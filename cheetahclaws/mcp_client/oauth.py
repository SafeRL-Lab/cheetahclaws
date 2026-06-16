"""OAuth 2.0 PKCE flow for MCP HTTP servers.

Implements the MCP Authorization spec:
  - Resource server metadata discovery (RFC 9728)
  - Authorization server metadata discovery (RFC 8414)
  - Dynamic client registration (RFC 7591) — used when no client_id configured
  - Authorization code + PKCE (S256) flow
  - Token refresh
  - Token persistence to ~/.cheetahclaws/mcp_oauth.json

Usage (from HttpTransport):
    from cheetahclaws.mcp_client.oauth import OAuthSession
    session = OAuthSession(server_name, resource_url, headers_config)
    token = session.get_token()   # blocks for browser auth on first call
    # then inject:  {"Authorization": f"Bearer {token}"}
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional

_TOKEN_STORE = Path.home() / ".cheetahclaws" / "mcp_oauth.json"


# ── Token persistence ─────────────────────────────────────────────────────────

def _load_store() -> dict:
    try:
        return json.loads(_TOKEN_STORE.read_text())
    except Exception:
        return {}


def _save_store(data: dict) -> None:
    _TOKEN_STORE.parent.mkdir(parents=True, exist_ok=True)
    # Restrict the parent dir too — best-effort, ignored on filesystems that
    # don't honour POSIX modes (e.g. some Windows setups).
    try:
        os.chmod(_TOKEN_STORE.parent, 0o700)
    except OSError:
        pass
    # Write atomically with 0600 perms so refresh tokens aren't world-readable.
    tmp = _TOKEN_STORE.with_suffix(_TOKEN_STORE.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass
    os.replace(tmp, _TOKEN_STORE)


# ── PKCE helpers ──────────────────────────────────────────────────────────────

def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


# ── OAuth metadata discovery ──────────────────────────────────────────────────

def _discover(resource_url: str, extra_headers: dict) -> dict:
    """Return authorization server metadata for the given MCP resource URL.

    Steps (per MCP spec):
      1. Fetch /.well-known/oauth-protected-resource[/<path>]
      2. Pick the first authorization_server from the response
      3. Fetch that server's /.well-known/oauth-authorization-server
    """
    import httpx
    parsed = urllib.parse.urlparse(resource_url)
    path_suffix = parsed.path.lstrip("/")
    well_known_candidates = []
    if path_suffix:
        well_known_candidates.append(
            f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource/{path_suffix}"
        )
    well_known_candidates.append(
        f"{parsed.scheme}://{parsed.netloc}/.well-known/oauth-protected-resource"
    )

    resource_meta = None
    for url in well_known_candidates:
        try:
            r = httpx.get(url, headers=extra_headers, timeout=10, follow_redirects=True)
            if r.status_code == 200:
                resource_meta = r.json()
                break
        except Exception:
            continue

    if not resource_meta:
        raise RuntimeError(f"Could not discover OAuth metadata for {resource_url}")

    auth_servers = resource_meta.get("authorization_servers", [])
    if not auth_servers:
        raise RuntimeError(f"No authorization_servers in OAuth metadata for {resource_url}")

    as_base = auth_servers[0].rstrip("/")
    as_meta_url = f"{as_base}/.well-known/oauth-authorization-server"
    r = httpx.get(as_meta_url, timeout=10, follow_redirects=True)
    if r.status_code != 200:
        raise RuntimeError(f"Failed to fetch authorization server metadata from {as_meta_url}: {r.status_code}")
    return r.json()


# ── Dynamic client registration ───────────────────────────────────────────────

def _register_client(registration_endpoint: str, server_name: str, redirect_uri: str) -> str:
    """Register a new OAuth client and return the client_id."""
    import httpx
    payload = {
        "client_name": f"cheetahclaws-{server_name}",
        "redirect_uris": [redirect_uri],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }
    r = httpx.post(registration_endpoint, json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["client_id"]


# ── Local callback server ─────────────────────────────────────────────────────

class _CallbackServer:
    """Tiny HTTP server that catches the OAuth redirect and extracts the code."""

    def __init__(self, port: int, state: str):
        self._port = port
        self._state = state
        self._code: Optional[str] = None
        self._error: Optional[str] = None
        self._event = threading.Event()
        self._server: Optional[http.server.HTTPServer] = None

    def start(self) -> None:
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parsed.query)
                if params.get("state", [""])[0] != parent._state:
                    parent._error = "state mismatch"
                elif "error" in params:
                    parent._error = params["error"][0]
                else:
                    parent._code = params.get("code", [None])[0]
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                msg = "Authorization complete. You can close this tab." \
                    if parent._code else f"Authorization failed: {parent._error}"
                self.wfile.write(f"<html><body><p>{msg}</p></body></html>".encode())
                parent._event.set()

            def log_message(self, *_):
                pass  # silence request logs

        self._server = http.server.HTTPServer(("127.0.0.1", self._port), Handler)
        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()

    def wait(self, timeout: int = 120) -> Optional[str]:
        self._event.wait(timeout=timeout)
        if self._server:
            self._server.shutdown()
        if self._error:
            raise RuntimeError(f"OAuth error: {self._error}")
        return self._code


# ── Token exchange & refresh ──────────────────────────────────────────────────

def _exchange_code(token_endpoint: str, client_id: str, code: str,
                   redirect_uri: str, verifier: str) -> dict:
    import httpx
    r = httpx.post(token_endpoint, data={
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
    }, timeout=15)
    r.raise_for_status()
    return r.json()


def _refresh_token(token_endpoint: str, client_id: str, refresh: str) -> dict:
    import httpx
    r = httpx.post(token_endpoint, data={
        "grant_type": "refresh_token",
        "refresh_token": refresh,
        "client_id": client_id,
    }, timeout=15)
    r.raise_for_status()
    return r.json()


# ── OAuthSession ──────────────────────────────────────────────────────────────

class OAuthSession:
    """Manages the full OAuth lifecycle for one MCP server.

    Call get_token() before each request. It returns a valid access token,
    refreshing or re-authorizing transparently as needed.
    """

    def __init__(self, server_name: str, resource_url: str,
                 extra_headers: dict | None = None):
        self._name = server_name
        self._resource_url = resource_url
        self._extra_headers = extra_headers or {}
        self._lock = threading.Lock()
        self._as_meta: Optional[dict] = None   # cached auth server metadata

    # ── Public API ────────────────────────────────────────────────────────────

    def get_token(self) -> str:
        """Return a valid access token, refreshing or re-authorizing as needed."""
        with self._lock:
            store = _load_store()
            entry = store.get(self._name, {})

            # Valid non-expired token
            if entry.get("access_token") and not self._is_expired(entry):
                return entry["access_token"]

            # Try refresh first
            if entry.get("refresh_token"):
                try:
                    meta = self._as_metadata()
                    tokens = _refresh_token(
                        meta["token_endpoint"],
                        entry["client_id"],
                        entry["refresh_token"],
                    )
                    entry = self._merge_tokens(entry, tokens)
                    store[self._name] = entry
                    _save_store(store)
                    return entry["access_token"]
                except Exception:
                    pass  # fall through to full re-auth

            # Full interactive auth
            entry = self._authorize(entry)
            store[self._name] = entry
            _save_store(store)
            return entry["access_token"]

    def clear(self) -> None:
        """Remove stored tokens (force re-auth on next get_token())."""
        store = _load_store()
        store.pop(self._name, None)
        _save_store(store)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _as_metadata(self) -> dict:
        if self._as_meta is None:
            self._as_meta = _discover(self._resource_url, self._extra_headers)
        return self._as_meta

    @staticmethod
    def _is_expired(entry: dict) -> bool:
        exp = entry.get("expires_at")
        if exp is None:
            return False
        # Refresh 60 s before actual expiry
        return time.time() >= exp - 60

    @staticmethod
    def _merge_tokens(entry: dict, tokens: dict) -> dict:
        entry["access_token"] = tokens["access_token"]
        if "refresh_token" in tokens:
            entry["refresh_token"] = tokens["refresh_token"]
        if "expires_in" in tokens:
            entry["expires_at"] = time.time() + int(tokens["expires_in"])
        else:
            entry.pop("expires_at", None)
        return entry

    def _authorize(self, entry: dict) -> dict:
        meta = self._as_metadata()
        auth_endpoint  = meta["authorization_endpoint"]
        token_endpoint = meta["token_endpoint"]
        reg_endpoint   = meta.get("registration_endpoint")

        # Pick the loopback port ONCE — registration and callback must use the
        # same redirect_uri or strict OAuth servers (per the MCP spec) reject it.
        port = self._pick_port()
        redirect_uri = f"http://localhost:{port}/callback"

        # Ensure we have a client_id (register if not)
        client_id = entry.get("client_id")
        if not client_id and reg_endpoint:
            client_id = _register_client(reg_endpoint, self._name, redirect_uri)
            entry["client_id"] = client_id
        elif not client_id:
            raise RuntimeError(
                f"MCP server '{self._name}' requires OAuth but has no client_id "
                "and does not support dynamic registration. "
                "Add 'oauth_client_id' to its mcp.json entry."
            )

        verifier, challenge = _pkce_pair()
        state = secrets.token_urlsafe(32)

        # Pick a scope the server actually advertises; omit otherwise.
        auth_params = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        scope = self._pick_scope(meta)
        if scope:
            auth_params["scope"] = scope

        params = urllib.parse.urlencode(auth_params)
        auth_url = f"{auth_endpoint}?{params}"

        cb = _CallbackServer(port, state)
        cb.start()

        print(f"\n🔐 OAuth required for MCP server '{self._name}'")
        print(f"   Opening browser… if it doesn't open, visit:\n   {auth_url}\n")
        webbrowser.open(auth_url)

        code = cb.wait(timeout=120)
        if not code:
            raise RuntimeError(f"OAuth timed out waiting for callback for '{self._name}'")

        tokens = _exchange_code(token_endpoint, client_id, code, redirect_uri, verifier)
        entry["client_id"] = client_id
        return self._merge_tokens(entry, tokens)

    @staticmethod
    def _pick_port() -> int:
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    @staticmethod
    def _pick_scope(meta: dict) -> Optional[str]:
        """Choose a scope the AS advertises. Prefer 'mcp', else the first one,
        else None (which means: don't send a scope parameter at all).

        Hardcoding 'mcp' broke servers whose scopes_supported didn't include it
        (invalid_scope error from the AS).
        """
        supported = meta.get("scopes_supported") or []
        if "mcp" in supported:
            return "mcp"
        if supported:
            return supported[0]
        return None
