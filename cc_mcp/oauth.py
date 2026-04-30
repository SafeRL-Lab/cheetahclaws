"""OAuth 2.0 + PKCE flow for MCP HTTP servers.

Handles:
  - Dynamic client registration (RFC 7591)
  - Authorization Code + PKCE (S256)
  - Token refresh
  - Token persistence in ~/.cheetahclaws/mcp_oauth_tokens.json
"""
from __future__ import annotations

import base64
import hashlib
import http.server
import json
import re
import secrets
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path
from typing import Optional

TOKEN_STORE = Path.home() / ".cheetahclaws" / "mcp_oauth_tokens.json"
REDIRECT_PORT = 54321
REDIRECT_URI = f"http://localhost:{REDIRECT_PORT}/callback"


def _http():
    try:
        import httpx
        return httpx
    except ImportError:
        raise RuntimeError("httpx is required for MCP OAuth: pip install httpx")


# ── Token persistence ─────────────────────────────────────────────────────────

def _load_tokens() -> dict:
    if TOKEN_STORE.exists():
        try:
            return json.loads(TOKEN_STORE.read_text())
        except Exception:
            pass
    return {}


def _save_tokens(data: dict) -> None:
    TOKEN_STORE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_STORE.write_text(json.dumps(data, indent=2))


def get_cached_token(server_url: str) -> Optional[str]:
    data = _load_tokens()
    entry = data.get(server_url)
    if not entry:
        return None
    if entry.get("expires_at", 0) < time.time() + 60:
        refreshed = _try_refresh(server_url, entry)
        if refreshed:
            return refreshed
        return None
    return entry.get("access_token")


def _try_refresh(server_url: str, entry: dict) -> Optional[str]:
    refresh_token = entry.get("refresh_token")
    token_uri = entry.get("token_uri")
    client_id = entry.get("client_id")
    if not (refresh_token and token_uri and client_id):
        return None
    try:
        httpx = _http()
        resp = httpx.post(token_uri, data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
        }, timeout=15)
        resp.raise_for_status()
        token_data = resp.json()
        _cache_token(server_url, token_data, token_uri, client_id)
        return token_data.get("access_token")
    except Exception:
        return None


def _cache_token(server_url: str, token_data: dict, token_uri: str, client_id: str) -> None:
    data = _load_tokens()
    expires_in = token_data.get("expires_in", 3600)
    data[server_url] = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": time.time() + expires_in,
        "token_uri": token_uri,
        "client_id": client_id,
    }
    _save_tokens(data)


# ── OAuth metadata discovery ──────────────────────────────────────────────────

def _parse_www_authenticate(header: str) -> dict:
    result = {}
    for match in re.finditer(r'(\w+)="([^"]*)"', header):
        result[match.group(1)] = match.group(2)
    return result


def _fetch_json(url: str) -> dict:
    try:
        httpx = _http()
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def _discover_endpoints(www_auth_header: str) -> tuple[str, str]:
    params = _parse_www_authenticate(www_auth_header)
    auth_uri = params.get("authorization_uri", "")
    token_uri = params.get("token_uri", "")

    if not auth_uri or not token_uri:
        metadata_url = params.get("resource_metadata", "")
        if metadata_url:
            resource_meta = _fetch_json(metadata_url)
            for as_url in resource_meta.get("authorization_servers", []):
                as_meta = _fetch_json(f"{as_url.rstrip('/')}/.well-known/oauth-authorization-server")
                if as_meta:
                    auth_uri = auth_uri or as_meta.get("authorization_endpoint", "")
                    token_uri = token_uri or as_meta.get("token_endpoint", "")
                    break

    return auth_uri, token_uri


# ── Dynamic client registration ───────────────────────────────────────────────

def _register_client(registration_endpoint: str) -> str:
    httpx = _http()
    resp = httpx.post(registration_endpoint, json={
        "client_name": "cheetahclaws",
        "redirect_uris": [REDIRECT_URI],
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "token_endpoint_auth_method": "none",
    }, timeout=15)
    resp.raise_for_status()
    return resp.json()["client_id"]


def _get_or_register_client(server_url: str, token_uri: str) -> str:
    data = _load_tokens()
    reg_key = f"__client__{server_url}"
    if reg_key in data:
        return data[reg_key]["client_id"]

    base = token_uri.rsplit("/token", 1)[0]
    registration_endpoint = f"{base}/register"
    try:
        client_id = _register_client(registration_endpoint)
    except Exception as e:
        raise RuntimeError(
            f"Dynamic client registration failed for {server_url}: {e}\n"
            "Add a static 'Authorization' header to this server's entry in mcp.json."
        )

    data[reg_key] = {"client_id": client_id}
    _save_tokens(data)
    return client_id


# ── PKCE helpers ──────────────────────────────────────────────────────────────

def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    return verifier, challenge


# ── Local callback server ─────────────────────────────────────────────────────

class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    code: Optional[str] = None
    error: Optional[str] = None
    done = threading.Event()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        if "code" in params:
            _CallbackHandler.code = params["code"]
        else:
            _CallbackHandler.error = params.get("error", "unknown error")
        _CallbackHandler.done.set()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>Authentication complete. You can close this tab.</h2>")

    def log_message(self, *_):
        pass


def _run_callback_server() -> http.server.HTTPServer:
    _CallbackHandler.code = None
    _CallbackHandler.error = None
    _CallbackHandler.done.clear()
    server = http.server.HTTPServer(("localhost", REDIRECT_PORT), _CallbackHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# ── Main OAuth flow ───────────────────────────────────────────────────────────

def acquire_token(server_url: str, www_auth_header: str) -> str:
    """Run the full OAuth 2.0 + PKCE browser flow and return an access token."""
    auth_uri, token_uri = _discover_endpoints(www_auth_header)
    if not auth_uri or not token_uri:
        raise RuntimeError(
            f"Cannot discover OAuth endpoints for {server_url}. "
            "Set an Authorization header manually in mcp.json."
        )

    client_id = _get_or_register_client(server_url, token_uri)
    verifier, challenge = _pkce_pair()
    state = secrets.token_urlsafe(16)

    auth_url = auth_uri + "?" + urllib.parse.urlencode({
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": REDIRECT_URI,
        "scope": "mcp",
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    })

    callback_server = _run_callback_server()
    print(f"\n[MCP OAuth] Opening browser for {server_url} ...")
    print(f"[MCP OAuth] If the browser doesn't open, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    if not _CallbackHandler.done.wait(timeout=120):
        callback_server.shutdown()
        raise RuntimeError(f"OAuth timeout waiting for callback from {server_url}")
    callback_server.shutdown()

    if _CallbackHandler.error:
        raise RuntimeError(f"OAuth error from {server_url}: {_CallbackHandler.error}")

    httpx = _http()
    resp = httpx.post(token_uri, data={
        "grant_type": "authorization_code",
        "code": _CallbackHandler.code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "code_verifier": verifier,
    }, timeout=15)
    resp.raise_for_status()
    token_data = resp.json()

    if "access_token" not in token_data:
        raise RuntimeError(f"Token exchange failed for {server_url}: {token_data}")

    _cache_token(server_url, token_data, token_uri, client_id)
    print(f"[MCP OAuth] Authenticated successfully with {server_url}\n")
    return token_data["access_token"]
