"""MCP client: stdio and HTTP/SSE transports, JSON-RPC 2.0 protocol."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional


# Env vars that a malicious mcp config could use to hijack the subprocess —
# LD_PRELOAD shims libc, PYTHONPATH/PYTHONSTARTUP can run arbitrary code on
# Python interpreter MCP servers, etc. They're stripped from caller-supplied
# env unless the operator opts in.
_MCP_BLOCKED_ENV_KEYS = frozenset({
    "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT",
    "DYLD_INSERT_LIBRARIES", "DYLD_LIBRARY_PATH",
    "PYTHONPATH", "PYTHONSTARTUP", "PYTHONHOME", "PYTHONEXECUTABLE",
    "NODE_OPTIONS", "NODE_PATH",
    "BASH_ENV", "ENV",
})


def _sanitized_mcp_env(user_env: Optional[dict]) -> dict[str, str]:
    """Return os.environ updated with caller-supplied env, minus dangerous keys.

    Set CHEETAHCLAWS_MCP_TRUST_ENV=1 to bypass (for power users wiring up
    a server that legitimately needs LD_LIBRARY_PATH etc.).
    """
    base = dict(os.environ)
    if not user_env:
        return base
    trust = os.environ.get("CHEETAHCLAWS_MCP_TRUST_ENV", "0") == "1"
    dropped: list[str] = []
    for k, v in user_env.items():
        if (not trust) and k in _MCP_BLOCKED_ENV_KEYS:
            dropped.append(k)
            continue
        base[k] = v
    if dropped:
        print(
            f"[mcp] Dropped potentially-dangerous env keys from server "
            f"config: {sorted(dropped)}. Set CHEETAHCLAWS_MCP_TRUST_ENV=1 "
            f"to allow.",
            file=sys.stderr, flush=True,
        )
    return base

from .types import (
    MCPServerConfig, MCPServerState, MCPTool, MCPTransport,
    INIT_PARAMS, make_notification, make_request,
)


# ── Stdio transport ───────────────────────────────────────────────────────────

class StdioTransport:
    """Bidirectional JSON-RPC over a subprocess's stdin/stdout.

    Messages are newline-delimited JSON objects (one per line).
    Responses are matched to requests by 'id'.
    """

    def __init__(self, config: MCPServerConfig):
        self._config = config
        self._process: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._next_id = 1
        self._pending: Dict[int, dict] = {}   # id → {"event": Event, "result": ...}
        self._reader: Optional[threading.Thread] = None
        self._stderr_reader: Optional[threading.Thread] = None
        self._running = False
        self._stderr_lines: List[str] = []

    def start(self) -> None:
        env = _sanitized_mcp_env(self._config.env)
        cmd = [self._config.command] + list(self._config.args or [])
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_reader.start()

    def _read_loop(self) -> None:
        while self._running and self._process:
            try:
                raw = self._process.stdout.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                msg = json.loads(line)
            except Exception:
                continue
            # Dispatch: response (has "id") vs notification (no "id").
            # Use pop() with a fallback so a late-arriving response after a
            # request timed out (and removed its holder) is silently dropped
            # instead of racing on dict membership-then-index.
            msg_id = msg.get("id")
            if msg_id is None:
                continue
            holder = self._pending.pop(msg_id, None)
            if holder is None:
                continue
            holder["result"] = msg
            holder["event"].set()

    def _stderr_loop(self) -> None:
        while self._running and self._process:
            try:
                raw = self._process.stderr.readline()
                if not raw:
                    break
                self._stderr_lines.append(raw.decode("utf-8", errors="replace").rstrip())
            except Exception:
                break

    def _send_raw(self, msg: dict) -> None:
        line = (json.dumps(msg) + "\n").encode("utf-8")
        with self._lock:
            self._process.stdin.write(line)
            self._process.stdin.flush()

    def request(self, method: str, params: Optional[dict] = None, timeout: Optional[int] = None) -> dict:
        """Send a JSON-RPC request and wait for the response."""
        with self._lock:
            req_id = self._next_id
            self._next_id += 1
        event = threading.Event()
        holder: dict = {"event": event, "result": None}
        self._pending[req_id] = holder
        msg = make_request(method, params, req_id)
        self._send_raw(msg)
        wait_secs = timeout or self._config.timeout
        signalled = event.wait(timeout=wait_secs)
        # Pop unconditionally — if signalled, the reader already set result
        # before set(); if timeout, we drop the holder so a late response
        # in _read_loop's pop() returns None and is discarded.
        self._pending.pop(req_id, None)
        result = holder["result"]
        if not signalled or result is None:
            raise TimeoutError(f"MCP server '{self._config.name}' timed out on '{method}'")
        if "error" in result:
            err = result["error"]
            raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")
        return result.get("result", {})

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        """Send a JSON-RPC notification (no response expected)."""
        self._send_raw(make_notification(method, params))

    def stop(self) -> None:
        self._running = False
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                pass
            self._process = None

    @property
    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def stderr_output(self) -> str:
        return "\n".join(self._stderr_lines[-20:])


# ── HTTP / SSE transport ──────────────────────────────────────────────────────

class HttpTransport:
    """HTTP-based MCP transport (POST-based streamable HTTP or SSE endpoint).

    For SSE servers: sends messages via POST to the SSE session endpoint.
    For HTTP servers: sends messages via POST and reads response directly.
    Supports OAuth 2.0 PKCE: when the server returns 401, an OAuthSession is
    created automatically to obtain and refresh Bearer tokens.
    """

    def __init__(self, config: MCPServerConfig):
        self._config = config
        self._session_url: Optional[str] = None
        self._lock = threading.Lock()
        self._oauth_lock = threading.Lock()  # serialises httpx client rebuilds
        self._next_id = 1
        self._client = None   # httpx.Client, loaded lazily
        self._sse_thread: Optional[threading.Thread] = None
        self._sse_pending: Dict[int, dict] = {}
        self._running = False
        self._oauth: Optional[Any] = None  # OAuthSession, created on first 401

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                # Accept both JSON and SSE — required by servers like sap-jira
                # that return 406 when only one content type is advertised.
                headers = {"Accept": "application/json, text/event-stream",
                           **self._config.headers}
                self._client = httpx.Client(
                    headers=headers,
                    timeout=self._config.timeout,
                    follow_redirects=True,
                )
            except ImportError:
                raise RuntimeError("httpx is required for HTTP/SSE MCP transport: pip install httpx")
        return self._client

    def _ensure_oauth(self) -> None:
        """Initialise OAuthSession on first 401 and inject the token into the client.

        Guarded by _oauth_lock so concurrent 401-retries don't race on the
        httpx.Client (close + recreate is otherwise not safe).
        """
        with self._oauth_lock:
            if self._oauth is None:
                from cc_mcp.oauth import OAuthSession
                # Pass any non-auth headers already configured (e.g. custom SAP headers)
                extra = {k: v for k, v in self._config.headers.items()
                         if not k.lower().startswith("authorization")}
                self._oauth = OAuthSession(self._config.name, self._config.url, extra)
            token = self._oauth.get_token()
            # Rebuild the httpx client with the fresh token injected.
            if self._client:
                self._client.close()
                self._client = None
            import httpx
            headers = {"Accept": "application/json, text/event-stream",
                       **self._config.headers, "Authorization": f"Bearer {token}"}
            self._client = httpx.Client(
                headers=headers,
                timeout=self._config.timeout,
                follow_redirects=True,
            )

    def start(self) -> None:
        """For SSE transport: connect to the /sse endpoint and get session URL."""
        if self._config.transport == MCPTransport.SSE:
            self._start_sse()
        else:
            # Streamable HTTP: no persistent connection needed; probe for 401.
            self._session_url = self._config.url
            self._probe_oauth()

    def _probe_oauth(self) -> None:
        """HEAD the resource URL; if 401, run OAuth flow before first request."""
        import httpx
        try:
            r = httpx.get(self._config.url, headers=self._config.headers,
                          timeout=5, follow_redirects=True)
            if r.status_code == 401:
                self._ensure_oauth()
        except Exception:
            pass  # network errors surface properly during request()

    def _start_sse(self) -> None:
        """Open SSE stream to get session endpoint, then start background reader."""
        import httpx
        client = self._get_client()
        self._running = True

        # Initial SSE connect — first event should be 'endpoint' with session URL
        endpoint_event = threading.Event()
        endpoint_holder: dict = {"url": None, "error": None}

        def _sse_reader():
            try:
                with client.stream("GET", self._config.url) as resp:
                    resp.raise_for_status()
                    event_type = None
                    for line in resp.iter_lines():
                        if not self._running:
                            break
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                            if event_type == "endpoint":
                                # Session URL may be relative or absolute
                                base = self._config.url.rsplit("/sse", 1)[0]
                                session_url = data if data.startswith("http") else base + data
                                endpoint_holder["url"] = session_url
                                self._session_url = session_url
                                endpoint_event.set()
                            elif event_type == "message":
                                try:
                                    msg = json.loads(data)
                                    msg_id = msg.get("id")
                                    if msg_id is not None:
                                        holder = self._sse_pending.pop(msg_id, None)
                                        if holder is not None:
                                            holder["result"] = msg
                                            holder["event"].set()
                                except Exception:
                                    pass
            except Exception as e:
                endpoint_holder["error"] = str(e)
                endpoint_event.set()

        self._sse_thread = threading.Thread(target=_sse_reader, daemon=True)
        self._sse_thread.start()
        endpoint_event.wait(timeout=10)
        if endpoint_holder.get("error"):
            raise RuntimeError(f"SSE connect failed: {endpoint_holder['error']}")
        if not self._session_url:
            raise RuntimeError("SSE server did not send 'endpoint' event")

    def request(self, method: str, params: Optional[dict] = None, timeout: Optional[int] = None) -> dict:
        with self._lock:
            req_id = self._next_id
            self._next_id += 1

        msg = make_request(method, params, req_id)
        client = self._get_client()
        wait_secs = timeout or self._config.timeout

        if self._config.transport == MCPTransport.SSE:
            # For SSE: POST to session URL, wait for response on SSE stream
            event = threading.Event()
            holder: dict = {"event": event, "result": None}
            self._sse_pending[req_id] = holder
            client.post(self._session_url, json=msg)
            signalled = event.wait(timeout=wait_secs)
            self._sse_pending.pop(req_id, None)
            result = holder["result"] if signalled else None
        else:
            # Streamable HTTP: POST returns an SSE stream; read first data: line.
            # Retry once on 401 (token expired mid-session).
            url = self._session_url or self._config.url
            result = None
            for _attempt in range(2):
                with client.stream("POST", url, json=msg, timeout=wait_secs) as resp:
                    if resp.status_code == 401 and _attempt == 0:
                        resp.read()  # drain
                        self._ensure_oauth()
                        client = self._get_client()
                        continue
                    resp.raise_for_status()
                    ct = resp.headers.get("content-type", "")
                    if "text/event-stream" in ct:
                        for line in resp.iter_lines():
                            if line.startswith("data:"):
                                result = json.loads(line[5:].strip())
                                break
                    else:
                        result = resp.json()
                    break  # success

        if result is None:
            raise TimeoutError(f"MCP server '{self._config.name}' timed out on '{method}'")
        if "error" in result:
            err = result["error"]
            raise RuntimeError(f"MCP error {err.get('code')}: {err.get('message')}")
        return result.get("result", {})

    def notify(self, method: str, params: Optional[dict] = None) -> None:
        client = self._get_client()
        msg = make_notification(method, params)
        url = self._session_url or self._config.url
        try:
            client.post(url, json=msg)
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    @property
    def alive(self) -> bool:
        return self._session_url is not None or self._config.transport == MCPTransport.HTTP


# ── High-level MCP client ─────────────────────────────────────────────────────

class MCPClient:
    """Manages the lifecycle of one MCP server connection.

    Protocol flow:
        connect() → initialize handshake → notifications/initialized
        list_tools() → tools/list
        call_tool()  → tools/call
        disconnect() → cleanup
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.state = MCPServerState.DISCONNECTED
        self._transport: Optional[Any] = None
        self._server_info: dict = {}
        self._capabilities: dict = {}
        self._tools: List[MCPTool] = []
        self._error: str = ""

    # ── Connection ────────────────────────────────────────────────────────────

    def connect(self) -> None:
        if self.state == MCPServerState.CONNECTED:
            return
        self.state = MCPServerState.CONNECTING
        self._error = ""
        try:
            self._transport = self._make_transport()
            self._transport.start()
            self._handshake()
            self.state = MCPServerState.CONNECTED
        except Exception as e:
            self.state = MCPServerState.ERROR
            self._error = str(e)
            raise

    def _make_transport(self):
        t = self.config.transport
        if t == MCPTransport.STDIO:
            return StdioTransport(self.config)
        if t in (MCPTransport.SSE, MCPTransport.HTTP):
            return HttpTransport(self.config)
        raise ValueError(f"Unsupported MCP transport: {t}")

    def _handshake(self) -> None:
        result = self._transport.request("initialize", INIT_PARAMS, timeout=15)
        self._server_info = result.get("serverInfo", {})
        self._capabilities = result.get("capabilities", {})
        self._transport.notify("notifications/initialized")

    def disconnect(self) -> None:
        if self._transport:
            self._transport.stop()
            self._transport = None
        self.state = MCPServerState.DISCONNECTED

    def reconnect(self) -> None:
        self.disconnect()
        self.connect()

    @property
    def alive(self) -> bool:
        return (
            self.state == MCPServerState.CONNECTED
            and self._transport is not None
            and self._transport.alive
        )

    # ── Tool discovery ────────────────────────────────────────────────────────

    def list_tools(self) -> List[MCPTool]:
        """Fetch tool list from server and cache as MCPTool objects."""
        if self.state != MCPServerState.CONNECTED:
            raise RuntimeError(f"MCP server '{self.config.name}' is not connected")

        if "tools" not in self._capabilities:
            self._tools = []
            return self._tools

        result = self._transport.request("tools/list", timeout=15)
        raw_tools = result.get("tools", [])
        self._tools = [self._parse_tool(t) for t in raw_tools]
        return self._tools

    def _parse_tool(self, raw: dict) -> MCPTool:
        tool_name = raw.get("name", "")
        qualified = f"mcp__{self.config.name}__{tool_name}"
        # Sanitize: replace non-alphanumeric with _ for API compatibility
        qualified = "".join(c if c.isalnum() or c == "_" else "_" for c in qualified)

        annotations = raw.get("annotations", {})
        read_only = bool(annotations.get("readOnlyHint", False))

        schema = raw.get("inputSchema", {"type": "object", "properties": {}})
        # Ensure minimum valid JSON schema
        if not isinstance(schema, dict):
            schema = {"type": "object", "properties": {}}

        return MCPTool(
            server_name=self.config.name,
            tool_name=tool_name,
            qualified_name=qualified,
            description=raw.get("description", ""),
            input_schema=schema,
            read_only=read_only,
        )

    # ── Tool invocation ───────────────────────────────────────────────────────

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Call a tool by its original (non-qualified) name.

        Returns the text content from the response, or an error string.
        """
        if self.state != MCPServerState.CONNECTED:
            raise RuntimeError(f"MCP server '{self.config.name}' is not connected")

        params = {"name": tool_name, "arguments": arguments}
        result = self._transport.request("tools/call", params, timeout=self.config.timeout)

        is_error = result.get("isError", False)
        content = result.get("content", [])

        # Collect text content blocks
        parts: List[str] = []
        for block in content:
            btype = block.get("type", "")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "image":
                parts.append(f"[image: {block.get('mimeType', 'unknown')}]")
            elif btype == "resource":
                res = block.get("resource", {})
                parts.append(f"[resource: {res.get('uri', '')}]")

        text = "\n".join(parts) if parts else str(result)
        if is_error:
            return f"[MCP tool error]\n{text}"
        return text

    # ── Status ────────────────────────────────────────────────────────────────

    def status_line(self) -> str:
        icon = {"connected": "✓", "connecting": "…", "disconnected": "○", "error": "✗"}.get(
            self.state.value, "?"
        )
        server = self._server_info.get("name", self.config.name)
        version = self._server_info.get("version", "")
        tool_count = len(self._tools)
        line = f"{icon} {self.config.name}"
        if server and server != self.config.name:
            line += f" ({server}"
            if version:
                line += f" v{version}"
            line += ")"
        if self.state == MCPServerState.CONNECTED:
            line += f"  [{tool_count} tool(s)]"
        if self.state == MCPServerState.ERROR:
            line += f"  error: {self._error}"
        return line


# ── Manager ───────────────────────────────────────────────────────────────────

class MCPManager:
    """Singleton that manages all configured MCP server connections."""

    def __init__(self):
        self._clients: Dict[str, MCPClient] = {}

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Normalize a server name to match the qualified tool name format."""
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name)

    def add_server(self, config: MCPServerConfig) -> MCPClient:
        """Register a server. Replaces any existing client with the same name."""
        key = self._sanitize_name(config.name)
        if key in self._clients:
            try:
                self._clients[key].disconnect()
            except Exception:
                pass
        client = MCPClient(config)
        self._clients[key] = client
        return client

    def connect_all(self) -> Dict[str, Optional[str]]:
        """Connect to all registered servers. Returns {name: error_or_None}."""
        errors: Dict[str, Optional[str]] = {}
        for name, client in self._clients.items():
            if client.config.disabled:
                errors[name] = "disabled"
                continue
            try:
                client.connect()
                client.list_tools()
                errors[name] = None
            except Exception as e:
                errors[name] = str(e)
        return errors

    def connect_server(self, name: str) -> MCPClient:
        """Connect (or reconnect) a single server by name."""
        client = self._clients.get(self._sanitize_name(name))
        if client is None:
            raise KeyError(f"MCP server '{name}' not configured")
        if client.state != MCPServerState.CONNECTED:
            client.connect()
            client.list_tools()
        return client

    def all_tools(self) -> List[MCPTool]:
        """Return all tools from all connected servers."""
        tools: List[MCPTool] = []
        for client in self._clients.values():
            if client.state == MCPServerState.CONNECTED:
                tools.extend(client._tools)
        return tools

    def call_tool(self, qualified_name: str, arguments: dict) -> str:
        """Dispatch a tool call by qualified name (mcp__server__tool)."""
        # Parse server and tool name from qualified name
        parts = qualified_name.split("__", 2)
        if len(parts) != 3 or parts[0] != "mcp":
            raise ValueError(f"Invalid MCP tool name: {qualified_name}")
        server_name = parts[1]
        tool_name = parts[2]

        client = self._clients.get(server_name)
        if client is None:
            raise RuntimeError(f"MCP server '{server_name}' not configured")

        # Auto-reconnect if dropped
        if not client.alive:
            client.reconnect()
            client.list_tools()

        # Find the original tool name (un-sanitized)
        original_name = tool_name
        for t in client._tools:
            if t.qualified_name == qualified_name:
                original_name = t.tool_name
                break

        return client.call_tool(original_name, arguments)

    def list_servers(self) -> List[MCPClient]:
        return list(self._clients.values())

    def disconnect_all(self) -> None:
        for client in self._clients.values():
            try:
                client.disconnect()
            except Exception:
                pass

    def reload_server(self, name: str) -> None:
        client = self._clients.get(self._sanitize_name(name))
        if client:
            client.reconnect()
            client.list_tools()


# ── Module-level singleton ────────────────────────────────────────────────────

_manager: Optional[MCPManager] = None


def get_mcp_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
