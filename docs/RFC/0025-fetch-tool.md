# Design Note: Fetch Tool — bounded HTTP for agents

- **Status:** Draft
- **Tracking issue:** _to be filed_
- **Author:** @shangdinggu
- **Last updated:** 2026-05-08
- **Builds on:** [`0005-capability-model.md`](./0005-capability-model.md), [`0021-tool-dispatch.md`](./0021-tool-dispatch.md), [`0023-shell-exec-tool.md`](./0023-shell-exec-tool.md)

The second-most-dangerous tool the kernel ships (after Exec).
HTTP requests open the door to **SSRF** (the agent fetching
internal services), **DNS rebinding** (allowed-domain.com →
internal IP), **redirect-driven privilege escalation**,
**credential exfiltration** (Authorization headers leaking
across hops), **size bombs**, and **slowloris** holds.

This RFC defines the boundary that makes outbound HTTP
survivable for our single-user, single-host threat model. The
tool ships **opt-in** (not in ``register_builtin_tools``), like
Exec.

## 1. Threat model

**Defended against:**

| Threat | Defence |
|---|---|
| SSRF to localhost | Resolved IP must NOT be in private/loopback ranges (127/8, 10/8, 172.16/12, 192.168/16, 169.254/16, ::1, fc00::/7, fe80::/10, 0/8) |
| DNS rebinding | DNS resolved at the start of EACH request hop; private IPs blocked even if domain was on net_grants |
| Cross-domain redirect | Redirects (when enabled) re-validate scheme + hostname + IP for each hop |
| Auth/Cookie leak across redirect | ``Authorization`` and ``Cookie`` headers stripped on redirect |
| Unbounded download | Streaming read with size cap (default 4 MB, max 16 MB) |
| Long-running fetch | Wall-clock timeout (default 30 s, max 120 s) |
| Non-HTTP schemes (file://, gopher://, ftp://, data:) | Scheme allowlist — only ``http`` / ``https`` accepted |
| Cookie persistence | No cookie jar — each request is fresh |
| Env-based proxy attack | ``http.client`` doesn't honour ``HTTP_PROXY`` env |
| TLS downgrade | TLS verification on, no opt-out |

**Out of scope:**

- The agent itself misusing the data fetched.
- Vulnerabilities in the remote server.
- Network-level eavesdropping (not the kernel's job).
- Exfil via DNS query metadata (DNS lookups happen, by design).
- IP spoofing / BGP hijacking.

## 2. Tool spec

### Name

``Fetch``

### Args

```jsonc
{
  "url":              "https://api.example.com/v1/users",
  "method":           "GET",         // GET | HEAD | POST
  "headers":          {"Accept": "application/json"},
  "body":             "...",          // for POST; str or base64-encoded bytes
  "max_bytes":        4194304,        // default 4 MB, max 16 MB
  "timeout_s":        30,             // default 30, max 120
  "follow_redirects": false,          // default false
  "max_redirects":    3               // only relevant if follow_redirects=true
}
```

### Validation

- ``url``: parsed by ``urllib.parse.urlparse``. Scheme must be
  ``http`` or ``https``. Hostname must be present.
- ``method``: one of ``GET``, ``HEAD``, ``POST``.
- ``headers``: dict[str, str]. Each header name + value must be
  printable ASCII. Header name not in
  ``{"host", "content-length"}`` (we set those).
- ``body``: only valid with ``POST``; str (sent as UTF-8) or
  ``{"_b64": "..."}`` base64 envelope. Max 1 MB.
- ``max_bytes``: 1 KB ≤ x ≤ 16 MB.
- ``timeout_s``: 1 ≤ x ≤ 120.
- ``max_redirects``: 0 ≤ x ≤ 5.

### Capability requirements

- ``tool_grants`` must include ``"Fetch"``.
- ``net_grants`` must cover the URL's hostname (see
  ``kernel.cap.check_net``). Globs ``*.example.com`` work as
  documented in RFC 0005 §2.

### Per-hop checks (initial + each redirect)

1. Re-parse the URL (initial or Location header value).
2. Scheme in {``http``, ``https``}.
3. ``cap.check_net(pid, hostname)`` passes.
4. ``socket.gethostbyname(hostname)`` resolves to a public IP
   (not in the private/loopback table above).
5. On redirect: strip ``Authorization`` and ``Cookie`` headers
   before the next request.

### Result

```jsonc
{
  "url":             "https://api.example.com/v1/users",  // final URL after redirects
  "status":          200,
  "headers":         {"content-type": "application/json", ...},
  "body":            "{\"users\": [...]}",
  "encoding":        "utf-8",                      // or "base64" for non-UTF-8
  "body_bytes":      1234,                         // raw byte count
  "truncated":       false,
  "redirects":       [{"from": "...", "to": "...", "status": 302}],
  "duration_s":      0.123
}
```

If the body decodes as UTF-8 cleanly, ``body`` is the text and
``encoding`` is ``"utf-8"``. Otherwise ``body`` is base64 and
``encoding`` is ``"base64"`` — the LLM caller can decide whether
to ingest it.

## 3. New error

A new ``ToolNetDenied`` subclass of ``ToolError`` (slug
``net_denied``). Raised when:

- ``cap.check_net`` returns False for the hostname.
- DNS resolves the hostname to a private/loopback IP.
- A redirect target fails the same check.

The supervisor's existing dispatch path (RFC 0021) translates
this to ``tool_response.ok=false, error=net_denied`` exactly
like other errors.

## 4. Implementation

The handler uses ``http.client`` (not ``urllib.request``,
because the latter auto-follows redirects in a way that's hard
to inject re-validation into, and not third-party libs because
we keep the kernel stdlib-only).

Sketch:

```python
def fetch_handler(args, ctx):
    url = _validate_url(args["url"])
    method = _validate_method(args.get("method", "GET"))
    headers = _validate_headers(args.get("headers"))
    body_bytes = _build_body(args.get("body"), method)
    max_bytes = _validate_max_bytes(args.get("max_bytes"))
    timeout_s = _validate_timeout(args.get("timeout_s"))
    follow = bool(args.get("follow_redirects", False))
    max_redirects = _validate_max_redirects(args.get("max_redirects", 3))

    redirects = []
    current_url = url
    current_headers = dict(headers)

    for hop in range(max_redirects + 1):
        scheme, host, port, path = _split(current_url)
        # Capability + IP checks (each hop).
        if ctx.kernel and not ctx.kernel.cap.check_net(ctx.pid, host):
            raise ToolNetDenied(f"net not granted: {host}")
        ip = _resolve(host)                   # socket.gethostbyname
        if _is_private_ip(ip):
            raise ToolNetDenied(f"resolved to private IP: {ip}")
        # Send.
        resp = _http_request(scheme, host, port, path, method,
                              current_headers, body_bytes, timeout_s)
        if 300 <= resp.status < 400 and follow:
            loc = resp.headers.get("location")
            if not loc:
                break
            current_url = _absolutise(current_url, loc)
            redirects.append({"from": ..., "to": current_url,
                              "status": resp.status})
            # Strip auth on redirect.
            current_headers.pop("Authorization", None)
            current_headers.pop("Cookie", None)
            continue
        break
    else:
        raise ToolFailed("too many redirects")

    body, body_bytes_count, truncated = _stream_read(
        resp, max_bytes,
    )
    return {
        "url":         current_url,
        "status":      resp.status,
        "headers":     dict(resp.headers),
        "body":        body,
        "encoding":    enc,
        "body_bytes":  body_bytes_count,
        "truncated":   truncated,
        "redirects":   redirects,
        "duration_s":  duration,
    }
```

## 5. Backwards compatibility

- New file ``kernel/tools/fetch_tool.py``.
- ``register_builtin_tools`` is unchanged; Fetch is opt-in via
  ``register_fetch_tool(registry)``.
- ``ToolNetDenied`` is a new exception in ``registry.py`` —
  additive subclass of existing ``ToolError``.

## 6. Open questions

1. **Cookie support.** Some real workflows need session cookies.
   Proposed: ``args.cookies`` dict-style; each cookie is a
   single-flight key-value. Out of scope for v1.
2. **HTTP/2.** ``http.client`` is HTTP/1.1 only. Future RFC may
   wrap a third-party HTTP/2 client when needed.
3. **DNS resolver.** v1 uses ``socket.gethostbyname``. A future
   RFC could pin to a specific DNS resolver to avoid the host
   resolver picking up local hosts file entries / mDNS /
   ``/etc/hosts`` overrides that map allowed-domain → 127.0.0.1.
   v1 trusts the system resolver; documented as a known caveat.
4. **IPv6.** ``gethostbyname`` returns IPv4 only. For IPv6
   targets, future work uses ``getaddrinfo``. v1 fetches over
   IPv4-resolvable addresses only.

## 7. Acceptance criteria

A PR claiming this RFC must:

1. ``register_fetch_tool`` is NOT called by
   ``register_builtin_tools``.
2. URL with non-http/https scheme rejected.
3. Method outside GET/HEAD/POST rejected.
4. ``cap.check_net`` denial → ``net_denied``.
5. Hostname resolving to 127.0.0.1 → ``net_denied``.
6. Hostname resolving to 10.0.0.x → ``net_denied``.
7. Hostname resolving to 169.254.169.254 (cloud metadata) →
   ``net_denied``.
8. Real HTTP fetch via local ``http.server`` succeeds; size
   cap + truncation observed; status / headers correct.
9. Redirect target with private IP → ``net_denied`` (defense
   against DNS-rebinding via redirect).
10. Redirect strips Authorization header (verified by serving
    a redirect that echoes incoming headers).
11. POST with body works.
12. No file outside ``kernel/``, ``tests/``, ``docs/RFC/``
    modified.
