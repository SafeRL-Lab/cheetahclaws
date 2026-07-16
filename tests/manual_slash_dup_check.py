"""Manual repro for issue #111 — slash command duplicate replies.

Simulates the chat UI: opens a WS subscriber + POSTs a slash command via
HTTP. Counts how many `command_result` events the client would render
(HTTP `data.events` + WS `onmessage`). Pre-fix this prints 2; post-fix 1.

Usage:
    # In one terminal: start the server
    cd cheetahclaws && python cheetahclaws.py --web --no-auth --port 8080

    # In another terminal:
    python tests/manual_slash_dup_check.py http://127.0.0.1:8080
"""
import json
import sys
import threading
import time
import urllib.request
import urllib.parse
from http.cookiejar import CookieJar

try:
    from websocket import create_connection  # pip install websocket-client
except ImportError:
    print("Need: pip install websocket-client")
    sys.exit(2)


def http(opener, method, url, body=None, headers=None):
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    return opener.open(req, timeout=10)


def main(base_url):
    cj = CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    # --no-auth mode: skip register/login (server returns uid=1 implicitly).
    # If you need an authed run, log in first and the cookie jar handles it.

    r = http(opener, "POST", f"{base_url}/api/prompt",
             {"prompt": "", "session_id": ""})
    sid = json.loads(r.read())["session_id"]
    print(f"[+] session: {sid}")

    ws_url = base_url.replace("http://", "ws://").replace("https://", "wss://")
    cookies = [f"{c.name}={c.value}" for c in cj]
    headers = [f"Cookie: {'; '.join(cookies)}"] if cookies else []
    ws = create_connection(f"{ws_url}/api/events", header=headers)
    ws.send(json.dumps({"session_id": sid}))

    ws_events = []
    def reader():
        try:
            while True:
                msg = ws.recv()
                if not msg:
                    break
                ws_events.append(json.loads(msg))
        except Exception:
            pass

    threading.Thread(target=reader, daemon=True).start()
    time.sleep(0.3)  # let WS attach

    for cmd in ["/status", "/model", "/cost"]:
        r = http(opener, "POST", f"{base_url}/api/prompt",
                 {"prompt": cmd, "session_id": sid})
        http_events = json.loads(r.read())["events"]
        time.sleep(0.5)  # let WS catch up

        http_results = sum(1 for e in http_events
                            if e["type"] == "command_result"
                            and e["data"]["command"] == cmd)
        ws_results = sum(1 for e in ws_events
                          if e.get("type") == "command_result"
                          and e["data"]["command"] == cmd)
        total = http_results + ws_results
        marker = "OK " if total == 1 else "BUG"
        print(f"  [{marker}] {cmd}: HTTP={http_results}  WS={ws_results}  "
              f"client_renders={total}")
        ws_events.clear()

    ws.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080")
