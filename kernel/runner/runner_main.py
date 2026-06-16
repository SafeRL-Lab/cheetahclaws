"""runner_main.py — minimal subprocess entry point for kernel-managed agents.

This is the **substrate** runner. It implements the JSON-line protocol
defined in RFC 0016 §4 and runs a trivially simple agent loop suitable
for testing the supervisor end-to-end.

For a real LLM-driven agent, write a different ``__main__`` that
performs the same handshake and emits the same messages. The
supervisor doesn't care about the runner's internal logic — only the
protocol.

Usage:
    python -m kernel.runner.runner_main

Reads init from stdin, sends ready, executes the optional ``payload.py``
field as a Python expression in a tightly bounded namespace (for
testing), emits charge messages if asked, then exits cleanly.

Set environment variable ``CC_RUNNER_BEHAVIOR`` to control:
  - ``echo``          (default) emit init payload back, exit completed
  - ``loop``          busy loop until SIGTERM (tests sandbox/wall-kill)
  - ``crash``         exit non-zero immediately
  - ``charge=K:N``    emit a charge message for dim=K, amount=N before exit
  - ``slow=N``        sleep N seconds, then exit
"""
from __future__ import annotations

import json
import os
import sys
import time

from .ipc import JsonLineChannel


def main() -> int:
    chan = JsonLineChannel(sys.stdin.buffer, sys.stdout.buffer)
    behavior = os.environ.get("CC_RUNNER_BEHAVIOR", "echo")

    # 1) Read init from supervisor.
    try:
        init = chan.recv(timeout=10.0)
    except Exception as e:
        sys.stderr.write(f"runner: init recv failed: {e}\n")
        return 2

    if init.get("op") != "init":
        sys.stderr.write(f"runner: expected init, got {init!r}\n")
        return 2

    pid = init.get("pid")
    payload = init.get("payload") or {}

    # 2) Send ready.
    chan.send({"op": "ready", "pid": pid})

    # 3) Behavior dispatch.
    exit_kind = "completed"
    summary = "ok"
    rc = 0

    if behavior == "loop":
        # Busy loop until killed externally. Used by tests to verify
        # the wall-clock killer works.
        try:
            while True:
                # Look for a stop message non-blocking-ish.
                # In practice the wall-killer arrives before we'd
                # poll — this loop just chews CPU.
                pass
        except KeyboardInterrupt:
            exit_kind = "cancelled"
            summary = "interrupted"

    elif behavior == "crash":
        chan.send({"op": "log", "level": "error",
                   "msg": "intentional crash"})
        return 1

    elif behavior.startswith("slow="):
        try:
            n = float(behavior.split("=", 1)[1])
        except ValueError:
            n = 1.0
        time.sleep(n)
        chan.send({"op": "iteration_done", "iter": 1,
                   "tokens": 0, "cost_micro": 0})

    elif behavior.startswith("chunks="):
        # RFC 0026 test path: emit N text chunks then exit.
        try:
            n = int(behavior.split("=", 1)[1])
        except ValueError:
            n = 1
        for i in range(1, n + 1):
            chan.send({
                "op":       "chunk",
                "kind":     "text",
                "content":  f"chunk-{i}",
                "metadata": {"i": i, "of": n},
            })

    elif behavior.startswith("charge="):
        spec = behavior.split("=", 1)[1]
        try:
            dim, amount_s = spec.split(":", 1)
            amount = int(amount_s)
        except ValueError:
            chan.send({"op": "log", "level": "error",
                       "msg": f"bad CC_RUNNER_BEHAVIOR={behavior!r}"})
            return 2
        chan.send({"op": "charge", "dim": dim, "amount": amount})

    elif behavior.startswith("tool_call="):
        # Test path for RFC 0021. Body is JSON like:
        #   {"tool": "Read", "args": {"path": "/tmp/x"},
        #    "tool_call_id": "abc"}
        # Emits one tool_call, waits for tool_response, logs the
        # response, then exits with exit_kind reflecting the
        # response's ok flag.
        spec = behavior.split("=", 1)[1]
        try:
            call_body = json.loads(spec)
        except json.JSONDecodeError as e:
            chan.send({"op": "log", "level": "error",
                       "msg": f"bad tool_call body: {e}"})
            return 2
        call_msg = {
            "op":           "tool_call",
            "tool_call_id": call_body.get("tool_call_id", "test-1"),
            "tool":         call_body.get("tool", ""),
            "args":         call_body.get("args", {}),
        }
        chan.send(call_msg)
        try:
            response = chan.recv(timeout=10.0)
        except Exception as e:
            chan.send({"op": "log", "level": "error",
                       "msg": f"tool_response recv failed: {e}"})
            chan.send({"op": "exit", "exit_kind": "failed",
                       "summary": f"recv failed: {e}",
                       "metadata": {"tool_response": None}})
            return 1
        chan.send({"op": "log", "level": "info",
                   "msg": f"tool_response: ok={response.get('ok')} "
                          f"error={response.get('error')}"})
        # Carry the tool_response back to the supervisor via exit
        # metadata so tests can assert on the full payload.
        ok = bool(response.get("ok"))
        exit_kind = "completed" if ok else "failed"
        # Note: don't fail on tool denial — the runner's job is to
        # surface the response to the caller.
        chan.send({
            "op":         "exit",
            "exit_kind":  "completed",
            "summary":    f"tool_response.ok={ok}",
            "text":       json.dumps(response),
            "metadata":   {"tool_response": response},
        })
        return 0

    else:  # echo (default)
        chan.send({"op": "iteration_start", "iter": 1})
        chan.send({"op": "log", "level": "info",
                   "msg": f"received payload: {json.dumps(payload)}"})
        chan.send({"op": "iteration_done", "iter": 1,
                   "tokens": 0, "cost_micro": 0})

    # 4) Send exit message.
    chan.send({
        "op":        "exit",
        "exit_kind": exit_kind,
        "summary":   summary,
    })
    return rc


if __name__ == "__main__":
    sys.exit(main())
