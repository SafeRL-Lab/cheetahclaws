"""Persistent Python console (REPL) tool — working memory outside the context.

The `Python` tool runs code in a LONG-LIVED subprocess kernel whose namespace
survives across calls within a session. Variables, imports, and loaded data stay
alive between invocations, so the model can:

  * explore large data structures incrementally (scan a directory tree ONCE into
    a variable, then filter/query it across many turns), and
  * keep that data in the kernel's heap instead of the conversation — it never
    pays tokens to re-read or re-transmit it. Only the small printed slices land
    in the context. This is the whole point: a Python heap as working memory that
    lives OUTSIDE the context window.

Design (why a subprocess, not in-process exec):
  * Isolation. Arbitrary model-written code runs in loops; an infinite loop,
    `os._exit()`, or a segfault kills the KERNEL, never CheetahClaws. The parent detects
    the dead child and transparently restarts it on the next call.
  * Clean timeouts. The parent enforces a wall-clock timeout by waiting on a
    reader thread; a runaway cell is killed (its state is lost — that's the cost
    of an infinite loop) and CheetahClaws keeps going.

Protocol: newline-delimited JSON over the child's stdin/stdout. Code and captured
output are base64-encoded so arbitrary text (newlines, unicode) stays on one line
and never collides with the framing. The worker redirects stdout/stderr into a
buffer during exec, so the model's `print()`s are captured while the framing
channel (`sys.__stdout__`) stays clean.

This module is BOTH the parent (imported as `pyconsole`, registers the tool) and
the worker (run as `python pyconsole.py --pykernel-worker`). Top-level imports are
stdlib-only on purpose: the worker process must not drag in the CheetahClaws runtime, so
the `tool_registry` import is done lazily inside `register_python_console()`.
"""
from __future__ import annotations

import ast
import base64
import io
import json
import os
import queue
import reprlib
import subprocess
import sys
import threading
import traceback

# Hard cap on how many characters any single stream (stdout / stderr / the echoed
# value) may contribute to a result. This is the real cost/caching guardrail: the
# ~80-line cap alone does NOT stop one enormous line (e.g. print('x' * 10_000_000)
# or repr() of a 100 MB structure) from flooding the context and being re-sent
# every turn — the exact bug that made a big Edit diff so expensive. We truncate
# at the SOURCE, before anything crosses the pipe, so megabytes never travel.
_MAX_STREAM_CHARS = 20_000

# Bounded repr for the REPL echo: reprlib limits container/str sizes so echoing a
# huge object shows its SHAPE (e.g. [0, 1, 2, ..., 9998, 9999]) without ever
# materialising a gigabyte-long repr string. Limits are generous so normal small
# results still print in full.
_repr = reprlib.Repr()
_repr.maxlist = _repr.maxtuple = _repr.maxset = _repr.maxfrozenset = 200
_repr.maxdict = 100
_repr.maxstring = _repr.maxother = 2_000
_repr.maxlevel = 8


# ── framing helpers ──────────────────────────────────────────────────────────

def _b64e(s: str) -> str:
    return base64.b64encode((s or "").encode("utf-8")).decode("ascii")


def _b64d(s: str) -> str:
    return base64.b64decode((s or "").encode("ascii")).decode("utf-8", "replace")


def _clean_tb(e: BaseException) -> str:
    """Format a traceback showing only the model's own code frames, dropping the
    worker's internal exec/eval frames so errors read like a plain REPL."""
    tb = e.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_filename != "<pyconsole>":
        tb = tb.tb_next
    if tb is None:
        # No user-code frame (e.g. SyntaxError at parse time) — just the error,
        # which for SyntaxError still carries the source line and caret.
        return "".join(traceback.format_exception_only(type(e), e))
    return "".join(traceback.format_exception(type(e), e, tb))


def _trunc_chars(text: str, limit: int = _MAX_STREAM_CHARS) -> str:
    """Cap a single stream's character count at the source (before transmit)."""
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    return (
        text[:limit]
        + f"\n[... {dropped} more chars truncated — keep large results in a "
        "variable and print only the slice you need ...]"
    )


def _safe_repr(v: object) -> str:
    """Bounded repr for the REPL echo — never builds a gigabyte-long string."""
    try:
        s = _repr.repr(v)
    except Exception:
        try:
            return f"<{type(v).__name__} (unreprable)>"
        except Exception:
            return "<unreprable>"
    return _trunc_chars(s)


# ── worker (child process) ───────────────────────────────────────────────────

def _worker_main() -> None:
    """Run the persistent kernel loop. One JSON request per line on stdin; one
    JSON response per line on the real stdout. exec state lives in `ns`."""
    real_out = sys.__stdout__ or sys.stdout

    def send(obj: dict) -> None:
        real_out.write(json.dumps(obj) + "\n")
        real_out.flush()

    ns: dict = {"__name__": "__pyconsole__", "__builtins__": __builtins__}

    send({"ready": True})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        code = _b64d(req.get("code_b64", ""))

        out, err = io.StringIO(), io.StringIO()
        value = None
        error = None
        ok = True
        old_out, old_err, old_in = sys.stdout, sys.stderr, sys.stdin
        sys.stdout, sys.stderr = out, err
        # Redirect stdin to an empty stream: code that calls input() gets a clean
        # EOFError instead of blocking on — and eating — the JSON protocol stream
        # that the kernel loop reads from the REAL stdin.
        sys.stdin = io.StringIO("")
        try:
            block = ast.parse(code, mode="exec")
            # REPL nicety: if the last statement is a bare expression, eval it
            # separately so its value can be echoed like an interactive prompt.
            last_expr = None
            if block.body:
                last_node = block.body[-1]
                if isinstance(last_node, ast.Expr):
                    block.body.pop()
                    last_expr = ast.Expression(last_node.value)
            if block.body:
                exec(compile(block, "<pyconsole>", "exec"), ns)
            if last_expr is not None:
                v = eval(compile(last_expr, "<pyconsole>", "eval"), ns)
                if v is not None:
                    value = _safe_repr(v)
        except SystemExit:
            error = "SystemExit was raised inside the kernel and ignored. The " \
                    "namespace is intact; pass reset=true to restart if needed."
            ok = False
        except BaseException as e:
            error = _clean_tb(e)
            ok = False
        finally:
            sys.stdout, sys.stderr, sys.stdin = old_out, old_err, old_in

        # Cap each stream at the source so megabytes never cross the pipe or land
        # in the conversation (the cost/caching guardrail).
        out_s = _trunc_chars(out.getvalue())
        err_s = _trunc_chars(err.getvalue())

        send({
            "stdout_b64": _b64e(out_s),
            "stderr_b64": _b64e(err_s),
            "value_b64": _b64e(_trunc_chars(value or "")),
            "error_b64": _b64e(_trunc_chars(error or "")),
            "ok": ok,
        })


# ── parent (kernel manager) ──────────────────────────────────────────────────

class _PyKernel:
    """Manages one persistent worker subprocess for the whole CheetahClaws session.

    A single reader thread drains the child's stdout into a Queue so the parent
    can wait for a response with a wall-clock timeout (cross-platform — no select
    on pipes, which Windows can't do). All access is serialised by a lock so
    concurrent sub-agent calls can't interleave on the pipe.
    """

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.q: "queue.Queue[str | None]" = queue.Queue()
        self.lock = threading.RLock()
        self.cwd = os.getcwd()

    def _drain(self, proc: subprocess.Popen, q: "queue.Queue[str | None]") -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                q.put(line)
        except Exception:
            pass
        finally:
            q.put(None)  # sentinel: child stdout closed (process died)

    def _start(self) -> str | None:
        """Launch a fresh worker. Returns an error string, or None on success."""
        self.q = queue.Queue()
        try:
            self.proc = subprocess.Popen(
                [sys.executable, "-u", os.path.abspath(__file__), "--pykernel-worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=self.cwd,
            )
        except Exception as e:
            self.proc = None
            return f"Failed to start Python kernel: {e}"
        threading.Thread(
            target=self._drain, args=(self.proc, self.q), daemon=True
        ).start()
        # Wait for the ready handshake.
        try:
            first = self.q.get(timeout=15)
        except queue.Empty:
            self._kill()
            return "Python kernel did not come up (handshake timeout)."
        if first is None:
            self._kill()
            return "Python kernel died during startup."
        return None

    def _kill(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def reset(self) -> None:
        with self.lock:
            self._kill()

    def run(self, code: str, timeout: float) -> dict:
        with self.lock:
            if self.proc is None or self.proc.poll() is not None:
                err = self._start()
                if err:
                    return {"fatal": err}
            assert self.proc is not None and self.proc.stdin is not None
            # Swallow any unsolicited output left in the queue between calls (e.g.
            # a background thread the previous cell spawned that printed after its
            # response). Each call fully consumes its own response under the lock,
            # so anything here now is noise that must not be mistaken for a reply.
            while True:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break
            try:
                self.proc.stdin.write(json.dumps({"code_b64": _b64e(code)}) + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                self._kill()
                return {"fatal": f"Kernel pipe broke while sending: {e}"}
            try:
                line = self.q.get(timeout=timeout)
            except queue.Empty:
                # Runaway cell (infinite loop / blocking call). Kill it — its
                # in-memory state is forfeit — and let the NEXT call restart clean.
                self._kill()
                return {"timeout": True}
            if line is None:
                self._kill()
                return {"fatal": "Kernel process exited unexpectedly during exec."}
            try:
                return json.loads(line)
            except Exception as e:
                self._kill()
                return {"fatal": f"Malformed kernel response: {e}"}


# One kernel per CheetahClaws process (sub-agents share it; the lock keeps it safe).
_KERNEL = _PyKernel()

_MAX_OUT_LINES = 80
_MAX_OUT_CHARS = 24_000  # backstop: total result never exceeds this in context


def _cap(text: str) -> str:
    lines = text.splitlines()
    if len(lines) > _MAX_OUT_LINES:
        remaining = len(lines) - _MAX_OUT_LINES
        text = (
            "\n".join(lines[:_MAX_OUT_LINES])
            + f"\n\n[... {remaining} more lines — keep large results in a variable "
            "and print only the slice you need ...]"
        )
    # Char backstop: even within the line cap, guard against a few very long lines
    # so a tool result can never balloon the context (and get re-sent every turn).
    if len(text) > _MAX_OUT_CHARS:
        text = text[:_MAX_OUT_CHARS] + "\n[... output truncated — print a smaller slice ...]"
    return text


def _format(resp: dict) -> str:
    if "fatal" in resp:
        return f"Error: {resp['fatal']}"
    if resp.get("timeout"):
        return (
            "Error: execution exceeded the timeout and the kernel was killed "
            "(its in-memory state was lost). An infinite loop or a blocking call "
            "is the usual cause. The next Python call starts a fresh kernel."
        )
    stdout = _b64d(resp.get("stdout_b64", ""))
    stderr = _b64d(resp.get("stderr_b64", ""))
    value = _b64d(resp.get("value_b64", ""))
    error = _b64d(resp.get("error_b64", ""))

    parts: list[str] = []
    if stdout:
        parts.append(stdout.rstrip("\n"))
    if stderr:
        parts.append("[stderr]\n" + stderr.rstrip("\n"))
    if error:
        parts.append(error.rstrip("\n"))
    elif value:
        parts.append(f"=> {value}")

    if not parts:
        return "(ok — no output; the statement ran and its result was None)"
    return _cap("\n".join(parts))


def _py_console(params: dict, config: dict) -> str:
    code = params.get("code", "")
    if not isinstance(code, str) or not code.strip():
        return "Error: 'code' must be a non-empty string of Python to run."
    if params.get("reset"):
        _KERNEL.reset()
        if not code.strip():
            return "Python kernel namespace reset."
    try:
        timeout = float(params.get("timeout", 30))
    except (TypeError, ValueError):
        timeout = 30.0
    timeout = max(1.0, min(timeout, 300.0))
    resp = _KERNEL.run(code, timeout)
    return _format(resp)


# ── registration ─────────────────────────────────────────────────────────────

_DESCRIPTION = (
    "Persistent Python console (REPL). State PERSISTS across calls within the "
    "session — variables, imports, and any data you assign stay alive between "
    "invocations. Runs in an ISOLATED subprocess kernel, so an infinite loop or "
    "a crash cannot take CheetahClaws down.\n\n"
    "USE IT AS WORKING MEMORY OUTSIDE THE CONTEXT WINDOW. When you explore large "
    "data (a big directory tree, a parsed file, an API dump, a dataframe), load "
    "it ONCE into a variable, then filter and query it on later calls and print "
    "only small slices. The data lives in the kernel's heap, NOT in the "
    "conversation, so you never spend tokens re-reading or re-transmitting it.\n\n"
    "Example flow:\n"
    "  1) paths = list(Path('.').rglob('*'))     # scan once, held in memory\n"
    "  2) len(paths)                             # 48213  (echoed like a REPL)\n"
    "  3) [p for p in paths if 'config' in p.name.lower()][:10] # cheap slice\n\n"
    "The value of a trailing bare expression is echoed as `=> <repr>`. Output is "
    "capped at ~80 lines — keep big results in variables and print summaries. "
    "Pass reset=true to wipe the namespace and start a fresh kernel."
)


def register_python_console() -> None:
    from cheetahclaws.tool_registry import ToolDef, register_tool

    register_tool(ToolDef(
        name="Python",
        schema={
            "name": "Python",
            "description": _DESCRIPTION,
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python source to execute in the persistent kernel. "
                            "May span multiple lines. A trailing bare expression "
                            "is echoed like an interactive prompt."
                        ),
                    },
                    "reset": {
                        "type": "boolean",
                        "description": (
                            "Wipe the namespace and restart the kernel before "
                            "running (default false)."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            "Wall-clock seconds before the cell is killed "
                            "(default 30, max 300)."
                        ),
                    },
                },
                "required": ["code"],
            },
        },
        func=_py_console,
        read_only=False,
        concurrent_safe=False,
    ))


if __name__ == "__main__":
    # Launched as the kernel worker subprocess (see _PyKernel._start). Runs the
    # exec loop only — never imports the cheetahclaws package, keeping the worker
    # lightweight and isolated.
    if "--pykernel-worker" in sys.argv:
        _worker_main()
else:
    # Normal import (cheetahclaws.tools auto-imports this submodule) — register
    # the tool, matching the self-registering pattern of the other tools/ modules.
    try:
        register_python_console()
    except Exception:
        pass  # never block import over the Python console tool
