"""exec_tool.py — argv-only bounded-shell tool (RFC 0023).

The most dangerous tool we ship. The threat model + invariants
are documented in RFC 0023; the boundary in this module is:

  * No shell=True — ever.
  * argv[0] must be absolute path, file, fs_grants("r") covered.
  * Default env scrubbed to a minimal allowlist + caller-provided
    additions; secrets like ANTHROPIC_API_KEY are dropped.
  * RLIMIT (cpu, memory, fsize, nofile) enforced via the existing
    sandbox primitives.
  * stdout / stderr each truncated to 256 KB by default.
  * Wall-clock timeout enforced (default 60s, max 600s).

NOT registered by register_builtin_tools — operators must call
register_exec_tool(registry) explicitly. Agents must hold "Exec"
in tool_grants AND have the binary path covered by fs_grants
"r".
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..sandbox import (
    SandboxPolicy,
    apply_rlimits_in_child,
    run_sandboxed,
)
from ..sandbox import _RunControl, _wall_clock_killer    # RFC 0028
from .registry import (
    Tool,
    ToolContext,
    ToolFailed,
    ToolFsDenied,
    ToolInvalidArgs,
    ToolRegistry,
)

if TYPE_CHECKING:
    from ..api import Kernel


# ── Defaults ────────────────────────────────────────────────────────────


DEFAULT_TIMEOUT_S      = 60
MAX_TIMEOUT_S          = 600
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024
MAX_OUTPUT_BYTES_LIMIT   = 4 * 1024 * 1024
MIN_OUTPUT_BYTES_LIMIT   = 1024

DEFAULT_MEMORY_BYTES   = 512 * 1024 * 1024     # 512 MB
DEFAULT_FSIZE_BYTES    = 64 * 1024 * 1024      # 64 MB
DEFAULT_NOFILE         = 256


# Env keys safe to expose to children. Anything not in this set
# AND not in caller-provided ``env`` is dropped.
_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "LANG", "LC_ALL", "USER", "TERM", "SHELL",
})

# Default env values when those keys aren't set in the parent.
_DEFAULT_ENV = {
    "PATH":   "/usr/local/bin:/usr/bin:/bin",
    "HOME":   "/tmp",
    "LANG":   "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "TERM":   "dumb",
    "SHELL":  "/bin/sh",
}


# ── Handler ─────────────────────────────────────────────────────────────


def _scrub_env(parent_env: dict, user_env: dict) -> dict:
    """Build the child env: safe-list from parent, then
    user-supplied additions. Reserved-key checks are caller's
    responsibility (Exec validates ``args.env``)."""
    out = dict(_DEFAULT_ENV)
    for k in _SAFE_ENV_KEYS:
        if k in parent_env:
            out[k] = parent_env[k]
    # Caller additions override (but can't break the type contract).
    for k, v in user_env.items():
        out[k] = v
    return out


def _validate_argv(argv) -> list:
    if not isinstance(argv, list) or not argv:
        raise ToolInvalidArgs("'argv' must be a non-empty list")
    for i, a in enumerate(argv):
        if not isinstance(a, str) or not a:
            raise ToolInvalidArgs(
                f"argv[{i}] must be a non-empty string, got {type(a).__name__}",
            )
    if not argv[0].startswith("/"):
        raise ToolInvalidArgs(
            f"argv[0] must be an absolute path, got {argv[0]!r}; "
            "Exec does not perform PATH lookup",
        )
    p = Path(argv[0])
    if not p.is_file():
        raise ToolFailed(f"argv[0] not a file: {argv[0]}")
    return list(argv)


def _validate_user_env(user_env) -> dict:
    if user_env is None:
        return {}
    if not isinstance(user_env, dict):
        raise ToolInvalidArgs(
            f"'env' must be a dict, got {type(user_env).__name__}",
        )
    out = {}
    for k, v in user_env.items():
        if not isinstance(k, str) or not k:
            raise ToolInvalidArgs(
                f"env key must be non-empty str, got {k!r}",
            )
        if k.startswith("_"):
            raise ToolInvalidArgs(
                f"env keys starting with '_' are reserved: {k!r}",
            )
        if not isinstance(v, str):
            raise ToolInvalidArgs(
                f"env[{k!r}] must be str, got {type(v).__name__}",
            )
        out[k] = v
    return out


def _validate_timeout(t) -> int:
    if t is None:
        return DEFAULT_TIMEOUT_S
    if not isinstance(t, (int, float)):
        raise ToolInvalidArgs(
            f"'timeout_s' must be a number, got {type(t).__name__}",
        )
    t_int = int(t)
    if t_int < 1 or t_int > MAX_TIMEOUT_S:
        raise ToolInvalidArgs(
            f"'timeout_s' must be in [1, {MAX_TIMEOUT_S}], got {t_int}",
        )
    return t_int


def _validate_max_output(n) -> int:
    if n is None:
        return DEFAULT_MAX_OUTPUT_BYTES
    if not isinstance(n, int) or \
            n < MIN_OUTPUT_BYTES_LIMIT or n > MAX_OUTPUT_BYTES_LIMIT:
        raise ToolInvalidArgs(
            f"'max_output_bytes' must be int in "
            f"[{MIN_OUTPUT_BYTES_LIMIT}, {MAX_OUTPUT_BYTES_LIMIT}], "
            f"got {n!r}",
        )
    return n


def _validate_stream(s) -> bool:
    if s is None:
        return False
    if not isinstance(s, bool):
        raise ToolInvalidArgs(
            f"'stream' must be bool, got {type(s).__name__}",
        )
    return s


def _validate_cwd(cwd) -> str:
    if cwd is None:
        return None  # type: ignore[return-value]
    if not isinstance(cwd, str) or not cwd:
        raise ToolInvalidArgs(
            f"'cwd' must be non-empty str, got {cwd!r}",
        )
    if not cwd.startswith("/"):
        raise ToolInvalidArgs(
            f"'cwd' must be absolute, got {cwd!r}",
        )
    p = Path(cwd)
    if not p.is_dir():
        raise ToolFailed(f"cwd not a directory: {cwd}")
    return cwd


def _run_streaming(
    argv: list, policy: SandboxPolicy, env: dict,
    cwd, max_output_bytes: int, ctx: ToolContext,
) -> dict:
    """RFC 0028: Popen + per-line reader threads + queue-serialized
    chunk emission. Same wall-clock + RLIMIT enforcement as
    run_sandboxed; output truncated identically."""
    import queue
    import subprocess
    import threading
    import time

    preexec = apply_rlimits_in_child(policy)
    start = time.monotonic()
    proc = subprocess.Popen(
        argv,
        stdin  = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        env    = env,
        cwd    = cwd,
        preexec_fn = preexec,
    )

    q: "queue.Queue[tuple[str, bytes | None]]" = queue.Queue()

    def _reader(pipe, kind: str) -> None:
        try:
            for line in iter(pipe.readline, b""):
                if not line:
                    break
                q.put((kind, line))
        except Exception:
            pass
        finally:
            q.put((kind, None))    # EOF marker
            try:
                pipe.close()
            except Exception:
                pass

    t_out = threading.Thread(
        target=_reader, args=(proc.stdout, "stdout"),
        daemon=True, name="exec-stream-stdout",
    )
    t_err = threading.Thread(
        target=_reader, args=(proc.stderr, "stderr"),
        daemon=True, name="exec-stream-stderr",
    )
    t_out.start()
    t_err.start()

    # Wall-clock killer (same primitive as run_sandboxed).
    ctl = _RunControl()
    killer: threading.Thread = None    # type: ignore[assignment]
    if policy.wall_seconds is not None:
        deadline = time.monotonic() + policy.wall_seconds
        killer = threading.Thread(
            target=_wall_clock_killer,
            args=(proc, deadline, ctl),
            daemon=True, name="exec-stream-killer",
        )
        killer.start()

    out_bytes_list: list = []
    err_bytes_list: list = []
    out_total = 0
    err_total = 0
    eof_count = 0

    while eof_count < 2:
        try:
            kind, line = q.get(timeout=0.5)
        except queue.Empty:
            if proc.poll() is not None and not (t_out.is_alive() or t_err.is_alive()):
                break
            continue
        if line is None:
            eof_count += 1
            continue
        try:
            text = line.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        # Always emit chunk — even past truncation cap, so the UI
        # keeps showing progress.
        try:
            ctx.on_chunk({
                "op":       "chunk",
                "kind":     kind,
                "content":  text,
                "metadata": {"tool": "Exec"},
            })
        except Exception:
            pass
        if kind == "stdout":
            if out_total < max_output_bytes:
                out_bytes_list.append(line)
                out_total += len(line)
        else:
            if err_total < max_output_bytes:
                err_bytes_list.append(line)
                err_total += len(line)

    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
        proc.wait()
    ctl.finished = True
    if killer is not None:
        killer.join(timeout=2.0)

    duration = time.monotonic() - start
    out_truncated = out_total > max_output_bytes
    err_truncated = err_total > max_output_bytes
    out_bytes = b"".join(out_bytes_list)
    err_bytes = b"".join(err_bytes_list)
    if out_truncated:
        out_bytes = out_bytes[:max_output_bytes]
    if err_truncated:
        err_bytes = err_bytes[:max_output_bytes]
    return {
        "exit_code":        int(proc.returncode),
        "stdout":           out_bytes.decode("utf-8", errors="replace"),
        "stderr":           err_bytes.decode("utf-8", errors="replace"),
        "stdout_truncated": out_truncated,
        "stderr_truncated": err_truncated,
        "duration_s":       round(float(duration), 4),
        "timed_out":        bool(ctl.timed_out),
    }


def exec_handler(args: dict, ctx: ToolContext) -> dict:
    """The Exec tool's handler. Wraps run_sandboxed under a tight
    SandboxPolicy and a scrubbed env."""
    import os as _os

    argv             = _validate_argv(args.get("argv"))
    user_env         = _validate_user_env(args.get("env"))
    timeout_s        = _validate_timeout(args.get("timeout_s"))
    max_output_bytes = _validate_max_output(args.get("max_output_bytes"))
    cwd              = _validate_cwd(args.get("cwd"))
    stream_requested = _validate_stream(args.get("stream"))

    # fs_grants check on argv[0]: handler does this directly because
    # ``requires_fs`` only supports top-level args_key extraction.
    if ctx.kernel is not None:
        if not ctx.kernel.cap.check_fs(ctx.pid, argv[0], "r"):
            raise ToolFsDenied(
                f"agent {ctx.pid} not granted 'r' on {argv[0]!r}",
            )
        if cwd is not None and not ctx.kernel.cap.check_fs(
            ctx.pid, cwd, "r",
        ):
            raise ToolFsDenied(
                f"agent {ctx.pid} not granted 'r' on cwd {cwd!r}",
            )

    # Build scrubbed env.
    env = _scrub_env(_os.environ, user_env)

    # Build sandbox policy.
    policy = SandboxPolicy(
        cpu_seconds   = max(timeout_s, 1),
        memory_bytes  = DEFAULT_MEMORY_BYTES,
        fsize_bytes   = DEFAULT_FSIZE_BYTES,
        nofile        = DEFAULT_NOFILE,
        wall_seconds  = float(timeout_s),
        new_session   = True,
    )

    # RFC 0028: streaming path requires both opt-in and a
    # supervisor-supplied chunk callback. If either is missing,
    # fall back to the buffered run_sandboxed path — byte-for-byte
    # identical to the pre-RFC behaviour.
    if stream_requested and ctx.on_chunk is not None:
        return _run_streaming(
            argv, policy, env, cwd, max_output_bytes, ctx,
        )

    # Run.
    result = run_sandboxed(
        argv, policy,
        env=env, cwd=cwd,
        capture_stdout=True, capture_stderr=True,
    )

    # Truncate output. The supervisor side decoding errors=replace
    # so we never crash on weird bytes.
    out_bytes = result.stdout or b""
    err_bytes = result.stderr or b""
    out_truncated = len(out_bytes) > max_output_bytes
    err_truncated = len(err_bytes) > max_output_bytes
    if out_truncated:
        out_bytes = out_bytes[:max_output_bytes]
    if err_truncated:
        err_bytes = err_bytes[:max_output_bytes]
    out_text = out_bytes.decode("utf-8", errors="replace")
    err_text = err_bytes.decode("utf-8", errors="replace")

    return {
        "exit_code":        int(result.exit_code),
        "stdout":           out_text,
        "stderr":           err_text,
        "stdout_truncated": out_truncated,
        "stderr_truncated": err_truncated,
        "duration_s":       round(float(result.duration_s), 4),
        "timed_out":        bool(result.timed_out),
    }


EXEC_TOOL = Tool(
    name="Exec",
    description=(
        "Execute a binary with a fixed argv list (no shell). "
        "Requires 'Exec' tool capability AND fs_grants 'r' on argv[0]. "
        "Output bounded; env scrubbed; RLIMITs enforced. "
        "See RFC 0023 for the threat model."
    ),
    handler=exec_handler,
    requires_capability=True,
    requires_fs=(),  # Handler does custom fs_grants check on argv[0] / cwd.
)


def register_exec_tool(
    registry: ToolRegistry,
    *,
    kernel=None,
) -> str:
    """Register the Exec tool. **Opt-in** — not called by
    ``register_builtin_tools``. Operators must explicitly enable
    this tool because the threat surface is significantly larger
    than Echo / Read / Write.

    Returns the tool name registered.
    """
    del kernel  # unused; reserved for symmetry with builtins
    registry.register(EXEC_TOOL)
    return EXEC_TOOL.name


__all__ = ["EXEC_TOOL", "exec_handler", "register_exec_tool"]
