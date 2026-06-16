"""git_tool.py — read-only git inspector (RFC 0032).

Wraps a small allowlist of read-only git subcommands behind
the same RLIMIT/wall-clock sandbox primitives Exec uses.
Strict op + flag allowlist; ref / path validators; ``-C
<repo>`` always set so the child can't traverse out of the
configured repo. Opt-in (NOT in register_builtin_tools).
"""
from __future__ import annotations

import os as _os
import re
import shutil
from pathlib import Path

from ..sandbox import SandboxPolicy, run_sandboxed
from .exec_tool import _scrub_env, _DEFAULT_ENV
from .registry import (
    Tool,
    ToolContext,
    ToolFailed,
    ToolFsDenied,
    ToolInvalidArgs,
    ToolRegistry,
)


# ── Defaults / limits ──────────────────────────────────────────────────


DEFAULT_GIT_TIMEOUT_S = 30
MIN_GIT_TIMEOUT_S     = 1
MAX_GIT_TIMEOUT_S     = 120

DEFAULT_GIT_STDOUT_CAP = 1024 * 1024
DEFAULT_GIT_STDERR_CAP = 256 * 1024

DEFAULT_MEMORY_BYTES = 512 * 1024 * 1024
DEFAULT_FSIZE_BYTES  = 64 * 1024 * 1024
DEFAULT_NOFILE       = 256


# ── Op + flag allowlist ────────────────────────────────────────────────


_REF_RE  = re.compile(r"^[A-Za-z0-9_./~^@-]{1,200}$")
_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]{1,512}$")

# Per-op allowed extra flags. Keys must be the user-facing op name.
# Each entry is a frozenset of allowed flag prefixes (we match prefix
# so ``-n`` accepts both ``-n`` and the value following it).
_ALLOWED_FLAGS = {
    "status":   frozenset({"--porcelain", "-z", "--branch"}),
    "log":      frozenset({"--oneline", "--graph", "-n", "--since",
                            "--until", "--author", "--max-count",
                            "--no-color", "--date", "--name-only",
                            "--name-status", "--abbrev-commit"}),
    "diff":     frozenset({"--no-color", "--stat", "--shortstat",
                            "--name-only", "--name-status",
                            "--unified", "-U", "--cached",
                            "--word-diff"}),
    "show":     frozenset({"--no-color", "--stat", "--shortstat",
                            "--name-only", "--name-status",
                            "--no-patch", "-s"}),
    "branch":   frozenset({"-a", "-r", "-v", "--no-color",
                            "--list"}),
    "blame":    frozenset({"-l", "-w", "-M", "-C",
                            "--no-color", "-L", "--since"}),
    "ls_files": frozenset({"-z", "--cached", "--others",
                            "--exclude-standard", "--full-name"}),
    "rev_parse": frozenset({"--short", "--abbrev-ref",
                              "--show-toplevel", "--git-dir",
                              "HEAD"}),
}

_OP_BASE_ARGV = {
    "status":   ["status", "--porcelain"],
    "log":      ["log"],
    "diff":     ["diff"],
    "show":     ["show"],
    "branch":   ["branch", "-a", "--no-color"],
    "blame":    ["blame"],
    "ls_files": ["ls-files"],
    "rev_parse": ["rev-parse"],
}


# ── Validation ─────────────────────────────────────────────────────────


def _validate_op(op) -> str:
    if not isinstance(op, str) or op not in _ALLOWED_FLAGS:
        raise ToolInvalidArgs(
            f"'op' must be one of {sorted(_ALLOWED_FLAGS)}, "
            f"got {op!r}",
        )
    return op


def _validate_repo(repo) -> str:
    if not isinstance(repo, str) or not repo:
        raise ToolInvalidArgs("'repo' must be non-empty string")
    if not repo.startswith("/"):
        raise ToolInvalidArgs(
            f"'repo' must be absolute, got {repo!r}",
        )
    p = Path(repo)
    if not p.is_dir():
        raise ToolFailed(f"repo not a directory: {repo!r}")
    if not (p / ".git").exists() and not (p / "HEAD").exists():
        raise ToolFailed(
            f"repo has no .git/ (and no HEAD): {repo!r}",
        )
    return repo


def _validate_ref(ref):
    if ref is None:
        return None
    if not isinstance(ref, str) or not ref:
        raise ToolInvalidArgs("'ref' must be non-empty string")
    if not _REF_RE.match(ref):
        raise ToolInvalidArgs(
            f"'ref' contains disallowed chars: {ref!r}",
        )
    return ref


def _validate_path(path_arg, repo: str):
    if path_arg is None:
        return None
    if not isinstance(path_arg, str) or not path_arg:
        raise ToolInvalidArgs("'path' must be non-empty string")
    if path_arg.startswith("/"):
        raise ToolInvalidArgs(
            f"'path' must be relative to repo, got {path_arg!r}",
        )
    if ".." in Path(path_arg).parts:
        raise ToolInvalidArgs(
            f"'path' may not contain '..' segments: {path_arg!r}",
        )
    if not _PATH_RE.match(path_arg):
        raise ToolInvalidArgs(
            f"'path' contains disallowed chars: {path_arg!r}",
        )
    return path_arg


def _validate_args(args_in, op: str) -> list:
    if args_in is None:
        return []
    if not isinstance(args_in, list):
        raise ToolInvalidArgs("'args' must be a list of strings")
    allowed = _ALLOWED_FLAGS[op]
    out = []
    for a in args_in:
        if not isinstance(a, str) or not a:
            raise ToolInvalidArgs("each arg must be non-empty string")
        if any(ch in a for ch in ("\n", "\r", "\x00")):
            raise ToolInvalidArgs(
                f"arg contains control char: {a!r}",
            )
        # Numeric argv (the value after -n / -U / etc.) — allow it.
        if a.lstrip("-").replace(".", "", 1).isdigit():
            out.append(a)
            continue
        # Flag form: must be prefix-allowlisted.
        if a.startswith("-"):
            base = a.split("=", 1)[0]
            if base not in allowed:
                raise ToolInvalidArgs(
                    f"flag {a!r} not allowed for op {op!r}; "
                    f"allowed: {sorted(allowed)}",
                )
        # Positional non-flag — allow only if it matches the path
        # regex or the ref regex (defensive — the dedicated `ref`
        # / `path` keys are preferred).
        elif _REF_RE.match(a) or _PATH_RE.match(a):
            pass
        else:
            raise ToolInvalidArgs(
                f"positional arg {a!r} contains disallowed chars",
            )
        out.append(a)
    return out


def _validate_timeout(t) -> int:
    if t is None:
        return DEFAULT_GIT_TIMEOUT_S
    if not isinstance(t, (int, float)):
        raise ToolInvalidArgs("'timeout_s' must be a number")
    n = int(t)
    if n < MIN_GIT_TIMEOUT_S or n > MAX_GIT_TIMEOUT_S:
        raise ToolInvalidArgs(
            f"'timeout_s' must be in "
            f"[{MIN_GIT_TIMEOUT_S}, {MAX_GIT_TIMEOUT_S}]",
        )
    return n


# ── Handler ────────────────────────────────────────────────────────────


def _resolve_git_binary() -> str | None:
    return shutil.which("git")


def git_handler(args: dict, ctx: ToolContext) -> dict:
    # Validation order: cheap argv-shape checks first (fail-fast on
    # malformed input), expensive disk check last. Ensures a malformed
    # ref/path/args/timeout produces a clear ``invalid_args`` error
    # without requiring the repo to exist on disk yet.
    op        = _validate_op(args.get("op"))
    ref       = _validate_ref(args.get("ref"))
    path_arg  = _validate_path(args.get("path"), repo=None)
    extra     = _validate_args(args.get("args"), op)
    timeout_s = _validate_timeout(args.get("timeout_s"))
    repo      = _validate_repo(args.get("repo"))

    git_bin = _resolve_git_binary()
    if git_bin is None:
        raise ToolFailed("git binary not available on this host")

    # fs check: agent needs "r" on the repo AND the git binary.
    if ctx.kernel is not None:
        if not ctx.kernel.cap.check_fs(ctx.pid, repo, "r"):
            raise ToolFsDenied(
                f"agent {ctx.pid} not granted 'r' on repo {repo!r}",
            )
        if not ctx.kernel.cap.check_fs(ctx.pid, git_bin, "r"):
            raise ToolFsDenied(
                f"agent {ctx.pid} not granted 'r' on {git_bin!r}",
            )

    # Build argv: ``git -C <repo> <op-base> [extra] [ref] [-- path]``
    argv: list = [git_bin, "-C", repo] + list(_OP_BASE_ARGV[op])
    # Special handling: blame requires path; show requires ref.
    if op == "blame":
        if path_arg is None:
            raise ToolInvalidArgs("'path' required for op='blame'")
    if op == "show":
        if ref is None:
            raise ToolInvalidArgs("'ref' required for op='show'")
    if extra:
        argv.extend(extra)
    if ref is not None and op != "blame":
        argv.append(ref)
    if path_arg is not None:
        if op == "blame":
            argv.append(path_arg)
        elif op in ("log", "diff", "show", "ls_files"):
            argv.extend(["--", path_arg])

    # Sandbox policy + scrubbed env (reuse Exec primitives).
    policy = SandboxPolicy(
        cpu_seconds  = max(timeout_s, 1),
        memory_bytes = DEFAULT_MEMORY_BYTES,
        fsize_bytes  = DEFAULT_FSIZE_BYTES,
        nofile       = DEFAULT_NOFILE,
        wall_seconds = float(timeout_s),
        new_session  = True,
    )
    env = _scrub_env(_os.environ, {})
    # Disable any user gitconfig-borne hooks / aliases.
    env["GIT_CONFIG_NOSYSTEM"] = "1"
    env["HOME"] = "/tmp"
    env["GIT_TERMINAL_PROMPT"] = "0"

    result = run_sandboxed(
        argv, policy,
        env=env, cwd=repo,
        capture_stdout=True, capture_stderr=True,
    )

    out_bytes = result.stdout or b""
    err_bytes = result.stderr or b""
    out_truncated = len(out_bytes) > DEFAULT_GIT_STDOUT_CAP
    err_truncated = len(err_bytes) > DEFAULT_GIT_STDERR_CAP
    if out_truncated:
        out_bytes = out_bytes[:DEFAULT_GIT_STDOUT_CAP]
    if err_truncated:
        err_bytes = err_bytes[:DEFAULT_GIT_STDERR_CAP]

    return {
        "op":               op,
        "exit_code":        int(result.exit_code),
        "stdout":           out_bytes.decode("utf-8", errors="replace"),
        "stderr":           err_bytes.decode("utf-8", errors="replace"),
        "stdout_truncated": out_truncated,
        "stderr_truncated": err_truncated,
        "duration_s":       round(float(result.duration_s), 4),
        "timed_out":        bool(result.timed_out),
        "cmd":              list(argv),
    }


GIT_TOOL = Tool(
    name="Git",
    description=(
        "Read-only git inspector. ops: status / log / diff / show / "
        "branch / blame / ls_files / rev_parse. Argv allowlisted per "
        "op; ref/path validated; gitconfig disabled. Requires 'Git' "
        "tool capability AND 'r' fs_grants on repo + git binary."
    ),
    handler=git_handler,
    requires_capability=True,
    requires_fs=(),    # handler does its own fs check.
)


def register_git_tool(
    registry: ToolRegistry, *, kernel=None,
) -> str:
    """Register the Git tool. **Opt-in** — not called by
    ``register_builtin_tools``."""
    del kernel
    registry.register(GIT_TOOL)
    return GIT_TOOL.name


__all__ = ["GIT_TOOL", "git_handler", "register_git_tool"]
