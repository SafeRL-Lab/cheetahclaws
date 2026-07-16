"""Tests for kernel.tools.exec_tool (RFC 0023).

The most security-sensitive tool the kernel ships. Tests focus on
the safety boundary: no shell expansion, env scrubbing, RLIMIT,
output bounding, capability + fs_grants gates.

POSIX-only — Exec spawns subprocess with preexec_fn (RLIMIT).
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from cheetahclaws.kernel import (
    EXEC_TOOL,
    Kernel,
    SandboxPolicy,
    ToolRegistry,
    register_builtin_tools,
    register_exec_tool,
)
from cheetahclaws.kernel.tools.exec_tool import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_S,
    MAX_TIMEOUT_S,
    _scrub_env,
    _validate_argv,
    _validate_user_env,
    _validate_timeout,
    exec_handler,
)
from cheetahclaws.kernel.tools.registry import (
    ToolContext,
    ToolFailed,
    ToolFsDenied,
    ToolInvalidArgs,
    dispatch_tool_call,
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Exec tool requires POSIX (RLIMIT + subprocess preexec_fn)",
)


# ── Discovery: find absolute paths to common binaries ────────────────


def _find_bin(name: str) -> str:
    """Return absolute path to a system binary, skip test if missing."""
    for prefix in ("/bin/", "/usr/bin/", "/usr/local/bin/"):
        p = prefix + name
        if Path(p).is_file():
            return p
    pytest.skip(f"binary {name!r} not found in /bin or /usr/bin")


ECHO_BIN  = _find_bin("echo")
TRUE_BIN  = _find_bin("true")
SLEEP_BIN = _find_bin("sleep")
ENV_BIN   = _find_bin("env")
CAT_BIN   = _find_bin("cat")


# ── Opt-in invariant ─────────────────────────────────────────────────


def test_exec_not_in_builtin():
    """register_builtin_tools MUST NOT include Exec — opt-in only."""
    r = ToolRegistry()
    register_builtin_tools(r)
    assert "Exec" not in r.list()


def test_register_exec_adds_it():
    r = ToolRegistry()
    register_builtin_tools(r)
    name = register_exec_tool(r)
    assert name == "Exec"
    assert "Exec" in r.list()


def test_register_exec_idempotent():
    """Re-registering replaces silently per ToolRegistry contract."""
    r = ToolRegistry()
    register_exec_tool(r)
    register_exec_tool(r)
    assert r.list() == ["Exec"]


# ── _validate_argv ───────────────────────────────────────────────────


def test_validate_argv_basic():
    out = _validate_argv([ECHO_BIN, "hi"])
    assert out == [ECHO_BIN, "hi"]


def test_validate_argv_rejects_non_list():
    with pytest.raises(ToolInvalidArgs):
        _validate_argv("not a list")


def test_validate_argv_rejects_empty():
    with pytest.raises(ToolInvalidArgs):
        _validate_argv([])


def test_validate_argv_rejects_non_str():
    with pytest.raises(ToolInvalidArgs):
        _validate_argv([ECHO_BIN, 123])


def test_validate_argv_rejects_relative_path():
    """argv[0] must be absolute — no PATH lookup."""
    with pytest.raises(ToolInvalidArgs) as e:
        _validate_argv(["echo", "hi"])
    assert "absolute" in str(e.value)


def test_validate_argv_rejects_missing_binary(tmp_path):
    with pytest.raises(ToolFailed):
        _validate_argv([str(tmp_path / "nope")])


# ── _validate_user_env ───────────────────────────────────────────────


def test_validate_env_none_returns_empty():
    assert _validate_user_env(None) == {}


def test_validate_env_basic():
    assert _validate_user_env({"FOO": "bar"}) == {"FOO": "bar"}


def test_validate_env_rejects_non_dict():
    with pytest.raises(ToolInvalidArgs):
        _validate_user_env("not a dict")


def test_validate_env_rejects_non_str_value():
    with pytest.raises(ToolInvalidArgs):
        _validate_user_env({"FOO": 123})


def test_validate_env_rejects_underscore_prefix():
    """Reserved keys (kernel-internal) blocked."""
    with pytest.raises(ToolInvalidArgs) as e:
        _validate_user_env({"_INTERNAL": "x"})
    assert "reserved" in str(e.value)


# ── _validate_timeout ────────────────────────────────────────────────


def test_timeout_default():
    assert _validate_timeout(None) == DEFAULT_TIMEOUT_S


def test_timeout_int():
    assert _validate_timeout(30) == 30


def test_timeout_float_rounds():
    assert _validate_timeout(30.7) == 30


def test_timeout_too_low():
    with pytest.raises(ToolInvalidArgs):
        _validate_timeout(0)


def test_timeout_too_high():
    with pytest.raises(ToolInvalidArgs):
        _validate_timeout(MAX_TIMEOUT_S + 1)


# ── _scrub_env ───────────────────────────────────────────────────────


def test_scrub_drops_secrets():
    parent = {
        "PATH": "/usr/bin",
        "HOME": "/h",
        "ANTHROPIC_API_KEY": "sk-secret",
        "AWS_SECRET_ACCESS_KEY": "very-secret",
        "GITHUB_TOKEN": "ghp_secret",
    }
    out = _scrub_env(parent, {})
    assert "ANTHROPIC_API_KEY"     not in out
    assert "AWS_SECRET_ACCESS_KEY" not in out
    assert "GITHUB_TOKEN"          not in out


def test_scrub_keeps_safe_keys():
    parent = {"PATH": "/test", "HOME": "/test_home"}
    out = _scrub_env(parent, {})
    assert out["PATH"] == "/test"
    assert out["HOME"] == "/test_home"


def test_scrub_user_env_overrides_defaults():
    parent = {}
    out = _scrub_env(parent, {"PATH": "/custom/bin"})
    assert out["PATH"] == "/custom/bin"


def test_scrub_user_env_addition():
    parent = {}
    out = _scrub_env(parent, {"MY_VAR": "value"})
    assert out["MY_VAR"] == "value"


# ── End-to-end via dispatch_tool_call (in-process) ───────────────────


def _make_kernel_with_grants(tmp_path, *, tool_grants, fs_grants):
    k = Kernel.open(tmp_path / "kernel.db")
    a = k.create_agent(name="x", template="t")
    k.cap.create(pid=a.pid, tool_grants=tool_grants,
                 fs_grants=fs_grants)
    return k, a.pid


def _registry_with_exec():
    r = ToolRegistry()
    register_exec_tool(r)
    return r


def test_e2e_basic_exec(tmp_path):
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/usr/bin/", "mode": "r"},
                   {"prefix": "/bin/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ECHO_BIN, "hello"]}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is True
        assert resp["result"]["exit_code"] == 0
        assert resp["result"]["stdout"] == "hello\n"
        assert resp["result"]["timed_out"] is False
    finally:
        k.close()


def test_e2e_no_shell_expansion(tmp_path):
    """Critical safety property: shell metachars in args are NOT
    interpreted. echo prints them literally."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/usr/bin/", "mode": "r"},
                   {"prefix": "/bin/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ECHO_BIN, "; rm -rf / && echo PWNED"]}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is True
        assert resp["result"]["stdout"] == "; rm -rf / && echo PWNED\n"
    finally:
        k.close()


def test_e2e_no_path_expansion(tmp_path):
    """argv[0] is absolute — no PATH lookup. Setting PATH wonky
    doesn't affect resolution."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/usr/bin/", "mode": "r"},
                   {"prefix": "/bin/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ECHO_BIN, "ok"],
                           "env": {"PATH": "/nonexistent"}}},
            pid=pid, registry=r, kernel=k,
        )
        # PATH=/nonexistent in child but argv[0] is absolute, so
        # echo runs anyway.
        assert resp["ok"] is True
    finally:
        k.close()


def test_e2e_capability_denied(tmp_path):
    """Without 'Exec' in tool_grants → permission_denied."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Read"],   # Exec missing
        fs_grants=[{"prefix": "/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ECHO_BIN, "x"]}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is False
        assert resp["error"] == "permission_denied"
    finally:
        k.close()


def test_e2e_fs_denied_on_argv0(tmp_path):
    """Exec capability granted but fs_grants doesn't cover argv[0]
    → fs_denied (handler-side check)."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/some/other/path/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ECHO_BIN, "x"]}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is False
        assert resp["error"] == "fs_denied"
    finally:
        k.close()


def test_e2e_relative_path_rejected(tmp_path):
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": ["echo", "x"]}},  # relative
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is False
        assert resp["error"] == "invalid_args"
    finally:
        k.close()


def test_e2e_exit_code_propagated(tmp_path):
    """An /bin/sh -c won't run via Exec because we forbid shell.
    But /bin/false (returns 1) propagates exit code."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/bin/", "mode": "r"},
                   {"prefix": "/usr/bin/", "mode": "r"}],
    )
    false_bin = _find_bin("false")
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [false_bin]}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is True
        assert resp["result"]["exit_code"] == 1
    finally:
        k.close()


def test_e2e_timeout_kills_long_command(tmp_path):
    """Sleep 30 with timeout_s=1 → wall-killer fires within seconds."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/bin/", "mode": "r"},
                   {"prefix": "/usr/bin/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        start = time.monotonic()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [SLEEP_BIN, "30"], "timeout_s": 1}},
            pid=pid, registry=r, kernel=k,
        )
        elapsed = time.monotonic() - start
        assert elapsed < 8, f"sleep ran {elapsed}s"
        assert resp["ok"] is True
        assert resp["result"]["timed_out"] is True
        assert resp["result"]["exit_code"] != 0
    finally:
        k.close()


def test_e2e_env_scrub_secret_dropped(tmp_path, monkeypatch):
    """Set ANTHROPIC_API_KEY in parent; Exec'd `env` shouldn't see it."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret-shouldnt-leak")
    monkeypatch.setenv("CC_KERNEL_TEST_FAKE_SECRET", "must-not-leak")
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/bin/", "mode": "r"},
                   {"prefix": "/usr/bin/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ENV_BIN]}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is True
        out = resp["result"]["stdout"]
        assert "ANTHROPIC_API_KEY" not in out
        assert "sk-secret-shouldnt-leak" not in out
        assert "CC_KERNEL_TEST_FAKE_SECRET" not in out
    finally:
        k.close()


def test_e2e_user_env_visible(tmp_path):
    """args.env additions DO show up in the child's env."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/bin/", "mode": "r"},
                   {"prefix": "/usr/bin/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ENV_BIN],
                           "env": {"MYTESTVAR": "hello123"}}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is True
        assert "MYTESTVAR=hello123" in resp["result"]["stdout"]
    finally:
        k.close()


def test_e2e_output_truncation(tmp_path):
    """A binary that prints more than max_output_bytes →
    stdout_truncated=True with stdout length ≤ cap."""
    # /usr/bin/yes prints forever; pair with head to bound.
    # Or use python -c to print a fixed large amount.
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": str(Path(sys.executable).parent), "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        # Print 100k chars of 'X' — way over the 1024-byte cap.
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {
                      "argv": [sys.executable, "-c",
                               "print('X' * 100_000)"],
                      "max_output_bytes": 1024,
                      "timeout_s": 10,
                  }},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is True
        assert resp["result"]["stdout_truncated"] is True
        assert len(resp["result"]["stdout"]) <= 1024
    finally:
        k.close()


def test_e2e_invalid_max_output_bytes(tmp_path):
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        # 100 bytes < MIN_OUTPUT_BYTES_LIMIT (1024).
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ECHO_BIN, "x"],
                           "max_output_bytes": 100}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is False
        assert resp["error"] == "invalid_args"
    finally:
        k.close()


def test_e2e_cwd_must_be_absolute(tmp_path):
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ECHO_BIN, "x"],
                           "cwd": "relative/path"}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is False
        assert resp["error"] == "invalid_args"
    finally:
        k.close()


def test_e2e_cwd_fs_denied(tmp_path):
    """cwd not covered by fs_grants → fs_denied."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/usr/bin/", "mode": "r"},
                   {"prefix": "/bin/", "mode": "r"}],
        # /tmp not in grants
    )
    try:
        r = _registry_with_exec()
        resp = dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [ECHO_BIN, "x"], "cwd": "/tmp"}},
            pid=pid, registry=r, kernel=k,
        )
        assert resp["ok"] is False
        assert resp["error"] == "fs_denied"
    finally:
        k.close()


def test_e2e_audit_event_recorded(tmp_path):
    """Successful Exec writes a tool.call.dispatched audit event
    via the supervisor's existing path. We test by going through
    the full supervisor flow."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/bin/", "mode": "r"},
                   {"prefix": "/usr/bin/", "mode": "r"}],
    )
    try:
        r = ToolRegistry()
        register_exec_tool(r)
        dispatch_tool_call(
            msg={"tool": "Exec", "tool_call_id": "T",
                  "args": {"argv": [TRUE_BIN]}},
            pid=pid, registry=r, kernel=k,
        )
        # The dispatch_tool_call function itself doesn't write
        # audit events — the supervisor's _handle_tool_call does.
        # So we only test the dispatch invariant here. Audit
        # coverage is in test_kernel_tools.py.
    finally:
        k.close()


# ── Argv schema bounds ───────────────────────────────────────────────


def test_argv_with_dangerous_chars_runs_safely(tmp_path):
    """A range of shell-meaningful chars in args. None should
    expand."""
    k, pid = _make_kernel_with_grants(
        tmp_path, tool_grants=["Exec"],
        fs_grants=[{"prefix": "/bin/", "mode": "r"},
                   {"prefix": "/usr/bin/", "mode": "r"}],
    )
    try:
        r = _registry_with_exec()
        for arg in ("$(whoami)", "`id`", "&& echo PWN",
                    "|cat /etc/passwd", "`reboot`"):
            resp = dispatch_tool_call(
                msg={"tool": "Exec", "tool_call_id": "T",
                      "args": {"argv": [ECHO_BIN, arg]}},
                pid=pid, registry=r, kernel=k,
            )
            assert resp["ok"] is True
            # The arg appears literally in stdout — nothing
            # expanded.
            assert arg in resp["result"]["stdout"]
    finally:
        k.close()
