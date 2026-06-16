"""Tests for kernel.tools (RFC 0021)."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

from cheetahclaws.kernel import (
    Kernel,
    SandboxPolicy,
    Tool,
    ToolContext,
    ToolDenied,
    ToolError,
    ToolFailed,
    ToolFsDenied,
    ToolInvalidArgs,
    ToolNotFound,
    ToolRegistry,
    dispatch_tool_call,
    register_builtin_tools,
)


pytestmark_subprocess = pytest.mark.skipif(
    os.name != "posix",
    reason="end-to-end tests spawn POSIX subprocesses",
)


# ── Registry CRUD ───────────────────────────────────────────────────────


def test_registry_register_and_get():
    r = ToolRegistry()
    t = Tool(name="X", description="x", handler=lambda a, c: {"ok": 1})
    r.register(t)
    assert r.get("X") is t
    assert r.has("X")
    assert "X" in r.list()


def test_registry_register_replaces():
    r = ToolRegistry()
    t1 = Tool(name="X", description="v1", handler=lambda a, c: {"v": 1})
    t2 = Tool(name="X", description="v2", handler=lambda a, c: {"v": 2})
    r.register(t1)
    r.register(t2)
    assert r.get("X") is t2


def test_registry_get_missing_raises():
    r = ToolRegistry()
    with pytest.raises(ToolNotFound):
        r.get("nope")


def test_registry_unregister():
    r = ToolRegistry()
    r.register(Tool(name="X", description="x",
                    handler=lambda a, c: {}))
    assert r.unregister("X") is True
    assert r.unregister("X") is False  # idempotent
    assert not r.has("X")


def test_registry_register_validates_name():
    r = ToolRegistry()
    with pytest.raises(ToolError):
        r.register(Tool(name="", description="x",
                          handler=lambda a, c: {}))


def test_registry_register_validates_handler():
    r = ToolRegistry()
    with pytest.raises(ToolError):
        r.register(Tool(name="X", description="x",
                          handler="not callable"))  # type: ignore[arg-type]


def test_registry_register_validates_requires_fs():
    r = ToolRegistry()
    with pytest.raises(ToolError):
        r.register(Tool(name="X", description="x",
                          handler=lambda a, c: {},
                          requires_fs=(("write", "path"),)))  # bad mode


# ── register_builtin_tools ─────────────────────────────────────────────


def test_register_builtin_registers_standard_set():
    """Builtin set as of RFC 0031:
    Echo + Read + Write + Glob + List + Diff + AST."""
    r = ToolRegistry()
    names = register_builtin_tools(r)
    assert sorted(names) == ["AST", "Diff", "Echo", "Glob",
                              "List", "Read", "Write"]
    for n in names:
        assert r.has(n)


def test_register_builtin_idempotent():
    r = ToolRegistry()
    register_builtin_tools(r)
    register_builtin_tools(r)  # no error
    assert sorted(r.list()) == ["AST", "Diff", "Echo", "Glob",
                                 "List", "Read", "Write"]


# ── dispatch_tool_call (pure) ─────────────────────────────────────────


def _basic_registry():
    r = ToolRegistry()
    register_builtin_tools(r)
    return r


class _FakeKernel:
    """Minimal kernel-like stub for dispatch tests that don't need
    a real kernel.db."""
    def __init__(self, *, allowed_tools=(), allowed_fs=()):
        self._allowed_tools = set(allowed_tools)
        self._allowed_fs = set(allowed_fs)  # tuples of (path_prefix, mode)
        self.cap = self

    def check_tool(self, pid, tool):
        return tool in self._allowed_tools

    def check_fs(self, pid, path, mode):
        for prefix, m in self._allowed_fs:
            if path.startswith(prefix) and (mode == "r" or m == "rw"):
                return True
        return False


def test_dispatch_echo_success():
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Echo"])
    resp = dispatch_tool_call(
        msg={"op": "tool_call", "tool_call_id": "T",
              "tool": "Echo", "args": {"text": "hi"}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is True
    assert resp["result"]["text"] == "hi"
    assert resp["tool_call_id"] == "T"


def test_dispatch_unknown_tool():
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Echo"])
    resp = dispatch_tool_call(
        msg={"tool": "Nope", "tool_call_id": "T", "args": {}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "tool_not_found"


def test_dispatch_missing_tool_field():
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Echo"])
    resp = dispatch_tool_call(
        msg={"tool_call_id": "T", "args": {}},  # no 'tool'
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "tool_not_found"


def test_dispatch_capability_denied():
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=())  # nothing allowed
    resp = dispatch_tool_call(
        msg={"tool": "Echo", "tool_call_id": "T",
              "args": {"text": "hi"}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "permission_denied"


def test_dispatch_fs_denied(tmp_path):
    r = _basic_registry()
    target = tmp_path / "a.txt"
    target.write_text("hello")
    # Tool is allowed but the path is not in fs_grants.
    k = _FakeKernel(allowed_tools=["Read"], allowed_fs=())
    resp = dispatch_tool_call(
        msg={"tool": "Read", "tool_call_id": "T",
              "args": {"path": str(target)}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "fs_denied"


def test_dispatch_fs_required_arg_missing():
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Read"],
                     allowed_fs=[("/", "rw")])
    resp = dispatch_tool_call(
        msg={"tool": "Read", "tool_call_id": "T", "args": {}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "invalid_args"


def test_dispatch_handler_raises_tool_failed():
    r = ToolRegistry()
    def bad(args, ctx):
        raise RuntimeError("boom")
    r.register(Tool(name="X", description="x", handler=bad,
                    requires_capability=False))
    resp = dispatch_tool_call(
        msg={"tool": "X", "tool_call_id": "T", "args": {}},
        pid=1, registry=r, kernel=None,
    )
    assert resp["ok"] is False
    assert resp["error"] == "tool_failed"
    assert "boom" in resp["message"]


def test_dispatch_handler_raises_tool_invalid_args():
    r = ToolRegistry()
    def bad(args, ctx):
        raise ToolInvalidArgs("missing X")
    r.register(Tool(name="X", description="x", handler=bad,
                    requires_capability=False))
    resp = dispatch_tool_call(
        msg={"tool": "X", "tool_call_id": "T", "args": {}},
        pid=1, registry=r, kernel=None,
    )
    assert resp["ok"] is False
    assert resp["error"] == "invalid_args"


def test_dispatch_handler_returns_non_dict():
    r = ToolRegistry()
    r.register(Tool(name="X", description="x",
                    handler=lambda a, c: "not a dict",
                    requires_capability=False))
    resp = dispatch_tool_call(
        msg={"tool": "X", "tool_call_id": "T", "args": {}},
        pid=1, registry=r, kernel=None,
    )
    assert resp["ok"] is False
    assert resp["error"] == "tool_failed"


def test_dispatch_no_kernel_no_cap_check():
    """When kernel is None, capability checks pass through (useful
    for test setups). fs checks similarly skip."""
    r = _basic_registry()
    resp = dispatch_tool_call(
        msg={"tool": "Echo", "tool_call_id": "T",
              "args": {"text": "hi"}},
        pid=1, registry=r, kernel=None,
    )
    assert resp["ok"] is True


# ── Built-in tool: Read ────────────────────────────────────────────────


def test_builtin_read_success(tmp_path):
    target = tmp_path / "test.txt"
    target.write_text("hello world", encoding="utf-8")
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Read"],
                     allowed_fs=[(str(tmp_path), "rw")])
    resp = dispatch_tool_call(
        msg={"tool": "Read", "tool_call_id": "T",
              "args": {"path": str(target)}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"]
    assert resp["result"]["content"] == "hello world"
    assert resp["result"]["size"] == 11
    assert resp["result"]["encoding"] == "utf-8"


def test_builtin_read_missing_file_returns_failed(tmp_path):
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Read"],
                     allowed_fs=[(str(tmp_path), "rw")])
    resp = dispatch_tool_call(
        msg={"tool": "Read", "tool_call_id": "T",
              "args": {"path": str(tmp_path / "nope.txt")}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "tool_failed"


def test_builtin_read_directory_fails(tmp_path):
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Read"],
                     allowed_fs=[(str(tmp_path), "rw")])
    resp = dispatch_tool_call(
        msg={"tool": "Read", "tool_call_id": "T",
              "args": {"path": str(tmp_path)}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "tool_failed"


def test_builtin_read_binary_falls_back_to_base64(tmp_path):
    target = tmp_path / "bin.dat"
    target.write_bytes(bytes(range(256)))
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Read"],
                     allowed_fs=[(str(tmp_path), "rw")])
    resp = dispatch_tool_call(
        msg={"tool": "Read", "tool_call_id": "T",
              "args": {"path": str(target)}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"]
    assert resp["result"]["encoding"] == "base64"


# ── Built-in tool: Write ───────────────────────────────────────────────


def test_builtin_write_success(tmp_path):
    target = tmp_path / "out.txt"
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Write"],
                     allowed_fs=[(str(tmp_path), "rw")])
    resp = dispatch_tool_call(
        msg={"tool": "Write", "tool_call_id": "T",
              "args": {"path": str(target),
                       "content": "hello"}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"]
    assert target.read_text() == "hello"


def test_builtin_write_requires_rw_capability(tmp_path):
    """Write needs 'rw' fs grant; 'r'-only is not enough."""
    target = tmp_path / "out.txt"
    r = _basic_registry()
    # FakeKernel with read-only access.
    k = _FakeKernel(allowed_tools=["Write"],
                     allowed_fs=[(str(tmp_path), "r")])
    resp = dispatch_tool_call(
        msg={"tool": "Write", "tool_call_id": "T",
              "args": {"path": str(target), "content": "x"}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "fs_denied"


def test_builtin_write_invalid_content_type(tmp_path):
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Write"],
                     allowed_fs=[(str(tmp_path), "rw")])
    resp = dispatch_tool_call(
        msg={"tool": "Write", "tool_call_id": "T",
              "args": {"path": str(tmp_path / "x.txt"),
                       "content": 12345}},  # wrong type
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "invalid_args"


def test_builtin_write_missing_parent_dir(tmp_path):
    r = _basic_registry()
    k = _FakeKernel(allowed_tools=["Write"],
                     allowed_fs=[(str(tmp_path), "rw")])
    resp = dispatch_tool_call(
        msg={"tool": "Write", "tool_call_id": "T",
              "args": {"path": str(tmp_path / "deep" / "nested" / "file"),
                       "content": "x"}},
        pid=1, registry=r, kernel=k,
    )
    assert resp["ok"] is False
    assert resp["error"] == "tool_failed"


# ── Supervisor end-to-end ─────────────────────────────────────────────


@pytestmark_subprocess
def test_supervisor_dispatches_echo_via_runner(tmp_path):
    """Spin up the kernel + supervisor with a registry, run runner_main
    in tool_call mode, verify the tool_response is correct."""
    with Kernel.open(tmp_path / "kernel.db") as k:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        a = k.create_agent(name="x", template="t")
        k.cap.create(pid=a.pid, tool_grants=["Echo"])
        sup = k.make_supervisor(tool_registry=registry)
        call_body = json.dumps({
            "tool": "Echo", "tool_call_id": "abc",
            "args": {"text": "hi from agent"},
        })
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"],
            policy=SandboxPolicy(wall_seconds=10),
            env={**os.environ, "CC_RUNNER_BEHAVIOR": f"tool_call={call_body}"},
        )
        info = sup.wait(a.pid, timeout=15)
        assert info.exit_kind == "completed", info.stderr_tail
        tr = info.metadata["tool_response"]
        assert tr["ok"] is True
        assert tr["result"]["text"] == "hi from agent"
        assert tr["tool_call_id"] == "abc"


@pytestmark_subprocess
def test_supervisor_dispatches_read_with_fs_grant(tmp_path):
    target = tmp_path / "data.txt"
    target.write_text("on disk")
    with Kernel.open(tmp_path / "kernel.db") as k:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        a = k.create_agent(name="x", template="t")
        k.cap.create(
            pid=a.pid,
            tool_grants=["Read"],
            fs_grants=[{"prefix": str(tmp_path), "mode": "r"}],
        )
        sup = k.make_supervisor(tool_registry=registry)
        body = json.dumps({"tool": "Read", "tool_call_id": "T",
                            "args": {"path": str(target)}})
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"],
            policy=SandboxPolicy(wall_seconds=10),
            env={**os.environ, "CC_RUNNER_BEHAVIOR": f"tool_call={body}"},
        )
        info = sup.wait(a.pid, timeout=15)
        tr = info.metadata["tool_response"]
        assert tr["ok"] is True
        assert tr["result"]["content"] == "on disk"


@pytestmark_subprocess
def test_supervisor_capability_denied_via_runner(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        a = k.create_agent(name="x", template="t")
        # No Echo in tool_grants.
        k.cap.create(pid=a.pid, tool_grants=["OtherTool"])
        sup = k.make_supervisor(tool_registry=registry)
        body = json.dumps({"tool": "Echo", "tool_call_id": "T",
                            "args": {"text": "x"}})
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"],
            policy=SandboxPolicy(wall_seconds=10),
            env={**os.environ, "CC_RUNNER_BEHAVIOR": f"tool_call={body}"},
        )
        info = sup.wait(a.pid, timeout=15)
        tr = info.metadata["tool_response"]
        assert tr["ok"] is False
        assert tr["error"] == "permission_denied"


@pytestmark_subprocess
def test_supervisor_without_registry_returns_tool_not_found(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        a = k.create_agent(name="x", template="t")
        k.cap.create(pid=a.pid, tool_grants=["Echo"])
        sup = k.make_supervisor()  # no tool_registry
        body = json.dumps({"tool": "Echo", "tool_call_id": "T",
                            "args": {"text": "x"}})
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"],
            policy=SandboxPolicy(wall_seconds=10),
            env={**os.environ, "CC_RUNNER_BEHAVIOR": f"tool_call={body}"},
        )
        info = sup.wait(a.pid, timeout=15)
        tr = info.metadata["tool_response"]
        assert tr["ok"] is False
        assert tr["error"] == "tool_not_found"


# ── Audit events ───────────────────────────────────────────────────────


@pytestmark_subprocess
def test_supervisor_records_tool_dispatched_event(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        a = k.create_agent(name="x", template="t")
        k.cap.create(pid=a.pid, tool_grants=["Echo"])
        sup = k.make_supervisor(tool_registry=registry)
        body = json.dumps({"tool": "Echo", "tool_call_id": "T",
                            "args": {"text": "hi"}})
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"],
            policy=SandboxPolicy(wall_seconds=10),
            env={**os.environ, "CC_RUNNER_BEHAVIOR": f"tool_call={body}"},
        )
        sup.wait(a.pid, timeout=15)
        events = k.process.events_tail(pid=a.pid,
                                         kind="tool.call.dispatched")
        assert len(events) == 1
        ev = events[0]
        assert ev.payload["tool"] == "Echo"
        assert ev.payload["tool_call_id"] == "T"
        assert ev.payload["ok"] is True


@pytestmark_subprocess
def test_supervisor_records_tool_denied_event(tmp_path):
    with Kernel.open(tmp_path / "kernel.db") as k:
        registry = ToolRegistry()
        register_builtin_tools(registry)
        a = k.create_agent(name="x", template="t")
        k.cap.create(pid=a.pid, tool_grants=[])  # nothing
        sup = k.make_supervisor(tool_registry=registry)
        body = json.dumps({"tool": "Echo", "tool_call_id": "T",
                            "args": {"text": "hi"}})
        sup.spawn(
            pid=a.pid,
            argv=[sys.executable, "-m", "cheetahclaws.kernel.runner.runner_main"],
            policy=SandboxPolicy(wall_seconds=10),
            env={**os.environ, "CC_RUNNER_BEHAVIOR": f"tool_call={body}"},
        )
        sup.wait(a.pid, timeout=15)
        events = k.process.events_tail(pid=a.pid,
                                         kind="tool.call.denied")
        assert len(events) == 1
        assert events[0].payload["error"] == "permission_denied"
