"""Tests for the Git built-in (RFC 0032)."""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from cheetahclaws.kernel.tools.git_tool import (
    GIT_TOOL,
    git_handler,
    register_git_tool,
)
from cheetahclaws.kernel.tools.registry import (
    ToolContext,
    ToolFsDenied,
    ToolInvalidArgs,
    ToolRegistry,
)


_GIT_BIN = shutil.which("git")
needs_git = pytest.mark.skipif(
    _GIT_BIN is None, reason="git binary required",
)


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="Git tool exec uses POSIX subprocess primitives",
)


# ── Stubs ─────────────────────────────────────────────────────────


class _AllowAll:
    class _Cap:
        def check_fs(self, pid, path, mode): return True
        def check_tool(self, pid, t):        return True
    cap = _Cap()


class _DenyFs:
    class _Cap:
        def check_fs(self, pid, path, mode): return False
        def check_tool(self, pid, t):        return True
    cap = _Cap()


@pytest.fixture
def repo(tmp_path):
    """Init a git repo in a tmp dir with a single commit."""
    if _GIT_BIN is None:
        pytest.skip("git binary required")
    r = tmp_path / "repo"
    r.mkdir()
    subprocess.run([_GIT_BIN, "init", "-q"], cwd=r, check=True)
    subprocess.run([_GIT_BIN, "config", "user.email",
                     "test@example.com"], cwd=r, check=True)
    subprocess.run([_GIT_BIN, "config", "user.name", "Test"],
                    cwd=r, check=True)
    (r / "hello.txt").write_text("first\n")
    subprocess.run([_GIT_BIN, "add", "."], cwd=r, check=True)
    subprocess.run([_GIT_BIN, "commit", "-q", "-m", "initial"],
                    cwd=r, check=True)
    return r


# ── Args validation ───────────────────────────────────────────────


def test_op_required():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        git_handler({"repo": "/tmp"}, ctx)


def test_op_must_be_allowlisted():
    ctx = ToolContext(pid=1, kernel=None)
    for bad in ("push", "commit", "fetch", "clone",
                "remote", "pull", "merge"):
        with pytest.raises(ToolInvalidArgs):
            git_handler({"op": bad, "repo": "/tmp"}, ctx)


def test_repo_must_be_absolute():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        git_handler({"op": "status", "repo": "rel/path"}, ctx)


def test_repo_must_have_dotgit(tmp_path):
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(Exception):
        git_handler({"op": "status", "repo": str(tmp_path)}, ctx)


def test_ref_disallowed_chars():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        git_handler({
            "op": "show", "repo": "/tmp",
            "ref": "HEAD; rm -rf /",
        }, ctx)


def test_path_must_be_relative():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        git_handler({"op": "log", "repo": "/tmp",
                       "path": "/etc/passwd"}, ctx)


def test_path_no_dotdot():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        git_handler({"op": "log", "repo": "/tmp",
                       "path": "../../../etc/passwd"}, ctx)


def test_disallowed_flag_for_op():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        git_handler({
            "op": "log", "repo": "/tmp",
            "args": ["--exec-path=/tmp/evil"],
        }, ctx)


def test_arg_with_newline_rejected():
    ctx = ToolContext(pid=1, kernel=None)
    with pytest.raises(ToolInvalidArgs):
        git_handler({
            "op": "log", "repo": "/tmp",
            "args": ["--oneline\nrm"],
        }, ctx)


# ── Real-repo execution (needs git) ──────────────────────────────


@needs_git
def test_status_returns_zero(repo):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = git_handler({"op": "status", "repo": str(repo)}, ctx)
    assert r["exit_code"] == 0
    assert r["timed_out"] is False
    # Clean repo → empty porcelain output.
    assert r["stdout"] == ""


@needs_git
def test_log_oneline(repo):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = git_handler({
        "op": "log", "repo": str(repo),
        "args": ["--oneline", "-n", "1"],
    }, ctx)
    assert r["exit_code"] == 0
    # One line of "<sha> initial".
    assert "initial" in r["stdout"]
    assert r["stdout"].count("\n") == 1


@needs_git
def test_status_after_change(repo):
    (repo / "hello.txt").write_text("modified\n")
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = git_handler({"op": "status", "repo": str(repo)}, ctx)
    assert r["exit_code"] == 0
    # Modified file shows in porcelain.
    assert "hello.txt" in r["stdout"]


@needs_git
def test_diff_path_filter(repo):
    (repo / "hello.txt").write_text("changed\n")
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = git_handler({
        "op": "diff", "repo": str(repo),
        "path": "hello.txt",
    }, ctx)
    assert r["exit_code"] == 0
    assert "hello.txt" in r["stdout"]


@needs_git
def test_show_requires_ref(repo):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    with pytest.raises(ToolInvalidArgs):
        git_handler({"op": "show", "repo": str(repo)}, ctx)


@needs_git
def test_show_with_ref(repo):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = git_handler({
        "op": "show", "repo": str(repo), "ref": "HEAD",
        "args": ["-s", "--no-color"],
    }, ctx)
    assert r["exit_code"] == 0
    assert "initial" in r["stdout"]


@needs_git
def test_blame_requires_path(repo):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    with pytest.raises(ToolInvalidArgs):
        git_handler({"op": "blame", "repo": str(repo)}, ctx)


@needs_git
def test_blame_with_path(repo):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = git_handler({
        "op": "blame", "repo": str(repo),
        "path": "hello.txt",
    }, ctx)
    assert r["exit_code"] == 0
    assert "first" in r["stdout"]


@needs_git
def test_rev_parse_head(repo):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = git_handler({
        "op": "rev_parse", "repo": str(repo),
        "args": ["--short", "HEAD"],
    }, ctx)
    assert r["exit_code"] == 0
    assert r["stdout"].strip()


@needs_git
def test_branch_list(repo):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = git_handler({"op": "branch", "repo": str(repo)}, ctx)
    assert r["exit_code"] == 0
    assert r["stdout"].strip() != ""


@needs_git
def test_ls_files(repo):
    ctx = ToolContext(pid=1, kernel=_AllowAll())
    r = git_handler({"op": "ls_files", "repo": str(repo)}, ctx)
    assert r["exit_code"] == 0
    assert "hello.txt" in r["stdout"]


@needs_git
def test_fs_denied_on_repo(repo):
    ctx = ToolContext(pid=1, kernel=_DenyFs())
    with pytest.raises(ToolFsDenied):
        git_handler({"op": "status", "repo": str(repo)}, ctx)


# ── Registry ──────────────────────────────────────────────────────


def test_git_not_in_register_builtin_tools():
    """Git is opt-in — must not be auto-registered."""
    from cheetahclaws.kernel.tools.builtin import register_builtin_tools
    reg = ToolRegistry()
    register_builtin_tools(reg)
    assert not reg.has("Git")


def test_register_git_tool():
    reg = ToolRegistry()
    name = register_git_tool(reg)
    assert name == "Git"
    assert reg.has("Git")


def test_git_tool_metadata():
    assert GIT_TOOL.name == "Git"
    assert GIT_TOOL.requires_capability is True
    assert GIT_TOOL.requires_fs == ()
