"""Tests for kernel.sandbox (RFC 0008).

Strategy: spawn real subprocesses that try to violate each enforced
limit and verify the kernel sandbox catches them. POSIX-only; the whole
file is skipped on Windows.

bubblewrap-dependent tests skip cleanly when bwrap isn't installed.
"""
from __future__ import annotations

import os
import shutil
import sys
import textwrap
import time
from pathlib import Path

import pytest

from cheetahclaws.kernel import (
    SANDBOX_DEFAULT,
    SANDBOX_OFF,
    SANDBOX_STRICT,
    SandboxPolicy,
    SandboxPolicyError,
    SandboxNotAvailable,
    apply_rlimits_in_child,
    detect_isolation_tools,
    run_sandboxed,
    wrap_with_bubblewrap,
)


# Whole-file skip on non-POSIX. RLIMIT and process-group semantics are
# Linux/macOS-only; the kernel sandbox tolerates Windows via a no-op
# preexec_fn but the assertions below assume real enforcement.
pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="kernel.sandbox tests require POSIX (RLIMIT + setsid)",
)


HAS_BWRAP = shutil.which("bwrap") is not None


def _py_script(body: str) -> list[str]:
    """Build an argv that runs ``body`` as a one-liner Python script."""
    return [sys.executable, "-c", textwrap.dedent(body)]


def _python_bind_dirs() -> tuple[str, ...]:
    """Return the bwrap bind paths needed to run sys.executable.

    Distros install python at /usr/bin/python, conda at $HOME/anaconda3,
    venvs at ~/.virtualenvs/<name>. We bind the executable's bin/ AND
    the conda env root (so site-packages is visible) read-only.

    We deliberately do NOT call ``Path.resolve()`` — when $HOME contains
    a symlink (e.g. /home/x -> /srv/home/x), resolve() rewrites the
    binding's source path while sys.executable still names the original;
    bwrap then binds /srv/... but the child tries to exec /home/...
    Without resolve(), the paths stay consistent.
    """
    paths = ["/usr", "/lib", "/lib64", "/bin", "/etc"]
    py = Path(sys.executable)
    if py.parent.name == "bin":
        env_root = py.parent.parent
        if str(env_root) not in paths:
            paths.append(str(env_root))
    elif str(py.parent) not in paths:
        paths.append(str(py.parent))
    return tuple(p for p in paths if Path(p).exists())


# ── Policy validation ──────────────────────────────────────────────────────


def test_policy_accepts_defaults():
    p = SandboxPolicy()
    assert p.cpu_seconds is None
    assert p.use_bubblewrap is False
    assert p.new_session is True


def test_policy_rejects_negative_cpu():
    with pytest.raises(SandboxPolicyError):
        SandboxPolicy(cpu_seconds=0)
    with pytest.raises(SandboxPolicyError):
        SandboxPolicy(cpu_seconds=-1)


def test_policy_rejects_non_int_memory():
    with pytest.raises(SandboxPolicyError):
        SandboxPolicy(memory_bytes="100M")  # type: ignore[arg-type]


def test_policy_rejects_relative_bind():
    with pytest.raises(SandboxPolicyError):
        SandboxPolicy(bind_ro=("relative/path",))


def test_policy_rejects_deny_network_without_bwrap():
    with pytest.raises(SandboxPolicyError):
        SandboxPolicy(deny_network=True, use_bubblewrap=False)


def test_policy_accepts_named_profiles():
    # Profiles are pre-validated; constructing them should not raise.
    assert SANDBOX_OFF.cpu_seconds is None
    assert SANDBOX_DEFAULT.cpu_seconds == 300
    assert SANDBOX_STRICT.use_bubblewrap is True


# ── Tool detection ─────────────────────────────────────────────────────────


def test_detect_returns_dict_with_keys():
    tools = detect_isolation_tools()
    assert set(tools.keys()) == {"bubblewrap", "firejail"}
    # Each value is either a path string or None.
    for v in tools.values():
        assert v is None or isinstance(v, str)


# ── apply_rlimits_in_child returns a callable ──────────────────────────────


def test_apply_rlimits_returns_callable():
    fn = apply_rlimits_in_child(SandboxPolicy())
    assert callable(fn)


# ── run_sandboxed: clean exit ──────────────────────────────────────────────


def test_clean_exit_returns_zero():
    result = run_sandboxed(
        _py_script("print('hello')"),
        SandboxPolicy(wall_seconds=10),
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == b"hello"
    assert not result.timed_out
    assert not result.killed


def test_capture_stderr_separately():
    result = run_sandboxed(
        _py_script("import sys; sys.stderr.write('err'); print('out')"),
        SandboxPolicy(wall_seconds=10),
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == b"out"
    assert result.stderr.strip() == b"err"


def test_stdin_pipe():
    result = run_sandboxed(
        _py_script("import sys; print(sys.stdin.read().upper())"),
        SandboxPolicy(wall_seconds=10),
        stdin=b"hello\n",
    )
    assert result.exit_code == 0
    assert result.stdout.strip() == b"HELLO"


# ── RLIMIT_AS: virtual memory ──────────────────────────────────────────────


def test_rlimit_as_kills_runaway_alloc():
    # Try to allocate 4 GB; cap at 256 MB. The Python child should die
    # with MemoryError or be killed; in either case exit_code != 0.
    body = """
        try:
            x = b'x' * (4 * 1024 * 1024 * 1024)
            print('SOMEHOW_OK', len(x))
        except MemoryError:
            print('CAUGHT_MEMERR')
    """
    result = run_sandboxed(
        _py_script(body),
        SandboxPolicy(memory_bytes=256 * 1024 * 1024, wall_seconds=15),
    )
    # Either Python caught the MemoryError (exit 0, output CAUGHT_MEMERR)
    # or the kernel killed it. Both prove RLIMIT_AS is enforced.
    assert b"SOMEHOW_OK" not in result.stdout, \
        f"4GB alloc succeeded under 256MB cap! result={result}"


# ── RLIMIT_CPU: CPU time ──────────────────────────────────────────────────


def test_rlimit_cpu_kills_busy_loop():
    body = """
        i = 0
        while True:
            i += 1
            # Avoid sleep — we want CPU time consumption.
    """
    result = run_sandboxed(
        _py_script(body),
        # 1s soft CPU limit. SIGXCPU after 1s; SIGKILL shortly after if
        # the child doesn't exit. wall_seconds is the safety net.
        SandboxPolicy(cpu_seconds=1, wall_seconds=15),
    )
    assert result.exit_code != 0
    # Either SIGXCPU (-24) or rlimit-classified as 'cpu' or wall-killed.
    # We accept anything that didn't run forever. The duration must be
    # bounded by wall_seconds.
    assert result.duration_s < 15
    assert result.killed or result.rlimit_hit == "cpu"


# ── RLIMIT_FSIZE: max file size ───────────────────────────────────────────


def test_rlimit_fsize_caps_writes(tmp_path):
    target = tmp_path / "big.bin"
    body = f"""
        import os
        # Try to write 100 MB; cap at 4 MB.
        with open({str(target)!r}, 'wb') as f:
            for _ in range(100):
                f.write(b'x' * 1024 * 1024)
        print('WROTE_FULL')
    """
    result = run_sandboxed(
        _py_script(body),
        SandboxPolicy(fsize_bytes=4 * 1024 * 1024, wall_seconds=10),
    )
    assert b"WROTE_FULL" not in result.stdout
    # File size is capped at our limit.
    if target.exists():
        assert target.stat().st_size <= 4 * 1024 * 1024


# ── Wall-clock killer ──────────────────────────────────────────────────────


def test_wall_seconds_kills_sleeper():
    body = "import time; time.sleep(60); print('SOMEHOW_DONE')"
    start = time.monotonic()
    result = run_sandboxed(
        _py_script(body),
        SandboxPolicy(wall_seconds=1.5),
    )
    elapsed = time.monotonic() - start
    assert result.timed_out
    assert result.killed
    assert b"SOMEHOW_DONE" not in result.stdout
    # Should be killed in roughly wall_seconds + 1s grace, not 60s.
    assert elapsed < 10, f"sleeper ran for {elapsed}s — wall-killer broken"


def test_wall_seconds_does_not_fire_for_quick_jobs():
    result = run_sandboxed(
        _py_script("print('done')"),
        SandboxPolicy(wall_seconds=10),
    )
    assert not result.timed_out
    assert result.exit_code == 0


def test_no_wall_seconds_means_no_killer_thread():
    # Quick exit shouldn't time out if wall_seconds is None.
    result = run_sandboxed(
        _py_script("print('q')"),
        SandboxPolicy(),
    )
    assert not result.timed_out
    assert result.exit_code == 0


# ── Process group / setsid ─────────────────────────────────────────────────


def test_new_session_kills_descendants():
    """An agent that forks a child must have the whole tree killed
    when the wall-clock fires."""
    body = """
        import os, time
        # Fork a long-sleeping grandchild.
        if os.fork() == 0:
            time.sleep(60)
            os._exit(0)
        else:
            time.sleep(60)
            print('PARENT_DONE')
    """
    start = time.monotonic()
    result = run_sandboxed(
        _py_script(body),
        SandboxPolicy(wall_seconds=1.5),
    )
    elapsed = time.monotonic() - start
    assert elapsed < 10
    assert result.timed_out


# ── Bubblewrap layer (skipped if not installed) ────────────────────────────


@pytest.mark.skipif(not HAS_BWRAP, reason="bwrap not installed")
def test_wrap_with_bubblewrap_argv_shape():
    argv = ["/usr/bin/python3", "-c", "print('x')"]
    cmd = wrap_with_bubblewrap(
        argv,
        SandboxPolicy(use_bubblewrap=True,
                      bind_ro=("/usr",), bind_rw=("/tmp",)),
    )
    # bwrap is the entrypoint; user argv ends the list.
    assert cmd[0].endswith("bwrap")
    assert "--unshare-pid" in cmd
    assert "--unshare-ipc" in cmd
    assert "--die-with-parent" in cmd
    # ro-bind / bind args present.
    assert "--ro-bind" in cmd
    assert "--bind" in cmd
    # User argv at the tail.
    assert cmd[-3:] == argv


def test_wrap_with_bubblewrap_raises_when_missing(monkeypatch):
    # Force shutil.which to claim bwrap is missing.
    monkeypatch.setattr(shutil, "which",
                        lambda name: None if name == "bwrap" else shutil.which(name))
    with pytest.raises(SandboxNotAvailable):
        wrap_with_bubblewrap(
            ["/bin/true"],
            SandboxPolicy(use_bubblewrap=True),
        )


@pytest.mark.skipif(not HAS_BWRAP, reason="bwrap not installed")
def test_bubblewrap_isolates_filesystem(tmp_path):
    """An agent under bubblewrap with no bind to ~/.ssh should not be
    able to read it."""
    home = Path.home()
    ssh = home / ".ssh"
    if not ssh.exists():
        # Nothing to hide — the test would be meaningless.
        pytest.skip("no ~/.ssh on this host")

    body = f"""
        import os
        path = {str(ssh)!r}
        try:
            os.listdir(path)
            print('READ_OK')
        except (FileNotFoundError, PermissionError):
            print('READ_BLOCKED')
    """
    result = run_sandboxed(
        _py_script(body),
        SandboxPolicy(
            use_bubblewrap=True,
            bind_ro=_python_bind_dirs(),
            wall_seconds=10,
        ),
    )
    assert b"READ_BLOCKED" in result.stdout, \
        f"sandbox failed to hide ~/.ssh — stdout={result.stdout!r} stderr={result.stderr!r}"


@pytest.mark.skipif(not HAS_BWRAP, reason="bwrap not installed")
def test_bubblewrap_blocks_network():
    """deny_network=True + bubblewrap should make socket() fail."""
    body = """
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            s.connect(('1.1.1.1', 80))
            print('CONNECTED')
        except (OSError, socket.gaierror) as e:
            print('BLOCKED')
    """
    result = run_sandboxed(
        _py_script(body),
        SandboxPolicy(
            use_bubblewrap=True,
            deny_network=True,
            bind_ro=_python_bind_dirs(),
            wall_seconds=10,
        ),
    )
    assert b"CONNECTED" not in result.stdout
    assert b"BLOCKED" in result.stdout


# ── env / cwd plumbing ─────────────────────────────────────────────────────


def test_env_is_passed_through():
    result = run_sandboxed(
        _py_script("import os; print(os.environ.get('CC_KERNEL_TEST', 'absent'))"),
        SandboxPolicy(wall_seconds=10),
        env={"CC_KERNEL_TEST": "present", "PATH": os.environ.get("PATH", "")},
    )
    assert result.stdout.strip() == b"present"


def test_cwd_is_respected(tmp_path):
    result = run_sandboxed(
        _py_script("import os; print(os.getcwd())"),
        SandboxPolicy(wall_seconds=10),
        cwd=str(tmp_path),
    )
    # macOS may resolve /tmp -> /private/tmp; allow both forms.
    out = result.stdout.strip().decode()
    assert out == str(tmp_path) or os.path.realpath(out) == os.path.realpath(str(tmp_path))


# ── Bind path validation ──────────────────────────────────────────────────


def test_run_sandboxed_rejects_nonexistent_bind():
    with pytest.raises(SandboxPolicyError):
        run_sandboxed(
            ["/bin/true"],
            SandboxPolicy(use_bubblewrap=HAS_BWRAP,
                          bind_ro=("/this/path/does/not/exist",)),
        )
