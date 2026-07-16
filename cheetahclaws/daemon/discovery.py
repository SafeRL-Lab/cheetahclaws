"""Discovery file: tells clients where a running daemon lives.

The daemon writes ``~/.cheetahclaws/daemon.json`` when it binds, and removes
it on clean exit.  Clients call :func:`locate` to learn whether a daemon is
running and how to reach it.

Schema::

    { "pid":        12345,
      "started_at": "2026-04-30T12:00:00Z",
      "transport":  "unix" | "tcp",
      "address":    "/run/user/1000/cheetahclaws/daemon.sock"
                  | "127.0.0.1:8765",
      "version":    "3.5.72",
      "schema":     1 }

Atomic write semantics: writes go through a sibling ``.tmp`` file then
``os.replace``, so an interrupted write never leaves the discovery file
half-overwritten.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 1
DEFAULT_FILENAME = "daemon.json"


# ── Path resolution ────────────────────────────────────────────────────────

def get_default_path() -> Path:
    """Default discovery-file location: ``~/.cheetahclaws/daemon.json``."""
    from cheetahclaws.config import CONFIG_DIR
    return CONFIG_DIR / DEFAULT_FILENAME


def _resolve(path: Optional[Path]) -> Path:
    return path if path is not None else get_default_path()


# ── Info builder ───────────────────────────────────────────────────────────

def make_info(*, pid: int, transport: str, address: str,
              version: str, token_path: Optional[str] = None) -> dict:
    """Build a discovery dict ready for :func:`write`.

    ``token_path`` is recorded only when the daemon was started with a
    non-default --token-path so daemon-control verbs (status / stop /
    rotate-token) can find the token the daemon is actually using.
    Schema stays at version 1 — this is a strictly additive field.
    """
    info: dict = {
        "pid":        pid,
        "started_at": datetime.datetime.now(datetime.timezone.utc)
                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "transport":  transport,
        "address":    address,
        "version":    version,
        "schema":     SCHEMA_VERSION,
    }
    if token_path is not None:
        info["token_path"] = token_path
    return info


# ── Read / write / clear ───────────────────────────────────────────────────

def write(info: dict, *, path: Optional[Path] = None) -> None:
    """Atomically write *info* to the discovery file (mode 0600).

    Writes through a sibling ``.tmp`` file then ``os.replace`` so a crash
    mid-write cannot corrupt an existing discovery file.  If ``os.replace``
    fails, the temp file is best-effort removed and the exception bubbles.
    """
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    tmp = p.with_suffix(p.suffix + ".tmp")
    data = json.dumps(info, indent=2).encode("utf-8")
    # Open with 0600 so the file mode is correct from the moment data lands;
    # avoids a window where a world-readable temp file could be observed.
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)

    try:
        os.replace(str(tmp), str(p))
    except Exception:
        # Best-effort cleanup of the temp file; do not mask the real error.
        try:
            os.unlink(str(tmp))
        except OSError:
            pass
        raise

    # On POSIX, re-assert mode in case umask or filesystem altered it.
    if os.name != "nt":
        os.chmod(str(p), 0o600)


def read(*, path: Optional[Path] = None) -> Optional[dict]:
    """Return the parsed discovery file, or ``None`` if absent / unreadable."""
    p = _resolve(path)
    try:
        text = p.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def clear(*, path: Optional[Path] = None) -> None:
    """Remove the discovery file.  Idempotent — missing file is not an error."""
    p = _resolve(path)
    try:
        p.unlink()
    except FileNotFoundError:
        pass


# ── Liveness probe ─────────────────────────────────────────────────────────

def pid_alive(pid: int) -> bool:
    """Best-effort cross-platform check that *pid* is currently running."""
    if pid <= 0:
        return False
    if os.name == "nt":
        return _pid_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't own it; for our purposes that counts.
        return True
    except OSError:
        return False
    return True


def _pid_alive_windows(pid: int) -> bool:
    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION,
                                   False, pid)
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


# ── Locate ─────────────────────────────────────────────────────────────────

def locate(*, path: Optional[Path] = None) -> Optional[dict]:
    """Return discovery info if a live daemon is registered.

    If the file exists but the recorded pid is no longer running the file is
    auto-cleared and ``None`` is returned, so callers do not need a separate
    stale-file step.
    """
    info = read(path=path)
    if info is None:
        return None
    pid = info.get("pid")
    if not isinstance(pid, int) or not pid_alive(pid):
        clear(path=path)
        return None
    return info
