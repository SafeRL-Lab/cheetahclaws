"""
bridges/interactive_session.py — PTY-based interactive terminal sessions for bridges.

Key design decisions
--------------------
* pyte Screen is PERSISTENT for the session lifetime — it accumulates all bytes
  from the process so cursor-positioning escape sequences are applied correctly.
  (Recreating it on each flush loses screen state and garbles output.)

* send_input() sends <text>\r (carriage-return) not \n.  Claude Code / Ink and
  most raw-mode TUI programs treat CR as "submit", not LF.

* Flush is edge-triggered: we only send when the rendered screen content has
  actually changed from what we last sent, and only after output has been silent
  for _SETTLE_TIMEOUT seconds (so we don't spam partial renders).

* For programs that stream long responses (like claude), _SETTLE_TIMEOUT is
  deliberately generous (3 s) so the user gets a complete answer in one message.
"""
from __future__ import annotations

import fcntl
import os
import re
import select
import struct
import subprocess
import threading
import time
from typing import Callable

import logging_utils as _log

# ── pyte: proper vt100 terminal emulator ─────────────────────────────────
try:
    import pyte as _pyte
    _HAVE_PYTE = True
except ImportError:
    _HAVE_PYTE = False

# ── Registry ──────────────────────────────────────────────────────────────
_sessions: dict[str, "InteractiveSession"] = {}
_sessions_lock = threading.Lock()


def get_session(key: str) -> "InteractiveSession | None":
    with _sessions_lock:
        s = _sessions.get(key)
        if s and not s.is_alive:
            _sessions.pop(key, None)
            return None
        return s


def set_session(key: str, session: "InteractiveSession") -> None:
    with _sessions_lock:
        old = _sessions.pop(key, None)
        if old:
            old.kill()
        _sessions[key] = session


def remove_session(key: str) -> bool:
    with _sessions_lock:
        s = _sessions.pop(key, None)
    if s:
        s.kill()
        return True
    return False


# ── Fallback text cleaning (when pyte not available) ──────────────────────
_ANSI_RE    = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1B\\))')
_CTRL_RE    = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_BOX_RE     = re.compile(r'[\u2500-\u259f\u25a0-\u25ff]')
_MULTI_SP   = re.compile(r' {3,}')
_BLANK4     = re.compile(r'\n{4,}')


def _clean_fallback(raw: str) -> str:
    t = _ANSI_RE.sub('', raw)
    t = _CTRL_RE.sub('', t)
    t = _BOX_RE.sub('', t)
    t = _MULTI_SP.sub('  ', t)
    t = _BLANK4.sub('\n\n', t)
    return t


# ── InteractiveSession ─────────────────────────────────────────────────────

_PTY_COLS = 80
_PTY_ROWS = 24

_MAX_CHUNK        = 3500  # chars per bridge message
_SETTLE           = 3.0   # seconds of silence before flushing (generous for API calls)
_SETTLE_AFTER_INPUT = 1.5 # shorter settle used for 8 s after user sends input —
                          # quick commands (nvidia-smi, ls) settle faster so the user
                          # doesn't think their input was ignored and resend it
_POST_INPUT_WINDOW  = 8.0 # seconds after input to use the shorter settle
_FORCE_FLUSH      = 8.0   # always flush after this long even if output still streaming


class InteractiveSession:
    """A running process on a pseudo-TTY, bridged to a phone via send_fn."""

    def __init__(self, cmd: str, send_fn: Callable[[str], None],
                 session_key: str = "") -> None:
        # Enabled by default; bridges already enforce owner-only via chat_id
        # whitelist. Operators can hard-disable with CHEETAHCLAWS_BRIDGE_TERMINAL=0.
        if os.environ.get("CHEETAHCLAWS_BRIDGE_TERMINAL", "1") == "0":
            raise RuntimeError(
                "Remote interactive session is disabled "
                "(CHEETAHCLAWS_BRIDGE_TERMINAL=0)."
            )
        if not isinstance(cmd, str) or "\x00" in cmd or len(cmd) > 4096:
            raise RuntimeError("Refused: command empty, too long, or contains NUL.")
        try:
            from tools.shell import _bash_hard_denied
            denied = _bash_hard_denied(cmd)
        except Exception:
            denied = None
        if denied:
            raise RuntimeError(denied)

        self.cmd         = cmd
        self.send_fn     = send_fn
        self.session_key = session_key
        self._dead           = False
        self._last_sent      = ""          # deduplicate identical screen renders
        self._last_input_time: float = 0.0  # monotonic time of last send_input() call
        # _next_force_flush: when set by send_input(), _read_loop will flush at that
        # monotonic time even if output is still streaming.  Resets to inf after flush.
        self._next_force_flush: float = float("inf")

        # ── pyte persistent screen ────────────────────────────────────────
        if _HAVE_PYTE:
            self._screen: "_pyte.Screen | None" = _pyte.Screen(_PTY_COLS, _PTY_ROWS)
            self._pyte_stream: "_pyte.ByteStream | None" = _pyte.ByteStream(self._screen)
        else:
            self._screen = None
            self._pyte_stream = None

        # Raw bytes accumulated since last flush
        self._raw_buf: bytearray = bytearray()

        # ── Open PTY ─────────────────────────────────────────────────────
        try:
            self.master_fd, slave_fd = os.openpty()
        except OSError as exc:
            raise RuntimeError(f"openpty failed: {exc}") from exc

        try:
            import termios
            # Set window size (80×24) so TUI apps render correctly
            winsize = struct.pack("HHHH", _PTY_ROWS, _PTY_COLS, 0, 0)
            fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, winsize)
            # Keep echo ON — TUI programs manage their own echo
        except Exception:
            pass

        self.proc = subprocess.Popen(
            cmd,
            shell=True,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            env={**os.environ,
                 "TERM": "xterm-256color",
                 "COLUMNS": str(_PTY_COLS),
                 "LINES": str(_PTY_ROWS)},
        )
        os.close(slave_fd)

        # Non-blocking reads on master
        flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
        fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        _log.info("interactive_session_start",
                  key=session_key, cmd=cmd[:200],
                  pid=self.proc.pid, pyte=_HAVE_PYTE)

        self._reader = threading.Thread(
            target=self._read_loop, daemon=True,
            name=f"pty-{session_key}"
        )
        self._reader.start()

    # ── Output reader ──────────────────────────────────────────────────────

    def _read_loop(self) -> None:
        last_data  = time.monotonic()
        last_flush = time.monotonic()

        while not self._dead:
            try:
                r, _, _ = select.select([self.master_fd], [], [], 0.2)
            except (ValueError, OSError):
                break

            if r:
                try:
                    chunk = os.read(self.master_fd, 4096)
                    self._raw_buf.extend(chunk)
                    last_data = time.monotonic()
                except (OSError, IOError):
                    break

            now     = time.monotonic()
            silence = now - last_data

            # Use a shorter settle window for _POST_INPUT_WINDOW seconds after the
            # user sends input.  This means quick commands (nvidia-smi, ls, etc.)
            # appear on the phone within ~2 s instead of ~3 s, preventing the user
            # from thinking their input was ignored and resending it.
            post_input = (now - self._last_input_time) < _POST_INPUT_WINDOW
            settle = _SETTLE_AFTER_INPUT if post_input else _SETTLE

            # Normal flush: data buffered + silence settled or force-flush timeout
            if self._raw_buf and (
                silence >= settle
                or now - last_flush >= _FORCE_FLUSH
            ):
                self._flush()
                last_flush = now

            # Post-input force flush: fires 3.5 s after send_input() regardless of
            # whether _raw_buf is empty.  Re-renders pyte screen so the user sees
            # the result even when Claude Code hasn't produced output yet.
            if now >= self._next_force_flush:
                self._next_force_flush = float("inf")   # consume timer
                self._last_sent = ""                    # defeat dedup
                if self._raw_buf:
                    self._flush()
                elif self._screen is not None:
                    text = self._render_screen().strip()
                    if not text:
                        # Screen mid-clear — wait briefly and retry
                        time.sleep(0.8)
                        text = self._render_screen().strip()
                    if text:
                        self._last_sent = text
                        for i in range(0, len(text), _MAX_CHUNK):
                            try:
                                self.send_fn(f"```\n{text[i:i+_MAX_CHUNK]}\n```")
                            except Exception:
                                pass
                last_flush = now

            # Process exited
            if self.proc.poll() is not None:
                self._dead = True
                if self._raw_buf:
                    self._flush()
                rc = self.proc.returncode
                self.send_fn(f"⏹ Session ended (exit {rc}).")
                _log.info("interactive_session_end",
                          key=self.session_key, returncode=rc)
                with _sessions_lock:
                    _sessions.pop(self.session_key, None)
                break

        try:
            os.close(self.master_fd)
        except OSError:
            pass

    def _flush(self) -> None:
        """Feed accumulated bytes to pyte and send the rendered screen."""
        if not self._raw_buf:
            return
        raw = bytes(self._raw_buf)
        self._raw_buf.clear()

        # Feed into persistent pyte screen (or fallback clean)
        if self._pyte_stream is not None and self._screen is not None:
            self._pyte_stream.feed(raw)
            text = self._render_screen()
        else:
            text = _clean_fallback(raw.decode("utf-8", errors="replace"))

        text = text.strip()
        if not text:
            return

        # Deduplicate: don't resend identical screen content
        if text == self._last_sent:
            return
        self._last_sent = text

        for i in range(0, len(text), _MAX_CHUNK):
            try:
                self.send_fn(f"```\n{text[i:i+_MAX_CHUNK]}\n```")
            except Exception:
                pass

    def _render_screen(self) -> str:
        """Extract visible text from the persistent pyte screen."""
        assert self._screen is not None
        lines = []
        for y in range(self._screen.lines):
            line = "".join(
                self._screen.buffer[y][x].data
                for x in range(self._screen.columns)
            ).rstrip()
            lines.append(line)
        text = "\n".join(lines)
        # Trim blank lines at top/bottom and collapse runs of 4+
        text = _BLANK4.sub("\n\n", text).strip()
        return text

    # ── Input ──────────────────────────────────────────────────────────────

    def send_input(self, text: str) -> None:
        """Send a line to the process.

        Uses \\r (carriage-return) not \\n — raw-mode TUI programs (Claude Code,
        Python REPL, bash) all treat CR as "submit/Enter".

        Also clears _last_sent so the next flush is always delivered even if
        the screen content hasn't changed (avoids dedup-silencing responses).

        Schedules a 6-second delayed flush so the response reaches the user even
        when the process generates continuous output that prevents _SETTLE from
        firing (e.g. nvidia-smi or a fast-running command).
        """
        if self._dead:
            return

        stripped = text.strip()

        # ── Numbered-menu detection ────────────────────────────────────────
        # Ink's SelectInput only responds to arrow keys + Enter, NOT to digit
        # keys.  Claude Code's permission prompt shows "❯ 1. Yes" options.
        # When the user sends a single digit (1/2/3), we translate it to the
        # correct ANSI escape sequence:
        #   1 → \r                  (cursor already on item 1)
        #   2 → ESC[B \r            (one Down arrow, then Enter)
        #   3 → ESC[B ESC[B \r      (two Down arrows, then Enter)
        # Without this translation, digit '2' can be misread by Ink as two
        # Down presses → landing on "No" → command rejected → loop.
        payload = text
        if len(stripped) == 1 and stripped.isdigit() and self._screen is not None:
            n = int(stripped)
            if 1 <= n <= 9:
                screen_text = self._render_screen()
                # Match Claude Code's permission prompt ("❯ 1." style menu)
                if "❯ 1." in screen_text:
                    _DOWN = "\x1b[B"          # ANSI cursor-down
                    payload = _DOWN * (n - 1) # (n-1) downs from the first item

        try:
            os.write(self.master_fd, (payload + "\r").encode("utf-8"))
            now = time.monotonic()
            self._last_sent = ""
            self._last_input_time = now
            self._next_force_flush = now + 3.5
            _log.debug("interactive_input_sent",
                       key=self.session_key, text=text[:40], payload_len=len(payload))
        except OSError as exc:
            _log.warn("interactive_write_error",
                      key=self.session_key, error=str(exc))

    def force_flush(self) -> None:
        """Force-send current screen content regardless of dedup state.

        Handles two cases:
        * Raw bytes buffered → feed them into pyte and send the rendered screen.
        * No raw bytes       → re-render the current pyte state and send.

        If the rendered screen is empty (happens when the process just cleared the
        screen mid-redraw), we wait 1.5 s and retry once so the user gets the new
        content rather than silence.
        """
        self._last_sent = ""   # defeat deduplication

        if self._raw_buf:
            self._flush()
            return

        if self._screen is None:
            return

        def _render_and_send() -> bool:
            text = self._render_screen().strip()
            if not text:
                return False
            self._last_sent = text
            for i in range(0, len(text), _MAX_CHUNK):
                try:
                    self.send_fn(f"```\n{text[i:i+_MAX_CHUNK]}\n```")
                except Exception:
                    pass
            return True

        if not _render_and_send():
            # Screen temporarily empty (mid clear/redraw) — wait and retry once
            time.sleep(1.5)
            if self._raw_buf:
                self._flush()   # new data arrived while we waited
            else:
                _render_and_send()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def kill(self) -> None:
        self._dead = True
        try:
            self.proc.kill()
        except Exception:
            pass
        try:
            os.close(self.master_fd)
        except Exception:
            pass
        _log.info("interactive_session_killed", key=self.session_key)

    @property
    def is_alive(self) -> bool:
        return not self._dead and self.proc.poll() is None
