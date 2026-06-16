"""ipc.py — newline-delimited JSON channel for runner IPC.

Wraps a pair of binary streams (stdin/stdout pipes) and exposes
``send(dict)`` / ``recv(timeout) -> dict``. Tests can construct
JsonLineChannel directly over io.BytesIO pairs to verify the
protocol without spawning real subprocesses.
"""
from __future__ import annotations

import json
import select
import threading
import time
from typing import Optional


class IpcReadTimeout(Exception):
    """recv() timed out without receiving a complete line."""


class JsonLineChannel:
    """Bidirectional JSON-line channel.

    ``inbound`` is a binary file (must support .readline / .read /
    .fileno). ``outbound`` is a binary file (must support .write /
    .flush). For subprocess.Popen pipes the supervisor passes:

        inbound  = proc.stdout    (read from runner)
        outbound = proc.stdin     (write to runner)

    The runner side does the inverse.

    ``recv`` is line-buffered: it reads one full line then parses. If
    the line is longer than ``max_line_bytes`` (default 1 MB), the
    overflow truncates and the remainder is discarded — protects
    against a runaway runner flooding the pipe with one giant line.

    Thread safety: send and recv each acquire their own lock; the
    channel is safe to use from one reader thread + one writer thread
    concurrently.
    """

    DEFAULT_MAX_LINE_BYTES = 1 * 1024 * 1024  # 1 MB

    def __init__(
        self,
        inbound,
        outbound,
        *,
        max_line_bytes: int = DEFAULT_MAX_LINE_BYTES,
    ) -> None:
        self._in = inbound
        self._out = outbound
        self._max_line_bytes = max_line_bytes
        self._send_lock = threading.Lock()
        self._recv_lock = threading.Lock()
        # Cross-call read buffer. A subprocess that emits multiple
        # lines in a single write would otherwise lose the trailing
        # lines if we only returned the first newline-delimited slice.
        self._rx_buf = bytearray()

    def send(self, message: dict) -> None:
        if not isinstance(message, dict):
            raise TypeError(f"message must be dict, got {type(message).__name__}")
        line = json.dumps(message, separators=(",", ":"),
                          sort_keys=True).encode("utf-8") + b"\n"
        with self._send_lock:
            self._out.write(line)
            self._out.flush()

    def recv(self, timeout: Optional[float] = None) -> dict:
        """Read one JSON object. Raises IpcReadTimeout on timeout, or
        EOFError on EOF before a complete line."""
        with self._recv_lock:
            line = self._read_one_line(timeout)
            if not line:
                raise EOFError("runner closed pipe before sending a line")
            if len(line) > self._max_line_bytes:
                line = line[:self._max_line_bytes]
            try:
                obj = json.loads(line.decode("utf-8", errors="replace"))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"runner sent invalid JSON: {line[:200]!r}"
                ) from e
            if not isinstance(obj, dict):
                raise ValueError(
                    f"runner sent non-dict JSON: {type(obj).__name__}"
                )
            return obj

    def _read_one_line(self, timeout: Optional[float]) -> bytes:
        """Return one newline-terminated line (the trailing \\n
        included), preserving any post-newline bytes in self._rx_buf
        for the next call. On EOF before a newline, returns whatever
        was buffered (possibly empty)."""
        # Already have a complete line in the buffer?
        nl = self._rx_buf.find(b"\n")
        if nl != -1:
            line = bytes(self._rx_buf[:nl + 1])
            del self._rx_buf[:nl + 1]
            return line

        # Need to read more.
        try:
            fd = self._in.fileno()
            selectable = True
        except (AttributeError, OSError, ValueError):
            selectable = False

        if not selectable:
            # Test path: BytesIO etc. readline returns one line.
            line = self._in.readline()
            if not line:
                # EOF; return whatever's buffered (likely empty).
                if self._rx_buf:
                    leftover = bytes(self._rx_buf)
                    self._rx_buf.clear()
                    return leftover
                return b""
            return line

        if timeout is None:
            # Blocking read — readline on the buffered file goes
            # straight through.
            chunk = self._in.readline()
            if chunk:
                # readline returns ONE complete line (or partial on
                # EOF) — store nothing extra.
                return chunk
            # EOF
            if self._rx_buf:
                leftover = bytes(self._rx_buf)
                self._rx_buf.clear()
                return leftover
            return b""

        # Bounded read with select.
        deadline = time.monotonic() + timeout
        while True:
            nl = self._rx_buf.find(b"\n")
            if nl != -1:
                line = bytes(self._rx_buf[:nl + 1])
                del self._rx_buf[:nl + 1]
                return line
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise IpcReadTimeout(
                    f"no line within {timeout}s (buffered {len(self._rx_buf)} bytes)"
                )
            ready, _, _ = select.select([fd], [], [], min(remaining, 0.5))
            if not ready:
                continue
            chunk = self._in.read1(4096) if hasattr(self._in, "read1") \
                else self._in.read(4096)
            if not chunk:
                # EOF — return buffered leftover (which has no
                # newline since we'd have returned earlier).
                if self._rx_buf:
                    leftover = bytes(self._rx_buf)
                    self._rx_buf.clear()
                    return leftover
                return b""
            self._rx_buf.extend(chunk)
            if len(self._rx_buf) > self._max_line_bytes and b"\n" not in self._rx_buf:
                # Single line is way too big — return it truncated;
                # outer recv() will truncate JSON parse-friendly.
                line = bytes(self._rx_buf[:self._max_line_bytes])
                self._rx_buf = self._rx_buf[self._max_line_bytes:]
                return line

    def close(self) -> None:
        """Best-effort close of both streams."""
        for stream in (self._in, self._out):
            try:
                stream.close()
            except Exception:
                pass
