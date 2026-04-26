"""Incremental parser for thinking XML tags in text stream, with loop detection."""

import re

OPEN_TAG = "<" + "thinking>"
CLOSE_TAG = "</" + "thinking>"
_THINKING_RE = re.compile(r"<" + r"thinking>.*?</" + r"thinking>", re.DOTALL)


class ThinkingStreamParser:
    """Parses a stream of text chunks, separating thinking blocks from regular text."""

    def __init__(self):
        self._buffer = ""
        self._in_thinking = False
        self._thinking_text: list[str] = []

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        """Feed a chunk. Returns list of ("text", content) or ("thinking", content) tuples."""
        self._buffer += chunk
        results: list[tuple[str, str]] = []

        while self._buffer:
            if self._in_thinking:
                end_idx = self._buffer.find(CLOSE_TAG)
                if end_idx != -1:
                    thinking_content = self._buffer[:end_idx]
                    if thinking_content:
                        results.append(("thinking", thinking_content))
                        self._thinking_text.append(thinking_content)
                    self._buffer = self._buffer[end_idx + len(CLOSE_TAG):]
                    self._in_thinking = False
                else:
                    for i in range(1, min(len(CLOSE_TAG), len(self._buffer) + 1)):
                        if CLOSE_TAG[:i] == self._buffer[-i:]:
                            emit = self._buffer[:-i]
                            if emit:
                                results.append(("thinking", emit))
                                self._thinking_text.append(emit)
                            self._buffer = self._buffer[-i:]
                            return results
                    results.append(("thinking", self._buffer))
                    self._thinking_text.append(self._buffer)
                    self._buffer = ""
            else:
                start_idx = self._buffer.find(OPEN_TAG)
                if start_idx != -1:
                    text_before = self._buffer[:start_idx]
                    if text_before:
                        results.append(("text", text_before))
                    self._buffer = self._buffer[start_idx + len(OPEN_TAG):]
                    self._in_thinking = True
                else:
                    for i in range(1, min(len(OPEN_TAG), len(self._buffer) + 1)):
                        if OPEN_TAG[:i] == self._buffer[-i:]:
                            emit = self._buffer[:-i]
                            if emit:
                                results.append(("text", emit))
                            self._buffer = self._buffer[-i:]
                            return results
                    results.append(("text", self._buffer))
                    self._buffer = ""

        return results

    def finalize(self) -> list[tuple[str, str]]:
        """Flush remaining buffer."""
        results: list[tuple[str, str]] = []
        if self._buffer:
            kind = "thinking" if self._in_thinking else "text"
            results.append((kind, self._buffer))
            if self._in_thinking:
                self._thinking_text.append(self._buffer)
            self._buffer = ""
        return results

    @property
    def full_thinking_text(self) -> str:
        return "".join(self._thinking_text)

    @property
    def in_thinking(self) -> bool:
        return self._in_thinking


class LoopDetector:
    """Detects repetitive patterns in a stream of text."""

    WINDOW_SIZE = 2000
    PATTERN_LENGTHS = list(range(20, 51)) + list(range(55, 201, 5))
    MIN_REPEATS = 8

    def __init__(self):
        self._window = ""

    def feed(self, text: str) -> bool:
        """Feed text. Returns True if a loop is detected."""
        self._window += text
        if len(self._window) > self.WINDOW_SIZE:
            self._window = self._window[-self.WINDOW_SIZE:]
        if len(self._window) < 200:
            return False
        return any(self._check_pattern(plen) for plen in self.PATTERN_LENGTHS)

    def _check_pattern(self, pattern_length: int) -> bool:
        needed = pattern_length * self.MIN_REPEATS
        if len(self._window) < needed:
            return False
        tail = self._window[-needed:]
        pattern = tail[:pattern_length]
        return all(
            tail[i : i + pattern_length] == pattern
            for i in range(pattern_length, needed, pattern_length)
        )


def strip_thinking_tags(content: str) -> str:
    """Remove all thinking blocks from content."""
    return _THINKING_RE.sub("", content).strip()
