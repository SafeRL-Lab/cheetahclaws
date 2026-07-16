"""Tests for the adaptive streaming tiers in ui.render.

Covers:
  - auto_stream_mode device routing (live / commit / plain)
  - _safe_commit_point block detection (incl. code fences)
  - commit-mode stream_text / flush_response (append-only progressive Markdown)
  - the bounded, self-healing in-progress preview
"""
import platform

import pytest

import cheetahclaws.ui.render as render


# ── auto_stream_mode routing ────────────────────────────────────────────────

class _Console:
    def __init__(self, is_terminal=True, is_dumb_terminal=False, height=100, width=80):
        self.is_terminal = is_terminal
        self.is_dumb_terminal = is_dumb_terminal
        self.height = height
        self.width = width
        self.printed = []

    def print(self, value):
        self.printed.append(value)


@pytest.fixture
def clean_env(monkeypatch):
    """A real-TTY console + a baseline env with no terminal-identifying vars."""
    for var in ("SSH_CLIENT", "SSH_TTY", "TERM_PROGRAM", "TERM", "WT_SESSION",
                "KITTY_WINDOW_ID", "ALACRITTY_WINDOW_ID", "WEZTERM_PANE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(render, "_RICH", True)
    monkeypatch.setattr(render, "console", _Console())
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    return monkeypatch


def test_explicit_stream_mode_wins(clean_env):
    assert render.auto_stream_mode({"stream_mode": "plain"}) == "plain"
    assert render.auto_stream_mode({"stream_mode": "commit"}) == "commit"
    assert render.auto_stream_mode({"stream_mode": "live"}) == "live"


def test_legacy_rich_live_flag(clean_env):
    assert render.auto_stream_mode({"rich_live": True}) == "live"
    # Legacy False now maps to the rich append-only tier, not raw plain.
    assert render.auto_stream_mode({"rich_live": False}) == "commit"


def test_no_rich_is_plain(clean_env):
    clean_env.setattr(render, "_RICH", False)
    assert render.auto_stream_mode({}) == "plain"


def test_local_tty_gets_live(clean_env):
    assert render.auto_stream_mode({}) == "live"


def test_dumb_terminal_gets_commit(clean_env):
    clean_env.setattr(render, "console", _Console(is_dumb_terminal=True))
    assert render.auto_stream_mode({}) == "commit"


def test_non_tty_gets_commit(clean_env):
    clean_env.setattr(render, "console", _Console(is_terminal=False))
    assert render.auto_stream_mode({}) == "commit"


def test_unknown_ssh_terminal_gets_commit(clean_env):
    clean_env.setenv("SSH_CLIENT", "1.2.3.4 5555 22")
    assert render.auto_stream_mode({}) == "commit"


def test_modern_terminal_over_ssh_gets_live(clean_env):
    clean_env.setenv("SSH_CLIENT", "1.2.3.4 5555 22")
    clean_env.setenv("TERM_PROGRAM", "vscode")
    assert render.auto_stream_mode({}) == "live"


def test_windows_terminal_over_ssh_gets_live(clean_env):
    clean_env.setenv("SSH_TTY", "/dev/pts/0")
    clean_env.setenv("WT_SESSION", "abc-123")
    assert render.auto_stream_mode({}) == "live"


def test_apple_terminal_gets_commit(clean_env):
    clean_env.setattr(platform, "system", lambda: "Darwin")
    clean_env.setenv("TERM_PROGRAM", "Apple_Terminal")
    assert render.auto_stream_mode({}) == "commit"


def test_iterm_on_macos_gets_live(clean_env):
    clean_env.setattr(platform, "system", lambda: "Darwin")
    clean_env.setenv("TERM_PROGRAM", "iTerm.app")
    assert render.auto_stream_mode({}) == "live"


# ── _safe_commit_point ──────────────────────────────────────────────────────

def test_commit_point_no_complete_block():
    text = "still typing the first paragraph"
    assert render._safe_commit_point(text, 0) == 0


def test_commit_point_commits_completed_paragraph():
    text = "first paragraph\n\nsecond, in progress"
    # Boundary is just after the "\n\n".
    assert render._safe_commit_point(text, 0) == len("first paragraph\n\n")


def test_commit_point_does_not_split_open_code_fence():
    # A blank line INSIDE an unclosed ``` fence must not be a commit point.
    text = "intro\n\n```python\ncode line\n\nmore code"
    assert render._safe_commit_point(text, 0) == len("intro\n\n")


def test_commit_point_commits_after_fence_closes():
    text = "intro\n\n```python\ncode\n\nmore\n```\n\nafter"
    point = render._safe_commit_point(text, 0)
    # Everything up to and including the blank line after the closing fence.
    assert text[:point].endswith("```\n\n")
    assert "after" not in text[:point]


# ── commit-mode streaming ───────────────────────────────────────────────────

@pytest.fixture
def commit_mode(monkeypatch):
    fake = _Console(is_terminal=True, height=40)   # even on a TTY, commit is append-only
    monkeypatch.setattr(render, "_RICH", True)
    monkeypatch.setattr(render, "console", fake)
    monkeypatch.setattr(render, "_STREAM_MODE", "commit")
    monkeypatch.setattr(render, "_make_renderable", lambda text: text)
    monkeypatch.setattr(render, "_accumulated_text", [])
    monkeypatch.setattr(render, "_commit_idx", 0)
    return fake


def test_commit_mode_commits_blocks_appendonly(commit_mode, capsys):
    render.stream_text("# Title\n\n")        # completes a block → committed
    render.stream_text("body still going")   # incomplete → buffered, no commit
    assert commit_mode.printed == ["# Title"]

    render.flush_response()
    assert commit_mode.printed == ["# Title", "body still going"]
    assert render._commit_idx == 0           # state reset after flush


def test_commit_mode_emits_no_cursor_sequences(commit_mode, capsys):
    """Regression: commit mode must NEVER issue cursor-up / erase ANSI, even on a
    TTY — that was the source of duplicated frames over SSH / with CJK text."""
    for chunk in ["第一段，正在", "输入中的内容", "\n\n", "第二段也在写", "更多文字"]:
        render.stream_text(chunk)
    render.flush_response()
    out = capsys.readouterr().out
    assert "\x1b[" not in out                 # no cursor control of any kind
    # Each block rendered exactly once → no duplication.
    assert commit_mode.printed == ["第一段，正在输入中的内容", "第二段也在写更多文字"]


def test_commit_mode_streaming_chunks_commit_each_block_once(commit_mode):
    """A long block streamed token-by-token commits exactly once when it closes
    (not re-emitted on every chunk)."""
    text = "这是一个很长的段落" * 20 + "\n\n尾巴"
    for ch in text:                           # one char at a time, like a real stream
        render.stream_text(ch)
    render.flush_response()
    assert commit_mode.printed == ["这是一个很长的段落" * 20, "尾巴"]


def test_commit_mode_renders_full_fenced_block_atomically(commit_mode):
    for chunk in ["```py\n", "x = 1\n", "\n", "y = 2\n", "```\n\n", "done"]:
        render.stream_text(chunk)
    render.flush_response()
    # The whole code fence is one committed block; "done" is the trailing block.
    assert commit_mode.printed[0].startswith("```py")
    assert commit_mode.printed[0].rstrip().endswith("```")
    assert commit_mode.printed[-1] == "done"
