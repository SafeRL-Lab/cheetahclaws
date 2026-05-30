"""Tests for thinking_parser module."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from thinking_parser import ThinkingStreamParser, LoopDetector, strip_thinking_tags, OPEN_TAG, CLOSE_TAG

# Build test strings using the tag constants to avoid XML transport issues
def _wrap(inner):
    return OPEN_TAG + inner + CLOSE_TAG


class TestThinkingStreamParser:
    def test_no_thinking_tags(self):
        p = ThinkingStreamParser()
        events = p.feed("Hello world")
        events += p.finalize()
        texts = "".join(t for k, t in events if k == "text")
        assert texts == "Hello world"
        assert not any(k == "thinking" for k, _ in events)

    def test_basic_thinking_block(self):
        p = ThinkingStreamParser()
        events = p.feed(_wrap("reasoning") + "answer")
        events += p.finalize()
        thinking = "".join(t for k, t in events if k == "thinking")
        text = "".join(t for k, t in events if k == "text")
        assert thinking == "reasoning"
        assert text == "answer"

    def test_multiple_blocks(self):
        p = ThinkingStreamParser()
        events = p.feed(_wrap("A") + "X" + _wrap("B") + "Y")
        events += p.finalize()
        thinking = "".join(t for k, t in events if k == "thinking")
        text = "".join(t for k, t in events if k == "text")
        assert thinking == "AB"
        assert text == "XY"

    def test_partial_open_tag_across_chunks(self):
        p = ThinkingStreamParser()
        all_events = []
        # Split the open tag across two chunks
        all_events += p.feed("before" + OPEN_TAG[:4])
        all_events += p.feed(OPEN_TAG[4:] + "inside" + CLOSE_TAG + "after")
        all_events += p.finalize()
        thinking = "".join(t for k, t in all_events if k == "thinking")
        text = "".join(t for k, t in all_events if k == "text")
        assert thinking == "inside"
        assert text == "beforeafter"

    def test_partial_close_tag_across_chunks(self):
        p = ThinkingStreamParser()
        all_events = []
        all_events += p.feed(OPEN_TAG + "inside" + CLOSE_TAG[:6])
        all_events += p.feed(CLOSE_TAG[6:] + "outside")
        all_events += p.finalize()
        thinking = "".join(t for k, t in all_events if k == "thinking")
        text = "".join(t for k, t in all_events if k == "text")
        assert thinking == "inside"
        assert text == "outside"

    def test_unclosed_tag_flushed_on_finalize(self):
        p = ThinkingStreamParser()
        events = p.feed(OPEN_TAG + "unclosed")
        events += p.finalize()
        thinking = "".join(t for k, t in events if k == "thinking")
        assert thinking == "unclosed"

    def test_text_before_thinking(self):
        p = ThinkingStreamParser()
        events = p.feed("before" + _wrap("during") + "after")
        events += p.finalize()
        assert events == [
            ("text", "before"),
            ("thinking", "during"),
            ("text", "after"),
        ]

    def test_empty_thinking_block(self):
        p = ThinkingStreamParser()
        events = p.feed(_wrap("") + "text")
        events += p.finalize()
        text = "".join(t for k, t in events if k == "text")
        assert text == "text"

    def test_character_by_character(self):
        p = ThinkingStreamParser()
        full = _wrap("abc") + "xyz"
        events = []
        for ch in full:
            events += p.feed(ch)
        events += p.finalize()
        thinking = "".join(t for k, t in events if k == "thinking")
        text = "".join(t for k, t in events if k == "text")
        assert thinking == "abc"
        assert text == "xyz"

    def test_newlines_preserved(self):
        p = ThinkingStreamParser()
        events = p.feed(_wrap("line1\nline2\n"))
        events += p.finalize()
        thinking = "".join(t for k, t in events if k == "thinking")
        assert thinking == "line1\nline2\n"

    def test_full_thinking_text_property(self):
        p = ThinkingStreamParser()
        p.feed(_wrap("part1") + "gap" + _wrap("part2"))
        assert p.full_thinking_text == "part1part2"


class TestLoopDetector:
    def test_no_repetition(self):
        d = LoopDetector()
        assert not d.feed("Normal text without any repeating patterns at all here.")

    def test_short_text(self):
        d = LoopDetector()
        assert not d.feed("abc" * 10)

    def test_long_repeating_pattern(self):
        d = LoopDetector()
        pattern = "I need to analyze this carefully now. ok " * 10
        assert d.feed(pattern)

    def test_incremental_detection(self):
        d = LoopDetector()
        pattern = "I keep repeating this same thought!! "
        for _ in range(6):
            assert not d.feed(pattern)
        for _ in range(6):
            if d.feed(pattern):
                return
        assert False, "Should have detected loop"


class TestStripThinkingTags:
    def test_basic(self):
        assert strip_thinking_tags(_wrap("r") + "answer") == "answer"

    def test_multiline(self):
        result = strip_thinking_tags(_wrap("line1\nline2\n") + "answer")
        assert result == "answer"

    def test_multiple_blocks(self):
        assert strip_thinking_tags(_wrap("a") + "X" + _wrap("b") + "Y") == "XY"

    def test_no_tags(self):
        assert strip_thinking_tags("just text") == "just text"
