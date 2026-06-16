import pytest
import cheetahclaws.ui.render as render


class _FakeLive:
    def __init__(self, *args, **kwargs):
        self.started = False
        self.stopped = False
        self.updates = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def update(self, value, refresh=False):
        self.updates.append((value, refresh))


class _FakeConsole:
    height = 100
    width = 80
    options = object()

    def __init__(self):
        self.printed = []

    def print(self, value):
        self.printed.append(value)


@pytest.fixture
def patched_render(monkeypatch):
    previous_state = (
        list(render._accumulated_text),
        render._current_live,
        render._plain_streaming_response,
        render._live_shows_full,
    )
    fake_console = _FakeConsole()
    live_instances = []

    def fake_live(*args, **kwargs):
        live = _FakeLive(*args, **kwargs)
        live_instances.append(live)
        return live

    monkeypatch.setattr(render, "_RICH", True)
    monkeypatch.setattr(render, "_RICH_LIVE", True)
    monkeypatch.setattr(render, "console", fake_console)
    monkeypatch.setattr(render, "Live", fake_live)
    monkeypatch.setattr(render, "_make_renderable", lambda text: text)
    monkeypatch.setattr(render, "_live_line_limit", lambda: 80)
    # Identifiable tail renderable so tests can assert which lines were shown.
    monkeypatch.setattr(render, "_lines_renderable", lambda lines: ("TAIL", list(lines)))

    render._accumulated_text.clear()
    render._current_live = None
    render._plain_streaming_response = False
    render._live_shows_full = False

    yield fake_console, live_instances

    render._accumulated_text[:] = previous_state[0]
    render._current_live = previous_state[1]
    render._plain_streaming_response = previous_state[2]
    render._live_shows_full = previous_state[3]


def test_short_response_streams_full_in_one_live_and_freezes_on_flush(
    patched_render, monkeypatch, capsys
):
    """A response that fits the viewport shows the full text live; flush freezes
    it in place without re-printing (no flicker, no duplicate)."""
    fake_console, live_instances = patched_render
    # Cheap estimate well under the limit → fast path, precise render never used.
    monkeypatch.setattr(render, "_cheap_line_estimate", lambda _: 5)
    render_calls = []
    monkeypatch.setattr(render, "_render_to_lines", lambda r: render_calls.append(r))

    render.stream_text("first")
    render.stream_text(" second")

    assert render_calls == []  # fast path never rendered precisely
    assert capsys.readouterr().out == ""
    assert fake_console.printed == []
    assert len(live_instances) == 1
    live = live_instances[0]
    assert live.started is True
    assert live.stopped is False
    assert live.updates == [("first", True), ("first second", True)]
    assert render._live_shows_full is True
    assert render._accumulated_text == ["first", " second"]
    assert render._plain_streaming_response is False

    render.flush_response()

    assert live.stopped is True
    assert live.updates == [("first", True), ("first second", True)]  # no clear
    assert fake_console.printed == []  # full frame already on screen, no re-print
    assert render._accumulated_text == []
    assert render._current_live is None
    assert render._live_shows_full is False


def test_long_response_uses_tail_window_then_commits_full_on_flush(
    patched_render, monkeypatch
):
    """Once the response would overflow, Live only shows the last `limit` rendered
    lines; flush clears that window and commits the complete output once."""
    fake_console, live_instances = patched_render
    # Force the precise-render path (cheap estimate over the gate).
    monkeypatch.setattr(render, "_cheap_line_estimate", lambda _: 100)
    # Precise render reports 90 lines (> limit 80) → tail window of the last 80.
    monkeypatch.setattr(render, "_render_to_lines", lambda _r: list(range(90)))

    render.stream_text("x")

    assert len(live_instances) == 1
    live = live_instances[0]
    assert live.started is True
    assert live.updates == [(("TAIL", list(range(10, 90))), True)]
    assert render._live_shows_full is False
    assert render._plain_streaming_response is False

    render.flush_response()

    # Tail window cleared (update "" then stop), full output committed once.
    assert live.updates[-1] == ("", True)
    assert live.stopped is True
    assert fake_console.printed == ["x"]
    assert render._accumulated_text == []
    assert render._current_live is None
    assert render._live_shows_full is False
    assert render._plain_streaming_response is False


def test_transition_from_full_to_tail_window(patched_render, monkeypatch):
    """A response that starts within the limit and grows past it switches from a
    full frame to a tail window mid-stream, staying on one Live renderer."""
    fake_console, live_instances = patched_render
    monkeypatch.setattr(render, "_cheap_line_estimate", lambda _: 100)  # force precise
    rendered = iter([list(range(10)), list(range(90))])  # fits, then overflows
    monkeypatch.setattr(render, "_render_to_lines", lambda _r: next(rendered))

    render.stream_text("a")    # 10 lines → full frame
    render.stream_text("b")    # 90 lines → tail window

    assert len(live_instances) == 1
    live = live_instances[0]
    assert live.updates == [
        ("ab"[:1], True),                      # full frame shows the rendered text "a"
        (("TAIL", list(range(10, 90))), True),  # tail window of last 80 lines
    ]
    assert render._live_shows_full is False

    render.flush_response()

    assert live.updates[-1] == ("", True)       # tail window cleared
    assert fake_console.printed == ["ab"]       # full output committed
    assert render._live_shows_full is False


def test_falls_back_to_plain_when_precise_render_fails(patched_render, monkeypatch, capsys):
    """If the precise render can't produce lines, this response degrades to plain
    streaming (the safety net) instead of risking a bad Live frame."""
    fake_console, live_instances = patched_render
    monkeypatch.setattr(render, "_cheap_line_estimate", lambda _: 100)  # force precise
    monkeypatch.setattr(render, "_render_to_lines", lambda _r: None)    # render failed

    render.stream_text("first")
    render.stream_text(" second")

    assert live_instances == []
    assert fake_console.printed == ["first"]
    assert capsys.readouterr().out == " second"
    assert render._accumulated_text == []
    assert render._current_live is None
    assert render._plain_streaming_response is True

    render.flush_response()

    assert capsys.readouterr().out == "\n"
    assert render._plain_streaming_response is False


def test_falls_back_to_plain_when_terminal_too_small_to_bound_window(
    patched_render, monkeypatch
):
    """When the viewport is smaller than the live limit, no tail window can be
    bounded safely → plain streaming."""
    fake_console, live_instances = patched_render
    fake_console.height = 10  # limit (80) > height → cannot bound a Live window

    render.stream_text("first")

    assert live_instances == []
    assert fake_console.printed == ["first"]
    assert render._accumulated_text == []
    assert render._plain_streaming_response is True


def test_lines_renderable_joins_segment_lines_without_trailing_break():
    """Real _lines_renderable (no mocks) wraps rendered segment-lines into a
    Segments renderable, inserting a line break between lines but not after the
    last one. Guards against Rich's segment API drifting under us."""
    from rich.segment import Segment, Segments

    lines = [[Segment("alpha")], [Segment("beta"), Segment("!")]]
    result = render._lines_renderable(lines)

    assert isinstance(result, Segments)
    assert [seg.text for seg in result.segments] == ["alpha", "\n", "beta", "!"]


def test_real_tail_window_end_to_end_commits_full_output(monkeypatch):
    """Integration: drive the real Live + Segments tail-window path with nothing
    mocked, and confirm flush commits the complete output without error (the head
    that scrolled out of the tail window is recovered on flush)."""
    import io
    from rich.console import Console

    buf = io.StringIO()
    con = Console(file=buf, force_terminal=True, width=60, height=18)
    monkeypatch.setattr(render, "console", con)
    monkeypatch.setattr(render, "_RICH", True)
    monkeypatch.setattr(render, "_RICH_LIVE", True)
    monkeypatch.setattr(render, "_current_live", None)
    monkeypatch.setattr(render, "_plain_streaming_response", False)
    monkeypatch.setattr(render, "_live_shows_full", False)
    monkeypatch.setattr(render, "_accumulated_text", [])

    # 40 list items at height 18 → far past the limit → tail-window path.
    body = "# Title\n\n" + "".join(f"- entry {i} short\n" for i in range(40))
    for j in range(0, len(body), 32):
        render.stream_text(body[j:j + 32])
    render.flush_response()

    out = buf.getvalue()
    assert "entry 0 short" in out       # first item (scrolled out of tail) committed
    assert "entry 39 short" in out      # last item committed
    assert render._current_live is None
    assert render._plain_streaming_response is False
    assert render._live_shows_full is False
    assert render._accumulated_text == []
