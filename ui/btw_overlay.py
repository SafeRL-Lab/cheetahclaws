"""ui/btw_overlay.py — /btw Rich overlay: side-session Q&A while agent runs.

Opens a three-column layout:
  left   — dim padding (agent output scrolls behind via patch_stdout)
  center — focused white-on-dark panel; streams the answer
  right  — dim padding (mirror)

The question is answered by a fresh no-tools API call so it never
touches the main conversation state.
"""
from __future__ import annotations

import sys
from typing import Iterator


def _side_stream(question: str, model: str, config: dict) -> Iterator[str]:
    """Stream a quick no-tools answer for the /btw question."""
    from providers import stream as _stream, AssistantTurn, TextChunk
    messages = [{"role": "user", "content": question}]
    system = "Answer concisely and helpfully. No markdown headers needed."
    for event in _stream(
        model=model,
        system=system,
        messages=messages,
        tool_schemas=[],
        config=config,
    ):
        if isinstance(event, TextChunk):
            yield event.text


def run_btw_overlay(question: str, model: str, config: dict) -> None:
    """Render the /btw overlay and stream an answer into it."""
    try:
        from rich.console import Console
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel
        from rich.markdown import Markdown
        from rich.text import Text
    except ImportError:
        # Rich not available — plain fallback
        print(f"\n\033[35m[/btw]\033[0m {question}")
        for chunk in _side_stream(question, model, config):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()
        return

    console = Console()
    layout = Layout()
    layout.split_row(
        Layout(name="left",   ratio=1),
        Layout(name="center", ratio=3),
        Layout(name="right",  ratio=1),
    )
    dim_panel = Panel(Text(""), border_style="dim")
    layout["left"].update(dim_panel)
    layout["right"].update(dim_panel)
    layout["center"].update(
        Panel(Text("…", style="dim"), title="[cyan]/btw[/cyan]", border_style="cyan")
    )

    buf: list[str] = []

    def _render_center():
        text = "".join(buf)
        try:
            renderable = Markdown(text) if any(c in text for c in ("#", "*", "`", "_", "[")) else text
        except Exception:
            renderable = text
        layout["center"].update(
            Panel(renderable, title="[cyan]/btw[/cyan]", border_style="cyan")
        )

    with Live(layout, console=console, auto_refresh=True, refresh_per_second=12,
              vertical_overflow="visible"):
        try:
            for chunk in _side_stream(question, model, config):
                buf.append(chunk)
                _render_center()
        except KeyboardInterrupt:
            pass

    # Leave a static copy so the user can scroll back
    final = "".join(buf).strip()
    if final:
        console.print(
            Panel(Markdown(final) if any(c in final for c in ("#", "*", "`", "_", "[")) else final,
                  title="[cyan]/btw[/cyan]", border_style="dim")
        )
