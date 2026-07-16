#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws tmux integration using PIL.
Simulates: user asks AI to split screen → AI calls TmuxSplitWindow +
TmuxSendKeys → runs server in one pane + tests in another → TmuxCapture
to read output → reports results.
"""
from PIL import Image, ImageDraw, ImageFont
import os

# ── Catppuccin Mocha palette ─────────────────────────────────────────────
BG      = (30,  30,  46)
SURFACE = (49,  50,  68)
TEXT    = (205, 214, 244)
SUBTEXT = (108, 112, 134)
CYAN    = (137, 220, 235)
GREEN   = (166, 227, 161)
YELLOW  = (249, 226, 175)
RED     = (243, 139, 168)
MAUVE   = (203, 166, 247)
BLUE    = (137, 180, 250)
PEACH   = (250, 179, 135)

W, H = 960, 720
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_SIZE = 14
LINE_H    = 20
PAD_X     = 18
PAD_Y     = 16


def make_font(size=FONT_SIZE, bold=False):
    path = FONT_BOLD if bold else FONT_PATH
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


FONT   = make_font()
FONT_B = make_font(bold=True)


def seg(t, c=TEXT, b=False):
    return (t, c, b)


def render_line(draw, y, segments, x_start=PAD_X):
    x = x_start
    for text, color, bold in segments:
        font = FONT_B if bold else FONT
        draw.text((x, y), text, font=font, fill=color)
        x += font.getlength(text)
    return y + LINE_H


def blank_frame():
    return Image.new("RGB", (W, H), BG)


def draw_frame(lines_segments):
    img = blank_frame()
    d   = ImageDraw.Draw(img)
    y   = PAD_Y
    for item in lines_segments:
        if item is None:
            y += LINE_H
        elif isinstance(item, list):
            y = render_line(d, y, item)
        else:
            y = render_line(d, y, [item])
    return img


# ── Reusable line builders ────────────────────────────────────────────────

BANNER = [
    [seg("╭─ CheetahClaws v3.5.5 ──────────────────────────────────╮", SUBTEXT)],
    [seg("│  ", SUBTEXT), seg("Model: ", SUBTEXT), seg("claude-sonnet-4-6", CYAN, True)],
    [seg("│  ", SUBTEXT), seg("Permissions: ", SUBTEXT), seg("auto", YELLOW)],
    [seg("│  Type /help for commands, Ctrl+C to cancel                  │", SUBTEXT)],
    [seg("╰────────────────────────────────────────────────────────────╯", SUBTEXT)],
    None,
]


def prompt_line(text="", cursor=False):
    cur = "█" if cursor else ""
    return [
        seg("[cheetahclaws] ", SUBTEXT),
        seg("» ", CYAN, True),
        seg(text + cur, TEXT),
    ]


def ok_line(msg):
    return [seg("✓  ", GREEN, True), seg(msg, TEXT)]


def info_line(msg):
    return [seg("  ", SUBTEXT), seg(msg, SUBTEXT)]


def claude_header():
    return [
        seg("╭─ CheetahClaws ", SUBTEXT),
        seg("●", GREEN),
        seg(" ─────────────────────────────────────────────", SUBTEXT),
    ]


def claude_sep():
    return [seg("╰──────────────────────────────────────────────────────────", SUBTEXT)]


def text_line(t, indent=2):
    return [seg(" " * indent + t, TEXT)]


def tool_line(icon, name, arg):
    return [
        seg(f"  {icon}  ", SUBTEXT),
        seg(name, CYAN),
        seg("(", SUBTEXT),
        seg(arg, TEXT),
        seg(")", SUBTEXT),
    ]


def tool_ok(msg):
    return [seg("  ✓ ", GREEN), seg(msg, SUBTEXT)]


def pane_box(title, lines, color=SUBTEXT, active=False):
    """Render a mini tmux pane box."""
    border = CYAN if active else SUBTEXT
    header = [seg(f"  ┌─ {title} ", border),
              seg("─" * max(1, 44 - len(title)), border),
              (seg("(active)", CYAN, True) if active else seg("", SUBTEXT)),
              seg("┐", border)]
    rows = []
    for l in lines:
        padded = l[:50].ljust(50)
        rows.append([seg("  │ ", border), seg(padded, color), seg("│", border)])
    footer = [seg("  └" + "─" * 52 + "┘", border)]
    return [header] + rows + [footer]


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms))

    USER_MSG = "Split screen: run uvicorn on the left, pytest on the right"

    # ── 0: Banner + empty prompt ─────────────────────────────────────────
    add(BANNER + [prompt_line(cursor=True)], 900)

    # ── 1: Type user message ──────────────────────────────────────────────
    for i in range(0, len(USER_MSG) + 1, 4):
        add(BANNER + [prompt_line(USER_MSG[:i], cursor=(i < len(USER_MSG)))], 55)
    add(BANNER + [prompt_line(USER_MSG)], 500)

    base0 = BANNER + [prompt_line(USER_MSG), None, claude_header()]

    # ── 2: TmuxListSessions ───────────────────────────────────────────────
    add(base0 + [tool_line("⊞", "TmuxListSessions", "")], 500)
    add(base0 + [
        tool_line("⊞", "TmuxListSessions", ""),
        tool_ok("→ main: 1 windows (created ...) [220x50]"),
    ], 700)

    # ── 3: TmuxSplitWindow ────────────────────────────────────────────────
    add(base0 + [
        tool_line("⊞", "TmuxListSessions", ""),
        tool_ok("→ main: 1 windows (created ...) [220x50]"),
        tool_line("⊞", "TmuxSplitWindow", "target=main:0  direction=horizontal"),
    ], 500)
    add(base0 + [
        tool_line("⊞", "TmuxListSessions", ""),
        tool_ok("→ main: 1 windows (created ...) [220x50]"),
        tool_line("⊞", "TmuxSplitWindow", "target=main:0  direction=horizontal"),
        tool_ok("→ (ok)  pane 0 and pane 1 now visible"),
    ], 700)

    # ── 4: TmuxSendKeys to pane 0 (uvicorn) ──────────────────────────────
    add(base0 + [
        tool_line("⊞", "TmuxListSessions", ""),
        tool_ok("→ main: 1 windows (created ...) [220x50]"),
        tool_line("⊞", "TmuxSplitWindow", "target=main:0  direction=horizontal"),
        tool_ok("→ (ok)  pane 0 and pane 1 now visible"),
        tool_line("⌨", "TmuxSendKeys", "target=main:0.0  keys=uvicorn main:app --reload"),
    ], 500)
    add(base0 + [
        tool_line("⊞", "TmuxListSessions", ""),
        tool_ok("→ main: 1 windows (created ...) [220x50]"),
        tool_line("⊞", "TmuxSplitWindow", "target=main:0  direction=horizontal"),
        tool_ok("→ (ok)  pane 0 and pane 1 now visible"),
        tool_line("⌨", "TmuxSendKeys", "target=main:0.0  keys=uvicorn main:app --reload"),
        tool_ok("→ (ok)"),
    ], 500)

    # ── 5: TmuxSendKeys to pane 1 (pytest) ───────────────────────────────
    add(base0 + [
        tool_line("⊞", "TmuxListSessions", ""),
        tool_ok("→ main: 1 windows (created ...) [220x50]"),
        tool_line("⊞", "TmuxSplitWindow", "target=main:0  direction=horizontal"),
        tool_ok("→ (ok)  pane 0 and pane 1 now visible"),
        tool_line("⌨", "TmuxSendKeys", "target=main:0.0  keys=uvicorn main:app --reload"),
        tool_ok("→ (ok)"),
        tool_line("⌨", "TmuxSendKeys", "target=main:0.1  keys=pytest -v --tb=short"),
    ], 500)
    add(base0 + [
        tool_line("⊞", "TmuxListSessions", ""),
        tool_ok("→ main: 1 windows (created ...) [220x50]"),
        tool_line("⊞", "TmuxSplitWindow", "target=main:0  direction=horizontal"),
        tool_ok("→ (ok)  pane 0 and pane 1 now visible"),
        tool_line("⌨", "TmuxSendKeys", "target=main:0.0  keys=uvicorn main:app --reload"),
        tool_ok("→ (ok)"),
        tool_line("⌨", "TmuxSendKeys", "target=main:0.1  keys=pytest -v --tb=short"),
        tool_ok("→ (ok)"),
    ], 800)

    # ── 6: Show pane layout ───────────────────────────────────────────────
    uvicorn_lines = [
        "INFO:     Uvicorn running on http://0.0.0.0:8000",
        "INFO:     Started reloader process [12847]",
        "INFO:     Application startup complete.",
        "INFO:     127.0.0.1 - GET /health 200 OK",
    ]
    pytest_lines = [
        "====================== test session starts ==============",
        "collected 12 items",
        "",
        "tests/test_api.py::test_health PASSED             [ 8%]",
        "tests/test_api.py::test_login  PASSED             [16%]",
    ]

    add(base0 + [
        tool_ok("→ (ok)"),
        None,
        [seg("  ── Terminal layout (both panes running) ──", SUBTEXT)],
        None,
    ] + pane_box("pane 0 · uvicorn", uvicorn_lines, GREEN, active=True)
      + [None]
      + pane_box("pane 1 · pytest",  pytest_lines,  YELLOW), 1200)

    # ── 7: TmuxCapture pane 1 after tests finish ──────────────────────────
    pytest_done = [
        "tests/test_api.py::test_auth   PASSED             [25%]",
        "tests/test_api.py::test_users  PASSED             [33%]",
        "tests/test_api.py::test_items  PASSED             [41%]",
        "...",
        "12 passed in 4.31s",
    ]

    add(base0 + [
        tool_ok("→ (ok)"),
        None,
        tool_line("📋", "TmuxCapture", "target=main:0.1  lines=20"),
    ], 600)
    add(base0 + [
        tool_ok("→ (ok)"),
        None,
        tool_line("📋", "TmuxCapture", "target=main:0.1  lines=20"),
        tool_ok("→ captured 20 lines"),
        None,
        [seg("  ── Captured output (pane 1) ──", SUBTEXT)],
    ] + pane_box("pane 1 · pytest", pytest_done, GREEN), 1000)

    # ── 8: AI summary response ────────────────────────────────────────────
    summary = [
        "Both panes are running:",
        "",
        "  Left  (pane 0): uvicorn serving on :8000, reloader active",
        "  Right (pane 1): pytest finished — 12/12 passed in 4.31s ✓",
        "",
        "All tests green. The server is live and watching for changes.",
        "Use TmuxCapture anytime to check new output.",
    ]

    add(BANNER + [prompt_line(USER_MSG), None, claude_header()] + [
        [seg("│ ", SUBTEXT)],
    ] + [text_line(l, 2) for l in summary] + [claude_sep()], 1000)

    # ── 9: New prompt ─────────────────────────────────────────────────────
    add(BANNER + [prompt_line(USER_MSG), None, claude_header()] + [
        [seg("│ ", SUBTEXT)],
    ] + [text_line(l, 2) for l in summary] + [claude_sep()] + [
        None, prompt_line(cursor=True),
    ], 2500)

    return scenes


# ── Palette + render ──────────────────────────────────────────────────────

def _build_palette():
    theme = [
        BG, SURFACE, TEXT, SUBTEXT,
        CYAN, GREEN, YELLOW, RED, MAUVE, BLUE, PEACH,
        (255, 255, 255), (0, 0, 0),
        (50, 55, 80), (90, 95, 120), (160, 166, 200),
    ]
    flat = []
    for c in theme:
        flat.extend(c)
    while len(flat) < 256 * 3:
        flat.extend((0, 0, 0))
    return flat


def render_gif(output_path):
    print("Building scenes...")
    scenes = build_scenes()
    print(f"  {len(scenes)} scenes")

    pal_ref = Image.new("P", (1, 1))
    pal_ref.putpalette(_build_palette())

    print("  Rendering frames...")
    rgb_frames, durations = [], []
    for i, (lines, ms) in enumerate(scenes):
        rgb_frames.append(draw_frame(lines))
        durations.append(ms)
        if i % 20 == 0:
            print(f"  {i}/{len(scenes)}...")

    print("  Quantizing...")
    p_frames = [f.quantize(palette=pal_ref, dither=0) for f in rgb_frames]

    print(f"Saving → {output_path}  ({len(p_frames)} frames)")
    p_frames[0].save(
        output_path,
        save_all=True,
        append_images=p_frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    size_kb = os.path.getsize(output_path) // 1024
    print(f"Done! {size_kb} KB")


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "docs", "tmux_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
