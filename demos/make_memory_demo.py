#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws Memory system using PIL.
Simulates: session 1 → user tells AI preferences + project context → AI saves memory
→ session 2 (new day) → AI recalls context automatically → continues seamlessly.
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

def banner(session_label, date_str):
    return [
        [seg("╭─ CheetahClaws v3.5.5 ──────────────────────────────────╮", SUBTEXT)],
        [seg("│  ", SUBTEXT), seg("Model: ", SUBTEXT), seg("claude-sonnet-4-6", CYAN, True),
         seg(f"   {session_label}", SUBTEXT)],
        [seg("│  ", SUBTEXT), seg("Session: ", SUBTEXT), seg(date_str, YELLOW)],
        [seg("│  Type /help for commands, Ctrl+C to cancel                  │", SUBTEXT)],
        [seg("╰────────────────────────────────────────────────────────────╯", SUBTEXT)],
        None,
    ]


BANNER1 = banner("Session 1", "Mon Apr 07")
BANNER2 = banner("Session 2", "Tue Apr 08  — new session")


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


def text_line(t, indent=2, color=TEXT):
    return [seg(" " * indent + t, color)]


def mem_save_line(key, value, color=MAUVE):
    return [
        seg("  💾  ", MAUVE),
        seg("memory: ", SUBTEXT),
        seg(f"{key} ", color, True),
        seg("→ ", SUBTEXT),
        seg(value, TEXT),
    ]


def mem_load_line(key, value):
    return [
        seg("  🧠  ", CYAN),
        seg(f"{key}: ", SUBTEXT),
        seg(value, TEXT),
    ]


def divider(label):
    pad = "─" * max(0, 54 - len(label))
    return [seg(f"  ── {label} {pad}", SUBTEXT)]


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms))

    def type_it(base, text, ms_char=60, ms_hold=400):
        for i in range(0, len(text) + 1, 3):
            add(base + [prompt_line(text[:i], cursor=(i < len(text)))], ms_char)
        add(base + [prompt_line(text)], ms_hold)

    # ════════════════════════════════════════════════════════════════════
    # SESSION 1
    # ════════════════════════════════════════════════════════════════════
    add(BANNER1 + [
        divider("Session 1  —  first time using CheetahClaws"),
        None,
        prompt_line(cursor=True),
    ], 900)

    base1 = BANNER1 + [divider("Session 1  —  first time using CheetahClaws"), None]

    # User sets preferences
    q1 = "I prefer pytest over unittest, and always use type hints"
    type_it(base1, q1)

    add(BANNER1 + [
        divider("Session 1  —  first time using CheetahClaws"),
        None,
        prompt_line(q1), None, claude_header(),
        text_line("Got it — I'll default to pytest and add type hints", 2),
        text_line("to all code I write for you.", 2),
        claude_sep(),
        None,
        mem_save_line("test_framework", "pytest"),
        mem_save_line("style",          "always use type hints"),
        None,
        prompt_line(cursor=True),
    ], 1000)

    # User shares project context
    after1 = BANNER1 + [
        divider("Session 1  —  first time using CheetahClaws"),
        None,
        prompt_line(q1),
        mem_save_line("test_framework", "pytest"),
        mem_save_line("style",          "always use type hints"),
        None,
    ]

    q2 = "this is a FastAPI service, Postgres backend, deployed on GCP"
    type_it(after1, q2)

    add(BANNER1 + [
        divider("Session 1  —  first time using CheetahClaws"),
        None,
        prompt_line(q1),
        mem_save_line("test_framework", "pytest"),
        mem_save_line("style",          "always use type hints"),
        None,
        prompt_line(q2), None, claude_header(),
        text_line("Project context saved. I'll keep GCP deployment", 2),
        text_line("considerations in mind for infra-related tasks.", 2),
        claude_sep(),
        None,
        mem_save_line("stack",      "FastAPI + Postgres"),
        mem_save_line("deploy",     "GCP"),
        None,
        ok_line("4 memories saved  —  will persist across sessions"),
        None,
        prompt_line(cursor=True),
    ], 1100)

    # ════════════════════════════════════════════════════════════════════
    # SESSION 2  (next day, brand-new session)
    # ════════════════════════════════════════════════════════════════════
    add(BANNER2 + [
        divider("Session 2  —  starting fresh the next day"),
        None,
        info_line("Loading memories…"),
        None,
        mem_load_line("test_framework", "pytest"),
        mem_load_line("style",          "always use type hints"),
        mem_load_line("stack",          "FastAPI + Postgres"),
        mem_load_line("deploy",         "GCP"),
        None,
        ok_line("4 memories restored  —  context ready"),
        None,
        prompt_line(cursor=True),
    ], 1200)

    base2 = BANNER2 + [
        divider("Session 2  —  starting fresh the next day"),
        None,
        ok_line("4 memories restored  —  context ready"),
        None,
    ]

    # User asks something without re-explaining
    q3 = "write a test for the new /users endpoint"
    type_it(base2, q3)

    test_resp = [
        "# tests/test_users.py  (pytest, type hints — as you prefer)",
        "import pytest",
        "from httpx import AsyncClient",
        "from app.main import app",
        "",
        "async def test_list_users(client: AsyncClient) -> None:",
        '    resp = await client.get("/users")',
        "    assert resp.status_code == 200",
        '    assert isinstance(resp.json(), list)',
    ]

    add(BANNER2 + [
        divider("Session 2  —  starting fresh the next day"),
        None,
        ok_line("4 memories restored  —  context ready"),
        None,
        prompt_line(q3), None, claude_header(),
    ] + [text_line(l, 2, SUBTEXT if l.startswith("import") or l.startswith("from") or
                         l.startswith("#") else TEXT)
         for l in test_resp] + [
        claude_sep(),
        None,
        info_line("Used memory: pytest · type hints · FastAPI — no re-explanation needed."),
        None,
        prompt_line(cursor=True),
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
                       "..", "docs", "memory_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
