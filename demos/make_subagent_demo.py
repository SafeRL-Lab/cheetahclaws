#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws sub-agents (parallel multi-agent) using PIL.
Simulates: user asks to review a PR → AI spawns coder agent + reviewer agent in parallel
→ agents work simultaneously → results merged → summary presented.
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
ORANGE  = (254, 100,  11)

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


def text_line(t, indent=2, color=TEXT):
    return [seg(" " * indent + t, color)]


def agent_header(name, color, role):
    return [
        seg("  ┌─ ", SUBTEXT),
        seg(f"Agent: {name}", color, True),
        seg(f"  [{role}]", SUBTEXT),
        seg(" ─────────────────────────────", SUBTEXT),
    ]


def agent_footer():
    return [seg("  └─────────────────────────────────────────────────────────", SUBTEXT)]


def agent_line(t, color=TEXT):
    return [seg("  │  ", SUBTEXT), seg(t, color)]


def agent_tool(icon, name, arg, color=CYAN):
    return [
        seg("  │  ", SUBTEXT),
        seg(f"{icon}  ", SUBTEXT),
        seg(name, color),
        seg(f"({arg})", SUBTEXT),
    ]


def agent_ok(msg):
    return [seg("  │  ✓ ", GREEN), seg(msg, SUBTEXT)]


def spawn_line(name, role, color):
    return [
        seg("  ⟳  ", CYAN, True),
        seg("Spawning ", SUBTEXT),
        seg(name, color, True),
        seg(f"  ({role})", SUBTEXT),
    ]


def parallel_bar(agents_running):
    parts = [seg("  ◈  Running in parallel: ", SUBTEXT)]
    for i, (name, color) in enumerate(agents_running):
        if i > 0:
            parts.append(seg("  +  ", SUBTEXT))
        parts.append(seg(name, color, True))
    return parts


def merge_line(msg):
    return [seg("  ⊕  ", MAUVE, True), seg(msg, TEXT)]


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms))

    def type_it(base, text, ms_char=60, ms_hold=400):
        for i in range(0, len(text) + 1, 3):
            add(base + [prompt_line(text[:i], cursor=(i < len(text)))], ms_char)
        add(base + [prompt_line(text)], ms_hold)

    # ── 0: Banner + empty prompt ─────────────────────────────────────────
    add(BANNER + [prompt_line(cursor=True)], 900)

    # ── 1: User asks to review PR ─────────────────────────────────────────
    task = "review the new auth PR — check correctness AND security"
    type_it(BANNER, task)

    base0 = BANNER + [prompt_line(task), None, claude_header()]

    # ── 2: AI decides to spawn sub-agents ────────────────────────────────
    add(base0 + [
        text_line("This task benefits from parallel analysis.", 2, SUBTEXT),
        text_line("Spawning 2 sub-agents…", 2, SUBTEXT),
        None,
        spawn_line("coder-agent",    "implementation review", GREEN),
        spawn_line("security-agent", "security audit",        RED),
    ], 1000)

    # ── 3: Both agents running in parallel ────────────────────────────────
    parallel_header = [
        None,
        parallel_bar([("coder-agent", GREEN), ("security-agent", RED)]),
        None,
    ]

    add(base0 + parallel_header + [
        agent_header("coder-agent", GREEN, "implementation review"),
        agent_tool("⚙", "Read", "src/auth/auth.py", GREEN),
        agent_footer(),
        None,
        agent_header("security-agent", RED, "security audit"),
        agent_tool("⚙", "Grep", "pattern=password|token|secret", RED),
        agent_footer(),
    ], 700)

    add(base0 + parallel_header + [
        agent_header("coder-agent", GREEN, "implementation review"),
        agent_tool("⚙", "Read",  "src/auth/auth.py", GREEN),
        agent_ok("→ 143 lines"),
        agent_tool("⚙", "Read",  "src/auth/middleware.py", GREEN),
        agent_ok("→ 67 lines"),
        agent_footer(),
        None,
        agent_header("security-agent", RED, "security audit"),
        agent_tool("⚙", "Grep", "pattern=password|token|secret", RED),
        agent_ok("→ 9 matches"),
        agent_tool("⚙", "Read",  "src/auth/auth.py", RED),
        agent_footer(),
    ], 700)

    # ── 4: Agents complete ────────────────────────────────────────────────
    add(base0 + parallel_header + [
        agent_header("coder-agent", GREEN, "implementation review"),
        agent_line("✓  JWT signing logic looks correct", GREEN),
        agent_line("✓  Token refresh flow is well structured", GREEN),
        agent_line("⚠  Missing unit tests for edge cases", YELLOW),
        agent_footer(),
        None,
        agent_header("security-agent", RED, "security audit"),
        agent_line("✓  HS256 algorithm is acceptable", GREEN),
        agent_line("✗  Token stored in localStorage — XSS risk", RED),
        agent_line("✗  No rate limiting on /auth endpoints", RED),
        agent_footer(),
    ], 1100)

    # ── 5: Merge + summary ────────────────────────────────────────────────
    add(base0 + [
        None,
        merge_line("Merging results from 2 agents…"),
        None,
        [seg("│ ", SUBTEXT)],
        text_line("PR Review — auth feature branch:", 2, CYAN),
        None,
        text_line("Implementation (coder-agent):", 2, SUBTEXT),
        text_line("  ✓  JWT logic correct, refresh flow solid", 2, GREEN),
        text_line("  ⚠  Add unit tests for edge cases", 2, YELLOW),
        None,
        text_line("Security (security-agent):", 2, SUBTEXT),
        text_line("  ✗  Move token to httpOnly cookie (XSS risk)", 2, RED),
        text_line("  ✗  Add rate limiting on /auth/login", 2, RED),
        None,
        text_line("Verdict: 2 blockers before merge.", 2, YELLOW),
        claude_sep(),
    ], 1000)

    add(base0 + [
        None,
        merge_line("Merging results from 2 agents…"),
        None,
        [seg("│ ", SUBTEXT)],
        text_line("PR Review — auth feature branch:", 2, CYAN),
        None,
        text_line("Implementation (coder-agent):", 2, SUBTEXT),
        text_line("  ✓  JWT logic correct, refresh flow solid", 2, GREEN),
        text_line("  ⚠  Add unit tests for edge cases", 2, YELLOW),
        None,
        text_line("Security (security-agent):", 2, SUBTEXT),
        text_line("  ✗  Move token to httpOnly cookie (XSS risk)", 2, RED),
        text_line("  ✗  Add rate limiting on /auth/login", 2, RED),
        None,
        text_line("Verdict: 2 blockers before merge.", 2, YELLOW),
        claude_sep(),
        None,
        prompt_line(cursor=True),
    ], 2500)

    return scenes


# ── Palette + render ──────────────────────────────────────────────────────

def _build_palette():
    theme = [
        BG, SURFACE, TEXT, SUBTEXT,
        CYAN, GREEN, YELLOW, RED, MAUVE, BLUE, PEACH, ORANGE,
        (255, 255, 255), (0, 0, 0),
        (50, 55, 80), (160, 166, 200),
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
                       "..", "docs", "subagent_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
