#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws checkpoint/rewind using PIL.
Simulates: AI edits a file with a bug → user notices → /checkpoint list
→ /checkpoint rewind → file restored to working state.
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


def err_line(msg):
    return [seg("  Error: ", RED, True), seg(msg, RED)]


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


def ckpt_row(idx, ts, msg, files, active=False):
    col = CYAN if active else SUBTEXT
    marker = seg("  ▶ ", CYAN, True) if active else seg("    ", SUBTEXT)
    return [
        marker,
        seg(f"[{idx}]", col, active),
        seg(f"  {ts}  ", SUBTEXT),
        seg(msg.ljust(32), TEXT if active else SUBTEXT),
        seg(f"  {files} file(s)", SUBTEXT),
    ]


def diff_line(prefix, text, color):
    return [seg(f"  {prefix} ", color, True), seg(text, color)]


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms))

    # ── 0: Banner + empty prompt ─────────────────────────────────────────
    add(BANNER + [prompt_line(cursor=True)], 900)

    # ── 1: User asks AI to refactor ──────────────────────────────────────
    cmd = "refactor the database connection pool"
    for i in range(0, len(cmd) + 1, 4):
        add(BANNER + [prompt_line(cmd[:i], cursor=(i < len(cmd)))], 60)
    add(BANNER + [prompt_line(cmd)], 400)

    base0 = BANNER + [prompt_line(cmd), None, claude_header()]

    # ── 2: AI reads and edits ─────────────────────────────────────────────
    add(base0 + [
        tool_line("⚙", "Read", "src/db/pool.py"),
        tool_ok("→ 87 lines"),
        tool_line("✏", "Edit", "src/db/pool.py"),
        tool_ok("→ connection pool refactored  (+12 lines)"),
        tool_line("✏", "Edit", "src/db/config.py"),
        tool_ok("→ pool size config updated"),
    ], 700)

    resp1 = [
        "Refactored the connection pool:",
        "  • Switched to asyncpg for async connections",
        "  • Pool size: min=2, max=10 (was hardcoded at 5)",
        "  • Added connection health checks on acquire",
    ]
    add(base0 + [
        tool_ok("→ pool size config updated"),
        None, [seg("│ ", SUBTEXT)],
    ] + [text_line(l, 2) for l in resp1] + [claude_sep()], 800)

    # Auto-snapshot indicator
    add(base0 + [
        tool_ok("→ pool size config updated"),
        None, [seg("│ ", SUBTEXT)],
    ] + [text_line(l, 2) for l in resp1] + [claude_sep()] + [
        None,
        info_line("📸  Auto-snapshot saved  →  checkpoint #3"),
        None,
        prompt_line(cursor=True),
    ], 1000)

    # ── 3: Tests fail ─────────────────────────────────────────────────────
    after1 = BANNER + [
        prompt_line(cmd), None,
        info_line("📸  Auto-snapshot saved  →  checkpoint #3"),
        None,
    ]
    cmd2 = "!pytest tests/ -q"
    for i in range(0, len(cmd2) + 1, 3):
        add(after1 + [prompt_line(cmd2[:i], cursor=(i < len(cmd2)))], 60)
    add(after1 + [prompt_line(cmd2)], 400)

    add(after1 + [
        prompt_line(cmd2),
        [seg("  $ ", SUBTEXT), seg("pytest tests/ -q", YELLOW)],
        text_line("FAILED tests/test_db.py::test_connection - RuntimeError", 2, RED),
        text_line("FAILED tests/test_db.py::test_pool_size  - AssertionError", 2, RED),
        text_line("2 failed, 18 passed in 3.24s", 2, YELLOW),
        None,
        prompt_line(cursor=True),
    ], 1100)

    # ── 4: User lists checkpoints ─────────────────────────────────────────
    after2 = BANNER + [
        prompt_line(cmd), None,
        info_line("📸  Auto-snapshot saved  →  checkpoint #3"),
        None,
        prompt_line(cmd2),
        text_line("2 failed, 18 passed in 3.24s", 2, YELLOW),
        None,
    ]
    cmd3 = "/checkpoint"
    for i in range(0, len(cmd3) + 1, 2):
        add(after2 + [prompt_line(cmd3[:i], cursor=(i < len(cmd3)))], 70)
    add(after2 + [prompt_line(cmd3)], 400)

    add(after2 + [
        prompt_line("/checkpoint"),
        None,
        [seg("  Checkpoints (4 total):", CYAN, True)],
        None,
        ckpt_row(3, "14:32:07", "refactor db connection pool", 2, active=True),
        ckpt_row(2, "14:18:44", "add user auth endpoints",    1),
        ckpt_row(1, "14:05:12", "initial project setup",      3),
        ckpt_row(0, "13:51:30", "session start",              0),
        None,
        info_line("▶ = current  |  /checkpoint <id> to rewind"),
        None,
        prompt_line(cursor=True),
    ], 1200)

    # ── 5: User rewinds to checkpoint 2 ──────────────────────────────────
    after3 = BANNER + [
        prompt_line("/checkpoint"),
        None,
        [seg("  Checkpoints (4 total):", CYAN, True)],
        None,
        ckpt_row(3, "14:32:07", "refactor db connection pool", 2, active=True),
        ckpt_row(2, "14:18:44", "add user auth endpoints",    1),
        ckpt_row(1, "14:05:12", "initial project setup",      3),
        ckpt_row(0, "13:51:30", "session start",              0),
        None,
    ]
    cmd4 = "/checkpoint 2"
    for i in range(0, len(cmd4) + 1, 2):
        add(after3 + [prompt_line(cmd4[:i], cursor=(i < len(cmd4)))], 70)
    add(after3 + [prompt_line(cmd4)], 400)

    # ── 6: Rewind in progress ─────────────────────────────────────────────
    add(BANNER + [
        prompt_line("/checkpoint 2"),
        None,
        [seg("  ⠿  ", CYAN), seg("Rewinding to checkpoint #2…", SUBTEXT)],
        info_line("Restoring conversation history…"),
        info_line("Restoring files…"),
    ], 700)

    add(BANNER + [
        prompt_line("/checkpoint 2"),
        None,
        [seg("  ⠿  ", CYAN), seg("Rewinding to checkpoint #2…", SUBTEXT)],
        info_line("Restoring conversation history…"),
        info_line("Restoring files…"),
        None,
        [seg("  ", SUBTEXT), seg("src/db/pool.py  ", TEXT),
         seg("→ restored", GREEN, True)],
        [seg("  ", SUBTEXT), seg("src/db/config.py", TEXT),
         seg("→ restored", GREEN, True)],
    ], 700)

    # ── 7: Rewind complete + diff ─────────────────────────────────────────
    add(BANNER + [
        prompt_line("/checkpoint 2"),
        None,
        ok_line("Rewound to checkpoint #2  (14:18:44)"),
        ok_line("2 files restored to pre-refactor state"),
        None,
        [seg("  Changes undone:", SUBTEXT)],
        diff_line("-", "asyncpg pool (broken)", RED),
        diff_line("+", "psycopg2 pool (original, working)", GREEN),
        None,
        info_line("Conversation history also rewound — 1 turn removed"),
        None,
        prompt_line(cursor=True),
    ], 1200)

    # ── 8: Tests pass again ───────────────────────────────────────────────
    after4 = BANNER + [
        ok_line("Rewound to checkpoint #2  (14:18:44)"),
        ok_line("2 files restored to pre-refactor state"),
        None,
    ]
    cmd5 = "!pytest tests/ -q"
    for i in range(0, len(cmd5) + 1, 3):
        add(after4 + [prompt_line(cmd5[:i], cursor=(i < len(cmd5)))], 60)
    add(after4 + [prompt_line(cmd5)], 400)

    add(after4 + [
        prompt_line(cmd5),
        [seg("  $ ", SUBTEXT), seg("pytest tests/ -q", YELLOW)],
        text_line("20 passed in 2.87s", 2, GREEN),
        None,
        ok_line("All tests green — safe state restored"),
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
                       "..", "docs", "checkpoint_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
