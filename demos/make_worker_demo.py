#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws /worker batch task execution using PIL.
Simulates: brainstorm generates todo_list → /worker auto-implements each item
→ progress bar → each task completed in sequence → final summary.
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


def text_line(t, indent=2, color=TEXT):
    return [seg(" " * indent + t, color)]


def todo_item(n, text, state="pending"):
    # state: pending | active | done
    if state == "done":
        icon = seg("  ✓ ", GREEN, True)
        col  = SUBTEXT
    elif state == "active":
        icon = seg("  ⟳ ", CYAN, True)
        col  = TEXT
    else:
        icon = seg("  ○ ", SUBTEXT)
        col  = SUBTEXT
    return [icon, seg(f"{n}. ", SUBTEXT), seg(text, col)]


def worker_header(current, total):
    pct = int(current / total * 100)
    filled = int(current / total * 40)
    bar = "█" * filled + "░" * (40 - filled)
    return [
        [seg("  ── /worker  batch mode ─────────────────────────────────", CYAN)],
        [seg(f"  [{bar}] ", CYAN), seg(f"{pct}%  ", YELLOW, True),
         seg(f"({current}/{total} tasks)", SUBTEXT)],
        None,
    ]


def worker_active(name):
    return [seg("  ⟳  ", CYAN, True), seg("Working on: ", SUBTEXT), seg(name, TEXT, True)]


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


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms))

    def type_it(base, text, ms_char=60, ms_hold=400):
        for i in range(0, len(text) + 1, 3):
            add(base + [prompt_line(text[:i], cursor=(i < len(text)))], ms_char)
        add(base + [prompt_line(text)], ms_hold)

    TASKS = [
        "Add input validation to all API endpoints",
        "Write unit tests for auth module",
        "Add OpenAPI docs (docstrings + schema)",
        "Set up rate limiting middleware",
        "Create CI workflow  (.github/workflows/ci.yml)",
    ]

    # ── 0: Banner + empty prompt ─────────────────────────────────────────
    add(BANNER + [prompt_line(cursor=True)], 900)

    # ── 1: User brainstorms tasks ─────────────────────────────────────────
    q = "brainstorm what's needed to harden this API for production"
    type_it(BANNER, q)

    base0 = BANNER + [prompt_line(q), None, claude_header()]

    brainstorm_resp = [
        "Here's a production-readiness checklist:",
        "",
    ] + [f"  {i+1}. {t}" for i, t in enumerate(TASKS)]

    streamed = []
    for line in brainstorm_resp:
        streamed.append(text_line(line, 2, SUBTEXT if line.startswith("  ") else TEXT))
        add(base0 + streamed, 55 if line else 20)

    add(base0 + [text_line(l, 2, SUBTEXT if l.startswith("  ") else TEXT)
                 for l in brainstorm_resp] + [claude_sep()], 800)

    # ── 2: User runs /worker ──────────────────────────────────────────────
    after_brainstorm = BANNER + [
        prompt_line(q), None,
    ] + [text_line(f"  {i+1}. {t}", 2, SUBTEXT) for i, t in enumerate(TASKS)] + [None]

    type_it(after_brainstorm, "/worker implement all 5", ms_hold=500)

    # Worker activated
    add(BANNER + [
        ok_line("/worker  batch mode activated  —  5 tasks queued"),
        info_line("Each task runs in sequence; checkpoints saved between tasks."),
        None,
    ] + worker_header(0, 5) + [
        todo_item(1, TASKS[0], "pending"),
        todo_item(2, TASKS[1], "pending"),
        todo_item(3, TASKS[2], "pending"),
        todo_item(4, TASKS[3], "pending"),
        todo_item(5, TASKS[4], "pending"),
        None,
        prompt_line(cursor=True),
    ], 1100)

    # ── Task 1 ────────────────────────────────────────────────────────────
    def task_doing(done_indices, active_idx, tool_lines):
        items = []
        for i, t in enumerate(TASKS):
            if i in done_indices:
                items.append(todo_item(i+1, t, "done"))
            elif i == active_idx:
                items.append(todo_item(i+1, t, "active"))
            else:
                items.append(todo_item(i+1, t, "pending"))
        return (BANNER + worker_header(len(done_indices), 5) +
                items + [None] + tool_lines)

    # Task 1 in progress
    add(task_doing(set(), 0, [
        worker_active(TASKS[0]),
        tool_line("⚙", "Glob", "src/api/**/*.py"),
        tool_ok("→ 8 route files"),
        tool_line("✏", "Edit", "src/api/routes.py"),
        tool_ok("→ added Pydantic validators  (+34 lines)"),
    ]), 800)

    # Task 1 done → Task 2 starting
    add(task_doing({0}, 1, [
        worker_active(TASKS[1]),
        tool_line("⚙", "Read", "src/auth/auth.py"),
        tool_ok("→ 143 lines"),
        tool_line("✎", "Write", "tests/test_auth.py"),
        tool_ok("→ created  (+78 lines, 12 test cases)"),
    ]), 800)

    # Task 2 done → Task 3 starting
    add(task_doing({0, 1}, 2, [
        worker_active(TASKS[2]),
        tool_line("⚙", "Glob", "src/**/*.py"),
        tool_ok("→ 14 files"),
        tool_line("✏", "Edit", "src/api/routes.py"),
        tool_ok("→ added docstrings + Pydantic schema  (+22 lines)"),
    ]), 800)

    # Task 3 done → Task 4 starting
    add(task_doing({0, 1, 2}, 3, [
        worker_active(TASKS[3]),
        tool_line("⌨", "Bash", "pip install flask-limiter"),
        tool_ok("→ installed flask-limiter-3.5.0"),
        tool_line("✎", "Write", "src/middleware/rate_limit.py"),
        tool_ok("→ created  (+28 lines)"),
    ]), 800)

    # Task 4 done → Task 5 starting
    add(task_doing({0, 1, 2, 3}, 4, [
        worker_active(TASKS[4]),
        tool_line("✎", "Write", ".github/workflows/ci.yml"),
        tool_ok("→ created  (+42 lines)  — lint + test + type-check"),
    ]), 800)

    # ── All tasks done ────────────────────────────────────────────────────
    add(BANNER + worker_header(5, 5) + [
        todo_item(1, TASKS[0], "done"),
        todo_item(2, TASKS[1], "done"),
        todo_item(3, TASKS[2], "done"),
        todo_item(4, TASKS[3], "done"),
        todo_item(5, TASKS[4], "done"),
        None,
        ok_line("All 5 tasks complete  ─  5 checkpoints saved"),
        info_line("Run /checkpoint to review history or rewind any step."),
        None,
        [seg("  ── Files changed ──────────────────────────────────────", SUBTEXT)],
        [seg("    +", GREEN, True), seg("  src/api/routes.py  src/middleware/rate_limit.py", TEXT)],
        [seg("    +", GREEN, True), seg("  tests/test_auth.py  .github/workflows/ci.yml", TEXT)],
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
                       "..", "docs", "worker_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
