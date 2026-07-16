#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws /plan command using PIL.
Simulates: /plan → read-only analysis → write plan file → /plan done
→ implementation begins → files edited.
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

BANNER_PLAN = [
    [seg("╭─ CheetahClaws v3.5.5 ──────────────────────────────────╮", SUBTEXT)],
    [seg("│  ", SUBTEXT), seg("Model: ", SUBTEXT), seg("claude-sonnet-4-6", CYAN, True)],
    [seg("│  ", SUBTEXT), seg("Permissions: ", SUBTEXT), seg("plan", MAUVE, True),
     seg("  [PLAN MODE — read only]", MAUVE)],
    [seg("│  Type /help for commands, Ctrl+C to cancel                  │", SUBTEXT)],
    [seg("╰────────────────────────────────────────────────────────────╯", SUBTEXT)],
    None,
]


def prompt_line(text="", cursor=False, plan=False):
    cur = "█" if cursor else ""
    label = "[plan-mode]  " if plan else "[cheetahclaws] "
    label_col = MAUVE if plan else SUBTEXT
    return [
        seg(label, label_col),
        seg("» ", CYAN, True),
        seg(text + cur, TEXT),
    ]


def ok_line(msg):
    return [seg("✓  ", GREEN, True), seg(msg, TEXT)]


def info_line(msg):
    return [seg("  ", SUBTEXT), seg(msg, SUBTEXT)]


def warn_line(msg):
    return [seg("  ⚠  ", YELLOW, True), seg(msg, YELLOW)]


def claude_header(plan=False):
    mode = seg(" [PLAN MODE]", MAUVE, True) if plan else seg("", SUBTEXT)
    return [
        seg("╭─ CheetahClaws ", SUBTEXT),
        seg("●", MAUVE if plan else GREEN),
        seg(" ─────────────────────────────────────────────", SUBTEXT),
        mode,
    ]


def claude_sep():
    return [seg("╰──────────────────────────────────────────────────────────", SUBTEXT)]


def text_line(t, indent=2, color=TEXT):
    return [seg(" " * indent + t, color)]


def tool_line(icon, name, arg, blocked=False):
    col = RED if blocked else CYAN
    return [
        seg(f"  {icon}  ", SUBTEXT),
        seg(name, col),
        seg("(", SUBTEXT),
        seg(arg, TEXT),
        seg(")", SUBTEXT),
    ]


def tool_ok(msg):
    return [seg("  ✓ ", GREEN), seg(msg, SUBTEXT)]


def tool_blocked(name):
    return [seg("  ✗ ", RED, True), seg(f"{name} blocked in plan mode", RED)]


def plan_section(title):
    return [seg(f"  ## {title}", CYAN, True)]


def plan_item(n, text):
    return [seg(f"    {n}. ", YELLOW), seg(text, TEXT)]


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms))

    PLAN_FILE = "plan.md"
    TASK = "add rate limiting to the API"

    # ── 0: Banner + empty prompt ─────────────────────────────────────────
    add(BANNER + [prompt_line(cursor=True)], 900)

    # ── 1: Type /plan add rate limiting ──────────────────────────────────
    cmd = f"/plan {TASK}"
    for i in range(0, len(cmd) + 1, 4):
        add(BANNER + [prompt_line(cmd[:i], cursor=(i < len(cmd)))], 60)
    add(BANNER + [prompt_line(cmd)], 500)

    # ── 2: Plan mode activated ────────────────────────────────────────────
    add(BANNER_PLAN + [
        ok_line("Plan mode activated  →  plan.md"),
        warn_line("Write/Edit blocked until /plan done"),
        info_line("AI will analyse and write a plan — no code changes yet."),
        None,
        prompt_line(TASK, plan=True),
    ], 1100)

    base_plan = BANNER_PLAN + [
        ok_line("Plan mode activated  →  plan.md"),
        warn_line("Write/Edit blocked until /plan done"),
        None,
        prompt_line(TASK, plan=True),
        None,
    ]

    # ── 3: AI reads codebase (read-only tools allowed) ────────────────────
    add(base_plan + [claude_header(plan=True),
        tool_line("⚙", "Glob", "**/*.py")], 500)
    add(base_plan + [claude_header(plan=True),
        tool_line("⚙", "Glob", "**/*.py"),
        tool_ok("→ 14 files"),
        tool_line("⚙", "Read", "src/api/routes.py")], 500)
    add(base_plan + [claude_header(plan=True),
        tool_line("⚙", "Glob", "**/*.py"),
        tool_ok("→ 14 files"),
        tool_line("⚙", "Read", "src/api/routes.py"),
        tool_ok("→ 198 lines"),
        tool_line("⚙", "Grep", "pattern=@app.route  |  middleware")], 500)
    add(base_plan + [claude_header(plan=True),
        tool_line("⚙", "Glob", "**/*.py"),
        tool_ok("→ 14 files"),
        tool_line("⚙", "Read", "src/api/routes.py"),
        tool_ok("→ 198 lines"),
        tool_line("⚙", "Grep", "pattern=@app.route  |  middleware"),
        tool_ok("→ 12 matches  (no rate-limit middleware found)"),
    ], 700)

    # ── 4: Attempt write (blocked) ────────────────────────────────────────
    add(base_plan + [
        claude_header(plan=True),
        tool_ok("→ 12 matches  (no rate-limit middleware found)"),
        tool_line("✏", "Edit", "src/api/routes.py", blocked=True),
        tool_blocked("Edit"),
        warn_line("Plan mode: writes blocked. Writing plan to plan.md instead."),
    ], 1000)

    # ── 5: Write plan.md ─────────────────────────────────────────────────
    plan_lines = [
        "# Plan: Add Rate Limiting to API",
        "",
        "## Analysis",
        "  - 14 Python files, no existing rate-limit middleware",
        "  - Entry point: src/api/routes.py (198 lines)",
        "  - Flask app with 8 endpoints, no auth decorators",
        "",
        "## Implementation Steps",
        "  1. Install flask-limiter  (pip install flask-limiter)",
        "  2. Create src/middleware/rate_limit.py",
        "     - Global: 100 req/min per IP",
        "     - /auth endpoints: 10 req/min (stricter)",
        "  3. Register limiter in src/app.py  (3 lines)",
        "  4. Add tests: tests/test_rate_limit.py",
        "",
        "## Risk: None — additive change, no existing logic modified",
    ]
    add(base_plan + [
        claude_header(plan=True),
        tool_line("✎", "Write", "plan.md"),
    ], 400)
    streamed = []
    for line in plan_lines:
        streamed.append(text_line(line, 4, SUBTEXT if line.startswith("  ") else TEXT))
        add(base_plan + [claude_header(plan=True),
            tool_line("✎", "Write", "plan.md")] + streamed, 55 if line else 20)

    add(base_plan + [claude_header(plan=True),
        tool_line("✎", "Write", "plan.md"),
        tool_ok("→ plan.md saved  (16 lines)"),
        None,
        [seg("│ ", SUBTEXT)],
        text_line("Plan written. Run /plan done to begin implementation.", 2),
        claude_sep(),
    ], 1000)

    # ── 6: User runs /plan done ───────────────────────────────────────────
    after_plan = BANNER_PLAN + [
        ok_line("Plan mode activated  →  plan.md"),
        warn_line("Write/Edit blocked until /plan done"),
        None,
        prompt_line(TASK, plan=True),
        None,
        claude_header(plan=True),
        tool_ok("→ plan.md saved  (16 lines)"),
        text_line("Plan written. Run /plan done to begin implementation.", 2),
        claude_sep(),
        None,
    ]
    cmd2 = "/plan done"
    for i in range(0, len(cmd2) + 1, 3):
        add(after_plan + [prompt_line(cmd2[:i], plan=True,
                          cursor=(i < len(cmd2)))], 70)
    add(after_plan + [prompt_line(cmd2, plan=True)], 400)

    # ── 7: Exit plan mode → implement ────────────────────────────────────
    add(BANNER + [
        ok_line("Plan approved — exiting plan mode"),
        ok_line("Permissions restored: auto"),
        info_line("Starting implementation from plan.md…"),
        None,
        prompt_line(cursor=True),
    ], 1000)

    impl_base = BANNER + [
        ok_line("Implementing plan…"),
        None,
        prompt_line("implement the plan"),
        None,
        claude_header(),
    ]

    # ── 8: Implementation ─────────────────────────────────────────────────
    add(impl_base + [
        tool_line("⚙", "Read", "plan.md"),
        tool_ok("→ 16 lines"),
        tool_line("⌨", "Bash", "pip install flask-limiter"),
        tool_ok("→ Successfully installed flask-limiter-3.5.0"),
        tool_line("✎", "Write", "src/middleware/rate_limit.py"),
        tool_ok("→ created  (+28 lines)"),
        tool_line("✏", "Edit", "src/app.py"),
        tool_ok("→ limiter registered  (+3 lines)"),
        tool_line("✎", "Write", "tests/test_rate_limit.py"),
        tool_ok("→ created  (+41 lines)"),
    ], 900)

    done_resp = [
        "Implementation complete:",
        "",
        "  ✓  flask-limiter installed",
        "  ✓  src/middleware/rate_limit.py  — 100 req/min global, 10/min on /auth",
        "  ✓  src/app.py                    — limiter registered",
        "  ✓  tests/test_rate_limit.py      — 6 test cases",
        "",
        "Run  pytest tests/test_rate_limit.py  to verify.",
    ]
    add(impl_base + [
        tool_ok("→ created  (+41 lines)"),
        None, [seg("│ ", SUBTEXT)],
    ] + [text_line(l, 2) for l in done_resp] + [claude_sep()], 1000)

    add(impl_base + [
        tool_ok("→ created  (+41 lines)"),
        None, [seg("│ ", SUBTEXT)],
    ] + [text_line(l, 2) for l in done_resp] + [claude_sep()] + [
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
                       "..", "docs", "plan_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
