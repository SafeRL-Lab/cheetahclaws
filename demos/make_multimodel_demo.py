#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws multi-model switching using PIL.
Simulates: start with Claude → /model gpt-4o → query → /model ollama/qwen2.5
→ query → /model claude-sonnet-4-6 → back. Shows same session, different models.
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


# ── Model color mapping ───────────────────────────────────────────────────

MODEL_COLORS = {
    "claude": MAUVE,
    "gpt":    GREEN,
    "gemini": BLUE,
    "ollama": PEACH,
    "custom": CYAN,
}

def model_color(name):
    n = name.lower()
    for k, c in MODEL_COLORS.items():
        if k in n:
            return c
    return CYAN


def make_banner(model, provider):
    col = model_color(model)
    return [
        [seg("╭─ CheetahClaws v3.5.5 ──────────────────────────────────╮", SUBTEXT)],
        [seg("│  ", SUBTEXT), seg("Model: ", SUBTEXT), seg(model, col, True),
         seg(f"  ({provider})", SUBTEXT)],
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


def claude_header(model):
    col = model_color(model)
    return [
        seg("╭─ CheetahClaws ", SUBTEXT),
        seg("●", col),
        seg(" ─────────────────────────────────────────────", SUBTEXT),
    ]


def claude_sep():
    return [seg("╰──────────────────────────────────────────────────────────", SUBTEXT)]


def text_line(t, indent=2, color=TEXT):
    return [seg(" " * indent + t, color)]


def model_switch_banner(old, new):
    old_col = model_color(old)
    new_col = model_color(new)
    return [
        [seg("  ── Model switched ──────────────────────────────────────", SUBTEXT)],
        [seg("  ", SUBTEXT), seg(old, old_col, True),
         seg("  →  ", SUBTEXT), seg(new, new_col, True)],
        [seg("  Conversation history preserved", SUBTEXT)],
        [seg("  ─────────────────────────────────────────────────────────", SUBTEXT)],
    ]


def available_models():
    return [
        [seg("  ── Available models ─────────────────────────────────────", SUBTEXT)],
        [seg("  Cloud   ", SUBTEXT),
         seg("claude-sonnet-4-6", MAUVE), seg("  claude-opus-4-6", MAUVE),
         seg("  gpt-4o", GREEN), seg("  gemini-2.0-flash", BLUE)],
        [seg("  Local   ", SUBTEXT),
         seg("ollama/qwen2.5-coder", PEACH), seg("  ollama/llama3.3", PEACH),
         seg("  ollama/deepseek-r1", PEACH)],
        [seg("  Custom  ", SUBTEXT),
         seg("custom/Qwen/Qwen2.5-14B", CYAN)],
        [seg("  ─────────────────────────────────────────────────────────", SUBTEXT)],
    ]


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms))

    def type_it(base, text, ms_char=60, ms_hold=400, is_cmd=False):
        for i in range(0, len(text) + 1, 3):
            add(base + [prompt_line(text[:i], cursor=(i < len(text)))], ms_char)
        add(base + [prompt_line(text)], ms_hold)

    # ── 0: Start — Claude ────────────────────────────────────────────────
    B_claude = make_banner("claude-sonnet-4-6", "anthropic")
    add(B_claude + [prompt_line(cursor=True)], 900)

    # ── 1: /model — show available ────────────────────────────────────────
    type_it(B_claude, "/model", ms_hold=400)
    add(B_claude + [prompt_line("/model")] + available_models() + [
        None, prompt_line(cursor=True),
    ], 1200)

    # ── 2: Ask Claude a question ──────────────────────────────────────────
    q1 = "explain async/await in Python in 2 lines"
    base_c = B_claude + available_models() + [None]
    type_it(base_c, q1, ms_hold=400)

    claude_resp = [
        "`async def` defines a coroutine; `await` suspends it until",
        "the awaited task completes, letting other tasks run meanwhile.",
    ]
    add(base_c + [prompt_line(q1), None, claude_header("claude")] +
        [text_line(l, 2) for l in claude_resp] + [claude_sep()], 1000)

    # ── 3: Switch to GPT-4o ───────────────────────────────────────────────
    after_claude = B_claude + [
        prompt_line(q1), None, claude_header("claude"),
    ] + [text_line(l, 2) for l in claude_resp] + [claude_sep(), None]

    type_it(after_claude, "/model gpt-4o", ms_hold=400)

    B_gpt = make_banner("gpt-4o", "openai")
    add(B_gpt + model_switch_banner("claude-sonnet-4-6", "gpt-4o") + [
        None, prompt_line(cursor=True),
    ], 1000)

    # ── 4: Ask GPT same question ──────────────────────────────────────────
    base_g = B_gpt + model_switch_banner("claude-sonnet-4-6", "gpt-4o") + [None]
    type_it(base_g, q1, ms_hold=400)

    gpt_resp = [
        "`async/await` lets you write non-blocking code: `await` pauses",
        "execution to yield control, resuming when the I/O is ready.",
    ]
    add(base_g + [prompt_line(q1), None, claude_header("gpt-4o")] +
        [text_line(l, 2) for l in gpt_resp] + [claude_sep()], 1000)

    # ── 5: Switch to local Ollama ─────────────────────────────────────────
    after_gpt = B_gpt + [
        prompt_line(q1), None, claude_header("gpt-4o"),
    ] + [text_line(l, 2) for l in gpt_resp] + [claude_sep(), None]

    type_it(after_gpt, "/model ollama/qwen2.5-coder", ms_hold=400)

    B_ollama = make_banner("ollama/qwen2.5-coder", "ollama · local · offline")
    add(B_ollama + model_switch_banner("gpt-4o", "ollama/qwen2.5-coder") + [
        None,
        info_line("Running locally — no API key required, fully offline"),
        None,
        prompt_line(cursor=True),
    ], 1100)

    # ── 6: Ask Ollama ─────────────────────────────────────────────────────
    base_o = B_ollama + model_switch_banner("gpt-4o", "ollama/qwen2.5-coder") + [
        info_line("Running locally — no API key required, fully offline"),
        None,
    ]
    type_it(base_o, q1, ms_hold=400)

    ollama_resp = [
        "Use `async def` to mark a function as async, then `await`",
        "before coroutines — Python schedules them cooperatively.",
    ]
    add(base_o + [prompt_line(q1), None, claude_header("ollama/qwen2.5-coder")] +
        [text_line(l, 2) for l in ollama_resp] + [claude_sep()], 1000)

    # ── 7: Switch back to Claude ──────────────────────────────────────────
    after_ollama = B_ollama + [
        prompt_line(q1), None, claude_header("ollama/qwen2.5-coder"),
    ] + [text_line(l, 2) for l in ollama_resp] + [claude_sep(), None]

    type_it(after_ollama, "/model claude-sonnet-4-6", ms_hold=400)

    add(B_claude + model_switch_banner("ollama/qwen2.5-coder", "claude-sonnet-4-6") + [
        None,
        ok_line("Switched back to claude-sonnet-4-6  —  full history intact"),
        None,
        [seg("  ── Models used this session ──────────────────────────────", SUBTEXT)],
        [seg("    ✓  ", MAUVE), seg("claude-sonnet-4-6", MAUVE, True),
         seg("  (anthropic)", SUBTEXT)],
        [seg("    ✓  ", GREEN), seg("gpt-4o            ", GREEN, True),
         seg("  (openai)", SUBTEXT)],
        [seg("    ✓  ", PEACH), seg("ollama/qwen2.5-coder", PEACH, True),
         seg("  (local)", SUBTEXT)],
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
                       "..", "docs", "multimodel_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
