#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws /image visual input using PIL.
Simulates: user pastes a UI screenshot → AI analyses the design → suggests
improvements. Then: user pastes a code screenshot with a bug → AI spots the issue.
"""
from PIL import Image, ImageDraw, ImageFont
import os
import math

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
FONT_S = make_font(size=11)


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


def _draw_ui_screenshot(draw, x, y, w, h):
    """Mock browser / UI screenshot showing a login form."""
    # Browser chrome bar
    draw.rectangle([x, y, x+w, y+h], fill=(40, 40, 60), outline=SUBTEXT, width=1)
    # URL bar
    bar_y = y + 6
    draw.rectangle([x+8, bar_y, x+w-8, bar_y+14], fill=(55, 55, 75), outline=(80, 80, 100))
    draw.text((x+12, bar_y+2), "http://localhost:3000/login", font=FONT_S, fill=SUBTEXT)
    # Page content area
    content_y = bar_y + 20
    draw.rectangle([x+8, content_y, x+w-8, y+h-8], fill=(35, 35, 50))
    # Login form
    form_x = x + w//2 - 80
    form_y = content_y + 12
    draw.text((form_x, form_y), "Login", font=FONT_B, fill=TEXT)
    # Username field
    draw.rectangle([form_x, form_y+22, form_x+160, form_y+36], fill=(50, 52, 70), outline=(80, 83, 110))
    draw.text((form_x+4, form_y+24), "Username", font=FONT_S, fill=SUBTEXT)
    # Password field
    draw.rectangle([form_x, form_y+44, form_x+160, form_y+58], fill=(50, 52, 70), outline=(80, 83, 110))
    draw.text((form_x+4, form_y+46), "Password", font=FONT_S, fill=SUBTEXT)
    # Button — very small, hard to click (bug)
    btn_y = form_y + 66
    draw.rectangle([form_x+60, btn_y, form_x+100, btn_y+14], fill=(80, 120, 200))
    draw.text((form_x+66, btn_y+2), "Login", font=FONT_S, fill=(240, 240, 255))
    # No password visibility toggle (missing feature)
    # Missing error state
    # No "forgot password" link


def _draw_code_screenshot(draw, x, y, w, h):
    """Mock code editor screenshot showing Python with an off-by-one bug."""
    draw.rectangle([x, y, x+w, y+h], fill=(28, 28, 42), outline=SUBTEXT, width=1)
    # Editor title bar
    draw.rectangle([x, y, x+w, y+16], fill=(40, 40, 58))
    draw.text((x+8, y+2), "paginate.py", font=FONT_S, fill=TEXT)
    # Line numbers + code
    code_lines = [
        ("1 ", SUBTEXT, "def paginate(items, page, size):"),
        ("2 ", SUBTEXT, "    start = page * size"),
        ("3 ", SUBTEXT, "    end   = start + size"),
        ("4 ", SUBTEXT, "    return items[start:end]"),
        ("5 ", SUBTEXT, ""),
        ("6 ", SUBTEXT, "# page=0 → items[0:10]  ✓"),
        ("7 ", RED,     "# page=1 → items[10:20] ✓  BUT"),
        ("8 ", RED,     "# page=0, size=0 → ZeroDivisionError"),
        ("9 ", SUBTEXT, ""),
    ]
    cy = y + 22
    for ln, lc, code in code_lines:
        draw.text((x+4,  cy), ln,   font=FONT_S, fill=lc)
        draw.text((x+20, cy), code, font=FONT_S, fill=RED if lc == RED else TEXT)
        cy += 14


def draw_frame(lines_segments, screenshot=None):
    img = blank_frame()
    d   = ImageDraw.Draw(img)
    y   = PAD_Y
    for item in lines_segments:
        if item is None:
            y += LINE_H
        elif isinstance(item, tuple) and item[0] == "__screenshot__":
            _, kind, sx, sy, sw, sh = item
            if kind == "ui":
                _draw_ui_screenshot(d, sx, sy, sw, sh)
            elif kind == "code":
                _draw_code_screenshot(d, sx, sy, sw, sh)
            y = sy + sh + 4
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


def image_prompt_line(filename):
    return [
        seg("[cheetahclaws] ", SUBTEXT),
        seg("» ", CYAN, True),
        seg("📎 ", YELLOW),
        seg(filename, YELLOW, True),
        seg("  attached", SUBTEXT),
    ]


def ok_line(msg):
    return [seg("✓  ", GREEN, True), seg(msg, TEXT)]


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


def vision_badge():
    return [
        seg("  👁  ", CYAN),
        seg("Vision input  —  analysing image…", SUBTEXT),
    ]


def screenshot_placeholder(kind, sx, sy, sw, sh):
    """Sentinel tuple that draw_frame() intercepts to draw the mock screenshot."""
    return ("__screenshot__", kind, sx, sy, sw, sh)


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms))

    def type_it(base, text, ms_char=60, ms_hold=400):
        for i in range(0, len(text) + 1, 3):
            add(base + [prompt_line(text[:i], cursor=(i < len(text)))], ms_char)
        add(base + [prompt_line(text)], ms_hold)

    # ── 0: Banner + label ─────────────────────────────────────────────────
    add(BANNER + [
        [seg("  ── /image  —  paste or attach any image for AI analysis ──", SUBTEXT)],
        None,
        prompt_line(cursor=True),
    ], 900)

    base_label = BANNER + [
        [seg("  ── /image  —  paste or attach any image for AI analysis ──", SUBTEXT)],
        None,
    ]

    # ══════════════════════════════════════════════════════════════════════
    # Scene A: UI screenshot review
    # ══════════════════════════════════════════════════════════════════════

    # User types /image and attaches file
    type_it(base_label, "/image login_screen.png  review the UI design")

    # Show image attached confirmation + screenshot preview
    SS_X, SS_Y, SS_W, SS_H = PAD_X, 130, 460, 170
    add(BANNER + [
        image_prompt_line("login_screen.png"),
        [seg("  review the UI design", TEXT)],
        None,
        vision_badge(),
        None,
        screenshot_placeholder("ui", SS_X, SS_Y, SS_W, SS_H),
    ], 900)

    # AI response streams in
    ui_resp = [
        "UI Review — login_screen.png",
        "",
        "  Issues found:",
        "  ✗  Login button too small (40px) — WCAG min is 44px",
        "  ✗  No password visibility toggle",
        "  ✗  No 'Forgot password?' link",
        "  ✗  No visible error state for failed login",
        "",
        "  Looks good:",
        "  ✓  Clean layout, good contrast ratio",
    ]

    add(BANNER + [
        image_prompt_line("login_screen.png"),
        [seg("  review the UI design", TEXT)],
        None,
        screenshot_placeholder("ui", SS_X, SS_Y, SS_W, SS_H),
        claude_header(),
    ] + [text_line(l, 2, RED if l.strip().startswith("✗") else
                         (GREEN if l.strip().startswith("✓") else
                          (CYAN if l == "  Issues found:" or l == "  Looks good:" else
                           (TEXT if not l.startswith("  ") else SUBTEXT))))
         for l in ui_resp] + [claude_sep()], 1100)

    add(BANNER + [
        image_prompt_line("login_screen.png"),
        [seg("  review the UI design", TEXT)],
        None,
        screenshot_placeholder("ui", SS_X, SS_Y, SS_W, SS_H),
        claude_header(),
    ] + [text_line(l, 2, RED if l.strip().startswith("✗") else
                         (GREEN if l.strip().startswith("✓") else
                          (CYAN if l == "  Issues found:" or l == "  Looks good:" else
                           (TEXT if not l.startswith("  ") else SUBTEXT))))
         for l in ui_resp] + [claude_sep(), None, prompt_line(cursor=True)], 1000)

    # ══════════════════════════════════════════════════════════════════════
    # Scene B: Code screenshot with bug
    # ══════════════════════════════════════════════════════════════════════
    after_ui = BANNER + [
        image_prompt_line("login_screen.png"),
        [seg("  review the UI design", TEXT)],
        None,
    ]

    type_it(after_ui, "/image paginate_bug.png  spot any bugs")

    SS2_X, SS2_Y, SS2_W, SS2_H = PAD_X, 165, 460, 145
    add(BANNER + [
        image_prompt_line("login_screen.png"),
        [seg("  review the UI design", TEXT)],
        None,
        image_prompt_line("paginate_bug.png"),
        [seg("  spot any bugs", TEXT)],
        None,
        vision_badge(),
        None,
        screenshot_placeholder("code", SS2_X, SS2_Y, SS2_W, SS2_H),
    ], 900)

    code_resp = [
        "Bug found in paginate.py:",
        "",
        "  Line 2:  start = page * size",
        "  When size=0 → start=0, end=0 → empty slice (silent, not crash)",
        "  When page=-1 → negative index — returns wrong data",
        "",
        "  Fix:",
        "    if size <= 0: raise ValueError('size must be > 0')",
        "    if page < 0:  raise ValueError('page must be >= 0')",
    ]

    add(BANNER + [
        image_prompt_line("login_screen.png"),
        [seg("  review the UI design", TEXT)],
        None,
        image_prompt_line("paginate_bug.png"),
        [seg("  spot any bugs", TEXT)],
        None,
        screenshot_placeholder("code", SS2_X, SS2_Y, SS2_W, SS2_H),
        claude_header(),
    ] + [text_line(l, 2, RED   if "Bug" in l or "Line" in l or "When" in l else
                         (GREEN if "Fix" in l or l.strip().startswith("if ") else
                          (CYAN if l.strip().startswith("if ") else TEXT)))
         for l in code_resp] + [claude_sep()], 1100)

    add(BANNER + [
        image_prompt_line("login_screen.png"),
        [seg("  review the UI design", TEXT)],
        None,
        image_prompt_line("paginate_bug.png"),
        [seg("  spot any bugs", TEXT)],
        None,
        screenshot_placeholder("code", SS2_X, SS2_Y, SS2_W, SS2_H),
        claude_header(),
    ] + [text_line(l, 2, RED   if "Bug" in l or "Line" in l or "When" in l else
                         (GREEN if "Fix" in l or l.strip().startswith("if ") else TEXT))
         for l in code_resp] + [claude_sep(), None, prompt_line(cursor=True)], 2500)

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
                       "..", "docs", "image_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
