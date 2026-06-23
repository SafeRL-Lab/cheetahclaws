#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws /cloudsave (Cloud Sync) using PIL.
Simulates: long session with code changes → /cloudsave → encrypted upload →
switch to laptop → /cloudload → session fully restored with files + history.
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

def banner(machine):
    return [
        [seg("╭─ CheetahClaws v3.5.5 ──────────────────────────────────╮", SUBTEXT)],
        [seg("│  ", SUBTEXT), seg("Model: ", SUBTEXT), seg("claude-sonnet-4-6", CYAN, True)],
        [seg("│  ", SUBTEXT), seg("Machine: ", SUBTEXT), seg(machine, YELLOW, True)],
        [seg("│  Type /help for commands, Ctrl+C to cancel                  │", SUBTEXT)],
        [seg("╰────────────────────────────────────────────────────────────╯", SUBTEXT)],
        None,
    ]


BANNER_DESKTOP = banner("desktop  (work)")
BANNER_LAPTOP  = banner("laptop   (home)")


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


def upload_line(file, size):
    return [
        seg("  ↑  ", CYAN, True),
        seg(file.ljust(36), TEXT),
        seg(size, SUBTEXT),
    ]


def download_line(file, state):
    col = GREEN if state == "ok" else CYAN
    icon = "✓" if state == "ok" else "↓"
    return [
        seg(f"  {icon}  ", col, True),
        seg(file, TEXT),
        seg("  restored", col) if state == "ok" else seg("  downloading…", SUBTEXT),
    ]


def progress_bar(pct, label=""):
    filled = int(pct / 100 * 44)
    bar = "█" * filled + "░" * (44 - filled)
    return [seg(f"  [{bar}] ", CYAN), seg(f"{pct}%", YELLOW, True),
            seg(f"  {label}", SUBTEXT)]


def divider(label):
    pad = "─" * max(0, 52 - len(label))
    return [seg(f"  ── {label} {pad}", SUBTEXT)]


def cloud_badge(msg, color=CYAN):
    return [seg("  ☁  ", color, True), seg(msg, color)]


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
    # DESKTOP: work session in progress
    # ════════════════════════════════════════════════════════════════════
    add(BANNER_DESKTOP + [
        divider("Desktop  —  end of work session"),
        None,
        info_line("Session: 2h 14min  |  14 AI turns  |  7 files modified"),
        None,
        [seg("  Modified files:", SUBTEXT)],
        [seg("    • ", CYAN), seg("src/api/routes.py      ", TEXT), seg("+89 lines", GREEN)],
        [seg("    • ", CYAN), seg("src/auth/middleware.py ", TEXT), seg("+34 lines", GREEN)],
        [seg("    • ", CYAN), seg("tests/test_auth.py     ", TEXT), seg("+78 lines", GREEN)],
        [seg("    • ", CYAN), seg("src/config.py          ", TEXT), seg("+12 lines", GREEN)],
        None,
        prompt_line(cursor=True),
    ], 1100)

    base_d = BANNER_DESKTOP + [
        divider("Desktop  —  end of work session"),
        None,
        info_line("Session: 2h 14min  |  14 AI turns  |  7 files modified"),
        None,
    ]

    # User types /cloudsave
    type_it(base_d + [
        [seg("    • ", CYAN), seg("src/api/routes.py", TEXT)],
        [seg("    • ", CYAN), seg("src/auth/middleware.py", TEXT)],
        [seg("    • ", CYAN), seg("tests/test_auth.py", TEXT)],
        None,
    ], "/cloudsave")

    # Uploading in progress
    add(BANNER_DESKTOP + [
        divider("Desktop  —  /cloudsave"),
        None,
        cloud_badge("Encrypting session snapshot…"),
        None,
        progress_bar(0,  "preparing…"),
    ], 500)

    add(BANNER_DESKTOP + [
        divider("Desktop  —  /cloudsave"),
        None,
        cloud_badge("Uploading to cloud…"),
        None,
        progress_bar(30, "conversation history"),
        upload_line("session/history.json.enc", "28 KB"),
    ], 500)

    add(BANNER_DESKTOP + [
        divider("Desktop  —  /cloudsave"),
        None,
        cloud_badge("Uploading to cloud…"),
        None,
        progress_bar(65, "modified files"),
        upload_line("session/history.json.enc",  "28 KB  ✓"),
        upload_line("files/src_api_routes.py",   "14 KB"),
        upload_line("files/src_auth_middleware.py", "6 KB"),
    ], 500)

    add(BANNER_DESKTOP + [
        divider("Desktop  —  /cloudsave"),
        None,
        cloud_badge("Uploading to cloud…"),
        None,
        progress_bar(90, "memories + checkpoints"),
        upload_line("session/history.json.enc",     "28 KB  ✓"),
        upload_line("files/src_api_routes.py",      "14 KB  ✓"),
        upload_line("files/src_auth_middleware.py",  "6 KB  ✓"),
        upload_line("session/checkpoints.tar.enc",  "52 KB"),
    ], 500)

    # Upload complete
    add(BANNER_DESKTOP + [
        divider("Desktop  —  /cloudsave"),
        None,
        progress_bar(100, "complete"),
        None,
        ok_line("Session saved to cloud  (AES-256 encrypted)"),
        ok_line("Share ID:  cc-a3f9b2  (valid 7 days)"),
        None,
        info_line("On another machine:  /cloudload cc-a3f9b2"),
        None,
        prompt_line(cursor=True),
    ], 1200)

    # ════════════════════════════════════════════════════════════════════
    # LAPTOP: loading from cloud
    # ════════════════════════════════════════════════════════════════════
    add(BANNER_LAPTOP + [
        divider("Laptop  —  continuing from cloud"),
        None,
        prompt_line(cursor=True),
    ], 900)

    base_l = BANNER_LAPTOP + [
        divider("Laptop  —  continuing from cloud"),
        None,
    ]

    type_it(base_l, "/cloudload cc-a3f9b2")

    # Downloading
    add(BANNER_LAPTOP + [
        divider("Laptop  —  /cloudload cc-a3f9b2"),
        None,
        cloud_badge("Fetching session cc-a3f9b2…"),
        None,
        progress_bar(0, "connecting…"),
    ], 500)

    add(BANNER_LAPTOP + [
        divider("Laptop  —  /cloudload cc-a3f9b2"),
        None,
        cloud_badge("Downloading + decrypting…"),
        None,
        progress_bar(40, "conversation history"),
        download_line("session/history.json.enc",  "ok"),
    ], 500)

    add(BANNER_LAPTOP + [
        divider("Laptop  —  /cloudload cc-a3f9b2"),
        None,
        cloud_badge("Downloading + decrypting…"),
        None,
        progress_bar(75, "modified files"),
        download_line("session/history.json.enc",     "ok"),
        download_line("files/src_api_routes.py",      "ok"),
        download_line("files/src_auth_middleware.py", "ok"),
    ], 500)

    add(BANNER_LAPTOP + [
        divider("Laptop  —  /cloudload cc-a3f9b2"),
        None,
        progress_bar(100, "complete"),
        None,
        ok_line("Session cc-a3f9b2 fully restored"),
        None,
        [seg("  Restored:", SUBTEXT)],
        [seg("    ✓ ", GREEN), seg("14 conversation turns", TEXT)],
        [seg("    ✓ ", GREEN), seg("4 modified files", TEXT)],
        [seg("    ✓ ", GREEN), seg("3 checkpoints", TEXT)],
        [seg("    ✓ ", GREEN), seg("memories (pytest · FastAPI · GCP)", TEXT)],
        None,
        info_line("Continuing exactly where you left off."),
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
                       "..", "docs", "cloudsave_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
