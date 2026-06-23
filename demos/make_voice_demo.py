#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws /voice command using PIL.
Simulates the full voice pipeline: /voice status → record → waveform
→ STT transcription → auto-submit → AI response.
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


def draw_frame(lines_segments, waveform_data=None):
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

    # Draw waveform if provided: (cx, cy, amplitude_list, color)
    if waveform_data:
        cx, cy, amps, color = waveform_data
        bar_w = 4
        gap   = 3
        n     = len(amps)
        total = n * (bar_w + gap) - gap
        x0    = cx - total // 2
        for i, amp in enumerate(amps):
            bx = x0 + i * (bar_w + gap)
            half = max(2, int(amp))
            d.rectangle([bx, cy - half, bx + bar_w - 1, cy + half], fill=color)

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


def status_row(icon, label, status, detail=""):
    if status == "ok":
        tick = seg("  ✓  ", GREEN, True)
        col  = GREEN
    else:
        tick = seg("  ✗  ", RED, True)
        col  = RED
    return [tick, seg(f"{label}: ", SUBTEXT), seg(detail, col)]


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


def dim_line(t, indent=2):
    return [seg(" " * indent + t, SUBTEXT)]


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


def recording_box(elapsed, db_level=0.6):
    """Build the recording indicator lines."""
    bar_filled = int(db_level * 20)
    bar = "█" * bar_filled + "░" * (20 - bar_filled)
    return [
        [seg("  ╔══════════════════════════════════════╗", RED)],
        [seg("  ║  ", RED), seg("🎙  Recording…", TEXT, True),
         seg(f"  {elapsed:.1f}s", YELLOW), seg("                  ║", RED)],
        [seg("  ║  ", RED), seg("Level: ", SUBTEXT), seg(f"[{bar}]", CYAN),
         seg("  ║", RED)],
        [seg("  ║  ", RED), seg("Press ", SUBTEXT), seg("Enter", GREEN, True),
         seg(" to stop · ", SUBTEXT), seg("Ctrl+C", RED),
         seg(" to cancel      ║", RED)],
        [seg("  ╚══════════════════════════════════════╝", RED)],
    ]


# ── Waveform helpers ──────────────────────────────────────────────────────

def make_waveform(t, n=40, base=18):
    """Generate animated bar waveform amplitudes based on time t."""
    amps = []
    for i in range(n):
        phase = t * 6 + i * 0.7
        val = base + 10 * math.sin(phase) + 5 * math.sin(phase * 2.3 + 1)
        amps.append(max(3, val))
    return amps


def make_waveform_flat(n=40):
    return [3] * n


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120, waveform=None):
        scenes.append((lines, ms, waveform))

    TRANSCRIBED = "Refactor the authentication module to use JWT tokens"

    # ── 0: Banner + empty prompt ─────────────────────────────────────────
    add(BANNER + [prompt_line(cursor=True)], 1000)

    # ── 1: Type /voice status ─────────────────────────────────────────────
    cmd = "/voice status"
    for i in range(0, len(cmd) + 1, 3):
        add(BANNER + [prompt_line(cmd[:i], cursor=(i < len(cmd)))], 70)
    add(BANNER + [prompt_line(cmd)], 500)

    # ── 2: Status output ──────────────────────────────────────────────────
    base0 = BANNER + [prompt_line(cmd)]
    add(base0 + [
        None,
        [seg("  Voice status:", CYAN, True)],
        status_row("✓", "Recording backend", "ok", "sounddevice"),
        status_row("✓", "STT backend      ", "ok", "faster-whisper  (base model, offline)"),
        status_row("✓", "Microphone       ", "ok", "[0] Default Input Device"),
        [seg("  Language: ", SUBTEXT), seg("auto-detect", YELLOW)],
        None,
        info_line("Tip: /voice device to switch mic · /voice lang zh for Chinese"),
    ], 1200)

    # ── 3: New prompt → type /voice ───────────────────────────────────────
    status_tail = base0 + [
        None,
        [seg("  Voice status:", CYAN, True)],
        status_row("✓", "Recording backend", "ok", "sounddevice"),
        status_row("✓", "STT backend      ", "ok", "faster-whisper  (base model, offline)"),
        status_row("✓", "Microphone       ", "ok", "[0] Default Input Device"),
        [seg("  Language: ", SUBTEXT), seg("auto-detect", YELLOW)],
        None,
        info_line("Tip: /voice device to switch mic · /voice lang zh for Chinese"),
        None,
    ]
    add(status_tail + [prompt_line(cursor=True)], 800)

    cmd2 = "/voice"
    for i in range(0, len(cmd2) + 1, 2):
        add(status_tail + [prompt_line(cmd2[:i], cursor=(i < len(cmd2)))], 80)
    add(status_tail + [prompt_line(cmd2)], 400)

    # ── 4: Recording starts (waveform frames) ────────────────────────────
    base1 = status_tail + [prompt_line("/voice")]

    # Initial recording box, flat waveform
    WX, WY = W // 2, 540   # waveform center
    add(base1 + recording_box(0.0, 0.0), 200,
        waveform=(WX, WY, make_waveform_flat(), CYAN))

    # Animate recording for ~2.5 seconds (25 frames × 100ms)
    steps = 25
    for k in range(steps):
        t      = k / 8.0
        elapsed = k * 0.1
        db     = 0.3 + 0.5 * abs(math.sin(t * 1.2))
        amps   = make_waveform(t, n=40, base=16)
        add(base1 + recording_box(elapsed, db), 100,
            waveform=(WX, WY, amps, CYAN))

    # ── 5: Stop recording ─────────────────────────────────────────────────
    add(base1 + [
        [seg("  ╔══════════════════════════════════════╗", SUBTEXT)],
        [seg("  ║  ", SUBTEXT), seg("🎙  Recording stopped", GREEN, True),
         seg("  2.5s captured        ║", SUBTEXT)],
        [seg("  ╚══════════════════════════════════════╝", SUBTEXT)],
        None,
        info_line("Transcribing with faster-whisper (base)…"),
    ], 700, waveform=(WX, WY, make_waveform_flat(), SUBTEXT))

    # ── 6: Transcription progress ─────────────────────────────────────────
    add(base1 + [
        [seg("  ╔══════════════════════════════════════╗", SUBTEXT)],
        [seg("  ║  ", SUBTEXT), seg("🎙  Recording stopped", GREEN, True),
         seg("  2.5s captured        ║", SUBTEXT)],
        [seg("  ╚══════════════════════════════════════╝", SUBTEXT)],
        None,
        info_line("Transcribing with faster-whisper (base)…"),
        [seg("  ⠿  ", CYAN), seg("Processing audio…", SUBTEXT)],
    ], 900)

    # ── 7: Transcription result ───────────────────────────────────────────
    add(base1 + [
        None,
        ok_line("Transcription complete  (1.1 s)"),
        None,
        [seg("  📝  ", CYAN), seg("\"", SUBTEXT),
         seg(TRANSCRIBED, TEXT, True), seg("\"", SUBTEXT)],
        None,
        info_line("Submitting to model…"),
    ], 1000)

    # ── 8: Auto-submit — show as user query ───────────────────────────────
    base2 = BANNER + [
        [seg("[cheetahclaws] ", SUBTEXT), seg("» ", CYAN, True),
         seg(TRANSCRIBED, TEXT)],
    ]
    add(base2, 600)

    # ── 9: Claude header + spinner ────────────────────────────────────────
    add(base2 + [
        None,
        claude_header(),
        [seg("  ⠿  ", CYAN), seg("Thinking…", SUBTEXT)],
    ], 700)

    # ── 10: Tool use — Read auth module ──────────────────────────────────
    add(base2 + [
        None,
        claude_header(),
        tool_line("⚙", "Read", "src/auth/auth.py"),
    ], 500)
    add(base2 + [
        None,
        claude_header(),
        tool_line("⚙", "Read", "src/auth/auth.py"),
        tool_ok("→ 142 lines"),
        tool_line("⚙", "Read", "src/auth/middleware.py"),
    ], 500)
    add(base2 + [
        None,
        claude_header(),
        tool_line("⚙", "Read", "src/auth/auth.py"),
        tool_ok("→ 142 lines"),
        tool_line("⚙", "Read", "src/auth/middleware.py"),
        tool_ok("→ 89 lines"),
    ], 600)

    # ── 11: Stream AI response ────────────────────────────────────────────
    response_lines = [
        "I'll refactor the authentication module to use JWT tokens.",
        "",
        "**Plan:**",
        "  1. Replace session-based auth with `python-jose` JWT",
        "  2. Add `create_access_token()` and `verify_token()` helpers",
        "  3. Update middleware to validate Bearer tokens",
        "  4. Add token expiry (15 min access / 7 day refresh)",
        "",
        "Starting with `src/auth/auth.py`…",
    ]

    tool_block = [
        tool_line("⚙", "Read", "src/auth/auth.py"),
        tool_ok("→ 142 lines"),
        tool_line("⚙", "Read", "src/auth/middleware.py"),
        tool_ok("→ 89 lines"),
        [seg("│ ", SUBTEXT)],
    ]

    streamed = []
    for line in response_lines:
        streamed.append(text_line(line, 2))
        add(base2 + [None, claude_header()] + tool_block + streamed,
            65 if line else 25)

    # ── 12: Write tool ────────────────────────────────────────────────────
    full_response = [text_line(l, 2) for l in response_lines]
    add(base2 + [None, claude_header()] + tool_block + full_response + [
        None,
        tool_line("✏", "Edit", "src/auth/auth.py"),
    ], 500)
    add(base2 + [None, claude_header()] + tool_block + full_response + [
        None,
        tool_line("✏", "Edit", "src/auth/auth.py"),
        tool_ok("→ JWT helpers added  (+38 lines)"),
        tool_line("✏", "Edit", "src/auth/middleware.py"),
    ], 500)
    add(base2 + [None, claude_header()] + tool_block + full_response + [
        None,
        tool_line("✏", "Edit", "src/auth/auth.py"),
        tool_ok("→ JWT helpers added  (+38 lines)"),
        tool_line("✏", "Edit", "src/auth/middleware.py"),
        tool_ok("→ Bearer token validation updated"),
    ], 700)

    # ── 13: Done response ─────────────────────────────────────────────────
    done_lines = [
        "Done! Both files updated:",
        "",
        "  • auth.py      — create_access_token(), verify_token(), refresh logic",
        "  • middleware.py — reads Authorization: Bearer <token>, validates, injects user",
        "",
        "JWT secret reads from JWT_SECRET env var. Test with:",
        "  pytest tests/test_auth.py",
    ]
    add(base2 + [None, claude_header()] + [
        tool_line("✏", "Edit", "src/auth/auth.py"),
        tool_ok("→ JWT helpers added  (+38 lines)"),
        tool_line("✏", "Edit", "src/auth/middleware.py"),
        tool_ok("→ Bearer token validation updated"),
        [seg("│ ", SUBTEXT)],
    ] + [text_line(l, 2) for l in done_lines] + [claude_sep()], 1000)

    # ── 14: New voice prompt ───────────────────────────────────────────────
    add(base2 + [None, claude_header()] + [
        tool_line("✏", "Edit", "src/auth/auth.py"),
        tool_ok("→ JWT helpers added  (+38 lines)"),
        tool_line("✏", "Edit", "src/auth/middleware.py"),
        tool_ok("→ Bearer token validation updated"),
        [seg("│ ", SUBTEXT)],
    ] + [text_line(l, 2) for l in done_lines] + [claude_sep()] + [
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
    for i, (lines, ms, waveform) in enumerate(scenes):
        rgb_frames.append(draw_frame(lines, waveform_data=waveform))
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
                       "..", "docs", "voice_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
