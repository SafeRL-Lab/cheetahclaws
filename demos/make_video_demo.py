#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws /video command using PIL.
Simulates the full video factory pipeline: wizard → story generation
→ TTS narration → image search → subtitle burn → FFmpeg assembly → .mp4 output.
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


def draw_frame(lines_segments, video_preview=False):
    img = blank_frame()
    d   = ImageDraw.Draw(img)

    if video_preview:
        _draw_video_preview(img, d)
        return img

    y = PAD_Y
    for item in lines_segments:
        if item is None:
            y += LINE_H
        elif isinstance(item, list):
            y = render_line(d, y, item)
        else:
            y = render_line(d, y, [item])
    return img


def _draw_video_preview(img, d):
    """Draw a simulated video player frame showing the generated video."""
    # ── Background: dark gradient panels simulating video frames ─────────
    # Sky gradient (top half of video area)
    VX, VY, VW, VH = 60, 80, 840, 480
    for row in range(VH // 2):
        ratio = row / (VH // 2)
        r = int(10  + ratio * 30)
        g = int(15  + ratio * 40)
        b = int(60  + ratio * 80)
        d.rectangle([VX, VY + row, VX + VW, VY + row + 1], fill=(r, g, b))
    # Ground / cityscape (bottom half)
    for row in range(VH // 2):
        ratio = row / (VH // 2)
        r = int(8  + ratio * 20)
        g = int(8  + ratio * 12)
        b = int(20 + ratio * 20)
        d.rectangle([VX, VY + VH // 2 + row, VX + VW, VY + VH // 2 + row + 1],
                    fill=(r, g, b))

    # ── Circuit / neural network decorative lines ─────────────────────────
    nodes = [
        (180, 200), (340, 160), (520, 220), (680, 170), (820, 210),
        (260, 310), (450, 280), (620, 330), (760, 290),
        (150, 390), (380, 360), (560, 400), (730, 370), (860, 410),
    ]
    edges = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(1,5),(2,6),(3,6),(3,7),(4,7),
        (5,6),(6,7),(8,7),(5,9),(6,10),(7,11),(8,12),(9,10),(10,11),(11,12),(12,13),
    ]
    node_color  = (60, 140, 200, 180)
    edge_color  = (40, 100, 160)
    for a, b_ in edges:
        if a < len(nodes) and b_ < len(nodes):
            ax, ay = nodes[a]
            bx, by = nodes[b_]
            d.line([VX + ax, VY + ay, VX + bx, VY + by], fill=edge_color, width=1)
    for nx, ny in nodes:
        r = 5
        d.ellipse([VX+nx-r, VY+ny-r, VX+nx+r, VY+ny+r], fill=(80, 180, 240))

    # ── Glowing headline text in video ────────────────────────────────────
    title_font = make_font(28, bold=True)
    title = "The Rise of Artificial Intelligence"
    tw = title_font.getlength(title)
    tx = VX + (VW - tw) // 2
    ty = VY + 30
    # Glow effect (offset shadow)
    for ox, oy in [(-2,0),(2,0),(0,-2),(0,2)]:
        d.text((tx+ox, ty+oy), title, font=title_font, fill=(30, 100, 200))
    d.text((tx, ty), title, font=title_font, fill=(200, 230, 255))

    # ── Subtitle bar at bottom of video ───────────────────────────────────
    sub_y = VY + VH - 68
    d.rectangle([VX, sub_y, VX + VW, VY + VH], fill=(0, 0, 0))
    sub_font = make_font(17, bold=True)
    subtitle = "...the machines we built are now teaching us"
    sw = sub_font.getlength(subtitle)
    sx = VX + (VW - sw) // 2
    d.text((sx, sub_y + 10), subtitle, font=sub_font, fill=(255, 255, 200))
    sub2 = "what it means to think."
    sw2 = sub_font.getlength(sub2)
    d.text((VX + (VW - sw2) // 2, sub_y + 34), sub2, font=sub_font, fill=(255, 255, 200))

    # ── Video player border ───────────────────────────────────────────────
    for i in range(3):
        d.rectangle([VX-i, VY-i, VX+VW+i, VY+VH+i], outline=CYAN)

    # ── Player controls bar ───────────────────────────────────────────────
    cy = VY + VH + 6
    ctrl_h = 28
    d.rectangle([VX, cy, VX+VW, cy+ctrl_h], fill=SURFACE)
    # Progress bar (at 42%)
    prog_w = int(VW * 0.42)
    d.rectangle([VX+4, cy+10, VX+4+prog_w, cy+18], fill=CYAN)
    d.rectangle([VX+4+prog_w, cy+10, VX+VW-4, cy+18], fill=SUBTEXT)
    # Playhead dot
    d.ellipse([VX+4+prog_w-5, cy+7, VX+4+prog_w+5, cy+21], fill=GREEN)
    # Time labels
    tf = make_font(11)
    d.text((VX+8, cy+8), "0:24", font=tf, fill=TEXT)
    d.text((VX+VW-36, cy+8), "0:58", font=tf, fill=SUBTEXT)

    # ── Caption below player ──────────────────────────────────────────────
    cap_y = cy + ctrl_h + 10
    cap_font = make_font(13)
    cap = f"output_20260410_143022.mp4  ·  1280×720  ·  24 fps  ·  58 s"
    cw = cap_font.getlength(cap)
    d.text((VX + (VW - cw)//2, cap_y), cap, font=cap_font, fill=SUBTEXT)

    # ── Watermark badge ───────────────────────────────────────────────────
    badge_font = make_font(11, bold=True)
    badge = "  Generated by CheetahClaws /video  "
    bw = badge_font.getlength(badge)
    bx = VX + VW - int(bw) - 4
    d.rectangle([bx-2, VY+4, bx+int(bw)+2, VY+20], fill=(30, 30, 46))
    d.text((bx, VY+6), badge, font=badge_font, fill=MAUVE)


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


def step_line(icon, label, value="", done=False):
    tick = seg("✓ ", GREEN, True) if done else seg("  ", SUBTEXT)
    return [tick, seg(f"{icon} {label}: ", SUBTEXT), seg(value, CYAN if not done else GREEN)]


def wizard_prompt(question, answer="", cursor=False):
    cur = "█" if cursor else ""
    return [
        seg("  ┃ ", CYAN),
        seg(question, TEXT),
        seg(" → ", SUBTEXT),
        seg(answer + cur, YELLOW, True),
    ]


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


def pipeline_line(step, label, status="pending"):
    if status == "done":
        icon = seg("  ✓ ", GREEN, True)
        col  = GREEN
    elif status == "active":
        icon = seg("  ⠿ ", CYAN)
        col  = CYAN
    else:
        icon = seg("  · ", SUBTEXT)
        col  = SUBTEXT
    return [icon, seg(f"[{step}] ", col, status == "active"), seg(label, col)]


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms, False))

    TOPIC   = "The Rise of Artificial Intelligence"
    OUTFILE = "output_20260410_143022.mp4"

    # ── 0: Banner + empty prompt ─────────────────────────────────────────
    add(BANNER + [prompt_line(cursor=True)], 1000)

    # ── 1: Type /video ───────────────────────────────────────────────────
    cmd = "/video"
    for i in range(0, len(cmd) + 1, 2):
        add(BANNER + [prompt_line(cmd[:i], cursor=(i < len(cmd)))], 80)
    add(BANNER + [prompt_line(cmd)], 600)

    # ── 2: Wizard header ─────────────────────────────────────────────────
    base0 = BANNER + [prompt_line(cmd)]
    wizard_header = [
        None,
        [seg("  ╔══════════════════════════════════════════════╗", CYAN)],
        [seg("  ║  ", CYAN), seg("🎬  Video Content Factory", TEXT, True), seg("                  ║", CYAN)],
        [seg("  ╚══════════════════════════════════════════════╝", CYAN)],
        None,
    ]
    add(base0 + wizard_header, 800)

    # ── 3: Niche selection ────────────────────────────────────────────────
    niches = [
        "  1. Viral Tech Explainer     6. Documentary Style",
        "  2. AI & Future Insights     7. Educational Deep-Dive",
        "  3. Science Breakthrough     8. Motivational Story",
        "  4. Nature & Environment     9. News Summary",
        "  5. Health & Wellness       10. Custom Topic",
    ]
    niche_block = base0 + wizard_header + [
        [seg("  Content niche:", TEXT, True)],
    ] + [[seg(n, SUBTEXT)] for n in niches] + [
        None,
        wizard_prompt("Select niche", cursor=True),
    ]
    add(niche_block, 1000)

    # User types "2"
    add(base0 + wizard_header + [
        [seg("  Content niche:", TEXT, True)],
    ] + [[seg(n, SUBTEXT)] for n in niches] + [
        None,
        wizard_prompt("Select niche", "2"),
    ], 600)

    # ── 4: Topic input ────────────────────────────────────────────────────
    base1 = base0 + wizard_header + [
        step_line("🎯", "Niche", "AI & Future Insights", done=True),
        None,
    ]
    add(base1 + [wizard_prompt("Topic", cursor=True)], 700)

    topic_short = TOPIC
    for i in range(0, len(topic_short) + 1, 4):
        add(base1 + [wizard_prompt("Topic", topic_short[:i], cursor=(i < len(topic_short)))], 55)
    add(base1 + [wizard_prompt("Topic", topic_short)], 500)

    # ── 5: Format selection ───────────────────────────────────────────────
    base2 = base0 + wizard_header + [
        step_line("🎯", "Niche",  "AI & Future Insights", done=True),
        step_line("📌", "Topic",  TOPIC,                  done=True),
        None,
    ]
    add(base2 + [
        [seg("  Format:", TEXT, True)],
        [seg("    1. Landscape (16:9) — YouTube, LinkedIn", SUBTEXT)],
        [seg("    2. Short    (9:16) — Reels, TikTok, Shorts", SUBTEXT)],
        None,
        wizard_prompt("Format", cursor=True),
    ], 800)

    add(base2 + [
        [seg("  Format:", TEXT, True)],
        [seg("    1. Landscape (16:9) — YouTube, LinkedIn", SUBTEXT)],
        [seg("    2. Short    (9:16) — Reels, TikTok, Shorts", SUBTEXT)],
        None,
        wizard_prompt("Format", "1"),
    ], 500)

    # ── 6: Wizard complete → pipeline starts ──────────────────────────────
    base3 = base0 + wizard_header + [
        step_line("🎯", "Niche",    "AI & Future Insights",    done=True),
        step_line("📌", "Topic",    TOPIC,                     done=True),
        step_line("📐", "Format",   "Landscape 16:9",          done=True),
        step_line("🌐", "Language", "English (auto-detected)", done=True),
        step_line("💬", "Subtitles","Story text",              done=True),
        None,
    ]
    add(base3 + [info_line("Starting video pipeline...")], 800)

    # ── 7: Pipeline progress display ─────────────────────────────────────
    pipeline_steps = [
        ("1/5", "AI Story Generation"),
        ("2/5", "TTS Narration"),
        ("3/5", "Image Search"),
        ("4/5", "Subtitle Rendering"),
        ("5/5", "FFmpeg Assembly"),
    ]

    # Step 1 active
    add(base3 + [
        pipeline_line("1/5", "AI Story Generation",  "active"),
        pipeline_line("2/5", "TTS Narration",        "pending"),
        pipeline_line("3/5", "Image Search",         "pending"),
        pipeline_line("4/5", "Subtitle Rendering",   "pending"),
        pipeline_line("5/5", "FFmpeg Assembly",       "pending"),
    ], 700)

    # ── 8: Claude header + streaming story ───────────────────────────────
    story_lines = [
        "In the span of just a few decades, artificial intelligence",
        "has transformed from a distant dream into the backbone of",
        "modern civilization. From diagnosing cancer to composing",
        "symphonies, the machines we built are now teaching us what",
        "it means to think — and what it means to be human.",
        "",
        "But the story is only beginning. As models grow deeper and",
        "data grows richer, we stand at an inflection point. The",
        "next wave will not merely automate tasks — it will reshape",
        "entire industries, redefine creativity, and challenge our",
        "understanding of intelligence itself.",
    ]

    pipe_s1_active = [
        pipeline_line("1/5", "AI Story Generation", "active"),
        pipeline_line("2/5", "TTS Narration",       "pending"),
        pipeline_line("3/5", "Image Search",        "pending"),
        pipeline_line("4/5", "Subtitle Rendering",  "pending"),
        pipeline_line("5/5", "FFmpeg Assembly",      "pending"),
    ]

    add(base3 + pipe_s1_active + [None, claude_header()], 400)

    streamed = []
    for line in story_lines:
        streamed.append(text_line(line, 2))
        add(base3 + pipe_s1_active + [None, claude_header()] + streamed,
            65 if line else 25)

    story_done_block = base3 + [
        pipeline_line("1/5", "AI Story Generation", "done"),
        pipeline_line("2/5", "TTS Narration",       "pending"),
        pipeline_line("3/5", "Image Search",        "pending"),
        pipeline_line("4/5", "Subtitle Rendering",  "pending"),
        pipeline_line("5/5", "FFmpeg Assembly",      "pending"),
    ]

    add(story_done_block + [
        None,
        claude_header(),
    ] + [text_line(l, 2) for l in story_lines] + [claude_sep()], 700)

    # ── 9: TTS narration ─────────────────────────────────────────────────
    tts_block = base3 + [
        pipeline_line("1/5", "AI Story Generation", "done"),
        pipeline_line("2/5", "TTS Narration",       "active"),
        pipeline_line("3/5", "Image Search",        "pending"),
        pipeline_line("4/5", "Subtitle Rendering",  "pending"),
        pipeline_line("5/5", "FFmpeg Assembly",      "pending"),
    ]
    add(tts_block + [
        None,
        tool_line("🔊", "EdgeTTS", "en-US-AriaNeural  →  narration.mp3"),
    ], 600)
    add(tts_block + [
        None,
        tool_line("🔊", "EdgeTTS", "en-US-AriaNeural  →  narration.mp3"),
        tool_ok("→ 11 chunks synthesized  |  duration: 58.3 s"),
    ], 700)

    # ── 10: Image search ─────────────────────────────────────────────────
    img_block = base3 + [
        pipeline_line("1/5", "AI Story Generation", "done"),
        pipeline_line("2/5", "TTS Narration",       "done"),
        pipeline_line("3/5", "Image Search",        "active"),
        pipeline_line("4/5", "Subtitle Rendering",  "pending"),
        pipeline_line("5/5", "FFmpeg Assembly",      "pending"),
    ]
    add(img_block + [
        None,
        tool_line("🖼 ", "Pexels", "\"artificial intelligence technology\""),
    ], 500)
    add(img_block + [
        None,
        tool_line("🖼 ", "Pexels", "\"artificial intelligence technology\""),
        tool_ok("→ 6 images downloaded"),
        tool_line("🖼 ", "Pexels", "\"futuristic neural network\""),
    ], 500)
    add(img_block + [
        None,
        tool_line("🖼 ", "Pexels", "\"artificial intelligence technology\""),
        tool_ok("→ 6 images downloaded"),
        tool_line("🖼 ", "Pexels", "\"futuristic neural network\""),
        tool_ok("→ 6 images downloaded  |  12 total"),
    ], 700)

    # ── 11: Subtitle rendering ────────────────────────────────────────────
    sub_block = base3 + [
        pipeline_line("1/5", "AI Story Generation", "done"),
        pipeline_line("2/5", "TTS Narration",       "done"),
        pipeline_line("3/5", "Image Search",        "done"),
        pipeline_line("4/5", "Subtitle Rendering",  "active"),
        pipeline_line("5/5", "FFmpeg Assembly",      "pending"),
    ]
    add(sub_block + [
        None,
        tool_line("💬", "PIL", "rendering subtitles  →  NotoSansSC font"),
    ], 500)
    add(sub_block + [
        None,
        tool_line("💬", "PIL", "rendering subtitles  →  NotoSansSC font"),
        tool_ok("→ 23 subtitle frames burned"),
    ], 700)

    # ── 12: FFmpeg assembly ───────────────────────────────────────────────
    ffmpeg_block = base3 + [
        pipeline_line("1/5", "AI Story Generation", "done"),
        pipeline_line("2/5", "TTS Narration",       "done"),
        pipeline_line("3/5", "Image Search",        "done"),
        pipeline_line("4/5", "Subtitle Rendering",  "done"),
        pipeline_line("5/5", "FFmpeg Assembly",      "active"),
    ]
    add(ffmpeg_block + [
        None,
        tool_line("🎞 ", "ffmpeg", "images + audio + subtitles  →  mp4"),
    ], 600)
    add(ffmpeg_block + [
        None,
        tool_line("🎞 ", "ffmpeg", "images + audio + subtitles  →  mp4"),
        [seg("  ⠿  ", CYAN), seg("Encoding:  ", SUBTEXT),
         seg("████████████████████  100%", GREEN)],
    ], 800)

    # ── 13: Complete ──────────────────────────────────────────────────────
    done_block = base3 + [
        pipeline_line("1/5", "AI Story Generation", "done"),
        pipeline_line("2/5", "TTS Narration",       "done"),
        pipeline_line("3/5", "Image Search",        "done"),
        pipeline_line("4/5", "Subtitle Rendering",  "done"),
        pipeline_line("5/5", "FFmpeg Assembly",      "done"),
    ]
    add(done_block + [
        None,
        tool_line("🎞 ", "ffmpeg", "images + audio + subtitles  →  mp4"),
        [seg("  ✓  ", GREEN, True), seg("Encoding:  ", SUBTEXT),
         seg("████████████████████  100%", GREEN)],
        None,
        ok_line(f"Video saved: {OUTFILE}  (58 s  ·  1280×720  ·  24 fps)"),
    ], 1000)

    # ── 14: Claude summary response ───────────────────────────────────────
    summary = [
        "Your AI video is ready!",
        "",
        f"  📄  Script  : 11 sentences  |  ~290 words",
        f"  🔊  Audio   : narration.mp3  |  58.3 s  |  en-US-AriaNeural",
        f"  🖼   Images  : 12 frames from Pexels",
        f"  💬  Subs    : 23 subtitle overlays  (PIL / NotoSansSC)",
        f"  🎬  Output  : {OUTFILE}  |  1280×720  |  24 fps",
    ]
    add(done_block + [
        None,
        ok_line(f"Video saved: {OUTFILE}  (58 s  ·  1280×720  ·  24 fps)"),
        None,
        claude_header(),
    ] + [text_line(l, 2) for l in summary] + [claude_sep()], 900)

    # ── 15: Video preview (3 frames: fade-in hold) ───────────────────────
    scenes.append(([], 200, True))   # video_preview flag
    scenes.append(([], 2800, True))  # hold on preview
    scenes.append(([], 200, True))   # brief hold before returning

    # ── 16: New prompt ────────────────────────────────────────────────────
    add(done_block + [
        None,
        ok_line(f"Video saved: {OUTFILE}  (58 s  ·  1280×720  ·  24 fps)"),
        None,
        claude_header(),
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
    for i, (lines, ms, is_preview) in enumerate(scenes):
        rgb_frames.append(draw_frame(lines, video_preview=is_preview))
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
                       "..", "docs", "video_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
