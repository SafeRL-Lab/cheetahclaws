#!/usr/bin/env python3
"""
Generate animated GIF demo of cheetahclaws shell escape (!command) using PIL.
Shows several ! commands interspersed with normal AI queries to highlight
the contrast: ! = direct shell, no AI; normal input = AI response.
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


def shell_prompt(cmd):
    """Prompt line with ! highlighted in yellow."""
    return [
        seg("[cheetahclaws] ", SUBTEXT),
        seg("» ", CYAN, True),
        seg("!", YELLOW, True),
        seg(cmd, TEXT),
    ]


def shell_output_line(t, color=TEXT):
    return [seg("  " + t, color)]


def shell_cmd_line(cmd):
    """The '  $ cmd' echo line shown when ! executes."""
    return [seg("  $ ", SUBTEXT), seg(cmd, YELLOW)]


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


def label_line(text):
    return [seg(f"  ── {text} ──", SUBTEXT)]


# ── Typing helper ─────────────────────────────────────────────────────────

def type_cmd(scenes_list, prefix_lines, full_cmd, hold_ms=500, char_ms=65, is_shell=False):
    """Animate typing full_cmd, appending to prefix_lines."""
    for i in range(0, len(full_cmd) + 1, 2):
        chunk = full_cmd[:i]
        if is_shell:
            line = [
                seg("[cheetahclaws] ", SUBTEXT),
                seg("» ", CYAN, True),
                seg("!", YELLOW, True),
                seg(chunk + ("█" if i < len(full_cmd) else ""), TEXT),
            ]
        else:
            line = [
                seg("[cheetahclaws] ", SUBTEXT),
                seg("» ", CYAN, True),
                seg(chunk + ("█" if i < len(full_cmd) else ""), TEXT),
            ]
        scenes_list.append((prefix_lines + [line], char_ms))
    scenes_list.append((prefix_lines + [
        shell_prompt(full_cmd) if is_shell else prompt_line(full_cmd)
    ], hold_ms))


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def add(lines, ms=120):
        scenes.append((lines, ms))

    # ── 0: Banner + label ─────────────────────────────────────────────────
    add(BANNER + [
        label_line("Shell Escape  —  prefix any command with ! to run it directly"),
        None,
        prompt_line(cursor=True),
    ], 1000)

    base = BANNER + [
        label_line("Shell Escape  —  prefix any command with ! to run it directly"),
        None,
    ]

    # ══════════════════════════════════════════════════════════════════════
    # Example 1: !git status
    # ══════════════════════════════════════════════════════════════════════
    type_cmd(scenes, base, "git status", hold_ms=400, is_shell=True)

    git_out = [
        "On branch main",
        "Your branch is up to date with 'origin/main'.",
        "",
        "Changes not staged for commit:",
        "  modified:   src/auth/auth.py",
        "  modified:   src/auth/middleware.py",
        "",
        "no changes added to commit",
    ]
    after_git = base + [shell_prompt("git status"), shell_cmd_line("git status")]
    for i, line in enumerate(git_out):
        after_git = after_git + [shell_output_line(
            line,
            RED if "modified" in line else (SUBTEXT if not line else TEXT)
        )]
        add(list(after_git), 60 if line else 25)
    add(list(after_git) + [None, prompt_line(cursor=True)], 900)

    # ══════════════════════════════════════════════════════════════════════
    # Example 2: !python --version
    # ══════════════════════════════════════════════════════════════════════
    block1 = base + [
        shell_prompt("git status"),
        shell_cmd_line("git status"),
    ] + [shell_output_line(l, RED if "modified" in l else (SUBTEXT if not l else TEXT))
         for l in git_out] + [None]

    type_cmd(scenes, block1, "python --version", hold_ms=400, is_shell=True)

    after_py = block1 + [
        shell_prompt("python --version"),
        shell_cmd_line("python --version"),
        shell_output_line("Python 3.11.7", GREEN),
        None,
        prompt_line(cursor=True),
    ]
    add(after_py, 900)

    # ══════════════════════════════════════════════════════════════════════
    # Example 3: !ls -la src/auth/
    # ══════════════════════════════════════════════════════════════════════
    block2 = block1 + [
        shell_prompt("python --version"),
        shell_cmd_line("python --version"),
        shell_output_line("Python 3.11.7", GREEN),
        None,
    ]

    type_cmd(scenes, block2, "ls -la src/auth/", hold_ms=400, is_shell=True)

    ls_out = [
        "total 24",
        "drwxr-xr-x  auth.py",
        "drwxr-xr-x  middleware.py",
        "drwxr-xr-x  utils.py",
        "drwxr-xr-x  __init__.py",
    ]
    after_ls = block2 + [
        shell_prompt("ls -la src/auth/"),
        shell_cmd_line("ls -la src/auth/"),
    ]
    for l in ls_out:
        after_ls = after_ls + [shell_output_line(l, CYAN if ".py" in l else SUBTEXT)]
        add(list(after_ls), 70)
    add(list(after_ls) + [None, prompt_line(cursor=True)], 900)

    # ══════════════════════════════════════════════════════════════════════
    # Contrast: normal AI query (no !)
    # ══════════════════════════════════════════════════════════════════════
    block3 = block2 + [
        shell_prompt("ls -la src/auth/"),
        shell_cmd_line("ls -la src/auth/"),
    ] + [shell_output_line(l, CYAN if ".py" in l else SUBTEXT) for l in ls_out] + [None]

    ai_msg = "What does auth.py do?"
    type_cmd(scenes, block3, ai_msg, hold_ms=400, is_shell=False)

    # AI responds
    ai_response = [
        "auth.py handles JWT authentication:",
        "",
        "  • create_access_token() — signs a JWT with HS256",
        "  • verify_token()        — validates and decodes Bearer tokens",
        "  • Token expiry: 15 min access / 7 day refresh",
    ]
    add(block3 + [prompt_line(ai_msg), None, claude_header()], 500)

    streamed = []
    for line in ai_response:
        streamed.append(text_line(line, 2))
        add(block3 + [prompt_line(ai_msg), None, claude_header()] + streamed,
            70 if line else 25)

    add(block3 + [prompt_line(ai_msg), None, claude_header()] +
        [text_line(l, 2) for l in ai_response] + [claude_sep()], 900)

    # ── Final prompt ──────────────────────────────────────────────────────
    add(block3 + [prompt_line(ai_msg), None, claude_header()] +
        [text_line(l, 2) for l in ai_response] + [claude_sep()] +
        [None, prompt_line(cursor=True)], 2500)

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
                       "..", "docs", "shell_escape_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
