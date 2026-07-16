#!/usr/bin/env python3
"""
Generate animated GIF demo of the CheetahClaws Web UI (`/web` chat) using PIL.

Simulates the web chat interface:
  - Sidebar with sessions + search + "+ New"
  - Topbar with title, status dot, theme/settings icons
  - User message bubbles (right)
  - Assistant markdown bubbles (left) — headers, bold, bullets, code
  - Tool cards (Running → Done) with colored border-left + status badge
  - Approval card (Allow / Deny)
  - Activity indicator (spinner + label + detail + progress bar)
  - Input area with textarea + Send / Stop button
"""
from PIL import Image, ImageDraw, ImageFont
import os
import copy
import math

# ── Dark palette from web/chat.html ───────────────────────────────────────
BG          = (11,  11,  14)
SURFACE     = (17,  17,  22)
PANEL       = (24,  24,  31)
PANEL2      = (30,  30,  40)
BORDER      = (40,  40,  54)
BORDER_DIM  = (28,  28,  36)
TEXT        = (220, 220, 232)
TEXT_DIM    = (148, 148, 174)
TEXT_MUTED  = (110, 110, 136)
ACCENT      = (232, 160, 69)
ACCENT_DIM  = (48,  34,  18)
GREEN       = (74,  222, 128)
GREEN_DIM   = (24,  54,  32)
RED         = (248, 113, 113)
RED_DIM     = (54,  24,  28)
BLUE        = (96,  165, 250)
BLUE_DIM    = (22,  36,  62)
YELLOW      = (251, 191,  36)
BLACK       = (0,   0,   0)
WHITE       = (255, 255, 255)

W, H = 960, 720
SIDEBAR_W = 220
TOPBAR_H = 46
INPUT_H = 74

FONT_REG    = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BLD    = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_MONO   = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_MONO_B = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"


def _ft(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


F10B  = _ft(FONT_BLD,    10)
F11   = _ft(FONT_REG,    11)
F11M  = _ft(FONT_MONO,   11)
F12   = _ft(FONT_REG,    12)
F12B  = _ft(FONT_BLD,    12)
F12M  = _ft(FONT_MONO,   12)
F12MB = _ft(FONT_MONO_B, 12)
F13   = _ft(FONT_REG,    13)
F13B  = _ft(FONT_BLD,    13)
F14   = _ft(FONT_REG,    14)
F14B  = _ft(FONT_BLD,    14)
F15B  = _ft(FONT_BLD,    15)
F16B  = _ft(FONT_BLD,    16)


# ── Helpers ───────────────────────────────────────────────────────────────

def rr(d, box, r, fill=None, outline=None, width=1):
    """Rounded rectangle."""
    d.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def wrap(text, font, max_w):
    words = text.split(" ")
    lines, cur = [], ""
    for w in words:
        t = (cur + (" " if cur else "") + w)
        if font.getlength(t) <= max_w:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def draw_spinner(d, cx, cy, r, frame, color=BLUE, dim=BLUE_DIM):
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=dim, width=2)
    start = (frame * 38) % 360
    d.arc([cx - r, cy - r, cx + r, cy + r], start, start + 110, fill=color, width=2)


# ── Sidebar ──────────────────────────────────────────────────────────────

SESSIONS = [
    ("Refactor web API routes",         "2m ago", "busy"),
    ("Brainstorm: ML paper ablations",  "1h ago", "idle"),
    ("Fix TypeScript strict errors",    "3h ago", "idle"),
    ("Deploy script review",            "yest.",  "idle"),
    ("Memory module rewrite",           "2d ago", "idle"),
    ("Plugin system design",            "4d ago", "idle"),
]


def draw_sidebar(d, active_idx=0, search_text=""):
    # Background
    d.rectangle([0, 0, SIDEBAR_W, H], fill=SURFACE)
    d.line([SIDEBAR_W, 0, SIDEBAR_W, H], fill=BORDER, width=1)

    # Header
    d.rectangle([0, 0, SIDEBAR_W, 54], fill=SURFACE)
    d.line([0, 54, SIDEBAR_W, 54], fill=BORDER, width=1)
    d.text((16, 18), "CheetahClaws", font=F15B, fill=ACCENT)
    # + New button
    btn_x, btn_y, btn_w, btn_h = SIDEBAR_W - 62, 14, 50, 26
    rr(d, [btn_x, btn_y, btn_x + btn_w, btn_y + btn_h], 6, fill=ACCENT)
    d.text((btn_x + 9, btn_y + 6), "+ New", font=F12B, fill=BLACK)

    # Search
    sy = 64
    rr(d, [10, sy, SIDEBAR_W - 10, sy + 28], 6, fill=PANEL, outline=BORDER, width=1)
    placeholder = search_text or "Search sessions..."
    placeholder_color = TEXT if search_text else TEXT_MUTED
    d.text((18, sy + 7), placeholder, font=F12, fill=placeholder_color)

    # Session list
    ly = 104
    for i, (title, when, state) in enumerate(SESSIONS):
        row_h = 48
        is_active = (i == active_idx)
        if is_active:
            rr(d, [8, ly, SIDEBAR_W - 8, ly + row_h], 6,
               fill=PANEL, outline=ACCENT, width=1)
        # Status dot
        dot_color = GREEN if state == "busy" else TEXT_MUTED
        d.ellipse([18, ly + 10, 24, ly + 16], fill=dot_color)
        d.text((32, ly + 6), title[:24], font=F12B if is_active else F12,
               fill=TEXT if is_active else TEXT_DIM)
        d.text((32, ly + 26), when, font=F11, fill=TEXT_MUTED)
        ly += row_h + 4

    # Footer
    d.line([0, H - 42, SIDEBAR_W, H - 42], fill=BORDER, width=1)
    d.text((16, H - 30), "shangding", font=F12, fill=TEXT_DIM)
    d.text((SIDEBAR_W - 70, H - 30), "Sign out", font=F12, fill=ACCENT)


# ── Topbar ────────────────────────────────────────────────────────────────

def draw_topbar(d, status_kind, status_text):
    d.rectangle([SIDEBAR_W, 0, W, TOPBAR_H], fill=SURFACE)
    d.line([SIDEBAR_W, TOPBAR_H, W, TOPBAR_H], fill=BORDER, width=1)
    d.text((SIDEBAR_W + 20, 14), "Chat", font=F15B, fill=ACCENT)

    # Status on the right
    rx = W - 170
    dot_color = {"idle": GREEN, "busy": YELLOW, "off": RED}.get(status_kind, GREEN)
    d.ellipse([rx, 19, rx + 8, 27], fill=dot_color)
    d.text((rx + 14, 14), status_text, font=F12, fill=TEXT_DIM)

    # Theme + settings buttons
    for i, ch in enumerate(["☾", "⚙"]):
        bx = W - 72 + i * 30
        rr(d, [bx, 11, bx + 24, 35], 5, outline=BORDER, width=1)
        d.text((bx + 6, 14), ch, font=F14, fill=TEXT_DIM)


# ── Input area ────────────────────────────────────────────────────────────

def draw_input(d, text, sending=False, cursor=False):
    iy = H - INPUT_H
    d.rectangle([SIDEBAR_W, iy, W, H], fill=SURFACE)
    d.line([SIDEBAR_W, iy, W, iy], fill=BORDER, width=1)

    pad = 14
    ta_x0 = SIDEBAR_W + pad
    ta_y0 = iy + 14
    btn_w = 78
    ta_x1 = W - pad - btn_w - 8
    ta_y1 = H - 14
    border_col = ACCENT if (cursor or text) else BORDER
    rr(d, [ta_x0, ta_y0, ta_x1, ta_y1], 10, fill=PANEL, outline=border_col, width=1)
    if text:
        for i, line in enumerate(text.split("\n")):
            d.text((ta_x0 + 12, ta_y0 + 10 + i * 18), line, font=F13, fill=TEXT)
        if cursor:
            last = text.split("\n")[-1]
            cx = ta_x0 + 12 + F13.getlength(last)
            cy = ta_y0 + 10 + (text.count("\n")) * 18
            d.rectangle([cx, cy, cx + 2, cy + 16], fill=ACCENT)
    else:
        d.text((ta_x0 + 12, ta_y0 + 10),
               "Type a message... (Shift+Enter for newline)",
               font=F13, fill=TEXT_MUTED)

    # Send / Stop button
    bx0 = ta_x1 + 8
    btn_color = RED if sending else ACCENT
    btn_text = "Stop" if sending else "Send"
    rr(d, [bx0, ta_y0, bx0 + btn_w, ta_y1], 10, fill=btn_color)
    tw = F13B.getlength(btn_text)
    d.text((bx0 + (btn_w - tw) / 2, ta_y0 + 11), btn_text, font=F13B, fill=BLACK)


# ── Messages ──────────────────────────────────────────────────────────────

MSG_X0 = SIDEBAR_W + 20
MSG_X1 = W - 20


def draw_user_bubble(d, y, text):
    max_w = 500
    lines = wrap(text, F13, max_w - 28)
    line_h = 18
    bh = 10 + len(lines) * line_h + 10
    bw = max(F13.getlength(ln) for ln in lines) + 28
    bw = min(bw, max_w)
    x1 = MSG_X1
    x0 = x1 - bw
    # role tag
    d.text((x1 - F10B.getlength("YOU") - 4, y), "YOU", font=F10B, fill=TEXT_MUTED)
    y += 14
    rr(d, [x0, y, x1, y + bh], 10, fill=PANEL, outline=BORDER, width=1)
    # Speech-bubble tail hint: flatten bottom-right corner
    d.rectangle([x1 - 6, y + bh - 6, x1, y + bh], fill=PANEL)
    d.line([x1, y + bh, x1, y + bh - 6], fill=BORDER)
    d.line([x1, y + bh, x1 - 6, y + bh], fill=BORDER)
    for i, ln in enumerate(lines):
        d.text((x0 + 14, y + 10 + i * line_h), ln, font=F13, fill=TEXT)
    return y + bh + 18


# Markdown-ish line types: ("h2", text), ("p", text), ("b", text), ("li", text),
# ("code", text), ("blank", None)
def draw_assistant_bubble(d, y, md_lines):
    # role tag
    d.text((MSG_X0, y), "ASSISTANT", font=F10B, fill=TEXT_MUTED)
    y += 14
    max_w = 640
    line_h = 19
    for kind, txt in md_lines:
        if kind == "blank":
            y += 8
            continue
        if kind == "h2":
            d.text((MSG_X0, y), txt, font=F15B, fill=TEXT)
            y += 24
            continue
        if kind == "h3":
            d.text((MSG_X0, y), txt, font=F14B, fill=TEXT)
            y += 22
            continue
        if kind == "code":
            box_h = 14 + 16 * len(txt) + 6
            rr(d, [MSG_X0, y, MSG_X0 + max_w, y + box_h], 6,
               fill=PANEL, outline=BORDER, width=1)
            for i, cl in enumerate(txt):
                d.text((MSG_X0 + 12, y + 8 + i * 16), cl, font=F11M, fill=TEXT)
            y += box_h + 6
            continue
        if kind == "li":
            d.ellipse([MSG_X0 + 4, y + 7, MSG_X0 + 9, y + 12], fill=TEXT_DIM)
            segs, is_bold = [], False
            for p in txt.split("**"):
                if p:
                    segs.append((p, is_bold))
                is_bold = not is_bold
            x = MSG_X0 + 18
            for p, b in segs:
                f = F13B if b else F13
                d.text((x, y), p, font=f, fill=TEXT)
                x += f.getlength(p)
            y += line_h + 2
            continue
        # paragraph: support **bold** inline via a simple split on **
        segs = []
        parts = txt.split("**")
        is_bold = False
        for p in parts:
            if p:
                segs.append((p, is_bold))
            is_bold = not is_bold
        # naive flow: assume short enough to fit one line; wrap plainly if not
        if sum(F13.getlength(p) + (0 if not b else 0) for p, b in segs) <= max_w:
            x = MSG_X0
            for p, b in segs:
                f = F13B if b else F13
                d.text((x, y), p, font=f, fill=TEXT)
                x += f.getlength(p)
            y += line_h
        else:
            # fallback: just wrap without bold
            for ln in wrap(txt.replace("**", ""), F13, max_w):
                d.text((MSG_X0, y), ln, font=F13, fill=TEXT)
                y += line_h
    return y + 10


def draw_tool_card(d, y, name, args, state, result=None, spinner_frame=0):
    max_w = 540
    title_h = 34
    body_h = 0
    # body lines
    body_lines = []
    if state == "done":
        body_lines.append(("arg", f"{name}({args})"))
        if result:
            body_lines.append(("result", result))
    elif state == "running":
        body_lines.append(("arg", f"{name}({args})"))
    elif state == "denied":
        body_lines.append(("arg", f"{name}({args})"))
        body_lines.append(("denied", "Denied by user"))

    if body_lines:
        body_h = 10 + 18 * len(body_lines) + 6

    card_h = title_h + body_h
    x0 = MSG_X0
    x1 = x0 + max_w
    # Outer card
    rr(d, [x0, y, x1, y + card_h], 6, fill=SURFACE, outline=BORDER, width=1)
    # Colored left stripe
    stripe = {"running": BLUE, "done": GREEN, "denied": RED}[state]
    d.rectangle([x0, y + 1, x0 + 3, y + card_h - 1], fill=stripe)

    # Title row
    # spinner / check / X
    ix = x0 + 16
    iy = y + 11
    if state == "running":
        draw_spinner(d, ix + 6, iy + 6, 6, spinner_frame, color=BLUE, dim=BLUE_DIM)
    elif state == "done":
        d.ellipse([ix, iy, ix + 14, iy + 14], fill=GREEN_DIM, outline=GREEN)
        d.text((ix + 3, iy - 2), "✓", font=F11M, fill=GREEN)
    else:
        d.ellipse([ix, iy, ix + 14, iy + 14], fill=RED_DIM, outline=RED)
        d.text((ix + 3, iy - 2), "✕", font=F11M, fill=RED)

    d.text((ix + 24, iy - 1), name, font=F12MB, fill=TEXT)
    # badge
    badges = {"running": ("RUNNING", BLUE, BLUE_DIM),
              "done":    ("DONE",    GREEN, GREEN_DIM),
              "denied":  ("DENIED",  RED,   RED_DIM)}
    btxt, bcol, bbg = badges[state]
    bw = F10B.getlength(btxt) + 14
    bx = x1 - bw - 12
    rr(d, [bx, y + 9, bx + bw, y + 24], 3, fill=bbg)
    d.text((bx + 7, y + 11), btxt, font=F10B, fill=bcol)

    # Body
    by = y + title_h
    d.line([x0 + 1, by, x1 - 1, by], fill=BORDER)
    for kind, text in body_lines:
        by += 10
        if kind == "arg":
            d.text((x0 + 16, by - 2), text, font=F11M, fill=TEXT_DIM)
        elif kind == "result":
            d.text((x0 + 16, by - 2), text, font=F11M, fill=GREEN)
        elif kind == "denied":
            d.text((x0 + 16, by - 2), text, font=F11M, fill=RED)
        by += 8
    return y + card_h + 10


def draw_approval_card(d, y, tool, args, resolved=None):
    max_w = 520
    x0 = MSG_X0
    x1 = x0 + max_w
    hdr_h = 28
    desc_h = 52
    btn_h = 36
    card_h = hdr_h + desc_h + btn_h + 8
    rr(d, [x0, y, x1, y + card_h], 10,
       fill=ACCENT_DIM, outline=ACCENT, width=1)

    # Header
    d.text((x0 + 16, y + 10), "⚠  APPROVAL REQUIRED", font=F12B, fill=ACCENT)

    # Description
    dy = y + hdr_h + 8
    d.text((x0 + 16, dy),
           f"Allow tool {tool} to modify:", font=F13, fill=TEXT_DIM)
    d.text((x0 + 16, dy + 20), args, font=F12M, fill=TEXT)

    # Buttons
    by = y + hdr_h + desc_h + 6
    allow_w = 80
    deny_w = 70
    if resolved == "allow":
        rr(d, [x0 + 16, by, x0 + 16 + allow_w, by + 28], 6, fill=GREEN)
        d.text((x0 + 40, by + 7), "Allow", font=F12B, fill=BLACK)
        rr(d, [x0 + 16 + allow_w + 8, by, x0 + 16 + allow_w + 8 + deny_w, by + 28], 6,
           fill=PANEL, outline=BORDER, width=1)
        d.text((x0 + 16 + allow_w + 8 + 22, by + 7), "Deny", font=F12, fill=TEXT_MUTED)
        d.text((x0 + 16 + allow_w + 8 + deny_w + 14, by + 10),
               "✓ Allowed", font=F11, fill=GREEN)
    elif resolved == "deny":
        rr(d, [x0 + 16, by, x0 + 16 + allow_w, by + 28], 6,
           fill=PANEL, outline=BORDER, width=1)
        d.text((x0 + 40, by + 7), "Allow", font=F12, fill=TEXT_MUTED)
        rr(d, [x0 + 16 + allow_w + 8, by, x0 + 16 + allow_w + 8 + deny_w, by + 28], 6,
           fill=RED)
        d.text((x0 + 16 + allow_w + 8 + 20, by + 7), "Deny", font=F12B, fill=BLACK)
    else:
        rr(d, [x0 + 16, by, x0 + 16 + allow_w, by + 28], 6, fill=GREEN)
        d.text((x0 + 40, by + 7), "Allow", font=F12B, fill=BLACK)
        rr(d, [x0 + 16 + allow_w + 8, by, x0 + 16 + allow_w + 8 + deny_w, by + 28], 6,
           fill=RED)
        d.text((x0 + 16 + allow_w + 8 + 20, by + 7), "Deny", font=F12B, fill=BLACK)

    return y + card_h + 10


def draw_activity(d, y, label, detail, kind, spinner_frame):
    max_w = 520
    x0 = MSG_X0
    x1 = x0 + max_w
    h = 40
    border_col = BLUE if kind == "thinking" else ACCENT
    bg_col = BLUE_DIM if kind == "thinking" else ACCENT_DIM
    rr(d, [x0, y, x1, y + h], 6, fill=bg_col, outline=border_col, width=1)
    draw_spinner(d, x0 + 22, y + 20, 8, spinner_frame,
                 color=border_col, dim=bg_col)
    d.text((x0 + 44, y + 12), label, font=F13B, fill=border_col)
    d.text((x0 + 44 + F13B.getlength(label) + 10, y + 13),
           detail, font=F11M, fill=TEXT_MUTED)
    # animated progress bar
    px0 = x0 + 260
    px1 = x1 - 14
    d.rectangle([px0, y + 19, px1, y + 22], fill=BORDER)
    p_w = px1 - px0
    seg_w = int(p_w * 0.28)
    off = int(((spinner_frame * 20) % (p_w + seg_w)) - seg_w)
    bar_x0 = max(px0, px0 + off)
    bar_x1 = min(px1, px0 + off + seg_w)
    if bar_x1 > bar_x0:
        d.rectangle([bar_x0, y + 19, bar_x1, y + 22], fill=border_col)
    return y + h + 10


# ── Scene renderer ───────────────────────────────────────────────────────

MSG_Y_TOP = TOPBAR_H + 18
MSG_Y_BOTTOM = H - INPUT_H - 10


def render_scene(state):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # Chat background
    d.rectangle([SIDEBAR_W, TOPBAR_H, W, H - INPUT_H], fill=BG)

    # Paint messages first (bottom-aligned-ish by measuring total height).
    # Simplest: top-anchor; drop oldest if overflow.
    msgs = state["msgs"]
    # Measure heights to implement "scroll to bottom" behavior:
    # Just show the tail that fits.
    estimated = []
    for m in msgs:
        estimated.append(_estimate_h(m))
    total_h = sum(estimated)
    available = MSG_Y_BOTTOM - MSG_Y_TOP
    start = 0
    while total_h > available and start < len(msgs):
        total_h -= estimated[start]
        start += 1
    y = MSG_Y_TOP
    if total_h < available:
        # center-ish: keep top aligned
        pass

    for m in msgs[start:]:
        y = _draw_msg(d, y, m, state.get("spinner_frame", 0))

    # Empty state placeholder
    if not msgs:
        txt = "Send a message to start a conversation."
        tw = F13.getlength(txt)
        d.text((SIDEBAR_W + (W - SIDEBAR_W - tw) / 2, H / 2 - 20),
               txt, font=F13, fill=TEXT_MUTED)

    # Overlays
    draw_sidebar(d, active_idx=state.get("active_session", 0))
    draw_topbar(d,
                state.get("status_kind", "idle"),
                state.get("status_text", "idle"))
    draw_input(d,
               state.get("input_text", ""),
               sending=state.get("sending", False),
               cursor=state.get("input_cursor", False))
    return img


def _estimate_h(m):
    t = m["t"]
    if t == "user":
        max_w = 500
        lines = wrap(m["text"], F13, max_w - 28)
        return 14 + 10 + 18 * len(lines) + 10 + 18
    if t == "assistant":
        h = 14
        for kind, txt in m["lines"]:
            if kind == "blank":
                h += 8
            elif kind == "h2":
                h += 24
            elif kind == "h3":
                h += 22
            elif kind == "code":
                h += 14 + 16 * len(txt) + 6 + 6
            elif kind == "li":
                lns = max(1, len(wrap(txt, F13, 640 - 24)))
                h += 19 * lns + 2
            else:
                h += 19
        return h + 10
    if t == "tool":
        lines = 1 + (1 if m.get("result") else 0) + (1 if m.get("state") == "denied" else 0)
        return 34 + 10 + 18 * lines + 6 + 10
    if t == "approval":
        return 28 + 52 + 36 + 8 + 10
    if t == "activity":
        return 40 + 10
    return 0


def _draw_msg(d, y, m, spinner_frame):
    t = m["t"]
    if t == "user":
        return draw_user_bubble(d, y, m["text"])
    if t == "assistant":
        return draw_assistant_bubble(d, y, m["lines"])
    if t == "tool":
        return draw_tool_card(d, y, m["name"], m["args"], m["state"],
                              result=m.get("result"),
                              spinner_frame=spinner_frame)
    if t == "approval":
        return draw_approval_card(d, y, m["tool"], m["args"],
                                  resolved=m.get("resolved"))
    if t == "activity":
        return draw_activity(d, y, m["label"], m["detail"], m["kind"],
                             spinner_frame)
    return y


# ── Scene builder ─────────────────────────────────────────────────────────

def build_scenes():
    scenes = []

    def push(state, ms=120):
        scenes.append((copy.deepcopy(state), ms))

    state = {
        "msgs": [],
        "input_text": "",
        "input_cursor": True,
        "sending": False,
        "status_kind": "idle",
        "status_text": "idle",
        "active_session": 0,
        "spinner_frame": 0,
    }

    # ── 1: Empty UI with cursor blink ──────────────────────────────────
    push(state, 1100)

    # ── 2: User types message ──────────────────────────────────────────
    msg = "Read web/server.py and explain the auth flow"
    for i in range(1, len(msg) + 1, 2):
        state["input_text"] = msg[:i]
        push(state, 55)
    state["input_text"] = msg
    push(state, 500)

    # ── 3: Send → user bubble + activity ───────────────────────────────
    state["msgs"].append({"t": "user", "text": msg})
    state["input_text"] = ""
    state["input_cursor"] = False
    state["sending"] = True
    state["status_kind"] = "busy"
    state["status_text"] = "thinking"
    state["msgs"].append({
        "t": "activity",
        "label": "Thinking",
        "detail": "claude-sonnet-4-6",
        "kind": "thinking",
    })
    for f in range(8):
        state["spinner_frame"] = f
        push(state, 90)

    # ── 4: Tool call — Read(web/server.py) running ─────────────────────
    state["msgs"].pop()  # remove activity
    state["msgs"].append({
        "t": "tool",
        "name": "Read",
        "args": "web/server.py",
        "state": "running",
    })
    state["status_text"] = "tool: Read"
    for f in range(6):
        state["spinner_frame"] = f + 8
        push(state, 100)

    # ── 5: Tool done ──────────────────────────────────────────────────
    state["msgs"][-1] = {
        "t": "tool",
        "name": "Read",
        "args": "web/server.py",
        "state": "done",
        "result": "→ 1761 lines (54.2 KB)",
    }
    push(state, 600)

    # ── 6: Assistant begins — stream header + first paragraph ──────────
    state["msgs"].append({"t": "activity",
                          "label": "Generating",
                          "detail": "streaming markdown",
                          "kind": "thinking"})
    for f in range(5):
        state["spinner_frame"] = f + 14
        push(state, 90)
    state["msgs"].pop()

    assistant_full = [
        ("h2",    "Auth flow in web/server.py"),
        ("blank", None),
        ("p",     "The server uses **cookie-based sessions** backed by SQLite."),
        ("p",     "The relevant pieces are:"),
        ("li",    "**POST /api/login** — verifies bcrypt hash, sets HttpOnly cookie"),
        ("li",    "**auth.py** — issues signed tokens (HMAC-SHA256)"),
        ("li",    "**require_auth** — decorator that 401s on missing/invalid cookie"),
        ("blank", None),
        ("h3",    "Token format"),
        ("code",  ["{",
                   "  \"user_id\": 1,",
                   "  \"exp\": 1712345678,",
                   "  \"sig\": \"<hmac-sha256>\"",
                   "}"]),
    ]
    # Stream one entry at a time
    state["msgs"].append({"t": "assistant", "lines": []})
    for item in assistant_full:
        state["msgs"][-1]["lines"] = state["msgs"][-1]["lines"] + [item]
        push(state, 200 if item[0] == "p" or item[0] == "h2" else 130)

    push(state, 500)

    # ── 7: Another tool — Edit with approval ───────────────────────────
    state["msgs"].append({
        "t": "approval",
        "tool": "Edit",
        "args": "web/auth.py  (add token rotation on each request)",
    })
    state["status_text"] = "awaiting approval"
    push(state, 1100)

    # Highlight Allow (simulated click)
    state["msgs"][-1]["resolved"] = "allow"
    push(state, 600)

    # ── 8: Edit tool running ──────────────────────────────────────────
    state["msgs"].append({
        "t": "tool",
        "name": "Edit",
        "args": "web/auth.py",
        "state": "running",
    })
    state["status_text"] = "tool: Edit"
    for f in range(5):
        state["spinner_frame"] = f + 22
        push(state, 100)

    # ── 9: Edit done ──────────────────────────────────────────────────
    state["msgs"][-1] = {
        "t": "tool",
        "name": "Edit",
        "args": "web/auth.py",
        "state": "done",
        "result": "→ 1 hunk applied (+14 / -3 lines)",
    }
    push(state, 500)

    # ── 10: Final assistant summary ───────────────────────────────────
    final = [
        ("blank", None),
        ("p",     "Added **sliding-window rotation**: each authenticated request"),
        ("p",     "re-issues the cookie with a fresh 24h `exp`. Idle users are"),
        ("p",     "logged out after one full expiry window."),
    ]
    state["msgs"].append({"t": "assistant", "lines": []})
    for item in final:
        state["msgs"][-1]["lines"] = state["msgs"][-1]["lines"] + [item]
        push(state, 160)

    # ── 11: Back to idle ──────────────────────────────────────────────
    state["sending"] = False
    state["status_kind"] = "idle"
    state["status_text"] = "idle"
    state["input_cursor"] = True
    push(state, 2200)

    return scenes


# ── Palette + render ──────────────────────────────────────────────────────

def _build_palette():
    theme = [
        BG, SURFACE, PANEL, PANEL2, BORDER, BORDER_DIM,
        TEXT, TEXT_DIM, TEXT_MUTED,
        ACCENT, ACCENT_DIM,
        GREEN, GREEN_DIM, RED, RED_DIM, BLUE, BLUE_DIM,
        YELLOW, BLACK, WHITE,
        (40, 40, 60), (60, 60, 80), (90, 90, 110),
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
    for i, (state, ms) in enumerate(scenes):
        rgb_frames.append(render_scene(state))
        durations.append(ms)
        if i % 20 == 0:
            print(f"    {i}/{len(scenes)}...")

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
                       "..", "docs", "web_demo.gif")
    render_gif(out)
    print(f"\nGIF saved: {out}")
