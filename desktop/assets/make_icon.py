import math, numpy as np
from PIL import Image, ImageDraw, ImageFilter

S = 2048  # supersample, downscale to 1024 at the end
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

# ── rounded-square background with a subtle vertical gradient ──
ys = np.linspace(0, 1, S)[:, None]
top = np.array([26, 34, 50]); bot = np.array([10, 14, 20])
col = (top[None, :] * (1 - ys) + bot[None, :] * ys)        # (S,3)
grad = np.broadcast_to(col[:, None, :], (S, S, 3)).astype("uint8")
gimg = Image.fromarray(grad, "RGB").convert("RGBA")
margin = int(S * 0.055); radius = int(S * 0.225)
mask = Image.new("L", (S, S), 0)
ImageDraw.Draw(mask).rounded_rectangle([margin, margin, S - margin, S - margin],
                                       radius, fill=255)
bg = Image.new("RGBA", (S, S), (0, 0, 0, 0))
bg.paste(gimg, (0, 0), mask)
img = Image.alpha_composite(img, bg)

# ── three tapered, slightly-curved claw slashes ──
def blade(cx, cy, length, maxw, curve, angle_deg):
    a = math.radians(angle_deg); L, R = [], []
    for i in range(61):
        t = i / 60
        px = curve * math.sin(math.pi * t)
        py = (t - 0.5) * length
        w = maxw * (math.sin(math.pi * t) ** 0.65)          # pointed ends, fat middle
        for xx, store in ((px - w / 2, L), (px + w / 2, R)):
            rx = xx * math.cos(a) - py * math.sin(a)
            ry = xx * math.sin(a) + py * math.cos(a)
            store.append((cx + rx, cy + ry))
    return L + R[::-1]

claw = Image.new("RGBA", (S, S), (0, 0, 0, 0))
cd = ImageDraw.Draw(claw)
cx0, cy0 = S * 0.5, S * 0.52
specs = [(-0.145, 0.86, 16), (0.0, 1.0, 9), (0.145, 0.86, 2)]   # (x-offset frac, len-scale, tilt)
for dx, ls, tilt in specs:
    poly = blade(cx0 + dx * S, cy0, S * 0.52 * ls, S * 0.115, S * 0.05, tilt)
    cd.polygon(poly, fill=(240, 136, 62, 255))                 # brand orange
    # inner highlight
    poly2 = blade(cx0 + dx * S, cy0 - S * 0.015, S * 0.46 * ls, S * 0.055, S * 0.05, tilt)
    cd.polygon(poly2, fill=(255, 190, 120, 255))

# soft glow under the claws
glow = claw.filter(ImageFilter.GaussianBlur(S * 0.012))
img = Image.alpha_composite(img, glow)
img = Image.alpha_composite(img, claw)
# clip claws to the rounded square
out = Image.new("RGBA", (S, S), (0, 0, 0, 0))
out.paste(img, (0, 0), mask)

icon = out.resize((1024, 1024), Image.LANCZOS)
icon.save("assets/icon.png")
icon.resize((256, 256), Image.LANCZOS).save("/tmp/icon_preview.png")
print("wrote assets/icon.png (1024x1024) + /tmp/icon_preview.png")
