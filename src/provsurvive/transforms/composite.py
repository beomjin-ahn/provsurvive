"""Composite family: a screenshot of the image as displayed on a phone, then cropped.

Pipeline:
1. Fit the image into a 1170-px-wide portrait display with Lanczos resampling (display
   scaling).
2. Composite it onto a synthetic UI frame (status bar, toolbar, background).
3. Capture as PNG.
4. Crop back to the image bounds plus a 2% margin, as people do before sharing.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from .base import Output, decode_rgb, encode, register

DISPLAY_W, DISPLAY_H = 1170, 2532
STATUS_H, TOOLBAR_H = 141, 250
UI_BG = (18, 18, 18)
UI_FG = (235, 235, 235)


def screenshot(
    img: Image.Image, margin: float = 0.02
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Returns (cropped screenshot, box of the image inside the crop)."""
    avail_h = DISPLAY_H - STATUS_H - TOOLBAR_H
    s = min(DISPLAY_W / img.width, avail_h / img.height)
    w, h = round(img.width * s), round(img.height * s)
    shown = img.resize((w, h), Image.LANCZOS)
    screen = Image.new("RGB", (DISPLAY_W, DISPLAY_H), UI_BG)
    x0 = (DISPLAY_W - w) // 2
    y0 = STATUS_H + (avail_h - h) // 2
    screen.paste(shown, (x0, y0))
    d = ImageDraw.Draw(screen)
    d.rectangle((0, 0, DISPLAY_W, STATUS_H), fill=UI_BG)
    d.text((60, 60), "9:41", fill=UI_FG)
    d.rectangle((0, DISPLAY_H - TOOLBAR_H, DISPLAY_W, DISPLAY_H), fill=UI_BG)
    for i in range(4):
        cx = 150 + i * 290
        d.ellipse((cx - 30, DISPLAY_H - 160, cx + 30, DISPLAY_H - 100), outline=UI_FG, width=4)
    mx, my = round(w * margin), round(h * margin)
    crop = (
        max(0, x0 - mx),
        max(0, y0 - my),
        min(DISPLAY_W, x0 + w + mx),
        min(DISPLAY_H, y0 + h + my),
    )
    out = screen.crop(crop)
    box = (x0 - crop[0], y0 - crop[1], x0 - crop[0] + w, y0 - crop[1] + h)
    return out, box


@register("screenshot", "composite", {"display": [DISPLAY_W, DISPLAY_H], "margin": 0.02})
def _screenshot(data, p, seed, ctx=None):
    out, box = screenshot(decode_rgb(data), p["margin"])
    return Output(encode(out, "png"), "png", alignment="approximate", geometry={"image_box": box})
