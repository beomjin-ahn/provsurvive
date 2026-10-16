"""Content edits that complete the C2PA action coverage.

Each edit stands for a pre-defined C2PA 2.4 action that the first suite lacked:
``c2pa.enhanced`` (sharpen, denoise), ``c2pa.filtered`` (blur), ``c2pa.drawing`` (brush
strokes), ``c2pa.addedText`` (caption), ``c2pa.placed`` (sticker) and ``c2pa.deleted``
(non-generative removal). The removal edits reuse the generative inpainting masks (same seed
key), so classical and diffusion fills can be compared on identical regions.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .base import Output, decode_rgb, encode, reencode, register
from .generative import ellipse_mask, ellipse_params
from .jumbf import sniff_format


@register("sharpen", "enhance", {"radius": 2.0, "percent": 120, "threshold": 2})
def _sharpen(data, p, seed, ctx=None):
    f = ImageFilter.UnsharpMask(p["radius"], p["percent"], p["threshold"])
    return reencode(data, decode_rgb(data).filter(f))


@register("denoise", "enhance", {"h": 5, "template": 7, "search": 21})
def _denoise(data, p, seed, ctx=None):
    import cv2

    arr = np.asarray(decode_rgb(data))[:, :, ::-1].copy()
    out = cv2.fastNlMeansDenoisingColored(arr, None, p["h"], p["h"], p["template"], p["search"])
    return reencode(data, Image.fromarray(out[:, :, ::-1]))


@register("blur", "filter", {"sigma": 1.5})
def _blur(data, p, seed, ctx=None):
    return reencode(data, decode_rgb(data).filter(ImageFilter.GaussianBlur(p["sigma"])))


@register("brush_strokes", "draw", {"strokes": 3, "width_frac": 0.012})
def _brush(data, p, seed, ctx=None):
    img = decode_rgb(data).copy()
    w, h = img.size
    rng = np.random.default_rng(seed)
    d = ImageDraw.Draw(img)
    width = max(2, round(p["width_frac"] * min(w, h)))
    for _ in range(p["strokes"]):
        n = 6
        x0, y0 = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        pts = [(x0, y0)]
        for _ in range(n):
            x0 = float(np.clip(x0 + rng.normal(0, 0.06 * w), 0, w - 1))
            y0 = float(np.clip(y0 + rng.normal(0, 0.06 * h), 0, h - 1))
            pts.append((x0, y0))
        color = tuple(int(v) for v in rng.integers(0, 256, 3))
        d.line(pts, fill=color, width=width, joint="curve")
    return reencode(data, img)


@register("caption_text", "overlay", {"text": "Seen on the street today", "height_frac": 0.06})
def _caption(data, p, seed, ctx=None):
    img = decode_rgb(data).copy()
    w, h = img.size
    size = max(10, round(p["height_frac"] * h))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=size)
    tw = d.textlength(p["text"], font=font)
    x, y = (w - tw) / 2, h - 1.8 * size
    d.rectangle((0, y - 0.3 * size, w, y + 1.3 * size), fill=(0, 0, 0))
    d.text((x, y), p["text"], fill=(255, 255, 255), font=font)
    return reencode(data, img)


@register("sticker", "overlay", {"area": 0.05})
def _sticker(data, p, seed, ctx=None):
    """A flat graphic (circle + star) pasted at a seeded position, as a sticker would be."""
    img = decode_rgb(data).copy()
    w, h = img.size
    rng = np.random.default_rng(seed)
    side = round((p["area"] * w * h) ** 0.5)
    st = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    d = ImageDraw.Draw(st)
    d.ellipse(
        (0, 0, side - 1, side - 1),
        fill=(255, 205, 40, 255),
        outline=(40, 40, 40, 255),
        width=max(2, side // 30),
    )
    cx = cy = side / 2
    r1, r2 = side * 0.38, side * 0.16
    star = [
        (
            cx + (r1 if k % 2 == 0 else r2) * np.cos(np.pi / 2 + k * np.pi / 5),
            cy - (r1 if k % 2 == 0 else r2) * np.sin(np.pi / 2 + k * np.pi / 5),
        )
        for k in range(10)
    ]
    d.polygon(star, fill=(230, 60, 60, 255))
    x = int(rng.integers(0, max(1, w - side)))
    y = int(rng.integers(0, max(1, h - side)))
    img.paste(st, (x, y), st)
    return reencode(data, img, box=[x, y, x + side, y + side])


def _classical_fill(data, p, seed, ctx=None):
    """Remove the masked region and fill it without a generative model (OpenCV Telea)."""
    import cv2

    master = decode_rgb(data)
    ell = ellipse_params(*master.size, p["area"], seed)
    mask = ellipse_mask(master.size, ell).astype(np.uint8) * 255
    arr = np.asarray(master)[:, :, ::-1].copy()
    out = cv2.inpaint(arr, mask, 5, cv2.INPAINT_TELEA)[:, :, ::-1]
    img = Image.fromarray(out)
    geometry = {"ellipse": ell, "size": list(master.size)}
    fmt = sniff_format(data)
    return Output(encode(img, fmt), fmt, alignment="masked", geometry=geometry)


# same seed keys as the generative inpaints -> identical masks
for _a in (10, 30):
    register(
        f"remove_classical_{_a}",
        "removal",
        {"area": _a / 100, "fill": "telea"},
        seed_id=f"inpaint_{_a}",
    )(_classical_fill)
