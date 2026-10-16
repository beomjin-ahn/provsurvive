"""Geometric family. Outputs keep the master's container format."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .base import decode_rgb, reencode, register


def _resize(data, p, seed, ctx=None):
    img = decode_rgb(data)
    w, h = img.size
    size = (max(1, round(w * p["scale"])), max(1, round(h * p["scale"])))
    return reencode(data, img.resize(size, Image.LANCZOS), "upsampled")


for s in (0.9, 0.75, 0.5, 0.35, 0.25, 0.2):
    register(f"resize_{s}", "geometric", {"scale": s})(_resize)


def crop_box(
    w: int, h: int, keep_area: float, aspect: float | None, rng
) -> tuple[int, int, int, int]:
    """Box keeping ``keep_area`` of the image; ``aspect`` = w/h of the crop (None: keep)."""
    aspect = aspect or w / h
    ch = math.sqrt(keep_area * w * h / aspect)
    cw = ch * aspect
    if cw > w:  # aspect change limited by width
        cw, ch = w, w / aspect
    if ch > h:
        ch, cw = h, h * aspect
    cw, ch = int(round(cw)), int(round(ch))
    left = int(rng.integers(0, w - cw + 1))
    top = int(rng.integers(0, h - ch + 1))
    return left, top, left + cw, top + ch


def _crop(data, p, seed, ctx=None):
    img = decode_rgb(data)
    rng = np.random.default_rng(seed)
    box = crop_box(*img.size, 1.0 - p["remove_area"], None, rng)
    return reencode(data, img.crop(box), "cropped_reference", box=box)


for a in (5, 10, 20, 30, 40):
    register(f"crop_{a}", "geometric", {"remove_area": a / 100})(_crop)


@register("reframe_4x5", "geometric", {"aspect": 4 / 5})
def _reframe(data, p, seed, ctx=None):
    img = decode_rgb(data)
    rng = np.random.default_rng(seed)
    box = crop_box(*img.size, 1.0, p["aspect"], rng)  # largest 4:5 crop
    return reencode(data, img.crop(box), "cropped_reference", box=box)


@register("rotate_90", "geometric")
def _rot90(data, p, seed, ctx=None):
    return reencode(data, decode_rgb(data).transpose(Image.ROTATE_90), "inverted", op="rotate_90")


@register("flip_h", "geometric")
def _flip(data, p, seed, ctx=None):
    img = decode_rgb(data).transpose(Image.FLIP_LEFT_RIGHT)
    return reencode(data, img, "inverted", op="flip_h")


# --- matched-area crops: separate area removed from aspect-ratio change ------------------
# keep a fraction (fw, fh) of width and height; aspect change = fw / fh (as a factor).


def _crop_frac(data, p, seed, ctx=None):
    img = decode_rgb(data)
    w, h = img.size
    cw, ch = max(1, round(w * p["keep_w"])), max(1, round(h * p["keep_h"]))
    rng = np.random.default_rng(seed)
    left = int(rng.integers(0, w - cw + 1))
    top = int(rng.integers(0, h - ch + 1))
    box = (left, top, left + cw, top + ch)
    return reencode(data, img.crop(box), "cropped_reference", box=box)


_FRAC_CROPS = {
    # name: (keep_w, keep_h)    area kept   aspect factor
    "crop_ap_20": (0.8944, 0.8944),  # 0.80 1.00
    "crop_ap_50": (0.7071, 0.7071),  # 0.50 1.00
    "crop_ar125_20": (0.8, 1.0),  # 0.80 1.25 (width only)
    "crop_ar125_50": (0.6325, 0.7906),  # 0.50 1.25
    "crop_ar2_50": (0.5, 1.0),  # 0.50 2.00 (width only)
}
for _n, (_kw, _kh) in _FRAC_CROPS.items():
    register(_n, "geometric", {"keep_w": _kw, "keep_h": _kh})(_crop_frac)
