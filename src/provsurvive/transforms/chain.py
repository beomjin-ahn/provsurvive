"""Chains of everyday operations."""

from __future__ import annotations

from PIL import Image

from .base import Output, decode_rgb, encode, register
from .color import apply_lut, exposure


@register("export_edit", "chain", {"ev": 0.3, "lut": "film", "jpeg_quality": 90})
def _export_edit(data, p, seed, ctx=None):
    """Edit then export: exposure -> film LUT -> JPEG."""
    img = apply_lut(exposure(decode_rgb(data), p["ev"]))
    return Output(encode(img, "jpeg", quality=p["jpeg_quality"], subsampling="4:2:0"), "jpeg")


@register("share_small", "chain", {"long_side": 1080, "jpeg_quality": 75})
def _share_small(data, p, seed, ctx=None):
    """Downscale-and-share: long side 1080 -> JPEG q75 (metadata dropped by re-encoding)."""
    img = decode_rgb(data)
    s = p["long_side"] / max(img.size)
    if s < 1:
        img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
    out = encode(img, "jpeg", quality=p["jpeg_quality"], subsampling="4:2:0")
    return Output(out, "jpeg", alignment="upsampled")
