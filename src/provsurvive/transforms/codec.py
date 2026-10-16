"""Codec family: decode and re-encode, as any export dialog does."""

from __future__ import annotations

from .base import Output, decode_rgb, encode, register
from .jumbf import sniff_format


def _jpeg(data, p, seed, ctx=None):
    return Output(
        encode(decode_rgb(data), "jpeg", quality=p["quality"], subsampling="4:2:0"), "jpeg"
    )


for q in (95, 90, 85, 80, 75, 70, 60, 50, 40, 30):
    register(f"jpeg_q{q}", "codec", {"quality": q})(_jpeg)


@register("webp_q80", "codec", {"quality": 80})
def _webp(data, p, seed, ctx=None):
    return Output(encode(decode_rgb(data), "webp", quality=p["quality"], method=4), "webp")


@register("avif_q60", "codec", {"quality": 60})
def _avif(data, p, seed, ctx=None):
    return Output(encode(decode_rgb(data), "avif", quality=p["quality"], speed=6), "avif")


@register("png_reencode", "codec")
def _png(data, p, seed, ctx=None):
    return Output(encode(decode_rgb(data), "png"), "png")


@register("identity", "control")
def _identity(data, p, seed, ctx=None):
    """The master itself: every layer must be intact (known-answer test)."""
    return Output(data, sniff_format(data))
