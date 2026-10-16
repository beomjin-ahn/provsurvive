"""Metadata family: container-level changes that isolate the embedded manifest."""

from __future__ import annotations

from . import jumbf
from .base import Output, decode_rgb, reencode, register
from .color import exposure


@register("strip_metadata", "metadata")
def _strip(data, p, seed, ctx=None):
    """Remove only the C2PA blocks; pixels and all other bytes are untouched."""
    return Output(jumbf.strip_c2pa(data), jumbf.sniff_format(data))


@register("carry_manifest_edit", "metadata", {"ev": 0.3})
def _carry(data, p, seed, ctx=None):
    """Edit pixels, then copy the old manifest back in (a metadata-preserving editor)."""
    edited = reencode(data, exposure(decode_rgb(data), p["ev"]))
    return Output(jumbf.insert_c2pa(edited.data, jumbf.extract_c2pa(data)), edited.format)
