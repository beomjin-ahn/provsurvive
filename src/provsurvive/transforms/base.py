"""Transform registry. Transforms take and return file bytes."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .jumbf import sniff_format


@dataclass(frozen=True)
class Output:
    data: bytes
    format: str  # "png", "jpeg", "webp", "avif"
    # How to align the master with this output for quality metrics.
    alignment: str = "direct"
    geometry: dict[str, Any] = field(default_factory=dict)


TransformFn = Callable[..., Output]  # (data, params, seed, ctx) -> Output


@dataclass(frozen=True)
class TransformSpec:
    id: str
    family: str
    fn: TransformFn
    params: dict[str, Any]
    impl_version: str = "1"
    gpu: bool = False
    seed_id: str | None = None  # transforms sharing a seed_id get the same per-cell seed

    def seed_key(self) -> tuple[str, dict[str, Any]]:
        """What the per-cell seed is derived from. Variants that differ only in how a shared
        random draw is used (e.g. inpaint composite vs raw) share it, so they stay paired."""
        params = {k: v for k, v in self.params.items() if k not in _UNSEEDED}
        return self.seed_id or self.id, params

    def __call__(self, data: bytes, seed: int, ctx: dict[str, Any] | None = None) -> Output:
        """``ctx``: source metadata (kind, prompt, ...) for transforms that need it."""
        return self.fn(data, self.params, seed, ctx or {})


REGISTRY: dict[str, TransformSpec] = {}

# Parameters that change how a shared random draw is used, not the draw itself.
_UNSEEDED = {"composite", "fill", "editor", "prompt"}


def register(
    id: str,
    family: str,
    params: dict[str, Any] | None = None,
    *,
    impl_version="1",
    gpu=False,
    seed_id: str | None = None,
):
    def deco(fn: TransformFn) -> TransformFn:
        REGISTRY[id] = TransformSpec(id, family, fn, dict(params or {}), impl_version, gpu, seed_id)
        return fn

    return deco


def get(id: str) -> TransformSpec:
    _load_builtin()
    try:
        return REGISTRY[id]
    except KeyError:
        raise KeyError(f"unknown transform {id!r}; known: {sorted(REGISTRY)}") from None


def _load_builtin() -> None:
    from . import (  # noqa: F401  (registration side effects)
        chain,
        codec,
        color,
        composite,
        content,
        generative,
        geometric,
        metadata,
    )


# --- helpers ----------------------------------------------------------------

CACHE_KEY_VERSION = 2


def cache_key(digest: str, size: tuple[int, int], settings: dict | None = None) -> str:
    """Key of a cached image (a marked master or a generative output). Version 2 adds to the
    version 1 digest of the input pixels and the edit what that digest left out: the input size
    and the generation settings. Version 1 caches can so be re-keyed from their files alone."""
    blob = json.dumps([CACHE_KEY_VERSION, digest, list(size), settings], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def decode_rgb(data: bytes) -> Image.Image:
    """The image as displayed: EXIF orientation applied."""
    img = Image.open(io.BytesIO(data))
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode.startswith("I;16"):  # convert() would clip 16-bit gray to 8 bits
        img = Image.fromarray((np.asarray(img) // 257).astype(np.uint8))
    return img.convert("RGB")


def encode(img: Image.Image, fmt: str, **kw: Any) -> bytes:
    buf = io.BytesIO()
    pil_fmt = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP", "avif": "AVIF"}[fmt]
    if fmt == "png":
        kw.setdefault("compress_level", 6)
    img.save(buf, pil_fmt, **kw)
    return buf.getvalue()


def reencode(data: bytes, img: Image.Image, alignment: str = "direct", **geometry) -> Output:
    """``img`` in the container format of ``data`` (JPEG at quality 95)."""
    fmt = sniff_format(data)
    kw = {"quality": 95} if fmt == "jpeg" else {}
    return Output(encode(img, fmt, **kw), fmt, alignment=alignment, geometry=geometry)
