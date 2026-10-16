"""Source images for a run: a corpus directory, or procedural images."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

Kind = Literal["generated", "photo", "synthetic"]

# digitalSourceType (IPTC) used when signing a master of each kind.
SOURCE_TYPE: dict[str, str] = {
    "generated": "trainedAlgorithmicMedia",
    "photo": "digitalCapture",  # signed as if by a camera, so post-edit truthfulness is measurable
    "synthetic": "algorithmicMedia",
}


@dataclass
class SourceImage:
    id: str
    kind: Kind
    subset: str
    image: Image.Image
    meta: dict[str, Any] = field(default_factory=dict)


def synthetic(n: int, size: int = 512, seed: int = 0) -> list[SourceImage]:
    """Procedural test images: gradients, shapes and texture. License-free, deterministic."""
    out = []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        yy, xx = np.mgrid[0:size, 0:size] / size
        c0, c1 = rng.random(3), rng.random(3)
        t = (xx * np.cos(i) + yy * np.sin(i)) % 1.0
        base = (c0[None, None] * (1 - t[..., None]) + c1[None, None] * t[..., None]) * 255
        noise = rng.normal(0, 6 + 4 * (i % 3), (size, size, 3))
        img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        for _ in range(6):
            x0, y0 = rng.integers(0, size - 64, 2)
            x1, y1 = x0 + rng.integers(32, size // 2), y0 + rng.integers(32, size // 2)
            color = tuple(int(v) for v in rng.integers(0, 256, 3))
            (draw.ellipse if rng.random() < 0.5 else draw.rectangle)((x0, y0, x1, y1), fill=color)
        out.append(
            SourceImage(f"syn-{i:03d}", "synthetic", "synthetic", img, {"license": "CC0-1.0"})
        )
    return out


def from_directory(root: Path, subset: str | None = None) -> list[SourceImage]:
    """A directory with images and an ``index.json``: [{"file", "id", "kind", ...meta}].

    ``subset`` defaults to each entry's kind ("generated" / "photo").
    """
    index = json.loads((root / "index.json").read_text())
    out = []
    for entry in index:
        meta = {k: v for k, v in entry.items() if k not in ("file", "id", "kind")}
        img = Image.open(root / entry["file"])
        img.load()
        out.append(
            SourceImage(
                entry["id"], entry["kind"], subset or entry["kind"], img.convert("RGB"), meta
            )
        )
    return out
