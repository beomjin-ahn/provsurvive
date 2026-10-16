"""ISCC Content-Code Image as a fingerprint soft binding (registry id ``io.iscc.v0``).

A fingerprint does not change pixels (zero quality cost). The manifest records the 64-bit
image code of the master. Recovery is a nearest-neighbour search: the resolver returns the
manifest whose code is closest in Hamming distance, if within ``threshold`` bits.

Normalization follows iscc-sdk (EXIF transpose, grayscale, bicubic 32x32) without its
default trimming of uniform borders. The default threshold (12 bits) lies below the smallest
distance between distinct corpus images (16 over 19,900 pairs).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps

from .base import Decoded

BITS = 64


def image_code(img: Image.Image) -> str:
    import iscc_core as ic

    g = ImageOps.exif_transpose(img).convert("L").resize((32, 32), Image.BICUBIC)
    return ic.gen_image_code_v0(np.asarray(g).reshape(-1).tolist(), bits=BITS)["iscc"]


def code_bits(code: str) -> np.ndarray:
    import iscc_core as ic

    raw = ic.decode_base32(code.split(":", 1)[1])[2:]  # drop the 2-byte ISCC header
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8))


def hamming(a: str, b: str) -> int:
    return int((code_bits(a) != code_bits(b)).sum())


class IsccBinding:
    kind: Literal["fingerprint"] = "fingerprint"
    alg = "io.iscc.v0"

    def __init__(self, threshold: int = 12, device: str = "") -> None:
        self.threshold = threshold
        self.name = f"iscc-image-t{threshold}"

    def config(self) -> dict[str, Any]:
        from importlib.metadata import version

        return {"bits": BITS, "threshold": self.threshold, "iscc_core": version("iscc-core")}

    def new_payload(self, seed: int) -> str:
        return ""  # nothing is embedded

    def embed(self, img: Image.Image, payload: str) -> Image.Image:
        return img.convert("RGB")

    def assertion_value(self, img: Image.Image, payload: str) -> str:
        return image_code(img)

    def coded_bits(self, payload: str) -> None:
        return None

    def decode(self, img: Image.Image) -> Decoded:
        code = image_code(img)
        # payload carries the code itself so results can be re-matched against other registries
        return Decoded(detected=True, payload=code, fingerprint=code)

    def distance(self, decoded: Decoded, stored_value: str) -> int | None:
        if decoded.fingerprint is None:
            return None
        return hamming(decoded.fingerprint, stored_value)

    def match(self, decoded: Decoded, stored_value: str) -> float | None:
        d = self.distance(decoded, stored_value)
        if d is None or d > self.threshold:
            return None
        return 1.0 - d / BITS
