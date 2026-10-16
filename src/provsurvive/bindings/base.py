"""Soft binding plugin contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from PIL import Image


@dataclass(frozen=True)
class Decoded:
    detected: bool
    payload: str | None  # after error correction, None if nothing decoded
    raw_bits: tuple[int, ...] | None = None  # before error correction, if exposed
    fingerprint: str | None = None  # fingerprints only
    schema: int | None = None  # ECC / payload schema reported by the decoder, if any


class SoftBinding(Protocol):
    name: str
    alg: str  # soft binding registry identifier, e.g. "com.adobe.trustmark.Q"
    kind: Literal["watermark", "fingerprint"]

    def config(self) -> dict[str, Any]: ...

    def new_payload(self, seed: int) -> str: ...

    def embed(self, img: Image.Image, payload: str) -> Image.Image: ...

    def assertion_value(self, img: Image.Image, payload: str) -> str: ...

    def decode(self, img: Image.Image) -> Decoded: ...

    def coded_bits(self, payload: str) -> tuple[int, ...] | None: ...

    def match(self, decoded: Decoded, stored_value: str) -> float | None:
        """Score in [0, 1] if ``decoded`` matches the stored assertion value, else None."""
        ...
