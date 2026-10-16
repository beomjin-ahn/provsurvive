"""Build soft binding plugins from a JSON-able spec (so worker processes can rebuild them)."""

from __future__ import annotations

from typing import Any

from .base import SoftBinding


def make_plugin(spec: dict[str, Any]) -> SoftBinding:
    spec = dict(spec)
    name = spec.pop("name")
    if name == "trustmark":
        from .trustmark import TrustMarkBinding

        return TrustMarkBinding(**spec)
    if name == "pixelseal":
        from .pixelseal import PixelSealBinding

        return PixelSealBinding(**spec)
    if name == "iscc":
        from .iscc import IsccBinding

        return IsccBinding(**spec)
    raise ValueError(f"unknown soft binding plugin {name!r}")
