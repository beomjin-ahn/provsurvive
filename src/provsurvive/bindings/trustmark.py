"""TrustMark soft binding (github.com/adobe/trustmark, MIT).

Registry ids: com.adobe.trustmark.Q / .C / .P (B is not registered). The assertion
value follows TrustMark's C2PA example: ``"<encoding>*<payload bits>"``.

Raw (pre-ECC) bits are read from the decoder network directly, mirroring
``TrustMark.subimage_decode``; this touches internals, so the trustmark version is
pinned in run.json and covered by a test.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Literal

import numpy as np
from PIL import Image

from .base import Decoded

_REGISTERED = {"Q", "C", "P"}
# what `detected` means, for the registry results file
DETECTION_FLAG = (
    "the TrustMark decoder's detection flag, with the ECC schema auto-detected among four BCH "
    "schemas; detections in the deployed schema are counted separately"
)


class TrustMarkBinding:
    kind: Literal["watermark"] = "watermark"

    def __init__(
        self,
        variant: str = "Q",
        encoding: str = "BCH_5",
        strength: float = 1.0,
        rotation_search: bool = False,
        device: str = "",
    ) -> None:
        if variant not in _REGISTERED:
            raise ValueError(f"TrustMark variant {variant!r} is not in the C2PA registry")
        self.variant = variant
        self.encoding = encoding
        self.strength = strength
        self.rotation_search = rotation_search
        self.device = device
        self.name = f"trustmark-{variant}" + ("-rot" if rotation_search else "")
        self.alg = f"com.adobe.trustmark.{variant}"

    @cached_property
    def _tm(self):
        from trustmark import TrustMark

        enc = getattr(TrustMark.Encoding, self.encoding)
        return TrustMark(
            verbose=False,
            model_type=self.variant,
            encoding_type=enc,
            loadRemover=False,
            device=self.device,
        )

    @property
    def _enc_code(self) -> int:
        from trustmark import TrustMark

        return int(getattr(TrustMark.Encoding, self.encoding))

    def config(self) -> dict[str, Any]:
        from importlib.metadata import version

        return {
            "variant": self.variant,
            "encoding": self.encoding,
            "strength": self.strength,
            "rotation_search": self.rotation_search,
            "trustmark_version": version("trustmark"),
        }

    def capacity(self) -> int:
        return int(self._tm.schemaCapacity())

    def new_payload(self, seed: int) -> str:
        rng = np.random.default_rng(seed)
        return "".join(rng.choice(["0", "1"], self.capacity()))

    def embed(self, img: Image.Image, payload: str) -> Image.Image:
        return self._tm.encode(
            img.convert("RGB"), payload, MODE="binary", WM_STRENGTH=self.strength
        )

    def assertion_value(self, img: Image.Image, payload: str) -> str:
        return f"{self._enc_code}*{payload}"

    def coded_bits(self, payload: str) -> tuple[int, ...]:
        arr = np.asarray(self._tm.ecc.encode_binary([payload])).reshape(-1)
        return tuple(int(b) for b in arr)

    def _raw_bits(self, img: Image.Image) -> tuple[int, ...]:
        import torch
        from torchvision import transforms

        tm = self._tm
        x = tm.get_the_image_for_processing(img)
        x = x.resize((tm.model_resolution_dec, tm.model_resolution_dec), Image.BILINEAR)
        t = transforms.ToTensor()(x).unsqueeze(0).to(tm.decoder.device) * 2.0 - 1.0
        with torch.no_grad():
            bits = (tm.decoder.decoder(t) > 0).cpu().numpy()[0]
        return tuple(int(b) for b in bits)

    def decode(self, img: Image.Image) -> Decoded:
        img = img.convert("RGB")
        secret, detected, schema = self._tm.decode(
            img, MODE="binary", ROTATION=self.rotation_search
        )
        return Decoded(
            detected=bool(detected),
            payload=secret if detected and secret else None,
            raw_bits=self._raw_bits(img),
            schema=int(schema) if detected else None,
        )

    def expected_schema(self) -> int:
        return self._enc_code

    def match(self, decoded: Decoded, stored_value: str) -> float | None:
        if decoded.payload is None:
            return None
        _, _, bits = stored_value.partition("*")
        return 1.0 if decoded.payload == bits else None
