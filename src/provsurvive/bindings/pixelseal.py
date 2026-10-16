"""PixelSeal (Meta FAIR, MIT code and weights) as a second, non-TrustMark watermark family.

The C2PA soft binding registry lists a third-party deployment of this model
(``com.aiwatermark.pixelseal.1``, id 31) whose payload format is not public, so the plugin uses
its own, unregistered ``alg`` (C2PA 2.4 allows one).

PixelSeal carries 256 bits without error correction. A stored value matches if the decoded
bits are within ``threshold`` Hamming distance. At the default threshold (64, i.e. at least 192
of 256 bits agree) unrelated bits match with probability 2.4e-16 per comparison, about 2.4e-7
for a registry of 1e9.

The model is loaded from a videoseal checkout pinned to a commit (it reads model cards and
configs relative to the working directory); see ``VIDEOSEAL_COMMIT``. ``provenance()`` records
the checkout actually loaded and the checkpoint's SHA-256 in run.json.

Detection: the extractor's first output channel is what the release calls a detection bit, and
the release itself does not use it ("binary detection bit (not used for now)",
videoseal/models/videoseal.py). We read it at logit 0 for the ``detected`` flag; presence is
decided by the payload rule above, never by this flag.
"""

from __future__ import annotations

import contextlib
import os
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from .base import Decoded

VIDEOSEAL_COMMIT = "870ca7fb33578b90f14c602016b6c2788096226e"
# what `detected` means, for the registry results file
DETECTION_FLAG = (
    "the extractor's first output, which the release describes as a detection bit and does not "
    "use itself, read at logit 0; presence is decided by payload matching (false matches)"
)
NBITS = 256


@contextlib.contextmanager
def _chdir(path: Path):
    """contextlib.chdir for Python 3.10."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def videoseal_dir() -> Path:
    """Checkout of facebookresearch/videoseal at VIDEOSEAL_COMMIT (see README)."""
    if os.environ.get("PROVSURVIVE_VIDEOSEAL"):
        return Path(os.environ["PROVSURVIVE_VIDEOSEAL"])
    return Path(os.environ.get("PROVSURVIVE_DATA", "data")) / "vendor" / "videoseal"


class PixelSealBinding:
    kind: Literal["watermark"] = "watermark"
    alg = "io.github.provsurvive.pixelseal-meta.v1"  # unregistered, see module doc

    def __init__(self, threshold: int = 64, device: str = "") -> None:
        self.threshold = threshold
        self.device = device or "cpu"
        self.name = f"pixelseal-t{threshold}"

    @cached_property
    def _model(self):
        with _chdir(videoseal_dir()):
            import videoseal

            return videoseal.load("pixelseal").eval().to(self.device)

    def config(self) -> dict[str, Any]:
        return {
            "nbits": NBITS,
            "threshold": self.threshold,
            "videoseal_commit": VIDEOSEAL_COMMIT,
            "model_card": "pixelseal",
        }

    def provenance(self) -> dict[str, Any]:
        """What was actually loaded (not part of config(), which keys the masters cache)."""
        import hashlib
        import subprocess

        root = videoseal_dir()

        def git(*args: str) -> str | None:
            try:
                return subprocess.run(
                    ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                return None

        ckpt = root / "ckpts" / "pixelseal_checkpoint.pth"
        digest = None
        if ckpt.exists():
            h = hashlib.sha256()
            with open(ckpt, "rb") as f:
                for block in iter(lambda: f.read(1 << 24), b""):
                    h.update(block)
            digest = h.hexdigest()
        status = git("status", "--porcelain")
        return {
            "videoseal_head": git("rev-parse", "HEAD"),
            "videoseal_dirty": None if status is None else bool(status),
            "checkpoint": "ckpts/pixelseal_checkpoint.pth",
            "checkpoint_sha256": digest,
        }

    def new_payload(self, seed: int) -> str:
        rng = np.random.default_rng(seed)
        return "".join(rng.choice(["0", "1"], NBITS))

    def _to_tensor(self, img: Image.Image):
        import torch

        x = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(x).permute(2, 0, 1)[None].to(self.device)

    def embed(self, img: Image.Image, payload: str) -> Image.Image:
        import torch

        msg = torch.tensor([[int(b) for b in payload]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self._model.embed(self._to_tensor(img), msgs=msg, is_video=False)
        w = out["imgs_w"][0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(np.round(w * 255.0).astype(np.uint8))

    def assertion_value(self, img: Image.Image, payload: str) -> str:
        return payload

    def coded_bits(self, payload: str) -> tuple[int, ...]:
        return tuple(int(b) for b in payload)

    def decode(self, img: Image.Image) -> Decoded:
        import torch

        with torch.no_grad():
            preds = self._model.detect(self._to_tensor(img), is_video=False)["preds"][0]
        bits = tuple(int(b) for b in (preds[1:] > 0).cpu().numpy())
        return Decoded(
            detected=bool(preds[0] > 0),  # the release's unused detection bit; see module doc
            payload="".join(map(str, bits)),
            raw_bits=bits,
        )

    def distance(self, decoded: Decoded, stored_value: str) -> int | None:
        if decoded.payload is None:
            return None
        return sum(a != b for a, b in zip(decoded.payload, stored_value, strict=True))

    def match(self, decoded: Decoded, stored_value: str) -> float | None:
        d = self.distance(decoded, stored_value)
        if d is None or d > self.threshold:
            return None
        return 1.0 - d / NBITS
