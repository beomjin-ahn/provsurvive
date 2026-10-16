"""Image quality metrics with explicit alignment rules."""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

LPIPS_LONG_SIDE = 512


def align(master: Image.Image, output: Image.Image, alignment: str, geometry: dict[str, Any]):
    """Return (reference, candidate, retained_area) with identical sizes."""
    mw, mh = master.size
    if alignment in ("direct", "masked"):
        ref, cand = master, output
    elif alignment == "upsampled":
        ref, cand = master, output.resize((mw, mh), Image.LANCZOS)
    elif alignment == "cropped_reference":
        box = tuple(geometry["box"])
        ref, cand = master.crop(box), output
    elif alignment == "approximate":
        # screenshot: the image region inside the capture, resized back to the master size
        box = tuple(geometry["image_box"])
        ref, cand = master, output.crop(box).resize((mw, mh), Image.LANCZOS)
    elif alignment == "inverted":
        op = geometry["op"]
        undo = {"rotate_90": Image.ROTATE_270, "flip_h": Image.FLIP_LEFT_RIGHT}[op]
        ref, cand = master, output.transpose(undo)
    else:
        raise ValueError(f"alignment {alignment!r} not supported here")
    if ref.size != cand.size:
        raise ValueError(f"aligned sizes differ: {ref.size} vs {cand.size}")
    retained = (ref.size[0] * ref.size[1]) / (mw * mh)
    return ref, cand, retained


def alignment_for(
    master: tuple[int, int], output: tuple[int, int], policy: str = "exact"
) -> tuple[str | None, str | None]:
    """How to compare a file processed outside provsurvive with its master. ``exact``: only a
    file of the master's size. ``rescale``: also one of the master's aspect ratio, rescaled to
    the master's size, which assumes it was not cropped. Returns (alignment, why not)."""
    (mw, mh), (w, h) = master, output
    if (w, h) == (mw, mh):
        return "direct", None
    if abs(round(mh * w / mw) - h) > 1 and abs(round(mw * h / mh) - w) > 1:
        return None, f"{w}x{h} vs the master's {mw}x{mh}: aspect ratio differs"
    if policy == "rescale":
        return "upsampled", None
    return None, f"{w}x{h} vs the master's {mw}x{mh}: rescaled; --align rescale measures it"


def psnr(a: np.ndarray, b: np.ndarray) -> float | None:
    if np.array_equal(a, b):
        return None  # infinite; stored as null and reported as "identical"
    return float(peak_signal_noise_ratio(a, b, data_range=255))


def ssim(a: np.ndarray, b: np.ndarray) -> float | None:
    if min(a.shape[:2]) < 7:  # smaller than SSIM's window
        return None
    return float(structural_similarity(a, b, channel_axis=-1, data_range=255))


@lru_cache(maxsize=2)
def _lpips_model(device: str):
    import lpips

    with warnings.catch_warnings():  # lpips loads AlexNet with torchvision's deprecated arguments
        warnings.simplefilter("ignore", UserWarning)
        return lpips.LPIPS(net="alex", verbose=False).eval().to(device)


def _for_lpips(img: Image.Image):
    import torch

    w, h = img.size
    s = LPIPS_LONG_SIDE / max(w, h)
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return torch.from_numpy(np.asarray(img, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)[None]


def lpips_distance(a: Image.Image, b: Image.Image, device: str = "cpu") -> float | None:
    import torch

    if min(a.size) * min(1, LPIPS_LONG_SIDE / max(a.size)) < 32:  # too small for AlexNet
        return None
    model = _lpips_model(device)
    with torch.no_grad():
        return float(model(_for_lpips(a).to(device), _for_lpips(b).to(device)).item())


def compare_masked(
    ref: Image.Image, cand: Image.Image, mask: np.ndarray
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """PSNR and mean absolute error inside and outside a boolean mask (LPIPS is full-image only)."""
    a = np.asarray(ref.convert("RGB"), dtype=np.float64)
    b = np.asarray(cand.convert("RGB"), dtype=np.float64)

    def stats(sel: np.ndarray) -> dict[str, float | None]:
        if not sel.any():
            return {"psnr": None, "mae": None}
        mse = float(np.mean((a[sel] - b[sel]) ** 2))
        return {
            "psnr": None if mse == 0 else float(10 * np.log10(255.0**2 / mse)),
            "mae": float(np.mean(np.abs(a[sel] - b[sel]))),
        }

    return stats(mask), stats(~mask)


def compare(
    ref: Image.Image, cand: Image.Image, *, with_lpips: bool, device: str = "cpu"
) -> dict[str, float | None]:
    a = np.asarray(ref.convert("RGB"))
    b = np.asarray(cand.convert("RGB"))
    return {
        "psnr": psnr(a, b),
        "ssim": ssim(a, b),
        "lpips": lpips_distance(ref, cand, device) if with_lpips else None,
    }
