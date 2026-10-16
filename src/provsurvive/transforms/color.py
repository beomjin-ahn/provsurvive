"""Color family: global tonal and color edits, re-encoded in the master's format."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import decode_rgb, reencode, register


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _to_img(x: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8))


def exposure(img: Image.Image, ev: float) -> Image.Image:
    lin = srgb_to_linear(np.asarray(img, dtype=np.float64) / 255.0)
    return _to_img(linear_to_srgb(lin * 2.0**ev))


def _exposure(data, p, seed, ctx=None):
    return reencode(data, exposure(decode_rgb(data), p["ev"]))


for _ev in (-1.5, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 1.5):
    register(f"exposure_{_ev:+g}ev", "color", {"ev": _ev})(_exposure)


# --- OKLab (Ottosson 2020) -------------------------------------------------------

_M1 = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_M2 = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)


def linear_to_oklab(rgb: np.ndarray) -> np.ndarray:
    return np.cbrt(rgb @ _M1.T) @ _M2.T


def oklab_to_linear(lab: np.ndarray) -> np.ndarray:
    return (lab @ np.linalg.inv(_M2).T) ** 3 @ np.linalg.inv(_M1).T


def saturation(img: Image.Image, factor: float) -> Image.Image:
    lin = srgb_to_linear(np.asarray(img, dtype=np.float64) / 255.0)
    lab = linear_to_oklab(lin)
    lab[..., 1:] *= factor  # scale chroma, keep lightness and hue
    return _to_img(linear_to_srgb(oklab_to_linear(lab)))


@register("saturation_1.3", "color", {"factor": 1.3})
def _saturation(data, p, seed, ctx=None):
    return reencode(data, saturation(decode_rgb(data), p["factor"]))


# --- 3D LUT -----------------------------------------------------------------------


def film_lut_table(size: int = 33) -> list[float]:
    """A procedural 'film' look: gentle S-curve, lifted blacks, warm highlights, cool shadows.

    Table order follows Pillow's Color3DLUT: red varies fastest.
    """
    g = np.linspace(0.0, 1.0, size)
    b, gg, r = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([r, gg, b], axis=-1)
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722])
    s_curve = lambda x: 0.5 - 0.5 * np.cos(np.pi * x)  # noqa: E731
    out = 0.75 * rgb + 0.25 * s_curve(rgb)
    out = 0.04 + 0.94 * out  # lift blacks, soften whites
    warm = np.array([0.035, 0.01, -0.03])
    cool = np.array([-0.02, 0.0, 0.03])
    out += luma[..., None] * warm + (1 - luma[..., None]) * cool
    out = luma[..., None] + 0.9 * (out - luma[..., None])  # slight desaturation
    return np.clip(out, 0, 1).reshape(-1).tolist()


def apply_lut(img: Image.Image, size: int = 33) -> Image.Image:
    from PIL import ImageFilter

    return img.filter(ImageFilter.Color3DLUT(size, film_lut_table(size)))


@register("lut_film", "color", {"lut": "film", "size": 33})
def _lut(data, p, seed, ctx=None):
    return reencode(data, apply_lut(decode_rgb(data), p["size"]))


# --- Reinhard color transfer ---------------------------------------------------------

# Target Lab statistics (mean, std per channel) of three hand-set reference "looks", fixed
# here so the transform needs no reference image files. The seed picks the look per cell.
REFERENCE_LOOKS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "golden_hour": ((62.0, 9.0, 28.0), (22.0, 9.0, 14.0)),
    "cool_overcast": ((58.0, -3.0, -9.0), (20.0, 5.0, 8.0)),
    "faded_film": ((60.0, 4.0, 8.0), (16.0, 6.0, 9.0)),
}


def reinhard(img: Image.Image, target_mean, target_std) -> Image.Image:
    """Reinhard et al. 2001 statistics transfer, done in CIELAB (a common variant)."""
    from skimage import color as skcolor

    lab = skcolor.rgb2lab(np.asarray(img, dtype=np.float64) / 255.0)
    mu = lab.reshape(-1, 3).mean(0)
    sd = lab.reshape(-1, 3).std(0) + 1e-6
    lab = (lab - mu) / sd * np.asarray(target_std) + np.asarray(target_mean)
    return _to_img(np.clip(skcolor.lab2rgb(lab), 0, 1))


@register("color_transfer", "color", {"method": "reinhard2001"}, impl_version="1")
def _color_transfer(data, p, seed, ctx=None):
    names = sorted(REFERENCE_LOOKS)
    look = names[seed % len(names)]
    mean, std = REFERENCE_LOOKS[look]
    return reencode(data, reinhard(decode_rgb(data), mean, std), look=look)
