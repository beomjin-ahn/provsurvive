"""Generative family (GPU): Z-Image-Turbo inpainting and img2img.

- Working resolution: long side 1024 (multiples of 16). Larger masters are downscaled for
  diffusion and the result is resized back with Lanczos, so every output has the master's size.
- Inpaint masks: a seeded ellipse covering exactly ``area`` of the image. By default only the
  masked region is composited back onto the master (feathered 3 px), as editors do;
  ``composite=False`` keeps the raw pipeline output, which also alters unmasked pixels.
- Prompts: the source's own prompt for generated images, a fixed neutral prompt otherwise.
- Outputs are cached on disk by master pixels and size, edit, params, seed, prompt, and every
  generation setting (``generation_settings``); the cache doubles as the "frozen edits" that let
  CPU-only users reproduce a run.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from .base import Output, cache_key, decode_rgb, encode, register

ZIMAGE_REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
KLEIN_REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"  # FLUX.2-klein-4B, Apache-2.0
KLEIN_STEPS, KLEIN_GUIDANCE = 4, 1.0  # model-card defaults for the distilled model
MODELS = {  # editor -> (Hugging Face repository, revision)
    "zimage": ("Tongyi-MAI/Z-Image-Turbo", ZIMAGE_REVISION),
    "klein": ("black-forest-labs/FLUX.2-klein-4B", KLEIN_REVISION),
}
NEUTRAL_PROMPT = "a natural, high quality photograph"
WORK_LONG_SIDE = 1024
INPAINT_STEPS = 9
IMG2IMG_STEPS = 20
ZIMAGE_GUIDANCE = 0.0
FEATHER_PX = 3


def generation_settings() -> dict:
    """What a generative output depends on besides its input, edit, seed and prompt."""
    return {
        "models": {k: {"repo": r, "revision": v} for k, (r, v) in MODELS.items()},
        "inpaint_steps": INPAINT_STEPS,
        "img2img_schedule_steps": IMG2IMG_STEPS,
        "zimage_guidance": ZIMAGE_GUIDANCE,
        "klein_steps_guidance": [KLEIN_STEPS, KLEIN_GUIDANCE],
        "work_long_side": WORK_LONG_SIDE,
    }


def model_info() -> dict:
    """What a run's generative edits depend on, for run.json; the cache key covers all of it."""
    return generation_settings() | {
        "feather_px": FEATHER_PX,
        "neutral_prompt": NEUTRAL_PROMPT,
        "cache_key": "v2: sha256([2, v1 digest of input pixels, edit, params, seed, prompt, "
        "editor and revision, input size, generation settings])",
    }


def cache_dir() -> Path:
    root = os.environ.get("PROVSURVIVE_CACHE", str(Path.home() / ".provsurvive" / "cache"))
    return Path(root) / "generative"


def ellipse_params(w: int, h: int, area: float, seed: int) -> list[float]:
    """(cx, cy, rx, ry, angle_deg) of a seeded ellipse with the given area fraction.

    Aspect and rotation are resampled until the rotated ellipse fits inside the frame with a
    2% border; if that never happens (large areas on narrow frames), fall back to an
    axis-aligned ellipse with the frame's aspect ratio.
    """
    rng = np.random.default_rng(seed)
    target = area * w * h
    pad = 0.02 * min(w, h)
    for _ in range(200):
        aspect = float(rng.uniform(0.6, 1.6))
        angle = float(rng.uniform(0, 180))
        ry = math.sqrt(target / (math.pi * aspect))
        rx = aspect * ry
        t = math.radians(angle)
        ex = math.hypot(rx * math.cos(t), ry * math.sin(t))  # half extents of the rotated box
        ey = math.hypot(rx * math.sin(t), ry * math.cos(t))
        if 2 * (ex + pad) <= w and 2 * (ey + pad) <= h:
            cx = float(rng.uniform(ex + pad, w - ex - pad))
            cy = float(rng.uniform(ey + pad, h - ey - pad))
            return [cx, cy, rx, ry, angle]
    aspect = w / h
    ry = math.sqrt(target / (math.pi * aspect))
    return [w / 2, h / 2, aspect * ry, ry, 0.0]


def ellipse_mask(size: tuple[int, int], params: list[float]) -> np.ndarray:
    w, h = size
    cx, cy, rx, ry, angle = params
    yy, xx = np.mgrid[0:h, 0:w]
    t = math.radians(angle)
    x, y = xx - cx, yy - cy
    u = x * math.cos(t) + y * math.sin(t)
    v = -x * math.sin(t) + y * math.cos(t)
    return (u / rx) ** 2 + (v / ry) ** 2 <= 1.0


def _work_size(w: int, h: int) -> tuple[int, int]:
    s = min(1.0, WORK_LONG_SIDE / max(w, h))
    return (max(16, int(round(w * s / 16)) * 16), max(16, int(round(h * s / 16)) * 16))


def _cache_key(
    img: Image.Image, tid: str, params: dict, seed: int, prompt: str, editor: str = "zimage"
) -> str:
    h = hashlib.sha256(img.tobytes())  # the version 1 digest, kept inside the version 2 key
    if editor == "zimage":
        h.update(json.dumps([tid, params, seed, prompt, ZIMAGE_REVISION], sort_keys=True).encode())
    else:
        rev = {"klein": KLEIN_REVISION, "klein_masked": KLEIN_REVISION}[editor]
        h.update(json.dumps([tid, params, seed, prompt, editor, rev], sort_keys=True).encode())
    return cache_key(h.hexdigest(), img.size, generation_settings())


def _cached(key: str) -> Image.Image | None:
    path = cache_dir() / f"{key}.png"
    if path.exists():
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    return None


def _store(key: str, img: Image.Image) -> None:
    path = cache_dir() / f"{key}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    img.save(tmp, "PNG")
    tmp.replace(path)


def _prompt(ctx: dict | None, mode: str = "source") -> str:
    """'source': a generated image's own prompt (neutral for photos); 'neutral': always neutral."""
    if mode == "neutral":
        return NEUTRAL_PROMPT
    return (ctx or {}).get("prompt") or NEUTRAL_PROMPT


def _model():
    from ..gpu import zimage

    return zimage(ZIMAGE_REVISION)


def _klein():
    from ..gpu import klein_inpaint

    return klein_inpaint(KLEIN_REVISION)


def _run_diffusion(kind: str, master: Image.Image, prompt: str, seed: int, **kw) -> Image.Image:
    from ..gpu import retry_oom

    if os.environ.get("PROVSURVIVE_CACHE_ONLY"):
        # CPU-stage runs must never start a diffusion model (e.g. 12 workers on a shared GPU)
        raise RuntimeError(f"generative cache miss for {kind} (PROVSURVIVE_CACHE_ONLY is set)")

    return retry_oom(_diffuse, kind, master, prompt, seed, **kw)


def _diffuse(kind: str, master: Image.Image, prompt: str, seed: int, **kw) -> Image.Image:
    """Diffusion at working resolution, resized back to the master's size."""
    ws = _work_size(*master.size)
    src = master.resize(ws, Image.LANCZOS) if ws != master.size else master
    mask = kw.get("mask")
    if mask is not None and ws != master.size:
        mask = mask.resize(ws, Image.NEAREST)
    if kw.get("editor", "zimage").startswith("klein"):
        k = _klein()
        if kw["editor"] == "klein_masked":
            # The pipeline always conditions on the input image as a reference, so with the
            # full image it re-renders the masked content almost unchanged. Graying out the
            # masked region removes that content from both the start latents and the reference.
            src = Image.composite(Image.new("RGB", src.size, (128, 128, 128)), src, mask)
        out = k.pipe(
            prompt=prompt,
            image=src,
            mask_image=mask,
            strength=1.0,
            height=ws[1],
            width=ws[0],
            num_inference_steps=KLEIN_STEPS,
            guidance_scale=KLEIN_GUIDANCE,
            generator=k.generator(seed),
        ).images[0]
    elif kind == "vae":
        out = _model().vae_roundtrip(src)
    else:
        m = _model()
        common = dict(
            prompt_embeds=m.embed(prompt),
            guidance_scale=ZIMAGE_GUIDANCE,
            generator=m.generator(seed),
        )
        if kind == "inpaint":
            out = m.inpaint(
                image=src,
                mask_image=mask,
                strength=1.0,
                num_inference_steps=INPAINT_STEPS,
                **common,
            ).images[0]
        else:
            out = m.img2img(
                image=src, strength=kw["strength"], num_inference_steps=IMG2IMG_STEPS, **common
            ).images[0]
    return out.resize(master.size, Image.LANCZOS) if out.size != master.size else out


def _inpaint(data, p, seed, ctx=None):
    master = decode_rgb(data)
    editor = p.get("editor", "zimage")
    prompt = _prompt(ctx, p.get("prompt", "source"))
    ell = ellipse_params(*master.size, p["area"], seed)
    mask = Image.fromarray((ellipse_mask(master.size, ell) * 255).astype(np.uint8))
    # raw output is shared between composite and raw variants
    key = _cache_key(master, "inpaint", {"area": p["area"]}, seed, prompt, editor)
    raw = _cached(key)
    if raw is None:
        raw = _run_diffusion("inpaint", master, prompt, seed, editor=editor, mask=mask)
        _store(key, raw)
    if p["composite"]:
        out = Image.composite(raw, master, mask.filter(ImageFilter.GaussianBlur(FEATHER_PX)))
    else:
        out = raw
    geometry = {
        "ellipse": ell,
        "size": list(master.size),
        "editor": editor,
        "revision": MODELS["zimage" if editor == "zimage" else "klein"][1],
        "steps": INPAINT_STEPS if editor == "zimage" else KLEIN_STEPS,
        "prompt": prompt,
        "work_size": list(_work_size(*master.size)),
        "cache_key": key,
    }
    return Output(encode(out, "png"), "png", alignment="masked", geometry=geometry)


def _img2img(data, p, seed, ctx=None):
    master = decode_rgb(data)
    prompt = _prompt(ctx)
    key = _cache_key(master, "img2img", {"strength": p["strength"]}, seed, prompt)
    out = _cached(key)
    if out is None:
        out = _run_diffusion("img2img", master, prompt, seed, strength=p["strength"])
        _store(key, out)
    geometry = {
        "editor": "zimage",
        "revision": ZIMAGE_REVISION,
        "prompt": prompt,
        # as diffusers' get_timesteps: n - int(n - n*s), i.e. 20s rounded up when not whole
        "effective_steps": IMG2IMG_STEPS - int(IMG2IMG_STEPS - IMG2IMG_STEPS * p["strength"]),
        "work_size": list(_work_size(*master.size)),
        "cache_key": key,
    }
    return Output(encode(out, "png"), "png", alignment="direct", geometry=geometry)


for _a in (5, 10, 20, 30, 40, 50):
    register(f"inpaint_{_a}", "generative", {"area": _a / 100, "composite": True}, gpu=True)(
        _inpaint
    )
register(
    "inpaint_30_raw",
    "generative",
    {"area": 0.30, "composite": False},
    gpu=True,
    seed_id="inpaint_30",
)(_inpaint)
for _s in (0.05, 0.1, 0.2, 0.3, 0.4, 0.6):
    register(f"img2img_{_s}", "generative", {"strength": _s}, gpu=True)(_img2img)


def mask_from_geometry(geometry: dict) -> np.ndarray:
    return ellipse_mask(tuple(geometry["size"]), geometry["ellipse"])


def _vae(data, p, seed, ctx=None):
    """Encode and decode through the editor's VAE only: separates the autoencoder's effect from
    denoising (the composite inpaints never pass unmasked pixels through it)."""
    master = decode_rgb(data)
    key = _cache_key(master, "vae", {}, 0, "")
    out = _cached(key)
    if out is None:
        out = _run_diffusion("vae", master, "", 0)
        _store(key, out)
    return Output(
        encode(out, "png"),
        "png",
        alignment="direct",
        geometry={"work_size": list(_work_size(*master.size)), "cache_key": key},
    )


# a diagnostic pass through the editor's VAE, not a generative edit: excluded from family
# statistics and from the disclosure ground truth (which keys on family == "generative")
register("vae_roundtrip", "diagnostic", {"vae": "zimage"}, gpu=True)(_vae)

# Second editor, same masks as the Z-Image inpaints (seed_id), for a paired comparison.
for _a in (5, 10, 20):
    register(
        f"inpaint_klein_{_a}",
        "generative",
        {"area": _a / 100, "composite": True, "editor": "klein"},
        gpu=True,
        seed_id=f"inpaint_{_a}",
    )(_inpaint)
# Same masks with the masked region grayed out in the input, so the second editor replaces the
# content instead of re-rendering it from the reference (see _diffuse).
for _a in (5, 10, 20):
    register(
        f"inpaint_kleinm_{_a}",
        "generative",
        {"area": _a / 100, "composite": True, "editor": "klein_masked"},
        gpu=True,
        seed_id=f"inpaint_{_a}",
    )(_inpaint)
# Same masks, one neutral prompt for every master (removes the prompt difference between
# generated masters, which use their own prompt, and photos).
register(
    "inpaint_10_neutral",
    "generative",
    {"area": 0.10, "composite": True, "prompt": "neutral"},
    gpu=True,
    seed_id="inpaint_10",
)(_inpaint)
