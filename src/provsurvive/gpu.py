"""Diffusion model loading with a placement that fits a 24 GB GPU.

Z-Image-Turbo in bf16 is ~19 GiB (transformer 11.5 + Qwen3-4B text encoder 7.5 + VAE 0.2),
which does not fit next to 1024² activations. The text encoder therefore visits the GPU only
to encode prompts; the transformer and VAE stay resident. Sibling pipelines (img2img,
inpaint) share module objects via ``cls(**components)``: ``from_pipe`` re-places modules and
pulled the text encoder back onto the GPU in the spike.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any


class StagedZImage:
    """Z-Image-Turbo T2I / img2img / inpaint with staged GPU placement.

    The text encoder and the transformer are never on the GPU together: prompts are encoded
    up front (``prepare``) or, for an unseen prompt, by temporarily moving the transformer off.
    Peak use is then ~14 GiB, which leaves room for other jobs on a shared 24 GB GPU.
    """

    repo = "Tongyi-MAI/Z-Image-Turbo"

    def __init__(self, revision: str | None, device: str = "cuda") -> None:
        import torch
        from diffusers import ZImageImg2ImgPipeline, ZImageInpaintPipeline, ZImagePipeline

        self.torch = torch
        self.device = device
        self.revision = revision
        base = ZImagePipeline.from_pretrained(self.repo, revision=revision, dtype=torch.bfloat16)

        def sibling(cls):
            comps = dict(base.components)
            comps["scheduler"] = base.scheduler.__class__.from_config(base.scheduler.config)
            return cls(**comps)

        self.t2i = base
        self.img2img = sibling(ZImageImg2ImgPipeline)
        self.inpaint = sibling(ZImageInpaintPipeline)
        self._embeds: dict[str, Any] = {}
        self._denoiser_on_gpu = False

    def _denoiser(self, on_gpu: bool) -> None:
        if on_gpu == self._denoiser_on_gpu:
            return
        target = self.device if on_gpu else "cpu"
        self.t2i.transformer.to(target)
        self.t2i.vae.to(target)
        self._denoiser_on_gpu = on_gpu
        self.torch.cuda.empty_cache()

    def prepare(self, prompts: list[str]) -> None:
        """Encode all prompts with only the text encoder on the GPU."""
        todo = [p for p in dict.fromkeys(prompts) if p not in self._embeds]
        if not todo:
            return
        self._denoiser(False)
        te = self.t2i.text_encoder
        te.to(self.device)
        with self.torch.no_grad():
            for p in todo:
                pe = self.t2i.encode_prompt(
                    p, device=self.device, do_classifier_free_guidance=False
                )[0]
                self._embeds[p] = [x.to("cpu") for x in pe]
        te.to("cpu")
        self.torch.cuda.empty_cache()

    def embed(self, prompt: str):
        if prompt not in self._embeds:
            self.prepare([prompt])
        self._denoiser(True)
        return [x.to(self.device) for x in self._embeds[prompt]]

    def generator(self, seed: int):
        return self.torch.Generator("cpu").manual_seed(seed)

    def vae_roundtrip(self, img):
        """VAE encode/decode only (no denoising), at the given size."""
        self._denoiser(True)
        pipe = self.t2i
        vae = pipe.vae
        x = pipe.image_processor.preprocess(img).to(self.device, dtype=vae.dtype)
        with self.torch.no_grad():
            lat = vae.encode(x).latent_dist.mode()
            y = vae.decode(lat).sample
        return pipe.image_processor.postprocess(y, output_type="pil")[0]


class KleinInpaint:
    """FLUX.2-klein-4B inpainting (second editor). Model CPU offload keeps it under ~10 GiB."""

    repo = "black-forest-labs/FLUX.2-klein-4B"

    def __init__(self, revision: str | None, device: str = "cuda") -> None:
        import torch
        from diffusers import Flux2KleinInpaintPipeline

        self.torch = torch
        self.pipe = Flux2KleinInpaintPipeline.from_pretrained(
            self.repo, revision=revision, dtype=torch.bfloat16
        )
        self.pipe.enable_model_cpu_offload()

    def generator(self, seed: int):
        return self.torch.Generator("cpu").manual_seed(seed)


@lru_cache(maxsize=1)
def klein_inpaint(revision: str | None = None) -> KleinInpaint:
    return KleinInpaint(revision)


def retry_oom(fn, *args, attempts: int = 6, wait_s: float = 30.0, **kwargs):
    """Retry on CUDA OOM (another job on a shared GPU may hold memory for a while)."""
    import time

    import torch

    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except torch.OutOfMemoryError:
            if i == attempts - 1:
                raise
            torch.cuda.empty_cache()
            time.sleep(wait_s * (i + 1))


class FullFlux2Klein:
    """FLUX.2-klein-4B fits entirely on a 24 GB GPU (~16 GiB bf16)."""

    repo = "black-forest-labs/FLUX.2-klein-4B"

    def __init__(self, revision: str | None, device: str = "cuda") -> None:
        import torch
        from diffusers import Flux2KleinPipeline

        self.torch = torch
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.repo, revision=revision, dtype=torch.bfloat16
        ).to(device)

    def generator(self, seed: int):
        return self.torch.Generator("cpu").manual_seed(seed)


@lru_cache(maxsize=1)
def zimage(revision: str | None = None, device: str = "cuda") -> StagedZImage:
    return StagedZImage(revision, device)
