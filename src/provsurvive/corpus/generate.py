"""Generate the T2I half of the corpus from the packaged prompts.yaml (GPU, [gen] extra)."""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import yaml

PROMPTS_FILE = Path(__file__).with_name("prompts.yaml")  # the committed prompt list
SIZES = [(1024, 1024), (1216, 832), (832, 1216)]


def load_prompts(path: Path) -> tuple[dict, list[dict]]:
    """Generator specs and one entry per image; ids, generators, sizes and seeds follow the
    index as prompts.yaml describes."""
    doc = yaml.safe_load(path.read_text())
    gens = list(doc["generators"])
    pairs = [(cat, p) for cat, ps in doc["prompts"].items() for p in ps]
    images = [
        {
            "id": f"gen-{i:03d}",
            "category": cat,
            "prompt": prompt,
            "generator": gens[i % 2],
            "width": SIZES[i % 3][0],
            "height": SIZES[i % 3][1],
            "seed": 1000 + i,
        }
        for i, (cat, prompt) in enumerate(pairs)
    ]
    return doc["generators"], images


def _sample(model, gen_name: str, prompt: str, common: dict):
    if gen_name == "zimage-turbo":
        return model.t2i(prompt_embeds=model.embed(prompt), **common)
    return model.pipe(prompt=prompt, **common)


def generate(prompts_path: Path, out_dir: Path, only: str | None = None, limit: int | None = None):
    import torch

    from ..gpu import FullFlux2Klein, StagedZImage, retry_oom

    specs, images = load_prompts(prompts_path)
    images = images[:limit] if limit else images
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.json"
    index = {e["id"]: e for e in json.loads(index_path.read_text())} if index_path.exists() else {}

    for gen_name, spec in specs.items():
        if only and gen_name != only:
            continue
        todo = [e for e in images if e["generator"] == gen_name and e["id"] not in index]
        if not todo:
            continue
        model = (StagedZImage if gen_name == "zimage-turbo" else FullFlux2Klein)(spec["revision"])
        if gen_name == "zimage-turbo":
            model.prepare([e["prompt"] for e in todo])
        for e in todo:
            t0 = time.time()
            common = dict(
                height=e["height"],
                width=e["width"],
                num_inference_steps=spec["steps"],
                guidance_scale=spec["guidance_scale"],
                generator=model.generator(e["seed"]),
            )
            img = retry_oom(_sample, model, gen_name, e["prompt"], common).images[0]
            path = out_dir / f"{e['id']}.png"
            img.save(path)
            index[e["id"]] = {
                "file": path.name,
                "id": e["id"],
                "kind": "generated",
                "category": e["category"],
                "prompt": e["prompt"],
                "seed": e["seed"],
                "model": f"{spec['repo']}@{spec['revision']}",
                "license": "CC0-1.0",
                "model_license": spec["license"],
                "width": img.width,
                "height": img.height,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "seconds": round(time.time() - t0, 2),
            }
            index_path.write_text(
                json.dumps(sorted(index.values(), key=lambda x: x["id"]), indent=1)
            )
            print(f"{e['id']} {gen_name} {img.size} {time.time() - t0:.1f}s", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return index_path
