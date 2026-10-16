"""Run a suite: mark -> sign -> transform -> probe -> results."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import runinfo
from .bindings.base import SoftBinding
from .metrics import quality
from .probe import manifest as mprobe
from .probe.state import derive_state
from .resolver.store import ManifestRepository
from .schema import (
    AssetRef,
    BindingRef,
    CellResult,
    OutputRef,
    Quality,
    RecoveryObs,
    SoftBindingObs,
    TransformRef,
)
from .sign.c2pa_io import MIME, C2paSession, master_manifest
from .sources import SOURCE_TYPE, SourceImage
from .transforms.base import TransformSpec, cache_key, decode_rgb, encode
from .transforms.generative import model_info as generative_model_info

JPEG_QUALITY = 95  # masters stored as JPEG


def masters_cache_dir() -> Path:
    root = os.environ.get("PROVSURVIVE_CACHE", str(Path.home() / ".provsurvive" / "cache"))
    return Path(root) / "masters"


def marked_master(plugin: SoftBinding, image: Image.Image, payload: str) -> Image.Image:
    """Embed the soft binding, reusing a cached master if one exists.

    Learned watermark encoders are not bit-identical across CPU types (float kernels differ),
    and generative outputs are cached by master pixels. Caching the marked master itself makes
    every downstream byte reproducible on any machine that has the cache. The key covers the
    source pixels and size, the binding and its config, and the payload.
    """
    cfg = {k: v for k, v in plugin.config().items() if k != "device"}
    rgb = image.convert("RGB")
    h = hashlib.sha256(rgb.tobytes())  # the version 1 digest, kept inside the version 2 key
    h.update(json.dumps([plugin.name, plugin.alg, cfg, payload], sort_keys=True).encode())
    path = masters_cache_dir() / f"{cache_key(h.hexdigest(), rgb.size)}.png"
    if path.exists():
        img = Image.open(path)
        img.load()
        return img.convert("RGB")
    marked = plugin.embed(image, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    marked.save(tmp, "PNG")
    tmp.replace(path)
    return marked


def derive_seed(*parts: Any) -> int:
    h = hashlib.sha256("|".join(json.dumps(p, sort_keys=True) for p in parts).encode())
    return int(h.hexdigest()[:8], 16)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Master:
    source: SourceImage
    data: bytes  # signed master file (or unmarked, unsigned file for controls)
    image: Image.Image  # pixels of the master
    label: str | None
    payload: str | None
    control: bool
    value: str | None = None  # soft-binding assertion value (fingerprint code for fingerprints)
    fmt: str = "png"

    @property
    def truth_ai(self) -> bool:
        return self.source.kind == "generated"


@dataclass
class RunConfig:
    suite: str
    transforms: list[TransformSpec]
    seed: int = 20261031
    device: str = "cpu"
    lpips: bool = True
    thumbnail: bool = True
    controls: bool = True
    controls_gpu: bool = False  # generative transforms on unmarked controls double GPU time
    workers: int = 1


class Runner:
    def __init__(
        self,
        cfg: RunConfig,
        plugin: SoftBinding,
        session: C2paSession,
        out_dir: Path,
        repo_root: Path | None = None,
        plugin_spec: dict[str, Any] | None = None,
        repo: ManifestRepository | None = None,
    ) -> None:
        self.cfg = cfg
        self.plugin = plugin
        self.plugin_spec = plugin_spec
        self.session = session
        self.out_dir = out_dir
        self.repo = repo or ManifestRepository()
        self.repo_root = repo_root or Path.cwd()
        self.run_id = ""

    # --- preparation ---------------------------------------------------------

    def make_master(
        self, src: SourceImage, *, fmt: str = "png", use_cache: bool = True
    ) -> tuple[Master, dict[str, Any]]:
        """``fmt``: png, or jpeg at quality 95, whose decoded pixels then are the master's.
        ``use_cache``: reuse the masters cache (a prepared case keeps its own masters)."""
        t0 = time.perf_counter()
        payload = self.plugin.new_payload(derive_seed(self.cfg.seed, src.id, "payload"))
        if use_cache:
            marked = marked_master(self.plugin, src.image, payload)
        else:
            marked = self.plugin.embed(src.image, payload)
        unsigned = encode(marked, fmt, **({"quality": JPEG_QUALITY} if fmt == "jpeg" else {}))
        if fmt != "png":
            marked = decode_rgb(unsigned)
        value = self.plugin.assertion_value(marked, payload)
        manifest = master_manifest(
            title=src.id,
            digital_source_type=SOURCE_TYPE[src.kind],
            software_agent=src.meta.get("model"),
            soft_binding=(self.plugin.alg, value),
            watermarked=self.plugin.kind == "watermark",
            source={"asset_id": src.id, **{k: v for k, v in src.meta.items() if v is not None}},
        )
        signed, store = self.session.sign(unsigned, fmt, manifest)
        with self.session.read(signed, MIME[fmt]) as reader:
            label = json.loads(reader.json())["active_manifest"]
        self.repo.add(
            label=label,
            asset_id=src.id,
            mime=MIME[fmt],
            store=store,
            alg=self.plugin.alg,
            value=value,
        )
        cost = quality.compare(src.image, marked, with_lpips=self.cfg.lpips, device=self.cfg.device)
        record = {
            "id": src.id,
            "subset": src.subset,
            "kind": src.kind,
            "meta": src.meta,
            "source_sha256": sha256(encode(src.image, "png")),
            "master_sha256": sha256(signed),
            "master_format": fmt,
            "width": src.image.width,
            "height": src.image.height,
            "binding": {"plugin": self.plugin.name, "alg": self.plugin.alg, "value": value},
            "manifest_label": label,
            "manifest_bytes": len(signed) - len(unsigned),
            "mark_cost": cost,
            "timing_ms": {"mark_sign": (time.perf_counter() - t0) * 1000},
        }
        return Master(src, signed, marked, label, payload, False, value, fmt), record

    def make_control(self, src: SourceImage) -> Master:
        return Master(src, encode(src.image, "png"), src.image, None, None, control=True)

    # --- one cell ------------------------------------------------------------

    def run_cell(self, master: Master, t: TransformSpec) -> CellResult:
        seed = derive_seed(self.cfg.seed, master.source.id, *t.seed_key())
        cell_id = sha256(
            json.dumps(
                [master.source.id, t.id, t.params, self.plugin.name, master.control]
            ).encode()
        )[:16]
        base = dict(
            run_id=self.run_id,
            cell_id=cell_id,
            control=master.control,
            asset=AssetRef(
                id=master.source.id,
                subset=master.source.subset,
                master_format=master.fmt,
                master_sha256=sha256(master.data),
            ),
            transform=TransformRef(
                id=t.id,
                family=t.family,
                params=t.params,
                seed=seed,
                impl_version=t.impl_version,
                gpu=t.gpu,
            ),
            binding=BindingRef(
                plugin=self.plugin.name, alg=self.plugin.alg, config=self.plugin.config()
            ),
        )
        timing: dict[str, float] = {}
        try:
            t0 = time.perf_counter()
            ctx = {"kind": master.source.kind, "prompt": master.source.meta.get("prompt")}
            out = t(master.data, seed, ctx)
            timing["transform"] = (time.perf_counter() - t0) * 1000
            obs = self.probe(
                master,
                out.data,
                out.format,
                out.alignment,
                out.geometry,
                truth_ai=master.truth_ai or t.family == "generative",
                timing=timing,
            )
            return CellResult(
                **base, **obs, edit=out.geometry or None, timing_ms=timing, error=None
            )
        except Exception as e:
            return CellResult(
                **base,
                output=None,
                manifest=None,
                softbinding=None,
                recovery=None,
                state=None,
                quality=None,
                timing_ms=timing,
                error=f"{type(e).__name__}: {e}",
                status="error",
            )

    def probe(
        self,
        master: Master,
        data: bytes,
        fmt: str,
        alignment: str | None,
        geometry: dict[str, Any],
        *,
        truth_ai: bool | None,
        timing: dict[str, float],
    ) -> dict[str, Any]:
        """Observe one output file of ``master``: embedded manifest, soft binding, recovery,
        states and quality. ``truth_ai``: whether the output contains AI-generated content, None
        if unknown. ``alignment`` None: no basis to align output and master, so no quality."""
        t0 = time.perf_counter()
        mime = MIME[fmt]
        out_img = decode_rgb(data)
        output = OutputRef(
            format=fmt,
            width=out_img.width,
            height=out_img.height,
            bytes=len(data),
            sha256=sha256(data),
        )
        mobs, active = mprobe.observe(self.session, data, mime, master.label or "")
        sobs, decoded = self._observe_softbinding(master, out_img)
        robs, recovered = self._recover(master, decoded, data, mime)
        timing["probe"] = (time.perf_counter() - t0) * 1000

        available = None
        if mobs.present and mobs.signature_ok and mobs.hard_binding == "match":
            # intact or superseded: the embedded manifest describes these bytes
            available = mprobe.digital_source_type(active)
        elif robs.outcome == "correct":
            available = mprobe.digital_source_type(recovered)
        state = derive_state(
            mobs, robs, truth_involves_ai=truth_ai, available_source_type=available
        )

        qual = None
        if alignment is not None:
            t0 = time.perf_counter()
            ref, cand, retained = quality.align(master.image, out_img, alignment, geometry)
            q = quality.compare(ref, cand, with_lpips=self.cfg.lpips, device=self.cfg.device)
            timing["quality"] = (time.perf_counter() - t0) * 1000
            inside = outside = None
            if alignment == "masked":
                from .transforms.generative import mask_from_geometry

                inside, outside = quality.compare_masked(ref, cand, mask_from_geometry(geometry))
            qual = Quality(
                alignment=alignment,
                retained_area=retained,
                inside_mask=inside,
                outside_mask=outside,
                **q,
            )
        return dict(
            output=output,
            manifest=mobs,
            softbinding=sobs,
            recovery=robs,
            state=state,
            quality=qual,
        )

    def _observe_softbinding(self, master: Master, img: Image.Image):
        decoded = self.plugin.decode(img)
        bit_acc = None
        if master.payload is not None and decoded.raw_bits is not None:
            coded = self.plugin.coded_bits(master.payload)
            if coded is not None and len(coded) == len(decoded.raw_bits):
                bit_acc = float(np.mean(np.array(coded) == np.array(decoded.raw_bits)))
        expected = getattr(self.plugin, "expected_schema", None)
        distance = threshold = None
        if hasattr(self.plugin, "distance") and master.value is not None:
            distance = float(self.plugin.distance(decoded, master.value))
            threshold = float(self.plugin.threshold)
        return SoftBindingObs(
            detected=decoded.detected,
            payload=decoded.payload,
            ecc_schema=decoded.schema,
            expected_schema=(
                decoded.schema == expected() if decoded.detected and expected else None
            ),
            payload_match=(decoded.payload == master.payload) if master.payload else None,
            bit_accuracy=bit_acc,
            distance=distance,
            distance_threshold=threshold,
        ), decoded

    def _recover(self, master: Master, decoded, data: bytes, mime: str):
        candidates = self.repo.match_by_binding(self.plugin, decoded)
        if not candidates:
            return RecoveryObs(outcome="none", candidates=[], recovered_signature_ok=None), None
        if len(candidates) > 1 and candidates[1].score == candidates[0].score:
            # equally good manifests at the top: which one a resolver returns is arbitrary
            return (
                RecoveryObs(
                    outcome="ambiguous", candidates=candidates, recovered_signature_ok=None
                ),
                None,
            )
        top = candidates[0].label
        outcome = "correct" if top == master.label else "wrong"
        with self.session.read_manifest_store(self.repo.get(top)[1], mime, data) as reader:
            codes = mprobe.status_codes(reader.get_validation_results())
            sig_ok = mprobe.signature_ok(codes)
            rec_hb = mprobe.hard_binding(codes)
            recovered = mprobe.active_manifest(reader)
        return RecoveryObs(
            outcome=outcome,
            candidates=candidates,
            recovered_signature_ok=sig_ok,
            recovered_hard_binding=rec_hb,
        ), recovered

    # --- whole run -----------------------------------------------------------

    def _cells_for(self, master: Master, transform_ids: list[str]) -> list[tuple[str, bool]]:
        from .transforms import base as tbase

        out = []
        for tid in transform_ids:
            r = self.run_cell(master, tbase.get(tid))
            out.append((r.model_dump_json(), r.error is not None))
        return out

    def _run_jobs(self, jobs, run_dir: Path) -> list[list[tuple[str, bool]]]:
        if self.cfg.workers <= 1 or self.plugin_spec is None:
            return [self._cells_for(m, ids) for m, ids in jobs]
        from concurrent.futures import ProcessPoolExecutor, as_completed

        init = (
            self.cfg,
            self.plugin_spec,
            self.session.ca,
            self.session.thumbnail,
            str(run_dir / "manifests.sqlite"),
            self.run_id,
            str(self.out_dir),
            str(self.repo_root),
        )
        with ProcessPoolExecutor(self.cfg.workers, initializer=_worker_init, initargs=init) as ex:
            futures = [ex.submit(_worker_job, m, ids) for m, ids in jobs]
            t0 = time.perf_counter()
            for done, _ in enumerate(as_completed(futures), 1):
                if done % 10 == 0 or done == len(futures):
                    rate = done / (time.perf_counter() - t0)
                    eta = (len(futures) - done) / rate
                    print(
                        f"[cells] {done}/{len(futures)} jobs, eta {eta / 60:.0f} min",
                        file=sys.stderr,
                        flush=True,
                    )
            return [f.result() for f in futures]

    def run(self, sources: list[SourceImage]) -> Path:
        started = dt.datetime.now(dt.timezone.utc)
        # the code and packages that run the cells, read before the first cell (the working
        # tree may change while a long run is in progress)
        tool, environment = runinfo.tool_info(self.repo_root), runinfo.environment()
        cfg_blob = {
            "suite": self.cfg.suite,
            "transforms": [[t.id, t.params, t.impl_version] for t in self.cfg.transforms],
            "seed": self.cfg.seed,
            "lpips": self.cfg.lpips,
            "thumbnail": self.cfg.thumbnail,
            "controls": self.cfg.controls,
            "controls_gpu": self.cfg.controls_gpu,
            "binding": self.plugin.config(),
            "alg": self.plugin.alg,
            "sources": [s.id for s in sources],
        }
        cfg_sha = sha256(json.dumps(cfg_blob, sort_keys=True).encode())
        self.run_id = started.strftime("%Y-%m-%dT%H-%M-%SZ") + "-" + cfg_sha[:6]
        run_dir = self.out_dir / self.run_id
        run_dir.mkdir(parents=True, exist_ok=False)

        self.repo = ManifestRepository(run_dir / "manifests.sqlite")

        stage: dict[str, float] = {}
        t0 = time.perf_counter()
        masters, assets = [], []
        for i, src in enumerate(sources, 1):
            m, rec = self.make_master(src)
            masters.append(m)
            assets.append(rec)
            if i % 20 == 0 or i == len(sources):
                print(f"[masters] {i}/{len(sources)}", file=sys.stderr, flush=True)
        self.repo.db.commit()
        stage["mark_sign"] = time.perf_counter() - t0

        jobs: list[tuple[Master, list[str]]] = [
            (m, [t.id for t in self.cfg.transforms]) for m in masters
        ]
        # an unmarked copy has its master's fingerprint and rightly finds its manifest (§6)
        skip_controls = None
        if self.cfg.controls and self.plugin.kind == "fingerprint":
            skip_controls = "fingerprint: unmarked copies find their masters; see misattribution"
        elif self.cfg.controls:
            ctrl_ids = [t.id for t in self.cfg.transforms if self.cfg.controls_gpu or not t.gpu]
            jobs += [(self.make_control(src), ctrl_ids) for src in sources]

        t0 = time.perf_counter()
        results = self._run_jobs(jobs, run_dir)
        stage["cells"] = time.perf_counter() - t0

        n_cells = n_ctrl = n_err = 0
        with (
            open(run_dir / "results.jsonl", "w") as fr,
            open(run_dir / "controls.jsonl", "w") as fc,
        ):
            for (m, _), lines in zip(jobs, results, strict=True):
                for line, err in lines:
                    (fc if m.control else fr).write(line + "\n")
                    n_err += err
                    if m.control:
                        n_ctrl += 1
                    else:
                        n_cells += 1

        with open(run_dir / "assets.jsonl", "w") as f:
            for rec in assets:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

        finished = dt.datetime.now(dt.timezone.utc)
        run_json = {
            "schema_version": "1.1",  # 1.1: generative, binding_provenance, CPU model
            "run_id": self.run_id,
            "tool": tool,
            "config": {"suite": self.cfg.suite, "config_sha256": cfg_sha, "resolved": cfg_blob},
            "seeds": {
                "global": self.cfg.seed,
                "per_cell": (
                    "sha256(global|asset_id|seed_key)[:8]; seed_key = the transform's seed_id "
                    "or id and its params, minus those that only change how a shared draw is "
                    "used, so paired variants share seeds"
                ),
            },
            "environment": environment,
            "device": self.cfg.device,
            # what a binding actually loaded, where it has more than its config (PixelSeal)
            "binding_provenance": (
                self.plugin.provenance() if hasattr(self.plugin, "provenance") else None
            ),
            # generative edits come from a separate GPU stage through the cache; what they
            # depend on (models, revisions, steps) is recorded here and in each cell's edit
            "generative": (
                generative_model_info() if any(t.gpu for t in self.cfg.transforms) else None
            ),
            "signing": {
                "ca_root_fingerprint_sha256": self.session.ca.root_fingerprint,
                "alg": "es256",
                "tsa": None,
                "thumbnail": self.cfg.thumbnail,
                "test_only": True,
            },
            "timing": {
                "started": started.isoformat(),
                "finished": finished.isoformat(),
                "stages_s": stage,
            },
            "controls_not_run": skip_controls,
            "counts": {
                "assets": len(sources),
                "transforms": len(self.cfg.transforms),
                "cells": n_cells,
                "controls": n_ctrl,
                "errors": n_err,
            },
        }
        (run_dir / "run.json").write_text(json.dumps(run_json, indent=2) + "\n")
        return run_dir


_WORKER: dict[str, Runner] = {}


def _worker_init(cfg, plugin_spec, ca, thumbnail, repo_path, run_id, out_dir, repo_root) -> None:
    import torch

    from .bindings.registry import make_plugin

    torch.set_num_threads(1)
    runner = Runner(
        cfg,
        make_plugin(plugin_spec),
        C2paSession(ca, thumbnail=thumbnail),
        Path(out_dir),
        repo_root=Path(repo_root),
        plugin_spec=plugin_spec,
        repo=ManifestRepository(repo_path),
    )
    runner.run_id = run_id
    _WORKER["runner"] = runner


def _worker_job(master: Master, transform_ids: list[str]) -> list[tuple[str, bool]]:
    return _WORKER["runner"]._cells_for(master, transform_ids)
