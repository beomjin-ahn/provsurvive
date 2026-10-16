"""Emit the benchmarking record proposed for the C2PA soft binding registry, and the results
file it points to.

The record (``benchmarkingInformation`` under a registry entry's ``entryMetadata``) has four
fields: a description, the covered media types, the benchmark date, and the URL and SHA-256
digest of a results file. What the results file contains is left open.

``benchmark-results.json`` is that file for a run: the numbers a registry reader needs, namely
recovery per edit family with image-clustered intervals, the disclosure-inconsistency rate after
generative edits, false detections on unmarked controls with the detection rule, misattribution
within the corpus, and the mark's quality cost. ``benchmarking-information.json`` is the
one-item ``benchmarkingInformation`` list that points to it. Its ``url`` is a URN placeholder
until the results file is published; ``PROVSURVIVE_RESULTS_URL_BASE`` or ``write(results_url=)``
supplies the location.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

from .. import HOMEPAGE
from ..metrics.stats import cluster_bootstrap
from .summary import _load, schema_detections, wilson

RESULTS_FILE = "benchmark-results.json"
RECORD_FILE = "benchmarking-information.json"
# v2: expected / evaluated / errors next to every rate; controls that ran but all failed
RESULTS_SCHEMA = "provsurvive.benchmark-results.v2"


def build_results(run_dir: Path) -> dict:
    """The measured contents of a registry results file. Rates are over the evaluated cells;
    the cells that failed are counted next to them, and a rate with no evaluated cell is null."""
    run = json.loads((run_dir / "run.json").read_text())
    every = _load(run_dir / "results.jsonl")
    cells = [c for c in every if c["error"] is None]
    fam = defaultdict(list)
    for c in every:
        if c["transform"]["family"] not in ("control", "diagnostic"):
            fam[c["transform"]["family"]].append(c)

    families = {}
    for f, cs in sorted(fam.items()):
        # a family pools several edits of each image, so the interval resamples images
        one_edit = len({c["transform"]["id"] for c in cs}) == 1
        families[f] = _recovery(cs, one_edit) | {
            "by_subset": {
                sub: _recovery([c for c in cs if c["asset"]["subset"] == sub], one_edit)
                for sub in sorted({c["asset"]["subset"] for c in cs})
            }
        }
    # photographs carry a capture claim, so a recovered manifest after a generative edit is
    # disclosure-inconsistent; generated masters already declare AI and cannot be
    gen_photo = [
        c
        for c in every
        if c["transform"]["family"] == "generative" and c["asset"]["subset"] == "photo"
    ]
    inconsistent = {}
    for t in sorted({c["transform"]["id"] for c in gen_photo}):
        cs = [c for c in gen_photo if c["transform"]["id"] == t]
        ok = [c for c in cs if c["error"] is None]
        rate = {"rate": None, "ci95": None}
        if ok:
            # one edit, one cell per image (a bootstrap of 0/100 has zero width)
            r, lo, hi, _ = _single_interval(
                ok, lambda c: c["state"]["disclosure_consistent"] is False
            )
            rate = {"rate": round(r, 4), "ci95": [round(lo, 4), round(hi, 4)]}
        inconsistent[t] = _counts(cs) | rate | {"ci_method": "Wilson"}

    assets = _load(run_dir / "assets.jsonl")
    cost = {}
    for k in ("psnr", "ssim", "lpips"):
        v = [a["mark_cost"][k] for a in assets if a.get("mark_cost", {}).get(k) is not None]
        cost[k] = round(sum(v) / len(v), 4) if v else None
    # a fingerprint leaves the pixels untouched: PSNR is infinite, which JSON cannot hold
    cost["pixels_changed"] = any(a.get("mark_cost", {}).get("psnr") is not None for a in assets)
    if not cost["pixels_changed"]:
        cost["psnr_null"] = "identical pixels: PSNR is infinite, which JSON cannot hold"

    return {
        "schema": RESULTS_SCHEMA,
        "tool": {
            "name": "provsurvive",
            "version": run["tool"]["version"],
            "git_commit": run["tool"].get("git_commit"),
        },
        "run_id": run["run_id"],
        "suite": run["config"]["suite"],
        "alg": run["config"]["resolved"]["alg"],
        "binding_config": run["config"]["resolved"]["binding"],
        "methodology": f"{HOMEPAGE}/blob/main/docs/METHODOLOGY.md",
        "corpus": corpus_description(assets),
        "cells": _counts(every),
        "rates_over": "evaluated cells",
        "recovery_by_family": families,
        "disclosure_inconsistent_after_generative_edit": {
            "population": "photographs (digitalCapture masters)",
            "by_edit": inconsistent,
        },
        "controls": _controls(
            _load(run_dir / "controls.jsonl"),
            run["config"]["resolved"],
            run.get("controls_not_run"),
        ),
        "misattribution": {
            "registry_size": run["counts"]["assets"],
            "within_corpus_cells": sum(c["state"]["misattributed"] for c in cells),
        },
        "mark_cost": cost,
    }


def _counts(cells: list[dict]) -> dict:
    ok = sum(c["error"] is None for c in cells)
    return {"expected": len(cells), "evaluated": ok, "errors": len(cells) - ok}


def _recovery(cells: list[dict], one_edit: bool) -> dict:
    """Recovery of the original manifest over the evaluated cells, with its interval."""
    ok = [c for c in cells if c["error"] is None]
    if not ok:
        return _counts(cells) | {"recovered": None, "ci95": None, "ci_method": None}
    interval = _single_interval if one_edit else _pooled_interval
    r, lo, hi, method = interval(ok, lambda c: c["state"]["recovery"] == "correct")
    return _counts(cells) | {
        "recovered": round(r, 4),
        "ci95": [round(lo, 4), round(hi, 4)],
        "ci_method": method,
    }


def _single_interval(cells: list[dict], pred) -> tuple[float, float, float, str]:
    """One edit, one cell per image: a Wilson interval."""
    k = sum(map(pred, cells))
    lo, hi = wilson(k, len(cells))
    return k / len(cells), lo, hi, "Wilson"


def _pooled_interval(cells: list[dict], pred) -> tuple[float, float, float, str]:
    """Image-clustered bootstrap for a rate pooled over several edits of each image. At 0% or
    100% every resample is identical and the interval collapses; then report a Wilson interval
    that counts each image once (conservative, since an image's cells are not independent)."""
    r, lo, hi = cluster_bootstrap(cells, pred)
    if lo < hi:
        return r, lo, hi, "image-clustered percentile bootstrap"
    n_img = len({c["asset"]["id"] for c in cells})
    lo, hi = wilson(round(r * n_img), n_img)
    return r, lo, hi, "Wilson over images (the bootstrap is degenerate at this rate)"


def detection_flag(alg: str) -> str:
    """What the decoder's `detected` flag is for this algorithm."""
    from ..bindings import pixelseal, trustmark

    if alg.startswith("com.adobe.trustmark."):
        return trustmark.DETECTION_FLAG
    if alg == pixelseal.PixelSealBinding.alg:
        return pixelseal.DETECTION_FLAG
    return "the decoder's detection flag"


def _controls(every: list[dict], resolved: dict, not_run: str | None = None) -> dict:
    flag = detection_flag(resolved["alg"])
    if resolved["binding"].get("rotation_search"):
        flag += ", decoded with rotation search over four orientations"
    rule = (
        f"false detection: {flag}, on unmarked, unsigned copies of every source; "
        "false match: the resolver returns any registered manifest"
    )
    if not every:  # not run: nothing measured, so null, not zero
        return {"detection_rule": rule, "run": False} | ({"why_not": not_run} if not_run else {})
    ctrl = [c for c in every if c["error"] is None]
    counts = {
        "false_detection_cells": sum(c["softbinding"]["detected"] for c in ctrl),
        "false_detection_images": len(
            {c["asset"]["id"] for c in ctrl if c["softbinding"]["detected"]}
        ),
        "false_detection_deployed_schema_cells": schema_detections(ctrl),
        "false_match_cells": sum(c["state"]["recovery"] != "none" for c in ctrl),
    }
    if not ctrl:  # run, but every cell failed
        counts = dict.fromkeys(counts)
    return {"detection_rule": rule, "run": True, **_counts(every), **counts}


def corpus_description(assets: list[dict]) -> str:
    """What the run's masters are, counted from its assets (a run may use one subset only)."""
    kinds = {
        "generated": "generated with Apache-2.0 T2I models",
        "photo": "CC0 Wikimedia Commons photographs",
        "synthetic": "procedural test images",
    }
    n = {k: sum(a["kind"] == k for a in assets) for k in kinds}
    parts = [f"{n[k]} {what}" for k, what in kinds.items() if n[k]]
    return f"provsurvive corpus: {len(assets)} images ({', '.join(parts)})"


def results_bytes(results: dict) -> bytes:
    """The exact bytes of the results file; the record's digest is computed over them."""
    return (json.dumps(results, indent=2) + "\n").encode()


def results_url(run_id: str, url: str | None = None) -> str:
    """The published location of the results file, or a URN placeholder until it exists."""
    if url:
        return url
    base = os.environ.get("PROVSURVIVE_RESULTS_URL_BASE")
    if base:
        return f"{base.rstrip('/')}/{run_id}/{RESULTS_FILE}"
    return f"urn:provsurvive:run:{run_id}:{RESULTS_FILE}"


def build_record(results: dict, url: str | None = None) -> list[dict]:
    """A one-item ``benchmarkingInformation`` list for the registry entry of ``results['alg']``."""
    run_id = results["run_id"]
    return [
        {
            "description": (
                f"provsurvive {results['tool']['version']}, suite '{results['suite']}', "
                f"{results['alg']}, {results['corpus'].removeprefix('provsurvive corpus: ')}: "
                "recovery of the original manifest per "
                "edit family, disclosure-inconsistent recoveries after generative edits, false "
                "detections on unmarked controls, misattribution, and mark cost"
            ),
            "coveredMediaTypes": ["image"],
            "benchmarkDate": run_id[:10],
            "results": {
                "url": results_url(run_id, url),
                "sha256": hashlib.sha256(results_bytes(results)).hexdigest(),
            },
        }
    ]


def write(run_dir: Path, results_url: str | None = None) -> Path:
    results = build_results(run_dir)
    (run_dir / RESULTS_FILE).write_bytes(results_bytes(results))
    out = run_dir / RECORD_FILE
    out.write_text(json.dumps(build_record(results, results_url), indent=2) + "\n")
    return out
