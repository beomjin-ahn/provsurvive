"""Aggregate results.jsonl into summary.csv and summary.md, with Wilson 95% CIs."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from statistics import mean

EMBEDDED = ["intact", "superseded", "stale", "invalid", "absent"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _mean(xs: list[float | None]) -> float | None:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def aggregate(cells: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for c in cells:
        groups[(c["transform"]["family"], c["transform"]["id"])].append(c)
    rows = []
    for (family, transform), cs in groups.items():
        ok = [c for c in cs if c["error"] is None]
        n = len(ok)
        emb = Counter(c["state"]["embedded"] for c in ok)
        rec = Counter(c["state"]["recovery"] for c in ok)
        avail = sum(c["state"]["provenance_available"] for c in ok)
        lo, hi = wilson(avail, n) if n else (None, None)
        rows.append(
            {
                "family": family,
                "transform": transform,
                "n": n,
                "errors": len(cs) - n,
                **{f"embedded_{e}": emb[e] for e in EMBEDDED},
                **{f"recovery_{r}": rec[r] for r in ("correct", "wrong", "ambiguous")},
                "detected": sum(c["softbinding"]["detected"] for c in ok),
                "provenance_available": avail,
                "available_rate": avail / n if n else None,
                "available_ci_lo": lo,
                "available_ci_hi": hi,
                "bit_accuracy_mean": _mean([c["softbinding"]["bit_accuracy"] for c in ok]),
                "psnr_mean": _mean([c["quality"]["psnr"] for c in ok]),
                "ssim_mean": _mean([c["quality"]["ssim"] for c in ok]),
                "lpips_mean": _mean([c["quality"]["lpips"] for c in ok]),
            }
        )
    return rows


def schema_detections(ok: list[dict]) -> int | None:
    """False detections in the deployed ECC schema; None if no cell recorded the schema (runs
    before schema 1.1, or a binding without one)."""
    if not any(
        "encoding" in c["binding"]["config"] and "ecc_schema" in c["softbinding"] for c in ok
    ):
        return None
    return sum(bool(c["softbinding"].get("expected_schema")) for c in ok)


def _fmt(x, nd=3) -> str:
    """Round half up, as the paper does (format() rounds 0.705 to 0.70)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if not isinstance(x, float):
        return str(x)
    # round to 9 decimals first: 141/200 is 0.70499999... in binary floating point
    q = Decimal(str(round(x, 9))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    return f"{q:.{nd}f}"


def write(run_dir: Path) -> Path:
    run = json.loads((run_dir / "run.json").read_text())
    rows = aggregate(_load(run_dir / "results.jsonl"))
    ctrl = _load(run_dir / "controls.jsonl")
    assets = _load(run_dir / "assets.jsonl")

    if rows:
        _write_csv(run_dir / "summary.csv", rows, list(rows[0]))

    lines = [
        f"# provsurvive run `{run['run_id']}`",
        "",
        f"Suite `{run['config']['suite']}` · {run['counts']['assets']} assets × "
        f"{run['counts']['transforms']} transforms · "
        f"binding `{run['config']['resolved']['alg']}` · "
        f"errors: {run['counts']['errors']}",
        "",
        "## Survival by transform",
        "",
        "| family | transform | n | intact | superseded | stale | invalid | absent | recovered | "
        "wrong | ambiguous | available (95% CI) | bit acc | PSNR | SSIM | LPIPS |",
        "|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|--:|",
    ]
    counts = [*(f"embedded_{e}" for e in EMBEDDED), "recovery_correct", "recovery_wrong",
              "recovery_ambiguous"]  # fmt: skip
    for r in rows:
        lo, hi = _fmt(r["available_ci_lo"], 2), _fmt(r["available_ci_hi"], 2)
        lines.append(
            f"| {r['family']} | `{r['transform']}` | {r['n']} | "
            + " | ".join(str(r[k]) for k in counts)
            + f" | {_fmt(r['available_rate'], 2)} ({lo}–{hi}) | {_fmt(r['bit_accuracy_mean'])} | "
            f"{_fmt(r['psnr_mean'], 1)} | {_fmt(r['ssim_mean'])} | {_fmt(r['lpips_mean'])} |"
        )
    lines += [
        "",
        'PSNR "—" for a transform means every output was pixel-identical to its reference.',
        "",
        "## Cost of the mark (source → master)",
        "",
        "| asset | PSNR | SSIM | LPIPS | manifest bytes |",
        "|---|--:|--:|--:|--:|",
    ]
    for a in assets:
        c = a["mark_cost"]
        lines.append(
            f"| {a['id']} | {_fmt(c['psnr'], 1)} | {_fmt(c['ssim'])} | {_fmt(c['lpips'])} | "
            f"{a['manifest_bytes']} |"
        )
    if run.get("controls_not_run"):
        lines += ["", "## Controls", "", f"Not run: {run['controls_not_run']}."]
    ok = [c for c in ctrl if c["error"] is None]
    if ctrl and not ok:
        lines += ["", "## Controls", "", f"All {len(ctrl)} control cells failed: nothing measured."]
    elif ctrl:
        fd = sum(c["softbinding"]["detected"] for c in ok)
        fm = sum(c["state"]["recovery"] != "none" for c in ok)
        lo_d, hi_d = wilson(fd, len(ok))
        lo_m, hi_m = wilson(fm, len(ok))
        fd_assets = {c["asset"]["id"] for c in ok if c["softbinding"]["detected"]}
        n_assets = len({c["asset"]["id"] for c in ok})
        schema = schema_detections(ok)
        lines += [
            "",
            "## Controls (unmarked, unsigned sources through the same transforms)",
            "",
            f"- Images with any false detection: {len(fd_assets)}/{n_assets}"
            + (f" ({', '.join(sorted(fd_assets))})" if fd_assets else ""),
            f"- False detection: {fd}/{len(ok)} (95% CI {_fmt(lo_d, 3)}–{_fmt(hi_d, 3)})",
            "- False detection in the deployed ECC schema only: "
            + (
                "n/a (no ECC schema recorded for this soft binding)"
                if schema is None
                else f"{schema}/{len(ok)} (detections in other auto-detected schemas are discarded)"
            ),
            f"- False match (any manifest returned): {fm}/{len(ok)} "
            f"(95% CI {_fmt(lo_m, 3)}–{_fmt(hi_m, 3)})",
        ]
    out = run_dir / "summary.md"
    out.write_text("\n".join(lines) + "\n")
    return out
