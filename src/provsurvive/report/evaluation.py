"""Reports for an evaluation of externally processed files (``provsurvive evaluate``).

files.csv has one row per setting and master, missing and failed files included; summary.csv
and summary.md aggregate by setting. Each count has its own population, stated in summary.md:
manifest states and recovery outcomes count every evaluated file; the recovery rate counts the
evaluated files whose master was recoverable unprocessed (its baseline); quality counts the
files with a quality measurement. A rate with nothing to count is n/a, never 0.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .summary import EMBEDDED, _fmt, _load, _mean, _write_csv, wilson

RECOVERY = ["correct", "wrong", "ambiguous", "none"]
FILE_FIELDS = [
    "setting", "master", "file", "status", "error", "format", "width", "height", "embedded",
    "signature_ok", "hard_binding", "detected", "payload_match", "bit_accuracy", "distance",
    "recovery", "recovered_signature_ok", "baseline_ok", "alignment", "psnr", "ssim", "lpips",
    "not_evaluated",
]  # fmt: skip
SHOW = 8  # problem files listed on the console per kind
NOT_MEASURED = (
    "false detection and false match (no unmarked images went through the pipeline), and "
    "disclosure consistency (whether the processing used AI is not declared)"
)
ALIGN = {
    "exact": "quality is measured on outputs of the master's size",
    "rescale": "quality is measured on outputs of the master's size or aspect ratio, the latter "
    "rescaled to the master's size and assumed uncropped",
}


def file_rows(cells: list[dict], baseline_ok: dict[str, bool]) -> list[dict]:
    rows = []
    for c in cells:
        st, mo, sb = c["state"] or {}, c["manifest"] or {}, c["softbinding"] or {}
        rc, q, o = c["recovery"] or {}, c["quality"] or {}, c["output"] or {}
        rows.append(
            {
                "setting": c["transform"]["id"],
                "master": c["asset"]["id"],
                "file": "; ".join((c["edit"] or {}).get("files", [])),
                "status": c["status"],
                "error": c["error"],
                "format": o.get("format"),
                "width": o.get("width"),
                "height": o.get("height"),
                "embedded": st.get("embedded"),
                "signature_ok": mo.get("signature_ok"),
                "hard_binding": mo.get("hard_binding"),
                "detected": sb.get("detected"),
                "payload_match": sb.get("payload_match"),
                "bit_accuracy": sb.get("bit_accuracy"),
                "distance": sb.get("distance"),
                "recovery": st.get("recovery"),
                "recovered_signature_ok": rc.get("recovered_signature_ok"),
                "baseline_ok": baseline_ok.get(c["asset"]["id"]),
                "alignment": q.get("alignment"),
                # PSNR of a pixel-identical file is infinite
                "psnr": ("inf" if q.get("psnr") is None else q["psnr"]) if q else None,
                "ssim": q.get("ssim"),
                "lpips": q.get("lpips"),
                "not_evaluated": "; ".join(
                    f"{k}: {v}" for k, v in (c["not_evaluated"] or {}).items()
                ),
            }
        )
    return rows


def by_setting(cells: list[dict], baseline_ok: dict[str, bool]) -> list[dict]:
    rows = []
    for s in dict.fromkeys(c["transform"]["id"] for c in cells):
        cs = [c for c in cells if c["transform"]["id"] == s]
        ok = [c for c in cs if c["status"] == "ok"]
        recoverable = [c for c in ok if baseline_ok[c["asset"]["id"]]]
        recovered = sum(c["state"]["recovery"] == "correct" for c in recoverable)
        emb = Counter(c["state"]["embedded"] for c in ok)
        rec = Counter(c["state"]["recovery"] for c in ok)
        qs = [c["quality"] for c in ok if c["quality"]]
        lo, hi = wilson(recovered, len(recoverable)) if recoverable else (None, None)
        rows.append(
            {
                "setting": s,
                "expected": len(cs),
                "evaluated": len(ok),
                "errors": sum(c["status"] == "error" for c in cs),
                "missing": sum(c["status"] == "missing" for c in cs),
                **{e: emb[e] for e in EMBEDDED},
                **{r: rec[r] for r in RECOVERY},
                "available": sum(c["state"]["provenance_available"] for c in ok),
                "baseline_failed": len(ok) - len(recoverable),
                "recoverable": len(recoverable),
                "recovered": recovered,
                "recovered_rate": recovered / len(recoverable) if recoverable else None,
                "recovered_ci_lo": lo,
                "recovered_ci_hi": hi,
                "quality_n": len(qs),
                "identical": sum(q["psnr"] is None for q in qs),
                "psnr_mean": _mean([q["psnr"] for q in qs]),
                "ssim_mean": _mean([q["ssim"] for q in qs]),
                "lpips_mean": _mean([q["lpips"] for q in qs]),
            }
        )
    return rows


def _recovered(r: dict) -> str:
    return f"{r['recovered']}/{r['recoverable']}" if r["recoverable"] else "n/a"


def _quality(r: dict, metric: str, nd: int = 3) -> str:
    if not r["quality_n"]:
        return "n/a"
    if metric == "psnr" and r["identical"] == r["quality_n"]:
        return "identical"
    mean = r[f"{metric}_mean"]
    return "n/a" if mean is None else _fmt(mean, nd)


def _problems(cells: list[dict], run: dict) -> dict[str, list[tuple[str, str]]]:
    """(where, why) per kind of problem, for the console and summary.md."""

    def where(c):
        files = (c["edit"] or {}).get("files") or []
        return files[0] if len(files) == 1 else f"{c['transform']['id']}/{c['asset']['id']}"

    failed = sorted(k for k, v in run["case"]["masters"].items() if not v["baseline_ok"])
    mapped = isinstance(run["config"]["resolved"]["matching"], dict)
    unmatched = "not in the map" if mapped else "no master of this name; use --map"
    # a setting with nothing at all is one item, so that single missing files stay in view
    n_masters = len(run["case"]["masters"])
    gone = Counter(c["transform"]["id"] for c in cells if c["status"] == "missing")
    empty = {s for s, n in gone.items() if n == n_masters}
    missing = [(f"{s}/", f"all {n_masters} masters missing") for s in sorted(empty)]
    missing += [(where(c), c["error"]) for c in cells
                if c["status"] == "missing" and c["transform"]["id"] not in empty]  # fmt: skip
    return {
        "missing": missing,
        "error": [(where(c), c["error"]) for c in cells if c["status"] == "error"],
        "unmatched": [(f, unmatched) for f in run["unmatched"]],
        "no quality": [
            (where(c), c["not_evaluated"]["quality"])
            for c in cells
            if c["status"] == "ok" and "quality" in (c["not_evaluated"] or {})
        ],
        "baseline failed": [(k, "not recoverable unprocessed") for k in failed],
    }


def _counts_line(k: dict) -> str:
    return (
        f"expected {k['expected']} ({k['settings']} setting{'s' * (k['settings'] != 1)} × "
        f"{k['masters']} masters) · "
        f"present {k['present']} · evaluated {k['evaluated']} · errors {k['errors']} · "
        f"missing {k['missing']} · unmatched {k['unmatched']} · ignored {k['ignored']}"
    )


def write(report: Path) -> str:
    """Write files.csv, summary.csv and summary.md; return the console summary."""
    run = json.loads((report / "run.json").read_text())
    cells = _load(report / "results.jsonl")
    baseline_ok = {k: v["baseline_ok"] for k, v in run["case"]["masters"].items()}
    rows = by_setting(cells, baseline_ok)
    _write_csv(report / "files.csv", file_rows(cells, baseline_ok), FILE_FIELDS)
    _write_csv(report / "summary.csv", rows, list(rows[0]))

    k, case, problems = run["counts"], run["case"], _problems(cells, run)
    resolved = run["config"]["resolved"]
    n_masters, alg = k["masters"], resolved["alg"]
    md = [
        f"# provsurvive evaluation `{run['run_id']}`",
        "",
        f"Case `{case['case_id']}` ({n_masters} masters, `{alg}`) · outputs "
        f"`{resolved['outputs']}`" + (f" · note: {run['note']}" if run["note"] else ""),
        "",
        f"**Counts:** {_counts_line(k)}",
        "",
        "Manifest states, recovery outcomes and *available* (intact, superseded or correct) "
        "count the evaluated files. *Recovered* counts correct recoveries among the evaluated "
        f"files whose master was recoverable unprocessed; a lookup searches the case's "
        f"{n_masters} masters. Quality counts the files with a measurement "
        f"({resolved['align']}: {ALIGN[resolved['align']]}).",
        "",
        "## By setting",
        "",
        "| setting | expected | evaluated | errors | missing | intact | superseded | stale | "
        "invalid | absent | correct | wrong | ambiguous | none | available | baseline failed | "
        "recovered (95% CI) | quality n | PSNR | SSIM | LPIPS |",
        "|---|" + "--:|" * 15 + "---|" + "--:|" * 4,
    ]
    for r in rows:
        ci = (
            f" ({_fmt(r['recovered_ci_lo'], 2)}–{_fmt(r['recovered_ci_hi'], 2)})"
            if r["recoverable"]
            else ""
        )
        md.append(
            f"| {r['setting']} | {r['expected']} | {r['evaluated']} | {r['errors']} | "
            f"{r['missing']} | "
            + " | ".join(str(r[x]) for x in [*EMBEDDED, *RECOVERY, "available", "baseline_failed"])
            + f" | {_recovered(r)}{ci} | {r['quality_n']} | {_quality(r, 'psnr', 1)} | "
            f"{_quality(r, 'ssim')} | {_quality(r, 'lpips')} |"
        )
    identical = sum(r["identical"] for r in rows)
    md += [
        "",
        "PSNR averages the files that are not pixel-identical to their master"
        + (f" ({identical} are)." if identical else "."),
        "",
        f"Not measured: {NOT_MEASURED}.",
        "",
        "## Files to look at",
        "",
    ]
    listed = [(kind, w, why) for kind, items in problems.items() for w, why in items]
    if listed:
        md += ["| file | problem | detail |", "|---|---|---|"]
        md += [f"| {w} | {kind} | {why} |" for kind, w, why in listed]
    else:
        md.append("None.")
    md += ["", "Every file's states: `files.csv`. Raw observations: `results.jsonl`."]
    (report / "summary.md").write_text("\n".join(md) + "\n")

    # console
    out = [f"case {case['case_id']} · {n_masters} masters · {alg}", _counts_line(k), ""]
    cols = ["evaluated", *EMBEDDED, *RECOVERY, "recovered", "PSNR"]
    table = [
        [r["evaluated"], *(r[x] for x in [*EMBEDDED, *RECOVERY]), _recovered(r),
         _quality(r, "psnr", 1)]
        for r in rows
    ]  # fmt: skip
    widths = [max(len(c), 5, *(len(str(t[i])) for t in table)) for i, c in enumerate(cols)]
    width = max(len("setting"), *(len(r["setting"]) for r in rows))
    names = ["setting", *(r["setting"] for r in rows)]
    for name, vals in zip(names, [cols, *table], strict=True):
        padded = (f"{v:>{w}}" for v, w in zip(vals, widths, strict=True))
        out.append(f"{name:<{width}}  " + "  ".join(padded))
    out.append(
        "recovered: correct among files whose master was recoverable unprocessed; "
        f"quality: {resolved['align']}"
    )
    for kind, items in problems.items():
        if not items:
            continue
        out.append("")
        out += [f"{kind}: {w}  ({why})" for w, why in items[:SHOW]]
        if len(items) > SHOW:
            out.append(f"{kind}: … and {len(items) - SHOW} more (summary.md)")
    return "\n".join(out)
