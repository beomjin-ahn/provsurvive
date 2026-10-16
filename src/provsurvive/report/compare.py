"""Compare two reports of the same masters (``provsurvive compare``): typically two
evaluations of one case, before and after a change to the pipeline; two runs work too.

Rows are joined on (setting or edit, master). What differs between the reports themselves
(inputs, binding, evaluation config, evaluator versions, notes) is stated first. A master is
compared only if both reports hash its input and the hashes agree. A row regressed if it lost
something it had (a readable output, an intact or superseded manifest, the found original
manifest) or a lookup started to return another master's manifest; it improved in the opposite
cases, and changed if anything else in its states, or its error, moved. Recovery counts only
under the same soft binding. Per setting, among the files read in both reports: how many were
measured and pixel-identical, the mean PSNR of those measured and not identical in both, and
(between evaluations) how many differ with the same states.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .summary import _load, _write_csv

PACKAGES = ["provsurvive", "c2pa-python", "c2pa_sdk", "trustmark", "iscc-core", "Pillow"]
DIFF_FIELDS = [
    "setting", "master", "change", "transitions", "a_status", "b_status", "a_embedded",
    "b_embedded", "a_recovery", "b_recovery", "a_psnr", "b_psnr", "differ", "a_error", "b_error",
    "a_file", "b_file",
]  # fmt: skip
VIEW = ("status", "error", "embedded", "recovery", "file")
ORDER = ["regressed", "improved", "changed", "same", "only in A", "only in B"]
SHOW = 8


def _report(d: Path) -> tuple[dict, dict[tuple, dict]]:
    run = json.loads((d / "run.json").read_text())
    cells = {}
    for name in ("results.jsonl", "controls.jsonl"):
        for c in _load(d / name):
            setting = c["transform"]["id"] + (" (control)" if c["control"] else "")
            cells[(setting, c["asset"]["id"])] = c
    return run, cells


def _inputs(d: Path, run: dict) -> dict[str, str | None]:
    if "case" in run:
        return {k: v["input_sha256"] for k, v in run["case"]["masters"].items()}
    return {a["id"]: a.get("source_sha256") for a in _load(d / "assets.jsonl")}


def _identity(run: dict) -> str:
    return run["case"]["case_id"] if "case" in run else run["config"]["config_sha256"][:12]


def _view(c: dict | None) -> dict[str, Any]:
    if c is None:
        return dict.fromkeys([*VIEW, "sha", "detected"])
    st = c.get("state") or {}
    return {
        "status": c["status"],
        "error": c["error"],
        "sha": (c.get("output") or {}).get("sha256"),
        "embedded": st.get("embedded"),
        "recovery": st.get("recovery"),
        "detected": (c.get("softbinding") or {}).get("detected"),
        "file": "; ".join((c.get("edit") or {}).get("files", [])),
    }


def _psnr(c: dict | None) -> float | None:
    """None if not measured; infinite for a pixel-identical file."""
    q = (c or {}).get("quality")
    return None if not q else math.inf if q["psnr"] is None else q["psnr"]


def classify(a: dict, b: dict, same_binding: bool) -> tuple[str, list[str]]:
    """(regressed | improved | changed | same, transitions) for one row."""
    kept = {"intact", "superseded"}
    lost = gained = False
    if a["status"] != b["status"]:
        fields: tuple[str, ...] = ("status",)
        lost, gained = a["status"] == "ok", b["status"] == "ok"
    elif a["status"] != "ok":  # a file that is missing or failed has no states to compare
        fields = ("error",)
    else:
        fields = ("embedded", "recovery", "detected")
        lost = a["embedded"] in kept and b["embedded"] not in kept
        gained = b["embedded"] in kept and a["embedded"] not in kept
        if same_binding:
            lost |= a["recovery"] == "correct" and b["recovery"] != "correct"
            gained |= b["recovery"] == "correct" and a["recovery"] != "correct"
            lost |= b["recovery"] == "wrong" != a["recovery"]
            gained |= a["recovery"] == "wrong" != b["recovery"]
    moves = [
        "error changed" if k == "error" else f"{k} {a[k]}→{b[k]}"
        for k in fields
        if a[k] != b[k] and (same_binding or k in ("status", "error", "embedded"))
    ]
    if lost and not gained:
        return "regressed", moves
    if gained and not lost:
        return "improved", moves
    return ("changed" if moves else "same"), moves


def compare(a_dir: Path, b_dir: Path) -> dict[str, Any]:
    ra, ca = _report(a_dir)
    rb, cb = _report(b_dir)
    conf_a, conf_b = ra["config"]["resolved"], rb["config"]["resolved"]
    ia, ib = _identity(ra), _identity(rb)
    kind = "case" if "case" in ra and "case" in rb else "configuration"
    header = [f"same {kind} {ia}" if ia == ib else f"different {kind}: {ia} vs {ib}"]

    # inputs are checked whatever the identifiers say: a run's configuration names its
    # sources but does not hash them
    inputs_a, inputs_b = _inputs(a_dir, ra), _inputs(b_dir, rb)
    skip = {}
    for m in {k[1] for k in set(ca) & set(cb)}:
        ha, hb = inputs_a.get(m), inputs_b.get(m)
        if ha is None or hb is None:
            skip[m] = "input unknown"
        elif ha != hb:
            skip[m] = "input differs"
    for why in ("input differs", "input unknown"):
        ms = sorted(m for m, w in skip.items() if w == why)
        if ms:
            header.append(f"{why}, not compared: {', '.join(ms)}")

    same_binding = conf_a["alg"] == conf_b["alg"]
    if not same_binding:
        header.append(
            f"soft binding: {conf_a['alg']} vs {conf_b['alg']}; recovery and detection are not "
            "compared"
        )
    header += _moved("binding config", conf_a["binding"], conf_b["binding"])
    rest = set(conf_a) | set(conf_b)
    rest -= {"alg", "binding", "outputs", "settings", "sources", "transforms", "case_id"}
    header += _moved("evaluation config", {k: _config(conf_a, k) for k in rest},
                     {k: _config(conf_b, k) for k in rest})  # fmt: skip
    pa, pb = ra["environment"]["packages"], rb["environment"]["packages"]
    header += _moved(
        "evaluator", {p: pa.get(p) for p in PACKAGES}, {p: pb.get(p) for p in PACKAGES}
    )
    if ra.get("note") or rb.get("note"):
        header.append(f"note: {ra.get('note') or '-'} → {rb.get('note') or '-'}")

    # files evaluated against one case are the pipeline's; a run's are re-signed every time, and
    # another case has other masters, so theirs always differ
    hashed = "case" in ra and "case" in rb and ia == ib
    rows = []
    for key in sorted(set(ca) | set(cb)):
        setting, master = key
        a, b = ca.get(key), cb.get(key)
        va, vb = _view(a), _view(b)
        if a is None or b is None:
            change, moves = ("only in B" if a is None else "only in A"), []
        elif master in skip:
            change, moves = f"not compared ({skip[master]})", []
        else:
            change, moves = classify(va, vb, same_binding)
        rows.append(
            {
                "setting": setting,
                "master": master,
                "change": change,
                "transitions": "; ".join(moves),
                **{f"a_{k}": va[k] for k in VIEW},
                **{f"b_{k}": vb[k] for k in VIEW},
                "a_psnr": _psnr(a),
                "b_psnr": _psnr(b),
                "differ": hashed and None not in (va["sha"], vb["sha"]) and va["sha"] != vb["sha"],
                "_moves": moves,
            }
        )
    return {"a": str(a_dir), "b": str(b_dir), "header": header, "rows": rows}


def _config(conf: dict, k: str) -> Any:
    """A resolved setting as compared: a map by its digest, wherever it lies."""
    v = conf.get(k)
    return f"map {v['sha256'][:12]}" if k == "matching" and isinstance(v, dict) else v


def _moved(what: str, a: dict, b: dict) -> list[str]:
    moved = [
        f"{k} {a.get(k)} → {b.get(k)}" for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)
    ]
    return [f"{what}: " + ", ".join(moved)] if moved else []


def _quality(rows: list[dict]) -> list[str]:
    """Quality and file movement among the files read in both reports."""
    both = [(r["a_psnr"], r["b_psnr"]) for r in rows
            if r["change"] in ORDER[:4] and r["a_status"] == r["b_status"] == "ok"]  # fmt: skip
    out = []
    for what, test in (("quality measured", lambda v: v is not None),
                       ("pixel-identical", lambda v: v == math.inf)):  # fmt: skip
        na, nb = sum(test(a) for a, _ in both), sum(test(b) for _, b in both)
        if na != nb:
            out.append(f"{what}: {na} → {nb} file{'s' * (nb != 1)}")
    differ = sum(r["differ"] for r in rows if r["change"] == "same")
    if differ:
        out.append(f"different files, same states: {differ}")
    pairs = [(a, b) for a, b in both if a not in (None, math.inf) and b not in (None, math.inf)]
    if pairs:
        pa, pb = (f"{sum(p[i] for p in pairs) / len(pairs):.1f}" for i in (0, 1))
        if pa != pb:
            out.append(f"PSNR {pa} → {pb} dB ({len(pairs)} file{'s' * (len(pairs) != 1)})")
    return out


def render(result: dict[str, Any]) -> str:
    out = [f"A: {result['a']}", f"B: {result['b']}", *result["header"], ""]
    per = defaultdict(list)
    for r in result["rows"]:
        per[r["setting"]].append(r)
    width = max([len(s) for s in per] + [7])
    for setting, rows in per.items():
        n = Counter(r["change"] for r in rows)
        quality = [f"{'':<{width}}    {q}" for q in _quality(rows)]
        if set(n) <= {"same"}:
            states = "states unchanged" if quality else "no change"
            out += [f"{setting:<{width}}  {states} ({n['same']} same)", *quality]
            continue
        if len(n) == 1 and set(n) <= {"only in A", "only in B"}:
            which = "A" if "only in A" in n else "B"
            out.append(f"{setting:<{width}}  only in {which} ({len(rows)} rows)")
            continue
        out.append(f"{setting:<{width}}  " + " · ".join(f"{k} {v}" for k, v in sorted(
            n.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else len(ORDER)
        )))  # fmt: skip
        moves = defaultdict(list)
        for r in rows:
            for m in r["_moves"]:
                moves[m].append(r["master"])
        for m, masters in sorted(moves.items(), key=lambda x: -len(x[1])):
            more = f" … +{len(masters) - SHOW}" if len(masters) > SHOW else ""
            out.append(f"{'':<{width}}    {m}: {len(masters)} ({', '.join(masters[:SHOW])}{more})")
        out += quality
    return "\n".join(out)


def write(result: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    moved = [r for r in result["rows"]
             if r["change"] != "same" or r["a_psnr"] != r["b_psnr"] or r["differ"]]  # fmt: skip
    _write_csv(out / "diff.csv", moved, DIFF_FIELDS)
    (out / "diff.md").write_text("```\n" + render(result) + "\n```\n")
