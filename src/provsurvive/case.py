"""Test your own pipeline: prepare signed, marked test masters from a folder of images, put
them through your pipeline, and evaluate what comes out (``provsurvive prepare | evaluate``).

A case directory holds everything evaluation needs:

    case/
      masters/<id>.png   the files to put through your pipeline (.jpg with --format jpeg)
      case.json          binding, payloads and values, digests, and each master's baseline
      manifests.sqlite   the resolver's registry: this case's masters
      ca-root.pem        the test CA's root certificate

Evaluation reads the case only. Outputs are matched to masters by file name without
extension, or by an explicit map.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import UnidentifiedImageError

from . import runinfo
from .bindings.registry import make_plugin
from .metrics import quality
from .resolver.store import ManifestRepository
from .runner import JPEG_QUALITY, Master, RunConfig, Runner, derive_seed, sha256
from .schema import AssetRef, BindingRef, CellResult, TransformRef
from .sign.c2pa_io import C2paSession
from .sign.test_ca import TestCA, TrustAnchor
from .sources import SourceImage, from_directory
from .transforms.base import decode_rgb

CASE_FILE = "case.json"
CASE_SCHEMA = "provsurvive.case.v1"
IMAGE_SUFFIXES = {
    ".jpg", ".jpeg", ".png", ".webp", ".avif", ".heic", ".heif", ".tif", ".tiff", ".gif",
    ".bmp", ".jxl",
}  # fmt: skip
MASTER_SUFFIX = {"png": "png", "jpeg": "jpg"}
REPO_ROOT = Path(__file__).resolve().parents[2]


class CaseError(Exception):
    """A problem with the user's files that stops prepare or evaluate."""


def sniff(data: bytes) -> str | None:
    """The container (jpeg, png, webp or avif) from the leading bytes; None for anything else."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] == b"ftyp" and b"avif" in data[8 : int.from_bytes(data[:4], "big")]:
        return "avif"
    return None


def _why(e: Exception) -> str:
    if isinstance(e, UnidentifiedImageError):
        return "not a readable image"
    if isinstance(e, OSError) and e.strerror:
        return e.strerror
    return f"{type(e).__name__}: {e}"


def _walk(root: Path):
    """(folder relative to ``root``, its subfolders, its files) for every folder under ``root``,
    hidden names left out and symlinked folders followed."""

    def fail(e: OSError) -> None:
        raise CaseError(f"cannot read {e.filename}: {e.strerror}")

    up = {str(root): {os.path.realpath(root)}}  # each folder's real path and its parents'
    for d, dirs, names in os.walk(root, followlinks=True, onerror=fail):
        seen, keep = up.pop(d), []
        for x in sorted(dirs):
            real = os.path.realpath(os.path.join(d, x))
            if not x.startswith(".") and real not in seen:  # a link to a parent is a loop
                up[os.path.join(d, x)] = seen | {real}
                keep.append(x)
        dirs[:] = keep
        yield Path(d).relative_to(root), dirs, sorted(x for x in names if not x.startswith("."))


# --- prepare ---------------------------------------------------------------------------


@dataclass
class _Input:
    src: SourceImage
    rel: str  # path relative to the images folder
    data: bytes


def _corpus_files(images: Path) -> dict[str, str] | None:
    """id -> file of a provsurvive corpus (a folder with index.json); None for any other folder."""
    try:
        index = json.loads((images / "index.json").read_text())
    except (OSError, ValueError):
        return None
    if isinstance(index, list) and all(
        isinstance(e, dict) and {"id", "file", "kind"} <= set(e) for e in index
    ):
        return {e["id"]: e["file"] for e in index}
    return None


def _inputs(images: Path, kind: str) -> tuple[list[_Input], list[dict[str, str]]]:
    """The images under ``images`` (hidden files skipped), with ids from their file names, and
    the files skipped. A provsurvive corpus (with index.json) keeps its ids and kinds."""
    files = _corpus_files(images)
    if files is not None:
        return [
            _Input(s, files[s.id], (images / files[s.id]).read_bytes())
            for s in from_directory(images)
        ], []
    found, skipped = [], []
    for folder, _, names in _walk(images):
        for name in names:
            p, rel = images / folder / name, (folder / name).as_posix()
            try:
                data = p.read_bytes()
            except OSError as e:
                skipped.append({"file": rel, "reason": f"cannot read: {_why(e)}"})
                continue
            if p.suffix.lower() not in IMAGE_SUFFIXES and sniff(data) is None:
                skipped.append({"file": rel, "reason": "not an image"})
                continue
            try:
                img = decode_rgb(data)
            except Exception as e:
                skipped.append({"file": rel, "reason": f"cannot decode: {_why(e)}"})
                continue
            found.append(_Input(SourceImage(p.stem, kind, kind, img), rel, data))  # type: ignore[arg-type]
    names = defaultdict(list)
    for i in found:
        names[i.src.id.lower()].append(i.rel)  # some file systems ignore case
    clash = {k: v for k, v in names.items() if len(v) > 1}
    if clash:
        lines = "; ".join(", ".join(v) for _, v in sorted(clash.items()))
        raise CaseError(f"inputs share a file name, which outputs are matched by: {lines}")
    return found, skipped


def prepare(
    images: Path,
    out: Path,
    plugin_spec: dict[str, Any],
    ca: TestCA,
    *,
    kind: str = "photo",
    fmt: str = "png",
    seed: int = 20261031,
    device: str = "cpu",
    lpips: bool = True,
) -> dict[str, Any]:
    """Mark, sign and register a master for every image under ``images``; write the case to
    ``out`` and return case.json. Each master is probed once as it is (its baseline)."""
    if out.exists() and any(out.iterdir()):
        raise CaseError(f"{out} exists and is not empty")
    if fmt not in MASTER_SUFFIX:
        raise CaseError(f"master format {fmt!r}: use png or jpeg")
    if kind not in ("photo", "generated"):
        raise CaseError(f"kind {kind!r}: use photo or generated")
    inputs, skipped = _inputs(images, kind)
    if not inputs:
        why = "; ".join(f"{k['file']} ({k['reason']})" for k in skipped[:8])
        raise CaseError(f"no images under {images}" + (f"; skipped: {why}" if why else ""))
    plugin = make_plugin(plugin_spec)
    started = dt.datetime.now(dt.timezone.utc)
    (out / "masters").mkdir(parents=True)
    (out / "ca-root.pem").write_bytes(ca.root_pem)
    # payloads depend on the inputs too, so cases with the same file names do not share them
    payload_seed = derive_seed(seed, sorted(sha256(i.data) for i in inputs))
    runner = Runner(
        RunConfig("prepare", [], seed=payload_seed, device=device, lpips=lpips),
        plugin,
        C2paSession(ca),
        out,
        repo=ManifestRepository(out / "manifests.sqlite"),
    )
    made = []
    for inp in inputs:
        m, rec = runner.make_master(inp.src, fmt=fmt, use_cache=False)
        name = f"masters/{inp.src.id}.{MASTER_SUFFIX[fmt]}"
        (out / name).write_bytes(m.data)
        made.append((inp, m, rec, name))

    masters = []
    for inp, m, rec, name in made:  # baselines once every master is registered
        obs = runner.probe(m, m.data, fmt, None, {}, truth_ai=None, timing={})
        st, sb = obs["state"], obs["softbinding"]
        masters.append(
            {
                "id": m.source.id,
                "kind": m.source.kind,
                "file": name,
                "input": inp.rel,
                "input_sha256": sha256(inp.data),
                "width": rec["width"],
                "height": rec["height"],
                "master_sha256": rec["master_sha256"],
                "manifest_label": m.label,
                "manifest_bytes": rec["manifest_bytes"],
                "payload": m.payload,
                "value": m.value,
                "mark_cost": rec["mark_cost"],
                "baseline": {
                    "embedded": st.embedded,
                    "recovery": st.recovery,
                    "detected": sb.detected,
                    "payload_match": sb.payload_match,
                    "distance": sb.distance,
                },
                "baseline_ok": st.embedded == "intact" and st.recovery == "correct",
            }
        )
    case = {
        "schema": CASE_SCHEMA,
        "case_id": sha256(json.dumps(sorted(m["master_sha256"] for m in masters)).encode())[:12],
        "created": started.isoformat(),
        "images": str(images.resolve()),
        "tool": runinfo.tool_info(REPO_ROOT),
        "environment": runinfo.environment(),
        "binding": {
            "spec": plugin_spec,
            "plugin": plugin.name,
            "alg": plugin.alg,
            "kind": plugin.kind,
            "config": plugin.config(),
        },
        "seed": seed,
        "master_format": fmt,
        "jpeg_quality": JPEG_QUALITY if fmt == "jpeg" else None,
        "ca_root_fingerprint_sha256": ca.root_fingerprint,
        "masters": masters,
        "skipped": skipped,
    }
    (out / CASE_FILE).write_text(json.dumps(case, indent=1) + "\n")
    return case


# --- evaluate --------------------------------------------------------------------------


def load_case(case_dir: Path) -> tuple[dict[str, Any], list[Master]]:
    """case.json and the masters, each checked against its digest."""
    path = case_dir / CASE_FILE
    if not path.exists():
        raise CaseError(f"{case_dir} has no {CASE_FILE}; create a case with `provsurvive prepare`")
    case = json.loads(path.read_text())
    masters = []
    for e in case["masters"]:
        f = case_dir / e["file"]
        data = f.read_bytes() if f.exists() else b""
        if sha256(data) != e["master_sha256"]:
            raise CaseError(f"{f} is missing or changed since prepare (processed in place?)")
        img = decode_rgb(data)
        src = SourceImage(e["id"], e["kind"], e["kind"], img)
        label, fmt = e["manifest_label"], case["master_format"]
        masters.append(Master(src, data, img, label, e["payload"], False, e["value"], fmt))
    return case, masters


def output_files(outputs: Path) -> tuple[list[str], list[str], list[str]]:
    """Files under ``outputs`` ('/'-separated, relative) that may be images, the other files, and
    the innermost folders without an image (a stage that wrote nothing). Hidden ones are left
    out."""
    files, ignored, empty = [], [], []
    for folder, subfolders, names in _walk(outputs):
        images = [n for n in names if _maybe_image(outputs / folder / n)]
        files += [(folder / n).as_posix() for n in images]
        ignored += [(folder / n).as_posix() for n in names if n not in images]
        if not subfolders and not images and folder != Path("."):
            empty.append(folder.as_posix())
    return files, ignored, empty


def _maybe_image(p: Path) -> bool:
    if p.suffix.lower() in IMAGE_SUFFIXES:
        return True
    try:
        with open(p, "rb") as f:
            return sniff(f.read(64)) is not None
    except OSError:
        return False


def load_map(path: Path, master_ids: set[str]) -> dict[str, tuple[str, str]]:
    """An ``output,master[,setting]`` CSV, output paths relative to the outputs folder, as
    output -> (setting, master). The setting defaults to the output's folder."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
    except UnicodeDecodeError:
        raise CaseError(f"{path} is not UTF-8; save it as CSV UTF-8") from None
    if not fields or {"output", "master"} - set(fields):
        raise CaseError(f"{path}: the first row must be the header 'output,master'")
    mapping: dict[str, tuple[str, str]] = {}
    taken: dict[tuple[str, str], str] = {}
    lower = {m.lower(): m for m in master_ids}
    for i, r in enumerate(rows, 2):
        raw = (r["output"] or "").strip().replace("\\", "/")
        if not raw or raw.startswith("/"):
            raise CaseError(f"{path} line {i}: the output must be a path relative to OUTPUTS")
        out = PurePosixPath(raw).as_posix()
        mid = (r["master"] or "").strip()
        mid = lower.get(mid.lower(), mid)
        if mid not in master_ids:
            raise CaseError(
                f"{path} line {i}: the case has no master {mid!r} (a master's file name without "
                "extension)"
            )
        setting = (r.get("setting") or "").strip() or str(PurePosixPath(out).parent)
        if mapping.get(out, (setting, mid)) != (setting, mid):
            raise CaseError(f"{path} line {i}: {out!r} is mapped twice")
        if taken.setdefault((setting, mid), out) != out:
            raise CaseError(
                f"{path} line {i}: master {mid!r} already has an output in setting {setting!r}"
            )
        mapping[out] = (setting, mid)
    return mapping


def match_outputs(
    files: list[str], master_ids: set[str], mapping: dict[str, tuple[str, str]] | None = None
) -> tuple[dict[tuple[str, str], list[str]], list[str]]:
    """Assign output files to (setting, master); return the assignment and the unmatched files.

    The setting is the file's folder ("." at the top) and the master the one whose id is the file
    name without extension, or, with ``mapping``, the setting and master the mapping names (then
    the mapping is the only rule). Several files for one (setting, master) stay together, for the
    caller to report as duplicates."""
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    unmatched = []
    lower = {m.lower(): m for m in master_ids}  # names match whatever their case
    for f in sorted(files):
        p = PurePosixPath(f)
        stem = PurePosixPath(p.stem)  # beach.png.webp is beach's too
        if p.stem.lower() not in lower and stem.suffix.lower() in IMAGE_SUFFIXES:
            stem = PurePosixPath(stem.stem)
        by_name = (str(p.parent), lower.get(str(stem).lower()))
        setting, mid = mapping.get(f, ("", None)) if mapping is not None else by_name
        if mid in master_ids:
            groups[(setting, mid)].append(f)
        else:
            unmatched.append(f)
    return dict(groups), unmatched


def _conflict(m: Master, obs: dict[str, Any], owner: dict[str, str], kind: str) -> str | None:
    """Evidence that the file was made from another master of the case: it carries that
    master's valid manifest, or decodes to its watermark payload (random payloads do not
    collide). A fingerprint's nearest neighbour is no such evidence."""
    mobs, robs = obs["manifest"], obs["recovery"]
    label = mobs.active_label
    if mobs.signature_ok and label in owner and label != m.label:
        return f"mapping conflict: the file carries the manifest of master {owner[label]!r}"
    if mobs.validation_state == "Trusted" and label not in owner:
        # only provsurvive's test CA is trusted here
        return "wrong case: the file carries the manifest of another provsurvive case"
    if kind == "watermark" and robs.outcome == "wrong":
        top = robs.candidates[0].label
        return f"mapping conflict: the watermark decodes to master {owner.get(top, top)!r}"
    return None


def _cell(
    runner: Runner,
    m: Master,
    setting: str,
    files: list[str],
    outputs: Path,
    binding: BindingRef,
    owner: dict[str, str],
    align: str,
    mapped: str | None = None,
) -> CellResult:
    base = dict(
        run_id=runner.run_id,
        cell_id=sha256(json.dumps([setting, m.source.id]).encode())[:16],
        control=False,
        asset=AssetRef(
            id=m.source.id,
            subset=m.source.subset,
            master_format=m.fmt,
            master_sha256=sha256(m.data),
        ),
        transform=TransformRef(
            id=setting, family="external", params={}, seed=0, impl_version="external", gpu=False
        ),
        binding=binding,
        edit={"files": files},
    )
    timing: dict[str, float] = {}

    def failed(status: str, error: str) -> CellResult:
        none = dict(output=None, manifest=None, softbinding=None, recovery=None, state=None)
        return CellResult(
            **base, **none, quality=None, timing_ms=timing, status=status, error=error
        )

    if not files:
        map_note = f"; map: {mapped}" if mapped else ""
        return failed("missing", f"no output for master {m.source.id!r}{map_note}")
    if len(files) > 1:
        return failed(
            "error", f"duplicate: {len(files)} files for master {m.source.id!r}: {', '.join(files)}"
        )
    try:
        data = (outputs / files[0]).read_bytes()
        fmt = sniff(data)
        if fmt is None:
            return failed("error", "not a JPEG, PNG, WebP or AVIF file" if data else "empty file")
        size = decode_rgb(data).size
        alignment, why = quality.alignment_for(m.image.size, size, align)
        obs = runner.probe(m, data, fmt, alignment, {}, truth_ai=None, timing=timing)
    except Exception as e:
        return failed("error", f"cannot read: {_why(e)}")
    conflict = _conflict(m, obs, owner, runner.plugin.kind)
    return CellResult(
        **base,
        **obs,
        timing_ms=timing,
        status="error" if conflict else "ok",
        error=conflict,
        not_evaluated={"quality": why} if alignment is None else None,
    )


def evaluate(
    case_dir: Path,
    outputs: Path,
    out: Path,
    *,
    mapping: Path | None = None,
    note: str | None = None,
    align: str = "exact",
    device: str = "cpu",
    lpips: bool = True,
) -> Path:
    """Evaluate every (setting, master) of the files under ``outputs``; write the report to
    ``out``: run.json and results.jsonl (one CellResult per setting and master, missing ones
    included)."""
    if out.exists() and any(out.iterdir()):
        raise CaseError(f"{out} exists and is not empty")
    if align not in ("exact", "rescale"):
        raise CaseError(f"alignment {align!r}: use exact or rescale")
    started = dt.datetime.now(dt.timezone.utc)
    case, masters = load_case(case_dir)
    ids = {m.source.id for m in masters}
    files, ignored, empty = output_files(outputs)
    # with a map, its settings; otherwise every folder with images and every innermost folder
    # without one (a stage that wrote nothing). A master without a file there is missing.
    if mapping is None:
        table = None
        rule: Any = "file name without extension"
        settings = sorted({str(PurePosixPath(f).parent) for f in files} | set(empty))
    else:
        table = load_map(mapping, ids)
        # a mapped file is an output whatever its name; one that is no image fails as such
        files = sorted(set(files) | {f for f in table if (outputs / f).is_file()})
        ignored = [f for f in ignored if f not in table]
        rule = {"map": str(mapping.resolve()), "sha256": sha256(mapping.read_bytes())}
        settings = sorted({s for s, _ in table.values()})
    groups, unmatched = match_outputs(files, ids, table)
    if not files:
        raise CaseError(f"no image under {outputs}")
    if not groups and table is not None:
        raise CaseError(f"no output in {mapping} is under {outputs} (paths are relative to it)")
    if not groups:
        raise CaseError(f"no image under {outputs} is named after a master; rename or use --map")

    plugin = make_plugin({**case["binding"]["spec"], "device": device})
    config = plugin.config()
    runner = Runner(
        RunConfig("evaluate", [], device=device, lpips=lpips),
        plugin,
        C2paSession(TrustAnchor((case_dir / "ca-root.pem").read_bytes())),
        out,
        repo=ManifestRepository(case_dir / "manifests.sqlite", readonly=True),
    )
    resolved = {
        "case_id": case["case_id"],
        "outputs": str(outputs.resolve()),
        "matching": rule,
        "settings": settings,
        "alg": plugin.alg,
        "binding": config,
        "align": align,
        "lpips": lpips,
    }
    cfg_sha = sha256(json.dumps(resolved, sort_keys=True).encode())
    runner.run_id = started.strftime("%Y-%m-%dT%H-%M-%SZ") + "-" + cfg_sha[:6]
    owner = {e["manifest_label"]: e["id"] for e in case["masters"]}
    binding = BindingRef(plugin=plugin.name, alg=plugin.alg, config=config)
    expected = {v: k for k, v in (table or {}).items()}
    cells = [
        _cell(
            runner,
            m,
            s,
            groups.get((s, m.source.id), []),
            outputs,
            binding,
            owner,
            align,
            expected.get((s, m.source.id)),
        )
        for s in settings
        for m in masters
    ]

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for c in cells:
            f.write(c.model_dump_json() + "\n")
    status = Counter(c.status for c in cells)
    run_json = {
        "schema_version": "1.1",
        "kind": "evaluate",
        "run_id": runner.run_id,
        "tool": runinfo.tool_info(REPO_ROOT),
        "environment": runinfo.environment(),
        "config": {"suite": "evaluate", "config_sha256": cfg_sha, "resolved": resolved},
        "note": note,
        "case": {
            "case_id": case["case_id"],
            "path": str(case_dir.resolve()),
            "master_format": case["master_format"],
            "masters": {
                e["id"]: {
                    "input": e["input"],
                    "input_sha256": e["input_sha256"],
                    "baseline_ok": e["baseline_ok"],
                }
                for e in case["masters"]
            },
        },
        "counts": {
            "masters": len(masters),
            "settings": len(settings),
            "expected": len(cells),
            "present": len(cells) - status["missing"],
            "evaluated": status["ok"],
            "errors": status["error"],
            "missing": status["missing"],
            "unmatched": len(unmatched),
            "ignored": len(ignored),
        },
        "unmatched": unmatched,
        "ignored": ignored,
        "timing": {
            "started": started.isoformat(),
            "finished": dt.datetime.now(dt.timezone.utc).isoformat(),
        },
    }
    (out / "run.json").write_text(json.dumps(run_json, indent=2) + "\n")
    return out
