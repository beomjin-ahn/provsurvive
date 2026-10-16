"""Command-line interface of provsurvive."""

from __future__ import annotations

import importlib.util
import json
import os
from importlib import resources
from pathlib import Path

import typer
import yaml

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Does C2PA provenance survive your image pipeline? prepare, evaluate and compare test "
    "your own pipeline; run, precompute and corpus are the benchmark suites.",
)
corpus_app = typer.Typer(no_args_is_help=True, help="Build or fetch the corpus.")
app.add_typer(corpus_app, name="corpus")


def home() -> Path:
    return Path(os.environ.get("PROVSURVIVE_HOME", Path.home() / ".provsurvive"))


def load_suite(name_or_path: str) -> tuple[str, list, list[str]]:
    from .transforms import base

    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        spec = yaml.safe_load(path.read_text())
    else:
        text = resources.files("provsurvive.suites").joinpath(f"{name_or_path}.yaml").read_text()
        spec = yaml.safe_load(text)
    ids = list(spec["transforms"])
    if spec.get("include"):
        ids = load_suite(spec["include"])[2] + ids
    return spec["name"], [base.get(t) for t in ids], ids


def _plugin_spec(binding: str, variant: str, device: str, rotation_search: bool = False) -> dict:
    if binding not in ("trustmark", "iscc", "pixelseal"):
        _fail(f"binding {binding!r}: use trustmark, iscc or pixelseal")
    if binding == "iscc" and importlib.util.find_spec("iscc_core") is None:
        _fail('the ISCC binding needs its extra: pip install -e ".[iscc]"')
    if binding != "trustmark":
        return {"name": binding, "device": device}
    if variant not in ("Q", "C", "P"):
        _fail(f"TrustMark variant {variant!r}: use Q, C or P")
    return {
        "name": "trustmark",
        "variant": variant,
        "device": device,
        "rotation_search": rotation_search,
    }


def _folder(help: str) -> Path:
    return typer.Argument(..., exists=True, file_okay=False, help=help)


def _fail(message: str) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def _sources(spec: str, limit: int | None):
    from . import sources

    out = []
    for part in spec.split(","):
        if part.startswith("synthetic"):
            n = int(part.split(":")[1]) if ":" in part else 8
            out += sources.synthetic(n)
        else:
            chunk = sources.from_directory(Path(part))
            out += chunk[:limit] if limit else chunk
    return out


@app.command()
def run(
    suite: str = typer.Option("quick", help="Suite name (bundled) or path to a suite YAML."),
    source: str = typer.Option(
        "synthetic:8", help="'synthetic[:N]' or a directory with index.json."
    ),
    out: Path = typer.Option(Path("runs"), help="Directory for run outputs."),
    device: str = typer.Option("cpu", help="Torch device for TrustMark and LPIPS."),
    lpips: bool = typer.Option(True, help="Compute LPIPS (slower on CPU)."),
    controls: bool = typer.Option(True, help="Also run unmarked control images."),
    thumbnail: bool = typer.Option(True, help="Let the C2PA SDK embed a thumbnail."),
    seed: int = typer.Option(20261031),
    limit: int = typer.Option(None, help="Use only the first N sources of each directory."),
    workers: int = typer.Option(1, help="Parallel worker processes for the CPU stage."),
    binding: str = typer.Option(
        "trustmark", help="Soft binding plugin: trustmark | pixelseal | iscc."
    ),
    variant: str = typer.Option("Q", help="TrustMark variant (registered: Q, C, P)."),
    rotation_search: bool = typer.Option(False, help="Decode with TrustMark ROTATION=True."),
    controls_gpu: bool = typer.Option(False, help="Also run generative transforms on controls."),
) -> None:
    """Mark, sign, transform and probe every source; write runs/<run_id>/."""
    from .bindings.registry import make_plugin
    from .report import benchinfo, summary
    from .runner import RunConfig, Runner
    from .sign.c2pa_io import C2paSession
    from .sign.test_ca import TestCA

    suite_name, transforms, _ = load_suite(suite)
    ca = TestCA.load_or_create(home() / "test-ca")
    cfg = RunConfig(
        suite=suite_name,
        transforms=transforms,
        seed=seed,
        device=device,
        lpips=lpips,
        thumbnail=thumbnail,
        controls=controls,
        controls_gpu=controls_gpu,
        workers=workers,
    )
    plugin_spec = _plugin_spec(binding, variant, device, rotation_search)
    runner = Runner(
        cfg,
        make_plugin(plugin_spec),
        C2paSession(ca, thumbnail=thumbnail),
        out,
        repo_root=Path(__file__).resolve().parents[2],
        plugin_spec=plugin_spec,
    )
    srcs = _sources(source, limit)
    typer.echo(f"{len(srcs)} sources × {len(transforms)} transforms, device={device}")
    run_dir = runner.run(srcs)
    summary.write(run_dir)
    benchinfo.write(run_dir)
    typer.echo(f"wrote {run_dir}")


@corpus_app.command("generate")
def corpus_generate(
    out: Path = typer.Option(..., help="Output directory for images and index.json."),
    prompts: Path = typer.Option(None, help="Prompt list; default: the packaged prompts.yaml."),
    only: str = typer.Option(None, help="Only this generator (zimage-turbo | flux2-klein-4b)."),
    limit: int = typer.Option(None, help="Only the first N prompts (smoke test)."),
) -> None:
    """Generate the T2I images (GPU; needs the [gen] extra). Resumable."""
    from .corpus.generate import PROMPTS_FILE, generate

    typer.echo(generate(prompts or PROMPTS_FILE, out, only=only, limit=limit))


@corpus_app.command("fetch-photos")
def corpus_fetch_photos(
    out: Path = typer.Option(..., help="Output directory for photos and index.json."),
    n: int = typer.Option(100),
    seed: int = typer.Option(20261031),
) -> None:
    """Download CC0 photographs from Wikimedia Commons (resumable, license re-checked)."""
    from .corpus.fetch_commons import fetch

    typer.echo(fetch(out, n=n, seed=seed))


@app.command()
def precompute(
    suite: str = typer.Option("full"),
    source: str = typer.Option(..., help="Directories with index.json, comma-separated."),
    seed: int = typer.Option(20261031),
    limit: int = typer.Option(None),
    shard: str = typer.Option("0/1", help="i/n: only sources with index %% n == i."),
    binding: str = typer.Option("trustmark", help="trustmark | pixelseal (masters to edit)."),
    variant: str = typer.Option("Q", help="TrustMark variant."),
    unmarked: bool = typer.Option(False, help="Edit the unmarked sources (controls, ISCC)."),
) -> None:
    """GPU stage: run the suite's generative transforms on every master and cache the outputs.

    Masters are re-created exactly as `run` does (same payload seeds, CPU TrustMark), so a later
    `run` finds every generative output in the cache (PROVSURVIVE_CACHE) and needs no GPU. Each
    invocation writes a record to <cache>/precompute/: code commit, packages, GPUs, models and
    revisions, suite, binding, sources, and the cache key of every output.
    """
    import datetime as dt
    import time

    from . import runinfo
    from .bindings.registry import make_plugin
    from .runner import derive_seed, marked_master
    from .transforms.base import encode
    from .transforms.generative import cache_dir, model_info

    started = dt.datetime.now(dt.timezone.utc)
    record = {
        "tool": runinfo.tool_info(Path(__file__).resolve().parents[2]),
        "environment": runinfo.environment(),
        "generative": model_info(),
        "args": {
            "suite": suite,
            "source": source,
            "seed": seed,
            "limit": limit,
            "shard": shard,
            "binding": binding,
            "variant": variant,
            "unmarked": unmarked,
        },
        "started": started.isoformat(),
        "outputs": [],
    }

    _, transforms, _ = load_suite(suite)
    gpu_ts = [t for t in transforms if t.gpu]
    if binding == "trustmark":
        plugin = make_plugin({"name": "trustmark", "variant": variant, "device": "cpu"})
    else:
        plugin = make_plugin({"name": binding, "device": "cpu"})
    i, n = (int(x) for x in shard.split("/"))
    srcs = [s for k, s in enumerate(_sources(source, limit)) if k % n == i]
    typer.echo(f"{len(srcs)} sources × {len(gpu_ts)} generative transforms")
    for src in srcs:
        if unmarked:  # exactly what the runner's control cells feed to each edit
            master = encode(src.image, "png")
        else:
            payload = plugin.new_payload(derive_seed(seed, src.id, "payload"))
            master = encode(marked_master(plugin, src.image, payload), "png")
        ctx = {"kind": src.kind, "prompt": src.meta.get("prompt")}
        for t in gpu_ts:
            t0 = time.time()
            out = t(master, derive_seed(seed, src.id, *t.seed_key()), ctx)
            record["outputs"].append([src.id, t.id, out.geometry.get("cache_key")])
            typer.echo(f"{src.id} {t.id} {time.time() - t0:.1f}s")
    record["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
    log = cache_dir().parent / "precompute" / f"{started.strftime('%Y-%m-%dT%H-%M-%SZ')}.json"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(json.dumps(record, indent=1) + "\n")
    typer.echo(f"wrote {log}")


@app.command()
def prepare(
    images: Path = _folder("Folder of your images (or a corpus with index.json)."),
    out: Path = typer.Option(..., help="Case directory to create."),
    binding: str = typer.Option("trustmark", help="Soft binding: trustmark | iscc | pixelseal."),
    variant: str = typer.Option("Q", help="TrustMark variant (registered: Q, C, P)."),
    kind: str = typer.Option(
        "photo",
        help="Test claim signed into the masters: photo (digitalCapture) | generated "
        "(trainedAlgorithmicMedia).",
    ),
    fmt: str = typer.Option("png", "--format", help="Master format: png | jpeg (quality 95)."),
    seed: int = typer.Option(20261031, help="Seeds the watermark payloads."),
    device: str = typer.Option("cpu"),
    lpips: bool = typer.Option(True, help="Compute LPIPS for the mark's cost."),
) -> None:
    """Make signed, marked TEST masters from your images, to put through your pipeline."""
    from .case import CaseError
    from .case import prepare as make_case
    from .sign.test_ca import TestCA

    try:
        case = make_case(
            images,
            out,
            _plugin_spec(binding, variant, device),
            TestCA.load_or_create(home() / "test-ca"),
            kind=kind,
            fmt=fmt,
            seed=seed,
            device=device,
            lpips=lpips,
        )
    except CaseError as e:
        _fail(str(e))
    ms = case["masters"]
    ok = sum(m["baseline_ok"] for m in ms)
    typer.echo(f"{len(ms)} masters in {out / 'masters'} ({case['binding']['alg']})")
    psnr = [m["mark_cost"]["psnr"] for m in ms if m["mark_cost"]["psnr"] is not None]
    if psnr:
        typer.echo(f"mark cost (master vs your image): PSNR {sum(psnr) / len(psnr):.1f} dB")
    typer.echo(f"baseline (intact and recovered as they are): {ok}/{len(ms)}")
    for m in ms:
        if not m["baseline_ok"]:
            bl = m["baseline"]
            typer.echo(f"  {m['id']}: {bl['embedded']}, recovery {bl['recovery']}")
    for sk in case["skipped"]:
        typer.echo(f"skipped {sk['file']}: {sk['reason']}")


@app.command()
def evaluate(
    case: Path = _folder("Case directory made by `prepare`."),
    outputs: Path = _folder("Your pipeline's outputs: one folder per setting."),
    out: Path = typer.Option(..., help="Report directory to create."),
    map_csv: Path = typer.Option(
        None,
        "--map",
        exists=True,
        dir_okay=False,
        help="CSV 'output,master[,setting]': output paths relative to OUTPUTS, masters by file "
        "name without extension; then the only rule.",
    ),
    note: str = typer.Option(None, help="What this pipeline run was, e.g. 'libvips 8.15, q80'."),
    align: str = typer.Option(
        "exact",
        help="Quality on outputs of the master's size (exact), or also on outputs of its aspect "
        "ratio, rescaled and assumed uncropped (rescale).",
    ),
    device: str = typer.Option("cpu"),
    lpips: bool = typer.Option(True, help="Compute LPIPS."),
) -> None:
    """Evaluate the files your pipeline made from a case's masters."""
    from .case import CaseError
    from .case import evaluate as run_evaluation
    from .report import evaluation

    try:
        report = run_evaluation(case, outputs, out, mapping=map_csv, note=note, align=align,
                                device=device, lpips=lpips)  # fmt: skip
    except CaseError as e:
        _fail(str(e))
    typer.echo(evaluation.write(report))
    typer.echo(f"wrote {report}/: summary.md files.csv summary.csv results.jsonl run.json")


@app.command()
def compare(
    a: Path = _folder("Report before (from evaluate or run)."),
    b: Path = _folder("Report after."),
    out: Path = typer.Option(None, help="Also write diff.csv and diff.md here."),
) -> None:
    """Compare two reports file by file: what regressed, improved or changed."""
    from .report import compare as cmp

    for d in (a, b):
        if not (d / "run.json").exists():
            _fail(f"{d} is not a report (no run.json)")
    result = cmp.compare(a, b)
    typer.echo(cmp.render(result))
    if out is not None:
        cmp.write(result, out)
        typer.echo(f"wrote {out}/: diff.csv diff.md")


if __name__ == "__main__":
    app()
