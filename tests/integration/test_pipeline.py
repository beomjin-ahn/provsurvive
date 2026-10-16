"""Prepare -> your pipeline -> evaluate -> compare, with known answers."""

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from provsurvive import case as pcase
from provsurvive import sources
from provsurvive.report import compare, evaluation
from provsurvive.sign.c2pa_io import C2paSession
from provsurvive.transforms import jumbf

pytestmark = pytest.mark.slow

TRUSTMARK = {"name": "trustmark", "variant": "Q", "device": "cpu", "rotation_search": False}
NAMES = ["beach.jpg", "harbor.png", "sub/park.png"]
CONSOLE: dict[Path, str] = {}


def _images(root: Path, names=NAMES, size=320) -> Path:
    for s, name in zip(sources.synthetic(len(names), size=size), names, strict=True):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        s.image.save(root / name, quality=92)
    return root


def _digest(d: Path) -> dict[str, str]:
    return {
        str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(d.rglob("*"))
        if p.is_file()
    }


@pytest.fixture(scope="module")
def case_dir(tmp_path_factory, session):
    root = tmp_path_factory.mktemp("pipeline")
    imgs = _images(root / "imgs")
    (imgs / "notes.txt").write_text("not an image")
    (imgs / "broken.jpg").write_bytes(b"\xff\xd8\xff\xe0 truncated")
    pcase.prepare(imgs, root / "case", TRUSTMARK, session.ca, lpips=False)
    return root / "case"


def _pipeline(case_dir: Path, out: Path) -> Path:
    """What a user's pipeline might hand back, one folder per setting."""
    masters = {p.stem: p for p in sorted((case_dir / "masters").iterdir())}

    def put(setting, name, data=None, img=None, **kw):
        (out / setting).mkdir(parents=True, exist_ok=True)
        if img is not None:
            img.save(out / setting / name, **kw)
        else:
            (out / setting / name).write_bytes(data)

    for k, m in masters.items():
        data, img = m.read_bytes(), Image.open(m).convert("RGB")
        put("copy", m.name, data)  # known good
        put("strip", m.name, jumbf.strip_c2pa(data))  # metadata lost, pixels kept
        put("resave", f"{k}.png", img=img)  # Pillow drops caBX: known loss
        put("reframe", f"{k}.jpg", img=img.crop((32, 0, 288, 320)), quality=90)  # 4:5
        put("half", f"{k}.png", img=img.resize((160, 160)))  # rescaled
    put("messy", "beach.jpg", masters["beach"].read_bytes()[:2000])  # truncated
    put("messy", "harbor.png", masters["harbor"].read_bytes())
    put("messy", "harbor.webp", img=Image.open(masters["harbor"]).convert("RGB"))  # duplicate
    put("messy", "IMG_0001.png", masters["park"].read_bytes())  # unmatched; park is missing
    put("messy", "readme.txt", b"ignored")
    put("messy", ".DS_Store", b"hidden")
    put("swapped", "beach.png", masters["harbor"].read_bytes())  # names swapped by mistake
    put("swapped", "harbor.png", masters["beach"].read_bytes())
    return out


@pytest.fixture(scope="module")
def report(case_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("eval")
    outputs = _pipeline(case_dir, root / "processed")
    before = _digest(case_dir)
    mp = pytest.MonkeyPatch()
    mp.setattr(C2paSession, "sign", lambda *a, **k: pytest.fail("evaluate signed something"))
    try:
        rep = pcase.evaluate(case_dir, outputs, root / "report", note="v1", lpips=False)
    finally:
        mp.undo()
    assert _digest(case_dir) == before  # the case is only read
    CONSOLE[rep] = evaluation.write(rep)
    return rep


def _cells(report: Path) -> dict[tuple[str, str], dict]:
    rows = [json.loads(line) for line in (report / "results.jsonl").read_text().splitlines()]
    return {(r["transform"]["id"], r["asset"]["id"]): r for r in rows}


def test_prepare_writes_the_case(case_dir):
    c = json.loads((case_dir / "case.json").read_text())
    assert [m["id"] for m in c["masters"]] == ["beach", "harbor", "park"]
    assert {s["file"] for s in c["skipped"]} == {"notes.txt", "broken.jpg"}
    assert all(m["baseline_ok"] for m in c["masters"])
    assert c["binding"]["kind"] == "watermark"
    assert not (case_dir / "leaf.key").exists() and (case_dir / "ca-root.pem").exists()
    for m in c["masters"]:
        assert m["payload"] and m["value"].endswith(m["payload"])
        assert (case_dir / m["file"]).suffix == ".png"


def test_known_states(report):
    cells = _cells(report)
    for master in ("beach", "harbor", "park"):
        assert cells[("copy", master)]["state"]["embedded"] == "intact"
        for setting in ("strip", "resave"):
            c = cells[(setting, master)]
            assert c["status"] == "ok"
            assert c["state"]["embedded"] == "absent"
            assert c["state"]["recovery"] == "correct"
            assert c["quality"]["alignment"] == "direct" and c["quality"]["psnr"] is None
        for setting in ("copy", "strip", "resave"):
            assert cells[(setting, master)]["state"]["disclosure_consistent"] is None
        reframed = cells[("reframe", master)]
        assert reframed["quality"] is None
        assert "aspect ratio differs" in reframed["not_evaluated"]["quality"]
        # the default alignment measures no rescaled output, but keeps its other observations
        half = cells[("half", master)]
        assert half["quality"] is None and "rescaled" in half["not_evaluated"]["quality"]
        assert half["state"]["embedded"] == "absent" and half["status"] == "ok"


def test_problems_are_counted_not_hidden(report):
    cells = _cells(report)
    assert cells[("messy", "beach")]["status"] == "error"
    assert "cannot read" in cells[("messy", "beach")]["error"]
    assert cells[("messy", "harbor")]["status"] == "error"
    assert cells[("messy", "harbor")]["error"].startswith("duplicate")
    assert cells[("messy", "park")]["status"] == "missing"
    for k in ("beach", "harbor"):
        assert "mapping conflict" in cells[("swapped", k)]["error"]
    run = json.loads((report / "run.json").read_text())
    assert run["counts"] == {
        "masters": 3,
        "settings": 7,
        "expected": 21,
        "present": 19,
        "evaluated": 15,
        "errors": 4,
        "missing": 2,  # park in messy and in swapped
        "unmatched": 1,
        "ignored": 1,
    }
    assert run["unmatched"] == ["messy/IMG_0001.png"] and run["ignored"] == ["messy/readme.txt"]
    ok = {m: True for m in ("beach", "harbor", "park")}
    rows = {r["setting"]: r for r in evaluation.by_setting(list(cells.values()), ok)}
    messy = rows["messy"]
    assert (messy["evaluated"], messy["errors"], messy["missing"]) == (0, 2, 1)
    assert messy["recovered_rate"] is None and messy["quality_n"] == 0  # n/a, not 0
    md = (report / "summary.md").read_text()
    assert "errors 4 · missing 2 · unmatched 1" in md
    assert "Not measured: false detection and false match" in md
    assert "messy/IMG_0001.png" in md and "messy/park" in md
    assert "missing 2" in CONSOLE[report]


def test_compare_shows_what_changed(case_dir, report, tmp_path):
    outputs = tmp_path / "processed"
    shutil.copytree(report.parent / "processed", outputs)
    shutil.rmtree(outputs / "messy")
    beach = outputs / "copy" / "beach.png"
    beach.write_bytes(jumbf.strip_c2pa(beach.read_bytes()))  # the new version strips it
    (outputs / "copy" / "park.png").unlink()  # and loses one
    again = pcase.evaluate(case_dir, outputs, tmp_path / "report", note="v2", lpips=False)
    result = compare.compare(report, again)
    assert result["header"][0].startswith("same case")
    rows = {(r["setting"], r["master"]): r for r in result["rows"]}
    assert rows[("copy", "beach")]["change"] == "regressed"
    assert rows[("copy", "beach")]["transitions"] == "embedded intact→absent"
    assert rows[("copy", "park")]["transitions"] == "status ok→missing"
    assert rows[("copy", "harbor")]["change"] == "same"
    assert rows[("messy", "beach")]["change"] == "only in A"
    assert "regressed 2" in compare.render(result)


def test_rescale_alignment_is_explicit(case_dir, report, tmp_path):
    outputs = tmp_path / "processed"
    shutil.copytree(report.parent / "processed" / "half", outputs / "half")
    rep = pcase.evaluate(case_dir, outputs, tmp_path / "report", align="rescale", lpips=False)
    for c in _cells(rep).values():
        assert c["quality"]["alignment"] == "upsampled" and c["not_evaluated"] is None
    assert json.loads((rep / "run.json").read_text())["config"]["resolved"]["align"] == "rescale"
    evaluation.write(rep)
    assert "rescale: quality is measured" in (rep / "summary.md").read_text()


def test_compare_never_compares_different_inputs(tmp_path, session):
    """Same configuration and image ids, different pixels: not a pipeline change."""
    pytest.importorskip("iscc_core")
    from provsurvive.bindings.iscc import IsccBinding
    from provsurvive.runner import RunConfig, Runner
    from provsurvive.transforms import base

    def run(seed, out):
        cfg = RunConfig(suite="t", transforms=[base.get("identity")], lpips=False, controls=False)
        return Runner(cfg, IsccBinding(), session, out).run(sources.synthetic(2, 256, seed))

    a, b = run(0, tmp_path / "a"), run(100, tmp_path / "b")
    result = compare.compare(a, b)
    assert result["header"][0].startswith("same configuration")
    assert "input differs, not compared: syn-000, syn-001" in result["header"]
    assert {r["change"] for r in result["rows"]} == {"not compared (input differs)"}
    assert {r["change"] for r in compare.compare(a, run(0, tmp_path / "c"))["rows"]} == {"same"}


def test_refuses_bad_inputs(case_dir, tmp_path, session):
    imgs = _images(tmp_path / "dup", names=["a.jpg", "sub/a.png"])
    with pytest.raises(pcase.CaseError, match="share a file name"):
        pcase.prepare(imgs, tmp_path / "c", TRUSTMARK, session.ca)
    with pytest.raises(pcase.CaseError, match="not empty"):
        pcase.prepare(imgs, case_dir, TRUSTMARK, session.ca)
    copy = tmp_path / "case"
    shutil.copytree(case_dir, copy)
    master = next((copy / "masters").iterdir())
    Image.open(master).save(master)  # e.g. a pipeline that processed the case in place
    with pytest.raises(pcase.CaseError, match="changed since prepare"):
        pcase.load_case(copy)


def test_jpeg_masters(tmp_path, session):
    imgs = _images(tmp_path / "imgs", size=384)
    c = pcase.prepare(imgs, tmp_path / "case", TRUSTMARK, session.ca, fmt="jpeg", lpips=False)
    for m in c["masters"]:
        assert pcase.sniff((tmp_path / "case" / m["file"]).read_bytes()) == "jpeg"
        assert m["baseline_ok"], m["baseline"]


def test_fingerprint_duplicates_are_ambiguous(tmp_path, session):
    pytest.importorskip("iscc_core")
    imgs = _images(tmp_path / "imgs", names=["a.png", "b.png"])
    shutil.copy(imgs / "a.png", imgs / "a_copy.png")
    c = pcase.prepare(imgs, tmp_path / "case", {"name": "iscc", "device": "cpu"}, session.ca,
                      lpips=False)  # fmt: skip
    assert c["binding"]["kind"] == "fingerprint"
    ok = {m["id"]: m["baseline_ok"] for m in c["masters"]}
    assert ok == {"a": False, "a_copy": False, "b": True}
    assert {m["id"]: m["baseline"]["recovery"] for m in c["masters"]}["a"] == "ambiguous"
    # unprocessed copies: every manifest is intact, and the tie stays visible
    shutil.copytree(tmp_path / "case" / "masters", tmp_path / "out" / "copy")
    rep = pcase.evaluate(tmp_path / "case", tmp_path / "out", tmp_path / "report", lpips=False)
    evaluation.write(rep)
    row = next(csv.DictReader(open(rep / "summary.csv")))
    assert (row["evaluated"], row["intact"], row["correct"], row["ambiguous"]) == (
        "3",
        "3",
        "1",
        "2",
    )
    assert (row["baseline_failed"], row["recovered"], row["recoverable"]) == ("2", "1", "1")
