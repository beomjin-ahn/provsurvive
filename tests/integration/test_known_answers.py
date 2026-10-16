"""Known-answer tests: these define what "working" means."""

import hashlib
import json
import shutil

import pytest

from provsurvive import sources
from provsurvive.runner import RunConfig, Runner
from provsurvive.transforms import base

pytestmark = pytest.mark.slow

KNOWN = [
    "identity",
    "strip_metadata",
    "png_reencode",
    "carry_manifest_edit",
    "jpeg_q90",
    "rotate_90",
]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory, session, trustmark):
    cfg = RunConfig(suite="test", transforms=[base.get(t) for t in KNOWN], lpips=False)
    runner = Runner(cfg, trustmark, session, tmp_path_factory.mktemp("runs"))
    return runner.run(sources.synthetic(2, size=384))


def _cells(run_dir, name="results.jsonl"):
    rows = [json.loads(line) for line in (run_dir / name).read_text().splitlines()]
    return {(r["asset"]["id"], r["transform"]["id"]): r for r in rows}


def test_no_errors(run_dir):
    for r in _cells(run_dir).values():
        assert r["error"] is None, r["error"]


@pytest.mark.parametrize(
    "tid,embedded,recovery",
    [
        ("identity", "intact", "correct"),
        ("strip_metadata", "absent", "correct"),
        ("png_reencode", "absent", "correct"),  # Pillow drops caBX: tested, not assumed
        ("carry_manifest_edit", "stale", "correct"),
        ("jpeg_q90", "absent", "correct"),
    ],
)
def test_known_states(run_dir, tid, embedded, recovery):
    for (asset, t), r in _cells(run_dir).items():
        if t == tid:
            assert r["state"]["embedded"] == embedded, asset
            assert r["state"]["recovery"] == recovery, asset


def test_lossless_paths_decode_like_the_master(run_dir):
    cells = _cells(run_dir)
    for (asset, t), r in cells.items():
        if t in ("strip_metadata", "png_reencode"):
            ref = cells[(asset, "identity")]["softbinding"]
            assert r["softbinding"]["bit_accuracy"] == ref["bit_accuracy"]  # same pixels, same bits
            assert r["softbinding"]["payload_match"] is True
            assert r["quality"]["psnr"] is None  # pixel-identical to the master


def test_intact_is_trusted_by_test_ca(run_dir):
    for (_, t), r in _cells(run_dir).items():
        if t == "identity":
            assert r["manifest"]["validation_state"] == "Trusted"
            assert r["manifest"]["hard_binding"] == "match"


def test_stale_reports_hash_mismatch_with_valid_signature(run_dir):
    for (_, t), r in _cells(run_dir).items():
        if t == "carry_manifest_edit":
            codes = [c["code"] for c in r["manifest"]["status_codes"]["failure"]]
            assert "assertion.dataHash.mismatch" in codes
            assert r["manifest"]["signature_ok"] is True


def test_recovered_manifest_signature_validates(run_dir):
    for (_, t), r in _cells(run_dir).items():
        if r["state"]["recovery"] == "correct":
            assert r["recovery"]["recovered_signature_ok"] is True, t


def test_controls_never_match(run_dir):
    for r in _cells(run_dir, "controls.jsonl").values():
        assert r["error"] is None
        assert r["state"]["embedded"] == "absent"
        # an unmarked image has no manifest of its own, so any match at all is a false match
        assert r["state"]["recovery"] == "none"


def test_disclosure_for_synthetic_is_consistent(run_dir):
    # synthetic sources are algorithmicMedia (not trained AI), no generative edits here
    for r in _cells(run_dir).values():
        if r["state"]["provenance_available"]:
            assert r["state"]["disclosure_consistent"] is True


def _projection(run_dir):
    keep = {}
    for key, r in _cells(run_dir).items():
        keep[key] = {
            "state": r["state"],
            "softbinding": r["softbinding"],
            "quality": r["quality"],
            "output_sha": r["output"]["sha256"]
            if key[1] not in ("identity", "carry_manifest_edit")
            else None,
        }
    return keep


def test_reproducible(run_dir, tmp_path, session, trustmark):
    cfg = RunConfig(suite="test", transforms=[base.get(t) for t in KNOWN], lpips=False)
    again = Runner(cfg, trustmark, session, tmp_path).run(sources.synthetic(2, size=384))
    assert _projection(again) == _projection(run_dir)


def test_marked_master_decodes_in_the_deployed_schema(run_dir):
    for (_, t), r in _cells(run_dir).items():
        if t == "identity":
            assert r["softbinding"]["expected_schema"] is True
            assert r["softbinding"]["ecc_schema"] == 1  # TrustMark Encoding.BCH_5


# The proposed `benchmarkingInformation` fragment of the registry schema, so the test fails when
# the shape drifts.
REGISTRY_BENCHMARKING_INFORMATION = {
    "type": "array",
    "minItems": 1,
    "items": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "coveredMediaTypes": {
                "type": "array",
                "minItems": 1,
                "items": {"enum": ["application", "audio", "image", "model", "text", "video"]},
            },
            "benchmarkDate": {
                "type": "string",
                "format": "date",
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
            },
            "results": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri"},
                    "sha256": {"type": "string", "pattern": "^[A-Fa-f0-9]{64}$"},
                },
                "required": ["url", "sha256"],
            },
        },
        "required": ["description", "coveredMediaTypes", "benchmarkDate", "results"],
    },
}


def test_reports_render(run_dir):
    from provsurvive.report import benchinfo, summary

    summary.write(run_dir)
    results = benchinfo.build_results(run_dir)
    assert results["alg"] == "com.adobe.trustmark.Q"
    assert results["cells"]["errors"] == 0 and results["rates_over"] == "evaluated cells"
    assert results["controls"]["false_detection_deployed_schema_cells"] is not None
    item = benchinfo.build_record(results)[0]
    assert set(item) == {"description", "coveredMediaTypes", "benchmarkDate", "results"}
    assert item["results"]["sha256"] == hashlib.sha256(benchinfo.results_bytes(results)).hexdigest()
    assert item["results"]["url"].startswith("urn:provsurvive:run:")
    benchinfo.write(run_dir)
    written = (run_dir / benchinfo.RESULTS_FILE).read_bytes()
    assert hashlib.sha256(written).hexdigest() == item["results"]["sha256"]
    jsonschema = pytest.importorskip("jsonschema")
    jsonschema.validate(
        json.loads((run_dir / benchinfo.RECORD_FILE).read_text()), REGISTRY_BENCHMARKING_INFORMATION
    )


def test_results_file_counts_errors_and_failed_controls(run_dir, tmp_path):
    """A failed cell is counted, not dropped; controls that ran and all failed are not 'not run'."""
    from provsurvive.report import benchinfo

    d = tmp_path / "run"
    shutil.copytree(run_dir, d)

    def failed(c):
        c = dict(c, status="error", error="RuntimeError: boom")
        return (
            c
            | {k: None for k in ("output", "manifest", "softbinding", "recovery", "state")}
            | {"quality": None}
        )

    cells = [json.loads(line) for line in (d / "results.jsonl").read_text().splitlines()]
    jpeg = next(c for c in cells if c["transform"]["id"] == "jpeg_q90")
    with open(d / "results.jsonl", "a") as f:
        f.write(json.dumps(failed(jpeg)) + "\n")
    ctrl = [json.loads(line) for line in (d / "controls.jsonl").read_text().splitlines()]
    (d / "controls.jsonl").write_text("".join(json.dumps(failed(c)) + "\n" for c in ctrl))

    r = benchinfo.build_results(d)
    assert r["cells"] == {"expected": len(cells) + 1, "evaluated": len(cells), "errors": 1}
    assert r["recovery_by_family"]["codec"]["errors"] == 1
    assert r["controls"]["run"] is True and r["controls"]["errors"] == len(ctrl)
    assert r["controls"]["evaluated"] == 0 and r["controls"]["false_detection_cells"] is None


def test_recovered_manifest_hard_binding_reported(run_dir):
    for (_, t), r in _cells(run_dir).items():
        if t == "jpeg_q90" and r["state"]["recovery"] == "correct":
            # the original manifest's data hash cannot match the re-encoded file
            assert r["recovery"]["recovered_hard_binding"] == "mismatch"
        if t == "identity":
            assert r["recovery"]["recovered_hard_binding"] == "match"


def test_c2pa_aware_editor_is_superseded_and_discloses_the_edit(session):
    """The case the `superseded` state exists for: an editor that records its AI edit re-signs
    the edited asset with the master as a parentOf ingredient and declares
    compositeWithTrainedAlgorithmicMedia on its editing action. The verifier then holds a valid
    manifest bound to the edited bytes that is not the master's, and it is accurate about AI."""
    from provsurvive.probe.manifest import digital_source_type, observe
    from provsurvive.probe.state import derive_state, embedded_state
    from provsurvive.schema import RecoveryObs
    from provsurvive.sign.c2pa_io import IPTC, MIME, master_manifest

    src = sources.synthetic(1, size=256)[0]
    manifest = master_manifest(
        title=src.id,
        digital_source_type="digitalCapture",
        software_agent=None,
        soft_binding=None,
        watermarked=False,
        source={"asset_id": src.id},
    )
    master, _ = session.sign(base.encode(src.image, "png"), "png", manifest)
    with session.read(master, MIME["png"]) as reader:
        master_label = json.loads(reader.json())["active_manifest"]

    edited = base.get("exposure_+0.5ev")(master, 0, None).data  # re-encoded, no manifest
    ai_edit = IPTC + "compositeWithTrainedAlgorithmicMedia"
    editor_manifest = {
        "claim_generator_info": [{"name": "c2pa-aware-editor", "version": "0"}],
        "title": "edited",
        "assertions": [
            {
                "label": "c2pa.actions",
                "data": {
                    "actions": [
                        {"action": "c2pa.opened", "parameters": {"ingredientIds": ["parent"]}},
                        {"action": "c2pa.edited", "digitalSourceType": ai_edit},
                    ]
                },
            }
        ],
    }
    resigned, _ = session.sign(edited, "png", editor_manifest, parent=(master, "png"))

    obs, active = observe(session, resigned, MIME["png"], master_label)
    assert embedded_state(obs) == "superseded"
    assert obs.hard_binding == "match" and obs.is_original is False
    ingredient = active["ingredients"][0]
    assert ingredient["relationship"] == "parentOf"
    assert ingredient.get("active_manifest") == master_label
    dst = digital_source_type(active)
    assert dst == ai_edit
    state = derive_state(
        obs,
        RecoveryObs(outcome="none", candidates=[], recovered_signature_ok=None),
        truth_involves_ai=True,
        available_source_type=dst,
    )
    assert state.provenance_available and state.disclosure_consistent is True


def test_fingerprint_runs_no_unmarked_controls(tmp_path, session):
    """An unmarked copy of a master has the master's fingerprint and rightly finds its manifest;
    counting that as a false match would contradict METHODOLOGY §6."""
    pytest.importorskip("iscc_core")
    from provsurvive.bindings.iscc import IsccBinding
    from provsurvive.report import benchinfo, summary

    cfg = RunConfig(suite="test", transforms=[base.get("identity")], lpips=False)
    run_dir = Runner(cfg, IsccBinding(), session, tmp_path).run(sources.synthetic(2, size=256))
    assert (run_dir / "controls.jsonl").read_text() == ""
    assert "misattribution" in json.loads((run_dir / "run.json").read_text())["controls_not_run"]
    assert "Not run: fingerprint" in summary.write(run_dir).read_text()
    controls = benchinfo.build_results(run_dir)["controls"]
    assert controls["run"] is False and controls["why_not"]
    assert "false_detection_cells" not in controls
