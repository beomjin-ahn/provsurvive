import itertools

import pytest

from provsurvive.probe.state import AI_SOURCE_TYPES, derive_state, embedded_state
from provsurvive.schema import ManifestObs, RecoveryObs

AI = next(iter(AI_SOURCE_TYPES))
CAPTURE = "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"


def mobs(present, read_error=None, sig=None, hb=None, orig=None):
    return ManifestObs(
        present=present,
        read_error=read_error,
        validation_state=None,
        status_codes=None,
        signature_ok=sig,
        hard_binding=hb,
        active_label=None,
        is_original=orig,
    )


def expected_embedded(present, err, sig, hb, orig):
    if not present:
        return "absent"
    if err or not sig:
        return "invalid"
    if hb == "match":
        return "intact" if orig else "superseded"
    return "stale"


@pytest.mark.parametrize(
    "present,err,sig,hb,orig",
    list(
        itertools.product(
            [True, False],
            [None, "boom"],
            [True, False],
            ["match", "mismatch", "missing", None],
            [True, False],
        )
    ),
)
def test_embedded_truth_table(present, err, sig, hb, orig):
    m = mobs(present, err, sig, hb, orig)
    assert embedded_state(m) == expected_embedded(present, err, sig, hb, orig)


@pytest.mark.parametrize("outcome", ["correct", "wrong", "ambiguous", "none"])
@pytest.mark.parametrize("embedded", ["intact", "superseded", "absent"])
@pytest.mark.parametrize("truth_ai", [True, False, None])  # None: unknown, e.g. external edits
@pytest.mark.parametrize("dst", [AI, CAPTURE, None])
def test_derived_flags(outcome, embedded, truth_ai, dst):
    original = embedded == "intact"
    m = mobs(False) if embedded == "absent" else mobs(True, None, True, "match", original)
    s = derive_state(
        m,
        RecoveryObs(outcome=outcome, candidates=[], recovered_signature_ok=None),
        truth_involves_ai=truth_ai,
        available_source_type=dst,
    )
    available = embedded in ("intact", "superseded") or outcome == "correct"
    assert s.provenance_available is available
    assert s.misattributed is (outcome == "wrong")
    if not available or dst is None or truth_ai is None:
        assert s.disclosure_consistent is None
    else:
        assert s.disclosure_consistent is ((dst == AI) == truth_ai)


IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"


def _manifest(*actions):
    return {"assertions": [{"label": "c2pa.actions.v2", "data": {"actions": list(actions)}}]}


@pytest.mark.parametrize(
    "actions,expected",
    [
        # a creating tool: the source type sits on c2pa.created
        ([{"action": "c2pa.created", "digitalSourceType": CAPTURE}], CAPTURE),
        # an editor that re-signs: opens the parent without a source type and declares the AI
        # edit on the editing action
        (
            [
                {"action": "c2pa.opened"},
                {
                    "action": "c2pa.edited",
                    "digitalSourceType": IPTC + "compositeWithTrainedAlgorithmicMedia",
                },
            ],
            IPTC + "compositeWithTrainedAlgorithmicMedia",
        ),
        # a declaration of AI involvement on any action wins over a capture claim
        (
            [
                {"action": "c2pa.opened", "digitalSourceType": CAPTURE},
                {"action": "c2pa.placed", "digitalSourceType": AI},
            ],
            AI,
        ),
        ([{"action": "c2pa.watermarked.bound"}], None),
    ],
)
def test_digital_source_type_reads_every_action(actions, expected):
    from provsurvive.probe.manifest import digital_source_type

    assert digital_source_type(_manifest(*actions)) == expected
    assert digital_source_type(None) is None
