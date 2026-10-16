"""Pure derivation of survival states from raw observations (METHODOLOGY.md §5)."""

from __future__ import annotations

from ..schema import Embedded, ManifestObs, Recovery, RecoveryObs, State

AI_SOURCE_TYPES = {
    "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia",
    "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia",
}


def embedded_state(m: ManifestObs) -> Embedded:
    if not m.present:
        return "absent"
    if m.read_error is not None or not m.signature_ok:
        return "invalid"
    if m.hard_binding == "match":
        # a valid manifest bound to these bytes: ours (intact) or a successor's, e.g. from a
        # C2PA-aware editor that re-signed the edited asset (superseded)
        return "intact" if m.is_original else "superseded"
    return "stale"


def derive_state(
    manifest: ManifestObs,
    recovery: RecoveryObs,
    *,
    truth_involves_ai: bool | None,
    available_source_type: str | None,
) -> State:
    """Combine the embedded and recovery axes into the derived flags.

    ``truth_involves_ai``: whether the transformed image actually contains AI-generated
    content (generated master, or a generative edit); None if unknown, e.g. for a file processed
    outside provsurvive, which leaves ``disclosure_consistent`` unevaluated.
    ``available_source_type``: the ``digitalSourceType`` of the manifest a verifier would get
    (embedded if intact or superseded, otherwise the recovered one), or None when no provenance
    is available.
    """
    embedded = embedded_state(manifest)
    rec: Recovery = recovery.outcome
    available = embedded in ("intact", "superseded") or rec == "correct"
    if not available or available_source_type is None or truth_involves_ai is None:
        disclosure = None
    else:
        disclosure = (available_source_type in AI_SOURCE_TYPES) == truth_involves_ai
    return State(
        embedded=embedded,
        recovery=rec,
        provenance_available=available,
        misattributed=rec == "wrong",
        disclosure_consistent=disclosure,
    )
