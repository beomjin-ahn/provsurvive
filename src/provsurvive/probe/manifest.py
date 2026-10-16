"""Turn a c2pa Reader result into raw observations (METHODOLOGY.md §4)."""

from __future__ import annotations

import json
from typing import Any

import c2pa

from ..schema import HardBinding, ManifestObs, StatusCode, StatusCodes
from ..sign.c2pa_io import C2paSession

_NOT_FOUND = ("ManifestNotFound", "JumbfNotFound")


def _codes(entries: list[dict[str, Any]] | None) -> list[StatusCode]:
    return [
        StatusCode(code=e["code"], url=e.get("url"), explanation=e.get("explanation"))
        for e in entries or []
    ]


def status_codes(results: dict[str, Any] | None) -> StatusCodes:
    active = (results or {}).get("activeManifest") or {}
    return StatusCodes(
        success=_codes(active.get("success")),
        informational=_codes(active.get("informational")),
        failure=_codes(active.get("failure")),
    )


def hard_binding(codes: StatusCodes) -> HardBinding | None:
    fail = {c.code for c in codes.failure}
    ok = {c.code for c in codes.success}
    if "claim.hardBindings.missing" in fail:
        return "missing"
    if any(c.endswith("Hash.mismatch") for c in fail):
        return "mismatch"
    if any(c.endswith("Hash.match") for c in ok):
        return "match"
    return None


def signature_ok(codes: StatusCodes) -> bool:
    fail = {c.code for c in codes.failure}
    ok = {c.code for c in codes.success}
    return "claimSignature.validated" in ok and not any(
        c.startswith("claimSignature") for c in fail
    )


def active_manifest(reader: c2pa.Reader) -> dict[str, Any] | None:
    store = json.loads(reader.json())
    label = store.get("active_manifest")
    return store.get("manifests", {}).get(label) if label else None


def digital_source_type(manifest: dict[str, Any] | None) -> str | None:
    """The digitalSourceType a manifest declares for its asset.

    A creating tool declares it on ``c2pa.created``; an editor that re-signs an edited asset
    opens the parent (``c2pa.opened``, often without a source type) and declares AI involvement
    on the editing action itself, e.g. ``c2pa.edited`` with compositeWithTrainedAlgorithmicMedia.
    So every action counts, and a declaration of AI involvement on any of them wins.
    """
    from .state import AI_SOURCE_TYPES

    declared = [
        act["digitalSourceType"]
        for a in (manifest or {}).get("assertions", [])
        if a.get("label", "").startswith("c2pa.actions")
        for act in a.get("data", {}).get("actions", [])
        if act.get("digitalSourceType")
    ]
    ai = [t for t in declared if t in AI_SOURCE_TYPES]
    return (ai or declared or [None])[0]


def observe(
    session: C2paSession, data: bytes, mime: str, original_label: str
) -> tuple[ManifestObs, dict[str, Any] | None]:
    """Returns the observation and the active manifest (for disclosure checks)."""
    try:
        reader = session.read(data, mime)
    except c2pa.C2paError as e:
        if any(tag in type(e).__name__ or tag in str(e) for tag in _NOT_FOUND):
            return _absent(), None
        return ManifestObs(
            present=True,
            read_error=f"{type(e).__name__}: {e}",
            validation_state=None,
            status_codes=None,
            signature_ok=False,
            hard_binding=None,
            active_label=None,
            is_original=None,
        ), None
    with reader:
        codes = status_codes(reader.get_validation_results())
        manifest = active_manifest(reader)
        label = json.loads(reader.json()).get("active_manifest")
        return ManifestObs(
            present=True,
            read_error=None,
            validation_state=reader.get_validation_state(),
            status_codes=codes,
            signature_ok=signature_ok(codes),
            hard_binding=hard_binding(codes),
            active_label=label,
            is_original=label == original_label,
        ), manifest


def _absent() -> ManifestObs:
    return ManifestObs(
        present=False,
        read_error=None,
        validation_state=None,
        status_codes=None,
        signature_ok=None,
        hard_binding=None,
        active_label=None,
        is_original=None,
    )
