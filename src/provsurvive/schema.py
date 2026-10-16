"""Result records. See docs/METHODOLOGY.md §4–5.

Every nullable field is emitted as an explicit ``null``; nothing is omitted.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# 1.1: softbinding.ecc_schema / expected_schema; 1.2: recovered_hard_binding;
# 1.3: embedded state "superseded" (a valid manifest that is not ours now describes the bytes);
# 1.4: edit; 1.5: recovery "ambiguous", status, not_evaluated (externally processed files)
SCHEMA_VERSION = "1.5"

Embedded = Literal["intact", "superseded", "stale", "invalid", "absent"]
Recovery = Literal["correct", "wrong", "ambiguous", "none"]
HardBinding = Literal["match", "mismatch", "missing"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetRef(_Strict):
    id: str
    subset: Literal["generated", "photo", "fixture", "synthetic"]
    master_format: str
    master_sha256: str


class TransformRef(_Strict):
    id: str
    family: str
    params: dict[str, Any]
    seed: int
    impl_version: str
    gpu: bool


class OutputRef(_Strict):
    format: str
    width: int
    height: int
    bytes: int
    sha256: str


class BindingRef(_Strict):
    plugin: str
    alg: str
    config: dict[str, Any]


class StatusCode(_Strict):
    code: str
    url: str | None = None
    explanation: str | None = None


class StatusCodes(_Strict):
    success: list[StatusCode]
    informational: list[StatusCode]
    failure: list[StatusCode]


class ManifestObs(_Strict):
    present: bool
    read_error: str | None
    validation_state: Literal["Invalid", "Valid", "Trusted"] | None
    status_codes: StatusCodes | None
    signature_ok: bool | None
    hard_binding: HardBinding | None
    active_label: str | None
    is_original: bool | None


class SoftBindingObs(_Strict):
    detected: bool
    payload: str | None
    # Error-correction schema the decoder settled on, and whether it is the one the deployment
    # embeds. TrustMark auto-detects among four BCH schemas; low-redundancy schemas can decode
    # random bits, so detections in an unexpected schema are likely false.
    ecc_schema: int | None = None
    expected_schema: bool | None = None
    payload_match: bool | None  # None for fingerprints (no payload; see distance)
    bit_accuracy: float | None
    distance: float | None
    distance_threshold: float | None


class Candidate(_Strict):
    label: str
    score: float


class RecoveryObs(_Strict):
    outcome: Recovery
    candidates: list[Candidate]
    recovered_signature_ok: bool | None
    # hard binding of the recovered manifest against the *edited* asset: a mismatch is expected
    # (C2PA 2.4 soft binding lookup) and is the only machine-readable hint that the asset changed
    recovered_hard_binding: HardBinding | None = None


class State(_Strict):
    embedded: Embedded
    recovery: Recovery
    provenance_available: bool
    misattributed: bool
    disclosure_consistent: bool | None


class Quality(_Strict):
    alignment: Literal[
        "direct", "upsampled", "cropped_reference", "inverted", "approximate", "masked"
    ]
    psnr: float | None
    ssim: float | None
    lpips: float | None
    retained_area: float
    inside_mask: dict[str, float | None] | None = None
    outside_mask: dict[str, float | None] | None = None


class CellResult(_Strict):
    schema_version: str = SCHEMA_VERSION
    run_id: str
    cell_id: str
    control: bool = Field(description="true for unmarked-source control cells")
    asset: AssetRef
    transform: TransformRef
    output: OutputRef | None
    binding: BindingRef
    manifest: ManifestObs | None
    softbinding: SoftBindingObs | None
    recovery: RecoveryObs | None
    state: State | None
    quality: Quality | None
    edit: dict[str, Any] | None = Field(
        None,
        description="what a parametrized edit did: e.g. the inpainting ellipse, editor, model "
        "revision, prompt, steps and generative cache key (schema 1.4)",
    )
    timing_ms: dict[str, float]
    error: str | None
    status: Literal["ok", "error", "missing"] = Field(
        "ok", description="missing: no output file for this master (externally processed files)"
    )
    not_evaluated: dict[str, str] | None = Field(
        None, description="field -> why it was not evaluated (schema 1.5)"
    )
