"""Signing and reading C2PA manifests with c2pa-python, configured for the test CA."""

from __future__ import annotations

import io
import json
from typing import Any

import c2pa

from .test_ca import TestCA, TrustAnchor

IPTC = "http://cv.iptc.org/newscodes/digitalsourcetype/"
SOURCE_ASSERTION = "io.github.provsurvive.source"

MIME = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
    "avif": "image/avif",
}


def settings_json(ca: TestCA | TrustAnchor, *, thumbnail: bool) -> str:
    return json.dumps(
        {
            "version": 1,
            "trust": {
                "anchors": [{"trust_anchors": ca.root_pem.decode(), "trust_kind": "manifest"}]
            },
            "verify": {"verify_trust": True},
            "builder": {"thumbnail": {"enabled": thumbnail}},
        }
    )


class C2paSession:
    """Holds the c2pa context (trust anchors, builder settings) and the test signer; built on a
    ``TrustAnchor``, it can only read."""

    def __init__(self, ca: TestCA | TrustAnchor, *, thumbnail: bool = True) -> None:
        self.ca = ca
        self.thumbnail = thumbnail
        self._settings = settings_json(ca, thumbnail=thumbnail)

    def _context(self) -> c2pa.Context:
        return c2pa.Context(c2pa.Settings.from_json(self._settings))

    def _signer(self) -> c2pa.Signer:
        info = c2pa.C2paSignerInfo(b"es256", self.ca.chain_pem, self.ca.leaf_key_pem, None)
        return c2pa.Signer.from_info(info)

    def sign(
        self,
        image: bytes,
        fmt: str,
        manifest: dict[str, Any],
        parent: tuple[bytes, str] | None = None,
    ) -> tuple[bytes, bytes]:
        """Returns (signed asset bytes, manifest store bytes).

        ``parent``: (asset bytes, format) of a signed asset this one was edited from, added as
        a ``parentOf`` ingredient labelled "parent", as a C2PA-aware editor records it."""
        mime = MIME[fmt]
        manifest = {**manifest, "format": mime}
        out = io.BytesIO()
        with c2pa.Builder(json.dumps(manifest), context=self._context()) as builder:
            if parent is not None:
                data, pfmt = parent
                builder.add_ingredient(
                    {"title": "parent", "relationship": "parentOf", "label": "parent"},
                    MIME[pfmt],
                    io.BytesIO(data),
                )
            store = builder.sign(self._signer(), mime, io.BytesIO(image), out)
        return out.getvalue(), bytes(store)

    def read(self, data: bytes, mime: str) -> c2pa.Reader:
        """Raises c2pa.C2paError (ManifestNotFound etc.) like the underlying Reader."""
        return c2pa.Reader(mime, io.BytesIO(data), context=self._context())

    def read_manifest_store(
        self, store: bytes, mime: str, asset: bytes | None = None
    ) -> c2pa.Reader:
        """Read a detached manifest store (as recovered from the resolver)."""
        return c2pa.Reader(
            mime,
            io.BytesIO(asset or b""),
            manifest_data=store,
            context=self._context(),
        )


def master_manifest(
    *,
    title: str,
    digital_source_type: str,
    software_agent: str | None,
    soft_binding: tuple[str, str] | None,
    watermarked: bool,
    source: dict[str, Any],
) -> dict[str, Any]:
    """``watermarked``: the soft binding is a watermark embedded in the pixels, which the
    manifest records as a ``c2pa.watermarked.bound`` action; a fingerprint embeds nothing."""
    created: dict[str, Any] = {
        "action": "c2pa.created",
        "digitalSourceType": IPTC + digital_source_type,
    }
    if software_agent:
        created["softwareAgent"] = {"name": software_agent}
    actions = [created]
    assertions = [{"label": "c2pa.actions", "data": {"actions": actions}}]
    if soft_binding is not None:
        alg, value = soft_binding
        if watermarked:
            actions.append({"action": "c2pa.watermarked.bound"})
        assertions.append(
            {
                "label": "c2pa.soft-binding",
                "data": {"alg": alg, "blocks": [{"scope": {}, "value": value}]},
            }
        )
    assertions.append({"label": SOURCE_ASSERTION, "data": source})
    return {
        "claim_generator_info": [{"name": "provsurvive", "version": _version()}],
        "title": title,
        "assertions": assertions,
    }


def _version() -> str:
    from .. import __version__

    return __version__
