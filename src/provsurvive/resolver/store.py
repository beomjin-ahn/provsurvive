"""Local manifest repository with Soft Binding Resolution API matching semantics.

Mirrors ``/matches/byBinding`` (C2PA 2.4 §18.10.5): a match needs the same ``alg``
and a value that matches "in the manner prescribed by the specified algorithm",
which is delegated to the plugin's ``match``. In-process only; no network.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from ..bindings.base import Decoded, SoftBinding
from ..schema import Candidate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS manifests (
    label TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL,
    mime TEXT NOT NULL,
    store BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS bindings (
    label TEXT NOT NULL REFERENCES manifests(label),
    alg TEXT NOT NULL,
    value TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS bindings_alg ON bindings(alg);
"""


class ManifestRepository:
    def __init__(self, path: Path | str = ":memory:", *, readonly: bool = False) -> None:
        if readonly:
            uri = Path(path).resolve().as_uri() + "?mode=ro"
            self.db = sqlite3.connect(uri, uri=True, timeout=60)
            return
        self.db = sqlite3.connect(str(path), timeout=60)
        self.db.executescript(_SCHEMA)

    def add(
        self, *, label: str, asset_id: str, mime: str, store: bytes, alg: str, value: str
    ) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO manifests VALUES (?, ?, ?, ?)",
                (label, asset_id, mime, store),
            )
            self.db.execute("INSERT INTO bindings VALUES (?, ?, ?)", (label, alg, value))

    def get(self, label: str) -> tuple[str, bytes] | None:
        row = self.db.execute(
            "SELECT mime, store FROM manifests WHERE label = ?", (label,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def match_by_binding(
        self, plugin: SoftBinding, decoded: Decoded, top_k: int = 5
    ) -> list[Candidate]:
        rows = self.db.execute("SELECT label, value FROM bindings WHERE alg = ?", (plugin.alg,))
        scored = []
        for label, value in rows:
            score = plugin.match(decoded, value)
            if score is not None:
                scored.append(Candidate(label=label, score=score))
        scored.sort(key=lambda c: (-c.score, c.label))
        return scored[:top_k]
