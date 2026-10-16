"""Image-clustered bootstrap intervals for rates: cells from one image are not independent."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable

import numpy as np


def cluster_bootstrap(
    cells: list[dict],
    pred: Callable[[dict], bool],
    n_boot: int = 2000,
    seed: int = 0,
    cluster: Callable[[dict], str] = lambda c: c["asset"]["id"],
) -> tuple[float, float, float]:
    """Rate of ``pred`` with a 95% percentile interval, resampling images (clusters)."""
    groups: dict[str, list[bool]] = defaultdict(list)
    for c in cells:
        groups[cluster(c)].append(bool(pred(c)))
    keys = sorted(groups)
    k = np.array([sum(groups[g]) for g in keys], dtype=float)
    n = np.array([len(groups[g]) for g in keys], dtype=float)
    rate = k.sum() / n.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    boots = k[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(rate), float(lo), float(hi)
