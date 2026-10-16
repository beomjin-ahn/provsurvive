"""Capture the execution environment for run.json."""

from __future__ import annotations

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PACKAGES = [
    "provsurvive",
    "c2pa-python",
    "trustmark",
    "Pillow",
    "numpy",
    "scikit-image",
    "torch",
    "torchvision",
    "lpips",
    "diffusers",
    "transformers",
    "iscc-core",
]


def _pkg(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    dirty = run("status", "--porcelain")
    return {"git_commit": commit, "dirty": bool(dirty) if dirty is not None else None}


def _gpus() -> list[dict[str, Any]]:
    try:
        import torch
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [
        {
            "name": torch.cuda.get_device_name(i),
            "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1),
            "cuda": torch.version.cuda,
        }
        for i in range(torch.cuda.device_count())
    ]


def _codecs() -> dict[str, str | None]:
    from PIL import features

    out = {}
    for key in ("jpg", "libjpeg_turbo", "webp", "avif", "zlib"):
        try:
            out[key] = features.version(key)
        except ValueError:
            out[key] = None
    return out


def _cpu_model() -> str | None:
    """The CPU model (learned encoders are not bit-identical across CPU types)."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def environment() -> dict[str, Any]:
    import c2pa

    packages = {p: _pkg(p) for p in PACKAGES}
    packages["c2pa_sdk"] = c2pa.sdk_version()
    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cpu": _cpu_model() or platform.processor() or platform.machine(),
        "gpus": _gpus(),
        "packages": packages,
        "codecs": _codecs(),
    }


def tool_info(root: Path) -> dict[str, Any]:
    from . import __version__

    return {"name": "provsurvive", "version": __version__, **_git(root)}
