"""A toy image pipeline for the demo.

  python toy_pipeline.py images DIR             four procedural test images
  python toy_pipeline.py v1|v2 MASTERS OUT      process the masters, one folder per setting:

    copy/   the file as it is
    jpeg/   re-encoded as JPEG at quality 85, which drops the manifest
    strip/  the manifest removed, pixels kept

v2 changes copy/: it strips the first file's manifest and loses the second file.
"""

import sys
from pathlib import Path

from PIL import Image

from provsurvive.sources import synthetic
from provsurvive.transforms.jumbf import strip_c2pa


def images(out: Path) -> None:
    out.mkdir(parents=True)
    for s in synthetic(4):
        s.image.save(out / f"{s.id}.png")


def pipeline(version: str, masters: Path, out: Path) -> None:
    for setting in ("copy", "jpeg", "strip"):
        (out / setting).mkdir(parents=True)
    for m in sorted(masters.iterdir()):
        data = m.read_bytes()
        (out / "copy" / m.name).write_bytes(data)
        Image.open(m).convert("RGB").save(out / "jpeg" / f"{m.stem}.jpg", quality=85)
        (out / "strip" / m.name).write_bytes(strip_c2pa(data))
    if version == "v2":
        first, second = sorted((out / "copy").iterdir())[:2]
        first.write_bytes(strip_c2pa(first.read_bytes()))
        second.unlink()


if __name__ == "__main__":
    if sys.argv[1] == "images":
        images(Path(sys.argv[2]))
    else:
        pipeline(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]))
