"""Collect CC0 photographs from Wikimedia Commons with per-file license records.

Selection: files in both Category:CC-Zero and Category:Quality_images (photographs vetted by
the community), excluding the "Images from Unsplash" imports, bitmap, long side >= 2048 px.
The license is re-checked per file from the API's extmetadata; anything not CC0 is dropped.
Images are downloaded as 2048-px-wide Commons renderings and resized to 2048 px on the long
side, so portrait images are resampled twice.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from .. import HOMEPAGE, __version__

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia's User-Agent policy asks for a way to contact the operator.
UA = (
    f"provsurvive-corpus/{__version__} ({os.environ.get('PROVSURVIVE_CONTACT', HOMEPAGE)}; "
    "provenance robustness benchmark) python-requests"
)
SEARCH = (
    'incategory:"CC-Zero" incategory:"Quality_images" -incategory:"Images_from_Unsplash" '
    "filetype:bitmap filemime:image/jpeg filew:>2047"
)
LONG_SIDE = 2048


def _get(session: requests.Session, **params: Any) -> dict[str, Any]:
    params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    for attempt in range(8):
        time.sleep(1.0)  # stay well under Wikimedia's API rate limits
        r = session.post(API, data=params, timeout=60)  # POST: long title lists
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("Retry-After", 2 ** (attempt + 2))))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def search_titles(session: requests.Session, want: int) -> list[str]:
    titles: list[str] = []
    offset = 0
    while len(titles) < want:
        data = _get(
            session,
            action="query",
            list="search",
            srnamespace=6,
            srsearch=SEARCH,
            srlimit=50,
            sroffset=offset,
        )
        hits = data.get("query", {}).get("search", [])
        if not hits:
            break
        titles += [h["title"] for h in hits]
        offset = data.get("continue", {}).get("sroffset")
        if offset is None:
            break
    return titles


def file_info(session: requests.Session, titles: list[str]) -> list[dict[str, Any]]:
    out = []
    for i in range(0, len(titles), 20):
        data = _get(
            session,
            action="query",
            titles="|".join(titles[i : i + 20]),
            prop="imageinfo",
            iiprop="url|size|mime|extmetadata|sha1",
            iiurlwidth=LONG_SIDE,
        )
        for page in data.get("query", {}).get("pages", []):
            info = (page.get("imageinfo") or [None])[0]
            if info:
                out.append({"title": page["title"], **info})
    return out


def _meta(info: dict[str, Any], key: str) -> str | None:
    v = info.get("extmetadata", {}).get(key, {}).get("value")
    return str(v) if v is not None else None


def is_cc0(info: dict[str, Any]) -> bool:
    short = (_meta(info, "LicenseShortName") or "").lower()
    lic = (_meta(info, "License") or "").lower()
    return short in ("cc0", "cc-zero") or lic == "cc0"


def fetch(out_dir: Path, n: int = 100, seed: int = 20261031, pool: int = 600) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = UA
    titles = search_titles(session, pool)
    infos = [i for i in file_info(session, titles) if is_cc0(i) and i.get("mime") == "image/jpeg"]
    infos = [i for i in infos if max(i["width"], i["height"]) >= LONG_SIDE]
    # TrustMark Q and C center-crop inputs with aspect ratio > 2 (P always marks only the central
    # square); keep the corpus within 2 so that Q and C cover the whole frame.
    infos = [i for i in infos if max(i["width"], i["height"]) <= 2 * min(i["width"], i["height"])]
    rng = random.Random(seed)
    rng.shuffle(infos)

    index_path = out_dir / "index.json"
    index: list[dict[str, Any]] = json.loads(index_path.read_text()) if index_path.exists() else []
    have = {e["source_title"] for e in index}
    for info in infos:
        if len(index) >= n:
            break
        if info["title"] in have:
            continue
        url = info.get("thumburl") or info["url"]
        r = session.get(url, timeout=120)
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After", 30)))
            r = session.get(url, timeout=120)
        if r.status_code != 200:
            continue
        img = Image.open(io.BytesIO(r.content))
        img = img.convert("RGB")
        w, h = img.size
        s = LONG_SIDE / max(w, h)
        if s < 1:
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        pid = f"photo-{len(index):03d}"
        path = out_dir / f"{pid}.png"
        img.save(path)
        index.append(
            {
                "file": path.name,
                "id": pid,
                "kind": "photo",
                "license": "CC0-1.0",
                "license_short_name": _meta(info, "LicenseShortName"),
                "source_title": info["title"],
                "source_page": info.get("descriptionurl"),
                "source_url": info["url"],
                "source_sha1": info.get("sha1"),
                "artist_html": _meta(info, "Artist"),
                "original_size": [info["width"], info["height"]],
                "width": img.width,
                "height": img.height,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        index_path.write_text(json.dumps(index, indent=1))
        time.sleep(2.0)  # be polite to Wikimedia
    return index_path
