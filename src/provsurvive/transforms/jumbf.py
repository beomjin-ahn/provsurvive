"""Byte-level handling of embedded C2PA manifest stores (JUMBF).

Only the containers used for masters are supported: JPEG (APP11 segments whose
payload starts with the JPEG XT common identifier ``JP``) and PNG (``caBX``
chunks). Everything else in the file is copied byte for byte.
"""

from __future__ import annotations

import struct

JPEG_SOI = b"\xff\xd8"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_C2PA_CHUNK = b"caBX"
_APP11 = 0xEB
_SOS = 0xDA
_STANDALONE = {0x01, *range(0xD0, 0xD8)}


def sniff_format(data: bytes) -> str:
    if data.startswith(JPEG_SOI):
        return "jpeg"
    if data.startswith(PNG_SIGNATURE):
        return "png"
    raise ValueError("unsupported container (only JPEG and PNG masters are handled)")


# --- JPEG -------------------------------------------------------------------


def _jpeg_segments(data: bytes) -> list[tuple[int, bytes]]:
    """Split a JPEG into (marker, raw bytes) up to and including the scan data."""
    if not data.startswith(JPEG_SOI):
        raise ValueError("not a JPEG")
    segments: list[tuple[int, bytes]] = [(0xD8, data[:2])]
    i = 2
    while i < len(data):
        if data[i] != 0xFF:
            raise ValueError(f"expected marker at offset {i}")
        j = i
        while data[j + 1] == 0xFF:  # fill bytes
            j += 1
        marker = data[j + 1]
        if marker == _SOS:
            segments.append((marker, data[i:]))  # scan data and everything after it
            break
        if marker in _STANDALONE:
            segments.append((marker, data[i : j + 2]))
            i = j + 2
            continue
        (length,) = struct.unpack(">H", data[j + 2 : j + 4])
        end = j + 2 + length
        segments.append((marker, data[i:end]))
        i = end
    return segments


def _is_c2pa_app11(marker: int, raw: bytes) -> bool:
    # raw = FF EB <len:2> <CI "JP":2> ...
    return marker == _APP11 and raw[4:6] == b"JP"


def _jpeg_strip(data: bytes) -> bytes:
    return b"".join(raw for m, raw in _jpeg_segments(data) if not _is_c2pa_app11(m, raw))


def _jpeg_extract(data: bytes) -> list[bytes]:
    return [raw for m, raw in _jpeg_segments(data) if _is_c2pa_app11(m, raw)]


def _jpeg_insert(data: bytes, blocks: list[bytes]) -> bytes:
    segments = _jpeg_segments(_jpeg_strip(data))
    # Place after SOI and any leading APP0/APP1 (JFIF/EXIF), where writers put it.
    k = 1
    while k < len(segments) and segments[k][0] in (0xE0, 0xE1):
        k += 1
    raws = [raw for _, raw in segments]
    return b"".join(raws[:k] + blocks + raws[k:])


# --- PNG --------------------------------------------------------------------


def _png_chunks(data: bytes) -> list[tuple[bytes, bytes]]:
    """Split a PNG into (chunk type, raw chunk bytes incl. length and CRC)."""
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not a PNG")
    chunks = []
    i = len(PNG_SIGNATURE)
    while i < len(data):
        (length,) = struct.unpack(">I", data[i : i + 4])
        ctype = data[i + 4 : i + 8]
        end = i + 12 + length
        chunks.append((ctype, data[i:end]))
        i = end
        if ctype == b"IEND":
            break
    return chunks


def _png_strip(data: bytes) -> bytes:
    return PNG_SIGNATURE + b"".join(raw for t, raw in _png_chunks(data) if t != PNG_C2PA_CHUNK)


def _png_extract(data: bytes) -> list[bytes]:
    return [raw for t, raw in _png_chunks(data) if t == PNG_C2PA_CHUNK]


def _png_insert(data: bytes, blocks: list[bytes]) -> bytes:
    chunks = _png_chunks(_png_strip(data))
    raws = [raw for _, raw in chunks]
    return PNG_SIGNATURE + b"".join(raws[:1] + blocks + raws[1:])  # right after IHDR


# --- public API -------------------------------------------------------------


def strip_c2pa(data: bytes) -> bytes:
    """Remove the embedded C2PA manifest store; leave every other byte untouched."""
    return {"jpeg": _jpeg_strip, "png": _png_strip}[sniff_format(data)](data)


def extract_c2pa(data: bytes) -> list[bytes]:
    """Return the raw container blocks (APP11 segments / caBX chunks) holding C2PA data."""
    return {"jpeg": _jpeg_extract, "png": _png_extract}[sniff_format(data)](data)


def insert_c2pa(data: bytes, blocks: list[bytes]) -> bytes:
    """Replace any C2PA blocks in ``data`` with ``blocks`` (copied from another file)."""
    return {"jpeg": _jpeg_insert, "png": _png_insert}[sniff_format(data)](data, blocks)
