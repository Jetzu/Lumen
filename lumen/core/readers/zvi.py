"""Zeiss AxioVision ZVI reader (pure Python).

A ZVI file is an OLE2 compound document.  ``Image/Contents`` holds the image header, every plane is an
``Image/Item(n)/Contents`` stream and its tags (Z / C / T index, channel name, exposure …) live in
``Image/Item(n)/Tags/Contents``; ``Image/Tags/Contents`` carries the global tags (scale factors and units,
acquisition date, objective …).  Values are COM ``VARIANT``s: a 2-byte ``VT_*`` type code followed by the data
(``VT_I4`` = 4-byte int, ``VT_R8`` = double, ``VT_BSTR`` = 4-byte byte length + UTF-16LE text, ``VT_BLOB`` /
``VT_STORED_OBJECT`` = 4-byte length + bytes, …).

A plane's pixels sit at the end of its ``Contents`` stream behind a 28-byte raw header (version, width, height,
depth, bytes per pixel, pixel format, valid bits per pixel); the reader anchors on that header and falls back to
the Image header's sizes when it is absent.  Pixel formats: 1 = 8-bit BGR, 2 = 8-bit BGRA, 3 = 8-bit gray,
4 = 16-bit, 5 = 32-bit int, 6 = 32-bit float, 7 = 64-bit float, 8 = 16-bit BGR, 9 = 32-bit BGR.
"""
from __future__ import annotations

import os
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from . import NativeSeries
from ._ole import OleContainer, OleError, is_ole

FORMAT_NAME = "Zeiss ZVI"

# VT_* codes
VT_EMPTY, VT_NULL, VT_I2, VT_I4, VT_R4, VT_R8, VT_DATE, VT_BSTR, VT_DISPATCH, VT_BOOL, VT_UNKNOWN = 0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13
VT_I1, VT_UI1, VT_UI2, VT_UI4, VT_I8, VT_UI8, VT_INT, VT_UINT = 16, 17, 18, 19, 20, 21, 22, 23
VT_ARRAY, VT_BLOB, VT_STREAM, VT_STORAGE, VT_STREAMED_OBJECT, VT_STORED_OBJECT, VT_CLSID = 63, 65, 66, 67, 68, 69, 72

_FIXED = {VT_I2: "<h", VT_I4: "<i", VT_R4: "<f", VT_R8: "<d", VT_DATE: "<d", VT_BOOL: "<h", VT_I1: "<b", VT_UI1: "<B", VT_UI2: "<H",
          VT_UI4: "<I", VT_I8: "<q", VT_UI8: "<Q", VT_INT: "<i", VT_UINT: "<I"}

#: pixel format -> (numpy dtype, samples per pixel, bgr order)
PIXEL_FORMATS: dict[int, tuple[str, int, bool]] = {1: ("u1", 3, True), 2: ("u1", 4, True), 3: ("u1", 1, False), 4: ("u2", 1, False),
                                                   5: ("i4", 1, False), 6: ("f4", 1, False), 7: ("f8", 1, False), 8: ("u2", 3, True),
                                                   9: ("i4", 3, True)}

#: tag IDs (the AxioVision tag table as Bio-Formats names them)
TAG_NAMES: dict[int, str] = {
    222: "Compression", 258: "BlackValue", 259: "WhiteValue", 260: "ImageDataMappingAutoRange", 261: "Thumbnail", 262: "GammaValue",
    264: "ImageOverExposure", 265: "ImageRelativeTime1", 266: "ImageRelativeTime2", 267: "ImageRelativeTime3", 268: "ImageRelativeTime4",
    333: "RelFocusPosition1", 334: "RelFocusPosition2", 513: "ObjectType", 515: "ImageWidth", 516: "ImageHeight", 517: "Number Raw Count",
    518: "PixelType", 519: "NumberOfRawImages", 520: "ImageSize", 523: "Acquisition pause annotation", 530: "Document Subtype",
    531: "Acquisition Bit Depth", 532: "Image Memory Usage (RAM)", 534: "Z-Stack single representative", 769: "Scale Factor for X",
    770: "Scale Unit for X", 771: "Scale Width", 772: "Scale Factor for Y", 773: "Scale Unit for Y", 774: "Scale Height",
    775: "Scale Factor for Z", 776: "Scale Unit for Z", 777: "Scale Depth", 778: "Scaling Parent", 1001: "Date", 1002: "code", 1003: "Source",
    1004: "Message", 1025: "Acquisition Date", 1026: "8-bit acquisition", 1027: "Camera Bit Depth", 1029: "MonoReferenceLow",
    1030: "MonoReferenceHigh", 1031: "RedReferenceLow", 1032: "RedReferenceHigh", 1033: "GreenReferenceLow", 1034: "GreenReferenceHigh",
    1035: "BlueReferenceLow", 1036: "BlueReferenceHigh", 1041: "FramesizeX", 1042: "FramesizeY", 1281: "Camera Frame Width",
    1282: "Camera Frame Height", 1537: "Camera Exposure Time", 2049: "Camera Name", 2050: "Objective Name", 2070: "Image Index Z",
    2071: "Image Index C", 2072: "Image Index T", 2073: "Image Index S", 2074: "Image Index RAW", 2075: "Image Tile Index",
    2076: "Image acquisition Index", 2077: "Image Count Tiles", 2078: "Image Count A", 2080: "Image Count Z", 2081: "Image Count C",
    2082: "Image Count T", 2244: "Channel Name", 2245: "Exposure Time [ms]", 2246: "Emission Wavelength", 2247: "Excitation Wavelength",
    2258: "Stage Position X", 2259: "Stage Position Y", 2260: "Stage Position Z", 2261: "Focus Position",
}
_SCALE_UNITS = {72: "µm", 76: "mm", 77: "cm", 80: "nm", 0: "µm"}


class ZVIError(ValueError):
    pass


# ---------------------------------------------------------------------------- VARIANT parsing
class _Stream:
    def __init__(self, data: bytes, name: str):
        self.data, self.pos, self.name = data, 0, name

    def eof(self) -> bool:
        return self.pos >= len(self.data)

    def _take(self, n: int) -> bytes:
        if n < 0 or self.pos + n > len(self.data):
            raise ZVIError(f"{self.name}: truncated value")
        out = self.data[self.pos:self.pos + n]
        self.pos += n
        return out

    def int32(self) -> int:
        return struct.unpack("<i", self._take(4))[0]

    def variant(self) -> tuple[int, Any]:
        """The next VARIANT as (type code, value)."""
        vt = struct.unpack("<H", self._take(2))[0]
        if vt in (VT_EMPTY, VT_NULL):
            return vt, None
        if vt in _FIXED:
            fmt = _FIXED[vt]
            v = struct.unpack(fmt, self._take(struct.calcsize(fmt)))[0]
            return vt, (bool(v) if vt == VT_BOOL else v)
        if vt == VT_BSTR:
            n = self.int32()
            return vt, self._take(n).decode("utf-16-le", "replace").rstrip("\x00")
        if vt in (VT_DISPATCH, VT_UNKNOWN, VT_CLSID):
            return vt, self._take(16)
        if vt in (VT_BLOB, VT_STREAMED_OBJECT, VT_ARRAY):
            n = self.int32()
            return vt, self._take(n)
        if vt == VT_STORED_OBJECT:
            n = self.int32()
            return vt, self._take(n).decode("utf-16-le", "replace").rstrip("\x00")
        if vt in (VT_STREAM, VT_STORAGE):
            n = struct.unpack("<H", self._take(2))[0]
            return vt, self._take(n).decode("utf-16-le", "replace").rstrip("\x00")
        raise ZVIError(f"{self.name}: unknown VARIANT type {vt} at offset {self.pos - 2}")


def parse_tags(data: bytes, name: str = "Tags") -> dict[int, Any]:
    """``Tags/Contents``: version (VT_I4), count (VT_I4), then count × (value VARIANT, id VT_I4, attribute VT_I4)."""
    s = _Stream(data, name)
    out: dict[int, Any] = {}
    try:
        s.variant()  # version
        _, count = s.variant()
    except ZVIError:
        return out
    if not isinstance(count, int):
        return out
    for _ in range(max(0, count)):
        if s.eof():
            break
        try:
            _, value = s.variant()
            _, tag_id = s.variant()
            _, _attr = s.variant()
        except ZVIError:
            break
        if isinstance(tag_id, int):
            out[tag_id] = value
    return out


def parse_image_header(data: bytes, name: str = "Image/Contents") -> dict[str, Any]:
    """The 10 leading VARIANTs of an Image / Item Contents stream: version, type, type description, file name, width,
    height, depth, pixel format, count, valid bits per pixel (whatever parses before the first unknown type)."""
    s = _Stream(data, name)
    keys = ["version", "type", "type_description", "file_name", "width", "height", "depth", "pixel_format", "count", "valid_bits"]
    out: dict[str, Any] = {}
    for key in keys:
        if s.eof():
            break
        try:
            _, v = s.variant()
        except ZVIError:
            break
        out[key] = v
    out["_header_end"] = s.pos
    return out


_ITEM_RE = re.compile(r"^Image/Item\((\d+)\)/Contents$", re.IGNORECASE)


def _plane_block(stream: bytes, hint: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """(offset of the pixel bytes, width, height, pixel format, bytes per pixel) of an Item Contents stream."""
    n = len(stream)
    w, h, pf = hint.get("width"), hint.get("height"), hint.get("pixel_format")
    if isinstance(w, int) and isinstance(h, int) and isinstance(pf, int) and pf in PIXEL_FORMATS and w > 0 and h > 0:
        dt, spp, _ = PIXEL_FORMATS[pf]
        bpp = np.dtype(dt).itemsize * spp
        size = w * h * bpp
        pos = n - size - 28
        if pos >= 0:
            hv = struct.unpack_from("<7i", stream, pos)
            if hv[1] == w and hv[2] == h and hv[4] == bpp and hv[5] == pf:
                return pos + 28, w, h, pf, bpp
        if n - size >= 0:
            return n - size, w, h, pf, bpp  # no raw header: the pixels are simply the tail of the stream
    # scan for a plausible raw header whose pixel block ends exactly at the stream end
    for pos in range(max(0, n - 28), -1, -4):
        hv = struct.unpack_from("<7i", stream, pos)
        if hv[5] in PIXEL_FORMATS and hv[1] > 0 and hv[2] > 0:
            dt, spp, _ = PIXEL_FORMATS[hv[5]]
            bpp = np.dtype(dt).itemsize * spp
            if hv[4] == bpp and pos + 28 + hv[1] * hv[2] * bpp == n:
                return pos + 28, hv[1], hv[2], hv[5], bpp
    raise ZVIError("could not locate the pixel data of an image item")


class ZVIFile:
    """Parsed structure of a ZVI file: header, global tags and one entry per image item (plane)."""

    def __init__(self, path: str | os.PathLike, prefer_olefile: bool = True):
        self.path = Path(path)
        if not is_ole(self.path):
            raise ZVIError(f"{self.path.name} is not a ZVI (OLE compound) file")
        try:
            self.ole = OleContainer(self.path, prefer_olefile=prefer_olefile)
        except OleError as e:
            raise ZVIError(str(e)) from e
        try:
            self._parse()
        except Exception:
            self.ole.close()
            raise

    def close(self) -> None:
        self.ole.close()

    def __enter__(self) -> "ZVIFile":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _parse(self) -> None:
        streams = self.ole.listdir()
        lower = {s.lower(): s for s in streams}
        self.header = parse_image_header(self.ole.read(lower["image/contents"]), "Image/Contents") if "image/contents" in lower else {}
        self.global_tags = parse_tags(self.ole.read(lower["image/tags/contents"]), "Image/Tags/Contents") if "image/tags/contents" in lower else {}
        items: list[dict[str, Any]] = []
        for s in streams:
            m = _ITEM_RE.match(s)
            if not m:
                continue
            n = int(m.group(1))
            tag_stream = lower.get(f"image/item({n})/tags/contents")
            tags = parse_tags(self.ole.read(tag_stream), tag_stream) if tag_stream else {}
            items.append({"item": n, "stream": s, "tags": tags})
        if not items:
            raise ZVIError(f"{self.path.name}: no Image/Item(n)/Contents streams – not an AxioVision image")
        items.sort(key=lambda d: d["item"])
        # Z / C / T coordinates from the item tags; items without them are consecutive Z planes
        for order, it in enumerate(items):
            t = it["tags"]
            it["z"] = int(t[2070]) if isinstance(t.get(2070), int) else None
            it["c"] = int(t[2071]) if isinstance(t.get(2071), int) else None
            it["t"] = int(t[2072]) if isinstance(t.get(2072), int) else None
            it["order"] = order
        have_idx = all(it["z"] is not None or it["c"] is not None or it["t"] is not None for it in items)
        if not have_idx:
            for it in items:
                it["z"], it["c"], it["t"] = it["order"], 0, 0
        else:
            for it in items:
                it["z"], it["c"], it["t"] = it["z"] or 0, it["c"] or 0, it["t"] or 0
        self.items = items
        self.nz = max(it["z"] for it in items) + 1
        self.nc = max(it["c"] for it in items) + 1
        self.nt = max(it["t"] for it in items) + 1
        # first plane's geometry (all planes of a ZVI share it)
        first = self.ole.read(items[0]["stream"])
        hint = parse_image_header(first, items[0]["stream"])
        for key in ("width", "height", "pixel_format"):
            if not isinstance(hint.get(key), int) or hint[key] <= 0:
                hint[key] = self.header.get(key)
        _off, self.width, self.height, self.pixel_format, self.bpp = _plane_block(first, hint)
        if self.pixel_format not in PIXEL_FORMATS:
            raise ZVIError(f"{self.path.name}: unsupported pixel format {self.pixel_format}")
        self._item_hint = {"width": self.width, "height": self.height, "pixel_format": self.pixel_format}

    # ------------------------------------------------------------------ metadata
    def _tag(self, tag_id: int, item: dict[str, Any] | None = None) -> Any:
        if item is not None and tag_id in item["tags"]:
            return item["tags"][tag_id]
        return self.global_tags.get(tag_id)

    def channel_names(self) -> list[str]:
        names: dict[int, str] = {}
        for it in self.items:
            n = it["tags"].get(2244)
            if isinstance(n, str) and n.strip() and it["c"] not in names:
                names[it["c"]] = n.strip()
        return [names.get(c, f"C{c + 1}") for c in range(self.nc)] if names else []

    def info(self) -> NativeSeries:
        dt, spp, _bgr = PIXEL_FORMATS[self.pixel_format]
        rgb = spp >= 3
        axes = "TZCYX" + ("S" if rgb else "")
        shape = (self.nt, self.nz, self.nc, self.height, self.width) + ((3,) if rgb else ())
        info = NativeSeries(index=0, name=self.path.stem, shape=shape, axes=axes, dtype=np.dtype(dt), rgb=rgb and np.dtype(dt) == np.uint8,
                            format=FORMAT_NAME, channel_names=[] if rgb else self.channel_names())
        sx, sy, sz = (self._tag(769), self._tag(772), self._tag(775))
        if isinstance(sx, (int, float)) and sx > 0:
            unit_code = self._tag(770)
            info.pixel_width = float(sx)
            info.pixel_height = float(sy) if isinstance(sy, (int, float)) and sy > 0 else float(sx)
            info.unit = _SCALE_UNITS.get(unit_code if isinstance(unit_code, int) else 72, "µm")
            if isinstance(sz, (int, float)) and sz > 0:
                info.pixel_depth = float(sz)
        meta: dict[str, Any] = {"zvi_pixel_format": self.pixel_format, "zvi_valid_bits": self.header.get("valid_bits"), "zvi_planes": len(self.items),
                                "ole_backend": self.ole.backend}
        for tid, v in sorted(self.global_tags.items()):
            if isinstance(v, (bool, int, float, str)):
                meta[TAG_NAMES.get(tid, f"Tag {tid}")] = v
        first = self.items[0]["tags"]
        for tid in (1537, 2245, 2246, 2247, 2049, 2050, 2258, 2259, 2260):
            if tid in first and isinstance(first[tid], (bool, int, float, str)):
                meta.setdefault(TAG_NAMES[tid], first[tid])
        info.metadata = {k: v for k, v in meta.items() if v is not None}
        return info

    # ------------------------------------------------------------------ pixels
    def read_plane(self, item: dict[str, Any]) -> np.ndarray:
        stream = self.ole.read(item["stream"])
        off, w, h, pf, bpp = _plane_block(stream, self._item_hint)
        dt, spp, bgr = PIXEL_FORMATS[pf]
        size = w * h * bpp
        buf = np.frombuffer(stream, np.dtype("<" + dt), count=w * h * spp, offset=off)
        if len(stream) - off < size:
            raise ZVIError(f"{item['stream']}: truncated plane")
        if spp == 1:
            return buf.reshape(h, w)
        px = buf.reshape(h, w, spp)
        return np.ascontiguousarray(px[:, :, [2, 1, 0]] if bgr else px[:, :, :3])

    def read(self) -> tuple[np.ndarray, NativeSeries]:
        info = self.info()
        data = np.zeros(info.shape, dtype=info.dtype)
        for it in self.items:
            plane = self.read_plane(it)
            if plane.shape[:2] != (self.height, self.width):
                raise ZVIError(f"{it['stream']}: plane size {plane.shape[1]}×{plane.shape[0]} differs from the image's {self.width}×{self.height}")
            data[it["t"], it["z"], it["c"]] = plane
        return data, info


def list_series(path: str | os.PathLike) -> list[NativeSeries]:
    with ZVIFile(path) as z:
        return [z.info()]


def read(path: str | os.PathLike, series: int = 0) -> tuple[np.ndarray, NativeSeries]:
    with ZVIFile(path) as z:
        return z.read()
