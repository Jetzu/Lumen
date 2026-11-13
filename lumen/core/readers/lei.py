"""Leica LEI reader (Leica Confocal Software / LCS: a ``.lei`` header plus one TIFF per plane).

The ``.lei`` file starts with a byte-order marker (``IIII`` little-endian, ``MMMM`` big-endian), 8 reserved bytes
and the address of the first series record.  A series record is a TIFF-IFD-like list of (block id, block offset)
pairs ended by id 0 and followed by the address of the next record (0 = last).  A block has 12 header bytes, a
4-byte size and the data.  Blocks: 10 SERIES, 15 IMAGES (plane count, width, height, bits, samples, then the
plane file names), 20 DIMDESCR (voxel type, bytes per pixel, the dimension list with sizes and physical lengths,
series name / description), 40 TIMEINFO (time stamps), 70 LUTDESC (channel LUT names), 80 CHANDESC …  Strings
are a 4-byte character count followed by UTF-16LE text.  Dimension ids are the first two UTF-16 characters of the
dimension name: 120 ``x``, 121 ``y``, 122 ``z``, 116 ``t``, 6815843 ``ch``, 6357100 ``la`` (lambda) …

The pixels come from the sibling TIFF files (``<series>_z000_ch00.tif`` …) the IMAGES block lists; they are looked
up next to the ``.lei`` (case-insensitively) and in a sub-folder named after the series.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

import numpy as np

from . import NativeSeries

FORMAT_NAME = "Leica LEI"
BLOCK_SERIES, BLOCK_IMAGES, BLOCK_DIMDESCR, BLOCK_FILTERSET, BLOCK_TIMEINFO, BLOCK_SCANNERSET = 10, 15, 20, 30, 40, 50
BLOCK_EXPERIMENT, BLOCK_LUTDESC, BLOCK_CHANDESC, BLOCK_SEQUENTIALSET = 60, 70, 80, 90
DIM_X, DIM_Y, DIM_Z, DIM_T, DIM_CH, DIM_LAMBDA = 120, 121, 122, 116, 6815843, 6357100
_DIM_AXIS = {DIM_X: "X", DIM_Y: "Y", DIM_Z: "Z", DIM_T: "T", DIM_CH: "C", DIM_LAMBDA: "C"}
_IMAGE_SUFFIXES = (".tif", ".tiff")
_UNIT_TO_UM = {"m": 1e6, "mm": 1e3, "um": 1.0, "µm": 1.0, "μm": 1.0, "nm": 1e-3}


class LEIError(ValueError):
    pass


class _Reader:
    def __init__(self, data: bytes, order: str, name: str):
        self.data, self.order, self.pos, self.name = data, order, 0, name

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def int32(self) -> int:
        if self.pos + 4 > len(self.data):
            raise LEIError(f"{self.name}: truncated block")
        v = struct.unpack_from(self.order + "i", self.data, self.pos)[0]
        self.pos += 4
        return v

    def double(self) -> float:
        if self.pos + 8 > len(self.data):
            raise LEIError(f"{self.name}: truncated block")
        v = struct.unpack_from(self.order + "d", self.data, self.pos)[0]
        self.pos += 8
        return v

    def skip(self, n: int) -> None:
        self.pos += n

    def string(self) -> str:
        """4-byte character count + UTF-16 text."""
        n = self.int32()
        if n < 0 or self.pos + 2 * n > len(self.data):
            raise LEIError(f"{self.name}: truncated string")
        raw = self.data[self.pos:self.pos + 2 * n]
        self.pos += 2 * n
        enc = "utf-16-le" if self.order == "<" else "utf-16-be"
        return raw.decode(enc, "replace").rstrip("\x00")


def _plausible_dim(dim: dict[str, Any]) -> bool:
    return 0 < dim["size"] <= 1_000_000 and (dim["id"] in _DIM_AXIS or 0 < dim["id"] < 1 << 30)


def _parse_dimdescr(block: bytes, order: str) -> dict[str, Any]:
    """Voxel type, bytes per pixel, the dimension list and the series name / description."""
    head = _Reader(block, order, "DIMDESCR")
    out: dict[str, Any] = {"voxel_version": head.int32(), "voxel_type": head.int32(), "bytes_per_pixel": head.int32(), "resolution": head.int32(),
                           "dims": []}
    base = head.pos
    for n_strings in (3, 2, 4, 1, 0):
        r = _Reader(block, order, "DIMDESCR")
        r.pos = base
        try:
            strings = [r.string() for _ in range(n_strings)]
            n_dims = r.int32()
            if not 1 <= n_dims <= 12:
                continue
            dims = []
            for _ in range(n_dims):
                d = {"id": r.int32(), "size": r.int32(), "distance": r.int32(), "length": r.string(), "origin": r.string()}
                if not _plausible_dim(d):
                    raise LEIError("implausible dimension")
                dims.append(d)
            out["dims"] = dims
            if len(strings) >= 2:
                out["max_intensity"], out["min_intensity"] = strings[0], strings[1]
            try:
                out["series_name"] = r.string()
                out["description"] = r.string()
            except LEIError:
                pass
            break
        except LEIError:
            continue
    return out


def _parse_images(block: bytes, order: str) -> dict[str, Any]:
    r = _Reader(block, order, "IMAGES")
    out: dict[str, Any] = {"count": r.int32(), "width": r.int32(), "height": r.int32(), "bits": r.int32(), "samples": r.int32(), "files": []}
    files: list[str] = []
    while r.remaining() >= 4 and len(files) < max(out["count"], 0):
        try:
            s = r.string()
        except LEIError:
            break
        if s and ("." in Path(s.replace("\\", "/")).name):
            files.append(s)
    out["files"] = files
    return out


def _parse_lutdesc(block: bytes, order: str) -> list[dict[str, Any]]:
    r = _Reader(block, order, "LUTDESC")
    out = []
    try:
        n = r.int32()
        for _ in range(max(0, min(n, 64))):
            version = r.int32()
            inverted = bool(block[r.pos]) if r.pos < len(block) else False
            r.skip(1)
            desc, fname, lut = r.string(), r.string(), r.string()
            r.skip(8)
            out.append({"version": version, "inverted": inverted, "description": desc, "file": fname, "lut": lut})
    except LEIError:
        pass
    return out


def _parse_timeinfo(block: bytes, order: str) -> list[str]:
    r = _Reader(block, order, "TIMEINFO")
    stamps: list[str] = []
    try:
        n_dims = r.int32()
        r.int32()
        for _ in range(max(0, min(n_dims, 16))):
            r.int32(), r.int32(), r.int32()
        n = r.int32()
        enc = "utf-16-le" if order == "<" else "utf-16-be"
        for _ in range(max(0, min(n, 100000))):
            if r.remaining() < 64:
                break
            stamps.append(block[r.pos:r.pos + 64].decode(enc, "replace").rstrip("\x00"))
            r.skip(64)
    except LEIError:
        pass
    return stamps


def _parse_series_block(block: bytes, order: str) -> dict[str, Any]:
    r = _Reader(block, order, "SERIES")
    out: dict[str, Any] = {}
    try:
        out["version"], out["series_count"], out["name_length"], ext_len = r.int32(), r.int32(), r.int32(), r.int32()
        enc = "utf-16-le" if order == "<" else "utf-16-be"
        if 0 <= ext_len <= r.remaining():
            out["extension"] = block[r.pos:r.pos + ext_len].decode(enc, "replace").rstrip("\x00")
    except LEIError:
        pass
    return out


def _physical(length: str, size: int) -> tuple[float | None, str]:
    """'0.000375 m' over ``size`` elements -> (per-element value, unit) with metres / millimetres turned into µm."""
    parts = length.strip().split()
    if not parts:
        return None, ""
    try:
        total = float(parts[0].replace(",", "."))
    except ValueError:
        return None, ""
    unit = parts[1] if len(parts) > 1 else ""
    per = total / max(size, 1)
    if unit in _UNIT_TO_UM:
        return per * _UNIT_TO_UM[unit], "µm"
    return per, unit


def _timestamp_seconds(stamp: str) -> float | None:
    """'2007:06:21,14:32:07:531' -> seconds since midnight of that day (good enough for an interval)."""
    try:
        date, clock = stamp.split(",", 1)
        h, m, s, ms = (clock.split(":") + ["0"] * 4)[:4]
        y, mo, d = (date.split(":") + ["1", "1"])[:3]
        import datetime as _dt
        t = _dt.datetime(int(y), int(mo), int(d), int(h), int(m), int(s), int(ms) * 1000)
        return t.timestamp()
    except Exception:
        return None


class LEIFile:
    """The parsed ``.lei`` header: one entry per series with its dimensions, calibration and plane files."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            raw = f.read()
        if len(raw) < 16:
            raise LEIError(f"{self.path.name}: not a Leica LEI file")
        marker = raw[:4]
        if marker == b"IIII":
            self.order = "<"
        elif marker == b"MMMM":
            self.order = ">"
        else:
            raise LEIError(f"{self.path.name}: not a Leica LEI file (no IIII / MMMM marker)")
        self.raw = raw
        self.series: list[dict[str, Any]] = []
        addr = struct.unpack_from(self.order + "i", raw, 12)[0]
        seen: set[int] = set()
        while addr > 0 and addr + 8 <= len(raw) and addr not in seen and len(self.series) < 10000:
            seen.add(addr)
            blocks: dict[int, bytes] = {}
            pos = addr + 4
            while pos + 8 <= len(raw):
                tag, offset = struct.unpack_from(self.order + "ii", raw, pos)
                pos += 8
                if tag == 0:
                    pos -= 4
                    break
                if 0 <= offset and offset + 16 <= len(raw):
                    size = struct.unpack_from(self.order + "i", raw, offset + 12)[0]
                    if 0 <= size <= len(raw) - offset - 16:
                        blocks[tag] = raw[offset + 16:offset + 16 + size]
            self.series.append(self._series(blocks, len(self.series)))
            addr = struct.unpack_from(self.order + "i", raw, pos)[0] if pos + 4 <= len(raw) else 0
        if not self.series:
            raise LEIError(f"{self.path.name}: no series records")

    def _series(self, blocks: dict[int, bytes], index: int) -> dict[str, Any]:
        s: dict[str, Any] = {"index": index, "blocks": sorted(blocks)}
        s["images"] = _parse_images(blocks[BLOCK_IMAGES], self.order) if BLOCK_IMAGES in blocks else {"count": 0, "width": 0, "height": 0, "files": []}
        s["dimdescr"] = _parse_dimdescr(blocks[BLOCK_DIMDESCR], self.order) if BLOCK_DIMDESCR in blocks else {"dims": []}
        s["luts"] = _parse_lutdesc(blocks[BLOCK_LUTDESC], self.order) if BLOCK_LUTDESC in blocks else []
        s["timestamps"] = _parse_timeinfo(blocks[BLOCK_TIMEINFO], self.order) if BLOCK_TIMEINFO in blocks else []
        s["series_block"] = _parse_series_block(blocks[BLOCK_SERIES], self.order) if BLOCK_SERIES in blocks else {}
        return s

    # ------------------------------------------------------------------ geometry
    def info(self, index: int) -> NativeSeries:
        s = self.series[index]
        dd, im = s["dimdescr"], s["images"]
        sizes = {"X": im.get("width") or 0, "Y": im.get("height") or 0, "Z": 1, "C": 1, "T": 1}
        extra_axes: list[str] = []
        physical: dict[str, tuple[float | None, str]] = {}
        meta: dict[str, Any] = {"lei_series_index": index, "series_count": len(self.series), "plane_files": len(im.get("files", [])),
                                "bytes_per_pixel": dd.get("bytes_per_pixel"), "voxel_type": dd.get("voxel_type")}
        for k in ("series_name", "description", "max_intensity", "min_intensity"):
            if dd.get(k):
                meta[k] = dd[k]
        for d in dd.get("dims", []):
            ax = _DIM_AXIS.get(d["id"])
            if ax is None:
                ax = "T" if "T" not in extra_axes and sizes["T"] == 1 else "Z"
                meta[f"dimension_{d['id']}"] = d["size"]
            if ax in ("X", "Y"):
                sizes[ax] = d["size"] or sizes[ax]
            else:
                sizes[ax] = sizes[ax] * d["size"] if ax in extra_axes else d["size"]
                if ax not in extra_axes:
                    extra_axes.append(ax)
            physical[ax] = _physical(d["length"], d["size"])
            meta[f"dimension_{ax}_length"] = d["length"]
            if d.get("origin"):
                meta[f"dimension_{ax}_origin"] = d["origin"]
        n_files = len(im.get("files", []))
        if extra_axes:
            expected = sizes["Z"] * sizes["C"] * sizes["T"]
            if n_files and expected != n_files:
                meta["dimension_mismatch"] = f"{expected} planes declared, {n_files} files listed"
        elif n_files > 1:
            sizes["Z"] = n_files
            extra_axes = ["Z"]
        channel_names = [lut["lut"] or lut["description"] for lut in s["luts"]] if s["luts"] else []
        info = NativeSeries(index=index, name=dd.get("series_name") or f"{self.path.stem} series {index + 1}",
                            shape=(sizes["T"], sizes["Z"], sizes["C"], sizes["Y"], sizes["X"]), axes="TZCYX", dtype=np.uint8,
                            format=FORMAT_NAME, channel_names=channel_names[:sizes["C"]] if len(channel_names) >= sizes["C"] else [])
        bpp = dd.get("bytes_per_pixel") or 1
        info.dtype = {1: np.uint8, 2: np.uint16, 3: np.uint8, 4: np.float32}.get(int(bpp), np.uint8)
        if bpp == 3:
            info.rgb = True
        px, py, pz = physical.get("X"), physical.get("Y"), physical.get("Z")
        if px and px[0]:
            info.pixel_width, info.unit = px[0], px[1] or "µm"
            info.pixel_height = py[0] if py and py[0] else px[0]
            if pz and pz[0]:
                info.pixel_depth = pz[0]
        pt = physical.get("T")
        if pt and pt[0] and sizes["T"] > 1:
            info.frame_interval, info.time_unit = pt[0], {"s": "sec", "sec": "sec", "ms": "ms", "min": "min", "h": "h"}.get(pt[1], pt[1] or "sec")
        elif sizes["T"] > 1 and len(s["timestamps"]) >= 2:
            t0, t1 = _timestamp_seconds(s["timestamps"][0]), _timestamp_seconds(s["timestamps"][-1])
            if t0 is not None and t1 is not None and t1 > t0:
                info.frame_interval, info.time_unit = (t1 - t0) / (len(s["timestamps"]) - 1), "sec"
        if s["timestamps"]:
            meta["first_timestamp"] = s["timestamps"][0]
        for lut in s["luts"]:
            if lut["description"]:
                meta[f"LUT {lut['lut'] or '?'} description"] = lut["description"]
        meta["dimension_order"] = "XY" + "".join(extra_axes) + "".join(a for a in "CZT" if a not in extra_axes)
        info.metadata = meta
        s["_extra_axes"] = extra_axes
        return info

    # ------------------------------------------------------------------ plane files
    def plane_paths(self, index: int) -> list[Path]:
        s = self.series[index]
        files = s["images"].get("files", [])
        folder = self.path.parent
        listing: dict[str, Path] | None = None
        out: list[Path] = []
        missing: list[str] = []
        series_name = s["dimdescr"].get("series_name") or ""
        for entry in files:
            name = Path(entry.replace("\\", "/")).name
            candidates = [folder / name]
            if series_name:
                candidates.append(folder / series_name / name)
            found = next((c for c in candidates if c.is_file()), None)
            if found is None:
                if listing is None:
                    listing = {}
                    for sub in [folder] + [p for p in folder.iterdir() if p.is_dir()] if folder.is_dir() else []:
                        for f in sub.iterdir() if sub.is_dir() else []:
                            if f.is_file():
                                listing.setdefault(f.name.lower(), f)
                found = listing.get(name.lower())
            if found is None:
                missing.append(name)
            else:
                out.append(found)
        if missing:
            raise FileNotFoundError(f"{self.path.name}: {len(missing)} of {len(files)} plane TIFF(s) are missing next to the .lei "
                                    f"(first: {missing[0]}) – copy the whole Leica folder, not only the .lei")
        return out

    def read(self, index: int) -> tuple[np.ndarray, NativeSeries]:
        import tifffile
        info = self.info(index)
        paths = self.plane_paths(index)
        if not paths:
            raise LEIError(f"{self.path.name}: series {index} lists no image files")
        planes: list[np.ndarray] = []
        for p in paths:
            with tifffile.TiffFile(str(p)) as tf:
                if len(paths) == 1 and len(tf.pages) > 1:
                    planes.extend(np.asarray(page.asarray()) for page in tf.pages)
                else:
                    planes.append(np.asarray(tf.pages[0].asarray()))
        nt, nz, nc = info.shape[0], info.shape[1], info.shape[2]
        first = planes[0]
        rgb = first.ndim == 3 and first.shape[-1] in (3, 4) and first.dtype == np.uint8
        h, w = first.shape[:2]
        n_planes = nt * nz * nc
        if len(planes) != n_planes:
            # trust the files: fold everything the header did not describe into Z
            nz = len(planes) // max(nt * nc, 1) if len(planes) % max(nt * nc, 1) == 0 else len(planes)
            if nz * nt * nc != len(planes):
                nt, nc, nz = 1, 1, len(planes)
            n_planes = len(planes)
        dtype = first.dtype
        data = np.zeros((nt, nz, nc, h, w) + ((3,) if rgb else ()), dtype)
        extra = self.series[index].get("_extra_axes") or ["Z"]
        sizes = {"Z": nz, "C": nc, "T": nt}
        for i, plane in enumerate(planes[:n_planes]):
            if plane.shape[:2] != (h, w):
                raise LEIError(f"{paths[min(i, len(paths) - 1)].name}: plane size {plane.shape[1]}×{plane.shape[0]} differs from {w}×{h}")
            coords, rem = {"Z": 0, "C": 0, "T": 0}, i
            for ax in extra:  # first listed extra dimension varies fastest
                coords[ax] = rem % sizes[ax]
                rem //= sizes[ax]
            data[coords["T"], coords["Z"], coords["C"]] = plane[:, :, :3] if rgb else plane
        info.shape = data.shape
        info.axes = "TZCYX" + ("S" if rgb else "")
        info.dtype = dtype
        info.rgb = rgb
        info.metadata["plane_files_read"] = len(paths)
        return data, info


def list_series(path: str | os.PathLike) -> list[NativeSeries]:
    lei = LEIFile(path)
    return [lei.info(i) for i in range(len(lei.series))]


def read(path: str | os.PathLike, series: int = 0) -> tuple[np.ndarray, NativeSeries]:
    lei = LEIFile(path)
    index = min(max(int(series or 0), 0), len(lei.series) - 1)
    return lei.read(index)
