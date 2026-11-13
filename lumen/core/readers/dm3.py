"""Gatan DigitalMicrograph DM3 / DM4 reader (pure Python).

The file is a tag directory: a header (version 3 or 4, root length, byte order) followed by the root TagGroup.
Structural fields (counts, lengths, type codes) are always big-endian; tag *data* follows the header's byte-order
flag (1 = little-endian, the usual case).  Images live under ``ImageList/<i>/ImageData`` (``Data``, ``DataType``,
``Dimensions``, ``Calibrations``) with the image name in ``ImageList/<i>/Name`` and the microscope metadata in
``ImageList/<i>/ImageTags``.  Image 0 of a DM3 is usually the RGB thumbnail; the real data follows it.

Pixel arrays are located while parsing (offset + count) and only read when :func:`read` asks for them, so listing
the series of a multi-GB DM4 touches only the tag tree.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from . import NativeSeries

FORMAT_NAMES = {3: "Gatan DM3", 4: "Gatan DM4"}

# tag data type codes -> struct format (data byte order applied at read time)
_SIMPLE = {2: "h", 3: "i", 4: "H", 5: "I", 6: "f", 7: "d", 8: "?", 9: "b", 10: "B", 11: "q", 12: "Q"}
_TYPE_STRUCT, _TYPE_STRING, _TYPE_ARRAY = 15, 18, 20
#: ImageData/DataType -> numpy dtype (None = packed complex / unsupported)
DATA_TYPES: dict[int, Any] = {
    1: "i2", 2: "f4", 3: "c8", 5: None, 6: "u1", 7: "i4", 8: [("b", "u1"), ("g", "u1"), ("r", "u1"), ("a", "u1")], 9: "i1",
    10: "u2", 11: "u4", 12: "f8", 13: "c16", 14: "?", 23: [("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")], 27: None, 28: None,
}
_RGB_TYPES = {8: "bgra", 23: "rgba"}
#: how many bytes of a tag array are read eagerly; larger arrays stay on disk as :class:`LazyArray`
_EAGER_BYTES = 65536


class DMError(ValueError):
    pass


class LazyArray:
    """A tag array left on disk: ``offset`` into the file, ``count`` elements of ``dtype``."""
    __slots__ = ("offset", "count", "dtype")

    def __init__(self, offset: int, count: int, dtype: np.dtype):
        self.offset, self.count, self.dtype = offset, count, dtype

    @property
    def nbytes(self) -> int:
        return self.count * self.dtype.itemsize

    def load(self, f: BinaryIO) -> np.ndarray:
        f.seek(self.offset)
        raw = f.read(self.nbytes)
        if len(raw) < self.nbytes:
            raise DMError("truncated DM file: pixel data ends before the dimensions say it should")
        return np.frombuffer(raw, self.dtype)

    def __repr__(self) -> str:
        return f"LazyArray(offset={self.offset}, count={self.count}, dtype={self.dtype})"


class _Parser:
    def __init__(self, f: BinaryIO, path: Path):
        self.f, self.path = f, path
        head = f.read(4)
        if len(head) < 4:
            raise DMError(f"{path.name}: not a DM3/DM4 file")
        self.version = struct.unpack(">i", head)[0]
        if self.version not in (3, 4):
            raise DMError(f"{path.name}: not a DM3/DM4 file (version {self.version})")
        self.root_length = self._long()
        byte_order = struct.unpack(">i", f.read(4))[0]
        self.endian = "<" if byte_order == 1 else ">"
        self.tags = self._group()

    # ------------------------------------------------------------------ primitives (always big-endian)
    def _long(self) -> int:
        if self.version == 4:
            return struct.unpack(">q", self.f.read(8))[0]
        return struct.unpack(">i", self.f.read(4))[0]

    def _group(self) -> dict[str, Any]:
        f = self.f
        f.read(2)  # is_sorted, is_open
        n = self._long()
        if n < 0 or n > 10_000_000:
            raise DMError(f"{self.path.name}: corrupt tag group (count {n})")
        out: dict[str, Any] = {}
        for i in range(n):
            tb = f.read(1)
            if not tb:
                break
            tag_type = tb[0]
            if tag_type == 0:
                break  # end marker (trailing padding in some writers)
            name_len = struct.unpack(">H", f.read(2))[0]
            name = f.read(name_len).decode("latin-1") if name_len else str(i)
            if self.version == 4:
                f.read(8)  # total bytes of this entry (not needed: we parse it)
            if tag_type == 21:
                out[name] = self._data()
            elif tag_type == 20:
                out[name] = self._group()
            else:
                raise DMError(f"{self.path.name}: unknown tag entry type {tag_type} at {f.tell()}")
        return out

    def _data(self) -> Any:
        f = self.f
        if f.read(4) != b"%%%%":
            raise DMError(f"{self.path.name}: tag data without %%%% delimiter at {f.tell()}")
        n_info = self._long()
        if n_info < 1 or n_info > 1_000_000:
            raise DMError(f"{self.path.name}: corrupt tag info array")
        info = [self._long() for _ in range(n_info)]
        return self._typed(info)

    def _typed(self, info: list[int]) -> Any:
        f = self.f
        t = info[0]
        if t in _SIMPLE:
            fmt = self.endian + _SIMPLE[t]
            size = struct.calcsize(fmt)
            return struct.unpack(fmt, f.read(size))[0]
        if t == _TYPE_STRING:
            n = info[1] if len(info) > 1 else 0
            return f.read(n).decode("latin-1", "replace")
        if t == _TYPE_STRUCT:
            dtype = self._struct_dtype(info, 1)[0]
            raw = f.read(dtype.itemsize)
            return tuple(np.frombuffer(raw, dtype)[0])
        if t == _TYPE_ARRAY:
            elem = info[1]
            if elem == _TYPE_STRUCT:
                dtype, _end = self._struct_dtype(info, 2)
                count = info[-1]
            elif elem in _SIMPLE:
                dtype = np.dtype(self.endian + _SIMPLE[elem])
                count = info[2]
            else:
                raise DMError(f"{self.path.name}: unsupported array element type {elem}")
            if count < 0:
                raise DMError(f"{self.path.name}: negative array length")
            nbytes = count * dtype.itemsize
            if nbytes > _EAGER_BYTES:
                lazy = LazyArray(f.tell(), count, dtype)
                f.seek(nbytes, 1)
                return lazy
            raw = f.read(nbytes)
            if len(raw) < nbytes:
                raise DMError(f"{self.path.name}: truncated tag array")
            return np.frombuffer(raw, dtype)
        raise DMError(f"{self.path.name}: unsupported tag data type {t}")

    def _struct_dtype(self, info: list[int], start: int) -> tuple[np.dtype, int]:
        """info[start] = struct name length (0), info[start+1] = n fields, then (name length, type) pairs."""
        n_fields = info[start + 1]
        fields = []
        pos = start + 2
        for k in range(n_fields):
            ftype = info[pos + 1]
            if ftype not in _SIMPLE:
                raise DMError(f"{self.path.name}: unsupported struct field type {ftype}")
            fields.append((f"f{k}", self.endian + _SIMPLE[ftype]))
            pos += 2
        return np.dtype(fields), pos


# ---------------------------------------------------------------------------- helpers
def _as_str(v: Any) -> str | None:
    """DM stores strings as arrays of uint16 (UTF-16LE) – or as type-18 strings."""
    if isinstance(v, str):
        return v
    if isinstance(v, np.ndarray) and v.dtype.kind in "uU" and v.dtype.itemsize == 2 and v.ndim == 1 and v.size <= 4096:
        try:
            return v.astype("<u2").tobytes().decode("utf-16-le", "replace").rstrip("\x00")
        except Exception:
            return None
    return None


def _scalar(v: Any) -> Any:
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.generic):
        return v.item()
    s = _as_str(v)
    if s is not None:
        return s
    if isinstance(v, tuple) and len(v) <= 8:
        return ", ".join(str(_scalar(x)) for x in v)
    return None


def flatten_tags(group: dict[str, Any], prefix: str = "", out: dict[str, Any] | None = None, limit: int = 2000) -> dict[str, Any]:
    """``{"Microscope Info.Voltage": 300000.0, …}`` – scalars and strings only, nested groups joined with dots."""
    out = {} if out is None else out
    for k, v in group.items():
        if len(out) >= limit:
            break
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            flatten_tags(v, key + ".", out, limit)
        else:
            s = _scalar(v)
            if s is not None:
                out[key] = s
    return out


def _dims(image: dict[str, Any]) -> list[int]:
    d = image.get("ImageData", {}).get("Dimensions", {})
    if not isinstance(d, dict):
        return []
    out = []
    for k in sorted(d, key=lambda s: int(s) if s.isdigit() else 1 << 30):
        v = d[k]
        if isinstance(v, (int, np.integer)):
            out.append(int(v))
    return out


def _calibration(image: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-dimension (X, Y, Z …) ``{"scale", "origin", "unit"}`` from ImageData/Calibrations/Dimension."""
    cal = image.get("ImageData", {}).get("Calibrations", {})
    dims = cal.get("Dimension", {}) if isinstance(cal, dict) else {}
    out = []
    if isinstance(dims, dict):
        for k in sorted(dims, key=lambda s: int(s) if s.isdigit() else 1 << 30):
            d = dims[k]
            if not isinstance(d, dict):
                continue
            out.append({"scale": float(d.get("Scale", 1.0) or 1.0), "origin": float(d.get("Origin", 0.0) or 0.0),
                        "unit": (_as_str(d.get("Units")) or "").strip()})
    return out


def _unit_name(u: str) -> str:
    u = u.replace("\xb5", "µ").replace("μ", "µ")
    return {"um": "µm", "micron": "µm", "microns": "µm", "A": "Å"}.get(u, u)


def _is_thumbnail(image: dict[str, Any], dims: list[int], dtype_code: int, n_images: int) -> bool:
    return n_images > 1 and dtype_code in _RGB_TYPES and len(dims) == 2 and max(dims) <= 512


class DMFile:
    """The parsed tag tree of a DM3 / DM4 file plus the list of (non-thumbnail) images."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            p = _Parser(f, self.path)
        self.version, self.endian, self.tags = p.version, p.endian, p.tags
        self.images: list[dict[str, Any]] = []
        image_list = self.tags.get("ImageList", {})
        entries = [image_list[k] for k in sorted(image_list, key=lambda s: int(s) if s.isdigit() else 1 << 30)] if isinstance(image_list, dict) else []
        entries = [e for e in entries if isinstance(e, dict) and isinstance(e.get("ImageData"), dict)]
        for raw_index, img in enumerate(entries):
            dims = _dims(img)
            dtype_code = int(img["ImageData"].get("DataType", 0) or 0)
            thumb = _is_thumbnail(img, dims, dtype_code, len(entries))
            self.images.append({"raw_index": raw_index, "tags": img, "dims": dims, "dtype_code": dtype_code, "thumbnail": thumb})
        main = [i for i in self.images if not i["thumbnail"]]
        self.series = main or self.images

    def series_info(self, index: int) -> NativeSeries:
        entry = self.series[index]
        img = entry["tags"]
        dims = entry["dims"]
        code = entry["dtype_code"]
        dt = DATA_TYPES.get(code)
        if dt is None:
            raise DMError(f"{self.path.name}: unsupported DM data type {code} (packed complex)")
        rgb = code in _RGB_TYPES
        dtype = np.dtype("u1") if rgb else np.dtype(dt)
        if dtype.kind == "c":
            dtype = np.dtype("f4") if dtype.itemsize == 8 else np.dtype("f8")  # modulus of complex data
        nd = len(dims)
        if nd < 2:
            raise DMError(f"{self.path.name}: image {index} has {nd} dimension(s)")
        # DM stores X first (fastest), then Y, then the stack / series axes
        axes = {2: "YX", 3: "ZYX", 4: "TZYX"}.get(nd)
        if axes is None:
            raise DMError(f"{self.path.name}: {nd}-D images are not supported")
        shape = tuple(reversed(dims))
        if rgb:
            axes += "S"
            shape = shape + (3,)
        info = NativeSeries(index=index, name=_as_str(img.get("Name")) or f"{self.path.stem} image {entry['raw_index']}", shape=shape, axes=axes,
                            dtype=dtype, rgb=rgb, format=FORMAT_NAMES[self.version])
        cal = _calibration(img)
        if len(cal) >= 2 and cal[0]["scale"] > 0:
            ux, uy = _unit_name(cal[0]["unit"]), _unit_name(cal[1]["unit"])
            if ux and ux == uy:
                info.pixel_width, info.pixel_height, info.unit = cal[0]["scale"], cal[1]["scale"] or cal[0]["scale"], ux
                if len(cal) >= 3 and nd >= 3 and _unit_name(cal[2]["unit"]) == ux and cal[2]["scale"] > 0:
                    info.pixel_depth = cal[2]["scale"]
        meta: dict[str, Any] = {"dm_version": self.version, "dm_data_type": code, "dm_image_index": entry["raw_index"], "series_count": len(self.series)}
        for i, c in enumerate(cal):
            ax = "XYZT"[i] if i < 4 else str(i)
            meta[f"calibration_{ax}_scale"], meta[f"calibration_{ax}_origin"], meta[f"calibration_{ax}_unit"] = c["scale"], c["origin"], _unit_name(c["unit"])
        bright = img.get("ImageData", {}).get("Calibrations", {}).get("Brightness") if isinstance(img.get("ImageData", {}).get("Calibrations"), dict) else None
        if isinstance(bright, dict) and _as_str(bright.get("Units")):
            meta["intensity_unit"] = _as_str(bright.get("Units"))
        if isinstance(img.get("ImageTags"), dict):
            flatten_tags(img["ImageTags"], "", meta)
        info.metadata = meta
        return info

    def read(self, index: int) -> tuple[np.ndarray, NativeSeries]:
        info = self.series_info(index)
        entry = self.series[index]
        code = entry["dtype_code"]
        data_tag = entry["tags"]["ImageData"].get("Data")
        count = int(np.prod(entry["dims"]))
        if isinstance(data_tag, LazyArray):
            with open(self.path, "rb") as f:
                arr = data_tag.load(f)
        elif isinstance(data_tag, np.ndarray):
            arr = data_tag
        else:
            raise DMError(f"{self.path.name}: image {index} has no pixel data")
        raw = arr.view(np.uint8) if arr.dtype.fields else arr
        dt = DATA_TYPES[code]
        if code in _RGB_TYPES:
            px = raw.view(np.uint8).reshape(-1, 4)[:count]
            if px.shape[0] < count:
                raise DMError(f"{self.path.name}: pixel data shorter than the dimensions")
            rgb = px[:, [2, 1, 0]] if _RGB_TYPES[code] == "bgra" else px[:, :3]
            data = np.ascontiguousarray(rgb).reshape(info.shape)
        else:
            dtype = np.dtype(dt).newbyteorder(self.endian)
            vals = raw.view(np.uint8)[:count * dtype.itemsize].view(dtype)
            if vals.size < count:
                raise DMError(f"{self.path.name}: pixel data shorter than the dimensions")
            vals = vals[:count]
            if dtype.kind == "c":
                vals = np.abs(vals).astype(info.dtype)
            data = vals.reshape(info.shape)
        return data, info


def list_series(path: str | os.PathLike) -> list[NativeSeries]:
    dm = DMFile(path)
    return [dm.series_info(i) for i in range(len(dm.series))]


def read(path: str | os.PathLike, series: int = 0) -> tuple[np.ndarray, NativeSeries]:
    dm = DMFile(path)
    if not dm.series:
        raise DMError(f"{Path(path).name}: no images in the ImageList")
    index = min(max(int(series or 0), 0), len(dm.series) - 1)
    return dm.read(index)
