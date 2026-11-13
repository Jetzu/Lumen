"""Native readers for formats that otherwise need Java Bio-Formats (Gatan DM3/DM4, Zeiss ZVI, Leica LEI), the mocked
Java Bio-Formats bridge and its place in the reader fallback chain, and the Plugins Configuration rows.

The synthetic files are written by small in-test writers that follow the format descriptions in the reader modules
(tag directory for DM, an OLE2 compound file with VARIANT streams for ZVI, the IFD-like header for LEI)."""
from __future__ import annotations

import struct
import os
import sys
import types
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest
import tifffile

from lumen.core import ops  # noqa: F401  (registers the commands)
from lumen.core import io
from lumen.core.readers import _ole, bioformats_bridge, dm3, lei, zvi
from lumen.core.session import Session


# ============================================================================ Gatan DM3 / DM4 writer
_DM_CODES = {"i2": 2, "i4": 3, "u2": 4, "u4": 5, "f4": 6, "f8": 7, "i1": 9, "u1": 10, "i8": 11, "u8": 12}
_DM_DATA_TYPES = {"i2": 1, "f4": 2, "u1": 6, "i4": 7, "i1": 9, "u2": 10, "u4": 11, "f8": 12}


def _dm_long(v: int, version: int) -> bytes:
    return struct.pack(">q", v) if version == 4 else struct.pack(">i", v)


def _dm_data(value, version: int, end: str) -> bytes:
    if isinstance(value, bool):
        info, payload = [8], struct.pack("?", value)
    elif isinstance(value, int):
        info, payload = [3], struct.pack(end + "i", value)
    elif isinstance(value, float):
        info, payload = [7], struct.pack(end + "d", value)
    elif isinstance(value, str):
        a = np.frombuffer(value.encode("utf-16-le"), "<u2").astype(end + "u2")
        info, payload = [20, 4, a.size], a.tobytes()
    elif isinstance(value, np.ndarray):
        if value.dtype.fields:
            names = value.dtype.names
            info = [20, 15, 0, len(names)]
            for n in names:
                info += [0, _DM_CODES[value.dtype[n].str[1:]]]
            info.append(value.size)
            payload = value.astype(np.dtype([(n, end + value.dtype[n].str[1:]) for n in names])).tobytes()
        else:
            info, payload = [20, _DM_CODES[value.dtype.str[1:]], value.size], np.ascontiguousarray(value).astype(end + value.dtype.str[1:]).tobytes()
    else:
        raise TypeError(type(value))
    return b"%%%%" + _dm_long(len(info), version) + b"".join(_dm_long(i, version) for i in info) + payload


def _dm_group(d: dict, version: int, end: str) -> bytes:
    body = b"\x00\x00" + _dm_long(len(d), version)
    for name, value in d.items():
        is_group = isinstance(value, dict)
        data = _dm_group(value, version, end) if is_group else _dm_data(value, version, end)
        label = b"" if name.isdigit() else name.encode("latin-1")
        body += bytes([20 if is_group else 21]) + struct.pack(">H", len(label)) + label + (struct.pack(">q", len(data)) if version == 4 else b"") + data
    return body


def dm_image(data: np.ndarray, name: str, scale=(1.0, 1.0, 1.0), units=("nm", "nm", "nm"), data_type: int | None = None, tags: dict | None = None) -> dict:
    """An ImageList entry: ImageData (Calibrations, Data, DataType, Dimensions), ImageTags, Name."""
    nd = data.ndim if data_type not in (8, 23) else data.ndim - 1
    dims = {str(i): int(n) for i, n in enumerate(reversed(data.shape[:nd]))}  # X first
    cal = {"Brightness": {"Origin": 0.0, "Scale": 1.0, "Units": ""},
           "Dimension": {str(i): {"Origin": 0.0, "Scale": float(scale[i]), "Units": units[i]} for i in range(nd)},
           "DisplayCalibratedUnits": False}
    code = data_type if data_type is not None else _DM_DATA_TYPES[data.dtype.str[1:]]
    flat = data.reshape(-1) if data_type not in (8, 23) else data.reshape(-1)
    return {"ImageData": {"Calibrations": cal, "Data": flat, "DataType": code, "Dimensions": dims, "PixelDepth": int(data.dtype.itemsize)},
            "ImageTags": tags or {}, "Name": name, "UniqueID": 1}


def write_dm(path: Path, images: list[dict], version: int = 3, little_endian: bool = True) -> Path:
    end = "<" if little_endian else ">"
    root = {"ApplicationBounds": np.array([0, 0, 1000, 1500], "<i4"), "DocumentObjectList": {"0": {"ImageSource": 0}}, "ImageList": {str(i): im for i, im in enumerate(images)},
            "InImageMode": True, "Version": 3}
    body = _dm_group(root, version, end)
    header = struct.pack(">i", version) + _dm_long(len(body), version) + struct.pack(">i", 1 if little_endian else 0)
    path.write_bytes(header + body + b"\x00" * 8)
    return path


# ============================================================================ OLE2 / ZVI writer
_FREESECT, _ENDOFCHAIN, _FATSECT, _NOSTREAM = 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD, 0xFFFFFFFF


def _dir_entry(name: str, kind: int, left: int, right: int, child: int, start: int, size: int) -> bytes:
    raw = (name + "\x00").encode("utf-16-le") if name else b""
    return (raw.ljust(64, b"\x00") + struct.pack("<HBB", len(raw), kind, 1) + struct.pack("<III", left, right, child) + b"\x00" * 16 + b"\x00" * 4
            + b"\x00" * 16 + struct.pack("<I", start) + struct.pack("<Q", size))


def write_ole(path: Path, streams: dict[str, bytes]) -> Path:
    """A version-3 compound file (512-byte sectors, mini stream for streams < 4096 bytes, DIFAT in the header)."""
    sector, mini, cutoff = 512, 64, 4096
    # directory tree: storages for every path prefix, children chained through the right-sibling pointer
    nodes: dict[str, dict] = {"": {"name": "Root Entry", "kind": 5, "children": [], "data": None}}
    for full in sorted(streams):
        parts = full.split("/")
        for i in range(1, len(parts)):
            key = "/".join(parts[:i])
            if key not in nodes:
                nodes[key] = {"name": parts[i - 1], "kind": 1, "children": [], "data": None}
                nodes["/".join(parts[:i - 1])]["children"].append(key)
        nodes[full] = {"name": parts[-1], "kind": 2, "children": [], "data": streams[full]}
        nodes["/".join(parts[:-1])]["children"].append(full)
    order = list(nodes)  # root first
    sid = {k: i for i, k in enumerate(order)}
    # mini stream for small streams
    ministream = bytearray()
    mini_start: dict[str, int] = {}
    minifat: list[int] = []
    for key in order:
        n = nodes[key]
        if n["kind"] == 2 and 0 < len(n["data"]) < cutoff:
            n_sect = -(-len(n["data"]) // mini)
            start = len(minifat)
            mini_start[key] = start
            minifat += [start + i + 1 for i in range(n_sect - 1)] + [_ENDOFCHAIN]
            ministream += n["data"].ljust(n_sect * mini, b"\x00")
    big = [(key, nodes[key]["data"]) for key in order if nodes[key]["kind"] == 2 and len(nodes[key]["data"]) >= cutoff]
    minifat_bytes = struct.pack("<%dI" % len(minifat), *minifat) if minifat else b""
    n_dir = -(-len(order) * 128 // sector)
    n_minifat = -(-len(minifat_bytes) // sector)
    n_ministream = -(-len(ministream) // sector)
    n_big = sum(-(-len(d) // sector) for _, d in big)
    n_fat = 1
    while True:
        total = n_fat + n_dir + n_minifat + n_ministream + n_big
        need = -(-total // (sector // 4))
        if need <= n_fat:
            break
        n_fat = need
    fat = [_FREESECT] * (n_fat * (sector // 4))
    for i in range(n_fat):
        fat[i] = _FATSECT
    pos = n_fat

    def chain(n: int) -> int:
        nonlocal pos
        start = pos
        for i in range(n):
            fat[pos + i] = pos + i + 1 if i < n - 1 else _ENDOFCHAIN
        pos += n
        return start if n else _ENDOFCHAIN

    dir_start = chain(n_dir)
    minifat_start = chain(n_minifat)
    ministream_start = chain(n_ministream)
    big_start = {key: chain(-(-len(d) // sector)) for key, d in big}
    entries = []
    for key in order:
        n = nodes[key]
        kids = n["children"]
        child = sid[kids[0]] if kids else _NOSTREAM
        if n["kind"] == 5:
            entries.append(_dir_entry(n["name"], 5, _NOSTREAM, _NOSTREAM, child, ministream_start if ministream else _ENDOFCHAIN, len(ministream)))
        elif n["kind"] == 1:
            entries.append(_dir_entry(n["name"], 1, _NOSTREAM, _NOSTREAM, child, 0, 0))
        else:
            start = mini_start.get(key, big_start.get(key, _ENDOFCHAIN))
            entries.append(_dir_entry(n["name"], 2, _NOSTREAM, _NOSTREAM, _NOSTREAM, start, len(n["data"])))
    # right-sibling chains
    for key in order:
        kids = nodes[key]["children"]
        for a, b in zip(kids, kids[1:]):
            e = bytearray(entries[sid[a]])
            struct.pack_into("<I", e, 72, sid[b])
            entries[sid[a]] = bytes(e)
    directory = b"".join(entries).ljust(n_dir * sector, b"\x00")
    header = (_ole.MAGIC + b"\x00" * 16 + struct.pack("<HHHHH", 0x3E, 3, 0xFFFE, 9, 6) + b"\x00" * 6 + struct.pack("<I", 0)
              + struct.pack("<IIIIIIII", n_fat, dir_start, 0, cutoff, minifat_start, n_minifat, _ENDOFCHAIN, 0))
    difat = list(range(n_fat)) + [_FREESECT] * (109 - n_fat)
    header += struct.pack("<109I", *difat)
    body = struct.pack("<%dI" % len(fat), *fat) + directory + minifat_bytes.ljust(n_minifat * sector, b"\x00") + bytes(ministream).ljust(n_ministream * sector, b"\x00")
    for _, d in big:
        body += d.ljust(-(-len(d) // sector) * sector, b"\x00")
    path.write_bytes(header + body)
    return path


def _vt_i4(v: int) -> bytes:
    return struct.pack("<Hi", 3, v)


def _vt_r8(v: float) -> bytes:
    return struct.pack("<Hd", 5, v)


def _vt_bstr(s: str) -> bytes:
    b = s.encode("utf-16-le")
    return struct.pack("<Hi", 8, len(b)) + b


def _vt_blob(b: bytes) -> bytes:
    return struct.pack("<Hi", 65, len(b)) + b


def _vt_stored(s: str) -> bytes:
    b = s.encode("utf-16-le")
    return struct.pack("<Hi", 69, len(b)) + b


def _zvi_header(w: int, h: int, pf: int, count: int, bits: int) -> bytes:
    return (_vt_i4(0x20000001) + _vt_i4(0) + _vt_bstr("Zeiss AxioVision image") + _vt_bstr("") + _vt_i4(w) + _vt_i4(h) + _vt_i4(1) + _vt_i4(pf)
            + _vt_i4(count) + _vt_i4(bits) + _vt_blob(b"\x00" * 16) + _vt_blob(b"") + _vt_stored("Layers") + _vt_stored("Scaling") + _vt_blob(b"\x01\x02"))


def _zvi_tags(tags: list[tuple[int, bytes]]) -> bytes:
    return _vt_i4(1) + _vt_i4(len(tags)) + b"".join(value + _vt_i4(tid) + _vt_i4(0) for tid, value in tags)


def write_zvi(path: Path, planes: dict[tuple[int, int, int], np.ndarray], pixel_format: int, scale=(0.645, 0.645, 1.5), unit_code: int = 72,
              channel_names: dict[int, str] | None = None, index_tags: bool = True, raw_header: bool = True) -> Path:
    first = next(iter(planes.values()))
    h, w = first.shape[:2]
    dt, spp, _bgr = zvi.PIXEL_FORMATS[pixel_format]
    bpp = np.dtype(dt).itemsize * spp
    bits = 8 * np.dtype(dt).itemsize
    streams: dict[str, bytes] = {"Image/Contents": _zvi_header(w, h, pixel_format, len(planes), bits),
                                 "Image/Tags/Contents": _zvi_tags([(769, _vt_r8(scale[0])), (770, _vt_i4(unit_code)), (772, _vt_r8(scale[1])), (773, _vt_i4(unit_code)),
                                                                   (775, _vt_r8(scale[2])), (776, _vt_i4(unit_code)), (1025, _vt_bstr("2009-03-04 10:11:12")),
                                                                   (2050, _vt_bstr("Plan-Apochromat 20x/0.8"))]),
                                 "SummaryInformation": b"\xfe\xff\x00\x00" + b"\x00" * 40}
    for n, ((t, z, c), plane) in enumerate(planes.items()):
        arr = np.ascontiguousarray(plane.astype("<" + dt))
        body = _zvi_header(w, h, pixel_format, 1, bits)
        if raw_header:
            body += struct.pack("<7i", 1, w, h, 1, bpp, pixel_format, bits)
        streams[f"Image/Item({n})/Contents"] = body + arr.tobytes()
        tags: list[tuple[int, bytes]] = [(2245, _vt_r8(12.5 + n))]
        if index_tags:
            tags += [(2070, _vt_i4(z)), (2071, _vt_i4(c)), (2072, _vt_i4(t))]
        if channel_names and c in channel_names:
            tags.append((2244, _vt_bstr(channel_names[c])))
        streams[f"Image/Item({n})/Tags/Contents"] = _zvi_tags(tags)
    return write_ole(path, streams)


# ============================================================================ Leica LEI writer
def _lei_str(s: str) -> bytes:
    return struct.pack("<i", len(s)) + s.encode("utf-16-le")


def _lei_block(data: bytes) -> bytes:
    return b"\x00" * 12 + struct.pack("<i", len(data)) + data


def write_lei(path: Path, series: list[dict]) -> Path:
    """``series``: dicts with name, width, height, bpp, dims [(id, size, length, origin)], files, luts, timestamps."""
    out = bytearray(b"IIII" + b"\x00" * 8 + struct.pack("<i", 0))
    ifd_addrs: list[int] = []
    for s in series:
        blocks: dict[int, bytes] = {}
        blocks[lei.BLOCK_SERIES] = struct.pack("<iiii", 0, len(series), 64, 8) + ".tif".encode("utf-16-le")
        images = struct.pack("<iiiii", len(s["files"]), s["width"], s["height"], 8 * s.get("bpp", 1), 1)
        images += b"".join(_lei_str("C:\\Leica\\Experiment\\" + f) for f in s["files"])
        blocks[lei.BLOCK_IMAGES] = images
        dd = struct.pack("<iiii", 1, 20, s.get("bpp", 1), 0) + _lei_str("255") + _lei_str("0") + _lei_str("") + struct.pack("<i", len(s["dims"]))
        for dim_id, size, length, origin in s["dims"]:
            dd += struct.pack("<iii", dim_id, size, 1) + _lei_str(length) + _lei_str(origin)
        dd += _lei_str(s["name"]) + _lei_str(s.get("description", "test series"))
        blocks[lei.BLOCK_DIMDESCR] = dd
        if s.get("luts"):
            lut = struct.pack("<i", len(s["luts"]))
            for name in s["luts"]:
                lut += struct.pack("<i", 1) + b"\x00" + _lei_str(f"{name} channel") + _lei_str(f"{name.lower()}.lut") + _lei_str(name) + b"\x00" * 8
            blocks[lei.BLOCK_LUTDESC] = lut
        if s.get("timestamps"):
            ti = struct.pack("<ii", 1, lei.DIM_T) + struct.pack("<iii", lei.DIM_T, len(s["timestamps"]), 1) + struct.pack("<i", len(s["timestamps"]))
            for st in s["timestamps"]:
                ti += st.encode("utf-16-le").ljust(64, b"\x00")
            blocks[lei.BLOCK_TIMEINFO] = ti
        offsets = {}
        for tag, data in blocks.items():
            offsets[tag] = len(out)
            out += _lei_block(data)
        ifd_addrs.append(len(out))
        out += struct.pack("<i", len(blocks))
        for tag in blocks:
            out += struct.pack("<ii", tag, offsets[tag])
        out += struct.pack("<i", 0) + struct.pack("<i", 0)  # end tag + next address (patched below)
    struct.pack_into("<i", out, 12, ifd_addrs[0])
    for i, addr in enumerate(ifd_addrs[:-1]):
        n_blocks = struct.unpack_from("<i", out, addr)[0]
        struct.pack_into("<i", out, addr + 4 + n_blocks * 8 + 4, ifd_addrs[i + 1])
    path.write_bytes(bytes(out))
    return path


def _lei_zc(tmp_path: Path, folder_name: str = "exp", subfolder: bool = False, nz: int = 2, nc: int = 2, h: int = 12, w: int = 15) -> tuple[Path, np.ndarray]:
    folder = tmp_path / folder_name
    folder.mkdir()
    planes = np.zeros((nz, nc, h, w), np.uint8)
    files = []
    target = folder / "Series003" if subfolder else folder
    target.mkdir(exist_ok=True)
    for c in range(nc):
        for z in range(nz):  # z varies fastest: dims listed x, y, z, ch
            planes[z, c] = 10 * z + 100 * c + np.arange(h * w, dtype=np.uint8).reshape(h, w) % 7
            name = f"Series003_z{z:03d}_ch{c:02d}.tif"
            tifffile.imwrite(target / name, planes[z, c])
            files.append(name)
    dims = [(lei.DIM_X, w, f"{w * 2.5e-6:.9f} m", "0 m"), (lei.DIM_Y, h, f"{h * 2.5e-6:.9f} m", "0 m"), (lei.DIM_Z, nz, f"{nz * 5e-6:.9f} m", "0 m"),
            (lei.DIM_CH, nc, "2", "0")]
    p = write_lei(folder / "experiment.lei", [{"name": "Series003", "width": w, "height": h, "bpp": 1, "dims": dims, "files": files, "luts": ["Green", "Red"]}])
    return p, planes


# ============================================================================ DM3 / DM4
def test_dm3_uint16_image_calibration_and_tags(tmp_path):
    rng = np.random.default_rng(0)
    data = rng.integers(0, 65535, (20, 30), dtype=np.uint16)
    tags = {"Microscope Info": {"Voltage": 300000.0, "Name": "Titan", "Indicated Magnification": 50000.0}, "Acquisition": {"Device": {"Name": "US1000"}}}
    p = write_dm(tmp_path / "a.dm3", [dm_image(data, "HAADF", scale=(0.25, 0.25), units=("nm", "nm"), tags=tags)])
    arr, info = dm3.read(p)
    assert arr.shape == (20, 30) and arr.dtype == np.uint16 and np.array_equal(arr, data)
    assert info.axes == "YX" and info.name == "HAADF" and info.format == "Gatan DM3"
    assert info.pixel_width == pytest.approx(0.25) and info.unit == "nm" and info.pixel_depth is None
    assert info.metadata["Microscope Info.Voltage"] == 300000.0 and info.metadata["Acquisition.Device.Name"] == "US1000"
    img = io.read_image(p)
    assert img.shape == (1, 1, 1, 20, 30) and img.dtype == np.uint16 and img.calibration.unit == "nm" and img.calibration.pixel_width == pytest.approx(0.25)
    assert img.metadata["format"] == "Gatan DM3" and img.metadata["Microscope Info.Name"] == "Titan" and "series_count" not in img.metadata
    infos = io.list_series(p)
    assert len(infos) == 1 and infos[0].size_x == 30 and infos[0].size_y == 20 and infos[0].pixel_type == "uint16" and infos[0].format == "Gatan DM3"
    assert infos[0].unit == "nm" and infos[0].metadata["Microscope Info.Voltage"] == 300000.0


def test_dm4_float_stack_big_endian_lazy_array(tmp_path):
    rng = np.random.default_rng(1)
    data = rng.random((3, 120, 150)).astype(np.float32)  # 216 KB > the eager limit: the Data tag stays lazy
    p = write_dm(tmp_path / "s.dm4", [dm_image(data, "stack", scale=(1.2, 1.2, 4.0), units=("µm", "µm", "µm"))], version=4, little_endian=False)
    dm = dm3.DMFile(p)
    assert dm.version == 4 and dm.endian == ">" and isinstance(dm.images[0]["tags"]["ImageData"]["Data"], dm3.LazyArray)
    arr, info = dm.read(0)
    assert arr.shape == (3, 120, 150) and info.axes == "ZYX" and np.allclose(arr, data) and info.format == "Gatan DM4"
    assert info.pixel_width == pytest.approx(1.2) and info.pixel_depth == pytest.approx(4.0) and info.unit == "µm"
    img = io.read_image(p)
    assert img.shape == (1, 3, 1, 120, 150) and img.dtype == np.float32 and np.allclose(img.data[0, :, 0], data)
    assert img.calibration.pixel_depth == pytest.approx(4.0) and img.calibration.unit == "µm"
    info = io.list_series(p)[0]
    assert (info.size_z, info.size_c, info.size_t) == (3, 1, 1) and info.pixel_depth == pytest.approx(4.0) and info.dimension_order == "XYZCT"


def test_dm3_thumbnail_skipped_and_signed_types(tmp_path):
    thumb = np.zeros((64, 64, 4), np.uint8)
    thumb[..., 0] = 200
    real = (np.arange(40 * 50, dtype=np.int16).reshape(40, 50) - 1000).astype(np.int16)
    p = write_dm(tmp_path / "t.dm3", [dm_image(thumb, "thumbnail", data_type=23), dm_image(real, "spectrum image", scale=(2.0, 2.0), units=("1/nm", "1/nm"))])
    dm = dm3.DMFile(p)
    assert [i["thumbnail"] for i in dm.images] == [True, False] and len(dm.series) == 1
    arr, info = dm3.read(p)
    assert arr.dtype == np.int16 and np.array_equal(arr, real) and info.name == "spectrum image" and info.unit == "1/nm"
    img = io.read_image(p)
    assert img.dtype == np.float32 and float(img.data.min()) == -1000.0  # signed data become 32-bit float in Lumen
    assert len(io.list_series(p)) == 1


def test_dm3_rgb_image_and_struct_arrays(tmp_path):
    rgb = np.zeros((10, 12, 4), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2], rgb[..., 3] = 10, 20, 30, 255  # B G R A for type 8
    tags = {"Colour": np.array([(1, 2.5)], dtype=[("a", "<i4"), ("b", "<f8")])}
    p = write_dm(tmp_path / "rgb.dm3", [dm_image(rgb, "colour", data_type=8, tags=tags)])
    arr, info = dm3.read(p)
    assert info.rgb and info.axes == "YXS" and arr.shape == (10, 12, 3) and tuple(arr[0, 0]) == (30, 20, 10)
    img = io.read_image(p)
    assert img.rgb and img.shape == (1, 1, 3, 10, 12) and img.data[0, 0, :, 0, 0].tolist() == [30, 20, 10]
    assert io.list_series(p)[0].rgb


def test_dm_errors(tmp_path):
    bad = tmp_path / "bad.dm3"
    bad.write_bytes(b"\x00\x00\x00\x07" + b"\x00" * 40)
    with pytest.raises(dm3.DMError, match="not a DM3/DM4"):
        dm3.read(bad)
    with pytest.raises(ValueError):
        io.read_image(bad)
    with pytest.raises(ValueError):
        io.list_series(bad)
    assert ".dm3" in io.IMAGE_EXTS and ".dm4" in io.IMAGE_EXTS and ".zvi" in io.IMAGE_EXTS and ".lei" in io.IMAGE_EXTS


# ============================================================================ OLE / ZVI
def test_builtin_ole_reader_mini_and_regular_streams(tmp_path):
    small = bytes(range(256)) * 3  # 768 bytes -> mini stream (several mini sectors)
    big = np.arange(3000, dtype="<u2").tobytes()  # 6000 bytes -> regular sectors
    streams = {"Image/Contents": small, "Image/Item(0)/Contents": big, "Image/Item(0)/Tags/Contents": b"tags", "Empty": b"", "Root/Deep/Leaf": b"x" * 5000}
    p = write_ole(tmp_path / "c.ole", streams)
    assert _ole.is_ole(p) and not _ole.is_ole(tmp_path / "nope")
    with _ole.OleContainer(p, prefer_olefile=False) as ole:
        assert ole.backend == "built-in"
        assert ole.listdir() == sorted(streams)
        for name, data in streams.items():
            assert ole.exists(name) and ole.read(name) == data, name
        assert ole.exists("image/contents") and not ole.exists("Image/Nope")
        with pytest.raises(KeyError):
            ole.read("Image/Nope")
    try:
        import olefile  # noqa: F401
    except ImportError:
        return
    with _ole.OleContainer(p) as ole:  # the same file through olefile when it is installed
        assert ole.backend == "olefile" and ole.read("Image/Item(0)/Contents") == big and ole.read("Image/Contents") == small


def test_zvi_zct_stack_uint16(tmp_path):
    rng = np.random.default_rng(2)
    planes = {}
    for t in range(2):
        for z in range(3):
            for c in range(2):
                planes[(t, z, c)] = rng.integers(0, 4000, (16, 20), dtype=np.uint16) + 10000 * c
    p = write_zvi(tmp_path / "stack.zvi", planes, pixel_format=4, channel_names={0: "DAPI", 1: "GFP"})
    arr, info = zvi.read(p)
    assert arr.shape == (2, 3, 2, 16, 20) and arr.dtype == np.uint16 and info.axes == "TZCYX" and not info.rgb
    for (t, z, c), plane in planes.items():
        assert np.array_equal(arr[t, z, c], plane)
    assert info.channel_names == ["DAPI", "GFP"] and info.pixel_width == pytest.approx(0.645) and info.pixel_depth == pytest.approx(1.5) and info.unit == "µm"
    assert info.metadata["Acquisition Date"] == "2009-03-04 10:11:12" and info.metadata["Objective Name"].startswith("Plan-Apo")
    assert info.metadata["Exposure Time [ms]"] == 12.5 and info.metadata["zvi_pixel_format"] == 4 and info.metadata["ole_backend"] in ("built-in", "olefile")
    img = io.read_image(p)
    assert img.shape == (2, 3, 2, 16, 20) and img.dtype == np.uint16 and [ch.name for ch in img.display.channels] == ["DAPI", "GFP"]
    assert img.calibration.unit == "µm" and img.calibration.pixel_width == pytest.approx(0.645) and img.calibration.pixel_depth == pytest.approx(1.5)
    assert img.metadata["format"] == "Zeiss ZVI"
    infos = io.list_series(p)
    assert len(infos) == 1 and (infos[0].size_x, infos[0].size_y, infos[0].size_z, infos[0].size_c, infos[0].size_t) == (20, 16, 3, 2, 2)
    assert infos[0].pixel_type == "uint16" and infos[0].channel_names == ["DAPI", "GFP"] and infos[0].format == "Zeiss ZVI" and infos[0].image_count == 12


def test_zvi_bgr_colour_and_untagged_planes(tmp_path):
    plane = np.zeros((8, 9, 3), np.uint8)
    plane[..., 0], plane[..., 1], plane[..., 2] = 5, 50, 200  # stored B, G, R
    p = write_zvi(tmp_path / "rgb.zvi", {(0, 0, 0): plane}, pixel_format=1)
    arr, info = zvi.read(p)
    assert info.rgb and arr.shape == (1, 1, 1, 8, 9, 3) and tuple(arr[0, 0, 0, 0, 0]) == (200, 50, 5)
    img = io.read_image(p)
    assert img.rgb and img.shape == (1, 1, 3, 8, 9) and img.data[0, 0, :, 0, 0].tolist() == [200, 50, 5]
    # planes without Z / C / T index tags (and without the raw 28-byte header) are consecutive Z sections
    planes = {(0, z, 0): np.full((6, 7), 3 * z + 1, np.uint8) for z in range(4)}
    p2 = write_zvi(tmp_path / "plain.zvi", planes, pixel_format=3, index_tags=False, raw_header=False)
    arr, info = zvi.read(p2)
    assert arr.shape == (1, 4, 1, 6, 7) and [int(arr[0, z, 0, 0, 0]) for z in range(4)] == [1, 4, 7, 10] and info.channel_names == []


def test_zvi_errors(tmp_path):
    bad = tmp_path / "bad.zvi"
    bad.write_bytes(b"not an ole file" * 50)
    with pytest.raises(zvi.ZVIError, match="not a ZVI"):
        zvi.read(bad)
    with pytest.raises(ValueError):
        io.read_image(bad)
    empty = write_ole(tmp_path / "empty.zvi", {"SummaryInformation": b"\x00" * 16})
    with pytest.raises(zvi.ZVIError, match="no Image/Item"):
        zvi.list_series(empty)


# ============================================================================ Leica LEI
def test_lei_two_channel_zstack(tmp_path):
    p, planes = _lei_zc(tmp_path)
    infos = lei.list_series(p)
    assert len(infos) == 1 and infos[0].shape == (1, 2, 2, 12, 15) and infos[0].name == "Series003" and infos[0].channel_names == ["Green", "Red"]
    assert infos[0].pixel_width == pytest.approx(2.5) and infos[0].pixel_depth == pytest.approx(5.0) and infos[0].unit == "µm"
    assert infos[0].metadata["dimension_order"] == "XYZCT" and infos[0].metadata["plane_files"] == 4
    arr, info = lei.read(p)
    assert arr.shape == (1, 2, 2, 12, 15) and arr.dtype == np.uint8
    for z in range(2):
        for c in range(2):
            assert np.array_equal(arr[0, z, c], planes[z, c])
    img = io.read_image(p)
    assert img.shape == (1, 2, 2, 12, 15) and [ch.name for ch in img.display.channels] == ["Green", "Red"]
    assert img.calibration.pixel_width == pytest.approx(2.5) and img.calibration.pixel_depth == pytest.approx(5.0) and img.calibration.unit == "µm"
    assert img.metadata["format"] == "Leica LEI" and img.metadata["series_name"] == "Series003" and img.metadata["description"] == "test series"
    si = io.list_series(p)[0]
    assert (si.size_x, si.size_y, si.size_z, si.size_c, si.size_t) == (15, 12, 2, 2, 1) and si.format == "Leica LEI" and si.channel_names == ["Green", "Red"]
    assert si.pixel_depth == pytest.approx(5.0)


def test_lei_multiple_series_timepoints_and_subfolder(tmp_path):
    folder = tmp_path / "exp"
    folder.mkdir()
    (folder / "Series001").mkdir()
    (folder / "Series002").mkdir()
    files1, files2 = [], []
    for t in range(3):
        name = f"Series001_t{t:02d}_ch00.tif"
        tifffile.imwrite(folder / "Series001" / name, np.full((5, 6), 7 * t + 1, np.uint16))
        files1.append(name)
    for z in range(2):
        name = f"Series002_z{z:03d}_ch00.tif"
        tifffile.imwrite(folder / "Series002" / name, np.full((4, 4), 20 + z, np.uint8))
        files2.append(name)
    s1 = {"name": "Series001", "width": 6, "height": 5, "bpp": 2, "files": files1,
          "dims": [(lei.DIM_X, 6, "0.000012 m", "0 m"), (lei.DIM_Y, 5, "0.000010 m", "0 m"), (lei.DIM_T, 3, "6 s", "0 s")],
          "timestamps": ["2007:06:21,14:32:07:000", "2007:06:21,14:32:09:000", "2007:06:21,14:32:11:000"]}
    s2 = {"name": "Series002", "width": 4, "height": 4, "bpp": 1, "files": files2,
          "dims": [(lei.DIM_X, 4, "0.000008 m", "0 m"), (lei.DIM_Y, 4, "0.000008 m", "0 m"), (lei.DIM_Z, 2, "0.000004 m", "0 m")]}
    p = write_lei(folder / "two.lei", [s1, s2])
    infos = io.list_series(p)
    assert [i.name for i in infos] == ["Series001", "Series002"] and [i.index for i in infos] == [0, 1]
    assert (infos[0].size_t, infos[0].size_z, infos[0].size_x) == (3, 1, 6) and infos[0].time_increment == pytest.approx(2.0) and infos[0].time_unit == "sec"
    assert infos[0].pixel_type == "uint16" and infos[1].size_z == 2 and infos[1].pixel_depth == pytest.approx(2.0)
    a = io.read_image(p, series=0)
    assert a.shape == (3, 1, 1, 5, 6) and a.dtype == np.uint16 and [int(a.data[t, 0, 0, 0, 0]) for t in range(3)] == [1, 8, 15]
    assert a.calibration.frame_interval == pytest.approx(2.0) and a.metadata["series_count"] == 2 and a.metadata["series"] == 0
    assert a.metadata["source"] == {"kind": "file", "path": str(p), "series": 0}
    b = io.read_image(p, series=1)
    assert b.shape == (1, 2, 1, 4, 4) and [int(b.data[0, z, 0, 0, 0]) for z in range(2)] == [20, 21] and b.metadata["series_name"] == "Series002"


def test_lei_missing_planes_and_bad_files(tmp_path):
    p, _planes = _lei_zc(tmp_path, "missing")
    (tmp_path / "missing" / "Series003_z001_ch01.tif").unlink()
    with pytest.raises(FileNotFoundError, match="Series003_z001_ch01.tif"):
        io.read_image(p)
    assert len(io.list_series(p)) == 1  # listing needs only the header
    bad = tmp_path / "bad.lei"
    bad.write_bytes(b"XXXX" + b"\x00" * 30)
    with pytest.raises(lei.LEIError, match="not a Leica LEI"):
        lei.read(bad)
    with pytest.raises(ValueError):
        io.read_image(bad)


def test_lei_case_insensitive_lookup(tmp_path):
    p, planes = _lei_zc(tmp_path, "case")
    for f in (tmp_path / "case").glob("*.tif"):
        f.rename(f.with_name(f.name.upper()))
    img = io.read_image(p)
    assert img.shape == (1, 2, 2, 12, 15) and np.array_equal(img.data[0, 1, 1], planes[1, 1])


# ============================================================================ Java Bio-Formats bridge (mocked)
_OME = """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
  <Image ID="Image:0" Name="slide A"><AcquisitionDate>2024-05-06T07:08:09</AcquisitionDate>
    <Pixels ID="Pixels:0" DimensionOrder="XYCZT" Type="uint16" SizeX="12" SizeY="10" SizeZ="2" SizeC="2" SizeT="1"
            PhysicalSizeX="0.5" PhysicalSizeXUnit="µm" PhysicalSizeY="0.5" PhysicalSizeYUnit="µm" PhysicalSizeZ="2" PhysicalSizeZUnit="µm">
      <Channel ID="Channel:0:0" Name="Hoechst" SamplesPerPixel="1"/><Channel ID="Channel:0:1" Name="Alexa488" SamplesPerPixel="1"/>
      <Plane TheZ="0" TheC="0" TheT="0" DeltaT="0" ExposureTime="0.1"/>
    </Pixels></Image>
  <Image ID="Image:1" Name="overview"><Pixels ID="Pixels:1" DimensionOrder="XYCZT" Type="uint8" SizeX="6" SizeY="5" SizeZ="1" SizeC="1" SizeT="3"
            PhysicalSizeX="250" PhysicalSizeXUnit="nm" PhysicalSizeY="250" PhysicalSizeYUnit="nm" TimeIncrement="1.5" TimeIncrementUnit="s">
      <Channel ID="Channel:1:0" Name="" SamplesPerPixel="1"/></Pixels></Image>
</OME>"""
_PPS = namedtuple("PhysicalPixelSizes", ["Z", "Y", "X"])


class _FakeDims:
    def __init__(self, order: str, shape: tuple[int, ...]):
        self.order, self.shape = order, shape


class _FakeOME:
    def to_xml(self) -> str:
        return _OME


class _FakeBioImage:
    """Mimics bioio.BioImage over bioio_bioformats.Reader for two scenes (the OME-XML above)."""
    opened: list[tuple[str, object]] = []
    scenes = ("slide A", "overview")
    _arrays = {0: np.arange(2 * 2 * 10 * 12, dtype=np.uint16).reshape(1, 2, 2, 10, 12),  # T C Z Y X
               1: np.arange(3 * 5 * 6, dtype=np.uint8).reshape(3, 1, 1, 5, 6)}
    _names = {0: ["Hoechst", "Alexa488"], 1: ["Channel:1:0"]}
    _pps = {0: _PPS(2.0, 0.5, 0.5), 1: _PPS(None, 0.25, 0.25)}

    def __init__(self, path, reader=None, **kw):
        if reader is None:
            raise ValueError(f"No reader plugin for {path}")
        self.path = path
        self.current_scene_index = 0
        _FakeBioImage.opened.append((path, reader))
        self.resolution_levels = (0, 1)
        self.resolution_level_dims = {0: (1, 2, 2, 10, 12), 1: (1, 2, 2, 5, 6)}
        self._level = 0

    def set_scene(self, i):
        self.current_scene_index = int(i)

    def set_resolution_level(self, lv):
        self._level = int(lv)

    @property
    def dims(self):
        return _FakeDims("TCZYX", self._arrays[self.current_scene_index].shape)

    @property
    def shape(self):
        return self._arrays[self.current_scene_index].shape

    @property
    def dtype(self):
        return self._arrays[self.current_scene_index].dtype

    @property
    def data(self):
        return self._arrays[self.current_scene_index]

    def get_image_data(self, order: str):
        arr = self._arrays[self.current_scene_index]
        if self._level:
            arr = arr[..., ::2, ::2]
        return np.transpose(arr, ["TCZYX".index(ch) for ch in order])

    @property
    def physical_pixel_sizes(self):
        return self._pps[self.current_scene_index]

    @property
    def channel_names(self):
        return self._names[self.current_scene_index]

    @property
    def ome_metadata(self):
        return _FakeOME()


@pytest.fixture
def fake_bridge(monkeypatch):
    """Inject fake ``bioio`` / ``bioio_bioformats`` modules and pretend a JDK is present; nothing Java-related runs."""
    bioio = types.ModuleType("bioio")
    bioio.BioImage = _FakeBioImage
    bf = types.ModuleType("bioio_bioformats")
    bf.Reader = type("Reader", (), {})
    monkeypatch.setitem(sys.modules, "bioio", bioio)
    monkeypatch.setitem(sys.modules, "bioio_bioformats", bf)
    monkeypatch.setattr(bioformats_bridge, "_java_available", lambda: (True, "fake java 21"))
    monkeypatch.setenv("LUMEN_BIOFORMATS_MODE", "bioio")  # the Fiji-jars mode of this machine must not stand in
    bioformats_bridge.reset()
    _FakeBioImage.opened.clear()
    yield bioformats_bridge
    bioformats_bridge.reset()


@pytest.fixture
def no_bridge(monkeypatch):
    for name in ("bioio", "bioio_bioformats", "cjdk"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(bioformats_bridge, "_importable", lambda name: False)
    monkeypatch.setenv("LUMEN_BIOFORMATS_MODE", "bioio")  # a JDK + Fiji on this machine would otherwise provide the fiji-jars mode
    bioformats_bridge.reset()
    yield bioformats_bridge
    bioformats_bridge.reset()


def test_bridge_status_without_package(no_bridge):
    st = no_bridge.status()
    assert not st["available"] and not st["package"] and "lumen-imaging[bioformats]" in st["detail"] and "JDK" in st["install"]
    assert no_bridge.available() is False
    with pytest.raises(RuntimeError, match="not available"):
        no_bridge.list_series("x.vsi")


def test_bridge_status_without_java(monkeypatch):
    monkeypatch.setattr(bioformats_bridge, "_importable", lambda name: name in ("bioio", "bioio_bioformats"))
    monkeypatch.setattr(bioformats_bridge, "_java_available", lambda: (False, "no JDK found (JAVA_HOME unset, no java on PATH)"))
    monkeypatch.setenv("LUMEN_BIOFORMATS_MODE", "bioio")
    bioformats_bridge.reset()
    try:
        st = bioformats_bridge.status()
        assert st["package"] and not st["java"] and not st["available"] and "Temurin" in st["detail"]
    finally:
        bioformats_bridge.reset()


def test_bridge_java_detection(monkeypatch, tmp_path):
    monkeypatch.setattr(bioformats_bridge.shutil, "which", lambda name: None)
    monkeypatch.setattr(bioformats_bridge, "_importable", lambda name: False)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    assert bioformats_bridge._java_available()[0] is False
    jdk = tmp_path / "jdk"
    (jdk / "bin").mkdir(parents=True)
    (jdk / "bin" / ("java.exe" if sys.platform == "win32" else "java")).write_bytes(b"")
    monkeypatch.setenv("JAVA_HOME", str(jdk))
    ok, detail = bioformats_bridge._java_available()
    assert ok and "JAVA_HOME" in detail
    monkeypatch.delenv("JAVA_HOME")
    monkeypatch.setattr(bioformats_bridge.shutil, "which", lambda name: "/usr/bin/java")
    assert bioformats_bridge._java_available() == (True, "java at /usr/bin/java")


def test_bridge_list_and_read_use_ome_metadata(fake_bridge, tmp_path):
    p = tmp_path / "scan.vsi"
    p.write_bytes(b"\x00" * 64)
    assert fake_bridge.available()
    infos = fake_bridge.list_series(p)
    assert [i.name for i in infos] == ["slide A", "overview"] and infos[0].axes == "TCZYX" and infos[0].shape == (1, 2, 2, 10, 12)
    assert infos[0].metadata["ome_xml"].startswith("<?xml") and infos[0].metadata["levels"] == [(10, 12), (5, 6)]
    data, cal, names, meta = fake_bridge.read(p, 0)
    assert data.shape == (1, 2, 2, 10, 12) and meta["axes"] == "TZCYX" and names == ["Hoechst", "Alexa488"]
    assert np.array_equal(data[0, :, :, :, :], np.transpose(_FakeBioImage._arrays[0][0], (1, 0, 2, 3)))
    assert cal.pixel_width == pytest.approx(0.5) and cal.pixel_depth == pytest.approx(2.0) and cal.unit == "µm"
    assert meta["series_count"] == 2 and meta["series_name"] == "slide A" and meta["planes"][0]["exposure_time"] == 0.1 and meta["acquisition_date"].startswith("2024")
    data, cal, names, meta = fake_bridge.read(p, 1)
    assert data.shape == (3, 1, 1, 5, 6) and cal.unit == "nm" and cal.pixel_width == pytest.approx(250) and cal.frame_interval == pytest.approx(1.5)
    assert cal.time_unit == "sec" and names == ["C1"] and meta["series"] == 1
    data, cal, _names, meta = fake_bridge.read(p, 0, level=1)
    assert data.shape == (1, 2, 2, 5, 6) and meta["level"] == 1 and cal.pixel_width == pytest.approx(1.0)


def test_io_falls_back_to_the_bridge_for_unknown_formats(fake_bridge, tmp_path):
    p = tmp_path / "plate.sld"
    p.write_bytes(b"\x00" * 64)
    infos = io.list_series(p)
    assert [i.name for i in infos] == ["slide A", "overview"] and infos[0].format == "Bio-Formats (Java bridge)"
    assert (infos[0].size_x, infos[0].size_y, infos[0].size_z, infos[0].size_c, infos[0].size_t) == (12, 10, 2, 2, 1) and infos[0].pixel_type == "uint16"
    assert infos[0].pixel_width == pytest.approx(0.5) and infos[0].unit == "µm" and infos[0].pixel_depth == pytest.approx(2.0)
    assert infos[0].channel_names == ["Hoechst", "Alexa488"] and infos[0].levels == [(10, 12), (5, 6)] and infos[0].planes[0]["exposure_time"] == 0.1
    assert infos[0].metadata["AcquisitionDate"].startswith("2024") and infos[1].time_increment == pytest.approx(1.5) and infos[1].unit == "nm"
    assert infos[0].dimension_order == "XYZCT"
    # plain bioio (no plugin) is tried first and fails, then the bridge opens the file with the Bio-Formats reader
    img = io.read_image(p)
    assert len(_FakeBioImage.opened) >= 1 and _FakeBioImage.opened[-1][1] is sys.modules["bioio_bioformats"].Reader
    assert img.shape == (1, 2, 2, 10, 12) and img.dtype == np.uint16 and [ch.name for ch in img.display.channels] == ["Hoechst", "Alexa488"]
    assert img.calibration.pixel_width == pytest.approx(0.5) and img.calibration.pixel_depth == pytest.approx(2.0) and img.metadata["format"] == "Bio-Formats (Java bridge)"
    assert img.metadata["series_count"] == 2 and img.metadata["source"] == {"kind": "file", "path": str(p), "series": 0} and "ome_xml" not in img.metadata
    b = io.read_image(p, series=1)
    assert b.shape == (3, 1, 1, 5, 6) and b.calibration.frame_interval == pytest.approx(1.5) and b.metadata["series_name"] == "overview"
    # the importer op goes through the same path
    s = Session()
    out = s.run_op("file.bioformats_series", {"path": str(p)})
    assert [r[1] for r in out["table"]["rows"]] == ["slide A", "overview"] and out["table"]["rows"][0][13] == "Bio-Formats (Java bridge)"
    out = s.run_op("file.bioformats_import", {"path": str(p), "series": "1"})
    assert s.images[out["new_images"][0]].shape == (3, 1, 1, 5, 6)


def test_io_bridge_rescues_a_failed_native_reader(fake_bridge, tmp_path):
    corrupt = tmp_path / "broken.dm3"
    corrupt.write_bytes(b"\x00\x00\x00\x03" + b"\xff" * 60)  # DM3 magic, garbage tag tree
    with pytest.raises(dm3.DMError):
        dm3.read(corrupt)
    img = io.read_image(corrupt)  # the native reader fails -> Java bridge
    assert img.metadata["format"] == "Bio-Formats (Java bridge)" and img.shape == (1, 2, 2, 10, 12)
    assert [i.name for i in io.list_series(corrupt)] == ["slide A", "overview"]
    # errors unrelated to the format are not swallowed
    with pytest.raises(FileNotFoundError):
        io.read_image(tmp_path / "nothing.dm3")
    raw = tmp_path / "x.raw"
    raw.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="raw"):
        io.read_image(raw)


def test_io_errors_name_the_bioformats_extra(no_bridge, tmp_path):
    p = tmp_path / "plate.sld"
    p.write_bytes(b"\x00" * 64)
    with pytest.raises(RuntimeError) as e:
        io.read_image(p)
    msg = str(e.value)
    assert "No reader for '.sld'" in msg and "lumen-imaging[bioformats]" in msg and "JDK" in msg and "bfconvert" in msg
    with pytest.raises(RuntimeError, match=r"lumen-imaging\[bioformats\]"):
        io.list_series(p)
    corrupt = tmp_path / "broken.dm4"
    corrupt.write_bytes(b"\x00\x00\x00\x04" + b"\xff" * 60)
    with pytest.raises(dm3.DMError):  # without a bridge the native reader's error surfaces unchanged
        io.read_image(corrupt)


# ============================================================================ Plugins Configuration rows
def test_config_table_lists_native_readers_and_bridge(no_bridge):
    out = Session().run_op("file.bioformats_config", {})
    rows = {r[0]: r for r in out["table"]["rows"]}
    for key in ("Gatan DigitalMicrograph DM3 / DM4", "Zeiss AxioVision ZVI", "Leica LEI"):
        row = next(r for name, r in rows.items() if name.startswith(key))
        assert row[3] == "enabled" and row[2].startswith("built in (lumen.core.readers.")
    bridge = next(r for name, r in rows.items() if "Java bridge" in name)
    assert bridge[2] == "Java Bio-Formats (bridge)" and bridge[3].startswith("missing (") and "lumen-imaging[bioformats]" in bridge[3]
    vsi = rows["Olympus VSI (cellSens)"]
    assert vsi[2] == "Java Bio-Formats (bridge)" and "not available" in vsi[3] and "[bioformats]" in vsi[3]
    assert not any(name.startswith(("Leica LEI (old", "Zeiss ZVI /", "Gatan DM3 / DM4")) for name in rows)
    assert "bridge not installed" in out["message"]


def test_config_table_with_bridge_available(fake_bridge):
    out = Session().run_op("file.bioformats_config", {})
    rows = {r[0]: r for r in out["table"]["rows"]}
    bridge = next(r for name, r in rows.items() if "Java bridge" in name)
    assert bridge[3].startswith("enabled (") and "fake java 21" in bridge[3]
    assert rows["Olympus VSI (cellSens)"][3] == "enabled (Java bridge)" and "bridge enabled" in out["message"]


# ============================================================================ Java Bio-Formats bridge, mode "fiji-jars" (mocked)
from lumen.core.readers import fiji_jars  # noqa: E402

#: what the Groovy runner prints for a two-series file (the sizes of _OME above; series 0 has two resolution levels)
_INFO = {"format": "CellSens VSI", "series_count": "2", "global.Objective": "20x", "global.Note": "a b",
         "series.0.name": "slide A", "series.0.sizeX": "12", "series.0.sizeY": "10", "series.0.sizeZ": "2", "series.0.sizeC": "2", "series.0.sizeT": "1",
         "series.0.dimension_order": "XYCZT", "series.0.pixel_type": "uint16", "series.0.rgb": "false", "series.0.rgb_channel_count": "1", "series.0.interleaved": "false",
         "series.0.little_endian": "true", "series.0.indexed": "false", "series.0.resolutions": "2", "series.0.level.0": "12x10", "series.0.level.1": "6x5",
         "series.0.physical_size_x": "0.5 µm", "series.0.physical_size_y": "0.5 µm", "series.0.physical_size_z": "2.0 µm", "series.0.time_increment": "",
         "series.0.channel.0": "Hoechst", "series.0.channel.1": "Alexa488",
         "series.1.name": "overview", "series.1.sizeX": "6", "series.1.sizeY": "5", "series.1.sizeZ": "1", "series.1.sizeC": "1", "series.1.sizeT": "3",
         "series.1.dimension_order": "XYCZT", "series.1.pixel_type": "uint8", "series.1.rgb": "false", "series.1.rgb_channel_count": "1", "series.1.interleaved": "false",
         "series.1.little_endian": "true", "series.1.indexed": "false", "series.1.resolutions": "1", "series.1.level.0": "6x5",
         "series.1.physical_size_x": "250.0 nm", "series.1.physical_size_y": "250.0 nm", "series.1.physical_size_z": "", "series.1.time_increment": "1.5 s", "series.1.channel.0": "",
         "done": "1"}
_FIJI = {"available": True, "detail": "Fiji's Bio-Formats jars (formats-gpl-8.1.1.jar in /fake/jars/bio-formats) with openjdk 21 through a groovy runner",
         "java": "/fake/jdk/bin/java", "java_version": "openjdk version 21", "jar": "/fake/jars/bio-formats/formats-gpl-8.1.1.jar", "runner": "groovy", "jars": "/fake/jars"}


@pytest.fixture
def fake_fiji_mode(monkeypatch):
    """No bioio, a pretend Fiji + JDK (discovery patched) and the runner replaced by a fake that answers like the real
    one – the ``convert`` step writes the series as an OME-TIFF with tifffile, as OMETiffWriter would."""
    import tifffile
    for name in ("bioio", "bioio_bioformats", "cjdk"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(bioformats_bridge, "_importable", lambda name: False)
    monkeypatch.delenv("LUMEN_BIOFORMATS_MODE", raising=False)
    monkeypatch.setattr(bioformats_bridge, "fiji_status", lambda refresh=False: dict(_FIJI))
    calls: list[tuple] = []

    def fake_runner(mode, path, *, series=0, level=0, out=None, nbytes=None):
        calls.append((mode, str(path), int(series), int(level), out, nbytes))
        info = dict(_INFO)
        info.update({"ome_xml": _OME, "seconds": 0.5, "command": ["java"]})
        if mode == "convert":
            si = min(max(int(series), 0), 1)
            lv = int(level) if si == 0 else 0
            arr = np.transpose(_FakeBioImage._arrays[si], (0, 2, 1, 3, 4))  # T C Z Y X -> T Z C Y X
            if lv:
                arr = arr[..., ::2, ::2]
            meta = {"axes": "TZCYX", "PhysicalSizeX": 0.5 * (2 ** lv), "PhysicalSizeXUnit": "µm", "PhysicalSizeY": 0.5 * (2 ** lv), "PhysicalSizeYUnit": "µm"} if si == 0 else \
                   {"axes": "TZCYX", "PhysicalSizeX": 0.25, "PhysicalSizeXUnit": "µm", "TimeIncrement": 1.5, "TimeIncrementUnit": "s"}
            if si == 0:
                meta.update({"PhysicalSizeZ": 2.0, "PhysicalSizeZUnit": "µm", "Channel": {"Name": ["Hoechst", "Alexa488"]}})
            tifffile.imwrite(str(out), arr, ome=True, metadata=meta)
            info.update({"series": str(si), "level": str(lv), "planes": str(arr.shape[0] * arr.shape[1] * arr.shape[2]), "output": str(out)})
        return info

    monkeypatch.setattr(bioformats_bridge, "run_fiji_runner", fake_runner)
    bioformats_bridge._info_cache.clear()
    bioformats_bridge.reset()
    yield calls
    bioformats_bridge._info_cache.clear()
    bioformats_bridge.reset()


def test_fiji_mode_status_and_mode_env(fake_fiji_mode, monkeypatch):
    st = bioformats_bridge.status()
    assert st["available"] and st["mode"] == "fiji-jars" and not st["package"] and st["java"] and "no Python package needed" in st["detail"] and "Fiji" in st["install"]
    assert bioformats_bridge.available()
    monkeypatch.setenv("LUMEN_BIOFORMATS_MODE", "bioio")
    bioformats_bridge.reset()
    st = bioformats_bridge.status()
    assert not st["available"] and st["mode"] is None and "lumen-imaging[bioformats]" in st["detail"] and st["fiji"]["detail"].startswith("disabled")
    monkeypatch.setenv("LUMEN_BIOFORMATS_MODE", "fiji-jars")
    bioformats_bridge.reset()
    assert bioformats_bridge.status()["mode"] == "fiji-jars"
    for v, want in (("", "auto"), ("FIJI", "fiji-jars"), ("python", "bioio"), ("x", "auto")):
        monkeypatch.setenv("LUMEN_BIOFORMATS_MODE", v)
        assert bioformats_bridge.mode_requested() == want


def test_fiji_mode_list_read_and_io(fake_fiji_mode, tmp_path):
    calls = fake_fiji_mode
    p = tmp_path / "scan.vsi"
    p.write_bytes(b"\x00" * 64)
    infos = bioformats_bridge.list_series(p)
    assert [i.name for i in infos] == ["slide A", "overview"] and infos[0].axes == "TZCYX" and infos[0].shape == (1, 2, 2, 10, 12) and infos[0].dtype == np.uint16
    assert infos[0].pixel_width == pytest.approx(0.5) and infos[0].unit == "µm" and infos[0].pixel_depth == pytest.approx(2.0) and infos[0].channel_names == ["Hoechst", "Alexa488"]
    assert infos[0].metadata["levels"] == [(10, 12), (5, 6)] and infos[0].metadata["ome_xml"].startswith("<?xml") and infos[0].metadata["bioformats_format"] == "CellSens VSI"
    assert infos[0].metadata["Objective"] == "20x" and infos[0].format == "Bio-Formats (Fiji jars)" and infos[0].metadata["series_count"] == 2
    assert infos[1].shape == (3, 1, 1, 5, 6) and infos[1].dtype == np.uint8 and infos[1].frame_interval == pytest.approx(1.5) and infos[1].time_unit == "s" and infos[1].unit == "nm"
    assert len(calls) == 1 and calls[0][0] == "info"
    bioformats_bridge.list_series(p)
    assert len(calls) == 1  # cached per file mtime
    data, cal, names, meta = bioformats_bridge.read(p, 0)
    assert calls[-1][0] == "convert" and calls[-1][2] == 0 and calls[-1][5] == 2 * 2 * 10 * 12 * 2  # the heap is sized from the listing
    assert data.shape == (1, 2, 2, 10, 12) and data.dtype == np.uint16 and meta["axes"] == "TZCYX" and names == ["Hoechst", "Alexa488"]
    assert np.array_equal(data, np.transpose(_FakeBioImage._arrays[0], (0, 2, 1, 3, 4)))
    assert cal.pixel_width == pytest.approx(0.5) and cal.pixel_depth == pytest.approx(2.0) and cal.unit == "µm"
    assert meta["format"] == "Bio-Formats (Fiji jars)" and meta["reader"] == "fiji-jars" and meta["series_count"] == 2 and meta["series_names"] == ["slide A", "overview"]
    assert meta["source"] == {"kind": "file", "path": str(p), "series": 0} and meta["bioformats_format"] == "CellSens VSI" and meta["planes"][0]["exposure_time"] == 0.1
    data, cal, names, meta = bioformats_bridge.read(p, 1)
    assert data.shape == (3, 1, 1, 5, 6) and cal.frame_interval == pytest.approx(1.5) and meta["series"] == 1 and meta["series_name"] == "overview"
    data, cal, _n, meta = bioformats_bridge.read(p, 0, level=1)
    assert data.shape == (1, 2, 2, 5, 6) and meta["level"] == 1 and cal.pixel_width == pytest.approx(1.0) and meta["source"]["level"] == 1
    # the whole chain: unknown extension -> no bioio -> the bridge in fiji-jars mode
    si = io.list_series(p)
    assert [i.name for i in si] == ["slide A", "overview"] and si[0].format == "Bio-Formats (Fiji jars)" and si[0].channel_names == ["Hoechst", "Alexa488"]
    assert (si[0].size_x, si[0].size_y, si[0].size_z, si[0].size_c, si[0].size_t) == (12, 10, 2, 2, 1) and si[0].levels == [(10, 12), (5, 6)] and si[0].pixel_width == pytest.approx(0.5)
    img = io.read_image(p)
    assert img.shape == (1, 2, 2, 10, 12) and img.dtype == np.uint16 and [c.name for c in img.display.channels] == ["Hoechst", "Alexa488"]
    assert img.metadata["format"] == "Bio-Formats (Fiji jars)" and img.metadata["bioformats_format"] == "CellSens VSI" and img.metadata["series_count"] == 2 and "ome_xml" not in img.metadata
    s = Session()
    out = s.run_op("file.bioformats_import", {"path": str(p), "series": "1"})
    assert s.images[out["new_images"][0]].shape == (3, 1, 1, 5, 6)
    cfg = Session().run_op("file.bioformats_config", {})
    bridge = next(r for r in cfg["table"]["rows"] if "Java bridge" in r[0])
    assert bridge[3].startswith("enabled (Fiji's Bio-Formats jars") and "bridge enabled" in cfg["message"]


def test_fiji_runner_invocation(monkeypatch, tmp_path):
    """run_fiji_runner builds the java command line (headless, quiet logback, the Groovy main class) and parses the output."""
    monkeypatch.setattr(bioformats_bridge, "fiji_status", lambda refresh=False: dict(_FIJI))
    monkeypatch.setattr(fiji_jars, "classpath_for", lambda comps, **kw: (["/fake/jars/groovy-4.0.23.jar", "/fake/jars/bio-formats/*", "/fake/jars/*"], []))
    calls = []

    def fake_java(args, *, cwd=None, timeout=600.0, java=None, env=None):
        calls.append([java] + [str(a) for a in args])
        xml = Path(args[-1] if args[-4] == "info" else args[-4])
        i = args.index("info") if "info" in args else args.index("convert")
        Path(args[i + 2]).write_text(_OME, encoding="utf-8")
        if fake_java.fail:
            return fiji_jars.JavaResult(command=calls[-1], returncode=1, stdout="", stderr="SLF4J: noise\nloci.formats.FormatException: bad file\n", seconds=0.1)
        return fiji_jars.JavaResult(command=calls[-1], returncode=0, stdout="format=OME-TIFF\nseries_count=1\nseries.0.sizeX=3\nglobal.a=b\ndone=1\n", stderr="", seconds=0.2)

    fake_java.fail = False
    monkeypatch.setattr(fiji_jars, "run_java", fake_java)
    f = tmp_path / "x.ics"
    f.write_bytes(b"")
    info = bioformats_bridge.run_fiji_runner("info", f)
    assert info["format"] == "OME-TIFF" and info["series.0.sizeX"] == "3" and info["global.a"] == "b" and info["ome_xml"].startswith("<?xml") and info["seconds"] == 0.2
    cmd = calls[0]
    assert cmd[0] == "/fake/jdk/bin/java" and "-Djava.awt.headless=true" in cmd and "-Dapple.awt.UIElement=true" in cmd and any(o.startswith("-Dlogback.configurationFile=") for o in cmd)
    i = cmd.index("-cp")
    assert cmd[i + 1] == os.pathsep.join(["/fake/jars/groovy-4.0.23.jar", "/fake/jars/bio-formats/*", "/fake/jars/*"]) and cmd[i + 2] == "groovy.ui.GroovyMain"
    assert cmd[i + 3].endswith("runner.groovy") and cmd[i + 4] == "info" and cmd[i + 5] == str(f.resolve()) and cmd[i + 6].endswith("meta.ome.xml") and len(cmd) == i + 7
    info = bioformats_bridge.run_fiji_runner("convert", f, series=1, level=0, out=tmp_path / "o.ome.tif", nbytes=10 ** 9)
    cmd = calls[1]
    assert cmd[cmd.index("convert") + 3:] == ["1", "0", str(tmp_path / "o.ome.tif")] and "-Xmx3373m" in cmd
    fake_java.fail = True
    with pytest.raises(RuntimeError, match="could not read x.ics.*exited with code 1") as e:
        bioformats_bridge.run_fiji_runner("info", f)
    assert "FormatException: bad file" in str(e.value) and "SLF4J" not in str(e.value)
    assert "runner.bsh" not in str(calls)
    monkeypatch.setattr(bioformats_bridge, "fiji_status", lambda refresh=False: {**_FIJI, "runner": "beanshell"})
    fake_java.fail = False
    bioformats_bridge.run_fiji_runner("info", f)
    assert calls[-1][calls[-1].index("-cp") + 2] == "bsh.Interpreter" and calls[-1][calls[-1].index("-cp") + 3].endswith("runner.bsh")


# ============================================================================ Java Bio-Formats bridge, mode "fiji-jars" – real runs (a JDK + Fiji; skipped in CI)
REAL_FIJI = bioformats_bridge.fiji_status()["available"] and not os.environ.get("LUMEN_SKIP_JVM_TESTS")
needs_fiji = pytest.mark.skipif(not REAL_FIJI, reason="needs a JDK and a Fiji installation (jars/bio-formats + groovy or bsh)")


def _write_multi_ome(p: Path) -> tuple[np.ndarray, np.ndarray]:
    import tifffile
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4000, (3, 2, 4, 20, 24)).astype(np.uint16)
    b = rng.integers(0, 255, (5, 10, 12)).astype(np.uint8)
    with tifffile.TiffWriter(str(p), ome=True) as tw:
        tw.write(a, metadata={"axes": "TZCYX", "Name": "cells", "PhysicalSizeX": 0.25, "PhysicalSizeXUnit": "µm", "PhysicalSizeY": 0.25, "PhysicalSizeYUnit": "µm",
                              "PhysicalSizeZ": 1.5, "PhysicalSizeZUnit": "µm", "TimeIncrement": 2.0, "TimeIncrementUnit": "s", "Channel": {"Name": ["DAPI", "GFP", "RFP", "Cy5"]}})
        tw.write(b, metadata={"axes": "ZYX", "Name": "overview", "PhysicalSizeX": 4.0, "PhysicalSizeXUnit": "µm"})
    return a, b


_ICS_WRITER = r"""
import loci.formats.*
import loci.formats.out.ICSWriter
def r = new ImageReader(); def meta = MetadataTools.createOMEXMLMetadata(); r.setMetadataStore(meta); r.setId(args[0]); r.setSeries(0)
def w = new ICSWriter(); w.setMetadataRetrieve(meta); w.setId(args[1])
for (int i = 0; i < r.getImageCount(); i++) w.saveBytes(i, r.openBytes(i))
w.close(); r.close(); println "done=1"
"""


def _write_ics_with_bioformats(src: Path, dst: Path) -> None:
    """An ICS file (a format tifffile cannot read) written by Bio-Formats' own ICSWriter from an OME-TIFF."""
    fs = bioformats_bridge.fiji_status()
    runner = dst.parent / "ics_writer.groovy"
    runner.write_text(_ICS_WRITER, encoding="utf-8")
    cp, missing = fiji_jars.classpath_for(["groovy", "bioformats"])
    assert not missing
    home = dst.parent / "home"
    home.mkdir(exist_ok=True)
    opts = fiji_jars.base_options(headless=True, home=home, heap="-Xmx1g") + [fiji_jars.logback_option(home)]
    res = fiji_jars.run_java(opts + ["-cp", os.pathsep.join(cp), "groovy.ui.GroovyMain", str(runner), str(src), str(dst)], cwd=home, timeout=300, java=fs["java"])
    assert res.ok and "done=1" in res.stdout, (res.stdout, res.stderr)


@needs_fiji
def test_real_fiji_jars_lists_and_reads_a_multi_series_ome_tiff(tmp_path, monkeypatch):
    import time
    monkeypatch.setenv("LUMEN_BIOFORMATS_MODE", "fiji-jars")
    bioformats_bridge.reset()
    p = tmp_path / "multi.ome.tif"
    a, b = _write_multi_ome(p)
    t0 = time.perf_counter()
    infos = bioformats_bridge.list_series(p)
    assert time.perf_counter() - t0 < 60
    assert [i.name for i in infos] == ["cells", "overview"] and infos[0].shape == (3, 2, 4, 20, 24) and infos[0].dtype == np.uint16 and infos[1].shape == (1, 5, 1, 10, 12)
    assert infos[0].pixel_width == pytest.approx(0.25) and infos[0].unit == "µm" and infos[0].pixel_depth == pytest.approx(1.5) and infos[0].frame_interval == pytest.approx(2.0)
    assert infos[0].channel_names == ["DAPI", "GFP", "RFP", "Cy5"] and infos[0].metadata["bioformats_format"] == "OME-TIFF" and infos[0].metadata["ome_xml"].lstrip().startswith("<")
    data, cal, names, meta = bioformats_bridge.read(p, 0)
    assert np.array_equal(data, a) and names == ["DAPI", "GFP", "RFP", "Cy5"] and cal.pixel_width == pytest.approx(0.25) and cal.pixel_depth == pytest.approx(1.5)
    assert cal.frame_interval == pytest.approx(2.0) and cal.unit == "µm" and meta["series_count"] == 2 and meta["series_names"] == ["cells", "overview"] and meta["format"] == "Bio-Formats (Fiji jars)"
    data, cal, _n, meta = bioformats_bridge.read(p, 1)
    assert np.array_equal(data[0, :, 0], b) and cal.pixel_width == pytest.approx(4.0) and meta["series_name"] == "overview"


@needs_fiji
def test_real_fiji_jars_reads_ics_through_io_and_the_importer(tmp_path, monkeypatch):
    import tifffile
    monkeypatch.setenv("LUMEN_BIOFORMATS_MODE", "fiji-jars")
    bioformats_bridge.reset()
    src = tmp_path / "cells.ome.tif"
    a, _b = _write_multi_ome(src)
    ics = tmp_path / "cells.ics"
    _write_ics_with_bioformats(src, ics)
    assert ics.is_file() and ics.stat().st_size > 1000
    with pytest.raises(Exception):
        tifffile.imread(str(ics))  # not a TIFF: only Bio-Formats reads it
    infos = io.list_series(ics)
    assert len(infos) == 1 and infos[0].format == "Bio-Formats (Fiji jars)" and (infos[0].size_x, infos[0].size_y, infos[0].size_z, infos[0].size_c, infos[0].size_t) == (24, 20, 2, 4, 3)
    assert infos[0].pixel_type == "uint16" and infos[0].pixel_width == pytest.approx(0.25) and infos[0].pixel_depth == pytest.approx(1.5) and infos[0].time_increment == pytest.approx(2.0)
    img = io.read_image(ics)
    assert img.shape == (3, 2, 4, 20, 24) and np.array_equal(img.data, a) and img.metadata["format"] == "Bio-Formats (Fiji jars)"
    assert img.metadata["bioformats_format"] == "Image Cytometry Standard" and img.calibration.pixel_width == pytest.approx(0.25) and img.calibration.unit == "µm"
    s = Session()
    out = s.run_op("file.bioformats_import", {"path": str(ics)})
    assert s.images[out["new_images"][0]].shape == (3, 2, 4, 20, 24)
    rows = s.run_op("file.bioformats_series", {"path": str(ics)})["table"]["rows"]
    assert rows[0][13] == "Bio-Formats (Fiji jars)"
