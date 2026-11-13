"""Bio-Formats parity: the importer dialog (series, colour mode, view, ranges, crop, split, grouping, virtual stacks,
metadata / OME-XML display), the remote importer, the exporter (compression, one file per Z / C / T, save range,
metadata embedding), the series chooser table, TIFF virtual stacks, the Stack Slicer and the plugins configuration
table.  The readers themselves live in ``lumen.core.io`` (``list_series`` / ``read_image``)."""
from __future__ import annotations

import itertools
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..image import LumenImage
from ..io import (IMAGE_EXTS, SeriesInfo, group_file_pattern, is_virtual, list_series, ome_xml_of, read_image, read_sequence,
                  strip_image_ext, write_image)
from ..luts import LUT_NAMES
from ..registry import OpResult, Param, op
from .stacks import plane_index, slice_labels

IMPORT = "File/Import"
EXPORT = "File/Export"
_COMMON = dict(kind="image", requires_image=False, scope_param=False, preview=False, roi_aware=False)
_NOIMG = dict(kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False)

COLOR_MODES = ["Default", "Composite", "Colorized", "Grayscale", "Custom"]
VIEWS = ["Hyperstack", "Standard ImageJ", "Data Browser"]
STACK_ORDERS = ["Default", "XYCZT", "XYCTZ", "XYZCT", "XYZTC", "XYTCZ", "XYTZC"]
COMPRESSIONS = ["Uncompressed", "LZW", "zlib", "JPEG", "JPEG-2000", "PackBits", "Zstd"]
MAX_SPLIT_IMAGES = 500

SERIES_COLUMNS = ["Index", "Name", "SizeX", "SizeY", "SizeZ", "SizeC", "SizeT", "Type", "Order", "Levels", "Pixel size", "Unit", "Channels", "Format"]

#: the Bio-Formats format list as Lumen covers it: (format, extensions, reader / library, pip name or None when built in)
FORMAT_TABLE: list[tuple[str, str, str, str | None]] = [
    ("TIFF / BigTIFF / ImageJ TIFF", "tif tiff", "tifffile", None),
    ("OME-TIFF (multi-series, pyramids)", "ome.tif ome.tiff", "tifffile", None),
    ("Zeiss LSM", "lsm", "tifffile", None),
    ("MetaMorph STK / TIFF", "stk tif", "tifffile", None),
    ("Micro-Manager / NDTiff", "tif", "tifffile", None),
    ("Olympus FluoView TIFF", "tif", "tifffile", None),
    ("Aperio SVS (whole slide)", "svs", "tifffile", None),
    ("Hamamatsu NDPI (whole slide)", "ndpi", "tifffile", None),
    ("Leica SCN (whole slide)", "scn", "tifffile", None),
    ("Ventana BIF (whole slide)", "bif", "tifffile", None),
    ("Philips TIFF (whole slide)", "tif", "tifffile", None),
    ("PerkinElmer QPTIFF", "qptiff", "tifffile", None),
    ("PNG / JPEG / BMP / GIF / WebP / PNM / TGA / JPEG-2000", "png jpg bmp gif webp pgm ppm pbm pnm tga jp2", "imageio", None),
    ("NumPy", "npy", "numpy", None),
    ("Nikon ND2", "nd2", "nd2", "nd2"),
    ("Zeiss CZI", "czi", "czifile", "czifile"),
    ("Leica LIF", "lif", "readlif", "readlif"),
    ("Olympus OIF / OIB", "oif oib", "oiffile", "oiffile"),
    ("DICOM (files and folders)", "dcm dicom ima", "pydicom", "pydicom"),
    ("NIfTI", "nii nii.gz", "nibabel", "nibabel"),
    ("MRC / CCP4", "mrc rec map mrcs", "mrcfile", "mrcfile"),
    ("HDF5 / Imaris IMS (resolution levels)", "h5 hdf5 ims", "h5py", "h5py"),
    ("Zarr / OME-Zarr (multiscales, bioformats2raw)", "zarr", "zarr", "zarr"),
    ("Video (AVI, MP4, MOV, MKV, WebM, WMV)", "avi mp4 mov mkv webm wmv", "imageio-ffmpeg", "imageio-ffmpeg"),
    ("FITS", "fits fit fts", "astropy", "astropy"),
    ("MetaImage", "mha mhd", "built in", None),
    ("NRRD", "nrrd nhdr", "pynrrd", "pynrrd"),
    ("Raw / text image", "raw bin txt csv", "built in", None),
    ("Image sequences (folders)", "", "built in", None),
    ("Gatan DigitalMicrograph DM3 / DM4 (stacks, calibration, tags)", "dm3 dm4", "built in (lumen.core.readers.dm3)", None),
    ("Zeiss AxioVision ZVI (Z / C / T stacks, scaling)", "zvi", "built in (lumen.core.readers.zvi; olefile when installed)", None),
    ("Leica LEI (TCS / LCS header + TIFF planes)", "lei", "built in (lumen.core.readers.lei)", None),
    ("Other formats via bioio reader plugins", "*", "bioio", "bioio"),
]

#: the Java Bio-Formats bridge row: pip name is the extra; its status comes from lumen.core.readers.bioformats_bridge
JAVA_BRIDGE_ROW = ("Every other Bio-Formats format via the Java bridge (bioio-bioformats + JDK)", "*", "Java Bio-Formats (bridge)", "lumen-imaging[bioformats]")

#: proprietary formats that only the Java bridge reads (no native Python reader) – the honest parity note
UNREADABLE_WITHOUT_JAVA = ["Olympus VSI (cellSens)", "SlideBook SLD", "Nikon NIS ND (legacy)",
                           "Andor IQ", "Hamamatsu VMS / VMU", "3i SlideBook", "Amira", "FEI / Thermo EMI / SER",
                           "Becker & Hickl SDT", "PicoQuant PTU / BIN", "Imspector MSR", "Molecular Devices MetaXpress HTD",
                           "Perkin Elmer Volocity / UltraVIEW", "Deltavision DV / R3D (partial via bioio-dv)", "Bruker MRI", "Varian FDF",
                           "Inveon", "Li-Cor L2D", "Prairie / Bruker PV", "Cellomics C01", "BD Pathway", "InCell", "Opera / Columbus", "MIAS",
                           "SPIDER", "Zeiss LMS", "ZEISS CZI with JPEG-XR (needs imagecodecs + czifile)", "TillVision", "TrestleTIFF", "Ventana TIFF (non-BIF)"]


# ============================================================================ helpers
def _expand(path: str) -> Path:
    p = Path(str(path or "").strip()).expanduser()
    if not str(p) or str(p) == ".":
        raise ValueError("Give a file path")
    return p


def parse_series_spec(spec: Any, infos: list[SeriesInfo], open_all: bool = False) -> list[int]:
    """Series selection → 0-based indices: '' / None → [0]; 'all' / '*' → every series; '0,2-4' → those indices
    (0-based, like read_image); anything else → series whose name contains the text (case-insensitive)."""
    n = len(infos)
    if open_all:
        return list(range(n))
    s = str(spec if spec is not None else "").strip()
    if not s:
        return [0] if n else []
    if s.lower() in ("all", "*"):
        return list(range(n))
    if re.fullmatch(r"[\d,\s\-]+", s):
        out: list[int] = []
        for part in re.split(r"[,\s]+", s):
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                lo, hi = int(a), int(b)
                out += list(range(lo, hi + 1))
            else:
                out.append(int(part))
        bad = [i for i in out if not 0 <= i < n]
        if bad:
            raise ValueError(f"Series {bad[0]} does not exist: the file has {n} series (0–{n - 1})")
        return list(dict.fromkeys(out))
    low = s.lower()
    hits = [i for i, info in enumerate(infos) if low in str(info.name).lower()]
    if not hits:
        raise ValueError(f"No series named {s!r}; available: " + ", ".join(f"{i}: {info.name}" for i, info in enumerate(infos)))
    return hits


def _range(begin: int, end: int, step: int, n: int) -> slice | None:
    """A 1-based begin/end/step (0 = default) → slice over ``n`` entries, or None when it is the whole axis."""
    b = max(1, int(begin or 1))
    e = int(end or 0)
    e = n if e <= 0 else min(e, n)
    st = max(1, int(step or 1))
    if b > e:
        raise ValueError(f"Range {b}-{e} is empty (the axis has {n} entries)")
    if b == 1 and e == n and st == 1:
        return None
    return slice(b - 1, e, st)


def _carry_labels(src: LumenImage, out: LumenImage, ts: list[int], zs: list[int], cs: list[int]) -> None:
    """Slice labels follow their planes: ``out`` holds the planes (t, z, c) of ``src`` for t in ts, z in zs, c in cs
    (in that nesting, i.e. ImageJ plane order).  ``LumenImage.copy`` carries the source's full list, which would be
    the wrong length / order for a sub-image, so it is replaced (or dropped when the source has no labels)."""
    if "slice_labels" not in src.metadata:
        out.metadata.pop("slice_labels", None)
        return
    labels = slice_labels(src)
    out.metadata["slice_labels"] = [labels[plane_index(src, t, z, c)] for t in ts for z in zs for c in cs]


def select_range(img: LumenImage, c: tuple[int, int, int] = (0, 0, 0), z: tuple[int, int, int] = (0, 0, 0),
                 t: tuple[int, int, int] = (0, 0, 0)) -> LumenImage:
    """Bio-Formats "specify range for each series": keep only the given C / Z / T sub-ranges (1-based begin, end, step)."""
    sl_t = _range(*t, img.nt)
    sl_z = _range(*z, img.nz)
    sl_c = None if img.rgb else _range(*c, img.nc)
    if sl_t is None and sl_z is None and sl_c is None:
        return img
    data = img.data[sl_t or slice(None), sl_z or slice(None), sl_c or slice(None)]
    out = img.copy(data=np.ascontiguousarray(data))
    out.rgb = img.rgb
    if sl_c is not None:
        keep = list(range(*sl_c.indices(img.nc)))
        out.display.channels = [type(img.display.channels[k])(**img.display.channels[k].to_dict()) for k in keep]
    out.display.t = out.display.z = out.display.c = 0
    out.sync_channels()
    out.metadata["range"] = {"c": list(c), "z": list(z), "t": list(t)}
    _carry_labels(img, out, list(range(*(sl_t or slice(None)).indices(img.nt))), list(range(*(sl_z or slice(None)).indices(img.nz))),
                  list(range(*(sl_c or slice(None)).indices(img.nc))))
    return out


def crop_image(img: LumenImage, x: int, y: int, w: int, h: int) -> LumenImage:
    """Bio-Formats "crop on import": the (x, y, w, h) window of every plane (w / h ≤ 0 = to the edge)."""
    x, y = max(0, int(x)), max(0, int(y))
    w = img.width - x if int(w) <= 0 else min(int(w), img.width - x)
    h = img.height - y if int(h) <= 0 else min(int(h), img.height - y)
    if w <= 0 or h <= 0:
        raise ValueError(f"The crop window ({x}, {y}, {w}×{h}) lies outside the {img.width}×{img.height} image")
    if (x, y, w, h) == (0, 0, img.width, img.height):
        return img
    out = img.copy(data=np.ascontiguousarray(img.data[:, :, :, y:y + h, x:x + w]))
    out.rgb = img.rgb
    out.metadata["crop"] = {"x": x, "y": y, "w": w, "h": h}
    return out


def swap_dimensions(img: LumenImage, order: str) -> LumenImage:
    """Bio-Formats "swap dimensions": re-label the stored Z, C and T axes.  ``order`` gives the new meaning of the stored
    (Z, C, T) axes, e.g. 'TCZ' = what the file calls Z is time and what it calls T is depth."""
    o = str(order or "ZCT").upper().replace("X", "").replace("Y", "")
    if sorted(o) != ["C", "T", "Z"]:
        raise ValueError(f"swap_order must be a permutation of Z, C and T (got {order!r})")
    if o == "ZCT":
        return img
    if img.rgb and o[1] != "C":
        raise ValueError("The channel axis of an RGB image cannot be swapped")
    stored_axis = {"Z": 1, "C": 2, "T": 0}  # position of the stored axes in the (T, Z, C, Y, X) array
    new_source = {o[i]: "ZCT"[i] for i in range(3)}  # new label → stored letter
    perm = [stored_axis[new_source["T"]], stored_axis[new_source["Z"]], stored_axis[new_source["C"]], 3, 4]
    out = img.copy(data=np.ascontiguousarray(np.transpose(img.data, perm)))
    out.rgb = img.rgb
    out.display.t = out.display.z = out.display.c = 0
    out.reset_display(keep_luts=False)
    out.metadata["swap_dimensions"] = o
    if "slice_labels" in img.metadata:  # the labels move with their planes
        labels = slice_labels(img)
        moved = []
        for t, z, c in ((t, z, c) for t in range(out.nt) for z in range(out.nz) for c in range(out.nc)):
            src = [0, 0, 0]
            src[perm[0]], src[perm[1]], src[perm[2]] = t, z, c
            moved.append(labels[plane_index(img, *src)])
        out.metadata["slice_labels"] = moved
    return out


def apply_color_mode(img: LumenImage, mode: str, custom_luts: str = "") -> None:
    """The importer's colour modes: Default (reader / Lumen defaults), Composite, Colorized (one coloured channel at a
    time), Grayscale, Custom (LUT names from ``custom_luts``)."""
    if img.rgb or mode == "Default":
        return
    chans = img.display.channels
    if mode == "Composite":
        img.display.mode = "composite" if img.nc > 1 else "single"
    elif mode == "Colorized":
        img.display.mode = "single"
    elif mode == "Grayscale":
        for ch in chans:
            ch.lut = "Grays"
        img.display.mode = "grayscale" if img.nc > 1 else "single"
    elif mode == "Custom":
        names = [n.strip() for n in str(custom_luts or "").split(",") if n.strip()]
        low = {n.lower(): n for n in LUT_NAMES}
        for c, ch in enumerate(chans):
            if c < len(names):
                if names[c].lower() not in low:
                    raise ValueError(f"Unknown LUT {names[c]!r} in custom_luts")
                ch.lut = low[names[c].lower()]
        img.display.mode = "composite" if img.nc > 1 else "single"


def apply_autoscale(img: LumenImage, autoscale: bool) -> None:
    """Bio-Formats "autoscale": display range = the data's min/max per channel (over the whole stack, or the current
    plane of a virtual stack); off = the full range of the pixel type (float images keep their data range)."""
    chans = img.display.channels
    virtual = is_virtual(img.data)
    d = img.display
    for c, ch in enumerate(chans):
        if autoscale:
            sub = img.data[d.t, d.z, c] if virtual else img.data[:, :, c]
            sub = np.asarray(sub)
            if sub.dtype == np.float32:
                sub = sub[np.isfinite(sub)]
            if sub.size == 0:
                continue
            lo, hi = float(sub.min()), float(sub.max())
            if hi <= lo:
                hi = lo + 1
            ch.min, ch.max = lo, hi
        elif img.dtype == np.uint8:
            ch.min, ch.max = 0.0, 255.0
        elif img.dtype == np.uint16:
            ch.min, ch.max = 0.0, 65535.0


def split_image(img: LumenImage, by_c: bool = False, by_z: bool = False, by_t: bool = False,
                limit: int | None = MAX_SPLIT_IMAGES) -> list[LumenImage]:
    """Fiji's Stack Slicer / importer "split channels / focal planes / timepoints": one image per index, named
    ``<name> - C=<c>`` / ``Z=<z>`` / ``T=<t>`` (0-based like Bio-Formats).  ``limit`` caps the number of pieces – the
    default protects the UI from opening hundreds of windows; the exporter, which only writes files, passes None.
    The pieces are numpy views of the source along T / Z (no pixel copy), so the list costs little memory."""
    by_c = by_c and not img.rgb and img.nc > 1
    by_z = by_z and img.nz > 1
    by_t = by_t and img.nt > 1
    if not (by_c or by_z or by_t):
        return [img]
    cs = range(img.nc) if by_c else [None]
    zs = range(img.nz) if by_z else [None]
    ts = range(img.nt) if by_t else [None]
    total = len(cs) * len(zs) * len(ts)
    if limit is not None and total > limit:
        raise ValueError(f"Splitting would open {total} images (limit {limit}) – split fewer dimensions")
    out = []
    base = img.name
    for c, z, t in itertools.product(cs, zs, ts):
        sl = (slice(t, t + 1) if t is not None else slice(None), slice(z, z + 1) if z is not None else slice(None),
              slice(c, c + 1) if c is not None else slice(None))
        name = base
        for key, v in (("C", c), ("Z", z), ("T", t)):
            if v is not None:
                name += f" - {key}={v}"
        piece = img.copy(name=name, data=np.ascontiguousarray(img.data[sl]))
        piece.rgb = img.rgb
        if c is not None:
            piece.display.channels = [type(img.display.channels[c])(**img.display.channels[c].to_dict())]
            piece.display.mode = "single"
        piece.display.t = piece.display.z = piece.display.c = 0
        piece.sync_channels()
        # each piece keeps only the labels of its own planes (the inherited full list would label every piece with plane 1's)
        _carry_labels(img, piece, [t] if t is not None else list(range(img.nt)), [z] if z is not None else list(range(img.nz)),
                      [c] if c is not None else list(range(img.nc)))
        out.append(piece)
    return out


def flatten_to_stack(img: LumenImage, stack_order: str = "XYCZT") -> LumenImage:
    """Bio-Formats "Standard ImageJ" view: the C / Z / T planes become one plain Z stack in ``stack_order``
    (the dimension named first after XY varies fastest; RGB images keep their three samples)."""
    order = (stack_order if stack_order and stack_order != "Default" else "XYCZT").upper()
    if sorted(order) != sorted("XYCZT"):
        raise ValueError(f"stack_order must be a permutation of XYCZT (got {stack_order!r})")
    dims = order[2:]  # fastest first
    axis_of = {"T": 0, "Z": 1, "C": 2}
    if img.rgb:
        dims = dims.replace("C", "")
    slow_to_fast = [axis_of[d] for d in reversed(dims)]
    keep = [a for a in (0, 1, 2) if a not in slow_to_fast]
    data = np.transpose(img.data, slow_to_fast + keep + [3, 4])
    n = int(np.prod([img.data.shape[a] for a in slow_to_fast])) if slow_to_fast else 1
    data = data.reshape((1, n) + tuple(img.data.shape[a] for a in keep) + img.data.shape[3:])
    if not img.rgb:
        data = data.reshape((1, n, 1) + img.data.shape[3:])
    out = img.copy(data=np.ascontiguousarray(data))
    out.rgb = img.rgb
    if not img.rgb:
        out.display.channels = [type(img.display.channels[0])(**img.display.channels[0].to_dict())]
        out.display.mode = "single"
    out.display.t = out.display.z = out.display.c = 0
    out.sync_channels()
    labels = []
    for idx in range(n):
        rem, parts = idx, {}
        for d in dims:
            size = img.data.shape[axis_of[d]]
            parts[d] = rem % size
            rem //= size
        labels.append("; ".join(f"{d.lower()}:{parts[d] + 1}/{img.data.shape[axis_of[d]]}" for d in "CZT" if d in parts))
    # the project's slice-label key (read by stacks.slice_labels, Slice Labels to Results, getInfo("slice.label"), Make
    # Substack, TIFF save …), one entry per plane of the flat stack, uncapped – Bio-Formats attaches exactly these
    # "c:1/2; z:1/3; t:1/2" strings as the slice labels of a Standard ImageJ view
    out.metadata["slice_labels"] = labels
    out.metadata["stack_order"] = order
    return out


def metadata_table(info: SeriesInfo | None, img: LumenImage, name: str = "Metadata") -> dict[str, Any]:
    rows: list[list[Any]] = list(info.metadata_rows()) if info is not None else []
    seen = {r[0] for r in rows}
    cal = img.calibration
    extra = [["Image", img.name], ["Width", img.width], ["Height", img.height], ["Slices", img.nz], ["Channels", img.nc], ["Frames", img.nt],
             ["BitDepth", img.bit_depth], ["Calibration", f"{cal.pixel_width:g}×{cal.pixel_height:g}×{cal.pixel_depth:g} {cal.unit}"]]
    for k, v in sorted(img.metadata.items()):
        if isinstance(v, (str, int, float, bool)) and k not in seen:
            extra.append([k, v])
    return {"name": name, "columns": ["Key", "Value"], "rows": rows + [r for r in extra if r[0] not in seen]}


def _export_base_name(name: str) -> str:
    """'cells.ome.tif' / 'cells.ome.zarr' / 'cells.zarr' / 'cells.tif' → 'cells' (any other known image extension is stripped too)."""
    low = name.lower()
    for suffix in (".ome.zarr", ".ome.tiff", ".ome.tif", ".zarr", ".tiff", ".tif"):
        if low.endswith(suffix):
            return name[:-len(suffix)] or "image"
    return strip_image_ext(name) or "image"


def _group_open(p: Path, pattern: str, img_hint: LumenImage | None = None) -> tuple[LumenImage, list[Path]]:
    folder, regex, axis = group_file_pattern(p, pattern or None)
    if axis is None:
        first = img_hint or read_image(p)
        axis = "Z" if first.nz == 1 else ("T" if first.nt == 1 else "C")
    from ..io import list_sequence_files
    files = list_sequence_files(folder, regex, sort="natural", exts=IMAGE_EXTS)
    if len(files) < 1:
        raise ValueError(f"No files match the pattern {regex} in {folder}")
    # read with the same extension set the check above used (read_sequence's default would drop .stk / .lsm / .nd2 … siblings)
    img = read_sequence(folder, pattern=regex, sort="natural", axis=axis, name=strip_image_ext(p.name), exts=IMAGE_EXTS)
    img.metadata["group_pattern"] = regex
    img.metadata["group_axis"] = axis
    img.metadata["used_files"] = [str(f) for f in files]
    return img, files


def import_file(path: str | os.PathLike, series: Any = "0", open_all_series: bool = False, color_mode: str = "Default", custom_luts: str = "",
                view: str = "Hyperstack", stack_order: str = "Default", autoscale: bool = True, split_channels: bool = False,
                split_focal_planes: bool = False, split_timepoints: bool = False, group_files: bool = False, group_pattern: str = "",
                virtual: bool = False, specify_range: bool = False, c_range: tuple[int, int, int] = (0, 0, 0), z_range: tuple[int, int, int] = (0, 0, 0),
                t_range: tuple[int, int, int] = (0, 0, 0), crop: bool = False, crop_box: tuple[int, int, int, int] = (0, 0, 0, 0),
                swap_order: str = "", level: int = 0, name: str | None = None) -> tuple[list[LumenImage], list[SeriesInfo], list[int], list[str]]:
    """The importer pipeline for one file: (images, series infos, chosen series indices, notes).  Used by the importer
    command, the remote importer and the macro extension ``Ext.openImagePlus``."""
    p = Path(path)
    infos = list_series(p)
    chosen = parse_series_spec(series, infos, open_all_series)
    notes: list[str] = []
    images: list[LumenImage] = []
    for i in chosen:
        info = infos[i] if i < len(infos) else None
        if group_files:
            img, _files = _group_open(p, group_pattern)
            if virtual:
                notes.append("group_files reads every file into memory (no virtual stack)")
        else:
            img = read_image(p, name=name or p.name, series=i, level=level, virtual=virtual)
            if virtual and not is_virtual(img.data):
                notes.append("virtual stacks are only available for TIFF, HDF5 / Imaris, Zarr / OME-Zarr and npy files; the image was read into memory")
        if len(infos) > 1:
            img.name = f"{strip_image_ext(name or p.name)} - {info.name if info and info.name else f'Series {i}'}"
        if swap_order:
            img = swap_dimensions(img, swap_order)
        if specify_range:
            img = select_range(img, c_range, z_range, t_range)
        if crop:
            img = crop_image(img, *crop_box)
        apply_color_mode(img, color_mode, custom_luts)
        apply_autoscale(img, autoscale)
        if view == "Standard ImageJ":
            img = flatten_to_stack(img, stack_order)
        pieces = split_image(img, split_channels, split_focal_planes, split_timepoints)
        for piece in pieces:
            piece.metadata.setdefault("series", i)
            piece.metadata.setdefault("bioformats_series_name", info.name if info else "")
        images += pieces
    return images, infos, chosen, notes


def _import_params() -> list[Param]:
    return [
        Param("series", "string", "0", help="Series to open: an index (0-based), several (0,2-4), 'all', or part of a series name – see Bio-Formats Series… for the list"),
        Param("open_all_series", "bool", False, label="Open all series"),
        Param("color_mode", "choice", "Default", label="Color mode", choices=COLOR_MODES,
              help="Composite: overlay; Colorized: one coloured channel at a time; Grayscale; Custom: LUT names from custom_luts"),
        Param("custom_luts", "string", "", label="Custom LUTs", help="Comma-separated LUT names for color_mode=Custom (e.g. Cyan, Magenta)", advanced=True),
        Param("view", "choice", "Hyperstack", label="View stack with", choices=VIEWS, help="Standard ImageJ flattens C/Z/T into one stack in stack_order; Data Browser = Hyperstack"),
        Param("stack_order", "choice", "Default", label="Stack order", choices=STACK_ORDERS, help="Plane order of the flat stack (Standard ImageJ view); Default = XYCZT"),
        Param("autoscale", "bool", True, help="Display range from the data (per channel) instead of the full pixel-type range"),
        Param("split_channels", "bool", False, label="Split channels"),
        Param("split_focal_planes", "bool", False, label="Split focal planes"),
        Param("split_timepoints", "bool", False, label="Split timepoints"),
        Param("group_files", "bool", False, label="Group files with similar names", help="Open the numbered siblings of the file (img_t001.tif, img_t002.tif …) as one stack"),
        Param("group_pattern", "string", "", label="File pattern", help="Bio-Formats pattern (img_t<1-30>.tif), regular expression or glob; empty = derived from the file name", advanced=True),
        Param("virtual", "bool", False, label="Use virtual stack", help="TIFF, HDF5 / Imaris, Zarr / OME-Zarr and npy: keep the pixels in the file and read planes on demand (see TIFF Virtual Stack…)"),
        Param("specify_range", "bool", False, label="Specify range for each series"),
        Param("c_begin", "int", 0, min=0, label="C begin", advanced=True, help="1-based; 0 = first"), Param("c_end", "int", 0, min=0, label="C end", advanced=True, help="0 = last"),
        Param("c_step", "int", 1, min=1, label="C step", advanced=True),
        Param("z_begin", "int", 0, min=0, label="Z begin", advanced=True), Param("z_end", "int", 0, min=0, label="Z end", advanced=True), Param("z_step", "int", 1, min=1, label="Z step", advanced=True),
        Param("t_begin", "int", 0, min=0, label="T begin", advanced=True), Param("t_end", "int", 0, min=0, label="T end", advanced=True), Param("t_step", "int", 1, min=1, label="T step", advanced=True),
        Param("crop", "bool", False, label="Crop on import"),
        Param("crop_x", "int", 0, min=0, label="Crop X", advanced=True), Param("crop_y", "int", 0, min=0, label="Crop Y", advanced=True),
        Param("crop_width", "int", 0, min=0, label="Crop width", advanced=True, help="0 = to the right edge"), Param("crop_height", "int", 0, min=0, label="Crop height", advanced=True, help="0 = to the bottom edge"),
        Param("display_metadata", "bool", False, label="Display metadata", help="Open the series' metadata as a 'Metadata' table (Key / Value)"),
        Param("display_ome_xml", "bool", False, label="Display OME-XML metadata", help="The file's OME-XML (synthesised for other formats) → 'OME-XML' table and the Log"),
        Param("swap_dimensions", "bool", False, label="Swap dimensions"),
        Param("swap_order", "string", "ZCT", label="Dimension meaning", advanced=True, help="New meaning of the stored Z, C, T axes, e.g. TCZ = the file's Z is time and its T is depth"),
        Param("level", "int", 0, min=0, label="Resolution level", advanced=True, help="Pyramid level of whole-slide / OME pyramids / Imaris / OME-Zarr files (0 = full resolution)"),
    ]


def _run_import(ctx, p: Path, params: dict[str, Any], name: str | None = None, source_label: str | None = None) -> OpResult:
    images, infos, chosen, notes = import_file(
        p, series=params["series"], open_all_series=params["open_all_series"], color_mode=params["color_mode"], custom_luts=params["custom_luts"],
        view=params["view"], stack_order=params["stack_order"], autoscale=params["autoscale"], split_channels=params["split_channels"],
        split_focal_planes=params["split_focal_planes"], split_timepoints=params["split_timepoints"], group_files=params["group_files"],
        group_pattern=params["group_pattern"], virtual=params["virtual"], specify_range=params["specify_range"],
        c_range=(params["c_begin"], params["c_end"], params["c_step"]), z_range=(params["z_begin"], params["z_end"], params["z_step"]),
        t_range=(params["t_begin"], params["t_end"], params["t_step"]), crop=params["crop"],
        crop_box=(params["crop_x"], params["crop_y"], params["crop_width"], params["crop_height"]),
        swap_order=params["swap_order"] if params["swap_dimensions"] else "", level=params["level"], name=name)
    s = ctx.session
    table = None
    logs: list[str] = []
    if params["display_ome_xml"]:
        xml = ome_xml_of(p, infos)
        lines = xml.splitlines()
        s.tables["OME-XML"] = {"name": "OME-XML", "columns": ["OME-XML"], "rows": [[ln] for ln in lines]}
        logs.append(xml)
    if params["display_metadata"]:
        first = chosen[0] if chosen else 0
        table = metadata_table(infos[first] if first < len(infos) else None, images[0] if images else LumenImage(np.zeros((1, 1), np.uint8)))
    label = source_label or p.name
    what = ", ".join(f"{im.name} {im.shape[1:]}" for im in images[:4]) + (" …" if len(images) > 4 else "")
    msg = f"{label}: opened {len(images)} image(s) from {len(chosen)} of {len(infos)} series – {what}"
    if len(infos) > 1:
        logs.append(f"{label} has {len(infos)} series: " + "; ".join(f"{i}: {info.name} ({info.size_x}×{info.size_y}, Z={info.size_z} C={info.size_c} T={info.size_t})" for i, info in enumerate(infos)))
    if any(is_virtual(im.data) for im in images):
        msg += " (virtual)"
    if notes:
        msg += "; " + "; ".join(dict.fromkeys(notes))
    return OpResult(new_images=images, message=msg, log="\n".join(logs) if logs else None, table=table)


# ============================================================================ commands
@op("file.bioformats_series", "Bio-Formats Series...", IMPORT, **_NOIMG, keywords=["bio-formats", "series", "scenes", "positions", "pyramid", "levels", "chooser"],
    description="List the image series of a file (OME-TIFF images, whole-slide pyramid series, ND2 positions, CZI scenes, LIF images, HDF5 datasets, "
                "OME-Zarr multiscales …) with their sizes, pixel type, dimension order, resolution levels, calibration and channel names – "
                "the Bio-Formats series chooser as a table; use the Index (or Name) with Bio-Formats… / Open.",
    params=[Param("path", "string", "", help="Image file (or Zarr / DICOM folder)")])
def bioformats_series(ctx, path=""):
    p = _expand(path)
    infos = list_series(p)
    rows = [info.to_row() for info in infos]
    msg = f"{p.name}: {len(infos)} series" + ("" if len(infos) == 1 else "; open one with Bio-Formats… series=<index> or series=all")
    return OpResult(table={"name": "Bio-Formats Series", "columns": SERIES_COLUMNS, "rows": rows}, message=msg,
                    log="\n".join(f"{info.index}: {info.name} – {info.size_x}×{info.size_y}, Z={info.size_z} C={info.size_c} T={info.size_t}, {info.pixel_type}"
                                  + (f", {info.level_count} levels" if info.level_count > 1 else "") for info in infos))


@op("file.bioformats_import", "Bio-Formats...", IMPORT, **_COMMON,
    keywords=["bio-formats", "importer", "series", "scene", "position", "nd2", "czi", "lif", "ome", "svs", "ndpi", "virtual", "split", "crop", "range", "metadata"],
    description="Open a file the way Fiji's Bio-Formats Importer does: choose the series (or all of them), the colour mode, hyperstack or flat stack view "
                "and stack order, autoscaling, splitting by channel / focal plane / timepoint, grouping of numbered sibling files, a TIFF virtual stack, "
                "a C/Z/T sub-range, a crop window, dimension swapping, the resolution level, and show the metadata / OME-XML. "
                "Native readers cover TIFF/OME-TIFF (incl. pyramids: SVS, NDPI, SCN, BIF, Philips), LSM, STK, ND2, CZI, LIF, OIF/OIB, DICOM, NIfTI, "
                "MRC, HDF5/Imaris, Zarr/OME-Zarr, FITS, MetaImage, NRRD, video, Gatan DM3/DM4, Zeiss ZVI, Leica LEI and the common raster formats; "
                "every other Bio-Formats format opens through the optional Java bridge (the [bioformats] extra plus a JDK) or a bioio reader plugin.",
    params=[Param("path", "string", "", help="Image file (or Zarr / DICOM folder)")] + _import_params())
def bioformats_import(ctx, path="", **params):
    p = _expand(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return _run_import(ctx, p, params)


@op("file.bioformats_remote_import", "Bio-Formats Remote...", IMPORT, **_COMMON,
    keywords=["bio-formats", "remote", "url", "download", "internet", "importer"],
    description="Bio-Formats Remote Importer: download an image from a URL into a temporary file and open it with the Bio-Formats importer options "
                "(series, colour mode, view, autoscale, splitting, ranges, crop, metadata).",
    params=[Param("url", "string", "", help="http(s):// address of an image file")] + _import_params())
def bioformats_remote_import(ctx, url="", **params):
    from .import_export import download_url
    target, fname = download_url(url)
    res = _run_import(ctx, target, params, name=fname, source_label=fname)
    for im in res.new_images:
        im.path = None if not is_virtual(im.data) else im.path
        im.metadata["url"] = str(url).strip()
    return res


@op("file.import_tiff_virtual", "TIFF Virtual Stack...", IMPORT, **_COMMON, keywords=["virtual", "lazy", "memmap", "large", "tiff", "big", "out of core"],
    description="Open a TIFF without loading it into memory (File ▸ Import ▸ TIFF Virtual Stack): uncompressed files are memory-mapped (copy-on-write), "
                "compressed / tiled files read each plane on demand with a small cache. Display, slicing and analysis read only the planes they need; "
                "the first command that modifies the pixels (or needs the whole array, e.g. a 32-bit value range, Stack to Images, arithmetic) copies "
                "the stack into memory – undo and Save then work as usual. Limits: one series / resolution level at a time (pick a pyramid level for "
                "whole-slide images – planes are read whole), only TIFF-family files (OME-TIFF, LSM, STK), ranges / crops read the selection into memory, "
                "16-bit and float display ranges come from the current plane.",
    params=[Param("path", "string", "", help="TIFF / OME-TIFF file"), Param("series", "int", 0, min=0, help="Series index (0-based)"),
            Param("level", "int", 0, min=0, label="Resolution level", help="Pyramid level, 0 = full resolution")])
def import_tiff_virtual(ctx, path="", series=0, level=0):
    p = _expand(path)
    if p.suffix.lower() not in (".tif", ".tiff", ".lsm", ".stk") and not p.name.lower().endswith((".ome.tif", ".ome.tiff")):
        raise ValueError("TIFF Virtual Stack opens TIFF-family files only (tif, tiff, ome.tif, lsm, stk) – use Bio-Formats… or Open for other formats")
    if not p.exists():
        raise FileNotFoundError(str(p))
    img = read_image(p, series=series, level=level, virtual=True)
    kind = type(img.data).__name__
    virtual = is_virtual(img.data)
    how = ("memory-mapped" if kind == "memmap" else
           "memory-mapped, planes byte-swapped on demand" if getattr(img.data, "memmapped", False) else "planes on demand")
    msg = f"{img.name} {img.shape[1:]} {img.bit_depth}" + (f" – virtual ({how})" if virtual else " – read into memory (layout not mappable)")
    return OpResult(new_images=[img], message=msg, log=msg)


@op("file.bioformats_export", "Bio-Formats Exporter...", EXPORT, kind="image", scope_param=False, preview=False, roi_aware=False,
    keywords=["bio-formats", "exporter", "ome-tiff", "ome-zarr", "compression", "lzw", "jpeg", "split", "save"],
    description="Save the image as OME-TIFF (or OME-Zarr / ImageJ TIFF) like Fiji's Bio-Formats Exporter: compression (Uncompressed, LZW, zlib, JPEG, "
                "JPEG-2000, PackBits, Zstd), one file per Z section / timepoint / channel (name_Z0_C0_T0.ome.tif, optionally zero-padded), a C/Z/T save "
                "range, and OME-XML metadata embedding (channel names and colours, physical sizes, time increment, description, the image's metadata keys).",
    params=[Param("path", "string", "", help="Output file; the extension is set from the format"),
            Param("format", "choice", "OME-TIFF", choices=["OME-TIFF", "OME-Zarr", "ImageJ TIFF"]),
            Param("compression", "choice", "Uncompressed", choices=COMPRESSIONS, help="TIFF codecs (JPEG needs 8-bit data); OME-Zarr uses its default codec"),
            Param("write_each_z_section", "bool", False, label="Write each Z section to a separate file"),
            Param("write_each_timepoint", "bool", False, label="Write each timepoint to a separate file"),
            Param("write_each_channel", "bool", False, label="Write each channel to a separate file"),
            Param("padded", "bool", False, help="Zero-pad the Z / C / T indices in the file names", advanced=True),
            Param("c_begin", "int", 0, min=0, label="C begin", advanced=True, help="Save range, 1-based; 0 = first / last"), Param("c_end", "int", 0, min=0, label="C end", advanced=True),
            Param("z_begin", "int", 0, min=0, label="Z begin", advanced=True), Param("z_end", "int", 0, min=0, label="Z end", advanced=True),
            Param("t_begin", "int", 0, min=0, label="T begin", advanced=True), Param("t_end", "int", 0, min=0, label="T end", advanced=True),
            Param("embed_metadata", "bool", True, label="Embed metadata", help="Write the image's metadata keys into the OME-XML (Lumen MapAnnotation) so they are restored on open"),
            Param("description", "string", "", help="OME Image Description", advanced=True)])
def bioformats_export(ctx, path="", format="OME-TIFF", compression="Uncompressed", write_each_z_section=False, write_each_timepoint=False,
                      write_each_channel=False, padded=False, c_begin=0, c_end=0, z_begin=0, z_end=0, t_begin=0, t_end=0, embed_metadata=True, description=""):
    img = ctx.img
    p = _expand(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmt = {"OME-TIFF": "ome.tif", "OME-Zarr": "ome.zarr", "ImageJ TIFF": "tif"}[format]
    ext = {"ome.tif": ".ome.tif", "ome.zarr": ".zarr", "tif": ".tif"}[fmt]
    base = _export_base_name(p.name)
    sub = select_range(img, (c_begin, c_end, 1), (z_begin, z_end, 1), (t_begin, t_end, 1))
    by_c = write_each_channel and not sub.rgb and sub.nc > 1
    by_z = write_each_z_section and sub.nz > 1
    by_t = write_each_timepoint and sub.nt > 1
    jobs: list[tuple[str, LumenImage, int | None, int | None, int | None]] = []
    if by_c or by_z or by_t:
        width = {"Z": len(str(sub.nz)), "C": len(str(sub.nc)), "T": len(str(sub.nt))}  # Fiji pads to the digits of the dimension size
        total = (sub.nc if by_c else 1) * (sub.nz if by_z else 1) * (sub.nt if by_t else 1)
        if total > 10000:
            raise ValueError(f"The export would write {total} files – choose fewer split dimensions")
        # limit=None: the exporter only writes files (its own 10000 cap applies above); split_image's default 500 is the
        # "do not open hundreds of windows" cap of the importer / Stack Slicer
        for piece in split_image(sub, by_c, by_z, by_t, limit=None):
            # only the " - C=.." / " - Z=.." / " - T=.." that THIS split appended: an image that was itself a Stack Slicer
            # or importer piece ("cells - C=1") must not carry its old index into the file names and the table
            parts = dict(re.findall(r" - ([CZT])=(\d+)", piece.name[len(sub.name):]))
            suffix = ""
            for key in "ZCT":
                if key in parts:
                    suffix += f"_{key}{int(parts[key]):0{width[key] if padded else 1}d}"
            piece.name = base + suffix
            jobs.append((base + suffix + ext, piece, int(parts.get("T", -1)) if "T" in parts else None,
                         int(parts.get("Z", -1)) if "Z" in parts else None, int(parts.get("C", -1)) if "C" in parts else None))
    else:
        jobs.append((base + ext, sub, None, None, None))
    rows = []
    n = len(jobs)
    opts: dict[str, Any] = {"compression": compression, "embed_metadata": embed_metadata}
    if description:
        opts["description"] = description
    if n == 1 and is_virtual(jobs[0][1].data):
        # a virtual stack is streamed page by page by write_image: the progress bar and Cancel follow the pages
        opts["progress"], opts["check_cancel"] = ctx.progress, ctx.check_cancel
    for i, (fname, piece, t, z, c) in enumerate(jobs):
        ctx.check_cancel()
        out = write_image(piece, p.parent / fname, fmt, **opts)
        size = sum(f.stat().st_size for f in Path(out).rglob("*") if f.is_file()) if Path(out).is_dir() else os.path.getsize(out)
        rows.append([Path(out).name, "" if t is None else t, "" if z is None else z, "" if c is None else c, size])
        if n > 1:
            ctx.progress((i + 1) / n, f"File {i + 1}/{n}")
    where = rows[0][0] if n == 1 else f"{n} files in {p.parent}"
    msg = f"Saved {img.name} as {format}" + (f" ({compression})" if compression != "Uncompressed" and fmt != "ome.zarr" else "") + f" → {where}"
    return OpResult(message=msg, log=msg, table={"name": "Export", "columns": ["File", "T", "Z", "C", "Bytes"], "rows": rows})


@op("file.stack_slicer", "Stack Slicer...", "Plugins/Bio-Formats", kind="image", scope_param=False, preview=False, roi_aware=False,
    keywords=["bio-formats", "stack slicer", "split", "channels", "slices", "frames", "hyperstack"],
    description="Fiji's Bio-Formats Stack Slicer: split the hyperstack into separate images by channel, Z slice and/or timepoint "
                "(named '<title> - C=0', 'Z=0', 'T=0'); optionally keep the original open.",
    params=[Param("split_channels", "bool", True, label="Split channels"), Param("split_z", "bool", False, label="Split Z slices"),
            Param("split_timepoints", "bool", False, label="Split timepoints"),
            Param("stack_order", "choice", "XYCZT", choices=STACK_ORDERS[1:], help="Accepted for Fiji compatibility (the pieces are hyperstacks)", advanced=True),
            Param("keep_original", "bool", True, label="Keep original")])
def stack_slicer(ctx, split_channels=True, split_z=False, split_timepoints=False, stack_order="XYCZT", keep_original=True):
    img = ctx.img
    pieces = split_image(img, split_channels, split_z, split_timepoints)
    if pieces == [img]:
        raise ValueError("Nothing to split: the image has no second channel / slice / frame along the chosen dimension(s)")
    return OpResult(new_images=pieces, close_image=not keep_original, message=f"Split {img.name} into {len(pieces)} image(s)")


@op("file.bioformats_config", "Bio-Formats Plugins Configuration", "Plugins/Bio-Formats", **_NOIMG, keywords=["bio-formats", "configuration", "formats", "readers", "libraries", "plugins"],
    description="The Bio-Formats Plugins Configuration window as a table: every format Lumen's importer handles, its extensions, the Python reader "
                "behind it and whether that reader is installed – plus the proprietary formats that still need Java Bio-Formats.")
def bioformats_config(ctx):
    import importlib.util
    from ..readers import bioformats_bridge
    rows = []
    for fmt, exts, lib, pip in FORMAT_TABLE:
        if pip is None:
            ok = True
        else:
            mod = {"imageio-ffmpeg": "imageio_ffmpeg", "pynrrd": "nrrd"}.get(pip, pip)
            try:
                ok = mod in sys.modules or importlib.util.find_spec(mod) is not None
            except (ImportError, ValueError):
                ok = False
        rows.append([fmt, exts, lib, "enabled" if ok else f"missing (pip install {pip})"])
    bridge = bioformats_bridge.status()
    fmt, exts, lib, pip = JAVA_BRIDGE_ROW
    bridge_status = f"enabled ({bridge['detail']})" if bridge["available"] else f"missing ({bridge['detail']})"
    rows.append([fmt, exts, lib, bridge_status])
    for name in UNREADABLE_WITHOUT_JAVA:
        rows.append([name, "", "Java Bio-Formats (bridge)", "enabled (Java bridge)" if bridge["available"] else
                     "not available (install the [bioformats] extra + a JDK, or convert with bfconvert / bioformats2raw to OME-TIFF / OME-Zarr)"])
    n_ok = sum(1 for r in rows[:len(FORMAT_TABLE)] if r[3] == "enabled")
    msg = f"{n_ok} of {len(FORMAT_TABLE)} reader groups enabled; Java Bio-Formats bridge {'enabled' if bridge['available'] else 'not installed'} " \
          f"({len(UNREADABLE_WITHOUT_JAVA)} proprietary formats {'read through it' if bridge['available'] else 'need it'})"
    return OpResult(table={"name": "Bio-Formats Plugins Configuration", "columns": ["Format", "Extensions", "Reader", "Status"], "rows": rows}, message=msg)
