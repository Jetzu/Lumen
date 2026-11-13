"""Reading and writing images (and ImageJ-compatible ROI files).

Readers are dispatched on the file extension.  Formats that need an optional library
import it lazily and raise ``RuntimeError("Reading .xyz needs: pip install …")`` when it
is missing, so the application imports and runs without any of them.

Read:   tif/tiff (ImageJ, OME, LSM, STK, MicroManager, MetaMorph), png/jpg/bmp/gif/webp/
        pgm/ppm/pbm/pnm/tga/jp2 (imageio), npy, nd2, czi, lif, oif/oib (oiffile), dcm +
        DICOM directories (pydicom), nii/nii.gz (nibabel), mrc/rec/map (mrcfile), h5/hdf5/
        ims (h5py), zarr + OME-Zarr (zarr), avi/mp4/mov/mkv/webm/wmv (imageio-ffmpeg),
        fits (astropy), mha/mhd (pure numpy), nrrd (pynrrd or SimpleITK), raw / text
        matrices (``read_raw`` / ``read_text_image``), image-sequence folders, Gatan dm3/dm4,
        Zeiss zvi and Leica lei (pure-Python readers in ``lumen.core.readers``), a generic
        fallback through bioio / aicsimageio and – last – the optional Java Bio-Formats bridge
        (``lumen.core.readers.bioformats_bridge``, the ``[bioformats]`` extra plus a JDK).
Write:  tif (ImageJ hyperstack, the default), ome.tif, npy, png/jpg/bmp/webp/pgm/ppm/tga,
        gif, nii/nii.gz, mrc, h5, zarr (OME-Zarr 0.4), avi/mp4/mov/mkv, mha/mhd, fits,
        raw, txt (text image).
Series: ``list_series(path)`` lists the Bio-Formats style series of a file (OME images, whole-slide
        pyramid series, ND2 positions, CZI scenes, LIF images, HDF5 datasets / Imaris levels, OME-Zarr
        multiscales, bioio scenes) as :class:`SeriesInfo`; ``read_image(path, series=i, level=k,
        virtual=True)`` reads one series / resolution level, lazily for TIFFs (:class:`VirtualStack`).
"""
from __future__ import annotations

import os
import re
import struct
import sys
import threading
import uuid
import zipfile
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .image import AXES, Calibration, Display, LumenImage
from .lazy import (ImarisPlaneSource, LazyArray, TiffPageSource, auto_virtual_bytes, hdf5_source, is_virtual as _lazy_is_virtual,
                   iter_planes as _lazy_iter_planes, npy_source, page_iterator, view_source, zarr_source)
from .readers import READER_BY_EXT, NativeSeries

# Extensions the file browser marks as openable images.
RASTER_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".pgm", ".ppm", ".pbm", ".pnm", ".tga", ".jp2", ".npy"}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".wmv", ".m4v", ".mpg", ".mpeg"}
IMAGE_EXTS = RASTER_EXTS | VIDEO_EXTS | {
    ".nd2", ".czi", ".lif", ".ome.tif", ".ome.tiff", ".lsm", ".stk", ".oif", ".oib",
    ".dcm", ".dicom", ".nii", ".nii.gz", ".mrc", ".rec", ".map", ".mrcs", ".h5", ".hdf5", ".ims",
    ".zarr", ".fits", ".fit", ".fts", ".mha", ".mhd", ".nrrd", ".nhdr",
    ".dm3", ".dm4", ".zvi", ".lei",
}

_UNIT_ALIASES = {"micron": "µm", "microns": "µm", "um": "µm", "μm": "µm", "micrometer": "µm", "micrometre": "µm",
                 "millimeter": "mm", "millimetre": "mm", "nanometer": "nm", "nanometre": "nm", "meter": "m", "metre": "m",
                 "angstrom": "Å", "centimeter": "cm", "inch": "inch", "pixel": "pixel", "pixels": "pixel", "unknown": "pixel", "": "pixel"}
_OME_UNITS = {"µm": "micrometer", "mm": "millimeter", "nm": "nanometer", "m": "meter", "Å": "angstrom", "cm": "centimeter", "inch": "inch"}
_TO_ANGSTROM = {"Å": 1.0, "nm": 10.0, "µm": 1e4, "mm": 1e7, "cm": 1e8, "m": 1e10, "inch": 2.54e8}
_TO_MM = {k: v / 1e7 for k, v in _TO_ANGSTROM.items()}  # MetaIO / ITK store spacings in millimetres
_ASCII_UNIT = {"µm": "um", "Å": "A"}  # for headers that must stay 7-bit ASCII
# Lumen time units <-> OME TimeIncrementUnit symbols
_TIME_TO_OME = {"sec": "s", "s": "s", "ms": "ms", "µs": "us", "us": "us", "ns": "ns", "min": "min", "h": "h", "hour": "h", "d": "d"}
_TIME_FROM_OME = {"s": "sec", "ms": "ms", "us": "µs", "µs": "µs", "ns": "ns", "min": "min", "h": "h", "d": "d"}
_TIME_TO_SEC = {"sec": 1.0, "s": 1.0, "min": 60.0, "h": 3600.0, "hour": 3600.0, "d": 86400.0}
# OME-TIFF: Lumen's display state (LUT names, ranges, mode, inverted, visible) travels in a MapAnnotation
# with this Namespace, referenced from the Image (see _lumen_display_map); Channel Name/Color stay standard OME.
_LUMEN_OME_NS = "lumen.display"
# Primary LUTs an OME Channel Color (one colour) can express, with the colour at the top of their ramp
_PRIMARY_LUT_RGB = {"Red": (255, 0, 0), "Green": (0, 255, 0), "Blue": (0, 0, 255), "Cyan": (0, 255, 255),
                    "Magenta": (255, 0, 255), "Yellow": (255, 255, 0), "Grays": (255, 255, 255)}
_LUT_MATCH_TOL = 8.0  # mean |difference| per table entry below which a foreign LUT counts as one of ours (closest distinct pair: 15.9)


def _norm_unit(u: Any) -> str:
    s = str(u or "").strip()
    return _UNIT_ALIASES.get(s.lower(), s or "pixel").replace("\\u00B5", "µ")


def _need(pkg: str, ext: str, pip: str | None = None, verb: str = "Reading"):
    """Import an optional package or raise a helpful RuntimeError."""
    import importlib
    try:
        return importlib.import_module(pkg)
    except ImportError as e:
        raise RuntimeError(f"{verb} {ext} needs: pip install {pip or pkg}") from e


def lumen_home() -> Path:
    """Lumen's per-user folder (recent files, plugins, sample cache): ``$LUMEN_HOME`` or ``~/.lumen``.

    Only the location is decided here (nothing is created).  ``LUMEN_HOME`` may use ``~``; without it
    the folder is ``~/.lumen`` on every platform (``C:\\Users\\<you>\\.lumen`` on Windows).
    """
    custom = os.environ.get("LUMEN_HOME", "").strip()
    return Path(custom).expanduser() if custom else Path.home() / ".lumen"


def _full_ext(p: Path) -> str:
    """Extension including double suffixes such as .ome.tif / .nii.gz."""
    n = p.name.lower()
    for double in (".ome.tif", ".ome.tiff", ".nii.gz", ".ome.zarr"):
        if n.endswith(double):
            return double
    return p.suffix.lower()


# Extensions a writer may *replace* when it is asked for another format ('cells.png' + tif -> cells.tif).
_OUT_STRIP_EXTS = IMAGE_EXTS | {".txt", ".csv", ".tsv", ".raw", ".bin", ".zip", ".roi"}


def _with_ext(p: Path, ext: str, *accept: str) -> Path:
    """Return ``p`` ending in ``.ext``.

    Keeps a suffix listed in ``accept`` (aliases such as .tiff for tif), replaces a *known*
    image/text extension, and otherwise appends – never ``Path.with_suffix``, which eats a
    dotted stem ('cells.v2' -> 'cells.tif', 'exp-1.5x' -> 'exp-1.png').
    """
    ext = "." + ext.lower().lstrip(".")
    ok = {a.lower() for a in accept} | {ext}
    cur = _full_ext(p)
    if cur in ok:
        return p
    if cur and cur in _OUT_STRIP_EXTS:
        return p.with_name(p.name[:-len(cur)] + ext)
    return p.with_name(p.name + ext)


def strip_image_ext(name: str) -> str:
    """'sample.ome.tif' -> 'sample', 'cells.v2.tif' -> 'cells.v2', 'cells.v2' -> 'cells.v2'."""
    p = Path(name)
    e = _full_ext(p)
    return p.name[:-len(e)] if e and e in _OUT_STRIP_EXTS else p.name


# ============================================================================ series (Bio-Formats style)
_BF_PIXEL_TYPES = {"uint8": "uint8", "int8": "int8", "uint16": "uint16", "int16": "int16", "uint32": "uint32", "int32": "int32",
                   "float32": "float", "float64": "double", "bool": "bit"}
_FORMAT_BY_EXT = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "BMP", ".gif": "GIF", ".webp": "WebP", ".pgm": "PNM", ".ppm": "PNM",
                  ".pbm": "PNM", ".pnm": "PNM", ".tga": "TGA", ".jp2": "JPEG-2000", ".npy": "NumPy", ".nd2": "Nikon ND2", ".czi": "Zeiss CZI",
                  ".lif": "Leica LIF", ".oif": "Olympus OIF", ".oib": "Olympus OIB", ".dcm": "DICOM", ".dicom": "DICOM", ".nii": "NIfTI",
                  ".nii.gz": "NIfTI", ".mrc": "MRC", ".rec": "MRC", ".map": "MRC", ".mrcs": "MRC", ".h5": "HDF5", ".hdf5": "HDF5", ".hdf": "HDF5",
                  ".ims": "Imaris", ".zarr": "Zarr", ".ome.zarr": "OME-Zarr", ".fits": "FITS", ".fit": "FITS", ".fts": "FITS", ".mha": "MetaImage",
                  ".mhd": "MetaImage", ".nrrd": "NRRD", ".nhdr": "NRRD", ".txt": "Text image", ".csv": "Text image", ".tsv": "Text image",
                  ".dm3": "Gatan DM3", ".dm4": "Gatan DM4", ".zvi": "Zeiss ZVI", ".lei": "Leica LEI"}


@dataclass
class SeriesInfo:
    """One image series of a file – what Bio-Formats calls a series: an OME Image, a CZI scene, an ND2 position, a
    LIF image, an HDF5 dataset, a Zarr multiscale image … – with its sizes, pixel type, dimension order, calibration,
    channel names and resolution (pyramid) levels.  ``list_series(path)`` returns them; ``read_image(path, series=i,
    level=k)`` reads one."""
    index: int = 0
    name: str = ""
    size_x: int = 0
    size_y: int = 0
    size_z: int = 1
    size_c: int = 1
    size_t: int = 1
    dtype: str = "uint8"
    dimension_order: str = "XYCZT"
    rgb: bool = False
    pixel_width: float | None = None
    pixel_height: float | None = None
    pixel_depth: float | None = None
    unit: str = ""
    time_increment: float | None = None
    time_unit: str = ""
    channel_names: list[str] = field(default_factory=list)
    #: (height, width) of every resolution level, full resolution first (one entry for ordinary images)
    levels: list[tuple[int, int]] = field(default_factory=list)
    axes: str = ""
    format: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    #: per-plane timing / position (OME Plane elements): z, c, t, delta_t, exposure_time, position_x, position_y, position_z
    planes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def pixel_type(self) -> str:
        """The Bio-Formats pixel type name (uint8, uint16, int16, float, double …)."""
        return _BF_PIXEL_TYPES.get(str(self.dtype), str(self.dtype))

    @property
    def effective_size_c(self) -> int:
        return 1 if self.rgb else max(1, self.size_c)

    @property
    def rgb_channel_count(self) -> int:
        return self.size_c if self.rgb else 1

    @property
    def image_count(self) -> int:
        """Number of planes as Bio-Formats counts them (an RGB plane is one image)."""
        return max(1, self.size_z) * max(1, self.size_t) * self.effective_size_c

    @property
    def level_count(self) -> int:
        return max(1, len(self.levels))

    def _dim_sizes(self) -> dict[str, int]:
        return {"Z": max(1, self.size_z), "C": self.effective_size_c, "T": max(1, self.size_t)}

    def plane_index(self, z: int, c: int, t: int) -> int:
        """Bio-Formats ``getIndex(z, c, t)``: the plane number in this series' dimension order."""
        sizes = self._dim_sizes()
        coords = {"Z": int(z), "C": int(c), "T": int(t)}
        index, stride = 0, 1
        for ax in self.dimension_order[2:].upper():
            if ax in sizes:
                index += coords[ax] * stride
                stride *= sizes[ax]
        return index

    def zct_coords(self, index: int) -> tuple[int, int, int]:
        """Bio-Formats ``getZCTCoords(index)``: (z, c, t) of plane ``index``."""
        sizes = self._dim_sizes()
        coords = {"Z": 0, "C": 0, "T": 0}
        rem = int(index)
        for ax in self.dimension_order[2:].upper():
            if ax in sizes:
                coords[ax] = rem % sizes[ax]
                rem //= sizes[ax]
        return coords["Z"], coords["C"], coords["T"]

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("metadata", "planes")}
        d["pixel_type"] = self.pixel_type
        d["image_count"] = self.image_count
        d["metadata"] = {k: v for k, v in self.metadata.items() if isinstance(v, (str, int, float, bool))}
        return d

    def to_row(self) -> list[Any]:
        """A row for the series table: Index, Name, SizeX, SizeY, SizeZ, SizeC, SizeT, Type, Order, Levels, Pixel size, Unit, Channels, Format."""
        return [self.index, self.name, self.size_x, self.size_y, self.size_z, self.size_c, self.size_t, self.pixel_type + (" (RGB)" if self.rgb else ""),
                self.dimension_order, self.level_count, self.pixel_width if self.pixel_width is not None else "", self.unit or "",
                ", ".join(self.channel_names), self.format]

    def metadata_rows(self) -> list[list[Any]]:
        """Key/Value rows (the Bio-Formats "Original metadata" window): the core sizes first, then the reader's metadata."""
        rows: list[list[Any]] = [["Series", self.index], ["Name", self.name], ["Format", self.format], ["SizeX", self.size_x], ["SizeY", self.size_y],
                                 ["SizeZ", self.size_z], ["SizeC", self.size_c], ["SizeT", self.size_t], ["PixelType", self.pixel_type],
                                 ["DimensionOrder", self.dimension_order], ["RGB", self.rgb], ["ImageCount", self.image_count],
                                 ["ResolutionLevels", self.level_count]]
        if self.pixel_width is not None:
            rows += [["PhysicalSizeX", self.pixel_width], ["PhysicalSizeY", self.pixel_height if self.pixel_height is not None else self.pixel_width],
                     ["PhysicalSizeUnit", self.unit]]
            if self.pixel_depth is not None:
                rows.append(["PhysicalSizeZ", self.pixel_depth])
        if self.time_increment is not None:
            rows += [["TimeIncrement", self.time_increment], ["TimeIncrementUnit", self.time_unit]]
        for i, n in enumerate(self.channel_names):
            rows.append([f"Channel {i} Name", n])
        for i, (h, w) in enumerate(self.levels[1:], 1):
            rows.append([f"Level {i} size", f"{w}×{h}"])
        for k, v in sorted(self.metadata.items()):
            if isinstance(v, (str, int, float, bool)):
                rows.append([str(k), v])
        for pl in self.planes[:2000]:
            tag = f"Plane Z={pl.get('z', 0)} C={pl.get('c', 0)} T={pl.get('t', 0)}"
            for key in ("delta_t", "exposure_time", "position_x", "position_y", "position_z"):
                if pl.get(key) is not None:
                    rows.append([f"{tag} {key}", pl[key]])
        return rows

    @classmethod
    def from_image(cls, img: LumenImage, index: int = 0, name: str | None = None, format: str = "") -> "SeriesInfo":
        cal = img.calibration
        info = cls(index=index, name=name or strip_image_ext(img.name), size_x=img.width, size_y=img.height, size_z=img.nz, size_c=img.nc,
                   size_t=img.nt, dtype=str(img.dtype), dimension_order="XYCZT", rgb=img.rgb, axes="TZCYX", format=format,
                   levels=[(img.height, img.width)], channel_names=[] if img.rgb else [c.name for c in img.display.channels],
                   metadata={k: v for k, v in img.metadata.items() if isinstance(v, (str, int, float, bool))})
        if cal.scaled:
            info.pixel_width, info.pixel_height, info.pixel_depth, info.unit = cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit
        if cal.frame_interval:
            info.time_increment, info.time_unit = cal.frame_interval, cal.time_unit
        return info


def _canonical_axes(axes: str) -> str:
    """A reader's axes string with the letters LumenImage.to_canonical folds them to: 'S' → C, unknown → Z (then T)."""
    axes = axes.upper().replace("S", "C")
    for ch in axes:
        if ch not in AXES:
            axes = axes.replace(ch, "Z" if "Z" not in axes else "T")
    return axes


def _dimension_order_from_axes(axes: str) -> str:
    """Bio-Formats DimensionOrder (fastest varying first after XY) for a storage-order axes string ('TZCYX' → 'XYCZT')."""
    lead = [ch for ch in _canonical_axes(axes) if ch in "TZC"]
    order = "".join(reversed(list(dict.fromkeys(lead))))
    for ch in "CZT":
        if ch not in order:
            order += ch
    return "XY" + order


def _series_info_from_axes(index: int, axes: str, shape: tuple[int, ...], dtype: Any) -> SeriesInfo:
    """Sizes / RGB flag / dimension order from a reader's axes + shape (the folding LumenImage applies)."""
    axes = str(axes).upper()
    shape = tuple(int(v) for v in shape)
    sizes = dict(zip(axes, shape))
    dt = np.dtype(dtype)
    rgb = "S" in axes and sizes["S"] in (3, 4) and dt == np.uint8
    counts = {"T": 1, "Z": 1, "C": 1, "Y": 1, "X": 1}
    for ch, n in zip(_canonical_axes(axes), shape):
        counts[ch] = counts.get(ch, 1) * n
    if rgb:
        counts["C"] = 3
    return SeriesInfo(index=index, size_x=counts["X"], size_y=counts["Y"], size_z=counts["Z"], size_c=counts["C"], size_t=counts["T"],
                      dtype=str(dt), dimension_order=_dimension_order_from_axes(axes), rgb=rgb, axes=axes, levels=[(counts["Y"], counts["X"])])


def _float_or_none(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _ome_images_info(xml: str | None) -> list[dict[str, Any]]:
    """Per-Image information from OME-XML: name, Pixels attributes, channel names / colours, Plane timings and positions,
    acquisition date, description – plus the file's ``OriginalMetadata`` key/value annotations under ``global``."""
    out: list[dict[str, Any]] = []
    if not xml:
        return out
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
    except Exception:
        return out

    def tag(e) -> str:
        return e.tag.rsplit("}", 1)[-1]

    original: dict[str, Any] = {}
    for om in root.iter():
        if tag(om) == "OriginalMetadata":
            key = value = None
            for ch in om:
                if tag(ch) == "Key":
                    key = (ch.text or "").strip()
                elif tag(ch) == "Value":
                    value = (ch.text or "").strip()
            if key:
                original[key] = value
    for image in root.iter():
        if tag(image) != "Image":
            continue
        info: dict[str, Any] = {"name": image.get("Name") or "", "id": image.get("ID") or "", "global": original, "pixels": {},
                                "channel_names": [], "channel_colors": [], "planes": []}
        for child in image:
            t = tag(child)
            if t == "AcquisitionDate":
                info["acquisition_date"] = (child.text or "").strip()
            elif t == "Description":
                info["description"] = (child.text or "").strip()
            elif t == "Pixels":
                info["pixels"] = dict(child.attrib)
                for el in child:
                    et = tag(el)
                    if et == "Channel":
                        info["channel_names"].append(el.get("Name") or "")
                        info["channel_colors"].append(el.get("Color"))
                    elif et == "Plane":
                        info["planes"].append({"z": int(el.get("TheZ", 0) or 0), "c": int(el.get("TheC", 0) or 0), "t": int(el.get("TheT", 0) or 0),
                                               "delta_t": _float_or_none(el.get("DeltaT")), "exposure_time": _float_or_none(el.get("ExposureTime")),
                                               "position_x": _float_or_none(el.get("PositionX")), "position_y": _float_or_none(el.get("PositionY")),
                                               "position_z": _float_or_none(el.get("PositionZ"))})
        out.append(info)
    return out


def _apply_ome_pixels(target: Any, px: dict[str, Any]) -> None:
    """Copy PhysicalSize* / TimeIncrement of an OME Pixels element onto a Calibration or a SeriesInfo."""
    is_info = isinstance(target, SeriesInfo)
    if px.get("PhysicalSizeX"):
        w = float(px["PhysicalSizeX"])
        h = float(px.get("PhysicalSizeY") or w)
        unit = _norm_unit(px.get("PhysicalSizeXUnit", "µm"))
        if is_info:
            target.pixel_width, target.pixel_height, target.unit = w, h, unit
            target.pixel_depth = float(px["PhysicalSizeZ"]) if px.get("PhysicalSizeZ") else None
        else:
            target.pixel_width, target.pixel_height, target.unit = w, h, unit
            if px.get("PhysicalSizeZ"):
                target.pixel_depth = float(px["PhysicalSizeZ"])
    elif px.get("PhysicalSizeZ") and not is_info:
        target.pixel_depth = float(px["PhysicalSizeZ"])
    if px.get("TimeIncrement"):
        ti = float(px["TimeIncrement"])
        tu = _TIME_FROM_OME.get(px.get("TimeIncrementUnit") or "s", px.get("TimeIncrementUnit") or "sec")
        if is_info:
            target.time_increment, target.time_unit = ti, tu
        else:
            target.frame_interval, target.time_unit = ti, tu


def _tiff_format_name(tf: Any) -> str:
    for flag, name in (("ome", "OME-TIFF"), ("imagej", "ImageJ TIFF"), ("svs", "Aperio SVS"), ("ndpi", "Hamamatsu NDPI"), ("scn", "Leica SCN"),
                       ("bif", "Ventana BIF"), ("philips", "Philips TIFF"), ("qpi", "PerkinElmer QPTIFF"), ("lsm", "Zeiss LSM"), ("stk", "MetaMorph STK"),
                       ("micromanager", "Micro-Manager TIFF"), ("mmstack", "Micro-Manager TIFF"), ("ndtiff", "NDTiff"), ("fluoview", "Olympus FluoView TIFF"),
                       ("metaseries", "MetaMorph TIFF"), ("nih", "NIH Image"), ("mdgel", "Molecular Dynamics GEL"), ("geotiff", "GeoTIFF"),
                       ("bigtiff", "BigTIFF")):
        try:
            if bool(getattr(tf, "is_" + flag, False)):
                return name
        except Exception:
            continue
    return "TIFF"


def _tiff_global_metadata(tf: Any) -> dict[str, Any]:
    """Scalar metadata of a TIFF file (ImageJ / LSM / STK / Micro-Manager / MetaMorph tags, a few baseline tags, the slide
    scanner description fields) for the Metadata table and ``Ext.getMetadataValue``."""
    meta: dict[str, Any] = {}

    def scalars(prefix: str, d: Any) -> None:
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (str, int, float, bool)) and not isinstance(v, np.ndarray):
                    meta[f"{prefix}{k}"] = v
                elif isinstance(v, np.generic):
                    meta[f"{prefix}{k}"] = v.item()
    try:
        page = tf.pages[0]
        for name in ("Software", "DateTime", "Make", "Model", "Artist", "ImageWidth", "ImageLength", "BitsPerSample", "SamplesPerPixel",
                     "ResolutionUnit", "Orientation", "PlanarConfiguration", "TileWidth", "TileLength", "RowsPerStrip"):
            t = page.tags.get(name)
            if t is not None and isinstance(t.value, (str, int, float)):
                meta[name] = t.value
        try:
            meta["Compression"] = page.compression.name
            meta["PhotometricInterpretation"] = page.photometric.name
        except Exception:
            pass
        desc = str(getattr(page, "description", "") or "")
        if desc and not desc.lstrip().startswith("<"):
            if "|" in desc and "=" in desc:  # Aperio SVS style "Aperio Image Library ...|AppMag = 20|MPP = 0.5"
                for part in desc.split("|"):
                    if "=" in part:
                        k, v = part.split("=", 1)
                        meta[k.strip()] = v.strip()
            elif len(desc) <= 2000:
                meta["ImageDescription"] = desc
    except Exception:
        pass
    for attr, prefix in (("imagej_metadata", ""), ("lsm_metadata", "LSM "), ("stk_metadata", "STK "), ("metaseries_metadata", "MetaSeries "),
                         ("fluoview_metadata", "FluoView "), ("philips_metadata", "Philips "), ("ndpi_metadata", "NDPI ")):
        try:
            d = getattr(tf, attr, None)
        except Exception:
            d = None
        if isinstance(d, dict):
            scalars(prefix, d)
            for sub in ("PlaneInfo", "Summary"):
                scalars(prefix, d.get(sub))
    try:
        mm = getattr(tf, "micromanager_metadata", None)
        if isinstance(mm, dict):
            scalars("MicroManager ", mm.get("Summary"))
    except Exception:
        pass
    try:
        page = tf.pages[0]
        nd = getattr(page, "ndpi_tags", None)
        if isinstance(nd, dict):
            scalars("NDPI ", nd)
    except Exception:
        pass
    return meta


def _tiff_levels(series: Any) -> list[tuple[int, int]]:
    out = []
    try:
        for lvl in (series.levels or [series]):
            ax = str(lvl.axes)
            out.append((int(lvl.shape[ax.index("Y")]) if "Y" in ax else 1, int(lvl.shape[ax.index("X")]) if "X" in ax else 1))
    except Exception:
        pass
    return out


def _list_series_tiff(p: Path) -> list[SeriesInfo]:
    import tifffile
    out: list[SeriesInfo] = []
    with tifffile.TiffFile(str(p)) as tf:
        cal, fmeta, fnames, _disp = _ij_metadata_to_calibration(tf)
        ome_images = _ome_images_info(getattr(tf, "ome_metadata", None))
        fmt = _tiff_format_name(tf)
        gmeta = _tiff_global_metadata(tf)
        n = len(tf.series)
        for i, s in enumerate(tf.series):
            info = _series_info_from_axes(i, s.axes, s.shape, s.dtype)
            info.format = fmt
            info.name = str(getattr(s, "name", "") or "")
            info.levels = _tiff_levels(s) or info.levels
            info.pixel_width, info.pixel_height, info.pixel_depth, info.unit = cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit
            if cal.frame_interval:
                info.time_increment, info.time_unit = cal.frame_interval, cal.time_unit
            if fnames and len(fnames) == info.size_c and not info.rgb:
                info.channel_names = list(fnames)
            info.metadata = dict(gmeta)
            oi = ome_images[i] if i < len(ome_images) else None
            if oi:
                if oi.get("name"):
                    info.name = oi["name"]
                # an OME Image carries its own Pixels: never inherit the first Image's sizes (Bio-Formats reports null → NaN)
                info.pixel_width = info.pixel_height = info.pixel_depth = info.time_increment = None
                info.unit = info.time_unit = ""
                _apply_ome_pixels(info, oi.get("pixels", {}))
                if oi.get("pixels", {}).get("DimensionOrder"):
                    info.dimension_order = oi["pixels"]["DimensionOrder"]
                names = oi.get("channel_names") or []
                if names and all(names) and len(names) == info.effective_size_c and not info.rgb:
                    info.channel_names = [str(x) for x in names]
                info.planes = list(oi.get("planes") or [])
                info.metadata.update(oi.get("global") or {})
                for key in ("acquisition_date", "description", "id"):
                    if oi.get(key):
                        info.metadata[{"acquisition_date": "AcquisitionDate", "description": "Description", "id": "ImageID"}[key]] = oi[key]
            if not info.name:
                info.name = strip_image_ext(p.name) if n == 1 else f"{strip_image_ext(p.name)} #{i + 1}"
            out.append(info)
    return out


def _list_series_nd2(p: Path) -> list[SeriesInfo]:
    nd2 = _need("nd2", ".nd2")
    out: list[SeriesInfo] = []
    with nd2.ND2File(str(p)) as f:
        sizes = dict(f.sizes)
        n_pos = int(sizes.get("P", 1) or 1)
        axes = "".join(k for k in sizes if k != "P")
        shape = tuple(int(sizes[k]) for k in sizes if k != "P")
        try:
            dtype = np.dtype(getattr(f, "dtype", None) or np.uint16)
        except Exception:
            dtype = np.dtype(np.uint16)
        cal = Calibration()
        try:
            vx = f.voxel_size()
            cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit = float(vx.x), float(vx.y), float(vx.z), "µm"
        except Exception:
            pass
        names: list[str] = []
        try:
            names = [str(c.channel.name) for c in f.metadata.channels]
        except Exception:
            pass
        meta: dict[str, Any] = {"positions": n_pos}
        try:
            for k, v in (getattr(f, "text_info", None) or {}).items():
                if isinstance(v, (str, int, float)) and len(str(v)) < 500:
                    meta[str(k)] = v
        except Exception:
            pass
        for i in range(n_pos):
            info = _series_info_from_axes(i, axes, shape, dtype)
            info.name = strip_image_ext(p.name) + (f" (position {i + 1})" if n_pos > 1 else "")
            info.format = "Nikon ND2"
            if cal.unit == "µm":
                info.pixel_width, info.pixel_height, info.pixel_depth, info.unit = cal.pixel_width, cal.pixel_height, cal.pixel_depth, "µm"
            if names and len(names) == info.size_c:
                info.channel_names = list(names)
            info.metadata = dict(meta)
            out.append(info)
    return out


def _czi_scaling(f: Any) -> Calibration | None:
    """Pixel sizes from the CZI metadata (Scaling/Items/Distance, in metres) when czifile exposes it."""
    try:
        md = f.metadata(raw=False) if callable(getattr(f, "metadata", None)) else None
        items = md["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
        if isinstance(items, dict):
            items = [items]
        cal = Calibration(unit="µm")
        for it in items:
            v = float(it.get("Value")) * 1e6
            if it.get("Id") == "X":
                cal.pixel_width = v
            elif it.get("Id") == "Y":
                cal.pixel_height = v
            elif it.get("Id") == "Z":
                cal.pixel_depth = v
        return cal
    except Exception:
        return None


def _list_series_czi(p: Path) -> list[SeriesInfo]:
    czifile = _need("czifile", ".czi")
    out: list[SeriesInfo] = []
    with czifile.CziFile(str(p)) as f:
        cal = _czi_scaling(f)
        if hasattr(f, "axes") and hasattr(f, "shape"):
            axes = "".join("S" if ch == "0" else ("P" if ch == "S" else ch) for ch in str(f.axes))
            shape = tuple(int(v) for v in f.shape)
            n_scenes = shape[axes.index("P")] if "P" in axes else 1
            keep = [i for i, ch in enumerate(axes) if ch != "P"]
            sub_axes, sub_shape = "".join(axes[i] for i in keep), tuple(shape[i] for i in keep)
            dtype = getattr(f, "dtype", np.uint16)
            for i in range(n_scenes):
                info = _series_info_from_axes(i, sub_axes, sub_shape, dtype)
                info.name = strip_image_ext(p.name) + (f" (scene {i + 1})" if n_scenes > 1 else "")
                info.format = "Zeiss CZI"
                out.append(info)
        else:
            scenes = list(getattr(f, "scenes").items())
            for i, (key, im) in enumerate(scenes):
                shape = getattr(im, "shape", None)
                dtype = getattr(im, "dtype", None)
                if shape is None or dtype is None:
                    arr = np.asarray(im.asarray())
                    shape, dtype = arr.shape, arr.dtype
                info = _series_info_from_axes(i, str(im.axes), shape, dtype)
                info.name = str(key) if len(scenes) > 1 else strip_image_ext(p.name)
                info.format = "Zeiss CZI"
                out.append(info)
    if cal is not None:
        for info in out:
            info.pixel_width, info.pixel_height, info.pixel_depth, info.unit = cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit
    return out


def _list_series_lif(p: Path) -> list[SeriesInfo]:
    try:
        from readlif.reader import LifFile
    except ImportError as e:
        raise RuntimeError("Reading .lif needs: pip install readlif") from e
    lif = LifFile(str(p))
    n_images = int(getattr(lif, "num_images", 0) or len(getattr(lif, "image_list", [])) or 1)
    out: list[SeriesInfo] = []
    for i in range(n_images):
        im = lif.get_image(i)
        dims = im.dims
        nt, nz, nc = int(getattr(dims, "t", 1) or 1), int(getattr(dims, "z", 1) or 1), int(getattr(im, "channels", 1) or 1)
        bd = getattr(im, "bit_depth", None)
        if isinstance(bd, (list, tuple)):
            bd = bd[0] if bd else None
        dtype = "uint8" if (bd is not None and int(bd) <= 8) else "uint16"
        info = SeriesInfo(index=i, name=str(getattr(im, "name", "") or f"Image {i}"), size_x=int(dims.x), size_y=int(dims.y), size_z=nz, size_c=nc,
                          size_t=nt, dtype=dtype, dimension_order="XYCZT", axes="TZCYX", format="Leica LIF", levels=[(int(dims.y), int(dims.x))])
        try:
            sx, sy, sz, st = (list(im.scale) + [None] * 4)[:4]
            if sx:
                info.pixel_width, info.pixel_height, info.unit = 1.0 / sx, 1.0 / (sy or sx), "µm"
                info.pixel_depth = 1.0 / sz if sz else 1.0
            if st and nt > 1:
                info.time_increment, info.time_unit = 1.0 / st, "sec"
        except Exception:
            pass
        if int(getattr(dims, "m", 1) or 1) > 1:
            info.metadata["mosaic_tiles"] = int(dims.m)
        out.append(info)
    return out


def _ims_levels(f: Any) -> list[str]:
    ds = f.get("DataSet")
    if ds is None:
        return []
    keys = [k for k in ds.keys() if k.startswith("ResolutionLevel")]
    return sorted(keys, key=lambda s: int(s.split()[-1]))


def _ims_level_sizes(level: Any) -> tuple[int, int, int] | None:
    """(Z, Y, X) of an Imaris resolution level from its ImageSize* attributes (the Data arrays are padded)."""
    try:
        tp = sorted((k for k in level.keys() if k.startswith("TimePoint")), key=lambda s: int(s.split()[-1]))[0]
        ch = sorted((k for k in level[tp].keys() if k.startswith("Channel")), key=lambda s: int(s.split()[-1]))[0]
        g = level[tp][ch]
        sx, sy, sz = (int(_ims_attr(g, k)) for k in ("ImageSizeX", "ImageSizeY", "ImageSizeZ"))
        return sz, sy, sx
    except Exception:
        return None


def _h5_image_datasets(f: Any) -> list[str]:
    names: list[str] = []

    def visit(name, obj):
        import h5py
        if isinstance(obj, h5py.Dataset) and 2 <= obj.ndim <= 5 and obj.dtype.kind in "uif":
            names.append(name)
    f.visititems(visit)
    return names


def _h5_axes_guess(ds: Any) -> str | None:
    attrs = dict(ds.attrs)
    axes = attrs.get("axes")
    if isinstance(axes, bytes):
        axes = axes.decode()
    if axes is not None and len(str(axes)) == ds.ndim:
        return str(axes)
    if "DIMENSION_LABELS" in attrs:
        try:
            labels = "".join(str(x.decode() if isinstance(x, bytes) else x)[0].upper() for x in attrs["DIMENSION_LABELS"])
            if len(labels) == ds.ndim:
                return labels
        except Exception:
            pass
    return None


def _list_series_hdf5(p: Path) -> list[SeriesInfo]:
    h5py = _need("h5py", p.suffix)
    out: list[SeriesInfo] = []
    with h5py.File(str(p), "r") as f:
        if "DataSet" in f and "DataSetInfo" in f:
            levels = _ims_levels(f)
            lvl0 = f["DataSet"][levels[0]]
            tps = sorted((k for k in lvl0.keys() if k.startswith("TimePoint")), key=lambda s: int(s.split()[-1]))
            chans = sorted((k for k in lvl0[tps[0]].keys() if k.startswith("Channel")), key=lambda s: int(s.split()[-1]))
            d0 = lvl0[tps[0]][chans[0]]["Data"]
            sizes = _ims_level_sizes(lvl0) or tuple(int(v) for v in d0.shape)
            info = SeriesInfo(index=0, name=strip_image_ext(p.name), size_x=sizes[2], size_y=sizes[1], size_z=sizes[0], size_c=len(chans), size_t=len(tps),
                              dtype=str(d0.dtype), dimension_order="XYZCT", axes="TCZYX", format="Imaris (HDF5)")
            info.levels = [(s[1], s[2]) for s in (_ims_level_sizes(f["DataSet"][k]) for k in levels) if s] or [(sizes[1], sizes[2])]
            cal = _ims_calibration(f, sizes[2], sizes[1], sizes[0])
            if cal.unit != "pixel":
                info.pixel_width, info.pixel_height, info.pixel_depth, info.unit = cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit
            for i, _ in enumerate(chans):
                ci = f.get(f"DataSetInfo/Channel {i}")
                info.channel_names.append(str(_ims_attr(ci, "Name", f"C{i + 1}")) if ci is not None else f"C{i + 1}")
            img_info = f.get("DataSetInfo/Image")
            if img_info is not None:
                for k in img_info.attrs.keys():
                    v = _ims_attr(img_info, k)
                    if isinstance(v, (str, int, float)) and len(str(v)) < 200:
                        info.metadata[str(k)] = v
            return [info]
        for i, key in enumerate(_h5_image_datasets(f)):
            ds = f[key]
            axes = _h5_axes_guess(ds)
            if axes is None:
                axes = {2: "YX", 3: "YXS" if (ds.shape[-1] in (3, 4) and ds.dtype == np.uint8) else "ZYX", 4: "ZCYX", 5: "TZCYX"}.get(ds.ndim, "ZYX")
            info = _series_info_from_axes(i, axes, ds.shape, ds.dtype)
            info.name, info.format = key, "HDF5"
            attrs = dict(ds.attrs)
            if "element_size_um" in attrs:
                esu = np.atleast_1d(np.asarray(attrs["element_size_um"], float))
                if esu.size >= 2:
                    info.pixel_height, info.pixel_width, info.unit = float(esu[-2]), float(esu[-1]), "µm"
                    info.pixel_depth = float(esu[-3]) if esu.size >= 3 else 1.0
            if "pixel_width" in attrs:
                info.pixel_width = float(attrs["pixel_width"])
                info.pixel_height = float(attrs.get("pixel_height", attrs["pixel_width"]))
                info.pixel_depth = float(attrs.get("pixel_depth", 1.0))
                u = attrs.get("unit", "pixel")
                info.unit = _norm_unit(u.decode() if isinstance(u, bytes) else u)
            names = attrs.get("channel_names")
            if names is not None:
                info.channel_names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
            info.metadata = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in attrs.items() if isinstance(v, (str, int, float, np.generic))}
            out.append(info)
    if not out:
        raise ValueError(f"No image dataset found in {p.name}")
    return out


def _zarr_array_shape(node: Any, path: str) -> tuple[tuple[int, ...], Any] | None:
    try:
        a = node[path]
        return tuple(int(v) for v in a.shape), a.dtype
    except Exception:
        return None


def _list_series_zarr(p: Path, index_offset: int = 0, name: str | None = None) -> list[SeriesInfo]:
    zarr = _need("zarr", ".zarr")
    node = zarr.open(str(p), mode="r")
    out: list[SeriesInfo] = []
    if not hasattr(node, "arrays"):  # a bare array
        attrs = {}
        try:
            attrs = dict(node.attrs.asdict()) if hasattr(node.attrs, "asdict") else dict(node.attrs)
        except Exception:
            pass
        dims = attrs.get("_ARRAY_DIMENSIONS") or attrs.get("axes")
        axes = ("".join(str(d)[0].upper() for d in dims) if not isinstance(dims, str) else dims) if dims and len(dims) == node.ndim else None
        if axes is None or any(ch not in "TZCYXS" for ch in axes):
            axes = {2: "YX", 3: "ZYX", 4: "ZCYX", 5: "TZCYX"}.get(node.ndim, "ZYX")
        info = _series_info_from_axes(index_offset, axes, node.shape, node.dtype)
        info.name, info.format = name or strip_image_ext(p.name), "Zarr"
        return [info]
    attrs = _zarr_group_attrs(node)
    ms = attrs.get("multiscales")
    if ms:
        m0 = ms[0]
        datasets = m0.get("datasets") or []
        ax = m0.get("axes")
        axes = "".join((a["name"] if isinstance(a, dict) else str(a))[0].upper() for a in ax) if ax else None
        first = _zarr_array_shape(node, datasets[0]["path"]) if datasets else None
        if first is None:
            raise ValueError(f"No multiscale array found in {p.name}")
        shape, dtype = first
        if axes is None or len(axes) != len(shape):
            axes = {2: "YX", 3: "ZYX", 4: "CZYX", 5: "TCZYX"}.get(len(shape), "ZYX")
        info = _series_info_from_axes(index_offset, axes, shape, dtype)
        info.name = name or str(m0.get("name") or strip_image_ext(p.name))
        info.format = "OME-Zarr"
        info.levels = []
        for d in datasets:
            sd = _zarr_array_shape(node, d["path"])
            if sd:
                info.levels.append((sd[0][axes.index("Y")] if "Y" in axes else 1, sd[0][axes.index("X")] if "X" in axes else 1))
        scale = None
        for tr in (datasets[0].get("coordinateTransformations", []) if datasets else []) + m0.get("coordinateTransformations", []):
            if tr.get("type") == "scale" and scale is None:
                scale = tr["scale"]
        if ax and scale:
            for i, a in enumerate(ax):
                if not isinstance(a, dict) or i >= len(scale):
                    continue
                nm, unit, sv = a["name"].lower(), a.get("unit"), float(scale[i])
                if nm == "x":
                    info.pixel_width = sv
                    info.unit = _norm_unit(unit) if unit else info.unit
                elif nm == "y":
                    info.pixel_height = sv
                elif nm == "z":
                    info.pixel_depth = sv
                elif nm == "t":
                    info.time_increment = sv
                    info.time_unit = {"second": "sec", "millisecond": "ms", "minute": "min"}.get(unit, unit) if unit else "sec"
        if info.pixel_width is not None and info.pixel_height is None:
            info.pixel_height = info.pixel_width
        chans = (attrs.get("omero") or {}).get("channels") or []
        if chans and all(c.get("label") for c in chans) and len(chans) == info.effective_size_c:
            info.channel_names = [str(c["label"]) for c in chans]
        info.metadata = {"version": str(m0.get("version", "")), "levels": len(datasets)}
        return [info]
    # bioformats2raw layout: one OME-Zarr image per numbered sub-group
    subs = []
    try:
        subs = [(nm, g) for nm, g in node.groups() if _zarr_group_attrs(g).get("multiscales")]
    except Exception:
        pass
    if subs:
        subs.sort(key=lambda kv: (not kv[0].isdigit(), int(kv[0]) if kv[0].isdigit() else kv[0]))
        for i, (nm, _g) in enumerate(subs):
            out += _list_series_zarr(p / nm, index_offset + i, name=f"{strip_image_ext(p.name)} #{nm}")
        for info in out:
            info.metadata["bioformats2raw"] = True
        return out
    found = []
    stack = [(node, "")]
    while stack:
        g, prefix = stack.pop()
        try:
            arrays, groups = list(g.arrays()), list(g.groups())
        except Exception:
            continue
        for nm, arr in arrays:
            if 2 <= arr.ndim <= 5:
                found.append((f"{prefix}{nm}", arr))
        for nm, sub in groups:
            stack.append((sub, f"{prefix}{nm}/"))
    found.sort(key=lambda kv: kv[0])
    for i, (key, arr) in enumerate(found):
        axes = {2: "YX", 3: "ZYX", 4: "ZCYX", 5: "TZCYX"}.get(arr.ndim, "ZYX")
        info = _series_info_from_axes(index_offset + i, axes, arr.shape, arr.dtype)
        info.name, info.format = key, "Zarr"
        out.append(info)
    if not out:
        raise ValueError(f"No array found in {p.name}")
    return out


def _bioio_open(p: Path):
    BioImage = None
    try:
        from bioio import BioImage  # type: ignore
    except ImportError:
        try:
            from aicsimageio import AICSImage as BioImage  # type: ignore
        except ImportError:
            pass
    if BioImage is None:
        return None
    return BioImage(str(p))


def _list_series_bioio(p: Path, ext: str) -> list[SeriesInfo]:
    bi = _bioio_open(p)
    if bi is None:
        raise RuntimeError(f"No reader for {ext or p.name!r}: pip install bioio (plus a bioio reader plugin for this format)")
    out: list[SeriesInfo] = []
    scenes = list(getattr(bi, "scenes", ()) or ())
    reader = type(getattr(bi, "reader", bi)).__name__
    for i in range(max(1, len(scenes))):
        try:
            bi.set_scene(i)
        except Exception:
            pass
        dims = bi.dims
        axes = str(getattr(dims, "order", "TCZYX"))
        shape = tuple(int(v) for v in getattr(dims, "shape", bi.shape))
        info = _series_info_from_axes(i, axes, shape, bi.dtype)
        info.name = str(scenes[i]) if i < len(scenes) else strip_image_ext(p.name)
        info.format = f"bioio ({reader})"
        try:
            pps = bi.physical_pixel_sizes
            if pps.X:
                info.pixel_width, info.pixel_height, info.pixel_depth, info.unit = float(pps.X), float(pps.Y or pps.X), float(pps.Z or 1.0), "µm"
        except Exception:
            pass
        try:
            info.channel_names = [str(n) for n in bi.channel_names]
        except Exception:
            pass
        out.append(info)
    return out


def _bridge():
    """The optional Java Bio-Formats bridge module (imported lazily so tests can swap its backend)."""
    from .readers import bioformats_bridge
    return bioformats_bridge


def _native_reader(ext: str):
    import importlib
    return importlib.import_module(f".readers.{READER_BY_EXT[ext]}", __package__)


def _series_info_from_native(info: NativeSeries) -> SeriesInfo:
    """A :class:`SeriesInfo` from what a native reader / the Java bridge reports (OME-XML applied when it carries one)."""
    si = _series_info_from_axes(info.index, info.axes, info.shape, info.dtype)
    si.name, si.format = info.name, info.format
    si.rgb = si.rgb or bool(info.rgb)
    if si.rgb:
        si.size_c = 3
    si.channel_names = [] if si.rgb else list(info.channel_names)
    if info.pixel_width:
        si.pixel_width, si.pixel_height, si.unit = float(info.pixel_width), float(info.pixel_height or info.pixel_width), _norm_unit(info.unit or "µm")
        si.pixel_depth = float(info.pixel_depth) if info.pixel_depth else None
    if info.frame_interval:
        si.time_increment, si.time_unit = float(info.frame_interval), info.time_unit or "sec"
    levels = info.metadata.get("levels")
    if isinstance(levels, list) and levels:
        si.levels = [(int(h), int(w)) for h, w in levels]
    si.metadata = {k: v for k, v in info.metadata.items() if isinstance(v, (str, int, float, bool)) and k != "ome_xml"}
    xml = info.metadata.get("ome_xml")
    if xml:
        images = _ome_images_info(xml)
        if images:
            ome = images[min(info.index, len(images) - 1)]
            _apply_ome_pixels(si, ome.get("pixels", {}))
            if ome.get("channel_names") and not si.rgb:
                si.channel_names = [n or f"C{i + 1}" for i, n in enumerate(ome["channel_names"])]
            si.planes = ome.get("planes", [])
            if ome.get("acquisition_date"):
                si.metadata["AcquisitionDate"] = ome["acquisition_date"]
    return si


def _lumen_from_native(data: np.ndarray, info: NativeSeries, name: str, p: Path, cal: Calibration | None = None, names: list[str] | None = None,
                       meta: dict[str, Any] | None = None) -> LumenImage:
    if cal is None:
        cal = Calibration()
        if info.pixel_width:
            cal.pixel_width, cal.pixel_height, cal.unit = float(info.pixel_width), float(info.pixel_height or info.pixel_width), _norm_unit(info.unit or "µm")
            if info.pixel_depth:
                cal.pixel_depth = float(info.pixel_depth)
        if info.frame_interval:
            cal.frame_interval, cal.time_unit = float(info.frame_interval), info.time_unit or "sec"
    if meta is None:
        meta = {k: v for k, v in info.metadata.items() if k not in ("ome_xml", "series_count")}
        meta["format"] = info.format
        n_series = int(info.metadata.get("series_count", 1) or 1)
        if n_series > 1:
            meta.update({"series_count": n_series, "series": info.index, "series_name": info.name,
                         "source": {"kind": "file", "path": str(p), "series": info.index}})
    names = info.channel_names if names is None else names
    axes = info.axes if len(info.axes) == data.ndim else ""
    rgb = bool(info.rgb) or ("S" in axes and data.shape[axes.index("S")] in (3, 4) and data.dtype == np.uint8)
    if axes and "S" in axes and "C" in axes:
        # interleaved samples next to a channel axis (RGB planes of a multi-channel file): one channel keeps RGB,
        # several fold their samples into channels
        ci, si = axes.index("C"), axes.index("S")
        if data.shape[ci] == 1:
            data = np.take(data, 0, axis=ci)
            axes = axes[:ci] + axes[ci + 1:]
        else:
            data = np.moveaxis(data, si, ci + 1)
            data = data.reshape(data.shape[:ci] + (data.shape[ci] * data.shape[ci + 1],) + data.shape[ci + 2:])
            axes = axes.replace("S", "")
            rgb = False
            names = []
    return LumenImage(data, name=name, axes=axes or None, rgb=rgb, calibration=cal,
                      channel_names=list(names) if names and not rgb else None, metadata=meta, path=str(p))


def _list_series_native(p: Path, ext: str) -> list[SeriesInfo]:
    return [_series_info_from_native(i) for i in _native_reader(ext).list_series(p)]


def _read_native(p: Path, name: str, ext: str, series: int | None) -> LumenImage:
    data, info = _native_reader(ext).read(p, int(series or 0))
    return _lumen_from_native(data, info, name, p)


def _list_series_bridge(p: Path) -> list[SeriesInfo]:
    return [_series_info_from_native(i) for i in _bridge().list_series(p)]


def _read_bridge(p: Path, name: str, series: int | None = None, level: int = 0) -> LumenImage:
    data, cal, names, meta = _bridge().read(p, int(series or 0), int(level or 0))
    axes = str(meta.get("axes") or "")
    info = NativeSeries(index=int(meta.get("series", 0) or 0), name=str(meta.get("series_name") or strip_image_ext(p.name)), shape=tuple(data.shape),
                        axes=axes if len(axes) == data.ndim else {5: "TZCYX", 6: "TZCYXS"}.get(data.ndim, ""), dtype=data.dtype,
                        format=str(meta.get("format") or "Bio-Formats (Java bridge)"))
    meta = {k: v for k, v in meta.items() if k not in ("axes", "ome_xml")}
    meta.setdefault("format", info.format)
    return _lumen_from_native(data, info, name, p, cal=cal, names=names, meta=meta)


def _no_reader_message(p: Path, ext: str, errors: list[str]) -> str:
    st = _bridge().status()
    where = "Java Bio-Formats bridge is installed but could not read it" if st["available"] else f"Java Bio-Formats bridge not available ({st['detail']})"
    msg = (f"No reader for {ext or p.name!r}: {where}. Install the bridge – {st['install']} – or a bioio reader plugin for this format, "
           f"or convert the file with bfconvert / bioformats2raw to OME-TIFF / OME-Zarr")
    return msg + ("; tried: " + "; ".join(errors) if errors else "")


def _bioio_installed() -> bool:
    import importlib.util
    import sys
    for name in ("bioio", "aicsimageio"):
        if name in sys.modules:
            return True
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError):
            pass
    return False


def _list_series_fallback(p: Path, ext: str) -> list[SeriesInfo]:
    """Unknown extensions: bioio (if it has a plugin for the file), then the Java Bio-Formats bridge."""
    errors: list[str] = []
    if not _bioio_installed():
        errors.append("bioio not installed")
    else:
        try:
            return _list_series_bioio(p, ext)
        except Exception as e:  # noqa: BLE001 – every failure falls through to the bridge
            errors.append(f"bioio: {e}")
    if _bridge().available():
        try:
            return _list_series_bridge(p)
        except Exception as e:  # noqa: BLE001
            errors.append(f"Java Bio-Formats: {e}")
    raise RuntimeError(_no_reader_message(p, ext, errors))


def _read_fallback(p: Path, name: str, ext: str, series: int | None = None, level: int = 0) -> LumenImage:
    """Unknown extensions: bioio (if it has a plugin for the file), then the Java Bio-Formats bridge, else a clear error."""
    errors: list[str] = []
    if not _bioio_installed():
        errors.append("bioio not installed")
    else:
        try:
            return _read_bioio(p, name, ext, series=series)
        except Exception as e:  # noqa: BLE001
            errors.append(f"bioio: {e}")
    if _bridge().available():
        try:
            return _read_bridge(p, name, series=series, level=level)
        except Exception as e:  # noqa: BLE001
            errors.append(f"Java Bio-Formats: {e}")
    raise RuntimeError(_no_reader_message(p, ext, errors))


def list_series(path: str | os.PathLike, dataset: str | None = None) -> list[SeriesInfo]:
    """The image series of a file (Bio-Formats' series list) without reading the pixels where the reader allows it.

    Multi-series OME-TIFF, pyramidal / whole-slide TIFFs (NDPI, SVS, SCN, BIF, Philips, OME pyramids – one entry per
    tifffile series, resolution levels in ``levels``), ND2 positions, CZI scenes, LIF images, HDF5 datasets / Imaris
    resolution levels, OME-Zarr multiscales (and bioformats2raw collections) and bioio scenes.  Every other format
    reports one series (read through the ordinary reader).
    """
    p = Path(path)
    if p.is_dir():
        if _looks_like_zarr(p):
            return _list_series_zarr(p)
        if _looks_like_dicom_dir(p):
            return [SeriesInfo.from_image(_read_dicom_dir(p, p.name, dicom_reader_options()), format="DICOM")]
        return [SeriesInfo.from_image(read_sequence(p), format="Image sequence")]
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = _full_ext(p)
    if ext in (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".lsm", ".stk"):
        return _list_series_tiff(p)
    if ext == ".nd2":
        return _list_series_nd2(p)
    if ext == ".czi":
        return _list_series_czi(p)
    if ext == ".lif":
        return _list_series_lif(p)
    if ext in (".h5", ".hdf5", ".ims", ".hdf"):
        if dataset is not None:
            return [SeriesInfo.from_image(_read_hdf5(p, p.name, dataset), name=dataset, format="HDF5")]
        return _list_series_hdf5(p)
    if ext in READER_BY_EXT:
        try:
            return _list_series_native(p, ext)
        except Exception as e:  # noqa: BLE001 – a file the native reader cannot parse may still open through Java
            if not _bridge().available():
                raise
            try:
                return _list_series_bridge(p)
            except Exception:
                raise e from None
    native = ext in IMAGE_EXTS or ext in (".txt", ".csv", ".tsv") or _is_dicom_file(p)
    if not native:
        return _list_series_fallback(p, ext)
    img = read_image(p)
    return [SeriesInfo.from_image(img, format=_FORMAT_BY_EXT.get(ext, "Video" if ext in VIDEO_EXTS else ext.lstrip(".").upper()))]


def ome_xml_from_series(infos: list[SeriesInfo], creator: str = "Lumen") -> str:
    """A minimal OME-XML document describing ``infos`` (what Bio-Formats synthesises for formats without OME metadata)."""
    import xml.etree.ElementTree as ET
    ns = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
    ET.register_namespace("", ns)
    root = ET.Element(f"{{{ns}}}OME", {"Creator": creator})
    for info in infos:
        image = ET.SubElement(root, f"{{{ns}}}Image", {"ID": f"Image:{info.index}", "Name": info.name})
        if info.metadata.get("AcquisitionDate"):
            ET.SubElement(image, f"{{{ns}}}AcquisitionDate").text = str(info.metadata["AcquisitionDate"])
        if info.metadata.get("Description"):
            ET.SubElement(image, f"{{{ns}}}Description").text = str(info.metadata["Description"])
        attrs = {"ID": f"Pixels:{info.index}", "DimensionOrder": info.dimension_order, "Type": info.pixel_type, "SizeX": str(info.size_x),
                 "SizeY": str(info.size_y), "SizeZ": str(info.size_z), "SizeC": str(info.size_c), "SizeT": str(info.size_t), "Interleaved": str(info.rgb).lower()}
        if info.pixel_width is not None:
            unit = _OME_UNITS.get(info.unit, info.unit or "pixel")
            attrs.update({"PhysicalSizeX": repr(float(info.pixel_width)), "PhysicalSizeXUnit": unit,
                          "PhysicalSizeY": repr(float(info.pixel_height if info.pixel_height is not None else info.pixel_width)), "PhysicalSizeYUnit": unit})
            if info.pixel_depth is not None:
                attrs.update({"PhysicalSizeZ": repr(float(info.pixel_depth)), "PhysicalSizeZUnit": unit})
        if info.time_increment is not None:
            attrs.update({"TimeIncrement": repr(float(info.time_increment)), "TimeIncrementUnit": _TIME_TO_OME.get(info.time_unit, info.time_unit or "s")})
        pixels = ET.SubElement(image, f"{{{ns}}}Pixels", attrs)
        for c in range(info.effective_size_c):
            ch = {"ID": f"Channel:{info.index}:{c}", "SamplesPerPixel": str(info.rgb_channel_count)}
            if c < len(info.channel_names) and info.channel_names[c]:
                ch["Name"] = info.channel_names[c]
            ET.SubElement(pixels, f"{{{ns}}}Channel", ch)
        for pl in info.planes:
            a = {"TheZ": str(pl.get("z", 0)), "TheC": str(pl.get("c", 0)), "TheT": str(pl.get("t", 0))}
            for key, name in (("delta_t", "DeltaT"), ("exposure_time", "ExposureTime"), ("position_x", "PositionX"), ("position_y", "PositionY"), ("position_z", "PositionZ")):
                if pl.get(key) is not None:
                    a[name] = repr(float(pl[key]))
            ET.SubElement(pixels, f"{{{ns}}}Plane", a)
    try:
        ET.indent(root)
    except Exception:
        pass
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def ome_xml_of(path: str | os.PathLike, infos: list[SeriesInfo] | None = None) -> str:
    """The OME-XML of a file: the embedded document of an OME-TIFF (re-indented), otherwise one synthesised from its series."""
    p = Path(path)
    if _full_ext(p) in (".tif", ".tiff", ".ome.tif", ".ome.tiff"):
        try:
            import tifffile
            with tifffile.TiffFile(str(p)) as tf:
                xml = getattr(tf, "ome_metadata", None)
            if xml:
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(xml)
                    ET.indent(root)
                    ns = root.tag.split("}")[0].strip("{") if "}" in root.tag else ""
                    if ns:
                        ET.register_namespace("", ns)
                    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
                except Exception:
                    return str(xml)
        except Exception:
            pass
    return ome_xml_from_series(infos if infos is not None else list_series(p))


def group_file_pattern(path: str | os.PathLike, pattern: str | None = None) -> tuple[Path, str, str | None]:
    """Bio-Formats "group files with similar names": ``(folder, regex, axis)`` for the files that belong with ``path``.

    Without ``pattern`` the last number in the file name is the varying index (``img_t003.tif`` → ``^img_t\\d+\\.tif$``);
    a Bio-Formats pattern (``img_t<1-30>.tif``) or a regular expression / glob can be given instead.  ``axis`` is the
    guessed dimension of the numbering ('T', 'Z', 'C' from the letter before the number) or None when unknown.
    """
    p = Path(path).expanduser()
    folder = p.parent if not p.is_dir() else p
    name = p.name if not p.is_dir() else ""

    def guess(prefix: str) -> str | None:
        s = prefix.rstrip("_- ").lower()
        if s.endswith(("ch", "_c", "-c", "w")) or s[-1:] == "c":
            return "C"
        if s[-1:] == "t" or s.endswith(("time", "frame")):
            return "T"
        if s[-1:] == "z" or s.endswith(("slice", "plane")):
            return "Z"
        return None

    if pattern:
        pat = Path(pattern).name
        if "<" in pat:
            parts = re.split(r"(<[^>]*>)", pat)
            regex = "^" + "".join(r"\d+" if part.startswith("<") else re.escape(part) for part in parts) + "$"
            axis = guess(pat.split("<", 1)[0])
        else:
            regex = compile_file_pattern(pat)
            axis = None
        return folder, regex, axis
    m = None
    for m in re.finditer(r"\d+", name):
        pass
    if m is None:
        raise ValueError(f"{name!r} has no number to group files on – give a pattern such as name_<1-10>.tif")
    regex = "^" + re.escape(name[:m.start()]) + r"\d+" + re.escape(name[m.end():]) + "$"
    return folder, regex, guess(name[:m.start()])


# ============================================================================ virtual (lazy) TIFF stacks
class VirtualStack(LazyArray):
    """Lazily read (T, Z, C, Y, X) planes of a TIFF series – File ▸ Import ▸ TIFF Virtual Stack for compressed / tiled files
    (uncompressed files use ``numpy.memmap`` instead).  The TIFF specialisation of :class:`lumen.core.lazy.LazyArray`:
    planes come from a :class:`lumen.core.lazy.TiffPageSource` (one page decode per plane, through the shared plane
    cache) or, for big-endian memmaps, from the mapping itself byte-swapped one plane at a time (``_mm`` / ``memmapped``).

    Behaves like a read-mostly 5-D array: ``vs[t, z, c]`` reads one plane from the file, slices read only the planes
    they cover, and anything that needs the whole array (``np.asarray``, ``copy()``, in-place writes, arithmetic,
    reductions) materialises it in memory – copy-on-write.  Plane commands and undo snapshots copy the data, so the
    first modifying command turns the image into an ordinary in-memory image.
    """

    def __init__(self, path: str | os.PathLike, series: int, level: int, axes: str, shape: tuple[int, ...], dtype: Any, rgb: bool,
                 cache_planes: int = 32) -> None:
        super().__init__(TiffPageSource(path, series, level, axes, shape, dtype, rgb))
        self.path = str(path)
        self.series_index = int(series)
        self.level = int(level)
        self._mm_view: Any = None

    @property
    def _mm(self) -> Any:
        """A (T, Z, C, Y, X) view of a numpy memmap in a foreign byte order (big-endian 'MM' files – ImageJ/Fiji's
        default): planes are read from the mapping and byte-swapped one at a time instead of from the pages (a page
        read would return zeros for the planes of a single-IFD "truncated" ImageJ hyperstack)."""
        return self.__dict__.get("_mm_view")

    @_mm.setter
    def _mm(self, view: Any) -> None:
        self._mm_view = view
        if view is not None:
            self._source.close()
            self._source = view_source(view, self.rgb, path=self.path)

    @property
    def memmapped(self) -> bool:
        """True while the planes come from a memory-mapped file (foreign byte order, swapped per plane on read)."""
        return self._mm_view is not None and self._mat is None

    def materialize(self) -> np.ndarray:
        arr = super().materialize()
        self._mm_view = None  # release the mapping once the stack lives in memory
        return arr

    def __repr__(self) -> str:  # pragma: no cover
        return f"<VirtualStack {self.shape} {self.dtype} {'in memory' if self._mat is not None else 'lazy'} {self.path!r}>"


def is_virtual(data: Any) -> bool:
    """True for pixel data that still lives in the file: a lazy array (VirtualStack, HDF5 / Zarr / NPY LazyArray) that was
    not materialised, or a numpy memmap view of an uncompressed TIFF / NPY file."""
    return _lazy_is_virtual(data)


def _view_5d(data: np.ndarray, axes: str, rgb: bool) -> np.ndarray:
    """(T, Z, C, Y, X) *view* of a reader array (no copy – for memmaps), folding axes like LumenImage.to_canonical."""
    axes = axes.upper()
    if "S" in axes:
        si = axes.index("S")
        if data.shape[si] == 4 and rgb:
            data = data[(slice(None),) * si + (slice(0, 3),)]
        axes = axes.replace("S", "C")
    for ch in axes:
        if ch not in AXES:
            axes = axes.replace(ch, "Z" if "Z" not in axes else "T")
    if len(set(axes)) != len(axes):
        raise ValueError(f"axes {axes!r} cannot be mapped onto T/Z/C without merging")
    for ch in AXES:
        if ch not in axes:
            data = np.expand_dims(data, 0)
            axes = ch + axes
    return np.transpose(data, [axes.index(ch) for ch in AXES])


def _tiff_lazy_5d(p: Path, series_index: int, level: int, lvl: Any) -> tuple[Any, bool] | None:
    """A lazy (T, Z, C, Y, X) array for a TIFF series level: a copy-on-write memmap for uncompressed, contiguous data,
    else a VirtualStack reading pages on demand; None when the layout cannot be mapped (the caller reads normally)."""
    import tifffile
    axes, shape, dtype = str(lvl.axes), tuple(int(v) for v in lvl.shape), np.dtype(lvl.dtype)
    rgb = "S" in axes and shape[axes.index("S")] in (3, 4) and dtype == np.uint8
    if dtype in (np.uint8, np.uint16, np.float32):
        # (tifffile reports the series dtype in native order even for big-endian files, so the memmap's own dtype decides)
        try:
            m = tifffile.memmap(str(p), series=series_index, level=level, mode="c")
            if tuple(m.shape) != shape:
                m = m.reshape(shape)
            view = _view_5d(m, axes, rgb)
            if m.dtype.isnative:  # single-byte dtypes report '|' and count as native
                return view, rgb
            # big-endian ('MM', the ImageJ/Fiji default) or otherwise foreign byte order: a '>u2' / '>f4' array must
            # never reach LumenImage (bit_depth / dtype checks expect native types), and byte-swapping the whole
            # mapping would copy it – so keep the mapping and hand out native planes on demand through a VirtualStack
            # (page reads are no alternative: a single-IFD "truncated" ImageJ hyperstack only exposes its first plane)
            vs = VirtualStack(p, series_index, level, axes, shape, dtype, rgb)
            vs._mm = view
            return vs, rgb
        except (ValueError, TypeError, OSError, NotImplementedError, AttributeError):
            pass
    try:
        return VirtualStack(p, series_index, level, axes, shape, dtype, rgb), rgb
    except ValueError:
        return None


def _lumen_from_lazy(data: Any, name: str, rgb: bool, cal: Calibration, meta: dict[str, Any], path: Path | str | None,
                     channel_names: list[str] | None) -> LumenImage:
    """A LumenImage over lazy pixel data (memmap view / VirtualStack): like ``LumenImage(...)`` but never copies the pixels."""
    img = LumenImage.__new__(LumenImage)
    img.id = uuid.uuid4().hex[:10]
    img.name = name
    img.path = str(path) if path is not None else None
    img.rgb = bool(rgb)
    img.metadata = meta if meta is not None else {}
    img.calibration = cal or Calibration()
    img.data = data
    img.display = Display()
    img.lut_is_binary = False
    img.overlay = []
    img.rev = 0
    img.reset_display(channel_names)
    return img


def _guess_axes(shape: tuple[int, ...], dtype: Any, rgb: bool = False) -> str:
    """The axes ``LumenImage.to_canonical`` assumes for a reader array without an axes string."""
    nd = len(shape)
    if nd == 2:
        return "YX"
    if nd == 3:
        return "YXS" if (rgb or (shape[-1] in (3, 4) and np.dtype(dtype) == np.uint8)) else "ZYX"
    if nd == 4:
        return "ZCYX"
    if nd == 5:
        return "TZCYX"
    raise ValueError(f"Unsupported image dimensionality: {shape}")


def _read_npy_lazy(p: Path, name: str) -> LumenImage | None:
    """A ``.npy`` file without reading it: a copy-on-write memmap view for supported native dtypes, else a LazyArray
    that coerces planes on read; None when the file cannot be mapped (the caller reads it normally)."""
    try:
        m = np.load(str(p), mmap_mode="c")
    except (ValueError, OSError):  # pickled objects, a damaged header …
        return None
    if not isinstance(m, np.memmap) or not 2 <= m.ndim <= 5:
        return None
    axes = "TZCYX" if m.ndim == 5 else _guess_axes(m.shape, m.dtype, False)
    rgb = "S" in axes
    meta: dict[str, Any] = {"source": {"kind": "file", "path": str(p), "virtual": True}}
    shape, dtype = tuple(int(v) for v in m.shape), m.dtype
    try:
        if dtype.isnative and dtype in (np.uint8, np.uint16, np.float32):
            return _lumen_from_lazy(_view_5d(m, axes, rgb), name, rgb, Calibration(), meta, p, None)
        del m  # a foreign byte order or an unsupported dtype: planes are coerced one at a time
        return _lumen_from_lazy(LazyArray(npy_source(p, axes, shape, dtype, rgb)), name, rgb, Calibration(), meta, p, None)
    except ValueError:
        return None


def _decoded_bytes(shape: tuple[int, ...], dtype: Any) -> int:
    """Size of the in-memory array Lumen builds from a reader array of ``shape`` / ``dtype`` (int16 / int32 / float64
    data is promoted to float32 on read)."""
    item = LumenImage.coerce_dtype(np.zeros(1, np.dtype(dtype))).dtype.itemsize
    return int(np.prod([int(v) for v in shape])) * int(item)


def _series_bytes(infos: list[SeriesInfo], series: int | None, level: int, dataset: str | None) -> int | None:
    if not infos:
        return None
    info = next((i for i in infos if i.name == dataset), None) if dataset is not None else None
    if info is None:
        info = infos[min(max(int(series or 0), 0), len(infos) - 1)]
    ny, nx = info.size_y, info.size_x
    li = int(level or 0)
    if info.levels and 0 <= li < len(info.levels):
        ny, nx = info.levels[li]
    return _decoded_bytes((max(1, info.size_t), max(1, info.size_z), max(1, info.size_c), ny, nx), info.dtype)


def estimate_decoded_bytes(path: str | os.PathLike, *, series: int | None = None, level: int = 0, dataset: str | None = None) -> int | None:
    """The decoded (T, Z, C, Y, X) size in bytes of what :func:`read_image` would build, from the file header alone –
    for the formats that can be opened lazily (TIFF family, ``.npy``, HDF5 / Imaris, Zarr / OME-Zarr); None otherwise."""
    p = Path(path)
    try:
        if p.is_dir():
            return _series_bytes(_list_series_zarr(p), series, level, dataset) if _looks_like_zarr(p) else None
        ext = _full_ext(p)
        if ext == ".npy":
            m = np.load(str(p), mmap_mode="r")
            return _decoded_bytes(m.shape, m.dtype) if isinstance(m, np.memmap) else None
        if ext in (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".lsm", ".stk"):
            import tifffile
            with tifffile.TiffFile(str(p)) as tf:
                s = tf.series[min(max(int(series or 0), 0), len(tf.series) - 1)]
                levels = s.levels or [s]
                lvl = levels[min(max(int(level or 0), 0), len(levels) - 1)]
                return _decoded_bytes(tuple(lvl.shape), lvl.dtype)
        if ext in (".h5", ".hdf5", ".ims", ".hdf"):
            return _series_bytes(_list_series_hdf5(p), series, level, dataset)
    except Exception:  # noqa: BLE001 – an unreadable header: read_image reports the real error
        return None
    return None


def should_open_virtual(path: str | os.PathLike, *, series: int | None = None, level: int = 0, dataset: str | None = None,
                        limit: int | None = None) -> bool:
    """True when the file can be opened lazily and its decoded size exceeds ``limit`` (default: the ``auto_virtual_mb``
    preference – half of the physical RAM unless changed, 0 = never; see :func:`lumen.core.lazy.auto_virtual_bytes`)."""
    lim = auto_virtual_bytes() if limit is None else int(limit)
    if lim >= sys.maxsize:
        return False
    est = estimate_decoded_bytes(path, series=series, level=level, dataset=dataset)
    return est is not None and est > lim


# ============================================================================ TIFF
def _ij_metadata_to_calibration(tf) -> tuple[Calibration, dict[str, Any], list[str] | None, dict[str, Any]]:
    cal = Calibration()
    meta: dict[str, Any] = {}
    channel_names = None
    display: dict[str, Any] = {}
    page = tf.pages[0]
    try:
        xres = page.tags.get("XResolution")
        yres = page.tags.get("YResolution")
        if xres is not None and yres is not None:
            xn, xd = xres.value
            yn, yd = yres.value
            if xn and yn:
                cal.pixel_width = xd / xn
                cal.pixel_height = yd / yn
    except Exception:
        pass
    ij = tf.imagej_metadata or {}
    if ij:
        meta.update({k: v for k, v in ij.items() if isinstance(v, (str, int, float, bool))})
        if "unit" in ij:
            cal.unit = _norm_unit(str(ij["unit"]))
        if "spacing" in ij:
            cal.pixel_depth = float(ij["spacing"])
        if "finterval" in ij:
            cal.frame_interval = float(ij["finterval"])
        if ij.get("tunit"):  # ImageJ writes tunit only when it is not "sec"
            cal.time_unit = str(ij["tunit"])
        if "Labels" in ij and isinstance(ij["Labels"], (list, tuple)):
            # one per plane (ImageJ order: C fastest, then Z, then T): channel names when consistent per channel,
            # otherwise slice labels – _apply_ij_display decides once the plane count is known
            display["labels"] = [str(v) for v in ij["Labels"]]
        if "Ranges" in ij:
            display["ranges"] = list(ij["Ranges"])
        if "min" in ij and "max" in ij:
            # ImageJ stores the display range of a non-composite image as min=/max= (Ranges= is only for CompositeImages)
            try:
                display["min"], display["max"] = float(ij["min"]), float(ij["max"])
            except (TypeError, ValueError):
                pass
        if "LUTs" in ij:
            display["ij_luts"] = ij["LUTs"]
        if "mode" in ij:
            display["mode"] = ij["mode"]
    ome = getattr(tf, "ome_metadata", None)
    if ome:
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(ome)
            ns = {"ome": root.tag.split("}")[0].strip("{")} if "}" in root.tag else {}
            px = root.find(".//ome:Pixels", ns) if ns else root.find(".//Pixels")
            if px is not None:
                if px.get("PhysicalSizeX"):
                    cal.pixel_width = float(px.get("PhysicalSizeX"))
                    cal.unit = _norm_unit(px.get("PhysicalSizeXUnit", "µm"))
                if px.get("PhysicalSizeY"):
                    cal.pixel_height = float(px.get("PhysicalSizeY"))
                if px.get("PhysicalSizeZ"):
                    cal.pixel_depth = float(px.get("PhysicalSizeZ"))
                if px.get("TimeIncrement"):
                    cal.frame_interval = float(px.get("TimeIncrement"))
                    tu = px.get("TimeIncrementUnit")
                    if tu:
                        cal.time_unit = _TIME_FROM_OME.get(tu, tu)
                chans = px.findall("ome:Channel", ns) if ns else px.findall("Channel")
                names = [c.get("Name") for c in chans]
                if names and all(names):
                    channel_names = names
                colors: list[int | None] = []
                for c in chans:
                    try:
                        colors.append(int(c.get("Color")) if c.get("Color") not in (None, "") else None)
                    except ValueError:
                        colors.append(None)
                if any(v is not None for v in colors):
                    display["ome_colors"] = colors
            # Lumen's own display state: the MapAnnotation in our namespace that the (first) Image references
            image = root.find("ome:Image", ns) if ns else root.find("Image")
            refs = {a.get("ID") for a in (image.findall("ome:AnnotationRef", ns) if ns else image.findall("AnnotationRef"))} if image is not None else set()
            for ann in (root.findall(".//ome:MapAnnotation", ns) if ns else root.findall(".//MapAnnotation")):
                if ann.get("Namespace") != _LUMEN_OME_NS or (refs and ann.get("ID") not in refs):
                    continue
                kv = {m.get("K"): (m.text or "") for m in ann.iter() if m.tag.rsplit("}", 1)[-1] == "M" and m.get("K")}
                display["lumen"] = kv
                # image metadata embedded by the Bio-Formats exporter (embed_metadata): "meta.<key>" entries
                for k, v in kv.items():
                    if k.startswith("meta.") and k[5:]:
                        meta[k[5:]] = _parse_scalar(v)
                break
            desc = image.find("ome:Description", ns) if (image is not None and ns) else (image.find("Description") if image is not None else None)
            if desc is not None and (desc.text or "").strip():
                meta.setdefault("description", desc.text.strip())
        except Exception:
            pass
    # Zeiss LSM: voxel sizes are in metres
    lsm = getattr(tf, "lsm_metadata", None)
    if lsm:
        try:
            if lsm.get("VoxelSizeX"):
                cal.pixel_width = float(lsm["VoxelSizeX"]) * 1e6
                cal.pixel_height = float(lsm.get("VoxelSizeY", lsm["VoxelSizeX"])) * 1e6
                cal.pixel_depth = float(lsm.get("VoxelSizeZ", 0) or 0) * 1e6 or cal.pixel_depth
                cal.unit = "µm"
            if lsm.get("TimeIntervall"):
                cal.frame_interval = float(lsm["TimeIntervall"])
            names = (lsm.get("ChannelColors") or {}).get("ColorNames")
            if names and all(names):
                channel_names = [str(n) for n in names]
            for k in ("DimensionX", "DimensionY", "DimensionZ", "DimensionChannels", "DimensionTime", "ScanType"):
                if k in lsm:
                    meta[f"lsm_{k}"] = lsm[k]
        except Exception:
            pass
    # MetaMorph STK
    stk = getattr(tf, "stk_metadata", None)
    if stk:
        try:
            if stk.get("XCalibration"):
                cal.pixel_width = float(stk["XCalibration"])
                cal.pixel_height = float(stk.get("YCalibration") or stk["XCalibration"])
                cal.unit = _norm_unit(stk.get("CalibrationUnits", "µm"))
            zd = stk.get("ZDistance")
            if zd is not None:
                zd = np.atleast_1d(np.asarray(zd, dtype=float))
                nz = [float(v) for v in zd if v != 0]
                if nz:
                    cal.pixel_depth = float(np.median(np.abs(nz)))
            if stk.get("Name"):
                meta["stk_name"] = str(stk["Name"])
            wl = stk.get("Wavelengths")
            if wl is not None:
                meta["wavelengths"] = [float(v) for v in np.atleast_1d(wl)[:16]]
        except Exception:
            pass
    # MetaMorph single-plane TIFFs
    ms = getattr(tf, "metaseries_metadata", None)
    if ms:
        try:
            pi = ms.get("PlaneInfo", {})
            if pi.get("spatial-calibration-x"):
                cal.pixel_width = float(pi["spatial-calibration-x"])
                cal.pixel_height = float(pi.get("spatial-calibration-y", pi["spatial-calibration-x"]))
                cal.unit = _norm_unit(pi.get("spatial-calibration-units", "µm"))
        except Exception:
            pass
    # Micro-Manager
    mm = getattr(tf, "micromanager_metadata", None)
    if mm:
        try:
            summ = mm.get("Summary", {})
            if summ.get("PixelSize_um"):
                cal.pixel_width = cal.pixel_height = float(summ["PixelSize_um"])
                cal.unit = "µm"
            if summ.get("z-step_um"):
                cal.pixel_depth = abs(float(summ["z-step_um"])) or cal.pixel_depth
            if summ.get("Interval_ms"):
                cal.frame_interval = float(summ["Interval_ms"]) / 1000.0
            if summ.get("ChNames"):
                channel_names = [str(n) for n in summ["ChNames"]]
        except Exception:
            pass
    if cal.unit in ("pixel", "") and (cal.pixel_width != 1.0 or cal.pixel_height != 1.0):
        # Plain TIFF (scanner / PIL / camera software): only X/YResolution are known, so follow
        # ImageJ's TiffDecoder + FileOpener on tag 296 ResolutionUnit (1 = none, 2 = inch, 3 = cm).
        ru = None
        try:
            ru_tag = page.tags.get("ResolutionUnit")
            ru = int(ru_tag.value) if ru_tag is not None else None
        except Exception:
            pass
        if ru == 2:
            if abs(cal.pixel_width - 1.0 / 72.0) < 1e-12:
                cal.pixel_width = cal.pixel_height = 1.0  # 72 dpi screen default: ImageJ treats it as uncalibrated
            else:
                cal.unit = "inch"
        elif ru == 3:
            if cal.pixel_width <= 0.001:  # ImageJ (Prefs.convertToMicrons, default on): <= 10 µm pixels stored in cm -> µm
                cal.pixel_width *= 10000.0
                cal.pixel_height *= 10000.0
                cal.unit = "µm"
            else:
                cal.unit = "cm"
        elif ru == 1:
            cal.unit = "unit"  # NONE: ImageJ keeps the scale with a blank unit
        else:
            cal.pixel_width = cal.pixel_height = 1.0  # tag absent: ImageJ leaves the image uncalibrated
    return cal, meta, channel_names, display


def _parse_scalar(s: str) -> Any:
    """'12' → 12, '1.5' → 1.5, 'true' → True, anything else stays text (round-trips the metadata map of an OME-TIFF)."""
    t = s.strip()
    if t.lower() in ("true", "false"):
        return t.lower() == "true"
    try:
        return int(t) if re.fullmatch(r"[-+]?\d+", t) else float(t) if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+|\d+)([eE][-+]?\d+)?", t) else s
    except ValueError:
        return s


def _lut_name_from_ij(lut: Any) -> str | None:
    """Primary LUT name from the colour at the top of an ImageJ colour table (the old heuristic)."""
    try:
        arr = np.asarray(lut)
        if arr.shape[0] == 3:
            last = arr[:, -1]
        else:
            last = arr[-1]
        r, g, b = (int(v) > 127 for v in last)
        key = (r, g, b)
        return {(1, 0, 0): "Red", (0, 1, 0): "Green", (0, 0, 1): "Blue", (0, 1, 1): "Cyan", (1, 0, 1): "Magenta", (1, 1, 0): "Yellow", (1, 1, 1): "Grays"}.get(key)
    except Exception:
        return None


_LUT_TABLE_CACHE: list[tuple[str, np.ndarray]] | None = None


def _known_lut_tables() -> list[tuple[str, np.ndarray]]:
    """Lumen's LUTs as (name, (256, 3) int16 table), built once."""
    global _LUT_TABLE_CACHE
    if _LUT_TABLE_CACHE is None:
        from .luts import LUT_NAMES, get_lut
        _LUT_TABLE_CACHE = [(n, get_lut(n).astype(np.int16)) for n in LUT_NAMES]
    return _LUT_TABLE_CACHE


def _lut_from_ij_table(lut: Any) -> tuple[str, bool] | None:
    """``(LUT name, inverted)`` for an ImageJ (3, 256) / (256, 3) colour table, or None.

    Exact match against Lumen's tables first, then the nearest table within ``_LUT_MATCH_TOL``
    (ImageJ interpolates its control points a little differently from ``luts._interp``; HiLo is
    only one entry away from Grays, so it matches exactly or not at all); both are also tried on
    the reversed table, which is what ImageJ's Invert LUT writes.  Falls back to the top-colour
    heuristic, again on both orientations.
    """
    try:
        arr = np.asarray(lut)
        table = (arr.T if arr.ndim == 2 and arr.shape[0] == 3 and arr.shape[1] != 3 else arr).astype(np.int16)
    except Exception:
        return None
    if table.shape != (256, 3):
        return None
    for inverted, cand in ((False, table), (True, table[::-1])):
        best, best_d = None, _LUT_MATCH_TOL
        for name, known in _known_lut_tables():
            d = float(np.abs(known - cand).mean())
            if d == 0.0:
                return name, inverted
            if d <= best_d and name != "HiLo":
                best, best_d = name, d
        if best is not None:
            return best, inverted
    for inverted, cand in ((False, table), (True, table[::-1])):
        name = _lut_name_from_ij(cand)
        if name:
            return name, inverted
    return None


def _lut_name_from_ome_color(value: Any) -> str | None:
    """Nearest primary LUT for an OME Channel ``Color`` (signed 32-bit RGBA int); None for black/unset."""
    try:
        v = int(value) & 0xFFFFFFFF
    except (TypeError, ValueError):
        return None
    rgb = ((v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255)
    if max(rgb) < 32:
        return None
    return min(_PRIMARY_LUT_RGB, key=lambda n: sum((a - b) ** 2 for a, b in zip(_PRIMARY_LUT_RGB[n], rgb)))


def _ome_color_from_lut(name: str) -> int:
    """OME Channel Color (signed 32-bit RGBA) from the colour at the top of a LUT's ramp (white = -1, the OME default)."""
    from .luts import get_lut
    try:
        r, g, b = (int(v) for v in get_lut(name)[255])
    except KeyError:
        r, g, b = 255, 255, 255
    v = (r << 24) | (g << 16) | (b << 8) | 255
    return v - (1 << 32) if v >= (1 << 31) else v


def _channel_names_from_labels(labels: list[str] | None, nt: int, nz: int, nc: int) -> list[str] | None:
    """Channel names from ImageJ slice labels: one per plane in ImageJ order (C fastest), identical for every plane of a channel."""
    if not labels or len(labels) != nt * nz * nc:
        return None
    names = [str(v) for v in labels[:nc]]
    if not all(n.strip() for n in names) or any(str(labels[i]) != names[i % nc] for i in range(len(labels))):
        return None
    return names


def _lumen_display_map(img: LumenImage) -> dict[str, str]:
    """Display state as flat key/value text for the OME MapAnnotation (read back by _apply_lumen_display)."""
    kv = {"mode": img.display.mode}
    for c, ch in enumerate(img.display.channels):
        k = f"channel.{c}."
        kv.update({k + "lut": ch.lut, k + "min": repr(float(ch.min)), k + "max": repr(float(ch.max)),
                   k + "inverted": str(bool(ch.inverted)).lower(), k + "visible": str(bool(ch.visible)).lower()})
        if ch.gamma != 1.0:
            kv[k + "gamma"] = repr(float(ch.gamma))
    return kv


def _apply_lumen_display(img: LumenImage, kv: dict[str, str]) -> None:
    from .luts import LUT_NAMES
    mode = kv.get("mode")
    if mode in ("composite", "single", "grayscale") and img.nc > 1:
        img.display.mode = mode
    for c, ch in enumerate(img.display.channels):
        k = f"channel.{c}."
        if kv.get(k + "lut") in LUT_NAMES:
            ch.lut = kv[k + "lut"]
        try:
            lo, hi = float(kv[k + "min"]), float(kv[k + "max"])
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                ch.min, ch.max = lo, hi
        except (KeyError, ValueError):
            pass
        for attr in ("inverted", "visible"):
            if k + attr in kv:
                setattr(ch, attr, kv[k + attr].strip().lower() == "true")
        try:
            gamma = float(kv[k + "gamma"])
            if np.isfinite(gamma) and gamma > 0:
                ch.gamma = gamma
        except (KeyError, ValueError):
            pass


def _read_tiff(p: Path, name: str, series_index: int = 0, level: int = 0, virtual: bool = False) -> LumenImage:
    """One series (and resolution level) of a TIFF.  ``virtual`` keeps the pixels in the file: a copy-on-write memmap
    for uncompressed data, a :class:`VirtualStack` for compressed / tiled pages (see File ▸ Import ▸ TIFF Virtual Stack)."""
    import tifffile
    lazy = None
    with tifffile.TiffFile(str(p)) as tf:
        idx = min(max(int(series_index or 0), 0), len(tf.series) - 1)
        series = tf.series[idx]
        levels = series.levels or [series]
        lvl_i = min(max(int(level or 0), 0), len(levels) - 1)
        lvl = levels[lvl_i]
        axes = str(lvl.axes)
        cal, meta, channel_names, disp = _ij_metadata_to_calibration(tf)
        ome_images = _ome_images_info(getattr(tf, "ome_metadata", None)) if (idx > 0 and getattr(tf, "is_ome", False)) else []
        if idx < len(ome_images):  # _ij_metadata_to_calibration read the first Image's Pixels: this series has its own
            oi = ome_images[idx]
            _apply_ome_pixels(cal, oi.get("pixels", {}))
            names = oi.get("channel_names") or []
            channel_names = [str(x) for x in names] if names and all(names) else None
            colors = []
            for c in oi.get("channel_colors") or []:
                try:
                    colors.append(int(c) if c not in (None, "") else None)
                except ValueError:
                    colors.append(None)
            disp["ome_colors"] = colors if any(v is not None for v in colors) else None
            if oi.get("name"):
                meta["series_name"] = oi["name"]
        if len(tf.series) > 1:
            meta["series_count"] = len(tf.series)
            meta["series"] = idx
            meta.setdefault("series_name", str(getattr(series, "name", "") or "") or f"Series {idx}")
            meta["series_names"] = [str(getattr(s, "name", "") or "") or f"Series {i}" for i, s in enumerate(tf.series)]
            meta["source"] = {"kind": "file", "path": str(p), "series": idx}  # so File ▸ Revert re-reads the same series
        if len(levels) > 1:
            meta["levels"] = len(levels)
            meta["level"] = lvl_i
            meta.setdefault("source", {"kind": "file", "path": str(p)})["level"] = lvl_i
            if lvl_i > 0:  # pixels of a lower pyramid level are bigger by the downsampling factor
                try:
                    fy, fx = series.shape[series.axes.index("Y")] / lvl.shape[axes.index("Y")], series.shape[series.axes.index("X")] / lvl.shape[axes.index("X")]
                    cal.pixel_width *= fx
                    cal.pixel_height *= fy
                except (ValueError, ZeroDivisionError):
                    pass
        if virtual:
            lazy = _tiff_lazy_5d(p, idx, lvl_i, lvl)
            if lazy is not None:
                meta.setdefault("source", {"kind": "file", "path": str(p)})["virtual"] = True
        data = None if lazy is not None else lvl.asarray()
    if lazy is not None:
        data5d, rgb = lazy
        if channel_names and not rgb and len(channel_names) != data5d.shape[2]:
            channel_names = None
        img = _lumen_from_lazy(data5d, name, rgb, cal, meta, p, channel_names)
        _apply_ij_display(img, disp, names_known=channel_names is not None)
        return img
    # tifffile may report an 'I' (sequence), 'Q' (unknown) or 'P' axis; LumenImage folds those into Z/T
    rgb = "S" in axes and data.shape[axes.index("S")] in (3, 4) and data.dtype == np.uint8
    if rgb and data.shape[axes.index("S")] == 4:
        data = np.take(data, [0, 1, 2], axis=axes.index("S"))
    if "S" in axes and not rgb:
        axes = axes.replace("S", "C")
    if channel_names and "C" in axes and len(channel_names) != data.shape[axes.index("C")]:
        channel_names = None
    img = LumenImage(data, name=name, axes=axes, rgb=rgb, calibration=cal, metadata=meta, path=str(p), channel_names=channel_names)
    _apply_ij_display(img, disp, names_known=channel_names is not None)
    return img


def _apply_ij_display(img: LumenImage, disp: dict[str, Any], names_known: bool = False) -> None:
    """Apply the display state gathered by _ij_metadata_to_calibration: ImageJ tags first, then OME on top.

    ``names_known`` means the channel names came from format metadata (OME Channel Name, LSM, …)
    and ImageJ slice labels must not override them.
    """
    labels = disp.get("labels")
    if labels and not img.rgb:
        names = _channel_names_from_labels(labels, img.nt, img.nz, img.nc)
        if names:
            # the same label on every plane of a channel = channel names (what write_image emits); they are NOT also
            # kept as slice labels, otherwise a channel rename followed by Save would write the stale labels back
            if not names_known:
                for ch, n in zip(img.display.channels, names):
                    ch.name = n
        else:
            # per-slice labels (Image Sequence file names, Micro-Manager / MetaMorph time stamps, Set Label…): the
            # project's slice-label key, one entry per (t, z, c) plane – ImageJ page order equals stacks.plane_index
            # order, so no reordering; like ImageJ's FileOpener they only apply when there are no more than planes
            n = img.nt * img.nz * img.nc
            lab = ["" if v is None else str(v) for v in labels]
            if len(lab) <= n and any(v.strip() for v in lab):
                img.metadata["slice_labels"] = lab + [""] * (n - len(lab))
    ranges = disp.get("ranges")
    covered = False
    if ranges is not None:
        ranges = list(np.asarray(ranges).ravel())
        if len(ranges) >= 2 * img.nc:
            for c, ch in enumerate(img.display.channels):
                ch.min, ch.max = float(ranges[2 * c]), float(ranges[2 * c + 1])
            covered = True
    # ImageJ's FileOpener applies the description-level min=/max= to non-composite images (skipping 0/0 = unset)
    lo, hi = disp.get("min"), disp.get("max")
    if not covered and not img.rgb and lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi) \
            and not (lo == 0.0 and hi == 0.0) and hi > lo:
        for ch in img.display.channels:
            ch.min, ch.max = float(lo), float(hi)
    luts = disp.get("ij_luts")
    if luts is not None and not isinstance(luts, (list, tuple)):
        # tifffile returns one LUT as a (3, 256) array and several as (n, 3, 256)
        arr = np.asarray(luts)
        luts = [arr] if arr.ndim == 2 else (list(arr) if arr.ndim == 3 else [])
    if luts is not None and len(luts) >= img.nc and not img.rgb:
        for c, ch in enumerate(img.display.channels):
            found = _lut_from_ij_table(luts[c])
            if found:
                ch.lut, ch.inverted = found
    if luts is None and disp.get("mode") in ("composite", "color") and img.nc > 1 and not img.rgb:
        # an ImageJ hyperstack with a mode but no LUT tables: ImageJ/Fiji show it with CompositeImage's
        # red / green / blue / gray / cyan / magenta / yellow channel colours
        try:
            from .ops._imagej_types import COMPOSITE_LUTS
            for c, ch in enumerate(img.display.channels):
                if c < len(COMPOSITE_LUTS):
                    ch.lut = COMPOSITE_LUTS[c]
        except Exception:  # noqa: BLE001 – keep Lumen's defaults when the port is unavailable
            pass
    mode = disp.get("mode")
    if mode in ("composite", "color", "grayscale") and img.nc > 1:
        img.display.mode = {"composite": "composite", "color": "single", "grayscale": "grayscale"}[mode]
    # OME: Channel Color -> nearest primary LUT; Lumen's MapAnnotation then restores the exact LUT names,
    # ranges, mode, inverted and visible flags (it wins over the ImageJ tags, which Lumen never writes alongside)
    colors = disp.get("ome_colors")
    if colors and not img.rgb and len(colors) >= img.nc:
        for c, ch in enumerate(img.display.channels):
            n = _lut_name_from_ome_color(colors[c])
            if n:
                ch.lut = n
    lumen = disp.get("lumen")
    if lumen and not img.rgb:
        _apply_lumen_display(img, lumen)


# ============================================================================ main dispatcher
def read_image(path: str | os.PathLike, name: str | None = None, *, dataset: str | None = None, series: int | None = None,
               level: int = 0, virtual: bool | str = False, dicom_options: dict[str, Any] | None = None) -> LumenImage:
    """Open any supported file (or folder) as a LumenImage.

    ``dataset`` selects a dataset/array inside HDF5 or Zarr containers; ``series`` the image series (see
    :func:`list_series`): a TIFF / OME-TIFF series, a CZI scene, an ND2 position, a LIF image, an HDF5 dataset,
    an OME-Zarr image of a bioformats2raw collection or a bioio scene – ``None`` keeps the reader's default
    (series 0; ND2 positions and CZI scenes without a time axis are folded into frames).  ``level`` picks a
    resolution level of a pyramidal TIFF, an Imaris file or an OME-Zarr multiscale (0 = full resolution).
    ``virtual`` keeps the pixels in the file and reads planes on demand (:mod:`lumen.core.lazy`): TIFF / OME-TIFF /
    LSM / STK (a copy-on-write memmap or a :class:`VirtualStack`, see File ▸ Import ▸ TIFF Virtual Stack), HDF5
    datasets and Imaris resolution levels, Zarr / OME-Zarr arrays and ``.npy`` files; other formats ignore it.
    ``virtual="auto"`` opens lazily when the decoded size exceeds the ``auto_virtual_mb`` preference
    (:func:`should_open_virtual`).  ``dicom_options`` overrides the DICOM reader preferences (Edit ▸ Options ▸
    DICOM, see :func:`dicom_reader_options`); by default they are read from the global session settings.
    """
    p = Path(path)
    name = name or p.name
    if virtual == "auto":
        virtual = should_open_virtual(p, series=series, level=level, dataset=dataset)
    virtual = bool(virtual)
    if p.is_dir():
        if _looks_like_zarr(p):
            return _read_zarr(p, name, dataset, series=series, level=level, virtual=virtual)
        if _looks_like_dicom_dir(p):
            return _read_dicom_dir(p, name, dicom_reader_options(dicom_options))
        return read_sequence(p)
    if not p.exists():
        raise FileNotFoundError(str(p))
    ext = _full_ext(p)
    try:
        return _read_file(p, name, ext, dataset=dataset, series=series, level=level, virtual=virtual, dicom_options=dicom_options)
    except (FileNotFoundError, MemoryError, KeyboardInterrupt):
        raise
    except Exception as e:  # noqa: BLE001
        # a format Lumen knows but could not read (unusual variant, missing optional package): the Java bridge is the last resort
        if ext in (".raw", ".bin", ".txt", ".csv", ".tsv") or not _bridge().available():
            raise
        try:
            return _read_bridge(p, name, series=series, level=level)
        except Exception:  # noqa: BLE001
            raise e from None


def _read_file(p: Path, name: str, ext: str, *, dataset: str | None, series: int | None, level: int, virtual: bool,
               dicom_options: dict[str, Any] | None) -> LumenImage:
    """The extension-dispatched readers behind :func:`read_image` (files only)."""
    s0 = int(series or 0)
    if ext in (".tif", ".tiff", ".ome.tif", ".ome.tiff", ".lsm", ".stk"):
        return _read_tiff(p, name, s0, level=level, virtual=virtual)
    if ext == ".npy":
        if virtual:
            lazy_img = _read_npy_lazy(p, name)
            if lazy_img is not None:
                return lazy_img
        data = np.load(str(p))
        # 5-D files are Lumen's own (T, Z, C, Y, X); anything else is a foreign array and keeps the ndim-based guess
        return LumenImage(data, name=name, axes="TZCYX" if data.ndim == 5 else None, path=str(p))
    if ext == ".nd2":
        nd2 = _need("nd2", ".nd2")
        with nd2.ND2File(str(p)) as f:
            data = f.asarray()
            sizes = dict(f.sizes)  # ordered like data.shape; nd2 emits 'P' for XY positions, 'T' for time loops
            axes = "".join(sizes)
            meta: dict[str, Any] = {}
            if "P" in sizes and series is not None:
                # an explicit series picks one XY position (a Bio-Formats series) instead of folding them into frames
                n_pos = sizes["P"]
                pos = min(max(s0, 0), n_pos - 1)
                data = np.take(data, pos, axis=axes.index("P"))
                axes = axes.replace("P", "")
                meta = {"series_count": n_pos, "series": pos, "series_name": f"position {pos + 1}", "source": {"kind": "file", "path": str(p), "series": pos}}
            elif "P" in sizes and "T" in sizes:
                # multi-point time-lapse: fold the positions into T (position-major: all frames of position 0,
                # then position 1, …).  A plain .replace("P", "T") would produce two T's and crash to_canonical.
                n_pos, n_t = sizes["P"], sizes["T"]
                data = np.moveaxis(data, (axes.index("P"), axes.index("T")), (0, 1))
                data = data.reshape((n_pos * n_t,) + data.shape[2:])
                axes = "T" + axes.replace("P", "").replace("T", "")
                meta = {"positions": n_pos, "frames_per_position": n_t}
            else:
                axes = axes.replace("P", "T")  # positions only: use them as frames
            cal = Calibration()
            try:
                vx = f.voxel_size()
                cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit = vx.x, vx.y, vx.z, "µm"
            except Exception:
                pass
        return LumenImage(data, name=name, axes=axes, calibration=cal, metadata=meta, path=str(p))
    if ext == ".czi":
        return _read_czi(p, name, series)
    if ext == ".lif":
        return _read_lif(p, name, s0)
    if ext in (".oif", ".oib"):
        return _read_oif(p, name)
    if ext in READER_BY_EXT:
        return _read_native(p, name, ext, series)
    if ext in (".dcm", ".dicom"):
        return _read_dicom(p, name, dicom_reader_options(dicom_options))
    if ext in (".nii", ".nii.gz"):
        return _read_nifti(p, name)
    if ext in (".mrc", ".rec", ".map", ".mrcs"):
        return _read_mrc(p, name)
    if ext in (".h5", ".hdf5", ".ims", ".hdf"):
        return _read_hdf5(p, name, dataset, series=series, level=level, virtual=virtual)
    if ext in VIDEO_EXTS:
        return _read_video(p, name)
    if ext in (".fits", ".fit", ".fts"):
        return _read_fits(p, name)
    if ext in (".mha", ".mhd"):
        return _read_metaimage(p, name)
    if ext in (".nrrd", ".nhdr"):
        return _read_nrrd(p, name)
    if ext in RASTER_EXTS:
        return _read_imageio(p, name)
    if ext in (".raw", ".bin"):
        raise ValueError(f"{p.name} is a headerless raw file – use File ▸ Import ▸ Raw… (file.import_raw) and give width, height and type")
    if ext in (".txt", ".csv", ".tsv"):
        return read_text_image(p, name)
    if _is_dicom_file(p):
        return _read_dicom(p, name, dicom_reader_options(dicom_options))
    return _read_fallback(p, name, ext, series=series, level=level)


def _read_czi(p: Path, name: str, series: int | None = 0) -> LumenImage:
    """Zeiss CZI through czifile.  A scene is one Bio-Formats series: ``series`` picks it when the file
    also has a time axis (or when it is given explicitly), otherwise the scenes are opened as frames (like
    ND2 positions)."""
    czifile = _need("czifile", ".czi")
    meta: dict[str, Any] = {}
    explicit = series is not None
    s0 = int(series or 0)
    with czifile.CziFile(str(p)) as f:
        cal = _czi_scaling(f)
        if hasattr(f, "axes") and hasattr(f, "shape"):
            # classic czifile (<= 2019.7.x): '0' = sample (RGB, already BGR->RGB), 'S' = scene, 'M' never
            # appears in CziFile.axes.  Rename before LumenImage sees the letters ('S' means samples there).
            data, axes = np.asarray(f.asarray()), str(f.axes)
            axes = "".join("S" if ch == "0" else ("P" if ch == "S" else ch) for ch in axes)
        else:
            # czifile >= 2026: scenes are separate CziImage objects and their axes use 'S' for samples
            scenes = list(getattr(f, "scenes").items())
            if len(scenes) > 1:
                meta["series_count"] = len(scenes)
                meta["series_names"] = [str(k) for k, _ in scenes]
            si = min(max(s0, 0), len(scenes) - 1)
            key, im = scenes[si]
            if len(scenes) > 1:
                meta["series"], meta["series_name"] = si, str(key)
                meta["source"] = {"kind": "file", "path": str(p), "series": si}
            data, axes = np.asarray(im.asarray()), str(im.axes)
    if len(axes) != data.ndim:
        raise ValueError(f"{p.name}: czifile axes {axes!r} do not match the array shape {data.shape}")
    # drop singleton axes, keeping data and axes string in sync (always keep Y and X)
    keep = [i for i, n in enumerate(data.shape) if n > 1 or axes[i] in "YX"]
    data = data.reshape([data.shape[i] for i in keep])
    axes = "".join(axes[i] for i in keep)
    if "P" in axes:  # several scenes (classic API)
        i = axes.index("P")
        if "T" not in axes and not explicit:
            axes = axes.replace("P", "T")  # scenes as frames
            meta["positions"] = data.shape[i]
        else:
            meta["series_count"] = data.shape[i]
            meta["series"] = min(max(s0, 0), data.shape[i] - 1)
            meta["series_name"] = f"scene {meta['series'] + 1}"
            meta["source"] = {"kind": "file", "path": str(p), "series": meta["series"]}
            data = np.take(data, meta["series"], axis=i)
            axes = axes[:i] + axes[i + 1:]
    rgb = "S" in axes and data.shape[axes.index("S")] in (3, 4) and data.dtype == np.uint8
    if "S" in axes and not rgb:
        axes = axes.replace("S", "C")  # e.g. Bgr48 → 3-channel composite
    return LumenImage(np.ascontiguousarray(data), name=name, axes=axes, rgb=rgb, calibration=cal, metadata=meta, path=str(p))


def _read_lif(p: Path, name: str, series: int = 0) -> LumenImage:
    """Leica LIF through readlif: the full (T, Z, C, Y, X) hyperstack of image ``series``."""
    try:
        from readlif.reader import LifFile
    except ImportError as e:
        raise RuntimeError("Reading .lif needs: pip install readlif") from e
    lif = LifFile(str(p))
    n_images = int(getattr(lif, "num_images", 0) or len(getattr(lif, "image_list", [])) or 1)
    idx = min(max(int(series), 0), n_images - 1)
    im = lif.get_image(idx)
    nt, nz, nc = int(getattr(im.dims, "t", 1) or 1), int(getattr(im.dims, "z", 1) or 1), int(im.channels or 1)
    first = np.asarray(im.get_frame(z=0, t=0, c=0))
    data = np.empty((nt, nz, nc) + first.shape, dtype=first.dtype)
    for t in range(nt):
        for z in range(nz):
            for c in range(nc):
                data[t, z, c] = first if (t, z, c) == (0, 0, 0) else np.asarray(im.get_frame(z=z, t=t, c=c))
    cal = Calibration()
    try:
        sx, sy, sz, st = (list(im.scale) + [None] * 4)[:4]  # readlif: px/µm for x, y, z; frames/s for t; entries may be None
        if sx:
            cal.pixel_width, cal.pixel_height, cal.unit = 1.0 / sx, 1.0 / (sy or sx), "µm"
        if sz:
            cal.pixel_depth = 1.0 / sz
        if st and nt > 1:
            cal.frame_interval, cal.time_unit = 1.0 / st, "sec"
    except Exception:
        pass
    meta: dict[str, Any] = {"lif_image_index": idx, "lif_image_name": str(getattr(im, "name", "") or "")}
    if n_images > 1:
        meta["series_count"] = n_images  # same key _read_tiff uses
        meta["series"], meta["series_name"] = idx, meta["lif_image_name"] or f"Image {idx}"
        meta["source"] = {"kind": "file", "path": str(p), "series": idx}
        try:
            meta["series_names"] = [str(e.get("name", "") if isinstance(e, dict) else getattr(e, "name", "")) or f"Image {i}"
                                    for i, e in enumerate(getattr(lif, "image_list", []) or [])][:n_images] or None
            if meta["series_names"] is None:
                del meta["series_names"]
        except Exception:
            pass
    if int(getattr(im.dims, "m", 1) or 1) > 1:
        meta["lif_mosaic_tiles"] = int(im.dims.m)  # tiles beyond m=0 are not loaded
    return LumenImage(data, name=name, axes="TZCYX", calibration=cal, metadata=meta, path=str(p))


def _read_indexed(p: Path, name: str) -> LumenImage | None:
    """A single-frame palette image (indexed GIF / PNG / BMP) the way ImageJ opens it: the 8-bit *indices* are the pixel
    values and the palette becomes the display LUT - a grey ramp is Grays, a reversed grey ramp is an inverting LUT
    (ImageJ's blobs.gif: raw values average 103.27, which every Fiji tutorial measures), anything else a runtime LUT.
    None when the file is not a single-frame palette image (the caller reads it as RGB / grey as before)."""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    try:
        with Image.open(p) as im:
            if im.mode != "P" or int(getattr(im, "n_frames", 1) or 1) > 1 or "transparency" in im.info and im.format == "PNG":
                return None
            idx = np.asarray(im, dtype=np.uint8).copy()
            pal = np.asarray(im.getpalette() or [], dtype=np.uint8)
    except Exception:  # noqa: BLE001
        return None
    if pal.size < 3:
        return None
    pal = pal[: (pal.size // 3) * 3].reshape(-1, 3)[:256]
    table = np.zeros((256, 3), np.uint8)
    table[: len(pal)] = pal
    used = int(idx.max()) + 1 if idx.size else 1
    grey = bool(np.all(table[:used, 0] == table[:used, 1]) and np.all(table[:used, 1] == table[:used, 2]))
    img = LumenImage(idx, name=name, path=str(p))
    ch = img.display.channels[0]
    ch.min, ch.max = 0.0, 255.0
    ramp = np.arange(256, dtype=np.int16)
    if grey and np.array_equal(table[:used, 0].astype(np.int16), ramp[:used]):
        pass  # a plain grey ramp: Grays
    elif grey and np.array_equal(table[:used, 0].astype(np.int16), 255 - ramp[:used]):
        ch.inverted = True  # ImageJ's "inverting LUT"
        img.metadata["inverting_lut"] = True
    else:
        from .luts import register_lut
        ch.lut = register_lut(f"{Path(name).stem} palette", table)
    img.metadata["indexed_color"] = True
    return img


def _read_imageio(p: Path, name: str) -> LumenImage:
    import imageio.v3 as iio
    ext = p.suffix.lower()
    if ext in (".gif", ".png", ".bmp"):
        indexed = _read_indexed(p, name)
        if indexed is not None:
            return indexed
    data = iio.imread(str(p))
    if ext == ".gif" and data.ndim == 4:
        data = data[..., :3]
        return LumenImage(data, name=name, axes="TYXS", rgb=True, path=str(p))
    if data.ndim == 3 and data.shape[-1] == 2:
        # grey + alpha (LA png/tga/webp): keep the grey plane and drop alpha, as for RGBA and as ImageJ does
        data = np.ascontiguousarray(data[..., 0])
    rgb = data.ndim == 3 and data.shape[-1] in (3, 4)
    if rgb and data.dtype != np.uint8:
        data = (data.astype(np.float32) / data.max() * 255).astype(np.uint8) if data.max() > 0 else data.astype(np.uint8)
    img = LumenImage(data, name=name, axes="YXS" if rgb else None, rgb=rgb, path=str(p))
    if rgb and data.shape[-1] == 3 and np.array_equal(data[..., 0], data[..., 1]) and np.array_equal(data[..., 1], data[..., 2]):
        img = LumenImage(data[..., 0], name=name, path=str(p))  # grayscale saved as RGB
    return img


def _read_oif(p: Path, name: str) -> LumenImage:
    oiffile = _need("oiffile", p.suffix)
    with oiffile.OifFile(str(p)) as oif:
        data = oif.asarray()
        axes = str(getattr(oif, "axes", "") or "")
        cal = Calibration()
        try:
            main = oif.mainfile
            ref = main.get("Reference Image Parameter", {})
            if ref.get("WidthConvertValue"):
                cal.pixel_width = float(ref["WidthConvertValue"])
                cal.pixel_height = float(ref.get("HeightConvertValue", ref["WidthConvertValue"]))
                cal.unit = _norm_unit(ref.get("WidthUnit", "µm"))
            for k, v in main.items():
                if k.startswith("Axis ") and v.get("AxisCode") == "Z" and v.get("Interval"):
                    cal.pixel_depth = abs(float(v["Interval"])) / 1000.0 if str(v.get("UnitName", "")).lower().startswith("nm") else abs(float(v["Interval"]))
        except Exception:
            pass
    return LumenImage(data, name=name, axes=axes if len(axes) == data.ndim else None, calibration=cal, path=str(p))


# ============================================================================ DICOM
DICOM_OPTION_DEFAULTS: dict[str, bool] = {
    "dicom_open_as_32bit": False,       # True: float32 with RescaleSlope / RescaleIntercept applied; False: native dtype, raw stored values
    "dicom_ignore_rescale": False,      # True: never apply the rescale slope / intercept
    "dicom_calibrate_images": True,     # PixelSpacing / SliceThickness / FrameTime → spatial & temporal calibration
    "dicom_window_level_on_open": True, # WindowCenter / WindowWidth → display range
}


def dicom_reader_options(override: dict[str, Any] | None = None) -> dict[str, bool]:
    """The DICOM reader preferences set by ``edit.options_dicom``: the defaults, updated from the global session
    settings (``lumen.core.session.SESSION.settings``, keys ``dicom_*``), then from ``override`` (keys with or
    without the ``dicom_`` prefix – used by tests and scripts)."""
    opts = dict(DICOM_OPTION_DEFAULTS)
    try:
        from .session import SESSION  # imported here: session imports io
        settings = SESSION.settings
        for k in opts:
            if k in settings:
                opts[k] = bool(settings[k])
    except Exception:
        pass
    for k, v in (override or {}).items():
        key = k if k.startswith("dicom_") else "dicom_" + k
        if key in opts:
            opts[key] = bool(v)
    return opts


def _is_dicom_file(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _looks_like_dicom_dir(p: Path) -> bool:
    n = 0
    for f in p.iterdir():
        if f.name.startswith(".") or not f.is_file():
            continue
        if f.suffix.lower() in (".dcm", ".dicom"):
            return True
        if f.suffix.lower() in IMAGE_EXTS:
            return False
        if f.suffix == "" or f.suffix.lower() == ".ima":
            if _is_dicom_file(f):
                return True
        n += 1
        if n > 25:
            break
    return False


def _dicom_calibration(ds) -> Calibration:
    cal = Calibration()
    try:
        ps = getattr(ds, "PixelSpacing", None) or getattr(ds, "ImagerPixelSpacing", None)
        if ps is not None and len(ps) >= 2:
            cal.pixel_height, cal.pixel_width = float(ps[0]), float(ps[1])
            cal.unit = "mm"
        depth = getattr(ds, "SpacingBetweenSlices", None) or getattr(ds, "SliceThickness", None)
        if depth:
            cal.pixel_depth = abs(float(depth))
            if cal.unit == "pixel":
                cal.unit = "mm"
        ft = getattr(ds, "FrameTime", None)
        if ft:
            cal.frame_interval = float(ft) / 1000.0
    except Exception:
        pass
    return cal


def _dicom_pixels(ds, opts: dict[str, bool] | None = None) -> np.ndarray:
    """Pixel array honouring the DICOM preferences: native dtype by default (the raw stored values; the rescale
    slope / intercept stay in the metadata), float32 with the rescale applied when ``dicom_open_as_32bit`` is on
    (unless ``dicom_ignore_rescale``).  RGB (SamplesPerPixel 3) data is never rescaled."""
    opts = opts or DICOM_OPTION_DEFAULTS
    arr = np.asarray(ds.pixel_array)
    if int(getattr(ds, "SamplesPerPixel", 1) or 1) != 1 or not opts.get("dicom_open_as_32bit", False):
        return arr
    arr = arr.astype(np.float32)
    if opts.get("dicom_ignore_rescale", False):
        return arr
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
    if slope != 1.0 or inter != 0.0:
        arr = arr * slope + inter
    return arr


def _dicom_meta(ds) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key in ("PatientName", "PatientID", "StudyDescription", "SeriesDescription", "Modality", "StudyDate", "SeriesNumber",
                "InstanceNumber", "PhotometricInterpretation", "WindowCenter", "WindowWidth", "BitsStored", "Manufacturer", "SOPClassUID",
                "RescaleSlope", "RescaleIntercept", "RescaleType", "PixelRepresentation"):
        v = getattr(ds, key, None)
        if v is None:
            continue
        try:
            if isinstance(v, (int, float, str)):
                meta[key] = v
            else:
                meta[key] = str(v)
        except Exception:
            pass
    return meta


def _read_dicom(p: Path, name: str, opts: dict[str, bool] | None = None) -> LumenImage:
    pydicom = _need("pydicom", ".dcm")
    opts = opts or dicom_reader_options()
    ds = pydicom.dcmread(str(p), force=True)
    arr = _dicom_pixels(ds, opts)
    rgb = int(getattr(ds, "SamplesPerPixel", 1)) == 3
    nframes = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if rgb:
        axes = "ZYXS" if arr.ndim == 4 else "YXS"
    else:
        axes = "ZYX" if (arr.ndim == 3 or nframes > 1) else "YX"
    cal = _dicom_calibration(ds) if opts.get("dicom_calibrate_images", True) else Calibration()
    img = LumenImage(arr, name=name, axes=axes, rgb=rgb, calibration=cal, metadata=_dicom_meta(ds), path=str(p))
    _dicom_window(img, ds, opts)
    return img


def _dicom_window(img: LumenImage, ds, opts: dict[str, bool] | None = None) -> None:
    """WindowCenter / WindowWidth → display range (skipped when ``dicom_window_level_on_open`` is off).  The window
    is expressed in rescaled units; when the pixels keep their raw stored values it is mapped back through the
    rescale slope / intercept."""
    opts = opts or DICOM_OPTION_DEFAULTS
    if not opts.get("dicom_window_level_on_open", True):
        return
    try:
        wc, ww = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
        if wc is None or ww is None or img.rgb:
            return
        wc = float(wc[0] if isinstance(wc, (list, tuple, pydicom_multivalue())) else wc)
        ww = float(ww[0] if isinstance(ww, (list, tuple, pydicom_multivalue())) else ww)
        if ww <= 0:
            return
        lo, hi = wc - ww / 2, wc + ww / 2
        rescaled = opts.get("dicom_open_as_32bit", False) and not opts.get("dicom_ignore_rescale", False)
        if not rescaled:  # raw stored values: undo the rescale on the window
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
            if slope != 0:
                lo, hi = (lo - inter) / slope, (hi - inter) / slope
        for ch in img.display.channels:
            ch.min, ch.max = lo, hi
    except Exception:
        pass


def pydicom_multivalue():
    try:
        from pydicom.multival import MultiValue
        return MultiValue
    except Exception:  # pragma: no cover
        return tuple


def _read_dicom_dir(p: Path, name: str, opts: dict[str, bool] | None = None) -> LumenImage:
    pydicom = _need("pydicom", ".dcm")
    opts = opts or dicom_reader_options()
    files = [f for f in sorted(p.iterdir()) if f.is_file() and not f.name.startswith(".") and (f.suffix.lower() in (".dcm", ".dicom", ".ima") or (f.suffix == "" and _is_dicom_file(f)))]
    if not files:
        raise FileNotFoundError(f"No DICOM files in {p}")
    headers = []
    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        except Exception:
            continue
        inst = getattr(ds, "InstanceNumber", None)
        pos = getattr(ds, "ImagePositionPatient", None)
        zpos = float(pos[2]) if pos is not None and len(pos) >= 3 else None
        headers.append((f, int(inst) if inst is not None else None, zpos))
    if not headers:
        raise ValueError(f"No readable DICOM files in {p}")
    if all(h[1] is not None for h in headers):
        headers.sort(key=lambda h: (h[1], h[0].name))
    elif all(h[2] is not None for h in headers):
        headers.sort(key=lambda h: (h[2], h[0].name))
    planes = []
    first = None
    for f, _, _ in headers:
        ds = pydicom.dcmread(str(f), force=True)
        if first is None:
            first = ds
        arr = _dicom_pixels(ds, opts)
        if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[-1] == 3):
            planes.append(arr)
        else:
            planes.extend(list(arr))
    data = np.stack(planes, axis=0)
    rgb = int(getattr(first, "SamplesPerPixel", 1)) == 3
    calibrate = opts.get("dicom_calibrate_images", True)
    cal = _dicom_calibration(first) if calibrate else Calibration()
    zs = [h[2] for h in headers if h[2] is not None] if calibrate else []
    if len(zs) >= 2:
        d = np.diff(np.sort(np.array(zs, float)))
        d = d[d > 0]
        if d.size:
            cal.pixel_depth = float(np.median(d))
    img = LumenImage(data, name=name, axes="ZYXS" if rgb else "ZYX", rgb=rgb, calibration=cal, metadata=_dicom_meta(first), path=str(p))
    _dicom_window(img, first, opts)
    return img


# ============================================================================ NIfTI
def _read_nifti(p: Path, name: str) -> LumenImage:
    nib = _need("nibabel", ".nii")
    ni = nib.load(str(p))
    try:
        ni = nib.as_closest_canonical(ni)
    except Exception:
        pass
    data = np.asanyarray(ni.dataobj)
    zooms = [float(z) for z in ni.header.get_zooms()]
    try:
        xyz_unit, t_unit = ni.header.get_xyzt_units()
    except Exception:
        xyz_unit, t_unit = "unknown", "unknown"
    nd = data.ndim
    if nd == 2:
        arr, axes = data.T, "YX"
    elif nd == 3:
        arr, axes = np.transpose(data, (2, 1, 0)), "ZYX"
    elif nd == 4:
        arr, axes = np.transpose(data, (3, 2, 1, 0)), "TZYX"
    elif nd >= 5:
        data = data.reshape(data.shape[:4] + (-1,))
        arr, axes = np.transpose(data, (3, 2, 4, 1, 0)), "TZCYX"
    else:
        raise ValueError(f"Unsupported NIfTI dimensionality {data.shape}")
    arr = arr[..., ::-1, :]  # canonical RAS → anterior at the top row
    cal = Calibration(unit=_norm_unit(xyz_unit))
    if len(zooms) >= 1:
        cal.pixel_width = zooms[0]
    if len(zooms) >= 2:
        cal.pixel_height = zooms[1]
    if len(zooms) >= 3:
        cal.pixel_depth = zooms[2]
    if len(zooms) >= 4 and nd >= 4:
        cal.frame_interval = zooms[3]
        cal.time_unit = {"sec": "sec", "msec": "ms", "usec": "µs"}.get(t_unit, "sec")
    meta = {"nifti_datatype": str(data.dtype), "qform_code": int(ni.header["qform_code"]), "sform_code": int(ni.header["sform_code"])}
    try:
        meta["description"] = ni.header["descrip"].tobytes().decode("ascii", "ignore").strip("\x00")
    except Exception:
        pass
    return LumenImage(np.ascontiguousarray(arr), name=name, axes=axes, calibration=cal, metadata=meta, path=str(p))


def _write_nifti(img: LumenImage, p: Path, gz: bool) -> str:
    nib = _need("nibabel", ".nii", verb="Writing")
    cal = img.calibration
    data = img.data[:, :, :, ::-1, :]  # undo the display flip
    out = np.transpose(data, (4, 3, 1, 0, 2))  # X Y Z T C
    if img.nc == 1 and img.nt == 1:
        out = out[:, :, :, 0, 0]
    elif img.nc == 1:
        out = out[:, :, :, :, 0]
    out = np.ascontiguousarray(out)
    affine = np.diag([cal.pixel_width, cal.pixel_height, cal.pixel_depth, 1.0])
    ni = nib.Nifti1Image(out, affine)
    zooms = [cal.pixel_width, cal.pixel_height, cal.pixel_depth]
    # NIfTI can only express sec/msec/usec: convert minutes/hours/days to seconds instead of writing 'unknown'
    fi, tu = cal.frame_interval or 1.0, cal.time_unit
    factor = _TIME_TO_SEC.get(tu)
    if factor and factor != 1.0:
        fi, tu = fi * factor, "sec"
    if out.ndim >= 4:
        zooms.append(fi)
    if out.ndim >= 5:
        zooms.append(1.0)
    ni.header.set_zooms(tuple(zooms))
    xyz = {"mm": "mm", "µm": "micron", "m": "meter"}.get(cal.unit, "unknown")
    t = {"sec": "sec", "s": "sec", "ms": "msec", "µs": "usec", "us": "usec"}.get(tu, "unknown")
    try:
        ni.header.set_xyzt_units(xyz=xyz, t=t)
    except Exception:
        pass
    ni.header["descrip"] = f"Lumen {img.name}"[:79].encode()
    suffix = ".nii.gz" if gz else ".nii"
    base = re.sub(r"\.nii(\.gz)?$", "", str(p), flags=re.I)
    out_path = base + suffix
    nib.save(ni, out_path)
    return out_path


# ============================================================================ MRC
def _read_mrc(p: Path, name: str) -> LumenImage:
    mrcfile = _need("mrcfile", p.suffix)
    with mrcfile.open(str(p), permissive=True) as m:
        data = np.asarray(m.data)
        vs = m.voxel_size
        cal = Calibration()
        try:
            vx, vy, vz = float(vs.x), float(vs.y), float(vs.z)
            if vx > 0:
                cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit = vx, vy or vx, vz or vx, "Å"
        except Exception:
            pass
        try:
            is_vol_stack = m.is_volume_stack()
        except Exception:
            is_vol_stack = False
    if np.iscomplexobj(data):
        data = np.abs(data).astype(np.float32)
    if data.ndim == 2:
        axes = "YX"
    elif data.ndim == 3:
        axes = "ZYX"
    elif data.ndim == 4:
        axes = "TZYX"
    else:
        raise ValueError(f"Unsupported MRC shape {data.shape}")
    return LumenImage(data, name=name, axes=axes, calibration=cal, metadata={"volume_stack": is_vol_stack}, path=str(p))


def _write_mrc(img: LumenImage, p: Path) -> str:
    mrcfile = _need("mrcfile", ".mrc", verb="Writing")
    if img.nc > 1:
        raise ValueError("MRC files cannot store channels – split channels first")
    data = img.data[:, :, 0]  # T Z Y X
    if img.nt == 1:
        data = data[0]
        if data.shape[0] == 1:
            data = data[0]
    data = np.ascontiguousarray(data)
    if data.dtype == np.uint8:
        data = data.astype(np.uint16)  # mrcfile has no unsigned 8-bit mode
    with mrcfile.new(str(p), overwrite=True) as m:
        m.set_data(data)
        cal = img.calibration
        f = _TO_ANGSTROM.get(cal.unit)
        if f:
            m.voxel_size = (cal.pixel_width * f, cal.pixel_height * f, cal.pixel_depth * f)
        if img.nt > 1 and hasattr(m, "set_volume_stack"):
            try:
                m.set_volume_stack()
            except Exception:
                pass
    return str(p)


# ============================================================================ HDF5 / Imaris
def _h5_largest_dataset(f) -> str | None:
    best: tuple[int, str] | None = None

    def visit(name, obj):
        nonlocal best
        import h5py
        if isinstance(obj, h5py.Dataset) and 2 <= obj.ndim <= 5 and obj.dtype.kind in "uif":
            size = int(np.prod(obj.shape))
            if best is None or size > best[0]:
                best = (size, name)
    f.visititems(visit)
    return best[1] if best else None


def _ims_attr(obj, key: str, default: Any = None) -> Any:
    try:
        v = obj.attrs[key]
        if isinstance(v, np.ndarray):
            return b"".join(bytes(x) if isinstance(x, (bytes, np.bytes_)) else str(x).encode() for x in v.ravel()).decode("utf-8", "ignore")
        if isinstance(v, bytes):
            return v.decode("utf-8", "ignore")
        return v
    except Exception:
        return default


def _ims_calibration(f, sx: int | None, sy: int | None, sz: int | None) -> Calibration:
    """Voxel size of an Imaris file from DataSetInfo/Image extents and the full-resolution sizes."""
    info = f.get("DataSetInfo/Image")
    cal = Calibration()
    if info is None or not sx or not sy:
        return cal
    try:
        ext = [float(_ims_attr(info, f"ExtMin{i}")) for i in range(3)], [float(_ims_attr(info, f"ExtMax{i}")) for i in range(3)]
        cal.pixel_width = (ext[1][0] - ext[0][0]) / sx
        cal.pixel_height = (ext[1][1] - ext[0][1]) / sy
        cal.pixel_depth = (ext[1][2] - ext[0][2]) / sz if sz else 1.0
        cal.unit = _norm_unit(_ims_attr(info, "Unit", "µm"))
    except Exception:
        return Calibration()
    return cal


def _read_imaris(f, p: Path, name: str, level: int = 0, virtual: bool = False) -> LumenImage:
    levels = _ims_levels(f) or ["ResolutionLevel 0"]
    li = min(max(int(level or 0), 0), len(levels) - 1)
    lvl = f["DataSet"][levels[li]]
    tps = sorted((k for k in lvl.keys() if k.startswith("TimePoint")), key=lambda s: int(s.split()[-1]))
    chans = sorted((k for k in lvl[tps[0]].keys() if k.startswith("Channel")), key=lambda s: int(s.split()[-1]))
    info = f.get("DataSetInfo/Image")
    sx = sy = sz = None
    if info is not None:
        try:
            sx, sy, sz = (int(_ims_attr(info, k)) for k in ("X", "Y", "Z"))
        except Exception:
            sx = sy = sz = None
    cal = _ims_calibration(f, sx, sy, sz)
    sizes = _ims_level_sizes(lvl)  # (Z, Y, X) of this level – the Data arrays are padded to chunk multiples
    if sizes is None and li == 0 and sz is not None:
        sizes = (sz, sy, sx)
    lazy: ImarisPlaneSource | None = None
    data: np.ndarray | None = None
    if virtual:
        d0 = lvl[tps[0]][chans[0]]["Data"]
        lazy = ImarisPlaneSource(p, levels[li], tps, chans, sizes or tuple(int(v) for v in d0.shape[-3:]), d0.dtype)
        nz_, ny_, nx_ = lazy.sizes
    else:
        vol = []
        for t in tps:
            cs = []
            for c in chans:
                d = lvl[t][c]["Data"]
                arr = d[()]
                if sizes is not None:
                    arr = arr[:sizes[0], :sizes[1], :sizes[2]]
                cs.append(arr)
            vol.append(np.stack(cs))
        data = np.stack(vol)  # T C Z Y X
        nz_, ny_, nx_ = data.shape[-3], data.shape[-2], data.shape[-1]
    if li > 0 and sx and sy and cal.unit != "pixel":  # coarser level: bigger voxels
        cal.pixel_width *= sx / nx_
        cal.pixel_height *= sy / ny_
        if sz and nz_:
            cal.pixel_depth *= sz / nz_
    channel_names = []
    for i, _ in enumerate(chans):
        ci = f.get(f"DataSetInfo/Channel {i}")
        channel_names.append(str(_ims_attr(ci, "Name", f"C{i + 1}")) if ci is not None else f"C{i + 1}")
    meta: dict[str, Any] = {"format": "Imaris"}
    if len(levels) > 1:
        meta["levels"], meta["level"] = len(levels), li
        meta["source"] = {"kind": "file", "path": str(p), "level": li}
    if lazy is not None:  # the planes stay in the file: DataSet/ResolutionLevel/TimePoint/Channel/Data read on demand
        meta.setdefault("source", {"kind": "file", "path": str(p), "level": li})["virtual"] = True
        return _lumen_from_lazy(LazyArray(lazy), name, False, cal, meta, p, channel_names)
    return LumenImage(data, name=name, axes="TCZYX", calibration=cal, channel_names=channel_names, metadata=meta, path=str(p))


def _read_hdf5(p: Path, name: str, dataset: str | None, series: int | None = None, level: int = 0, virtual: bool = False) -> LumenImage:
    h5py = _need("h5py", p.suffix)
    with h5py.File(str(p), "r") as f:
        if dataset is None and "DataSet" in f and "DataSetInfo" in f:
            return _read_imaris(f, p, name, level=level, virtual=virtual)
        ds_names = _h5_image_datasets(f)
        if dataset is None and series is not None and ds_names:
            dataset = ds_names[min(max(int(series), 0), len(ds_names) - 1)]
        key = dataset or _h5_largest_dataset(f)
        if key is None or key not in f:
            raise ValueError(f"No image dataset found in {p.name}" + (f" (asked for {dataset!r})" if dataset else ""))
        ds = f[key]
        shape, ndim, src_dtype = tuple(int(v) for v in ds.shape), int(ds.ndim), np.dtype(ds.dtype)
        attrs = dict(ds.attrs)
        axes = attrs.get("axes")
        if isinstance(axes, bytes):
            axes = axes.decode()
        axes = str(axes) if axes is not None else None
        if axes and len(axes) != ndim:
            axes = None
        if axes is None and "DIMENSION_LABELS" in attrs:
            try:
                labels = "".join(str(x.decode() if isinstance(x, bytes) else x)[0].upper() for x in attrs["DIMENSION_LABELS"])
                if len(labels) == ndim:
                    axes = labels
            except Exception:
                pass
        cal = Calibration()
        esu = attrs.get("element_size_um")
        if esu is not None:
            esu = np.atleast_1d(np.asarray(esu, float))
            if esu.size >= 2:
                cal.pixel_height, cal.pixel_width = float(esu[-2]), float(esu[-1])
                cal.unit = "µm"
            if esu.size >= 3:
                cal.pixel_depth = float(esu[-3])
        for k in ("pixel_width", "pixel_height", "pixel_depth", "frame_interval"):
            if k in attrs:
                setattr(cal, k, float(attrs[k]))
        if "unit" in attrs:
            cal.unit = _norm_unit(attrs["unit"].decode() if isinstance(attrs["unit"], bytes) else attrs["unit"])
        if "time_unit" in attrs:
            tu = attrs["time_unit"]
            cal.time_unit = tu.decode() if isinstance(tu, bytes) else str(tu)
        names = attrs.get("channel_names")
        channel_names = [n.decode() if isinstance(n, bytes) else str(n) for n in names] if names is not None else None
        rgb = bool(attrs.get("rgb", False)) and src_dtype == np.uint8
        lazy = None
        if virtual:
            try:
                lazy = hdf5_source(p, key, axes or _guess_axes(shape, src_dtype, rgb), shape, src_dtype, rgb)
            except ValueError:  # axes that cannot be mapped onto T/Z/C without merging: read into memory
                lazy = None
        data = None if lazy is not None else ds[()]
    meta = {"dataset": key, **{k: (v.item() if isinstance(v, np.generic) else v) for k, v in attrs.items() if isinstance(v, (str, int, float, np.generic))}}
    meta["source"] = {"kind": "file", "path": str(p), "dataset": key}  # so File ▸ Revert re-reads the same dataset
    if key in ds_names:
        meta["source"]["series"] = ds_names.index(key)  # undo snapshots re-open a virtual stack by series index
    if lazy is not None:
        meta["source"]["virtual"] = True
        return _lumen_from_lazy(LazyArray(lazy), name, rgb, cal, meta, p, channel_names)
    return LumenImage(data, name=name, axes=axes, rgb=rgb, calibration=cal, channel_names=channel_names, metadata=meta, path=str(p))


def _write_hdf5(img: LumenImage, p: Path) -> str:
    h5py = _need("h5py", ".h5", verb="Writing")
    cal = img.calibration
    with h5py.File(str(p), "w") as f:
        chunks = (1, 1, 1, img.height, img.width)
        if is_virtual(img.data):  # a virtual stack is written plane by plane
            ds = f.create_dataset("data", shape=tuple(img.data.shape), dtype=np.dtype(img.dtype), compression="gzip", compression_opts=4, chunks=chunks)
            for (t, z, c), plane in _lazy_iter_planes(img.data):
                ds[t, z, c] = plane
        else:
            ds = f.create_dataset("data", data=img.data, compression="gzip", compression_opts=4, chunks=chunks)
        ds.attrs["axes"] = "TZCYX"
        ds.attrs["unit"] = cal.unit
        for k in ("pixel_width", "pixel_height", "pixel_depth", "frame_interval"):
            ds.attrs[k] = float(getattr(cal, k))
        ds.attrs["time_unit"] = cal.time_unit
        if cal.unit == "µm":
            ds.attrs["element_size_um"] = np.array([cal.pixel_depth, cal.pixel_height, cal.pixel_width], np.float32)
        # variable-length UTF-8 strings: dtype="S" would encode with ASCII and crash on 'µ', '–', accents…
        ds.attrs["channel_names"] = np.array([c.name for c in img.display.channels], dtype=h5py.string_dtype("utf-8"))
        ds.attrs["rgb"] = bool(img.rgb)
        ds.attrs["name"] = img.name
    return str(p)


# ============================================================================ Zarr / OME-Zarr
def _looks_like_zarr(p: Path) -> bool:
    if p.name.lower().endswith(".zarr"):
        return True
    return any((p / f).exists() for f in ("zarr.json", ".zarray", ".zgroup"))


def _zarr_group_attrs(node) -> dict[str, Any]:
    try:
        attrs = dict(node.attrs.asdict()) if hasattr(node.attrs, "asdict") else dict(node.attrs)
    except Exception:
        attrs = {}
    if "ome" in attrs and isinstance(attrs["ome"], dict):  # OME-Zarr 0.5 nests everything under "ome"
        merged = dict(attrs)
        merged.update(attrs["ome"])
        return merged
    return attrs


def _zarr_largest_array(group) -> tuple[Any, str] | None:
    best = None
    stack = [(group, "")]
    while stack:
        g, prefix = stack.pop()
        try:
            arrays = list(g.arrays())
            groups = list(g.groups())
        except Exception:
            continue
        for nm, arr in arrays:
            if 2 <= arr.ndim <= 5:
                size = int(np.prod(arr.shape))
                if best is None or size > best[0]:
                    best = (size, arr, f"{prefix}{nm}")
        for nm, sub in groups:
            stack.append((sub, f"{prefix}{nm}/"))
    return (best[1], best[2]) if best else None


def _read_zarr(p: Path, name: str, dataset: str | None, series: int | None = None, level: int = 0, virtual: bool = False) -> LumenImage:
    zarr = _need("zarr", ".zarr")
    node = zarr.open(str(p), mode="r")
    cal = Calibration()
    channel_names = None
    axes = None
    meta: dict[str, Any] = {}
    arr = None
    if hasattr(node, "arrays"):  # group
        attrs = _zarr_group_attrs(node)
        ms = attrs.get("multiscales")
        if dataset is None and not ms:
            # bioformats2raw layout: numbered sub-groups, each an OME-Zarr image (= one Bio-Formats series)
            try:
                subs = [nm for nm, g in node.groups() if _zarr_group_attrs(g).get("multiscales")]
            except Exception:
                subs = []
            if subs:
                subs.sort(key=lambda nm: (not nm.isdigit(), int(nm) if nm.isdigit() else nm))
                si = min(max(int(series or 0), 0), len(subs) - 1)
                img = _read_zarr(p / subs[si], name, None, None, level, virtual=virtual)
                img.path = str(p)
                img.metadata.update({"series_count": len(subs), "series": si, "series_name": f"{strip_image_ext(p.name)} #{subs[si]}",
                                     "series_names": [f"{strip_image_ext(p.name)} #{nm}" for nm in subs]})
                img.metadata["source"] = {"kind": "file", "path": str(p), "series": si, "level": int(level or 0)}
                return img
        if dataset is not None:
            arr = node[dataset]
            meta["dataset"] = dataset
            meta["source"] = {"kind": "file", "path": str(p), "dataset": dataset}  # so File ▸ Revert re-reads the same array
        elif ms:
            m0 = ms[0]
            li = min(max(int(level or 0), 0), len(m0["datasets"]) - 1)
            d0 = m0["datasets"][li]
            arr = node[d0["path"]]
            meta["dataset"] = d0["path"]
            meta["multiscale_levels"] = len(m0["datasets"])
            if len(m0["datasets"]) > 1:
                meta["levels"], meta["level"] = len(m0["datasets"]), li
                meta["source"] = {"kind": "file", "path": str(p), "level": li}
            ax = m0.get("axes")
            if ax:
                axes = "".join((a["name"] if isinstance(a, dict) else str(a))[0].upper() for a in ax)
                scale = None
                for tr in d0.get("coordinateTransformations", []) + m0.get("coordinateTransformations", []):
                    if tr.get("type") == "scale" and scale is None:
                        scale = tr["scale"]
                for i, a in enumerate(ax):
                    if not isinstance(a, dict):
                        continue
                    nm = a["name"].lower()
                    unit = a.get("unit")
                    sv = float(scale[i]) if scale and i < len(scale) else None
                    if nm == "x" and sv:
                        cal.pixel_width = sv
                        if unit:
                            cal.unit = _norm_unit(unit)
                    elif nm == "y" and sv:
                        cal.pixel_height = sv
                    elif nm == "z" and sv:
                        cal.pixel_depth = sv
                    elif nm == "t" and sv:
                        cal.frame_interval = sv
                        if unit:
                            cal.time_unit = {"second": "sec", "millisecond": "ms", "minute": "min"}.get(unit, unit)
            if axes is None or len(axes) != arr.ndim:
                # OME-Zarr 0.1 / 0.2 have no "axes" key (bioformats2raw <= 0.2, OMERO): the spec fixes the layout to
                # T C Z Y X – the same default _list_series_zarr uses, so the Series table and the opened image agree
                # (LumenImage's own guess would be T Z C Y X and swap channels and slices); a malformed "axes" entry
                # of the wrong length is treated the same way on both sides
                axes = {2: "YX", 3: "ZYX", 4: "CZYX", 5: "TCZYX"}.get(arr.ndim)
            omero = attrs.get("omero") or {}
            chans = omero.get("channels") or []
            if chans and all(c.get("label") for c in chans):
                channel_names = [str(c["label"]) for c in chans]
        else:
            found = _zarr_largest_array(node)
            if found is None:
                raise ValueError(f"No array found in {p.name}")
            arr, meta["dataset"] = found
    else:
        arr = node
    shape, ndim, src_dtype = tuple(int(v) for v in arr.shape), int(arr.ndim), np.dtype(arr.dtype)
    aattrs = {}
    try:
        aattrs = dict(arr.attrs.asdict()) if hasattr(arr.attrs, "asdict") else dict(arr.attrs)
    except Exception:
        pass
    if axes is None:
        dims = aattrs.get("_ARRAY_DIMENSIONS") or aattrs.get("axes")
        if dims and len(dims) == ndim:
            axes = "".join(str(d)[0].upper() for d in dims) if not isinstance(dims, str) else dims
    if axes is not None and (len(axes) != ndim or any(ch not in "TZCYXS" for ch in axes)):
        axes = None
    rgb = bool(aattrs.get("rgb", False)) and src_dtype == np.uint8
    lazy = None
    if virtual:
        try:
            key = meta.get("dataset") if hasattr(node, "arrays") else None
            lazy = zarr_source(p, key, axes or _guess_axes(shape, src_dtype, rgb), shape, src_dtype, rgb,
                               description=f"OME-Zarr level {meta['level']}" if "level" in meta else f"Zarr array {key or p.name}")
        except ValueError:  # axes that cannot be mapped onto T/Z/C without merging: read into memory
            lazy = None
    if lazy is not None:  # the chunks stay in the store: planes (and regions) are read on demand
        meta.setdefault("source", {"kind": "file", "path": str(p)})["virtual"] = True
        img = _lumen_from_lazy(LazyArray(lazy), name, rgb, cal, meta, p, channel_names)
    else:
        data = np.asarray(arr[...])
        img = LumenImage(data, name=name, axes=axes, rgb=rgb, calibration=cal, channel_names=channel_names, metadata=meta, path=str(p))
    try:  # omero display windows / colors
        omero = (_zarr_group_attrs(node) if hasattr(node, "arrays") else {}).get("omero") or {}
        for c, ch in enumerate(omero.get("channels") or []):
            if c < img.nc:
                win = ch.get("window") or {}
                if "start" in win and "end" in win:
                    img.display.channels[c].min, img.display.channels[c].max = float(win["start"]), float(win["end"])
                col = (ch.get("color") or "").lower()
                lut = {"ff0000": "Red", "00ff00": "Green", "0000ff": "Blue", "00ffff": "Cyan", "ff00ff": "Magenta", "ffff00": "Yellow", "ffffff": "Grays"}.get(col)
                if lut:
                    img.display.channels[c].lut = lut
    except Exception:
        pass
    return img


_LUT_HEX = {"Red": "FF0000", "Green": "00FF00", "Blue": "0000FF", "Cyan": "00FFFF", "Magenta": "FF00FF", "Yellow": "FFFF00", "Grays": "FFFFFF"}


def _write_zarr(img: LumenImage, p: Path) -> str:
    zarr = _need("zarr", ".zarr", verb="Writing")
    import shutil
    cal = img.calibration
    if p.suffix.lower() != ".zarr":
        p = p.with_name(p.name + ".zarr")
    if p.exists():
        shutil.rmtree(p)
    lazy = is_virtual(img.data)
    data = None if lazy else np.ascontiguousarray(np.transpose(img.data, (0, 2, 1, 3, 4)))  # OME-Zarr axis order: T C Z Y X
    shape = (img.nt, img.nc, img.nz, img.height, img.width)
    chunks = (1, 1, 1, min(img.height, 1024), min(img.width, 1024))
    try:
        g = zarr.open_group(str(p), mode="w", zarr_format=2)
    except TypeError:  # zarr 2.x
        g = zarr.open_group(str(p), mode="w")
    if hasattr(g, "create_array"):
        a = g.create_array("0", shape=shape, chunks=chunks, dtype=np.dtype(img.dtype))
    else:  # zarr 2.x
        a = g.create_dataset("0", shape=shape, chunks=chunks, dtype=np.dtype(img.dtype))
    if lazy:  # a virtual stack is written plane by plane
        for (t, z, c), plane in _lazy_iter_planes(img.data):
            a[t, c, z] = plane
    else:
        a[...] = data
    unit = _OME_UNITS.get(cal.unit)
    axes = [{"name": "t", "type": "time"}, {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"}, {"name": "y", "type": "space"}, {"name": "x", "type": "space"}]
    if unit:
        for a in axes[2:]:
            a["unit"] = unit
    if cal.frame_interval:
        axes[0]["unit"] = {"sec": "second", "s": "second", "ms": "millisecond", "min": "minute"}.get(cal.time_unit, cal.time_unit)
    scale = [cal.frame_interval or 1.0, 1.0, cal.pixel_depth, cal.pixel_height, cal.pixel_width]
    g.attrs["multiscales"] = [{
        "version": "0.4", "name": img.name, "axes": axes,
        "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": scale}]}],
        "type": "lumen", "metadata": {"description": "Written by Lumen"},
    }]
    lo, hi = img.value_range()
    g.attrs["omero"] = {
        "name": img.name, "version": "0.4",
        "channels": [{"label": ch.name, "color": _LUT_HEX.get(ch.lut, "FFFFFF"), "active": ch.visible,
                      "window": {"min": lo, "max": hi, "start": ch.min, "end": ch.max}} for ch in img.display.channels],
        "rdefs": {"model": "color" if img.display.mode == "composite" else "greyscale", "defaultT": img.display.t, "defaultZ": img.display.z},
    }
    return str(p)


# ============================================================================ video
def _read_video(p: Path, name: str) -> LumenImage:
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"Reading {p.suffix} needs: pip install imageio-ffmpeg") from e
    import imageio.v2 as iio2
    reader = iio2.get_reader(str(p), format="FFMPEG")
    try:
        md = reader.get_meta_data() or {}
        frames = [np.asarray(fr) for fr in reader]
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames in {p.name}")
    data = np.stack(frames)  # T Y X 3
    if data.ndim == 4 and data.shape[-1] == 3 and np.array_equal(data[..., 0], data[..., 1]) and np.array_equal(data[..., 1], data[..., 2]):
        data = data[..., 0]
        axes, rgb = "TYX", False
    else:
        axes, rgb = "TYXS", True
    cal = Calibration()
    fps = float(md.get("fps") or 0)
    if fps > 0:
        cal.frame_interval = 1.0 / fps
        cal.time_unit = "sec"
    meta = {"fps": fps, "codec": str(md.get("codec", "")), "duration": float(md.get("duration") or 0)}
    return LumenImage(data, name=name, axes=axes, rgb=rgb, calibration=cal, metadata=meta, path=str(p))


def video_frames(img: LumenImage, axis: str = "auto") -> list[np.ndarray]:
    """Render every frame of the stack along ``axis`` ('T', 'Z' or 'auto') to RGB."""
    from .render import render_rgb
    if axis == "auto":
        axis = "T" if img.nt > 1 else "Z"
    d = img.display
    if axis.upper() == "T":
        return [render_rgb(img, t=t, z=d.z) for t in range(img.nt)]
    return [render_rgb(img, t=d.t, z=z) for z in range(img.nz)]


def write_video(frames: list[np.ndarray], path: str | os.PathLike, fmt: str, fps: float = 10.0, quality: int = 8) -> str:
    """Write rendered RGB frames (H, W, 3) uint8 to avi/mp4/mov/mkv/webm or gif."""
    p = Path(path)
    fmt = fmt.lower().lstrip(".")
    p = _with_ext(p, fmt)
    frames = [np.ascontiguousarray(f if f.ndim == 3 else np.stack([f] * 3, -1)) for f in frames]
    if fmt == "gif":
        import imageio.v3 as iio
        iio.imwrite(str(p), np.stack(frames), duration=max(1, int(round(1000.0 / max(fps, 0.001)))), loop=0)
        return str(p)
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"Writing .{fmt} needs: pip install imageio-ffmpeg") from e
    import imageio.v2 as iio2
    # encoders need even frame sizes: pad by one pixel if needed
    H, W = frames[0].shape[:2]
    if H % 2 or W % 2:
        frames = [np.pad(f, ((0, H % 2), (0, W % 2), (0, 0)), mode="edge") for f in frames]
    if fmt == "avi":
        kw: dict[str, Any] = {"codec": "mjpeg", "pixelformat": "yuvj420p", "quality": quality}
    elif fmt == "webm":
        kw = {"codec": "libvpx-vp9", "pixelformat": "yuv420p", "quality": quality}
    else:
        kw = {"codec": "libx264", "pixelformat": "yuv420p", "quality": quality}
    w = iio2.get_writer(str(p), format="FFMPEG", mode="I", fps=float(fps), macro_block_size=1, **kw)
    try:
        for f in frames:
            w.append_data(f)
    finally:
        w.close()
    return str(p)


# ============================================================================ FITS
def _read_fits(p: Path, name: str) -> LumenImage:
    try:
        from astropy.io import fits
    except ImportError as e:
        raise RuntimeError("Reading .fits needs: pip install astropy") from e
    with fits.open(str(p), memmap=False) as hdul:
        hdu = next((h for h in hdul if h.data is not None and np.ndim(h.data) >= 2), None)
        if hdu is None:
            raise ValueError(f"No image data in {p.name}")
        data = np.asarray(hdu.data)
        hdr = hdu.header
        meta = {k: hdr[k] for k in ("OBJECT", "TELESCOP", "INSTRUME", "DATE-OBS", "EXPTIME", "BUNIT", "BITPIX") if k in hdr and isinstance(hdr[k], (str, int, float))}
    if data.dtype.byteorder not in ("=", "|"):
        data = data.astype(data.dtype.newbyteorder("="))
    data = data[..., ::-1, :]  # FITS stores the bottom row first
    axes = {2: "YX", 3: "ZYX", 4: "TZYX"}.get(data.ndim)
    return LumenImage(np.ascontiguousarray(data), name=name, axes=axes, metadata=meta, path=str(p))


def _write_fits(img: LumenImage, p: Path) -> str:
    try:
        from astropy.io import fits
    except ImportError as e:
        raise RuntimeError("Writing .fits needs: pip install astropy") from e
    data = np.squeeze(img.data[..., ::-1, :])
    if data.ndim < 2:
        data = img.data[0, 0, 0, ::-1, :]
    hdu = fits.PrimaryHDU(np.ascontiguousarray(data))
    hdu.header["ORIGIN"] = "Lumen"
    hdu.writeto(str(p), overwrite=True)
    return str(p)


# ============================================================================ MetaImage (.mha/.mhd) – pure numpy
_MET_TYPES = {"MET_UCHAR": np.uint8, "MET_CHAR": np.int8, "MET_USHORT": np.uint16, "MET_SHORT": np.int16, "MET_UINT": np.uint32,
              "MET_INT": np.int32, "MET_ULONG": np.uint32, "MET_LONG": np.int32, "MET_ULONG_LONG": np.uint64, "MET_LONG_LONG": np.int64,
              "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64}
_MET_NAMES = {np.dtype(np.uint8): "MET_UCHAR", np.dtype(np.uint16): "MET_USHORT", np.dtype(np.float32): "MET_FLOAT"}


def _read_metaimage(p: Path, name: str) -> LumenImage:
    header: dict[str, str] = {}
    data_start = 0
    with open(p, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                break
            try:
                s = line.decode("utf-8").strip()
            except UnicodeDecodeError:  # foreign header in a legacy 8-bit code page
                s = line.decode("latin-1").strip()
            if "=" in s:
                k, v = s.split("=", 1)
                header[k.strip()] = v.strip()
                if k.strip() == "ElementDataFile":
                    data_start = f.tell()
                    break
    if "DimSize" not in header:
        raise ValueError(f"{p.name}: not a MetaImage header")
    dims = [int(v) for v in header["DimSize"].split()]
    nchan = int(header.get("ElementNumberOfChannels", 1))
    dtype = np.dtype(_MET_TYPES[header.get("ElementType", "MET_UCHAR")])
    if header.get("BinaryDataByteOrderMSB", "False").lower() == "true":
        dtype = dtype.newbyteorder(">")
    compressed = header.get("CompressedData", "False").lower() == "true"
    datafile = header.get("ElementDataFile", "LOCAL")
    count = int(np.prod(dims)) * nchan
    if datafile.upper() == "LOCAL":
        with open(p, "rb") as f:
            f.seek(data_start + int(header.get("HeaderSize", 0) or 0))
            raw = f.read()
    elif datafile.upper() == "LIST":
        raise ValueError("MetaImage with ElementDataFile = LIST is not supported")
    else:
        raw = (p.parent / datafile).read_bytes()
        hs = int(header.get("HeaderSize", 0) or 0)
        if hs > 0:
            raw = raw[hs:]
        elif hs < 0:
            raw = raw[len(raw) - count * dtype.itemsize:]
    if compressed:
        import zlib
        raw = zlib.decompress(raw)
    arr = np.frombuffer(raw, dtype=dtype, count=count)
    shape = list(reversed(dims)) + ([nchan] if nchan > 1 else [])
    arr = arr.reshape(shape)
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("="))
    nd = len(dims)
    base = {2: "YX", 3: "ZYX", 4: "TZYX"}.get(nd)
    if base is None:
        raise ValueError(f"Unsupported MetaImage NDims {nd}")
    rgb = nchan == 3 and arr.dtype == np.uint8
    axes = base + ("S" if rgb else ("C" if nchan > 1 else ""))
    cal = Calibration()
    sp = header.get("ElementSpacing") or header.get("ElementSize")
    if sp:
        vals = [float(v) for v in sp.split()]
        cal.pixel_width = vals[0]
        cal.pixel_height = vals[1] if len(vals) > 1 else vals[0]
        cal.pixel_depth = vals[2] if len(vals) > 2 else 1.0
        cal.unit = "mm"
        m = re.search(r"Lumen unit=(\S+)", header.get("Comment", ""))
        if m:  # written by Lumen: restore the original unit (foreign files keep the MetaIO mm convention)
            u = {"um": "µm", "A": "Å"}.get(m.group(1), _norm_unit(m.group(1)))
            k = _TO_MM.get(u)
            if k:
                cal.pixel_width, cal.pixel_height, cal.pixel_depth = cal.pixel_width / k, cal.pixel_height / k, cal.pixel_depth / k
            cal.unit = u
    meta = {k: v for k, v in header.items() if k in ("ObjectType", "ElementType", "AnatomicalOrientation", "Offset", "TransformMatrix")}
    return LumenImage(np.ascontiguousarray(arr), name=name, axes=axes, rgb=rgb, calibration=cal, metadata=meta, path=str(p))


def _write_metaimage(img: LumenImage, p: Path, fmt: str) -> str:
    cal = img.calibration
    data = img.data  # T Z C Y X
    nchan = img.nc
    if img.nt > 1:
        arr = np.transpose(data, (0, 1, 3, 4, 2))  # T Z Y X C
        dims = [img.width, img.height, img.nz, img.nt]
    elif img.nz > 1:
        arr = np.transpose(data[0], (0, 2, 3, 1))  # Z Y X C
        dims = [img.width, img.height, img.nz]
    else:
        arr = np.transpose(data[0, 0], (1, 2, 0))  # Y X C
        dims = [img.width, img.height]
    arr = np.ascontiguousarray(arr)
    # MetaIO / ITK ElementSpacing is in millimetres: convert the spatial spacings (never the 4th, time, entry);
    # 'pixel' / unknown units are written verbatim and the source unit is kept in the user-defined Comment tag.
    unit = _norm_unit(cal.unit)
    k = _TO_MM.get(unit, 1.0)
    spacing = [cal.pixel_width * k, cal.pixel_height * k, cal.pixel_depth * k, cal.frame_interval or 1.0][:len(dims)]
    hdr = ["ObjectType = Image", f"NDims = {len(dims)}", "BinaryData = True", "BinaryDataByteOrderMSB = False", "CompressedData = False",
           "TransformMatrix = " + " ".join("1" if i == j else "0" for i in range(len(dims)) for j in range(len(dims))),
           "Offset = " + " ".join("0" for _ in dims), "CenterOfRotation = " + " ".join("0" for _ in dims),
           "ElementSpacing = " + " ".join(f"{v:g}" for v in spacing), "DimSize = " + " ".join(str(d) for d in dims)]
    if unit != "mm":
        tag = "Comment = Lumen unit=" + _ASCII_UNIT.get(unit, unit)
        try:
            tag.encode("ascii")
            hdr.append(tag)
        except UnicodeEncodeError:
            pass
    if nchan > 1:
        hdr.append(f"ElementNumberOfChannels = {nchan}")
    hdr.append(f"ElementType = {_MET_NAMES[np.dtype(arr.dtype)]}")
    if fmt == "mhd":
        p = _with_ext(p, "mhd")
        rawpath = p.with_name(p.stem + ".raw")
        hdr.append(f"ElementDataFile = {rawpath.name}")
        p.write_text("\n".join(hdr) + "\n", encoding="utf-8", newline="\n")
        arr.astype(arr.dtype.newbyteorder("<")).tofile(str(rawpath))
    else:
        p = _with_ext(p, "mha")
        hdr.append("ElementDataFile = LOCAL")
        with open(p, "wb") as f:
            f.write(("\n".join(hdr) + "\n").encode("ascii"))
            f.write(arr.astype(arr.dtype.newbyteorder("<")).tobytes())
    return str(p)


# ============================================================================ NRRD
def _read_nrrd(p: Path, name: str) -> LumenImage:
    try:
        import nrrd  # pynrrd
    except ImportError:
        nrrd = None
    if nrrd is not None:
        data, hdr = nrrd.read(str(p))
        arr = np.transpose(data, list(reversed(range(data.ndim))))  # Fortran (x, y, z) → (z, y, x)
        kinds = [str(k).lower() for k in hdr.get("kinds", [])]
        cal = Calibration()
        try:
            sd = hdr.get("space directions")
            if sd is not None:
                vecs = [v for v in sd if v is not None and not (isinstance(v, float) and np.isnan(v))]
                vecs = [np.asarray(v, float) for v in vecs if np.all(np.isfinite(np.asarray(v, float)))]
                if vecs:
                    mags = [float(np.linalg.norm(v)) for v in vecs]
                    cal.pixel_width, cal.pixel_height = mags[0], mags[1] if len(mags) > 1 else mags[0]
                    cal.pixel_depth = mags[2] if len(mags) > 2 else 1.0
                    cal.unit = _norm_unit((hdr.get("space units") or ["mm"])[0]) if hdr.get("space units") else "mm"
            elif hdr.get("spacings") is not None:
                sp = [float(v) for v in hdr["spacings"] if np.isfinite(float(v))]
                if sp:
                    cal.pixel_width, cal.pixel_height, cal.pixel_depth = sp[0], sp[1] if len(sp) > 1 else sp[0], sp[2] if len(sp) > 2 else 1.0
                    cal.unit = "mm"
        except Exception:
            pass
        axes = None
        if kinds and len(kinds) == data.ndim:
            m = {"space": None, "domain": None, "time": "T", "list": "C", "vector": "C", "rgb-color": "S", "rgba-color": "S", "3-color": "S"}
            rev = list(reversed(kinds))
            spatial = "ZYX"[-sum(1 for k in rev if m.get(k, None) is None):] if any(m.get(k, None) is None for k in rev) else ""
            out = []
            si = 0
            for k in rev:
                a = m.get(k, None)
                if a is None:
                    out.append(spatial[si]); si += 1
                else:
                    out.append(a)
            axes = "".join(out)
        rgb = bool(axes and "S" in axes and arr.dtype == np.uint8)
        return LumenImage(np.ascontiguousarray(arr), name=name, axes=axes, rgb=rgb, calibration=cal, path=str(p))
    try:
        import SimpleITK as sitk
    except ImportError as e:
        raise RuntimeError("Reading .nrrd needs: pip install pynrrd  (or SimpleITK)") from e
    im = sitk.ReadImage(str(p))
    arr = sitk.GetArrayFromImage(im)
    sp = im.GetSpacing()
    cal = Calibration(pixel_width=sp[0], pixel_height=sp[1] if len(sp) > 1 else sp[0], pixel_depth=sp[2] if len(sp) > 2 else 1.0, unit="mm")
    ncomp = im.GetNumberOfComponentsPerPixel()
    base = {2: "YX", 3: "ZYX", 4: "TZYX"}[arr.ndim - (1 if ncomp > 1 else 0)]
    rgb = ncomp == 3 and arr.dtype == np.uint8
    axes = base + ("S" if rgb else ("C" if ncomp > 1 else ""))
    return LumenImage(arr, name=name, axes=axes, rgb=rgb, calibration=cal, path=str(p))


# ============================================================================ generic fallback
def _read_bioio(p: Path, name: str, ext: str, series: int | None = None) -> LumenImage:
    bi = _bioio_open(p)
    if bi is None:
        raise RuntimeError(f"No reader for {ext or p.name!r}: pip install bioio (plus a bioio reader plugin for this format)")
    meta: dict[str, Any] = {}
    scenes = list(getattr(bi, "scenes", ()) or ())
    if len(scenes) > 1:
        si = min(max(int(series or 0), 0), len(scenes) - 1)
        try:
            bi.set_scene(si)
        except Exception:
            pass
        meta = {"series_count": len(scenes), "series": si, "series_name": str(scenes[si]), "series_names": [str(s) for s in scenes],
                "source": {"kind": "file", "path": str(p), "series": si}}
    data = np.asarray(bi.data)
    axes = str(getattr(bi.dims, "order", "TCZYX"))
    cal = Calibration()
    try:
        pps = bi.physical_pixel_sizes
        if pps.X:
            cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit = float(pps.X), float(pps.Y or pps.X), float(pps.Z or 1.0), "µm"
    except Exception:
        pass
    names = None
    try:
        names = [str(n) for n in bi.channel_names]
    except Exception:
        pass
    return LumenImage(data, name=name, axes=axes if len(axes) == data.ndim else None, calibration=cal, channel_names=names, metadata=meta, path=str(p))


# ============================================================================ raw & text
RAW_TYPES = {
    "8-bit": ("u1", 1), "16-bit unsigned": ("u2", 2), "16-bit signed": ("i2", 2), "32-bit unsigned": ("u4", 4), "32-bit signed": ("i4", 4),
    "32-bit real": ("f4", 4), "64-bit real": ("f8", 8), "24-bit RGB": ("u1", 3), "1-bit bitmap": ("u1", 0),
}


def read_raw(path: str | os.PathLike, width: int, height: int, image_type: str = "8-bit", offset: int = 0, n_images: int = 1,
             gap: int = 0, little_endian: bool = False, white_is_zero: bool = False, name: str | None = None) -> LumenImage:
    """Open a headerless raw file as an image or stack (ImageJ File ▸ Import ▸ Raw)."""
    p = Path(path)
    if image_type not in RAW_TYPES:
        raise ValueError(f"Unknown raw type {image_type!r}; choose one of {list(RAW_TYPES)}")
    code, itemsize = RAW_TYPES[image_type]
    if image_type == "1-bit bitmap":
        raise ValueError("1-bit bitmaps are not supported")
    rgb = image_type == "24-bit RGB"
    dtype = np.dtype(("<" if little_endian else ">") + code) if not rgb else np.dtype("u1")
    per_image = width * height * (3 if rgb else 1)
    bytes_per_image = per_image * dtype.itemsize
    size = p.stat().st_size
    avail = max(0, size - offset)
    stride = bytes_per_image + gap
    max_images = (avail + gap) // stride if stride > 0 else 0
    if n_images <= 0:
        n_images = max(1, max_images)
    n_images = min(n_images, max(1, max_images))
    if max_images < 1:
        raise ValueError(f"{p.name}: file too small for a {width}×{height} {image_type} image at offset {offset}")
    planes = []
    with open(p, "rb") as f:
        for i in range(n_images):
            f.seek(offset + i * stride)
            buf = f.read(bytes_per_image)
            if len(buf) < bytes_per_image:
                break
            planes.append(np.frombuffer(buf, dtype=dtype, count=per_image))
    data = np.stack(planes).reshape((len(planes), height, width, 3) if rgb else (len(planes), height, width))
    if data.dtype.byteorder == ">":
        data = data.astype(data.dtype.newbyteorder("="))
    data = data.copy()
    if white_is_zero and not rgb:
        if data.dtype.kind == "u":
            data = np.iinfo(data.dtype).max - data
        else:
            data = -data
    if image_type == "16-bit signed":
        data = data.astype(np.float32)
    img = LumenImage(data, name=name or p.name, axes="ZYXS" if rgb else "ZYX", rgb=rgb, path=str(p))
    img.metadata.update({"raw_type": image_type, "raw_offset": offset, "raw_gap": gap, "raw_little_endian": little_endian})
    img.metadata["source"] = {"kind": "raw", "path": str(p), **_raw_params(width, height, image_type, offset, n_images, gap, little_endian, white_is_zero)}
    return img


def _raw_params(width: int, height: int, image_type: str, offset: int, n_images: int, gap: int, little_endian: bool, white_is_zero: bool) -> dict[str, Any]:
    return {"width": int(width), "height": int(height), "image_type": image_type, "offset": int(offset), "n_images": int(n_images),
            "gap": int(gap), "little_endian": bool(little_endian), "white_is_zero": bool(white_is_zero)}


def read_raw_folder(folder: str | os.PathLike, ext: str, width: int, height: int, image_type: str = "8-bit", offset: int = 0, n_images: int = 1,
                    gap: int = 0, little_endian: bool = False, white_is_zero: bool = False, name: str | None = None) -> LumenImage:
    """Open every raw file with extension ``ext`` in ``folder`` as one stack (Import ▸ Raw… "Open all files in folder")."""
    folder = Path(folder)
    ext = ext.lower()
    files = sorted(f for f in folder.iterdir() if f.is_file() and f.suffix.lower() == ext)
    if not files:
        raise FileNotFoundError(f"No {ext or 'raw'} files in {folder}")
    imgs = [read_raw(f, width, height, image_type, offset, n_images, gap, little_endian, white_is_zero) for f in files]
    data = np.concatenate([im.data for im in imgs], axis=1)
    img = LumenImage(data, name=name or folder.name, axes="TZCYX", rgb=imgs[0].rgb, path=str(folder))
    img.metadata["source"] = {"kind": "raw_folder", "path": str(folder), "ext": ext,
                              **_raw_params(width, height, image_type, offset, n_images, gap, little_endian, white_is_zero)}
    return img


def read_text_image(path: str | os.PathLike, name: str | None = None) -> LumenImage:
    """A tab/comma/space separated numeric matrix → 32-bit image (ImageJ File ▸ Import ▸ Text Image)."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"{p.name} is empty")
    delim = "\t" if "\t" in lines[0] else ("," if "," in lines[0] else (";" if ";" in lines[0] else None))
    rows = []
    for ln in lines:
        parts = [s.strip() for s in (ln.split(delim) if delim else ln.split())]
        vals = []
        for s in parts:
            if s == "":
                continue
            try:
                vals.append(float(s))
            except ValueError:
                vals.append(np.nan)
        if vals:
            rows.append(vals)
    if rows and all(np.isnan(v) for v in rows[0]):
        rows = rows[1:]  # header line
    width = max(len(r) for r in rows)
    data = np.full((len(rows), width), np.nan, np.float32)
    for i, r in enumerate(rows):
        data[i, :len(r)] = r
    return LumenImage(data, name=name or p.name, path=str(p))


def write_text_image(img: LumenImage, path: str | os.PathLike) -> str:
    p = _with_ext(Path(path), "txt", ".txt", ".csv", ".tsv")
    plane = img.current_plane()
    delim = "," if p.suffix.lower() == ".csv" else "\t"
    fmt = "%d" if plane.dtype.kind in "ui" else "%.6g"
    np.savetxt(str(p), plane, fmt=fmt, delimiter=delim)
    return str(p)


# ============================================================================ sequences
_SEQ_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".pgm", ".ppm", ".pbm", ".pnm", ".webp", ".tga", ".npy", ".dcm"}


def _natural_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _looks_like_glob(pattern: str) -> bool:
    """A shell glob rather than a regex: has a ``*`` / ``?`` wildcard, none of the regex-only metacharacters and none of
    the regex idioms ``.*`` / ``.?`` / ``.+`` (which contain a wildcard too but mean "any characters")."""
    return (any(ch in pattern for ch in "*?") and not any(ch in pattern for ch in "\\()|^$+{}")
            and re.search(r"\.[*?+]", pattern) is None)


def compile_file_pattern(pattern: str | None) -> str:
    """Validate a user file pattern and return a regex for ``re.search``: a shell glob (``*.tif``, ``img_t*.tif``,
    ``img_?.tif``) is translated to a whole-name regex, anything else is a regular expression; one that does not
    compile raises ``ValueError`` (HTTP 400)."""
    pattern = str(pattern or "")
    if not pattern:
        return ""
    if _looks_like_glob(pattern):
        # decided BEFORE re.compile: the common globs img_t*.tif / img_*.tif are valid regexes too ('img_' + zero or
        # more 't' + any character + 'tif') that match none of the files they are meant for
        import fnmatch
        return "^" + fnmatch.translate(pattern)  # img_t*.tif -> ^(?s:img_t.*\.tif)\Z : the whole name, like pathlib.glob
    try:
        re.compile(pattern)
        return pattern
    except re.error as e:
        raise ValueError(f"Invalid file pattern {pattern!r}: {e} (give a glob such as img_*.tif or a regular expression "
                         "such as img_.*\\.tif$)") from e


def list_sequence_files(folder: str | os.PathLike, pattern: str = "", sort: str = "name", recursive: bool = False,
                        exts: set[str] | list[str] | tuple[str, ...] | None = None) -> list[Path]:
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    exts = set(exts) if exts else _SEQ_EXTS  # a list when it comes back from a metadata["source"] record (File ▸ Revert)
    pattern = compile_file_pattern(pattern)
    it = folder.rglob("*") if recursive else folder.iterdir()
    files = [f for f in it if f.is_file() and not f.name.startswith(".") and (f.suffix.lower() in exts or _full_ext(f) in exts)
             and (not pattern or re.search(pattern, f.name))]
    if sort in ("natural", "numeric"):
        files.sort(key=lambda f: _natural_key(str(f.relative_to(folder))))
    elif sort in ("modification time", "mtime", "date"):
        files.sort(key=lambda f: (f.stat().st_mtime, f.name))
    else:
        files.sort(key=lambda f: str(f.relative_to(folder)))
    return files


def read_sequence(folder: str | os.PathLike, pattern: str = "", start: int = 1, count: int = 0, step: int = 1, scale: float = 100.0,
                  sort: str = "name", axis: str = "Z", recursive: bool = False, name: str | None = None,
                  exts: set[str] | list[str] | tuple[str, ...] | None = None) -> LumenImage:
    """Open all images in a folder as one stack (ImageJ File ▸ Import ▸ Image Sequence).

    ``axis`` chooses where the files go: 'Z' (slices), 'T' (frames) or 'C' (channels).
    Each file may itself be a stack; files are concatenated along the chosen axis.
    ``exts`` widens the file-type filter (default: the common raster formats) – the Bio-Formats "group files"
    option passes ``IMAGE_EXTS`` so numbered STK / LSM / ND2 … siblings group too.
    """
    folder = Path(folder).expanduser()
    ext_set = set(exts) if exts else None
    files = list_sequence_files(folder, pattern, sort, recursive, exts=ext_set)
    if not files:
        raise FileNotFoundError(f"No images in {folder}" + (f" matching /{pattern}/" if pattern else ""))
    start = max(int(start), 1)
    step = max(int(step), 1)
    files = files[start - 1::step]
    if count and count > 0:
        files = files[:count]
    if not files:
        raise FileNotFoundError("The start/count/step selection matches no files")
    axis = (axis or "Z")[0].upper()
    ax_index = {"T": 0, "Z": 1, "C": 2}.get(axis, 1)
    arrays = []
    first: LumenImage | None = None
    names = []
    for f in files:
        im = read_image(f)
        if first is None:
            first = im
        data = im.data
        if scale and scale != 100.0:
            data = _rescale_stack(data, scale / 100.0)
        if arrays and data.shape[3:] != arrays[0].shape[3:]:
            raise ValueError(f"{f.name} has a different size ({data.shape[4]}×{data.shape[3]} vs {arrays[0].shape[4]}×{arrays[0].shape[3]})")
        if data.dtype != first.data.dtype:
            data = data.astype(np.float32) if first.data.dtype == np.float32 or data.dtype == np.float32 else data.astype(np.uint16)
        if arrays and data.dtype != arrays[0].dtype:
            arrays = [a.astype(np.float32) for a in arrays]
            data = data.astype(np.float32)
        if axis == "C" and data.shape[2] > 1 and data.shape[0] == 1 and data.shape[1] == 1:
            pass
        arrays.append(data)
        names.append(f.stem)
    assert first is not None
    # make the non-concatenation axes agree
    for i in (0, 1, 2):
        if i == ax_index:
            continue
        sizes = {a.shape[i] for a in arrays}
        if len(sizes) > 1:
            raise ValueError(f"Files differ along axis {'TZC'[i]}: {sorted(sizes)}")
    data = np.concatenate(arrays, axis=ax_index)
    rgb = first.rgb and data.shape[2] == 3 and data.dtype == np.uint8 and axis != "C"
    cal = Calibration(**first.calibration.to_dict())
    if scale and scale != 100.0:
        cal.pixel_width /= scale / 100.0
        cal.pixel_height /= scale / 100.0
    channel_names = names if axis == "C" and data.shape[2] == len(names) else [c.name for c in first.display.channels]
    img = LumenImage(data, name=name or folder.name, axes="TZCYX", rgb=rgb, calibration=cal, path=str(folder), channel_names=channel_names)
    img.metadata.update({"sequence_files": len(files), "sequence_first": files[0].name, "sequence_last": files[-1].name})
    # how to re-read this image identically (File ▸ Revert); only valid while img.path is still this folder
    img.metadata["source"] = {"kind": "sequence", "path": str(folder), "pattern": pattern, "start": start, "count": count, "step": step,
                              "scale": scale, "sort": sort, "axis": axis, "recursive": recursive,
                              "exts": sorted(ext_set) if ext_set else None}  # a JSON-safe list; Revert hands it back as exts=
    if not rgb and axis != "C":
        for c, ch in enumerate(first.display.channels[:img.nc]):
            img.display.channels[c].lut = ch.lut
    return img


def _rescale_stack(data: np.ndarray, factor: float) -> np.ndarray:
    from skimage.transform import resize
    from .registry import cast_to
    H, W = data.shape[3:]
    nh, nw = max(1, int(round(H * factor))), max(1, int(round(W * factor)))
    out = np.empty(data.shape[:3] + (nh, nw), data.dtype)
    for t in range(data.shape[0]):
        for z in range(data.shape[1]):
            for c in range(data.shape[2]):
                r = resize(data[t, z, c].astype(np.float32), (nh, nw), order=1, anti_aliasing=factor < 1, preserve_range=True)
                out[t, z, c] = cast_to(r, data.dtype)
    return out


# ============================================================================ writers
#: Bio-Formats exporter compression names → tifffile compression codecs (None = uncompressed)
TIFF_COMPRESSIONS: dict[str, str | None] = {"uncompressed": None, "none": None, "": None, "lzw": "lzw", "zlib": "zlib", "deflate": "zlib",
                                              "jpeg": "jpeg", "jpeg-2000": "jpeg2000", "jpeg2000": "jpeg2000", "jpeg-2000 lossy": "jpeg2000",
                                              "packbits": "packbits", "zstd": "zstd", "lzma": "lzma"}


def _write_tiff_pages(p: Path, pages: Any, shape: tuple[int, ...], dtype: Any, kw: dict[str, Any]) -> None:
    """Write TIFF pages from an iterator (a virtual stack streamed plane by plane); a failed or cancelled write leaves
    no half-written file behind."""
    import tifffile
    if kw.get("compression"):
        # tifffile's threaded encoder submits every page of an iterator up front (executor.map), which would pull the
        # whole stack into memory; a single worker consumes the pages one at a time (measured: +13 MB for 800 MB of
        # pages instead of +500 MB).  Uncompressed pages are never encoded and stream either way.
        kw.setdefault("maxworkers", 1)
    try:
        tifffile.imwrite(str(p), pages, shape=tuple(int(v) for v in shape), dtype=np.dtype(dtype), **kw)
    except BaseException:
        try:
            Path(p).unlink()
        except OSError:
            pass
        raise


def tiff_compression(name: Any, img: LumenImage | None = None) -> str | None:
    """The tifffile codec for an exporter compression name (``Uncompressed`` / LZW / zlib / JPEG / JPEG-2000 / PackBits / Zstd),
    checking that the image can take it (JPEG needs 8-bit data) and that the codec is available (imagecodecs)."""
    key = str(name or "").strip().lower()
    if key not in TIFF_COMPRESSIONS:
        raise ValueError(f"Unknown compression {name!r}; choose one of Uncompressed, LZW, zlib, JPEG, JPEG-2000, PackBits, Zstd")
    codec = TIFF_COMPRESSIONS[key]
    if codec == "jpeg" and img is not None and img.dtype != np.uint8:
        raise ValueError("JPEG compression needs 8-bit (or RGB) data – convert the image to 8-bit first, or choose zlib / LZW / JPEG-2000")
    if codec in ("jpeg", "jpeg2000", "zstd", "lzma"):
        try:
            import imagecodecs  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"{name} compression needs: pip install imagecodecs") from e
    return codec


#: the ImageJ version Lumen's command semantics are cross-validated against; written as ``ImageJ=…`` into the TIFF description
IMAGEJ_VERSION = "1.54p"
#: prefix of the processing-log lines in the ImageJ Info property (Fiji: Image ▸ Show Info)
PROCESSING_PREFIX = "Lumen processing:"


def _software_tag() -> str:
    from .. import __version__
    return f"Lumen {__version__}"


def _owning_session(img: LumenImage):
    """The session that holds ``img`` (its History lives there): the session the server is serving when the server
    module is loaded, else the shared default session.  None when the image belongs to neither (a plain
    ``write_image`` call on an image that was never added to a session)."""
    candidates = []
    app = sys.modules.get("lumen.server.app")
    if app is not None:
        candidates.append(getattr(app, "_CURRENT_SESSION", None))
    try:
        from .session import SESSION
        candidates.append(SESSION)
    except Exception:  # noqa: BLE001 – io must stay importable without the session machinery
        pass
    for s in candidates:
        if s is not None and getattr(s, "images", {}).get(img.id) is img:
            return s
    return None


def history_steps(img: LumenImage, history: Any = None) -> list[tuple[str, str]]:
    """``[(label, detail)]`` of the undo steps applied to ``img`` in this session, oldest first.  ``history`` is a
    ``lumen.core.undo.History`` (or anything with ``labels()``), an explicit list of ``(label, detail)`` pairs, or None
    to look the image up in the session that owns it (see :func:`_owning_session`)."""
    if history is None:
        s = _owning_session(img)
        history = s.histories.get(img.id) if s is not None else None
    if history is None:
        return []
    if hasattr(history, "labels"):
        lab = history.labels()
        labels, details = list(lab.get("undo", [])), list(lab.get("undo_detail", []))
        details += [""] * (len(labels) - len(details))
        return [(str(a), str(b)) for a, b in zip(labels, details)]
    return [(str(a), str(b)) for a, b in history]


def _same_file(a: Any, b: Any) -> bool:
    try:
        return Path(str(a)).expanduser().resolve() == Path(str(b)).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return str(a) == str(b)


def processing_log(img: LumenImage, history: Any = None, target: Any = None) -> list[str]:
    """The ``Lumen processing:`` lines for the Info property: the Lumen version and date, where the image came from
    (``Derived from:`` / ``Source:`` of ``img.metadata``, E7-17) and one numbered line per History step with its
    parameters (the non-default ones, as the History panel shows them)."""
    import time
    lines = [f"{PROCESSING_PREFIX} {_software_tag()}, written {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    derived = img.metadata.get("derived_from")
    if derived:
        lines.append(f"{PROCESSING_PREFIX} Derived from: {derived}")
    source = img.metadata.get("source_path") or img.path
    if source and target is not None and _same_file(source, target):
        source = None  # a re-save of a file that was never opened from disk: it is not its own source (E11)
    if source:
        lines.append(f"{PROCESSING_PREFIX} Source: {source}")
    steps = history_steps(img, history)
    for n, (label, detail) in enumerate(steps, 1):
        lines.append(f"{PROCESSING_PREFIX} {n}. {label}" + (f" ({detail})" if detail else ""))
    return lines


def processing_info(img: LumenImage, history: Any = None, target: Any = None) -> str:
    """The ImageJ Info text to write: the image's existing Info (``img.metadata["Info"]`` – what the file it was opened
    from carried, including earlier Lumen logs) followed by this session's :func:`processing_log`."""
    existing = str(img.metadata.get("Info") or "").rstrip()
    block = "\n".join(processing_log(img, history, target))
    return f"{existing}\n{block}" if existing else block


def write_image(img: LumenImage, path: str | os.PathLike, fmt: str | None = None, compress: bool = False, **options: Any) -> str:
    """Write ``img`` to ``path``.  ``fmt`` defaults to the path's extension, then ImageJ TIFF.

    ImageJ TIFF: the description says ``ImageJ=1.54p`` (:data:`IMAGEJ_VERSION`) and the Info property holds the
    processing log (:func:`processing_info`); ``history`` (a ``History`` or ``[(label, detail)]`` list) names the
    steps to log explicitly, otherwise they are looked up in the session that holds the image.

    Options: ``fps`` (videos / gif), ``axis`` ('T' | 'Z' | 'auto', videos), ``quality`` (videos), ``compression``
    (TIFF / OME-TIFF: Uncompressed, LZW, zlib, JPEG, JPEG-2000, PackBits, Zstd – ``compress=True`` is zlib),
    ``embed_metadata`` (OME-TIFF: the image's scalar metadata travels in Lumen's MapAnnotation and is restored on
    open) and ``description`` (OME-TIFF Image Description).  A virtual stack (:func:`is_virtual`) is written plane
    by plane for TIFF / OME-TIFF / npy / h5 / zarr – never read into memory – and the optional ``progress(fraction,
    message)`` / ``check_cancel()`` callables (``OpContext.progress`` / ``check_cancel``) are called per page.
    """
    from .render import render_rgb
    p = Path(path)
    fmt = (fmt or _full_ext(p).lstrip(".") or "tif").lower().lstrip(".")
    if fmt == "tiff":
        fmt = "tif"
    codec = tiff_compression(options["compression"], img) if options.get("compression") is not None else ("zlib" if compress else None)
    if fmt in ("tif", "tiff"):
        import tifffile
        cal = img.calibration
        lazy = is_virtual(img.data)
        if img.rgb:
            data = None if lazy else np.transpose(img.data, (0, 1, 3, 4, 2))  # T Z Y X S
            axes = "TZYXS"
        else:
            data = None if lazy else img.data
            axes = "TZCYX"
        unit = {"µm": "micron", "μm": "micron"}.get(cal.unit, cal.unit)
        try:
            unit.encode("ascii")
        except UnicodeEncodeError:
            unit = "unit"
        meta: dict[str, Any] = {"axes": axes, "unit": unit, "spacing": cal.pixel_depth, "finterval": cal.frame_interval}
        if cal.frame_interval and cal.time_unit and cal.time_unit != "sec":
            tunit = {"µs": "us"}.get(cal.time_unit, cal.time_unit)  # ImageJ: tunit= only when not seconds
            try:
                tunit.encode("ascii")
                meta["tunit"] = tunit
            except UnicodeEncodeError:
                pass
        if not img.rgb:
            meta["mode"] = {"composite": "composite", "single": "color", "grayscale": "grayscale"}[img.display.mode]
            lo_hi = []
            for ch in img.display.channels:
                lo_hi += [ch.min, ch.max]
            meta["Ranges"] = lo_hi  # per-channel ranges (CompositeImage)
            meta["LUTs"] = _ij_luts(img)  # reversed tables for inverted channels (ImageJ's Invert LUT)
            meta["Labels"] = _ij_labels(img)  # the image's slice labels, else the channel name on every plane (ImageJ shows them as slice labels)
            # ImageJ restores the window of a non-composite image from the description's min=/max= only
            cur = img.display.channels[min(img.display.c, img.nc - 1)]
            meta["min"], meta["max"] = float(cur.min), float(cur.max)
        res = (1.0 / cal.pixel_width, 1.0 / cal.pixel_height) if cal.pixel_width > 0 else None
        # provenance (E7-16): the description names the ImageJ version Lumen's semantics follow (tifffile would write
        # its own 1.11a default) and the Info property – Fiji's Image ▸ Show Info – carries the processing log
        meta["ImageJ"] = IMAGEJ_VERSION
        meta["Info"] = processing_info(img, history=options.get("history"), target=p)
        kw: dict[str, Any] = {"imagej": True, "metadata": meta, "software": _software_tag()}
        if res:
            kw["resolution"] = res
        if codec:
            kw["compression"] = codec
        p = _with_ext(p, "tif", ".tif", ".tiff")
        if lazy:  # streamed page by page (progress / cancel through the options)
            pages, pshape = page_iterator(img.data, img.rgb, options.get("progress"), options.get("check_cancel"))
            _write_tiff_pages(p, pages, pshape, img.dtype, kw)
        else:
            tifffile.imwrite(str(p), np.asarray(data), **kw)
        return str(p)
    if fmt in ("ome.tif", "ome.tiff", "ome"):
        import tifffile
        cal = img.calibration
        base = re.sub(r"\.(ome\.)?tiff?$", "", str(p), flags=re.I)
        p = Path(base + ".ome.tif")
        md: dict[str, Any] = {"axes": "TZCYX", "PhysicalSizeX": cal.pixel_width, "PhysicalSizeY": cal.pixel_height,
                              "PhysicalSizeZ": cal.pixel_depth, "PhysicalSizeXUnit": cal.unit, "PhysicalSizeYUnit": cal.unit,
                              "PhysicalSizeZUnit": cal.unit}
        if cal.frame_interval:
            md["TimeIncrement"] = cal.frame_interval
            md["TimeIncrementUnit"] = _TIME_TO_OME.get(cal.time_unit, cal.time_unit or "s")
        if options.get("description"):
            md["Description"] = str(options["description"])
        kw: dict[str, Any] = {"compression": codec} if codec else {}
        lazy = is_virtual(img.data)
        data = None if lazy else np.asarray(img.data)
        display_map = _lumen_display_map(img)
        if options.get("embed_metadata"):
            # Bio-Formats exporter "embed metadata": the image's scalar metadata rides along in the same MapAnnotation
            # as "meta.<key>" entries (read back into img.metadata by _ij_metadata_to_calibration)
            for k, v in img.metadata.items():
                if isinstance(v, (str, int, float, bool)) and k not in ("source",) and not isinstance(v, np.ndarray):
                    display_map[f"meta.{k}"] = str(v) if not isinstance(v, bool) else str(v).lower()
        if img.rgb:
            # one interleaved Channel with SamplesPerPixel=3, as Bio-Formats writes RGB; tifffile then
            # reports an 'S' axis on read so _read_tiff restores rgb=True (and Red/Green/Blue LUTs).
            # Per-sample names are meaningless (tifffile would keep only the first), so no Channel element.
            if not lazy:
                data = np.ascontiguousarray(np.transpose(data, (0, 1, 3, 4, 2)))  # T Z Y X S
            md["axes"] = "TZYXS"
            kw["photometric"] = "rgb"
            if options.get("embed_metadata"):
                md["MapAnnotation"] = {"Namespace": _LUMEN_OME_NS, "Value": {k: v for k, v in display_map.items() if k.startswith("meta.")}}
        else:
            # Standard OME: Channel Name + Color (RGBA int of the LUT's top colour, so Bio-Formats/OMERO colour the
            # channels).  Everything OME cannot express (LUT name, range, mode, inverted, visible) goes into a
            # MapAnnotation in Lumen's namespace, which stays schema-valid and is ignored by other readers.
            kw["photometric"] = "minisblack"  # never let tifffile guess RGB samples from a trailing X (or Y) of 3 / 4 pixels
            md["Channel"] = {"Name": [c.name for c in img.display.channels],
                             "Color": [_ome_color_from_lut(c.lut) for c in img.display.channels]}
            md["MapAnnotation"] = {"Namespace": _LUMEN_OME_NS, "Value": display_map}
        if lazy:
            pages, pshape = page_iterator(img.data, img.rgb, options.get("progress"), options.get("check_cancel"))
            _write_tiff_pages(p, pages, pshape, img.dtype, {"ome": True, "metadata": md, **kw})
        else:
            tifffile.imwrite(str(p), data, ome=True, metadata=md, **kw)
        return str(p)
    if fmt == "npy":
        # Lumen .npy files always hold the full (T, Z, C, Y, X) hyperstack so the reader never
        # has to guess axes (RGB is stored as C=3; the rgb flag itself is not representable).
        p = _with_ext(p, "npy")
        if is_virtual(img.data):  # written plane by plane into a memory-mapped file
            out = np.lib.format.open_memmap(str(p), mode="w+", dtype=np.dtype(img.dtype), shape=tuple(int(v) for v in img.data.shape))
            try:
                for (t, z, c), plane in _lazy_iter_planes(img.data):
                    out[t, z, c] = plane
                out.flush()
            finally:
                del out
            return str(p)
        np.save(str(p), np.ascontiguousarray(img.data))
        return str(p)
    if fmt in ("png", "jpg", "jpeg", "bmp", "webp", "tga", "pgm", "ppm", "pnm"):
        import imageio.v3 as iio
        ext = "jpg" if fmt == "jpeg" else fmt
        p = _with_ext(p, ext, ".jpeg" if ext == "jpg" else ext)
        plane = img.current_plane()
        grey_raw = (not img.rgb) and img.nc == 1 and plane.dtype in (np.uint8, np.uint16)
        if grey_raw and (fmt == "pgm" or (fmt == "png" and options.get("raw16", False))):
            iio.imwrite(str(p), plane)  # keep the 8/16-bit grey values instead of rendering
            return str(p)
        rgb = render_rgb(img)
        if fmt == "pgm":
            rgb = rgb[..., 0]
        iio.imwrite(str(p), rgb)
        return str(p)
    if fmt == "gif":
        frames = video_frames(img, options.get("axis", "auto"))
        return write_video(frames, p, "gif", fps=float(options.get("fps", 10)))
    if fmt in ("avi", "mp4", "mov", "mkv", "webm"):
        frames = video_frames(img, options.get("axis", "auto"))
        return write_video(frames, p, fmt, fps=float(options.get("fps", 10)), quality=int(options.get("quality", 8)))
    if fmt in ("nii", "nii.gz", "nifti"):
        return _write_nifti(img, p, gz=(fmt == "nii.gz"))
    if fmt in ("mrc", "rec", "map"):
        return _write_mrc(img, _with_ext(p, fmt, ".mrc", ".rec", ".map"))
    if fmt in ("h5", "hdf5"):
        return _write_hdf5(img, _with_ext(p, fmt, ".h5", ".hdf5"))
    if fmt in ("zarr", "ome.zarr"):
        return _write_zarr(img, p)
    if fmt in ("mha", "mhd"):
        return _write_metaimage(img, p, fmt)
    if fmt in ("fits", "fit", "fts"):
        return _write_fits(img, _with_ext(p, fmt, ".fits", ".fit", ".fts"))
    if fmt in ("raw", "bin"):
        p = _with_ext(p, fmt, ".raw", ".bin")
        data = np.transpose(img.data, (0, 1, 3, 4, 2)) if img.rgb else img.data
        np.ascontiguousarray(data).tofile(str(p))
        return str(p)
    if fmt in ("txt", "csv", "text"):
        return write_text_image(img, p if fmt != "text" else _with_ext(p, "txt", ".txt", ".csv", ".tsv"))
    raise ValueError(f"Unsupported format {fmt!r}")


def _ij_luts(img: LumenImage) -> list[np.ndarray]:
    """ImageJ (3, 256) colour tables, one per channel; an inverted channel gets the reversed table (ImageJ's Invert LUT)."""
    from .luts import get_lut
    out = []
    for ch in img.display.channels:
        try:
            table = get_lut(ch.lut)
        except KeyError:
            table = get_lut("Grays")
        if ch.inverted:
            table = table[::-1]
        out.append(np.ascontiguousarray(table.T).astype(np.uint8))
    return out


def _slice_labels_of(img: LumenImage) -> list[str] | None:
    """The image's per-plane slice labels (metadata["slice_labels"], ImageJ plane order) as a full-length list, or
    None when there are none – the same reading as stacks.slice_labels (a per-Z list is expanded over C and T)."""
    labels = img.metadata.get("slice_labels")
    if not isinstance(labels, (list, tuple)):
        return None
    n = img.nt * img.nz * img.nc
    lab = ["" if v is None else str(v) for v in labels]
    if len(lab) == img.nz and n != img.nz:
        lab = [lab[z] for _t in range(img.nt) for z in range(img.nz) for _c in range(img.nc)]
    lab = (lab + [""] * n)[:n]
    return lab if any(v.strip() for v in lab) else None


def _ij_labels(img: LumenImage) -> list[str]:
    """One ImageJ slice label per plane, in ImageJ's plane order (C fastest, then Z, then T): the image's slice labels
    when it has any (written verbatim, empties included, like Fiji's FileSaver), else the channel name on every plane
    (which _apply_ij_display turns back into channel names on open).

    tifffile stores labels as UTF-16 in the IJMetadata tag, exactly what ImageJ's TiffDecoder decodes,
    so non-ASCII names are safe here (only the ImageDescription values must stay ASCII).
    """
    labels = _slice_labels_of(img)
    if labels is not None:
        return labels
    names = [ch.name for ch in img.display.channels]
    return [names[c] for _ in range(img.nt) for _ in range(img.nz) for c in range(img.nc)]


def list_dir(path: str) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(path)
    entries = []
    for child in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower())):
        if child.name.startswith("."):
            continue
        try:
            is_dir = child.is_dir()
            if is_dir:
                is_image = child.name.lower().endswith((".zarr", ".ome.zarr"))
            else:
                is_image = child.suffix.lower() in IMAGE_EXTS or _full_ext(child) in IMAGE_EXTS
            entries.append({"name": child.name, "path": str(child), "dir": is_dir,
                            "size": 0 if is_dir else child.stat().st_size,
                            "image": is_image})
        except OSError:
            continue
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "entries": entries}


# ============================================================================ ImageJ .roi / .zip
# Binary layout of ImageJ's RoiEncoder/RoiDecoder (ij/io/RoiDecoder.java, public domain).
_IJ_VERSION = 228
_IJ_HEADER2_SIZE = 64
_IJ_TYPES = {"polygon": 0, "rect": 1, "oval": 2, "line": 3, "freeline": 4, "polyline": 5, "noroi": 6, "freehand": 7, "traced": 8, "angle": 9, "point": 10}
_IJ_TYPE_NAMES = {v: k for k, v in _IJ_TYPES.items()}
_IJ_SUB_PIXEL = 128
_IJ_SPLINE_FIT = 1
_IJ_SUBTYPE_ELLIPSE = 3
_IJ_SUBTYPE_ROTATED_RECT = 5
# header-1 offsets
_H_VERSION, _H_TYPE, _H_TOP, _H_LEFT, _H_BOTTOM, _H_RIGHT, _H_NCOORDS = 4, 6, 8, 10, 12, 14, 16
_H_X1, _H_Y1, _H_X2, _H_Y2 = 18, 22, 26, 30
_H_STROKE_WIDTH, _H_SHAPE_ROI_SIZE, _H_STROKE_COLOR, _H_FILL_COLOR, _H_SUBTYPE, _H_OPTIONS = 34, 36, 40, 44, 48, 50
_H_FLOAT_PARAM, _H_POSITION, _H_HEADER2_OFFSET, _H_COORDS = 52, 56, 60, 64
# header-2 offsets
_H2_C, _H2_Z, _H2_T, _H2_NAME_OFFSET, _H2_NAME_LENGTH, _H2_LABEL_COLOR, _H2_FONT_SIZE = 4, 8, 12, 16, 20, 24, 28
_H2_GROUP, _H2_OPACITY, _H2_IMAGE_SIZE, _H2_FLOAT_STROKE_WIDTH, _H2_PROPS_OFFSET, _H2_PROPS_LENGTH, _H2_COUNTERS_OFFSET = 30, 31, 32, 36, 40, 44, 48


def _color_to_argb(color: Any) -> int:
    if not color:
        return 0
    if isinstance(color, int):
        return int(color)
    s = str(color).lstrip("#")
    try:
        if len(s) == 6:
            return (0xFF << 24) | int(s, 16)
        if len(s) == 8:  # rrggbbaa
            return (int(s[6:8], 16) << 24) | int(s[:6], 16)
    except ValueError:
        pass
    named = {"red": 0xFFFF0000, "green": 0xFF00FF00, "blue": 0xFF0000FF, "yellow": 0xFFFFFF00, "cyan": 0xFF00FFFF, "magenta": 0xFFFF00FF, "white": 0xFFFFFFFF, "black": 0xFF000000}
    return named.get(s.lower(), 0)


def _argb_to_color(v: int) -> str | None:
    v &= 0xFFFFFFFF
    if (v >> 24) == 0:
        return None
    return f"#{v & 0xFFFFFF:06x}"


def encode_ij_roi(roi: dict[str, Any]) -> bytes:
    """Encode a Lumen ROI dict as an ImageJ ``.roi`` byte string."""
    from . import roi as roimod
    roi = roimod.normalize(roi)
    t = roi["type"]
    name = roi.get("name")
    pts: np.ndarray | None = None
    x1 = y1 = x2 = y2 = None
    options = 0
    subpixel = False
    stroke_width = float(roi.get("stroke_width", roi.get("width", 0)) or 0) if t in ("line", "polyline") else float(roi.get("stroke_width", 0) or 0)
    if t in ("rect", "oval"):
        ij_type = _IJ_TYPES[t]
        x, y, w, h = float(roi["x"]), float(roi["y"]), float(roi["w"]), float(roi["h"])
        subpixel = any(abs(v - round(v)) > 1e-6 for v in (x, y, w, h))
        left, top = int(np.floor(x)) if subpixel else int(round(x)), int(np.floor(y)) if subpixel else int(round(y))
        right, bottom = int(np.ceil(x + w)) if subpixel else int(round(x + w)), int(np.ceil(y + h)) if subpixel else int(round(y + h))
        if subpixel:
            x1, y1, x2, y2 = x, y, w, h  # XD, YD, WIDTHD, HEIGHTD
    elif t == "line":
        ij_type = _IJ_TYPES["line"]
        x1, y1, x2, y2 = float(roi["x1"]), float(roi["y1"]), float(roi["x2"]), float(roi["y2"])
        left, right = int(np.floor(min(x1, x2))), int(np.ceil(max(x1, x2)))
        top, bottom = int(np.floor(min(y1, y2))), int(np.ceil(max(y1, y2)))
        if right == left:
            right += 1
        if bottom == top:
            bottom += 1
    elif t in ("polygon", "freehand", "polyline", "point", "mask"):
        if t == "mask":
            pts = roimod.polygon_points(roi)
            ij_type = _IJ_TYPES["traced"]
        else:
            pts = np.asarray(roi.get("points", []), dtype=float).reshape(-1, 2)
            ij_type = _IJ_TYPES[t]
        if pts.size == 0:
            raise ValueError("ROI has no points")
        subpixel = bool(np.any(np.abs(pts - np.round(pts)) > 1e-6))
        if subpixel:
            left, top = int(np.floor(pts[:, 0].min())), int(np.floor(pts[:, 1].min()))
            right, bottom = int(np.ceil(pts[:, 0].max())), int(np.ceil(pts[:, 1].max()))
        else:
            left, top = int(round(pts[:, 0].min())), int(round(pts[:, 1].min()))
            right, bottom = int(round(pts[:, 0].max())), int(round(pts[:, 1].max()))
        if t == "point":
            right += 1
            bottom += 1
    else:
        raise ValueError(f"Cannot export ROI type {t!r} to ImageJ")
    n = 0 if pts is None else len(pts)
    shape_vals: np.ndarray | None = None
    if n > 65535 and pts is not None:
        # The 16-bit N_COORDINATES field holds at most 65535 points.  ImageJ stores bigger point ROIs as
        # count 0 + an int at SIZE (offset 18), and bigger outlines as a composite ShapeRoi (a java.awt.geom
        # path of float segments) – we do the same instead of writing a count that disagrees with the data.
        if t in ("polygon", "freehand", "mask"):
            segs = np.ones(n, np.float32)
            segs[0] = 0.0  # MOVETO, then LINETO…, CLOSE
            shape_vals = np.concatenate([np.column_stack([segs, pts[:, 0], pts[:, 1]]).ravel(), [4.0]]).astype(np.float32)
            ij_type, subpixel, pts, n = _IJ_TYPES["rect"], False, None, 0
        elif t != "point":
            raise ValueError(f"ImageJ .roi files can hold at most 65535 points for a {t!r} ROI ({n} given)")
    if subpixel:
        options |= _IJ_SUB_PIXEL
    if roi.get("spline"):
        options |= _IJ_SPLINE_FIT
    coords_size = 4 * len(shape_vals) if shape_vals is not None else n * 4 + (n * 8 if subpixel and pts is not None else 0)
    hdr2_offset = _H_COORDS + coords_size
    name_bytes = name.encode("utf-16-be") if name else b""
    total = hdr2_offset + _IJ_HEADER2_SIZE + len(name_bytes)
    buf = bytearray(total)
    buf[0:4] = b"Iout"
    struct.pack_into(">h", buf, _H_VERSION, _IJ_VERSION)
    buf[_H_TYPE] = ij_type
    struct.pack_into(">hhhh", buf, _H_TOP, _clip16(top), _clip16(left), _clip16(bottom), _clip16(right))
    if n > 65535:  # point ROI: RoiDecoder reads getUnsignedShort(N_COORDINATES) and falls back to getInt(SIZE) when it is 0
        struct.pack_into(">H", buf, _H_NCOORDS, 0)
        struct.pack_into(">i", buf, _H_X1, n)
    else:
        struct.pack_into(">H", buf, _H_NCOORDS, n)  # unsigned, like RoiEncoder.putShort / RoiDecoder.getUnsignedShort
    if x1 is not None:
        struct.pack_into(">ffff", buf, _H_X1, x1, y1, x2, y2)
    if shape_vals is not None:
        struct.pack_into(">i", buf, _H_SHAPE_ROI_SIZE, len(shape_vals))
        buf[_H_COORDS:_H_COORDS + coords_size] = shape_vals.astype(">f4").tobytes()
    struct.pack_into(">h", buf, _H_STROKE_WIDTH, _clip16(int(round(stroke_width))))
    struct.pack_into(">I", buf, _H_STROKE_COLOR, _color_to_argb(roi.get("color")) & 0xFFFFFFFF)
    struct.pack_into(">I", buf, _H_FILL_COLOR, _color_to_argb(roi.get("fill_color") or roi.get("fill")) & 0xFFFFFFFF)
    struct.pack_into(">h", buf, _H_OPTIONS, options)
    pos = roi.get("position") or {}
    struct.pack_into(">i", buf, _H_POSITION, 0)
    struct.pack_into(">i", buf, _H_HEADER2_OFFSET, hdr2_offset)
    if pts is not None and n:
        base1, base2 = _H_COORDS, _H_COORDS + 2 * n
        xi = np.round(pts[:, 0]).astype(int) - left if not subpixel else np.floor(pts[:, 0]).astype(int) - left
        yi = np.round(pts[:, 1]).astype(int) - top if not subpixel else np.floor(pts[:, 1]).astype(int) - top
        for i in range(n):
            struct.pack_into(">h", buf, base1 + i * 2, _clip16(int(xi[i])))
            struct.pack_into(">h", buf, base2 + i * 2, _clip16(int(yi[i])))
        if subpixel:
            base1 = _H_COORDS + 4 * n
            base2 = base1 + 4 * n
            for i in range(n):
                struct.pack_into(">f", buf, base1 + i * 4, float(pts[i, 0]))
                struct.pack_into(">f", buf, base2 + i * 4, float(pts[i, 1]))
    # header 2
    h2 = hdr2_offset
    if pos:
        struct.pack_into(">iii", buf, h2 + _H2_C, int(pos.get("c", -1)) + 1, int(pos.get("z", -1)) + 1, int(pos.get("t", -1)) + 1)
    if name_bytes:
        # NAME_LENGTH counts UTF-16 code units (Java String.length()), not code points: emoji take two
        struct.pack_into(">ii", buf, h2 + _H2_NAME_OFFSET, h2 + _IJ_HEADER2_SIZE, len(name_bytes) // 2)
        buf[h2 + _IJ_HEADER2_SIZE:h2 + _IJ_HEADER2_SIZE + len(name_bytes)] = name_bytes
    struct.pack_into(">f", buf, h2 + _H2_FLOAT_STROKE_WIDTH, stroke_width)
    return bytes(buf)


def _clip16(v: int) -> int:
    return int(max(-32768, min(32767, v)))


def decode_ij_roi(data: bytes, default_name: str | None = None) -> dict[str, Any]:
    """Decode an ImageJ ``.roi`` byte string into a Lumen ROI dict."""
    from . import roi as roimod
    if len(data) < 64 or data[0:4] != b"Iout":
        raise ValueError("Not an ImageJ ROI file")
    version = struct.unpack_from(">h", data, _H_VERSION)[0]
    ij_type = data[_H_TYPE]
    top, left, bottom, right = struct.unpack_from(">hhhh", data, _H_TOP)
    n = struct.unpack_from(">H", data, _H_NCOORDS)[0]
    if n == 0 and ij_type == _IJ_TYPES["point"]:
        # ImageJ stores point ROIs with > 65535 points as count 0 + int at SIZE (offset 18); for other types
        # that offset holds the X1/XD floats, so the fallback is restricted to points (as RoiEncoder writes it)
        n = max(struct.unpack_from(">i", data, _H_X1)[0], 0)
    x1, y1, x2, y2 = struct.unpack_from(">ffff", data, _H_X1)
    stroke_width = struct.unpack_from(">h", data, _H_STROKE_WIDTH)[0]
    shape_size = struct.unpack_from(">i", data, _H_SHAPE_ROI_SIZE)[0]
    stroke_color = struct.unpack_from(">i", data, _H_STROKE_COLOR)[0]
    fill_color = struct.unpack_from(">i", data, _H_FILL_COLOR)[0]
    subtype = struct.unpack_from(">h", data, _H_SUBTYPE)[0]
    options = struct.unpack_from(">h", data, _H_OPTIONS)[0]
    position = struct.unpack_from(">i", data, _H_POSITION)[0]
    hdr2 = struct.unpack_from(">i", data, _H_HEADER2_OFFSET)[0]
    subpixel = bool(options & _IJ_SUB_PIXEL) and version >= 222
    name = None
    c = z = t = 0
    float_stroke = 0.0
    if hdr2 > 0 and hdr2 + _IJ_HEADER2_SIZE <= len(data):
        c, z, t = struct.unpack_from(">iii", data, hdr2 + _H2_C)
        name_off, name_len = struct.unpack_from(">ii", data, hdr2 + _H2_NAME_OFFSET)
        if name_off > 0 and name_len > 0 and name_off + 2 * name_len <= len(data):
            name = data[name_off:name_off + 2 * name_len].decode("utf-16-be", "ignore")
        float_stroke = struct.unpack_from(">f", data, hdr2 + _H2_FLOAT_STROKE_WIDTH)[0]
    width = float_stroke if float_stroke > 0 else float(stroke_width)
    tname = _IJ_TYPE_NAMES.get(ij_type, "polygon")
    roi: dict[str, Any]
    if tname == "rect" and shape_size > 0:
        roi = _decode_shape_roi(data, shape_size, (left, top, right, bottom))
    elif tname in ("rect", "oval"):
        if subpixel:
            roi = {"type": tname, "x": float(x1), "y": float(y1), "w": float(x2), "h": float(y2)}
        else:
            roi = {"type": tname, "x": float(left), "y": float(top), "w": float(right - left), "h": float(bottom - top)}
        if tname == "rect" and subtype == _IJ_SUBTYPE_ROTATED_RECT and n >= 4:
            roi = {"type": "polygon", "points": _decode_coords(data, n, left, top, subpixel)}
    elif tname == "line":
        roi = {"type": "line", "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "width": max(width, 1.0)}
    else:
        pts = _decode_coords(data, n, left, top, subpixel)
        if tname in ("polygon",):
            roi = {"type": "polygon", "points": pts}
        elif tname in ("freehand", "traced"):
            roi = {"type": "freehand", "points": pts}
            if subtype == _IJ_SUBTYPE_ELLIPSE and not pts:
                roi = {"type": "oval", "x": float(min(x1, x2)), "y": float(min(y1, y2)), "w": float(abs(x2 - x1)), "h": float(abs(y2 - y1))}
        elif tname in ("polyline", "freeline", "angle"):
            roi = {"type": "polyline", "points": pts, "width": max(width, 1.0)}
        elif tname == "point":
            roi = {"type": "point", "points": pts}
        else:
            roi = {"type": "polygon", "points": pts}
    if name:
        roi["name"] = name
    elif default_name:
        roi["name"] = default_name
    col = _argb_to_color(stroke_color)
    if col:
        roi["color"] = col
    fcol = _argb_to_color(fill_color)
    if fcol:
        roi["fill_color"] = roi["fill"] = fcol  # 'fill' is the key the renderers read (selection.properties keeps both in sync)
    if width > 0 and tname not in ("line", "polyline", "freeline", "angle"):
        roi["stroke_width"] = width
    if c > 0 or z > 0 or t > 0:
        roi["position"] = {"t": max(t - 1, 0), "z": max(z - 1, 0), "c": max(c - 1, 0)}
    elif position > 0:
        roi["position"] = {"t": 0, "z": position - 1, "c": 0}
    if options & _IJ_SPLINE_FIT:
        roi["spline"] = True
    return roimod.normalize(roi)


def _decode_coords(data: bytes, n: int, left: int, top: int, subpixel: bool) -> list[list[float]]:
    if n <= 0:
        return []
    if _H_COORDS + 4 * n > len(data):
        raise ValueError(f"Corrupt ImageJ ROI: {n} coordinates do not fit in {len(data)} bytes")
    base1, base2 = _H_COORDS, _H_COORDS + 2 * n
    xs = [left + struct.unpack_from(">h", data, base1 + i * 2)[0] for i in range(n)]
    ys = [top + struct.unpack_from(">h", data, base2 + i * 2)[0] for i in range(n)]
    if subpixel:
        b1 = _H_COORDS + 4 * n
        b2 = b1 + 4 * n
        if b2 + 4 * n <= len(data):
            xs = [struct.unpack_from(">f", data, b1 + i * 4)[0] for i in range(n)]
            ys = [struct.unpack_from(">f", data, b2 + i * 4)[0] for i in range(n)]
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def _decode_shape_roi(data: bytes, size: int, bounds: tuple[int, int, int, int]) -> dict[str, Any]:
    """Composite ('shape') ROIs are stored as a java.awt.geom path; rasterise to a mask ROI."""
    from . import roi as roimod
    vals = [struct.unpack_from(">f", data, _H_COORDS + i * 4)[0] for i in range(size)]
    polys: list[list[list[float]]] = []
    cur: list[list[float]] = []
    i = 0
    while i < len(vals):
        seg = int(vals[i])
        i += 1
        if seg == 0:  # MOVETO
            if len(cur) >= 3:
                polys.append(cur)
            cur = [[vals[i], vals[i + 1]]]
            i += 2
        elif seg == 1:  # LINETO
            cur.append([vals[i], vals[i + 1]])
            i += 2
        elif seg == 2:  # QUADTO
            cur.append([vals[i + 2], vals[i + 3]])
            i += 4
        elif seg == 3:  # CUBICTO
            cur.append([vals[i + 4], vals[i + 5]])
            i += 6
        elif seg == 4:  # CLOSE
            if len(cur) >= 3:
                polys.append(cur)
            cur = []
        else:
            break
    if len(cur) >= 3:
        polys.append(cur)
    if len(polys) == 1:
        return {"type": "polygon", "points": polys[0]}
    left, top, right, bottom = bounds
    shape = (max(bottom, int(max(p[1] for poly in polys for p in poly)) + 2), max(right, int(max(p[0] for poly in polys for p in poly)) + 2))
    mask = np.zeros(shape, bool)
    for poly in polys:
        mask ^= roimod.to_mask({"type": "polygon", "points": poly}, shape)
    m = roimod.from_mask(mask)
    return m or {"type": "rect", "x": left, "y": top, "w": right - left, "h": bottom - top}


def read_ij_rois(path: str | os.PathLike) -> list[dict[str, Any]]:
    """Read an ImageJ ``.roi`` file or a ``.zip`` of them (as the Fiji ROI Manager saves)."""
    p = Path(path)
    if zipfile.is_zipfile(p):
        out = []
        with zipfile.ZipFile(p) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".roi"):
                    continue
                out.append(decode_ij_roi(zf.read(info), default_name=Path(info.filename).stem))
        return out
    return [decode_ij_roi(p.read_bytes(), default_name=p.stem)]


def write_ij_rois(rois: list[dict[str, Any]], path: str | os.PathLike) -> str:
    """Write ROIs as a ``.zip`` of ImageJ ``.roi`` files (or a single ``.roi`` when the path ends in .roi)."""
    p = Path(path)
    if p.suffix.lower() == ".roi":
        if len(rois) != 1:
            raise ValueError("A .roi file holds exactly one ROI – use .zip for several")
        p.write_bytes(encode_ij_roi(rois[0]))
        return str(p)
    p = _with_ext(p, "zip")
    used: set[str] = set()
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, r in enumerate(rois):
            base = re.sub(r"[\\/:*?\"<>|]", "_", str(r.get("name") or f"{i + 1:04d}")) or f"{i + 1:04d}"
            name = base
            k = 1
            while name.lower() in used:  # unique on case-insensitive file systems (APFS, NTFS) too
                name = f"{base}-{k}"
                k += 1
            used.add(name.lower())
            zf.writestr(f"{name}.roi", encode_ij_roi({**r, "name": r.get("name") or name}))
    return str(p)
