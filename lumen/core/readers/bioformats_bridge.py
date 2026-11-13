"""Optional Java Bio-Formats bridge: every format the real Bio-Formats reads – through ``bioio-bioformats`` (mode
``bioio``) or, without any Python package, through the Bio-Formats jars of a Fiji installation run as a ``java``
subprocess (mode ``fiji-jars``).

Mode ``fiji-jars`` (``LUMEN_BIOFORMATS_MODE=auto|bioio|fiji-jars``, default ``auto`` = ``bioio`` when it is
installed, else ``fiji-jars`` when a JDK and Fiji's ``jars/bio-formats`` are found – see
:mod:`lumen.core.readers.fiji_jars` for the discovery): Fiji ships no ``bioformats_tools`` (``ImageInfo`` /
``ImageConverter``), so a small Groovy (or BeanShell) runner over ``loci.formats.ImageReader`` lists the series
(``key=value`` lines: sizes, dimension order, pixel type, RGB, resolution levels, physical sizes, channel names,
the global metadata, plus the OME-XML Bio-Formats builds) and converts one series / resolution level to an
OME-TIFF (``loci.formats.out.OMETiffWriter``, plane by plane, calibration and channel names carried over) in a temp
folder, which :func:`lumen.core.io.read_image` then reads.  One ``java`` run (~1 s) per listing or read.

Mode ``bioio`` – install the extra and a JDK::

    pip install "lumen-imaging[bioformats]"      # bioio + bioio-bioformats (scyjava, JPype)
    # plus a Java 11+ runtime: Eclipse Temurin / OpenJDK (notarized builds – no macOS Gatekeeper prompt),
    # found through JAVA_HOME or ``java`` on PATH; without one, scyjava's cjdk helper fetches a JDK on first use.

The first open downloads the Bio-Formats jars through Maven into the user's ``~/.m2`` / ``~/.jgo`` cache (an
internet connection is needed once).  Everything here degrades gracefully: :func:`available` is ``False`` with a
reason in :func:`status` when the package or Java is missing, and nothing Java-related is imported until a file is
actually read.  :mod:`lumen.core.io` uses the bridge as the *last* fallback (after the native readers and the
plain ``bioio`` plugins) for unknown or failed formats; ``file.bioformats_config`` reports its availability.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import NativeSeries

PIP_EXTRA = 'pip install "lumen-imaging[bioformats]"'
JDK_HINT = "install a JDK 11+ (Eclipse Temurin / OpenJDK; set JAVA_HOME or put java on PATH)"
FIJI_HINT = "or, without any Python package, install Fiji (its jars/bio-formats folder is used; LUMEN_FIJI_JARS when it is not in a standard place) plus a JDK"
INSTALL_HINT = f"{PIP_EXTRA} and {JDK_HINT}; the first use downloads the Bio-Formats jars via Maven – {FIJI_HINT}"
PACKAGES = ("bioio", "bioio_bioformats")
ENV_MODE = "LUMEN_BIOFORMATS_MODE"
ENV_TIMEOUT = "LUMEN_BIOFORMATS_TIMEOUT"
MODES = ("auto", "bioio", "fiji-jars")
_status_cache: dict[str, Any] | None = None


def _importable(name: str) -> bool:
    if name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _java_available() -> tuple[bool, str]:
    """(found, description) – JAVA_HOME, ``java`` on PATH, or scyjava's cjdk auto-install helper."""
    home = os.environ.get("JAVA_HOME", "").strip()
    if home:
        exe = Path(home) / "bin" / ("java.exe" if os.name == "nt" else "java")
        if exe.exists():
            return True, f"JAVA_HOME={home}"
    exe = shutil.which("java")
    if exe:
        return True, f"java at {exe}"
    if _importable("cjdk"):
        return True, "no JDK installed – scyjava will download one with cjdk on first use"
    return False, "no JDK found (JAVA_HOME unset, no java on PATH)"


def mode_requested() -> str:
    """``LUMEN_BIOFORMATS_MODE`` normalised: ``auto`` (default), ``bioio`` or ``fiji-jars``."""
    s = os.environ.get(ENV_MODE, "").strip().lower().replace("_", "-")
    s = {"fiji": "fiji-jars", "jars": "fiji-jars", "subprocess": "fiji-jars", "bioio-bioformats": "bioio", "python": "bioio"}.get(s, s)
    return s if s in MODES else "auto"


def status(refresh: bool = False) -> dict[str, Any]:
    """``{"available", "package", "java", "detail", "install", "mode", "fiji"}`` – cached (``refresh=True`` re-checks).
    ``mode`` is the one reads use: ``"bioio"``, ``"fiji-jars"`` or ``None``; ``fiji`` describes the jars mode."""
    global _status_cache
    if _status_cache is not None and not refresh:
        return dict(_status_cache)
    package = all(_importable(n) for n in PACKAGES)
    java, java_detail = _java_available() if package else (False, "not checked")
    want = mode_requested()
    bioio_ok = bool(package and java)
    if want == "bioio":
        fiji = {"available": False, "detail": f"disabled ({ENV_MODE}=bioio)", "java": None, "java_version": "", "jar": None, "runner": None, "jars": None}
    else:
        fiji = fiji_status(refresh)
    if want == "bioio":
        mode = "bioio" if bioio_ok else None
    elif want == "fiji-jars":
        mode = "fiji-jars" if fiji["available"] else None
    else:
        mode = "bioio" if bioio_ok else ("fiji-jars" if fiji["available"] else None)
    if mode == "bioio":
        detail = f"bioio-bioformats installed, {java_detail}"
    elif mode == "fiji-jars":
        detail = f"{fiji['detail']} – no Python package needed"
    elif not package:
        missing = [n for n in PACKAGES if not _importable(n)]
        detail = f"{', '.join(missing)} not installed – {PIP_EXTRA}"
        if want != "bioio":
            detail += f"; {fiji['detail']}"
    else:
        detail = f"{java_detail} – {JDK_HINT}"
        if want != "bioio":
            detail += f"; {fiji['detail']}"
    _status_cache = {"available": mode is not None, "package": package, "java": bool(java) or mode == "fiji-jars", "detail": detail, "install": INSTALL_HINT,
                     "mode": mode, "fiji": fiji}
    return dict(_status_cache)


def available(refresh: bool = False) -> bool:
    return bool(status(refresh)["available"])


def reset() -> None:
    """Forget the cached status (tests / after installing the extra)."""
    global _status_cache
    _status_cache = None


# ---------------------------------------------------------------------------- bioio-bioformats access
def _open(path: str | os.PathLike):
    from bioio import BioImage  # type: ignore
    from bioio_bioformats import Reader  # type: ignore
    return BioImage(str(path), reader=Reader)


def _ome_xml(img: Any) -> str | None:
    for attr in ("ome_metadata", "metadata"):
        try:
            ome = getattr(img, attr)
        except Exception:
            continue
        if ome is None:
            continue
        if isinstance(ome, str) and ome.lstrip().startswith("<"):
            return ome
        to_xml = getattr(ome, "to_xml", None)
        if callable(to_xml):
            try:
                return str(to_xml())
            except Exception:
                continue
    return None


def _set_scene(img: Any, series: int) -> int:
    scenes = list(getattr(img, "scenes", ()) or ())
    si = min(max(int(series or 0), 0), max(len(scenes) - 1, 0))
    if len(scenes) > 1:
        img.set_scene(si)
    return si


def _levels(img: Any) -> list[tuple[int, int]]:
    """(height, width) per resolution level when the bioio reader exposes them."""
    try:
        levels = list(getattr(img, "resolution_levels", ()) or ())
        dims = getattr(img, "resolution_level_dims", None) or {}
        out = []
        for lv in levels:
            shape = dims.get(lv)
            if shape is None:
                continue
            order = str(img.dims.order)
            out.append((int(shape[order.index("Y")]), int(shape[order.index("X")])))
        return out
    except Exception:
        return []


def _series_from_bioimage(img: Any, si: int, name: str) -> NativeSeries:
    dims = img.dims
    axes = str(getattr(dims, "order", "TCZYX")).upper()
    shape = tuple(int(v) for v in getattr(dims, "shape", img.shape))
    dtype = np.dtype(img.dtype)
    sizes = dict(zip(axes, shape))
    rgb = "S" in axes and sizes.get("S", 1) in (3, 4) and dtype == np.uint8
    info = NativeSeries(index=si, name=name, shape=shape, axes=axes, dtype=dtype, rgb=rgb, format="Bio-Formats (Java bridge)")
    try:
        pps = img.physical_pixel_sizes
        if pps.X:
            info.pixel_width, info.pixel_height, info.unit = float(pps.X), float(pps.Y or pps.X), "µm"
            info.pixel_depth = float(pps.Z) if pps.Z else None
    except Exception:
        pass
    try:
        info.channel_names = [str(n) for n in img.channel_names]
    except Exception:
        pass
    xml = _ome_xml(img)
    info.metadata = {"reader": "bioio-bioformats", "series_count": len(list(getattr(img, "scenes", ()) or ()))}
    if xml:
        info.metadata["ome_xml"] = xml
    lv = _levels(img)
    if lv:
        info.metadata["levels"] = lv
    return info


def list_series(path: str | os.PathLike) -> list[NativeSeries]:
    """One :class:`NativeSeries` per Bio-Formats series (bioio scene), with the OME-XML in ``metadata["ome_xml"]``."""
    st = status()
    if not st["available"]:
        raise RuntimeError(f"Java Bio-Formats bridge not available: {st['detail']}")
    if st["mode"] == "fiji-jars":
        return list_series_fiji(path)
    img = _open(path)
    scenes = list(getattr(img, "scenes", ()) or ())
    out = []
    for i in range(max(1, len(scenes))):
        si = _set_scene(img, i)
        out.append(_series_from_bioimage(img, si, str(scenes[si]) if scenes else Path(path).stem))
    return out


def read(path: str | os.PathLike, series: int = 0, level: int = 0) -> tuple[np.ndarray, Any, list[str], dict[str, Any]]:
    """Read one series / resolution level: ``(data (T, Z, C, Y, X[, S]), Calibration, channel names, metadata)``.

    The calibration comes from the OME metadata Bio-Formats produces (physical sizes with their units, time
    increment), falling back to bioio's ``physical_pixel_sizes``; ``metadata`` carries the OME-XML, the series
    count / name and the plane timings (``planes``).
    """
    st = status()
    if not st["available"]:
        raise RuntimeError(f"Java Bio-Formats bridge not available: {st['detail']}")
    if st["mode"] == "fiji-jars":
        return read_fiji(path, series, level)
    from ..image import Calibration
    img = _open(path)
    si = _set_scene(img, series)
    scenes = list(getattr(img, "scenes", ()) or ())
    levels = _levels(img)
    lv = min(max(int(level or 0), 0), max(len(levels) - 1, 0))
    if lv and hasattr(img, "set_resolution_level"):
        try:
            img.set_resolution_level(list(img.resolution_levels)[lv])
        except Exception:
            lv = 0
    info = _series_from_bioimage(img, si, str(scenes[si]) if scenes else Path(path).stem)
    axes = info.axes
    want = "TZCYX" + ("S" if "S" in axes else "")
    get = getattr(img, "get_image_data", None)
    if callable(get):
        data = np.asarray(get(want))
        axes = want
    else:
        data = np.asarray(img.data)
    if len(axes) != data.ndim:
        axes = ""
    cal = Calibration()
    names = list(info.channel_names)
    meta: dict[str, Any] = {"format": info.format, "reader": "bioio-bioformats", "series": si, "series_name": info.name, "level": lv}
    if len(scenes) > 1:
        meta.update({"series_count": len(scenes), "series_names": [str(s) for s in scenes], "source": {"kind": "file", "path": str(path), "series": si}})
    xml = info.metadata.get("ome_xml")
    applied = False
    if xml:
        from ..io import _apply_ome_pixels, _ome_images_info
        images = _ome_images_info(xml)
        if images:
            ome = images[min(si, len(images) - 1)]
            _apply_ome_pixels(cal, ome.get("pixels", {}))
            applied = cal.scaled
            if ome.get("channel_names"):
                names = [n or f"C{i + 1}" for i, n in enumerate(ome["channel_names"])]
            meta["ome_xml"] = xml
            if ome.get("planes"):
                meta["planes"] = ome["planes"]
            if ome.get("acquisition_date"):
                meta["acquisition_date"] = ome["acquisition_date"]
            for k, v in (ome.get("global") or {}).items():
                if isinstance(v, (str, int, float, bool)) and len(meta) < 2000:
                    meta.setdefault(k, v)
    if not applied and info.pixel_width:
        cal.pixel_width, cal.pixel_height, cal.unit = info.pixel_width, info.pixel_height or info.pixel_width, "µm"
        if info.pixel_depth:
            cal.pixel_depth = info.pixel_depth
    if lv and levels:
        scale = levels[0][1] / max(levels[lv][1], 1)
        cal.pixel_width *= scale
        cal.pixel_height *= scale
    meta["axes"] = axes or str(getattr(img.dims, "order", ""))
    return data, cal, names, meta


# ---------------------------------------------------------------------------- mode "fiji-jars": Fiji's Bio-Formats jars + a JDK, no Python package
#: the runner (Groovy; the BeanShell twin below is used when only bsh is available): ``<mode> <file> <xml out> [series level <ome-tiff out>]``
#: prints ``key=value`` lines – the format, the series count and per series the sizes, dimension order, pixel type,
#: RGB / interleaving, resolution levels, physical sizes, time increment, channel names and the global metadata –
#: writes the OME-XML Bio-Formats built to ``<xml out>`` and, in ``convert`` mode, one series / level as OME-TIFF.
FIJI_RUNNER_GROOVY = r"""
import loci.formats.*
import loci.formats.out.OMETiffWriter
import loci.formats.services.OMEXMLService
import loci.common.services.ServiceFactory

def mode = args[0]; def path = args[1]; def xmlOut = args[2]
def q = { v -> v == null ? "" : (v.value().toString() + " " + v.unit().getSymbol()) }
def e = { s -> s == null ? "" : s.toString().replace("\r", " ").replace("\n", " ").replace("\t", " ") }
def r = new ChannelFiller(new ImageReader())
r.setFlattenedResolutions(false)
def meta = MetadataTools.createOMEXMLMetadata()
r.setMetadataStore(meta)
r.setId(path)
println "format=" + e(r.getFormat())
println "series_count=" + r.getSeriesCount()
int nm = 0
for (k in r.getGlobalMetadata().keySet()) { if (nm++ >= 500) break; def v = e(r.getGlobalMetadata().get(k)); println "global." + e(k) + "=" + (v.length() > 500 ? v.substring(0, 500) : v) }
for (int s = 0; s < r.getSeriesCount(); s++) {
    r.setSeries(s); r.setResolution(0)
    def p = "series." + s + "."
    println p + "name=" + e(meta.getImageName(s))
    println p + "sizeX=" + r.getSizeX(); println p + "sizeY=" + r.getSizeY(); println p + "sizeZ=" + r.getSizeZ(); println p + "sizeC=" + r.getSizeC(); println p + "sizeT=" + r.getSizeT()
    println p + "dimension_order=" + r.getDimensionOrder()
    println p + "pixel_type=" + FormatTools.getPixelTypeString(r.getPixelType())
    println p + "rgb=" + r.isRGB(); println p + "rgb_channel_count=" + r.getRGBChannelCount(); println p + "interleaved=" + r.isInterleaved()
    println p + "little_endian=" + r.isLittleEndian(); println p + "indexed=" + r.isIndexed(); println p + "resolutions=" + r.getResolutionCount()
    for (int lv = 0; lv < r.getResolutionCount(); lv++) { r.setResolution(lv); println p + "level." + lv + "=" + r.getSizeX() + "x" + r.getSizeY() }
    r.setResolution(0)
    println p + "physical_size_x=" + q(meta.getPixelsPhysicalSizeX(s)); println p + "physical_size_y=" + q(meta.getPixelsPhysicalSizeY(s))
    println p + "physical_size_z=" + q(meta.getPixelsPhysicalSizeZ(s)); println p + "time_increment=" + q(meta.getPixelsTimeIncrement(s))
    for (int c = 0; c < meta.getChannelCount(s); c++) println p + "channel." + c + "=" + e(meta.getChannelName(s, c))
}
if (mode == "convert") {
    int s = Math.min(Math.max(args[3] as int, 0), r.getSeriesCount() - 1)
    r.setSeries(s)
    int lv = Math.min(Math.max(args[4] as int, 0), r.getResolutionCount() - 1)
    r.setResolution(lv)
    def out = args[5]
    def om = MetadataTools.createOMEXMLMetadata()
    def name = e(meta.getImageName(s))
    MetadataTools.populateMetadata(om, 0, name ? name : "series " + s, r.isLittleEndian(), r.getDimensionOrder(), FormatTools.getPixelTypeString(r.getPixelType()),
                                   r.getSizeX(), r.getSizeY(), r.getSizeZ(), r.getSizeC(), r.getSizeT(), r.getRGBChannelCount())
    if (meta.getPixelsPhysicalSizeX(s) != null) om.setPixelsPhysicalSizeX(meta.getPixelsPhysicalSizeX(s), 0)
    if (meta.getPixelsPhysicalSizeY(s) != null) om.setPixelsPhysicalSizeY(meta.getPixelsPhysicalSizeY(s), 0)
    if (meta.getPixelsPhysicalSizeZ(s) != null) om.setPixelsPhysicalSizeZ(meta.getPixelsPhysicalSizeZ(s), 0)
    if (meta.getPixelsTimeIncrement(s) != null) om.setPixelsTimeIncrement(meta.getPixelsTimeIncrement(s), 0)
    int nch = Math.min(meta.getChannelCount(s), om.getChannelCount(0))
    for (int c = 0; c < nch; c++) { def n = meta.getChannelName(s, c); if (n != null && n.length() > 0) om.setChannelName(n, 0, c) }
    def w = new OMETiffWriter()
    w.setMetadataRetrieve(om)
    w.setInterleaved(r.isInterleaved())
    w.setId(out)
    int n = r.getImageCount()
    for (int i = 0; i < n; i++) w.saveBytes(i, r.openBytes(i))
    w.close()
    println "series=" + s; println "level=" + lv; println "planes=" + n; println "output=" + out
}
def xml = new ServiceFactory().getInstance(OMEXMLService.class).getOMEXML(meta)
new File(xmlOut).write(xml, "UTF-8")
r.close()
println "done=1"
"""

FIJI_RUNNER_BSH = r"""
import loci.formats.*;
import loci.formats.out.OMETiffWriter;
import loci.formats.services.OMEXMLService;
import loci.common.services.ServiceFactory;
import java.io.*;
import java.util.*;

String q(Object v) { if (v == null) return ""; return v.value().toString() + " " + v.unit().getSymbol(); }
String e(Object s) { if (s == null) return ""; return s.toString().replace("\r", " ").replace("\n", " ").replace("\t", " "); }
String mode = bsh.args[0]; String path = bsh.args[1]; String xmlOut = bsh.args[2];
r = new ChannelFiller(new ImageReader());
r.setFlattenedResolutions(false);
meta = MetadataTools.createOMEXMLMetadata();
r.setMetadataStore(meta);
r.setId(path);
print("format=" + e(r.getFormat()));
print("series_count=" + r.getSeriesCount());
int nm = 0;
Hashtable gm = r.getGlobalMetadata();
for (Iterator it = gm.keySet().iterator(); it.hasNext() && nm < 500; nm++) { Object k = it.next(); String v = e(gm.get(k)); print("global." + e(k) + "=" + (v.length() > 500 ? v.substring(0, 500) : v)); }
for (int s = 0; s < r.getSeriesCount(); s++) {
    r.setSeries(s); r.setResolution(0);
    String p = "series." + s + ".";
    print(p + "name=" + e(meta.getImageName(s)));
    print(p + "sizeX=" + r.getSizeX()); print(p + "sizeY=" + r.getSizeY()); print(p + "sizeZ=" + r.getSizeZ()); print(p + "sizeC=" + r.getSizeC()); print(p + "sizeT=" + r.getSizeT());
    print(p + "dimension_order=" + r.getDimensionOrder());
    print(p + "pixel_type=" + FormatTools.getPixelTypeString(r.getPixelType()));
    print(p + "rgb=" + r.isRGB()); print(p + "rgb_channel_count=" + r.getRGBChannelCount()); print(p + "interleaved=" + r.isInterleaved());
    print(p + "little_endian=" + r.isLittleEndian()); print(p + "indexed=" + r.isIndexed()); print(p + "resolutions=" + r.getResolutionCount());
    for (int lv = 0; lv < r.getResolutionCount(); lv++) { r.setResolution(lv); print(p + "level." + lv + "=" + r.getSizeX() + "x" + r.getSizeY()); }
    r.setResolution(0);
    print(p + "physical_size_x=" + q(meta.getPixelsPhysicalSizeX(s))); print(p + "physical_size_y=" + q(meta.getPixelsPhysicalSizeY(s)));
    print(p + "physical_size_z=" + q(meta.getPixelsPhysicalSizeZ(s))); print(p + "time_increment=" + q(meta.getPixelsTimeIncrement(s)));
    for (int c = 0; c < meta.getChannelCount(s); c++) print(p + "channel." + c + "=" + e(meta.getChannelName(s, c)));
}
if (mode.equals("convert")) {
    int s = Math.min(Math.max(Integer.parseInt(bsh.args[3]), 0), r.getSeriesCount() - 1);
    r.setSeries(s);
    int lv = Math.min(Math.max(Integer.parseInt(bsh.args[4]), 0), r.getResolutionCount() - 1);
    r.setResolution(lv);
    String out = bsh.args[5];
    om = MetadataTools.createOMEXMLMetadata();
    String name = e(meta.getImageName(s));
    MetadataTools.populateMetadata(om, 0, name.length() > 0 ? name : "series " + s, r.isLittleEndian(), r.getDimensionOrder(), FormatTools.getPixelTypeString(r.getPixelType()),
                                   r.getSizeX(), r.getSizeY(), r.getSizeZ(), r.getSizeC(), r.getSizeT(), r.getRGBChannelCount());
    if (meta.getPixelsPhysicalSizeX(s) != null) om.setPixelsPhysicalSizeX(meta.getPixelsPhysicalSizeX(s), 0);
    if (meta.getPixelsPhysicalSizeY(s) != null) om.setPixelsPhysicalSizeY(meta.getPixelsPhysicalSizeY(s), 0);
    if (meta.getPixelsPhysicalSizeZ(s) != null) om.setPixelsPhysicalSizeZ(meta.getPixelsPhysicalSizeZ(s), 0);
    if (meta.getPixelsTimeIncrement(s) != null) om.setPixelsTimeIncrement(meta.getPixelsTimeIncrement(s), 0);
    int nch = Math.min(meta.getChannelCount(s), om.getChannelCount(0));
    for (int c = 0; c < nch; c++) { String n = meta.getChannelName(s, c); if (n != null && n.length() > 0) om.setChannelName(n, 0, c); }
    w = new OMETiffWriter();
    w.setMetadataRetrieve(om);
    w.setInterleaved(r.isInterleaved());
    w.setId(out);
    int n = r.getImageCount();
    for (int i = 0; i < n; i++) w.saveBytes(i, r.openBytes(i));
    w.close();
    print("series=" + s); print("level=" + lv); print("planes=" + n); print("output=" + out);
}
String xml = new ServiceFactory().getInstance(OMEXMLService.class).getOMEXML(meta);
Writer wr = new OutputStreamWriter(new FileOutputStream(xmlOut), "UTF-8"); wr.write(xml); wr.close();
r.close();
print("done=1");
"""

_BF_DTYPES = {"int8": np.int8, "uint8": np.uint8, "int16": np.int16, "uint16": np.uint16, "int32": np.int32, "uint32": np.uint32, "float": np.float32,
              "double": np.float64, "bit": np.uint8}
_info_cache: dict[str, tuple[float, dict[str, str]]] = {}


def fiji_status(refresh: bool = False) -> dict[str, Any]:
    """The ``fiji-jars`` mode on its own: ``{"available", "detail", "java", "java_version", "jar", "runner", "jars"}``."""
    from . import fiji_jars
    st = fiji_jars.status(refresh)
    jar = st["found"].get("bioformats")
    runner = "groovy" if st["found"].get("groovy") else ("beanshell" if st["found"].get("beanshell") else None)
    ok = st["java"] is not None and bool(jar) and runner is not None
    if ok:
        detail = f"Fiji's Bio-Formats jars ({Path(jar).name} in {Path(jar).parent}) with {st['java_version'] or 'java'} through a {runner} runner"
    elif st["java"] is None:
        detail = f"Fiji jars mode: {st['java_detail']}"
    elif not jar:
        detail = f"Fiji jars mode: no Bio-Formats jars ({st['jars_detail']})"
    else:
        detail = "Fiji jars mode: no groovy / bsh jar next to the Bio-Formats jars (one is needed as the runner)"
    return {"available": ok, "detail": detail, "java": st["java"], "java_version": st["java_version"], "jar": jar, "runner": runner, "jars": st["jars"]}


def _timeout() -> float:
    try:
        return max(5.0, float(os.environ.get(ENV_TIMEOUT, "900") or 900))
    except ValueError:
        return 900.0


def _parse_kv(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith(" "):
            k, v = line.split("=", 1)
            k = k.strip()
            if k and k not in out:
                out[k] = v.strip()
    return out


def run_fiji_runner(mode: str, path: str | os.PathLike, *, series: int = 0, level: int = 0, out: str | os.PathLike | None = None,
                    nbytes: int | None = None) -> dict[str, Any]:
    """Run the Bio-Formats runner (``info`` or ``convert``) and return its ``key=value`` output as a dict, plus
    ``ome_xml``, ``seconds`` and ``command``.  Raises ``RuntimeError`` with the JVM's stderr when it fails."""
    import shutil
    import tempfile
    from . import fiji_jars
    fs = fiji_status()
    if not fs["available"]:
        raise RuntimeError(f"Bio-Formats (Fiji jars) not available: {fs['detail']}")
    work = Path(tempfile.mkdtemp(prefix="lumen-bf-"))
    try:
        home = work / "home"
        home.mkdir()
        if fs["runner"] == "groovy":
            runner, main, comps = work / "runner.groovy", "groovy.ui.GroovyMain", ["groovy", "bioformats"]
            runner.write_text(FIJI_RUNNER_GROOVY, encoding="utf-8")
        else:
            runner, main, comps = work / "runner.bsh", "bsh.Interpreter", ["beanshell", "bioformats"]
            runner.write_text(FIJI_RUNNER_BSH, encoding="utf-8")
        cp, missing = fiji_jars.classpath_for(comps)
        if missing:
            raise RuntimeError(f"Bio-Formats (Fiji jars): missing jars {', '.join(missing)}")
        xml_out = work / "meta.ome.xml"
        opts = fiji_jars.base_options(headless=True, home=home, heap=fiji_jars.heap_option(nbytes)) + [fiji_jars.logback_option(work)]
        args = opts + ["-cp", os.pathsep.join(cp), main, str(runner), mode, str(Path(path).resolve()), str(xml_out)]
        if mode == "convert":
            args += [str(int(series or 0)), str(int(level or 0)), str(out)]
        res = fiji_jars.run_java(args, cwd=home, timeout=_timeout(), java=fs["java"])
        info: dict[str, Any] = dict(_parse_kv(res.stdout))
        if not res.ok or info.get("done") != "1":
            err = fiji_jars.filter_stderr(res.stderr)
            tail = "\n".join((err or res.stdout).splitlines()[-12:])
            raise RuntimeError(f"Bio-Formats (Fiji jars) could not read {Path(path).name}: {res.error or f'the JVM exited with code {res.returncode}'}"
                               + (f"\n{tail}" if tail else ""))
        info["ome_xml"] = xml_out.read_text(encoding="utf-8", errors="replace") if xml_out.is_file() else None
        info["seconds"] = res.seconds
        info["command"] = res.command
        return info
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _quantity(s: str | None) -> tuple[float | None, str]:
    """``"0.25 µm"`` → ``(0.25, "µm")``; empty → ``(None, "")``."""
    if not s:
        return None, ""
    parts = s.split(None, 1)
    try:
        v = float(parts[0])
    except ValueError:
        return None, ""
    return v, (parts[1].strip() if len(parts) > 1 else "")


def _series_from_info(info: dict[str, Any], i: int, path: Path) -> NativeSeries:
    p = f"series.{i}."

    def g(k: str, d: str = "") -> str:
        return str(info.get(p + k, d))

    sx, sy, sz, sc, st_ = (int(g(k, "1") or 1) for k in ("sizeX", "sizeY", "sizeZ", "sizeC", "sizeT"))
    dtype = np.dtype(_BF_DTYPES.get(g("pixel_type", "uint8"), np.uint8))
    rgbc = int(g("rgb_channel_count", "1") or 1)
    rgb = g("rgb") == "true" and rgbc in (3, 4) and dtype == np.uint8
    if rgb:
        shape: tuple[int, ...] = (st_, sz, max(sc // rgbc, 1), sy, sx, rgbc)
        axes = "TZCYXS"
    else:
        shape, axes = (st_, sz, sc, sy, sx), "TZCYX"
    name = g("name") or (f"{path.stem} #{i + 1}" if int(info.get("series_count", 1) or 1) > 1 else path.stem)
    out = NativeSeries(index=i, name=name, shape=shape, axes=axes, dtype=dtype, rgb=rgb, format="Bio-Formats (Fiji jars)")
    px, ux = _quantity(g("physical_size_x"))
    py, _uy = _quantity(g("physical_size_y"))
    pz, _uz = _quantity(g("physical_size_z"))
    if px:
        out.pixel_width, out.pixel_height, out.unit = px, py or px, ux or "µm"
        out.pixel_depth = pz
    ti, tu = _quantity(g("time_increment"))
    if ti:
        out.frame_interval, out.time_unit = ti, tu or "s"
    out.channel_names = [str(info.get(f"{p}channel.{c}", "") or "") for c in range(sc // (rgbc if rgb else 1)) if f"{p}channel.{c}" in info]
    if rgb:
        out.channel_names = []
    out.metadata = {"reader": "fiji-jars", "series_count": int(info.get("series_count", 1) or 1), "bioformats_format": str(info.get("format", "")),
                    "dimension_order": g("dimension_order", "XYCZT"), "little_endian": g("little_endian") == "true", "interleaved": g("interleaved") == "true"}
    nres = int(g("resolutions", "1") or 1)
    if nres > 1:
        levels = []
        for lv in range(nres):
            dims = g(f"level.{lv}")
            if "x" in dims:
                w, h = dims.split("x", 1)
                levels.append((int(h), int(w)))
        if levels:
            out.metadata["levels"] = levels
    for k, v in info.items():
        if k.startswith("global.") and len(out.metadata) < 600:
            out.metadata[k[7:]] = v
    if info.get("ome_xml"):
        out.metadata["ome_xml"] = info["ome_xml"]
    return out


def _cached_info(path: Path) -> dict[str, Any] | None:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _info_cache.get(str(path))
    return hit[1] if hit and hit[0] == mtime else None


def list_series_fiji(path: str | os.PathLike) -> list[NativeSeries]:
    """``fiji-jars`` mode: one :class:`NativeSeries` per Bio-Formats series (one ``java`` run, cached per file mtime)."""
    p = Path(path)
    info = _cached_info(p)
    if info is None:
        info = run_fiji_runner("info", p)
        try:
            _info_cache[str(p)] = (p.stat().st_mtime, info)
        except OSError:
            pass
    n = int(info.get("series_count", 1) or 1)
    return [_series_from_info(info, i, p) for i in range(max(n, 1))]


def read_fiji(path: str | os.PathLike, series: int = 0, level: int = 0) -> tuple[np.ndarray, Any, list[str], dict[str, Any]]:
    """``fiji-jars`` mode: convert one series / level to a temp OME-TIFF and read it – ``(data (T, Z, C, Y, X[, S]),
    Calibration, channel names, metadata)`` like :func:`read`."""
    import shutil
    import tempfile
    from ..io import is_virtual, read_image
    p = Path(path)
    cached = _cached_info(p)
    nbytes = None
    if cached is not None:
        try:
            q = f"series.{int(series or 0)}."
            n = 1
            for k in ("sizeX", "sizeY", "sizeZ", "sizeC", "sizeT"):
                n *= int(cached.get(q + k, 1) or 1)
            nbytes = n * np.dtype(_BF_DTYPES.get(cached.get(q + "pixel_type", "uint8"), np.uint8)).itemsize
        except (TypeError, ValueError):
            nbytes = None
    work = Path(tempfile.mkdtemp(prefix="lumen-bf-read-"))
    try:
        out = work / "series.ome.tif"
        info = run_fiji_runner("convert", p, series=series, level=level, out=out, nbytes=nbytes)
        try:
            _info_cache[str(p)] = (p.stat().st_mtime, info)
        except OSError:
            pass
        si, lv = int(info.get("series", series or 0)), int(info.get("level", level or 0))
        native = _series_from_info(info, si, p)
        img = read_image(out, name=native.name)
        data = np.asarray(img.data)
        if is_virtual(img.data) or isinstance(data, np.memmap):
            data = np.array(data)
        data = np.ascontiguousarray(data)
        if img.rgb:
            data = np.moveaxis(data, 2, -1)[:, :, None]  # (T, Z, 1, Y, X, S)
            axes, names = "TZCYXS", []
        else:
            axes, names = "TZCYX", [c.name for c in img.display.channels]
    finally:
        shutil.rmtree(work, ignore_errors=True)
    cal = img.calibration
    n_series = int(info.get("series_count", 1) or 1)
    meta: dict[str, Any] = {"format": "Bio-Formats (Fiji jars)", "reader": "fiji-jars", "series": si, "series_name": native.name, "level": lv, "axes": axes,
                            "bioformats_format": native.metadata.get("bioformats_format", ""), "seconds": round(float(info.get("seconds", 0.0)), 3)}
    if n_series > 1:
        meta.update({"series_count": n_series, "series_names": [_series_from_info(info, i, p).name for i in range(n_series)],
                     "source": {"kind": "file", "path": str(p), "series": si}})
    if lv:
        meta["source"] = {"kind": "file", "path": str(p), "series": si, "level": lv}
    if info.get("ome_xml"):
        meta["ome_xml"] = info["ome_xml"]
        from ..io import _ome_images_info
        images = _ome_images_info(info["ome_xml"])
        if images and si < len(images):
            ome = images[si]
            if ome.get("planes"):
                meta["planes"] = ome["planes"]
            if ome.get("acquisition_date"):
                meta["acquisition_date"] = ome["acquisition_date"]
    for k, v in native.metadata.items():
        if k not in ("ome_xml", "reader", "series_count", "levels") and isinstance(v, (str, int, float, bool)) and len(meta) < 2000:
            meta.setdefault(k, v)
    return data, cal, names, meta
