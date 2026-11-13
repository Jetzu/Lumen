"""File menu gaps: internal clipboard, next/previous/recent files, ImageJ sample images,
.lut import/export, table/text/coordinate/stack-list imports and zip/text exports.

ImageJ parity for File ▸ New ▸ Internal Clipboard, File ▸ Open Next, File ▸ Open Recent,
File ▸ Open Samples, File ▸ Import ▸ LUT… / Results… / Table… / Text File… / XY Coordinates… /
Stack From List… and File ▸ Save As ▸ ZIP… / Text….
"""
from __future__ import annotations

import csv
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .. import roi as roimod
from ..image import LumenImage
from ..io import IMAGE_EXTS, read_image, strip_image_ext, write_image
from ..luts import get_lut, read_lut_file, register_lut, write_lut_file
from ..registry import OpResult, Param, op

FILE = "File"
IMPORT = "File/Import"
EXPORT = "File/Export"
_NOIMG = dict(kind="image", requires_image=False, scope_param=False, preview=False, roi_aware=False)
_ZIP_MAGIC = b"PK\x03\x04"  # OOXML (.xlsx) workbook, possibly saved under a .xls name
_OLE_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"  # legacy binary Excel 97-2003 (BIFF) workbook


def _expand(path: str) -> Path:
    p = Path(str(path or "").strip()).expanduser()
    if not str(p) or str(p) == ".":
        raise ValueError("Give a path")
    return p


def _existing(path: str) -> Path:
    p = _expand(path)
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def _natural_key(s: str) -> list[Any]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _is_image_file(p: Path) -> bool:
    n = p.name.lower()
    return p.is_file() and not n.startswith(".") and any(n.endswith(e) for e in IMAGE_EXTS)


def _recent(session) -> list[str]:
    rec = getattr(session, "recent", None)
    return [str(p) for p in rec] if isinstance(rec, (list, tuple)) else []


def _add_recent(session, path: str) -> None:
    fn = getattr(session, "add_recent", None)
    if callable(fn):
        fn(str(path))


# ============================================================================ File ▸ New ▸ Internal Clipboard
@op("file.new_from_clipboard", "New ▸ Internal Clipboard", FILE, **_NOIMG, keywords=["clipboard", "paste new", "new image from copy"],
    description="Open the internal clipboard (the pixels copied with Edit ▸ Copy / Cut) as a new image; a non-rectangular copy becomes its selection.",
    params=[Param("title", "string", "Clipboard"),
            Param("outside", "choice", "Keep pixels", label="Outside the copied selection", choices=["Keep pixels", "Background"],
                  help="Pixels of the copied rectangle that lie outside a non-rectangular selection")])
def new_from_clipboard(ctx, title="Clipboard", outside="Keep pixels"):
    s = ctx.session
    clip = s.settings.get("clipboard")
    if not clip or clip.get("data") is None:
        raise ValueError("The internal clipboard is empty – use Edit ▸ Copy first")
    data = np.asarray(clip["data"])
    if data.ndim == 2:
        data = data[None]
    mask = clip.get("mask")
    rgb = bool(clip.get("rgb")) and data.shape[0] == 3 and data.dtype == np.uint8
    data = data.copy()
    if mask is not None and outside == "Background":
        bg = float(s.settings.get("background", 0))
        data[:, ~np.asarray(mask, bool)] = np.array(bg).astype(data.dtype) if data.dtype != np.float32 else bg
    img = LumenImage(data, name=title or "Clipboard", axes="CYX", rgb=rgb)
    res = OpResult(new_images=[img], message=f"{img.width}×{img.height} px, {img.bit_depth}")
    if mask is not None:
        r = roimod.from_mask(np.asarray(mask, bool), "Clipboard")
        if r is not None:
            res.roi = r
    return res


# ============================================================================ Open Next / Previous / Recent
def _neighbour(ctx, step: int, close_current: bool) -> OpResult:
    img = ctx.img
    if not img.path:
        raise ValueError("The current image was not opened from a file")
    cur = Path(img.path)
    folder = cur.parent
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder {folder} no longer exists")
    files = sorted((f for f in folder.iterdir() if _is_image_file(f)), key=lambda f: _natural_key(f.name))
    names = [f.name for f in files]
    if cur.name in names:
        i = names.index(cur.name)
    else:
        # the current file is gone / not an image extension: start from where it would sort
        i = sum(1 for n in names if _natural_key(n) < _natural_key(cur.name)) - (1 if step > 0 else 0)
    if not files or (len(files) == 1 and files[0].name == cur.name):
        return OpResult(message="No other image files in this folder")
    nxt = files[(i + step) % len(files)]
    new = read_image(nxt)
    new.path = str(nxt)
    _add_recent(ctx.session, str(nxt))
    return OpResult(new_images=[new], close_image=bool(close_current),
                    message=f"Opened {nxt.name} ({(i + step) % len(files) + 1}/{len(files)})")


@op("file.open_next", "Open Next", FILE, kind="image", scope_param=False, preview=False, roi_aware=False, shortcut="Shift+O",
    keywords=["next file", "folder", "browse"],
    params=[Param("close_current", "bool", True, label="Close the current image")],
    description="Open the next image file (sorted by name) in the folder of the current image, closing the current one like ImageJ's File ▸ Open Next.")
def open_next(ctx, close_current=True):
    return _neighbour(ctx, 1, close_current)


@op("file.open_previous", "Open Previous", FILE, kind="image", scope_param=False, preview=False, roi_aware=False, shortcut="Alt+Shift+O",
    keywords=["previous file", "folder", "browse"],
    params=[Param("close_current", "bool", True, label="Close the current image")],
    description="Open the previous image file (sorted by name) in the folder of the current image.")
def open_previous(ctx, close_current=True):
    return _neighbour(ctx, -1, close_current)


@op("file.open_recent", "Open Recent", FILE, **_NOIMG, keywords=["recent files", "history", "reopen"],
    params=[Param("index", "int", 0, min=0, help="0 = the most recently opened file"),
            Param("path", "string", "", help="Open this path instead of the indexed entry")],
    description="Re-open one of the recently opened files (newest first).")
def open_recent(ctx, index=0, path=""):
    s = ctx.session
    if str(path or "").strip():
        p = _existing(path)
    else:
        rec = _recent(s)
        if not rec:
            raise ValueError("No recent files")
        if index >= len(rec):
            raise ValueError(f"Only {len(rec)} recent file(s)")
        p = Path(rec[index])
        if not p.exists():
            raise FileNotFoundError(f"{p} no longer exists")
    img = read_image(p)
    img.path = str(p)
    _add_recent(s, str(p))
    return OpResult(new_images=[img], message=f"Opened {img.name} {img.shape[1:]} {img.bit_depth}")


# ============================================================================ ImageJ sample images
# Name → candidate file names on https://imagej.net/images/ (ij/IJ_Props.txt, "Open Samples" menu).
IJ_SAMPLES: dict[str, list[str]] = {
    "AuPbSn 40": ["AuPbSn40.jpg"],
    "Bat Cochlea Volume": ["bat-cochlea-volume.zip", "bat-cochlea-volume.tif"],
    "Bat Cochlea Renderings": ["bat-cochlea-renderings.zip", "bat-cochlea-renderings.tif"],
    "Blobs (25K)": ["blobs.gif"],
    "Boats": ["boats.gif"],
    "Bridge": ["bridge.gif"],
    "Cardio (RGB DICOM)": ["cardio.dcm.zip"],
    "Cell Colony": ["Cell_Colony.jpg"],
    "Clown": ["clown.jpg"],
    "Confocal Series": ["confocal-series.zip", "confocal-series.tif"],
    "Dot Blot": ["Dot_Blot.jpg"],
    "Embryos": ["embryos.jpg"],
    "Fluorescent Cells": ["FluorescentCells.zip", "FluorescentCells.tif"],
    "Flybrain": ["flybrain.zip", "flybrain.tif"],
    "Gel": ["gel.gif"],
    "HeLa Cells": ["hela-cells.zip", "hela-cells.tif"],
    "Image with Overlay": ["ImageWithOverlay.zip", "ImageWithOverlay.tif"],
    "Leaf": ["leaf.jpg"],
    "Line Graph": ["LineGraph.jpg"],
    "M51 Galaxy": ["m51.zip", "m51.tif"],
    "Mitosis": ["Spindly-GFP.zip", "mitosis.tif"],
    "Neuron": ["Rat_Hippocampal_Neuron.zip", "Rat_Hippocampal_Neuron.tif"],
    "Nile Bend": ["NileBend.jpg"],
    "Organ of Corti": ["organ-of-corti.zip", "organ-of-corti.tif"],
    "Particles": ["particles.gif"],
    "T1 Head": ["t1-head.zip", "t1-head.tif"],
    "T1 Head Renderings": ["t1-head-renderings.zip", "t1-head-renderings.tif"],
    "TEM Filter": ["TEM_filter_sample.jpg"],
    "Tree Rings": ["Tree_Rings.jpg"],
}
SAMPLE_BASES = ["https://imagej.net/images/", "https://imagej.nih.gov/ij/images/", "https://wsr.imagej.net/images/"]
IJ_SAMPLE_NAMES = list(IJ_SAMPLES)


def _samples_dir() -> Path:
    from ..io import lumen_home
    d = Path(os.environ.get("LUMEN_SAMPLES_DIR") or (lumen_home() / "samples"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, target: Path, timeout: float = 60.0) -> None:
    """Fetch ``url`` into ``target`` (replaced by tests)."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Lumen/0.1 (ImageJ sample fetch)"})
    tmp = target.with_name(target.name + ".part")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        shutil.copyfileobj(resp, out, 1 << 20)
    os.replace(tmp, target)


def _fetch_sample(fname: str, cache: Path) -> Path:
    target = cache / fname
    if target.is_file() and target.stat().st_size > 0:
        return target
    errors = []
    for base in SAMPLE_BASES:
        url = base + fname
        try:
            _download(url, target)
            if target.is_file() and target.stat().st_size > 0:
                return target
            errors.append(f"{url}: empty download")
        except Exception as e:  # URLError, HTTPError, socket.timeout, OSError …
            errors.append(f"{url}: {e}")
            try:
                target.with_name(target.name + ".part").unlink(missing_ok=True)
            except OSError:
                pass
    raise RuntimeError("Could not download the ImageJ sample " + fname + " (are you offline?):\n" + "\n".join(errors))


def _safe_extract(zpath: Path, dest: Path) -> list[Path]:
    out = []
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir() or name.startswith("/") or ".." in name.split("/") or name.startswith("__MACOSX"):
                continue
            safe_name = re.sub(r'[\\/:*?"<>|]', "_", Path(name).name) or "sample"  # Windows-safe entry name
            target = dest / safe_name  # flatten: samples hold a single file
            with zf.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            out.append(target)
    return out


def _sample_image_path(fname: str, cache: Path) -> Path:
    """The openable file for a sample: the download itself, or the image inside the sample zip."""
    local = _fetch_sample(fname, cache)
    if not fname.lower().endswith(".zip"):
        return local
    folder = cache / (fname[:-4])
    files = sorted((f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")), key=lambda f: f.name) if folder.is_dir() else []
    if not files:
        folder.mkdir(parents=True, exist_ok=True)
        if not zipfile.is_zipfile(local):
            raise RuntimeError(f"{local.name} is not a zip archive (incomplete download? delete it and retry)")
        files = _safe_extract(local, folder)
    images = [f for f in files if _is_image_file(f)] or files
    if not images:
        raise RuntimeError(f"{fname} contains no image")
    return sorted(images, key=lambda f: f.name)[0]


@op("file.open_imagej_sample", "Open ImageJ Sample…", FILE, **_NOIMG, keywords=["samples", "imagej", "fiji", "demo", "download", "blobs", "t1 head", "clown"],
    params=[Param("name", "choice", "Blobs (25K)", choices=IJ_SAMPLE_NAMES),
            Param("use_cache", "bool", True, label="Use the local copy", help="Downloads are kept in ~/.lumen/samples", advanced=True)],
    description="Download one of ImageJ's classic sample images from imagej.net/images (cached in ~/.lumen/samples) and open it.")
def open_imagej_sample(ctx, name="Blobs (25K)", use_cache=True):
    if name not in IJ_SAMPLES:
        raise ValueError(f"Unknown sample {name!r}")
    cache = Path(_samples_dir())
    cache.mkdir(parents=True, exist_ok=True)
    errors = []
    for fname in IJ_SAMPLES[name]:
        if not use_cache:
            for stale in (cache / fname, cache / fname[:-4] if fname.lower().endswith(".zip") else None):
                if stale is None:
                    continue
                if stale.is_dir():
                    shutil.rmtree(stale, ignore_errors=True)
                elif stale.is_file():
                    stale.unlink()
        try:
            p = _sample_image_path(fname, cache)
            img = read_image(p)
            img.path = str(p)
            img.metadata.setdefault("source_url", SAMPLE_BASES[0] + fname)
            img.metadata["imagej_sample"] = name
            _add_recent(ctx.session, str(p))
            return OpResult(new_images=[img], message=f"{name}: {img.width}×{img.height}, {img.nz} slice(s), {img.nc} channel(s), {img.bit_depth}")
        except RuntimeError as e:
            errors.append(str(e))
    raise RuntimeError("\n".join(errors) if errors else f"Could not open sample {name}")


# ============================================================================ LUT files
def _apply_lut_to_active(session, lut_name: str, all_channels: bool = False) -> str:
    img = session.active
    if img is None:
        return ""
    patch = [{"lut": lut_name} if (all_channels or i == img.display.c) else {} for i in range(img.nc)]
    session.update_display(img.id, {"channels": patch})
    return f"; applied to {img.name}"


@op("file.import_lut", "LUT…", IMPORT, kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    keywords=["lookup table", "lut file", "fiji luts", "colormap", "import"],
    params=[Param("path", "string", "", help="ImageJ .lut file (768-byte binary, NIH Image 'ICOL', or a 256-line text table)"),
            Param("name", "string", "", help="LUT name (default: the file name)"),
            Param("apply", "bool", True, label="Apply to the current channel")],
    description="Load an ImageJ / Fiji .lut file as a new lookup table (available to every LUT command) and apply it to the current channel.")
def import_lut(ctx, path="", name="", apply=True):
    p = _existing(path)
    table = read_lut_file(p)
    lut_name = register_lut(str(name or "").strip() or p.stem, table)
    msg = f"LUT {lut_name!r} loaded ({len(table)} entries)"
    if apply:
        msg += _apply_lut_to_active(ctx.session, lut_name)
    return OpResult(message=msg, log=msg)


@op("file.import_luts_folder", "LUTs from Folder…", IMPORT, kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    keywords=["lookup tables", "fiji luts folder", "import all luts"],
    params=[Param("folder", "string", "", help="Folder with .lut files (e.g. Fiji.app/luts)"), Param("recursive", "bool", False, label="Include subfolders")],
    description="Register every .lut file of a folder (e.g. Fiji's luts/ directory) as lookup tables.")
def import_luts_folder(ctx, folder="", recursive=False):
    d = _expand(folder)
    if not d.is_dir():
        raise FileNotFoundError(f"{d} is not a folder")
    it = d.rglob("*.lut") if recursive else d.glob("*.lut")
    rows, ok = [], 0
    for f in sorted(it, key=lambda f: _natural_key(f.name)):
        try:
            nm = register_lut(f.stem, read_lut_file(f))
            rows.append([f.name, nm, "ok"])
            ok += 1
        except Exception as e:
            rows.append([f.name, "", f"error: {e}"])
    if not rows:
        raise ValueError(f"No .lut files in {d}")
    return OpResult(message=f"Loaded {ok} of {len(rows)} LUT file(s) from {d}", table={"name": "LUTs", "columns": ["File", "LUT", "Status"], "rows": rows})


@op("file.export_lut", "LUT…", EXPORT, kind="analysis", preview=False, roi_aware=False, scope_param=False, keywords=["lookup table", "lut file", "save lut"],
    params=[Param("path", "string", "", help="Output .lut file"), Param("lut", "string", "", help="LUT name (default: the current channel's)")],
    description="Save the current channel's lookup table as a 768-byte ImageJ .lut file.")
def export_lut(ctx, path="", lut=""):
    img = ctx.img
    ch = img.display.channels[img.display.c]
    name = str(lut or "").strip() or ch.lut
    table = get_lut(name)
    if not lut and ch.inverted:
        table = table[::-1]
    p = _expand(path)
    if p.is_dir() or str(path).endswith(("/", os.sep)):
        # LUT names may hold path separators ('Red/Green') or Windows-illegal characters (runtime-registered names come
        # from user input): same sanitiser as the ROI zip entries (io.py) and Stack ▸ Images names (import_export.py)
        safe = re.sub(r'[\\/:*?"<>|]', "_", name).strip() or "lut"
        p = p / (safe + ".lut")
    out = write_lut_file(table, p)
    return OpResult(message=f"Saved LUT {name!r} to {out}", log=f"LUT saved: {out}")


# ============================================================================ tables & text
def _num(v: Any) -> Any:
    if isinstance(v, (int, float)) or v is None:
        return v
    s = str(v).strip()
    if s == "":
        return ""
    if re.fullmatch(r"[-+]?\d+", s):
        try:
            return int(s)
        except ValueError:
            return s
    if s.lower() in ("nan", "inf", "-inf", "infinity", "-infinity"):
        return float(s)
    try:
        return float(s)
    except ValueError:
        return s


def _sniff_delim(sample: str, suffix: str) -> str:
    if suffix in (".tsv", ".tab"):
        return "\t"
    if suffix == ".csv":
        return ","
    if suffix == ".xls" and "\t" in sample:
        return "\t"  # ImageJ's saveAs("Results", "x.xls") writes tab-separated text for every non-.csv extension
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;| ").delimiter
    except csv.Error:
        if "\t" in sample:
            return "\t"
        if "," in sample:
            return ","
        # hand-aligned columns with unequal space runs ('  12   3.5') defeat the sniffer: treat space runs as one separator
        return " " if re.search(r"\S[ ]+\S", sample) else ","


def _read_workbook_rows(path: Path) -> list[list[Any]]:
    """Rows of the active sheet of an OOXML workbook (.xlsx / .xlsm, or a zip-content file saved as .xls)."""
    try:
        import openpyxl
    except ImportError as e:
        raise RuntimeError("Reading .xlsx needs: pip install openpyxl") from e
    # Pass a file object: given a path string openpyxl rejects by *extension* before reading a byte,
    # so an OOXML workbook saved under a .xls name would raise InvalidFileException.
    with open(path, "rb") as fh:
        wb = openpyxl.load_workbook(fh, read_only=True, data_only=True)
        try:
            return [[("" if c is None else c) for c in r] for r in wb.active.iter_rows(values_only=True)]
        finally:
            wb.close()


def _read_biff_rows(path: Path) -> list[list[Any]]:
    """Rows of the first sheet of a legacy binary Excel 97-2003 workbook (needs the optional xlrd package)."""
    try:
        import xlrd
    except ImportError as e:
        raise RuntimeError(f"{path.name} is a binary Excel 97-2003 (BIFF) workbook: install xlrd (pip install xlrd) or save it as "
                           ".xlsx / .csv (ImageJ's own .xls Results files are tab-separated text and load as-is)") from e
    sh = xlrd.open_workbook(str(path)).sheet_by_index(0)
    return [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(sh.nrows)]


def _read_text_rows(path: Path, ext: str) -> list[list[Any]]:
    """Rows of a delimited text table (CSV / TSV / whitespace-aligned / ImageJ's tab-separated '.xls')."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    delim = _sniff_delim("\n".join(lines[:20]), ext)
    if delim == " ":
        # whitespace-aligned columns: a run of spaces is one separator (like ImageJ's TextReader) while quoted
        # labels with spaces survive; leading spaces of right-aligned numbers would otherwise yield an empty first cell
        return [row for row in csv.reader([ln.strip() for ln in lines], delimiter=" ", skipinitialspace=True)]
    return [row for row in csv.reader(lines, delimiter=delim)]  # tab / comma / ; / | keep genuine empty cells


def read_table_file(path: Path) -> tuple[list[str], list[list[Any]]]:
    """(columns, rows) from CSV / TSV / text / XLSX / XLS; a leading ImageJ row-number column (' ') is dropped.

    ``.xls`` is decided by content, not by name: ImageJ's ``saveAs("Results", "Results.xls")`` writes tab-separated
    text, an OOXML workbook may be saved under the old name, and a genuine BIFF workbook needs xlrd.
    """
    ext = path.suffix.lower()
    if ext in (".xlsx", ".xlsm"):
        raw = _read_workbook_rows(path)
    elif ext == ".xls":
        with open(path, "rb") as fh:
            head = fh.read(8)
        if head.startswith(_ZIP_MAGIC):
            raw = _read_workbook_rows(path)
        elif head.startswith(_OLE_MAGIC):
            raw = _read_biff_rows(path)
        else:
            raw = _read_text_rows(path, ext)
    else:
        raw = _read_text_rows(path, ext)
    raw = [r for r in raw if any(str(c).strip() for c in r)]
    if not raw:
        raise ValueError(f"{path.name} is empty")
    header = [str(c).strip() for c in raw[0]]
    body = raw[1:]
    numeric_header = all(isinstance(_num(h), (int, float)) for h in header if h != "")
    if numeric_header and header:
        body = raw
        header = [f"C{i + 1}" for i in range(len(header))]
    width = max(len(header), max((len(r) for r in body), default=0))
    header = header + [f"C{i + 1}" for i in range(len(header), width)]
    rows = [[_num(c) for c in (list(r) + [""] * (width - len(r)))] for r in body]
    if header and header[0] in ("", " ") and all(isinstance(r[0], int) for r in rows if r):
        header, rows = header[1:], [r[1:] for r in rows]  # ImageJ row numbers
    header = [h or f"C{i + 1}" for i, h in enumerate(header)]
    return header, rows


@op("file.import_results", "Results…", IMPORT, kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    keywords=["results table", "measurements", "csv", "xlsx", "xls", "import"],
    params=[Param("path", "string", "", help=".csv / .tsv / .txt / .xls (ImageJ tab-separated text) / .xlsx"),
            Param("replace", "bool", True, label="Replace the current Results")],
    description="Load a CSV / TSV / text / Excel file (including ImageJ's tab-separated Results.xls) into the Results table.")
def import_results(ctx, path="", replace=True):
    p = _existing(path)
    cols, rows = read_table_file(p)
    recs = [{c: v for c, v in zip(cols, r)} for r in rows]
    return OpResult(results=recs, results_replace=bool(replace), message=f"{len(recs)} row(s), {len(cols)} column(s) from {p.name}")


@op("file.import_table", "Table…", IMPORT, kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    keywords=["csv", "tsv", "xlsx", "xls", "spreadsheet", "import"],
    params=[Param("path", "string", "", help=".csv / .tsv / .txt / .xls (ImageJ tab-separated text) / .xlsx"),
            Param("name", "string", "", help="Table name (default: the file name)")],
    description="Open a CSV / TSV / text / Excel file (including ImageJ's tab-separated Results.xls) as a named table window.")
def import_table(ctx, path="", name=""):
    p = _existing(path)
    cols, rows = read_table_file(p)
    tname = str(name or "").strip() or p.stem
    return OpResult(table={"name": tname, "columns": cols, "rows": rows}, message=f"Table {tname!r}: {len(rows)} row(s), {len(cols)} column(s)")


@op("file.import_text", "Text File…", IMPORT, kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    keywords=["text", "notes", "log", "import"],
    params=[Param("path", "string", ""), Param("to_log", "bool", True, label="Also show in the Log")],
    description="Open a text file as a table named after the file (one row per line) and show it in the Log.")
def import_text(ctx, path="", to_log=True):
    p = _existing(path)
    text = p.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.splitlines()
    rows = [[i + 1, ln] for i, ln in enumerate(lines)]
    res = OpResult(table={"name": p.name, "columns": ["Line", "Text"], "rows": rows}, message=f"{p.name}: {len(lines)} line(s)")
    if to_log:
        res.log = f"--- {p.name} ---\n" + (text if len(text) <= 20000 else text[:20000] + "\n… (truncated)")
    return res


def _read_xy(path: Path) -> np.ndarray:
    pts = []
    for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//")):
            continue
        parts = [t for t in re.split(r"[,\t; ]+", line) if t]
        try:
            vals = [float(t) for t in parts[:2]]
        except ValueError:
            continue  # header
        if len(vals) == 2:
            pts.append(vals)
    if not pts:
        raise ValueError(f"No X Y coordinate pairs in {path.name}")
    return np.asarray(pts, float)


@op("file.import_xy_coordinates", "XY Coordinates…", IMPORT, kind="analysis", preview=False, roi_aware=False, scope_param=False,
    keywords=["coordinates", "polygon", "points", "selection", "import"],
    params=[Param("path", "string", "", help="Text file with two columns (x y), comma/tab/space separated; a header line is allowed"),
            Param("type", "choice", "polygon", choices=["polygon", "polyline", "point"]),
            Param("invert_y", "bool", False, label="Invert Y (origin bottom-left)")],
    description="Create a polygon (default), segmented line or multi-point selection from a two-column XY text file (File ▸ Import ▸ XY Coordinates).")
def import_xy_coordinates(ctx, path="", type="polygon", invert_y=False):
    img = ctx.img
    pts = _read_xy(_existing(path))
    if invert_y:
        pts[:, 1] = img.height - pts[:, 1]
    if type == "polygon" and len(pts) < 3:
        raise ValueError("A polygon needs at least 3 points")
    if type == "polyline" and len(pts) < 2:
        raise ValueError("A line needs at least 2 points")
    roi: dict[str, Any] = {"type": type, "points": pts.round(4).tolist(), "name": Path(path).stem}
    if type == "polyline":
        roi["width"] = 1
    return OpResult(roi=roi, message=f"{len(pts)} point(s) → {type} selection")


@op("file.import_stack_from_list", "Stack From List…", IMPORT, **_NOIMG, keywords=["list of files", "stack", "import"],
    params=[Param("path", "string", "", help="Text file with one image path per line (relative paths are taken from the list's folder)"),
            Param("stack_as", "choice", "Z stack", label="Open as", choices=["Z stack", "T series", "Channels"]),
            Param("title", "string", "", help="Default: the list file's name")],
    description="Open the images listed in a text file (one path per line) as one stack (File ▸ Import ▸ Stack From List).")
def import_stack_from_list(ctx, path="", stack_as="Z stack", title=""):
    lp = _existing(path)
    paths = []
    for line in lp.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip().strip('"')
        if not line or line.startswith("#"):
            continue
        q = Path(line).expanduser()
        if not q.is_absolute():
            q = lp.parent / q
        paths.append(q)
    if not paths:
        raise ValueError(f"{lp.name} lists no files")
    planes, missing = [], []
    first: LumenImage | None = None
    for q in paths:
        if not q.exists():
            missing.append(str(q))
            continue
        im = read_image(q)
        if first is None:
            first = im
        if (im.height, im.width) != (first.height, first.width):
            raise ValueError(f"{q.name} is {im.width}×{im.height} but {first.name} is {first.width}×{first.height}")
        # every (t, z) of every listed file contributes a plane; channels are kept when they agree
        sub = im.data.reshape(-1, im.nc, im.height, im.width)
        if sub.shape[1] != first.nc:
            sub = sub[:, :1]
        planes.extend(list(sub))
    if missing and first is None:
        raise FileNotFoundError("None of the listed files exist: " + ", ".join(missing[:5]))
    assert first is not None
    nc = min(p.shape[0] for p in planes)
    dtype = np.result_type(*[p.dtype for p in planes])
    arr = np.stack([p[:nc].astype(dtype) for p in planes])  # (N, C, Y, X)
    if stack_as == "T series":
        data = arr[:, None]  # T Z C Y X
        axes = "TZCYX"
    elif stack_as == "Channels":
        data = arr.reshape(1, 1, -1, first.height, first.width)
        axes = "TZCYX"
    else:
        data = arr[None]  # T Z C Y X
        axes = "TZCYX"
    img = LumenImage(data, name=str(title or "").strip() or (lp.stem + ".tif"), axes=axes, rgb=first.rgb and stack_as != "Channels" and nc == 3 and dtype == np.uint8,
                     calibration=type(first.calibration)(**first.calibration.to_dict()))
    img.metadata["source"] = {"kind": "list", "path": str(lp), "files": [str(q) for q in paths]}
    msg = f"{len(planes)} plane(s) from {len(paths) - len(missing)} file(s) → {img.shape[1:]}"
    if missing:
        msg += f"; {len(missing)} missing"
    return OpResult(new_images=[img], message=msg, log=(msg + "\nMissing: " + "\n".join(missing)) if missing else None)


# ============================================================================ exports
@op("file.export_zip", "ZIP…", EXPORT, kind="analysis", preview=False, roi_aware=False, scope_param=False, keywords=["zip", "compressed tiff", "save as zip"],
    params=[Param("path", "string", "", help="Output .zip (the ImageJ TIFF inside is named after the image)")],
    description="Save the image as an ImageJ TIFF inside a .zip archive (File ▸ Save As ▸ ZIP).")
def export_zip(ctx, path=""):
    img = ctx.img
    p = _expand(path)
    if p.suffix.lower() != ".zip":
        p = p.with_name(p.name + ".zip")
    p.parent.mkdir(parents=True, exist_ok=True)
    inner = strip_image_ext(img.name) + ".tif"
    tmpdir = Path(tempfile.mkdtemp(prefix="lumen_zip_"))
    try:
        tif = write_image(img, tmpdir / inner, "tif")
        with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(tif, inner)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return OpResult(message=f"Saved {inner} into {p} ({p.stat().st_size // 1024} kB)", log=f"ZIP saved: {p}")


TEXT_SOURCES = ["Results", "Table", "Log", "Macro"]


def _table_text(columns: list[str], rows: list[list[Any]], delim: str, row_numbers: bool) -> str:
    import io as _io
    buf = _io.StringIO()
    w = csv.writer(buf, delimiter=delim, lineterminator="\n")
    w.writerow(([" "] if row_numbers else []) + list(columns))
    for i, r in enumerate(rows, 1):
        w.writerow(([i] if row_numbers else []) + [_fmt_cell(v) for v in r])
    return buf.getvalue()


def _fmt_cell(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return "NaN" if not np.isfinite(v) else (f"{v:.6f}".rstrip("0").rstrip(".") if v != int(v) else int(v))
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return v


@op("file.export_text", "Text…", EXPORT, kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    keywords=["save text", "results txt", "log", "macro", "export"],
    params=[Param("path", "string", "", help="Output .txt / .csv / .tsv"), Param("source", "choice", "Results", choices=TEXT_SOURCES),
            Param("table", "string", "", help="Name of the table when source = Table"), Param("row_numbers", "bool", True, label="Row numbers (tables)")],
    description="Save the Results table, a named table, the Log or the recorded macro as a text file (File ▸ Save As ▸ Text).")
def export_text(ctx, path="", source="Results", table="", row_numbers=True):
    s = ctx.session
    p = _expand(path)
    if p.suffix.lower() not in (".txt", ".csv", ".tsv", ".py", ".log"):
        p = p.with_name(p.name + (".csv" if source in ("Results", "Table") else ".txt"))
    delim = "," if p.suffix.lower() == ".csv" else "\t"
    if source == "Results":
        if not s.results.rows:
            raise ValueError("The Results table is empty")
        text = _table_text(list(s.results.columns), [[r.get(c) for c in s.results.columns] for r in s.results.rows], delim, row_numbers)
        n = len(s.results.rows)
    elif source == "Table":
        tb = s.tables.get(table or "")
        if tb is None:
            raise ValueError(f"No table named {table!r}; open tables: {list(s.tables) or 'none'}")
        text = _table_text(list(tb.get("columns", [])), [list(r) for r in tb.get("rows", [])], delim, row_numbers)
        n = len(tb.get("rows", []))
    elif source == "Log":
        text = "\n".join(str(e.get("msg", "")) for e in s.log) + "\n"
        n = len(s.log)
    else:
        text = "\n".join(s.macro) + "\n"
        n = len(s.macro)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return OpResult(message=f"Saved {source} ({n} line(s)) to {p}", log=f"Text saved: {p}")
