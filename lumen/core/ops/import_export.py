"""File ▸ Import / File ▸ Export commands: raw & text images, image sequences, URLs,
ImageJ .roi/.zip compatibility, sequence/video export and table/coordinate export."""
from __future__ import annotations

import csv
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .. import roi as roimod
from ..io import (RAW_TYPES, read_ij_rois, read_image, read_raw, read_raw_folder, read_sequence, read_text_image,
                  strip_image_ext, video_frames, write_ij_rois, write_image, write_video)
from ..registry import Param, OpResult, op

IMPORT = "File/Import"
EXPORT = "File/Export"
_COMMON = dict(kind="image", requires_image=False, scope_param=False, preview=False, roi_aware=False)


def _expand(path: str) -> Path:
    p = Path(str(path or "").strip()).expanduser()
    if not str(p):
        raise ValueError("Give a path")
    return p


def _ensure_dir(path: str) -> Path:
    p = _expand(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ============================================================================ imports
@op("file.import_raw", "Raw...", IMPORT, **_COMMON, description="Open a headerless raw binary file as an image or stack (File ▸ Import ▸ Raw).",
    keywords=["binary", "headerless", "import"],
    params=[Param("path", "string", "", help="Raw file to open"),
            Param("image_type", "choice", "8-bit", label="Image type", choices=list(RAW_TYPES)),
            Param("width", "int", 512, min=1, max=100000, unit="px"), Param("height", "int", 512, min=1, max=100000, unit="px"),
            Param("offset", "int", 0, min=0, label="Offset to first image", unit="bytes"),
            Param("n_images", "int", 1, min=0, label="Number of images", help="0 = as many as fit in the file"),
            Param("gap", "int", 0, min=0, label="Gap between images", unit="bytes"),
            Param("little_endian", "bool", False, label="Little-endian byte order"),
            Param("white_is_zero", "bool", False, label="White is zero"),
            Param("open_all", "bool", False, label="Open all files in folder", advanced=True, help="Open every file with the same extension in the folder as a stack")])
def import_raw(ctx, path="", image_type="8-bit", width=512, height=512, offset=0, n_images=1, gap=0, little_endian=False, white_is_zero=False, open_all=False):
    p = _expand(path)
    if not p.is_file():
        raise FileNotFoundError(str(p))
    if open_all:
        img = read_raw_folder(p.parent, p.suffix, width, height, image_type, offset, n_images, gap, little_endian, white_is_zero)
    else:
        img = read_raw(p, width, height, image_type, offset, n_images, gap, little_endian, white_is_zero)
    return OpResult(new_images=[img], message=f"{img.name}: {img.width}×{img.height}, {img.nz} image(s), {img.bit_depth}")


@op("file.import_sequence", "Image Sequence...", IMPORT, **_COMMON, description="Open all images in a folder as one stack (File ▸ Import ▸ Image Sequence).",
    keywords=["folder", "stack", "series", "import"],
    params=[Param("folder", "string", "", help="Folder with the image files"),
            Param("pattern", "string", "", label="File name pattern (regex)", help="Only files whose name matches this regular expression, e.g. _ch1\\.tif$"),
            Param("start", "int", 1, min=1, label="Starting image"), Param("count", "int", 0, min=0, label="Number of images", help="0 = all"),
            Param("step", "int", 1, min=1, label="Increment"), Param("scale", "float", 100.0, min=1, max=400, unit="%"),
            Param("sort", "choice", "name", choices=["name", "natural", "modification time"], label="Sort by"),
            Param("stack_as", "choice", "Z stack", label="Open as", choices=["Z stack", "T series", "Channels"]),
            Param("recursive", "bool", False, label="Include subfolders"),
            Param("virtual", "bool", False, label="Use virtual stack", help="Accepted for ImageJ macro compatibility; Lumen always loads into memory", advanced=True)])
def import_sequence(ctx, folder="", pattern="", start=1, count=0, step=1, scale=100.0, sort="name", stack_as="Z stack", recursive=False, virtual=False):
    folder_p = _expand(folder)
    axis = {"Z stack": "Z", "T series": "T", "Channels": "C"}[stack_as]
    img = read_sequence(folder_p, pattern=pattern, start=start, count=count, step=step, scale=scale, sort=sort, axis=axis, recursive=recursive)
    n = img.metadata.get("sequence_files", 0)
    return OpResult(new_images=[img], message=f"{n} file(s) → {img.name} {img.shape[1:]}")


@op("file.import_text_image", "Text Image...", IMPORT, **_COMMON, description="Open a tab/comma separated numeric matrix as a 32-bit image.",
    keywords=["csv", "matrix", "ascii"], params=[Param("path", "string", "", help=".txt / .csv / .tsv file")])
def import_text_image(ctx, path=""):
    img = read_text_image(_expand(path))
    return OpResult(new_images=[img], message=f"{img.width}×{img.height} 32-bit image")


def download_url(url: str, timeout: float = 60.0) -> tuple[Path, str]:
    """Download ``url`` into a fresh temporary folder; returns ``(local file, original file name)``.

    The local name keeps the compound suffix (.ome.tif, .nii.gz) so the readers dispatch on it.  Shared by
    File ▸ Import ▸ URL and the Bio-Formats remote importer."""
    import urllib.request
    from urllib.parse import unquote, urlparse
    url = str(url or "").strip()
    if not re.match(r"^https?://", url, re.I):
        raise ValueError("Give an http:// or https:// URL")
    fname = unquote(Path(urlparse(url).path).name) or "download"
    suffix = "".join(Path(fname).suffixes[-2:]) if fname.lower().endswith((".ome.tif", ".ome.tiff", ".nii.gz")) else Path(fname).suffix
    tmpdir = Path(tempfile.mkdtemp(prefix="lumen_url_"))
    stem = Path(fname).name[:-len(suffix)] if suffix else Path(fname).name
    target = tmpdir / (stem + (suffix or ".tif"))
    req = urllib.request.Request(url, headers={"User-Agent": "Lumen/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(target, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
    return target, fname


@op("file.open_url", "URL...", IMPORT, **_COMMON, description="Download an image from a URL into a temporary file and open it.",
    keywords=["download", "http", "internet"], params=[Param("url", "string", "", help="http(s):// address of an image file")])
def open_url(ctx, url=""):
    target, fname = download_url(url)
    img = read_image(target, name=fname)
    img.path = None
    img.metadata["url"] = str(url).strip()
    return OpResult(new_images=[img], message=f"Opened {fname} ({target.stat().st_size // 1024} kB)")


@op("file.import_ij_rois", "ImageJ ROIs (.roi/.zip)...", IMPORT, **_COMMON, description="Load ImageJ / Fiji ROI files (.roi or a ROI Manager .zip) into the ROI manager.",
    keywords=["roi manager", "fiji", "selection", "import"],
    params=[Param("path", "string", "", help="A .roi file or a .zip saved by the ImageJ ROI Manager"),
            Param("add_to_manager", "bool", True, label="Add to ROI manager"),
            Param("clear_first", "bool", False, label="Clear the ROI manager first")])
def import_ij_rois(ctx, path="", add_to_manager=True, clear_first=False):
    rois = read_ij_rois(_expand(path))
    if not rois:
        raise ValueError("No ROIs found in the file")
    s = ctx.session
    img = ctx.image or s.active
    if add_to_manager:
        if clear_first:
            s.roi_manager.clear()
        for r in rois:
            s.roi_manager.add(r, image_id=img.id if img else None, position=r.get("position"))
        s._changed("rois")
    res = OpResult(message=f"Imported {len(rois)} ROI(s)")
    if len(rois) == 1:
        res.roi = rois[0]
    return res


@op("file.import_rois_json", "ROIs from JSON...", IMPORT, **_COMMON, description="Load ROIs saved by 'Export ROIs as JSON'.", keywords=["roi", "json"],
    params=[Param("path", "string", ""), Param("clear_first", "bool", False, label="Clear the ROI manager first")])
def import_rois_json(ctx, path="", clear_first=False):
    data = json.loads(_expand(path).read_text(encoding="utf-8"))
    rois = data.get("rois", data) if isinstance(data, dict) else data
    if not isinstance(rois, list):
        raise ValueError("Expected a JSON list of ROIs")
    s = ctx.session
    img = ctx.image or s.active
    if clear_first:
        s.roi_manager.clear()
    for r in rois:
        r = {k: v for k, v in r.items() if k not in ("id", "image_id")}
        s.roi_manager.add(r, image_id=img.id if img else None, position=r.get("position"))
    s._changed("rois")
    return OpResult(message=f"Imported {len(rois)} ROI(s)")


# ============================================================================ exports
def _rois_to_export(ctx, which: str) -> list[dict[str, Any]]:
    s = ctx.session
    if which.startswith("Current"):
        if ctx.roi is None:
            raise ValueError("There is no current selection")
        return [dict(ctx.roi)]
    rois = list(s.roi_manager.to_list())
    if not rois:
        raise ValueError("The ROI manager is empty")
    return rois


@op("file.export_ij_rois", "ImageJ ROIs (.roi/.zip)...", EXPORT, **_COMMON, description="Save the ROI manager (or the current selection) as ImageJ-compatible .roi / .zip files.",
    keywords=["roi manager", "fiji", "selection", "export", "save"],
    params=[Param("path", "string", "", help="Output .zip (several ROIs) or .roi (single ROI)"),
            Param("which", "choice", "ROI Manager", choices=["ROI Manager", "Current selection"])])
def export_ij_rois(ctx, path="", which="ROI Manager"):
    rois = _rois_to_export(ctx, which)
    p = _expand(path)
    if p.suffix.lower() not in (".zip", ".roi"):
        p = p.with_suffix(".roi" if len(rois) == 1 else ".zip")
    p.parent.mkdir(parents=True, exist_ok=True)
    out = write_ij_rois(rois, p)
    return OpResult(message=f"Saved {len(rois)} ROI(s) to {out}", log=f"Saved ROIs: {out}")


@op("file.export_rois_json", "ROIs as JSON...", EXPORT, **_COMMON, description="Save the ROI manager as a JSON file (Lumen's native ROI format).",
    keywords=["roi", "json", "export"],
    params=[Param("path", "string", ""), Param("which", "choice", "ROI Manager", choices=["ROI Manager", "Current selection"])])
def export_rois_json(ctx, path="", which="ROI Manager"):
    rois = _rois_to_export(ctx, which)
    p = _expand(path)
    if p.suffix.lower() != ".json":
        p = p.with_suffix(".json")
    p.parent.mkdir(parents=True, exist_ok=True)
    img = ctx.image or ctx.session.active
    payload = {"format": "lumen-rois", "version": 1, "image": img.name if img else None,
               "image_size": {"width": img.width, "height": img.height} if img else None, "rois": rois}
    p.write_text(json.dumps(payload, indent=1, default=_json_default), encoding="utf-8")
    return OpResult(message=f"Saved {len(rois)} ROI(s) to {p}")


def _json_default(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


@op("file.export_sequence", "Image Sequence...", EXPORT, kind="image", scope_param=False, preview=False, roi_aware=False,
    description="Save every plane / slice / frame of the stack as a separate file.", keywords=["save planes", "split files", "export"],
    params=[Param("folder", "string", "", help="Output folder (created if missing)"),
            Param("format", "choice", "tif", choices=["tif", "png", "jpg", "bmp", "npy", "pgm"]),
            Param("name_pattern", "string", "{name}_z{z:04d}_t{t:04d}", label="Name pattern",
                  help="Python format with {name}, {z}, {t}, {c} (1-based) and {i} (running index); the extension is added"),
            Param("split", "choice", "Every slice (all channels)", label="One file per", choices=["Every plane", "Every slice (all channels)", "Every frame (whole Z stack)"]),
            Param("start_index", "int", 1, min=0, label="First index", advanced=True)])
def export_sequence(ctx, folder="", format="tif", name_pattern="{name}_z{z:04d}_t{t:04d}", split="Every slice (all channels)", start_index=1):
    img = ctx.img
    out_dir = _ensure_dir(folder)
    base = strip_image_ext(img.name) or "image"
    if not name_pattern.strip():
        name_pattern = "{name}_z{z:04d}_t{t:04d}"
    rows = []
    i = start_index
    jobs: list[tuple[dict[str, int], np.ndarray]] = []
    if split.startswith("Every plane"):
        pattern = name_pattern if "{c" in name_pattern else name_pattern + "_c{c:02d}"
        for t in range(img.nt):
            for z in range(img.nz):
                for c in range(img.nc):
                    jobs.append(({"t": t + 1, "z": z + 1, "c": c + 1}, img.data[t:t + 1, z:z + 1, c:c + 1]))
    elif split.startswith("Every frame"):
        pattern = name_pattern.replace("_z{z:04d}", "").replace("{z:04d}", "").replace("{z}", "")
        for t in range(img.nt):
            jobs.append(({"t": t + 1, "z": 1, "c": 1}, img.data[t:t + 1]))
    else:
        pattern = name_pattern
        for t in range(img.nt):
            for z in range(img.nz):
                jobs.append(({"t": t + 1, "z": z + 1, "c": 1}, img.data[t:t + 1, z:z + 1]))
    for pos, sub in jobs:
        try:
            fname = pattern.format(name=base, i=i, **pos)
        except (KeyError, IndexError, ValueError) as e:
            raise ValueError(f"Bad name pattern {name_pattern!r}: {e}") from e
        fname = re.sub(r"[\\/:*?\"<>|]", "_", fname)
        part = img.copy(name=fname, data=np.ascontiguousarray(sub))
        part.rgb = img.rgb and part.nc == 3
        part.display.t = part.display.z = 0
        part.display.c = min(img.display.c, part.nc - 1)
        out = write_image(part, out_dir / fname, format)
        rows.append([Path(out).name, pos["t"], pos["z"], pos["c"], os.path.getsize(out)])
        i += 1
    return OpResult(message=f"Wrote {len(rows)} file(s) to {out_dir}", log=f"Exported {len(rows)} files to {out_dir}",
                    table={"name": "Export", "columns": ["File", "T", "Z", "C", "Bytes"], "rows": rows})


@op("file.export_video", "Video (AVI/MP4/GIF)...", EXPORT, kind="image", scope_param=False, preview=False, roi_aware=False,
    description="Render the stack along Z or T with the current display settings and save it as a video or animated GIF.",
    keywords=["movie", "animation", "avi", "mp4", "gif"],
    params=[Param("path", "string", "", help="Output file (extension is set from the format)"),
            Param("format", "choice", "mp4", choices=["mp4", "avi", "gif", "mov", "mkv", "webm"]),
            Param("fps", "float", 10.0, min=0.1, max=240, label="Frames per second"),
            Param("axis", "choice", "auto", choices=["auto", "T", "Z"], help="auto = T if the image has frames, else Z"),
            Param("quality", "int", 8, min=1, max=10, advanced=True)])
def export_video(ctx, path="", format="mp4", fps=10.0, axis="auto", quality=8):
    img = ctx.img
    p = _expand(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    frames = video_frames(img, axis)
    if len(frames) < 1:
        raise ValueError("Nothing to export")
    out = write_video(frames, p, format, fps=fps, quality=quality)
    return OpResult(message=f"Saved {len(frames)} frame(s) at {fps:g} fps → {out}", log=f"Video: {out}")


@op("file.save_xy_coordinates", "XY Coordinates...", EXPORT, kind="analysis", requires_roi="any", preview=False, roi_aware=False, scope_param=False,
    description="Save the outline / line / point coordinates of the current selection as CSV (pixels and calibrated units).",
    keywords=["selection", "outline", "points", "csv", "export"],
    params=[Param("path", "string", "", help="Output .csv / .txt"), Param("invert_y", "bool", False, label="Invert Y (origin bottom-left)"),
            Param("oval_points", "int", 72, min=8, max=720, label="Points for ovals", advanced=True)])
def save_xy_coordinates(ctx, path="", invert_y=False, oval_points=72):
    roi = ctx.roi
    img = ctx.img
    cal = img.calibration
    t = roi["type"]
    if t in ("line", "polyline"):
        pts = np.asarray(roimod.line_points(roi), float)
    elif t == "point":
        pts = np.asarray(roi["points"], float)
    elif t == "oval":
        pts = roimod.polygon_points(roi, n_oval=oval_points)
    else:
        pts = roimod.polygon_points(roi)
    if pts.size == 0:
        raise ValueError("The selection has no coordinates")
    p = _expand(path)
    if p.suffix.lower() not in (".csv", ".txt", ".tsv"):
        p = p.with_suffix(".csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    ys = (img.height - pts[:, 1]) if invert_y else pts[:, 1]
    delim = "\t" if p.suffix.lower() in (".txt", ".tsv") else ","
    unit = cal.unit if cal.scaled else "pixel"
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=delim)
        w.writerow(["X", "Y", f"X ({unit})", f"Y ({unit})"])
        for (x, _), y in zip(pts, ys):
            w.writerow([f"{x:.4f}".rstrip("0").rstrip("."), f"{y:.4f}".rstrip("0").rstrip("."),
                        f"{x * cal.pixel_width:.6g}", f"{y * cal.pixel_height:.6g}"])
    return OpResult(message=f"Saved {len(pts)} coordinate(s) to {p}")


@op("file.export_results", "Results Table...", EXPORT, kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    description="Save the Results table (or another table) as CSV, TSV or Excel.", keywords=["results", "measurements", "csv", "excel", "xlsx", "save"],
    params=[Param("path", "string", "", help="Output file"), Param("format", "choice", "csv", choices=["csv", "tsv", "xlsx"]),
            Param("table", "string", "Results", help="'Results' or the name of another table (e.g. Summary, Batch)"),
            Param("row_numbers", "bool", True, label="Include row numbers")])
def export_results(ctx, path="", format="csv", table="Results", row_numbers=True):
    s = ctx.session
    if table in ("", "Results"):
        columns = list(s.results.columns)
        rows = [[r.get(c) for c in columns] for r in s.results.rows]
    else:
        tb = s.tables.get(table)
        if tb is None:
            raise ValueError(f"No table named {table!r}; open tables: {list(s.tables) or 'none'}")
        columns = list(tb.get("columns", []))
        rows = [list(r) for r in tb.get("rows", [])]
    if not rows:
        raise ValueError(f"The {table or 'Results'} table is empty")
    p = _expand(path)
    if p.suffix.lower() != "." + format:
        p = p.with_suffix("." + format)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = ([" "] if row_numbers else []) + columns
    body = [([i] if row_numbers else []) + [_cell(v) for v in r] for i, r in enumerate(rows, 1)]
    if format == "xlsx":
        try:
            import openpyxl
        except ImportError as e:
            raise RuntimeError("Writing .xlsx needs: pip install openpyxl") from e
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = (table or "Results")[:31]
        ws.append(header)
        for r in body:
            ws.append(r)
        wb.save(str(p))
    else:
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter="\t" if format == "tsv" else ",")
            w.writerow(header)
            w.writerows(body)
    return OpResult(message=f"Saved {len(rows)} row(s) to {p}", log=f"Table saved: {p}")


def _cell(v: Any) -> Any:
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        if not np.isfinite(v):
            return "NaN"
        return round(v, 6)
    if isinstance(v, (list, dict)):
        return json.dumps(v)
    return v
