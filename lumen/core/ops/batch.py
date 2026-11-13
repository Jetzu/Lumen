"""Plugins ▸ Batch: run a Lumen script (or the recorded macro) on every file in a folder
or on every open image.

Each file is opened with ``session.open_file``, the script runs with ``lumen.image`` bound
to it (``run_script``), the resulting image is saved, and the images opened for that file
are closed again.  Every file is wrapped in try/except so one bad file cannot abort the
batch (unless ``continue_on_error`` is off) and the session is restored afterwards
(previously active image re-selected, recorder unchanged).
"""
from __future__ import annotations

import csv
import re
import time
import traceback
from pathlib import Path
from typing import Any

from ..io import list_sequence_files, IMAGE_EXTS, _full_ext
from ..recorder_lines import resolve_undo
from ..registry import Param, OpResult, op
from ..scripting import run_script
from ..session import ResultsTable

CAT = "Plugins/Batch"
# scoped (?i:...) so upper-case extensions (A01.TIF, scan.CZI ...) match and the user can still prepend to the pre-filled value
DEFAULT_PATTERN = r"\.(?i:tiff?|png|jpe?g|nd2|czi|lif)$"
_COMMON = dict(kind="image", requires_image=False, scope_param=False, preview=False, roi_aware=False)
_FOLDER_PARAMS = [
    Param("input_folder", "string", "", label="Input folder"),
    Param("pattern", "string", DEFAULT_PATTERN, label="File pattern (regex)", help="Only file names matching this regular expression are processed"),
    Param("recursive", "bool", False, label="Include subfolders"),
    Param("output_folder", "string", "", label="Output folder", help="Empty = <input folder>/lumen_output"),
    Param("output_format", "choice", "tif", label="Output format", choices=["tif", "ome.tif", "png", "jpg", "npy", "none"]),
]
_TAIL_PARAMS = [
    Param("save_results", "bool", True, label="Save Results table", help="Append measurements the script produced to Results.csv in the output folder (with a File column)"),
    Param("close_after", "bool", True, label="Close images after each file"),
    Param("continue_on_error", "bool", True, label="Continue on error"),
    Param("max_files", "int", 0, min=0, label="Max files", help="0 = all"),
]
# lines that name files or images of the recording session (open / save / close / select / display / undo) or run a
# batch themselves are not replayed on every file of a folder; the preview endpoint lists them as "skipped".  Undo /
# redo lines are first resolved by ``recorder_lines.resolve_undo`` (the undone commands go with them, E7-5); only an
# undo / redo that cancelled nothing is left for this pattern to skip
_SKIP_MACRO = re.compile(r'^\s*(lumen\.(open|save|close|select|undo|redo|display)\(|lumen\.run\("(file\.sample|file\.new|file\.open_|file\.import_|file\.close|batch\.|rois\.restore_cleared|results\.restore_cleared)|#)')
_SKIP_ALWAYS = re.compile(r'^\s*lumen\.run\("batch\.')


def _short_error(err: str) -> str:
    lines = [ln for ln in err.strip().splitlines() if ln.strip()]
    return lines[-1][:300] if lines else "error"


def _compile_pattern(pattern: str) -> str:
    """Validate the user's file pattern (a regex; a glob such as ``*.tif`` is translated) – shared with lumen.core.io."""
    from ..io import compile_file_pattern
    return compile_file_pattern(pattern)


def _write_results_csv(path: Path, file_name: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["File"] + columns)
        for r in rows:
            w.writerow([file_name] + [r.get(c, "") for c in columns])


def _run_batch(ctx, *, input_folder: str, pattern: str, recursive: bool, output_folder: str, output_format: str, script: str,
               save_results: bool, close_after: bool, continue_on_error: bool, max_files: int, label: str) -> OpResult:
    s = ctx.session
    in_dir = Path(str(input_folder or "")).expanduser()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")
    files = list_sequence_files(in_dir, _compile_pattern(pattern or ""), sort="natural", recursive=recursive, exts=IMAGE_EXTS)
    out_dir = Path(str(output_folder or "")).expanduser() if str(output_folder or "").strip() else in_dir / "lumen_output"
    files = [f for f in files if out_dir not in f.parents]
    if max_files and max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files in {in_dir} match /{pattern}/")
    if output_format != "none" or save_results:
        out_dir.mkdir(parents=True, exist_ok=True)
    results_csv = out_dir / "Results.csv"
    script = script or ""
    prev_active = s.active_id
    macro_before = list(s.macro)
    rows: list[list[Any]] = []
    csv_columns: list[str] = []  # stable column order for Results.csv across files
    n_ok = n_err = 0
    t_batch = time.time()
    try:
        for i, f in enumerate(files):
            ctx.check_cancel()
            ctx.progress(i / len(files), f"{f.name} ({i + 1}/{len(files)})")
            t0 = time.time()
            before_ids = set(s.images.keys())
            # the script measures into a fresh per-file table, so rows survive a `Clear Results` /
            # results_replace inside the script; they are merged back into the session table below
            session_results, s.results = s.results, ResultsTable()
            status, msg, out_path = "ok", "", ""
            try:
                img = s.open_file(str(f))
                if script.strip():
                    res = run_script(s, script)
                    if res["error"]:
                        raise RuntimeError(_short_error(res["error"]))
                    if res["stdout"].strip():
                        msg = res["stdout"].strip().splitlines()[-1][:200]
                # the image to save: whatever the script left active, provided it was created for this file
                # (the opened image itself or e.g. a projection / duplicate); if the script left a pre-existing
                # image active, fall back to the opened image, and save nothing if the script closed it
                act = s.active
                target = act if (act is not None and act.id not in before_ids) else s.images.get(img.id)
                if output_format != "none" and target is not None:
                    rel = f.relative_to(in_dir) if recursive else Path(f.name)
                    ext = _full_ext(f)  # ".tif", ".ome.tif", ... (never "": list_sequence_files filters by IMAGE_EXTS)
                    base = f.name[:len(f.name) - len(ext)]  # keep dotted tails: "exp1.ch0", "cells_2024.03.12"
                    dest = out_dir / rel.parent / f"{base}.{output_format}"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    out_path = s.save_image(target.id, str(dest), output_format)
                if save_results and s.results.rows:
                    csv_columns += [c for c in s.results.columns if c not in csv_columns]
                    _write_results_csv(results_csv, f.name, list(s.results.rows), csv_columns)
                n_ok += 1
            except Exception as e:  # noqa: BLE001 – one bad file must not kill the batch
                status = "error"
                msg = f"{type(e).__name__}: {e}"[:300]
                n_err += 1
                s.add_log(f"[batch] {f.name}: {msg}\n{traceback.format_exc()[-600:]}", level="error")
            finally:
                file_results, s.results = s.results, session_results  # restore the session table, keep this file's rows
                if file_results.rows:
                    s.results.append(file_results.rows)
                if close_after:
                    for iid in list(s.images.keys()):
                        if iid not in before_ids:
                            s.close_image(iid)
            rows.append([f.name, status, round(time.time() - t0, 3), Path(out_path).name if out_path else "", msg])
            if status == "error" and not continue_on_error:
                break
    finally:
        # restore the session even if a BaseException escapes a script: previous active image,
        # recorder without the per-file open/save lines
        if prev_active and prev_active in s.images:
            s.select(prev_active)
        s.macro = macro_before
        s._changed("images")
        s._changed("results")
    ctx.check_cancel()  # cancelled during the last file (the script's command stopped): the session is restored, now stop
    dt = time.time() - t_batch
    summary = f"{label}: {n_ok} ok, {n_err} failed of {len(files)} file(s) in {dt:.1f} s" + (f" → {out_dir}" if output_format != "none" else "")
    if n_err and not continue_on_error and n_ok + n_err < len(files):
        summary += " (stopped at first error)"
    return OpResult(message=summary, log=summary,
                    table={"name": "Batch", "columns": ["File", "Status", "Time (s)", "Output", "Message"], "rows": rows,
                           "summary": {"ok": n_ok, "error": n_err, "total": len(files), "output_folder": str(out_dir),
                                       "results_csv": str(results_csv) if save_results and results_csv.exists() else None}})


@op("batch.process_folder", "Process Folder...", CAT, **_COMMON,
    description="Open every matching file in a folder, run a Lumen Python script on it (lumen.image is the open file) and save the result.",
    keywords=["batch", "folder", "script", "macro", "automate"],
    params=_FOLDER_PARAMS + [
        Param("script", "text", 'lumen.run("filters.gaussian", sigma=2)', help="Python using the lumen API, e.g. lumen.run(\"threshold.auto\", method=\"Otsu\")"),
    ] + _TAIL_PARAMS)
def process_folder(ctx, input_folder="", pattern=DEFAULT_PATTERN, recursive=False, output_folder="", output_format="tif",
                   script="", save_results=True, close_after=True, continue_on_error=True, max_files=0):
    return _run_batch(ctx, input_folder=input_folder, pattern=pattern, recursive=recursive, output_folder=output_folder, output_format=output_format,
                      script=script, save_results=save_results, close_after=close_after, continue_on_error=continue_on_error, max_files=max_files, label="Batch")


def macro_as_script(macro: list[str]) -> str:
    """The recorded macro without the lines that open/save/close/select images or run batch commands, and without the
    steps the user undid (see :func:`preview_lines`)."""
    return preview_lines(macro)["script"]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def preview_summary(n_lines: int, n_skipped: int, n_undone: int) -> str:
    """The sentence the Apply Recorded Macro sheet shows under the command list."""
    parts = []
    if n_skipped:
        parts.append(f"{_plural(n_skipped, 'open / save / select / display / undo line')} left out")
    if n_undone:
        parts.append(f"{_plural(n_undone, 'undone step')} removed")
    tail = f" ({'; '.join(parts)})" if parts else ""
    return f"{_plural(n_lines, 'recorded line')} from this session's recorder will run on every file{tail}. Edit the list above before Apply."


def preview_lines(macro: list[str]) -> dict[str, Any]:
    """What Apply Recorded Macro to Folder runs from a recorder: ``lines`` (kept, in order), ``skipped`` (the
    open/save/close/select/display/undo and batch lines it drops), ``undone`` (the commands a recorded ``lumen.undo``
    cancelled – dropped together with the undo / redo lines, so a batch never applies a step the user took back),
    ``script`` (the kept lines joined) and ``summary`` (the sentence for the sheet)."""
    _resolved, dropped = resolve_undo(macro)
    removed = {d["index"]: d["kind"] for d in dropped}
    kept, skipped, undone = [], [], []
    for i, ln in enumerate(macro):
        if not ln.strip():
            continue
        kind = removed.get(i)
        if kind == "undone":
            undone.append(ln)
        elif kind is not None or _SKIP_MACRO.match(ln):
            skipped.append(ln)
        else:
            kept.append(ln)
    return {"lines": kept, "skipped": skipped, "undone": undone, "script": "\n".join(kept),
            "summary": preview_summary(len(kept), len(skipped), len(undone))}


@op("batch.apply_recorded_macro", "Apply Recorded Macro to Folder...", CAT, **_COMMON,
    description="Run the recorded commands (shown and editable in 'lines'; default: this session's recorder minus the open/save/close/select/display lines and minus the steps you undid) on every matching file in a folder.",
    keywords=["batch", "recorder", "macro", "replay"],
    params=_FOLDER_PARAMS + [Param("lines", "text", "", label="Commands to run", help="One Python line per command (lumen.run(...)); empty = the recorder's lines minus the open / save / close / select / display / undo ones and minus the steps an undo cancelled. GET /api/recorder/preview shows exactly those.")] + _TAIL_PARAMS)
def apply_recorded_macro(ctx, input_folder="", pattern=DEFAULT_PATTERN, recursive=False, output_folder="", output_format="tif",
                         lines="", save_results=True, close_after=True, continue_on_error=True, max_files=0):
    undone: list[str] = []
    if str(lines or "").strip():
        # explicit lines (edited in the sheet) run as given – only nested batch calls are dropped
        script = "\n".join(ln for ln in str(lines).splitlines() if ln.strip() and not _SKIP_ALWAYS.match(ln))
    else:
        pv = preview_lines(ctx.session.macro)
        script, undone = pv["script"], pv["undone"]
    if not script.strip():
        raise ValueError("The recorder is empty – apply some commands first" + (f" ({_plural(len(undone), 'recorded step')} undone and left out)" if undone else ""))
    res = _run_batch(ctx, input_folder=input_folder, pattern=pattern, recursive=recursive, output_folder=output_folder, output_format=output_format,
                     script=script, save_results=save_results, close_after=close_after, continue_on_error=continue_on_error, max_files=max_files, label="Macro batch")
    if undone:
        note = f"; {_plural(len(undone), 'undone step')} removed from the recording before the run"
        res.message = (res.message or "") + note
        res.log = (res.log or "") + note + ": " + " | ".join(ln.strip() for ln in undone)
        if res.table is not None:
            res.table.setdefault("summary", {})["undone"] = list(undone)
    return res


@op("batch.run_script_on_open_images", "Run Script on All Open Images...", CAT, **_COMMON,
    description="Run a Lumen Python script once per open image (each becomes lumen.image in turn).",
    keywords=["batch", "open images", "script", "all images"],
    params=[Param("script", "text", 'lumen.run("filters.gaussian", sigma=2)'),
            Param("continue_on_error", "bool", True, label="Continue on error")])
def run_script_on_open_images(ctx, script="", continue_on_error=True):
    s = ctx.session
    if not s.images:
        raise ValueError("No images are open")
    if not str(script or "").strip():
        raise ValueError("The script is empty")
    prev_active = s.active_id
    macro_before = list(s.macro)
    rows: list[list[Any]] = []
    n_ok = n_err = 0
    try:
        items = list(s.images.items())
        for i, (iid, img) in enumerate(items):
            ctx.check_cancel()
            ctx.progress(i / len(items), f"{img.name} ({i + 1}/{len(items)})")
            if iid not in s.images:
                continue  # closed by an earlier run
            t0 = time.time()
            name = img.name
            status, msg = "ok", ""
            try:
                s.select(iid)
                res = run_script(s, script)
                if res["error"]:
                    raise RuntimeError(_short_error(res["error"]))
                if res["stdout"].strip():
                    msg = res["stdout"].strip().splitlines()[-1][:200]
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                status, msg = "error", f"{type(e).__name__}: {e}"[:300]
                n_err += 1
                s.add_log(f"[batch] {name}: {msg}", level="error")
            rows.append([name, status, round(time.time() - t0, 3), "", msg])
            if status == "error" and not continue_on_error:
                break
    finally:
        # restore the session even if a BaseException escapes a script
        if prev_active and prev_active in s.images:
            s.select(prev_active)
        s.macro = macro_before
        s._changed("images")
    ctx.check_cancel()
    summary = f"Script ran on {n_ok} image(s), {n_err} failed"
    return OpResult(message=summary, log=summary,
                    table={"name": "Batch", "columns": ["File", "Status", "Time (s)", "Output", "Message"], "rows": rows,
                           "summary": {"ok": n_ok, "error": n_err, "total": len(rows)}})


@op("batch.measure", "Batch Measure...", "Process/Batch", **_COMMON,
    description="Open every matching image in a folder, optionally threshold it, run Measure with the current Set Measurements and append the rows (with a File column) to the Results table (ImageJ's Process ▸ Batch ▸ Measure).",
    keywords=["batch", "measure", "folder"],
    params=[Param("input_folder", "string", "", label="Input folder"),
            Param("pattern", "string", DEFAULT_PATTERN, label="File pattern (regex)"), Param("recursive", "bool", False, label="Include subfolders"),
            Param("threshold", "choice", "None", choices=["None", "Default", "Huang", "Intermodes", "IsoData", "Li", "MaxEntropy", "Mean", "MinError", "Minimum", "Moments", "Otsu", "Percentile", "RenyiEntropy", "Shanbhag", "Triangle", "Yen"],
                  help="Auto-threshold method applied before measuring (measurements are then limited to the thresholded pixels)"),
            Param("dark_background", "bool", True), Param("all_planes", "bool", False, label="Measure every slice / frame"),
            Param("max_files", "int", 0, min=0, label="Max files", help="0 = all"), Param("continue_on_error", "bool", True)])
def batch_measure(ctx, input_folder="", pattern=DEFAULT_PATTERN, recursive=False, threshold="None", dark_background=True, all_planes=False, max_files=0, continue_on_error=True):
    from ..measure import measure_roi
    from ..threshold import auto_threshold
    s = ctx.session
    in_dir = Path(str(input_folder or "")).expanduser()
    if not in_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")
    files = list_sequence_files(in_dir, _compile_pattern(pattern or ""), sort="natural", recursive=recursive, exts=IMAGE_EXTS)
    if max_files and max_files > 0:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"No files in {in_dir} match /{pattern}/")
    prev_active = s.active_id
    macro_before = list(s.macro)
    table_rows: list[list[Any]] = []
    results: list[dict[str, Any]] = []
    n_ok = n_err = 0
    t_batch = time.time()
    try:
        for i, f in enumerate(files):
            ctx.check_cancel()
            ctx.progress(i / len(files), f"{f.name} ({i + 1}/{len(files)})")
            t0 = time.time()
            status, msg = "ok", ""
            img = None
            try:
                img = s.open_file(str(f))
                d = img.display
                planes = [(t, z, d.c) for t in range(img.nt) for z in range(img.nz)] if all_planes else [(d.t, d.z, d.c)]
                meas = list(dict.fromkeys(s.measurements + (["stack_position"] if all_planes else [])))
                for (t, z, c) in planes:
                    thr = None
                    if threshold != "None":
                        if img.rgb:
                            raise ValueError("Thresholding needs a grayscale image")
                        thr = auto_threshold(img.data[t, z, c], threshold, None, dark_background)
                    for row in measure_roi(img, None, meas, t=t, z=z, c=c, limit_to_threshold=thr, label=f.name, options=s.measure_options):
                        results.append({"File": f.name, **row})
                msg = f"{len(planes)} plane(s)"
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                status, msg = "error", f"{type(e).__name__}: {e}"[:300]
                n_err += 1
                s.add_log(f"[batch measure] {f.name}: {msg}", level="error")
            finally:
                if img is not None and img.id in s.images:
                    s.close_image(img.id)
            table_rows.append([f.name, status, round(time.time() - t0, 3), "", msg])
            if status == "error" and not continue_on_error:
                break
    finally:
        if prev_active and prev_active in s.images:
            s.select(prev_active)
        s.macro = macro_before
        s._changed("images")
    ctx.check_cancel()
    summary = f"Batch measure: {n_ok} ok, {n_err} failed of {len(files)} file(s) in {time.time() - t_batch:.1f} s"
    return OpResult(results=results, message=summary, log=summary,
                    table={"name": "Batch", "columns": ["File", "Status", "Time (s)", "Output", "Message"], "rows": table_rows,
                           "summary": {"ok": n_ok, "error": n_err, "total": len(files)}})
