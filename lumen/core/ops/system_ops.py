"""System / housekeeping commands that close the last ImageJ parity gaps.

* Plugins ▸ Utilities: ``help.properties`` (ImageJ Properties), ``help.threads``, ``help.benchmark``,
  ``help.reset``, ``help.monitor_events``, ``help.search_commands`` (Search…) and ``help.reload_plugins``.
* Edit ▸ Options: ``edit.options_memory`` (Memory & Threads…), ``edit.options_proxy`` (Proxy Settings…),
  ``help.compiler_info`` (Compiler…) and ``edit.options_dicom`` (DICOM…).
* Process ▸ Batch ▸ Virtual Stack…: ``batch.virtual_stack`` – a folder processed plane by plane.
* Help ▸ Check for Updates…: ``help.check_updates`` (never installs anything).
* Help ▸ Update Lumen…: ``help.update`` – ``git pull --ff-only`` in a checkout / ``pip install --upgrade`` for a package
  install (list argv, no shell; ``dry_run`` / ``check_only``); never restarts the server.
* Plugins ▸ Utilities ▸ Search…: ``help.search_files`` – greps the plugins / macros / scripts folders, the open images'
  names and metadata, the Log and the recorded script (ImageJ's Search…).

Settings written here (all plain JSON values in ``session.settings``):

``undo_budget_mb``            applied to ``History.default_budget_bytes`` and every open history
``max_threads``               BLAS / OpenMP pool size via ``threadpoolctl`` when it is installed (0 = all cores)
``proxy`` / ``no_proxy``      mirrored into ``HTTP_PROXY`` / ``HTTPS_PROXY`` / ``NO_PROXY`` (sample + URL downloads)
``dicom_open_as_32bit``, ``dicom_ignore_rescale``, ``dicom_calibrate_images``, ``dicom_window_level_on_open``
                              stored for the DICOM reader (``file.open`` consults them once wired)
"""
from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy import ndimage as ndi

import lumen
from ..registry import REGISTRY, OpResult, Param, op
from ..undo import History

UTIL = "Plugins/Utilities"
OPT = "Edit/Options"
HELP = "Help"
_NOIMG = dict(kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False)

PYPI_NAME = "lumen-imaging"
DEFAULT_UNDO_BUDGET_MB = 1500
_MB = 1_000_000


def _mb(n: int | float | None) -> str:
    return "n/a" if n is None else f"{n / _MB:,.1f} MB"


def _plugin_dirs() -> list[Path]:
    from ..io import lumen_home
    dirs = [lumen_home() / "plugins"]
    if os.environ.get("LUMEN_PLUGINS"):
        dirs += [Path(p) for p in os.environ["LUMEN_PLUGINS"].split(os.pathsep) if p]
    return dirs


def _module_version(module: str, dist: str | None = None) -> str:
    try:
        v = getattr(importlib.import_module(module), "__version__", None)
        if v:
            return str(v)
    except Exception:
        pass
    try:
        return importlib.metadata.version(dist or module)
    except Exception:
        return "not installed"


def process_rss_bytes() -> int | None:
    """Resident set size of this process: psutil if present, else ``resource`` (peak RSS; None on Windows)."""
    try:
        import psutil  # type: ignore[import-not-found]
        return int(psutil.Process().memory_info().rss)
    except Exception:
        pass
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss) if sys.platform == "darwin" else int(rss) * 1024  # macOS reports bytes, Linux kB
    except Exception:
        return None


def _session_memory(s) -> tuple[int, int]:
    images = int(sum(im.nbytes for im in s.images.values()))
    undo = int(sum(sum(x.nbytes for x in h.undo) + sum(x.nbytes for x in h.redo) for h in s.histories.values()))
    return images, undo


# ============================================================================ ImageJ Properties
@op("help.properties", "Lumen Properties", UTIL, **_NOIMG, keywords=["imagej properties", "system", "versions", "memory", "environment", "about"],
    description="A table of system properties: Lumen / Python / library versions, platform, CPU count, process memory, virtual environment, "
                "plugin folders, open images and the memory they hold (ImageJ's Plugins ▸ Utilities ▸ ImageJ Properties).")
def properties(ctx):
    s = ctx.session
    img_bytes, undo_bytes = _session_memory(s)
    rss = process_rss_bytes()
    dirs = _plugin_dirs()
    rows: list[list[Any]] = [
        ["Lumen version", lumen.__version__],
        ["Python", f"{platform.python_version()} ({platform.python_implementation()})"],
        ["Executable", sys.executable],
        ["Environment (sys.prefix)", sys.prefix],
        ["Platform", platform.platform()],
        ["Machine", platform.machine() or "unknown"],
        ["CPU count", os.cpu_count() or "unknown"],
        ["numpy", _module_version("numpy")],
        ["scipy", _module_version("scipy")],
        ["scikit-image", _module_version("skimage", "scikit-image")],
        ["scikit-learn", _module_version("sklearn", "scikit-learn")],
        ["tifffile", _module_version("tifffile")],
        ["threadpoolctl", _module_version("threadpoolctl")],
        ["Process RSS", _mb(rss) + ("" if rss is None or _has_psutil() else " (peak)")],
        ["Python threads", threading.active_count()],
        ["Max threads (BLAS/OpenMP)", s.settings.get("max_threads") or "all cores"],
        ["Undo budget", _mb(History.default_budget_bytes)],
        ["Proxy", s.settings.get("proxy") or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or "none"],
        ["Plugin folders", os.pathsep.join(str(d) for d in dirs)],
        ["Plugin files", sum(len(list(d.glob("*.py"))) for d in dirs if d.is_dir())],
        ["Commands", len(REGISTRY.ops)],
        ["Open images", len(s.images)],
        ["Image memory", _mb(img_bytes)],
        ["Undo memory", _mb(undo_bytes)],
        ["Results rows", len(s.results.rows)],
        ["ROI manager", len(s.roi_manager.rois)],
        ["Log entries", len(s.log)],
        ["Session version", s.version],
        ["Working directory", os.getcwd()],
    ]
    return OpResult(table={"name": "Properties", "columns": ["Property", "Value"], "rows": rows},
                    message=f"Lumen {lumen.__version__} on Python {platform.python_version()} / {platform.system()} – "
                            f"{len(s.images)} image(s), {_mb(img_bytes + undo_bytes)} in images + undo, process {_mb(rss)}")


def _has_psutil() -> bool:
    try:
        import psutil  # type: ignore[import-not-found]  # noqa: F401
        return True
    except Exception:
        return False


# ============================================================================ Threads
@op("help.threads", "Threads", UTIL, **_NOIMG, keywords=["threads", "running", "workers", "stack"],
    description="List the live Python threads of the Lumen server with their ids, daemon flag and the code line each is executing "
                "(ImageJ's Plugins ▸ Utilities ▸ Threads).")
def threads(ctx):
    frames = {}
    try:
        frames = sys._current_frames()
    except Exception:  # pragma: no cover
        pass
    main_id = threading.main_thread().ident
    rows: list[list[Any]] = []
    for t in sorted(threading.enumerate(), key=lambda t: (t.ident != main_id, t.name)):
        where = ""
        f = frames.get(t.ident) if t.ident is not None else None
        if f is not None:
            where = f"{Path(f.f_code.co_filename).name}:{f.f_lineno} {f.f_code.co_name}"
        rows.append([t.name, t.ident, getattr(t, "native_id", None), t.daemon, t.is_alive(), "main" if t.ident == main_id else "", where])
    return OpResult(table={"name": "Threads", "columns": ["Name", "Id", "Native id", "Daemon", "Alive", "Role", "Where"], "rows": rows},
                    message=f"{len(rows)} live thread(s), {sum(1 for r in rows if r[3])} daemon")


# ============================================================================ Benchmark
def _benchmark_image(size: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.normal(90.0, 15.0, (size, size)).astype(np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    n = max(4, size // 32)
    for cy, cx, r in zip(rng.integers(0, size, n), rng.integers(0, size, n), rng.integers(size // 40 + 2, size // 12 + 3, n)):
        img[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] += 110.0
    return np.clip(img, 0, 255).astype(np.uint8)


def benchmark_steps(img8: np.ndarray) -> list[tuple[str, Any]]:
    """(name, callable) pairs timed by ``help.benchmark``; each callable returns a short result note."""
    from skimage import filters, measure
    from ..threshold import auto_threshold
    f32 = img8.astype(np.float32)
    k5 = np.ones((5, 5), np.float32) / 25.0

    def threshold_particles():
        lo, hi = auto_threshold(img8, "Otsu", None, True)
        mask = (img8 >= lo) & (img8 <= hi)
        lbl, n = ndi.label(mask)
        props = measure.regionprops(lbl)
        return f"{n} particles, mean area {np.mean([p.area for p in props]) if props else 0:.0f} px"

    def fft_roundtrip():
        spec = np.fft.fft2(f32)
        back = np.fft.ifft2(spec).real
        return f"max error {np.abs(back - f32).max():.2g}"

    return [
        ("Invert", lambda: f"{(255 - img8).mean():.1f} mean"),
        ("Flip vertical", lambda: str(img8[::-1].copy().shape)),
        ("Rotate 90°", lambda: str(np.rot90(img8).copy().shape)),
        ("Scale 50 % (bilinear)", lambda: str(ndi.zoom(img8, 0.5, order=1).shape)),
        ("Gaussian blur σ=2", lambda: f"{ndi.gaussian_filter(f32, 2.0).mean():.1f} mean"),
        ("Median 3×3", lambda: f"{ndi.median_filter(img8, size=3).mean():.1f} mean"),
        ("Unsharp mask (σ=1, 0.6)", lambda: f"{np.clip(f32 + 0.6 * (f32 - ndi.gaussian_filter(f32, 1.0)), 0, 255).mean():.1f} mean"),
        ("Find edges (Sobel)", lambda: f"{filters.sobel(f32).max():.1f} max"),
        ("Convolve 5×5", lambda: f"{ndi.convolve(f32, k5).mean():.1f} mean"),
        ("FFT + inverse FFT", fft_roundtrip),
        ("Histogram (256 bins)", lambda: f"{int(np.bincount(img8.ravel(), minlength=256).argmax())} mode"),
        ("32-bit math (add, multiply, sqrt)", lambda: f"{np.sqrt((f32 + 10.0) * 1.5).mean():.1f} mean"),
        ("Threshold (Otsu) + Analyze Particles", threshold_particles),
    ]


@op("help.benchmark", "Benchmark", UTIL, **_NOIMG, keywords=["benchmark", "speed", "performance", "timing"],
    params=[Param("size", "int", 512, min=64, max=4096, label="Image size (px)", help="Square synthetic 8-bit test image"),
            Param("repeats", "int", 1, min=1, max=20, help="Each operation runs this many times; the best time is reported")],
    description="Time a fixed set of operations (invert, flip, rotate, scale, Gaussian, median, unsharp mask, edges, convolve, FFT, histogram, "
                "32-bit math, Otsu threshold + particles) on a synthetic 512×512 image, like ImageJ's Plugins ▸ Utilities ▸ Benchmark.")
def benchmark(ctx, size=512, repeats=1):
    img8 = _benchmark_image(int(size))
    rows: list[list[Any]] = []
    total = 0.0
    for name, fn in benchmark_steps(img8):
        best = None
        note = ""
        for _ in range(int(repeats)):
            t0 = time.perf_counter()
            note = str(fn())
            dt = (time.perf_counter() - t0) * 1000.0
            best = dt if best is None else min(best, dt)
        total += best or 0.0
        rows.append([name, round(best or 0.0, 2), note])
    rows.append(["Total", round(total, 2), f"{len(rows)} operations on {size}×{size} px, best of {repeats}"])
    msg = f"Benchmark: {total:.0f} ms for {len(rows) - 1} operations on a {size}×{size} image (best of {repeats})"
    return OpResult(table={"name": "Benchmark", "columns": ["Operation", "Time (ms)", "Result"], "rows": rows}, message=msg, log=msg)


# ============================================================================ Reset
@op("help.reset", "Reset…", UTIL, **_NOIMG, keywords=["reset", "clear all", "fresh start", "clean"],
    params=[Param("clear_results", "bool", True, label="Clear Results table"),
            Param("clear_rois", "bool", True, label="Clear ROI Manager"),
            Param("clear_tables", "bool", True, label="Clear named tables and the last plot"),
            Param("clear_thresholds", "bool", True, label="Clear thresholds"),
            Param("clear_macro", "bool", True, label="Clear the recorded macro"),
            Param("clear_log", "bool", False, label="Clear the Log"),
            Param("close_images", "bool", False, label="Close all images"),
            Param("collect_garbage", "bool", True, label="Run the garbage collector")],
    description="Clear the Results table, ROI Manager, named tables, plots, thresholds and the recorded macro, optionally the Log and every "
                "open image (ImageJ's Plugins ▸ Utilities ▸ Reset…). Preferences are kept – use Edit ▸ Options ▸ Reset Preferences for those.")
def reset(ctx, clear_results=True, clear_rois=True, clear_tables=True, clear_thresholds=True, clear_macro=True, clear_log=False,
          close_images=False, collect_garbage=True):
    s = ctx.session
    done: list[str] = []
    if clear_results and (s.results.rows or s.results.columns):
        s.results.clear()
        s._changed("results")
        done.append("Results")
    if clear_rois and s.roi_manager.rois:
        s.roi_manager.clear()
        s._changed("rois")
        done.append("ROI Manager")
    if clear_tables and (s.tables or s.last_plot is not None):
        s.tables.clear()
        s.last_plot = None
        done.append("tables and plot")
    if clear_thresholds and s.thresholds:
        s.thresholds.clear()
        done.append("thresholds")
    if clear_macro and s.macro:
        s.macro = []
        done.append("recorded macro")
    if clear_log and s.log:
        s.log = []
        done.append("Log")
    if close_images and s.images:
        n = len(s.images)
        for iid in list(s.images):
            s.close_image(iid)
        done.append(f"{n} image(s)")
    if collect_garbage:
        gc.collect()
    msg = "Reset: cleared " + ", ".join(done) if done else "Reset: nothing to clear"
    return OpResult(message=msg, log=msg)


# ============================================================================ Monitor Events
@op("help.monitor_events", "Monitor Events", UTIL, **_NOIMG, keywords=["events", "log", "monitor", "history", "errors"],
    params=[Param("n", "int", 50, min=1, max=500, label="Entries", help="How many of the most recent log entries to show"),
            Param("level", "choice", "all", choices=["all", "info", "warning", "error"], help="Only entries of this level")],
    description="The most recent session events (commands with their run time, opened and saved files, errors) as a table – a snapshot of the "
                "Log, newest last (ImageJ's Plugins ▸ Utilities ▸ Monitor Events…; the Log panel shows them live).")
def monitor_events(ctx, n=50, level="all"):
    s = ctx.session
    entries = list(enumerate(s.log, 1))
    if level != "all":
        entries = [(i, e) for i, e in entries if str(e.get("level", "info")) == level]
    entries = entries[-int(n):]
    rows = [[i, time.strftime("%H:%M:%S", time.localtime(float(e.get("t", 0)))), e.get("level", "info"), str(e.get("msg", ""))] for i, e in entries]
    return OpResult(table={"name": "Events", "columns": ["#", "Time", "Level", "Message"], "rows": rows},
                    message=f"{len(rows)} of {len(s.log)} log entries (session version {s.version}, {len(s.listeners)} live listener(s))")


# ============================================================================ Search commands
def score_command(spec, query: str) -> int:
    """Palette-style fuzzy score of a command against a query (0 = no match); every word must match."""
    words = [w for w in query.lower().split() if w]
    if not words:
        return 0
    name = spec.name.lower()
    total = 0
    for w in words:
        if name == w:
            s = 100
        elif name.startswith(w):
            s = 80
        elif any(t.startswith(w) for t in name.replace("…", " ").replace("/", " ").replace("-", " ").split()):
            s = 65
        elif w in name:
            s = 50
        elif any(w in k.lower() for k in spec.keywords):
            s = 40
        elif w in (spec.menu or spec.category).lower():
            s = 30
        elif w in spec.id.lower():
            s = 28
        elif w in spec.description.lower():
            s = 20
        else:
            return 0
        total += s
    return total


@op("help.search_commands", "Search Commands…", UTIL, **_NOIMG, keywords=["search", "find", "command finder", "what does"],
    params=[Param("query", "string", "", help="Words to look for in command names, keywords, menus, ids and descriptions"),
            Param("max_results", "int", 50, min=1, max=500)],
    description="Find Lumen commands by name, keyword, menu or description and list them with their id, menu, shortcut and description "
                "(the ⌘K palette does the same interactively and also searches the ImageJ wiki and the image.sc forum).")
def search_commands(ctx, query="", max_results=50):
    q = str(query or "").strip()
    if not q:
        raise ValueError("Type one or more words to search for")
    hits = [(score_command(sp, q), sp) for sp in REGISTRY.ops.values()]
    hits = sorted(((sc, sp) for sc, sp in hits if sc > 0), key=lambda x: (-x[0], x[1].name.lower()))[:int(max_results)]
    rows = [[sp.id, sp.name, sp.menu or sp.category, sp.shortcut or "", sp.description, sc] for sc, sp in hits]
    return OpResult(table={"name": "Search", "columns": ["Id", "Name", "Menu", "Shortcut", "Description", "Score"], "rows": rows},
                    message=f"{len(rows)} command(s) match {q!r}" if rows else f"No command matches {q!r} – try the ImageJ wiki or image.sc search in the ⌘K palette")


# ============================================================================ Search… (files, images, Log, script)
SEARCH_SUFFIXES = {".py", ".ijm", ".txt", ".md", ".js", ".json", ".csv", ".tsv", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".groovy", ".bsh", ".r", ".jl"}
SEARCH_MAX_FILE_BYTES = 2_000_000


def search_folders() -> list[Path]:
    """The folders Search… greps: <LUMEN_HOME>/plugins, /macros, /scripts and every $LUMEN_PLUGINS folder."""
    from ..io import lumen_home
    home = lumen_home()
    dirs = [home / "plugins", home / "macros", home / "scripts"]
    for d in _plugin_dirs()[1:]:
        if d not in dirs:
            dirs.append(d)
    return dirs


def iter_search_files(folders: list[Path]) -> Iterator[tuple[str, Path]]:
    """(folder label, file) for every text-like file under the folders (hidden / __pycache__ entries and files > 2 MB skipped)."""
    seen: set[Path] = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        for f in sorted(folder.rglob("*")):
            try:
                if not f.is_file() or f.suffix.lower() not in SEARCH_SUFFIXES:
                    continue
                if any(part.startswith(".") or part == "__pycache__" for part in f.relative_to(folder).parts):
                    continue
                if f.stat().st_size > SEARCH_MAX_FILE_BYTES:
                    continue
                rf = f.resolve()
            except OSError:
                continue
            if rf in seen:
                continue
            seen.add(rf)
            yield folder.name, f


def flatten_metadata(obj: Any, prefix: str = "", depth: int = 0, out: list[tuple[str, str]] | None = None) -> list[tuple[str, str]]:
    """Nested metadata → [(dotted key, text value)] (lists of ≤ 50 entries expanded, deeper levels repr'd)."""
    if out is None:
        out = []
    if depth > 5:
        out.append((prefix, repr(obj)[:300]))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten_metadata(v, f"{prefix}.{k}" if prefix else str(k), depth + 1, out)
    elif isinstance(obj, (list, tuple)) and len(obj) <= 50:
        for i, v in enumerate(obj):
            flatten_metadata(v, f"{prefix}[{i}]", depth + 1, out)
    else:
        out.append((prefix, str(obj)[:300]))
    return out


@op("help.search_files", "Search…", UTIL, **_NOIMG, keywords=["search", "grep", "find text", "macros folder", "plugins folder", "metadata", "log", "script"],
    params=[Param("query", "string", "", help="Text to look for (plain substring, not a regular expression)"),
            Param("case_sensitive", "bool", False),
            Param("files", "bool", True, label="Plugins / macros / scripts folders", help="Every text-like file under <LUMEN_HOME>/plugins, /macros, /scripts and $LUMEN_PLUGINS"),
            Param("images", "bool", True, label="Open images", help="Image names and metadata"),
            Param("log", "bool", True, label="Log"),
            Param("script", "bool", True, label="Recorded script"),
            Param("max_results", "int", 500, min=1, max=100000)],
    description="Search for a string in the plugins / macros / scripts folders, the open images' names and metadata, the Log and the recorded script; the matches "
                "(where, file, line, text) go to a 'Search Files' table. Nothing is opened (ImageJ's Plugins ▸ Utilities ▸ Search…).")
def search_files(ctx, query="", case_sensitive=False, files=True, images=True, log=True, script=True, max_results=500):
    q = str(query or "")
    if not q.strip():
        raise ValueError("Type the text to search for")
    needle = q if case_sensitive else q.lower()

    def hit(text: str) -> bool:
        return needle in (text if case_sensitive else text.lower())

    rows: list[list[Any]] = []
    limit = int(max_results)
    n_files = 0
    truncated = False

    def add(where: str, where_file: str, line: Any, text: str) -> bool:
        nonlocal truncated
        if len(rows) >= limit:
            truncated = True
            return False
        rows.append([where, where_file, line, text.strip()[:300]])
        return True

    if files:
        found = list(iter_search_files(search_folders()))
        for i, (label, f) in enumerate(found):
            ctx.check_cancel()
            n_files += 1
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for ln, line in enumerate(text.splitlines(), 1):
                if hit(line) and not add(label, str(f), ln, line):
                    break
            if truncated:
                break
            if len(found) > 20:
                ctx.progress((i + 1) / len(found), f"{f.name}")
    s = ctx.session
    if images and not truncated:
        for im in s.images.values():
            if hit(im.name):
                add("image", im.name, "", im.name)
            for key, val in flatten_metadata(im.metadata):
                if hit(key) or hit(val):
                    if not add("metadata", im.name, key, val):
                        break
            if truncated:
                break
    if log and not truncated:
        for i, e in enumerate(s.log, 1):
            msg = str(e.get("msg", ""))
            if hit(msg) and not add("Log", "", i, msg):
                break
    if script and not truncated:
        for i, line in enumerate(s.macro, 1):
            if hit(line) and not add("script", "recorded script", i, line):
                break
    where = ", ".join(w for w, on in (("files", files), ("images", images), ("Log", log), ("script", script)) if on) or "nothing"
    msg = f"{len(rows)} match(es) for {q!r} in {where}" + (f" ({n_files} file(s) searched)" if files else "") + (f"; stopped after {limit} results" if truncated else "")
    return OpResult(table={"name": "Search Files", "columns": ["Where", "File", "Line", "Text"], "rows": rows}, message=msg)


# ============================================================================ Memory & Threads
_THREAD_LIMITER: Any = None


def apply_thread_limit(max_threads: int) -> str:
    """Limit the BLAS / OpenMP thread pools numpy, scipy and scikit-learn use (0 = restore all cores)."""
    global _THREAD_LIMITER
    try:
        import threadpoolctl  # type: ignore[import-not-found]
    except ImportError:
        return "stored only (threadpoolctl is not installed)"
    if _THREAD_LIMITER is not None:
        try:
            _THREAD_LIMITER.restore_original_limits()
        except Exception:  # pragma: no cover
            pass
        _THREAD_LIMITER = None
    if max_threads <= 0:
        return "no limit (all cores)"
    _THREAD_LIMITER = threadpoolctl.threadpool_limits(limits=int(max_threads))
    return f"BLAS/OpenMP pools limited to {int(max_threads)} thread(s)"


def apply_undo_budget(session, undo_budget_mb: float) -> int:
    """Set the undo budget of every open history and of histories created later; returns the budget in bytes."""
    budget = max(0, int(round(float(undo_budget_mb) * _MB)))
    History.set_default_budget(budget)
    for h in session.histories.values():
        h.set_budget(budget)
    session.settings["undo_budget_mb"] = int(round(float(undo_budget_mb)))
    return budget


@op("edit.options_memory", "Memory & Threads…", OPT, **_NOIMG, keywords=["memory", "threads", "undo buffer", "ram", "parallel", "cores"],
    params=[Param("undo_budget_mb", "int", DEFAULT_UNDO_BUDGET_MB, min=0, max=1_000_000, label="Undo memory per image (MB)",
                  help="Snapshots older than this budget are dropped from each image's undo history (0 keeps only the last step)"),
            Param("max_threads", "int", 0, min=0, max=1024, label="Parallel threads", help="BLAS / OpenMP threads for numpy-based commands; 0 = all cores"),
            Param("collect_garbage", "bool", False, label="Run the garbage collector now")],
    description="The undo memory budget per image (applied to every open history and to new ones) and the number of threads numpy / scipy / "
                "scikit-learn may use (via threadpoolctl when installed; otherwise the value is stored only) – ImageJ's Edit ▸ Options ▸ Memory & Threads.")
def options_memory(ctx, undo_budget_mb=DEFAULT_UNDO_BUDGET_MB, max_threads=0, collect_garbage=False):
    s = ctx.session
    budget = apply_undo_budget(s, undo_budget_mb)
    s.settings["max_threads"] = int(max_threads)
    how = apply_thread_limit(int(max_threads))
    freed = f", collected {gc.collect()} objects" if collect_garbage else ""
    _img, undo_bytes = _session_memory(s)
    msg = f"Undo budget {_mb(budget)} per image ({len(s.histories)} histories, {_mb(undo_bytes)} in use); threads: {how}{freed}"
    return OpResult(message=msg, log=msg)


# ============================================================================ Proxy
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")
_NO_PROXY_VARS = ("NO_PROXY", "no_proxy")


def apply_proxy(proxy: str, no_proxy: str = "") -> None:
    """Mirror the proxy settings into the environment variables urllib / pooch / requests honour."""
    for var in _PROXY_VARS:
        if proxy:
            os.environ[var] = proxy
        else:
            os.environ.pop(var, None)
    for var in _NO_PROXY_VARS:
        if no_proxy:
            os.environ[var] = no_proxy
        else:
            os.environ.pop(var, None)
    # urllib.request.urlopen caches one global OpenerDirector whose ProxyHandler snapshots the environment when it is
    # first built and never rebuilds it: rebuild it here so later urlopen() calls (the PyPI check, sample / URL downloads)
    # go through the new proxy - or directly again once it was cleared
    urllib.request.install_opener(urllib.request.build_opener())


@op("edit.options_proxy", "Proxy Settings…", OPT, **_NOIMG, keywords=["proxy", "network", "firewall", "download", "http"],
    params=[Param("proxy", "string", "", label="Proxy server", help="host:port or http://user:pass@host:port; empty = direct connection"),
            Param("no_proxy", "string", "", label="Bypass for", help="Comma-separated hosts that never go through the proxy, e.g. localhost,.local")],
    description="HTTP/HTTPS proxy used when Lumen downloads ImageJ samples or opens URLs: stored in the preferences and exported as "
                "HTTP_PROXY / HTTPS_PROXY (and NO_PROXY) for this server process – ImageJ's Edit ▸ Options ▸ Proxy Settings.")
def options_proxy(ctx, proxy="", no_proxy=""):
    s = ctx.session
    proxy = str(proxy or "").strip()
    no_proxy = ",".join(h.strip() for h in str(no_proxy or "").split(",") if h.strip())
    if proxy and "://" not in proxy:
        proxy = "http://" + proxy
    if proxy and any(ch.isspace() for ch in proxy):
        raise ValueError(f"Invalid proxy URL {proxy!r}")
    s.settings["proxy"] = proxy
    s.settings["no_proxy"] = no_proxy
    apply_proxy(proxy, no_proxy)
    msg = f"Proxy: {proxy}" + (f" (bypass {no_proxy})" if no_proxy else "") if proxy else "Proxy disabled (direct connection)"
    return OpResult(message=msg, log=msg)


# ============================================================================ Compiler (plugins)
@op("help.compiler_info", "Compiler… (Python plugins)", OPT, **_NOIMG, keywords=["compiler", "plugin", "javac", "develop", "reload", "script"],
    description="There is no Java compiler: Lumen plugins are Python files in ~/.lumen/plugins (or $LUMEN_PLUGINS) whose @op functions register "
                "as commands when the server starts. This lists the plugin folders and files and explains how to reload them.")
def compiler_info(ctx):
    dirs = _plugin_dirs()
    files = [f for d in dirs if d.is_dir() for f in sorted(d.glob("*.py"))]
    loaded = {}
    for spec in REGISTRY.ops.values():
        mod = getattr(spec.func, "__module__", "") or ""
        if mod.startswith("lumen_plugin_"):
            loaded.setdefault(mod[len("lumen_plugin_"):], []).append(spec.id)
    rows: list[list[Any]] = [
        ["Language", "Python 3 – no compilation step; the ImageJ macro language is translated by script.run_ij_macro / ij_compat"],
        ["Plugin folders", os.pathsep.join(str(d) for d in dirs)],
        ["Plugin files", str(len(files))],
        ["Minimal plugin", 'from lumen.core.registry import op, Param\n@op("my.invert", "My Invert", "Plugins/My")\ndef my_invert(plane, ctx): return 255 - plane'],
        ["Reload", "Plugins ▸ Utilities ▸ Reload Plugins (help.reload_plugins) re-imports the plugin files and replaces their commands; "
                   "the menus pick them up after a page reload. A restart of `lumen` does the same."],
        ["Script console", "lumen.run(\"<id>\", …), lumen.image, np – the Python API is documented in the Command Reference (Help menu)"],
    ]
    for f in files:
        ids = loaded.get(f.stem, [])
        rows.append([f"plugin {f.name}", f"{len(ids)} command(s): {', '.join(ids)}" if ids else "not loaded (no @op commands, import error or added since start-up)"])
    msg = f"Lumen plugins are Python files: {len(files)} file(s) in {len(dirs)} folder(s), {sum(len(v) for v in loaded.values())} plugin command(s) loaded"
    return OpResult(table={"name": "Compiler", "columns": ["Property", "Value"], "rows": rows}, message=msg)


@op("help.reload_plugins", "Reload Plugins", UTIL, **_NOIMG, keywords=["reload", "plugins", "update menus", "refresh"],
    description="Re-import every plugin file in ~/.lumen/plugins and $LUMEN_PLUGINS: the commands they registered are removed first so edited "
                "plugins replace themselves (ImageJ's Help ▸ Update Menus). Reload the page to see new menu entries.")
def reload_plugins(ctx):
    from . import load_user_plugins
    before = [oid for oid, sp in REGISTRY.ops.items() if (getattr(sp.func, "__module__", "") or "").startswith("lumen_plugin_")]
    for oid in before:
        REGISTRY.ops.pop(oid, None)
    files = load_user_plugins()
    try:  # ImageJ macro-language tool sets (.ijm) are rescanned too, like plugins.reload
        from .macro_tools import load_macro_tools
        load_macro_tools()
    except Exception as exc:  # noqa: BLE001
        ctx.session.add_log(f"macro tool sets: {exc}")
    after = [oid for oid, sp in REGISTRY.ops.items() if (getattr(sp.func, "__module__", "") or "").startswith("lumen_plugin_")]
    # broadcast the "ops" event: the UI refetches /api/meta and rebuilds palette + menus, no page reload needed
    from .plugins_ops import _ops_changed
    _ops_changed(ctx.session)
    msg = f"Reloaded {len(files)} plugin file(s): {len(after)} plugin command(s) ({len(before)} before); menus and palette updated"
    rows = [[oid, REGISTRY.ops[oid].name, REGISTRY.ops[oid].menu or REGISTRY.ops[oid].category] for oid in after]
    return OpResult(table={"name": "Plugins", "columns": ["Id", "Name", "Menu"], "rows": rows}, message=msg, log=msg)


# ============================================================================ DICOM
DICOM_KEYS = {"open_as_32bit": "dicom_open_as_32bit", "ignore_rescale": "dicom_ignore_rescale",
              "calibrate_images": "dicom_calibrate_images", "window_level_on_open": "dicom_window_level_on_open"}


@op("edit.options_dicom", "DICOM…", OPT, **_NOIMG, keywords=["dicom", "medical", "rescale", "hounsfield", "window level"],
    params=[Param("open_as_32bit", "bool", False, label="Open as 32-bit float", help="Apply rescale slope/intercept and keep float values (e.g. Hounsfield units)"),
            Param("ignore_rescale", "bool", False, label="Ignore rescale slope/intercept"),
            Param("calibrate_images", "bool", True, label="Calibrate images", help="Use PixelSpacing / SliceThickness as the spatial calibration"),
            Param("window_level_on_open", "bool", True, label="Apply window center/width on open", help="Set the display range from the file's WindowCenter / WindowWidth")],
    description="Preferences for the DICOM reader, stored as dicom_open_as_32bit, dicom_ignore_rescale, dicom_calibrate_images and "
                "dicom_window_level_on_open in the session settings (ImageJ's Edit ▸ Options ▸ DICOM). file.open consults "
                "session.settings[\"dicom_open_as_32bit\"] and friends once the reader is wired to them.")
def options_dicom(ctx, open_as_32bit=False, ignore_rescale=False, calibrate_images=True, window_level_on_open=True):
    s = ctx.session
    values = {"open_as_32bit": bool(open_as_32bit), "ignore_rescale": bool(ignore_rescale),
              "calibrate_images": bool(calibrate_images), "window_level_on_open": bool(window_level_on_open)}
    for k, v in values.items():
        s.settings[DICOM_KEYS[k]] = v
    msg = "DICOM: " + ", ".join(f"{k}={'on' if v else 'off'}" for k, v in values.items())
    return OpResult(message=msg)


# ============================================================================ Virtual stack (batch)
@op("batch.virtual_stack", "Virtual Stack (process a folder plane by plane)…", "Process/Batch",
    kind="image", requires_image=False, scope_param=False, preview=False, roi_aware=False,
    keywords=["virtual stack", "batch", "folder", "lazy", "large", "out of core", "plane by plane"],
    params=[Param("input_folder", "string", "", label="Input folder"),
            Param("pattern", "string", r"\.(?i:tiff?|png|jpe?g|nd2|czi|lif)$", label="File pattern (regex)", help="Only file names matching this regular expression are processed"),
            Param("recursive", "bool", False, label="Include subfolders"),
            Param("output_folder", "string", "", label="Output folder", help="Empty = <input folder>/lumen_output"),
            Param("script", "text", 'lumen.run("filters.gaussian", sigma=2)', help="Python using the lumen API, run once per file with lumen.image bound to it"),
            Param("format", "choice", "tif", label="Output format", choices=["tif", "ome.tif", "png", "jpg", "npy", "none"]),
            Param("save_results", "bool", True, label="Save Results table", help="Append measurements to Results.csv in the output folder (with a File column)"),
            Param("continue_on_error", "bool", True, label="Continue on error"),
            Param("max_files", "int", 0, min=0, label="Max files", help="0 = all")],
    description="ImageJ's Process ▸ Batch ▸ Virtual Stack… opens a folder lazily and processes it slice by slice. Lumen's image model holds "
                "whole stacks in memory, so this processes the folder plane by plane instead: each file is opened on its own, the script runs "
                "on it, the result is saved and the file is closed before the next one is read – the whole stack is never held at once.")
def virtual_stack(ctx, input_folder="", pattern=r"\.(?i:tiff?|png|jpe?g|nd2|czi|lif)$", recursive=False, output_folder="", script="",
                  format="tif", save_results=True, continue_on_error=True, max_files=0):
    from .batch import _run_batch
    return _run_batch(ctx, input_folder=input_folder, pattern=pattern, recursive=recursive, output_folder=output_folder, output_format=format,
                      script=script, save_results=save_results, close_after=True, continue_on_error=continue_on_error, max_files=max_files,
                      label="Virtual stack")


# ============================================================================ Check for updates
def source_tree_version() -> tuple[str | None, Path | None]:
    """The version declared in pyproject.toml next to the lumen package (a source checkout), if any."""
    try:
        pp = Path(lumen.__file__).resolve().parents[1] / "pyproject.toml"
        if not pp.is_file():
            return None, None
        import tomllib
        data = tomllib.loads(pp.read_text(encoding="utf-8"))
        return str(data.get("project", {}).get("version") or "") or None, pp
    except Exception:
        return None, None


def pypi_version(name: str = PYPI_NAME, timeout: float = 3.0) -> tuple[str | None, str]:
    """(latest version on PyPI, status) – status is 'ok', 'no network', 'not on PyPI' or an HTTP error."""
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # the global opener is rebuilt by apply_proxy (edit.options_proxy)
            data = json.load(r)
        v = str(data.get("info", {}).get("version") or "")
        return (v or None), ("ok" if v else "no version on PyPI")
    except urllib.error.HTTPError as e:
        return None, "not on PyPI" if e.code == 404 else f"HTTP {e.code}"
    except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
        return None, "no network" if not isinstance(e, ValueError) else f"bad response: {e}"


def compare_versions(a: str, b: str) -> int:
    """-1, 0, 1 for a <, ==, > b (PEP 440 when `packaging` is available, else a numeric fallback)."""
    try:
        from packaging.version import Version
        va, vb = Version(a), Version(b)
        return (va > vb) - (va < vb)
    except Exception:
        def key(v: str) -> list[int]:
            out = []
            for part in v.split("."):
                digits = "".join(ch for ch in part if ch.isdigit())
                out.append(int(digits) if digits else 0)
            return out
        ka, kb = key(a), key(b)
        return (ka > kb) - (ka < kb)


@op("help.check_updates", "Check for Updates…", HELP, **_NOIMG, keywords=["update", "upgrade", "version", "pypi", "new release"],
    params=[Param("check_pypi", "bool", True, label="Ask PyPI", help="Query pypi.org for the latest lumen-imaging release (3 s timeout); off = offline check only")],
    description="Compare the running Lumen version with the source tree's pyproject.toml and, when the network is allowed, with the latest "
                "lumen-imaging release on PyPI. Nothing is installed: upgrade with `pip install -U lumen-imaging` (or `git pull` in a checkout).")
def check_updates(ctx, check_pypi=True):
    installed = str(lumen.__version__)
    src_version, pp = source_tree_version()
    rows: list[list[Any]] = [["Installed (lumen.__version__)", installed, sys.prefix]]
    status: list[str] = []
    if src_version:
        rel = compare_versions(src_version, installed)
        note = "same as installed" if rel == 0 else ("source tree is newer – restart lumen / reinstall (pip install -e .)" if rel > 0 else "source tree is older")
        rows.append(["Source tree (pyproject.toml)", src_version, f"{pp} – {note}"])
        if rel > 0:
            status.append(f"source tree declares {src_version}")
    else:
        rows.append(["Source tree (pyproject.toml)", "not found", "installed as a package, not a checkout"])
    if check_pypi:
        latest, st = pypi_version()
        if latest:
            rel = compare_versions(latest, installed)
            if rel > 0:
                rows.append([f"PyPI {PYPI_NAME}", latest, f"newer than installed – upgrade with: pip install -U {PYPI_NAME}"])
                status.append(f"{latest} is available on PyPI (pip install -U {PYPI_NAME})")
            else:
                rows.append([f"PyPI {PYPI_NAME}", latest, "up to date" if rel == 0 else "installed version is newer than the PyPI release"])
        else:
            rows.append([f"PyPI {PYPI_NAME}", "unknown", st])
            if st == "no network":
                status.append("no network – PyPI could not be reached")
    else:
        rows.append([f"PyPI {PYPI_NAME}", "not checked", "check_pypi is off"])
    msg = f"Lumen {installed}: " + ("; ".join(status) if status else "up to date (nothing newer found)") + " – nothing was installed"
    return OpResult(table={"name": "Updates", "columns": ["Source", "Version", "Note"], "rows": rows}, message=msg, log=msg)


# ============================================================================ Update Lumen (the actual updater)
def package_root() -> Path:
    """The folder that holds the ``lumen`` package: a checkout's repository root, or site-packages for a wheel install."""
    return Path(lumen.__file__).resolve().parents[1]


def install_kind(root: Path | None = None) -> str:
    """'git' for a git checkout, 'source' for a plain source folder (pyproject.toml, no .git), else 'pip' (installed package)."""
    root = package_root() if root is None else Path(root)
    if (root / ".git").exists():
        return "git"
    if (root / "pyproject.toml").is_file():
        return "source"
    return "pip"


def update_argv(kind: str, git: str = "git") -> list[str]:
    """The update command as a list argv (never a shell string): git pull in a checkout, pip upgrade for a package install."""
    if kind == "git":
        return [git, "pull", "--ff-only"]
    if kind == "pip":
        return [sys.executable, "-m", "pip", "install", "--upgrade", "--no-input", PYPI_NAME]
    raise ValueError(f"A {kind} installation cannot be updated automatically")


def _noninteractive_env() -> dict[str, str]:
    """Environment for the update subprocesses: fail fast instead of prompting (the op holds the session lock, so a git / ssh /
    pip prompt on the server's terminal would stall every client until the timeout).  Credential helpers keep working."""
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")  # git: never ask for credentials on /dev/tty or the Windows console
    env.setdefault("GCM_INTERACTIVE", "never")  # Git Credential Manager: no GUI dialog
    env.setdefault("SSH_ASKPASS_REQUIRE", "never")  # OpenSSH ≥ 8.4: no askpass popup (no controlling tty either, below)
    env.setdefault("PIP_NO_INPUT", "1")  # pip: never prompt for an index password
    env.setdefault("PYTHONIOENCODING", "utf-8")  # Python children (pip) print UTF-8, as the output is decoded (Windows: cp1252 otherwise)
    return env


def _kill(p: "subprocess.Popen[str]") -> None:
    """Kill the command and, on POSIX, its whole process group (git-remote-https / ssh helpers)."""
    try:
        if os.name != "nt":
            import signal

            os.killpg(p.pid, signal.SIGKILL)
    except (OSError, AttributeError, ProcessLookupError):
        pass
    try:
        p.kill()
    except OSError:
        pass
    try:
        p.communicate(timeout=5)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass


def run_command(argv: list[str], cwd: Path | None = None, timeout: float = 600.0, cancel: "threading.Event | None" = None) -> tuple[int | None, str]:
    """Run a command (list argv, no shell, UTF-8 output, never interactive) → (exit code, or None when it could not run; combined
    stdout + stderr).  stdin is closed and the child is detached from the controlling terminal (POSIX) with a non-interactive
    environment, so git / ssh / pip fail fast instead of waiting for a prompt; ``cancel`` (the job's cancel flag) kills it."""
    try:
        p = subprocess.Popen(list(argv), cwd=str(cwd) if cwd else None, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, encoding="utf-8", errors="replace", env=_noninteractive_env(), start_new_session=(os.name != "nt"))
    except FileNotFoundError as e:
        return None, f"command not found: {e}"
    except OSError as e:
        return None, str(e)
    deadline = time.monotonic() + float(timeout)
    while True:
        try:
            out, err = p.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if cancel is not None and cancel.is_set():
                _kill(p)
                return None, "cancelled"
            if time.monotonic() >= deadline:
                _kill(p)
                return None, f"timed out after {timeout:g} s"
    out, err = (out or "").strip(), (err or "").strip()
    return int(p.returncode), (out + ("\n" + err if err else "")).strip()


def _status(code: int | None) -> str:
    return "exit 0" if code == 0 else (f"exit {code}" if code is not None else "failed")


@op("help.update", "Update Lumen…", HELP, **_NOIMG, keywords=["update", "upgrade", "git pull", "pip install", "fiji update", "new version", "updater"],
    params=[Param("check_only", "bool", False, label="Check only", help="Report whether an update is available (git fetch / PyPI) without installing anything"),
            Param("dry_run", "bool", False, label="Dry run", help="Show the command that would run; nothing is executed"),
            Param("timeout", "int", 600, min=10, max=3600, unit="s", advanced=True)],
    description="Update the running Lumen in place: `git pull --ff-only` in a git checkout, `python -m pip install --upgrade lumen-imaging` for a package install "
                "(a plain source folder without git cannot be updated automatically). The command's output goes to the Log and an 'Update' table. The server is never "
                "restarted – restart `lumen` afterwards to run the new version (ImageJ's Help ▸ Update ImageJ… / Fiji's Update…).")
def update(ctx, check_only=False, dry_run=False, timeout=600):
    root = package_root()
    kind = install_kind(root)
    installed = str(lumen.__version__)
    where = str(root) if kind != "pip" else sys.prefix
    rows: list[list[Any]] = [["Installed", installed, f"{kind} install – {where}"]]
    table = {"name": "Update", "columns": ["Step", "Status", "Output"], "rows": rows}
    git = shutil.which("git") if kind == "git" else None
    if kind == "git" and not git:
        rows.append(["git", "not found", "install git (or update the checkout by hand)"])
        msg = f"Lumen {installed} runs from the git checkout {root} but git was not found on PATH – nothing was changed"
        return OpResult(table=table, message=msg, log=msg)
    if kind == "source":
        rows.append(["Source folder", "cannot update automatically", "replace the folder with a newer copy (or clone it with git) and run `pip install -e .` again"])
        msg = f"Lumen {installed} runs from the source folder {root} without git – nothing was changed"
        return OpResult(table=table, message=msg, log=msg)
    outputs: list[str] = []
    cancel = ctx.job["cancel"] if ctx.job else None  # Cancel in the jobs panel kills the subprocess

    def run(argv, cwd):
        code, out = run_command(argv, cwd, timeout, cancel)
        if out == "cancelled":
            ctx.check_cancel()
        return code, out

    if check_only:
        if kind == "git":
            steps = [[git, "fetch", "--quiet"], [git, "rev-list", "--count", "HEAD..@{upstream}"]]
            if dry_run:
                rows += [[" ".join(a), "dry run", "not executed"] for a in steps]
                note = "dry run – would fetch and count the commits behind the upstream branch"
            else:
                code, out = run(steps[0], root)
                rows.append([" ".join(steps[0]), _status(code), out[:4000]])
                outputs.append(out)
                if code == 0:
                    code2, out2 = run(steps[1], root)
                    rows.append([" ".join(steps[1]), _status(code2), out2[:4000]])
                    outputs.append(out2)
                    behind = int(out2.strip()) if code2 == 0 and out2.strip().isdigit() else None
                    if behind == 0:
                        note = "up to date with the upstream branch"
                    elif behind:
                        note = f"{behind} commit(s) behind the upstream branch – run Update Lumen (without 'check only') to pull them"
                    else:
                        note = f"could not compare with the upstream branch ({out2.strip().splitlines()[0] if out2.strip() else 'no upstream configured?'})"
                else:
                    note = f"git fetch failed ({_status(code)}): {out.strip().splitlines()[0] if out.strip() else 'no output'}"
        else:
            latest, st = pypi_version()
            if latest:
                rel = compare_versions(latest, installed)
                rows.append([f"PyPI {PYPI_NAME}", latest, "newer than installed" if rel > 0 else ("up to date" if rel == 0 else "installed version is newer")])
                note = f"{latest} is available on PyPI – run Update Lumen (without 'check only') to install it" if rel > 0 else "up to date (nothing newer on PyPI)"
            else:
                rows.append([f"PyPI {PYPI_NAME}", "unknown", st])
                note = f"PyPI could not be checked ({st})"
        msg = f"Lumen {installed}: {note}; nothing was installed"
        return OpResult(table=table, message=msg, log="\n".join([msg, *[o for o in outputs if o]]))
    argv = update_argv(kind, git or "git")
    shown = " ".join(argv)
    if dry_run:
        rows.append([shown, "dry run", "not executed"])
        msg = f"Dry run: would run `{shown}` in {where}; nothing was changed"
        return OpResult(table=table, message=msg, log=msg)
    code, out = run(argv, root if kind == "git" else None)
    rows.append([shown, _status(code), out[:4000]])
    if code == 0:
        msg = f"Update finished (`{shown}`). Restart Lumen (the server process) to run the new version – nothing was restarted"
    else:
        first = out.strip().splitlines()[0] if out.strip() else "no output"
        msg = f"Update failed ({_status(code)}: {first}) – see the Update table; nothing was changed or restarted"
    return OpResult(table=table, message=msg, log=msg + ("\n" + out if out else ""))
