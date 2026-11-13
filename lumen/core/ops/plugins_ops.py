"""Plugins ▸ Macros and Plugins ▸ Utilities: startup scripts, macro install/run, plugin templates, hot reload,
DICOM header dump and the command list / finder.

Files live under the Lumen home folder (``$LUMEN_HOME`` or ``~/.lumen``):

``startup.py`` / ``startup.ijm``   run once when ``lumen`` starts (``run_startup_scripts``; errors go to the Log)
``plugins/*.py``                   plugin modules – their ``@op`` functions become commands (``tool=True`` ones: tool-rail tools)
``plugins/<name>.ijm``             an installed ImageJ macro, wrapped by ``plugins/<name>_macro.py`` into a command
``macros/*.py``                    installed plain Python scripts (wrapped by ``plugins/<name>_script.py``) and recordings
``plugins|macros/<set>.ijm``       an ImageJ *tool macro set* (``macro "Name Tool - icon" { … }``): its tools go on the tool
                                   rail without a wrapper (``lumen.core.ops.macro_tools``), also from ``startup.ijm``

Hot reload: ``plugins.reload`` re-imports every plugin file (``lumen.core.ops.reload_user_plugins``) and rescans the
macro tools, replacing the commands they registered before.  Every change to the registry bumps
``session.settings["ops_version"]`` and broadcasts an ``{"type": "ops", "ops_version": n}`` event so the UI can refetch
``/api/meta``.

Plugin frames (ImageJ's Plugins ▸ New ▸ Plugin Frame): a command declared with ``frame=True`` / ``kind="frame"`` whose
output is shown in its own docked sub-panel of the inspector section "Plugin Frames" (``frames.image_statistics`` is the
built-in example; the "Plugin Frame" template writes one).
"""
from __future__ import annotations

import builtins
import difflib
import keyword
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..registry import REGISTRY, OpResult, Param, op

MACROS = "Plugins/Macros"
UTIL = "Plugins/Utilities"
FRAMES = "Plugins/Frames"
_NOIMG = dict(kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False)
_IMG = dict(kind="analysis", preview=False, roi_aware=False, scope_param=False)
_FRAME = dict(kind="frame", requires_image=False, roi_aware=False, scope_param=False)

STARTUP_PY = "startup.py"
STARTUP_IJM = "startup.ijm"
MACRO_EXTS = (".ijm", ".txt")
TEMPLATE_KINDS = ["Python command", "Python filter (plane)", "ImageJ macro", "Script with dialog", "Plugin Tool", "Macro Tool", "Plugin Frame"]


# ============================================================================ folders & helpers
def lumen_home() -> Path:
    from . import lumen_home as _home
    return _home()


def plugins_dir(create: bool = True) -> Path:
    d = lumen_home() / "plugins"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def macros_dir(create: bool = True) -> Path:
    d = lumen_home() / "macros"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def startup_paths() -> list[Path]:
    home = lumen_home()
    return [home / STARTUP_PY, home / STARTUP_IJM]


# module-level names the templates / wrappers bind and call inside the generated function body (lower-case only
# matters: the slug is lower-cased)
_RESERVED_SLUGS = frozenset({"np", "ndi", "math", "op", "cast_to"})


def _slug(name: str) -> str:
    """File / op-id / ``def`` name for a command name: ``[0-9a-z_]+`` starting with a letter or ``m_``.  The slug is also
    the generated function's name, so Python keywords (``def pass(…)`` is a SyntaxError that breaks every later plugin
    load), builtins (``def float(…)`` shadows what the template body calls) and the template's own imports get a
    trailing underscore (``Pass`` → ``pass_``, ``Float`` → ``float_``, ``np`` → ``np_``)."""
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name)).strip("_").lower()
    if not s:
        s = "macro"
    if s[0].isdigit():
        s = "m_" + s
    if keyword.iskeyword(s) or s in _RESERVED_SLUGS or hasattr(builtins, s):
        s += "_"
    return s


def _lines_table(name: str, text: str) -> dict[str, Any]:
    return {"name": name, "columns": ["Line", "Text"], "rows": [[i, ln] for i, ln in enumerate(text.splitlines(), 1)]}


def _ops_changed(session: Any) -> int:
    """Record that the set of commands changed: bump ``settings["ops_version"]`` and broadcast an ``ops`` event."""
    n = int(session.settings.get("ops_version", 0) or 0) + 1
    session.settings["ops_version"] = n
    session._changed("ops", ops_version=n, commands=len(REGISTRY.ops))
    return n


def _plugin_file_rows(file: Path) -> tuple[str, str]:
    """(commands, status) of a plugin file from the package's bookkeeping."""
    from . import PLUGIN_ERRORS, PLUGIN_OPS
    key = str(file)
    ids = [oid for oid in PLUGIN_OPS.get(key, []) if oid in REGISTRY.ops]
    if key in PLUGIN_ERRORS:
        return ", ".join(ids), f"error: {PLUGIN_ERRORS[key]}"
    if key in PLUGIN_OPS:
        return ", ".join(ids), f"loaded ({len(ids)} command(s))" if ids else "loaded (no @op commands)"
    return "", "not loaded – Plugins ▸ Utilities ▸ Reload Plugins"


# ============================================================================ running files
def run_python_file(session: Any, path: "str | Path") -> str:
    """Run a Lumen Python script file (``lumen`` API); returns what it printed, raises ValueError on error."""
    from ..scripting import run_script
    p = Path(path)
    out = run_script(session, p.read_text(encoding="utf-8"))
    if out.get("error"):
        err = str(out["error"]).rstrip()
        session.add_log(f"{p.name}:\n{err}", level="error")
        last = err.splitlines()[-1] if err.splitlines() else err
        raise ValueError(f"{p.name}: {last}")
    return out.get("stdout", "")


def run_macro_file(session: Any, path: "str | Path", roi: dict[str, Any] | None = None) -> OpResult:
    """Run an ImageJ macro file through the compatibility layer; the result carries the Log text and the macro's selection."""
    from ..ij_compat import IJ
    p = Path(path)
    ij = IJ(session)
    ij.roi = roi
    output = ij.macro(p.read_text(encoding="utf-8"))
    res = OpResult(log=output or None, message=f"{p.name} finished" + (f" – {len(output.splitlines())} line(s) of output" if output else ""))
    if ij.roi is not None:
        res.roi = ij.roi
    elif roi is not None:
        res.clear_roi = True
    return res


def run_file(session: Any, path: "str | Path", roi: dict[str, Any] | None = None) -> OpResult:
    """Run a ``.py`` (Lumen Python) or ``.ijm`` (ImageJ macro) file and return an OpResult for the UI."""
    p = Path(str(path)).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {p}")
    ext = p.suffix.lower()
    if ext == ".py":
        output = run_python_file(session, p)
        return OpResult(log=output or None, message=f"{p.name} finished" + (f" – {len(output.splitlines())} line(s) of output" if output else ""))
    if ext in MACRO_EXTS:
        return run_macro_file(session, p, roi=roi)
    raise ValueError(f"{p.name}: only .py (Lumen Python) and .ijm (ImageJ macro) files can be run")


def run_startup_scripts(session: Any) -> list[dict[str, Any]]:
    """Run ``startup.py`` (Python, ``lumen`` API) then ``startup.ijm`` (ImageJ macro) from the Lumen home folder if they exist.

    Never raises: every failure is written to the session log (level ``error``) and reported in the returned list."""
    results: list[dict[str, Any]] = []
    for p in startup_paths():
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        entry: dict[str, Any] = {"path": str(p), "ok": True, "output": "", "error": None}
        t0 = time.time()
        try:
            if p.suffix.lower() == ".py":
                from ..scripting import run_script
                out = run_script(session, p.read_text(encoding="utf-8"))
                entry["output"] = out.get("stdout", "") or ""
                if out.get("error"):
                    raise RuntimeError(str(out["error"]).rstrip())
            else:
                from ..ij_compat import IJ
                entry["output"] = IJ(session).macro(p.read_text(encoding="utf-8")) or ""
        except BaseException as e:  # noqa: BLE001 – a startup script must never take the server down
            msg = str(e).rstrip() or type(e).__name__
            entry["ok"] = False
            entry["error"] = msg
            session.add_log(f"Startup script {p.name} failed: {msg}", level="error")
        else:
            session.add_log(f"Startup script {p.name} ran ({(time.time() - t0) * 1000:.0f} ms)")
            if entry["output"].strip():
                session.add_log(entry["output"].rstrip())
        results.append(entry)
    return results


# ============================================================================ wrappers & templates
_WRAPPER_MACRO = '''"""__NAME__ – generated by Lumen (Plugins ▸ Macros ▸ Install): runs the ImageJ macro "__FILE__" next to this file.

Edit the .ijm file freely – it is re-read every time the command runs.  Delete both files to uninstall.
"""
from pathlib import Path

from lumen.core.registry import op

MACRO_FILE = Path(__file__).with_name("__FILE__")


@op("__OP_ID__", "__NAME__", "__MENU__", kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    description="ImageJ macro __FILE__ (installed macro)", keywords=["macro", "ijm", "installed"])
def __SLUG__(ctx):
    from lumen.core.ops.plugins_ops import run_macro_file
    return run_macro_file(ctx.session, MACRO_FILE, roi=ctx.roi)
'''

_WRAPPER_SCRIPT = '''"""__NAME__ – generated by Lumen (Plugins ▸ Macros ▸ Install): runs the Lumen Python script "__FILE__".

The script lives in <lumen home>/macros and is re-read every time the command runs.  Delete both files to uninstall.
"""
from pathlib import Path

from lumen.core.registry import op

SCRIPT_FILE = Path(__file__).parent.parent / "macros" / "__FILE__"
if not SCRIPT_FILE.is_file():
    SCRIPT_FILE = Path(r"__ABS__")


@op("__OP_ID__", "__NAME__", "__MENU__", kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    description="Lumen Python script __FILE__ (installed script)", keywords=["macro", "script", "installed"])
def __SLUG__(ctx):
    from lumen.core.ops.plugins_ops import run_file
    return run_file(ctx.session, SCRIPT_FILE, roi=ctx.roi)
'''

_TEMPLATE_COMMAND = '''"""__NAME__ – a Lumen command plugin (Plugins ▸ Utilities ▸ New Plugin Template).

Every ``@op`` function in this file registers one command.  Its ``Param`` list drives the dialog with live
preview, the macro recorder line and the scripting call ``lumen.run("__OP_ID__", offset=10)`` – nothing else
to declare.  After editing: Plugins ▸ Utilities ▸ Reload Plugins (``plugins.reload``) replaces the command.

kind="image" gives full control (``fn(ctx, **params) -> OpResult``): return ``data=`` to replace the pixels
(undoable), ``new_images=[...]`` to open results, or ``results=``/``table=``/``plot=`` for analysis output.
``ctx.img`` is the active LumenImage – ``.data`` is a (T, Z, C, Y, X) numpy array, ``ctx.mask`` the area selection.
"""
import numpy as np

from lumen.core.image import LumenImage
from lumen.core.registry import OpResult, Param, cast_to, op


@op("__OP_ID__", "__NAME__", "Plugins", kind="image",
    description="Add a constant to every pixel – as a new image or in place.",
    params=[Param("offset", "float", 10.0, min=-65535, max=65535, step=1, help="Value added to every pixel"),
            Param("as_new", "bool", True, label="Create a new image")],
    keywords=["template", "example"])
def __SLUG__(ctx, offset=10.0, as_new=True):
    img = ctx.img
    data = cast_to(img.data.astype(np.float32) + offset, img.data.dtype)   # keep the image's bit depth
    if as_new:
        return OpResult(new_images=[LumenImage(data, name=f"{img.name}+{offset:g}", rgb=img.rgb)], message="Created a new image")
    return OpResult(data=data, message=f"Added {offset:g} to {img.name}")
'''

_TEMPLATE_FILTER = '''"""__NAME__ – a Lumen plane filter plugin (Plugins ▸ Utilities ▸ New Plugin Template).

kind="plane" is the simplest kind: ``fn(plane, ctx, **params) -> plane``.  Lumen calls it for every plane
selected by the "apply to" scope (plane / slice / stack), restricts the change to the area selection, keeps the
image's bit depth, records the undo step and drives the live preview.  Reload with Plugins ▸ Utilities ▸ Reload Plugins.
"""
import numpy as np
from scipy import ndimage as ndi

from lumen.core.registry import Param, cast_to, op


@op("__OP_ID__", "__NAME__", "Plugins", kind="plane",
    description="Gaussian smoothing followed by a contrast stretch of each plane.",
    params=[Param("sigma", "float", 2.0, min=0, max=50, step=0.5, unit="px", help="Gaussian radius (0 = none)"),
            Param("stretch", "bool", True, label="Stretch contrast to the full range")],
    keywords=["template", "example", "smooth"])
def __SLUG__(plane, ctx, sigma=2.0, stretch=True):
    out = plane.astype(np.float32)
    if sigma > 0:
        out = ndi.gaussian_filter(out, sigma)
    if stretch:
        lo, hi = float(out.min()), float(out.max())
        full = 255.0 if plane.dtype == np.uint8 else (65535.0 if plane.dtype == np.uint16 else 1.0)
        if hi > lo:
            out = (out - lo) / (hi - lo) * full
    return cast_to(out, plane.dtype)   # same shape as ``plane``; returning float32 instead would convert the image to 32-bit
'''

_TEMPLATE_DIALOG = '''"""__NAME__ – a Lumen script with a dialog (Plugins ▸ Utilities ▸ New Plugin Template).

The ``Param`` list IS the dialog: string, float (slider), int, choice (drop-down), bool (checkbox) and text
(multi-line) parameters are rendered automatically, remembered by the recorder and accepted by
``lumen.run("__OP_ID__", threshold=100, method="Median")``.  kind="analysis" never changes pixels (no undo
snapshot) and returns rows for the Results table plus a Log line.  Reload with Plugins ▸ Utilities ▸ Reload Plugins.
"""
import numpy as np

from lumen.core.registry import OpResult, Param, op


@op("__OP_ID__", "__NAME__", "Plugins", kind="analysis",
    description="Shows every dialog parameter type and reports statistics of the current image.",
    params=[Param("label", "string", "sample", help="Free text"),
            Param("threshold", "float", 128.0, min=0, max=65535, step=1, help="Numbers get a slider"),
            Param("iterations", "int", 3, min=1, max=100),
            Param("method", "choice", "Mean", choices=["Mean", "Median", "Max"]),
            Param("per_plane", "bool", False, label="Report every plane"),
            Param("notes", "text", "", help="Multi-line text", advanced=True)],
    keywords=["template", "example", "dialog"])
def __SLUG__(ctx, label="sample", threshold=128.0, iterations=3, method="Mean", per_plane=False, notes=""):
    img = ctx.img
    planes = list(img.iter_planes()) if per_plane else [(img.display.t, img.display.z, img.display.c)]
    rows = []
    for (t, z, c) in planes:
        p = img.data[t, z, c].astype(np.float32)
        if ctx.mask is not None:          # restrict to the area selection, if there is one
            p = p[ctx.mask]
        stat = {"Mean": float(p.mean()), "Median": float(np.median(p)), "Max": float(p.max())}[method]
        rows.append({"Label": label, "T": t, "Z": z, "C": c, method: stat, "Above threshold (%)": 100.0 * float((p > threshold).mean())})
    note = f" – {notes}" if notes else ""
    return OpResult(results=rows, log=f"__NAME__: {len(rows)} row(s), {iterations} iteration(s){note}",
                    message=f"{len(rows)} row(s) added to the Results table")
'''

_TEMPLATE_TOOL = '''"""__NAME__ – a Lumen plugin tool (Plugins ▸ Utilities ▸ New Plugin Template ▸ Plugin Tool).

``tool=True`` puts the command on the tool rail after the built-in tools (``tool_icon``: an emoji or 1–2 characters,
``tool_cursor``: a CSS cursor name such as crosshair, cell, pointer, copy).  While the tool is selected, every click
or drag on the image runs the command with the interaction added to its parameters:

    x, y      image pixel of the press                 event    "click", "drag-start" or "drag-end"
    x2, y2    release position of a drag (else None)   button   0 left, 1 middle, 2 right
    shift / alt / ctrl   modifier keys (True / False)

The ``Param`` list is the tool-options bar above the canvas.  Return an OpResult like any command: ``results=`` rows,
``overlay=`` the image's new overlay list, ``roi=`` a selection, ``message=`` a status toast; kind="image" tools may
return ``data=`` to change pixels (undoable).  Reload with Plugins ▸ Utilities ▸ Reload Plugins – the tool rail follows.
Every interaction is recorded as ``lumen.run("__OP_ID__", x=…, y=…, event="click")`` so a macro can replay the clicks.
"""
import math

from lumen.core.registry import OpResult, Param, op


@op("__OP_ID__", "__NAME__", "Plugins/Tools", kind="analysis", preview=False, roi_aware=False, scope_param=False,
    tool=True, tool_icon="✚", tool_cursor="crosshair",
    description="Click to mark a spot (overlay circle + Results row); a drag marks the midpoint of the dragged segment; ⌥-click removes the nearest mark.",
    params=[Param("radius", "float", 8.0, min=1, max=200, step=1, unit="px", help="Marker radius in image pixels"),
            Param("color", "color", "#4ade80")],
    keywords=["template", "example", "tool"])
def __SLUG__(ctx, radius=8.0, color="#4ade80", x=0.0, y=0.0, x2=None, y2=None, button=0, shift=False, alt=False, ctrl=False, event="click"):
    img = ctx.img
    if event == "drag-start":
        return OpResult()                                   # nothing to do until the drag ends
    if event == "drag-end" and x2 is not None and y2 is not None:
        x, y = (x + x2) / 2, (y + y2) / 2                  # a drag marks the midpoint of the dragged segment
    marks = [o for o in img.overlay if str(o.get("id", "")).startswith("__SLUG__-")]
    others = [o for o in img.overlay if o not in marks]
    if alt:                                                 # ⌥-click: remove the nearest mark
        if not marks:
            return OpResult(message="No marks to remove")
        nearest = min(marks, key=lambda o: math.hypot(o["x"] + o["w"] / 2 - x, o["y"] + o["h"] / 2 - y))
        marks.remove(nearest)
        return OpResult(overlay=others + marks, message=f"Removed {nearest['name']} – {len(marks)} mark(s) left")
    n = len(marks) + 1
    d = img.display
    px, py = int(x), int(y)
    value = float(img.data[d.t, d.z, d.c, py, px]) if 0 <= px < img.width and 0 <= py < img.height else None
    mark = {"type": "oval", "id": f"__SLUG__-{n}", "name": f"Mark {n}", "x": x - radius, "y": y - radius, "w": 2 * radius, "h": 2 * radius,
            "color": color, "stroke_width": 1.5, "label": str(n), "position": {"t": d.t, "z": d.z, "c": d.c}}
    row = {"Label": f"Mark {n}", "X": round(x, 2), "Y": round(y, 2), "Value": value, "T": d.t, "Z": d.z, "Button": button}
    return OpResult(overlay=others + marks + [mark], results=[row], message=f"__NAME__: {n} mark(s)")
'''

_TEMPLATE_MACRO = '''// __NAME__ – an ImageJ macro installed as the Lumen command "Plugins ▸ Macros ▸ __NAME__"
// (Plugins ▸ Utilities ▸ New Plugin Template).  Edit this file freely: the command re-reads it every time it runs.
// Lumen's macro layer supports run("…"), selectWindow, makeRectangle, setAutoThreshold, roiManager, nResults, getResult,
// loops, functions, Array.* / String.* / Math.* …  Plugins ▸ Scripting ▸ ImageJ Command List shows every run() command.
if (nImages == 0) run("Blobs (25K)");
title = getTitle();
run("Duplicate...", "title=" + title + "-mask");
run("Gaussian Blur...", "sigma=2");
setAutoThreshold("Otsu dark");
run("Convert to Mask");
run("Analyze Particles...", "size=20-Infinity display clear");
print("__NAME__: " + nResults + " particles in " + title);
'''

_TEMPLATE_MACRO_TOOL = '''// __NAME__ – an ImageJ macro tool set (Plugins ▸ Utilities ▸ New Plugin Template ▸ Macro Tool, ImageJ's Plugins ▸ New ▸ Macro Tool).
// Every `macro "Name Tool - <icon>" { ... }` block of this file is a tool on the tool rail: a click or drag on the image runs
// its body with getCursorLoc(x, y, z, flags) giving the press (flags: 16 left button, 4 right, 1 shift, 2 ctrl, 8 alt) and,
// when the macro loops `while (flags&16!=0)`, the release of a drag.  The icon string is ImageJ's (C = colour rgb, O oval,
// F filled rect, L line, T text ...): its colour and first letters / shape become the rail icon.
// "Name Tool Options" runs with the tool's "Tool Options" checkbox (ImageJ: double-click on the tool); `var` globals keep
// their values between clicks.  Edit this file freely – it is re-read when it changes; Plugins ▸ Utilities ▸ Compile and
// Run / Update Menus rescans the tool rail.
var radius = 12;
var count = 0;

macro "__NAME__ Tool - C0c0O11cc" {
    leftButton = 16; rightButton = 4; shiftKey = 1; altKey = 8;
    getCursorLoc(x, y, z, flags);
    x0 = x; y0 = y;
    while (flags&leftButton != 0) {       // a drag: follow the cursor to the release point
        getCursorLoc(x, y, z, flags);
        wait(10);
    }
    if (isKeyDown("alt")) {               // alt-click: report the pixel only
        showStatus("__NAME__: (" + x0 + ", " + y0 + ") = " + getPixel(x0, y0));
        exit;
    }
    r = radius;
    if (x != x0 || y != y0)               // the drag length sets the radius
        r = maxOf(1, round(sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0))));
    count++;
    setColor(0, 200, 0);
    drawOval(x0 - r, y0 - r, 2 * r, 2 * r);
    setResult("Label", nResults, "__NAME__ " + count);
    setResult("X", nResults - 1, x0);
    setResult("Y", nResults - 1, y0);
    setResult("Radius", nResults - 1, r);
    updateResults();
    showStatus("__NAME__: circle " + count + " at (" + x0 + ", " + y0 + ") r=" + r);
}

macro "__NAME__ Tool Options" {
    Dialog.create("__NAME__ Options");
    Dialog.addNumber("Default radius (px)", radius);
    Dialog.show();
    radius = Dialog.getNumber();
    count = 0;
    print("__NAME__: radius " + radius + ", counter reset");
}
'''

_TEMPLATE_FRAME = '''"""__NAME__ – a Lumen plugin frame (Plugins ▸ Utilities ▸ New Plugin Template ▸ Plugin Frame, ImageJ's Plugins ▸ New ▸ Plugin Frame).

ImageJ's PlugInFrame is a plugin with its own always-visible window.  In Lumen a frame is a command declared with
``frame=True`` (or ``kind="frame"``): its output – a table, a plot, Log text, a message – is shown in the inspector
section "Plugin Frames" as its own docked sub-panel, with a Run / Refresh button and the command's Params as a compact
form, instead of in the shared Table / Plot sections.  Everything else is an ordinary analysis command: it runs from
the palette, menus, macros (``lumen.run("__OP_ID__", decimals=1)``) and the Python console too.
Reload with Plugins ▸ Utilities ▸ Reload Plugins.
"""
import numpy as np

from lumen.core.registry import OpResult, Param, op


@op("__OP_ID__", "__NAME__", "Plugins/Frames", kind="frame", requires_image=False,
    description="Lists every open image with its size, type and the intensity statistics of its current plane (a plugin frame: "
                "shown in the Plugin Frames inspector section).",
    params=[Param("decimals", "int", 2, min=0, max=6),
            Param("plot_means", "bool", True, label="Plot the mean of every image")],
    keywords=["template", "example", "frame", "statistics"])
def __SLUG__(ctx, decimals=2, plot_means=True):
    rows = []
    for img in ctx.session.images.values():
        d = img.display
        plane = img.data[d.t, d.z, min(d.c, img.nc - 1)].astype(np.float64)
        rows.append([img.name, f"{img.width}×{img.height}", f"{img.nz}/{img.nc}/{img.nt}", str(img.dtype), round(float(plane.mean()), decimals),
                     round(float(plane.std()), decimals), round(float(plane.min()), decimals), round(float(plane.max()), decimals)])
    table = {"name": "__NAME__", "columns": ["Image", "Size", "Z/C/T", "Type", "Mean", "Std", "Min", "Max"], "rows": rows}
    plot = None
    if plot_means and rows:
        plot = {"title": "__NAME__", "x": list(range(len(rows))), "series": [{"name": "Mean", "values": [r[4] for r in rows]}],
                "x_label": "image", "y_label": "mean"}
    return OpResult(table=table, plot=plot, message=f"{len(rows)} open image(s)")
'''

_STARTUP_PY_TEMPLATE = '''# Lumen startup script – runs once when `lumen` starts (Plugins ▸ Macros ▸ Edit Startup Script).
# The `lumen` scripting API, numpy (np), scipy and skimage are available, as are the ImageJ macro helpers
# (run, selectWindow, makeRectangle, nResults, getResult …) and `IJ`.  Errors are written to the Log.
# Examples:
#   lumen.session.settings["foreground"] = 255
#   lumen.run("file.sample", name="Blobs")
#   lumen.log("Hello from startup.py")
'''

_STARTUP_IJM_TEMPLATE = '''// Lumen startup macro – ImageJ macro code run once when `lumen` starts (after startup.py).
// Example:
//   print("Hello from startup.ijm");
//   run("Blobs (25K)");
'''


def _fill(template: str, **vals: str) -> str:
    for k, v in vals.items():
        template = template.replace(f"__{k.upper()}__", v)
    return template


def _unique_op_id(prefix: str, slug: str, owner: Path) -> str:
    """``<prefix>.<slug>`` unless another file already registered it – then suffix a counter."""
    from . import PLUGIN_OPS
    mine = set(PLUGIN_OPS.get(str(owner), []))
    oid, k = f"{prefix}.{slug}", 1
    while oid in REGISTRY.ops and oid not in mine:
        k += 1
        oid = f"{prefix}.{slug}_{k}"
    return oid


def _load_wrapper(session: Any, wrapper: Path) -> list[str]:
    from . import load_plugin_file
    try:
        ids = load_plugin_file(wrapper)
    except Exception as e:
        raise ValueError(f"{wrapper.name} failed to load: {e}") from e
    _ops_changed(session)
    return ids


def install_macro_file(session: Any, src: Path, name: str = "", menu: str = MACROS, overwrite: bool = True) -> tuple[Path, Path, list[str]]:
    """Copy an ImageJ macro into the plugins folder and register it as a command through a wrapper module.
    Returns (installed macro, wrapper module, op ids).

    A *tool macro set* (``macro "Name Tool - icon" { … }`` blocks) gets no wrapper: its tools are registered on the tool
    rail and its other named macros as commands (ImageJ's Install… semantics, see ``lumen.core.ops.macro_tools``); the
    second element is then the macro file itself."""
    from . import macro_tools
    pdir = plugins_dir()
    dest = pdir / (src.stem + ".ijm")
    wrapper = pdir / f"{src.stem}_macro.py"
    if dest.exists() and not overwrite and src.resolve() != dest.resolve():
        raise FileExistsError(f"{dest} exists – enable overwrite to replace it")
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    if macro_tools.has_tool_macros(dest.read_text(encoding="utf-8")):
        if wrapper.exists():  # an earlier plain install of the same file: the wrapper command would run the set inline
            from . import PLUGIN_OPS
            for oid in PLUGIN_OPS.pop(str(wrapper), []):
                REGISTRY.ops.pop(oid, None)
            wrapper.unlink()
        ids = macro_tools.register_file(dest, session)
        return dest, dest, ids
    name = str(name or "").strip() or src.stem.replace("_", " ")
    slug = _slug(src.stem)
    oid = _unique_op_id("macros", slug, wrapper)
    wrapper.write_text(_fill(_WRAPPER_MACRO, name=name.replace('"', "'"), file=dest.name, op_id=oid, menu=menu or MACROS, slug=slug), encoding="utf-8")
    return dest, wrapper, _load_wrapper(session, wrapper)


def install_script_file(session: Any, src: Path, name: str = "", menu: str = MACROS, overwrite: bool = True) -> tuple[Path, Path, list[str]]:
    """Copy a plain Lumen Python script into <home>/macros and register it as a command through a wrapper module."""
    mdir = macros_dir()
    dest = mdir / src.name
    wrapper = plugins_dir() / f"{src.stem}_script.py"
    if dest.exists() and not overwrite and src.resolve() != dest.resolve():
        raise FileExistsError(f"{dest} exists – enable overwrite to replace it")
    if src.resolve() != dest.resolve():
        shutil.copyfile(src, dest)
    name = str(name or "").strip() or src.stem.replace("_", " ")
    slug = _slug(src.stem)
    oid = _unique_op_id("macros", slug, wrapper)
    wrapper.write_text(_fill(_WRAPPER_SCRIPT, name=name.replace('"', "'"), file=dest.name, abs=str(dest), op_id=oid, menu=menu or MACROS, slug=slug), encoding="utf-8")
    return dest, wrapper, _load_wrapper(session, wrapper)


def _is_plugin_module(text: str) -> bool:
    return "@op(" in text or "lumen.core.registry" in text


# ============================================================================ Plugins ▸ Macros
@op("macros.edit_startup", "Edit Startup Script…", MACROS, **_NOIMG, keywords=["startup", "autostart", "init", "options"],
    params=[Param("kind", "choice", "Python (startup.py)", choices=["Python (startup.py)", "ImageJ macro (startup.ijm)"]),
            Param("create", "bool", True, label="Create a documented file if it is missing")],
    description="Show the startup script that runs when Lumen starts (<lumen home>/startup.py with the Python API, then startup.ijm as an ImageJ macro; "
                "$LUMEN_HOME overrides ~/.lumen). The path is reported so you can edit the file; its contents are listed line by line.")
def edit_startup(ctx, kind="Python (startup.py)", create=True):
    is_py = kind.startswith("Python")
    p = startup_paths()[0 if is_py else 1]
    if not p.exists() and create:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_STARTUP_PY_TEMPLATE if is_py else _STARTUP_IJM_TEMPLATE, encoding="utf-8")
    text = p.read_text(encoding="utf-8") if p.exists() else ""
    code = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#" if is_py else "//")]
    msg = f"Startup script: {p}" + ("" if p.exists() else " (does not exist)") + f" – {len(code)} code line(s); it runs once when lumen starts"
    return OpResult(table=_lines_table(p.name, text), message=msg, log=f"{msg}\n{text}" if text else msg)


@op("macros.install", "Install…", MACROS, **_NOIMG, keywords=["install", "macro", "plugin", "ijm", "script"],
    params=[Param("path", "string", "", help="A .py plugin module (its @op commands register), a plain .py Lumen script or an .ijm ImageJ macro"),
            Param("name", "string", "", help="Command name (default: the file name)"),
            Param("menu", "string", MACROS, help="Menu for wrapped macros / scripts"),
            Param("overwrite", "bool", True)],
    description="Copy a plugin, macro or script into the plugins folder and register it as a command right away: .py files with @op commands become "
                "plugins; .ijm macros and plain Python scripts are wrapped into a command under Plugins ▸ Macros (ImageJ's Plugins ▸ Macros ▸ Install…).")
def install(ctx, path="", name="", menu=MACROS, overwrite=True):
    s = ctx.session
    src = Path(str(path or "")).expanduser()
    if not str(path).strip():
        raise ValueError("Choose a .py or .ijm file to install")
    if not src.is_file():
        raise FileNotFoundError(f"No such file: {src}")
    ext = src.suffix.lower()
    if ext in MACRO_EXTS:
        dest, wrapper, ids = install_macro_file(s, src, name, menu, overwrite)
        what = "ImageJ macro"
    elif ext == ".py":
        text = src.read_text(encoding="utf-8")
        if _is_plugin_module(text):
            dest = plugins_dir() / src.name
            if dest.exists() and not overwrite and src.resolve() != dest.resolve():
                raise FileExistsError(f"{dest} exists – enable overwrite to replace it")
            if src.resolve() != dest.resolve():
                shutil.copyfile(src, dest)
            wrapper = dest
            ids = _load_wrapper(s, dest)
            what = "plugin module"
        else:
            dest, wrapper, ids = install_script_file(s, src, name, menu, overwrite)
            what = "Python script"
    else:
        raise ValueError(f"{src.name}: only .py and .ijm files can be installed")
    rows = [[oid, REGISTRY.ops[oid].name, REGISTRY.ops[oid].menu or REGISTRY.ops[oid].category] for oid in ids if oid in REGISTRY.ops]
    msg = f"Installed {what} {src.name} → {dest}" + (f"; command(s): {', '.join(r[1] for r in rows)}" if rows else "; no commands registered")
    return OpResult(table={"name": "Installed", "columns": ["Id", "Name", "Menu"], "rows": rows}, message=msg, log=msg)


@op("macros.run_file", "Run Macro File…", MACROS, **_NOIMG, keywords=["run", "macro", "script", "file", "ijm", "py"],
    params=[Param("path", "string", "", help="A .py Lumen Python script or an .ijm ImageJ macro")],
    description="Run a Python script (the lumen API, numpy, scipy, skimage and the ImageJ helpers are available) or an ImageJ macro file now.")
def run_macro_file_op(ctx, path=""):
    if not str(path).strip():
        raise ValueError("Choose a .py or .ijm file to run")
    return run_file(ctx.session, path, roi=ctx.roi)


@op("macros.record_to_file", "Save Recorded Macro…", MACROS, **_NOIMG, keywords=["recorder", "record", "save macro", "script"],
    params=[Param("path", "string", "", help="Destination .py (default: <lumen home>/macros/Recorded_<date>.py)"),
            Param("title", "string", "", help="Comment written at the top of the file"),
            Param("clear_recorder", "bool", False, label="Clear the recorder afterwards", help="Only this command's own line stays recorded")],
    description="Write the lines recorded so far (the Recorder panel) to a Python script that replays them with lumen.run(...). "
                "Run it with Run Macro File…, or Install… it as a command.")
def record_to_file(ctx, path="", title="", clear_recorder=False):
    s = ctx.session
    lines = [ln for ln in s.macro if ln.strip() and '"macros.record_to_file"' not in ln]  # earlier saves are not part of the workflow
    if not lines:
        raise ValueError("The recorder is empty – run some commands first")
    p = Path(str(path)).expanduser() if str(path).strip() else macros_dir() / f"Recorded_{time.strftime('%Y%m%d_%H%M%S')}.py"
    if p.suffix.lower() != ".py":
        p = p.with_suffix(".py")
    p.parent.mkdir(parents=True, exist_ok=True)
    head = f"# {title.strip()}\n" if str(title).strip() else ""
    head += f"# Recorded by Lumen on {time.strftime('%Y-%m-%d %H:%M')} – run with Plugins ▸ Macros ▸ Run Macro File… or install it as a command.\n"
    text = head + "\n".join(lines) + "\n"
    p.write_text(text, encoding="utf-8")
    if clear_recorder:
        s.macro.clear()
    msg = f"Saved {len(lines)} recorded line(s) to {p}"
    return OpResult(table=_lines_table(p.name, text), message=msg, log=msg)


# ============================================================================ Plugins ▸ Utilities (plugins)
@op("plugins.new_template", "New Plugin Template…", UTIL, **_NOIMG, keywords=["new", "plugin", "template", "create", "develop"],
    params=[Param("kind", "choice", "Python command", choices=TEMPLATE_KINDS),
            Param("name", "string", "My Plugin", help="Command name; the file is named after it"),
            Param("overwrite", "bool", False), Param("load", "bool", True, label="Register the command right away")],
    description="Write a documented plugin template into the plugins folder (ImageJ's Plugins ▸ New ▸ …): a Python command, a plane filter, an ImageJ "
                "macro (installed as a command), a script with an auto-generated dialog, a plugin tool (tool=True: lives on the tool rail and "
                "receives clicks / drags), an ImageJ macro tool set (.ijm with `macro \"Name Tool - icon\"` blocks on the tool rail) or a plugin "
                "frame (frame=True: its output gets a docked sub-panel in the Plugin Frames inspector section). Returns the file's path and contents.")
def new_template(ctx, kind="Python command", name="My Plugin", overwrite=False, load=True):
    from . import macro_tools
    s = ctx.session
    name = str(name or "").strip() or "My Plugin"
    if kind == "Macro Tool":
        name = re.sub(r"\s+tool$", "", name, flags=re.IGNORECASE).strip() or "My"  # the template appends " Tool" to the block names
    slug = _slug(name)
    pdir = plugins_dir()
    if kind == "Macro Tool":
        src = pdir / f"{slug}.ijm"
        if src.exists() and not overwrite:
            raise FileExistsError(f"{src.name} already exists in {pdir} – choose another name or enable overwrite")
        src.write_text(_fill(_TEMPLATE_MACRO_TOOL, name=name.replace('"', "'")), encoding="utf-8")
        ids = macro_tools.register_file(src, s) if load else []
        text, path = src.read_text(encoding="utf-8"), src
    elif kind == "ImageJ macro":
        src = pdir / f"{slug}.ijm"
        wrapper = pdir / f"{slug}_macro.py"
        if (src.exists() or wrapper.exists()) and not overwrite:
            raise FileExistsError(f"{src.name} already exists in {pdir} – choose another name or enable overwrite")
        src.write_text(_fill(_TEMPLATE_MACRO, name=name), encoding="utf-8")
        ids: list[str] = []
        if load:
            _, _, ids = install_macro_file(s, src, name, MACROS, overwrite=True)
        else:
            oid = _unique_op_id("macros", slug, wrapper)
            wrapper.write_text(_fill(_WRAPPER_MACRO, name=name.replace('"', "'"), file=src.name, op_id=oid, menu=MACROS, slug=slug), encoding="utf-8")
        text, path = src.read_text(encoding="utf-8"), src
    else:
        path = pdir / f"{slug}.py"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path.name} already exists in {pdir} – choose another name or enable overwrite")
        oid = _unique_op_id("user", slug, path)
        tpl = {"Python command": _TEMPLATE_COMMAND, "Python filter (plane)": _TEMPLATE_FILTER, "Script with dialog": _TEMPLATE_DIALOG,
               "Plugin Tool": _TEMPLATE_TOOL, "Plugin Frame": _TEMPLATE_FRAME}[kind]
        text = _fill(tpl, name=name.replace('"', "'"), op_id=oid, slug=slug)
        path.write_text(text, encoding="utf-8")
        ids = _load_wrapper(s, path) if load else []
    cmds = ", ".join(f"{REGISTRY.ops[i].name} ({i})" for i in ids if i in REGISTRY.ops)
    msg = f"{kind} template written to {path}" + (f" – command registered: {cmds}" if cmds else " – reload plugins to register it")
    return OpResult(table=_lines_table(path.name, text), message=msg, log=msg)


def _reveal(path: Path) -> str:
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        elif os.name == "nt":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return "opened in the file browser"
    except Exception as e:
        return f"could not open the file browser ({e})"


def _file_rows(dirs: list[Path]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for d in dirs:
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.name.startswith(".") or not f.is_file() or f.suffix.lower() not in (".py", ".ijm", ".txt"):
                continue
            st = f.stat()
            if f.suffix.lower() == ".py":
                cmds, status = _plugin_file_rows(f)
                ftype = "plugin"
            else:
                cmds, status, ftype = _macro_file_rows(f)
            rows.append([f.name, str(d), ftype, st.st_size, time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)), cmds, status])
    return rows


def _macro_file_rows(f: Path) -> tuple[str, str, str]:
    """(commands, status, type) of an .ijm / .txt macro file: wrapped command, tool macro set, or not installed."""
    from .macro_tools import MACRO_TOOL_ERRORS, MACRO_TOOL_FILES
    wrapper = f.with_name(f.stem + "_macro.py")
    if wrapper.exists():
        cmds, status = _plugin_file_rows(wrapper)
        return cmds, status, "ImageJ macro"
    if str(f) in MACRO_TOOL_ERRORS:
        return "", f"error: {MACRO_TOOL_ERRORS[str(f)]}", "ImageJ macro"
    if MACRO_TOOL_FILES.get(str(f)):
        ids = [oid for oid in MACRO_TOOL_FILES[str(f)] if oid in REGISTRY.ops]
        n_tools = sum(1 for oid in ids if REGISTRY.ops[oid].tool)
        return ", ".join(ids), f"loaded ({n_tools} tool(s), {len(ids) - n_tools} macro command(s))", "ImageJ macro tool set"
    return "", "macro not installed – Plugins ▸ Macros ▸ Install…", "ImageJ macro"


@op("plugins.open_folder", "Open Plugins Folder", UTIL, **_NOIMG, keywords=["plugins", "folder", "directory", "reveal", "finder", "explorer"],
    params=[Param("reveal", "bool", False, label="Open it in the system file browser")],
    description="Report (and create) the plugins folder <lumen home>/plugins and list the files in it; optionally reveal it in Finder / Explorer.")
def open_folder(ctx, reveal=False):
    from . import plugin_dirs
    d = plugins_dir()
    rows = _file_rows([d])
    msg = f"Plugins folder: {d} ({len(rows)} file(s))"
    extra = [str(x) for x in plugin_dirs()[1:]]
    if extra:
        msg += f"; also $LUMEN_PLUGINS: {os.pathsep.join(extra)}"
    if reveal:
        msg += " – " + _reveal(d)
    return OpResult(table={"name": "Plugins folder", "columns": ["File", "Folder", "Type", "Bytes", "Modified", "Commands", "Status"], "rows": rows},
                    message=msg, log=msg)


@op("plugins.list", "List Plugins", UTIL, **_NOIMG, keywords=["plugins", "list", "installed", "commands"],
    description="Table of every plugin file in <lumen home>/plugins and $LUMEN_PLUGINS with the commands it registered and its load status.")
def list_plugins(ctx):
    from . import plugin_dirs
    dirs = plugin_dirs()
    rows = _file_rows(dirs)
    ids = {c for r in rows for c in r[5].split(", ") if c}
    msg = f"{len(rows)} plugin file(s) in {len([d for d in dirs if d.is_dir()])} folder(s), {len(ids)} plugin command(s)"
    return OpResult(table={"name": "Plugins", "columns": ["File", "Folder", "Type", "Bytes", "Modified", "Commands", "Status"], "rows": rows}, message=msg)


@op("plugins.reload", "Compile and Run / Update Menus", UTIL, **_NOIMG, keywords=["reload", "compile", "update menus", "refresh", "hot reload", "plugins"],
    description="Re-import every plugin file (hot reload): commands registered by plugins are removed and registered again from the current files, "
                "so edited, added or deleted plugins take effect without restarting; the ImageJ tool macro sets (.ijm) are rescanned too. "
                "The menus refetch the command list afterwards.")
def reload_plugins(ctx):
    from . import reload_user_plugins
    from .macro_tools import load_macro_tools
    s = ctx.session
    info = reload_user_plugins()
    tools = load_macro_tools()
    try:
        from .fiji_plugins import register_installed
        register_installed()  # the commands of the Fiji plugin jars in <lumen home>/jars
    except Exception:  # noqa: BLE001
        pass
    rows: list[list[Any]] = []
    for f, ids in info["ops"].items():
        err = info["errors"].get(f)
        rows.append([Path(f).name, str(Path(f).parent), f"error: {err}" if err else "loaded", ", ".join(ids)])
    for f, err in info["errors"].items():
        if f not in info["ops"]:
            rows.append([Path(f).name, str(Path(f).parent), f"error: {err}", ""])
    for f, ids in tools["ops"].items():
        rows.append([Path(f).name, str(Path(f).parent), "loaded (macro tools)", ", ".join(ids)])
    for f, err in tools["errors"].items():
        rows.append([Path(f).name, str(Path(f).parent), f"error: {err}", ""])
    n = _ops_changed(s)
    n_ops = sum(len(v) for v in info["ops"].values())
    n_tools = sum(len(v) for v in tools["ops"].values())
    msg = (f"Reloaded {len(info['files'])} plugin file(s): {n_ops} command(s) now, {len(info['removed'])} replaced/removed"
           + (f", {len(info['errors'])} file(s) failed" if info["errors"] else "")
           + (f"; {n_tools} macro tool command(s) from {len(tools['files'])} .ijm file(s)" if n_tools or tools["removed"] else "")
           + (f", {len(tools['errors'])} macro file(s) failed" if tools["errors"] else "") + f" (ops version {n})")
    return OpResult(table={"name": "Reloaded plugins", "columns": ["File", "Folder", "Status", "Commands"], "rows": rows}, message=msg, log=msg)


# ============================================================================ Plugins ▸ Frames
@op("frames.image_statistics", "Image Statistics Frame", FRAMES, **_FRAME, keywords=["frame", "plugin frame", "statistics", "images", "mean", "std", "window"],
    params=[Param("plane", "choice", "Current plane", choices=["Current plane", "Whole stack"], help="Statistics of the displayed plane or of every pixel"),
            Param("decimals", "int", 2, min=0, max=6),
            Param("use_selection", "bool", False, label="Restrict to the selection", help="Only the active image's selection is known; other images use all pixels")],
    description="A plugin frame (ImageJ's Plugins ▸ New ▸ Plugin Frame): lists every open image with its size, type and the mean, standard deviation, "
                "minimum and maximum of its current plane (or the whole stack). Its table is shown in the Plugin Frames inspector section as a "
                "docked sub-panel with a Refresh button and these parameters as a compact form.")
def image_statistics_frame(ctx, plane="Current plane", decimals=2, use_selection=False):
    from .. import roi as roimod
    s = ctx.session
    dec = int(decimals)
    rows: list[list[Any]] = []
    images = list(s.images.values())
    active = s.active
    mask = None  # the selection belongs to the active image (the op needs no image, so ctx.mask is not available)
    if use_selection and ctx.roi is not None and active is not None and roimod.is_area(ctx.roi):
        mask = roimod.to_mask(ctx.roi, (active.height, active.width))
    for i, img in enumerate(images):
        ctx.check_cancel()
        d = img.display
        if plane == "Whole stack":
            data = img.data
        elif img.rgb:
            data = img.data[d.t, d.z]
        else:
            data = img.data[d.t, d.z, min(d.c, img.nc - 1)]
        if mask is not None and img is active and plane != "Whole stack":
            data = data[..., mask]
        arr = np.asarray(data, dtype=np.float64)
        if arr.size == 0:
            mean = std = lo = hi = float("nan")
        else:
            mean, std, lo, hi = float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())
        rows.append([img.name, f"{img.width}×{img.height}", f"{img.nz}/{img.nc}/{img.nt}", "RGB" if img.rgb else str(img.dtype),
                     round(mean, dec), round(std, dec), round(lo, dec), round(hi, dec), int(arr.size)])
        if len(images) > 1:
            ctx.progress((i + 1) / len(images), img.name)
    table = {"name": "Image Statistics", "columns": ["Image", "Size", "Z/C/T", "Type", "Mean", "Std", "Min", "Max", "Pixels"], "rows": rows}
    return OpResult(table=table, message=f"{len(rows)} open image(s)" + (" – open an image to see its statistics here" if not rows else ""))


# ============================================================================ Plugins ▸ Utilities (DICOM, commands)
def _dicom_source(img: Any) -> Path:
    from ..io import _is_dicom_file, _looks_like_dicom_dir
    cand = img.metadata.get("dicom_path") or img.path
    if not cand:
        raise ValueError("The current image was not opened from a DICOM file")
    p = Path(str(cand))
    if p.is_dir():
        if not _looks_like_dicom_dir(p):
            raise ValueError(f"{p} is not a DICOM folder")
        files = [f for f in sorted(p.iterdir()) if f.is_file() and not f.name.startswith(".") and (f.suffix.lower() in (".dcm", ".dicom", ".ima") or _is_dicom_file(f))]
        if not files:
            raise ValueError(f"No DICOM files in {p}")
        return files[0]
    if p.suffix.lower() in (".dcm", ".dicom", ".ima") or _is_dicom_file(p):
        return p
    raise ValueError(f"{img.name} is not a DICOM image ({p.name})")


def _dicom_value(v: Any) -> str:
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        text = ", ".join(str(x) for x in v)
    else:
        text = str(v)
    return text if len(text) <= 300 else text[:297] + "…"


def _dicom_rows(ds: Any, rows: list[list[Any]], depth: int, include_private: bool) -> None:
    indent = "  " * depth
    for el in ds:
        if el.tag.group == 0x7FE0 or el.tag.group == 0xFFFC:  # pixel data / data set trailing padding
            continue
        if el.tag.is_private and not include_private:
            continue
        tag = f"{el.tag.group:04X},{el.tag.element:04X}"
        name = el.name or ("Private tag" if el.tag.is_private else "Unknown")
        if el.VR == "SQ":
            items = list(el.value or [])
            rows.append([tag, indent + name, "SQ", f"{len(items)} item(s)", el.keyword or ""])
            for i, item in enumerate(items, 1):
                rows.append(["", f"{indent}  Item {i}", "", "", ""])
                _dicom_rows(item, rows, depth + 2, include_private)
        else:
            rows.append([tag, indent + name, str(el.VR), _dicom_value(el.value), el.keyword or ""])


@op("utilities.dicom_metadata", "DICOM Metadata", UTIL, **_IMG, keywords=["dicom", "header", "tags", "metadata", "show info", "medical"],
    params=[Param("filter", "string", "", help="Only tags whose name, keyword, tag number or value contains this text"),
            Param("include_private", "bool", True, label="Include private tags")],
    description="Table of every DICOM header tag (tag, name, VR, value) of the current image, read again from its .dcm file or folder with pydicom "
                "(ImageJ's Image ▸ Show Info for DICOM files).")
def dicom_metadata(ctx, filter="", include_private=True):
    img = ctx.img
    src = _dicom_source(img)
    try:
        import pydicom
    except ImportError as e:
        raise RuntimeError("Reading DICOM headers needs: pip install pydicom") from e
    ds = pydicom.dcmread(str(src), stop_before_pixels=True, force=True)
    rows: list[list[Any]] = []
    meta = getattr(ds, "file_meta", None)
    if meta is not None and len(meta) > 0:
        _dicom_rows(meta, rows, 0, include_private)
    _dicom_rows(ds, rows, 0, include_private)
    q = str(filter or "").strip().lower()
    if q:
        rows = [r for r in rows if any(q in str(x).lower() for x in r)]
    desc = ", ".join(str(getattr(ds, k)) for k in ("Modality", "PatientName", "StudyDescription", "SeriesDescription") if getattr(ds, k, None))
    msg = f"{len(rows)} DICOM tag(s) in {src.name}" + (f" ({desc})" if desc else "") + (f" matching {q!r}" if q else "")
    return OpResult(table={"name": "DICOM", "columns": ["Tag", "Name", "VR", "Value", "Keyword"], "rows": rows}, message=msg)


def _param_text(spec: Any) -> str:
    parts = []
    for p in spec.params:
        t = f"{p.name}: {p.type}"
        if p.default is not None:
            t += f" = {p.default!r}"
        if p.choices:
            t += " [" + "|".join(p.choices) + "]"
        parts.append(t)
    return "; ".join(parts)


@op("utilities.list_commands", "List Commands", UTIL, **_NOIMG, keywords=["commands", "list", "menu", "reference", "all"],
    params=[Param("menu", "string", "", help="Only commands whose menu contains this text (e.g. Process/Filters)"),
            Param("kind", "choice", "All", choices=["All", "plane", "image", "analysis"])],
    description="Table of every registered command: id, name, menu, kind, shortcut, parameters (name: type = default [choices]) and description.")
def list_commands(ctx, menu="", kind="All"):
    q = str(menu or "").strip().lower()
    specs = sorted(REGISTRY.ops.values(), key=lambda sp: ((sp.menu or sp.category).lower(), sp.name.lower()))
    rows = [[sp.id, sp.name, sp.menu or sp.category, sp.kind, sp.shortcut or "", _param_text(sp), sp.description]
            for sp in specs if (not q or q in (sp.menu or sp.category).lower()) and (kind == "All" or sp.kind == kind)]
    return OpResult(table={"name": "Commands", "columns": ["Id", "Name", "Menu", "Kind", "Shortcut", "Parameters", "Description"], "rows": rows},
                    message=f"{len(rows)} of {len(REGISTRY.ops)} command(s)")


def _is_subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def fuzzy_score(spec: Any, query: str, search_descriptions: bool = True) -> int:
    """Command Finder score (0 = no match): every query word must match the name, id, keywords, menu or description –
    exactly, as a prefix, as a substring, as a character subsequence (``gausblur``) or by close spelling (``gausian``)."""
    words = [w for w in str(query or "").lower().split() if w]
    if not words:
        return 0
    name = spec.name.lower()
    tokens = [t for t in re.split(r"[\s/…\-_().]+", name) if t]
    compact = re.sub(r"[^0-9a-z]", "", name)
    total = 0
    for w in words:
        if name == w:
            best = 100
        elif name.startswith(w):
            best = 85
        elif any(t.startswith(w) for t in tokens):
            best = 70
        elif w in name:
            best = 55
        elif w in spec.id.lower():
            best = 45
        elif any(w in k.lower() for k in spec.keywords):
            best = 40
        elif w in (spec.menu or spec.category).lower():
            best = 30
        else:
            best = 0
            if len(w) >= 3 and _is_subsequence(re.sub(r"[^0-9a-z]", "", w), compact):
                best = 25 + min(10, int(10 * len(w) / max(len(compact), 1)))
            ratio = max((difflib.SequenceMatcher(None, w, t).ratio() for t in tokens if abs(len(t) - len(w)) <= 3), default=0.0)
            if ratio >= 0.75:
                best = max(best, int(20 + 30 * ratio))
            if search_descriptions and w in spec.description.lower():
                best = max(best, 15)
        if best == 0:
            return 0
        total += best + (5 * len(w)) // max(len(name), 1)  # tie-break: the shorter (closer) name wins
    return total


@op("utilities.command_finder", "Command Finder…", UTIL, **_NOIMG, keywords=["find", "search", "command finder", "palette", "fuzzy", "where is"],
    params=[Param("query", "string", "", help="Words (or parts of words, typos allowed) to look for"),
            Param("max_results", "int", 25, min=1, max=500),
            Param("search_descriptions", "bool", True, label="Also search descriptions")],
    description="Fuzzy search over every command (ImageJ's Plugins ▸ Utilities ▸ Find Commands…): name, id, keywords, menu and description; "
                "partial words, character subsequences and misspellings match. Lists id, menu, shortcut and description.")
def command_finder(ctx, query="", max_results=25, search_descriptions=True):
    q = str(query or "").strip()
    if not q:
        raise ValueError("Type one or more words to search for")
    hits = sorted(((fuzzy_score(sp, q, search_descriptions), sp) for sp in REGISTRY.ops.values()), key=lambda x: (-x[0], x[1].name.lower()))
    hits = [(sc, sp) for sc, sp in hits if sc > 0][:int(max_results)]
    rows = [[sp.id, sp.name, sp.menu or sp.category, sp.shortcut or "", sc, sp.description] for sc, sp in hits]
    return OpResult(table={"name": "Command Finder", "columns": ["Id", "Name", "Menu", "Shortcut", "Score", "Description"], "rows": rows},
                    message=f"{len(rows)} command(s) match {q!r}" if rows else f"No command matches {q!r}")
