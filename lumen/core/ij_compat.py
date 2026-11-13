"""ImageJ-1 compatibility layer.

* ``parse_options("sigma=2 stack")`` – the ImageJ option-string parser.
* ``IJ`` – ``IJ.run(command, options)`` maps ~400 ImageJ / Fiji command names (File, Edit, Image, Process,
  Analyze, the classic sample images and the common Fiji plugins) to Lumen ops with their ImageJ option keys,
  plus the usual macro functions (``selectWindow``, ``makeRectangle``, ``setAutoThreshold``, ``roiManager``,
  ``nResults``, ``getResult``, ``saveAs`` …) as methods.  ``command_table()`` lists every mapping with its notes.
* ``IJ.macro(text)`` – transpiles ImageJ-1 macro code to Python and executes it with those
  helpers in scope (``IJ.transpile(text)`` shows the generated Python).
* ``IJ.Ext`` – the Bio-Formats Macro Extensions (``run("Bio-Formats Macro Extensions"); Ext.setId(path);
  Ext.getSeriesCount(n); Ext.getSizeX(w); Ext.openImage(title, no) …``) over Lumen's native readers.
"""
from __future__ import annotations

import inspect
import io
import math
import os
import re
import time
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Callable

import numpy as np

from . import roi as roimod
from .image import Calibration, LumenImage
from .luts import LUT_NAMES
from .measure import ALL_MEASUREMENTS, measure_roi
from .registry import REGISTRY, TOOL_PARAMS
from .session import Session
from .threshold import METHODS


# ====================================================================== option strings
class Options(dict):
    """Parsed ImageJ option string. Values are the raw option text (str) or True for bare flags – like ImageJ's
    Macro.getValue(), text is never re-typed (``title=007`` stays "007"); use num()/rng()/flag()/str() to interpret."""

    def flag(self, key: str) -> bool:
        return bool(self.get(key, False)) if not isinstance(self.get(key), str) else True

    def num(self, key: str, default: float = 0.0) -> float:
        """Parse the value as a number on demand (int for integer literals, like the old typed parser)."""
        v = self.get(key)
        if isinstance(v, bool) or v is None:
            return default
        if isinstance(v, (int, float)):
            return v
        s = str(v).strip()
        try:
            return int(s) if re.fullmatch(r"[-+]?\d+", s) else float(s)
        except ValueError:
            return default

    def str(self, key: str, default: str = "") -> str:  # noqa: A003
        v = self.get(key)
        if v is None or v is True:
            return default
        return str(v)

    def rng(self, key: str, default: tuple[float, float] = (0.0, 0.0)) -> tuple[float, float]:
        """'20-Infinity' → (20, inf); '0.50-1.00' → (0.5, 1.0)."""
        v = self.get(key)
        if v is None or v is True:
            return default
        if isinstance(v, (int, float)):
            return float(v), default[1]
        parts = re.split(r"(?<=[\d.yY])\s*-\s*", str(v).strip(), maxsplit=1)
        try:
            lo = float(parts[0])
            hi = float("inf") if len(parts) < 2 or parts[1].lower().startswith("inf") else float(parts[1])
            return lo, hi
        except ValueError:
            return default


_NUM_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")


def parse_options(options: str | None) -> Options:
    """Parse an ImageJ option string into a dict: ``key=value`` pairs (values may be ``[bracketed]`` or
    ``"quoted"`` to contain spaces) and bare words, which become ``True`` flags."""
    out = Options()
    s = (options or "").strip()
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i].isspace():
            i += 1
        if i >= n:
            break
        j = i
        while j < n and not s[j].isspace() and s[j] != "=":
            j += 1
        key = s[i:j]
        if j < n and s[j] == "=":
            j += 1
            if j < n and s[j] == "[":
                depth, k = 1, j + 1
                while k < n and depth:
                    if s[k] == "[":
                        depth += 1
                    elif s[k] == "]":
                        depth -= 1
                    k += 1
                val = s[j + 1:k - 1] if depth == 0 else s[j + 1:k]
                i = k
            elif j < n and s[j] in "\"'":
                q = s[j]
                k = j + 1
                while k < n and s[k] != q:
                    k += 1
                val = s[j + 1:k]
                i = k + 1
            else:
                k = j
                while k < n and not s[k].isspace():
                    k += 1
                val = s[j:k]
                i = k
            if key:
                out[key] = val  # raw text: ImageJ only parses numbers in numeric dialog fields (see Options.num)
        else:
            if key:
                out[key] = True
            i = j
    return out


# ====================================================================== command table
_Handler = Callable[["IJ", Options], Any]
COMMANDS: dict[str, tuple[_Handler, str, str]] = {}


def cmd(*names: str, lumen: str = "", notes: str = ""):
    def deco(fn: _Handler) -> _Handler:
        for n in names:
            COMMANDS[n] = (fn, lumen, notes)
        return fn
    return deco


def command_table() -> list[dict[str, str]]:
    seen: dict[int, list[str]] = {}
    order: list[tuple[_Handler, str, str]] = []
    for name, (fn, lumen, notes) in COMMANDS.items():
        if id(fn) not in seen:
            seen[id(fn)] = []
            order.append((fn, lumen, notes))
        seen[id(fn)].append(name)
    rows = [{"command": " / ".join(f'"{n}"' for n in seen[id(fn)]), "lumen": lumen, "notes": notes} for fn, lumen, notes in order]
    rows.append({"command": '"<LUT name>" (' + ", ".join(LUT_NAMES[:6]) + " …)", "lumen": "color.lut", "notes": "Any Lumen LUT name"})
    return rows


def _color_hex(name: Any, default: str = "#ffff00") -> str:
    from .overlay_render import parse_color
    if name is None or name is True:
        return default
    r, g, b = parse_color(name, None) or (None, None, None)
    if r is None:
        return default
    return f"#{r:02x}{g:02x}{b:02x}"


def _scope(o: Options) -> str:
    return "stack" if o.flag("stack") else "plane"


def _collect_params(o: Options, params: dict[str, Any] | None) -> dict[str, Any]:
    """Op params from an option string: `params` maps op param → option key, a callable(o) or a constant (wrapped in `const`)."""
    p: dict[str, Any] = {}
    for k, src in (params or {}).items():
        if isinstance(src, const):
            p[k] = src.value
        elif callable(src):
            v = src(o)
            if v is not None:
                p[k] = v
        elif src in o:
            p[k] = o[src]
    return p


#: op id → [(ImageJ command names, op param → option reader, stack_flag)] for every `simple()` registration; the
#: recorder → ImageJ-macro export (ij_export.py) inverts these mappings
SIMPLE_MAPPINGS: dict[str, list[tuple[tuple[str, ...], dict[str, Any], bool]]] = {}


def simple(*names: str, lumen: str, params: dict[str, Any] | None = None, notes: str = "", stack_flag: bool = True):
    """Register a command that maps option keys to op params.  `params` maps op param → option key,
    a callable(o) or a constant (wrapped in `const`)."""
    def handler(ij: "IJ", o: Options):
        return ij._run(lumen, _collect_params(o, params), scope=_scope(o) if stack_flag else "stack")
    SIMPLE_MAPPINGS.setdefault(lumen, []).append((tuple(names), dict(params or {}), stack_flag))
    return cmd(*names, lumen=lumen, notes=notes or ", ".join(f"{v}→{k}" for k, v in (params or {}).items() if isinstance(v, str)))(handler)


def _current_planes(img: LumenImage) -> set[tuple[int, int, int]]:
    d = img.display
    return {(d.t, d.z, c) for c in range(img.nc)} if img.rgb else {(d.t, d.z, d.c)}


def transform(*names: str, lumen: str, params: dict[str, Any] | None = None, notes: str = ""):
    """Register an Image ▸ Transform command whose ImageJ filter goes through IJ.setupDialog (Flip, Rotate, Translate):
    without the 'stack' keyword only the current plane is transformed.  The Lumen ops are whole-stack image ops, so the
    command runs them (one undo step) and then puts the untouched planes back.  For Rotate with 'enlarge' the canvas grows
    for every slice (ImageJ's Rotator.enlargeCanvas = Canvas Size on the whole stack) but only the current plane is rotated."""
    def handler(ij: "IJ", o: Options):
        p = _collect_params(o, params)
        img = ij.img
        keep = set(img.iter_planes()) - _current_planes(img)
        if o.flag("stack") or not keep:
            return ij._run(lumen, p)
        before = img.data.copy()
        out = ij._run(lumen, p)
        data = img.data
        H, W = before.shape[-2:]
        nh, nw = data.shape[-2:]
        oy, ox = (nh - H) // 2, (nw - W) // 2
        for (t, z, c) in keep:
            if (nh, nw) == (H, W):
                data[t, z, c] = before[t, z, c]
            else:
                data[t, z, c] = 0
                data[t, z, c, oy:oy + H, ox:ox + W] = before[t, z, c]
        img.replace_data(data, rgb=img.rgb)  # the undo snapshot run_op took still holds the untouched stack
        return out
    return cmd(*names, lumen=lumen, notes=notes)(handler)


class const:
    def __init__(self, value: Any) -> None:
        self.value = value


def _keyed(fn, key: str, kind: str):
    # the option key and kind are kept on the reader so the recorder → ImageJ-macro export (ij_export.py) can invert it
    fn.key = key
    fn.kind = kind
    return fn


def flag(key: str):
    return _keyed(lambda o: o.flag(key), key, "flag")


def num(key: str, default: float | None = None):
    return _keyed(lambda o: (o.num(key, default) if (key in o or default is not None) else None), key, "num")


def text(key: str, default: str | None = None):
    return _keyed(lambda o: (o.str(key, default) if (key in o or default is not None) else None), key, "text")


# ---- Process ▸ Filters
def _gauss_sigma(o: Options) -> float:
    # ImageJ GaussianBlur.showDialog: legacy macros pass radius=, converted as sigma = radius / 2.5
    if "sigma" not in o and "radius" in o:
        return o.num("radius", 5.0) / 2.5
    return o.num("sigma", 2.0)


simple("Gaussian Blur...", lumen="filters.gaussian", params={"sigma": _keyed(_gauss_sigma, "sigma", "num"), "scaled_units": flag("scaled")}, notes="sigma (or legacy radius/2.5) → sigma; 'scaled' → sigma in calibrated units; 'stack' → all planes")
simple("Median...", lumen="filters.median", params={"radius": num("radius", 2.0)})
simple("Mean...", lumen="filters.mean", params={"radius": num("radius", 2.0)})
simple("Minimum...", lumen="filters.minimum", params={"radius": num("radius", 2.0)})
simple("Maximum...", lumen="filters.maximum", params={"radius": num("radius", 2.0)})
simple("Variance...", lumen="filters.variance", params={"radius": num("radius", 2.0)})
simple("Unsharp Mask...", lumen="filters.unsharp", params={"radius": num("radius", 1.0), "weight": num("mask", 0.6)}, notes="radius → radius, mask → weight")
simple("Convolve...", lumen="filters.convolve", params={"kernel": text("text1", ""), "normalize": flag("normalize")}, notes="text1 → kernel")
simple("Top Hat...", lumen="filters.tophat", params={"radius": num("radius", 15.0), "light_background": flag("light"), "dont_subtract": flag("don't")})
simple("Gaussian Blur 3D...", lumen="filters.gaussian3d", params={"sigma_xy": num("x", 2.0), "sigma_z": num("z", 1.0)}, stack_flag=False)
for _n, _f in (("Median 3D...", "Median"), ("Mean 3D...", "Mean"), ("Minimum 3D...", "Minimum"), ("Maximum 3D...", "Maximum"), ("Variance 3D...", "Variance")):
    simple(_n, lumen="filters.rank3d", params={"filter": const(_f), "radius_xy": num("x", 2.0), "radius_z": num("z", 1.0)}, stack_flag=False, notes="x → radius_xy, z → radius_z")
simple("Smooth", lumen="process.smooth")
simple("Sharpen", lumen="process.sharpen")
simple("Find Edges", lumen="process.find_edges")
simple("Enhance Contrast", "Enhance Contrast...", lumen="process.enhance_contrast",
       params={"saturated": num("saturated", 0.35), "normalize": lambda o: o.flag("normalize") or (o.flag("process_all") and not o.flag("equalize")),
               "equalize": flag("equalize"), "process_all": flag("process_all"), "use_stack_histogram": flag("use")}, stack_flag=False,
       notes="saturated normalize equalize process_all use; ImageJ's ContrastEnhancer normalizes every slice when 'process_all' is given without 'equalize'")
simple("Enhance Local Contrast (CLAHE)", "CLAHE", lumen="process.clahe",
       params={"block_size": num("blocksize", 127), "bins": num("histogram", 256), "clip_limit": lambda o: min(0.2, o.num("maximum", 3.0) / 100.0)}, notes="blocksize, histogram, maximum/100 → clip_limit")
simple("Subtract Background...", lumen="process.subtract_background",
       params={"radius": num("rolling", 50.0), "light_background": flag("light"), "create_background": flag("create"), "smooth": lambda o: not o.flag("disable"),
               "method": lambda o: "Sliding paraboloid" if o.flag("sliding") else "Rolling ball"}, notes="rolling → radius; light; create; sliding → Sliding paraboloid; disable → no smoothing")
simple("Bandpass Filter...", lumen="process.fft_bandpass",
       params={"large": num("filter_large", 40.0), "small": num("filter_small", 3.0), "suppress": text("suppress", "None"), "tolerance": num("tolerance", 5.0),
               "autoscale": flag("autoscale"), "saturate": flag("saturate")})
simple("FFT", lumen="process.fft", stack_flag=False)
simple("Add Noise", lumen="noise.add")
simple("Add Specified Noise...", lumen="noise.add_specified", params={"sd": num("standard", 25.0)}, notes="standard → sd")
simple("Salt and Pepper", lumen="noise.salt_pepper", params={"percent": num("percent", 2.5)})
simple("Despeckle", lumen="noise.despeckle")
simple("Remove Outliers...", lumen="noise.remove_outliers", params={"radius": num("radius", 2.0), "threshold": num("threshold", 50.0), "which": text("which", "Bright")})
simple("Remove NaNs...", lumen="noise.remove_nans", params={"radius": num("radius", 2.0)})
for _dir in ("North", "Northeast", "East", "Southeast", "South", "Southwest", "West", "Northwest"):
    simple(_dir, lumen="process.shadows", params={"direction": const(_dir)}, notes="Process ▸ Shadows")
# ---- Process ▸ Math
def _ij_display_sync(ij: "IJ") -> None:
    """Give an image that still carries Lumen's open-time display defaults ImageJ's (every channel at the min–max of channel 1,
    red/green/blue… composite colours – see _imagej_types.imagej_display_patch) before a command whose pixels depend on the
    display range or LUTs: type conversions, RGB Color, Apply LUT, 16/32-bit Math."""
    from .ops._imagej_types import imagej_display_patch
    img = ij.session.active
    if img is None:
        return
    patch = imagej_display_patch(img)
    if patch:
        ij.session.update_display(img.id, patch)


def _math_cmd(*names: str, lumen: str, params: dict[str, Any] | None = None, notes: str = ""):
    """A Process ▸ Math command: ImageJ's process(op, value) scales 16-bit Gamma / Log / Exp and the 32-bit Invert by the display range,
    so the image first gets ImageJ's open-time display state (_ij_display_sync); 'stack' → every plane."""
    def handler(ij: "IJ", o: Options):
        _ij_display_sync(ij)
        return ij._run(lumen, _collect_params(o, params), scope=_scope(o))
    return cmd(*names, lumen=lumen, notes=notes or "ImageJ's ImageProcessor.process arithmetic; " + _SETUP_DIALOG_MATH)(handler)


_SETUP_DIALOG_MATH = "'stack' → every plane, otherwise only the current plane"
for _n, _op, _d in (("Add...", "math.add", 25.0), ("Subtract...", "math.subtract", 25.0), ("Multiply...", "math.multiply", 1.25), ("Divide...", "math.divide", 1.25),
                    ("Min...", "math.min", 25.0), ("Max...", "math.max", 230.0), ("Gamma...", "math.gamma", 0.5), ("Set...", "math.set", 25.0)):
    _math_cmd(_n, lumen=_op, params={"value": num("value", _d)}, notes="value → value; " + _SETUP_DIALOG_MATH)
for _n, _op in (("AND...", "math.and"), ("OR...", "math.or"), ("XOR...", "math.xor")):
    _math_cmd(_n, lumen=_op, params={"value": text("value", "11110000")}, notes="value (binary) → value; " + _SETUP_DIALOG_MATH)
for _n, _op in (("Log", "math.log"), ("Exp", "math.exp"), ("Square", "math.square"), ("Square Root", "math.sqrt"), ("Reciprocal", "math.reciprocal"), ("Abs", "math.abs"),
                ("NaN Background", "math.nan_background"), ("Invert", "edit.invert")):
    _math_cmd(_n, lumen=_op)
simple("Macro...", lumen="math.macro", params={"code": text("code", "v=v")}, notes="Process ▸ Math ▸ Macro: code=[v=v+50*sin(d/10)] with v x y z w h a d (ImageJ's pixel-wise macro); 'stack' → all planes")
# ---- Edit
simple("Fill", lumen="edit.fill")
simple("Clear", lumen="edit.clear")
simple("Clear Outside", lumen="edit.clear_outside")
@cmd("Draw", lumen="edit.draw", notes="Edit ▸ Draw with the global line width (Edit ▸ Options ▸ Line Width); 'stack' → every plane")
def _draw(ij: "IJ", o: Options):
    return ij._run("edit.draw", {"line_width": max(int(ij.line_width), 1)}, scope=_scope(o))
# ---- Image
simple("Crop", lumen="image.crop", stack_flag=False)
simple("Rotate 90 Degrees Right", lumen="image.rotate90", params={"direction": const("Right")}, stack_flag=False)
simple("Rotate 90 Degrees Left", lumen="image.rotate90", params={"direction": const("Left")}, stack_flag=False)
_SETUP_DIALOG = "'stack' → every plane, otherwise only the current plane (IJ.setupDialog)"
transform("Flip Horizontally", lumen="image.flip_h", notes=_SETUP_DIALOG)
transform("Flip Vertically", lumen="image.flip_v", notes=_SETUP_DIALOG)
simple("Flip Z", lumen="image.flip_z", stack_flag=False)
transform("Rotate... ", lumen="image.rotate", params={"angle": num("angle", 15.0), "interpolation": text("interpolation", "Bilinear"), "enlarge": flag("enlarge")},
          notes="Image ▸ Transform ▸ Rotate (note the trailing space, as in ImageJ); " + _SETUP_DIALOG + "; 'enlarge' grows the canvas of every slice")
transform("Translate...", lumen="image.translate", params={"dx": num("x", 0.0), "dy": num("y", 0.0), "interpolation": text("interpolation", "None")}, notes=_SETUP_DIALOG)
simple("Canvas Size...", lumen="image.canvas_size", params={"width": num("width"), "height": num("height"), "position": text("position", "Center"), "zero_fill": flag("zero")}, stack_flag=False)
simple("Bin...", lumen="image.bin", params={"x": num("x", 2), "y": num("y", 2), "z": num("z", 1), "method": lambda o: o.str("bin", "average").capitalize()}, stack_flag=False)
simple("Rename...", lumen="image.rename", params={"title": text("title", "")}, stack_flag=False)
simple("Coordinates...", lumen="adjust.coordinates", params={"origin_x": num("left", 0.0), "origin_y": num("top", 0.0)}, stack_flag=False)
simple("Bleach Correction", lumen="adjust.bleach_correction", params={"method": text("correction", "Simple Ratio"), "background": num("background", 0)},
       notes="Image ▸ Adjust ▸ Bleach Correction (Miura's CorrectBleach): correction=[Simple Ratio|Exponential Fit|Histogram Matching] background=0")
# ---- Image ▸ Type: ImageJ's TypeConverter / RGBStackConverter arithmetic (lumen.core.ops._imagej_types); the conversion options of
# Edit ▸ Options ▸ Conversions and setOption("ScaleConversions") are remembered on the IJ object (ImageConverter.doScaling, ColorProcessor weights)
def _conversion_prefs(ij: "IJ") -> dict[str, bool]:
    return {"scale": bool(getattr(ij, "scale_conversions", True)), "weighted": bool(getattr(ij, "weighted_rgb", False))}


def _conversions(ij: "IJ", o: Options):
    """Edit ▸ Options ▸ Conversions (registered at the end of the module, after the interactive-dialog no-op list that also names it)."""
    ij.scale_conversions = o.flag("scale")
    ij.weighted_rgb = o.flag("weighted")
    return _conversion_prefs(ij)


@cmd("8-bit", lumen="type.8bit", notes="TypeConverter: 16-bit (v−min)·256/(max−min+1)+0.5 and 32-bit (v−min)·255/(max−min)+0.5 from the display range (composites: per channel, "
                                     "ImageJ gives every channel the range of channel 1 on opening); setOption('ScaleConversions', false) clamps instead; RGB → unweighted mean unless Conversions… weighted")
def _to_8bit(ij: "IJ", o: Options):
    _ij_display_sync(ij)
    p = _conversion_prefs(ij)
    return ij._run("type.8bit", {"scale": p["scale"], "weighted": p["weighted"]})


@cmd("16-bit", lumen="type.16bit", notes="TypeConverter: 32-bit (v−min)·65535/(max−min)+0.5 from the display range, or the rounded values with setOption('ScaleConversions', false)")
def _to_16bit(ij: "IJ", o: Options):
    _ij_display_sync(ij)
    return ij._run("type.16bit", {"scale": _conversion_prefs(ij)["scale"]})


@cmd("32-bit", lumen="type.32bit", notes="values unchanged; RGB → (un)weighted gray mean (Conversions…)")
def _to_32bit(ij: "IJ", o: Options):
    return ij._run("type.32bit", {"weighted": _conversion_prefs(ij)["weighted"]})


@cmd("RGB Color", lumen="type.rgb", notes="ImageJ's convertToRGB: the display (LUT index (v−min)·256/(max−min) for 8-bit, 256/(max−min+1) for 16-bit) baked into RGB pixels; a composite "
                                        "is rendered like Stack to RGB (channels summed, clamped) into a new '<title> (RGB)' image")
def _rgb_color(ij: "IJ", o: Options):
    img = ij.img
    if img.rgb:
        return None
    _ij_display_sync(ij)
    return ij._run("type.rgb", {"create_new": img.nc > 1})


@cmd("Stack to RGB", lumen="type.rgb / type.slices_to_rgb", notes="RGBStackConverter: a composite is rendered (channel LUTs and ranges, summed and clamped) into '<title> (RGB)'; a 2–3 slice "
                                                               "single-channel stack takes its slices as R, G, B (16-bit slices scaled from the display range into a new image)")
def _stack_to_rgb(ij: "IJ", o: Options):
    img = ij.img
    if img.rgb:
        return None
    _ij_display_sync(ij)
    if img.nc > 1:
        return ij._run("type.rgb", {"create_new": True})
    if img.nt == 1 and img.nz in (2, 3):
        return ij._run("type.slices_to_rgb", {"create_new": img.data.dtype != np.uint8})
    raise ValueError("Stack to RGB: a 2 or 3 image stack, or a composite image, is required")


simple("RGB Stack", lumen="type.rgb_stack", stack_flag=False)
simple("HSB Stack", lumen="type.hsb_stack", stack_flag=False)
simple("Lab Stack", lumen="type.lab_stack", stack_flag=False)
# ---- Image ▸ Color / LUTs
simple("Split Channels", lumen="color.split", stack_flag=False)
simple("Arrange Channels...", lumen="color.arrange", params={"order": lambda o: ",".join(ch for ch in o.str("new", "") if ch.isdigit())}, stack_flag=False)
simple("Invert LUT", lumen="color.invert_lut", stack_flag=False)
@cmd("Apply LUT", lumen="type.apply_lut", notes="ContrastAdjuster.apply: 8-bit (v−min)·256/(max−min) truncated, 16-bit ·65536 (v ≤ min → 0, v ≥ max → maximum), current channel; every plane of a stack")
def _apply_lut(ij: "IJ", o: Options):
    _ij_display_sync(ij)
    return ij._run("type.apply_lut", {})
# ---- Binary: Process ▸ Binary with ImageJ's semantics; iterations / count / pad edges / black background / EDM output are the
# "Options..." settings (ImageJ's Prefs.binaryOptions) remembered on the IJ object
def _binary_prefs(ij: "IJ") -> dict[str, Any]:
    return {"iterations": int(getattr(ij, "binary_iterations", 1)), "count": int(getattr(ij, "binary_count", 1)),
            "pad_edges": bool(getattr(ij, "binary_pad_edges", False))}


def _edm_output(ij: "IJ") -> str:
    return str(getattr(ij, "edm_output", "Overwrite"))


@cmd("Erode", lumen="binary.erode", notes="ImageJ's 8-neighbour erosion with the Options... iterations / count / pad edges")
def _erode(ij: "IJ", o: Options):
    return ij._run("binary.erode", _binary_prefs(ij), scope=_scope(o))


@cmd("Dilate", lumen="binary.dilate", notes="ImageJ's 8-neighbour dilation with the Options... iterations / count")
def _dilate(ij: "IJ", o: Options):
    return ij._run("binary.dilate", _binary_prefs(ij), scope=_scope(o))


@cmd("Open", lumen="binary.open", notes="iterations × erode, then iterations × dilate (Options... iterations / count / pad edges)")
def _open(ij: "IJ", o: Options):
    return ij._run("binary.open", _binary_prefs(ij), scope=_scope(o))


@cmd("Close-", lumen="binary.close", notes="iterations × dilate, then iterations × erode (Options... iterations / count / pad edges)")
def _close(ij: "IJ", o: Options):
    return ij._run("binary.close", _binary_prefs(ij), scope=_scope(o))


@cmd("Outline", lumen="binary.outline", notes="ImageJ's Outline on a 0/255 mask: with a black background (Options... black) the ring of background pixels around each object; with 'black' off ImageJ's result (everything but that ring)")
def _outline(ij: "IJ", o: Options):
    from .ops.binary import OUTLINE_IJ, OUTLINE_IJ_WHITE
    return ij._run("binary.outline", {"mode": OUTLINE_IJ if getattr(ij, "black_background", True) else OUTLINE_IJ_WHITE}, scope=_scope(o))


simple("Fill Holes", lumen="binary.fill_holes")
simple("Skeletonize", lumen="binary.skeletonize", params={"method": text("method")}, notes="ImageJ's lookup-table thinning; method=[Zhang-Suen|Lee|Thin (Guo-Hall)|Medial axis] is a Lumen extension")


@cmd("Distance Map", lumen="binary.distance_map", notes="EDM output type from Options... edm=[Overwrite|8-bit|16-bit|32-bit]; default Overwrite = the 8-bit EDM in place")
def _distance_map(ij: "IJ", o: Options):
    return ij._run("binary.distance_map", {"output": _edm_output(ij)}, scope="stack")


@cmd("Ultimate Points", lumen="binary.ultimate_points", notes="EDM output type from Options... (default Overwrite: the image becomes the UEPs carrying their EDM value); output=Image → 32-bit UEP image, output=[Point selection] is a Lumen extension")
def _ultimate_points(ij: "IJ", o: Options):
    out = o.str("output", "")
    if out.lower().startswith("image"):
        output = "Image"
    elif out.lower().startswith("point"):
        output = "Point selection"
    else:
        output = {"32-bit": "Image"}.get(_edm_output(ij), _edm_output(ij))
    return ij._run("binary.ultimate_points", {"output": output}, scope="stack")


simple("Watershed", lumen="binary.watershed", params={"method": text("method"), "tolerance": num("tolerance")}, notes="ImageJ's EDM/UEP watershed; method=[Distance transform (h-maxima)] / tolerance= are Lumen extensions")


@cmd("Voronoi", lumen="binary.voronoi", notes="ImageJ's Voronoi: the lines carry the EDM value of the background (EDM output Overwrite)")
def _voronoi(ij: "IJ", o: Options):
    return ij._run("binary.voronoi", {"values": "EDM"}, scope=_scope(o))


def _local_threshold_method(o: Options) -> str:
    spec = REGISTRY.get("threshold.local")
    choices = next(p.choices for p in spec.params if p.name == "method")
    m = o.str("method", "Sauvola")
    low = {c.lower(): c for c in choices}
    return low.get(m.lower().replace("midgray", "midgrey"), m)


simple("Auto Local Threshold", lumen="threshold.local", params={"method": _local_threshold_method, "radius": num("radius", 15), "parameter_1": num("parameter_1", 0.0), "white_objects": flag("white")},
       notes="Fiji: method=Bernsen|Contrast|Mean|Median|MidGrey|Niblack|Otsu|Phansalkar|Sauvola radius= parameter_1= white (parameter_2 is ignored)")
# ---- Stacks
@cmd("Z Project...", lumen="stacks.z_project", notes="projection start stop; nSlices==1 → projects the frames (ImageJ ZProjector); 'all' → every frame, else only the current frame")
def _z_project(ij: "IJ", o: Options):
    img = ij.img
    over_t = img.nz == 1 and img.nt > 1  # ZProjector.doHyperStackProjection: if (slices==1) slices = nFrames
    out = ij._run("stacks.z_project", {"method": o.str("projection", "Max Intensity"), "start": int(o.num("start", 1)), "stop": int(o.num("stop", 0)), "axis": "T" if over_t else "Z"})
    if not over_t and img.nt > 1 and not o.flag("all"):  # ZProjector: firstFrame = lastFrame = imp.getFrame() unless 'all'
        new = ij.session.images[out["new_images"][0]]
        new.replace_data(np.ascontiguousarray(np.take(new.data, [img.display.t], axis=0)), rgb=new.rgb)
        new.display.t = 0
        ij.session._changed("images")
    return out


simple("Grouped Z Project...", lumen="stacks.grouped_z_project", params={"method": text("projection", "Max Intensity"), "group_size": num("group", 2)}, stack_flag=False)
simple("Reverse", lumen="stacks.reverse", stack_flag=False)
simple("Delete Slice", lumen="stacks.delete_slice", params={"axis": lambda o: {"channel": "C", "frame": "T"}.get(o.str("delete", "slice"), "Z")}, stack_flag=False)
@cmd("Add Slice", lumen="stacks.add_slice", notes="add=channel/slice/frame (ImageJ hyperstack dialog) → C/Z/T")
def _add_slice(ij: "IJ", o: Options):
    axis = {"channel": "C", "frame": "T"}.get(o.str("add", "slice"), "Z")
    spec = REGISTRY.get("stacks.add_slice")
    if axis != "C" or "C" in next(p.choices for p in spec.params if p.name == "axis"):
        return ij._run("stacks.add_slice", {"axis": axis})
    # stacks.add_slice only knows Z/T: insert a blank channel after the current one here (as ImageJ does)
    from .image import ChannelDisplay, _default_lut_for
    img = ij.img
    if img.rgb:
        raise ValueError("Add Slice cannot add a channel to an RGB image – convert to an RGB stack first")
    cur = img.display.c
    blank = np.zeros_like(img.data[:, :, cur:cur + 1])
    data = np.concatenate([img.data[:, :, :cur + 1], blank, img.data[:, :, cur + 1:]], axis=2)
    ij.session.histories[img.id].push(img, spec.undo_label or spec.name)
    nc = img.nc + 1
    lo, hi = (0.0, 255.0) if img.data.dtype == np.uint8 else (0.0, 1.0)
    img.display.channels.insert(cur + 1, ChannelDisplay(name=f"C{nc}", lut=_default_lut_for(cur + 1, nc, False), min=lo, max=hi, visible=True))
    img.replace_data(data, rgb=False)
    ij.session._changed("images")
    return {"image_id": img.id}
simple("Make Montage...", lumen="stacks.montage", params={"columns": num("columns", 0), "rows": num("rows", 0), "scale": num("scale", 1.0), "border": num("border", 0)}, stack_flag=False,
       notes="first/last/increment/label are ignored")
simple("Stack to Images", lumen="stacks.to_images", stack_flag=False)
simple("Images to Stack", lumen="stacks.images_to_stack", params={"title": text("name", "Stack"), "name_filter": text("title", "")}, stack_flag=False)
simple("Concatenate...", lumen="stacks.concatenate", params={"image1": text("image1"), "image2": text("image2"), "title": text("title", "Concatenated")}, stack_flag=False, notes="image3+ are ignored")
simple("Stack to Hyperstack...", lumen="stacks.to_hyperstack", params={"channels": num("channels", 1), "slices": num("slices", 1), "frames": num("frames", 1), "order": lambda o: o.str("order", "xyczt").split("(")[0]}, stack_flag=False)
simple("Plot Z-axis Profile", lumen="stacks.z_profile", stack_flag=False)
simple("Reduce...", lumen="stacks.reduce", params={"factor": num("reduction", 2)}, stack_flag=False)
simple("3D Project...", lumen="stacks.project3d", params={"method": text("projection", "Brightest Point"), "axis": text("axis", "Y-Axis"), "angle_step": num("rotation", 10), "total_rotation": num("total", 360),
                                                            "initial_angle": num("initial", 0), "lower_threshold": num("lower", 1), "upper_threshold": num("upper", 255), "depth_cueing": num("surface", 0),
                                                            "interpolate": flag("interpolate")}, stack_flag=False)
simple("Orthogonal Views", lumen="stacks.orthogonal_views", stack_flag=False)
simple("Combine...", lumen="stacks.combine", params={"image1": text("stack1"), "image2": text("stack2"), "vertical": flag("combine")}, stack_flag=False)
simple("Straighten...", lumen="stacks.straighten", params={"line_width": num("line", 20), "all_planes": flag("process")}, stack_flag=False)
def _op_params(op_id: str) -> set[str]:
    return {p.name for p in REGISTRY.get(op_id).params}


@cmd("Label...", lumen="stacks.label_slices",
     notes="format=0/0000/Text [text=] starting= interval=; numbers the frames (nFrames>1) else the slices, like ImageJ's StackLabeler (label = starting + (n-1)*interval when the op supports start/interval); x/y/range ignored")
def _label(ij: "IJ", o: Options):
    img = ij.img
    n_key = "frame" if img.nt > 1 else "slice"  # StackLabeler.updateIndex: count frames when nFrames > 1, else Z slices
    p: dict[str, Any] = {"position": "Upper Left", "font_size": o.num("font", 18)}
    if {"start", "interval"} <= _op_params("stacks.label_slices"):
        p["start"], p["interval"] = o.num("starting", 0.0), o.num("interval", 1.0)
        number = "{label}"
    else:  # the op has no start/interval yet: plain 1-based plane numbers (= starting=1 interval=1)
        number = "{%s}" % n_key
    fmt = o.str("format", "0").lower()
    text = o.str("text", "")
    if fmt == "text":
        p["format"] = text
    elif fmt in ("00:00", "00:00:00"):
        p["format"] = "{time:.0f} {unit}"
    elif fmt == "0000":
        p["format"] = (text + " " + number[:-1] + ":04.0f}").strip()  # StackLabeler: text first for the zero-padded format
    else:
        p["format"] = (number + " " + text).strip()  # ImageJ: number first, then the optional text
    return ij._run("stacks.label_slices", p)


@cmd("Time Stamper", lumen="overlay.time_stamp", notes="starting= (start time) interval= (time per frame) decimal= or= (suffix) font=; x/y/anti-aliased ignored, stamp is an overlay (Burn In to make it permanent)")
def _time_stamper(ij: "IJ", o: Options):
    p: dict[str, Any] = {"format": "{time:.%df} %s" % (int(o.num("decimal", 0)), o.str("or", "sec")), "font_size": o.num("font", 12)}
    if {"start_time", "interval"} <= _op_params("overlay.time_stamp"):
        # ImageJ's Time_Stamper: time = starting + (frame-1)*interval, never the calibration (defaults 0 / 1)
        p["start_time"], p["interval"] = o.num("starting", 0.0), o.num("interval", 1.0)
    # else: the op can only express a 0-based frame index and the calibration interval, so starting/interval cannot be honoured
    # ('starting' is a time, not a frame index, so it must not become start_frame)
    return ij._run("overlay.time_stamp", p)
# ---- Selection
simple("Select All", lumen="selection.all", stack_flag=False)
simple("Select None", lumen="selection.none", stack_flag=False)
def _px_size(v: float) -> float:
    """Calibrated size of one pixel, falling back to 1 for an unset/invalid calibration."""
    return float(v) if v and v > 0 else 1.0


@cmd("Enlarge...", lumen="selection.enlarge", notes="enlarge= is in calibrated units unless 'pixel' is given (ImageJ RoiEnlarger: pixels = n / pixelWidth)")
def _enlarge(ij: "IJ", o: Options):
    n = o.num("enlarge", 5.0)
    if "enlarge" in o and not o.flag("pixel"):
        n = n / _px_size(ij.img.calibration.pixel_width)
    return ij._run("selection.enlarge", {"pixels": n})


@cmd("Make Band...", lumen="selection.band", notes="band= is always in calibrated units (ImageJ: round(size / pixelWidth))")
def _make_band(ij: "IJ", o: Options):
    w = o.num("band", 15.0)
    if "band" in o:
        w = w / _px_size(ij.img.calibration.pixel_width)
    return ij._run("selection.band", {"width": w})


simple("Convex Hull", lumen="selection.convex_hull", stack_flag=False)
simple("Fit Ellipse", lumen="selection.fit_ellipse", stack_flag=False)
simple("To Bounding Box", lumen="selection.bounding_box", stack_flag=False)
simple("Rotate...", lumen="selection.rotate", params={"angle": num("angle", 15.0)}, stack_flag=False, notes="Edit ▸ Selection ▸ Rotate (no trailing space); falls back to Image ▸ Rotate without a selection")
simple("Scale... ", lumen="selection.scale", params={"x_scale": num("x", 1.5), "y_scale": num("y", 1.5), "centered": flag("centered")}, stack_flag=False, notes="Edit ▸ Selection ▸ Scale (trailing space)")
simple("Interpolate", lumen="selection.interpolate", params={"interval": num("interval", 1.0), "smooth": flag("smooth"), "adjust": flag("adjust")}, stack_flag=False,
       notes="interval smooth adjust (ImageJ's macro default interval is 1)")
simple("Line Width...", lumen="selection.line_width", params={"width": num("line", 1)}, stack_flag=False)
@cmd("Specify...", lumen="selection.specify", notes="width height x y oval centered; 'scaled' → width/x in units ÷ pixel_width, height/y ÷ pixel_height (ImageJ SpecifyROI)")
def _specify(ij: "IJ", o: Options):
    w, h, x, y = o.num("width", 100), o.num("height", 100), o.num("x", 0), o.num("y", 0)
    if o.flag("scaled"):
        cal = ij.img.calibration
        pw, ph = _px_size(cal.pixel_width), _px_size(cal.pixel_height)
        w, h, x, y = w / pw, h / ph, x / pw, y / ph
    return ij._run("selection.specify", {"width": w, "height": h, "x": x, "y": y, "oval": o.flag("oval"), "centered": o.flag("centered")})
simple("Create Mask", lumen="selection.create_mask", stack_flag=False)
simple("Create Selection", lumen="selection.create", stack_flag=False)
simple("Make Inverse", lumen="selection.inverse", stack_flag=False)
simple("Line to Area", lumen="selection.line_to_area", stack_flag=False)
simple("Area to Line", lumen="selection.area_to_line", stack_flag=False)
# ---- Overlay
simple("Flatten", lumen="overlay.flatten", stack_flag=False)
simple("Remove Overlay", lumen="overlay.remove", stack_flag=False)
simple("From ROI Manager", lumen="overlay.from_manager", stack_flag=False)
simple("To ROI Manager", lumen="overlay.to_roi_manager", stack_flag=False)
simple("List Elements", lumen="overlay.list", stack_flag=False)
simple("Scale Bar...", lumen="overlay.scale_bar", params={"length": num("width", 0), "thickness": lambda o: o.num("height", o.num("thickness", 6)), "font_size": num("font", 14),
                                                          "position": text("location", "Lower Right"), "color": lambda o: _color_hex(o.get("color"), "#ffffff"), "show_label": lambda o: not o.flag("hide")}, stack_flag=False)
simple("Calibration Bar...", lumen="overlay.calibration_bar", params={"position": text("location", "Upper Right"), "ticks": num("number", 5), "decimals": num("decimal", 0), "font_size": num("font", 12),
                                                                      "color": lambda o: _color_hex(o.get("label"), "#ffffff")}, stack_flag=False)
simple("Burn In Overlay", lumen="overlay.burn_in", stack_flag=False)
# ---- Analyze
simple("Measure", lumen="analyze.measure", stack_flag=False)
simple("Clear Results", lumen="analyze.clear_results", stack_flag=False)
simple("Summarize", lumen="analyze.summarize", stack_flag=False)
simple("Plot Profile", lumen="analyze.plot_profile", stack_flag=False)
simple("Distribution...", lumen="analyze.distribution", params={"column": text("parameter", "Area"), "bins": num("or", 20)}, stack_flag=False)
simple("Find Maxima...", lumen="analyze.find_maxima", params={"prominence": lambda o: o.num("prominence", o.num("noise", 10.0)), "output": text("output", "Point Selection"), "exclude_edges": flag("exclude"),
                                                             "light_background": flag("light"), "strict": flag("strict"), "above_lower_threshold": flag("above")}, stack_flag=False)
simple("Curve Fitting...", lumen="analyze.curve_fit", params={"function": text("fit", "Straight Line")}, stack_flag=False)
simple("Fractal Box Count...", lumen="analyze.fractal_box_count", params={"box_sizes": text("box", "2,3,4,6,8,12,16,32,64")}, stack_flag=False)
simple("Surface Plot...", lumen="analyze.surface_plot", params={"smoothing": lambda o: 1.0 if o.flag("smooth") else 0.0, "wireframe": flag("wireframe"), "invert": flag("invert")}, stack_flag=False)
simple("Label", lumen="analyze.label", stack_flag=False)
simple("Multi Measure", lumen="analyze.multi_measure", params={"one_row_per_slice": lambda o: o.flag("one")}, stack_flag=False)
simple("Angle", lumen="analyze.angle", stack_flag=False)
simple("Gel Analyzer", lumen="analyze.gel_lanes", stack_flag=False, notes="Lumen's one-step gel analysis: lanes = rectangles in the ROI Manager")


def _calibrate_function(o: Options) -> str:
    choices = next(p.choices for p in REGISTRY.get("analyze.calibrate").params if p.name == "function")
    f = o.str("function", "Straight Line")
    low = {c.lower(): c for c in choices}
    return low.get(f.lower(), f)


simple("Calibrate...", lumen="analyze.calibrate", params={"function": _calibrate_function, "unit": text("unit", "OD"), "pixel_values": text("text1", ""), "standards": text("text2", "")}, stack_flag=False,
       notes="function=[None|Straight Line|2nd Degree Polynomial|3rd Degree Polynomial|Exponential|Exponential with Offset|Exponential Recovery|Power|Log|Rodbard|Rodbard (NIH Image)|Gamma Variate|Chapman-Richards|Uncalibrated OD] unit= text1=[pixel values] text2=[standards]")
# ---- File
simple("Close", lumen="file.close", stack_flag=False)
simple("Close All", lumen="file.close_all", stack_flag=False)
simple("Revert", lumen="file.revert", stack_flag=False)


# ---- commands that need custom logic
_SAMPLES = {"blobs": "Blobs", "blobs (25k)": "Blobs", "t1 head": "T1 head (MRI stack)", "t1 head (2.4m, 16-bits)": "T1 head (MRI stack)", "mitosis": "Human mitosis",
            "mitosis (26mb, 5d stack)": "Human mitosis", "fluorescent cells": "Synthetic nuclei (3 channels)", "fluorescent cells (400k)": "Synthetic nuclei (3 channels)",
            "hela cells": "Synthetic nuclei (3 channels)", "hela cells (1.3m, 48-bit rgb)": "Synthetic nuclei (3 channels)", "confocal series": "Cells (3-D, 2 channels)",
            "confocal series (2.2mb)": "Cells (3-D, 2 channels)", "clown": "Astronaut (RGB)", "clown (14k)": "Astronaut (RGB)", "cell colony": "Coins", "cell colony (31k)": "Coins",
            "coins": "Coins", "moon": "Moon", "astronaut": "Astronaut (RGB)", "nuclei timelapse": "Nuclei timelapse (synthetic)", "cells": "Cells (3-D, 2 channels)"}


# ImageJ's File ▸ Open Samples menu (ImageJ 1.54): "<name> (<size>)" → the name file.open_imagej_sample downloads from imagej.net/images
_IJ_SAMPLE_MENU = {"AuPbSn 40 (56K)": "AuPbSn 40", "Bat Cochlea Volume (19K)": "Bat Cochlea Volume", "Bat Cochlea Renderings (449K)": "Bat Cochlea Renderings", "Blobs (25K)": "Blobs (25K)",
                   "Boats (356K)": "Boats", "Bridge (174K)": "Bridge", "Cardio (768K, RGB DICOM)": "Cardio (RGB DICOM)", "Cell Colony (31K)": "Cell Colony", "Clown (14K)": "Clown",
                   "Confocal Series (2.2MB)": "Confocal Series", "Dot Blot (7K)": "Dot Blot", "Embryos (42K)": "Embryos", "Fluorescent Cells (400K)": "Fluorescent Cells", "Fly Brain (1MB)": "Flybrain",
                   "Gel (105K)": "Gel", "HeLa Cells (1.3M, 48-bit RGB)": "HeLa Cells", "Image with Overlay (1.3M)": "Image with Overlay", "Leaf (36K)": "Leaf", "Line Graph (21K)": "Line Graph",
                   "M51 Galaxy (177K, 16-bits)": "M51 Galaxy", "Mitosis (26MB, 5D stack)": "Mitosis", "Neuron (1.6M, 5 channels)": "Neuron", "Nile Bend (1.9M)": "Nile Bend",
                   "Organ of Corti (2.8M, 4D stack)": "Organ of Corti", "Particles (75K)": "Particles", "T1 Head (2.4M, 16-bits)": "T1 Head", "T1 Head Renderings (736K)": "T1 Head Renderings",
                   "TEM Filter (112K)": "TEM Filter", "Tree Rings (48K)": "Tree Rings"}
_IJ_SAMPLE_LOOKUP = {k.lower(): v for k, v in _IJ_SAMPLE_MENU.items()}
_IJ_SAMPLE_LOOKUP.update({v.lower(): v for v in _IJ_SAMPLE_MENU.values()})
_IJ_SAMPLE_LOOKUP.update({"flybrain": "Flybrain", "fly brain": "Flybrain", "cardio": "Cardio (RGB DICOM)", "cardio (rgb dicom)": "Cardio (RGB DICOM)", "blobs": "Blobs (25K)"})


@cmd(*_IJ_SAMPLE_MENU, *(v for v in _IJ_SAMPLE_MENU.values() if v not in _IJ_SAMPLE_MENU), "Blobs", "Fly Brain", "Cardio",
     lumen="file.sample / file.open_imagej_sample",
     notes="the 29 ImageJ sample names (with or without the size suffix): Blobs, T1 Head, Mitosis, Fluorescent Cells, HeLa Cells, Confocal Series, Clown and Cell Colony open the bundled "
           "offline Lumen stand-ins (file.sample); the others are downloaded from imagej.net/images (cached in ~/.lumen/samples); 'download' forces the original ImageJ file")
def _sample(ij: "IJ", o: Options, command: str = ""):
    key = command.lower().strip()
    builtin = _SAMPLES.get(key)
    if builtin is not None and not o.flag("download"):
        return ij._run("file.sample", {"name": builtin})
    name = _IJ_SAMPLE_LOOKUP.get(key)
    if name is None:
        raise ValueError(f"Unknown sample image {command!r}")
    return ij._run("file.open_imagej_sample", {"name": name, "use_cache": not o.flag("reload")})


@cmd("Duplicate...", lumen="image.duplicate",
     notes="'duplicate'/'stack' → whole stack; else the current plane (all channels of the current slice for a composite-mode hyperstack, as ImageJ's Duplicator); "
           "range=first-last (plain stacks) or channels=/slices=/frames= (hyperstacks) with 'duplicate'; title=")
def _duplicate(ij: "IJ", o: Options):
    img = ij.img
    whole = o.flag("duplicate") or o.flag("stack")
    # ImageJ Duplicator.duplicateHyperstack: without 'duplicate', all channels of the current slice are kept only for a
    # CompositeImage in COMPOSITE mode with > 1 channel that is not a "single composite" (nz == nt == 1).
    composite_slice = not img.rgb and img.nc > 1 and img.display.mode == "composite" and img.nz * img.nt > 1
    what = "Whole stack" if whole else ("Current slice (all channels)" if (img.rgb or composite_slice) else "Current plane")
    out = ij._run("image.duplicate", {"title": o.str("title", f"{img.name}-1"), "what": what, "crop_to_selection": ij.roi is not None})
    new = ij.session.images[out["new_images"][0]]
    from .ops.stacks import _parse_range
    timelapse = img.nz == 1 and img.nt > 1  # ImageJ treats a (c=1, z=1, t>1) image as a plain stack whose "slices" are the frames
    # Plain (non-hyper, non-composite) stacks take ImageJ's range=first-last: a single number means first..end,
    # first > last or junk → whole stack, values are clamped. Hyperstacks ignore range= and use channels=/slices=/frames=.
    plain = (img.nc == 1 or img.rgb) and (img.nz > 1) != (img.nt > 1)
    if plain and whole and "range" in o and o["range"] is not True:
        key, n = ("frames", img.nt) if timelapse else ("slices", img.nz)
        if key not in o:
            bits = [b for b in re.split(r"[\s-]+", str(o["range"]).strip()) if b]
            first = int(float(bits[0])) if bits and _NUM_RE.match(bits[0]) else 1
            last = int(float(bits[1])) if len(bits) >= 2 and _NUM_RE.match(bits[1]) else n
            first, last = max(1, first), min(n, last)
            if first > last:
                first, last = 1, n
            o = Options(o, **{key: f"{first}-{last}"})
    for key, ax in (("channels", 2), ("slices", 0 if timelapse else 1), ("frames", 0)):
        if whole and key in o and o[key] is not True and new.data.shape[ax] > 1:  # ImageJ ignores the ranges unless 'duplicate' is set
            idx = _parse_range(str(o[key]), new.data.shape[ax])
            data = np.take(new.data, idx, axis=ax)
            if ax == 2 and not new.rgb:
                new.display.channels = [type(new.display.channels[i])(**new.display.channels[i].to_dict()) for i in idx]
            new.replace_data(np.ascontiguousarray(data), rgb=new.rgb and ax != 2)
            new.display.t = new.display.z = new.display.c = 0
    ij.session._changed("images")
    return out


@cmd("Scale...", lumen="image.scale", notes="x y z width height depth interpolation average create title; ImageJ's Scaler: 'create' → ImageProcessor.resize (averaging kernel when downsizing) "
                                           "into a new window, otherwise ImageProcessor.scale inside the current canvas")
def _scale(ij: "IJ", o: Options):
    img = ij.img
    if not o.flag("create"):
        _ij_display_sync(ij)  # the in-place scale fills uncovered 16/32-bit pixels with the display minimum
    p: dict[str, Any] = {"x_scale": o.num("x", 1.0) or 1.0, "y_scale": o.num("y", o.num("x", 1.0)) or 1.0, "z_scale": o.num("z", 1.0) or 1.0, "interpolation": o.str("interpolation", "Bilinear"),
                         "create_new": o.flag("create"), "average_when_downsizing": o.flag("average"), "keep_canvas": not o.flag("create")}
    if "width" in o and o["width"] is not True and o.num("width", -1) >= 0:  # ImageJ records 'width=-' placeholders
        p["width"] = int(o.num("width"))
    if "height" in o and o["height"] is not True and o.num("height", -1) >= 0:
        p["height"] = int(o.num("height"))
    if "depth" in o and o["depth"] is not True and img.nz > 1 and o.num("depth", 0) > 0:
        p["z_scale"] = o.num("depth") / img.nz
    out = ij._run("image.scale", p)
    if o.flag("create") and o.str("title"):
        new = ij.session.images[out["new_images"][0]]
        new.name = ij.session.unique_name(o.str("title"))
    return out


@cmd("Size...", lumen="image.scale", notes="width height depth → in-place resize")
def _size(ij: "IJ", o: Options):
    img = ij.img
    p: dict[str, Any] = {"width": int(o.num("width", img.width)), "height": int(o.num("height", img.height)), "interpolation": o.str("interpolation", "Bilinear"), "create_new": False,
                         "average_when_downsizing": o.flag("average")}
    if o.flag("constrain") and "height" not in o:
        p["height"] = int(round(img.height * p["width"] / img.width))
    if "depth" in o and img.nz > 1:
        p["z_scale"] = o.num("depth", img.nz) / img.nz
    return ij._run("image.scale", p)


@cmd("Properties...", lumen="image.properties / stacks.to_hyperstack", notes="pixel_width pixel_height voxel_depth unit frame=[5 sec]; channels/slices/frames re-shape the stack; with a selection and only ROI keys (name stroke width fill position) it is the selection dialog ('Properties... ')")
def _properties(ij: "IJ", o: Options):
    img = ij.img
    if ij.roi is not None and o and any(k in o for k in _ROI_KEYS) and not any(k in o for k in ("channels", "slices", "frames", "pixel_width", "pixel_height", "voxel_depth", "unit", "frame")):
        return _roi_properties(ij, o)
    cal = img.calibration
    fi, tu = cal.frame_interval, cal.time_unit
    fr = o.str("frame", "")
    if fr:
        m = re.match(r"\s*([-+\d.eE]+)\s*(\S*)", fr)
        if m:
            fi = float(m.group(1)); tu = m.group(2) or tu
    out = ij._run("image.properties", {"pixel_width": o.num("pixel_width", cal.pixel_width), "pixel_height": o.num("pixel_height", cal.pixel_height), "voxel_depth": o.num("voxel_depth", cal.pixel_depth),
                                       "unit": o.str("unit", cal.unit), "frame_interval": fi, "time_unit": tu})
    c, z, t = int(o.num("channels", img.nc)), int(o.num("slices", img.nz)), int(o.num("frames", img.nt))
    if (c, z, t) != (img.nc, img.nz, img.nt) and c * z * t == img.nc * img.nz * img.nt and not img.rgb:
        ij._run("stacks.to_hyperstack", {"channels": c, "slices": z, "frames": t})
    return out


@cmd("Set Scale...", lumen="image.properties", notes="distance known pixel unit → pixel size = known/distance")
def _set_scale(ij: "IJ", o: Options):
    img = ij.img
    cal = img.calibration
    dist, known = o.num("distance", 0.0), o.num("known", 0.0)
    aspect = o.num("pixel", 1.0) or 1.0
    unit = o.str("unit", "pixel")
    if dist <= 0 or known <= 0 or unit.lower() in ("pixel", "pixels", ""):
        pw, ph, unit = 1.0, 1.0, "pixel"
    else:
        pw = known / dist
        ph = pw * aspect
    unit = {"um": "µm", "micron": "µm", "microns": "µm"}.get(unit, unit)
    return ij._run("image.properties", {"pixel_width": pw, "pixel_height": ph, "voxel_depth": cal.pixel_depth if unit != "pixel" else 1.0, "unit": unit, "frame_interval": cal.frame_interval, "time_unit": cal.time_unit})


@cmd("Threshold...", lumen="threshold.compute", notes="opens the threshold panel in ImageJ; here computes the Default threshold")
def _threshold_dialog(ij: "IJ", o: Options):
    return ij._run("threshold.compute", {"method": o.str("method", "Default"), "dark_background": o.flag("dark") or not o.flag("light")})


@cmd("Auto Threshold", lumen="threshold.apply", notes="method=Otsu white [stack | use_stack_histogram] → binary mask ('white' = white objects on black background); without 'stack' only the current slice is thresholded")
def _auto_threshold(ij: "IJ", o: Options):
    method = o.str("method", "Default")
    if method not in METHODS:
        low = {m.lower(): m for m in METHODS}
        method = low.get(method.lower(), "Default")
    img = ij.img
    whole = o.flag("stack") or o.flag("use_stack_histogram")
    single = not whole and img.nt * img.nz * img.nc > 1
    before = img.data if single and img.data.dtype != np.uint8 else None
    d = img.display
    pos = (d.t, d.z, d.c)
    out = ij._run("threshold.apply", {"method": method, "dark_background": o.flag("white"), "per_plane": o.flag("stack") and not o.flag("use_stack_histogram"),
                                      "only_current_plane": single})
    if before is not None and img.data.shape == before.shape:
        # threshold.apply only keeps the untouched planes of an 8-bit stack; like the Fiji plugin, a single slice of a
        # 16-bit / float stack stays in its own type with the thresholded slice written as 0 / max.
        hi = np.iinfo(before.dtype).max if np.issubdtype(before.dtype, np.integer) else 1.0
        data = before.copy()
        data[pos] = (img.data[pos] > 0).astype(before.dtype) * hi
        img.replace_data(data, rgb=False)
        img.lut_is_binary = False
        for ch in img.display.channels:
            ch.min, ch.max = 0.0, float(hi)
        ij.session._changed("images")
    return out


@cmd("Convert to Mask", "Make Binary", lumen="threshold.apply",
     notes="ImageJ's Thresholder: a threshold set with setThreshold / setAutoThreshold is applied; without one the legacy ISODATA auto threshold "
           "(mode-corrected, object side from the histogram) is used. method= and background= only count for stacks with 'calculate' (per-slice levels "
           "without the +1, 'Default'/'Default' = legacy ISODATA); 'only' converts the current slice; the stack dialog's 'black' flag resets Black background like ImageJ")
def _convert_to_mask(ij: "IJ", o: Options):
    from .ops.threshold_ops import LEGACY
    img = ij.img
    has_thr = ij.session.thresholds.get(img.id) is not None
    stack = img.nt * img.nz * img.nc > 1
    only = o.flag("only")
    if stack:
        # Thresholder.convertStack: the dialog's "Black background" checkbox writes Prefs.blackBackground (false unless 'black' is in the options)
        ij.black_background = o.flag("black")
    if stack and o.flag("calculate") and not has_thr:  # a set threshold switches 'calculate' off (useLocal = false)
        method = o.str("method", "Default")
        bg = o.str("background", "Default")
        if method == "Default" and bg == "Default":
            return ij._run("threshold.apply", {"method": LEGACY, "per_plane": True, "only_current_plane": only})
        if method not in METHODS:
            low = {m.lower(): m for m in METHODS}
            method = low.get(method.lower(), "Default")
        return ij._run("threshold.apply", {"method": method, "dark_background": bg.lower() != "light", "per_plane": True, "add_one": False, "only_current_plane": only})
    if has_thr:
        return ij._run("threshold.apply", {"method": "Current", "only_current_plane": only})
    return ij._run("threshold.apply", {"method": LEGACY, "only_current_plane": only})


@cmd("Options...", lumen="binary.* ",
     notes="Process ▸ Binary ▸ Options: iterations= count= black pad edm=[Overwrite|8-bit|16-bit|32-bit] are remembered for the binary commands (ImageJ's "
           "Prefs.binaryOptions; like ImageJ a flag is off unless present); do=Erode/Dilate/Open/Close/Outline/Fill Holes/Skeletonize is executed")
def _binary_options(ij: "IJ", o: Options):
    do = o.str("do", "Nothing")
    ij.binary_iterations = max(1, min(int(o.num("iterations", 1)), 100))
    ij.binary_count = max(1, min(int(o.num("count", 1)), 8))
    ij.black_background = o.flag("black")
    ij.binary_pad_edges = o.flag("pad")
    edm = o.str("edm", "Overwrite")
    ij.edm_output = edm if edm in ("Overwrite", "8-bit", "16-bit", "32-bit") else "Overwrite"
    ops = {"erode": "binary.erode", "dilate": "binary.dilate", "open": "binary.open", "close": "binary.close", "outline": "binary.outline", "fill holes": "binary.fill_holes", "skeletonize": "binary.skeletonize"}
    if do.lower() in ops:
        if do.lower() in ("erode", "dilate", "open", "close"):
            p: dict[str, Any] = _binary_prefs(ij)
        elif do.lower() == "outline":
            from .ops.binary import OUTLINE_IJ, OUTLINE_IJ_WHITE
            p = {"mode": OUTLINE_IJ if ij.black_background else OUTLINE_IJ_WHITE}
        else:
            p = {}
        return ij._run(ops[do.lower()], p, scope=_scope(o))
    return None


_SET_MEASUREMENT_FLAGS = {"display": "display_label", "nan": "nan_empty_cells", "scientific": "scientific_notation", "add": "add_to_overlay", "invert": "invert_y"}


@cmd("Set Measurements...", lumen="analyze.set_measurements", notes="area mean standard modal min centroid center perimeter bounding fit shape feret's integrated median skewness kurtosis area_fraction stack limit decimal= display nan scientific add invert")
def _set_measurements(ij: "IJ", o: Options):
    table = {"area": "area", "mean": "mean", "standard": "std", "modal": "mode", "min": "min", "max": "max", "centroid": "centroid", "center": "center_of_mass", "perimeter": "perimeter",
             "bounding": "bounding_rect", "fit": "fit_ellipse", "shape": "shape_descriptors", "feret's": "feret", "feret": "feret", "integrated": "integrated_density", "median": "median",
             "skewness": "skewness", "kurtosis": "kurtosis", "area_fraction": "area_fraction", "stack": "stack_position", "display": None, "nan": None, "scientific": None, "add": None, "invert": None, "limit": None}
    meas = []
    for k in o:
        m = table.get(k)
        if m and m not in meas:
            meas.append(m)
        if k == "min" and "max" not in meas:
            meas.append("max")
    params = {"measurements": meas or ["area", "mean", "min", "max"], "limit_to_threshold": o.flag("limit"), "decimals": int(o.num("decimal", 3))}
    for flag, key in _SET_MEASUREMENT_FLAGS.items():  # ImageJ: display / nan / scientific / add / invert check-boxes
        params[key] = o.flag(flag)
    return ij._run("analyze.set_measurements", params)


@cmd("Analyze Particles...", lumen="analyze.particles", notes="size=min-max [pixel] circularity=min-max show= display clear summarize add exclude include stack; like ImageJ it needs a thresholded or 8-bit binary image (auto_threshold stays off)")
def _analyze_particles(ij: "IJ", o: Options):
    img = ij.img
    lo, hi = o.rng("size", (0.0, float("inf")))
    if o.flag("pixel"):
        f = img.calibration.pixel_width * img.calibration.pixel_height
        lo, hi = lo * f, hi * f
    clo, chi = o.rng("circularity", (0.0, 1.0))
    show = o.str("show", "Nothing")
    show = {"nothing": "Nothing", "outlines": "Outlines", "bare outlines": "Bare Outlines", "masks": "Masks", "count masks": "Count Masks", "ellipses": "Ellipses", "overlay": "Nothing", "overlay masks": "Nothing",
            "[overlay]": "Nothing", "labels": "Labels"}.get(show.lower(), "Nothing")
    p = {"min_size": lo, "max_size": 0.0 if not math.isfinite(hi) else hi, "min_circularity": clo, "max_circularity": min(chi, 1.0), "show": show, "display_results": o.flag("display"),
         "clear_results": o.flag("clear"), "summarize": o.flag("summarize"), "add_to_manager": o.flag("add"), "exclude_on_edges": o.flag("exclude"), "include_holes": o.flag("include"), "all_planes": o.flag("stack"),
         "record_starts": o.flag("record"), "composite_rois": o.flag("composite"), "in_situ_show": o.flag("in_situ")}
    return ij._run("analyze.particles", p)


@cmd("Histogram", lumen="analyze.histogram / analyze.stack_histogram", notes="bins=; 'stack' → whole stack")
def _histogram(ij: "IJ", o: Options):
    if o.flag("stack"):
        return ij._run("analyze.stack_histogram", {"bins": int(o.num("bins", 256))})
    return ij._run("analyze.histogram", {"bins": int(o.num("bins", 256))})


@cmd("Image Calculator...", lumen="process.calculator", notes="image1= operation= image2= create 32-bit")
def _calculator(ij: "IJ", o: Options):
    op_name = o.str("operation", "Subtract")
    for k in ("create", "32-bit", "stack"):
        op_name = op_name.replace(k, "").strip()
    a = ij._image(o.str("image1")) if o.str("image1") else ij.img
    b = ij._image(o.str("image2")) if o.str("image2") else ij.img
    create = o.flag("create")
    # ImageJ: without 'create' the result replaces image1 (not the active image), so run the op on image1
    out = ij._run("process.calculator", {"image1": a.id, "operation": op_name.capitalize() if op_name.lower() not in ("and", "or", "xor") else op_name.upper(), "image2": b.id,
                                         "create_new": create, "float_result": o.flag("32-bit")}, image_id=a.id)
    if not create:
        ij.session.select(a.id)  # ImageJ makes image1 the current window after an in-place calculation
    return out


@cmd("Merge Channels...", lumen="color.merge", notes="c1= … c7= create keep")
def _merge(ij: "IJ", o: Options):
    p: dict[str, Any] = {"create_composite": o.flag("create"), "keep_source": o.flag("keep")}
    for k in range(1, 8):
        v = o.str(f"c{k}")
        if v and v != "*None*":
            p[f"c{k}"] = ij._image(v).id
    return ij._run("color.merge", p)


@cmd("Make Composite", lumen="display mode", notes="display=Composite/Color/Grayscale → view mode")
def _make_composite(ij: "IJ", o: Options):
    img = ij.img
    mode = {"composite": "composite", "color": "single", "grayscale": "grayscale"}.get(o.str("display", "Composite").lower(), "composite")
    ij.session.update_display(img.id, {"mode": mode})
    ij.session._changed("images")
    return {"mode": mode}


@cmd("Make Substack...", lumen="stacks.substack", notes="slices= channels= frames= (all given ranges apply together, like ImageJ's SubHyperstackMaker; slices= means frames on a time-lapse); delete (single range only)")
def _substack(ij: "IJ", o: Options):
    from .ops.stacks import _label_grid, _parse_range, _store_labels
    img = ij.img
    z_axis = "T" if (img.nz == 1 and img.nt > 1) else "Z"  # ImageJ: a (c=1, z=1, t>1) image is a plain stack whose slices are the frames
    axes = [(k, ax) for k, ax in (("channels", "C"), ("slices", z_axis), ("frames", "T")) if k in o and o[k] is not True]
    if not axes:
        raise ValueError("Make Substack: give slices=, channels= or frames=")
    key, ax = axes[0]
    # ImageJ offers 'Delete slices from original' only for plain stacks; a hyperstack subset never deletes
    out = ij._run("stacks.substack", {"slices": str(o[key]), "axis": ax, "delete_from_original": o.flag("delete") and len(axes) == 1})
    new = ij.session.images[out["new_images"][0]]
    for key, ax in axes[1:]:
        axi = {"T": 0, "Z": 1, "C": 2}[ax]
        if new.data.shape[axi] <= 1:
            continue
        idx = _parse_range(str(o[key]), new.data.shape[axi])
        grid = _label_grid(new)  # read before the data shrinks: the labels of the kept planes follow them (as stacks.substack does for the first axis)
        if axi == 2 and not new.rgb:
            new.display.channels = [type(new.display.channels[i])(**new.display.channels[i].to_dict()) for i in idx]
        new.replace_data(np.ascontiguousarray(np.take(new.data, idx, axis=axi)), rgb=new.rgb and axi != 2)
        if grid is not None:
            _store_labels(new, np.take(grid, idx, axis=axi))
        new.display.t = new.display.z = new.display.c = 0
    if len(axes) > 1:
        new.name = ij.session.unique_name("Substack (" + " ".join(f"{k}={o[k]}" for k, _ in axes) + ")")
    ij.session._changed("images")
    return out


@cmd("Re-order Hyperstack ...", "Re-order Hyperstack...", lumen="stacks.reorder", notes="channels=[Slices (z)] slices=[Channels (c)] frames=[Frames (t)] → the swapped pair")
def _reorder(ij: "IJ", o: Options):
    def letter(key: str, default: str) -> str:
        m = re.search(r"\((\w)\)", o.str(key, ""))
        return m.group(1).lower() if m else default
    mapping = {"c": letter("channels", "c"), "z": letter("slices", "z"), "t": letter("frames", "t")}
    swapped = {k for k, v in mapping.items() if k != v}
    if not swapped:
        return None
    if swapped == {"z", "t"}:
        order = "Swap Z and T"
    elif swapped == {"z", "c"}:
        order = "Swap Z and C"
    elif swapped == {"c", "t"}:
        order = "Swap C and T"
    else:
        raise ValueError("Re-order Hyperstack: only pairwise swaps are supported")
    return ij._run("stacks.reorder", {"order": order})


@cmd("Insert...", lumen="stacks.insert", notes="source= destination= x= y=")
def _insert(ij: "IJ", o: Options):
    if o.str("destination"):
        ij.selectWindow(o.str("destination"))
    return ij._run("stacks.insert", {"source": ij._image(o.str("source")).id if o.str("source") else None, "x": int(o.num("x", 0)), "y": int(o.num("y", 0))})


@cmd("Add Selection...", lumen="overlay.add / overlay.arrow", notes="stroke= width=; an arrow selection (makeArrow) becomes an arrow item")
def _add_selection(ij: "IJ", o: Options):
    if ij.roi is None:
        raise ValueError("Add Selection: no selection")
    color = _color_hex(o.get("stroke"), "#ffff00")
    if ij.roi.get("arrow"):
        return ij._run("overlay.arrow", {"color": color, "stroke_width": o.num("width", 2.0)})
    return ij._run("overlay.add", {"color": color, "stroke_width": o.num("width", 1.5)})


@cmd("Next Slice [>]", lumen="setSlice(n+1)")
def _next_slice(ij: "IJ", o: Options):
    ij.setSlice(min(ij.getSliceNumber() + 1, ij.nSlices()))


@cmd("Previous Slice [<]", lumen="setSlice(n-1)")
def _prev_slice(ij: "IJ", o: Options):
    ij.setSlice(max(ij.getSliceNumber() - 1, 1))


@cmd("First Slice", lumen="setSlice(1)")
def _first_slice(ij: "IJ", o: Options):
    ij.setSlice(1)


@cmd("Last Slice", lumen="setSlice(nSlices)")
def _last_slice(ij: "IJ", o: Options):
    ij.setSlice(ij.nSlices())


@cmd("Save", lumen="session.save_image", notes="saves to the image's path (TIFF)")
def _save(ij: "IJ", o: Options):
    img = ij.img
    if not img.path:
        raise ValueError("Save: the image has no file path – use saveAs(format, path)")
    return ij.session.save_image(img.id, img.path)


@cmd("Open...", lumen="session.open_file", notes="path=")
def _open(ij: "IJ", o: Options):
    return ij.open(o.str("path", o.str("open", "")))


@cmd("Tiff...", "PNG...", "Jpeg...", lumen="session.save_image", notes="path= (or save=)")
def _save_as(ij: "IJ", o: Options):
    path = o.str("path", o.str("save", ""))
    if not path:
        raise ValueError("Save: no path given")
    return ij.session.save_image(ij.img.id, path)


@cmd("Brightness/Contrast...", "Window/Level...", "Color Balance...", "Channels Tool...", "Show Overlay", "Hide Overlay", "Tile", "Cascade",
     "Put Behind [tab]", "Show Info...", "Conversions...", "Line Width... ", "Memory & Threads...", "Proxy Settings...", "Compiler...", "Wand Tool...", "Collect Garbage", "Monitor Memory...",
     "ImageJ [NEW]...", "Record...", "Startup...", "Reset...", "Select Window...", "Animation Options...", "Synchronize Windows",
     "Manual Tracking", "3D Viewer", "Capture Screen", "Capture Image", "Help on Menu Item", "Help on Menu Item...", "ROI Manager...", "Threshold Panel", "Control Panel...",
     lumen="(no-op)", notes="interactive dialog / window command: ignored (Manual Tracking → the tracking.manual_* commands, 3D Viewer → the Volume view, Capture Screen → the browser)")
def _noop(ij: "IJ", o: Options):
    return None


_CT_NAMES = {"HSB": ("hue", "saturation", "brightness"), "RGB": ("red", "green", "blue"), "Lab": ("l", "a", "b"), "YUV": ("y", "u", "v")}


@cmd("Color Threshold...", lumen="adjust.color_threshold",
     notes="best effort: space/method=HSB|RGB|Lab|YUV, <component>_min/_max (hue, saturation, brightness / red, green, blue / l, a, b / y, u, v or c1..c3), "
           "<component>_filter=pass|stop (or <component>=stop), output=Mask|Selection|Overlay")
def _color_threshold(ij: "IJ", o: Options):
    space = o.str("space", o.str("method", o.str("color_space", "HSB")))
    low = {k.lower(): k for k in _CT_NAMES}
    space = low.get(space.lower(), "HSB")
    p: dict[str, Any] = {"space": space}
    for k, name in enumerate(_CT_NAMES[space], start=1):
        for key in (name, f"c{k}"):
            if f"{key}_min" in o:
                p[f"c{k}_min"] = o.num(f"{key}_min", 0)
            if f"{key}_max" in o:
                p[f"c{k}_max"] = o.num(f"{key}_max", 255)
            if f"{key}_filter" in o:
                p[f"c{k}_filter"] = "stop" if o.str(f"{key}_filter", "pass").lower().startswith("stop") else "pass"
            elif o.str(key, "").lower().startswith("stop"):
                p[f"c{k}_filter"] = "stop"
        if f"{name}_stop" in o or f"c{k}_stop" in o:
            p[f"c{k}_filter"] = "stop"
    out = o.str("output", o.str("macro", "Mask"))
    for choice in ("Mask", "Selection", "Overlay"):
        if out.lower().startswith(choice.lower()):
            p["output"] = choice
    return ij._run("adjust.color_threshold", p, scope="plane")


@cmd("Fit Spline", lumen="selection.interpolate", notes="smooth=True; interval → interval (default 5 px)")
def _fit_spline(ij: "IJ", o: Options):
    if ij.roi is None:
        raise ValueError("Fit Spline: selection required")
    p: dict[str, Any] = {"smooth": True}
    if "interval" in o:
        p["interval"] = o.num("interval", 5.0)
    return ij._run("selection.interpolate", p, scope="plane")


@cmd("Image Sequence...", lumen="file.import_sequence",
     notes="open=[folder or first file] / dir=[folder] → folder, file= / filter= → pattern, number → count, starting → start, increment → step, scale, use → virtual")
def _image_sequence(ij: "IJ", o: Options):
    folder = o.str("dir", o.str("open", o.str("path", o.str("folder", ""))))
    if not folder:
        raise ValueError("Image Sequence: no folder given (use open=[folder] or dir=[folder])")
    folder = os.path.expanduser(folder)
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    p: dict[str, Any] = {"folder": folder}
    pattern = o.str("file", o.str("filter", ""))
    if pattern:
        p["pattern"] = pattern
    if "number" in o:
        p["count"] = int(o.num("number", 0))
    if "starting" in o:
        p["start"] = int(o.num("starting", 1))
    if "increment" in o:
        p["step"] = int(o.num("increment", 1))
    if "scale" in o:
        p["scale"] = o.num("scale", 100.0)
    if o.flag("use"):
        p["virtual"] = True
    if o.flag("sort"):
        p["sort"] = "name"
    return ij._run("file.import_sequence", p)


# ====================================================================== Edit ▸ clipboard / selection / options
def _image_param(ij: "IJ", o: Options, *keys: str) -> str | None:
    """Image id for an ``image=`` style option (title, ImageJ id or Lumen id); None when the option is absent."""
    for k in keys:
        v = o.str(k, "")
        if v and v not in ("*None*", "<None>", "None"):
            return ij._image(v).id
    return None


def _paste_mode(name: Any) -> str:
    from .ops.edit_extra import PASTE_MODES
    n = str(name or "Copy").strip()
    low = {m.lower(): m for m in PASTE_MODES}
    if n.lower() in low:
        return low[n.lower()]
    if n.lower() in ("transparent-zero", "transparent zero", "transparent_zero"):
        return "Transparent-zero"
    if n.lower() in ("transparent-white", "transparent", "transparent white", "transparent_white"):
        return "Transparent-white"
    raise ValueError(f"Paste Control: unknown transfer mode {n!r} (one of {', '.join(PASTE_MODES)})")


simple("Cut", lumen="edit.cut", stack_flag=False, notes="all channels of the current plane; the selection is cleared to the background value")
simple("Copy", lumen="edit.copy", stack_flag=False, notes="internal clipboard")
simple("Paste", lumen="edit.paste_with_mode", params={"mode": const("Current setting")}, stack_flag=False, notes="at the selection or centred; honours the Paste Control transfer mode (setPasteMode)")


@cmd("Paste Control...", lumen="edit.paste_control", notes="transfer=Copy/Blend/Average/Difference/Transparent-white/Transparent-zero/AND/OR/XOR/Add/Subtract/Multiply/Divide/Min/Max (all 15 ImageJ transfer modes, Transparent-white included)")
def _paste_control(ij: "IJ", o: Options):
    return ij._run("edit.paste_control", {"mode": _paste_mode(o.str("transfer", "Copy"))})


simple("Internal Clipboard", lumen="file.new_from_clipboard", stack_flag=False, notes="File ▸ New ▸ Internal Clipboard: the pixels copied with Copy / Cut become a new image")
simple("Fit Circle", lumen="selection.fit_circle", stack_flag=False, notes="Edit ▸ Selection ▸ Fit Circle (no options)")
simple("Fit Rectangle", lumen="selection.fit_rectangle", stack_flag=False, notes="Edit ▸ Selection ▸ Fit Rectangle: minimum-area rotated rectangle")
simple("Select Bounding Box", "Select Bounding Box (guess background color)", lumen="selection.select_bounding_box", stack_flag=False, notes="Fiji Edit ▸ Selection ▸ Select Bounding Box; the background is guessed from the image border")


@cmd("Image to Selection...", lumen="selection.image_to_selection", notes="image= (title of an open mask / image) opacity=")
def _image_to_selection(ij: "IJ", o: Options):
    p: dict[str, Any] = {"opacity": int(o.num("opacity", 75))}
    iid = _image_param(ij, o, "image")
    if iid is None:
        others = [im for im in ij.session.images.values() if im.id != ij.img.id]
        if not others:
            raise ValueError("Image to Selection: open the image to convert first (image=)")
        iid = others[-1].id
    p["image"] = iid
    return ij._run("selection.image_to_selection", p, scope="plane")


@cmd("Restore Selection", lumen="ij.last_roi", notes="re-activates the previous selection (the one removed by Select None, Fill, Measure of another image …); frontend-only in ImageJ")
def _restore_selection(ij: "IJ", o: Options):
    ij.restoreSelection()
    return {"roi": ij.roi}


def _roi_position(ij: "IJ", value: str) -> str:
    """ImageJ 'position=' (none, a stack index or 'c,z,t') → Lumen 't,z,c' (1-based)."""
    v = value.strip().lower()
    if v in ("", "none", "0"):
        return "none" if v else ""
    parts = [p for p in re.split(r"[,\s]+", v) if p]
    if len(parts) == 3:
        c, z, t = (int(float(x)) for x in parts)
        return f"{t},{z},{c}"
    img = ij.img
    n = int(float(parts[0]))
    nc = 1 if img.rgb else img.nc
    i = max(n - 1, 0)
    return f"{i // (nc * img.nz) + 1},{(i // nc) % img.nz + 1},{i % nc + 1}"


@cmd("Properties... ", lumen="selection.properties", notes="Edit ▸ Selection ▸ Properties (trailing space, as in ImageJ): name= stroke= (colour) width= fill= position=none|n|c,z,t; group= is stored on the ROI")
def _roi_properties(ij: "IJ", o: Options):
    if ij.roi is None:
        raise ValueError("Properties: selection required")
    p: dict[str, Any] = {}
    if o.str("name"):
        p["name"] = o.str("name")
    if o.str("stroke"):
        p["color"] = _color_hex(o.get("stroke"), "#ffff00")
    if "width" in o and o.num("width", 0) > 0:
        p["stroke_width"] = o.num("width")
    if o.str("fill"):
        p["fill"] = "none" if o.str("fill").lower() in ("none", "no", "-") else _color_hex(o.get("fill"), "#ffff00")
    if o.str("position"):
        p["position"] = _roi_position(ij, o.str("position"))
    out = ij._run("selection.properties", p, scope="plane")
    if "group" in o and ij.roi is not None:
        r = dict(ij.roi); r["group"] = int(o.num("group", 0)); ij.roi = r
    return out


_ROI_KEYS = ("name", "stroke", "fill", "width", "position", "group")


def _prefs(ij: "IJ", op_id: str, given: dict[str, Any]) -> dict[str, Any] | None:
    """Run an Edit ▸ Options op with only the given keys changed: the Edit ▸ Options ops write just the keys they are passed
    (None = keep), and the other parameters are sent with their current session values, so nothing else moves."""
    given = {k: v for k, v in given.items() if v is not None}
    if not given:
        return None
    spec = REGISTRY.get(op_id)
    p = {q.name: ij.session.settings.get(q.name, q.default) for q in spec.params}
    p.update(given)
    return ij._run(op_id, p)


_POINT_SIZES = {"tiny": "1", "small": "3", "medium": "4", "large": "6", "extra large": "8", "xxl": "12", "xxxl": "16"}
_POINT_TYPES = {"hybrid": "Hybrid", "cross": "Cross", "crosshair": "Cross", "dot": "Dot", "circle": "Circle"}
_ARROW_STYLES = {"filled": "Filled", "notched": "Notched", "open": "Open", "headless": "Headless", "bar": "Bar"}
_FONT_FAMILIES = {"sansserif": "SansSerif", "serif": "Serif", "monospaced": "Monospaced", "monospace": "monospace", "sans-serif": "sans-serif", "dialog": "SansSerif"}


@cmd("Colors...", lumen="setForegroundColor / setBackgroundColor / edit.options_roi_defaults", notes="Edit ▸ Options ▸ Colors: foreground= background= selection= (colour names)")
def _colors_dialog(ij: "IJ", o: Options):
    if o.str("foreground"):
        ij.setForegroundColor(o.str("foreground"))
    if o.str("background"):
        ij.setBackgroundColor(o.str("background"))
    if o.str("selection"):
        return _prefs(ij, "edit.options_roi_defaults", {"roi_color": _color_hex(o.get("selection"), "#ffff00")})
    return None


@cmd("Input/Output...", lumen="edit.options_io", notes="Edit ▸ Options ▸ Input/Output: jpeg= → jpeg_quality (gif= file= use_file copy_column copy_row save_column save_row are ignored)")
def _io_dialog(ij: "IJ", o: Options):
    return _prefs(ij, "edit.options_io", {"jpeg_quality": int(o.num("jpeg")) if "jpeg" in o else None})


@cmd("Point Tool...", lumen="edit.options_point_tool", notes="Edit ▸ Options ▸ Point Tool: type=Hybrid|Cross|Dot|Circle color= size=Tiny|Small|Medium|Large|[Extra Large]|XXL|XXXL (label show counter= auto-measure auto-next add are ignored)")
def _point_tool_dialog(ij: "IJ", o: Options):
    return _prefs(ij, "edit.options_point_tool", {"point_type": _POINT_TYPES.get(o.str("type", "").lower()) if o.str("type") else None,
                                                   "point_color": _color_hex(o.get("color"), "#ffff00") if o.str("color") else None,
                                                   "point_size": _POINT_SIZES.get(o.str("size", "").lower(), None) if o.str("size") else None})


@cmd("Arrow Tool...", lumen="edit.options_arrow_tool", notes="Edit ▸ Options ▸ Arrow Tool: width= size= (head size) style=Filled|Notched|Open|Headless|Bar double; color= outline keep are ignored")
def _arrow_tool_dialog(ij: "IJ", o: Options):
    if not o:
        return None
    return _prefs(ij, "edit.options_arrow_tool", {"arrow_width": o.num("width") if "width" in o else None, "arrow_head_size": o.num("size") if "size" in o else None,
                                                   "arrow_style": _ARROW_STYLES.get(o.str("style", "").lower()) if o.str("style") else None, "arrow_double": o.flag("double")})


@cmd("Fonts...", lumen="edit.options_fonts", notes="Edit ▸ Options ▸ Fonts is a non-modal panel in ImageJ (recorded as setFont); accepts font= size= style=[bold italic antialiased] antialiased")
def _fonts_dialog(ij: "IJ", o: Options):
    style = o.str("style", "").lower()
    aa = True if (o.flag("antialiased") or "antialiased" in style) else None
    if o.str("font") or "size" in o:
        ij.setFont(o.str("font", "SansSerif"), o.num("size", ij.font_size), style)
    return _prefs(ij, "edit.options_fonts", {"font_family": _FONT_FAMILIES.get(o.str("font", "").lower(), o.str("font")) if o.str("font") else None,
                                              "font_size": int(o.num("size")) if "size" in o else None, "antialiased_text": aa})


@cmd("Appearance...", lumen="edit.options_appearance", notes="Edit ▸ Options ▸ Appearance: interpolate → interpolate_zoom, auto (auto contrast stacks), antialiased; open black no use menu= gui= 16-bit= are ignored")
def _appearance_dialog(ij: "IJ", o: Options):
    if not o:
        return None
    return _prefs(ij, "edit.options_appearance", {"interpolate_zoom": o.flag("interpolate"), "auto_contrast_stacks": o.flag("auto"), "antialiased_text": o.flag("antialiased")})


@cmd("Plots...", "Profile Plot Options...", lumen="edit.options_plots", notes="Edit ▸ Options ▸ Plots: width= height= minimum= maximum= interpolate (font= draw sub-pixel auto-close list vertical fixed are ignored)")
def _plots_dialog(ij: "IJ", o: Options):
    if not o:
        return None
    return _prefs(ij, "edit.options_plots", {"plot_width": int(o.num("width")) if "width" in o else None, "plot_height": int(o.num("height")) if "height" in o else None,
                                              "plot_y_min": o.num("minimum") if "minimum" in o else None, "plot_y_max": o.num("maximum") if "maximum" in o else None,
                                              "plot_interpolate": o.flag("interpolate")})


@cmd("Rounded Rect Tool...", lumen="edit.options_rounded_rect", notes="Edit ▸ Options ▸ Rounded Rect Tool: corner= is the corner *diameter* (→ radius = corner/2), width= stroke width; stroke= fill= colours are ignored")
def _rounded_rect_dialog(ij: "IJ", o: Options):
    return _prefs(ij, "edit.options_rounded_rect", {"rounded_rect_radius": o.num("corner") / 2.0 if "corner" in o else None,
                                                     "rounded_rect_stroke_width": o.num("width") if "width" in o else None})


@cmd("Misc...", lumen="edit.options_misc", notes="Edit ▸ Options ▸ Misc: divide=Infinity|max|NaN|0 → divide_by_zero, require (command key for shortcuts); use hide move run enhanced reverse debug are ignored")
def _misc_dialog(ij: "IJ", o: Options):
    if not o:
        return None
    dv = o.str("divide", "")
    div = {"infinity": "infinity", "max": "max", "nan": "NaN", "0": "0", "zero": "0"}.get(dv.lower()) if dv else None
    return _prefs(ij, "edit.options_misc", {"divide_by_zero": div, "require_control_key": o.flag("require")})


# ====================================================================== File ▸ Import / Save As
def _path_opt(o: Options, *keys: str) -> str:
    for k in keys:
        v = o.str(k, "")
        if v:
            return os.path.expanduser(v)
    return ""


def _require_path(o: Options, command: str, *keys: str) -> str:
    p = _path_opt(o, *keys)
    if not p:
        raise ValueError(f"{command}: no file given (use {keys[0]}=[path])")
    return p


simple("Open Next", lumen="file.open_next", params={"close_current": const(True)}, stack_flag=False, notes="File ▸ Open Next: the next image file of the current image's folder replaces it")


@cmd("LUT... ", "LUT...", lumen="file.import_lut / file.export_lut", notes="open=[file.lut] → File ▸ Import ▸ LUT (ImageJ: 'LUT... ' with a trailing space); save=[file.lut] → File ▸ Save As ▸ LUT")
def _lut_file(ij: "IJ", o: Options, command: str = ""):
    if "open" in o:
        return ij._run("file.import_lut", {"path": _require_path(o, command, "open"), "name": o.str("name", ""), "apply": ij.session.active is not None})
    return ij._run("file.export_lut", {"path": _require_path(o, command, "save", "path"), "lut": o.str("lut", "")})


@cmd("Text File... ", "Text File...", lumen="file.import_text", notes="File ▸ Import ▸ Text File: open=[file.txt] → table + Log")
def _text_file(ij: "IJ", o: Options, command: str = ""):
    return ij._run("file.import_text", {"path": _require_path(o, command, "open", "path"), "to_log": True})


def _results_format(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {".xlsx": "xlsx", ".tsv": "tsv", ".txt": "tsv"}.get(ext, "csv")


@cmd("Results... ", "Results...", lumen="file.import_results / file.export_results", notes="open=[file.csv] → File ▸ Import ▸ Results (replaces the table); save=[file.csv|.tsv|.xlsx] → File ▸ Save As ▸ Results")
def _results_file(ij: "IJ", o: Options, command: str = ""):
    if "open" in o:
        return ij._run("file.import_results", {"path": _require_path(o, command, "open"), "replace": True})
    p = _require_path(o, command, "save", "path")
    return ij._run("file.export_results", {"path": p, "format": _results_format(p), "table": "Results", "row_numbers": True})


@cmd("Table... ", "Table...", lumen="file.import_table / table.new", notes="File ▸ Import ▸ Table: open=[file.csv] (name= optional); without open= it is File ▸ New ▸ Table: name= headings=[A,B,C]")
def _table_file(ij: "IJ", o: Options, command: str = ""):
    if "open" in o or "path" in o:
        return ij._run("file.import_table", {"path": _require_path(o, command, "open", "path"), "name": o.str("name", "")})
    return ij._run("table.new", {"name": o.str("name", "Table"), "columns": o.str("headings", o.str("columns", ""))})


@cmd("XY Coordinates... ", "XY Coordinates...", lumen="file.import_xy_coordinates / file.save_xy_coordinates", notes="open=[file.txt] → File ▸ Import ▸ XY Coordinates (polygon selection; type=polyline|point); save=[file] → File ▸ Save As ▸ XY Coordinates")
def _xy_file(ij: "IJ", o: Options, command: str = ""):
    if "open" in o:
        return ij._run("file.import_xy_coordinates", {"path": _require_path(o, command, "open"), "type": o.str("type", "polygon"), "invert_y": o.flag("invert")}, scope="plane")
    if ij.roi is None:
        raise ValueError("XY Coordinates: selection required")
    return ij._run("file.save_xy_coordinates", {"path": _require_path(o, command, "save", "path"), "invert_y": o.flag("invert")}, scope="plane")


@cmd("Stack From List...", lumen="file.import_stack_from_list", notes="File ▸ Import ▸ Stack From List: open=[list.txt] (use → virtual stack is ignored)")
def _stack_from_list(ij: "IJ", o: Options, command: str = ""):
    return ij._run("file.import_stack_from_list", {"path": _require_path(o, command, "open", "path"), "title": o.str("title", "")})


@cmd("ZIP...", lumen="file.export_zip", notes="File ▸ Save As ▸ ZIP: save=[file.zip] (an ImageJ TIFF inside the archive)")
def _zip_file(ij: "IJ", o: Options, command: str = ""):
    return ij._run("file.export_zip", {"path": _require_path(o, command, "save", "path")})


@cmd("Text...", lumen="file.export_text", notes="File ▸ Save As ▸ Text: save=[file.txt]; saves the Results table when it has rows, else the Log (source=Results|Table|Log|Macro to choose)")
def _text_save(ij: "IJ", o: Options, command: str = ""):
    src = o.str("source", "")
    if not src:
        src = "Results" if ij.session.results.rows else "Log"
    return ij._run("file.export_text", {"path": _require_path(o, command, "save", "path"), "source": src, "table": o.str("table", ""), "row_numbers": True})


_RAW_TYPES = {"8-bit": "8-bit", "16-bit signed": "16-bit signed", "16-bit unsigned": "16-bit unsigned", "32-bit signed": "32-bit signed", "32-bit unsigned": "32-bit unsigned",
              "32-bit real": "32-bit real", "64-bit real": "64-bit real", "24-bit rgb": "24-bit RGB", "1-bit bitmap": "1-bit bitmap"}


@cmd("Raw...", lumen="file.import_raw", notes="File ▸ Import ▸ Raw: open= image=[16-bit Unsigned] width= height= offset= number= gap= little-endian white open (all files) use")
def _raw_import(ij: "IJ", o: Options, command: str = ""):
    t = o.str("image", "8-bit").lower()
    if t not in _RAW_TYPES:
        raise ValueError(f"Raw: unsupported image type {o.str('image')!r} (one of {', '.join(_RAW_TYPES.values())})")
    return ij._run("file.import_raw", {"path": _require_path(o, command, "open", "path"), "image_type": _RAW_TYPES[t], "width": int(o.num("width", 512)), "height": int(o.num("height", 512)),
                                       "offset": int(o.num("offset", 0)), "n_images": int(o.num("number", 1)), "gap": int(o.num("gap", 0)), "little_endian": o.flag("little-endian"),
                                       "white_is_zero": o.flag("white"), "open_all": o.flag("open")})


@cmd("Text Image... ", "Text Image...", lumen="file.import_text_image", notes="File ▸ Import ▸ Text Image: open=[file.txt] (saving a text image is not available)")
def _text_image(ij: "IJ", o: Options, command: str = ""):
    if "open" not in o:
        raise ValueError("Text Image: only the import (open=[file]) is available in Lumen")
    return ij._run("file.import_text_image", {"path": _require_path(o, command, "open")})


@cmd("URL...", lumen="file.open_url", notes="File ▸ Import ▸ URL: url=")
def _url(ij: "IJ", o: Options, command: str = ""):
    return ij._run("file.open_url", {"url": o.str("url", o.str("open", ""))})


_SEQ_FORMATS = {"tiff": "tif", "tif": "tif", "png": "png", "jpeg": "jpg", "jpg": "jpg", "bmp": "bmp", "pgm": "pgm", "gif": "png", "zip": "tif", "raw": "npy", "text image": "npy"}


@cmd("Image Sequence... ", lumen="file.export_sequence", notes="File ▸ Save As ▸ Image Sequence (trailing space): dir=[folder] format=TIFF|PNG|JPEG|BMP|PGM name= digits= start=; GIF → PNG, ZIP → TIFF")
def _export_sequence(ij: "IJ", o: Options, command: str = ""):
    folder = _require_path(o, command, "dir", "save", "path", "folder")
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    fmt = _SEQ_FORMATS.get(o.str("format", "TIFF").lower(), "tif")
    p: dict[str, Any] = {"folder": folder, "format": fmt, "start_index": int(o.num("start", 1))}
    name = o.str("name", "")
    if name:
        digits = int(o.num("digits", 4))
        img = ij.img
        axis = "t" if (img.nz == 1 and img.nt > 1) else "z"
        p["name_pattern"] = f"{name}{{{axis}:0{digits}d}}"
    return ij._run("file.export_sequence", p)


@cmd("AVI... ", lumen="file.export_video", notes="File ▸ Save As ▸ AVI (trailing space): save=[file.avi] frame= (fps); compression= is ignored; needs the optional [video] extra")
def _export_avi(ij: "IJ", o: Options, command: str = ""):
    p = _require_path(o, command, "save", "path")
    ext = os.path.splitext(p)[1].lower().lstrip(".") or "avi"
    return ij._run("file.export_video", {"path": p, "format": ext if ext in ("mp4", "avi", "gif", "mov", "mkv", "webm") else "avi", "fps": o.num("frame", 10.0)})


# ====================================================================== Image
@cmd("8-bit Color", lumen="type.8bit_color / type.8bit", notes="number= colours (median cut) for RGB images; grayscale images are converted to 8-bit")
def _8bit_color(ij: "IJ", o: Options):
    if ij.img.rgb:
        return ij._run("type.8bit_color", {"colors": int(o.num("number", 256))})
    return ij._run("type.8bit", {})


simple("Auto Crop", "Auto Crop (guess background color)", lumen="image.auto_crop", stack_flag=False, notes="Fiji Image ▸ Crop ▸ Auto Crop: trims to the bounding box of everything that differs from the border colour")
simple("Show LUT", lumen="color.show_lut", stack_flag=False, notes="Image ▸ Color ▸ Show LUT (no options)")
simple("Display LUTs", lumen="color.display_luts", stack_flag=False, notes="Image ▸ Color ▸ Display LUTs (no options)")


@cmd("Edit LUT...", lumen="color.edit_lut", notes="interactive in ImageJ (no options → nothing happens); name= start= end= or table=[r g b lines] build and apply a LUT")
def _edit_lut(ij: "IJ", o: Options):
    if not o:
        return None
    p: dict[str, Any] = {"name": o.str("name", "Custom"), "apply": not o.flag("no_apply")}
    if o.str("table"):
        p["table"] = o.str("table").replace(";", "\n")
    if o.str("start"):
        p["start"] = _color_hex(o.get("start"), "#000000")
    if o.str("end"):
        p["end"] = _color_hex(o.get("end"), "#ffffff")
    return ij._run("color.edit_lut", p)


simple("Montage to Stack...", lumen="stacks.montage_to_stack", params={"columns": lambda o: int(o.num("columns", o.num("images_per_row", 2))), "rows": lambda o: int(o.num("rows", o.num("images_per_column", 2))),
                                                                       "border": num("border", 0)}, stack_flag=False, notes="Image ▸ Stacks ▸ Tools ▸ Montage to Stack: columns= rows= (ImageJ's StackMaker: images_per_row= images_per_column=) border=")
simple("Remove Slice Labels", lumen="stacks.remove_slice_labels", stack_flag=False, notes="Image ▸ Stacks ▸ Tools ▸ Remove Slice Labels")
simple("Set Label...", lumen="stacks.set_slice_label", params={"label": text("label", "")}, stack_flag=False, notes="Image ▸ Stacks ▸ Tools ▸ Set Label: label= → the current plane's label")
simple("Statistics", lumen="stacks.statistics", stack_flag=False, notes="Image ▸ Stacks ▸ Statistics: whole-stack voxel statistics (table 'Stack Statistics' + Results row)")


@cmd("Radial Reslice", lumen="stacks.radial_reslice", notes="Fiji Image ▸ Stacks ▸ Radial Reslice: angle= degrees_per_slice= (→ n_angles = angle/degrees_per_slice); direction= and rotate are ignored (slices run counter-clockwise from the line / point selection)")
def _radial_reslice(ij: "IJ", o: Options):
    angle = o.num("angle", 360.0)
    step = o.num("degrees_per_slice", 1.0) or 1.0
    return ij._run("stacks.radial_reslice", {"n_angles": max(int(round(angle / step)), 1), "total_rotation": min(max(angle, 1.0), 360.0), "interpolate_z": not o.flag("avoid")})


@cmd("Reslice [/]...", "Reslice [/]... ", "Reslice...", lumen="stacks.reslice / stacks.reslice_line", notes="output= (spacing) start=Top|Bottom|Left|Right avoid (no Z interpolation) flip rotate; with a line / polyline selection: slice_count= rotate (ImageJ Slicer)")
def _reslice(ij: "IJ", o: Options):
    p: dict[str, Any] = {"interpolate_z": not o.flag("avoid"), "output_spacing": o.num("output", 1.0) or 1.0, "slice_count": max(int(o.num("slice_count", 1)), 1), "rotate": o.flag("rotate")}
    if ij.roi is not None and roimod.is_line(ij.roi):
        out = ij._run("stacks.reslice_line", p, scope="plane")
    else:
        p["direction"] = "Top (XZ)" if o.str("start", "Top") in ("Top", "Bottom") else "Left (YZ)"
        out = ij._run("stacks.reslice", p, scope="plane")
    if o.flag("flip") and out.get("new_images"):
        ij._run("image.flip_v", {}, image_id=out["new_images"][0])
    return out


@cmd("Hyperstack to Stack", lumen="stacks.hyperstack_to_stack", notes="flattens channels/slices/frames into one Z stack (ImageJ order c, z, t) with 'c:… z:… t:…' slice labels; a plain stack or RGB image is left as it is")
def _hyper_to_stack(ij: "IJ", o: Options):
    img = ij.img
    if img.rgb or not (img.nc > 1 or (img.nz > 1 and img.nt > 1)):
        return None
    return ij._run("stacks.hyperstack_to_stack", {})


@cmd("Reduce Dimensionality...", lumen="stacks.reduce_dimensionality", notes="channels slices frames keep (ImageJ HyperStackReducer: a missing keyword drops that dimension to the current position; keep = keep the source)")
def _reduce_dimensionality(ij: "IJ", o: Options):
    img = ij.img
    return ij._run("stacks.reduce_dimensionality", {"keep_channels": o.flag("channels") or img.nc == 1 or img.rgb, "keep_slices": o.flag("slices") or img.nz == 1,
                                                    "keep_frames": o.flag("frames") or img.nt == 1, "keep_source": o.flag("keep")})


simple("Make Subset...", lumen="stacks.make_subset", params={"channels": text("channels", ""), "slices": text("slices", ""), "frames": text("frames", "")}, stack_flag=False,
       notes="ImageJ SubHyperstackMaker: channels=1-2 slices=1-5-2 frames=1,3 (all ranges apply together)")
simple("Image to Results", lumen="image.to_results", stack_flag=False, notes="Image ▸ Transform ▸ Image to Results: pixel values of the current plane (or selection bounds) → Results")
simple("Results to Image", lumen="image.from_results", stack_flag=False, notes="Image ▸ Transform ▸ Results to Image: numeric Results columns → 32-bit image")
_LANDMARK_MODELS = {"Interactive Affine": "Affine", "Interactive Rigid": "Rigid body", "Interactive Similarity": "Similarity", "Interactive Perspective": "Projective"}


@cmd(*_LANDMARK_MODELS, lumen="image.transform_landmarks", notes="Fiji's interactive landmark transforms have no macro options; here source=[x1,y1; x2,y2; …] (or the point selection / source_roi=) target=[…] (or target_roi=) interpolation= create")
def _interactive_transform(ij: "IJ", o: Options, command: str = ""):
    p: dict[str, Any] = {"model": _LANDMARK_MODELS[command], "source_points": o.str("source", o.str("source_points", "")), "target_points": o.str("target", o.str("target_points", "")),
                         "source_roi": o.str("source_roi", ""), "target_roi": o.str("target_roi", ""), "interpolation": o.str("interpolation", "Linear"), "create_new": o.flag("create") or not o.flag("in_place")}
    return ij._run("image.transform_landmarks", p, scope="plane")


@cmd("Polar Transformer", lumen="image.warp_polar", notes="Fiji Polar Transformer: method=Polar (Cartesian is not available) number= (angle samples) default_center | polar=[x y] / center_x= center_y= radius= log; degrees= is ignored (always 360°)")
def _polar_transformer(ij: "IJ", o: Options):
    if o.str("method", "Polar").lower().startswith("cart"):
        raise ValueError("Polar Transformer: only the polar (not the Cartesian) transform is available in Lumen")
    p: dict[str, Any] = {"scaling": "log" if o.flag("log") else "linear", "angles": int(o.num("number", 360)), "radius": o.num("radius", 0), "interpolation": o.str("interpolation", "Linear")}
    centre = o.str("polar", "")
    if centre and not o.flag("default_center"):
        bits = [b for b in re.split(r"[,\s]+", centre) if b]
        if len(bits) >= 2:
            p["center_x"], p["center_y"] = float(bits[0]), float(bits[1])
    for k in ("center_x", "center_y"):
        if k in o:
            p[k] = o.num(k)
    return ij._run("image.warp_polar", p, scope="plane", use_roi=not o.flag("default_center"))  # default_center → the image centre even with a selection


@cmd("Add Image...", lumen="overlay.add_image", notes="Image ▸ Overlay ▸ Add Image: image= x= y= opacity= (zero / transparent background is ignored)")
def _add_image(ij: "IJ", o: Options):
    p: dict[str, Any] = {"x": o.num("x", 0.0), "y": o.num("y", 0.0), "opacity": o.num("opacity", 100.0)}
    iid = _image_param(ij, o, "image")
    if iid is not None:
        p["image"] = iid
    return ij._run("overlay.add_image", p, scope="plane")


_GRID_TYPES = {"lines": "Lines", "horizontal lines": "Horizontal Lines", "crosses": "Crosses", "points": "Points", "combined": "Combined"}


@cmd("Grid...", lumen="overlay.grid", notes="Analyze ▸ Tools ▸ Grid: grid=Lines|[Horizontal Lines]|Crosses|Points|Combined|None area= color= bold random (center is ignored); None removes the grid")
def _grid(ij: "IJ", o: Options):
    kind = o.str("grid", "Lines")
    if kind.lower() == "none":
        img = ij.img
        img.overlay = [it for it in img.overlay if not it.get("grid")]
        ij.session._changed("images")
        return None
    p: dict[str, Any] = {"type": _GRID_TYPES.get(kind.lower(), "Lines"), "area_per_point": o.num("area", 0.0), "random_offset": o.flag("random"), "stroke_width": 2.0 if o.flag("bold") else 0.0}
    if o.str("color"):
        p["color"] = _color_hex(o.get("color"), "#00ffff")
    return ij._run("overlay.grid", p, scope="plane")


# ====================================================================== Process
simple("Dither", lumen="binary.dither", notes="Floyd–Steinberg dithering of the current plane ('stack' → all planes)")
simple("Show Circular Masks", lumen="filters.show_circular_masks", stack_flag=False, notes="Process ▸ Filters ▸ Show Circular Masks (no options)")
simple("Normalize Local Contrast", lumen="filters.normalize_local_contrast", params={"block_radius_x": num("block_radius_x", 40), "block_radius_y": num("block_radius_y", 40), "standard_deviations": num("standard_deviations", 3.0),
                                                                                   "center": flag("center"), "stretch": flag("stretch")}, notes="Fiji Process ▸ Filters: block_radius_x= block_radius_y= standard_deviations= center stretch stack")
simple("ROF Denoise", lumen="filters.rof_denoise", params={"theta": num("theta", 25.0)}, notes="Fiji Process ▸ Filters ▸ ROF Denoise: theta=")
simple("Retinex", lumen="process.retinex", params={"level": text("level", "Uniform"), "scale": num("scale", 240.0), "scale_division": num("scale_division", 3), "dynamic": num("dynamic", 1.2)}, stack_flag=False,
       notes="Fiji Process ▸ Retinex: level=Uniform|Low|High scale= scale_division= dynamic=")
simple("Shape Index Map", lumen="filters.shape_index", params={"sigma": lambda o: o.num("sigma", o.num("gaussian", o.num("radius", 2.0)))}, notes="Fiji Process ▸ Shape Index Map: sigma= (gaussian= / radius= accepted)")
simple("Exact Euclidean Distance Transform (3D)", "Exact Signed Euclidean Distance Transform (3D)", lumen="binary.distance_map_3d", stack_flag=False,
       notes="Fiji Process ▸ Binary: 3-D EDT (32-bit, calibrated); the signed variant is approximated by the unsigned transform")


@cmd("Measure...", "Batch Measure", lumen="batch.measure", notes="Process ▸ Batch ▸ Measure: input=[folder] (dir= / folder= / open=) file= (name filter) threshold= (Lumen: auto-threshold before measuring)")
def _batch_measure(ij: "IJ", o: Options, command: str = ""):
    folder = _require_path(o, command, "input", "dir", "folder", "open")
    if os.path.isfile(folder):
        folder = os.path.dirname(folder)
    p: dict[str, Any] = {"input_folder": folder, "all_planes": o.flag("stack")}
    if o.str("file"):
        p["pattern"] = re.escape(o.str("file"))
    if o.str("threshold"):
        p["threshold"] = o.str("threshold")
    return ij._run("batch.measure", p)


@cmd("FFT Options...", lumen="process.fft_options", notes="Process ▸ FFT ▸ FFT Options: fft (display 8-bit FFT window) raw (raw power spectrum) complex (complex output) do (do forward transform now); fast (FHT) is ignored, window = Hann window (Lumen)")
def _fft_options(ij: "IJ", o: Options):
    out = ij._run("process.fft_options", {"do_forward": o.flag("fft") or not o, "raw_power_spectrum": o.flag("raw"), "complex_output": o.flag("complex"), "fft_window": o.flag("window")})
    if o.flag("do") and ij.session.active is not None:
        return ij._run("process.fft", {}, scope="plane")
    return out


simple("Redisplay Power Spectrum", lumen="process.fft_redisplay_power_spectrum", stack_flag=False, notes="Process ▸ FFT ▸ Redisplay Power Spectrum (no options)")
simple("Inverse FFT", lumen="process.inverse_fft", params={"output": lambda o: "32-bit" if o.flag("32-bit") else "Original type"}, stack_flag=False,
       notes="Process ▸ FFT ▸ Inverse FFT of a power-spectrum image (black / white painted regions filter); '32-bit' keeps a float result (Lumen)")
simple("Swap Quadrants", lumen="process.fft_swap_quadrants", notes="Process ▸ FFT ▸ Swap Quadrants ('stack' → all planes)")


@cmd("FD Math...", lumen="process.fd_math", notes="Process ▸ FFT ▸ FD Math: image1= operation=Correlate|Convolve|Deconvolve image2= result= do (inverse transform; without it the result stays in the frequency domain)")
def _fd_math(ij: "IJ", o: Options):
    a = _image_param(ij, o, "image1") or ij.img.id
    b = _image_param(ij, o, "image2") or a
    op_name = o.str("operation", "Correlate")
    low = {k.lower(): k for k in ("Correlate", "Convolve", "Deconvolve")}
    return ij._run("process.fd_math", {"image1": a, "operation": low.get(op_name.lower(), "Correlate"), "image2": b, "result": o.str("result", "Result"), "do_inverse": o.flag("do")}, image_id=a)


@cmd("Custom Filter...", lumen="process.fft_custom_filter", notes="Process ▸ FFT ▸ Custom Filter: filter=[filter image title] ('stack' / process → every plane)")
def _custom_filter(ij: "IJ", o: Options):
    fid = _image_param(ij, o, "filter")
    if fid is None:
        raise ValueError("Custom Filter: give the filter image (filter=[title])")
    return ij._run("process.fft_custom_filter", {"filter": fid, "all_planes": o.flag("stack") or o.flag("process")}, scope="plane")


# ====================================================================== Analyze
simple("Analyze Line Graph", lumen="analyze.line_graph", stack_flag=False, notes="Analyze ▸ Tools ▸ Analyze Line Graph (no options); a rectangle selection limits the plot area")


@cmd("Directionality", lumen="analyze.directionality", notes="Fiji Analyze ▸ Directionality: method=[Local gradient orientation]|[Fourier components] nbins= histogram_start= (histogram=) ; histogram_end display_table are ignored; stacks are processed plane by plane")
def _directionality(ij: "IJ", o: Options):
    img = ij.img
    m = o.str("method", "Local gradient orientation")
    method = "Fourier components" if m.lower().startswith("fourier") else "Local gradient orientation"
    return ij._run("analyze.directionality", {"method": method, "n_bins": int(o.num("nbins", 90)), "bin_start": o.num("histogram_start", o.num("histogram", -90.0)), "all_planes": img.nz * img.nt > 1}, scope="plane")


def _gel_params(ij: "IJ", o: Options, *keys: str) -> dict[str, Any]:
    g = ij.gel_options
    p: dict[str, Any] = {}
    if "orientation" in keys:
        p["orientation"] = "Horizontal" if o.flag("horizontal") else ("Vertical" if o.flag("vertical") else g.get("orientation", "Vertical"))
    if "dark_bands" in keys:
        p["dark_bands"] = not o.flag("invert") if o else g.get("dark_bands", True)
    if "label" in keys:
        p["label"] = "Percent" if (o.flag("label") or (not o and g.get("label") == "Percent")) else g.get("label", "Area")
    return p


@cmd("Gel Analyzer Options...", lumen="(ij.gel_options)", notes="Analyze ▸ Gels ▸ Gel Analyzer Options: vertical / horizontal (lane direction) invert (peaks) label (with percentages); uncalibrated outline are ignored")
def _gel_options(ij: "IJ", o: Options):
    ij.gel_options = {"orientation": "Horizontal" if o.flag("horizontal") else "Vertical", "dark_bands": not o.flag("invert"), "label": "Percent" if o.flag("label") else "Area"}
    return None


@cmd("Select First Lane", "Select Next Lane", lumen="analyze.gel_select_lane", notes="Analyze ▸ Gels: the rectangle selection becomes lane 1 (Select First Lane drops the previous lanes of the image) / the next lane")
def _gel_select_lane(ij: "IJ", o: Options, command: str = ""):
    if command == "Select First Lane":
        from .ops.gel import gel_lanes_of
        img = ij.img
        old = [r["id"] for r in gel_lanes_of(ij.session, img) if r.get("gel_lane") and r.get("id")]
        if old:
            ij.session.roi_manager.remove(old)
            ij.session._changed("rois")
    return ij._run("analyze.gel_select_lane", {}, scope="plane")


@cmd("Plot Lanes", lumen="analyze.gel_plot_lanes", notes="Analyze ▸ Gels ▸ Plot Lanes: the lanes selected with Select First/Next Lane (honours Gel Analyzer Options)")
def _gel_plot_lanes(ij: "IJ", o: Options):
    return ij._run("analyze.gel_plot_lanes", {**_gel_params(ij, o, "orientation", "dark_bands"), "make_image": True}, scope="plane")


@cmd("Label Peaks", lumen="analyze.gel_label_peaks", notes="Analyze ▸ Gels ▸ Label Peaks: band areas (or percentages with 'Label with percentages') as Results rows + labels")
def _gel_label_peaks(ij: "IJ", o: Options):
    return ij._run("analyze.gel_label_peaks", _gel_params(ij, o, "orientation", "dark_bands", "label"), scope="plane")


@cmd("Analyze Skeleton (2D/3D)", "Analyze Skeleton", lumen="analyze.skeleton", notes="Fiji AnalyzeSkeleton: prune=[none|shortest branch|lowest intensity voxel|lowest intensity branch] (intensity variants → shortest branch) prune_0 (prune ends) ; calculate show display exclude are ignored; stacks are analysed plane by plane; skeletonize (Lumen) thins the mask first")
def _analyze_skeleton(ij: "IJ", o: Options):
    img = ij.img
    prune = o.str("prune", "none").lower()
    return ij._run("analyze.skeleton", {"skeletonize_first": o.flag("skeletonize"), "prune_ends": 1e9 if o.flag("prune_0") else 0.0, "prune_cycles": "none" if prune in ("", "none") else "shortest branch",
                                        "show": "Tagged skeleton", "all_planes": img.nz * img.nt > 1}, scope="plane")


@cmd("Coloc 2", lumen="analyze.coloc_costes", notes="Fiji Coloc 2: channel_1= channel_2= (image titles; two images are combined into a temporary 2-channel image, one multi-channel image uses channel_a=/channel_b=) psf= (→ block size) costes_randomisations= ; roi_or_mask= threshold_regression= and the display check-boxes are ignored")
def _coloc2(ij: "IJ", o: Options):
    a = ij._image(o.str("channel_1")) if o.str("channel_1") else ij.img
    b = ij._image(o.str("channel_2")) if o.str("channel_2") else a
    p: dict[str, Any] = {"scramble_iterations": int(o.num("costes_randomisations", 10)), "block_size": max(int(o.num("psf", 3)), 1), "show_scatter": o.flag("2d_intensity_histogram") or not o}
    if a.id == b.id:
        p["channel_a"], p["channel_b"] = int(o.num("channel_a", 1)), int(o.num("channel_b", 2))
        return ij._run("analyze.coloc_costes", p, image_id=a.id)
    if (a.height, a.width) != (b.height, b.width):
        raise ValueError(f"Coloc 2: {a.name} and {b.name} must have the same size")
    pa, pb = a.current_plane(), b.current_plane()
    dtype = np.result_type(pa.dtype, pb.dtype)
    tmp = LumenImage(np.stack([pa.astype(dtype), pb.astype(dtype)])[None, None], name=f"Coloc 2: {a.name} + {b.name}", axes="TZCYX")
    tmp.calibration = Calibration(**a.calibration.to_dict())
    ij.session.add_image(tmp, activate=False)
    try:
        return ij._run("analyze.coloc_costes", {**p, "channel_a": 1, "channel_b": 2}, image_id=tmp.id)
    finally:
        ij.session.close_image(tmp.id)


# ====================================================================== Plugins (Fiji)
def _flow_axis(img: LumenImage) -> str:
    return "T" if (img.nz == 1 and img.nt > 1) else "Z"


@cmd("Elastic Stack Alignment", lumen="registration.elastic_align", notes="Fiji Plugins ▸ Registration ▸ Elastic Stack Alignment: best effort – the slices/frames are aligned onto the previous one with dense optical flow; Fiji's block-matching parameters (layer_scale search_radius block_radius …) have no equivalent and are ignored")
def _elastic_stack_alignment(ij: "IJ", o: Options):
    img = ij.img
    return ij._run("registration.elastic_align", {"axis": _flow_axis(img), "reference": "Previous slice", "create_new": o.flag("create")}, scope="plane")


_BUNWARPJ_MODES = {"fast": "Fast", "accurate": "Accurate", "mono": "Mono"}


@cmd("bUnwarpJ", lumen="registration.bunwarpj",
     notes="Fiji bUnwarpJ: source_image= (made current) target_image= registration=[Fast|Accurate|Mono] image_subsampling_factor= initial_deformation=[Very Coarse|Coarse|Fine|Very Fine] "
           "final_deformation=[Very Coarse|Coarse|Fine|Very Fine|Super Fine] divergence_weight= curl_weight= landmark_weight= image_weight= consistency_weight= stop_threshold= "
           "save_transformations (→ deformation field image) save_direct= save_inverse= (transform files); verbose is ignored")
def _bunwarpj(ij: "IJ", o: Options):
    if o.str("source_image"):
        ij.selectWindow(o.str("source_image"))
    tid = _image_param(ij, o, "target_image", "target")
    if tid is None:
        raise ValueError("bUnwarpJ: give the target image (target_image=)")
    p: dict[str, Any] = {"target": tid, "mode": _BUNWARPJ_MODES.get(o.str("registration", "Fast").strip().lower(), "Fast"), "subsampling": int(o.num("image_subsampling_factor", 0)),
                         "output_field": o.flag("save_transformations"), "store": True}
    for key, default in (("initial_deformation", "Very Coarse"), ("final_deformation", "Fine")):
        if key in o:
            p[key] = _choice("registration.bunwarpj", key, o.str(key), default)
    for key in ("divergence_weight", "curl_weight", "landmark_weight", "image_weight", "consistency_weight", "stop_threshold"):
        if key in o:
            p[key] = o.num(key)
    for key in ("save_direct", "save_inverse"):
        if o.str(key):
            p[key] = _path_opt(o, key)
    return ij._run("registration.bunwarpj", p, scope="plane")


_RVSS_MODES = {"translation": "Translation", "rigid": "Rigid body", "similarity": "Affine", "affine": "Affine", "elastic": "Elastic", "moving": "Elastic"}


@cmd("Register Virtual Stack Slices", lumen="registration.align_folder", notes="Fiji Plugins ▸ Registration ▸ Register Virtual Stack Slices: source=[folder] output=[folder] registration=[Translation|Rigid|Similarity|Affine|Elastic|Moving least squares …] (feature= when registration= is absent; Similarity → Affine, Moving least squares → Elastic); advanced shrinkage save are ignored")
def _register_virtual_stack(ij: "IJ", o: Options, command: str = ""):
    model = (o.str("registration", "") or o.str("feature", "Translation")).strip().split()[0].lower()
    return ij._run("registration.align_folder", {"folder": _require_path(o, command, "source", "input", "dir"), "output_folder": _path_opt(o, "output"), "mode": _RVSS_MODES.get(model, "Translation"),
                                                 "reference": "Previous image"})


@cmd("Phase Correlation", lumen="registration.phase_correlation", notes="Fiji Stitching ▸ Phase Correlation: first_image= (made current) second_image= (target); subpixel_accuracy → 10× sub-pixel, else integer shifts; number_of_peaks= and the fusion options are ignored")
def _phase_correlation(ij: "IJ", o: Options):
    if o.str("first_image") or o.str("image1"):
        ij.selectWindow(o.str("first_image") or o.str("image1"))
    tid = _image_param(ij, o, "second_image", "image2", "target")
    if tid is None:
        others = [im for im in ij.session.images.values() if im.id != ij.img.id]
        if not others:
            raise ValueError("Phase Correlation: open a second image (second_image=)")
        tid = others[-1].id
    return ij._run("registration.phase_correlation", {"target": tid, "upsample": 10 if (o.flag("subpixel_accuracy") or o.flag("subpixel") or not o) else 1, "large_shifts": o.flag("large")}, scope="plane")


@cmd("Pairwise stitching", lumen="registration.pairwise_stitch", notes="Fiji Stitching ▸ Pairwise stitching: first_image= second_image= fusion_method=[Linear Blending|Max. Intensity|Average|Overlay into composite image] fused_image= (title) check_peaks=")
def _pairwise_stitch(ij: "IJ", o: Options):
    a = _image_param(ij, o, "first_image", "image1") or ij.img.id
    b = _image_param(ij, o, "second_image", "image2")
    if b is None:
        raise ValueError("Pairwise stitching: give the second image (second_image=)")
    fm = o.str("fusion_method", "Linear Blending").lower()
    fusion = "Max" if fm.startswith("max") else "Average" if fm.startswith("av") else "Overlay (channels)" if fm.startswith("overlay") else "Linear blend"
    return ij._run("registration.pairwise_stitch", {"image1": a, "image2": b, "fusion": fusion, "check_peaks": int(o.num("check_peaks", 5)), "title": o.str("fused_image", "Stitched")}, image_id=a)


_FEATURE_MODELS = {"translation": "Translation", "rigid": "Rigid body", "similarity": "Similarity", "affine": "Affine", "perspective": "Projective", "projective": "Projective"}


@cmd("Extract SIFT Correspondences", "Extract MOPS Correspondences", lumen="registration.extract_correspondences",
     notes="Fiji Plugins ▸ Feature Extraction: source_image= (made current) target_image= initial_gaussian_blur= steps_per_scale_octave= minimum_image_size= maximum_image_size= feature_descriptor_size= "
           "feature_descriptor_orientation_bins= closest/next_closest_ratio= maximal_alignment_error= inlier_ratio= expected_transformation=[Translation|Rigid|Similarity|Affine|Perspective] → "
           "SIFT / MOPS matches as point ROIs on both images and a Correspondences table; filter is implied")
def _feature_correspondences(ij: "IJ", o: Options, command: str = ""):
    if o.str("source_image"):
        ij.selectWindow(o.str("source_image"))
    tid = _image_param(ij, o, "target_image", "target")
    if tid is None:
        raise ValueError(f"{command}: give the target image (target_image=)")
    p: dict[str, Any] = {"target": tid, "method": "MOPS" if "MOPS" in command else "SIFT", "model": _FEATURE_MODELS.get(o.str("expected_transformation", "Rigid").strip().lower(), "Rigid body"),
                         "initial_blur": o.num("initial_gaussian_blur", 1.6), "steps_per_octave": int(o.num("steps_per_scale_octave", 3)), "min_image_size": int(o.num("minimum_image_size", 64)),
                         "max_image_size": int(o.num("maximum_image_size", 1024)), "descriptor_size": int(o.num("feature_descriptor_size", 0)),
                         "orientation_bins": int(o.num("feature_descriptor_orientation_bins", 8)), "max_ratio": o.num("closest/next_closest_ratio", 0.92),
                         "max_alignment_error": o.num("maximal_alignment_error", 25), "min_inlier_ratio": o.num("inlier_ratio", 0.05), "add_rois": True}
    return ij._run("registration.extract_correspondences", p, scope="plane")


@cmd("Load a TrackMate file", lumen="tracking.import_trackmate_xml", notes="Fiji Plugins ▸ Tracking ▸ Load a TrackMate file: open=[file.xml] → tracks of the current image")
def _load_trackmate(ij: "IJ", o: Options, command: str = ""):
    return ij._run("tracking.import_trackmate_xml", {"path": _require_path(o, command, "open", "path"), "only_filtered": True}, scope="plane")


@cmd("Color Clustering", lumen="segment.color_clustering", notes="Fiji Plugins ▸ Segmentation ▸ Color Clustering is interactive; number_of_clusters= (clusters=) color_space=RGB|Lab|HSB → k-means label image + 'Clusters' table")
def _color_clustering(ij: "IJ", o: Options):
    n = int(o.num("number_of_clusters", o.num("clusters", o.num("n_clusters", 3))))
    return ij._run("segment.color_clustering", {"n_clusters": max(n, 2), "color_space": o.str("color_space", "RGB"), "color_image": o.flag("color_image")}, scope="plane")


simple("About Plugins", "About Plugins...", lumen="help.about_plugins", stack_flag=False, notes="Help ▸ About Plugins: table of the loaded command modules")


# ====================================================================== Bio-Formats (loci.plugins) commands
_BF_COLOR_MODES = {"default": "Default", "composite": "Composite", "colorized": "Colorized", "grayscale": "Grayscale", "custom": "Custom"}
_BF_VIEWS = {"hyperstack": "Hyperstack", "standard imagej": "Standard ImageJ", "data browser": "Data Browser", "image5d": "Hyperstack", "view5d": "Hyperstack",
             "metadata only": "Hyperstack"}
_BF_STACK_ORDERS = {"XYCZT", "XYCTZ", "XYZCT", "XYZTC", "XYTCZ", "XYTZC"}
_BF_COMPRESSIONS = {"uncompressed": "Uncompressed", "none": "Uncompressed", "lzw": "LZW", "zlib": "zlib", "deflate": "zlib", "jpeg": "JPEG",
                    "jpeg-2000": "JPEG-2000", "jpeg-2000 lossy": "JPEG-2000", "jpeg2000": "JPEG-2000", "packbits": "PackBits", "zstd": "Zstd"}


def _bf_opt(o: Options, key: str, default: float | None = None) -> float | None:
    """A Bio-Formats importer option that may carry a series suffix (``c_begin_1``) or none (``c_begin``)."""
    for k in (key, key + "_1"):
        if k in o:
            return o.num(k, default if default is not None else 0)
    for k in o:
        if re.fullmatch(re.escape(key) + r"_\d+", k):
            return o.num(k, default if default is not None else 0)
    return default


def _bf_import_params(o: Options) -> dict[str, Any]:
    """Bio-Formats Importer option string → file.bioformats_import params (series_N flags are 1-based, Lumen's series are 0-based)."""
    p: dict[str, Any] = {}
    sel = sorted(int(m.group(1)) - 1 for k in o for m in [re.fullmatch(r"series_(\d+)", k)] if m and o.flag(k) and int(m.group(1)) >= 1)
    if o.flag("open_all_series"):
        p["open_all_series"] = True
    elif "series_list" in o:
        p["series"] = ",".join(str(max(int(x) - 1, 0)) for x in re.split(r"[,\s]+", o.str("series_list")) if x.strip().isdigit()) or "0"
    elif sel:
        p["series"] = ",".join(str(i) for i in sel)
    p["color_mode"] = _BF_COLOR_MODES.get(o.str("color_mode", "Default").strip().lower(), "Default")
    p["view"] = _BF_VIEWS.get(o.str("view", "Hyperstack").strip().lower(), "Hyperstack")
    so = o.str("stack_order", "Default").strip().upper()
    p["stack_order"] = so if so in _BF_STACK_ORDERS else "Default"
    for key in ("autoscale", "split_channels", "split_focal_planes", "split_timepoints", "group_files", "specify_range", "crop", "display_metadata", "swap_dimensions"):
        p[key] = o.flag(key)
    p["display_ome_xml"] = o.flag("display_ome-xml") or o.flag("display_ome_xml")
    p["virtual"] = o.flag("virtual") or o.flag("use_virtual_stack")
    pattern = o.str("name", o.str("pattern", o.str("group_pattern", "")))
    if pattern and p["group_files"]:
        p["group_pattern"] = pattern
    if p["specify_range"]:
        for ax in "czt":
            for part in ("begin", "end", "step"):
                v = _bf_opt(o, f"{ax}_{part}")
                if v is not None:
                    p[f"{ax}_{part}"] = int(v)
    if p["crop"]:
        for key, opt in (("crop_x", "x_coordinate"), ("crop_y", "y_coordinate"), ("crop_width", "width"), ("crop_height", "height")):
            v = _bf_opt(o, opt)
            if v is not None:
                p[key] = int(v)
    if p["swap_dimensions"]:
        letters = "".join((o.str(f"{ax}_1", o.str(ax, ax.upper())) or ax.upper())[0].upper() for ax in "zct")
        p["swap_order"] = letters
    lvl = _bf_opt(o, "level")
    if lvl is None:
        lvl = _bf_opt(o, "resolution")
    if lvl is not None:
        p["level"] = int(lvl)
    return p


@cmd("Bio-Formats Importer", "Bio-Formats", "Bio-Formats (Windowless)", "Bio-Formats Windowless Importer", lumen="file.bioformats_import",
     notes="open=[file] color_mode=Default|Composite|Colorized|Grayscale|Custom view=Hyperstack|[Standard ImageJ]|[Data Browser] stack_order= series_N (1-based) / "
           "series_list= / open_all_series autoscale split_channels split_focal_planes split_timepoints group_files (name=[pattern]) virtual|use_virtual_stack "
           "specify_range c_begin_1= c_end_1= c_step_1= z_* t_* crop x_coordinate_1= y_coordinate_1= width_1= height_1= display_metadata display_ome-xml "
           "swap_dimensions z_1= c_1= t_1=")
def _bioformats_importer(ij: "IJ", o: Options, command: str = ""):
    path = _require_path(o, command, "open", "path", "file", "id")
    return ij._run("file.bioformats_import", {"path": path, **_bf_import_params(o)})


@cmd("Bio-Formats Remote Importer", lumen="file.bioformats_remote_import", notes="url=[http://…] (or open=) plus the Bio-Formats Importer options")
def _bioformats_remote_importer(ij: "IJ", o: Options, command: str = ""):
    url = o.str("url", o.str("open", o.str("path", "")))
    if not url:
        raise ValueError(f"{command}: no URL given (use url=[http://…])")
    return ij._run("file.bioformats_remote_import", {"url": url, **_bf_import_params(o)})


@cmd("Bio-Formats Exporter", lumen="file.bioformats_export",
     notes="save=[file.ome.tif] compression=Uncompressed|LZW|zlib|JPEG|JPEG-2000|PackBits|Zstd write_each_z_section write_each_timepoint write_each_channel padded; "
           ".zarr → OME-Zarr, any .tif → OME-TIFF")
def _bioformats_exporter(ij: "IJ", o: Options, command: str = ""):
    path = _require_path(o, command, "save", "path", "file")
    comp_key = o.str("compression", "Uncompressed").strip().lower()
    if comp_key not in _BF_COMPRESSIONS:
        raise ValueError(f"{command}: unknown compression {o.str('compression')!r} (Uncompressed, LZW, zlib, JPEG, JPEG-2000, PackBits, Zstd)")
    p: dict[str, Any] = {"path": path, "format": "OME-Zarr" if path.lower().endswith(".zarr") else "OME-TIFF", "compression": _BF_COMPRESSIONS[comp_key],
                         "write_each_z_section": o.flag("write_each_z_section"), "write_each_timepoint": o.flag("write_each_timepoint"),
                         "write_each_channel": o.flag("write_each_channel"), "padded": o.flag("padded")}
    return ij._run("file.bioformats_export", p)


@cmd("TIFF Virtual Stack...", lumen="file.import_tiff_virtual", notes="File ▸ Import ▸ TIFF Virtual Stack: open=[file.tif] (series= level= optional)")
def _tiff_virtual_stack(ij: "IJ", o: Options, command: str = ""):
    path = _require_path(o, command, "open", "path")
    return ij._run("file.import_tiff_virtual", {"path": path, "series": int(o.num("series", 0)), "level": int(o.num("level", 0))})


@cmd("Stack Slicer", lumen="file.stack_slicer", notes="Plugins ▸ Bio-Formats ▸ Stack Slicer: split_channels split_z split_timepoints stack_order= keep_original")
def _stack_slicer(ij: "IJ", o: Options, command: str = ""):
    so = o.str("stack_order", "XYCZT").strip().upper()
    return ij._run("file.stack_slicer", {"split_channels": o.flag("split_channels"), "split_z": o.flag("split_z") or o.flag("split_focal_planes"),
                                         "split_timepoints": o.flag("split_timepoints"), "stack_order": so if so in _BF_STACK_ORDERS else "XYCZT",
                                         "keep_original": o.flag("keep_original")})


@cmd("Bio-Formats Plugins Configuration", lumen="file.bioformats_config", notes="→ the 'Bio-Formats Plugins Configuration' table (formats, readers, availability)")
def _bioformats_plugins_configuration(ij: "IJ", o: Options, command: str = ""):
    return ij._run("file.bioformats_config", {})


@cmd("Bio-Formats Macro Extensions", lumen="Ext.*", notes="enables the Ext.setId / getSeriesCount / setSeries / getSizeX … / openImage macro functions (see IJ.Ext)")
def _bioformats_macro_extensions(ij: "IJ", o: Options, command: str = ""):
    ij.Ext._enable()
    return None


@cmd("AVI...", lumen="session.open_file / stacks.substack", notes="File ▸ Import ▸ AVI: open=[file.avi] first= last= (frame range), convert → 8-bit grayscale; use (virtual) is accepted")
def _import_avi(ij: "IJ", o: Options, command: str = ""):
    path = _require_path(o, command, "open", "path")
    img = ij.open(path)
    first, last = int(o.num("first", 1)), int(o.num("last", 0))
    if img.nt > 1 and (first > 1 or (0 < last < img.nt)):
        ij._run("stacks.substack", {"slices": f"{max(first, 1)}-{last if 0 < last <= img.nt else img.nt}", "axis": "T"}, image_id=img.id)
        ij.session.close_image(img.id)
    if o.flag("convert") and ij.img.rgb:
        ij._run("type.8bit", {})
    return None


# ====================================================================== Selection / Overlay / Utilities (phase 5)
simple("Fit Circle to Image", lumen="selection.fit_circle_to_image", params={"method": "method"}, stack_flag=False,
       notes="Edit ▸ Selection ▸ Fit Circle to Image: circle through the thresholded / non-zero pixels (method= optional)")
simple("Points from Mask", lumen="selection.points_from_mask", params={"step": "step", "max_points": "max_points"}, stack_flag=False,
       notes="Edit ▸ Selection ▸ Points from Mask: step= max_points= → multi-point selection of the mask pixels")


@cmd("Labels...", lumen="analyze.label", notes="Image ▸ Overlay ▸ Labels…: color= font= show use (names as labels) bold draw (label background) → index / name labels of the overlay items")
def _overlay_labels(ij: "IJ", o: Options):
    return ij._run("analyze.label", {"what": "Index", "use_names": o.flag("use"), "font_size": int(o.num("font", 0)), "color": o.str("color", ""), "bold": o.flag("bold"),
                                     "background": o.flag("background") or o.flag("draw"), "source": "Overlay"})


@cmd("Measure Overlay", lumen="overlay.measure", notes="Image ▸ Overlay ▸ Measure Overlay: one Results row per overlay item")
def _measure_overlay(ij: "IJ", o: Options):
    return ij._run("overlay.measure", {}, scope="plane")


@cmd("Overlay Options...", lumen="edit.options_overlay", notes="Image ▸ Overlay ▸ Overlay Options…: width= fill= show (labels) use (names) font= color= background; the stroke colour and 'apply' are ignored")
def _overlay_options(ij: "IJ", o: Options):
    p: dict[str, Any] = {"overlay_stroke_width": o.num("width") or None, "overlay_default_fill": o.str("fill", "") or None, "overlay_show_labels": True if o.flag("show") else None,
                         "overlay_use_names": True if o.flag("use") else None, "overlay_label_font_size": int(o.num("font", 0)) or None, "overlay_label_color": o.str("color", "") or None,
                         "overlay_label_background": True if o.flag("background") else None}
    return ij._run("edit.options_overlay", {k: v for k, v in p.items() if v is not None})


@cmd("Search...", lumen="help.search_files", notes="Plugins ▸ Utilities ▸ Search…: search= (or _=) case (sensitive); files, images, the Log and the script are all searched")
def _search_files(ij: "IJ", o: Options):
    return ij._run("help.search_files", {"query": o.str("search", o.str("_", "")), "case_sensitive": o.flag("case")})


simple("Linear Gradient", lumen="edit.linear_gradient", params={"angle": "angle"}, stack_flag=False, notes="Image ▸ Drawing ▸ Linear Gradient: angle= (foreground → background colour)")
simple("Radial Gradient", lumen="edit.radial_gradient", params={"radius": "radius"}, stack_flag=False, notes="Image ▸ Drawing ▸ Radial Gradient: radius= (0 = to the corners)")


@cmd("Update ImageJ...", "Update...", "Update Fiji...", lumen="help.update", notes="Help ▸ Update…: check (only report what is available) dry (dry run) → upgrades the Lumen package with pip")
def _update(ij: "IJ", o: Options):
    return ij._run("help.update", {"check_only": o.flag("check"), "dry_run": o.flag("dry")})


@cmd("Text Window", "Text Window...", lumen="file.new_text", notes="File ▸ New ▸ Text Window: name= text=; print(\"[name]\", s) appends to an open text window")
def _text_window(ij: "IJ", o: Options):
    return ij._run("file.new_text", {"name": o.str("name", "Untitled.txt"), "text": o.str("text", "")})


# Plugins ▸ New ▸ … (ij.plugin.NewPlugin): the template kinds of plugins.new_template
_NEW_PLUGIN_KINDS = {"macro tool": "Macro Tool", "plugin frame": "Plugin Frame", "plugin tool": "Plugin Tool", "plugin": "Python command",
                     "plugin filter": "Python filter (plane)", "macro": "ImageJ macro", "javascript": "Python command", "python": "Python command",
                     "python command": "Python command", "python filter": "Python filter (plane)", "script with dialog": "Script with dialog",
                     "imagej macro": "ImageJ macro", "text window": "Text Window", "table": "Table"}


@cmd("Macro Tool", "Plugin Frame", "Plugin Tool", "Plugin", "Plugin Filter", "Macro", "JavaScript", "New... ", "New...",
     lumen="plugins.new_template", notes="Plugins ▸ New ▸ Macro Tool / Plugin Frame / Plugin Tool / Plugin / Plugin Filter / Macro: name=[My Tool] writes the template "
                                        "(Macro Tool → an .ijm tool macro set on the tool rail, Plugin Frame → a docked plugin frame, Plugin → Python command, "
                                        "Plugin Filter → plane filter, Macro / JavaScript → ImageJ macro / Python command); run(\"New...\", \"type=[Plugin Frame] "
                                        "name=X\") picks the kind with type= (Text Window / Table open those instead); overwrite load")
def _new_plugin(ij: "IJ", o: Options, command: str = ""):
    kind_text = o.str("type", o.str("kind", command.strip())).strip()
    kind = _NEW_PLUGIN_KINDS.get(kind_text.lower())
    if kind is None:
        raise ValueError(f"New...: unknown template type {kind_text!r} (Macro Tool, Plugin Frame, Plugin Tool, Plugin, Plugin Filter, Macro, Text Window, Table)")
    if kind == "Text Window":
        return _text_window(ij, o)
    if kind == "Table":
        return ij._run("table.new", {"name": o.str("name", "Table"), "columns": o.str("headings", o.str("columns", ""))})
    name = o.str("name", o.str("title", "")).strip() or {"Macro Tool": "My Tool", "Plugin Tool": "My Tool", "Plugin Frame": "My Frame"}.get(kind, "My Plugin")
    params: dict[str, Any] = {"kind": kind, "name": name, "overwrite": o.flag("overwrite")}
    if "load" in o:
        params["load"] = o.flag("load")
    return ij._run("plugins.new_template", params, use_roi=False)


# ====================================================================== MorphoLibJ (Plugins ▸ MorphoLibJ)
def _morpho_mode(command: str) -> str:
    return "3D (whole stack)" if "3D" in command else "2D (current plane)"


def _morpho_conn(o: Options, default: str = "4") -> str:
    """connectivity=4|8 (2D) or 6|26 / C6|C26 (3D) → the ops' 4 (faces only) / 8 (faces + diagonals)."""
    v = re.sub(r"\D", "", o.str("connectivity", ""))
    return default if not v else ("4" if v in ("4", "6") else "8")


def _choice(op_id: str, param: str, value: str, default: str) -> str:
    """``value`` matched case-insensitively (then by prefix) against the choices of an op parameter."""
    p = next((p for p in REGISTRY.get(op_id).params if p.name == param), None)
    v = str(value or "").strip().lower()
    if p is None or not v:
        return default
    low = {c.lower(): c for c in (p.choices or [])}
    if v in low:
        return low[v]
    for c in p.choices or []:
        if c.lower().startswith(v):
            return c
    return default


_MLJ_LUTS = {"glasbey dark": "Glasbey on dark", "glasbey_dark": "Glasbey on dark", "glasbey bright": "Glasbey", "glasbey_bright": "Glasbey", "mixed colors": "Random", "mixed_colors": "Random",
             "golden_angle": "Golden angle", "golden angle": "Golden angle"}


@cmd("Labels To RGB", lumen="morpho.labels_to_rgb", notes="MorphoLibJ Labels ▸ Labels To RGB: colormap=[Glasbey|Glasbey Dark|Golden Angle|Mixed Colors|Grays|Fire|Ice|Spectrum|…] background=White|Black shuffle")
def _mlj_labels_to_rgb(ij: "IJ", o: Options):
    cm = o.str("colormap", o.str("lut", "Glasbey"))
    lut = _MLJ_LUTS.get(cm.strip().lower()) or _choice("morpho.labels_to_rgb", "lut", cm, "Glasbey")
    return ij._run("morpho.labels_to_rgb", {"lut": lut, "background": o.str("background", "Black").strip().lower() or "black", "shuffle": o.flag("shuffle")}, scope="plane")


@cmd("Assign Measure to Label", lumen="morpho.assign_measure_to_label", notes="MorphoLibJ Labels ▸ Assign Measure to Label: results=[table] column=[name] → the label image coloured by the measurement")
def _mlj_assign_measure(ij: "IJ", o: Options):
    return ij._run("morpho.assign_measure_to_label", {"table": o.str("results", o.str("table", "Results")), "column": o.str("column", "Area")}, scope="plane")


_MLJ_MICRO = (("area", "Area / volume density"), ("volume", "Area / volume density"), ("perimeter", "Perimeter / surface density"), ("surface", "Perimeter / surface density"),
              ("mean_breadth", "Mean breadth density"), ("breadth", "Mean breadth density"), ("euler", "Euler density"))


@cmd("Microstructure Analysis", "Microstructure Analysis 3D", lumen="morpho.microstructure_analysis",
     notes="MorphoLibJ Analyze ▸ Microstructure Analysis (3D): area|volume perimeter|surface mean_breadth euler flags (none = all), directions=2|4 (3|13 in 3D) connectivity=4|8 (6|26)")
def _mlj_microstructure(ij: "IJ", o: Options, command: str = ""):
    three_d = "3D" in command
    feats: list[str] = []
    for key in o:
        if not o.flag(key) or key in ("directions", "connectivity"):
            continue
        for prefix, name in _MLJ_MICRO:
            if key.lower().startswith(prefix) and name not in feats:
                feats.append(name)
    p: dict[str, Any] = {"mode": _morpho_mode(command), "connectivity": _morpho_conn(o, "8") if "connectivity" in o else "8"}
    if feats:
        p["features"] = feats
    if "directions" in o:
        d = re.sub(r"\D", "", o.str("directions", ""))
        if d:
            p["directions_3d" if three_d else "directions_2d"] = _choice("morpho.microstructure_analysis", "directions_3d" if three_d else "directions_2d", d, "13" if three_d else "4")
    return ij._run("morpho.microstructure_analysis", p)


_MLJ_REGIONS_3D = (("volume", "Volume"), ("surface_area", "Surface area"), ("sphericity", "Sphericity"), ("euler_number", "Euler number"), ("bounding_box", "Bounding box"), ("centroid", "Centroid"),
                   ("equivalent_ellipsoid", "Equivalent ellipsoid"), ("ellipsoid_elongations", "Ellipsoid elongations"), ("max._inscribed", "Inscribed sphere"), ("max_inscribed", "Inscribed sphere"),
                   ("inscribed", "Inscribed sphere"), ("mean_breadth", "Mean breadth"))


@cmd("Analyze Regions 3D", lumen="morpho.analyze_regions_3d",
     notes="MorphoLibJ Analyze ▸ Analyze Regions 3D: volume surface_area sphericity euler_number bounding_box centroid equivalent_ellipsoid ellipsoid_elongations max._inscribed mean_breadth flags (none = all), "
           "surface_area_method=[Crofton (3 dirs.)|Crofton (13 dirs.)] euler_connectivity=C6|C26")
def _mlj_analyze_regions_3d(ij: "IJ", o: Options):
    feats = list(dict.fromkeys(name for key, name in _MLJ_REGIONS_3D if o.flag(key)))
    p: dict[str, Any] = {"directions": "3" if re.search(r"\b3\b", o.str("surface_area_method", "13")) else "13",
                         "connectivity": "6" if re.sub(r"\D", "", o.str("euler_connectivity", "26")) == "6" else "26"}
    if feats:
        p["features"] = feats
    return ij._run("morpho.analyze_regions_3d", p)


_MLJ_ELEMENTS = {"disk": ("Disk", None), "square": ("Square", None), "diamond": ("Diamond", None), "octagon": ("Octagon", None), "horizontal line": ("Line", 0.0),
                 "vertical line": ("Line", 90.0), "line 45 degrees": ("Line", 45.0), "line 135 degrees": ("Line", 135.0)}


@cmd("Morphological Filters", lumen="morpho.filters",
     notes="MorphoLibJ Filters ▸ Morphological Filters: operation=[Erosion|Dilation|Opening|Closing|White Top Hat|Black Top Hat|Gradient|Laplacian|Internal Gradient|External Gradient] "
           "element=[Disk|Square|Diamond|Octagon|Horizontal Line|Vertical Line|Line 45 degrees|Line 135 degrees] radius= (stack → every plane)")
def _mlj_filters(ij: "IJ", o: Options):
    el = o.str("element", "Disk").strip().lower()
    element, angle = _MLJ_ELEMENTS.get(el, ("Line" if "line" in el else "Disk", None))
    p: dict[str, Any] = {"operation": _choice("morpho.filters", "operation", o.str("operation", "Dilation"), "Dilation"), "element": element, "radius": int(o.num("radius", 2))}
    if element == "Line":
        m = re.search(r"\d+", el)
        p["line_angle"] = angle if angle is not None else (float(m.group(0)) if m else 0.0)
    return ij._run("morpho.filters", p, scope=_scope(o))


@cmd("Morphological Filters (3D)", lumen="morpho.filters_3d", notes="MorphoLibJ Filters ▸ Morphological Filters (3D): operation= element=[Ball|Cube|Cross] x-radius= y-radius= z-radius=")
def _mlj_filters_3d(ij: "IJ", o: Options):
    return ij._run("morpho.filters_3d", {"operation": _choice("morpho.filters_3d", "operation", o.str("operation", "Dilation"), "Dilation"),
                                         "element": _choice("morpho.filters_3d", "element", o.str("element", "Ball"), "Ball"),
                                         "radius_x": int(o.num("x-radius", o.num("x_radius", 2))), "radius_y": int(o.num("y-radius", o.num("y_radius", 2))),
                                         "radius_z": int(o.num("z-radius", o.num("z_radius", 1)))})


@cmd("Regional Min & Max", "Regional Min & Max 3D", lumen="morpho.regional_min_max", notes="MorphoLibJ Minima & Maxima ▸ Regional Min & Max (3D): operation=[Regional Minima|Regional Maxima] connectivity=4|8 (6|26)")
def _mlj_regional(ij: "IJ", o: Options, command: str = ""):
    return ij._run("morpho.regional_min_max", {"operation": _choice("morpho.regional_min_max", "operation", o.str("operation", ""), "Regional Minima"), "connectivity": _morpho_conn(o),
                                               "mode": _morpho_mode(command)})


@cmd("Extended Min & Max", "Extended Min & Max 3D", lumen="morpho.extended_min_max", notes="MorphoLibJ Minima & Maxima ▸ Extended Min & Max (3D): operation=[Extended Minima|Extended Maxima] dynamic= connectivity=")
def _mlj_extended(ij: "IJ", o: Options, command: str = ""):
    return ij._run("morpho.extended_min_max", {"operation": _choice("morpho.extended_min_max", "operation", o.str("operation", ""), "Extended Minima"), "dynamic": o.num("dynamic", 10),
                                               "connectivity": _morpho_conn(o), "mode": _morpho_mode(command)})


def _mlj_marker(ij: "IJ", o: Options, command: str, current_key: str, *marker_keys: str) -> str:
    """Select the ``original= / mask= / input=`` image and return the id of the ``marker=`` image (required)."""
    if o.str(current_key) and o.str(current_key) not in ("None", "*None*"):
        ij.selectWindow(o.str(current_key))
    mid = _image_param(ij, o, *marker_keys)
    if mid is None:
        raise ValueError(f"{command}: give the marker image ({marker_keys[0]}=)")
    return mid


@cmd("Impose Min & Max", "Impose Min & Max 3D", lumen="morpho.impose_min_max", notes="MorphoLibJ Minima & Maxima ▸ Impose Min & Max (3D): original= (made current) marker= operation=[Impose Minima|Impose Maxima] connectivity=")
def _mlj_impose(ij: "IJ", o: Options, command: str = ""):
    mid = _mlj_marker(ij, o, command, "original", "marker", "marker_image")
    return ij._run("morpho.impose_min_max", {"operation": _choice("morpho.impose_min_max", "operation", o.str("operation", ""), "Impose Minima"), "marker_image": mid,
                                             "connectivity": _morpho_conn(o), "mode": _morpho_mode(command)})


_MLJ_WEIGHTS = {"chessboard": "Chessboard", "city-block": "City-Block", "cityblock": "City-Block", "city block": "City-Block", "quasi-euclidean": "Quasi-Euclidean", "quasi euclidean": "Quasi-Euclidean",
                "borgefors": "Borgefors", "chessknight": "Chessknight", "chess-knight": "Chessknight"}


def _mlj_weights(o: Options) -> str:
    w = o.str("distances", "Borgefors").split("(")[0].strip().lower()
    return _MLJ_WEIGHTS.get(w, "Borgefors")  # Weights (3,4,5) / Svensson (3,4,5,7) → Borgefors


@cmd("Chamfer Distance Map", "Chamfer Distance Map 3D", lumen="morpho.chamfer_distance_map",
     notes="MorphoLibJ Binary Images ▸ Chamfer Distance Map (3D): distances=[Chessboard (1,1)|City-Block (1,2)|Quasi-Euclidean (1,1.41)|Borgefors (3,4)|Chessknight (5,7,11)] output=[16 bits|32 bits] normalize")
def _mlj_chamfer(ij: "IJ", o: Options, command: str = ""):
    out = "16-bit" if re.search(r"\b(8|16)\b", o.str("output", "32 bits")) else "32-bit"
    return ij._run("morpho.chamfer_distance_map", {"weights": _mlj_weights(o), "normalize": o.flag("normalize"), "output": out, "mode": _morpho_mode(command)})


@cmd("Geodesic Distance Map", lumen="morpho.geodesic_distance_map", notes="MorphoLibJ Binary Images ▸ Geodesic Distance Map: marker= mask= (made current) distances=[…] normalize")
def _mlj_geodesic(ij: "IJ", o: Options, command: str = ""):
    mid = _mlj_marker(ij, o, command, "mask", "marker", "marker_image")
    return ij._run("morpho.geodesic_distance_map", {"marker_image": mid, "weights": _mlj_weights(o), "normalize": o.flag("normalize")})


@cmd("Marker-controlled Watershed", lumen="morpho.marker_watershed",
     notes="MorphoLibJ Segmentation ▸ Marker-controlled Watershed: input= (made current) marker= mask= compactness= binary calculate (dams) use (diagonal connectivity); 3-D when the input is a Z stack")
def _mlj_marker_watershed(ij: "IJ", o: Options, command: str = ""):
    mid = _mlj_marker(ij, o, command, "input", "marker", "marker_image")
    p: dict[str, Any] = {"markers": "Marker image", "marker_image": mid, "input": "Image as is", "calculate_dams": o.flag("calculate"), "connectivity": "8" if o.flag("use") else "4",
                         "mode": "3D (whole stack)" if ij.img.nz > 1 else "2D (current plane)"}
    mask = _image_param(ij, o, "mask", "mask_image")
    if mask is not None:
        p["mask_image"] = mask
    return ij._run("morpho.marker_watershed", p)


@cmd("Morphological Reconstruction", "Morphological Reconstruction 3D", lumen="morpho.reconstruct", notes="MorphoLibJ Morphological Reconstruction (3D): marker= mask= (made current) type=[By Dilation|By Erosion] connectivity=")
def _mlj_reconstruct(ij: "IJ", o: Options, command: str = ""):
    mid = _mlj_marker(ij, o, command, "mask", "marker", "marker_image")
    return ij._run("morpho.reconstruct", {"operation": _choice("morpho.reconstruct", "operation", o.str("type", o.str("operation", "")), "By Dilation"), "marker_image": mid,
                                          "connectivity": _morpho_conn(o), "mode": _morpho_mode(command)})


simple("Kill Borders", lumen="morpho.kill_borders", notes="MorphoLibJ Binary Images ▸ Kill Borders (stack → every plane)")
simple("Fill Holes (Binary/Gray)", lumen="morpho.fill_holes", notes="MorphoLibJ Binary Images ▸ Fill Holes (Binary/Gray): grayscale hole filling by reconstruction")
simple("Area Opening", lumen="morpho.area_opening", params={"min_area": "pixel"}, notes="MorphoLibJ Binary Images ▸ Area Opening: pixel= (minimum area)")
simple("Directional Filtering", lumen="morpho.directional_filter", params={"type": "type", "operation": "operation", "line_length": "line", "n_directions": "direction"},
       notes="MorphoLibJ Filters ▸ Directional Filtering: type=Max|Min operation=[Opening|Closing|Erosion|Dilation|Mean|Median] line= (length) direction= (number of directions)")


@cmd("Binary Overlay", lumen="morpho.binary_overlay", notes="MorphoLibJ Binary Images ▸ Binary Overlay: reference= (made current) binary= overlay=Red|Green|… (colour)")
def _mlj_binary_overlay(ij: "IJ", o: Options, command: str = ""):
    mid = _mlj_marker(ij, o, command, "reference", "binary", "mask", "mask_image")
    return ij._run("morpho.binary_overlay", {"mask_image": mid, "color": _color_hex(o.get("overlay"), "#ff0000")})


@cmd("Label Overlay", lumen="morpho.label_overlay", notes="MorphoLibJ Labels ▸ Label Overlay: reference= (made current) labels= opacity= (%)")
def _mlj_label_overlay(ij: "IJ", o: Options, command: str = ""):
    mid = _mlj_marker(ij, o, command, "reference", "labels", "label", "label_image")
    opacity = o.num("opacity", 50)
    return ij._run("morpho.label_overlay", {"label_image": mid, "opacity": opacity / 100.0 if opacity > 1 else opacity})


def _mlj_labels(o: Options) -> str:
    return o.str("label(s)", o.str("labels", o.str("label", "")))


@cmd("Merge Labels", lumen="morpho.merge_labels", notes="MorphoLibJ Label Edition ▸ Merge Labels (interactive in Fiji): label(s)=1,2,3 (or labels=) merged into the first one (new_value= optional)")
def _mlj_merge_labels(ij: "IJ", o: Options):
    p: dict[str, Any] = {"labels": _mlj_labels(o)}
    if "new_value" in o:
        p["new_value"] = int(o.num("new_value", 0))
    return ij._run("morpho.merge_labels", p, scope=_scope(o))


simple("Label Size Opening", lumen="morpho.label_size_opening", params={"min_size": "min_size"}, notes="MorphoLibJ Labels ▸ Label Size Opening: min_size= (pixels / voxels)")


@cmd("Select Label(s)", lumen="morpho.select_label", notes="MorphoLibJ Labels ▸ Select Label(s): label(s)=1,2,3 → the outline of each listed label becomes a selection (several → all added to the ROI Manager, the last one stays selected)")
def _mlj_select_labels(ij: "IJ", o: Options):
    labels = [int(float(x)) for x in re.split(r"[,\s;]+", _mlj_labels(o).strip()) if x.strip()]
    if not labels:
        return ij._run("morpho.select_label", {"label": 0})
    out: dict[str, Any] = {}
    for k in labels:
        out = ij._run("morpho.select_label", {"label": k, "add_to_manager": len(labels) > 1})
    return out


@cmd("Remove Label(s)", lumen="morpho.replace_label", notes="MorphoLibJ Labels ▸ Remove Label(s): label(s)=1,2,3 → set to 0")
def _mlj_remove_labels(ij: "IJ", o: Options):
    return ij._run("morpho.replace_label", {"labels": _mlj_labels(o), "new_value": 0}, scope=_scope(o))


# ====================================================================== Registration estimators (SIFT / StackReg / Descriptor-based) and Time Lapse
_SIFT_MODES = {"translation": "Translation", "rigid": "Rigid body", "similarity": "Scaled rotation", "affine": "Affine"}
_STACKREG_MODES = {"translation": "Translation", "rigid body": "Rigid body", "rigid": "Rigid body", "scaled rotation": "Scaled rotation", "affine": "Affine", "bilinear": "Bilinear"}


def _align_axis(img: LumenImage) -> str:
    return "T" if (img.nt > 1 and img.nz == 1) else "Z"


@cmd("Linear Stack Alignment with SIFT", "Linear Stack Alignment with SIFT MultiChannel", lumen="registration.align_stack",
     notes="Fiji Plugins ▸ Registration ▸ Linear Stack Alignment with SIFT: initial_gaussian_blur= steps_per_scale_octave= minimum_image_size= maximum_image_size= feature_descriptor_size= "
           "feature_descriptor_orientation_bins= closest/next_closest_ratio= maximal_alignment_error= inlier_ratio= expected_transformation=[Translation|Rigid|Similarity|Affine] interpolate "
           "show_info show_transformation_matrix registration_channel= → SIFT alignment onto the previous slice as a new stack")
def _sift_stack_alignment(ij: "IJ", o: Options):
    p: dict[str, Any] = {"axis": _align_axis(ij.img), "estimator": "SIFT feature matching", "mode": _SIFT_MODES.get(o.str("expected_transformation", "Rigid").strip().lower(), "Rigid body"),
                         "reference": "Previous slice", "initial_blur": o.num("initial_gaussian_blur", 1.6), "steps_per_octave": int(o.num("steps_per_scale_octave", 3)),
                         "min_image_size": int(o.num("minimum_image_size", 64)), "max_image_size": int(o.num("maximum_image_size", 1024)), "descriptor_size": int(o.num("feature_descriptor_size", 4)),
                         "orientation_bins": int(o.num("feature_descriptor_orientation_bins", 8)), "max_ratio": o.num("closest/next_closest_ratio", 0.92),
                         "max_alignment_error": o.num("maximal_alignment_error", 25), "min_inlier_ratio": o.num("inlier_ratio", 0.05), "interpolation": "Linear" if o.flag("interpolate") else "Nearest",
                         "show_info": o.flag("show_info"), "show_matrix": o.flag("show_transformation_matrix"), "channel": int(o.num("registration_channel", 0)), "create_new": True}
    return ij._run("registration.align_stack", p)


@cmd("StackReg", lumen="registration.align_stack", notes="Fiji Plugins ▸ Registration ▸ StackReg: transformation=[Translation|Rigid Body|Scaled Rotation|Affine|Bilinear] → intensity-based pyramid alignment onto the current slice, in place")
def _stackreg(ij: "IJ", o: Options):
    return ij._run("registration.align_stack", {"axis": _align_axis(ij.img), "estimator": "Intensity (pyramid optimisation)",
                                                "mode": _STACKREG_MODES.get(o.str("transformation", "Rigid Body").strip().lower(), "Rigid body"), "reference": "Current slice", "create_new": False})


_DESC_THRESHOLDS = {"very low": 0.2, "low": 0.1, "medium": 0.05, "strong": 0.02, "very strong": 0.01}
_DESC_OBJECTS = {"minima only": "Dark", "minima & maxima": "Bright and dark", "minima and maxima": "Bright and dark"}
_DESC_MODELS = {"translation": "Translation", "rigid": "Rigid body", "similarity": "Similarity", "affine": "Affine"}


def _descriptor_params(o: Options) -> dict[str, Any]:
    m = re.search(r"\d+(?:\.\d+)?", o.str("approximate_size", "5"))
    model_word = (o.str("transformation_model", "Rigid").strip().split() or ["rigid"])[0].lower()
    return {"threshold": _DESC_THRESHOLDS.get(o.str("brightness_of_objects", "Medium").strip().lower(), 0.05), "size": float(m.group(0)) if m else 5.0,
            "objects": _DESC_OBJECTS.get(o.str("type_of_detections", "Maxima only").strip().lower(), "Bright"), "model": _DESC_MODELS.get(model_word, "Rigid body"),
            "n_neighbors": int(o.num("number_of_neighbors", 3)), "redundancy": int(o.num("redundancy", 1)), "significance": o.num("significance", 3),
            "ransac_threshold": o.num("allowed_error_for_ransac", 5)}


@cmd("Descriptor-based registration (2d/3d)", lumen="registration.descriptor_registration",
     notes="Fiji Plugins ▸ Registration ▸ Descriptor-based registration (2d/3d): first_image= (made current) second_image= (target) brightness_of_objects=[Very low|Low|Medium|Strong|Very strong] "
           "approximate_size=[5 px] type_of_detections=[Maxima only|Minima only|Minima & Maxima] transformation_model=[Translation|Rigid|Similarity|Affine (2d)] number_of_neighbors= redundancy= "
           "significance= allowed_error_for_ransac= choose_registration_channel_for_image_1= → registered copy of the first image")
def _descriptor_registration(ij: "IJ", o: Options, command: str = ""):
    if o.str("first_image"):
        ij.selectWindow(o.str("first_image"))
    tid = _image_param(ij, o, "second_image", "target")
    if tid is None:
        raise ValueError(f"{command}: give the second image (second_image=)")
    p: dict[str, Any] = {"target": tid, **_descriptor_params(o), "channel": int(o.num("choose_registration_channel_for_image_1", 0)), "create_new": True}
    return ij._run("registration.descriptor_registration", p)


@cmd("Descriptor-based series registration (2d/3d + t)", lumen="registration.descriptor_series",
     notes="Fiji Plugins ▸ Registration ▸ Descriptor-based series registration (2d/3d + t): the detection / model / RANSAC keys of the pairwise command plus register=[… first frame …] "
           "(else onto the previous frame) choose_registration_channel= → the stack is aligned in place")
def _descriptor_series(ij: "IJ", o: Options):
    ref = (o.str("register", "") + " " + o.str("registration", "")).lower()
    p: dict[str, Any] = {**_descriptor_params(o), "reference": "First frame" if "first" in ref else "Previous frame", "channel": int(o.num("choose_registration_channel", 0)), "create_new": False}
    return ij._run("registration.descriptor_series", p)


@cmd("Phase Map", lumen="timelapse.phase_map",
     notes="Fiji Plugins ▸ Time Lapse ▸ Phase Map: min_period= max_period= detrend=[Mean|Linear|None] window phase=[Degrees|Radians] output=[Period (frames)|Frequency (cycles/frame)|Frequency (Hz)|Period (seconds)] reference=[None|Mean signal|Selection]")
def _phase_map(ij: "IJ", o: Options):
    return ij._run("timelapse.phase_map", {"min_period": o.num("min_period", 2), "max_period": o.num("max_period", 0), "detrend": o.str("detrend", "Mean"), "window": o.flag("window") or "window" not in o,
                                           "phase_unit": o.str("phase", "Degrees"), "frequency_output": o.str("output", "Period (frames)"), "reference": o.str("reference", "None")}, scope="plane")


# ====================================================================== Trainable Weka Segmentation
def _weka_features(value: str) -> str:
    """Fiji's feature list (``Gaussian_blur Sobel_filter Hessian …``, space / comma separated) → Lumen's comma list (the op accepts Weka spellings)."""
    return ",".join(t.replace("_", " ") for t in re.split(r"[,\s;]+", str(value or "").strip()) if t)


def _weka_classifier(value: str) -> str:
    """weka.classifiers.trees.FastRandomForest → 'Random forest' (the op's classifier choices via its Weka aliases; unknown names → Random forest)."""
    from .ops.classifier_ops import normalize_classifier
    name = str(value or "").strip().rsplit(".", 1)[-1] or "Random forest"
    try:
        return normalize_classifier(name)
    except ValueError:
        return "Random forest"


def _weka_train_params(o: Options, op_id: str) -> dict[str, Any]:
    allowed = _op_params(op_id)
    p: dict[str, Any] = {}
    if "classes" in o:
        p["classes"] = o.str("classes")
    if "features" in o:
        p["features"] = _weka_features(o.str("features"))
    if "classifier" in o:
        p["classifier"] = _weka_classifier(o.str("classifier"))
    for key in ("min_sigma", "max_sigma", "membrane_thickness", "membrane_patch_size", "n_trees", "max_depth", "features_per_node", "overlay_opacity", "max_memory_mb"):
        if key in o and key in allowed:
            p[key] = o.num(key)
    for key in ("channels", "output", "class_colors"):
        if key in o:
            p[key] = o.str(key)
    if "balance" in o or "balance_classes" in o:
        p["balance_classes"] = o.flag("balance") or o.flag("balance_classes")
    if "apply_now" in o:
        p["apply_now"] = o.flag("apply_now")
    return {k: v for k, v in p.items() if k in allowed}


@cmd("Trainable Weka Segmentation", "Trainable Weka Segmentation 3D", lumen="segment.train_classifier / segment3d.train_classifier",
     notes="Fiji Plugins ▸ Segmentation ▸ Trainable Weka Segmentation (3D): interactive in Fiji – with options (classes= features=[Gaussian_blur Sobel_filter …] min_sigma= max_sigma= classifier=[…FastRandomForest] "
           "n_trees= output=) it trains on the class-named ROIs of the manager right away; without options it only opens the Weka session for the "
           "call(\"trainableSegmentation.Weka_Segmentation.*\") statics (createNewClass, changeClassName, addTrace, setFeature, setMinimumSigma, setMaximumSigma, setClassifier, trainClassifier, "
           "getResult, getProbability, loadClassifier, saveClassifier, applyClassifier, toggleOverlay, setOpacity)")
def _weka(ij: "IJ", o: Options, command: str = ""):
    st = ij._weka_state()
    st["three_d"] = "3D" in command
    if not o:
        return None
    return ij._run(ij._weka_op("train_classifier"), _weka_train_params(o, ij._weka_op("train_classifier")), scope="plane")


# ====================================================================== Grid/Collection stitching, Elastic Montaging, Landmark Correspondences, Moving Least Squares
_STITCH_ORDERS = (("snake by rows", "Snake by rows"), ("snake by columns", "Snake by columns"), ("column-by-column", "Column-by-column"), ("column by column", "Column-by-column"),
                  ("row-by-row", "Row-by-row"))
_STITCH_FUSIONS = (("linear", "Linear blending"), ("average", "Average"), ("median", "Median"), ("max", "Max intensity"), ("min", "Min intensity"), ("overlay", "Overlay into composite"),
                   ("do not", "Do not fuse"))


@cmd("Grid/Collection stitching", lumen="registration.stitch_collection",
     notes="Fiji Plugins ▸ Stitching ▸ Grid/Collection stitching: type=[Grid: row-by-row|Grid: column-by-column|Grid: snake by rows|Grid: snake by columns|Positions from file|Unknown position] "
           "grid_size_x= grid_size_y= tile_overlap= directory= file_names=tile_{ii}.tif layout_file= fusion_method=[Linear Blending|Average|Median|Max. Intensity|Min. Intensity|Overlay into composite image|Do not fuse images …] "
           "regression_threshold= max/avg_displacement_threshold= absolute_displacement_threshold= compute_overlap subpixel_accuracy downsample_tiles image_output=[Fuse and display|Write to disk] output_directory=; "
           "computation_parameters= use_virtual_input_images ignore_z_stage invert_x invert_y are ignored (no 'Filename defined position' / 'Sequential images' types)")
def _grid_collection_stitching(ij: "IJ", o: Options, command: str = ""):
    t = o.str("type", "Grid: row-by-row").strip().lower()
    p: dict[str, Any] = {"folder": _require_path(o, command, "directory", "folder", "dir")}
    if t.startswith("grid") or t.startswith("filename"):
        p["layout"] = "Grid (folder)"
        p["order"] = next((v for k, v in _STITCH_ORDERS if k in t), "Row-by-row")
        p["grid_x"], p["grid_y"], p["overlap_percent"] = int(o.num("grid_size_x", 2)), int(o.num("grid_size_y", 2)), o.num("tile_overlap", 10)
    elif t.startswith("positions"):
        p["layout"] = "Positions from file"
        if o.str("layout_file"):
            p["layout_file"] = o.str("layout_file")
    elif t.startswith("unknown") or t.startswith("sequential"):
        p["layout"] = "Unknown positions (folder)"
    else:
        raise ValueError(f"{command}: type=[{o.str('type')}] is not available (Grid: …, Positions from file, Unknown position)")
    if o.str("file_names"):
        p["pattern"] = re.sub(r"\{[^}]*\}", "*", o.str("file_names"))
    fm = o.str("fusion_method", "Linear Blending").strip().lower()
    p["fusion"] = next((v for k, v in _STITCH_FUSIONS if fm.startswith(k)), "Linear blending")
    for key, pk in (("regression_threshold", "regression_threshold"), ("max/avg_displacement_threshold", "max_avg_displacement"), ("absolute_displacement_threshold", "abs_displacement")):
        if key in o:
            p[pk] = o.num(key)
    p["compute_overlap"] = o.flag("compute_overlap")
    p["subpixel"] = o.flag("subpixel_accuracy")
    if o.flag("downsample_tiles"):
        p["downsample"] = 2
    if o.str("image_output", "Fuse and display").strip().lower().startswith("write"):
        p["output"] = "Write to file"
        p["output_path"] = os.path.join(_path_opt(o, "output_directory") or p["folder"], "fused.tif")
    else:
        p["output"] = "Open image"
    return ij._run("registration.stitch_collection", p)


@cmd("Elastic Montaging", lumen="registration.elastic_montage",
     notes="Fiji Plugins ▸ Registration ▸ Elastic Montaging (on the open images): stiffness= minimal_pmcc_r= (→ min_correlation) layer_scale= (→ downscale) resolution= (→ mesh spacing); "
           "search_radius= block_radius= maximal_curvature_ratio= maximal_second_best_r/best_r= maximal_stretch= maximal_iterations= maximal_plateauwidth= are ignored")
def _elastic_montaging(ij: "IJ", o: Options):
    p: dict[str, Any] = {"layout": "Open images"}
    if "stiffness" in o:
        p["stiffness"] = o.num("stiffness", 1.0)
    if "minimal_pmcc_r" in o:
        p["min_correlation"] = o.num("minimal_pmcc_r", 0.3)
    ls = o.num("layer_scale", 0)
    if ls > 0:
        p["downscale"] = max(int(round(1 / ls)), 1)
    res = o.num("resolution", 0)
    if res > 0 and ij.session.active is not None:
        p["mesh_spacing"] = max(int(round(ij.img.width / res)), 1)
    return ij._run("registration.elastic_montage", p)


_LC_MODELS = {"translation": "Translation", "rigid": "Rigid body", "similarity": "Similarity", "affine": "Affine", "perspective": "Projective", "projective": "Projective"}


@cmd("Landmark Correspondences", lumen="image.transform_landmark_correspondences",
     notes="Fiji Plugins ▸ Transform ▸ Landmark Correspondences: source_image= (made current) template_image= (point selections on both, or source=[x1,y1; …] target=[…] / source_roi= target_roi=) "
           "transformation_method=[Translation|Rigid|Similarity|Affine|Perspective|Moving Least Squares (non-linear)] transformation_class=[Rigid|Similarity|Affine] (MLS) alpha= interpolate; mesh_resolution= is ignored")
def _landmark_correspondences(ij: "IJ", o: Options, command: str = ""):
    if o.str("source_image"):
        ij.selectWindow(o.str("source_image"))
    tid = _image_param(ij, o, "template_image", "template", "target_image")
    if tid is None:
        raise ValueError(f"{command}: give the template image (template_image=)")
    tm = o.str("transformation_method", "Affine").strip().lower()
    if tm.startswith("moving"):
        cls = (o.str("transformation_class", "Affine").strip().split() or ["affine"])[0].lower()
        model = _choice("image.transform_landmark_correspondences", "model", f"Moving least squares ({cls})", "Moving least squares (affine)")
    else:
        model = _LC_MODELS.get((tm.split() or ["affine"])[0], "Affine")
    p: dict[str, Any] = {"target": tid, "model": model, "alpha": o.num("alpha", 1.0), "interpolation": "Linear" if o.flag("interpolate") else "Nearest"}
    for key, opt in (("source_points", "source"), ("target_points", "target"), ("source_roi", "source_roi"), ("target_roi", "target_roi")):
        if o.str(opt) or o.str(key):
            p[key] = o.str(opt) or o.str(key)
    return ij._run("image.transform_landmark_correspondences", p, scope="plane")


@cmd("Moving Least Squares", "Interactive Moving Least Squares", lumen="image.transform_mls",
     notes="Fiji Plugins ▸ Transform ▸ Interactive Moving Least Squares (no macro options in Fiji): source=[x1,y1; x2,y2; …] (or the point selection / source_roi=) target=[…] (or target_roi=) "
           "model=[Affine|Similarity|Rigid] alpha= interpolation= create")
def _moving_least_squares(ij: "IJ", o: Options):
    p: dict[str, Any] = {"model": _choice("image.transform_mls", "model", o.str("model", "Affine"), "Affine"), "source_points": o.str("source", o.str("source_points", "")),
                         "target_points": o.str("target", o.str("target_points", "")), "source_roi": o.str("source_roi", ""), "target_roi": o.str("target_roi", ""), "alpha": o.num("alpha", 1.0),
                         "interpolation": o.str("interpolation", "Linear"), "create_new": o.flag("create") or not o.flag("in_place")}
    return ij._run("image.transform_mls", p, scope="plane")


_WEKA_PREFIX = "trainableSegmentation.Weka_Segmentation."

#: ImageJ's ij.gui.Toolbar tool ids (toolID(), call("ij.gui.Toolbar.getToolId")); custom (macro) tools start at CUSTOM1 = 15
IJ_TOOL_IDS: dict[str, int] = {"rectangle": 0, "roundrect": 0, "rotrect": 0, "oval": 1, "ellipse": 1, "brush": 1, "polygon": 2, "freehand": 3, "line": 4,
                               "arrow": 4, "polyline": 5, "freeline": 6, "point": 7, "multipoint": 7, "wand": 8, "text": 9, "zoom": 11, "magnifier": 11,
                               "hand": 12, "dropper": 13, "angle": 14}
IJ_CUSTOM_TOOL_ID = 15
#: suffixes of ImageJ macro-set block names that define toolbar tools rather than runnable macros (ij.macro.MacroInstaller)
TOOL_MACRO_SUFFIXES = (" Tool", " Tool Options", " Tool Selected", " Action Tool", " Menu Tool")


def is_tool_macro_name(name: str) -> bool:
    """True for ``macro "Circle Tool - C00cO11cc"`` style block names: ``<Name> Tool[ Options| Selected]`` / ``Action Tool`` / ``Menu Tool``
    before the optional `` - <icon>`` part."""
    base = str(name).split(" - ", 1)[0].rstrip()
    return any(base.endswith(s) for s in TOOL_MACRO_SUFFIXES)


MACRO_FUNCTION_NAMES: list[str] = []


def _mf(fn):
    MACRO_FUNCTION_NAMES.append(fn.__name__.rstrip("_"))
    return fn


class MacroExit(Exception):
    pass


class IJArray(list):
    """ImageJ arrays grow when you assign past the end (setOption('ExpandableArrays') behaviour)."""

    def __setitem__(self, i, v):
        if isinstance(i, int) and i >= len(self):
            self.extend([0] * (i - len(self) + 1))
        super().__setitem__(i, v)


_JAVA_INT_MIN, _JAVA_INT_MAX = -2**31, 2**31 - 1


def ij_d2s(n: float, decimals: int) -> str:
    """Port of IJ.d2s(double, int). Negative decimals → DecimalFormat("0.0000E0")-style notation."""
    n = float(n)
    if math.isnan(n):
        return "NaN"
    if math.isinf(n):
        return "Infinity" if n > 0 else "-Infinity"
    if decimals < 0:
        d = min(-decimals, 9)
        mant, exp = f"{n:.{d}e}".split("e")
        return f"{mant}E{int(exp)}"  # Java prints '3.0000E-5' / '3.0000E9': no '+', no zero padding
    d = min(decimals, 9)
    if d == 0:  # ImageJ's 0-decimal formatter rounds HALF_UP
        return str(int(Decimal(n).quantize(Decimal(1), rounding=ROUND_HALF_UP)))
    return f"{n:.{d}f}"


def ij_d2s_sig(x: float, significant: int = 4, max_digits: int = 9) -> str:
    """Port of IJ.d2s(double x, int significantDigits, int maxDigits)."""
    x = float(x)
    if math.isnan(x) or math.isinf(x) or x == 0.0:
        return ij_d2s(x, significant) if x != 0.0 else "0"
    magnitude = math.ceil(math.log10(abs(x)) + 0.223 * 10 ** (-max_digits))
    decimals = max_digits - magnitude
    if decimals < 0 or magnitude < significant + 1 - max_digits:
        return ij_d2s(x, -significant)  # exp notation for large and small numbers
    if decimals > significant:
        decimals = max(significant, decimals - max_digits + significant)
    return ij_d2s(x, decimals)


def _num_to_str(f: float) -> str:
    """Port of ij.macro.Interpreter.toString(double): integers in the Java int range print plainly,
    everything else is d2s(x, 4, 9) with trailing zeros stripped (unless in E notation)."""
    if math.isnan(f):
        return "NaN"
    if math.isinf(f):
        return "Infinity" if f > 0 else "-Infinity"
    if f == int(f) and _JAVA_INT_MIN <= f <= _JAVA_INT_MAX:  # Java '(int)x == x'
        return str(int(f))
    s = ij_d2s_sig(f, 4, 9)
    if "." in s and "E" not in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _str(v: Any) -> str:
    """ImageJ-style string conversion: integers print as integers, other numbers with 4 significant digits
    (IJ.d2s(x, 4, 9), trailing zeros stripped, E notation for very small / large values)."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        i = int(v)
        return str(i) if _JAVA_INT_MIN <= i <= _JAVA_INT_MAX else _num_to_str(float(i))  # ImageJ only has doubles: 3e9 → 3.0000E9
    if isinstance(v, (float, np.floating)):
        return _num_to_str(float(v))
    if isinstance(v, (list, tuple)):
        return ",".join(_str(x) for x in v)
    if v is None:
        return "NaN"
    return str(v)


def _add(a: Any, b: Any) -> Any:
    if isinstance(a, str) or isinstance(b, str):
        return _str(a) + _str(b)
    return a + b


def _div(a: Any, b: Any) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        if a == 0:
            return float("nan")
        return math.copysign(float("inf"), a)


def _idx(i: Any) -> int:
    return int(i)


def _len(v: Any) -> int:
    """``.length`` – a private helper so a macro variable named ``len`` cannot shadow it."""
    return len(v)


def _outs(value: Any, n: int) -> tuple:
    seq = list(value) if isinstance(value, (list, tuple)) else [value]
    seq = seq[:n] + [0] * max(0, n - len(seq))
    return tuple(seq)


def _band(a, b): return int(a) & int(b)
def _bor(a, b): return int(a) | int(b)
def _bxor(a, b): return int(a) ^ int(b)
def _shl(a, b): return int(a) << int(b)
def _shr(a, b): return int(a) >> int(b)
def _bnot(a): return ~int(a)
def _mod(a, b):
    """ImageJ ``%``: Java's double remainder (the sign follows the dividend); NaN on a zero divisor."""
    try:
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)) and not isinstance(a, bool) and not isinstance(b, bool):
            r = abs(int(a)) % abs(int(b))  # exact truncated remainder, stays an int
            return -r if a < 0 else r
        return math.fmod(a, b)
    except (ZeroDivisionError, ValueError, OverflowError):
        return float("nan")


# ====================================================================== helper objects
class _ArrayFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij

    def getStatistics(self, arr):
        a = np.asarray([float(x) for x in arr], float) if len(arr) else np.zeros(1)
        return float(a.min()), float(a.max()), float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0

    def sort(self, arr):
        arr.sort()
        return arr

    def reverse(self, arr):
        arr.reverse()
        return arr

    def concat(self, *parts):
        out = IJArray()
        for p in parts:
            out.extend(p if isinstance(p, (list, tuple)) else [p])
        return out

    def copy(self, arr):
        return IJArray(arr)

    def slice(self, arr, start, end=None):
        return IJArray(arr[int(start):(int(end) if end is not None else len(arr))])

    def trim(self, arr, n):
        return IJArray(arr[:int(n)])

    def fill(self, arr, value):
        for i in range(len(arr)):
            arr[i] = value
        return arr

    def getSequence(self, n):
        return IJArray(range(int(n)))

    def rankPositions(self, arr):
        return IJArray(int(i) for i in np.argsort(np.asarray(arr, float), kind="stable"))

    def print(self, arr):  # noqa: A003
        self.ij.print(",".join(_str(x) for x in arr))

    def show(self, *args):
        self.ij.print(",".join(_str(x) for x in args[-1])) if args else None

    def deleteValue(self, arr, value):
        return IJArray(x for x in arr if x != value)

    def deleteIndex(self, arr, i):
        return IJArray(x for k, x in enumerate(arr) if k != int(i))

    def findMaxima(self, arr, tolerance, *rest):
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(np.asarray(arr, float), prominence=float(tolerance))
        return IJArray(int(p) for p in peaks)

    def findMinima(self, arr, tolerance, *rest):
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(-np.asarray(arr, float), prominence=float(tolerance))
        return IJArray(int(p) for p in peaks)

    def resample(self, arr, n):
        a = np.asarray(arr, float)
        return IJArray(np.interp(np.linspace(0, len(a) - 1, int(n)), np.arange(len(a)), a).tolist())

    def filter(self, arr, s):  # noqa: A003
        return IJArray(x for x in arr if s in str(x))

    def getVertexAngles(self, xs, ys, arm):
        raise ValueError("Unsupported macro function: Array.getVertexAngles")


class _MathFns:
    abs = staticmethod(abs); sqrt = staticmethod(math.sqrt); pow = staticmethod(math.pow); exp = staticmethod(math.exp); log = staticmethod(math.log); log10 = staticmethod(math.log10)
    sin = staticmethod(math.sin); cos = staticmethod(math.cos); tan = staticmethod(math.tan); atan = staticmethod(math.atan); atan2 = staticmethod(math.atan2); asin = staticmethod(math.asin)
    acos = staticmethod(math.acos); sinh = staticmethod(math.sinh); cosh = staticmethod(math.cosh); tanh = staticmethod(math.tanh); floor = staticmethod(math.floor); ceil = staticmethod(math.ceil)
    sqr = staticmethod(lambda x: x * x); signum = staticmethod(lambda x: (x > 0) - (x < 0)); min = staticmethod(min); max = staticmethod(max); erf = staticmethod(math.erf)
    round = staticmethod(lambda x: int(math.floor(x + 0.5)))
    toRadians = staticmethod(math.radians); toDegrees = staticmethod(math.degrees)
    constrain = staticmethod(lambda v, lo, hi: min(max(v, lo), hi))


class _StringFns:
    @staticmethod
    def format(fmt, *args):
        return fmt % tuple(args) if "%" in fmt else fmt

    @staticmethod
    def pad(s, n):
        s = _str(s)
        return s.rjust(int(n), "0")

    @staticmethod
    def join(arr, sep=","):
        return sep.join(_str(x) for x in arr)

    @staticmethod
    def trim(s):
        return str(s).strip()

    @staticmethod
    def replace(s, a, b):
        return _replace(s, a, b)

    @staticmethod
    def length(s):
        return len(str(s))

    @staticmethod
    def substring(s, a, b=None):
        return str(s)[int(a):(int(b) if b is not None else None)]

    @staticmethod
    def toUpperCase(s):
        return str(s).upper()

    @staticmethod
    def toLowerCase(s):
        return str(s).lower()

    @staticmethod
    def split(s, delim=None):
        return _split(s, delim)

    @staticmethod
    def append(s, sep=None):
        return str(s)

    @staticmethod
    def getResultsHeadings():
        return "\t".join(["Label"] if False else [])

    @staticmethod
    def copy(s):
        return str(s)

    @staticmethod
    def paste():
        return ""

    @staticmethod
    def buffer():
        return ""

    @staticmethod
    def show():
        return None


def _java_replacement(template: str) -> Callable[[re.Match], str]:
    """Expand a Java String.replaceAll replacement: $n / ${name} group references, \\x = literal x."""
    def expand(m: re.Match) -> str:
        out: list[str] = []
        i, n, ngroups = 0, len(template), m.re.groups
        while i < n:
            c = template[i]
            if c == "\\" and i + 1 < n:  # Java: a backslash quotes the next character
                out.append(template[i + 1]); i += 2
            elif c == "$" and i + 1 < n and template[i + 1] == "{" and template.find("}", i + 2) > 0:
                j = template.find("}", i + 2)
                name = template[i + 2:j]
                out.append((m.group(name) or "") if name in m.re.groupindex else template[i:j + 1])
                i = j + 1
            elif c == "$" and i + 1 < n and template[i + 1].isdigit():
                j = i + 1
                num = int(template[j]); j += 1
                while j < n and template[j].isdigit() and num * 10 + int(template[j]) <= ngroups:  # Java: largest legal group number
                    num = num * 10 + int(template[j]); j += 1
                out.append((m.group(num) or "") if num <= ngroups else "")
                i = j
            else:
                out.append(c); i += 1
        return "".join(out)
    return expand


def _replace(s: Any, a: str, b: str) -> str:
    s = str(s)
    if len(a) == 1:  # ImageJ ≥ 1.53d: literal char replace whenever `old` is one character (any `new`)
        return s.replace(a, b)
    try:
        return re.sub(a, _java_replacement(b), s)  # Java replaceAll: $1, ${name}, \x escapes
    except re.error:
        return s.replace(a, b)


def _split(s: Any, delim: str | None = None) -> IJArray:
    s = str(s)
    if delim is None or delim == "":
        return IJArray(s.split())
    if len(delim) >= 3 and delim.startswith("(") and delim.endswith(")"):
        # ImageJ Functions.split(): strings = s1.split(s2.substring(1, s2.length()-1), -1) – Java never returns
        # captured-group text (Python's re.split does) and limit -1 keeps trailing empty strings.
        if s == "":
            return IJArray([])  # ImageJ returns a zero-length array for an empty string
        pat = re.compile(delim[1:-1])
        parts = pat.split(s)[::pat.groups + 1]  # drop every captured group, nested or named
        m = pat.match(s)
        if m is not None and m.end() == 0 and parts and parts[0] == "":
            parts = parts[1:]  # Java: a zero-width match at index 0 never yields a leading ""
        return IJArray(parts)
    if delim in ("\n", "\\n"):
        return IJArray(s.split("\n"))
    parts = re.split("[" + re.escape(delim) + "]+", s)
    return IJArray(p for p in parts if p != "" or len(delim) > 1)


class _FileFns:
    separator = os.sep
    directory = ""
    name = ""
    nameWithoutExtension = ""

    def __init__(self, ij: "IJ") -> None:
        self.ij = ij

    def exists(self, p): return os.path.exists(str(p))
    def isDirectory(self, p): return os.path.isdir(str(p))
    def isFile(self, p): return os.path.isfile(str(p))
    def getName(self, p): return os.path.basename(str(p))
    def getNameWithoutExtension(self, p): return os.path.splitext(os.path.basename(str(p)))[0]
    def getDirectory(self, p): return os.path.dirname(str(p)) + os.sep
    def getParent(self, p): return os.path.dirname(str(p))
    def length(self, p): return os.path.getsize(str(p)) if os.path.exists(str(p)) else 0
    def lastModified(self, p): return int(os.path.getmtime(str(p)) * 1000) if os.path.exists(str(p)) else 0
    def makeDirectory(self, p): os.makedirs(str(p), exist_ok=True)
    def openAsString(self, p):
        with open(str(p), encoding="utf-8", errors="replace") as f:
            return f.read()
    def openAsRawString(self, p, n=5000):
        with open(str(p), "rb") as f:
            return f.read(int(n)).decode("latin-1")
    def saveString(self, s, p):
        with open(str(p), "w", encoding="utf-8") as f:
            f.write(str(s))
    def append(self, s, p):
        with open(str(p), "a", encoding="utf-8") as f:
            f.write(str(s) + "\n")
    def delete(self, p):
        if os.path.isfile(str(p)):
            os.remove(str(p)); return 1
        return 0
    def rename(self, a, b): os.rename(str(a), str(b)); return 1
    def copy(self, a, b):
        import shutil
        shutil.copy(str(a), str(b))
    def openDialog(self, title=""): return ""
    def getDefaultDir(self): return os.getcwd()
    def setDefaultDir(self, d): os.chdir(str(d))
    def openSequence(self, d, *rest): raise ValueError("Unsupported macro function: File.openSequence")
    def close(self, f): return None


class _DialogFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij
        self._values: list[Any] = []
        self._i = 0

    def create(self, title=""): self._values = []; self._i = 0
    def createNonBlocking(self, title=""): self.create(title)
    def addNumber(self, label, default, *rest): self._values.append(default)
    def addSlider(self, label, lo, hi, default, *rest): self._values.append(default)
    def addString(self, label, default, *rest): self._values.append(default)
    def addCheckbox(self, label, default): self._values.append(bool(default))
    def addChoice(self, label, items, default=None): self._values.append(default if default is not None else (items[0] if items else ""))
    def addRadioButtonGroup(self, label, items, rows, cols, default): self._values.append(default)
    def addMessage(self, *a): pass
    def addHelp(self, *a): pass
    def addCheckboxGroup(self, rows, cols, labels, defaults): self._values.extend(bool(d) for d in defaults)
    def addDirectory(self, label, default=""): self._values.append(default)
    def addFile(self, label, default=""): self._values.append(default)
    def addImageChoice(self, label, default=None): self._values.append(default or self.ij.getTitle())
    def setInsets(self, *a): pass
    def setLocation(self, *a): pass
    def show(self): self._i = 0
    def _next(self):
        if self._i >= len(self._values):
            raise ValueError("Dialog.get*: more values requested than were added")
        v = self._values[self._i]; self._i += 1
        return v
    def getNumber(self): return self._next()
    def getString(self): return self._next()
    def getCheckbox(self): return self._next()
    def getChoice(self): return self._next()
    def getRadioButton(self): return self._next()
    def getImageChoice(self): return self._next()


class _ListFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij
        self._d: dict[str, Any] = {}
    def set(self, k, v): self._d[str(k)] = v
    def get(self, k): return self._d.get(str(k), "")
    def getValue(self, k):
        try:
            return float(self._d.get(str(k), float("nan")))
        except (TypeError, ValueError):
            return float("nan")  # ImageJ: Tools.parseDouble → NaN for a missing or non-numeric value
    def size(self): return len(self._d)
    def clear(self): self._d.clear()
    def indexOf(self, k): return list(self._d).index(str(k)) if str(k) in self._d else -1
    def getList(self): return "\n".join(f"{k}={v}" for k, v in self._d.items())
    def setList(self, s):
        self.clear()
        for line in str(s).splitlines():
            if "=" in line:
                k, v = line.split("=", 1); self._d[k] = v
    def setMeasurements(self, *a):
        """List.setMeasurements / List.setMeasurements("limit"): measure the image (or selection) with every measurement
        into the list without touching the Results table (ImageJ: ALL_STATS + SLICE [+ LIMIT]); values are stored as
        strings like ImageJ's ""+double, List.getValue parses them back."""
        ij = self.ij
        thr = None
        if any("limit" in str(x).lower() for x in a):
            lo, hi = ij.getThreshold()
            if (lo, hi) != (-1.0, -1.0):
                thr = (lo, hi)
        row = measure_roi(ij.img, ij.roi, ALL_MEASUREMENTS, limit_to_threshold=thr)[0]
        self.clear()
        for k, v in row.items():
            if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):  # Label is not a measurement
                self._d[k] = str(float(v))
    def setCommands(self, *a): self.clear(); self._d.update({k: v[1] for k, v in COMMANDS.items()})


class _FitFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij
        self._res: dict[str, Any] | None = None

    def doFit(self, equation, xs, ys, initial=None):
        from .fitting import FIT_NAMES, fit
        eq = str(equation)
        name = {"straight line": "Straight Line", "linear": "Straight Line", "2nd degree polynomial": "2nd Degree Polynomial", "3rd degree polynomial": "3rd Degree Polynomial", "4th degree polynomial": "4th Degree Polynomial",
                "exponential": "Exponential", "exponential with offset": "Exponential with Offset", "exponential recovery": "Exponential Recovery", "power": "Power", "log": "Log", "gaussian": "Gaussian",
                "rodbard": "Rodbard", "rational": "Rational", "sigmoid": "Sigmoid (Boltzmann)", "boltzmann": "Sigmoid (Boltzmann)"}.get(eq.lower())
        if name is None:
            low = {n.lower(): n for n in FIT_NAMES}
            name = low.get(eq.lower())
        if name is None:
            raise ValueError(f"Fit.doFit: unknown equation {equation!r}")
        self._res = fit(name, [float(v) for v in xs], [float(v) for v in ys])

    def p(self, i): return self._r["params"][int(i)]
    def rSquared(self): return self._r["r2"]
    def nParams(self): return len(self._r["params"])
    def f(self, x):
        from .fitting import evaluate
        return float(evaluate(self._r["function"], self._r["params"], [x])[0])
    def getEquation(self, i=None, *rest): return self._r["formula"]
    def nEquations(self):
        from .fitting import FIT_NAMES
        return len(FIT_NAMES)
    def logResults(self): self.ij.print(f"{self._r['function']}: {self._r['params']} R²={self._r['r2']:.4f}")
    def plot(self): pass
    def showDialog(self): pass
    @property
    def _r(self):
        if self._res is None:
            raise ValueError("Fit.doFit was not called")
        return self._res


class _StackFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij

    def setChannel(self, c): self.ij.img.display.c = int(c) - 1; self.ij.img.clamp_position()
    def setSlice(self, z): self.ij.img.display.z = int(z) - 1; self.ij.img.clamp_position()
    def setFrame(self, t): self.ij.img.display.t = int(t) - 1; self.ij.img.clamp_position()
    def setPosition(self, c, z, t):
        d = self.ij.img.display
        d.c, d.z, d.t = int(c) - 1, int(z) - 1, int(t) - 1
        self.ij.img.clamp_position()
    def getPosition(self):
        d = self.ij.img.display
        return d.c + 1, d.z + 1, d.t + 1
    def getDimensions(self): return self.ij.getDimensions()
    def isHyperstack(self): return self.ij.img.nc > 1 or (self.ij.img.nz > 1 and self.ij.img.nt > 1)
    def setDimensions(self, c, z, t): self.ij._run("stacks.to_hyperstack", {"channels": int(c), "slices": int(z), "frames": int(t)})
    def setDisplayMode(self, mode):
        self.ij.session.update_display(self.ij.img.id, {"mode": {"composite": "composite", "color": "single", "grayscale": "grayscale"}.get(str(mode).lower(), "composite")})
    def getDisplayMode(self): return {"composite": "composite", "single": "color", "grayscale": "grayscale"}[self.ij.img.display.mode]
    def setActiveChannels(self, s):
        img = self.ij.img
        for i, ch in enumerate(img.display.channels):
            if i < len(str(s)):
                ch.visible = str(s)[i] == "1"
    def getActiveChannels(self): return "".join("1" if ch.visible else "0" for ch in self.ij.img.display.channels)
    def getStatistics(self):
        img = self.ij.img
        d = img.display
        mask = self.ij._mask()
        sub = img.data[:, :, d.c]
        vals = sub[:, :, mask].ravel() if mask is not None else sub.ravel()
        vals = vals.astype(np.float64)
        return int(vals.size), float(vals.mean()), float(vals.min()), float(vals.max()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    def setXUnit(self, u): self.ij.img.calibration.unit = str(u)
    def setYUnit(self, u): pass
    def setZUnit(self, u): pass
    def setTUnit(self, u): self.ij.img.calibration.time_unit = str(u)
    def getUnits(self):
        cal = self.ij.img.calibration
        return cal.unit, cal.unit, cal.unit, cal.time_unit, cal.value_unit or "gray"
    def setFrameRate(self, fps): self.ij.img.calibration.frame_interval = 1.0 / float(fps) if fps else 0.0
    def getFrameRate(self):
        fi = self.ij.img.calibration.frame_interval
        return 1.0 / fi if fi else 0.0
    def setFrameInterval(self, s): self.ij.img.calibration.frame_interval = float(s)
    def getFrameInterval(self): return self.ij.img.calibration.frame_interval
    def swap(self, a, b): raise ValueError("Unsupported macro function: Stack.swap")
    def setSliceLabel(self, label, n=None): _set_slice_label(self.ij, str(label), n)
    def getSliceLabel(self, n=None): return _get_slice_label(self.ij, n)
    def setOrthoViews(self, *a): pass
    def stopOrthoViews(self): pass
    def getOrthoViewsID(self): return 0
    def setZ(self, z): self.setSlice(z)
    def setT(self, t): self.setFrame(t)
    def setC(self, c): self.setChannel(c)
    def toggleChannel(self, c):
        ch = self.ij.img.display.channels[int(c) - 1]
        ch.visible = not ch.visible


def _slice_index(ij: "IJ", n: Any = None) -> int:
    """0-based ImageJ stack index (channels fastest) of slice ``n`` (1-based) or of the current plane."""
    from .ops.stacks import plane_index
    img = ij.img
    if n is not None and int(n) > 0:
        return int(n) - 1
    d = img.display
    return plane_index(img, d.t, d.z, 0 if img.rgb else d.c)


def _get_slice_label(ij: "IJ", n: Any = None) -> str:
    from .ops.stacks import slice_labels
    labels = slice_labels(ij.img)
    i = _slice_index(ij, n)
    return labels[i] if 0 <= i < len(labels) else ""


def _set_slice_label(ij: "IJ", label: str, n: Any = None) -> None:
    from .ops.stacks import slice_labels
    img = ij.img
    labels = slice_labels(img)
    i = _slice_index(ij, n)
    if 0 <= i < len(labels):
        labels[i] = label
        img.metadata["slice_labels"] = labels
        img.rev += 1


class _RoiFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij

    def _r(self):
        if self.ij.roi is None:
            raise ValueError("Roi.*: there is no selection")
        return self.ij.roi

    def getBounds(self): return self.ij.getSelectionBounds()
    def getFloatBounds(self): return self.ij.getSelectionBounds()
    def getCoordinates(self): return self.ij.getSelectionCoordinates()
    def getFloatCoordinates(self): return self.ij.getSelectionCoordinates()
    def getName(self): return self._r().get("name", "")
    def setName(self, n): self._r()["name"] = str(n)
    def getType(self): return {0: "rectangle", 1: "oval", 2: "polygon", 3: "freehand", 5: "line", 6: "polyline", 10: "point"}.get(self.ij.selectionType(), "none")
    def setStrokeColor(self, *c): self._r()["color"] = _color_hex(c[0] if len(c) == 1 else c, "#ffff00")
    def setFillColor(self, *c): self._r()["fill"] = _color_hex(c[0] if len(c) == 1 else c, "#ffff00")
    def getStrokeColor(self): return self._r().get("color", "yellow")
    def setStrokeWidth(self, w): self._r()["stroke_width"] = float(w)
    def getStrokeWidth(self): return float(self._r().get("stroke_width", 1))
    def move(self, x, y):
        bx, by, _, _ = roimod.bounding_box(self._r())
        self.ij.roi = roimod.translate(self._r(), float(x) - bx, float(y) - by)
    def translate(self, dx, dy): self.ij.roi = roimod.translate(self._r(), float(dx), float(dy))
    def size(self): return len(roimod.line_points(self._r())) if roimod.is_line(self._r()) or self._r()["type"] == "point" else len(roimod.polygon_points(self._r()))
    def contains(self, x, y): return roimod.contains_point(self._r(), float(x), float(y))
    def remove(self): self.ij.roi = None
    def selectNone(self): self.ij.roi = None
    def setPosition(self, *a): pass
    def getPosition(self): return 0, 0, 0
    def setProperty(self, k, v): self._r().setdefault("properties", {})[str(k)] = v
    def getProperty(self, k): return self._r().get("properties", {}).get(str(k), "")
    def getProperties(self): return "\n".join(f"{k}: {v}" for k, v in self._r().get("properties", {}).items())
    def setPolygonSplineAnchors(self, xs, ys): self.ij.makeSelection("polygon", xs, ys)
    def setPolylineSplineAnchors(self, xs, ys): self.ij.makeSelection("polyline", xs, ys)
    def getSplineAnchors(self): return self.ij.getSelectionCoordinates()
    def getDefaultColor(self): return "yellow"
    def setDefaultColor(self, c): pass
    def setJustification(self, j): pass
    def setFontSize(self, s): pass
    def getFeretPoints(self): raise ValueError("Unsupported macro function: Roi.getFeretPoints")
    def getFeretValues(self):
        from .measure import feret_diameters
        r = self._r()
        pts = roimod.polygon_points(r) if roimod.is_area(r) else np.asarray(roimod.line_points(r), float)
        maxf, ang, minf, fx, fy = feret_diameters(pts)
        return IJArray([maxf, ang, minf, fx, fy])
    def setGroup(self, g): self._r()["group"] = int(g)
    def getGroup(self): return int(self._r().get("group", 0))


class _OverlayFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij
        self._pending: dict[str, Any] = {}

    def addSelection(self, stroke=None, width=None, fill=None):
        if self.ij.roi is None:
            raise ValueError("Overlay.addSelection: no selection")
        p: dict[str, Any] = {}
        if stroke:
            p["color"] = _color_hex(stroke)
        if width:
            p["stroke_width"] = float(width)
        self.ij._run("overlay.arrow" if self.ij.roi.get("arrow") else "overlay.add", p)
    def remove(self): self.ij._run("overlay.remove", {})
    def clear(self): self.remove()
    def show(self): pass
    def hide(self): pass
    def hidden(self): return False
    def size(self): return len(self.ij.img.overlay)
    def drawString(self, text, x, y): self.ij._run("overlay.text", {"text": str(text), "x": float(x), "y": float(y) - self.ij.font_size, "font_size": self.ij.font_size, "color": self.ij._color_hex(), "at_selection": False})
    def drawLine(self, x1, y1, x2, y2):
        self.ij.img.overlay.append({"id": roimod.new_id(), "type": "line", "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "width": self.ij.line_width, "color": self.ij._color_hex(), "stroke_width": self.ij.line_width})
    def drawRect(self, x, y, w, h):
        self.ij.img.overlay.append({"id": roimod.new_id(), "type": "rect", "x": float(x), "y": float(y), "w": float(w), "h": float(h), "color": self.ij._color_hex(), "stroke_width": self.ij.line_width})
    def drawEllipse(self, x, y, w, h):
        self.ij.img.overlay.append({"id": roimod.new_id(), "type": "oval", "x": float(x), "y": float(y), "w": float(w), "h": float(h), "color": self.ij._color_hex(), "stroke_width": self.ij.line_width})
    def add(self): self.ij.session._changed("images")
    def removeSelection(self, i):
        ov = self.ij.img.overlay
        if 0 <= int(i) < len(ov):
            ov.pop(int(i))
    def activateSelection(self, i):
        ov = self.ij.img.overlay
        if 0 <= int(i) < len(ov) and (roimod.is_area(ov[int(i)]) or roimod.is_line(ov[int(i)]) or ov[int(i)].get("type") == "point"):
            self.ij.roi = dict(ov[int(i)])
    def setPosition(self, *a): pass
    def getBounds(self, i):
        o = self.ij.img.overlay[int(i)]
        return roimod.bounding_box(o) if (roimod.is_area(o) or roimod.is_line(o) or o.get("type") == "point") else (o.get("x", 0), o.get("y", 0), 0, 0)
    def removeRois(self, name):
        self.ij.img.overlay = [o for o in self.ij.img.overlay if o.get("name") != name]
    def measure(self):
        for o in self.ij.img.overlay:
            if roimod.is_area(o) or roimod.is_line(o):
                self.ij.session.run_op("analyze.measure", {}, roi=o, record=False)
    def copy(self): self._pending = {"copy": [dict(o) for o in self.ij.img.overlay]}
    def paste(self): self.ij.img.overlay = [dict(o) for o in self._pending.get("copy", [])]
    def setStrokeColor(self, c):
        for o in self.ij.img.overlay:
            o["color"] = _color_hex(c)
    def setStrokeWidth(self, w):
        for o in self.ij.img.overlay:
            o["stroke_width"] = float(w)
    def drawLabels(self, b): pass
    def useNamesAsLabels(self, b): pass
    def setLabelFontSize(self, *a): pass
    def setLabelColor(self, *a): pass
    def indexAt(self, x, y):
        for i, o in enumerate(self.ij.img.overlay):
            if roimod.is_area(o) and roimod.contains_point(o, float(x), float(y)):
                return i
        return -1
    def flatten(self): self.ij._run("overlay.flatten", {})
    def cropAndSave(self, *a): raise ValueError("Unsupported macro function: Overlay.cropAndSave")
    def getType(self, i):
        return self.ij.img.overlay[int(i)].get("type", "")


class _TableFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij

    def size(self, *a): return self.ij.nResults()
    def get(self, col, row, *a): return self.ij.getResult(col, row)
    def getString(self, col, row, *a): return _str(self.ij.getResult(col, row))
    def set(self, col, row, value, *a): self.ij.setResult(col, row, value)
    def update(self, *a): self.ij.updateResults()
    def reset(self, *a): self.ij.session.results.clear(); self.ij.session._changed("results")
    def create(self, title=""): pass
    def getColumn(self, col, *a): return IJArray(r.get(col, float("nan")) for r in self.ij.session.results.rows)
    def setColumn(self, col, arr, *a):
        for i, v in enumerate(arr):
            self.ij.setResult(col, i, v)
    def headings(self, *a): return "\t".join(self.ij.session.results.columns)
    def title(self): return "Results"
    def deleteRows(self, first, last, *a): self.ij._delete_rows(int(first), int(last))
    def deleteColumn(self, col, *a):
        rt = self.ij.session.results
        for r in rt.rows:
            r.pop(col, None)
        rt.columns = [c for c in rt.columns if c != col]
    def columnExists(self, col, *a): return col in self.ij.session.results.columns
    def rename(self, a, b=None): pass
    def save(self, path, *a):
        with open(str(path), "w", encoding="utf-8") as f:
            f.write(self.ij.session.results.to_csv())
    def open(self, path, *a):  # noqa: A003
        import csv
        with open(str(path), encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        out = []
        for r in rows:
            row: dict[str, Any] = {}
            for k, v in r.items():
                if k in (None, " ", ""):
                    continue
                try:
                    row[k] = float(v) if v not in ("", "NaN") else float("nan")
                except ValueError:
                    row[k] = v
            out.append(row)
        self.ij.session.results.clear(); self.ij.session.results.append(out); self.ij.session._changed("results")
    def sort(self, col, *a):
        rt = self.ij.session.results
        rt.rows.sort(key=lambda r: (r.get(col) is None, r.get(col)))
    def showRowNumbers(self, *a): pass
    def showRowIndexes(self, *a): pass
    def setLocationAndSize(self, *a): pass
    def applyMacro(self, *a): raise ValueError("Unsupported macro function: Table.applyMacro")
    def getSelectionStart(self, *a): return -1
    def getSelectionEnd(self, *a): return -1
    def setSelection(self, *a): pass
    def showArrays(self, title, *arrays): pass


class _ColorFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij
    def setForeground(self, *c): self.ij.setForegroundColor(*c)
    def setBackground(self, *c): self.ij.setBackgroundColor(*c)
    def foreground(self): return self.ij._fg_hex
    def background(self): return self.ij._bg_hex
    def toString(self, r, g, b): return f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    def toArray(self, s):
        from .overlay_render import parse_color
        return IJArray(parse_color(s))
    def setLut(self, r, g, b): pass
    def getLut(self): return self.ij.getLut()
    def wavelengthToColor(self, wl): return "#ffffff"
    def set(self, *c): self.ij.setColor(*c)


class _PropertyFns:
    def __init__(self, ij: "IJ") -> None:
        self.ij = ij
    def set(self, k, v): self.ij.img.metadata[str(k)] = v
    def get(self, k): return self.ij.img.metadata.get(str(k), "")
    def getNumber(self, k):
        try:
            return float(self.ij.img.metadata.get(str(k)))
        except (TypeError, ValueError):
            return float("nan")
    def getInfo(self): return "\n".join(f"{k}={v}" for k, v in self.ij.img.metadata.items() if isinstance(v, (str, int, float)))
    def setInfo(self, s): self.ij.img.metadata["Info"] = str(s)
    def getSliceLabel(self): return self.ij.Stack.getSliceLabel()
    def setSliceLabel(self, s, n=None): self.ij.Stack.setSliceLabel(s, n)
    def getList(self): return self.getInfo()
    def setList(self, s):
        for line in str(s).splitlines():
            if "=" in line:
                k, v = line.split("=", 1); self.ij.img.metadata[k] = v


# ====================================================================== the IJ object
class _ExtFns:
    """``Ext.*`` – the Bio-Formats Macro Extensions (``run("Bio-Formats Macro Extensions")`` in a macro): a Bio-Formats
    style reader over ``lumen.core.io.list_series`` / ``read_image``.  Output-argument functions return their value and
    the transpiler assigns it to the macro variable (``Ext.getSizeX(w)`` → ``w = Ext.getSizeX()``), like ImageJ.
    Missing metadata reads back as ``"null"`` (what a Java null string prints as) and unknown physical sizes as NaN."""

    def __init__(self, ij: "IJ") -> None:
        self.ij = ij
        self.group_files = False
        self.open_all_series = False
        self.virtual = False
        self.normalized = False
        self.metadata_collected = True
        self.original_metadata_populated = True
        self.metadata_filtered = False
        self._enable()

    def _enable(self) -> None:
        """(Re)initialise the reader state – what run("Bio-Formats Macro Extensions") does."""
        self.enabled = True
        self.path: str | None = None
        self.infos: list[Any] = []
        self.series = 0
        self._images: dict[int, LumenImage] = {}
        self._used_files: list[str] = []
        self._format_cache: tuple[str, str] | None = None

    # ------------------------------------------------------------ reader state
    def _info(self):
        if self.path is None or not self.infos:
            raise ValueError("Ext: call Ext.setId(path) first")
        return self.infos[min(self.series, len(self.infos) - 1)]

    def setId(self, id: Any) -> None:  # noqa: A002
        from pathlib import Path
        from .io import IMAGE_EXTS, group_file_pattern, list_sequence_files, list_series
        p = Path(str(id)).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"Ext.setId: {p} does not exist")
        self.path = str(p)
        self.infos = list_series(p)
        self.series = 0
        self._images = {}
        self._used_files = [str(p)]
        if self.group_files and p.is_file():
            try:
                folder, regex, _axis = group_file_pattern(p)
                self._used_files = [str(f) for f in list_sequence_files(folder, regex, sort="natural", exts=IMAGE_EXTS)] or [str(p)]
            except Exception:
                pass

    def close(self) -> None:
        self._enable()

    def closeFileOnly(self) -> None:
        self._images = {}

    def getSeriesCount(self) -> int:
        return len(self.infos) if self.path is not None else 0

    def setSeries(self, n: Any) -> None:
        self._info()
        n = int(n)
        if not 0 <= n < len(self.infos):
            raise ValueError(f"Ext.setSeries: series {n} is out of range (0–{len(self.infos) - 1})")
        self.series = n

    def getSeries(self) -> int:
        return self.series

    def getSeriesName(self) -> str:
        return str(self._info().name)

    def getSizeX(self) -> int: return int(self._info().size_x)
    def getSizeY(self) -> int: return int(self._info().size_y)
    def getSizeZ(self) -> int: return int(self._info().size_z)
    def getSizeC(self) -> int: return int(self._info().size_c)
    def getSizeT(self) -> int: return int(self._info().size_t)
    def getImageCount(self) -> int: return int(self._info().image_count)
    def getPixelType(self) -> str: return str(self._info().pixel_type)
    def getDimensionOrder(self) -> str: return str(self._info().dimension_order)
    def getEffectiveSizeC(self) -> int: return int(self._info().effective_size_c)
    def getRGBChannelCount(self) -> int: return int(self._info().rgb_channel_count)
    def isInterleaved(self) -> int: return 1 if self._info().rgb else 0
    def isInterleavedSubC(self, sub_c: Any = 0) -> int: return self.isInterleaved()
    def isIndexed(self) -> int: return 0
    def isFalseColor(self) -> int: return 0
    def isLittleEndian(self) -> int: return 1
    def isOrderCertain(self) -> int: return 1
    def isThumbnailSeries(self) -> int: return 1 if "thumb" in str(self._info().name).lower() else 0
    def isMetadataComplete(self) -> int: return 1
    def getChannelDimCount(self) -> int: return 1
    def getChannelDimLength(self, dim: Any = 0) -> int: return int(self._info().effective_size_c)
    def getChannelDimType(self, dim: Any = 0) -> str: return "Channel"
    def getResolutionCount(self) -> int: return int(self._info().level_count)

    def _size(self, v: Any) -> float:
        return float("nan") if v is None else float(v)

    def getPixelsPhysicalSizeX(self) -> float: return self._size(self._info().pixel_width)
    def getPixelsPhysicalSizeY(self) -> float: return self._size(self._info().pixel_height)
    def getPixelsPhysicalSizeZ(self) -> float: return self._size(self._info().pixel_depth)
    def getPixelsTimeIncrement(self) -> float: return self._size(self._info().time_increment)

    def getImageCreationDate(self) -> str:
        return str(self._info().metadata.get("AcquisitionDate") or self._info().metadata.get("DateTime") or "")

    def _meta_value(self, meta: dict[str, Any], field: Any) -> str:
        key = str(field)
        if key in meta:
            v = meta[key]
        else:
            low = {str(k).lower(): v for k, v in meta.items()}
            if key.lower() not in low:
                return "null"
            v = low[key.lower()]
        return str(v).lower() if isinstance(v, bool) else _str(v) if isinstance(v, (int, float)) else str(v)

    def getMetadataValue(self, field: Any) -> str:
        merged: dict[str, Any] = {}
        for info in reversed(self.infos):
            merged.update(info.metadata)
        return self._meta_value(merged, field)

    def getSeriesMetadataValue(self, field: Any) -> str:
        return self._meta_value(self._info().metadata, field)

    def getFormat(self, id: Any) -> str:  # noqa: A002
        from .io import list_series
        key = str(id)
        if self._format_cache and self._format_cache[0] == key:
            return self._format_cache[1]
        infos = self.infos if key == self.path and self.infos else list_series(key)
        fmt = str(infos[0].format) if infos else "null"
        self._format_cache = (key, fmt)
        return fmt

    def getCurrentFile(self) -> str:
        return self.path if self.path is not None else "null"

    def getUsedFileCount(self) -> int:
        return len(self._used_files)

    def getUsedFile(self, i: Any) -> str:
        i = int(i)
        if not 0 <= i < len(self._used_files):
            raise ValueError(f"Ext.getUsedFile: index {i} is out of range (0–{len(self._used_files) - 1})")
        return self._used_files[i]

    def _plane_entry(self, no: Any) -> dict[str, Any]:
        info = self._info()
        z, c, t = info.zct_coords(int(no))
        for pl in info.planes:
            if (pl.get("z", 0), pl.get("c", 0), pl.get("t", 0)) == (z, c, t):
                return pl
        return {}

    def getPlaneTimingDeltaT(self, no: Any = 0) -> float: return self._size(self._plane_entry(no).get("delta_t"))
    def getPlaneTimingExposureTime(self, no: Any = 0) -> float: return self._size(self._plane_entry(no).get("exposure_time"))
    def getPlanePositionX(self, no: Any = 0) -> float: return self._size(self._plane_entry(no).get("position_x"))
    def getPlanePositionY(self, no: Any = 0) -> float: return self._size(self._plane_entry(no).get("position_y"))
    def getPlanePositionZ(self, no: Any = 0) -> float: return self._size(self._plane_entry(no).get("position_z"))

    @staticmethod
    def _truthy(v: Any) -> bool:
        return v if isinstance(v, bool) else (str(v).strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v))

    def setGroupFiles(self, flag: Any) -> None: self.group_files = self._truthy(flag)
    def isGroupFiles(self) -> int: return int(self.group_files)
    def setOpenAllSeries(self, flag: Any) -> None: self.open_all_series = self._truthy(flag)
    def isOpenAllSeries(self) -> int: return int(self.open_all_series)
    def setVirtual(self, flag: Any) -> None: self.virtual = self._truthy(flag)
    def isVirtual(self) -> int: return int(self.virtual)
    def setNormalized(self, flag: Any) -> None: self.normalized = self._truthy(flag)
    def isNormalized(self) -> int: return int(self.normalized)
    def setMetadataCollected(self, flag: Any) -> None: self.metadata_collected = self._truthy(flag)
    def isMetadataCollected(self) -> int: return int(self.metadata_collected)
    def setOriginalMetadataPopulated(self, flag: Any) -> None: self.original_metadata_populated = self._truthy(flag)
    def isOriginalMetadataPopulated(self) -> int: return int(self.original_metadata_populated)
    def setMetadataFiltered(self, flag: Any) -> None: self.metadata_filtered = self._truthy(flag)
    def isMetadataFiltered(self) -> int: return int(self.metadata_filtered)
    def fileGroupOption(self, id: Any = "") -> str: return "CAN_GROUP"  # noqa: A002

    def getIndex(self, z: Any, c: Any, t: Any) -> int:
        return int(self._info().plane_index(int(z), int(c), int(t)))

    def getZCTCoords(self, index: Any) -> tuple[int, int, int]:
        return self._info().zct_coords(int(index))

    def isThisType(self, name: Any) -> int:
        from .io import IMAGE_EXTS, _full_ext
        from pathlib import Path
        p = Path(str(name))
        return 1 if (_full_ext(p) in IMAGE_EXTS or p.suffix.lower() in IMAGE_EXTS) else 0

    def isThisTypeFast(self, name: Any) -> int:
        return self.isThisType(name)

    def getVersionNumber(self) -> str:
        from .. import __version__
        return f"Lumen {__version__} (native readers)"

    def getRevision(self) -> str:
        return "lumen-native"

    def getBuildDate(self) -> str:
        return ""

    # ------------------------------------------------------------ opening images
    def _series_image(self) -> LumenImage:
        from .io import read_image
        self._info()
        img = self._images.get(self.series)
        if img is None:
            img = read_image(self.path, series=self.series, virtual=True)
            self._images[self.series] = img
        return img

    def _open_plane(self, title: Any, no: Any, box: tuple[int, int, int, int] | None = None, thumb: bool = False) -> LumenImage:
        src = self._series_image()
        info = self._info()
        z, c, t = info.zct_coords(int(no))
        t, z = min(t, src.nt - 1), min(z, src.nz - 1)
        if src.rgb:
            plane = np.asarray(src.data[t, z])  # (3, Y, X)
        else:
            plane = np.asarray(src.data[t, z, min(c, src.nc - 1)])
        if box is not None:
            x, y, w, h = (int(v) for v in box)
            plane = plane[..., y:y + h, x:x + w]
        if thumb:
            step = max(1, int(np.ceil(max(plane.shape[-2:]) / 128)))
            plane = plane[..., ::step, ::step]
        img = LumenImage(np.ascontiguousarray(plane), name=str(title) or f"{info.name} plane {int(no)}", axes="CYX" if src.rgb else "YX", rgb=src.rgb,
                         calibration=Calibration(**src.calibration.to_dict()), metadata={"series": self.series, "plane": int(no), "source_file": self.path})
        if thumb and step > 1:
            img.calibration.pixel_width *= step
            img.calibration.pixel_height *= step
        if not src.rgb:
            ch = src.display.channels[min(c, src.nc - 1)]
            img.display.channels[0].lut, img.display.channels[0].min, img.display.channels[0].max = ch.lut, ch.min, ch.max
        self.ij.session.add_image(img)
        return img

    def openImage(self, title: Any, no: Any = 0) -> None:
        self._open_plane(title, no)

    def openSubImage(self, title: Any, no: Any, x: Any, y: Any, w: Any, h: Any) -> None:
        self._open_plane(title, no, box=(int(x), int(y), int(w), int(h)))

    def openThumbImage(self, title: Any, no: Any = 0) -> None:
        self._open_plane(title, no, thumb=True)

    def _open_file(self, path: Any, thumb: bool = False) -> None:
        from .ops.bioformats_ops import import_file
        p = str(path or self.path or "")
        if not p:
            raise ValueError("Ext.openImagePlus: no file given")
        images, _infos, _chosen, _notes = import_file(p, series=str(self.series), open_all_series=self.open_all_series, group_files=self.group_files,
                                                      virtual=self.virtual and not thumb)
        for img in images:
            if thumb:
                step = max(1, int(np.ceil(max(img.height, img.width) / 128)))
                if step > 1:
                    small = img.copy(data=np.ascontiguousarray(np.asarray(img.data)[..., ::step, ::step]))
                    small.rgb = img.rgb
                    small.calibration.pixel_width *= step
                    small.calibration.pixel_height *= step
                    img = small
            self.ij.session.add_image(img)

    def openImagePlus(self, path: Any = None) -> None:
        self._open_file(path)

    def openThumbImagePlus(self, path: Any = None) -> None:
        self._open_file(path, thumb=True)


class IJ:
    """The ImageJ façade: ``IJ.run(cmd, options)`` plus the macro functions as methods.

    ``ij.roi`` is the current selection (shared with ``lumen.roi`` when created from the scripting API)."""

    def __init__(self, session: Session, api: Any = None) -> None:
        self.session = session
        self._api = api
        self._own_roi: dict[str, Any] | None = None
        self._ids: dict[str, int] = {}
        self._out = io.StringIO()
        self._macro_depth = 0
        self._args: list[str] = []  # getArgument() stack: one entry per running macro (runMacro(path, arg) / eval(code, arg)); "" at top level
        self.selected_rois: list[int] = []
        self.line_width = 1
        self.font_size = 12
        self._fg = (255, 255, 255)
        self._bg = (0, 0, 0)
        self._color: tuple[int, int, int] | None = None
        self._value: float | None = None
        self.binary_iterations = 1
        self.batch_mode = False
        self._last_roi: dict[str, Any] | None = None  # Edit ▸ Selection ▸ Restore Selection
        self.gel_options: dict[str, Any] = {}  # Analyze ▸ Gels ▸ Gel Analyzer Options
        # macro tools (lumen.core.ops.macro_tools): the cursor states getCursorLoc() hands out one after the other (the last one
        # is repeated, so `while (flags&16!=0) getCursorLoc(…)` loops end), the keys setKeyDown()/isKeyDown() see, the tool name
        self._cursor: list[tuple[float, float, int, int]] = []
        self._keys: set[str] = set()
        self._tool: str = "rectangle"
        self.status: str = ""  # the last showStatus() text
        self.Array = _ArrayFns(self); self.Math = _MathFns(); self.String = _StringFns(); self.File = _FileFns(self); self.Dialog = _DialogFns(self); self.List = _ListFns(self)
        self.Fit = _FitFns(self); self.Stack = _StackFns(self); self.Roi = _RoiFns(self); self.Overlay = _OverlayFns(self); self.Table = _TableFns(self); self.Color = _ColorFns(self)
        self.Property = _PropertyFns(self)

    # ------------------------------------------------------------- state
    @property
    def roi(self) -> dict[str, Any] | None:
        return self._api.roi if self._api is not None else self._own_roi

    @roi.setter
    def roi(self, value: dict[str, Any] | None) -> None:
        value = roimod.normalize(value) if value is not None else None
        prev = self._api.roi if self._api is not None else self._own_roi
        if prev is not None and prev != value:
            self._last_roi = dict(prev)  # ImageJ keeps the previous selection for Restore Selection
        if self._api is not None:
            self._api.roi = value
        else:
            self._own_roi = value

    @property
    def img(self) -> LumenImage:
        img = self.session.active
        if img is None:
            raise ValueError("There are no images open")
        return img

    @property
    def log_text(self) -> str:
        return self._out.getvalue()

    def _image(self, key: Any) -> LumenImage:
        if isinstance(key, (int, float, np.integer, np.floating)) and not isinstance(key, bool):
            k = int(key)
            if k < 0:
                for iid, n in self._ids.items():
                    if n == k and iid in self.session.images:
                        return self.session.images[iid]
                raise ValueError(f"Image {k} not found")
            imgs = list(self.session.images.values())
            if 1 <= k <= len(imgs):
                return imgs[k - 1]
            raise ValueError(f"Image {k} not found")
        key = str(key)
        if key in self.session.images:
            return self.session.images[key]
        for im in self.session.images.values():
            if im.name == key:
                return im
        for im in self.session.images.values():  # case-insensitive / without extension ("blobs.gif" finds "blobs.tif")
            if im.name.lower() == key.lower() or os.path.splitext(im.name)[0].lower() == os.path.splitext(key)[0].lower():
                return im
        raise ValueError(f'No image named "{key}"')

    def _mask(self) -> np.ndarray | None:
        if self.roi is None or not roimod.is_area(self.roi):
            return None
        return roimod.to_mask(self.roi, (self.img.height, self.img.width))

    def _run(self, op_id: str, params: dict[str, Any] | None = None, scope: str = "stack", image_id: str | None = None, use_roi: bool = True) -> dict[str, Any]:
        out = self.session.run_op(op_id, params or {}, image_id=image_id, roi=self.roi if use_roi else None, scope=scope, record=False)
        if out.get("clear_roi"):
            self.roi = None
        elif out.get("roi") is not None:
            self.roi = out["roi"]
        return out

    def _color_hex(self) -> str:
        c = self._color or self._fg
        return f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}"

    @property
    def Ext(self) -> _ExtFns:
        """The Bio-Formats Macro Extensions namespace (``Ext.setId`` … – see :class:`_ExtFns`)."""
        ext = self.__dict__.get("_ext")
        if ext is None:
            ext = self.__dict__["_ext"] = _ExtFns(self)
        return ext

    @property
    def _fg_hex(self) -> str:
        return f"#{self._fg[0]:02x}{self._fg[1]:02x}{self._fg[2]:02x}"

    @property
    def _bg_hex(self) -> str:
        return f"#{self._bg[0]:02x}{self._bg[1]:02x}{self._bg[2]:02x}"

    # ------------------------------------------------------------- run()
    @_mf
    def run(self, command: str, options: str = "") -> Any:
        """Execute an ImageJ menu command with an option string."""
        command = str(command)
        o = parse_options(str(options) if options is not None else "")
        entry = COMMANDS.get(command)
        if entry is None and command.strip() != command:
            entry = COMMANDS.get(command.strip())
        if entry is None and command == "Rotate..." and self.roi is None:
            entry = COMMANDS.get("Rotate... ")
        if entry is None and command == "Scale..." and self.roi is not None and o.flag("centered"):
            entry = COMMANDS.get("Scale... ")
        if entry is None:
            if command in LUT_NAMES:
                return self._run("color.lut", {"lut": command})
            low = {n.lower(): n for n in LUT_NAMES}
            if command.lower() in low:
                return self._run("color.lut", {"lut": low[command.lower()]})
            # fall back to a Lumen op id / name
            try:
                spec = REGISTRY.get(command)
            except KeyError:
                raise ValueError(f"Unsupported ImageJ command: {command!r}") from None
            if getattr(spec.func, "fiji_plugin", None):  # an installed Fiji plugin (ops/fiji_plugins.py): the options go to its dialog verbatim
                return self._run(spec.id, {"options": str(options or "")})
            # plugin tools (tool=True) also take the interaction parameters (x, y, x2, y2, button, shift, alt, ctrl, event) so
            # run("Spot Click Counter", "x=10 y=12") works; values stay raw option text / True flags – run_op's coerce_params types them
            names = list(spec.params) + (TOOL_PARAMS if spec.tool else [])
            params = {p.name: o[p.name] for p in names if p.name in o}
            return self._run(spec.id, params, scope=_scope(o))
        fn = entry[0]
        try:  # guard only the signature introspection, never the handler call
            takes_cmd = "command" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            takes_cmd = False
        return fn(self, o, command) if takes_cmd else fn(self, o)

    def transpile(self, text: str) -> str:
        return transpile(text)

    def macro(self, text: str, arg: Any = "", env: "dict[str, Any] | None" = None) -> str:
        """Run ImageJ-1 macro code; returns what the macro printed.  ``arg`` is what getArgument() returns inside it.
        Nested calls (eval(code), runMacro(path)) share the caller's Log buffer, like ImageJ's single Log window,
        and return only their own output.

        ``env`` is an optional persistent namespace: an installed macro set's ``var`` globals and functions live in it
        between runs (ImageJ keeps them while the set is installed – a tool's "Tool Options" macro sets a ``var`` the tool
        reads on the next click).  The macro functions are (re)bound to this IJ instance on every call; names the macro
        code itself (re)defined, such as a function shadowing a built-in, are kept."""
        py = transpile(text)
        try:
            code = compile(py, "<ij-macro>", "exec")
        except SyntaxError as e:  # the transpiler emitted Python that does not compile: report it like any other unsupported construct
            raise ValueError(f"Unsupported macro construct: {(e.text or '').strip()!r} ({e.msg})") from e
        base = self.macro_globals()
        if env is None:
            g = base
        else:
            mine = env.get("__ij_names__") or set()
            for k, v in base.items():
                if k not in env or k in mine:
                    env[k] = v
            g = env
        depth = self._macro_depth
        if depth == 0:
            self._out = io.StringIO()
        buf, start = self._out, len(self._out.getvalue())
        self._macro_depth = depth + 1
        self._args.append(arg if isinstance(arg, str) else _str(arg))  # _str: a numeric argument reads "2", not "2.0"
        try:
            exec(code, g)
        except MacroExit as e:
            if str(e):
                self.print(str(e))
        except NameError as e:
            name = getattr(e, "name", None) or str(e).split("'")[1] if "'" in str(e) else str(e)
            raise ValueError(f"Unsupported macro function or undefined variable: {name}") from e
        except AttributeError as e:
            raise ValueError(f"Unsupported macro function: {e}") from e
        finally:
            self._args.pop()
            self._macro_depth = depth  # an error inside eval() must not leave the depth > 0 (nor a stale argument)
            if env is not None:  # remember which names are still ours, so the next call rebinds them (and only them)
                env["__ij_names__"] = {k for k, v in base.items() if env.get(k) is v}
            self.session._changed("images")
        full = self._out.getvalue()
        # outermost call, or print("\\Clear") swapped the buffer during the nested call → whole buffer; else only this call's text
        return full if depth == 0 or self._out is not buf else full[start:]

    def macro_globals(self) -> dict[str, Any]:
        g: dict[str, Any] = {"__name__": "__ijmacro__", "__builtins__": __builtins__, "_add": _add, "_div": _div, "_idx": _idx, "_len": _len, "_outs": _outs, "_str": _str,
                             "_band": _band, "_bor": _bor, "_bxor": _bxor, "_shl": _shl, "_shr": _shr, "_bnot": _bnot, "_mod": _mod, "_call0": _call0, "_MacroExit": MacroExit, "math": math, "np": np,
                             "PI": math.pi, "NaN": float("nan"), "IJ": self, "true": True, "false": False}
        for name in MACRO_FUNCTION_NAMES:
            g[name] = getattr(self, name + "_" if name in ("is", "print", "open") and hasattr(self, name + "_") else name)
        g["is"] = self.is_
        g["print"] = self.print
        g["open"] = self.open
        for name in ("Array", "Math", "String", "File", "Dialog", "List", "Fit", "Stack", "Roi", "Overlay", "Table", "Color", "Property", "Ext"):
            g[name] = getattr(self, name)
        g["nResults"] = self.nResults; g["nSlices"] = self.nSlices; g["nImages"] = self.nImages
        for k in list(g):  # the transpiler renames Python keywords (is → is_, in → in_ …)
            if k in _KEYWORDS_PY:
                g[k + "_"] = g[k]
        return g

    def script_globals(self) -> dict[str, Any]:
        """Names safe to inject into a Python script namespace (no shadowing of builtins)."""
        import builtins
        g = self.macro_globals()
        return {k: v for k, v in g.items() if not k.startswith("_") and not hasattr(builtins, k) and k not in ("math", "np", "true", "false", "exit")}

    # ------------------------------------------------------------- images / windows
    @_mf
    def selectWindow(self, title: Any) -> None:
        if str(title) in ("Results", "Log", "ROI Manager", "Summary", "Recorder"):
            return
        self.session.select(self._image(title).id)

    @_mf
    def selectImage(self, id_or_title: Any) -> None:
        self.session.select(self._image(id_or_title).id)

    @_mf
    def getTitle(self) -> str:
        return self.img.name

    @_mf
    def getImageID(self) -> int:
        img = self.img
        if img.id not in self._ids:
            self._ids[img.id] = -(len(self._ids) + 1)
        return self._ids[img.id]

    @_mf
    def getWidth(self) -> int:
        return self.img.width

    @_mf
    def getHeight(self) -> int:
        return self.img.height

    @_mf
    def nSlices(self) -> int:
        img = self.img
        return img.nt * img.nz * (1 if img.rgb else img.nc)

    @_mf
    def nImages(self) -> int:
        return len(self.session.images)

    @_mf
    def getDimensions(self) -> tuple[int, int, int, int, int]:
        img = self.img
        return img.width, img.height, 1 if img.rgb else img.nc, img.nz, img.nt

    @_mf
    def getSliceNumber(self) -> int:
        img = self.img
        d = img.display
        nc = 1 if img.rgb else img.nc
        return (d.t * img.nz + d.z) * nc + (0 if img.rgb else d.c) + 1

    @_mf
    def setSlice(self, n: Any) -> None:
        img = self.img
        n = int(n)
        if not 1 <= n <= self.nSlices():
            raise ValueError(f"setSlice: {n} is out of range (1–{self.nSlices()})")
        nc = 1 if img.rgb else img.nc
        i = n - 1
        c = i % nc
        z = (i // nc) % img.nz
        t = i // (nc * img.nz)
        d = img.display
        d.t, d.z = t, z
        if not img.rgb:
            d.c = c
        img.clamp_position()

    @_mf
    def isOpen(self, key: Any) -> bool:
        try:
            self._image(key)
            return True
        except ValueError:
            return False

    @_mf
    def isActive(self, key: Any) -> bool:
        try:
            return self._image(key).id == self.img.id
        except ValueError:
            return False

    @_mf
    def bitDepth(self) -> int:
        img = self.img
        return 24 if img.rgb else {np.dtype(np.uint8): 8, np.dtype(np.uint16): 16, np.dtype(np.float32): 32}[img.data.dtype]

    def is_(self, what: str) -> bool:
        w = str(what).lower()
        img = self.session.active
        if w == "batch mode":
            return self.batch_mode
        if img is None:
            return False
        if w == "binary":
            return img.is_binary
        if w == "inverting lut":
            return img.display.channels[img.display.c].inverted
        if w == "composite":
            return img.nc > 1 and not img.rgb
        if w == "hyperstack":
            return img.nc > 1 or (img.nz > 1 and img.nt > 1)
        if w == "grayscale":
            return not img.rgb
        if w == "area":
            return roimod.is_area(self.roi)
        if w == "line":
            return roimod.is_line(self.roi)
        if w == "virtual stack":
            return False
        if w == "global scale":
            return False
        if w == "changes":
            return bool(self.session.histories.get(img.id) and self.session.histories[img.id].can_undo())
        if w == "animated":
            return False
        if w == "invertingLUT".lower():
            return img.display.channels[img.display.c].inverted
        if w == "locked":
            return False
        if w == "caps lock set":
            return False
        return False

    MACRO_FUNCTION_NAMES.append("is")

    @_mf
    def rename(self, title: Any) -> None:
        self._run("image.rename", {"title": str(title)})

    @_mf
    def close(self, pattern: Any = None) -> None:
        if pattern is None:
            if self.session.active is not None:
                self._run("file.close", {})
            return
        p = str(pattern)
        if p == "*":
            self._run("file.close_all", {})
            return
        if p in ("Results", "Summary"):
            self.session.results.clear(); self.session._changed("results")
            return
        if p in ("ROI Manager", "Log", "Recorder", "Threshold", "B&C"):
            return
        rx = re.escape(p).replace(r"\*", ".*")
        for im in list(self.session.images.values()):
            if re.fullmatch(rx, im.name):
                self.session.close_image(im.id)

    @_mf
    def open(self, path: Any = "", n: Any = None) -> LumenImage:  # noqa: A003
        if not path:
            raise ValueError("open(): a file path is required")
        img = self.session.open_file(str(path))
        # ImageJ's open-time display state (channel-1 range for every channel, red/green/blue composite colours) so that
        # conversions, Math and RGB rendering in the macro layer start from what ImageJ shows
        from .ops._imagej_types import imagej_display_patch
        p = imagej_display_patch(img)
        if p:
            self.session.update_display(img.id, p)
        return img

    @_mf
    def saveAs(self, fmt: Any, path: Any = None) -> str:
        if path is None:
            path, fmt = fmt, None
        f = str(fmt or "").lower().strip()
        p = str(path)
        if f in ("results", "measurements"):
            ext = os.path.splitext(p)[1].lower()
            if ext in (".xlsx", ".tsv"):
                self._run("file.export_results", {"path": p, "format": _results_format(p), "table": "Results", "row_numbers": True})
                return p
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(self.session.results.to_csv())
            return p
        if f == "text":
            # ImageJ saves the frontmost text window: the macro's Log output when it printed anything, else the Results table (or the session log)
            if self._out.getvalue():
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(self._out.getvalue())
                return p
            self._run("file.export_text", {"path": p, "source": "Results" if self.session.results.rows else "Log", "row_numbers": True})
            return p
        if f == "selection":
            if p.lower().endswith((".roi", ".zip")):
                if self.roi is None:
                    raise ValueError('saveAs("Selection"): no selection')
                self._run("file.export_ij_rois", {"path": p, "which": "Current selection"}, scope="plane")
                return p
            import json
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(self.roi, fh)
            return p
        if f in ("xy coordinates", "xy", "coordinates"):
            if self.roi is None:
                raise ValueError('saveAs("XY Coordinates"): no selection')
            self._run("file.save_xy_coordinates", {"path": p}, scope="plane")
            return p
        if f == "zip":
            return self._run("file.export_zip", {"path": p}).get("path", p if p.lower().endswith(".zip") else p + ".zip")
        if f == "lut":
            self._run("file.export_lut", {"path": p})
            return p
        fmt_map = {"tiff": "tif", "tif": "tif", "png": "png", "jpeg": "jpg", "jpg": "jpg", "": None}
        ext = fmt_map.get(f, f or None)
        p = str(path)
        if ext and not p.lower().endswith("." + ext):
            p = os.path.splitext(p)[0] + "." + ext
        return self.session.save_image(self.img.id, p, ext)

    @_mf
    def save(self, path: Any) -> str:
        return self.saveAs(None, path)

    @_mf
    def newImage(self, title: Any, type_: Any, width: Any, height: Any, *dims: Any) -> LumenImage:
        t = str(type_).lower()
        bits = "RGB" if "rgb" in t else ("16-bit" if "16" in t else ("32-bit" if "32" in t else "8-bit"))
        fill = "White" if "white" in t else ("Ramp" if "ramp" in t else ("Noise" if ("random" in t or "noise" in t) else "Black"))
        if len(dims) >= 3:
            c, z, f = (int(v) for v in dims[:3])
        else:
            c, z, f = 1, int(dims[0]) if dims else 1, 1
        out = self._run("file.new", {"title": str(title), "type": bits, "width": int(width), "height": int(height), "slices": z, "channels": c, "frames": f, "fill": fill})
        return self.session.images[out["new_images"][0]]

    @_mf
    def getInfo(self, key: Any = "") -> str:
        k = str(key)
        img = self.session.active
        if k in ("image.filename",):
            return os.path.basename(img.path) if img and img.path else ""
        if k in ("image.directory",):
            return os.path.dirname(img.path) + os.sep if img and img.path else ""
        if k in ("image.title", "window.title"):
            return img.name if img else ""
        if k == "image.subtitle":
            return f"{img.width}x{img.height} pixels; {img.bit_depth}" if img else ""
        if k == "selection.name":
            return (self.roi or {}).get("name", "")
        if k == "slice.label":
            return self.Stack.getSliceLabel()
        if k == "micrometer.abbreviation":
            return "µm"
        if k == "os.name":
            import platform
            return platform.system()
        if k in ("user.dir",):
            return os.getcwd()
        if k == "user.home":
            return os.path.expanduser("~")
        if k == "version":
            return "1.54-lumen"
        if k == "window.type":
            return "Image" if img else ""
        if k == "threshold.method":
            return (self.session.thresholds.get(img.id, {}) if img else {}).get("method", "Default")
        if k == "window.contents":
            return self._out.getvalue()
        if k == "log":
            return self._out.getvalue()
        if k == "overlay":
            return f"{len(img.overlay)} elements" if img and img.overlay else ""
        if k == "roi.manager":
            return str(len(self.session.roi_manager.rois))
        if k == "font.name":
            return "SansSerif"
        if k.startswith("image.") or k in ("Info",):
            return str(img.metadata.get("Info", "")) if img else ""
        return str(img.metadata.get(k, "")) if img and k in img.metadata else ""

    @_mf
    def getImageInfo(self) -> str:
        img = self.img
        cal = img.calibration
        return (f"Title: {img.name}\nWidth: {img.width} pixels\nHeight: {img.height} pixels\nBits per pixel: {self.bitDepth()}\nChannels: {img.nc}\nSlices: {img.nz}\nFrames: {img.nt}\n"
                f"Resolution: {1 / cal.pixel_width:.4g} pixels per {cal.unit}\n")

    @_mf
    def getDirectory(self, what: Any = "") -> str:
        w = str(what).lower()
        img = self.session.active
        if w == "image":
            return os.path.dirname(img.path) + os.sep if img and img.path else ""
        if w in ("temp",):
            import tempfile
            return tempfile.gettempdir() + os.sep
        if w in ("home", "startup", "imagej", "plugins", "macros", "luts", "file", "current", "cwd", ""):
            return (os.path.expanduser("~") if w == "home" else os.getcwd()) + os.sep
        return os.getcwd() + os.sep

    getDir = getDirectory
    MACRO_FUNCTION_NAMES.append("getDir")

    @_mf
    def getFileList(self, directory: Any) -> IJArray:
        d = str(directory)
        if not os.path.isdir(d):
            return IJArray()
        return IJArray(sorted(n + (os.sep if os.path.isdir(os.path.join(d, n)) else "") for n in os.listdir(d) if not n.startswith(".")))

    @_mf
    def getMetadata(self, key: Any = "Label") -> str:
        if str(key).lower() == "label":
            return self.Stack.getSliceLabel()
        return str(self.img.metadata.get("Info", ""))

    @_mf
    def setMetadata(self, key: Any, value: Any = None) -> None:
        if value is None:
            key, value = "Label", key
        if str(key).lower() == "label":
            self.Stack.setSliceLabel(value)
        else:
            self.img.metadata["Info"] = str(value)

    @_mf
    def snapshot(self) -> None:
        img = self.img
        self.session.histories[img.id].push(img, "Snapshot")

    @_mf
    def reset(self) -> None:
        self.session.undo(self.img.id)

    @_mf
    def updateDisplay(self) -> None:
        img = self.img
        img.rev += 1
        img.sync_channels()
        self.session._changed("images")

    @_mf
    def getList(self, what: Any) -> IJArray:
        w = str(what)
        if w == "image.titles":
            return IJArray(im.name for im in self.session.images.values())
        if w == "threshold.methods":
            return IJArray(METHODS)
        if w == "LUTs":
            return IJArray(LUT_NAMES)
        if w == "window.titles":
            return IJArray()
        if w == "java.properties":
            return IJArray()
        return IJArray()

    # ------------------------------------------------------------- calibration & display
    @_mf
    def getVoxelSize(self) -> tuple[float, float, float, str]:
        cal = self.img.calibration
        return cal.pixel_width, cal.pixel_height, cal.pixel_depth, cal.unit

    @_mf
    def getPixelSize(self) -> tuple[str, float, float, float]:
        cal = self.img.calibration
        return cal.unit, cal.pixel_width, cal.pixel_height, cal.pixel_depth

    @_mf
    def setVoxelSize(self, w: Any, h: Any, d: Any, unit: Any) -> None:
        img = self.img
        img.calibration = Calibration(**{**img.calibration.to_dict(), "pixel_width": float(w), "pixel_height": float(h), "pixel_depth": float(d), "unit": str(unit)})

    @_mf
    def getMinAndMax(self) -> tuple[float, float]:
        ch = self.img.display.channels[self.img.display.c]
        return ch.min, ch.max

    @_mf
    def setMinAndMax(self, lo: Any, hi: Any, channels: Any = None) -> None:
        img = self.img
        patch = [{"min": float(lo), "max": float(hi)} if (channels is None and i == img.display.c) or (channels is not None and (int(channels) >> i) & 1) or img.rgb else {} for i in range(img.nc)]
        self.session.update_display(img.id, {"channels": patch})

    @_mf
    def resetMinAndMax(self) -> None:
        img = self.img
        c = img.display.c
        lo, hi = (0.0, 255.0) if img.data.dtype == np.uint8 else img.auto_range(c, 0.0)
        self.session.update_display(img.id, {"channels": [{"min": lo, "max": hi} if i == c else {} for i in range(img.nc)]})

    @_mf
    def getThreshold(self) -> tuple[float, float]:
        thr = self.session.thresholds.get(self.img.id)
        if thr is None:
            return -1.0, -1.0
        return float(thr["lower"]), float(thr["upper"])

    @_mf
    def setThreshold(self, lower: Any, upper: Any, mode: Any = None) -> None:
        self._run("threshold.set", {"lower": float(lower), "upper": float(upper)})

    @_mf
    def setAutoThreshold(self, method: Any = "Default") -> None:
        words = str(method).replace("dark", " dark ").split()
        name = words[0] if words else "Default"
        if name not in METHODS:
            low = {m.lower(): m for m in METHODS}
            name = low.get(name.lower(), "Default")
        self._run("threshold.compute", {"method": name, "dark_background": "dark" in [w.lower() for w in words]})

    @_mf
    def resetThreshold(self) -> None:
        img = self.session.active
        if img is not None:
            self.session.thresholds.pop(img.id, None)
            self.session._changed("images")

    @_mf
    def getLut(self) -> tuple[IJArray, IJArray, IJArray]:
        from .luts import get_lut
        t = get_lut(self.img.display.channels[self.img.display.c].lut)
        return IJArray(t[:, 0].tolist()), IJArray(t[:, 1].tolist()), IJArray(t[:, 2].tolist())

    @_mf
    def setLut(self, r: Any, g: Any, b: Any) -> None:
        raise ValueError("Unsupported macro function: setLut (use run(\"<LUT name>\"))")

    # ------------------------------------------------------------- selections
    def _set(self, roi: dict[str, Any]) -> None:
        self.roi = roi

    @_mf
    def makeRectangle(self, x: Any, y: Any, w: Any, h: Any, arc: Any = 0) -> None:
        self._set({"type": "rect", "x": float(x), "y": float(y), "w": float(w), "h": float(h)})

    @_mf
    def makeOval(self, x: Any, y: Any, w: Any, h: Any) -> None:
        self._set({"type": "oval", "x": float(x), "y": float(y), "w": float(w), "h": float(h)})

    @_mf
    def makeEllipse(self, x1: Any, y1: Any, x2: Any, y2: Any, aspect: Any) -> None:
        cx, cy = (float(x1) + float(x2)) / 2, (float(y1) + float(y2)) / 2
        a = math.hypot(float(x2) - float(x1), float(y2) - float(y1)) / 2
        b = a * float(aspect)
        ang = math.atan2(float(y2) - float(y1), float(x2) - float(x1))
        th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        xs = cx + a * np.cos(th) * math.cos(ang) - b * np.sin(th) * math.sin(ang)
        ys = cy + a * np.cos(th) * math.sin(ang) + b * np.sin(th) * math.cos(ang)
        self._set({"type": "polygon", "subtype": "ellipse", "points": np.stack([xs, ys], 1).tolist()})

    @_mf
    def makeLine(self, *coords: Any) -> None:
        c = [float(v) for v in coords]
        if len(c) == 4:
            self._set({"type": "line", "x1": c[0], "y1": c[1], "x2": c[2], "y2": c[3], "width": self.line_width})
        elif len(c) == 5:
            self._set({"type": "line", "x1": c[0], "y1": c[1], "x2": c[2], "y2": c[3], "width": int(c[4])})
        elif len(c) >= 6 and len(c) % 2 == 0:
            self._set({"type": "polyline", "points": [[c[i], c[i + 1]] for i in range(0, len(c), 2)], "width": self.line_width})
        else:
            raise ValueError("makeLine needs 4 coordinates (or x1,y1,x2,y2,width, or an even list of points)")

    @_mf
    def makeArrow(self, x1: Any, y1: Any, x2: Any, y2: Any, style: Any = "") -> None:
        self._set({"type": "line", "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "width": self.line_width, "arrow": True, "arrow_style": str(style)})

    @_mf
    def makePolygon(self, *coords: Any) -> None:
        c = [float(v) for v in coords]
        if len(c) < 6 or len(c) % 2:
            raise ValueError("makePolygon needs at least three x,y pairs")
        self._set({"type": "polygon", "points": [[c[i], c[i + 1]] for i in range(0, len(c), 2)]})

    @_mf
    def makePoint(self, x: Any, y: Any, options: Any = "") -> None:
        self._set({"type": "point", "points": [[float(x), float(y)]]})

    @_mf
    def makeSelection(self, type_: Any, xs: Any, ys: Any) -> None:
        pts = [[float(a), float(b)] for a, b in zip(xs, ys)]
        t = str(type_).lower() if not isinstance(type_, (int, float)) else {0: "rectangle", 1: "oval", 2: "polygon", 3: "freehand", 4: "traced", 5: "line", 6: "polyline", 7: "freeline", 8: "angle", 10: "point"}.get(int(type_), "polygon")
        if t in ("point", "multipoint"):
            self._set({"type": "point", "points": pts})
        elif t in ("line", "straight line") and len(pts) == 2:
            self._set({"type": "line", "x1": pts[0][0], "y1": pts[0][1], "x2": pts[1][0], "y2": pts[1][1], "width": self.line_width})
        elif t in ("polyline", "freeline", "angle", "segmented line", "line"):
            self._set({"type": "polyline", "points": pts, "width": self.line_width})
        elif t in ("rectangle", "rect"):
            xs_, ys_ = [p[0] for p in pts], [p[1] for p in pts]
            self._set({"type": "rect", "x": min(xs_), "y": min(ys_), "w": max(xs_) - min(xs_), "h": max(ys_) - min(ys_)})
        elif t == "oval":
            xs_, ys_ = [p[0] for p in pts], [p[1] for p in pts]
            self._set({"type": "oval", "x": min(xs_), "y": min(ys_), "w": max(xs_) - min(xs_), "h": max(ys_) - min(ys_)})
        else:
            self._set({"type": "freehand" if t in ("freehand", "traced") else "polygon", "points": pts})

    @_mf
    def makeText(self, text: Any, x: Any, y: Any) -> None:
        self._run("overlay.text", {"text": str(text), "x": float(x), "y": float(y), "font_size": self.font_size, "color": self._color_hex(), "at_selection": False})

    @_mf
    def makeRotatedRectangle(self, x1: Any, y1: Any, x2: Any, y2: Any, width: Any) -> None:
        from .ops._imagej_measure import rotated_rect_polygon
        xs, ys = rotated_rect_polygon(float(x1), float(y1), float(x2), float(y2), float(width))   # ImageJ RotatedRectRoi.makeRectangle
        self._set({"type": "polygon", "subtype": "rotated_rect", "points": [[float(x), float(y)] for x, y in zip(xs, ys)]})

    @_mf
    def selectionType(self) -> int:
        r = self.roi
        if r is None:
            return -1
        return {"rect": 0, "oval": 1, "polygon": 2, "freehand": 3, "mask": 9, "line": 5, "polyline": 6, "point": 10}.get(r["type"], 9)

    @_mf
    def selectionName(self) -> str:
        return (self.roi or {}).get("name", "")

    @_mf
    def setSelectionName(self, name: Any) -> None:
        if self.roi is not None:
            r = dict(self.roi); r["name"] = str(name); self.roi = r

    @_mf
    def getSelectionBounds(self) -> tuple[int, int, int, int]:
        """ImageJ's ``Roi.getBounds()``: integer bounds (floor of the float bounds, width rounded) – the whole image without a selection."""
        if self.roi is None:
            return 0, 0, int(self.img.width), int(self.img.height)
        return roimod.int_bounds(self.roi)

    getBoundingRect = getSelectionBounds
    MACRO_FUNCTION_NAMES.append("getBoundingRect")

    @_mf
    def getSelectionCoordinates(self) -> tuple[IJArray, IJArray]:
        r = self.roi
        if r is None:
            raise ValueError("getSelectionCoordinates: no selection")
        if roimod.is_line(r) or r["type"] == "point":
            pts = np.asarray(roimod.line_points(r) if roimod.is_line(r) else r["points"], float)
        else:
            pts = roimod.polygon_points(r)
        return IJArray(pts[:, 0].tolist()), IJArray(pts[:, 1].tolist())

    @_mf
    def getLine(self) -> tuple[float, float, float, float, float]:
        r = self.roi
        if r is None or r.get("type") != "line":
            return -1.0, -1.0, -1.0, -1.0, float(self.line_width)
        return r["x1"], r["y1"], r["x2"], r["y2"], float(r.get("width", 1))

    @_mf
    def getProfile(self) -> IJArray:
        """``getProfile()``: ImageJ's ``ProfilePlot`` values of the current line or rectangle selection – a straight line is
        sampled at its own coordinates (``ImageProcessor.getLine``, bilinear), a wide line through ``Straightener``, a
        rectangle gives its column averages (row averages with the alt key in ImageJ – not in Lumen's macro layer)."""
        if self.roi is None:
            raise ValueError("Selection required")
        from .ops._imagej_roi import profile_plot, rgb_gray_plane
        img = self.img
        d = img.display
        plane = rgb_gray_plane(img.data[d.t, d.z]) if img.rgb else img.data[d.t, d.z, d.c]
        return IJArray([float(v) for v in profile_plot(plane, self.roi)])

    @_mf
    def setLineWidth(self, w: Any) -> None:
        self.line_width = max(int(w), 1)
        if self.roi is not None and roimod.is_line(self.roi):
            r = dict(self.roi); r["width"] = self.line_width; self.roi = r

    @_mf
    def restoreSelection(self) -> None:
        """Re-activate the previous selection (the one replaced or removed since), like ImageJ's Edit ▸ Selection ▸ Restore Selection."""
        if self._last_roi is not None:
            self.roi = dict(self._last_roi)

    @_mf
    def setKeyDown(self, keys: Any) -> None:
        """``setKeyDown("shift")``, ``setKeyDown("alt space")``, ``setKeyDown("none")``: the keys isKeyDown() reports from now on
        (ImageJ also feeds them to the next selection action; Lumen only records them)."""
        words = {w for w in str(keys).lower().replace(",", " ").split() if w}
        if "none" in words or not words:
            self._keys.clear()
            return
        self._keys |= {{"control": "ctrl", "option": "alt", "command": "meta", "cmd": "meta"}.get(w, w) for w in words}

    @_mf
    def isKeyDown(self, key: Any) -> bool:
        """``isKeyDown("shift"|"alt"|"ctrl"|"space"|"meta")``: keys of the interaction a macro tool was run with, or set with setKeyDown()."""
        k = str(key).lower().strip()
        return {"control": "ctrl", "option": "alt", "command": "meta", "cmd": "meta"}.get(k, k) in self._keys

    def set_cursor(self, states: "list[tuple[float, float, int, int]]", keys: "set[str] | None" = None) -> None:
        """Feed the cursor states (x, y, z, flags) a macro tool's getCursorLoc() calls return, in order – the last one repeats;
        ``keys`` are the modifier names isKeyDown() reports (lumen.core.ops.macro_tools)."""
        self._cursor = [(float(x), float(y), int(z), int(f)) for x, y, z, f in states]
        if keys is not None:
            self._keys = set(keys)

    @_mf
    def setTool(self, name: Any) -> None:
        """``setTool("oval")`` / ``setTool(1)``: the tool getToolName() / toolID() report (the UI's tool rail is not switched)."""
        if isinstance(name, (int, float, np.integer, np.floating)) and not isinstance(name, bool):
            k = int(name)
            self._tool = next((n for n, i in IJ_TOOL_IDS.items() if i == k), self._tool if k >= IJ_CUSTOM_TOOL_ID else "rectangle")
            return
        s = str(name).strip()
        self._tool = s.lower() if s.lower() in IJ_TOOL_IDS else s

    @_mf
    def toolID(self) -> int:
        """ImageJ's Toolbar tool ids: 0 rectangle, 1 oval, 2 polygon, 3 freehand, 4 line, 5 polyline, 6 freeline, 7 point, 8 wand,
        9 text, 11 zoom, 12 hand, 13 dropper, 14 angle; macro tools get ids from 15 (CUSTOM1) in the order they were registered."""
        t = self._tool
        if t in IJ_TOOL_IDS:
            return IJ_TOOL_IDS[t]
        try:
            from .ops import macro_tools
            return macro_tools.tool_id_of(t)
        except Exception:
            return IJ_CUSTOM_TOOL_ID

    # ------------------------------------------------------------- pixels & stats
    @_mf
    def getPixel(self, x: Any, y: Any) -> float:
        img = self.img
        d = img.display
        xi, yi = int(x), int(y)
        if not (0 <= xi < img.width and 0 <= yi < img.height):
            return 0.0
        if img.rgb:
            r, g, b = (int(img.data[d.t, d.z, c, yi, xi]) for c in range(3))
            return (r << 16) | (g << 8) | b
        v = img.data[d.t, d.z, d.c, yi, xi]
        return int(v) if img.data.dtype != np.float32 else float(v)

    @_mf
    def getValue(self, *args: Any) -> Any:
        if len(args) == 2 and all(isinstance(a, (int, float)) for a in args):
            return self.getPixel(*args)
        key = str(args[0]) if args else ""
        img = self.session.active
        if key == "results.count":
            return self.nResults()
        if key == "selection.size":
            return self.Roi.size() if self.roi is not None else 0
        if key == "selection.width":
            return self.line_width
        if key in ("rgb.foreground", "color.foreground"):
            return (self._fg[0] << 16) | (self._fg[1] << 8) | self._fg[2]
        if key in ("rgb.background", "color.background"):
            return (self._bg[0] << 16) | (self._bg[1] << 8) | self._bg[2]
        if key == "image.size":
            return float(img.nbytes) if img else 0.0
        if key == "font.size":
            return self.font_size
        if key == "font.height":
            return self.font_size
        if key == "Mean raw" or key == "Mean":
            rows = measure_roi(img, self.roi, ["mean"])
            return rows[0].get("Mean", float("nan"))
        col = key.split(" raw")[0]
        from .ops.extras import _measurement_for_column
        rows = measure_roi(img, self.roi, [_measurement_for_column(col)])
        if col in rows[0]:
            return rows[0][col]
        raise ValueError(f"getValue: unknown key {key!r}")

    @_mf
    def setPixel(self, x: Any, y: Any, value: Any) -> None:
        img = self.img
        d = img.display
        xi, yi = int(x), int(y)
        if not (0 <= xi < img.width and 0 <= yi < img.height):
            return
        if img.rgb:
            v = int(value)
            for c, sh in ((0, 16), (1, 8), (2, 0)):
                img.data[d.t, d.z, c, yi, xi] = (v >> sh) & 255
        else:
            from .registry import cast_to
            img.data[d.t, d.z, d.c, yi, xi] = cast_to(np.array(float(value)), img.data.dtype)
        img.rev += 1

    @_mf
    def getStatistics(self) -> tuple[Any, ...]:
        img = self.img
        rows = measure_roi(img, self.roi, ["area", "mean", "min", "max", "std"])
        r = rows[0]
        counts, _edges = self.getHistogram(256)
        return r.get("Area", 0.0), r.get("Mean", float("nan")), r.get("Min", 0.0), r.get("Max", 0.0), r.get("StdDev", 0.0), counts

    @_mf
    def getRawStatistics(self) -> tuple[Any, ...]:
        img = self.img
        d = img.display
        p = img.data[d.t, d.z, d.c]
        mask = self._mask()
        vals = (p[mask] if mask is not None else p.ravel()).astype(np.float64)
        counts, _ = self.getHistogram(256)
        return int(vals.size), float(vals.mean()), float(vals.min()), float(vals.max()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0, counts

    @_mf
    def getHistogram(self, n_bins: Any = 256, hist_min: Any = None, hist_max: Any = None) -> tuple[IJArray, IJArray]:
        """``getHistogram(values, counts, nBins[, histMin, histMax])`` with ImageJ's bin assignment (``ImageStatistics``:
        8-bit 256 bins = the gray values, otherwise ``(int)((v − min) / binSize)`` over the selection's min … max)."""
        from .ops._imagej_measure import macro_histogram
        img = self.img
        d = img.display
        p = img.data[d.t, d.z, d.c]
        rect, mask = roimod.roi_rect_mask(self.roi if (self.roi is not None and roimod.is_area(self.roi)) else None, p.shape)
        values, counts = macro_histogram(p, rect, mask, int(n_bins), None if hist_min is None else float(hist_min), None if hist_max is None else float(hist_max))
        return IJArray(values.tolist()), IJArray(counts.tolist())

    # ------------------------------------------------------------- drawing
    def _rgb_from_args(self, *c: Any) -> tuple[int, int, int]:
        from .overlay_render import parse_color
        if len(c) == 3:
            return int(c[0]), int(c[1]), int(c[2])
        v = c[0]
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            iv = int(v)
            if self.session.active is not None and not self.session.active.rgb:
                g = max(0, min(255, iv)) if self.session.active.data.dtype == np.uint8 else iv
                return g, g, g
            return (iv >> 16) & 255, (iv >> 8) & 255, iv & 255
        return parse_color(v, (255, 255, 255))

    @_mf
    def setColor(self, *c: Any) -> None:
        if len(c) == 1 and isinstance(c[0], (int, float)) and not isinstance(c[0], bool):
            self._value = float(c[0])
            self._color = self._rgb_from_args(*c)
        else:
            self._color = self._rgb_from_args(*c)
            self._value = None

    @_mf
    def setForegroundColor(self, *c: Any) -> None:
        self._fg = self._rgb_from_args(*c)
        self.session.settings["foreground"] = self._gray(self._fg)

    @_mf
    def setBackgroundColor(self, *c: Any) -> None:
        self._bg = self._rgb_from_args(*c)
        self.session.settings["background"] = self._gray(self._bg)

    def _gray(self, rgb: tuple[int, int, int]) -> float:
        img = self.session.active
        v = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        if img is not None and img.data.dtype == np.uint16:
            return v / 255.0 * 65535.0
        return float(v)

    def _draw_value(self) -> float:
        if self._value is not None:
            return self._value
        return self._gray(self._color or self._fg)

    def _with_fg(self, op_id: str, params: dict[str, Any], roi: dict[str, Any]) -> None:
        saved = self.session.settings.get("foreground")
        self.session.settings["foreground"] = self._draw_value()
        try:
            self.session.run_op(op_id, params, roi=roi, scope="plane", record=False)
        finally:
            self.session.settings["foreground"] = saved

    @_mf
    def drawLine(self, x1: Any, y1: Any, x2: Any, y2: Any) -> None:
        self._with_fg("edit.draw", {"line_width": self.line_width}, {"type": "line", "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "width": self.line_width})

    @_mf
    def drawRect(self, x: Any, y: Any, w: Any, h: Any) -> None:
        self._with_fg("edit.draw", {"line_width": self.line_width}, {"type": "rect", "x": float(x), "y": float(y), "w": float(w), "h": float(h)})

    @_mf
    def drawOval(self, x: Any, y: Any, w: Any, h: Any) -> None:
        self._with_fg("edit.draw", {"line_width": self.line_width}, {"type": "oval", "x": float(x), "y": float(y), "w": float(w), "h": float(h)})

    @_mf
    def fillRect(self, x: Any, y: Any, w: Any, h: Any) -> None:
        self._with_fg("edit.fill", {"value": -1}, {"type": "rect", "x": float(x), "y": float(y), "w": float(w), "h": float(h)})

    @_mf
    def fillOval(self, x: Any, y: Any, w: Any, h: Any) -> None:
        self._with_fg("edit.fill", {"value": -1}, {"type": "oval", "x": float(x), "y": float(y), "w": float(w), "h": float(h)})

    @_mf
    def fill(self) -> None:
        if self.roi is None or not roimod.is_area(self.roi):
            raise ValueError("fill(): needs an area selection")
        self._with_fg("edit.fill", {"value": -1}, self.roi)

    @_mf
    def drawString(self, text: Any, x: Any, y: Any, background: Any = None) -> None:
        from .overlay_render import draw_text_into_plane
        img = self.img
        d = img.display
        self.session.histories[img.id].push(img, "drawString")
        chans = range(img.nc) if img.rgb else [d.c]
        col = self._color or self._fg
        for c in chans:
            value = col[c] if img.rgb else self._draw_value()
            img.data[d.t, d.z, c] = draw_text_into_plane(img.data[d.t, d.z, c], str(text), int(x), int(y), value, self.font_size)
        img.rev += 1
        self.session._changed("images")

    @_mf
    def setFont(self, name: Any = "SansSerif", size: Any = 12, style: Any = "") -> None:
        if isinstance(name, (int, float)) and not isinstance(name, bool):
            size = name
        self.font_size = max(int(size), 4)

    @_mf
    def setJustification(self, j: Any) -> None:
        pass

    @_mf
    def getStringWidth(self, s: Any) -> int:
        return int(len(str(s)) * self.font_size * 0.6)

    @_mf
    def floodFill(self, x: Any, y: Any, connectivity: Any = "4-connected") -> None:
        out = self.session.run_op("selection.wand", {"x": int(x), "y": int(y), "tolerance": 0, "connectivity": "8" if "8" in str(connectivity) else "4"}, record=False)
        if out.get("roi"):
            self._with_fg("edit.fill", {"value": -1}, out["roi"])

    @_mf
    def changeValues(self, v1: Any, v2: Any, v3: Any) -> None:
        img = self.img
        d = img.display
        self.session.histories[img.id].push(img, "changeValues")
        p = img.data[d.t, d.z, d.c]
        m = (p >= float(v1)) & (p <= float(v2))
        mask = self._mask()
        if mask is not None:
            m &= mask
        from .registry import cast_to
        p[m] = cast_to(np.array(float(v3)), img.data.dtype)
        img.rev += 1

    @_mf
    def setPasteMode(self, mode: Any) -> None:
        self.session.settings["paste_mode"] = _paste_mode(mode)

    # ------------------------------------------------------------- ROI manager
    def _rm(self) -> roimod.RoiManager:
        return self.session.roi_manager

    @_mf
    def roiManager(self, command: Any, *args: Any) -> Any:
        c = str(command).strip().lower()
        rm = self._rm()
        s = self.session
        if c == "add":
            if self.roi is None:
                raise ValueError("roiManager(\"Add\"): no selection")
            img = s.active
            # the manager owns its ids: the selection carries one (and after roiManager("select") the id of a manager
            # entry), so strip it or Add / Add + Delete would remove both entries (RoiManager.remove / update are id-keyed)
            r = rm.add({k: v for k, v in self.roi.items() if k != "id"}, image_id=img.id if img else None,
                       position={"t": img.display.t, "z": img.display.z, "c": img.display.c} if img else None)
            if args and isinstance(args[0], str):
                r["color"] = _color_hex(args[0])
            s._changed("rois")
            return None
        if c == "count":
            return len(rm.rois)
        if c == "index":
            return self.selected_rois[0] if self.selected_rois else -1
        if c == "select":
            idx = args[0] if args else 0
            if isinstance(idx, (list, tuple)):
                self.selected_rois = [int(i) for i in idx]
            else:
                self.selected_rois = [int(idx)]
            for i in self.selected_rois:
                if not 0 <= i < len(rm.rois):
                    raise ValueError(f"roiManager(\"Select\", {i}): index out of range (0–{len(rm.rois) - 1})")
            r = rm.rois[self.selected_rois[0]]
            if r.get("image_id") and r["image_id"] in s.images and len(args) > 1 and isinstance(args[0], str):
                s.select(r["image_id"])
            pos = r.get("position")
            img = s.active
            if pos and img is not None and (img.nz > 1 or img.nt > 1):
                img.display.t = min(pos.get("t", img.display.t), img.nt - 1); img.display.z = min(pos.get("z", img.display.z), img.nz - 1)
            self.roi = {k: v for k, v in r.items() if k not in ("image_id",)}
            return None
        if c == "deselect":
            self.selected_rois = []
            self.roi = None
            return None
        if c == "delete":
            if self.selected_rois:
                rm.remove([rm.rois[i]["id"] for i in self.selected_rois if i < len(rm.rois)])
            else:
                rm.clear()
            self.selected_rois = []
            s._changed("rois")
            return None
        if c == "reset":
            rm.clear(); self.selected_rois = []; s._changed("rois")
            return None
        if c == "update":
            if self.selected_rois and self.roi is not None:
                rm.update(rm.rois[self.selected_rois[0]]["id"], {k: v for k, v in self.roi.items() if k != "id"})
                s._changed("rois")
            return None
        if c == "rename":
            if self.selected_rois:
                rm.rois[self.selected_rois[0]]["name"] = str(args[0]) if args else ""
                s._changed("rois")
            return None
        if c == "measure":
            idx = self.selected_rois or list(range(len(rm.rois)))
            # the Set Measurements options apply exactly as for Measure and POST /api/rois/measure: invert Y, NaN empty
            # cells, display label, limit to threshold (the measured image's threshold, read from the session)
            opts = s.measure_options
            thr = None
            if opts.get("limit_to_threshold"):
                rec = s.thresholds.get(self.img.id)
                if rec is not None:
                    thr = (float(rec["lower"]), float(rec["upper"]))
            for i in idx:
                r = rm.rois[i]
                pos = r.get("position") or {}
                rows = measure_roi(self.img, r, s.measurements, t=pos.get("t"), z=pos.get("z"), c=pos.get("c"),
                                   limit_to_threshold=thr, label=r.get("name"), options=opts)
                if opts.get("display_label", True):  # apply_measure_options just dropped the column when 'Display label' is off
                    for row in rows:
                        row["Label"] = f"{self.img.name}:{r.get('name', i)}"
                s.results.append(rows)
            s._changed("results")
            return None
        if c in ("multi measure", "multi-measure", "multi_measure"):
            return self._run("analyze.multi_measure", {"one_row_per_slice": "one" in " ".join(str(a) for a in args).lower() or not args})
        if c in ("show all", "show none", "show all with labels", "show all without labels", "usenames", "remove slice info", "associate", "centered", "list", "sort by name"):
            if c == "sort by name":
                rm.rois.sort(key=lambda r: r.get("name", ""))
            return None
        if c == "sort":
            rm.rois.sort(key=lambda r: r.get("name", "")); s._changed("rois")
            return None
        if c in ("combine", "or"):
            idx = self.selected_rois or list(range(len(rm.rois)))
            r = roimod.combine([rm.rois[i] for i in idx], (self.img.height, self.img.width), "or")
            self.roi = r
            return None
        if c == "and":
            idx = self.selected_rois or list(range(len(rm.rois)))
            self.roi = roimod.combine([rm.rois[i] for i in idx], (self.img.height, self.img.width), "and")
            return None
        if c == "xor":
            idx = self.selected_rois or list(range(len(rm.rois)))
            self.roi = roimod.combine([rm.rois[i] for i in idx], (self.img.height, self.img.width), "xor")
            return None
        if c == "split":
            if self.roi is None:
                raise ValueError("roiManager(\"Split\"): no selection")
            from skimage.measure import label as sk_label
            m = roimod.to_mask(self.roi, (self.img.height, self.img.width))
            for r in roimod.label_to_rois(sk_label(m, connectivity=2)):
                rm.add(r, image_id=self.img.id)
            s._changed("rois")
            return None
        if c in ("fill", "draw"):
            idx = self.selected_rois or list(range(len(rm.rois)))
            for i in idx:
                self._with_fg("edit.fill" if c == "fill" else "edit.draw", {"value": -1} if c == "fill" else {"line_width": self.line_width}, rm.rois[i])
            return None
        if c == "set color":
            for i in (self.selected_rois or range(len(rm.rois))):
                rm.rois[i]["color"] = _color_hex(args[0] if args else "yellow")
            s._changed("rois")
            return None
        if c in ("set line width",):
            for i in (self.selected_rois or range(len(rm.rois))):
                rm.rois[i]["stroke_width"] = float(args[0]) if args else 1.0
            return None
        if c == "translate":
            dx, dy = float(args[0]), float(args[1])
            for i in (self.selected_rois or range(len(rm.rois))):
                rm.rois[i] = roimod.normalize(roimod.translate(rm.rois[i], dx, dy))
            s._changed("rois")
            return None
        if c == "save":
            import json
            path = str(args[0]) if args else "rois.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump([rm.rois[i] for i in (self.selected_rois or range(len(rm.rois)))], f)
            return None
        if c == "open":
            import json
            path = str(args[0]) if args else ""
            with open(path, encoding="utf-8") as f:
                rois = json.load(f)
            for r in rois:
                rm.add({k: v for k, v in r.items() if k not in ("id", "image_id")}, image_id=s.active_id)  # fresh ids, like file.import_rois_json
            s._changed("rois")
            return None
        if c in ("show all with labels", "remove channel info", "remove frame info"):
            return None
        raise ValueError(f"Unsupported roiManager command: {command!r}")

    # ------------------------------------------------------------- results
    @_mf
    def nResults(self) -> int:
        return len(self.session.results.rows)

    @_mf
    def getResult(self, column: Any, row: Any = None) -> float:
        rt = self.session.results
        if not rt.rows:
            return float("nan")
        i = len(rt.rows) - 1 if row is None else int(row)
        if not 0 <= i < len(rt.rows):
            raise ValueError(f"getResult: row {i} out of range (0–{len(rt.rows) - 1})")
        v = rt.rows[i].get(str(column))
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        if v is None:
            return float("nan")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    @_mf
    def getResultString(self, column: Any, row: Any = None) -> str:
        rt = self.session.results
        if not rt.rows:
            return "null"
        i = len(rt.rows) - 1 if row is None else int(row)
        v = rt.rows[i].get(str(column))
        return _str(v) if v is not None else "null"

    @_mf
    def getResultLabel(self, row: Any) -> str:
        rt = self.session.results
        return str(rt.rows[int(row)].get("Label", "")) if 0 <= int(row) < len(rt.rows) else ""

    @_mf
    def setResult(self, column: Any, row: Any, value: Any) -> None:
        rt = self.session.results
        i = int(row)
        col = str(column)
        if i == len(rt.rows):
            rt.append([{col: value}])
            return
        if not 0 <= i < len(rt.rows):
            raise ValueError(f"setResult: row {i} out of range")
        rt.rows[i][col] = value
        if col not in rt.columns:
            rt.columns.append(col)

    @_mf
    def updateResults(self) -> None:
        self.session._changed("results")

    def _delete_rows(self, first: int, last: int) -> None:
        rt = self.session.results
        rt.rows = [r for i, r in enumerate(rt.rows) if not first <= i <= last]
        self.session._changed("results")

    # ------------------------------------------------------------- misc
    @_mf
    def print(self, *args: Any) -> None:  # noqa: A003
        if args and isinstance(args[0], str) and args[0] in ("\\Clear",):
            self._out = io.StringIO()
            return
        if args and isinstance(args[0], str) and args[0].startswith("[") and args[0].endswith("]"):
            target, args = args[0][1:-1], args[1:]  # print("[window]", …): a text window of that name gets the line; other targets are ignored
            from .ops.text_ops import is_text_table
            tb = self.session.tables.get(target)
            if tb is not None and is_text_table(tb):
                self._run("text.append", {"name": target, "text": " ".join(_str(a) for a in args)})
                return
        text = " ".join(_str(a) for a in args)
        self._out.write(text + "\n")
        self.session.add_log(text)

    @_mf
    def log(self, *args: Any) -> None:
        self.print(*args)

    @_mf
    def wait(self, ms: Any) -> None:
        pass

    @_mf
    def setBatchMode(self, arg: Any = True) -> None:
        self.batch_mode = bool(arg) if not isinstance(arg, str) else arg.lower() not in ("false", "exit and display", "show")

    @_mf
    def getTime(self) -> int:
        return int(time.time() * 1000)

    @_mf
    def getDateAndTime(self) -> tuple[int, ...]:
        t = time.localtime()
        return t.tm_year, t.tm_mon - 1, (t.tm_wday + 1) % 7, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, int((time.time() % 1) * 1000)

    @_mf
    def exit(self, message: Any = "") -> None:  # noqa: A003
        raise MacroExit(str(message))

    @_mf
    def showMessage(self, *args: Any) -> None:
        self.print(*args)

    @_mf
    def showMessageWithCancel(self, *args: Any) -> None:
        self.print(*args)

    @_mf
    def showStatus(self, *args: Any) -> None:
        """The status-bar text: kept as ``ij.status`` (macro tools show it as their toast)."""
        self.status = " ".join(_str(a) for a in args).strip()

    @_mf
    def showProgress(self, *args: Any) -> None:
        pass

    @_mf
    def beep(self) -> None:
        pass

    @_mf
    def requires(self, version: Any) -> None:
        pass

    @_mf
    def setOption(self, name: Any, value: Any = True) -> None:
        # "ScaleConversions" = ImageConverter.setDoScaling: 16/32-bit → 8-bit and 32-bit → 16-bit scale from the display range (true,
        # the default) or clamp / round the raw values (false); "BlackBackground" = Prefs.blackBackground for the Binary commands;
        # the other options have no effect in Lumen
        key = str(name).strip().lower()
        if key == "scaleconversions":
            self.scale_conversions = bool(value)
        elif key == "blackbackground":
            self.black_background = bool(value)

    @_mf
    def getBoolean(self, message: Any = "", *rest: Any) -> bool:
        return True

    @_mf
    def getNumber(self, message: Any, default: Any = 0) -> float:
        return default

    @_mf
    def getString(self, message: Any, default: Any = "") -> str:
        return str(default)

    @_mf
    def getArgument(self) -> str:
        return self._args[-1] if self._args else ""

    @_mf
    def getVersion(self) -> str:
        return "1.54"

    @_mf
    def getZoom(self) -> float:
        return 1.0

    @_mf
    def setLocation(self, *args: Any) -> None:
        pass

    @_mf
    def getLocationAndSize(self) -> tuple[int, int, int, int]:
        return 0, 0, self.img.width, self.img.height

    @_mf
    def getCursorLoc(self) -> tuple[float, float, int, int]:
        """``getCursorLoc(x, y, z, flags)``: inside a macro tool the interaction's cursor states (see :meth:`set_cursor`): the press
        with the button flag set (16 left, 8 middle, 4 right/meta; 1 shift, 2 ctrl, 8 alt – ImageJ's AWT constants), for a drag the
        release position still pressed, then the release with the button bits cleared – so ImageJ's ``while (flags&16!=0)`` loops
        see the drag end.  Outside a tool interaction the cursor is unknown: (-1, -1, 0, 0)."""
        if not self._cursor:
            return -1, -1, 0, 0
        state = self._cursor.pop(0) if len(self._cursor) > 1 else self._cursor[0]
        return state

    @_mf
    def doCommand(self, command: Any) -> None:
        self.run(command)

    @_mf
    def pad(self, n: Any, length: Any) -> str:
        return _StringFns.pad(n, length)

    @_mf
    def deleteRows(self, first: Any, last: Any) -> None:
        self._delete_rows(int(first), int(last))

    @_mf
    def renameResults(self, *args: Any) -> None:
        pass

    @_mf
    def getToolName(self) -> str:
        """``IJ.getToolName()``: the running macro tool's name ("Circle Tool"), the tool set with setTool(), else "rectangle"."""
        return self._tool

    @_mf
    def freeMemory(self) -> str:
        return ""

    @_mf
    def currentMemory(self) -> int:
        return 0

    @_mf
    def maxMemory(self) -> int:
        return 0

    @_mf
    def redirectErrorMessages(self, *args: Any) -> None:
        pass

    @_mf
    def getFullVersion(self) -> str:
        return "1.54-lumen"

    @_mf
    def checksum(self, *args: Any) -> str:
        return ""

    @_mf
    def newArray(self, *args: Any) -> IJArray:
        if len(args) == 1 and isinstance(args[0], (int, float)) and not isinstance(args[0], bool):
            return IJArray([0] * int(args[0]))
        return IJArray(args)

    @_mf
    def lengthOf(self, x: Any) -> int:
        return len(x)

    @_mf
    def d2s(self, n: Any, decimals: Any) -> str:
        return ij_d2s(float(n), int(decimals))

    @_mf
    def toString(self, n: Any, decimals: Any = None) -> str:
        if decimals is not None:
            return self.d2s(n, decimals)
        return _str(n)

    @_mf
    def parseInt(self, s: Any, radix: Any = 10) -> float:
        try:
            return int(str(s).strip(), int(radix))
        except ValueError:
            try:
                return int(float(str(s).strip()))
            except ValueError:
                return float("nan")

    @_mf
    def parseFloat(self, s: Any) -> float:
        try:
            return float(str(s).strip())
        except ValueError:
            return float("nan")

    @_mf
    def isNaN(self, v: Any) -> bool:
        try:
            return math.isnan(float(v))
        except (TypeError, ValueError):
            return True

    @_mf
    def substring(self, s: Any, a: Any, b: Any = None) -> str:
        return str(s)[int(a):(int(b) if b is not None else None)]

    @_mf
    def indexOf(self, s: Any, sub: Any, start: Any = 0) -> int:
        return str(s).find(str(sub), int(start))

    @_mf
    def lastIndexOf(self, s: Any, sub: Any) -> int:
        return str(s).rfind(str(sub))

    @_mf
    def startsWith(self, s: Any, prefix: Any) -> bool:
        return str(s).startswith(str(prefix))

    @_mf
    def endsWith(self, s: Any, suffix: Any) -> bool:
        return str(s).endswith(str(suffix))

    @_mf
    def matches(self, s: Any, regex: Any) -> bool:
        try:
            return re.fullmatch(str(regex), str(s)) is not None
        except re.error:
            return str(s) == str(regex)

    @_mf
    def replace(self, s: Any, a: Any, b: Any) -> str:
        return _replace(s, str(a), str(b))

    @_mf
    def toUpperCase(self, s: Any) -> str:
        return str(s).upper()

    @_mf
    def toLowerCase(self, s: Any) -> str:
        return str(s).lower()

    @_mf
    def split(self, s: Any, delim: Any = None) -> IJArray:
        return _split(s, None if delim is None else str(delim))

    @_mf
    def charCodeAt(self, s: Any, i: Any) -> int:
        return ord(str(s)[int(i)])

    @_mf
    def fromCharCode(self, *codes: Any) -> str:
        return "".join(chr(int(c)) for c in codes)

    @_mf
    def toHex(self, n: Any) -> str:
        return format(int(n), "x")

    @_mf
    def toBinary(self, n: Any) -> str:
        return format(int(n), "b")

    @_mf
    def round(self, x: Any) -> int:  # noqa: A003
        return int(math.floor(float(x) + 0.5))

    @_mf
    def floor(self, x: Any) -> int:
        return int(math.floor(float(x)))

    @_mf
    def abs(self, x: Any) -> float:  # noqa: A003
        return abs(x)

    @_mf
    def sqrt(self, x: Any) -> float:
        return math.sqrt(x) if x >= 0 else float("nan")

    @_mf
    def pow(self, a: Any, b: Any) -> float:  # noqa: A003
        try:
            return math.pow(a, b)
        except (ValueError, OverflowError):
            return float("nan")

    @_mf
    def exp(self, x: Any) -> float:
        try:
            return math.exp(x)
        except OverflowError:
            return float("inf")

    @_mf
    def log(self, x: Any) -> float:  # noqa: F811
        return math.log(x) if x > 0 else (float("-inf") if x == 0 else float("nan"))

    @_mf
    def sin(self, x: Any) -> float:
        return math.sin(x)

    @_mf
    def cos(self, x: Any) -> float:
        return math.cos(x)

    @_mf
    def tan(self, x: Any) -> float:
        return math.tan(x)

    @_mf
    def atan(self, x: Any) -> float:
        return math.atan(x)

    @_mf
    def atan2(self, y: Any, x: Any) -> float:
        return math.atan2(y, x)

    @_mf
    def asin(self, x: Any) -> float:
        return math.asin(x)

    @_mf
    def acos(self, x: Any) -> float:
        return math.acos(x)

    @_mf
    def random(self, *args: Any) -> float:
        import random as _random
        if args and str(args[0]).lower() == "seed" and len(args) > 1:
            _random.seed(int(args[1]))
            return 0.0
        if args and str(args[0]).lower() == "gaussian":
            return _random.gauss(0.0, 1.0)
        return _random.random()

    @_mf
    def maxOf(self, a: Any, b: Any) -> Any:
        return max(a, b)

    @_mf
    def minOf(self, a: Any, b: Any) -> Any:
        return min(a, b)

    @_mf
    def eval(self, *args: Any) -> Any:  # noqa: A003
        if len(args) == 2 and str(args[0]).lower() in ("python", "script"):
            from .scripting import run_script
            out = run_script(self.session, str(args[1]))
            if out["error"]:
                raise ValueError(out["error"])
            return out["stdout"]
        if len(args) == 1:
            return self.macro(str(args[0]))
        if len(args) == 2:
            return self.macro(str(args[0]), args[1])  # eval(macroCode, arg): the argument is read with getArgument()
        raise ValueError("Unsupported macro function: eval (only eval(macroCode[, arg]) and eval(\"python\", code))")

    @_mf
    def call(self, *args: Any) -> Any:
        """``call("class.method", args…)``: only the Trainable Weka Segmentation statics are mapped (see :meth:`_weka_call`)."""
        name = str(args[0]) if args else ""
        if name.startswith(_WEKA_PREFIX):
            return self._weka_call(name[len(_WEKA_PREFIX):], tuple(args[1:]))
        # the Toolbar / Prefs statics macro tools use (ImageJ's call() returns strings)
        if name in ("ij.gui.Toolbar.getToolId", "ij.gui.Toolbar.getToolID"):
            return str(self.toolID())
        if name in ("ij.gui.Toolbar.getToolName", "ij.IJ.getToolName"):
            return self.getToolName()
        if name in ("ij.gui.Toolbar.setTool", "ij.IJ.setTool") and len(args) > 1:
            self.setTool(args[1])
            return ""
        if name == "ij.Prefs.get" and len(args) > 1:
            return str(self.session.settings.get("ij.prefs", {}).get(str(args[1]), args[2] if len(args) > 2 else ""))
        if name == "ij.Prefs.set" and len(args) > 2:
            self.session.settings.setdefault("ij.prefs", {})[str(args[1])] = str(args[2])
            return ""
        raise ValueError(f"Unsupported macro function: call({name!r}) – Java static methods are not available (only trainableSegmentation.Weka_Segmentation.*, "
                         "ij.gui.Toolbar.getToolId/getToolName/setTool and ij.Prefs.get/set are mapped)")

    # ------------------------------------------------------------- Trainable Weka Segmentation statics
    def _weka_state(self) -> dict[str, Any]:
        st = self.__dict__.get("_weka")
        if st is None:
            st = self.__dict__["_weka"] = {"three_d": False, "classes": ["class 1", "class 2"], "features": {}, "params": {}, "traces": 0}
        return st

    def _weka_op(self, name: str) -> str:
        return ("segment3d." if self._weka_state()["three_d"] else "segment.") + name

    def _weka_call(self, fn: str, args: tuple[Any, ...]) -> Any:
        """``call("trainableSegmentation.Weka_Segmentation.<fn>", …)`` over the segment.* classifier ops: the settings statics
        (createNewClass, changeClassName, setFeature, setMinimumSigma … setClassifier) are collected and used by trainClassifier()."""
        st = self._weka_state()
        a = [str(x) for x in args]

        def arg(i: int, default: str = "") -> str:
            return a[i] if i < len(a) else default

        def truthy(s: str) -> bool:
            return str(s).strip().lower() in ("1", "true", "yes", "on")

        def ensure_class(i: int) -> None:
            while len(st["classes"]) <= i:
                st["classes"].append(f"class {len(st['classes']) + 1}")

        if fn == "loadClassifier":
            return self._run(self._weka_op("load_classifier"), {"path": arg(0)})
        if fn == "saveClassifier":
            return self._run(self._weka_op("save_classifier"), {"path": arg(0)})
        if fn == "trainClassifier":
            op_id = self._weka_op("train_classifier")
            p: dict[str, Any] = dict(st["params"])
            p["classes"] = ",".join(st["classes"])
            if st["features"]:
                p["features"] = ",".join(k for k, v in st["features"].items() if v)
            allowed = _op_params(op_id)
            p = {k: v for k, v in p.items() if k in allowed}
            p["apply_now"] = False
            return self._run(op_id, p, scope="plane")
        if fn in ("getResult", "getProbability"):
            return self._run(self._weka_op("apply_classifier"), {"output": "Labels" if fn == "getResult" else "Probabilities"}, scope="plane")
        if fn == "applyClassifier":  # applyClassifier(inputDir, fileName, showResults, storeResults, probabilityMaps, storeDir)
            p = {"input_folder": arg(0), "output": "Probabilities" if truthy(arg(4, "false")) else "Labels", "use_3d": st["three_d"]}
            if arg(1):
                p["pattern"] = re.escape(arg(1)) + "$"
            if truthy(arg(3, "false")) and arg(5):
                p["output_folder"] = arg(5)
            return self._run("segment.apply_classifier_folder", p)
        if fn == "toggleOverlay":
            return self._run("segment.classifier_overlay", {"action": "Toggle"})
        if fn == "setOpacity":
            v = float(arg(0, "33") or 33)
            return self._run("segment.classifier_overlay", {"action": "Show", "opacity": v / 100.0 if v > 1 else v})
        if fn == "setFeature":  # setFeature("Gaussian_blur=true")
            name, _, val = arg(0).partition("=")
            st["features"][name.replace("_", " ").strip()] = truthy(val or "true")
            return None
        if fn in ("setMinimumSigma", "setMaximumSigma", "setMembraneThickness", "setMembranePatchSize"):
            key = {"setMinimumSigma": "min_sigma", "setMaximumSigma": "max_sigma", "setMembraneThickness": "membrane_thickness", "setMembranePatchSize": "membrane_patch_size"}[fn]
            st["params"][key] = float(arg(0, "0") or 0)
            return None
        if fn == "setClassBalance":
            st["params"]["balance_classes"] = truthy(arg(0, "true"))
            return None
        if fn == "setClassifier":  # setClassifier("weka.classifiers.trees.FastRandomForest", "-I 200 -K 2 -S 1")
            st["params"]["classifier"] = _weka_classifier(arg(0))
            for flag, key in (("-I", "n_trees"), ("-K", "features_per_node"), ("-depth", "max_depth")):
                m = re.search(re.escape(flag) + r"\s+(\d+)", arg(1))
                if m:
                    st["params"][key] = int(m.group(1))
            return None
        if fn == "createNewClass":
            st["classes"].append(arg(0) or f"class {len(st['classes']) + 1}")
            return None
        if fn == "changeClassName":
            i = int(float(arg(0, "0") or 0))
            ensure_class(i)
            st["classes"][i] = arg(1) or st["classes"][i]
            return None
        if fn == "addTrace":  # addTrace(classIndex, slice): the current selection becomes a training region of that class
            if self.roi is None:
                raise ValueError("Weka_Segmentation.addTrace: draw a selection first")
            i = int(float(arg(0, "0") or 0))
            ensure_class(i)
            img = self.img
            n = max(int(float(arg(1, "1") or 1)), 1) - 1
            pos = {"t": n if (img.nt > 1 and img.nz == 1) else img.display.t, "z": n if img.nz > 1 else img.display.z, "c": img.display.c}
            st["traces"] += 1
            r = dict(self.roi)
            r["name"] = f"{st['classes'][i]} {st['traces']}"
            self.session.roi_manager.add(r, image_id=img.id, position=pos)
            self.session._changed("rois")
            return None
        raise ValueError(f'Unsupported macro function: call("trainableSegmentation.Weka_Segmentation.{fn}")')

    @_mf
    def runMacro(self, path: Any, arg: Any = "") -> str:
        with open(str(path), encoding="utf-8") as f:
            return self.macro(f.read(), arg)


# the log() name clashes with the IJ.log (print) helper: in macros bare log(x) is the natural logarithm.
IJ.log = IJ.log  # type: ignore[method-assign]


# ====================================================================== macro transpiler
_KEYWORDS_PY = {"and", "as", "assert", "async", "await", "class", "def", "del", "elif", "except", "finally", "from", "global", "import", "in", "is", "lambda", "nonlocal", "not", "or",
                "pass", "raise", "try", "with", "yield", "None", "True", "False"}
# functions whose arguments are *output* variables: name -> (leading input args, max outputs)
OUT_FUNCS: dict[str, tuple[int, int]] = {
    "getDimensions": (0, 5), "getStatistics": (0, 6), "getRawStatistics": (0, 6), "getVoxelSize": (0, 4), "getPixelSize": (0, 4), "getMinAndMax": (0, 2), "getThreshold": (0, 2),
    "getSelectionBounds": (0, 4), "getBoundingRect": (0, 4), "getLine": (0, 5), "getCursorLoc": (0, 4), "getSelectionCoordinates": (0, 2), "getHistogram": (0, 2), "getLut": (0, 3),
    "getDateAndTime": (0, 8), "getLocationAndSize": (0, 4), "Stack.getDimensions": (0, 5), "Stack.getPosition": (0, 3), "Stack.getStatistics": (0, 5), "Stack.getUnits": (0, 5),
    "Array.getStatistics": (1, 4), "Roi.getBounds": (0, 4), "Roi.getFloatBounds": (0, 4), "Roi.getCoordinates": (0, 2), "Roi.getFloatCoordinates": (0, 2), "Roi.getSplineAnchors": (0, 2),
    "Overlay.getBounds": (1, 4), "Fit.getEquation": (1, 2),
    # Bio-Formats Macro Extensions (loci.plugins.macro.LociFunctions): output arguments after the inputs, except the
    # per-plane functions whose plane number follows the output (Ext.getPlaneTimingDeltaT(deltaT, no))
    **{f"Ext.{n}": (0, 1) for n in ("getSeriesCount", "getSeries", "getSeriesName", "getSizeX", "getSizeY", "getSizeZ", "getSizeC", "getSizeT", "getImageCount",
                                      "getPixelType", "getDimensionOrder", "getEffectiveSizeC", "getRGBChannelCount", "isInterleaved", "isIndexed", "isFalseColor",
                                      "isLittleEndian", "isOrderCertain", "isThumbnailSeries", "isMetadataComplete", "getChannelDimCount", "getResolutionCount",
                                      "getPixelsPhysicalSizeX", "getPixelsPhysicalSizeY", "getPixelsPhysicalSizeZ", "getPixelsTimeIncrement",
                                      "getImageCreationDate", "getCurrentFile", "getUsedFileCount", "isGroupFiles", "isOpenAllSeries", "isVirtual", "isNormalized",
                                      "isMetadataCollected", "isOriginalMetadataPopulated", "isMetadataFiltered", "getVersionNumber", "getRevision", "getBuildDate",
                                      "getPlaneTimingDeltaT", "getPlaneTimingExposureTime", "getPlanePositionX", "getPlanePositionY", "getPlanePositionZ")},
    **{f"Ext.{n}": (1, 1) for n in ("getMetadataValue", "getSeriesMetadataValue", "getFormat", "getUsedFile", "getChannelDimLength", "getChannelDimType",
                                      "isInterleavedSubC", "fileGroupOption", "isThisType", "isThisTypeFast")},
    "Ext.getIndex": (3, 1), "Ext.getZCTCoords": (1, 3),
}
_BARE_CALLS = {"nResults", "nSlices", "nImages", "getTime", "getTitle", "getWidth", "getHeight", "getImageID", "getSliceNumber", "selectionType", "bitDepth", "getZoom", "getVersion", "random",
               "toolID",  # tool macros: `if (toolID==15)`
               # statement-level no-argument built-ins: ImageJ's exit() checks nextToken()=='(' and Interpreter.getParens() is optional, so `exit;`, `beep;` … are legal
               "exit", "snapshot", "reset", "beep", "updateResults", "updateDisplay", "resetThreshold", "resetMinAndMax"}
# ImageJ namespaces whose members may be used without parentheses (`List.setMeasurements;`, `x = List.size;`, `File.separator`):
# the transpiler wraps a bare member in _call0(), which calls functions and passes attribute values through
_NAMESPACES = {"Array", "Math", "String", "File", "Dialog", "List", "Fit", "Stack", "Roi", "Overlay", "Table", "Color", "Property", "Ext"}


def _call0(v: Any) -> Any:
    return v() if callable(v) else v


class _Tok:
    __slots__ = ("kind", "value", "line")

    def __init__(self, kind: str, value: str, line: int) -> None:
        self.kind, self.value, self.line = kind, value, line

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.kind}:{self.value!r}"


_OPS = ["++", "--", "+=", "-=", "*=", "/=", "==", "!=", "<=", ">=", "&&", "||", "<<", ">>", "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~", "(", ")", "{", "}", "[", "]", ",", ";", ".", "?", ":"]


def tokenize(src: str) -> list[_Tok]:
    toks: list[_Tok] = []
    i, n, line = 0, len(src), 1
    while i < n:
        ch = src[i]
        if ch == "\n":
            line += 1; i += 1; continue
        if ch.isspace():
            i += 1; continue
        if src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += src.count("\n", i, j)
            i = j
            continue
        if ch in "\"'":
            j = i + 1
            buf = []
            while j < n and src[j] != ch:
                if src[j] == "\\" and j + 1 < n:
                    esc = src[j + 1]
                    buf.append({"n": "\n", "t": "\t", "r": "\r", "\\": "\\", '"': '"', "'": "'", "u": "\\u"}.get(esc, "\\" + esc))
                    j += 2
                    continue
                if src[j] == "\n":
                    line += 1
                buf.append(src[j]); j += 1
            toks.append(_Tok("STR", "".join(buf), line))
            i = j + 1
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and src[i + 1].isdigit()):
            m = re.match(r"0[xX][0-9a-fA-F]+|\d+\.?\d*(?:[eE][-+]?\d+)?|\.\d+(?:[eE][-+]?\d+)?", src[i:])
            toks.append(_Tok("NUM", m.group(0), line))
            i += len(m.group(0))
            continue
        if ch.isalpha() or ch == "_":
            m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", src[i:])
            toks.append(_Tok("ID", m.group(0), line))
            i += len(m.group(0))
            continue
        for op_ in _OPS:
            if src.startswith(op_, i):
                toks.append(_Tok("OP", op_, line))
                i += len(op_)
                break
        else:
            raise ValueError(f"Unsupported macro construct: unexpected character {ch!r} on line {line}")
    toks.append(_Tok("EOF", "", line))
    return toks


class _Transpiler:
    def __init__(self, src: str) -> None:
        self.src_lines = src.splitlines()
        self.toks = tokenize(src)
        self.i = 0
        self.lines: list[str] = []
        self.funcs: list[str] = []
        self.indent = 0
        self.var_names: list[str] = []
        self.loop_steps: list[str | None] = []
        self._do_n = 0
        self.in_function = False
        self.func_params: list[str] = []
        self._target = self.lines

    # ---- token helpers
    @property
    def tok(self) -> _Tok:
        return self.toks[self.i]

    def peek(self, k: int = 1) -> _Tok:
        return self.toks[min(self.i + k, len(self.toks) - 1)]

    def at(self, value: str, kind: str = "OP") -> bool:
        return self.tok.kind == kind and self.tok.value == value

    def at_id(self, value: str) -> bool:
        return self.tok.kind == "ID" and self.tok.value == value

    def take(self) -> _Tok:
        t = self.tok
        self.i += 1
        return t

    def expect(self, value: str, kind: str = "OP") -> _Tok:
        if not self.at(value, kind):
            self.fail(f"expected {value!r}")
        return self.take()

    def fail(self, why: str = "") -> None:
        t = self.tok
        text = self.src_lines[t.line - 1].strip() if 0 < t.line <= len(self.src_lines) else t.value
        raise ValueError(f"Unsupported macro construct: {text!r} (line {t.line}{', ' + why if why else ''})")

    def emit(self, line: str) -> None:
        self._target.append("    " * self.indent + line)

    def end_statement(self) -> None:
        if self.at(";"):
            while self.at(";"):
                self.take()
            return
        if self.tok.kind == "EOF" or self.at("}"):
            return
        prev = self.toks[self.i - 1]
        if self.tok.line > prev.line:
            return
        self.fail("missing ';'")

    # ---- program
    def run(self) -> str:
        while self.tok.kind != "EOF":
            self.statement()
        body = self.lines or ["pass"]
        return "\n".join(self.funcs + body) + "\n"

    def block_or_statement(self) -> None:
        self.indent += 1
        start = len(self._target)
        if self.at("{"):
            self.take()
            while not self.at("}"):
                if self.tok.kind == "EOF":
                    self.fail("missing '}'")
                self.statement()
            self.take()
        else:
            self.statement()
        if len(self._target) == start:
            self.emit("pass")
        self.indent -= 1

    def statement(self) -> None:
        if self.at(";"):  # ImageJ: ';' is an empty statement (so `if (c);` has an empty body)
            self.take()
            return
        t = self.tok
        if self.at("{"):
            self.take()
            while not self.at("}"):
                if self.tok.kind == "EOF":
                    self.fail("missing '}'")
                self.statement()
            self.take()
            return
        if t.kind == "ID":
            v = t.value
            if v == "if":
                return self.if_statement()
            if v == "while":
                return self.while_statement()
            if v == "for":
                return self.for_statement()
            if v == "do":
                return self.do_statement()
            if v == "function":
                return self.function_statement()
            if v == "var":
                return self.var_statement()
            if v == "return":
                self.take()
                if self.at(";") or self.at("}") or self.tok.kind == "EOF" or self.tok.line > t.line:
                    self.emit("return")
                else:
                    self.emit(f"return {self.expression()}")
                return self.end_statement()
            if v == "break":
                self.take(); self.emit("break"); return self.end_statement()
            if v == "continue":
                self.take()
                if self.loop_steps and self.loop_steps[-1]:
                    self.emit(self.loop_steps[-1])
                self.emit("continue")
                return self.end_statement()
            if v == "macro":
                self.take()
                name = ""
                if self.tok.kind == "STR":
                    name = self.take().value
                self.expect("{")
                # a tool macro ("… Tool", "… Tool Options", "… Tool Selected", "… Action Tool", "… Menu Tool") is a definition that
                # only runs from the tool rail (lumen.core.ops.macro_tools), never when the file is run as a whole: parse it and drop it
                saved, saved_funcs = self._target, len(self.funcs)
                if is_tool_macro_name(name):
                    self._target = []
                try:
                    while not self.at("}"):
                        if self.tok.kind == "EOF":
                            self.fail("missing '}'")
                        self.statement()
                    self.take()
                finally:
                    if self._target is not saved:
                        self._target = saved
                        del self.funcs[saved_funcs:]
                return
            if v == "else":
                self.fail("'else' without 'if'")
        # assignment or expression statement
        self.simple_statement()
        self.end_statement()

    def simple_statement(self) -> None:
        t = self.tok
        # output-argument calls: getDimensions(w, h, c, z, t);
        if t.kind == "ID":
            name = t.value
            if self.peek().kind == "OP" and self.peek().value == "." and self.peek(2).kind == "ID":
                name = f"{t.value}.{self.peek(2).value}"
                offset = 3
            else:
                offset = 1
            if name in OUT_FUNCS and self.peek(offset).kind == "OP" and self.peek(offset).value == "(":
                for _ in range(offset):
                    self.take()
                self.take()  # (
                args: list[str] = []
                arg_is_name: list[bool] = []
                while not self.at(")"):
                    if self.tok.kind == "ID" and self.peek().kind == "OP" and self.peek().value in (",", ")"):
                        args.append(self.pyname(self.take().value)); arg_is_name.append(True)
                    else:
                        args.append(self.expression()); arg_is_name.append(False)
                    if self.at(","):
                        self.take()
                self.take()  # )
                lead, nout = OUT_FUNCS[name]
                inputs = args[:lead]
                outs = args[lead:lead + nout]
                trailing = args[lead + nout:]
                if not outs:
                    self.emit(f"{name}({', '.join(inputs + trailing)})")
                    return
                self.emit(f"{', '.join(outs)}{',' if len(outs) == 1 else ''} = _outs({name}({', '.join(inputs + trailing)}), {len(outs)})")
                return
        expr = self.expression(allow_assign=True)
        self.emit(expr)

    def if_statement(self) -> None:
        self.take()
        self.expect("(")
        cond = self.expression()
        self.expect(")")
        self.emit(f"if {cond}:")
        self.block_or_statement()
        while self.at_id("else"):
            self.take()
            if self.at_id("if"):
                self.take()
                self.expect("(")
                cond = self.expression()
                self.expect(")")
                self.emit(f"elif {cond}:")
                self.block_or_statement()
            else:
                self.emit("else:")
                self.block_or_statement()
                break

    def while_statement(self) -> None:
        self.take()
        self.expect("(")
        cond = self.expression()
        self.expect(")")
        self.emit(f"while {cond}:")
        self.loop_steps.append(None)
        self.block_or_statement()
        self.loop_steps.pop()

    def do_statement(self) -> None:
        # `do { body } while (cond);` → `_doN = True; while _doN or (cond): _doN = False; body` so that a bare
        # Python `continue` re-tests the condition (C/Java semantics) instead of looping forever.
        self.take()
        flag = f"_do{self._do_n}"
        self._do_n += 1
        self.emit(f"{flag} = True")
        hdr = len(self._target)
        self.emit("while True:")  # placeholder, patched below once the condition is known
        self.loop_steps.append(None)
        self.indent += 1
        self.emit(f"{flag} = False")
        if self.at("{"):
            self.take()
            while not self.at("}"):
                if self.tok.kind == "EOF":
                    self.fail("missing '}'")
                self.statement()
            self.take()
        else:
            self.statement()
        self.indent -= 1
        self.loop_steps.pop()
        if not self.at_id("while"):
            self.fail("expected 'while' after do-block")
        self.take()
        self.expect("(")
        cond = self.expression()
        self.expect(")")
        self._target[hdr] = "    " * self.indent + f"while {flag} or ({cond}):"
        self.end_statement()

    def for_statement(self) -> None:
        self.take()
        self.expect("(")
        # init
        init = ""
        if not self.at(";"):
            if self.at_id("var"):
                self.take()
            init = self.expression(allow_assign=True)
        self.expect(";")
        cond = "True" if self.at(";") else self.expression()
        self.expect(";")
        step = "" if self.at(")") else self.expression(allow_assign=True)
        self.expect(")")
        # Always the C lowering (init; while cond: body; step): the loop variable keeps its value after the loop
        # (and is bound even for zero iterations), body assignments to it are honoured and the condition is
        # re-evaluated every pass – exactly ImageJ's Interpreter.doFor.
        if init:
            self.emit(init)
        self.emit(f"while {cond}:")
        self.loop_steps.append(step or None)
        self.block_or_statement()
        if step:
            self.indent += 1
            self.emit(step)
            self.indent -= 1
        self.loop_steps.pop()

    def var_statement(self) -> None:
        self.take()
        while True:
            if self.tok.kind != "ID":
                self.fail("expected a variable name after 'var'")
            name = self.pyname(self.take().value)
            if not self.in_function and name not in self.var_names:
                self.var_names.append(name)
            if self.at("="):
                self.take()
                self.emit(f"{name} = {self.expression()}")
            else:
                self.emit(f"{name} = 0")
            if self.at(","):
                self.take()
                continue
            break
        self.end_statement()

    def function_statement(self) -> None:
        self.take()
        if self.tok.kind != "ID":
            self.fail("expected a function name")
        name = self.pyname(self.take().value)
        self.expect("(")
        params = []
        while not self.at(")"):
            if self.tok.kind != "ID":
                self.fail("bad parameter list")
            params.append(self.pyname(self.take().value))
            if self.at(","):
                self.take()
        self.take()
        saved_target, saved_indent, saved_in = self._target, self.indent, self.in_function
        self._target = self.funcs
        self.indent = 0
        self.in_function = True
        self.func_params = params
        self.emit(f"def {name}({', '.join(params)}):")
        self.indent = 1
        # declare the file-level 'var' globals (they may be declared later in the file: collect them first)
        all_vars = [v for v in self._scan_vars() if v not in params]
        if all_vars:
            self.emit("global " + ", ".join(all_vars))
        self.indent = 0
        self.block_or_statement()
        self._target, self.indent, self.in_function = saved_target, saved_indent, saved_in

    def _scan_vars(self) -> list[str]:
        names: list[str] = []
        depth = 0
        in_func = 0
        for k, t in enumerate(self.toks):
            if t.kind == "OP" and t.value == "{":
                depth += 1
            elif t.kind == "OP" and t.value == "}":
                depth -= 1
                if depth == in_func - 1 and in_func:
                    in_func = 0
            elif t.kind == "ID" and t.value == "function":
                in_func = depth + 1
            elif t.kind == "ID" and t.value == "var" and not in_func:
                j = k + 1
                while j < len(self.toks) and self.toks[j].kind == "ID":
                    names.append(self.pyname(self.toks[j].value))
                    # skip "= expr" up to ',' or ';'
                    j += 1
                    d2 = 0
                    while j < len(self.toks) and not (d2 == 0 and self.toks[j].kind == "OP" and self.toks[j].value in (",", ";")) and self.toks[j].kind != "EOF":
                        if self.toks[j].kind == "OP" and self.toks[j].value in ("(", "["):
                            d2 += 1
                        elif self.toks[j].kind == "OP" and self.toks[j].value in (")", "]"):
                            d2 -= 1
                        j += 1
                    if j < len(self.toks) and self.toks[j].kind == "OP" and self.toks[j].value == ",":
                        j += 1
                        continue
                    break
        return list(dict.fromkeys(names))

    # ---- expressions (precedence climbing, emitted as Python source)
    @staticmethod
    def pyname(name: str) -> str:
        if name in _KEYWORDS_PY:
            return name + "_"
        return name

    def expression(self, allow_assign: bool = False) -> str:
        if allow_assign:
            # assignment: target (= | += | -= | *= | /=) expr   or   target++ / target--
            save = self.i
            target = self.unary(as_target=True)
            if self.tok.kind == "OP" and self.tok.value in ("=", "+=", "-=", "*=", "/="):
                op_ = self.take().value
                rhs = self.expression()
                if op_ == "=":
                    return f"{target} = {rhs}"
                if op_ == "+=":
                    return f"{target} = _add({target}, ({rhs}))"
                if op_ == "-=":
                    return f"{target} = {target} - ({rhs})"
                if op_ == "*=":
                    return f"{target} = {target} * ({rhs})"
                return f"{target} = _div({target}, ({rhs}))"
            if self.tok.kind == "OP" and self.tok.value in ("++", "--"):
                op_ = self.take().value
                return f"{target} = _add({target}, {1 if op_ == '++' else -1})"
            self.i = save
        return self.ternary()

    def ternary(self) -> str:
        cond = self.binary(0)
        if self.at("?"):
            self.take()
            a = self.ternary()
            self.expect(":")
            b = self.ternary()
            return f"(({a}) if ({cond}) else ({b}))"
        return cond

    # ImageJ's precedence (ij.macro.Interpreter.getTerm), not Java's: the bitwise operators & | ^ << >> bind as tightly as
    # * / %, so the tool-macro idiom `while (flags&16!=0)` reads (flags&16) != 0; comparisons share one level
    _PREC = [("||",), ("&&",), ("==", "!=", "<", ">", "<=", ">="), ("+", "-"), ("*", "/", "%", "&", "|", "^", "<<", ">>")]

    def binary(self, level: int) -> str:
        if level >= len(self._PREC):
            return self.unary()
        left = self.binary(level + 1)
        while self.tok.kind == "OP" and self.tok.value in self._PREC[level]:
            op_ = self.take().value
            right = self.binary(level + 1)
            if op_ == "||":
                left = f"({left} or {right})"
            elif op_ == "&&":
                left = f"({left} and {right})"
            elif op_ == "+":
                left = f"_add({left}, {right})"
            elif op_ == "/":
                left = f"_div({left}, {right})"
            elif op_ == "%":
                left = f"_mod({left}, {right})"
            elif op_ == "&":
                left = f"_band({left}, {right})"
            elif op_ == "|":
                left = f"_bor({left}, {right})"
            elif op_ == "^":
                left = f"_bxor({left}, {right})"
            elif op_ == "<<":
                left = f"_shl({left}, {right})"
            elif op_ == ">>":
                left = f"_shr({left}, {right})"
            else:
                left = f"({left} {op_} {right})"
        return left

    def unary(self, as_target: bool = False) -> str:
        if self.tok.kind == "OP":
            v = self.tok.value
            if v == "!":
                self.take(); return f"(not {self.unary()})"
            if v == "-":
                self.take(); return f"(-{self.unary()})"
            if v == "+":
                self.take(); return self.unary()
            if v == "~":
                self.take(); return f"_bnot({self.unary()})"
        return self.postfix(as_target)

    def postfix(self, as_target: bool = False) -> str:
        expr = self.primary()
        while True:
            if self.at("("):
                self.take()
                args = []
                while not self.at(")"):
                    if self.tok.kind == "EOF":
                        self.fail("missing ')'")
                    args.append(self.expression())
                    if self.at(","):
                        self.take()
                self.take()
                expr = f"{expr}({', '.join(args)})"
            elif self.at("["):
                self.take()
                idx = self.expression()
                self.expect("]")
                expr = f"{expr}[_idx({idx})]"
            elif self.at(".") and self.peek().kind == "ID":
                self.take()
                attr = self.take().value
                if attr == "length" and not self.at("("):
                    expr = f"_len({expr})"
                elif expr in _NAMESPACES and not self.at("(") and not as_target:
                    expr = f"_call0({expr}.{self.pyname(attr)})"  # List.setMeasurements; / List.clear; / x = Overlay.size;
                else:
                    expr = f"{expr}.{self.pyname(attr)}"
            else:
                break
        if not as_target and self.tok.kind == "OP" and self.tok.value in ("++", "--"):
            self.fail("postfix ++/-- inside an expression")
        return expr

    def primary(self) -> str:
        t = self.tok
        if t.kind == "NUM":
            self.take()
            v = t.value
            if v.lower().startswith("0x"):
                return str(int(v, 16))
            if re.fullmatch(r"\d+", v):
                return str(int(v, 10))  # ImageJ's tokenizer reads 007 / 010 as decimal 7 / 10; Python 3 rejects leading zeros in int literals
            return "0" + v if v.startswith(".") else v  # float literals: Python accepts 007.5, 00.5, 007e1 (and 1e400 must stay a literal → inf)
        if t.kind == "STR":
            self.take()
            return repr(t.value)
        if t.kind == "ID":
            self.take()
            v = t.value
            if v == "true":
                return "True"
            if v == "false":
                return "False"
            if v in _BARE_CALLS and not self.at("("):
                return f"{v}()"
            return self.pyname(v)
        if self.at("("):
            self.take()
            e = self.expression()
            self.expect(")")
            return f"({e})"
        self.fail()
        return ""  # pragma: no cover


def transpile(text: str) -> str:
    """ImageJ-1 macro source → Python source (executed by IJ.macro)."""
    return _Transpiler(text).run()


# the IJ "log" macro function is the natural logarithm; IJ.log(...) (the object method) prints.
def _ij_log_dispatch(self: IJ, *args: Any) -> Any:
    if len(args) == 1 and isinstance(args[0], (int, float)) and not isinstance(args[0], bool):
        x = args[0]
        return math.log(x) if x > 0 else (float("-inf") if x == 0 else float("nan"))
    self.print(*args)
    return None


IJ.log = _ij_log_dispatch  # type: ignore[assignment]
# Edit ▸ Options ▸ Conversions is registered last so that it wins over the interactive-dialog no-op list that also names it
cmd("Conversions...", lumen="(conversion options)", notes="Edit ▸ Options ▸ Conversions: 'scale' = scale when converting (ImageConverter.setDoScaling), "
                                                        "'weighted' = weighted RGB conversions (0.299/0.587/0.114); both remembered for 8-bit / 16-bit / 32-bit")(_conversions)
MACRO_FUNCTION_NAMES[:] = list(dict.fromkeys(MACRO_FUNCTION_NAMES))
