"""Recorder → ImageJ-1 macro language (E7-14: "the recorder can't emit ImageJ macro language").

The recorder writes Python (``lumen.run("filters.gaussian", sigma=2.0)``, ``lumen.select("t")``, ``lumen.display(...)``).
This module turns those lines into an ``.ijm`` macro that Fiji – and Lumen's own macro layer – can run:

* ``simple()`` registrations of :mod:`lumen.core.ij_compat` are inverted automatically (the option readers carry their
  key, see ``SIMPLE_MAPPINGS``), so every ``run("Median...", "radius=2")``-style command converts both ways;
* the hand-written handlers that matter in a recording (Z Project, Analyze Particles, thresholds, Duplicate, New,
  type conversions, LUTs, substacks, properties, the ROI Manager, the Image Calculator …) have explicit reverse rules;
* a command whose parameters are all at their defaults becomes a bare ``run("Name")``.

Every input line is accounted for: a line that has no ImageJ equivalent (bundled samples, batch commands, a parameter
that ImageJ's dialog does not have) stays in the script as ``// not converted (reason): <line>`` and is listed in
``skipped``; a line that converts with a caveat gets a trailing ``// note`` and the note is listed in ``notes`` *and*
written as ``// note: …`` lines at the top of the script (after the session marker).  Undo / redo lines are resolved
before anything is converted (:func:`lumen.core.recorder_lines.resolve_undo`): the steps the user undid are left out
together with the ``lumen.undo`` / ``lumen.redo`` lines (``undone`` / ``removed`` in the result) instead of emitting
``run("Undo")``, which is single-level in ImageJ (E8-3).  Nothing is converted silently into something that would run
differently.
"""
from __future__ import annotations

import ast
import json
import os
import re
from typing import Any, Callable

from . import ij_compat
from .ij_compat import COMMANDS, SIMPLE_MAPPINGS, const
from .recorder_lines import resolve_undo, undone_lines
from .registry import REGISTRY

_ARROW = re.compile(r"^(.*?)\s+#\s*→\s*(.*)$")


class _Skip(Exception):
    """Raised by a rule when the line cannot be expressed in ImageJ macro language."""


# ----------------------------------------------------------------------------------------------- value formatting
def _num(v: Any) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if v != v or v in (float("inf"), float("-inf")):
            return "Infinity" if v > 0 else "-Infinity"
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(round(v, 6))
    return str(v)


def _opt(key: str, v: Any) -> str:
    """One ``key=value`` of an ImageJ option string (values with spaces, '=' or brackets are bracketed, like the
    Recorder does)."""
    if isinstance(v, bool):
        return key if v else ""
    if isinstance(v, (int, float)):
        return f"{key}={_num(v)}"
    s = str(v)
    if s == "" or any(ch in s for ch in " =[]"):
        s = "[" + s.replace("]", ")") + "]"
    return f"{key}={s}"


def _opts(parts: list[str]) -> str:
    return " ".join(p for p in parts if p)


def _s(v: Any) -> str:
    """A macro string literal."""
    return json.dumps(str(v), ensure_ascii=False)


def _run(name: str, options: list[str] | None = None) -> str:
    o = _opts(options or [])
    return f'run({_s(name)}, {_s(o)});' if o else f'run({_s(name)});'


# ----------------------------------------------------------------------------------------------- parsing
def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception as e:  # noqa: BLE001
        raise _Skip(f"argument is not a literal ({ast.dump(node)[:40]})") from e


def _parse(line: str) -> tuple[str, list[Any], dict[str, Any]] | None:
    """``lumen.<method>(args, kw)`` → (method, args, kwargs); None for anything else."""
    try:
        tree = ast.parse(line.strip(), mode="eval")
    except SyntaxError:
        return None
    call = tree.body
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute) or not isinstance(call.func.value, ast.Name) or call.func.value.id != "lumen":
        return None
    args = [_literal(a) for a in call.args]
    kw = {k.arg: _literal(k.value) for k in call.keywords if k.arg}
    return call.func.attr, args, kw


# ----------------------------------------------------------------------------------------------- ROIs and positions
def _xy(points: list[Any]) -> tuple[str, str]:
    xs = ", ".join(_num(float(p[0])) for p in points)
    ys = ", ".join(_num(float(p[1])) for p in points)
    return xs, ys


def roi_lines(roi: dict[str, Any] | None) -> list[str]:
    """The ``makeRectangle`` / ``makeOval`` / ``makePolygon`` / ``makeLine`` / ``makeSelection`` call for a ROI dict."""
    if not roi:
        return []
    t = roi.get("type")
    if t in ("rect", "oval"):
        fn = "makeRectangle" if t == "rect" else "makeOval"
        return [f"{fn}({_num(float(roi['x']))}, {_num(float(roi['y']))}, {_num(float(roi['w']))}, {_num(float(roi['h']))});"]
    if t == "polygon":
        pts = roi.get("points") or []
        return [f"makePolygon({', '.join(_num(float(c)) for p in pts for c in p[:2])});"]
    if t in ("freehand", "polyline", "point"):
        xs, ys = _xy(roi.get("points") or [])
        return [f"makeSelection({_s(t)}, newArray({xs}), newArray({ys}));"]
    if t == "line":
        w = roi.get("width")
        tail = f", {_num(float(w))}" if w and float(w) > 1 else ""
        return [f"makeLine({_num(float(roi['x1']))}, {_num(float(roi['y1']))}, {_num(float(roi['x2']))}, {_num(float(roi['y2']))}{tail});"]
    raise _Skip(f"a {t or 'unknown'} selection has no macro constructor")


# ----------------------------------------------------------------------------------------------- the converter
class _Ctx:
    def __init__(self, titles: dict[str, str] | None) -> None:
        self.current: str | None = None  # the title the macro's active image has, as far as the lines say
        self.titles = titles or {}
        self.notes: list[str] = []

    def select(self, title: str | None) -> list[str]:
        if not title or title == self.current:
            return []
        self.current = title
        return [f"selectWindow({_s(title)});"]


def _defaults(op_id: str) -> dict[str, Any]:
    spec = REGISTRY.ops.get(op_id)
    return {p.name: p.default for p in spec.params} if spec else {}


def _ij_name(names: tuple[str, ...]) -> str:
    """The name Fiji's Recorder would write: the one with the dialog ellipsis when there is one."""
    for n in names:
        if n.endswith("..."):
            return n
    return names[0]


def _names_for(op_id: str) -> list[tuple[str, ...]]:
    """ImageJ command names registered (by hand) for an op id, grouped per handler."""
    groups: dict[int, list[str]] = {}
    for name, (fn, lumen, _notes) in COMMANDS.items():
        if lumen == op_id:
            groups.setdefault(id(fn), []).append(name)
    return [tuple(g) for g in groups.values()]


def _simple(op_id: str, kw: dict[str, Any], scope: str, defaults: dict[str, Any]) -> list[str] | None:
    """Invert a ``simple()`` registration; None when the op has none."""
    entries = SIMPLE_MAPPINGS.get(op_id)
    if not entries:
        return None
    reasons: list[str] = []
    for names, params, stack_flag in entries:
        opts: list[str] = []
        ok = True
        for param, src in params.items():
            if isinstance(src, const):
                if param in kw and kw[param] != src.value:
                    ok = False  # this ImageJ name stands for another constant (e.g. Median 3D vs Mean 3D)
                    break
                continue
            if param not in kw:
                continue
            v = kw[param]
            key = getattr(src, "key", None)
            if key is None:
                # a hand-written reader: convertible only when ImageJ's dialog defaults (an empty option string)
                # make the reader produce exactly the recorded value, so that leaving the option out is faithful
                try:
                    implied = src(ij_compat.Options())
                except Exception:  # noqa: BLE001
                    implied = object()
                if implied is None:
                    implied = defaults.get(param)
                if v == implied:
                    continue
                reasons.append(f"{param}={v!r} has no direct option in {_ij_name(names)}")
                ok = False
                break
            if getattr(src, "kind", "") == "flag":
                if v:
                    opts.append(key)
            else:
                opts.append(_opt(key, v))
        if not ok:
            continue
        for param, v in kw.items():
            if param not in params and v != defaults.get(param):
                reasons.append(f"{param}={v!r} is not an option of {_ij_name(names)}")
                ok = False
                break
        if not ok:
            continue
        spec = REGISTRY.ops.get(op_id)
        if stack_flag and spec is not None and spec.kind == "plane" and scope == "stack":
            opts.append("stack")
        return [_run(_ij_name(names), opts)]
    raise _Skip("; ".join(reasons) if reasons else "no matching ImageJ command variant")


# ---- hand-written reverse rules: op id → f(kw, ctx) -> macro lines (raise _Skip when impossible)
_RULES: dict[str, Callable[[dict[str, Any], _Ctx], list[str]]] = {}


def _rule(*ids: str):
    def deco(fn):
        for i in ids:
            _RULES[i] = fn
        return fn
    return deco


@_rule("stacks.z_project")
def _r_zproject(kw, ctx):
    if kw.get("axis", "Z") == "T":
        raise _Skip("projection along T: ImageJ's Z Project projects frames only when the stack has a single slice")
    o = [_opt("projection", kw.get("method", "Max Intensity"))]
    if kw.get("start", 1) not in (1, None):
        o.append(_opt("start", kw["start"]))
    if kw.get("stop", 0):
        o.append(_opt("stop", kw["stop"]))
    o.append("all")  # Lumen projects every frame of a time series; ImageJ only the current one without 'all'
    return [_run("Z Project...", o)]


@_rule("analyze.particles")
def _r_particles(kw, ctx):
    if kw.get("intensity_image"):
        raise _Skip("'Measure intensities on' names an image id that the macro cannot resolve (use redirect= in ImageJ)")
    lo, hi = float(kw.get("min_size", 0) or 0), float(kw.get("max_size", 0) or 0)
    o = [f"size={_num(lo)}-{_num(hi) if hi > 0 else 'Infinity'}"]
    if kw.get("pixel_units"):
        o.append("pixel")
    o.append(f"circularity={_num(float(kw.get('min_circularity', 0.0)))}-{_num(float(kw.get('max_circularity', 1.0)))}")
    o.append(_opt("show", kw.get("show", "Nothing")))
    for param, key in (("display_results", "display"), ("clear_results", "clear"), ("summarize", "summarize"), ("add_to_manager", "add"),
                       ("exclude_on_edges", "exclude"), ("include_holes", "include"), ("all_planes", "stack"), ("record_starts", "record"),
                       ("composite_rois", "composite"), ("in_situ_show", "in_situ")):
        if kw.get(param):
            o.append(key)
    out = [_run("Analyze Particles...", o)]
    if kw.get("auto_threshold"):
        out[-1] += " // Lumen auto-thresholded the image first; ImageJ needs setAutoThreshold / setThreshold before this line"
        ctx.notes.append("Analyze Particles ran with 'Auto-threshold if needed'; set a threshold before it in ImageJ")
    return out


def _method(kw: dict[str, Any]) -> str:
    m = str(kw.get("method", "Default"))
    return m + (" dark" if kw.get("dark_background", True) else "")


@_rule("threshold.compute")
def _r_thr_compute(kw, ctx):
    return [f"setAutoThreshold({_s(_method(kw))});"]


def _black_background(ctx: _Ctx) -> list[str]:
    # Lumen masks are always foreground = 255 with a normal LUT; ImageJ needs 'Black background' for the same bytes
    if getattr(ctx, "black_bg", False):
        return []
    ctx.black_bg = True
    return ['setOption("BlackBackground", true);']


@_rule("threshold.apply")
def _r_thr_apply(kw, ctx):
    method = str(kw.get("method", "Current"))
    o: list[str] = []
    if kw.get("per_plane") or kw.get("only_current_plane"):
        o += [_opt("method", "Default" if method == "Current" else method), _opt("background", "Dark" if kw.get("dark_background", True) else "Light")]
        if kw.get("per_plane"):
            o.append("calculate")
        if kw.get("only_current_plane"):
            o.append("only")
        return _black_background(ctx) + [_run("Convert to Mask", o)]
    out = _black_background(ctx) + ([] if method == "Current" else [f"setAutoThreshold({_s(_method(kw))});"])
    return out + [_run("Convert to Mask")]


@_rule("analyze.measure")
def _r_measure(kw, ctx):
    return [_run("Measure")]


@_rule("analyze.clear_results")
def _r_clear_results(kw, ctx):
    return [_run("Clear Results")]


@_rule("rois.clear")
def _r_rois_clear(kw, ctx):
    return ['roiManager("reset");']


@_rule("rois.delete")
def _r_rois_delete(kw, ctx):
    idx: list[int] = []
    for part in str(kw.get("indices", "")).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            idx += list(range(int(a), int(b) + 1))
        else:
            idx.append(int(part))
    if not idx:
        raise _Skip("no ROI indices")
    arr = ", ".join(str(i) for i in sorted(set(idx)))
    return [f'roiManager("select", newArray({arr}));', 'roiManager("delete");']


@_rule("rois.restore_cleared", "results.restore_cleared")
def _r_restore(kw, ctx):
    raise _Skip("ImageJ cannot restore a cleared ROI Manager or Results table")


@_rule("selection.none")
def _r_sel_none(kw, ctx):
    return [_run("Select None")]


@_rule("selection.all")
def _r_sel_all(kw, ctx):
    return [_run("Select All")]


@_rule("file.close_all")
def _r_close_all(kw, ctx):
    ctx.current = None
    return ['close("*");']


@_rule("file.close")
def _r_close(kw, ctx):
    ctx.current = None
    return ["close();"]


_FILLS = {"Black": "black", "White": "white", "Ramp": "ramp", "Noise": "random"}


@_rule("file.new")
def _r_new(kw, ctx):
    fill = str(kw.get("fill", "Black"))
    if fill not in _FILLS:
        raise _Skip(f"newImage has no '{fill}' fill (black, white, ramp, random only)")
    typ = str(kw.get("type", "8-bit"))
    kind = f"{'RGB' if typ == 'RGB' else typ} {_FILLS[fill]}"
    c, z, t = int(kw.get("channels", 1)), int(kw.get("slices", 1)), int(kw.get("frames", 1))
    title = str(kw.get("title", "Untitled"))
    if c > 1 or t > 1:
        line = f"newImage({_s(title)}, {_s(kind)}, {int(kw.get('width', 512))}, {int(kw.get('height', 512))}, {c}, {z}, {t});"
    else:
        line = f"newImage({_s(title)}, {_s(kind)}, {int(kw.get('width', 512))}, {int(kw.get('height', 512))}, {z});"
    return [line]


@_rule("file.open_imagej_sample")
def _r_ij_sample(kw, ctx):
    return [_run(str(kw.get("name", "Blobs (25K)")))]


@_rule("file.sample")
def _r_sample(kw, ctx):
    raise _Skip(f"the bundled sample '{kw.get('name')}' is not an ImageJ sample")


@_rule("image.duplicate")
def _r_duplicate(kw, ctx):
    what = str(kw.get("what", "Whole stack"))
    if what not in ("Whole stack", "Current plane"):
        raise _Skip(f"Duplicate '{what}' needs the channel / slice number ImageJ would record (channels= / slices=)")
    o = []
    if kw.get("title"):
        o.append(_opt("title", kw["title"]))
    if what == "Whole stack":
        o.append("duplicate")
    return [_run("Duplicate...", o)]


@_rule("image.crop")
def _r_crop(kw, ctx):
    return [_run("Crop")]


def _conversion(kw: dict[str, Any], name: str) -> list[str]:
    out: list[str] = []
    scale, weighted = kw.get("scale", True), kw.get("weighted", False)
    if scale is False or weighted:
        out.append(_run("Conversions...", ["scale" if scale is not False else "", "weighted" if weighted else ""]))
    out.append(_run(name))
    if scale is False or weighted:
        out.append(_run("Conversions...", ["scale"]))  # back to ImageJ's defaults
    return out


@_rule("type.8bit")
def _r_8bit(kw, ctx):
    return _conversion(kw, "8-bit")


@_rule("type.16bit")
def _r_16bit(kw, ctx):
    return _conversion(kw, "16-bit")


@_rule("type.32bit")
def _r_32bit(kw, ctx):
    return _conversion(kw, "32-bit")


@_rule("type.rgb")
def _r_rgb(kw, ctx):
    return [_run("RGB Color")]


@_rule("color.lut")
def _r_lut(kw, ctx):
    out = [_run(str(kw.get("lut", "Grays")))]
    if kw.get("all_channels"):
        out[-1] += " // Lumen applied the LUT to every channel; ImageJ sets the current channel only"
        ctx.notes.append("Set LUT on all channels: ImageJ's run(\"<LUT>\") changes the current channel only")
    return out


@_rule("type.apply_lut", "adjust.brightness_contrast_apply")
def _r_apply_lut(kw, ctx):
    out = [_run("Apply LUT")]
    if kw.get("all_channels"):
        out[-1] += " // every channel in Lumen; ImageJ applies the current channel's range"
        ctx.notes.append("Apply LUT on all channels: ImageJ applies the current channel only")
    return out


@_rule("stacks.substack")
def _r_substack(kw, ctx):
    key = {"Z": "slices", "T": "frames", "C": "channels"}.get(str(kw.get("axis", "Z")), "slices")
    o = [_opt(key, kw.get("slices", "1-10"))]
    if kw.get("delete_from_original"):
        o.append("delete")
    return [_run("Make Substack...", o)]


@_rule("image.properties")
def _r_properties(kw, ctx):
    o = [_opt("unit", kw.get("unit", "pixel")), _opt("pixel_width", float(kw.get("pixel_width", 1.0))), _opt("pixel_height", float(kw.get("pixel_height", 1.0))),
         _opt("voxel_depth", float(kw.get("voxel_depth", 1.0)))]
    if kw.get("frame_interval"):
        o.append(_opt("frame", f"{_num(float(kw['frame_interval']))} {kw.get('time_unit', 'sec')}"))
    return [_run("Properties...", o)]


@_rule("process.calculator")
def _r_calculator(kw, ctx):
    a, b = ctx.titles.get(str(kw.get("image1"))), ctx.titles.get(str(kw.get("image2")))
    if not a or not b:
        raise _Skip("Image Calculator operands are image ids of a session that is no longer open")
    o = [_opt("image1", a), _opt("operation", kw.get("operation", "Subtract")), _opt("image2", b)]
    if kw.get("create_new", True):
        o.append("create")
    if kw.get("float_result"):
        o.append("32-bit")
    return [_run("Image Calculator...", o)]


@_rule("script.run_ij_macro")
def _r_macro(kw, ctx):
    code = str(kw.get("macro", "")).strip("\n")
    return code.splitlines() if code else []


def _convert_run(args: list[Any], kw: dict[str, Any], ctx: _Ctx) -> list[str]:
    if not args or not isinstance(args[0], str):
        raise _Skip("lumen.run without a command id")
    op_id = args[0]
    kw = dict(kw)
    if op_id.startswith("batch."):
        raise _Skip("batch commands run Lumen scripts; write a loop over getFileList() in ImageJ")
    out: list[str] = []
    image = kw.pop("image", None)
    if image is not None:
        out += ctx.select(ctx.titles.get(str(image), str(image)))
    channel = kw.pop("channel", None)
    if channel:
        out.append(f"Stack.setChannel({int(channel)});")
    scope = kw.pop("scope", "stack")
    roi = kw.pop("roi", None)
    out += roi_lines(roi)
    defaults = _defaults(op_id)
    if scope == "channel":
        ctx.notes.append(f"{op_id}: 'current channel on every plane' has no ImageJ scope; converted as the current plane")
        scope = "slice"
    rule = _RULES.get(op_id)
    if rule is not None:
        body = rule(kw, ctx)
    else:
        body = _simple(op_id, kw, scope, defaults)
        if body is None:
            names = _names_for(op_id)
            if not names:
                raise _Skip("no ImageJ command maps to this Lumen command")
            extra = {k: v for k, v in kw.items() if v != defaults.get(k)}
            if extra:
                raise _Skip(f"{_ij_name(names[0])}: parameters {', '.join(f'{k}={v!r}' for k, v in extra.items())} have no option-string mapping")
            spec = REGISTRY.ops.get(op_id)
            body = [_run(_ij_name(names[0]), ["stack"] if spec is not None and spec.kind == "plane" and scope == "stack" else [])]
    return out + body


_SAVE_FORMATS = {".tif": "Tiff", ".tiff": "Tiff", ".png": "PNG", ".jpg": "Jpeg", ".jpeg": "Jpeg", ".gif": "Gif", ".bmp": "BMP", ".pgm": "PGM",
                 ".zip": "ZIP", ".txt": "Text", ".avi": "AVI", ".fits": "FITS", ".raw": "Raw Data", ".lut": "LUT", ".csv": "Results"}

_MODES = {"composite": "composite", "single": "color", "grayscale": "grayscale"}


def _convert_display(args: list[Any], kw: dict[str, Any], ctx: _Ctx) -> list[str]:
    out = ctx.select(str(args[0])) if args else []
    kw = dict(kw)
    if "channel" in kw:
        out.append(f"Stack.setChannel({int(kw.pop('channel'))});")
    lo, hi = kw.pop("min", None), kw.pop("max", None)
    if lo is not None and hi is not None:
        out.append(f"setMinAndMax({_num(float(lo))}, {_num(float(hi))});")
    elif lo is not None or hi is not None:
        # only one end was changed: ImageJ's idiom keeps the other end with getMinAndMax (E8-3)
        out.append("getMinAndMax(__lo, __hi);")
        out.append(f"setMinAndMax({_num(float(lo)) if lo is not None else '__lo'}, {_num(float(hi)) if hi is not None else '__hi'});")
    if "lut" in kw:
        out.append(_run(str(kw.pop("lut"))))
    if "inverted" in kw:
        out.append(_run("Invert LUT"))
        kw.pop("inverted")
    if "visible" in kw:
        v = kw.pop("visible")
        ctx.notes.append(f"channel visibility ({'shown' if v else 'hidden'}) needs Stack.setActiveChannels(\"…\") with every channel's state")
        out.append(f'// channel {"shown" if v else "hidden"}: use Stack.setActiveChannels("1101") with one digit per channel')
    if "gamma" in kw:
        kw.pop("gamma")
        raise _Skip("display gamma has no ImageJ equivalent")
    if "mode" in kw:
        out.append(f"Stack.setDisplayMode({_s(_MODES.get(str(kw.pop('mode')), 'composite'))});")
    for k, fn in (("z", "Stack.setSlice"), ("c", "Stack.setChannel"), ("t", "Stack.setFrame")):
        if k in kw:
            out.append(f"{fn}({int(kw.pop(k)) + 1});")
    if kw:
        raise _Skip(f"unknown display settings {sorted(kw)}")
    return out


def convert_line(line: str, ctx: _Ctx) -> tuple[list[str] | None, str | None]:
    """One recorder line → (macro lines, None) or (None, reason)."""
    raw = line.rstrip()
    if not raw.strip():
        return [""], None
    if raw.lstrip().startswith("#"):
        return ["// " + raw.lstrip()[1:].strip()], None
    created: str | None = None
    m = _ARROW.match(raw)
    if m:
        raw, tail = m.group(1), m.group(2)
        try:
            created = json.loads(tail.split(",")[0])
        except ValueError:
            created = None
    try:
        parsed = _parse(raw)
        if parsed is None:
            raise _Skip("not a recorder line")
        method, args, kw = parsed
        if method == "run":
            out = _convert_run(args, kw, ctx)
        elif method == "select":
            out = ctx.select(str(args[0]) if args else None)
        elif method == "close":
            if args:
                out = [f"close({_s(args[0])});"]
                if ctx.current == args[0]:
                    ctx.current = None
            else:
                out = ["close();"]
                ctx.current = None
        elif method == "open":
            out = [f"open({_s(args[0])});"]
        elif method == "save":
            path = str(args[0]) if args else ""
            fmt = _SAVE_FORMATS.get(os.path.splitext(path)[1].lower())
            if fmt is None:
                raise _Skip(f"saveAs has no format for '{os.path.splitext(path)[1] or path}'")
            out = [f"saveAs({_s(fmt)}, {_s(path)});"]
        elif method == "undo":
            # an undo that cancelled a recorded step was resolved away by recorder_to_ijm (both lines left out);
            # one that reaches this point had nothing recorded to cancel, and run("Undo") would undo something else
            raise _Skip("this undo cancels no recorded step (ImageJ's single-level Undo would act on the previous command)")
        elif method == "redo":
            raise _Skip("this redo restores no recorded step (ImageJ has no Redo)")
        elif method == "display":
            out = _convert_display(args, kw, ctx)
        else:
            raise _Skip(f"lumen.{method}() has no macro equivalent")
    except _Skip as e:
        return None, str(e)
    if created:
        ctx.current = created
        if out:
            out[-1] += f" // → {_s(created)}"
    return out, None


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def recorder_to_ijm(lines: list[str], titles: dict[str, str] | None = None) -> dict[str, Any]:
    """Convert recorder lines to an ImageJ macro. Returns ``{"script", "lines", "converted", "skipped": [{"line", "reason"}],
    "notes", "count", "undone", "removed"}``; ``count`` is the number of input command lines (comments excluded),
    ``undone`` the recorded commands a ``lumen.undo`` cancelled and ``removed`` the number of lines left out for it
    (the undone commands plus their undo / redo lines), so ``converted + len(skipped) + removed == count``.  The
    notes are written into the script as ``// note: …`` lines after the session-marker comment."""
    ctx = _Ctx(titles)
    kept, dropped = resolve_undo(lines)
    undone = undone_lines(dropped)
    removed = sum(1 for d in dropped if d["line"].strip() and not d["line"].lstrip().startswith("#"))
    out: list[str] = []
    skipped: list[dict[str, str]] = []
    converted = 0
    count = removed
    for line in kept:
        is_cmd = bool(line.strip()) and not line.lstrip().startswith("#")
        count += is_cmd
        macro, reason = convert_line(line, ctx)
        if macro is None:
            skipped.append({"line": line, "reason": reason or ""})
            out.append(f"// not converted ({reason}): {line.strip()}")
            continue
        converted += is_cmd
        out.extend(macro)
    if undone:
        n = len(undone)
        ctx.notes.append(f"{n} recorded step{'s' if n != 1 else ''} the session undid {'were' if n != 1 else 'was'} left out together with the undo / redo lines "
                         f"(ImageJ's Undo is single-level, so run(\"Undo\") is never written): " + "; ".join(ln.strip() for ln in undone))
    notes = _unique(ctx.notes)
    if notes:
        header = [f"// note: {n}" for n in notes]
        at = 1 if out and out[0].startswith("// Lumen recorder") else 0
        out[at:at] = header
    script = "\n".join(out).rstrip("\n") + ("\n" if out else "")
    return {"script": script, "lines": out, "converted": converted, "skipped": skipped, "notes": notes, "count": count,
            "undone": undone, "removed": removed}


__all__ = ["recorder_to_ijm", "convert_line", "roi_lines"]
