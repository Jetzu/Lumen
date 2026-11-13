"""Process ▸ Math and Edit ▸ Invert – ImageJ's ``ImageProcessor.process(op, value)`` arithmetic per pixel type
(see ``_imagej_types``): 8-bit through a 256-entry table with ``(int)`` truncation (Gamma/Log/Exp/Sqrt) and ``Math.round``
(Multiply); 16-bit with Gamma/Log/Exp scaled by the current display range (``getMin()/getMax()``) and truncated;
32-bit in float32 (Gamma = exp(γ·ln v), v ≤ 0 → 0; Log/Exp/Sqrt via double and cast back)."""
from __future__ import annotations

import numpy as np

from ..registry import Param, op
from ._imagej_types import math_plane
from ._util import as_float, back_to

CAT = "Process/Math"


def _val(name="value", default=25.0):
    return Param(name, "float", default, step=1.0)


def _display_range(ctx) -> tuple[float, float]:
    """ImageJ's ``ip.getMin()/getMax()``: the display range of the current channel (applied to every slice of a stack)."""
    img = ctx.image
    if img is None or not img.display.channels:
        return 0.0, 0.0
    ch = img.display.channels[min(img.display.c, len(img.display.channels) - 1)]
    return float(ch.min), float(ch.max)


def _math(plane, ctx, op_name: str, value: float = 0.0):
    lo, hi = _display_range(ctx)
    return math_plane(plane, op_name, value, lo, hi)


def _binary(value) -> int:
    try:
        return int(str(value).strip(), 2)
    except ValueError:
        raise ValueError(f"Binary number required ({value!r})") from None


@op("math.add", "Add", CAT, params=[_val()], description="Add a constant (integer images add the truncated value, like ImageJ).")
def add(plane, ctx, value=25.0):
    return _math(plane, ctx, "add", value)


@op("math.subtract", "Subtract", CAT, params=[_val()], description="Subtract a constant.")
def subtract(plane, ctx, value=25.0):
    return _math(plane, ctx, "add", -value)


@op("math.multiply", "Multiply", CAT, params=[_val("value", 1.25)], description="Multiply by a constant (integer results rounded half up, like ImageJ).")
def multiply(plane, ctx, value=1.25):
    return _math(plane, ctx, "mult", value)


@op("math.divide", "Divide", CAT, params=[_val("value", 1.25)], description="Divide by a constant (ImageJ multiplies by 1/value).")
def divide(plane, ctx, value=1.25):
    if value == 0:
        raise ValueError("Cannot divide by zero")
    return _math(plane, ctx, "mult", 1.0 / value)


@op("math.and", "AND", CAT, params=[Param("value", "string", "11110000", help="Binary constant")], description="Bitwise AND with a binary constant.", supports_rgb=False)
def and_(plane, ctx, value="11110000"):
    if plane.dtype == np.float32:
        raise ValueError("Bitwise operations need an integer image")
    return _math(plane, ctx, "and", _binary(value))


@op("math.or", "OR", CAT, params=[Param("value", "string", "11110000")], description="Bitwise OR.", supports_rgb=False)
def or_(plane, ctx, value="11110000"):
    if plane.dtype == np.float32:
        raise ValueError("Bitwise operations need an integer image")
    return _math(plane, ctx, "or", _binary(value))


@op("math.xor", "XOR", CAT, params=[Param("value", "string", "11110000")], description="Bitwise XOR.", supports_rgb=False)
def xor_(plane, ctx, value="11110000"):
    if plane.dtype == np.float32:
        raise ValueError("Bitwise operations need an integer image")
    return _math(plane, ctx, "xor", _binary(value))


@op("math.min", "Min", CAT, params=[_val()], description="Pixels below the value are set to the value.")
def min_(plane, ctx, value=25.0):
    return _math(plane, ctx, "min", value)


@op("math.max", "Max", CAT, params=[_val("value", 230.0)], description="Pixels above the value are set to the value.")
def max_(plane, ctx, value=230.0):
    return _math(plane, ctx, "max", value)


@op("math.gamma", "Gamma", CAT, params=[Param("value", "float", 0.5, min=0.05, max=5.0, step=0.05)],
    description="ImageJ's Gamma: 8-bit exp(γ·ln(v/255))·255, 16-bit exp(γ·ln((v−min)/(max−min)))·(max−min)+min with the display range, 32-bit exp(γ·ln v) (v ≤ 0 → 0).")
def gamma(plane, ctx, value=0.5):
    return _math(plane, ctx, "gamma", value)


@op("math.set", "Set", CAT, params=[_val()], description="Fill with a constant (within the selection).")
def set_(plane, ctx, value=25.0):
    return _math(plane, ctx, "set", value)


@op("math.log", "Log", CAT, description="Natural logarithm like ImageJ: 8-bit ln(v)·255/ln(255), 16-bit ln(v)·max/ln(max) with the display maximum (truncated), 32-bit ln v.")
def log_(plane, ctx):
    return _math(plane, ctx, "log")


@op("math.exp", "Exp", CAT, description="Exponential like ImageJ: 8-bit exp(v·ln(255)/255), 16-bit exp(v·ln(max)/max) with the display maximum (truncated), 32-bit exp v.")
def exp_(plane, ctx):
    return _math(plane, ctx, "exp")


@op("math.square", "Square", CAT, description="I² (clamped to the type range)")
def square(plane, ctx):
    return _math(plane, ctx, "sqr")


@op("math.sqrt", "Square Root", CAT, description="√I (integer results truncated, like ImageJ)")
def sqrt(plane, ctx):
    return _math(plane, ctx, "sqrt")


@op("math.reciprocal", "Reciprocal", CAT, description="1 / I (32-bit images)")
def reciprocal(plane, ctx):
    f = as_float(plane)
    with np.errstate(divide="ignore"):
        return np.where(f != 0, 1.0 / f, np.nan).astype(np.float32)


@op("math.nan_background", "NaN Background", CAT, description="Set pixels outside the current threshold to NaN (32-bit).")
def nan_background(plane, ctx):
    from ._util import current_threshold
    thr = current_threshold(ctx)
    if thr is None:
        raise ValueError("Set a threshold first")
    f = as_float(plane).copy()
    f[(f < thr[0]) | (f > thr[1])] = np.nan
    return f


@op("math.abs", "Abs", CAT, description="|I| (32-bit images; unsigned integer images are unchanged, like ImageJ)")
def abs_(plane, ctx):
    return _math(plane, ctx, "abs")


@op("math.expression", "Expression", CAT, description="Evaluate a numpy expression per pixel. Variables: v (value), x, y, w, h, np.",
    params=[Param("expression", "string", "v * (1 + 0.5 * np.sin(x / 20))")])
def expression(plane, ctx, expression="v"):
    h, w = plane.shape
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    env = {"v": as_float(plane), "x": x, "y": y, "w": w, "h": h, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "abs": np.abs, "pi": np.pi}
    out = eval(expression, {"__builtins__": {}}, env)
    return back_to(np.broadcast_to(np.asarray(out, np.float32), plane.shape), plane)


@op("edit.invert", "Invert", "Edit", shortcut="Shift+I",
    description="Invert pixel values like ImageJ: 8-bit 255−v; 16-bit max−(v−min) with the plane's own min and max (ShortProcessor.invert); 32-bit max−(v−min) with the display range.")
def invert(plane, ctx):
    if plane.dtype == np.uint8:
        return 255 - plane
    if plane.dtype == np.uint16:
        mn, mx = float(plane.min()), float(plane.max())
        return math_plane(plane, "invert", 0.0, mn, mx)
    lo, hi = _display_range(ctx)
    if ctx.image is None or not np.isfinite([lo, hi]).all():
        f = as_float(plane)
        lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
    return math_plane(plane, "invert", 0.0, lo, hi)
