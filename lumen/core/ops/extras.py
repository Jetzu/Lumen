"""Remaining ImageJ/Fiji core commands: Image ▸ Adjust extras, Analyze extras, Image ▸ Overlay items
(text, arrows, calibration bars, time stamps, burn-in) and Plugins ▸ Scripting ▸ ImageJ macro."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .. import roi as roimod
from ..fitting import CUSTOM, FIT_NAMES, compile_equation, evaluate, fit, parse_numbers, residuals
from ..image import Calibration, LumenImage
from ..luts import LUT_NAMES
from ..measure import DEFAULT_MEASUREMENTS, _pixel_stats, measure_roi
from ..overlay_render import POSITIONS, burn_overlay
from ..registry import OpResult, Param, cast_to, op
from ._util import binary_image, current_threshold

ADJ = "Image/Adjust"
ANA = "Analyze"
OVL = "Image/Overlay"


# ====================================================================== Image ▸ Adjust
COLOR_SPACES = ["HSB", "RGB", "Lab", "YUV"]
_SPACE_LABELS = {"HSB": ("Hue", "Saturation", "Brightness"), "RGB": ("Red", "Green", "Blue"), "Lab": ("L*", "a*", "b*"), "YUV": ("Y", "U", "V")}


def color_components(rgb_hwc: np.ndarray, space: str) -> np.ndarray:
    """(H, W, 3) uint8 RGB → (3, H, W) float32 components scaled to 0–255 (ImageJ Color Threshold convention)."""
    f = rgb_hwc.astype(np.float32)
    if space == "RGB":
        return f.transpose(2, 0, 1)
    if space == "HSB":
        from skimage.color import rgb2hsv
        return (rgb2hsv(rgb_hwc) * 255.0).astype(np.float32).transpose(2, 0, 1)
    if space == "Lab":
        from skimage.color import rgb2lab
        lab = rgb2lab(rgb_hwc)  # D65/2°, same math as ij.process.ColorSpaceConverter
        # ImageJ ColorThresholder.getLab: L1 = (int)(L*2.55); a1/b1 = floor(1.0625*a + 128 + 0.5), clamped to 0..255
        out = np.stack([np.floor(lab[..., 0] * 2.55), np.floor(1.0625 * lab[..., 1] + 128.5), np.floor(1.0625 * lab[..., 2] + 128.5)])
        return np.clip(out, 0, 255).astype(np.float32)
    if space == "YUV":
        r, g, b = f[..., 0], f[..., 1], f[..., 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.28886 * g + 0.436 * b + 128.0
        v = 0.615 * r - 0.51499 * g - 0.10001 * b + 128.0
        return np.clip(np.stack([y, u, v]), 0, 255).astype(np.float32)
    raise ValueError(space)


def _component_params() -> list[Param]:
    ps: list[Param] = [Param("space", "choice", "HSB", choices=COLOR_SPACES, label="Color space")]
    for i in range(1, 4):
        ps += [Param(f"c{i}_min", "int", 0, min=0, max=255, label=f"Component {i} min"), Param(f"c{i}_max", "int", 255, min=0, max=255, label=f"Component {i} max"),
               Param(f"c{i}_filter", "choice", "pass", choices=["pass", "stop"], label=f"Component {i} filter")]
    ps.append(Param("output", "choice", "Mask", choices=["Mask", "Selection", "Overlay"]))
    return ps


@op("adjust.color_threshold", "Color Threshold", ADJ, kind="image", preview=False, roi_aware=False, scope_param=False,
    description="Threshold an RGB image in HSB, RGB, Lab or YUV space with per-component min/max and pass/stop filters (components are scaled to 0–255, like ImageJ).",
    params=_component_params(), keywords=["hue", "saturation", "segment colour", "rgb threshold"])
def color_threshold(ctx, space="HSB", c1_min=0, c1_max=255, c1_filter="pass", c2_min=0, c2_max=255, c2_filter="pass", c3_min=0, c3_max=255, c3_filter="pass", output="Mask"):
    img = ctx.img
    d = img.display
    if not (img.nc == 3 and img.data.dtype == np.uint8):
        raise ValueError("Color Threshold needs an RGB (3 × 8-bit) image")
    rgb = img.data[d.t, d.z].transpose(1, 2, 0)
    comps = color_components(rgb, space)
    inside = np.ones(rgb.shape[:2], bool)
    for k, (lo, hi, filt) in enumerate(((c1_min, c1_max, c1_filter), (c2_min, c2_max, c2_filter), (c3_min, c3_max, c3_filter))):
        c = comps[k]
        if lo <= hi:
            m = (c >= lo) & (c <= hi)
        else:  # wrap-around (hue)
            m = (c >= lo) | (c <= hi)
        if filt == "stop":
            m = ~m
        inside &= m
    if ctx.mask is not None:
        inside &= ctx.mask
    frac = 100.0 * float(inside.mean())
    msg = f"{space} threshold: {frac:.1f}% of pixels selected"
    if output == "Mask":
        return OpResult(new_images=[binary_image(inside, f"Mask of {img.name}", like=img)], message=msg, select_image=img.id)
    r = roimod.from_mask(inside, "Color threshold")
    if r is None:
        raise ValueError("No pixels within the colour ranges")
    if output == "Selection":
        return OpResult(roi=r, message=msg)
    r["color"] = "#ff0000"
    r["stroke_width"] = 1.0
    return OpResult(overlay=[*img.overlay, r], message=msg)


@op("adjust.window_level", "Window/Level", ADJ, kind="analysis", preview=False, roi_aware=False,
    params=[Param("window", "float", 256.0, min=0, step=1, help="Width of the displayed intensity range"), Param("level", "float", 128.0, step=1, help="Centre of the displayed range"),
            Param("all_channels", "bool", False, help="Apply to every channel (RGB images always get the same range on all three components)")],
    description="Set the display range of the current channel (all three components of an RGB image) from window (width) and level (centre); pixel values are unchanged.",
    keywords=["brightness", "contrast", "display range"])
def window_level(ctx, window=256.0, level=128.0, all_channels=False):
    img = ctx.img
    lo, hi = level - window / 2.0, level + window / 2.0
    if hi <= lo:
        hi = lo + 1e-6
    patch = [{"min": lo, "max": hi} if (all_channels or img.rgb or i == img.display.c) else {} for i in range(img.nc)]  # RGB = one range for all components (ImageJ)
    return OpResult(display={"channels": patch}, message=f"Display range {lo:g} – {hi:g}")


@op("adjust.coordinates", "Coordinates (origin)", ADJ, kind="analysis", preview=False, roi_aware=False,
    params=[Param("origin_x", "float", 0.0, step=1, help="In calibrated units"), Param("origin_y", "float", 0.0, step=1), Param("origin_z", "float", 0.0, step=1)],
    description="Set the coordinate origin of the image (stored in the metadata as 'origin').", keywords=["origin", "offset"])
def coordinates(ctx, origin_x=0.0, origin_y=0.0, origin_z=0.0):
    img = ctx.img
    if hasattr(img.calibration, "origin_x"):
        img.calibration.origin_x = origin_x; img.calibration.origin_y = origin_y; img.calibration.origin_z = origin_z  # pragma: no cover
    img.metadata["origin"] = {"x": origin_x, "y": origin_y, "z": origin_z}
    img.metadata["origin_x"], img.metadata["origin_y"], img.metadata["origin_z"] = origin_x, origin_y, origin_z
    return OpResult(message=f"Origin set to ({origin_x:g}, {origin_y:g}, {origin_z:g}) {img.calibration.unit}")


@op("adjust.brightness_contrast_apply", "Apply Display Range to Pixels (B&C Apply)", ADJ, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("all_channels", "bool", False, help="Apply every channel's display range (default: current channel only)")],
    description="DESTRUCTIVE: bake the current display range into the pixel values of the whole stack, like the Apply button of ImageJ's Brightness/Contrast dialog (and ImageJ's 'Apply LUT'). To only change the colour map use Set LUT.",
    keywords=["apply lut", "brightness", "contrast", "bake", "display range"])
def bc_apply(ctx, all_channels=False):
    img = ctx.img
    if img.data.dtype == np.float32:
        raise ValueError("Apply is not available for 32-bit images – convert to 16-bit first")
    full = 255.0 if img.data.dtype == np.uint8 else 65535.0
    out = img.data.copy()
    chans = range(img.nc) if (all_channels or img.rgb) else [img.display.c]
    patch: list[dict[str, Any]] = [{} for _ in range(img.nc)]
    for c in chans:
        ch = img.display.channels[c]
        lo, hi = ch.min, ch.max
        from ._imagej_types import apply_lut_table  # ImageJ's ContrastAdjuster.apply lookup table (bit-exact)
        table = apply_lut_table(img.data.dtype, float(lo), float(hi))
        out[:, :, c] = table[img.data[:, :, c]]
        patch[c] = {"min": 0.0, "max": full}
    return OpResult(data=out, rgb=img.rgb, display={"channels": patch}, message="Display range applied to pixel values")


# ====================================================================== Analyze
CAL_FUNCTIONS = ["None", "Straight Line", "2nd Degree Polynomial", "3rd Degree Polynomial", "4th Degree Polynomial", "Exponential", "Exponential with Offset", "Exponential Recovery",
                 "Exponential (linear regression)", "Power", "Power (linear regression)", "Log", "Log2", "Gaussian", "Gaussian (no offset)", "Error Function", "Rodbard", "Rodbard (NIH Image)",
                 "Inverse Rodbard", "Gamma Variate", "Chapman-Richards", "Uncalibrated OD", CUSTOM]
CURVE_FIT_FUNCTIONS = [*FIT_NAMES, CUSTOM]


def intensity_calibration(img: LumenImage) -> dict[str, Any] | None:
    cal = img.metadata.get("intensity_calibration")
    return cal if isinstance(cal, dict) and cal.get("function") not in (None, "None") else None


def calibrated_values(img: LumenImage, values: Any) -> np.ndarray:
    """Convert raw pixel values through the image's intensity calibration (identity when none)."""
    cal = intensity_calibration(img)
    arr = np.asarray(values, dtype=np.float64)
    if cal is None:
        return arr
    return evaluate(cal["function"], list(cal.get("coefficients", [])), arr, equation=cal.get("equation"))


@op("analyze.calibrate", "Calibrate (intensity)", ANA, kind="analysis", preview=False, roi_aware=False, supports_rgb=False,
    params=[Param("function", "choice", "Straight Line", choices=CAL_FUNCTIONS), Param("pixel_values", "text", "0 128 255", help="Measured pixel values, one per line or space separated"),
            Param("standards", "text", "0 0.5 1.0", help="Known standard values (same order)"), Param("unit", "string", "OD", label="Value unit"),
            Param("custom_equation", "text", "", help="Function = Custom: equation in x with parameters a–f, e.g. y = a + b*exp(c*x)")],
    description="Fit a function mapping pixel values to known standards (density calibration); Measure Calibrated then reports calibrated Mean/Min/Max.", keywords=["density", "OD", "standard curve"])
def calibrate(ctx, function="Straight Line", pixel_values="", standards="", unit="OD", custom_equation=""):
    img = ctx.img
    if function == "None":
        img.metadata.pop("intensity_calibration", None)
        return OpResult(calibration={"value_unit": ""}, message="Intensity calibration removed")
    if img.data.dtype.kind == "f":
        # ImageJ Calibrator: 'Function must be "None" for 32-bit images, but you can change the Unit'
        raise ValueError('Function must be "None" for 32-bit images (their values are not density calibrated), but you can change the value unit')
    if function == "Uncalibrated OD":
        if img.data.dtype != np.uint8:
            # ImageJ: "Uncalibrated OD is not supported on 16-bit images" – the formula is only defined for 8-bit values
            raise ValueError("Uncalibrated OD is only supported on 8-bit images – convert to 8-bit first")
        img.metadata["intensity_calibration"] = {"function": function, "coefficients": [], "unit": "Uncalibrated OD"}
        return OpResult(calibration={"value_unit": "Uncalibrated OD"}, message="Uncalibrated OD: value = log10(255 / (255 − v))")
    x = parse_numbers(pixel_values)
    y = parse_numbers(standards)
    if x.size != y.size or x.size < 2:
        raise ValueError("Pixel values and standards must have the same number of entries (at least 2)")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]  # same rule as fitting.fit(); the plot below must see the same finite pairs
    if x.size < 2:
        raise ValueError("Need at least 2 finite pixel-value / standard pairs")
    equation = str(custom_equation or "").strip()
    if function == CUSTOM:
        if not equation:
            raise ValueError("Custom calibration: enter an equation in x with parameters a–f, e.g. y = a + b*exp(c*x)")
        res = fit(CUSTOM, x, y, equation=equation, restarts=2)
    else:
        equation = ""
        res = fit(function, x, y)
    cal: dict[str, Any] = {"function": function, "coefficients": res["params"], "unit": unit, "r2": res["r2"]}
    if equation:
        cal["equation"] = equation
    img.metadata["intensity_calibration"] = cal
    row: dict[str, Any] = {"Label": img.name, "Function": function if not equation else res["formula"]}
    for n, p in zip(res["names"], res["params"]):
        row[n] = p
    row["R²"] = res["r2"]
    xs = np.linspace(float(x.min()), float(x.max()), 100)
    plot = {"title": f"Calibration of {img.name}: {res['formula']}", "x": xs.tolist(),
            "series": [{"name": "fit", "values": evaluate(function, res["params"], xs, equation=equation or None).tolist()}, {"name": "standards", "values": np.interp(xs, x[np.argsort(x)], y[np.argsort(x)]).tolist()}],
            "x_label": "Pixel value", "y_label": unit, "points": {"x": x.tolist(), "y": y.tolist()}}
    return OpResult(results=[row], plot=plot, calibration={"value_unit": unit}, message=f"{res['formula'] if equation else function}: R² = {res['r2']:.4f}")


@op("analyze.calibrated_values", "Calibrated Values (32-bit)", ANA, kind="image", preview=False, roi_aware=False, scope_param=False, supports_rgb=False,
    description="New 32-bit image whose pixels are the calibrated values (after Analyze ▸ Calibrate) of this image.")
def calibrated_values_op(ctx):
    img = ctx.img
    cal = intensity_calibration(img)
    if cal is None:
        raise ValueError("No intensity calibration – run Calibrate first")
    out = calibrated_values(img, img.data).astype(np.float32)
    new = img.copy(name=f"Calibrated {img.name}", data=out)
    new.metadata.pop("intensity_calibration", None)  # the pixels now ARE the calibrated values (ImageJ drops the function on the 32-bit result)
    new.calibration.value_unit = cal.get("unit", "") or new.calibration.value_unit
    new.reset_display(keep_luts=True)
    return OpResult(new_images=[new], message=f"Values converted with {cal['function']} ({cal.get('unit', '')})")


@op("analyze.measure_calibrated", "Measure (calibrated values)", ANA, kind="analysis", preview=False, roi_aware=True,
    description="Measure like Measure, but with Mean / StdDev / Min / Max / Median / Mode / IntDen (and Skew, Kurt, XM, YM) computed from the calibrated pixel values "
                "(Analyze ▸ Calibrate); RawIntDen stays raw, like ImageJ.")
def measure_calibrated(ctx):
    s = ctx.session
    img = ctx.img
    thr = current_threshold(ctx) if s.measure_options.get("limit_to_threshold") else None
    rows = measure_roi(img, ctx.roi, s.measurements, limit_to_threshold=thr, options=s.measure_options)
    cal = intensity_calibration(img)
    unit = (cal.get("unit") if cal else img.calibration.value_unit) or "gray"
    if cal is None:
        for r in rows:
            r["Value unit"] = unit
        return OpResult(results=rows)
    # ImageJ computes the statistics OF the calibrated values (cTable), not the calibration of the raw statistics:
    # f(mean) != mean(f) for non-linear functions and Min/Max swap for the (usual) decreasing density calibrations.
    roi = ctx.roi
    plane = img.current_plane()
    H, W = plane.shape
    pw, ph = img.calibration.pixel_width, img.calibration.pixel_height
    wanted = set(s.measurements or DEFAULT_MEASUREMENTS) - {"area", "mode"}  # Mode: f(raw mode) like ImageJ's dmode = cTable[mode]
    for r in rows:
        r["Value unit"] = unit
        if roi is not None and roi.get("type") == "point":
            if isinstance(r.get("Value"), float) and np.isfinite(r["Value"]):
                r["Value"] = float(calibrated_values(img, r["Value"]))  # ImageJ: cal.getCValue(pixel)
            continue
        if roi is not None and roimod.is_line(roi):
            from ..profile import line_profile
            vals = calibrated_values(img, np.asarray(line_profile(plane, roi, img.calibration)["values"], dtype=np.float64))
        else:
            mask = np.ones((H, W), bool) if roi is None else roimod.to_mask(roi, (H, W))
            if thr is not None:  # the threshold limit applies to the RAW pixel values (ImageJ)
                lo, hi = thr
                mask &= (plane >= lo) & (plane <= hi)
            vals = calibrated_values(img, plane[mask])
            if "center_of_mass" in wanted and mask.any():
                ys, xs = np.nonzero(mask)
                w_ = np.where(np.isfinite(vals), vals, 0.0)
                sw = w_.sum()
                if sw != 0:
                    r["XM"] = float((xs * w_).sum() / sw + 0.5) * pw
                    r["YM"] = float((ys * w_).sum() / sw + 0.5) * ph
        cs = _pixel_stats(vals, wanted, pw * ph, img)  # "mode" excluded: _pixel_stats would bincount the float values for integer images
        for k in ("Mean", "StdDev", "Min", "Max", "Median", "IntDen", "Skew", "Kurt"):
            if k in cs:
                r[k] = cs[k]  # IntDen = Area × calibrated Mean (ImageJ); RawIntDen intentionally stays raw
        if isinstance(r.get("Mode"), float) and np.isfinite(r["Mode"]):
            r["Mode"] = float(calibrated_values(img, r["Mode"]))
    return OpResult(results=rows)


@op("analyze.angle", "Angle", ANA, kind="analysis", preview=False, requires_roi="line", roi_aware=False,
    description="Angle of a line selection (degrees, counter-clockwise), or the angle at the vertex of a 3-point segmented line (the angle tool).", keywords=["angle tool", "degrees"])
def angle(ctx):
    img = ctx.img
    roi = ctx.roi
    # ImageJ's Roi.getFloatAngle scales dx by pixelWidth and dy by pixelHeight (unconditionally, like Length), so an
    # anisotropic calibration reports the physical angle, not the pixel-grid one
    cal = img.calibration
    pw, ph = cal.pixel_width, cal.pixel_height
    pts = roimod.line_points(roi)
    if len(pts) >= 3:
        (x0, y0), (x1, y1), (x2, y2) = pts[0], pts[1], pts[2]
        v1 = np.array([(x0 - x1) * pw, (y0 - y1) * ph], float)
        v2 = np.array([(x2 - x1) * pw, (y2 - y1) * ph], float)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            raise ValueError("Degenerate angle selection")
        ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))  # = |180 - |a1 - a2|| of the two calibrated getFloatAngle values
        kind = "vertex angle"
    else:
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        ang = math.degrees(math.atan2((y1 - y2) * ph, (x2 - x1) * pw))  # roimod.line_angle, calibrated (degrees, counter-clockwise, y up)
        kind = "line angle"
    return OpResult(results=[{"Label": img.name, "Angle": ang}], message=f"{kind}: {ang:.2f}°")


@op("analyze.curve_fit", "Curve Fitting", ANA, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("source", "choice", "Results columns", choices=["Results columns", "Last plot", "Values"]), Param("x_column", "string", "X", help="Results column (or space-separated values when source = Values)"),
            Param("y_column", "string", "Y"), Param("function", "choice", "Straight Line", choices=CURVE_FIT_FUNCTIONS), Param("series", "int", 1, min=1, help="Plot series to fit (source = Last plot)"),
            Param("custom_equation", "text", "", help="User-defined function of x with parameters a–f (used when given or Function = Custom), e.g. y = a*exp(-b*x) + c; operators + - * / ^, "
                                                      "functions sin cos exp log sqrt pow abs tanh erf …"),
            Param("initial_params", "string", "", help="Starting values (space separated, one per parameter); empty = automatic guess"),
            Param("restarts", "int", 2, min=0, max=50, help="Extra random starting points (multi-start, the best fit wins) – ImageJ's 'Number of restarts'"),
            Param("show_residuals", "bool", False, help="Also report y − f(x) per point (plot 'residuals' and a Residuals table)")],
    description="Least-squares fit of x/y data (two Results columns, the last plot, or typed values) to a line, polynomial, exponential, power, log, Gaussian, error function, sigmoid, Rodbard, rational, "
                "Michaelis-Menten … or a user-defined equation (ImageJ CurveFitter parity).",
    keywords=["fit", "regression", "exponential", "gaussian", "custom", "user-defined"])
def curve_fit(ctx, source="Results columns", x_column="X", y_column="Y", function="Straight Line", series=1, custom_equation="", initial_params="", restarts=2, show_residuals=False):
    s = ctx.session
    if source == "Last plot":
        plot = s.last_plot
        if not plot or not plot.get("series"):
            raise ValueError("No plot available – make a plot first")
        ser = plot["series"][min(series, len(plot["series"])) - 1]
        x = np.asarray(ser.get("x") or plot.get("x") or range(len(ser["values"])), float)
        if len(x) != len(ser["values"]):  # series shorter than the shared axis (e.g. Analyze Line Graph curves)
            x = x[:len(ser["values"])]
        y = np.asarray(ser["values"], float)
        xl, yl = plot.get("x_label", "x"), plot.get("y_label", "y")
    elif source == "Values":
        x, y = parse_numbers(x_column), parse_numbers(y_column)
        xl, yl = "x", "y"
    else:
        rt = s.results
        pairs = [(r[x_column], r[y_column]) for r in rt.rows if isinstance(r.get(x_column), (int, float)) and isinstance(r.get(y_column), (int, float))]
        if not pairs:
            raise ValueError(f"No numeric data in columns {x_column!r} / {y_column!r}. Columns: {', '.join(rt.columns)}")
        x = np.array([p[0] for p in pairs], float); y = np.array([p[1] for p in pairs], float)
        xl, yl = x_column, y_column
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]  # same rule as fitting.fit(); the plot / points below must see the same finite pairs
    equation = str(custom_equation or "").strip()
    if function == CUSTOM and not equation:
        raise ValueError("Custom fit: enter an equation such as y = a*exp(-b*x) + c in 'Custom equation'")
    if equation:
        compile_equation(equation)  # validate (and report) a bad equation before touching the data
        res = fit(CUSTOM, x, y, equation=equation, initial_params=initial_params, restarts=restarts)
        function = CUSTOM
    else:
        res = fit(function, x, y, initial_params=initial_params, restarts=restarts)
    row: dict[str, Any] = {"Label": function if not equation else f"Custom: {res['formula']}", "Formula": res["formula"]}
    for n, p in zip(res["names"], res["params"]):
        row[n] = p
    row["R²"] = res["r2"]
    row["RMS"] = res["rms"]
    row["N"] = res["n"]
    order = np.argsort(x)
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    plot = {"title": f"{function}: {res['formula']}  (R² = {res['r2']:.4f})", "x": xs.tolist(),
            "series": [{"name": "fit", "values": evaluate(function, res["params"], xs, equation=equation or None).tolist()},
                       {"name": "data", "values": np.interp(xs, x[order], y[order]).tolist()}],
            "x_label": xl, "y_label": yl, "points": {"x": x.tolist(), "y": y.tolist()}, "fit": {k: v for k, v in res.items()}}
    table = None
    if show_residuals:
        resid = residuals(function, res["params"], x, y, equation=equation or None)
        fitted = y - resid
        plot["residuals"] = {"x": x.tolist(), "values": resid.tolist(), "title": f"Residuals of {res['formula']}", "x_label": xl, "y_label": f"{yl} − fit"}
        table = {"name": "Residuals", "columns": ["X", "Y", "Fit", "Residual"], "rows": [[float(a), float(b), float(c), float(d)] for a, b, c, d in zip(x, y, fitted, resid)]}
    ptxt = ", ".join(f"{n} = {p:.4g}" for n, p in zip(res["names"], res["params"]))
    return OpResult(results=[row], plot=plot, table=table, message=f"{res['formula'] if equation else function}: {ptxt}; R² = {res['r2']:.4f}, RMS = {res['rms']:.4g}")


@op("analyze.fractal_box_count", "Fractal Box Count", ANA, kind="analysis", preview=False, roi_aware=False, supports_rgb=False,
    params=[Param("box_sizes", "string", "2,3,4,6,8,12,16,32,64", help="Comma-separated box sizes in pixels")],
    description="Box-counting fractal dimension of the binary (or thresholded) current plane: slope of log(count) vs log(1/size).", keywords=["fractal dimension", "box counting"])
def fractal_box_count(ctx, box_sizes="2,3,4,6,8,12,16,32,64"):
    img = ctx.img
    p = img.current_plane()
    thr = current_threshold(ctx)
    b = ((p >= thr[0]) & (p <= thr[1])) if thr is not None else (p > 0)
    if ctx.mask is not None:
        b &= ctx.mask
    if not b.any():
        raise ValueError("No foreground pixels (image must be binary or thresholded)")
    sizes = sorted({int(v) for v in parse_numbers(box_sizes) if v >= 1})
    if len(sizes) < 2:
        raise ValueError("Need at least two box sizes")
    H, W = b.shape
    counts = []
    for sz in sizes:
        best = None
        # minimum over a few grid offsets removes most of the grid-alignment bias
        for oy in sorted({0, sz // 2}):
            for ox in sorted({0, sz // 2}):
                ph, pw = (-(H + oy)) % sz, (-(W + ox)) % sz
                bp = np.pad(b, ((oy, ph), (ox, pw)))
                blocks = bp.reshape(bp.shape[0] // sz, sz, bp.shape[1] // sz, sz).any(axis=(1, 3))
                n = int(blocks.sum())
                best = n if best is None else min(best, n)
        counts.append(max(best or 0, 1))
    lx = np.log(np.array(sizes, float)); ly = np.log(np.array(counts, float))
    slope, intercept = np.polyfit(lx, ly, 1)
    D = float(-slope)
    pred = slope * lx + intercept
    r2 = 1 - float(np.sum((ly - pred) ** 2) / max(np.sum((ly - ly.mean()) ** 2), 1e-12))
    row = {"Label": img.name, "D": D, "R²": r2}
    for sz, c in zip(sizes, counts):
        row[f"C{sz}"] = c
    plot = {"title": f"Fractal box count of {img.name}: D = {D:.3f}", "x": lx.tolist(), "series": [{"name": "log(count)", "values": ly.tolist()}, {"name": "fit", "values": pred.tolist()}],
            "x_label": "log(box size)", "y_label": "log(count)"}
    return OpResult(results=[row], plot=plot, message=f"D = {D:.3f} (R² = {r2:.3f})")


@op("analyze.surface_plot", "Surface Plot", ANA, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("smoothing", "float", 1.0, min=0, max=20, step=0.5, help="Gaussian sigma applied before plotting (0 = none)"), Param("wireframe", "bool", False),
            Param("lut", "choice", "Current", choices=["Current"] + LUT_NAMES), Param("invert", "bool", False, help="Plot the inverted values"),
            Param("elevation", "float", 35.0, min=0, max=90, step=5), Param("azimuth", "float", -60.0, min=-180, max=180, step=5),
            Param("width", "int", 640, min=100, max=4000), Param("height", "int", 480, min=100, max=4000), Param("max_points", "int", 160, min=16, max=1024, advanced=True, help="Resample the plane to at most this many points per axis")],
    description="3-D surface rendering of the current plane (or the selection) as a new RGB image.", keywords=["3d plot", "surface", "mesh"])
def surface_plot(ctx, smoothing=1.0, wireframe=False, lut="Current", invert=False, elevation=35.0, azimuth=-60.0, width=640, height=480, max_points=160):
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure
    from ..luts import get_lut
    img = ctx.img
    d = img.display
    p = img.current_plane().astype(np.float32)
    if ctx.roi is not None and roimod.is_area(ctx.roi):
        x, y, w, h = roimod.bounding_box(ctx.roi)
        x0, y0 = max(int(x), 0), max(int(y), 0)
        p = p[y0:int(y + h) + 1, x0:int(x + w) + 1]
    if smoothing > 0:
        p = ndi.gaussian_filter(p, smoothing)
    if invert:
        p = -p
    step = max(int(math.ceil(max(p.shape) / max_points)), 1)
    p = p[::step, ::step]
    ys, xs = np.mgrid[0:p.shape[0], 0:p.shape[1]]
    ys = ys * step * img.calibration.pixel_height
    xs = xs * step * img.calibration.pixel_width
    lut_name = img.display.channels[d.c].lut if lut == "Current" else lut
    cmap = ListedColormap(get_lut(lut_name).astype(np.float32) / 255.0)
    dpi = 100
    fig = Figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    if wireframe:
        ax.plot_wireframe(xs, ys, p, rstride=1, cstride=1, linewidth=0.5, color="#444444")
    else:
        ax.plot_surface(xs, ys, p, cmap=cmap, linewidth=0, antialiased=False, rstride=1, cstride=1)
    ax.view_init(elev=elevation, azim=azimuth)
    unit = img.calibration.unit
    ax.set_xlabel(f"x ({unit})"); ax.set_ylabel(f"y ({unit})"); ax.set_zlabel(img.calibration.value_unit or "value")
    ax.set_title(f"{img.name}")
    fig.tight_layout()
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[..., :3])
    new = LumenImage(rgb, name=f"Surface Plot of {img.name}", axes="YXS", rgb=True)
    return OpResult(new_images=[new], message=f"Surface plot ({matplotlib.__version__})")


@op("analyze.stack_histogram", "Stack Histogram", ANA, kind="analysis", preview=False,
    params=[Param("bins", "int", 256, min=2, max=65536), Param("all_channels", "bool", False)],
    description="Histogram (and statistics) of every slice and frame of the current channel, within the selection.", keywords=["histogram", "stack"])
def stack_histogram(ctx, bins=256, all_channels=False):
    img = ctx.img
    d = img.display
    mask = ctx.mask
    sub = img.data if all_channels else img.data[:, :, d.c:d.c + 1]
    if mask is not None:
        vals = sub[..., mask].ravel()
    else:
        vals = sub.ravel()
    if vals.dtype == np.float32:
        vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No pixels")
    if vals.dtype == np.uint8 and bins == 256:
        counts = np.bincount(vals.astype(np.int64), minlength=256)[:256]
        edges = np.arange(257)
    else:
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            hi = lo + 1
        counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    stats = {"count": int(vals.size), "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0, "min": float(vals.min()), "max": float(vals.max()),
             "mode": float(edges[int(np.argmax(counts))]), "median": float(np.median(vals))}
    row = {"Label": f"{img.name} (stack)", "N": stats["count"], "Mean": stats["mean"], "StdDev": stats["std"], "Min": stats["min"], "Max": stats["max"], "Mode": stats["mode"], "Median": stats["median"]}
    return OpResult(histogram={"counts": counts.tolist(), "edges": edges.tolist(), "stats": stats, "title": f"Stack histogram of {img.name}"}, results=[row],
                    message=f"{stats['count']} voxels, mean {stats['mean']:.4g}")


def _manager_rois(ctx, only_this_image: bool = False) -> list[dict[str, Any]]:
    img = ctx.img
    rois = [r for r in ctx.session.roi_manager.to_list() if (not only_this_image or r.get("image_id") in (None, img.id))]
    if not rois and ctx.roi is not None:
        rois = [roimod.normalize(ctx.roi)]
    return rois


@op("analyze.multi_measure", "Multi Measure (ROI Manager)", ANA, kind="analysis", preview=False, roi_aware=False,
    params=[Param("axis", "choice", "Z", choices=["Z", "T", "Z and T"], label="Iterate over"), Param("one_row_per_slice", "bool", True, help="Columns Mean1, Mean2 … per ROI; otherwise one row per ROI and slice"),
            Param("only_this_image", "bool", False, help="Skip ROIs that were saved from other images"), Param("append", "bool", False, help="Append to the Results table instead of replacing it")],
    description="Measure every ROI of the ROI Manager on every slice / frame of the current channel (ROI Manager ▸ Multi Measure).", keywords=["roi manager", "measure all", "time series"])
def multi_measure(ctx, axis="Z", one_row_per_slice=True, only_this_image=False, append=False):
    s = ctx.session
    img = ctx.img
    d = img.display
    rois = _manager_rois(ctx, only_this_image)
    if not rois:
        raise ValueError("The ROI Manager is empty and there is no selection")
    zs = range(img.nz) if axis in ("Z", "Z and T") else [d.z]
    ts = range(img.nt) if axis in ("T", "Z and T") else [d.t]
    meas = [m for m in s.measurements if m != "stack_position"] or ["mean"]
    rows: list[dict[str, Any]] = []
    for t in ts:
        for z in zs:
            n = (t * img.nz + z + 1) if axis == "Z and T" else (z + 1 if axis == "Z" else t + 1)
            if one_row_per_slice:
                row: dict[str, Any] = {"Label": img.name, "Slice": n}
                for i, r in enumerate(rois, 1):
                    m = measure_roi(img, r, meas, t=t, z=z, c=d.c, options=ctx.session.measure_options)[0]
                    for k, v in m.items():
                        if k == "Label":
                            continue
                        row[f"{k}{i}"] = v
                rows.append(row)
            else:
                for i, r in enumerate(rois, 1):
                    m = measure_roi(img, r, meas, t=t, z=z, c=d.c, label=r.get("name") or f"ROI {i}", options=ctx.session.measure_options)[0]
                    m["Label"] = f"{img.name}:{r.get('name') or i}"
                    m["Slice"] = n
                    rows.append(m)
    return OpResult(results=rows, results_replace=not append, message=f"{len(rows)} rows ({len(rois)} ROIs × {len(ts) * len(zs)} slices)")


def roi_centroid(roi: dict[str, Any], shape: tuple[int, int]) -> tuple[float, float]:
    t = roi.get("type")
    if t in ("rect", "oval"):
        return roi["x"] + roi["w"] / 2, roi["y"] + roi["h"] / 2
    if t == "line":
        return (roi["x1"] + roi["x2"]) / 2, (roi["y1"] + roi["y2"]) / 2
    if t in ("polyline", "point"):
        pts = np.asarray(roi["points"], float)
        if len(pts) == 0:
            return 0.0, 0.0
        return float(pts[:, 0].mean()), float(pts[:, 1].mean())
    m = roimod.to_mask(roi, shape)
    if m.any():
        cy, cx = ndi.center_of_mass(m)
        return float(cx) + 0.5, float(cy) + 0.5
    x, y, w, h = roimod.bounding_box(roi)
    return x + w / 2, y + h / 2


LABEL_FORMATS = ["Index", "Name", "Measurement", "Value"]
LABEL_SOURCES = ["ROI Manager", "Overlay"]
LABEL_POSITIONS = ["Centroid", "Bounding box"]


def _overlay_shaped(img: LumenImage) -> list[dict[str, Any]]:
    return [o for o in img.overlay if o.get("type") in roimod.AREA_TYPES or o.get("type") in roimod.LINE_TYPES or o.get("type") == "point"]


def _roi_plane(roi: dict[str, Any], img: LumenImage) -> tuple[int, int, int]:
    """(t, z, c) of a ROI pinned to a plane (its 'position'), else the current plane."""
    d = img.display
    pos = roi.get("position") if isinstance(roi.get("position"), dict) else {}
    t = pos.get("t") if pos.get("t") is not None else d.t
    z = pos.get("z") if pos.get("z") is not None else d.z
    c = pos.get("c") if pos.get("c") is not None else d.c
    return min(max(int(t), 0), img.nt - 1), min(max(int(z), 0), img.nz - 1), min(max(int(c), 0), img.nc - 1)


def _format_value(v: Any, decimals: int) -> str:
    if isinstance(v, float):
        return f"{v:.{decimals}f}" if np.isfinite(v) else "NaN"
    return str(v if v is not None else "?")


@op("analyze.label", "Labels… (ROI names / values as overlay)", ANA, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("what", "choice", "Index", choices=LABEL_FORMATS, label="Label format",
                  help="Index = 1, 2, …; Name = the ROI name; Measurement = a Results column measured on the ROI; Value = the pixel value at the label position"),
            Param("column", "string", "Mean", help="Results column to show when what = Measurement"),
            Param("decimals", "int", 2, min=0, max=9),
            Param("use_names", "bool", False, label="Use names as labels", help="Index labels show the ROI's name when it has one (ImageJ's 'Use names as labels')"),
            Param("font_size", "int", 0, min=0, max=200, help="0 = the Overlay Options label font size (default 12)"),
            Param("bold", "bool", False),
            Param("color", "string", "", label="Label colour", help="#rrggbb or a colour name; empty = the Overlay Options label colour (default #ffff00)"),
            Param("background", "bool", False, label="Draw background", help="Opaque box behind the text"),
            Param("at", "choice", "Centroid", choices=LABEL_POSITIONS, label="Draw at", help="The ROI's centroid, or the centre of its bounding box (where ImageJ draws overlay labels)"),
            Param("source", "choice", "ROI Manager", choices=LABEL_SOURCES, help="Label the ROI Manager's ROIs (or the selection), or the ROI-shaped items already in the overlay"),
            Param("only_this_image", "bool", False, help="Skip ROI Manager ROIs that were saved from other images")],
    description="Add a text overlay item to every ROI-Manager ROI (or the selection, or every ROI-shaped overlay item) showing its index, name, a measurement or the pixel "
                "value, with ImageJ's Overlay ▸ Labels… options (use names, font size, bold, colour, background, centroid / bounding box). Labels of ROIs pinned to a plane "
                "stay on that plane.", keywords=["label", "annotate", "numbers", "overlay labels", "names"])
def label_rois(ctx, what="Index", column="Mean", decimals=2, use_names=False, font_size=0, bold=False, color="", background=False, at="Centroid",
               source="ROI Manager", only_this_image=False):
    img = ctx.img
    s = ctx.session
    if source == "Overlay":
        rois = _overlay_shaped(img)
        if not rois:
            raise ValueError("The overlay has no ROI-shaped items – use Overlay ▸ From ROI Manager / Add Selection first")
    else:
        rois = _manager_rois(ctx, only_this_image)
        if not rois:
            raise ValueError("The ROI Manager is empty and there is no selection")
    fs = int(font_size) if font_size and int(font_size) > 0 else int(s.settings.get("overlay_label_font_size", 12) or 12)
    col = str(color or "").strip() or str(s.settings.get("overlay_label_color", "#ffff00") or "#ffff00")
    if col and not col.startswith("#"):
        from ..luts import parse_color
        r_, g_, b_ = parse_color(col)
        col = f"#{r_:02x}{g_:02x}{b_:02x}"
    items = []
    n = len(rois)
    # overlay items are numbered by their position among ALL overlay items (the rule the viewport, Measure Overlay and
    # the burn-in share); ROI Manager entries by their position in the manager
    overlay_index = {id(o): k for k, o in enumerate(img.overlay, 1)} if source == "Overlay" else {}
    for i, r in enumerate(rois, 1):
        ctx.check_cancel()
        i = overlay_index.get(id(r), i)
        t, z, c = _roi_plane(r, img)
        if at == "Bounding box":
            bx, by, bw, bh = roimod.bounding_box(r)
            cx, cy = bx + bw / 2.0, by + bh / 2.0
        else:
            cx, cy = roi_centroid(r, (img.height, img.width))
        name = r.get("name")
        if what == "Name" or (what == "Index" and use_names and name):
            text = str(name or i)
        elif what == "Measurement":
            meas = list(dict.fromkeys(s.measurements + [_measurement_for_column(column)]))
            row = measure_roi(img, r, meas, t=t, z=z, c=c)[0]
            text = _format_value(row.get(column), decimals)
        elif what == "Value":
            xi, yi = int(cx), int(cy)
            if 0 <= yi < img.height and 0 <= xi < img.width:
                if img.rgb:
                    text = ",".join(str(int(v)) for v in img.data[t, z, :, yi, xi])
                else:
                    v = img.data[t, z, c, yi, xi]
                    text = _format_value(float(v), decimals) if img.data.dtype.kind == "f" else str(int(v))
            else:
                text = "?"
        else:
            text = str(i)
        item: dict[str, Any] = {"id": roimod.new_id(), "type": "text", "x": float(cx) - 0.25 * fs * len(text), "y": float(cy) - fs / 2.0, "text": text,
                                "font_size": fs, "color": col, "background": bool(background), "bold": bool(bold), "roi_id": r.get("id")}
        if isinstance(r.get("position"), dict):
            item["position"] = dict(r["position"])
        items.append(item)
        if n > 20:
            ctx.progress(i / n, f"Label {i}/{n}")
    return OpResult(overlay=[*img.overlay, *items], message=f"{len(items)} labels added to the overlay")


def _measurement_for_column(column: str) -> str:
    col = column.lower()
    table = {"area": "area", "mean": "mean", "stddev": "std", "mode": "mode", "min": "min", "max": "max", "x": "centroid", "y": "centroid", "xm": "center_of_mass", "ym": "center_of_mass",
             "perim.": "perimeter", "bx": "bounding_rect", "by": "bounding_rect", "width": "bounding_rect", "height": "bounding_rect", "major": "fit_ellipse", "minor": "fit_ellipse", "angle": "fit_ellipse",
             "circ.": "shape_descriptors", "ar": "shape_descriptors", "round": "shape_descriptors", "solidity": "shape_descriptors",
             "feret": "feret", "feretangle": "feret", "minferet": "feret", "feretx": "feret", "ferety": "feret", "intden": "integrated_density", "rawintden": "integrated_density",
             "median": "median", "skew": "skewness", "kurt": "kurtosis", "%area": "area_fraction", "ch": "stack_position", "slice": "stack_position", "frame": "stack_position"}
    return table.get(col, "mean")


# ====================================================================== Image ▸ Overlay
@op("overlay.text", "Add Text", OVL, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("text", "string", "Text"), Param("x", "float", 10.0, step=1, help="Ignored when a selection exists and 'at selection' is on"), Param("y", "float", 10.0, step=1),
            Param("font_size", "int", 18, min=4, max=300), Param("color", "string", "#ffff00"), Param("background", "bool", False), Param("at_selection", "bool", True, help="Place the text at the selection's bounding box")],
    description="Add a text item to the overlay (burn it in with Overlay ▸ Burn In).", keywords=["annotate", "label", "caption"])
def overlay_text(ctx, text="Text", x=10.0, y=10.0, font_size=18, color="#ffff00", background=False, at_selection=True):
    img = ctx.img
    if at_selection and ctx.roi is not None:
        bx, by, bw, bh = roimod.bounding_box(ctx.roi)
        x, y = bx, by
        if roimod.is_area(ctx.roi) and bh >= 6:
            font_size = int(max(min(bh, 300), 4))
    item = {"id": roimod.new_id(), "type": "text", "x": float(x), "y": float(y), "text": str(text), "font_size": int(font_size), "color": color, "background": bool(background)}
    return OpResult(overlay=[*img.overlay, item], message=f"Text added at ({x:g}, {y:g})")


@op("overlay.arrow", "Add Arrow", OVL, kind="image", preview=False, roi_aware=False, scope_param=False, requires_roi="line",
    params=[Param("color", "string", "#ffff00"), Param("stroke_width", "float", 2.0, min=0.5, max=50, step=0.5), Param("head_size", "float", 12.0, min=1, max=200, step=1), Param("double_headed", "bool", False)],
    description="Turn the line selection into an arrow overlay item (first point = tail, last point = head).", keywords=["annotate", "pointer"])
def overlay_arrow(ctx, color="#ffff00", stroke_width=2.0, head_size=12.0, double_headed=False):
    img = ctx.img
    pts = roimod.line_points(ctx.roi)
    (x1, y1), (x2, y2) = pts[0], pts[-1]
    item = {"id": roimod.new_id(), "type": "arrow", "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2), "color": color,
            "stroke_width": float(stroke_width), "head_size": float(head_size), "double_headed": bool(double_headed)}
    return OpResult(overlay=[*img.overlay, item], clear_roi=True, message="Arrow added")


@op("overlay.calibration_bar", "Calibration Bar", OVL, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("position", "choice", "Upper Right", choices=POSITIONS), Param("lut", "choice", "Current", choices=["Current"] + LUT_NAMES),
            Param("width", "int", 20, min=2, max=500, label="Bar width (px)"), Param("height", "int", 128, min=10, max=4000, label="Bar height (px)"), Param("font_size", "int", 12, min=4, max=200),
            Param("color", "string", "#ffffff", label="Text colour"), Param("decimals", "int", 0, min=0, max=9), Param("ticks", "int", 5, min=2, max=20), Param("fill", "bool", True, label="Opaque background")],
    description="Add a LUT colour bar with the current channel's display range (and value unit) to the overlay.", keywords=["colorbar", "legend", "lut bar"])
def calibration_bar(ctx, position="Upper Right", lut="Current", width=20, height=128, font_size=12, color="#ffffff", decimals=0, ticks=5, fill=True):
    img = ctx.img
    ch = img.display.channels[img.display.c]
    lut_name = ch.lut if lut == "Current" else lut
    lo, hi = ch.min, ch.max
    cal = intensity_calibration(img)
    unit = (cal.get("unit") if cal else img.calibration.value_unit) or ""
    if cal:
        lo, hi = float(calibrated_values(img, lo)), float(calibrated_values(img, hi))
    item = {"id": roimod.new_id(), "type": "calibration_bar", "position": position, "lut": lut_name, "inverted": bool(getattr(ch, "inverted", False)),
            "min": float(lo), "max": float(hi), "unit": unit, "width": int(width), "height": int(height),
            "font_size": int(font_size), "color": color, "decimals": int(decimals), "ticks": int(ticks), "fill": bool(fill)}
    overlay = [o for o in img.overlay if o.get("type") != "calibration_bar"] + [item]
    return OpResult(overlay=overlay, message=f"Calibration bar: {lut_name} {lo:g} – {hi:g} {unit}".rstrip())


def time_stamp_item(format: str, position: str, color: str, font_size: int, start_frame: int, start_time: float = 0.0, interval: float = 0.0,
                    label_start: int = 1, label_interval: int = 1) -> dict[str, Any]:
    """The overlay item both Time Stamp and Label Slices add; rendered by overlay_render.time_stamp_text (and the viewport's
    timeStampText): time = start_time + (t - start_frame) * (interval or frame_interval or 1), label = label_start + k * label_interval."""
    return {"id": roimod.new_id(), "type": "time_stamp", "position": position, "format": format, "color": color, "font_size": int(font_size), "start_frame": int(start_frame),
            "start_time": float(start_time), "interval": float(interval), "label_start": int(label_start), "label_interval": int(label_interval)}


@op("overlay.time_stamp", "Time Stamp", OVL, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("format", "string", "t = {time:.1f} {unit}", help="Placeholders: {time} {unit} {frame} {slice}"), Param("position", "choice", "Upper Left", choices=POSITIONS),
            Param("color", "string", "#ffffff"), Param("font_size", "int", 16, min=4, max=300), Param("start_frame", "int", 0, min=0, help="Frame index (0-based) that corresponds to t = start time"),
            Param("start_time", "float", 0.0, label="Start time", help="Time shown at the start frame"),
            Param("interval", "float", 0.0, min=0, help="Time per frame (0 = the calibration's frame interval)")],
    description="Add a time stamp overlay that shows the elapsed time of the current frame (start time + frames × interval; the interval defaults to the frame interval).",
    keywords=["timestamp", "time series", "label frames"])
def time_stamp(ctx, format="t = {time:.1f} {unit}", position="Upper Left", color="#ffffff", font_size=16, start_frame=0, start_time=0.0, interval=0.0):
    img = ctx.img
    item = time_stamp_item(format, position, color, font_size, start_frame, start_time, interval)
    overlay = [o for o in img.overlay if o.get("type") != "time_stamp"] + [item]
    return OpResult(overlay=overlay, message="Time stamp added")


@op("overlay.list", "List Overlay Elements", OVL, kind="analysis", preview=False, roi_aware=False, description="List every overlay item (type, position, size, text) in the Results table.")
def overlay_list(ctx):
    img = ctx.img
    rows = []
    for i, o in enumerate(img.overlay, 1):
        typ = o.get("type", "?")
        row: dict[str, Any] = {"Label": img.name, "Index": i, "Type": typ, "Name": o.get("name", "") or o.get("text", "") or o.get("format", "")}
        if typ in roimod.AREA_TYPES or typ in roimod.LINE_TYPES or typ == "point":
            x, y, w, h = roimod.bounding_box(o)
        elif typ == "arrow":
            x, y = min(o["x1"], o["x2"]), min(o["y1"], o["y2"]); w, h = abs(o["x2"] - o["x1"]), abs(o["y2"] - o["y1"])
        elif typ == "text":
            x, y, w, h = o.get("x", 0), o.get("y", 0), 0, o.get("font_size", 0)
        else:
            x, y, w, h = 0, 0, o.get("width", 0), o.get("height", 0)
        row.update({"X": float(x), "Y": float(y), "Width": float(w), "Height": float(h), "Color": o.get("color", ""), "Position": o.get("position", "") if isinstance(o.get("position"), str) else ""})
        rows.append(row)
    if not rows:
        return OpResult(message="The overlay is empty")
    return OpResult(results=rows, message=f"{len(rows)} overlay elements")


@op("overlay.to_roi_manager", "Overlay to ROI Manager", OVL, kind="analysis", preview=False, roi_aware=False,
    params=[Param("remove", "bool", False, help="Remove the transferred items from the overlay")], description="Copy every ROI-shaped overlay item into the ROI Manager.")
def overlay_to_manager(ctx, remove=False):
    img = ctx.img
    shaped = [o for o in img.overlay if o.get("type") in roimod.AREA_TYPES or o.get("type") in roimod.LINE_TYPES or o.get("type") == "point"]
    if not shaped:
        raise ValueError("No ROI-shaped overlay items")
    if remove:
        img.overlay = [o for o in img.overlay if o not in shaped]
    # strip the overlay item's id so the manager assigns a fresh one (items that came from the manager, or a previous run, would otherwise duplicate ids)
    return OpResult(rois_add=[{k: v for k, v in o.items() if k != "id"} for o in shaped], message=f"{len(shaped)} items added to the ROI Manager")


@op("overlay.burn_in", "Burn In Overlay", OVL, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("all_planes", "bool", False, help="Render every slice / frame (time stamps update per frame)"), Param("keep_overlay", "bool", False, help="Keep the overlay on the new image")],
    description="Rasterise text, arrows, calibration bars, time stamps, scale bars and ROI outlines of the overlay into a new RGB image.", keywords=["flatten", "rasterize", "export"])
def burn_in(ctx, all_planes=False, keep_overlay=False):
    from ..render import render_rgb
    img = ctx.img
    if not img.overlay:
        raise ValueError("The overlay is empty")
    d = img.display
    if all_planes:
        out = np.zeros((img.nt, img.nz, 3, img.height, img.width), np.uint8)
        for t in range(img.nt):
            for z in range(img.nz):
                out[t, z] = burn_overlay(render_rgb(img, t=t, z=z), img.overlay, img, t=t, z=z, settings=ctx.session.settings).transpose(2, 0, 1)
    else:
        out = burn_overlay(render_rgb(img), img.overlay, img, t=d.t, z=d.z, settings=ctx.session.settings).transpose(2, 0, 1)[None, None]
    new = img.copy(name=f"{img.name} (burned)", data=out)
    new.metadata.pop("intensity_calibration", None)  # the RGB bytes are display colours (ImageJ drops the density calibration and value unit for RGB)
    new.calibration.value_unit = ""
    new.rgb = True
    new.reset_display()
    new.display.mode = "composite"
    if not keep_overlay:
        new.overlay = []
    return OpResult(new_images=[new], message=f"{len(img.overlay)} overlay items burned in")


# ====================================================================== Plugins ▸ Scripting
@op("script.run_ij_macro", "Run ImageJ Macro", "Plugins/Scripting", kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("macro", "text", 'run("Gaussian Blur...", "sigma=2");\nrun("Measure");\nprint("done: " + nResults());', label="ImageJ 1.x macro code")],
    description="Run ImageJ-1 macro code (run(\"…\"), makeRectangle, setAutoThreshold, roiManager, nResults, loops, functions …) through the compatibility layer.",
    keywords=["imagej", "fiji", "macro", "ijm", "compat"])
def run_ij_macro(ctx, macro=""):
    from ..ij_compat import IJ
    ij = IJ(ctx.session)
    ij.roi = ctx.roi
    had_roi = ctx.roi is not None
    output = ij.macro(macro)
    res = OpResult(log=output or None, message="Macro finished" + (f" – {len(output.splitlines())} line(s) of output" if output else ""))
    if ij.roi is not None:
        res.roi = ij.roi
    elif had_roi:
        res.clear_roi = True
    return res


@op("script.ij_command_list", "ImageJ Command List", "Plugins/Scripting", kind="analysis", requires_image=False, preview=False, roi_aware=False,
    description="Table of the ImageJ run(\"…\") commands and macro functions supported by the compatibility layer and the Lumen command each maps to.", keywords=["imagej", "compat", "help"])
def ij_command_list(ctx):
    from ..ij_compat import command_table, MACRO_FUNCTION_NAMES
    rows = command_table()
    table = {"name": "ImageJ commands", "columns": ["IJ command", "Lumen command", "Notes"], "rows": [[r["command"], r["lumen"], r["notes"]] for r in rows]}
    return OpResult(table=table, message=f"{len(rows)} run() commands, {len(MACRO_FUNCTION_NAMES)} macro functions")


# ---------------------------------------------------------------------- JVM languages through the optional Java bridge (lumen.core.ijshim.jvm_bridge)
JVM_LANGUAGE_CHOICES = ["groovy", "beanshell", "rhino", "jruby", "clojure"]
_JVM_EXAMPLE = 'import ij.IJ\nimp = IJ.getImage()\nIJ.run(imp, "Invert", "")\nIJ.log("inverted " + imp.getTitle())\n'


@op("script.run_jvm_script", "Run JVM Script (Groovy / BeanShell / Rhino)", "Plugins/Scripting", kind="analysis", requires_image=False, preview=False, roi_aware=False,
    scope_param=False,
    params=[Param("code", "text", _JVM_EXAMPLE, label="Script"),
            Param("language", "choice", "groovy", choices=JVM_LANGUAGE_CHOICES, label="Language",
                  help="JVM script language – needs a JDK plus Fiji's jars (subprocess mode, no Python package; macOS and the default without jpype) "
                       'or pip install "lumen-imaging[jvm]" (in-process mode, Linux / Windows); LUMEN_JVM_MODE=auto|subprocess|inprocess')],
    description="Run a Groovy, BeanShell, JavaScript (Rhino), JRuby or Clojure script with the real ImageJ 1.x classes – as a java subprocess over a Fiji "
                "installation's jars (no Python package needed) or on the optional in-process JPype bridge: the active image is bound as imp (changes come "
                "back in place, with undo), images the script shows become new images, and its output, Results table and ROI Manager entries are copied into the session.",
    keywords=["groovy", "beanshell", "rhino", "jruby", "clojure", "jvm", "java", "fiji", "script editor"])
def run_jvm_script_op(ctx, code="", language="groovy"):
    from ..ijshim.jvm_bridge import LANGUAGES, normalize_language, run_jvm_script
    had_roi = ctx.roi is not None
    out = run_jvm_script(ctx.session, code, language, roi=ctx.roi)
    if out.get("error"):
        raise RuntimeError(str(out["error"]) + (f"\n--- script output ---\n{out['stdout']}" if out.get("stdout") else ""))
    lang = normalize_language(language) or str(language)
    parts = []
    if out.get("changed"):
        parts.append("image changed")
    if out.get("new_images"):
        parts.append(f"{len(out['new_images'])} new image(s)")
    if out.get("results_rows"):
        parts.append(f"{out['results_rows']} result row(s)")
    if out.get("rois_copied"):
        parts.append(f"{out['rois_copied']} ROI Manager entries")
    msg = f"{LANGUAGES.get(lang, {}).get('label', lang)} script finished" + (f" – {', '.join(parts)}" if parts else "")
    res = OpResult(log=out.get("stdout") or None, message=msg)
    if "roi" in out:
        if out["roi"] is not None:
            res.roi = out["roi"]
        elif had_roi:
            res.clear_roi = True
    return res


@op("script.jvm_bridge_status", "JVM Script Bridge Status", "Plugins/Scripting", kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False,
    description="What the Java bridge for Groovy / BeanShell / Rhino / JRuby / Clojure scripts needs and has, for both modes: the subprocess mode (a JDK plus "
                "Fiji's jars – ij.jar, the language jars, Bio-Formats; no Python package) and the in-process mode (jpype1 + scyjava, Maven or local jars, "
                "whether its JVM is running); LUMEN_JVM_MODE picks the mode.",
    keywords=["jvm", "java", "groovy", "beanshell", "rhino", "bridge", "configuration", "status", "fiji", "jars"])
def jvm_bridge_status(ctx):
    from pathlib import Path
    from ..ijshim import jvm_bridge as jb
    st = jb.status(refresh=True)
    sub = st.get("subprocess") or {}
    inp = st.get("inprocess") or {}
    eff = st.get("effective_mode", "inprocess")
    rows = [["jpype1 + scyjava", "installed" if st["package"] else f"missing – {jb.PIP_EXTRA}"],
            ["JDK", (f"{sub.get('java_version')} – {sub.get('java_detail')}" if sub.get("java") else sub.get("java_detail") or jb._java_available()[1])],
            ["ij.jar", st["ij"]],
            ["JVM", "running" if st["started"] else ("not started (in-process mode starts it with the first script; the first run downloads the jars through Maven)"
                                                      if eff == "inprocess" else "one java process per script (subprocess mode)")],
            ["Headless", ("yes – no ROI Manager exchange (LUMEN_JVM_HEADLESS=0 to enable)" if jb.headless_requested() else "no (LUMEN_JVM_HEADLESS=0)")
             if eff == "inprocess" else "no – ImageJ batch mode in a java subprocess (hidden ROI Manager, no windows)"],
            ["Mode", f"{eff} (LUMEN_JVM_MODE={st.get('mode', 'auto')}; auto = subprocess on macOS or without jpype, else in-process)"],
            ["Fiji jars", sub.get("jars_detail") or "not checked"],
            ["Subprocess mode", ("available – " if sub.get("available") else "not available – ") + str(sub.get("detail", ""))],
            ["In-process mode", ("available – " if inp.get("available") else "not available – ") + str(inp.get("detail", ""))]]
    for lang, info in jb.LANGUAGES.items():
        if eff == "subprocess":
            ls = (sub.get("languages") or {}).get(lang) or {}
            text = f"ready – {Path(ls['jar']).name}" if ls.get("available") and ls.get("jar") else str(ls.get("detail", "not available"))
            rows.append([f"{info['label']} ({lang})", text + (f" (in-process alternative: {info['endpoint']})" if lang in ("jruby", "clojure") else "")])
        else:
            rows.append([f"{info['label']} ({lang})", info["endpoint"] + (" – enabled" if lang in st["languages"] else f" – optional, enable with LUMEN_JVM_LANGUAGES={lang}")])
    for p in st["classpath"]:
        rows.append(["classpath", p])
    msg = f"Java bridge available ({st['detail']})" if st["available"] else f"Java bridge not available: {st['detail']}"
    return OpResult(table={"name": "JVM Script Bridge", "columns": ["Item", "Status"], "rows": rows}, message=msg)
