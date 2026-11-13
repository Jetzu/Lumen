"""Image ▸ Adjust ▸ Bleach Correction – Fiji's CorrectBleach plugin (Kota Miura, EMBL CMCI) on a time series.

Three methods, as in the plugin's "Correction Method" choice:

* **Simple Ratio** – every frame is multiplied by ``(mean(frame 0) − background) / (mean(frame t) − background)``;
  the background intensity is the plugin's second dialog (default 0) and the means are taken inside the area
  selection when there is one.
* **Exponential Fit** – the per-frame means are fitted with ImageJ's "Exponential with Offset"
  (``y = a·exp(−b·t) + c``, the CurveFitter model the plugin uses; ``lumen.core.fitting`` is the port, scipy's
  ``curve_fit`` the fallback) and every frame is multiplied by ``fit(0) / fit(t)``.
* **Histogram Matching** – every frame's histogram is matched to the first frame's
  (``skimage.exposure.match_histograms``).

The time axis is T; a plain Z stack is treated as a time series (slices = time points), as the plugin does with any
stack.  A 3-D time series (Z and T) gets one factor per time point from the mean over the whole volume.  Channels are
corrected independently.  The result is a new image ``Bleach_corrected_<title>`` (the plugin corrects a duplicate):
32-bit for the ratio / fit methods unless ``keep_type`` rounds and clips the frames back to the input type (what the
plugin's in-place multiply on an 8 / 16-bit stack does); Histogram Matching keeps the type.  The per-frame means
before / after, the ratios and the fit parameters come back as the table "Bleach correction", the log and a plot.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..registry import Param, OpResult, cast_to, op

CAT = "Image/Adjust"
METHODS = ["Simple Ratio", "Exponential Fit", "Histogram Matching"]
NAME_PREFIX = "Bleach_corrected_"
TABLE_NAME = "Bleach correction"


def _frame_means(series: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Mean of every time point of an (N, Z, Y, X) array, inside ``mask`` (Y, X) when given (float64)."""
    n = series.shape[0]
    sub = series[:, :, mask] if mask is not None else series
    return sub.reshape(n, -1).mean(axis=1, dtype=np.float64)


def fit_exponential(t: Any, y: Any) -> tuple[float, float, float]:
    """``(a, b, c)`` of ``y = a·exp(−b·t) + c`` through the frame means: ImageJ's "Exponential with Offset" fit (the
    CurveFitter model CorrectBleach uses, ported in ``lumen.core.fitting``), scipy's ``curve_fit`` when that fails."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    params: tuple[float, float, float] | None = None
    try:
        from ..fitting import fit
        res = fit("Exponential with Offset", t, y)
        a, b, c = (float(v) for v in res["params"][:3])
        if all(math.isfinite(v) for v in (a, b, c)) and float(res.get("r2", 0.0)) > 0.5:
            params = (a, b, c)
    except Exception:  # noqa: BLE001 – fall through to scipy
        params = None
    if params is None:
        from scipy.optimize import curve_fit
        p0 = (max(float(y[0] - y[-1]), 1e-6), 1.0 / max(len(t) - 1, 1), float(y[-1]))
        try:
            (a, b, c), _ = curve_fit(lambda x, a, b, c: a * np.exp(-b * x) + c, t, y, p0=p0, maxfev=20000)
        except (RuntimeError, ValueError) as e:
            raise ValueError(f"Exponential Fit: a·exp(−b·t) + c does not fit the frame means ({e}); try Simple Ratio") from e
        params = (float(a), float(b), float(c))
    return params


def _fmt(v: float) -> str:
    return f"{v:.4g}"


@op("adjust.bleach_correction", "Bleach Correction...", CAT, kind="image", scope_param=False, preview=False, supports_rgb=False,
    description="Compensate photobleaching in a time series (Fiji's Image ▸ Adjust ▸ Bleach Correction, Miura's CorrectBleach): "
                "Simple Ratio scales every frame to the first frame's mean, Exponential Fit scales by a fitted a·exp(−b·t) + c decay, "
                "Histogram Matching matches every frame's histogram to the first frame's. Slices of a plain stack count as time points.",
    keywords=["bleach", "bleaching", "photobleaching", "decay", "fading", "correct bleach", "CorrectBleach", "time series", "intensity drift", "exponential fit", "histogram matching"],
    params=[Param("method", "choice", "Simple Ratio", choices=METHODS, label="Correction method",
                  help="Simple Ratio: frame × mean(frame 1 − background) / mean(frame t − background); Exponential Fit: frame × fit(0) / fit(t) of a·exp(−b·t) + c through the frame means; Histogram Matching: every frame's histogram becomes frame 1's"),
            Param("background", "float", 0.0, min=0.0, label="Background intensity", help="Simple Ratio only: subtracted from the frame means before the ratios are formed (the plugin's second dialog)"),
            Param("keep_type", "bool", False, label="Keep the image type", help="Ratio / fit methods: round and clip the corrected frames to the input type (as the plugin's in-place multiply does) instead of writing 32-bit")])
def bleach_correction(ctx, method="Simple Ratio", background=0.0, keep_type=False):
    img = ctx.img
    if img.rgb:
        raise ValueError("Bleach Correction needs a grayscale time series – split the RGB channels first (Image ▸ Color ▸ Split Channels)")
    if method not in METHODS:
        raise ValueError(f"Bleach Correction: unknown method {method!r} (Simple Ratio, Exponential Fit or Histogram Matching)")
    axis = 0 if img.nt > 1 else 1  # T, else the slices of a plain stack are the time points (as the plugin does)
    n = int(img.data.shape[axis])
    if n < 2:
        raise ValueError("Bleach Correction needs a time series: more than one frame, or a stack whose slices are time points")
    mask = ctx.mask
    series = np.moveaxis(np.asarray(img.data), axis, 0)  # (N, other, C, Y, X)
    other, nc = series.shape[1], series.shape[2]
    integer = np.issubdtype(img.data.dtype, np.integer)
    if method == "Histogram Matching" or (keep_type and integer):
        out_dtype = img.data.dtype
    else:
        out_dtype = np.float32
    out = np.empty(series.shape, dtype=out_dtype)
    t = np.arange(n, dtype=np.float64)
    rows: list[list[Any]] = []
    plot_series: list[dict[str, Any]] = []
    fits: dict[str, dict[str, float]] = {}
    ratio_summary: list[tuple[float, float]] = []
    after_summary: list[tuple[float, float]] = []
    for c in range(nc):
        ctx.check_cancel()
        ctx.progress(c / nc, f"Channel {c + 1}/{nc}")
        sc = series[:, :, c].astype(np.float32)  # (N, other, Y, X)
        means_before = _frame_means(sc, mask)
        tag = f"C{c + 1} " if nc > 1 else ""
        ratios: np.ndarray | None = None
        fit_curve: np.ndarray | None = None
        if method == "Simple Ratio":
            ref = means_before[0] - background
            denom = means_before - background
            if ref <= 0 or np.any(denom <= 0):
                k = int(np.argmax(denom <= 0)) if np.any(denom <= 0) else 0
                raise ValueError(f"Simple Ratio: the mean of frame {k + 1} ({_fmt(float(means_before[k]))}) is not above the background {background:g}")
            ratios = ref / denom
            corrected = sc * ratios[:, None, None, None].astype(np.float32)
        elif method == "Exponential Fit":
            a, b, c0 = fit_exponential(t, means_before)
            f = a * np.exp(-b * t) + c0
            if not np.all(np.isfinite(f)) or np.any(f <= 0):
                raise ValueError("Exponential Fit: the fitted curve a·exp(−b·t) + c is not positive on every frame – check the series or use Simple Ratio")
            ratios = f[0] / f
            corrected = sc * ratios[:, None, None, None].astype(np.float32)
            half = math.log(2.0) / b if b > 0 else float("inf")
            fits[f"C{c + 1}"] = {"a": a, "b": b, "c": c0, "half_life_frames": half}
            fit_curve = f
        else:
            from skimage.exposure import match_histograms
            corrected = np.empty_like(sc)
            corrected[0] = sc[0]
            for k in range(1, n):
                ctx.check_cancel()
                for z in range(other):
                    corrected[k, z] = match_histograms(sc[k, z], sc[0, z])
        means_after = _frame_means(corrected, mask)
        out[:, :, c] = cast_to(corrected, out_dtype) if out_dtype != np.float32 else corrected
        for k in range(n):
            row: list[Any] = [k + 1]
            if nc > 1:
                row.append(c + 1)
            row += [float(means_before[k]), float(means_after[k]), (float(ratios[k]) if ratios is not None else "")]
            rows.append(row)
        plot_series.append({"name": f"{tag}Mean before", "values": means_before.tolist()})
        plot_series.append({"name": f"{tag}Mean after", "values": means_after.tolist()})
        if fit_curve is not None:
            plot_series.append({"name": f"{tag}Fit", "values": fit_curve.tolist()})
        if ratios is not None:
            ratio_summary.append((float(ratios.min()), float(ratios.max())))
        after_summary.append((float(means_after.mean()), float(means_after.std())))
        if c == 0:
            first_before = (float(means_before[0]), float(means_before[-1]))
    result = np.ascontiguousarray(np.moveaxis(out, 0, axis))
    res = img.copy(name=f"{NAME_PREFIX}{img.name}", data=result)
    res.reset_display(keep_luts=True)
    res.metadata["bleach_correction"] = method + (f" (background {background:g})" if method == "Simple Ratio" and background else "")
    columns = ["Frame"] + (["Channel"] if nc > 1 else []) + ["Mean before", "Mean after", "Ratio"]
    time_axis = "T" if axis == 0 else "Z"
    summary: dict[str, Any] = {"method": method, "background": float(background), "frames": n, "time_axis": time_axis, "channels": nc,
                               "output_type": str(np.dtype(out_dtype)), "fit": fits}
    parts = [f"Bleach Correction ({method}): {n} {'frames' if time_axis == 'T' else 'slices treated as time points'}"]
    if fits:
        f0 = fits["C1"]
        half = f0["half_life_frames"]
        parts.append(f"y = {_fmt(f0['a'])}·exp(−{_fmt(f0['b'])}·t) + {_fmt(f0['c'])}" + (f", half-life {_fmt(half)} frames" if math.isfinite(half) else ""))
    parts.append(f"mean {_fmt(first_before[0])} → {_fmt(first_before[1])} before, {_fmt(after_summary[0][0])} ± {_fmt(after_summary[0][1])} after")
    if ratio_summary:
        parts.append(f"ratios {_fmt(ratio_summary[0][0])}–{_fmt(ratio_summary[0][1])}")
    msg = "; ".join(parts) + (f" (channel 1 of {nc})" if nc > 1 else "")
    plot = {"title": f"Bleach correction of {img.name}", "x": [k + 1 for k in range(n)], "series": plot_series,
            "x_label": "Frame" if time_axis == "T" else "Slice (time point)", "y_label": "Mean intensity"}
    return OpResult(new_images=[res], table={"name": TABLE_NAME, "columns": columns, "rows": rows, "summary": summary}, plot=plot, message=msg, log=msg)
