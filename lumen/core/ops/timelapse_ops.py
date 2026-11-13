"""Plugins ▸ Time Lapse — Phase Map: per-pixel amplitude, phase and frequency of a periodic signal
over the frames of a time series (Fiji Plugins ▸ Time Lapse ▸ Phase Map).

The Circle / LOI interpolators of that menu are ``stacks.interpolate_rois`` (transform_extra.py), which
already interpolates oval ROIs (centre and radii) between keyframes.
"""
from __future__ import annotations

import math

import numpy as np

from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op
from .registration import _channel_index, _f32

CAT = "Plugins/Time Lapse"
DETREND = ["Mean", "Linear", "None"]
PHASE_UNITS = ["Degrees", "Radians"]
FREQ_OUTPUT = ["Period (frames)", "Frequency (cycles/frame)", "Frequency (Hz)", "Period (seconds)"]


def phase_map(signal: np.ndarray, min_period: float = 2.0, max_period: float = 0.0, detrend: str = "Mean", window: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dominant periodic component of ``signal`` (T, …) along axis 0 by FFT within a period range.

    Returns (amplitude, phase (radians, of a cosine: s(t) ≈ A·cos(2π·f·t + φ)), frequency (cycles per frame),
    the mean power spectrum over the pixels (per rfft bin) and the rfft frequencies). Bins outside
    ``[1/max_period, 1/min_period]`` are ignored (``max_period`` ≤ 0 = the whole series)."""
    s = _f32(signal).astype(np.float64)
    T = s.shape[0]
    if T < 3:
        raise ValueError("Needs at least 3 frames")
    t = np.arange(T, dtype=np.float64)
    if detrend == "Linear":
        tc = t - t.mean()
        slope = np.tensordot(tc, s - s.mean(axis=0), axes=(0, 0)) / float((tc ** 2).sum())
        s = s - s.mean(axis=0) - tc.reshape((T,) + (1,) * (s.ndim - 1)) * slope
    elif detrend == "Mean":
        s = s - s.mean(axis=0)
    w = np.hanning(T) if window else np.ones(T)
    s = s * w.reshape((T,) + (1,) * (s.ndim - 1))
    X = np.fft.rfft(s, axis=0)
    freqs = np.fft.rfftfreq(T, d=1.0)
    f_lo = 1.0 / float(max_period) if max_period and max_period > 0 else 0.0
    f_hi = 1.0 / max(float(min_period), 2.0)
    ok = (freqs > 0) & (freqs >= f_lo - 1e-12) & (freqs <= f_hi + 1e-12)
    if not ok.any():
        raise ValueError(f"No frequency bin within the period range {min_period:g}–{max_period if max_period > 0 else T:g} frames (the series has {T} frames)")
    mag = np.abs(X)
    mag_ok = np.where(ok.reshape((len(freqs),) + (1,) * (s.ndim - 1)), mag, -1.0)
    k = np.argmax(mag_ok, axis=0)
    best = np.take_along_axis(X, k[None], axis=0)[0]
    amplitude = 2.0 * np.abs(best) / float(w.sum())
    phase = np.angle(best)
    frequency = freqs[k]
    power = (mag ** 2).reshape(len(freqs), -1).mean(axis=1)
    return amplitude.astype(np.float32), phase.astype(np.float32), frequency.astype(np.float32), power, freqs


@op("timelapse.phase_map", "Phase Map", CAT, kind="analysis", preview=False, roi_aware=True, supports_rgb=False,
    params=[Param("channel", "int", 0, min=0, help="Channel to analyse (0 = current)"),
            Param("min_period", "float", 2.0, min=2, step=1, label="Minimum period (frames)"),
            Param("max_period", "float", 0.0, min=0, step=1, label="Maximum period (frames)", help="0 = the whole series"),
            Param("detrend", "choice", "Mean", choices=DETREND, help="Remove the mean or a linear trend of every pixel before the FFT"),
            Param("window", "bool", True, label="Hann window", help="Taper the series to reduce spectral leakage"),
            Param("phase_unit", "choice", "Degrees", choices=PHASE_UNITS),
            Param("frequency_output", "choice", "Period (frames)", choices=FREQ_OUTPUT, label="Third channel", help="Hz / seconds need the frame interval of the image calibration"),
            Param("reference", "choice", "None", choices=["None", "Mean signal", "Selection"], label="Phase relative to",
                  help="Subtract the phase of the mean signal of the image (or of the selection) at every pixel's frequency"),
            Param("all_slices", "bool", False, help="Z stacks: analyse every slice (otherwise the current slice)")],
    description="Fiji Plugins ▸ Time Lapse ▸ Phase Map: for every pixel the dominant periodic component over the frames (FFT within a period range): a 3-channel 32-bit image with the amplitude, the phase (of a cosine, optionally relative to the mean signal) and the period / frequency, plus the mean power spectrum as a plot and a summary row (dominant period of the mean signal, mean amplitude) in the Results table.",
    keywords=["phase map", "oscillation", "periodic", "frequency", "fft", "time lapse", "calcium", "wave"])
def phase_map_op(ctx, channel=0, min_period=2.0, max_period=0.0, detrend="Mean", window=True, phase_unit="Degrees", frequency_output="Period (frames)", reference="None", all_slices=False):
    img = ctx.img
    if img.nt < 3:
        raise ValueError("Needs a time series with at least 3 frames (T > 2)")
    c = _channel_index(img, channel)
    cal = img.calibration
    dt = float(cal.frame_interval) if cal.frame_interval and cal.frame_interval > 0 else 0.0
    if frequency_output in ("Frequency (Hz)", "Period (seconds)") and dt <= 0:
        raise ValueError("The image has no frame interval – set it in Image ▸ Properties or choose a frame-based output")
    zs = list(range(img.nz)) if all_slices and img.nz > 1 else [img.display.z]
    H, W = img.height, img.width
    out = np.zeros((1, len(zs), 3, H, W), np.float32)
    mask = ctx.mask
    T = img.nt
    rows = []
    spectrum = None
    freqs = None
    for i, z in enumerate(zs):
        ctx.check_cancel()
        ctx.progress(i / len(zs), f"Slice {z + 1}")
        series = img.data[:, z, c]
        amp, ph, freq, power, freqs = phase_map(series, min_period, max_period, detrend, window)
        if reference != "None":
            region = series.reshape(T, -1).mean(axis=1) if (reference == "Mean signal" or mask is None) else series[:, mask].reshape(T, -1).mean(axis=1)
            _, _, _, _, _ = phase_map(region, min_period, max_period, detrend, window)
            # phase of the reference at every pixel's own frequency bin
            r = _f32(region).astype(np.float64)
            if detrend == "Linear":
                tt = np.arange(T) - (T - 1) / 2.0
                r = r - r.mean() - tt * float((tt * (r - r.mean())).sum() / (tt ** 2).sum())
            elif detrend == "Mean":
                r = r - r.mean()
            R = np.fft.rfft(r * (np.hanning(T) if window else 1.0))
            k = np.rint(freq * T).astype(int)
            ph = np.angle(np.exp(1j * ph) * np.conj(np.exp(1j * np.angle(R[np.clip(k, 0, len(R) - 1)])))).astype(np.float32)
        if phase_unit == "Degrees":
            ph = np.degrees(ph).astype(np.float32)
        with np.errstate(divide="ignore"):
            if frequency_output == "Period (frames)":
                third = np.where(freq > 0, 1.0 / freq, 0.0)
            elif frequency_output == "Frequency (cycles/frame)":
                third = freq
            elif frequency_output == "Frequency (Hz)":
                third = freq / dt
            else:
                third = np.where(freq > 0, dt / freq, 0.0)
        out[0, i, 0], out[0, i, 1], out[0, i, 2] = amp, ph, third
        spectrum = power if spectrum is None else spectrum + power
        mean_sig = series.reshape(T, -1).mean(axis=1) if mask is None else series[:, mask].reshape(T, -1).mean(axis=1)
        a_m, p_m, f_m, _, _ = phase_map(mean_sig, min_period, max_period, detrend, window)
        f_m = float(f_m)
        sel = amp if mask is None else amp[mask]
        rows.append({"Label": f"{img.name}:z{z + 1}" if len(zs) > 1 else img.name, "Frames": T, "Dominant period (frames)": (1.0 / f_m) if f_m > 0 else 0.0,
                     **({"Dominant period (s)": (dt / f_m) if f_m > 0 else 0.0, "Frequency (Hz)": f_m / dt} if dt > 0 else {"Frequency (cycles/frame)": f_m}),
                     "Mean-signal amplitude": float(a_m), "Mean-signal phase (°)": float(math.degrees(float(p_m))), "Mean amplitude": float(sel.mean()), "Max amplitude": float(sel.max()),
                     "Median period (frames)": float(np.median(np.where(freq > 0, 1.0 / np.maximum(freq, 1e-12), 0.0)[mask] if mask is not None else np.where(freq > 0, 1.0 / np.maximum(freq, 1e-12), 0.0)))})
    third_name = {"Period (frames)": "Period", "Frequency (cycles/frame)": "Frequency", "Frequency (Hz)": "Frequency", "Period (seconds)": "Period"}[frequency_output]
    res = LumenImage(out, name=f"Phase map {img.name}", axes="TZCYX", channel_names=["Amplitude", f"Phase ({'°' if phase_unit == 'Degrees' else 'rad'})", third_name])
    res.calibration = Calibration(**cal.to_dict())
    res.display.channels[0].lut = "Grays"
    res.display.channels[1].lut = "Spectrum"
    res.display.channels[1].min, res.display.channels[1].max = (-180.0, 180.0) if phase_unit == "Degrees" else (-math.pi, math.pi)
    res.display.channels[2].lut = "Fire"
    res.display.mode = "single"
    res.display.c = 1
    res.metadata.update({"phase_map": True, "min_period": float(min_period), "max_period": float(max_period), "detrend": detrend, "window": "Hann" if window else "none",
                         "phase_unit": phase_unit, "third_channel": frequency_output, "frame_interval": dt})
    res.sync_channels()
    spec = spectrum / len(zs)
    f_ok = freqs > 0
    x = (freqs[f_ok] / dt).tolist() if dt > 0 else freqs[f_ok].tolist()
    plot = {"title": f"Mean power spectrum of {img.name}", "x": x, "series": [{"name": "Power", "values": spec[f_ok].tolist()}],
            "x_label": "Frequency (Hz)" if dt > 0 else "Frequency (cycles/frame)", "y_label": "Power"}
    r0 = rows[0]
    msg = (f"Phase map over {T} frames: dominant period of the mean signal {r0['Dominant period (frames)']:.2f} frames"
           + (f" ({r0['Dominant period (s)']:.3g} s, {r0['Frequency (Hz)']:.4g} Hz)" if dt > 0 else "") + f"; mean amplitude {r0['Mean amplitude']:.3g}")
    return OpResult(new_images=[res], results=rows, plot=plot, message=msg, log=msg)
