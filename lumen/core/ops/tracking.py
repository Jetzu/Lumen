"""Plugins ▸ Tracking — a TrackMate-style workflow.

Detect spots on every frame (LoG / DoG / local maxima with quality values), link them
across frames with the Jaqaman LAP (frame-to-frame linking, then gap closing on track
segments), measure track statistics, draw the tracks as overlays, export CSV, build
kymographs.

Spot coordinates follow the rest of Lumen: ``x``/``y`` are continuous pixel
coordinates where pixel ``i`` spans ``[i, i + 1)`` (so a maximum at pixel index ``i``
sits at ``i + 0.5``). Calibrated coordinates are ``x * pixel_width``. Frames are
0-based (TrackMate convention).

TrackMate puts the centre of pixel ``i`` at ``i * pixel_width`` (its detectors emit
index × calibration and its SpotOverlay adds half a pixel when drawing), so the TrackMate
XML / CSV exporters write ``(x - 0.5) * pixel_width`` and the XML importer reads
``POSITION_X / pixel_width + 0.5`` (see ``_tm_xy``); ``z`` is an integer slice index in
both tools and is not shifted.

Track model (``session.settings``): ``spots`` is the detection (one dict per spot with a stable
``id``), ``tracks`` the list of track dicts ``{id, name, color, spots, edges, stats}`` of the image
``tracks_image`` (other images' tracks are parked in ``tracks_archive``).  ``edges`` are the lineage
links ``[source spot id, target spot id]`` – a track is a tree (or DAG) with splits (a spot with two
successors) and merges (two predecessors), like TrackMate's.  A track without an ``edges`` key is
the chain of its spots in frame order.  Every track is drawn as one overlay item per linear branch
(``track-3``, ``track-3-b1`` …, all carrying ``track_id``), so the viewer's polylines show lineages.
Manual edits (``tracking_extra``) snapshot the model in ``tracks_undo`` first.
"""
from __future__ import annotations

import csv
import math
import re
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max

from .. import roi as roimod
from ..image import Calibration, LumenImage
from ..profile import line_profile
from ..registry import Param, OpResult, op

CAT = "Plugins/Tracking"
DETECTORS = ["LoG", "DoG", "Find maxima"]
UNITS = ["pixels", "calibrated"]
Z_MODES = ["Current slice", "Max projection"]
TRACKERS = ["Simple LAP tracker", "LAP tracker", "Nearest neighbour"]
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4",
           "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff"]

#: spot features: display name → key in the spot dict (TrackMate's spot features, one channel)
SPOT_FEATURES: dict[str, str] = {
    "Quality": "quality", "Mean intensity": "intensity", "Median intensity": "median", "Min intensity": "min",
    "Max intensity": "max", "Total intensity": "total", "Std intensity": "std", "Contrast": "contrast", "SNR": "snr",
    "Radius": "r", "X": "x", "Y": "y", "Z": "z", "Frame": "t", "Visibility": "visibility",
}
#: TrackMate feature keys → spot dict keys (accepted in penalty strings, read from / written to TrackMate XML)
TM_SPOT_KEYS: dict[str, str] = {
    "QUALITY": "quality", "MEAN_INTENSITY_CH1": "intensity", "MEAN_INTENSITY": "intensity", "MEDIAN_INTENSITY_CH1": "median",
    "MEDIAN_INTENSITY": "median", "MIN_INTENSITY_CH1": "min", "MIN_INTENSITY": "min", "MAX_INTENSITY_CH1": "max", "MAX_INTENSITY": "max",
    "TOTAL_INTENSITY_CH1": "total", "TOTAL_INTENSITY": "total", "STD_INTENSITY_CH1": "std", "STD_INTENSITY": "std", "STANDARD_DEVIATION": "std",
    "CONTRAST_CH1": "contrast", "CONTRAST": "contrast", "SNR_CH1": "snr", "SNR": "snr", "RADIUS": "r", "POSITION_X": "x", "POSITION_Y": "y",
    "POSITION_Z": "z", "FRAME": "t", "VISIBILITY": "visibility",
}
#: track features (keys of track["stats"]) that Filter Tracks / Set Display accept
TRACK_FEATURES = ["N spots", "Start frame", "End frame", "Duration (frames)", "Net displacement", "Total distance", "Mean speed", "Max speed",
                  "Min speed", "Median speed", "Mean intensity", "Mean quality", "Confinement ratio", "N gaps", "Longest gap", "N splits", "N merges"]
PENALTY_HELP = ("Feature penalties 'feature=weight, …' (e.g. 'Quality=1, Mean intensity=0.5'): the link cost d² is multiplied by "
                "(1 + 1.5·Σ weight·|a−b|/mean(a,b))², like TrackMate. Features: " + ", ".join(SPOT_FEATURES))

DETECT_PARAMS = [
    Param("detector", "choice", "LoG", choices=DETECTORS, help="LoG: Laplacian of Gaussian (TrackMate default). DoG: faster approximation. Find maxima: smoothed local maxima."),
    Param("radius", "float", 2.5, min=0, step=0.1, label="Spot radius", help="Expected spot radius (in the chosen units)"),
    Param("threshold", "float", 0.0, min=0, step=1, label="Quality threshold", help="Minimum detector response (intensity units). 0 = automatic"),
    Param("subpixel", "bool", True, label="Sub-pixel localization"),
    Param("median_filter", "bool", False, label="Median filter (3×3) first"),
    Param("units", "choice", "pixels", choices=UNITS, help="Units of radius / distances"),
    Param("z_mode", "choice", "Current slice", choices=Z_MODES, help="For Z stacks: detect on the current slice or on a per-frame max projection", advanced=True),
]
LINK_PARAMS = [
    Param("tracker", "choice", "Simple LAP tracker", choices=TRACKERS,
          help="Simple LAP: frame-to-frame LAP linking + gap closing. LAP: adds track splitting and merging (lineage trees). Nearest neighbour: greedy closest-spot linking (each target once), no gap closing"),
    Param("max_displacement", "float", 10.0, min=0, step=1, label="Max linking distance", help="Largest frame-to-frame displacement (in the chosen units)"),
    Param("gap_closing_frames", "int", 2, min=0, max=100, label="Max frame gap", help="Close gaps of up to N missing frames (0 = no gap closing)"),
    Param("max_gap_distance", "float", 15.0, min=0, step=1, label="Max gap-closing distance"),
    Param("allow_splitting", "bool", False, label="Allow track splitting", help="LAP tracker: a track may divide (cell division) – a segment starting next to a spot of another segment in the previous frame is attached to it"),
    Param("split_max_distance", "float", 15.0, min=0, step=1, label="Max splitting distance"),
    Param("allow_merging", "bool", False, label="Allow track merging", help="LAP tracker: two tracks may fuse – a segment ending next to a spot of another segment in the next frame is attached to it"),
    Param("merge_max_distance", "float", 15.0, min=0, step=1, label="Max merging distance"),
    Param("min_track_length", "int", 2, min=1, label="Min spots per track"),
    Param("alternative_cost_factor", "float", 1.05, min=1.0, max=10.0, step=0.05, label="Alternative linking cost factor",
          help="Cost of leaving a spot or segment unlinked = factor × the largest candidate cost of that step (TrackMate's 1.05)", advanced=True),
    Param("linking_penalties", "string", "", label="Linking feature penalties", help=PENALTY_HELP, advanced=True),
    Param("gap_penalties", "string", "", label="Gap-closing feature penalties", help=PENALTY_HELP, advanced=True),
    Param("split_penalties", "string", "", label="Splitting feature penalties", help=PENALTY_HELP, advanced=True),
    Param("merge_penalties", "string", "", label="Merging feature penalties", help=PENALTY_HELP, advanced=True),
]
LINK_KEYS = [p.name for p in LINK_PARAMS]


# ============================================================================ helpers
def _cal_units(img: LumenImage, units: str) -> tuple[float, float, bool]:
    """(pixel_width, pixel_height, calibrated?) for distance computations."""
    cal = img.calibration
    if units == "calibrated" and cal.scaled:
        return float(cal.pixel_width), float(cal.pixel_height), True
    return 1.0, 1.0, False


def _radius_px(img: LumenImage, radius: float, units: str) -> float:
    pw, _, scaled = _cal_units(img, units)
    return float(radius / pw) if scaled else float(radius)


def _response(plane: np.ndarray, detector: str, radius_px: float, median_filter: bool) -> np.ndarray:
    """Detector response (float32, bright blobs positive, ~intensity units)."""
    p = plane.astype(np.float32, copy=True)
    if p.dtype == np.float32 and not np.all(np.isfinite(p)):
        p = np.nan_to_num(p)
    if median_filter:
        p = ndi.median_filter(p, size=3)
    sigma = max(radius_px / math.sqrt(2.0), 0.5)
    if detector == "LoG":
        # scale-normalised LoG; a Gaussian spot of amplitude A gives ~A/2 at the matched scale
        return (-(sigma ** 2) * ndi.gaussian_laplace(p, sigma)).astype(np.float32)
    if detector == "DoG":
        k = 1.6
        g1 = ndi.gaussian_filter(p, sigma)
        g2 = ndi.gaussian_filter(p, sigma * k)
        return ((g1 - g2) / (k - 1.0)).astype(np.float32)
    if detector == "Find maxima":
        sm = ndi.gaussian_filter(p, max(sigma / 2.0, 0.5))
        return (sm - float(np.median(sm))).astype(np.float32)
    raise ValueError(f"Unknown detector {detector!r}")


def _refine(resp: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Quadratic sub-pixel refinement of (row, col) integer peaks → float (row, col)."""
    out = peaks.astype(np.float64)
    if len(peaks) == 0:
        return out
    h, w = resp.shape
    yy, xx = peaks[:, 0], peaks[:, 1]
    inside = (yy > 0) & (yy < h - 1) & (xx > 0) & (xx < w - 1)
    y, x = yy[inside], xx[inside]
    c = resp[y, x].astype(np.float64)
    for axis, (lo, hi) in ((1, (resp[y, x - 1], resp[y, x + 1])), (0, (resp[y - 1, x], resp[y + 1, x]))):
        lo = lo.astype(np.float64)
        hi = hi.astype(np.float64)
        denom = 2.0 * (2.0 * c - lo - hi)
        delta = np.where(denom > 1e-12, (hi - lo) / np.where(denom > 1e-12, denom, 1.0), 0.0)
        out[inside, axis] += np.clip(delta, -1.0, 1.0)
    return out


def _disk_means(plane: np.ndarray, xs: np.ndarray, ys: np.ndarray, r: float) -> np.ndarray:
    """Mean of ``plane`` inside a disk of radius r (px) around each (x, y) pixel coordinate."""
    ri = max(int(math.ceil(r)), 1)
    gy, gx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    disk = (gx ** 2 + gy ** 2) <= max(r, 1.0) ** 2 + 1e-6
    h, w = plane.shape
    out = np.full(len(xs), np.nan)
    for k in range(len(xs)):
        xi, yi = int(math.floor(xs[k])), int(math.floor(ys[k]))
        y0, y1 = max(yi - ri, 0), min(yi + ri + 1, h)
        x0, x1 = max(xi - ri, 0), min(xi + ri + 1, w)
        if y1 <= y0 or x1 <= x0:
            continue
        m = disk[y0 - (yi - ri):y1 - (yi - ri), x0 - (xi - ri):x1 - (xi - ri)]
        sub = plane[y0:y1, x0:x1]
        if m.any():
            out[k] = float(sub[m].mean())
    return out


INTENSITY_KEYS = ("intensity", "median", "min", "max", "total", "std", "contrast", "snr")


def _disk_stats(plane: np.ndarray, xs: np.ndarray, ys: np.ndarray, r: float) -> dict[str, np.ndarray]:
    """TrackMate's spot intensity features inside a disk of radius r (px) around each (x, y) pixel coordinate:
    mean (``intensity``), median, min, max, total, std, plus ``contrast`` = (in − out) / (in + out) and
    ``snr`` = (in − out) / std_in, where 'out' is the ring between r and 2r."""
    r = max(float(r), 1.0)
    ri = int(math.ceil(2.0 * r))
    gy, gx = np.mgrid[-ri:ri + 1, -ri:ri + 1]
    d2 = gx ** 2 + gy ** 2
    disk = d2 <= r ** 2 + 1e-6
    ring = (~disk) & (d2 <= 4.0 * r ** 2 + 1e-6)
    h, w = plane.shape
    n = len(xs)
    out = {k: np.full(n, np.nan) for k in INTENSITY_KEYS}
    for k in range(n):
        xi, yi = int(math.floor(xs[k])), int(math.floor(ys[k]))
        y0, y1 = max(yi - ri, 0), min(yi + ri + 1, h)
        x0, x1 = max(xi - ri, 0), min(xi + ri + 1, w)
        if y1 <= y0 or x1 <= x0:
            continue
        sl = (slice(y0 - (yi - ri), y1 - (yi - ri)), slice(x0 - (xi - ri), x1 - (xi - ri)))
        sub = plane[y0:y1, x0:x1]
        inside = sub[disk[sl]]
        if not inside.size:
            continue
        inside = inside.astype(np.float64)
        m = float(inside.mean())
        sd = float(inside.std())
        out["intensity"][k] = m
        out["median"][k] = float(np.median(inside))
        out["min"][k] = float(inside.min())
        out["max"][k] = float(inside.max())
        out["total"][k] = float(inside.sum())
        out["std"][k] = sd
        outside = sub[ring[sl]]
        if outside.size:
            mo = float(outside.astype(np.float64).mean())
            out["contrast"][k] = (m - mo) / (m + mo) if (m + mo) != 0 else 0.0
            out["snr"][k] = (m - mo) / sd if sd > 0 else 0.0
    return out


def _spot_feature_key(name: str) -> str:
    """The spot dict key of a feature given by display name ('Mean intensity'), TrackMate key ('MEAN_INTENSITY_CH1') or dict key."""
    n = str(name).strip()
    if n in SPOT_FEATURES:
        return SPOT_FEATURES[n]
    up = n.upper().replace(" ", "_")
    if up in TM_SPOT_KEYS:
        return TM_SPOT_KEYS[up]
    low = n.lower()
    for disp, key in SPOT_FEATURES.items():
        if disp.lower() == low or key == low:
            return key
    raise ValueError(f"Unknown spot feature {name!r}; one of {', '.join(SPOT_FEATURES)}")


def _spot_value(sp: dict[str, Any], key: str) -> float:
    """A spot feature as float (NaN when missing); 'visibility' defaults to 1."""
    v = sp.get(key, 1 if key == "visibility" else None)
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _parse_penalties(text: Any) -> dict[str, float]:
    """'Quality=1, Mean intensity=0.5' → {'quality': 1.0, 'intensity': 0.5}."""
    out: dict[str, float] = {}
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part and ":" not in part:
            raise ValueError(f"Feature penalty {part!r}: expected 'feature=weight'")
        name, _, w = part.replace(":", "=").partition("=")
        key = _spot_feature_key(name)
        try:
            weight = float(w)
        except ValueError:
            raise ValueError(f"Feature penalty {part!r}: the weight is not a number") from None
        if weight < 0:
            raise ValueError(f"Feature penalty {part!r}: the weight must be ≥ 0")
        out[key] = weight
    return out


def _penalty_factor(ia: np.ndarray, ib: np.ndarray, penalties: dict[str, float], feats: dict[str, np.ndarray]) -> np.ndarray:
    """TrackMate's link penalty 1 + 1.5·Σ w·|a − b| / mean(a, b) for the candidate pairs (ia[k], ib[k])."""
    p = np.ones(len(ia))
    if not penalties or len(ia) == 0:
        return p
    for key, w in penalties.items():
        f = feats[key]
        a, b = f[ia], f[ib]
        mean = (a + b) / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            nd = np.where(mean != 0, np.abs(a - b) / np.abs(mean), 0.0)
        p += 1.5 * w * np.where(np.isfinite(nd), nd, 0.0)
    return p


def _impulse_gain(fn) -> float:
    """L2 norm of a linear filter's impulse response (= std of its output on unit white noise)."""
    d = np.zeros((65, 65), np.float32)
    d[32, 32] = 1.0
    return float(np.sqrt((fn(d).astype(np.float64) ** 2).sum()))


_RESID_GAIN = _impulse_gain(lambda d: d - ndi.gaussian_filter(d, 1.0))
NOISE_K = 5.0  # a response maximum below NOISE_K sigma of the response noise is indistinguishable from noise


def _noise_sigma(p: np.ndarray) -> float:
    """Pixel noise of a float plane: MAD of the 1-px high-pass residual on background pixels
    (smoothed value at or below the median), so that dense spots do not inflate it."""
    sm = ndi.gaussian_filter(p, 1.0)
    res = (p - sm)[sm <= np.median(sm)]
    if res.size < 16:
        return 0.0
    return 1.4826 * float(np.median(np.abs(res - np.median(res)))) / _RESID_GAIN


def _noise_floor(plane: np.ndarray, gain: float, median_filter: bool) -> float:
    """Quality below which a local maximum of the response is indistinguishable from noise;
    ``gain`` is the detector's impulse gain (see ``_impulse_gain``)."""
    p = plane.astype(np.float32, copy=True)
    if not np.all(np.isfinite(p)):
        p = np.nan_to_num(p)
    if median_filter:
        p = ndi.median_filter(p, size=3)
    return NOISE_K * _noise_sigma(p) * gain


def _auto_threshold(qualities: np.ndarray, noise_floor: float = 0.0) -> float:
    """Automatic quality threshold on the pooled candidate responses.

    Exact (un-binned) Otsu split of the sorted qualities with the threshold placed in the
    middle of the gap between the two classes, never below ``noise_floor`` (NOISE_K sigma of
    the detector response on the image's noise, so that an image without real spots does not
    pass noise maxima). The floor is estimated from the image, not from the candidates: a
    statistic of the candidate maxima would be dominated by the spots themselves on dense
    or clean images and reject them all. When the lower Otsu class is itself well above the
    floor the candidates contain no noise class and Otsu would merely bisect the real spots;
    the noise floor is then the threshold.
    """
    q = np.sort(qualities[np.isfinite(qualities)]).astype(np.float64)
    noise_floor = max(float(noise_floor), 0.0)
    if q.size < 3 or q[-1] <= q[0]:
        return noise_floor
    if q.size > 200_000:
        q = q[:: int(math.ceil(q.size / 200_000))]
    n = q.size
    k = np.arange(1, n)  # split: first k values are the lower class
    cs = np.cumsum(q)[:-1]
    w0 = k / n
    w1 = 1.0 - w0
    mu0 = cs / k
    mu1 = (cs[-1] + q[-1] - cs) / (n - k)
    between = w0 * w1 * (mu0 - mu1) ** 2
    best = int(np.argmax(between))
    if mu0[best] > max(noise_floor, 0.1 * mu1[best]):
        return noise_floor
    otsu = 0.5 * (q[best] + q[best + 1])
    return max(float(otsu), noise_floor)


def _detect(ctx, detector: str, radius: float, threshold: float, subpixel: bool, median_filter: bool, units: str, z_mode: str) -> tuple[list[dict[str, Any]], float, float]:
    """Run the detector on every frame of the current channel. Returns (spots, radius_px, threshold_used)."""
    img = ctx.img
    d = img.display
    c = d.c
    r_px = _radius_px(img, radius, units)
    if r_px <= 0:
        raise ValueError("Spot radius must be positive")
    use_proj = z_mode == "Max projection" and img.nz > 1
    mask = ctx.mask
    md = max(1, int(round(r_px)))
    thr_fixed = float(threshold) if threshold > 0 else None
    gain = _impulse_gain(lambda p: _response(p, detector, r_px, False)) if thr_fixed is None else 0.0
    # One frame at a time, keeping only per-candidate data: the response of every frame would cost
    # 4*T*H*W bytes (and the projection planes another T*H*W*itemsize) until the op finishes.
    # Refinement and the disk mean are per-peak operations, so running them on all candidates
    # before the pooled threshold gives the same spots.
    frames: list[dict[str, Any]] = []
    floors: list[float] = []
    for t in range(img.nt):
        ctx.check_cancel()
        ctx.progress(t / img.nt, f"Detecting spots in frame {t + 1}/{img.nt}")
        if use_proj:
            vol = img.data[t, :, c]
            plane = vol.max(axis=0)
        else:
            vol = None
            plane = img.data[t, d.z, c]
        resp = _response(plane, detector, r_px, median_filter)
        if thr_fixed is None:
            floors.append(_noise_floor(plane, gain, median_filter))
        if mask is not None:
            resp = np.where(mask, resp, 0.0).astype(np.float32)
        peaks = peak_local_max(resp, min_distance=md, threshold_abs=1e-6, exclude_border=False, num_peaks=50000)
        quality = resp[peaks[:, 0], peaks[:, 1]].astype(np.float64) if len(peaks) else np.zeros(0)
        if thr_fixed is not None:  # explicit threshold: filter now, nothing to pool
            keep = quality >= thr_fixed
            peaks, quality = peaks[keep], quality[keep]
        if len(peaks) == 0:
            continue
        pos = _refine(resp, peaks) if subpixel else peaks.astype(np.float64)
        ys = pos[:, 0] + 0.5
        xs = pos[:, 1] + 0.5
        zs = vol[:, peaks[:, 0], peaks[:, 1]].argmax(axis=0) if vol is not None else np.full(len(peaks), int(d.z))
        frames.append({"t": t, "quality": quality, "xs": xs, "ys": ys, "zs": zs})
        del resp, plane, vol
    pooled = np.concatenate([f["quality"] for f in frames]) if frames else np.zeros(0)
    thr = thr_fixed if thr_fixed is not None else _auto_threshold(pooled, float(np.median(floors)) if floors else 0.0)
    # Second pass: the intensity features of the spots that passed, measured on the raw plane (TrackMate's spot
    # analyzers); only the kept candidates are measured, and no plane is retained between frames.
    kept = [(f, np.nonzero(f["quality"] >= thr)[0]) for f in frames]
    kept = [(f, k) for f, k in kept if len(k)]
    spots: list[dict[str, Any]] = []
    for n, (f, keep) in enumerate(kept):
        ctx.check_cancel()
        ctx.progress(n / len(kept), f"Measuring spots in frame {f['t'] + 1}/{img.nt}")
        t = int(f["t"])
        plane = img.data[t, :, c].max(axis=0) if use_proj else img.data[t, d.z, c]
        xs, ys = f["xs"][keep], f["ys"][keep]
        st = _disk_stats(plane.astype(np.float32, copy=False), xs, ys, r_px)
        for i, k in enumerate(keep):
            sp: dict[str, Any] = {"id": len(spots), "t": t, "x": float(xs[i]), "y": float(ys[i]), "z": int(f["zs"][k]), "r": float(r_px),
                                  "quality": float(f["quality"][k])}
            for key in INTENSITY_KEYS:
                sp[key] = float(st[key][i])
            spots.append(sp)
        del plane
    return spots, r_px, thr


def _spot_rows(img: LumenImage, spots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cal = img.calibration
    rows = []
    for i, s in enumerate(spots):
        rows.append({"Label": f"{img.name}:spot{i}", "Spot ID": int(s.get("id", i)), "Frame": s["t"], "X": s["x"] * cal.pixel_width, "Y": s["y"] * cal.pixel_height,
                     "Z": s["z"], "X (px)": s["x"], "Y (px)": s["y"], "Radius": s["r"] * cal.pixel_width, "Quality": s["quality"],
                     "Mean intensity": s.get("intensity", float("nan")), "Median intensity": s.get("median", float("nan")),
                     "Std intensity": s.get("std", float("nan")), "Contrast": s.get("contrast", float("nan")), "SNR": s.get("snr", float("nan"))})
    return rows


def _ensure_ids(*groups: "list[dict[str, Any]] | None") -> None:
    """Give every spot of the given lists a unique integer ``id`` (spots without one get the next free number)."""
    used: set[int] = set()
    missing: list[dict[str, Any]] = []
    for g in groups:
        for sp in g or []:
            sid = sp.get("id")
            if isinstance(sid, (int, np.integer)) and int(sid) not in used:
                used.add(int(sid))
            elif sid is None or not isinstance(sid, (int, np.integer)):
                missing.append(sp)
    nxt = (max(used) + 1) if used else 0
    for sp in missing:
        sp["id"] = nxt
        used.add(nxt)
        nxt += 1


def _next_spot_id(*groups: "list[dict[str, Any]] | None") -> int:
    ids = [int(sp["id"]) for g in groups for sp in (g or []) if isinstance(sp.get("id"), (int, np.integer))]
    return (max(ids) + 1) if ids else 0


def _spot_rois(img: LumenImage, spots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One point ROI per frame (for ``rois_add``), each carrying its frame position."""
    by_t: dict[int, list[list[float]]] = {}
    zs: dict[int, int] = {}
    for sp in spots:
        by_t.setdefault(sp["t"], []).append([sp["x"], sp["y"]])
        zs.setdefault(sp["t"], sp["z"])
    return [{"type": "point", "points": by_t[t], "name": f"Spots t{t}", "color": "#ffd700",
             "position": {"t": int(t), "z": int(zs[t]), "c": int(img.display.c)}} for t in sorted(by_t)]


# ------------------------------------------------------------------------ LAP linking
def _candidates(pa: np.ndarray, pb: np.ndarray, max_dist: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, distance) for every pair with |pa[i] - pb[j]| <= max_dist (inclusive; zero-distance pairs kept)."""
    if len(pa) == 0 or len(pb) == 0 or max_dist <= 0:
        return np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0)
    rec = cKDTree(pa).sparse_distance_matrix(cKDTree(pb), max_dist, output_type="ndarray")
    return rec["i"].astype(np.intp), rec["j"].astype(np.intp), rec["v"].astype(np.float64)


def _lap(n: int, m: int, ii: np.ndarray, jj: np.ndarray, cost: np.ndarray, alt: float) -> list[tuple[int, int]]:
    """Solve the augmented linear assignment problem of Jaqaman et al. (2008) on a sparse candidate list.

    ``ii[k], jj[k]`` are the allowed (row, col) pairs with cost ``cost[k]`` (every other pair is
    forbidden); ``alt`` is the cost of leaving a row/column unassigned. Returns the (i, j) pairs
    that were linked. Memory is O(n + m + len(ii)) instead of the (n + m)^2 of the dense matrix,
    which is what makes gap closing over tens of thousands of segments possible.
    """
    if n == 0 or m == 0 or len(ii) == 0:
        return []
    cost = np.asarray(cost, dtype=np.float64)
    eps = float(cost.min())
    # min_weight_full_bipartite_matching silently drops explicit zero weights, and zero costs do occur
    # (coincident spots, eps == 0): shift every stored weight to be strictly positive. Every perfect
    # matching has exactly n + m edges, so a uniform offset does not change the optimum.
    off = 1.0 - min(eps, alt, 0.0)
    r = np.concatenate([ii, np.arange(n), n + np.arange(m), n + jj])
    c = np.concatenate([jj, m + np.arange(n), np.arange(m), m + ii])
    v = np.concatenate([cost + off, np.full(n, alt + off), np.full(m, alt + off), np.full(len(ii), eps + off)])
    a = coo_matrix((v, (r, c)), shape=(n + m, n + m)).tocsr()  # candidate pairs are unique: no summed duplicates
    rows, cols = min_weight_full_bipartite_matching(a)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if i < n and j < m]


def _alt_cost(cost: np.ndarray, factor: float, scale: float) -> float:
    """The cost of leaving a row/column unassigned: ``factor`` × the largest candidate cost (TrackMate's alternative
    cost), floored so that a matrix of (near-)zero costs still prefers linking over not linking."""
    return float(factor) * max(float(cost.max()) if len(cost) else 0.0, 1e-3 * float(scale) ** 2, 1e-12)


def _frame_links(xy: np.ndarray, ts: np.ndarray, frames: list[int], idx_by_t: dict[int, np.ndarray], max_disp: float, penalties: dict[str, float],
                 feats: dict[str, np.ndarray], alt_factor: float, greedy: bool, ctx) -> list[tuple[int, int]]:
    """Frame-to-frame links (spot index pairs) between consecutive frames: the Jaqaman LAP, or the greedy
    nearest-neighbour assignment (closest pairs first, every spot used at most once)."""
    edges: list[tuple[int, int]] = []
    if max_disp <= 0:
        return edges
    for k, (ta, tb) in enumerate(zip(frames[:-1], frames[1:])):
        if ctx is not None:
            ctx.check_cancel()
            ctx.progress(k / max(len(frames) - 1, 1), f"Linking frame {tb + 1}/{frames[-1] + 1}")
        if tb - ta != 1:
            continue
        ia, ib = idx_by_t[ta], idx_by_t[tb]
        i, j, dist = _candidates(xy[ia], xy[ib], max_disp)
        if len(i) == 0:
            continue
        cost = dist ** 2 * _penalty_factor(ia[i], ib[j], penalties, feats) ** 2
        if greedy:
            used_a: set[int] = set()
            used_b: set[int] = set()
            for q in np.argsort(cost, kind="stable"):
                a, b = int(i[q]), int(j[q])
                if a in used_a or b in used_b:
                    continue
                used_a.add(a)
                used_b.add(b)
                edges.append((int(ia[a]), int(ib[b])))
        else:
            for a, b in _lap(len(ia), len(ib), i, j, cost, _alt_cost(cost, alt_factor, max_disp)):
                edges.append((int(ia[a]), int(ib[b])))
    return edges


def _chains(n: int, edges: list[tuple[int, int]], ts: np.ndarray) -> list[list[int]]:
    """The segments made by frame-to-frame links (each spot has at most one predecessor and one successor)."""
    nxt = np.full(n, -1, dtype=int)
    prv = np.full(n, -1, dtype=int)
    for a, b in edges:
        nxt[a] = b
        prv[b] = a
    segments: list[list[int]] = []
    for s in np.argsort(ts, kind="stable"):
        if prv[s] != -1:
            continue
        chain = [int(s)]
        cur = int(s)
        while nxt[cur] != -1:
            cur = int(nxt[cur])
            chain.append(cur)
        segments.append(chain)
    return segments


def _segment_links(segments: list[list[int]], xy: np.ndarray, ts: np.ndarray, feats: dict[str, np.ndarray], gap_frames: int, max_gap: float,
                   allow_split: bool, split_max: float, allow_merge: bool, merge_max: float, alt_factor: float,
                   penalties: dict[str, dict[str, float]]) -> list[tuple[int, int]]:
    """TrackMate's second LAP step on the track segments (Jaqaman et al. 2008): gap closing (segment end → segment
    start, 1 … max_frame_gap + 1 frames later), splitting (a spot that is not the last of its segment → a segment
    start in the next frame) and merging (a segment end → a spot that is not the first of its segment, next frame),
    solved together in one sparse assignment problem. Returns the new links as spot index pairs."""
    n_seg = len(segments)
    if n_seg == 0:
        return []
    ends = np.array([s[-1] for s in segments], dtype=np.intp)
    starts = np.array([s[0] for s in segments], dtype=np.intp)
    seg_of = np.full(len(ts), -1, dtype=int)
    for k, s in enumerate(segments):
        seg_of[s] = k
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    costs: list[np.ndarray] = []
    if gap_frames > 0 and max_gap > 0 and n_seg > 1:
        i, j, dist = _candidates(xy[ends], xy[starts], max_gap)  # (end i, start j) within reach
        dt = ts[starts[j]] - ts[ends[i]]
        keep = (dt >= 1) & (dt <= gap_frames + 1) & (i != j)
        i, j, dist = i[keep], j[keep], dist[keep]
        rows.append(i)
        cols.append(j)
        costs.append(dist ** 2 * _penalty_factor(ends[i], starts[j], penalties.get("gap", {}), feats) ** 2)
    split_mid = np.array([i for s in segments for i in s[:-1]], dtype=np.intp) if allow_split and split_max > 0 and n_seg > 1 else np.zeros(0, np.intp)
    if len(split_mid):
        i, j, dist = _candidates(xy[split_mid], xy[starts], split_max)
        keep = (ts[starts[j]] - ts[split_mid[i]] == 1) & (seg_of[split_mid[i]] != j)
        i, j, dist = i[keep], j[keep], dist[keep]
        rows.append(n_seg + i)
        cols.append(j)
        costs.append(dist ** 2 * _penalty_factor(split_mid[i], starts[j], penalties.get("split", {}), feats) ** 2)
    merge_mid = np.array([i for s in segments for i in s[1:]], dtype=np.intp) if allow_merge and merge_max > 0 and n_seg > 1 else np.zeros(0, np.intp)
    if len(merge_mid):
        i, j, dist = _candidates(xy[ends], xy[merge_mid], merge_max)
        keep = (ts[merge_mid[j]] - ts[ends[i]] == 1) & (seg_of[merge_mid[j]] != i)
        i, j, dist = i[keep], j[keep], dist[keep]
        rows.append(i)
        cols.append(n_seg + j)
        costs.append(dist ** 2 * _penalty_factor(ends[i], merge_mid[j], penalties.get("merge", {}), feats) ** 2)
    if not rows:
        return []
    ii = np.concatenate(rows).astype(np.intp)
    jj = np.concatenate(cols).astype(np.intp)
    cost = np.concatenate(costs).astype(np.float64)
    if len(ii) == 0:
        return []
    alt = _alt_cost(cost, alt_factor, max(max_gap, split_max if allow_split else 0.0, merge_max if allow_merge else 0.0))
    out: list[tuple[int, int]] = []
    for r, c in _lap(n_seg + len(split_mid), n_seg + len(merge_mid), ii, jj, cost, alt):
        src = ends[r] if r < n_seg else split_mid[r - n_seg]
        dst = starts[c] if c < n_seg else merge_mid[c - n_seg]
        out.append((int(src), int(dst)))
    return out


def _components(n: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Connected-component label of every spot index (union–find)."""
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for a, b in edges:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb
    return np.array([find(i) for i in range(n)], dtype=int)


def _link_graph(spots: list[dict[str, Any]], *, tracker: str = "Simple LAP tracker", max_disp: float = 10.0, gap_frames: int = 2, max_gap: float = 15.0,
                allow_split: bool = False, split_max: float = 15.0, allow_merge: bool = False, merge_max: float = 15.0, alt_factor: float = 1.05,
                penalties: dict[str, dict[str, float]] | None = None, min_len: int = 2, pw: float = 1.0, ph: float = 1.0,
                ctx=None) -> list[tuple[list[int], list[tuple[int, int]]]]:
    """Link spots into tracks with lineage. Returns one ``(spot indices sorted by frame, edges)`` per track, where the
    edges are (source, target) spot index pairs; tracks are the connected components of the link graph.

    ``tracker``: 'Simple LAP tracker' (frame-to-frame LAP + gap closing), 'LAP tracker' (+ splitting / merging when
    allowed) or 'Nearest neighbour' (greedy, no second step). ``penalties`` maps the step ('link', 'gap', 'split',
    'merge') to its feature penalties. ``ctx`` (an OpContext) is optional: with it, linking reports progress per frame
    and honours cancellation."""
    if not spots:
        return []
    penalties = penalties or {}
    if tracker not in TRACKERS:
        raise ValueError(f"Unknown tracker {tracker!r}; one of {TRACKERS}")
    n = len(spots)
    ts = np.array([s["t"] for s in spots], dtype=int)
    xy = np.array([[s["x"] * pw, s["y"] * ph] for s in spots], dtype=np.float64)
    keys = set().union(*(set(p) for p in penalties.values())) if penalties else set()
    feats = {k: np.array([_spot_value(s, k) for s in spots], dtype=np.float64) for k in keys}
    frames = sorted(set(ts.tolist()))
    idx_by_t = {t: np.nonzero(ts == t)[0] for t in frames}
    edges = _frame_links(xy, ts, frames, idx_by_t, max_disp, penalties.get("link", {}), feats, alt_factor, tracker == "Nearest neighbour", ctx)
    if tracker != "Nearest neighbour":
        lap = tracker == "LAP tracker"
        if (gap_frames > 0 and max_gap > 0) or (lap and (allow_split or allow_merge)):
            if ctx is not None:
                ctx.check_cancel()
                ctx.progress(1.0, "Linking track segments (gap closing, splits, merges)")
            segments = _chains(n, edges, ts)
            edges += _segment_links(segments, xy, ts, feats, gap_frames, max_gap, lap and allow_split, split_max, lap and allow_merge, merge_max,
                                    alt_factor, penalties)
    comp = _components(n, edges)
    members: dict[int, list[int]] = defaultdict(list)
    for i in np.argsort(ts, kind="stable"):
        members[int(comp[i])].append(int(i))
    comp_edges: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for a, b in edges:
        comp_edges[int(comp[a])].append((int(a), int(b)))
    tracks = [(idx, sorted(comp_edges.get(c, []), key=lambda e: (ts[e[0]], e[0], ts[e[1]], e[1]))) for c, idx in members.items() if len(idx) >= max(min_len, 1)]
    tracks.sort(key=lambda tr: (ts[tr[0][0]], xy[tr[0][0], 0], xy[tr[0][0], 1]))
    return tracks


def _link(spots: list[dict[str, Any]], max_disp: float, gap_frames: int, max_gap: float, min_len: int, pw: float = 1.0, ph: float = 1.0,
          ctx=None) -> list[list[int]]:
    """Link spots into tracks with the simple LAP tracker. Returns a list of tracks, each a list of spot indices sorted by frame."""
    return [idx for idx, _ in _link_graph(spots, max_disp=max_disp, gap_frames=gap_frames, max_gap=max_gap, min_len=min_len, pw=pw, ph=ph, ctx=ctx)]


def _track_stats(img: LumenImage, spots: list[dict[str, Any]], track: list[int], edges: list[tuple[int, int]] | None = None) -> dict[str, Any]:
    """Track features of the spots ``spots[i] for i in track``; ``edges`` are (source, target) index pairs of the
    lineage (None = the chain in frame order). Speeds / gaps are measured along the edges, splits and merges are the
    spots with several successors / predecessors."""
    cal = img.calibration
    pw, ph = float(cal.pixel_width), float(cal.pixel_height)
    dt_unit = float(cal.frame_interval) if cal.frame_interval and cal.frame_interval > 0 else 0.0
    if not track:
        # the same keys, in the same order, as a populated track: the Tracks table must not depend on which track comes first
        out = {"N spots": 0, "Start frame": -1, "End frame": -1, "Duration (frames)": 0}
        if dt_unit > 0:
            out[f"Duration ({cal.time_unit})"] = 0.0
        out.update({"Net displacement": 0.0, "Total distance": 0.0, "Mean speed": 0.0, "Max speed": 0.0,
                    "Mean intensity": float("nan"), "Confinement ratio": 0.0, "Min speed": 0.0, "Median speed": 0.0, "Mean quality": float("nan"),
                    "N gaps": 0, "Longest gap": 0, "N splits": 0, "N merges": 0})
        return out
    idx = sorted(track, key=lambda i: (spots[i]["t"], i))
    if edges is None:
        edges = list(zip(idx[:-1], idx[1:]))
    pos = {i: (spots[i]["x"] * pw, spots[i]["y"] * ph) for i in idx}
    steps = np.array([math.hypot(pos[b][0] - pos[a][0], pos[b][1] - pos[a][1]) for a, b in edges], dtype=np.float64)
    dts = np.array([spots[b]["t"] - spots[a]["t"] for a, b in edges], dtype=np.float64)
    dts_time = dts * dt_unit if dt_unit > 0 else dts
    speeds = steps / np.where(dts_time > 0, dts_time, 1.0) if len(steps) else np.zeros(0)
    total = float(steps.sum())
    first, last = idx[0], idx[-1]
    net = float(math.hypot(pos[last][0] - pos[first][0], pos[last][1] - pos[first][1])) if len(idx) > 1 else 0.0
    start, end = int(spots[first]["t"]), int(spots[last]["t"])
    out: dict[str, Any] = {"N spots": len(idx), "Start frame": start, "End frame": end, "Duration (frames)": end - start}
    if dt_unit > 0:
        out[f"Duration ({cal.time_unit})"] = (end - start) * dt_unit
    inten = np.array([_spot_value(spots[i], "intensity") for i in idx], dtype=np.float64)
    qual = np.array([_spot_value(spots[i], "quality") for i in idx], dtype=np.float64)
    outdeg = Counter(a for a, _ in edges)
    indeg = Counter(b for _, b in edges)
    out.update({"Net displacement": net, "Total distance": total,
                "Mean speed": float(speeds.mean()) if len(speeds) else 0.0, "Max speed": float(speeds.max()) if len(speeds) else 0.0,
                "Mean intensity": float(inten[np.isfinite(inten)].mean()) if np.isfinite(inten).any() else float("nan"),
                "Confinement ratio": (net / total) if total > 0 else 0.0,
                "Min speed": float(speeds.min()) if len(speeds) else 0.0, "Median speed": float(np.median(speeds)) if len(speeds) else 0.0,
                "Mean quality": float(qual[np.isfinite(qual)].mean()) if np.isfinite(qual).any() else float("nan"),
                "N gaps": int(np.sum(dts > 1)), "Longest gap": int(max(dts.max() - 1, 0)) if len(dts) else 0,
                "N splits": sum(1 for v in outdeg.values() if v > 1), "N merges": sum(1 for v in indeg.values() if v > 1)})
    return out


# ------------------------------------------------------------------------ lineage helpers
def _spot_key(sp: dict[str, Any]) -> tuple[int, int]:
    return int(sp["t"]), int(sp.get("id", 0))


def _track_edges(track: dict[str, Any]) -> list[tuple[int, int]]:
    """The (source id, target id) links of a track: its explicit lineage, else the chain of its spots in frame order."""
    ed = track.get("edges")
    if ed is not None:
        return [(int(a), int(b)) for a, b in ed]
    sp = sorted(track["spots"], key=_spot_key)
    return [(int(a["id"]), int(b["id"])) for a, b in zip(sp[:-1], sp[1:])]


def _branches(track: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """The linear branches of a track's lineage (lists of spot dicts in frame order), for drawing: a split
    continues one branch and starts another one at the mother spot; a merge ends a branch at the merged spot."""
    spots = track["spots"]
    if not spots:
        return []
    by_id = {int(s["id"]): s for s in spots}
    edges = [(a, b) for a, b in _track_edges(track) if a in by_id and b in by_id]
    succ: dict[int, list[int]] = defaultdict(list)
    pred: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        succ[a].append(b)
        pred[b].append(a)
    key = lambda i: _spot_key(by_id[i])  # noqa: E731
    for v in succ.values():
        v.sort(key=key)
    roots = sorted((i for i in by_id if not pred.get(i)), key=key)
    visited: set[int] = set()
    branches: list[list[dict[str, Any]]] = []
    queue: deque[tuple[int, int | None]] = deque((r, None) for r in roots)
    while queue:
        start, prefix = queue.popleft()
        pts = [by_id[prefix]] if prefix is not None else []
        cur = start
        while True:
            if cur in visited:  # merge into a branch walked before: end here so the lines meet
                pts.append(by_id[cur])
                break
            visited.add(cur)
            pts.append(by_id[cur])
            ch = succ.get(cur, [])
            if not ch:
                break
            for extra in ch[1:]:
                queue.append((extra, cur))
            cur = ch[0]
        branches.append(pts)
    return branches


def _speed_unit(img: LumenImage) -> str:
    cal = img.calibration
    du = cal.unit if cal.scaled else "px"
    tu = cal.time_unit if cal.frame_interval and cal.frame_interval > 0 else "frame"
    return f"{du}/{tu}"


def _spots_for(ctx) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s = ctx.session
    spots = s.settings.get("spots")
    if not spots:
        raise ValueError("Run Detect Spots first")
    meta = s.settings.get("spots_meta") or {}
    return spots, meta


def _overlay_id(track: dict[str, Any]) -> str:
    """Id of the overlay item drawing a track ('track-<id>'; manual tracks carry their own)."""
    return str(track.get("overlay_id") or f"track-{track['id']}")


def _item_track_key(o: dict[str, Any]) -> str:
    """The track overlay id ('track-3') that a branch item ('track-3-b1') belongs to."""
    return re.sub(r"-b\d+$", "", str(o.get("id", "")))


def _overlay_without(overlay: list[dict[str, Any]], oid: str) -> list[dict[str, Any]]:
    """The overlay minus every branch item of the track with overlay id ``oid``."""
    return [o for o in overlay if not (o.get("type") == "track" and _item_track_key(o) == oid)]


def _track_overlay_items(track: dict[str, Any]) -> list[dict[str, Any]]:
    """One 'track' overlay item per linear branch of the track (the first keeps the plain id)."""
    base = _overlay_id(track)
    items = []
    for k, br in enumerate(_branches(track)):
        items.append({"type": "track", "id": base if k == 0 else f"{base}-b{k}", "name": track["name"], "color": track["color"], "track_id": str(track["id"]),
                      "points": [[round(float(sp["x"]), 3), round(float(sp["y"]), 3), int(sp["t"])] for sp in br]})
    if not items:  # an empty (manual) track keeps an empty item so it stays listed
        items.append({"type": "track", "id": base, "name": track["name"], "color": track["color"], "track_id": str(track["id"]), "points": []})
    return items


def _all_track_overlay(img: LumenImage, tracks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The image overlay with every track item replaced by the items of ``tracks``."""
    return [o for o in img.overlay if o.get("type") != "track"] + [it for t in tracks for it in _track_overlay_items(t)]


def _refresh_stats(img: LumenImage, track: dict[str, Any]) -> None:
    """Recompute a track's statistics from its spots and lineage edges (spots are sorted by frame first)."""
    track["spots"].sort(key=_spot_key)
    sp = track["spots"]
    index_of = {int(s["id"]): k for k, s in enumerate(sp)}
    edges = [(index_of[a], index_of[b]) for a, b in _track_edges(track) if a in index_of and b in index_of]
    track["stats"] = _track_stats(img, sp, list(range(len(sp))), edges)


UNDO_DEPTH = 25


def _copy_track(t: dict[str, Any]) -> dict[str, Any]:
    out = dict(t)
    out["spots"] = list(t.get("spots") or [])
    if t.get("edges") is not None:
        out["edges"] = [list(e) for e in t["edges"]]
    out["stats"] = dict(t.get("stats") or {})
    return out


def _push_undo(session, img: LumenImage, tracks: list[dict[str, Any]] | None, label: str) -> None:
    """Snapshot the track model of ``img`` for ``tracking.undo_edit``. Lists are copied shallowly: the edit ops
    never change a spot dict in place (they replace or remove spots), so shared spot dicts stay valid.
    ``tracks`` is None when the image has no track list yet (a spot filter before linking)."""
    s = session.settings
    stack = s.setdefault("tracks_undo", [])
    spots = s.get("spots")
    stack.append({"image": img.id, "label": label, "tracks": [_copy_track(t) for t in tracks] if tracks is not None else None,
                  "spots": list(spots) if isinstance(spots, list) else None, "spots_meta": dict(s.get("spots_meta") or {})})
    del stack[:-UNDO_DEPTH]


def _forget_undo(session, img: LumenImage) -> None:
    """Drop the undo snapshots of ``img`` (after re-tracking / import the old model is gone)."""
    stack = session.settings.get("tracks_undo")
    if stack:
        session.settings["tracks_undo"] = [e for e in stack if e.get("image") != img.id]


def _tracks_table(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    """The 'Tracks' table: one row per track with its statistics and its overlay colour (hex, so the
    UI can colour the rows). Track summaries live here only, never in the shared Results table."""
    rows = [{"Label": t["name"], **t["stats"], "Color": t["color"]} for t in tracks]
    # the ordered union of all rows' keys (not the first row's), Label first and Color last: a track whose statistics
    # were computed under another calibration (or an empty manual track) must not hide a column of the others
    cols = ["Label"]
    for r in rows:
        for k in r:
            if k not in cols and k != "Color":
                cols.append(k)
    cols.append("Color")
    return {"name": "Tracks", "columns": cols, "rows": [[r.get(c) for c in cols] for r in rows]}


DENSE_SPOTS_PER_FRAME = 400


def _dense_hint(img: LumenImage, spots: list[dict[str, Any]], threshold: float) -> str:
    """A hint for the message when the automatic threshold passes implausibly many spots per frame."""
    if threshold > 0 or img.nt < 1:
        return ""
    per_frame = len(spots) / img.nt
    if per_frame <= DENSE_SPOTS_PER_FRAME:
        return ""
    return f"; {per_frame:.0f} spots per frame with the automatic threshold – consider a larger radius or threshold"


def _link_and_report(ctx, spots: list[dict[str, Any]], units: str, lp: dict[str, Any]) -> OpResult:
    """Link ``spots`` with the LINK_PARAMS values ``lp``, store the tracks of the current image and report."""
    img = ctx.img
    pw, ph, _ = _cal_units(img, units)
    penalties = {step: _parse_penalties(lp.get(f"{key}_penalties")) for step, key in (("link", "linking"), ("gap", "gap"), ("split", "split"), ("merge", "merge"))}
    _ensure_ids(spots)
    tracker = lp.get("tracker", "Simple LAP tracker")
    graph = _link_graph(spots, tracker=tracker, max_disp=lp["max_displacement"], gap_frames=lp["gap_closing_frames"], max_gap=lp["max_gap_distance"],
                        allow_split=lp.get("allow_splitting", False), split_max=lp.get("split_max_distance", 15.0), allow_merge=lp.get("allow_merging", False),
                        merge_max=lp.get("merge_max_distance", 15.0), alt_factor=lp.get("alternative_cost_factor", 1.05), penalties=penalties,
                        min_len=lp["min_track_length"], pw=pw, ph=ph, ctx=ctx)
    tracks: list[dict[str, Any]] = []
    for k, (tr, ed) in enumerate(graph):
        track = {"id": str(k), "name": f"Track {k}", "color": PALETTE[k % len(PALETTE)], "spots": [spots[i] for i in tr],
                 "edges": [[int(spots[a]["id"]), int(spots[b]["id"])] for a, b in ed], "stats": _track_stats(img, spots, tr, ed)}
        tracks.append(track)
    s = ctx.session
    _tracks_for_image(ctx, img, create=True)  # park another image's tracks
    s.settings["tracks"] = tracks
    s.settings["tracks_image"] = img.id
    _forget_undo(s, img)
    n_linked = sum(len(t["spots"]) for t in tracks)
    n_split = sum(t["stats"]["N splits"] for t in tracks)
    n_merge = sum(t["stats"]["N merges"] for t in tracks)
    n_gaps = sum(t["stats"]["N gaps"] for t in tracks)
    extras = []
    if tracker == "LAP tracker" and (lp.get("allow_splitting") or lp.get("allow_merging")):
        extras.append(f"{n_split} split{'s' if n_split != 1 else ''}, {n_merge} merge{'s' if n_merge != 1 else ''}")
    if n_gaps:
        extras.append(f"{n_gaps} gap{'s' if n_gaps != 1 else ''} closed")
    msg = f"{len(tracks)} tracks from {n_linked} of {len(spots)} spots ({'; '.join(extras + [f'speeds in {_speed_unit(img)}'])})"
    return OpResult(overlay=_all_track_overlay(img, tracks), table=_tracks_table(tracks), message=msg, log=msg)


# ================================================================================ ops
@op("tracking.detect_spots", "Detect Spots", CAT, kind="analysis", preview=False, supports_rgb=False,
    params=DETECT_PARAMS + [Param("add_to_manager", "bool", True, label="Point ROI per frame"),
                            Param("add_to_results", "bool", True, label="Spot rows in Results", help="Append one row per spot (frame, position, quality, intensity) to the Results table")],
    description="TrackMate-style spot detection on every frame of the current channel (LoG / DoG / local maxima with a quality value).",
    keywords=["trackmate", "blobs", "particles", "spots", "log", "dog"])
def detect_spots(ctx, detector="LoG", radius=2.5, threshold=0.0, subpixel=True, median_filter=False, units="pixels", z_mode="Current slice",
                 add_to_manager=True, add_to_results=True):
    img = ctx.img
    spots, r_px, thr = _detect(ctx, detector, radius, threshold, subpixel, median_filter, units, z_mode)
    s = ctx.session
    s.settings["spots"] = spots
    s.settings["spots_meta"] = {"image_id": img.id, "image_name": img.name, "channel": img.display.c, "radius_px": r_px, "threshold": thr, "detector": detector}
    _drop_tracks(ctx, img)  # a fresh detection invalidates this image's tracks only; another image's model is parked
    rois = _spot_rois(img, spots) if add_to_manager else []
    frames_with = len({sp["t"] for sp in spots})
    msg = f"{len(spots)} spots in {frames_with} frame{'s' if frames_with != 1 else ''} (quality ≥ {thr:.4g})"
    if rois:
        msg += f", {len(rois)} point ROIs added"
    msg += _dense_hint(img, spots, threshold)
    return OpResult(results=_spot_rows(img, spots) if add_to_results else None, rois_add=rois, message=msg, log=msg)


@op("tracking.link_spots", "Link Spots (LAP)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=LINK_PARAMS + [Param("units", "choice", "pixels", choices=UNITS)],
    description="Link the detected spots into tracks: TrackMate's LAP tracker (frame-to-frame linking, then gap closing and – with the full LAP tracker – track splitting and merging into lineage trees, with optional feature penalties) or the greedy nearest-neighbour tracker; draws track overlays and measures each track.",
    keywords=["trackmate", "lap", "tracks", "linking", "splitting", "merging", "lineage", "gap closing"])
def link_spots(ctx, tracker="Simple LAP tracker", max_displacement=10.0, gap_closing_frames=2, max_gap_distance=15.0, allow_splitting=False, split_max_distance=15.0,
               allow_merging=False, merge_max_distance=15.0, min_track_length=2, alternative_cost_factor=1.05, linking_penalties="", gap_penalties="",
               split_penalties="", merge_penalties="", units="pixels"):
    img = ctx.img
    spots, meta = _spots_for(ctx)
    if meta.get("image_id") and meta["image_id"] != img.id and meta.get("image_name") != img.name:
        raise ValueError(f"The stored spots belong to {meta.get('image_name')!r}; select that image or run Detect Spots again")
    lp = {k: v for k, v in locals().items() if k in LINK_KEYS}
    return _link_and_report(ctx, spots, units, lp)


@op("tracking.track", "Track (detect + link)", CAT, kind="analysis", preview=False, supports_rgb=False,
    params=DETECT_PARAMS + LINK_PARAMS + [Param("add_to_manager", "bool", False, label="Point ROI per frame"),
                                          Param("add_to_results", "bool", False, label="Spot rows in Results", help="Append one row per spot to the Results table (track summaries always go to the Tracks table)")],
    description="One-click TrackMate: detect spots on every frame and link them into tracks (summaries in the Tracks table).", keywords=["trackmate", "particle tracking"])
def track(ctx, detector="LoG", radius=2.5, threshold=0.0, subpixel=True, median_filter=False, units="pixels", z_mode="Current slice",
          tracker="Simple LAP tracker", max_displacement=10.0, gap_closing_frames=2, max_gap_distance=15.0, allow_splitting=False, split_max_distance=15.0,
          allow_merging=False, merge_max_distance=15.0, min_track_length=2, alternative_cost_factor=1.05, linking_penalties="", gap_penalties="",
          split_penalties="", merge_penalties="", add_to_manager=False, add_to_results=False):
    lp = {k: v for k, v in locals().items() if k in LINK_KEYS}
    img = ctx.img
    spots, r_px, thr = _detect(ctx, detector, radius, threshold, subpixel, median_filter, units, z_mode)
    s = ctx.session
    s.settings["spots"] = spots
    s.settings["spots_meta"] = {"image_id": img.id, "image_name": img.name, "channel": img.display.c, "radius_px": r_px, "threshold": thr, "detector": detector}
    res = _link_and_report(ctx, spots, units, lp)
    if add_to_manager:
        res.rois_add = _spot_rois(img, spots)
    if add_to_results:
        res.results = _spot_rows(img, spots)
    res.message = f"{len(spots)} spots (quality ≥ {thr:.4g}); " + (res.message or "") + _dense_hint(img, spots, threshold)
    res.log = res.message
    return res


@op("tracking.export_tracks", "Export Tracks (CSV)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("path", "string", "", help="Output .csv file"),
            Param("layout", "choice", "TrackMate", choices=["TrackMate", "Lumen"], help="TrackMate: TRACK_ID, FRAME, POSITION_X … columns (calibrated). Lumen: pixel and calibrated columns."),
            Param("include_unlinked", "bool", False, help="Also write spots that belong to no track (TRACK_ID = -1)")],
    description="Write one row per spot of every track of the current image to a CSV file.", keywords=["save", "csv", "trackmate"])
def export_tracks(ctx, path="", layout="TrackMate", include_unlinked=False):
    img, tracks, spots = _export_model(ctx)  # the current image's own model and calibration, never another image's
    if not path or not str(path).strip():
        raise ValueError("Give an output file path")
    cal = img.calibration
    dt = cal.frame_interval if cal.frame_interval and cal.frame_interval > 0 else 1.0
    p = Path(path).expanduser()
    if p.suffix == "":
        p = p.with_suffix(".csv")
    p.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[Any]] = []
    spot_id = 0
    linked_ids: set[int] = set()
    for tr in tracks:
        for sp in tr["spots"]:
            linked_ids.add(id(sp))
            rows.append(_export_row(layout, spot_id, int(tr["id"]), sp, cal, dt))
            spot_id += 1
    if include_unlinked:
        for sp in spots:
            if id(sp) not in linked_ids:
                rows.append(_export_row(layout, spot_id, -1, sp, cal, dt))
                spot_id += 1
    if layout == "TrackMate":
        header = ["ID", "TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z", "POSITION_T", "RADIUS", "QUALITY", "MEAN_INTENSITY"]
    else:
        header = ["Spot", "Track", "Frame", "X (px)", "Y (px)", "X", "Y", "Z", "Time", "Radius", "Quality", "Mean intensity"]
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    msg = f"Wrote {len(rows)} spots of {len(tracks)} tracks to {p}"
    return OpResult(message=msg, log=msg)


def _export_row(layout: str, spot_id: int, track_id: int, sp: dict[str, Any], cal: Calibration, dt: float) -> list[Any]:
    x_cal, y_cal, z_cal = sp["x"] * cal.pixel_width, sp["y"] * cal.pixel_height, sp["z"] * cal.pixel_depth
    if layout == "TrackMate":
        x_tm, y_tm = _tm_xy(sp["x"], sp["y"], cal.pixel_width, cal.pixel_height)  # TrackMate's pixel-centre convention
        return [spot_id, track_id, sp["t"], _r(x_tm), _r(y_tm), _r(z_cal), _r(sp["t"] * dt), _r(sp["r"] * cal.pixel_width), _r(sp["quality"]), _r(sp["intensity"])]
    return [spot_id, track_id, sp["t"], _r(sp["x"]), _r(sp["y"]), _r(x_cal), _r(y_cal), sp["z"], _r(sp["t"] * dt), _r(sp["r"] * cal.pixel_width), _r(sp["quality"]), _r(sp["intensity"])]


def _r(v: float) -> float:
    return round(float(v), 4) if np.isfinite(v) else float("nan")


@op("tracking.kymograph", "Kymograph", CAT, kind="analysis", preview=False, requires_roi="line", roi_aware=False, supports_rgb=False,
    params=[Param("axis", "choice", "T", choices=["T", "Z"], help="Stack axis that becomes the vertical axis"),
            Param("all_channels", "bool", False, help="One kymograph channel per image channel")],
    description="Sample the line selection on every frame and stack the profiles into a (frames × distance) image.", keywords=["space-time plot", "line profile", "multi kymograph"])
def kymograph(ctx, axis="T", all_channels=False):
    img = ctx.img
    d = img.display
    roi = ctx.roi
    if roi is None or not roimod.is_line(roi):
        raise ValueError("Kymograph needs a line or polyline selection")
    n = img.nt if axis == "T" else img.nz
    if n < 2:
        raise ValueError("Needs a time series (T > 1)" if axis == "T" else "Needs a Z stack (Z > 1)")
    cal = img.calibration
    chans = list(range(img.nc)) if all_channels else [d.c]
    planes = []
    dist = None
    for c in chans:
        rows = []
        for k in range(n):
            plane = img.data[k, d.z, c] if axis == "T" else img.data[d.t, k, c]
            pr = line_profile(plane, roi, cal)
            rows.append(np.asarray(pr["values"], dtype=np.float32))
            dist = pr["x"]
        planes.append(np.stack(rows, axis=0))
    data = np.stack(planes, axis=0)[None, None]  # (1, 1, C, N, L)
    out = LumenImage(data.astype(np.float32), name=f"Kymograph of {img.name}", axes="TZCYX")
    length = float(dist[-1]) if dist else 0.0
    step = length / max(len(dist) - 1, 1) if dist and len(dist) > 1 else cal.pixel_width
    if axis == "T":
        y_step = cal.frame_interval if cal.frame_interval and cal.frame_interval > 0 else 1.0
        y_unit = cal.time_unit if cal.frame_interval and cal.frame_interval > 0 else "frame"
    else:
        y_step = cal.pixel_depth
        y_unit = cal.unit
    out.calibration = Calibration(pixel_width=step, pixel_height=y_step, pixel_depth=1.0, unit=cal.unit,
                                  frame_interval=cal.frame_interval, time_unit=cal.time_unit)
    out.metadata.update({"kymograph": True, "x_axis": f"Distance ({cal.unit})", "y_axis": f"{'Time' if axis == 'T' else 'Z'} ({y_unit})",
                         "source": img.name, "line": {k: v for k, v in roi.items() if k in ("type", "x1", "y1", "x2", "y2", "points", "width")}})
    for c, ch in zip(chans, out.display.channels):
        ch.lut = img.display.channels[c].lut if c < len(img.display.channels) else ch.lut
        ch.name = img.display.channels[c].name if c < len(img.display.channels) else ch.name
    out.reset_display(keep_luts=True)
    msg = f"Kymograph {out.width} × {out.height} ({'frames' if axis == 'T' else 'slices'} × distance)"
    return OpResult(new_images=[out], message=msg)


@op("tracking.clear_tracks", "Clear Tracks", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("forget_spots", "bool", False, help="Also discard the stored spot list")],
    description="Remove the track overlays (and the stored tracks) from the current image.")
def clear_tracks(ctx, forget_spots=False):
    img = ctx.img
    s = ctx.session
    n = sum(1 for o in img.overlay if o.get("type") == "track")
    # image-scoped: drops this image's model (active or parked) and its undo snapshots, parks another image's model
    _drop_tracks(ctx, img)
    if forget_spots:
        meta = s.settings.get("spots_meta") or {}
        if not meta.get("image_id") or meta.get("image_id") == img.id:  # only this image's detection
            s.settings.pop("spots", None)
            s.settings.pop("spots_meta", None)
    return OpResult(overlay=[o for o in img.overlay if o.get("type") != "track"], message=f"Removed {n} track overlays")


@op("tracking.filter_tracks", "Filter Tracks", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("min_length", "int", 0, min=0, label="Min spots per track", help="0 = off"),
            Param("min_duration", "int", 0, min=0, label="Min duration (frames)", help="Last frame − first frame; 0 = off"),
            Param("min_displacement", "float", 0.0, min=0, step=1, label="Min net displacement", help="Start-to-end distance in the units of the Tracks table (calibrated when the image is); 0 = off"),
            Param("max_speed", "float", 0.0, min=0, step=1, label="Max speed", help="Largest frame-to-frame speed allowed, in the units of the Tracks table; 0 = off"),
            Param("feature", "choice", "None", choices=["None"] + TRACK_FEATURES, help="Any track feature of the Tracks table (TrackMate's track filters); 'None' = only the criteria above"),
            Param("threshold", "float", 0.0, step=1, label="Feature threshold"),
            Param("above", "bool", True, help="Keep the tracks whose feature is ≥ the threshold (unticked: ≤)")],
    description="Keep only the tracks that pass the length / duration / displacement / speed criteria and/or a threshold on any track feature (N splits, N gaps, Mean quality …); the track overlays and the Tracks table are updated.",
    keywords=["trackmate", "track filter", "spurious tracks", "short tracks", "feature filter"])
def filter_tracks(ctx, min_length=0, min_duration=0, min_displacement=0.0, max_speed=0.0, feature="None", threshold=0.0, above=True):
    img = ctx.img
    s = ctx.session
    try:
        tracks = _tracks_for_image(ctx, img)
    except ValueError:
        raise ValueError("Run Link Spots (or Track) on this image first")
    if not tracks:
        raise ValueError("Run Link Spots (or Track) on this image first")
    kept = []
    for t in tracks:
        st = t.get("stats") or {}
        if min_length > 0 and st.get("N spots", len(t.get("spots", []))) < min_length:
            continue
        if min_duration > 0 and st.get("Duration (frames)", 0) < min_duration:
            continue
        if min_displacement > 0 and st.get("Net displacement", 0.0) < min_displacement:
            continue
        if max_speed > 0 and st.get("Max speed", 0.0) > max_speed:
            continue
        if feature != "None":
            v = st.get(feature)
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
            if not np.isfinite(v) or (v < threshold if above else v > threshold):
                continue
        kept.append(t)
    keep_ids = {_overlay_id(t) for t in kept}
    overlay = [o for o in img.overlay if o.get("type") != "track" or _item_track_key(o) in keep_ids]
    s.settings["tracks"] = kept
    removed = len(tracks) - len(kept)
    msg = f"Kept {len(kept)} of {len(tracks)} tracks ({removed} removed)"
    if feature != "None":
        msg += f" – {feature} {'≥' if above else '≤'} {threshold:g}"
    return OpResult(overlay=overlay, table=_tracks_table(kept), message=msg, log=msg)


# ================================================================================ manual tracking
MANUAL_RADIUS = 2.5


def _tracks_for_image(ctx, img: LumenImage, create: bool = False) -> list[dict[str, Any]]:
    """The session's track list for ``img`` (one track model per image, like TrackMate).

    Tracks of another image are parked in ``settings["tracks_archive"]`` and restored when that image is
    tracked again, so switching images never loses or mixes tracks.  ``create`` starts a fresh list."""
    s = ctx.session
    archive = s.settings.setdefault("tracks_archive", {})
    tracks = s.settings.get("tracks")
    tid = s.settings.get("tracks_image")
    if tracks is not None and tid and tid != img.id:
        # another image's model is active: park it only once ``img`` is known to have (or may create) a model, so a
        # failing call leaves tracks / tracks_image / tracks_archive exactly as they were (a failed command changes nothing)
        if not create and img.id not in archive:
            raise ValueError("No tracks for this image – start one with 'Manual Tracking: New Track' (or run Track)")
        archive[tid] = tracks
        tracks = archive.pop(img.id, None)
        s.settings["tracks"] = tracks
    elif tracks is None and img.id in archive:
        tracks = archive.pop(img.id)
        s.settings["tracks"] = tracks
    if tracks is None:
        if not create:
            raise ValueError("No tracks for this image – start one with 'Manual Tracking: New Track' (or run Track)")
        tracks = []
        s.settings["tracks"] = tracks
    s.settings["tracks_image"] = img.id
    return tracks


def _drop_tracks(ctx, img: LumenImage) -> None:
    """Forget the track model of ``img`` only – wherever it lives (active, or parked in ``tracks_archive``) – together
    with its undo snapshots.  Another image's active model is parked in the archive, never deleted."""
    s = ctx.session
    _tracks_for_image(ctx, img, create=True)  # parks another image's list, pulls img's own out of the archive
    s.settings.pop("tracks", None)
    s.settings.pop("tracks_image", None)
    (s.settings.get("tracks_archive") or {}).pop(img.id, None)
    _forget_undo(s, img)


def _spot_pool(session, img: LumenImage) -> list[dict[str, Any]]:
    """The stored detection only when it belongs to ``img`` (never another image's spots)."""
    meta = session.settings.get("spots_meta") or {}
    spots = session.settings.get("spots")
    if not isinstance(spots, list):
        return []
    if meta.get("image_id") == img.id or (not meta.get("image_id") and meta.get("image_name") == img.name):
        return spots
    return []


def _export_model(ctx) -> tuple[LumenImage, list[dict[str, Any]], list[dict[str, Any]]]:
    """(image, tracks, detection) of the current image for the exporters – read-only: the tracks are the active model
    or the image's parked copy in ``tracks_archive`` (the active model is never swapped by an export), and the detection
    is included only when it belongs to the same image."""
    s = ctx.session
    img = ctx.img
    if s.settings.get("tracks_image") == img.id:
        tracks = s.settings.get("tracks")
    else:
        tracks = (s.settings.get("tracks_archive") or {}).get(img.id)
    if not tracks:
        raise ValueError("No tracks for this image – run Track / Link Spots or create manual tracks first")
    return img, tracks, _spot_pool(s, img)


def _find_track(tracks: list[dict[str, Any]], track_id: int) -> dict[str, Any]:
    if track_id < 0:
        manual = [t for t in tracks if t.get("manual")]
        if not manual and not tracks:
            raise ValueError("No tracks – start one with 'Manual Tracking: New Track'")
        return (manual or tracks)[-1]
    for t in tracks:
        if str(t.get("id")) == str(track_id):
            return t
    raise ValueError(f"No track with id {track_id}")


def _manual_spot(img: LumenImage, x: float, y: float, t: int, radius: float = MANUAL_RADIUS, spot_id: int | None = None) -> dict[str, Any]:
    """A spot placed by hand (or by Add Spot): full intensity features measured on the current plane, quality 1."""
    d = img.display
    plane = img.data[t, d.z, d.c]
    st = _disk_stats(plane.astype(np.float32, copy=False), np.array([x]), np.array([y]), max(float(radius), 1.0))
    sp: dict[str, Any] = {"t": int(t), "x": float(x), "y": float(y), "z": int(d.z), "r": float(radius), "quality": 1.0, "manual": True}
    for key in INTENSITY_KEYS:
        sp[key] = float(st[key][0])
    if spot_id is not None:
        sp["id"] = int(spot_id)
    return sp


def _track_overlay_item(track: dict[str, Any]) -> dict[str, Any]:
    """The first (main-branch) overlay item of a track."""
    return _track_overlay_items(track)[0]


def _attach_spot(track: dict[str, Any], spot: dict[str, Any], replaced: dict[str, Any] | None = None) -> None:
    """Keep a track's explicit lineage consistent when ``spot`` is added to it (or replaces ``replaced``, same frame):
    a replaced spot takes over the links of the old one; a new spot is linked from the latest earlier spot (re-routing
    that spot's link to the next one through the new spot) or to the earliest later spot."""
    if track.get("edges") is None:
        return
    edges = [list(e) for e in track["edges"]]
    nid = int(spot["id"])
    if replaced is not None:
        rid = int(replaced["id"])
        for e in edges:
            if e[0] == rid:
                e[0] = nid
            if e[1] == rid:
                e[1] = nid
        track["edges"] = edges
        return
    others = [s for s in track["spots"] if s is not spot]
    prev = max((s for s in others if s["t"] < spot["t"]), key=_spot_key, default=None)
    nxt = min((s for s in others if s["t"] > spot["t"]), key=_spot_key, default=None)
    if prev is not None:
        if nxt is not None and [int(prev["id"]), int(nxt["id"])] in edges:
            edges.remove([int(prev["id"]), int(nxt["id"])])
            edges.append([nid, int(nxt["id"])])
        edges.append([int(prev["id"]), nid])
    elif nxt is not None:
        edges.append([nid, int(nxt["id"])])
    track["edges"] = edges


def _detach_spot(track: dict[str, Any], spot: dict[str, Any], bridge: bool = True) -> None:
    """Remove the links of ``spot`` from a track's explicit lineage; with ``bridge`` its single predecessor and
    successor are joined so a manual track stays in one piece."""
    if track.get("edges") is None:
        return
    sid = int(spot["id"])
    preds = [int(e[0]) for e in track["edges"] if int(e[1]) == sid]
    succs = [int(e[1]) for e in track["edges"] if int(e[0]) == sid]
    edges = [list(e) for e in track["edges"] if sid not in (int(e[0]), int(e[1]))]
    if bridge and len(preds) == 1 and len(succs) == 1 and [preds[0], succs[0]] not in edges:
        edges.append([preds[0], succs[0]])
    track["edges"] = edges


def _refresh_track(ctx, img: LumenImage, track: dict[str, Any]) -> list[dict[str, Any]]:
    """Recompute a track's statistics and return the image overlay with its items replaced."""
    _ensure_ids(track["spots"], ctx.session.settings.get("spots"))
    _refresh_stats(img, track)
    overlay = _overlay_without(img.overlay, _overlay_id(track))
    overlay.extend(_track_overlay_items(track))
    return overlay


def _next_track_number(tracks: list[dict[str, Any]]) -> int:
    nums = []
    for t in tracks:
        try:
            nums.append(int(t["id"]))
        except (TypeError, ValueError, KeyError):
            pass
    return (max(nums) + 1) if nums else 0


@op("tracking.manual_new_track", "Manual Tracking: New Track", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("name", "string", "", help="Track name (default 'Manual <n>')"), Param("color", "string", "", help="Overlay colour (default: palette)")],
    description="Start a new manual track (empty track overlay 'manual-<n>'); add points frame by frame with 'Manual Tracking: Add Point'. The new track id is reported in the message.",
    keywords=["manual tracking", "mtrackj", "click", "new track"])
def manual_new_track(ctx, name="", color=""):
    img = ctx.img
    tracks = _tracks_for_image(ctx, img, create=True)
    _push_undo(ctx.session, img, tracks, "New Track")
    n = _next_track_number(tracks)
    track = {"id": str(n), "name": name or f"Manual {n}", "color": color or PALETTE[n % len(PALETTE)], "spots": [], "stats": {}, "manual": True, "overlay_id": f"manual-{n}"}
    tracks.append(track)
    overlay = _refresh_track(ctx, img, track)
    msg = f"New track {n} ({track['name']}): add points with 'Manual Tracking: Add Point' (track_id = {n})"
    return OpResult(overlay=overlay, table=_tracks_table(tracks), message=msg, log=msg)


@op("tracking.manual_add_point", "Manual Tracking: Add Point", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("track_id", "int", -1, min=-1, help="Track id from 'New Track' (−1 = the last manual track)"),
            Param("x", "float", -1, step=0.5, help="Position (−1 = the current point selection)"), Param("y", "float", -1, step=0.5),
            Param("t", "int", -1, min=-1, label="Frame", help="0-based frame (−1 = the current frame)"),
            Param("radius", "float", MANUAL_RADIUS, min=0.5, step=0.5, label="Spot radius (px)"),
            Param("advance", "bool", True, help="Move the display to the next frame afterwards")],
    description="Add (or replace) the point of a manual track on a frame: the given x, y or the current point selection, on the current frame. The overlay, the stored tracks and the Tracks table are updated.",
    keywords=["manual tracking", "mtrackj", "click", "add point"])
def manual_add_point(ctx, track_id=-1, x=-1.0, y=-1.0, t=-1, radius=MANUAL_RADIUS, advance=True):
    img = ctx.img
    d = img.display
    tracks = _tracks_for_image(ctx, img)
    track = _find_track(tracks, int(track_id))
    if x < 0 or y < 0:
        roi = ctx.roi
        if roi is None or roi.get("type") != "point" or not roi.get("points"):
            raise ValueError("Give x and y or make a point selection")
        x, y = (float(v) for v in roi["points"][-1])
    frame = int(d.t) if t < 0 else int(t)
    if frame >= img.nt:
        raise ValueError(f"Frame {frame} is beyond the last frame ({img.nt - 1})")
    s = ctx.session
    _ensure_ids(s.settings.get("spots"), *(tr["spots"] for tr in tracks))
    _push_undo(s, img, tracks, "Add Point")
    spot = _manual_spot(img, x, y, frame, radius, spot_id=_next_spot_id(s.settings.get("spots"), *(tr["spots"] for tr in tracks)))
    replaced = False
    for i, sp in enumerate(track["spots"]):
        if sp["t"] == frame:
            track["spots"][i] = spot
            _attach_spot(track, spot, replaced=sp)
            replaced = True
            break
    if not replaced:
        track["spots"].append(spot)
        _attach_spot(track, spot)
    overlay = _refresh_track(ctx, img, track)
    res = OpResult(overlay=overlay, table=_tracks_table(tracks))
    res.message = f"{track['name']}: point ({x:.1f}, {y:.1f}) {'replaced' if replaced else 'added'} on frame {frame} ({len(track['spots'])} spots)"
    res.log = res.message
    if advance and frame + 1 < img.nt and t < 0:
        res.display = {"t": frame + 1}
    return res


@op("tracking.manual_delete_point", "Manual Tracking: Delete Point", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("track_id", "int", -1, min=-1, help="Track id (−1 = the last manual track)"), Param("t", "int", -1, min=-1, label="Frame", help="0-based frame (−1 = the current frame)"),
            Param("delete_track", "bool", False, help="Remove the whole track instead")],
    description="Remove the point of a manual track on a frame (or the whole track).", keywords=["manual tracking", "delete point", "remove track"])
def manual_delete_point(ctx, track_id=-1, t=-1, delete_track=False):
    img = ctx.img
    tracks = _tracks_for_image(ctx, img)
    track = _find_track(tracks, int(track_id))
    if delete_track:
        _push_undo(ctx.session, img, tracks, "Delete Track")
        tracks.remove(track)
        overlay = _overlay_without(img.overlay, _overlay_id(track))
        msg = f"Removed {track['name']}"
        return OpResult(overlay=overlay, table=_tracks_table(tracks), message=msg, log=msg)
    frame = int(img.display.t) if t < 0 else int(t)
    gone = [sp for sp in track["spots"] if sp["t"] == frame]
    if not gone:
        raise ValueError(f"{track['name']} has no point on frame {frame}")
    _ensure_ids(ctx.session.settings.get("spots"), *(tr["spots"] for tr in tracks))
    _push_undo(ctx.session, img, tracks, "Delete Point")
    track["spots"] = [sp for sp in track["spots"] if sp["t"] != frame]
    for sp in gone:
        _detach_spot(track, sp, bridge=True)
    overlay = _refresh_track(ctx, img, track)
    msg = f"{track['name']}: point on frame {frame} removed ({len(track['spots'])} spots left)"
    return OpResult(overlay=overlay, table=_tracks_table(tracks), message=msg, log=msg)


# ================================================================================ TrackMate XML
_SPOT_FEATURES = [  # (feature, name, shortname, dimension, isint)
    ("QUALITY", "Quality", "Quality", "QUALITY", "false"), ("POSITION_X", "X", "X", "POSITION", "false"), ("POSITION_Y", "Y", "Y", "POSITION", "false"),
    ("POSITION_Z", "Z", "Z", "POSITION", "false"), ("POSITION_T", "T", "T", "TIME", "false"), ("FRAME", "Frame", "Frame", "NONE", "true"),
    ("RADIUS", "Radius", "R", "LENGTH", "false"), ("VISIBILITY", "Visibility", "Visibility", "NONE", "true"),
    ("MEAN_INTENSITY_CH1", "Mean intensity ch1", "Mean ch1", "INTENSITY", "false"), ("MEDIAN_INTENSITY_CH1", "Median intensity ch1", "Median ch1", "INTENSITY", "false"),
    ("MIN_INTENSITY_CH1", "Min intensity ch1", "Min ch1", "INTENSITY", "false"), ("MAX_INTENSITY_CH1", "Max intensity ch1", "Max ch1", "INTENSITY", "false"),
    ("TOTAL_INTENSITY_CH1", "Sum intensity ch1", "Sum ch1", "INTENSITY", "false"), ("STD_INTENSITY_CH1", "Std intensity ch1", "Std ch1", "INTENSITY", "false"),
    ("CONTRAST_CH1", "Contrast ch1", "Ctrst ch1", "NONE", "false"), ("SNR_CH1", "Signal/Noise ratio ch1", "SNR ch1", "NONE", "false"),
]
#: TrackMate spot feature key → Lumen spot dict key, for the exporters (intensity features; positions are handled separately)
_TM_SPOT_INTENSITY = [("MEAN_INTENSITY_CH1", "intensity"), ("MEDIAN_INTENSITY_CH1", "median"), ("MIN_INTENSITY_CH1", "min"), ("MAX_INTENSITY_CH1", "max"),
                      ("TOTAL_INTENSITY_CH1", "total"), ("STD_INTENSITY_CH1", "std"), ("CONTRAST_CH1", "contrast"), ("SNR_CH1", "snr")]
_EDGE_FEATURES = [
    ("SPOT_SOURCE_ID", "Source spot ID", "Source ID", "NONE", "true"), ("SPOT_TARGET_ID", "Target spot ID", "Target ID", "NONE", "true"),
    ("LINK_COST", "Edge cost", "Cost", "COST", "false"), ("DIRECTIONAL_CHANGE_RATE", "Directional change rate", "γ rate", "ANGLE_RATE", "false"),
    ("SPEED", "Speed", "Speed", "VELOCITY", "false"), ("DISPLACEMENT", "Displacement", "Disp.", "LENGTH", "false"),
    ("EDGE_TIME", "Edge time", "Edge T", "TIME", "false"), ("EDGE_X_LOCATION", "Edge X", "Edge X", "POSITION", "false"),
    ("EDGE_Y_LOCATION", "Edge Y", "Edge Y", "POSITION", "false"), ("EDGE_Z_LOCATION", "Edge Z", "Edge Z", "POSITION", "false"),
]
_TRACK_FEATURES = [
    ("TRACK_INDEX", "Track index", "Index", "NONE", "true"), ("TRACK_ID", "Track ID", "ID", "NONE", "true"),
    ("NUMBER_SPOTS", "Number of spots in track", "N spots", "NONE", "true"), ("NUMBER_GAPS", "Number of gaps", "N gaps", "NONE", "true"),
    ("NUMBER_SPLITS", "Number of split events", "N splits", "NONE", "true"), ("NUMBER_MERGES", "Number of merge events", "N merges", "NONE", "true"),
    ("NUMBER_COMPLEX", "Number of complex points", "N complex", "NONE", "true"), ("LONGEST_GAP", "Longest gap", "Lgst gap", "NONE", "true"),
    ("TRACK_DURATION", "Track duration", "Duration", "TIME", "false"), ("TRACK_START", "Track start", "Track start", "TIME", "false"),
    ("TRACK_STOP", "Track stop", "Track stop", "TIME", "false"), ("TRACK_DISPLACEMENT", "Track displacement", "Track disp.", "LENGTH", "false"),
    ("TRACK_X_LOCATION", "Track mean X", "Track X", "POSITION", "false"), ("TRACK_Y_LOCATION", "Track mean Y", "Track Y", "POSITION", "false"),
    ("TRACK_Z_LOCATION", "Track mean Z", "Track Z", "POSITION", "false"), ("TRACK_MEAN_SPEED", "Track mean speed", "Mean sp.", "VELOCITY", "false"),
    ("TRACK_MAX_SPEED", "Track max speed", "Max speed", "VELOCITY", "false"), ("TRACK_MIN_SPEED", "Track min speed", "Min speed", "VELOCITY", "false"),
    ("TRACK_MEDIAN_SPEED", "Track median speed", "Med. speed", "VELOCITY", "false"), ("TRACK_STD_SPEED", "Track std speed", "Std speed", "VELOCITY", "false"),
    ("TRACK_MEAN_QUALITY", "Track mean quality", "Mean Q", "QUALITY", "false"), ("TOTAL_DISTANCE_TRAVELED", "Total distance traveled", "Total dist.", "LENGTH", "false"),
    ("MAX_DISTANCE_TRAVELED", "Max distance traveled", "Max dist.", "LENGTH", "false"), ("CONFINEMENT_RATIO", "Confinement ratio", "Cfn. ratio", "NONE", "false"),
    ("MEAN_STRAIGHT_LINE_SPEED", "Mean straight line speed", "Mn. v. line", "VELOCITY", "false"),
    ("LINEARITY_OF_FORWARD_PROGRESSION", "Linearity of forward progression", "Fwd. progr.", "NONE", "false"),
    ("MEAN_DIRECTIONAL_CHANGE_RATE", "Mean directional change rate", "Mn. γ rate", "ANGLE_RATE", "false"),
]
_OME_UNIT = {"µm": "µm", "um": "µm", "micron": "µm", "pixel": "pixel", "pixels": "pixel", "": "pixel"}


def _xml_num(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "0.0"
    if not np.isfinite(f):
        return "NaN"
    return repr(f) if f != int(f) or abs(f) >= 1e15 else f"{f:.1f}"


#: Lumen puts the centre of pixel i at i + 0.5 px; TrackMate puts it at i * pixel size (its detectors emit
#: index * calibration and SpotOverlay adds half a pixel when drawing). Shift by this at the file boundary only.
TM_OFFSET = 0.5


def _tm_xy(x: float, y: float, pw: float, ph: float) -> tuple[float, float]:
    """Lumen pixel coordinates (x, y) → TrackMate calibrated coordinates; the inverse is ``v / pixel_size + TM_OFFSET``."""
    return (float(x) - TM_OFFSET) * pw, (float(y) - TM_OFFSET) * ph


def _feature_declarations(parent, tag: str, feats: list[tuple[str, str, str, str, str]]) -> None:
    import xml.etree.ElementTree as ET
    el = ET.SubElement(parent, tag)
    for feature, name, short, dim, isint in feats:
        ET.SubElement(el, "Feature", feature=feature, name=name, shortname=short, dimension=dim, isint=isint)


# ---- TrackMate feature values (numeric), shared by the XML exporter and the feature tables / CSV exports
def _tm_scale(cal: Calibration) -> tuple[float, float, float, float]:
    """(pixel width, pixel height, voxel depth, frame interval) with TrackMate's defaults for uncalibrated images."""
    dt = float(cal.frame_interval) if cal.frame_interval and cal.frame_interval > 0 else 1.0
    return float(cal.pixel_width), float(cal.pixel_height), float(cal.pixel_depth), dt


def _tm_spot_features(sp: dict[str, Any], tm_id: int, cal: Calibration) -> dict[str, Any]:
    """TrackMate's spot features (calibrated, pixel-centre convention) of a Lumen spot."""
    pw, ph, pd, dt = _tm_scale(cal)
    tx, ty = _tm_xy(sp["x"], sp["y"], pw, ph)
    out: dict[str, Any] = {"ID": int(tm_id), "name": str(sp.get("name") or f"ID{tm_id}"), "VISIBILITY": int(_spot_value(sp, "visibility")) if np.isfinite(_spot_value(sp, "visibility")) else 1,
                           "QUALITY": float(sp.get("quality", 1.0)), "POSITION_X": float(tx), "POSITION_Y": float(ty), "POSITION_Z": float(sp.get("z", 0)) * pd,
                           "POSITION_T": float(sp["t"]) * dt, "FRAME": int(sp["t"]), "RADIUS": float(sp.get("r", MANUAL_RADIUS)) * pw}
    for key, lkey in _TM_SPOT_INTENSITY:
        out[key] = _spot_value(sp, lkey)
    return out


def _tm_edge_features(sa: dict[str, Any], sb: dict[str, Any], ida: int, idb: int, cal: Calibration, prev: dict[str, Any] | None = None) -> dict[str, Any]:
    """TrackMate's edge features of the link sa → sb (``prev``: the unique predecessor of sa, for the directional change rate)."""
    pw, ph, pd, dt = _tm_scale(cal)
    dx, dy = (sb["x"] - sa["x"]) * pw, (sb["y"] - sa["y"]) * ph
    disp = math.hypot(dx, dy)
    edt = max((sb["t"] - sa["t"]) * dt, 1e-12)
    ex, ey = _tm_xy((sa["x"] + sb["x"]) / 2.0, (sa["y"] + sb["y"]) / 2.0, pw, ph)
    rate = float("nan")
    if prev is not None:
        px, py = (sa["x"] - prev["x"]) * pw, (sa["y"] - prev["y"]) * ph
        if math.hypot(px, py) > 0 and disp > 0:
            rate = abs(math.atan2(px * dy - py * dx, px * dx + py * dy)) / edt
    return {"SPOT_SOURCE_ID": int(ida), "SPOT_TARGET_ID": int(idb), "LINK_COST": disp ** 2, "DIRECTIONAL_CHANGE_RATE": rate, "SPEED": disp / edt,
            "DISPLACEMENT": disp, "EDGE_TIME": (sa["t"] + sb["t"]) / 2.0 * dt, "EDGE_X_LOCATION": float(ex), "EDGE_Y_LOCATION": float(ey),
            "EDGE_Z_LOCATION": (sa.get("z", 0) + sb.get("z", 0)) / 2.0 * pd}


def _tm_track_features(track: dict[str, Any], index: int, cal: Calibration) -> dict[str, Any]:
    """TrackMate's track features of a track (lineage-aware: speeds along the edges, split / merge / complex counts)."""
    pw, ph, pd, dt = _tm_scale(cal)
    try:
        tid = int(track["id"])
    except (TypeError, ValueError):
        tid = index
    spots = sorted(track["spots"], key=_spot_key)
    by_id = {int(sp["id"]): sp for sp in spots}
    edges = [(a, b) for a, b in _track_edges(track) if a in by_id and b in by_id]
    pts = np.array([[*_tm_xy(sp["x"], sp["y"], pw, ph), sp.get("z", 0) * pd] for sp in spots], float).reshape(-1, 3)
    ts = np.array([sp["t"] for sp in spots], float)
    preds: dict[int, list[int]] = defaultdict(list)
    succs: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        preds[b].append(a)
        succs[a].append(b)
    steps, dts, rates = [], [], []
    for a, b in edges:
        sa, sb = by_id[a], by_id[b]
        prev = by_id[preds[a][0]] if len(preds[a]) == 1 else None
        ef = _tm_edge_features(sa, sb, a, b, cal, prev)
        steps.append(ef["DISPLACEMENT"])
        dts.append(sb["t"] - sa["t"])
        rates.append(ef["DIRECTIONAL_CHANGE_RATE"])
    steps_a = np.array(steps, float)
    dts_a = np.array(dts, float)
    speeds = steps_a / np.where(dts_a > 0, dts_a * dt, 1e-12) if len(steps_a) else np.zeros(0)
    rates_a = np.array([r for r in rates if np.isfinite(r)], float)
    net = float(np.hypot(*(pts[-1, :2] - pts[0, :2]))) if len(spots) > 1 else 0.0
    total = float(steps_a.sum())
    dur = float((ts[-1] - ts[0]) * dt) if len(spots) else 0.0
    far = float(np.max(np.hypot(pts[:, 0] - pts[0, 0], pts[:, 1] - pts[0, 1]))) if len(spots) else 0.0
    mean_speed = float(speeds.mean()) if len(speeds) else 0.0
    return {"LABEL": str(track.get("name") or f"Track_{tid}"), "TRACK_ID": tid, "TRACK_INDEX": int(index), "NUMBER_SPOTS": len(spots),
            "NUMBER_GAPS": int(np.sum(dts_a > 1)) if len(dts_a) else 0, "NUMBER_SPLITS": sum(1 for v in succs.values() if len(v) > 1),
            "NUMBER_MERGES": sum(1 for v in preds.values() if len(v) > 1), "NUMBER_COMPLEX": sum(1 for i in by_id if len(succs.get(i, [])) > 1 and len(preds.get(i, [])) > 1),
            "LONGEST_GAP": int(max(dts_a.max() - 1, 0)) if len(dts_a) else 0, "TRACK_DURATION": dur, "TRACK_START": float(ts[0] * dt) if len(ts) else 0.0,
            "TRACK_STOP": float(ts[-1] * dt) if len(ts) else 0.0, "TRACK_DISPLACEMENT": net,
            "TRACK_X_LOCATION": float(pts[:, 0].mean()) if len(pts) else 0.0, "TRACK_Y_LOCATION": float(pts[:, 1].mean()) if len(pts) else 0.0,
            "TRACK_Z_LOCATION": float(pts[:, 2].mean()) if len(pts) else 0.0, "TRACK_MEAN_SPEED": mean_speed,
            "TRACK_MAX_SPEED": float(speeds.max()) if len(speeds) else 0.0, "TRACK_MIN_SPEED": float(speeds.min()) if len(speeds) else 0.0,
            "TRACK_MEDIAN_SPEED": float(np.median(speeds)) if len(speeds) else 0.0, "TRACK_STD_SPEED": float(speeds.std()) if len(speeds) else 0.0,
            "TRACK_MEAN_QUALITY": float(np.mean([sp.get("quality", 1.0) for sp in spots])) if spots else 0.0, "TOTAL_DISTANCE_TRAVELED": total,
            "MAX_DISTANCE_TRAVELED": far, "CONFINEMENT_RATIO": net / total if total > 0 else 0.0, "MEAN_STRAIGHT_LINE_SPEED": net / dur if dur > 0 else 0.0,
            "LINEARITY_OF_FORWARD_PROGRESSION": (net / dur) / mean_speed if dur > 0 and mean_speed > 0 else 0.0,
            "MEAN_DIRECTIONAL_CHANGE_RATE": float(rates_a.mean()) if len(rates_a) else 0.0}


def _xml_attrs(feats: dict[str, Any]) -> dict[str, str]:
    """Feature values as XML attribute strings (ints without a decimal point, floats via _xml_num, strings as they are)."""
    out: dict[str, str] = {}
    for k, v in feats.items():
        if isinstance(v, str):
            out[k] = v
        elif isinstance(v, (bool, int, np.integer)):
            out[k] = str(int(v))
        else:
            out[k] = _xml_num(v)
    return out


@op("tracking.export_trackmate_xml", "Export Tracks (TrackMate XML)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("path", "string", "", help="Output .xml file"),
            Param("include_unlinked", "bool", False, help="Also write spots that belong to no track")],
    description="Write the current image's tracks as a TrackMate XML file (Model with SpotCollection / AllTracks / FilteredTracks, feature declarations and image settings) that TrackMate (Fiji ▸ Plugins ▸ Tracking ▸ Load a TrackMate file) can open.",
    keywords=["trackmate", "xml", "save tracks", "fiji"])
def export_trackmate_xml(ctx, path="", include_unlinked=False):
    import xml.etree.ElementTree as ET
    img, tracks, spots = _export_model(ctx)  # the current image's own model and calibration, never another image's
    if not path or not str(path).strip():
        raise ValueError("Give an output file path")
    cal = img.calibration
    pw, ph, pd = float(cal.pixel_width), float(cal.pixel_height), float(cal.pixel_depth)
    dt = float(cal.frame_interval) if cal.frame_interval and cal.frame_interval > 0 else 1.0
    unit = cal.unit if cal.scaled else "pixel"
    tunit = cal.time_unit if cal.frame_interval and cal.frame_interval > 0 else "frame"
    p = Path(path).expanduser()
    if p.suffix.lower() != ".xml":
        p = p.with_name(p.name + ".xml")
    p.parent.mkdir(parents=True, exist_ok=True)
    root = ET.Element("TrackMate", version="7.11.1")
    model = ET.SubElement(root, "Model", spatialunits=unit, timeunits=tunit)
    decl = ET.SubElement(model, "FeatureDeclarations")
    _feature_declarations(decl, "SpotFeatures", _SPOT_FEATURES)
    _feature_declarations(decl, "EdgeFeatures", _EDGE_FEATURES)
    _feature_declarations(decl, "TrackFeatures", _TRACK_FEATURES)
    # ---- spots (every spot once; TrackMate IDs are the Lumen spot ids, renumbered only on a clash)
    _ensure_ids(spots, *(tr["spots"] for tr in tracks))
    spot_ids: dict[int, int] = {}  # id(spot dict) → TrackMate ID
    used_ids: set[int] = set()
    by_frame: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    fresh = [0]

    def register(sp: dict[str, Any]) -> int:
        k = id(sp)
        if k in spot_ids:
            return spot_ids[k]
        sid = sp.get("id")
        if not isinstance(sid, (int, np.integer)) or int(sid) in used_ids:
            while fresh[0] in used_ids:
                fresh[0] += 1
            sid = fresh[0]
        sid = int(sid)
        used_ids.add(sid)
        spot_ids[k] = sid
        by_frame.setdefault(int(sp["t"]), []).append((sid, sp))
        return sid

    for tr in tracks:
        for sp in tr["spots"]:
            register(sp)
    if include_unlinked:
        for sp in spots:
            register(sp)
    all_spots = ET.SubElement(model, "AllSpots", nspots=str(len(spot_ids)))
    for frame in sorted(by_frame):
        sif = ET.SubElement(all_spots, "SpotsInFrame", frame=str(frame))
        for sid, sp in by_frame[frame]:
            ET.SubElement(sif, "Spot", **_xml_attrs(_tm_spot_features(sp, sid, cal)))
    # ---- tracks + lineage edges (gap closing, splits and merges are plain edges in TrackMate's model)
    all_tracks = ET.SubElement(model, "AllTracks")
    filtered = ET.SubElement(model, "FilteredTracks")
    for index, tr in enumerate(tracks):
        feats = _tm_track_features(tr, index, cal)
        tid = feats["TRACK_ID"]
        attrs = {"name": feats.pop("LABEL")}
        attrs.update(_xml_attrs(feats))
        tel = ET.SubElement(all_tracks, "Track", **attrs)
        by_lumen = {int(sp["id"]): sp for sp in tr["spots"]}
        edges = [(a, b) for a, b in _track_edges(tr) if a in by_lumen and b in by_lumen]
        preds: dict[int, list[int]] = defaultdict(list)
        for a, b in edges:
            preds[b].append(a)
        for a, b in edges:
            sa, sb = by_lumen[a], by_lumen[b]
            prev = by_lumen[preds[a][0]] if len(preds[a]) == 1 else None
            ET.SubElement(tel, "Edge", **_xml_attrs(_tm_edge_features(sa, sb, spot_ids[id(sa)], spot_ids[id(sb)], cal, prev)))
        ET.SubElement(filtered, "TrackID", TRACK_ID=str(tid))
    # ---- settings (image geometry, manual detector / tracker so TrackMate needs no parameters)
    settings = ET.SubElement(root, "Settings")
    if img is not None:
        ET.SubElement(settings, "ImageData", filename=Path(img.path).name if img.path else f"{img.name}.tif", folder=str(Path(img.path).parent) if img.path else "",
                      width=str(img.width), height=str(img.height), nslices=str(img.nz), nframes=str(img.nt), pixelwidth=_xml_num(pw), pixelheight=_xml_num(ph),
                      voxeldepth=_xml_num(pd), timeinterval=_xml_num(dt))
        ET.SubElement(settings, "BasicSettings", xstart="0", xend=str(img.width - 1), ystart="0", yend=str(img.height - 1), zstart="0", zend=str(img.nz - 1),
                      tstart="0", tend=str(img.nt - 1))
    ET.SubElement(settings, "DetectorSettings", DETECTOR_NAME="MANUAL_DETECTOR", RADIUS=_xml_num(MANUAL_RADIUS * pw))
    ET.SubElement(settings, "InitialSpotFilter", feature="QUALITY", value="0.0", isabove="true")
    ET.SubElement(settings, "SpotFilterCollection")
    ET.SubElement(settings, "TrackerSettings", TRACKER_NAME="MANUAL_TRACKER")
    ET.SubElement(settings, "TrackFilterCollection")
    analyzers = ET.SubElement(settings, "AnalyzerCollection")
    sa = ET.SubElement(analyzers, "SpotAnalyzers")
    ET.SubElement(sa, "Analyzer", key="Spot intensity")
    ea = ET.SubElement(analyzers, "EdgeAnalyzers")
    for key in ("Edge target", "Edge speed", "Edge location", "Directional change"):
        ET.SubElement(ea, "Analyzer", key=key)
    ta = ET.SubElement(analyzers, "TrackAnalyzers")
    for key in ("Branching analyzer", "Track duration", "Track index", "Track location", "Track speed", "Track quality", "Track motility analysis"):
        ET.SubElement(ta, "Analyzer", key=key)
    ET.SubElement(root, "GUIState", state="ConfigureViews")
    ET.SubElement(root, "DisplaySettings").text = "{}"
    try:
        ET.indent(root)
    except AttributeError:  # pragma: no cover – Python < 3.9
        pass
    ET.ElementTree(root).write(str(p), encoding="utf-8", xml_declaration=True)
    msg = f"Wrote {len(spot_ids)} spots of {len(tracks)} tracks to {p} (TrackMate XML)"
    return OpResult(message=msg, log=msg)


def _attr_float(el, key: str, default: float = float("nan")) -> float:
    v = el.get(key)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


@op("tracking.import_trackmate_xml", "Import Tracks (TrackMate XML)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("path", "string", "", help="A TrackMate .xml file"),
            Param("only_filtered", "bool", True, help="Only the tracks listed in FilteredTracks (visible in TrackMate)"),
            Param("units", "choice", "From file", choices=["From file", "Pixels"], help="From file: positions are physical units, divided by the pixel size of the file (else of the image). Pixels: positions are taken as pixels. Either way half a pixel is added: TrackMate puts the centre of pixel i at i, Lumen at i + 0.5"),
            Param("add_to_manager", "bool", False, label="Point ROI per frame")],
    description="Read a TrackMate XML file (AllSpots / AllTracks / edges, with the spot and track features) into track overlays, the Tracks table and the stored spots of the current image. Lineages (splits / merges / gap-closing links) are kept as the tracks' edges.",
    keywords=["trackmate", "xml", "load tracks", "fiji", "import", "lineage"])
def import_trackmate_xml(ctx, path="", only_filtered=True, units="From file", add_to_manager=False):
    import xml.etree.ElementTree as ET
    img = ctx.img
    if not path or not str(path).strip():
        raise ValueError("Give the TrackMate XML file")
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"{p} does not exist")
    try:
        root = ET.parse(str(p)).getroot()
    except ET.ParseError as e:
        raise ValueError(f"{p.name} is not valid XML: {e}") from e
    model = root.find("Model") if root.tag != "Model" else root
    if model is None:
        raise ValueError(f"{p.name} has no <Model> element – not a TrackMate file")
    # pixel size: the file's image settings, else the image's calibration
    pw = ph = pd = 1.0
    dt = 1.0
    imd = root.find("Settings/ImageData")
    if units == "From file":
        if imd is not None and imd.get("pixelwidth"):
            pw, ph, pd = _attr_float(imd, "pixelwidth", 1.0), _attr_float(imd, "pixelheight", 1.0), _attr_float(imd, "voxeldepth", 1.0)
            dt = _attr_float(imd, "timeinterval", 1.0)
        elif img.calibration.scaled:
            pw, ph, pd = img.calibration.pixel_width, img.calibration.pixel_height, img.calibration.pixel_depth
        pw, ph, pd = (v if v and np.isfinite(v) and v > 0 else 1.0 for v in (pw, ph, pd))
    spots_by_id: dict[int, dict[str, Any]] = {}
    for sif in model.iter("SpotsInFrame"):
        frame_attr = sif.get("frame")
        for el in sif.findall("Spot"):
            sid = int(float(el.get("ID")))
            frame = el.get("FRAME")
            t = int(float(frame)) if frame is not None else (int(float(frame_attr)) if frame_attr is not None else int(round(_attr_float(el, "POSITION_T", 0) / (dt or 1.0))))
            inten = _attr_float(el, "MEAN_INTENSITY_CH1", _attr_float(el, "MEAN_INTENSITY", float("nan")))
            r = _attr_float(el, "RADIUS", MANUAL_RADIUS * pw) / pw
            z = _attr_float(el, "POSITION_Z", 0.0) / pd
            # TrackMate's pixel centre i → Lumen's i + 0.5, in both units modes (z is a slice index in both tools: not shifted)
            sp: dict[str, Any] = {"id": sid, "t": t, "x": _attr_float(el, "POSITION_X", 0.0) / pw + TM_OFFSET, "y": _attr_float(el, "POSITION_Y", 0.0) / ph + TM_OFFSET,
                                  "z": int(round(z)) if np.isfinite(z) else 0, "r": float(r) if np.isfinite(r) and r > 0 else MANUAL_RADIUS,
                                  "quality": _attr_float(el, "QUALITY", 1.0), "intensity": float(inten), "trackmate_id": sid}
            for key, lkey in _TM_SPOT_INTENSITY[1:]:
                v = _attr_float(el, key, _attr_float(el, key.replace("_CH1", ""), float("nan")))
                if np.isfinite(v):
                    sp[lkey] = v
            vis = el.get("VISIBILITY")
            if vis is not None:
                try:
                    sp["visibility"] = int(float(vis))
                except ValueError:
                    pass
            name = el.get("name")
            if name and name != f"ID{sid}":
                sp["name"] = name
            spots_by_id[sid] = sp
    if not spots_by_id:
        raise ValueError(f"{p.name} contains no spots")
    keep: set[int] | None = None
    ft = model.find("FilteredTracks")
    if only_filtered and ft is not None and len(ft):
        keep = {int(float(e.get("TRACK_ID"))) for e in ft.findall("TrackID") if e.get("TRACK_ID") is not None}
    tracks: list[dict[str, Any]] = []
    for tel in model.iter("Track"):
        tid_attr = tel.get("TRACK_ID")
        tid = int(float(tid_attr)) if tid_attr is not None else len(tracks)
        if keep is not None and tid not in keep:
            continue
        edges: list[tuple[int, int]] = []
        for e in tel.findall("Edge"):
            va, vb = e.get("SPOT_SOURCE_ID"), e.get("SPOT_TARGET_ID")
            if va is None or vb is None:
                continue
            a, b = int(float(va)), int(float(vb))
            if a == b or a not in spots_by_id or b not in spots_by_id:
                continue
            if spots_by_id[a]["t"] > spots_by_id[b]["t"]:  # edges always point forward in time
                a, b = b, a
            if (a, b) not in edges:
                edges.append((a, b))
        ids = sorted({i for e in edges for i in e}, key=lambda i: (spots_by_id[i]["t"], i))
        if not ids:
            continue
        tracks.append({"tm_id": tid, "name": tel.get("name") or f"Track_{tid}", "ids": ids, "edges": edges})
    if not tracks:
        raise ValueError(f"{p.name} contains no tracks" + (" (FilteredTracks is empty – untick 'only filtered')" if only_filtered and keep is not None else ""))
    spots: list[dict[str, Any]] = [spots_by_id[i] for i in sorted(spots_by_id)]
    index_of = {int(sp["id"]): k for k, sp in enumerate(spots)}
    out_tracks: list[dict[str, Any]] = []
    n_lineage = 0
    for k, tr in enumerate(tracks):
        idx = [index_of[i] for i in tr["ids"]]
        eidx = [(index_of[a], index_of[b]) for a, b in tr["edges"]]
        stats = _track_stats(img, spots, idx, eidx)
        if stats["N splits"] or stats["N merges"]:
            n_lineage += 1
        out_tracks.append({"id": str(tr["tm_id"]), "name": tr["name"], "color": PALETTE[k % len(PALETTE)], "spots": [spots[i] for i in idx],
                           "edges": [[int(a), int(b)] for a, b in tr["edges"]], "stats": stats, "trackmate_id": tr["tm_id"]})
    s = ctx.session
    _tracks_for_image(ctx, img, create=True)  # park another image's tracks
    s.settings["spots"] = spots
    s.settings["spots_meta"] = {"image_id": img.id, "image_name": img.name, "channel": img.display.c, "radius_px": float(np.median([sp["r"] for sp in spots])),
                                "threshold": 0.0, "detector": "TrackMate XML"}
    s.settings["tracks"] = out_tracks
    s.settings["tracks_image"] = img.id
    _forget_undo(s, img)
    msg = f"Imported {len(out_tracks)} tracks ({sum(len(t['spots']) for t in out_tracks)} linked of {len(spots)} spots) from {p.name}"
    if n_lineage:
        msg += f"; {n_lineage} track{'s' if n_lineage != 1 else ''} with splits / merges"
    tmax = max(sp["t"] for sp in spots)
    if tmax >= img.nt:
        msg += f"; warning: frames up to {tmax} but the image has {img.nt}"
    rois = _spot_rois(img, spots) if add_to_manager else []
    return OpResult(overlay=_all_track_overlay(img, out_tracks), table=_tracks_table(out_tracks), rois_add=rois, message=msg, log=msg)


@op("tracking.tracks_to_rois", "Tracks to ROIs", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("width", "float", 1.0, min=0.5, max=50, step=0.5, label="Line width")],
    description="Add every track as a polyline ROI (its spot positions in frame order; single-spot tracks as point ROIs) to the ROI Manager, named and coloured like the track.",
    keywords=["trackmate", "roi manager", "polyline", "export tracks"])
def tracks_to_rois(ctx, width=1.0):
    img = ctx.img
    tracks = _tracks_for_image(ctx, img)
    if not tracks:
        raise ValueError("No tracks – run Track / Link Spots or create manual tracks first")
    d = img.display
    rois = []
    _ensure_ids(ctx.session.settings.get("spots"), *(tr["spots"] for tr in tracks))
    for tr in tracks:
        branches = _branches(tr)
        for k, br in enumerate(branches):
            pts = [[round(float(sp["x"]), 3), round(float(sp["y"]), 3)] for sp in br]
            if not pts:
                continue
            name = tr["name"] if len(branches) == 1 else f"{tr['name']} (branch {k + 1})"
            pos = {"t": int(br[0]["t"]), "z": int(br[0].get("z", d.z)), "c": int(d.c)}
            if len(pts) == 1:
                rois.append({"type": "point", "points": pts, "name": name, "color": tr["color"], "position": pos, "image_id": img.id})
            else:
                rois.append({"type": "polyline", "points": pts, "width": float(width), "name": name, "color": tr["color"], "position": pos, "image_id": img.id})
    if not rois:
        raise ValueError("The tracks have no points yet")
    msg = f"{len(rois)} track ROIs added to the ROI Manager"
    return OpResult(rois_add=rois, message=msg, log=msg)
