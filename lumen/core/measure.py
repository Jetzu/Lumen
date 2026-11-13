"""ImageJ-compatible measurements (Analyze ▸ Measure / Set Measurements)."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from . import roi as roimod
from .image import LumenImage

ALL_MEASUREMENTS = [
    "area", "mean", "std", "mode", "min", "max", "centroid", "center_of_mass", "perimeter",
    "bounding_rect", "fit_ellipse", "shape_descriptors", "feret", "integrated_density", "median",
    "skewness", "kurtosis", "area_fraction", "stack_position",
]
DEFAULT_MEASUREMENTS = ["area", "mean", "min", "max"]

# Set Measurements options (session.measure_options) beyond the column list; see apply_measure_options()
MEASURE_OPTION_DEFAULTS = {
    "limit_to_threshold": False, "decimals": 3, "display_label": True,
    "invert_y": False,            # Y / YM / BY measured from the bottom edge (ImageJ 'Invert Y coordinates')
    "scientific_notation": False, # formatting hint for the Results table (values stay numeric)
    "add_to_overlay": False,      # Measure adds the measured ROI to the image overlay (done by the measure command)
    "nan_empty_cells": False,     # columns a row cannot provide (e.g. Perim. of a point) are written as NaN instead of left empty
}

# every column a measurement key can produce (used to fill NaN for 'nan_empty_cells')
MEASUREMENT_COLUMNS = {
    "area": ["Area"], "mean": ["Mean"], "std": ["StdDev"], "mode": ["Mode"], "min": ["Min"], "max": ["Max"],
    "centroid": ["X", "Y"], "center_of_mass": ["XM", "YM"], "perimeter": ["Perim."], "bounding_rect": ["BX", "BY", "Width", "Height"],
    "fit_ellipse": ["Major", "Minor", "Angle"], "shape_descriptors": ["Circ.", "AR", "Round", "Solidity"],
    "feret": ["Feret", "FeretAngle", "MinFeret", "FeretX", "FeretY"], "integrated_density": ["IntDen", "RawIntDen"], "median": ["Median"],
    "skewness": ["Skew"], "kurtosis": ["Kurt"], "area_fraction": ["%Area"], "stack_position": ["Ch", "Slice", "Frame"],
}
_Y_COLUMNS = ("Y", "YM", "FeretY")


def apply_measure_options(rows: list[dict[str, Any]], img: LumenImage, options: dict[str, Any] | None,
                          measurements: list[str] | set[str] | None = None) -> list[dict[str, Any]]:
    """Post-process measurement rows according to the Set Measurements options:

    * ``nan_empty_cells`` – every column of the requested measurements that is missing from a row is filled with NaN
    * ``display_label`` False – the Label column is dropped

    ``invert_y`` is applied while measuring (ImageJ's ``Calibration.getY(y, height)``: Y, YM and BY become
    (height − y − 1)·pixel_height; FeretY stays a pixel coordinate), so it is not touched here.
    """
    if not options:
        return rows
    opts = {**MEASURE_OPTION_DEFAULTS, **options}
    if opts.get("nan_empty_cells") and measurements:
        cols = [c for m in measurements for c in MEASUREMENT_COLUMNS.get(m, [])]
        for r in rows:
            for c in cols:
                if c not in r:
                    r[c] = float("nan")
    if opts.get("display_label") is False:
        for r in rows:
            r.pop("Label", None)
    return rows

MEASUREMENT_LABELS = {
    "area": "Area", "mean": "Mean gray value", "std": "Standard deviation", "mode": "Modal gray value",
    "min": "Min gray value", "max": "Max gray value", "centroid": "Centroid", "center_of_mass": "Center of mass",
    "perimeter": "Perimeter", "bounding_rect": "Bounding rectangle", "fit_ellipse": "Fit ellipse",
    "shape_descriptors": "Shape descriptors", "feret": "Feret's diameter", "integrated_density": "Integrated density",
    "median": "Median", "skewness": "Skewness", "kurtosis": "Kurtosis", "area_fraction": "Area fraction",
    "stack_position": "Stack position",
}


def feret_diameters(coords_xy: np.ndarray) -> tuple[float, float, float, float, float]:
    """(max_feret, angle_deg, min_feret, feret_x, feret_y) from boundary/hull points (x, y)."""
    if len(coords_xy) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    pts = np.unique(coords_xy, axis=0)
    if len(pts) == 1:
        return 1.0, 0.0, 1.0, float(pts[0, 0]), float(pts[0, 1])
    try:
        from scipy.spatial import ConvexHull
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            hp = pts[hull.vertices]
        else:
            hp = pts
    except Exception:
        hp = pts
    # max feret: brute force over hull points
    d = hp[:, None, :] - hp[None, :, :]
    dist = np.hypot(d[..., 0], d[..., 1])
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    maxf = float(dist[i, j])
    p1, p2 = hp[i], hp[j]
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    angle = math.degrees(math.atan2(p1[1] - p2[1], p2[0] - p1[0]))
    if angle < 0:
        angle += 180
    # min feret: rotating calipers – min over hull edges of max distance to edge line
    minf = math.inf
    n = len(hp)
    if n >= 2:
        for k in range(n):
            a, b = hp[k], hp[(k + 1) % n]
            e = b - a
            L = math.hypot(e[0], e[1])
            if L == 0:
                continue
            dists = np.abs((hp[:, 0] - a[0]) * e[1] - (hp[:, 1] - a[1]) * e[0]) / L
            minf = min(minf, float(dists.max()))
    if not math.isfinite(minf):
        minf = maxf
    return maxf, angle, minf, float(p1[0]), float(p1[1])


def _ijm():
    """The ImageJ ports (lazy: the ops package imports every op module, and those import this module)."""
    from .ops import _imagej_measure as m
    return m


def _jdiv(a: float, b: float) -> float:
    """Java double division: x/0 → ±Infinity, 0/0 → NaN instead of a ZeroDivisionError."""
    if b == 0:
        return float("nan") if a == 0 or math.isnan(a) else math.copysign(math.inf, a)
    return a / b


def _stat_flags(wanted: set[str]) -> dict[str, bool]:
    """Which optional ``ImageStatistics`` parts a measurement set needs (``Byte/Short/FloatStatistics`` constructors)."""
    return {
        "ellipse": bool(wanted & {"fit_ellipse", "shape_descriptors"}),
        "centroid": "centroid" in wanted,
        "moments": bool(wanted & {"center_of_mass", "skewness", "kurtosis"}),
        "median": "median" in wanted,
        "area_fraction": "area_fraction" in wanted,
    }


def _pixel_stats(vals: np.ndarray, wanted: set[str], cal_area: float, img: LumenImage, hist_bins: int = 256) -> dict[str, Any]:
    """Statistics of a bare value array (``measure_calibrated``: the calibrated values of a selection) with ImageJ's
    ``FloatStatistics`` arithmetic: Area, Mean, StdDev, Mode (256-bin histogram), Min, Max, IntDen, RawIntDen,
    Median, Skew, Kurt – the keys :func:`_build_row` would write for the same measurements."""
    ijm = _ijm()
    vals = np.asarray(vals, dtype=np.float64).ravel()
    finite = vals[np.isfinite(vals)]
    row: dict[str, Any] = {}
    n = int(finite.size)
    if "area" in wanted:
        row["Area"] = n * cal_area
    if n == 0:
        for k in ("Mean", "StdDev", "Mode", "Min", "Max", "IntDen", "RawIntDen", "Median", "Skew", "Kurt"):
            row[k] = float("nan")
        return row
    cal = ijm.Cal(1.0, cal_area, False, 1)
    st = ijm.image_statistics(finite.astype(np.float32)[None, :], None, None, cal=cal, moments=bool(wanted & {"skewness", "kurtosis"}),
                              median="median" in wanted, n_bins=hist_bins)
    if "mean" in wanted:
        row["Mean"] = st.mean
    if "std" in wanted:
        row["StdDev"] = st.std_dev
    if "mode" in wanted:
        row["Mode"] = st.dmode
    if "min" in wanted:
        row["Min"] = st.min
    if "max" in wanted:
        row["Max"] = st.max
    if "integrated_density" in wanted:
        row["IntDen"] = st.area * st.mean
        row["RawIntDen"] = st.pixel_count * st.umean
    if "median" in wanted:
        row["Median"] = st.median
    if "skewness" in wanted:
        row["Skew"] = st.skewness
    if "kurtosis" in wanted:
        row["Kurt"] = st.kurtosis
    return row


def _cal_of(img: LumenImage, invert_y: bool, height: int | None = None):
    cal = img.calibration
    return _ijm().Cal(cal.pixel_width, cal.pixel_height, bool(invert_y), img.height if height is None else height)


def _build_row(st: Any, wanted: set[str], label: str, *, is_area: bool, perimeter: float, feret: tuple[float, ...] | None,
               hull_area: float | None, line_bounds: tuple[float, float, float, float] | None, cal: Any,
               limit_on: bool, rgb: bool, stack: tuple[int, int, int] | None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """``Analyzer.saveResults``: the Results columns for one ImageStatistics in ImageJ's column order.

    ``line_bounds`` (BX/BY/Width/Height from ``roi.getBounds()``, calibrated here) is used for line selections, every
    other ROI reports the statistics' rectangle; ``feret`` = (Feret, FeretAngle, MinFeret, FeretX, FeretY)."""
    row: dict[str, Any] = {"Label": label}
    if "area" in wanted:
        row["Area"] = st.area
    if "mean" in wanted:
        row["Mean"] = st.mean
    if "std" in wanted:
        row["StdDev"] = st.std_dev
    if "mode" in wanted:
        row["Mode"] = st.dmode
    if "min" in wanted:
        row["Min"] = st.min
    if "max" in wanted:
        row["Max"] = st.max
    if "centroid" in wanted:
        row["X"] = st.x_centroid
        row["Y"] = st.y_centroid
    if "center_of_mass" in wanted:
        row["XM"] = st.x_center_of_mass
        row["YM"] = st.y_center_of_mass
    shape = "shape_descriptors" in wanted
    if "perimeter" in wanted:
        row["Perim."] = perimeter
    if "bounding_rect" in wanted:
        if line_bounds is not None:
            bx, by, bw, bh = line_bounds
            row.update(BX=cal.get_x(bx), BY=cal.get_y(by), Width=bw * cal.pw, Height=bh * cal.ph)
        else:
            row.update(BX=st.roi_x, BY=st.roi_y, Width=st.roi_width, Height=st.roi_height)
    if "fit_ellipse" in wanted:
        row["Major"] = st.major
        row["Minor"] = st.minor
        row["Angle"] = st.angle
    if shape:
        circ = 0.0 if perimeter == 0.0 else 4.0 * math.pi * (st.area / (perimeter * perimeter))
        row["Circ."] = min(circ, 1.0)
    if "feret" in wanted and feret is not None:
        row["Feret"] = feret[0]
    if "integrated_density" in wanted:
        row["IntDen"] = st.area * st.mean
    if "median" in wanted:
        row["Median"] = st.median
    if "skewness" in wanted:
        row["Skew"] = st.skewness
    if "kurtosis" in wanted:
        row["Kurt"] = st.kurtosis
    if "area_fraction" in wanted:
        row["%Area"] = st.area_fraction
    if "integrated_density" in wanted:
        row["RawIntDen"] = st.pixel_count * st.umean
    if "stack_position" in wanted and stack is not None:
        c, z, t = stack
        row.update(Ch=c + 1, Slice=z + 1, Frame=t + 1)
    if "feret" in wanted and feret is not None:
        row["FeretX"] = feret[3]
        row["FeretY"] = feret[4]
        row["FeretAngle"] = feret[1]
        row["MinFeret"] = feret[2]
    if shape:
        row["AR"] = _jdiv(st.major, st.minor) if is_area else 0.0
        row["Round"] = _jdiv(4.0 * st.area, math.pi * st.major * st.major) if is_area else 0.0
        row["Solidity"] = _jdiv(float(st.pixel_count), float(hull_area if hull_area is not None else st.pixel_count)) if is_area else float("nan")
    if limit_on and not rgb:
        row["MinThr"] = st.lower_threshold
        row["MaxThr"] = st.upper_threshold
    if extra:
        row.update(extra)
    return row


def _measure_points(img: LumenImage, plane: np.ndarray, roi: dict[str, Any], wanted: set[str], label: str, cal: Any,
                    limit: tuple[float, float] | None, threshold: tuple[float, float] | None, stack: tuple[int, int, int]) -> list[dict[str, Any]]:
    """``Analyzer.measurePoint`` + ``savePoints``: one row per point, the statistics of the 1×1 pixel at the rounded
    position, Area overwritten with 0 and X/Y with the point coordinates (calibrated, invert-Y aware).  ``Value`` (the
    pixel value, Lumen's own column) is kept for scripts."""
    ijm = _ijm()
    H, W = plane.shape
    flags = _stat_flags(wanted)
    rows = []
    pts = roi.get("points", [])
    for k, (x, y) in enumerate(pts):
        xi, yi = ijm.jround(float(x)), ijm.jround(float(y))
        st = ijm.image_statistics(plane, (xi, yi, 1, 1), None, cal=cal, limit=limit, threshold=threshold, **flags)
        fer = ijm.feret_values([float(x)], [float(y)], cal.pw, cal.ph)
        lab = label if len(pts) == 1 else f"{label} (pt {k + 1})"
        row = _build_row(st, wanted, lab, is_area=False, perimeter=0.0, feret=fer, hull_area=0.0, line_bounds=None, cal=cal,
                         limit_on=limit is not None, rgb=bool(img.rgb), stack=stack)
        if "area" in wanted:
            row["Area"] = 0
        row["X"] = cal.get_x(float(x))
        row["Y"] = cal.get_y(float(y))
        vx, vy = int(x), int(y)
        row["Value"] = float(plane[vy, vx]) if 0 <= vy < H and 0 <= vx < W else float("nan")
        rows.append(row)
    return rows


def _measure_line(img: LumenImage, plane: np.ndarray, roi: dict[str, Any], wanted: set[str], label: str, cal: Any,
                  limit: tuple[float, float] | None, stack: tuple[int, int, int]) -> list[dict[str, Any]]:
    """``Analyzer.measureLength``: a line of width ≤ 1 is sampled with ``ImageProcessor.getLine`` (bilinear
    interpolation, ``ProfilePlot``) and measured as a 1-row float image; a wider straight line as the area of its
    ``RotatedRectRoi``.  Only Area/Mean/SD/Mode/Min/Max/Median are computed (XM/YM, %Area, Skew/Kurt stay 0, the
    ellipse 0); the centroid is the midpoint, Perim. the length, Feret from the end points, plus Length and Angle."""
    ijm = _ijm()
    H, W = plane.shape
    pw, ph = cal.pw, cal.ph
    straight = roi["type"] == "line"
    pts = roimod.line_points(roi)
    length = roimod.line_length(roi, pw, ph)
    angle = ijm.float_angle(roi["x1"], roi["y1"], roi["x2"], roi["y2"], pw, ph) if straight else None
    extra: dict[str, Any] = {"Length": length}
    if angle is not None:
        extra["Angle"] = angle
    if not wanted & {"area", "mean", "std", "mode", "min", "max", "centroid", "median"}:
        row = {"Label": label, **extra}
        if "stack_position" in wanted:
            c, z, t = stack
            row.update(Ch=c + 1, Slice=z + 1, Frame=t + 1)
        return [row]
    width = float(roi.get("width", 1) or 1)
    lw = ijm.jround(width)
    if lw > 1:
        if straight:
            rect, mask = roimod.roi_rect_mask(roi, (H, W))
        else:
            rect, mask = (0, 0, W, H), roimod.to_mask(roi, (H, W))
        st = ijm.image_statistics(plane, rect, mask, cal=cal, limit=limit, median=True)
    else:
        if straight:
            values = ijm.get_line(plane, roi["x1"], roi["y1"], roi["x2"], roi["y2"], interpolate=True)
        else:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            values = ijm.irregular_profile(plane, xs, ys, interpolate=True)
        prof: np.ndarray = values.astype(np.float32)[None, :]          # new FloatProcessor(values.length, 1, values)
        if limit is not None and plane.dtype in (np.uint8, np.uint16):  # convertToOriginalDepth (no scaling)
            top = 255.0 if plane.dtype == np.uint8 else 65535.0
            prof = np.floor(np.clip(prof.astype(np.float64), 0.0, top) + 0.5).astype(plane.dtype)
        st = ijm.image_statistics(prof, None, None, cal=cal, limit=limit, median=True)
    if straight and "centroid" in wanted:
        (x1, y1), (x2, y2) = pts[0], pts[-1]
        st.x_centroid = cal.get_x(x1 + (x2 - x1) / 2.0)
        st.y_centroid = cal.get_y(y1 + (y2 - y1) / 2.0)
    fer = roimod.feret_values(roi, pw, ph) if "feret" in wanted else None
    bounds = roimod.int_bounds(roi)
    return [_build_row(st, wanted, label, is_area=False, perimeter=length, feret=fer, hull_area=None, line_bounds=bounds, cal=cal,
                       limit_on=limit is not None, rgb=bool(img.rgb), stack=stack, extra=extra)]


def _measure_area(img: LumenImage, plane: np.ndarray, roi: dict[str, Any] | None, wanted: set[str], label: str, cal: Any,
                  limit: tuple[float, float] | None, threshold: tuple[float, float] | None, stack: tuple[int, int, int]) -> list[dict[str, Any]]:
    """``Analyzer.measure`` for an area selection (or the whole plane): ``ImageStatistics`` over the ROI's rectangle
    and mask, perimeter / convex hull / Feret from the ROI's own geometry (``Roi.getLength``, ``getConvexHull``,
    ``getFeretValues``); without a selection the perimeter is 2·(W + H) in pixels, as ImageJ reports it."""
    ijm = _ijm()
    H, W = plane.shape
    pw, ph = cal.pw, cal.ph
    rect, mask = roimod.roi_rect_mask(roi, (H, W))
    st = ijm.image_statistics(plane, rect, mask, cal=cal, limit=limit, threshold=threshold, **_stat_flags(wanted))
    need_per = bool(wanted & {"perimeter", "shape_descriptors"})
    if roi is None:
        perimeter = float(W * 2 + H * 2) if need_per else 0.0
        hull_area: float | None = float(st.pixel_count)
    else:
        perimeter = roimod.perimeter(roi, pw, ph) if need_per else 0.0
        hull_area = roimod.convex_hull_area(roi, (H, W)) if "shape_descriptors" in wanted else None
    fer = roimod.feret_values(roi, pw, ph, (H, W)) if "feret" in wanted else None
    return [_build_row(st, wanted, label, is_area=True, perimeter=perimeter, feret=fer, hull_area=hull_area, line_bounds=None, cal=cal,
                       limit_on=limit is not None, rgb=bool(img.rgb), stack=stack)]


def measure_roi(img: LumenImage, roi: dict[str, Any] | None, measurements: list[str] | None = None,
                t: int | None = None, z: int | None = None, c: int | None = None,
                limit_to_threshold: tuple[float, float] | None = None, label: str | None = None,
                options: dict[str, Any] | None = None, threshold: tuple[float, float] | None = None) -> list[dict[str, Any]]:
    """Measure one ROI (or the whole plane) on the plane (t, z, c) exactly as ImageJ's Analyzer does.  Returns the
    list of result rows (one per point of a point selection).

    ``limit_to_threshold`` – the threshold when 'Limit to threshold' is on (statistics over the thresholded pixels,
    MinThr/MaxThr columns); ``threshold`` – the image's threshold when it is set but not limiting (ImageJ's %Area is
    the fraction of thresholded pixels whenever a threshold exists).  ``options`` (session.measure_options) applies
    invert_y (inside the statistics), nan_empty_cells and display_label, see apply_measure_options().
    """
    rows = _measure_roi(img, roi, measurements, t, z, c, limit_to_threshold, label, threshold=threshold,
                        invert_y=bool(options and options.get("invert_y")))
    return apply_measure_options(rows, img, options, measurements or DEFAULT_MEASUREMENTS)


def _measure_roi(img: LumenImage, roi: dict[str, Any] | None, measurements: list[str] | None = None,
                 t: int | None = None, z: int | None = None, c: int | None = None,
                 limit_to_threshold: tuple[float, float] | None = None, label: str | None = None,
                 threshold: tuple[float, float] | None = None, invert_y: bool = False) -> list[dict[str, Any]]:
    wanted = set(measurements or DEFAULT_MEASUREMENTS)
    d = img.display
    # a ROI's stored position may refer to another image's dimensions (ROI Manager entries survive image switches):
    # clamp every axis into this image instead of raising an IndexError
    t = d.t if t is None else max(0, min(int(t), img.nt - 1))
    z = d.z if z is None else max(0, min(int(z), img.nz - 1))
    c = d.c if c is None else max(0, min(int(c), img.nc - 1))
    plane = img.data[t, z, c]
    cal = _cal_of(img, invert_y)
    lab = label or (roi.get("name") if roi else None) or img.name
    limit = (float(limit_to_threshold[0]), float(limit_to_threshold[1])) if limit_to_threshold is not None else None
    thr = limit if limit is not None else ((float(threshold[0]), float(threshold[1])) if threshold is not None else None)
    stack = (c, z, t)
    if roi is not None and roi.get("type") == "point":
        return _measure_points(img, plane, roi, wanted, lab, cal, limit, thr, stack)
    if roi is not None and roimod.is_line(roi):
        return _measure_line(img, plane, roi, wanted, lab, cal, limit, stack)
    return _measure_area(img, plane, roi, wanted, lab, cal, limit, thr, stack)


def particles_from_labels(labels: np.ndarray) -> list[Any]:
    """Every label of a label image as an ImageJ-style particle (bounds, mask, wand outline, first pixel), in label
    order – the labels of ``skimage.measure.label`` / ``ndimage.label`` are already numbered in raster order of their
    first pixel, which is ImageJ's particle order."""
    ijm = _ijm()
    labels = np.asarray(labels)
    out = []
    for lab_value, sl in enumerate(ndi.find_objects(labels), 1):
        if sl is None:
            continue
        ys, xs = sl
        y0, x0 = int(ys.start), int(xs.start)
        sub = labels[ys, xs] == lab_value
        h, w = sub.shape
        yy, xx = np.nonzero(sub)
        k = int(np.argmin(yy * w + xx))
        sx, sy = int(xx[k]) + x0, int(yy[k]) + y0
        outline = ijm.wand_outline(sub, int(xx[k]), int(yy[k]))
        if outline is None:
            ox, oy = [x0, x0 + w, x0 + w, x0], [y0, y0, y0 + h, y0 + h]
        else:
            ox = [v + x0 for v in outline[0]]
            oy = [v + y0 for v in outline[1]]
        out.append(ijm.Particle((x0, y0, w, h), sub if (w > 1 and h > 1) else None, ox, oy, (sx, sy), int(sub.sum())))
    return out


def measure_particles(img: LumenImage, particles: list[Any], measurements: list[str], t: int, z: int, c: int,
                      intensity_plane: np.ndarray | None = None, options: dict[str, Any] | None = None,
                      threshold: tuple[float, float] | None = None, first_number: int = 1) -> list[dict[str, Any]]:
    """``ParticleAnalyzer.analyzeParticle`` → ``Analyzer.saveResults`` for a list of particles
    (:func:`lumen.core.ops._imagej_measure.find_particles` or :func:`particles_from_labels`): the statistics of the
    particle's mask on ``intensity_plane`` (default: the image), the traced perimeter, the convex hull of the traced
    outline (Feret, Solidity) and the EllipseFitter of the mask.  'Limit to threshold' never applies (ImageJ clears
    LIMIT); ``threshold`` only feeds %Area.  ``options`` as in :func:`measure_roi`."""
    ijm = _ijm()
    wanted = set(measurements)
    plane = intensity_plane if intensity_plane is not None else img.data[t, z, c]
    cal = _cal_of(img, bool(options and options.get("invert_y")), plane.shape[0])
    pw, ph = cal.pw, cal.ph
    flags = _stat_flags(wanted)
    need_per = bool(wanted & {"perimeter", "shape_descriptors"})
    thr = (float(threshold[0]), float(threshold[1])) if threshold is not None else None
    rows = []
    for k, part in enumerate(particles, first_number):
        st = ijm.image_statistics(plane, part.bounds, part.mask, cal=cal, limit=None, threshold=thr, **flags)
        perimeter = ijm.traced_perimeter(part.outline_x, part.outline_y, pw, ph) if need_per else 0.0
        hull_area = ijm.hull_int_area(part.outline_x, part.outline_y) if "shape_descriptors" in wanted else None
        fer = None
        if "feret" in wanted:
            hx, hy = ijm.convex_hull(part.outline_x, part.outline_y)
            fer = ijm.feret_values(hx, hy, pw, ph)
        rows.append(_build_row(st, wanted, f"{img.name}:{k}", is_area=True, perimeter=perimeter, feret=fer, hull_area=hull_area,
                               line_bounds=None, cal=cal, limit_on=False, rgb=bool(img.rgb), stack=(c, z, t)))
    return apply_measure_options(rows, img, options, measurements)


def measure_labels(img: LumenImage, labels: np.ndarray, measurements: list[str], t: int, z: int, c: int,
                   intensity_plane: np.ndarray | None = None, options: dict[str, Any] | None = None,
                   threshold: tuple[float, float] | None = None) -> list[dict[str, Any]]:
    """Measure every labelled particle of a label image like Analyze Particles (see :func:`measure_particles`)."""
    return measure_particles(img, particles_from_labels(labels), measurements, t, z, c, intensity_plane, options, threshold)
