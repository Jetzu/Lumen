"""Image ▸ Transform extras — landmark (interactive-style) rigid / similarity / affine / perspective
transforms, perspective rectification, shear, (log-)polar warps, and keyframe ROI interpolation
for time-lapse stacks (Image ▸ Stacks ▸ Tools).

Point coordinates follow Lumen's ROI convention (pixel ``i`` spans ``[i, i + 1)``); they are
shifted by −0.5 to scikit-image's pixel-centre convention before a transform is estimated.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from skimage.transform import ProjectiveTransform, warp, warp_polar

from .. import roi as roimod
from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op, cast_to
from .registration import INTERP, ORDER_MAP
from .registration_extra import CORRESPONDENCES_KEY, FEATURE_MODELS, _apply_matrix, estimate_point_transform, transform_params

CAT = "Image/Transform"
LANDMARK_MODELS = ["Rigid body", "Similarity", "Affine", "Projective", "Translation"]


# ============================================================================ helpers
def parse_points(text: str) -> np.ndarray:
    """'x1,y1; x2,y2; …' (also newline / space separated, optional brackets) → (N, 2) float array."""
    s = str(text or "").strip()
    if not s:
        return np.zeros((0, 2))
    nums = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)]
    if len(nums) % 2:
        raise ValueError("Point list needs pairs of coordinates: 'x1,y1; x2,y2; …'")
    return np.asarray(nums, float).reshape(-1, 2)


def _points_of_roi(roi: dict[str, Any] | None) -> np.ndarray:
    if not roi:
        return np.zeros((0, 2))
    t = roi.get("type")
    if t in ("point", "polygon", "polyline", "freehand"):
        return np.asarray(roi.get("points", []), float).reshape(-1, 2)
    if t == "line":
        return np.array([[roi["x1"], roi["y1"]], [roi["x2"], roi["y2"]]], float)
    if t == "mask":  # every part of a composite mask, not only the largest contour
        parts = roimod.outlines(roi)
        return np.vstack(parts) if parts else np.zeros((0, 2))
    if t in ("rect", "oval"):
        return roimod.polygon_points(roi) if t != "oval" else np.asarray([[roi["x"], roi["y"]], [roi["x"] + roi["w"], roi["y"]], [roi["x"] + roi["w"], roi["y"] + roi["h"]], [roi["x"], roi["y"] + roi["h"]]], float)
    return np.zeros((0, 2))


def _manager_roi(ctx, name: str) -> dict[str, Any] | None:
    if not name:
        return None
    for r in ctx.session.roi_manager.rois:
        if r.get("name") == name or r.get("id") == name:
            return r
    raise ValueError(f"No ROI named {name!r} in the ROI Manager")


def _warp_all(img: LumenImage, inverse_map, out_shape: tuple[int, int], order: int) -> np.ndarray:
    out = np.zeros((img.nt, img.nz, img.nc) + tuple(out_shape), np.float32)
    for t, z, c in img.iter_planes():
        out[t, z, c] = warp(img.data[t, z, c].astype(np.float32), inverse_map, output_shape=out_shape, order=order, mode="constant", cval=0.0, preserve_range=True)
    return out


def _matrix_result(ctx, img: LumenImage, m: np.ndarray, out_shape: tuple[int, int], order: int, create_new: bool, name: str, rows: list[dict[str, Any]] | None, msg: str) -> OpResult:
    tf = ProjectiveTransform(matrix=m)
    data = cast_to(_warp_all(img, tf.inverse, out_shape, order), img.data.dtype)
    if create_new:
        res = img.copy(name=name, data=data)
        res.overlay = []
        res.sync_channels()
        return OpResult(new_images=[res], results=rows, message=msg, log=msg)
    return OpResult(data=data, rgb=img.rgb, results=rows, clear_roi=True, message=msg, log=msg)


# ============================================================================ landmark transforms
@op("image.transform_landmarks", "Landmark Transform (Rigid / Similarity / Affine / Perspective)", CAT, kind="image", preview=False, scope_param=False, roi_aware=False,
    params=[Param("model", "choice", "Affine", choices=LANDMARK_MODELS, label="Transformation"),
            Param("target_points", "text", "", help="Target positions 'x1,y1; x2,y2; …' matching the source points in order (empty = use target_roi / stored correspondences)"),
            Param("source_points", "text", "", help="Source positions (empty = the current point / polygon selection, or source_roi)"),
            Param("source_roi", "string", "", help="Name of a point ROI in the ROI Manager with the source points"),
            Param("target_roi", "string", "", help="Name of a point ROI in the ROI Manager with the target points"),
            Param("width", "int", 0, min=0, help="Output width (0 = same as the image)"), Param("height", "int", 0, min=0, help="Output height (0 = same)"),
            Param("interpolation", "choice", "Linear", choices=INTERP), Param("create_new", "bool", True, label="Create new image")],
    description="Warp the image so that the source landmarks (point selection) land on the target landmarks: rigid, similarity, affine or perspective (least squares over all pairs). Without explicit points the correspondences from 'Extract SIFT Correspondences' are used. The transform parameters go to the Results table.",
    keywords=["interactive affine", "interactive rigid", "interactive similarity", "interactive perspective", "landmarks", "warp", "register points"])
def transform_landmarks(ctx, model="Affine", target_points="", source_points="", source_roi="", target_roi="", width=0, height=0, interpolation="Linear", create_new=True):
    img = ctx.img
    src = parse_points(source_points)
    if src.size == 0:
        src = _points_of_roi(_manager_roi(ctx, source_roi)) if source_roi else _points_of_roi(ctx.roi)
    dst = parse_points(target_points)
    if dst.size == 0 and target_roi:
        dst = _points_of_roi(_manager_roi(ctx, target_roi))
    if src.size == 0 and dst.size == 0:
        corr = ctx.session.settings.get(CORRESPONDENCES_KEY)
        if corr and corr.get("source") == img.id:
            src, dst = np.asarray(corr["src"], float), np.asarray(corr["dst"], float)
    if src.size == 0:
        raise ValueError("Give the source landmarks: a point (or polygon) selection, source_points or source_roi")
    if dst.size == 0:
        raise ValueError("Give the target landmarks: target_points 'x1,y1; x2,y2; …' or target_roi")
    if len(src) != len(dst):
        raise ValueError(f"{len(src)} source points but {len(dst)} target points – they must correspond one to one")
    m, _ = estimate_point_transform(src - 0.5, dst - 0.5, model, use_ransac=False)
    order = ORDER_MAP[interpolation]
    out_shape = (int(height) if height > 0 else img.height, int(width) if width > 0 else img.width)
    resid = np.hypot(*(_apply_matrix(m, src - 0.5) - (dst - 0.5)).T)
    params = transform_params(m)
    row = {"Label": img.name, "Model": model, "N landmarks": int(len(src)), **params, "RMS residual": float(np.sqrt(np.mean(resid ** 2)))}
    msg = f"{model} transform from {len(src)} landmarks: dx {params['dx']:+.2f}, dy {params['dy']:+.2f}, angle {params['Angle']:+.2f}°, scale {params['Scale']:.3f} (RMS residual {row['RMS residual']:.2f} px)"
    ctx.session.settings["landmark_transform"] = {"matrix": m.tolist(), "model": model}
    return _matrix_result(ctx, img, m, out_shape, order, create_new, f"{img.name} ({model.lower()})", [row], msg)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Four points → top-left, top-right, bottom-right, bottom-left."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])  # clockwise in image coordinates (y down)
    order = np.argsort(ang)
    pts = pts[order]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))  # top-left = smallest x + y
    return np.roll(pts, -start, axis=0)


@op("image.perspective", "Perspective Correction (4 corners)", CAT, kind="image", preview=False, scope_param=False, roi_aware=False,
    params=[Param("corners", "text", "", help="Four corner points 'x,y; x,y; x,y; x,y' (empty = the current 4-point / polygon selection); any order"),
            Param("width", "int", 0, min=0, help="Output width (0 = from the quadrilateral)"), Param("height", "int", 0, min=0, help="Output height (0 = from the quadrilateral)"),
            Param("interpolation", "choice", "Linear", choices=INTERP), Param("create_new", "bool", True, label="Create new image")],
    description="Rectify a perspective-distorted rectangle: the four corners of the selection (or given points) are mapped onto a width × height rectangle.",
    keywords=["perspective", "rectify", "keystone", "homography", "quadrilateral"])
def perspective(ctx, corners="", width=0, height=0, interpolation="Linear", create_new=True):
    img = ctx.img
    pts = parse_points(corners)
    if pts.size == 0:
        pts = _points_of_roi(ctx.roi)
    if len(pts) != 4:
        raise ValueError(f"Perspective correction needs exactly 4 corner points ({len(pts)} given)")
    q = _order_corners(pts) - 0.5  # the corners are region edges: pixel-centre coordinates of the outer pixel borders
    w_est = (np.hypot(*(q[1] - q[0])) + np.hypot(*(q[2] - q[3]))) / 2
    h_est = (np.hypot(*(q[3] - q[0])) + np.hypot(*(q[2] - q[1]))) / 2
    W = int(width) if width > 0 else max(int(round(w_est)), 2)
    H = int(height) if height > 0 else max(int(round(h_est)), 2)
    # the quadrilateral's edges land on the output image's outer borders (an axis-aligned rectangle = a plain crop)
    dst = np.array([[-0.5, -0.5], [W - 0.5, -0.5], [W - 0.5, H - 0.5], [-0.5, H - 0.5]], float)
    m, _ = estimate_point_transform(q, dst, "Projective", use_ransac=False)
    order = ORDER_MAP[interpolation]
    msg = f"Rectified the quadrilateral to {W} × {H} px"
    return _matrix_result(ctx, img, m, (H, W), order, create_new, f"Rectified {img.name}", None, msg)


@op("image.shear", "Shear", CAT, kind="image", preview=True, scope_param=False, roi_aware=False,
    params=[Param("shear_x", "float", 0.2, min=-10, max=10, step=0.05, label="Horizontal shear", help="x' = x + shear_x · y (relative to the image centre)"),
            Param("shear_y", "float", 0.0, min=-10, max=10, step=0.05, label="Vertical shear", help="y' = y + shear_y · x"),
            Param("enlarge", "bool", False, help="Enlarge the canvas to fit the sheared image"), Param("interpolation", "choice", "Linear", choices=INTERP)],
    description="Shear the image horizontally and/or vertically about its centre.", keywords=["skew", "slant", "affine"])
def shear(ctx, shear_x=0.2, shear_y=0.0, enlarge=False, interpolation="Linear"):
    img = ctx.img
    H, W = img.height, img.width
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    lin = np.array([[1.0, float(shear_x)], [float(shear_y), 1.0]])
    m = np.eye(3)
    m[:2, :2] = lin
    m[:2, 2] = np.array([cx, cy]) - lin @ np.array([cx, cy])
    out_shape = (H, W)
    if enlarge:
        corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], float)
        tc = _apply_matrix(m, corners)
        x0, y0 = tc.min(axis=0)
        x1, y1 = tc.max(axis=0)
        W2, H2 = int(math.ceil(x1 - x0)) + 1, int(math.ceil(y1 - y0)) + 1
        shift = np.eye(3)
        shift[0, 2], shift[1, 2] = -x0, -y0
        m = shift @ m
        out_shape = (H2, W2)
    order = ORDER_MAP[interpolation]
    msg = f"Sheared by ({shear_x:g}, {shear_y:g})" + (f"; canvas {out_shape[1]} × {out_shape[0]}" if enlarge else "")
    return _matrix_result(ctx, img, m, out_shape, order, False, img.name, None, msg)


@op("image.warp_polar", "Polar Transform", CAT, kind="image", preview=False, scope_param=False, roi_aware=False,
    params=[Param("scaling", "choice", "linear", choices=["linear", "log"], help="log = log-polar transform"),
            Param("center_x", "float", -1, step=1, help="Centre (−1 = the selection centre, else the image centre)"), Param("center_y", "float", -1, step=1),
            Param("radius", "float", 0, min=0, step=1, help="Maximum radius (0 = to the farthest image corner)"),
            Param("angles", "int", 360, min=8, max=3600, label="Output height (angle samples)"), Param("radii", "int", 0, min=0, label="Output width (radius samples; 0 = radius)"),
            Param("interpolation", "choice", "Linear", choices=INTERP)],
    description="Polar or log-polar transform about a centre: rows = angle (0–360°, counter-clockwise from +x), columns = radius. Rotations become shifts along the angle axis (and scalings along the radius axis in log-polar).",
    keywords=["log-polar", "polar", "radial", "unwrap", "rotation invariant"])
def warp_polar_op(ctx, scaling="linear", center_x=-1, center_y=-1, radius=0, angles=360, radii=0, interpolation="Linear"):
    img = ctx.img
    H, W = img.height, img.width
    if center_x < 0 or center_y < 0:
        if ctx.roi is not None:
            bx, by, bw, bh = roimod.bounding_box(ctx.roi)
            cx, cy = bx + bw / 2.0, by + bh / 2.0
        else:
            cx, cy = W / 2.0, H / 2.0
    else:
        cx, cy = float(center_x), float(center_y)
    cx, cy = cx - 0.5, cy - 0.5  # pixel-centre coordinates
    if radius <= 0:
        radius = max(math.hypot(x - cx, y - cy) for x in (0, W - 1) for y in (0, H - 1))
    n_r = int(radii) if radii > 0 else max(int(round(radius)), 2)
    out_shape = (int(angles), n_r)
    order = ORDER_MAP[interpolation]
    out = np.zeros((img.nt, img.nz, img.nc) + out_shape, np.float32)
    for t, z, c in img.iter_planes():
        out[t, z, c] = warp_polar(img.data[t, z, c].astype(np.float32), center=(cy, cx), radius=float(radius), output_shape=out_shape, scaling=scaling, order=order, mode="constant", cval=0.0, preserve_range=True)
    res = img.copy(name=f"{'Log-polar' if scaling == 'log' else 'Polar'} {img.name}", data=cast_to(out, img.data.dtype))
    res.overlay = []
    res.calibration = Calibration(pixel_width=(float(radius) / n_r) * img.calibration.pixel_width if scaling == "linear" else 1.0,
                                  pixel_height=360.0 / int(angles), unit=img.calibration.unit if scaling == "linear" else "pixel",
                                  frame_interval=img.calibration.frame_interval, time_unit=img.calibration.time_unit)
    res.metadata.update({"polar": scaling, "center": [cx + 0.5, cy + 0.5], "radius": float(radius), "x_axis": "radius" + (" (log)" if scaling == "log" else ""), "y_axis": "angle (°)"})
    res.sync_channels()
    msg = f"{'Log-polar' if scaling == 'log' else 'Polar'} transform about ({cx + 0.5:g}, {cy + 0.5:g}), radius {radius:.1f}: {out_shape[1]} × {out_shape[0]}"
    return OpResult(new_images=[res], message=msg, log=msg)


# ============================================================================ keyframe ROI interpolation
def _resample_closed(pts: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to n vertices equally spaced along its perimeter."""
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0) if len(pts) else np.zeros((n, 2))
    closed = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)
    s = np.linspace(0, total, n, endpoint=False)
    return np.stack([np.interp(s, cum, closed[:, 0]), np.interp(s, cum, closed[:, 1])], axis=1)


def _resample_open(pts: np.ndarray, n: int) -> np.ndarray:
    pts = np.asarray(pts, float)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0) if len(pts) else np.zeros((n, 2))
    seg = np.hypot(*np.diff(pts, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)
    s = np.linspace(0, total, n)
    return np.stack([np.interp(s, cum, pts[:, 0]), np.interp(s, cum, pts[:, 1])], axis=1)


def _signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _align_closed(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Re-index closed polygon b (same vertex count as a) so that vertices correspond to a: same orientation, best cyclic start."""
    if _signed_area(a) * _signed_area(b) < 0:
        b = b[::-1]
    best, best_d = b, np.inf
    for k in range(len(b)):
        rb = np.roll(b, -k, axis=0)
        dsum = float(np.sum(np.hypot(*(rb - a).T)))
        if dsum < best_d:
            best, best_d = rb, dsum
    return best


def interpolate_roi(a: dict[str, Any], b: dict[str, Any], w: float, n_vertices: int = 0) -> dict[str, Any]:
    """ROI between keyframes a (w = 0) and b (w = 1); same type required (mask ROIs become polygons)."""
    ta, tb = a["type"], b["type"]
    for r in (a, b):
        if r["type"] == "mask" and len(roimod.outlines(r)) > 1:
            raise ValueError("Keyframe interpolation needs single-part selections (a composite mask has several outlines)")
    if ta == "mask":
        a = {"type": "polygon", "points": roimod.polygon_points(a).tolist()}
        ta = "polygon"
    if tb == "mask":
        b = {"type": "polygon", "points": roimod.polygon_points(b).tolist()}
        tb = "polygon"
    if ta != tb and {ta, tb} != {"polygon", "freehand"}:
        raise ValueError(f"Keyframe ROIs must have the same type ({ta} vs {tb})")
    out: dict[str, Any] = {"type": ta}
    if ta in ("rect", "oval"):
        for k in ("x", "y", "w", "h"):
            out[k] = (1 - w) * float(a[k]) + w * float(b[k])
    elif ta == "line":
        for k in ("x1", "y1", "x2", "y2"):
            out[k] = (1 - w) * float(a[k]) + w * float(b[k])
        out["width"] = a.get("width", 1)
    elif ta == "point":
        pa, pb = np.asarray(a["points"], float), np.asarray(b["points"], float)
        if len(pa) != len(pb):
            raise ValueError(f"Point keyframes have different point counts ({len(pa)} vs {len(pb)})")
        out["points"] = ((1 - w) * pa + w * pb).round(3).tolist()
    elif ta in ("polygon", "freehand"):
        pa, pb = np.asarray(a["points"], float), np.asarray(b["points"], float)
        n = int(n_vertices) if n_vertices > 0 else max(len(pa), len(pb))
        if len(pa) != len(pb) or n_vertices > 0:
            pa, pb = _resample_closed(pa, n), _resample_closed(pb, n)
        pb = _align_closed(pa, pb)
        out["points"] = ((1 - w) * pa + w * pb).round(3).tolist()
    elif ta == "polyline":
        pa, pb = np.asarray(a["points"], float), np.asarray(b["points"], float)
        n = int(n_vertices) if n_vertices > 0 else max(len(pa), len(pb))
        if len(pa) != len(pb) or n_vertices > 0:
            pa, pb = _resample_open(pa, n), _resample_open(pb, n)
        if np.sum(np.hypot(*(pb[::-1] - pa).T)) < np.sum(np.hypot(*(pb - pa).T)):
            pb = pb[::-1]
        out["points"] = ((1 - w) * pa + w * pb).round(3).tolist()
        out["width"] = a.get("width", 1)
    else:
        raise ValueError(f"Cannot interpolate ROIs of type {ta!r}")
    for k in ("color", "stroke_width"):
        if k in a:
            out[k] = a[k]
    return out


@op("stacks.interpolate_rois", "Interpolate ROIs (keyframes)", "Image/Stacks", kind="analysis", preview=False, roi_aware=False, menu="Image/Stacks/Tools",
    params=[Param("axis", "choice", "T", choices=["T", "Z"], help="Stack axis along which the keyframes are placed"),
            Param("name_filter", "string", "", help="Only ROIs whose name starts with this text (empty = every ROI of the image in the manager)"),
            Param("n_vertices", "int", 0, min=0, help="Polygons / polylines: resample to this many vertices (0 = the larger keyframe count)"),
            Param("prefix", "string", "", help="Name of the interpolated ROIs (default: '<keyframe name> t<frame>')")],
    description="LOI / shape interpolator: the ROIs of the ROI Manager on different frames (slices) of the current image are keyframes; the ROIs for the frames in between are interpolated (rectangles / ovals / lines by their parameters, polygons and polylines by vertex interpolation after resampling, points one by one) and added to the manager.",
    keywords=["loi interpolator", "circle interpolator", "keyframe", "morph", "tween", "time-lapse", "roi"])
def interpolate_rois(ctx, axis="T", name_filter="", n_vertices=0, prefix=""):
    img = ctx.img
    s = ctx.session
    d = img.display
    ax = "t" if axis == "T" else "z"
    keys = []
    for r in s.roi_manager.rois:
        if r.get("image_id") not in (None, img.id):
            continue
        if name_filter and not str(r.get("name", "")).startswith(name_filter):
            continue
        pos = r.get("position") or {}
        if ax not in pos:
            continue
        keys.append((int(pos[ax]), r))
    if len(keys) < 2:
        raise ValueError("Needs at least two keyframe ROIs on different frames in the ROI Manager" + (f" (names starting with {name_filter!r})" if name_filter else ""))
    keys.sort(key=lambda kr: kr[0])
    frames = [k for k, _ in keys]
    if len(set(frames)) != len(frames):
        dup = next(f for f in frames if frames.count(f) > 1)
        raise ValueError(f"Two keyframe ROIs on frame {dup + 1}: use the name filter to pick one ROI series")
    added = []
    for (fa, ra), (fb, rb) in zip(keys[:-1], keys[1:]):
        for f in range(fa + 1, fb):
            w = (f - fa) / (fb - fa)
            roi = interpolate_roi(ra, rb, w, n_vertices)
            roi["name"] = f"{prefix}{axis.lower()}{f + 1}" if prefix else f"{ra.get('name', 'ROI')} {axis.lower()}{f + 1}"
            pos_a = ra.get("position") or {}
            roi["position"] = {"t": f if ax == "t" else int(pos_a.get("t", d.t)), "z": f if ax == "z" else int(pos_a.get("z", d.z)), "c": int(pos_a.get("c", d.c))}
            roi["image_id"] = img.id
            added.append(roi)
    if not added:
        raise ValueError("The keyframes are on consecutive frames – nothing to interpolate")
    msg = f"Interpolated {len(added)} ROIs between {len(keys)} keyframes (frames {frames[0] + 1}–{frames[-1] + 1})"
    return OpResult(rois_add=added, message=msg, log=msg)
