"""Plugins ▸ MorphoLibJ — region analysis, intensity measurements, label tools and morphological segmentation.

A re-implementation of the most used MorphoLibJ commands (Legland, Arganda-Carreras & Andrey 2016)
on top of numpy / scipy / scikit-image:

* ``morpho.analyze_regions``: per-label geometry of a label image (area, Crofton perimeter,
  circularity, Euler number, bounding box, centroid, equivalent ellipse, convexity, Feret,
  oriented box, geodesic diameter, tortuosity, largest inscribed circle, average thickness)
  with optional overlays.
* ``morpho.intensity_measurements``: per-label gray statistics of a second image (2-D or 3-D).
* label tools: keep / remove the largest label, crop a label, replace labels, label boundaries,
  constrained label dilation, label morphological filters, the region adjacency graph and
  label overlap measures between two label images.
* ``morpho.morphological_segmentation``: gradient + extended minima + watershed.

The rest of MorphoLibJ (Labels To RGB, Assign Measure to Label, Microstructure Analysis, Analyze
Regions 3D, the Label Edition operations, the gray-level morphological filters, minima / maxima,
chamfer and geodesic distance maps, marker-controlled watershed, reconstruction, overlays) lives in
``morpho_extra.py``; the "Edit labels" guided workflow (``workflows.py``) strings the label edition
commands together.

Conventions: label images are read with ``_as_labels`` (any dtype; a 0/255 mask is labelled into
its 8-connected components first); results are calibrated with the image's pixel size; angles
follow ImageJ (degrees, counter-clockwise, y axis pointing up, 0 ≤ angle < 180).  Label-editing
commands take ``label = 0`` / an empty label list as "the label picked with the value picker":
the picker (I) stores the clicked value as the session's foreground value (``edit.set_foreground``).
"""
from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage import filters as skfilt, measure as skmeas, morphology as skm, segmentation as skseg

from .. import roi as roimod
from ..image import LumenImage
from ..luts import get_lut
from ..measure import feret_diameters
from ..registry import Param, OpResult, op
from ._util import back_to, binary_image, label_image
from .edit_extra import min_area_rect

CAT = "Plugins/MorphoLibJ"

REGION_FEATURES = ["Area", "Perimeter", "Circularity", "Euler number", "Bounding box", "Centroid", "Equivalent ellipse",
                   "Ellipse elongation", "Convexity", "Max Feret diameter", "Oriented box", "Geodesic diameter", "Tortuosity",
                   "Max inscribed circle", "Average thickness"]
REGION_OVERLAYS = ["Equivalent ellipses", "Oriented boxes", "Inscribed circles", "Geodesic paths"]
INTENSITY_MEASUREMENTS = ["Mean", "StdDev", "Max", "Min", "Median", "Mode", "Skewness", "Kurtosis", "NumberOfVoxels", "Volume"]
MODES_2D_3D = ["2D (current plane)", "3D (whole stack)"]
BOUNDARY_MODES = ["inner", "outer", "thick"]
LABEL_FILTERS = ["Dilation", "Erosion", "Opening", "Closing"]
GRADIENTS = ["Morphological gradient", "Sobel"]
SEGMENTATION_INPUTS = ["Object image", "Gradient image"]
SEGMENTATION_OUTPUTS = ["Overlaid basins", "Overlaid dams", "Catchment basins", "Watershed lines"]
OVERLAP_MATCHING = ["Same label value", "Best overlap"]
OVERLAY_COLORS = {"Equivalent ellipses": "#00e5ff", "Oriented boxes": "#ffd54a", "Inscribed circles": "#ff4dff", "Geodesic paths": "#66ff66"}


# =============================================================================== helpers
def _as_labels(plane: np.ndarray, binary_to_components: bool = True) -> np.ndarray:
    """Interpret a plane (or volume) as an int32 label image (a 0/255 uint8 mask is labelled first)."""
    if binary_to_components and plane.dtype == np.uint8 and np.isin(np.unique(plane), (0, 255)).all():
        return skmeas.label(plane > 0, connectivity=plane.ndim).astype(np.int32)
    lab = np.rint(np.nan_to_num(plane)).astype(np.int32)
    lab[lab < 0] = 0
    return lab


def _planes(img: LumenImage, all_planes: bool) -> list[tuple[int, int, int]]:
    d = img.display
    return list(img.iter_planes()) if all_planes else [(d.t, d.z, d.c)]


def _position_cols(img: LumenImage, t: int, z: int, c: int, all_planes: bool) -> dict[str, int]:
    return {"Ch": c + 1, "Slice": z + 1, "Frame": t + 1} if all_planes else {}


def _depth(img: LumenImage) -> float:
    cal = img.calibration
    return float(cal.pixel_depth) if cal.pixel_depth and cal.pixel_depth > 0 else float(cal.pixel_width)


def _labels_image(labels: np.ndarray, name: str, like: LumenImage) -> LumenImage:
    """``label_image`` that stores more than 65535 labels as 32-bit instead of wrapping them."""
    n = int(labels.max()) if labels.size else 0
    if n < 65536:
        return label_image(labels, name, like=like)
    img = LumenImage(np.asarray(labels).astype(np.float32), name=name)
    img.calibration = type(like.calibration)(**like.calibration.to_dict())
    ch = img.display.channels[0]
    ch.lut = "Glasbey on dark"  # Fiji-exact Glasbey has a white index 0; the dark variant keeps the background black
    ch.min, ch.max = 0.0, float(n)
    return img


def _parse_label_list(text: str) -> list[int]:
    """'3, 5-7, 12' → [3, 5, 6, 7, 12]."""
    out: list[int] = []
    for tok in str(text or "").replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok[1:]:
            a, b = tok.split("-", 1)
            lo, hi = int(a), int(b)
            out.extend(range(min(lo, hi), max(lo, hi) + 1))
        else:
            out.append(int(tok))
    if not out:
        raise ValueError("Give one or more label values, e.g. '3, 5-7'")
    return out


def _picked_label(ctx) -> int:
    """The label value picked with the value-picker tool (the session's foreground value), > 0."""
    try:
        k = int(round(float(ctx.session.settings.get("foreground", 0))))
    except (TypeError, ValueError):
        k = 0
    if k <= 0:
        raise ValueError("Pick a label first: click it with the value picker (I) or enter its value")
    return k


def _overlay_item(kind: str, name: str, t: int, z: int, c: int, **shape: Any) -> dict[str, Any]:
    return {"id": roimod.new_id(), "name": name, "color": OVERLAY_COLORS.get(kind, "#00e5ff"), "stroke_width": 1.0,
            "position": {"t": int(t), "z": int(z), "c": int(c)}, "morpho": kind, **shape}


def _corner_hull(ys: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Convex hull (N, 2) of the pixel *corners* of a region, as (x, y) in pixel units.

    Every pixel is a unit square: the hull of the corners is the exact convex hull of the region
    (used for the convex area, the Feret diameters and the oriented box, as MorphoLibJ does).
    """
    from scipy.spatial import ConvexHull
    pts = np.stack([xs, ys], axis=1).astype(float)
    if len(pts) >= 3:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except Exception:
            pass
    corners = np.concatenate([pts + [dx, dy] for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)])
    corners = np.unique(corners, axis=0)
    hull = ConvexHull(corners)
    return corners[hull.vertices]


def _polygon_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _equivalent_ellipse(ys: np.ndarray, xs: np.ndarray, pw: float, ph: float) -> dict[str, float]:
    """Centre, radii and orientation of the ellipse with the same second-order moments as the region
    (MorphoLibJ: the moments get the 1/12 pixel term). Centre/radii calibrated; ``theta`` is the major-axis
    angle in image coordinates (radians, y down), ``orientation`` the ImageJ angle in degrees (y up, 0–180)."""
    cx, cy = float(xs.mean()), float(ys.mean())
    dx, dy = (xs - cx) * pw, (ys - cy) * ph
    ixx = float(np.mean(dx * dx)) + pw * pw / 12.0
    iyy = float(np.mean(dy * dy)) + ph * ph / 12.0
    ixy = float(np.mean(dx * dy))
    common = math.sqrt((ixx - iyy) ** 2 + 4.0 * ixy * ixy)
    r1 = math.sqrt(2.0) * math.sqrt(max(ixx + iyy + common, 0.0))
    r2 = math.sqrt(2.0) * math.sqrt(max(ixx + iyy - common, 0.0))
    theta = 0.5 * math.atan2(2.0 * ixy, ixx - iyy)
    return {"cx": (cx + 0.5) * pw, "cy": (cy + 0.5) * ph, "r1": r1, "r2": r2, "theta": theta,
            "orientation": (-math.degrees(theta)) % 180.0}


def _ellipse_polygon(cx: float, cy: float, r1: float, r2: float, theta: float, n: int = 72) -> list[list[float]]:
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ex, ey = r1 * np.cos(t), r2 * np.sin(t)
    c, s = math.cos(theta), math.sin(theta)
    return np.stack([cx + c * ex - s * ey, cy + s * ex + c * ey], axis=1).round(3).tolist()


def _oriented_box(hull_xy: np.ndarray, pw: float, ph: float) -> dict[str, Any]:
    """Minimum-area enclosing rectangle of the (pixel-corner) hull: calibrated centre, length, width, orientation
    and the corners in pixel coordinates (for the overlay)."""
    cal = hull_xy * [pw, ph]
    corners = min_area_rect(cal)
    e1, e2 = corners[1] - corners[0], corners[2] - corners[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    long_edge = e1 if l1 >= l2 else e2
    orient = math.degrees(math.atan2(-long_edge[1], long_edge[0])) % 180.0
    centre = corners.mean(axis=0)
    return {"cx": float(centre[0]), "cy": float(centre[1]), "length": max(l1, l2), "width": min(l1, l2), "orientation": orient,
            "corners_px": (corners / [pw, ph]).round(3).tolist()}


def _chamfer_graph(mask: np.ndarray) -> tuple[Any, np.ndarray]:
    """Sparse 8-connected graph over the pixels of ``mask`` with chamfer weights 3 (orthogonal) / 4 (diagonal).

    Returns (csr matrix, index map: -1 outside the mask, node index inside)."""
    idx = np.full(mask.shape, -1, np.int64)
    n = int(mask.sum())
    idx[mask] = np.arange(n)
    rows, cols, w = [], [], []
    for (dy, dx), weight in (((0, 1), 3.0), ((1, 0), 3.0), ((1, 1), 4.0), ((1, -1), 4.0)):
        a = idx[max(0, -dy):mask.shape[0] - max(0, dy), max(0, -dx):mask.shape[1] - max(0, dx)]
        b = idx[max(0, dy):mask.shape[0] - max(0, -dy), max(0, dx):mask.shape[1] - max(0, -dx)]
        m = (a >= 0) & (b >= 0)
        rows.append(a[m])
        cols.append(b[m])
        w.append(np.full(int(m.sum()), weight))
    if rows and sum(len(r) for r in rows):
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(w)
        g = coo_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n)).tocsr()
    else:
        g = coo_matrix((n, n)).tocsr()
    return g, idx


def _geodesic_diameter(mask: np.ndarray, start: tuple[int, int]) -> dict[str, Any]:
    """MorphoLibJ's geodesic diameter of one region (8-connected, chamfer 3-4 normalised by 3):
    propagate from ``start`` (the inscribed-circle centre) to its farthest pixel p1, then from p1 to its
    farthest pixel p2; the diameter is d(p1, p2) + √2 (pixel thickness). Returns the diameter in pixels,
    the extremities and the geodesic path (list of (y, x)) between them."""
    g, idx = _chamfer_graph(mask)
    n = g.shape[0]
    if n == 0:
        return {"diameter": 0.0, "p1": start, "p2": start, "path": [start]}
    nodes = np.argwhere(mask)  # node k ↔ nodes[k] (same order as idx assignment)
    s = int(idx[start])
    d0 = dijkstra(g, directed=False, indices=s)
    d0[~np.isfinite(d0)] = -1
    k1 = int(np.argmax(d0))
    d1, pred = dijkstra(g, directed=False, indices=k1, return_predecessors=True)
    d1[~np.isfinite(d1)] = -1
    k2 = int(np.argmax(d1))
    diam = float(d1[k2]) / 3.0 + math.sqrt(2.0)
    path = []
    k = k2
    while k >= 0 and k != k1 and len(path) <= n:
        path.append(tuple(int(v) for v in nodes[k]))
        k = int(pred[k])
    path.append(tuple(int(v) for v in nodes[k1]))
    return {"diameter": diam, "p1": tuple(int(v) for v in nodes[k1]), "p2": tuple(int(v) for v in nodes[k2]), "path": path[::-1]}


def _region_row(rp, pw: float, ph: float, wanted: set[str], crofton: int, conn: int, overlays: set[str],
                t: int, z: int, c: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measurements of one regionprops region plus its overlay items."""
    mask = rp.image
    y0, x0, y1, x1 = rp.bbox
    n = int(rp.area)
    row: dict[str, Any] = {}
    items: list[dict[str, Any]] = []
    area = n * pw * ph
    if "Area" in wanted:
        row["Area"] = area
    perim = None
    if "Perimeter" in wanted or "Circularity" in wanted:
        perim = float(skmeas.perimeter_crofton(mask, directions=crofton)) * math.sqrt(pw * ph)
    if "Perimeter" in wanted:
        row["Perimeter"] = perim
    if "Circularity" in wanted:
        row["Circularity"] = 4.0 * math.pi * area / (perim * perim) if perim and perim > 0 else float("nan")
    if "Euler number" in wanted:
        row["EulerNumber"] = int(skmeas.euler_number(mask, connectivity=1 if conn == 4 else 2))
    if "Bounding box" in wanted:
        row.update({"Box.X.Min": x0 * pw, "Box.X.Max": x1 * pw, "Box.Y.Min": y0 * ph, "Box.Y.Max": y1 * ph})
    ys_l, xs_l = np.nonzero(mask)
    ys, xs = ys_l + y0, xs_l + x0
    if "Centroid" in wanted:
        row["Centroid.X"] = (float(xs.mean()) + 0.5) * pw
        row["Centroid.Y"] = (float(ys.mean()) + 0.5) * ph
    need_ellipse = "Equivalent ellipse" in wanted or "Ellipse elongation" in wanted or "Equivalent ellipses" in overlays
    if need_ellipse:
        ell = _equivalent_ellipse(ys, xs, pw, ph)
        if "Equivalent ellipse" in wanted:
            row.update({"Ellipse.Center.X": ell["cx"], "Ellipse.Center.Y": ell["cy"], "Ellipse.Radius1": ell["r1"],
                        "Ellipse.Radius2": ell["r2"], "Ellipse.Orientation": ell["orientation"]})
        if "Ellipse elongation" in wanted:
            row["Ellipse.Elong"] = ell["r1"] / ell["r2"] if ell["r2"] > 0 else float("nan")
        if "Equivalent ellipses" in overlays:
            pts = _ellipse_polygon(ell["cx"] / pw, ell["cy"] / ph, ell["r1"] / pw, ell["r2"] / ph, ell["theta"])
            items.append(_overlay_item("Equivalent ellipses", f"Ellipse L{rp.label}", t, z, c, type="polygon", points=pts))
    need_hull = any(k in wanted for k in ("Convexity", "Max Feret diameter", "Oriented box", "Tortuosity")) or "Oriented boxes" in overlays
    feret = None
    if need_hull:
        hull = _corner_hull(ys, xs)
        if "Convexity" in wanted:
            row["Convexity"] = area / (_polygon_area(hull) * pw * ph)
        if "Max Feret diameter" in wanted or "Tortuosity" in wanted:
            feret, fang, _, _, _ = feret_diameters(hull * [pw, ph])
            if "Max Feret diameter" in wanted:
                row["MaxFeretDiam"] = feret
                row["MaxFeretAngle"] = fang
        if "Oriented box" in wanted or "Oriented boxes" in overlays:
            if len(hull) >= 3:
                box = _oriented_box(hull, pw, ph)
                if "Oriented box" in wanted:
                    row.update({"OrientedBox.Center.X": box["cx"], "OrientedBox.Center.Y": box["cy"], "OrientedBox.Length": box["length"],
                                "OrientedBox.Width": box["width"], "OrientedBox.Orientation": box["orientation"]})
                if "Oriented boxes" in overlays:
                    items.append(_overlay_item("Oriented boxes", f"Box L{rp.label}", t, z, c, type="polygon", points=box["corners_px"]))
            elif "Oriented box" in wanted:
                row.update({k: float("nan") for k in ("OrientedBox.Center.X", "OrientedBox.Center.Y", "OrientedBox.Length", "OrientedBox.Width", "OrientedBox.Orientation")})
    need_edt = any(k in wanted for k in ("Geodesic diameter", "Tortuosity", "Max inscribed circle", "Average thickness")) or \
        "Inscribed circles" in overlays or "Geodesic paths" in overlays
    if need_edt:
        edt = ndi.distance_transform_edt(np.pad(mask, 1), sampling=(ph, pw))[1:-1, 1:-1]
        iy, ix = np.unravel_index(int(np.argmax(edt)), edt.shape)
        radius = float(edt[iy, ix])
        if "Max inscribed circle" in wanted:
            row.update({"InscrCircle.Center.X": float(x0 + ix + 0.5) * pw, "InscrCircle.Center.Y": float(y0 + iy + 0.5) * ph, "InscrCircle.Radius": radius})
        if "Inscribed circles" in overlays:
            rx, ry = radius / pw, radius / ph
            items.append(_overlay_item("Inscribed circles", f"Circle L{rp.label}", t, z, c, type="oval",
                                       x=x0 + ix + 0.5 - rx, y=y0 + iy + 0.5 - ry, w=2 * rx, h=2 * ry))
        if "Average thickness" in wanted:
            row["AverageThickness"] = 2.0 * float(edt[mask].mean())
        if "Geodesic diameter" in wanted or "Tortuosity" in wanted or "Geodesic paths" in overlays:
            geo = _geodesic_diameter(mask, (int(iy), int(ix)))
            gd = geo["diameter"] * pw
            if "Geodesic diameter" in wanted:
                row["GeodesicDiameter"] = gd
            if "Tortuosity" in wanted:
                row["Tortuosity"] = gd / feret if feret else float("nan")
            if "Geodesic paths" in overlays and len(geo["path"]) >= 2:
                pts = [[float(x0 + px + 0.5), float(y0 + py + 0.5)] for py, px in geo["path"]]
                items.append(_overlay_item("Geodesic paths", f"Geodesic L{rp.label}", t, z, c, type="polyline", points=pts, width=1))
    return row, items


@op("morpho.analyze_regions", "Analyze Regions", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("features", "multichoice", REGION_FEATURES, choices=REGION_FEATURES, help="Measurements per label"),
            Param("crofton_directions", "choice", "4", choices=["4", "8"], label="Crofton perimeter directions"),
            Param("connectivity", "choice", "4", choices=["4", "8"], help="Euler number connectivity"),
            Param("overlays", "multichoice", [], choices=REGION_OVERLAYS, help="Shapes added to the image overlay (one item per label)"),
            Param("all_planes", "bool", False, label="Process whole stack", help="Measure the labels of every plane (3-D labels: one row per label and slice)")],
    description="MorphoLibJ's Analyze Regions: per-label area, Crofton perimeter, circularity, Euler number, bounding box, centroid, equivalent ellipse (centre, radii, orientation, elongation), convexity, max Feret diameter and angle, oriented (min-area) box, geodesic diameter (chamfer 3-4), tortuosity, largest inscribed circle and average thickness of a label image; optional ellipse / box / circle / geodesic-path overlays.",
    keywords=["morpholibj", "label", "region", "geodesic", "feret", "ellipse", "inscribed circle", "oriented box", "euler", "crofton", "thickness"])
def analyze_regions(ctx, features=None, crofton_directions="4", connectivity="4", overlays=None, all_planes=False):
    img = ctx.img
    cal = img.calibration
    pw, ph = float(cal.pixel_width), float(cal.pixel_height)
    wanted = set(features if features is not None else REGION_FEATURES)
    ovl = set(overlays or [])
    crofton = int(crofton_directions)
    conn = int(connectivity)
    rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    for (t, z, c) in _planes(img, all_planes):
        lab = _as_labels(img.data[t, z, c])
        pos = _position_cols(img, t, z, c, all_planes)
        for rp in skmeas.regionprops(lab):
            row, its = _region_row(rp, pw, ph, wanted, crofton, conn, ovl, t, z, c)
            rows.append({"Label": f"{img.name}:{rp.label}", **row, **pos})
            items.extend(its)
    unit = cal.unit if cal.scaled else "px"
    msg = f"{len(rows)} region{'s' if len(rows) != 1 else ''} measured (lengths in {unit})"
    res = OpResult(results=rows if rows else None, message=msg, log=msg)
    if ovl:
        keep = [o for o in img.overlay if o.get("morpho") not in ovl]
        res.overlay = keep + items
        msg += f"; {len(items)} overlay items"
        res.message = res.log = msg
    return res


# ======================================================================= intensity measurements
def _group_slices(lab: np.ndarray, vals: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """[(label, values of that label), …] for every label > 0 (one sort of the whole array)."""
    flat = lab.ravel()
    order = np.argsort(flat, kind="stable")
    sl = flat[order]
    sv = vals.ravel()[order]
    labels, starts = np.unique(sl, return_index=True)
    ends = np.append(starts[1:], sl.size)
    return [(int(k), sv[a:b]) for k, a, b in zip(labels.tolist(), starts.tolist(), ends.tolist()) if k > 0]


def _intensity_stats(v: np.ndarray, wanted: set[str], integer_values: bool) -> dict[str, float]:
    v = v.astype(np.float64)
    v = v[np.isfinite(v)]
    out: dict[str, float] = {}
    n = v.size
    if n == 0:
        return {k: float("nan") for k in wanted if k not in ("NumberOfVoxels", "Volume")}
    mean = float(v.mean())
    if "Mean" in wanted:
        out["Mean"] = mean
    if "StdDev" in wanted:
        out["StdDev"] = float(v.std(ddof=1)) if n > 1 else 0.0
    if "Max" in wanted:
        out["Max"] = float(v.max())
    if "Min" in wanted:
        out["Min"] = float(v.min())
    if "Median" in wanted:
        out["Median"] = float(np.median(v))
    if "Mode" in wanted:
        if integer_values and v.min() >= 0:
            out["Mode"] = float(np.argmax(np.bincount(v.astype(np.int64))))
        else:
            h, edges = np.histogram(v, bins=256)
            k = int(np.argmax(h))
            out["Mode"] = float((edges[k] + edges[k + 1]) / 2)
    if "Skewness" in wanted or "Kurtosis" in wanted:
        sd = float(v.std())
        zz = (v - mean) / sd if sd > 0 else None
        if "Skewness" in wanted:
            out["Skewness"] = float(np.mean(zz ** 3)) if zz is not None else 0.0
        if "Kurtosis" in wanted:
            out["Kurtosis"] = float(np.mean(zz ** 4) - 3) if zz is not None else 0.0
    return out


@op("morpho.intensity_measurements", "Intensity Measurements 2D/3D", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("intensity_image", "image", None, help="Image providing the gray values (same width and height as the label image; its current channel and frame are used)"),
            Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D, help="3D: the labels of the whole z-stack (number of voxels and volume use the voxel depth)"),
            Param("measurements", "multichoice", INTENSITY_MEASUREMENTS, choices=INTENSITY_MEASUREMENTS)],
    description="MorphoLibJ's Intensity Measurements: mean, SD, max, min, median, mode, skewness, kurtosis, number of voxels and volume of the gray values of an intensity image inside every label of the current label image (2-D plane or 3-D stack).",
    keywords=["morpholibj", "label", "intensity", "statistics", "mean", "volume", "3d"])
def intensity_measurements(ctx, intensity_image=None, mode="2D (current plane)", measurements=None):
    img = ctx.img
    d = img.display
    if not intensity_image:
        raise ValueError("Choose the intensity image (the current image provides the labels)")
    inten = ctx.session.get_image(intensity_image)
    if (inten.height, inten.width) != (img.height, img.width):
        raise ValueError("The intensity image must have the same width and height as the label image")
    wanted = set(measurements if measurements is not None else INTENSITY_MEASUREMENTS)
    cal = img.calibration
    pw, ph = float(cal.pixel_width), float(cal.pixel_height)
    three_d = str(mode).startswith("3D")
    ti, ci = min(d.t, inten.nt - 1), min(inten.display.c, inten.nc - 1)
    if three_d:
        if inten.nz != img.nz:
            raise ValueError("3D: the intensity image must have the same number of slices as the label image")
        lab = _as_labels(img.data[d.t, :, d.c])
        vals = inten.data[ti, :, ci]
        voxel = pw * ph * _depth(img)
    else:
        lab = _as_labels(img.data[d.t, d.z, d.c])
        vals = inten.data[ti, min(d.z, inten.nz - 1), ci]
        voxel = pw * ph
    integer_values = np.issubdtype(vals.dtype, np.integer)
    rows = []
    for k, v in _group_slices(lab, vals):
        row: dict[str, Any] = {"Label": f"{img.name}:{k}"}
        row.update(_intensity_stats(v, wanted, integer_values))
        if "NumberOfVoxels" in wanted:
            row["NumberOfVoxels"] = int(v.size)
        if "Volume" in wanted:
            row["Volume"] = float(v.size) * voxel
        rows.append(row)
    msg = f"{len(rows)} label{'s' if len(rows) != 1 else ''} measured on {inten.name}{' (3D)' if three_d else ''}"
    return OpResult(results=rows if rows else None, message=msg, log=msg)


# ================================================================================ label tools
def _largest_label(lab: np.ndarray) -> int:
    if lab.size == 0 or lab.max() <= 0:
        return 0
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return int(np.argmax(counts))


@op("morpho.keep_largest_label", "Keep Largest Label", CAT, kind="plane", supports_rgb=False, roi_aware=False,
    description="Keep only the label with the largest area (its value is kept; everything else becomes 0).", keywords=["morpholibj", "label", "largest", "biggest"])
def keep_largest_label(plane, ctx):
    lab = _as_labels(plane)
    k = _largest_label(lab)
    return back_to(np.where(lab == k, lab, 0) if k else np.zeros_like(lab), plane)


@op("morpho.remove_largest_label", "Remove Largest Label", CAT, kind="plane", supports_rgb=False, roi_aware=False,
    description="Set the label with the largest area to 0 (e.g. drop the background-like region).", keywords=["morpholibj", "label", "largest", "remove"])
def remove_largest_label(plane, ctx):
    lab = _as_labels(plane)
    k = _largest_label(lab)
    return back_to(np.where(lab == k, 0, lab), plane)


@op("morpho.replace_label", "Replace / Remove Labels", CAT, kind="plane", supports_rgb=False, roi_aware=False,
    params=[Param("labels", "string", "1", help="Label values to replace, e.g. '3, 5-7'; empty = the label picked with the value picker (I)"),
            Param("new_value", "int", 0, min=0, help="New value (0 removes the labels)")],
    description="Replace the given label values by another value (0 = remove them); an empty list means the label picked with the value picker.",
    keywords=["morpholibj", "label", "replace", "remove", "merge", "label edition"])
def replace_label(plane, ctx, labels="1", new_value=0):
    lab = _as_labels(plane)
    values = _parse_label_list(labels) if str(labels or "").strip() else [_picked_label(ctx)]
    sel = np.isin(lab, values)
    return back_to(np.where(sel, int(new_value), lab), plane)


@op("morpho.crop_label", "Crop Label", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("label", "int", 1, min=1, help="Label value to crop"), Param("border", "int", 0, min=0, max=1000, help="Extra pixels around the bounding box")],
    description="New binary image of one label cropped to its bounding box (every slice of the current channel and frame: a 3-D label gives a 3-D crop).",
    keywords=["morpholibj", "label", "crop", "extract"])
def crop_label(ctx, label=1, border=0):
    img = ctx.img
    d = img.display
    vol = _as_labels(img.data[d.t, :, d.c]) == int(label)
    if not vol.any():
        raise ValueError(f"Label {label} does not exist on the current channel / frame")
    zz, yy, xx = np.nonzero(vol)
    b = int(border)
    z0, z1 = int(zz.min()), int(zz.max()) + 1
    y0, y1 = max(int(yy.min()) - b, 0), min(int(yy.max()) + 1 + b, img.height)
    x0, x1 = max(int(xx.min()) - b, 0), min(int(xx.max()) + 1 + b, img.width)
    sub = vol[z0:z1, y0:y1, x0:x1]
    out = binary_image(sub[None, :, None], f"{img.name}-L{label}", like=img)
    msg = f"Label {label}: {int(sub.sum())} voxels, box {x1 - x0} × {y1 - y0}" + (f" × {z1 - z0}" if img.nz > 1 else "") + f" at ({x0}, {y0}" + (f", {z0})" if img.nz > 1 else ")")
    return OpResult(new_images=[out], message=msg, log=msg)


@op("morpho.labels_to_boundaries", "Label Boundaries", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("mode", "choice", "inner", choices=BOUNDARY_MODES, help="inner: label pixels next to another label or the background; outer: background pixels next to a label (both sides where two labels touch); thick: both"),
            Param("connectivity", "choice", "8", choices=["4", "8"]),
            Param("all_planes", "bool", False, label="Process whole stack")],
    description="Binary mask of the label boundaries (inner, outer or thick).", keywords=["morpholibj", "label", "boundary", "edges", "outline"])
def labels_to_boundaries(ctx, mode="inner", connectivity="8", all_planes=False):
    img = ctx.img
    planes = _planes(img, all_planes)
    out = np.zeros(img.shape if all_planes else (1, 1, 1, img.height, img.width), bool)
    conn = 2 if connectivity == "8" else 1
    n = 0
    for (t, z, c) in planes:
        lab = _as_labels(img.data[t, z, c])
        b = skseg.find_boundaries(lab, connectivity=conn, mode=mode)
        n += int(b.sum())
        out[(t, z, c) if all_planes else (0, 0, 0)] = b
    msg = f"{n} boundary pixels ({mode})"
    return OpResult(new_images=[binary_image(out, f"Boundaries of {img.name}", like=img)], message=msg, log=msg)


@op("morpho.dilate_labels", "Dilate Labels", CAT, kind="image", preview=False, supports_rgb=False, roi_aware=False, scope_param=False,
    params=[Param("radius", "float", 2.0, min=0.0, max=500.0, step=1.0, unit="px", help="Every label grows by this distance without overlapping its neighbours (nearest label wins)"),
            Param("mask_image", "image", None, help="Optional binary / label image: labels only grow where it is non-zero (constrained dilation)"),
            Param("all_planes", "bool", False, label="Process whole stack")],
    description="MorphoLibJ's Dilate Labels: grow every label by a distance without overlapping other labels, optionally constrained to a mask.",
    keywords=["morpholibj", "label", "dilate", "grow", "expand", "constrained"], undo_label="Dilate Labels")
def dilate_labels(ctx, radius=2.0, mask_image=None, all_planes=False):
    img = ctx.img
    mask_img = ctx.session.get_image(mask_image) if mask_image else None
    if mask_img is not None and (mask_img.height, mask_img.width) != (img.height, img.width):
        raise ValueError("The mask image must have the same width and height as the label image")
    out = img.data.copy()
    for (t, z, c) in _planes(img, all_planes):
        lab = _as_labels(img.data[t, z, c])
        grown = skseg.expand_labels(lab, distance=float(radius))
        if mask_img is not None:
            m = mask_img.data[min(t, mask_img.nt - 1), min(z, mask_img.nz - 1), min(mask_img.display.c, mask_img.nc - 1)] != 0
            grown = np.where(m | (lab > 0), grown, 0)
        out[t, z, c] = back_to(grown, img.data[t, z, c])
    return OpResult(data=out, message=f"Labels dilated by {radius:g} px" + (" inside the mask" if mask_img is not None else ""))


@op("morpho.label_morphology", "Label Morphological Filters", CAT, kind="plane", supports_rgb=False, roi_aware=False,
    params=[Param("operation", "choice", "Dilation", choices=LABEL_FILTERS),
            Param("radius", "int", 1, min=1, max=100, unit="px")],
    description="MorphoLibJ's Label Morphological Filters: dilation (labels grow into the background without overlapping), erosion (a pixel keeps its label only when its whole neighbourhood carries it), opening and closing of a label image.",
    keywords=["morpholibj", "label", "morphology", "dilation", "erosion", "opening", "closing"])
def label_morphology(plane, ctx, operation="Dilation", radius=1):
    lab = _as_labels(plane)
    fp = skm.disk(int(radius))

    def dil(a: np.ndarray) -> np.ndarray:
        return np.where(a > 0, a, skseg.expand_labels(a, distance=float(radius)))

    def ero(a: np.ndarray) -> np.ndarray:
        mn = ndi.grey_erosion(a, footprint=fp, mode="nearest")
        mx = ndi.grey_dilation(a, footprint=fp, mode="nearest")
        return np.where((mn == mx) & (a > 0), a, 0)

    if operation == "Dilation":
        out = dil(lab)
    elif operation == "Erosion":
        out = ero(lab)
    elif operation == "Opening":
        out = dil(ero(lab))
    else:
        out = ero(dil(lab))
    return back_to(out, plane)


def _adjacent_pairs(lab: np.ndarray, full: bool) -> tuple[np.ndarray, np.ndarray]:
    """Unique (label a < label b) pairs of touching labels and the number of adjacent pixel pairs for each.

    ``full``: 8- (2-D) / 26- (3-D) connectivity, else 4 / 6."""
    nd = lab.ndim
    import itertools
    offsets = [o for o in itertools.product((-1, 0, 1), repeat=nd) if any(o)]
    offsets = [o for o in offsets if o > tuple([0] * nd)]  # one of each ± pair
    if not full:
        offsets = [o for o in offsets if sum(abs(v) for v in o) == 1]
    codes = []
    for o in offsets:
        sa = tuple(slice(max(0, -v), lab.shape[i] - max(0, v)) for i, v in enumerate(o))
        sb = tuple(slice(max(0, v), lab.shape[i] - max(0, -v)) for i, v in enumerate(o))
        a, b = lab[sa], lab[sb]
        m = (a > 0) & (b > 0) & (a != b)
        if m.any():
            pa, pb = a[m].astype(np.int64), b[m].astype(np.int64)
            lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
            codes.append(lo * (int(lab.max()) + 1) + hi)
    if not codes:
        return np.zeros((0, 2), np.int64), np.zeros(0, np.int64)
    allc = np.concatenate(codes)
    u, cnt = np.unique(allc, return_counts=True)
    base = int(lab.max()) + 1
    return np.stack([u // base, u % base], axis=1), cnt


@op("morpho.region_adjacency_graph", "Region Adjacency Graph", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D),
            Param("connectivity", "choice", "4", choices=["4", "8"], help="2D: 4 / 8; 3D: 6 / 26 neighbours"),
            Param("show_overlay", "bool", True, help="Draw a line between the centroids of every touching pair on the overlay")],
    description="MorphoLibJ's Region Adjacency Graph: the pairs of labels that touch each other (table + Results rows with the number of adjacent pixel pairs) and optional lines between their centroids.",
    keywords=["morpholibj", "label", "adjacency", "graph", "neighbours", "touching", "rag"])
def region_adjacency_graph(ctx, mode="2D (current plane)", connectivity="4", show_overlay=True):
    img = ctx.img
    d = img.display
    three_d = str(mode).startswith("3D")
    lab = _as_labels(img.data[d.t, :, d.c] if three_d else img.data[d.t, d.z, d.c])
    pairs, counts = _adjacent_pairs(lab, full=(connectivity == "8"))
    rows = [{"Label": f"{img.name}:{int(a)}-{int(b)}", "Label 1": int(a), "Label 2": int(b), "Adjacent pairs": int(n)} for (a, b), n in zip(pairs.tolist(), counts.tolist())]
    table = {"name": "Region adjacency graph", "columns": ["Label 1", "Label 2", "Adjacent pairs"], "rows": [[r["Label 1"], r["Label 2"], r["Adjacent pairs"]] for r in rows]}
    msg = f"{len(rows)} adjacent label pair{'s' if len(rows) != 1 else ''} ({'3D' if three_d else '2D'}, {connectivity}-connectivity)"
    res = OpResult(results=rows if rows else None, table=table, message=msg, log=msg)
    if show_overlay:
        labels = np.unique(pairs) if len(pairs) else np.zeros(0, np.int64)
        cents = {int(k): c for k, c in zip(labels.tolist(), ndi.center_of_mass(lab > 0, lab, labels.tolist()))} if len(labels) else {}
        items = []
        for (a, b) in pairs.tolist():
            ca, cb = cents[int(a)], cents[int(b)]
            item = {"type": "line", "id": roimod.new_id(), "x1": float(ca[-1] + 0.5), "y1": float(ca[-2] + 0.5), "x2": float(cb[-1] + 0.5), "y2": float(cb[-2] + 0.5),
                    "width": 1, "color": "#ff9f43", "stroke_width": 1.0, "name": f"RAG {a}-{b}", "morpho": "rag"}
            if not three_d:
                item["position"] = {"t": d.t, "z": d.z, "c": d.c}
            items.append(item)
        res.overlay = [o for o in img.overlay if o.get("morpho") != "rag"] + items
    return res


def _overlap_rows(a: np.ndarray, b: np.ndarray, matching: str, name_a: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Per-label overlap measures between label images ``a`` (source) and ``b`` (target), plus the all-labels row."""
    base = int(max(a.max(), b.max())) + 1
    sizes_a = np.bincount(a.ravel(), minlength=base)
    sizes_b = np.bincount(b.ravel(), minlength=base)
    both = (a > 0) & (b > 0)
    codes = a[both].astype(np.int64) * base + b[both].astype(np.int64)
    u, cnt = np.unique(codes, return_counts=True)
    inter: dict[tuple[int, int], int] = {(int(k // base), int(k % base)): int(n) for k, n in zip(u.tolist(), cnt.tolist())}
    rows = []
    for la in np.unique(a[a > 0]).tolist():
        if matching == "Same label value":
            lb = la
            n = inter.get((la, lb), 0)
        else:
            cands = [(n, lb) for (xa, lb), n in inter.items() if xa == la]
            n, lb = max(cands) if cands else (0, 0)
        na, nb = int(sizes_a[la]), int(sizes_b[lb]) if lb < base else 0
        union = na + nb - n
        rows.append({"Label": f"{name_a}:{la}", "Source label": int(la), "Target label": int(lb), "Overlap": n,
                     "Jaccard": n / union if union else 0.0, "Dice": 2 * n / (na + nb) if (na + nb) else 0.0,
                     "Target overlap": n / nb if nb else 0.0, "Source overlap": n / na if na else 0.0,
                     "Volume similarity": 2 * (na - nb) / (na + nb) if (na + nb) else 0.0,
                     "False negative error": (nb - n) / nb if nb else 0.0, "False positive error": (na - n) / na if na else 0.0})
    fa, fb = a > 0, b > 0
    n = int((fa & fb).sum())
    na, nb = int(fa.sum()), int(fb.sum())
    union = na + nb - n
    total = {"Overlap": n, "Jaccard": n / union if union else 0.0, "Dice": 2 * n / (na + nb) if (na + nb) else 0.0,
             "Target overlap": n / nb if nb else 0.0, "Source overlap": n / na if na else 0.0,
             "Volume similarity": 2 * (na - nb) / (na + nb) if (na + nb) else 0.0,
             "False negative error": (nb - n) / nb if nb else 0.0, "False positive error": (na - n) / na if na else 0.0}
    return rows, total


@op("morpho.label_overlap_measures", "Label Overlap Measures", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("reference_image", "image", None, label="Target label image", help="Label image to compare the current (source) label image with; same width and height"),
            Param("matching", "choice", "Same label value", choices=OVERLAP_MATCHING, help="Compare each source label with the same label value of the target (MorphoLibJ) or with the target label it overlaps most"),
            Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D)],
    description="MorphoLibJ's Label Overlap Measures between two label images: overlap, Jaccard index, Dice coefficient, target / source overlap, volume similarity and false negative / positive errors per label pair plus an all-labels row (Results + table).",
    keywords=["morpholibj", "label", "overlap", "jaccard", "dice", "segmentation accuracy", "compare"])
def label_overlap_measures(ctx, reference_image=None, matching="Same label value", mode="2D (current plane)"):
    img = ctx.img
    d = img.display
    if not reference_image:
        raise ValueError("Choose the target label image")
    ref = ctx.session.get_image(reference_image)
    if (ref.height, ref.width) != (img.height, img.width):
        raise ValueError("The target label image must have the same width and height as the source")
    three_d = str(mode).startswith("3D")
    rt, rc = min(d.t, ref.nt - 1), min(ref.display.c, ref.nc - 1)
    if three_d:
        if ref.nz != img.nz:
            raise ValueError("3D: both label images must have the same number of slices")
        a, b = _as_labels(img.data[d.t, :, d.c]), _as_labels(ref.data[rt, :, rc])
    else:
        a, b = _as_labels(img.data[d.t, d.z, d.c]), _as_labels(ref.data[rt, min(d.z, ref.nz - 1), rc])
    rows, total = _overlap_rows(a, b, matching, img.name)
    rows.append({"Label": f"{img.name} vs {ref.name} (all labels)", "Source label": 0, "Target label": 0, **total})
    cols = ["Source label", "Target label", "Overlap", "Jaccard", "Dice", "Target overlap", "Source overlap", "Volume similarity", "False negative error", "False positive error"]
    table = {"name": "Label overlap", "columns": cols, "rows": [[r[c] for c in cols] for r in rows]}
    jac = [r["Jaccard"] for r in rows[:-1]]
    msg = (f"{len(rows) - 1} source label{'s' if len(rows) != 2 else ''} vs {ref.name}: mean Jaccard {np.mean(jac) if jac else 0:.3f}, "
           f"all labels Jaccard {total['Jaccard']:.3f} / Dice {total['Dice']:.3f}")
    return OpResult(results=rows, table=table, message=msg, log=msg)


# ================================================================== morphological segmentation
def _gradient(p: np.ndarray, kind: str, radius: int) -> np.ndarray:
    if kind == "Sobel":
        return skfilt.sobel(p).astype(np.float32)
    fp = skm.disk(max(int(radius), 1))
    return (ndi.grey_dilation(p, footprint=fp, mode="nearest") - ndi.grey_erosion(p, footprint=fp, mode="nearest")).astype(np.float32)


def _segment_plane(p: np.ndarray, input_kind: str, gradient: str, radius: int, smooth: float, tolerance: float, conn: int,
                   dams: bool) -> tuple[np.ndarray, np.ndarray]:
    """(basins label image, gradient) of one plane."""
    p = np.nan_to_num(p.astype(np.float32))
    if smooth > 0:
        p = ndi.gaussian_filter(p, smooth)
    grad = p if input_kind == "Gradient image" else _gradient(p, gradient, radius)
    if tolerance > 0:
        # extended minima of dynamic h = the regional minima of the h-minima transform (MorphoLibJ's
        # MinimaAndMaxima.extendedMinima): a basin whose bottom is a set of disconnected plateaus is one marker
        fp = np.ones((3, 3), np.uint8) if conn == 8 else np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)  # skimage's reconstruction sets arr.shape (numpy ≥ 2.5 deprecation)
            rec = skm.reconstruction(grad + np.float32(tolerance), grad, method="erosion", footprint=fp)
        minima = skm.local_minima(rec, connectivity=2 if conn == 8 else 1)
    else:
        minima = skm.local_minima(grad, connectivity=2 if conn == 8 else 1)
    markers = skmeas.label(minima > 0, connectivity=2 if conn == 8 else 1)
    basins = skseg.watershed(grad, markers, connectivity=2 if conn == 8 else 1, watershed_line=dams)
    return basins.astype(np.int32), grad


def _overlay_rgb(p: np.ndarray, lo: float, hi: float, basins: np.ndarray, show_basins: bool, dams: bool) -> np.ndarray:
    """Gray plane blended with Glasbey basin colours (and black / red dams)."""
    if hi <= lo:
        hi = lo + 1.0
    g = np.clip((np.nan_to_num(p.astype(np.float32)) - lo) / (hi - lo), 0, 1)
    rgb = np.repeat((g * 255).astype(np.float32)[..., None], 3, axis=2)
    if show_basins:
        lut = get_lut("Glasbey on dark").astype(np.float32)
        col = lut[np.where(basins > 0, (basins - 1) % 255 + 1, 0)]  # label k → LUT entry 1..255
        col[basins == 0] = rgb[basins == 0]
        rgb = 0.5 * rgb + 0.5 * col
    if dams:
        rgb[basins == 0] = [255, 0, 0] if not show_basins else [0, 0, 0]
    return np.clip(rgb, 0, 255).astype(np.uint8)


@op("morpho.morphological_segmentation", "Morphological Segmentation", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("input", "choice", "Object image", choices=SEGMENTATION_INPUTS, help="Object image: a gradient is computed first. Gradient image: the image already is a gradient / border image"),
            Param("gradient", "choice", "Morphological gradient", choices=GRADIENTS),
            Param("gradient_radius", "int", 1, min=1, max=20, help="Radius of the morphological gradient"),
            Param("tolerance", "float", 10.0, min=0.0, step=1.0, help="Dynamic of the extended minima used as markers (gray levels of the gradient); 0 = every regional minimum"),
            Param("connectivity", "choice", "4", choices=["4", "8"]),
            Param("calculate_dams", "bool", True, help="Separate the basins by one-pixel watershed lines"),
            Param("output", "choice", "Overlaid basins", choices=SEGMENTATION_OUTPUTS, help="Overlaid basins / dams: RGB image; Catchment basins: label image; Watershed lines: binary mask"),
            Param("smooth", "float", 0.0, min=0.0, max=20.0, step=0.5, advanced=True, help="Gaussian sigma applied before the gradient"),
            Param("all_planes", "bool", False, label="Process whole stack")],
    description="MorphoLibJ's Morphological Segmentation: gradient (morphological or Sobel) of the object image → extended minima of the given dynamic (tolerance) as markers → marker-controlled watershed; outputs the basins overlaid in colour, the dams, the catchment basins as a label image or the watershed lines as a mask.",
    keywords=["morpholibj", "watershed", "segmentation", "extended minima", "dynamic", "gradient", "basins", "dams"])
def morphological_segmentation(ctx, input="Object image", gradient="Morphological gradient", gradient_radius=1, tolerance=10.0, connectivity="4",
                               calculate_dams=True, output="Overlaid basins", smooth=0.0, all_planes=False):
    img = ctx.img
    conn = int(connectivity)
    planes = _planes(img, all_planes)
    dams = bool(calculate_dams) or output == "Watershed lines"
    shape = img.shape if all_planes else (1, 1, 1, img.height, img.width)
    rgb_out = output in ("Overlaid basins", "Overlaid dams")
    if rgb_out:
        out = np.zeros((shape[0], shape[1], 3, shape[3], shape[4]), np.uint8)
    elif output == "Watershed lines":
        out = np.zeros(shape, bool)
    else:
        out = np.zeros(shape, np.int32)
    n_basins = 0
    for (t, z, c) in planes:
        p = img.data[t, z, c]
        basins, _ = _segment_plane(p, input, gradient, gradient_radius, smooth, tolerance, conn, dams)
        n_basins += int(len(np.unique(basins)) - (1 if (basins == 0).any() else 0))
        tt, zz = (t, z) if all_planes else (0, 0)
        if rgb_out:
            ch = img.display.channels[min(c, len(img.display.channels) - 1)]
            out[tt, zz] = np.moveaxis(_overlay_rgb(p, float(ch.min), float(ch.max), basins, output == "Overlaid basins", dams), 2, 0)
        elif output == "Watershed lines":
            out[tt, zz, 0] = basins == 0
        else:
            out[tt, zz, 0] = basins
    msg = f"{n_basins} catchment basin{'s' if n_basins != 1 else ''} (tolerance {tolerance:g}, {conn}-connectivity)"
    if rgb_out:
        new = LumenImage(out, name=f"{img.name}-{'basins' if output == 'Overlaid basins' else 'dams'}", rgb=True)
        new.calibration = type(img.calibration)(**img.calibration.to_dict())
    elif output == "Watershed lines":
        new = binary_image(out, f"{img.name}-watershed-lines", like=img)
    else:
        new = _labels_image(out, f"{img.name}-basins", img)
    return OpResult(new_images=[new], message=msg, log=msg)
