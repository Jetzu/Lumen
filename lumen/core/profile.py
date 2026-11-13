"""Line / rectangle intensity profiles (Analyze ▸ Plot Profile)."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from . import roi as roimod
from .image import Calibration


def line_profile(plane: np.ndarray, roi: dict[str, Any], cal: Calibration, interpolate: bool = True) -> dict[str, Any]:
    pts = roimod.line_points(roi)
    width = int(roi.get("width", 1) or 1)
    xs_all: list[np.ndarray] = []
    ys_all: list[np.ndarray] = []
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        n = max(int(round(np.hypot(xb - xa, yb - ya))), 1)
        xs_all.append(np.linspace(xa, xb, n, endpoint=False))
        ys_all.append(np.linspace(ya, yb, n, endpoint=False))
    xs = np.concatenate(xs_all + [np.array([pts[-1][0]])])
    ys = np.concatenate(ys_all + [np.array([pts[-1][1]])])
    order = 1 if interpolate else 0
    if width <= 1:
        vals = ndi.map_coordinates(plane.astype(np.float32), [ys - 0.5, xs - 0.5], order=order, mode="nearest")
    else:
        # average across the line width using perpendicular offsets
        dx = np.gradient(xs); dy = np.gradient(ys)
        L = np.hypot(dx, dy); L[L == 0] = 1
        nx, ny = -dy / L, dx / L
        offs = np.linspace(-(width - 1) / 2, (width - 1) / 2, width)
        acc = np.zeros_like(xs, dtype=np.float64)
        for o in offs:
            acc += ndi.map_coordinates(plane.astype(np.float32), [ys + o * ny - 0.5, xs + o * nx - 0.5], order=order, mode="nearest")
        vals = acc / width
    dist = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xs) * cal.pixel_width, np.diff(ys) * cal.pixel_height))])
    return {"x": dist.tolist(), "values": vals.astype(float).tolist(), "x_label": f"Distance ({cal.unit})", "y_label": "Gray value"}


def rect_profile(plane: np.ndarray, roi: dict[str, Any], cal: Calibration, vertical: bool = False) -> dict[str, Any]:
    x0, y0 = int(round(roi["x"])), int(round(roi["y"]))
    x1, y1 = int(round(roi["x"] + roi["w"])), int(round(roi["y"] + roi["h"]))
    sub = plane[max(y0, 0):y1, max(x0, 0):x1].astype(np.float64)
    if sub.size == 0:
        return {"x": [], "values": [], "x_label": "", "y_label": ""}
    if vertical:
        vals = sub.mean(axis=1)
        x = np.arange(vals.size) * cal.pixel_height
    else:
        vals = sub.mean(axis=0)
        x = np.arange(vals.size) * cal.pixel_width
    return {"x": x.tolist(), "values": vals.tolist(), "x_label": f"Distance ({cal.unit})", "y_label": "Gray value"}
