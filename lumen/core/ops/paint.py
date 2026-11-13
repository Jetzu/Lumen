"""Painting tools (brush, bucket) and raw overlay-item management.

The brush and bucket tools of the viewport call these commands; every stroke / fill is a
normal command, so it is recorded by the macro recorder and creates one undo step.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
from skimage.draw import disk as sk_disk, line as sk_line, polygon as sk_polygon
from skimage.segmentation import flood

from .. import roi as roimod
from ..registry import Param, OpResult, cast_to, op
from .edit import _fill_value

CAT = "Edit"
OVL = "Image/Overlay"

OVERLAY_ITEM_TYPES = ["text", "arrow", "track", "calibration_bar", "time_stamp", "scalebar"]


# ------------------------------------------------------------------ helpers
def _parse_points(points: Any) -> list[tuple[float, float]]:
    """Accept a JSON string or a list of [x, y] pairs."""
    if isinstance(points, str):
        try:
            points = json.loads(points) if points.strip() else []
        except json.JSONDecodeError as e:
            raise ValueError(f"points: invalid JSON ({e})") from e
    out: list[tuple[float, float]] = []
    for p in points or []:
        if not isinstance(p, (list, tuple)) or len(p) < 2:
            raise ValueError("points must be a list of [x, y] pairs")
        x, y = float(p[0]), float(p[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        out.append((x, y))
    return out


def stroke_mask(points: list[tuple[float, float]], width: float, shape: tuple[int, int]) -> np.ndarray:
    """Rasterise a polyline with round caps and joins into a boolean (H, W) mask.

    Coordinates follow the ROI convention: integer values are pixel *edges*, so pixel
    (i, j) is centred at (j + 0.5, i + 0.5).
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)
    if not points:
        return mask
    # drop consecutive duplicates
    pts: list[tuple[float, float]] = [points[0]]
    for p in points[1:]:
        if abs(p[0] - pts[-1][0]) > 1e-9 or abs(p[1] - pts[-1][1]) > 1e-9:
            pts.append(p)
    width = max(float(width), 0.5)
    if width <= 1.5:
        # one-pixel pencil line
        if len(pts) == 1:
            xi, yi = int(np.floor(pts[0][0])), int(np.floor(pts[0][1]))
            if 0 <= yi < H and 0 <= xi < W:
                mask[yi, xi] = True
            return mask
        for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
            rr, cc = sk_line(int(np.floor(ya)), int(np.floor(xa)), int(np.floor(yb)), int(np.floor(xb)))
            ok = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            mask[rr[ok], cc[ok]] = True
        return mask
    r = width / 2.0
    # round joints / caps: one disc per vertex
    for x, y in pts:
        if x + r < 0 or y + r < 0 or x - r > W or y - r > H:
            continue
        rr, cc = sk_disk((y - 0.5, x - 0.5), r, shape=(H, W))
        mask[rr, cc] = True
    # thick body of every segment
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        poly = roimod.thick_segment_polygon(xa, ya, xb, yb, width)
        rr, cc = sk_polygon(poly[:, 1] - 0.5, poly[:, 0] - 0.5, shape=(H, W))
        mask[rr, cc] = True
    return mask


def _put(plane: np.ndarray, mask: np.ndarray, value: float) -> None:
    plane[mask] = cast_to(np.array([value], dtype=np.float64), plane.dtype)[0]


# ------------------------------------------------------------------ brush / bucket
@op("edit.paint_stroke", "Paint Stroke", CAT, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("points", "text", "[]", help="JSON list of [x, y] image coordinates along the stroke"),
            Param("width", "float", 8.0, min=0.5, max=500, step=0.5, unit="px"),
            Param("erase", "bool", False, help="Paint with the background value instead of the foreground"),
            Param("all_channels", "bool", False, help="Paint every channel of the current slice (RGB images are always painted on all three)")],
    description="Burn a brush stroke (polyline with round caps) into the current plane using the foreground value. Used by the brush tool.",
    keywords=["brush", "pencil", "draw", "paint"], undo_label="Paint stroke")
def paint_stroke(ctx, points="[]", width=8.0, erase=False, all_channels=False):
    img = ctx.img
    pts = _parse_points(points)
    if not pts:
        raise ValueError("A stroke needs at least one point")
    mask = stroke_mask(pts, width, (img.height, img.width))
    n = int(mask.sum())
    if n == 0:
        return OpResult(message="Stroke lies outside the image")
    value = _fill_value(ctx, "background" if erase else "foreground")
    d = img.display
    chans = range(img.nc) if (img.rgb or all_channels) else [d.c]
    data = img.data.copy()
    for c in chans:
        _put(data[d.t, d.z, c], mask, _fill_value(ctx, "background" if erase else "foreground", img.data[d.t, d.z, c]))  # per-channel colour on RGB
    return OpResult(data=data, rgb=img.rgb)


@op("edit.flood_fill", "Flood Fill", CAT, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("x", "int", 0, min=0), Param("y", "int", 0, min=0),
            Param("tolerance", "float", 0.0, min=0, step=1, help="Pixels within ± tolerance of the clicked value are filled"),
            Param("connectivity", "choice", "4", choices=["4", "8"]),
            Param("value", "float", -1, help="-1 = foreground colour setting")],
    description="Fill the connected region of similar pixels around a point with the foreground value (the bucket tool).",
    keywords=["bucket", "fill", "paint"], undo_label="Flood fill")
def flood_fill(ctx, x=0, y=0, tolerance=0.0, connectivity="4", value=-1.0):
    img = ctx.img
    if not (0 <= x < img.width and 0 <= y < img.height):
        raise ValueError("Point outside the image")
    v = float(value) if value >= 0 else _fill_value(ctx, "foreground")
    d = img.display
    conn = 2 if str(connectivity) == "8" else 1
    data = img.data.copy()
    if img.rgb:
        lum = data[d.t, d.z].astype(np.float32).mean(axis=0)
        m = flood(lum, (int(y), int(x)), tolerance=float(tolerance), connectivity=conn)
        for c in range(img.nc):
            _put(data[d.t, d.z, c], m, v if value >= 0 else _fill_value(ctx, "foreground", img.data[d.t, d.z, c]))
    else:
        plane = data[d.t, d.z, d.c]
        m = flood(plane.astype(np.float32), (int(y), int(x)), tolerance=float(tolerance), connectivity=conn)
        _put(plane, m, v)
    return OpResult(data=data, rgb=img.rgb, message=f"Filled {int(m.sum())} pixels")


# ------------------------------------------------------------------ overlay items
def _parse_item(item: Any) -> dict[str, Any]:
    if isinstance(item, str):
        try:
            item = json.loads(item) if item.strip() else {}
        except json.JSONDecodeError as e:
            raise ValueError(f"item: invalid JSON ({e})") from e
    if not isinstance(item, dict) or not item.get("type"):
        raise ValueError('item must be a JSON object with a "type" key')
    obj = dict(item)
    if obj["type"] in roimod.AREA_TYPES | roimod.LINE_TYPES | {"point"}:
        obj = roimod.normalize(obj)
    obj.setdefault("id", roimod.new_id())
    return obj


@op("overlay.add_item", "Add Overlay Item", OVL, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("item", "text", "{}", help='JSON item, e.g. {"type": "text", "x": 10, "y": 10, "text": "hello", "font_size": 14, "color": "#fff"}')],
    description="Append a raw overlay item (text, arrow, track, calibration bar, time stamp or ROI shape) to the image overlay.",
    keywords=["annotation", "overlay", "text", "arrow", "track"], undo_label="Add overlay item")
def overlay_add_item(ctx, item="{}"):
    img = ctx.img
    obj = _parse_item(item)
    return OpResult(overlay=[*img.overlay, obj], message=f"Added {obj['type']} to overlay")


@op("overlay.remove_item", "Remove Overlay Item", OVL, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("id", "string", "", help="Id of the overlay item to remove")],
    description="Remove one overlay item by id.", keywords=["annotation", "overlay"], undo_label="Remove overlay item")
def overlay_remove_item(ctx, id=""):
    img = ctx.img
    ids = {s.strip() for s in str(id).split(",") if s.strip()}
    if not ids:
        raise ValueError("Give the id of the item to remove")
    keep = [o for o in img.overlay if o.get("id") not in ids]
    removed = len(img.overlay) - len(keep)
    if removed == 0:
        raise ValueError(f"No overlay item with id {id!r}")
    return OpResult(overlay=keep, message=f"Removed {removed} overlay item{'s' if removed != 1 else ''}")


@op("overlay.clear_type", "Clear Overlay Items of Type", OVL, kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("type", "string", "text", help="text, arrow, track, calibration_bar, time_stamp, scalebar, or a ROI type such as rect")],
    description="Remove every overlay item of the given type (e.g. all tracks).", keywords=["annotation", "overlay", "tracks"], undo_label="Clear overlay items")
def overlay_clear_type(ctx, type="text"):
    img = ctx.img
    kinds = {s.strip() for s in str(type).split(",") if s.strip()}
    keep = [o for o in img.overlay if o.get("type") not in kinds]
    removed = len(img.overlay) - len(keep)
    return OpResult(overlay=keep, message=f"Removed {removed} {'/'.join(sorted(kinds))} item{'s' if removed != 1 else ''}")
