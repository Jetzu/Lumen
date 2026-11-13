"""Edit menu: fill/clear/draw and the Selection submenu."""
from __future__ import annotations

import math
import re

import numpy as np
from scipy import ndimage as ndi

from .. import roi as roimod
from ..registry import Param, OpResult, cast_to, op
from ._util import binary_image

CAT = "Edit"
SEL = "Edit/Selection"


def _fill_value(ctx, which: str, plane: np.ndarray | None = None) -> float:
    """The foreground / background value for the plane being processed.

    RGB images use the colour picked in the Color Picker (``settings["foreground_color"]`` /
    ``["background_color"]``, "#rrggbb") per channel when the plane's channel can be recovered
    (``plane`` is the view ``img.data[t, z, c]`` that ``apply_plane_op`` passes); otherwise the
    grey value setting applies to every channel."""
    img = ctx.img
    settings = ctx.session.settings
    if img.rgb and plane is not None:
        hex_col = str(settings.get(f"{which}_color") or "").strip()
        if re.fullmatch(r"#[0-9a-fA-F]{6}", hex_col):
            from .process_extra import _plane_index
            idx = _plane_index(plane, img)
            if idx is not None:
                rgb = [int(hex_col[i:i + 2], 16) for i in (1, 3, 5)]
                return float(rgb[min(idx[2], 2)])
    v = settings.get(which, 255 if which == "foreground" else 0)
    if img.data.dtype == np.uint16 and v == 255:
        return 65535.0
    return float(v)


def _pixel(v: float, dtype) -> np.generic | float:
    """``v`` as a pixel of ``dtype``: like ImageJ's ``ImageProcessor.setValue()`` an out-of-range value is clamped to
    the type's range (0…255 / 0…65535) instead of raising numpy 2's OverflowError (foreground 300 or background −5
    picked for another image depth); float32 keeps the value as it is.  Same conversion as paint.py's ``_put``."""
    if dtype == np.float32:
        return v
    return cast_to(np.array([v], np.float64), dtype)[0]


@op("edit.fill", "Fill", CAT, kind="plane", requires_roi="area", preview=False, shortcut="F", description="Fill the selection with the foreground value.",
    params=[Param("value", "float", -1, help="-1 = foreground colour setting")])
def fill(plane, ctx, value=-1):
    v = value if value >= 0 else _fill_value(ctx, "foreground", plane)
    out = plane.copy()
    out[...] = _pixel(v, plane.dtype)
    return out


@op("edit.clear", "Clear", CAT, kind="plane", requires_roi="area", preview=False, description="Set the selection to the background value.")
def clear(plane, ctx):
    out = plane.copy()
    out[...] = _pixel(_fill_value(ctx, "background", plane), plane.dtype)
    return out


@op("edit.clear_outside", "Clear Outside", CAT, kind="plane", requires_roi="area", preview=False, roi_aware=False, description="Set everything outside the selection to the background value.")
def clear_outside(plane, ctx):
    m = ctx.mask
    out = plane.copy()
    out[~m] = _pixel(_fill_value(ctx, "background", plane), plane.dtype)
    return out


@op("edit.draw", "Draw", CAT, kind="plane", requires_roi="any", preview=False, roi_aware=False,
    description="Burn the selection outline into the pixels (ImageJ's Roi.drawPixels: rectangles, ovals, polygons and lines are drawn vertex by vertex with ImageProcessor.lineTo).",
    params=[Param("line_width", "int", 1, min=1, max=50, help="Edit ▸ Options ▸ Line Width: dots of this diameter along the outline")])
def draw(plane, ctx, line_width=1):
    from ._imagej_roi import draw_roi
    roi = ctx.roi
    H, W = plane.shape
    out = plane.copy()
    value = _pixel(_fill_value(ctx, "foreground", plane), plane.dtype)
    if draw_roi(out, roi, value, int(line_width)):
        return out
    # point sets and composite masks: the outline pixels of the mask (ImageJ draws the flattened shape path)
    if roimod.is_area(roi):
        m = roimod.to_mask(roi, (H, W))
        edge = m & ~ndi.binary_erosion(m, border_value=0)
        if line_width > 1:
            edge = ndi.binary_dilation(edge, iterations=line_width // 2)
    else:
        edge = roimod.to_mask({**roi, "width": line_width}, (H, W))
    out[edge] = value
    return out


@op("edit.set_colors", "Colour Settings", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("foreground", "float", 255), Param("background", "float", 0)], description="Values used by Fill, Clear and Draw.")
def set_colors(ctx, foreground=255.0, background=0.0):
    ctx.session.settings.update(foreground=foreground, background=background)
    return OpResult(message=f"Foreground {foreground:g}, background {background:g}")


# ------------------------------------------------------------------ selection
@op("selection.all", "Select All", SEL, kind="analysis", preview=False, shortcut="A", roi_aware=False, description="Select the whole image.")
def select_all(ctx):
    img = ctx.img
    return OpResult(roi={"type": "rect", "x": 0, "y": 0, "w": img.width, "h": img.height})


@op("selection.none", "Select None", SEL, kind="analysis", preview=False, shortcut="Shift+A", roi_aware=False, description="Remove the selection.")
def select_none(ctx):
    return OpResult(clear_roi=True)


@op("selection.enlarge", "Enlarge", SEL, kind="analysis", preview=False, requires_roi="area", roi_aware=False,
    params=[Param("pixels", "float", 5, step=1, help="Negative values shrink")],
    description="Grow or shrink the selection (ImageJ's RoiEnlarger: rectangles and ovals grow by round(n) on every side; other selections take the pixels within n of the selection, or – shrinking – the pixels more than n inside it, from the Euclidean distance map).")
def enlarge(ctx, pixels=5.0):
    from ._imagej_roi import enlarge_roi
    img = ctx.img
    return OpResult(roi=enlarge_roi(ctx.roi, float(pixels), (img.height, img.width)))


@op("selection.band", "Make Band", SEL, kind="analysis", preview=False, requires_roi="area", roi_aware=False,
    params=[Param("width", "float", 15, min=1, max=255, step=1, help="Rounded to whole pixels (at most 255)")],
    description="Band of the given width around the selection (ImageJ's Make Band: the pixels whose rounded distance to the selection is ≤ width, minus the selection).")
def make_band(ctx, width=15.0):
    from ._imagej_roi import make_band as ij_make_band
    img = ctx.img
    return OpResult(roi=ij_make_band(ctx.roi, float(width), (img.height, img.width)))


@op("selection.convex_hull", "Convex Hull", SEL, kind="analysis", preview=False, requires_roi="area", roi_aware=False, description="Replace the selection with its convex hull polygon.")
def convex_hull(ctx):
    parts = roimod.outlines(ctx.roi)          # every part of a composite mask, not only the largest blob
    pts = np.vstack(parts) if parts else np.zeros((0, 2))
    from scipy.spatial import ConvexHull
    if len(pts) < 3:
        raise ValueError("Need at least three points")
    hull = ConvexHull(pts)
    return OpResult(roi={"type": "polygon", "points": pts[hull.vertices].tolist(), "name": "Hull"})


@op("selection.fit_ellipse", "Fit Ellipse", SEL, kind="analysis", preview=False, requires_roi="area", roi_aware=False,
    description="Replace the selection with the best-fit ellipse (ImageJ's EllipseFitter – same area and second moments – as a 72-vertex EllipseRoi polygon).")
def fit_ellipse(ctx):
    from ._imagej_measure import clip_rect_mask, fit_ellipse_polygon
    img = ctx.img
    rect, m = roimod.roi_rect_mask(ctx.roi, (img.height, img.width))
    (x, y, w, h), m = clip_rect_mask(rect, m, img.width, img.height)
    if w <= 0 or h <= 0:
        raise ValueError("The selection lies outside the image")
    xs, ys = fit_ellipse_polygon(m, x, y, w, h)
    # 'ellipse': an ImageJ EllipseRoi – its integer bounds follow makePolygonRelative (FloatPolygon.getBounds)
    return OpResult(roi={"type": "polygon", "subtype": "ellipse", "points": np.stack([xs.astype(float), ys.astype(float)], 1).tolist(), "name": "Ellipse"})


@op("selection.bounding_box", "To Bounding Box", SEL, kind="analysis", preview=False, requires_roi="any", roi_aware=False, description="Replace the selection with its bounding rectangle.")
def to_bbox(ctx):
    x, y, w, h = roimod.bounding_box(ctx.roi)
    return OpResult(roi={"type": "rect", "x": x, "y": y, "w": w, "h": h})


@op("selection.rotate", "Rotate Selection", SEL, kind="analysis", preview=False, requires_roi="any", roi_aware=False,
    params=[Param("angle", "float", 15, step=1)], description="Rotate the selection about the centre of its bounds (ImageJ's RoiRotator).")
def rotate_sel(ctx, angle=15.0):
    from ._imagej_roi import rotate_roi_ij
    roi = ctx.roi
    r = rotate_roi_ij(roi, float(angle))
    if r is not None:
        return OpResult(roi=r)
    if roi["type"] == "point":
        parts = [np.asarray(roi["points"], float).reshape(-1, 2)]
    elif roimod.is_area(roi):
        parts = roimod.outlines(roi)          # every part and hole of a composite mask, not only the largest blob
    else:
        parts = [np.asarray(roimod.line_points(roi), float).reshape(-1, 2)]
    if not parts or not len(parts[0]):
        raise ValueError("Empty selection")
    cx, cy = np.vstack(parts).mean(axis=0)
    a = math.radians(angle)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    rotated = [(p - [cx, cy]) @ R.T + [cx, cy] for p in parts]
    new = rotated[0]
    if roi["type"] in ("line", "polyline", "point"):
        if roi["type"] == "line":
            return OpResult(roi={**roi, "x1": new[0, 0], "y1": new[0, 1], "x2": new[1, 0], "y2": new[1, 1]})
        return OpResult(roi={**roi, "points": new.tolist()})
    if len(rotated) > 1:   # composite mask: rasterise every rotated outline with even-odd parity (holes stay holes)
        r = roimod.polygons_to_mask_roi(rotated, roi.get("name"))
        if r is None:
            raise ValueError("Empty selection")
        return OpResult(roi=r)
    return OpResult(roi={"type": "polygon", "points": new.tolist(), "name": roi.get("name")})


@op("selection.scale", "Scale Selection", SEL, kind="analysis", preview=False, requires_roi="any", roi_aware=False,
    params=[Param("x_scale", "float", 1.5, min=0.01, step=0.1), Param("y_scale", "float", 1.5, min=0.01, step=0.1), Param("centered", "bool", True)],
    description="Scale the selection (ImageJ's RoiScaler: about the origin, or 'centered' about the selection's centroid with the vertices rounded to pixels; rectangles and ovals become polygons).")
def scale_sel(ctx, x_scale=1.5, y_scale=1.5, centered=True):
    from ._imagej_roi import scale_roi_ij
    roi = ctx.roi
    img = ctx.img
    r = scale_roi_ij(roi, float(x_scale), float(y_scale), bool(centered), (img.height, img.width))
    if r is not None:
        return OpResult(roi=r)
    # composite (mask) selections: ImageJ scales the shape; Lumen resamples the mask
    bx, by, bw, bh = roimod.bounding_box(roi)
    cx, cy = bx + bw / 2, by + bh / 2
    r = roimod.scale_roi(roi, x_scale, y_scale)
    if centered:
        nbx, nby, nbw, nbh = roimod.bounding_box(r)
        r = roimod.translate(r, cx - (nbx + nbw / 2), cy - (nby + nbh / 2))
    return OpResult(roi=r)


@op("selection.interpolate", "Interpolate / Smooth", SEL, kind="analysis", preview=False, requires_roi="any", roi_aware=False,
    params=[Param("interval", "float", 5, min=0.01, step=0.5, help="Point spacing in pixels"),
            Param("smooth", "bool", True, help="3-point running average of freehand / traced outlines first (ImageJ: polygons are not smoothed)"),
            Param("adjust", "bool", False, label="Adjust interval to match", help="Let the spacing shrink (≤ 20 %) so that the last segment matches too")],
    description="Resample the selection outline at a fixed point spacing (ImageJ's Interpolate: Roi.getInterpolatedPolygon).")
def interpolate(ctx, interval=5.0, smooth=True, adjust=False):
    from ._imagej_roi import interpolated_polygon, smooth_float_polygon
    roi = ctx.roi
    t = roi["type"]
    if t == "point":
        return OpResult(roi=roi)
    signed = -abs(float(interval)) if adjust else abs(float(interval))
    is_line = roimod.is_line(roi)
    if t == "mask":
        parts = [p for p in roimod.outlines(roi) if len(p) >= 3]   # every part and hole of a composite mask (ImageJ: ShapeRoi.getRois, POLYGON type: no smoothing)
        if not parts:
            raise ValueError("Need an outline")
        resampled = []
        for p in parts:
            xs, ys = interpolated_polygon(p[:, 0], p[:, 1], signed, False)
            resampled.append(np.stack([xs.astype(np.float64), ys.astype(np.float64)], 1))
        if len(resampled) > 1:   # rasterised back into a mask ROI
            r = roimod.polygons_to_mask_roi(resampled, roi.get("name"))
            if r is None:
                raise ValueError("Selection is empty after interpolation")
            return OpResult(roi=r)
        return OpResult(roi={"type": "polygon", "points": resampled[0].tolist(), "name": roi.get("name")})
    if t == "line":
        xs, ys = np.asarray([roi["x1"], roi["x2"]], np.float32), np.asarray([roi["y1"], roi["y2"]], np.float32)
    elif t == "polyline":
        pts = np.asarray(roi["points"], np.float64).reshape(-1, 2)
        xs, ys = pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32)
    else:
        pts = roimod.polygon_points(roi)
        xs, ys = pts[:, 0].astype(np.float32), pts[:, 1].astype(np.float32)
    if smooth and t == "freehand":
        xs, ys = smooth_float_polygon(xs, ys, closed=True)
    xs, ys = interpolated_polygon(xs, ys, signed, is_line)
    pts_out = [[float(x), float(y)] for x, y in zip(xs, ys)]
    keep = {k: v for k, v in roi.items() if k in ("name", "color", "stroke_width", "position")}
    if is_line:
        return OpResult(roi={**keep, "type": "polyline", "points": pts_out, "width": roi.get("width", 1)})
    # ImageJ: a polygon interpolated at > 1 px (or any selection at ≥ 8 px) stays a POLYGON, otherwise FREEROI
    new_type = "polygon" if ((t == "polygon" and abs(interval) > 1.0) or abs(interval) >= 8.0) else "freehand"
    return OpResult(roi={**keep, "type": new_type, "points": pts_out})


@op("selection.create_mask", "Create Mask", SEL, kind="analysis", preview=False, requires_roi="area", roi_aware=False, description="New binary image from the selection.")
def create_mask(ctx):
    img = ctx.img
    m = roimod.to_mask(ctx.roi, (img.height, img.width))
    return OpResult(new_images=[binary_image(m, "Mask", like=img)])


@op("selection.create", "Create Selection", SEL, kind="analysis", preview=False, roi_aware=False, supports_rgb=False,
    description="Selection from the thresholded / binary pixels of the current plane.")
def create_selection(ctx):
    from ._util import current_threshold
    img = ctx.img
    p = img.current_plane()
    thr = current_threshold(ctx)
    if thr is not None:
        m = (p >= thr[0]) & (p <= thr[1])
    else:
        m = p > 0
    r = roimod.from_mask(m, "Selection")
    if r is None:
        raise ValueError("No foreground pixels")
    return OpResult(roi=r)


@op("selection.add_to_manager", "Add to ROI Manager", SEL, kind="analysis", preview=False, requires_roi="any", roi_aware=False, shortcut="T", description="Save the selection in the ROI manager.")
def add_to_manager(ctx):
    return OpResult(rois_add=[ctx.roi], message="Added to ROI manager")


@op("selection.specify", "Specify", SEL, kind="analysis", preview=False, roi_aware=False,
    params=[Param("width", "float", 100, min=1), Param("height", "float", 100, min=1), Param("x", "float", 0), Param("y", "float", 0), Param("oval", "bool", False), Param("centered", "bool", False)],
    description="Create a rectangle or oval with exact coordinates.")
def specify(ctx, width=100.0, height=100.0, x=0.0, y=0.0, oval=False, centered=False):
    if centered:
        x, y = x - width / 2, y - height / 2
    return OpResult(roi={"type": "oval" if oval else "rect", "x": x, "y": y, "w": width, "h": height})


@op("selection.line_width", "Line Width", SEL, kind="analysis", preview=False, requires_roi="line", roi_aware=False,
    params=[Param("width", "int", 1, min=1, max=200)], description="Set the width of the line selection (for profiles and measurements).")
def line_width(ctx, width=1):
    return OpResult(roi={**ctx.roi, "width": width})


# ------------------------------------------------------------------ overlay
@op("overlay.add", "Add Selection to Overlay", "Image/Overlay", kind="image", preview=False, requires_roi="any", roi_aware=False, shortcut="B", scope_param=False,
    description="Draw the selection as a non-destructive overlay.", params=[Param("color", "string", "#ffd700"), Param("stroke_width", "float", 1.5, min=0.5, max=20)])
def overlay_add(ctx, color="#ffd700", stroke_width=1.5):
    img = ctx.img
    roi = roimod.normalize(ctx.roi)
    roi["color"] = color
    roi["stroke_width"] = stroke_width
    return OpResult(overlay=[*img.overlay, roi], clear_roi=True)


@op("overlay.from_manager", "Overlay from ROI Manager", "Image/Overlay", kind="image", preview=False, roi_aware=False, scope_param=False, description="Show every ROI of the manager as an overlay.")
def overlay_from_manager(ctx):
    img = ctx.img
    rois = [dict(r, color=r.get("color", "#ffd700")) for r in ctx.session.roi_manager.to_list()]
    return OpResult(overlay=rois)


@op("overlay.remove", "Remove Overlay", "Image/Overlay", kind="image", preview=False, roi_aware=False, scope_param=False, description="Clear the overlay.")
def overlay_remove(ctx):
    return OpResult(overlay=[])


@op("overlay.flatten", "Flatten", "Image/Overlay", kind="image", preview=False, roi_aware=False, scope_param=False, description="Burn the overlay into a new RGB image.")
def flatten(ctx):
    from ..render import render_rgb
    try:
        from ..overlay_render import burn_overlay
    except Exception:  # pragma: no cover
        burn_overlay = None
    img = ctx.img
    d = img.display
    rgb = render_rgb(img).copy()
    if burn_overlay is not None:
        rgb = burn_overlay(rgb, img.overlay, img, t=d.t, z=d.z, settings=ctx.session.settings)
    else:
        H, W = rgb.shape[:2]
        for o in img.overlay:
            if o.get("type") == "scalebar":
                _burn_scalebar(rgb, o)
                continue
            if o.get("type") not in roimod.AREA_TYPES | roimod.LINE_TYPES | {"point"}:
                continue
            col = o.get("color", "#ffd700").lstrip("#")
            rgbcol = tuple(int(col[i:i + 2], 16) for i in (0, 2, 4)) if len(col) == 6 else (255, 215, 0)
            if roimod.is_area(o):
                m = roimod.to_mask(o, (H, W))
                edge = m & ~ndi.binary_erosion(m, border_value=0)
            else:
                edge = roimod.to_mask(o, (H, W))
            rgb[edge] = rgbcol
    new = img.copy(name=f"{img.name} (flat)", data=np.ascontiguousarray(rgb).transpose(2, 0, 1)[None, None])
    new.rgb = True
    new.reset_display()
    return OpResult(new_images=[new])


@op("selection.wand", "Wand (trace region)", SEL, kind="analysis", preview=False, roi_aware=False,
    params=[Param("x", "int", 0, min=0), Param("y", "int", 0, min=0), Param("tolerance", "float", 0.0, min=0, step=1), Param("connectivity", "choice", "8", choices=["4", "8"]),
            Param("add", "bool", False, help="Combine with the current selection")],
    description="Select the connected region of similar pixels around a point (the wand tool).")
def wand(ctx, x=0, y=0, tolerance=0.0, connectivity="8", add=False):
    from skimage.segmentation import flood
    img = ctx.img
    p = img.current_plane()
    if not (0 <= x < img.width and 0 <= y < img.height):
        raise ValueError("Point outside the image")
    from ._util import current_threshold
    thr = current_threshold(ctx)
    if thr is not None and thr[0] <= float(p[y, x]) <= thr[1]:
        src = ((p >= thr[0]) & (p <= thr[1])).astype(np.uint8)
        m = flood(src, (y, x), connectivity=2 if connectivity == "8" else 1)
    else:
        m = flood(p.astype(np.float32), (y, x), tolerance=float(tolerance), connectivity=2 if connectivity == "8" else 1)
    if add and ctx.roi is not None and roimod.is_area(ctx.roi):
        m = m | roimod.to_mask(ctx.roi, m.shape)
    r = roimod.from_mask(m, "Wand")
    return OpResult(roi=r)


@op("selection.inverse", "Make Inverse", SEL, kind="analysis", preview=False, requires_roi="area", roi_aware=False, description="Select everything outside the current selection.")
def make_inverse(ctx):
    img = ctx.img
    m = roimod.to_mask(ctx.roi, (img.height, img.width))
    r = roimod.from_mask(~m, "Inverse")
    if r is None:
        return OpResult(clear_roi=True)
    return OpResult(roi=r)


@op("selection.line_to_area", "Line to Area", SEL, kind="analysis", preview=False, requires_roi="line", roi_aware=False,
    description="Convert a (thick) line selection into an area selection (ImageJ's Roi.convertLineToArea: a straight line becomes its rotated rectangle polygon, a segmented line the mask of its stroke).")
def line_to_area(ctx):
    from ._imagej_roi import line_to_area_roi
    img = ctx.img
    r = line_to_area_roi(ctx.roi)
    if r is not None:
        r.setdefault("name", "Line area")
        return OpResult(roi=r)
    m = roimod.to_mask(ctx.roi, (img.height, img.width))
    return OpResult(roi=roimod.from_mask(m, "Line area"))


@op("selection.area_to_line", "Area to Line", SEL, kind="analysis", preview=False, requires_roi="area", roi_aware=False, description="Convert the outline of an area selection into a segmented line.")
def area_to_line(ctx):
    parts = roimod.outlines(ctx.roi)
    if len(parts) > 1:   # ImageJ: "Composite selections cannot be converted to lines" (one polyline cannot hold several parts / holes)
        raise ValueError("Composite selections (several parts or holes) cannot be converted to lines")
    pts = parts[0] if parts else np.zeros((0, 2))
    if len(pts) < 2:
        raise ValueError("Empty outline")
    closed = np.vstack([pts, pts[:1]])
    return OpResult(roi={"type": "polyline", "points": closed.round(2).tolist(), "width": 1})


@op("edit.copy", "Copy", CAT, kind="analysis", preview=False, roi_aware=False, shortcut="Cmd+C", description="Copy the selection's pixels (all channels of the current plane) to the internal clipboard.")
def copy(ctx):
    img = ctx.img
    d = img.display
    if ctx.roi is not None and roimod.is_area(ctx.roi):
        x, y, w, h = roimod.bounding_box(ctx.roi)
        x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
        x1, y1 = min(int(round(x + w)), img.width), min(int(round(y + h)), img.height)
        mask = roimod.to_mask(ctx.roi, (img.height, img.width))[y0:y1, x0:x1]
    else:
        x0, y0, x1, y1 = 0, 0, img.width, img.height
        mask = None
    ctx.session.settings["clipboard"] = {"data": img.data[d.t, d.z, :, y0:y1, x0:x1].copy(), "mask": mask, "rgb": img.rgb}
    return OpResult(message=f"Copied {x1 - x0}×{y1 - y0} px")


@op("edit.paste", "Paste", CAT, kind="image", preview=False, roi_aware=False, scope_param=False, shortcut="Cmd+V", description="Paste the clipboard into the current plane, at the selection (or centred).")
def paste(ctx):
    # Edit ▸ Paste honours the Paste Control / Preferences transfer mode (Copy, Blend, Transparent-white, Add …)
    # exactly like ImageJ; "Copy" is pixel- and ROI-identical to a plain paste.
    from .edit_extra import current_paste_mode, paste_with_mode  # edit.py is imported before edit_extra.py
    return paste_with_mode(ctx, current_paste_mode(ctx.session))


@op("overlay.scale_bar", "Scale Bar", "Image/Overlay", kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("length", "float", 0, min=0, help="In calibrated units; 0 = automatic"), Param("thickness", "int", 6, min=1, max=100),
            Param("position", "choice", "Lower Right", choices=["Lower Right", "Lower Left", "Upper Right", "Upper Left"]), Param("color", "string", "#ffffff"),
            Param("show_label", "bool", True), Param("font_size", "int", 14, min=6, max=120)],
    description="Add a calibrated scale bar to the overlay (use Flatten to burn it in).", keywords=["calibration bar", "micron bar"])
def scale_bar(ctx, length=0.0, thickness=6, position="Lower Right", color="#ffffff", show_label=True, font_size=14):
    img = ctx.img
    cal = img.calibration
    if length <= 0:
        width_units = img.width * cal.pixel_width
        raw = width_units / 5
        mag = 10 ** np.floor(np.log10(raw)) if raw > 0 else 1
        length = float(max([m for m in (1, 2, 5) if m * mag <= raw] or [1]) * mag)
    bar = {"id": roimod.new_id(), "type": "scalebar", "length": length, "unit": cal.unit, "px": length / cal.pixel_width, "thickness": thickness,
           "position": position, "color": color, "show_label": show_label, "font_size": font_size}
    overlay = [o for o in img.overlay if o.get("type") != "scalebar"] + [bar]
    return OpResult(overlay=overlay, message=f"Scale bar: {length:g} {cal.unit}")


def _burn_scalebar(rgb: np.ndarray, o: dict) -> None:
    H, W = rgb.shape[:2]
    px = int(round(o["px"]))
    th = int(o.get("thickness", 6))
    margin = max(int(min(H, W) * 0.03), 8)
    col = o.get("color", "#ffffff").lstrip("#")
    rgbcol = tuple(int(col[i:i + 2], 16) for i in (0, 2, 4)) if len(col) == 6 else (255, 255, 255)
    pos = o.get("position", "Lower Right")
    x0 = W - margin - px if "Right" in pos else margin
    y0 = H - margin - th if "Lower" in pos else margin
    rgb[max(y0, 0):y0 + th, max(x0, 0):x0 + px] = rgbcol
