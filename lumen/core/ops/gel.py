"""Analyze ▸ Gels: lane profiles and band quantification (the basics of ImageJ's Gel Analyzer)."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .. import roi as roimod
from ..registry import OpResult, Param, op


def lane_profile(plane: np.ndarray, roi: dict[str, Any], vertical: bool = True) -> np.ndarray:
    x0, y0 = int(round(roi["x"])), int(round(roi["y"]))
    x1, y1 = int(round(roi["x"] + roi["w"])), int(round(roi["y"] + roi["h"]))
    sub = plane[max(y0, 0):y1, max(x0, 0):x1].astype(np.float64)
    if sub.size == 0:
        raise ValueError("Lane rectangle is outside the image")
    return sub.mean(axis=1) if vertical else sub.mean(axis=0)


def clip_lane(roi: dict[str, Any], shape: tuple[int, ...]) -> dict[str, Any]:
    """The part of a lane rectangle that lies inside the image (ImageJ clips a rectangle ROI to the image the same way).

    ``lane_profile`` samples the clipped rectangle, so profile index 0 is the *clipped* origin: profiling the clipped
    lane keeps the band positions, the overlay labels (``roi.y + peak``) and the label column (``roi.x + roi.w``) in
    agreement for a lane dragged past an image edge – otherwise every label of a lane starting ``y0 < 0`` sits
    ``|y0|`` px above its band."""
    h, w = int(shape[0]), int(shape[1])
    x0 = min(max(float(roi["x"]), 0.0), float(w))
    y0 = min(max(float(roi["y"]), 0.0), float(h))
    x1 = min(max(float(roi["x"] + roi["w"]), 0.0), float(w))
    y1 = min(max(float(roi["y"] + roi["h"]), 0.0), float(h))
    return {**roi, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def band_areas(profile: np.ndarray, min_prominence: float) -> list[dict[str, Any]]:
    """Peaks of a lane profile with a straight baseline between the local minima on either side."""
    from scipy.signal import find_peaks
    rng = float(profile.max() - profile.min())
    if rng <= 0:
        return []
    peaks, props = find_peaks(profile, prominence=max(min_prominence * rng, 1e-9))
    out = []
    n = len(profile)
    for k, p in enumerate(peaks):
        left_edge = peaks[k - 1] if k > 0 else 0
        right_edge = peaks[k + 1] if k + 1 < len(peaks) else n - 1
        l = int(left_edge + np.argmin(profile[left_edge:p + 1]))
        r = int(p + np.argmin(profile[p:right_edge + 1]))
        if r <= l:
            continue
        xs = np.arange(l, r + 1)
        baseline = profile[l] + (profile[r] - profile[l]) * (xs - l) / max(r - l, 1)
        area = float(np.sum(np.maximum(profile[l:r + 1] - baseline, 0)))
        out.append({"peak": int(p), "left": l, "right": r, "area": area, "height": float(profile[p] - baseline[p - l]), "prominence": float(props["prominences"][k])})
    return out


@op("analyze.gel_lanes", "Gel Analyzer (lanes)", "Analyze/Gels", kind="analysis", preview=False, roi_aware=False, supports_rgb=False,
    params=[Param("orientation", "choice", "Vertical", choices=["Vertical", "Horizontal"], help="Direction of migration (profile axis)"),
            Param("dark_bands", "bool", True, help="Dark bands on a light background (profile is inverted so bands become peaks)"),
            Param("min_prominence", "float", 0.05, min=0.0, max=1.0, step=0.01, help="Minimum peak prominence as a fraction of the profile range"),
            Param("smooth", "float", 1.0, min=0, max=20, step=0.5, help="Gaussian smoothing (px) of the profile before peak detection"),
            Param("only_this_image", "bool", False),
            Param("percent_per_lane", "bool", False, help="Percent of the lane's own band total instead of the total of all bands in all lanes (ImageJ)")],
    description="Treat every rectangle in the ROI Manager (or the selection) as a gel lane: plot the lane profiles and quantify each band as the area above a straight baseline between the neighbouring minima "
                "(Percent = share of the total of all bands in all lanes, as in ImageJ's Gel Analyzer).",
    keywords=["gel", "lanes", "bands", "densitometry", "western"])
def gel_lanes(ctx, orientation="Vertical", dark_bands=True, min_prominence=0.05, smooth=1.0, only_this_image=False, percent_per_lane=False):
    img = ctx.img
    s = ctx.session
    lanes = [r for r in s.roi_manager.to_list() if r.get("type") == "rect" and (not only_this_image or r.get("image_id") in (None, img.id))]
    if not lanes and ctx.roi is not None and ctx.roi.get("type") == "rect":
        lanes = [roimod.normalize(ctx.roi)]
    if not lanes:
        raise ValueError("Add one rectangle per lane to the ROI Manager (or draw one rectangle) first")
    if orientation == "Vertical":
        lanes = sorted(lanes, key=lambda r: r["x"])
    else:
        lanes = sorted(lanes, key=lambda r: r["y"])
    plane = img.current_plane()
    lanes = [clip_lane(r, plane.shape) for r in lanes]  # profile index 0 = the lane's first in-image row / column
    top = float(plane.max()) if plane.dtype == np.float32 else (255.0 if plane.dtype == np.uint8 else 65535.0)
    cal = img.calibration
    step = cal.pixel_height if orientation == "Vertical" else cal.pixel_width
    rows: list[dict[str, Any]] = []
    series = []
    x_axis: list[float] = []
    detected: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for li, roi in enumerate(lanes, 1):
        prof = lane_profile(plane, roi, vertical=(orientation == "Vertical"))
        if dark_bands:
            prof = top - prof
        if smooth > 0:
            prof = ndi.gaussian_filter1d(prof, smooth)
        detected.append((li, roi, band_areas(prof, min_prominence)))
        series.append({"name": roi.get("name") or f"Lane {li}", "values": prof.tolist()})
        if len(prof) > len(x_axis):
            x_axis = [i * step for i in range(len(prof))]
    # ImageJ (Label Peaks): Percent = area / total of all measured peaks in all lanes, so lanes are comparable
    grand = sum(b["area"] for _, _, bands in detected for b in bands) or 1.0
    for li, roi, bands in detected:
        total = (sum(b["area"] for b in bands) or 1.0) if percent_per_lane else grand
        name = roi.get("name") or f"Lane {li}"
        for bi, b in enumerate(bands, 1):
            rows.append({"Label": f"{img.name}:{name}", "Lane": li, "Peak": bi, "Position": b["peak"] * step, "Start": b["left"] * step, "End": b["right"] * step,
                         "Height": b["height"], "Area": b["area"], "Percent": 100.0 * b["area"] / total})
    plot = {"title": f"Lane profiles of {img.name}", "x": x_axis, "series": series, "x_label": f"Distance ({cal.unit})", "y_label": "Intensity (inverted)" if dark_bands else "Intensity"}
    return OpResult(results=rows, plot=plot, message=f"{len(lanes)} lanes, {len(rows)} bands")


# ---------------------------------------------------------------- Fiji-style Gel Analyzer workflow
def gel_lanes_of(session, img, only_this_image: bool = True) -> list[dict[str, Any]]:
    """The lane rectangles of an image: ROI-manager rectangles tagged by Select Lane (``gel_lane``), else every rectangle."""
    rects = [r for r in session.roi_manager.to_list() if r.get("type") == "rect" and (not only_this_image or r.get("image_id") in (None, img.id))]
    tagged = [r for r in rects if r.get("gel_lane")]
    lanes = tagged or rects
    return sorted(lanes, key=lambda r: (r.get("gel_lane") or 0, r["x"], r["y"]))


@op("analyze.gel_select_lane", "Gels ▸ Select Lane", "Analyze/Gels", kind="analysis", preview=False, roi_aware=False, requires_roi="area",
    params=[Param("color", "string", "", help="Outline colour (empty = session ROI colour)")],
    description="Add the current rectangle selection to the ROI Manager as the next numbered gel lane (Select First Lane / Select Next Lane).",
    keywords=["gel", "lane", "first lane", "next lane"])
def gel_select_lane(ctx, color=""):
    img = ctx.img
    s = ctx.session
    roi = roimod.normalize(ctx.roi)
    if roi.get("type") != "rect":
        raise ValueError("Gel lanes must be rectangle selections")
    existing = gel_lanes_of(s, img)
    n = max([int(r.get("gel_lane") or 0) for r in existing] + [0]) + 1
    if n > 1:
        ref = existing[-1]
        if abs(ref["w"] - roi["w"]) > 1e-6 or abs(ref["h"] - roi["h"]) > 1e-6:
            s.add_log(f"Lane {n} has a different size than lane {n - 1} ({roi['w']:.0f}×{roi['h']:.0f} vs {ref['w']:.0f}×{ref['h']:.0f})", level="warning")
    lane = {**{k: v for k, v in roi.items() if k != "id"}, "name": f"Lane {n}", "gel_lane": n,
            "color": color or s.settings.get("roi_color", "#ffd700"), "stroke_width": s.settings.get("roi_stroke_width", 1.5)}
    return OpResult(rois_add=[lane], message=f"Lane {n} selected ({roi['w']:.0f}×{roi['h']:.0f} px)")


@op("analyze.gel_plot_lanes", "Gels ▸ Plot Lanes", "Analyze/Gels", kind="analysis", preview=False, roi_aware=False, supports_rgb=False,
    params=[Param("orientation", "choice", "Vertical", choices=["Vertical", "Horizontal"], help="Direction of migration"),
            Param("dark_bands", "bool", True, help="Invert so that dark bands become peaks"),
            Param("smooth", "float", 0.0, min=0, max=20, step=0.5), Param("make_image", "bool", True, help="Also draw the lane profiles into a new 8-bit plot image (Fiji style: one profile below the other)")],
    description="Plot the density profile of every selected gel lane (ROI Manager rectangles) – the lanes are stacked one below the other like Fiji's Gel Analyzer plot.",
    keywords=["gel", "lanes", "profile", "densitometry"])
def gel_plot_lanes(ctx, orientation="Vertical", dark_bands=True, smooth=0.0, make_image=True):
    img = ctx.img
    s = ctx.session
    lanes = gel_lanes_of(s, img)
    if not lanes and ctx.roi is not None and ctx.roi.get("type") == "rect":
        lanes = [roimod.normalize(ctx.roi)]
    if not lanes:
        raise ValueError("Select the lanes first (Gels ▸ Select Lane with a rectangle)")
    plane = img.current_plane()
    lanes = [clip_lane(r, plane.shape) for r in lanes]  # profile index 0 = the lane's first in-image row / column
    top = float(plane.max()) if plane.dtype == np.float32 else (255.0 if plane.dtype == np.uint8 else 65535.0)
    cal = img.calibration
    step = cal.pixel_height if orientation == "Vertical" else cal.pixel_width
    profiles = []
    for roi in lanes:
        prof = lane_profile(plane, roi, vertical=(orientation == "Vertical"))
        if dark_bands:
            prof = top - prof
        if smooth > 0:
            prof = ndi.gaussian_filter1d(prof, smooth)
        profiles.append(prof)
    n = max(len(p) for p in profiles)
    x = [i * step for i in range(n)]
    series = [{"name": r.get("name") or f"Lane {i + 1}", "values": p.tolist()} for i, (r, p) in enumerate(zip(lanes, profiles))]
    plot = {"title": f"Lane profiles of {img.name}", "x": x, "series": series, "x_label": f"Distance ({cal.unit})", "y_label": "Density" if dark_bands else "Intensity"}
    s.settings["gel_profiles"] = {"image_id": img.id, "orientation": orientation, "dark_bands": dark_bands, "smooth": smooth,
                                  "lanes": [dict(r) for r in lanes], "profiles": [p.tolist() for p in profiles]}
    out = OpResult(plot=plot, message=f"{len(lanes)} lane profile(s)")
    if make_image:
        out.new_images = [lane_plot_image(profiles, [sr["name"] for sr in series], f"Plots of {img.name}")]
    return out


def lane_plot_image(profiles: list[np.ndarray], names: list[str], title: str, height: int = 120, margin: int = 10):
    """8-bit plot image with one profile per row block (black curve on white), Fiji Gel Analyzer style."""
    from ..image import LumenImage
    from ..overlay_render import draw_text_into_plane
    n = max(len(p) for p in profiles)
    width = max(n + 2 * margin, 64)
    H = len(profiles) * (height + margin) + margin
    out = np.full((H, width), 255, np.uint8)
    vmax = max(float(np.max(p)) for p in profiles) or 1.0
    vmin = min(float(np.min(p)) for p in profiles)
    span = (vmax - vmin) or 1.0
    for k, p in enumerate(profiles):
        y_top = margin + k * (height + margin)
        base = y_top + height
        out[base, margin:margin + n] = 0  # baseline
        ys = base - np.round((p - vmin) / span * (height - 2)).astype(int)
        xs = margin + np.arange(len(p))
        from skimage.draw import line as sk_line
        for i in range(len(p) - 1):
            rr, cc = sk_line(int(ys[i]), int(xs[i]), int(ys[i + 1]), int(xs[i + 1]))
            out[np.clip(rr, 0, H - 1), np.clip(cc, 0, width - 1)] = 0
        out = draw_text_into_plane(out, names[k], margin + 2, y_top + 2, 0, font_size=12)
    img = LumenImage(out, name=title)
    img.metadata["gel_plot"] = True
    img.metadata["slice_labels"] = list(names)
    return img


@op("analyze.gel_label_peaks", "Gels ▸ Label Peaks", "Analyze/Gels", kind="image", preview=False, roi_aware=False, scope_param=False, supports_rgb=False,
    params=[Param("min_prominence", "float", 0.05, min=0.0, max=1.0, step=0.01, help="Minimum peak prominence as a fraction of the profile range"),
            Param("orientation", "choice", "Vertical", choices=["Vertical", "Horizontal"]), Param("dark_bands", "bool", True), Param("smooth", "float", 1.0, min=0, max=20, step=0.5),
            Param("label", "choice", "Area", choices=["Area", "Percent", "Peak number"], help="Percent = share of the total of all bands in all lanes (ImageJ 'Label with percentages')"),
            Param("font_size", "int", 0, min=0, max=72, help="0 = session font size"),
            Param("color", "string", "", help="Text colour (empty = session ROI colour)"),
            Param("percent_per_lane", "bool", False, help="Percent of the lane's own band total instead of the total of all bands in all lanes (ImageJ)")],
    description="Integrate every band of every lane between its baseline points (the neighbouring minima) → Results rows (Lane, Peak, Area, Percent of all bands in all lanes) and a text overlay with the value at each band.",
    keywords=["gel", "peaks", "integrate", "bands"], undo_label="Label peaks")
def gel_label_peaks(ctx, min_prominence=0.05, orientation="Vertical", dark_bands=True, smooth=1.0, label="Area", font_size=0, color="", percent_per_lane=False):
    img = ctx.img
    s = ctx.session
    lanes = gel_lanes_of(s, img)
    if not lanes and ctx.roi is not None and ctx.roi.get("type") == "rect":
        lanes = [roimod.normalize(ctx.roi)]
    if not lanes:
        raise ValueError("Select the lanes first (Gels ▸ Select Lane with a rectangle)")
    plane = img.current_plane()
    lanes = [clip_lane(r, plane.shape) for r in lanes]  # profile index 0 = the lane's first in-image row / column
    top = float(plane.max()) if plane.dtype == np.float32 else (255.0 if plane.dtype == np.uint8 else 65535.0)
    cal = img.calibration
    vertical = orientation == "Vertical"
    step = cal.pixel_height if vertical else cal.pixel_width
    fs = int(font_size) if font_size > 0 else int(s.settings.get("font_size", 12))
    col = color or s.settings.get("roi_color", "#ffd700")
    rows: list[dict[str, Any]] = []
    items: list[dict[str, Any]] = []
    detected: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for li, roi in enumerate(lanes, 1):
        prof = lane_profile(plane, roi, vertical=vertical)
        if dark_bands:
            prof = top - prof
        if smooth > 0:
            prof = ndi.gaussian_filter1d(prof, smooth)
        detected.append((li, roi, band_areas(prof, min_prominence)))
    # ImageJ Label Peaks: the percentage is the band area divided by the sum of all measured peaks from all lanes
    # (so lanes are comparable, e.g. against a loading control); the overlay text embeds it, hence the second pass
    grand = sum(b["area"] for _, _, bands in detected for b in bands) or 1.0
    for li, roi, bands in detected:
        total = (sum(b["area"] for b in bands) or 1.0) if percent_per_lane else grand
        name = roi.get("name") or f"Lane {li}"
        for bi, b in enumerate(bands, 1):
            pct = 100.0 * b["area"] / total
            rows.append({"Label": f"{img.name}:{name}", "Lane": li, "Peak": bi, "Position": b["peak"] * step, "Start": b["left"] * step, "End": b["right"] * step,
                         "Height": b["height"], "Area": b["area"], "Percent": pct})
            txt = f"{b['area']:.0f}" if label == "Area" else (f"{pct:.1f}%" if label == "Percent" else str(bi))
            if vertical:
                x, y = roi["x"] + roi["w"] + 2, roi["y"] + b["peak"] - fs / 2
            else:
                x, y = roi["x"] + b["peak"] - fs / 2, roi["y"] + roi["h"] + 2
            items.append({"type": "text", "id": roimod.new_id(), "x": float(x), "y": float(y), "text": txt, "font_size": fs, "color": col, "gel_peak": True})
    keep = [o for o in img.overlay if not o.get("gel_peak")]
    return OpResult(results=rows, overlay=keep + items, message=f"{len(rows)} band(s) in {len(lanes)} lane(s)")
