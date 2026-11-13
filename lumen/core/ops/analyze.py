"""Analyze menu."""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage as ndi

from .. import roi as roimod
from ..image import LumenImage
from ..measure import ALL_MEASUREMENTS, MEASUREMENT_LABELS, measure_labels, measure_particles, measure_roi
from ..profile import line_profile, rect_profile
from ..registry import Param, OpResult, cast_to, op
from ..threshold import METHODS, auto_threshold, histogram256
from . import _imagej_measure as ijm
from ._util import current_threshold, label_image, binary_image, is_binary_plane

# analyze_particles has a parameter called auto_threshold (ImageJ's behaviour is opt-in); the function keeps this alias
_auto_threshold = auto_threshold

CAT = "Analyze"

# spellings of length units as they appear in file metadata / macros -> the display form used in messages and tables
_UNIT_ALIASES = {
    "micron": "µm", "microns": "µm", "um": "µm", "μm": "µm", "µm": "µm", "micrometer": "µm", "micrometers": "µm", "micrometre": "µm", "micrometres": "µm",
    "nanometer": "nm", "nanometers": "nm", "nanometre": "nm", "nanometres": "nm", "millimeter": "mm", "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "centimeter": "cm", "centimeters": "cm", "centimetre": "cm", "centimetres": "cm", "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "inches": "inch", "angstrom": "Å", "angstroms": "Å", "pixel": "pixel", "pixels": "pixel", "px": "pixel", "": "pixel",
}


def normalize_unit(unit: str | None) -> str:
    """Canonical spelling of a length unit for display and comparison: 'micron', 'um', 'μm' (Greek mu) → 'µm',
    'pixels'/'' → 'pixel', 'millimeter' → 'mm' …; unknown units are returned stripped but otherwise unchanged.
    (The io/image layer still stores whatever the file said – normalising there is pending.)"""
    u = (unit or "").strip()
    return _UNIT_ALIASES.get(u.lower(), u)


def redirect_image(session, img: LumenImage) -> LumenImage | None:
    """The 'Redirect to' image of Set Measurements (session.measure_options['redirect']), or None.

    ImageJ's Analyzer measures the selection of the current image on the redirect image's pixels; it must have the
    same width and height (ImageJ: "Redirect image ... must have the same dimensions").
    """
    key = session.measure_options.get("redirect")
    if not key:
        return None
    try:
        target = session.get_image(key)
    except Exception:
        raise ValueError(f"Redirect image {key!r} (Set Measurements ▸ Redirect to) is not open") from None
    if target.id == img.id:
        return None
    if (target.width, target.height) != (img.width, img.height):
        raise ValueError(f"Redirect image {target.name!r} must have the same size as {img.name!r} ({img.width}×{img.height})")
    return target


def _redirect_position(target: LumenImage, t: int, z: int, c: int) -> tuple[int, int, int]:
    """Plane of the redirect image matching (t, z, c) of the measured image (clamped to the target's dimensions)."""
    return min(t, target.nt - 1), min(z, target.nz - 1), min(c, target.nc - 1)


#: the channel to measure (1-based, 0 = the current one); recorded explicitly on multi-channel images (E7-11)
CHANNEL_PARAM = Param("channel", "int", 0, min=0, advanced=True, help="1-based channel to measure; 0 = the current channel")


def _channel_note(img: LumenImage, cc: int) -> str:
    """" on channel 1 (DAPI)" for a multi-channel image, "" otherwise – so a toast or message says which channel was
    analysed (E9-2); the Results rows carry it in the Ch column."""
    if img.nc <= 1 or img.rgb:
        return ""
    name = ""
    try:
        name = img.display.channels[cc].name or ""
    except Exception:  # noqa: BLE001
        pass
    return f" on channel {cc + 1}" + (f" ({name})" if name else "")


def _channel_index(img: LumenImage, channel) -> int:
    c = int(channel or 0)
    if c == 0:
        return img.display.c
    if c > img.nc:
        raise ValueError(f"{img.name} has {img.nc} channel(s), channel {c} does not exist")
    return c - 1


@op("analyze.measure", "Measure", CAT, kind="analysis", preview=False, shortcut="M", roi_aware=True,
    description="Measure the selection (or the whole plane) and append a row to the Results table (on the 'Redirect to' image of Set Measurements when one is set).",
    keywords=["area", "mean", "intensity", "redirect"], params=[CHANNEL_PARAM])
def measure(ctx, channel=0):
    s = ctx.session
    img = ctx.img
    image_thr = current_threshold(ctx)   # ImageJ's %Area counts the thresholded pixels whenever a threshold is set
    thr = image_thr if s.measure_options.get("limit_to_threshold") else None
    target = redirect_image(s, img)
    d = img.display
    cc = _channel_index(img, channel)
    if target is not None:
        # ImageJ: the selection comes from the current image, the pixel values (and the Label) from the redirect image
        t, z, c = _redirect_position(target, d.t, d.z, cc)
        rows = measure_roi(target, ctx.roi, s.measurements, t=t, z=z, c=c, limit_to_threshold=thr, options=s.measure_options, threshold=image_thr)
    else:
        rows = measure_roi(img, ctx.roi, s.measurements, t=d.t, z=d.z, c=cc, limit_to_threshold=thr, options=s.measure_options, threshold=image_thr)
    res = OpResult(results=rows, message=(f"Measured{_channel_note(img, cc)}" if _channel_note(img, cc) else None))
    if s.measure_options.get("add_to_overlay") and ctx.roi is not None:
        from .. import roi as roimod
        item = roimod.normalize(ctx.roi)
        item["color"] = s.settings.get("roi_color", "#ffd54a")
        res.overlay = [*img.overlay, item]
    return res


@op("analyze.measure_stack", "Measure Stack", CAT, kind="analysis", preview=False,
    params=[Param("axis", "choice", "Z", choices=["Z", "T", "Z and T"])], description="Measure the selection on every slice / frame of the current channel.")
def measure_stack(ctx, axis="Z"):
    s = ctx.session
    img = ctx.img
    d = img.display
    rows = []
    zs = range(img.nz) if axis in ("Z", "Z and T") else [d.z]
    ts = range(img.nt) if axis in ("T", "Z and T") else [d.t]
    meas = list(dict.fromkeys(s.measurements + ["stack_position"]))
    target = redirect_image(s, img)
    image_thr = current_threshold(ctx)
    thr = image_thr if s.measure_options.get("limit_to_threshold") else None
    for t in ts:
        for z in zs:
            if target is not None:
                tt, zz, cc = _redirect_position(target, t, z, d.c)
                rows += measure_roi(target, ctx.roi, meas, t=tt, z=zz, c=cc, limit_to_threshold=thr, options=s.measure_options, threshold=image_thr)
            else:
                rows += measure_roi(img, ctx.roi, meas, t=t, z=z, c=d.c, limit_to_threshold=thr, options=s.measure_options, threshold=image_thr)
    return OpResult(results=rows)


@op("analyze.set_measurements", "Set Measurements", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("measurements", "multichoice", ["area", "mean", "min", "max"], choices=ALL_MEASUREMENTS),
            Param("limit_to_threshold", "bool", False), Param("decimals", "int", 3, min=0, max=9),
            Param("display_label", "bool", True, help="Include the image/ROI label column"), Param("invert_y", "bool", False, help="Y coordinates measured from the bottom edge"),
            Param("scientific_notation", "bool", False, help="Format hint for the Results panel"), Param("add_to_overlay", "bool", False, help="Measure also adds the selection to the overlay"),
            Param("nan_empty_cells", "bool", False, help="Report NaN instead of leaving a cell empty"),
            Param("redirect", "string", "", label="Redirect to", help="Image (name or id) whose pixel values Measure / Analyze Particles read for the selection; empty = none")],
    description="Choose which columns Measure and Analyze Particles report, and how coordinates and labels are reported.")
def set_measurements(ctx, measurements=None, limit_to_threshold=False, decimals=3, display_label=True, invert_y=False, scientific_notation=False, add_to_overlay=False, nan_empty_cells=False,
                     redirect=""):
    s = ctx.session
    s.measurements = [m for m in (measurements or []) if m in ALL_MEASUREMENTS] or ["area"]
    s.measure_options.update(limit_to_threshold=limit_to_threshold, decimals=decimals, display_label=display_label, invert_y=invert_y,
                             scientific_notation=scientific_notation, add_to_overlay=add_to_overlay, nan_empty_cells=nan_empty_cells)
    redirect = str(redirect or "").strip()
    if redirect and redirect.lower() != "none":
        try:
            target = s.get_image(redirect)
        except Exception:
            raise ValueError(f"Redirect image {redirect!r} is not open") from None
        s.measure_options["redirect"] = target.id  # the id survives renames; resolved again at measure time
        extra = f" (redirected to {target.name})"
    else:
        s.measure_options["redirect"] = ""
        extra = ""
    return OpResult(message="Measurements: " + ", ".join(MEASUREMENT_LABELS[m] for m in s.measurements) + extra)


@op("analyze.clear_results", "Clear Results", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    description="Empty the Results table (one step restorable with results.restore_cleared).")
def clear_results(ctx):
    n = ctx.session.clear_results(record=False)  # restorable (results.restore_cleared); run_op records the line
    return OpResult(message=f"Results cleared ({n} row(s); Results ▸ Restore Cleared puts them back)" if n else "Results cleared")


def _show_in_situ(session, img: LumenImage, data: np.ndarray, show: str, plane_index: tuple[int, int, int], all_planes: bool) -> None:
    """ImageJ 'In situ Show': the Show image replaces the processed plane(s) of the source image (undoable)."""
    from ..undo import History
    snap = History.snapshot(img, "Analyze Particles (in situ)")
    labels = show in ("Count Masks", "Labels")
    n = int(data.max()) if (labels and data.size) else 0
    label_dtype = np.uint8 if n < 256 else np.uint16 if n < 65536 else np.float32  # as label_image: a count mask never wraps
    if all_planes:
        new = data.astype(label_dtype if labels else np.uint8)
    else:
        new = img.data
        if labels and np.issubdtype(new.dtype, np.integer) and n > np.iinfo(new.dtype).max:
            new = new.astype(label_dtype)  # more particles than the image's dtype can number
        new = new.copy()
        t, z, c = plane_index
        new[t, z, c] = cast_to(data, new.dtype)
    img.replace_data(new)
    img.lut_is_binary = not labels
    for ch in img.display.channels:
        if labels:
            ch.lut = "Glasbey on dark"  # Fiji-exact Glasbey has a white index 0; the dark variant keeps the background black
            ch.min, ch.max = 0.0, float(max(int(new.max()), 1))
        elif new.dtype != np.uint8:
            ch.min, ch.max = 0.0, 255.0
    hist = session.histories.get(img.id)
    if hist is not None:
        hist.undo.append(snap)
        hist.redo.clear()
        hist._trim()


@op("analyze.particles", "Analyze Particles", CAT, kind="analysis", preview=False, supports_rgb=False,
    description="Find connected objects in a binary (or thresholded) image, filter them by size and circularity, measure them, and add outlines to the ROI manager.",
    params=[Param("min_size", "float", 0.0, min=0, step=1, label="Min size (calibrated units²)"), Param("max_size", "float", 0.0, min=0, step=1, label="Max size (0 = ∞)"),
            Param("min_circularity", "float", 0.0, min=0, max=1, step=0.05), Param("max_circularity", "float", 1.0, min=0, max=1, step=0.05),
            Param("show", "choice", "Nothing", choices=["Nothing", "Outlines", "Bare Outlines", "Masks", "Count Masks", "Labels", "Ellipses"], help="Extra output image (ImageJ's 'Show:'; Labels = Count Masks)"),
            Param("display_results", "bool", True), Param("clear_results", "bool", False), Param("summarize", "bool", True), Param("add_to_manager", "bool", True),
            Param("exclude_on_edges", "bool", False), Param("include_holes", "bool", True), Param("all_planes", "bool", False, label="Process whole stack"),
            Param("intensity_image", "image", None, label="Measure intensities on", help="Optional: an open image to read gray values from (e.g. before thresholding); default = the Set Measurements 'Redirect to' image"),
            Param("pixel_units", "bool", False, help="Size and circularity limits in pixels² / pixels instead of calibrated units (ImageJ 'Pixel units')"),
            Param("record_starts", "bool", False, help="Add XStart / YStart (first pixel of each particle, pixel coordinates, for doWand) to the Results"),
            Param("in_situ_show", "bool", False, label="In situ show", help="The Show image replaces the processed plane(s) of this image instead of opening a new window"),
            Param("composite_rois", "bool", True, help="Particles with holes (Include holes off) become exact mask ROIs; off = outer outline polygons (ImageJ's default ROIs)"),
            Param("auto_threshold", "bool", False, label="Auto-threshold if needed", advanced=True,
                  help="Off (ImageJ): a thresholded or 8-bit binary image is required. On: an unthresholded gray image gets ImageJ's Default auto threshold on each plane"),
            CHANNEL_PARAM],
    keywords=["count", "segment", "objects", "cells", "nuclei"])
def analyze_particles(ctx, min_size=0.0, max_size=0.0, min_circularity=0.0, max_circularity=1.0, show="Outlines", display_results=True, clear_results=False,
                      summarize=True, add_to_manager=True, exclude_on_edges=False, include_holes=True, all_planes=False, intensity_image=None,
                      pixel_units=False, record_starts=False, in_situ_show=False, composite_rois=True, auto_threshold=False, channel=0):
    s = ctx.session
    img = ctx.img
    d = img.display
    cc = _channel_index(img, channel)
    cal = img.calibration
    cal_area = cal.pixel_width * cal.pixel_height
    unit = normalize_unit(cal.unit)
    size_unit = "pixel" if pixel_units else unit
    unit_squared = 1.0 if pixel_units else cal_area        # ParticleAnalyzer: the size limits are converted to pixels
    min_px = min_size / unit_squared
    max_px = (max_size / unit_squared) if max_size > 0 else float("inf")
    thr = current_threshold(ctx)
    mask_roi = ctx.mask
    intensity = s.get_image(intensity_image) if intensity_image else redirect_image(s, img)
    if intensity is not None and (intensity.width, intensity.height) != (img.width, img.height):
        raise ValueError(f"Intensity image {intensity.name!r} must have the same size as {img.name!r}")
    planes = list(img.iter_planes()) if all_planes else [(d.t, d.z, cc)]
    if thr is None and not auto_threshold and not all(is_binary_plane(img.data[t, z, c]) for (t, z, c) in planes):
        # ImageJ: "A thresholded image or 8-bit binary image is required."  The silent Default auto threshold
        # that used to run here hid wrong counts (docs/USABILITY.md M2); it is now the explicit auto_threshold option.
        raise ValueError("Analyze Particles needs a thresholded or binary image (Image ▸ Adjust ▸ Threshold…), "
                         "or tick 'Auto-threshold if needed' to apply the Default auto threshold")
    show = {"Labels": "Count Masks"}.get(show, show)
    count_masks = show == "Count Masks"
    drawing = show in ("Outlines", "Bare Outlines", "Ellipses")
    # ImageJ draws the Outlines / Ellipses image black on white; in situ with a black background it is white on black
    draw_bg, draw_fg = (0, 255) if (in_situ_show or not drawing) else (255, 0)
    all_rows: list[dict] = []
    rois_add: list[dict] = []
    summary_rows = []
    # count masks are numbered in int32 here; the output picks uint16 (ImageJ's ShortProcessor) or float32 beyond 65535 particles
    out_data = np.zeros(img.shape, np.int32 if count_masks else np.uint8) if show != "Nothing" and all_planes else None
    out_plane = None
    total_count = 0
    H, W = img.height, img.width
    rect = (0, 0, W, H)
    if mask_roi is not None and mask_roi.any():
        ys, xs = np.nonzero(mask_roi)
        rect = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    for (t, z, c) in planes:
        p = img.data[t, z, c]
        if thr is not None:
            b = (p >= thr[0]) & (p <= thr[1])
        elif is_binary_plane(p):
            b = p > 0
        else:  # auto_threshold=True (checked above)
            lo, hi = _auto_threshold(p, "Default", mask_roi, True)
            b = (p >= lo) & (p <= hi)
        if mask_roi is not None:
            b &= mask_roi
        parts = ijm.find_particles(b, rect=rect, include_holes=include_holes, exclude_on_edges=exclude_on_edges, min_size=min_px, max_size=max_px,
                                   min_circ=min_circularity, max_circ=max_circularity)
        n = len(parts)
        meas = list(dict.fromkeys(s.measurements + (["stack_position"] if all_planes else [])))
        ip = intensity.data[min(t, intensity.nt - 1), min(z, intensity.nz - 1), min(c, intensity.nc - 1)] if intensity is not None else None
        rows = measure_particles(img, parts, meas, t, z, c, intensity_plane=ip, options=s.measure_options, threshold=thr, first_number=total_count + 1)
        if record_starts:
            for part, r in zip(parts, rows):
                r["XStart"], r["YStart"] = part.start
        all_rows += rows   # ImageJ (batch mode / no Results window) records the rows even without 'Display results'
        if summarize:
            areas = [part.pixel_count * cal_area for part in parts]
            means = [r.get("Mean", float("nan")) for r in rows if "Mean" in r]
            summary_rows.append({"Slice": f"{img.name} t{t + 1} z{z + 1} c{c + 1}" if all_planes else img.name, "Count": n,
                                 "Total Area": float(np.sum(areas)) if areas else 0.0, "Average Size": float(np.mean(areas)) if areas else 0.0,
                                 "%Area": 100.0 * float(np.sum(areas)) / (img.width * img.height * cal_area) if areas else 0.0,
                                 "Mean": float(np.mean(means)) if means else float("nan")})
        if add_to_manager and n:
            for k, part in enumerate(parts, total_count + 1):
                bx, by, bw, bh = part.bounds
                name = f"{k:04d}-{by + bh // 2:04d}-{bx + bw // 2:04d}"   # ImageJ RoiManager label: number-yc-xc
                if composite_rois and not include_holes and part.mask is not None and not part.mask.all():
                    new = roimod._crop_mask_roi(part.mask, bx, by, name)     # exact pixels, holes excluded
                else:
                    new = {"id": roimod.new_id(), "type": "polygon", "name": name,
                           "points": [[float(x), float(y)] for x, y in zip(part.outline_x, part.outline_y)]}   # the traced outline
                if new is not None:
                    new["position"] = {"t": t, "z": z, "c": c}
                    rois_add.append(new)
        if show != "Nothing":
            plane_out = np.full((H, W), draw_bg, np.int32 if count_masks else np.uint8)
            if count_masks:
                plane_out[:] = 0
            for k, part in enumerate(parts, total_count + 1):
                bx, by, bw, bh = part.bounds
                sub = plane_out[by:by + bh, bx:bx + bw]
                if count_masks or show == "Masks":
                    value = k if count_masks else 255
                    if part.mask is None:
                        sub[:] = value
                    else:
                        sub[part.mask] = value
                elif show == "Ellipses":
                    ef = ijm.ellipse_fitter(part.mask, bx, by, bw, bh)
                    ijm.draw_polyline(plane_out, ijm.ellipse_outline_points(ef, H), draw_fg)
                else:  # Outlines / Bare Outlines (the particle numbers of 'Outlines' are not drawn)
                    ijm.draw_polyline(plane_out, list(zip(part.outline_x, part.outline_y)), draw_fg, close=True)
            if out_data is not None:
                out_data[t, z, c] = plane_out
            else:
                out_plane = plane_out
        total_count += n
    log = f"{total_count} particles" + _channel_note(img, cc)
    msg = log
    if min_size > 0 or max_size > 0:
        hi = f"{max_size:g}" if max_size > 0 else "∞"
        msg += f" (size {min_size:g}–{hi} {size_unit}²" + (f", circularity {min_circularity:g}–{max_circularity:g}" if (min_circularity > 0 or max_circularity < 1) else "") + ")"
    elif min_circularity > 0 or max_circularity < 1:
        msg += f" (circularity {min_circularity:g}–{max_circularity:g})"
    if intensity is not None:
        msg += f", intensities from {intensity.name}"
    res = OpResult(results=all_rows, results_replace=clear_results, rois_add=rois_add, message=msg, log=log)
    if summarize:
        res.table = {"name": "Particle Summary", "columns": list(summary_rows[0].keys()) if summary_rows else [], "rows": [list(r.values()) for r in summary_rows], "unit": unit}
    if show != "Nothing":
        data = out_data if out_data is not None else out_plane
        if in_situ_show:
            _show_in_situ(s, img, data, show, planes[0], all_planes)
            res.message += " – shown in situ"
        else:
            # ImageJ's output window ("Mask of …", "Count Masks of …" 16-bit; ImageJ calls the outline / ellipse
            # drawing "Drawing of …") becomes the current image
            if count_masks:
                # ImageJ's count mask is 16-bit (counts clamp at 65535); above that Lumen keeps the exact numbers in float32
                out_img = LumenImage(data.astype(np.uint16 if total_count < 65536 else np.float32), name=f"Count Masks of {img.name}")
                out_img.calibration = type(cal)(**cal.to_dict())
                ch = out_img.display.channels[0]
                ch.lut = "Glasbey on dark"
                ch.min, ch.max = 0.0, float(max(total_count, 1))
            elif show == "Masks":
                out_img = binary_image(data, f"Mask of {img.name}", like=img)
                out_img.display.channels[0].inverted = True   # ImageJ: an inverting LUT (particles 255, shown black on white)
            else:
                out_img = LumenImage(data.astype(np.uint8), name=f"{show} of {img.name}")
                out_img.calibration = type(cal)(**cal.to_dict())
            res.new_images = [out_img]
    return res


@op("analyze.histogram", "Histogram", CAT, kind="analysis", preview=False, shortcut="H",
    description="Histogram of the current plane (or selection) with statistics – ImageJ's Histogram dialog: bins, fixed X range, fixed Y max, log scale, stack histogram, list to Results.",
    params=[Param("bins", "int", 256, min=2, max=65536),
            Param("all_planes", "bool", False, label="Stack histogram", help="All slices and frames of the current channel (ImageJ 'Stack histogram')"),
            Param("x_min", "float", 0.0, help="Fixed value range; x_min = x_max → use the pixel value range"), Param("x_max", "float", 0.0),
            Param("y_max", "float", 0.0, help="Fixed count axis maximum (0 = auto)"),
            Param("log_scale", "bool", False, help="Draw the counts on a logarithmic axis (stored with the histogram)"),
            Param("copy_to_results", "bool", False, help="Bin starts and counts → Results table (ImageJ 'List')")],
    keywords=["histogram", "stack", "log", "list"])
def histogram(ctx, bins=256, all_planes=False, x_min=0.0, x_max=0.0, y_max=0.0, log_scale=False, copy_to_results=False):
    img = ctx.img
    d = img.display
    mask = ctx.mask
    if all_planes:
        sub = img.data[:, :, d.c]  # (T, Z, H, W) of the current channel
        vals = sub[..., mask].ravel() if mask is not None else sub.ravel()
    else:
        p = img.current_plane()
        vals = p[mask] if mask is not None else p.ravel()
    if vals.dtype.kind == "f":
        vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No pixels to histogram")
    fixed = x_min != x_max
    if fixed:
        lo, hi = (float(min(x_min, x_max)), float(max(x_min, x_max)))
    if not fixed and img.data.dtype == np.uint8 and bins == 256:
        counts = np.bincount(vals.astype(np.int64), minlength=256)[:256]
        edges = np.arange(257)
    elif not all_planes and img.data.dtype in (np.uint8, np.uint16, np.float32):
        # ImageJ's bins (ImagePlus.getStatistics(…, nBins, histMin, histMax)): a 16-bit bin is (max − min + 1)/nBins wide,
        # an 8-bit image with another bin count is treated as 16-bit over 0–256, 32-bit bins are (max − min)/nBins
        src = p if img.data.dtype != np.uint8 else p.astype(np.uint16)
        rect, rmask = roimod.roi_rect_mask(ctx.roi if (ctx.roi is not None and roimod.is_area(ctx.roi)) else None, p.shape)
        if img.data.dtype == np.uint8:
            lo, hi = (lo, hi) if fixed else (0.0, 256.0)
        st = ijm.image_statistics(src, rect, rmask, n_bins=int(bins), hist_min=lo if (fixed or img.data.dtype == np.uint8) else 0.0,
                                  hist_max=hi if (fixed or img.data.dtype == np.uint8) else 0.0)
        counts = st.histogram
        edges = st.hist_min + np.arange(bins + 1) * st.bin_size
    else:
        if not fixed:
            lo, hi = float(vals.min()), float(vals.max())
            if hi <= lo:
                hi = lo + 1
        counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    stats = {"count": int(vals.size), "mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if vals.size > 1 else 0,
             "min": float(vals.min()), "max": float(vals.max()), "mode": float(edges[int(np.argmax(counts))]),
             "median": float(np.median(vals))}
    title = f"{'Stack histogram' if all_planes else 'Histogram'} of {img.name}"
    hist = {"counts": counts.tolist(), "edges": edges.tolist(), "stats": stats, "title": title, "bins": int(len(counts)),
            "log_scale": bool(log_scale), "y_max": float(y_max) if y_max > 0 else None, "x_range": [float(edges[0]), float(edges[-1])],
            "fixed_range": bool(fixed), "stack": bool(all_planes)}
    results = None
    if copy_to_results:
        results = [{"Bin Start": float(e), "Count": int(n)} for e, n in zip(edges[:-1], counts)]
    msg = f"{stats['count']} pixels, mean {stats['mean']:.4g}, mode {stats['mode']:g}" + (" (log scale)" if log_scale else "")
    return OpResult(histogram=hist, results=results, message=msg)


@op("analyze.plot_profile", "Plot Profile", CAT, kind="analysis", preview=False, shortcut="K", requires_roi="line_or_rect",
    params=[Param("vertical", "bool", False, help="Rectangle: average columns instead of rows"), Param("all_channels", "bool", True)],
    description="Intensity profile along a line, or column/row averages of a rectangle.")
def plot_profile(ctx, vertical=False, all_channels=True):
    img = ctx.img
    d = img.display
    roi = ctx.roi
    chans = range(img.nc) if (all_channels and img.nc > 1 and d.mode == "composite") else [d.c]
    series = []
    x = None
    for c in chans:
        p = img.data[d.t, d.z, c]
        if roimod.is_line(roi):
            pr = line_profile(p, roi, img.calibration)
        elif roi["type"] == "rect":
            pr = rect_profile(p, roi, img.calibration, vertical)
        else:
            raise ValueError("Plot Profile needs a line or rectangle selection")
        x = pr["x"]
        series.append({"name": img.display.channels[c].name or f"C{c + 1}", "values": pr["values"], "lut": img.display.channels[c].lut})
    plot = {"title": f"Profile of {img.name}", "x": x, "series": series, "x_label": pr["x_label"], "y_label": pr["y_label"]}
    return OpResult(plot=plot)


@op("analyze.find_maxima", "Find Maxima", CAT, kind="analysis", preview=False, supports_rgb=False,
    params=[Param("prominence", "float", 10.0, min=0, step=1, help="Minimum height of a peak above its surroundings (noise tolerance)"),
            Param("output", "choice", "Point Selection", choices=["Point Selection", "Count", "Single Points", "Maxima Within Tolerance", "Segmented Particles", "List"]),
            Param("exclude_edges", "bool", False), Param("light_background", "bool", False), Param("strict", "bool", False, advanced=True),
            Param("above_lower_threshold", "bool", False, advanced=True, help="Only maxima above the image's lower threshold (ImageJ 'Above lower threshold')")],
    description="Local maxima with a prominence (noise tolerance) criterion – ImageJ's Find Maxima (MaximumFinder), including its watershed segmentation.",
    keywords=["peaks", "spots", "seeds", "segment"])
def find_maxima(ctx, prominence=10.0, output="Point Selection", exclude_edges=False, light_background=False, strict=False, above_lower_threshold=False):
    img = ctx.img
    d = img.display
    plane = img.current_plane()
    out_type = {"Single Points": ijm.OUT_SINGLE_POINTS, "Maxima Within Tolerance": ijm.OUT_IN_TOLERANCE, "Segmented Particles": ijm.OUT_SEGMENTED,
                "Point Selection": ijm.OUT_POINT_SELECTION, "List": ijm.OUT_LIST, "Count": ijm.OUT_COUNT}.get(output, ijm.OUT_POINT_SELECTION)
    rect = None
    rmask = None
    mask = ctx.mask
    if mask is not None and mask.any() and out_type != ijm.OUT_SEGMENTED:   # ImageJ drops the selection for Segmented Particles
        ys, xs = np.nonzero(mask)
        rect = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
        rmask = mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
    threshold = None
    if above_lower_threshold:
        thr = current_threshold(ctx)
        if thr is not None:
            threshold = float(thr[0])
    if light_background:
        threshold = None
        plane = ijm.invert_for_minima(plane, rect)
    res = ijm.find_maxima(plane, float(prominence), strict, threshold, out_type, exclude_edges, rect, rmask)
    pts = res.points
    n = len(pts)
    if out_type == ijm.OUT_COUNT:
        return OpResult(results=[{"Label": img.name, "Count": n}], message=f"{n} maxima")
    if out_type == ijm.OUT_LIST:   # ImageJ resets the Results table and lists X, Y in pixel coordinates
        return OpResult(results=[{"X": x, "Y": y} for x, y in pts], results_replace=True, message=f"{n} maxima")
    if out_type == ijm.OUT_POINT_SELECTION:
        if n == 0:
            return OpResult(message="0 maxima")
        return OpResult(roi={"type": "point", "points": [[float(x), float(y)] for x, y in pts], "name": "Maxima"}, message=f"{n} maxima")
    name = img.name + (f"({d.z + 1})" if img.nz > 1 else "") + (" Segmented" if out_type == ijm.OUT_SEGMENTED else " Maxima")
    out_img = binary_image(res.output, name, like=img)
    count = int(ndi.label(res.output > 0, structure=np.ones((3, 3), bool))[1]) if out_type == ijm.OUT_SEGMENTED else n
    return OpResult(new_images=[out_img], message=f"{count} {'particles' if out_type == ijm.OUT_SEGMENTED else 'maxima'}")


_SUMMARY_STATS = (("Mean", np.nanmean), ("SD", lambda v: np.nanstd(v, ddof=1) if len(v) > 1 else 0.0), ("Min", np.nanmin), ("Max", np.nanmax))


@op("analyze.summarize", "Summarize Results", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("group_by", "string", "", help="Optional column (e.g. Label, Slice): one Mean/SD/Min/Max block per distinct value instead of one for the whole table")],
    description="Mean, SD, min and max of every numeric Results column (optionally per group of a label column).")
def summarize(ctx, group_by=""):
    rt = ctx.session.results
    if not rt.rows:
        raise ValueError("Results table is empty")
    group_by = str(group_by or "").strip()
    if group_by and group_by not in rt.columns:
        raise ValueError(f"No column {group_by!r} in the Results table. Columns: {', '.join(rt.columns)}")
    cols = [c for c in rt.columns if c not in ("Label", group_by) and any(isinstance(r.get(c), (int, float)) for r in rt.rows)]

    def block(rows_, extra: dict | None = None) -> list[dict]:
        out = []
        for stat, fn in _SUMMARY_STATS:
            row = {"Label": stat, **(extra or {})}
            for c in cols:
                vals = np.array([r.get(c) for r in rows_ if isinstance(r.get(c), (int, float))], dtype=float)
                row[c] = float(fn(vals)) if vals.size else float("nan")
            out.append(row)
        return out

    if not group_by:
        rows = block(rt.rows)
        header = ["Label"] + cols
    else:
        groups: dict = {}
        for r in rt.rows:  # groups keep the order of first appearance
            groups.setdefault(r.get(group_by), []).append(r)
        gcol = group_by if group_by != "Label" else "Group"  # the first column already holds the statistic's name
        rows = []
        for key, members in groups.items():
            rows += block(members, {gcol: key, "N": len(members)})
        header = ["Label", gcol, "N"] + cols
    return OpResult(table={"name": "Summary", "columns": header, "rows": [[r.get(c) for c in header] for r in rows]},
                    message=f"Summary of {len(rt.rows)} rows" + (f" in {len(rows) // len(_SUMMARY_STATS)} groups of {group_by}" if group_by else ""))


@op("analyze.label_objects", "Measure Labels", CAT, kind="analysis", preview=False, supports_rgb=False,
    params=[Param("intensity_image", "image", None, help="Optional image providing gray values")], description="Measure every object of a label image (as produced by Label Connected Components).")
def measure_label_image(ctx, intensity_image=None):
    img = ctx.img
    d = img.display
    lab = img.current_plane().astype(np.int32)
    s = ctx.session
    ip = s.get_image(intensity_image).data[d.t, d.z, min(d.c, s.get_image(intensity_image).nc - 1)] if intensity_image else None
    rows = measure_labels(img, lab, s.measurements, d.t, d.z, d.c, intensity_plane=ip, options=s.measure_options)
    return OpResult(results=rows, message=f"{len(rows)} objects")


@op("analyze.colocalization", "Colocalization (2 channels)", CAT, kind="analysis", preview=False, supports_rgb=False,
    params=[Param("channel_a", "int", 1, min=1), Param("channel_b", "int", 2, min=1), Param("threshold_a", "float", 0, help="0 = Otsu"), Param("threshold_b", "float", 0, help="0 = Otsu")],
    description="Pearson, Spearman and Manders coefficients between two channels (within the selection).")
def coloc(ctx, channel_a=1, channel_b=2, threshold_a=0.0, threshold_b=0.0):
    img = ctx.img
    d = img.display
    if img.nc < 2:
        raise ValueError("Needs at least two channels")
    a = img.data[d.t, d.z, channel_a - 1].astype(np.float64)
    b = img.data[d.t, d.z, channel_b - 1].astype(np.float64)
    mask = ctx.mask if ctx.mask is not None else np.ones(a.shape, bool)
    av, bv = a[mask], b[mask]
    pearson = float(np.corrcoef(av, bv)[0, 1]) if av.size > 1 else float("nan")
    from scipy.stats import spearmanr
    spear = float(spearmanr(av, bv).statistic) if av.size > 1 else float("nan")
    ta = threshold_a or auto_threshold(img.data[d.t, d.z, channel_a - 1], "Otsu", ctx.mask)[0]
    tb = threshold_b or auto_threshold(img.data[d.t, d.z, channel_b - 1], "Otsu", ctx.mask)[0]
    m1 = float(av[bv >= tb].sum() / av.sum()) if av.sum() else float("nan")
    m2 = float(bv[av >= ta].sum() / bv.sum()) if bv.sum() else float("nan")
    row = {"Label": img.name, "Pearson": pearson, "Spearman": spear, "Manders M1": m1, "Manders M2": m2, "Threshold A": ta, "Threshold B": tb}
    return OpResult(results=[row], message=f"Pearson r = {pearson:.3f}")


def _parse_range(text: str) -> tuple[float, float] | None:
    """'min-max', 'min max' or 'min,max' → (lo, hi); empty → None."""
    import re
    t = str(text or "").strip()
    if not t:
        return None
    parts = [p for p in re.split(r"(?<=[\d.])\s*-\s*|[\s,;]+", t) if p]
    if len(parts) != 2:
        raise ValueError(f"Range must be 'min-max' (got {text!r})")
    lo, hi = float(parts[0]), float(parts[1])
    return (min(lo, hi), max(lo, hi))


@op("analyze.distribution", "Distribution", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("column", "string", "Area", label="Parameter"),
            Param("auto_binning", "bool", False, help="Number of bins from Scott's rule (3.49·SD·n^-1/3), like ImageJ's 'Automatic binning'"),
            Param("bins", "int", 20, min=2, max=500, help="Number of bins when not automatic"),
            Param("range", "string", "", help="Fixed value range 'min-max' (empty = data range)")],
    description="Histogram of a Results-table column (ImageJ Analyze ▸ Distribution: automatic binning or a fixed number of bins and range).")
def distribution(ctx, column="Area", auto_binning=False, bins=20, range=""):
    rt = ctx.session.results
    vals = np.array([r[column] for r in rt.rows if isinstance(r.get(column), (int, float)) and np.isfinite(r[column])], dtype=float)
    if vals.size == 0:
        raise ValueError(f"No numeric values in column {column!r}. Columns: {', '.join(rt.columns)}")
    rng = _parse_range(range)
    lo, hi = rng if rng is not None else (float(vals.min()), float(vals.max()))
    if hi <= lo:
        hi = lo + 1.0
    n_bins = int(bins)
    if auto_binning:
        sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
        width = 3.49 * sd * vals.size ** (-1.0 / 3.0)  # Scott's method (ImageJ Distribution)
        n_bins = int(math.ceil((hi - lo) / width)) if width > 0 else 1
        n_bins = max(1, min(n_bins, 1000))
    counts, edges = np.histogram(vals, bins=n_bins, range=(lo, hi))
    centers = ((edges[:-1] + edges[1:]) / 2).tolist()
    plot = {"title": f"Distribution of {column} (n={vals.size})", "x": centers, "series": [{"name": column, "values": counts.tolist()}], "x_label": column, "y_label": "Count",
            "bins": int(n_bins), "range": [float(lo), float(hi)], "edges": edges.tolist(), "auto_binning": bool(auto_binning)}
    return OpResult(plot=plot, message=f"{column}: mean {vals.mean():.4g}, SD {vals.std(ddof=1) if vals.size > 1 else 0:.4g}, {n_bins} bins")
