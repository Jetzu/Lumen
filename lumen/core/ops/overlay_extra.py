"""Image ▸ Overlay extras: ``overlay.measure`` (ImageJ's Measure Overlay).

Every ROI-shaped overlay item (areas, lines, multi-points) is measured with the Set Measurements configuration
(columns, limit to threshold, redirect image, decimals / invert-Y options), each on its own plane when it carries a
``position`` and labelled ``<image>:<name or index>`` like ImageJ.
"""
from __future__ import annotations

from typing import Any

from .. import roi as roimod
from ..measure import measure_roi
from ..registry import OpResult, Param, op
from ._util import current_threshold
from .analyze import _redirect_position, redirect_image
from .extras import _overlay_shaped, _roi_plane

OVL = "Image/Overlay"


def overlay_item_label(img_name: str, item: dict[str, Any], index: int) -> str:
    """ImageJ's Measure Overlay row label: ``image:name`` (the index when the item has no name)."""
    return f"{img_name}:{item.get('name') or index}"


@op("overlay.measure", "Measure Overlay", OVL, kind="analysis", preview=False, roi_aware=False, keywords=["measure overlay", "overlay", "measure all", "roi"],
    params=[Param("append", "bool", True, help="Append to the Results table (off = replace it)"),
            Param("current_plane_only", "bool", False, label="Current slice / frame only", help="Skip items pinned to other slices or frames")],
    description="Measure every ROI-shaped overlay item (areas, lines, points) with the Set Measurements configuration – each on its own plane when it is pinned to one – "
                "and add one row per item labelled image:name to the Results table (ImageJ's Image ▸ Overlay ▸ Measure Overlay).")
def measure_overlay(ctx, append=True, current_plane_only=False):
    s = ctx.session
    img = ctx.img
    d = img.display
    items = [(i, o) for i, o in enumerate(img.overlay, 1) if o in _overlay_shaped(img)]
    if not items:
        raise ValueError("The overlay has no ROI-shaped items – use Overlay ▸ Add Selection / From ROI Manager first")
    thr = current_threshold(ctx) if s.measure_options.get("limit_to_threshold") else None
    target = redirect_image(s, img)
    rows: list[dict[str, Any]] = []
    measured = 0
    n = len(items)
    for k, (i, o) in enumerate(items):
        ctx.check_cancel()
        t, z, c = _roi_plane(o, img)
        pos = o.get("position") if isinstance(o.get("position"), dict) else None
        if current_plane_only and pos and (t != d.t or z != d.z):
            continue
        roi = roimod.normalize({k2: v for k2, v in o.items() if k2 != "position"})
        label = overlay_item_label(img.name, o, i)
        if target is not None:  # ImageJ: the selection comes from this image, the pixels from the redirect image
            tt, zz, cc = _redirect_position(target, t, z, c)
            rows.extend(measure_roi(target, roi, s.measurements, t=tt, z=zz, c=cc, limit_to_threshold=thr, label=label, options=s.measure_options))
        else:
            rows.extend(measure_roi(img, roi, s.measurements, t=t, z=z, c=c, limit_to_threshold=thr, label=label, options=s.measure_options))
        measured += 1
        if n > 10:
            ctx.progress((k + 1) / n, f"Item {k + 1}/{n}")
    if not rows:
        raise ValueError("No overlay item lies on the current slice / frame")
    return OpResult(results=rows, results_replace=not append, message=f"{len(rows)} row(s) from {measured} overlay item(s)")
