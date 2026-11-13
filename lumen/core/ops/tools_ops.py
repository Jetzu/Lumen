"""Plugin tools: commands that live on the tool rail (``@op(..., tool=True)``) and run on clicks / drags.

Two built-in tools prove the path (Fiji's Plugins ▸ New ▸ Macro Tool / Plugin Tool):

``tools.spot_counter``      Spot Click Counter – a click marks a spot (overlay marker + Results row), ⌥-click removes the nearest
``tools.measure_distance``  Measure Distance Tool – a drag measures a calibrated distance (Results row + labelled overlay line)

Besides its own ``Param`` list (the tool-options bar) a tool op receives the interaction, see
``lumen.core.registry.TOOL_PARAMS``: ``x, y`` (image px of the press), ``x2, y2`` (release position of a drag, else
None), ``button`` (0 left, 1 middle, 2 right), ``shift / alt / ctrl`` and ``event`` ("click", "drag-start", "drag-end").
Every interaction is recorded as ``lumen.run("tools.…", x=…, y=…, event="click")`` so a macro can replay it, and the
same call works from the Python console, an ImageJ macro (``run("Spot Click Counter", "x=10 y=12")``) or JavaScript.

ImageJ macro-language tools (``macro "Name Tool - C00cO11cc" { getCursorLoc(x, y, z, flags); … }`` in an installed
``.ijm`` file) are registered the same way by ``lumen.core.ops.macro_tools`` (ids ``macrotool.<slug>``).
"""
from __future__ import annotations

import math
from typing import Any

from ..registry import OpResult, Param, op

TOOLS = "Plugins/Tools"
_TOOL = dict(kind="analysis", preview=False, roi_aware=False, scope_param=False, tool=True)


def _value_at(img: Any, x: float, y: float) -> Any:
    """Pixel value under (x, y) on the displayed plane: a float, ``[r, g, b]`` for RGB, None outside the image."""
    px, py = int(math.floor(x)), int(math.floor(y))
    if not (0 <= px < img.width and 0 <= py < img.height):
        return None
    d = img.display
    if img.rgb:
        return [float(v) for v in img.data[d.t, d.z, :, py, px]]
    return float(img.data[d.t, d.z, min(d.c, img.nc - 1), py, px])


def _position(img: Any) -> dict[str, int]:
    d = img.display
    return {"t": int(d.t), "z": int(d.z), "c": int(d.c)}


def _same_plane(item: dict[str, Any], img: Any) -> bool:
    pos = item.get("position")
    if not isinstance(pos, dict):
        return True
    d = img.display
    return (pos.get("t") in (None, d.t)) and (pos.get("z") in (None, d.z))


# ============================================================================ Spot Click Counter
@op("tools.spot_counter", "Spot Click Counter", TOOLS, **_TOOL, tool_icon="◎", tool_cursor="crosshair",
    keywords=["tool", "count", "click", "spots", "cell counter", "point", "marker"],
    params=[Param("marker_size", "int", 6, min=1, max=60, unit="px", help="Marker radius drawn in the overlay (screen px)"),
            Param("color", "color", "#ff5252"),
            Param("all_planes", "bool", False, label="Show spots on every plane", help="Otherwise a spot is pinned to the plane it was clicked on")],
    description="A plugin tool (tool rail): click on the image to mark a spot – an overlay marker plus a Results row with its position, "
                "value and the running count; a drag marks the release point; ⌥-click removes the nearest spot; ⇧-click only reports "
                "the value. Shows how a tool=True command receives clicks (x, y, x2, y2, button, shift, alt, ctrl, event).")
def spot_counter(ctx, marker_size=6, color="#ff5252", all_planes=False, x=0.0, y=0.0, x2=None, y2=None, button=0,
                 shift=False, alt=False, ctrl=False, event="click"):
    img = ctx.img
    if event == "drag-start":
        return OpResult()
    if event == "drag-end" and x2 is not None and y2 is not None:
        x, y = float(x2), float(y2)
    x, y = float(x), float(y)
    spots = [o for o in img.overlay if str(o.get("id", "")).startswith("spot-")]
    others = [o for o in img.overlay if not str(o.get("id", "")).startswith("spot-")]
    value = _value_at(img, x, y)
    if shift:
        return OpResult(message=f"({x:.0f}, {y:.0f}) = {value} – {len(spots)} spot(s)")
    if alt:
        here = [o for o in spots if _same_plane(o, img)]
        if not here:
            return OpResult(message="No spot to remove on this plane")
        nearest = min(here, key=lambda o: math.hypot(o["points"][0][0] - x, o["points"][0][1] - y))
        dist = math.hypot(nearest["points"][0][0] - x, nearest["points"][0][1] - y)
        if dist > max(12.0, 3.0 * float(marker_size)):
            return OpResult(message=f"No spot near ({x:.0f}, {y:.0f}) – the nearest is {dist:.0f} px away")
        spots.remove(nearest)
        return OpResult(overlay=others + spots, message=f"Removed {nearest.get('name', nearest['id'])} – {len(spots)} spot(s) left")
    n = max([int(str(o["id"]).split("-")[-1]) for o in spots if str(o["id"]).split("-")[-1].isdigit()] + [0]) + 1
    pos = _position(img)
    spot: dict[str, Any] = {"type": "point", "id": f"spot-{n}", "name": f"Spot {n}", "points": [[round(x, 2), round(y, 2)]], "color": color,
                            "stroke_width": 1.5, "label": str(n), "point_size": int(marker_size)}
    if not all_planes:
        spot["position"] = pos
    row = {"Label": f"Spot {n}", "X": round(x, 2), "Y": round(y, 2), "Value": ",".join(f"{v:g}" for v in value) if isinstance(value, list) else value,
           "T": pos["t"], "Z": pos["z"], "C": pos["c"], "Count": len(spots) + 1}
    return OpResult(overlay=others + spots + [spot], results=[row], message=f"{len(spots) + 1} spot(s) – ⌥-click removes the nearest")


# ============================================================================ Measure Distance Tool
@op("tools.measure_distance", "Measure Distance Tool", TOOLS, **_TOOL, tool_icon="↔", tool_cursor="crosshair",
    keywords=["tool", "distance", "measure", "length", "ruler", "line", "calibrated"],
    params=[Param("add_to_overlay", "bool", True, label="Keep a labelled line in the overlay"),
            Param("color", "color", "#ffd54a"),
            Param("line_width", "float", 1.5, min=0.5, max=20, step=0.5, unit="px"),
            Param("decimals", "int", 2, min=0, max=6)],
    description="A plugin tool (tool rail): drag on the image to measure the distance between two points in calibrated units – a Results "
                "row with the end points, length (unit) and angle, the line as the selection and optionally a labelled overlay line. "
                "A plain click reports the pixel position and value.")
def measure_distance(ctx, add_to_overlay=True, color="#ffd54a", line_width=1.5, decimals=2, x=0.0, y=0.0, x2=None, y2=None, button=0,
                     shift=False, alt=False, ctrl=False, event="click"):
    img = ctx.img
    cal = img.calibration
    if event == "drag-start":
        return OpResult()
    x, y = float(x), float(y)
    if event == "click" or x2 is None or y2 is None:
        return OpResult(message=f"({x:.0f}, {y:.0f}) = {_value_at(img, x, y)} – drag to measure a distance")
    x2, y2 = float(x2), float(y2)
    pw = float(cal.pixel_width or 1.0)
    ph = float(cal.pixel_height or 1.0)
    dx = (x2 - x) * pw
    dy = (y2 - y) * ph
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(-dy, dx))  # ImageJ convention: counter-clockwise from the x axis, y pointing up
    unit = cal.unit or "pixel"
    dec = int(decimals)
    n = sum(1 for r in ctx.session.results.rows if str(r.get("Label", "")).startswith("Distance ")) + 1
    row = {"Label": f"Distance {n}", "X1": round(x, 2), "Y1": round(y, 2), "X2": round(x2, 2), "Y2": round(y2, 2),
           "Length": round(length, dec), "Angle": round(angle, 2), "Unit": unit}
    label = f"{length:.{dec}f} {unit}"
    res = OpResult(results=[row], roi={"type": "line", "x1": x, "y1": y, "x2": x2, "y2": y2},
                   message=f"Length {label}, angle {angle:.1f}°")
    if add_to_overlay:
        item = {"type": "line", "id": f"distance-{n}", "name": f"Distance {n}", "x1": round(x, 2), "y1": round(y, 2), "x2": round(x2, 2), "y2": round(y2, 2),
                "color": color, "stroke_width": float(line_width), "label": label, "position": _position(img)}
        res.overlay = list(img.overlay) + [item]
    return res
