"""Single-value colour settings used by the value-picker tool.

``edit.set_colors`` (edit.py) is the two-field Colour Settings dialog and always writes both
values; the picker only wants to change one of them, so it needs ops that leave the other alone.
"""
from __future__ import annotations

from ..registry import OpResult, Param, op

CAT = "Edit"


@op("edit.set_foreground", "Set Foreground Value", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("value", "float", 255)], keywords=["picker", "colour", "color"],
    description="Set only the foreground value (used by the value-picker tool); the background value is left unchanged.")
def set_foreground(ctx, value=255.0):
    ctx.session.settings["foreground"] = float(value)
    return OpResult(message=f"Foreground {value:g}")


@op("edit.set_background", "Set Background Value", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("value", "float", 0)], keywords=["picker", "colour", "color"],
    description="Set only the background value (used by the value-picker tool with Alt); the foreground value is left unchanged.")
def set_background(ctx, value=0.0):
    ctx.session.settings["background"] = float(value)
    return OpResult(message=f"Background {value:g}")
