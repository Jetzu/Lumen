"""Process ▸ Image Calculator."""
from __future__ import annotations

import numpy as np

from ..registry import Param, OpResult, op
from ..image import LumenImage

OPS = ["Add", "Subtract", "Multiply", "Divide", "AND", "OR", "XOR", "Min", "Max", "Average", "Difference", "Copy", "Transparent-zero"]


@op("process.calculator", "Image Calculator", "Process", kind="image", scope_param=False, preview=False,
    description="Pixel-wise arithmetic between two images: result = image1 ⊕ image2.",
    params=[Param("image1", "image", None, label="Image 1"), Param("operation", "choice", "Subtract", choices=OPS),
            Param("image2", "image", None, label="Image 2"), Param("create_new", "bool", True, label="Create new window"),
            Param("float_result", "bool", False, label="32-bit (float) result")])
def calculator(ctx, image1=None, operation="Subtract", image2=None, create_new=True, float_result=False):
    s = ctx.session
    a = s.get_image(image1) if image1 else ctx.img
    b = s.get_image(image2) if image2 else ctx.img
    da, db = a.data, b.data
    # broadcast single-plane second image over stacks
    if db.shape != da.shape:
        try:
            db = np.broadcast_to(db, da.shape)
        except ValueError:
            raise ValueError(f"Image sizes differ: {da.shape} vs {db.shape}")
    fa, fb = da.astype(np.float32), db.astype(np.float32)
    o = operation
    if o == "Add":
        r = fa + fb
    elif o == "Subtract":
        r = fa - fb
    elif o == "Multiply":
        r = fa * fb
    elif o == "Divide":
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(fb != 0, fa / fb, 0 if not float_result else np.nan)
    elif o in ("AND", "OR", "XOR"):
        if da.dtype == np.float32 or db.dtype == np.float32:
            raise ValueError("Bitwise operations need integer images")
        ia, ib = da.astype(np.int64), db.astype(np.int64)
        r = {"AND": ia & ib, "OR": ia | ib, "XOR": ia ^ ib}[o].astype(np.float32)
    elif o == "Min":
        r = np.minimum(fa, fb)
    elif o == "Max":
        r = np.maximum(fa, fb)
    elif o == "Average":
        r = (fa + fb) / 2
        if not float_result and da.dtype != np.float32:
            r = np.floor(r)  # ImageJ's Byte/ShortBlitter AVERAGE: integer division (a+b)/2
    elif o == "Difference":
        r = np.abs(fa - fb)
    elif o == "Copy":
        r = fb.copy()
    elif o == "Transparent-zero":
        r = np.where(fb == 0, fa, fb)
    else:
        raise ValueError(o)
    from ..registry import cast_to
    out = r.astype(np.float32) if float_result else cast_to(r, da.dtype)
    if create_new:
        name = f"Result of {a.name}"
        img = a.copy(name=name, data=out)
        img.rgb = a.rgb and out.dtype == np.uint8 and out.shape[2] == 3
        img.reset_display(keep_luts=True)
        return OpResult(new_images=[img])
    return OpResult(data=out, rgb=a.rgb and out.dtype == np.uint8)
