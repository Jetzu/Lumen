"""Server-side rendering of a hyperstack plane to RGB (used for RGB conversion,
flatten, PNG export).  The browser does the same thing on the client."""
from __future__ import annotations

import numpy as np

from .image import LumenImage
from .luts import apply_lut


def render_rgb(img: LumenImage, t: int | None = None, z: int | None = None, c: int | None = None,
               mode: str | None = None) -> np.ndarray:
    d = img.display
    t = d.t if t is None else t
    z = d.z if z is None else z
    mode = mode or d.mode
    H, W = img.height, img.width
    if img.rgb:
        out = np.zeros((H, W, 3), np.uint8)
        for ci in range(3):
            ch = d.channels[ci]
            out[..., ci] = apply_lut(img.data[t, z, ci], "Grays", ch.min, ch.max)[..., 0]
        return out
    if mode == "composite" and img.nc > 1:
        acc = np.zeros((H, W, 3), np.float32)
        for ci, ch in enumerate(d.channels):
            if not ch.visible:
                continue
            acc += apply_lut(img.data[t, z, ci], ch.lut, ch.min, ch.max, ch.inverted, ch.gamma).astype(np.float32)
        return np.clip(acc, 0, 255).astype(np.uint8)
    ci = d.c if c is None else c
    ch = d.channels[ci]
    lut = "Grays" if mode == "grayscale" else ch.lut
    return apply_lut(img.data[t, z, ci], lut, ch.min, ch.max, ch.inverted, ch.gamma)
