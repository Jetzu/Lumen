from __future__ import annotations

import numpy as np

from ..image import LumenImage
from ..registry import cast_to


def as_float(plane: np.ndarray) -> np.ndarray:
    return plane.astype(np.float32, copy=False)


def back_to(plane: np.ndarray, like: np.ndarray) -> np.ndarray:
    return cast_to(plane, like.dtype)


def footprint(radius: float, shape: str = "disk") -> np.ndarray:
    from skimage.morphology import disk, square
    r = max(int(round(radius)), 1)
    if shape == "square":
        return square(2 * r + 1)
    return disk(r)


def label_image(labels: np.ndarray, name: str, like: LumenImage | None = None) -> LumenImage:
    """Make a label image (Glasbey on dark LUT, 0 = background) in the narrowest dtype that holds every label: uint8 up to 255
    labels, uint16 up to 65535, else float32 (integers are exact up to 2**24; every label consumer reads labels
    dtype-independently) – a label count above 65535 never wraps."""
    labels = np.asarray(labels)
    n = int(labels.max()) if labels.size else 0
    if n > 2 ** 24:
        raise ValueError(f"Too many labels ({n}) for a 32-bit label image")
    dtype = np.uint8 if n < 256 else np.uint16 if n < 65536 else np.float32
    img = LumenImage(labels.astype(dtype), name=name)
    if like is not None:
        img.calibration = type(like.calibration)(**like.calibration.to_dict())
    ch = img.display.channels[0]
    ch.lut = "Glasbey on dark"  # Fiji-exact Glasbey has a white index 0; the dark variant keeps the background black
    ch.min, ch.max = 0.0, float(max(n, 1))
    return img


def binary_image(mask: np.ndarray, name: str, like: LumenImage | None = None) -> LumenImage:
    img = LumenImage((mask > 0).astype(np.uint8) * 255, name=name)
    if like is not None:
        img.calibration = type(like.calibration)(**like.calibration.to_dict())
    img.lut_is_binary = True
    return img


def is_binary_plane(plane: np.ndarray) -> bool:
    if plane.dtype != np.uint8:
        return False
    return bool(np.isin(np.unique(plane), (0, 255)).all())


def current_threshold(ctx) -> tuple[float, float] | None:
    img = ctx.image
    if img is None:
        return None
    thr = ctx.session.thresholds.get(img.id)
    if thr is None:
        return None
    return float(thr["lower"]), float(thr["upper"])
