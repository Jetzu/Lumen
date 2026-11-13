"""Image ▸ Adjust ▸ Threshold and friends."""
from __future__ import annotations

import numpy as np
from skimage import filters as skf

from ..registry import Param, OpResult, op
from ..threshold import METHODS, auto_threshold, histogram256, legacy_isodata_threshold
from ._util import current_threshold

CAT = "Image/Adjust"
LEGACY = "ImageJ legacy (ISODATA)"  # the Thresholder's auto threshold when no threshold is set
#: the channel a threshold / measurement acts on; 0 = the current one. The recorder writes the explicit 1-based channel on
#: multi-channel images so the replay does not depend on the channel slider (E7-11)
CHANNEL_PARAM = Param("channel", "int", 0, min=0, advanced=True, help="1-based channel to work on; 0 = the current channel")


def channel_index(img, channel) -> int:
    """0-based channel index for a ``channel`` parameter (1-based, 0 = current); validates against the image."""
    c = int(channel or 0)
    if c == 0:
        return img.display.c
    if c > img.nc:
        raise ValueError(f"{img.name} has {img.nc} channel(s), channel {c} does not exist")
    return c - 1


@op("threshold.compute", "Compute Auto Threshold", CAT, kind="analysis", roi_aware=False, supports_rgb=False,
    description="Compute an automatic threshold for the current plane and show it as an overlay (nothing is applied).",
    params=[Param("method", "choice", "Default", choices=METHODS), Param("dark_background", "bool", True, help="Objects are brighter than the background"), CHANNEL_PARAM], keywords=["otsu", "segment"])
def compute(ctx, method="Default", dark_background=True, channel=0):
    img = ctx.img
    d = img.display
    plane = img.data[d.t, d.z, channel_index(img, channel)]
    lo, hi = auto_threshold(plane, method, ctx.mask, dark_background)
    thr = {"lower": lo, "upper": hi, "method": method, "dark_background": dark_background}
    return OpResult(threshold=thr, message=f"{method}: {lo:.4g} – {hi:.4g}")


@op("threshold.set", "Set Threshold", CAT, kind="analysis", roi_aware=False, supports_rgb=False,
    description="Set the threshold range manually (shown as an overlay).",
    params=[Param("lower", "float", 128.0), Param("upper", "float", 255.0)], shortcut="Shift+T")
def set_threshold(ctx, lower=128.0, upper=255.0):
    thr = {"lower": float(lower), "upper": float(upper), "method": "Manual"}
    return OpResult(threshold=thr)


def _level(plane, method, mask, dark_background, add_one):
    if method == LEGACY:
        return legacy_isodata_threshold(plane, mask)
    return auto_threshold(plane, method, mask, dark_background, add_one=add_one)


@op("threshold.apply", "Apply Threshold → Mask", CAT, kind="image", preview=True, supports_rgb=False, scope_param=False,
    description="Convert the image to a binary mask (255 inside the threshold range, 0 outside). Uses the current threshold, or computes one with the chosen method.",
    params=[Param("method", "choice", "Current", choices=["Current"] + METHODS + [LEGACY],
                  help="Current: the threshold set in the panel (without one ImageJ's Convert to Mask falls back to the legacy ISODATA auto threshold); ImageJ legacy (ISODATA): that fallback explicitly"),
            Param("dark_background", "bool", True),
            Param("per_plane", "bool", False, label="Calculate threshold for each plane"), Param("only_current_plane", "bool", False),
            Param("add_one", "bool", True, advanced=True, label="Foreground from level + 1",
                  help="ImageJ's setAutoThreshold() and the Auto Threshold plugin select the pixels above the computed level; Process ▸ Binary ▸ Convert to Mask on a stack ('calculate threshold for each image') selects from the level itself"),
            CHANNEL_PARAM],
    keywords=["binary", "make binary", "convert to mask"])
def apply_threshold(ctx, method="Current", dark_background=True, per_plane=False, only_current_plane=False, add_one=True, channel=0):
    img = ctx.img
    d = img.display
    cc = channel_index(img, channel)  # the channel the threshold belongs to (the current one unless given)
    thr = current_threshold(ctx)
    if method == "Current" and (thr is None or per_plane):
        # no stored range: ImageJ's Thresholder runs its legacy ISODATA auto threshold (the mode-corrected
        # iterative intermeans with the object side chosen from the histogram, no +1). "Calculate threshold for
        # each plane" re-computes per plane with the method the threshold panel used (ImageJ's Convert Stack to
        # Binary dialog is pre-filled with it, and converts without the +1); a manual range carries no method.
        rec = ctx.session.thresholds.get(img.id) or {}
        stored = rec.get("method")
        if per_plane:
            method = stored if stored in METHODS else LEGACY
            add_one = False
        else:
            method = LEGACY
        if "dark_background" in rec:
            dark_background = bool(rec["dark_background"])
    if method != "Current" and not per_plane:
        lo, hi = _level(img.data[d.t, d.z, cc], method, ctx.mask, dark_background, add_one)
        thr = (lo, hi)
    out = np.zeros(img.shape, np.uint8)
    mask = ctx.mask
    planes = [(d.t, d.z, cc)] if only_current_plane else list(img.iter_planes())
    if only_current_plane:
        if img.data.dtype == np.uint8:
            out = img.data.copy()
        else:
            # keep the other planes: scale them to 8-bit through their display range (like Image ▸ Type ▸ 8-bit)
            from ..registry import cast_to
            scaled = np.zeros(img.shape, np.float32)
            for c, ch in enumerate(img.display.channels):
                lo, hi = ch.min, ch.max
                scaled[:, :, c] = (img.data[:, :, c].astype(np.float32) - lo) / max(hi - lo, 1e-12) * 255.0
            out = cast_to(scaled, np.uint8)
    for (t, z, c) in planes:
        p = img.data[t, z, c]
        if per_plane and method != "Current":
            lo, hi = _level(p, method, mask, dark_background, add_one)
        else:
            lo, hi = thr
        b = (p >= lo) & (p <= hi)
        if mask is not None:
            b &= mask
        out[t, z, c] = b.astype(np.uint8) * 255
    # a preview must leave the session and the image untouched (Session._preview restores only the pixel data):
    # dropping the threshold here would make the final Apply with method "Current" fall back to "Default"
    if not ctx.preview:
        ctx.session.thresholds.pop(img.id, None)
    if img.nc > 1 and not img.rgb and not only_current_plane:
        # multi-channel composite: the threshold belongs to the current channel → new single-channel mask image
        from ._util import binary_image
        # keep the channel axis, (T, Z, 1, Y, X): a 4-D slab is read as ZCYX by to_canonical, so for Z > 1 the mask was
        # born with Z channels and a Cyan LUT that the later reshape + sync_channels(keep_luts) preserved
        mask_img = binary_image(out[:, :, cc:cc + 1], f"Mask of {img.name}", like=img)
        return OpResult(new_images=[mask_img], message=f"Mask of channel {cc + 1} created as a new image")
    if not ctx.preview:  # only an applied threshold turns the image into a binary mask
        img.lut_is_binary = True
    return OpResult(data=out, rgb=False, display={"channels": [{"min": 0, "max": 255} for _ in range(img.nc)]})


@op("threshold.local", "Auto Local Threshold", CAT, kind="plane", supports_rgb=False,
    description="Adaptive threshold with a local neighbourhood (Bernsen, Contrast, Mean, Median, MidGrey, Gaussian, Niblack, Otsu, Phansalkar, Sauvola).",
    params=[Param("method", "choice", "Sauvola", choices=["Bernsen", "Contrast", "Mean", "Median", "MidGrey", "Gaussian", "Niblack", "Otsu", "Sauvola", "Phansalkar"]),
            Param("radius", "int", 15, min=1, max=500), Param("parameter_1", "float", 0.0, help="Niblack/Sauvola k offset, Bernsen contrast threshold, MidGrey / Mean / Median / Gaussian offset C"),
            Param("white_objects", "bool", True)])
def local_threshold(plane, ctx, method="Sauvola", radius=15, parameter_1=0.0, white_objects=True):
    f = plane.astype(np.float32)
    block = 2 * radius + 1
    if method in ("Contrast", "MidGrey"):
        from scipy import ndimage as ndi
        from skimage.morphology import disk
        fp = disk(radius)
        mx = ndi.maximum_filter(f, footprint=fp, mode="nearest")
        mn = ndi.minimum_filter(f, footprint=fp, mode="nearest")
        if method == "Contrast":
            # Fiji Auto Local Threshold "Contrast": object if the pixel is closer to the local maximum than to the local minimum
            b = (mx - f) <= (f - mn)
        else:
            b = f > (mx + mn) / 2.0 - parameter_1
    elif method == "Otsu":
        from skimage.filters import rank
        from skimage.morphology import disk
        if plane.dtype == np.uint8:
            p8 = plane
        else:
            lo, hi = float(f.min()), float(f.max())
            p8 = np.clip((f - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).astype(np.uint8)
        t = rank.otsu(p8, disk(radius))
        b = p8 > t
    elif method == "Bernsen":
        from scipy import ndimage as ndi
        from skimage.morphology import disk
        fp = disk(radius)
        mx = ndi.maximum_filter(f, footprint=fp, mode="nearest")
        mn = ndi.minimum_filter(f, footprint=fp, mode="nearest")
        contrast_thr = parameter_1 if parameter_1 > 0 else (15.0 if plane.dtype == np.uint8 else (f.max() - f.min()) * 0.06)
        # Fiji Auto_Local_Threshold.Bernsen: mid_gray = (min + max) / 2 in integer arithmetic for 8-bit, pixel >= mid_gray
        # → object, and a low-contrast window → object iff mid_gray >= 128 (the median for other bit depths). Black objects
        # only swap the output values, i.e. the mask is the bitwise inverse of the white-objects mask, so Bernsen is
        # computed in the white-objects sense here and goes through the common ~b below like every other method.
        mid = np.floor((mx + mn) / 2) if plane.dtype == np.uint8 else (mx + mn) / 2
        low_contrast = (mx - mn) < contrast_thr
        flat_is_object = mid >= 128 if plane.dtype == np.uint8 else mid >= np.median(f)
        b = np.where(low_contrast, flat_is_object, f >= mid)
    elif method in ("Mean", "Median", "Gaussian"):
        t = skf.threshold_local(f, block, method=method.lower(), offset=parameter_1)
        b = f > t
    elif method == "Niblack":
        k = parameter_1 if parameter_1 != 0 else 0.2
        t = skf.threshold_niblack(f, window_size=block, k=k)
        b = f > t
    elif method == "Sauvola":
        k = parameter_1 if parameter_1 != 0 else 0.2
        r = 128 if plane.dtype == np.uint8 else (f.max() - f.min()) / 2
        t = skf.threshold_sauvola(f, window_size=block, k=k, r=r)
        b = f > t
    elif method == "Phansalkar":
        k = parameter_1 if parameter_1 != 0 else 0.25
        n = (f - f.min()) / max(float(f.max() - f.min()), 1e-9)
        from scipy import ndimage as ndi
        m = ndi.uniform_filter(n, block, mode="nearest")
        s = np.sqrt(np.maximum(ndi.uniform_filter(n * n, block, mode="nearest") - m * m, 0))
        t = m * (1 + 2.0 * np.exp(-10.0 * m) + k * (s / 0.5 - 1))
        b = n > t
    else:
        raise ValueError(method)
    if not white_objects:
        b = ~b
    return b.astype(np.uint8) * 255


@op("threshold.histogram", "Threshold Histogram", CAT, kind="analysis", roi_aware=False, supports_rgb=False, preview=False,
    description="256-bin histogram of the current plane (used by the threshold panel).")
def thr_hist(ctx):
    plane = ctx.img.current_plane()
    h, lo, hi = histogram256(plane, ctx.mask)
    return OpResult(histogram={"counts": h.tolist(), "min": lo, "max": hi})
