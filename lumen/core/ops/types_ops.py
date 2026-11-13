"""Image ▸ Type conversions."""
from __future__ import annotations

import numpy as np

from ..image import ChannelDisplay
from ..registry import Param, OpResult, op, cast_to
from ..render import render_rgb
from . import _imagej_types as ijt

CAT = "Image/Type"

_WEIGHTED_HELP = "RGB → gray with the weights 0.299 / 0.587 / 0.114 (ImageJ's 'Weighted RGB conversions'); unchecked: the unweighted mean (r+g+b)/3 like ImageJ's default"


def _gray_display(img) -> None:
    img.display.channels = [ChannelDisplay(name="Gray", lut="Grays", min=0, max=255)]
    img.display.c = 0


@op("type.8bit", "8-bit", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("scale", "bool", True, help="Scale the display range to 0–255 (ImageJ's 'Scale when converting'); unchecked: 16-bit values are clamped at 255, 32-bit values rounded and clamped"),
            Param("weighted", "bool", False, help=_WEIGHTED_HELP)],
    description="Convert to 8-bit (0–255) with ImageJ's TypeConverter arithmetic: 16-bit (v−min)·256/(max−min+1)+0.5, 32-bit (v−min)·255/(max−min)+0.5 from each "
                "channel's display range; RGB becomes the (un)weighted gray mean. Multi-channel images keep their channels.")
def to_8bit(ctx, scale=True, weighted=False):
    img = ctx.img
    weights = ijt.WEIGHTED if weighted else ijt.UNWEIGHTED
    if img.rgb:
        out = np.zeros((img.nt, img.nz, 1, img.height, img.width), np.uint8)
        for t in range(img.nt):
            for z in range(img.nz):
                out[t, z, 0] = ijt.rgb_to_byte(img.data[t, z], weights)
        _gray_display(img)
        return OpResult(data=out, rgb=False, display={"mode": "single"})
    if img.data.dtype == np.uint8:
        return OpResult(message="Already 8-bit")
    out = np.zeros(img.shape, np.uint8)
    for (t, z, c) in img.iter_planes():
        ctx.check_cancel()
        ch = img.display.channels[c]
        plane = img.data[t, z, c]
        out[t, z, c] = ijt.short_to_byte(plane, ch.min, ch.max, scale) if plane.dtype == np.uint16 else ijt.float_to_byte(plane, ch.min, ch.max, scale)
    return OpResult(data=out, display={"channels": [{"min": 0, "max": 255} for _ in img.display.channels]})


@op("type.16bit", "16-bit", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("scale", "bool", True, help="For 32-bit sources: scale the display range to 0–65535 (ImageJ's 'Scale when converting'); unchecked: round and clamp the values")],
    description="Convert to 16-bit unsigned with ImageJ's TypeConverter arithmetic (32-bit: (v−min)·65535/(max−min)+0.5 from the display range; RGB: the unweighted gray mean).")
def to_16bit(ctx, scale=True):
    img = ctx.img
    if img.rgb:
        out = np.zeros((img.nt, img.nz, 1, img.height, img.width), np.uint16)
        for t in range(img.nt):
            for z in range(img.nz):
                out[t, z, 0] = ijt.rgb_to_byte(img.data[t, z]).astype(np.uint16)
        _gray_display(img)
        return OpResult(data=out, rgb=False, display={"mode": "single"})
    if img.data.dtype == np.uint16:
        return OpResult(message="Already 16-bit")
    if img.data.dtype == np.uint8:
        data = img.data.astype(np.uint16)
        return OpResult(data=data, display={"channels": [{"min": c.min, "max": c.max} for c in img.display.channels]})
    out = np.zeros(img.shape, np.uint16)
    for (t, z, c) in img.iter_planes():
        ctx.check_cancel()
        ch = img.display.channels[c]
        out[t, z, c] = ijt.float_to_short(img.data[t, z, c], ch.min, ch.max, scale)
    if scale:
        return OpResult(data=out, display={"channels": [{"min": 0, "max": 65535} for _ in img.display.channels]})
    return OpResult(data=out, display={"channels": [{"min": max(0.0, c.min), "max": min(65535.0, max(c.max, 1.0))} for c in img.display.channels]})


@op("type.32bit", "32-bit", CAT, kind="image", scope_param=False, preview=False, params=[Param("weighted", "bool", False, help=_WEIGHTED_HELP)],
    description="Convert to 32-bit floating point (values unchanged; RGB becomes the (un)weighted gray mean like ImageJ).")
def to_32bit(ctx, weighted=False):
    img = ctx.img
    if img.rgb:
        weights = ijt.WEIGHTED if weighted else ijt.UNWEIGHTED
        out = np.zeros((img.nt, img.nz, 1, img.height, img.width), np.float32)
        for t in range(img.nt):
            for z in range(img.nz):
                out[t, z, 0] = ijt.rgb_to_float(img.data[t, z], weights)
        _gray_display(img)
        return OpResult(data=out, rgb=False, display={"mode": "single"})
    if img.data.dtype == np.float32:
        return OpResult(message="Already 32-bit")
    return OpResult(data=img.data.astype(np.float32), display={"channels": [{"min": c.min, "max": c.max} for c in img.display.channels]})


def flatten_rgb(img, t: int, z: int) -> np.ndarray:
    """(H, W, 3) uint8 of plane (t, z) as ImageJ's RGB Color / Stack to RGB bake it (display index per pixel type, LUT, composite
    sum with clamping); channels with a Lumen gamma ≠ 1 fall back to Lumen's renderer."""
    if any(ch.gamma != 1.0 for ch in img.display.channels[:img.nc]):
        return render_rgb(img, t=t, z=z)
    return ijt.render_rgb(img, t=t, z=z)


@op("type.rgb", "RGB Color", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("create_new", "bool", False, label="Create new window", help="Put the RGB image in a new tab '<name> (RGB)' (ImageJ's Stack to RGB); unchecked: convert in place")],
    description="Flatten the current display (LUTs, ranges, composite) into an RGB image for every slice/frame with ImageJ's pixel arithmetic "
                "(8-bit LUT index (v−min)·256/(max−min), 16-bit (v−min)·256/(max−min+1)+0.5, 32-bit (v−min)·255/(max−min)+0.5; composite channels summed and clamped).",
    keywords=["flatten", "composite to rgb", "stack to rgb"])
def to_rgb(ctx, create_new=False):
    img = ctx.img
    if img.rgb:
        return OpResult(message="Already RGB")
    out = np.zeros((img.nt, img.nz, 3, img.height, img.width), np.uint8)
    for t in range(img.nt):
        for z in range(img.nz):
            ctx.check_cancel()
            out[t, z] = flatten_rgb(img, t, z).transpose(2, 0, 1)
    rgb_channels = [ChannelDisplay(name=n, lut=n, min=0, max=255) for n in ("Red", "Green", "Blue")]
    if create_new:
        new = img.copy(name=f"{img.name} (RGB)", data=out)
        new.rgb = True
        new.display.channels = rgb_channels
        new.display.c = 0
        new.display.mode = "composite"
        return OpResult(new_images=[new])
    img.display.channels = rgb_channels
    img.display.c = 0
    img.display.mode = "composite"
    return OpResult(data=out, rgb=True)


@op("type.slices_to_rgb", "Stack to RGB (slices as R, G, B)", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("create_new", "bool", False, label="Create new window")],
    description="Merge the 2 or 3 slices of a single-channel stack into one RGB image like ImageJ's Stack to RGB: 8-bit slices are taken as they are, 16/32-bit slices are "
                "scaled from the display range (TypeConverter) first; a missing third slice gives blue = 0.", keywords=["stack to rgb", "rgb stack"])
def slices_to_rgb(ctx, create_new=False):
    img = ctx.img
    if img.rgb or img.nc != 1 or img.nt != 1 or img.nz not in (2, 3):
        raise ValueError("Stack to RGB needs a single-channel stack of 2 or 3 slices (composites use RGB Color)")
    ch = img.display.channels[0]
    out = np.zeros((1, 1, 3, img.height, img.width), np.uint8)
    for z in range(img.nz):
        plane = img.data[0, z, 0]
        if plane.dtype == np.uint8:
            out[0, 0, z] = plane
        elif plane.dtype == np.uint16:
            out[0, 0, z] = ijt.short_to_byte(plane, ch.min, ch.max, True)
        else:
            out[0, 0, z] = ijt.float_to_byte(plane, ch.min, ch.max, True)
    rgb_channels = [ChannelDisplay(name=n, lut=n, min=0, max=255) for n in ("Red", "Green", "Blue")]
    if create_new:
        new = img.copy(name=f"{img.name} (RGB)", data=out)
        new.rgb = True
        new.display.channels = rgb_channels
        new.display.c = 0
        new.display.z = 0
        new.display.mode = "composite"
        return OpResult(new_images=[new])
    img.display.channels = rgb_channels
    img.display.c = 0
    img.display.z = 0
    img.display.mode = "composite"
    return OpResult(data=out, rgb=True)


@op("type.apply_lut", "Apply LUT", "Image/Lookup Tables", kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("all_channels", "bool", False, help="Apply every channel's display range (default: the current channel only, like ImageJ)")],
    description="Bake the display range into the pixel values like ImageJ's Image ▸ Lookup Tables ▸ Apply LUT (ContrastAdjuster.apply): 8-bit pixels map through "
                "(v−min)·256/(max−min) truncated, 16-bit through ·65536, values at or below min → 0, at or above max → the maximum; RGB images are rescaled per channel. "
                "Not available for 32-bit images.", keywords=["apply lut", "brightness", "contrast", "display range"])
def apply_lut(ctx, all_channels=False):
    img = ctx.img
    if img.data.dtype == np.float32:
        raise ValueError('"Apply LUT" does not work with 32-bit images – convert to 16-bit first')
    full = 255.0 if img.data.dtype == np.uint8 else 65535.0
    out = img.data.copy()
    chans = range(img.nc) if (all_channels or img.rgb) else [img.display.c]
    patch = [{} for _ in range(img.nc)]
    for c in chans:
        ch = img.display.channels[c]
        if img.rgb:
            out[:, :, c] = ijt.rgb_display_index(img.data[:, :, c], ch.min, ch.max)
        else:
            out[:, :, c] = ijt.apply_lut_table(img.data.dtype, ch.min, ch.max)[img.data[:, :, c]].astype(img.data.dtype)
        patch[c] = {"min": 0.0, "max": full}
    return OpResult(data=out, rgb=img.rgb, display={"channels": patch}, message="Display range applied to pixel values")


@op("type.rgb_stack", "RGB Stack", CAT, kind="image", scope_param=False, preview=False, description="Turn an RGB image into a 3-channel 8-bit stack (R, G, B).")
def rgb_stack(ctx):
    img = ctx.img
    if not img.rgb:
        raise ValueError("Not an RGB image")
    img.display.channels = [ChannelDisplay(name=n, lut=n, min=0, max=255) for n in ("Red", "Green", "Blue")]
    return OpResult(data=img.data.copy(), rgb=False, display={"mode": "composite"})


@op("type.hsb_stack", "HSB Stack", CAT, kind="image", scope_param=False, preview=False, description="Convert RGB to a hue / saturation / brightness 3-channel stack.")
def hsb_stack(ctx):
    img = ctx.img
    if not img.rgb:
        raise ValueError("Not an RGB image")
    from skimage.color import rgb2hsv
    out = np.zeros_like(img.data)
    for t in range(img.nt):
        for z in range(img.nz):
            hsv = rgb2hsv(img.data[t, z].transpose(1, 2, 0))
            out[t, z] = (hsv.transpose(2, 0, 1) * 255).astype(np.uint8)
    img.display.channels = [ChannelDisplay(name=n, lut="Grays", min=0, max=255) for n in ("Hue", "Saturation", "Brightness")]
    return OpResult(data=out, rgb=False, display={"mode": "single"})


@op("type.lab_stack", "Lab Stack", CAT, kind="image", scope_param=False, preview=False, description="Convert RGB to CIE L*a*b* (32-bit, 3 channels).")
def lab_stack(ctx):
    img = ctx.img
    if not img.rgb:
        raise ValueError("Not an RGB image")
    from skimage.color import rgb2lab
    out = np.zeros(img.shape, np.float32)
    for t in range(img.nt):
        for z in range(img.nz):
            out[t, z] = rgb2lab(img.data[t, z].transpose(1, 2, 0)).transpose(2, 0, 1)
    img.display.channels = [ChannelDisplay(name=n, lut="Grays") for n in ("L*", "a*", "b*")]
    return OpResult(data=out, rgb=False, display={"mode": "single"})
