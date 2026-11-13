"""Image ▸ Crop / Scale / Rotate / Transform / Duplicate / Properties."""
from __future__ import annotations

import numpy as np
from skimage.transform import resize

from .. import roi as roimod
from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op, cast_to
from . import _imagej_types as ijt

CAT = "Image"


@op("image.crop", "Crop", CAT, kind="image", requires_roi="area", scope_param=False, preview=False, shortcut="Shift+X",
    description="Crop the whole stack to the bounding box of the selection.")
def crop(ctx):
    img = ctx.img
    x, y, w, h = roimod.bounding_box(ctx.roi)
    x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
    x1, y1 = min(int(round(x + w)), img.width), min(int(round(y + h)), img.height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Selection is empty")
    data = img.data[:, :, :, y0:y1, x0:x1].copy()
    new_roi = roimod.translate(ctx.roi, -x0, -y0) if ctx.roi["type"] != "rect" else None
    return OpResult(data=data, rgb=img.rgb, roi=new_roi, clear_roi=new_roi is None)


@op("image.duplicate", "Duplicate", CAT, kind="image", scope_param=False, preview=False, shortcut="Shift+D",
    description="Copy the image (or the current plane / channel) into a new tab.",
    params=[Param("title", "string", ""), Param("what", "choice", "Whole stack", choices=["Whole stack", "Current plane", "Current channel", "Current slice (all channels)"]),
            Param("crop_to_selection", "bool", True)])
def duplicate(ctx, title="", what="Whole stack", crop_to_selection=True):
    img = ctx.img
    d = img.display
    data = img.data
    if what == "Current plane":
        data = data[d.t:d.t + 1, d.z:d.z + 1, d.c:d.c + 1] if not img.rgb else data[d.t:d.t + 1, d.z:d.z + 1]
    elif what == "Current channel":
        data = data[:, :, d.c:d.c + 1]
    elif what == "Current slice (all channels)":
        data = data[d.t:d.t + 1, d.z:d.z + 1]
    roi = ctx.roi
    if crop_to_selection and roi is not None and roimod.is_area(roi):
        x, y, w, h = roimod.bounding_box(roi)
        x0, y0 = max(int(round(x)), 0), max(int(round(y)), 0)
        x1, y1 = min(int(round(x + w)), img.width), min(int(round(y + h)), img.height)
        data = data[:, :, :, y0:y1, x0:x1]
    new = img.copy(name=title or f"{img.name}-1", data=data.copy())
    if what == "Current channel" and not img.rgb:
        new.display.channels = [type(img.display.channels[d.c])(**img.display.channels[d.c].to_dict())]
        new.display.c = 0
        new.display.mode = "single"
    elif what == "Current plane" and not img.rgb:
        new.display.channels = [type(img.display.channels[d.c])(**img.display.channels[d.c].to_dict())]
        new.display.c = 0; new.display.z = 0; new.display.t = 0
    elif what == "Current slice (all channels)":
        new.display.z = 0; new.display.t = 0
    new.sync_channels()
    return OpResult(new_images=[new])


def _java_round(x: float) -> int:
    """``(int)Math.round(x)``."""
    return int(np.floor(x + 0.5))


@op("image.scale", "Scale", CAT, kind="image", scope_param=False, preview=False,
    description="Resize by factors (or to a target size) with ImageJ's resampling (ImageProcessor.resize: pixel centres offset by scale/4, bilinear / Catmull-Rom bicubic, "
                "'average when downsizing' = ImageJ's kernel averaging); updates the calibration. 'Keep canvas' scales the content inside the current canvas instead (ImageJ's Scale… without a new window).",
    params=[Param("x_scale", "float", 0.5, min=0.01, max=50, step=0.05), Param("y_scale", "float", 0.5, min=0.01, max=50, step=0.05), Param("z_scale", "float", 1.0, min=0.01, max=50, step=0.05),
            Param("width", "int", 0, min=0, help="Target width (overrides x scale if > 0)"), Param("height", "int", 0, min=0, help="Target height (overrides y scale if > 0)"),
            Param("interpolation", "choice", "Bilinear", choices=["None", "Bilinear", "Bicubic"]), Param("create_new", "bool", True, label="Create new window"), Param("average_when_downsizing", "bool", True),
            Param("keep_canvas", "bool", False, help="Scale the content about the image centre inside the current canvas (background 0 / display minimum); only without a new window")])
def scale(ctx, x_scale=0.5, y_scale=0.5, z_scale=1.0, width=0, height=0, interpolation="Bilinear", create_new=True, average_when_downsizing=True, keep_canvas=False):
    img = ctx.img
    H, W = img.height, img.width
    method = ijt.INTERPOLATION[interpolation]
    if keep_canvas and not create_new:
        xs = width / W if width > 0 else x_scale
        ys = height / H if height > 0 else y_scale
        out = img.data.copy()
        for (t, z, c) in img.iter_planes():
            ctx.check_cancel()
            ch = img.display.channels[c]
            fill = 0.0 if img.data.dtype == np.uint8 else float(ch.min)  # ImageJ: bgColor for bytes, getMin() for short / float
            out[t, z, c] = ijt.scale_in_place(img.data[t, z, c], xs, ys, method, bg=0.0, fill=fill)
        return OpResult(data=out, rgb=img.rgb, clear_roi=True)
    nw = width if width > 0 else max(_java_round(W * x_scale), 1)
    nh = height if height > 0 else max(_java_round(H * y_scale), 1)
    nz = max(_java_round(img.nz * z_scale), 1)
    planes = np.zeros((img.nt, img.nz, img.nc, nh, nw), img.data.dtype)
    n = img.nt * img.nz * img.nc
    for i, (t, z, c) in enumerate(img.iter_planes()):
        ctx.check_cancel()
        planes[t, z, c] = ijt.resize(img.data[t, z, c], nw, nh, method, average_when_downsizing, bg=0.0)
        if n > 1:
            ctx.progress((i + 1) / n, f"Plane {i + 1}/{n}")
    if nz != img.nz:
        order = {"None": 0, "Bilinear": 1, "Bicubic": 3}[interpolation]
        out = np.zeros((img.nt, nz, img.nc, nh, nw), np.float32)
        for t in range(img.nt):
            for c in range(img.nc):
                out[t, :, c] = resize(planes[t, :, c].astype(np.float32), (nz, nh, nw), order=order, anti_aliasing=average_when_downsizing and order > 0 and nz < img.nz, preserve_range=True)
        data = cast_to(out, img.data.dtype)
    else:
        data = planes
    cal = img.calibration.to_dict()
    cal["pixel_width"] *= W / nw
    cal["pixel_height"] *= H / nh
    cal["pixel_depth"] *= img.nz / nz
    if create_new:
        new = img.copy(name=f"{img.name}-scaled", data=data)
        new.calibration = Calibration(**cal)
        return OpResult(new_images=[new])
    return OpResult(data=data, rgb=img.rgb, calibration=cal, clear_roi=True)


@op("image.rotate", "Rotate", CAT, kind="image", scope_param=False, preview=True,
    description="Rotate by an arbitrary angle (degrees, clockwise positive) like ImageJ's Image ▸ Transform ▸ Rotate: about the pixel centre ((w−1)/2, (h−1)/2), nearest / bilinear / "
                "Catmull-Rom bicubic, uncovered pixels 0; 'enlarge' grows the canvas to the rotated bounds first.",
    params=[Param("angle", "float", 15.0, min=-360, max=360, step=1), Param("interpolation", "choice", "Bilinear", choices=["None", "Bilinear", "Bicubic"]), Param("enlarge", "bool", False, help="Enlarge the canvas to fit")])
def rotate(ctx, angle=15.0, interpolation="Bilinear", enlarge=False):
    img = ctx.img
    method = ijt.INTERPOLATION[interpolation]
    data = img.data
    if enlarge:
        w2, h2 = ijt.enlarged_canvas(img.width, img.height, angle)
        ox, oy = ijt.canvas_offset(img.width, img.height, w2, h2, "Center")
        data = ijt.expand_canvas(data, w2, h2, ox, oy, 0)
    out = np.empty_like(data)
    n = img.nt * img.nz * img.nc
    for i, (t, z, c) in enumerate(img.iter_planes()):
        ctx.check_cancel()
        out[t, z, c] = ijt.rotate(data[t, z, c], angle, method, 0.0)
        if n > 1:
            ctx.progress((i + 1) / n, f"Plane {i + 1}/{n}")
    return OpResult(data=out, rgb=img.rgb, clear_roi=True)


@op("image.rotate90", "Rotate 90°", "Image/Transform", kind="image", scope_param=False, preview=False,
    params=[Param("direction", "choice", "Right", choices=["Right", "Left"])], description="Rotate the stack by 90 degrees.")
def rotate90(ctx, direction="Right"):
    img = ctx.img
    k = -1 if direction == "Right" else 1
    return OpResult(data=np.ascontiguousarray(np.rot90(img.data, k=k, axes=(3, 4))), rgb=img.rgb, clear_roi=True,
                    calibration={"pixel_width": img.calibration.pixel_height, "pixel_height": img.calibration.pixel_width})


@op("image.flip_h", "Flip Horizontally", "Image/Transform", kind="image", scope_param=False, preview=False, description="Mirror left ↔ right.")
def flip_h(ctx):
    img = ctx.img
    data = img.data.copy()
    mask = ctx.mask
    if mask is not None:
        x, y, w, h = roimod.bounding_box(ctx.roi)
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        data[..., y0:y1, x0:x1] = data[..., y0:y1, x0:x1][..., ::-1]
        return OpResult(data=data, rgb=img.rgb)
    return OpResult(data=np.ascontiguousarray(data[..., ::-1]), rgb=img.rgb)


@op("image.flip_v", "Flip Vertically", "Image/Transform", kind="image", scope_param=False, preview=False, description="Mirror top ↔ bottom.")
def flip_v(ctx):
    img = ctx.img
    data = img.data.copy()
    if ctx.mask is not None:
        x, y, w, h = roimod.bounding_box(ctx.roi)
        x0, y0, x1, y1 = int(x), int(y), int(x + w), int(y + h)
        data[..., y0:y1, x0:x1] = data[..., y0:y1, x0:x1][..., ::-1, :]
        return OpResult(data=data, rgb=img.rgb)
    return OpResult(data=np.ascontiguousarray(data[..., ::-1, :]), rgb=img.rgb)


@op("image.flip_z", "Flip Z", "Image/Transform", kind="image", scope_param=False, preview=False, description="Reverse the slice order.")
def flip_z(ctx):
    return OpResult(data=np.ascontiguousarray(ctx.img.data[:, ::-1]), rgb=ctx.img.rgb)


@op("image.translate", "Translate", "Image/Transform", kind="image", scope_param=False, preview=True,
    params=[Param("dx", "float", 10.0, step=1), Param("dy", "float", 10.0, step=1), Param("interpolation", "choice", "None", choices=["None", "Bilinear", "Bicubic"])],
    description="Shift the image content like ImageJ's Translate (integer offsets copy pixels, fractional ones interpolate; uncovered pixels 0).")
def translate(ctx, dx=10.0, dy=10.0, interpolation="None"):
    img = ctx.img
    method = ijt.INTERPOLATION[interpolation]
    out = np.zeros_like(img.data)
    for (t, z, c) in img.iter_planes():
        ctx.check_cancel()
        out[t, z, c] = ijt.translate(img.data[t, z, c], float(dx), float(dy), method)
    return OpResult(data=out, rgb=img.rgb)


@op("image.canvas_size", "Canvas Size", "Image/Adjust", kind="image", scope_param=False, preview=False,
    params=[Param("width", "int", 512, min=1), Param("height", "int", 512, min=1),
            Param("position", "choice", "Center", choices=["Center", "Top-Left", "Top-Center", "Top-Right", "Center-Left", "Center-Right", "Bottom-Left", "Bottom-Center", "Bottom-Right"]),
            Param("zero_fill", "bool", True)], description="Change the canvas size without scaling the content.")
def canvas_size(ctx, width=512, height=512, position="Center", zero_fill=True):
    img = ctx.img
    H, W = img.height, img.width
    fill = 0 if zero_fill else (255 if img.data.dtype == np.uint8 else 0)
    out = np.full((img.nt, img.nz, img.nc, height, width), fill, img.data.dtype)
    v, h_ = position.split("-") if "-" in position else ("Center", "Center")
    if position == "Center":
        v, h_ = "Center", "Center"
    oy = {"Top": 0, "Center": (height - H) // 2, "Bottom": height - H}[v]
    ox = {"Left": 0, "Center": (width - W) // 2, "Right": width - W}[h_]
    sy0, sx0 = max(0, -oy), max(0, -ox)
    dy0, dx0 = max(0, oy), max(0, ox)
    hh = min(H - sy0, height - dy0)
    ww = min(W - sx0, width - dx0)
    if hh > 0 and ww > 0:
        out[..., dy0:dy0 + hh, dx0:dx0 + ww] = img.data[..., sy0:sy0 + hh, sx0:sx0 + ww]
    return OpResult(data=out, rgb=img.rgb, clear_roi=True)


@op("image.bin", "Bin", "Image/Transform", kind="image", scope_param=False, preview=False,
    params=[Param("x", "int", 2, min=1, max=64), Param("y", "int", 2, min=1, max=64), Param("z", "int", 1, min=1, max=64),
            Param("method", "choice", "Average", choices=["Average", "Median", "Max", "Min", "Sum"])], description="Reduce the size by combining blocks of pixels.")
def bin_(ctx, x=2, y=2, z=1, method="Average"):
    img = ctx.img
    d = img.data
    nz, H, W = d.shape[1] // z * z, d.shape[3] // y * y, d.shape[4] // x * x
    d = d[:, :nz, :, :H, :W].astype(np.float32)
    r = d.reshape(d.shape[0], nz // z, z, d.shape[2], H // y, y, W // x, x)
    fn = {"Average": np.mean, "Median": np.median, "Max": np.max, "Min": np.min, "Sum": np.sum}[method]
    out = fn(fn(fn(r, axis=7), axis=5), axis=2)
    cal = img.calibration.to_dict()
    cal["pixel_width"] *= x; cal["pixel_height"] *= y; cal["pixel_depth"] *= z
    return OpResult(data=cast_to(out, img.data.dtype), rgb=img.rgb, calibration=cal, clear_roi=True)


@op("image.properties", "Properties / Set Scale", CAT, kind="analysis", scope_param=False, preview=False, roi_aware=False,
    params=[Param("pixel_width", "float", 1.0, min=0, step=0.01), Param("pixel_height", "float", 1.0, min=0, step=0.01), Param("voxel_depth", "float", 1.0, min=0, step=0.01),
            Param("unit", "string", "pixel"), Param("frame_interval", "float", 0.0, min=0, step=0.1), Param("time_unit", "string", "sec")],
    description="Set the spatial / temporal calibration of the image.", keywords=["calibration", "scale", "micron"])
def properties(ctx, pixel_width=1.0, pixel_height=1.0, voxel_depth=1.0, unit="pixel", frame_interval=0.0, time_unit="sec"):
    img = ctx.img
    img.calibration = Calibration(pixel_width=pixel_width, pixel_height=pixel_height, pixel_depth=voxel_depth, unit=unit, frame_interval=frame_interval, time_unit=time_unit)
    return OpResult(message=f"Calibration: {pixel_width:g} × {pixel_height:g} {unit}/px")


@op("image.set_scale_from_line", "Set Scale from Line", CAT, kind="analysis", requires_roi="line", roi_aware=False, preview=False,
    params=[Param("known_distance", "float", 100.0, min=0, step=1), Param("unit", "string", "µm")], description="Calibrate using a line selection of known length.")
def set_scale_from_line(ctx, known_distance=100.0, unit="µm"):
    img = ctx.img
    L = roimod.line_length(ctx.roi)
    if L <= 0 or known_distance <= 0:
        raise ValueError("Line length and known distance must be > 0")
    pw = known_distance / L
    img.calibration = Calibration(**{**img.calibration.to_dict(), "pixel_width": pw, "pixel_height": pw, "unit": unit})
    return OpResult(message=f"Scale: {1 / pw:.4g} px/{unit}")


@op("image.rename", "Rename", CAT, kind="analysis", preview=False, roi_aware=False, params=[Param("title", "string", "")], description="Rename the image tab.")
def rename(ctx, title=""):
    if not title.strip():
        raise ValueError("Title is empty")
    return OpResult(rename=title.strip())
