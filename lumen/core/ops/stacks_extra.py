"""Image ▸ Stacks extras: 3D Project, Orthogonal Views, Reduce, Label, Combine, Insert, Straighten."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .. import roi as roimod
from ..image import Calibration, LumenImage
from ..overlay_render import POSITIONS
from ..registry import OpResult, Param, cast_to, op
from .stacks import _label_grid, _store_labels

CAT = "Image/Stacks"
PROJ3D = ["Brightest Point", "Nearest Point", "Mean Value"]


def _z_factor(img: LumenImage) -> float:
    cal = img.calibration
    if cal.pixel_width > 0 and cal.pixel_depth > 0:
        return float(cal.pixel_depth / cal.pixel_width)
    return 1.0


@op("stacks.project3d", "3D Project", CAT, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("method", "choice", "Brightest Point", choices=PROJ3D), Param("axis", "choice", "Y-Axis", choices=["Y-Axis", "X-Axis", "Z-Axis"], label="Rotation axis"),
            Param("angle_step", "int", 10, min=1, max=180, label="Rotation angle increment (°)"), Param("total_rotation", "int", 360, min=1, max=360), Param("initial_angle", "int", 0, min=-360, max=360),
            Param("lower_threshold", "int", 1, min=0, max=255, label="Lower transparency bound (0–255)"), Param("upper_threshold", "int", 255, min=0, max=255, label="Upper transparency bound (0–255)"),
            Param("depth_cueing", "int", 0, min=0, max=100, label="Surface depth-cueing (%)", help="Nearest Point only: the first voxel on each ray is dimmed by up to this percentage with its depth"),
            Param("interior_depth_cueing", "int", 50, min=0, max=100, label="Interior depth-cueing (%)", help="Brightest Point only: the brightest voxel on each ray is dimmed by up to this percentage with its depth (ImageJ default 50; 0 = off)"),
            Param("interpolate", "bool", True), Param("all_channels", "bool", True),
            Param("max_size", "int", 256, min=32, max=1024, advanced=True, help="Larger volumes are downsampled to this many voxels per axis")],
    description="Rotating projections (brightest point / nearest point / mean) of the volume about the Y, X or Z axis → a time series with one frame per angle (ImageJ 3D Project).",
    keywords=["3d", "rotation", "volume rendering", "projection"])
def project3d(ctx, method="Brightest Point", axis="Y-Axis", angle_step=10, total_rotation=360, initial_angle=0, lower_threshold=1, upper_threshold=255, depth_cueing=0,
              interior_depth_cueing=50, interpolate=True, all_channels=True, max_size=256):
    img = ctx.img
    d = img.display
    if img.nz < 2:
        raise ValueError("3D Project needs a Z stack")
    chans = range(img.nc) if (all_channels and not img.rgb) or img.rgb else [d.c]
    # ImageJ Projector: views at 0, step, 2*step, ... while angle <= total (the end angle is included), minus one when the
    # sequence passes 360° so a full turn does not repeat the 0° view. The over-360 test is on the un-offset angle
    # (ImageJ starts counting at 0 regardless of the initial angle).
    n_proj = int(total_rotation // angle_step) + 1
    if n_proj * angle_step > 360:
        n_proj -= 1
    n_proj = max(n_proj, 1)
    angles = np.arange(n_proj, dtype=float) * angle_step + initial_angle
    order = 1 if interpolate else 0
    zf = _z_factor(img)
    notes = []
    frames: list[np.ndarray] = []
    out_shape = None
    sx = sy = 1.0  # effective XY downsampling factors (input px / output px)
    n_views = len(chans) * len(angles)
    n_done = 0
    for c in chans:
        ch = img.display.channels[c]
        vol = img.data[d.t, :, c].astype(np.float32)  # (Z, Y, X)
        # 8-bit scale for the transparency bounds (relative to the display range)
        vol = np.clip((vol - ch.min) / max(ch.max - ch.min, 1e-12) * 255.0, 0, 255)
        if abs(zf - 1.0) > 1e-3:
            vol = ndi.zoom(vol, (zf, 1, 1), order=1)
        big = max(vol.shape)
        if big > max_size:
            f = max_size / big
            vol = ndi.zoom(vol, f, order=1)
            sx, sy = img.width / vol.shape[2], img.height / vol.shape[1]  # ndi.zoom rounds the shape: use the real ratio
            if c == chans[0]:
                notes.append(f"volume downsampled ×{1 / f:.2f} to {vol.shape[2]}×{vol.shape[1]}×{vol.shape[0]}")
                if img.calibration.pixel_width > 0:
                    notes.append(f"pixel size now {img.calibration.pixel_width * sx:.4g} {img.calibration.unit}")
        nz, ny, nx = vol.shape
        # scipy.ndimage.rotate sorts the axes pair, so the tuple order cannot choose the rotation sense; `sign` makes each
        # series match ImageJ's Projector (about Y: xnew = x cos + z sin; about X: ynew = y cos - z sin;
        # about Z: xnew = x cos - y sin, ynew = x sin + y cos, i.e. clockwise on screen like image.rotate).
        if axis == "Y-Axis":
            rot_axes, sign, D = (0, 2), 1.0, int(math.ceil(math.hypot(nz, nx)))
            pad = ((D - nz) // 2, D - nz - (D - nz) // 2), (0, 0), ((D - nx) // 2, D - nx - (D - nx) // 2)
        elif axis == "X-Axis":
            rot_axes, sign, D = (0, 1), -1.0, int(math.ceil(math.hypot(nz, ny)))
            pad = ((D - nz) // 2, D - nz - (D - nz) // 2), ((D - ny) // 2, D - ny - (D - ny) // 2), (0, 0)
        else:
            rot_axes, sign, D = (1, 2), -1.0, int(math.ceil(math.hypot(ny, nx)))
            pad = (0, 0), ((D - ny) // 2, D - ny - (D - ny) // 2), ((D - nx) // 2, D - nx - (D - nx) // 2)
        volp = np.pad(vol, pad)
        depth = volp.shape[0]
        planes = []
        for ang in angles:
            ctx.check_cancel()
            ctx.progress(n_done / n_views, f"Angle {ang:g}°" + (f" (C{c + 1})" if len(chans) > 1 else ""))
            n_done += 1
            rot = volp if ang % 360 == 0 else ndi.rotate(volp, sign * ang, axes=rot_axes, reshape=False, order=order, prefilter=False, mode="constant", cval=0.0)
            inside = (rot >= lower_threshold) & (rot <= upper_threshold)
            if method == "Brightest Point":
                masked = np.where(inside, rot, 0.0)
                proj = masked.max(axis=0)
                if interior_depth_cueing > 0:
                    # ImageJ picks the raw-brightest voxel and stores its depth-cued value:
                    # dcInt*v/100 + (100-dcInt)*v*(zmax-z)/(zmax-zmin) in dialog units; nearest wins ties (argmax = first hit)
                    zi = masked.argmax(axis=0)
                    proj = proj * (1.0 - interior_depth_cueing / 100.0 * zi / max(depth - 1, 1))
            elif method == "Mean Value":
                cnt = inside.sum(axis=0)
                proj = np.where(cnt > 0, np.where(inside, rot, 0.0).sum(axis=0) / np.maximum(cnt, 1), 0.0)
            else:  # Nearest Point: first voxel inside the bounds along the viewing ray, depth cued
                first = inside.argmax(axis=0)
                has = inside.any(axis=0)
                val = np.take_along_axis(rot, first[None], axis=0)[0]
                if depth_cueing > 0:
                    val = val * (1.0 - depth_cueing / 100.0 * first / max(depth - 1, 1))
                proj = np.where(has, val, 0.0)
            planes.append(proj.astype(np.float32))
        stack = np.stack(planes)  # (angles, H, W)
        # back to the original value range of the channel
        stack = stack / 255.0 * (ch.max - ch.min) + ch.min
        frames.append(stack)
        out_shape = stack.shape
    data = np.stack(frames, axis=1)[:, None]  # (T=angles, Z=1, C, H, W)
    new = LumenImage(cast_to(data, img.data.dtype), name=f"Projections of {img.name}", axes="TZCYX", rgb=img.rgb and len(chans) == 3)
    cal = img.calibration.to_dict()
    cal["pixel_width"] *= sx
    cal["pixel_height"] *= sy
    if sx != 1.0 or sy != 1.0:
        cal["pixel_depth"] = cal["pixel_width"]  # Z was resampled to isotropic (pixel_width) spacing before the downsampling
    cal["frame_interval"] = float(angle_step); cal["time_unit"] = "deg"
    new.calibration = Calibration(**cal)
    if not img.rgb:
        for k, c in enumerate(chans):
            src = img.display.channels[c]
            new.display.channels[k].lut = src.lut; new.display.channels[k].min = src.min; new.display.channels[k].max = src.max; new.display.channels[k].name = src.name
        new.display.mode = "composite" if len(chans) > 1 else "single"
    msg = f"{len(angles)} projections ({method}, {axis})" + (f"; {'; '.join(notes)}" if notes else "")
    return OpResult(new_images=[new], message=msg, log=msg if notes else None)


@op("stacks.orthogonal_views", "Orthogonal Views", CAT, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("x", "int", -1, min=-1, help="Column of the YZ section (-1 = point selection or image centre)"), Param("y", "int", -1, min=-1, help="Row of the XZ section (-1 = point selection or centre)"),
            Param("scale_z", "bool", True, help="Stretch Z by the voxel depth / pixel width ratio")],
    description="XZ and YZ cross-sections through the point selection (or the image centre) as two new images.", keywords=["cross section", "xz", "yz", "ortho"])
def orthogonal_views(ctx, x=-1, y=-1, scale_z=True):
    img = ctx.img
    d = img.display
    if img.nz < 2:
        raise ValueError("Orthogonal Views needs a Z stack")
    if (x < 0 or y < 0) and ctx.roi is not None:
        if ctx.roi.get("type") == "point" and ctx.roi.get("points"):
            px, py = ctx.roi["points"][0]
        else:
            bx, by, bw, bh = roimod.bounding_box(ctx.roi)
            px, py = bx + bw / 2, by + bh / 2
        x = int(px) if x < 0 else x
        y = int(py) if y < 0 else y
    x = img.width // 2 if x < 0 else min(x, img.width - 1)
    y = img.height // 2 if y < 0 else min(y, img.height - 1)
    zf = _z_factor(img) if scale_z else 1.0
    xz = img.data[d.t, :, :, y, :]  # (Z, C, X)
    yz = img.data[d.t, :, :, :, x]  # (Z, C, Y)
    xz = np.ascontiguousarray(xz.transpose(1, 0, 2))  # (C, Z, X)
    yz = np.ascontiguousarray(yz.transpose(1, 2, 0))  # (C, Y, Z)
    if abs(zf - 1.0) > 1e-3:
        xz = cast_to(ndi.zoom(xz.astype(np.float32), (1, zf, 1), order=1), img.data.dtype)
        yz = cast_to(ndi.zoom(yz.astype(np.float32), (1, 1, zf), order=1), img.data.dtype)
    cal = img.calibration
    # physical size of one XZ row / YZ column along Z: pixel_width after the zoom, else one slice = pixel_depth
    # (ImageJ Orthogonal_Views.calibrate: cal_xz.pixelHeight = cal_yz.pixelWidth = depth / az)
    z_size = cal.pixel_depth / zf if cal.pixel_depth > 0 else cal.pixel_width
    out = []
    for arr, name, zkey in ((xz, f"XZ {img.name}", "pixel_height"), (yz, f"YZ {img.name}", "pixel_width")):
        im = img.copy(name=name, data=arr[None, None])
        im.rgb = img.rgb
        im.display.t = im.display.z = 0
        im.sync_channels()
        im.calibration = Calibration(**{**cal.to_dict(), zkey: z_size, "pixel_depth": cal.pixel_width})
        out.append(im)
    return OpResult(new_images=out, roi={"type": "point", "points": [[x + 0.5, y + 0.5]]}, message=f"Sections at x = {x}, y = {y}")


@op("stacks.reduce", "Reduce", CAT, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("factor", "int", 2, min=1, max=1000, label="Reduction factor (keep every n-th)"), Param("axis", "choice", "Z", choices=["Z", "T"])],
    description="Keep every n-th slice / frame (the calibration depth / interval is scaled accordingly).", keywords=["subsample", "decimate"])
def reduce(ctx, factor=2, axis="Z"):
    img = ctx.img
    # ImageJ's StackReducer reduces frames when the hyperstack has a single slice (and slices when it has a single frame)
    if axis == "Z" and img.nz == 1 and img.nt > 1:
        axis = "T"
    elif axis == "T" and img.nt == 1 and img.nz > 1:
        axis = "Z"
    ax = 1 if axis == "Z" else 0
    n = img.data.shape[ax]
    if n < 2:
        raise ValueError("Reduce needs a stack (more than one slice or frame)")
    idx = list(range(0, n, factor))
    data = np.ascontiguousarray(np.take(img.data, idx, axis=ax))
    grid = _label_grid(img)
    if grid is not None:  # in place: run_op installs `data` afterwards, so the kept planes keep their labels (undo brings the full list back)
        _store_labels(img, np.take(grid, idx, axis=ax))
    cal = {}
    if axis == "Z":
        cal["pixel_depth"] = img.calibration.pixel_depth * factor
    else:
        cal["frame_interval"] = img.calibration.frame_interval * factor
    if axis == "Z":
        img.display.z = min(img.display.z // factor, len(idx) - 1)
    else:
        img.display.t = min(img.display.t // factor, len(idx) - 1)
    return OpResult(data=data, rgb=img.rgb, calibration=cal, message=f"{n} → {len(idx)} {'slices' if axis == 'Z' else 'frames'}")


@op("stacks.label_slices", "Label Slices (time stamp)", CAT, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("format", "string", "{frame}", help="Placeholders: {label} {frame} {slice} {time} {unit}; e.g. '{label:04d}' or 't = {time:.1f} {unit}'"),
            Param("position", "choice", "Upper Left", choices=POSITIONS),
            Param("color", "string", "#ffffff"), Param("font_size", "int", 18, min=4, max=300), Param("start_frame", "int", 0, min=0, help="Frame (0-based) that corresponds to t = 0"),
            Param("start", "int", 1, label="Starting value", help="{label} of the first frame / slice (ImageJ Label…: starting value)"),
            Param("interval", "int", 1, help="{label} step per frame / slice (ImageJ Label…: interval)")],
    description="Label every frame / slice with its number or elapsed time (a time_stamp overlay that follows the current frame; Burn In to make it permanent). "
                "{label} = start + n × interval, counting frames when the image has several, else slices.",
    keywords=["label", "time stamp", "frame number"])
def label_slices(ctx, format="{frame}", position="Upper Left", color="#ffffff", font_size=18, start_frame=0, start=1, interval=1):
    from .extras import time_stamp_item
    img = ctx.img
    item = time_stamp_item(format, position, color, font_size, start_frame, label_start=start, label_interval=interval)
    overlay = [o for o in img.overlay if o.get("type") != "time_stamp"] + [item]
    return OpResult(overlay=overlay, message="Slice labels added to the overlay")


def _pad_to(arr: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    pads = [(0, max(s - a, 0)) for a, s in zip(arr.shape, shape)]
    return np.pad(arr, pads) if any(p[1] for p in pads) else arr


@op("stacks.combine", "Combine", CAT, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("image1", "image", None, label="Stack 1"), Param("image2", "image", None, label="Stack 2"), Param("vertical", "bool", False, help="Stack 2 below stack 1 instead of to the right")],
    description="Place two stacks side by side (horizontally or vertically) in a new stack; smaller stacks are zero-padded.", keywords=["side by side", "concatenate xy"])
def combine(ctx, image1=None, image2=None, vertical=False):
    s = ctx.session
    a = s.get_image(image1) if image1 else ctx.img
    b = s.get_image(image2) if image2 else next((im for im in s.images.values() if im.id != a.id), None)
    if b is None:
        raise ValueError("Need two open images")
    if a.nc != b.nc:
        raise ValueError(f"Channel count differs ({a.nc} vs {b.nc})")
    # ImageJ's StackCombiner refuses stacks of different types; like images_to_stack / concatenate we widen instead
    # (uint8 < uint16 < float32: never narrower than either source, so astype is lossless – cast_to clipped a 16-bit
    # stack to 255 and rounded floats when stack 1 was the narrower type)
    out_dtype = np.result_type(a.data.dtype, b.data.dtype)
    da = a.data.astype(out_dtype, copy=False)
    db = b.data.astype(out_dtype, copy=False)
    nt, nz = max(a.nt, b.nt), max(a.nz, b.nz)
    if vertical:
        W = max(a.width, b.width)
        da = _pad_to(da, (nt, nz, a.nc, a.height, W)); db = _pad_to(db, (nt, nz, a.nc, b.height, W))
        data = np.concatenate([da, db], axis=3)
    else:
        H = max(a.height, b.height)
        da = _pad_to(da, (nt, nz, a.nc, H, a.width)); db = _pad_to(db, (nt, nz, a.nc, H, b.width))
        data = np.concatenate([da, db], axis=4)
    new = a.copy(name="Combined Stacks", data=np.ascontiguousarray(data))
    new.display.t = new.display.z = 0
    promoted = out_dtype != a.data.dtype
    if promoted:
        new.reset_display(keep_luts=True)  # copy() kept stack 1's 8-bit 0–255 range, which would show the widened half saturated
    new.sync_channels()
    return OpResult(new_images=[new], message=f"{a.name} + {b.name} → {new.width}×{new.height}" + (f" ({np.dtype(out_dtype).name})" if promoted else ""))


@op("stacks.insert", "Insert", CAT, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("source", "image", None, help="Image to paste into the current one"), Param("x", "int", 0, min=0), Param("y", "int", 0, min=0)],
    description="Paste another stack into the current one at (x, y); every matching slice / frame / channel is copied (ImageJ Stack Inserter).", keywords=["paste stack", "inset"])
def insert(ctx, source=None, x=0, y=0):
    s = ctx.session
    img = ctx.img
    src = s.get_image(source) if source else next((im for im in s.images.values() if im.id != img.id), None)
    if src is None:
        raise ValueError("Pick a source image")
    # ImageJ's StackInserter refuses a source of another type; like Combine / images_to_stack the destination is
    # widened instead (lossless, undoable), so a 16-bit source pasted into an 8-bit stack no longer clips to 255
    out_dtype = np.result_type(img.data.dtype, src.data.dtype)
    promoted = out_dtype != img.data.dtype
    data = img.data.astype(out_dtype, copy=True)
    hh = min(src.height, img.height - y)
    ww = min(src.width, img.width - x)
    if hh <= 0 or ww <= 0:
        raise ValueError("Insert position is outside the image")
    for t in range(min(img.nt, src.nt)):
        for z in range(min(img.nz, src.nz)):
            for c in range(img.nc):
                sc = min(c, src.nc - 1)
                data[t, z, c, y:y + hh, x:x + ww] = src.data[t, z, sc, :hh, :ww].astype(out_dtype, copy=False)
    display = None
    if promoted:
        # replace_data() would keep the old 8-bit 0–255 range (it lies inside the wider type's range): hand the
        # viewer a range that fits the widened data, keeping the LUTs
        tmp = img.copy(data=data)
        tmp.reset_display(keep_luts=True)
        display = {"channels": [{"min": ch.min, "max": ch.max} for ch in tmp.display.channels]}
    msg = f"{src.name} inserted at ({x}, {y})" + (f" ({img.data.dtype.name} promoted to {np.dtype(out_dtype).name})" if promoted else "")
    return OpResult(data=data, rgb=bool(img.rgb and not promoted), display=display,
                    roi={"type": "rect", "x": x, "y": y, "w": ww, "h": hh}, message=msg)


def straighten_coordinates(points: list[tuple[float, float]], width: int, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Fit a spline through the polyline points and return (ys, xs) sampling grids of shape (width, L)."""
    pts = np.asarray(points, float)
    keep = [0] + [i for i in range(1, len(pts)) if np.hypot(*(pts[i] - pts[i - 1])) > 1e-6]
    pts = pts[keep]
    if len(pts) < 2:
        raise ValueError("Need a line with at least two distinct points")
    seg = np.hypot(*np.diff(pts, axis=0).T)
    total = float(seg.sum())
    n_dense = max(int(total * 4), 16)
    if len(pts) == 2:
        u = np.linspace(0, 1, n_dense)
        xd = pts[0, 0] + u * (pts[1, 0] - pts[0, 0]); yd = pts[0, 1] + u * (pts[1, 1] - pts[0, 1])
    else:
        from scipy.interpolate import splev, splprep
        k = min(3, len(pts) - 1)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, k=k)
        xd, yd = splev(np.linspace(0, 1, n_dense), tck)
        xd, yd = np.asarray(xd), np.asarray(yd)
    cum = np.concatenate([[0], np.cumsum(np.hypot(np.diff(xd), np.diff(yd)))])
    L = max(int(round(cum[-1] / spacing)), 2)
    s_even = np.linspace(0, cum[-1], L)
    xs = np.interp(s_even, cum, xd); ys = np.interp(s_even, cum, yd)
    tx, ty = np.gradient(xs), np.gradient(ys)
    norm = np.hypot(tx, ty); norm[norm == 0] = 1
    nx, ny = -ty / norm, tx / norm
    offs = (np.arange(width) - (width - 1) / 2.0)[:, None]
    gx = xs[None, :] + offs * nx[None, :]
    gy = ys[None, :] + offs * ny[None, :]
    return gy, gx


@op("stacks.straighten", "Straighten", CAT, kind="image", scope_param=False, preview=False, roi_aware=False, requires_roi="line",
    params=[Param("line_width", "int", 20, min=1, max=2000), Param("all_planes", "bool", True, help="Process every slice / frame / channel")],
    description="Straighten a curved (segmented / freehand) line selection: a spline is fitted through the points and the image is resampled along it (ImageJ Edit ▸ Selection ▸ Straighten).",
    keywords=["unbend", "curved line", "spline", "profile"])
def straighten(ctx, line_width=20, all_planes=True):
    img = ctx.img
    d = img.display
    roi = ctx.roi
    width = int(line_width) if line_width > 0 else int(roi.get("width", 1) or 1)
    gy, gx = straighten_coordinates(roimod.line_points(roi), width)
    planes = list(img.iter_planes()) if all_planes else [(d.t, d.z, c) for c in (range(img.nc) if img.rgb else [d.c])]
    ts = sorted({p[0] for p in planes}); zs = sorted({p[1] for p in planes}); cs = sorted({p[2] for p in planes})
    out = np.zeros((len(ts), len(zs), len(cs), width, gx.shape[1]), np.float32)
    for (t, z, c) in planes:
        out[ts.index(t), zs.index(z), cs.index(c)] = ndi.map_coordinates(img.data[t, z, c].astype(np.float32), [gy - 0.5, gx - 0.5], order=1, mode="nearest")
    new = img.copy(name=f"Straightened {img.name}", data=cast_to(out, img.data.dtype))
    if not all_planes and not img.rgb:
        new.display.channels = [type(img.display.channels[d.c])(**img.display.channels[d.c].to_dict())]
    new.display.t = new.display.z = new.display.c = 0
    new.sync_channels()
    return OpResult(new_images=[new], message=f"Straightened: {gx.shape[1]} × {width} px")
