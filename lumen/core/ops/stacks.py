"""Image ▸ Stacks and Hyperstacks."""
from __future__ import annotations

import numpy as np

from ..image import LumenImage
from ..lazy import get_plane, is_virtual, stream_project, stream_statistics, take_planes
from ..registry import Param, OpResult, op, cast_to
from ._util import as_float

CAT = "Image/Stacks"
PROJ = ["Max Intensity", "Min Intensity", "Average Intensity", "Sum Slices", "Standard Deviation", "Median"]
#: ImageJ ZProjector.makeTitle prefixes
PROJ_PREFIX = {"Max Intensity": "MAX", "Min Intensity": "MIN", "Average Intensity": "AVG", "Sum Slices": "SUM", "Standard Deviation": "STD", "Median": "MED"}


def _project(vol: np.ndarray, method: str) -> np.ndarray:
    """ImageJ's ``ZProjector`` ray functions on an (N, …, Y, X) volume, NaN-aware (NaN planes are skipped per pixel):
    Average / Sum accumulate in float32 plane by plane and divide in float32, the Standard Deviation sums in double
    (``sqrt((n·Σv² − (Σv)²) / n / (n − 1))``), the Median sorts the float32 values and averages the two middle ones in
    float32.  The result is float32; the caller rounds it into the image type like ``makeOutputImage``."""
    f = vol.astype(np.float32, copy=False)
    if method == "Max Intensity":
        return np.nanmax(f, axis=0)
    if method == "Min Intensity":
        return np.nanmin(f, axis=0)
    shape = f.shape[1:]
    if method in ("Average Intensity", "Sum Slices"):
        acc = np.zeros(shape, np.float32)
        cnt = np.zeros(shape, np.int32)
        for k in range(f.shape[0]):
            p = f[k]
            ok = ~np.isnan(p)
            acc += np.where(ok, p, np.float32(0))
            cnt += ok
        if method == "Sum Slices":
            return acc
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(cnt > 0, acc / cnt.astype(np.float32), np.float32(np.nan)).astype(np.float32)
    if method == "Standard Deviation":
        s = np.zeros(shape, np.float64)
        s2 = np.zeros(shape, np.float64)
        cnt = np.zeros(shape, np.int64)
        for k in range(f.shape[0]):
            p = f[k].astype(np.float64)
            ok = ~np.isnan(p)
            pv = np.where(ok, p, 0.0)
            s += pv
            s2 += pv * pv
            cnt += ok
        n = cnt.astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (n * s2 - s * s) / n
            sd = np.where((cnt > 1) & (var > 0), np.sqrt(var / (n - 1.0)), 0.0).astype(np.float32)
        return np.where(cnt > 0, sd, np.float32(np.nan)).astype(np.float32)
    if method == "Median":
        srt = np.sort(f, axis=0)                      # NaNs sort last
        cnt = (~np.isnan(f)).sum(axis=0)
        m = cnt // 2
        last = f.shape[0] - 1
        hi = np.take_along_axis(srt, np.minimum(m, last)[None], axis=0)[0]
        lo = np.take_along_axis(srt, np.clip(m - 1, 0, last)[None], axis=0)[0]
        med = np.where((cnt & 1) == 0, (lo + hi) / np.float32(2), hi).astype(np.float32)
        return np.where(cnt > 0, med, np.float32(np.nan)).astype(np.float32)
    raise ValueError(method)


def _projection_dtype(method: str, dtype: np.dtype) -> np.dtype:
    """The type of ImageJ's projection image: Sum and SD are 32-bit, the Median of anything but an 8-bit stack is
    32-bit too (``doMedianProjection`` only converts 8-bit stacks back), the rest keep the stack's type."""
    if method in ("Sum Slices", "Standard Deviation"):
        return np.dtype(np.float32)
    if method == "Median":
        return np.dtype(np.uint8) if dtype == np.uint8 else np.dtype(np.float32)
    return np.dtype(dtype)


def _to_projection_type(out: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """``ZProjector.makeOutputImage`` / ``ByteProcessor.setPixels(0, fp)``: ``(int)(v + 0.5f)`` into 8/16-bit, float as is."""
    if dtype == np.float32:
        return out.astype(np.float32, copy=False)
    from ._imagej import round_half_up
    return round_half_up(out, dtype)


@op("stacks.z_project", "Z Project", CAT, kind="image", scope_param=False, preview=False, shortcut="",
    description="Collapse the Z axis (max / min / mean / sum / SD / median) for every channel and frame.",
    params=[Param("method", "choice", "Max Intensity", choices=PROJ), Param("start", "int", 1, min=1), Param("stop", "int", 0, min=0, help="0 = last slice"),
            Param("axis", "choice", "Z", choices=["Z", "T"], advanced=True)], keywords=["maximum projection", "mip"])
def z_project(ctx, method="Max Intensity", start=1, stop=0, axis="Z"):
    img = ctx.img
    if axis == "Z" and img.nz == 1 and img.nt > 1:
        axis = "T"  # ImageJ ZProjector.doHyperStackProjection: a single-slice time series is projected over its frames (E10-6)
    ax = 1 if axis == "Z" else 0
    n = img.data.shape[ax]
    if n < 2:
        raise ValueError(f"Z Project needs a stack with more than one {'slice' if axis == 'Z' else 'frame'}")
    what = "slice" if axis == "Z" else "frame"
    a, b = max(start - 1, 0), min(stop if stop > 0 else n, n)
    if a >= b:
        # empty selection (start beyond the stack, or start > stop): Max / Min used to surface numpy's raw
        # "zero-size array to reduction operation" error and the other methods silently projected to all zeros
        raise ValueError(f"Z Project: start {what} {start} must be ≤ stop {what} {b} (the stack has {n} {what}s)")
    if is_virtual(img.data):
        # a virtual stack is projected plane by plane (one streaming pass for max / min / mean / sum, two for the SD,
        # row tiles for the median): the whole stack is never read into memory
        out = stream_project(img.data, ax, method, a, b, progress=ctx.progress, check_cancel=ctx.check_cancel)
    else:
        sub = np.take(img.data, range(a, b), axis=ax)
        moved = np.moveaxis(sub, ax, 0)  # (N, ..., Y, X)
        out = _project(moved, method)[None]  # (1, ...)
        out = np.moveaxis(out, 0, ax)
    dtype = _projection_dtype(method, img.data.dtype)
    res = img.copy(name=f"{PROJ_PREFIX.get(method, method.split()[0].upper())}_{img.name}", data=_to_projection_type(out, dtype))
    _store_labels(res, None)  # a projection has no per-slice identity (ImageJ's ZProjector result is unlabelled)
    res.display.z = 0 if axis == "Z" else res.display.z
    res.display.t = 0 if axis == "T" else res.display.t
    res.reset_display(keep_luts=True)
    return OpResult(new_images=[res])


@op("stacks.grouped_z_project", "Grouped Z Project", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("group_size", "int", 2, min=1), Param("method", "choice", "Max Intensity", choices=PROJ)], description="Project every group of N consecutive slices.")
def grouped_z_project(ctx, group_size=2, method="Max Intensity"):
    img = ctx.img
    nz = img.nz // group_size
    if nz < 1:
        raise ValueError("Group size larger than the stack")
    out = np.stack([_project(img.data[:, i * group_size:(i + 1) * group_size].transpose(1, 0, 2, 3, 4), method) for i in range(nz)], axis=1)
    dtype = _projection_dtype(method, img.data.dtype)
    res = img.copy(name=f"{PROJ_PREFIX.get(method, method.split()[0].upper())}_{img.name}", data=_to_projection_type(out, dtype))
    _store_labels(res, None)
    res.reset_display(keep_luts=True)
    return OpResult(new_images=[res])


@op("stacks.substack", "Make Substack", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("slices", "string", "1-10", help="e.g. 1-10, 1-20-2 (step), or 3,5,8"), Param("axis", "choice", "Z", choices=["Z", "T", "C"]), Param("delete_from_original", "bool", False)],
    description="Extract a range of slices / frames / channels into a new stack.")
def substack(ctx, slices="1-10", axis="Z", delete_from_original=False):
    img = ctx.img
    ax = {"Z": 1, "T": 0, "C": 2}[axis]
    n = img.data.shape[ax]
    idx = _parse_range(slices, n)
    key = {"Z": "z_idx", "T": "t_idx", "C": "c_idx"}[axis]
    if is_virtual(img.data):  # only the planes of the range are read, one at a time: the substack of a virtual stack stays small
        data = take_planes(img.data, **{key: idx}, progress=ctx.progress, check_cancel=ctx.check_cancel)
    else:
        data = np.take(img.data, idx, axis=ax).copy()
    res = img.copy(name=f"Substack ({slices})", data=data)
    grid = _label_grid(img)
    _store_labels(res, np.take(grid, idx, axis=ax) if grid is not None else None)  # labels travel with their planes
    if axis == "C" and not img.rgb:
        res.display.channels = [type(img.display.channels[i])(**img.display.channels[i].to_dict()) for i in idx]
    res.display.z = res.display.t = res.display.c = 0
    res.sync_channels()
    out = OpResult(new_images=[res])
    if delete_from_original:
        keep = [i for i in range(n) if i not in set(idx)]
        if not keep:
            raise ValueError("Cannot delete every slice")
        out.data = take_planes(img.data, **{key: keep}) if is_virtual(img.data) else np.take(img.data, keep, axis=ax).copy()
        # an RGB image that loses a channel becomes a plain 8-bit composite (as delete_slice / make_subset do):
        # rgb=True on a C != 3 array breaks render_rgb, rgb.png and the frontend composite
        out.rgb = img.rgb and axis != "C"
        if axis == "C":
            # run_op installs out.data afterwards; the surviving channels keep their own names / LUTs
            # (sync_channels() would otherwise re-label the shortened list by index: GFP/RFP data shown as DAPI/GFP)
            img.display.channels = [img.display.channels[i] for i in keep]
            if len(keep) == 1:
                img.display.mode = "single"
        if grid is not None:  # run_op swaps img.data for out.data afterwards: the original keeps the labels of the kept planes
            _store_labels(img, np.take(grid, keep, axis=ax))
    return out


def _parse_range(spec: str, n: int) -> list[int]:
    spec = spec.strip()
    idx: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            bits = [int(b) for b in part.split("-")]
            a, b = bits[0], bits[1]
            step = bits[2] if len(bits) > 2 else 1
            idx += list(range(a - 1, min(b, n), step))
        else:
            idx.append(int(part) - 1)
    idx = [i for i in idx if 0 <= i < n]
    if not idx:
        raise ValueError(f"No valid slices in {spec!r} (stack has {n})")
    return idx


@op("stacks.delete_slice", "Delete Slice", CAT, kind="image", scope_param=False, preview=False, params=[Param("axis", "choice", "Z", choices=["Z", "T", "C"])], description="Remove the current slice / frame / channel.")
def delete_slice(ctx, axis="Z"):
    img = ctx.img
    ax = {"Z": 1, "T": 0, "C": 2}[axis]
    cur = {"Z": img.display.z, "T": img.display.t, "C": img.display.c}[axis]
    n = img.data.shape[ax]
    if n <= 1:
        raise ValueError("Only one slice left")
    data = np.delete(img.data, cur, axis=ax)
    if axis == "C" and not img.rgb:
        img.display.channels.pop(cur)
    grid = _label_grid(img)  # built from the old shape; run_op installs `data` afterwards, so the shortened list then matches
    if grid is not None:
        _store_labels(img, np.delete(grid, cur, axis=ax))
    return OpResult(data=data, rgb=img.rgb and axis != "C")


@op("stacks.add_slice", "Add Slice", CAT, kind="image", scope_param=False, preview=False, params=[Param("axis", "choice", "Z", choices=["Z", "T", "C"])], description="Insert a blank slice / frame / channel after the current one.")
def add_slice(ctx, axis="Z"):
    img = ctx.img
    if axis == "C":
        if img.rgb:
            raise ValueError("Cannot add a channel to an RGB image – convert it to an RGB stack first")
        cur = img.display.c
        from ..image import ChannelDisplay
        blank = np.zeros_like(img.data[:, :, cur:cur + 1])
        data = np.concatenate([img.data[:, :, :cur + 1], blank, img.data[:, :, cur + 1:]], axis=2)
        src = img.display.channels[cur]
        img.display.channels.insert(cur + 1, ChannelDisplay(name=f"C{cur + 2}", lut="Grays", min=src.min, max=src.max, visible=True))
        _insert_blank_label(img, cur + 1, 2)
        return OpResult(data=data, rgb=False)
    ax = 1 if axis == "Z" else 0
    cur = img.display.z if axis == "Z" else img.display.t
    blank = np.zeros_like(np.take(img.data, [cur], axis=ax))
    _insert_blank_label(img, cur + 1, ax)
    return OpResult(data=np.concatenate([np.take(img.data, range(0, cur + 1), axis=ax), blank, np.take(img.data, range(cur + 1, img.data.shape[ax]), axis=ax)], axis=ax), rgb=img.rgb)


def _insert_blank_label(img: LumenImage, pos: int, ax: int) -> None:
    """Add Slice: the new plane at `pos` along axis `ax` (0 = T, 1 = Z, 2 = C) gets an empty label, every later label moves with its plane."""
    grid = _label_grid(img)
    if grid is not None:
        _store_labels(img, np.insert(grid, pos, "", axis=ax))


@op("stacks.reverse", "Reverse", CAT, kind="image", scope_param=False, preview=False, params=[Param("axis", "choice", "Z", choices=["Z", "T"])], description="Reverse the slice order.")
def reverse(ctx, axis="Z"):
    ax = 1 if axis == "Z" else 0
    grid = _label_grid(ctx.img)
    if grid is not None:
        _store_labels(ctx.img, np.flip(grid, axis=ax))
    return OpResult(data=np.ascontiguousarray(np.flip(ctx.img.data, axis=ax)), rgb=ctx.img.rgb)


@op("stacks.montage", "Make Montage", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("columns", "int", 0, min=0, help="0 = automatic"), Param("rows", "int", 0, min=0), Param("scale", "float", 1.0, min=0.05, max=1.0, step=0.05),
            Param("border", "int", 0, min=0, max=50), Param("axis", "choice", "Z", choices=["Z", "T", "C"])], description="Tile the slices into one image.")
def montage(ctx, columns=0, rows=0, scale=1.0, border=0, axis="Z"):
    img = ctx.img
    ax = {"Z": 1, "T": 0, "C": 2}[axis]
    n = img.data.shape[ax]
    if columns <= 0:
        columns = int(np.ceil(np.sqrt(n)))
    if rows <= 0:
        rows = int(np.ceil(n / columns))
    from skimage.transform import resize
    h = max(int(round(img.height * scale)), 1)
    w = max(int(round(img.width * scale)), 1)
    other = [s for i, s in enumerate(img.data.shape[:3]) if i != ax]
    out_shape = (other[0], other[1], 1, rows * h + border * (rows - 1), columns * w + border * (columns - 1))
    out = np.zeros(out_shape, np.float32)
    moved = np.moveaxis(img.data, ax, 0)  # (n, A, B, Y, X)
    for k in range(min(n, rows * columns)):
        r, c = divmod(k, columns)
        tile = moved[k].astype(np.float32)
        if scale != 1.0:
            tile = resize(tile, tile.shape[:2] + (h, w), order=1, preserve_range=True, anti_aliasing=True)
        out[:, :, 0, r * (h + border):r * (h + border) + h, c * (w + border):c * (w + border) + w] = tile
    # restore the axes layout: out has (A, B, 1, Y, X) with A,B the two untouched axes in order
    full = np.zeros((1, 1, 1, out_shape[3], out_shape[4]), np.float32)
    shape5 = [1, 1, 1, out_shape[3], out_shape[4]]
    others = [i for i in range(3) if i != ax]
    shape5[others[0]] = other[0]
    shape5[others[1]] = other[1]
    full = out.reshape(shape5)
    res = img.copy(name=f"Montage of {img.name}", data=cast_to(full, img.data.dtype))
    _store_labels(res, None)  # the tiled planes no longer exist as planes
    if axis == "C":
        res.display.channels = res.display.channels[:1]
        res.display.mode = "single"
    res.display.z = res.display.t = res.display.c = 0
    res.sync_channels()
    cal = res.calibration.to_dict()
    cal["pixel_width"] /= scale; cal["pixel_height"] /= scale
    res.calibration = type(res.calibration)(**cal)
    return OpResult(new_images=[res])


@op("stacks.to_images", "Stack to Images", CAT, kind="image", scope_param=False, preview=False, params=[Param("axis", "choice", "Z", choices=["Z", "T", "C"])], description="Open every slice as a separate image.")
def stack_to_images(ctx, axis="Z"):
    img = ctx.img
    ax = {"Z": 1, "T": 0, "C": 2}[axis]
    n = img.data.shape[ax]
    if n > 200:
        raise ValueError("Refusing to open more than 200 images")
    out = []
    grid = _label_grid(img)
    for k in range(n):
        data = np.take(img.data, [k], axis=ax).copy()
        im = img.copy(name=f"{img.name}-{axis}{k + 1:04d}", data=data)
        _store_labels(im, np.take(grid, [k], axis=ax) if grid is not None else None)
        if axis == "C" and not img.rgb:
            im.display.channels = [type(img.display.channels[k])(**img.display.channels[k].to_dict())]
            im.display.c = 0
        im.sync_channels()
        out.append(im)
    return OpResult(new_images=out)


@op("stacks.images_to_stack", "Images to Stack", CAT, kind="image", requires_image=False, scope_param=False, preview=False,
    params=[Param("title", "string", "Stack"), Param("axis", "choice", "Z", choices=["Z", "T", "C"]), Param("name_filter", "string", ""), Param("close_sources", "bool", False)],
    description="Combine all open images of the same size into one stack.")
def images_to_stack(ctx, title="Stack", axis="Z", name_filter="", close_sources=False):
    s = ctx.session
    imgs = [im for im in s.images.values() if (not name_filter or name_filter in im.name)]
    if len(imgs) < 2:
        raise ValueError("Need at least two open images")
    ref = imgs[0]
    for im in imgs:
        if (im.height, im.width) != (ref.height, ref.width):
            raise ValueError(f"{im.name} has a different size")
    ax = {"Z": 1, "T": 0, "C": 2}[axis]
    # ImageJ's ImagesToStack promotes every image to the widest type (8 → 16 → 32 bit) without scaling; a plain
    # astype would wrap / truncate floats and 16-bit values into the first image's type
    out_dtype = np.result_type(*(im.data.dtype for im in imgs))  # uint8 < uint16 < float32: never narrower than any source, so astype is lossless
    data = np.concatenate([im.data.astype(out_dtype, copy=False) for im in imgs], axis=ax)
    res = ref.copy(name=title, data=data)
    _store_labels(res, _concat_label_grids([im for im in imgs], ax))
    # RGB survives only when every source is RGB and the sample axis is untouched (ImageJ's ImagesToStack keeps an
    # RGB stack); reset_display() then picks Red/Green/Blue instead of the Cyan/Magenta/Yellow composite defaults
    res.rgb = bool(all(im.rgb for im in imgs) and ax != 2 and data.shape[2] == 3 and data.dtype == np.uint8)
    res.reset_display()
    if close_sources:
        for im in imgs:
            s.close_image(im.id)
    return OpResult(new_images=[res])


@op("stacks.concatenate", "Concatenate", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("image1", "image", None), Param("image2", "image", None), Param("axis", "choice", "T", choices=["T", "Z", "C"]), Param("title", "string", "Concatenated")],
    description="Join two stacks along an axis.")
def concatenate(ctx, image1=None, image2=None, axis="T", title="Concatenated"):
    s = ctx.session
    a = s.get_image(image1) if image1 else ctx.img
    b = s.get_image(image2) if image2 else ctx.img
    ax = {"T": 0, "Z": 1, "C": 2}[axis]
    out_dtype = np.result_type(a.data.dtype, b.data.dtype)  # widest type wins (lossless widening), no wrap-around of the narrower image
    data = np.concatenate([a.data.astype(out_dtype, copy=False), b.data.astype(out_dtype, copy=False)], axis=ax)
    res = a.copy(name=title, data=data)
    _store_labels(res, _concat_label_grids([a, b], ax))
    res.reset_display(keep_luts=True)
    return OpResult(new_images=[res])


@op("stacks.reorder", "Re-order Hyperstack", "Image/Hyperstacks", kind="image", scope_param=False, preview=False,
    params=[Param("order", "choice", "Swap Z and T", choices=["Swap Z and T", "Swap Z and C", "Swap C and T"])], description="Swap two hyperstack dimensions.")
def reorder(ctx, order="Swap Z and T"):
    img = ctx.img
    perm = {"Swap Z and T": (1, 0, 2), "Swap Z and C": (0, 2, 1)}.get(order, (2, 1, 0))
    data = np.ascontiguousarray(img.data.transpose(perm + (3, 4)))
    grid = _label_grid(img)
    if grid is not None:
        _store_labels(img, grid.transpose(perm))  # in place: run_op installs `data` on img afterwards
    if order == "Swap Z and T":
        # the channel axis is untouched: every channel's LUT, display range, visibility and gamma stay valid (and an
        # RGB stack stays RGB) – Fiji's Hyperstack_rearranger copies imp.getLuts() in exactly this case. Returning no
        # display lets replace_data() → sync_channels() keep the ChannelDisplay list; the view follows the plane that
        # was on screen (old (t, z) now sits at (z, t); sync_channels() clamps it to the new shape).
        d = img.display
        d.t, d.z = d.z, d.t
        return OpResult(data=data, rgb=bool(img.rgb))
    # Z or T planes become channels: fresh default LUTs (as Fiji), and an RGB source's samples are scattered off C
    res = img.copy(name=img.name, data=data)
    res.rgb = False
    res.reset_display()
    return OpResult(data=data, rgb=False, display=res.display.to_dict())


@op("stacks.to_hyperstack", "Stack to Hyperstack", "Image/Hyperstacks", kind="image", scope_param=False, preview=False,
    params=[Param("channels", "int", 1, min=1), Param("slices", "int", 1, min=1), Param("frames", "int", 1, min=1), Param("order", "choice", "xyczt", choices=["xyczt", "xyzct", "xyctz", "xytcz", "xyztc", "xytzc"])],
    description="Re-interpret a plain stack as channels × slices × frames.")
def to_hyperstack(ctx, channels=1, slices=1, frames=1, order="xyczt"):
    img = ctx.img
    n = img.nt * img.nz * img.nc
    if channels * slices * frames != n:
        raise ValueError(f"c×z×t = {channels * slices * frames} does not match the {n} planes")
    flat = img.data.reshape(n, img.height, img.width)
    inner = order[2:]  # fastest-varying first, e.g. 'czt'
    dims = {"c": channels, "z": slices, "t": frames}
    shape = [dims[inner[2]], dims[inner[1]], dims[inner[0]]]
    arr = flat.reshape(shape + [img.height, img.width])
    letters = [inner[2], inner[1], inner[0]]
    perm = [letters.index("t"), letters.index("z"), letters.index("c")]
    data = np.transpose(arr, perm + [3, 4])
    res = img.copy(name=img.name, data=np.ascontiguousarray(data))
    # an RGB source stays RGB only when its three samples land on the channel axis again: channels == 3 and c the
    # fastest-varying dimension (xyczt / xyctz); every other order scatters R, G, B over z or t. The flag must also
    # reach the OpResult: run_op passes it to replace_data(), where a hard-coded False (or None) would undo it.
    res.rgb = bool(img.rgb and channels == 3 and inner[0] == "c")
    res.reset_display()
    grid = _label_grid(img)
    if grid is not None:  # the flat plane order is the ImageJ order, so the labels are re-shaped exactly like the planes
        _store_labels(img, np.transpose(grid.reshape(shape), perm))
    return OpResult(data=res.data, rgb=res.rgb, display=res.display.to_dict())


@op("stacks.z_profile", "Plot Z-axis Profile", CAT, kind="analysis", preview=False,
    params=[Param("axis", "choice", "Z", choices=["Z", "T"])], description="Mean intensity of the selection (or whole image) per slice / frame.")
def z_profile(ctx, axis="Z"):
    img = ctx.img
    d = img.display
    mask = ctx.mask
    ax = 1 if axis == "Z" else 0
    n = img.data.shape[ax]
    series = []
    chans = range(img.nc) if img.nc > 1 and d.mode == "composite" else [d.c]
    lazy = is_virtual(img.data)
    total = len(chans) * n
    for ci, c in enumerate(chans):
        vals = []
        for k in range(n):
            ctx.check_cancel()
            # one plane at a time; a virtual stack reads it from the file in a single streaming pass (no caching)
            p = get_plane(img.data, d.t, k, c) if axis == "Z" else get_plane(img.data, k, d.z, c)
            v = p[mask] if mask is not None else p
            vals.append(float(np.nanmean(v.astype(np.float32))))
            if lazy:
                ctx.progress((ci * n + k + 1) / total, f"Plane {ci * n + k + 1}/{total}")
        series.append({"name": img.display.channels[c].name or f"C{c + 1}", "values": vals, "lut": img.display.channels[c].lut})
    cal = img.calibration
    if axis == "Z":
        x = [k * cal.pixel_depth for k in range(n)]
        xl = f"Z ({cal.unit})" if cal.scaled else "Slice"
    else:
        x = [k * (cal.frame_interval or 1) for k in range(n)]
        xl = f"Time ({cal.time_unit})" if cal.frame_interval else "Frame"
    plot = {"title": f"{axis} profile of {img.name}", "x": x, "series": series, "x_label": xl, "y_label": "Mean intensity"}
    return OpResult(plot=plot, results=[{"Label": f"{img.name}:{s['name']}", **{f"{axis}{k + 1}": v for k, v in enumerate(s["values"])}} for s in series] if n <= 64 else None)


def _interpolate_z(img: LumenImage, data: np.ndarray) -> np.ndarray:
    cal = img.calibration
    if cal.pixel_depth != cal.pixel_width and img.nz > 1 and cal.pixel_width > 0:
        from scipy.ndimage import zoom
        f = cal.pixel_depth / cal.pixel_width
        return cast_to(zoom(data.astype(np.float32), (1, f, 1, 1, 1), order=1), img.data.dtype)
    return data


def _z_row_spacing(img: LumenImage, data: np.ndarray, interpolate_z: bool) -> float:
    """Calibrated spacing of the output rows of a reslice (the original Z axis): pixel_width when Z was resampled to
    isotropic voxels, otherwise one row = one slice = pixel_depth (ImageJ Slicer: pixelHeight = pixelDepth / zSpacing,
    zSpacing = 1 with 'avoid interpolation')."""
    cal = img.calibration
    return cal.pixel_width if (interpolate_z and data is not img.data) else cal.pixel_depth


def _line_samples(roi: dict, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """(xs, ys) sample positions along a line / polyline selection, `spacing` pixels apart."""
    from .. import roi as roimod
    pts = roimod.line_points(roi)
    xs_all, ys_all = [], []
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        n = max(int(round(np.hypot(xb - xa, yb - ya) / max(spacing, 1e-6))), 1)
        xs_all.append(np.linspace(xa, xb, n, endpoint=False))
        ys_all.append(np.linspace(ya, yb, n, endpoint=False))
    xs = np.concatenate(xs_all + [np.array([pts[-1][0]])])
    ys = np.concatenate(ys_all + [np.array([pts[-1][1]])])
    return xs, ys


def reslice_along_line(data: np.ndarray, xs: np.ndarray, ys: np.ndarray, dtype) -> np.ndarray:
    """(T, Z, C, Y, X) → (T, 1, C, nz, len(xs)): one slice through the stack along the sampled path (bilinear)."""
    from scipy.ndimage import map_coordinates
    nt, nz, nc = data.shape[:3]
    out = np.zeros((nt, 1, nc, nz, xs.shape[0]), np.float32)
    coords = [ys - 0.5, xs - 0.5]
    for t in range(nt):
        for c in range(nc):
            for z in range(nz):
                out[t, 0, c, z] = map_coordinates(data[t, z, c].astype(np.float32), coords, order=1, mode="nearest")
    return cast_to(out, dtype)


@op("stacks.reslice", "Reslice", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("direction", "choice", "Top (XZ)", choices=["Top (XZ)", "Left (YZ)"], help="Ignored when a line / polyline selection is active: the stack is then resliced along the line"),
            Param("interpolate_z", "bool", True, help="Resample Z to isotropic voxels"),
            Param("output_spacing", "float", 1.0, min=0.1, max=100, step=0.5, help="Line mode: distance (px) between the parallel slices"),
            Param("slice_count", "int", 1, min=1, max=1000, help="Line mode: number of parallel slices, shifted perpendicular to the straight line"),
            Param("rotate", "bool", False, help="Line mode: rotate the output 90° (depth along X)")],
    description="Build an orthogonal view stack (Top = XZ, Left = YZ), or – with a line / polyline selection – the slice(s) through the stack along the line.",
    keywords=["orthogonal", "xz", "yz", "line reslice"])
def reslice(ctx, direction="Top (XZ)", interpolate_z=True, output_spacing=1.0, slice_count=1, rotate=False):
    img = ctx.img
    cal = img.calibration
    from .. import roi as roimod
    if ctx.roi is not None and roimod.is_line(ctx.roi):
        return reslice_line(ctx, output_spacing=output_spacing, slice_count=slice_count, rotate=rotate, interpolate_z=interpolate_z)
    data = img.data
    if interpolate_z:
        data = _interpolate_z(img, data)
    row = _z_row_spacing(img, data, interpolate_z)
    if direction.startswith("Top"):
        out = np.ascontiguousarray(data.transpose(0, 3, 2, 1, 4))  # Y becomes slices, Z becomes rows
        new_cal = {"pixel_width": cal.pixel_width, "pixel_height": row, "pixel_depth": cal.pixel_height}
    else:
        out = np.ascontiguousarray(data.transpose(0, 4, 2, 1, 3))  # X becomes slices, Z becomes rows, Y becomes columns
        new_cal = {"pixel_width": cal.pixel_height, "pixel_height": row, "pixel_depth": cal.pixel_width}
    res = img.copy(name=f"Reslice of {img.name}", data=out)
    res.calibration = type(cal)(**{**cal.to_dict(), **new_cal})
    _store_labels(res, None)
    res.display.z = 0
    res.sync_channels()
    return OpResult(new_images=[res])


@op("stacks.reslice_line", "Reslice (line)", CAT, kind="image", scope_param=False, preview=False, requires_roi="line",
    params=[Param("output_spacing", "float", 1.0, min=0.1, max=100, step=0.5, help="Distance (px) between parallel slices"),
            Param("slice_count", "int", 1, min=1, max=1000, help="Number of parallel slices (straight lines only), shifted perpendicular to the line"),
            Param("rotate", "bool", False, help="Rotate the output 90° (depth along X)"), Param("interpolate_z", "bool", True, help="Resample Z to isotropic voxels")],
    description="Slice through the stack along the line / polyline selection; several parallel slices are produced by shifting a straight line perpendicular to itself.",
    keywords=["reslice", "line", "xz"])
def reslice_line(ctx, output_spacing=1.0, slice_count=1, rotate=False, interpolate_z=True):
    img = ctx.img
    from .. import roi as roimod
    roi = ctx.roi
    if roi is None or not roimod.is_line(roi):
        raise ValueError("Reslice (line) needs a line or polyline selection")
    cal = img.calibration
    data = _interpolate_z(img, img.data) if interpolate_z else img.data
    xs, ys = _line_samples(roi)
    slices = []
    if roi["type"] == "line" and slice_count > 1:
        dx, dy = roi["x2"] - roi["x1"], roi["y2"] - roi["y1"]
        L = float(np.hypot(dx, dy)) or 1.0
        nx, ny = -dy / L, dx / L
        for k in range(slice_count):
            off = k * output_spacing
            slices.append(reslice_along_line(data, xs + off * nx, ys + off * ny, img.data.dtype))
    else:
        slices.append(reslice_along_line(data, xs, ys, img.data.dtype))
    out = np.concatenate(slices, axis=1)  # (T, n, C, nz, L)
    pw, ph = cal.pixel_width, _z_row_spacing(img, data, interpolate_z)
    if rotate:
        out = np.ascontiguousarray(np.swapaxes(out, 3, 4)[:, :, :, ::-1, :])
        pw, ph = ph, pw  # depth now runs along X
    res = img.copy(name=f"Reslice of {img.name}", data=np.ascontiguousarray(out))
    res.calibration = type(cal)(**{**cal.to_dict(), "pixel_width": pw, "pixel_height": ph, "pixel_depth": output_spacing * cal.pixel_width})
    _store_labels(res, None)
    res.display.z = 0
    res.overlay = []
    res.sync_channels()
    return OpResult(new_images=[res], message=f"{out.shape[1]} slice(s) of {out.shape[4]}×{out.shape[3]} px")


@op("stacks.radial_reslice", "Radial Reslice", CAT, kind="image", scope_param=False, preview=False,
    params=[Param("n_angles", "int", 36, min=1, max=3600, help="Number of reslices over the rotation"), Param("total_rotation", "float", 360.0, min=1, max=360, step=1),
            Param("length", "float", 0.0, min=0, step=1, help="Length of the rotating line (px); 0 = from the centre to the nearest image edge (or the line length when a line is selected)"),
            Param("interpolate_z", "bool", True, help="Resample Z to isotropic voxels")],
    description="Rotate a line around a point selection (or the image centre; a line selection rotates around its first end) and reslice the stack along it at every angle → stack of radial slices (Fiji's Radial Reslice).",
    keywords=["radial", "reslice", "rotate", "cylindrical"])
def radial_reslice(ctx, n_angles=36, total_rotation=360.0, length=0.0, interpolate_z=True):
    img = ctx.img
    from .. import roi as roimod
    roi = ctx.roi
    angle0 = 0.0
    if roi is not None and roi.get("type") == "point" and roi.get("points"):
        cx, cy = roi["points"][0]
    elif roi is not None and roimod.is_line(roi):
        pts = roimod.line_points(roi)
        (cx, cy), (ex, ey) = pts[0], pts[-1]
        if length <= 0:
            length = float(np.hypot(ex - cx, ey - cy))
        angle0 = float(np.degrees(np.arctan2(ey - cy, ex - cx)))
    else:
        cx, cy = img.width / 2.0, img.height / 2.0
    if length <= 0:
        length = float(min(cx, cy, img.width - cx, img.height - cy))
    n = max(int(round(length)), 1)
    data = _interpolate_z(img, img.data) if interpolate_z else img.data
    # n = round(length) samples length/n ≈ 1 px apart, end point excluded (ImageJ Slicer.getLine, like _line_samples):
    # column k then lies k px from the centre, which is what the output calibration (one pixel_width per column) claims
    r = np.linspace(0.0, length, n, endpoint=False)
    slices = []
    for k in range(n_angles):
        ang = np.radians(angle0 + k * total_rotation / n_angles)
        xs, ys = cx + r * np.cos(ang), cy + r * np.sin(ang)
        slices.append(reslice_along_line(data, xs, ys, img.data.dtype))
    out = np.ascontiguousarray(np.concatenate(slices, axis=1))
    res = img.copy(name=f"Radial Reslice of {img.name}", data=out)
    cal = img.calibration
    res.calibration = type(cal)(**{**cal.to_dict(), "pixel_height": _z_row_spacing(img, data, interpolate_z), "pixel_depth": cal.pixel_width})
    res.metadata["slice_labels"] = [f"{angle0 + k * total_rotation / n_angles:.1f}°" for k in range(n_angles)]
    res.display.z = 0
    res.overlay = []
    res.sync_channels()
    return OpResult(new_images=[res], message=f"{n_angles} radial slices around ({cx:.1f}, {cy:.1f})")


# ------------------------------------------------------------------ slice labels
def plane_index(img: LumenImage, t: int, z: int, c: int) -> int:
    """ImageJ stack index (0-based) of a plane: channels vary fastest, then slices, then frames."""
    return (t * img.nz + z) * img.nc + c


def slice_labels(img: LumenImage) -> list[str]:
    """Per-plane labels (length nt·nz·nc, '' = none); a per-Z list is expanded over channels and frames."""
    n = img.nt * img.nz * img.nc
    labels = img.metadata.get("slice_labels")
    if not isinstance(labels, (list, tuple)):
        return [""] * n
    labels = ["" if v is None else str(v) for v in labels]
    if len(labels) == n:
        return labels
    if len(labels) == img.nz and n != img.nz:
        return [labels[z] for t in range(img.nt) for z in range(img.nz) for c in range(img.nc)]
    return (labels + [""] * n)[:n]


def set_slice_label(img: LumenImage, label: str, t: int, z: int, c: int) -> None:
    labels = slice_labels(img)
    labels[plane_index(img, t, z, c)] = label
    img.metadata["slice_labels"] = labels


def _label_grid(img: LumenImage) -> np.ndarray | None:
    """The per-plane labels as an (nt, nz, nc) object array (ravel() gives the ImageJ plane order), or None when the
    image has no labels.  Stack editors index / insert / flip / transpose this grid exactly like the pixel data so
    that every label stays with its plane (ImageJ ImageStack.deleteSlice / addSlice, SubstackMaker, StackReverser)."""
    labels = slice_labels(img)
    if not any(labels):
        return None
    return np.array(labels, dtype=object).reshape(img.nt, img.nz, img.nc)


def _store_labels(target: LumenImage, grid: np.ndarray | None) -> None:
    """Attach the labels of `grid` (any shape whose ravel() is the plane order of `target`) to `target`; None drops the
    (stale, copied) label list, e.g. for projections whose planes no longer correspond to source planes."""
    if grid is None:
        target.metadata.pop("slice_labels", None)
    else:
        target.metadata["slice_labels"] = [str(v) for v in np.asarray(grid, dtype=object).ravel()]


def _concat_label_grids(imgs: list[LumenImage], ax: int) -> np.ndarray | None:
    """Label grid of images stacked along axis `ax` (0 = T, 1 = Z, 2 = C); None when no source is labelled."""
    grids = [_label_grid(im) for im in imgs]
    if not any(g is not None for g in grids):
        return None
    return np.concatenate([g if g is not None else np.full((im.nt, im.nz, im.nc), "", dtype=object) for g, im in zip(grids, imgs)], axis=ax)


@op("stacks.set_slice_label", "Set Slice Label", CAT, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("label", "string", "")], description="Set the label of the current plane (stored per plane in the image metadata; shown in Slice Labels to Results and used by Montage).",
    keywords=["slice label", "set label"], undo_label="Set slice label")
def set_slice_label_op(ctx, label=""):
    img = ctx.img
    d = img.display
    set_slice_label(img, str(label), d.t, d.z, d.c)
    img.rev += 1  # metadata changed – make the client reload
    return OpResult(message=f"Slice {plane_index(img, d.t, d.z, d.c) + 1}: {label!r}")


@op("stacks.remove_slice_labels", "Remove Slice Labels", CAT, kind="image", scope_param=False, preview=False, roi_aware=False,
    description="Delete every per-plane label of the image.", keywords=["slice label"], undo_label="Remove slice labels")
def remove_slice_labels(ctx):
    img = ctx.img
    had = any(slice_labels(img))  # same notion of "labelled" as _label_grid / Slice Labels to Results
    img.metadata.pop("slice_labels", None)  # a stale / all-empty list is dropped too, but that is no change worth an undo entry
    if had:
        img.rev += 1  # metadata changed – make the client reload and let run_op record the undo entry (and clear redo)
    return OpResult(message="Slice labels removed" if had else "The image has no slice labels")


@op("stacks.slice_labels_to_results", "Slice Labels to Results", CAT, kind="analysis", preview=False, roi_aware=False,
    description="List the label of every plane (index, channel, slice, frame, label) in the Results table.", keywords=["slice label"])
def slice_labels_to_results(ctx):
    img = ctx.img
    labels = slice_labels(img)
    rows = [{"Label": img.name, "Index": plane_index(img, t, z, c) + 1, "Ch": c + 1, "Slice": z + 1, "Frame": t + 1, "Slice Label": labels[plane_index(img, t, z, c)]}
            for (t, z, c) in img.iter_planes()]
    return OpResult(results=rows, message=f"{sum(1 for v in labels if v)} of {len(labels)} planes labelled")


# ------------------------------------------------------------------ montage ⇄ stack, hyperstack tools
@op("stacks.montage_to_stack", "Montage to Stack", CAT, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("columns", "int", 2, min=1, max=1000), Param("rows", "int", 2, min=1, max=1000), Param("border", "int", 0, min=0, max=200, help="Width of the gap between tiles")],
    description="Cut the current plane into columns × rows tiles (optionally separated by a border) and stack them as Z slices – the inverse of Make Montage.",
    keywords=["montage", "tiles", "grid"])
def montage_to_stack(ctx, columns=2, rows=2, border=0):
    img = ctx.img
    d = img.display
    tw = (img.width - border * (columns - 1)) // columns
    th = (img.height - border * (rows - 1)) // rows
    if tw < 1 or th < 1:
        raise ValueError(f"{columns}×{rows} tiles with a border of {border} px do not fit into {img.width}×{img.height}")
    plane = img.data[d.t, d.z]  # (C, Y, X)
    tiles = []
    for r in range(rows):
        for c in range(columns):
            y0, x0 = r * (th + border), c * (tw + border)
            tiles.append(plane[:, y0:y0 + th, x0:x0 + tw])
    data = np.stack(tiles, axis=0)[None]  # (1, n, C, th, tw)
    res = img.copy(name=f"Stack of {img.name}", data=np.ascontiguousarray(data))
    _store_labels(res, None)
    res.rgb = img.rgb
    res.display.t = res.display.z = 0
    res.overlay = []
    res.sync_channels()
    return OpResult(new_images=[res], message=f"{rows * columns} tiles of {tw}×{th} px")


@op("stacks.hyperstack_to_stack", "Hyperstack to Stack", "Image/Hyperstacks", kind="image", scope_param=False, preview=False, roi_aware=False, supports_rgb=False,
    description="Flatten channels, slices and frames into one plain Z stack (ImageJ order: channels vary fastest, then slices, then frames); every plane gets a 'c:… z:… t:…' slice label.",
    keywords=["flatten", "hyperstack"])
def hyperstack_to_stack(ctx):
    img = ctx.img
    n = img.nt * img.nz * img.nc
    data = img.data.reshape(1, n, 1, img.height, img.width).copy()
    labels = slice_labels(img)
    res = img.copy(name=img.name, data=data)
    res.rgb = False
    res.display.channels = res.display.channels[:1]
    res.display.mode = "single"
    res.display.t = res.display.z = res.display.c = 0
    res.sync_channels()
    res.metadata["slice_labels"] = [(labels[plane_index(img, t, z, c)] + " " if labels[plane_index(img, t, z, c)] else "") + f"c:{c + 1}/{img.nc} z:{z + 1}/{img.nz} t:{t + 1}/{img.nt}"
                                    for (t, z, c) in img.iter_planes()]
    res.metadata["hyperstack_dims"] = {"c": img.nc, "z": img.nz, "t": img.nt}
    return OpResult(new_images=[res], message=f"{n} planes")


@op("stacks.reduce_dimensionality", "Reduce Dimensionality", "Image/Hyperstacks", kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("keep_channels", "bool", True, label="Channels"), Param("keep_slices", "bool", True, label="Slices"), Param("keep_frames", "bool", True, label="Frames"),
            Param("keep_source", "bool", True, help="Keep the original image open")],
    description="Drop hyperstack dimensions: an unticked axis is reduced to its current position (ImageJ's Reduce Dimensionality).",
    keywords=["hyperstack", "reduce", "extract"])
def reduce_dimensionality(ctx, keep_channels=True, keep_slices=True, keep_frames=True, keep_source=True):
    img = ctx.img
    d = img.display
    data = img.data
    if not keep_frames:
        data = data[d.t:d.t + 1]
    if not keep_slices:
        data = data[:, d.z:d.z + 1]
    if not keep_channels and not img.rgb:
        data = data[:, :, d.c:d.c + 1]
    if data.shape == img.data.shape:
        raise ValueError("Untick at least one dimension to reduce")
    res = img.copy(name=f"{img.name}-1", data=np.ascontiguousarray(data.copy()))
    res.rgb = img.rgb and data.shape[2] == 3
    if not keep_channels and not img.rgb:
        res.display.channels = [type(d.channels[d.c])(**d.channels[d.c].to_dict())]
        res.display.mode = "single"
    res.display.t = 0 if not keep_frames else d.t
    res.display.z = 0 if not keep_slices else d.z
    res.display.c = 0 if not keep_channels else d.c
    res.sync_channels()
    labels = slice_labels(img)
    if any(labels):
        ts = range(img.nt) if keep_frames else [d.t]
        zs = range(img.nz) if keep_slices else [d.z]
        cs = range(img.nc) if (keep_channels or img.rgb) else [d.c]
        res.metadata["slice_labels"] = [labels[plane_index(img, t, z, c)] for t in ts for z in zs for c in cs]
    out = OpResult(new_images=[res])
    if not keep_source:
        out.close_image = True
    return out


@op("stacks.make_subset", "Make Subset", "Image/Hyperstacks", kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("channels", "string", "", help="e.g. 1-2 or 1,3 (empty = all)"), Param("slices", "string", "", help="e.g. 1-10-2 (step), empty = all"), Param("frames", "string", "", help="empty = all")],
    description="Extract a sub-hyperstack from ranges of channels, slices and frames applied together (ImageJ's Make Subset).",
    keywords=["hyperstack", "subset", "substack"])
def make_subset(ctx, channels="", slices="", frames=""):
    img = ctx.img
    cs = _parse_range(channels, img.nc) if str(channels).strip() and not img.rgb else list(range(img.nc))
    zs = _parse_range(slices, img.nz) if str(slices).strip() else list(range(img.nz))
    ts = _parse_range(frames, img.nt) if str(frames).strip() else list(range(img.nt))
    data = img.data[np.ix_(ts, zs, cs)].copy()
    res = img.copy(name=f"{img.name}-1", data=np.ascontiguousarray(data))
    res.rgb = img.rgb and data.shape[2] == 3
    if not img.rgb:
        res.display.channels = [type(img.display.channels[i])(**img.display.channels[i].to_dict()) for i in cs]
        if len(cs) == 1:
            res.display.mode = "single"
    res.display.t = res.display.z = res.display.c = 0
    res.sync_channels()
    labels = slice_labels(img)
    if any(labels):
        res.metadata["slice_labels"] = [labels[plane_index(img, t, z, c)] for t in ts for z in zs for c in cs]
    return OpResult(new_images=[res], message=f"{len(cs)} channel(s) × {len(zs)} slice(s) × {len(ts)} frame(s)")


def _ij_hist_median(hist: np.ndarray, count: int) -> float:
    """``ImageStatistics.calculateMedian`` of an integer histogram: the first value whose cumulative count exceeds half
    the pixels (ImageJ never averages the two middle values)."""
    if count <= 0 or hist.size == 0:
        return float("nan")
    half = count / 2.0
    cum = np.cumsum(hist)
    idx = np.nonzero(cum > half)[0]
    return float(idx[0]) if idx.size else float(hist.size - 1)


@op("stacks.statistics", "Stack Statistics", CAT, kind="analysis", preview=False,
    description="Voxel count, volume, % of the stack (or selection) volume, mean, SD, min, max, mode and median over the whole stack (every slice and frame, inside the selection) per channel → table 'Stack Statistics' (ImageJ's Image ▸ Stacks ▸ Statistics columns).",
    keywords=["stack", "statistics", "voxels", "histogram"])
def stack_statistics(ctx):
    img = ctx.img
    mask = ctx.mask
    rows = []
    results = []
    cal = img.calibration
    vol = cal.pixel_width * cal.pixel_height * cal.pixel_depth
    lazy = is_virtual(img.data)
    # ImageJ's %Volume: voxels / (selection pixels × planes) · 100 – the whole image or the rectangle bounds without a
    # selection, the mask's pixel count for other selections
    planes = img.data.shape[0] * img.data.shape[1]
    total = float((int(mask.sum()) if mask is not None else img.width * img.height) * planes)
    for c in range(img.nc):
        if lazy:
            # a virtual stack is scanned plane by plane: one pass with an exact histogram for 8 / 16-bit data, two passes
            # (min / max, then a 65536-bin histogram for the mode and median) for float data – never read into memory
            st = stream_statistics(img.data, c, mask, progress=lambda f, m, c=c: ctx.progress((c + f) / img.nc, m), check_cancel=ctx.check_cancel)
            n, mean, sd, mn, mx, mode, med = st["n"], st["mean"], st["sd"], st["min"], st["max"], st["mode"], st["median"]
        else:
            vals = img.data[:, :, c]
            if mask is not None:
                vals = vals[:, :, mask]
            vals = vals.astype(np.float64).ravel()
            vals = vals[np.isfinite(vals)]
            n = int(vals.size)
            if n == 0:
                mean = sd = mn = mx = mode = med = float("nan")
            else:
                mean, sd = float(vals.mean()), float(vals.std(ddof=1)) if n > 1 else 0.0
                mn, mx = float(vals.min()), float(vals.max())
                if img.data.dtype in (np.uint8, np.uint16):
                    hist = np.bincount(vals.astype(np.int64))
                    mode = float(np.argmax(hist))
                    med = _ij_hist_median(hist, n)   # ImageJ: the histogram median
                else:
                    med = float(np.median(vals))
                    h, edges = np.histogram(vals, bins=256)
                    k = int(np.argmax(h))
                    mode = float((edges[k] + edges[k + 1]) / 2)
        name = img.display.channels[c].name if c < len(img.display.channels) else f"C{c + 1}"
        label = f"{img.name}:{name}" if img.nc > 1 else img.name
        pct = n * 100.0 / total if total > 0 else float("nan")
        results.append({"Label": label, "Voxels": n, "Volume": n * vol, "%Volume": pct, "Mean": mean, "StdDev": sd, "Min": mn, "Max": mx, "Mode": mode, "Median": med})
        rows.append([label, n, n * vol] + [v if np.isfinite(v) else None for v in (mean, sd, mn, mx, mode, med)] + [pct if np.isfinite(pct) else None])
    unit = cal.unit if cal.scaled else "pixel"
    table = {"name": "Stack Statistics", "columns": ["Label", "Voxels", f"Volume ({unit}³)", "Mean", "StdDev", "Min", "Max", "Mode", "Median", "%Volume"], "rows": rows}
    first_mean = rows[0][3]  # None when channel 1 has no finite voxel (empty selection, all-NaN data): report it, do not crash
    mean_txt = "NaN" if first_mean is None else f"{first_mean:.4g}"
    return OpResult(table=table, results=results, message=f"{rows[0][1]} voxels, mean {mean_txt}")
