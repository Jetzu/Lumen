"""Image ▸ Adjust ▸ Auto Crop, Image ⇄ Results, image overlays, the frequency-domain tool set
(Inverse FFT, Swap Quadrants, FD Math, Custom Filter, FFT Options), Retinex and Process ▸ Math ▸ Macro."""
from __future__ import annotations

import ast
import re
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from .. import roi as roimod
from ..image import LumenImage
from ..registry import Param, OpResult, op, cast_to
from ._util import as_float, back_to
from ._imagej import fht2d, power_spectrum_8bit_from_fht, round_half_up
from .enhance import FFT_OPTION_DEFAULTS, fft_store, fft_options, forward_fft, make_fft_image, power_spectrum_8bit, stored_fft

ADJ = "Image/Adjust"
FFT = "Process/FFT"


# ====================================================================== Auto Crop
def border_mode(plane: np.ndarray) -> np.ndarray:
    """Most frequent value of the 1-pixel border (per channel for (C, Y, X) input)."""
    chans = plane if plane.ndim == 3 else plane[None]
    out = []
    for p in chans:
        border = np.concatenate([p[0], p[-1], p[:, 0], p[:, -1]]).astype(np.float64)
        border = border[np.isfinite(border)]
        if border.size == 0:
            out.append(0.0)
            continue
        if p.dtype in (np.uint8, np.uint16):
            out.append(float(np.argmax(np.bincount(border.astype(np.int64)))))
        else:
            vals, counts = np.unique(np.round(border, 6), return_counts=True)
            out.append(float(vals[int(np.argmax(counts))]))
    return np.asarray(out)


def auto_crop_rect(planes: np.ndarray, tolerance: float = 0.0) -> tuple[int, int, int, int]:
    """(x, y, w, h) bounding box of the pixels that differ from the border background in any channel of (C, Y, X)."""
    bg = border_mode(planes)
    diff = np.zeros(planes.shape[1:], bool)
    for p, b in zip(planes, bg):
        f = p.astype(np.float64)
        diff |= np.abs(np.nan_to_num(f, nan=b) - b) > tolerance
    ys, xs = np.nonzero(diff)
    if ys.size == 0:
        return 0, 0, planes.shape[2], planes.shape[1]
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


@op("image.auto_crop", "Auto Crop", ADJ, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("tolerance", "float", 0.0, min=0, step=1, help="Pixels within ± tolerance of the background value still count as background"),
            Param("all_planes", "bool", True, help="Find the bounding box over every slice / frame / channel (otherwise over the current plane only)")],
    description="Crop the image to the bounding box of everything that differs from the background (the most frequent value along the image border).",
    keywords=["crop", "trim", "bounding box", "background"], undo_label="Auto crop")
def auto_crop(ctx, tolerance=0.0, all_planes=True):
    img = ctx.img
    x, y, w, h = _auto_rect(img, tolerance, all_planes)
    if (w, h) == (img.width, img.height):
        return OpResult(message="Nothing to crop: the whole image differs from the background")
    data = np.ascontiguousarray(img.data[:, :, :, y:y + h, x:x + w])
    overlay = [roimod.translate(o, -x, -y) if o.get("type") in roimod.AREA_TYPES | roimod.LINE_TYPES | {"point"} else o for o in img.overlay]
    return OpResult(data=data, rgb=img.rgb, overlay=overlay, message=f"Cropped to {w}×{h} at ({x}, {y})")


def _auto_rect(img: LumenImage, tolerance: float, all_planes: bool) -> tuple[int, int, int, int]:
    d = img.display
    if all_planes:
        rects = [auto_crop_rect(img.data[t, z], tolerance) for t in range(img.nt) for z in range(img.nz)]
        x0 = min(r[0] for r in rects); y0 = min(r[1] for r in rects)
        x1 = max(r[0] + r[2] for r in rects); y1 = max(r[1] + r[3] for r in rects)
        return x0, y0, x1 - x0, y1 - y0
    planes = img.data[d.t, d.z] if img.rgb else img.data[d.t, d.z, d.c:d.c + 1]
    return auto_crop_rect(planes, tolerance)


@op("image.auto_crop_selection", "Auto Crop (selection)", ADJ, kind="analysis", preview=False, roi_aware=False,
    params=[Param("tolerance", "float", 0.0, min=0, step=1), Param("all_planes", "bool", True)],
    description="Select (instead of cropping to) the bounding rectangle of everything that differs from the border background.",
    keywords=["crop", "bounding box", "select"])
def auto_crop_selection(ctx, tolerance=0.0, all_planes=True):
    img = ctx.img
    x, y, w, h = _auto_rect(img, tolerance, all_planes)
    return OpResult(roi={"type": "rect", "x": x, "y": y, "w": w, "h": h, "name": "Auto crop"}, message=f"{w}×{h} at ({x}, {y})")


# ====================================================================== Image ⇄ Results
@op("image.to_results", "Image to Results", "Image/Transform", kind="analysis", preview=False, roi_aware=True,
    params=[Param("replace", "bool", True, help="Clear the Results table first")],
    description="Write the pixel values of the current plane (or of the selection's bounding box) into the Results table: one row per image row, columns X0, X1, …",
    keywords=["pixels", "table", "results", "export"])
def image_to_results(ctx, replace=True):
    img = ctx.img
    plane = img.current_plane()
    x0 = y0 = 0
    if ctx.roi is not None and roimod.is_area(ctx.roi):
        bx, by, bw, bh = roimod.bounding_box(ctx.roi)
        x0, y0 = max(int(bx), 0), max(int(by), 0)
        plane = plane[y0:min(int(by + bh), img.height), x0:min(int(bx + bw), img.width)]
    if plane.shape[1] > 10000:
        raise ValueError("Image to Results is limited to 10 000 columns – select a smaller region")
    cols = [f"X{x0 + x}" for x in range(plane.shape[1])]
    values = plane.astype(np.float64).tolist() if plane.dtype == np.float32 else plane.tolist()   # Python floats / ints in C
    rows = [dict(zip(cols, r)) for r in values]
    return OpResult(results=rows, results_replace=replace, message=f"{plane.shape[0]} rows × {plane.shape[1]} columns")


@op("image.from_results", "Results to Image", "Image/Transform", kind="image", requires_image=False, scope_param=False, preview=False, roi_aware=False,
    params=[Param("title", "string", "Results Table"), Param("exclude_label", "bool", True, help="Skip non-numeric columns such as Label")],
    description="Create a 32-bit image from the numeric columns of the Results table (one row per Results row, one column per numeric column).",
    keywords=["table", "results", "import", "matrix"])
def results_to_image(ctx, title="Results Table", exclude_label=True):
    s = ctx.session
    tbl = s.results
    if not tbl.rows:
        raise ValueError("The Results table is empty")
    cols = []
    for c in tbl.columns:
        vals = [r.get(c) for r in tbl.rows]
        numeric = all(v is None or isinstance(v, (int, float, np.integer, np.floating)) for v in vals)
        if numeric or not exclude_label:
            cols.append(c)
    if not cols:
        raise ValueError("The Results table has no numeric columns")
    def _num(v):
        try:
            return float(v) if v is not None else np.nan
        except (TypeError, ValueError):
            return np.nan
    arr = np.array([[_num(r.get(c)) for c in cols] for r in tbl.rows], np.float64).reshape(len(tbl.rows), len(cols)).astype(np.float32)
    res = LumenImage(arr, name=title)
    res.metadata["columns"] = ",".join(cols)
    return OpResult(new_images=[res], message=f"{arr.shape[0]}×{arr.shape[1]} ({', '.join(cols[:6])}{'…' if len(cols) > 6 else ''})")


# ====================================================================== image overlays
@op("overlay.add_image", "Add Image to Overlay", "Image/Overlay", kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("image", "image", None, help="The image to show on top (rendered with its own display settings)"), Param("x", "float", 0.0, step=1), Param("y", "float", 0.0, step=1),
            Param("opacity", "float", 100.0, min=0, max=100, step=5, label="Opacity (%)"), Param("scale", "float", 1.0, min=0.01, max=100, step=0.1)],
    description="Add another open image as a non-destructive overlay item (type 'image') at x, y with the given opacity and scale (Image ▸ Overlay ▸ Add Image).",
    keywords=["overlay", "inset", "picture in picture"], undo_label="Add image overlay")
def overlay_add_image(ctx, image=None, x=0.0, y=0.0, opacity=100.0, scale=1.0):
    img = ctx.img
    other = ctx.session.get_image(image) if image else None
    if other is None:
        others = [im for im in ctx.session.images.values() if im.id != img.id]
        if not others:
            raise ValueError("Open the image to overlay first")
        other = others[-1]
    if other.id == img.id:
        raise ValueError("Choose a different image to overlay")
    item = {"type": "image", "id": roimod.new_id(), "image_id": other.id, "x": float(x), "y": float(y), "opacity": float(opacity) / 100.0,
            "scale": float(scale), "name": other.name}
    return OpResult(overlay=[*img.overlay, item], message=f"{other.name} added as an image overlay at ({x:g}, {y:g})")


def blend_image_item(rgb: np.ndarray, item: dict[str, Any], session) -> np.ndarray:
    """Alpha-blend the rendered image referenced by an 'image' overlay item into an (H, W, 3) uint8 array."""
    from ..render import render_rgb
    other = session.images.get(item.get("image_id"))
    if other is None:
        return rgb
    src = render_rgb(other).astype(np.float32)
    scale = float(item.get("scale", 1.0) or 1.0)
    if scale != 1.0:
        src = ndi.zoom(src, (scale, scale, 1), order=1)
    alpha = float(np.clip(item.get("opacity", 1.0), 0.0, 1.0))
    H, W = rgb.shape[:2]
    x0, y0 = int(round(float(item.get("x", 0)))), int(round(float(item.get("y", 0))))
    sh, sw = src.shape[:2]
    ya, yb = max(y0, 0), min(y0 + sh, H)
    xa, xb = max(x0, 0), min(x0 + sw, W)
    if yb <= ya or xb <= xa:
        return rgb
    sub = src[ya - y0:yb - y0, xa - x0:xb - x0]
    dst = rgb[ya:yb, xa:xb].astype(np.float32)
    rgb[ya:yb, xa:xb] = np.clip(dst * (1 - alpha) + sub * alpha, 0, 255).astype(np.uint8)
    return rgb


@op("overlay.burn_image", "Burn Image Overlay", "Image/Overlay", kind="image", preview=False, roi_aware=False, scope_param=False,
    params=[Param("image", "image", None, help="Blend this image (leave empty to burn every 'image' overlay item)"), Param("x", "float", 0.0, step=1), Param("y", "float", 0.0, step=1),
            Param("opacity", "float", 100.0, min=0, max=100, step=5, label="Opacity (%)"), Param("scale", "float", 1.0, min=0.01, max=100, step=0.1)],
    description="Blend an image (or every 'image' overlay item) into an RGB copy of the current plane at x, y with the given opacity → new RGB image.",
    keywords=["overlay", "flatten", "blend", "composite"])
def overlay_burn_image(ctx, image=None, x=0.0, y=0.0, opacity=100.0, scale=1.0):
    from ..render import render_rgb
    img = ctx.img
    rgb = render_rgb(img).copy()
    items = [{"image_id": ctx.session.get_image(image).id, "x": x, "y": y, "opacity": opacity / 100.0, "scale": scale}] if image else [o for o in img.overlay if o.get("type") == "image"]
    if not items:
        raise ValueError("No image overlay to burn in – add one with Overlay ▸ Add Image or choose an image")
    for it in items:
        rgb = blend_image_item(rgb, it, ctx.session)
    res = LumenImage(rgb, name=f"{img.name} (burned)", rgb=True)
    res.calibration = type(img.calibration)(**img.calibration.to_dict())
    return OpResult(new_images=[res], message=f"{len(items)} image overlay(s) burned in")


# ====================================================================== FFT tools
def _fft_mask(ps: np.ndarray) -> np.ndarray | None:
    """ImageJ's ``FFT.doMasking`` mask for an edited power spectrum (centred layout, values 0…255): white (255)
    regions pass exclusively, otherwise black (0) regions are blocked; every edited pixel is mirrored through the
    centre pixel (``changeValuesAndSymmetrize``) and the mask is smoothed three times with a periodic 3×3 integer mean
    (``(sum+4)/9``).  None when no pixel was painted 0 or 255 (ImageJ then filters nothing)."""
    ps = np.asarray(ps)
    if ps.dtype != np.uint8:
        ps = round_half_up(ps.astype(np.float32), np.uint8)
    n, m = ps.shape
    has_white = bool(np.any(ps == 255))
    has_black = bool(np.any(ps == 0))
    if not has_white and not has_black:
        return None
    v1, v2 = (255, 0) if has_white else (0, 255)
    sel = ps == v1
    yy, xx = np.nonzero(sel)
    mask = np.full((n, m), v2, np.int64)
    mask[sel] = v1
    mask[(n - yy) % n, (m - xx) % m] = v1
    for _ in range(3):
        s = sum(np.roll(np.roll(mask, dy, axis=0), dx, axis=1) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        mask = (s + 4) // 9
    return mask.astype(np.uint8)


@op("process.inverse_fft", "Inverse FFT", FFT, kind="image", scope_param=False, preview=False, roi_aware=False, supports_rgb=False,
    params=[Param("output", "choice", "Original type", choices=["Original type", "32-bit"]), Param("masking", "bool", True, help="Honour regions painted black (blocked) or white (passed) in the power spectrum")],
    description="Reconstruct the spatial image from the transform stored with a power-spectrum image. Regions painted black (0) in the spectrum are filtered out, regions painted white (255) are passed exclusively (like ImageJ).",
    keywords=["fourier", "ifft", "filter", "mask"])
def inverse_fft(ctx, output="Original type", masking=True):
    entry, img = stored_fft(ctx)
    F = entry["F"]
    fht = entry.get("fht")
    ps = img.current_plane()
    mask = _fft_mask(ps) if masking and ps.shape == F.shape else None
    if fht is not None:                        # ImageJ's float32 Hartley transform: bit-exact inverse
        h = fht
        if mask is not None:                   # FFT.doMasking: fht[i] = (float)(fht[i]*mask/255.0), mask in the FFT layout
            h = ((h.astype(np.float64) * np.fft.ifftshift(mask).astype(np.float64)) / 255.0).astype(np.float32)
        spatial = fht2d(h, inverse=True)
    else:
        if mask is not None:
            F = F * (mask.astype(np.float64) / 255.0)
        spatial = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    oh, ow = entry.get("orig_shape", spatial.shape)
    spatial = np.ascontiguousarray(spatial[:oh, :ow]).astype(np.float32)   # FFT.doInverseTransform crops the padding
    src = ctx.session.images.get(entry.get("source_id"))
    dtype = np.float32 if output == "32-bit" else np.dtype(entry.get("dtype", "float32"))
    if dtype == np.uint8:
        data = round_half_up(spatial, np.uint8)                              # convertToByte(false)
    elif dtype == np.uint16:
        data = np.trunc(np.clip(spatial.astype(np.float64), 0.0, 65535.0) + 0.5).astype(np.uint16)   # convertToShort(false)
    else:
        data = cast_to(spatial, dtype)
    base = src if src is not None else img
    res = base.copy(name=f"Inverse FFT of {entry.get('name', img.name)}", data=data[None, None, None])
    res.rgb = False
    res.lut_is_binary = False
    res.metadata.pop("fft", None)
    res.metadata.pop("fft_source", None)
    res.overlay = []
    res.display.t = res.display.z = res.display.c = 0
    res.sync_channels()
    if dtype == np.float32:
        res.display.channels[0].min, res.display.channels[0].max = float(data.min()), float(data.max() or 1.0)
    return OpResult(new_images=[res])


@op("process.fft_swap_quadrants", "Swap Quadrants", FFT, roi_aware=False,
    description="Swap the diagonal quadrants of the plane (move the origin between the corner and the centre). Like ImageJ, applying it twice restores the plane.",
    keywords=["fourier", "fftshift"])
def fft_swap_quadrants(plane, ctx):
    # ImageJ's FHT.swapQuadrants: exchange the four half-blocks (h = H//2, w = W//2); on odd sizes the last row/column
    # stays put, which makes the op an involution for every size (np.fft.fftshift is not for odd sizes: shifting twice
    # moves the plane by one pixel). For even sizes this is identical to fftshift.
    h, w = plane.shape[0] // 2, plane.shape[1] // 2
    out = plane.copy()
    out[:h, :w], out[h:2 * h, w:2 * w] = plane[h:2 * h, w:2 * w], plane[:h, :w]
    out[:h, w:2 * w], out[h:2 * h, :w] = plane[h:2 * h, :w], plane[:h, w:2 * w]
    return out


@op("process.fft_redisplay_power_spectrum", "Redisplay Power Spectrum", FFT, kind="image", scope_param=False, preview=False, roi_aware=False, supports_rgb=False,
    description="Redraw the 8-bit power spectrum from the stored transform (discarding anything painted into it).", keywords=["fourier", "restore"])
def fft_redisplay(ctx):
    entry, img = stored_fft(ctx)
    fht = entry.get("fht")
    ps = power_spectrum_8bit_from_fht(fht) if fht is not None else power_spectrum_8bit(entry["F"])
    return OpResult(data=ps[None, None, None], rgb=False, message="Power spectrum redisplayed")


@op("process.fft_options", "FFT Options", FFT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("fft_window", "bool", FFT_OPTION_DEFAULTS["fft_window"], help="Multiply by a Hann window before the forward transform (reduces edge artefacts)"),
            Param("raw_power_spectrum", "bool", FFT_OPTION_DEFAULTS["raw_power_spectrum"], help="Also output the unscaled 32-bit power spectrum |F|²"),
            Param("complex_output", "bool", FFT_OPTION_DEFAULTS["complex_output"], help="Also output a 2-slice 32-bit stack (real, imaginary)"),
            Param("do_forward", "bool", True, label="Display 8-bit power spectrum", help="Create the 8-bit, log-scaled power spectrum image")],
    description="Options for Process ▸ FFT ▸ FFT: window function, raw power spectrum, complex output.", keywords=["fourier", "settings"])
def fft_options_op(ctx, fft_window=False, raw_power_spectrum=False, complex_output=False, do_forward=True):
    opts = {"fft_window": bool(fft_window), "raw_power_spectrum": bool(raw_power_spectrum), "complex_output": bool(complex_output), "do_forward": bool(do_forward)}
    ctx.session.settings["fft_options"] = opts
    on = [k for k, v in opts.items() if v]
    return OpResult(message="FFT options: " + (", ".join(on) if on else "defaults"))


def _transform_of(ctx, image_key: str | None) -> tuple[np.ndarray, LumenImage]:
    """Centred complex transform of an open image: the stored one for a power-spectrum image, else the FFT of its current plane."""
    img = ctx.session.get_image(image_key) if image_key else ctx.img
    entry = fft_store(ctx.session).get(img.id)
    if entry is not None:
        return entry["F"], img
    if img.rgb:
        raise ValueError(f"{img.name} is RGB – convert it to 8-bit first")
    return forward_fft(img.current_plane(), window=bool(fft_options(ctx.session).get("fft_window"))), img


@op("process.fd_math", "FD Math", FFT, kind="image", scope_param=False, preview=False, roi_aware=False,
    params=[Param("image1", "image", None), Param("operation", "choice", "Correlate", choices=["Correlate", "Convolve", "Deconvolve"]), Param("image2", "image", None),
            Param("result", "string", "Result", label="Result title"), Param("do_inverse", "bool", True, label="Do inverse transform", help="Return the spatial result (32-bit, origin centred); off = a frequency-domain image"),
            Param("regularization", "float", 1e-3, min=0, step=1e-3, advanced=True, help="Deconvolve: Wiener-style damping relative to the maximum of |F2|²")],
    description="Frequency-domain maths of two same-sized images: correlation (F1·conj(F2)), convolution (F1·F2) or deconvolution (F1 / F2). Images may be spatial images or power-spectrum images with a stored transform.",
    keywords=["fourier", "correlate", "convolve", "deconvolve", "cross-correlation"])
def fd_math(ctx, image1=None, operation="Correlate", image2=None, result="Result", do_inverse=True, regularization=1e-3):
    F1, im1 = _transform_of(ctx, image1)
    F2, im2 = _transform_of(ctx, image2 or image1)
    if F1.shape != F2.shape:
        raise ValueError(f"{im1.name} ({F1.shape[1]}×{F1.shape[0]}) and {im2.name} ({F2.shape[1]}×{F2.shape[0]}) must have the same size")
    if operation == "Correlate":
        # ImageJ's FHT.conjugateMultiply(h1, h2) is the DHT of F1·conj(F2): the spatial correlation peaks at
        # centre − shift (image2 shifted right by dx shows the peak at −dx). conj(F1)·F2 would be its point reflection.
        F = F1 * np.conj(F2)
    elif operation == "Convolve":
        F = F1 * F2
    else:
        p2 = np.abs(F2) ** 2
        eps = float(regularization) * float(p2.max() or 1.0)
        F = F1 * np.conj(F2) / (p2 + eps)
    title = result or f"{operation} of {im1.name}"
    if not do_inverse:
        return OpResult(new_images=[make_fft_image(F, im1, ctx.session, name=title)], message=f"{operation}: frequency-domain result")
    spatial = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    spatial = np.fft.fftshift(spatial).astype(np.float32)  # origin in the centre like ImageJ's FD Math
    res = im1.copy(name=title, data=spatial[None, None, None])
    res.rgb = False
    res.lut_is_binary = False
    res.metadata.pop("fft", None)
    res.metadata.pop("fft_source", None)
    res.overlay = []
    res.display.t = res.display.z = res.display.c = 0
    res.sync_channels()
    res.display.channels[0].min, res.display.channels[0].max = float(spatial.min()), float(spatial.max() or 1.0)
    return OpResult(new_images=[res], message=f"{operation}: {im1.name} ⊛ {im2.name}")


@op("process.fft_custom_filter", "Custom Filter (FFT)", FFT, kind="image", scope_param=False, preview=False, supports_rgb=False,
    params=[Param("filter", "image", None, help="8-bit filter image shown like a power spectrum (origin centred): 255 = pass, 0 = block; it is resized to the image size when needed"),
            Param("all_planes", "bool", True, label="Process entire stack")],
    description="Multiply the Fourier transform of every plane by a filter image (255 = pass, 0 = block) and transform back (Process ▸ FFT ▸ Custom Filter).",
    keywords=["fourier", "filter", "bandpass", "notch"])
def fft_custom_filter(ctx, filter=None, all_planes=True):
    img = ctx.img
    if not filter:
        raise ValueError("Choose the filter image")
    filt_img = ctx.session.get_image(filter)
    f = as_float(filt_img.current_plane()).astype(np.float64)
    f = f / (255.0 if filt_img.data.dtype == np.uint8 else (float(f.max()) or 1.0))
    if f.shape != (img.height, img.width):
        from skimage.transform import resize
        f = resize(f, (img.height, img.width), order=1, preserve_range=True, anti_aliasing=False)
    f = np.clip(f, 0.0, 1.0)
    d = img.display
    planes = list(img.iter_planes()) if all_planes else [(d.t, d.z, d.c)]
    out = img.data.astype(np.float32).copy()
    mask = ctx.mask
    for (t, z, c) in planes:
        F = np.fft.fftshift(np.fft.fft2(as_float(img.data[t, z, c]).astype(np.float64)))
        res = np.real(np.fft.ifft2(np.fft.ifftshift(F * f))).astype(np.float32)
        out[t, z, c] = np.where(mask, res, out[t, z, c]) if mask is not None else res
    return OpResult(data=back_to(out, img.data), rgb=False, message=f"Custom filter {filt_img.name} applied to {len(planes)} plane(s)")


# ====================================================================== Retinex
def multi_scale_retinex(chan: np.ndarray, scales: list[float]) -> np.ndarray:
    logi = np.log1p(chan)
    acc = np.zeros_like(chan)
    for s in scales:
        acc += logi - np.log1p(ndi.gaussian_filter(chan, s, mode="nearest"))
    return acc / len(scales)


@op("process.retinex", "Retinex", "Process", kind="image", scope_param=False, preview=True,
    params=[Param("level", "choice", "Uniform", choices=["Uniform", "Low", "High"], help="Weighting of the scales: Low favours the small scale, High the large one"),
            Param("scale", "float", 240.0, min=16, max=250, step=1, help="Largest Gaussian scale (px)"), Param("scale_division", "int", 3, min=1, max=8, help="Number of scales"),
            Param("dynamic", "float", 1.2, min=0.05, max=4, step=0.1, help="Contrast: how many standard deviations around the mean map to the full range")],
    description="Multi-scale retinex with colour restoration (MSRCR) – compresses the dynamic range and enhances local contrast; grayscale images use plain multi-scale retinex (Fiji's Retinex).",
    keywords=["msrcr", "dynamic range", "illumination", "colour constancy"])
def retinex(ctx, level="Uniform", scale=240.0, scale_division=3, dynamic=1.2):
    img = ctx.img
    n = max(int(scale_division), 1)
    if n == 1:
        scales = [float(scale)]
    elif level == "Low":
        scales = [float(scale) * (0.5 ** (n - 1 - k)) for k in range(n)]  # dense at small scales
        scales = sorted(set(max(2.0, s) for s in scales))
    elif level == "High":
        scales = sorted(set(max(2.0, float(scale) * (1 - k / (2.0 * n))) for k in range(n)))
    else:
        scales = sorted(set(max(2.0, float(scale) * (k + 1) / n) for k in range(n)))
    d = img.display
    out = img.data.astype(np.float32).copy()
    top = 255.0 if img.data.dtype == np.uint8 else (65535.0 if img.data.dtype == np.uint16 else float(np.nanmax(img.data) or 1.0))
    alpha, beta, gain, offset = 125.0, 46.0, 1.0, 0.0
    tz = [(t, z) for t in range(img.nt) for z in range(img.nz)] if not ctx.preview else [(d.t, d.z)]
    for (t, z) in tz:
        chans = img.data[t, z].astype(np.float64) + 1.0  # (C, Y, X), strictly positive
        msr = np.stack([multi_scale_retinex(ch, scales) for ch in chans])
        if img.rgb and img.nc == 3:
            total = chans.sum(axis=0)
            crf = beta * (np.log1p(alpha * chans) - np.log1p(total))
            msr = gain * (crf * msr + offset)
        for c in range(img.nc):
            v = msr[c]
            mean, sd = float(v.mean()), float(v.std()) or 1.0
            lo, hi = mean - dynamic * sd, mean + dynamic * sd
            out[t, z, c] = np.clip((v - lo) / (hi - lo), 0, 1) * top
    return OpResult(data=back_to(out, img.data), rgb=img.rgb)


# ====================================================================== Math ▸ Macro
_MACRO_FUNCS = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin, "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
    "exp": np.exp, "log": np.log, "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs, "pow": np.power, "round": np.round, "floor": np.floor, "ceil": np.ceil,
    "min": np.minimum, "max": np.maximum, "minOf": np.minimum, "maxOf": np.maximum, "random": None, "PI": np.pi, "E": np.e, "NaN": np.nan,
}
#: the only AST node types a macro expression may contain (numeric literals, names, arithmetic / comparison operators,
#: plain function calls, `cond ? a : b`-style ast.IfExp); attribute access, subscripts, lambdas, comprehensions etc. are rejected
_MACRO_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Compare, ast.Call, ast.Name, ast.Constant, ast.IfExp,
                ast.operator, ast.unaryop, ast.cmpop, ast.expr_context)


def _check_macro_tree(tree: ast.Expression, ns: dict[str, Any]) -> None:
    """Whitelist the expression on its AST, not on its text: numeric literals such as 2.5e-05 or 1e+16 (the repr that
    ``ast.unparse`` prints) are Constants, never a name 'e'; every Name must be a known variable or function."""
    for node in ast.walk(tree):
        if not isinstance(node, _MACRO_NODES):
            raise ValueError(f"Unsupported syntax ({type(node).__name__}) in macro expression")
        if isinstance(node, ast.Name) and node.id not in ns:
            raise ValueError(f"Unknown variable or function {node.id!r}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported literal {node.value!r} in macro expression")
        if isinstance(node, ast.Call) and (not isinstance(node.func, ast.Name) or node.keywords):
            raise ValueError("Only plain function calls such as sin(v) or pow(v, 2) are supported")


def _plane_index(plane: np.ndarray, img: LumenImage | None) -> tuple[int, int, int] | None:
    """(t, z, c) of the plane a plane op is processing, or None.

    ``apply_plane_op`` calls a plane op once per selected plane with the view ``img.data[t, z, c]`` but does not say
    which plane it is, so the index is recovered from the view's offset into the (C-contiguous) 5-D array; a copy or a
    plane of another image gives None."""
    if img is None or plane.ndim != 2:
        return None
    data = img.data
    if data.ndim != 5 or not np.may_share_memory(plane, data):
        return None
    off = plane.__array_interface__["data"][0] - data.__array_interface__["data"][0]
    st, sz, sc = data.strides[:3]
    if off < 0 or min(st, sz, sc) <= 0:
        return None
    t, off = divmod(off, st)
    z, off = divmod(off, sz)
    c, off = divmod(off, sc)
    if off != 0 or t >= data.shape[0] or z >= data.shape[1] or c >= data.shape[2]:
        return None
    cand = data[t, z, c]
    if cand.shape != plane.shape or cand.strides != plane.strides or cand.__array_interface__["data"][0] != plane.__array_interface__["data"][0]:
        return None
    return int(t), int(z), int(c)


class _LogicToBitwise(ast.NodeTransformer):
    """`a and b` → `(a) & (b)`, `a or b` → `(a) | (b)`, `not a` → `~(a)` so that comparisons work element-wise on arrays."""

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        values = [self.visit(v) for v in node.values]
        opcls = ast.BitAnd if isinstance(node.op, ast.And) else ast.BitOr
        out = values[0]
        for v in values[1:]:
            out = ast.BinOp(left=out, op=opcls(), right=v)
        return ast.copy_location(out, node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        node = self.generic_visit(node)  # type: ignore[assignment]
        if isinstance(node.op, ast.Not):
            return ast.copy_location(ast.UnaryOp(op=ast.Invert(), operand=node.operand), node)
        return node


def _translate_expr(expr: str) -> str:
    """ImageJ macro expression → Python/numpy source: && || ! become element-wise &, |, ~ with correct precedence."""
    expr = expr.replace("&&", " and ").replace("||", " or ")
    expr = re.sub(r"!(?!=)", "~", expr).strip()  # unary ! → ~ (same precedence; element-wise logical not on comparisons)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in {expr.strip()!r}: {e.msg}") from e
    tree = ast.fix_missing_locations(_LogicToBitwise().visit(tree))
    return ast.unparse(tree)


def macro_to_python(code: str) -> list[tuple[str, str]]:
    """Statements `v = expr;` (ImageJ Process ▸ Math ▸ Macro) → [(target, python_expr)]."""
    code = re.sub(r"//.*", "", code)
    stmts = []
    for raw in re.split(r"[;\n]+", code):
        st = raw.strip()
        if not st:
            continue
        m = re.match(r"^(v)\s*(\+|-|\*|/)?=\s*(.+)$", st)
        if not m:
            raise ValueError(f"Cannot parse {st!r}: use statements of the form v = expression (v, x, y, z, w, h, a, d available)")
        target, aug, rhs = m.groups()
        py = _translate_expr(rhs)
        if aug:
            py = f"(v) {aug} ({py})"
        stmts.append((target, py))
    if not stmts:
        raise ValueError("The macro must assign v, e.g. v = v * 2")
    return stmts


@op("math.macro", "Math ▸ Macro…", "Process/Math", description="Evaluate an ImageJ-style macro expression for every pixel: v (value), x, y, z (0-based slice of the plane being processed; also t and c, its 0-based frame and channel), w, h, a (angle from the centre, radians), d (distance from the centre). Example: v = v + 50*sin(d/10).",
    params=[Param("code", "text", "v = v + 50*sin(d/10)", help="Statements separated by ; or newlines; functions: sin cos tan exp log sqrt abs pow round floor ceil min max atan2; operators + - * / % && || ! < <= == != > >=")],
    keywords=["expression", "formula", "macro", "math"], supports_rgb=True)
def math_macro(plane, ctx, code="v = v"):
    stmts = macro_to_python(code)
    H, W = plane.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    cx, cy = W / 2.0, H / 2.0
    img = ctx.image
    # z/t/c of the plane under evaluation (a stack run visits every plane with its own index, like ImageJ);
    # a direct call with a copied plane falls back to the displayed plane
    tzc = _plane_index(plane, img)
    if tzc is None:
        tzc = (img.display.t, img.display.z, img.display.c) if img is not None else (0, 0, 0)
    ns: dict[str, Any] = {k: v for k, v in _MACRO_FUNCS.items() if v is not None}
    rng = np.random.default_rng()
    ns["random"] = lambda *a: rng.random((H, W))
    ns.update(v=plane.astype(np.float64), x=xx, y=yy, z=float(tzc[1]), t=float(tzc[0]), c=float(tzc[2]), w=float(W), h=float(H),
              a=np.arctan2(-(yy - cy), xx - cx), d=np.hypot(xx - cx, yy - cy))
    for target, py in stmts:
        tree = ast.parse(py, mode="eval")  # py came from ast.unparse, so it always re-parses
        _check_macro_tree(tree, ns)
        with np.errstate(all="ignore"):
            val = eval(compile(tree, "<macro>", "eval"), {"__builtins__": {}}, ns)  # noqa: S307 – the tree is whitelisted above
        ns[target] = np.broadcast_to(np.asarray(val, dtype=np.float64), (H, W)).copy() if np.ndim(val) == 0 or np.shape(val) != (H, W) else np.asarray(val, dtype=np.float64)
    v = ns["v"]
    if plane.dtype != np.float32:  # ImageJ's ImageMath.applyMacro truncates ((int) v) for integer images, like the other 8-bit Math ops
        v = np.floor(np.asarray(v, dtype=np.float64))
    return back_to(v, plane)
