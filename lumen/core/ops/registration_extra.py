"""Plugins ▸ Registration extras — elastic (optical-flow) alignment, bUnwarpJ-style pairwise
elastic registration with stored deformation fields, folder registration ("Register Virtual
Stack Slices"), phase correlation and SIFT/ORB correspondences; plus the dense optic-flow
analysis (Plugins ▸ Tracking).

Dense flows are ``(2, H, W)`` float32 arrays ``[v, u]`` (row and column displacement) in the
scikit-image convention: ``moving[r + v, c + u] ≈ reference[r, c]``, so warping the moving
image with ``coords = [rows + v, cols + u]`` registers it onto the reference. Rigid / affine
transforms reuse the 3×3 (x, y) matrices of ``registration.py``.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import warp, resize

from .. import roi as roimod
from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op, cast_to
from .registration import (INTERP, MODES, ORDER_MAP, IDENTITY, _alignment_table, _channel_index, _decompose, _estimate, _estimation_plane,
                           _f32, _feature_matrix, _norm01, _overlap_ncc, _phase_shift, _translation, _translation_any_offset, _warp)

CAT = "Plugins/Registration"
FLOW_METHODS = ["TV-L1", "ILK"]
REFS_FLOW = ["Previous slice", "First slice", "Middle slice"]
FOLDER_MODES = MODES + ["Elastic"]
FEATURE_MODELS = ["Translation", "Rigid body", "Similarity", "Affine", "Projective"]
DEFORMATION_KEY = "deformation"
CORRESPONDENCES_KEY = "correspondences"

FLOW_PARAMS = [
    Param("method", "choice", "TV-L1", choices=FLOW_METHODS, help="TV-L1: accurate, keeps sharp motion boundaries. ILK (iterative Lucas-Kanade): faster, smoother"),
    Param("downscale", "int", 1, min=1, max=8, help="Estimate the flow at 1/n resolution (multi-scale: faster, smoother), then upsample it"),
    Param("smoothing", "float", 2.0, min=0, max=50, step=0.5, label="Flow smoothing (px)", help="Gaussian sigma applied to the displacement field (0 = none)"),
    Param("attachment", "float", 15.0, min=0.1, max=100, step=1, advanced=True, help="TV-L1: data-term weight (smaller = smoother flow)"),
    Param("tightness", "float", 0.3, min=0.01, max=2, step=0.05, advanced=True, help="TV-L1 tightness"),
    Param("num_warp", "int", 5, min=1, max=20, advanced=True, help="Warpings per pyramid level"),
    Param("num_iter", "int", 10, min=1, max=100, advanced=True, help="TV-L1 iterations per warping"),
    Param("radius", "int", 7, min=1, max=50, advanced=True, help="ILK window radius (px)"),
]


# ============================================================================ flow helpers
def _flow_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    return {k: params[k] for k in ("method", "downscale", "smoothing", "attachment", "tightness", "num_warp", "num_iter", "radius") if k in params}


def estimate_flow(ref: np.ndarray, mov: np.ndarray, method: str = "TV-L1", downscale: int = 1, smoothing: float = 2.0, attachment: float = 15.0,
                  tightness: float = 0.3, num_warp: int = 5, num_iter: int = 10, radius: int = 7) -> np.ndarray:
    """Dense displacement field (2, H, W) float32 = [v, u] registering ``mov`` onto ``ref``."""
    from skimage.registration import optical_flow_ilk, optical_flow_tvl1
    if ref.shape != mov.shape:
        raise ValueError(f"Planes differ in size: {ref.shape} vs {mov.shape}")
    a, b = _norm01(ref), _norm01(mov)
    H, W = a.shape
    ds = max(int(downscale), 1)
    if ds > 1:
        shp = (max(H // ds, 8), max(W // ds, 8))
        a = resize(a, shp, order=1, anti_aliasing=True, preserve_range=True)
        b = resize(b, shp, order=1, anti_aliasing=True, preserve_range=True)
    if min(a.shape) < 8:
        raise ValueError("Image too small for optical flow (needs at least 8 px per side)")
    if method == "ILK":
        flow = optical_flow_ilk(a, b, radius=max(int(radius), 1), num_warp=max(int(num_warp), 1), gaussian=False, prefilter=True)
    else:
        flow = optical_flow_tvl1(a, b, attachment=float(attachment), tightness=float(tightness), num_warp=max(int(num_warp), 1),
                                 num_iter=max(int(num_iter), 1), prefilter=True)
    flow = np.asarray(flow, dtype=np.float32)
    if ds > 1:
        sy, sx = H / a.shape[0], W / a.shape[1]
        v = resize(flow[0], (H, W), order=1, preserve_range=True) * sy
        u = resize(flow[1], (H, W), order=1, preserve_range=True) * sx
        flow = np.stack([v, u]).astype(np.float32)
    if smoothing and smoothing > 0:
        flow = np.stack([ndi.gaussian_filter(flow[0], smoothing), ndi.gaussian_filter(flow[1], smoothing)]).astype(np.float32)
    if not np.all(np.isfinite(flow)):
        flow = np.nan_to_num(flow)
    return flow


def warp_flow(plane: np.ndarray, flow: np.ndarray, order: int = 1) -> np.ndarray:
    """Warp a 2-D plane with a (2, H, W) displacement field [v, u] (float32 result, zeros outside)."""
    p = _f32(plane)
    H, W = p.shape
    if flow.shape[1:] != (H, W):
        raise ValueError(f"Deformation field is {flow.shape[2]} × {flow.shape[1]}, image is {W} × {H}")
    rr, cc = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")
    coords = np.stack([rr + flow[0], cc + flow[1]])
    return warp(p, coords, order=order, mode="constant", cval=0.0, preserve_range=True).astype(np.float32)


def _flow_magnitude(flow: np.ndarray) -> np.ndarray:
    return np.hypot(flow[0], flow[1]).astype(np.float32)


def _flow_direction(flow: np.ndarray) -> np.ndarray:
    """Direction of the motion in degrees, ImageJ convention (counter-clockwise, y up, 0 = +x)."""
    return np.degrees(np.arctan2(-flow[0], flow[1])).astype(np.float32)


def _field_image(flow: np.ndarray, name: str, like: LumenImage | None = None) -> LumenImage:
    """A 2-channel 32-bit image (C1 = dx, C2 = dy) holding a displacement field."""
    data = np.stack([flow[1], flow[0]])[None, None].astype(np.float32)  # (1, 1, 2, H, W)
    out = LumenImage(data, name=name, axes="TZCYX", channel_names=["dx", "dy"])
    if like is not None:
        out.calibration = Calibration(**like.calibration.to_dict())
    out.metadata["deformation_field"] = True
    out.display.mode = "single"
    return out


def _flow_from_image(field: LumenImage) -> np.ndarray:
    if field.nc < 2:
        raise ValueError(f"{field.name} is not a deformation field (needs two channels: dx, dy)")
    d = field.display
    dx = _f32(field.data[min(d.t, field.nt - 1), min(d.z, field.nz - 1), 0])
    dy = _f32(field.data[min(d.t, field.nt - 1), min(d.z, field.nz - 1), 1])
    return np.stack([dy, dx]).astype(np.float32)


def _padded_to(p: np.ndarray, h: int, w: int) -> np.ndarray:
    """Zero-pad / crop a plane to (h, w)."""
    out = np.zeros((h, w), np.float32)
    hh, ww = min(h, p.shape[0]), min(w, p.shape[1])
    out[:hh, :ww] = _f32(p)[:hh, :ww]
    return out


def _displacement_rows(label: str, index_name: str, k: int, flow: np.ndarray | None) -> dict[str, Any]:
    """Alignment-table row of an elastic correction. 'Mean dx/dy' is the mean shift *applied* to the moving
    image (the same convention as the rigid rows of align_stack), i.e. the negated flow."""
    if flow is None:
        return {"Label": label, index_name: k, "Mean displacement": 0.0, "Max displacement": 0.0, "Mean dx": 0.0, "Mean dy": 0.0}
    mag = _flow_magnitude(flow)
    return {"Label": label, index_name: k, "Mean displacement": float(mag.mean()), "Max displacement": float(mag.max()),
            "Mean dx": -float(flow[1].mean()), "Mean dy": -float(flow[0].mean())}


# ================================================================================ elastic ops
@op("registration.elastic_align", "Elastic Alignment (Optical Flow)", CAT, kind="image", preview=False, scope_param=False, roi_aware=False,
    params=[Param("axis", "choice", "Z", choices=["Z", "T"]), Param("reference", "choice", "Previous slice", choices=REFS_FLOW),
            Param("channel", "int", 0, min=0, help="Channel used to estimate the flow (0 = current); applied to all channels")]
           + FLOW_PARAMS + [Param("interpolation", "choice", "Linear", choices=INTERP),
                            Param("output_displacement", "bool", False, label="Displacement magnitude stack", help="Also create a 32-bit stack of the per-pixel displacement (px)"),
                            Param("create_new", "bool", False, label="Create new image")],
    description="Non-rigid (dense optical-flow) alignment of the slices or frames of a stack: every slice is warped onto the reference with a smooth per-pixel displacement field (TV-L1 or iterative Lucas-Kanade, multi-scale). Per-slice displacements go to the 'Alignment' table.",
    keywords=["elastic", "non-rigid", "optical flow", "deformable", "bunwarpj", "register"])
def elastic_align(ctx, axis="Z", reference="Previous slice", channel=0, method="TV-L1", downscale=1, smoothing=2.0, attachment=15.0, tightness=0.3,
                  num_warp=5, num_iter=10, radius=7, interpolation="Linear", output_displacement=False, create_new=False):
    img = ctx.img
    d = img.display
    ax = 1 if axis == "Z" else 0
    n = img.data.shape[ax]
    if n < 2:
        raise ValueError("Needs a time series (T > 1)" if axis == "T" else "Needs a Z stack (Z > 1)")
    c = _channel_index(img, channel)
    order = ORDER_MAP[interpolation]
    fk = dict(method=method, downscale=downscale, smoothing=smoothing, attachment=attachment, tightness=tightness, num_warp=num_warp, num_iter=num_iter, radius=radius)

    def get_plane(k: int) -> np.ndarray:
        return img.data[d.t, k, c] if axis == "Z" else img.data[k, d.z, c]

    def idx(k: int, other: int, cc: int) -> tuple[int, int, int]:
        return (k, other, cc) if axis == "T" else (other, k, cc)

    out = img.data.astype(np.float32, copy=True)
    n_other = img.data.shape[1 - ax]
    flows: list[np.ndarray | None] = [None] * n
    r = 0 if reference != "Middle slice" else n // 2
    if reference == "Previous slice":
        ref = _f32(get_plane(0))
        for k in range(1, n):
            ctx.check_cancel()
            ctx.progress(k / n, f"Slice {k + 1}/{n}")
            flow = estimate_flow(ref, get_plane(k), **fk)
            flows[k] = flow
            for other in range(n_other):
                for cc in range(img.nc):
                    i = idx(k, other, cc)
                    out[i] = warp_flow(img.data[i], flow, order)
            ref = out[idx(k, d.z if axis == "T" else d.t, c)]  # the next slice is registered onto the aligned one
    else:
        ref = get_plane(r)
        for k in range(n):
            if k == r:
                continue
            ctx.check_cancel()
            ctx.progress(k / n, f"Slice {k + 1}/{n}")
            flow = estimate_flow(ref, get_plane(k), **fk)
            flows[k] = flow
            for other in range(n_other):
                for cc in range(img.nc):
                    i = idx(k, other, cc)
                    out[i] = warp_flow(img.data[i], flow, order)
    rows = [_displacement_rows(f"{img.name}:{axis}{k + 1}", "Slice", k + 1, flows[k]) for k in range(n)]
    dmax = max(rw["Max displacement"] for rw in rows)
    msg = f"Elastically aligned {n} {'slices' if axis == 'Z' else 'frames'} ({method}, ref: {reference.lower()}); max displacement {dmax:.2f} px"
    new_images = []
    if output_displacement:
        H, W = img.height, img.width
        mags = np.stack([_flow_magnitude(f) if f is not None else np.zeros((H, W), np.float32) for f in flows])
        data = mags[None, :, None] if axis == "Z" else mags[:, None, None]
        disp = LumenImage(data.astype(np.float32), name=f"Displacement of {img.name}", axes="TZCYX")
        disp.calibration = Calibration(**img.calibration.to_dict())
        disp.display.channels[0].lut = "Fire"
        disp.metadata["displacement_magnitude"] = True
        new_images.append(disp)
    data = cast_to(out, img.data.dtype)
    table = _alignment_table(rows)
    if create_new:
        res = img.copy(name=f"Elastic {img.name}", data=data)
        res.sync_channels()
        return OpResult(new_images=[*new_images, res], table=table, message=msg, log=msg)  # the aligned image ends up active
    return OpResult(data=data, rgb=img.rgb, new_images=new_images, select_image=img.id if new_images else None, table=table, message=msg, log=msg)


@op("registration.elastic_pair", "Elastic Registration (bUnwarpJ-style)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("target", "image", None, help="The open image to register onto")] + FLOW_PARAMS
           + [Param("interpolation", "choice", "Linear", choices=INTERP),
              Param("output_field", "bool", False, label="Output deformation field", help="Also create a 2-channel 32-bit image (dx, dy) of the deformation field"),
              Param("store", "bool", True, label="Store the field", help="Keep the field for 'Apply Deformation' (session)")],
    description="Register the current image onto a target image with a dense optical-flow deformation field (applied to every plane); the field can be output as a 2-channel 32-bit image and re-applied to other images with 'Apply Deformation'.",
    keywords=["bunwarpj", "elastic", "non-rigid", "deformation field", "warp"])
def elastic_pair(ctx, target=None, method="TV-L1", downscale=1, smoothing=2.0, attachment=15.0, tightness=0.3, num_warp=5, num_iter=10, radius=7,
                 interpolation="Linear", output_field=False, store=True):
    img = ctx.img
    if not target:
        raise ValueError("Choose a target image")
    tgt = ctx.session.get_image(target)
    if tgt.id == img.id:
        raise ValueError("Target must be a different image")
    order = ORDER_MAP[interpolation]
    ht, wt = tgt.height, tgt.width
    hh, ww = max(ht, img.height), max(wt, img.width)
    tp, ip = _padded_to(_estimation_plane(tgt), hh, ww), _padded_to(_estimation_plane(img), hh, ww)
    flow = estimate_flow(tp, ip, method=method, downscale=downscale, smoothing=smoothing, attachment=attachment, tightness=tightness, num_warp=num_warp, num_iter=num_iter, radius=radius)
    out = np.zeros((img.nt, img.nz, img.nc, ht, wt), np.float32)
    for t in range(img.nt):
        for z in range(img.nz):
            for c in range(img.nc):
                out[t, z, c] = warp_flow(_padded_to(img.data[t, z, c], hh, ww), flow, order)[:ht, :wt]
    res = img.copy(name=f"{img.name} on {tgt.name} (elastic)", data=cast_to(out, img.data.dtype))
    res.calibration = Calibration(**tgt.calibration.to_dict())
    res.overlay = []
    res.sync_channels()
    field = flow[:, :ht, :wt]
    new_images = [res]
    if output_field:
        new_images.append(_field_image(field, f"Deformation field {img.name} → {tgt.name}", like=tgt))
    if store:
        ctx.session.settings[DEFORMATION_KEY] = {"flow": np.ascontiguousarray(field), "source": img.name, "target": tgt.name, "shape": [ht, wt], "method": method}
    row = _displacement_rows(f"{img.name} → {tgt.name}", "Slice", 1, field)
    row.pop("Slice", None)
    msg = f"Registered {img.name} onto {tgt.name} ({method}): mean displacement {row['Mean displacement']:.2f} px, max {row['Max displacement']:.2f} px"
    if store:
        msg += "; field stored for 'Apply Deformation'"
    return OpResult(new_images=new_images, results=[row], message=msg, log=msg)


@op("registration.apply_deformation", "Apply Deformation", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("field", "image", None, help="A 2-channel 32-bit deformation-field image (dx, dy); empty = the field stored by 'Elastic Registration'"),
            Param("interpolation", "choice", "Linear", choices=INTERP), Param("invert", "bool", False, help="Apply the negated field (approximate inverse)")],
    description="Warp every plane of the current image with a stored deformation field (from 'Elastic Registration') or with a 2-channel (dx, dy) field image.",
    keywords=["bunwarpj", "warp", "deformation field", "elastic"])
def apply_deformation(ctx, field=None, interpolation="Linear", invert=False):
    img = ctx.img
    order = ORDER_MAP[interpolation]
    if field:
        fimg = ctx.session.get_image(field)
        flow = _flow_from_image(fimg)
        src = fimg.name
    else:
        stored = ctx.session.settings.get(DEFORMATION_KEY)
        if not stored:
            raise ValueError("No stored deformation field – run 'Elastic Registration' first or choose a field image")
        flow = np.asarray(stored["flow"], np.float32)
        src = f"{stored.get('source')} → {stored.get('target')}"
    if invert:
        flow = -flow
    H, W = flow.shape[1:]
    # Warp on the union canvas of the field and the source – the max(target, moving) canvas that elastic_pair /
    # block matching estimated the field on – and crop to the field size afterwards, exactly as they do: flow
    # vectors that point beyond the field's extent then sample real source pixels instead of zero padding, so
    # re-applying a stored field (or its field image) reproduces 'Elastic Registration' on the other channels.
    hh, ww = max(H, img.height), max(W, img.width)
    full = flow
    if (hh, ww) != (H, W):
        full = np.zeros((2, hh, ww), np.float32)
        full[:, :H, :W] = flow
    out = np.zeros((img.nt, img.nz, img.nc, H, W), np.float32)
    for t in range(img.nt):
        for z in range(img.nz):
            for c in range(img.nc):
                out[t, z, c] = warp_flow(_padded_to(img.data[t, z, c], hh, ww), full, order)[:H, :W]
    res = img.copy(name=f"{img.name} (deformed)", data=cast_to(out, img.data.dtype))
    res.overlay = []
    res.sync_channels()
    mag = _flow_magnitude(flow)
    msg = f"Applied deformation field ({src}) to {img.nt * img.nz * img.nc} planes; mean displacement {mag.mean():.2f} px"
    return OpResult(new_images=[res], message=msg, log=msg)


# ================================================================================ folder registration
@op("registration.align_folder", "Register Virtual Stack Slices (Folder)", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("folder", "string", "", help="Folder with the images to register (natural sort order)"),
            Param("pattern", "string", "", help="Regular expression the file names must match (empty = all images)"),
            Param("output_folder", "string", "", help="Empty = <folder>/registered"),
            Param("mode", "choice", "Translation", choices=FOLDER_MODES, label="Transformation"),
            Param("reference", "choice", "First image", choices=["First image", "Previous image"]),
            Param("channel", "int", 0, min=0, help="Channel used to estimate the transform (0 = first)"),
            Param("upsample", "int", 10, min=1, max=100, label="Sub-pixel factor"), Param("interpolation", "choice", "Linear", choices=INTERP),
            Param("method", "choice", "TV-L1", choices=FLOW_METHODS, advanced=True, help="Elastic: optical-flow method"),
            Param("downscale", "int", 1, min=1, max=8, advanced=True, help="Elastic: flow estimated at 1/n resolution"),
            Param("smoothing", "float", 2.0, min=0, max=50, step=0.5, advanced=True, help="Elastic: flow smoothing (px)"),
            Param("output_format", "choice", "tif", choices=["tif", "png", "npy"])],
    description="Register every image of a folder (loaded one at a time) onto the first / previous image and save the registered images to an output folder, with the transforms in the 'Alignment' table and transforms.csv (Fiji: Register Virtual Stack Slices).",
    keywords=["register virtual stack", "folder", "batch registration", "sequence"])
def align_folder(ctx, folder="", pattern="", output_folder="", mode="Translation", reference="First image", channel=0, upsample=10, interpolation="Linear",
                 method="TV-L1", downscale=1, smoothing=2.0, output_format="tif"):
    from ..io import list_sequence_files, read_image, strip_image_ext, write_image
    from .batch import _compile_pattern  # validates the regex (ValueError with a readable message) and translates the '*.tif' glob habit
    if not folder or not str(folder).strip():
        raise ValueError("Give the folder that contains the images")
    # both folders resolved, so '..', symlinks and relative spellings compare consistently with the f.parents filter below
    root = Path(folder).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder")
    pattern = _compile_pattern(str(pattern or "").strip())
    out_dir = (Path(output_folder).expanduser() if str(output_folder or "").strip() else root / "registered").resolve()
    if out_dir == root or out_dir in root.parents:
        # the registered images would overwrite the originals (or every input would be skipped as 'inside the output folder')
        raise ValueError(f"The output folder must differ from the input folder and must not contain it: {out_dir}")
    files = [f for f in list_sequence_files(root, pattern, sort="natural") if out_dir not in f.parents]
    if not files:
        raise ValueError(f"No images in {root}" + (f" matching /{pattern}/" if pattern else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    order = ORDER_MAP[interpolation]
    elastic = mode == "Elastic"
    rows: list[dict[str, Any]] = []
    ref_plane: np.ndarray | None = None
    H = W = 0
    cum = IDENTITY.copy()
    written = []
    c_idx = max(int(channel) - 1, 0)
    for k, f in enumerate(files):
        try:
            im = read_image(f)
        except Exception as e:
            raise ValueError(f"Could not read {f.name}: {e}") from e
        if k == 0:
            H, W = im.height, im.width
        c = min(c_idx, im.nc - 1)
        plane = _padded_to(im.data[min(im.display.t, im.nt - 1), im.nz // 2, c], H, W)
        label = f.name
        if k == 0:
            res_data = np.zeros((im.nt, im.nz, im.nc, H, W), np.float32)
            for t, z, cc in im.iter_planes():
                res_data[t, z, cc] = _padded_to(im.data[t, z, cc], H, W)
            rows.append(_displacement_rows(label, "Image", 1, None) if elastic else {"Label": label, "Image": 1, **_decompose(IDENTITY)})
            ref_plane = plane
        else:
            res_data = np.zeros((im.nt, im.nz, im.nc, H, W), np.float32)
            if elastic:
                flow = estimate_flow(ref_plane, plane, method=method, downscale=downscale, smoothing=smoothing)
                for t, z, cc in im.iter_planes():
                    res_data[t, z, cc] = warp_flow(_padded_to(im.data[t, z, cc], H, W), flow, order)
                rows.append(_displacement_rows(label, "Image", k + 1, flow))
                if reference == "Previous image":
                    ref_plane = warp_flow(plane, flow, order)
            else:
                m, _ = _estimate(ref_plane, plane, mode, upsample)
                if reference == "Previous image":
                    cum = cum @ m
                    m_total = cum
                    ref_plane = plane
                else:
                    m_total = m
                for t, z, cc in im.iter_planes():
                    res_data[t, z, cc] = _warp(_padded_to(im.data[t, z, cc], H, W), m_total, order)
                rows.append({"Label": label, "Image": k + 1, **_decompose(m_total)})
        res = im.copy(name=strip_image_ext(f.name), data=cast_to(res_data, im.data.dtype))
        res.sync_channels()
        dest = out_dir / f"{strip_image_ext(f.name)}.{output_format}"
        written.append(write_image(res, dest, output_format))
    table = _alignment_table(rows)
    with open(out_dir / "transforms.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(table["columns"])
        w.writerows(table["rows"])
    msg = f"Registered {len(files)} images ({mode}, ref: {reference.lower()}) → {out_dir}"
    return OpResult(table=table, message=msg, log=msg)


# ================================================================================ phase correlation / correspondences
def _phase_peak(ref: np.ndarray, mov: np.ndarray) -> float:
    """Height of the (circular) phase-correlation peak, normalised to [0, 1]."""
    a, b = _f32(ref).astype(np.float64), _f32(mov).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    if not a.any() or not b.any():
        return 0.0
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    cross = cross / np.where(mag > 1e-12, mag, 1.0)
    corr = np.real(np.fft.ifft2(cross))
    return float(np.clip(corr.max(), 0.0, 1.0))


@op("registration.phase_correlation", "Phase Correlation", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("target", "image", None, help="The open image to compare the current image with"),
            Param("upsample", "int", 10, min=1, max=100, label="Sub-pixel factor"),
            Param("large_shifts", "bool", False, help="Allow offsets beyond half the image size / images of different size (zero-padded linear correlation)")],
    description="Translation between the current image and another open image by phase correlation: dx, dy (the shift that aligns the current image onto the target), the correlation peak and the normalised cross-correlation of the overlap, as a Results row.",
    keywords=["phase correlation", "shift", "translation", "cross correlation"])
def phase_correlation(ctx, target=None, upsample=10, large_shifts=False):
    img = ctx.img
    if not target:
        raise ValueError("Choose a target image")
    tgt = ctx.session.get_image(target)
    tp, ip = _estimation_plane(tgt), _estimation_plane(img)
    if large_shifts or tp.shape != ip.shape:
        dy, dx = _translation_any_offset(tp, ip, upsample)
    else:
        dy, dx = _phase_shift(tp, ip, upsample)
    h, w = max(tp.shape[0], ip.shape[0]), max(tp.shape[1], ip.shape[1])
    peak = _phase_peak(_padded_to(tp, h, w), _padded_to(ip, h, w))
    ncc = _overlap_ncc(_f32(tp), _f32(ip), int(round(dy)), int(round(dx)))
    row = {"Label": f"{img.name} → {tgt.name}", "dx": float(dx), "dy": float(dy), "Peak": peak, "Correlation": float(ncc)}
    msg = f"{img.name} is shifted by ({dx:+.2f}, {dy:+.2f}) px relative to {tgt.name} (peak {peak:.3f}, R = {ncc:.3f})"
    return OpResult(results=[row], message=msg, log=msg)


def _keypoints(plane: np.ndarray, detector: str, max_features: int) -> tuple[np.ndarray, np.ndarray, str]:
    """(keypoints (row, col), descriptors, detector used) – SIFT with an ORB fallback."""
    from skimage.feature import ORB, SIFT
    a = _norm01(plane)
    if detector == "SIFT":
        try:
            s = SIFT()
            s.detect_and_extract(a)
            kp, desc = s.keypoints, s.descriptors
            if kp is not None and len(kp) >= 4:
                if len(kp) > max_features:
                    sel = np.argsort(-s.scales)[:max_features] if getattr(s, "scales", None) is not None and len(s.scales) == len(kp) else np.arange(max_features)
                    kp, desc = kp[sel], desc[sel]
                return np.asarray(kp, float), np.asarray(desc), "SIFT"
        except (RuntimeError, ValueError, IndexError):
            pass
    try:
        orb = ORB(n_keypoints=max(int(max_features), 10), fast_threshold=0.05)
        orb.detect_and_extract(a)
        kp, desc = orb.keypoints, orb.descriptors
    except (RuntimeError, ValueError, IndexError) as e:
        raise ValueError(f"Feature detection failed: {e}") from e
    if kp is None or len(kp) < 4:
        raise ValueError("Too few features found – the image has not enough texture")
    return np.asarray(kp, float), np.asarray(desc), "ORB"


def _ransac_translation(src: np.ndarray, dst: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Translation (3×3 matrix) with the most inliers; inlier mask."""
    diffs = dst - src
    best, best_in = np.zeros(2), np.zeros(len(src), bool)
    for dvec in diffs:
        res = np.hypot(*(diffs - dvec).T)
        inl = res <= threshold
        if inl.sum() > best_in.sum():
            best, best_in = dvec, inl
    if best_in.any():
        best = diffs[best_in].mean(axis=0)
        best_in = np.hypot(*(diffs - best).T) <= threshold
    return _translation(float(best[0]), float(best[1])), best_in


def estimate_point_transform(src: np.ndarray, dst: np.ndarray, model: str, residual_threshold: float = 3.0, use_ransac: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(3×3 matrix mapping src (x, y) → dst (x, y), inlier mask) for a FEATURE_MODELS model."""
    from skimage.measure import ransac
    from skimage.transform import AffineTransform, EuclideanTransform, ProjectiveTransform, SimilarityTransform, estimate_transform
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    if model == "Translation":
        if n < 1:
            raise ValueError("Translation needs at least one point pair")
        if use_ransac and n >= 2:
            return _ransac_translation(src, dst, residual_threshold)
        m = _translation(*(dst - src).mean(axis=0))
        return m, np.ones(n, bool)
    cls, min_samples, ttype = {"Rigid body": (EuclideanTransform, 2, "euclidean"), "Similarity": (SimilarityTransform, 2, "similarity"),
                               "Affine": (AffineTransform, 3, "affine"), "Projective": (ProjectiveTransform, 4, "projective")}[model]
    if n < min_samples:
        raise ValueError(f"{model} needs at least {min_samples} point pairs ({n} given)")
    if use_ransac and n > min_samples:
        try:
            tf, inl = ransac((src, dst), cls, min_samples=min_samples, residual_threshold=float(residual_threshold), max_trials=2000, rng=0)
        except Exception:
            tf, inl = None, None
        if tf is not None and inl is not None and int(inl.sum()) >= min_samples and np.all(np.isfinite(tf.params)):
            return np.asarray(tf.params, float), np.asarray(inl, bool)
    tf = estimate_transform(ttype, src, dst)
    m = np.asarray(tf.params, float)
    if not np.all(np.isfinite(m)):
        raise ValueError("Degenerate point configuration – the transform could not be estimated")
    res = np.hypot(*(tf(src) - dst).T)
    return m, res <= max(float(residual_threshold), 1e-9) if use_ransac else np.ones(n, bool)


def _apply_matrix(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    h = np.c_[pts, np.ones(len(pts))] @ m.T
    return h[:, :2] / np.where(np.abs(h[:, 2:3]) > 1e-12, h[:, 2:3], 1e-12)


def transform_params(m: np.ndarray) -> dict[str, float]:
    """dx, dy, Angle, Scale (+ perspective terms) of a 3×3 matrix for the Results table."""
    out = _decompose(m)
    out["Shear"] = float(math.degrees(math.atan2(m[0, 0] * m[0, 1] + m[1, 0] * m[1, 1], m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]))) if abs(np.linalg.det(m[:2, :2])) > 1e-12 else 0.0
    if abs(m[2, 0]) > 1e-12 or abs(m[2, 1]) > 1e-12:
        out["Perspective x"] = float(m[2, 0])
        out["Perspective y"] = float(m[2, 1])
    return out


@op("registration.feature_correspondences", "Extract SIFT Correspondences", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("target", "image", None, help="The open image to match the current image with"),
            Param("detector", "choice", "SIFT", choices=["SIFT", "ORB"]),
            Param("model", "choice", "Affine", choices=FEATURE_MODELS, label="Transformation", help="Model fitted with RANSAC to filter the matches"),
            Param("max_features", "int", 500, min=10, max=5000), Param("max_ratio", "float", 0.8, min=0.1, max=1.0, step=0.05, help="Lowe ratio test for descriptor matching"),
            Param("residual_threshold", "float", 3.0, min=0.1, max=100, step=0.5, label="Inlier residual (px)"),
            Param("add_rois", "bool", True, label="Point ROIs 'Matches'", help="Add the inlier points as point ROIs on both images")],
    description="SIFT (fallback ORB) features matched between the current image and a target image, filtered with RANSAC on the chosen transformation model: inlier points as 'Matches' point ROIs on both images, a 'Correspondences' table and the estimated transform in the Results table.",
    keywords=["sift", "mops", "correspondences", "feature matching", "ransac", "landmarks"])
def feature_correspondences(ctx, target=None, detector="SIFT", model="Affine", max_features=500, max_ratio=0.8, residual_threshold=3.0, add_rois=True):
    from skimage.feature import match_descriptors
    img = ctx.img
    if not target:
        raise ValueError("Choose a target image")
    tgt = ctx.session.get_image(target)
    if tgt.id == img.id:
        raise ValueError("Target must be a different image")
    pa, pb = _estimation_plane(img), _estimation_plane(tgt)
    ka, da, used_a = _keypoints(pa, detector, max_features)
    kb, db, used_b = _keypoints(pb, detector, max_features)
    if used_a != used_b:  # SIFT succeeded on one image only: redo both with ORB so the descriptors are comparable
        ka, da, used_a = _keypoints(pa, "ORB", max_features)
        kb, db, used_b = _keypoints(pb, "ORB", max_features)
    matches = match_descriptors(da, db, cross_check=True, max_ratio=float(max_ratio))
    if len(matches) < 2:
        raise ValueError(f"Only {len(matches)} descriptor matches between {img.name} and {tgt.name} – not enough to estimate a transform")
    src = ka[matches[:, 0]][:, ::-1] + 0.5  # current image (x, y) in Lumen ROI coordinates (pixel i spans [i, i+1))
    dst = kb[matches[:, 1]][:, ::-1] + 0.5
    m, inl = estimate_point_transform(src - 0.5, dst - 0.5, model, residual_threshold, use_ransac=True)
    n_in = int(inl.sum())
    if n_in < 2:
        raise ValueError(f"No consistent matches ({len(matches)} descriptor matches, {n_in} inliers)")
    s_in, d_in = src[inl], dst[inl]
    resid = np.hypot(*(_apply_matrix(m, s_in - 0.5) - (d_in - 0.5)).T)
    cols = ["Match", "X1", "Y1", "X2", "Y2", "Residual"]
    table = {"name": "Correspondences", "columns": cols,
             "rows": [[i + 1, round(float(a[0]), 2), round(float(a[1]), 2), round(float(b[0]), 2), round(float(b[1]), 2), round(float(r), 3)] for i, (a, b, r) in enumerate(zip(s_in, d_in, resid))]}
    params = transform_params(m)
    row = {"Label": f"{img.name} → {tgt.name}", "Model": model, "Detector": used_a, "N matches": int(len(matches)), "N inliers": n_in, **params, "RMS residual": float(np.sqrt(np.mean(resid ** 2)))}
    ctx.session.settings[CORRESPONDENCES_KEY] = {"source": img.id, "target": tgt.id, "src": s_in.tolist(), "dst": d_in.tolist(), "matrix": m.tolist(), "model": model}
    rois = []
    if add_rois:
        rois = [{"type": "point", "points": s_in.round(3).tolist(), "name": "Matches", "color": "#00ff00", "image_id": img.id,
                 "position": {"t": img.display.t, "z": img.display.z, "c": img.display.c}},
                {"type": "point", "points": d_in.round(3).tolist(), "name": "Matches", "color": "#00ff00", "image_id": tgt.id,
                 "position": {"t": tgt.display.t, "z": tgt.display.z, "c": tgt.display.c}}]
    msg = (f"{n_in} inliers of {len(matches)} {used_a} matches ({model}): dx {params['dx']:+.2f}, dy {params['dy']:+.2f}, "
           f"angle {params['Angle']:+.2f}°, scale {params['Scale']:.3f}")
    return OpResult(results=[row], table=table, rois_add=rois, message=msg, log=msg)


# ================================================================================ dense optic flow (time-lapse)
@op("process.optic_flow", "Optic Flow (Dense)", "Process", kind="analysis", preview=False, roi_aware=False, supports_rgb=False, menu="Plugins/Tracking",
    params=[Param("channel", "int", 0, min=0, help="Channel to analyse (0 = current)")] + FLOW_PARAMS
           + [Param("magnitude", "bool", True, label="Magnitude image (px/frame)"), Param("direction", "bool", True, label="Direction image (degrees)"),
              Param("components", "bool", False, label="Component image (dx, dy)"),
              Param("vector_overlay", "bool", False, label="Arrow overlay", help="Draw the flow as arrows on a grid: one set per frame pair, pinned (overlay 'position') to the first frame of the pair"),
              Param("grid", "int", 16, min=4, max=256, label="Arrow grid spacing (px)"), Param("vector_scale", "float", 1.0, min=0.01, max=100, step=0.5, label="Arrow length scale"),
              Param("min_magnitude", "float", 0.0, min=0, step=0.1, label="Hide arrows shorter than (px)"), Param("color", "string", "#00ffff", label="Arrow colour")],
    description="Dense optical flow (TV-L1 or iterative Lucas-Kanade) between consecutive frames of a time series: 32-bit magnitude (px/frame) and direction (degrees, counter-clockwise from +x) stacks with one plane per frame pair, the mean flow per frame in the 'Optic flow' table and optionally an arrow overlay on a grid.",
    keywords=["optical flow", "motion", "velocity field", "piv", "farneback", "lucas-kanade"])
def optic_flow(ctx, channel=0, method="TV-L1", downscale=1, smoothing=2.0, attachment=15.0, tightness=0.3, num_warp=5, num_iter=10, radius=7,
               magnitude=True, direction=True, components=False, vector_overlay=False, grid=16, vector_scale=1.0, min_magnitude=0.0, color="#00ffff"):
    img = ctx.img
    d = img.display
    if img.nt < 2:
        raise ValueError("Needs a time series (T > 1)")
    c = _channel_index(img, channel)
    fk = dict(method=method, downscale=downscale, smoothing=smoothing, attachment=attachment, tightness=tightness, num_warp=num_warp, num_iter=num_iter, radius=radius)
    H, W = img.height, img.width
    n = img.nt - 1
    mags = np.zeros((n, H, W), np.float32) if magnitude else None
    dirs = np.zeros((n, H, W), np.float32) if direction else None
    comps = np.zeros((n, 2, H, W), np.float32) if components else None
    rows = []
    arrows: list[dict[str, Any]] = []
    g = max(int(grid), 4)
    for k in range(n):
        flow = estimate_flow(img.data[k, d.z, c], img.data[k + 1, d.z, c], **fk)
        mag = _flow_magnitude(flow)
        if mags is not None:
            mags[k] = mag
        if dirs is not None:
            dirs[k] = _flow_direction(flow)
        if comps is not None:
            comps[k, 0], comps[k, 1] = flow[1], flow[0]
        mdx, mdy = float(flow[1].mean()), float(flow[0].mean())
        rows.append({"Label": f"{img.name}:t{k + 1}→t{k + 2}", "Frame": k + 1, "Mean dx": mdx, "Mean dy": mdy, "Mean magnitude": float(mag.mean()),
                     "Max magnitude": float(mag.max()), "Mean direction": float(math.degrees(math.atan2(-mdy, mdx)))})
        if vector_overlay:
            for gy in range(0, H, g):
                for gx in range(0, W, g):
                    cell = flow[:, gy:gy + g, gx:gx + g]
                    v, u = float(cell[0].mean()), float(cell[1].mean())
                    if math.hypot(u, v) < max(float(min_magnitude), 1e-6):
                        continue
                    cx, cy = gx + min(g, W - gx) / 2.0, gy + min(g, H - gy) / 2.0
                    arrows.append({"id": roimod.new_id(), "type": "arrow", "x1": round(cx, 2), "y1": round(cy, 2), "x2": round(cx + u * vector_scale, 2), "y2": round(cy + v * vector_scale, 2),
                                   "color": color, "stroke_width": 1.0, "head_size": max(3.0, min(g / 3.0, 8.0)), "double_headed": False,
                                   "position": {"t": k, "z": d.z, "c": d.c}, "flow": True})
    new_images = []
    cal = Calibration(**img.calibration.to_dict())
    if mags is not None:
        im = LumenImage(mags[:, None, None], name=f"Flow magnitude of {img.name}", axes="TZCYX")
        im.calibration = cal
        im.display.channels[0].lut = "Fire"
        im.metadata.update({"optic_flow": "magnitude", "unit": "px/frame"})
        new_images.append(im)
    if dirs is not None:
        im = LumenImage(dirs[:, None, None], name=f"Flow direction of {img.name}", axes="TZCYX")
        im.calibration = Calibration(**cal.to_dict())
        im.display.channels[0].lut = "Spectrum"
        im.display.channels[0].min, im.display.channels[0].max = -180.0, 180.0
        im.metadata.update({"optic_flow": "direction", "unit": "degrees"})
        new_images.append(im)
    if comps is not None:
        im = LumenImage(comps[:, None], name=f"Flow components of {img.name}", axes="TZCYX", channel_names=["dx", "dy"])
        im.calibration = Calibration(**cal.to_dict())
        im.metadata.update({"optic_flow": "components", "unit": "px/frame"})
        new_images.append(im)
    cols = list(rows[0].keys())
    table = {"name": "Optic flow", "columns": cols, "rows": [[r.get(k) for k in cols] for r in rows]}
    overlay = None
    if vector_overlay:
        overlay = [o for o in img.overlay if not o.get("flow")] + arrows
    mean_mag = float(np.mean([r["Mean magnitude"] for r in rows]))
    msg = f"Optic flow ({method}) over {n} frame pairs: mean magnitude {mean_mag:.2f} px/frame"
    if vector_overlay:
        msg += f"; {len(arrows)} arrows"
    return OpResult(new_images=new_images, table=table, overlay=overlay, message=msg, log=msg)

