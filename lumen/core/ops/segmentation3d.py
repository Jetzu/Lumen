"""Plugins ▸ Segmentation – volumetric (3-D) commands.

The 2-D commands of :mod:`segmentation` work plane by plane; the commands here treat the Z
stack of the current frame / channel as one volume:

* a trainable 3-D pixel (voxel) classifier whose features are computed on the volume with
  Gaussians scaled for the z anisotropy of the calibration,
* 3-D label tools (26-connectivity labelling, measurements with volume / surface / sphericity,
  filtering, distance-transform watershed, label expansion, labels → per-slice ROIs),
* a seeded 3-D watershed and a 3-D random walker.

Conventions (shared with the 2-D module): training / seed ROIs come from the ROI manager and
carry the slice they were drawn on in ``position.z``; a ROI belongs to a class when its name
starts with the class name; the first class is background (label 0).  Label stacks are
uint8 / uint16 (32-bit above 65535 labels) with the Glasbey LUT, masks 0/255 uint8 and
probability stacks 32-bit with one channel per class.  Voxel distances are in x/y pixels: the
z spacing is ``pixel_depth / pixel_width`` of the calibration (1 when uncalibrated).
"""
from __future__ import annotations

import inspect
import math
import os
import warnings
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import filters as skfilt, measure as skmeas, morphology as skm, segmentation as skseg

from .. import roi as roimod
from ..image import LumenImage
from ..registry import Param, OpResult, op
from . import classifier_ops as clf
from ._util import binary_image, current_threshold, label_image
from .segmentation import (OUTPUT_CHOICES, _CLASSIFIER_PARAMS, _any_count_label_image, _attach_overlay, _class_of, _class_rois,
                           _fit_and_store, _intensity_scale, _parse_classes, _prob_image, _roi_plane, _rois_for_image, _scaled)

CAT = "Plugins/Segmentation"

#: the first five keep the order of earlier versions (saved models list their features in this order)
FEATURE_CHOICES_3D = ["Gaussian", "Difference of Gaussians", "Sobel", "Hessian", "Variance", "Laplacian", "Mean", "Minimum", "Maximum", "Median", "Structure"]
DEFAULT_FEATURES_3D = ["Gaussian", "Difference of Gaussians", "Sobel", "Hessian", "Variance"]
CLASSIFIER3D_KEY = "classifier3d"
MODEL_KIND = "classifier3d"


# =============================================================================== helpers
def _zratio(img: LumenImage) -> float:
    """z spacing in units of the x/y pixel size (1 for uncalibrated or isotropic data)."""
    cal = img.calibration
    z = float(cal.pixel_depth / cal.pixel_width) if cal.pixel_width and cal.pixel_depth else 1.0
    return z if math.isfinite(z) and z > 0 else 1.0


def _spacing(img: LumenImage) -> tuple[float, float, float]:
    """(z, y, x) voxel spacing in x/y pixel units."""
    return (_zratio(img), 1.0, 1.0)


def _calibrated_spacing(img: LumenImage) -> tuple[float, float, float]:
    """(z, y, x) voxel spacing in calibrated units (1, 1, 1 for uncalibrated images)."""
    cal = img.calibration
    if cal.scaled:
        return (float(cal.pixel_depth) or 1.0, float(cal.pixel_height) or 1.0, float(cal.pixel_width) or 1.0)
    return (1.0, 1.0, 1.0)


def _unit(img: LumenImage) -> str:
    cal = img.calibration
    return cal.unit if cal.scaled else "pixel"


def _sigma3(sigma: float, zr: float) -> tuple[float, float, float]:
    """3-D Gaussian sigma (z, y, x) for an x/y sigma in pixels, scaled for the z anisotropy."""
    return (float(sigma) / zr, float(sigma), float(sigma))


def _frames(img: LumenImage, all_frames: bool) -> list[int]:
    return list(range(img.nt)) if all_frames else [img.display.t]


def _is_binary_volume(vol: np.ndarray) -> bool:
    if vol.dtype != np.uint8:
        return False
    sub = vol[::max(1, vol.shape[0] // 16), ::max(1, vol.shape[1] // 64), ::max(1, vol.shape[2] // 64)]
    if not np.isin(np.unique(sub), (0, 255)).all():
        return False
    return bool(np.isin(np.unique(vol), (0, 255)).all())


def _binary_volume(ctx, img: LumenImage, t: int, c: int) -> np.ndarray:
    """Boolean (Z, Y, X) foreground of one frame / channel: a 0/255 mask, else the current threshold."""
    vol = img.data[t, :, c]
    if _is_binary_volume(vol):
        m = vol > 0
    else:
        thr = current_threshold(ctx)
        if thr is None:
            raise ValueError("This command needs a binary (0/255) stack or a thresholded image (Image ▸ Adjust ▸ Threshold)")
        v = np.nan_to_num(vol.astype(np.float32))
        m = (v >= thr[0]) & (v <= thr[1])
    if ctx.mask is not None:
        m = m & ctx.mask[None]
    return m


def _label_volume(vol: np.ndarray, binary_to_components: bool = True) -> np.ndarray:
    """Interpret a (Z, Y, X) volume as int32 labels (a 0/255 mask is labelled in 26-connectivity first)."""
    if binary_to_components and _is_binary_volume(vol):
        return skmeas.label(vol > 0, connectivity=3).astype(np.int32)
    lab = np.rint(np.nan_to_num(vol)).astype(np.int32)
    lab[lab < 0] = 0
    return lab


def _stack5(vols: dict[int, np.ndarray], img: LumenImage, all_frames: bool, dtype=np.int32) -> np.ndarray:
    """(T, Z, 1, Y, X) array from per-frame (Z, Y, X) volumes (T = 1 unless the whole stack was processed)."""
    T = img.nt if all_frames else 1
    out = np.zeros((T, img.nz, 1, img.height, img.width), dtype)
    for t, v in vols.items():
        out[t if all_frames else 0, :, 0] = v
    return out


def _label_stack_image(vols: dict[int, np.ndarray], img: LumenImage, all_frames: bool, name: str) -> LumenImage:
    return _any_count_label_image(_stack5(vols, img, all_frames), name, like=img)


def _slice_rois(lab: np.ndarray, t: int, c: int, min_voxels: int = 0, max_rois: int = 2000, simplify: bool = True,
                step: int = 1) -> tuple[list[dict[str, Any]], bool]:
    """Per-slice outline ROIs ("L<label> z<slice>", with position) of every object of a (Z, Y, X) label volume.

    Returns (rois, truncated): at most ``max_rois`` ROIs are produced, walking the slices in order."""
    lab = np.asarray(lab, np.int32)
    if min_voxels > 0 and lab.max() > 0:
        sizes = np.bincount(lab.ravel())
        lab = np.where(sizes[lab] >= int(min_voxels), lab, 0)
    out: list[dict[str, Any]] = []
    for z in range(0, lab.shape[0], max(1, int(step))):
        plane = lab[z]
        if not plane.any():
            continue
        for r in roimod.label_to_rois(plane, simplify=simplify):
            if len(out) >= max_rois:
                return out, True
            head = str(r.get("name", "")).split("-")[0]
            r["name"] = f"L{int(head)} z{z + 1}" if head.isdigit() else f"{r.get('name')} z{z + 1}"
            r["position"] = {"t": int(t), "z": int(z), "c": int(c)}
            out.append(r)
    return out, False


def _add_slice_rois(ctx, img: LumenImage, rois: list[dict[str, Any]]) -> int:
    for r in rois:
        pos = r.pop("position")
        ctx.session.roi_manager.add(r, image_id=img.id, position=pos)
    if rois:
        ctx.session._changed("rois")
    return len(rois)


# ===================================================================== voxel features (3-D)
def _sigmas(max_sigma: float) -> list[float]:
    """Scales 1, 2, 4 … up to ``max_sigma`` (the historical ladder, capped at 16 for volumes)."""
    return clf.sigma_list(1.0, min(float(max_sigma), 16.0))


def _settings3d(channels: str, features: list[str] | None, max_sigma: float, dtype, zr: float, min_sigma: float = 1.0) -> dict[str, Any]:
    feats = clf.normalize_features(features if features is not None else DEFAULT_FEATURES_3D, FEATURE_CHOICES_3D)
    return {"channels": channels, "features": feats, "sigmas": clf.sigma_list(min_sigma, min(float(max_sigma), 16.0)),
            "min_sigma": float(min_sigma), "max_sigma": float(max_sigma), "intensity_scale": _intensity_scale(dtype), "trained_z_ratio": float(zr),
            "feature_version": clf.FEATURE_VERSION}


_PER_SIGMA_3D = {"Gaussian": 1, "Difference of Gaussians": 1, "Sobel": 1, "Hessian": 3, "Variance": 1, "Laplacian": 1, "Mean": 1, "Minimum": 1,
                 "Maximum": 1, "Median": 1, "Structure": 3}


def _n_features(settings: dict[str, Any], n_channels: int) -> int:
    F = set(settings["features"])
    ns = len(settings["sigmas"])
    per = 1 + ns * sum(n for f, n in _PER_SIGMA_3D.items() if f in F)
    return per * n_channels


def _box3(sigma: float, zr: float) -> tuple[int, int, int]:
    """Odd (z, y, x) footprint size of radius ``sigma`` x/y pixels, the z extent scaled for the anisotropy."""
    r = max(1, int(round(sigma)))
    rz = max(0, int(round(sigma / zr)))
    return (2 * rz + 1, 2 * r + 1, 2 * r + 1)


def _halo(settings: dict[str, Any], zr: float) -> int:
    """Slices of context needed above / below a z chunk so that its features equal the whole-volume ones."""
    smax = max(float(s) for s in settings["sigmas"])
    return int(math.ceil(4.0 * smax / zr)) + 3


def _volume_features(vol: np.ndarray, settings: dict[str, Any], zr: float) -> tuple[np.ndarray, list[str]]:
    """(Z, Y, X, F) float32 feature stack of a (Z, Y, X) volume plus the feature names."""
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    p = _scaled(vol, settings.get("intensity_scale"))
    F = set(settings["features"])
    sigmas = [float(s) for s in settings["sigmas"]]
    feats: list[np.ndarray] = [p]
    names: list[str] = ["Original"]
    gs = {s: ndi.gaussian_filter(p, _sigma3(s, zr)).astype(np.float32) for s in sigmas}
    if "Gaussian" in F:
        for s in sigmas:
            feats.append(gs[s]); names.append(f"Gaussian3D_{s:g}")
    if "Difference of Gaussians" in F:
        prev, prev_s = p, 0.0
        for s in sigmas:
            feats.append(gs[s] - prev); names.append(f"DoG3D_{prev_s:g}_{s:g}")
            prev, prev_s = gs[s], s
    if "Sobel" in F:
        for s in sigmas:
            feats.append(skfilt.sobel(gs[s]).astype(np.float32)); names.append(f"Sobel3D_{s:g}")
    if "Hessian" in F:
        for s in sigmas:
            H = hessian_matrix(p, sigma=_sigma3(s, zr), order="rc", use_gaussian_derivatives=False)
            ev = hessian_matrix_eigvals(H)  # (3, Z, Y, X), descending
            del H
            for k, tag in enumerate(("max", "mid", "min")):
                feats.append((ev[k] * s * s).astype(np.float32)); names.append(f"Hessian3D_{tag}_{s:g}")
            del ev
    if "Variance" in F:
        for s in sigmas:
            m = gs[s]
            feats.append(np.maximum(ndi.gaussian_filter(p * p, _sigma3(s, zr)) - m * m, 0).astype(np.float32)); names.append(f"Variance3D_{s:g}")
    # volumetric subset of the Weka bank (box footprints of radius sigma, z extent scaled for the anisotropy)
    if "Laplacian" in F:
        for s in sigmas:
            feats.append((ndi.gaussian_laplace(p, _sigma3(s, zr)) * s * s).astype(np.float32)); names.append(f"Laplacian3D_{s:g}")
    if "Mean" in F:
        for s in sigmas:
            feats.append(ndi.uniform_filter(p, size=_box3(s, zr), mode="reflect").astype(np.float32)); names.append(f"Mean3D_{s:g}")
    if "Minimum" in F:
        for s in sigmas:
            feats.append(ndi.minimum_filter(p, size=_box3(s, zr), mode="reflect").astype(np.float32)); names.append(f"Minimum3D_{s:g}")
    if "Maximum" in F:
        for s in sigmas:
            feats.append(ndi.maximum_filter(p, size=_box3(s, zr), mode="reflect").astype(np.float32)); names.append(f"Maximum3D_{s:g}")
    if "Median" in F:
        for s in sigmas:
            feats.append(ndi.median_filter(p, size=_box3(s, zr), mode="reflect").astype(np.float32)); names.append(f"Median3D_{s:g}")
    if "Structure" in F:
        from skimage.feature import structure_tensor, structure_tensor_eigenvalues
        for s in sigmas:
            A = structure_tensor(p, sigma=_sigma3(s, zr), order="rc")
            ev = structure_tensor_eigenvalues(A)  # (3, Z, Y, X), descending
            del A
            for k, tag in enumerate(("largest", "mid", "smallest")):
                feats.append((ev[k] * s * s).astype(np.float32)); names.append(f"Structure3D_{tag}_{s:g}")
            del ev
    stack = np.stack([f.astype(np.float32, copy=False) for f in feats], axis=-1)
    return np.nan_to_num(stack), names


def _chunk_features(vol: np.ndarray, z0: int, z1: int, settings: dict[str, Any], zr: float) -> tuple[np.ndarray, list[str]]:
    """Features of slices z0..z1 (exclusive) computed with a halo of neighbouring slices → (z1 - z0, Y, X, F)."""
    h = _halo(settings, zr)
    a, b = max(0, z0 - h), min(vol.shape[0], z1 + h)
    F, names = _volume_features(vol[a:b], settings, zr)
    return F[z0 - a:z1 - a], names


def _image_chunk_features(img: LumenImage, t: int, z0: int, z1: int, settings: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    zr = _zratio(img)
    chans = range(img.nc) if settings.get("channels") == "All channels" else [img.display.c]
    parts, names = [], []
    for c in chans:
        f, n = _chunk_features(img.data[t, :, c], z0, z1, settings, zr)
        parts.append(f)
        names += [f"C{c + 1}_{x}" for x in n] if len(chans) > 1 else n
    return (np.concatenate(parts, axis=-1) if len(parts) > 1 else parts[0]), names


def _chunk_slices(img: LumenImage, settings: dict[str, Any], n_features: int, max_memory_mb: float) -> int:
    """Number of z slices per feature chunk so that the feature stack (plus its temporaries) stays within the budget."""
    bytes_per_voxel = 4.0 * (n_features + 2 * len(settings["sigmas"]) + 12)
    budget = max(float(max_memory_mb), 1.0) * 1024 * 1024
    slices = int(budget // (img.height * img.width * bytes_per_voxel))
    return int(min(img.nz, max(1, slices - 2 * _halo(settings, _zratio(img)))))


def _get_model3d(session) -> dict[str, Any]:
    model = session.settings.get(CLASSIFIER3D_KEY)
    if not model or "model" not in model:
        raise ValueError("No trained 3-D pixel classifier. Run Train 3D Pixel Classifier or Load 3D Classifier first.")
    return model


def _overlay_pins_frames() -> bool:
    """Whether segmentation._attach_overlay can pin an overlay to one frame with z free (``pin_t``); until it can, a
    single classified frame of a time series is zero-padded to the series length instead."""
    try:
        return "pin_t" in inspect.signature(_attach_overlay).parameters
    except (TypeError, ValueError):
        return False


def _apply_model3d(img: LumenImage, model: dict[str, Any], all_frames: bool, output: str, smooth: float = 0.0, mask_class: str = "",
                   probability_threshold: float = 0.5, max_memory_mb: float = 512.0, ctx=None, session=None,
                   overlay_opacity: float = 0.33) -> tuple[list[LumenImage], str]:
    """Classify the current frame (or every frame) of ``img`` in z chunks.  ``ctx`` (an OpContext) is optional: with it,
    classification reports progress per chunk and honours cancellation; ``session`` (or ``ctx.session``) is needed for
    the Overlay output."""
    est = model["model"]
    names: list[str] = list(model["classes"])
    colors = clf.parse_colors(model.get("colors"), len(names))
    settings = model["settings"]
    K = len(names)
    Z, H, W = img.nz, img.height, img.width
    frames = _frames(img, all_frames)
    probs = np.zeros((len(frames), Z, K, H, W), np.float32)
    step = _chunk_slices(img, settings, int(model["n_features"]), max_memory_mb)
    n_chunks = len(frames) * max(1, math.ceil(Z / step))
    done = 0
    for ti, t in enumerate(frames):
        for z0 in range(0, Z, step):
            z1 = min(Z, z0 + step)
            if ctx is not None:
                ctx.check_cancel()
                ctx.progress(done / n_chunks, f"Classifying slices {z0 + 1}–{z1} of frame {t + 1}")
            F, _ = _image_chunk_features(img, t, z0, z1, settings)
            if F.shape[-1] != model["n_features"]:
                raise ValueError(f"The classifier expects {model['n_features']} features but this image yields {F.shape[-1]} "
                                 f"(it was trained with channels = {settings.get('channels')!r} on {model.get('trained_on', '?')})")
            n = z1 - z0
            full = clf.predict_proba(est, F.reshape(-1, F.shape[-1]), K)
            del F
            probs[ti, z0:z1] = np.moveaxis(full.T.reshape(K, n, H, W), 0, 1)
            del full
            done += 1
        if smooth > 0:
            zr = _zratio(img)
            P = np.stack([ndi.gaussian_filter(probs[ti, :, k], _sigma3(smooth, zr)) for k in range(K)], axis=1)
            probs[ti] = P / np.maximum(P.sum(axis=1, keepdims=True), 1e-6)
    new: list[LumenImage] = []
    if output in ("Probabilities", "Both"):
        new.append(_prob_image(probs, names, f"Probabilities of {img.name}", img))
    lab = probs.argmax(axis=2)
    if output in ("Labels", "Both"):
        lim = label_image(lab[:, :, None].astype(np.uint8), f"Classification of {img.name}", like=img)
        lim.display.channels[0].max = float(max(K - 1, 1))
        lim.metadata["classes"] = ", ".join(names)
        lim.metadata["class_colors"] = ", ".join(colors)
        new.append(lim)
    if output == "Overlay":
        sess = session or (ctx.session if ctx is not None else None)
        if sess is None:
            raise ValueError("The Overlay output needs a session")
        # the whole z stack of the frame(s) is classified: the overlay covers every slice (all frames when requested).
        # A single frame of a time series is pinned to that frame with z left free, so the unclassified frames show no
        # overlay at all: padding the series with zeros would paint the first class colour over frames that were never
        # classified, indistinguishable from a genuine 'all background' result
        lab5 = lab[:, :, None].astype(np.uint8)
        pin_t = None if all_frames or img.nt == 1 else int(img.display.t)
        if pin_t is not None and _overlay_pins_frames():
            _attach_overlay(sess, img, lab5, names, colors, True, overlay_opacity, suffix=" (3D)", pin_t=pin_t)
        else:
            if pin_t is not None:  # helper without frame pins (legacy): pad the series, unclassified frames get class 0
                full5 = np.zeros((img.nt, Z, 1, H, W), np.uint8)
                full5[pin_t] = lab5[0]
                lab5 = full5
            _attach_overlay(sess, img, lab5, names, colors, True, overlay_opacity, suffix=" (3D)")
    if mask_class:
        k = _class_of(mask_class, names)
        if k is None:
            raise ValueError(f"Unknown class {mask_class!r}; classes are {', '.join(names)}")
        m = probs[:, :, k] >= probability_threshold
        new.append(binary_image(m[:, :, None], f"{names[k]} mask of {img.name}", like=img))
    summary = ", ".join(f"{names[k]} {100.0 * float((lab == k).mean()):.1f}%" for k in range(K))
    trained_dtype = model.get("trained_dtype")
    if trained_dtype and trained_dtype != str(img.dtype):
        summary += (f" (note: the classifier was trained on a {trained_dtype} image and this image is {img.dtype}; "
                    "the result is only meaningful when the intensities are on the same scale)")
    return new, summary


# ==================================================================== 3-D pixel classifier
@op("segment3d.train_classifier", "Train 3D Pixel Classifier", CAT, kind="analysis", preview=False, roi_aware=False, supports_rgb=True,
    description="Trainable 3-D segmentation: learn a random forest on volumetric features (3-D Gaussians scaled for the z spacing, "
                "DoG, Sobel magnitude, Hessian eigenvalues, local variance) from ROI-manager regions named after the classes – each ROI "
                "labels the slice it was drawn on – then classify every voxel of the current frame into probability / label stacks.",
    keywords=["weka", "3d", "machine learning", "random forest", "trainable", "voxel classification", "volume"],
    params=[Param("classes", "string", "background,foreground", help="Comma-separated class names. A ROI whose name starts with a class name (or is 'class:<name>') is a training region for that class on its slice (first class = background = label 0)"),
            Param("channels", "choice", "Current channel", choices=["Current channel", "All channels"], help="Compute the features on the current channel only or on every channel"),
            Param("features", "multichoice", DEFAULT_FEATURES_3D, choices=FEATURE_CHOICES_3D, help="The volumetric subset of the Weka bank (Mean / Minimum / Maximum / Median use box footprints)"),
            Param("min_sigma", "float", 1.0, min=0.5, max=16.0, step=0.5, help="Smallest feature scale (x/y pixels)"),
            Param("max_sigma", "float", 4.0, min=0.5, max=16.0, step=1.0, help="Feature scales min, 2·min, 4·min … up to this sigma (x/y pixels; the z sigma is divided by the z spacing)"),
            *_CLASSIFIER_PARAMS,
            Param("max_voxels_per_class", "int", 50000, min=100, max=5_000_000, label="Max samples per class", advanced=True, help="Classes with more training voxels are randomly subsampled"),
            Param("class_colors", "string", "", advanced=True, help="Comma-separated hex colours of the classes (empty = Weka's red, green, blue, cyan, magenta …); used by the Overlay output"),
            Param("max_memory_mb", "float", 512.0, min=1.0, max=65536.0, step=64.0, advanced=True, help="Feature stacks are computed in z chunks that fit this budget"),
            Param("apply_now", "bool", True, help="Classify the current frame right after training"),
            Param("output", "choice", "Both", choices=OUTPUT_CHOICES, help="Probability stacks (one channel per class), a label stack, both, or a transparent class-colour overlay on every slice"),
            Param("overlay_opacity", "float", 0.33, min=0.05, max=1.0, step=0.05, advanced=True)])
def train_classifier3d(ctx, classes="background,foreground", channels="Current channel", features=None, min_sigma=1.0, max_sigma=4.0,
                       classifier="Random forest", n_trees=100, max_depth=0, features_per_node=0, balance_classes=True, svm_max_samples=4000,
                       k_neighbors=5, hidden_units=64, max_voxels_per_class=50000, class_colors="", max_memory_mb=512.0, apply_now=True, output="Both",
                       overlay_opacity=0.33):
    img = ctx.img
    names = _parse_classes(classes)
    K = len(names)
    colors = clf.parse_colors(class_colors, K)
    zr = _zratio(img)
    settings = _settings3d(channels, features if features is not None else DEFAULT_FEATURES_3D, max_sigma, img.dtype, zr, min_sigma=min_sigma)
    pairs = _class_rois(ctx.session, img, names)
    if not pairs:
        raise ValueError(f"No training ROIs found. Add ROIs to the ROI manager named after the classes ({', '.join(names)}), e.g. 'foreground 1' "
                         "or 'class:foreground', on one or more slices of the stack.")
    H, W = img.height, img.width
    by_slice: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
    for k, r in pairs:
        by_slice.setdefault(_roi_plane(r, img), []).append((k, r))
    n_feat = _n_features(settings, img.nc if channels == "All channels" else 1)
    step = _chunk_slices(img, settings, n_feat, max_memory_mb)
    X_parts: list[list[np.ndarray]] = [[] for _ in range(K)]
    feature_names: list[str] = []
    n_groups = done = 0
    for t in sorted({t for (t, _z) in by_slice}):
        zs = sorted({z for (tt, z) in by_slice if tt == t})
        # group the labelled slices into z runs no longer than one chunk so that a run's features are computed once
        groups: list[list[int]] = []
        for z in zs:
            if groups and z - groups[-1][0] < step:
                groups[-1].append(z)
            else:
                groups.append([z])
        n_groups += len(groups)
        for g in groups:
            ctx.check_cancel()
            ctx.progress(0.6 * done / max(n_groups, 1), f"Features of training slices {g[0] + 1}–{g[-1] + 1}")
            z0, z1 = g[0], g[-1] + 1
            F, feature_names = _image_chunk_features(img, t, z0, z1, settings)
            for z in g:
                lab = np.full((H, W), -1, np.int16)
                for k, r in by_slice[(t, z)]:
                    lab[roimod.to_mask(r, (H, W))] = k
                flat = F[z - z0].reshape(-1, F.shape[-1])
                for k in range(K):
                    idx = np.flatnonzero(lab.ravel() == k)
                    if idx.size:
                        X_parts[k].append(flat[idx])
            del F
            done += 1
    model, row = _fit_and_store(ctx, img, MODEL_KIND, CLASSIFIER3D_KEY, "Classifier 3D", names, colors, settings, X_parts, feature_names, classifier,
                                n_trees, max_depth, features_per_node, balance_classes, max_voxels_per_class, svm_max_samples, k_neighbors, hidden_units,
                                extra={"training_slices": len(by_slice)})
    row["Training slices"] = len(by_slice)
    for k in range(K):
        row[f"Voxels '{names[k]}'"] = model["n_samples"][k]
    acc = model["accuracy"]
    acc_s = f"; {model['accuracy_kind']} accuracy {acc:.3f}" if math.isfinite(acc) else ""
    msg = (f"Trained {model['classifier']} on {model['n_fitted']} voxels from {len(by_slice)} slice(s), {model['n_features']} features "
           f"in {model['train_time']:.2f} s{acc_s}")
    res = OpResult(results=[row], table=clf.publish_tables(ctx.session, model, "Classifier 3D"), message=msg, log=msg)
    if apply_now:
        ctx.progress(0.8, "Classifying the current frame")
        new, summary = _apply_model3d(img, model, False, output, max_memory_mb=max_memory_mb, session=ctx.session, overlay_opacity=overlay_opacity)
        res.new_images = new
        res.message = f"{msg}. {summary}"
    return res


@op("segment3d.apply_classifier", "Apply 3D Pixel Classifier", CAT, kind="analysis", preview=False, roi_aware=False, supports_rgb=True,
    description="Classify every voxel of the current frame (or every frame) of the current – or another open – image with the trained / loaded "
                "3-D pixel classifier; features are computed in z chunks so that memory stays bounded.",
    keywords=["weka", "3d", "classify", "predict", "probability", "volume", "overlay"],
    params=[Param("image", "image", None, help="Image to classify (empty = the current image)"),
            Param("all_frames", "bool", False, label="Process every frame"),
            Param("output", "choice", "Both", choices=OUTPUT_CHOICES, help="Probability stacks (one channel per class), a label stack, both, or a transparent class-colour overlay on every slice"),
            Param("smooth", "float", 0.0, min=0.0, max=10.0, step=0.5, help="3-D Gaussian sigma (x/y pixels) applied to the probability stacks before choosing the most likely class"),
            Param("mask_class", "string", "", help="Optional: name of a class to threshold into an extra binary mask stack"),
            Param("probability_threshold", "float", 0.5, min=0.0, max=1.0, step=0.05),
            Param("max_memory_mb", "float", 512.0, min=1.0, max=65536.0, step=64.0, advanced=True, help="Feature stacks are computed in z chunks that fit this budget"),
            Param("overlay_opacity", "float", 0.33, min=0.05, max=1.0, step=0.05, advanced=True)])
def apply_classifier3d(ctx, image=None, all_frames=False, output="Both", smooth=0.0, mask_class="", probability_threshold=0.5, max_memory_mb=512.0,
                       overlay_opacity=0.33):
    img = ctx.session.get_image(image) if image else ctx.img
    model = _get_model3d(ctx.session)
    new, summary = _apply_model3d(img, model, bool(all_frames), output, smooth=smooth, mask_class=mask_class,
                                  probability_threshold=probability_threshold, max_memory_mb=max_memory_mb, ctx=ctx, session=ctx.session,
                                  overlay_opacity=overlay_opacity)
    return OpResult(new_images=new, message=summary, log=f"Applied 3-D classifier to {img.name}: {summary}")


@op("segment3d.save_classifier", "Save 3D Classifier", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    description="Write the trained 3-D pixel classifier (estimator, feature settings, class names and colours, training summary) to a .joblib file "
                "that loads on any platform.",
    keywords=["3d", "classifier", "save", "model"],
    params=[Param("path", "string", "classifier3d.joblib", help="File to write (a directory gets classifier3d.joblib inside)")])
def save_classifier3d(ctx, path="classifier3d.joblib"):
    model = _get_model3d(ctx.session)
    out = clf.save_model(model, path, "classifier3d.joblib")
    model["path"] = str(out)
    msg = f"Saved 3-D classifier ({', '.join(model['classes'])}) to {out}"
    return OpResult(message=msg, log=msg)


@op("segment3d.load_classifier", "Load 3D Classifier", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    description="Load a 3-D pixel classifier saved with Save 3D Classifier so that Apply 3D Pixel Classifier can use it.",
    keywords=["3d", "classifier", "load", "model"],
    params=[Param("path", "string", "classifier3d.joblib")])
def load_classifier3d(ctx, path="classifier3d.joblib"):
    obj = clf.load_model(path)
    if obj.get("kind") != MODEL_KIND:
        raise ValueError(f"{path} is a 2-D pixel classifier – load it with Plugins ▸ Segmentation ▸ Load Classifier")
    ctx.session.settings[CLASSIFIER3D_KEY] = obj
    table = clf.publish_tables(ctx.session, obj, "Classifier 3D")
    msg = f"Loaded 3-D classifier ({', '.join(obj['classes'])}; {obj.get('n_features', '?')} features, trained on {obj.get('trained_on', '?')})"
    return OpResult(table=table, message=msg, log=msg)


# ===================================================================== connected components
_CONN = {"6": 1, "18": 2, "26": 3}


@op("segment3d.label_3d", "Label 3D (Connected Components)", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=True,
    description="Label the connected components of a binary stack (0/255, or the current threshold) in 3-D (26-connectivity by default) "
                "into a new label stack; every frame is labelled separately.",
    keywords=["3d", "label", "connected components", "objects", "volume"],
    params=[Param("connectivity", "choice", "26", choices=["6", "18", "26"], help="Voxels sharing a face (6), an edge (18) or a corner (26) are connected"),
            Param("all_frames", "bool", False, label="Process every frame")])
def label_3d(ctx, connectivity="26", all_frames=False):
    img = ctx.img
    c = img.display.c
    vols: dict[int, np.ndarray] = {}
    total = 0
    rows = []
    for t in _frames(img, all_frames):
        lab = skmeas.label(_binary_volume(ctx, img, t, c), connectivity=_CONN[str(connectivity)]).astype(np.int32)
        n = int(lab.max())
        vols[t] = lab
        total += n
        rows.append({"Label": img.name if img.nt == 1 else f"{img.name}:t{t + 1}", "Count": n})
    out = _label_stack_image(vols, img, all_frames, f"Labels 3D of {img.name}")
    msg = f"{total} objects ({connectivity}-connectivity)"
    return OpResult(new_images=[out], results=rows, message=msg, log=msg)


# ============================================================================ measurements
def _surface_marching_cubes(sub: np.ndarray, spacing: tuple[float, float, float]) -> float:
    from skimage.measure import marching_cubes, mesh_surface_area
    padded = np.pad(sub, 1).astype(np.float32)
    if min(padded.shape) < 2:
        return float("nan")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=spacing, allow_degenerate=False)
    except (ValueError, RuntimeError):
        return float("nan")
    return float(mesh_surface_area(verts, faces)) if len(faces) else 0.0


def _surface_crofton(sub: np.ndarray, spacing: tuple[float, float, float]) -> float:
    """Crofton surface estimate from the intercepts along the three axes: S ≈ 2/3 · Σ_axes (transitions · line area)."""
    dz, dy, dx = spacing
    padded = np.pad(sub, 1)
    line_area = {0: dy * dx, 1: dz * dx, 2: dz * dy}
    total = 0.0
    for ax in range(3):
        total += float(np.count_nonzero(np.diff(padded, axis=ax))) * line_area[ax]
    return 2.0 * total / 3.0


@op("segment3d.measure_labels_3d", "Measure Labels 3D", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    description="Per-object measurements of a label stack (or 0/255 mask) in 3-D: voxel count, calibrated volume, centroid, bounding box, "
                "intensity statistics from an intensity image, surface area (marching cubes or Crofton) and sphericity → Results rows and a 'Labels 3D' table.",
    keywords=["3d", "label", "measure", "volume", "surface", "sphericity", "objects", "particles"],
    params=[Param("intensity_image", "image", None, help="Image providing the gray values for Mean / StdDev / Min / Max (current channel & frame); empty = no intensity columns"),
            Param("surface", "choice", "Marching cubes", choices=["Marching cubes", "Crofton (3 directions)", "None"], help="Surface area estimate (marching cubes also gives the sphericity)"),
            Param("min_voxels", "int", 0, min=0, max=1_000_000_000, help="Skip objects smaller than this"),
            Param("clear_results", "bool", False, help="Empty the Results table first"),
            Param("all_frames", "bool", False, label="Process every frame")])
def measure_labels_3d(ctx, intensity_image=None, surface="Marching cubes", min_voxels=0, clear_results=False, all_frames=False):
    img = ctx.img
    d = img.display
    inten = ctx.session.get_image(intensity_image) if intensity_image else None
    if inten is not None and (inten.nz, inten.height, inten.width) != (img.nz, img.height, img.width):
        raise ValueError("The intensity image must have the same width, height and number of slices as the label stack")
    dz, dy, dx = _calibrated_spacing(img)
    unit = _unit(img)
    vox_vol = dz * dy * dx
    rows: list[dict[str, Any]] = []
    for t in _frames(img, all_frames):
        lab = _label_volume(img.data[t, :, d.c])
        n = int(lab.max())
        if n == 0:
            continue
        counts = np.bincount(lab.ravel(), minlength=n + 1)
        ids = [i for i in range(1, n + 1) if counts[i] >= max(int(min_voxels), 1)]
        if not ids:
            continue
        objs = ndi.find_objects(lab)
        cents = ndi.center_of_mass(np.ones(lab.shape, np.uint8), lab, ids)
        ip = None
        if inten is not None:
            ip = np.nan_to_num(inten.data[min(t, inten.nt - 1), :, min(inten.display.c, inten.nc - 1)].astype(np.float64))
            means = ndi.mean(ip, lab, ids)
            vars_ = ndi.variance(ip, lab, ids)
            mins = ndi.minimum(ip, lab, ids)
            maxs = ndi.maximum(ip, lab, ids)
        label_name = img.name if img.nt == 1 else f"{img.name}:t{t + 1}"
        for j, i in enumerate(ids):
            sl = objs[i - 1]
            nv = int(counts[i])
            vol = nv * vox_vol
            cz, cy, cx = cents[j]
            row: dict[str, Any] = {"Label": label_name, "ID": int(i), "Voxels": nv, "Volume": float(vol),
                                   "X": float((cx + 0.5) * dx), "Y": float((cy + 0.5) * dy), "Z": float((cz + 0.5) * dz),
                                   "BX": float(sl[2].start * dx), "BY": float(sl[1].start * dy), "BZ": float(sl[0].start * dz),
                                   "Width": float((sl[2].stop - sl[2].start) * dx), "Height": float((sl[1].stop - sl[1].start) * dy),
                                   "Depth": float((sl[0].stop - sl[0].start) * dz)}
            if ip is not None:
                sd = math.sqrt(max(float(vars_[j]), 0.0) * nv / (nv - 1)) if nv > 1 else 0.0
                row.update({"Mean": float(means[j]), "StdDev": sd, "Min": float(mins[j]), "Max": float(maxs[j])})
            if surface != "None":
                sub = lab[sl] == i
                area = _surface_crofton(sub, (dz, dy, dx)) if surface.startswith("Crofton") else _surface_marching_cubes(sub, (dz, dy, dx))
                row["Surface"] = float(area)
                row["Sphericity"] = float(min((math.pi ** (1.0 / 3.0)) * (6.0 * vol) ** (2.0 / 3.0) / area, 1.0)) if area and math.isfinite(area) and area > 0 else float("nan")
            rows.append(row)
    if not rows:
        raise ValueError("No objects found: the current slice stack has no labels (or none above the minimum size)")
    cols = list(rows[0].keys())
    head = {"Volume": f"Volume ({unit}³)", "Surface": f"Surface ({unit}²)"}
    table = {"name": "Labels 3D", "columns": [head.get(c, c) for c in cols],
             "rows": [[(None if isinstance(r.get(c), float) and not math.isfinite(r[c]) else r.get(c)) for c in cols] for r in rows]}
    msg = f"{len(rows)} objects measured"
    return OpResult(results=rows, results_replace=bool(clear_results), table=table, message=msg, log=msg)


# ================================================================================ filtering
@op("segment3d.filter_labels_3d", "Filter Labels 3D", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    description="Keep only the objects of a label stack whose volume lies within the limits (and that do not touch the edges), "
                "relabelled sequentially into a new label stack.",
    keywords=["3d", "label", "filter", "size", "volume", "exclude edges"],
    params=[Param("min_volume", "float", 0.0, min=0.0, step=1.0, help="Minimum volume (calibrated units³, voxels when uncalibrated)"),
            Param("max_volume", "float", 0.0, min=0.0, step=1.0, help="Maximum volume (0 = no limit)"),
            Param("in_voxels", "bool", False, help="Interpret the volume limits as voxel counts even when the image is calibrated"),
            Param("exclude_edges", "choice", "None", choices=["None", "X/Y edges", "All edges (incl. Z)"], help="Drop objects touching the image border"),
            Param("intensity_image", "image", None, help="Optional image for the mean-intensity criterion"),
            Param("min_mean", "float", 0.0, step=1.0, label="Min mean intensity"), Param("max_mean", "float", 0.0, step=1.0, label="Max mean intensity (0 = ∞)"),
            Param("all_frames", "bool", False, label="Process every frame")])
def filter_labels_3d(ctx, min_volume=0.0, max_volume=0.0, in_voxels=False, exclude_edges="None", intensity_image=None, min_mean=0.0, max_mean=0.0, all_frames=False):
    img = ctx.img
    d = img.display
    inten = ctx.session.get_image(intensity_image) if intensity_image else None
    if inten is not None and (inten.nz, inten.height, inten.width) != (img.nz, img.height, img.width):
        raise ValueError("The intensity image must have the same width, height and number of slices as the label stack")
    dz, dy, dx = (1.0, 1.0, 1.0) if in_voxels else _calibrated_spacing(img)
    vox_vol = dz * dy * dx
    vols: dict[int, np.ndarray] = {}
    kept = total = 0
    for t in _frames(img, all_frames):
        lab = _label_volume(img.data[t, :, d.c])
        n = int(lab.max())
        keep = np.zeros(n + 1, bool)
        if n:
            counts = np.bincount(lab.ravel(), minlength=n + 1).astype(np.float64)
            present = counts[1:] > 0
            total += int(present.sum())
            vol = counts * vox_vol
            ok = present & (vol[1:] >= float(min_volume)) & ((vol[1:] <= float(max_volume)) if max_volume > 0 else True)
            if exclude_edges != "None":
                border = np.zeros(lab.shape, bool)
                border[:, 0, :] = border[:, -1, :] = border[:, :, 0] = border[:, :, -1] = True
                if exclude_edges.startswith("All"):
                    border[0] = border[-1] = True
                touching = np.unique(lab[border])
                ok[touching[touching > 0] - 1] = False
            if inten is not None and (min_mean > 0 or max_mean > 0):
                ip = np.nan_to_num(inten.data[min(t, inten.nt - 1), :, min(inten.display.c, inten.nc - 1)].astype(np.float64))
                means = np.asarray(ndi.mean(ip, lab, np.arange(1, n + 1)), np.float64)
                ok &= (means >= float(min_mean)) & ((means <= float(max_mean)) if max_mean > 0 else True)
            keep[1:] = ok
            kept += int(ok.sum())
        filtered = np.where(keep[lab], lab, 0)
        filtered, _, _ = skseg.relabel_sequential(filtered)
        vols[t] = np.asarray(filtered, np.int32)
    out = _label_stack_image(vols, img, all_frames, f"Filtered labels of {img.name}")
    msg = f"Kept {kept} of {total} objects"
    return OpResult(new_images=[out], message=msg, log=msg, results=[{"Label": img.name, "Count": kept, "Total": total}])


# ====================================================================== distance watershed
def _h_maxima_markers(surface: np.ndarray, h: float, mask: np.ndarray | None) -> np.ndarray:
    """Labelled (26-connectivity) regional maxima of ``surface`` with a dynamic of at least ``h`` (inside the mask)."""
    peaks = skm.h_maxima(surface.astype(np.float32), max(float(h), 1e-3)).astype(bool)
    if mask is not None:
        peaks &= mask
    return skmeas.label(peaks, connectivity=3).astype(np.int32)


@op("segment3d.watershed_3d", "Distance Watershed 3D", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=True,
    description="Split touching objects of a binary stack with a 3-D watershed of the (anisotropy-aware) distance transform; "
                "the seeds are the distance maxima with a dynamic of at least the tolerance (h-maxima).",
    keywords=["3d", "watershed", "distance transform", "split", "binary", "touching"],
    params=[Param("tolerance", "float", 1.0, min=0.0, max=1000.0, step=0.5, help="Minimum depth (x/y pixels) of the valley between two distance maxima for them to seed separate objects"),
            Param("smooth", "float", 0.0, min=0.0, max=10.0, step=0.5, help="3-D Gaussian sigma applied to the distance map before the maxima are found"),
            Param("connectivity", "choice", "6", choices=["6", "26"], help="Flooding connectivity"),
            Param("watershed_line", "bool", False, help="Leave a one-voxel background line between the objects"),
            Param("output", "choice", "Labels", choices=["Labels", "Binary (split)"], help="A label stack, or the binary stack with watershed lines cut in"),
            Param("all_frames", "bool", False, label="Process every frame")])
def watershed_3d(ctx, tolerance=1.0, smooth=0.0, connectivity="6", watershed_line=False, output="Labels", all_frames=False):
    img = ctx.img
    c = img.display.c
    sp = _spacing(img)
    vols: dict[int, np.ndarray] = {}
    total = 0
    for t in _frames(img, all_frames):
        m = _binary_volume(ctx, img, t, c)
        if not m.any():
            vols[t] = np.zeros(m.shape, np.int32)
            continue
        edt = ndi.distance_transform_edt(m, sampling=sp).astype(np.float32)
        if smooth > 0:
            edt = ndi.gaussian_filter(edt, _sigma3(smooth, sp[0]))
        markers = _h_maxima_markers(edt, tolerance, m)
        if markers.max() == 0:
            markers = skmeas.label(m, connectivity=3).astype(np.int32)
        lab = skseg.watershed(-edt, markers, mask=m, connectivity=_CONN[str(connectivity)], watershed_line=bool(output.startswith("Binary") or watershed_line))
        lab, _, _ = skseg.relabel_sequential(np.asarray(lab, np.int32))
        vols[t] = lab
        total += int(lab.max())
    if output.startswith("Binary"):
        out = binary_image(_stack5({t: v > 0 for t, v in vols.items()}, img, all_frames, np.uint8), f"Watershed of {img.name}", like=img)
    else:
        out = _label_stack_image(vols, img, all_frames, f"Watershed of {img.name}")
    msg = f"{total} objects after watershed"
    return OpResult(new_images=[out], message=msg, log=msg, results=[{"Label": img.name, "Count": total}])


# ========================================================================== label expansion
@op("segment3d.expand_labels_3d", "Expand Labels 3D", CAT, kind="image", preview=False, supports_rgb=False, roi_aware=False, scope_param=False,
    description="Grow every label of a label stack outwards in 3-D by a distance (x/y pixels; the z spacing is respected) without overlapping its neighbours.",
    keywords=["3d", "label", "dilate", "grow", "expand"],
    params=[Param("distance", "float", 2.0, min=0.0, max=500.0, step=1.0, unit="px")])
def expand_labels_3d(ctx, distance=2.0):
    from ..registry import cast_to
    img = ctx.img
    sp = _spacing(img)
    out = np.empty_like(img.data)
    for t in range(img.nt):
        for c in range(img.nc):
            lab = _label_volume(img.data[t, :, c], binary_to_components=False)
            grown = skseg.expand_labels(lab, distance=float(distance), spacing=sp)
            out[t, :, c] = cast_to(np.asarray(grown), img.dtype)
    return OpResult(data=out, message=f"Labels expanded by {float(distance):g} px")


# ========================================================================= labels → ROIs
@op("segment3d.labels_to_rois_3d", "Labels to ROIs 3D", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    description="Add one outline ROI per object and slice of the current label stack (or 0/255 mask) to the ROI manager, named 'L<label> z<slice>' "
                "with the slice stored in the ROI position (at most max_rois ROIs).",
    keywords=["3d", "label", "roi manager", "outline", "convert", "slices"],
    params=[Param("min_voxels", "int", 0, min=0, max=1_000_000_000, help="Skip objects smaller than this (voxels)"),
            Param("max_rois", "int", 2000, min=1, max=1_000_000, help="Stop after this many ROIs"),
            Param("slice_step", "int", 1, min=1, max=1000, help="Outline every n-th slice only", advanced=True),
            Param("simplify", "bool", True, help="Polygon outlines (off: exact pixel masks)"),
            Param("clear_manager", "bool", False, help="Empty the ROI manager first")])
def labels_to_rois_3d(ctx, min_voxels=0, max_rois=2000, slice_step=1, simplify=True, clear_manager=False):
    img = ctx.img
    d = img.display
    lab = _label_volume(img.data[d.t, :, d.c])
    rois, truncated = _slice_rois(lab, d.t, d.c, min_voxels=int(min_voxels), max_rois=int(max_rois), simplify=bool(simplify), step=int(slice_step))
    if clear_manager:
        ctx.session.roi_manager.clear()
    n = _add_slice_rois(ctx, img, rois)
    labels = sorted({int(str(r["name"]).split(" ")[0][1:]) for r in rois if str(r["name"]).startswith("L")})
    msg = f"{n} ROIs ({len(labels)} objects) added to the ROI manager" + (f" – stopped at the limit of {int(max_rois)} ROIs" if truncated else "")
    return OpResult(message=msg, log=msg, results=[{"Label": img.name, "ROIs": n, "Objects": len(labels)}])


# ======================================================================= seeded watershed
def _seed_points(ctx, img: LumenImage) -> list[tuple[int, int, int]]:
    """(z, y, x) seeds: the active point selection (on the current slice), else the point ROIs of the manager on their slices."""
    H, W, Z = img.height, img.width, img.nz
    pts: list[tuple[int, float, float]] = []
    if ctx.roi is not None and ctx.roi.get("type") == "point":
        z = img.display.z
        pts = [(z, x, y) for x, y in ctx.roi.get("points", [])]
    else:
        for r in _rois_for_image(ctx.session, img):
            if r.get("type") == "point":
                _t, z = _roi_plane(r, img)
                pts += [(z, x, y) for x, y in r.get("points", [])]
    out, seen = [], set()
    for z, x, y in pts:
        xi, yi, zi = int(x), int(y), int(z)
        if 0 <= xi < W and 0 <= yi < H and 0 <= zi < Z and (zi, yi, xi) not in seen:
            seen.add((zi, yi, xi))
            out.append((zi, yi, xi))
    return out


def _mask_volume(ctx, img: LumenImage, t: int, raw: np.ndarray, use_threshold: bool, mask_image) -> np.ndarray | None:
    mask = None
    thr = current_threshold(ctx) if use_threshold else None
    if thr is not None:
        mask = (raw >= thr[0]) & (raw <= thr[1])
    if mask_image:
        mk = ctx.session.get_image(mask_image)
        if (mk.nz, mk.height, mk.width) != (img.nz, img.height, img.width):
            raise ValueError("The mask image must have the same width, height and number of slices as the image")
        mm = mk.data[min(t, mk.nt - 1), :, min(mk.display.c, mk.nc - 1)] > 0
        mask = mm if mask is None else (mask & mm)
    if ctx.mask is not None:
        mask = ctx.mask[None].repeat(img.nz, axis=0) if mask is None else (mask & ctx.mask[None])
    return mask


@op("segment3d.seeded_watershed_3d", "Seeded Watershed 3D", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=True,
    description="Marker-controlled 3-D watershed of the smoothed (optionally inverted) intensity or its gradient magnitude from point ROIs "
                "(each on its own slice), local maxima or a marker label stack, limited to the current threshold / a mask stack.",
    keywords=["3d", "watershed", "seeds", "markers", "split", "points", "volume"],
    params=[Param("markers", "choice", "Points", choices=["Points", "Local maxima", "Markers image"], help="Points: the active point selection or the point ROIs of the ROI manager (on their slices)"),
            Param("markers_image", "image", None, help="Label stack with the seeds (markers = Markers image)"),
            Param("smooth", "float", 1.0, min=0.0, max=20.0, step=0.5, help="3-D Gaussian sigma (x/y pixels) applied before flooding"),
            Param("invert", "bool", True, help="Flood the inverted image so bright objects become basins"),
            Param("use_gradient", "bool", False, help="Flood the 3-D Sobel gradient magnitude instead of the intensity"),
            Param("use_threshold", "bool", True, help="Limit the result to the current threshold when one is set"),
            Param("mask_image", "image", None, help="Optional binary stack limiting the flooding"),
            Param("prominence", "float", 10.0, min=0.0, step=1.0, advanced=True, help="Local maxima: minimum height of a seed above its surroundings"),
            Param("connectivity", "choice", "6", choices=["6", "26"], advanced=True, help="Flooding connectivity"),
            Param("compactness", "float", 0.0, min=0.0, max=1000.0, step=0.1, advanced=True), Param("watershed_line", "bool", False),
            Param("add_to_manager", "bool", False, help="Add per-slice outline ROIs of the objects to the ROI manager"),
            Param("max_rois", "int", 2000, min=1, max=1_000_000, advanced=True)])
def seeded_watershed_3d(ctx, markers="Points", markers_image=None, smooth=1.0, invert=True, use_gradient=False, use_threshold=True, mask_image=None,
                        prominence=10.0, connectivity="6", compactness=0.0, watershed_line=False, add_to_manager=False, max_rois=2000):
    img = ctx.img
    d = img.display
    t = d.t
    zr = _zratio(img)
    raw = np.nan_to_num(img.data[t, :, d.c].astype(np.float32))
    s = ndi.gaussian_filter(raw, _sigma3(smooth, zr)) if smooth > 0 else raw
    surface = skfilt.sobel(s).astype(np.float32) if use_gradient else (-s if invert else s)
    mask = _mask_volume(ctx, img, t, raw, bool(use_threshold), mask_image)
    seeds = np.zeros(raw.shape, np.int32)
    if markers == "Points":
        pts = _seed_points(ctx, img)
        if not pts:
            raise ValueError("Seeded watershed needs a point selection or point ROIs in the ROI manager (or choose markers = Local maxima)")
        for i, (z, y, x) in enumerate(pts, 1):
            seeds[z, y, x] = i
    elif markers == "Markers image":
        if not markers_image:
            raise ValueError("Choose a markers image (a label stack with the seeds)")
        mk = ctx.session.get_image(markers_image)
        if (mk.nz, mk.height, mk.width) != (img.nz, img.height, img.width):
            raise ValueError("The markers image must have the same width, height and number of slices as the image")
        seeds = _label_volume(mk.data[min(t, mk.nt - 1), :, min(mk.display.c, mk.nc - 1)])
        if seeds.max() == 0:
            raise ValueError("The markers image has no labels")
    else:
        seeds = _h_maxima_markers(-surface, prominence, mask)
        if seeds.max() == 0:
            raise ValueError("No local maxima found – lower the prominence")
    lab = skseg.watershed(surface, seeds, mask=mask, connectivity=_CONN[str(connectivity)], compactness=float(compactness), watershed_line=bool(watershed_line))
    lab, _, _ = skseg.relabel_sequential(np.asarray(lab, np.int32))
    n = int(lab.max())
    out = _label_stack_image({t: lab}, img, False, f"Watershed 3D of {img.name}")
    n_rois = 0
    truncated = False
    if add_to_manager:
        rois, truncated = _slice_rois(lab, t, d.c, max_rois=int(max_rois))
        n_rois = _add_slice_rois(ctx, img, rois)
    msg = f"{n} objects" + (f", {n_rois} ROIs added" + (" (limit reached)" if truncated else "") if add_to_manager else "")
    return OpResult(new_images=[out], message=msg, log=msg, results=[{"Label": img.name, "Count": n}])


# ========================================================================== random walker
def _block_reduce(a: np.ndarray, factors: tuple[int, int, int], func) -> np.ndarray:
    from skimage.measure import block_reduce
    return block_reduce(a, factors, func)


def _upsample(a: np.ndarray, factors: tuple[int, int, int], shape: tuple[int, int, int]) -> np.ndarray:
    for ax, f in enumerate(factors):
        if f > 1:
            a = np.repeat(a, f, axis=ax)
    pad = [(0, max(0, shape[i] - a.shape[i])) for i in range(3)]
    a = np.pad(a, pad, mode="edge") if any(p[1] for p in pad) else a
    return a[:shape[0], :shape[1], :shape[2]]


@op("segment3d.random_walker_3d", "Random Walker 3D", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=True,
    description="3-D random-walker segmentation of the current frame from seeds: class ROIs in the ROI manager (named after the classes, "
                "each on its own slice) or a marker label stack; optionally computed on a downsampled volume for speed.",
    keywords=["3d", "seeds", "markers", "graph", "diffusion", "random walker", "volume"],
    params=[Param("classes", "string", "background,foreground", help="Seed ROIs in the ROI manager named after these classes (ignored when a markers image is given)"),
            Param("markers_image", "image", None, help="Optional label stack with seeds (0 = unlabelled)"),
            Param("beta", "float", 130.0, min=1.0, max=100000.0, step=10.0, help="Edge penalty: higher = the walk stops more strictly at intensity edges"),
            Param("downsample", "int", 1, min=1, max=8, help="Solve on a volume reduced by this factor in x/y (and in z as far as the spacing allows), then upsample the result"),
            Param("mode", "choice", "cg_j", choices=["cg_j", "cg", "bf"], advanced=True, help="Linear solver (bf = exact but slow)"),
            Param("smooth", "float", 0.0, min=0.0, max=10.0, step=0.5, advanced=True, help="3-D Gaussian sigma (x/y pixels) applied before the walk"),
            Param("output", "choice", "Labels", choices=["Labels", "Labels and probabilities"])])
def random_walker_3d(ctx, classes="background,foreground", markers_image=None, beta=130.0, downsample=1, mode="cg_j", smooth=0.0, output="Labels"):
    img = ctx.img
    d = img.display
    t = d.t
    Z, H, W = img.nz, img.height, img.width
    zr = _zratio(img)
    p = np.nan_to_num(img.data[t, :, d.c].astype(np.float32))
    finite_lo, finite_hi = float(p.min()), float(p.max())
    p = (p - finite_lo) / (finite_hi - finite_lo) if finite_hi > finite_lo else np.zeros_like(p)
    if smooth > 0:
        p = ndi.gaussian_filter(p, _sigma3(smooth, zr))
    names: list[str] | None = None
    if markers_image:
        mk = ctx.session.get_image(markers_image)
        if (mk.nz, mk.height, mk.width) != (Z, H, W):
            raise ValueError("The markers image must have the same width, height and number of slices as the current image")
        markers = _label_volume(mk.data[min(t, mk.nt - 1), :, min(mk.display.c, mk.nc - 1)], binary_to_components=False)
        offset = 0
    else:
        names = _parse_classes(classes)
        markers = np.zeros((Z, H, W), np.int32)
        for k, r in _class_rois(ctx.session, img, names):
            _rt, z = _roi_plane(r, img)
            markers[z][roimod.to_mask(r, (H, W))] = k + 1
        offset = 1
    if ctx.mask is not None:
        markers = np.where(ctx.mask[None], markers, -1)
    present = np.unique(markers[markers > 0])
    if present.size < 2:
        raise ValueError("Random walker needs seeds of at least two classes (ROIs named after the classes on their slices, or a markers stack with ≥ 2 labels)")
    # solve on a reduced grid when asked: x/y by `downsample`, z by as much as keeps the voxels roughly isotropic
    f = int(max(1, downsample))
    fz = int(max(1, min(f, round(f / zr)))) if f > 1 else 1
    factors = (fz, f, f)
    if f > 1:
        data = _block_reduce(p, factors, np.mean).astype(np.float32)
        small = np.zeros(data.shape, np.int32)
        for v in present:
            small[_block_reduce(markers == v, factors, np.max)] = int(v)
        small[_block_reduce(markers == -1, factors, np.min)] = -1
        markers_s = small
        spacing = (zr * fz / f, 1.0, 1.0)
    else:
        data, markers_s, spacing = p, markers, (zr, 1.0, 1.0)
    present = np.unique(markers_s[markers_s > 0])
    if present.size < 2:
        raise ValueError("After downsampling fewer than two classes keep seeds – lower the downsample factor")
    reachable = ndi.binary_propagation(markers_s > 0, mask=markers_s >= 0) & (markers_s == 0)
    if not reachable.any():
        raise ValueError("Random walker has nothing to segment: every voxel is already a seed (the seeds must leave unlabelled voxels next to them)")
    lut = np.zeros(int(present.max()) + 1, np.int32)
    lut[present] = np.arange(1, present.size + 1, dtype=np.int32)
    ranked = np.where(markers_s > 0, lut[np.maximum(markers_s, 0)], markers_s).astype(np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = skseg.random_walker(data, ranked, beta=float(beta), mode=mode, spacing=spacing, return_full_prob=(output != "Labels"))
    new: list[LumenImage] = []
    if output != "Labels":
        probs = np.clip(np.asarray(res, np.float32), 0.0, 1.0)  # (present.size, z, y, x) in ascending order of `present`
        probs /= np.maximum(probs.sum(axis=0, keepdims=True), 1e-6)  # the iterative solver leaves the sums a little off 1
        lab = present[np.argmax(probs, axis=0)].astype(np.int32)
        if ctx.mask is not None:
            keep = _block_reduce(ctx.mask[None].repeat(Z, axis=0), factors, np.max) if f > 1 else ctx.mask[None].repeat(Z, axis=0)
            lab[~keep] = 0
            probs[:, ~keep] = 0
        if f > 1:
            probs = np.stack([_upsample(probs[k], factors, (Z, H, W)) for k in range(probs.shape[0])])
        pnames = [names[int(v) - 1] if names else f"Label {int(v)}" for v in present]
        new.append(_prob_image(np.moveaxis(probs, 0, 1)[None], pnames, f"Probabilities of {img.name}", img))
    else:
        r = np.asarray(res, np.int32)
        lab = np.where(r > 0, present[np.clip(r, 1, present.size) - 1], 0).astype(np.int32)
    if f > 1:
        lab = _upsample(lab, factors, (Z, H, W))
    if ctx.mask is not None:
        lab[:, ~ctx.mask] = 0
    if offset:
        lab = np.maximum(lab - 1, 0)
    lim = label_image(lab[None, :, None], f"Random walker 3D of {img.name}", like=img)
    if names:
        lim.metadata["classes"] = ", ".join(names)
    lim.display.channels[0].max = float(max(int(lab.max()), 1))
    new.insert(0, lim)
    return OpResult(new_images=new, message=f"Random walker 3D: {present.size} classes" + (f" (downsampled x{f})" if f > 1 else ""))
