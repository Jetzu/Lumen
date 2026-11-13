"""Plugins ▸ Segmentation.

Advanced segmentation commands: a trainable (Weka-style) pixel classifier, random walker,
active contours (Chan-Vese / morphological snakes), superpixels, multi-Otsu, deep-learning
wrappers (Cellpose, StarDist – optional dependencies) and label-image utilities.

Conventions shared by the commands in this module:

* Training / seed regions come from the ROI manager.  A ROI belongs to a class when its
  ``name`` starts with the class name (``"foreground 1"``, ``"foreground-2"`` → ``foreground``) or is
  ``class:<name>`` (the Pixel Classifier panel's traces).  The first class of the list is treated as
  background and gets label 0.
* Label images are uint8/uint16 (32-bit above 65535 labels) with the Glasbey LUT (``_util.label_image``), masks are
  0/255 uint8 (``_util.binary_image``), probability maps are 32-bit with one channel per class.
* The pixel classifier's feature bank, classifier choice, model files and summary tables live in
  :mod:`classifier_ops`; the trained model is ``session.settings["classifier"]`` and its summary is published as
  the ``Classifier`` / ``Classifier classes`` / ``Classifier summary`` tables of ``session.tables``.
"""
from __future__ import annotations

import math
import os
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import filters as skfilt, measure as skmeas, morphology as skm, segmentation as skseg

from .. import roi as roimod
from ..image import LumenImage
from ..registry import OperationCancelled, Param, OpResult, op
from . import classifier_ops as clf
from .classifier_ops import (CLASSIFIER_CHOICES, DEFAULT_FEATURES, FEATURE_CHOICES, MODEL_VERSION, intensity_scale as _intensity_scale,
                             membrane_projections_legacy as _membrane_projections, norm01 as _norm01, plane_features as _plane_features,
                             scaled as _scaled)
from ._util import back_to, binary_image, current_threshold, label_image

CAT = "Plugins/Segmentation"

CLASSIFIER_KEY = "classifier"
MODEL_KIND_2D = "classifier"
OUTPUT_CHOICES = ["Probabilities", "Labels", "Both", "Overlay"]
OVERLAY_SOURCE = "classifier"
OVERLAY_ITEM_ID = "classifier-overlay"


# =============================================================================== helpers
# _norm01 / _intensity_scale / _scaled / _plane_features / _membrane_projections are the classifier_ops functions
# (imported above under their historical names for the 3-D module and external callers).


def _as_labels(plane: np.ndarray, binary_to_components: bool = False) -> np.ndarray:
    """Interpret a plane as an int32 label image (a 0/255 mask is optionally labelled first)."""
    if binary_to_components and plane.dtype == np.uint8 and np.isin(np.unique(plane), (0, 255)).all():
        return skmeas.label(plane > 0, connectivity=2).astype(np.int32)
    lab = np.rint(np.nan_to_num(plane)).astype(np.int32)
    lab[lab < 0] = 0
    return lab


def _parse_classes(classes: str) -> list[str]:
    names: list[str] = []
    for n in str(classes or "").replace(";", ",").split(","):
        n = n.strip()
        if n and n.lower() not in [o.lower() for o in names]:
            names.append(n)
    if len(names) < 2:
        raise ValueError("Give at least two comma-separated class names, e.g. 'background,foreground'")
    return names


def _class_of(name: str | None, classes: list[str]) -> int | None:
    """Index of the class whose name is a prefix of the ROI name (longest match wins); a ``class:`` prefix
    (the Pixel Classifier panel's trace naming, ``class:foreground``) is stripped first."""
    n = (name or "").strip()
    if n.lower().startswith("class:"):
        n = n[6:].strip()
    n = n.lower()
    best: int | None = None
    for i, c in enumerate(classes):
        cl = c.strip().lower()
        if cl and n.startswith(cl) and (best is None or len(cl) > len(classes[best])):
            best = i
    return best


def _rois_for_image(session, img: LumenImage) -> list[dict[str, Any]]:
    """ROI-manager entries belonging to this image (ROIs without image id count as belonging to every image)."""
    rois = session.roi_manager.to_list()
    mine = [r for r in rois if not r.get("image_id") or r.get("image_id") == img.id]
    return mine or list(rois)


def _class_rois(session, img: LumenImage, classes: list[str]) -> list[tuple[int, dict[str, Any]]]:
    out = []
    for r in _rois_for_image(session, img):
        k = _class_of(r.get("name"), classes)
        if k is not None and (roimod.is_area(r) or roimod.is_line(r) or r.get("type") == "point"):
            out.append((k, r))
    return out


def _roi_plane(r: dict[str, Any], img: LumenImage) -> tuple[int, int]:
    """(t, z) plane a ROI was drawn on (its position, else the current plane)."""
    pos = r.get("position") or {}
    d = img.display
    t = int(pos.get("t", d.t)) if isinstance(pos, dict) else d.t
    z = int(pos.get("z", d.z)) if isinstance(pos, dict) else d.z
    return min(max(t, 0), img.nt - 1), min(max(z, 0), img.nz - 1)


def _on_current_plane(r: dict[str, Any], img: LumenImage) -> bool:
    return _roi_plane(r, img) == (img.display.t, img.display.z)


def _add_rois(ctx, img: LumenImage, rois: list[dict[str, Any]], t: int, z: int, c: int) -> int:
    """Add ROIs to the manager with an explicit stack position."""
    for r in rois:
        ctx.session.roi_manager.add(r, image_id=img.id, position={"t": int(t), "z": int(z), "c": int(c)})
    if rois:
        ctx.session._changed("rois")
    return len(rois)


def _name_rois_by_label(rois: list[dict[str, Any]]) -> None:
    """Rename ROIs made by ``roi.label_to_rois`` ("0012-y-x") to "L12"."""
    for r in rois:
        head = str(r.get("name", "")).split("-")[0]
        if head.isdigit():
            r["name"] = f"L{int(head)}"


def _any_count_label_image(labels: np.ndarray, name: str, like: LumenImage) -> LumenImage:
    """Alias of :func:`~lumen.core.ops._util.label_image` (kept for its callers): uint8 / uint16 / float32 by label
    count, so a label image never wraps above 65535 (float32 holds integers exactly up to 2**24; every label consumer
    reads labels dtype-independently through ``_as_labels``)."""
    return label_image(labels, name, like=like)


def _prob_image(probs: np.ndarray, names: list[str], name: str, like: LumenImage) -> LumenImage:
    """32-bit probability image with one Grays channel per class, display range 0–1."""
    pim = LumenImage(probs.astype(np.float32), name=name, axes="TZCYX" if probs.ndim == 5 else "CYX", channel_names=names)
    pim.calibration = type(like.calibration)(**like.calibration.to_dict())
    for ch in pim.display.channels:
        ch.lut = "Grays"
        ch.min, ch.max = 0.0, 1.0
    pim.display.mode = "single"
    pim.display.c = pim.nc - 1  # show the last (usually the foreground) class first
    return pim


# ============================================================ pixel features (Weka-style)
def _sigmas(max_sigma: float) -> list[float]:
    """Scales 1, 2, 4 … up to ``max_sigma`` (the historical ladder = ``sigma_list(1, max_sigma)``)."""
    return clf.sigma_list(1.0, max_sigma)


def _feature_settings(channels: str, features: list[str] | None, max_sigma: float, dtype=None, min_sigma: float = 1.0,
                      membrane_thickness: int = 1, membrane_patch_size: int = 19) -> dict[str, Any]:
    return clf.feature_settings(channels, features if features is not None else DEFAULT_FEATURES, min_sigma, max_sigma, dtype=dtype,
                                membrane_thickness=membrane_thickness, membrane_patch_size=membrane_patch_size)


def _image_features(img: LumenImage, t: int, z: int, settings: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    chans = range(img.nc) if settings.get("channels") == "All channels" else [img.display.c]
    parts, names = [], []
    for c in chans:
        f, n = _plane_features(img.data[t, z, c], settings)
        parts.append(f)
        names += [f"C{c + 1}_{x}" for x in n] if len(chans) > 1 else n
    return np.concatenate(parts, axis=-1), names


def _get_model(session) -> dict[str, Any]:
    model = session.settings.get(CLASSIFIER_KEY)
    if not model or "model" not in model:
        raise ValueError("No trained pixel classifier. Run Train Pixel Classifier or Load Classifier first.")
    return model


def _predict_probs(img: LumenImage, model: dict[str, Any], all_planes: bool, smooth: float = 0.0, ctx=None,
                   progress: tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """(T, Z, K, H, W) class probabilities of the current plane (T = Z = 1) or of every plane of ``img``.

    ``ctx`` (an OpContext) is optional: with it, classification reports progress per plane (mapped into the
    ``progress`` fraction window, for batch callers) and honours cancellation."""
    est = model["model"]
    names: list[str] = list(model["classes"])
    settings = model["settings"]
    K = len(names)
    d = img.display
    H, W = img.height, img.width
    planes = [(t, z) for t in range(img.nt) for z in range(img.nz)] if all_planes else [(d.t, d.z)]
    T, Z = (img.nt, img.nz) if all_planes else (1, 1)
    probs = np.zeros((T, Z, K, H, W), np.float32)
    p0, p1 = progress
    for i, (t, z) in enumerate(planes):
        if ctx is not None:
            ctx.check_cancel()
            ctx.progress(p0 + (p1 - p0) * i / len(planes), f"Classifying plane {i + 1}/{len(planes)}")
        F, _ = _image_features(img, t, z, settings)
        if F.shape[-1] != model["n_features"]:
            raise ValueError(f"The classifier expects {model['n_features']} features but this image yields {F.shape[-1]} "
                             f"(it was trained with channels = {settings.get('channels')!r} on {model.get('trained_on', '?')})")
        P = clf.predict_proba(est, F.reshape(-1, F.shape[-1]), K).T.reshape(K, H, W)
        if smooth > 0:
            P = np.stack([ndi.gaussian_filter(P[k], smooth) for k in range(K)])
            P /= np.maximum(P.sum(axis=0, keepdims=True), 1e-6)
        ti, zi = (t, z) if all_planes else (0, 0)
        probs[ti, zi] = P
    return probs


def _find_overlay_image(session, img: LumenImage) -> LumenImage | None:
    """The open colour image that carries the classifier overlay of ``img`` (None when none / closed)."""
    for im in session.images.values():
        if im.metadata.get("classifier_overlay_of") == img.id:
            return im
    return None


def _overlay_item(ov: LumenImage, opacity: float) -> dict[str, Any]:
    item: dict[str, Any] = {"type": "image", "id": OVERLAY_ITEM_ID, "image_id": ov.id, "opacity": float(min(max(opacity, 0.0), 1.0)),
                            "source": OVERLAY_SOURCE, "name": "Pixel classifier"}
    t, z = int(ov.metadata.get("classifier_overlay_t", -1)), int(ov.metadata.get("classifier_overlay_z", -1))
    if t >= 0 or z >= 0:  # a pinned axis is fixed, a free axis (-1) follows the source image's slider
        item["position"] = {"t": t if t >= 0 else None, "z": z if z >= 0 else None, "c": None}
    return item


def _attach_overlay(session, img: LumenImage, labels: np.ndarray, names: list[str], colors: list[str], all_planes: bool,
                    opacity: float, suffix: str = "", pin_t: int | None = None) -> LumenImage:
    """Show the classification as a transparent colour overlay on ``img``: the class colours become an RGB image
    (``Classifier overlay of <name>``, kept open but not activated, linked through its metadata) and an ``image``
    overlay item with the given opacity is put on the source – pinned to the classified plane, or covering every
    plane when the whole stack was classified (the Pixel Classifier panel keeps the overlay's slice in step).
    A previous classifier overlay of the image is replaced."""
    old = _find_overlay_image(session, img)
    rgb = clf.overlay_rgb(labels, colors)
    ov = LumenImage(rgb, name=f"Classifier overlay of {img.name}{suffix}", axes="TZCYX", rgb=True)
    ov.calibration = type(img.calibration)(**img.calibration.to_dict())
    ov.metadata["classifier_overlay_of"] = img.id
    ov.metadata["classes"] = ", ".join(names)
    ov.metadata["class_colors"] = ", ".join(colors)
    d = img.display
    ov.metadata["classifier_overlay_t"] = -1 if all_planes else int(d.t)
    ov.metadata["classifier_overlay_z"] = -1 if all_planes else int(d.z)
    if pin_t is not None:  # one frame of a time series, every slice (3-D classifier): frame pinned, z free
        ov.metadata["classifier_overlay_t"], ov.metadata["classifier_overlay_z"] = int(pin_t), -1
    session.add_image(ov, activate=False)
    if old is not None and old.id in session.images:
        session.close_image(old.id)
    img.overlay = [o for o in img.overlay if o.get("source") != OVERLAY_SOURCE] + [_overlay_item(ov, opacity)]
    return ov


def _outputs(session, img: LumenImage, model: dict[str, Any], probs: np.ndarray, output: str, mask_class: str = "",
             probability_threshold: float = 0.5, overlay_opacity: float = 0.33, all_planes: bool = False) -> tuple[list[LumenImage], str]:
    """The result images of a classification: probability maps, labels, the class overlay and / or a class mask."""
    names: list[str] = list(model["classes"])
    colors = clf.parse_colors(model.get("colors"), len(names))
    K = len(names)
    new: list[LumenImage] = []
    lab = probs.argmax(axis=2)[:, :, None].astype(np.uint8)
    if output in ("Probabilities", "Both", "All"):
        new.append(_prob_image(probs, names, f"Probabilities of {img.name}", img))
    if output in ("Labels", "Both", "All"):
        lim = label_image(lab, f"Classification of {img.name}", like=img)
        lim.display.channels[0].max = float(max(K - 1, 1))
        lim.metadata["classes"] = ", ".join(names)
        lim.metadata["class_colors"] = ", ".join(colors)
        new.append(lim)
    if output in ("Overlay", "All"):
        if session is None:
            raise ValueError("The Overlay output needs a session")
        _attach_overlay(session, img, lab, names, colors, all_planes, overlay_opacity)
    if mask_class:
        k = _class_of(mask_class, names)
        if k is None:
            raise ValueError(f"Unknown class {mask_class!r}; classes are {', '.join(names)}")
        m = probs[:, :, k] >= probability_threshold
        new.append(binary_image(m[:, :, None], f"{names[k]} mask of {img.name}", like=img))
    frac = lab[:, :, 0]
    summary = ", ".join(f"{names[k]} {100.0 * float((frac == k).mean()):.1f}%" for k in range(K))
    trained_dtype = model.get("trained_dtype")
    if trained_dtype and trained_dtype != str(img.dtype):
        summary += (f" (note: the classifier was trained on a {trained_dtype} image and this image is {img.dtype}; "
                    "the result is only meaningful when the intensities are on the same scale)")
    return new, summary


def _apply_model(img: LumenImage, model: dict[str, Any], all_planes: bool, output: str, smooth: float = 0.0,
                 mask_class: str = "", probability_threshold: float = 0.5, ctx=None, session=None,
                 overlay_opacity: float = 0.33) -> tuple[list[LumenImage], str]:
    """Classify ``img`` (current plane or every plane) and build the requested outputs.  ``ctx`` (an OpContext) is
    optional: with it, classification reports progress per plane and honours cancellation; ``session`` (or
    ``ctx.session``) is needed for the Overlay output."""
    probs = _predict_probs(img, model, all_planes, smooth=smooth, ctx=ctx)
    return _outputs(session or (ctx.session if ctx is not None else None), img, model, probs, output, mask_class=mask_class,
                    probability_threshold=probability_threshold, overlay_opacity=overlay_opacity, all_planes=all_planes)


def _collect_training_rows(img: LumenImage, pairs: list[tuple[int, dict[str, Any]]], K: int, settings: dict[str, Any], ctx=None
                           ) -> tuple[list[list[np.ndarray]], list[str], int]:
    """Feature rows under the class ROIs, grouped per class (features are computed once per annotated plane)."""
    H, W = img.height, img.width
    by_plane: dict[tuple[int, int], list[tuple[int, dict[str, Any]]]] = {}
    for k, r in pairs:
        by_plane.setdefault(_roi_plane(r, img), []).append((k, r))
    X_parts: list[list[np.ndarray]] = [[] for _ in range(K)]
    feature_names: list[str] = []
    for i, ((t, z), items) in enumerate(by_plane.items()):
        if ctx is not None:
            ctx.check_cancel()
            ctx.progress(0.6 * i / len(by_plane), f"Features of training plane {i + 1}/{len(by_plane)}")
        F, feature_names = _image_features(img, t, z, settings)
        lab = np.full((H, W), -1, np.int16)
        for k, r in items:
            lab[roimod.to_mask(r, (H, W))] = k
        flat = F.reshape(-1, F.shape[-1])
        for k in range(K):
            idx = np.flatnonzero(lab.ravel() == k)
            if idx.size:
                X_parts[k].append(flat[idx])
    return X_parts, feature_names, len(by_plane)


_CLASSIFIER_PARAMS = [
    Param("classifier", "choice", "Random forest", choices=CLASSIFIER_CHOICES,
          help="Random forest (Weka's FastRandomForest), Extra trees, histogram Gradient boosting, RBF SVM (training capped at 'SVM max samples'), "
               "k-nearest neighbours, Logistic regression or a one-hidden-layer MLP"),
    Param("n_trees", "int", 100, min=1, max=1000, label="Number of trees", help="Trees of the forests / boosting iterations"),
    Param("max_depth", "int", 0, min=0, max=100, help="Maximum tree depth, 0 = unlimited"),
    Param("features_per_node", "int", 0, min=0, max=10000, advanced=True, help="Random features tried at every split of a forest tree, 0 = square root of the feature count"),
    Param("balance_classes", "bool", True, help="Weight the classes inversely to their training counts (k-NN / MLP: subsample every class to the smallest one)"),
    Param("svm_max_samples", "int", 4000, min=100, max=200000, advanced=True, help="SVM: at most this many training samples (stratified subsample) – the RBF kernel is quadratic in the sample count"),
    Param("k_neighbors", "int", 5, min=1, max=100, advanced=True, help="k-NN: number of neighbours"),
    Param("hidden_units", "int", 64, min=1, max=1024, advanced=True, help="MLP: units of the hidden layer"),
]


def _fit_and_store(ctx, img: LumenImage, kind: str, key: str, prefix: str, names: list[str], colors: list[str], settings: dict[str, Any],
                   X_parts: list[list[np.ndarray]], feature_names: list[str], classifier: str, n_trees: int, max_depth: int,
                   features_per_node: int, balance_classes: bool, max_per_class: int, svm_max_samples: int, k_neighbors: int,
                   hidden_units: int, extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shared tail of the 2-D / 3-D training commands: assemble the samples, fit, store the model under ``key`` and
    publish its summary tables.  Returns ``(model, results_row)``."""
    K = len(names)
    classifier = clf.normalize_classifier(classifier)
    counts_raw = [sum(int(p.shape[0]) for p in X_parts[k]) for k in range(K)]
    missing = [names[k] for k in range(K) if counts_raw[k] == 0]
    if missing:
        unit = "voxels" if kind == "classifier3d" else "pixels"
        raise ValueError(f"No training {unit} for class(es): {', '.join(missing)}. Name ROIs after every class ({', '.join(names)}).")
    sample_balance = bool(balance_classes) and classifier in ("k-NN", "MLP")
    X, y, counts = clf.assemble_training_set(X_parts, int(max_per_class), sample_balance)
    if ctx is not None:
        ctx.check_cancel()
        ctx.progress(0.65, f"Training {classifier} on {X.shape[0]} samples")
    t0 = time.perf_counter()
    est, acc, acc_kind, n_used = clf.fit_classifier(X, y, classifier, n_trees=n_trees, max_depth=max_depth, features_per_node=features_per_node,
                                                     balance_classes=balance_classes, svm_max_samples=svm_max_samples, k_neighbors=k_neighbors,
                                                     hidden_units=hidden_units)
    dt = clf.elapsed(t0)
    params: dict[str, Any] = {"n_trees": int(n_trees), "max_depth": int(max_depth), "balance_classes": bool(balance_classes)}
    if classifier in ("Random forest", "Extra trees"):
        params["features_per_node"] = int(features_per_node) or "sqrt"
    if classifier == "SVM":
        params["svm_max_samples"] = int(svm_max_samples)
    if classifier == "k-NN":
        params["k_neighbors"] = int(k_neighbors)
    if classifier == "MLP":
        params["hidden_units"] = int(hidden_units)
    model: dict[str, Any] = {"kind": kind, "version": MODEL_VERSION, "model": est, "classifier": classifier, "classes": names, "colors": colors,
                             "settings": settings, "n_features": int(X.shape[1]), "feature_names": feature_names, "trained_on": img.name,
                             "trained_dtype": str(img.dtype), "n_samples": counts, "n_fitted": n_used, "train_time": float(dt), "accuracy": acc,
                             "accuracy_kind": acc_kind, "params": params, "importances": clf.feature_importances(est, int(X.shape[1])),
                             "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if extra:
        model.update(extra)
    ctx.session.settings[key] = model
    row: dict[str, Any] = {"Label": f"{'3D classifier' if kind == 'classifier3d' else 'Classifier'} ({img.name})", "Classifier": classifier, "Classes": K,
                           "Features": int(X.shape[1]), "Trees": int(n_trees), f"Training {'voxels' if kind == 'classifier3d' else 'pixels'}": int(X.shape[0]),
                           "Training time (s)": round(dt, 3), "OOB accuracy": acc if acc_kind == "out-of-bag" else float("nan"),
                           "Holdout accuracy": acc if acc_kind == "hold-out" else float("nan")}
    return model, row


# ======================================================================= pixel classifier
@op("segment.train_classifier", "Train Pixel Classifier", CAT, kind="analysis", preview=False, roi_aware=False, supports_rgb=True,
    description="Weka-style trainable segmentation: learn a classifier (random forest by default) from ROI-manager regions named after the "
                "classes – or the Pixel Classifier panel's 'class:<name>' traces – on the Weka feature bank, then classify every pixel of "
                "the current plane into probability maps, a label image or a transparent class overlay.",
    keywords=["weka", "machine learning", "random forest", "trainable", "pixel classification", "ilastik"],
    params=[Param("classes", "string", "background,foreground", help="Comma-separated class names. A ROI whose name starts with a class name (or is 'class:<name>') is a training region for that class (first class = background = label 0)"),
            Param("channels", "choice", "Current channel", choices=["Current channel", "All channels"], help="Compute the features on the current channel only or on every channel"),
            Param("features", "multichoice", DEFAULT_FEATURES, choices=FEATURE_CHOICES, help="Weka's training features; every scale-dependent one is computed at the sigmas min, 2·min, 4·min … max"),
            Param("min_sigma", "float", 1.0, min=0.5, max=32.0, step=0.5, help="Smallest feature scale (pixels)"),
            Param("max_sigma", "float", 8.0, min=0.5, max=32.0, step=1.0, help="Feature scales min, 2·min, 4·min … up to this sigma (pixels)"),
            Param("membrane_thickness", "int", 1, min=1, max=15, advanced=True, help="Membrane projections / Kuwahara: width of the line kernel (pixels)"),
            Param("membrane_patch_size", "int", 19, min=3, max=99, advanced=True, help="Membrane projections / Kuwahara: size of the line kernel (pixels)"),
            *_CLASSIFIER_PARAMS,
            Param("max_pixels_per_class", "int", 50000, min=100, max=5_000_000, label="Max samples per class", advanced=True, help="Classes with more training pixels are randomly subsampled"),
            Param("class_colors", "string", "", advanced=True, help="Comma-separated hex colours of the classes (empty = Weka's red, green, blue, cyan, magenta …); used by the Overlay output"),
            Param("apply_now", "bool", True, help="Classify the current plane right after training"),
            Param("output", "choice", "Both", choices=OUTPUT_CHOICES, help="Probability maps (one channel per class), a label image, both, or a transparent class-colour overlay on the image"),
            Param("overlay_opacity", "float", 0.33, min=0.05, max=1.0, step=0.05, advanced=True)])
def train_classifier(ctx, classes="background,foreground", channels="Current channel", features=None, min_sigma=1.0, max_sigma=8.0, membrane_thickness=1,
                     membrane_patch_size=19, classifier="Random forest", n_trees=100, max_depth=0, features_per_node=0, balance_classes=True,
                     svm_max_samples=4000, k_neighbors=5, hidden_units=64, max_pixels_per_class=50000, class_colors="", apply_now=True, output="Both",
                     overlay_opacity=0.33):
    img = ctx.img
    names = _parse_classes(classes)
    K = len(names)
    colors = clf.parse_colors(class_colors, K)
    settings = _feature_settings(channels, features if features is not None else DEFAULT_FEATURES, max_sigma, dtype=img.dtype, min_sigma=min_sigma,
                                 membrane_thickness=membrane_thickness, membrane_patch_size=membrane_patch_size)
    pairs = _class_rois(ctx.session, img, names)
    if not pairs:
        raise ValueError(f"No training ROIs found. Add ROIs to the ROI manager named after the classes ({', '.join(names)}), e.g. 'foreground 1' or 'class:foreground'.")
    X_parts, feature_names, n_planes = _collect_training_rows(img, pairs, K, settings, ctx=ctx)
    model, row = _fit_and_store(ctx, img, MODEL_KIND_2D, CLASSIFIER_KEY, "Classifier", names, colors, settings, X_parts, feature_names, classifier,
                                n_trees, max_depth, features_per_node, balance_classes, max_pixels_per_class, svm_max_samples, k_neighbors, hidden_units,
                                extra={"training_planes": n_planes})
    for k in range(K):
        row[f"Pixels '{names[k]}'"] = model["n_samples"][k]
    acc = model["accuracy"]
    acc_s = f"; {model['accuracy_kind']} accuracy {acc:.3f}" if math.isfinite(acc) else ""
    msg = f"Trained {model['classifier']} on {model['n_fitted']} pixels, {model['n_features']} features in {model['train_time']:.2f} s{acc_s}"
    res = OpResult(results=[row], table=clf.publish_tables(ctx.session, model, "Classifier"), message=msg, log=msg)
    if apply_now:
        ctx.progress(0.8, "Classifying the current plane")
        new, summary = _apply_model(img, model, False, output, session=ctx.session, overlay_opacity=overlay_opacity)
        res.new_images = new
        res.message = f"{msg}. {summary}"
    return res


@op("segment.apply_classifier", "Apply Pixel Classifier", CAT, kind="analysis", preview=False, roi_aware=False, supports_rgb=True,
    description="Classify the current image – or another open image – with the trained (or loaded) pixel classifier: the current plane or the "
                "whole stack, into probability maps, a label image or a transparent class overlay.",
    keywords=["weka", "classify", "predict", "probability", "overlay"],
    params=[Param("image", "image", None, help="Image to classify (empty = the current image)"),
            Param("all_planes", "bool", False, label="Process whole stack"),
            Param("output", "choice", "Both", choices=OUTPUT_CHOICES, help="Probability maps (one channel per class), a label image, both, or a transparent class-colour overlay on the image"),
            Param("smooth", "float", 0.0, min=0.0, max=10.0, step=0.5, help="Gaussian sigma applied to the probability maps before choosing the most likely class"),
            Param("mask_class", "string", "", help="Optional: name of a class to threshold into an extra binary mask image"),
            Param("probability_threshold", "float", 0.5, min=0.0, max=1.0, step=0.05),
            Param("overlay_opacity", "float", 0.33, min=0.05, max=1.0, step=0.05, advanced=True)])
def apply_classifier(ctx, image=None, all_planes=False, output="Both", smooth=0.0, mask_class="", probability_threshold=0.5, overlay_opacity=0.33):
    img = ctx.session.get_image(image) if image else ctx.img
    model = _get_model(ctx.session)
    new, summary = _apply_model(img, model, all_planes, output, smooth=smooth, mask_class=mask_class, probability_threshold=probability_threshold, ctx=ctx,
                                session=ctx.session, overlay_opacity=overlay_opacity)
    return OpResult(new_images=new, message=summary, log=f"Applied classifier to {img.name}: {summary}")


@op("segment.save_classifier", "Save Classifier", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    description="Write the trained pixel classifier (estimator, feature settings, class names and colours, training summary) to a .joblib file "
                "that loads on any platform.",
    keywords=["weka", "classifier", "save", "model"],
    params=[Param("path", "string", "classifier.joblib", help="File to write (a directory gets classifier.joblib inside)")])
def save_classifier(ctx, path="classifier.joblib"):
    model = _get_model(ctx.session)
    out = clf.save_model(model, path, "classifier.joblib")
    model["path"] = str(out)
    msg = f"Saved classifier ({', '.join(model['classes'])}) to {out}"
    return OpResult(message=msg, log=msg)


@op("segment.load_classifier", "Load Classifier", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    description="Load a pixel classifier saved with Save Classifier so that Apply Pixel Classifier can use it.",
    keywords=["weka", "classifier", "load", "model"],
    params=[Param("path", "string", "classifier.joblib")])
def load_classifier(ctx, path="classifier.joblib"):
    obj = clf.load_model(path)
    if obj.get("kind") == "classifier3d":
        raise ValueError(f"{path} is a 3-D pixel classifier – load it with Plugins ▸ Segmentation ▸ Load 3D Classifier")
    ctx.session.settings[CLASSIFIER_KEY] = obj
    table = clf.publish_tables(ctx.session, obj, "Classifier")
    msg = f"Loaded classifier ({', '.join(obj['classes'])}; {obj.get('n_features', '?')} features, trained on {obj.get('trained_on', '?')})"
    return OpResult(table=table, message=msg, log=msg)


@op("segment.classifier_info", "Classifier Info", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    description="Summary of the trained / loaded pixel classifier: feature importances table, class list with colours and training sample "
                "counts, classifier settings and training time (the 'Classifier', 'Classifier classes' and 'Classifier summary' tables).",
    keywords=["weka", "classifier", "importances", "features", "classes", "info"],
    params=[Param("kind", "choice", "2-D", choices=["2-D", "3-D"], help="The 2-D pixel classifier or the 3-D voxel classifier")])
def classifier_info(ctx, kind="2-D"):
    if kind == "3-D":
        from .segmentation3d import CLASSIFIER3D_KEY, _get_model3d
        model, prefix = _get_model3d(ctx.session), "Classifier 3D"
        key = CLASSIFIER3D_KEY
    else:
        model, prefix, key = _get_model(ctx.session), "Classifier", CLASSIFIER_KEY
    table = clf.publish_tables(ctx.session, model, prefix)
    msg = clf.describe(model)
    return OpResult(table=table, message=msg, log=f"{prefix} ({key}): {msg}")


@op("segment.classifier_overlay", "Classifier Overlay", CAT, kind="analysis", preview=False, roi_aware=False, supports_rgb=True,
    description="Show, hide or remove the transparent class overlay the pixel classifier put on the current image (Weka's 'Toggle overlay'), "
                "and set its opacity.",
    keywords=["weka", "overlay", "toggle", "classifier", "opacity"],
    params=[Param("action", "choice", "Toggle", choices=["Toggle", "Show", "Hide", "Remove"], help="Remove also closes the overlay colour image"),
            Param("opacity", "float", 0.33, min=0.05, max=1.0, step=0.05)])
def classifier_overlay(ctx, action="Toggle", opacity=0.33):
    img = ctx.img
    ov = _find_overlay_image(ctx.session, img)
    shown = any(o.get("source") == OVERLAY_SOURCE for o in img.overlay)
    keep = [o for o in img.overlay if o.get("source") != OVERLAY_SOURCE]
    if action == "Remove":
        if ov is not None and ov.id in ctx.session.images:
            ctx.session.close_image(ov.id)
        return OpResult(overlay=keep, message="Classifier overlay removed")
    show = action == "Show" or (action == "Toggle" and not shown)
    if show:
        if ov is None:
            raise ValueError("No classifier overlay for this image – apply the classifier with output 'Overlay' first")
        return OpResult(overlay=keep + [_overlay_item(ov, opacity)], message=f"Classifier overlay shown (opacity {float(opacity):.2f})")
    return OpResult(overlay=keep, message="Classifier overlay hidden")


def _trace_rois(session, img: LumenImage, class_name: str = "") -> list[dict[str, Any]]:
    """The ROI-manager traces ('class:<name>') of ``img`` (all classes, or one)."""
    want = f"class:{class_name.strip().lower()}" if class_name.strip() else None
    out = []
    for r in _rois_for_image(session, img):
        n = str(r.get("name", "")).strip().lower()
        if n.startswith("class:") and (want is None or n == want):
            out.append(r)
    return out


@op("segment.trace_tool", "Classifier Trace Tool", CAT, kind="analysis", preview=False, roi_aware=False, scope_param=False, supports_rgb=True,
    tool=True, tool_icon="✎", tool_cursor="crosshair",
    keywords=["weka", "trace", "pen", "tool", "annotate", "class", "scribble", "training"],
    params=[Param("class_name", "string", "foreground", help="The class the trace belongs to; the ROI is named 'class:<name>' (the training convention)"),
            Param("width", "float", 6.0, min=1.0, max=200.0, step=1.0, unit="px", help="Pen width of the trace"),
            Param("color", "color", "#00ff00", help="Colour of the trace (the class colour)")],
    description="A plugin tool (tool rail) for Weka-style training traces: drag on the image to add a pen stroke, click to add a dot – each "
                "becomes a ROI-manager entry 'class:<name>' on the current plane, in the class colour; ⌥-click removes the nearest trace of "
                "the class. The Pixel Classifier panel sets the class, width and colour.")
def trace_tool(ctx, class_name="foreground", width=6.0, color="#00ff00", x=0.0, y=0.0, x2=None, y2=None, button=0, shift=False, alt=False, ctrl=False,
               event="click"):
    img = ctx.img
    if event == "drag-start":
        return OpResult()
    name = str(class_name or "").strip()
    if not name:
        raise ValueError("Give the class the trace belongs to (class_name)")
    d = img.display
    session = ctx.session
    if alt or button == 2:
        traces = [r for r in _trace_rois(session, img, name) if _on_current_plane(r, img)]
        if not traces:
            raise ValueError(f"No '{name}' trace on this plane to remove")
        def _dist(r):
            bx, by, bw, bh = roimod.bounding_box(r)
            return math.hypot(bx + bw / 2 - x, by + bh / 2 - y)
        nearest = min(traces, key=_dist)
        session.roi_manager.remove([nearest["id"]])
        session._changed("rois")
        return OpResult(message=f"Removed a '{name}' trace ({len(traces) - 1} left on this plane)")
    w = float(width)
    if x2 is None or y2 is None or math.hypot(float(x2) - float(x), float(y2) - float(y)) < 0.5:
        roi: dict[str, Any] = {"type": "oval", "x": float(x) - w / 2, "y": float(y) - w / 2, "w": w, "h": w}
    else:
        roi = {"type": "polyline", "points": [[float(x), float(y)], [float(x2), float(y2)]], "width": w}
    roi.update(name=f"class:{name}", color=str(color or "#00ff00"))
    session.roi_manager.add(roi, image_id=img.id, position={"t": int(d.t), "z": int(d.z), "c": int(d.c)})
    session._changed("rois")
    n = len(_trace_rois(session, img, name))
    return OpResult(message=f"'{name}' trace added ({n} trace{'s' if n != 1 else ''} of this class)")


@op("segment.apply_classifier_folder", "Apply Pixel Classifier to Folder...", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    description="Batch: classify every image file of a folder with the trained / loaded pixel (or 3-D voxel) classifier and save the label images, "
                "probability maps or class-colour images next to them (progress per file, cancellable).",
    keywords=["weka", "batch", "folder", "classify", "apply"],
    params=[Param("input_folder", "string", "", label="Input folder"),
            Param("pattern", "string", r"\.(?i:tiff?|png|jpe?g|nd2|czi|lif)$", label="File pattern (regex)", help="Only file names matching this regular expression are processed"),
            Param("recursive", "bool", False, label="Include subfolders"),
            Param("output_folder", "string", "", label="Output folder", help="Empty = <input folder>/lumen_classified"),
            Param("output", "choice", "Labels", choices=["Labels", "Probabilities", "Both", "Overlay"], help="Overlay = the class-colour RGB image"),
            Param("output_format", "choice", "tif", choices=["tif", "ome.tif", "png", "npy"]),
            Param("use_3d", "bool", False, label="Use the 3-D classifier", help="Classify each file's z stack with the 3-D voxel classifier instead of the 2-D one"),
            Param("all_planes", "bool", True, label="Process whole stacks", help="Off: only the first plane (frame) of every file"),
            Param("smooth", "float", 0.0, min=0.0, max=10.0, step=0.5, help="Gaussian sigma applied to the probability maps"),
            Param("continue_on_error", "bool", True), Param("max_files", "int", 0, min=0, help="0 = all")])
def apply_classifier_folder(ctx, input_folder="", pattern=r"\.(?i:tiff?|png|jpe?g|nd2|czi|lif)$", recursive=False, output_folder="", output="Labels",
                            output_format="tif", use_3d=False, all_planes=True, smooth=0.0, continue_on_error=True, max_files=0):
    from ..io import IMAGE_EXTS, _full_ext, list_sequence_files, read_image, write_image
    from .batch import _compile_pattern
    if use_3d:
        from .segmentation3d import _apply_model3d, _get_model3d
        model = _get_model3d(ctx.session)
    else:
        model = _get_model(ctx.session)
    in_dir = Path(str(input_folder or "")).expanduser()
    if not in_dir.is_dir():
        raise ValueError(f"Input folder not found: {in_dir}")
    files = list_sequence_files(in_dir, _compile_pattern(pattern or ""), sort="natural", recursive=recursive, exts=IMAGE_EXTS)
    out_dir = Path(str(output_folder or "")).expanduser() if str(output_folder or "").strip() else in_dir / "lumen_classified"
    files = [f for f in files if out_dir not in f.parents]
    if max_files and int(max_files) > 0:
        files = files[:int(max_files)]
    if not files:
        raise ValueError(f"No files in {in_dir} match /{pattern}/")
    out_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = list(model["classes"])
    colors = clf.parse_colors(model.get("colors"), len(names))
    rows: list[dict[str, Any]] = []
    n_ok = n_err = 0
    t_batch = time.perf_counter()
    for i, f in enumerate(files):
        ctx.check_cancel()
        ctx.progress(i / len(files), f"{f.name} ({i + 1}/{len(files)})")
        row: dict[str, Any] = {"Label": f.name, "File": str(f), "Status": "ok"}
        try:
            img = read_image(str(f))
            want = "Labels" if output == "Overlay" else output
            if use_3d:
                new, summary = _apply_model3d(img, model, bool(all_planes), want, smooth=smooth, ctx=ctx, session=None)
            else:
                probs = _predict_probs(img, model, bool(all_planes), smooth=smooth, ctx=ctx, progress=(i / len(files), (i + 1) / len(files)))
                new, summary = _outputs(None, img, model, probs, want, all_planes=bool(all_planes))
            rel = f.relative_to(in_dir) if recursive else Path(f.name)
            ext = _full_ext(f)
            base = f.name[:len(f.name) - len(ext)] if ext else f.stem
            written = []
            for im in new:
                tag = "labels" if im.name.startswith("Classification") else "probabilities"
                if output == "Overlay":
                    im = LumenImage(clf.overlay_rgb(im.data, colors), name=f"{base} overlay", axes="TZCYX", rgb=True)
                    tag = "overlay"
                dest = out_dir / rel.parent / f"{base}_{tag}.{output_format}"
                dest.parent.mkdir(parents=True, exist_ok=True)
                written.append(write_image(im, str(dest), output_format))
            row["Output"] = "; ".join(written)
            row["Classes"] = summary
            n_ok += 1
        except OperationCancelled:
            raise
        except Exception as e:  # noqa: BLE001 – one bad file must not kill the batch
            row["Status"] = f"error: {str(e)[:200]}"
            n_err += 1
            if not continue_on_error:
                rows.append(row)
                raise ValueError(f"{f.name}: {e}") from e
        rows.append(row)
    ctx.check_cancel()
    msg = f"Classified {n_ok} of {len(files)} file(s) into {out_dir} in {time.perf_counter() - t_batch:.1f} s" + (f" ({n_err} failed)" if n_err else "")
    return OpResult(results=rows, message=msg, log=msg)


# ========================================================================= random walker
@op("segment.random_walker", "Random Walker", CAT, kind="analysis", preview=False, supports_rgb=False,
    description="Random-walker segmentation from seeds: class ROIs in the ROI manager (named after the classes) or a marker label image.",
    keywords=["seeds", "markers", "graph", "diffusion"],
    params=[Param("classes", "string", "background,foreground", help="Seed ROIs in the ROI manager named after these classes (ignored when a markers image is given)"),
            Param("markers_image", "image", None, help="Optional label image with seeds (0 = unlabelled)"),
            Param("beta", "float", 130.0, min=1.0, max=100000.0, step=10.0, help="Edge penalty: higher = the walk stops more strictly at intensity edges. Raise it (1000–5000) when regions leak through blurred edges; every object needs its own seed"),
            Param("mode", "choice", "cg_j", choices=["cg_j", "cg", "bf"], advanced=True, help="Linear solver (bf = exact but slow)"),
            Param("smooth", "float", 0.0, min=0.0, max=10.0, step=0.5, advanced=True),
            Param("output", "choice", "Labels", choices=["Labels", "Labels and probabilities"])])
def random_walker(ctx, classes="background,foreground", markers_image=None, beta=130.0, mode="cg_j", smooth=0.0, output="Labels"):
    img = ctx.img
    d = img.display
    H, W = img.height, img.width
    p = _norm01(img.current_plane())
    if smooth > 0:
        p = ndi.gaussian_filter(p, smooth)
    names: list[str] | None = None
    if markers_image:
        mk = ctx.session.get_image(markers_image)
        mp = mk.data[min(d.t, mk.nt - 1), min(d.z, mk.nz - 1), 0]
        if mp.shape != (H, W):
            raise ValueError("The markers image must have the same width and height as the current image")
        markers = _as_labels(mp)
        offset = 0
    else:
        names = _parse_classes(classes)
        markers = np.zeros((H, W), np.int32)
        for k, r in _class_rois(ctx.session, img, names):
            if _on_current_plane(r, img):
                markers[roimod.to_mask(r, (H, W))] = k + 1
        offset = 1
    if ctx.mask is not None:
        markers = np.where(ctx.mask, markers, -1)  # seeds outside the selection are dropped, so mask FIRST
    # The classes that actually have seeds inside the selection; skimage renumbers whatever positive values it
    # receives to consecutive 1..n (ascending) and returns those, so everything below maps back through `present`.
    present = np.unique(markers[markers > 0])
    if present.size < 2:
        if ctx.mask is not None and present.size == 1:
            cls = names[int(present[0]) - 1] if names else f"label {int(present[0])}"
            raise ValueError(f"Random walker needs seeds of at least two classes inside the selection "
                             f"(only {cls!r} has seeds inside it)")
        raise ValueError("Random walker needs seeds of at least two classes (ROIs named after the classes, or a markers image with ≥ 2 labels)")
    # skimage returns the seeds unchanged (and, with probabilities, an (H, W, n) array instead of (n, H, W)) when
    # no unlabelled pixel is reachable from a seed – gap-free markers, or zeros cut off by the selection.
    reachable = ndi.binary_propagation(markers > 0, mask=markers >= 0) & (markers == 0)
    if not reachable.any():
        where = " inside the selection" if ctx.mask is not None else ""
        raise ValueError(f"Random walker has nothing to segment: every pixel{where} is already a seed "
                         "(the seeds / markers image must leave unlabelled 0 pixels next to them)")
    # Feed skimage consecutive seed values 1..n so its internal renumbering is the identity on every code path.
    lut = np.zeros(int(present.max()) + 1, np.int32)
    lut[present] = np.arange(1, present.size + 1, dtype=np.int32)
    ranked = np.where(markers > 0, lut[np.maximum(markers, 0)], markers).astype(np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = skseg.random_walker(p, ranked, beta=beta, mode=mode, return_full_prob=(output != "Labels"))
    new: list[LumenImage] = []
    if output != "Labels":
        probs = np.asarray(res, np.float32)  # (present.size, H, W) in ascending order of `present`
        lab = present[np.argmax(probs, axis=0)].astype(np.int32)
        if ctx.mask is not None:
            lab[~ctx.mask] = 0
            probs[:, ~ctx.mask] = 0
        pnames = [names[int(v) - 1] if names else f"Label {int(v)}" for v in present]
        new.append(_prob_image(probs, pnames, f"Probabilities of {img.name}", img))
    else:
        r = np.asarray(res, np.int32)  # consecutive 1..n (-1 outside the selection) -> original class / marker values
        lab = np.where(r > 0, present[np.clip(r, 1, present.size) - 1], 0).astype(np.int32)
    if offset:
        lab = np.maximum(lab - 1, 0)
    lim = label_image(lab, f"Random walker of {img.name}", like=img)
    if names:
        lim.metadata["classes"] = ", ".join(names)
    lim.display.channels[0].max = float(max(int(lab.max()), 1))
    new.insert(0, lim)
    return OpResult(new_images=new, message=f"Random walker: {present.size} classes")


# ======================================================================= active contours
def _init_level_set(init: str, shape: tuple[int, int], ctx) -> np.ndarray | str:
    if init == "selection":
        if ctx.mask is None or not ctx.mask.any():
            raise ValueError("Initialise from selection: draw an area selection first (or choose checkerboard / disk)")
        return ctx.mask.copy()
    return init


def _orient(seg: np.ndarray, p: np.ndarray, dark_background: bool) -> np.ndarray:
    seg = seg.astype(bool)
    if seg.any() and (~seg).any():
        brighter_inside = float(p[seg].mean()) > float(p[~seg].mean())
        if brighter_inside != bool(dark_background):
            seg = ~seg
    return seg


@op("segment.active_contour", "Active Contour (Chan-Vese)", CAT, kind="analysis", preview=False, supports_rgb=False,
    description="Region-based level-set segmentation (Chan-Vese) of the current plane into a binary mask.", keywords=["level set", "snake", "chan vese"],
    params=[Param("mu", "float", 0.25, min=0.0, max=5.0, step=0.05, help="Edge length weight: higher = smoother contour"),
            Param("lambda1", "float", 1.0, min=0.0, max=10.0, step=0.1, help="Weight of the inside-region intensity difference"),
            Param("lambda2", "float", 1.0, min=0.0, max=10.0, step=0.1, help="Weight of the outside-region intensity difference"),
            Param("iterations", "int", 200, min=1, max=5000),
            Param("init", "choice", "checkerboard", choices=["checkerboard", "disk", "selection"], help="Initial level set; 'selection' uses the active area selection"),
            Param("tolerance", "float", 0.001, min=0.0, max=1.0, step=0.001, advanced=True), Param("dt", "float", 0.5, min=0.01, max=10.0, step=0.1, advanced=True),
            Param("dark_background", "bool", True, help="Report the brighter region as foreground"),
            Param("to_selection", "bool", False, label="Also create a selection")])
def active_contour(ctx, mu=0.25, lambda1=1.0, lambda2=1.0, iterations=200, init="checkerboard", tolerance=0.001, dt=0.5, dark_background=True, to_selection=False):
    img = ctx.img
    p = _norm01(img.current_plane())
    ls = _init_level_set(init, p.shape, ctx)
    if isinstance(ls, np.ndarray):
        ls = np.where(ls, 1.0, -1.0).astype(np.float32)
    seg = skseg.chan_vese(p, mu=mu, lambda1=lambda1, lambda2=lambda2, tol=tolerance, max_num_iter=int(iterations), dt=dt, init_level_set=ls)
    seg = _orient(np.asarray(seg), p, dark_background)
    res = OpResult(new_images=[binary_image(seg, f"Chan-Vese of {img.name}", like=img)], message=f"Chan-Vese: {int(seg.sum())} foreground pixels")
    if to_selection:
        res.roi = roimod.from_mask(seg, "Chan-Vese")
    return res


@op("segment.morph_snakes", "Morphological Snakes", CAT, kind="analysis", preview=False, supports_rgb=False,
    description="Morphological active contours (morphological Chan-Vese or geodesic active contour) started from the selection, a disk or a checkerboard.",
    keywords=["snake", "level set", "geodesic", "contour"],
    params=[Param("method", "choice", "Chan-Vese", choices=["Chan-Vese", "Geodesic active contour"]),
            Param("iterations", "int", 100, min=1, max=5000), Param("smoothing", "int", 1, min=0, max=10, help="Contour smoothing steps per iteration"),
            Param("init", "choice", "selection", choices=["selection", "disk", "checkerboard"]),
            Param("lambda1", "float", 1.0, min=0.0, max=10.0, step=0.1, help="Chan-Vese: inside weight"), Param("lambda2", "float", 1.0, min=0.0, max=10.0, step=0.1, help="Chan-Vese: outside weight"),
            Param("alpha", "float", 100.0, min=1.0, max=10000.0, step=10.0, help="Geodesic: edge sensitivity of the inverse Gaussian gradient"),
            Param("sigma", "float", 2.0, min=0.1, max=20.0, step=0.5, help="Geodesic: smoothing of the gradient image"),
            Param("balloon", "float", 0.0, min=-1.0, max=1.0, step=1.0, help="Geodesic: > 0 inflates, < 0 deflates the contour where edges are weak"),
            Param("dark_background", "bool", True, help="Chan-Vese: report the brighter region as foreground"),
            Param("to_selection", "bool", True, label="Also create a selection")])
def morph_snakes(ctx, method="Chan-Vese", iterations=100, smoothing=1, init="selection", lambda1=1.0, lambda2=1.0, alpha=100.0, sigma=2.0, balloon=0.0,
                 dark_background=True, to_selection=True):
    img = ctx.img
    p = _norm01(img.current_plane())
    ls = _init_level_set(init, p.shape, ctx)
    if isinstance(ls, np.ndarray):
        ls = ls.astype(np.int8)
    if method == "Chan-Vese":
        seg = skseg.morphological_chan_vese(p, num_iter=int(iterations), init_level_set=ls, smoothing=int(smoothing), lambda1=lambda1, lambda2=lambda2)
        seg = _orient(np.asarray(seg), p, dark_background)
        label = "Morphological Chan-Vese"
    else:
        g = skseg.inverse_gaussian_gradient(p, alpha=alpha, sigma=sigma)
        seg = skseg.morphological_geodesic_active_contour(g, num_iter=int(iterations), init_level_set=ls, smoothing=int(smoothing), balloon=balloon)
        seg = np.asarray(seg).astype(bool)
        label = "Geodesic active contour"
    res = OpResult(new_images=[binary_image(seg, f"{label} of {img.name}", like=img)], message=f"{label}: {int(seg.sum())} foreground pixels")
    if to_selection:
        res.roi = roimod.from_mask(seg, label)
    return res


# ============================================================================ superpixels
@op("segment.superpixels", "Superpixels", CAT, kind="analysis", preview=False, supports_rgb=True,
    description="Over-segment the current plane into superpixels (SLIC, Felzenszwalb, Quickshift or compact watershed) as a label image.",
    keywords=["slic", "felzenszwalb", "quickshift", "oversegmentation"],
    params=[Param("method", "choice", "SLIC", choices=["SLIC", "Felzenszwalb", "Quickshift", "Compact watershed"]),
            Param("n_segments", "int", 250, min=2, max=100000, help="SLIC / compact watershed: approximate number of segments"),
            Param("compactness", "float", 10.0, min=0.001, max=10000.0, step=1.0, help="SLIC: intensity vs. space weighting, higher = squarer (compact watershed uses compactness / 10000)"),
            Param("sigma", "float", 1.0, min=0.0, max=10.0, step=0.5, help="Pre-smoothing"),
            Param("scale", "float", 100.0, min=1.0, max=10000.0, step=10.0, help="Felzenszwalb: higher = larger segments", advanced=True),
            Param("min_size", "int", 20, min=1, max=100000, help="Felzenszwalb: minimum segment size (pixels)", advanced=True),
            Param("kernel_size", "float", 5.0, min=1.0, max=50.0, step=1.0, help="Quickshift: kernel width", advanced=True),
            Param("max_dist", "float", 10.0, min=1.0, max=100.0, step=1.0, help="Quickshift: cut-off distance", advanced=True),
            Param("ratio", "float", 0.5, min=0.0, max=1.0, step=0.1, help="Quickshift: colour vs. space weighting", advanced=True),
            Param("output", "choice", "Labels", choices=["Labels", "Boundaries", "Both"])])
def superpixels(ctx, method="SLIC", n_segments=250, compactness=10.0, sigma=1.0, scale=100.0, min_size=20, kernel_size=5.0, max_dist=10.0, ratio=0.5, output="Labels"):
    img = ctx.img
    d = img.display
    mask = ctx.mask
    if img.rgb:
        arr = np.moveaxis(img.data[d.t, d.z].astype(np.float32) / 255.0, 0, -1)
        multi = True
    else:
        arr = _norm01(img.current_plane())
        multi = False
    if method == "SLIC":
        x = arr if multi else arr * 100.0  # grey: same scale as the L channel of Lab, so compactness behaves like for colour
        lab = skseg.slic(x, n_segments=int(n_segments), compactness=compactness, sigma=sigma, channel_axis=-1 if multi else None, start_label=1, mask=mask)
    elif method == "Felzenszwalb":
        lab = skseg.felzenszwalb(arr, scale=scale, sigma=sigma, min_size=int(min_size), channel_axis=-1 if multi else None) + 1
    elif method == "Quickshift":
        a3 = arr if multi else np.stack([arr] * 3, axis=-1)
        lab = skseg.quickshift(a3, ratio=ratio, kernel_size=kernel_size, max_dist=max_dist, sigma=sigma, convert2lab=multi) + 1
    else:
        g = arr if not multi else arr.mean(axis=-1)
        edges = skfilt.sobel(ndi.gaussian_filter(g, sigma) if sigma > 0 else g)
        lab = skseg.watershed(edges, markers=int(n_segments), compactness=compactness / 10000.0, mask=mask)
    lab = np.asarray(lab, np.int32)
    if mask is not None:
        lab[~mask] = 0
    lab, _, _ = skseg.relabel_sequential(lab)
    n = int(lab.max())
    new: list[LumenImage] = []
    if output in ("Labels", "Both"):
        new.append(_any_count_label_image(lab, f"{method} of {img.name}", like=img))
    if output in ("Boundaries", "Both"):
        bd = skseg.find_boundaries(lab, mode="inner") & (lab > 0)
        new.append(binary_image(bd, f"{method} boundaries of {img.name}", like=img))
    return OpResult(new_images=new, message=f"{n} segments")


# ============================================================================ multi-Otsu
@op("segment.multi_otsu", "Multi-Otsu Threshold", CAT, kind="analysis", preview=False, supports_rgb=False,
    description="Split the intensities into several classes with the multi-Otsu method; makes a label image and reports the thresholds.",
    keywords=["threshold", "otsu", "classes"],
    params=[Param("classes", "int", 3, min=2, max=5), Param("output", "choice", "Labels", choices=["Labels", "Mask of brightest class", "Nothing"]),
            Param("all_planes", "bool", False, help="Apply the thresholds of the current plane to the whole stack", advanced=True)])
def multi_otsu(ctx, classes=3, output="Labels", all_planes=False):
    from skimage.filters import threshold_multiotsu
    img = ctx.img
    p = img.current_plane()
    vals = p[ctx.mask] if ctx.mask is not None else p
    if vals.dtype == np.float32:
        vals = vals[np.isfinite(vals)]
    try:
        thr = threshold_multiotsu(vals, classes=int(classes), nbins=256)
    except ValueError as e:
        raise ValueError(f"Multi-Otsu failed: {e}") from e
    row: dict[str, Any] = {"Label": img.name, "Classes": int(classes)}
    for i, t in enumerate(thr, 1):
        row[f"Threshold {i}"] = float(t)
    new: list[LumenImage] = []
    if output != "Nothing":
        src = img.data if all_planes else p
        lab = np.digitize(np.nan_to_num(src), thr).astype(np.uint8)
        if all_planes:
            lab = lab[:, :, [img.display.c]]
        if output == "Labels":
            lim = label_image(lab, f"Multi-Otsu of {img.name}", like=img)
            lim.display.channels[0].max = float(max(int(classes) - 1, 1))
            new.append(lim)
        else:
            new.append(binary_image(lab == int(classes) - 1, f"Multi-Otsu mask of {img.name}", like=img))
    msg = "Thresholds: " + ", ".join(f"{float(t):g}" for t in thr)
    return OpResult(results=[row], new_images=new, message=msg)


# ================================================================ deep learning (optional)
def _import_cellpose():
    try:
        from cellpose import models  # type: ignore
    except ImportError as e:
        raise ValueError("Cellpose is not installed. Lumen needs: pip install cellpose") from e
    return models


def _import_stardist():
    try:
        from stardist.models import StarDist2D  # type: ignore
    except ImportError as e:
        raise ValueError("StarDist is not installed. Lumen needs: pip install stardist tensorflow") from e
    return StarDist2D


def _label_output(ctx, img: LumenImage, labs: dict[tuple[int, int], np.ndarray], name: str, add_to_manager: bool) -> OpResult:
    """Common tail of the object-segmentation commands: label image (+ ROIs) from per-plane labels."""
    d = img.display
    H, W = img.height, img.width
    if len(labs) == 1:
        data = next(iter(labs.values())).astype(np.int32)
    else:
        data = np.zeros((img.nt, img.nz, 1, H, W), np.int32)
        for (t, z), lab in labs.items():
            data[t, z, 0] = lab
    total = sum(int(len(np.unique(l[l > 0]))) for l in labs.values())
    n_rois = 0
    if add_to_manager:
        for (t, z), lab in labs.items():
            rois = roimod.label_to_rois(lab.astype(np.int32))
            _name_rois_by_label(rois)
            n_rois += _add_rois(ctx, img, rois, t, z, d.c)
    out = label_image(data, name, like=img)
    msg = f"{total} objects" + (f", {n_rois} ROIs added" if add_to_manager else "")
    return OpResult(new_images=[out], message=msg, log=msg, results=[{"Label": img.name, "Count": total}])


def _cellpose_eval(net, x: np.ndarray, chans: list[int], diameter: float, flow_threshold: float, cellprob_threshold: float) -> np.ndarray:
    kwargs: dict[str, Any] = {"diameter": diameter or None, "flow_threshold": flow_threshold, "cellprob_threshold": cellprob_threshold}
    try:
        out = net.eval(x, channels=chans, **kwargs)
    except TypeError:  # cellpose ≥ 4 dropped the channels argument
        out = net.eval(x, **kwargs)
    masks = out[0] if isinstance(out, (tuple, list)) else out
    return np.asarray(masks, np.int32)


@op("segment.cellpose", "Cellpose", CAT, kind="analysis", preview=False, supports_rgb=True,
    description="Segment cells or nuclei with a Cellpose deep-learning model (needs the optional 'cellpose' package).",
    keywords=["deep learning", "cells", "nuclei", "cyto", "neural network"],
    params=[Param("model", "choice", "cyto3", choices=["cyto3", "cyto2", "cyto", "nuclei", "custom"]),
            Param("custom_model", "string", "", help="Path of a custom model file (model = custom)"),
            Param("diameter", "float", 0.0, min=0.0, max=5000.0, step=1.0, help="Expected object diameter in pixels, 0 = estimate automatically"),
            Param("cyto_channel", "int", 0, min=0, max=64, help="Channel (1-based) with the cytoplasm / objects, 0 = current channel (grayscale)"),
            Param("nuclei_channel", "int", 0, min=0, max=64, help="Optional nuclear channel (1-based), 0 = none"),
            Param("flow_threshold", "float", 0.4, min=0.0, max=3.0, step=0.1), Param("cellprob_threshold", "float", 0.0, min=-6.0, max=6.0, step=0.5),
            Param("use_gpu", "bool", False), Param("add_to_manager", "bool", True), Param("all_planes", "bool", False, label="Process whole stack")])
def cellpose(ctx, model="cyto3", custom_model="", diameter=0.0, cyto_channel=0, nuclei_channel=0, flow_threshold=0.4, cellprob_threshold=0.0, use_gpu=False,
             add_to_manager=True, all_planes=False):
    models = _import_cellpose()
    img = ctx.img
    d = img.display
    for label, ch in (("Cytoplasm channel", cyto_channel), ("Nuclei channel", nuclei_channel)):
        if int(ch) > img.nc:  # checked before the model is built so a typo does not first load / download weights
            raise ValueError(f"{label} {int(ch)} does not exist (image has {img.nc})")
    if model == "custom":
        if not custom_model:
            raise ValueError("Give the path of the custom Cellpose model")
        net = models.CellposeModel(gpu=use_gpu, pretrained_model=os.path.expanduser(custom_model))
    elif hasattr(models, "Cellpose"):
        net = models.Cellpose(gpu=use_gpu, model_type=model)
    else:  # cellpose ≥ 4: one generalist model
        try:
            net = models.CellposeModel(gpu=use_gpu, model_type=model)
        except TypeError:
            net = models.CellposeModel(gpu=use_gpu)
    planes = [(t, z) for t in range(img.nt) for z in range(img.nz)] if all_planes else [(d.t, d.z)]
    labs = {}
    for (t, z) in planes:
        if cyto_channel == 0 and nuclei_channel == 0:
            if img.rgb:
                x = np.moveaxis(img.data[t, z], 0, -1).astype(np.float32)
            else:
                x = img.data[t, z, d.c].astype(np.float32)
            chans = [0, 0]
        else:
            x = np.zeros((img.height, img.width, 3), np.float32)
            x[..., 0] = img.data[t, z, (cyto_channel - 1) if cyto_channel else d.c]
            if nuclei_channel:
                x[..., 1] = img.data[t, z, nuclei_channel - 1]
            chans = [1, 2 if nuclei_channel else 0]
        labs[(t, z)] = _cellpose_eval(net, x, chans, diameter, flow_threshold, cellprob_threshold)
    return _label_output(ctx, img, labs, f"Cellpose of {img.name}", add_to_manager)


@op("segment.stardist", "StarDist", CAT, kind="analysis", preview=False, supports_rgb=True,
    description="Segment star-convex objects (nuclei) with a StarDist 2D model (needs the optional 'stardist' package).",
    keywords=["deep learning", "nuclei", "star convex", "neural network"],
    params=[Param("model", "choice", "2D_versatile_fluo", choices=["2D_versatile_fluo", "2D_versatile_he", "2D_paper_dsb2018", "custom"]),
            Param("custom_model", "string", "", help="Folder of a custom StarDist model (model = custom)"),
            Param("prob_thresh", "float", 0.5, min=0.0, max=1.0, step=0.05, label="Probability threshold"),
            Param("nms_thresh", "float", 0.4, min=0.0, max=1.0, step=0.05, label="NMS overlap threshold"),
            Param("normalize", "bool", True, help="Percentile-normalise (1–99.8 %) before prediction"),
            Param("n_tiles", "int", 1, min=1, max=64, advanced=True, help="Tiles per axis for large images"),
            Param("add_to_manager", "bool", True), Param("all_planes", "bool", False, label="Process whole stack")])
def stardist(ctx, model="2D_versatile_fluo", custom_model="", prob_thresh=0.5, nms_thresh=0.4, normalize=True, n_tiles=1, add_to_manager=True, all_planes=False):
    StarDist2D = _import_stardist()
    img = ctx.img
    d = img.display
    if model == "custom":
        if not custom_model:
            raise ValueError("Give the folder of the custom StarDist model")
        folder = os.path.expanduser(custom_model.rstrip("/\\"))
        net = StarDist2D(None, name=os.path.basename(folder), basedir=os.path.dirname(folder) or ".")
    else:
        net = StarDist2D.from_pretrained(model)
    rgb_model = model == "2D_versatile_he"
    planes = [(t, z) for t in range(img.nt) for z in range(img.nz)] if all_planes else [(d.t, d.z)]
    labs = {}
    for (t, z) in planes:
        if rgb_model:
            x = np.moveaxis(img.data[t, z], 0, -1).astype(np.float32) if img.rgb else np.stack([img.data[t, z, d.c].astype(np.float32)] * 3, axis=-1)
        else:
            x = img.data[t, z, d.c].astype(np.float32)
        if normalize:
            try:
                from csbdeep.utils import normalize as cs_normalize  # type: ignore
                x = cs_normalize(x, 1, 99.8, axis=(0, 1))
            except ImportError:
                lo, hi = np.percentile(x, (1, 99.8))
                x = (x - lo) / max(hi - lo, 1e-6)
        tiles = (n_tiles, n_tiles, 1) if x.ndim == 3 else (n_tiles, n_tiles)
        lab, _details = net.predict_instances(x, prob_thresh=prob_thresh, nms_thresh=nms_thresh, n_tiles=tiles)
        labs[(t, z)] = np.asarray(lab, np.int32)
    return _label_output(ctx, img, labs, f"StarDist of {img.name}", add_to_manager)


# ====================================================================== labels ⇄ ROIs
@op("segment.labels_to_rois", "Labels to ROIs", CAT, kind="analysis", preview=False, supports_rgb=False,
    description="Add one ROI per object of the current label image (or 0/255 mask) to the ROI manager, named L<label>.",
    keywords=["label", "roi manager", "outline", "convert"],
    params=[Param("min_size", "int", 0, min=0, max=100000000, help="Skip objects smaller than this (pixels)"),
            Param("simplify", "bool", True, help="Polygon outlines (off: exact pixel masks)"),
            Param("clear_manager", "bool", False, help="Empty the ROI manager first")])
def labels_to_rois(ctx, min_size=0, simplify=True, clear_manager=False):
    img = ctx.img
    d = img.display
    lab = _as_labels(img.current_plane(), binary_to_components=True)
    if ctx.mask is not None:
        lab = np.where(ctx.mask, lab, 0)
    if min_size > 0 and lab.max() > 0:
        sizes = np.bincount(lab.ravel())
        lab = np.where(sizes[lab] >= min_size, lab, 0)
    rois = roimod.label_to_rois(lab, simplify=simplify)
    _name_rois_by_label(rois)
    if clear_manager:
        ctx.session.roi_manager.clear()
    n = _add_rois(ctx, img, rois, d.t, d.z, d.c)
    msg = f"{n} ROIs added to the ROI manager"
    return OpResult(message=msg, log=msg)


@op("segment.rois_to_labels", "ROIs to Labels", CAT, kind="analysis", preview=False, supports_rgb=True, roi_aware=False,
    description="Rasterise the area ROIs of the ROI manager into a new label image (label = position in the manager; later ROIs overwrite earlier ones).",
    keywords=["label", "roi manager", "rasterise", "convert"],
    params=[Param("use_positions", "bool", True, help="Stacks: draw each ROI on the plane stored in its position (off: everything on one plane)")])
def rois_to_labels(ctx, use_positions=True):
    img = ctx.img
    H, W = img.height, img.width
    rois = [r for r in _rois_for_image(ctx.session, img) if roimod.is_area(r)]
    if not rois:
        raise ValueError("The ROI manager has no area ROIs")
    stack = bool(use_positions) and (img.nt > 1 or img.nz > 1)
    lab = np.zeros((img.nt if stack else 1, img.nz if stack else 1, 1, H, W), np.int32)
    for i, r in enumerate(rois, 1):
        t, z = _roi_plane(r, img) if stack else (0, 0)
        lab[t, z, 0][roimod.to_mask(r, (H, W))] = i
    out = label_image(lab, f"Labels from ROIs ({img.name})", like=img)
    msg = f"{len(rois)} ROIs rasterised"
    return OpResult(new_images=[out], message=msg)


# ======================================================================= label filtering
@op("segment.label_filter", "Filter Labels", CAT, kind="analysis", preview=False, supports_rgb=False,
    description="Keep only the objects of a label image that pass area / circularity / solidity / mean-intensity criteria, relabelled sequentially into a new image.",
    keywords=["label", "filter", "size", "circularity", "solidity"],
    params=[Param("min_area", "float", 0.0, min=0.0, step=1.0, label="Min area (px)"), Param("max_area", "float", 0.0, min=0.0, step=1.0, label="Max area (px, 0 = ∞)"),
            Param("min_circularity", "float", 0.0, min=0.0, max=1.0, step=0.05), Param("max_circularity", "float", 1.0, min=0.0, max=1.0, step=0.05),
            Param("min_solidity", "float", 0.0, min=0.0, max=1.0, step=0.05), Param("max_solidity", "float", 1.0, min=0.0, max=1.0, step=0.05),
            Param("intensity_image", "image", None, help="Image providing the gray values for the mean-intensity criterion"),
            Param("min_mean", "float", 0.0, step=1.0, label="Min mean intensity"), Param("max_mean", "float", 0.0, step=1.0, label="Max mean intensity (0 = ∞)"),
            Param("exclude_edges", "bool", False, help="Drop objects touching the image border"),
            Param("all_planes", "bool", False, label="Process whole stack")])
def label_filter(ctx, min_area=0.0, max_area=0.0, min_circularity=0.0, max_circularity=1.0, min_solidity=0.0, max_solidity=1.0, intensity_image=None,
                 min_mean=0.0, max_mean=0.0, exclude_edges=False, all_planes=False):
    img = ctx.img
    d = img.display
    H, W = img.height, img.width
    inten = ctx.session.get_image(intensity_image) if intensity_image else None
    if inten is not None and (inten.height, inten.width) != (H, W):
        raise ValueError("The intensity image must have the same width and height as the label image")
    planes = [(t, z) for t in range(img.nt) for z in range(img.nz)] if all_planes else [(d.t, d.z)]
    out = np.zeros((img.nt if all_planes else 1, img.nz if all_planes else 1, 1, H, W), np.int32)
    kept = total = 0
    for (t, z) in planes:
        lab = _as_labels(img.data[t, z, d.c], binary_to_components=True)
        if exclude_edges:
            lab = skseg.clear_border(lab)
        ip = inten.data[min(t, inten.nt - 1), min(z, inten.nz - 1), min(d.c, inten.nc - 1)].astype(np.float32) if inten is not None else None
        keep = np.zeros(int(lab.max()) + 1, bool)
        for rp in skmeas.regionprops(lab, intensity_image=ip):
            total += 1
            area = float(rp.area)
            if area < min_area or (max_area > 0 and area > max_area):
                continue
            if min_circularity > 0 or max_circularity < 1:
                per = skmeas.perimeter(rp.image, neighborhood=4)
                circ = min(1.0, 4 * math.pi * area / (per * per)) if per > 0 else 1.0
                if circ < min_circularity or circ > max_circularity:
                    continue
            if min_solidity > 0 or max_solidity < 1:
                try:
                    sol = float(rp.solidity)
                except Exception:
                    sol = 1.0
                if sol < min_solidity or sol > max_solidity:
                    continue
            if ip is not None and (min_mean > 0 or max_mean > 0):
                mean = float(rp.intensity_mean)
                if mean < min_mean or (max_mean > 0 and mean > max_mean):
                    continue
            keep[rp.label] = True
            kept += 1
        filtered = np.where(keep[lab], lab, 0)
        filtered, _, _ = skseg.relabel_sequential(filtered)
        out[t if all_planes else 0, z if all_planes else 0, 0] = filtered
    res_img = label_image(out, f"Filtered labels of {img.name}", like=img)
    msg = f"Kept {kept} of {total} objects"
    return OpResult(new_images=[res_img], message=msg, log=msg)


# ==================================================================== label editing (plane ops)
@op("segment.expand_labels", "Expand Labels", CAT, kind="plane", supports_rgb=False,
    description="Grow every label outwards by a distance without overlapping its neighbours (like a dilation that respects other labels).",
    keywords=["label", "dilate", "grow"], params=[Param("distance", "float", 2.0, min=0.0, max=500.0, step=1.0, unit="px")])
def expand_labels(plane, ctx, distance=2.0):
    lab = _as_labels(plane)
    return back_to(skseg.expand_labels(lab, distance=float(distance)), plane)


@op("segment.shrink_labels", "Shrink Labels", CAT, kind="plane", supports_rgb=False,
    description="Erode every label by a radius (a pixel survives when its whole neighbourhood carries the same label).",
    keywords=["label", "erode", "shrink"], params=[Param("distance", "int", 1, min=1, max=100, unit="px")])
def shrink_labels(plane, ctx, distance=1):
    lab = _as_labels(plane)
    fp = skm.disk(int(distance))
    mn = ndi.grey_erosion(lab, footprint=fp, mode="nearest")
    mx = ndi.grey_dilation(lab, footprint=fp, mode="nearest")
    return back_to(np.where((mn == mx) & (lab > 0), lab, 0), plane)


@op("segment.remove_border_labels", "Remove Border Labels", CAT, kind="plane", supports_rgb=False,
    description="Delete every label that touches the image border.", keywords=["label", "clear border", "edge"],
    params=[Param("buffer", "int", 0, min=0, max=100, help="Also remove labels within this distance of the border")])
def remove_border_labels(plane, ctx, buffer=0):
    lab = _as_labels(plane)
    return back_to(skseg.clear_border(lab, buffer_size=int(buffer)), plane)


@op("segment.merge_touching_labels", "Merge Touching Labels", CAT, kind="plane", supports_rgb=False,
    description="Merge labels that touch each other into one object (connected components of the label foreground), relabelled sequentially.",
    keywords=["label", "merge", "join"], params=[Param("connectivity", "choice", "8", choices=["4", "8"])])
def merge_touching_labels(plane, ctx, connectivity="8"):
    lab = _as_labels(plane)
    merged = skmeas.label(lab > 0, connectivity=2 if connectivity == "8" else 1)
    n = int(merged.max())
    if np.issubdtype(plane.dtype, np.integer) and n > np.iinfo(plane.dtype).max:
        # back_to would clip every object numbered above the type maximum into one label (MorphoLibJ refuses too)
        bits = plane.dtype.itemsize * 8
        raise ValueError(f"Merge Touching Labels found {n} objects, more than a {bits}-bit image can hold "
                         f"({np.iinfo(plane.dtype).max}); convert the image to {'16' if bits == 8 else '32'}-bit first")
    return back_to(merged, plane)


# ===================================================================== seeded watershed
def _marker_points(ctx, img: LumenImage) -> list[tuple[int, int]]:
    H, W = img.height, img.width
    pts: list[list[float]] = []
    if ctx.roi is not None and ctx.roi.get("type") == "point":
        pts = list(ctx.roi.get("points", []))
    else:
        for r in _rois_for_image(ctx.session, img):
            if r.get("type") == "point" and _on_current_plane(r, img):
                pts += list(r.get("points", []))
    out, seen = [], set()
    for x, y in pts:
        xi, yi = int(x), int(y)
        if 0 <= xi < W and 0 <= yi < H and (xi, yi) not in seen:
            seen.add((xi, yi))
            out.append((xi, yi))
    return out


@op("segment.seeded_watershed", "Seeded Watershed", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=True,
    description="Marker-controlled watershed of the (smoothed, optionally inverted) intensity from a point selection / point ROIs, limited to the current threshold.",
    keywords=["watershed", "seeds", "markers", "split", "points"],
    params=[Param("markers", "choice", "Points", choices=["Points", "Local maxima"], help="Points: the active point selection or point ROIs of the ROI manager"),
            Param("smooth", "float", 1.0, min=0.0, max=20.0, step=0.5, help="Gaussian sigma applied before flooding"),
            Param("invert", "bool", True, help="Flood the inverted image so bright objects become basins"),
            Param("use_gradient", "bool", False, help="Flood the Sobel gradient magnitude instead of the intensity"),
            Param("use_threshold", "bool", True, help="Limit the result to the current threshold when one is set"),
            Param("prominence", "float", 10.0, min=0.0, step=1.0, advanced=True, help="Local maxima: minimum height of a seed above its surroundings"),
            Param("compactness", "float", 0.0, min=0.0, max=1000.0, step=0.1, advanced=True), Param("watershed_line", "bool", False),
            Param("add_to_manager", "bool", True)])
def seeded_watershed(ctx, markers="Points", smooth=1.0, invert=True, use_gradient=False, use_threshold=True, prominence=10.0, compactness=0.0,
                     watershed_line=False, add_to_manager=True):
    img = ctx.img
    d = img.display
    H, W = img.height, img.width
    p = np.nan_to_num(img.current_plane().astype(np.float32))
    s = ndi.gaussian_filter(p, smooth) if smooth > 0 else p
    surface = skfilt.sobel(s).astype(np.float32) if use_gradient else (-s if invert else s)
    mask = None
    thr = current_threshold(ctx) if use_threshold else None
    if thr is not None:
        mask = (p >= thr[0]) & (p <= thr[1])
    if ctx.mask is not None:
        mask = ctx.mask if mask is None else (mask & ctx.mask)
    seeds = np.zeros((H, W), np.int32)
    if markers == "Points":
        pts = _marker_points(ctx, img)
        if not pts:
            raise ValueError("Seeded watershed needs a point selection or point ROIs in the ROI manager (or choose markers = Local maxima)")
        for i, (x, y) in enumerate(pts, 1):
            seeds[y, x] = i
    else:
        peaks = skm.h_maxima(-surface, max(float(prominence), 1e-6)).astype(bool)
        if mask is not None:
            peaks &= mask
        seeds = skmeas.label(peaks, connectivity=2).astype(np.int32)
        if seeds.max() == 0:
            raise ValueError("No local maxima found – lower the prominence")
    lab = skseg.watershed(surface, seeds, mask=mask, compactness=float(compactness), watershed_line=bool(watershed_line))
    lab, _, _ = skseg.relabel_sequential(np.asarray(lab, np.int32))
    return _label_output(ctx, img, {(d.t, d.z): lab}, f"Watershed of {img.name}", add_to_manager)


# ================================================================================ colour clustering
COLOR_SPACES = ["RGB", "Lab", "HSB"]


def _cluster_features(img: LumenImage, color_space: str) -> tuple[np.ndarray, list[str], np.ndarray | None]:
    """(features (H, W, F) in ~[0, 1], feature names, RGB (H, W, 3) uint8 of the plane when it is colour)."""
    d = img.display
    if img.rgb:
        rgb = np.transpose(img.data[d.t, d.z], (1, 2, 0)).astype(np.uint8)
        from skimage import color as skcol
        if color_space == "Lab":
            lab = skcol.rgb2lab(rgb)
            feats = np.stack([lab[..., 0] / 100.0, (lab[..., 1] + 128) / 255.0, (lab[..., 2] + 128) / 255.0], axis=-1)
            names = ["L", "a", "b"]
        elif color_space == "HSB":
            hsv = skcol.rgb2hsv(rgb)
            # hue is circular: encode it as (cos, sin) scaled by the saturation so grey pixels do not get a random hue
            feats = np.stack([np.cos(2 * np.pi * hsv[..., 0]) * hsv[..., 1] * 0.5, np.sin(2 * np.pi * hsv[..., 0]) * hsv[..., 1] * 0.5, hsv[..., 1], hsv[..., 2]], axis=-1)
            names = ["Hue (cos)", "Hue (sin)", "Saturation", "Brightness"]
        else:
            feats = rgb.astype(np.float32) / 255.0
            names = ["R", "G", "B"]
        return feats.astype(np.float32), names, rgb
    planes = []
    names = []
    for c in range(img.nc):
        p = np.nan_to_num(img.data[d.t, d.z, c].astype(np.float32))
        lo, hi = float(p.min()), float(p.max())
        planes.append((p - lo) / (hi - lo) if hi > lo else np.zeros_like(p))
        ch = img.display.channels[c] if c < len(img.display.channels) else None
        names.append(ch.name if ch is not None and ch.name else f"C{c + 1}")
    return np.stack(planes, axis=-1).astype(np.float32), names, None


@op("segment.color_clustering", "Color Clustering", CAT, kind="analysis", preview=False, supports_rgb=True, roi_aware=True,
    params=[Param("n_clusters", "int", 3, min=2, max=64), Param("color_space", "choice", "RGB", choices=COLOR_SPACES, help="Colour features of RGB images (multi-channel images use their channels)"),
            Param("spatial_weight", "float", 0.0, min=0, max=10, step=0.1, help="Weight of the normalised x, y position as extra features (0 = colour only)"),
            Param("max_samples", "int", 100000, min=1000, max=5000000, advanced=True, help="Pixels sampled to fit the k-means centres (all pixels are then assigned)"),
            Param("seed", "int", 0, min=0, advanced=True), Param("color_image", "bool", False, help="Also create an image with every pixel replaced by its cluster colour")],
    description="Segment the current plane by k-means clustering of its pixel colours (RGB / Lab / HSB, or the channel values of a multi-channel image), optionally with the pixel position as extra feature: a label image (clusters ordered by brightness) plus a 'Clusters' table with the size and mean colour of every cluster.",
    keywords=["k-means", "kmeans", "color clustering", "colour segmentation", "clusters", "weka"])
def color_clustering(ctx, n_clusters=3, color_space="RGB", spatial_weight=0.0, max_samples=100000, seed=0, color_image=False):
    from sklearn.cluster import KMeans
    img = ctx.img
    feats, names, rgb = _cluster_features(img, color_space)
    H, W, F = feats.shape
    if spatial_weight > 0:
        yy, xx = np.mgrid[0:H, 0:W]
        feats = np.concatenate([feats, (xx / max(W - 1, 1) * spatial_weight)[..., None].astype(np.float32), (yy / max(H - 1, 1) * spatial_weight)[..., None].astype(np.float32)], axis=-1)
        names = names + ["x", "y"]
    mask = ctx.mask
    X = feats.reshape(-1, feats.shape[-1])
    sel = np.nonzero(mask.ravel())[0] if mask is not None else np.arange(X.shape[0])
    if sel.size < n_clusters:
        raise ValueError(f"Only {sel.size} pixels to cluster but {n_clusters} clusters requested")
    rng = np.random.default_rng(int(seed))
    fit_idx = sel if sel.size <= max_samples else rng.choice(sel, size=int(max_samples), replace=False)
    km = KMeans(n_clusters=int(n_clusters), n_init=4, random_state=int(seed))
    km.fit(X[fit_idx])
    assign = km.predict(X[sel])
    centers = km.cluster_centers_
    # order the clusters by brightness so that label 1 is the darkest cluster (deterministic colours in the Glasbey LUT)
    bright = centers[:, :F].mean(axis=1) if color_space != "HSB" or rgb is None else centers[:, 3]
    order = np.argsort(bright)
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    lab = np.zeros(H * W, np.int32)
    lab[sel] = rank[assign] + 1
    lab = lab.reshape(H, W)
    lim = label_image(lab, f"Clusters of {img.name}", like=img)
    counts = np.bincount(lab.ravel(), minlength=n_clusters + 1)[1:]
    total = int(counts.sum())
    rows = []
    for k in range(n_clusters):
        c = centers[order[k]]
        row: dict[str, Any] = {"Cluster": k + 1, "Pixels": int(counts[k]), "Fraction (%)": float(100.0 * counts[k] / max(total, 1))}
        for nm, v in zip(names, c):
            row[f"Mean {nm}"] = float(v)
        if rgb is not None:
            m = lab == k + 1
            mean_rgb = rgb[m].mean(axis=0) if m.any() else np.zeros(3)
            row["R"], row["G"], row["B"] = (int(round(v)) for v in mean_rgb)
            row["Color"] = "#%02x%02x%02x" % tuple(int(round(v)) for v in mean_rgb)
        rows.append(row)
    cols = list(rows[0].keys())
    table = {"name": "Clusters", "columns": cols, "rows": [[r.get(cn) for cn in cols] for r in rows]}
    new_images = [lim]
    if color_image and rgb is not None:
        out = np.zeros((H, W, 3), np.uint8)
        for k, r in enumerate(rows):
            out[lab == k + 1] = (r["R"], r["G"], r["B"])
        cim = LumenImage(out, name=f"Cluster colours of {img.name}", rgb=True)
        cim.calibration = type(img.calibration)(**img.calibration.to_dict())
        new_images.append(cim)
    msg = f"{n_clusters} colour clusters ({color_space if rgb is not None else f'{F} channels'}" + (f", spatial weight {spatial_weight:g}" if spatial_weight > 0 else "") + ")"
    return OpResult(new_images=new_images, table=table, message=msg, log=msg)
