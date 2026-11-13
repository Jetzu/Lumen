"""Trainable (Weka-style) pixel classification – the machinery shared by the 2-D and 3-D commands.

This module holds no commands of its own; :mod:`segmentation` (``segment.*``) and :mod:`segmentation3d`
(``segment3d.*``) import from it.  It must not import either of them (they import it).

* the Weka feature bank for 2-D planes (:func:`plane_features`): Gaussian blur, Sobel, Hessian (module, trace,
  determinant, eigenvalues, orientation, eigenvalue difference …), Difference of Gaussians, Membrane projections,
  Variance, Mean, Minimum, Maximum, Median, Anisotropic diffusion, Bilateral, Lipschitz, Kuwahara, Gabor,
  Derivatives, Laplacian, Structure, Entropy, Neighbors – at the doubling scales min_sigma … max_sigma.
  ``settings["feature_version"]`` keeps classifiers saved by earlier versions valid: version 2 models (no key)
  are computed exactly as before (two Hessian eigenvalues, 12-angle membrane projections …).
* the classifier choice (:func:`fit_classifier`): Random forest (default, the FastRandomForest equivalent),
  Extra trees, Gradient boosting (histogram-based), SVM (RBF, with a sample cap), k-NN, Logistic regression
  and MLP – with class balancing, per-class sample caps, an accuracy estimate (out-of-bag for the forests, a
  stratified 20 % hold-out otherwise) and feature importances where the estimator provides them.
* model files (:func:`save_model` / :func:`load_model`): a joblib archive (a pickle) holding the estimator,
  the feature settings, the class names and colours and the training summary – portable across platforms.
* the summary tables the commands publish in ``session.tables`` (:func:`model_tables`) and the class-colour
  overlay (:func:`overlay_rgb`).

Approximations (documented here because Fiji's implementations differ in detail): *Anisotropic diffusion* is a
regularised Perona–Malik diffusion (Catté et al.: conductance from the gradient of the Gaussian-smoothed image at
the feature scale, 20 iterations, exponential conductance with kappa = 0.1 of the intensity range).  *Lipschitz*
is the top-hat against the lower Lipschitz envelope under the L1 (city-block) cone with slopes 5, 10, 15 and 20
grey levels / pixel on the 8-bit scale.  *Bilateral* uses the histogram bilateral mean on an 8-bit copy (radius
5 / 10, range 50 / 100) and *Median* / *Entropy* use rank filters on an 8-bit copy, like Weka's 8-bit features.
The 8-bit copy spans the plane's own finite min / max (Weka's ``convertToByte(true)``; a single hot pixel
compresses it there too) and the Median / Bilateral outputs and the Lipschitz slopes are expressed in the scaled
plane's units again, so the rank features keep their resolution on 12-bit-in-16-bit or narrow-range data and a
model transfers between images holding the same values whatever each image's range.  Models saved with feature
version 3 keep the earlier copy (0..1 of the type range) so that their features are unchanged.
*Gabor* takes the maximum and mean magnitude over four orientations of a Gabor kernel with wavelength 4·sigma.
*Kuwahara* picks, per pixel, the line orientation (membrane patch) with the smallest variance (or variance /
mean, variance / mean²) and returns its mean.  *Derivatives* are the magnitudes of the Gaussian derivatives of
order 2–5, *Structure* the eigenvalues of the structure tensor at the feature scale and *Neighbors* the 8 pixels
at distance sigma.  Scale-dependent features are gamma-normalised (multiplied by sigma^order).
"""
from __future__ import annotations

import datetime as _dt
import math
import os
import re
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage import filters as skfilt, morphology as skm

# --------------------------------------------------------------------------- choices
#: the first eight keep the order of earlier versions (saved models list their features in this order)
FEATURE_CHOICES = ["Gaussian", "Difference of Gaussians", "Sobel", "Laplacian", "Hessian", "Variance", "Entropy", "Membrane projections",
                   "Mean", "Minimum", "Maximum", "Median", "Anisotropic diffusion", "Bilateral", "Lipschitz", "Kuwahara", "Gabor",
                   "Derivatives", "Structure", "Neighbors"]
DEFAULT_FEATURES = ["Gaussian", "Difference of Gaussians", "Sobel", "Laplacian", "Hessian", "Variance"]
#: Weka's spelling of the same features (accepted everywhere a feature list is given)
FEATURE_ALIASES = {"gaussian blur": "Gaussian", "sobel filter": "Sobel", "difference of gaussians": "Difference of Gaussians", "dog": "Difference of Gaussians",
                   "membrane projection": "Membrane projections", "membrane": "Membrane projections", "neighbours": "Neighbors", "anisotropic": "Anisotropic diffusion",
                   "structure tensor": "Structure", "gabor filter": "Gabor", "lipschitz filter": "Lipschitz", "kuwahara filter": "Kuwahara",
                   "bilateral filter": "Bilateral", "laplace": "Laplacian", "gaussian": "Gaussian", "sobel": "Sobel"}

CLASSIFIER_CHOICES = ["Random forest", "Extra trees", "Gradient boosting", "SVM", "k-NN", "Logistic regression", "MLP"]
CLASSIFIER_ALIASES = {"randomforest": "Random forest", "random_forest": "Random forest", "rf": "Random forest", "fastrandomforest": "Random forest",
                      "extratrees": "Extra trees", "extra_trees": "Extra trees", "et": "Extra trees",
                      "gradientboosting": "Gradient boosting", "gradient_boosting": "Gradient boosting", "hgb": "Gradient boosting", "boosting": "Gradient boosting",
                      "svc": "SVM", "svm (rbf)": "SVM", "knn": "k-NN", "k-nn": "k-NN", "nearest neighbours": "k-NN", "nearest neighbors": "k-NN",
                      "logistic": "Logistic regression", "logreg": "Logistic regression", "logistic regression": "Logistic regression",
                      "mlp": "MLP", "neural network": "MLP", "multilayer perceptron": "MLP"}

#: Weka's class colours (red, green, blue, cyan, magenta, yellow, orange, pink, grey, white) and a few more
DEFAULT_COLORS = ["#ff0000", "#00ff00", "#0000ff", "#00ffff", "#ff00ff", "#ffff00", "#ffa500", "#ff69b4", "#9e9e9e", "#ffffff",
                  "#8b4513", "#7fff00", "#4b0082", "#20b2aa", "#dc143c", "#f0e68c"]

FEATURE_VERSION = 4
MODEL_VERSION = 3

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


# --------------------------------------------------------------------------- small helpers
def normalize_features(features: Any, choices: list[str] | None = None) -> list[str]:
    """The requested feature names (list or comma string, Weka spellings allowed) in the order of ``choices``."""
    choices = choices or FEATURE_CHOICES
    if isinstance(features, str):
        features = features.replace(";", ",").split(",")
    wanted: set[str] = set()
    for f in features or []:
        key = str(f).strip().lower()
        if not key:
            continue
        key = FEATURE_ALIASES.get(key, key)
        wanted.add(key.lower())
    out = [c for c in choices if c.lower() in wanted]
    return out or [choices[0]]


def normalize_classifier(name: Any) -> str:
    key = str(name or "Random forest").strip()
    for c in CLASSIFIER_CHOICES:
        if c.lower() == key.lower():
            return c
    alias = CLASSIFIER_ALIASES.get(key.lower())
    if alias:
        return alias
    raise ValueError(f"Unknown classifier {key!r}; choose one of {', '.join(CLASSIFIER_CHOICES)}")


def sigma_list(min_sigma: float, max_sigma: float) -> list[float]:
    """Doubling scales min_sigma, 2·min_sigma … up to max_sigma (Weka's sigma ladder); ``[max_sigma]`` if max < min."""
    lo = float(min_sigma) if min_sigma and min_sigma > 0 else 1.0
    hi = float(max_sigma) if max_sigma and max_sigma > 0 else lo
    out: list[float] = []
    s = lo
    while s <= hi + 1e-9 and len(out) < 12:
        out.append(round(s, 6))
        s *= 2.0
    return out or [hi]


def parse_colors(spec: Any, n: int) -> list[str]:
    """``n`` class colours from a comma-separated list of hex colours (missing / invalid ones take the Weka defaults)."""
    out: list[str] = []
    parts = [str(p).strip() for p in (spec if isinstance(spec, (list, tuple)) else str(spec or "").replace(";", ",").split(","))]
    for i in range(n):
        v = parts[i] if i < len(parts) else ""
        m = _HEX.match(v) if v else None
        out.append(("#" + m.group(1)).lower() if m else DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
    return out


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    m = _HEX.match(str(color or ""))
    if not m:
        return (255, 255, 255)
    v = int(m.group(1), 16)
    return ((v >> 16) & 255, (v >> 8) & 255, v & 255)


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return f if math.isfinite(f) else None
    return v


# --------------------------------------------------------------------------- scaling (shared with the commands)
def intensity_scale(dtype) -> float:
    """Divisor that maps an integer image onto 0..1 (float images are used as they are)."""
    dtype = np.dtype(dtype)
    return 255.0 if dtype == np.uint8 else (65535.0 if dtype == np.uint16 else 1.0)


def scaled(plane: np.ndarray, divisor: float | None = None) -> np.ndarray:
    """Plane as float32 divided by ``divisor`` (the divisor of the image the classifier was trained on, so that a
    classifier transfers between 8/16/32-bit images holding the same values); without one the plane's own type
    maximum is used (classifiers saved before version 2)."""
    d = float(divisor) if divisor else intensity_scale(plane.dtype)
    return np.nan_to_num(plane.astype(np.float32) / d)


def norm01(plane: np.ndarray) -> np.ndarray:
    """Plane as float32 scaled to 0..1 by its (finite) min / max."""
    p = plane.astype(np.float32)
    finite = p[np.isfinite(p)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return np.nan_to_num((p - lo) / (hi - lo)).astype(np.float32)


def feature_settings(channels: str, features: Any, min_sigma: float, max_sigma: float, dtype=None, membrane_thickness: int = 1,
                     membrane_patch_size: int = 19) -> dict[str, Any]:
    """The feature configuration stored with a 2-D model (what :func:`plane_features` needs to reproduce the stack)."""
    out: dict[str, Any] = {"channels": channels, "features": normalize_features(features), "sigmas": sigma_list(min_sigma, max_sigma),
                           "min_sigma": float(min_sigma), "max_sigma": float(max_sigma), "membrane_thickness": int(max(1, membrane_thickness)),
                           "membrane_patch_size": int(max(3, membrane_patch_size)), "feature_version": FEATURE_VERSION}
    if dtype is not None:
        out["intensity_scale"] = intensity_scale(dtype)  # fixed at training time, reused when the model is applied
    return out


# --------------------------------------------------------------------------- 2-D feature bank
def _conv(p: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """True convolution with symmetric edge handling; FFT-based for large kernels."""
    if kernel.size <= 225:
        return ndi.convolve(p, kernel.astype(np.float32), mode="reflect").astype(np.float32)
    from scipy.signal import fftconvolve
    ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
    pad = np.pad(p, ((ph, ph), (pw, pw)), mode="symmetric")
    out = fftconvolve(pad, kernel, mode="same")
    out = out[ph:ph + p.shape[0], pw:pw + p.shape[1]]
    return (out.real if np.iscomplexobj(out) else out).astype(np.float32)


def _conv_complex(p: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Magnitude of the complex convolution (Gabor)."""
    from scipy.signal import fftconvolve
    ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
    pad = np.pad(p, ((ph, ph), (pw, pw)), mode="symmetric")
    out = fftconvolve(pad, kernel, mode="same")[ph:ph + p.shape[0], pw:pw + p.shape[1]]
    return np.abs(out).astype(np.float32)


def _disk(radius: float) -> np.ndarray:
    return skm.disk(max(1, int(round(radius))))


def _to_uint8_v3(p: np.ndarray) -> np.ndarray:
    """Feature version 3's 8-bit copy: the 0..1 scaled plane, or min/max normalised when it leaves that range (kept for
    saved models: on 16-bit data using a small part of the type range it collapses to a few grey levels)."""
    if float(np.nanmin(p)) >= 0.0 and float(np.nanmax(p)) <= 1.0:
        return np.clip(p * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return (norm01(p) * 255.0 + 0.5).astype(np.uint8)


def _to_uint8(p: np.ndarray) -> tuple[np.ndarray, float, float]:
    """8-bit copy for the rank filters plus the (lo, hi) range it spans: the plane's own finite min / max, like Weka's
    ``convertToByte(true)`` (a single hot pixel compresses the copy there as well).  Returns ``(q, lo, hi)``; a rank
    output ``r`` maps back onto the plane's units as ``lo + r * (hi - lo) / 255``."""
    finite = p[np.isfinite(p)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    q = np.clip((np.nan_to_num(p) - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return q, lo, hi


def membrane_projections_legacy(p: np.ndarray, size: int = 19, n_angles: int = 12) -> list[np.ndarray]:
    """Version-2 membrane projections: Bresenham line kernels at 12 orientations, projected (mean, max, min, std)."""
    from skimage.draw import line
    c = size // 2
    resp = []
    for k in range(n_angles):
        ang = math.pi * k / n_angles
        dx, dy = math.cos(ang) * c, math.sin(ang) * c
        ker = np.zeros((size, size), np.float32)
        rr, cc = line(int(round(c - dy)), int(round(c - dx)), int(round(c + dy)), int(round(c + dx)))
        ker[rr, cc] = 1.0
        ker /= ker.sum()
        resp.append(ndi.convolve(p, ker, mode="reflect"))
    R = np.stack(resp)
    return [R.mean(0), R.max(0), R.min(0), R.std(0)]


def line_kernels(size: int, thickness: int, n_angles: int = 30) -> list[np.ndarray]:
    """Normalised line kernels (``size`` × ``size``, ``thickness`` pixels wide) at ``n_angles`` orientations over 180°."""
    size = max(3, int(size) | 1)
    t = max(1, int(thickness))
    base = np.zeros((size, size), np.float32)
    c = size // 2
    y0 = c - (t - 1) // 2
    base[max(0, y0):min(size, y0 + t), :] = 1.0
    out = []
    for k in range(n_angles):
        ang = 180.0 * k / n_angles
        ker = base if k == 0 else ndi.rotate(base, ang, reshape=False, order=1, mode="constant", cval=0.0)
        ker = np.clip(ker, 0.0, None).astype(np.float32)
        s = float(ker.sum())
        out.append(ker / s if s > 0 else base / float(base.sum()))
    return out


def anisotropic_diffusion(p: np.ndarray, sigma: float, n_iter: int = 20, kappa: float = 0.1, gamma: float = 0.15) -> np.ndarray:
    """Regularised Perona–Malik diffusion (Catté et al.): the conductance comes from the gradient of the
    Gaussian-smoothed image at scale ``sigma`` (exponential conductance), explicit 4-neighbour scheme."""
    u = p.astype(np.float32, copy=True)
    k2 = float(kappa) ** 2
    for _ in range(int(n_iter)):
        g = ndi.gaussian_filter(u, sigma) if sigma > 0 else u
        pu = np.pad(u, 1, mode="edge")
        pg = np.pad(g, 1, mode="edge")
        acc = np.zeros_like(u)
        for du, dg in (((pu[:-2, 1:-1] - u), (pg[:-2, 1:-1] - g)), ((pu[2:, 1:-1] - u), (pg[2:, 1:-1] - g)),
                       ((pu[1:-1, :-2] - u), (pg[1:-1, :-2] - g)), ((pu[1:-1, 2:] - u), (pg[1:-1, 2:] - g))):
            acc += np.exp(-(dg * dg) / k2) * du
        u += float(gamma) * acc
    return u


def lipschitz_tophat(p: np.ndarray, slope: float) -> np.ndarray:
    """``p`` minus its lower Lipschitz envelope under the L1 cone of the given slope (top-hat with a cone)."""
    u = p.astype(np.float32, copy=True)
    s = np.float32(slope)
    H, W = u.shape
    for x in range(1, W):
        np.minimum(u[:, x], u[:, x - 1] + s, out=u[:, x])
    for x in range(W - 2, -1, -1):
        np.minimum(u[:, x], u[:, x + 1] + s, out=u[:, x])
    for y in range(1, H):
        np.minimum(u[y], u[y - 1] + s, out=u[y])
    for y in range(H - 2, -1, -1):
        np.minimum(u[y], u[y + 1] + s, out=u[y])
    return p - u


def _shifted(p: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """The plane shifted so that pixel (y, x) shows (y + dy, x + dx) (edge values repeat)."""
    H, W = p.shape
    r = max(abs(dy), abs(dx))
    pad = np.pad(p, r, mode="edge")
    return pad[r + dy:r + dy + H, r + dx:r + dx + W]


def plane_features(plane: np.ndarray, settings: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """(H, W, F) float32 feature stack of one 2-D plane plus the feature names."""
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals, structure_tensor, structure_tensor_eigenvalues
    p = scaled(plane, settings.get("intensity_scale"))
    version = int(settings.get("feature_version", 2))
    F = set(settings["features"])
    sigmas = [float(s) for s in settings["sigmas"]]
    feats: list[np.ndarray] = [p]
    names: list[str] = ["Original"]
    gs = {s: ndi.gaussian_filter(p, s).astype(np.float32) for s in sigmas}
    p8: np.ndarray | None = None
    p8_lo, p8_step = 0.0, 1.0 / 255.0  # one 8-bit grey level of the copy in the scaled plane's units (version >= 4)

    def u8() -> np.ndarray:
        nonlocal p8, p8_lo, p8_step
        if p8 is None:
            if version >= 4:
                p8, lo, hi = _to_uint8(p)
                p8_lo, p8_step = lo, (hi - lo) / 255.0
            else:
                p8 = _to_uint8_v3(p)
        return p8

    def from_u8(r: np.ndarray) -> np.ndarray:
        """A rank-filter output on the 8-bit copy mapped back onto the scaled plane's units."""
        return (r.astype(np.float32) * np.float32(p8_step) + np.float32(p8_lo)).astype(np.float32)

    if "Gaussian" in F:
        for s in sigmas:
            feats.append(gs[s]); names.append(f"Gaussian_{s:g}")
    if "Difference of Gaussians" in F:
        prev, prev_s = p, 0.0
        for s in sigmas:
            feats.append(gs[s] - prev); names.append(f"DoG_{prev_s:g}_{s:g}")
            prev, prev_s = gs[s], s
    if "Sobel" in F:
        for s in sigmas:
            feats.append(skfilt.sobel(gs[s]).astype(np.float32)); names.append(f"Sobel_{s:g}")
    if "Laplacian" in F:
        for s in sigmas:
            feats.append((ndi.gaussian_laplace(p, s) * s * s).astype(np.float32)); names.append(f"Laplacian_{s:g}")
    if "Hessian" in F:
        for s in sigmas:
            H = hessian_matrix(p, sigma=s, order="rc", use_gaussian_derivatives=False)
            if version < 3:
                ev = hessian_matrix_eigvals(H)
                feats.append((ev[0] * s * s).astype(np.float32)); names.append(f"Hessian_max_{s:g}")
                feats.append((ev[1] * s * s).astype(np.float32)); names.append(f"Hessian_min_{s:g}")
                continue
            a, b, c = (np.asarray(h, np.float32) * np.float32(s * s) for h in H)  # gamma-normalised d²/dr², d²/drdc, d²/dc²
            half = (a + c) * 0.5
            root = np.sqrt(((a - c) * 0.5) ** 2 + b * b)
            e1, e2 = half + root, half - root
            feats += [np.sqrt(a * a + 2.0 * b * b + c * c), a + c, a * c - b * b, e1, e2,
                      (0.5 * np.arctan2(2.0 * b, a - c)).astype(np.float32), e1 - e2, (e1 - e2) ** 2]
            names += [f"Hessian_{k}_{s:g}" for k in ("module", "trace", "det", "max", "min", "orient", "eigdiff", "sq_eigdiff")]
    if "Variance" in F:
        for s in sigmas:
            m = gs[s]
            feats.append(np.maximum(ndi.gaussian_filter(p * p, s) - m * m, 0).astype(np.float32)); names.append(f"Variance_{s:g}")
    if "Entropy" in F:
        from skimage.filters.rank import entropy
        q = (norm01(p) * 255).astype(np.uint8) if version < 3 else u8()
        for s in sigmas:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats.append(entropy(q, _disk(s)).astype(np.float32))
            names.append(f"Entropy_{s:g}")
    kernels: list[np.ndarray] | None = None
    responses: np.ndarray | None = None
    if version >= 3 and ("Membrane projections" in F or "Kuwahara" in F):
        kernels = line_kernels(int(settings.get("membrane_patch_size", 19)), int(settings.get("membrane_thickness", 1)), 30)
        responses = np.stack([_conv(p, k) for k in kernels])
    if "Membrane projections" in F:
        if version < 3:
            feats += membrane_projections_legacy(p)
            names += ["Membrane_mean", "Membrane_max", "Membrane_min", "Membrane_std"]
        else:
            R = responses
            feats += [R.sum(0), R.mean(0), R.std(0), np.median(R, axis=0).astype(np.float32), R.max(0), R.min(0)]
            names += [f"Membrane_{k}" for k in ("sum", "mean", "std", "median", "max", "min")]
    if version >= 3:
        if "Mean" in F:
            for s in sigmas:
                d = _disk(s).astype(np.float32)
                feats.append(_conv(p, d / d.sum())); names.append(f"Mean_{s:g}")
        if "Minimum" in F:
            for s in sigmas:
                feats.append(ndi.minimum_filter(p, footprint=_disk(s), mode="reflect").astype(np.float32)); names.append(f"Minimum_{s:g}")
        if "Maximum" in F:
            for s in sigmas:
                feats.append(ndi.maximum_filter(p, footprint=_disk(s), mode="reflect").astype(np.float32)); names.append(f"Maximum_{s:g}")
        if "Median" in F:
            from skimage.filters.rank import median
            for s in sigmas:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    feats.append(from_u8(median(u8(), _disk(s))))
                names.append(f"Median_{s:g}")
        if "Anisotropic diffusion" in F:
            for s in sigmas:
                feats.append(anisotropic_diffusion(p, s)); names.append(f"Anisotropic_{s:g}")
        if "Bilateral" in F:
            from skimage.filters.rank import mean_bilateral
            for r, rng_ in ((5, 50), (5, 100), (10, 50), (10, 100)):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    feats.append(from_u8(mean_bilateral(u8(), skm.disk(r), s0=rng_, s1=rng_)))
                names.append(f"Bilateral_{r}_{rng_}")
        if "Lipschitz" in F:
            if version >= 4:
                u8()  # slopes in grey levels of the plane's own 8-bit copy, as in Weka
            for slope in (5, 10, 15, 20):
                feats.append(lipschitz_tophat(p, slope * p8_step)); names.append(f"Lipschitz_{slope}")
        if "Kuwahara" in F:
            R = responses
            S = np.stack([_conv(p * p, k) for k in kernels])
            var = np.maximum(S - R * R, 0.0)
            mean_safe = np.maximum(np.abs(R), 1e-6)
            for tag, crit in (("var", var), ("var_mean", var / mean_safe), ("var_mean2", var / (mean_safe * mean_safe))):
                best = np.argmin(crit, axis=0)
                feats.append(np.take_along_axis(R, best[None], axis=0)[0].astype(np.float32)); names.append(f"Kuwahara_{tag}")
        if "Gabor" in F:
            from skimage.filters import gabor_kernel
            for s in sigmas:
                mags = [_conv_complex(p, np.asarray(gabor_kernel(frequency=1.0 / (4.0 * s), theta=th, sigma_x=s, sigma_y=s)))
                        for th in (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)]
                M = np.stack(mags)
                feats.append(M.max(0)); names.append(f"Gabor_max_{s:g}")
                feats.append(M.mean(0)); names.append(f"Gabor_mean_{s:g}")
        if "Derivatives" in F:
            for s in sigmas:
                for order in (2, 3, 4, 5):
                    dy = ndi.gaussian_filter(p, s, order=(order, 0)) * s ** order
                    dx = ndi.gaussian_filter(p, s, order=(0, order)) * s ** order
                    feats.append(np.sqrt(dx * dx + dy * dy).astype(np.float32)); names.append(f"Derivatives_{order}_{s:g}")
        if "Structure" in F:
            for s in sigmas:
                A = structure_tensor(p, sigma=s, order="rc")
                ev = structure_tensor_eigenvalues(A)
                feats.append((ev[0] * s * s).astype(np.float32)); names.append(f"Structure_largest_{s:g}")
                feats.append((ev[1] * s * s).astype(np.float32)); names.append(f"Structure_smallest_{s:g}")
        if "Neighbors" in F:
            for s in sigmas:
                r = max(1, int(round(s)))
                for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                    feats.append(_shifted(p, dy * r, dx * r)); names.append(f"Neighbor_{dy}_{dx}_{s:g}")
    stack = np.stack([np.asarray(f, np.float32) for f in feats], axis=-1)
    return np.nan_to_num(stack), names


# --------------------------------------------------------------------------- classifiers
def _subsample(X: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return X if X.shape[0] <= n else X[rng.choice(X.shape[0], int(n), replace=False)]


def assemble_training_set(parts: list[list[np.ndarray]], max_per_class: int, balance: bool, rng: np.random.Generator | None = None
                          ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Stack the per-class feature rows into (X, y) with at most ``max_per_class`` rows per class; with ``balance``
    every class is cut to the size of the smallest class (sample-level balancing, used where the estimator has no
    class weights).  Returns the per-class counts (0 for a class without samples)."""
    rng = rng or np.random.default_rng(0)
    Xs, ys, counts = [], [], []
    per_class = [np.concatenate(p) if p else None for p in parts]
    cap = int(max_per_class)
    if balance:
        present = [x.shape[0] for x in per_class if x is not None and x.shape[0] > 0]
        if present:
            cap = min(cap, min(present))
    for k, Xk in enumerate(per_class):
        if Xk is None or Xk.shape[0] == 0:
            counts.append(0)
            continue
        Xk = _subsample(Xk, cap, rng)
        Xs.append(Xk)
        ys.append(np.full(Xk.shape[0], k, np.int32))
        counts.append(int(Xk.shape[0]))
    if not Xs:
        return np.zeros((0, 0), np.float32), np.zeros(0, np.int32), counts
    return np.concatenate(Xs), np.concatenate(ys), counts


def make_estimator(classifier: str, *, n_trees: int = 100, max_depth: int = 0, features_per_node: int = 0, balance_classes: bool = True,
                   k_neighbors: int = 5, hidden_units: int = 64, random_state: int = 0):
    """An unfitted scikit-learn estimator for one of :data:`CLASSIFIER_CHOICES`."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    classifier = normalize_classifier(classifier)
    cw = "balanced" if balance_classes else None
    depth = int(max_depth) or None
    if classifier == "Random forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=int(n_trees), max_depth=depth, max_features=int(features_per_node) or "sqrt", class_weight=cw,
                                      oob_score=True, n_jobs=-1, random_state=random_state)
    if classifier == "Extra trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(n_estimators=int(n_trees), max_depth=depth, max_features=int(features_per_node) or "sqrt", class_weight=cw,
                                    bootstrap=True, oob_score=True, n_jobs=-1, random_state=random_state)
    if classifier == "Gradient boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=int(n_trees), max_depth=depth, class_weight=cw, random_state=random_state)
    if classifier == "SVM":
        from sklearn.svm import SVC
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight=cw, random_state=random_state))
    if classifier == "k-NN":
        from sklearn.neighbors import KNeighborsClassifier
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=max(1, int(k_neighbors)), weights="distance"))
    if classifier == "Logistic regression":
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight=cw))
    from sklearn.neural_network import MLPClassifier
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(max(1, int(hidden_units)),), max_iter=300, random_state=random_state))


def _final_estimator(est):
    return est.steps[-1][1] if hasattr(est, "steps") else est


def feature_importances(est, n_features: int) -> list[float] | None:
    """Per-feature importances (sum 1) where the estimator offers them: impurity importances of the forests,
    mean absolute standardised coefficients of the logistic regression; None for the others."""
    final = _final_estimator(est)
    imp = None
    if hasattr(final, "feature_importances_"):
        imp = np.asarray(final.feature_importances_, np.float64)
    elif hasattr(final, "coef_"):
        c = np.abs(np.asarray(final.coef_, np.float64))
        imp = c.mean(axis=0) if c.ndim == 2 else c
    if imp is None or imp.shape[0] != n_features:
        return None
    tot = float(imp.sum())
    return [float(v) for v in (imp / tot if tot > 0 else imp)]


def fit_classifier(X: np.ndarray, y: np.ndarray, classifier: str, *, n_trees: int = 100, max_depth: int = 0, features_per_node: int = 0,
                   balance_classes: bool = True, svm_max_samples: int = 4000, k_neighbors: int = 5, hidden_units: int = 64,
                   random_state: int = 0) -> tuple[Any, float, str, int]:
    """Fit the chosen estimator on (X, y).  Returns ``(estimator, accuracy, accuracy_kind, n_used)``: the accuracy is
    the out-of-bag score of the forests or a stratified 20 % hold-out estimate for the other classifiers (NaN when
    there are too few samples); ``n_used`` is the number of rows actually fitted (the SVM is capped)."""
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split
    classifier = normalize_classifier(classifier)
    rng = np.random.default_rng(random_state)
    if classifier == "SVM" and X.shape[0] > int(svm_max_samples):
        keep = []
        for k in np.unique(y):
            idx = np.flatnonzero(y == k)
            n = max(1, int(round(int(svm_max_samples) * idx.size / X.shape[0])))
            keep.append(rng.choice(idx, min(n, idx.size), replace=False))
        sel = np.sort(np.concatenate(keep))
        X, y = X[sel], y[sel]
    est = make_estimator(classifier, n_trees=n_trees, max_depth=max_depth, features_per_node=features_per_node, balance_classes=balance_classes,
                         k_neighbors=k_neighbors, hidden_units=hidden_units, random_state=random_state)
    acc, kind = float("nan"), "none"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if classifier in ("Random forest", "Extra trees"):
            est.fit(X, y)
            acc, kind = float(getattr(est, "oob_score_", float("nan"))), "out-of-bag"
        else:
            counts = np.bincount(y)
            if X.shape[0] >= 40 and counts[counts > 0].min() >= 4:
                Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
                probe = clone(est).fit(Xa, ya)
                acc, kind = float(probe.score(Xb, yb)), "hold-out"
            est.fit(X, y)
    return est, acc, kind, int(X.shape[0])


def predict_proba(est, X: np.ndarray, n_classes: int) -> np.ndarray:
    """(N, K) class probabilities with the columns of classes absent from training filled with 0."""
    pr = np.asarray(est.predict_proba(X), np.float32)
    full = np.zeros((pr.shape[0], n_classes), np.float32)
    full[:, np.asarray(est.classes_, dtype=int)] = pr
    return full


# --------------------------------------------------------------------------- persistence
def save_model(model: dict[str, Any], path: str, default_name: str) -> Path:
    """Write the model dict (estimator, settings, classes, colours, summary) as a joblib archive; a directory gets
    ``default_name`` inside, a bare name gets the ``.joblib`` extension.  Returns the file written."""
    import joblib
    p = Path(os.path.expanduser(str(path).strip() or default_name))
    if p.is_dir():
        p = p / default_name
    if not p.suffix:
        p = p.with_suffix(".joblib")
    p.parent.mkdir(parents=True, exist_ok=True)
    obj = {k: v for k, v in model.items() if not str(k).startswith("_")}
    try:
        import sklearn
        obj["sklearn_version"] = sklearn.__version__
    except Exception:  # pragma: no cover
        pass
    obj["saved"] = _dt.datetime.now().isoformat(timespec="seconds")
    joblib.dump(obj, p, compress=3)
    return p


def load_model(path: str) -> dict[str, Any]:
    """Read a model written by :func:`save_model` (or by an earlier Lumen version) and fill in missing keys."""
    import joblib
    p = Path(os.path.expanduser(str(path).strip()))
    if not p.is_file():
        raise ValueError(f"No such file: {p}")
    try:
        obj = joblib.load(p)
    except Exception as e:  # noqa: BLE001 – anything that is not our archive
        raise ValueError(f"{p} is not a Lumen pixel classifier ({e})") from e
    if not isinstance(obj, dict) or "model" not in obj or "classes" not in obj or "settings" not in obj:
        raise ValueError(f"{p} is not a Lumen pixel classifier")
    obj.setdefault("kind", "classifier")
    obj["classes"] = [str(c) for c in obj["classes"]]
    obj["colors"] = parse_colors(obj.get("colors"), len(obj["classes"]))
    obj.setdefault("classifier", "Random forest")
    obj.setdefault("n_samples", [0] * len(obj["classes"]))
    obj["path"] = str(p)
    return obj


# --------------------------------------------------------------------------- summaries
def model_tables(model: dict[str, Any], prefix: str = "Classifier") -> dict[str, dict[str, Any]]:
    """The tables published in ``session.tables``: ``<prefix>`` (feature importances), ``<prefix> classes``
    (index, name, colour, training samples) and ``<prefix> summary`` (key / value rows)."""
    names = list(model.get("feature_names") or [])
    imp = model.get("importances")
    rows: list[list[Any]] = []
    order = np.argsort(-np.asarray(imp, np.float64), kind="stable") if imp and len(imp) == len(names) else range(len(names))
    for rank, i in enumerate(order, 1):
        rows.append([rank, names[i], _json_safe(imp[i]) if imp and len(imp) == len(names) else None])
    importances = {"name": prefix, "columns": ["Rank", "Feature", "Importance"], "rows": rows}
    classes = list(model.get("classes") or [])
    colors = parse_colors(model.get("colors"), len(classes))
    counts = [int(n) for n in (model.get("n_samples") or [0] * len(classes))]
    total = max(sum(counts), 1)
    cls_rows = [[k, classes[k], colors[k], counts[k] if k < len(counts) else 0, round(100.0 * (counts[k] if k < len(counts) else 0) / total, 1)] for k in range(len(classes))]
    class_table = {"name": f"{prefix} classes", "columns": ["Index", "Class", "Colour", "Training samples", "Share %"], "rows": cls_rows}
    settings = model.get("settings") or {}
    params = model.get("params") or {}
    acc = model.get("accuracy")
    acc_s = f"{float(acc):.3f} ({model.get('accuracy_kind', '')})" if acc is not None and isinstance(acc, (int, float)) and math.isfinite(float(acc)) else "n/a"
    summary_rows = [
        ["Kind", "3-D (voxels)" if model.get("kind") == "classifier3d" else "2-D (pixels)"],
        ["Classifier", str(model.get("classifier", "Random forest"))],
        ["Settings", ", ".join(f"{k} = {v}" for k, v in params.items()) or "defaults"],
        ["Classes", ", ".join(classes)],
        ["Features", int(model.get("n_features", len(names)))],
        ["Feature groups", ", ".join(settings.get("features", []))],
        ["Sigmas", ", ".join(f"{float(s):g}" for s in settings.get("sigmas", []))],
        ["Channels", str(settings.get("channels", "Current channel"))],
        ["Training samples", int(sum(counts))],
        ["Training time (s)", _json_safe(round(float(model.get("train_time", 0.0)), 3))],
        ["Accuracy", acc_s],
        ["Trained on", str(model.get("trained_on", "?"))],
        ["Image type", str(model.get("trained_dtype", "?"))],
        ["File", str(model.get("path", ""))],
        ["Version", int(model.get("version", 1))],
    ]
    summary = {"name": f"{prefix} summary", "columns": ["Property", "Value"], "rows": summary_rows}
    return {prefix: importances, f"{prefix} classes": class_table, f"{prefix} summary": summary}


def publish_tables(session, model: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Store the class / summary tables in ``session.tables`` and return the importances table (for ``OpResult.table``)."""
    tables = model_tables(model, prefix)
    for name, table in tables.items():
        if name != prefix:
            session.tables[name] = table
    return tables[prefix]


def describe(model: dict[str, Any]) -> str:
    classes = ", ".join(model.get("classes", []))
    n = sum(int(x) for x in model.get("n_samples", []) or [])
    acc = model.get("accuracy")
    acc_s = f", accuracy {float(acc):.3f} ({model.get('accuracy_kind', '')})" if isinstance(acc, (int, float)) and math.isfinite(float(acc)) else ""
    return (f"{model.get('classifier', 'Random forest')} on {model.get('n_features', '?')} features; classes {classes}; "
            f"{n} training samples, {float(model.get('train_time', 0.0)):.2f} s{acc_s}; trained on {model.get('trained_on', '?')}")


# --------------------------------------------------------------------------- overlay
def overlay_rgb(labels: np.ndarray, colors: list[str]) -> np.ndarray:
    """(T, Z, 3, H, W) uint8 colour image of a (T, Z, 1, H, W) label array with one colour per class."""
    lut = np.asarray([hex_to_rgb(c) for c in colors] or [(255, 255, 255)], np.uint8)
    lab = np.asarray(labels)
    if lab.ndim == 5:
        lab = lab[:, :, 0]
    idx = np.clip(lab.astype(np.int64), 0, len(lut) - 1)
    rgb = lut[idx]  # (T, Z, H, W, 3)
    return np.ascontiguousarray(np.moveaxis(rgb, -1, 2))


def elapsed(t0: float) -> float:
    return time.perf_counter() - t0
