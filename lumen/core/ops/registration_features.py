"""Plugins ▸ Registration / Feature Extraction — the estimators behind *Align Stack* (SIFT feature
matching with Fiji's "Linear Stack Alignment with SIFT" options, StackReg / TurboReg-style
multi-resolution intensity optimisation), SIFT / MOPS / ORB / block-matching correspondences,
Preibisch-style descriptor-based registration (DoG detections + local geometric descriptors +
RANSAC, 2-D and 3-D, pairs and time series) and manual landmark pairs.

Conventions (shared with ``registration.py``): 2-D transforms are 3×3 matrices in (x, y) pixel-centre
coordinates mapping the *moving* image onto the *reference*; 3-D transforms are the 4×4 (x, y, z)
matrices of ``registration3d.py``. The one exception is :class:`BilinearTransform` (StackReg's
bilinear model), which stores the *inverse* map (reference → moving) as it cannot be inverted in
closed form; ``registration._warp`` / ``_decompose`` accept it.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Callable

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op, cast_to
from .registration import (ALIGN_MODES, CAT, ESTIMATORS, IDENTITY, INTERP, ORDER_MAP, _channel_index, _decompose, _estimation_plane, _f32, _finish,  # noqa: F401
                           _norm01, _phase_shift, _translation, _warp)
from .registration_extra import CORRESPONDENCES_KEY, FEATURE_MODELS, _apply_matrix, _padded_to, estimate_point_transform, transform_params
from .registration3d import REGISTRATION3D_KEY, _decompose3, _volume, _voxel_size, _warp_volume, block_match, detect_dog_nd

CORRESPONDENCE_METHODS = ["SIFT", "MOPS", "ORB", "Block matching"]
DESCRIPTOR_MODELS = ["Translation", "Rigid body", "Similarity", "Affine"]
OBJECTS = ["Bright", "Dark", "Bright and dark"]
LANDMARK_MODELS = ["Rigid body", "Similarity", "Affine", "Projective", "Translation"]
#: Fiji "Linear Stack Alignment with SIFT" defaults
SIFT_DEFAULTS: dict[str, Any] = dict(initial_blur=1.6, steps_per_octave=3, min_image_size=64, max_image_size=1024, descriptor_size=4, orientation_bins=8,
                                     max_ratio=0.92, max_alignment_error=25.0, min_inlier_ratio=0.05)
#: align_stack mode → point-transform model of registration_extra.estimate_point_transform
_POINT_MODEL = {"Translation": "Translation", "Rigid body": "Rigid body", "Scaled rotation": "Similarity", "Similarity": "Similarity", "Affine": "Affine",
                "Bilinear": "Affine", "Projective": "Projective"}


# ============================================================================ SIFT (scikit-image, Fiji's dialog options)
def _rescaled(a: np.ndarray, max_image_size: int) -> tuple[np.ndarray, float]:
    """Downscale so that the longer side is ≤ max_image_size (Fiji's 'maximum image size'); returns (image, factor)."""
    from skimage.transform import rescale
    mx = max(a.shape)
    if max_image_size and max_image_size > 0 and mx > max_image_size:
        f = max_image_size / mx
        return rescale(a, f, order=1, anti_aliasing=True, preserve_range=True).astype(np.float64), f
    return a, 1.0


def _unscale_keypoints(kp: np.ndarray, factor: float) -> np.ndarray:
    return (np.asarray(kp, float) + 0.5) / factor - 0.5 if factor != 1.0 else np.asarray(kp, float)


def _n_octaves(shape: tuple[int, int], min_image_size: int) -> int:
    return int(min(8, max(1, math.floor(math.log2(max(min(shape), 1) / max(int(min_image_size), 8))) + 1)))


def sift_features(plane: np.ndarray, initial_blur: float = 1.6, steps_per_octave: int = 3, min_image_size: int = 64, max_image_size: int = 1024,
                  descriptor_size: int = 4, orientation_bins: int = 8, max_features: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """SIFT keypoints ((row, col) at full resolution) and descriptors with the parameters of Fiji's SIFT dialog.

    ``initial_blur`` = sigma of the first scale-space level (the input is assumed to carry 0.5), ``steps_per_octave`` =
    DoG levels per octave, the min / max image sizes bound the octaves (the image is downscaled to max_image_size
    first), ``descriptor_size`` × ``descriptor_size`` histograms with ``orientation_bins`` bins each."""
    from skimage.feature import SIFT
    a, f = _rescaled(_norm01(plane), int(max_image_size))
    if min(a.shape) < 16:
        return np.zeros((0, 2)), np.zeros((0, 1), np.uint8)
    s = SIFT(upsampling=1, n_octaves=_n_octaves(a.shape, min_image_size), n_scales=max(int(steps_per_octave), 1), sigma_min=max(float(initial_blur), 0.6),
             sigma_in=0.5, n_hist=max(int(descriptor_size), 1), n_ori=max(int(orientation_bins), 2))
    try:
        s.detect_and_extract(a)
    except (RuntimeError, ValueError, IndexError):
        return np.zeros((0, 2)), np.zeros((0, 1), np.uint8)
    kp, desc = np.asarray(s.keypoints, float), np.asarray(s.descriptors)
    if kp is None or len(kp) == 0:
        return np.zeros((0, 2)), np.zeros((0, 1), np.uint8)
    if max_features and len(kp) > int(max_features):
        sc = getattr(s, "scales", None)
        sel = np.argsort(-np.asarray(sc, float))[:int(max_features)] if sc is not None and len(sc) == len(kp) else np.arange(int(max_features))
        kp, desc = kp[sel], desc[sel]
    return _unscale_keypoints(kp, f), desc


def orb_features(plane: np.ndarray, max_features: int = 500) -> tuple[np.ndarray, np.ndarray]:
    from skimage.feature import ORB
    a = _norm01(plane)
    try:
        orb = ORB(n_keypoints=max(int(max_features) if max_features else 500, 10), fast_threshold=0.05)
        orb.detect_and_extract(a)
    except (RuntimeError, ValueError, IndexError):
        return np.zeros((0, 2)), np.zeros((0, 1), bool)
    if orb.keypoints is None or len(orb.keypoints) == 0:
        return np.zeros((0, 2)), np.zeros((0, 1), bool)
    return np.asarray(orb.keypoints, float), np.asarray(orb.descriptors)


# ============================================================================ MOPS (Multi-Scale Oriented Patches)
def _parabolic(vm: np.ndarray, v0: np.ndarray, vp: np.ndarray) -> np.ndarray:
    den = vm - 2.0 * v0 + vp
    with np.errstate(divide="ignore", invalid="ignore"):
        off = np.where(np.abs(den) > 1e-12, 0.5 * (vm - vp) / den, 0.0)
    return np.clip(np.nan_to_num(off), -0.5, 0.5)


def _dog_scale_space(a: np.ndarray, initial_blur: float, steps: int, min_size: int, contrast: float = 0.01, edge_ratio: float = 10.0,
                     max_octaves: int = 8) -> tuple[list[tuple[int, int, float, float, float, float]], list[list[np.ndarray]]]:
    """Scale-space DoG extrema of a [0, 1] image.

    Returns keypoints ``(octave, level, y, x, sigma, response)`` in the coordinates of their octave and the per-octave
    Gaussian stacks (``octave k`` is the input subsampled ``2**k`` times, every level ``steps`` apart doubles sigma)."""
    img = a.astype(np.float64)
    if initial_blur > 0.5:
        img = ndi.gaussian_filter(img, math.sqrt(initial_blur ** 2 - 0.25))
    steps = max(int(steps), 1)
    k = 2.0 ** (1.0 / steps)
    kps: list[tuple[int, int, float, float, float, float]] = []
    stacks: list[list[np.ndarray]] = []
    octave = 0
    thr = (edge_ratio + 1.0) ** 2 / edge_ratio
    while min(img.shape) >= max(int(min_size), 8) and octave < max_octaves:
        sig = [initial_blur * k ** i for i in range(steps + 3)]
        gauss = [img]
        for i in range(1, steps + 3):
            gauss.append(ndi.gaussian_filter(gauss[-1], math.sqrt(sig[i] ** 2 - sig[i - 1] ** 2)))
        dogs = np.stack([gauss[i + 1] - gauss[i] for i in range(steps + 2)])
        mx = ndi.maximum_filter(dogs, size=3)
        mn = ndi.minimum_filter(dogs, size=3)
        for s in range(1, steps + 1):
            d = dogs[s]
            ext = ((d == mx[s]) | (d == mn[s])) & (np.abs(d) > contrast)
            ext[:2, :] = ext[-2:, :] = ext[:, :2] = ext[:, -2:] = False
            ys, xs = np.nonzero(ext)
            if len(ys) == 0:
                continue
            dxx = d[ys, xs + 1] - 2 * d[ys, xs] + d[ys, xs - 1]
            dyy = d[ys + 1, xs] - 2 * d[ys, xs] + d[ys - 1, xs]
            dxy = (d[ys + 1, xs + 1] - d[ys + 1, xs - 1] - d[ys - 1, xs + 1] + d[ys - 1, xs - 1]) / 4.0
            tr, det = dxx + dyy, dxx * dyy - dxy ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                ok = (det > 1e-12) & (tr ** 2 / np.where(det > 1e-12, det, 1.0) < thr)
            ox = _parabolic(d[ys, xs - 1], d[ys, xs], d[ys, xs + 1])
            oy = _parabolic(d[ys - 1, xs], d[ys, xs], d[ys + 1, xs])
            for y, x, o_x, o_y, good, resp in zip(ys, xs, ox, oy, ok, d[ys, xs]):
                if good:
                    kps.append((octave, s, float(y + o_y), float(x + o_x), float(sig[s]), float(resp)))
        stacks.append(gauss)
        img = gauss[steps][::2, ::2]
        octave += 1
    return kps, stacks


def _dominant_orientation(gy: np.ndarray, gx: np.ndarray, y: float, x: float, sigma: float, n_bins: int = 36) -> float:
    """Dominant gradient direction (radians) in a Gaussian window (1.5 sigma) around (y, x)."""
    h, w = gx.shape
    r = max(int(round(3.0 * sigma)), 2)
    yi, xi = int(round(y)), int(round(x))
    y0, y1, x0, x1 = max(yi - r, 0), min(yi + r + 1, h), max(xi - r, 0), min(xi + r + 1, w)
    if y1 - y0 < 2 or x1 - x0 < 2:
        return 0.0
    sy, sx = gy[y0:y1, x0:x1], gx[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    wgt = np.hypot(sx, sy) * np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * (1.5 * sigma) ** 2))
    ang = np.arctan2(sy, sx)
    bins = np.floor((ang + math.pi) / (2 * math.pi) * n_bins).astype(int) % n_bins
    hist = np.bincount(bins.ravel(), weights=wgt.ravel(), minlength=n_bins)
    for _ in range(2):  # circular smoothing
        hist = (np.roll(hist, 1) + hist + np.roll(hist, -1)) / 3.0
    b = int(np.argmax(hist))
    off = float(_parabolic(np.array(hist[(b - 1) % n_bins]), np.array(hist[b]), np.array(hist[(b + 1) % n_bins])))
    return (b + 0.5 + off) * 2 * math.pi / n_bins - math.pi


def mops_features(plane: np.ndarray, initial_blur: float = 1.6, steps_per_octave: int = 3, min_image_size: int = 64, max_image_size: int = 1024,
                  patch_size: int = 16, max_features: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-Scale Oriented Patches (Brown, Szeliski & Winder 2005; Fiji 'Extract MOPS Correspondences').

    DoG scale-space extrema (same detector as SIFT) with the dominant gradient orientation at the keypoint's scale;
    the descriptor is a ``patch_size`` × ``patch_size`` patch sampled at the keypoint's scale (sample spacing = sigma,
    from the next coarser Gaussian level) in the keypoint's orientation and normalised to zero mean / unit variance.
    Returns keypoints ((row, col) at full resolution), descriptors (N, patch_size²) float32 and the keypoint sigmas."""
    a, f = _rescaled(_norm01(plane), int(max_image_size))
    P = max(int(patch_size), 4)
    if min(a.shape) < 16:
        return np.zeros((0, 2)), np.zeros((0, P * P), np.float32), np.zeros(0)
    kps, stacks = _dog_scale_space(a, float(initial_blur), int(steps_per_octave), int(min_image_size))
    if not kps:
        return np.zeros((0, 2)), np.zeros((0, P * P), np.float32), np.zeros(0)
    if max_features and len(kps) > int(max_features):
        kps = sorted(kps, key=lambda kk: -abs(kk[5]))[:int(max_features)]
    grads: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    u = (np.arange(P) - (P - 1) / 2.0)
    uu, vv = np.meshgrid(u, u)  # patch columns (u) / rows (v)
    pts, descs, sigmas = [], [], []
    for octave, s, y, x, sigma, _resp in kps:
        gauss = stacks[octave]
        if (octave, s) not in grads:
            grads[(octave, s)] = np.gradient(gauss[s])
        gy, gx = grads[(octave, s)]
        theta = _dominant_orientation(gy, gx, y, x, sigma)
        c, sn = math.cos(theta), math.sin(theta)
        xs = x + (uu * c - vv * sn) * sigma
        ys = y + (uu * sn + vv * c) * sigma
        patch = ndi.map_coordinates(gauss[min(s + 1, len(gauss) - 1)], [ys.ravel(), xs.ravel()], order=1, mode="nearest")
        sd = patch.std()
        if sd <= 1e-9:
            continue
        descs.append(((patch - patch.mean()) / sd).astype(np.float32))
        factor = 2.0 ** octave
        pts.append((y * factor, x * factor))
        sigmas.append(sigma * factor)
    if not descs:
        return np.zeros((0, 2)), np.zeros((0, P * P), np.float32), np.zeros(0)
    kp = _unscale_keypoints(np.asarray(pts, float), f)
    return kp, np.stack(descs), np.asarray(sigmas) / f


def match_features(da: np.ndarray, db: np.ndarray, max_ratio: float = 0.92) -> np.ndarray:
    """Cross-checked nearest-neighbour descriptor matches with Lowe's ratio test → (M, 2) index pairs."""
    from skimage.feature import match_descriptors
    if len(da) < 2 or len(db) < 2:
        return np.zeros((0, 2), int)
    metric = "hamming" if da.dtype == bool else "euclidean"
    return np.asarray(match_descriptors(da, db, metric=metric, cross_check=True, max_ratio=float(max_ratio)), int).reshape(-1, 2)


def _feature_pairs(pa: np.ndarray, pb: np.ndarray, method: str, p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Candidate correspondences (src (x, y) on plane a, dst (x, y) on plane b) for a CORRESPONDENCE_METHODS method."""
    info: dict[str, Any] = {"method": method}
    if method == "Block matching":
        H, W = max(pa.shape[0], pb.shape[0]), max(pa.shape[1], pb.shape[1])
        g = int(p.get("grid_spacing", 0)) or int(p["block_size"])
        rows = block_match(_padded_to(pa, H, W), _padded_to(pb, H, W), int(p["block_size"]), int(p["search_radius"]), g)
        used = [r for r in rows if r["r"] >= float(p.get("min_correlation", 0.5))]
        info.update(features=(len(rows), len(rows)), matches=len(used))
        src = np.array([[r["x"], r["y"]] for r in used], float).reshape(-1, 2)
        dst = np.array([[r["x"] + r["dx"], r["y"] + r["dy"]] for r in used], float).reshape(-1, 2)
        return src, dst, info
    common = dict(initial_blur=p["initial_blur"], steps_per_octave=p["steps_per_octave"], min_image_size=p["min_image_size"], max_image_size=p["max_image_size"],
                  max_features=p.get("max_features", 0))
    if method == "SIFT":
        ka, da = sift_features(pa, descriptor_size=p["descriptor_size"], orientation_bins=p["orientation_bins"], **common)
        kb, db = sift_features(pb, descriptor_size=p["descriptor_size"], orientation_bins=p["orientation_bins"], **common)
    elif method == "MOPS":
        ka, da, _ = mops_features(pa, patch_size=p["patch_size"], **common)
        kb, db, _ = mops_features(pb, patch_size=p["patch_size"], **common)
    else:
        ka, da = orb_features(pa, p.get("max_features", 0) or 500)
        kb, db = orb_features(pb, p.get("max_features", 0) or 500)
    info["features"] = (int(len(ka)), int(len(kb)))
    m = match_features(da, db, float(p["max_ratio"])) if len(ka) and len(kb) else np.zeros((0, 2), int)
    info["matches"] = int(len(m))
    if len(m) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2)), info
    return ka[m[:, 0]][:, ::-1], kb[m[:, 1]][:, ::-1], info


def feature_transform(ref: np.ndarray, mov: np.ndarray, mode: str, method: str = "SIFT", **opts: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    """(3×3 moving → reference matrix, info) from matched features + RANSAC on the mode's model; None when the
    geometric consensus fails (fewer inliers than the model needs or than ``min_inlier_ratio`` of the matches)."""
    p = {**SIFT_DEFAULTS, "patch_size": 16, **opts}
    dst, src, info = _feature_pairs(ref, mov, method, p)  # dst on the reference, src on the moving image
    info["features_used"] = False
    model = _POINT_MODEL.get(mode, "Affine")
    n_min = {"Translation": 1, "Rigid body": 2, "Similarity": 2, "Affine": 3, "Projective": 4}[model]
    if len(src) < max(n_min, 2):
        info["inliers"] = 0
        return None, info
    try:
        m, inl = estimate_point_transform(src, dst, model, float(p["max_alignment_error"]), use_ransac=True)
    except ValueError:
        info["inliers"] = 0
        return None, info
    n_in = int(inl.sum())
    info["inliers"] = n_in
    if n_in < max(n_min, 2) or n_in < float(p["min_inlier_ratio"]) * len(src) or not np.all(np.isfinite(m)):
        return None, info
    sx, sy = math.hypot(m[0, 0], m[1, 0]), math.hypot(m[0, 1], m[1, 1])
    if not (0.1 < sx < 10 and 0.1 < sy < 10):
        return None, info
    info["features_used"] = True
    return np.asarray(m, float), info


# ============================================================================ StackReg / TurboReg: multi-resolution intensity optimisation
class BilinearTransform:
    """StackReg's bilinear model: the *inverse* map (reference → moving) ``x_m = c0 + c1·x + c2·y + c3·x·y`` (and the
    same for y) in coordinates centred on the image centre of a ``shape`` (H, W) plane."""

    def __init__(self, coef: np.ndarray, shape: tuple[int, int]):
        self.coef = np.asarray(coef, float).reshape(2, 4)
        self.shape = (int(shape[0]), int(shape[1]))

    @classmethod
    def from_matrix(cls, m: np.ndarray, shape: tuple[int, int]) -> "BilinearTransform":
        """From a forward (moving → reference) 3×3 matrix."""
        n = np.linalg.inv(np.asarray(m, float))
        cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
        c = np.array([cx, cy, 1.0])
        off = n @ c
        coef = np.array([[off[0] - cx, n[0, 0], n[0, 1], 0.0], [off[1] - cy, n[1, 0], n[1, 1], 0.0]])
        return cls(coef, shape)

    @property
    def matrix(self) -> np.ndarray:
        """Forward (moving → reference) affine approximation (the x·y terms dropped) for tables and messages."""
        cy, cx = (self.shape[0] - 1) / 2.0, (self.shape[1] - 1) / 2.0
        c = self.coef
        n = np.eye(3)
        n[0, :2], n[1, :2] = c[0, 1:3], c[1, 1:3]
        n[0, 2] = c[0, 0] + cx - (c[0, 1] * cx + c[0, 2] * cy)
        n[1, 2] = c[1, 0] + cy - (c[1, 1] * cx + c[1, 2] * cy)
        try:
            return np.linalg.inv(n)
        except np.linalg.LinAlgError:
            return IDENTITY.copy()

    def coords(self, shape: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(x_m, y_m) absolute moving-image coordinates sampled by every reference pixel."""
        h, w = shape or self.shape
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        xc, yc = xx - cx, yy - cy
        c = self.coef
        xm = cx + c[0, 0] + c[0, 1] * xc + c[0, 2] * yc + c[0, 3] * xc * yc
        ym = cy + c[1, 0] + c[1, 1] * xc + c[1, 2] * yc + c[1, 3] * xc * yc
        return xm, ym

    def warp(self, plane: np.ndarray, order: int = 1) -> np.ndarray:
        from skimage.transform import warp
        p = _f32(plane)
        xm, ym = self.coords(p.shape)
        return warp(p, np.stack([ym, xm]), order=order, mode="constant", cval=0.0, preserve_range=True).astype(np.float32)

    def rescaled(self, new_shape: tuple[int, int]) -> "BilinearTransform":
        """The same map expressed on a plane of another size (pyramid level): centred coordinates scale by the size ratio."""
        fy, fx = self.shape[0] / new_shape[0], self.shape[1] / new_shape[1]
        c = self.coef
        coef = np.array([[c[0, 0] / fx, c[0, 1], c[0, 2] * fy / fx, c[0, 3] * fy],
                         [c[1, 0] / fy, c[1, 1] * fx / fy, c[1, 2], c[1, 3] * fx]])
        return BilinearTransform(coef, new_shape)


def _level_matrix(old_shape: tuple[int, int], new_shape: tuple[int, int]) -> np.ndarray:
    """P with old_coords = P · new_coords (pixel-centre convention of skimage's resize)."""
    fy, fx = old_shape[0] / new_shape[0], old_shape[1] / new_shape[1]
    return np.array([[fx, 0.0, 0.5 * fx - 0.5], [0.0, fy, 0.5 * fy - 0.5], [0.0, 0.0, 1.0]])


def _rescale_transform(tf: Any, old_shape: tuple[int, int], new_shape: tuple[int, int]) -> Any:
    if isinstance(tf, BilinearTransform):
        return tf.rescaled(new_shape)
    P = _level_matrix(old_shape, new_shape)
    return np.linalg.inv(P) @ tf @ P


def _small_update(mode: str, p: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """Reference-frame update matrix of a Gauss-Newton step (dx, dy[, angle[, scale]] / affine terms) about the centre."""
    if mode == "Translation":
        lin, off = np.eye(2), np.array([p[0], p[1]])
    elif mode == "Rigid body":
        th = float(p[2])
        lin, off = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]), np.array([p[0], p[1]])
    elif mode == "Scaled rotation":
        th, sc = float(p[2]), math.exp(float(p[3]))
        lin, off = sc * np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]), np.array([p[0], p[1]])
    else:
        lin, off = np.array([[1 + p[2], p[3]], [p[4], 1 + p[5]]]), np.array([p[0], p[1]])
    d = np.eye(3)
    d[:2, :2] = lin
    d[:2, 2] = off + np.array([cx, cy]) - lin @ np.array([cx, cy])
    return d


def _gn_matrix_level(r: np.ndarray, mv: np.ndarray, m: np.ndarray, mode: str, iterations: int) -> tuple[np.ndarray, float]:
    """Forward-compositional Gauss-Newton on z-scored intensities at one pyramid level. Returns (matrix, final error)."""
    h, w = r.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    xm, ym = (xx - cx).astype(np.float64), (yy - cy).astype(np.float64)
    ones = np.ones_like(mv, dtype=np.float32)
    best, best_err, cur = m, np.inf, m
    converged = False
    for it in range(int(iterations) + 1):
        wv = _warp(mv, cur, 1).astype(np.float64)
        valid = _warp(ones, cur, 1) > 0.99
        valid[:1, :] = valid[-1:, :] = valid[:, :1] = valid[:, -1:] = False
        if valid.sum() < 64:
            break
        a, b = r[valid], wv[valid]
        sa, sb = a.std(), b.std()
        if sa <= 1e-12 or sb <= 1e-12:
            break
        an = (a - a.mean()) / sa
        bn = (wv - b.mean()) / sb
        resid = an - bn[valid]
        err = float(np.mean(resid ** 2))
        if err < best_err - 1e-9:
            best, best_err = cur, err
        elif err > best_err * 1.001:
            break
        if converged or it == int(iterations):
            break
        gy, gx = np.gradient(bn)
        gx, gy = gx[valid], gy[valid]
        x, y = xm[valid], ym[valid]
        if mode == "Translation":
            cols = [-gx, -gy]
        elif mode == "Rigid body":
            cols = [-gx, -gy, -(gy * x - gx * y)]
        elif mode == "Scaled rotation":
            cols = [-gx, -gy, -(gy * x - gx * y), -(gx * x + gy * y)]
        else:
            cols = [-gx, -gy, -gx * x, -gx * y, -gy * x, -gy * y]
        A = np.stack(cols, axis=1)
        try:
            p = np.linalg.lstsq(A, resid, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(p)):
            break
        # damp overly large steps (coarse levels with little overlap)
        lim = max(abs(p[0]) / (0.25 * w), abs(p[1]) / (0.25 * h), (np.abs(p[2:]).max() / 0.3 if len(p) > 2 else 0.0))
        if lim > 1:
            p = p / lim
        cur = _small_update(mode, p, cx, cy) @ cur
        if abs(p[0]) < 1e-3 and abs(p[1]) < 1e-3 and (len(p) == 2 or np.all(np.abs(p[2:]) < 1e-5)):
            converged = True
    return best, best_err


def _gn_bilinear_level(r: np.ndarray, mv: np.ndarray, tf: BilinearTransform, iterations: int) -> tuple[BilinearTransform, float]:
    """Forward-additive Gauss-Newton of the 8 bilinear coefficients at one pyramid level."""
    h, w = r.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xc, yc = xx - cx, yy - cy
    gy_m, gx_m = np.gradient(mv)
    best, best_err, cur = tf, np.inf, tf
    converged = False
    for it in range(int(iterations) + 1):
        xm, ym = cur.coords((h, w))
        inside = (xm >= 0) & (xm <= w - 1) & (ym >= 0) & (ym <= h - 1)
        inside[:1, :] = inside[-1:, :] = inside[:, :1] = inside[:, -1:] = False
        if inside.sum() < 64:
            break
        samp = [ym[inside], xm[inside]]
        wv = ndi.map_coordinates(mv, samp, order=1, mode="constant", cval=0.0)
        a = r[inside]
        sa, sb = a.std(), wv.std()
        if sa <= 1e-12 or sb <= 1e-12:
            break
        resid = (a - a.mean()) / sa - (wv - wv.mean()) / sb
        err = float(np.mean(resid ** 2))
        if err < best_err - 1e-9:
            best, best_err = cur, err
        elif err > best_err * 1.001:
            break
        if converged or it == int(iterations):
            break
        Gx = ndi.map_coordinates(gx_m, samp, order=1, mode="constant", cval=0.0) / sb
        Gy = ndi.map_coordinates(gy_m, samp, order=1, mode="constant", cval=0.0) / sb
        x, y = xc[inside], yc[inside]
        A = np.stack([Gx, Gx * x, Gx * y, Gx * x * y, Gy, Gy * x, Gy * y, Gy * x * y], axis=1)
        try:
            p = np.linalg.lstsq(A, resid, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(p)):
            break
        lim = max(abs(p[0]) / (0.25 * w), abs(p[4]) / (0.25 * h), max(abs(p[1]), abs(p[2]), abs(p[5]), abs(p[6])) / 0.3)
        if lim > 1:
            p = p / lim
        cur = BilinearTransform(cur.coef + p.reshape(2, 4), (h, w))
        if np.all(np.abs(p) < 1e-6):
            converged = True
    return best, best_err


def pyramid_transform(ref: np.ndarray, mov: np.ndarray, mode: str, levels: int = 0, iterations: int = 30, init: Any = None,
                      smooth: float = 0.8) -> tuple[Any, dict[str, Any]]:
    """StackReg / TurboReg-style estimate of the moving → reference transform by multi-resolution least-squares
    intensity optimisation (Gaussian pyramid, Gauss-Newton / Lucas-Kanade at every level, coarse to fine) for
    translation, rigid body, scaled rotation, affine or bilinear. Returns (transform, info) where the transform is
    a 3×3 matrix or a :class:`BilinearTransform`."""
    from skimage.transform import pyramid_reduce
    r0 = _f32(ref).astype(np.float64)
    m0 = _f32(mov).astype(np.float64)
    if r0.shape != m0.shape:
        H, W = max(r0.shape[0], m0.shape[0]), max(r0.shape[1], m0.shape[1])
        r0, m0 = _padded_to(r0, H, W).astype(np.float64), _padded_to(m0, H, W).astype(np.float64)
    H, W = r0.shape
    if int(levels) <= 0:
        levels = int(max(0, math.floor(math.log2(max(min(H, W), 1) / 32.0))))
    levels = int(min(levels, 6))
    pyr_r, pyr_m = [r0], [m0]
    for _ in range(levels):
        if min(pyr_r[-1].shape) < 16:
            break
        pyr_r.append(pyramid_reduce(pyr_r[-1], downscale=2, order=1, preserve_range=True))
        pyr_m.append(pyramid_reduce(pyr_m[-1], downscale=2, order=1, preserve_range=True))
    levels = len(pyr_r) - 1
    bilinear = mode == "Bilinear"
    gn_mode = "Affine" if bilinear else mode
    tf: Any = IDENTITY.copy() if init is None else init
    if bilinear and not isinstance(tf, BilinearTransform):
        tf = BilinearTransform.from_matrix(tf, (H, W))
    elif not bilinear and isinstance(tf, BilinearTransform):
        tf = tf.matrix
    tf = _rescale_transform(tf, (H, W), pyr_r[levels].shape)
    err = float("nan")
    for L in range(levels, -1, -1):
        if L < levels:
            tf = _rescale_transform(tf, pyr_r[L + 1].shape, pyr_r[L].shape)
        r = ndi.gaussian_filter(pyr_r[L], smooth) if smooth > 0 else pyr_r[L]
        m = ndi.gaussian_filter(pyr_m[L], smooth) if smooth > 0 else pyr_m[L]
        if bilinear:
            tf, err = _gn_bilinear_level(r, m, tf, iterations)
        else:
            tf, err = _gn_matrix_level(r, m, tf, gn_mode, iterations)
    corr = float("nan") if not np.isfinite(err) else float(max(-1.0, min(1.0, 1.0 - err / 2.0)))  # z-scored MSE ↔ correlation
    return tf, {"levels": levels + 1, "error": err, "correlation": corr, "features_used": False}


# ============================================================================ estimator factories for align_stack
def make_sift_estimator(mode: str, opts: dict[str, Any], upsample: int = 10) -> Callable[..., tuple[np.ndarray, dict[str, Any]]]:
    """estimate(ref, mov, k_ref, k_mov) → (matrix, info): SIFT + RANSAC, falling back to phase correlation."""
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    fk = {k: opts[k] for k in ("initial_blur", "steps_per_octave", "min_image_size", "max_image_size", "descriptor_size", "orientation_bins") if k in opts}

    def features(plane: np.ndarray, k: int | None) -> tuple[np.ndarray, np.ndarray]:
        if k is not None and k in cache:
            return cache[k]
        f = sift_features(plane, **fk)
        if k is not None:
            cache[k] = f
        return f

    def estimate(ref: np.ndarray, mov: np.ndarray, k_ref: int | None = None, k_mov: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        ka, da = features(ref, k_ref)
        kb, db = features(mov, k_mov)
        info: dict[str, Any] = {"features": (int(len(ka)), int(len(kb))), "matches": 0, "inliers": 0, "features_used": False}
        m = None
        if len(ka) >= 2 and len(kb) >= 2:
            mt = match_features(da, db, float(opts.get("max_ratio", 0.92)))
            info["matches"] = int(len(mt))
            model = _POINT_MODEL.get(mode, "Affine")
            n_min = {"Translation": 1, "Rigid body": 2, "Similarity": 2, "Affine": 3}[model]
            if len(mt) >= max(n_min, 2):
                dst, src = ka[mt[:, 0]][:, ::-1], kb[mt[:, 1]][:, ::-1]
                try:
                    mm, inl = estimate_point_transform(src, dst, model, float(opts.get("max_alignment_error", 25.0)), use_ransac=True)
                    n_in = int(inl.sum())
                    info["inliers"] = n_in
                    sx, sy = math.hypot(mm[0, 0], mm[1, 0]), math.hypot(mm[0, 1], mm[1, 1])
                    if n_in >= max(n_min, 2) and n_in >= float(opts.get("min_inlier_ratio", 0.05)) * len(mt) and np.all(np.isfinite(mm)) and 0.1 < sx < 10 and 0.1 < sy < 10:
                        m = np.asarray(mm, float)
                except ValueError:
                    pass
        if m is None:
            dy, dx = _phase_shift(ref, mov, upsample)
            return _translation(dx, dy), info
        info["features_used"] = True
        return m, info

    return estimate


def make_intensity_estimator(mode: str, levels: int, iterations: int) -> Callable[..., tuple[Any, dict[str, Any]]]:
    def estimate(ref: np.ndarray, mov: np.ndarray, k_ref: int | None = None, k_mov: int | None = None) -> tuple[Any, dict[str, Any]]:
        return pyramid_transform(ref, mov, mode, levels, iterations)

    return estimate


# ============================================================================ descriptor-based registration (Preibisch): detections + local geometric descriptors
def dog_sigma(size: float, ndim: int) -> float:
    """DoG sigma (px) for objects of ``size`` px diameter (blob radius ≈ sigma·√ndim)."""
    return max(float(size), 1.0) / (2.0 * math.sqrt(ndim))


def geometric_descriptors(pts: np.ndarray, n_neighbors: int = 3, redundancy: int = 1, invariance: str = "similarity") -> tuple[np.ndarray, np.ndarray]:
    """Local geometric descriptors of a point cloud (2-D or 3-D): every point is described by its ``n_neighbors``
    nearest neighbours (ordered by distance); with ``redundancy`` r every subset of n out of the n + r nearest
    neighbours is a descriptor of its own. ``invariance``: 'translation' → the neighbour offsets, 'rigid' → all pairwise
    distances of the point and its neighbours, 'similarity' → the same divided by the largest point–neighbour distance.
    Returns (descriptors (M, L), owner point index (M,))."""
    pts = np.asarray(pts, float)
    N = len(pts)
    n = max(int(n_neighbors), 1)
    k = n + max(int(redundancy), 0)
    if N <= k:
        raise ValueError(f"Needs more than {k} detections ({N} found): lower the threshold or the size, or reduce redundancy")
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=k + 1)
    idx = idx[:, 1:]
    descs, owners = [], []
    for comb in itertools.combinations(range(k), n):
        nb = pts[idx[:, list(comb)]]  # (N, n, D) ordered by distance to the point
        rel = nb - pts[:, None, :]
        if invariance == "translation":
            desc = rel.reshape(N, -1)
        else:
            dp = np.linalg.norm(rel, axis=2)
            pair = [np.linalg.norm(nb[:, i] - nb[:, j], axis=1) for i in range(n) for j in range(i + 1, n)]
            desc = np.column_stack([dp] + pair)
            if invariance == "similarity":
                sc = dp[:, -1:].copy()
                sc[sc <= 1e-12] = 1.0
                desc = desc / sc
        descs.append(desc)
        owners.append(np.arange(N))
    return np.vstack(descs), np.concatenate(owners)


def match_geometric(da: np.ndarray, oa: np.ndarray, db: np.ndarray, ob: np.ndarray, significance: float = 3.0) -> np.ndarray:
    """Candidate point pairs (ia, ib): the nearest descriptor of b must be ``significance`` times closer than the nearest
    descriptor that belongs to another point (Fiji's 'significance'); one-to-one by the best descriptor distance."""
    if len(da) == 0 or len(db) == 0:
        return np.zeros((0, 2), int)
    tree = cKDTree(db)
    kq = int(min(len(db), 12))
    dist, idx = tree.query(da, k=kq)
    if kq == 1:
        dist, idx = dist[:, None], idx[:, None]
    cands: dict[tuple[int, int], float] = {}
    for i in range(len(da)):
        ib = int(ob[idx[i, 0]])
        d0 = float(dist[i, 0])
        d1 = math.inf
        for j in range(1, kq):
            if int(ob[idx[i, j]]) != ib:
                d1 = float(dist[i, j])
                break
        if d0 <= 1e-12 or d1 >= float(significance) * d0:
            key = (int(oa[i]), ib)
            if d0 < cands.get(key, math.inf):
                cands[key] = d0
    best_a: dict[int, tuple[int, float]] = {}
    for (ia, ib), d in cands.items():
        if d < best_a.get(ia, (0, math.inf))[1]:
            best_a[ia] = (ib, d)
    best_b: dict[int, tuple[int, float]] = {}
    for ia, (ib, d) in best_a.items():
        if d < best_b.get(ib, (0, math.inf))[1]:
            best_b[ib] = (ia, d)
    pairs = sorted((ia, ib) for ib, (ia, _) in best_b.items())
    return np.asarray(pairs, int).reshape(-1, 2)


def fit_model_nd(src: np.ndarray, dst: np.ndarray, model: str) -> np.ndarray:
    """(D+1)×(D+1) homogeneous matrix mapping src points onto dst points: translation (mean), rigid / similarity
    (Umeyama / Kabsch), affine (least squares)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n, D = src.shape
    M = np.eye(D + 1)
    if model == "Translation":
        M[:D, D] = (dst - src).mean(axis=0)
        return M
    if model in ("Rigid body", "Similarity"):
        mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
        xs, xd = src - mu_s, dst - mu_d
        cov = xd.T @ xs / n
        U, S, Vt = np.linalg.svd(cov)
        d = np.ones(D)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            d[-1] = -1.0
        R = U @ np.diag(d) @ Vt
        var_s = float((xs ** 2).sum() / n)
        c = float((S * d).sum() / var_s) if (model == "Similarity" and var_s > 1e-12) else 1.0
        M[:D, :D] = c * R
        M[:D, D] = mu_d - c * R @ mu_s
        return M
    A = np.c_[src, np.ones(n)]
    sol, _, rank, _ = np.linalg.lstsq(A, dst, rcond=None)
    if rank < D + 1:
        raise ValueError("degenerate")
    M[:D, :D] = sol[:D].T
    M[:D, D] = sol[D]
    return M


def apply_nd(M: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, float)
    D = pts.shape[1]
    return pts @ M[:D, :D].T + M[:D, D]


def min_samples(model: str, ndim: int) -> int:
    return {"Translation": 1, "Rigid body": ndim, "Similarity": ndim, "Affine": ndim + 1}[model]


def ransac_nd(src: np.ndarray, dst: np.ndarray, model: str, threshold: float = 5.0, iterations: int = 1000, min_inlier_ratio: float = 0.1,
              min_inliers: int = 0, seed: int = 0) -> tuple[np.ndarray | None, np.ndarray]:
    """RANSAC over candidate pairs for a DESCRIPTOR_MODELS model in 2-D or 3-D: (matrix refined on the inliers, inlier mask),
    or (None, all False) when fewer than ``min_inliers`` (≥ the model's minimum) or ``min_inlier_ratio`` of the pairs agree."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    D = src.shape[1]
    ms = min_samples(model, D)
    none = np.zeros(n, bool)
    if n < ms:
        return None, none
    rng = np.random.default_rng(seed)
    best_inl: np.ndarray | None = None
    best_n, best_err = 0, math.inf
    tries = 1 if n == ms else max(int(iterations), 1)
    for _ in range(tries):
        sample = np.arange(n) if n == ms else rng.choice(n, ms, replace=False)
        try:
            M = fit_model_nd(src[sample], dst[sample], model)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(M)):
            continue
        det = abs(np.linalg.det(M[:D, :D]))
        if not (1e-3 < det < 1e3):
            continue
        res = np.linalg.norm(apply_nd(M, src) - dst, axis=1)
        inl = res <= float(threshold)
        k = int(inl.sum())
        err = float(res[inl].sum())
        if k > best_n or (k == best_n and err < best_err):
            best_inl, best_n, best_err = inl, k, err
            if k == n:
                break
    need = max(ms, int(min_inliers))
    if best_inl is None or best_n < need or best_n < float(min_inlier_ratio) * n:
        return None, none
    inl = best_inl
    M = None
    for _ in range(2):  # refit on the inliers, re-classify
        try:
            M = fit_model_nd(src[inl], dst[inl], model)
        except (ValueError, np.linalg.LinAlgError):
            break
        res = np.linalg.norm(apply_nd(M, src) - dst, axis=1)
        new = res <= float(threshold)
        if new.sum() < need:
            break
        inl = new
    if M is None or not np.all(np.isfinite(M)):
        return None, none
    return M, inl


def descriptor_match(pa: np.ndarray, pb: np.ndarray, model: str, n_neighbors: int = 3, redundancy: int = 1, significance: float = 3.0,
                     threshold: float = 5.0, iterations: int = 1000, min_inlier_ratio: float = 0.1, min_inliers: int = 0) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict[str, Any]]:
    """Descriptor-based matching of two point clouds (same dimensionality): (matrix a → b or None, inlier src points,
    inlier dst points, info with 'candidates' / 'inliers')."""
    inv = "translation" if model == "Translation" else ("rigid" if model == "Rigid body" else "similarity")
    info: dict[str, Any] = {"detections": (int(len(pa)), int(len(pb))), "candidates": 0, "inliers": 0}
    da, oa = geometric_descriptors(pa, n_neighbors, redundancy, inv)
    db, ob = geometric_descriptors(pb, n_neighbors, redundancy, inv)
    pairs = match_geometric(da, oa, db, ob, significance)
    info["candidates"] = int(len(pairs))
    if len(pairs) == 0:
        return None, np.zeros((0, pa.shape[1])), np.zeros((0, pa.shape[1])), info
    src, dst = pa[pairs[:, 0]], pb[pairs[:, 1]]
    M, inl = ransac_nd(src, dst, model, threshold, iterations, min_inlier_ratio, min_inliers)
    info["inliers"] = int(inl.sum())
    if M is None:
        return None, np.zeros((0, pa.shape[1])), np.zeros((0, pa.shape[1])), info
    return M, src[inl], dst[inl], info


DETECTION_PARAMS = [
    Param("objects", "choice", "Bright", choices=OBJECTS, label="Brightness of objects", help="Detect bright blobs (DoG maxima), dark blobs (minima) or both"),
    Param("size", "float", 5.0, min=1, max=200, step=0.5, label="Approximate object size (px)", help="Diameter of the detected objects (beads, nuclei…) in xy pixels"),
    Param("threshold", "float", 0.05, min=0.0, max=1.0, step=0.01, label="Detection threshold", help="Fraction of the strongest DoG response a detection must reach (lower = more detections)"),
    Param("max_detections", "int", 500, min=4, max=20000, help="Keep at most this many (strongest) detections per image"),
]
MATCH_PARAMS = [
    Param("n_neighbors", "int", 3, min=2, max=8, label="Number of neighbours", help="Neighbouring detections that make up a local descriptor"),
    Param("redundancy", "int", 1, min=0, max=4, help="Extra neighbours: every subset of n out of n + redundancy neighbours is tried"),
    Param("significance", "float", 3.0, min=1.0, max=50, step=0.5, help="The best descriptor match must be this many times closer than the second best"),
    Param("ransac_threshold", "float", 5.0, min=0.1, max=500, step=0.5, label="Allowed error for RANSAC (px)"),
    Param("ransac_iterations", "int", 1000, min=1, max=100000, label="RANSAC iterations", advanced=True),
    Param("min_inlier_ratio", "float", 0.1, min=0.0, max=1.0, step=0.05, label="Minimal inlier ratio", advanced=True),
    Param("min_inliers", "int", 6, min=1, max=10000, label="Minimal number of inliers", advanced=True),
]


def _detect_2d(plane: np.ndarray, objects: str, size: float, threshold: float, max_detections: int) -> np.ndarray:
    pts, _ = detect_dog_nd(plane, dog_sigma(size, 2), objects, threshold, int(max_detections))
    return pts


def _detect_3d(vol: np.ndarray, voxel: tuple[float, float, float], objects: str, size: float, threshold: float, max_detections: int) -> np.ndarray:
    """(N, 3) detections (x, y, z) in *calibrated* units."""
    sx, sy, sz = voxel
    s = dog_sigma(size, 3)
    pts, _ = detect_dog_nd(vol, (s * sx / sz, s * sx / sy, s), objects, threshold, int(max_detections))
    return pts * np.array([sx, sy, sz])


def _matrix_to_voxels(m_cal: np.ndarray, vox_in: tuple[float, float, float], vox_out: tuple[float, float, float]) -> np.ndarray:
    """A 4×4 matrix fitted in calibrated space → voxel space (source voxels → target voxels)."""
    s_in, s_out = np.diag(np.asarray(vox_in, float)), np.diag(1.0 / np.asarray(vox_out, float))
    m = np.eye(4)
    m[:3, :3] = s_out @ m_cal[:3, :3] @ s_in
    m[:3, 3] = s_out @ m_cal[:3, 3]
    return m


def _match_rois(img: LumenImage, pts: np.ndarray, name: str = "Matches", color: str = "#00ff00") -> list[dict[str, Any]]:
    """Point ROIs (Lumen coordinates) of inlier points; 3-D points are grouped per slice."""
    pts = np.asarray(pts, float)
    if pts.size == 0:
        return []
    d = img.display
    if pts.shape[1] == 2:
        return [{"type": "point", "points": (pts + 0.5).round(3).tolist(), "name": name, "color": color, "image_id": img.id, "position": {"t": d.t, "z": d.z, "c": d.c}}]
    out = []
    zs = np.clip(np.rint(pts[:, 2]).astype(int), 0, max(img.nz - 1, 0))
    for z in sorted(set(zs.tolist())):
        sel = pts[zs == z][:, :2] + 0.5
        out.append({"type": "point", "points": sel.round(3).tolist(), "name": f"{name} z{z + 1}", "color": color, "image_id": img.id, "position": {"t": d.t, "z": int(z), "c": d.c}})
    return out


def _correspondence_table(src: np.ndarray, dst: np.ndarray, resid: np.ndarray, ndim: int, offset: float = 0.5) -> dict[str, Any]:
    cols = ["Match", "X1", "Y1"] + (["Z1"] if ndim == 3 else []) + ["X2", "Y2"] + (["Z2"] if ndim == 3 else []) + ["Residual"]
    rows = []
    for i, (a, b, r) in enumerate(zip(src, dst, resid)):
        ra = [round(float(a[0]) + offset, 2), round(float(a[1]) + offset, 2)] + ([round(float(a[2]) + 1, 2)] if ndim == 3 else [])
        rb = [round(float(b[0]) + offset, 2), round(float(b[1]) + offset, 2)] + ([round(float(b[2]) + 1, 2)] if ndim == 3 else [])
        rows.append([i + 1] + ra + rb + [round(float(r), 3)])
    return {"name": "Correspondences", "columns": cols, "rows": rows}


# ================================================================================ ops
@op("registration.extract_correspondences", "Extract Correspondences (SIFT / MOPS / ORB / Block matching)", CAT, kind="analysis", preview=False, roi_aware=False, menu="Plugins/Feature Extraction",
    params=[Param("target", "image", None, help="The open image to match the current image with"),
            Param("method", "choice", "SIFT", choices=CORRESPONDENCE_METHODS, help="Feature detector / descriptor; Block matching = normalised cross-correlation of blocks on a grid"),
            Param("model", "choice", "Rigid body", choices=FEATURE_MODELS, label="Expected transformation", help="Model fitted with RANSAC to filter the matches (geometric consensus)"),
            Param("initial_blur", "float", 1.6, min=0.5, max=10, step=0.1, label="Initial Gaussian blur (px)"),
            Param("steps_per_octave", "int", 3, min=1, max=10, label="Steps per scale octave"),
            Param("min_image_size", "int", 64, min=8, max=4096, label="Minimum image size (px)"), Param("max_image_size", "int", 1024, min=16, max=16384, label="Maximum image size (px)"),
            Param("descriptor_size", "int", 0, min=0, max=64, label="Feature descriptor size", help="SIFT: histograms per side (Fiji default 4); MOPS: patch side in samples (Fiji default 16); 0 = the method's default"),
            Param("orientation_bins", "int", 8, min=2, max=36, label="Descriptor orientation bins (SIFT)"),
            Param("max_ratio", "float", 0.92, min=0.1, max=1.0, step=0.01, label="Closest / next closest ratio"),
            Param("max_alignment_error", "float", 25.0, min=0.1, max=1000, step=1, label="Maximal alignment error (px)", help="RANSAC inlier residual"),
            Param("min_inlier_ratio", "float", 0.05, min=0.0, max=1.0, step=0.01, label="Inlier ratio", help="Smallest accepted inliers / matches"),
            Param("max_features", "int", 0, min=0, max=20000, help="Keep at most this many features per image (0 = all)", advanced=True),
            Param("block_size", "int", 32, min=4, max=512, label="Block size (block matching)", advanced=True), Param("search_radius", "int", 8, min=1, max=256, label="Search radius (block matching)", advanced=True),
            Param("min_correlation", "float", 0.5, min=-1, max=1, step=0.05, label="Min block correlation", advanced=True),
            Param("add_rois", "bool", True, label="Point ROIs 'Matches'", help="Add the inlier points as point ROIs on both images")],
    description="Fiji Plugins ▸ Feature Extraction: SIFT, MOPS (multi-scale oriented patches), ORB or block-matching correspondences between the current image and a target image, filtered by RANSAC on the expected transformation: inlier points as 'Matches' point ROIs on both images, a 'Correspondences' table and the transform in the Results table (reused by 'Landmark Transform'). The older 'Extract SIFT Correspondences' command stays available.",
    keywords=["sift", "mops", "orb", "block matching", "correspondences", "feature extraction", "ransac", "landmarks", "matches"])
def extract_correspondences(ctx, target=None, method="SIFT", model="Rigid body", initial_blur=1.6, steps_per_octave=3, min_image_size=64, max_image_size=1024, descriptor_size=0,
                            orientation_bins=8, max_ratio=0.92, max_alignment_error=25.0, min_inlier_ratio=0.05, max_features=0, block_size=32, search_radius=8, min_correlation=0.5, add_rois=True):
    img = ctx.img
    if not target:
        raise ValueError("Choose a target image")
    tgt = ctx.session.get_image(target)
    if tgt.id == img.id:
        raise ValueError("Target must be a different image")
    pa, pb = _estimation_plane(img), _estimation_plane(tgt)
    p = dict(initial_blur=initial_blur, steps_per_octave=steps_per_octave, min_image_size=min_image_size, max_image_size=max_image_size,
             descriptor_size=int(descriptor_size) or 4, patch_size=int(descriptor_size) or 16, orientation_bins=orientation_bins, max_ratio=max_ratio, max_features=max_features,
             block_size=block_size, search_radius=search_radius, min_correlation=min_correlation)
    ctx.progress(0.1, f"{method} features")
    src, dst, info = _feature_pairs(pa, pb, method, p)
    n_min = {"Translation": 1, "Rigid body": 2, "Similarity": 2, "Affine": 3, "Projective": 4}[model]
    if len(src) < max(n_min, 2):
        raise ValueError(f"Only {len(src)} {method} matches between {img.name} and {tgt.name} ({info['features'][0]} / {info['features'][1]} features) – not enough to estimate a {model.lower()} transform")
    ctx.progress(0.7, "Geometric consensus (RANSAC)")
    m, inl = estimate_point_transform(src, dst, model, float(max_alignment_error), use_ransac=True)
    n_in = int(inl.sum())
    if n_in < max(n_min, 2) or n_in < float(min_inlier_ratio) * len(src):
        raise ValueError(f"No consistent matches: {n_in} inliers of {len(src)} {method} matches ({model}, max error {max_alignment_error:g} px)")
    s_in, d_in = src[inl], dst[inl]
    resid = np.hypot(*(_apply_matrix(m, s_in) - d_in).T)
    table = _correspondence_table(s_in, d_in, resid, 2)
    params = transform_params(m)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    row = {"Label": f"{img.name} → {tgt.name}", "Method": method, "Model": model, "N features": int(info["features"][0]), "N features 2": int(info["features"][1]),
           "N matches": int(len(src)), "N inliers": n_in, **params, "RMS residual": rms}
    ctx.session.settings[CORRESPONDENCES_KEY] = {"source": img.id, "target": tgt.id, "src": (s_in + 0.5).tolist(), "dst": (d_in + 0.5).tolist(), "matrix": m.tolist(), "model": model, "method": method}
    rois = _match_rois(img, s_in) + _match_rois(tgt, d_in) if add_rois else []
    msg = (f"{n_in} inliers of {len(src)} {method} matches ({model}; {info['features'][0]} / {info['features'][1]} features): dx {params['dx']:+.2f}, dy {params['dy']:+.2f}, "
           f"angle {params['Angle']:+.2f}°, scale {params['Scale']:.3f}, RMS {rms:.2f} px")
    return OpResult(results=[row], table=table, rois_add=rois, message=msg, log=msg)


def _descriptor_pair_2d(pa: np.ndarray, pb: np.ndarray, model: str, det: dict[str, Any], mp: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict[str, Any]]:
    a = _detect_2d(pa, det["objects"], det["size"], det["threshold"], det["max_detections"])
    b = _detect_2d(pb, det["objects"], det["size"], det["threshold"], det["max_detections"])
    try:
        return descriptor_match(a, b, model, **mp)
    except ValueError as e:
        return None, np.zeros((0, 2)), np.zeros((0, 2)), {"detections": (int(len(a)), int(len(b))), "candidates": 0, "inliers": 0, "error": str(e)}


def _descriptor_pair_3d(va: np.ndarray, vox_a: tuple[float, float, float], vb: np.ndarray, vox_b: tuple[float, float, float], model: str, det: dict[str, Any],
                        mp: dict[str, Any]) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict[str, Any]]:
    """3-D pair in calibrated space; the returned matrix maps a voxels → b voxels, points are returned in voxels."""
    a = _detect_3d(va, vox_a, det["objects"], det["size"], det["threshold"], det["max_detections"])
    b = _detect_3d(vb, vox_b, det["objects"], det["size"], det["threshold"], det["max_detections"])
    mp = {**mp, "threshold": float(mp["threshold"]) * vox_b[0]}
    try:
        M, s_in, d_in, info = descriptor_match(a, b, model, **mp)
    except ValueError as e:
        return None, np.zeros((0, 3)), np.zeros((0, 3)), {"detections": (int(len(a)), int(len(b))), "candidates": 0, "inliers": 0, "error": str(e)}
    if M is None:
        return None, np.zeros((0, 3)), np.zeros((0, 3)), info
    return _matrix_to_voxels(M, vox_a, vox_b), s_in / np.array(vox_a), d_in / np.array(vox_b), info


def _mp(n_neighbors, redundancy, significance, ransac_threshold, ransac_iterations, min_inlier_ratio, min_inliers) -> dict[str, Any]:
    return dict(n_neighbors=int(n_neighbors), redundancy=int(redundancy), significance=float(significance), threshold=float(ransac_threshold), iterations=int(ransac_iterations),
                min_inlier_ratio=float(min_inlier_ratio), min_inliers=int(min_inliers))


def _warp_planes_2d(img: LumenImage, m: np.ndarray, out_shape: tuple[int, int], order: int) -> np.ndarray:
    """Every plane of ``img`` warped by the (x, y) matrix onto a canvas of ``out_shape``."""
    ht, wt = out_shape
    hh, ww = max(ht, img.height), max(wt, img.width)
    out = np.zeros((img.nt, img.nz, img.nc, ht, wt), np.float32)
    for t, z, c in img.iter_planes():
        canvas = np.zeros((hh, ww), np.float32)
        canvas[:img.height, :img.width] = img.data[t, z, c]
        out[t, z, c] = _warp(canvas, m, order)[:ht, :wt]
    return out


def _warp_volumes_3d(img: LumenImage, m: np.ndarray, out_shape: tuple[int, int, int], order: int) -> np.ndarray:
    out = np.zeros((img.nt, out_shape[0], img.nc, out_shape[1], out_shape[2]), np.float32)
    for t in range(img.nt):
        for c in range(img.nc):
            out[t, :, c] = _warp_volume(_volume(img, t, c), m, out_shape, order)
    return out


@op("registration.descriptor_registration", "Descriptor-based Registration (2D/3D)", CAT, kind="image", preview=False, scope_param=False, roi_aware=False, supports_rgb=False,
    params=[Param("target", "image", None, help="The open image to register the current image onto (2-D planes, or 3-D volumes when both are Z stacks)")] + DETECTION_PARAMS
           + [Param("model", "choice", "Rigid body", choices=DESCRIPTOR_MODELS, label="Transformation model")] + MATCH_PARAMS
           + [Param("channel", "int", 0, min=0, help="Channel the detections are taken from (0 = current); the transform is applied to all channels and frames"),
              Param("also_apply_to", "image", None, label="Re-apply to another image", help="Optional: another open image (e.g. a second channel or series) transformed with the same model → extra new image"),
              Param("interpolation", "choice", "Linear", choices=INTERP), Param("add_rois", "bool", True, label="Point ROIs 'Matches'"),
              Param("create_new", "bool", True, label="Create new image")],
    description="Fiji Descriptor-based registration (Preibisch): DoG detections (beads, nuclei …) of the current image and a target image are described by the geometry of their nearest neighbours (translation / rigid / similarity invariant descriptors with redundancy), matched by descriptor similarity (significance ratio) and filtered with RANSAC on the model. The transform is applied to every channel and frame (3-D when both images are Z stacks, fitted in calibrated space); inliers → 'Matches' point ROIs, a 'Correspondences' table and the transform in the Results table.",
    keywords=["descriptor-based registration", "beads", "dog", "geometric hashing", "preibisch", "register", "3d registration", "rigid", "similarity", "affine"])
def descriptor_registration(ctx, target=None, objects="Bright", size=5.0, threshold=0.05, max_detections=500, model="Rigid body", n_neighbors=3, redundancy=1, significance=3.0,
                            ransac_threshold=5.0, ransac_iterations=1000, min_inlier_ratio=0.1, min_inliers=6, channel=0, also_apply_to=None, interpolation="Linear", add_rois=True, create_new=True):
    img = ctx.img
    if not target:
        raise ValueError("Choose a target image")
    tgt = ctx.session.get_image(target)
    if tgt.id == img.id:
        raise ValueError("Target must be a different image")
    if tgt.rgb:
        raise ValueError("The target must not be an RGB image")
    c = _channel_index(img, channel)
    ct = min(c, tgt.nc - 1)
    order = ORDER_MAP[interpolation]
    det = dict(objects=objects, size=float(size), threshold=float(threshold), max_detections=int(max_detections))
    mp = _mp(n_neighbors, redundancy, significance, ransac_threshold, ransac_iterations, min_inlier_ratio, min_inliers)
    three_d = img.nz > 1 and tgt.nz > 1
    ctx.progress(0.1, "Detecting objects")
    if three_d:
        vox_a, vox_b = _voxel_size(img), _voxel_size(tgt)
        m, s_in, d_in, info = _descriptor_pair_3d(_volume(img, img.display.t, c), vox_a, _volume(tgt, min(tgt.display.t, tgt.nt - 1), ct), vox_b, model, det, mp)
    else:
        m, s_in, d_in, info = _descriptor_pair_2d(img.data[img.display.t, img.display.z, c], tgt.data[min(tgt.display.t, tgt.nt - 1), min(tgt.display.z, tgt.nz - 1), ct], model, det, mp)
    if m is None:
        raise ValueError(f"Descriptor matching failed ({info['detections'][0]} / {info['detections'][1]} detections, {info['candidates']} candidates, {info['inliers']} inliers)"
                         + (f": {info['error']}" if info.get("error") else " – adjust the object size / threshold, lower the significance or raise the allowed error"))
    ctx.check_cancel()
    ctx.progress(0.6, "Transforming")
    resid = np.linalg.norm(apply_nd(m, s_in) - d_in, axis=1)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    ndim = 3 if three_d else 2
    table = _correspondence_table(s_in, d_in, resid, ndim)
    if three_d:
        out_shape3 = (tgt.nz, tgt.height, tgt.width)
        out = _warp_volumes_3d(img, m, out_shape3, order)
        info_t = _decompose3(m, tgt)
        ctx.session.settings[REGISTRATION3D_KEY] = {"matrix": m.tolist(), "source": img.id, "target": tgt.id, "mode": f"Descriptor-based ({model})"}
    else:
        out = _warp_planes_2d(img, m, (tgt.height, tgt.width), order)
        info_t = transform_params(m)
        ctx.session.settings[CORRESPONDENCES_KEY] = {"source": img.id, "target": tgt.id, "src": (s_in + 0.5).tolist(), "dst": (d_in + 0.5).tolist(), "matrix": m.tolist(), "model": model, "method": "Descriptor-based"}
    row = {"Label": f"{img.name} → {tgt.name}", "Model": model, "N detections": info["detections"][0], "N detections 2": info["detections"][1], "N candidates": info["candidates"],
           "N inliers": info["inliers"], **info_t, "RMS residual": rms}
    rois = (_match_rois(img, s_in) + _match_rois(tgt, d_in)) if add_rois else []
    new_images = []
    if also_apply_to:
        other = ctx.session.get_image(also_apply_to)
        if three_d and other.nz < 2:
            raise ValueError(f"{other.name} is not a Z stack – the 3-D model cannot be re-applied to it")
        o_out = _warp_volumes_3d(other, m, out_shape3, order) if three_d else _warp_planes_2d(other, m, (tgt.height, tgt.width), order)
        res_o = other.copy(name=f"{other.name} on {tgt.name}", data=cast_to(o_out, other.data.dtype))
        res_o.calibration = Calibration(**tgt.calibration.to_dict())
        res_o.overlay = []
        res_o.sync_channels()
        new_images.append(res_o)
    msg = (f"Descriptor-based registration of {img.name} onto {tgt.name} ({model}, {'3-D' if three_d else '2-D'}): {info['inliers']} inliers of {info['candidates']} candidates "
           f"({info['detections'][0]} / {info['detections'][1]} detections); dx {info_t['dx']:+.2f}, dy {info_t['dy']:+.2f}" + (f", dz {info_t['dz']:+.2f}" if three_d else "")
           + f", angle {info_t['Angle']:+.2f}°, scale {info_t['Scale']:.3f}, RMS {rms:.2f}")
    data = cast_to(out, img.data.dtype)
    if create_new:
        res = img.copy(name=f"{img.name} on {tgt.name}", data=data)
        res.calibration = Calibration(**tgt.calibration.to_dict())
        res.overlay = []
        res.sync_channels()
        return OpResult(new_images=new_images + [res], results=[row], table=table, rois_add=rois, message=msg, log=msg)
    return OpResult(data=data, rgb=img.rgb, new_images=new_images, select_image=img.id if new_images else None, results=[row], table=table, rois_add=rois,
                    calibration=tgt.calibration.to_dict(), clear_roi=True, message=msg, log=msg)


@op("registration.descriptor_series", "Descriptor-based Series Registration (2D/3D + t)", CAT, kind="image", preview=False, scope_param=False, roi_aware=False, supports_rgb=False,
    params=DETECTION_PARAMS + [Param("model", "choice", "Rigid body", choices=DESCRIPTOR_MODELS, label="Transformation model"),
                               Param("reference", "choice", "Previous frame", choices=["Previous frame", "First frame"])] + MATCH_PARAMS
           + [Param("channel", "int", 0, min=0, help="Channel the detections are taken from (0 = current); applied to all channels"),
              Param("interpolation", "choice", "Linear", choices=INTERP), Param("create_new", "bool", False, label="Create new image")],
    description="Fiji Descriptor-based series registration: every frame of a time series is registered onto the previous (chained) or the first frame from DoG detections matched by local geometric descriptors + RANSAC (3-D on Z stacks, in calibrated space); the per-frame transforms (with detection / candidate / inlier counts) go to the 'Alignment' table; frames without a consistent match keep the previous transform.",
    keywords=["descriptor-based series registration", "time lapse", "beads", "register frames", "drift", "3d + t"])
def descriptor_series(ctx, objects="Bright", size=5.0, threshold=0.05, max_detections=500, model="Rigid body", reference="Previous frame", n_neighbors=3, redundancy=1, significance=3.0,
                      ransac_threshold=5.0, ransac_iterations=1000, min_inlier_ratio=0.1, min_inliers=6, channel=0, interpolation="Linear", create_new=False):
    img = ctx.img
    if img.nt < 2:
        raise ValueError("Needs a time series (T > 1)")
    c = _channel_index(img, channel)
    order = ORDER_MAP[interpolation]
    det = dict(objects=objects, size=float(size), threshold=float(threshold), max_detections=int(max_detections))
    mp = _mp(n_neighbors, redundancy, significance, ransac_threshold, ransac_iterations, min_inlier_ratio, min_inliers)
    three_d = img.nz > 1
    vox = _voxel_size(img)
    n = img.nt
    D = 3 if three_d else 2

    def points(t: int) -> np.ndarray:
        if three_d:
            return _detect_3d(_volume(img, t, c), vox, det["objects"], det["size"], det["threshold"], det["max_detections"])
        return _detect_2d(img.data[t, img.display.z, c], det["objects"], det["size"], det["threshold"], det["max_detections"])

    mats = [np.eye(D + 1) for _ in range(n)]
    infos: list[dict[str, Any]] = [{"detections": (0, 0), "candidates": 0, "inliers": 0} for _ in range(n)]
    pts_cache: dict[int, np.ndarray] = {}

    def pts(t: int) -> np.ndarray:
        if t not in pts_cache:
            pts_cache[t] = points(t)
        return pts_cache[t]

    mp3 = {**mp, "threshold": float(mp["threshold"]) * (vox[0] if three_d else 1.0)}
    failed = 0
    for t in range(1, n):
        ctx.check_cancel()
        ctx.progress(0.6 * t / n, f"Matching frame {t + 1}/{n}")
        r = t - 1 if reference == "Previous frame" else 0
        a, b = pts(t), pts(r)
        try:
            M, _, _, info = descriptor_match(a, b, model, **mp3)
        except ValueError as e:
            M, info = None, {"detections": (int(len(a)), int(len(b))), "candidates": 0, "inliers": 0, "error": str(e)}
        infos[t] = info
        if M is None:
            failed += 1
            mats[t] = mats[t - 1].copy() if reference == "Previous frame" else np.eye(D + 1)
            continue
        if three_d:
            M = _matrix_to_voxels(M, vox, vox)
        mats[t] = (mats[r] @ M) if reference == "Previous frame" else M
        if reference == "Previous frame":
            pts_cache.pop(t - 1, None)
    out = np.empty(img.shape, np.float32)
    for t in range(n):
        ctx.check_cancel()
        ctx.progress(0.6 + 0.4 * t / n, f"Transforming frame {t + 1}/{n}")
        m = mats[t]
        if np.allclose(m, np.eye(D + 1)):
            out[t] = img.data[t]
            continue
        for cc in range(img.nc):
            if three_d:
                out[t, :, cc] = _warp_volume(_volume(img, t, cc), m, (img.nz, img.height, img.width), order)
            else:
                for z in range(img.nz):
                    out[t, z, cc] = _warp(img.data[t, z, cc], m, order)
    rows = []
    for t in range(n):
        dec = _decompose3(mats[t], img) if three_d else _decompose(mats[t])
        rows.append({"Label": f"{img.name}:t{t + 1}", "Frame": t + 1, **dec, "Detections": infos[t]["detections"][0], "Candidates": infos[t]["candidates"], "Inliers": infos[t]["inliers"]})
    dmax = max(math.hypot(r_["dx"], r_["dy"]) for r_ in rows)
    msg = f"Descriptor-based series registration over {n} frames ({model}, {'3-D' if three_d else '2-D'}, ref: {reference.lower()}); max shift {dmax:.2f} px"
    if failed:
        msg += f"; {failed} frame{'s' if failed != 1 else ''} without a consistent match (previous transform kept)"
    return _finish(ctx, img, out, create_new, f"Registered {img.name}", rows, msg)


# ================================================================================ manual landmark pairs
def _landmark_rois(ctx, img: LumenImage, source_roi: str, target_roi: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rois = ctx.session.roi_manager.rois

    def by_name(name: str) -> dict[str, Any]:
        for r in rois:
            if r.get("name") == name or r.get("id") == name:
                if r.get("type") != "point":
                    raise ValueError(f"ROI {name!r} is not a point selection")
                return r
        raise ValueError(f"No ROI named {name!r} in the ROI Manager")

    if source_roi and target_roi:
        return by_name(source_roi), by_name(target_roi)
    pts = [r for r in rois if r.get("type") == "point" and len(r.get("points", [])) >= 1]
    named = [r for r in pts if str(r.get("name", "")).lower().startswith("landmark")]
    pool = named if len(named) >= 2 else pts
    if source_roi:
        src = by_name(source_roi)
        rest = [r for r in pool if r is not src] or [r for r in pts if r is not src]
        if not rest:
            raise ValueError("Add a second point ROI with the target landmarks to the ROI Manager")
        return src, rest[0]
    if target_roi:
        dst = by_name(target_roi)
        rest = [r for r in pool if r is not dst and r.get("image_id") in (None, img.id)] or [r for r in pts if r is not dst]
        if not rest:
            raise ValueError("Add a point ROI with the source landmarks (on the current image) to the ROI Manager")
        return rest[0], dst
    if len(pool) < 2:
        raise ValueError("Needs two multi-point ROIs in the ROI Manager (e.g. 'landmarks:A' on the current image and 'landmarks:B' on the target)")
    on_img = [r for r in pool if r.get("image_id") == img.id]
    others = [r for r in pool if r.get("image_id") != img.id]
    if on_img and others:
        return on_img[0], others[0]
    return pool[0], pool[1]


@op("registration.landmark_pairs", "Register with Manual Landmarks", CAT, kind="image", preview=False, scope_param=False, roi_aware=False,
    params=[Param("model", "choice", "Rigid body", choices=LANDMARK_MODELS, label="Transformation model"),
            Param("source_roi", "string", "", help="Name of the multi-point ROI with the landmarks on the current image (empty = the first point ROI named 'landmarks…' on this image)"),
            Param("target_roi", "string", "", help="Name of the multi-point ROI with the matching landmarks, in the same order (empty = the other 'landmarks…' point ROI)"),
            Param("target", "image", None, help="Optional: the image the target landmarks belong to (the result takes its size and calibration)"),
            Param("interpolation", "choice", "Linear", choices=INTERP),
            Param("apply", "bool", True, help="Transform the image (otherwise only compute and store the model)"),
            Param("create_new", "bool", True, label="Create new image")],
    description="Manual landmark selection (Fiji Descriptor-based registration ▸ manual): two multi-point ROIs from the ROI Manager – e.g. 'landmarks:A' clicked on the current image and 'landmarks:B' on the target, point i on A matching point i on B – give a translation / rigid / similarity / affine / projective model (least squares over all pairs) that is applied to every plane of the current image; the pairs go to the 'Correspondences' table and the model to the Results table and to 'Landmark Transform'.",
    keywords=["manual landmark selection", "landmarks", "point pairs", "register by points", "fiducials"])
def landmark_pairs(ctx, model="Rigid body", source_roi="", target_roi="", target=None, interpolation="Linear", apply=True, create_new=True):
    from .transform_extra import _matrix_result
    img = ctx.img
    ra, rb = _landmark_rois(ctx, img, source_roi, target_roi)
    src = np.asarray(ra.get("points", []), float).reshape(-1, 2)
    dst = np.asarray(rb.get("points", []), float).reshape(-1, 2)
    if len(src) != len(dst):
        raise ValueError(f"{ra.get('name')} has {len(src)} points but {rb.get('name')} has {len(dst)} – the landmarks must correspond one to one")
    if len(src) == 0:
        raise ValueError("The landmark ROIs are empty")
    tgt = ctx.session.get_image(target) if target else (ctx.session.get_image(rb["image_id"]) if rb.get("image_id") and rb.get("image_id") != img.id and rb["image_id"] in ctx.session.images else None)
    m, _ = estimate_point_transform(src - 0.5, dst - 0.5, model, use_ransac=False)
    resid = np.hypot(*(_apply_matrix(m, src - 0.5) - (dst - 0.5)).T)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    params = transform_params(m)
    row = {"Label": f"{img.name} → {tgt.name}" if tgt is not None else img.name, "Model": model, "Source ROI": ra.get("name"), "Target ROI": rb.get("name"), "N landmarks": int(len(src)),
           **params, "RMS residual": rms}
    table = _correspondence_table(src - 0.5, dst - 0.5, resid, 2)
    ctx.session.settings[CORRESPONDENCES_KEY] = {"source": img.id, "target": tgt.id if tgt is not None else None, "src": src.tolist(), "dst": dst.tolist(), "matrix": m.tolist(), "model": model, "method": "Manual landmarks"}
    ctx.session.settings["landmark_transform"] = {"matrix": m.tolist(), "model": model}
    msg = f"{model} model from {len(src)} landmark pairs ({ra.get('name')} → {rb.get('name')}): dx {params['dx']:+.2f}, dy {params['dy']:+.2f}, angle {params['Angle']:+.2f}°, scale {params['Scale']:.3f}, RMS residual {rms:.2f} px"
    if not apply:
        return OpResult(results=[row], table=table, message=msg, log=msg)
    order = ORDER_MAP[interpolation]
    out_shape = (tgt.height, tgt.width) if tgt is not None else (img.height, img.width)
    res = _matrix_result(ctx, img, m, out_shape, order, create_new, f"{img.name} ({model.lower()}, landmarks)", [row], msg)
    res.table = table
    if tgt is not None and res.new_images:
        res.new_images[0].calibration = Calibration(**tgt.calibration.to_dict())
    elif tgt is not None:
        res.calibration = tgt.calibration.to_dict()
    return res
