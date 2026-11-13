"""Plugins ▸ Registration — StackReg-style alignment, chromatic-shift correction, drift
correction and pairwise / grid stitching.

All transforms are 3×3 matrices in (x, y) pixel coordinates mapping the *moving* image
onto the *reference* image. Translation is estimated with phase correlation
(sub-pixel via ``upsample_factor``); rigid-body and affine transforms with ORB feature
matching + RANSAC, falling back to translation when there are too few matches.
"""
from __future__ import annotations

import inspect
import itertools
import math
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation

from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op, cast_to

CAT = "Plugins/Registration"
MODES = ["Translation", "Rigid body", "Affine"]
#: Align Stack models: StackReg's set (Scaled rotation = similarity; Bilinear only with the intensity estimator)
ALIGN_MODES = ["Translation", "Rigid body", "Scaled rotation", "Affine", "Bilinear"]
ESTIMATORS = ["Phase correlation + features", "SIFT feature matching", "Intensity (pyramid optimisation)"]
REFS = ["Previous slice", "First slice", "Middle slice", "Current slice"]
INTERP = ["Linear", "Nearest", "Cubic"]
FUSION = ["Linear blend", "Max", "Average", "Overlay (channels)"]
ORDERS = ["Row-by-row", "Column-by-column", "Snake by rows", "Snake by columns"]
ORDER_MAP = {"Nearest": 0, "Linear": 1, "Cubic": 3}
IDENTITY = np.eye(3)


# ============================================================================ helpers
def _f32(plane: np.ndarray) -> np.ndarray:
    p = plane.astype(np.float32, copy=False)
    if not np.all(np.isfinite(p)):
        p = np.nan_to_num(p)
    return p


def _norm01(plane: np.ndarray) -> np.ndarray:
    p = _f32(plane).astype(np.float64)
    lo, hi = np.percentile(p, (0.5, 99.5))
    if hi <= lo:
        lo, hi = float(p.min()), float(p.max())
    if hi <= lo:
        return np.zeros_like(p)
    return np.clip((p - lo) / (hi - lo), 0, 1)


def _translation(dx: float, dy: float) -> np.ndarray:
    m = np.eye(3)
    m[0, 2], m[1, 2] = dx, dy
    return m


def _is_translation(m: np.ndarray) -> bool:
    return bool(np.allclose(m[:2, :2], np.eye(2), atol=1e-9) and np.allclose(m[2], [0, 0, 1]))


def _is_identity(m: Any) -> bool:
    return isinstance(m, np.ndarray) and bool(np.allclose(m, IDENTITY))


def _compose(a: Any, b: Any) -> Any:
    """a ∘ b for chained slice transforms (b maps slice k onto k−1, a maps k−1 onto the reference)."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a @ b
    if _is_identity(a):
        return b
    if _is_identity(b):
        return a
    raise ValueError("Bilinear transforms cannot be chained – they are estimated against the aligned previous slice")


def _phase_shift(ref: np.ndarray, mov: np.ndarray, upsample: int = 10) -> tuple[float, float]:
    """(dy, dx) that shifts ``mov`` onto ``ref``."""
    if ref.shape != mov.shape:
        h, w = max(ref.shape[0], mov.shape[0]), max(ref.shape[1], mov.shape[1])
        ref = np.pad(ref, ((0, h - ref.shape[0]), (0, w - ref.shape[1])))
        mov = np.pad(mov, ((0, h - mov.shape[0]), (0, w - mov.shape[1])))
    a, b = _f32(ref), _f32(mov)
    a = a - a.mean()
    b = b - b.mean()
    if not a.any() or not b.any():
        return 0.0, 0.0
    up = max(int(upsample), 1)
    # whitened ("phase") correlation is sharper on textured images, plain cross-correlation is
    # more robust on noisy / smooth ones: compute both and keep the one whose overlap agrees best
    cands = []
    for norm in ("phase", None):
        try:
            shift = phase_cross_correlation(a, b, upsample_factor=up, normalization=norm)[0]
        except Exception:
            continue
        if np.all(np.isfinite(shift)):
            cands.append((float(shift[0]), float(shift[1])))
    if not cands:
        return 0.0, 0.0
    if len(cands) == 2 and (abs(cands[0][0] - cands[1][0]) > 0.5 or abs(cands[0][1] - cands[1][1]) > 0.5):
        scores = [_overlap_ncc(a, b, int(round(dy)), int(round(dx))) for dy, dx in cands]
        return cands[int(np.argmax(scores))]
    return cands[0]


def _feature_matrix(ref: np.ndarray, mov: np.ndarray, mode: str) -> np.ndarray | None:
    """ORB + RANSAC estimate of the (moving → reference) matrix, or None when it fails."""
    from skimage.feature import ORB, match_descriptors
    from skimage.measure import ransac
    from skimage.transform import AffineTransform, EuclideanTransform
    a, b = _norm01(ref), _norm01(mov)
    n_kp = int(min(500, max(50, a.size // 400)))
    try:
        orb = ORB(n_keypoints=n_kp, fast_threshold=0.05)
        orb.detect_and_extract(a)
        ka, da = orb.keypoints, orb.descriptors
        orb.detect_and_extract(b)
        kb, db = orb.keypoints, orb.descriptors
    except (RuntimeError, ValueError, IndexError):
        return None
    if ka is None or kb is None or len(ka) < 8 or len(kb) < 8:
        return None
    matches = match_descriptors(da, db, cross_check=True, max_ratio=0.85)
    if len(matches) < 8:
        return None
    dst = ka[matches[:, 0]][:, ::-1]  # reference (x, y)
    src = kb[matches[:, 1]][:, ::-1]  # moving (x, y)
    rigid = mode == "Rigid body"
    if mode == "Scaled rotation":
        from skimage.transform import SimilarityTransform
        model_cls = SimilarityTransform
    else:
        model_cls = EuclideanTransform if rigid else AffineTransform
    try:
        model, inliers = ransac((src, dst), model_cls, min_samples=3 if mode in ("Rigid body", "Scaled rotation") else 4, residual_threshold=2.0, max_trials=1000, rng=0)
    except Exception:
        return None
    if model is None or inliers is None or int(inliers.sum()) < 6:
        return None
    m = np.asarray(model.params, dtype=np.float64)
    if not np.all(np.isfinite(m)):
        return None
    sx = math.hypot(m[0, 0], m[1, 0])
    sy = math.hypot(m[0, 1], m[1, 1])
    if not (0.5 < sx < 2.0 and 0.5 < sy < 2.0):
        return None
    # refine the residual translation with phase correlation on the warped image
    warped = _warp(mov, m, 1)
    dy, dx = _phase_shift(ref, warped, 10)
    if abs(dx) < 0.25 * ref.shape[1] and abs(dy) < 0.25 * ref.shape[0]:
        m = _translation(dx, dy) @ m
    return _refine_intensity(ref, mov, m, mode)


def _refine_intensity(ref: np.ndarray, mov: np.ndarray, m: np.ndarray, mode: str, iters: int = 6) -> np.ndarray:
    """Gauss-Newton refinement of a rigid (3 parameters), scaled-rotation (4) or affine (6 parameters) matrix on
    intensities (StackReg-style). Starts from the feature-based estimate; keeps the best."""
    r = ndi.gaussian_filter(_f32(ref), 1.0)
    mv = ndi.gaussian_filter(_f32(mov), 1.0)
    h, w = r.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    ym, xm = (yy - cy).astype(np.float32), (xx - cx).astype(np.float32)
    ones = np.ones_like(mv)
    rigid = mode == "Rigid body"
    scaled = mode == "Scaled rotation"
    best_m, best_err = m, np.inf
    cur = m
    for it in range(iters + 1):
        wv = _warp(mv, cur, 1)
        valid = _warp(ones, cur, 1) > 0.99
        valid[:2, :] = valid[-2:, :] = valid[:, :2] = valid[:, -2:] = False
        if valid.sum() < 64:
            break
        a = r[valid]
        b = wv[valid]
        sa, sb = a.std(), b.std()
        if sa <= 0 or sb <= 0:
            break
        a = (a - a.mean()) / sa
        bn = (wv - b.mean()) / sb
        resid = a - bn[valid]
        err = float(np.mean(resid ** 2))
        if err < best_err - 1e-7:
            best_m, best_err = cur, err
        elif err > best_err:
            break
        if it == iters:
            break
        gy, gx = np.gradient(bn)
        gx, gy = gx[valid], gy[valid]
        x, y = xm[valid], ym[valid]
        if rigid:
            cols = [-gx, -gy, -(gy * x - gx * y)]
        elif scaled:
            cols = [-gx, -gy, -(gy * x - gx * y), -(gx * x + gy * y)]
        else:
            cols = [-gx, -gy, -gx * x, -gx * y, -gy * x, -gy * y]
        A = np.stack(cols, axis=1).astype(np.float64)
        try:
            p = np.linalg.lstsq(A, resid.astype(np.float64), rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(p)):
            break
        if rigid or scaled:
            dx, dy, th = p[:3]
            sc = math.exp(float(p[3])) if scaled else 1.0
            if abs(th) > 0.2 or abs(dx) > 5 or abs(dy) > 5 or abs(sc - 1) > 0.2:
                break
            c_, s_ = math.cos(th), math.sin(th)
            lin = sc * np.array([[c_, -s_], [s_, c_]])
            off = np.array([dx, dy])
        else:
            dx, dy, a11, a12, a21, a22 = p
            if abs(a11) > 0.2 or abs(a22) > 0.2 or abs(a12) > 0.2 or abs(a21) > 0.2 or abs(dx) > 5 or abs(dy) > 5:
                break
            lin = np.array([[1 + a11, a12], [a21, 1 + a22]])
            off = np.array([dx, dy])
        d = np.eye(3)
        d[:2, :2] = lin
        d[:2, 2] = off + np.array([cx, cy]) - lin @ np.array([cx, cy])
        cur = d @ cur
        if abs(p[0]) < 1e-3 and abs(p[1]) < 1e-3 and np.all(np.abs(p[2:]) < 1e-5):
            best_m = cur
            break
    return best_m


def _estimate(ref: np.ndarray, mov: np.ndarray, mode: str, upsample: int) -> tuple[np.ndarray, bool]:
    """(matrix, used_features). Bilinear has no feature model: it is estimated as an affine here."""
    if mode in ("Rigid body", "Scaled rotation", "Affine", "Bilinear"):
        m = _feature_matrix(ref, mov, "Affine" if mode == "Bilinear" else mode)
        if m is not None:
            return m, True
    dy, dx = _phase_shift(ref, mov, upsample)
    return _translation(dx, dy), False


def _translation_any_offset(ref: np.ndarray, mov: np.ndarray, upsample: int) -> tuple[float, float]:
    """(dy, dx) of ``mov``'s origin in ``ref``'s frame, valid for any offset (beyond half the
    canvas, planes of different size).

    Circular phase correlation only knows the shift modulo the canvas, so the integer offset
    comes from the zero-padded linear correlator (whose peaks are verified on the real overlap)
    and the sub-pixel residual from phase correlation on the overlapping windows only.
    """
    ref, mov = _f32(ref), _f32(mov)
    dy0, dx0, ncc = _find_offset(ref, mov)
    if ncc < 0:  # no verified overlap: keep the old circular estimate (pads internally)
        return _phase_shift(ref, mov, upsample)
    ha, wa = ref.shape
    hb, wb = mov.shape
    y0, y1 = max(0, dy0), min(ha, dy0 + hb)
    x0, x1 = max(0, dx0), min(wa, dx0 + wb)
    rdy, rdx = _phase_shift(ref[y0:y1, x0:x1], mov[y0 - dy0:y1 - dy0, x0 - dx0:x1 - dx0], upsample)
    if abs(rdy) > 1.5 or abs(rdx) > 1.5:  # the residual must be sub-pixel; anything else is a spurious peak
        rdy = rdx = 0.0
    return float(dy0 + rdy), float(dx0 + rdx)


def _warp(plane: np.ndarray, m: Any, order: int) -> np.ndarray:
    """Apply the (moving → reference) matrix (or a transform object with ``.warp``, e.g. the bilinear model of
    registration_features) to a 2-D plane; output has the same shape."""
    if not isinstance(m, np.ndarray):
        return m.warp(plane, order)
    p = _f32(plane)
    if _is_translation(m):
        if abs(m[0, 2]) < 1e-9 and abs(m[1, 2]) < 1e-9:
            return p.copy()
        return ndi.shift(p, (m[1, 2], m[0, 2]), order=order, mode="constant", cval=0.0).astype(np.float32)
    from skimage.transform import warp
    return warp(p, np.linalg.inv(m), order=order, mode="constant", cval=0.0, preserve_range=True).astype(np.float32)


def _decompose(m: Any) -> dict[str, float]:
    if not isinstance(m, np.ndarray):  # a transform object (bilinear): its affine approximation
        m = np.asarray(m.matrix, float)
    return {"dx": float(m[0, 2]), "dy": float(m[1, 2]), "Angle": float(math.degrees(math.atan2(m[1, 0], m[0, 0]))),
            "Scale": float(math.sqrt(abs(np.linalg.det(m[:2, :2]))))}


def _chain_seed(prev_tf: Any, dx: float, dy: float) -> Any:
    """``prev_tf`` (slice k−1 → reference) pre-composed with the raw inter-slice translation (slice k → k−1): the start
    value of a chained estimate.  For a bilinear model (an inverse map) a moving-side translation only shifts the
    constant terms."""
    if isinstance(prev_tf, np.ndarray):
        return prev_tf @ _translation(dx, dy)
    coef = np.array(prev_tf.coef, float)
    coef[0, 0] -= dx
    coef[1, 0] -= dy
    return type(prev_tf)(coef, prev_tf.shape)


def _rebase_transform(tf: Any, ref_origin: tuple[float, float], mov_origin: tuple[float, float], old_shape: tuple[int, int],
                      new_shape: tuple[int, int]) -> Any:
    """The same moving → reference transform expressed on windows of both planes: ``ref_origin`` / ``mov_origin`` are
    the (y, x) positions of the new reference / moving frame's origin inside the old frames (negative to go back from
    a window to the full frame), ``new_shape`` the window size.  Matrices just change origin; a bilinear model's
    coefficients are re-centred on the window centre (a change of variables of its inverse map)."""
    y0r, x0r = ref_origin
    y0m, x0m = mov_origin
    if isinstance(tf, np.ndarray):
        return _translation(-x0r, -y0r) @ tf @ _translation(x0m, y0m)
    cy, cx = (old_shape[0] - 1) / 2.0, (old_shape[1] - 1) / 2.0
    ny, nx = (new_shape[0] - 1) / 2.0, (new_shape[1] - 1) / 2.0
    ox, oy = x0r + nx - cx, y0r + ny - cy  # old centred coordinate of the new centre
    c = np.asarray(tf.coef, float)
    out = np.empty_like(c)
    for row, (cen_old, cen_new, org) in enumerate(((cx, nx, x0m), (cy, ny, y0m))):
        c0, c1, c2, c3 = c[row]
        out[row] = (cen_old - org - cen_new + c0 + c1 * ox + c2 * oy + c3 * ox * oy, c1 + c3 * oy, c2 + c3 * ox, c3)
    return type(tf)(out, (int(new_shape[0]), int(new_shape[1])))


def _moving_offset(tf: Any, y: float, x: float) -> tuple[float, float]:
    """(dy, dx) from the reference-frame point (y, x) to the moving-frame point it samples (the affine approximation
    for a bilinear model)."""
    m = tf if isinstance(tf, np.ndarray) else np.asarray(tf.matrix, float)
    try:
        p = np.linalg.inv(m) @ np.array([x, y, 1.0])
    except np.linalg.LinAlgError:
        return 0.0, 0.0
    if abs(p[2]) < 1e-12 or not np.all(np.isfinite(p)):
        return 0.0, 0.0
    return float(p[1] / p[2] - y), float(p[0] / p[2] - x)


def _chain_windows(valid: np.ndarray, tf: Any, min_size: int = 32) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    """Equal-sized windows (y0, y1, x0, x1) of the reference plane – the bounding box of its ``valid`` region – and
    of the moving plane (the same box shifted by the transform's offset at its centre, clipped), or None when the
    valid region is too small for an intensity fit."""
    rows, cols = np.flatnonzero(valid.any(axis=1)), np.flatnonzero(valid.any(axis=0))
    if rows.size < min_size or cols.size < min_size:
        return None
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    H, W = valid.shape
    ty, tx = _moving_offset(tf, (y0 + y1 - 1) / 2.0, (x0 + x1 - 1) / 2.0)
    ty, tx = int(round(ty)), int(round(tx))
    my0, my1, mx0, mx1 = y0 + ty, y1 + ty, x0 + tx, x1 + tx
    if my0 < 0:
        y0, my0 = y0 - my0, 0
    if mx0 < 0:
        x0, mx0 = x0 - mx0, 0
    if my1 > H:
        y1, my1 = y1 - (my1 - H), H
    if mx1 > W:
        x1, mx1 = x1 - (mx1 - W), W
    if y1 - y0 < min_size or x1 - x0 < min_size:
        return None
    return (y0, y1, x0, x1), (my0, my1, mx0, mx1)


def _estimate_series(get_plane: Callable[[int], np.ndarray], n: int, mode: str, reference: str, upsample: int, ctx=None, estimate: Callable[..., Any] | None = None,
                     current: int = 0, chain_on_aligned: bool = False, order: int = 1) -> tuple[list[Any], list[dict[str, Any]]]:
    """Transforms mapping every slice onto the reference frame. Returns (transforms, per-slice info dicts).

    ``estimate(ref, mov, k_ref, k_mov) -> (transform, info)`` is the pairwise estimator (``k_*`` = slice indices or None
    for a plane that is not a raw slice); the default is phase correlation + features (``info["features_used"]``).
    References: 'Previous slice' chains from the first slice, 'Current slice' chains outward from slice ``current``
    (StackReg), 'First' / 'Middle slice' register every slice directly. With ``chain_on_aligned`` a chained slice is
    registered onto the *aligned* previous slice (needed for models that cannot be composed, e.g. bilinear): the
    estimate then has to recover the whole cumulative transform, so it is seeded with the previous transform
    pre-composed with the raw inter-slice phase-correlation step (when ``estimate`` accepts ``init``), and both planes
    are cropped to the valid region of the warped reference so that the zero border of the accumulated shift does not
    enter the intensity fit (it would lock the fit onto a wrong, distorted solution once the shift grows).
    ``ctx`` (an OpContext) is optional: with it, the estimation reports progress (first half) and honours cancellation."""
    if estimate is None:
        def estimate(ref: np.ndarray, mov: np.ndarray, k_ref: int | None = None, k_mov: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
            m, feat = _estimate(ref, mov, mode, upsample)
            return m, {"features_used": bool(feat)}

    mats: list[Any] = [IDENTITY.copy() for _ in range(n)]
    infos: list[dict[str, Any]] = [{} for _ in range(n)]
    done = 0
    try:
        accepts_init = "init" in inspect.signature(estimate).parameters
    except (TypeError, ValueError):
        accepts_init = False

    def tick(k: int) -> None:
        nonlocal done
        done += 1
        if ctx is not None:
            ctx.check_cancel()
            ctx.progress(0.5 * done / n, f"Estimating slice {k + 1}/{n}")

    if reference in ("Previous slice", "Current slice"):
        r = 0 if reference == "Previous slice" else max(0, min(int(current), n - 1))
        for direction in (1, -1):
            k = r + direction
            while 0 <= k < n:
                tick(k)
                prev_k = k - direction
                cur = get_plane(k)
                if chain_on_aligned and not _is_identity(mats[prev_k]):
                    prev, pm = get_plane(prev_k), mats[prev_k]
                    ref = _warp(prev, pm, order)
                    if accepts_init:
                        sdy, sdx = _phase_shift(prev, cur, upsample)  # raw step k → k−1: large capture range
                        init = _chain_seed(pm, sdx, sdy)
                        valid = _warp(np.ones(prev.shape, np.float32), pm, 1) > 0.99
                        win = _chain_windows(valid, init)
                        if win is not None:
                            (y0, y1, x0, x1), (my0, my1, mx0, mx1) = win
                            shape_w = (y1 - y0, x1 - x0)
                            init_w = _rebase_transform(init, (y0, x0), (my0, mx0), prev.shape, shape_w)
                            m, info = estimate(ref[y0:y1, x0:x1], cur[my0:my1, mx0:mx1], None, k, init=init_w)
                            m = _rebase_transform(m, (-y0, -x0), (-my0, -mx0), shape_w, prev.shape)
                        else:
                            m, info = estimate(ref, cur, None, k, init=init)
                    else:
                        m, info = estimate(ref, cur, None, k)
                    mats[k] = m
                elif chain_on_aligned and accepts_init:
                    # first step of a chained model: a large inter-slice motion exceeds the intensity estimator's capture
                    # range from identity, so also start from the raw phase-correlation step and keep the better fit
                    prev = get_plane(prev_k)
                    m, info = estimate(prev, cur, prev_k, k)
                    sdy, sdx = _phase_shift(prev, cur, upsample)
                    if math.hypot(sdx, sdy) > 1.5:
                        m2, info2 = estimate(prev, cur, prev_k, k, init=_translation(sdx, sdy))
                        e1, e2 = float(info.get("error", np.inf)), float(info2.get("error", np.inf))
                        if np.isfinite(e2) and not (np.isfinite(e1) and e1 <= e2):
                            m, info = m2, info2
                    mats[k] = _compose(mats[prev_k], m)
                else:
                    m, info = estimate(get_plane(prev_k), cur, prev_k, k)
                    mats[k] = _compose(mats[prev_k], m)
                infos[k] = info
                k += direction
    else:
        r = 0 if reference == "First slice" else n // 2
        ref = get_plane(r)
        for k in range(n):
            if k == r:
                continue
            tick(k)
            m, info = estimate(ref, get_plane(k), r, k)
            mats[k] = m
            infos[k] = info
    return mats, infos


def _channel_index(img: LumenImage, channel: int) -> int:
    if channel <= 0:
        return img.display.c
    if channel > img.nc:
        raise ValueError(f"Channel {channel} does not exist (image has {img.nc})")
    return channel - 1


def _alignment_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The per-slice / per-channel / per-frame shifts as the 'Alignment' table (not the shared Results
    table, where one row per slice with dx/dy/Angle columns only leaves a wide, mostly empty table)."""
    cols = list(rows[0].keys()) if rows else ["Label"]
    return {"name": "Alignment", "columns": cols, "rows": [[r.get(c) for c in cols] for r in rows]}


def _finish(ctx, img: LumenImage, out: np.ndarray, create_new: bool, name: str, rows: list[dict[str, Any]], msg: str) -> OpResult:
    data = cast_to(out, img.data.dtype)
    table = _alignment_table(rows)
    if create_new:
        res = img.copy(name=name, data=data)
        res.sync_channels()
        return OpResult(new_images=[res], table=table, message=msg, log=msg)
    return OpResult(data=data, rgb=img.rgb, table=table, message=msg, log=msg)


# ------------------------------------------------------------------ stitching helpers
def _find_offset(a: np.ndarray, b: np.ndarray, n_peaks: int = 5, expected: tuple[float, float] | None = None,
                 max_dev: tuple[float, float] | None = None, min_overlap: int = 16) -> tuple[int, int, float]:
    """Position (dy, dx) of ``b``'s origin in ``a``'s pixel frame, with the normalised cross-correlation of the overlap.

    Zero-padded (linear, not circular) correlation; the best peaks are verified on the real overlap (Fiji 'check
    peaks' style).  Candidates come from two correlators, interleaved: the whitened (phase) correlation, sharp on
    textured images, and the plain cross-correlation per overlap pixel, which whitening loses on smooth / blurred
    images (it amplifies their empty high frequencies and the true offset drops out of the top peaks).  The plain
    list is restricted to credible overlaps (>= 10 % of the smaller image) because its per-pixel normalisation
    favours slivers whose chance correlation is high.  Among the verified candidates the winner is the most
    *significant* correlation (Fisher z, atanh(r)·sqrt(n − 3)) rather than the largest raw NCC, so a small overlap's
    chance NCC cannot beat a weaker but well-supported one; the returned value is the plain NCC.  The winner is then
    refined by exhaustive NCC within ±2 px (on smooth images the plain peak sits one pixel off).
    ``expected``/``max_dev`` restrict the candidates to a neighbourhood of a known approximate offset; NCC < 0 (−1)
    means no verified overlap.
    """
    a = _f32(a)
    b = _f32(b)
    ha, wa = a.shape
    hb, wb = b.shape
    H, W = ha + hb, wa + wb
    fa = np.fft.rfft2(a - a.mean(), s=(H, W))
    fb = np.fft.rfft2(b - b.mean(), s=(H, W))
    cross = fa * np.conj(fb)
    del fa, fb
    mag = np.abs(cross)
    corr_phase = np.fft.irfft2(cross / np.where(mag > 1e-12, mag, 1.0), s=(H, W)).astype(np.float32)
    del mag
    ones_a = np.fft.rfft2(np.ones(a.shape, np.float32), s=(H, W))
    ones_b = np.fft.rfft2(np.ones(b.shape, np.float32), s=(H, W))
    overlap = np.fft.irfft2(ones_a * np.conj(ones_b), s=(H, W))
    del ones_a, ones_b
    lim = max(float(min_overlap), 0.10 * min(a.size, b.size))
    corr_plain = (np.fft.irfft2(cross, s=(H, W)) / np.maximum(overlap, lim)).astype(np.float32)
    del cross
    corr_plain[overlap < lim - 0.5] = -np.inf
    del overlap
    k = min(max(n_peaks, 1) * 4, corr_phase.size)
    lists: list[list[int]] = []
    for corr in (corr_phase, corr_plain):
        peaks = corr == ndi.maximum_filter(corr, size=3, mode="wrap")
        flat = np.where(peaks.ravel(), corr.ravel(), -np.inf)
        cand = np.argpartition(-flat, k - 1)[:k]
        cand = cand[np.argsort(-flat[cand])]
        lists.append([int(i) for i in cand if np.isfinite(flat[i])])
    del corr_phase, corr_plain
    seen: set[int] = set()
    cands: list[int] = []
    for pair in itertools.zip_longest(*lists):  # best whitened, best plain, second whitened, …
        for i in pair:
            if i is not None and i not in seen:
                seen.add(i)
                cands.append(i)

    def allowed(dy: int, dx: int) -> bool:
        return expected is None or max_dev is None or (abs(dy - expected[0]) <= max_dev[0] and abs(dx - expected[1]) <= max_dev[1])

    def score(dy: int, dx: int) -> tuple[float, float]:
        ncc = _overlap_ncc(a, b, dy, dx, min_overlap)
        if ncc <= -1.0:
            return ncc, -np.inf
        n = max(0, min(ha, dy + hb) - max(0, dy)) * max(0, min(wa, dx + wb) - max(0, dx))
        return ncc, math.atanh(max(-0.999999, min(0.999999, ncc))) * math.sqrt(max(n - 3, 1))

    best, best_score, tested = (0, 0, -2.0), -np.inf, 0
    for idx in cands:
        py, px = divmod(idx, W)
        dy = py if py < ha else py - H
        dx = px if px < wa else px - W
        if not allowed(dy, dx):
            continue
        ncc, sc = score(dy, dx)
        tested += 1
        if sc > best_score:
            best, best_score = (int(dy), int(dx), float(ncc)), sc
        if tested >= 2 * max(n_peaks, 1):
            break
    if best[2] > -1.0:
        dy0, dx0 = best[0], best[1]
        for ddy in range(-2, 3):
            for ddx in range(-2, 3):
                if (ddy or ddx) and allowed(dy0 + ddy, dx0 + ddx):
                    ncc, sc = score(dy0 + ddy, dx0 + ddx)
                    if sc > best_score:
                        best, best_score = (dy0 + ddy, dx0 + ddx, float(ncc)), sc
    return best


def _overlap_ncc(a: np.ndarray, b: np.ndarray, dy: int, dx: int, min_overlap: int = 16) -> float:
    ha, wa = a.shape
    hb, wb = b.shape
    y0, y1 = max(0, dy), min(ha, dy + hb)
    x0, x1 = max(0, dx), min(wa, dx + wb)
    if y1 - y0 < 2 or x1 - x0 < 2 or (y1 - y0) * (x1 - x0) < min_overlap:
        return -1.0
    sa = a[y0:y1, x0:x1].astype(np.float64).ravel()
    sb = b[y0 - dy:y1 - dy, x0 - dx:x1 - dx].astype(np.float64).ravel()
    sa = sa - sa.mean()
    sb = sb - sb.mean()
    den = math.sqrt(float((sa * sa).sum()) * float((sb * sb).sum()))
    return float((sa * sb).sum() / den) if den > 0 else -1.0


def _blend_weights(h: int, w: int) -> np.ndarray:
    y = np.minimum(np.arange(h) + 1, h - np.arange(h)).astype(np.float32)
    x = np.minimum(np.arange(w) + 1, w - np.arange(w)).astype(np.float32)
    return np.minimum.outer(y, x)


def _fuse(tiles: list[tuple[np.ndarray, int, int]], fusion: str) -> np.ndarray:
    """Fuse 5-D tiles (data, y, x) placed at integer positions into one 5-D float32 canvas."""
    if not tiles:
        raise ValueError("No tiles to fuse")
    oy = min(y for _, y, _ in tiles)
    ox = min(x for _, _, x in tiles)
    H = max(y + d.shape[3] for d, y, _ in tiles) - oy
    W = max(x + d.shape[4] for d, _, x in tiles) - ox
    t0, z0 = tiles[0][0].shape[0], tiles[0][0].shape[1]
    for d, _, _ in tiles:
        if d.shape[0] != t0 or d.shape[1] != z0:
            raise ValueError("Tiles must have the same number of frames and slices")
    if fusion == "Overlay (channels)":
        nc = sum(d.shape[2] for d, _, _ in tiles)
        out = np.zeros((t0, z0, nc, H, W), np.float32)
        c = 0
        for d, y, x in tiles:
            yy, xx = y - oy, x - ox
            out[:, :, c:c + d.shape[2], yy:yy + d.shape[3], xx:xx + d.shape[4]] = d
            c += d.shape[2]
        return out
    nc = tiles[0][0].shape[2]
    for d, _, _ in tiles:
        if d.shape[2] != nc:
            raise ValueError("Tiles must have the same number of channels (or use 'Overlay (channels)')")
    if fusion == "Max":
        out = np.full((t0, z0, nc, H, W), -np.inf, np.float32)
        for d, y, x in tiles:
            yy, xx = y - oy, x - ox
            sl = (slice(None), slice(None), slice(None), slice(yy, yy + d.shape[3]), slice(xx, xx + d.shape[4]))
            out[sl] = np.maximum(out[sl], d.astype(np.float32))
        out[~np.isfinite(out)] = 0
        return out
    acc = np.zeros((t0, z0, nc, H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    for d, y, x in tiles:
        yy, xx = y - oy, x - ox
        w = _blend_weights(d.shape[3], d.shape[4]) if fusion == "Linear blend" else np.ones((d.shape[3], d.shape[4]), np.float32)
        acc[:, :, :, yy:yy + d.shape[3], xx:xx + d.shape[4]] += d.astype(np.float32) * w
        wsum[yy:yy + d.shape[3], xx:xx + d.shape[4]] += w
    return acc / np.where(wsum > 0, wsum, 1.0)


def _estimation_plane(img: LumenImage, t: int | None = None, z: int | None = None, c: int | None = None) -> np.ndarray:
    d = img.display
    t = min(d.t if t is None else t, img.nt - 1)
    z = min(d.z if z is None else z, img.nz - 1)
    c = min(d.c if c is None else c, img.nc - 1)
    return img.data[t, z, c]


def _read_tile(path: Path) -> LumenImage:
    """Read a tile with the standard reader, falling back to a plain TIFF read when the
    ImageJ-metadata parsing fails."""
    from ..io import read_image
    try:
        return read_image(path)
    except Exception:
        if path.suffix.lower() not in (".tif", ".tiff"):
            raise
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            series = tf.series[0]
            data = series.asarray()
            axes = series.axes
        return LumenImage(data, name=path.name, axes=axes if len(axes) == data.ndim else None, path=str(path))


def _visible_files(paths) -> list[Path]:
    """Regular, non-hidden files: skips '.DS_Store' and the macOS '._*' AppleDouble stubs (which
    pathlib's glob and an unanchored regex both match), like io.list_sequence_files / list_dir."""
    return [f for f in paths if f.is_file() and not f.name.startswith(".")]


def _natural_key(p: Path) -> list[Any]:
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", p.name)]


def _grid_order(index: int, gx: int, gy: int, order: str) -> tuple[int, int]:
    """(row, col) of the index-th tile."""
    if order == "Row-by-row":
        return divmod(index, gx)
    if order == "Column-by-column":
        c, r = divmod(index, gy)
        return r, c
    if order == "Snake by rows":
        r, c = divmod(index, gx)
        return r, (c if r % 2 == 0 else gx - 1 - c)
    if order == "Snake by columns":
        c, r = divmod(index, gy)
        return (r if c % 2 == 0 else gy - 1 - r), c
    raise ValueError(order)


# ================================================================================ ops
@op("registration.align_stack", "Align Stack (StackReg / SIFT)", CAT, kind="image", preview=False, scope_param=False, roi_aware=False,
    params=[Param("axis", "choice", "Z", choices=["Z", "T"]),
            Param("mode", "choice", "Translation", choices=ALIGN_MODES, label="Transformation", help="StackReg's models; Scaled rotation = similarity (SIFT dialog). Bilinear needs the intensity estimator (the feature estimators use an affine instead)"),
            Param("estimator", "choice", ESTIMATORS[0], choices=ESTIMATORS,
                  help="Phase correlation (+ ORB features for rotations); SIFT keypoints matched and filtered by RANSAC (Fiji 'Linear Stack Alignment with SIFT'); multi-resolution intensity optimisation (StackReg / TurboReg)"),
            Param("reference", "choice", "Previous slice", choices=REFS, help="Previous slice: chain from the first slice; Current slice: chain outward from the displayed slice (StackReg); First / Middle: every slice directly onto it"),
            Param("channel", "int", 0, min=0, help="Channel used to estimate the transform (0 = current); applied to all channels (MultiChannel)"),
            Param("upsample", "int", 10, min=1, max=100, label="Sub-pixel factor", help="Phase-correlation upsampling (1 = integer shifts)"),
            Param("interpolation", "choice", "Linear", choices=INTERP), Param("create_new", "bool", False, label="Create new image"),
            Param("initial_blur", "float", 1.6, min=0.5, max=10, step=0.1, label="SIFT: initial Gaussian blur (px)", advanced=True),
            Param("steps_per_octave", "int", 3, min=1, max=10, label="SIFT: steps per scale octave", advanced=True),
            Param("min_image_size", "int", 64, min=8, max=4096, label="SIFT: minimum image size (px)", advanced=True),
            Param("max_image_size", "int", 1024, min=16, max=16384, label="SIFT: maximum image size (px)", advanced=True),
            Param("descriptor_size", "int", 4, min=1, max=8, label="SIFT: feature descriptor size", advanced=True),
            Param("orientation_bins", "int", 8, min=2, max=36, label="SIFT: descriptor orientation bins", advanced=True),
            Param("max_ratio", "float", 0.92, min=0.1, max=1.0, step=0.01, label="SIFT: closest / next closest ratio", advanced=True),
            Param("max_alignment_error", "float", 25.0, min=0.1, max=1000, step=1, label="SIFT: maximal alignment error (px)", advanced=True),
            Param("min_inlier_ratio", "float", 0.05, min=0.0, max=1.0, step=0.01, label="SIFT: inlier ratio", advanced=True),
            Param("pyramid_levels", "int", 0, min=0, max=8, label="Intensity: pyramid levels (0 = auto)", advanced=True),
            Param("iterations", "int", 30, min=1, max=500, label="Intensity: iterations per level", advanced=True),
            Param("show_info", "bool", False, help="Add the feature / match / inlier counts (or the final correlation) of every slice to the Alignment table and log them"),
            Param("show_matrix", "bool", False, label="Show transformation matrices", help="Log the 3×3 matrix of every slice")],
    description="Register the slices (Z) or frames (T) of a stack to each other (StackReg / TurboReg, Linear Stack Alignment with SIFT): translation, rigid body, scaled rotation, affine or bilinear, estimated by phase correlation + features, by SIFT feature matching with RANSAC, or by multi-resolution intensity optimisation; estimated on one channel, applied to all. The per-slice transforms go to the 'Alignment' table.",
    keywords=["stackreg", "turboreg", "sift", "linear stack alignment", "multichannel", "register", "align slices", "jitter"])
def align_stack(ctx, axis="Z", mode="Translation", estimator=ESTIMATORS[0], reference="Previous slice", channel=0, upsample=10, interpolation="Linear", create_new=False,
                initial_blur=1.6, steps_per_octave=3, min_image_size=64, max_image_size=1024, descriptor_size=4, orientation_bins=8, max_ratio=0.92, max_alignment_error=25.0,
                min_inlier_ratio=0.05, pyramid_levels=0, iterations=30, show_info=False, show_matrix=False):
    from .registration_features import make_intensity_estimator, make_sift_estimator
    img = ctx.img
    d = img.display
    ax = 1 if axis == "Z" else 0
    n = img.data.shape[ax]
    if n < 2:
        raise ValueError("Needs a time series (T > 1)" if axis == "T" else "Needs a Z stack (Z > 1)")
    c = _channel_index(img, channel)
    order = ORDER_MAP[interpolation]
    current = d.z if axis == "Z" else d.t

    def get_plane(k: int) -> np.ndarray:
        return img.data[d.t, k, c] if axis == "Z" else img.data[k, d.z, c]

    est, chain_on_aligned, note = None, False, ""
    if estimator == ESTIMATORS[1]:
        est = make_sift_estimator(mode, dict(initial_blur=initial_blur, steps_per_octave=steps_per_octave, min_image_size=min_image_size, max_image_size=max_image_size,
                                             descriptor_size=descriptor_size, orientation_bins=orientation_bins, max_ratio=max_ratio, max_alignment_error=max_alignment_error,
                                             min_inlier_ratio=min_inlier_ratio), upsample)
        if mode == "Bilinear":
            note = " (bilinear → affine: SIFT has no bilinear model)"
    elif estimator == ESTIMATORS[2]:
        from .registration_features import pyramid_transform
        base_est = make_intensity_estimator(mode, int(pyramid_levels), int(iterations))

        def est(ref: np.ndarray, mov: np.ndarray, k_ref: int | None = None, k_mov: int | None = None, init: Any = None) -> tuple[Any, dict[str, Any]]:
            if init is None:
                return base_est(ref, mov, k_ref, k_mov)
            return pyramid_transform(ref, mov, mode, int(pyramid_levels), int(iterations), init=init)

        chain_on_aligned = mode == "Bilinear"
    elif mode == "Bilinear":
        note = " (bilinear → affine: the feature estimator has no bilinear model)"
    mats, infos = _estimate_series(get_plane, n, mode, reference, upsample, ctx=ctx, estimate=est, current=current, chain_on_aligned=chain_on_aligned, order=order)
    out = np.empty(img.shape, np.float32)
    for k in range(n):
        ctx.check_cancel()
        ctx.progress(0.5 + 0.5 * k / n, f"Aligning slice {k + 1}/{n}")
        m = mats[k]
        for other in range(img.data.shape[1 - ax]):
            for cc in range(img.nc):
                idx = (k, other, cc) if axis == "T" else (other, k, cc)
                src = img.data[idx[0], idx[1], idx[2]]
                out[idx[0], idx[1], idx[2]] = src if _is_identity(m) else _warp(src, m, order)
    rows = []
    log_lines = []
    for k in range(n):
        row: dict[str, Any] = {"Label": f"{img.name}:{axis}{k + 1}", "Slice": k + 1, **_decompose(mats[k])}
        info = infos[k]
        if show_info:
            if estimator == ESTIMATORS[1]:
                feats = info.get("features", (0, 0))
                row.update({"Features": int(feats[1]) if isinstance(feats, tuple) else 0, "Matches": int(info.get("matches", 0)), "Inliers": int(info.get("inliers", 0)),
                            "SIFT": bool(info.get("features_used", False))})
            elif estimator == ESTIMATORS[2]:
                row.update({"Correlation": round(float(info.get("correlation", float("nan"))), 4), "Levels": int(info.get("levels", 0))})
            else:
                row["Features"] = bool(info.get("features_used", False))
            log_lines.append(", ".join(f"{kk}: {vv}" for kk, vv in row.items() if kk not in ("Label",)))
        if show_matrix:
            m = mats[k] if isinstance(mats[k], np.ndarray) else mats[k].matrix
            log_lines.append(f"{row['Label']}: " + "; ".join(" ".join(f"{v:.5f}" for v in r_) for r_ in np.asarray(m, float)[:2]))
        rows.append(row)
    dmax = max(math.hypot(r["dx"], r["dy"]) for r in rows)
    label = {ESTIMATORS[1]: "SIFT", ESTIMATORS[2]: "intensity pyramid"}.get(estimator, "phase correlation + features")
    msg = f"Aligned {n} {'slices' if axis == 'Z' else 'frames'} ({mode}, {label}, ref: {reference.lower()}); max shift {dmax:.2f} px{note}"
    n_est = n - 1
    if estimator == ESTIMATORS[1]:
        n_feat = sum(1 for i in infos if i.get("features_used"))
        msg += f"; {n_feat} slices by SIFT matching, {n_est - n_feat} by phase correlation (too few inliers)"
    elif estimator == ESTIMATORS[0] and mode != "Translation":
        n_feat = sum(1 for i in infos if i.get("features_used"))
        msg += f"; {n_feat} feature-based, {n_est - n_feat} translation fallback"
    log = msg if not log_lines else msg + "\n" + "\n".join(log_lines)
    res = _finish(ctx, img, out, create_new, f"Aligned {img.name}", rows, msg)
    res.log = log
    return res


@op("registration.align_channels", "Align Channels", CAT, kind="image", preview=False, scope_param=False, roi_aware=False, supports_rgb=False,
    params=[Param("reference_channel", "int", 1, min=1), Param("estimate_on", "choice", "Current slice", choices=["Current slice", "Max projection (Z)"]),
            Param("upsample", "int", 10, min=1, max=100, label="Sub-pixel factor"), Param("interpolation", "choice", "Linear", choices=INTERP),
            Param("create_new", "bool", False, label="Create new image")],
    description="Chromatic-shift correction: estimate the xy shift of every channel relative to a reference channel by phase correlation and shift the whole channel. The per-channel shifts go to the 'Alignment' table.",
    keywords=["chromatic aberration", "channel shift", "colour shift"])
def align_channels(ctx, reference_channel=1, estimate_on="Current slice", upsample=10, interpolation="Linear", create_new=False):
    img = ctx.img
    d = img.display
    if img.nc < 2:
        raise ValueError("Needs at least two channels")
    r = _channel_index(img, reference_channel)
    order = ORDER_MAP[interpolation]

    def plane_for(c: int) -> np.ndarray:
        if estimate_on.startswith("Max") and img.nz > 1:
            return img.data[d.t, :, c].max(axis=0)
        return img.data[d.t, d.z, c]

    ref = plane_for(r)
    out = img.data.astype(np.float32, copy=True)
    rows = []
    for c in range(img.nc):
        if c == r:
            rows.append({"Label": f"{img.name}:C{c + 1}", "Channel": c + 1, "dx": 0.0, "dy": 0.0})
            continue
        dy, dx = _phase_shift(ref, plane_for(c), upsample)
        rows.append({"Label": f"{img.name}:C{c + 1}", "Channel": c + 1, "dx": dx, "dy": dy})
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        for t in range(img.nt):
            for z in range(img.nz):
                out[t, z, c] = ndi.shift(out[t, z, c], (dy, dx), order=order, mode="constant", cval=0.0)
    msg = "Channel shifts: " + ", ".join(f"C{r_['Channel']} ({r_['dx']:+.2f}, {r_['dy']:+.2f})" for r_ in rows if r_["Channel"] != r + 1)
    return _finish(ctx, img, out, create_new, f"Aligned {img.name}", rows, msg)


@op("registration.register_to_image", "Register to Image", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("target", "image", None, help="The open image to align onto"), Param("mode", "choice", "Translation", choices=MODES, label="Transformation"),
            Param("upsample", "int", 10, min=1, max=100, label="Sub-pixel factor"), Param("interpolation", "choice", "Linear", choices=INTERP)],
    description="Align the current image onto another open image (estimated on the current planes, applied to every plane); the result has the target's size.",
    keywords=["register", "align two images", "overlay"])
def register_to_image(ctx, target=None, mode="Translation", upsample=10, interpolation="Linear"):
    img = ctx.img
    if not target:
        raise ValueError("Choose a target image")
    tgt = ctx.session.get_image(target)
    if tgt.id == img.id:
        raise ValueError("Target must be a different image")
    order = ORDER_MAP[interpolation]
    ht, wt = tgt.height, tgt.width
    h, w = img.height, img.width
    hh, ww = max(ht, h), max(wt, w)

    def padded(p: np.ndarray) -> np.ndarray:
        return np.pad(_f32(p), ((0, hh - p.shape[0]), (0, ww - p.shape[1])))

    tp, ip = _estimation_plane(tgt), _estimation_plane(img)
    m, feat = None, False
    if mode in ("Rigid body", "Affine"):
        m = _feature_matrix(padded(tp), padded(ip), mode)
        feat = m is not None
    if m is None:  # Translation mode, or feature matching failed: a translation valid for any offset
        dy, dx = _translation_any_offset(tp, ip, upsample)
        m = _translation(dx, dy)
    out = np.zeros((img.nt, img.nz, img.nc, ht, wt), np.float32)
    for t in range(img.nt):
        for z in range(img.nz):
            for c in range(img.nc):
                canvas = np.zeros((hh, ww), np.float32)
                canvas[:h, :w] = img.data[t, z, c]
                out[t, z, c] = _warp(canvas, m, order)[:ht, :wt]
    res = img.copy(name=f"{img.name} on {tgt.name}", data=cast_to(out, img.data.dtype))
    res.calibration = Calibration(**tgt.calibration.to_dict())
    res.sync_channels()
    info = _decompose(m)
    rows = [{"Label": f"{img.name} → {tgt.name}", **info}]
    msg = f"Registered onto {tgt.name}: dx {info['dx']:+.2f}, dy {info['dy']:+.2f}, angle {info['Angle']:+.2f}°" + ("" if feat or mode == "Translation" else " (translation fallback)")
    return OpResult(new_images=[res], results=rows, message=msg, log=msg)


@op("registration.pairwise_stitch", "Pairwise Stitching", CAT, kind="image", requires_image=False, preview=False, scope_param=False, roi_aware=False,
    params=[Param("image1", "image", None), Param("image2", "image", None), Param("fusion", "choice", "Linear blend", choices=FUSION),
            Param("check_peaks", "int", 5, min=1, max=50, advanced=True, help="Phase-correlation peaks verified on the real overlap"),
            Param("title", "string", "Stitched")],
    description="Find the overlap between two images by phase correlation and fuse them into one canvas.", keywords=["stitch", "mosaic", "overlap"])
def pairwise_stitch(ctx, image1=None, image2=None, fusion="Linear blend", check_peaks=5, title="Stitched"):
    s = ctx.session
    if not image1 or not image2:
        raise ValueError("Choose two open images")
    a = s.get_image(image1)
    b = s.get_image(image2)
    if a.id == b.id:
        raise ValueError("Choose two different images")
    da = a.display
    pa = _estimation_plane(a)
    pb = _estimation_plane(b, da.t, da.z, da.c)
    dy, dx, ncc = _find_offset(pa, pb, n_peaks=check_peaks)
    if ncc < 0:
        raise ValueError("No overlap found between the two images")
    fused = _fuse([(a.data, 0, 0), (b.data, dy, dx)], fusion)
    res = a.copy(name=title, data=cast_to(fused, a.data.dtype))
    if fusion == "Overlay (channels)":
        res.rgb = False
        res.reset_display()
        # every tile is its own channel: show them overlaid (Fiji OverlayFusion -> CompositeImage.COMPOSITE);
        # copy() carried the source's mode ('single' for 1-channel tiles) and reset_display() keeps it
        res.display.mode = "composite" if res.nc > 1 else "single"
        res.display.c = 0
    else:
        res.sync_channels()
    res.display.t = res.display.z = 0
    res.overlay = []
    rows = [{"Label": f"{b.name} → {a.name}", "dx": dx, "dy": dy, "Correlation": ncc}]
    msg = f"{b.name} is at ({dx:+d}, {dy:+d}) relative to {a.name} (R = {ncc:.3f}); canvas {res.width} × {res.height}"
    return OpResult(new_images=[res], results=rows, message=msg, log=msg)


@op("registration.grid_stitch", "Grid Stitching", CAT, kind="image", requires_image=False, preview=False, scope_param=False, roi_aware=False,
    params=[Param("folder", "string", "", help="Folder containing the tile images"), Param("pattern", "string", "*.tif", help="Glob (e.g. tile_*.tif) or regular expression"),
            Param("grid_x", "int", 2, min=1, label="Grid size X"), Param("grid_y", "int", 2, min=1, label="Grid size Y"),
            Param("overlap_percent", "float", 10.0, min=0, max=90, step=1, label="Tile overlap (%)"), Param("order", "choice", "Row-by-row", choices=ORDERS, label="Tile order"),
            Param("refine", "bool", True, help="Refine every neighbour pair by phase correlation (otherwise pure grid placement)"),
            Param("fusion", "choice", "Linear blend", choices=FUSION),
            Param("min_correlation", "float", 0.3, min=-1, max=1, step=0.05, advanced=True, help="Pairs with a lower overlap correlation fall back to the grid position"),
            Param("title", "string", "", help="Name of the stitched image (default: folder name)")],
    description="Stitch a folder of tiles arranged in a regular grid (Grid/Collection stitching).", keywords=["mosaic", "tiles", "stitch", "collection"])
def grid_stitch(ctx, folder="", pattern="*.tif", grid_x=2, grid_y=2, overlap_percent=10.0, order="Row-by-row", refine=True, fusion="Linear blend",
                min_correlation=0.3, title=""):
    if not folder:
        raise ValueError("Give the folder that contains the tiles")
    root = Path(folder).expanduser()
    if not root.is_dir():
        raise ValueError(f"{root} is not a folder")
    pattern = pattern.strip() or "*"
    files = sorted(_visible_files(root.glob(pattern)), key=_natural_key) if any(ch in pattern for ch in "*?[") else []
    if not files:
        try:
            rx = re.compile(pattern)
            files = sorted((f for f in _visible_files(root.iterdir()) if rx.search(f.name)), key=_natural_key)
        except re.error:
            files = []
    n = grid_x * grid_y
    if len(files) < n:
        raise ValueError(f"Found {len(files)} tiles matching {pattern!r} in {root}, need {n} for a {grid_x}×{grid_y} grid")
    extra = files[n:]
    files = files[:n]
    tiles = []
    for i, f in enumerate(files):
        ctx.check_cancel()
        ctx.progress(0.3 * i / n, f"Reading tile {i + 1}/{n}")
        try:
            tiles.append(_read_tile(f))
        except Exception as e:  # grid position is assigned by list index: never skip a tile silently
            raise ValueError(f"Could not read tile {f.name}: {e}") from e
    first = tiles[0]
    for im, f in zip(tiles, files):
        if im.shape != first.shape:
            raise ValueError(f"{f.name} has shape {im.shape}, expected {first.shape}")
    h, w = first.height, first.width
    ov = max(0.0, min(overlap_percent, 90.0)) / 100.0
    rc = [_grid_order(i, grid_x, grid_y, order) for i in range(n)]
    nominal = np.array([[r * h * (1 - ov), c * w * (1 - ov)] for r, c in rc], dtype=np.float64)  # (y, x)
    pos = nominal.copy()
    pair_rows: list[dict[str, Any]] = []
    if refine and n > 1:
        by_rc = {rc[i]: i for i in range(n)}
        planes = [_estimation_plane(im, 0, im.nz // 2, 0) for im in tiles]
        eqs: list[tuple[int, int, float, float, float]] = []  # (i, j, dy, dx, weight)
        max_dev = (max(h * 0.5, 8.0), max(w * 0.5, 8.0))
        for i in range(n):
            ctx.check_cancel()
            ctx.progress(0.3 + 0.6 * i / n, f"Refining tile {i + 1}/{n}")
            r, c = rc[i]
            for (rr, cc) in ((r, c + 1), (r + 1, c)):
                j = by_rc.get((rr, cc))
                if j is None:
                    continue
                exp = (nominal[j, 0] - nominal[i, 0], nominal[j, 1] - nominal[i, 1])
                dy, dx, ncc = _find_offset(planes[i], planes[j], n_peaks=8, expected=exp, max_dev=max_dev)
                ok = ncc >= min_correlation
                pair_rows.append({"Label": f"{files[i].name} → {files[j].name}", "Tile A": i + 1, "Tile B": j + 1, "dx": dx, "dy": dy, "Correlation": ncc, "Used": ok})
                if ok:
                    eqs.append((i, j, float(dy), float(dx), max(float(ncc), 1e-3)))
        # global least squares: p_j - p_i = d_ij (weight ncc), weak prior p_i = nominal_i
        rows_a: list[np.ndarray] = []
        rhs_y: list[float] = []
        rhs_x: list[float] = []
        for i, j, dy, dx, wgt in eqs:
            row = np.zeros(n)
            row[i], row[j] = -wgt, wgt
            rows_a.append(row)
            rhs_y.append(wgt * dy)
            rhs_x.append(wgt * dx)
        prior = 1e-3
        for i in range(n):
            row = np.zeros(n)
            row[i] = prior
            rows_a.append(row)
            rhs_y.append(prior * nominal[i, 0])
            rhs_x.append(prior * nominal[i, 1])
        A = np.array(rows_a)
        py = np.linalg.lstsq(A, np.array(rhs_y), rcond=None)[0]
        px = np.linalg.lstsq(A, np.array(rhs_x), rcond=None)[0]
        pos = np.stack([py, px], axis=1)
    pos = pos - pos.min(axis=0)
    ipos = np.rint(pos).astype(int)
    ctx.check_cancel()
    ctx.progress(0.9, "Fusing tiles")
    fused = _fuse([(im.data, int(ipos[i, 0]), int(ipos[i, 1])) for i, im in enumerate(tiles)], fusion)
    res = first.copy(name=title or f"Stitched {root.name}", data=cast_to(fused, first.data.dtype))
    if fusion == "Overlay (channels)":
        res.rgb = False
        res.reset_display()
        res.display.mode = "composite" if res.nc > 1 else "single"  # see pairwise_stitch
    else:
        res.sync_channels()
    res.display.t = res.display.z = res.display.c = 0
    res.overlay = []
    res.metadata["stitch_positions"] = [[int(ipos[i, 1]), int(ipos[i, 0])] for i in range(n)]
    cols = ["Tile", "File", "Row", "Col", "X", "Y"]
    table = {"name": "Tile positions", "columns": cols, "rows": [[i + 1, files[i].name, rc[i][0] + 1, rc[i][1] + 1, int(ipos[i, 1]), int(ipos[i, 0])] for i in range(n)]}
    used = sum(1 for r in pair_rows if r["Used"])
    msg = f"Stitched {n} tiles ({grid_x}×{grid_y}, {order.lower()}) into {res.width} × {res.height}"
    if refine and n > 1:
        msg += f"; {used}/{len(pair_rows)} neighbour pairs refined"
    if extra:
        msg += f"; {len(extra)} further file{'s' if len(extra) != 1 else ''} matching {pattern!r} ignored ({', '.join(f.name for f in extra[:5])}{', …' if len(extra) > 5 else ''})"
    return OpResult(new_images=[res], results=pair_rows or None, table=table, message=msg, log=msg)


@op("registration.drift_correct", "Correct Drift (3D)", CAT, kind="image", preview=False, scope_param=False, roi_aware=False,
    params=[Param("channel", "int", 0, min=0, help="Channel used to estimate the drift (0 = current)"),
            Param("reference", "choice", "Previous frame", choices=["Previous frame", "First frame"]),
            Param("correct_z", "bool", False, label="Also correct Z drift", help="Estimate the Z shift on XZ max projections"),
            Param("upsample", "int", 10, min=1, max=100, label="Sub-pixel factor"), Param("interpolation", "choice", "Linear", choices=INTERP),
            Param("create_new", "bool", False, label="Create new image")],
    description="Correct sample drift in a time series: xy shift per frame from the max projection (optionally z from XZ projections), applied to all slices and channels. The per-frame shifts go to the 'Alignment' table.",
    keywords=["drift", "3d drift", "time lapse", "translation"])
def drift_correct(ctx, channel=0, reference="Previous frame", correct_z=False, upsample=10, interpolation="Linear", create_new=False):
    img = ctx.img
    if img.nt < 2:
        raise ValueError("Needs a time series (T > 1)")
    c = _channel_index(img, channel)
    order = ORDER_MAP[interpolation]
    do_z = correct_z and img.nz > 1

    def xy_proj(t: int) -> np.ndarray:
        return img.data[t, :, c].max(axis=0) if img.nz > 1 else img.data[t, 0, c]

    def xz_proj(t: int) -> np.ndarray:
        return img.data[t, :, c].max(axis=1)  # (Z, X)

    shifts = np.zeros((img.nt, 3))  # dz, dy, dx
    ref_xy, ref_xz = xy_proj(0), (xz_proj(0) if do_z else None)
    for t in range(1, img.nt):
        ctx.check_cancel()
        ctx.progress(0.5 * t / img.nt, f"Estimating drift of frame {t + 1}/{img.nt}")
        cur_xy = xy_proj(t)
        dy, dx = _phase_shift(ref_xy, cur_xy, upsample)
        dz = 0.0
        if do_z:
            cur_xz = xz_proj(t)
            dz, _ = _phase_shift(ref_xz, cur_xz, upsample)
        if reference == "Previous frame":
            shifts[t] = shifts[t - 1] + (dz, dy, dx)
            ref_xy, ref_xz = cur_xy, (cur_xz if do_z else None)
        else:
            shifts[t] = (dz, dy, dx)
    out = np.empty(img.shape, np.float32)
    for t in range(img.nt):
        ctx.check_cancel()
        ctx.progress(0.5 + 0.5 * t / img.nt, f"Shifting frame {t + 1}/{img.nt}")
        dz, dy, dx = shifts[t]
        for cc in range(img.nc):
            vol = _f32(img.data[t, :, cc])
            if abs(dz) < 1e-9 and abs(dy) < 1e-9 and abs(dx) < 1e-9:
                out[t, :, cc] = vol
            elif abs(dz) < 1e-9:
                for z in range(img.nz):
                    out[t, z, cc] = ndi.shift(vol[z], (dy, dx), order=order, mode="constant", cval=0.0)
            else:
                out[t, :, cc] = ndi.shift(vol, (dz, dy, dx), order=order, mode="constant", cval=0.0)
    rows = [{"Label": f"{img.name}:t{t + 1}", "Frame": t + 1, "dx": float(shifts[t, 2]), "dy": float(shifts[t, 1]), "dz": float(shifts[t, 0])} for t in range(img.nt)]
    dmax = float(np.max(np.hypot(shifts[:, 1], shifts[:, 2])))
    msg = f"Drift corrected over {img.nt} frames (max xy drift {dmax:.2f} px" + (f", max z drift {np.abs(shifts[:, 0]).max():.2f} slices)" if do_z else ")")
    return _finish(ctx, img, out, create_new, f"Drift corrected {img.name}", rows, msg)
