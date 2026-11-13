"""Plugins ▸ Transform — moving least squares (MLS) deformation from landmark pairs (Schaefer, McPhail & Warren
2006: affine / similarity / rigid) as in Fiji's *Interactive Moving Least Squares*, and the two-image *Landmark
Correspondences* command (point selections on a source and a target image; translation … perspective or MLS).

The deformation is applied as an inverse mapping: for every output pixel the MLS transform built from the *target*
landmarks onto the *source* landmarks gives the source position to sample, so the source landmarks land exactly on the
target landmarks. Point coordinates follow Lumen's ROI convention (pixel ``i`` spans ``[i, i + 1)``) and are shifted by
−0.5 to pixel centres before the transform is built.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import ProjectiveTransform

from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op, cast_to
from .registration import INTERP, ORDER_MAP
from .registration_extra import CORRESPONDENCES_KEY, _apply_matrix, estimate_point_transform, transform_params
from .transform_extra import _manager_roi, _points_of_roi, _warp_all, parse_points

CAT = "Image/Transform"
MLS_MODELS = ["Affine", "Similarity", "Rigid"]
CORRESPONDENCE_MODELS = ["Translation", "Rigid body", "Similarity", "Affine", "Projective",
                         "Moving least squares (affine)", "Moving least squares (similarity)", "Moving least squares (rigid)"]
MLS_KEY = "mls_transform"


# ============================================================================ MLS deformation
def mls_map(p: np.ndarray, q: np.ndarray, xx: np.ndarray, yy: np.ndarray, alpha: float = 1.0, model: str = "Affine") -> tuple[np.ndarray, np.ndarray]:
    """Moving-least-squares deformation ``f`` with ``f(p_i) = q_i``, evaluated at the points (xx, yy): returns (fx, fy).

    ``p`` / ``q`` are (N, 2) arrays of (x, y) control points; ``model`` is Affine, Similarity or Rigid; ``alpha`` is the
    weight exponent (``w_i = 1 / |p_i − v|^(2 α)``). Evaluated in chunks so that memory stays bounded.
    """
    p = np.asarray(p, np.float64).reshape(-1, 2)
    q = np.asarray(q, np.float64).reshape(-1, 2)
    n = len(p)
    if n == 0 or len(q) != n:
        raise ValueError("MLS needs the same number of source and target points (at least one)")
    xx = np.asarray(xx, np.float64)
    yy = np.asarray(yy, np.float64)
    shape = xx.shape
    vx, vy = xx.ravel(), yy.ravel()
    if n == 1:
        return (vx + (q[0, 0] - p[0, 0])).reshape(shape), (vy + (q[0, 1] - p[0, 1])).reshape(shape)
    if model not in MLS_MODELS:
        raise ValueError(f"Unknown MLS model {model!r}; choose one of {MLS_MODELS}")
    # Fiji's mpicbg models refuse ill-defined landmark sets (NotEnoughDataPointsException / IllDefinedDataPointsException)
    # instead of returning a warp: the affine moment matrix Σ w p̂ p̂ᵀ is singular at every pixel for fewer than three or
    # collinear landmarks, so the per-pixel solve would only amplify round-off; similarity / rigid are exact with two
    cen = p - p.mean(axis=0)
    ev = np.linalg.eigvalsh(cen.T @ cen)  # ascending
    if ev[-1] <= 1e-18:
        raise ValueError("The landmarks all lie at the same position – the set is ill-defined")
    if model == "Affine" and (n < 3 or ev[0] <= 1e-9 * ev[-1]):
        why = f"{n} given" if n < 3 else f"{n} given, collinear"
        raise ValueError(f"Affine moving least squares needs at least 3 non-collinear landmarks ({why}) – add landmarks or use Similarity / Rigid")
    fx, fy = np.empty_like(vx), np.empty_like(vy)
    chunk = int(max(1024, 2_000_000 // n))
    px, py = p[:, 0:1], p[:, 1:2]
    qx, qy = q[:, 0:1], q[:, 1:2]
    a = max(float(alpha), 1e-3)
    for start in range(0, vx.size, chunk):
        sl = slice(start, start + chunk)
        cx, cy = vx[sl], vy[sl]
        dxp, dyp = cx[None, :] - px, cy[None, :] - py  # (n, M)
        d2 = dxp * dxp + dyp * dyp
        # w_i = |p_i − v|^(−2α), normalised per pixel: the MLS solution is invariant to a positive per-pixel factor, and
        # the normalisation (done in log space so that large α cannot overflow) keeps the moment sums O(geometry)
        # instead of O(d^−4α), where an absolute tolerance would wrongly flag valid landmark sets as degenerate
        logw = -a * np.log(np.maximum(d2, 1e-12))
        w = np.exp(logw - logw.max(axis=0, keepdims=True))
        w /= w.sum(axis=0, keepdims=True)
        psx, psy = (w * px).sum(axis=0), (w * py).sum(axis=0)
        qsx, qsy = (w * qx).sum(axis=0), (w * qy).sum(axis=0)
        phx, phy = px - psx[None, :], py - psy[None, :]
        qhx, qhy = qx - qsx[None, :], qy - qsy[None, :]
        ex, ey = cx - psx, cy - psy  # v − p*
        # complex form of the similarity solution: f(v) = q* + a · (v − p*), a = Σ w q̂ conj(p̂) / Σ w |p̂|²; computed for
        # every model, as it is also the fallback of pixels whose affine moment matrix is numerically singular
        num_re = (w * (qhx * phx + qhy * phy)).sum(0)
        num_im = (w * (qhy * phx - qhx * phy)).sum(0)
        mu = (w * (phx * phx + phy * phy)).sum(0)
        if model == "Rigid":  # the rotation part of the similarity (unit modulus)
            mod = np.hypot(num_re, num_im)
            ok = mod > 1e-12 * mu
            den = np.where(ok, mod, 1.0)
        else:
            ok = mu > 1e-300
            den = np.where(ok, mu, 1.0)
        a_re, a_im = np.where(ok, num_re / den, 1.0), np.where(ok, num_im / den, 0.0)
        rx, ry = a_re * ex - a_im * ey + qsx, a_re * ey + a_im * ex + qsy
        if model == "Affine":
            aa, bb, dd = (w * phx * phx).sum(0), (w * phx * phy).sum(0), (w * phy * phy).sum(0)
            det = aa * dd - bb * bb
            good = det > 1e-12 * (aa + dd) ** 2  # relative test: ill-conditioned pixels keep the similarity solution
            det_s = np.where(good, det, 1.0)
            bxx, bxy = (w * phx * qhx).sum(0), (w * phx * qhy).sum(0)
            byx, byy = (w * phy * qhx).sum(0), (w * phy * qhy).sum(0)
            m11, m12 = (dd * bxx - bb * byx) / det_s, (dd * bxy - bb * byy) / det_s
            m21, m22 = (-bb * bxx + aa * byx) / det_s, (-bb * bxy + aa * byy) / det_s
            rx = np.where(good, ex * m11 + ey * m21 + qsx, rx)
            ry = np.where(good, ex * m12 + ey * m22 + qsy, ry)
        # exactly interpolating at the control points themselves
        near = np.argmin(d2, axis=0)
        at = d2[near, np.arange(d2.shape[1])] < 1e-9
        if at.any():
            rx[at], ry[at] = q[near[at], 0], q[near[at], 1]
        fx[sl], fy[sl] = rx, ry
    return fx.reshape(shape), fy.reshape(shape)


def mls_warp_planes(img: LumenImage, src: np.ndarray, dst: np.ndarray, out_shape: tuple[int, int], alpha: float, model: str, order: int) -> np.ndarray:
    """Warp every plane of ``img`` so that the source landmarks ``src`` land on ``dst`` (pixel-centre coordinates)."""
    yy, xx = np.mgrid[0:out_shape[0], 0:out_shape[1]]
    fx, fy = mls_map(dst, src, xx, yy, alpha=alpha, model=model)
    coords = [np.round(fy, 6), np.round(fx, 6)]  # float noise of exact transforms must not push edge pixels outside
    out = np.zeros((img.nt, img.nz, img.nc) + tuple(out_shape), np.float32)
    for t, z, c in img.iter_planes():
        out[t, z, c] = ndi.map_coordinates(img.data[t, z, c].astype(np.float32), coords, order=order, mode="constant", cval=0.0)
    return out


# ============================================================================ landmark sources
def _manager_points(ctx, img: LumenImage) -> np.ndarray:
    pts = np.zeros((0, 2))
    for r in ctx.session.roi_manager.rois:
        if r.get("type") == "point" and r.get("image_id") == img.id:
            pts = np.vstack([pts, np.asarray(r.get("points", []), float).reshape(-1, 2)])
    return pts


def _table_correspondences(ctx) -> tuple[np.ndarray, np.ndarray]:
    """(X1, Y1) / (X2, Y2) columns of a 'Correspondences' table in the session (Extract SIFT Correspondences)."""
    tab = ctx.session.tables.get("Correspondences")
    if not tab:
        return np.zeros((0, 2)), np.zeros((0, 2))
    cols = tab.get("columns", [])
    try:
        ix = [cols.index(k) for k in ("X1", "Y1", "X2", "Y2")]
    except ValueError:
        return np.zeros((0, 2)), np.zeros((0, 2))
    rows = [[float(r[i]) for i in ix] for r in tab.get("rows", [])]
    arr = np.asarray(rows, float).reshape(-1, 4)
    return arr[:, :2], arr[:, 2:]


def _landmark_pairs(ctx, img: LumenImage, source_points: str, target_points: str, source_roi: str, target_roi: str,
                    target_img: LumenImage | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Source / target landmark arrays (ROI coordinates) from text, ROI-Manager names, the selection, point ROIs
    attached to the images, the stored SIFT correspondences or the 'Correspondences' table."""
    src = parse_points(source_points)
    if src.size == 0 and source_roi:
        src = _points_of_roi(_manager_roi(ctx, source_roi))
    if src.size == 0:
        src = _points_of_roi(ctx.roi)
    if src.size == 0:
        src = _manager_points(ctx, img)
    dst = parse_points(target_points)
    if dst.size == 0 and target_roi:
        dst = _points_of_roi(_manager_roi(ctx, target_roi))
    if dst.size == 0 and target_img is not None:
        dst = _manager_points(ctx, target_img)
    if src.size == 0 or dst.size == 0:
        corr = ctx.session.settings.get(CORRESPONDENCES_KEY)
        if corr and corr.get("source") == img.id and (target_img is None or corr.get("target") == target_img.id):
            src, dst = np.asarray(corr["src"], float), np.asarray(corr["dst"], float)
        elif src.size == 0 and dst.size == 0:
            src, dst = _table_correspondences(ctx)
    if src.size == 0:
        raise ValueError("Give the source landmarks: a point selection, point ROIs of the source image in the ROI Manager, source_points 'x1,y1; x2,y2; …' or source_roi")
    if dst.size == 0:
        raise ValueError("Give the target landmarks: target_points 'x1,y1; x2,y2; …', target_roi" + (", point ROIs of the target image" if target_img is not None else "") + " or a Correspondences table")
    if len(src) != len(dst):
        raise ValueError(f"{len(src)} source points but {len(dst)} target points – they must correspond one to one")
    return src, dst


def _mls_result(ctx, img: LumenImage, src: np.ndarray, dst: np.ndarray, model: str, alpha: float, out_shape: tuple[int, int], order: int,
                create_new: bool, name: str, like: LumenImage | None, label: str) -> OpResult:
    data = mls_warp_planes(img, src - 0.5, dst - 0.5, out_shape, alpha, model, order)
    yy, xx = np.mgrid[0:out_shape[0]:max(out_shape[0] // 64, 1), 0:out_shape[1]:max(out_shape[1] // 64, 1)]
    fx, fy = mls_map(dst - 0.5, src - 0.5, xx, yy, alpha=alpha, model=model)
    mag = np.hypot(fx - xx, fy - yy)
    row = {"Label": label, "Model": f"Moving least squares ({model.lower()})", "N landmarks": int(len(src)), "Alpha": float(alpha),
           "Mean displacement": float(mag.mean()), "Max displacement": float(mag.max())}
    ctx.session.settings[MLS_KEY] = {"src": src.tolist(), "dst": dst.tolist(), "model": model, "alpha": float(alpha)}
    msg = f"Moving least squares ({model.lower()}, α = {alpha:g}) from {len(src)} landmarks: mean displacement {row['Mean displacement']:.2f} px, max {row['Max displacement']:.2f} px"
    out = cast_to(data, img.data.dtype)
    if create_new:
        res = img.copy(name=name, data=out)
        if like is not None:
            res.calibration = Calibration(**like.calibration.to_dict())
        res.overlay = []
        res.sync_channels()
        return OpResult(new_images=[res], results=[row], message=msg, log=msg)
    return OpResult(data=out, rgb=img.rgb, results=[row], clear_roi=True, message=msg, log=msg)


# ============================================================================ ops
@op("image.transform_mls", "Moving Least Squares Deformation", CAT, kind="image", preview=False, scope_param=False, roi_aware=False,
    params=[Param("model", "choice", "Affine", choices=MLS_MODELS, label="Local transformation", help="Affine: free local stretch; Similarity: local rotation + scale; Rigid: local rotation only"),
            Param("target_points", "text", "", help="Target positions 'x1,y1; x2,y2; …' matching the source points in order (empty = target_roi / stored correspondences / Correspondences table)"),
            Param("source_points", "text", "", help="Source positions (empty = the current point / polygon selection, source_roi or the point ROIs of this image)"),
            Param("source_roi", "string", "", help="Name of a point ROI in the ROI Manager with the source points"),
            Param("target_roi", "string", "", help="Name of a point ROI in the ROI Manager with the target points"),
            Param("alpha", "float", 1.0, min=0.1, max=5.0, step=0.1, help="Weight exponent: larger = more local influence of each landmark"),
            Param("width", "int", 0, min=0, help="Output width (0 = same as the image)"), Param("height", "int", 0, min=0, help="Output height (0 = same)"),
            Param("interpolation", "choice", "Linear", choices=INTERP), Param("create_new", "bool", True, label="Create new image")],
    description="Deform the image with moving least squares (Schaefer et al.) so that every source landmark lands exactly on its target landmark, with a locally affine / similarity / rigid deformation in between (Fiji: Interactive Moving Least Squares). Landmarks come from point selections, the ROI Manager, typed lists or the correspondences of 'Extract SIFT Correspondences'.",
    keywords=["moving least squares", "mls", "interactive moving least squares", "landmarks", "warp", "deformation", "non-rigid"])
def transform_mls(ctx, model="Affine", target_points="", source_points="", source_roi="", target_roi="", alpha=1.0, width=0, height=0, interpolation="Linear", create_new=True):
    img = ctx.img
    src, dst = _landmark_pairs(ctx, img, source_points, target_points, source_roi, target_roi)
    out_shape = (int(height) if height > 0 else img.height, int(width) if width > 0 else img.width)
    return _mls_result(ctx, img, src, dst, model, float(alpha), out_shape, ORDER_MAP[interpolation], create_new, f"{img.name} (mls)", None, img.name)


@op("image.transform_landmark_correspondences", "Landmark Correspondences (two images)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("target", "image", None, help="The open target image; the current image is the source"),
            Param("model", "choice", "Affine", choices=CORRESPONDENCE_MODELS, label="Transformation"),
            Param("source_points", "text", "", help="Source landmarks 'x1,y1; …' (empty = the point selection, source_roi or the point ROIs of the source image in the ROI Manager)"),
            Param("target_points", "text", "", help="Target landmarks in the same order (empty = target_roi or the point ROIs of the target image in the ROI Manager)"),
            Param("source_roi", "string", "", help="Name of a point ROI in the ROI Manager"), Param("target_roi", "string", ""),
            Param("alpha", "float", 1.0, min=0.1, max=5.0, step=0.1, help="Moving least squares: weight exponent"),
            Param("interpolation", "choice", "Linear", choices=INTERP, help="Nearest = Fiji's 'interpolate' unchecked"),
            Param("show_matrix", "bool", True, label="Transform parameters to Results")],
    description="Transform the current (source) image into the coordinate frame of a target image from corresponding point selections on both images: translation, rigid, similarity, affine, perspective (least squares) or moving least squares (locally affine / similarity / rigid). The result has the target's size (Fiji: Plugins ▸ Transform ▸ Landmark Correspondences).",
    keywords=["landmark correspondences", "point correspondences", "register landmarks", "moving least squares", "two images", "warp onto"])
def transform_landmark_correspondences(ctx, target=None, model="Affine", source_points="", target_points="", source_roi="", target_roi="", alpha=1.0,
                                       interpolation="Linear", show_matrix=True):
    img = ctx.img
    if not target:
        raise ValueError("Choose a target image")
    tgt = ctx.session.get_image(target)
    if tgt.id == img.id:
        raise ValueError("Target must be a different image")
    src, dst = _landmark_pairs(ctx, img, source_points, target_points, source_roi, target_roi, target_img=tgt)
    order = ORDER_MAP[interpolation]
    out_shape = (tgt.height, tgt.width)
    label = f"{img.name} → {tgt.name}"
    name = f"{img.name} on {tgt.name}"
    if model.startswith("Moving least squares"):
        local = model[model.index("(") + 1:-1].capitalize()
        res = _mls_result(ctx, img, src, dst, local, float(alpha), out_shape, order, True, name, tgt, label)
        if not show_matrix:
            res.results = None
        return res
    m, _ = estimate_point_transform(src - 0.5, dst - 0.5, model, use_ransac=False)
    data = cast_to(_warp_all(img, ProjectiveTransform(matrix=m).inverse, out_shape, order), img.data.dtype)
    res_img = img.copy(name=name, data=data)
    res_img.calibration = Calibration(**tgt.calibration.to_dict())
    res_img.overlay = []
    res_img.sync_channels()
    resid = np.hypot(*(_apply_matrix(m, src - 0.5) - (dst - 0.5)).T)
    params = transform_params(m)
    row = {"Label": label, "Model": model, "N landmarks": int(len(src)), **params, "RMS residual": float(np.sqrt(np.mean(resid ** 2)))}
    ctx.session.settings["landmark_transform"] = {"matrix": m.tolist(), "model": model, "source": img.id, "target": tgt.id}
    msg = (f"{model} transform of {img.name} onto {tgt.name} from {len(src)} landmarks: dx {params['dx']:+.2f}, dy {params['dy']:+.2f}, "
           f"angle {params['Angle']:+.2f}°, scale {params['Scale']:.3f} (RMS residual {row['RMS residual']:.2f} px)")
    return OpResult(new_images=[res_img], results=[row] if show_matrix else None, message=msg, log=msg)
