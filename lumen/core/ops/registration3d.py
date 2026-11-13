"""Plugins ▸ Registration — 3-D registration: descriptor-style volume registration onto a
target volume (translation / rigid / landmark affine), 'Correct 3D Drift' with full 3-D phase
correlation, 3-D chromatic-shift correction, extended depth of field (Stack Focuser) and
block matching (sparse displacement fields by normalised cross-correlation).

Conventions
-----------
* Volumes are ``(Z, Y, X)`` float32 arrays; shifts are ``(dz, dy, dx)`` in *voxels* and
  always the shift that moves the moving volume onto the reference
  (``ndi.shift(moving, shift) ≈ reference``).
* 3-D transforms are 4×4 homogeneous matrices in **(x, y, z) voxel-centre coordinates**
  mapping the moving volume onto the reference (same direction as the 3×3 matrices of
  ``registration.py``). They are converted to ``(z, y, x)`` index order only when applied with
  ``scipy.ndimage.affine_transform``.
* Anisotropy: landmark fits are done in calibrated space (voxel sizes from the image
  calibration) and converted back; rigid in-plane rotations are estimated on the (isotropic)
  XY plane; the tables report voxel shifts and, when calibrated, physical shifts as well.
"""
from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.signal.windows import tukey
from skimage.registration import phase_cross_correlation

from .. import roi as roimod
from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op, cast_to
from .registration import INTERP, ORDER_MAP, _alignment_table, _channel_index, _f32, _feature_matrix, _phase_shift, _translation, _warp
from .registration_extra import DEFORMATION_KEY, _field_image, warp_flow

CAT = "Plugins/Registration"
MODES_3D = ["Translation (3-D phase correlation)", "Rigid (MIP features + z)", "Affine (3-D landmarks)"]
FOCUS_METHODS = ["Sobel energy", "Local variance", "Laplacian energy"]
REGISTRATION3D_KEY = "registration3d"
_XYZ_TO_ZYX = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)


# ============================================================================ phase correlation (n-D)
def _window(shape: tuple[int, ...], alpha: float = 0.5) -> np.ndarray:
    """Separable Tukey window: suppresses the wrap-around / border artefacts of the FFT."""
    w = np.ones(shape, np.float32)
    for ax, n in enumerate(shape):
        if n < 3:
            continue
        t = tukey(n, alpha).astype(np.float32)
        shp = [1] * len(shape)
        shp[ax] = n
        w = w * t.reshape(shp)
    return w


def _zero_mean(a: np.ndarray) -> np.ndarray:
    a = _f32(a)
    return a - float(a.mean())


def _pad_like(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    shp = tuple(max(x, y) for x, y in zip(a.shape, b.shape))
    pa = np.zeros(shp, np.float32)
    pb = np.zeros(shp, np.float32)
    pa[tuple(slice(0, n) for n in a.shape)] = a
    pb[tuple(slice(0, n) for n in b.shape)] = b
    return pa, pb


def _overlap_slices(shape: tuple[int, ...], shift: tuple[float, ...]) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    """Index windows of the reference / moving arrays that overlap for an integer shift (or None)."""
    sa, sb = [], []
    for n, s in zip(shape, shift):
        s = int(round(s))
        lo, hi = max(0, s), min(n, n + s)
        if hi - lo < 2:
            return None
        sa.append(slice(lo, hi))
        sb.append(slice(lo - s, hi - s))
    return tuple(sa), tuple(sb)


def _overlap_ncc_nd(a: np.ndarray, b: np.ndarray, shift: tuple[float, ...]) -> float:
    """Normalised cross-correlation of the overlap of ``a`` and ``b`` shifted by an integer shift."""
    sl = _overlap_slices(a.shape, shift)
    if sl is None:
        return -1.0
    sa = a[sl[0]].astype(np.float64).ravel()
    sb = b[sl[1]].astype(np.float64).ravel()
    if sa.size < 8:
        return -1.0
    sa = sa - sa.mean()
    sb = sb - sb.mean()
    den = math.sqrt(float((sa * sa).sum()) * float((sb * sb).sum()))
    return float((sa * sb).sum() / den) if den > 0 else -1.0


def _integer_shift(a: np.ndarray, b: np.ndarray, max_shift: float = 0.0) -> tuple[float, ...]:
    """Integer shift (per axis) that moves ``b`` onto ``a``: the peaks of the windowed phase
    correlation and of the plain cross-correlation, verified on the real overlap (NCC); the
    search is restricted to ``|shift| <= max_shift`` per axis when given."""
    w = _window(a.shape)
    axes = tuple(range(a.ndim))
    fa = np.fft.rfftn(a * w, axes=axes)
    fb = np.fft.rfftn(b * w, axes=axes)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    cands: list[tuple[float, ...]] = []
    for norm in ("phase", None):
        c = cross / np.where(mag > 1e-12, mag, 1.0) if norm else cross
        corr = np.fft.irfftn(c, s=a.shape, axes=axes)
        if max_shift and max_shift > 0:
            for ax, n in enumerate(a.shape):
                idx = np.arange(n)
                signed = np.where(idx <= n // 2, idx, idx - n)
                ok = np.abs(signed) <= max_shift
                shp = [1] * a.ndim
                shp[ax] = n
                corr = np.where(ok.reshape(shp), corr, -np.inf)
        if not np.any(np.isfinite(corr)):
            continue
        peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
        cands.append(tuple(float(p if p <= n // 2 else p - n) for p, n in zip(peak, a.shape)))
    cands = list(dict.fromkeys(cands))
    if not cands:
        return tuple(0.0 for _ in a.shape)
    if len(cands) == 1:
        return cands[0]
    scores = [_overlap_ncc_nd(a, b, s) for s in cands]
    return cands[int(np.argmax(scores))]


def _subvoxel_residual(a: np.ndarray, b: np.ndarray, shift: tuple[float, ...], upsample: int) -> np.ndarray:
    """Sub-voxel residual of an integer shift: upsampled (plain, windowed) cross-correlation of the
    overlapping windows only; a residual beyond ±1.5 is a spurious peak and is discarded."""
    up = max(int(upsample), 1)
    if up <= 1:
        return np.zeros(a.ndim)
    sl = _overlap_slices(a.shape, shift)
    if sl is None:
        return np.zeros(a.ndim)
    ca, cb = a[sl[0]], b[sl[1]]
    if min(ca.shape) < 3 or ca.size < 16:
        return np.zeros(a.ndim)
    ca = _zero_mean(ca)
    cb = _zero_mean(cb)
    w = _window(ca.shape)
    try:
        r = phase_cross_correlation(ca * w, cb * w, upsample_factor=up, normalization=None)[0]
    except Exception:
        return np.zeros(a.ndim)
    r = np.asarray(r, np.float64)
    if not np.all(np.isfinite(r)) or np.any(np.abs(r) > 1.5):
        return np.zeros(a.ndim)
    return r


def phase_shift_nd(ref: np.ndarray, mov: np.ndarray, upsample: int = 10, max_shift: float = 0.0) -> np.ndarray:
    """Shift (per axis, sub-voxel) that moves ``mov`` onto ``ref`` (``ndi.shift(mov, shift) ≈ ref``).

    Two stages: integer peak (windowed phase correlation vs. plain cross-correlation, the one
    whose overlap correlates best wins; optionally restricted to ``±max_shift``), then the
    sub-voxel residual from an upsampled cross-correlation of the overlapping windows.
    Works for 1-D profiles, 2-D planes and 3-D volumes alike.
    """
    a, b = _pad_like(_f32(ref), _f32(mov))
    ndim = a.ndim
    keep = tuple(i for i, n in enumerate(a.shape) if n > 1)  # singleton axes (a single slice) carry no shift
    out = np.zeros(ndim)
    if not keep:
        return out
    a = _zero_mean(np.squeeze(a))
    b = _zero_mean(np.squeeze(b))
    if not a.any() or not b.any():
        return out
    s = _integer_shift(a, b, max_shift)
    est = np.asarray(s, np.float64) + _subvoxel_residual(a, b, s, upsample)
    if max_shift and max_shift > 0:
        est = np.clip(est, -float(max_shift), float(max_shift))
    out[list(keep)] = est
    return out


# ============================================================================ volumes / matrices
def _volume(img: LumenImage, t: int, c: int) -> np.ndarray:
    return _f32(img.data[t, :, c])


def _voxel_size(img: LumenImage) -> tuple[float, float, float]:
    """(sx, sy, sz) calibrated voxel size (1 where unknown)."""
    cal = img.calibration
    sx = float(cal.pixel_width) if cal.pixel_width and cal.pixel_width > 0 else 1.0
    sy = float(cal.pixel_height) if cal.pixel_height and cal.pixel_height > 0 else sx
    sz = float(cal.pixel_depth) if cal.pixel_depth and cal.pixel_depth > 0 else sx
    return sx, sy, sz


def _shift_volume(vol: np.ndarray, shift: np.ndarray, order: int) -> np.ndarray:
    dz, dy, dx = (float(v) for v in shift)
    if abs(dz) < 1e-9 and abs(dy) < 1e-9 and abs(dx) < 1e-9:
        return vol.astype(np.float32, copy=True)
    if abs(dz) < 1e-9:  # pure in-plane shift: no interpolation across slices
        out = np.empty_like(vol, dtype=np.float32)
        for z in range(vol.shape[0]):
            out[z] = ndi.shift(vol[z], (dy, dx), order=order, mode="constant", cval=0.0)
        return out
    return ndi.shift(vol, (dz, dy, dx), order=order, mode="constant", cval=0.0).astype(np.float32)


def _matrix_xyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = a
    m[:3, 3] = b
    return m


def _translation_xyz(dx: float, dy: float, dz: float) -> np.ndarray:
    return _matrix_xyz(np.eye(3), np.array([dx, dy, dz], dtype=np.float64))


def _rigid_xyz(m2: np.ndarray, dz: float, rdy: float = 0.0, rdx: float = 0.0) -> np.ndarray:
    """In-plane 3×3 (x, y) matrix + z translation (+ residual xy) → 4×4 (x, y, z)."""
    a = np.eye(3)
    a[:2, :2] = m2[:2, :2]
    b = np.array([m2[0, 2] + rdx, m2[1, 2] + rdy, dz], dtype=np.float64)
    return _matrix_xyz(a, b)


def _apply_matrix_xyz(m: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    return pts @ m[:3, :3].T + m[:3, 3]


def _warp_volume(vol: np.ndarray, m_xyz: np.ndarray, out_shape: tuple[int, int, int], order: int) -> np.ndarray:
    """Apply a (moving → reference, x/y/z) 4×4 matrix to a (Z, Y, X) volume; result in the reference grid."""
    a_zyx = _XYZ_TO_ZYX @ m_xyz[:3, :3] @ _XYZ_TO_ZYX
    b_zyx = _XYZ_TO_ZYX @ m_xyz[:3, 3]
    if np.allclose(a_zyx, np.eye(3), atol=1e-12):
        if tuple(vol.shape) == tuple(out_shape):
            return _shift_volume(vol, b_zyx, order)
        canvas = np.zeros(tuple(max(a, b) for a, b in zip(vol.shape, out_shape)), np.float32)
        canvas[tuple(slice(0, n) for n in vol.shape)] = vol
        return _shift_volume(canvas, b_zyx, order)[tuple(slice(0, n) for n in out_shape)]
    inv = np.linalg.inv(a_zyx)
    return ndi.affine_transform(_f32(vol), inv, offset=-inv @ b_zyx, output_shape=tuple(out_shape), order=order, mode="constant", cval=0.0).astype(np.float32)


def _decompose3(m: np.ndarray, img: LumenImage) -> dict[str, float]:
    """dx, dy, dz (voxels; plus calibrated when scaled), in-plane angle (°) and isotropic scale."""
    out = {"dx": float(m[0, 3]), "dy": float(m[1, 3]), "dz": float(m[2, 3]),
           "Angle": float(math.degrees(math.atan2(m[1, 0], m[0, 0]))), "Scale": float(abs(np.linalg.det(m[:3, :3])) ** (1.0 / 3.0))}
    cal = img.calibration
    if cal.scaled:
        sx, sy, sz = _voxel_size(img)
        u = cal.unit
        out[f"dx [{u}]"] = out["dx"] * sx
        out[f"dy [{u}]"] = out["dy"] * sy
        out[f"dz [{u}]"] = out["dz"] * sz
    return out


def _shift_row(label: str, index_name: str, k: int, shift: np.ndarray, img: LumenImage) -> dict[str, Any]:
    """Alignment-table row of a (dz, dy, dx) voxel shift (plus calibrated columns when scaled)."""
    row: dict[str, Any] = {"Label": label, index_name: k, "dx": float(shift[2]), "dy": float(shift[1]), "dz": float(shift[0])}
    cal = img.calibration
    if cal.scaled:
        sx, sy, sz = _voxel_size(img)
        row[f"dx [{cal.unit}]"] = float(shift[2]) * sx
        row[f"dy [{cal.unit}]"] = float(shift[1]) * sy
        row[f"dz [{cal.unit}]"] = float(shift[0]) * sz
    return row


# ============================================================================ landmarks
def parse_points3(text: str) -> np.ndarray:
    """'x1,y1,z1; x2,y2,z2; …' (any separators) → (N, 3) float array; z is the 1-based slice number."""
    s = str(text or "").strip()
    if not s:
        return np.zeros((0, 3))
    nums = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)]
    if len(nums) % 3:
        raise ValueError("3-D landmarks need triples 'x,y,z; x,y,z; …' (z = slice number)")
    return np.asarray(nums, float).reshape(-1, 3)


def _manager_landmarks(ctx, img: LumenImage) -> np.ndarray:
    """(N, 3) voxel-centre coordinates (x, y, z) of the point ROIs of ``img`` in the ROI Manager, in manager order."""
    pts = []
    for r in ctx.session.roi_manager.rois:
        if r.get("type") != "point" or r.get("image_id") != img.id:
            continue
        pos = r.get("position") or {}
        z = float(pos.get("z", img.display.z))
        for p in r.get("points", []):
            pts.append([float(p[0]) - 0.5, float(p[1]) - 0.5, z])
    return np.asarray(pts, float).reshape(-1, 3)


def _landmarks(ctx, img: LumenImage, text: str, what: str) -> np.ndarray:
    pts = parse_points3(text)
    if pts.size:
        pts = pts.copy()
        pts[:, :2] -= 0.5  # Lumen ROI coordinates → pixel centres
        pts[:, 2] -= 1.0   # slice numbers → slice indices
        return pts
    pts = _manager_landmarks(ctx, img)
    if pts.size == 0:
        raise ValueError(f"No landmarks on the {what} image {img.name!r}: add point ROIs (with their slice position) to the ROI Manager or give {what}_points 'x,y,z; …'")
    return pts


def fit_affine_3d(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares 4×4 affine (x, y, z) mapping src points onto dst points (≥ 4 non-coplanar pairs)."""
    src = np.asarray(src, float).reshape(-1, 3)
    dst = np.asarray(dst, float).reshape(-1, 3)
    n = len(src)
    if n < 4:
        raise ValueError(f"A 3-D affine needs at least 4 landmark pairs ({n} given)")
    A = np.c_[src, np.ones(n)]
    sol, _, rank, _ = np.linalg.lstsq(A, dst, rcond=None)
    if rank < 4:
        raise ValueError("The landmarks are coplanar (or collinear) – a 3-D affine needs points spread over x, y and z")
    m = np.eye(4)
    m[:3, :3] = sol[:3].T
    m[:3, 3] = sol[3]
    if not np.all(np.isfinite(m)):
        raise ValueError("Degenerate landmark configuration")
    return m


def _masked_ncc(ref: np.ndarray, mov: np.ndarray, m2: np.ndarray) -> float:
    """Normalised cross-correlation of ``ref`` and ``mov`` warped by the (x, y) matrix, over the warped-in pixels only."""
    w = _warp(mov, m2, 1)
    valid = _warp(np.ones_like(mov, dtype=np.float32), m2, 1) > 0.99
    if valid.sum() < 16:
        return -1.0
    a = ref[valid].astype(np.float64)
    b = w[valid].astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else -1.0


def _rigid_mip_estimate(ref: np.ndarray, mov: np.ndarray, upsample: int) -> tuple[np.ndarray, bool, float]:
    """(4×4 x/y/z matrix moving → reference, used_features, dz): in-plane rigid transform from
    ORB + RANSAC (+ intensity refinement) on the XY maximum projections, then the z shift from
    a 1-D phase correlation of the per-slice intensity profiles (verified against the XZ / YZ
    projections and refined in 3-D)."""
    hh, ww = max(ref.shape[1], mov.shape[1]), max(ref.shape[2], mov.shape[2])

    def padded(p: np.ndarray) -> np.ndarray:
        out = np.zeros((hh, ww), np.float32)
        out[:p.shape[0], :p.shape[1]] = p
        return out

    mip_r, mip_m = padded(ref.max(axis=0)), padded(mov.max(axis=0))
    dy, dx = _phase_shift(mip_r, mip_m, upsample)
    m2 = _translation(dx, dy)
    feat = False
    m_feat = _feature_matrix(mip_r, mip_m, "Rigid body")
    if m_feat is not None:
        # symmetric features (blobs, nuclei) can give a consistent but wrong match set: keep the
        # feature-based model only when it explains the projections at least as well as the pure shift
        if _masked_ncc(mip_r, mip_m, m_feat) >= _masked_ncc(mip_r, mip_m, m2) - 0.01:
            m2, feat = m_feat, True
    # warp every slice in-plane, then estimate z
    nz_r, nz_m = ref.shape[0], mov.shape[0]
    nz = max(nz_r, nz_m)
    mov_w = np.zeros((nz, hh, ww), np.float32)
    for z in range(nz_m):
        mov_w[z] = _warp(padded(mov[z]), m2, 1)
    ref_p = np.zeros((nz, hh, ww), np.float32)
    ref_p[:nz_r, :ref.shape[1], :ref.shape[2]] = ref
    if nz < 2:  # single planes: nothing to estimate along z
        resid = _subvoxel_residual(_zero_mean(ref_p[0]), _zero_mean(mov_w[0]), (0, 0), upsample)
        return _rigid_xyz(m2, 0.0, float(resid[0]), float(resid[1])), feat, 0.0
    prof_r = ref_p.reshape(nz, -1).sum(axis=1)
    prof_m = mov_w.reshape(nz, -1).sum(axis=1)
    cands = [float(phase_shift_nd(prof_r, prof_m, upsample)[0])]
    if hh > 2 and ww > 2:
        cands.append(float(phase_shift_nd(ref_p.max(axis=1), mov_w.max(axis=1), upsample)[0]))
        cands.append(float(phase_shift_nd(ref_p.max(axis=2), mov_w.max(axis=2), upsample)[0]))
    a, b = _zero_mean(ref_p), _zero_mean(mov_w)
    uniq = list(dict.fromkeys(int(round(c)) for c in cands))
    dz_int = max(uniq, key=lambda c: _overlap_ncc_nd(a, b, (c, 0, 0))) if len(uniq) > 1 else uniq[0]
    resid = _subvoxel_residual(a, b, (dz_int, 0, 0), upsample)
    dz = dz_int + float(resid[0])
    return _rigid_xyz(m2, dz, float(resid[1]), float(resid[2])), feat, dz


def _result_image(ctx, img: LumenImage, out: np.ndarray, create_new: bool, name: str, calibration: Calibration | None, table: dict[str, Any] | None,
                  rows: list[dict[str, Any]] | None, msg: str) -> OpResult:
    data = cast_to(out, img.data.dtype)
    if create_new:
        res = img.copy(name=name, data=data)
        if calibration is not None:
            res.calibration = Calibration(**calibration.to_dict())
        res.overlay = []
        res.sync_channels()
        return OpResult(new_images=[res], table=table, results=rows, message=msg, log=msg)
    return OpResult(data=data, rgb=img.rgb, table=table, results=rows, calibration=calibration.to_dict() if calibration is not None else None,
                    clear_roi=True, message=msg, log=msg)


# ================================================================================ ops
@op("registration.register_3d", "Register 3D Volume to Image", CAT, kind="image", preview=False, scope_param=False, roi_aware=False, supports_rgb=False,
    params=[Param("target", "image", None, help="The open 3-D image to align the current volume onto"),
            Param("mode", "choice", MODES_3D[0], choices=MODES_3D, label="Transformation",
                  help="Translation: 3-D phase correlation. Rigid: in-plane rotation + shift from features on the XY maximum projections, z from a 1-D phase correlation. Affine: least squares over ≥ 4 landmark pairs (point ROIs with slice positions on both images, or the point lists)"),
            Param("channel", "int", 0, min=0, help="Channel used for the estimate (0 = current); applied to all channels and frames"),
            Param("upsample", "int", 10, min=1, max=100, label="Sub-voxel factor"),
            Param("source_points", "text", "", help="Affine: landmarks on the current image 'x,y,z; x,y,z; …' (z = slice number; empty = its point ROIs in the ROI Manager)"),
            Param("target_points", "text", "", help="Affine: the matching landmarks on the target image (same order)"),
            Param("interpolation", "choice", "Linear", choices=INTERP), Param("create_new", "bool", True, label="Create new image")],
    description="Descriptor-style 3-D registration of the current volume (current frame) onto another open 3-D image: 3-D translation by phase correlation, rigid (in-plane rotation and shift from features on the XY projections + z shift) or a 3-D affine from landmark pairs fitted in calibrated space. The transform is applied to every channel and frame; the result has the target's volume size. dx/dy/dz, angle and scale go to the Results table.",
    keywords=["descriptor-based registration", "3d registration", "register volume", "rigid 3d", "affine 3d", "landmarks 3d", "phase correlation 3d"])
def register_3d(ctx, target=None, mode=MODES_3D[0], channel=0, upsample=10, source_points="", target_points="", interpolation="Linear", create_new=True):
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
    ref = _volume(tgt, min(tgt.display.t, tgt.nt - 1), ct)
    mov = _volume(img, img.display.t, c)
    feat, n_land, rms = False, 0, None
    if mode.startswith("Affine"):
        src = _landmarks(ctx, img, source_points, "source")
        dst = _landmarks(ctx, tgt, target_points, "target")
        if len(src) != len(dst):
            raise ValueError(f"{len(src)} source landmarks but {len(dst)} target landmarks – they must correspond one to one")
        s_src, s_dst = np.array(_voxel_size(img)), np.array(_voxel_size(tgt))
        m_cal = fit_affine_3d(src * s_src, dst * s_dst)  # fitted in calibrated (isotropic) space
        s_in, s_out = np.diag(s_src), np.diag(1.0 / s_dst)
        m = np.eye(4)
        m[:3, :3] = s_out @ m_cal[:3, :3] @ s_in
        m[:3, 3] = s_out @ m_cal[:3, 3]
        resid = np.linalg.norm((_apply_matrix_xyz(m_cal, src * s_src) - dst * s_dst), axis=1)
        rms, n_land = float(np.sqrt(np.mean(resid ** 2))), int(len(src))
    elif mode.startswith("Rigid"):
        m, feat, _ = _rigid_mip_estimate(ref, mov, upsample)
    else:
        dz, dy, dx = phase_shift_nd(ref, mov, upsample)
        m = _translation_xyz(dx, dy, dz)
    out_shape = (tgt.nz, tgt.height, tgt.width)
    out = np.zeros((img.nt, out_shape[0], img.nc, out_shape[1], out_shape[2]), np.float32)
    for t in range(img.nt):
        for cc in range(img.nc):
            out[t, :, cc] = _warp_volume(_volume(img, t, cc), m, out_shape, order)
    info = _decompose3(m, tgt)
    row: dict[str, Any] = {"Label": f"{img.name} → {tgt.name}", "Mode": mode.split(" (")[0], **info}
    if mode.startswith("Affine"):
        row["N landmarks"] = n_land
        row[f"RMS residual [{tgt.calibration.unit}]" if tgt.calibration.scaled else "RMS residual"] = rms
    ctx.session.settings[REGISTRATION3D_KEY] = {"matrix": m.tolist(), "source": img.id, "target": tgt.id, "mode": mode}
    msg = f"Registered {img.name} onto {tgt.name} ({mode.split(' (')[0].lower()}): dx {info['dx']:+.2f}, dy {info['dy']:+.2f}, dz {info['dz']:+.2f} voxels, angle {info['Angle']:+.2f}°, scale {info['Scale']:.3f}"
    if mode.startswith("Rigid") and not feat:
        msg += " (no features on the projections: translation only)"
    if mode.startswith("Affine"):
        msg += f"; {n_land} landmarks, RMS residual {rms:.2f}"
    if abs(_voxel_size(img)[2] / _voxel_size(tgt)[2] - 1) > 0.01 and mode.startswith(("Translation", "Rigid")):
        msg += " – note: the two images have different z spacings; shifts are in the target's voxels"
    return _result_image(ctx, img, out, create_new, f"{img.name} on {tgt.name}", tgt.calibration, None, [row], msg)


@op("registration.align_time_3d", "Correct 3D Drift (phase correlation)", CAT, kind="image", preview=False, scope_param=False, roi_aware=False, supports_rgb=False,
    params=[Param("channel", "int", 0, min=0, help="Channel used to estimate the drift (0 = current); applied to all channels"),
            Param("reference", "choice", "Previous frame", choices=["Previous frame", "First frame"]),
            Param("upsample", "int", 10, min=1, max=100, label="Sub-voxel factor", help="Phase-correlation upsampling (1 = whole voxels)"),
            Param("max_shift", "float", 0.0, min=0, step=1, label="Max shift (voxels)", help="Largest drift between the compared frames per axis (0 = unlimited)"),
            Param("interpolation", "choice", "Linear", choices=INTERP),
            Param("crop_to_common", "bool", False, label="Crop to common region", help="Keep only the region covered by every frame after the correction (no black borders)"),
            Param("create_new", "bool", False, label="Create new image")],
    description="Correct 3D drift: the 3-D translation of every frame against the previous (or first) frame by 3-D phase correlation on one channel, sub-voxel, applied to all channels; optionally cropped to the region valid in every frame. The per-frame shifts go to the 'Alignment' table.",
    keywords=["correct 3d drift", "drift", "time lapse", "3d translation", "register frames"])
def align_time_3d(ctx, channel=0, reference="Previous frame", upsample=10, max_shift=0.0, interpolation="Linear", crop_to_common=False, create_new=False):
    img = ctx.img
    if img.nt < 2:
        raise ValueError("Needs a time series (T > 1)")
    c = _channel_index(img, channel)
    order = ORDER_MAP[interpolation]
    shifts = np.zeros((img.nt, 3))  # dz, dy, dx
    ref = _volume(img, 0, c)
    for t in range(1, img.nt):
        cur = _volume(img, t, c)
        s = phase_shift_nd(ref, cur, upsample, max_shift)
        if reference == "Previous frame":
            shifts[t] = shifts[t - 1] + s
            ref = cur
        else:
            shifts[t] = s
    out = np.empty(img.shape, np.float32)
    for t in range(img.nt):
        for cc in range(img.nc):
            out[t, :, cc] = _shift_volume(_volume(img, t, cc), shifts[t], order)
    crop_msg = ""
    if crop_to_common:
        lo, hi = [], []
        for ax, n in enumerate((img.nz, img.height, img.width)):
            lo.append(max(0, int(math.ceil(shifts[:, ax].max() - 1e-6))))
            hi.append(min(n, int(math.floor(n - 1 + shifts[:, ax].min() + 1e-6)) + 1))
        if any(h - l_ < 1 for l_, h in zip(lo, hi)):
            raise ValueError("The drift is larger than the image – no region is covered by every frame")
        out = np.ascontiguousarray(out[:, lo[0]:hi[0], :, lo[1]:hi[1], lo[2]:hi[2]])
        crop_msg = f"; cropped to {hi[2] - lo[2]} × {hi[1] - lo[1]} × {hi[0] - lo[0]}"
    rows = [_shift_row(f"{img.name}:t{t + 1}", "Frame", t + 1, shifts[t], img) for t in range(img.nt)]
    dmax = float(np.max(np.hypot(shifts[:, 1], shifts[:, 2])))
    msg = f"3D drift corrected over {img.nt} frames (ref: {reference.lower()}; max xy drift {dmax:.2f} px, max z drift {np.abs(shifts[:, 0]).max():.2f} slices){crop_msg}"
    return _result_image(ctx, img, out, create_new, f"Drift corrected {img.name}", None, _alignment_table(rows), None, msg)


@op("registration.multi_channel_3d", "Align Channels (3D)", CAT, kind="image", preview=False, scope_param=False, roi_aware=False, supports_rgb=False,
    params=[Param("reference_channel", "int", 1, min=1),
            Param("estimate_on", "choice", "Current frame", choices=["Current frame", "All frames (mean shift)"]),
            Param("upsample", "int", 10, min=1, max=100, label="Sub-voxel factor"),
            Param("max_shift", "float", 0.0, min=0, step=1, label="Max shift (voxels)", help="Largest chromatic shift per axis (0 = unlimited)"),
            Param("interpolation", "choice", "Linear", choices=INTERP), Param("create_new", "bool", False, label="Create new image")],
    description="3-D chromatic-shift correction: the (dx, dy, dz) shift of every channel's volume relative to a reference channel by 3-D phase correlation (including the axial chromatic shift), applied to the whole channel in every frame. The per-channel shifts go to the 'Alignment' table.",
    keywords=["chromatic aberration", "chromatic shift z", "channel shift 3d", "axial chromatic"])
def multi_channel_3d(ctx, reference_channel=1, estimate_on="Current frame", upsample=10, max_shift=0.0, interpolation="Linear", create_new=False):
    img = ctx.img
    if img.nc < 2:
        raise ValueError("Needs at least two channels")
    r = _channel_index(img, reference_channel)
    order = ORDER_MAP[interpolation]
    frames = list(range(img.nt)) if estimate_on.startswith("All") else [img.display.t]
    shifts = np.zeros((img.nc, 3))
    for c in range(img.nc):
        if c == r:
            continue
        est = [phase_shift_nd(_volume(img, t, r), _volume(img, t, c), upsample, max_shift) for t in frames]
        shifts[c] = np.mean(est, axis=0)
    out = img.data.astype(np.float32, copy=True)
    for c in range(img.nc):
        if c == r or np.all(np.abs(shifts[c]) < 1e-9):
            continue
        for t in range(img.nt):
            out[t, :, c] = _shift_volume(_volume(img, t, c), shifts[c], order)
    rows = [_shift_row(f"{img.name}:C{c + 1}", "Channel", c + 1, shifts[c], img) for c in range(img.nc)]
    msg = "3-D channel shifts: " + ", ".join(f"C{c + 1} ({shifts[c, 2]:+.2f}, {shifts[c, 1]:+.2f}, {shifts[c, 0]:+.2f})" for c in range(img.nc) if c != r)
    return _result_image(ctx, img, out, create_new, f"Aligned {img.name}", None, _alignment_table(rows), None, msg)


# ================================================================================ extended depth of field
def focus_measure(plane: np.ndarray, method: str, kernel_size: int) -> np.ndarray:
    """Per-pixel sharpness of a 2-D plane: Sobel gradient energy, local variance or Laplacian energy, averaged over a kernel_size² window."""
    p = _f32(plane).astype(np.float64)
    k = max(int(kernel_size), 1)
    if method == "Local variance":
        mu = ndi.uniform_filter(p, k, mode="reflect")
        return np.maximum(ndi.uniform_filter(p * p, k, mode="reflect") - mu * mu, 0.0).astype(np.float32)
    if method == "Laplacian energy":
        e = ndi.laplace(p, mode="reflect") ** 2
    else:
        e = ndi.sobel(p, axis=0, mode="reflect") ** 2 + ndi.sobel(p, axis=1, mode="reflect") ** 2
    return ndi.uniform_filter(e, k, mode="reflect").astype(np.float32)


@op("registration.stack_focus", "Extended Depth of Field (Stack Focuser)", CAT, kind="analysis", preview=False, roi_aware=False, supports_rgb=False,
    params=[Param("method", "choice", "Sobel energy", choices=FOCUS_METHODS, label="Focus measure"),
            Param("kernel_size", "int", 11, min=3, max=201, help="Side of the window over which the focus measure is averaged (px)"),
            Param("channel", "int", 0, min=0, help="Channel the focus measure is computed on (0 = current); every channel is composed with the same height map"),
            Param("smooth_height", "int", 0, min=0, max=50, label="Smooth height map (px)", help="Median-filter radius applied to the height map before composing (0 = none)"),
            Param("height_map", "bool", True, label="Output height map", help="Also create the 32-bit map of the best-focus slice number per pixel")],
    description="Extended depth of field (Fiji Stack Focuser / EDF): for every pixel the sharpest slice of the Z stack (highest local Sobel energy, variance or Laplacian energy) is chosen and its value copied into a single all-in-focus 2-D image; the slice numbers form a 32-bit height map (topography).",
    keywords=["stack focuser", "edf", "extended depth of field", "all in focus", "focus stacking", "height map", "topography"])
def stack_focus(ctx, method="Sobel energy", kernel_size=11, channel=0, smooth_height=0, height_map=True):
    img = ctx.img
    if img.nz < 2:
        raise ValueError("Needs a Z stack (Z > 1)")
    c = _channel_index(img, channel)
    k = int(kernel_size) | 1  # odd
    focused = np.zeros((img.nt, 1, img.nc, img.height, img.width), np.float32)
    heights = np.zeros((img.nt, 1, 1, img.height, img.width), np.float32)
    for t in range(img.nt):
        fm = np.stack([focus_measure(img.data[t, z, c], method, k) for z in range(img.nz)])
        best = np.argmax(fm, axis=0)
        if smooth_height > 0:
            best = ndi.median_filter(best.astype(np.int32), size=2 * int(smooth_height) + 1, mode="nearest")
        heights[t, 0, 0] = best + 1
        idx = best[None]
        for cc in range(img.nc):
            focused[t, 0, cc] = np.take_along_axis(_f32(img.data[t, :, cc]), idx, axis=0)[0]
    res = img.copy(name=f"Focused {img.name}", data=cast_to(focused, img.data.dtype))
    res.overlay = []
    res.display.z = 0
    res.metadata["edf"] = method
    res.sync_channels()
    new_images = [res]
    if height_map:
        hm = LumenImage(heights, name=f"Height map {img.name}", axes="TZCYX")
        hm.calibration = Calibration(**img.calibration.to_dict())
        hm.display.channels[0].lut = "Fire"
        hm.display.channels[0].min, hm.display.channels[0].max = 1.0, float(img.nz)
        hm.metadata.update({"height_map": True, "unit": "slice"})
        new_images.append(hm)
    rng = (int(heights.min()), int(heights.max()))
    msg = f"Extended depth of field ({method.lower()}, {k} px window) over {img.nz} slices; best-focus slices {rng[0]}–{rng[1]}"
    return OpResult(new_images=new_images, select_image=res.id, message=msg, log=msg)


# ================================================================================ DoG detections (n-D)
def detect_dog_nd(vol: np.ndarray, sigma: float | tuple[float, ...], objects: str = "Bright", threshold: float = 0.05, max_detections: int = 500,
                  min_distance: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Difference-of-Gaussian blob detections of a 2-D plane or 3-D volume (the interest points of Fiji's
    Descriptor-based registration).

    ``sigma`` (one value or one per array axis, in voxels) is the inner Gaussian; the outer one is 1.6× wider.
    ``objects``: 'Bright' (DoG maxima), 'Dark' (minima) or 'Bright and dark'. A detection must be a local maximum
    (within ``min_distance``) and reach ``threshold`` × the strongest response. Returns (points (N, D) sub-voxel in
    **(x, y[, z]) pixel-centre coordinates**, strongest first, and their responses)."""
    a = _f32(vol).astype(np.float64)
    D = a.ndim
    sig = np.broadcast_to(np.asarray(sigma, float), (D,)).astype(float)
    sig = np.maximum(sig, 0.3)
    dog = ndi.gaussian_filter(a, sig) - ndi.gaussian_filter(a, sig * 1.6)
    if objects == "Dark":
        dog = -dog
    elif objects.startswith("Bright and"):
        dog = np.abs(dog)
    empty = np.zeros((0, D)), np.zeros(0)
    peak = float(dog.max()) if dog.size else 0.0
    if peak <= 0:
        return empty
    size = 2 * max(int(min_distance), 1) + 1
    cand = (dog == ndi.maximum_filter(dog, size=size, mode="nearest")) & (dog >= float(threshold) * peak) & (dog > 0)
    for ax in range(D):  # no border extrema (the parabolic fit needs both neighbours)
        sl = [slice(None)] * D
        sl[ax] = slice(0, 1)
        cand[tuple(sl)] = False
        sl[ax] = slice(-1, None)
        cand[tuple(sl)] = False
    idx = np.argwhere(cand)
    if len(idx) == 0:
        return empty
    resp = dog[tuple(idx.T)]
    order = np.argsort(-resp)[:max(int(max_detections), 1)]
    idx, resp = idx[order], resp[order]
    pts = idx.astype(float)
    for ax in range(D):  # sub-voxel refinement: parabola through the three samples along every axis
        lo, hi = idx.copy(), idx.copy()
        lo[:, ax] -= 1
        hi[:, ax] += 1
        vm, v0, vp = dog[tuple(lo.T)], resp, dog[tuple(hi.T)]
        den = vm - 2.0 * v0 + vp
        with np.errstate(divide="ignore", invalid="ignore"):
            off = np.where(np.abs(den) > 1e-12, 0.5 * (vm - vp) / den, 0.0)
        pts[:, ax] += np.clip(np.nan_to_num(off), -0.5, 0.5)
    return pts[:, ::-1], resp


# ================================================================================ block matching
def _parabolic_offset(cm: float, c0: float, cp: float) -> float:
    den = cm - 2.0 * c0 + cp
    if den >= 0:
        return 0.0
    return float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))


def block_match(src: np.ndarray, tgt: np.ndarray, block_size: int, search_radius: int, grid_spacing: int) -> list[dict[str, float]]:
    """Sparse displacement field: for every grid block of ``src`` the (dy, dx) with the best normalised
    cross-correlation in ``tgt`` within ±search_radius, sub-pixel by parabolic peak fit.
    Rows: y, x (block centre, pixel-centre coordinates), dy, dx, r."""
    from skimage.feature import match_template
    a, b = _f32(src), _f32(tgt)
    if a.shape != b.shape:
        raise ValueError(f"Images differ in size: {a.shape[1]} × {a.shape[0]} vs {b.shape[1]} × {b.shape[0]}")
    H, W = a.shape
    bs = max(int(block_size), 3)
    r = max(int(search_radius), 1)
    g = max(int(grid_spacing), 1)
    half = bs // 2
    margin = half + r
    ys = list(range(margin, H - (bs - half) - r + 1, g))
    xs = list(range(margin, W - (bs - half) - r + 1, g))
    if not ys or not xs:
        raise ValueError(f"The image ({W} × {H}) is too small for blocks of {bs} px with a search radius of {r} px")
    rows = []
    for y in ys:
        for x in xs:
            block = a[y - half:y - half + bs, x - half:x - half + bs]
            win = b[y - half - r:y - half + bs + r, x - half - r:x - half + bs + r]
            if block.std() <= 1e-6 or win.std() <= 1e-6:
                rows.append({"y": y + 0.0, "x": x + 0.0, "dy": 0.0, "dx": 0.0, "r": -1.0})
                continue
            corr = match_template(win, block, pad_input=False)  # (2r+1, 2r+1)
            corr = np.nan_to_num(corr, nan=-1.0)
            iy, ix = np.unravel_index(int(np.argmax(corr)), corr.shape)
            best = float(corr[iy, ix])
            oy = _parabolic_offset(corr[iy - 1, ix], best, corr[iy + 1, ix]) if 0 < iy < corr.shape[0] - 1 else 0.0
            ox = _parabolic_offset(corr[iy, ix - 1], best, corr[iy, ix + 1]) if 0 < ix < corr.shape[1] - 1 else 0.0
            rows.append({"y": y + 0.0, "x": x + 0.0, "dy": float(iy - r + oy), "dx": float(ix - r + ox), "r": best})
    return rows


def dense_field(rows: list[dict[str, float]], shape: tuple[int, int], smoothing: float = 0.0) -> np.ndarray:
    """(2, H, W) float32 flow [v, u] in the registration_extra convention (``moving[r + v, c + u] ≈ reference[r, c]``)
    interpolated from the matched blocks: the value −d is placed at the matched position (block + d) in the reference."""
    from scipy.interpolate import griddata
    H, W = shape
    pts = np.array([[rw["y"] + rw["dy"], rw["x"] + rw["dx"]] for rw in rows], float)
    vals = np.array([[-rw["dy"], -rw["dx"]] for rw in rows], float)
    rr, cc = np.mgrid[0:H, 0:W]
    flow = np.zeros((2, H, W), np.float32)
    if len(rows) == 0:
        return flow
    for k in range(2):
        lin = np.full((H, W), np.nan)
        if len(rows) >= 4:
            try:
                lin = griddata(pts, vals[:, k], (rr, cc), method="linear")
            except Exception:  # collinear block centres (a single row / column of blocks): nearest only
                pass
        near = griddata(pts, vals[:, k], (rr, cc), method="nearest")
        f = np.where(np.isfinite(lin), lin, near)
        if smoothing and smoothing > 0:
            f = ndi.gaussian_filter(f, float(smoothing))
        flow[k] = f
    return flow


@op("registration.block_matching_2d", "Block Matching (sparse displacement field)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("target", "image", None, help="The open image the current image's blocks are searched in"),
            Param("block_size", "int", 32, min=4, max=512, help="Side of the matched blocks (px)"),
            Param("search_radius", "int", 8, min=1, max=256, help="Largest displacement searched per axis (px)"),
            Param("grid_spacing", "int", 0, min=0, help="Distance between block centres (0 = block size)"),
            Param("min_correlation", "float", 0.5, min=-1, max=1, step=0.05, help="Blocks with a lower normalised cross-correlation are discarded"),
            Param("arrows", "bool", True, label="Arrow overlay", help="Draw the displacement of every used block as an arrow on the current image"),
            Param("vector_scale", "float", 1.0, min=0.1, max=50, step=0.5, help="Arrow length factor"), Param("color", "color", "#00ffff"),
            Param("warp", "bool", False, label="Dense warp", help="Interpolate the sparse field into a dense deformation and register the current image onto the target (new image)"),
            Param("smoothing", "float", 0.0, min=0, max=100, step=1, help="Gaussian smoothing of the dense field (px, 0 = none)"),
            Param("interpolation", "choice", "Linear", choices=INTERP),
            Param("output_field", "bool", False, label="Output deformation field", help="Also create a 2-channel 32-bit image (dx, dy) of the dense field"),
            Param("store", "bool", True, label="Store the field", help="Keep the dense field for 'Apply Deformation' (session)")],
    description="Elastic ▸ block matching: a sparse displacement field between the current image and a target image – every block on a grid is located in the target by normalised cross-correlation within a search radius (sub-pixel by parabolic peak fit). Arrows show the displacements, the 'Block matching' table lists them; optionally the field is interpolated into a dense deformation and used to warp the current image onto the target.",
    keywords=["block matching", "elastic", "displacement field", "cross correlation", "vectors", "piv"])
def block_matching_2d(ctx, target=None, block_size=32, search_radius=8, grid_spacing=0, min_correlation=0.5, arrows=True, vector_scale=1.0, color="#00ffff",
                      warp=False, smoothing=0.0, interpolation="Linear", output_field=False, store=True):
    img = ctx.img
    if not target:
        raise ValueError("Choose a target image")
    tgt = ctx.session.get_image(target)
    if tgt.id == img.id:
        raise ValueError("Target must be a different image")
    d = img.display
    src_plane = img.data[d.t, d.z, min(d.c, img.nc - 1)]
    tgt_plane = tgt.data[min(tgt.display.t, tgt.nt - 1), min(tgt.display.z, tgt.nz - 1), min(d.c, tgt.nc - 1)]
    H, W = max(src_plane.shape[0], tgt_plane.shape[0]), max(src_plane.shape[1], tgt_plane.shape[1])

    def padded(p: np.ndarray) -> np.ndarray:
        out = np.zeros((H, W), np.float32)
        out[:p.shape[0], :p.shape[1]] = _f32(p)
        return out

    g = int(grid_spacing) if grid_spacing > 0 else int(block_size)
    matches = block_match(padded(src_plane), padded(tgt_plane), block_size, search_radius, g)
    used = [m for m in matches if m["r"] >= float(min_correlation)]
    cols = ["Block", "X", "Y", "dx", "dy", "Correlation", "Used"]
    table = {"name": "Block matching", "columns": cols,
             "rows": [[i + 1, m["x"] + 0.5, m["y"] + 0.5, round(m["dx"], 3), round(m["dy"], 3), round(m["r"], 4), m["r"] >= float(min_correlation)] for i, m in enumerate(matches)]}
    overlay = None
    if arrows:
        items = []
        for m in used:
            cx, cy = m["x"] + 0.5, m["y"] + 0.5
            items.append({"id": roimod.new_id(), "type": "arrow", "x1": round(cx, 2), "y1": round(cy, 2), "x2": round(cx + m["dx"] * vector_scale, 2), "y2": round(cy + m["dy"] * vector_scale, 2),
                          "color": color, "stroke_width": 1.0, "head_size": max(3.0, min(g / 3.0, 8.0)), "double_headed": False,
                          "position": {"t": d.t, "z": d.z, "c": d.c}, "block_matching": True})
        overlay = [o for o in img.overlay if not o.get("block_matching")] + items
    new_images = []
    results = None
    if warp or output_field:
        if len(used) < 1:
            raise ValueError("No block matched above the minimum correlation – nothing to warp")
        flow = dense_field(used, (H, W), smoothing)
        ht, wt = tgt.height, tgt.width
        if warp:
            order = ORDER_MAP[interpolation]
            out = np.zeros((img.nt, img.nz, img.nc, ht, wt), np.float32)
            for t, z, c in img.iter_planes():
                out[t, z, c] = warp_flow(padded(img.data[t, z, c]), flow, order)[:ht, :wt]
            res = img.copy(name=f"{img.name} on {tgt.name} (block matching)", data=cast_to(out, img.data.dtype))
            res.calibration = Calibration(**tgt.calibration.to_dict())
            res.overlay = []
            res.sync_channels()
            new_images.append(res)
        field = np.ascontiguousarray(flow[:, :ht, :wt])
        if output_field:
            new_images.append(_field_image(field, f"Deformation field {img.name} → {tgt.name}", like=tgt))
        if store:
            ctx.session.settings[DEFORMATION_KEY] = {"flow": field, "source": img.name, "target": tgt.name, "shape": [ht, wt], "method": "Block matching"}
    if used:
        dxs = np.array([m["dx"] for m in used])
        dys = np.array([m["dy"] for m in used])
        results = [{"Label": f"{img.name} → {tgt.name}", "N blocks": len(matches), "N used": len(used), "Mean dx": float(dxs.mean()), "Mean dy": float(dys.mean()),
                    "Max displacement": float(np.hypot(dxs, dys).max()), "Mean correlation": float(np.mean([m["r"] for m in used]))}]
        msg = f"Block matching {img.name} → {tgt.name}: {len(used)}/{len(matches)} blocks matched (R ≥ {min_correlation:g}); mean displacement ({dxs.mean():+.2f}, {dys.mean():+.2f}) px"
    else:
        msg = f"Block matching {img.name} → {tgt.name}: none of the {len(matches)} blocks reached R ≥ {min_correlation:g}"
    if warp:
        msg += "; warped onto the target"
    return OpResult(new_images=new_images, results=results, table=table, overlay=overlay, message=msg, log=msg)
