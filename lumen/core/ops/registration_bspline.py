"""Plugins ▸ Registration — bUnwarpJ: cubic B-spline free-form deformation registration
(``registration.bunwarpj``) with bUnwarpJ's registration modes (Fast / Accurate / Mono = unidirectional),
coarse-to-fine control-point grids (Very Coarse = 1 interval … Super Fine = 16 intervals), the image (SSD),
divergence / curl, landmark and consistency (bidirectional) energy terms, and Fiji's elastic / raw
transform text files (load / save / apply / invert / compare / convert).

Conventions
-----------
A transform maps *target* pixel coordinates onto *source* coordinates: the registered source is
``source(x + ux(x, y), y + uy(x, y))`` for every target pixel. The B-spline model is bUnwarpJ's:
``intervals + 3`` coefficients per axis; coefficient ``k`` of the x grid sits at
``(k - 1) · (width - 1) / intervals`` and the basis at pixel ``x`` is ``β³(x · intervals / (width - 1) + 1 - k)``.
Elastic transform files store the *absolute* coefficients (identity + displacement) exactly like bUnwarpJ,
so files can be exchanged with Fiji. Deformation-field images are the 2-channel (dx, dy) images of
``registration.apply_deformation`` (``registration_extra``), so a bUnwarpJ field can be re-applied with it.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from skimage.transform import resize

from ..image import Calibration, LumenImage
from ..registry import Param, OpResult, op, cast_to
from .registration import INTERP, ORDER_MAP, _estimation_plane, _f32, _norm01
from .registration_extra import DEFORMATION_KEY, _field_image, _flow_from_image
from .transform_extra import _manager_roi, _points_of_roi, parse_points

CAT = "Plugins/Registration"
BUNWARPJ_KEY = "bunwarpj"
REG_MODES = ["Fast", "Accurate", "Mono"]
LEVELS = ["Very Coarse", "Coarse", "Fine", "Very Fine", "Super Fine"]
ITERATIONS = {"Fast": 40, "Accurate": 150, "Mono": 150}
COMPARE_MODES = ["Opposite directions (warping index)", "Same direction (mean distance)"]
CONVERT_MODES = ["Elastic → raw", "Raw → elastic (B-spline fit)"]


# ============================================================================ B-spline model
def _beta3(t: np.ndarray) -> np.ndarray:
    """Uniform cubic B-spline β³ (support |t| < 2)."""
    a = np.abs(np.asarray(t, np.float64))
    out = np.zeros_like(a)
    m = a < 1
    out[m] = 2.0 / 3.0 - a[m] ** 2 + a[m] ** 3 / 2.0
    m = (a >= 1) & (a < 2)
    out[m] = (2.0 - a[m]) ** 3 / 6.0
    return out


def _beta3_prime(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, np.float64)
    a = np.abs(t)
    s = np.sign(t)
    out = np.zeros_like(a)
    m = a < 1
    out[m] = (-2.0 * a[m] + 1.5 * a[m] ** 2) * s[m]
    m = (a >= 1) & (a < 2)
    out[m] = (-0.5 * (2.0 - a[m]) ** 2) * s[m]
    return out


@dataclass(frozen=True)
class BGrid:
    """bUnwarpJ's control-point layout: ``intervals`` intervals over an image of ``width`` × ``height`` pixels,
    ``intervals + 3`` coefficients per axis (coefficient matrices are indexed ``[y index, x index]``)."""
    intervals: int
    width: int
    height: int

    @property
    def n(self) -> int:
        return self.intervals + 3

    def _t(self, coords: np.ndarray, length: int) -> np.ndarray:
        return np.asarray(coords, np.float64) * self.intervals / max(length - 1, 1) + 1.0

    def basis(self, coords: np.ndarray, length: int, deriv: bool = False) -> np.ndarray:
        """Dense (len(coords), n) basis (or derivative with respect to the pixel coordinate)."""
        t = self._t(coords, length)
        d = t[:, None] - np.arange(self.n)[None, :]
        if deriv:
            return _beta3_prime(d) * (self.intervals / max(length - 1, 1))
        return _beta3(d)

    def bx(self, deriv: bool = False) -> np.ndarray:
        return self.basis(np.arange(self.width), self.width, deriv)

    def by(self, deriv: bool = False) -> np.ndarray:
        return self.basis(np.arange(self.height), self.height, deriv)

    def identity(self) -> tuple[np.ndarray, np.ndarray]:
        """Absolute coefficients of the identity transform (bUnwarpJ's file convention)."""
        k = np.arange(self.n, dtype=np.float64)
        cx = np.tile((k - 1) * (self.width - 1) / self.intervals, (self.n, 1))
        cy = np.tile(((k - 1) * (self.height - 1) / self.intervals)[:, None], (1, self.n))
        return cx, cy

    def field(self, dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dense displacement (ux, uy), each (height, width), from displacement coefficients."""
        bx, by = self.bx(), self.by()
        return by @ dx @ bx.T, by @ dy @ bx.T

    def fit(self, ux: np.ndarray, uy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Least-squares displacement coefficients reproducing the dense fields (exact for B-spline fields)."""
        py, px = np.linalg.pinv(self.by()), np.linalg.pinv(self.bx())
        return py @ np.asarray(ux, np.float64) @ px.T, py @ np.asarray(uy, np.float64) @ px.T


def _scatter_weights(grid: BGrid, px: np.ndarray, py: np.ndarray):
    """16-neighbour basis weights / derivative weights / coefficient indices at scattered points."""
    tx, ty = grid._t(px, grid.width), grid._t(py, grid.height)
    off = np.arange(4)[:, None]
    ix = (np.floor(tx).astype(np.int64) - 1)[None, :] + off
    iy = (np.floor(ty).astype(np.int64) - 1)[None, :] + off
    sx, sy = grid.intervals / max(grid.width - 1, 1), grid.intervals / max(grid.height - 1, 1)
    wx, dwx = _beta3(tx[None, :] - ix), _beta3_prime(tx[None, :] - ix) * sx
    wy, dwy = _beta3(ty[None, :] - iy), _beta3_prime(ty[None, :] - iy) * sy
    return np.clip(iy, 0, grid.n - 1), wy, dwy, np.clip(ix, 0, grid.n - 1), wx, dwx


def _scatter_eval(c: np.ndarray, iy, wy, ix, wx) -> np.ndarray:
    out = np.zeros(ix.shape[1])
    for a in range(4):
        for b in range(4):
            out += wy[a] * wx[b] * c[iy[a], ix[b]]
    return out


def _scatter_adjoint(vals: np.ndarray, iy, wy, ix, wx, shape: tuple[int, int]) -> np.ndarray:
    acc = np.zeros(shape[0] * shape[1])
    for a in range(4):
        for b in range(4):
            acc += np.bincount(iy[a] * shape[1] + ix[b], weights=vals * wy[a] * wx[b], minlength=acc.size)
    return acc.reshape(shape)


# ============================================================================ energy
def _spline_images(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cubic-spline coefficients of an image and of its x / y gradients (for ``_sample``).

    The derivative of the cubic-spline interpolant at a pixel centre is the central difference of its *coefficients*
    (β³'(±1) = ∓½), so the gradient images are exact on the pixel grid and only interpolated in between."""
    c = ndi.spline_filter(np.asarray(img, np.float64), order=3, mode="nearest")
    cp = np.pad(c, 1, mode="edge")  # the 'nearest' extension of the coefficients that map_coordinates uses
    # ∂S/∂x at a pixel centre = Σ_a β³(y − a) Σ_b c_ab β³'(x − b) = the (1, 4, 1)/6 y-smoothed central difference of the
    # coefficients along x (β³'(±1) = ∓½); likewise for ∂S/∂y
    dx = (cp[:, 2:] - cp[:, :-2]) / 2.0
    gx = (dx[:-2] + 4.0 * dx[1:-1] + dx[2:]) / 6.0
    dy = (cp[2:, :] - cp[:-2, :]) / 2.0
    gy = (dy[:, :-2] + 4.0 * dy[:, 1:-1] + dy[:, 2:]) / 6.0
    # beyond the edge the derivative along the extension axis vanishes within a pixel (zero padding), across it the
    # edge gradient continues (edge padding); sampled with the GRAD_PAD offset
    gx = np.pad(np.pad(gx, ((0, 0), (GRAD_PAD, GRAD_PAD))), ((GRAD_PAD, GRAD_PAD), (0, 0)), mode="edge")
    gy = np.pad(np.pad(gy, ((GRAD_PAD, GRAD_PAD), (0, 0))), ((0, 0), (GRAD_PAD, GRAD_PAD)), mode="edge")
    return c, ndi.spline_filter(gx, order=3, mode="nearest"), ndi.spline_filter(gy, order=3, mode="nearest")


GRAD_PAD = 2


def _sample(coeffs: np.ndarray, coords, pad: int = 0) -> np.ndarray:
    if pad:
        coords = [c + pad for c in coords]
    return ndi.map_coordinates(coeffs, coords, order=3, mode="nearest", prefilter=False)


class _Problem:
    """The registration energy of one resolution level and its gradient with respect to the coefficients.

    ``theta`` packs the displacement coefficients ``[Dx, Dy]`` of the direct transform (target → source) and,
    in the bidirectional modes, ``[Ix, Iy]`` of the inverse transform (source → target).
    """

    def __init__(self, target: np.ndarray, source: np.ndarray, gd: BGrid, gi: BGrid | None, weights: dict[str, float],
                 lm_t: np.ndarray | None, lm_s: np.ndarray | None):
        self.T, self.S = target, source
        # cubic-spline coefficient images (sampled with prefilter=False): a smooth interpolant without the kinks of
        # linear interpolation at integer positions, so line searches around the identity start make progress
        self.Sc, self.Sxc, self.Syc = _spline_images(source)
        self.gd, self.gi, self.w = gd, gi, weights
        self.Bxd, self.Byd, self.dBxd, self.dByd = gd.bx(), gd.by(), gd.bx(True), gd.by(True)
        self.yy_t, self.xx_t = (a.astype(np.float64) for a in np.mgrid[0:gd.height, 0:gd.width])
        self.n_d = gd.n * gd.n
        self.n_i = gi.n * gi.n if gi else 0
        self.stride_t = max(1, min(gd.height, gd.width) // 48)
        if gi:
            self.stride_s = max(1, min(gi.height, gi.width) // 48)
            self.Tc, self.Txc, self.Tyc = _spline_images(target)
            self.Bxi, self.Byi, self.dBxi, self.dByi = gi.bx(), gi.by(), gi.bx(True), gi.by(True)
            self.yy_s, self.xx_s = (a.astype(np.float64) for a in np.mgrid[0:gi.height, 0:gi.width])
        self.lm_t, self.lm_s = lm_t, lm_s
        if lm_t is not None and len(lm_t):
            self.lbx_d, self.lby_d = gd.basis(lm_t[:, 0], gd.width), gd.basis(lm_t[:, 1], gd.height)
            if gi:
                self.lbx_i, self.lby_i = gi.basis(lm_s[:, 0], gi.width), gi.basis(lm_s[:, 1], gi.height)

    def unpack(self, theta: np.ndarray):
        n, gd, gi = self.n_d, self.gd, self.gi
        dx, dy = theta[:n].reshape(gd.n, gd.n), theta[n:2 * n].reshape(gd.n, gd.n)
        if not gi:
            return dx, dy, None, None
        m = self.n_i
        return dx, dy, theta[2 * n:2 * n + m].reshape(gi.n, gi.n), theta[2 * n + m:].reshape(gi.n, gi.n)

    @staticmethod
    def _image_term(src, sx, sy, tgt, qx, qy):
        """SSD of ``src`` (spline coefficients) sampled at (qx, qy) against ``tgt``; E and the per-pixel gradient images.
        Samples beyond the edge take the edge value ('nearest'): continuous, and no reward for pushing pixels outside."""
        n = max(tgt.size, 1)
        coords = [qy, qx]
        r = _sample(src, coords) - tgt
        gx = r * _sample(sx, coords, GRAD_PAD) * (2.0 / n)
        gy = r * _sample(sy, coords, GRAD_PAD) * (2.0 / n)
        return float((r * r).sum() / n), gx, gy

    @staticmethod
    def _regularise(dx, dy, bx, by, dbx, dby, w_div, w_curl):
        n = by.shape[0] * bx.shape[0]
        e, gx, gy = 0.0, np.zeros_like(dx), np.zeros_like(dy)
        if w_div > 0:
            div = by @ dx @ dbx.T + dby @ dy @ bx.T
            e += w_div * float((div * div).sum() / n)
            gx += w_div * 2.0 / n * (by.T @ div @ dbx)
            gy += w_div * 2.0 / n * (dby.T @ div @ bx)
        if w_curl > 0:
            curl = by @ dy @ dbx.T - dby @ dx @ bx.T
            e += w_curl * float((curl * curl).sum() / n)
            gy += w_curl * 2.0 / n * (by.T @ curl @ dbx)
            gx -= w_curl * 2.0 / n * (dby.T @ curl @ bx)
        return e, gx, gy

    @staticmethod
    def _landmark_term(dx, dy, lbx, lby, pts_from, pts_to):
        ux = np.einsum("ij,jk,ik->i", lby, dx, lbx)
        uy = np.einsum("ij,jk,ik->i", lby, dy, lbx)
        ex = pts_from[:, 0] + ux - pts_to[:, 0]
        ey = pts_from[:, 1] + uy - pts_to[:, 1]
        n = len(ex)
        return float(((ex * ex + ey * ey).sum()) / n), 2.0 / n * (lby.T @ (ex[:, None] * lbx)), 2.0 / n * (lby.T @ (ey[:, None] * lbx))

    @staticmethod
    def _consistency(dx, dy, ix_, iy_, inner: BGrid, xx, yy, bx_o, by_o):
        """mean |p + U(p) + V(p + U(p)) - p|² over the outer grid's pixels; gradients for both coefficient sets."""
        ux, uy = by_o @ dx @ bx_o.T, by_o @ dy @ bx_o.T
        qx, qy = xx + ux, yy + uy
        inside = (qx >= 0) & (qx <= inner.width - 1) & (qy >= 0) & (qy <= inner.height - 1)
        n = max(xx.size, 1)
        gx_img, gy_img = np.zeros_like(xx), np.zeros_like(xx)
        g_ix, g_iy = np.zeros((inner.n, inner.n)), np.zeros((inner.n, inner.n))
        if not inside.any():
            return 0.0, np.zeros_like(dx), np.zeros_like(dy), g_ix, g_iy
        px, py = qx[inside], qy[inside]
        iy, wy, dwy, ix, wx, dwx = _scatter_weights(inner, px, py)
        vx, vy = _scatter_eval(ix_, iy, wy, ix, wx), _scatter_eval(iy_, iy, wy, ix, wx)
        vxx, vxy = _scatter_eval(ix_, iy, wy, ix, dwx), _scatter_eval(ix_, iy, dwy, ix, wx)
        vyx, vyy = _scatter_eval(iy_, iy, wy, ix, dwx), _scatter_eval(iy_, iy, dwy, ix, wx)
        ex, ey = ux[inside] + vx, uy[inside] + vy
        e = float((ex * ex + ey * ey).sum() / n)
        gx_img[inside] = 2.0 / n * (ex * (1.0 + vxx) + ey * vyx)
        gy_img[inside] = 2.0 / n * (ex * vxy + ey * (1.0 + vyy))
        g_ix = _scatter_adjoint(2.0 / n * ex, iy, wy, ix, wx, (inner.n, inner.n))
        g_iy = _scatter_adjoint(2.0 / n * ey, iy, wy, ix, wx, (inner.n, inner.n))
        return e, by_o.T @ gx_img @ bx_o, by_o.T @ gy_img @ bx_o, g_ix, g_iy

    def energy(self, theta: np.ndarray) -> tuple[float, np.ndarray]:
        dx, dy, ix_, iy_ = self.unpack(theta)
        w = self.w
        e = 0.0
        g_dx, g_dy = np.zeros_like(dx), np.zeros_like(dy)
        # ---- direct transform: target pixels sample the source
        ux, uy = self.Byd @ dx @ self.Bxd.T, self.Byd @ dy @ self.Bxd.T
        if w["image"] > 0:
            ei, gx, gy = self._image_term(self.Sc, self.Sxc, self.Syc, self.T, self.xx_t + ux, self.yy_t + uy)
            e += w["image"] * ei
            g_dx += w["image"] * (self.Byd.T @ gx @ self.Bxd)
            g_dy += w["image"] * (self.Byd.T @ gy @ self.Bxd)
        er, gx, gy = self._regularise(dx, dy, self.Bxd, self.Byd, self.dBxd, self.dByd, w["div"], w["curl"])
        e += er
        g_dx += gx
        g_dy += gy
        if self.lm_t is not None and w["landmark"] > 0:
            el, gx, gy = self._landmark_term(dx, dy, self.lbx_d, self.lby_d, self.lm_t, self.lm_s)
            e += w["landmark"] * el
            g_dx += w["landmark"] * gx
            g_dy += w["landmark"] * gy
        if ix_ is None:
            return e, np.concatenate([g_dx.ravel(), g_dy.ravel()])
        # ---- inverse transform: source pixels sample the target
        g_ix, g_iy = np.zeros_like(ix_), np.zeros_like(iy_)
        vx, vy = self.Byi @ ix_ @ self.Bxi.T, self.Byi @ iy_ @ self.Bxi.T
        if w["image"] > 0:
            ei, gx, gy = self._image_term(self.Tc, self.Txc, self.Tyc, self.S, self.xx_s + vx, self.yy_s + vy)
            e += w["image"] * ei
            g_ix += w["image"] * (self.Byi.T @ gx @ self.Bxi)
            g_iy += w["image"] * (self.Byi.T @ gy @ self.Bxi)
        er, gx, gy = self._regularise(ix_, iy_, self.Bxi, self.Byi, self.dBxi, self.dByi, w["div"], w["curl"])
        e += er
        g_ix += gx
        g_iy += gy
        if self.lm_t is not None and w["landmark"] > 0:
            el, gx, gy = self._landmark_term(ix_, iy_, self.lbx_i, self.lby_i, self.lm_s, self.lm_t)
            e += w["landmark"] * el
            g_ix += w["landmark"] * gx
            g_iy += w["landmark"] * gy
        if w["consistency"] > 0:
            # the consistency of two smooth fields is evaluated on a strided pixel subset (a few samples per interval suffice)
            wc = w["consistency"]
            st, si = self.stride_t, self.stride_s
            ec, a, b, c, d = self._consistency(dx, dy, ix_, iy_, self.gi, self.xx_t[::st, ::st], self.yy_t[::st, ::st], self.Bxd[::st], self.Byd[::st])
            e += wc * ec
            g_dx += wc * a
            g_dy += wc * b
            g_ix += wc * c
            g_iy += wc * d
            ec, a, b, c, d = self._consistency(ix_, iy_, dx, dy, self.gd, self.xx_s[::si, ::si], self.yy_s[::si, ::si], self.Bxi[::si], self.Byi[::si])
            e += wc * ec
            g_ix += wc * a
            g_iy += wc * b
            g_dx += wc * c
            g_dy += wc * d
        return e, np.concatenate([g_dx.ravel(), g_dy.ravel(), g_ix.ravel(), g_iy.ravel()])


# ============================================================================ multi-resolution driver
def _level_dims(h: int, w: int, f: int) -> tuple[int, int]:
    return max(int(round(h / f)), 8), max(int(round(w / f)), 8)


def _prepare(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The plane normalised to 0–255 (so the SSD term is weighted against the px² terms like bUnwarpJ's 8-bit
    intensities), resized to the level's shape and lightly smoothed."""
    p = _norm01(plane) * 255.0
    if tuple(p.shape) != tuple(shape):
        p = resize(p, shape, order=1, anti_aliasing=True, preserve_range=True)
    return ndi.gaussian_filter(np.asarray(p, np.float64), 0.8)


def _transfer(dx: np.ndarray, dy: np.ndarray, old: BGrid, new: BGrid) -> tuple[np.ndarray, np.ndarray]:
    """Carry displacement coefficients to another image size (scaled) and control-point density (refit)."""
    sx, sy = max(new.width - 1, 1) / max(old.width - 1, 1), max(new.height - 1, 1) / max(old.height - 1, 1)
    dx, dy = dx * sx, dy * sy
    if new.intervals == old.intervals:
        return dx, dy
    coarse = BGrid(old.intervals, new.width, new.height)
    ux, uy = coarse.field(dx, dy)
    return new.fit(ux, uy)


def _level_index(name: str) -> int:
    low = str(name).strip().lower()
    for i, lv in enumerate(LEVELS):
        if lv.lower() == low:
            return i
    try:
        return int(low)
    except ValueError:
        raise ValueError(f"Unknown deformation level {name!r}; choose one of {LEVELS}") from None


def register_bspline(target: np.ndarray, source: np.ndarray, mode: str = "Fast", subsampling: int = 0, initial_level: int = 0, final_level: int = 2,
                     weights: dict[str, float] | None = None, lm_t: np.ndarray | None = None, lm_s: np.ndarray | None = None,
                     stop_threshold: float = 0.01, ctx=None, max_iter: int | None = None) -> dict[str, Any]:
    """Register ``source`` (H_s, W_s) onto ``target`` (H_t, W_t) with a coarse-to-fine cubic B-spline FFD.

    Returns ``{"direct": (intervals, Dx, Dy), "inverse": (intervals, Ix, Iy) | None, "energy": E, "iterations": n}``
    with displacement coefficients in full-resolution pixels (direct grid over the target, inverse over the source).
    Landmarks ``lm_t`` (target, x/y pixel centres) and ``lm_s`` (source) correspond one to one.
    """
    w = {"image": 1.0, "div": 0.0, "curl": 0.0, "landmark": 1.0, "consistency": 10.0}
    w.update(weights or {})
    bidirectional = mode != "Mono"
    if not bidirectional:
        w["consistency"] = 0.0
    ht, wt = target.shape
    hs, ws = source.shape
    if min(ht, wt, hs, ws) < 8:
        raise ValueError("Images must be at least 8 × 8 pixels")
    lo, hi = sorted((int(initial_level), int(final_level)))
    levels = list(range(lo, hi + 1))
    iters = int(max_iter) if max_iter else ITERATIONS.get(mode, 60)
    tol = max(float(stop_threshold), 0.0) * 1e-3  # relative energy change per iteration (the objective is scaled to 100 at the start of a level)
    sub = 2 ** max(int(subsampling), 0)
    have_lm = lm_t is not None and lm_s is not None and len(lm_t) > 0 and w["landmark"] > 0
    dx = dy = ix_ = iy_ = None
    prev_d = prev_i = None
    total_it = 0
    energy = 0.0
    from scipy.optimize import minimize
    for li, s in enumerate(levels):
        intervals = 2 ** s
        f = sub * 2 ** (hi - s)
        while f > sub and min(_level_dims(ht, wt, f) + _level_dims(hs, ws, f)) < 32:
            f //= 2
        t_dims, s_dims = _level_dims(ht, wt, f), _level_dims(hs, ws, f)
        gd = BGrid(intervals, t_dims[1], t_dims[0])
        gi = BGrid(intervals, s_dims[1], s_dims[0]) if bidirectional else None
        tl, sl = _prepare(target, t_dims), _prepare(source, s_dims)
        lt = ls = None
        if have_lm:
            lt = lm_t * np.array([(gd.width - 1) / max(wt - 1, 1), (gd.height - 1) / max(ht - 1, 1)])
            ls = lm_s * np.array([(s_dims[1] - 1) / max(ws - 1, 1), (s_dims[0] - 1) / max(hs - 1, 1)])
        if dx is None:
            dx, dy = np.zeros((gd.n, gd.n)), np.zeros((gd.n, gd.n))
            if gi:
                ix_, iy_ = np.zeros((gi.n, gi.n)), np.zeros((gi.n, gi.n))
        else:
            dx, dy = _transfer(dx, dy, prev_d, gd)
            if gi:
                ix_, iy_ = _transfer(ix_, iy_, prev_i, gi)
        prob = _Problem(tl, sl, gd, gi, w, lt, ls)
        theta0 = np.concatenate([dx.ravel(), dy.ravel()] + ([ix_.ravel(), iy_.ravel()] if gi else []))
        e0 = prob.energy(theta0)[0]
        scale = 100.0 / e0 if e0 > 1e-12 else 1.0  # L-BFGS-B's ftol is relative to max(|f|, 1): work in 'percent of the start energy'

        def objective(x, _prob=prob, _scale=scale):
            e, g = _prob.energy(x)
            return e * _scale, g * _scale

        count = {"it": 0}

        last = {"f": 100.0, "x": theta0.copy()}

        def callback(_x, _li=li, _count=count, _prob=prob, _scale=scale, _last=last):
            _count["it"] += 1
            if ctx is not None:
                ctx.check_cancel()
                ctx.progress((_li + min(_count["it"] / iters, 1.0)) / len(levels), f"{LEVELS[min(s, 4)]} ({intervals} intervals): iteration {_count['it']}/{iters}")
            f = _prob.energy(_x)[0] * _scale
            improved = (_last["f"] - f) / max(abs(_last["f"]), 1e-12)
            _last["f"], _last["x"] = f, np.array(_x, copy=True)
            if _count["it"] >= 3 and improved < tol:  # bUnwarpJ's stop threshold: relative energy change per iteration
                raise StopIteration

        if ctx is not None:
            ctx.check_cancel()
            ctx.progress(li / len(levels), f"{LEVELS[min(s, 4)]} ({intervals} intervals)")
        res = minimize(objective, theta0, jac=True, method="L-BFGS-B", callback=callback,
                       options={"maxiter": iters, "maxfun": iters * 5, "ftol": 1e-15, "gtol": 1e-15, "maxcor": 20})
        theta, f_best = (res.x, float(res.fun)) if np.all(np.isfinite(res.x)) else (theta0, 100.0)
        if np.all(np.isfinite(last["x"])) and last["f"] < f_best:
            theta, f_best = last["x"], last["f"]
        if f_best > 100.0:
            theta, f_best = theta0, 100.0
        energy = float(f_best / scale)
        total_it += int(count["it"])
        dx, dy, ix_, iy_ = prob.unpack(theta)
        prev_d, prev_i = gd, gi
    full_d = BGrid(prev_d.intervals, wt, ht)
    dx, dy = _transfer(dx, dy, prev_d, full_d)
    inverse = None
    if prev_i is not None:
        full_i = BGrid(prev_i.intervals, ws, hs)
        ix_, iy_ = _transfer(ix_, iy_, prev_i, full_i)
        inverse = (full_i.intervals, ix_, iy_)
    return {"direct": (full_d.intervals, dx, dy), "inverse": inverse, "energy": energy, "iterations": total_it}


# ============================================================================ fields, warping, inversion
def coeffs_to_field(intervals: int, cx: np.ndarray, cy: np.ndarray, width: int, height: int, absolute: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Dense displacement (ux, uy) of an elastic transform on a width × height target (``absolute`` = file coefficients)."""
    grid = BGrid(int(intervals), int(width), int(height))
    cx, cy = np.asarray(cx, np.float64), np.asarray(cy, np.float64)
    if cx.shape != (grid.n, grid.n) or cy.shape != (grid.n, grid.n):
        raise ValueError(f"Coefficient grids must be {grid.n} × {grid.n} for {intervals} intervals (got {cx.shape})")
    if absolute:
        idx, idy = grid.identity()
        cx, cy = cx - idx, cy - idy
    return grid.field(cx, cy)


def field_to_coeffs(ux: np.ndarray, uy: np.ndarray, intervals: int) -> tuple[np.ndarray, np.ndarray]:
    """Absolute (file) coefficients of the elastic transform best fitting a dense displacement field."""
    h, w = ux.shape
    grid = BGrid(int(intervals), w, h)
    dx, dy = grid.fit(ux, uy)
    idx, idy = grid.identity()
    return dx + idx, dy + idy


def warp_field(plane: np.ndarray, ux: np.ndarray, uy: np.ndarray, order: int = 1) -> np.ndarray:
    """Sample ``plane`` (any size) at target pixel + displacement; output has the field's shape (zeros outside)."""
    h, w = ux.shape
    yy, xx = np.mgrid[0:h, 0:w]
    return ndi.map_coordinates(_f32(plane), [yy + uy, xx + ux], order=order, mode="constant", cval=0.0).astype(np.float32)


def invert_field(ux: np.ndarray, uy: np.ndarray, out_shape: tuple[int, int] | None = None, iterations: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Displacement (vx, vy) on ``out_shape`` with v(x + u(x)) ≈ −u(x) (fixed-point inversion)."""
    h, w = out_shape or ux.shape
    yy, xx = (a.astype(np.float64) for a in np.mgrid[0:h, 0:w])
    vx, vy = np.zeros((h, w)), np.zeros((h, w))
    # start from −u sampled on the output grid where the fields overlap
    if out_shape is None or tuple(out_shape) == ux.shape:
        vx, vy = -ux.astype(np.float64), -uy.astype(np.float64)
    for _ in range(max(int(iterations), 1)):
        coords = [yy + vy, xx + vx]
        vx = -ndi.map_coordinates(ux, coords, order=1, mode="nearest")
        vy = -ndi.map_coordinates(uy, coords, order=1, mode="nearest")
    return vx, vy


def warping_index(ux1, uy1, ux2, uy2) -> float:
    """Mean |T2(T1(x)) − x| over the pixels of T1's domain whose image lies inside T2's domain (bUnwarpJ's warping index)."""
    h, w = ux1.shape
    yy, xx = (a.astype(np.float64) for a in np.mgrid[0:h, 0:w])
    qx, qy = xx + ux1, yy + uy1
    inside = (qx >= 0) & (qx <= ux2.shape[1] - 1) & (qy >= 0) & (qy <= ux2.shape[0] - 1)
    if not inside.any():
        return float("nan")
    coords = [qy[inside], qx[inside]]
    zx = qx[inside] + ndi.map_coordinates(ux2, coords, order=1, mode="nearest")
    zy = qy[inside] + ndi.map_coordinates(uy2, coords, order=1, mode="nearest")
    return float(np.mean(np.hypot(zx - xx[inside], zy - yy[inside])))


# ============================================================================ transform files
def save_elastic_transform(path: str | Path, intervals: int, cx: np.ndarray, cy: np.ndarray) -> str:
    """bUnwarpJ elastic transform file (absolute coefficients)."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Intervals={int(intervals)}", "", "X Coeffs -----------------------------------"]
    lines += [" ".join(f"{v:.6f}" for v in row) for row in np.asarray(cx, np.float64)]
    lines += ["", "Y Coeffs -----------------------------------"]
    lines += [" ".join(f"{v:.6f}" for v in row) for row in np.asarray(cy, np.float64)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


def save_raw_transform(path: str | Path, x_map: np.ndarray, y_map: np.ndarray) -> str:
    """bUnwarpJ raw transform file: absolute source coordinates (X Trans, Y Trans) of every target pixel."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    h, w = x_map.shape
    lines = [f"Width={w}", f"Height={h}", "", "X Trans -----------------------------------"]
    lines += [" ".join(f"{v:.4f}" for v in row) for row in np.asarray(x_map, np.float64)]
    lines += ["", "Y Trans -----------------------------------"]
    lines += [" ".join(f"{v:.4f}" for v in row) for row in np.asarray(y_map, np.float64)]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


_NUM = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|NaN|nan|Infinity|-Infinity")


def _numbers(text: str) -> np.ndarray:
    return np.array([float(v.replace("Infinity", "inf")) for v in _NUM.findall(text)], np.float64)


def load_transform(path: str | Path) -> dict[str, Any]:
    """Read a bUnwarpJ elastic (``{"kind": "elastic", "intervals", "cx", "cy"}``) or raw
    (``{"kind": "raw", "x", "y", "width", "height"}``) transform file."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"Transform file not found: {p}")
    text = p.read_text(encoding="utf-8", errors="replace")
    head = text.lstrip()[:200].lower()
    if head.startswith("intervals"):
        m = re.search(r"Intervals\s*=\s*(\d+)", text, re.I)
        xi, yi = text.find("X Coeffs"), text.find("Y Coeffs")
        if not m or xi < 0 or yi < 0:
            raise ValueError(f"{p.name} is not a bUnwarpJ elastic transform file")
        n = int(m.group(1)) + 3
        cx, cy = _numbers(text[xi + 8:yi]), _numbers(text[yi + 8:])
        if cx.size != n * n or cy.size != n * n:
            raise ValueError(f"{p.name}: expected {n}×{n} coefficients per axis, found {cx.size} / {cy.size}")
        return {"kind": "elastic", "intervals": n - 3, "cx": cx.reshape(n, n), "cy": cy.reshape(n, n)}
    if head.startswith("width"):
        mw, mh = re.search(r"Width\s*=\s*(\d+)", text, re.I), re.search(r"Height\s*=\s*(\d+)", text, re.I)
        xi, yi = text.find("X Trans"), text.find("Y Trans")
        if not mw or not mh or xi < 0 or yi < 0:
            raise ValueError(f"{p.name} is not a bUnwarpJ raw transform file")
        w, h = int(mw.group(1)), int(mh.group(1))
        x, y = _numbers(text[xi + 7:yi]), _numbers(text[yi + 7:])
        if x.size != w * h or y.size != w * h:
            raise ValueError(f"{p.name}: expected {w}×{h} values per axis, found {x.size} / {y.size}")
        return {"kind": "raw", "width": w, "height": h, "x": x.reshape(h, w), "y": y.reshape(h, w)}
    raise ValueError(f"{p.name} is neither an elastic (Intervals=…) nor a raw (Width=…) bUnwarpJ transform file")


def transform_field(tr: dict[str, Any], width: int = 0, height: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Dense displacement (ux, uy) of a loaded / stored transform; elastic transforms need the target size."""
    if tr["kind"] == "raw":
        h, w = tr["x"].shape
        yy, xx = np.mgrid[0:h, 0:w]
        return tr["x"] - xx, tr["y"] - yy
    w = int(width or tr.get("width") or 0)
    h = int(height or tr.get("height") or 0)
    if w <= 0 or h <= 0:
        raise ValueError("An elastic transform needs the target image size (width / height)")
    return coeffs_to_field(tr["intervals"], tr["cx"], tr["cy"], w, h, absolute=True)


# ============================================================================ session helpers
def _default_size(ctx, width: int, height: int) -> tuple[int, int]:
    """(width, height) for an elastic transform: the given size, else the active image's (ops without a required image)."""
    if width > 0 and height > 0:
        return int(width), int(height)
    img = ctx.image if ctx.image is not None else ctx.session.active
    return (img.width, img.height) if img is not None else (0, 0)


def _stored(ctx) -> dict[str, Any]:
    st = ctx.session.settings.get(BUNWARPJ_KEY)
    if not st:
        raise ValueError("No bUnwarpJ transform in this session – run 'bUnwarpJ' or 'Load Elastic Transform' first")
    return st


def _entry(intervals: int, dx: np.ndarray, dy: np.ndarray, width: int, height: int) -> dict[str, Any]:
    grid = BGrid(int(intervals), int(width), int(height))
    idx, idy = grid.identity()
    return {"kind": "elastic", "intervals": int(intervals), "cx": dx + idx, "cy": dy + idy, "width": int(width), "height": int(height)}


def _store(ctx, direct: dict[str, Any], inverse: dict[str, Any] | None, source: str, target: str) -> None:
    ctx.session.settings[BUNWARPJ_KEY] = {"direct": direct, "inverse": inverse, "source": source, "target": target}
    ux, uy = transform_field(direct)
    ctx.session.settings[DEFORMATION_KEY] = {"flow": np.ascontiguousarray(np.stack([uy, ux]).astype(np.float32)), "source": source, "target": target,
                                             "shape": [direct["height"], direct["width"]], "method": "bUnwarpJ"}


def _landmarks(ctx, text: str, img: LumenImage, what: str) -> np.ndarray:
    """(N, 2) landmark pixel centres: 'x,y; …' text, a ROI-Manager name, or the point ROIs attached to ``img``."""
    s = str(text or "").strip()
    pts = np.zeros((0, 2))
    if s:
        try:
            pts = parse_points(s)
        except ValueError:
            pts = np.zeros((0, 2))
        if pts.size == 0:
            pts = _points_of_roi(_manager_roi(ctx, s))
    else:
        for r in ctx.session.roi_manager.rois:
            if r.get("type") == "point" and r.get("image_id") == img.id:
                pts = np.vstack([pts, np.asarray(r.get("points", []), float).reshape(-1, 2)])
    return pts - 0.5 if pts.size else pts


def _warp_image(img: LumenImage, ux: np.ndarray, uy: np.ndarray, order: int, name: str, like: LumenImage | None) -> LumenImage:
    h, w = ux.shape
    out = np.zeros((img.nt, img.nz, img.nc, h, w), np.float32)
    for t, z, c in img.iter_planes():
        out[t, z, c] = warp_field(img.data[t, z, c], ux, uy, order)
    res = img.copy(name=name, data=cast_to(out, img.data.dtype))
    if like is not None:
        res.calibration = Calibration(**like.calibration.to_dict())
    res.overlay = []
    res.sync_channels()
    return res


def _field_stats(ux: np.ndarray, uy: np.ndarray) -> tuple[float, float]:
    mag = np.hypot(ux, uy)
    return float(mag.mean()), float(mag.max())


# ============================================================================ ops
@op("registration.bunwarpj", "bUnwarpJ (B-spline Elastic Registration)", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("target", "image", None, help="The open image to register the current (source) image onto"),
            Param("mode", "choice", "Fast", choices=REG_MODES, label="Registration mode", help="Fast / Accurate: bidirectional (direct + inverse transform, consistency term); Mono: source → target only"),
            Param("subsampling", "int", 0, min=0, max=7, label="Image subsampling factor", help="Register on images reduced by 2^n (faster, smoother)"),
            Param("initial_deformation", "choice", "Very Coarse", choices=LEVELS[:4], help="Coarsest control-point grid (Very Coarse = 1 interval, Coarse = 2, Fine = 4, Very Fine = 8)"),
            Param("final_deformation", "choice", "Fine", choices=LEVELS, help="Finest control-point grid (Super Fine = 16 intervals)"),
            Param("divergence_weight", "float", 0.0, min=0, step=0.1, help="Penalises local expansion / contraction"),
            Param("curl_weight", "float", 0.0, min=0, step=0.1, help="Penalises local rotation"),
            Param("landmark_weight", "float", 1.0, min=0, step=0.1, help="Weight of the landmark term (only with landmarks)"),
            Param("image_weight", "float", 1.0, min=0, step=0.1, help="Weight of the image similarity (sum of squared differences)"),
            Param("consistency_weight", "float", 10.0, min=0, step=1, help="Fast / Accurate: penalises direct ∘ inverse ≠ identity"),
            Param("stop_threshold", "float", 0.01, min=0, step=0.001, help="Stop a resolution level when the relative energy change per iteration drops below threshold × 1e-3"),
            Param("source_landmarks", "text", "", help="Landmarks on the source: 'x1,y1; x2,y2; …' or the name of a point ROI in the ROI Manager (empty = the point ROIs attached to the source image)"),
            Param("target_landmarks", "text", "", help="Matching landmarks on the target, same order (empty = the point ROIs attached to the target image)"),
            Param("interpolation", "choice", "Linear", choices=INTERP),
            Param("output_field", "bool", False, label="Output deformation field", help="Also create the 2-channel 32-bit (dx, dy) field image usable by 'Apply Deformation'"),
            Param("output_target", "bool", False, label="Registered target image", help="Fast / Accurate: also warp the target onto the source with the inverse transform"),
            Param("save_direct", "string", "", label="Save direct transform to", help="Path of an elastic transform text file (bUnwarpJ format; empty = don't save)"),
            Param("save_inverse", "string", "", label="Save inverse transform to"),
            Param("store", "bool", True, label="Keep the transform", help="Keep the transform in the session for Apply / Save / Invert and 'Apply Deformation'"),
            Param("max_iterations", "int", 0, min=0, max=2000, advanced=True, help="Iterations per level (0 = by mode: Fast 40, Accurate / Mono 150)")],
    description="Register the current (source) image onto a target image with a cubic B-spline free-form deformation (Fiji bUnwarpJ): coarse-to-fine control-point grids, image (SSD), divergence / curl, landmark and consistency energies; bidirectional in Fast / Accurate mode. Outputs the registered source (and optionally the registered target, the deformation field and bUnwarpJ transform files).",
    keywords=["bunwarpj", "b-spline", "elastic", "non-rigid", "free-form deformation", "unwarp", "landmarks", "consistent"])
def bunwarpj(ctx, target=None, mode="Fast", subsampling=0, initial_deformation="Very Coarse", final_deformation="Fine", divergence_weight=0.0, curl_weight=0.0,
             landmark_weight=1.0, image_weight=1.0, consistency_weight=10.0, stop_threshold=0.01, source_landmarks="", target_landmarks="", interpolation="Linear",
             output_field=False, output_target=False, save_direct="", save_inverse="", store=True, max_iterations=0):
    img = ctx.img
    if not target:
        raise ValueError("Choose a target image")
    tgt = ctx.session.get_image(target)
    if tgt.id == img.id:
        raise ValueError("Target must be a different image")
    lm_s, lm_t = _landmarks(ctx, source_landmarks, img, "source"), _landmarks(ctx, target_landmarks, tgt, "target")
    if (len(lm_s) > 0) != (len(lm_t) > 0):
        raise ValueError("Give landmarks on both images (source and target) or on neither")
    if len(lm_s) != len(lm_t):
        raise ValueError(f"{len(lm_s)} source landmarks but {len(lm_t)} target landmarks – they must correspond one to one")
    if image_weight <= 0 and (len(lm_s) == 0 or landmark_weight <= 0):
        raise ValueError("With image_weight 0 the registration needs landmarks (and a landmark weight > 0)")
    weights = {"image": float(image_weight), "div": float(divergence_weight), "curl": float(curl_weight), "landmark": float(landmark_weight), "consistency": float(consistency_weight)}
    tp, sp = _estimation_plane(tgt), _estimation_plane(img)
    ctx.progress(0.0, "Preparing")
    res = register_bspline(tp, sp, mode=mode, subsampling=subsampling, initial_level=_level_index(initial_deformation), final_level=_level_index(final_deformation),
                           weights=weights, lm_t=lm_t if len(lm_t) else None, lm_s=lm_s if len(lm_s) else None, stop_threshold=stop_threshold, ctx=ctx,
                           max_iter=max_iterations or None)
    ctx.progress(0.97, "Warping")
    intervals, dx, dy = res["direct"]
    direct = _entry(intervals, dx, dy, tgt.width, tgt.height)
    ux, uy = BGrid(intervals, tgt.width, tgt.height).field(dx, dy)
    order = ORDER_MAP[interpolation]
    reg = _warp_image(img, ux, uy, order, f"{img.name} on {tgt.name} (bUnwarpJ)", like=tgt)
    inverse = None
    new_images = [reg]
    if res["inverse"] is not None:
        ii, ix_, iy_ = res["inverse"]
        inverse = _entry(ii, ix_, iy_, img.width, img.height)
        if output_target:
            vx, vy = BGrid(ii, img.width, img.height).field(ix_, iy_)
            new_images.append(_warp_image(tgt, vx, vy, order, f"{tgt.name} on {img.name} (bUnwarpJ inverse)", like=img))
    if output_field:
        new_images.append(_field_image(np.stack([uy, ux]).astype(np.float32), f"Deformation field {img.name} → {tgt.name}", like=tgt))
    written = []
    if str(save_direct or "").strip():
        written.append(save_elastic_transform(save_direct, direct["intervals"], direct["cx"], direct["cy"]))
    if str(save_inverse or "").strip():
        if inverse is None:
            raise ValueError("No inverse transform in Mono mode – choose Fast or Accurate to save the inverse")
        written.append(save_elastic_transform(save_inverse, inverse["intervals"], inverse["cx"], inverse["cy"]))
    if store:
        _store(ctx, direct, inverse, img.name, tgt.name)
    mean_d, max_d = _field_stats(ux, uy)
    row = {"Label": f"{img.name} → {tgt.name}", "Mode": mode, "Intervals": intervals, "Mean displacement": mean_d, "Max displacement": max_d,
           "Energy": res["energy"], "Iterations": res["iterations"], "N landmarks": int(len(lm_s))}
    if inverse is not None:
        vx, vy = transform_field(inverse)
        row["Warping index"] = warping_index(ux, uy, vx, vy)
    if len(lm_s):
        lx = np.einsum("ij,jk,ik->i", BGrid(intervals, tgt.width, tgt.height).basis(lm_t[:, 1], tgt.height), dx, BGrid(intervals, tgt.width, tgt.height).basis(lm_t[:, 0], tgt.width))
        ly = np.einsum("ij,jk,ik->i", BGrid(intervals, tgt.width, tgt.height).basis(lm_t[:, 1], tgt.height), dy, BGrid(intervals, tgt.width, tgt.height).basis(lm_t[:, 0], tgt.width))
        row["Landmark RMS"] = float(np.sqrt(np.mean((lm_t[:, 0] + lx - lm_s[:, 0]) ** 2 + (lm_t[:, 1] + ly - lm_s[:, 1]) ** 2)))
    msg = f"bUnwarpJ ({mode}, {initial_deformation} → {final_deformation}, {res['iterations']} iterations): {img.name} onto {tgt.name}; mean displacement {mean_d:.2f} px, max {max_d:.2f} px"
    if "Warping index" in row:
        msg += f"; warping index {row['Warping index']:.3f} px"
    if "Landmark RMS" in row:
        msg += f"; landmark RMS {row['Landmark RMS']:.2f} px"
    if written:
        msg += "; saved " + ", ".join(written)
    return OpResult(new_images=new_images, results=[row], message=msg, log=msg)


@op("registration.bunwarpj_apply", "bUnwarpJ ▸ Apply Elastic / Raw Transform", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("file", "string", "", label="Transform file", help="bUnwarpJ elastic or raw transform text file (empty = the transform kept in the session)"),
            Param("direction", "choice", "Direct", choices=["Direct", "Inverse"], help="Session transform: direct (source → target) or inverse; a file is always applied as given"),
            Param("target", "image", None, help="Image defining the output size of an elastic transform (empty = the transform's own target size, else the current image)"),
            Param("interpolation", "choice", "Linear", choices=INTERP),
            Param("invert", "bool", False, help="Apply the numerically inverted transform")],
    description="Warp every plane of the current image with a bUnwarpJ transform (session or file). Elastic transforms are evaluated on the target image size; raw transforms carry their own size.",
    keywords=["bunwarpj", "load elastic transformation", "load raw transformation", "apply transform", "warp"])
def bunwarpj_apply(ctx, file="", direction="Direct", target=None, interpolation="Linear", invert=False):
    img = ctx.img
    like = ctx.session.get_image(target) if target else None
    w, h = (like.width, like.height) if like is not None else (0, 0)
    if str(file or "").strip():
        tr = load_transform(file)
        label = Path(file).name
        if tr["kind"] == "elastic" and not w:
            w, h = img.width, img.height
    else:
        st = _stored(ctx)
        tr = st["inverse" if direction == "Inverse" else "direct"]
        if tr is None:
            raise ValueError("The session transform has no inverse (Mono mode) – run 'bUnwarpJ ▸ Invert' first")
        label = f"{st.get('source')} → {st.get('target')}" + (" (inverse)" if direction == "Inverse" else "")
    ux, uy = transform_field(tr, w, h)
    if invert:
        ux, uy = invert_field(ux, uy)
        label += " inverted"
    res = _warp_image(img, ux, uy, ORDER_MAP[interpolation], f"{img.name} (bUnwarpJ)", like=like)
    mean_d, max_d = _field_stats(ux, uy)
    msg = f"Applied bUnwarpJ transform {label} to {img.nt * img.nz * img.nc} planes → {res.width} × {res.height}; mean displacement {mean_d:.2f} px"
    return OpResult(new_images=[res], message=msg, log=msg)


@op("registration.bunwarpj_load", "bUnwarpJ ▸ Load Transform", CAT, kind="analysis", preview=False, roi_aware=False,
    params=[Param("file", "string", "", label="Transform file", help="bUnwarpJ elastic (Intervals=…) or raw (Width=…) transform text file"),
            Param("as_inverse", "bool", False, label="Load as the inverse transform", help="Keep it as the inverse (source → target) of the session transform"),
            Param("width", "int", 0, min=0, help="Target width of an elastic transform (0 = current image)"), Param("height", "int", 0, min=0),
            Param("output_field", "bool", False, label="Deformation field image", help="Also create the 2-channel (dx, dy) field image")],
    description="Load a bUnwarpJ elastic or raw transform file into the session (for Apply / Save / Invert / 'Apply Deformation'); raw files are fitted with a B-spline of the same density when kept as elastic.",
    keywords=["bunwarpj", "load elastic transformation", "load raw transformation", "transform file"])
def bunwarpj_load(ctx, file="", as_inverse=False, width=0, height=0, output_field=False):
    img = ctx.img
    if not str(file or "").strip():
        raise ValueError("Give the transform file to load")
    tr = load_transform(file)
    w, h = (int(width), int(height)) if width > 0 and height > 0 else (img.width, img.height)
    if tr["kind"] == "elastic":
        entry = {**tr, "width": w, "height": h}
    else:
        entry = tr
    ux, uy = transform_field(entry)
    st = ctx.session.settings.get(BUNWARPJ_KEY) or {"direct": None, "inverse": None, "source": "", "target": ""}
    if as_inverse:
        st = {**st, "inverse": entry}
        if st.get("direct") is None:
            st["direct"] = _entry(entry.get("intervals", 4), *BGrid(entry.get("intervals", 4), ux.shape[1], ux.shape[0]).fit(*invert_field(ux, uy)), ux.shape[1], ux.shape[0])
    else:
        st = {**st, "direct": entry, "source": st.get("source") or Path(file).stem, "target": st.get("target") or img.name}
    ctx.session.settings[BUNWARPJ_KEY] = st
    dux, duy = transform_field(st["direct"])
    ctx.session.settings[DEFORMATION_KEY] = {"flow": np.ascontiguousarray(np.stack([duy, dux]).astype(np.float32)), "source": st.get("source"), "target": st.get("target"),
                                             "shape": list(dux.shape), "method": "bUnwarpJ"}
    new_images = []
    if output_field:
        new_images.append(_field_image(np.stack([uy, ux]).astype(np.float32), f"Deformation field {Path(file).stem}", like=img))
    mean_d, max_d = _field_stats(ux, uy)
    msg = f"Loaded {tr['kind']} transform {Path(file).name} ({ux.shape[1]} × {ux.shape[0]}) as the {'inverse' if as_inverse else 'direct'} transform; mean displacement {mean_d:.2f} px"
    return OpResult(new_images=new_images, message=msg, log=msg)


@op("registration.bunwarpj_save", "bUnwarpJ ▸ Save Transform", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("file", "string", "", label="Direct transform file", help="Path of the elastic transform text file to write (empty = skip)"),
            Param("inverse_file", "string", "", label="Inverse transform file"),
            Param("format", "choice", "Elastic (B-spline coefficients)", choices=["Elastic (B-spline coefficients)", "Raw (pixel coordinates)"])],
    description="Write the session's bUnwarpJ transform(s) to elastic (coefficient) or raw (per-pixel coordinate) text files in Fiji's format.",
    keywords=["bunwarpj", "save transformation", "elastic transform file", "raw transform"])
def bunwarpj_save(ctx, file="", inverse_file="", format="Elastic (B-spline coefficients)"):
    st = _stored(ctx)
    written = []
    for path, key in ((file, "direct"), (inverse_file, "inverse")):
        if not str(path or "").strip():
            continue
        tr = st.get(key)
        if tr is None:
            raise ValueError(f"The session transform has no {key} part")
        if format.startswith("Raw") or tr["kind"] == "raw":
            ux, uy = transform_field(tr)
            h, w = ux.shape
            yy, xx = np.mgrid[0:h, 0:w]
            written.append(save_raw_transform(path, xx + ux, yy + uy))
        else:
            written.append(save_elastic_transform(path, tr["intervals"], tr["cx"], tr["cy"]))
    if not written:
        raise ValueError("Give at least one output file")
    msg = "Saved bUnwarpJ transform: " + ", ".join(written)
    return OpResult(message=msg, log=msg)


@op("registration.bunwarpj_invert", "bUnwarpJ ▸ Invert Transform", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("file", "string", "", label="Transform file", help="Elastic or raw transform to invert (empty = the session's direct transform)"),
            Param("width", "int", 0, min=0, help="Target size of an elastic transform file (0 = the current image)"), Param("height", "int", 0, min=0),
            Param("output_file", "string", "", help="Write the inverse as an elastic transform file (empty = keep in the session only)"),
            Param("swap", "bool", False, help="Make the inverse the session's direct transform (and vice versa)")],
    description="Numerically invert a bUnwarpJ transform (fixed-point inversion of the displacement field, refitted as a B-spline of the same density) and keep / save it.",
    keywords=["bunwarpj", "invert raw transformation", "inverse transform", "elastic"])
def bunwarpj_invert(ctx, file="", width=0, height=0, output_file="", swap=False):
    if str(file or "").strip():
        tr = load_transform(file)
        if tr["kind"] == "elastic":
            w, h = _default_size(ctx, width, height)
            tr = {**tr, "width": w, "height": h}
        st = ctx.session.settings.get(BUNWARPJ_KEY) or {"direct": tr, "inverse": None, "source": Path(file).stem, "target": ""}
        st = {**st, "direct": tr}
    else:
        st = _stored(ctx)
        tr = st["direct"]
    ux, uy = transform_field(tr)
    vx, vy = invert_field(ux, uy)
    intervals = int(tr.get("intervals", 4))
    inv = _entry(intervals, *BGrid(intervals, ux.shape[1], ux.shape[0]).fit(vx, vy), ux.shape[1], ux.shape[0])
    if swap:
        st = {**st, "direct": inv, "inverse": tr, "source": st.get("target", ""), "target": st.get("source", "")}
    else:
        st = {**st, "inverse": inv}
    ctx.session.settings[BUNWARPJ_KEY] = st
    written = save_elastic_transform(output_file, inv["intervals"], inv["cx"], inv["cy"]) if str(output_file or "").strip() else ""
    fx, fy = transform_field(inv)
    wi = warping_index(ux, uy, fx, fy)
    msg = f"Inverted the bUnwarpJ transform ({intervals} intervals): warping index direct ∘ inverse {wi:.3f} px" + (f"; saved {written}" if written else "") + ("; swapped" if swap else "")
    return OpResult(results=[{"Label": "Inverse", "Intervals": intervals, "Warping index": wi}], message=msg, log=msg)


@op("registration.bunwarpj_compare", "bUnwarpJ ▸ Compare Transforms", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("file1", "string", "", label="Transform 1", help="Elastic or raw transform file (empty = the session's direct transform)"),
            Param("file2", "string", "", label="Transform 2", help="Elastic or raw transform file (empty = the session's inverse transform)"),
            Param("mode", "choice", COMPARE_MODES[0], choices=COMPARE_MODES, help="Opposite: mean |T2(T1(x)) − x| (warping index); same direction: mean |T1(x) − T2(x)|"),
            Param("width", "int", 0, min=0, help="Target size of elastic transform files (0 = the current image)"), Param("height", "int", 0, min=0)],
    description="Compare two bUnwarpJ transforms (elastic or raw files, or the session's direct / inverse pair): the warping index of opposite transforms or the mean distance of two transforms of the same direction, as a Results row.",
    keywords=["bunwarpj", "compare opposite elastic transforms", "compare elastic and raw", "warping index"])
def bunwarpj_compare(ctx, file1="", file2="", mode=COMPARE_MODES[0], width=0, height=0):
    w, h = _default_size(ctx, width, height)

    def get(path: str, key: str) -> tuple[dict[str, Any], str]:
        if str(path or "").strip():
            tr = load_transform(path)
            return ({**tr, "width": w, "height": h} if tr["kind"] == "elastic" else tr), Path(path).name
        st = _stored(ctx)
        if st.get(key) is None:
            raise ValueError(f"The session transform has no {key} part – give {('file1' if key == 'direct' else 'file2')}")
        return st[key], f"session {key}"

    t1, n1 = get(file1, "direct")
    t2, n2 = get(file2, "inverse")
    ux1, uy1 = transform_field(t1)
    if mode.startswith("Opposite"):
        ux2, uy2 = transform_field(t2)
        value = warping_index(ux1, uy1, ux2, uy2)
        what = "Warping index"
    else:
        ux2, uy2 = transform_field(t2, ux1.shape[1], ux1.shape[0])
        if ux2.shape != ux1.shape:
            raise ValueError(f"Transforms of different size: {ux1.shape[::-1]} vs {ux2.shape[::-1]}")
        value = float(np.mean(np.hypot(ux1 - ux2, uy1 - uy2)))
        what = "Mean distance"
    row = {"Label": f"{n1} vs {n2}", "Mode": mode, what: value, "Width": ux1.shape[1], "Height": ux1.shape[0]}
    msg = f"{what} of {n1} and {n2}: {value:.3f} px"
    return OpResult(results=[row], message=msg, log=msg)


@op("registration.bunwarpj_convert", "bUnwarpJ ▸ Convert Transform (Elastic ↔ Raw)", CAT, kind="analysis", requires_image=False, preview=False, roi_aware=False,
    params=[Param("file", "string", "", label="Input transform file", help="Elastic (Intervals=…) or raw (Width=…) transform (empty = the session's direct transform)"),
            Param("output_file", "string", "", help="Output transform file"),
            Param("mode", "choice", CONVERT_MODES[0], choices=CONVERT_MODES),
            Param("intervals", "int", 4, min=1, max=64, help="Raw → elastic: control-point intervals of the fitted B-spline"),
            Param("width", "int", 0, min=0, help="Target size of an elastic input (0 = the current image)"), Param("height", "int", 0, min=0)],
    description="Convert a bUnwarpJ elastic transform to a raw (per-pixel source coordinate) file, or fit an elastic B-spline transform to a raw file.",
    keywords=["bunwarpj", "convert transformation to raw", "raw to elastic", "elastic to raw"])
def bunwarpj_convert(ctx, file="", output_file="", mode=CONVERT_MODES[0], intervals=4, width=0, height=0):
    if not str(output_file or "").strip():
        raise ValueError("Give the output file")
    w, h = _default_size(ctx, width, height)
    if str(file or "").strip():
        tr = load_transform(file)
        if tr["kind"] == "elastic":
            tr = {**tr, "width": w, "height": h}
        name = Path(file).name
    else:
        tr, name = _stored(ctx)["direct"], "session direct"
    ux, uy = transform_field(tr)
    hh, ww = ux.shape
    if mode.startswith("Elastic"):
        yy, xx = np.mgrid[0:hh, 0:ww]
        out = save_raw_transform(output_file, xx + ux, yy + uy)
        msg = f"Converted {name} to a raw transform ({ww} × {hh}): {out}"
    else:
        cx, cy = field_to_coeffs(ux, uy, int(intervals))
        fx, fy = coeffs_to_field(int(intervals), cx, cy, ww, hh)
        err = float(np.mean(np.hypot(fx - ux, fy - uy)))
        out = save_elastic_transform(output_file, int(intervals), cx, cy)
        msg = f"Fitted {name} with a {intervals}-interval B-spline (mean fit error {err:.3f} px): {out}"
    return OpResult(message=msg, log=msg)
