"""Plugins ▸ Stitching extras — ``registration.stitch_collection`` (Grid/Collection stitching: positions from a
TileConfiguration file, unknown positions with all-pairs phase correlation and Fiji's global optimisation thresholds,
3-D tiles, multi-channel / time-series tiles, every fusion method, sub-pixel accuracy, downsampling for the computation
and output to an open image or a file) and ``registration.elastic_montage`` (Elastic Montaging: pairwise optical-flow
alignment of the overlapping tiles blended with a spring-mesh-like least-squares relaxation of per-tile deformation
meshes, then linear blending).

Tile positions are kept internally as (z, y, x) pixel offsets of every tile's origin in the montage frame (z = 0 for
2-D tiles); TileConfiguration files store them as (x, y) or (x, y, z) like Fiji.
"""
from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage.registration import phase_cross_correlation

from ..image import LumenImage
from ..registry import Param, OpResult, op, cast_to
from .registration import INTERP, ORDER_MAP, ORDERS, _blend_weights, _f32, _grid_order, _natural_key, _read_tile, _visible_files
from .registration3d import _window
from .registration_extra import FLOW_METHODS, estimate_flow

CAT = "Plugins/Registration"
LAYOUTS = ["Positions from file", "Unknown positions (folder)", "Grid (folder)", "Open images"]
FUSIONS = ["Linear blending", "Average", "Median", "Max intensity", "Min intensity", "Overlay into composite", "Do not fuse"]
OUTPUTS = ["Open image", "Write to file", "Write to file and open"]
TIMELAPSE = ["Same positions for all frames", "Register every frame separately"]
REGISTERED_NAME = "TileConfiguration.registered.txt"


# ============================================================================ TileConfiguration files
_TC_LINE = re.compile(r"^\s*(?P<file>.+?)\s*;\s*(?P<series>[^;]*?)\s*;\s*\(\s*(?P<coords>[^)]*)\)\s*$")


def read_tile_configuration(path: str | Path) -> dict[str, Any]:
    """Parse a Fiji ``TileConfiguration.txt``: ``{"dim": 2 | 3, "entries": [{"file", "series", "position": [x, y(, z)]}, …]}``."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise ValueError(f"Tile configuration file not found: {p}")
    dim = None
    entries: list[dict[str, Any]] = []
    for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^dim\s*=\s*(\d+)\s*$", line, re.I)
        if m:
            dim = int(m.group(1))
            continue
        m = _TC_LINE.match(line)
        if not m:
            raise ValueError(f"{p.name}: cannot parse line {line!r} (expected 'file; series; (x, y[, z])')")
        coords = [float(v) for v in re.split(r"[,\s]+", m.group("coords").strip()) if v]
        series = m.group("series").strip()
        entries.append({"file": m.group("file").strip(), "series": int(series) if series.isdigit() else None, "position": coords})
    if not entries:
        raise ValueError(f"No tile entries in {p.name}")
    if dim is None:
        dim = len(entries[0]["position"])
    if dim not in (2, 3):
        raise ValueError(f"{p.name}: dim = {dim} is not supported (2 or 3)")
    for e in entries:
        if len(e["position"]) != dim:
            raise ValueError(f"{p.name}: {e['file']} has {len(e['position'])} coordinates, expected {dim}")
    return {"dim": dim, "entries": entries}


def write_tile_configuration(path: str | Path, entries, dim: int) -> str:
    """Write a Fiji ``TileConfiguration.txt``; ``entries`` are ``(file, series | None, (x, y[, z]))`` tuples."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Define the number of dimensions we are working on", f"dim = {int(dim)}", "", "# Define the image coordinates"]
    for file, series, pos in entries:
        coords = ", ".join(repr(round(float(v), 4)) for v in list(pos)[:dim])
        lines.append(f"{file}; {'' if series is None else int(series)}; ({coords})")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(p)


# ============================================================================ loading a collection
def _tile_files(root: Path, pattern: str) -> list[Path]:
    pattern = (pattern or "").strip() or "*"
    files = sorted(_visible_files(root.glob(pattern)), key=_natural_key) if any(ch in pattern for ch in "*?[") else []
    if not files:
        try:
            rx = re.compile(pattern)
            files = sorted((f for f in _visible_files(root.iterdir()) if rx.search(f.name)), key=_natural_key)
        except re.error:
            files = []
    return files


def _parse_positions(text: str) -> np.ndarray | None:
    """'x,y; x,y; …' or 'x,y,z; …' → (n, 3) array of (z, y, x), or None when empty."""
    s = str(text or "").strip()
    if not s:
        return None
    rows = []
    for part in re.split(r"[;\n]+", s):
        nums = [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", part)]
        if not nums:
            continue
        if len(nums) == 2:
            rows.append([0.0, nums[1], nums[0]])
        elif len(nums) == 3:
            rows.append([nums[2], nums[1], nums[0]])
        else:
            raise ValueError(f"Position {part.strip()!r} needs 2 or 3 coordinates (x,y or x,y,z)")
    return np.asarray(rows, np.float64) if rows else None


def _read_entry(path: Path, series: int | None) -> LumenImage:
    if series is not None:
        from ..io import read_image
        try:
            return read_image(path, series=series)
        except TypeError:
            pass
    return _read_tile(path)


def load_collection(ctx, layout: str, folder: str = "", pattern: str = "*.tif", layout_file: str = "", images: str = "", positions: str = "",
                    grid_x: int = 2, grid_y: int = 2, overlap_percent: float = 10.0, order: str = "Row-by-row") -> dict[str, Any]:
    """Tiles of a collection: ``{"tiles": [LumenImage], "names": [str], "series": [int | None], "nominal": (n, 3) | None, "root": Path | None}``."""
    layout = str(layout)
    tiles: list[LumenImage] = []
    names: list[str] = []
    series: list[int | None] = []
    nominal: np.ndarray | None = None
    root: Path | None = None
    if layout == "Open images":
        wanted = [s.strip() for s in re.split(r"[,;\n]+", str(images or "")) if s.strip()]
        if wanted:
            tiles = [ctx.session.get_image(w) for w in wanted]
        else:
            tiles = list(ctx.session.images.values())
        names = [t.name for t in tiles]
        series = [None] * len(tiles)
        nominal = _parse_positions(positions)
        if nominal is not None and len(nominal) != len(tiles):
            raise ValueError(f"{len(nominal)} positions given for {len(tiles)} images")
    else:
        if layout == "Positions from file":
            lf = Path(layout_file).expanduser() if str(layout_file or "").strip() else None
            if lf is None:
                if not str(folder or "").strip():
                    raise ValueError("Give the tile configuration file (or the folder that contains TileConfiguration.txt)")
                lf = Path(folder).expanduser() / "TileConfiguration.txt"
            root = Path(folder).expanduser() if str(folder or "").strip() else lf.parent
            conf = read_tile_configuration(lf)
            pos_rows = []
            for i, e in enumerate(conf["entries"]):
                ctx.check_cancel()
                ctx.progress(0.15 * i / len(conf["entries"]), f"Reading tile {i + 1}/{len(conf['entries'])}")
                fp = root / e["file"]
                if not fp.exists():
                    raise ValueError(f"Tile {e['file']} listed in {lf.name} not found in {root}")
                try:
                    tiles.append(_read_entry(fp, e["series"]))
                except Exception as exc:
                    raise ValueError(f"Could not read tile {e['file']}: {exc}") from exc
                names.append(e["file"])
                series.append(e["series"])
                x, y = e["position"][0], e["position"][1]
                z = e["position"][2] if conf["dim"] == 3 else 0.0
                pos_rows.append([z, y, x])
            nominal = np.asarray(pos_rows, np.float64)
        else:
            if not str(folder or "").strip():
                raise ValueError("Give the folder that contains the tiles")
            root = Path(folder).expanduser()
            if not root.is_dir():
                raise ValueError(f"{root} is not a folder")
            files = _tile_files(root, pattern)
            if layout == "Grid (folder)":
                n = int(grid_x) * int(grid_y)
                if len(files) < n:
                    raise ValueError(f"Found {len(files)} tiles matching {pattern!r} in {root}, need {n} for a {grid_x}×{grid_y} grid")
                files = files[:n]
            if not files:
                raise ValueError(f"No tiles matching {pattern!r} in {root}")
            for i, f in enumerate(files):
                ctx.check_cancel()
                ctx.progress(0.15 * i / len(files), f"Reading tile {i + 1}/{len(files)}")
                try:
                    tiles.append(_read_tile(f))
                except Exception as exc:
                    raise ValueError(f"Could not read tile {f.name}: {exc}") from exc
                names.append(f.name)
                series.append(None)
            if layout == "Grid (folder)":
                h, w = tiles[0].height, tiles[0].width
                ov = max(0.0, min(float(overlap_percent), 90.0)) / 100.0
                rc = [_grid_order(i, int(grid_x), int(grid_y), order) for i in range(len(tiles))]
                nominal = np.array([[0.0, r * h * (1 - ov), c * w * (1 - ov)] for r, c in rc], np.float64)
    if len(tiles) < 2:
        raise ValueError("A collection needs at least two tiles")
    nt = tiles[0].nt
    for t, nm in zip(tiles, names):
        if t.nt != nt:
            raise ValueError(f"{nm} has {t.nt} frames, {names[0]} has {nt}: all tiles need the same number of frames")
    return {"tiles": tiles, "names": names, "series": series, "nominal": nominal, "root": root}


# ============================================================================ n-D phase correlation
def _est_volume(img: LumenImage, channel: int, t: int, is3d: bool, downsample: int) -> tuple[np.ndarray, tuple[int, ...]]:
    """The float32 (Z, Y, X) volume or (Y, X) plane used for the registration (one channel or the channel average,
    frame ``t``), block-averaged by ``downsample`` where the axis is long enough; returns (volume, factors)."""
    c_idx = None if channel <= 0 else min(int(channel) - 1, img.nc - 1)
    data = img.data[min(t, img.nt - 1)]  # (Z, C, Y, X)
    vol = _f32(data[:, c_idx]) if c_idx is not None else _f32(data).mean(axis=1)
    if not is3d:
        vol = vol[vol.shape[0] // 2]
    ds = max(int(downsample), 1)
    factors = tuple(ds if n >= 4 * ds else 1 for n in vol.shape)
    if ds > 1 and any(f > 1 for f in factors):
        from skimage.transform import downscale_local_mean
        vol = downscale_local_mean(vol, factors).astype(np.float32)
    else:
        factors = tuple(1 for _ in vol.shape)
    return vol, factors


def _overlap_slices_nd(shape_a, shape_b, off) -> tuple[tuple[slice, ...], tuple[slice, ...]] | None:
    sa, sb = [], []
    for n_a, n_b, o in zip(shape_a, shape_b, off):
        o = int(round(o))
        lo, hi = max(0, o), min(n_a, o + n_b)
        if hi - lo < 2:
            return None
        sa.append(slice(lo, hi))
        sb.append(slice(lo - o, hi - o))
    return tuple(sa), tuple(sb)


def _ncc_at(a: np.ndarray, b: np.ndarray, off, min_overlap: int = 16) -> float:
    """Normalised cross-correlation of the overlap of ``a`` and ``b`` placed at integer offset ``off`` (b's origin in a)."""
    sl = _overlap_slices_nd(a.shape, b.shape, off)
    if sl is None:
        return -1.0
    sa = a[sl[0]].astype(np.float64).ravel()
    sb = b[sl[1]].astype(np.float64).ravel()
    if sa.size < min_overlap:
        return -1.0
    sa = sa - sa.mean()
    sb = sb - sb.mean()
    den = math.sqrt(float((sa * sa).sum()) * float((sb * sb).sum()))
    return float((sa * sb).sum() / den) if den > 0 else -1.0


def find_offset_nd(a: np.ndarray, b: np.ndarray, n_peaks: int = 5, expected=None, max_dev=None, min_overlap: int = 16) -> tuple[np.ndarray, float]:
    """Integer position of ``b``'s origin in ``a``'s frame (2-D or 3-D) by zero-padded phase correlation; the best
    ``n_peaks`` peaks are verified on the real overlap (NCC); ``expected`` / ``max_dev`` restrict the candidates."""
    a, b = _f32(a), _f32(b)
    full = tuple(sa + sb for sa, sb in zip(a.shape, b.shape))
    axes = tuple(range(a.ndim))
    # a credible tile overlap covers at least 2 % of the smaller tile: tiny overlaps correlate perfectly by chance
    min_overlap = int(max(min_overlap, 0.02 * min(a.size, b.size)))
    fa = np.fft.rfftn(a - a.mean(), s=full, axes=axes)
    fb = np.fft.rfftn(b - b.mean(), s=full, axes=axes)
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    corr_phase = np.fft.irfftn(cross / np.where(mag > 1e-12, mag, 1.0), s=full, axes=axes).astype(np.float32)
    # the plain cross-correlation normalised by the overlap size is robust on smooth images (whitening amplifies the
    # noise of their empty high frequencies); the candidates of both correlators are verified on the real overlap
    ones_a = np.fft.rfftn(np.ones(a.shape, np.float32), s=full, axes=axes)
    ones_b = np.fft.rfftn(np.ones(b.shape, np.float32), s=full, axes=axes)
    overlap = np.fft.irfftn(ones_a * np.conj(ones_b), s=full, axes=axes)
    corr_plain = (np.fft.irfftn(cross, s=full, axes=axes) / np.maximum(overlap, float(max(min_overlap, 1)))).astype(np.float32)
    corr_plain[overlap < min_overlap - 0.5] = -np.inf
    k = min(max(int(n_peaks), 1) * 4, corr_phase.size)
    lists = []
    for corr in (corr_phase, corr_plain):
        peaks = corr == ndi.maximum_filter(corr, size=3, mode="wrap")
        flat = np.where(peaks.ravel(), corr.ravel(), -np.inf)
        cand = np.argpartition(-flat, k - 1)[:k]
        cand = cand[np.argsort(-flat[cand])]
        lists.append([int(i) for i in cand if np.isfinite(flat[i])])
    seen: set[int] = set()
    cand = []
    for pair in itertools.zip_longest(*lists):
        for i in pair:
            if i is not None and i not in seen:
                seen.add(i)
                cand.append(i)
    best_off, best = tuple(0 for _ in a.shape), -2.0
    tested = 0
    for idx in cand:
        p = np.unravel_index(int(idx), full)
        off = tuple(int(pi) if pi < sa else int(pi) - fs for pi, sa, fs in zip(p, a.shape, full))
        if expected is not None and max_dev is not None and any(abs(o - e) > m for o, e, m in zip(off, expected, max_dev)):
            continue
        ncc = _ncc_at(a, b, off, min_overlap)
        tested += 1
        if ncc > best:
            best_off, best = off, float(ncc)
        if tested >= 2 * max(int(n_peaks), 1):
            break
    return np.array(best_off, np.float64), best


def _refine_local(a: np.ndarray, b: np.ndarray, off, radius) -> tuple[np.ndarray, float]:
    """Best integer offset within ``radius`` (per axis) of ``off`` by exhaustive NCC on the overlap."""
    best, best_ncc = tuple(int(round(o)) for o in off), _ncc_at(a, b, off)
    ranges = [range(int(round(o)) - int(r), int(round(o)) + int(r) + 1) for o, r in zip(off, radius)]
    for cand in itertools.product(*ranges):
        ncc = _ncc_at(a, b, cand)
        if ncc > best_ncc:
            best, best_ncc = cand, ncc
    return np.array(best, np.float64), best_ncc


def _subpixel_nd(a: np.ndarray, b: np.ndarray, off, upsample: int = 10) -> np.ndarray:
    """Sub-pixel residual of an integer offset from the upsampled cross-correlation of the overlapping windows."""
    up = max(int(upsample), 1)
    zero = np.zeros(len(off))
    if up <= 1:
        return zero
    sl = _overlap_slices_nd(a.shape, b.shape, off)
    if sl is None:
        return zero
    wa, wb = _f32(a[sl[0]]), _f32(b[sl[1]])
    if min(wa.shape) < 3 or wa.size < 16:
        return zero
    wa = wa - wa.mean()
    wb = wb - wb.mean()
    w = _window(wa.shape)
    try:
        r = phase_cross_correlation(wa * w, wb * w, upsample_factor=up, normalization=None)[0]
    except Exception:
        return zero
    r = np.asarray(r, np.float64)
    if not np.all(np.isfinite(r)) or np.any(np.abs(r) > 1.5):
        return zero
    return r


def _to3(off) -> np.ndarray:
    off = np.asarray(off, np.float64)
    return off if off.size == 3 else np.array([0.0, off[0], off[1]])


def _from3(off3, ndim: int) -> np.ndarray:
    return np.asarray(off3, np.float64) if ndim == 3 else np.asarray(off3, np.float64)[1:]


# ============================================================================ pairwise registration + global optimisation
def pairwise_offsets(ctx, vols: list[tuple[np.ndarray, tuple[int, ...]]], nominal: np.ndarray | None, check_peaks: int = 5, subpixel: bool = True,
                     full_vols: list[np.ndarray] | None = None, progress=(0.15, 0.7)) -> list[dict[str, Any]]:
    """Phase-correlation offsets (z, y, x) of every candidate tile pair: all pairs when the positions are unknown, else
    the pairs whose nominal boxes overlap (searched around the nominal offset). ``vols`` are the (downsampled) estimation
    volumes with their factors; ``full_vols`` the full-resolution ones for the local refinement."""
    n = len(vols)
    ndim = vols[0][0].ndim
    pairs = []
    cands = []
    for i in range(n):
        for j in range(i + 1, n):
            if nominal is None:
                cands.append((i, j, None))
                continue
            shp_i = np.array(full_vols[i].shape if full_vols else vols[i][0].shape, float)
            shp_j = np.array(full_vols[j].shape if full_vols else vols[j][0].shape, float)
            pi, pj = _from3(nominal[i], ndim), _from3(nominal[j], ndim)
            margin = np.maximum(0.5 * np.minimum(shp_i, shp_j), 8.0)
            if np.all(pj + shp_j + margin > pi) and np.all(pi + shp_i + margin > pj):
                cands.append((i, j, pj - pi))
    p0, p1 = progress
    for k, (i, j, exp) in enumerate(cands):
        ctx.check_cancel()
        ctx.progress(p0 + (p1 - p0) * k / max(len(cands), 1), f"Registering pair {k + 1}/{len(cands)}")
        a, fa = vols[i]
        b, fb = vols[j]
        f = np.array(fa, float)
        if exp is not None:
            exp_ds = exp / f
            max_dev = np.maximum(0.5 * np.minimum(np.array(a.shape, float), np.array(b.shape, float)), 8.0)
            off, ncc = find_offset_nd(a, b, n_peaks=check_peaks, expected=exp_ds, max_dev=max_dev)
        else:
            off, ncc = find_offset_nd(a, b, n_peaks=check_peaks)
        if full_vols is not None and any(x > 1 for x in fa):
            off = off * f
            off, ncc = _refine_local(full_vols[i], full_vols[j], off, [max(int(x), 1) for x in fa])
        base = (full_vols[i], full_vols[j]) if full_vols is not None else (a, b)
        if ncc > -1 and subpixel:
            off = off + _subpixel_nd(base[0], base[1], off)
        pairs.append({"i": i, "j": j, "d": _to3(off), "ncc": float(ncc)})
    return pairs


def _components(n: int, edges) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in edges:
        parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def optimise_positions(n: int, pairs: list[dict[str, Any]], nominal: np.ndarray | None, regression_threshold: float = 0.3,
                       max_avg_error: float = 2.5, abs_threshold: float = 3.5) -> tuple[np.ndarray, set[int], list[int], dict[int, float]]:
    """Fiji-style global optimisation: weighted least squares over the pairwise offsets, dropping the worst link while
    the largest displacement error exceeds ``abs_threshold`` or the mean exceeds ``max_avg_error``. Tiles outside the
    largest connected component keep their nominal position (or are dropped when the positions are unknown).

    Returns (positions (n, 3), indices of the used pairs, dropped tiles, displacement error per used pair)."""
    use = [k for k, p in enumerate(pairs) if p["ncc"] >= regression_threshold]
    errors: dict[int, float] = {}
    idx: dict[int, int] = {}
    sol = np.zeros((0, 3))
    while True:
        comps = _components(n, [(pairs[k]["i"], pairs[k]["j"]) for k in use])
        main = max(comps, key=lambda c: (len(c), -min(c)))
        idx = {t: a for a, t in enumerate(main)}
        m = len(main)
        rows, rhs = [], []
        for k in use:
            p = pairs[k]
            if p["i"] in idx and p["j"] in idx:
                row = np.zeros(m)
                wgt = max(float(p["ncc"]), 1e-3)
                row[idx[p["j"]]] = wgt
                row[idx[p["i"]]] = -wgt
                rows.append(row)
                rhs.append(wgt * np.asarray(p["d"], float))
        if nominal is not None:
            for t in main:
                row = np.zeros(m)
                row[idx[t]] = 1e-3
                rows.append(row)
                rhs.append(1e-3 * np.asarray(nominal[t], float))
        else:
            row = np.zeros(m)
            row[idx[main[0]]] = 1.0
            rows.append(row)
            rhs.append(np.zeros(3))
        A = np.array(rows)
        B = np.array(rhs).reshape(len(rows), 3)
        sol = np.linalg.lstsq(A, B, rcond=None)[0]
        errors = {}
        for k in use:
            p = pairs[k]
            if p["i"] in idx and p["j"] in idx:
                errors[k] = float(np.linalg.norm(sol[idx[p["j"]]] - sol[idx[p["i"]]] - np.asarray(p["d"], float)))
        if errors:
            worst = max(errors, key=errors.get)
            if errors[worst] > abs_threshold or float(np.mean(list(errors.values()))) > max_avg_error:
                use.remove(worst)
                continue
        break
    pos = nominal.astype(np.float64).copy() if nominal is not None else np.zeros((n, 3))
    for t, a in idx.items():
        pos[t] = sol[a]
    dropped = [t for t in range(n) if t not in idx] if nominal is None else []
    return pos, set(use), dropped, errors


# ============================================================================ fusion
def _blend_weights_nd(nz: int, h: int, w: int) -> np.ndarray:
    """(Z, Y, X) linear-blending weights: distance to the nearest tile border (z only for real stacks)."""
    w2 = _blend_weights(h, w)[None]
    if nz > 1:
        zw = np.minimum(np.arange(nz) + 1, nz - np.arange(nz)).astype(np.float32)
        return np.minimum(w2, zw[:, None, None])
    return np.broadcast_to(w2, (nz, h, w)).astype(np.float32)


def fuse_tiles(tiles: list[np.ndarray], pos: np.ndarray, fusion: str, ctx=None, progress=(0.75, 0.95),
               masks: list[np.ndarray | None] | None = None) -> np.ndarray:
    """Fuse 5-D tile arrays (T, Z, C, Y, X) placed at integer (z, y, x) offsets ``pos`` (≥ 0) into one float32 canvas.
    ``masks`` (one (Z, Y, X) bool array or None per tile) marks the voxels a tile contributes: the leading plane /
    row / column of a sub-pixel-shifted tile is a blend with the zero fill and is left to the neighbouring tile
    (Fiji fuses a tile on [ceil(min), floor(min + dim − 1)] only), otherwise a dark seam line appears."""
    if not tiles:
        raise ValueError("No tiles to fuse")
    pos = np.asarray(pos, int)
    if masks is None:
        masks = [None] * len(tiles)
    nt = tiles[0].shape[0]
    zc = int(max(p[0] + d.shape[1] for d, p in zip(tiles, pos)))
    hc = int(max(p[1] + d.shape[3] for d, p in zip(tiles, pos)))
    wc = int(max(p[2] + d.shape[4] for d, p in zip(tiles, pos)))
    p0, p1 = progress

    def tick(k: int) -> None:
        if ctx is not None:
            ctx.check_cancel()
            ctx.progress(p0 + (p1 - p0) * k / len(tiles), f"Fusing tile {k + 1}/{len(tiles)}")

    def region(d: np.ndarray, p) -> tuple[slice, ...]:
        return (slice(None), slice(p[0], p[0] + d.shape[1]), slice(None), slice(p[1], p[1] + d.shape[3]), slice(p[2], p[2] + d.shape[4]))

    if fusion == "Overlay into composite":
        nc = sum(d.shape[2] for d in tiles)
        out = np.zeros((nt, zc, nc, hc, wc), np.float32)
        c = 0
        for k, (d, p) in enumerate(zip(tiles, pos)):
            tick(k)
            out[:, p[0]:p[0] + d.shape[1], c:c + d.shape[2], p[1]:p[1] + d.shape[3], p[2]:p[2] + d.shape[4]] = d
            c += d.shape[2]
        return out
    nc = tiles[0].shape[2]
    for d in tiles:
        if d.shape[2] != nc:
            raise ValueError("Tiles must have the same number of channels (or use 'Overlay into composite')")
    if fusion in ("Max intensity", "Min intensity"):
        is_max = fusion == "Max intensity"
        out = np.full((nt, zc, nc, hc, wc), -np.inf if is_max else np.inf, np.float32)
        for k, (d, p, m) in enumerate(zip(tiles, pos, masks)):
            tick(k)
            sl = region(d, p)
            v = d.astype(np.float32)
            if m is not None:
                v = np.where(m[None, :, None], v, -np.inf if is_max else np.inf)
            out[sl] = np.maximum(out[sl], v) if is_max else np.minimum(out[sl], v)
        out[~np.isfinite(out)] = 0
        return out
    if fusion == "Median":
        out = np.zeros((nt, zc, nc, hc, wc), np.float32)
        for z in range(zc):
            if ctx is not None:
                ctx.check_cancel()
                ctx.progress(p0 + (p1 - p0) * z / zc, f"Median of plane {z + 1}/{zc}")
            cover = [(d, p, m) for d, p, m in zip(tiles, pos, masks) if p[0] <= z < p[0] + d.shape[1]]
            if not cover:
                continue
            stack = np.full((len(cover), nt, nc, hc, wc), np.nan, np.float32)
            for k, (d, p, m) in enumerate(cover):
                plane = d[:, z - p[0]]
                if m is not None:
                    plane = np.where(m[z - p[0]][None, None], plane, np.nan)
                stack[k, :, :, p[1]:p[1] + d.shape[3], p[2]:p[2] + d.shape[4]] = plane
            with np.errstate(all="ignore"):
                med = np.nanmedian(stack, axis=0)
            out[:, z] = np.nan_to_num(med)
        return out
    acc = np.zeros((nt, zc, nc, hc, wc), np.float32)
    wsum = np.zeros((zc, hc, wc), np.float32)
    for k, (d, p, m) in enumerate(zip(tiles, pos, masks)):
        tick(k)
        w = _blend_weights_nd(d.shape[1], d.shape[3], d.shape[4]) if fusion == "Linear blending" else np.ones((d.shape[1], d.shape[3], d.shape[4]), np.float32)
        if m is not None:
            w = w * m
        acc[region(d, p)] += d.astype(np.float32) * w[None, :, None]
        wsum[p[0]:p[0] + d.shape[1], p[1]:p[1] + d.shape[3], p[2]:p[2] + d.shape[4]] += w
    return acc / np.where(wsum > 0, wsum, 1.0)[None, :, None]


def _shift_fraction(data: np.ndarray, frac: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Shift a 5-D tile by a sub-pixel (z, y, x) fraction (linear interpolation) for sub-pixel placement.

    Returns (shifted tile, validity mask): the leading plane / row / column along a shifted axis is a blend of the
    zero fill with the first sample and must not be fused (Fiji contributes a tile only from ceil(position) on; the
    far partial pixel is dropped as well). The mask is None when nothing was shifted."""
    frac = np.asarray(frac, float)
    if np.all(np.abs(frac) < 1e-6):
        return data, None
    shift = (0.0, float(frac[0]) if data.shape[1] > 1 else 0.0, 0.0, float(frac[1]), float(frac[2]))
    out = ndi.shift(_f32(data), shift, order=1, mode="constant", cval=0.0)
    valid = np.ones((data.shape[1], data.shape[3], data.shape[4]), bool)  # (Z, Y, X)
    if shift[1] > 1e-6:
        valid[0] = False
    if shift[3] > 1e-6:
        valid[:, 0] = False
    if shift[4] > 1e-6:
        valid[:, :, 0] = False
    return out, valid


def _placed(datas: list[np.ndarray], positions: list[np.ndarray], subpixel: bool) -> tuple[list[np.ndarray], list[np.ndarray | None] | None]:
    """Tiles resampled at their fractional positions (with their validity masks), or as they are."""
    if not subpixel:
        return list(datas), None
    data, masks = [], []
    for d, p in zip(datas, positions):
        shifted, m = _shift_fraction(d, p - np.floor(p))
        data.append(shifted)
        masks.append(m)
    return data, masks


def _finish_stitched(first: LumenImage, fused: np.ndarray, fusion: str, name: str, pos: np.ndarray, kept: list[int]) -> LumenImage:
    res = first.copy(name=name, data=cast_to(fused, first.data.dtype))
    if fusion == "Overlay into composite":
        res.rgb = False
        res.reset_display()
        res.display.mode = "composite" if res.nc > 1 else "single"
    else:
        res.sync_channels()
    res.display.t = res.display.z = res.display.c = 0
    res.overlay = []
    res.metadata["stitch_positions"] = [[float(pos[i, 2]), float(pos[i, 1]), float(pos[i, 0])] for i in kept]
    return res


def _write_output(ctx, res: LumenImage, output: str, output_path: str) -> tuple[list[LumenImage], str]:
    written = ""
    if output != "Open image":
        if not str(output_path or "").strip():
            raise ValueError("Give the output file (output_path) to write the stitched image to")
        from ..io import write_image
        dest = Path(output_path).expanduser()
        dest.parent.mkdir(parents=True, exist_ok=True)
        written = write_image(res, str(dest))
    return ([res] if output != "Write to file" else []), written


# ============================================================================ ops
@op("registration.stitch_collection", "Grid/Collection Stitching (positions, unknown, 3-D)", CAT, kind="image", requires_image=False, preview=False, scope_param=False, roi_aware=False,
    params=[Param("layout", "choice", "Positions from file", choices=LAYOUTS, label="Type", help="Positions from file: a Fiji TileConfiguration.txt; Unknown positions: every pair is tested; Grid: regular grid with an overlap; Open images: the open images"),
            Param("folder", "string", "", help="Folder containing the tiles (Positions from file: empty = the configuration file's folder)"),
            Param("pattern", "string", "*.tif", help="Grid / unknown positions: glob or regular expression of the tile files"),
            Param("layout_file", "string", "", label="Tile configuration file", help="TileConfiguration.txt (empty = <folder>/TileConfiguration.txt); 'file; series; (x, y[, z])' lines"),
            Param("images", "text", "", help="Open images: comma-separated names (empty = all open images)"),
            Param("positions", "text", "", help="Open images: known positions 'x,y; x,y; …' (or 'x,y,z; …'); empty = unknown"),
            Param("grid_x", "int", 2, min=1, label="Grid size X"), Param("grid_y", "int", 2, min=1, label="Grid size Y"),
            Param("overlap_percent", "float", 10.0, min=0, max=90, step=1, label="Tile overlap (%)"), Param("order", "choice", "Row-by-row", choices=ORDERS, label="Tile order"),
            Param("compute_overlap", "bool", True, help="Refine the given / grid positions by phase correlation (always on for unknown positions)"),
            Param("channel", "int", 0, min=0, help="Channel used for the registration (0 = average of all channels); all channels are fused"),
            Param("timelapse", "choice", TIMELAPSE[0], choices=TIMELAPSE, label="Time series", help="Register on the first frame and apply to all, or register every frame separately"),
            Param("check_peaks", "int", 5, min=1, max=50, advanced=True, help="Phase-correlation peaks verified on the real overlap"),
            Param("regression_threshold", "float", 0.3, min=-1, max=1, step=0.05, help="Pairs with a lower overlap correlation are ignored"),
            Param("max_avg_displacement", "float", 2.5, min=0, step=0.5, label="Max/avg displacement threshold (px)", help="Links are dropped one by one while the mean displacement error exceeds this"),
            Param("abs_displacement", "float", 3.5, min=0, step=0.5, label="Absolute displacement threshold (px)", help="Links whose displacement error exceeds this are dropped"),
            Param("subpixel", "bool", True, label="Sub-pixel accuracy", help="Sub-pixel offsets (tiles are resampled at their fractional position)"),
            Param("downsample", "int", 1, min=1, max=16, help="Downsample the tiles by this factor for the registration (refined at full resolution afterwards)"),
            Param("fusion", "choice", "Linear blending", choices=FUSIONS, label="Fusion method", help="'Do not fuse' only writes TileConfiguration.registered.txt and the positions table"),
            Param("output", "choice", "Open image", choices=OUTPUTS), Param("output_path", "string", "", help="File to write the fused image to (tif, png, …)"),
            Param("registered_file", "string", "", help="Where to write the registered tile configuration (empty = <folder>/TileConfiguration.registered.txt; open images: not written)"),
            Param("title", "string", "", help="Name of the stitched image (default: folder name)")],
    description="Grid/Collection stitching with Fiji's options: tile positions from a TileConfiguration.txt, unknown positions (all-pairs phase correlation), a regular grid or the open images; global optimisation with regression / displacement thresholds; 2-D and 3-D tiles (3-D phase correlation), multi-channel and time-series tiles; fusion by linear blending, average, median, max, min, overlay into a composite or not at all (TileConfiguration.registered.txt); sub-pixel accuracy, downsampling for the computation, output to an open image or a file.",
    keywords=["stitching", "grid/collection stitching", "tile configuration", "unknown positions", "mosaic", "3d stitching", "fuse", "tiles"])
def stitch_collection(ctx, layout="Positions from file", folder="", pattern="*.tif", layout_file="", images="", positions="", grid_x=2, grid_y=2, overlap_percent=10.0,
                      order="Row-by-row", compute_overlap=True, channel=0, timelapse=TIMELAPSE[0], check_peaks=5, regression_threshold=0.3, max_avg_displacement=2.5,
                      abs_displacement=3.5, subpixel=True, downsample=1, fusion="Linear blending", output="Open image", output_path="", registered_file="", title=""):
    col = load_collection(ctx, layout, folder, pattern, layout_file, images, positions, grid_x, grid_y, overlap_percent, order)
    tiles, names, series, nominal, root = col["tiles"], col["names"], col["series"], col["nominal"], col["root"]
    n = len(tiles)
    is3d = any(t.nz > 1 for t in tiles)
    nt = tiles[0].nt
    per_frame = timelapse == TIMELAPSE[1] and nt > 1
    frames = list(range(nt)) if per_frame else [0]
    refine = compute_overlap or nominal is None
    pair_rows: list[dict[str, Any]] = []
    pos_rows: list[list[Any]] = []
    frame_results: list[tuple[np.ndarray, list[int]]] = []
    all_dropped: set[int] = set()
    for fi, fr in enumerate(frames):
        base = 0.15 + 0.6 * fi / len(frames)
        span = 0.6 / len(frames)
        used: set[int] = set()
        errors: dict[int, float] = {}
        pairs: list[dict[str, Any]] = []
        dropped: list[int] = []
        if refine:
            ctx.progress(base, "Preparing the registration" + (f" of frame {fr + 1}" if per_frame else ""))
            vols = [_est_volume(t, channel, fr, is3d, downsample) for t in tiles]
            full = [_est_volume(t, channel, fr, is3d, 1)[0] for t in tiles] if downsample > 1 else None
            pairs = pairwise_offsets(ctx, vols, nominal, check_peaks, subpixel, full, progress=(base, base + span * 0.9))
            if not any(p["ncc"] > -1 for p in pairs):
                raise ValueError("No overlap found between any two tiles")
            pos, used, dropped, errors = optimise_positions(n, pairs, nominal, regression_threshold, max_avg_displacement, abs_displacement)
            all_dropped.update(dropped)
        else:
            pos = nominal.astype(np.float64).copy()
        kept = [i for i in range(n) if i not in dropped]
        pos = pos - pos[kept].min(axis=0)
        # float noise of exact integer offsets (119.99999999999999) would otherwise become a full-pixel shift
        pos = np.where(np.abs(pos - np.rint(pos)) < 1e-6, np.rint(pos), pos)
        if not subpixel:
            pos = np.rint(pos)
        frame_results.append((pos, kept))
        for k, p in enumerate(pairs):
            row = {"Label": f"{names[p['i']]} → {names[p['j']]}", "Tile A": p["i"] + 1, "Tile B": p["j"] + 1, "dx": float(p["d"][2]), "dy": float(p["d"][1])}
            if is3d:
                row["dz"] = float(p["d"][0])
            row.update({"Correlation": p["ncc"], "Error": errors.get(k, float("nan")), "Used": k in used})
            if per_frame:
                row["Frame"] = fr + 1
            pair_rows.append(row)
        for i in kept:
            r = [i + 1, names[i], round(float(pos[i, 2]), 3), round(float(pos[i, 1]), 3)] + ([round(float(pos[i, 0]), 3)] if is3d else []) + ([fr + 1] if per_frame else [])
            pos_rows.append(r)
    cols = ["Tile", "File", "X", "Y"] + (["Z"] if is3d else []) + (["Frame"] if per_frame else [])
    table = {"name": "Tile positions", "columns": cols, "rows": pos_rows}
    # registered tile configuration (first frame's positions)
    pos0, kept0 = frame_results[0]
    reg_path = ""
    if str(registered_file or "").strip() or root is not None:
        target = Path(registered_file).expanduser() if str(registered_file or "").strip() else root / REGISTERED_NAME
        if target.is_dir():
            target = target / REGISTERED_NAME
        reg_path = write_tile_configuration(target, [(names[i], series[i], (pos0[i, 2], pos0[i, 1], pos0[i, 0])) for i in kept0], 3 if is3d else 2)
    n_used = sum(1 for r in pair_rows if r["Used"])
    msg = f"Registered {len(kept0)} tiles ({layout.lower()}{', 3-D' if is3d else ''}{', per frame' if per_frame else ''})"
    if pair_rows:
        msg += f"; {n_used}/{len(pair_rows)} tile pairs used"
    if all_dropped:
        msg += f"; dropped (unconnected): {', '.join(names[i] for i in sorted(all_dropped))}"
    if reg_path:
        msg += f"; wrote {Path(reg_path).name}"
    if fusion == "Do not fuse":
        return OpResult(results=pair_rows or None, table=table, message=msg, log=msg)
    first = tiles[kept0[0]]
    if per_frame:
        canvases = []
        for fi, (pos, kept) in enumerate(frame_results):
            ctx.check_cancel()
            ipos = np.floor(pos[kept]).astype(int)
            data, masks = _placed([tiles[i].data[fi:fi + 1] for i in kept], [pos[i] for i in kept], subpixel)
            canvases.append(fuse_tiles(data, ipos, fusion, ctx, progress=(0.75 + 0.2 * fi / nt, 0.75 + 0.2 * (fi + 1) / nt), masks=masks))
        zc, cc = max(c.shape[1] for c in canvases), max(c.shape[2] for c in canvases)
        hc, wc = max(c.shape[3] for c in canvases), max(c.shape[4] for c in canvases)
        fused = np.zeros((nt, zc, cc, hc, wc), np.float32)
        for fi, c in enumerate(canvases):
            fused[fi, :c.shape[1], :c.shape[2], :c.shape[3], :c.shape[4]] = c[0]
    else:
        ipos = np.floor(pos0[kept0]).astype(int)
        data, masks = _placed([tiles[i].data for i in kept0], [pos0[i] for i in kept0], subpixel)
        fused = fuse_tiles(data, ipos, fusion, ctx, masks=masks)
    name = title or (f"Stitched {root.name}" if root is not None else "Stitched")
    res = _finish_stitched(first, fused, fusion, name, pos0, kept0)
    ctx.progress(0.97, "Writing" if output != "Open image" else "Done")
    new_images, written = _write_output(ctx, res, output, output_path)
    msg += f"; fused ({fusion.lower()}) into {res.width} × {res.height}" + (f" × {res.nz}" if res.nz > 1 else "")
    if written:
        msg += f" → {written}"
    return OpResult(new_images=new_images, results=pair_rows or None, table=table, message=msg, log=msg)


# ============================================================================ elastic montaging
def _mesh(h: int, w: int, spacing: float) -> tuple[int, int, float, float]:
    gy = max(2, int(math.ceil((h - 1) / max(spacing, 1.0))) + 1)
    gx = max(2, int(math.ceil((w - 1) / max(spacing, 1.0))) + 1)
    return gy, gx, (h - 1) / (gy - 1), (w - 1) / (gx - 1)


def _bilinear(x: np.ndarray, y: np.ndarray, gy: int, gx: int, sy: float, sx: float) -> tuple[np.ndarray, np.ndarray]:
    """Node indices (4, N) and weights (4, N) of bilinear interpolation on a tile's mesh at points (x, y)."""
    fx = np.clip(np.asarray(x, float) / sx, 0, gx - 1 - 1e-9)
    fy = np.clip(np.asarray(y, float) / sy, 0, gy - 1 - 1e-9)
    ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
    tx, ty = fx - ix, fy - iy
    idx = np.stack([iy * gx + ix, iy * gx + ix + 1, (iy + 1) * gx + ix, (iy + 1) * gx + ix + 1])
    wts = np.stack([(1 - ty) * (1 - tx), (1 - ty) * tx, ty * (1 - tx), ty * tx])
    return idx, wts


def _dense_field(nodes: np.ndarray, h: int, w: int, sy: float, sx: float) -> np.ndarray:
    """(2, H, W) [dy, dx] forward displacement from a (gy, gx, 2) node grid (bilinear)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    coords = [yy / sy, xx / sx]
    return np.stack([ndi.map_coordinates(nodes[..., 1], coords, order=1, mode="nearest"), ndi.map_coordinates(nodes[..., 0], coords, order=1, mode="nearest")]).astype(np.float32)


def _warp_forward(img: LumenImage, field: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Warp every plane of a tile with a forward displacement field (pixel x moves to x + D(x)); returns the warped
    5-D data on a canvas enlarged by a margin, its blending weights (distance to the border × validity) and the margin."""
    h, w = field.shape[1:]
    m = int(math.ceil(float(np.abs(field).max()))) + 1
    hh, ww = h + 2 * m, w + 2 * m
    yy, xx = (a.astype(np.float64) for a in np.mgrid[0:hh, 0:ww])
    ty, tx = yy - m, xx - m  # output pixel in tile coordinates
    sy, sx = ty.copy(), tx.copy()
    for _ in range(3):  # fixed point of x = y − D(x)
        dy = ndi.map_coordinates(field[0], [sy, sx], order=1, mode="nearest")
        dx = ndi.map_coordinates(field[1], [sy, sx], order=1, mode="nearest")
        sy, sx = ty - dy, tx - dx
    coords = [sy, sx]
    valid = ((sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)).astype(np.float32)
    weights = ndi.map_coordinates(_blend_weights(h, w), coords, order=1, mode="nearest") * valid
    out = np.zeros((img.nt, img.nz, img.nc, hh, ww), np.float32)
    for t, z, c in img.iter_planes():
        out[t, z, c] = ndi.map_coordinates(_f32(img.data[t, z, c]), coords, order=order, mode="constant", cval=0.0) * valid
    return out, weights.astype(np.float32), m


@op("registration.elastic_montage", "Elastic Montaging", CAT, kind="image", requires_image=False, preview=False, scope_param=False, roi_aware=False,
    params=[Param("layout", "choice", "Open images", choices=LAYOUTS, label="Tiles", help="Open images (unknown positions unless given), a TileConfiguration.txt, a folder with unknown positions or a grid"),
            Param("folder", "string", ""), Param("pattern", "string", "*.tif"), Param("layout_file", "string", "", label="Tile configuration file"),
            Param("images", "text", "", help="Open images: comma-separated names (empty = all open images)"),
            Param("positions", "text", "", help="Open images: approximate positions 'x,y; x,y; …' (empty = unknown)"),
            Param("grid_x", "int", 2, min=1), Param("grid_y", "int", 2, min=1), Param("overlap_percent", "float", 10.0, min=0, max=90, step=1), Param("order", "choice", "Row-by-row", choices=ORDERS),
            Param("channel", "int", 0, min=0, help="Channel used for the alignment (0 = average of all channels)"),
            Param("method", "choice", "TV-L1", choices=FLOW_METHODS, help="Optical flow between overlapping tiles"),
            Param("downscale", "int", 2, min=1, max=8, help="Flow estimated at 1/n resolution (smoother, faster)"),
            Param("smoothing", "float", 3.0, min=0, max=50, step=0.5, label="Flow smoothing (px)"),
            Param("mesh_spacing", "int", 32, min=4, max=512, label="Spring mesh spacing (px)", help="Node spacing of the per-tile deformation mesh"),
            Param("stiffness", "float", 1.0, min=0, max=100, step=0.5, help="Weight of the mesh springs (higher = more rigid tiles)"),
            Param("min_correlation", "float", 0.3, min=-1, max=1, step=0.05, help="Tile pairs with a lower overlap correlation are not aligned"),
            Param("max_displacement", "float", 0.0, min=0, step=1, help="Discard flow vectors longer than this (0 = no limit)"),
            Param("interpolation", "choice", "Linear", choices=INTERP),
            Param("output", "choice", "Open image", choices=OUTPUTS), Param("output_path", "string", ""), Param("title", "string", "Elastic montage")],
    description="Elastic montaging: the tiles are placed by phase correlation + global optimisation, every overlapping pair is aligned with a dense optical-flow field, the per-tile deformations are relaxed on spring meshes (least squares: the matches pull the meshes together, the springs keep each tile smooth) and the warped tiles are blended linearly into one montage. The 'Montage' table lists the positions and deformations per tile; pair residuals before / after go to the Results table.",
    keywords=["elastic montaging", "montage", "spring mesh", "elastic stitching", "non-rigid stitching", "tiles", "optical flow"])
def elastic_montage(ctx, layout="Open images", folder="", pattern="*.tif", layout_file="", images="", positions="", grid_x=2, grid_y=2, overlap_percent=10.0, order="Row-by-row",
                    channel=0, method="TV-L1", downscale=2, smoothing=3.0, mesh_spacing=32, stiffness=1.0, min_correlation=0.3, max_displacement=0.0,
                    interpolation="Linear", output="Open image", output_path="", title="Elastic montage"):
    col = load_collection(ctx, layout, folder, pattern, layout_file, images, positions, grid_x, grid_y, overlap_percent, order)
    tiles, names, nominal, root = col["tiles"], col["names"], col["nominal"], col["root"]
    n = len(tiles)
    nc = tiles[0].nc
    for t, nm in zip(tiles, names):
        if t.nc != nc or t.nz != tiles[0].nz:
            raise ValueError(f"{nm} has {t.nc} channels / {t.nz} slices, {names[0]} has {nc} / {tiles[0].nz}: all tiles need the same layout")
    # 1. translations (2-D, on the middle slice)
    vols = [_est_volume(t, channel, 0, False, 1) for t in tiles]
    planes = [v[0] for v in vols]
    pairs = pairwise_offsets(ctx, vols, nominal, 5, False, None, progress=(0.15, 0.4))
    if not any(p["ncc"] > -1 for p in pairs):
        raise ValueError("No overlap found between any two tiles")
    pos, used, dropped, _ = optimise_positions(n, pairs, nominal, min_correlation, 1e9, 1e9)
    kept = [i for i in range(n) if i not in dropped]
    pos = np.rint(pos - pos[kept].min(axis=0)).astype(int)
    # 2. meshes
    meshes = [_mesh(t.height, t.width, float(mesh_spacing)) for t in tiles]
    offsets = np.cumsum([0] + [gy * gx for gy, gx, _, _ in meshes])
    n_nodes = int(offsets[-1])
    rows_i: list[np.ndarray] = []
    rows_j: list[np.ndarray] = []
    rows_v: list[np.ndarray] = []
    rhs: list[np.ndarray] = []
    row = 0
    pair_rows = []
    order_i = ORDER_MAP[interpolation]
    stride = max(4, int(mesh_spacing) // 4)
    overl = [k for k in sorted(used) if pairs[k]["i"] in kept and pairs[k]["j"] in kept]
    for q, k in enumerate(overl):
        ctx.check_cancel()
        ctx.progress(0.4 + 0.3 * q / max(len(overl), 1), f"Aligning pair {q + 1}/{len(overl)}")
        p = pairs[k]
        i, j = p["i"], p["j"]
        yi, xi = int(pos[i, 1]), int(pos[i, 2])
        yj, xj = int(pos[j, 1]), int(pos[j, 2])
        hi, wi = planes[i].shape
        hj, wj = planes[j].shape
        y0, y1 = max(yi, yj), min(yi + hi, yj + hj)
        x0, x1 = max(xi, xj), min(xi + wi, xj + wj)
        if y1 - y0 < 16 or x1 - x0 < 16:
            continue
        a = planes[i][y0 - yi:y1 - yi, x0 - xi:x1 - xi]
        b = planes[j][y0 - yj:y1 - yj, x0 - xj:x1 - xj]
        ds = max(1, min(int(downscale), min(a.shape) // 8))
        flow = estimate_flow(a, b, method=method, downscale=ds, smoothing=float(smoothing))
        rr, cc = np.mgrid[0:a.shape[0]:stride, 0:a.shape[1]:stride]
        v, u = flow[0][rr, cc], flow[1][rr, cc]
        ok = np.isfinite(u) & np.isfinite(v)
        if max_displacement > 0:
            ok &= np.hypot(u, v) <= max_displacement
        if not ok.any():
            continue
        rr, cc, u, v = rr[ok], cc[ok], u[ok], v[ok]
        gyi, gxi, syi, sxi = meshes[i]
        gyj, gxj, syj, sxj = meshes[j]
        idx_i, w_i = _bilinear(x0 - xi + cc, y0 - yi + rr, gyi, gxi, syi, sxi)
        idx_j, w_j = _bilinear(x0 - xj + cc + u, y0 - yj + rr + v, gyj, gxj, syj, sxj)
        wgt = max(float(p["ncc"]), 0.1)
        m = len(rr)
        r_idx = np.arange(row, row + m)
        for a_ in range(4):
            rows_i.append(r_idx)
            rows_j.append(idx_j[a_] + offsets[j])
            rows_v.append(wgt * w_j[a_])
            rows_i.append(r_idx)
            rows_j.append(idx_i[a_] + offsets[i])
            rows_v.append(-wgt * w_i[a_])
        rhs.append(wgt * np.stack([-u, -v], axis=1))
        row += m
        pair_rows.append({"Label": f"{names[i]} → {names[j]}", "Tile A": i + 1, "Tile B": j + 1, "dx": float(p["d"][2]), "dy": float(p["d"][1]), "Correlation": p["ncc"],
                          "Matches": int(m), "Residual before": float(np.hypot(u, v).mean()), "_rows": (row - m, row)})
    n_constraints = row
    if n_constraints == 0:
        raise ValueError("No overlapping tile pairs to align elastically (check the positions / min correlation)")
    # springs between neighbouring nodes and a weak prior (gauge; unconstrained tiles stay put)
    for ti, (gy, gx, _, _) in enumerate(meshes):
        if ti not in kept:
            continue
        base = offsets[ti]
        node = np.arange(gy * gx).reshape(gy, gx)
        links = np.concatenate([np.stack([node[:, :-1].ravel(), node[:, 1:].ravel()], axis=1), np.stack([node[:-1, :].ravel(), node[1:, :].ravel()], axis=1)])
        m = len(links)
        r_idx = np.arange(row, row + m)
        rows_i += [r_idx, r_idx]
        rows_j += [links[:, 0] + base, links[:, 1] + base]
        rows_v += [np.full(m, float(stiffness)), np.full(m, -float(stiffness))]
        rhs.append(np.zeros((m, 2)))
        row += m
        r_idx = np.arange(row, row + gy * gx)
        rows_i.append(r_idx)
        rows_j.append(np.arange(gy * gx) + base)
        rows_v.append(np.full(gy * gx, 1e-3))
        rhs.append(np.zeros((gy * gx, 2)))
        row += gy * gx
    ctx.progress(0.72, "Relaxing the spring meshes")
    A = sparse.coo_matrix((np.concatenate(rows_v), (np.concatenate(rows_i), np.concatenate(rows_j))), shape=(row, n_nodes)).tocsr()
    B = np.concatenate(rhs)
    ata = (A.T @ A).tocsc() + sparse.identity(n_nodes, format="csc") * 1e-9
    sol = np.asarray(spsolve(ata, A.T @ B)).reshape(n_nodes, 2)
    resid = A[:n_constraints] @ sol - B[:n_constraints]
    for r in pair_rows:
        a0, a1 = r.pop("_rows")
        r["Residual after"] = float(np.hypot(resid[a0:a1, 0], resid[a0:a1, 1]).mean())
    # 3. warp + blend
    fields = {}
    for ti in kept:
        gy, gx, sy, sx = meshes[ti]
        fields[ti] = _dense_field(sol[offsets[ti]:offsets[ti + 1]].reshape(gy, gx, 2), tiles[ti].height, tiles[ti].width, sy, sx)
    margin = {ti: int(math.ceil(float(np.abs(fields[ti]).max()))) + 1 for ti in kept}
    oy = min(int(pos[ti, 1]) - margin[ti] for ti in kept)
    ox = min(int(pos[ti, 2]) - margin[ti] for ti in kept)
    hc = max(int(pos[ti, 1]) + tiles[ti].height + margin[ti] for ti in kept) - oy
    wc = max(int(pos[ti, 2]) + tiles[ti].width + margin[ti] for ti in kept) - ox
    first = tiles[kept[0]]
    acc = np.zeros((first.nt, first.nz, nc, hc, wc), np.float32)
    wsum = np.zeros((hc, wc), np.float32)
    mont_rows = []
    for q, ti in enumerate(kept):
        ctx.check_cancel()
        ctx.progress(0.75 + 0.2 * q / len(kept), f"Warping tile {q + 1}/{len(kept)}")
        warped, weights, m = _warp_forward(tiles[ti], fields[ti], order_i)
        y, x = int(pos[ti, 1]) - m - oy, int(pos[ti, 2]) - m - ox
        acc[:, :, :, y:y + warped.shape[3], x:x + warped.shape[4]] += warped * weights
        wsum[y:y + warped.shape[3], x:x + warped.shape[4]] += weights
        mag = np.hypot(fields[ti][0], fields[ti][1])
        mont_rows.append([ti + 1, names[ti], int(pos[ti, 2]) - ox, int(pos[ti, 1]) - oy, round(float(mag.mean()), 3), round(float(mag.max()), 3)])
    fused = acc / np.where(wsum > 0, wsum, 1.0)
    res = first.copy(name=title or "Elastic montage", data=cast_to(fused, first.data.dtype))
    res.sync_channels()
    res.display.t = res.display.z = res.display.c = 0
    res.overlay = []
    res.metadata["stitch_positions"] = [[r[2], r[3]] for r in mont_rows]
    table = {"name": "Montage", "columns": ["Tile", "File", "X", "Y", "Mean deformation", "Max deformation"], "rows": mont_rows}
    new_images, written = _write_output(ctx, res, output, output_path)
    before = float(np.mean([r["Residual before"] for r in pair_rows]))
    after = float(np.mean([r["Residual after"] for r in pair_rows]))
    msg = f"Elastic montage of {len(kept)} tiles ({len(pair_rows)} overlapping pairs) into {res.width} × {res.height}: mean pair residual {before:.2f} → {after:.2f} px"
    if dropped:
        msg += f"; dropped (unconnected): {', '.join(names[i] for i in dropped)}"
    if written:
        msg += f" → {written}"
    return OpResult(new_images=new_images, results=pair_rows, table=table, message=msg, log=msg)
