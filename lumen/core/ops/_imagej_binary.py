"""Pixel-exact ports of ImageJ 1.54p's Process ▸ Binary algorithms.

* ``ij.process.ByteProcessor.erode / dilate`` with ``Prefs.binaryCount`` and ``Prefs.padEdges``
  (:func:`ij_erode`, :func:`ij_dilate`): a foreground pixel is removed when at least *count* of its 8 neighbours
  are background, a background pixel added when at least *count* are foreground; pixels outside the image count
  as background (as foreground for eroding with 'pad edges').
* ``ij.plugin.filter.Binary.outline`` (:func:`ij_outline`): with a black background ImageJ inverts the image,
  outlines and inverts back – but ``ByteProcessor.getPixelsCopy()`` hands ``BinaryProcessor.process`` the
  *pre-inversion* snapshot (``setSnapshotCopyMode(true)``), so the result is the one-pixel ring of background
  around every object rather than the object's boundary.  Reproduced as ImageJ does it.
* ``ij.process.BinaryProcessor.skeletonize`` (:func:`ij_skeletonize`): the lookup-table thinning with ImageJ's two
  tables (decoded from ij-1.54p.jar), pairs of passes until nothing changes.
* ``ij.plugin.filter.EDM`` (:func:`ij_distance_map`, :func:`ij_ultimate_points`, :func:`ij_voronoi`,
  :func:`ij_watershed`): float EDM, ``MaximumFinder`` with tolerance 0.5 (SINGLE_POINTS / SEGMENTED, EDM mode),
  8-bit 'Overwrite' output = ``(byte)(v + 0.5f)``.
* ``ij.plugin.filter.MaximumFinder`` (:func:`find_maxima`): getSortedMaxPoints, analyzeAndMarkMaxima (with
  ``trueEdmHeight`` and the sorting-error restart), make8bit, cleanupMaxima, the level-by-level
  ``watershedSegment`` with its fate table and direction sequence, cleanupExtraLines and watershedPostProcess.
  Float arithmetic is done in float32 where ImageJ uses ``float`` so that comparisons and ties come out the same.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

_FULL = np.ones((3, 3), np.int32)


# ====================================================================== erode / dilate / open / close
def _neighbour_count(mask: np.ndarray, outside: bool) -> np.ndarray:
    """Number of the 8 neighbours of every pixel that are True; pixels outside the image count as ``outside``."""
    m = mask.astype(np.int32)
    s = ndi.convolve(m, _FULL, mode="constant", cval=1 if outside else 0)
    return s - m


def ij_erode(b: np.ndarray, count: int = 1, pad_edges: bool = False) -> np.ndarray:
    """``ByteProcessor.erode(count, background)``: foreground pixels with ≥ count background neighbours become
    background; outside the image is background unless ``pad_edges`` (Prefs.padEdges)."""
    count = max(int(count), 1)
    n_bg = _neighbour_count(~b, outside=not pad_edges)
    return b & ~(n_bg >= count)


def ij_dilate(b: np.ndarray, count: int = 1) -> np.ndarray:
    """``ByteProcessor.dilate(count, background)``: background pixels with ≥ count foreground neighbours become
    foreground; outside the image is background."""
    count = max(int(count), 1)
    n_fg = _neighbour_count(b, outside=False)
    return b | (n_fg >= count)


def ij_open(b: np.ndarray, iterations: int = 1, count: int = 1, pad_edges: bool = False) -> np.ndarray:
    for _ in range(max(int(iterations), 1)):
        b = ij_erode(b, count, pad_edges)
    for _ in range(max(int(iterations), 1)):
        b = ij_dilate(b, count)
    return b


def ij_close(b: np.ndarray, iterations: int = 1, count: int = 1, pad_edges: bool = False) -> np.ndarray:
    for _ in range(max(int(iterations), 1)):
        b = ij_dilate(b, count)
    for _ in range(max(int(iterations), 1)):
        b = ij_erode(b, count, pad_edges)
    return b


# ====================================================================== outline
def ij_outline(b: np.ndarray, black_background: bool = True) -> np.ndarray:
    """``Binary.outline``: with a black background (Process ▸ Binary ▸ Options, Fiji's default) the result is the
    8-connected ring of background pixels around each object (see the module docstring); with a white background
    (inverting LUT) it is the object's own 8-connected boundary.  Border pixels are not processed and keep their
    value, as ``BinaryProcessor.process`` only visits the interior."""
    out = b.copy()
    if b.shape[0] < 3 or b.shape[1] < 3:
        return out
    if black_background:
        inner = (~b) & (_neighbour_count(b, outside=False) > 0)
    else:
        inner = b & (_neighbour_count(~b, outside=False) > 0)
    out[1:-1, 1:-1] = inner[1:-1, 1:-1]
    return out


def ij_outline_white_bg(b: np.ndarray) -> np.ndarray:
    """``Binary.outline`` with 'Black background' off on a 0/255 mask that has a normal (non-inverting) LUT – ImageJ
    then takes 255 for the background colour: the objects and the far background stay 255 and only the ring of
    background pixels around the objects becomes 0 (verified against ImageJ 1.54p); border pixels keep their value."""
    out = b.copy()
    if b.shape[0] < 3 or b.shape[1] < 3:
        return out
    inner = ~((~b) & (_neighbour_count(b, outside=False) > 0))
    out[1:-1, 1:-1] = inner[1:-1, 1:-1]
    return out


# ====================================================================== skeletonize
# ij.process.BinaryProcessor.table / table2 (ij-1.54p.jar): entry = 1 → delete on even passes, 2 → odd passes,
# 3 → both; index = p1 + 2·p2 + 4·p3 + 8·p6 + 16·p9 + 32·p8 + 64·p7 + 128·p4 (clockwise from the top-left)
_SKEL_TABLE = np.array([
    0, 0, 0, 0, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 3, 0, 3, 3,
    0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
    0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0], np.uint8)
_SKEL_TABLE2 = np.zeros(256, np.uint8)
for _i, _v in {3: 1, 24: 2, 26: 2, 27: 2, 46: 2, 48: 2, 58: 2, 129: 1, 154: 2, 184: 2, 226: 1}.items():
    _SKEL_TABLE2[_i] = _v


def _thin_pass(p: np.ndarray, pass_: int, table: np.ndarray) -> tuple[np.ndarray, int]:
    """One ``BinaryProcessor.thin`` pass over the interior (decisions from the state at the start of the pass)."""
    f = p
    idx = (f[:-2, :-2].astype(np.int32) * 1 + f[:-2, 1:-1] * 2 + f[:-2, 2:] * 4 + f[1:-1, 2:] * 8 + f[2:, 2:] * 16
           + f[2:, 1:-1] * 32 + f[2:, :-2] * 64 + f[1:-1, :-2] * 128)
    code = table[idx]
    if pass_ & 1:
        rem = (code == 2) | (code == 3)
    else:
        rem = (code == 1) | (code == 3)
    rem &= f[1:-1, 1:-1]
    out = p.copy()
    out[1:-1, 1:-1] &= ~rem
    return out, int(rem.sum())


def ij_skeletonize(b: np.ndarray) -> np.ndarray:
    """``BinaryProcessor.skeletonize``: the image is padded by one background pixel (``expand`` when objects touch
    the border; otherwise the frame ImageJ paints is already background), thinned with pairs of passes over
    ``table`` until nothing changes, then pairs of passes over ``table2`` ('stuck' pixels), and cropped."""
    p = np.pad(b.astype(bool), 1, constant_values=False)
    pass_ = 0
    for table in (_SKEL_TABLE, _SKEL_TABLE2):
        while True:
            p, n1 = _thin_pass(p, pass_, table)
            pass_ += 1
            p, n2 = _thin_pass(p, pass_, table)
            pass_ += 1
            if n1 + n2 <= 0:
                break
    return p[1:-1, 1:-1]


# ====================================================================== EDM
def edm_float(b: np.ndarray) -> np.ndarray:
    """``EDM.makeFloatEDM(ip, background, edgesAreBackground=false)``: the Euclidean distance of every foreground
    pixel to the nearest background pixel as float32 (``(float) Math.sqrt`` of the exact squared distance; pixels
    outside the image are not background)."""
    b = b.astype(bool)
    if not b.any():
        return np.zeros(b.shape, np.float32)
    if b.all():
        return np.full(b.shape, np.sqrt(np.float32(np.finfo(np.float32).max)))  # no background anywhere: sqrt(Float.MAX_VALUE)
    _, (iy, ix) = ndi.distance_transform_edt(b, return_indices=True)
    yy, xx = np.indices(b.shape)
    d2 = (yy - iy).astype(np.int64) ** 2 + (xx - ix).astype(np.int64) ** 2
    return np.sqrt(d2.astype(np.float64)).astype(np.float32)


def edm_to_byte(edm: np.ndarray) -> np.ndarray:
    """``ByteProcessor.setPixels(0, FloatProcessor)``: ``(byte)(v + 0.5f)`` clipped to 0…255."""
    v = edm.astype(np.float32) + np.float32(0.5)
    v = np.minimum(np.maximum(v, np.float32(0)), np.float32(255))
    return np.floor(v).astype(np.uint8)


# ====================================================================== MaximumFinder
MAXIMUM, LISTED, PROCESSED, MAX_AREA, EQUAL, MAX_POINT, ELIMINATED = 1, 2, 4, 8, 16, 32, 64
SINGLE_POINTS, IN_TOLERANCE, SEGMENTED = 0, 1, 2
_SQRT2 = np.float32(1.4142135)
_HALF_SQRT2 = np.float32(0.70710677)
_DIR_X = (0, 1, 1, 1, 0, -1, -1, -1)
_DIR_Y = (-1, -1, 0, 1, 1, 1, 0, -1)
_DIRECTION_SEQUENCE = (7, 3, 1, 5, 0, 4, 2, 6)


def true_edm_height(ip: np.ndarray) -> np.ndarray:
    """``MaximumFinder.trueEdmHeight`` for every pixel (float32 arithmetic like the Java code)."""
    v = ip.astype(np.float32, copy=False)
    out = v.copy()
    if v.shape[0] < 3 or v.shape[1] < 3:
        return out
    c = v[1:-1, 1:-1]
    n_, s_ = v[:-2, 1:-1], v[2:, 1:-1]
    ne, sw = v[:-2, 2:], v[2:, :-2]
    e_, w_ = v[1:-1, 2:], v[1:-1, :-2]
    se, nw = v[2:, 2:], v[:-2, :-2]
    true_h = c + _HALF_SQRT2
    ridge = np.zeros(c.shape, bool)
    for d, (v1, v2) in enumerate(((n_, s_), (ne, sw), (e_, w_), (se, nw))):
        cond = (c >= v1) & (c >= v2)
        h = np.where(cond, (v1 + v2) / np.float32(2), np.minimum(v1, v2)).astype(np.float32)
        ridge |= cond
        h = h + (np.float32(1) if d % 2 == 0 else _SQRT2)
        true_h = np.where(true_h > h, h, true_h).astype(np.float32)
    res = np.where(ridge, true_h, c)
    out[1:-1, 1:-1] = np.where(c == 0, c, res)
    return out


def _shift_max(a: np.ndarray, fill: float) -> list[np.ndarray]:
    """The 8 neighbour planes of ``a`` (direction order 0…7), out-of-image = fill."""
    p = np.pad(a, 1, constant_values=fill)
    H, W = a.shape
    return [p[1 + dy:1 + dy + H, 1 + dx:1 + dx + W] for dx, dy in zip(_DIR_X, _DIR_Y)]


def _sorted_max_points(ip: np.ndarray, vt: np.ndarray, exclude_edges_now: bool, is_edm: bool, global_min: np.float32,
                       global_max: np.float32, threshold: float | None) -> tuple[np.ndarray, np.ndarray]:
    """``getSortedMaxPoints``: (types with the MAXIMUM flag, pixel offsets sorted like ImageJ's long keys)."""
    H, W = ip.shape
    v = ip
    vtrue = vt if is_edm else v
    cand = v != global_min
    if exclude_edges_now:
        cand[0, :] = cand[-1, :] = False
        cand[:, 0] = cand[:, -1] = False
    if threshold is not None:
        cand &= ~(v.astype(np.float64) < threshold)
    neg = np.float32(-np.inf)
    higher = np.zeros(ip.shape, bool)
    for vn, vnt in zip(_shift_max(v, neg), _shift_max(vtrue, neg)):
        higher |= (vn > v) & (vnt > vtrue)
    is_max = cand & ~higher
    types = np.where(is_max, MAXIMUM, 0).astype(np.uint8).ravel()
    offsets = np.flatnonzero(is_max.ravel())
    if offsets.size == 0:
        return types, offsets
    v_factor = np.float32(2e9 / (float(global_max) - float(global_min)))
    f_value = vtrue.ravel()[offsets].astype(np.float32)
    i_value = np.trunc(((f_value - global_min) * v_factor).astype(np.float64)).astype(np.int64)
    order = np.lexsort((offsets, i_value))
    return types, offsets[order]


def _analyze_and_mark_maxima(ip: np.ndarray, types: np.ndarray, max_points: np.ndarray, exclude_edges_now: bool, is_edm: bool,
                             tolerance: float, strict: bool, max_sorting_error: np.float32, vt: np.ndarray) -> None:
    """``analyzeAndMarkMaxima``: flood every maximum (highest first) within the tolerance, mark MAX_AREA / MAX_POINT."""
    H, W = ip.shape
    n = H * W
    vals = (vt if is_edm else ip).ravel().tolist()  # exact float32 values as Python floats
    edm = ip.ravel().tolist() if is_edm else None
    tol32 = np.float32(tolerance)
    t = types  # uint8 array, modified in place
    n_max = len(max_points)
    xmax, ymax = W - 1, H - 1
    dir_off = (-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1)
    for i_max in range(n_max - 1, -1, -1):
        offset0 = int(max_points[i_max])
        if t[offset0] & PROCESSED:
            continue
        x0, y0 = offset0 % W, offset0 // W
        v0 = vals[offset0]
        while True:  # do … while (sortingError)
            v0_32 = np.float32(v0)
            hi = float(v0_32 + max_sorting_error)
            lo = float(v0_32 - tol32)
            plist = [offset0]
            t[offset0] |= EQUAL | LISTED
            list_i = 0
            is_edge_max = x0 == 0 or x0 == xmax or y0 == 0 or y0 == ymax
            sorting_error = False
            max_possible = True
            x_equal, y_equal, n_equal = float(x0), float(y0), 1
            while list_i < len(plist):
                offset = plist[list_i]
                x, y = offset % W, offset // W
                is_inner = y != 0 and y != ymax and x != 0 and x != xmax
                for d in range(8):
                    if not is_inner:
                        if d == 0 and not (y > 0):
                            continue
                        if d == 1 and not (x < xmax and y > 0):
                            continue
                        if d == 2 and not (x < xmax):
                            continue
                        if d == 3 and not (x < xmax and y < ymax):
                            continue
                        if d == 4 and not (y < ymax):
                            continue
                        if d == 5 and not (x > 0 and y < ymax):
                            continue
                        if d == 6 and not (x > 0):
                            continue
                        if d == 7 and not (x > 0 and y > 0):
                            continue
                    offset2 = offset + dir_off[d]
                    if t[offset2] & LISTED:
                        continue
                    if is_edm and edm[offset2] <= 0:
                        continue
                    if t[offset2] & PROCESSED:
                        max_possible = False
                        break
                    x2, y2 = x + _DIR_X[d], y + _DIR_Y[d]
                    v2 = vals[offset2]
                    if v2 > hi:
                        max_possible = False
                        break
                    if v2 >= lo:
                        if v2 > v0:
                            sorting_error = True
                            offset0, v0, x0, y0 = offset2, v2, x2, y2
                            v0_32 = np.float32(v0)
                            hi = float(v0_32 + max_sorting_error)
                            lo = float(v0_32 - tol32)
                        plist.append(offset2)
                        t[offset2] |= LISTED
                        if (x2 == 0 or x2 == xmax or y2 == 0 or y2 == ymax) and (strict or v2 >= v0):
                            is_edge_max = True
                            if exclude_edges_now:
                                max_possible = False
                                break
                        if v2 == v0:
                            t[offset2] |= EQUAL
                            x_equal += x2
                            y_equal += y2
                            n_equal += 1
                list_i += 1
            if sorting_error:
                for o in plist:
                    t[o] = 0
                continue
            reset_mask = (~LISTED) & 0xFF if max_possible else (~(LISTED | EQUAL)) & 0xFF
            x_equal /= n_equal
            y_equal /= n_equal
            min_dist2 = 1e20
            nearest_i = 0
            for li, o in enumerate(plist):
                t[o] &= reset_mask
                t[o] |= PROCESSED
                if max_possible:
                    t[o] |= MAX_AREA
                    if t[o] & EQUAL:
                        x, y = o % W, o // W
                        dist2 = (x_equal - x) * (x_equal - x) + (y_equal - y) * (y_equal - y)
                        if dist2 < min_dist2:
                            min_dist2 = dist2
                            nearest_i = li
            if max_possible:
                t[plist[nearest_i]] |= MAX_POINT
            break
    if n_max == 0:
        types[:] = PROCESSED | MAX_AREA


def _make_8bit(ip: np.ndarray, types: np.ndarray, is_edm: bool, global_min: float, global_max: float, threshold: float | None) -> np.ndarray:
    """``make8bit``: the image scaled to 1…254 (0 below the threshold, 255 inside the MAX_AREA regions)."""
    if is_edm:
        threshold = 0.5
        min_value = 1.0
    else:
        min_value = global_min if threshold is None else threshold
    offset = min_value - (global_max - min_value) * (1.0 / 253 / 2 - 1e-6)
    factor = 253 / (global_max - min_value) if global_max != min_value else np.inf
    if is_edm and factor > 1.0:
        factor = 1.0
    raw = ip.astype(np.float64).ravel()
    v = 1 + np.floor((raw - offset) * factor + 0.5)
    out = np.where(v < 1, 1, np.where(v <= 254, v, 254)).astype(np.uint8)
    out[(types & MAX_AREA) != 0] = 255
    if threshold is not None:
        out[raw < threshold] = 0
    return out


def _cleanup_maxima(pixels: np.ndarray, types: np.ndarray, max_points: np.ndarray, W: int, H: int) -> None:
    """``cleanupMaxima``: flatten the maxima that are not MAX_AREA down to the level of their saddle point."""
    xmax, ymax = W - 1, H - 1
    dir_off = (-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1)
    px = pixels
    t = types
    for i_max in range(len(max_points) - 1, -1, -1):
        offset0 = int(max_points[i_max])
        if t[offset0] & (MAX_AREA | ELIMINATED):
            continue
        level = int(px[offset0])
        lo_level = level + 1
        plist = [offset0]
        t[offset0] |= LISTED
        last_len = 1
        saddle_found = False
        while not saddle_found and lo_level > 0:
            lo_level -= 1
            last_len = len(plist)
            list_i = 0
            while list_i < len(plist):
                offset = plist[list_i]
                x, y = offset % W, offset // W
                is_inner = y != 0 and y != ymax and x != 0 and x != xmax
                for d in range(8):
                    if not is_inner:
                        if d == 0 and not (y > 0):
                            continue
                        if d == 1 and not (x < xmax and y > 0):
                            continue
                        if d == 2 and not (x < xmax):
                            continue
                        if d == 3 and not (x < xmax and y < ymax):
                            continue
                        if d == 4 and not (y < ymax):
                            continue
                        if d == 5 and not (x > 0 and y < ymax):
                            continue
                        if d == 6 and not (x > 0):
                            continue
                        if d == 7 and not (x > 0 and y > 0):
                            continue
                    offset2 = offset + dir_off[d]
                    if t[offset2] & LISTED:
                        continue
                    if (t[offset2] & MAX_AREA) or ((t[offset2] & ELIMINATED) and int(px[offset2]) >= lo_level):
                        saddle_found = True
                        break
                    if int(px[offset2]) >= lo_level and not (t[offset2] & ELIMINATED):
                        plist.append(offset2)
                        t[offset2] |= LISTED
                if saddle_found:
                    break
                list_i += 1
        for o in plist:
            t[o] &= (~LISTED) & 0xFF
        for o in plist[:last_len]:
            px[o] = lo_level
            t[o] |= ELIMINATED


def _make_fate_table() -> np.ndarray:
    table = np.zeros(256, np.int32)
    for item in range(256):
        is_set = [(item & (1 << i)) != 0 for i in range(8)]
        for i in range(8):
            if is_set[(i + 4) % 8]:
                table[item] |= 1 << i
        for i in range(0, 8, 2):
            if is_set[i]:
                is_set[(i + 1) % 8] = True
                is_set[(i + 7) % 8] = True
        transitions = sum(1 for i in range(8) if is_set[i] != is_set[(i + 1) % 8])
        if transitions >= 4:
            table[item] = 0
    return table


_FATE_TABLE = _make_fate_table()


def _process_level(pass_: int, pix: np.ndarray, coords: np.ndarray, W: int, H: int) -> np.ndarray:
    """``processLevel``: which of the level's pixels may be set to 255 in this pass (decided on the state at the
    start of the pass, as ImageJ applies its setPointList afterwards)."""
    x = coords % W
    y = coords // W
    p = pix
    is255 = lambda off: p[off] == 255  # noqa: E731
    xmax, ymax = W - 1, H - 1
    idx = np.zeros(coords.shape, np.int32)
    m = y > 0
    idx[m] |= is255(coords[m] - W).astype(np.int32) * 1
    m = (x < xmax) & (y > 0)
    idx[m] |= is255(coords[m] - W + 1).astype(np.int32) * 2
    m = x < xmax
    idx[m] |= is255(coords[m] + 1).astype(np.int32) * 4
    m = (x < xmax) & (y < ymax)
    idx[m] |= is255(coords[m] + W + 1).astype(np.int32) * 8
    m = y < ymax
    idx[m] |= is255(coords[m] + W).astype(np.int32) * 16
    m = (x > 0) & (y < ymax)
    idx[m] |= is255(coords[m] + W - 1).astype(np.int32) * 32
    m = x > 0
    idx[m] |= is255(coords[m] - 1).astype(np.int32) * 64
    m = (x > 0) & (y > 0)
    idx[m] |= is255(coords[m] - W - 1).astype(np.int32) * 128
    mask = 1 << pass_
    return (_FATE_TABLE[idx] & mask) == mask


def _watershed_segment(pix: np.ndarray, W: int, H: int) -> None:
    """``watershedSegment``: dilate the 255 regions level by level (highest first) through 8 directions, the fate
    table forbidding merges; unreached edge/border pixels of a level join the next lower level's task list."""
    hist = np.bincount(pix, minlength=256)
    coords: dict[int, np.ndarray] = {}
    all_off = np.flatnonzero((pix > 0) & (pix < 255))
    vals = pix[all_off]
    for v in range(1, 255):
        coords[v] = all_off[vals == v]
    highest = max([v for v in range(1, 255) if hist[v] > 0], default=0)
    xmax, ymax = W - 1, H - 1
    for level in range(highest, 0, -1):
        rem = coords.get(level, np.zeros(0, np.int64))
        idle = 0
        while rem.size > 0 and idle < 8:
            d_index = 0
            while True:
                changed = _process_level(_DIRECTION_SEQUENCE[d_index % 8], pix, rem, W, H)
                n = int(changed.sum())
                if n:
                    pix[rem[changed]] = 255
                    rem = rem[~changed]
                    idle = 0
                d_index += 1
                if rem.size > 0:
                    old = idle
                    idle += 1
                    if old < 8:
                        continue
                break
        if rem.size > 0 and level > 1:
            next_level = level
            while True:
                next_level -= 1
                if not (next_level > 1 and len(coords.get(next_level, ())) == 0):
                    break
            if next_level > 0:
                x = rem % W
                y = rem // W
                add = (x == 0) | (y == 0) | (x == xmax) | (y == ymax)
                inner = ~add
                if inner.any():
                    ri = rem[inner]
                    xi, yi = x[inner], y[inner]
                    zero_nb = np.zeros(ri.shape, bool)
                    for d, off in enumerate((-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1)):
                        if d == 0:
                            within = yi > 0
                        elif d == 1:
                            within = (xi < xmax) & (yi > 0)
                        elif d == 2:
                            within = xi < xmax
                        elif d == 3:
                            within = (xi < xmax) & (yi < ymax)
                        elif d == 4:
                            within = yi < ymax
                        elif d == 5:
                            within = (xi > 0) & (yi < ymax)
                        elif d == 6:
                            within = xi > 0
                        else:
                            within = (xi > 0) & (yi > 0)
                        sel = within & ~zero_nb
                        if sel.any():
                            zero_nb[sel] = pix[ri[sel] + off] == 0
                    add[inner] = zero_nb
                coords[next_level] = np.concatenate([coords.get(next_level, np.zeros(0, np.int64)), rem[add]])


def _n_radii(px: np.ndarray, x: int, y: int, W: int, H: int) -> int:
    """``nRadii``: number of lines (non-255 4-connected runs) leaving a pixel."""
    offset = x + y * W
    xmax, ymax = W - 1, H - 1
    count = 0
    prev_set = True
    first_set = True
    is_inner = y != 0 and y != ymax and x != 0 and x != xmax
    dir_off = (-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1)
    for d in range(8):
        pixel_set = prev_set
        if is_inner or _is_within(x, y, d, xmax, ymax):
            is_set = px[offset + dir_off[d]] != 255
            if (d & 1) == 0:
                pixel_set = is_set
            elif not is_set:
                pixel_set = False
        else:
            pixel_set = True
        if pixel_set and not prev_set:
            count += 1
        prev_set = pixel_set
        if d == 0:
            first_set = pixel_set
    if first_set and not prev_set:
        count += 1
    return count


def _is_within(x: int, y: int, d: int, xmax: int, ymax: int) -> bool:
    if d == 0:
        return y > 0
    if d == 1:
        return x < xmax and y > 0
    if d == 2:
        return x < xmax
    if d == 3:
        return x < xmax and y < ymax
    if d == 4:
        return y < ymax
    if d == 5:
        return x > 0 and y < ymax
    if d == 6:
        return x > 0
    return x > 0 and y > 0


def _remove_line_from(px: np.ndarray, x: int, y: int, W: int, H: int) -> None:
    xmax, ymax = W - 1, H - 1
    dir_off = (-W, -W + 1, 1, W + 1, W, W - 1, -1, -W - 1)
    px[x + W * y] = 255
    while True:
        continues = False
        is_inner = y != 0 and y != ymax and x != 0 and x != xmax
        for d in range(0, 8, 2):
            if is_inner or _is_within(x, y, d, xmax, ymax):
                v = px[x + W * y + dir_off[d]]
                if v != 255 and v != 0:
                    nr = _n_radii(px, x + _DIR_X[d], y + _DIR_Y[d], W, H)
                    if nr <= 1:
                        x += _DIR_X[d]
                        y += _DIR_Y[d]
                        px[x + W * y] = 255
                        continues = nr == 1
                        break
        if not continues:
            break


def _cleanup_extra_lines(px: np.ndarray, W: int, H: int) -> None:
    """``cleanupExtraLines``: remove watershed lines that end in a local minimum (non-EDM images)."""
    for p in np.flatnonzero((px != 255) & (px != 0)).tolist():
        v = px[p]
        if v == 255 or v == 0:
            continue
        x, y = p % W, p // W
        nr = _n_radii(px, x, y, W, H)
        if nr == 0:
            px[p] = 255
        elif nr == 1:
            _remove_line_from(px, x, y, W, H)


def find_maxima(ip: np.ndarray, tolerance: float = 0.5, output_type: int = SINGLE_POINTS, exclude_on_edges: bool = False,
                is_edm: bool = False, threshold: float | None = None, strict: bool = False) -> np.ndarray:
    """``MaximumFinder.findMaxima(ip, tolerance, strict, threshold, outputType, excludeOnEdges, isEDM)`` for a float32
    plane; returns the uint8 result image (255 = single maximum points / in-tolerance areas / segmented particles)."""
    ip = np.ascontiguousarray(ip, dtype=np.float32)
    H, W = ip.shape
    global_min = np.float32(ip.min())
    global_max = np.float32(ip.max())
    max_possible = bool(global_max > global_min)
    if strict and float(global_max) - float(global_min) <= tolerance:
        max_possible = False
    if threshold is not None:
        threshold = threshold - (float(global_max) - float(global_min)) * 1e-6
    exclude_edges_now = exclude_on_edges and output_type != SEGMENTED
    vt = true_edm_height(ip) if is_edm else ip
    if max_possible:
        types, max_points = _sorted_max_points(ip, vt, exclude_edges_now, is_edm, global_min, global_max, threshold)
    else:
        types, max_points = np.zeros(H * W, np.uint8), np.zeros(0, np.int64)
    if is_edm:
        max_sorting_error = np.float32(1.1) * _HALF_SQRT2
    else:
        max_sorting_error = np.float32(1.1) * ((global_max - global_min) / np.float32(2e9))
    _analyze_and_mark_maxima(ip, types, max_points, exclude_edges_now, is_edm, tolerance, strict, max_sorting_error, vt)
    if output_type == SEGMENTED:
        pix = _make_8bit(ip, types, is_edm, float(global_min), float(global_max), threshold)
        _cleanup_maxima(pix, types, max_points, W, H)
        _watershed_segment(pix, W, H)
        if not is_edm:
            _cleanup_extra_lines(pix, W, H)
        pix[pix < 255] = 0
        return pix.reshape(H, W)
    flag = MAX_POINT if output_type == SINGLE_POINTS else MAX_AREA
    return np.where((types & flag) != 0, 255, 0).astype(np.uint8).reshape(H, W)


# ====================================================================== EDM plugin commands
def ij_distance_map(b: np.ndarray) -> np.ndarray:
    """Process ▸ Binary ▸ Distance Map with EDM output 'Overwrite': the rounded 8-bit EDM."""
    return edm_to_byte(edm_float(b))


def ij_ultimate_points(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ultimate Points: (8-bit image holding the rounded EDM value at every UEP, the float EDM with the UEP values
    only) – ``EDM.run`` with MaximumFinder(SINGLE_POINTS) and ``resetMasked``."""
    edm = edm_float(b)
    if not b.any():
        return np.zeros(b.shape, np.uint8), edm
    pts = find_maxima(edm, 0.5, SINGLE_POINTS, False, True)
    edm = np.where(pts == 0, np.float32(0), edm).astype(np.float32)
    return edm_to_byte(edm), edm


def ij_watershed(b: np.ndarray) -> np.ndarray:
    """Watershed: ``EDM.toWatershed`` – MaximumFinder(SEGMENTED) on the float EDM (tolerance 0.5)."""
    if not b.any():
        return b.astype(bool)
    seg = find_maxima(edm_float(b), 0.5, SEGMENTED, False, True)
    return seg == 255


def ij_voronoi(b: np.ndarray) -> np.ndarray:
    """Voronoi with EDM output 'Overwrite': the rounded EDM of the background along the segmentation lines of the
    (negated) background EDM, 0 elsewhere."""
    bg = ~b.astype(bool)
    edm = edm_float(bg)
    if not b.any() or not bg.any():
        return np.zeros(b.shape, np.uint8)
    seg = find_maxima(-edm, 0.5, SEGMENTED, False, False)
    edm = np.where(seg == 255, np.float32(0), edm).astype(np.float32)
    return edm_to_byte(edm)
