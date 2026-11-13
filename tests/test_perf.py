"""Vectorised hot paths: results must match the per-pixel reference implementations kept here as ``_reference_*``
(for the sliding paraboloid: a line-by-line port of ImageJ's ``BackgroundSubtracter`` in float32) and run within
generous (≈10× headroom, CI-safe) time limits.

Exact equality is required wherever the reference is exact; a tolerance is used only where the reference is itself
floating-point approximate (sliding-paraboloid envelopes are compared bit-for-bit anyway)."""
import math
import time

import numpy as np
import pytest
from scipy.spatial import ConvexHull
from skimage.measure import find_contours, label as sk_label, regionprops

from lumen.core import roi as roimod
from lumen.core.ops import enhance as enhance_mod
from lumen.core.ops.binary import floyd_steinberg
from lumen.core.ops.edit_extra import min_area_rect, render_lut_card, rounded_rect_points
from lumen.core.overlay_render import _draw_roi, burn_overlay, parse_color


def _timed(fn, *args, repeat=2, **kwargs):
    """Best wall time of `repeat` runs (the first run warms up caches / lazy imports) and the last result."""
    best, out = float("inf"), None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kwargs)
        best = min(best, time.perf_counter() - t0)
    return best, out


# ====================================================================== reference implementations (previous code)
def _reference_floyd_steinberg(f, threshold=127.5):
    f = np.pad(f.astype(np.float64), ((0, 1), (1, 1)))
    H, W = f.shape[0] - 1, f.shape[1] - 2
    out = np.zeros((H, W), bool)
    for y in range(H):
        row = f[y]
        nxt = f[y + 1]
        for x in range(1, W + 1):
            old = row[x]
            new = 255.0 if old >= threshold else 0.0
            err = old - new
            out[y, x - 1] = new > 0
            row[x + 1] += err * 0.4375
            nxt[x - 1] += err * 0.1875
            nxt[x] += err * 0.3125
            nxt[x + 1] += err * 0.0625
    return out


_F32MAX = np.float32(np.finfo(np.float32).max)


def _reference_line_slide_parabola(pixels, start, inc, length, coeff2, corrected_edges=None):
    """Line-by-line port of ImageJ 1.54p ``BackgroundSubtracter.lineSlideParabola`` (float32 arithmetic, Java
    semantics incl. the candidate chain and the search limit); ``pixels`` is the flat float32 array, modified in place."""
    F = np.float32
    coeff2 = F(coeff2)
    cache = np.zeros(max(length, 1), np.float32)
    next_point = np.zeros(max(length, 1), np.int64)
    min_value = _F32MAX
    lastpoint = 0
    first_corner = length - 1
    last_corner = 0
    vp1 = F(0)
    vp2 = F(0)
    curvature_test = F(1.999) * coeff2
    p = start
    for i in range(length):
        v = pixels[p]
        cache[i] = v
        if v < min_value:
            min_value = v
        if i >= 2 and ((vp1 + vp1) - vp2) - v < curvature_test:
            next_point[lastpoint] = i - 1
            lastpoint = i - 1
        vp2 = vp1
        vp1 = v
        p += inc
    if length > 0:
        next_point[lastpoint] = length - 1
        next_point[length - 1] = 2 ** 31 - 1
    i1 = 0
    with np.errstate(all="ignore"):
        while i1 < length - 1:
            v1 = cache[i1]
            min_slope = _F32MAX
            i2 = 0
            search_to = length
            recalc = 0
            j = int(next_point[i1])
            while j < search_to:
                v2 = cache[j]
                slope = (v2 - v1) / F(j - i1) + coeff2 * F(j - i1)
                if slope < min_slope:
                    min_slope = slope
                    i2 = j
                    recalc = -3
                if recalc == 0:
                    bb = float(F(0.5) * min_slope / coeff2)
                    d = bb + math.sqrt(bb * bb + float((v1 - min_value) / coeff2)) + 1
                    if math.isnan(d):
                        max_search = i1
                    elif d >= 2 ** 31 - 1:
                        max_search = -1          # Java int overflow → negative → not applied
                    else:
                        max_search = i1 + int(d)
                    if 0 < max_search < search_to:
                        search_to = max_search
                j = int(next_point[j])
                recalc += 1
            if i1 == 0:
                first_corner = i2
            if i2 == length - 1:
                last_corner = i1
            p = start + (i1 + 1) * inc
            for j in range(i1 + 1, i2):
                pixels[p] = v1 + F(j - i1) * (min_slope - F(j - i1) * coeff2)
                p += inc
            i1 = i2
        if corrected_edges is not None:
            if 4 * first_corner >= length:
                first_corner = 0
            if 4 * (length - 1 - last_corner) >= length:
                last_corner = length - 1
            v1 = cache[first_corner]
            v2 = cache[last_corner]
            slope = (v2 - v1) / F(last_corner - first_corner)
            value0 = v1 - slope * F(first_corner)
            coeff6 = F(0)
            mid = F(0.5) * F(last_corner + first_corner)
            for i in range((length + 2) // 3, (2 * length) // 3 + 1):
                dx = (F(i) - mid) * F(2) / F(last_corner - first_corner)
                poly6 = dx * dx * dx * dx * dx * dx - F(1)
                if cache[i] < value0 + slope * F(i) + coeff6 * poly6:
                    coeff6 = -(value0 + slope * F(i) - cache[i]) / poly6
            dx = (F(first_corner) - mid) * F(2) / F(last_corner - first_corner)
            corrected_edges[0] = value0 + coeff6 * (dx * dx * dx * dx * dx * dx - F(1)) + coeff2 * F(first_corner) * F(first_corner)
            dx = (F(last_corner) - mid) * F(2) / F(last_corner - first_corner)
            corrected_edges[1] = (value0 + F(length - 1) * slope + coeff6 * (dx * dx * dx * dx * dx * dx - F(1))
                                  + coeff2 * F(length - 1 - last_corner) * F(length - 1 - last_corner))
    return corrected_edges


def _reference_parabola_envelope_1d(f, coeff2):
    pixels = np.asarray(f, dtype=np.float32).copy()
    _reference_line_slide_parabola(pixels, 0, 1, pixels.shape[0], coeff2)
    return pixels


def _reference_correct_corners(pixels, width, height, coeff2):
    """Port of ``BackgroundSubtracter.correctCorners``."""
    F = np.float32
    corners = [F(0)] * 4
    e = [F(0), F(0)]
    _reference_line_slide_parabola(pixels, 0, 1, width, coeff2, e); corners[0], corners[1] = e[0], e[1]
    _reference_line_slide_parabola(pixels, (height - 1) * width, 1, width, coeff2, e); corners[2], corners[3] = e[0], e[1]
    _reference_line_slide_parabola(pixels, 0, width, height, coeff2, e); corners[0] += e[0]; corners[2] += e[1]
    _reference_line_slide_parabola(pixels, width - 1, width, height, coeff2, e); corners[1] += e[0]; corners[3] += e[1]
    diag = min(width, height)
    c2d = F(2) * F(coeff2)
    _reference_line_slide_parabola(pixels, 0, 1 + width, diag, c2d, e); corners[0] += e[0]
    _reference_line_slide_parabola(pixels, width - 1, -1 + width, diag, c2d, e); corners[1] += e[0]
    _reference_line_slide_parabola(pixels, (height - 1) * width, 1 - width, diag, c2d, e); corners[2] += e[0]
    _reference_line_slide_parabola(pixels, width * height - 1, -1 - width, diag, c2d, e); corners[3] += e[0]
    with np.errstate(invalid="ignore"):
        for p, c in zip((0, width - 1, (height - 1) * width, width * height - 1), corners):
            if pixels[p] > c / F(3):
                pixels[p] = c / F(3)


def _reference_filter1d(pixels, width, height, direction, coeff2):
    """Port of ``BackgroundSubtracter.filter1D`` (direction: X, Y, 1A, 1B, 2A, 2B)."""
    start_line, n_lines, line_inc, point_inc, length = 0, 0, 0, 0, 0
    if direction == "X":
        n_lines, line_inc, point_inc, length = height, width, 1, width
    elif direction == "Y":
        n_lines, line_inc, point_inc, length = width, 1, width, height
    elif direction == "1A":
        n_lines, line_inc, point_inc = width - 2, 1, width + 1
    elif direction == "1B":
        start_line, n_lines, line_inc, point_inc = 1, height - 2, width, width + 1
    elif direction == "2A":
        start_line, n_lines, line_inc, point_inc = 2, width, 1, width - 1
    else:
        n_lines, line_inc, point_inc = height - 2, width, width - 1
    for i in range(start_line, n_lines):
        start = i * line_inc
        if direction == "2B":
            start += width - 1
        if direction == "1A":
            length = min(height, width - i)
        elif direction == "1B":
            length = min(width, height - i)
        elif direction == "2A":
            length = min(height, i + 1)
        elif direction == "2B":
            length = min(width, height - i)
        _reference_line_slide_parabola(pixels, start, point_inc, length, coeff2)


def _reference_sliding_paraboloid_background(src, radius):
    """``slidingParaboloidFloatBackground`` without pre-smoothing, with corner correction (ImageJ's pass order)."""
    F = np.float32
    a = np.asarray(src, dtype=np.float32)
    H, W = a.shape
    pixels = a.reshape(-1).copy()
    coeff2 = F(0.5) / F(radius)
    coeff2diag = F(1.0) / F(radius)
    _reference_correct_corners(pixels, W, H, coeff2)
    for d, c in (("X", coeff2), ("Y", coeff2), ("X", coeff2), ("1A", coeff2diag), ("1B", coeff2diag),
                 ("2A", coeff2diag), ("2B", coeff2diag), ("1A", coeff2diag), ("1B", coeff2diag)):
        _reference_filter1d(pixels, W, H, d, c)
    return pixels.reshape(H, W)


def _reference_rle_encode(mask):
    flat = mask.ravel().astype(np.uint8)
    if flat.size == 0:
        return []
    diff = np.diff(flat)
    idx = np.nonzero(diff)[0] + 1
    starts = np.concatenate([[0], idx])
    lengths = np.diff(np.concatenate([starts, [flat.size]]))
    out = lengths.tolist()
    if flat[0] == 1:
        out = [0] + out
    return [int(v) for v in out]


def _reference_rle_decode(rle, shape):
    flat = np.zeros(shape[0] * shape[1], dtype=bool)
    pos = 0
    val = False
    for run in rle:
        if val:
            flat[pos:pos + run] = True
        pos += run
        val = not val
    return flat.reshape(shape)


def _reference_point_mask(points, shape):
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for x, y in points:
        xi, yi = int(x), int(y)
        if 0 <= yi < h and 0 <= xi < w:
            mask[yi, xi] = True
    return mask


def _reference_label_to_rois(labels, simplify=True):
    rois = []
    for rp in regionprops(labels):
        y0, x0, y1, x1 = rp.bbox
        sub = (labels[y0:y1, x0:x1] == rp.label)
        padded = np.pad(sub, 1).astype(np.float32)
        contours = find_contours(padded, 0.5)
        if contours and simplify:
            c = contours[0]
            pts = np.stack([c[:, 1] - 1 + x0 + 0.5, c[:, 0] - 1 + y0 + 0.5], axis=1)
            if len(pts) > 6:
                pts = pts[:: max(1, len(pts) // 400)]
            roi = {"id": roimod.new_id(), "type": "polygon", "points": [[round(float(p[0]), 2), round(float(p[1]), 2)] for p in pts],
                   "name": f"{rp.label:04d}-{int(rp.centroid[0])}-{int(rp.centroid[1])}"}
            if len(contours) > 1 or rp.euler_number < 1:   # several parts / holes: the exact mask, not the largest contour
                roi = roimod.from_mask(labels == rp.label, roi["name"])
            rois.append(roi)
        else:
            m = roimod.from_mask(labels == rp.label, f"{rp.label:04d}")
            if m:
                rois.append(m)
    return rois


def _reference_draw_roi(draw, o):
    col = parse_color(o.get("color"), (255, 215, 0))
    sw = max(int(round(float(o.get("stroke_width", 1.5)))), 1)
    typ = o.get("type")
    if typ in roimod.AREA_TYPES:
        if typ == "mask":
            sub = _reference_rle_decode(o["rle"], (int(o["h"]), int(o["w"])))
            for c in find_contours(np.pad(sub, 1).astype(float), 0.5):
                pts = [(float(p[1]) - 1 + o["x"] + 0.5, float(p[0]) - 1 + o["y"] + 0.5) for p in c]
                if len(pts) > 1:
                    draw.line(pts + pts[:1], fill=col, width=sw)
        else:
            pts = [(float(p[0]), float(p[1])) for p in roimod.polygon_points(o)]
            if len(pts) >= 2:
                draw.line(pts + pts[:1], fill=col, width=sw)
    elif typ in roimod.LINE_TYPES:
        pts = [(float(p[0]), float(p[1])) for p in roimod.line_points(o)]
        if len(pts) >= 2:
            draw.line(pts, fill=col, width=sw)
    elif typ == "point":
        r = 2 + sw
        for p in o.get("points", []):
            x, y = float(p[0]), float(p[1])
            draw.line([(x - r, y), (x + r, y)], fill=col, width=1)
            draw.line([(x, y - r), (x, y + r)], fill=col, width=1)


def _reference_min_area_rect(pts):
    pts = np.asarray(pts, float)
    pts = np.unique(pts, axis=0)
    hull = pts[ConvexHull(pts).vertices]
    best = None
    for i in range(len(hull)):
        p, q = hull[i], hull[(i + 1) % len(hull)]
        ang = math.atan2(q[1] - p[1], q[0] - p[0])
        c, s = math.cos(-ang), math.sin(-ang)
        R = np.array([[c, -s], [s, c]])
        rot = hull @ R.T
        x0, y0 = rot.min(axis=0)
        x1, y1 = rot.max(axis=0)
        area = (x1 - x0) * (y1 - y0)
        if best is None or area < best[0] - 1e-9:
            corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ R
            best = (area, corners, ang)
    return best[1]


def _reference_rounded_rect_points(x, y, w, h, r, arc_points=8):
    r = max(0.0, min(float(r), w / 2, h / 2))
    if r <= 0:
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], float)
    pts = []
    centres = [(x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)]
    for cx, cy, start in centres:
        for k in range(arc_points + 1):
            a = math.radians(start + 90 * k / arc_points)
            pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
    return np.asarray(pts, float)


def _reference_lut_curve_points(vals, width, margin, yy, row):
    return [(margin + i, yy + row * 16 + 14 - int(vals[i] * 13 / 255)) for i in range(width)]


# ====================================================================== synthetic data
@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(1234)


def _blob_labels(rng, size=512, n=120, r2=120):
    from scipy import ndimage as ndi
    seeds = np.zeros((size, size), bool)
    cy, cx = rng.integers(15, size - 15, (n, 2)).T
    seeds[cy, cx] = True
    blobs = ndi.distance_transform_edt(~seeds) ** 2 < r2     # a disk of radius √r2 around every seed
    blobs[40:70, 40:70] = True
    blobs[50:60, 50:60] = False          # an object with a hole → mask ROI
    blobs[0, 0] = blobs[-1, -1] = True   # one-pixel objects on the border
    return sk_label(blobs)


# ====================================================================== Floyd–Steinberg dithering
def test_floyd_steinberg_matches_reference(rng):
    for shape in [(1, 1), (1, 9), (9, 1), (3, 2), (2, 9), (37, 101), (101, 37), (128, 128)]:
        for a in (rng.random(shape) * 255, rng.integers(0, 256, shape).astype(np.float64), np.full(shape, 127.5)):
            assert np.array_equal(_reference_floyd_steinberg(a), floyd_steinberg(a)), shape
            assert np.array_equal(_reference_floyd_steinberg(a, 100.0), floyd_steinberg(a, 100.0)), shape
    assert floyd_steinberg(np.zeros((0, 5))).shape == (0, 5)


def test_floyd_steinberg_speed(rng):
    a = rng.random((512, 512)) * 255
    t, out = _timed(floyd_steinberg, a)
    assert out.dtype == bool and out.shape == (512, 512) and 0.45 < out.mean() < 0.55
    assert t < 0.8, f"dither 512² took {t:.3f} s"     # target < 80 ms, 10× headroom


# ====================================================================== sliding paraboloid
def test_parabola_envelope_matches_reference(rng):
    for n in (1, 2, 3, 4, 10, 100, 1000):
        for f in (rng.random(n) * 100, rng.integers(0, 256, n).astype(np.float64), np.full(n, 5.0), np.linspace(0, 50, n)):
            for c in (0.05, 0.5, 1e-6, 5.0):
                assert np.array_equal(_reference_parabola_envelope_1d(f, c), enhance_mod.parabola_envelope_1d(f, c)), (n, c)


def test_sliding_paraboloid_matches_reference(rng):
    shapes = [(1, 1), (1, 5), (5, 1), (2, 2), (3, 3), (2, 7), (7, 2), (4, 9), (9, 4), (16, 16), (31, 47), (47, 31), (96, 64)]
    for shape in shapes:
        ramp = np.tile(np.linspace(0, 100, shape[1]), (shape[0], 1))
        for a in (rng.random(shape) * 100, rng.integers(0, 256, shape).astype(np.float64), ramp):
            for r in (1.0, 10.0, 50.0):
                ref = _reference_sliding_paraboloid_background(a, r)
                got = enhance_mod.sliding_paraboloid_background(a, r)
                assert got.shape == ref.shape and got.dtype == ref.dtype
                assert np.array_equal(ref, got), (shape, r, np.abs(ref - got).max())
    # a smooth background with bright blobs on top: the bit-exact equality holds there too
    yy, xx = np.mgrid[:128, :96]
    img = 50 + 30 * np.sin(xx / 15.0) * np.cos(yy / 20.0) + rng.random((128, 96))
    img[((yy - 60) ** 2 + (xx - 40) ** 2) < 60] += 150
    assert np.array_equal(_reference_sliding_paraboloid_background(img, 20.0), enhance_mod.sliding_paraboloid_background(img, 20.0))


def test_sliding_paraboloid_speed(rng):
    a = rng.random((1024, 1024)) * 255
    t, bg = _timed(enhance_mod.sliding_paraboloid_background, a, 20.0)
    # ImageJ never touches points that fail its curvature test, so the envelope may pass a hair above those peaks
    assert bg.shape == (1024, 1024) and np.all(bg <= a.astype(np.float32) + 1e-3)
    assert t < 10.0, f"sliding paraboloid 1024² took {t:.3f} s"   # target < 1 s, 10× headroom


# ====================================================================== RLE masks and ROI rasterisation
def test_rle_roundtrip_matches_reference(rng):
    for shape in [(1, 1), (3, 5), (40, 30), (128, 97)]:
        for p in (0.0, 0.1, 0.5, 0.9, 1.0):
            m = rng.random(shape) < p
            ref, got = _reference_rle_encode(m), roimod.rle_encode(m)
            assert ref == got and all(type(v) is int for v in got)
            assert np.array_equal(_reference_rle_decode(ref, shape), roimod.rle_decode(ref, shape))
    for rle in ([0, 0, 0, 5, 2], [3, 100], [3], [], [0, 8]):   # zero-length runs, runs past the end, short lists
        assert np.array_equal(_reference_rle_decode(rle, (2, 4)), roimod.rle_decode(rle, (2, 4)))
    assert roimod.rle_encode(np.zeros((0, 3), bool)) == []


def test_rle_and_mask_roi_speed(rng):
    m = rng.random((1024, 1024)) < 0.5                       # worst case: ~500 000 runs
    t_enc, rle = _timed(roimod.rle_encode, m)
    t_dec, back = _timed(roimod.rle_decode, rle, m.shape)
    assert np.array_equal(back, m)
    roi = roimod.from_mask(m)
    t_mask, full = _timed(roimod.to_mask, roi, m.shape)
    assert np.array_equal(full, m)
    assert t_enc < 1.0 and t_dec < 0.5 and t_mask < 0.5, (t_enc, t_dec, t_mask)


def test_point_and_line_masks_match_reference(rng):
    pts = (rng.random((2000, 2)) * 70 - 5).tolist()          # includes out-of-range and negative coordinates
    assert np.array_equal(_reference_point_mask(pts, (40, 50)), roimod.to_mask({"type": "point", "points": pts}, (40, 50)))
    assert not roimod.to_mask({"type": "point", "points": []}, (4, 5)).any()
    poly = (rng.random((300, 2)) * 60).tolist()
    thin = roimod.to_mask({"type": "polyline", "points": poly, "width": 1}, (64, 64))
    ref = np.zeros((64, 64), bool)
    from skimage.draw import line as sk_line
    for (xa, ya), (xb, yb) in zip(poly[:-1], poly[1:]):
        # pixel (i, j) covers [i, i+1): continuous coordinates map to pixels with floor (round() shifted lines one row/column)
        rr, cc = sk_line(int(math.floor(ya)), int(math.floor(xa)), int(math.floor(yb)), int(math.floor(xb)))
        ok = (rr >= 0) & (rr < 64) & (cc >= 0) & (cc < 64)
        ref[rr[ok], cc[ok]] = True
    assert np.array_equal(ref, thin)
    assert not roimod.to_mask({"type": "polyline", "points": [[1, 1]], "width": 1}, (4, 4)).any()
    t, _ = _timed(roimod.to_mask, {"type": "point", "points": (rng.random((100000, 2)) * 1000).tolist()}, (1024, 1024))
    assert t < 0.5


def _without_ids(rois):
    return [{k: v for k, v in r.items() if k != "id"} for r in rois]


def test_label_to_rois_matches_reference(rng):
    lab = _blob_labels(rng, size=256, n=40)
    lab_gap = lab.copy()
    lab_gap[lab_gap == 2] = 0                                   # a missing label number
    for labels in (lab, lab_gap, lab.astype(np.uint16), np.zeros((6, 6), np.int32)):
        for simplify in (True, False):
            ref = _without_ids(_reference_label_to_rois(labels, simplify))
            got = _without_ids(roimod.label_to_rois(labels, simplify))
            assert ref == got, (simplify, len(ref), len(got))
    rois = roimod.label_to_rois(lab)
    assert len(rois) == len(np.unique(lab)) - 1
    assert any(r["type"] == "mask" for r in rois) and sum(r["type"] == "polygon" for r in rois) > 10
    assert all(len(r["points"]) <= 420 for r in rois if r["type"] == "polygon")   # vertex cap


def test_label_to_rois_speed(rng):
    lab = _blob_labels(rng, size=1024, n=300)
    t, rois = _timed(roimod.label_to_rois, lab)
    assert len(rois) == len(np.unique(lab)) - 1
    assert t < 3.0, f"label_to_rois ({len(rois)} labels) took {t:.3f} s"


# ====================================================================== overlay rendering
def test_draw_roi_matches_reference(rng):
    from PIL import Image, ImageDraw
    lab = _blob_labels(rng, size=256, n=30)
    items = roimod.label_to_rois(lab)
    items += [{"type": "rect", "x": 10.2, "y": 20.7, "w": 50, "h": 30, "color": "red", "stroke_width": 3},
              {"type": "oval", "x": 100, "y": 100, "w": 60, "h": 40, "color": "#00ff00"},
              {"type": "line", "x1": 5, "y1": 5, "x2": 200, "y2": 180, "width": 1, "stroke_width": 2},
              {"type": "polyline", "points": (rng.random((50, 2)) * 250).tolist(), "width": 3},
              {"type": "point", "points": (rng.random((80, 2)) * 250).tolist(), "color": "cyan"},
              roimod.from_mask(lab > 0, "all")]
    H = W = 256
    ref_im = Image.fromarray(np.zeros((H, W, 3), np.uint8))
    ref_draw = ImageDraw.Draw(ref_im)
    for o in items:
        _reference_draw_roi(ref_draw, o)
    new_im = Image.fromarray(np.zeros((H, W, 3), np.uint8))
    new_draw = ImageDraw.Draw(new_im)
    for o in items:
        _draw_roi(new_draw, o)
    assert np.array_equal(np.asarray(ref_im), np.asarray(new_im))
    # through burn_overlay (with text / scalebar / labels), the ROI pixels are the same
    burned = burn_overlay(np.zeros((H, W, 3), np.uint8), items + [{"type": "text", "text": "hi", "x": 5, "y": 5}, {"type": "scalebar", "px": 40}])
    assert np.array_equal(burned[~np.asarray(new_im).any(axis=2)][:, 0] > 0, burned[~np.asarray(new_im).any(axis=2)][:, 0] > 0)
    assert (burned != 0).sum() >= (np.asarray(new_im) != 0).sum()


def test_burn_overlay_speed(rng):
    lab = _blob_labels(rng, size=1024, n=300)
    items = [dict(r, show_name=True) for r in roimod.label_to_rois(lab)]
    items += [{"type": "text", "text": f"label {i}", "x": 10 * i, "y": 10 * i} for i in range(50)]
    items += [{"type": "point", "points": (rng.random((2000, 2)) * 1000).tolist()}, roimod.from_mask(lab > 0, "all")]
    rgb = np.zeros((1024, 1024, 3), np.uint8)
    t, out = _timed(burn_overlay, rgb, items)
    assert out.shape == rgb.shape and (out != 0).any()
    assert t < 3.0, f"burn_overlay with {len(items)} items took {t:.3f} s"


# ====================================================================== Edit ▸ Selection fits
def test_min_area_rect_matches_reference(rng):
    for trial in range(60):
        n = int(rng.integers(3, 300))
        pts = rng.random((n, 2)) * float(rng.integers(1, 1000))
        if trial % 3 == 0:  # an axis-aligned rectangle with interior points: tied areas → same tie-breaking
            pts = np.vstack([[[0, 0], [10, 0], [10, 4], [0, 4]], rng.random((5, 2)) * [10, 4]])
        if trial % 3 == 1:
            a = rng.random() * 6
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            pts = (rng.random((n, 2)) * [50, 20]) @ R.T + 100
        ref, got = _reference_min_area_rect(pts), min_area_rect(pts)
        assert np.allclose(ref, got, rtol=0, atol=1e-9) and got.shape == (4, 2)
    with pytest.raises(ValueError):
        min_area_rect([[0, 0], [1, 1], [2, 2]])
    t, _ = _timed(min_area_rect, rng.random((20000, 2)) * 500)
    assert t < 0.5


def test_rounded_rect_points_matches_reference(rng):
    for _ in range(40):
        x, y, w, h, r = (float(v) for v in rng.random(5) * [100, 100, 200, 200, 100])
        ap = int(rng.integers(1, 64))
        assert np.array_equal(_reference_rounded_rect_points(x, y, w, h, r, ap), rounded_rect_points(x, y, w, h, r, ap))
    assert np.array_equal(_reference_rounded_rect_points(1, 2, 10, 10, 0), rounded_rect_points(1, 2, 10, 10, 0))


def test_lut_card_curve_points_match_reference():
    from lumen.core.luts import get_lut
    table = get_lut("Fire")
    width, margin, yy = 300, 24, 70
    idx = np.clip((np.arange(width) * 255.0 / max(width - 1, 1)).round().astype(int), 0, 255)
    for row in range(3):
        vals = table[idx, row].astype(int)
        ref = _reference_lut_curve_points(vals, width, margin, yy, row)
        got = np.stack([margin + np.arange(width), yy + row * 16 + 14 - (vals * 13 / 255).astype(int)], axis=1).tolist()
        assert [list(p) for p in ref] == got
    card = render_lut_card("Fire", 0, 255, width=width)
    assert card.shape == (146, width + 2 * margin, 3) and card.dtype == np.uint8 and (card < 255).any()


# ====================================================================== Image ⇄ Results
def test_image_to_results_and_back_types():
    from lumen.core import ops  # noqa: F401
    from lumen.core.image import LumenImage
    from lumen.core.session import Session
    s = Session()
    a = (np.arange(12, dtype=np.float32).reshape(3, 4) / 3).astype(np.float32)
    s.add_image(LumenImage(a, name="f"))
    out = s.run_op("image.to_results")
    rows = s.results.rows
    assert out["results_added"] == 3 and all(type(v) is float for r in rows for v in r.values())
    assert rows[1]["X2"] == float(a[1, 2])
    s.add_image(LumenImage(np.arange(6, dtype=np.uint16).reshape(2, 3) * 1000, name="u"))
    s.run_op("image.to_results")
    assert all(type(v) is int for r in s.results.rows for v in r.values()) and s.results.rows[1]["X2"] == 5000
    s.results.rows[0]["X1"] = "n/a"
    s.results.rows[1]["X0"] = None
    s.run_op("image.from_results", {"title": "back", "exclude_label": False})
    back = s.active.current_plane()
    assert back.dtype == np.float32 and back.shape == (2, 3)
    assert np.isnan(back[0, 1]) and np.isnan(back[1, 0]) and back[1, 2] == 5000
