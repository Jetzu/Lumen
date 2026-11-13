"""Exact ports of ImageJ 1.54p's ``ij.process.AutoThresholder`` (the 16 methods of Image ▸ Adjust ▸ Threshold).

Every function takes a 256-bin integer histogram and returns the bin index ImageJ returns (``-1`` where ImageJ
returns -1; ``get_threshold`` maps that to 0 like ``AutoThresholder.getThreshold``).  The code follows the Java
line by line – integer arithmetic where Java uses int/long (IsoData, Mean, MinError's A/B/C sums), sequential
double accumulation, the same tie rules (strict vs. non-strict comparisons), the same initial values
(``Double.MIN_VALUE`` for MaxEntropy / Yen, 0 for RenyiEntropy, ``Integer.MIN_VALUE`` for Otsu) and the same
truncation / rounding (``Math.round`` = floor(x + 0.5), ``(int)`` = truncation).  Bit depth handling, the
``addOne`` rule and the scaling to 16/32-bit values live in :mod:`lumen.core.threshold`.
"""
from __future__ import annotations

import math

import numpy as np

EPS = 2.220446049250313e-16  # DBL_EPSILON as written in the Java source
DOUBLE_MIN_VALUE = 4.9e-324   # Double.MIN_VALUE (smallest positive double)


def _hist(h) -> list[int]:
    return [int(v) for v in np.asarray(h).ravel().tolist()]


def java_round(x: float) -> int:
    """``(int) Math.round(double)``: floor(x + 0.5)."""
    return int(math.floor(x + 0.5))


def _int32(x: int) -> int:
    """Wrap to Java int (the A/B/C helper sums of MinError use int products)."""
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


# ------------------------------------------------------------------ bilevel short-cut (AutoThresholder.bilevel)
def bilevel(h, subtract_one: bool) -> int:
    """Histograms with one or two non-empty bins: ImageJ answers without running the method (-1 otherwise)."""
    data = _hist(h)
    first = second = -1
    count = 0
    for i, v in enumerate(data):
        if v > 0:
            count += 1
            if count > 2:
                return -1
            if first == -1:
                first = i
            else:
                second = i
    sub = 1 if subtract_one else 0
    if count == 1:
        return first - sub
    if count == 2:
        return second - sub
    return -1


# ------------------------------------------------------------------ Default / IJ_IsoData
def ij_isodata(h) -> int:
    """``AutoThresholder.IJIsoData`` == ``ImageProcessor.getAutoThreshold(int[])`` (the original ImageJ iterative
    intermeans; bins 0 and 255 are ignored)."""
    data = _hist(h)
    max_value = len(data) - 1
    data[0] = 0
    data[max_value] = 0
    lo = 0
    while data[lo] == 0 and lo < max_value:
        lo += 1
    hi = max_value
    while data[hi] == 0 and hi > 0:
        hi -= 1
    if lo >= hi:
        return len(data) // 2
    moving = lo
    while True:
        s1 = s2 = s3 = s4 = 0.0
        for i in range(lo, moving + 1):
            s1 += float(i) * data[i]
            s2 += data[i]
        for i in range(moving + 1, hi + 1):
            s3 += float(i) * data[i]
            s4 += data[i]
        result = (s1 / s2 + s3 / s4) / 2.0
        moving += 1
        if not ((moving + 1) <= result and moving < hi - 1):
            break
    return java_round(result)


def default_isodata(h) -> int:
    """``AutoThresholder.defaultIsoData``: the 'Default' method – the mode bin is cut down to 1.5× the second
    highest bin when it is more than twice as high, then IJIsoData."""
    data = _hist(h)
    max_count, mode = 0, 0
    for i, v in enumerate(data):
        if v > max_count:
            max_count, mode = v, i
    max_count2 = 0
    for i, v in enumerate(data):
        if v > max_count2 and i != mode:
            max_count2 = v
    if max_count > max_count2 * 2 and max_count2 != 0:
        data[mode] = int(max_count2 * 1.5)
    return ij_isodata(data)


# ------------------------------------------------------------------ Huang
def huang(h) -> int:
    data = _hist(h)
    n = len(data)
    first_bin = 0
    for ih in range(n):
        if data[ih] != 0:
            first_bin = ih
            break
    last_bin = n - 1
    for ih in range(n - 1, first_bin - 1, -1):
        if data[ih] != 0:
            last_bin = ih
            break
    term = 1.0 / float(last_bin - first_bin) if last_bin != first_bin else math.inf
    mu_0 = [0.0] * n
    sum_pix = num_pix = 0.0
    for ih in range(first_bin, n):
        sum_pix += float(ih) * data[ih]
        num_pix += data[ih]
        mu_0[ih] = sum_pix / num_pix
    mu_1 = [0.0] * n
    sum_pix = num_pix = 0.0
    for ih in range(last_bin, 0, -1):
        sum_pix += float(ih) * data[ih]
        num_pix += data[ih]
        mu_1[ih - 1] = sum_pix / num_pix
    threshold = -1
    min_ent = 1.7976931348623157e308
    for it in range(n):
        ent = 0.0
        for ih in range(it + 1):
            mu_x = 1.0 / (1.0 + term * abs(ih - mu_0[it]))
            if not (mu_x < 1e-06 or mu_x > 0.999999):
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        for ih in range(it + 1, n):
            mu_x = 1.0 / (1.0 + term * abs(ih - mu_1[it]))
            if not (mu_x < 1e-06 or mu_x > 0.999999):
                ent += data[ih] * (-mu_x * math.log(mu_x) - (1.0 - mu_x) * math.log(1.0 - mu_x))
        if ent < min_ent:
            min_ent = ent
            threshold = it
    return threshold


# ------------------------------------------------------------------ Intermodes / Minimum
def bimodal_test(y: list[float]) -> bool:
    modes = 0
    for k in range(1, len(y) - 1):
        if y[k - 1] < y[k] and y[k + 1] < y[k]:
            modes += 1
            if modes > 2:
                return False
    return modes == 2


def _smooth3(i_histo: list[float]) -> list[float]:
    n = len(i_histo)
    t = [0.0] * n
    for i in range(1, n - 1):
        t[i] = (i_histo[i - 1] + i_histo[i] + i_histo[i + 1]) / 3
    t[0] = (i_histo[0] + i_histo[1]) / 3
    t[n - 1] = (i_histo[n - 2] + i_histo[n - 1]) / 3
    return t


def intermodes(h) -> int:
    data = _hist(h)
    i_histo = [float(v) for v in data]
    it = 0
    while not bimodal_test(i_histo):
        i_histo = _smooth3(i_histo)
        it += 1
        if it > 10000:
            return -1
    tt = 0
    for i in range(1, len(data) - 1):
        if i_histo[i - 1] < i_histo[i] and i_histo[i + 1] < i_histo[i]:
            tt += i
    return int(math.floor(tt / 2.0))


def minimum(h) -> int:
    data = _hist(h)
    if len(data) < 2:
        return 0
    mx = -1
    i_histo = []
    for i, v in enumerate(data):
        i_histo.append(float(v))
        if v > 0:
            mx = i
    it = 0
    while not bimodal_test(i_histo):
        i_histo = _smooth3(i_histo)
        it += 1
        if it > 10000:
            return -1
    for i in range(1, mx):
        if i_histo[i - 1] > i_histo[i] and i_histo[i + 1] >= i_histo[i]:
            return i
    return -1


# ------------------------------------------------------------------ IsoData (Ridler-Calvard, integer arithmetic)
def isodata(h) -> int:
    data = _hist(h)
    n = len(data)
    g = 0
    for i in range(1, n):
        if data[i] > 0:
            g = i + 1
            break
    while True:
        l_ = totl = 0
        for i in range(g):
            totl += data[i]
            l_ += data[i] * i
        hh = toth = 0
        for i in range(g + 1, n):
            toth += data[i]
            hh += data[i] * i
        if totl > 0 and toth > 0:
            l_ //= totl
            hh //= toth
            if g == java_round((l_ + hh) / 2.0):
                break
        g += 1
        if g > n - 2:
            return -1
    return g


# ------------------------------------------------------------------ Li
def li(h) -> int:
    data = _hist(h)
    n = len(data)
    tolerance = 0.5
    num_pixels = sum(data)
    mean = 0.0
    for ih in range(n):
        mean += float(ih) * data[ih]
    mean /= num_pixels
    new_thresh = mean
    threshold = 0
    while True:
        old_thresh = new_thresh
        threshold = int(old_thresh + 0.5)
        sum_back = num_back = 0
        for ih in range(threshold + 1):
            sum_back += ih * data[ih]
            num_back += data[ih]
        mean_back = 0.0 if num_back == 0 else sum_back / float(num_back)
        sum_obj = num_obj = 0
        for ih in range(threshold + 1, n):
            sum_obj += ih * data[ih]
            num_obj += data[ih]
        mean_obj = 0.0 if num_obj == 0 else sum_obj / float(num_obj)
        # Math.log(0) = -inf and 0/-inf = 0 in Java; Python's math.log raises, so spell the IEEE result out
        lb = math.log(mean_back) if mean_back > 0 else -math.inf
        lo = math.log(mean_obj) if mean_obj > 0 else -math.inf
        denom = lb - lo
        num = mean_back - mean_obj
        if math.isnan(denom) or (math.isinf(denom) and math.isinf(num)):
            temp = math.nan
        elif denom == 0:
            temp = math.copysign(math.inf, num) if num != 0 else math.nan
        else:
            temp = num / denom
        if math.isnan(temp):
            # (int) NaN is 0 in Java
            new_thresh = 0.0
        elif temp < -EPS:
            new_thresh = float(int(temp - 0.5)) if math.isfinite(temp) else float(-2147483648)
        else:
            new_thresh = float(int(temp + 0.5)) if math.isfinite(temp) else float(2147483647)
        if not (abs(new_thresh - old_thresh) > tolerance):
            break
    return threshold


# ------------------------------------------------------------------ entropy methods (shared prefix)
def _norm_and_cum(data: list[int]) -> tuple[list[float], list[float], list[float], int, int]:
    n = len(data)
    total = sum(data)
    norm = [float(v) / total for v in data]
    p1 = [0.0] * n
    p2 = [0.0] * n
    p1[0] = norm[0]
    p2[0] = 1.0 - p1[0]
    for ih in range(1, n):
        p1[ih] = p1[ih - 1] + norm[ih]
        p2[ih] = 1.0 - p1[ih]
    first_bin = 0
    for ih in range(n):
        if not (abs(p1[ih]) < EPS):
            first_bin = ih
            break
    last_bin = n - 1
    for ih in range(n - 1, first_bin - 1, -1):
        if not (abs(p2[ih]) < EPS):
            last_bin = ih
            break
    return norm, p1, p2, first_bin, last_bin


def max_entropy(h) -> int:
    data = _hist(h)
    n = len(data)
    norm, p1, p2, first_bin, last_bin = _norm_and_cum(data)
    threshold = -1
    max_ent = DOUBLE_MIN_VALUE
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        for ih in range(it + 1):
            if data[ih] != 0:
                ent_back -= (norm[ih] / p1[it]) * math.log(norm[ih] / p1[it])
        ent_obj = 0.0
        for ih in range(it + 1, n):
            if data[ih] != 0:
                ent_obj -= (norm[ih] / p2[it]) * math.log(norm[ih] / p2[it])
        tot_ent = ent_back + ent_obj
        if max_ent < tot_ent:
            max_ent = tot_ent
            threshold = it
    return threshold


def renyi_entropy(h) -> int:
    data = _hist(h)
    n = len(data)
    norm, p1, p2, first_bin, last_bin = _norm_and_cum(data)
    # alpha = 1 (maximum entropy)
    threshold = 0
    max_ent = 0.0
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        for ih in range(it + 1):
            if data[ih] != 0:
                ent_back -= (norm[ih] / p1[it]) * math.log(norm[ih] / p1[it])
        ent_obj = 0.0
        for ih in range(it + 1, n):
            if data[ih] != 0:
                ent_obj -= (norm[ih] / p2[it]) * math.log(norm[ih] / p2[it])
        tot_ent = ent_back + ent_obj
        if max_ent < tot_ent:
            max_ent = tot_ent
            threshold = it
    t_star2 = threshold
    # alpha = 0.5
    threshold = 0
    max_ent = 0.0
    term = 1.0 / (1.0 - 0.5)
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        for ih in range(it + 1):
            ent_back += math.sqrt(norm[ih] / p1[it])
        ent_obj = 0.0
        for ih in range(it + 1, n):
            ent_obj += math.sqrt(norm[ih] / p2[it])
        prod = ent_back * ent_obj
        tot_ent = term * (math.log(prod) if prod > 0.0 else 0.0)
        if tot_ent > max_ent:
            max_ent = tot_ent
            threshold = it
    t_star1 = threshold
    # alpha = 2
    threshold = 0
    max_ent = 0.0
    term = 1.0 / (1.0 - 2.0)
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        for ih in range(it + 1):
            ent_back += (norm[ih] * norm[ih]) / (p1[it] * p1[it])
        ent_obj = 0.0
        for ih in range(it + 1, n):
            ent_obj += (norm[ih] * norm[ih]) / (p2[it] * p2[it])
        prod = ent_back * ent_obj
        tot_ent = term * (math.log(prod) if prod > 0.0 else 0.0)
        if tot_ent > max_ent:
            max_ent = tot_ent
            threshold = it
    t_star3 = threshold
    if t_star2 < t_star1:
        t_star1, t_star2 = t_star2, t_star1
    if t_star3 < t_star2:
        t_star2, t_star3 = t_star3, t_star2
    if t_star2 < t_star1:
        t_star1, t_star2 = t_star2, t_star1
    if abs(t_star1 - t_star2) <= 5:
        if abs(t_star2 - t_star3) <= 5:
            beta1, beta2, beta3 = 1, 2, 1
        else:
            beta1, beta2, beta3 = 0, 1, 3
    else:
        if abs(t_star2 - t_star3) <= 5:
            beta1, beta2, beta3 = 3, 1, 0
        else:
            beta1, beta2, beta3 = 1, 2, 1
    omega = p1[t_star3] - p1[t_star1]
    opt = t_star1 * (p1[t_star1] + 0.25 * omega * beta1) + 0.25 * t_star2 * omega * beta2 + t_star3 * (p2[t_star3] + 0.25 * omega * beta3)
    return int(opt)  # (int) truncation


def shanbhag(h) -> int:
    data = _hist(h)
    n = len(data)
    norm, p1, p2, first_bin, last_bin = _norm_and_cum(data)
    threshold = -1
    min_ent = 1.7976931348623157e308
    for it in range(first_bin, last_bin + 1):
        ent_back = 0.0
        term = 0.5 / p1[it]
        for ih in range(1, it + 1):
            ent_back -= norm[ih] * math.log(1.0 - term * p1[ih - 1])
        ent_back *= term
        ent_obj = 0.0
        term = 0.5 / p2[it]
        for ih in range(it + 1, n):
            ent_obj -= norm[ih] * math.log(1.0 - term * p2[ih])
        ent_obj *= term
        tot_ent = abs(ent_back - ent_obj)
        if tot_ent < min_ent:
            min_ent = tot_ent
            threshold = it
    return threshold


def yen(h) -> int:
    data = _hist(h)
    n = len(data)
    total = sum(data)
    norm = [float(v) / total for v in data]
    p1 = [0.0] * n
    p1[0] = norm[0]
    for ih in range(1, n):
        p1[ih] = p1[ih - 1] + norm[ih]
    p1_sq = [0.0] * n
    p1_sq[0] = norm[0] * norm[0]
    for ih in range(1, n):
        p1_sq[ih] = p1_sq[ih - 1] + norm[ih] * norm[ih]
    p2_sq = [0.0] * n
    p2_sq[n - 1] = 0.0
    for ih in range(n - 2, -1, -1):
        p2_sq[ih] = p2_sq[ih + 1] + norm[ih + 1] * norm[ih + 1]
    threshold = -1
    max_crit = DOUBLE_MIN_VALUE
    for it in range(n):
        a = p1_sq[it] * p2_sq[it]
        b = p1[it] * (1.0 - p1[it])
        crit = -1.0 * (math.log(a) if a > 0.0 else 0.0) + 2 * (math.log(b) if b > 0.0 else 0.0)
        if crit > max_crit:
            max_crit = crit
            threshold = it
    return threshold


# ------------------------------------------------------------------ Mean / MinError
def mean(h) -> int:
    data = _hist(h)
    tot = 0
    s = 0
    for i, v in enumerate(data):
        tot += v
        s += i * v
    if tot == 0:
        return -1
    return int(math.floor(float(s // tot)))  # long division, then floor of the (integral) double


def _a(y: list[int], j: int) -> float:
    x = 0.0
    for i in range(j + 1):
        x += y[i]
    return x


def _b(y: list[int], j: int) -> float:
    x = 0.0
    for i in range(j + 1):
        x += _int32(i * y[i])
    return x


def _c(y: list[int], j: int) -> float:
    x = 0.0
    for i in range(j + 1):
        x += _int32(_int32(i * i) * y[i])
    return x


def _jdiv(a: float, b: float) -> float:
    """Java double division (no ZeroDivisionError)."""
    if b == 0.0:
        if a == 0.0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a) * (1.0 if math.copysign(1.0, b) > 0 else -1.0)
    return a / b


def min_error(h) -> int:
    data = _hist(h)
    n = len(data)
    threshold = mean(data)
    if threshold < 0:
        return threshold
    t_prev = -2
    while threshold != t_prev:
        if threshold > n - 1:  # Java would throw ArrayIndexOutOfBounds here; stop instead
            break
        at = _a(data, threshold)
        an = _a(data, n - 1)
        mu = _jdiv(_b(data, threshold), at)
        nu = _jdiv(_b(data, n - 1) - _b(data, threshold), an - at)
        p = _jdiv(at, an)
        q = _jdiv(an - at, an)
        sigma2 = _jdiv(_c(data, threshold), at) - mu * mu
        tau2 = _jdiv(_c(data, n - 1) - _c(data, threshold), an - at) - nu * nu
        w0 = _jdiv(1.0, sigma2) - _jdiv(1.0, tau2)
        w1 = _jdiv(mu, sigma2) - _jdiv(nu, tau2)
        arg = _jdiv(sigma2 * (q * q), tau2 * (p * p))
        log10 = math.log10(arg) if arg > 0 else (-math.inf if arg == 0 else math.nan)
        w2 = _jdiv(mu * mu, sigma2) - _jdiv(nu * nu, tau2) + log10
        sqterm = (w1 * w1) - w0 * w2
        if sqterm < 0:
            break
        t_prev = threshold
        temp = _jdiv(w1 + (math.sqrt(sqterm) if not math.isnan(sqterm) else math.nan), w0)
        if math.isnan(temp):
            threshold = t_prev
        elif math.isinf(temp):
            threshold = 2147483647 if temp > 0 else -2147483648
            break
        else:
            threshold = int(math.floor(temp))
            if threshold < 0:
                break
    return threshold


# ------------------------------------------------------------------ Moments
def moments(h) -> int:
    data = _hist(h)
    n = len(data)
    total = 0.0
    for v in data:
        total += v
    histo = [float(v) / total for v in data]
    m0, m1, m2, m3 = 1.0, 0.0, 0.0, 0.0
    for i in range(n):
        di = float(i)
        m1 += di * histo[i]
        m2 += di * di * histo[i]
        m3 += di * di * di * histo[i]
    cd = m0 * m2 - m1 * m1
    c0 = _jdiv(-m2 * m2 + m1 * m3, cd)
    c1 = _jdiv(m0 * -m3 + m2 * m1, cd)
    disc = c1 * c1 - 4.0 * c0
    sq = math.sqrt(disc) if disc >= 0 else math.nan
    z0 = 0.5 * (-c1 - sq)
    z1 = 0.5 * (-c1 + sq)
    p0 = _jdiv(z1 - m1, z1 - z0)
    s = 0.0
    for i in range(n):
        s += histo[i]
        if s > p0:
            return i
    return -1


# ------------------------------------------------------------------ Otsu (Celebi's formulation, as in ImageJ ≥ 1.52)
def otsu(h) -> int:
    data = _hist(h)
    n = len(data)
    num_pixels = sum(data)
    term = 1.0 / float(num_pixels)
    histo = [term * v for v in data]
    cnh = [0.0] * n
    cnh[0] = histo[0]
    for ih in range(1, n):
        cnh[ih] = cnh[ih - 1] + histo[ih]
    mn = [0.0] * n
    for ih in range(1, n):
        mn[ih] = mn[ih - 1] + ih * histo[ih]
    total_mean = mn[n - 1]
    threshold = -2147483648
    max_bcv = 0.0
    for ih in range(n):
        bcv = total_mean * cnh[ih] - mn[ih]
        bcv = bcv * _jdiv(bcv, cnh[ih] * (1.0 - cnh[ih]))
        if max_bcv < bcv:
            max_bcv = bcv
            threshold = ih
    return threshold


# ------------------------------------------------------------------ Percentile
def percentile(h) -> int:
    data = _hist(h)
    n = len(data)
    ptile = 0.5
    threshold = -1
    total = _a(data, n - 1)
    temp = 1.0
    for i in range(n):
        a = abs(_jdiv(_a(data, i), total) - ptile)
        if a < temp:
            temp = a
            threshold = i
    return threshold


# ------------------------------------------------------------------ Triangle
def triangle(h) -> int:
    data = _hist(h)
    n = len(data)
    mn = 0
    for i in range(n):
        if data[i] > 0:
            mn = i
            break
    if mn > 0:
        mn -= 1
    min2 = 0
    for i in range(n - 1, 0, -1):
        if data[i] > 0:
            min2 = i
            break
    if min2 < n - 1:
        min2 += 1
    dmax = 0
    mx = 0
    for i in range(n):
        if data[i] > dmax:
            mx = i
            dmax = data[i]
    inverted = False
    if (mx - mn) < (min2 - mx):
        inverted = True
        data = data[::-1]
        mn = n - 1 - min2
        mx = n - 1 - mx
    if mn == mx:
        return mn
    nx = float(data[mx])
    ny = float(mn - mx)
    d = math.sqrt(nx * nx + ny * ny)
    nx /= d
    ny /= d
    d = nx * mn + ny * data[mn]
    split = mn
    split_distance = 0.0
    for i in range(mn + 1, mx + 1):
        new_distance = nx * i + ny * data[i] - d
        if new_distance > split_distance:
            split = i
            split_distance = new_distance
    split -= 1
    if inverted:
        return n - 1 - split
    return split


METHODS = {
    "Default": default_isodata, "IJ_IsoData": ij_isodata, "Huang": huang, "Intermodes": intermodes, "IsoData": isodata,
    "Li": li, "MaxEntropy": max_entropy, "Mean": mean, "MinError": min_error, "Minimum": minimum, "Moments": moments,
    "Otsu": otsu, "Percentile": percentile, "RenyiEntropy": renyi_entropy, "Shanbhag": shanbhag, "Triangle": triangle,
    "Yen": yen,
}


def get_threshold(method: str, h, subtract_one: bool = False) -> int:
    """``AutoThresholder.getThreshold(Method, int[])``: the bilevel short-cut, the method, -1 → 0."""
    t = bilevel(h, subtract_one)
    if t >= 0:
        return t
    t = METHODS[method](h)
    if t == -1:
        t = 0
    return t
