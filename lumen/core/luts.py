"""Look-up tables.  Each LUT is a (256, 3) uint8 array shared by the server-side RGB
flattening and the browser compositor (served as JSON once at startup).

The built-in set mirrors Fiji's ``luts/`` folder (Image ▸ Lookup Tables): ImageJ's core tables
(Grays, Fire, Ice, Spectrum, 3-3-2 RGB, Red/Green, HiLo – byte-identical to ``ij.plugin.LutLoader``),
the Fiji files decoded byte for byte (16 colors, Thermal, Phase, Glasbey and its on-dark / inverted
variants, the WCIF "Hot" family Red/Orange/Yellow/Cyan/Magenta Hot, Green Fire Blue, Rainbow RGB,
the NIH Image / NCSA PalEdit tables Royal, Sepia, Thal, Thallium, Union Jack, ICA/ICA2/ICA3,
brgbcmyw, 5 ramps, 6 shades, blue orange icb, cool, edges, gem, glow, physics, smart) and the
matplotlib perceptual maps (also under their ``mpl-*`` Fiji names).

Fiji's ``glasbey.lut`` starts with white, so label images use :data:`LABEL_LUT` (black at 0)."""
from __future__ import annotations

import base64
import zlib
from typing import Any

import numpy as np

_LUTS: dict[str, np.ndarray] = {}


def _ramp(r: float, g: float, b: float) -> np.ndarray:
    x = np.arange(256, dtype=np.float32)
    return np.stack([x * r, x * g, x * b], axis=1).clip(0, 255).astype(np.uint8)


def _interp(points: list[tuple[int, int, int]]) -> np.ndarray:
    """Interpolate ImageJ-style control points (len n) to 256 entries."""
    pts = np.asarray(points, dtype=np.float32)
    n = len(pts)
    xs = np.linspace(0, 255, n)
    out = np.zeros((256, 3), np.float32)
    for i in range(3):
        out[:, i] = np.interp(np.arange(256), xs, pts[:, i])
    return out.round().clip(0, 255).astype(np.uint8)


def _ij_interp(points: list[tuple[int, int, int]]) -> np.ndarray:
    """``ij.plugin.LutLoader.interpolate()``: knot ``k`` sits at index ``k * 256 / n`` (``scale = n / 256``,
    ``i1 = (int)(i * scale)``), the last knot is held flat (indices 248 … 255 for 32 knots) and every value is
    truncated to a byte – unlike :func:`_interp`, which spreads the knots over 0 … 255 and rounds."""
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    i = np.arange(256)
    scale = n / 256.0
    i1 = (i * scale).astype(int)
    i2 = np.minimum(i1 + 1, n - 1)
    frac = (i * scale - i1)[:, None]
    return np.floor((1.0 - frac) * pts[i1] + frac * pts[i2]).clip(0, 255).astype(np.uint8)


def _piecewise(knots: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Interpolate ``(index, r, g, b)`` control points that sit at explicit indices 0 … 255."""
    pts = np.asarray(knots, dtype=np.float32)
    xs = np.arange(256)
    out = np.stack([np.interp(xs, pts[:, 0], pts[:, c]) for c in (1, 2, 3)], axis=1)
    return out.round().clip(0, 255).astype(np.uint8)


def _unpack_blob(blob: str) -> np.ndarray:
    """Decode a table stored as base64(zlib(byte deltas of the R, G, B planes)) – see ``_FIJI_BLOBS``."""
    deltas = np.frombuffer(zlib.decompress(base64.b64decode(blob)), np.uint8).reshape(3, 256)
    return np.cumsum(deltas, axis=1, dtype=np.uint8).T.copy()  # uint8 cumsum wraps modulo 256, undoing the deltas


def _random_lut(seed: int = 12345) -> np.ndarray:
    """Random label colours: index 0 black, then 255 distinct colours that show on black (fixed seed → reproducible)."""
    rng = np.random.default_rng(seed)
    out = np.zeros((256, 3), np.uint8)
    seen = {(0, 0, 0)}
    k = 1
    while k < 256:
        c = tuple(int(v) for v in rng.integers(0, 256, 3))
        if max(c) < 96 or c in seen:
            continue
        seen.add(c)
        out[k] = c
        k += 1
    return out


# ------------------------------------------------------------------ Fiji luts/ folder
# Exact copies of the tables Fiji ships, as ImageJ's LutLoader decodes each .lut file (NIH-header tables
# included).  Piecewise-linear tables are stored as their control points, the others as compressed byte
# deltas of the colour planes (``_unpack_blob``).  Names are the Fiji file names without extension.
_FIJI_KNOTS: dict[str, list[tuple[int, int, int, int]]] = {
    "5 ramps": [(0, 0, 0, 1), (1, 0, 0, 5), (50, 0, 0, 250), (51, 0, 1, 0), (52, 0, 5, 0), (101, 0, 250, 0),
        (102, 1, 1, 0), (103, 5, 5, 0), (152, 250, 250, 0), (153, 1, 0, 0), (154, 5, 0, 0), (203, 250, 0, 0),
        (204, 1, 1, 1), (205, 5, 5, 5), (255, 255, 255, 255)],
    "6 shades": [(0, 0, 0, 0), (1, 3, 0, 0), (64, 255, 0, 0), (65, 247, 7, 0), (95, 7, 247, 0), (96, 0, 255, 0),
        (97, 0, 247, 7), (127, 0, 7, 247), (128, 0, 0, 255), (129, 0, 7, 255), (160, 0, 255, 255), (161, 7, 247, 255),
        (191, 247, 7, 255), (192, 255, 0, 255), (193, 255, 7, 247), (223, 255, 247, 7), (224, 255, 255, 0),
        (225, 255, 255, 7), (255, 255, 255, 247)],
    "blue orange icb": [(0, 1, 2, 1), (1, 1, 2, 1), (63, 1, 64, 1), (64, 2, 64, 2), (95, 64, 33, 64), (96, 65, 32, 65),
        (127, 127, 1, 127), (128, 129, 1, 125), (158, 189, 61, 65), (159, 191, 63, 64), (160, 192, 64, 62),
        (190, 252, 124, 2), (191, 254, 126, 1), (192, 254, 128, 4), (207, 254, 158, 64), (208, 254, 160, 67),
        (223, 254, 190, 127), (224, 254, 191, 131), (239, 254, 221, 191), (240, 254, 223, 194), (254, 254, 251, 250),
        (255, 254, 251, 250)],
    "brgbcmyw": [(0, 0, 0, 0), (31, 0, 0, 0), (32, 255, 0, 0), (63, 255, 0, 0), (64, 0, 255, 0), (95, 0, 255, 0),
        (96, 0, 0, 255), (127, 0, 0, 255), (128, 0, 255, 255), (159, 0, 255, 255), (160, 248, 0, 255),
        (164, 248, 0, 255), (165, 249, 0, 255), (169, 249, 0, 255), (170, 250, 0, 255), (173, 250, 0, 255),
        (174, 251, 0, 255), (178, 251, 0, 255), (179, 252, 0, 255), (182, 252, 0, 255), (183, 253, 0, 255),
        (187, 253, 0, 255), (188, 254, 0, 255), (191, 254, 0, 255), (192, 255, 255, 0), (223, 255, 255, 0),
        (224, 255, 255, 255), (255, 255, 255, 255)],
    "cool": [(0, 1, 3, 2), (1, 1, 3, 2), (31, 1, 63, 62), (32, 1, 64, 64), (33, 1, 66, 65), (64, 1, 128, 127),
        (65, 2, 126, 129), (96, 64, 64, 191), (97, 65, 63, 192), (128, 127, 1, 254), (129, 129, 2, 250),
        (143, 157, 30, 194), (144, 159, 32, 191), (160, 191, 64, 127), (161, 192, 65, 123), (175, 220, 93, 67),
        (176, 222, 95, 64), (191, 252, 125, 4), (192, 254, 127, 1), (193, 254, 129, 4), (208, 254, 159, 64),
        (209, 254, 161, 67), (224, 254, 191, 127), (225, 254, 192, 131), (240, 254, 222, 191), (241, 254, 224, 194),
        (254, 254, 250, 246), (255, 254, 250, 246)],
    "edges": [(0, 1, 1, 1), (106, 1, 1, 1), (116, 51, 51, 51), (117, 61, 61, 61), (118, 90, 90, 90),
        (119, 105, 105, 1), (120, 120, 120, 120), (124, 180, 180, 180), (125, 194, 194, 194), (126, 204, 204, 204),
        (127, 224, 224, 224), (128, 253, 253, 253), (129, 224, 224, 224), (130, 204, 204, 204), (131, 194, 194, 194),
        (132, 180, 180, 180), (137, 105, 105, 105), (138, 80, 80, 80), (139, 70, 70, 70), (140, 61, 61, 61),
        (141, 51, 51, 51), (151, 1, 1, 1), (255, 1, 1, 1)],
    "Sepia": [(0, 0, 0, 0), (1, 43, 28, 4), (2, 57, 35, 5), (3, 65, 35, 6), (21, 83, 53, 24), (22, 83, 53, 24),
        (46, 107, 77, 48), (47, 107, 77, 48), (65, 125, 95, 66), (66, 125, 95, 66), (128, 187, 157, 128),
        (129, 187, 157, 128), (197, 255, 225, 196), (201, 255, 225, 196), (202, 255, 225, 197), (205, 255, 225, 197),
        (206, 255, 226, 198), (210, 255, 226, 198), (211, 255, 226, 199), (215, 255, 226, 199), (216, 255, 227, 200),
        (220, 255, 227, 200), (221, 255, 227, 201), (225, 255, 227, 201), (226, 255, 228, 202), (231, 255, 228, 202),
        (232, 255, 228, 203), (236, 255, 228, 203), (237, 255, 229, 204), (242, 255, 229, 204), (243, 255, 229, 205),
        (247, 255, 229, 205), (248, 255, 230, 205), (249, 255, 230, 205), (250, 255, 231, 205), (251, 255, 231, 209),
        (252, 255, 233, 214), (253, 255, 233, 217), (254, 255, 242, 233), (255, 255, 255, 255)],
    "Union Jack": [(0, 0, 0, 0), (1, 0, 0, 2), (32, 0, 0, 95), (33, 0, 0, 97), (62, 0, 0, 184), (63, 0, 0, 190),
        (64, 0, 0, 190), (65, 3, 3, 191), (126, 247, 247, 252), (127, 255, 255, 255), (128, 255, 255, 255),
        (129, 253, 251, 251), (190, 192, 7, 7), (191, 190, 0, 0), (192, 190, 0, 0), (193, 191, 3, 3),
        (254, 252, 247, 247), (255, 255, 255, 255)],
}

_FIJI_BLOBS: dict[str, str] = {
    "Red Hot": (
        "eNpjYGBgYARiBmYQYAFCZigLgWEyqHxkeRZUPhMQMgwdAPb/f6qayMiIFkYseMKUGas8UpgyMjEyDSn/U9uBQP8TCE+SwhQUP1CjAW+z"
        "BP4="
    ),
    "Orange Hot": (
        "eNpjYGKmOmBnGEKAkYnqkIWBkWi1JBg60gENUiozCwCoMwL+"
    ),
    "Yellow Hot": (
        "eNpjYGJiBiE0AlOEBGE2BgZGEKQqwcRAG8DIxMQEwtgIXOIEJFmB7gUCJiIJ4hQyM4x0wMRMfcACAJ9VAv4="
    ),
    "Cyan Hot": (
        "eNpjYBgygImZ+oCFgZGRCQyhFB4G0SqAhsIkmJjIYWITYxk6oco+dNIUAwCmdwL+"
    ),
    "Magenta Hot": (
        "eNpjYGQCQfwEbikWBiYmZhBCIzBFsAuzMox0MNDhP8LjHwCpMQL+"
    ),
    "Green Fire Blue": (
        "eNpjYBhgwMxCCeCg2H5GRkYm8hEzAxMzEzN+hE+SbaCDn2L3//37Dx/6h1/+90CnHwC++UH+"
    ),
    "Rainbow RGB": (
        "eNq9jlEKwCAMQ6M2u/9963QKzo/ROvezBB4PQqFAT8DIFNNeZtdianUAcwmrExul1NJTJ3b0fvEDSCENiDewnK0OYC91cWIDyxxsfUIO"
        "gl7EG1SzqoXsDfgjKc5uaBx6AckQzbk="
    ),
    "Royal": (
        "eNq1UtEKxSAIbU6dwfr/32zPK7mWjI3uw7hs99jxZIJ4oCn8GVFiNIorM0URKxZxfTxfVas6mn71AWAGR9OxPd3Nt42ZeyILRKTj3hLS"
        "bI9nzT/vW7UexxulXmKHspdystfXCGlNaTWmTiIc/N74X8LL/gbknLfsaLq9/r8e+v8AYdZt+w=="
    ),
    "Thal": (
        "eNq9jr0OwCAIhFVItJO8/1PSHY0VXcrQTvKFn+HCcTE4g28Av7E6SJOndEhv0q1/Qkh6Bmv8xIne+W9mns17e/83dlcoSi6b0/m9IaJK"
        "i7mrkQdlhjL7"
    ),
    "Thallium": (
        "eNrdUUkOgDAI7LBo9Ob/H6tooKsxnjzIMDAtaQsp0neGjISickIaQQA6crVihzI1EBFdjAZR1lCbvTH09NBtNHV6peIccqp4A5S7guSy"
        "TgQvhyUftgVT3jPjK4jnCGI1PVXru07v5//h/68z9T0dzmgD+A=="
    ),
    "ICA": (
        "eNo78+8/CP5HomjG/v/37z8ohDP+scIBGyaLDZskAzXBmX9Q8J8ODGwOYEYDLIQEWBgZGZmIQEQpYvrPQEfw+fPnL+iYdFOYmJmY0RCG"
        "ANFiAEt4Ato="
    ),
    "ICA2": (
        "eNrVikEOwCAIBBdQ//9ZvQn15AFNJLaXDhtChrWuXWeO56uOYkOelPUquye+xAx+EFBXHdQ2cEsc6SQSEXEgoRLjf7CwuCwi7B6koLXL"
    ),
    "ICA3": (
        "eNr7/58BHTIQIUSWGoav34AAjeDh5eVFxwz0A2f+QcF/OjCwOYAVDtjgDEZGJjCkPcX0f6DjH1uYMDLBAIKFnUmpPBMAOej8yw=="
    ),
    "gem": (
        "eNqlj0EOACAIwxjw/x8rctUokqw7QbMEICTQmp9Xsk+en31QCS75XzO4bcW0E32Z/B8UZB0ya6LWw3znnL32sgCvrmH1"
    ),
    "glow": (
        "eNqNkgEOgzAMA+O4b94j9mmW2CkCpkkrlDhOqcqRQNSFTNbNBIkkqNlem5kr0yvSYxfIqJX9UiW9Sw/U1Qm0W20edkpViIjW9awHpBAO"
        "6KIEHNPxUtrm4SSxV8S9/u/g+3V99yl8LAtsBZ8P8EGhHPtzztkENBVoBickAQY42twEFkNQljyErcEasdbANFVt61NgS2Dj2ezs3eAg"
        "eGd3qoF4COgveMf8yYf5O/s2Ji0O8/Pmgwaj0InCEBIwtynP0C3LpRWdUe2o7lTLagVVAadtba/kB7bSCwA="
    ),
    "physics": (
        "eNqVUlsSwyAIFLS5To/XuxeBanyMD8Zp1mVZ9cOdkLeqSilpK7OJqwCAXE1Hk6wBdTb0eJA2r4Sr1DUs7zET/czgfcA7oOu62tMVnnMB"
        "FiyPT1WkNwthDwQPOWWfN8os3BF59XGU2NvND2AdYuu7QTBmPmL/2qvUhMItqhBRpA1fIrUmov/+Yvrw5AdyeZQc"
    ),
    "smart": (
        "eNq9UF0PgyAM7AcgiyTzbf//H+4dpGtBjTOL7sk7jtCmDe0RACIedB7+UwG/mq4kjEikb0JiZhOza3y9nYYqpQEbtZaq6809bLNQy0AP"
        "lGege/Yvc6mqUqvUBqkCokf5ddkxrVMjrru2Qe1Bjs0EtUKd8d55M4njI8QUfAiBFue0Cg0EtE6pdNrLgAtu2z+XknOe6wrZLS/b+tLS"
        "Ozcu0D86Xhu7WwoziHmIcUgpjWl6xnGcJs3DB8DpSvg="
    ),
    "16 colors": (
        "eNpjYEAFjAz4wX8S1aODvWh8RxL1/ybARwcv0fiKBNQT8k8uGt8WjW+Kxpcj0fyHaPy1aPyLaPxFBMzbgsb3JqB+NQH/yBNID+iAmUT1lIJ2NjY2"
        "diT8i0D8AwByMw7+"
    ),
    "Thermal": (
        "eNq1UgFuwyAMBEOgWf+1D+3xKcbgnU2qttImVZp2GBQCOs5nf4a3oEFVp07HGIhxrrab8/luTClvW9lqqRfDbrHv/l1rLaVsOadEMTqpkUjv0rk3"
        "Zm7cWuPjuB03oNkv6biiGojAWyu49uvHFUTGZXQIB2hBnIjCP4BSzngfj+HdarjYWGndEzMN9K6lL6b9SZuGGeajQDIEjnY4ugx1S33alhkn0kVk"
        "jPAVIjkSJQvzL1tguLWeEoz9yVRV74o1HWdacYHO8VtyVnsZfYldatkVs9ce1XfN3boDHfBcDOgF4rmsVxzLSvWODXd95ow+whY7w6VvY8bqBA=="
    ),
    "Phase": (
        "eNpjYGBgASEyCfJ1wggooIBFqdaB9f8fEEIjMEVIEyZB/x94oOBl0U5ygP0PAEjIcf0="
    ),
    "Glasbey": (
        "eNoBAAP//P8B/wEA/wH/AZpmeKfgskAN30NSBIzGwHeGHHPs3LdtADm1Eqv2A1tIjSewyjFapb2eAFZJo70B/KMIo7aRPvQ6RShLz9yfZ5jmLcGJ"
        "xZQuHl1JuwEcgv8IreA4a8mWkafIHrolNKycnQoif4Eo8+UpM8tYbzl8/SrHXbOH5/Im0lMzfPhzwnBhL2mW6taGZmPydolfkLGxbUNY77pin4kg"
        "4shBav/fjbOPUgGK5XIe5o39F90GXwn1iV/8HTiT1AntUCb1OPs3zQBGhf59T4PybPTYZCun3a0rdRBtXVzKyi8nvRhAQZDH6akAAMt5p9VnLySF"
        "jQJzP8ASifvpvIP/AQD/AQBTgMyuskBXFiA8h3Ea5mxBUm1Lzn72Thd2ePZlyZD1nW3PMq3JRkQAc+on9YY2yy3Fa6LAldPpPkcmiQqzgmbwltwZ"
        "n8S69cUZtPmDdoGFvn7ByOITAUUoUg76Inw2qjA2VenRZ7HIb+FaGSpYxZJuTbghelPI1OGxUywBo0SIqV07aAaqA/e/qSIpqStgjq9QypDdcnEN"
        "056FrSoprJnyuyiTTDqdqv2BZTgoQjedAEg2LpU+iCQIDSASZ4qZrerbji88qWxGsBSfCdkP7UiPAf9SCn1B6TeXkzk+XccBouF2cFOxlv0AbT37"
        "h439NrlZPtXMPShb5YRv/wABADODSgD/Q3wD12V06+a9AlRAj9wAn8iZ/5/1bABRGpRmaDN0/qSwqpHXw2SwVKbbfT8DeYg5PllZOseT/0Hg4FQ/"
        "bdNswX8v3fHuldNCMLvKscB9SeJXURaR0ICljhC8NAWEwv5nlgRzsUUCTSVbygFONmtBmxCI3Ab/pOi8dr4IhrHfeahrWjqKpWMxe9xmSRGxl3Ps"
        "/uIHOJ1YkdqIDWx7UP0eXkYXm8SdYFDsxmbLINDg/1f3Wb9K/aJqR6ATsDGLb8ETA7M9nBA64l2H/yexxIg1kG87Uz97MJo30V6Kyb3teRztA6dR"
        "2u+bUzPPnDaFaGr9zpzCmlodALgReKc="
    ),
    "Glasbey on dark": (
        "eNoBAAP//AD/AAGcvJkj7P+SzKP/eYiMxn3OIkGjtRMxZFyk8N3plki+9yVBIS/VK2F+IsU7xfmknaFfbPoXg5Z97cTJ/v0w7G3vMR2A"
        "kWf+PYtFrxjXFOI6Lna9iQjd8KNg9KyJUupL1lvpPpcDlfxkzD++QJxcB2pPzFL79BnFtw9RuI54JTonAu4ppMOBSo/Y853xiVRHNHd9"
        "VLsSuEQW+7wr2ONm7e8bsQByHhXMI8DbEsVG2n166wJ076vWmvumUckRxuQy8v4KLsEKJHjl2Ch0jXLeUKSK9HeoD8BPdWKPmAPzsWUt"
        "erZy5/HP9I3lfw3Ztuu/Hbx8AXQ3L9OTv93M1ZQA/xp6rcDrYHHj8XD/AWTlP0fJaNOYyBO5rNmM8FFZyEx++i8f5U8bLHznnp/ZiQCi"
        "tzY9CmRV83zYmVOjmrGxb8gkqdMZbn9IeVCz+elpsqrs7si0rMvf3kCCfoyppmcfu+ncFfZFN9u2kzIjCCAY6H+pUENYcPHl3siefLmJ"
        "XVD32Qv1W08BlqsZx03ZucSjApeeEBbLUWm2o5xb29wzM2OB28aouRXbraHWv/nCu9UhoO329zbW7y5NbjWV/UwzZjxUHvMeHZrruuwM"
        "dJPKs9z5hGwPRR3sYpcUgGtB55kh4/0abKMxDEI8oGeFONwduP2cHk2YNtdrkG1ZaPbsAAD/lG3HOAEBYZ1Al4kY6eq1u/fnOonQO0iT"
        "v49xAGQpcmX5SBs/vy+cd51cCP4eaxViAWHot31GE7dio4w/3rrMAAG07WY2AAA6NRoX6RcynCTGJTgKm6dKtQBj8Vy8YO2d2NNXrVs4"
        "aR2h1pM/mgDzp2YtYNWdBPyDqB+3vcHvYYMTJVHmHPQw/4h4sgmc4VG8OYOo9NLgp7OLdvPgUAMxe8VRb9+2bAGb2PgU1HP/IStcxISr"
        "dHBgSHYVZ1oDMJIBWXPsRrFU/Mv9rB4F4rAURGGd1efF76a2ZW5oYxjdnvX8RMgdEVl2KTjtfrmjYQTMT9CxyZ87ijWEwI2yz01Pj9g="
    ),
    "Glasbey inverted": (
        "eNoVkBHYxiAURi+GYRiGF8MwDMPwYhiGF8MwDIfDcDgcDofD4f/9/j7nnOcFkG80Aj0AmUeiV7hH/1oL1qK4XHoQtGy3cW8ota782sf0"
        "aZIWEpMA4XeRhEE6oX/wYT3ijHRkgZDouhSJ87dqt/XgYmUQGDbrAsRXuHS629hbeil1vqHFrkUEf0V8MLyQUAuoh/KfS4D6TpnLL+yV"
        "ssGTX/zKpx9tn7sKqev3Fvn99le4bG9ZgMoZhitworuLVBFcGFGgUa8pAP41lhZKwKgjdJlI2FCUrhCRYKgOFmIOxlp6Vmfr+CWdpW/7"
        "c5ZzxyiiJ2M16pQsGstBypq0zouziO0j/ftsxMe6seZrva/CUgA1t/W7Yr8rVO0MSyPYGBe8YSXiL3N7CQgpRNlPImuUneuYLH2B6pKQ"
        "B9/HNjdfRDRZeIf35uKHMSVp2n8NMDq1IGO6Uo6PCeNqp/clPYUUGn52CKKpbP49QQGXTiP1GZdt5cxm9/uGtFXuzObK2AbRJ9qrtszM"
        "XS23aqGZeQuduPs0Shqtt3D+O6L58Gwit7NTGbiaLXHSlubUK8aPiz3c6CxWHzWVmYYcsQ+OZffL03YWceur+R72GRqHvepZWjlUP+xh"
        "89YrznSGJfbGM3HhErjWH6Fsfo+lbu3HX022fDGXHI5iiTp9dgx3mPTzm8vvVPzcKnWmWv8A5Mc56A=="
    ),
}

# Order of the Fiji tables in Image ▸ Lookup Tables (after ImageJ's core set and the matplotlib maps).
_FIJI_ORDER = ("Red Hot", "Orange Hot", "Yellow Hot", "Cyan Hot", "Magenta Hot", "Green Fire Blue", "Rainbow RGB",
               "Royal", "Sepia", "Thal", "Thallium", "Union Jack", "ICA", "ICA2", "ICA3", "brgbcmyw", "5 ramps",
               "6 shades", "blue orange icb", "cool", "edges", "gem", "glow", "physics", "smart")
# Fiji's mpl-*.lut files are the same 256 matplotlib samples quantised by truncation (Lumen's own names round).
_MPL_ALIASES = (("mpl-inferno", "Inferno"), ("mpl-magma", "Magma"), ("mpl-plasma", "Plasma"), ("mpl-viridis", "Viridis"))


def _register_builtin() -> None:
    _LUTS["Grays"] = _ramp(1, 1, 1)
    _LUTS["Red"] = _ramp(1, 0, 0)
    _LUTS["Green"] = _ramp(0, 1, 0)
    _LUTS["Blue"] = _ramp(0, 0, 1)
    _LUTS["Cyan"] = _ramp(0, 1, 1)
    _LUTS["Magenta"] = _ramp(1, 0, 1)
    _LUTS["Yellow"] = _ramp(1, 1, 0)
    _LUTS["Orange"] = _interp([(0, 0, 0), (255, 128, 0)])
    # ImageJ "Fire" / "Ice" control points (LutLoader.fire() / ice()), interpolated exactly like LutLoader does
    fire_r = [0, 0, 1, 25, 49, 73, 98, 122, 146, 162, 173, 184, 195, 207, 217, 229, 240, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
    fire_g = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 35, 57, 79, 101, 117, 133, 147, 161, 175, 190, 205, 219, 234, 248, 255, 255, 255, 255]
    fire_b = [0, 61, 96, 130, 165, 192, 220, 227, 210, 181, 151, 122, 93, 64, 35, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 98, 160, 223, 255]
    _LUTS["Fire"] = _ij_interp(list(zip(fire_r, fire_g, fire_b)))
    ice_r = [0, 0, 0, 0, 0, 0, 19, 29, 50, 48, 79, 112, 134, 158, 186, 201, 217, 229, 242, 250, 250, 250, 250, 251, 250, 250, 250, 250, 251, 251, 243, 230]
    ice_g = [156, 165, 176, 184, 190, 196, 193, 184, 171, 162, 146, 125, 107, 93, 81, 87, 92, 97, 95, 93, 93, 90, 85, 69, 64, 54, 47, 35, 19, 0, 4, 0]
    ice_b = [140, 147, 158, 166, 170, 176, 209, 220, 234, 225, 236, 246, 250, 251, 250, 250, 245, 230, 230, 222, 202, 180, 163, 142, 123, 114, 106, 94, 84, 64, 26, 27]
    _LUTS["Ice"] = _ij_interp(list(zip(ice_r, ice_g, ice_b)))
    # Spectrum: hue sweep – LutLoader.spectrum() uses Color.getHSBColor(i/255f, 1f, 1f), which rounds (+0.5)
    import colorsys
    spec = np.array([[int(v * 255 + 0.5) for v in colorsys.hsv_to_rgb(i / 255.0, 1, 1)] for i in range(256)], np.uint8)
    _LUTS["Spectrum"] = spec
    # 3-3-2 RGB
    i = np.arange(256)
    _LUTS["3-3-2 RGB"] = np.stack([(i & 0xE0), (i << 3) & 0xE0, (i << 6) & 0xC0], axis=1).astype(np.uint8)
    # Red/Green (LutLoader.redGreen(): black→red over 0 … 127, black→green over 128 … 255)
    rg = np.zeros((256, 3), np.uint8)
    rg[:128, 0] = np.arange(128) * 2
    rg[128:, 1] = np.arange(128) * 2
    _LUTS["Red/Green"] = rg
    # HiLo: saturation check
    hilo = _ramp(1, 1, 1)
    hilo[0] = (0, 0, 255)
    hilo[255] = (255, 0, 0)
    _LUTS["HiLo"] = hilo
    # 16 colors: Fiji's 16_colors.lut (16 bands of 16 entries; band 208 … 223 is a magenta→pink ramp, not a flat colour)
    _LUTS["16 colors"] = _unpack_blob(_FIJI_BLOBS["16 colors"])
    # Glasbey et al. (2007) label colours as Fiji ships them (glasbey.lut: index 0 is WHITE – see LABEL_LUT) and the
    # cyclic phase.lut (blue → gray → red → … → blue)
    _LUTS["Glasbey"] = _unpack_blob(_FIJI_BLOBS["Glasbey"])
    _LUTS["Phase"] = _unpack_blob(_FIJI_BLOBS["Phase"])
    # matplotlib perceptual maps
    try:
        import matplotlib

        for name, key in [("Viridis", "viridis"), ("Magma", "magma"), ("Inferno", "inferno"), ("Plasma", "plasma"),
                          ("Cividis", "cividis"), ("Turbo", "turbo"), ("Jet", "jet"), ("Hot", "hot"), ("Cool", "cool"),
                          ("Rainbow", "rainbow")]:
            cm = matplotlib.colormaps[key]
            _LUTS[name] = (cm(np.linspace(0, 1, 256))[:, :3] * 255).round().astype(np.uint8)
        for fiji_name, ours in _MPL_ALIASES:  # byte-identical to Fiji's mpl-*.lut files
            _LUTS[fiji_name] = (matplotlib.colormaps[ours.lower()](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    except Exception:  # pragma: no cover
        pass
    _LUTS["Thermal"] = _unpack_blob(_FIJI_BLOBS["Thermal"])  # Fiji's Thermal.lut (purple → blue → … → white → red)
    # Fiji's luts/ folder under the names Image ▸ Lookup Tables shows
    for name in _FIJI_ORDER:
        _LUTS[name] = _piecewise(_FIJI_KNOTS[name]) if name in _FIJI_KNOTS else _unpack_blob(_FIJI_BLOBS[name])
    for fiji_name, ours in _MPL_ALIASES:
        if fiji_name not in _LUTS and ours in _LUTS:  # pragma: no cover – matplotlib missing
            _LUTS[fiji_name] = _LUTS[ours].copy()
    # label colours: Glasbey et al. (2007) variants shipped with Fiji / ImageJ 1.54, and reproducible random colours
    _LUTS["Glasbey on dark"] = _unpack_blob(_FIJI_BLOBS["Glasbey on dark"])
    _LUTS["Glasbey inverted"] = _unpack_blob(_FIJI_BLOBS["Glasbey inverted"])
    _LUTS["Random"] = _random_lut()


_register_builtin()

LUT_NAMES = list(_LUTS.keys())

#: Image ▸ Lookup Tables / the Display section's LUT select list the tables in these groups (GUI fix round 2, E5-1 /
#: E6-9 / E7-13): ImageJ's core set, the Fiji ``luts/`` folder, the matplotlib perceptual maps and the label maps.  Every
#: built-in name appears in exactly one group (``test_luts``); tables registered at runtime (Import LUT, Edit LUT …) are
#: listed by :func:`lut_groups` under "Imported".
LUT_GROUPS: list[tuple[str, list[str]]] = [
    ("ImageJ", ["Grays", "Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Orange", "Fire", "Ice", "Spectrum", "3-3-2 RGB",
                "Red/Green", "HiLo"]),
    ("Fiji", ["16 colors", "Thermal", "Phase", "Red Hot", "Orange Hot", "Yellow Hot", "Cyan Hot", "Magenta Hot", "Green Fire Blue",
              "Rainbow RGB", "Royal", "Sepia", "Thal", "Thallium", "Union Jack", "ICA", "ICA2", "ICA3", "brgbcmyw", "5 ramps",
              "6 shades", "blue orange icb", "cool", "edges", "gem", "glow", "physics", "smart"]),
    ("Perceptual (matplotlib)", ["Viridis", "Magma", "Inferno", "Plasma", "Cividis", "Turbo", "Jet", "Hot", "Cool", "Rainbow",
                                 "mpl-viridis", "mpl-magma", "mpl-inferno", "mpl-plasma"]),
    ("Labels", ["Glasbey", "Glasbey on dark", "Glasbey inverted", "Random"]),
]

#: Near-duplicate names and what they are, so a menu can label them ("mpl-viridis – Fiji's file, = Viridis"): the Fiji
#: ``mpl-*.lut`` files are the matplotlib maps quantised by truncation, Thallium is Thal's NIH-Image twin, "cool" (Fiji) and
#: "Cool" (matplotlib) are different ramps.
LUT_ALIASES: dict[str, str] = {
    "mpl-viridis": "Fiji's mpl-viridis.lut (= Viridis, truncated)", "mpl-magma": "Fiji's mpl-magma.lut (= Magma, truncated)",
    "mpl-inferno": "Fiji's mpl-inferno.lut (= Inferno, truncated)", "mpl-plasma": "Fiji's mpl-plasma.lut (= Plasma, truncated)",
    "Thallium": "NIH Image table, like Thal", "cool": "Fiji's cool.lut (not matplotlib's Cool)", "Cool": "matplotlib (not Fiji's cool.lut)",
    "Glasbey": "index 0 is white (Fiji's glasbey.lut)", "Glasbey on dark": "index 0 is black – label images",
}


def lut_groups() -> list[dict[str, Any]]:
    """``[{"name": group, "luts": [names…]}, …]`` for the menu / select: the built-in groups in ``LUT_GROUPS`` order (names
    that exist right now only) plus an "Imported" group holding every runtime-registered table, in registration order."""
    grouped: set[str] = set()
    out: list[dict[str, Any]] = []
    for name, members in LUT_GROUPS:
        present = [m for m in members if m in _LUTS]
        grouped.update(present)
        if present:
            out.append({"name": name, "luts": present})
    rest = [n for n in LUT_NAMES if n not in grouped]
    if rest:
        out.append({"name": "Imported", "luts": rest})
    return out

#: The LUT label images are displayed with: a Glasbey table whose index 0 (the background label) is black.  Fiji's
#: own ``glasbey.lut`` ("Glasbey") starts with white, so applying it to a label image paints the background white –
#: exactly as in Fiji, where ``glasbey_on_dark.lut`` exists for dark backgrounds.
LABEL_LUT = "Glasbey on dark"


def get_lut(name: str) -> np.ndarray:
    if name not in _LUTS:
        raise KeyError(f"Unknown LUT {name!r}")
    return _LUTS[name]


def all_luts() -> dict[str, list[list[int]]]:
    return {k: v.tolist() for k, v in _LUTS.items()}


def apply_lut(plane: np.ndarray, lut: str, vmin: float, vmax: float, inverted: bool = False, gamma: float = 1.0) -> np.ndarray:
    """Map a 2-D plane through a display range + LUT → (H, W, 3) uint8."""
    table = get_lut(lut)
    if inverted:
        table = table[::-1]
    span = max(vmax - vmin, 1e-12)
    norm = (plane.astype(np.float32) - vmin) / span
    norm = np.clip(np.nan_to_num(norm, nan=0.0), 0, 1)
    if gamma != 1.0:
        norm = norm ** (1.0 / gamma)
    idx = (norm * 255 + 0.5).astype(np.uint8)
    return table[idx]


# ----------------------------------------------------------------- runtime LUTs
BUILTIN_NAMES = tuple(LUT_NAMES)

#: Bumped every time a table is added, replaced or removed at runtime.  ``Session.state()`` exposes it as
#: ``lut_version`` so the browser refetches ``/api/luts`` after an in-place replacement (the name list alone
#: does not change when a runtime LUT is overwritten under the same name).  Read it via :func:`lut_version`.
LUT_VERSION = 0


def lut_version() -> int:
    """The current LUT-table revision (monotonically increasing; see ``LUT_VERSION``)."""
    return LUT_VERSION


def _as_table(table) -> np.ndarray:
    """Coerce anything list-like into a (256, 3) uint8 table (3×N / N×3 / N×4 with an index column; N != 256 is resampled)."""
    arr = np.asarray(table, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"A LUT table must be 2-D, got shape {arr.shape}")
    if arr.shape[0] in (3, 4) and arr.shape[1] not in (3, 4):
        arr = arr.T
    if arr.shape[1] == 4:  # index, r, g, b
        arr = arr[:, 1:]
    if arr.shape[1] != 3:
        raise ValueError(f"A LUT table needs 3 (r g b) or 4 (index r g b) columns, got {arr.shape[1]}")
    if arr.shape[0] < 2:
        raise ValueError("A LUT table needs at least two entries")
    if arr.max() <= 1.0 and arr.min() >= 0.0 and not np.array_equal(arr, arr.round()):
        arr = arr * 255.0  # 0–1 floats
    if arr.shape[0] != 256:
        xs = np.linspace(0, 255, arr.shape[0])
        arr = np.stack([np.interp(np.arange(256), xs, arr[:, i]) for i in range(3)], axis=1)
    return np.clip(np.nan_to_num(arr).round(), 0, 255).astype(np.uint8)


def register_lut(name: str, table, replace: bool = False) -> str:
    """Add (or replace) a LUT at runtime and return the name it was stored under.

    ``LUT_NAMES`` is extended in place, so every ``choices=LUT_NAMES`` parameter and the
    ``/api/luts`` listing pick the new table up immediately.  Without ``replace`` an
    existing name is kept when its table is identical (re-importing the same file is a
    no-op) and otherwise made unique ("Name 2", "Name 3", …).  Built-in tables are never
    overwritten: a clashing name is uniquified the same way whatever ``replace`` says, so
    ``replace`` only updates *runtime* LUTs in place.
    """
    name = str(name or "").strip() or "Custom"
    tab = _as_table(table)
    if name in BUILTIN_NAMES:
        replace = False  # Grays, Fire, … stay intact process-wide (render.py relies on 'Grays' being a gray ramp)
    if name in _LUTS and not replace:
        if np.array_equal(_LUTS[name], tab):
            return name
        base, k = name, 2
        while name in _LUTS:
            name = f"{base} {k}"
            k += 1
    _LUTS[name] = tab
    if name not in LUT_NAMES:
        LUT_NAMES.append(name)
    _invalidate_caches()
    return name


def unregister_lut(name: str) -> bool:
    """Remove a runtime LUT (built-ins stay)."""
    if name in BUILTIN_NAMES or name not in _LUTS:
        return False
    del _LUTS[name]
    if name in LUT_NAMES:
        LUT_NAMES.remove(name)
    _invalidate_caches()
    return True


def is_builtin(name: str) -> bool:
    return name in BUILTIN_NAMES


def _invalidate_caches() -> None:
    """Called on every mutation of the LUT table: bump ``LUT_VERSION`` and drop the io module's cached tables."""
    global LUT_VERSION
    LUT_VERSION += 1
    import sys
    io_mod = sys.modules.get(__name__.rsplit(".", 1)[0] + ".io")
    if io_mod is not None and hasattr(io_mod, "_LUT_TABLE_CACHE"):
        io_mod._LUT_TABLE_CACHE = None


# ----------------------------------------------------------------- .lut files
_ICOL = 0x49434F4C  # 'ICOL' – NIH Image colour table header


def parse_lut_bytes(raw: bytes) -> np.ndarray:
    """Decode an ImageJ ``.lut`` file (the formats ``ij.plugin.LutLoader`` reads):

    * 768 bytes: 256 reds, 256 greens, 256 blues (raw binary),
    * 800 bytes: the same behind a 32-byte NIH Image 'ICOL' header (fewer than 256 colours are interpolated),
    * text: 256 lines of ``r g b`` or ``index r g b`` (whitespace / comma / tab separated, header lines allowed).
    """
    n = len(raw)
    if n >= 32 and int.from_bytes(raw[:4], "big") == _ICOL:
        ncolors = int.from_bytes(raw[6:8], "big") or 256
        body = raw[32:32 + 3 * ncolors]
        if len(body) < 3 * ncolors:
            raise ValueError("Truncated NIH Image LUT")
        arr = np.frombuffer(body, np.uint8).reshape(3, ncolors).T
        return _as_table(arr)
    if n in (768, 970):
        arr = np.frombuffer(raw[:768], np.uint8).reshape(3, 256).T
        return arr.copy()
    text = raw.decode("utf-8", errors="replace")
    return parse_lut_text(text)


def parse_lut_text(text: str) -> np.ndarray:
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "//", ";")):
            continue
        parts = [p for p in line.replace(",", " ").replace("\t", " ").split(" ") if p]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue  # header line (e.g. "Index Red Green Blue")
        if len(vals) in (3, 4):
            rows.append(vals)
    if len(rows) < 2:
        raise ValueError("Not a LUT: expected 256 lines of 'r g b' (or 'index r g b') values")
    width = {len(r) for r in rows}
    if len(width) != 1:
        raise ValueError("Every line of a text LUT must have the same number of columns")
    return _as_table(np.asarray(rows))


def read_lut_file(path) -> np.ndarray:
    from pathlib import Path
    return parse_lut_bytes(Path(path).read_bytes())


def lut_bytes(table) -> bytes:
    """Encode a table as a 768-byte ImageJ ``.lut`` (R plane, G plane, B plane)."""
    tab = _as_table(table)
    return tab.T.astype(np.uint8).tobytes()


def write_lut_file(table, path) -> str:
    from pathlib import Path
    p = Path(path)
    if p.suffix.lower() != ".lut":
        p = p.with_name(p.name + ".lut")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(lut_bytes(table))
    return str(p)


def gradient_lut(start, end) -> np.ndarray:
    """Linear ramp between two colours (hex strings, 'r,g,b' strings or 3-tuples)."""
    a, b = parse_color(start), parse_color(end)
    t = np.linspace(0, 1, 256)[:, None]
    return (np.asarray(a, float) * (1 - t) + np.asarray(b, float) * t).round().clip(0, 255).astype(np.uint8)


_NAMED_COLORS = {"black": (0, 0, 0), "white": (255, 255, 255), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
                 "cyan": (0, 255, 255), "magenta": (255, 0, 255), "yellow": (255, 255, 0), "orange": (255, 200, 0), "gray": (128, 128, 128),
                 "grey": (128, 128, 128), "pink": (255, 175, 175), "darkgray": (64, 64, 64), "lightgray": (192, 192, 192)}


def parse_color(value) -> tuple[int, int, int]:
    """'#rrggbb', '#rgb', 'r,g,b', a colour name or a 3-sequence → (r, g, b)."""
    if isinstance(value, (tuple, list, np.ndarray)):
        if len(value) < 3:
            raise ValueError(f"Bad colour {value!r}")
        return tuple(int(np.clip(float(v), 0, 255)) for v in value[:3])  # type: ignore[return-value]
    s = str(value or "").strip()
    if not s:
        raise ValueError("Empty colour")
    if s.lower() in _NAMED_COLORS:
        return _NAMED_COLORS[s.lower()]
    h = s.lstrip("#")
    if "," in s or " " in s:
        parts = [p for p in s.replace(",", " ").split(" ") if p]
        if len(parts) >= 3:
            return tuple(int(np.clip(float(p), 0, 255)) for p in parts[:3])  # type: ignore[return-value]
    if len(h) == 3 and all(c in "0123456789abcdefABCDEF" for c in h):
        h = "".join(c * 2 for c in h)
    if len(h) in (6, 8) and all(c in "0123456789abcdefABCDEF" for c in h):
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    raise ValueError(f"Bad colour {value!r} (use #rrggbb, r,g,b or a colour name)")
