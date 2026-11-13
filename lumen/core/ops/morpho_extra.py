"""Plugins ▸ MorphoLibJ, part 2 — Labels To RGB, Assign Measure to Label, Microstructure Analysis, Analyze Regions
3D, the Label Edition operations, gray-level morphological filters (2-D / 3-D), regional / extended / imposed minima
and maxima, chamfer and geodesic distance maps, marker-controlled watershed, morphological reconstruction, kill
borders, gray-level fill holes, area opening, directional filtering and binary / label overlays.

Geometry (MorphoLibJ's ``IntrinsicVolumes2D`` / ``IntrinsicVolumes3D``): perimeter and Euler number of a 2-D set
come from the histogram of its 2 × 2 pixel configurations and a look-up table (Crofton formula with 2 or 4
directions; Ohser's Euler coefficients for 4- / 8-connectivity); the 3-D surface area is the Crofton estimate from
the intercept counts along 3 or 13 directions (direction weights = solid angles of the spherical Voronoi cells of
the directions), the mean breadth the Crofton integral of the Euler numbers of the planar sections perpendicular
to those directions (rectangular section lattices for the axes and face diagonals, triangular lattices for the
space diagonals) and the 3-D Euler number the 2 × 2 × 2 configuration estimate (6- / 26-connectivity).  Whole-object
measures pad the object by one voxel so its outer boundary counts; *densities* (Microstructure Analysis) count only
the configurations fully inside the image (edge correction) and divide by the sampling window, exactly as
MorphoLibJ does.

Label-editing commands take ``label = 0`` / an empty label list as "the label picked with the value picker".
"""
from __future__ import annotations

import inspect
import math
import warnings
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from skimage import filters as skfilt, measure as skmeas, morphology as skm, segmentation as skseg

from .. import roi as roimod
from ..image import LumenImage
from ..luts import LUT_NAMES, apply_lut, get_lut, parse_color
from ..registry import Param, OpResult, op
from ._util import back_to, binary_image, is_binary_plane
from .morpho_ops import (CAT, MODES_2D_3D, _as_labels, _depth, _labels_image, _parse_label_list, _picked_label, _planes,
                         _position_cols)

LABEL_LUTS = ["Glasbey", "Glasbey on dark", "Glasbey inverted", "Golden angle", "Random"]
LABEL_LUT_CHOICES = LABEL_LUTS + [n for n in LUT_NAMES if n not in LABEL_LUTS]
MICRO_FEATURES = ["Area / volume density", "Perimeter / surface density", "Mean breadth density", "Euler density"]
MICRO_REGIONS = ["Whole image", "Each label"]
REGION3D_FEATURES = ["Volume", "Surface area", "Sphericity", "Euler number", "Bounding box", "Centroid", "Equivalent ellipsoid",
                     "Ellipsoid elongations", "Inscribed sphere", "Mean breadth"]
MORPHO_OPERATIONS = ["Erosion", "Dilation", "Opening", "Closing", "White Top Hat", "Black Top Hat", "Gradient", "Laplacian",
                     "Internal Gradient", "External Gradient"]
ELEMENTS_2D = ["Disk", "Square", "Diamond", "Octagon", "Line"]
ELEMENTS_3D = ["Ball", "Cube", "Cross"]
MINMAX_REGIONAL = ["Regional Minima", "Regional Maxima"]
MINMAX_EXTENDED = ["Extended Minima", "Extended Maxima"]
MINMAX_IMPOSE = ["Impose Minima", "Impose Maxima"]
CHAMFER_NAMES = ["Chessboard", "City-Block", "Quasi-Euclidean", "Borgefors", "Chessknight"]
#: (2-D weights, 3-D weights) per chamfer mask name: orthogonal / diagonal / (knight | space diagonal)
CHAMFER_WEIGHTS: dict[str, tuple[tuple[float, ...], tuple[float, ...] | None]] = {
    "Chessboard": ((1.0, 1.0), (1.0, 1.0, 1.0)), "City-Block": ((1.0, 2.0), (1.0, 2.0, 3.0)),
    "Quasi-Euclidean": ((1.0, math.sqrt(2.0)), (1.0, math.sqrt(2.0), math.sqrt(3.0))),
    "Borgefors": ((3.0, 4.0), (3.0, 4.0, 5.0)), "Chessknight": ((5.0, 7.0, 11.0), None)}
WATERSHED_MARKERS = ["Marker image", "Point ROIs", "Area ROIs"]
WATERSHED_INPUTS = ["Image as is", "Sobel gradient", "Morphological gradient"]
RECONSTRUCTIONS = ["By Dilation", "By Erosion"]
AREA_OPERATIONS = ["Area Opening", "Area Closing"]
DIRECTIONAL_TYPES = ["Max", "Min"]
DIRECTIONAL_OPERATIONS = ["Opening", "Closing", "Erosion", "Dilation", "Mean", "Median"]

# Ohser's Euler coefficients per 2 × 2 configuration (bit order: top-left 8, top-right 2, bottom-left 4, bottom-right 1,
# as scikit-image) and the Crofton perimeter LUTs for 2 / 4 directions (unit pixels).
EULER_2D_4 = np.array([0, 1, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0], float)
EULER_2D_8 = np.array([0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, -1, 0], float)
_P2 = np.pi / 2
_P4 = np.pi / 4
_S2 = math.sqrt(2.0)
PERIMETER_2D_2 = np.array([0, _P2, 0, 0, 0, _P2, 0, 0, _P2, np.pi, 0, 0, _P2, np.pi, 0, 0], float)
PERIMETER_2D_4 = np.array([0, _P4 * (1 + 1 / _S2), _P4 / _S2, _P2 / _S2, 0, _P4 * (1 + 1 / _S2), 0, _P4 / _S2,
                           _P4, _P2, _P4 / _S2, _P4 / _S2, _P4, _P2, 0, 0], float)
# Euler coefficients per 2 × 2 × 2 configuration, 26-connectivity (reverse the table for 6-connectivity), × 1/8
EULER_3D_26 = np.array([
    0, 1, 1, 0, 1, 0, -2, -1, 1, -2, 0, -1, 0, -1, -1, 0, 1, 0, -2, -1, -2, -1, -1, -2, -6, -3, -3, -2, -3, -2, 0, -1,
    1, -2, 0, -1, -6, -3, -3, -2, -2, -1, -1, -2, -3, 0, -2, -1, 0, -1, -1, 0, -3, -2, 0, -1, -3, 0, -2, -1, 0, 1, 1, 0,
    1, -2, -6, -3, 0, -1, -3, -2, -2, -1, -3, 0, -1, -2, -2, -1, 0, -1, -3, -2, -1, 0, 0, -1, -3, 0, 0, 1, -2, -1, 1, 0,
    -2, -1, -3, 0, -3, 0, 0, 1, -1, 4, 0, 3, 0, 3, 1, 2, -1, -2, -2, -1, -2, -1, 1, 0, 0, 3, 1, 2, 1, 2, 2, 1,
    1, -6, -2, -3, -2, -3, -1, 0, 0, -3, -1, -2, -1, -2, -2, -1, -2, -3, -1, 0, -1, 0, 4, 3, -3, 0, 0, 1, 0, 1, 3, 2,
    0, -3, -1, -2, -3, 0, 0, 1, -1, 0, 0, -1, -2, 1, -1, 0, -1, -2, -2, -1, 0, 1, 3, 2, -2, 1, -1, 0, 1, 2, 2, 1,
    0, -3, -3, 0, -1, -2, 0, 1, -1, 0, -2, 1, 0, -1, -1, 0, -1, -2, 0, 1, -2, -1, 3, 2, -2, 1, 1, 2, -1, 0, 2, 1,
    -1, 0, -2, 1, -2, 1, 1, 2, -2, 3, -1, 2, -1, 2, 0, 1, 0, -1, -1, 0, -1, 0, 2, 1, -1, 2, 0, 1, 0, 1, 1, 0], float)
try:  # prefer scikit-image's own tables when they are available (identical values)
    from skimage.measure import _regionprops_utils as _rpu
    EULER_2D_4 = np.asarray(_rpu.EULER_COEFS2D_4, float)
    EULER_2D_8 = np.asarray(_rpu.EULER_COEFS2D_8, float)
    EULER_3D_26 = np.asarray(_rpu.EULER_COEFS3D_26, float)
except Exception:  # pragma: no cover
    pass

#: the 13 directions of the 3-D Crofton estimators as (dz, dy, dx) lattice offsets: axes, face diagonals, space diagonals
DIRS13: list[tuple[int, int, int]] = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1), (1, 1, 0), (1, -1, 0),
                                      (1, 1, 1), (1, 1, -1), (1, -1, 1), (1, -1, -1)]
#: for every direction (plane normal): the two lattice vectors spanning the planar section lattice and whether the
#: lattice is triangular (space diagonals: u, w and u + w are its three edge directions) or rectangular
SECTION_LATTICES: dict[tuple[int, int, int], tuple[tuple[int, int, int], tuple[int, int, int], bool]] = {
    (0, 0, 1): ((1, 0, 0), (0, 1, 0), False), (0, 1, 0): ((1, 0, 0), (0, 0, 1), False), (1, 0, 0): ((0, 1, 0), (0, 0, 1), False),
    (0, 1, 1): ((1, 0, 0), (0, 1, -1), False), (0, 1, -1): ((1, 0, 0), (0, 1, 1), False),
    (1, 0, 1): ((0, 1, 0), (1, 0, -1), False), (1, 0, -1): ((0, 1, 0), (1, 0, 1), False),
    (1, 1, 0): ((0, 0, 1), (1, -1, 0), False), (1, -1, 0): ((0, 0, 1), (1, 1, 0), False),
    (1, 1, 1): ((0, 1, -1), (1, -1, 0), True), (1, 1, -1): ((0, 1, 1), (1, -1, 0), True),
    (1, -1, 1): ((0, 1, 1), (1, 0, -1), True), (1, -1, -1): ((0, 1, -1), (1, 0, 1), True)}


# =========================================================================== intrinsic volumes
def _config_hist_2d(b: np.ndarray, pad: bool) -> np.ndarray:
    """Histogram (16) of the 2 × 2 configurations of a boolean plane.

    ``pad``: the windows overlapping the outside count too (whole-object measures); otherwise only the windows
    fully inside the image are counted (edge-corrected densities)."""
    b = np.pad(b, 1) if pad else np.asarray(b, bool)
    b = b.astype(np.uint8)
    code = 8 * b[:-1, :-1] + 2 * b[:-1, 1:] + 4 * b[1:, :-1] + b[1:, 1:]
    return np.bincount(code.ravel(), minlength=16).astype(float)


def _config_hist_3d(b: np.ndarray, pad: bool) -> np.ndarray:
    """Histogram (256) of the 2 × 2 × 2 configurations of a boolean volume (scikit-image's bit order)."""
    b = np.pad(b, 1) if pad else np.asarray(b, bool)
    b = b.astype(np.uint8)
    code = (128 * b[:-1, :-1, :-1] + 32 * b[:-1, :-1, 1:] + 64 * b[:-1, 1:, :-1] + 16 * b[:-1, 1:, 1:]
            + 8 * b[1:, :-1, :-1] + 2 * b[1:, :-1, 1:] + 4 * b[1:, 1:, :-1] + b[1:, 1:, 1:])
    return np.bincount(code.ravel(), minlength=256).astype(float)


def _perimeter_2d(b: np.ndarray, directions: int, pw: float, ph: float, pad: bool) -> float:
    """Crofton perimeter (2 or 4 directions) of a boolean plane, in calibrated units (isotropic approximation √(pw·ph))."""
    lut = PERIMETER_2D_2 if int(directions) == 2 else PERIMETER_2D_4
    return float(lut @ _config_hist_2d(b, pad)) * math.sqrt(pw * ph)


def _euler_2d(b: np.ndarray, conn: int, pad: bool) -> float:
    lut = EULER_2D_4 if int(conn) == 4 else EULER_2D_8
    return float(lut @ _config_hist_2d(b, pad))


def _euler_3d(b: np.ndarray, conn: int, pad: bool) -> float:
    lut = EULER_3D_26 if int(conn) == 26 else EULER_3D_26[::-1]
    return float(lut @ _config_hist_3d(b, pad)) / 8.0


@lru_cache(maxsize=32)
def _direction_weights(spacing: tuple[float, float, float], n_dirs: int) -> tuple[float, ...]:
    """Weights of the 3 / 13 directions (sum 1): the solid angle of each direction's spherical Voronoi cell, so that
    anisotropic voxels get their share of the sphere (MorphoLibJ's ``computeDirectionWeights3d13``)."""
    if int(n_dirs) == 3:
        return (1 / 3, 1 / 3, 1 / 3)
    from scipy.spatial import SphericalVoronoi
    dz, dy, dx = spacing
    vecs = np.array([(oz * dz, oy * dy, ox * dx) for oz, oy, ox in DIRS13], float)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    pts = np.concatenate([vecs, -vecs])
    sv = SphericalVoronoi(pts, radius=1.0, center=np.zeros(3))
    sv.sort_vertices_of_regions()
    areas = np.asarray(sv.calculate_areas())
    w = (areas[:13] + areas[13:]) / (4.0 * np.pi)
    return tuple(float(v) for v in w / w.sum())


def _views(b: np.ndarray, offsets: list[tuple[int, int, int]]) -> list[np.ndarray]:
    """``b[p + o]`` for every offset, over the points ``p`` for which all ``p + o`` lie inside ``b`` (same shapes)."""
    nd = b.ndim
    lo = [max(0, -min(o[a] for o in offsets)) for a in range(nd)]
    hi = [b.shape[a] - max(0, max(o[a] for o in offsets)) for a in range(nd)]
    return [b[tuple(slice(lo[a] + o[a], hi[a] + o[a]) for a in range(nd))] for o in offsets]


def _transitions(b: np.ndarray, off: tuple[int, int, int]) -> int:
    a, c = _views(b, [(0, 0, 0), off])
    return int(np.count_nonzero(a != c))


def _crofton_surface(b: np.ndarray, spacing: tuple[float, float, float], n_dirs: int, pad: bool) -> float:
    """Crofton surface area of a boolean (Z, Y, X) volume: 2 · Σ_d w_d · (intercepts along d) · (area per test line of d)."""
    b = np.pad(b, 1) if pad else np.asarray(b, bool)
    dz, dy, dx = spacing
    vol = dz * dy * dx
    dirs = DIRS13[:3] if int(n_dirs) == 3 else DIRS13
    weights = _direction_weights(tuple(float(v) for v in spacing), int(n_dirs))
    total = 0.0
    for (oz, oy, ox), wd in zip(dirs, weights):
        length = math.sqrt((oz * dz) ** 2 + (oy * dy) ** 2 + (ox * dx) ** 2)
        total += wd * _transitions(b, (oz, oy, ox)) * vol / length
    return 2.0 * total


def _section_euler_sum(b: np.ndarray, u: tuple[int, int, int], w: tuple[int, int, int], triangular: bool, conn2d: int) -> float:
    """Σ over the parallel planar sections spanned by (u, w) of their Euler numbers (every lattice cell owns one vertex,
    its edges and faces, so summing per cell over the whole volume sums the Euler numbers of all sections)."""
    uw = (u[0] + w[0], u[1] + w[1], u[2] + w[2])
    b00, b01, b10, b11 = [v.astype(np.int64) for v in _views(b, [(0, 0, 0), w, u, uw])]
    if triangular:  # V − E + F with edges along u, w, u + w and the two triangles of every parallelogram cell
        return float((b11 - b01 * b11 - b10 * b11 - b00 * b11 + b00 * b10 * b11 + b00 * b01 * b11).sum())
    code = 8 * b00 + 2 * b01 + 4 * b10 + b11
    hist = np.bincount(code.ravel(), minlength=16).astype(float)
    return float((EULER_2D_4 if int(conn2d) == 4 else EULER_2D_8) @ hist)


def _mean_breadth(b: np.ndarray, spacing: tuple[float, float, float], n_dirs: int, conn2d: int, pad: bool) -> float:
    """Crofton mean breadth of a boolean volume: Σ_d w_d · Σ_planes⊥d χ(section) · (plane spacing along d)."""
    b = np.pad(b, 1) if pad else np.asarray(b, bool)
    dz, dy, dx = spacing
    vol = dz * dy * dx
    normals = DIRS13[:3] if int(n_dirs) == 3 else DIRS13
    weights = _direction_weights(tuple(float(v) for v in spacing), int(n_dirs))
    total = 0.0
    for n, wd in zip(normals, weights):
        u, w, tri = SECTION_LATTICES[n]
        ucal = np.array([u[0] * dz, u[1] * dy, u[2] * dx])
        wcal = np.array([w[0] * dz, w[1] * dy, w[2] * dx])
        cell_area = float(np.linalg.norm(np.cross(ucal, wcal)))
        total += wd * _section_euler_sum(b, u, w, tri, conn2d) * vol / cell_area
    return total


# =============================================================================== helpers
def _other_plane(other: LumenImage, img: LumenImage, t: int, z: int) -> np.ndarray:
    """The plane of ``other`` matching plane (t, z) of ``img`` (its current channel)."""
    if (other.height, other.width) != (img.height, img.width):
        raise ValueError(f"{other.name} must have the same width and height as {img.name}")
    return other.data[min(t, other.nt - 1), min(z, other.nz - 1), min(other.display.c, other.nc - 1)]


def _other_volume(other: LumenImage, img: LumenImage, t: int) -> np.ndarray:
    if (other.nz, other.height, other.width) != (img.nz, img.height, img.width):
        raise ValueError(f"{other.name} must have the same width, height and number of slices as {img.name}")
    return other.data[min(t, other.nt - 1), :, min(other.display.c, other.nc - 1)]


def _spacing3(img: LumenImage) -> tuple[float, float, float]:
    cal = img.calibration
    return (_depth(img), float(cal.pixel_height), float(cal.pixel_width))


def _unit(img: LumenImage) -> str:
    cal = img.calibration
    return cal.unit if cal.scaled else "px"


def _reference_rgb(img: LumenImage, t: int, z: int) -> np.ndarray:
    """(H, W, 3) uint8 rendering of plane (t, z): the RGB data, or the current channel through its LUT / display range."""
    if img.rgb:
        return np.ascontiguousarray(np.moveaxis(img.data[t, z], 0, 2))
    c = img.display.c
    ch = img.display.channels[min(c, len(img.display.channels) - 1)]
    return np.asarray(apply_lut(img.data[t, z, c], ch.lut, float(ch.min), float(ch.max), bool(ch.inverted), float(ch.gamma)), np.uint8)


def _rgb_image(out: np.ndarray, name: str, like: LumenImage) -> LumenImage:
    new = LumenImage(out, name=name, rgb=True)
    new.calibration = type(like.calibration)(**like.calibration.to_dict())
    return new


def _structure(ndim: int, connectivity: str) -> np.ndarray:
    """Binary structure: minimal (4 / 6) for '4', full (8 / 26) for '8'."""
    return ndi.generate_binary_structure(ndim, 1 if str(connectivity) == "4" else ndim)


def _conn_index(ndim: int, connectivity: str) -> int:
    return 1 if str(connectivity) == "4" else ndim


def _reconstruct(seed: np.ndarray, mask: np.ndarray, method: str, fp: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # skimage's reconstruction sets arr.shape (numpy ≥ 2.5 deprecation)
        return skm.reconstruction(seed, mask, method=method, footprint=fp)


def _current_array(img: LumenImage, three_d: bool) -> np.ndarray:
    d = img.display
    return img.data[d.t, :, d.c] if three_d else img.data[d.t, d.z, d.c]


def _result_shape(img: LumenImage, three_d: bool) -> tuple[int, int, int, int, int]:
    return (1, img.nz if three_d else 1, 1, img.height, img.width)


def _label_colors(n: int, lut: str, shuffle: bool, seed: int) -> np.ndarray:
    """(n, 3) uint8 colours for label ranks 1..n: categorical LUTs cycle through their 255 colours, gradient LUTs are
    spread over the labels, 'Golden angle' rotates the hue by the golden angle (MorphoLibJ's ColorMaps)."""
    if n <= 0:
        return np.zeros((0, 3), np.uint8)
    if lut == "Golden angle":
        from skimage.color import hsv2rgb
        hues = ((np.arange(n) * 137.50776405) % 360.0) / 360.0
        hsv = np.stack([hues, np.full(n, 0.85), np.full(n, 1.0)], axis=1).reshape(n, 1, 3)
        cols = (hsv2rgb(hsv).reshape(n, 3) * 255 + 0.5).astype(np.uint8)
    else:
        table = np.asarray(get_lut(lut), np.uint8)
        if lut.startswith("Glasbey") or lut in ("Random", "16 colors"):
            idx = (np.arange(n) % 255) + 1
        else:
            idx = np.rint(np.arange(n) * 255.0 / max(n - 1, 1)).astype(int) if n > 1 else np.array([255])
        cols = table[np.clip(idx, 0, 255)]
    if shuffle:
        cols = cols[np.random.default_rng(int(seed)).permutation(n)]
    return cols


def _label_ranks(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(unique labels > 0, rank image: 0 for background, 1..n for the labels in increasing value order)."""
    uniq = np.unique(lab)
    uniq = uniq[uniq > 0]
    ranks = np.searchsorted(uniq, lab) + 1
    ranks[lab <= 0] = 0
    ranks[~np.isin(lab, uniq)] = 0
    return uniq, ranks


def _point_labels(ctx, lab: np.ndarray) -> list[int]:
    """Labels under the points of the active point selection (or the manager's point ROIs on the current plane)."""
    from .segmentation import _marker_points
    pts = _marker_points(ctx, ctx.img)
    H, W = lab.shape
    return sorted({int(lab[y, x]) for x, y in pts if 0 <= y < H and 0 <= x < W and lab[y, x] > 0})


# ============================================================================= Labels To RGB
@op("morpho.labels_to_rgb", "Labels To RGB", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("lut", "choice", "Glasbey", choices=LABEL_LUT_CHOICES, help="Glasbey / Random cycle through 255 distinct colours, Golden angle rotates the hue by 137.5°, any other LUT is spread over the labels"),
            Param("background", "string", "black", help="Background colour (name, #rrggbb or r,g,b)"),
            Param("shuffle", "bool", False, help="Shuffle the colour assignment (neighbouring label values get unrelated colours)"),
            Param("seed", "int", 1, min=0, advanced=True, help="Seed of the shuffle"),
            Param("all_planes", "bool", False, label="Process whole stack", help="Convert every plane (3-D labels keep one colour per label across slices)")],
    description="MorphoLibJ's Labels To RGB: a new RGB image with one colour per label (Glasbey, Glasbey on dark, Golden angle, Random or any LUT), a chosen background colour and optional colour shuffling.",
    keywords=["morpholibj", "label", "rgb", "colour", "glasbey", "golden angle", "render"])
def labels_to_rgb(ctx, lut="Glasbey", background="black", shuffle=False, seed=1, all_planes=False):
    img = ctx.img
    try:
        bg = parse_color(background)
    except ValueError as e:
        raise ValueError(f"Background colour: {e}") from e
    planes = _planes(img, all_planes)
    labs = {p: _as_labels(img.data[p]) for p in planes}
    uniq = np.unique(np.concatenate([np.unique(v) for v in labs.values()]))
    uniq = uniq[uniq > 0]
    n = int(len(uniq))
    table = np.concatenate([np.asarray([bg], np.uint8), _label_colors(n, str(lut), bool(shuffle), int(seed))], axis=0)
    shape = (img.nt, img.nz, 3, img.height, img.width) if all_planes else (1, 1, 3, img.height, img.width)
    out = np.zeros(shape, np.uint8)
    for i, (t, z, c) in enumerate(planes):
        ctx.check_cancel()
        lab = labs[(t, z, c)]
        ranks = np.searchsorted(uniq, lab) + 1
        ranks[(lab <= 0) | ~np.isin(lab, uniq)] = 0
        rgb = table[ranks]
        out[(t, z) if all_planes else (0, 0)] = np.moveaxis(rgb, 2, 0)
        if len(planes) > 1:
            ctx.progress((i + 1) / len(planes), f"Plane {i + 1}/{len(planes)}")
    msg = f"{n} label{'s' if n != 1 else ''} coloured with {lut}" + (" (shuffled)" if shuffle else "")
    return OpResult(new_images=[_rgb_image(out, f"{img.name}-rgb", img)], message=msg, log=msg)


# ===================================================================== Assign Measure to Label
def _table_rows(ctx, table: str) -> list[dict[str, Any]]:
    name = str(table or "Results").strip()
    if name.lower() in ("", "results", "results table"):
        return list(ctx.session.results.rows)
    tab = ctx.session.tables.get(name)
    if tab is None:
        low = {k.lower(): k for k in ctx.session.tables}
        if name.lower() in low:
            tab = ctx.session.tables[low[name.lower()]]
    if tab is None:
        raise ValueError(f"No table {name!r} (use 'Results' or one of: {', '.join(ctx.session.tables) or 'none'})")
    cols = list(tab.get("columns", []))
    return [dict(zip(cols, r)) for r in tab.get("rows", [])]


def _row_label(row: dict[str, Any], label_column: str) -> int | None:
    """The label value a row refers to: an explicit column, else a numeric 'Label' / 'ID' / 'Label value' column or a
    'Label' string that is a plain number or ends in ':<k>' ('image:12' rows of Analyze Regions / Analyze Particles).

    In automatic mode a string only counts in those two forms: 'labs:t2' (Measure Labels 3D on a time series) and
    'cells2' (an image name ending in digits) must fall through to the numeric 'ID' column instead of being read as
    the label 2. An explicitly named column may use any trailing digits ('Spot 12')."""
    import re
    cands = [label_column] if label_column else ["Label", "ID", "Label value"]
    pat = r"(\d+)\s*$" if label_column else r"(?:^|:)\s*(\d+)\s*$"
    for col in cands:
        if col not in row or row[col] is None:
            continue
        v = row[col]
        if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
            return int(round(float(v)))
        m = re.search(pat, str(v))
        if m:
            return int(m.group(1))
    return None


@op("morpho.assign_measure_to_label", "Assign Measure to Label", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("column", "string", "Area", help="Column of the table whose value every label receives"),
            Param("table", "string", "Results", help="'Results' or the name of a table (Labels 3D, Region adjacency graph …)"),
            Param("label_column", "string", "", help="Column holding the label value; empty = automatic (numeric Label / ID / Label value columns, 'image:12' or plain-number Label entries)"),
            Param("missing", "choice", "0", choices=["0", "NaN"], help="Value of labels without a row (and of the background)"),
            Param("all_planes", "bool", False, label="Process whole stack")],
    description="MorphoLibJ's Assign Measure to Label: a new 32-bit image where every label carries the value of a Results (or table) column measured for it – a parametric map of the measurements.",
    keywords=["morpholibj", "label", "measure", "parametric", "map", "results", "colour by"])
def assign_measure_to_label(ctx, column="Area", table="Results", label_column="", missing="0", all_planes=False):
    img = ctx.img
    rows = _table_rows(ctx, table)
    col = str(column).strip()
    if not rows:
        raise ValueError(f"The table {table!r} is empty")
    if not any(col in r for r in rows):
        low = {k.lower(): k for r in rows for k in r}
        if col.lower() in low:
            col = low[col.lower()]
        else:
            raise ValueError(f"No column {column!r} in {table} (columns: {', '.join(dict.fromkeys(k for r in rows for k in r))})")
    values: dict[int, float] = {}
    for r in rows:
        k = _row_label(r, str(label_column).strip())
        v = r.get(col)
        if k is None or v is None:
            continue
        try:
            values[k] = float(v)
        except (TypeError, ValueError):
            continue
    if not values:
        raise ValueError("No row of the table could be matched to a label (set the label column)")
    fill = float("nan") if missing == "NaN" else 0.0
    planes = _planes(img, all_planes)
    shape = img.shape if all_planes else (1, 1, 1, img.height, img.width)
    out = np.full(shape, fill, np.float32)
    matched: set[int] = set()
    for (t, z, c) in planes:
        ctx.check_cancel()
        lab = _as_labels(img.data[t, z, c])
        uniq = np.unique(lab)
        uniq = uniq[uniq > 0]
        lut = np.array([values.get(int(k), fill) for k in uniq], np.float32)
        matched |= {int(k) for k in uniq if int(k) in values}
        idx = np.searchsorted(uniq, lab)
        idx[idx >= len(uniq)] = 0
        plane = np.where((lab > 0) & (len(uniq) > 0), lut[idx] if len(uniq) else fill, fill).astype(np.float32)
        out[(t, z, c) if all_planes else (0, 0, 0)] = plane
    new = LumenImage(out, name=f"{img.name}-{col}")
    new.calibration = type(img.calibration)(**img.calibration.to_dict())
    ch = new.display.channels[0]
    ch.lut = "Fire"
    finite = out[np.isfinite(out)]
    if finite.size:
        ch.min, ch.max = float(finite.min()), float(finite.max()) if finite.max() > finite.min() else float(finite.min()) + 1.0
    msg = f"{col} of {table} assigned to {len(matched)} label{'s' if len(matched) != 1 else ''}"
    return OpResult(new_images=[new], message=msg, log=msg)


# ===================================================================== Microstructure Analysis
@op("morpho.microstructure_analysis", "Microstructure Analysis", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D),
            Param("regions", "choice", "Whole image", choices=MICRO_REGIONS, help="Whole image: the foreground (non-zero pixels) as one phase; Each label: one row per label"),
            Param("features", "multichoice", MICRO_FEATURES, choices=MICRO_FEATURES, help="2D: area, perimeter and Euler densities; 3D: volume, surface, mean breadth and Euler densities"),
            Param("directions_2d", "choice", "4", choices=["2", "4"], label="Crofton directions (2D)"),
            Param("directions_3d", "choice", "13", choices=["3", "13"], label="Crofton directions (3D)", help="Surface area and mean breadth"),
            Param("connectivity", "choice", "8", choices=["4", "8"], help="Euler number: 4 / 8 neighbours in 2D, 6 / 26 in 3D (the mean breadth uses the matching 2D connectivity of the sections)")],
    description="MorphoLibJ's Microstructure Analysis: densities of the intrinsic volumes of a phase – area (volume) density, Crofton perimeter (surface) density, mean breadth density and Euler number density – with edge correction (only configurations fully inside the image count), per unit area / volume of the sampling window; for the foreground or for every label.",
    keywords=["morpholibj", "microstructure", "density", "stereology", "crofton", "euler", "mean breadth", "porosity", "specific surface"])
def microstructure_analysis(ctx, mode="2D (current plane)", regions="Whole image", features=None, directions_2d="4", directions_3d="13", connectivity="8"):
    img = ctx.img
    three_d = str(mode).startswith("3D")
    wanted = set(features if features is not None else MICRO_FEATURES)
    dz, dy, dx = _spacing3(img)
    unit = _unit(img)
    arr = _current_array(img, three_d)
    lab = _as_labels(arr) if regions == "Each label" else None
    if lab is not None:
        ids = [int(k) for k in np.unique(lab) if k > 0]
        phases = [(f"{img.name}:{k}", lab == k) for k in ids]
    else:
        fg = np.nan_to_num(arr.astype(np.float32)) != 0
        phases = [(img.name, fg)]
    if not phases or not any(p[1].any() for p in phases):
        raise ValueError("No foreground: the image needs non-zero pixels (a mask or a label image)")
    rows: list[dict[str, Any]] = []
    for i, (name, b) in enumerate(phases):
        ctx.check_cancel()
        row: dict[str, Any] = {"Label": name}
        if three_d:
            nz, ny, nx = b.shape
            total = nz * ny * nx * dz * dy * dx
            window = max(nz - 1, 1) * max(ny - 1, 1) * max(nx - 1, 1) * dz * dy * dx
            if "Area / volume density" in wanted:
                row["VolumeDensity"] = float(b.sum()) * dz * dy * dx / total
            if "Perimeter / surface density" in wanted:
                row["SurfaceDensity"] = _crofton_surface(b, (dz, dy, dx), int(directions_3d), pad=False) / window
            if "Mean breadth density" in wanted:
                row["MeanBreadthDensity"] = _mean_breadth(b, (dz, dy, dx), int(directions_3d), 4 if connectivity == "4" else 8, pad=False) / window
            if "Euler density" in wanted:
                row["EulerDensity"] = _euler_3d(b, 6 if connectivity == "4" else 26, pad=False) / window
        else:
            ny, nx = b.shape
            total = ny * nx * dy * dx
            window = max(ny - 1, 1) * max(nx - 1, 1) * dy * dx
            if "Area / volume density" in wanted:
                row["AreaDensity"] = float(b.sum()) * dy * dx / total
            if "Perimeter / surface density" in wanted:
                row["PerimeterDensity"] = _perimeter_2d(b, int(directions_2d), dx, dy, pad=False) / window
            if "Euler density" in wanted:
                row["EulerDensity"] = _euler_2d(b, int(connectivity), pad=False) / window
        rows.append(row)
        if len(phases) > 1:
            ctx.progress((i + 1) / len(phases), f"Phase {i + 1}/{len(phases)}")
    cols = [c for c in rows[0] if c != "Label"]
    table = {"name": "Microstructure", "columns": ["Label"] + cols, "rows": [[r["Label"]] + [r.get(c) for c in cols] for r in rows]}
    dims = "3D" if three_d else "2D"
    msg = f"{len(rows)} phase{'s' if len(rows) != 1 else ''} analysed ({dims}, densities per {unit}{'³' if three_d else '²'})"
    return OpResult(results=rows, table=table, message=msg, log=msg)


# ========================================================================= Analyze Regions 3D
def _equivalent_ellipsoid(zz: np.ndarray, yy: np.ndarray, xx: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    """Centre, radii (R1 ≥ R2 ≥ R3) and ZYX Euler angles (azimuth, elevation, roll; degrees) of the ellipsoid with the
    same second-order moments as the voxel set (MorphoLibJ's EquivalentEllipsoid, with the 1/12 voxel term)."""
    dz, dy, dx = spacing
    pts = np.stack([xx * dx, yy * dy, zz * dz], axis=1).astype(np.float64)
    centre = pts.mean(axis=0)
    d = pts - centre
    cov = d.T @ d / len(pts) + np.diag([dx * dx, dy * dy, dz * dz]) / 12.0
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals, R = np.maximum(evals[order], 0.0), evecs[:, order].copy()
    if R[0, 0] < 0:  # deterministic signs: first axis points to +x, right-handed frame
        R[:, 0] *= -1
        R[:, 1] *= -1
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    radii = np.sqrt(5.0 * evals)
    cos_theta = math.hypot(R[2, 1], R[2, 2])
    theta = math.atan2(-R[2, 0], cos_theta)
    if cos_theta > 1e-12:
        psi = math.atan2(R[2, 1], R[2, 2])
        phi = math.atan2(R[1, 0], R[0, 0])
    else:
        psi = math.atan2(-R[1, 2], R[1, 1])
        phi = 0.0
    return {"cx": centre[0] + 0.5 * dx, "cy": centre[1] + 0.5 * dy, "cz": centre[2] + 0.5 * dz,
            "r1": float(radii[0]), "r2": float(radii[1]), "r3": float(radii[2]),
            "azim": math.degrees(phi), "elev": math.degrees(theta), "roll": math.degrees(psi)}


@op("morpho.analyze_regions_3d", "Analyze Regions 3D", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("features", "multichoice", REGION3D_FEATURES, choices=REGION3D_FEATURES, help="Measurements per label"),
            Param("directions", "choice", "13", choices=["3", "13"], label="Crofton directions", help="Surface area and mean breadth"),
            Param("connectivity", "choice", "26", choices=["6", "26"], help="Euler number (the mean breadth sections use 4- / 8-connectivity accordingly)"),
            Param("all_frames", "bool", False, label="Process every frame")],
    description="MorphoLibJ's Analyze Regions 3D: per label of the current z-stack – volume, Crofton surface area (3 / 13 directions), sphericity, Euler number (6 / 26), bounding box, centroid, equivalent ellipsoid (centre, radii, azimuth / elevation / roll), ellipsoid elongations, largest inscribed sphere and mean breadth, in calibrated units.",
    keywords=["morpholibj", "3d", "label", "region", "volume", "surface", "sphericity", "ellipsoid", "inscribed sphere", "mean breadth", "euler"])
def analyze_regions_3d(ctx, features=None, directions="13", connectivity="26", all_frames=False):
    img = ctx.img
    d = img.display
    wanted = set(features if features is not None else REGION3D_FEATURES)
    spacing = _spacing3(img)
    dz, dy, dx = spacing
    vox = dz * dy * dx
    n_dirs = int(directions)
    conn3 = 6 if str(connectivity) == "6" else 26
    conn2 = 4 if conn3 == 6 else 8
    frames = list(range(img.nt)) if all_frames else [d.t]
    rows: list[dict[str, Any]] = []
    jobs: list[tuple[int, int, tuple[slice, ...], np.ndarray]] = []
    for t in frames:
        lab = _as_labels(img.data[t, :, d.c])
        objs = ndi.find_objects(lab)
        for i, sl in enumerate(objs):
            if sl is not None:
                jobs.append((t, i + 1, sl, lab))
    for j, (t, k, sl, lab) in enumerate(jobs):
        ctx.check_cancel()
        sub = lab[sl] == k
        zz, yy, xx = np.nonzero(sub)
        z0, y0, x0 = sl[0].start, sl[1].start, sl[2].start
        n = int(zz.size)
        volume = n * vox
        row: dict[str, Any] = {"Label": f"{img.name}:{k}"}
        if all_frames and img.nt > 1:
            row["Frame"] = t + 1
        if "Volume" in wanted:
            row["Volume"] = volume
        surface = None
        if "Surface area" in wanted or "Sphericity" in wanted:
            surface = _crofton_surface(sub, spacing, n_dirs, pad=True)
        if "Surface area" in wanted:
            row["SurfaceArea"] = surface
        if "Sphericity" in wanted:
            row["Sphericity"] = 36.0 * math.pi * volume * volume / (surface ** 3) if surface and surface > 0 else float("nan")
        if "Euler number" in wanted:
            row["EulerNumber"] = int(skmeas.euler_number(sub, connectivity=1 if conn3 == 6 else 3))
        if "Bounding box" in wanted:
            row.update({"Box.X.Min": x0 * dx, "Box.X.Max": sl[2].stop * dx, "Box.Y.Min": y0 * dy, "Box.Y.Max": sl[1].stop * dy,
                        "Box.Z.Min": z0 * dz, "Box.Z.Max": sl[0].stop * dz})
        if "Centroid" in wanted:
            row.update({"Centroid.X": (float(xx.mean()) + x0 + 0.5) * dx, "Centroid.Y": (float(yy.mean()) + y0 + 0.5) * dy,
                        "Centroid.Z": (float(zz.mean()) + z0 + 0.5) * dz})
        if "Equivalent ellipsoid" in wanted or "Ellipsoid elongations" in wanted:
            ell = _equivalent_ellipsoid(zz + z0, yy + y0, xx + x0, spacing)
            if "Equivalent ellipsoid" in wanted:
                row.update({"Elli.Center.X": ell["cx"], "Elli.Center.Y": ell["cy"], "Elli.Center.Z": ell["cz"], "Elli.R1": ell["r1"],
                            "Elli.R2": ell["r2"], "Elli.R3": ell["r3"], "Elli.Azim": ell["azim"], "Elli.Elev": ell["elev"], "Elli.Roll": ell["roll"]})
            if "Ellipsoid elongations" in wanted:
                row.update({"Elli.R1/R2": ell["r1"] / ell["r2"] if ell["r2"] > 0 else float("nan"),
                            "Elli.R1/R3": ell["r1"] / ell["r3"] if ell["r3"] > 0 else float("nan"),
                            "Elli.R2/R3": ell["r2"] / ell["r3"] if ell["r3"] > 0 else float("nan")})
        if "Inscribed sphere" in wanted:
            edt = ndi.distance_transform_edt(np.pad(sub, 1), sampling=spacing)[1:-1, 1:-1, 1:-1]
            iz, iy, ix = np.unravel_index(int(np.argmax(edt)), edt.shape)
            row.update({"InscrBall.Center.X": (ix + x0 + 0.5) * dx, "InscrBall.Center.Y": (iy + y0 + 0.5) * dy,
                        "InscrBall.Center.Z": (iz + z0 + 0.5) * dz, "InscrBall.Radius": float(edt[iz, iy, ix])})
        if "Mean breadth" in wanted:
            row["MeanBreadth"] = _mean_breadth(sub, spacing, n_dirs, conn2, pad=True)
        rows.append(row)
        if len(jobs) > 1:
            ctx.progress((j + 1) / len(jobs), f"Region {j + 1}/{len(jobs)}")
    unit = _unit(img)
    msg = f"{len(rows)} region{'s' if len(rows) != 1 else ''} measured in 3D (lengths in {unit})"
    return OpResult(results=rows if rows else None, message=msg, log=msg)


# ============================================================================== Label Edition
@op("morpho.select_label", "Select Label", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("label", "int", 0, min=0, help="Label value; 0 = the label picked with the value picker (I)"),
            Param("add_to_manager", "bool", False, help="Also add the selection to the ROI manager")],
    description="Make the outline of one label the active selection (Label Edition's selection by value / value picker).",
    keywords=["morpholibj", "label", "select", "selection", "picker", "label edition"])
def select_label(ctx, label=0, add_to_manager=False):
    img = ctx.img
    k = int(label) or _picked_label(ctx)
    lab = _as_labels(img.current_plane())
    m = lab == k
    if not m.any():
        raise ValueError(f"Label {k} is not on the current plane")
    rois = roimod.label_to_rois(m.astype(np.int32), simplify=True)
    roi = rois[0] if rois else roimod.from_mask(m)
    roi["name"] = f"L{k}"
    res = OpResult(roi=roi, message=f"Label {k}: {int(m.sum())} px")
    if add_to_manager:
        res.rois_add = [dict(roi)]
    return res


_TOOL_KW: dict[str, Any] = ({"tool": True, "tool_icon": "⌫", "tool_cursor": "crosshair"}
                            if "tool" in inspect.signature(op).parameters else {})


@op("morpho.remove_label_at", "Remove Label at Point", CAT, kind="image", preview=False, supports_rgb=False, roi_aware=False, scope_param=False,
    params=[Param("x", "int", 0, min=0, help="Image x of a pixel of the label (px)"), Param("y", "int", 0, min=0, help="Image y (px)"),
            Param("all_planes", "bool", False, label="Whole stack", help="Remove the label from every plane (3-D labels)")],
    description="Remove the label under a point (Label Edition's click-to-remove; as a tool, every click on the image removes the label under the cursor).",
    keywords=["morpholibj", "label", "remove", "delete", "click", "label edition", "tool"], undo_label="Remove Label", **_TOOL_KW)
def remove_label_at(ctx, x=0, y=0, all_planes=False, **tool):
    if tool.get("event") == "drag-end":
        return OpResult()
    img = ctx.img
    d = img.display
    xi, yi = int(x), int(y)
    if not (0 <= xi < img.width and 0 <= yi < img.height):
        raise ValueError(f"({xi}, {yi}) is outside the image")
    cur = img.data[d.t, d.z, d.c]
    lab = _as_labels(cur)
    k = int(lab[yi, xi])
    if k == 0:
        return OpResult(message="Background clicked – nothing removed")
    out = img.data.copy()
    planes = list(img.iter_planes()) if (all_planes and not is_binary_plane(cur)) else [(d.t, d.z, d.c)]
    for (t, z, c) in planes:
        src = img.data[t, z, c]
        lp = lab if (t, z, c) == (d.t, d.z, d.c) else _as_labels(src)
        out[t, z, c] = np.where(lp == k, 0, src)
    return OpResult(data=out, message=f"Label {k} removed")


@op("morpho.merge_labels", "Merge Labels", CAT, kind="plane", supports_rgb=False, roi_aware=False,
    params=[Param("labels", "string", "", help="Labels to merge, e.g. '3, 5, 8-9'; empty = the labels under the points of the active point selection (shift-click adds points) or of the manager's point ROIs"),
            Param("new_value", "int", 0, min=0, help="Value of the merged label; 0 = the smallest of the merged labels")],
    description="MorphoLibJ's Merge Labels: give several labels one common value (listed, or clicked with the point tool).",
    keywords=["morpholibj", "label", "merge", "join", "combine", "label edition"])
def merge_labels(plane, ctx, labels="", new_value=0):
    lab = _as_labels(plane)
    values = _parse_label_list(labels) if str(labels or "").strip() else _point_labels(ctx, lab)
    values = sorted({int(v) for v in values if int(v) > 0})
    if len(values) < 2:
        raise ValueError("Merge Labels needs at least two labels: list them, or click them with the point tool (shift-click adds points)")
    target = int(new_value) or values[0]
    return back_to(np.where(np.isin(lab, values), target, lab), plane)


@op("morpho.fill_label_holes", "Fill Label Holes", CAT, kind="plane", supports_rgb=False, roi_aware=False,
    description="Fill the background holes enclosed by labels with the label around them (a hole bordered by several labels is split between the nearest ones).",
    keywords=["morpholibj", "label", "fill holes", "label edition"])
def fill_label_holes(plane, ctx):
    lab = _as_labels(plane)
    fg = lab > 0
    holes = ndi.binary_fill_holes(fg) & ~fg
    if not holes.any():
        return plane
    grown = skseg.expand_labels(lab, distance=float(max(lab.shape)))
    return back_to(np.where(holes, grown, lab), plane)


@op("morpho.label_size_opening", "Label Size Opening", CAT, kind="plane", supports_rgb=False, roi_aware=False,
    params=[Param("min_size", "int", 20, min=1, max=1_000_000_000, unit="px", help="Labels with fewer pixels are removed"),
            Param("max_size", "int", 0, min=0, max=1_000_000_000, unit="px", help="Labels with more pixels are removed (0 = no limit)")],
    description="MorphoLibJ's Label Size Opening: remove the labels smaller than a number of pixels (and optionally the larger ones).",
    keywords=["morpholibj", "label", "size", "opening", "filter", "small", "label edition"])
def label_size_opening(plane, ctx, min_size=20, max_size=0):
    lab = _as_labels(plane)
    if lab.max() <= 0:
        return plane
    counts = np.bincount(lab.ravel())
    keep = counts >= int(min_size)
    if int(max_size) > 0:
        keep &= counts <= int(max_size)
    keep[0] = False
    return back_to(np.where(keep[lab], lab, 0), plane)


# ======================================================================= Morphological Filters
def _line_footprint(length: int, angle: float) -> np.ndarray:
    from skimage.draw import line as sk_line
    L = max(int(length), 1)
    r = L // 2
    fp = np.zeros((2 * r + 1, 2 * r + 1), bool)
    th = math.radians(float(angle))
    # 2r + 1 pixels along the line whatever its angle (MorphoLibJ's diagonal lines reach the corners of the box)
    scale = r / max(abs(math.cos(th)), abs(math.sin(th)), 1e-9)
    x0, y0 = r + scale * math.cos(th), r - scale * math.sin(th)  # ImageJ angles: counter-clockwise, y up
    x1, y1 = r - scale * math.cos(th), r + scale * math.sin(th)
    rr, cc = sk_line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
    fp[np.clip(rr, 0, 2 * r), np.clip(cc, 0, 2 * r)] = True
    fp[r, r] = True
    return fp


def _mlj_octagon(r: int) -> np.ndarray:
    """MorphoLibJ's octagon of size n = 2r + 1, as its OctagonStrel(size) builds it: ``diagSize = round((n + 2) / (2 + √2))``,
    ``squareSize = n - 2·(diagSize - 1)``; the element is that square dilated by the 45° and 135° lines of length diagSize
    (a diamond of radius diagSize - 1), each anchored at ``floor((length - 1) / 2)`` - so for an even diagSize the result
    sits one pixel right of the centre.  Read from MorphoLibJ 1.6.5's OctagonStrel and checked pixel for pixel against the
    real plugin for r = 1 … 40.  Returned centred for scipy, with a spare column on each side for the shift."""
    n = 2 * r + 1
    diag = int(math.floor((n + 2) / (2.0 + math.sqrt(2.0)) + 0.5))  # Java's Math.round
    d = diag - 1
    a = r - d  # half-side of the square
    shift = d % 2
    yy, xx = np.mgrid[-r:r + 1, -r - 1:r + 2]
    x = xx - shift
    return (np.maximum(0, np.abs(x) - a) + np.maximum(0, np.abs(yy) - a)) <= d


def _footprint_2d(element: str, radius: int, angle: float) -> np.ndarray:
    r = max(int(radius), 1)
    if element == "Square":
        return np.ones((2 * r + 1, 2 * r + 1), bool)
    if element == "Diamond":
        # MorphoLibJ's DiamondStrel (a shifted 3x3 cross plus the two diagonal lines): an exact diamond, one pixel left of
        # the centre for odd radii from 3 on - measured with the real plugin for r = 1 … 14
        shift = -1 if (r % 2 == 1 and r >= 3) else 0
        yy, xx = np.mgrid[-r:r + 1, -r - 1:r + 2]
        return (np.abs(xx - shift) + np.abs(yy)) <= r
    if element == "Octagon":
        return _mlj_octagon(r)
    if element == "Line":
        return _line_footprint(2 * r + 1, angle)
    # MorphoLibJ's DiskStrel uses ImageJ's RankFilters circle: dx² + dy² <= r² + 1 (21 pixels for r = 2, not the 13 of a
    # Euclidean r² disk) - measured by dilating a single pixel with the real MorphoLibJ plugin through the Java bridge
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx * xx + yy * yy) <= r * r + 1


def _footprint_3d(element: str, rx: int, ry: int, rz: int) -> np.ndarray:
    rx, ry, rz = max(int(rx), 0), max(int(ry), 0), max(int(rz), 0)
    zz, yy, xx = np.mgrid[-rz:rz + 1, -ry:ry + 1, -rx:rx + 1]
    if element == "Cube":
        return np.ones(zz.shape, bool)
    if element == "Cross":
        return ((yy == 0) & (xx == 0)) | ((zz == 0) & (xx == 0)) | ((zz == 0) & (yy == 0))
    return ((zz / max(rz, 1e-9)) ** 2 + (yy / max(ry, 1e-9)) ** 2 + (xx / max(rx, 1e-9)) ** 2) <= 1.0 + 1e-9


def _mid_value(dtype) -> float:
    if dtype == np.uint8:
        return 128.0
    if dtype == np.uint16:
        return 32768.0
    return 0.0


def _morpho_op(p: np.ndarray, operation: str, fp: np.ndarray, mid: float, reverse_second: bool = False) -> np.ndarray:
    """Gray-level morphology of a float32 array with a boolean footprint, following MorphoLibJ (checked against the real
    plugin through the Java bridge, tests/test_fiji_plugins.py):

    * single-step operations (erosion, dilation, the gradients, the Laplacian) ignore pixels outside the image;
    * opening, closing and the top hats run on the image extended by its edge values, then cropped - so the opening's
      erosion also has values beyond the border for its dilation to pick up;
    * erosion uses the same window as dilation (scipy reflects the footprint for erosion, so it is flipped back) - this
      only matters for MorphoLibJ's off-centre elements (octagon, diamond);
    * ``reverse_second``: the opening / closing applies the reversed element in its second step (MorphoLibJ's
      ``strel.reverse()``, which differs from the element itself only for the octagon);
    * the Laplacian is dilation + erosion - 2·image, offset by the mid gray level on 8- / 16-bit images."""
    p = p.astype(np.float32, copy=False)
    k = max(fp.shape)
    sl = tuple(slice(k, -k) for _ in range(p.ndim))

    def dil(a, f=fp, ignore_outside=True):
        if ignore_outside:  # pixels beyond the border never win a maximum
            return ndi.grey_dilation(a, footprint=f, mode="constant", cval=-np.inf)
        return ndi.grey_dilation(a, footprint=f, mode="nearest")

    def ero(a, f=fp, ignore_outside=True):
        if ignore_outside:  # ... nor a minimum
            return ndi.grey_erosion(a, footprint=np.flip(f), mode="constant", cval=np.inf)
        return ndi.grey_erosion(a, footprint=np.flip(f), mode="nearest")

    if operation in ("Opening", "Closing", "White Top Hat", "Black Top Hat"):
        q = np.pad(p, k, mode="edge")
        second = np.flip(fp) if reverse_second else fp
        if operation in ("Opening", "White Top Hat"):
            opened = dil(ero(q, ignore_outside=False), second, ignore_outside=False)
            out = opened if operation == "Opening" else q - opened
        else:
            closed = ero(dil(q, ignore_outside=False), second, ignore_outside=False)
            out = closed if operation == "Closing" else closed - q
        return out[sl]
    q = p
    if operation == "Erosion":
        return ero(q)
    if operation == "Dilation":
        return dil(q)
    if operation == "Gradient":
        return dil(q) - ero(q)
    if operation == "Laplacian":
        return dil(q) + ero(q) - 2.0 * q + np.float32(mid)
    if operation == "Internal Gradient":
        return q - ero(q)
    if operation == "External Gradient":
        return dil(q) - q
    raise ValueError(f"Unknown operation {operation!r}")


@op("morpho.filters", "Morphological Filters", CAT, supports_rgb=True,
    params=[Param("operation", "choice", "Dilation", choices=MORPHO_OPERATIONS, help="Laplacian = dilation + erosion − 2 × image, offset by the mid gray level on 8- / 16-bit images (as MorphoLibJ)"),
            Param("element", "choice", "Disk", choices=ELEMENTS_2D, label="Element shape"),
            Param("radius", "int", 2, min=1, max=200, unit="px"),
            Param("line_angle", "float", 0.0, min=0.0, max=180.0, step=15.0, help="Line element only: 0 = horizontal, 90 = vertical, 45 / 135 = diagonals")],
    description="MorphoLibJ's Morphological Filters (2D): erosion, dilation, opening, closing, white / black top hat, gradient, Laplacian, internal / external gradient with a disk, square, diamond, octagon or line structuring element.",
    keywords=["morpholibj", "morphology", "erosion", "dilation", "opening", "closing", "top hat", "gradient", "laplacian", "structuring element"])
def morphological_filters(plane, ctx, operation="Dilation", element="Disk", radius=2, line_angle=0.0):
    fp = _footprint_2d(element, radius, line_angle)
    return back_to(_morpho_op(plane, operation, fp, _mid_value(plane.dtype), reverse_second=(element == "Octagon")), plane)


@op("morpho.filters_3d", "Morphological Filters (3D)", CAT, kind="image", scope_param=False, preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("operation", "choice", "Dilation", choices=MORPHO_OPERATIONS),
            Param("element", "choice", "Ball", choices=ELEMENTS_3D, label="Element shape"),
            Param("radius_x", "int", 2, min=0, max=100, unit="px"), Param("radius_y", "int", 2, min=0, max=100, unit="px"),
            Param("radius_z", "int", 1, min=0, max=100, unit="px", help="Radius along z (slices)")],
    description="MorphoLibJ's Morphological Filters (3D): the gray-level operations with a ball, cube or 3-D cross structuring element of independent X / Y / Z radii, applied to every channel and frame of the stack.",
    keywords=["morpholibj", "3d", "morphology", "erosion", "dilation", "opening", "closing", "top hat", "gradient", "ball", "cube"])
def morphological_filters_3d(ctx, operation="Dilation", element="Ball", radius_x=2, radius_y=2, radius_z=1):
    img = ctx.img
    fp = _footprint_3d(element, radius_x, radius_y, radius_z)
    out = np.zeros(img.shape, np.float32)
    mid = _mid_value(img.data.dtype)
    jobs = [(t, c) for t in range(img.nt) for c in range(img.nc)]
    for i, (t, c) in enumerate(jobs):
        ctx.check_cancel()
        out[t, :, c] = _morpho_op(img.data[t, :, c].astype(np.float32), operation, fp, mid)
        if len(jobs) > 1:
            ctx.progress((i + 1) / len(jobs), f"Volume {i + 1}/{len(jobs)}")
    return OpResult(data=back_to(out, img.data), message=f"{operation} ({element} {radius_x}×{radius_y}×{radius_z})")


# ================================================================== regional / extended / impose
def _extrema(a: np.ndarray, minima: bool, conn: int, dynamic: float | None) -> np.ndarray:
    """Regional (dynamic None) or extended (dynamic h) minima / maxima of a 2-D or 3-D array → boolean."""
    a = np.nan_to_num(a.astype(np.float32))
    if dynamic is not None and dynamic > 0:
        fp = ndi.generate_binary_structure(a.ndim, conn)
        h = np.float32(dynamic)
        if minima:
            rec = _reconstruct(a + h, a, "erosion", fp)
            return skm.local_minima(rec, connectivity=conn).astype(bool)
        rec = _reconstruct(a - h, a, "dilation", fp)
        return skm.local_maxima(rec, connectivity=conn).astype(bool)
    return (skm.local_minima(a, connectivity=conn) if minima else skm.local_maxima(a, connectivity=conn)).astype(bool)


def _extrema_result(ctx, operation: str, three_d: bool, conn_choice: str, dynamic: float | None) -> OpResult:
    img = ctx.img
    a = _current_array(img, three_d)
    conn = _conn_index(a.ndim, conn_choice)
    m = _extrema(a, operation.endswith("Minima"), conn, dynamic)
    out = np.zeros(_result_shape(img, three_d), bool)
    if three_d:
        out[0, :, 0] = m
    else:
        out[0, 0, 0] = m
    n = int(skmeas.label(m, connectivity=conn).max()) if m.any() else 0
    msg = f"{n} {operation.lower()} region{'s' if n != 1 else ''}" + (f" (dynamic {dynamic:g})" if dynamic else "")
    return OpResult(new_images=[binary_image(out, f"{operation} of {img.name}", like=img)], message=msg, log=msg)


@op("morpho.regional_min_max", "Regional Min & Max", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("operation", "choice", "Regional Minima", choices=MINMAX_REGIONAL),
            Param("connectivity", "choice", "4", choices=["4", "8"], help="2D: 4 / 8 neighbours; 3D: 6 / 26"),
            Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D)],
    description="MorphoLibJ's Regional Min & Max (2D / 3D): binary image of the regional minima or maxima (connected plateaus with only higher, or only lower, neighbours).",
    keywords=["morpholibj", "regional minima", "regional maxima", "extrema", "plateau"])
def regional_min_max(ctx, operation="Regional Minima", connectivity="4", mode="2D (current plane)"):
    return _extrema_result(ctx, operation, str(mode).startswith("3D"), connectivity, None)


@op("morpho.extended_min_max", "Extended Min & Max", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("operation", "choice", "Extended Minima", choices=MINMAX_EXTENDED),
            Param("dynamic", "float", 10.0, min=0.0, step=1.0, help="Minimum depth (height) of a minimum (maximum) in gray levels"),
            Param("connectivity", "choice", "4", choices=["4", "8"], help="2D: 4 / 8 neighbours; 3D: 6 / 26"),
            Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D)],
    description="MorphoLibJ's Extended Min & Max (2D / 3D): binary image of the minima (maxima) whose dynamic is at least the given value – the regional extrema of the h-minima / h-maxima transform; the usual markers of a watershed.",
    keywords=["morpholibj", "extended minima", "extended maxima", "dynamic", "h-minima", "h-maxima", "markers"])
def extended_min_max(ctx, operation="Extended Minima", dynamic=10.0, connectivity="4", mode="2D (current plane)"):
    return _extrema_result(ctx, operation, str(mode).startswith("3D"), connectivity, float(dynamic))


def _impose(a: np.ndarray, markers: np.ndarray, minima: bool, conn: int) -> np.ndarray:
    """Impose the marker pixels as the only regional minima (maxima) of ``a`` (Soille's minima imposition).

    Integer images follow MorphoLibJ: the markers become 0 (minima) or the type maximum (maxima) and every other
    pixel is raised (lowered) by one gray level; float images use the range below (above) their values."""
    integer = np.issubdtype(a.dtype, np.integer)
    type_max = float(np.iinfo(a.dtype).max) if integer else None
    a = np.nan_to_num(a.astype(np.float64))
    fp = ndi.generate_binary_structure(a.ndim, conn)
    lo, hi = float(a.min()), float(a.max())
    delta = 1.0 if (integer or hi - lo >= 1.0) else max((hi - lo) * 1e-3, 1e-6)
    if minima:
        m_val = 0.0 if integer else lo - delta
        seed = np.where(markers, m_val, hi + delta)
        mask = np.where(markers, m_val, np.minimum(a + delta, type_max) if integer else a + delta)
        return _reconstruct(seed, mask, "erosion", fp)
    m_val = type_max if integer else hi + delta
    seed = np.where(markers, m_val, lo - delta)
    mask = np.where(markers, m_val, np.maximum(a - delta, 0.0) if integer else a - delta)
    return _reconstruct(seed, mask, "dilation", fp)


@op("morpho.impose_min_max", "Impose Min & Max", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("operation", "choice", "Impose Minima", choices=MINMAX_IMPOSE),
            Param("marker_image", "image", None, help="Binary image of the markers (non-zero pixels) – same size as the image"),
            Param("connectivity", "choice", "4", choices=["4", "8"], help="2D: 4 / 8 neighbours; 3D: 6 / 26"),
            Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D)],
    description="MorphoLibJ's Impose Min & Max (2D / 3D): a new image whose only regional minima (maxima) are the marker pixels – every other extremum is filled by morphological reconstruction, so a watershed of the result floods exactly from the markers.",
    keywords=["morpholibj", "impose minima", "impose maxima", "markers", "reconstruction", "watershed"])
def impose_min_max(ctx, operation="Impose Minima", marker_image=None, connectivity="4", mode="2D (current plane)"):
    img = ctx.img
    if not marker_image:
        raise ValueError("Choose the marker image")
    other = ctx.session.get_image(marker_image)
    d = img.display
    three_d = str(mode).startswith("3D")
    a = _current_array(img, three_d)
    markers = (_other_volume(other, img, d.t) if three_d else _other_plane(other, img, d.t, d.z)) != 0
    if not markers.any():
        raise ValueError(f"{other.name} has no marker pixels")
    res = _impose(a, markers, operation.endswith("Minima"), _conn_index(a.ndim, connectivity))
    out = np.zeros(_result_shape(img, three_d), np.float32)
    if three_d:
        out[0, :, 0] = res
    else:
        out[0, 0, 0] = res
    new = img.copy(name=f"{operation} of {img.name}", data=back_to(out, img.data))
    n = int(skmeas.label(markers, connectivity=_conn_index(a.ndim, connectivity)).max())
    msg = f"{operation}: {n} marker region{'s' if n != 1 else ''} imposed"
    return OpResult(new_images=[new], message=msg, log=msg)


# ======================================================================== chamfer distance maps
def _scan_2d(d: np.ndarray, a: float, b: float, knight: float | None) -> None:
    """One forward raster scan (top → bottom, left → right) of the chamfer distance ``d`` (in place, float64)."""
    H, W = d.shape
    idx = np.arange(W, dtype=np.float64) * a
    for y in range(H):
        row = d[y]
        if y >= 1:
            prev = d[y - 1]
            np.minimum(row, prev + a, out=row)
            if W > 1:
                np.minimum(row[1:], prev[:-1] + b, out=row[1:])
                np.minimum(row[:-1], prev[1:] + b, out=row[:-1])
            if knight is not None and W > 2:
                np.minimum(row[2:], prev[:-2] + knight, out=row[2:])
                np.minimum(row[:-2], prev[2:] + knight, out=row[:-2])
        if knight is not None and y >= 2 and W > 1:
            p2 = d[y - 2]
            np.minimum(row[1:], p2[:-1] + knight, out=row[1:])
            np.minimum(row[:-1], p2[1:] + knight, out=row[:-1])
        # in-row propagation: d[x] = min over x' ≤ x of d[x'] + a·(x − x')
        row[:] = np.minimum.accumulate(row - idx) + idx


def _chamfer_2d(fg: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    """Chamfer distance of every foreground pixel to the nearest background pixel (two raster scans; exact for the
    3 × 3 masks and Borgefors' 5 × 5 chessknight mask)."""
    a, b = float(weights[0]), float(weights[1])
    knight = float(weights[2]) if len(weights) > 2 else None
    d = np.where(fg, np.inf, 0.0)
    _scan_2d(d, a, b, knight)
    back = d[::-1, ::-1]
    _scan_2d(back, a, b, knight)
    d[~np.isfinite(d)] = 0.0  # a foreground image without background: everything stays "far", shown as 0
    return d


def _chamfer_3d(fg: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    a, b, c = float(weights[0]), float(weights[1]), float(weights[2])
    d = np.where(fg, np.inf, 0.0)

    def forward(vol: np.ndarray) -> None:
        for z in range(vol.shape[0]):
            plane = vol[z]
            if z >= 1:
                prev = vol[z - 1]
                np.minimum(plane, prev + a, out=plane)
                np.minimum(plane[1:, :], prev[:-1, :] + b, out=plane[1:, :])
                np.minimum(plane[:-1, :], prev[1:, :] + b, out=plane[:-1, :])
                np.minimum(plane[:, 1:], prev[:, :-1] + b, out=plane[:, 1:])
                np.minimum(plane[:, :-1], prev[:, 1:] + b, out=plane[:, :-1])
                np.minimum(plane[1:, 1:], prev[:-1, :-1] + c, out=plane[1:, 1:])
                np.minimum(plane[1:, :-1], prev[:-1, 1:] + c, out=plane[1:, :-1])
                np.minimum(plane[:-1, 1:], prev[1:, :-1] + c, out=plane[:-1, 1:])
                np.minimum(plane[:-1, :-1], prev[1:, 1:] + c, out=plane[:-1, :-1])
            _scan_2d(plane, a, b, None)

    forward(d)
    forward(d[::-1, ::-1, ::-1])
    d[~np.isfinite(d)] = 0.0
    return d


@op("morpho.chamfer_distance_map", "Chamfer Distance Map", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("weights", "choice", "Borgefors", choices=CHAMFER_NAMES, help="2D: Chessboard (1,1), City-Block (1,2), Quasi-Euclidean (1,1.41), Borgefors (3,4), Chessknight (5,7,11); 3D: (1,1,1), (1,2,3), (1,1.41,1.73), Borgefors (3,4,5)"),
            Param("normalize", "bool", True, help="Divide by the first weight so the distances are in pixels"),
            Param("output", "choice", "32-bit", choices=["32-bit", "16-bit"], help="16-bit rounds the distances"),
            Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D, help="3D: distances through the z-stack (Chessknight is 2D only)"),
            Param("all_planes", "bool", False, label="Process whole stack", help="2D: every plane")],
    description="MorphoLibJ's Chamfer Distance Map (2D / 3D): distance of every foreground (non-zero) pixel to the nearest background pixel with integer chamfer weights – chessboard, city-block, quasi-Euclidean, Borgefors (3,4) / (3,4,5) or chessknight (5,7,11) – optionally normalised by the first weight.",
    keywords=["morpholibj", "distance map", "chamfer", "borgefors", "chessknight", "distance transform", "3d"])
def chamfer_distance_map(ctx, weights="Borgefors", normalize=True, output="32-bit", mode="2D (current plane)", all_planes=False):
    img = ctx.img
    d = img.display
    three_d = str(mode).startswith("3D")
    w2, w3 = CHAMFER_WEIGHTS[str(weights)]
    if three_d:
        if w3 is None:
            raise ValueError("Chessknight weights are 2D only – choose Borgefors (3,4,5) for a 3D map")
        vols = {t: _chamfer_3d(np.nan_to_num(img.data[t, :, d.c].astype(np.float32)) != 0, w3) for t in (range(img.nt) if all_planes else [d.t])}
        out = np.zeros((img.nt if all_planes else 1, img.nz, 1, img.height, img.width), np.float64)
        for i, (t, v) in enumerate(vols.items()):
            out[t if all_planes else 0, :, 0] = v
        norm = w3[0]
    else:
        planes = _planes(img, all_planes)
        out = np.zeros(img.shape if all_planes else (1, 1, 1, img.height, img.width), np.float64)
        for i, (t, z, c) in enumerate(planes):
            ctx.check_cancel()
            out[(t, z, c) if all_planes else (0, 0, 0)] = _chamfer_2d(np.nan_to_num(img.data[t, z, c].astype(np.float32)) != 0, w2)
            if len(planes) > 1:
                ctx.progress((i + 1) / len(planes), f"Plane {i + 1}/{len(planes)}")
        norm = w2[0]
    if normalize:
        out /= norm
    data = np.clip(np.rint(out), 0, 65535).astype(np.uint16) if output == "16-bit" else out.astype(np.float32)
    new = LumenImage(data, name=f"Chamfer distance of {img.name}")
    new.calibration = type(img.calibration)(**img.calibration.to_dict())
    new.display.channels[0].lut = "Fire"
    new.display.channels[0].min, new.display.channels[0].max = 0.0, float(max(out.max(), 1.0))
    msg = f"Chamfer distance map ({weights}{', normalised' if normalize else ''}, max {out.max():.1f})"
    return OpResult(new_images=[new], message=msg, log=msg)


# ========================================================================= geodesic distance map
def _chamfer_graph(mask: np.ndarray, weights: tuple[float, ...]) -> tuple[Any, np.ndarray]:
    """Sparse graph over the pixels of ``mask`` with chamfer edge weights (orthogonal, diagonal, optional knight moves)."""
    a, b = float(weights[0]), float(weights[1])
    offsets = [((0, 1), a), ((1, 0), a), ((1, 1), b), ((1, -1), b)]
    if len(weights) > 2:
        c = float(weights[2])
        offsets += [((1, 2), c), ((2, 1), c), ((1, -2), c), ((2, -1), c)]
    idx = np.full(mask.shape, -1, np.int64)
    n = int(mask.sum())
    idx[mask] = np.arange(n)
    rows, cols, vals = [], [], []
    H, W = mask.shape
    for (dy, dx), wgt in offsets:
        if H <= abs(dy) or W <= abs(dx):
            continue
        p = idx[max(0, -dy):H - max(0, dy), max(0, -dx):W - max(0, dx)]
        q = idx[max(0, dy):H - max(0, -dy), max(0, dx):W - max(0, -dx)]
        m = (p >= 0) & (q >= 0)
        if m.any():
            rows.append(p[m])
            cols.append(q[m])
            vals.append(np.full(int(m.sum()), wgt))
    if rows:
        r, c, v = np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
        g = coo_matrix((np.concatenate([v, v]), (np.concatenate([r, c]), np.concatenate([c, r]))), shape=(n, n)).tocsr()
    else:
        g = coo_matrix((n, n)).tocsr()
    return g, idx


def _geodesic_distance(mask: np.ndarray, markers: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    """Chamfer-weighted geodesic distance inside ``mask`` from the marker pixels (NaN outside, inf when unreachable)."""
    g, idx = _chamfer_graph(mask, weights)
    out = np.full(mask.shape, np.nan, np.float64)
    seeds = idx[markers & mask]
    seeds = seeds[seeds >= 0]
    if seeds.size == 0:
        raise ValueError("No marker lies inside the mask")
    dist = dijkstra(g, directed=False, indices=seeds, min_only=True)
    out[mask] = dist
    return out


@op("morpho.geodesic_distance_map", "Geodesic Distance Map", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=True,
    params=[Param("marker_image", "image", None, help="Binary marker image (non-zero = marker); empty = the active selection (points or area) or the manager's point ROIs"),
            Param("weights", "choice", "Borgefors", choices=CHAMFER_NAMES, help="Chamfer weights of the propagation"),
            Param("normalize", "bool", True, help="Divide by the first weight so the distances are in pixels")],
    description="MorphoLibJ's Geodesic Distance Map: the shortest chamfer-weighted path length from the markers to every pixel of the mask (the current binary image), staying inside the mask; NaN outside the mask, infinite where no path exists.",
    keywords=["morpholibj", "geodesic", "distance map", "propagation", "chamfer", "mask", "markers"])
def geodesic_distance_map(ctx, marker_image=None, weights="Borgefors", normalize=True):
    img = ctx.img
    d = img.display
    H, W = img.height, img.width
    mask = np.nan_to_num(img.current_plane().astype(np.float32)) != 0
    if not mask.any():
        raise ValueError("The current image (the mask) has no foreground pixels")
    if marker_image:
        markers = _other_plane(ctx.session.get_image(marker_image), img, d.t, d.z) != 0
    else:
        markers = np.zeros((H, W), bool)
        if ctx.roi is not None and ctx.roi.get("type") in roimod.AREA_TYPES | {"point"}:
            markers = roimod.to_mask(ctx.roi, (H, W))
        else:
            from .segmentation import _marker_points
            for x, y in _marker_points(ctx, img):
                markers[y, x] = True
        if not markers.any():
            raise ValueError("Give the markers: a marker image, a point / area selection or point ROIs in the ROI manager")
    w2, _ = CHAMFER_WEIGHTS[str(weights)]
    out = _geodesic_distance(mask, markers, w2)
    if normalize:
        out /= w2[0]
    finite = out[np.isfinite(out)]
    new = LumenImage(out.astype(np.float32)[None, None, None], name=f"Geodesic distance of {img.name}")
    new.calibration = type(img.calibration)(**img.calibration.to_dict())
    new.display.channels[0].lut = "Fire"
    new.display.channels[0].min, new.display.channels[0].max = 0.0, float(finite.max()) if finite.size and finite.max() > 0 else 1.0
    unreachable = int(np.isinf(out).sum())
    msg = f"Geodesic distance map ({weights}, max {finite.max() if finite.size else 0:.1f})" + (f"; {unreachable} unreachable px" if unreachable else "")
    return OpResult(new_images=[new], message=msg, log=msg)


# ================================================================== marker-controlled watershed
def _roi_markers(ctx, img: LumenImage, kind: str, three_d: bool) -> np.ndarray:
    """Marker labels from the point / area ROIs of the manager (and the active selection) on the right planes."""
    from .segmentation import _roi_plane, _rois_for_image
    d = img.display
    H, W = img.height, img.width
    shape = (img.nz, H, W) if three_d else (H, W)
    markers = np.zeros(shape, np.int32)
    rois = list(_rois_for_image(ctx.session, img))
    if ctx.roi is not None:
        rois.append(ctx.roi)
    n = 0
    for r in rois:
        t, z = _roi_plane(r, img)
        if not three_d and (t, z) != (d.t, d.z):
            continue
        if kind == "Point ROIs":
            if r.get("type") != "point":
                continue
            for x, y in r.get("points", []):
                xi, yi = int(x), int(y)
                if 0 <= xi < W and 0 <= yi < H:
                    n += 1
                    if three_d:
                        markers[z, yi, xi] = n
                    else:
                        markers[yi, xi] = n
        else:
            if not roimod.is_area(r):
                continue
            n += 1
            m = roimod.to_mask(r, (H, W))
            if three_d:
                markers[z][m] = n
            else:
                markers[m] = n
    if n == 0:
        raise ValueError(f"No {kind.lower()} found for this image (add them to the ROI manager or draw a selection)")
    return markers


@op("morpho.marker_watershed", "Marker-controlled Watershed", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=True,
    params=[Param("markers", "choice", "Marker image", choices=WATERSHED_MARKERS, help="Marker image: a label or binary image (its components become the markers); ROIs: points or areas of the ROI manager / the active selection"),
            Param("marker_image", "image", None),
            Param("mask_image", "image", None, help="Optional binary mask: the flooding stays inside its non-zero pixels"),
            Param("input", "choice", "Image as is", choices=WATERSHED_INPUTS, help="Flood the image itself (a gradient or inverted distance map) or a gradient computed from it"),
            Param("connectivity", "choice", "4", choices=["4", "8"], help="2D: 4 / 8 neighbours; 3D: 6 / 26"),
            Param("calculate_dams", "bool", True, help="One-pixel watershed lines (label 0) between the basins"),
            Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D),
            Param("add_to_manager", "bool", False, help="2D: add one ROI per basin to the ROI manager")],
    description="MorphoLibJ's Marker-controlled Watershed (2D / 3D): flood the image from markers given as a marker image (labels or binary), point ROIs or area ROIs, optionally inside a mask, with or without dams → a new label image of the catchment basins.",
    keywords=["morpholibj", "watershed", "markers", "seeded", "basins", "dams", "interactive", "3d"])
def marker_watershed(ctx, markers="Marker image", marker_image=None, mask_image=None, input="Image as is", connectivity="4", calculate_dams=True,
                     mode="2D (current plane)", add_to_manager=False):
    img = ctx.img
    d = img.display
    three_d = str(mode).startswith("3D")
    a = np.nan_to_num(_current_array(img, three_d).astype(np.float32))
    conn = _conn_index(a.ndim, connectivity)
    if input == "Sobel gradient":
        a = skfilt.sobel(a).astype(np.float32)
    elif input == "Morphological gradient":
        fp = ndi.generate_binary_structure(a.ndim, a.ndim)
        a = (ndi.grey_dilation(a, footprint=fp, mode="nearest") - ndi.grey_erosion(a, footprint=fp, mode="nearest")).astype(np.float32)
    if markers == "Marker image":
        if not marker_image:
            raise ValueError("Choose the marker image (or markers = Point ROIs / Area ROIs)")
        other = ctx.session.get_image(marker_image)
        mk = _other_volume(other, img, d.t) if three_d else _other_plane(other, img, d.t, d.z)
        if is_binary_plane(np.asarray(mk)) or (mk.dtype == np.uint8 and np.isin(np.unique(mk), (0, 255)).all()):
            seeds = skmeas.label(mk > 0, connectivity=conn).astype(np.int32)
        else:
            seeds = _as_labels(mk, binary_to_components=False)
    else:
        seeds = _roi_markers(ctx, img, markers, three_d)
    if seeds.max() <= 0:
        raise ValueError("No markers found")
    mask = None
    if mask_image:
        other = ctx.session.get_image(mask_image)
        mask = (_other_volume(other, img, d.t) if three_d else _other_plane(other, img, d.t, d.z)) != 0
    if not three_d and ctx.mask is not None and markers == "Marker image":
        mask = ctx.mask if mask is None else (mask & ctx.mask)
    if mask is not None:
        seeds = np.where(mask, seeds, 0)
    basins = skseg.watershed(a, seeds, connectivity=conn, mask=mask, watershed_line=bool(calculate_dams)).astype(np.int32)
    n = int(len(np.unique(basins[basins > 0])))
    out = np.zeros(_result_shape(img, three_d), np.int32)
    if three_d:
        out[0, :, 0] = basins
    else:
        out[0, 0, 0] = basins
    new = _labels_image(out, f"Watershed of {img.name}", img)
    res = OpResult(new_images=[new], message=f"{n} catchment basin{'s' if n != 1 else ''}" + (" with dams" if calculate_dams else ""))
    if add_to_manager and not three_d:
        from .segmentation import _add_rois, _name_rois_by_label
        rois = roimod.label_to_rois(basins, simplify=True)
        _name_rois_by_label(rois)
        k = _add_rois(ctx, img, rois, d.t, d.z, d.c)
        res.message += f", {k} ROIs added"
    res.log = res.message
    return res


# ================================================================================ reconstruction
@op("morpho.reconstruct", "Morphological Reconstruction", CAT, kind="analysis", preview=False, supports_rgb=False, roi_aware=False,
    params=[Param("operation", "choice", "By Dilation", choices=RECONSTRUCTIONS, help="By dilation: grow the marker under the mask (current image); by erosion: shrink it above the mask"),
            Param("marker_image", "image", None, help="Marker image (same size); clipped to the mask where needed"),
            Param("connectivity", "choice", "4", choices=["4", "8"], help="2D: 4 / 8 neighbours; 3D: 6 / 26"),
            Param("mode", "choice", "2D (current plane)", choices=MODES_2D_3D)],
    description="MorphoLibJ's Morphological Reconstruction (2D / 3D) of the marker image inside the current image (the mask), by dilation or by erosion – the engine behind fill holes, kill borders, h-minima and opening by reconstruction.",
    keywords=["morpholibj", "reconstruction", "geodesic", "dilation", "erosion", "marker", "mask"])
def reconstruct(ctx, operation="By Dilation", marker_image=None, connectivity="4", mode="2D (current plane)"):
    img = ctx.img
    d = img.display
    if not marker_image:
        raise ValueError("Choose the marker image (the current image is the mask)")
    other = ctx.session.get_image(marker_image)
    three_d = str(mode).startswith("3D")
    mask = np.nan_to_num(_current_array(img, three_d).astype(np.float32))
    marker = np.nan_to_num((_other_volume(other, img, d.t) if three_d else _other_plane(other, img, d.t, d.z)).astype(np.float32))
    fp = _structure(mask.ndim, connectivity)
    if operation == "By Dilation":
        rec = _reconstruct(np.minimum(marker, mask), mask, "dilation", fp)
    else:
        rec = _reconstruct(np.maximum(marker, mask), mask, "erosion", fp)
    out = np.zeros(_result_shape(img, three_d), np.float32)
    if three_d:
        out[0, :, 0] = rec
    else:
        out[0, 0, 0] = rec
    new = img.copy(name=f"Reconstruction of {img.name}", data=back_to(out, img.data))
    msg = f"Reconstruction {operation.lower()} of {other.name} in {img.name}"
    return OpResult(new_images=[new], message=msg, log=msg)


def _border_marker(p: np.ndarray, fill: float) -> np.ndarray:
    marker = np.full(p.shape, np.float32(fill), np.float32)
    marker[0, :], marker[-1, :], marker[:, 0], marker[:, -1] = p[0, :], p[-1, :], p[:, 0], p[:, -1]
    return marker


@op("morpho.kill_borders", "Kill Borders", CAT, supports_rgb=False, roi_aware=False,
    params=[Param("connectivity", "choice", "4", choices=["4", "8"])],
    description="MorphoLibJ's Kill Borders: remove the structures touching the image border (binary objects, or bright gray-level structures connected to the border) by reconstruction from the border.",
    keywords=["morpholibj", "kill borders", "clear border", "edge", "reconstruction"])
def kill_borders(plane, ctx, connectivity="4"):
    p = np.nan_to_num(plane.astype(np.float32))
    rec = _reconstruct(_border_marker(p, float(p.min())), p, "dilation", _structure(2, connectivity))
    return back_to(p - rec, plane)


@op("morpho.fill_holes", "Fill Holes (Binary / Gray)", CAT, supports_rgb=False, roi_aware=False,
    params=[Param("connectivity", "choice", "4", choices=["4", "8"])],
    description="MorphoLibJ's Fill Holes for binary and gray-level images: fill the regions (holes, dark basins) not connected to the image border by reconstruction by erosion from the border.",
    keywords=["morpholibj", "fill holes", "gray", "reconstruction", "basins"])
def fill_holes_gray(plane, ctx, connectivity="4"):
    p = np.nan_to_num(plane.astype(np.float32))
    rec = _reconstruct(_border_marker(p, float(p.max())), p, "erosion", _structure(2, connectivity))
    return back_to(rec, plane)


@op("morpho.area_opening", "Area Opening", CAT, supports_rgb=False, roi_aware=False,
    params=[Param("operation", "choice", "Area Opening", choices=AREA_OPERATIONS, help="Opening removes bright components smaller than the area, closing removes dark ones"),
            Param("min_area", "int", 64, min=1, max=100_000_000, unit="px", help="Components (at any gray level) smaller than this are flattened"),
            Param("connectivity", "choice", "4", choices=["4", "8"])],
    description="MorphoLibJ's Area Opening for binary and gray-level images: remove the connected components smaller than a number of pixels at every gray level (attribute opening / closing).",
    keywords=["morpholibj", "area opening", "area closing", "attribute", "remove small", "gray"])
def area_opening(plane, ctx, operation="Area Opening", min_area=64, connectivity="4"):
    conn = _conn_index(2, connectivity)
    fn = skm.area_opening if operation == "Area Opening" else skm.area_closing
    return back_to(fn(plane, area_threshold=int(min_area), connectivity=conn), plane)


@op("morpho.directional_filter", "Directional Filtering", CAT, supports_rgb=False, roi_aware=False,
    params=[Param("type", "choice", "Max", choices=DIRECTIONAL_TYPES, help="Combine the directional results with their maximum (bright line-like structures) or minimum (dark ones)"),
            Param("operation", "choice", "Opening", choices=DIRECTIONAL_OPERATIONS),
            Param("line_length", "int", 20, min=3, max=500, unit="px", label="Line size"),
            Param("n_directions", "int", 32, min=2, max=180, help="Number of line orientations over 180°")],
    description="MorphoLibJ's Directional Filtering: apply a morphological (or mean / median) filter with a line structuring element in many orientations and keep the maximum (or minimum) over the orientations – enhances curvilinear structures.",
    keywords=["morpholibj", "directional", "line", "orientation", "curvilinear", "opening", "fibres", "vessels"])
def directional_filter(plane, ctx, type="Max", operation="Opening", line_length=20, n_directions=32):
    p = np.nan_to_num(plane.astype(np.float32))
    n = max(int(n_directions), 1)
    acc = None
    for i in range(n):
        ctx.check_cancel()
        fp = _line_footprint(int(line_length), 180.0 * i / n)
        if operation == "Mean":
            r = ndi.convolve(p, fp.astype(np.float32) / fp.sum(), mode="nearest")
        elif operation == "Median":
            r = ndi.median_filter(p, footprint=fp, mode="nearest")
        else:
            r = _morpho_op(p, operation, fp, 0.0)
        acc = r if acc is None else (np.maximum(acc, r) if type == "Max" else np.minimum(acc, r))
        ctx.progress((i + 1) / n, f"Direction {i + 1}/{n}")
    return back_to(acc, plane)


# ========================================================================================= overlays
@op("morpho.binary_overlay", "Binary Overlay", CAT, kind="analysis", preview=False, supports_rgb=True, roi_aware=False,
    params=[Param("mask_image", "image", None, help="Binary (or label) image: its non-zero pixels are painted over the current image"),
            Param("color", "string", "#ff0000", help="Overlay colour (name, #rrggbb or r,g,b)"),
            Param("opacity", "float", 1.0, min=0.0, max=1.0, step=0.1),
            Param("all_planes", "bool", False, label="Process whole stack")],
    description="MorphoLibJ's Binary Overlay: a new RGB image of the current image (through its LUT / display range, or its RGB) with the foreground of a binary mask painted in a colour.",
    keywords=["morpholibj", "overlay", "binary", "mask", "rgb", "colour", "render"])
def binary_overlay(ctx, mask_image=None, color="#ff0000", opacity=1.0, all_planes=False):
    img = ctx.img
    if not mask_image:
        raise ValueError("Choose the binary mask image")
    other = ctx.session.get_image(mask_image)
    try:
        col = np.array(parse_color(color), np.float32)
    except ValueError as e:
        raise ValueError(f"Colour: {e}") from e
    planes = sorted({(t, z) for (t, z, _c) in _planes(img, all_planes)})
    shape = (img.nt, img.nz, 3, img.height, img.width) if all_planes else (1, 1, 3, img.height, img.width)
    out = np.zeros(shape, np.uint8)
    alpha = float(opacity)
    n = 0
    for i, (t, z) in enumerate(planes):
        ctx.check_cancel()
        rgb = _reference_rgb(img, t, z).astype(np.float32)
        m = np.nan_to_num(_other_plane(other, img, t, z).astype(np.float32)) != 0
        n += int(m.sum())
        rgb[m] = (1 - alpha) * rgb[m] + alpha * col
        out[(t, z) if all_planes else (0, 0)] = np.moveaxis(np.clip(rgb, 0, 255).astype(np.uint8), 2, 0)
        if len(planes) > 1:
            ctx.progress((i + 1) / len(planes), f"Plane {i + 1}/{len(planes)}")
    msg = f"{n} mask pixels overlaid in {color}"
    return OpResult(new_images=[_rgb_image(out, f"{img.name}-overlay", img)], message=msg, log=msg)


@op("morpho.label_overlay", "Label Overlay", CAT, kind="analysis", preview=False, supports_rgb=True, roi_aware=False,
    params=[Param("label_image", "image", None, help="Label image (or binary mask) to blend over the current image"),
            Param("lut", "choice", "Glasbey", choices=LABEL_LUT_CHOICES),
            Param("opacity", "float", 0.5, min=0.0, max=1.0, step=0.1, help="Weight of the label colours (background pixels stay untouched)"),
            Param("shuffle", "bool", False, help="Shuffle the colour assignment"),
            Param("all_planes", "bool", False, label="Process whole stack")],
    description="MorphoLibJ's Label Overlay: a new RGB image of the current image with every label of a label image blended over it in its own colour (Glasbey, Golden angle, Random or any LUT).",
    keywords=["morpholibj", "overlay", "label", "rgb", "blend", "glasbey", "render"])
def label_overlay(ctx, label_image=None, lut="Glasbey", opacity=0.5, shuffle=False, all_planes=False):
    img = ctx.img
    if not label_image:
        raise ValueError("Choose the label image")
    other = ctx.session.get_image(label_image)
    planes = sorted({(t, z) for (t, z, _c) in _planes(img, all_planes)})
    labs = {(t, z): _as_labels(_other_plane(other, img, t, z)) for (t, z) in planes}
    uniq = np.unique(np.concatenate([np.unique(v) for v in labs.values()]))
    uniq = uniq[uniq > 0]
    cols = _label_colors(int(len(uniq)), str(lut), bool(shuffle), 1).astype(np.float32)
    shape = (img.nt, img.nz, 3, img.height, img.width) if all_planes else (1, 1, 3, img.height, img.width)
    out = np.zeros(shape, np.uint8)
    alpha = float(opacity)
    for i, (t, z) in enumerate(planes):
        ctx.check_cancel()
        rgb = _reference_rgb(img, t, z).astype(np.float32)
        lab = labs[(t, z)]
        fg = (lab > 0) & np.isin(lab, uniq)
        if fg.any():
            ranks = np.searchsorted(uniq, lab[fg])
            rgb[fg] = (1 - alpha) * rgb[fg] + alpha * cols[ranks]
        out[(t, z) if all_planes else (0, 0)] = np.moveaxis(np.clip(rgb, 0, 255).astype(np.uint8), 2, 0)
        if len(planes) > 1:
            ctx.progress((i + 1) / len(planes), f"Plane {i + 1}/{len(planes)}")
    msg = f"{len(uniq)} label{'s' if len(uniq) != 1 else ''} of {other.name} overlaid ({lut}, opacity {alpha:g})"
    return OpResult(new_images=[_rgb_image(out, f"{img.name}-labels", img)], message=msg, log=msg)
