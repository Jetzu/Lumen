"""Process ▸ Binary (foreground = 255, background = 0, always).

Erode / Dilate / Open / Close, Outline, Skeletonize, Distance Map, Ultimate Points, Watershed and Voronoi reproduce
ImageJ 1.54p pixel for pixel by default (ports in :mod:`._imagej_binary`); the Lumen extensions (disk radii, other
skeletonization methods, the h-maxima watershed, 32-bit outputs) stay available as parameters.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology as skm, measure as skmeas, segmentation as skseg

from ..registry import Param, OpResult, op
from . import _imagej_binary as ijb
from ._util import footprint, is_binary_plane

CAT = "Process/Binary"
OUTLINE_IJ = "ImageJ (black background)"
OUTLINE_INNER = "Inner boundary"
OUTLINE_OUTER = "Outer ring"
OUTLINE_IJ_WHITE = "ImageJ (white background)"
OUTLINE_MODES = [OUTLINE_IJ, OUTLINE_INNER, OUTLINE_OUTER, OUTLINE_IJ_WHITE]
EDM_OUTPUTS = ["Overwrite", "8-bit", "16-bit", "32-bit"]
_BINARY_PARAMS = [Param("iterations", "int", 1, min=1, max=100, help="Repeat the operation (Process ▸ Binary ▸ Options ▸ Iterations)"),
                  Param("count", "int", 1, min=1, max=8, help="Number of background (foreground) neighbours needed to remove (add) a pixel (Options ▸ Count)"),
                  Param("pad_edges", "bool", False, label="Pad edges when eroding", help="Pixels outside the image count as foreground when eroding (ImageJ: objects touching the border are not eroded there)"),
                  Param("radius", "int", 1, min=1, max=50, advanced=True, help="1 = ImageJ's 3×3 neighbourhood; larger radii use a disk (Lumen extension, count / pad edges ignored)")]


def _edm_result(img, edm_planes: dict, output: str, name: str):
    """A Distance Map / Ultimate Points result in ImageJ's output type: 'Overwrite' (8-bit, in place) or a new
    8 / 16 / 32-bit image (``EDM.setOutputType``: 8/16-bit = (int)(v + 0.5) clipped, 32-bit = float)."""
    if output == "Overwrite":
        out = np.zeros(img.shape, np.uint8)
        for key, edm in edm_planes.items():
            out[key] = ijb.edm_to_byte(edm)
        return OpResult(data=out, rgb=False, display={"channels": [{"min": 0, "max": 255} for _ in range(img.nc)]})
    if output == "32-bit":
        out = np.zeros(img.shape, np.float32)
        for key, edm in edm_planes.items():
            out[key] = edm
    elif output == "16-bit":
        out = np.zeros(img.shape, np.uint16)
        for key, edm in edm_planes.items():
            out[key] = np.minimum(np.floor(edm.astype(np.float64) + 0.5), 65535).astype(np.uint16)
    else:
        out = np.zeros(img.shape, np.uint8)
        for key, edm in edm_planes.items():
            out[key] = ijb.edm_to_byte(edm)
    res = img.copy(name=name, data=out)
    res.rgb = False
    res.lut_is_binary = False
    res.sync_channels()
    for ch in res.display.channels:
        ch.lut = "Fire" if res.nc == 1 else ch.lut
        ch.min, ch.max = 0.0, float(out.max() or 1.0)
    return OpResult(new_images=[res])


def _bin(plane: np.ndarray) -> np.ndarray:
    return plane > 0 if plane.dtype != np.float32 else plane > 0


def _out(mask: np.ndarray) -> np.ndarray:
    return mask.astype(np.uint8) * 255


@op("binary.make", "Make Binary", CAT, kind="image", scope_param=False, supports_rgb=False, preview=True,
    description="Convert to a 0/255 mask using the current threshold (or the Default auto threshold).",
    params=[Param("method", "choice", "Current", choices=["Current", "Default", "Otsu", "Li", "Triangle", "Huang", "Yen", "MaxEntropy"]), Param("dark_background", "bool", True)])
def make_binary(ctx, method="Current", dark_background=True):
    from .threshold_ops import apply_threshold
    return apply_threshold(ctx, method=method, dark_background=dark_background)


@op("binary.erode", "Erode", CAT, description="Remove pixels from the edges of foreground objects (ImageJ: a pixel with at least 'count' of its 8 neighbours in the background is removed; the image border counts as background unless 'pad edges').",
    params=_BINARY_PARAMS)
def erode(plane, ctx, iterations=1, count=1, pad_edges=False, radius=1):
    b = _bin(plane)
    if radius > 1:
        return _out(ndi.binary_erosion(b, structure=footprint(radius), iterations=iterations, border_value=1))
    for _ in range(max(int(iterations), 1)):
        b = ijb.ij_erode(b, count, pad_edges)
    return _out(b)


@op("binary.dilate", "Dilate", CAT, description="Add pixels to the edges of foreground objects (ImageJ: a background pixel with at least 'count' foreground neighbours is added).",
    params=_BINARY_PARAMS)
def dilate(plane, ctx, iterations=1, count=1, pad_edges=False, radius=1):
    b = _bin(plane)
    if radius > 1:
        return _out(ndi.binary_dilation(b, structure=footprint(radius), iterations=iterations))
    for _ in range(max(int(iterations), 1)):
        b = ijb.ij_dilate(b, count)
    return _out(b)


@op("binary.open", "Open", CAT, description="Erosion followed by dilation ('iterations' of each, ImageJ's Open): removes small specks.",
    params=_BINARY_PARAMS)
def open_(plane, ctx, iterations=1, count=1, pad_edges=False, radius=1):
    b = _bin(plane)
    if radius > 1:
        return _out(ndi.binary_opening(b, structure=footprint(radius), iterations=iterations))
    return _out(ijb.ij_open(b, iterations, count, pad_edges))


@op("binary.close", "Close", CAT, description="Dilation followed by erosion ('iterations' of each, ImageJ's Close-): fills small gaps.",
    params=_BINARY_PARAMS)
def close_(plane, ctx, iterations=1, count=1, pad_edges=False, radius=1):
    b = _bin(plane)
    if radius > 1:
        return _out(ndi.binary_closing(b, structure=footprint(radius), iterations=iterations, border_value=0))
    return _out(ijb.ij_close(b, iterations, count, pad_edges))


@op("binary.outline", "Outline", CAT, description="One-pixel-wide outlines of foreground objects.",
    params=[Param("mode", "choice", OUTLINE_IJ, choices=OUTLINE_MODES,
                  help="ImageJ (black background): exactly what Process ▸ Binary ▸ Outline produces with 'Black background' set – the 8-connected ring of background pixels around every object (ImageJ outlines the inverted image but reads the pre-inversion snapshot); border pixels are left as they are. "
                       "Inner boundary: the object's own 8-connected boundary (ImageJ with a white background and an inverting LUT). Outer ring: the ring around the objects including the image border. "
                       "ImageJ (white background): what ImageJ makes of a 0/255 mask with 'Black background' off – everything 255 except the ring of background around the objects")])
def outline(plane, ctx, mode=OUTLINE_IJ):
    b = _bin(plane)
    if mode == OUTLINE_OUTER:
        return _out((~b) & ndi.binary_dilation(b, structure=np.ones((3, 3), bool)))
    if mode == OUTLINE_IJ_WHITE:
        return _out(ijb.ij_outline_white_bg(b))
    return _out(ijb.ij_outline(b, black_background=mode != OUTLINE_INNER))


@op("binary.fill_holes", "Fill Holes", CAT, description="Fill background regions enclosed by foreground.")
def fill_holes(plane, ctx):
    return _out(ndi.binary_fill_holes(_bin(plane)))


SKELETON_METHODS = ["ImageJ", "Zhang-Suen", "Lee", "Thin (Guo-Hall)", "Medial axis"]


@op("binary.skeletonize", "Skeletonize", CAT, description="Reduce objects to one-pixel-wide skeletons.",
    params=[Param("method", "choice", "ImageJ", choices=SKELETON_METHODS,
                  help="ImageJ: Process ▸ Binary ▸ Skeletonize's lookup-table thinning, pixel for pixel; Zhang-Suen, Lee (3-D style thinning), Guo-Hall thinning or the medial axis (ridge of the distance map) are scikit-image's")])
def skeletonize(plane, ctx, method="ImageJ"):
    b = _bin(plane)
    if method == "ImageJ":
        return _out(ijb.ij_skeletonize(b))
    if method == "Lee":
        return _out(skm.skeletonize(b, method="lee") > 0)
    if method.startswith("Thin"):
        return _out(skm.thin(b))
    if method == "Medial axis":
        return _out(skm.medial_axis(b))
    return _out(skm.skeletonize(b, method="zhang"))


@op("binary.distance_map", "Distance Map", CAT, kind="image", scope_param=False, preview=False,
    description="Euclidean distance of every foreground pixel to the nearest background pixel. 'Overwrite' replaces the image with the 8-bit EDM (ImageJ's default: (int)(distance + 0.5), clipped to 255); 8/16/32-bit create a new image (Process ▸ Binary ▸ Options ▸ EDM output).",
    params=[Param("output", "choice", "Overwrite", choices=EDM_OUTPUTS, help="Overwrite = 8-bit EDM in place (ImageJ's default); 32-bit = exact float distances as a new image")])
def distance_map(ctx, output="Overwrite"):
    img = ctx.img
    planes = {key: ijb.edm_float(_bin(img.data[key])) for key in img.iter_planes()}
    return _edm_result(img, planes, output, f"EDM of {img.name}")


def ultimate_eroded_points(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(points (N, 2) as (row, col), float EDM) of the ultimate eroded points of a boolean mask – ImageJ's
    EDM ▸ Ultimate Points: MaximumFinder(SINGLE_POINTS, tolerance 0.5, excludeOnEdges=false) on the EDM, one
    point per maximum (the pixel nearest the centroid of the equal-height plateau), maxima on the border kept."""
    edm = ijb.edm_float(b)
    if not b.any():
        return np.zeros((0, 2), int), edm
    pts = ijb.find_maxima(edm, 0.5, ijb.SINGLE_POINTS, False, True)
    ys, xs = np.nonzero(pts)
    return np.stack([ys, xs], axis=1).astype(int).reshape(-1, 2), edm


UEP_OUTPUTS = ["Overwrite", "Point selection", "Image", "8-bit", "16-bit"]


@op("binary.ultimate_points", "Ultimate Points", CAT, kind="image", scope_param=False, preview=False,
    description="Ultimate eroded points: the maxima of the distance map. 'Overwrite' replaces the image with the UEPs carrying their rounded EDM value (ImageJ's default), 'Point selection' makes them a point ROI, 'Image' (32-bit) / 8-bit / 16-bit create a new image holding the EDM value (= radius of the inscribed circle) at every UEP.",
    params=[Param("output", "choice", "Overwrite", choices=UEP_OUTPUTS)])
def ultimate_points(ctx, output="Overwrite"):
    img = ctx.img
    if output == "Point selection":
        pts, _ = ultimate_eroded_points(_bin(img.current_plane()))
        roi = {"type": "point", "points": [[float(x) + 0.5, float(y) + 0.5] for y, x in pts], "name": "UEPs"}
        return OpResult(roi=roi, message=f"{len(pts)} ultimate points")
    planes = {}
    n = 0
    for key in img.iter_planes():
        _, uep = ijb.ij_ultimate_points(_bin(img.data[key]))
        planes[key] = uep
        n += int((uep > 0).sum())
    res = _edm_result(img, planes, "32-bit" if output == "Image" else output, f"UEP of {img.name}")
    res.message = f"{n} ultimate points"
    return res


WATERSHED_METHODS = ["ImageJ EDM/UEP", "Distance transform (h-maxima)"]


def imagej_watershed(b: np.ndarray) -> np.ndarray:
    """ImageJ's binary watershed (``EDM.toWatershed``): MaximumFinder(SEGMENTED, tolerance 0.5) on the float EDM –
    the maxima are the UEPs, the EDM is quantised to 8 bits and flooded level by level with ImageJ's fate table."""
    return ijb.ij_watershed(b)


@op("binary.watershed", "Watershed", CAT, description="Split touching objects using the distance-transform watershed.",
    params=[Param("method", "choice", WATERSHED_METHODS[0], choices=WATERSHED_METHODS,
                  help="ImageJ EDM/UEP: Process ▸ Binary ▸ Watershed pixel for pixel (EDM maxima with tolerance 0.5, 8-bit flooding); h-maxima: scikit-image flooding where tolerance / min distance are honoured"),
            Param("tolerance", "float", 0.5, min=0.0, max=50.0, step=0.5, help="Minimum depth (h-maxima) of a basin to count as a separate object"), Param("min_distance", "int", 3, min=1, max=100, advanced=True)])
def watershed(plane, ctx, method=WATERSHED_METHODS[0], tolerance=0.5, min_distance=3):
    b = _bin(plane)
    if not b.any():
        return _out(b)
    if method.startswith("ImageJ"):
        return _out(imagej_watershed(b))
    edm = ndi.distance_transform_edt(b)
    lab = skmeas.label(b)
    from skimage.feature import peak_local_max
    pts = peak_local_max(edm, min_distance=min_distance, threshold_abs=max(tolerance, 0.5), labels=lab, exclude_border=False)
    markers = np.zeros_like(lab)
    for i, (y, x) in enumerate(pts, 1):
        markers[y, x] = i
    if tolerance > 0.5:
        hm = skm.h_maxima(edm, tolerance)
        markers = skmeas.label(hm & b)
    ws = skseg.watershed(-edm, markers, mask=b, watershed_line=True)
    return _out((ws > 0) & b)


@op("binary.voronoi", "Voronoi", CAT, description="Lines of points equidistant to the nearest two objects (background partition). ImageJ writes the (rounded) distance to the objects along the lines; '255' draws plain binary lines.",
    params=[Param("values", "choice", "EDM", choices=["EDM", "255"], help="EDM: the line pixels carry the 8-bit distance map value of the background (ImageJ's Voronoi with EDM output 'Overwrite'); 255: a binary line image")])
def voronoi(plane, ctx, values="EDM"):
    b = _bin(plane)
    out = ijb.ij_voronoi(b)
    if values == "255":
        return _out(out > 0)
    return out


@op("binary.remove_small", "Remove Small Objects", CAT, description="Delete foreground components smaller than a size (px²).",
    params=[Param("min_size", "int", 64, min=1, max=10_000_000)])
def remove_small(plane, ctx, min_size=64):
    return _out(skm.remove_small_objects(_bin(plane), max_size=min_size - 1))


@op("binary.label", "Label Connected Components", CAT, kind="image", scope_param=False, preview=False,
    description="Create a label image (one integer per connected object) with a categorical LUT.",
    params=[Param("connectivity", "choice", "8", choices=["4", "8"])])
def label(ctx, connectivity="8"):
    img = ctx.img
    conn = 2 if connectivity == "8" else 1
    labs = np.zeros(img.shape, np.int32)
    for (t, z, c) in img.iter_planes():
        labs[t, z, c] = skmeas.label(_bin(img.data[t, z, c]), connectivity=conn)
    n = int(labs.max())
    # uint8 up to 255 labels, uint16 up to 65535, 32-bit beyond (exact to 2**24): a premature uint16 cast would wrap
    # object 65536 to background and make objects 65537+ share the labels 1, 2, … with earlier objects
    from .segmentation import _any_count_label_image
    res = _any_count_label_image(labs, f"Labels of {img.name}", like=img)
    res.data = res.data.reshape(img.shape) if res.data.shape != img.shape else res.data
    res.display.channels[0].max = float(max(n, 1))
    return OpResult(new_images=[res], message=f"{n} objects")


@op("binary.convex_hull", "Convex Hull (binary)", CAT, description="Replace each object with its convex hull.")
def convex_hull(plane, ctx):
    return _out(skm.convex_hull_object(_bin(plane)))


@op("binary.dither", "Dither", CAT, description="Floyd–Steinberg error-diffusion dithering of a grayscale plane to a 0/255 binary pattern.", keywords=["halftone", "floyd", "steinberg"])
def dither(plane, ctx):
    f = plane.astype(np.float64)
    if plane.dtype != np.uint8:
        lo, hi = float(np.nanmin(f)), float(np.nanmax(f))
        f = (f - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(f)
    return _out(floyd_steinberg(np.nan_to_num(f)))


def floyd_steinberg(f: np.ndarray, threshold: float = 127.5) -> np.ndarray:
    """Serpentine-free Floyd–Steinberg dithering of a float array in 0…255. Returns a boolean mask.

    Error diffusion in raster order is sequential along a row, but every pixel on the anti-diagonal wavefront
    x + 2·y = k only depends on earlier wavefronts (its 7/16 neighbour to the left and the 3/16 neighbour above-right
    lie on k − 1, the 5/16 neighbour above on k − 2 and the 1/16 neighbour above-left on k − 3), so a whole wavefront
    is quantised at once.  In the padded buffer a wavefront is a strided slice (stride W), the four error terms are
    added in the order a raster scan would add them (1/16, 5/16, 3/16, then 7/16) and the result is bit-identical
    to the classic per-pixel loop.
    """
    f = np.asarray(f, dtype=np.float64)
    H, W = f.shape
    if H == 0 or W == 0:
        return np.zeros((H, W), bool)
    thr = float(threshold)
    Wp = W + 2                                   # one spare column on each side, one spare row below
    buf = np.zeros((H + 1) * Wp, np.float64)
    buf.reshape(H + 1, Wp)[:H, 1:W + 1] = f
    for k in range(W + 2 * H - 2):
        ylo = max(0, (k - W + 2) // 2)           # x = k − 2y must stay within 0 … W−1
        yhi = min(H - 1, k // 2)
        start = k + 1 + ylo * W                  # flat index of (ylo, k − 2·ylo) in the padded buffer
        stop = start + (yhi - ylo + 1) * W
        old = buf[start:stop:W]
        err = old - (old >= thr) * 255.0
        buf[start + Wp - 1:stop + Wp - 1:W] += err * 0.1875   # 3/16 → (y+1, x−1); same targets as the 7/16 term, raster order first
        buf[start + 1:stop + 1:W] += err * 0.4375             # 7/16 → (y, x+1)
        buf[start + Wp:stop + Wp:W] += err * 0.3125           # 5/16 → (y+1, x)
        buf[start + Wp + 1:stop + Wp + 1:W] += err * 0.0625   # 1/16 → (y+1, x+1)
    return buf.reshape(H + 1, Wp)[:H, 1:W + 1] >= thr


@op("binary.median", "Binary Median (majority)", CAT, description="Majority filter: a pixel becomes foreground when more than half of its circular neighbourhood is foreground (binary median, removes specks and fills pits).",
    params=[Param("radius", "float", 2.0, min=0.5, max=100, step=0.5)], keywords=["majority", "vote"])
def binary_median(plane, ctx, radius=2.0):
    fp = footprint(radius).astype(np.float32)
    count = ndi.convolve(_bin(plane).astype(np.float32), fp, mode="nearest")
    return _out(count > fp.sum() / 2.0)


@op("binary.signed_distance_map", "Signed Distance Map", CAT, kind="image", scope_param=False, preview=False, supports_rgb=False,
    description="Signed Euclidean distance transform as a new 32-bit image: positive inside foreground objects (distance to the nearest background pixel), negative outside (distance to the nearest object).",
    keywords=["edt", "sdf", "level set"])
def signed_distance_map(ctx):
    img = ctx.img
    out = np.zeros(img.shape, np.float32)
    for (t, z, c) in img.iter_planes():
        b = _bin(img.data[t, z, c])
        out[t, z, c] = ndi.distance_transform_edt(b) - ndi.distance_transform_edt(~b)
    res = img.copy(name=f"Signed EDM of {img.name}", data=out)
    res.rgb = False
    res.lut_is_binary = False
    res.sync_channels()
    for ch in res.display.channels:
        ch.min, ch.max = float(out.min()), float(out.max() or 1.0)
    return OpResult(new_images=[res])


@op("binary.distance_map_3d", "Distance Map 3D", CAT, kind="image", scope_param=False, preview=False, supports_rgb=False,
    description="3-D Euclidean distance transform over X, Y and Z of every channel/frame (new 32-bit image). Distances are in calibrated units and honour anisotropic voxels (pixel depth ≠ pixel width).",
    params=[Param("calibrated", "bool", True, help="Use the voxel size (pixel width/height/depth) so distances are in calibrated units; off = voxel units with isotropic voxels")],
    keywords=["edt", "3d", "anisotropic"])
def distance_map_3d(ctx, calibrated=True):
    img = ctx.img
    cal = img.calibration
    sampling = (cal.pixel_depth, cal.pixel_height, cal.pixel_width) if calibrated else None
    out = np.zeros(img.shape, np.float32)
    for t in range(img.nt):
        for c in range(img.nc):
            out[t, :, c] = ndi.distance_transform_edt(_bin(img.data[t, :, c]), sampling=sampling)
    res = img.copy(name=f"EDM 3D of {img.name}", data=out)
    res.rgb = False
    res.lut_is_binary = False
    res.sync_channels()
    for ch in res.display.channels:
        ch.lut = "Fire" if res.nc == 1 else ch.lut
        ch.min, ch.max = 0.0, float(out.max() or 1.0)
    return OpResult(new_images=[res])
