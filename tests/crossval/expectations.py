"""Per-case expectations of the ImageJ 1.54p cross-validation (see docs/CROSSVALIDATION.md for the table).

``EXPECT[case_id]`` → ``expect`` (exact | tolerance | mismatch), ``tol`` (the bound the test asserts) and ``note`` (why the
bound is needed, or – for a mismatch – the measured difference and a one-line diagnosis).  Cases without an entry must be
exact.  Policy: "tolerance" is used only for floating-point accumulation / kernel truncation and for the two documented
±1 grey-level conventions (ImageJ rounds half up where numpy rounds half to even; ImageJ truncates the 8-bit result of
Log / Exp / Square Root / Gamma / Find Edges); every other systematic difference is recorded as a mismatch.
"""
from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------- shared diagnoses
ROUND = "±1 grey level: ImageJ rounds half up ((int)(v+0.5)), Lumen rounds half to even (numpy rint) – every differing pixel is an x.5 case (verified 100%)"
TRUNC = "±1 grey level: ImageJ truncates ((int)v) the 8-bit result of this operation, Lumen rounds (verified: ImageJ == floor(v) on 100% of pixels)"
FLOAT = "float32 accumulation order (ImageJ sums in float, Lumen in float64 and casts)"
MATH16 = "ImageJ scales 16-bit Gamma / Log to the image's own [min, max] (121–4875 here: 4875 stays 4875), Lumen to the type range 0–65535 (4875 → 17874 / 50180)"
SCALE8 = "16→8-bit scaling: ImageJ uses (int)((v−min)·256/(max−min+1) + 0.5) (verified 100%), Lumen round((v−min)·255/(max−min)) → ±1 in 12–16% of pixels"
LUT8 = "8-bit display-range mapping: ImageJ truncates (v−min)·256/(max−min) (verified 100%), Lumen rounds ·255/(max−min) → ±1 in ~9% of pixels"
OVAL = "the oval's polygon differs (ImageJ OvalRoi integer polygon vs Lumen's 72-vertex ellipse): Feret, FeretAngle (0 vs 5.79), MinFeret, Solidity differ; Area/Mean/StdDev identical"
ELLIPSE = "Major/Minor: ImageJ's EllipseFitter (same area and moments, 67.70 for the 60×45 rectangle) vs Lumen's moment ellipse (69.27); identical only for ellipse-like shapes"
ANGLE = "ellipse Angle: ImageJ measures counter-clockwise from the x axis with y up (0–180°), Lumen reports 180°−θ (173.6 vs 6.4)"
POLY = "polygon rasterisation: ImageJ's PolygonFiller (even-odd, pixel centres) covers 6645 px, Lumen's skimage polygon 6655 px → Area/IntDen/centroid/shape differ slightly"
EXTRA = "Lumen adds Ch and Frame columns (ImageJ writes only Slice for a non-hyperstack)"
NOOP = "setOption() is a no-op in Lumen's macro layer"
BINARY_NB = "ImageJ's binary Erode/Dilate (count=1) use all 8 neighbours; Lumen changes fewer pixels (a 4-connected neighbourhood?): "


def _t(pix_abs: float, pix_frac: float, note: str) -> dict[str, Any]:
    return {"expect": "tolerance", "tol": {"pix_abs": pix_abs, "pix_frac": pix_frac}, "note": note}


def _m(note: str, **tol: float) -> dict[str, Any]:
    return {"expect": "mismatch", "tol": tol, "note": note}


EXPECT: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------ filters
    # ------------------------------------------------------------ enhance / background
    # ------------------------------------------------------------ thresholds: exact (ports of AutoThresholder + ImageProcessor.setAutoThreshold / scaleAndSetThreshold)
    # ------------------------------------------------------------ binary: exact (ports of ByteProcessor.erode/dilate, BinaryProcessor.outline/skeletonize, EDM + MaximumFinder)
    "binary_fill_holes_stack": _m("ImageJ side: the Convert Stack to Binary dialog resets Prefs.blackBackground (no 'black' key in the options) and in batch mode Fill Holes "
                                  "then ran with foreground = 0 – its output keeps only the border-connected 255 regions (3 slices empty, no hole filled; verified pixel-exactly); "
                                  "Lumen fills the holes of every slice (12% of voxels differ)"),
    # ------------------------------------------------------------ particles
    # analyze_particles_all / _filtered / _calibrated / _masks / _count_masks: exact (ImageJ ParticleAnalyzer + Analyzer ports, ops/_imagej_measure.py)
    # ------------------------------------------------------------ measure
    # measure_rect / _oval / _polygon / _whole_image / _slice4_zstack / _channel2 / _line / _line_wide / _point / _limit_threshold /
    # _invert_y, set_scale_measure, calibrated_file_measure: exact – ImageJ's Analyzer / ImageStatistics / EllipseFitter / PolygonFiller /
    # OvalRoi / Roi.getFeretValues / Line.getPixels ports (ops/_imagej_measure.py); Lumen's extra Ch / Frame / Value columns are reported, not failed
    # ------------------------------------------------------------ stacks
    # zproject_avg / _median / _sd / _max / _min / _sum / _range_2_4: exact (ZProjector port in ops/stacks.py: AVG_/MED_/STD_ titles, float32 accumulation,
    # (int)(v + 0.5f) into 8/16-bit, the SD's double sums, the float32 median of the two middle slices)
    "stack_statistics": {"expect": "tolerance", "tol": {"res_atol": 5e-4},
                         "note": "ImageJ writes the Stack Statistics Results row with the default 3 decimals (this macro sets no decimal=9): Mean 53.554 / StdDev 52.540 vs "
                                 "Lumen's 53.554326 / 52.540440; the printed Stack.getStatistics values agree to 6 decimals; %Volume, Voxels, Min, Max, Mode, Median identical "
                                 "(Lumen's extra Label / Volume columns are reported, not failed)"},
    # ------------------------------------------------------------ transforms: exact (ports of ImageProcessor.rotate / resize / scale / translate, DownsizeTable, Rotator.enlargeCanvas)
    # ------------------------------------------------------------ calculator / math: exact (ports of Byte/Short/FloatProcessor.process; Average = integer (a+b)/2 like ByteBlitter)
    # ------------------------------------------------------------ types: exact (ports of TypeConverter, ColorProcessor weighting factors, StackConverter per-channel ranges, setOption ScaleConversions)
    # ------------------------------------------------------------ FFT
    # ------------------------------------------------------------ maxima
    # find_maxima_count / _strict_exclude / _points / _single_points / _segmented / _list: exact (MaximumFinder port incl. its watershed, ops/_imagej_measure.py)
    # ------------------------------------------------------------ histogram / profile / statistics
    # histogram_8bit_256 / _16bit_256 / _32bit_64 / _roi: exact (IJ.getHistogram → ops/_imagej_measure.macro_histogram, ImageJ's (int)((v−min)/binSize) bins)
    # get_statistics[blobs8/16/32]: exact (ShortStatistics median port)
    # profile_line[blobs8/32] / profile_line_wide / profile_rect: exact (getProfile() = ProfilePlot port in ops/_imagej_roi.py: ImageProcessor.getLine at the line's own
    # coordinates, Straightener.straightenLine (spline in bounds-relative float32 coordinates, 1-px equidistant points, column averages) for wide lines, column averages for rectangles)
    # ------------------------------------------------------------ ROI geometry: exact (ops/_imagej_roi.py ports)
    # roi_polygon_mask / roi_oval_mask: exact (PolygonFiller / OvalRoi.getMask ports in roi.to_mask)
    # roi_enlarge_5 / _rect: exact (RoiEnlarger.enlargeRectOrOval); roi_enlarge_minus5: exact (shrink255: 8-bit EDM ≥ n + 1)
    # roi_band_5: exact (Selection.makeBand: 8-bit EDM of the outside ≤ n, legacy Wand trace, minus the selection)
    # roi_interpolate / _nosmooth: exact (Roi.getInterpolatedPolygon; polygons are not smoothed); roi_scale_1_5: exact (RoiScaler: rounded vertices about the mask centroid)
    # roi_rotate_30: exact (RoiRotator + PolygonRoi.setIntBounds bounds); roi_line_to_area: exact (RotatedRectRoi polygon, FloatPolygon.getBounds bounds)
    # roi_fit_ellipse: exact (EllipseFitter → 72-vertex EllipseRoi); roi_draw: exact (OvalRoi.getPolygon Wand outline + ImageProcessor.lineTo)
    # ------------------------------------------------------------ LUTs: exact (ByteProcessor.setMinAndMax index mapping, ContrastAdjuster.apply table, 16-bit create8BitImage)
}
