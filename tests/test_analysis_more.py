"""Analyze gap fillers: Analyze Line Graph, Grid, Compare Images, Directionality, measurement options,
new calibration fit functions, and the Fiji-style gel lane workflow."""
import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.fitting import FIT_NAMES, evaluate, fit
from lumen.core.image import LumenImage
from lumen.core.measure import MEASURE_OPTION_DEFAULTS, apply_measure_options, measure_labels, measure_roi
from lumen.core.session import Session


# ====================================================================== Analyze Line Graph
def _plot_image(two_curves: bool = False) -> np.ndarray:
    H, W = 100, 200
    rgb = np.full((H, W, 3), 255, np.uint8)
    rgb[90, 10:190] = 0
    rgb[10:90, 10] = 0  # axes
    xs = np.arange(12, 188)
    ys = (50 - 30 * np.sin(xs / 20.0)).astype(int)
    for x, y in zip(xs, ys):
        rgb[y - 1:y + 1, x] = (255, 0, 0)
    if two_curves:
        for x in xs:
            rgb[70, x] = (0, 0, 255)
    return rgb


def test_line_graph_extracts_curves():
    s = Session()
    s.add_image(LumenImage(_plot_image(), name="plot", rgb=True))
    out = s.run_op("analyze.line_graph", {"tolerance": 40})
    assert len(out["plot"]["series"]) == 1 and out["results_added"] == 176
    rows = s.results.rows
    xs = np.array([r["X"] for r in rows])
    ys = np.array([r["Y"] for r in rows])
    assert xs[0] == 12 and xs[-1] == 187
    expect = 99 - (50 - 30 * np.sin(xs / 20.0))  # invert_y: measured from the bottom row
    assert np.abs(ys - expect).max() <= 1.5
    # two colours → two curves; axis ranges map pixels to data units
    s.add_image(LumenImage(_plot_image(True), name="plot2", rgb=True))
    s.results.clear()
    out = s.run_op("analyze.line_graph", {"tolerance": 40, "x_min": 0, "x_max": 10, "y_min": 0, "y_max": 1})
    assert len(out["plot"]["series"]) == 2
    rows = [r for r in s.results.rows if r["Label"].endswith("curve 2") or r["Label"].endswith("curve 1")]
    assert all(0 <= r["X"] <= 10 and 0 <= r["Y"] <= 1 for r in rows)
    flat = s.add_image(LumenImage(np.full((20, 20, 3), 255, np.uint8), name="blank", rgb=True))
    with pytest.raises(ValueError):
        s.run_op("analyze.line_graph", {}, image_id=flat.id)


def test_line_graph_curves_keep_their_own_x():
    # two curves with different column ranges: every series carries its own x (as long as its values) and the plot's
    # x axis is the union of all of them – the second curve must not be reported on the first curve's x values
    H, W = 100, 200
    rgb = np.full((H, W, 3), 255, np.uint8)
    rgb[50, 10:190] = (255, 0, 0)
    rgb[20, 120:170] = (0, 0, 255)
    s = Session()
    s.add_image(LumenImage(rgb, name="plot", rgb=True))
    plot = s.run_op("analyze.line_graph", {"tolerance": 40})["plot"]
    assert len(plot["series"]) == 2
    red, blue = plot["series"]
    assert len(red["x"]) == len(red["values"]) == 180 and red["x"][0] == 10 and red["x"][-1] == 189
    assert len(blue["x"]) == len(blue["values"]) == 50 and blue["x"][0] == 120 and blue["x"][-1] == 169
    assert red["values"][0] == pytest.approx(99 - 50) and blue["values"] == pytest.approx([99 - 20] * 50)
    assert plot["x"] == sorted(set(red["x"]) | set(blue["x"])) and plot["x"][0] == 10 and plot["x"][-1] == 189
    rows = [r for r in s.results.rows if r["Curve"] == 2]
    assert [r["X"] for r in rows] == blue["x"] and all(r["Y"] == pytest.approx(79) for r in rows)
    # with a rectangle around part of the plot the x values are still image columns
    plot = s.run_op("analyze.line_graph", {"tolerance": 40}, roi={"type": "rect", "x": 100, "y": 0, "w": 100, "h": 100})["plot"]
    red, blue = plot["series"]
    assert red["x"][0] == 100 and blue["x"][0] == 120 and blue["x"][-1] == 169 and len(blue["x"]) == len(blue["values"]) == 50
    assert plot["x"] == sorted(set(red["x"]) | set(blue["x"]))


# ====================================================================== Grid
def test_grid_overlay_types():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    img = s.active
    out = s.run_op("overlay.grid", {"type": "Lines", "spacing": 64})
    lines = [o for o in img.overlay if o["type"] == "line"]
    assert len(lines) == 8 and all(o.get("grid") for o in lines) and "4096 pixel² per point" in out["message"]
    assert lines[0]["x1"] == pytest.approx(32) and lines[0]["y2"] == img.height
    s.run_op("overlay.grid", {"type": "Combined", "area_per_point": 4096})
    assert [o["type"] for o in img.overlay] == ["line"] * 8 + ["point"] and len(img.overlay[-1]["points"]) == 16  # the previous grid is replaced
    s.run_op("overlay.grid", {"type": "Crosses", "spacing": 64, "color": "#ff0000"})
    assert len(img.overlay) == 32 and all(o["color"] == "#ff0000" for o in img.overlay)
    s.run_op("overlay.grid", {"type": "Horizontal Lines", "spacing": 100, "random_offset": True})
    ys = [o["y1"] for o in img.overlay]
    assert len(ys) in (2, 3) and 0 <= ys[0] < 100 and ys[1] - ys[0] == pytest.approx(100)  # offset is random in [0, spacing)
    s.run_op("overlay.grid", {"type": "Points", "spacing": 128})
    assert len(img.overlay) == 1 and len(img.overlay[0]["points"]) == 4
    # calibrated area per point
    img.calibration.pixel_width = img.calibration.pixel_height = 0.5
    img.calibration.unit = "um"
    out = s.run_op("overlay.grid", {"type": "Points", "area_per_point": 100})  # 100 µm² = 400 px² → 20 px spacing
    assert "spacing 20.00 px" in out["message"]


# ====================================================================== Compare Images
def test_compare_images_metrics():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    a = s.active
    s.run_op("file.sample", {"name": "Blobs"})
    b = s.active
    out = s.run_op("analyze.compare_images", {"image1": a.id, "image2": b.id})
    row = s.results.rows[-1]
    assert row["Correlation"] == pytest.approx(1.0) and row["RMSE"] == 0 and row["MAE"] == 0 and row["SSIM"] == pytest.approx(1.0)
    assert out["table"]["name"] == "Compare Images" and out["table"]["rows"][5] == ["PSNR (dB)", None]  # infinite PSNR → null in the table
    diff = s.images[out["new_images"][0]]
    assert diff.dtype == np.float32 and diff.data.max() == 0
    s.run_op("filters.gaussian", {"sigma": 1.5}, image_id=b.id)
    out = s.run_op("analyze.compare_images", {"image1": a.id, "image2": b.id, "difference_image": False})
    row = s.results.rows[-1]
    assert 0.9 < row["Correlation"] < 1 and row["RMSE"] > row["MAE"] > 0 and 20 < row["PSNR"] < 40 and 0 < row["SSIM"] < 1 and not out["new_images"]
    small = s.add_image(LumenImage(np.zeros((5, 5), np.uint8), name="tiny"))
    with pytest.raises(ValueError, match="same size"):
        s.run_op("analyze.compare_images", {"image1": a.id, "image2": small.id})


def test_compare_images_ssim_honours_selection():
    # identical top half, different bottom half: every metric – SSIM included – describes the selection, not the plane
    rng = np.random.default_rng(1)
    pa = rng.integers(0, 255, (64, 64)).astype(np.uint8)
    pb = pa.copy()
    pb[32:] = rng.integers(0, 255, (32, 64))
    s = Session()
    a = s.add_image(LumenImage(pa, name="A"))
    b = s.add_image(LumenImage(pb, name="B"))
    params = {"image1": a.id, "image2": b.id, "difference_image": False}

    def run(roi):
        s.run_op("analyze.compare_images", params, roi=roi)
        return s.results.rows[-1]

    whole = run(None)
    assert 0.3 < whole["SSIM"] < 0.7 and whole["Pixels"] == 64 * 64
    top = run({"type": "rect", "x": 0, "y": 0, "w": 64, "h": 32})
    assert top["RMSE"] == 0 and top["Correlation"] == pytest.approx(1.0) and top["SSIM"] == pytest.approx(1.0) and top["Pixels"] == 2048
    bottom = run({"type": "rect", "x": 0, "y": 32, "w": 64, "h": 32})
    assert bottom["RMSE"] > 50 and bottom["SSIM"] < 0.1
    oval = run({"type": "oval", "x": 0, "y": 0, "w": 64, "h": 32})  # non-rectangular: the SSIM map is averaged over the mask
    assert oval["SSIM"] == pytest.approx(1.0) and oval["Pixels"] < 2048
    assert run({"type": "oval", "x": 0, "y": 32, "w": 64, "h": 32})["SSIM"] < 0.1
    assert run({"type": "rect", "x": -20, "y": -10, "w": 84, "h": 42})["SSIM"] == pytest.approx(1.0)  # partly outside the image
    assert run({"type": "rect", "x": 0, "y": 0, "w": 64, "h": 64})["SSIM"] == pytest.approx(whole["SSIM"])  # whole image = no selection
    thin = run({"type": "rect", "x": 0, "y": 0, "w": 64, "h": 2})  # too thin for a 3-pixel window: NaN, no error
    assert thin["RMSE"] == 0 and np.isnan(thin["SSIM"])


# ====================================================================== Directionality
def test_compare_images_ignores_selection_of_other_image():
    """The selection the client sends belongs to the ACTIVE image: when the compared pair are two other images it is
    ignored (a 128×128 mask on 64×64 planes used to raise a raw numpy IndexError), when the active image is one of
    the pair it is honoured."""
    rng = np.random.default_rng(2)
    s = Session()
    a = s.add_image(LumenImage(rng.integers(0, 255, (64, 64)).astype(np.uint8), name="A"))
    b = s.add_image(LumenImage(rng.integers(0, 255, (64, 64)).astype(np.uint8), name="B"))
    c = s.add_image(LumenImage(rng.integers(0, 255, (128, 128)).astype(np.uint8), name="C"))  # active, different size
    params = {"image1": a.id, "image2": b.id, "difference_image": False}
    out = s.run_op("analyze.compare_images", params, roi={"type": "rect", "x": 10, "y": 10, "w": 100, "h": 100})
    assert out["results_added"] == 1 and s.results.rows[-1]["Pixels"] == 64 * 64  # foreign selection ignored
    # same-sized but unrelated active image: the selection is meaningless there too
    d = s.add_image(LumenImage(rng.integers(0, 255, (64, 64)).astype(np.uint8), name="D"))
    s.select(d.id)
    s.run_op("analyze.compare_images", params, roi={"type": "rect", "x": 10, "y": 10, "w": 20, "h": 20})
    assert s.results.rows[-1]["Pixels"] == 64 * 64
    # the active image's own selection is still honoured, whichever of the pair it is
    for active in (a, b):
        s.select(active.id)
        s.run_op("analyze.compare_images", params, roi={"type": "rect", "x": 10, "y": 10, "w": 20, "h": 20})
        assert s.results.rows[-1]["Pixels"] == 400
    s.select(c.id)
    with pytest.raises(ValueError, match="same size"):
        s.run_op("analyze.compare_images", {"image1": a.id, "image2": c.id})


def _stripes(angle_deg: float, size: int = 128, period: float = 8.0) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    ang = np.radians(angle_deg)
    # structures run along (cos, sin) in y-up coordinates → intensity varies along the perpendicular
    phase = xx * np.sin(ang) + yy * np.cos(ang)
    return (127 + 100 * np.sin(2 * np.pi * phase / period)).astype(np.uint8)


@pytest.mark.parametrize("method", ["Local gradient orientation", "Fourier components"])
def test_directionality_finds_stripe_angle(method):
    s = Session()
    for angle in (30.0, -60.0):
        s.add_image(LumenImage(_stripes(angle), name=f"stripes{angle:+.0f}"))
        out = s.run_op("analyze.directionality", {"method": method, "n_bins": 90})
        row = s.results.rows[-1]
        assert row["Direction (°)"] == pytest.approx(angle, abs=3.0), (method, angle)
        assert row["Dispersion (°)"] < 6 and row["Amount"] > 0.5 and row["Goodness"] > 0.8
        assert len(out["plot"]["x"]) == 90 and out["plot"]["series"][1]["name"] == "Gaussian fit"
        assert out["table"]["name"] == "Directionality histograms"
    # isotropic noise: low goodness / wide dispersion
    rng = np.random.default_rng(0)
    s.add_image(LumenImage(rng.integers(0, 255, (128, 128)).astype(np.uint8), name="noise"))
    s.run_op("analyze.directionality", {"method": method})
    assert s.results.rows[-1]["Dispersion (°)"] > 6 or s.results.rows[-1]["Goodness"] < 0.5


# ====================================================================== Set Measurements options (measure.py)
def test_measure_options_invert_y_nan_label():
    img = LumenImage(np.zeros((40, 60), np.uint8), name="m")
    rect = {"type": "rect", "x": 10, "y": 5, "w": 20, "h": 10}
    meas = ["area", "centroid", "center_of_mass", "bounding_rect", "feret"]
    plain = measure_roi(img, rect, meas)[0]
    inv = measure_roi(img, rect, meas, options={"invert_y": True})[0]
    assert plain["Y"] == 10 and inv["Y"] == 29 and np.isnan(inv["YM"])             # ImageJ Calibration.getY: 40 − 10 − 1; zero mass → NaN centre of mass
    assert plain["BY"] == 5 and inv["BY"] == 34 and inv["BX"] == plain["BX"] == 10  # 40 − 5 − 1 (ImageJ inverts the top edge)
    assert inv["FeretY"] == plain["FeretY"] and inv["X"] == plain["X"]            # FeretX/Y are pixel coordinates, not inverted
    img.calibration.pixel_height = 0.5
    inv = measure_roi(img, rect, ["centroid"], options={"invert_y": True})[0]
    assert inv["Y"] == pytest.approx(14.5)                                         # (height − y − 1) · pixel height: (40 − 10 − 1) · 0.5
    pt = measure_roi(img, {"type": "point", "points": [[3, 4]]}, ["area", "perimeter", "centroid"], options={"nan_empty_cells": True})[0]
    assert pt["Area"] == 0 and pt["Perim."] == 0 and pt["X"] == 3                 # ImageJ: a point has Area 0 and Perim. 0 (not empty cells)
    assert measure_roi(img, {"type": "point", "points": [[3, 4]]}, ["area", "perimeter"])[0]["Perim."] == 0
    no_label = measure_roi(img, rect, ["area"], options={"display_label": False})[0]
    assert "Label" not in no_label and no_label["Area"] == 100  # 200 px × 0.5 (pixel height set above)
    assert "Label" in measure_roi(img, rect, ["area"], options={"display_label": True})[0]
    assert set(MEASURE_OPTION_DEFAULTS) >= {"invert_y", "scientific_notation", "add_to_overlay", "nan_empty_cells", "display_label"}
    # labels (Analyze Particles) get the same treatment
    labels = np.zeros((40, 60), np.int32)
    labels[5:15, 10:30] = 1
    rows = measure_labels(img, labels, ["area", "centroid", "bounding_rect"], 0, 0, 0, options={"invert_y": True})
    assert rows[0]["Y"] == pytest.approx(14.5) and rows[0]["BY"] == pytest.approx(17)   # (40 − 10 − 1) · 0.5 and (40 − 5 − 1) · 0.5
    # options=None leaves rows untouched
    rows = [{"Y": 1.0}]
    assert apply_measure_options(rows, img, None) is rows and rows[0]["Y"] == 1.0


# ====================================================================== Calibrate / curve-fit functions
@pytest.mark.parametrize("name,params", [
    ("Exponential with Offset", [10.0, 0.4, 2.0]),
    ("Exponential Recovery", [8.0, 0.5, 1.0]),
    ("Gamma Variate", [0.5, 2.0, 1.5, 2.0]),
    ("Chapman-Richards", [10.0, 0.5, 1.5]),
    ("Rodbard (NIH Image)", [-1.0, 2.0, 5.0, 12.0]),
])
def test_new_fit_functions_recover_parameters(name, params):
    assert name in FIT_NAMES
    x = np.linspace(0, 10, 40)
    y = evaluate(name, params, x)
    res = fit(name, x, y)
    assert res["r2"] > 0.999
    assert np.allclose(res["params"], params, rtol=0.05, atol=0.05), res["params"]


def test_fit_aliases_imagej_spelling():
    x = np.linspace(0, 10, 30)
    y = evaluate("Exponential with Offset", [10.0, 0.4, 2.0], x)
    assert fit("Exponential with offset", x, y)["function"] == "Exponential with Offset"
    assert fit("Inverse Rodbard", x, 1 + x)["function"] == "Rodbard (NIH Image)"
    assert evaluate("gamma variate", [0.0, 1.0, 1.0, 1.0], [0.0, 1.0])[1] == pytest.approx(np.exp(-1))
    with pytest.raises(ValueError):
        fit("Nope", x, y)


# ====================================================================== Gels: select lane / plot lanes / label peaks
def _gel_session():
    s = Session()
    gel = np.full((120, 100), 220, np.uint8)
    for x0 in (10, 40, 70):
        for yb, dark in ((30, 60), (70, 120)):
            gel[yb - 3:yb + 3, x0:x0 + 20] = dark
    img = s.add_image(LumenImage(gel, name="gel"))
    return s, img


def test_gel_select_lane_numbering():
    s, img = _gel_session()
    s.settings["roi_color"] = "#00ff00"
    for n, x0 in enumerate((10, 40, 70), 1):
        out = s.run_op("analyze.gel_select_lane", {}, roi={"type": "rect", "x": x0, "y": 5, "w": 20, "h": 110})
        assert out["message"].startswith(f"Lane {n}")
    lanes = s.roi_manager.to_list()
    assert [r["name"] for r in lanes] == ["Lane 1", "Lane 2", "Lane 3"] and [r["gel_lane"] for r in lanes] == [1, 2, 3]
    assert all(r["image_id"] == img.id and r["color"] == "#00ff00" for r in lanes)
    with pytest.raises(ValueError):
        s.run_op("analyze.gel_select_lane", {}, roi={"type": "oval", "x": 0, "y": 0, "w": 5, "h": 5})


def test_gel_plot_lanes_and_label_peaks():
    s, img = _gel_session()
    for x0 in (10, 40, 70):
        s.run_op("analyze.gel_select_lane", {}, roi={"type": "rect", "x": x0, "y": 5, "w": 20, "h": 110})
    out = s.run_op("analyze.gel_plot_lanes", {})
    plot = out["plot"]
    assert [sr["name"] for sr in plot["series"]] == ["Lane 1", "Lane 2", "Lane 3"] and len(plot["x"]) == 110
    prof = np.array(plot["series"][0]["values"])
    assert prof[25] > prof[10] and prof[25] > prof[65] > prof[10]  # the darker band (gray 60 at row 30) → higher inverted density
    plot_img = s.images[out["new_images"][0]]
    assert plot_img.dtype == np.uint8 and plot_img.metadata["gel_plot"] and plot_img.height == 3 * 130 + 10
    assert s.settings["gel_profiles"]["image_id"] == img.id
    out = s.run_op("analyze.gel_label_peaks", {"smooth": 0}, image_id=img.id)
    assert out["results_added"] == 6
    rows = s.results.rows[-6:]
    assert [r["Lane"] for r in rows] == [1, 1, 2, 2, 3, 3] and [r["Peak"] for r in rows] == [1, 2] * 3
    assert rows[0]["Position"] == pytest.approx(25, abs=1) and rows[1]["Position"] == pytest.approx(65, abs=1)
    # Percent is of all bands in all lanes (ImageJ): three identical lanes → each lane holds a third
    assert rows[0]["Area"] > rows[1]["Area"] and rows[0]["Percent"] + rows[1]["Percent"] == pytest.approx(100 / 3)
    assert sum(r["Percent"] for r in rows) == pytest.approx(100)
    texts = [o for o in img.overlay if o.get("gel_peak")]
    assert len(texts) == 6 and texts[0]["type"] == "text" and texts[0]["text"] == f"{rows[0]['Area']:.0f}"
    assert texts[0]["x"] == pytest.approx(10 + 20 + 2) and abs(texts[0]["y"] - (5 + 25)) < 10
    s.run_op("analyze.gel_label_peaks", {"smooth": 0, "label": "Percent"}, image_id=img.id)
    assert len([o for o in img.overlay if o.get("gel_peak")]) == 6 and img.overlay[-1]["text"].endswith("%")  # relabelled, not duplicated
    # without lanes a plain rectangle selection is used, and nothing at all is an error
    s.roi_manager.clear()
    out = s.run_op("analyze.gel_plot_lanes", {"make_image": False}, image_id=img.id, roi={"type": "rect", "x": 40, "y": 5, "w": 20, "h": 110})
    assert len(out["plot"]["series"]) == 1 and not out["new_images"]
    with pytest.raises(ValueError):
        s.run_op("analyze.gel_plot_lanes", {}, image_id=img.id)


def test_gel_percent_is_of_all_bands_in_all_lanes():
    # ImageJ Label Peaks: Percent = band area / sum of all measured peaks from all lanes, so lanes are comparable –
    # the single faint band of lane 2 is a few percent, not 100 %
    gel = np.full((120, 100), 220, np.uint8)
    gel[27:33, 10:30] = 60
    gel[67:73, 10:30] = 120
    gel[27:33, 40:60] = 200
    s = Session()
    img = s.add_image(LumenImage(gel, name="gel"))
    for x0 in (10, 40):
        s.run_op("analyze.gel_select_lane", {}, roi={"type": "rect", "x": x0, "y": 5, "w": 20, "h": 110})
    s.run_op("analyze.gel_label_peaks", {"smooth": 0, "label": "Percent"}, image_id=img.id)
    rows = s.results.rows
    assert [r["Lane"] for r in rows] == [1, 1, 2]
    areas = np.array([r["Area"] for r in rows])
    expect = list(100 * areas / areas.sum())
    assert [r["Percent"] for r in rows] == pytest.approx(expect) and rows[2]["Percent"] < 10
    assert [r["Percent"] for r in rows] == pytest.approx([57.1, 35.7, 7.1], abs=0.1)
    assert [o["text"] for o in img.overlay if o.get("gel_peak")] == [f"{r['Percent']:.1f}%" for r in rows]
    s.results.clear()
    s.run_op("analyze.gel_lanes", {"smooth": 0}, image_id=img.id)
    assert [r["Percent"] for r in s.results.rows] == pytest.approx(expect)
    # the per-lane normalisation is opt-in
    lane1 = areas[0] + areas[1]
    s.results.clear()
    s.run_op("analyze.gel_label_peaks", {"smooth": 0, "percent_per_lane": True}, image_id=img.id)
    assert [r["Percent"] for r in s.results.rows] == pytest.approx([100 * areas[0] / lane1, 100 * areas[1] / lane1, 100.0])
    s.results.clear()
    s.run_op("analyze.gel_lanes", {"smooth": 0, "percent_per_lane": True}, image_id=img.id)
    assert [r["Percent"] for r in s.results.rows] == pytest.approx([100 * areas[0] / lane1, 100 * areas[1] / lane1, 100.0])


def test_gel_lanes_beyond_the_image_edge_keep_labels_on_the_bands():
    """A lane dragged past the top / left edge is clipped to the image (ImageJ clips rectangle ROIs the same way):
    profile index 0 is the first in-image row, and the band labels sit next to the bands – not |y0| px above them."""
    from lumen.core.ops.gel import clip_lane
    assert clip_lane({"type": "rect", "x": 10, "y": -10, "w": 20, "h": 120, "name": "L"}, (120, 100)) == {"type": "rect", "x": 10.0, "y": 0.0, "w": 20.0, "h": 110.0, "name": "L"}
    assert clip_lane({"type": "rect", "x": 90, "y": 5, "w": 20, "h": 110}, (120, 100))["w"] == 10.0
    assert clip_lane({"type": "rect", "x": -30, "y": 5, "w": 20, "h": 110}, (120, 100))["w"] == 0.0  # entirely outside
    s, img = _gel_session()
    s.run_op("analyze.gel_select_lane", {}, roi={"type": "rect", "x": 10, "y": 5, "w": 20, "h": 110})     # inside
    s.run_op("analyze.gel_select_lane", {}, roi={"type": "rect", "x": 40, "y": -10, "w": 20, "h": 120})   # hangs over the top edge
    s.run_op("analyze.gel_select_lane", {}, roi={"type": "rect", "x": 75, "y": 5, "w": 40, "h": 110})     # past the right edge
    assert s.roi_manager.to_list()[1]["y"] == -10  # the ROI itself is left alone (lane outlines / size checks)
    out = s.run_op("analyze.gel_label_peaks", {"smooth": 0}, image_id=img.id)
    assert out["results_added"] == 6
    rows = s.results.rows[-6:]
    assert [r["Lane"] for r in rows] == [1, 1, 2, 2, 3, 3]
    # Position is relative to the first in-image row of the lane (= the Plot Lanes x axis): 24 for y=5, 29 for y=-10
    assert rows[0]["Position"] == pytest.approx(24, abs=1) and rows[2]["Position"] == pytest.approx(29, abs=1)
    texts = [o for o in img.overlay if o.get("gel_peak")]
    assert len(texts) == 6
    fs = int(s.settings.get("font_size", 12))
    assert texts[0]["y"] == texts[2]["y"] == pytest.approx(29 - fs / 2, abs=1) and texts[1]["y"] == texts[3]["y"]  # same band → same row
    assert texts[2]["x"] == pytest.approx(40 + 20 + 2) and texts[4]["x"] == pytest.approx(100 + 2)  # label column stays on the canvas
    assert texts[4]["y"] == pytest.approx(texts[0]["y"])
    # Plot Lanes: the stored lanes / profiles agree (110 in-image rows for the overhanging lane too)
    out = s.run_op("analyze.gel_plot_lanes", {"make_image": False}, image_id=img.id)
    assert [len(sr["values"]) for sr in out["plot"]["series"]] == [110, 110, 110] and len(out["plot"]["x"]) == 110
    stored = s.settings["gel_profiles"]
    assert stored["lanes"][1]["y"] == 0.0 and stored["lanes"][1]["h"] == 110.0 and stored["lanes"][2]["w"] == 25.0
    # horizontal lanes hanging over the left edge
    s2 = Session()
    img2 = s2.add_image(LumenImage(np.ascontiguousarray(img.current_plane().T), name="gel-h"))
    s2.run_op("analyze.gel_select_lane", {}, roi={"type": "rect", "x": 5, "y": 10, "w": 110, "h": 20})
    s2.run_op("analyze.gel_select_lane", {}, roi={"type": "rect", "x": -10, "y": 40, "w": 120, "h": 20})
    s2.run_op("analyze.gel_label_peaks", {"smooth": 0, "orientation": "Horizontal"}, image_id=img2.id)
    texts = [o for o in img2.overlay if o.get("gel_peak")]
    assert len(texts) == 4 and texts[0]["x"] == texts[2]["x"] == pytest.approx(29 - fs / 2, abs=1)
    assert texts[2]["y"] == pytest.approx(40 + 20 + 2)
    # a lane entirely outside the image is still an error
    s2.roi_manager.clear()
    with pytest.raises(ValueError, match="outside"):
        s2.run_op("analyze.gel_plot_lanes", {"make_image": False}, image_id=img2.id, roi={"type": "rect", "x": -50, "y": 10, "w": 20, "h": 20})
