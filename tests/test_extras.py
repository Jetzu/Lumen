"""Tests for the Adjust / Analyze / Overlay / Stacks extras (extras.py, stacks_extra.py, gel.py, fitting.py)."""
import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.session import Session


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


def _stack(session, nz=4, h=32, w=40, seed=0):
    rng = np.random.default_rng(seed)
    vol = rng.integers(0, 200, size=(nz, h, w), dtype=np.uint16).astype(np.uint16)
    for z in range(nz):
        vol[z, 8:16, 8:16] = 1000 + 100 * z
    return session.add_image(LumenImage(vol, name="stack.tif", axes="ZYX"))


# ---------------------------------------------------------------- Image ▸ Adjust
def test_color_threshold_mask_and_selection():
    s = Session()
    s.run_op("file.sample", {"name": "Astronaut (RGB)"})
    rgb = s.active
    out = s.run_op("adjust.color_threshold", {"space": "HSB", "c3_min": 128, "c3_max": 255, "output": "Mask"})
    mask = s.images[out["new_images"][0]]
    plane = mask.current_plane()
    assert plane.shape == (rgb.height, rgb.width)
    assert set(np.unique(plane)).issubset({0, 255})
    frac = (plane > 0).mean()
    assert 0.05 < frac < 0.95
    s.select(rgb.id)
    out = s.run_op("adjust.color_threshold", {"space": "RGB", "c1_min": 100, "output": "Selection"}, image_id=rgb.id)
    assert out["roi"] is not None and not out["new_images"]


def test_window_level(session):
    img = session.active
    session.run_op("adjust.window_level", {"window": 100, "level": 100})
    assert img.display.channels[0].min == pytest.approx(50) and img.display.channels[0].max == pytest.approx(150)


def test_coordinates_origin(session):
    session.run_op("adjust.coordinates", {"origin_x": 5, "origin_y": 7})
    origin = session.active.metadata["origin"]
    assert (origin["x"], origin["y"], origin["z"]) == (5, 7, 0)


# ---------------------------------------------------------------- Analyze
def test_curve_fit_exponential():
    s = Session()
    x = np.arange(0, 6, dtype=float)
    y = 3.0 * np.exp(0.5 * x)
    out = s.run_op("analyze.curve_fit", {"source": "Values", "x_column": " ".join(f"{v:g}" for v in x), "y_column": " ".join(f"{v:.6f}" for v in y), "function": "Exponential"})
    row = s.results.rows[-1]
    assert row["a"] == pytest.approx(3.0, rel=1e-3) and row["b"] == pytest.approx(0.5, rel=1e-3)
    assert row["R²"] > 0.999
    assert out["plot"]["series"][0]["name"] == "fit"


def test_curve_fit_from_results_columns():
    s = Session()
    s.results.append([{"X": float(i), "Y": 2.0 * i + 1.0} for i in range(8)])
    s.run_op("analyze.curve_fit", {"source": "Results columns", "x_column": "X", "y_column": "Y", "function": "Straight Line"})
    row = s.results.rows[-1]
    assert row["a"] == pytest.approx(1.0, abs=1e-6) and row["b"] == pytest.approx(2.0, abs=1e-6)


def test_fractal_box_count_of_square():
    s = Session()
    a = np.zeros((256, 256), np.uint8)
    a[32:224, 32:224] = 255
    s.add_image(LumenImage(a, name="square.tif"))
    out = s.run_op("analyze.fractal_box_count", {"box_sizes": "2,3,4,6,8,12,16,32,64"})
    row = s.results.rows[-1]
    assert abs(row["D"] - 2.0) < 0.1
    assert row["R²"] > 0.99 and out["plot"]


def test_angle_of_L_polyline(session):
    roi = {"type": "polyline", "points": [[0, 0], [0, 10], [10, 10]]}
    session.run_op("analyze.angle", {}, roi=roi)
    assert session.results.rows[-1]["Angle"] == pytest.approx(90.0)
    session.run_op("analyze.angle", {}, roi={"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0})
    assert session.results.rows[-1]["Angle"] == pytest.approx(0.0, abs=1e-6)


def test_surface_plot_makes_rgb(session):
    out = session.run_op("analyze.surface_plot", {"width": 200, "height": 150, "max_points": 32})
    img = session.images[out["new_images"][0]]
    assert img.rgb and img.nc == 3 and img.width == 200 and img.height == 150


def test_stack_histogram(session):
    img = _stack(session)
    out = session.run_op("analyze.stack_histogram", {"bins": 64})
    assert out["histogram"]["stats"]["count"] == img.nz * img.height * img.width
    assert session.results.rows[-1]["N"] == img.nz * img.height * img.width


def test_multi_measure_row_modes(session):
    img = _stack(session)
    session.roi_manager.add({"type": "rect", "x": 8, "y": 8, "w": 8, "h": 8}, image_id=img.id)
    session.roi_manager.add({"type": "rect", "x": 20, "y": 20, "w": 10, "h": 6}, image_id=img.id)
    out = session.run_op("analyze.multi_measure", {"axis": "Z", "one_row_per_slice": True})
    assert out["results_added"] == img.nz
    row = session.results.rows[0]
    assert "Mean1" in row and "Mean2" in row and row["Mean1"] == pytest.approx(1000)
    assert session.results.rows[-1]["Mean1"] == pytest.approx(1300)
    out = session.run_op("analyze.multi_measure", {"axis": "Z", "one_row_per_slice": False})
    assert out["results_added"] == 2 * img.nz and len(session.results.rows) == 2 * img.nz


def test_calibrate_and_measure_calibrated(session):
    img = session.active
    session.run_op("analyze.calibrate", {"function": "Straight Line", "pixel_values": "0 255", "standards": "0 1", "unit": "OD"})
    assert "intensity_calibration" in img.metadata
    session.run_op("analyze.measure_calibrated", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 50, "h": 50})
    row = session.results.rows[-1]
    assert 0.0 <= row["Mean"] <= 1.0
    out = session.run_op("analyze.calibrated_values", {})
    assert session.images[out["new_images"][0]].data.dtype == np.float32


def test_label_rois_adds_overlay(session):
    img = session.active
    session.roi_manager.add({"type": "rect", "x": 8, "y": 8, "w": 8, "h": 8}, image_id=img.id)
    session.roi_manager.add({"type": "oval", "x": 40, "y": 40, "w": 20, "h": 12}, image_id=img.id)
    session.run_op("analyze.label", {"what": "Index"})
    assert sum(1 for it in img.overlay if it.get("type") == "text") == 2


def test_gel_lanes_two_bands():
    s = Session()
    gel = np.full((100, 30), 200, np.uint8)
    gel[30:35, :] = 50
    gel[60:65, :] = 50
    s.add_image(LumenImage(gel, name="gel.tif"))
    out = s.run_op("analyze.gel_lanes", {"orientation": "Vertical", "dark_bands": True}, roi={"type": "rect", "x": 5, "y": 0, "w": 20, "h": 100})
    assert out["results_added"] == 2
    rows = s.results.rows[-2:]
    assert [r["Peak"] for r in rows] == [1, 2]
    assert abs(rows[0]["Position"] - 32) <= 2 and abs(rows[1]["Position"] - 62) <= 2
    assert rows[0]["Percent"] + rows[1]["Percent"] == pytest.approx(100.0)
    assert len(out["plot"]["series"]) == 1


# ---------------------------------------------------------------- Image ▸ Overlay
def test_overlays_and_burn_in(session):
    img = session.active
    session.run_op("overlay.text", {"text": "hello", "x": 10, "y": 10, "font_size": 20})
    session.run_op("overlay.arrow", {"color": "#ff0000"}, roi={"type": "line", "x1": 10, "y1": 10, "x2": 80, "y2": 60})
    session.run_op("overlay.calibration_bar", {"position": "Upper Right"})
    session.run_op("overlay.time_stamp", {"format": "t = {time:.1f} {unit}"})
    types = [it["type"] for it in img.overlay]
    assert types == ["text", "arrow", "calibration_bar", "time_stamp"]
    session.run_op("overlay.list", {})
    out = session.run_op("overlay.burn_in", {})
    burned = session.images[out["new_images"][0]]
    assert burned.rgb and burned.nc == 3 and burned.overlay == []
    gray = np.repeat(img.current_plane()[None], 3, axis=0)
    changed = int((burned.data[0, 0] != gray).any(axis=0).sum())
    assert changed > 100
    # overlay → ROI manager copies only ROI-shaped items (text / arrows / bars stay)
    session.select(img.id)
    with pytest.raises(ValueError):
        session.run_op("overlay.to_roi_manager", {}, image_id=img.id)
    session.run_op("overlay.add", {}, image_id=img.id, roi={"type": "rect", "x": 5, "y": 5, "w": 30, "h": 20})
    n0 = len(session.roi_manager.to_list())
    session.run_op("overlay.to_roi_manager", {"remove": True}, image_id=img.id)
    assert len(session.roi_manager.to_list()) == n0 + 1
    assert [it["type"] for it in img.overlay] == ["text", "arrow", "calibration_bar", "time_stamp"]


# ---------------------------------------------------------------- Image ▸ Stacks
def test_project3d_small_volume():
    s = Session()
    zz, yy, xx = np.mgrid[0:8, 0:24, 0:24]
    vol = (((zz - 4) ** 2 / 4.0 + (yy - 12) ** 2 / 36.0 + (xx - 12) ** 2 / 36.0) < 1).astype(np.uint8) * 200
    s.add_image(LumenImage(vol, name="ball.tif", axes="ZYX"))
    out = s.run_op("stacks.project3d", {"method": "Brightest Point", "axis": "Y-Axis", "angle_step": 90, "total_rotation": 360})
    proj = s.images[out["new_images"][0]]
    assert proj.nt == 4 and proj.nz == 1
    assert proj.data.max() > 0


def test_orthogonal_views(session):
    img = _stack(session)
    out = session.run_op("stacks.orthogonal_views", {"x": 10, "y": 12, "scale_z": False})
    xz, yz = (session.images[i] for i in out["new_images"])
    assert xz.name.startswith("XZ") and yz.name.startswith("YZ")
    assert (xz.height, xz.width) == (img.nz, img.width)
    assert (yz.height, yz.width) == (img.height, img.nz)
    assert out["roi"]["type"] == "point"


def test_reduce_stack(session):
    img = _stack(session, nz=6)
    session.run_op("stacks.reduce", {"factor": 2, "axis": "Z"})
    assert img.nz == 3


def test_straighten_curved_line():
    s = Session()
    a = np.zeros((64, 64), np.uint8)
    a[19:22, :] = 255
    s.add_image(LumenImage(a, name="line.tif"))
    out = s.run_op("stacks.straighten", {"line_width": 10}, roi={"type": "polyline", "points": [[5, 20.5], [30, 20.5], [55, 20.5]]})
    st = s.images[out["new_images"][0]]
    assert st.height == 10 and 44 <= st.width <= 58
    assert st.current_plane()[5].mean() > 200
    assert st.current_plane()[0].mean() < 30


def test_combine_and_insert(session):
    a = session.add_image(LumenImage(np.full((2, 10, 12), 10, np.uint8), name="a.tif", axes="ZYX"))
    b = session.add_image(LumenImage(np.full((2, 10, 8), 20, np.uint8), name="b.tif", axes="ZYX"))
    out = session.run_op("stacks.combine", {"image1": a.name, "image2": b.name})
    comb = session.images[out["new_images"][0]]
    assert comb.width == 20 and comb.height == 10 and comb.nz == 2
    session.select(a.id)
    session.run_op("stacks.insert", {"source": b.name, "x": 2, "y": 0}, image_id=a.id)
    assert a.current_plane()[0, 2] == 20 and a.current_plane()[0, 0] == 10


def test_label_slices_overlay(session):
    img = _stack(session)
    session.run_op("stacks.label_slices", {"format": "{frame}"})
    assert any(it.get("type") in ("time_stamp", "text") for it in img.overlay)


# ---------------------------------------------------------------- regression tests (review round 2)
def test_window_level_rgb_sets_all_channels():
    # ImageJ's Window/Level on an RGB image sets the same range on all three components (not only red)
    s = Session()
    s.run_op("file.sample", {"name": "Astronaut (RGB)"})
    s.run_op("adjust.window_level", {"window": 100, "level": 100})
    assert [(c.min, c.max) for c in s.active.display.channels] == [(50.0, 150.0)] * 3
    # non-RGB multichannel images still get only the current channel
    s.add_image(LumenImage(np.zeros((1, 1, 2, 8, 8), np.uint8), name="two.tif"))
    s.run_op("adjust.window_level", {"window": 100, "level": 100})
    assert s.active.display.channels[0].max == pytest.approx(150) and s.active.display.channels[1].max != pytest.approx(150)


def test_lab_components_match_imagej_color_thresholder():
    from lumen.core.ops.extras import color_components
    # ImageJ ColorThresholder.getLab: L = (int)(L*2.55), a/b = floor(1.0625*a + 128 + 0.5)
    assert tuple(color_components(np.array([[[255, 0, 0]]], np.uint8), "Lab")[:, 0, 0]) == (135, 213, 199)
    assert tuple(color_components(np.array([[[0, 255, 0]]], np.uint8), "Lab")[:, 0, 0]) == (223, 36, 216)
    assert tuple(color_components(np.array([[[0, 0, 255]]], np.uint8), "Lab")[:, 0, 0]) == (82, 212, 13)


def test_measure_calibrated_measures_calibrated_values():
    s = Session()
    a = np.zeros((20, 20), np.uint8)
    a[:, 10:] = 200
    s.add_image(LumenImage(a, name="halves.tif"))
    s.run_op("analyze.set_measurements", {"measurements": ["area", "mean", "min", "max", "std", "integrated_density", "mode"]})
    # decreasing density calibration (the usual OD case): statistics OF the calibrated values, like ImageJ's cTable
    s.run_op("analyze.calibrate", {"function": "Straight Line", "pixel_values": "0 255", "standards": "1 0", "unit": "OD"})
    s.run_op("analyze.measure_calibrated", {})
    row = s.results.rows[-1]
    assert row["Min"] == pytest.approx(1 - 200 / 255, abs=1e-6) and row["Max"] == pytest.approx(1.0)
    assert row["Min"] < row["Max"]
    assert row["Mean"] == pytest.approx(0.60784, abs=1e-4)
    assert row["StdDev"] == pytest.approx(0.39265, abs=1e-4)
    assert row["IntDen"] == pytest.approx(row["Area"] * row["Mean"]) and row["IntDen"] == pytest.approx(243.137, abs=1e-2)
    assert row["RawIntDen"] == 40000  # stays raw (ImageJ: pixelCount * umean)
    assert row["Mode"] == pytest.approx(1.0)  # f(raw mode) like ImageJ's dmode = cTable[mode]
    assert row["Value unit"] == "OD"
    # non-linear function: Mean must be mean(f(v)), not f(mean(v))
    s.run_op("analyze.calibrate", {"function": "Uncalibrated OD"})
    s.run_op("analyze.measure_calibrated", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 20, "h": 20})
    row = s.results.rows[-1]
    expected = np.log10(255.0 / np.maximum(255.0 - a.astype(float), 0.5))
    assert row["Mean"] == pytest.approx(expected.mean()) and row["StdDev"] == pytest.approx(expected.std(ddof=1))
    # the threshold limit applies to the raw pixel values
    s.measure_options["limit_to_threshold"] = True
    s.thresholds[s.active.id] = {"lower": 150, "upper": 255}
    s.run_op("analyze.measure_calibrated", {})
    row = s.results.rows[-1]
    assert row["Area"] == 200 and row["Mean"] == pytest.approx(float(expected[0, 15]))
    s.measure_options["limit_to_threshold"] = False
    # point selections report the calibrated Value
    s.run_op("analyze.measure_calibrated", {}, roi={"type": "point", "points": [[15, 3]]})
    assert s.results.rows[-1]["Value"] == pytest.approx(float(expected[3, 15]))


def test_calibrated_values_and_burn_in_drop_the_intensity_calibration(session):
    img = session.active
    session.run_op("analyze.calibrate", {"function": "Straight Line", "pixel_values": "0 255", "standards": "0 1", "unit": "OD"})
    out = session.run_op("analyze.calibrated_values", {})
    new = session.images[out["new_images"][0]]
    assert "intensity_calibration" not in new.metadata and new.calibration.value_unit == "OD"
    assert "intensity_calibration" in img.metadata  # the source keeps its calibration
    session.run_op("analyze.measure_calibrated", {}, image_id=new.id)
    assert session.results.rows[-1]["Mean"] == pytest.approx(float(new.data.mean()), rel=1e-5)
    with pytest.raises(ValueError):  # the function is not carried over, so it cannot be applied twice
        session.run_op("analyze.calibrated_values", {}, image_id=new.id)
    session.run_op("overlay.calibration_bar", {}, image_id=new.id)
    bar = new.overlay[-1]
    assert bar["min"] == pytest.approx(new.display.channels[0].min) and bar["max"] == pytest.approx(new.display.channels[0].max)
    session.run_op("overlay.text", {"text": "x"}, image_id=img.id)
    out = session.run_op("overlay.burn_in", {}, image_id=img.id)
    burned = session.images[out["new_images"][0]]
    assert "intensity_calibration" not in burned.metadata and burned.calibration.value_unit == ""


def test_overlay_to_roi_manager_assigns_fresh_ids(session):
    img = session.active
    session.roi_manager.add({"type": "rect", "x": 8, "y": 8, "w": 8, "h": 8}, image_id=img.id)
    session.run_op("overlay.from_manager", {})
    session.run_op("overlay.to_roi_manager", {})
    rois = session.roi_manager.rois
    assert len(rois) == 2 and len({r["id"] for r in rois}) == 2
    session.roi_manager.remove([rois[0]["id"]])
    assert len(session.roi_manager.rois) == 1
    session.run_op("overlay.to_roi_manager", {})  # running it again adds another distinct entry
    assert len({r["id"] for r in session.roi_manager.rois}) == 2


def test_calibrate_rejects_unsupported_bit_depths(session):
    s16 = _stack(session)
    with pytest.raises(ValueError, match="8-bit"):
        session.run_op("analyze.calibrate", {"function": "Uncalibrated OD"}, image_id=s16.id)
    assert "intensity_calibration" not in s16.metadata
    f32 = session.add_image(LumenImage(np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10), name="f.tif"))
    with pytest.raises(ValueError, match="32-bit"):
        session.run_op("analyze.calibrate", {"function": "Straight Line", "pixel_values": "0 1", "standards": "0 1"}, image_id=f32.id)
    session.run_op("analyze.calibrate", {"function": "None"}, image_id=f32.id)  # removing a calibration stays allowed
    rgb = Session()
    rgb.run_op("file.sample", {"name": "Astronaut (RGB)"})
    with pytest.raises(ValueError):
        rgb.run_op("analyze.calibrate", {"function": "Uncalibrated OD"})


def test_curve_fit_log_reports_positive_b_for_positive_x():
    from lumen.core.fitting import fit
    x = np.linspace(1, 100, 20)
    yn = 3.0 * np.log(0.2 * x) * (1 + 0.01 * np.random.default_rng(0).standard_normal(20))  # this seed drove the optimiser to b < 0
    r = fit("Log", x, yn)
    a, b = r["params"]
    assert b > 0 and b == pytest.approx(0.2, rel=2e-2) and a == pytest.approx(3.0, rel=2e-2) and r["r2"] > 0.999
    assert fit("Log", -x, 3.0 * np.log(-0.2 * -x))["params"][1] < 0  # all-negative x: b·x > 0 needs b < 0
    s = Session()
    s.run_op("analyze.curve_fit", {"source": "Values", "x_column": " ".join(map(str, x)), "y_column": " ".join(f"{v:.5f}" for v in yn), "function": "Log"})
    assert s.results.rows[-1]["b"] > 0


def test_label_measurement_knows_every_feret_and_stack_column(session):
    img = session.active
    session.roi_manager.add({"type": "rect", "x": 8, "y": 8, "w": 20, "h": 10}, image_id=img.id)
    expected = {"MinFeret": "10.00", "FeretX": "8.00", "FeretY": "8.00", "Ch": "1", "Frame": "1", "Slice": "1"}
    for col, text in expected.items():
        session.run_op("analyze.label", {"what": "Measurement", "column": col})
        assert img.overlay[-1]["text"] == text, col
    session.run_op("analyze.label", {"what": "Measurement", "column": "FeretAngle"})
    assert img.overlay[-1]["text"] != "?"
    from lumen.core.ij_compat import IJ
    assert IJ(session).macro('makeRectangle(10,10,30,20); print(getValue("MinFeret"));').strip() == "20"


def test_curve_fit_and_calibrate_drop_nan_from_plot():
    s = Session()
    out = s.run_op("analyze.curve_fit", {"source": "Values", "x_column": "1 2 3 nan", "y_column": "1 2 3 4", "function": "Straight Line"})
    assert s.results.rows[-1]["N"] == 3 and s.results.rows[-1]["b"] == pytest.approx(1.0)
    p = out["plot"]
    assert np.isfinite(p["x"]).all() and all(np.isfinite(ser["values"]).all() for ser in p["series"])
    assert p["points"]["x"] == [1.0, 2.0, 3.0]
    s.run_op("file.sample", {"name": "Blobs"})
    out = s.run_op("analyze.calibrate", {"function": "Straight Line", "pixel_values": "0 128 nan 255", "standards": "0 0.5 0.7 1"})
    assert np.isfinite(out["plot"]["x"]).all() and len(out["plot"]["points"]["x"]) == 3


def test_angle_uses_the_pixel_calibration(session):
    from lumen.core.image import Calibration
    session.active.calibration = Calibration(pixel_width=0.5, pixel_height=1.0, unit="um")
    session.run_op("analyze.angle", {}, roi={"type": "line", "x1": 0, "y1": 10, "x2": 10, "y2": 0})
    assert session.results.rows[-1]["Angle"] == pytest.approx(63.4349, abs=1e-3)  # ImageJ Roi.getFloatAngle: atan2(dy * pixelHeight, dx * pixelWidth)
    session.run_op("analyze.angle", {}, roi={"type": "polyline", "points": [[0, 0], [10, 10], [20, 0]]})
    assert session.results.rows[-1]["Angle"] == pytest.approx(53.1301, abs=1e-3)  # = |180 - |a1 - a2|| of the two calibrated arm angles
    session.active.calibration = Calibration()
    session.run_op("analyze.angle", {}, roi={"type": "line", "x1": 0, "y1": 10, "x2": 10, "y2": 0})
    assert session.results.rows[-1]["Angle"] == pytest.approx(45.0)


def test_time_stamp_start_time_and_interval():
    from lumen.core.overlay_render import format_time_stamp, time_stamp_text
    s = Session()
    mov = s.add_image(LumenImage(np.zeros((6, 16, 16), np.uint8), name="t", axes="TYX"))
    s.run_op("overlay.time_stamp", {"format": "{time:.1f} {unit}", "start_time": 10, "interval": 2, "start_frame": 1})
    item = [o for o in mov.overlay if o["type"] == "time_stamp"][0]
    assert (item["start_time"], item["interval"], item["start_frame"]) == (10.0, 2.0, 1)
    assert [time_stamp_text(item, mov, t, 0) for t in range(4)] == ["8.0 sec", "10.0 sec", "12.0 sec", "14.0 sec"]  # time = start_time + (t - start_frame) * interval
    mov.calibration.frame_interval = 3.0
    mov.calibration.time_unit = "ms"
    s.run_op("overlay.time_stamp", {"format": "{time:.1f} {unit}"})  # interval 0 → the calibration's frame interval
    item = [o for o in mov.overlay if o["type"] == "time_stamp"][0]
    assert [time_stamp_text(item, mov, t, 0) for t in range(3)] == ["0.0 ms", "3.0 ms", "6.0 ms"]
    assert format_time_stamp("{time} {unit} {label}", mov, 2, 0) == "6.0 ms 3"  # defaults: start_time 0, frame interval, 1-based {label}
    legacy = LumenImage(np.zeros((3, 8, 8), np.uint8), axes="TYX")
    assert time_stamp_text({"type": "time_stamp", "format": "{time:.0f} {unit}"}, legacy, 2, 0) == "2 frame"  # items saved before start_time / interval existed


# ---------------------------------------------------------------- Curve fitting parity (ImageJ CurveFitter)
def test_custom_equation_recovers_parameters_within_2_percent():
    from lumen.core.fitting import evaluate, fit
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 50)
    y = (3.0 * np.exp(-0.4 * x) + 1.5) * (1 + 0.005 * rng.standard_normal(x.size))
    r = fit("Custom", x, y, equation="y = a*exp(-b*x) + c", restarts=2)
    assert r["function"] == "Custom" and r["names"] == ["a", "b", "c"] and r["equation"] == "y = a*exp(-b*x) + c"
    for got, want in zip(r["params"], (3.0, 0.4, 1.5)):
        assert got == pytest.approx(want, rel=0.02)
    assert r["r2"] > 0.999 and r["rms"] < 0.05 and r["sd"] >= r["rms"]
    # through the op: a typed equation overrides Function, initial values are honoured, residuals are reported
    s = Session()
    out = s.run_op("analyze.curve_fit", {"source": "Values", "x_column": " ".join(f"{v:g}" for v in x), "y_column": " ".join(f"{v:.6f}" for v in y),
                                         "custom_equation": "y = a*exp(-b*x) + c", "initial_params": "2 0.3 1", "show_residuals": True})
    row = s.results.rows[-1]
    assert row["Label"].startswith("Custom") and row["Formula"] == "y = a*exp(-b*x) + c"
    assert row["a"] == pytest.approx(3.0, rel=0.02) and row["b"] == pytest.approx(0.4, rel=0.02) and row["c"] == pytest.approx(1.5, rel=0.02)
    assert row["RMS"] < 0.05 and row["N"] == x.size
    resid = out["plot"]["residuals"]["values"]
    assert len(resid) == x.size and abs(float(np.mean(resid))) < 0.05
    assert s.tables["Residuals"]["columns"] == ["X", "Y", "Fit", "Residual"] and len(s.tables["Residuals"]["rows"]) == x.size
    assert np.isfinite(out["plot"]["series"][0]["values"]).all()
    # ^ and pow(), pi and the fitted equation can be evaluated afterwards
    r2 = fit("Custom", x, 2 * x ** 2 + 1, equation="a*x^2 + b")
    assert r2["params"][0] == pytest.approx(2.0, rel=1e-3) and r2["params"][1] == pytest.approx(1.0, abs=1e-3)
    assert evaluate("Custom", r2["params"], [3.0], equation="a*x^2 + b")[0] == pytest.approx(19.0, rel=1e-3)
    r3 = fit("Custom", x, 2 + 0.5 * np.sin(1.3 * x + 0.4), equation="a + b*sin(c*x + d)", restarts=2)
    assert r3["r2"] > 0.999 and r3["params"][0] == pytest.approx(2.0, rel=0.02) and abs(r3["params"][1]) == pytest.approx(0.5, rel=0.02)  # phase/sign are degenerate
    assert fit("Custom", x, 3 * x + 1, equation="a*x + b*pi/pi")["params"] == pytest.approx([3.0, 1.0], abs=1e-6)
    with pytest.raises(ValueError):  # wrong number of initial values
        fit("Custom", x, y, equation="a*exp(-b*x) + c", initial_params="1 2")


def test_custom_equation_rejects_anything_outside_the_whitelist():
    from lumen.core.fitting import compile_equation
    bad = ["y = __import__('os').system('true')", "a.real*x", "x[0]*a", "(lambda: a)()", "open('f')*a", "a if x else b", "k*x", "a*'x'",
           "y = 1 + x", "f(x) = sin(a, x)", "y = a*x; b", "a*x == b", "exp(x, a)", "a*x.__class__", ""]
    for text in bad:
        with pytest.raises(ValueError):
            compile_equation(text)
    s = Session()
    with pytest.raises(ValueError):
        s.run_op("analyze.curve_fit", {"source": "Values", "x_column": "1 2 3", "y_column": "1 2 3", "custom_equation": "a*x.__class__"})
    with pytest.raises(ValueError):  # Function = Custom needs an equation
        s.run_op("analyze.curve_fit", {"source": "Values", "x_column": "1 2 3", "y_column": "1 2 3", "function": "Custom"})
    assert s.results.rows == []


def test_new_imagej_fit_functions():
    from scipy.special import erf
    from lumen.core.fitting import FIT_NAMES, canonical_name, evaluate, fit
    x = np.linspace(0, 10, 60)
    cases = {
        "Error Function": (2.0 + 1.5 * erf((x - 5) / 1.3), (2.0, 1.5, 5.0, 1.3)),
        "Gaussian (no offset)": (4.0 * np.exp(-(x - 6) ** 2 / (2 * 1.2 ** 2)), (4.0, 6.0, 1.2)),
        "Exponential (linear regression)": (3.0 * np.exp(0.2 * x), (3.0, 0.2)),
        "Log2": (1.0 + 2.0 * np.log(x + 0.5), (1.0, 2.0, -0.5)),
        "5th Degree Polynomial": (0.5 + x - 0.1 * x ** 5, (0.5, 1.0, 0.0, 0.0, 0.0, -0.1)),
    }
    for name, (y, want) in cases.items():
        r = fit(name, x, y)
        assert r["function"] == name and r["r2"] > 0.999, name
        for got, w in zip(r["params"], want):
            assert got == pytest.approx(w, rel=1e-2, abs=1e-3), name
    xp = x[1:]
    assert fit("Power (linear regression)", xp, 2 * xp ** 1.5)["params"] == pytest.approx([2.0, 1.5], rel=1e-6)
    xi = x[1:-1]
    yi = 2.0 * ((xi + 1) / (11 - xi)) ** (1 / 1.5)
    for name in ("Inverse Rodbard", "Rodbard2", "Rodbard (NIH Image)"):
        r = fit(name, xi, yi)
        assert r["params"] == pytest.approx([-1.0, 1.5, 2.0, 11.0], rel=0.02), name
        assert r["function"] == "Rodbard (NIH Image)"  # ImageJ's two spellings of the inverse Rodbard fit
    assert canonical_name("inverse rodbard") == "Rodbard (NIH Image)" and canonical_name("Error function") == "Error Function" and canonical_name("y = a+b*ln(x-c)") == "Log2"
    for name in ("Error Function", "Gaussian (no offset)", "Exponential (linear regression)", "Power (linear regression)", "Log2", "8th Degree Polynomial"):
        assert name in FIT_NAMES
    lg = evaluate("Log2", [1.0, 2.0, 3.0], [2.0, 4.0])
    assert np.isnan(lg[0]) and lg[1] == pytest.approx(1.0)  # ln(x − c) undefined left of c, like ImageJ
    with pytest.raises(ValueError):
        fit("Exponential (linear regression)", x, -np.ones_like(x))
    # the Curve Fitting op offers every function plus Custom and reports RMS
    from lumen.core.registry import REGISTRY
    choices = [p for p in REGISTRY.get("analyze.curve_fit").params if p.name == "function"][0].choices
    assert "Error Function" in choices and "Custom" in choices
    s = Session()
    s.run_op("analyze.curve_fit", {"source": "Values", "x_column": " ".join(f"{v:g}" for v in x), "y_column": " ".join(f"{v:.6f}" for v in cases['Error Function'][0]), "function": "Error Function"})
    row = s.results.rows[-1]
    assert row["c"] == pytest.approx(5.0, rel=1e-2) and row["d"] == pytest.approx(1.3, rel=1e-2) and row["RMS"] < 1e-3


def test_curve_fit_restarts_never_worsen_the_fit():
    from lumen.core.fitting import fit
    x = np.linspace(0, 10, 40)
    y = 5 * np.exp(-((x - 7) ** 2) / (2 * 0.5 ** 2))  # narrow peak far from an all-ones start
    r0 = fit("Custom", x, y, equation="a*exp(-(x-b)^2/(2*c^2))", initial_params=[1, 1, 1], restarts=0)
    r = fit("Custom", x, y, equation="a*exp(-(x-b)^2/(2*c^2))", initial_params=[1, 1, 1], restarts=20)
    assert r["sse"] <= r0["sse"] + 1e-9 and r["r2"] >= r0["r2"]
    assert fit("Custom", x, y, equation="a*exp(-(x-b)^2/(2*c^2))")["params"] == pytest.approx([5.0, 7.0, 0.5], rel=1e-3)  # automatic starts find it
    g = fit("Gaussian (no offset)", x, y, restarts=3)
    assert g["params"] == pytest.approx([5.0, 7.0, 0.5], rel=1e-3)
    s = Session()
    s.run_op("analyze.curve_fit", {"source": "Values", "x_column": " ".join(f"{v:g}" for v in x), "y_column": " ".join(f"{v:.6f}" for v in y), "function": "Gaussian (no offset)", "restarts": 5})
    assert s.results.rows[-1]["b"] == pytest.approx(7.0, rel=1e-3)


def test_calibrate_custom_equation(session):
    img = session.active
    pv = np.array([0, 64, 128, 192, 255], float)
    std = 0.1 + 2.0 * np.exp(-pv / 100.0)  # an exponential density calibration typed as a user-defined function
    out = session.run_op("analyze.calibrate", {"function": "Custom", "custom_equation": "y = a + b*exp(-x/c)", "pixel_values": " ".join(f"{v:g}" for v in pv),
                                               "standards": " ".join(f"{v:.6f}" for v in std), "unit": "OD"})
    cal = img.metadata["intensity_calibration"]
    assert cal["function"] == "Custom" and cal["equation"] == "y = a + b*exp(-x/c)" and cal["r2"] > 0.999
    assert cal["coefficients"] == pytest.approx([0.1, 2.0, 100.0], rel=0.02)
    assert out["plot"]["series"][0]["name"] == "fit" and np.isfinite(out["plot"]["series"][0]["values"]).all()
    from lumen.core.ops.extras import calibrated_values
    assert calibrated_values(img, np.array([0.0, 255.0])) == pytest.approx(std[[0, -1]], rel=1e-3)
    session.run_op("analyze.measure_calibrated", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10})
    plane = img.current_plane()[:10, :10].astype(float)
    assert session.results.rows[-1]["Mean"] == pytest.approx(float(np.mean(0.1 + 2.0 * np.exp(-plane / 100.0))), rel=1e-2)
    out = session.run_op("analyze.calibrated_values", {})
    assert session.images[out["new_images"][0]].data.max() == pytest.approx(float(0.1 + 2.0 * np.exp(-float(img.data.min()) / 100.0)), rel=1e-2)
    with pytest.raises(ValueError):
        session.run_op("analyze.calibrate", {"function": "Custom", "custom_equation": "", "pixel_values": "0 255", "standards": "0 1"}, image_id=img.id)
    with pytest.raises(ValueError):
        session.run_op("analyze.calibrate", {"function": "Custom", "custom_equation": "a*open(x)", "pixel_values": "0 255", "standards": "0 1"}, image_id=img.id)
    session.run_op("analyze.calibrate", {"function": "Error Function", "pixel_values": "0 64 128 192 255", "standards": "0 0.05 0.5 0.95 1"}, image_id=img.id)
    assert img.metadata["intensity_calibration"]["function"] == "Error Function" and len(img.metadata["intensity_calibration"]["coefficients"]) == 4


# ---------------------------------------------------------------- overlay fills are burned in
def test_burn_overlay_paints_roi_fill():
    from lumen.core.io import decode_ij_roi, encode_ij_roi
    from lumen.core.overlay_render import burn_overlay
    rgb = np.zeros((20, 20, 3), np.uint8)
    rect = {"type": "rect", "x": 4, "y": 4, "w": 10, "h": 8, "color": "#ff0000", "stroke_width": 1}
    out = burn_overlay(rgb, [{**rect, "fill": "#0000ff"}])
    assert tuple(out[8, 9]) == (0, 0, 255) and tuple(out[4, 9]) == (255, 0, 0)  # interior = fill, outline = stroke
    assert tuple(out[2, 2]) == (0, 0, 0)  # outside untouched
    out = burn_overlay(rgb, [{**rect, "fill_color": "#00ff00"}])  # legacy key (ImageJ .roi reader)
    assert tuple(out[8, 9]) == (0, 255, 0)
    out = burn_overlay(rgb, [{**rect, "fill": "#0000ff80"}])  # 8-digit hex: the alpha blends with the pixels underneath
    assert tuple(out[8, 9, :2]) == (0, 0) and abs(int(out[8, 9, 2]) - 128) <= 1
    out = burn_overlay(rgb, [rect])  # unfilled: the interior stays as it was
    assert tuple(out[8, 9]) == (0, 0, 0) and tuple(out[4, 9]) == (255, 0, 0)
    out = burn_overlay(rgb, [{**rect, "fill": "none"}])
    assert tuple(out[8, 9]) == (0, 0, 0)
    out = burn_overlay(rgb, [{"type": "oval", "x": 2, "y": 2, "w": 16, "h": 16, "color": "#ff0000", "fill": "#ffffff"}])
    assert tuple(out[10, 10]) == (255, 255, 255) and tuple(out[0, 0]) == (0, 0, 0)
    out = burn_overlay(rgb, [{"type": "line", "x1": 2, "y1": 2, "x2": 18, "y2": 18, "color": "#ff0000", "fill": "#0000ff"}])
    assert tuple(out[2, 10]) == (0, 0, 0) and tuple(out[10, 10]) == (255, 0, 0)  # lines stay unfilled
    # the fill survives the ImageJ .roi codec under both keys
    back = decode_ij_roi(encode_ij_roi({**rect, "fill": "#0000ff"}))
    assert back["fill"] == back["fill_color"] == "#0000ff"
    assert "fill" not in decode_ij_roi(encode_ij_roi(rect))


def test_burn_in_shows_roi_fill(session):
    img = session.active
    img.overlay = [{"type": "rect", "x": 10, "y": 10, "w": 30, "h": 20, "color": "#ff0000", "fill": "#00ff00"}]
    out = session.run_op("overlay.burn_in", {})
    burned = session.images[out["new_images"][0]]
    assert tuple(int(v) for v in burned.data[0, 0, :, 20, 25]) == (0, 255, 0)


# ---------------------------------------------------------------- bold text and Overlay Options… (session.settings) in the burn-in
def _lit_columns(a):
    cols = np.where(a.any(axis=2).any(axis=0))[0]
    return int(cols.max() - cols.min() + 1)


def test_burn_overlay_bold_text_is_wider_and_heavier(session):
    from lumen.core.overlay_render import burn_overlay
    rgb = np.zeros((40, 80, 3), np.uint8)
    item = {"type": "text", "text": "Hello", "x": 5, "y": 5, "color": "#ffffff"}
    regular = burn_overlay(rgb, [item], settings={})
    bold = burn_overlay(rgb, [{**item, "bold": True}], settings={})
    assert _lit_columns(bold) == _lit_columns(regular) + 1  # second pass 1 px to the right
    assert (bold.any(axis=2)).sum() > (regular.any(axis=2)).sum()
    assert (bold.astype(int) >= regular.astype(int)).all()  # every regular pixel stays lit (strokes only thicken)
    assert np.array_equal(burn_overlay(rgb, [{**item, "bold": False}], settings={}), regular)
    # the contrasting background box survives (and grows by the extra pixel) for bold text
    boxed = burn_overlay(rgb, [{**item, "color": "#000000", "background": True}], settings={})
    boxed_bold = burn_overlay(rgb, [{**item, "color": "#000000", "background": True, "bold": True}], settings={})
    white = lambda a: (a == 255).all(axis=2)  # noqa: E731
    assert white(boxed).any() and _lit_columns(boxed_bold) == _lit_columns(boxed) + 1
    assert (white(boxed_bold).sum(axis=1).max()) == white(boxed).sum(axis=1).max() + 1  # box one px wider
    # through the op: analyze.label's bold text items burn heavier
    img = session.active
    img.overlay = [item]
    plain = session.images[session.run_op("overlay.burn_in", {}, image_id=img.id)["new_images"][0]].data[0, 0]
    img.overlay = [{**item, "bold": True}]
    heavy = session.images[session.run_op("overlay.burn_in", {}, image_id=img.id)["new_images"][0]].data[0, 0]
    assert (heavy == 255).all(axis=0).sum() > (plain == 255).all(axis=0).sum()


def test_burn_overlay_uses_overlay_options_stroke_and_fill_defaults():
    from lumen.core.overlay_render import burn_overlay
    rgb = np.zeros((60, 80, 3), np.uint8)
    rect = {"type": "rect", "x": 10, "y": 10, "w": 30, "h": 20, "color": "#ff0000"}
    plain = burn_overlay(rgb, [rect], settings={})
    thick = burn_overlay(rgb, [rect], settings={"overlay_stroke_width": 4})
    assert (thick[..., 0] > 0).sum() > (plain[..., 0] > 0).sum()
    assert (plain[8:14, 25, 0] > 0).sum() == 2 and (thick[8:14, 25, 0] > 0).sum() == 4  # 1.5 px (rounded to 2) vs 4 px outline
    # an item with its own stroke width ignores the default
    own = {**rect, "stroke_width": 1}
    assert np.array_equal(burn_overlay(rgb, [own], settings={"overlay_stroke_width": 6}), burn_overlay(rgb, [own], settings={}))
    # the default fill paints unfilled area items; explicit fills (and an explicit 'none') win; lines stay unfilled
    filled = burn_overlay(rgb, [rect], settings={"overlay_default_fill": "#0000ff"})
    assert tuple(filled[20, 25]) == (0, 0, 255) and tuple(filled[10, 25]) == (255, 0, 0) and tuple(filled[5, 5]) == (0, 0, 0)
    assert tuple(burn_overlay(rgb, [{**rect, "fill": "#00ff00"}], settings={"overlay_default_fill": "#0000ff"})[20, 25]) == (0, 255, 0)
    assert tuple(burn_overlay(rgb, [{**rect, "fill_color": "#00ff00"}], settings={"overlay_default_fill": "#0000ff"})[20, 25]) == (0, 255, 0)
    assert tuple(burn_overlay(rgb, [{**rect, "fill": "none"}], settings={"overlay_default_fill": "#0000ff"})[20, 25]) == (0, 0, 0)
    assert tuple(burn_overlay(rgb, [{**rect, "fill": ""}], settings={"overlay_default_fill": "#0000ff"})[20, 25]) == (0, 0, 255)  # '' = unset
    half = burn_overlay(rgb, [rect], settings={"overlay_default_fill": "#0000ff80"})  # 8-digit hex keeps its alpha
    assert tuple(half[20, 25, :2]) == (0, 0) and abs(int(half[20, 25, 2]) - 128) <= 1
    line = {"type": "line", "x1": 2, "y1": 2, "x2": 50, "y2": 50, "color": "#ff0000"}
    assert tuple(burn_overlay(rgb, [line], settings={"overlay_default_fill": "#0000ff"})[2, 30]) == (0, 0, 0)
    for off in ("", "none"):
        assert np.array_equal(burn_overlay(rgb, [rect], settings={"overlay_default_fill": off}), plain)


def test_burn_overlay_draws_overlay_option_labels():
    from lumen.core.overlay_render import burn_overlay
    rgb = np.zeros((60, 100, 3), np.uint8)
    rect = {"type": "rect", "x": 10, "y": 10, "w": 30, "h": 20, "color": "#ff0000", "stroke_width": 1}
    oval = {"type": "oval", "x": 50, "y": 20, "w": 30, "h": 30, "color": "#ff0000", "stroke_width": 1, "name": "cell"}
    # not an ROI: never labelled, but it still takes index 1 (as in ImageJ, the viewport's Show labels and Measure Overlay)
    text = {"type": "text", "text": "ignored", "x": 0, "y": 50, "color": "#ff0000"}
    green = lambda a: ((a[..., 1] > 0) & (a[..., 0] == 0) & (a[..., 2] == 0))  # noqa: E731
    opts = {"overlay_show_labels": True, "overlay_label_color": "#00ff00", "overlay_label_font_size": 12}
    plain = burn_overlay(rgb, [text, rect, oval], settings={})
    assert not green(plain).any()
    assert np.array_equal(burn_overlay(rgb, [text, rect, oval], settings={**opts, "overlay_show_labels": False}), plain)
    # Show labels: the item's 1-based position in the overlay at its bounding-box corner, in the label colour / font size
    labelled = burn_overlay(rgb, [text, rect, oval], settings=opts)
    expect = burn_overlay(rgb, [text, rect, oval, {"type": "text", "text": "2", "x": 12, "y": 12, "font_size": 12, "color": "#00ff00"},
                                {"type": "text", "text": "3", "x": 52, "y": 22, "font_size": 12, "color": "#00ff00"}], settings={})
    assert green(labelled).any() and np.array_equal(labelled, expect)
    # the burnt number is the one Measure Overlay puts in the Results label (image:index) - the two sites must not drift apart
    from lumen.core.ops.overlay_extra import overlay_item_label
    assert [overlay_item_label("im", o, i) for i, o in enumerate([text, rect, oval], 1)][1:] == ["im:2", "im:cell"]
    # Use names as labels: the name when the item has one, else its index
    named = burn_overlay(rgb, [text, rect, oval], settings={**opts, "overlay_use_names": True})
    expect = burn_overlay(rgb, [text, rect, oval, {"type": "text", "text": "2", "x": 12, "y": 12, "font_size": 12, "color": "#00ff00"},
                                {"type": "text", "text": "cell", "x": 52, "y": 22, "font_size": 12, "color": "#00ff00"}], settings={})
    assert np.array_equal(named, expect) and not np.array_equal(named, labelled)
    # a scale bar / arrow before the ROIs shifts the numbers the same way the viewport shows them
    bar = {"type": "scalebar", "length": 10, "color": "#ff0000", "position": "Lower Right"}
    with_bar = burn_overlay(rgb, [bar, rect], settings=opts)
    expect = burn_overlay(rgb, [bar, rect, {"type": "text", "text": "2", "x": 12, "y": 12, "font_size": 12, "color": "#00ff00"}], settings={})
    assert np.array_equal(with_bar, expect)
    # an explicit label is kept (and drawn in the label colour), the font size follows the option, the background boxes the text
    explicit = burn_overlay(rgb, [{**rect, "label": "A"}, oval], settings=opts)
    expect = burn_overlay(rgb, [rect, oval, {"type": "text", "text": "A", "x": 12, "y": 12, "font_size": 12, "color": "#00ff00"},
                                {"type": "text", "text": "2", "x": 52, "y": 22, "font_size": 12, "color": "#00ff00"}], settings={})
    assert np.array_equal(explicit, expect)
    big = burn_overlay(rgb, [rect], settings={**opts, "overlay_label_font_size": 24})
    assert green(big).sum() > green(burn_overlay(rgb, [rect], settings=opts)).sum()
    boxed = burn_overlay(rgb, [rect], settings={**opts, "overlay_label_background": True})
    assert (boxed == 255).all(axis=2).any() and not (labelled == 255).all(axis=2).any()  # white box behind the (dark) green label
    # labels respect the item's plane: an item pinned to another frame is neither drawn nor labelled, but keeps its number
    img = LumenImage(np.zeros((3, 1, 1, 60, 100), np.uint8), name="mov.tif", axes="TZCYX")
    items = [{**rect, "position": {"t": 0}}, {**oval, "position": {"t": 2}}]
    on2 = burn_overlay(rgb, items, img, t=2, z=0, settings=opts)
    expect = burn_overlay(rgb, [oval, {"type": "text", "text": "2", "x": 52, "y": 22, "font_size": 12, "color": "#00ff00"}], settings={})
    assert np.array_equal(on2, expect)


def test_drawstring_clamps_out_of_range_values():
    # ImageJ's Byte/ShortProcessor.setValue clamps; NumPy 2.x raised OverflowError for `setColor(300); drawString(...)` on 8-bit
    from lumen.core.overlay_render import draw_text_into_plane
    for v in (300, 300.0, 65535.0, 1e9):
        assert draw_text_into_plane(np.zeros((20, 40), np.uint8), "Ab", 0, 15, v, 12).max() == 255
    assert draw_text_into_plane(np.zeros((20, 40), np.uint8), "Ab", 0, 15, -1.0, 12).max() == 0
    assert draw_text_into_plane(np.full((20, 40), 7, np.uint8), "Ab", 0, 15, float("nan"), 12).min() == 0  # Java (int) NaN == 0
    assert draw_text_into_plane(np.zeros((20, 40), np.uint16), "Ab", 0, 15, 70000.0, 12).max() == 65535
    assert draw_text_into_plane(np.zeros((20, 40), np.int16), "Ab", 0, 15, 65535, 12).max() == 32767
    assert draw_text_into_plane(np.zeros((20, 40), np.float32), "Ab", 0, 15, 1e9, 12).max() == np.float32(1e9)
    # in-range values are untouched and nothing outside the glyphs changes
    out = draw_text_into_plane(np.zeros((20, 40), np.uint8), "Ab", 0, 15, 200, 12)
    assert set(np.unique(out)) == {0, 200}
    # the macro path: setColor beyond the 8-bit range then drawString draws white instead of raising
    from lumen.core.ij_compat import IJ
    s = Session()
    im = s.add_image(LumenImage(np.zeros((20, 40), np.uint8), name="t.tif"))
    ij = IJ(s)
    ij.macro('setColor(300); drawString("x", 5, 15);')
    assert im.data.max() == 255


def test_burn_overlay_falls_back_to_the_live_session_settings(session, monkeypatch):
    from lumen.core import session as sessmod
    from lumen.core.overlay_render import burn_overlay
    rgb = np.zeros((60, 80, 3), np.uint8)
    rect = {"type": "rect", "x": 10, "y": 10, "w": 30, "h": 20, "color": "#ff0000"}
    opts = {"overlay_show_labels": True, "overlay_label_color": "#00ff00", "overlay_default_fill": "#0000ff", "overlay_stroke_width": 3}
    # without the keys nothing changes: {} and unrelated settings render like the plain defaults
    plain = burn_overlay(rgb, [rect], settings={})
    assert np.array_equal(burn_overlay(rgb, [rect], settings={"foreground": 255, "roi_color": "#00ff00"}), plain)
    monkeypatch.setattr(sessmod.SESSION, "settings", dict(sessmod.SESSION.settings, **opts))
    assert np.array_equal(burn_overlay(rgb, [rect]), burn_overlay(rgb, [rect], settings=opts))  # settings=None → SESSION.settings
    assert not np.array_equal(burn_overlay(rgb, [rect]), plain)
    # Image ▸ Overlay ▸ Burn In honours the options (the op's session and the live one both carry them)
    from lumen.core.render import render_rgb
    session.settings.update(opts)
    img = session.active
    img.overlay = [rect]
    burned = session.images[session.run_op("overlay.burn_in", {}, image_id=img.id)["new_images"][0]].data[0, 0].transpose(1, 2, 0)
    assert tuple(int(v) for v in burned[20, 25]) == (0, 0, 255)  # default fill
    assert np.array_equal(burned, burn_overlay(render_rgb(img), [rect], img, settings=opts))  # thick outline + index label
    assert not np.array_equal(burned, burn_overlay(render_rgb(img), [rect], img, settings={}))


# ---------------------------------------------------------------- Process ▸ Enhance Contrast / Subtract Background regressions
def test_enhance_contrast_constant_plane_keeps_float_display():
    from lumen.core.render import render_rgb
    for arr in (np.zeros((16, 16), np.uint8), np.full((16, 16), 7, np.uint16), np.full((16, 16), 2.5, np.float32)):
        s = Session()
        img = s.add_image(LumenImage(arr))
        s.run_op("process.enhance_contrast", {"saturated": 0.35})
        ch = img.display.channels[0]
        assert isinstance(ch.max, float) and ch.max == ch.min + 1            # max was the tuple (min, min + 1)
        render_rgb(img)                                                       # raised TypeError before
        s.run_op("process.enhance_contrast", {"saturated": 0.35, "normalize": True})   # raised TypeError before
        assert np.isfinite(img.data).all()


def _java_equalize_lut(hist, classic=False):
    """Line-by-line port of the loop in ImageJ's ContrastEnhancer.equalize() (Java Math.round = floor(x + 0.5))."""
    import math
    mx = len(hist) - 1

    def w(i):
        h = int(hist[i])
        return float(h) if (h < 2 or classic) else math.sqrt(float(h))
    total = w(0) + sum(2 * w(i) for i in range(1, mx)) + w(mx)
    scale = mx / total
    lut = [0] * (mx + 1)
    acc = w(0)
    for i in range(1, mx):
        d = w(i)
        acc += d
        lut[i] = int(math.floor(acc * scale + 0.5))
        acc += d
    lut[mx] = mx
    return np.array(lut)


def test_equalize_matches_imagej():
    from lumen.core.ops.enhance import ij_equalize_lut
    rng = np.random.default_rng(0)
    p = np.clip(rng.normal(60, 15, (128, 128)), 0, 255).astype(np.uint8)
    hist = np.bincount(p.ravel(), minlength=256)
    for classic in (False, True):
        assert np.array_equal(ij_equalize_lut(hist, classic), _java_equalize_lut(hist, classic))
    lut = ij_equalize_lut(hist)
    assert (lut[60], lut[30], lut[75], lut[0], lut[255]) == (130, 20, 197, 0, 255)   # skimage's plain CDF gave 133 / 6 / 218
    s = Session()
    img = s.add_image(LumenImage(p))
    s.run_op("process.enhance_contrast", {"equalize": True})
    assert np.array_equal(img.data[0, 0, 0], lut[p]) and img.display.channels[0].max == 255
    s = Session()
    img = s.add_image(LumenImage(p))
    s.run_op("process.enhance_contrast", {"equalize": True, "classic_equalization": True})   # ImageJ's Alt-key variant
    assert np.array_equal(img.data[0, 0, 0], ij_equalize_lut(hist, classic=True)[p])
    # the histogram comes from the selection, the LUT is applied to the whole plane (ImageJ resetRoi)
    s = Session()
    img = s.add_image(LumenImage(p))
    s.run_op("process.enhance_contrast", {"equalize": True}, roi={"type": "rect", "x": 0, "y": 0, "w": 64, "h": 64})
    assert np.array_equal(img.data[0, 0, 0], ij_equalize_lut(np.bincount(p[:64, :64].ravel(), minlength=256))[p])
    # 16-bit: a 65536-bin histogram over the full range
    u = rng.integers(1000, 1100, (64, 64)).astype(np.uint16)
    h16 = np.bincount(u.ravel(), minlength=65536)
    assert np.array_equal(ij_equalize_lut(h16), _java_equalize_lut(h16))
    s = Session()
    img = s.add_image(LumenImage(u))
    s.run_op("process.enhance_contrast", {"equalize": True})
    assert np.array_equal(img.data[0, 0, 0], ij_equalize_lut(h16)[u]) and img.display.channels[0].max == 65535
    assert ij_equalize_lut(np.bincount(np.full(100, 7, np.uint8), minlength=256))[7] == 128   # constant image → 128 like ImageJ
    s = Session()
    s.add_image(LumenImage(np.zeros((8, 8), np.float32)))
    with pytest.raises(ValueError, match="32-bit"):                                  # ImageJ refuses 32-bit equalization
        s.run_op("process.enhance_contrast", {"equalize": True})


@pytest.mark.parametrize("method", ["Rolling ball", "Sliding paraboloid"])
def test_subtract_background_thin_images(method):
    for shape, radius in [((1, 200), 20), ((2, 200), 50), ((4, 200), 150), ((200, 2), 50), ((1, 1), 150)]:
        s = Session()
        im = s.add_image(LumenImage(np.random.default_rng(0).integers(0, 255, shape).astype(np.uint8), name="thin"))
        s.run_op("process.subtract_background", {"radius": radius, "method": method})   # ZeroDivisionError / broadcast error before
        assert im.current_plane().shape == shape
    # ImageJ's shrinkImage: ceil-sized block minimum, so even a thin image keeps one row / column
    from lumen.core.ops.enhance import _shrink_plane
    a = np.random.default_rng(1).random((33, 17)).astype(np.float32)
    for shrink in (2, 4, 8):
        small = _shrink_plane(a, shrink)
        sh, sw = (33 + shrink - 1) // shrink, (17 + shrink - 1) // shrink
        assert small.shape == (sh, sw) and small.dtype == np.float32
        assert small[0, 0] == a[:shrink, :shrink].min() and small[-1, -1] == a[(sh - 1) * shrink:, (sw - 1) * shrink:].min()
    assert _shrink_plane(np.zeros((2, 200)), 4).shape == (1, 50)
