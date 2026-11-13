import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.registry import REGISTRY
from lumen.core.session import Session
from lumen.core.threshold import METHODS, auto_threshold
from lumen.core import roi as roimod
from lumen.core.ops.file_ops import _blobs


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


def test_canonical_layout():
    img = LumenImage(np.zeros((10, 20), np.uint8))
    assert img.shape == (1, 1, 1, 10, 20)
    img = LumenImage(np.zeros((5, 3, 10, 20), np.uint16), axes="ZCYX")
    assert img.shape == (1, 5, 3, 10, 20) and img.nc == 3
    rgb = LumenImage(np.zeros((10, 20, 3), np.uint8), axes="YXS", rgb=True)
    assert rgb.shape == (1, 1, 3, 10, 20) and rgb.rgb


def test_threshold_methods_reasonable():
    b = _blobs()
    for m in METHODS:
        lo, hi = auto_threshold(b, m)
        assert 0 < lo < 255 and hi == 255, m


def test_threshold_light_background():
    b = _blobs()
    lo, hi = auto_threshold(255 - b, "Otsu", dark_background=False)
    assert lo == 0 and 0 < hi < 255


def test_plane_op_scope_and_roi(session):
    img = session.active
    before = img.data.copy()
    roi = {"type": "rect", "x": 0, "y": 0, "w": 50, "h": 50}
    session.run_op("math.set", {"value": 0}, roi=roi)
    assert img.data[0, 0, 0, :50, :50].max() == 0
    assert np.array_equal(img.data[0, 0, 0, 60:, 60:], before[0, 0, 0, 60:, 60:])
    session.undo(img.id)
    assert np.array_equal(img.data, before)
    session.redo(img.id)
    assert img.data[0, 0, 0, :50, :50].max() == 0


def test_threshold_particles_pipeline(session):
    img = session.active
    session.run_op("filters.gaussian", {"sigma": 1.5})
    session.run_op("threshold.compute", {"method": "Otsu"})
    session.run_op("threshold.apply", {})
    assert img.is_binary
    session.run_op("binary.fill_holes", {})
    out = session.run_op("analyze.particles", {"min_size": 30, "show": "Count Masks", "add_to_manager": True})
    assert "particles" in out["message"]
    assert len(session.roi_manager.rois) > 10
    assert session.results.columns[:2] == ["Label", "Area"]
    assert len(session.images) == 2  # count mask image created


def test_measure_roi_types(session):
    session.measurements = ["area", "mean", "centroid", "perimeter", "shape_descriptors", "feret", "fit_ellipse", "bounding_rect"]
    session.run_op("analyze.measure", {}, roi={"type": "oval", "x": 10, "y": 10, "w": 40, "h": 20})
    row = session.results.rows[-1]
    assert abs(row["Area"] - np.pi * 20 * 10) / (np.pi * 200) < 0.05
    assert 0.8 < row["Circ."] < 0.9  # 2:1 ellipse
    assert row["Feret"] >= 39
    session.run_op("analyze.measure", {}, roi={"type": "line", "x1": 0, "y1": 0, "x2": 30, "y2": 40})
    assert abs(session.results.rows[-1]["Length"] - 50) < 1e-6
    session.run_op("analyze.measure", {}, roi={"type": "point", "points": [[5, 5], [10, 10]]})
    assert "Value" in session.results.rows[-1]


def test_roi_mask_roundtrip():
    m = np.zeros((30, 40), bool)
    m[5:15, 10:30] = True
    m[8, 12] = False
    r = roimod.from_mask(m)
    back = roimod.to_mask(r, (30, 40))
    assert np.array_equal(back, m)
    combo = roimod.combine([r, {"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5}], (30, 40), "or")
    assert roimod.to_mask(combo, (30, 40)).sum() == m.sum() + 25


def test_stack_ops():
    s = Session()
    s.run_op("file.sample", {"name": "Synthetic nuclei (3 channels)"})
    img = s.active
    assert img.nc == 3 and img.dtype == np.uint16
    with pytest.raises(ValueError):
        s.run_op("stacks.z_project", {"method": "Max Intensity"})  # single plane: nothing to project
    s.select(img.id)
    out = s.run_op("color.split", {})
    assert len(out["new_images"]) == 3
    ids = out["new_images"]
    out = s.run_op("color.merge", {"c1": ids[0], "c2": ids[1], "c3": ids[2]})
    merged = s.images[out["new_images"][0]]
    assert merged.nc == 3 and merged.display.mode == "composite"
    s.select(merged.id)
    s.run_op("type.rgb", {})
    assert merged.rgb and merged.data.dtype == np.uint8
    s.run_op("type.8bit", {})
    assert not merged.rgb and merged.nc == 1


def test_hyperstack_dims():
    s = Session()
    s.run_op("file.sample", {"name": "Nuclei timelapse (synthetic)"})
    img = s.active
    assert img.nt == 12 and img.nz == 1
    out = s.run_op("stacks.z_profile", {"axis": "T"})
    assert len(out["plot"]["x"]) == 12
    s.run_op("stacks.substack", {"slices": "1-4", "axis": "T"})
    assert s.active.nt == 4
    s.run_op("stacks.montage", {"axis": "T", "columns": 2})
    assert s.active.width == img.width * 2


def test_preview_does_not_modify(session):
    img = session.active
    before = img.data.copy()
    out = session.run_op("filters.gaussian", {"sigma": 5}, preview=True)
    assert out["preview"].shape == (1, 256, 256)
    assert np.array_equal(img.data, before)
    assert not session.histories[img.id].can_undo()


def test_calculator_and_math(session):
    img = session.active
    out = session.run_op("image.duplicate", {"what": "Whole stack"})
    dup = session.images[out["new_images"][0]]
    session.run_op("process.calculator", {"image1": img.id, "operation": "Subtract", "image2": dup.id, "create_new": True})
    assert session.active.data.max() == 0
    session.select(img.id)
    session.run_op("math.multiply", {"value": 2})
    assert img.data.max() == 255


def test_scripting(session):
    from lumen.core.scripting import run_script
    out = run_script(session, "lumen.run('filters.gaussian', sigma=2)\nprint(lumen.image.name, lumen.data.shape)\nlumen.run('analyze.measure', roi={'type':'rect','x':0,'y':0,'w':10,'h':10})\nprint(len(lumen.results))")
    assert out["error"] is None, out["error"]
    assert "blobs" in out["stdout"] and out["stdout"].strip().endswith("1")


def test_macro_roundtrip(session):
    from lumen.core.scripting import run_script
    session.run_op("filters.gaussian", {"sigma": 2})
    session.run_op("threshold.apply", {"method": "Otsu", "dark_background": True})
    code = "\n".join(session.macro)
    s2 = Session()
    out = run_script(s2, code)
    assert out["error"] is None, out["error"]
    assert s2.active.is_binary


def test_registry_schema():
    for spec in REGISTRY.list():
        assert spec["id"] and spec["name"] and spec["category"]
        for p in spec["params"]:
            assert p["type"] in ("int", "float", "bool", "choice", "string", "text", "image", "multichoice", "color")


def test_io_roundtrip(tmp_path):
    from lumen.core.io import read_image, write_image
    s = Session()
    s.run_op("file.sample", {"name": "Synthetic nuclei (3 channels)"})
    img = s.active
    img.display.channels[0].min, img.display.channels[0].max = 100.0, 3000.0
    out = write_image(img, str(tmp_path / "test.tif"))
    back = read_image(out)
    assert back.shape == img.shape and back.dtype == img.dtype
    assert abs(back.calibration.pixel_width - 0.325) < 1e-6 and back.calibration.unit == "µm"
    assert back.display.channels[0].max == 3000.0 and back.display.channels[0].lut == "Blue"
    png = write_image(img, str(tmp_path / "test.png"))
    rgb = read_image(png)
    assert rgb.rgb and rgb.height == img.height


def test_duplicate_current_plane_from_deep_slice():
    s = Session()
    s.run_op("file.sample", {"name": "Cells (3-D, 2 channels)"})
    img = s.active
    s.update_display(img.id, {"z": 30, "c": 1})
    out = s.run_op("image.duplicate", {"what": "Current plane"})
    dup = s.images[out["new_images"][0]]
    assert dup.shape == (1, 1, 1, img.height, img.width)
    assert np.array_equal(dup.data[0, 0, 0], img.data[0, 30, 1])


def test_picker_single_value_ops_keep_the_other_setting():
    """The value-picker tool must change one colour setting only (ImageJ: click = foreground, Alt-click = background)."""
    import json
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("edit.set_colors", {"foreground": 100, "background": 50})
    s.run_op("edit.set_foreground", {"value": 200})
    assert s.settings["foreground"] == 200.0 and s.settings["background"] == 50.0
    s.run_op("edit.paint_stroke", {"points": json.dumps([[4, 4]]), "width": 1, "erase": True})
    assert s.active.data[0, 0, 0, 4, 4] == 50
    s.run_op("edit.set_background", {"value": 7})
    assert s.settings["foreground"] == 200.0 and s.settings["background"] == 7.0


def test_format_time_stamp_contract():
    """Pins the backend formatter the viewport's timeStampText mirrors (uncalibrated stacks count frames,
    {frame}/{t}/{slice}/{z} are 1-based, start_frame only shifts {time}, unknown placeholders leave the format as is)."""
    from lumen.core.overlay_render import format_time_stamp
    img = LumenImage(np.zeros((3, 2, 1, 4, 4), np.uint8), axes="TZCYX")
    assert img.calibration.frame_interval == 0.0
    assert [format_time_stamp("t = {time:.1f} {unit}", img, t, 1, 0) for t in range(3)] == ["t = 0.0 frame", "t = 1.0 frame", "t = 2.0 frame"]
    assert [format_time_stamp("{frame}", img, t, 1, 0) for t in range(3)] == ["1", "2", "3"]
    assert format_time_stamp("{slice} {z} {t} {time} {frame:03d}", img, 1, 1, 0) == "2 2 2 1.0 002"
    assert format_time_stamp("{foo}", img, 1, 1, 0) == "{foo}"
    img.calibration.frame_interval = 2.0
    img.calibration.time_unit = "sec"
    assert [format_time_stamp("t = {time:.1f} {unit} f{frame}", img, t, 0, 1) for t in range(3)] == ["t = -2.0 sec f1", "t = 0.0 sec f2", "t = 2.0 sec f3"]


def test_apply_threshold_only_current_plane_keeps_other_planes_16bit():
    s = Session()
    s.run_op("file.sample", {"name": "Cells (3-D, 2 channels)"})
    img = s.active
    s.update_display(img.id, {"z": 10, "c": 1})
    s.run_op("threshold.apply", {"method": "Otsu", "only_current_plane": True})
    assert img.dtype == np.uint8
    cur = img.data[0, 10, 1]
    assert set(np.unique(cur).tolist()) <= {0, 255}
    other = img.data[0, 20, 1]
    assert other.max() > 0 and len(np.unique(other)) > 2  # scaled gray, not zeroed


def test_add_slice_channel_axis():
    s = Session()
    s.run_op("file.sample", {"name": "Synthetic nuclei (3 channels)"})
    img = s.active
    s.update_display(img.id, {"c": 1})
    s.run_op("stacks.add_slice", {"axis": "C"})
    assert img.nc == 4 and len(img.display.channels) == 4
    assert img.data[0, 0, 2].max() == 0 and img.data[0, 0, 3].max() > 0


def test_roi_manager_readd_gets_new_id(session):
    img = session.active
    r1 = session.roi_manager.add({"type": "rect", "x": 1, "y": 1, "w": 5, "h": 5}, image_id=img.id)
    r2 = session.roi_manager.add(dict(r1), image_id=img.id)
    assert r1["id"] != r2["id"] and len(session.roi_manager.rois) == 2
    session.roi_manager.remove([r2["id"]])
    assert [r["id"] for r in session.roi_manager.rois] == [r1["id"]]


def test_line_angle_uses_calibration():
    roi = {"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": -10}
    assert abs(roimod.line_angle(roi) - 45) < 1e-9
    assert abs(roimod.line_angle(roi, 1.0, 2.0) - 63.4349) < 1e-3


# ---------------------------------------------------------------- Analyze Particles / Histogram / Measure dialog parity
def _particle_image(session, unit="micron", pw=0.5):
    from lumen.core.image import Calibration
    a = np.zeros((40, 60), np.uint8)
    a[2:6, 2:6] = 255        # 16 px²  (4 µm² at 0.5 µm/px)
    a[10:20, 10:20] = 255    # 100 px² (25 µm²)
    a[25:37, 30:50] = 255    # 240 px² (60 µm²)
    img = session.add_image(LumenImage(a, name="parts.tif", calibration=Calibration(pw, pw, 1.0, unit)))
    img.lut_is_binary = True
    return img


def test_normalize_unit():
    from lumen.core.ops.analyze import normalize_unit
    for spelling in ("micron", "microns", "um", "µm", "μm", "Micrometer", " micrometres "):
        assert normalize_unit(spelling) == "µm", spelling
    assert normalize_unit("pixels") == "pixel" and normalize_unit("") == "pixel" and normalize_unit(None) == "pixel"
    assert normalize_unit("millimeter") == "mm" and normalize_unit("nm") == "nm" and normalize_unit("furlong") == "furlong"


def test_particles_pixel_units_filters_in_pixels():
    s = Session()
    _particle_image(s, unit="micron", pw=0.5)
    out = s.run_op("analyze.particles", {"min_size": 50, "show": "Nothing", "add_to_manager": False})  # calibrated µm²: only the 60 µm² particle
    assert s.tables["Particle Summary"]["rows"][0][1] == 1 and "µm²" in out["message"] and out["log"] == "1 particles"
    out = s.run_op("analyze.particles", {"min_size": 50, "show": "Nothing", "add_to_manager": False, "pixel_units": True})  # pixels²: 100 and 240 px²
    assert s.tables["Particle Summary"]["rows"][0][1] == 2 and "pixel²" in out["message"]
    out = s.run_op("analyze.particles", {"min_size": 50, "max_size": 150, "show": "Nothing", "add_to_manager": False, "pixel_units": True})
    assert s.tables["Particle Summary"]["rows"][0][1] == 1
    assert s.results.rows[-1]["Area"] == pytest.approx(25.0)  # the results stay calibrated
    s.run_op("analyze.particles", {"min_size": 0.9, "max_size": 1.1, "min_circularity": 0.75, "show": "Nothing", "add_to_manager": False, "pixel_units": True})
    assert s.tables["Particle Summary"]["rows"][0][1] == 0


def test_particles_record_starts_and_in_situ_show():
    s = Session()
    img = _particle_image(s)
    before = img.data.copy()
    s.run_op("analyze.particles", {"show": "Nothing", "record_starts": True, "add_to_manager": False})
    rows = s.results.rows[-3:]
    assert [(r["XStart"], r["YStart"]) for r in rows] == [(2, 2), (10, 10), (30, 25)]  # first pixel of each particle in raster order (pixels)
    n_images = len(s.images)
    out = s.run_op("analyze.particles", {"show": "Masks", "in_situ_show": True, "add_to_manager": False, "min_size": 50, "pixel_units": True})
    assert len(s.images) == n_images and not out["new_images"] and "in situ" in out["message"]
    plane = img.current_plane()
    assert plane[3, 3] == 0 and plane[15, 15] == 255 and plane[30, 40] == 255 and img.lut_is_binary  # the small particle was filtered out of the mask
    assert s.undo(img.id) and np.array_equal(img.data, before)  # in situ replacement is undoable
    s.run_op("analyze.particles", {"show": "Count Masks", "in_situ_show": True, "add_to_manager": False})
    assert sorted(np.unique(img.current_plane()).tolist()) == [0, 1, 2, 3] and img.display.channels[0].lut == "Glasbey on dark"
    # a stack: only the processed plane is replaced unless the whole stack is processed
    vol = np.zeros((3, 20, 20), np.uint8)
    vol[:, 5:10, 5:10] = 255
    stk = s.add_image(LumenImage(vol, name="stk.tif", axes="ZYX"))
    stk.lut_is_binary = True
    s.run_op("analyze.particles", {"show": "Outlines", "in_situ_show": True, "add_to_manager": False}, image_id=stk.id)
    assert stk.nz == 3 and stk.data[0, 0, 0, 7, 7] == 0 and stk.data[0, 1, 0, 7, 7] == 255  # plane z=0 is now the outline, z=1 untouched
    s.run_op("analyze.particles", {"show": "Masks", "in_situ_show": True, "all_planes": True, "add_to_manager": False}, image_id=stk.id)
    assert stk.data[0, 2, 0, 7, 7] == 255 and stk.data[0, 2, 0, 1, 1] == 0 and stk.data.dtype == np.uint8


def test_particles_composite_rois_keep_holes():
    from lumen.core import roi as roimod
    s = Session()
    a = np.zeros((30, 30), np.uint8)
    a[5:25, 5:25] = 255
    a[12:18, 12:18] = 0  # a square with a hole
    img = s.add_image(LumenImage(a, name="ring.tif"))
    img.lut_is_binary = True
    s.run_op("analyze.particles", {"show": "Nothing", "include_holes": False, "composite_rois": True})
    r = s.roi_manager.rois[-1]
    assert r["type"] == "mask" and roimod.to_mask(r, a.shape).sum() == 400 - 36  # exact pixels, hole excluded
    assert s.results.rows[-1]["Area"] == 364
    s.run_op("analyze.particles", {"show": "Nothing", "include_holes": False, "composite_rois": False})
    r = s.roi_manager.rois[-1]
    assert r["type"] == "polygon" and roimod.to_mask(r, a.shape).sum() == pytest.approx(400, abs=40)  # outer outline, hole filled
    assert s.results.rows[-1]["Area"] == 364  # measurements still exclude the hole


def test_histogram_dialog_options():
    s = Session()
    vol = np.zeros((2, 16, 16), np.uint8)
    vol[0] = 10
    vol[1] = 200
    img = s.add_image(LumenImage(vol, name="h.tif", axes="ZYX"))
    out = s.run_op("analyze.histogram", {"bins": 256})
    h = out["histogram"]
    assert h["counts"][10] == 256 and h["stats"]["count"] == 256 and h["log_scale"] is False and h["stack"] is False and "results_added" not in out
    out = s.run_op("analyze.histogram", {"bins": 256, "all_planes": True, "log_scale": True, "y_max": 300})
    h = out["histogram"]
    assert h["stats"]["count"] == 512 and h["counts"][10] == 256 and h["counts"][200] == 256 and h["log_scale"] is True and h["y_max"] == 300 and h["stack"] is True
    out = s.run_op("analyze.histogram", {"bins": 4, "x_min": 0, "x_max": 100, "all_planes": True, "copy_to_results": True})
    h = out["histogram"]
    assert h["x_range"] == [0.0, 100.0] and h["fixed_range"] is True and sum(h["counts"]) == 256 and h["counts"][0] == 256  # the 200s fall outside the fixed range
    assert h["stats"]["max"] == 200  # statistics still describe every pixel
    assert out["results_added"] == 4 and s.results.columns == ["Bin Start", "Count"] and s.results.rows[0] == {"Bin Start": 0.0, "Count": 256}
    s.run_op("analyze.histogram", {"bins": 256}, roi={"type": "rect", "x": 0, "y": 0, "w": 4, "h": 4}, image_id=img.id)
    assert img.display.z == 0


def test_measure_redirects_to_another_image():
    from lumen.core.image import Calibration
    s = Session()
    a = np.zeros((20, 20), np.uint8)
    a[5:15, 5:15] = 255
    src = s.add_image(LumenImage(a, name="mask.tif"))
    other = s.add_image(LumenImage(np.full((20, 20), 7, np.uint16), name="signal.tif", calibration=Calibration(2.0, 2.0, 1.0, "um")))
    s.select(src.id)
    s.run_op("analyze.set_measurements", {"measurements": ["area", "mean", "min", "max"], "redirect": "signal.tif"})
    assert s.measure_options["redirect"] == other.id
    roi = {"type": "rect", "x": 5, "y": 5, "w": 10, "h": 10}
    s.run_op("analyze.measure", {}, roi=roi, image_id=src.id)
    row = s.results.rows[-1]
    assert row["Mean"] == 7 and row["Max"] == 7 and row["Label"] == "signal.tif" and row["Area"] == 400  # values and calibration of the redirect image
    s.run_op("analyze.measure_stack", {}, roi=roi, image_id=src.id)
    assert s.results.rows[-1]["Mean"] == 7
    # Analyze Particles reads its intensities from the redirect image too (ImageJ)
    s.run_op("analyze.particles", {"show": "Nothing", "add_to_manager": False}, image_id=src.id)
    assert s.results.rows[-1]["Mean"] == 7
    s.run_op("analyze.set_measurements", {"measurements": ["area", "mean"], "redirect": ""})
    assert s.measure_options["redirect"] == ""
    s.run_op("analyze.measure", {}, roi=roi, image_id=src.id)
    assert s.results.rows[-1]["Mean"] == 255 and s.results.rows[-1]["Label"] == "mask.tif"
    with pytest.raises(ValueError):
        s.run_op("analyze.set_measurements", {"measurements": ["area"], "redirect": "nope.tif"})
    small = s.add_image(LumenImage(np.zeros((5, 5), np.uint8), name="small.tif"))
    s.run_op("analyze.set_measurements", {"measurements": ["area", "mean"], "redirect": small.id})
    with pytest.raises(ValueError, match="same size"):
        s.run_op("analyze.measure", {}, roi=roi, image_id=src.id)
    s.close_image(small.id)
    with pytest.raises(ValueError, match="not open"):
        s.run_op("analyze.measure", {}, roi=roi, image_id=src.id)


def test_distribution_and_grouped_summary():
    s = Session()
    rng = np.random.default_rng(0)
    s.results.append([{"Label": "a" if i < 40 else "b", "Area": float(v), "Mean": float(i)} for i, v in enumerate(rng.normal(50, 10, 100))])
    out = s.run_op("analyze.distribution", {"column": "Area", "bins": 10})
    assert len(out["plot"]["x"]) == 10 and sum(out["plot"]["series"][0]["values"]) == 100 and out["plot"]["bins"] == 10
    out = s.run_op("analyze.distribution", {"column": "Area", "auto_binning": True})
    p = out["plot"]
    sd = float(np.std([r["Area"] for r in s.results.rows], ddof=1))
    lo, hi = min(r["Area"] for r in s.results.rows), max(r["Area"] for r in s.results.rows)
    assert p["bins"] == int(np.ceil((hi - lo) / (3.49 * sd * 100 ** (-1 / 3)))) and p["auto_binning"] is True  # Scott's rule, like ImageJ
    out = s.run_op("analyze.distribution", {"column": "Area", "bins": 5, "range": "0-100"})
    assert out["plot"]["range"] == [0.0, 100.0] and out["plot"]["edges"][0] == 0.0 and out["plot"]["edges"][-1] == 100.0 and sum(out["plot"]["series"][0]["values"]) == 100
    with pytest.raises(ValueError):
        s.run_op("analyze.distribution", {"column": "Area", "range": "1-2-3"})
    # Summarize: whole table (as before) or one Mean/SD/Min/Max block per group
    out = s.run_op("analyze.summarize", {})
    t = out["table"]
    assert t["columns"] == ["Label", "Area", "Mean"] and [r[0] for r in t["rows"]] == ["Mean", "SD", "Min", "Max"]
    out = s.run_op("analyze.summarize", {"group_by": "Label"})
    t = out["table"]
    assert t["columns"] == ["Label", "Group", "N", "Area", "Mean"]  # the group column is renamed when it would collide with the statistic column
    assert s.run_op("analyze.summarize", {"group_by": "Mean"})["table"]["columns"][:3] == ["Label", "Mean", "N"]
    assert len(t["rows"]) == 8 and [r[1] for r in t["rows"]] == ["a"] * 4 + ["b"] * 4 and t["rows"][0][2] == 40 and t["rows"][4][2] == 60
    i_mean = t["columns"].index("Mean")
    assert t["rows"][0][i_mean] == pytest.approx(19.5) and t["rows"][4][i_mean] == pytest.approx(69.5)  # per-group means of the Mean column
    with pytest.raises(ValueError):
        s.run_op("analyze.summarize", {"group_by": "Nope"})


# ---------------------------------------------------------------- undo carries metadata (slice labels) and the binary-LUT flag
def test_undo_restores_slice_labels_and_binary_flag():
    from lumen.core.ops.stacks import set_slice_label
    s = Session()
    vol = np.stack([np.full((8, 8), 10 * (k + 1), np.uint8) for k in range(4)])
    img = s.add_image(LumenImage(vol, name="labelled", axes="ZYX"))
    img.metadata["slice_labels"] = ["a", "b", "c", "d"]
    img.display.z = 1
    s.run_op("stacks.delete_slice", {"axis": "Z"})
    assert img.nz == 3 and img.metadata["slice_labels"] == ["a", "c", "d"]
    assert s.undo(img.id) == "Delete Slice"
    assert img.nz == 4 and img.metadata["slice_labels"] == ["a", "b", "c", "d"]  # the labels came back with their planes
    assert s.redo(img.id) == "Delete Slice" and img.metadata["slice_labels"] == ["a", "c", "d"]
    s.undo(img.id)
    # a pure metadata edit (Set Slice Label) restores through the same snapshot
    hist = s.histories[img.id]
    hist.push(img, "Set slice label")
    set_slice_label(img, "renamed", 0, 1, 0)
    assert img.metadata["slice_labels"] == ["a", "renamed", "c", "d"]
    assert hist.do_undo(img) == "Set slice label" and img.metadata["slice_labels"] == ["a", "b", "c", "d"]
    assert hist.do_redo(img) == "Set slice label" and img.metadata["slice_labels"] == ["a", "renamed", "c", "d"]
    # the snapshot is a deep copy: editing the live list afterwards must not leak into the history
    hist.push(img, "x")
    img.metadata["slice_labels"][0] = "mutated"
    hist.do_undo(img)
    assert img.metadata["slice_labels"][0] == "a"
    # Make Binary sets the binary-mask flag; undoing it clears the flag again
    s.run_op("binary.make", {"method": "Otsu"})
    assert img.lut_is_binary and set(np.unique(img.data)) <= {0, 255}
    assert s.undo(img.id) == "Make Binary"
    assert not img.lut_is_binary and not set(np.unique(img.data)) <= {0, 255}


def test_merge_channels_promotes_mixed_dtypes():
    s = Session()
    a = s.add_image(LumenImage(np.full((4, 4), 200, np.uint8), name="a"))
    b = s.add_image(LumenImage(np.full((4, 4), 1234.5, np.float32), name="b"))
    out = s.run_op("color.merge", {"c1": a.id, "c2": b.id, "keep_source": True})
    m = s.images[out["new_images"][0]]
    assert m.dtype == np.float32 and m.nc == 2
    assert float(m.data[0, 0, 0, 0, 0]) == 200.0 and float(m.data[0, 0, 1, 0, 0]) == 1234.5  # was cast to uint8 (1234.5 → 210)
    u = s.add_image(LumenImage(np.full((4, 4), 1000, np.uint16), name="u"))
    out = s.run_op("color.merge", {"c1": a.id, "c2": u.id, "keep_source": True})
    m = s.images[out["new_images"][0]]
    assert m.dtype == np.uint16 and int(m.data[0, 0, 0, 0, 0]) == 200 and int(m.data[0, 0, 1, 0, 0]) == 1000


# ---------------------------------------------------------------- code-review regressions: roi.py / edit.py / color.py
def test_line_masks_follow_pixel_centre_convention():
    # pixel (i, j) covers [i, i+1): a 1-px line through y = 3.5 (the centre of row 3) marks row 3 – round() gave row 4
    import math
    for y in (1.5, 2.5, 3.5, 4.5, 5.5, 3.0, 4.0, 3.6):
        m = roimod.to_mask({"type": "line", "x1": 0.5, "y1": y, "x2": 6.5, "y2": y, "width": 1}, (8, 8))
        assert np.unique(np.nonzero(m)[0]).tolist() == [math.floor(y)], y
    thick = roimod.to_mask({"type": "line", "x1": 0.5, "y1": 3.5, "x2": 6.5, "y2": 3.5, "width": 3}, (8, 8))
    poly = roimod.to_mask({"type": "polygon", "points": [[0.5, 2], [6.5, 2], [6.5, 5], [0.5, 5]]}, (8, 8))
    assert np.array_equal(thick, poly) and np.unique(np.nonzero(thick)[0]).tolist() == [2, 3, 4]   # was rows 2..5
    vertical = roimod.to_mask({"type": "line", "x1": 3.5, "y1": 0.5, "x2": 3.5, "y2": 6.5, "width": 3}, (8, 8))
    assert np.unique(np.nonzero(vertical)[1]).tolist() == [2, 3, 4]
    # Line to Area and Draw now mark the row the viewport shows (and Plot Profile / Measure sample)
    s = Session()
    img = s.add_image(LumenImage(np.repeat(np.arange(8, dtype=np.uint8)[:, None] * 10, 8, axis=1)))
    line = {"type": "line", "x1": 0.5, "y1": 3.5, "x2": 6.5, "y2": 3.5, "width": 1}
    area = s.run_op("selection.line_to_area", {}, roi=line)["roi"]
    assert np.unique(np.nonzero(roimod.to_mask(area, (8, 8)))[0]).tolist() == [3]
    s.run_op("edit.set_colors", {"foreground": 255, "background": 0})
    s.run_op("edit.draw", {"line_width": 1}, roi=line)
    assert img.data[0, 0, 0, 3, 1] == 255 and img.data[0, 0, 0, 4, 1] == 40


def test_mask_roi_outlines_cover_every_part():
    s = Session()
    p = np.zeros((40, 40), np.uint8); p[5:15, 5:15] = 255; p[25:33, 25:33] = 255
    s.add_image(LumenImage(p))
    r = s.run_op("selection.create", {})["roi"]
    assert r["type"] == "mask" and roimod.to_mask(r, (40, 40)).sum() == 164
    assert len(roimod.outlines(r)) == 2 and len(roimod.polygon_points(r)) == 41       # polygon_points: largest part only
    assert roimod.perimeter(r) == pytest.approx(40 + 32, abs=1e-9)                   # both parts: ImageJ ShapeRoi.getLength = pixel-edge outline length (was the 0.5-level contour, 69.66)
    s.measurements = ["area", "perimeter", "shape_descriptors"]
    s.run_op("analyze.measure", {}, roi=r)
    row = s.results.rows[-1]
    assert row["Area"] == 164 and row["Perim."] == pytest.approx(72.0, abs=1e-9) and row["Circ."] < 1   # Circ. clipped to 1.0 before
    rot = s.run_op("selection.rotate", {"angle": 0}, roi=r)["roi"]
    assert rot["type"] == "mask" and np.array_equal(roimod.to_mask(rot, (40, 40)), roimod.to_mask(r, (40, 40)))
    assert roimod.to_mask(s.run_op("selection.rotate", {"angle": 90}, roi=r)["roi"], (60, 60)).sum() == 164
    hull = roimod.to_mask(s.run_op("selection.convex_hull", {}, roi=r)["roi"], (40, 40))
    assert hull.sum() > 164 and hull[5, 5] and hull[28, 28]
    interp = s.run_op("selection.interpolate", {}, roi=r)["roi"]
    assert interp["type"] == "mask" and roimod.to_mask(interp, (40, 40))[28:30, 28:30].all()   # second part kept
    with pytest.raises(ValueError, match="Composite"):
        s.run_op("selection.area_to_line", {}, roi=r)
    # a ring: the perimeter includes the hole boundary (ImageJ ShapeRoi.getLength) and rotation keeps the hole
    ring = np.zeros((30, 30), bool); ring[5:25, 5:25] = True; ring[10:20, 10:20] = False
    rr = roimod.from_mask(ring)
    assert roimod.perimeter(rr) == pytest.approx(80 + 40, abs=1e-9)   # pixel-edge outline of the square plus its hole (ImageJ ShapeRoi.getLength)
    assert np.array_equal(roimod.to_mask(s.run_op("selection.rotate", {"angle": 0}, roi=rr)["roi"], (30, 30)), ring)
    # single-part masks are unchanged: same perimeter as before, rotation still yields a polygon
    single = roimod.from_mask(np.pad(np.ones((10, 10), bool), 2))
    assert roimod.perimeter(single) == pytest.approx(40 - 4 * (2 - 2 ** 0.5), abs=1e-9)   # one hole-free blob: ImageJ's traced perimeter (corners count √2)
    assert s.run_op("selection.rotate", {"angle": 30}, roi=single)["roi"]["type"] == "polygon"
    assert s.run_op("selection.area_to_line", {}, roi=single)["roi"]["type"] == "polyline"


def test_off_image_rois_are_empty_not_errors():
    m = np.zeros((20, 20), bool); m[5:10, 5:10] = True
    base = roimod.from_mask(m)
    ys, xs = np.nonzero(m)
    for dx, dy in [(0, -15), (-15, 0), (0, 17), (17, 0), (0, 25), (-3, -3), (18, 18)]:
        got = roimod.to_mask(roimod.translate(base, dx, dy), (20, 20))                  # broadcast ValueError before
        ref = np.zeros((20, 20), bool)
        for y, x in zip(ys + dy, xs + dx):
            if 0 <= y < 20 and 0 <= x < 20:
                ref[y, x] = True
        assert np.array_equal(got, ref), (dx, dy)
    assert not roimod.to_mask({"type": "rect", "x": -150, "y": 0, "w": 100, "h": 100}, (100, 100)).any()   # selected the left half
    assert not roimod.to_mask({"type": "rect", "x": 0, "y": -15, "w": 100, "h": 5}, (20, 100)).any()
    assert roimod.to_mask({"type": "rect", "x": -5, "y": -5, "w": 10, "h": 10}, (20, 20)).sum() == 25
    s = Session()
    s.add_image(LumenImage(np.full((20, 20), 9, np.uint8)))
    s.measurements = ["area", "mean"]
    s.run_op("analyze.measure", {}, roi=roimod.translate(base, 0, -15))
    assert s.results.rows[-1]["Area"] == 0
    s.run_op("analyze.measure", {}, roi={"type": "rect", "x": -150, "y": 0, "w": 100, "h": 100})
    assert s.results.rows[-1]["Area"] == 0                                              # was 5000
    poly = {"type": "polygon", "points": [[-10, -10], [-3, -10], [-3, -3], [-10, -3]]}
    assert roimod.contains_point(poly, -5, -5) and not roimod.contains_point(poly, -2, -5)   # 'negative dimensions' before
    rect = {"type": "rect", "x": 2, "y": 2, "w": 4, "h": 4}
    assert roimod.contains_point(rect, 3.5, 3.5) and not roimod.contains_point(rect, 6.5, 3.5)
    shifted = roimod.translate(base, -3, -3)
    assert roimod.contains_point(shifted, 3, 3) and not roimod.contains_point(shifted, 1, 1)
    scaled = s.run_op("selection.scale", {"x_scale": 2, "y_scale": 2}, roi=roimod.translate(base, 0, -15))["roi"]
    assert scaled["type"] == "mask" and (scaled["w"], scaled["h"]) == (10, 10)
    assert roimod.scale_roi(roimod.translate(base, 0, -15), 2, 2)["y"] == -20                 # scaled about the origin
    with pytest.raises(ValueError, match="empty"):                                           # from_mask gave None (TypeError downstream)
        roimod.scale_roi(roimod.from_mask(np.pad(np.ones((1, 1), bool), 3)), 0.2, 0.2)


def test_merge_channels_rejects_multichannel_and_rgb_sources():
    s = Session()
    rgb = np.zeros((16, 16, 3), np.uint8); rgb[..., 1] = 200
    a = s.add_image(LumenImage(rgb, name="rgb", rgb=True))
    b = s.add_image(LumenImage(np.full((16, 16), 50, np.uint8), name="g"))
    with pytest.raises(ValueError, match="single-channel"):
        s.run_op("color.merge", {"c1": a.id, "c2": b.id})                                 # silently kept the red samples
    assert a.id in s.images and b.id in s.images                                            # nothing closed on failure
    two = s.add_image(LumenImage(np.zeros((2, 16, 16), np.uint8), name="two", axes="CYX"))
    with pytest.raises(ValueError, match="2-channel"):
        s.run_op("color.merge", {"c1": b.id, "c2": two.id})
    ids = s.run_op("color.split", {}, image_id=a.id)["new_images"]                          # the remedy the message names
    out = s.run_op("color.merge", {"c1": ids[0], "c2": ids[1], "c3": ids[2], "keep_source": True})
    comp = s.images[out["new_images"][0]]
    assert comp.nc == 3 and int(comp.data[0, 0, 1].max()) == 200


def test_label_to_rois_keeps_every_part():
    lab = np.zeros((30, 30), np.int32); lab[2:12, 2:12] = 1; lab[20:25, 20:25] = 1
    (r,) = roimod.label_to_rois(lab)
    assert r["type"] == "mask" and roimod.to_mask(r, (30, 30)).sum() == 125 and r["name"].startswith("0001-")   # polygon of 100 px before
    diag = np.zeros((6, 6), np.int32); diag[1, 1] = 1; diag[2, 2] = 1                      # 8-connected pair → two contours
    (r,) = roimod.label_to_rois(diag)
    assert r["type"] == "mask" and roimod.to_mask(r, (6, 6)).sum() == 2
    hp = np.zeros((30, 30), np.int32); hp[2:20, 2:20] = 1; hp[8:12, 8:12] = 0; hp[24:27, 24:27] = 1   # hole + part: Euler number 1
    (r,) = roimod.label_to_rois(hp)
    assert r["type"] == "mask" and roimod.to_mask(r, (30, 30)).sum() == 18 * 18 - 16 + 9
    lab[20:25, 20:25] = 0
    (r,) = roimod.label_to_rois(lab)                                                        # a single solid part stays a polygon
    assert r["type"] == "polygon" and roimod.to_mask(r, (30, 30)).sum() == 100


def test_fill_clear_draw_clamp_out_of_range_values():
    s = Session()
    img = s.add_image(LumenImage(np.full((16, 16), 100, np.uint8)))
    rect = {"type": "rect", "x": 2, "y": 2, "w": 5, "h": 5}
    s.run_op("edit.set_colors", {"foreground": 300, "background": -5})                    # picked for another image depth
    s.run_op("edit.fill", {}, roi=rect)
    assert img.data[0, 0, 0, 3, 3] == 255                                                   # OverflowError before (ImageJ clamps)
    s.run_op("edit.clear", {}, roi=rect)
    assert img.data[0, 0, 0, 3, 3] == 0
    s.run_op("edit.clear_outside", {}, roi=rect)
    assert img.data[0, 0, 0, 0, 0] == 0
    s.run_op("edit.draw", {}, roi=rect)
    assert img.data[0, 0, 0, 2, 2] == 255 and img.data[0, 0, 0, 3, 3] == 0
    s.run_op("edit.fill", {"value": 300}, roi=rect)                                         # explicit value bypasses the setting
    assert img.data[0, 0, 0, 3, 3] == 255
    u16 = s.add_image(LumenImage(np.zeros((16, 16), np.uint16)))
    s.run_op("edit.set_colors", {"foreground": 70000, "background": -5})
    s.run_op("edit.fill", {}, roi=rect, image_id=u16.id)
    assert u16.data[0, 0, 0, 3, 3] == 65535
    f32 = s.add_image(LumenImage(np.zeros((16, 16), np.float32)))
    s.run_op("edit.set_colors", {"foreground": 300, "background": -5})
    s.run_op("edit.fill", {}, roi=rect, image_id=f32.id)
    s.run_op("edit.clear_outside", {}, roi=rect, image_id=f32.id)
    assert f32.data[0, 0, 0, 3, 3] == 300 and f32.data[0, 0, 0, 0, 0] == -5                 # float keeps the value


def test_particles_need_threshold_or_binary_unless_auto_threshold():
    """docs/USABILITY.md M2: no silent auto threshold on an unthresholded gray image (ImageJ raises too)."""
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    img = s.active
    assert not img.is_binary and img.id not in s.thresholds
    with pytest.raises(ValueError, match="thresholded or binary image"):
        s.run_op("analyze.particles", {"show": "Nothing", "add_to_manager": False})
    assert not s.results.rows and "Particle Summary" not in s.tables  # nothing ran
    # explicit opt-in restores the old behaviour (Default auto threshold per plane)
    out = s.run_op("analyze.particles", {"show": "Nothing", "add_to_manager": False, "auto_threshold": True})
    n_auto = s.tables["Particle Summary"]["rows"][0][1]
    assert n_auto > 10 and "particles" in out["message"]
    # a threshold set on the image is used as before
    s.run_op("threshold.compute", {"method": "Default", "dark_background": True})
    s.run_op("analyze.particles", {"show": "Nothing", "add_to_manager": False})
    assert s.tables["Particle Summary"]["rows"][0][1] == n_auto
    # a binary image never needs a threshold; a stack with one non-binary plane is refused when the whole stack is processed
    bin_img = _particle_image(s)
    s.run_op("analyze.particles", {"show": "Nothing", "add_to_manager": False}, image_id=bin_img.id)
    assert s.tables["Particle Summary"]["rows"][0][1] == 3
    vol = np.zeros((2, 20, 20), np.uint8)
    vol[0, 5:10, 5:10] = 255
    vol[1, 5:10, 5:10] = 100  # gray plane
    stk = s.add_image(LumenImage(vol, name="mixed.tif", axes="ZYX"))
    s.run_op("analyze.particles", {"show": "Nothing", "add_to_manager": False}, image_id=stk.id)  # plane z=0 is binary
    with pytest.raises(ValueError, match="thresholded or binary image"):
        s.run_op("analyze.particles", {"show": "Nothing", "add_to_manager": False, "all_planes": True}, image_id=stk.id)
    spec = REGISTRY.get("analyze.particles")
    p = next(x for x in spec.params if x.name == "auto_threshold")
    assert p.type == "bool" and p.default is False and p.advanced
