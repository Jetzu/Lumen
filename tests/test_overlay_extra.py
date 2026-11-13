"""Tests for Overlay ▸ Labels… options (analyze.label, lumen/core/ops/extras.py) and Overlay ▸ Measure Overlay
(overlay.measure, lumen/core/ops/overlay_extra.py)."""
import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.registry import REGISTRY
from lumen.core.session import Session


def _session(stack=False):
    s = Session()
    if stack:
        vol = np.zeros((3, 40, 50), np.uint16)
        for z in range(3):
            vol[z, 10:20, 10:20] = 100 * (z + 1)
        s.add_image(LumenImage(vol, name="stack.tif", axes="ZYX"))
    else:
        a = np.zeros((40, 50), np.uint8)
        a[10:20, 10:20] = 200
        a[25:35, 30:45] = 50
        s.add_image(LumenImage(a, name="two.tif"))
    return s


def _texts(img):
    return [it for it in img.overlay if it.get("type") == "text"]


# ---------------------------------------------------------------- Labels…
def test_label_spec_and_menu():
    spec = REGISTRY.ops["analyze.label"]
    names = [p.name for p in spec.params]
    for key in ("what", "column", "decimals", "use_names", "font_size", "bold", "color", "background", "at", "source", "only_this_image"):
        assert key in names, key
    assert next(p for p in spec.params if p.name == "what").choices == ["Index", "Name", "Measurement", "Value"]
    assert REGISTRY.ops["overlay.measure"].to_dict()["menu"] == "Image/Overlay"
    assert REGISTRY.ops["edit.options_overlay"].to_dict()["menu"] == "Edit/Options"


def test_label_defaults_follow_overlay_options_and_explicit_values_win():
    s = _session()
    img = s.active
    s.roi_manager.add({"type": "rect", "x": 10, "y": 10, "w": 10, "h": 10, "name": "square"}, image_id=img.id)
    s.roi_manager.add({"type": "rect", "x": 30, "y": 25, "w": 15, "h": 10}, image_id=img.id)
    s.run_op("edit.options_overlay", {"overlay_label_font_size": 20, "overlay_label_color": "#00ff00"})
    s.run_op("analyze.label", {"what": "Index"})
    t = _texts(img)
    assert [it["text"] for it in t] == ["1", "2"]
    assert all(it["font_size"] == 20 and it["color"] == "#00ff00" and it["bold"] is False and it["background"] is False for it in t)
    assert t[0]["x"] == pytest.approx(15 - 0.25 * 20) and t[0]["y"] == pytest.approx(15 - 10)  # centred on the centroid (15, 15)
    assert t[0]["roi_id"] == s.roi_manager.rois[0]["id"] and "position" not in t[0]
    auto = s.roi_manager.rois[1]["name"]  # the manager auto-names ROIs ('yyyy-xxxx'), like ImageJ
    assert auto and auto != "2"
    s.run_op("analyze.label", {"what": "Index", "use_names": True, "font_size": 8, "color": "red", "bold": True, "background": True})
    t = _texts(img)[2:]
    assert [it["text"] for it in t] == ["square", auto]  # use names: the name when the ROI has one, else the index
    assert all(it["font_size"] == 8 and it["color"] == "#ff0000" and it["bold"] is True and it["background"] is True for it in t)
    s.run_op("analyze.label", {"what": "Name"})
    assert [it["text"] for it in _texts(img)[4:]] == ["square", auto]
    del s.roi_manager.rois[1]["name"]
    s.run_op("analyze.label", {"what": "Index", "use_names": True})
    assert [it["text"] for it in _texts(img)[6:]] == ["square", "2"]  # a nameless ROI falls back to its index
    assert s.undo(img.id) is not None and len(_texts(img)) == 6


def test_label_at_bounding_box_vs_centroid():
    s = _session()
    img = s.active
    # an L shape: bounding box centre (20, 20), centroid pulled towards the corner
    s.roi_manager.add({"type": "polygon", "points": [[10, 10], [30, 10], [30, 15], [15, 15], [15, 30], [10, 30]]}, image_id=img.id)
    s.run_op("analyze.label", {"what": "Index", "at": "Bounding box", "font_size": 10})
    bb = _texts(img)[-1]
    assert bb["x"] + 0.25 * 10 == pytest.approx(20) and bb["y"] + 5 == pytest.approx(20)
    s.run_op("analyze.label", {"what": "Index", "at": "Centroid", "font_size": 10})
    ce = _texts(img)[-1]
    assert ce["x"] + 0.25 * 10 < 19 and ce["y"] + 5 < 19


def test_label_value_measurement_decimals_and_rgb():
    s = _session()
    img = s.active
    s.roi_manager.add({"type": "rect", "x": 10, "y": 10, "w": 10, "h": 10}, image_id=img.id)
    s.run_op("analyze.label", {"what": "Value"})
    assert _texts(img)[-1]["text"] == "200"
    s.run_op("analyze.label", {"what": "Measurement", "column": "Mean", "decimals": 1})
    assert _texts(img)[-1]["text"] == "200.0"
    s.run_op("analyze.label", {"what": "Measurement", "column": "Area", "decimals": 0})
    assert _texts(img)[-1]["text"] == "100"
    f = s.add_image(LumenImage(np.full((20, 20), 1.25, np.float32), name="f.tif"))
    s.roi_manager.clear()
    s.roi_manager.add({"type": "oval", "x": 2, "y": 2, "w": 10, "h": 10}, image_id=f.id)
    s.run_op("analyze.label", {"what": "Value", "decimals": 3})
    assert _texts(f)[-1]["text"] == "1.250"
    rgb = s.add_image(LumenImage(np.tile(np.array([10, 20, 30], np.uint8), (8, 8, 1)), name="rgb.png", axes="YXS", rgb=True))
    s.roi_manager.clear()
    s.roi_manager.add({"type": "rect", "x": 1, "y": 1, "w": 4, "h": 4}, image_id=rgb.id)
    s.run_op("analyze.label", {"what": "Value"})
    assert _texts(rgb)[-1]["text"] == "10,20,30"


def test_label_uses_the_rois_plane_and_pins_the_label():
    s = _session(stack=True)
    img = s.active
    s.roi_manager.add({"type": "rect", "x": 10, "y": 10, "w": 10, "h": 10, "name": "deep"}, image_id=img.id, position={"t": 0, "z": 2, "c": 0})
    s.run_op("analyze.label", {"what": "Value"})
    it = _texts(img)[-1]
    assert it["text"] == "300" and it["position"] == {"t": 0, "z": 2, "c": 0}
    s.run_op("analyze.label", {"what": "Measurement", "column": "Mean"})
    assert _texts(img)[-1]["text"] == "300.00"


def test_label_overlay_items_as_source():
    s = _session()
    img = s.active
    with pytest.raises(ValueError, match="no ROI-shaped items"):
        s.run_op("analyze.label", {"source": "Overlay"})
    img.overlay = [{"id": "r1", "type": "rect", "x": 30, "y": 25, "w": 15, "h": 10, "name": "ov"}, {"id": "t1", "type": "text", "x": 0, "y": 0, "text": "x"}]
    s.run_op("analyze.label", {"source": "Overlay", "what": "Name"})
    labels = [it for it in _texts(img) if it.get("roi_id")]
    assert len(labels) == 1 and labels[0]["text"] == "ov" and labels[0]["roi_id"] == "r1"
    s.roi_manager.clear()
    with pytest.raises(ValueError, match="ROI Manager is empty"):
        s.run_op("analyze.label", {"source": "ROI Manager"})


# ---------------------------------------------------------------- Measure Overlay
def test_measure_overlay_rows_labels_and_planes():
    s = _session(stack=True)
    img = s.active
    s.measurements = ["area", "mean", "stack_position"]
    img.overlay = [
        {"id": "a", "type": "rect", "x": 10, "y": 10, "w": 10, "h": 10, "name": "deep", "position": {"t": 0, "z": 2, "c": 0}},
        {"id": "b", "type": "oval", "x": 10, "y": 10, "w": 10, "h": 10},
        {"id": "c", "type": "point", "points": [[15, 15], [0, 0]]},
        {"id": "d", "type": "text", "x": 0, "y": 0, "text": "ignored"},
    ]
    s.results.append([{"Label": "earlier", "Mean": 1.0}])
    out = s.run_op("overlay.measure")
    assert out["results_added"] == 4 and len(s.results.rows) == 5  # appended by default
    rows = s.results.rows[1:]
    assert rows[0]["Label"] == "stack.tif:deep" and rows[0]["Mean"] == pytest.approx(300) and rows[0]["Slice"] == 3  # its own plane (z = 2)
    assert rows[1]["Label"] == "stack.tif:2" and rows[1]["Mean"] == pytest.approx(100) and rows[1]["Slice"] == 1  # unpinned: the current plane
    assert rows[2]["Label"] == "stack.tif:3 (pt 1)" and rows[2]["Value"] == 100 and rows[3]["Value"] == 0
    out = s.run_op("overlay.measure", {"append": False})
    assert len(s.results.rows) == 4 and s.results.rows[0]["Label"] == "stack.tif:deep"
    out = s.run_op("overlay.measure", {"append": False, "current_plane_only": True})
    assert out["results_added"] == 3 and all(r["Label"] != "stack.tif:deep" for r in s.results.rows)
    img.display.z = 2
    out = s.run_op("overlay.measure", {"append": False, "current_plane_only": True})
    assert out["results_added"] == 4 and s.results.rows[1]["Mean"] == pytest.approx(300)  # the unpinned oval now measures z = 2 too


def test_measure_overlay_honours_set_measurements_options_and_errors():
    s = _session()
    img = s.active
    with pytest.raises(ValueError, match="no ROI-shaped items"):
        s.run_op("overlay.measure")
    img.overlay = [{"id": "a", "type": "rect", "x": 10, "y": 10, "w": 10, "h": 10, "name": "sq"}]
    s.measurements = ["area", "mean", "min", "max"]
    s.measure_options["limit_to_threshold"] = True
    s.thresholds[img.id] = {"lower": 100, "upper": 255}
    s.run_op("overlay.measure", {"append": False})
    assert s.results.rows[0]["Mean"] == pytest.approx(200) and s.results.rows[0]["Label"] == "two.tif:sq"
    s.measure_options["limit_to_threshold"] = False
    s.measure_options["display_label"] = False
    s.run_op("overlay.measure", {"append": False})
    assert "Label" not in s.results.rows[0] and s.results.rows[0]["Area"] == 100
    # the selection is not needed and is ignored; the macro recorder keeps the call replayable
    s.run_op("overlay.measure", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5})
    assert s.macro[-1].startswith('lumen.run("overlay.measure"')
