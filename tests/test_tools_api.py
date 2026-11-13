"""Plugin tools (``@op(..., tool=True)``, lumen/core/registry.py + lumen/core/ops/tools_ops.py): registry fields, the interaction
parameters, the two built-in tools on 8-bit / 16-bit / RGB images and stacks, their exposure in /api/ops, and the text-file
router used by the script editor (lumen/server/extra/textfiles.py)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core.image import LumenImage
from lumen.core.registry import REGISTRY, TOOL_PARAMS, OpResult, Param, op
from lumen.core.session import Session
from lumen.server.app import create_app


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


# ---------------------------------------------------------------- registry
def test_tool_fields_in_registry_and_api():
    spec = REGISTRY.ops["tools.spot_counter"]
    assert spec.tool and spec.tool_icon == "◎" and spec.tool_cursor == "crosshair" and spec.to_dict()["menu"] == "Plugins/Tools"
    d = spec.to_dict()
    assert d["tool"] is True and d["tool_icon"] == "◎" and d["tool_cursor"] == "crosshair"
    assert REGISTRY.ops["filters.gaussian"].to_dict()["tool"] is False
    # the interaction params are accepted by coerce_params but are not dialog params
    assert [p.name for p in spec.params] == ["marker_size", "color", "all_planes"]
    assert {p.name for p in TOOL_PARAMS} == {"x", "y", "x2", "y2", "button", "shift", "alt", "ctrl", "event"}
    out = spec.coerce_params({"x": "10", "y": 20, "event": "drag-end", "x2": 30, "shift": "true"})
    assert out["x"] == 10.0 and out["y"] == 20.0 and out["x2"] == 30.0 and out["y2"] is None
    assert out["shift"] is True and out["alt"] is False and out["event"] == "drag-end" and out["button"] == 0 and out["marker_size"] == 6
    with pytest.raises(ValueError):
        spec.coerce_params({"event": "hover"})
    with pytest.raises(ValueError, match="unknown parameter 'x' for filters.gaussian"):  # non-tools reject them (E7-8)
        REGISTRY.ops["filters.gaussian"].coerce_params({"x": 1})
    c = TestClient(create_app(Session()))
    ops = {o["id"]: o for o in c.get("/api/ops").json()}
    assert ops["tools.spot_counter"]["tool"] is True and ops["tools.measure_distance"]["tool"] is True and ops["tools.measure_distance"]["tool_icon"] == "↔"
    assert all("tool" in o and "tool_icon" in o and "tool_cursor" in o and "tool_color" in o and "frame" in o for o in ops.values())
    assert sum(1 for o in ops.values() if o["tool"]) >= 2
    # plugin frames (frame=True / kind="frame") and the icon colour of macro tools are part of the same record
    assert ops["frames.image_statistics"]["frame"] is True and ops["tools.spot_counter"]["frame"] is False and ops["tools.spot_counter"]["tool_color"] == ""
    assert REGISTRY.ops["frames.image_statistics"].kind == "analysis" and REGISTRY.ops["frames.image_statistics"].preview is False


def test_frame_kind_and_tool_color_in_decorator():
    @op("testframe.f", "F", "Plugins/Test", kind="frame", requires_image=False, params=[Param("n", "int", 2)])
    def f(ctx, n=2):
        return OpResult(table={"name": "F", "columns": ["n"], "rows": [[n]]}, message="ok")

    @op("testframe.t", "T2", "Plugins/Test", kind="analysis", tool=True, tool_color="#ff0000", requires_image=False)
    def t(ctx, x=0.0, y=0.0, x2=None, y2=None, button=0, shift=False, alt=False, ctrl=False, event="click"):
        return OpResult()

    try:
        spec = REGISTRY.ops["testframe.f"]
        assert spec.kind == "analysis" and spec.frame and spec.preview is False and spec.to_dict()["frame"] is True and spec.to_dict()["preview"] is False
        assert Session().run_op("testframe.f", {"n": 5})["table"]["rows"] == [[5]]
        assert REGISTRY.ops["testframe.t"].to_dict()["tool_color"] == "#ff0000" and REGISTRY.ops["testframe.t"].tool_icon == "⚙"
        with pytest.raises(ValueError, match="unknown op kind"):
            op("testframe.bad", "Bad", "Plugins/Test", kind="window")(lambda ctx: None)
    finally:
        REGISTRY.ops.pop("testframe.f", None)
        REGISTRY.ops.pop("testframe.t", None)
        REGISTRY.ops.pop("testframe.bad", None)


def test_tool_decorator_defaults_and_recorder():
    @op("testtool.t", "T", "Plugins/Test", kind="analysis", tool=True, requires_image=False)
    def t(ctx, x=0.0, y=0.0, x2=None, y2=None, button=0, shift=False, alt=False, ctrl=False, event="click"):
        return OpResult(message=f"{event} {x} {y} {x2} {button} {shift}")

    try:
        spec = REGISTRY.ops["testtool.t"]
        assert spec.tool_icon == "⚙" and spec.tool_cursor == "crosshair"  # defaults for a bare tool=True
        s = Session()
        out = s.run_op("testtool.t", {"x": 3, "y": 4, "event": "drag-end", "x2": 5, "button": 2, "shift": True})
        assert out["message"] == "drag-end 3.0 4.0 5.0 2 True"
        # every interaction is recorded (y2 = None is left out) so a macro replays the drag
        assert s.macro[-1] == 'lumen.run("testtool.t", x=3.0, y=4.0, x2=5.0, button=2, shift=True, alt=False, ctrl=False, event=\'drag-end\')'
        assert s.run_op("testtool.t")["message"] == "click 0.0 0.0 None 0 False"
    finally:
        REGISTRY.ops.pop("testtool.t", None)


# ---------------------------------------------------------------- Spot Click Counter
def test_spot_counter_marks_counts_and_removes(session):
    img = session.active
    out = session.run_op("tools.spot_counter", {"x": 10, "y": 12, "event": "click"})
    assert out["results_added"] == 1 and "1 spot" in out["message"]
    assert len(img.overlay) == 1 and img.overlay[0]["type"] == "point" and img.overlay[0]["points"] == [[10.0, 12.0]]
    assert img.overlay[0]["position"] == {"t": 0, "z": 0, "c": 0} and img.overlay[0]["id"] == "spot-1" and img.overlay[0]["point_size"] == 6
    row = session.results.rows[-1]
    assert row["Label"] == "Spot 1" and row["X"] == 10 and row["Y"] == 12 and row["Value"] == float(img.data[0, 0, 0, 12, 10]) and row["Count"] == 1
    # a drag marks the release point; drag-start does nothing
    out = session.run_op("tools.spot_counter", {"x": 40, "y": 50, "x2": 44, "y2": 52, "event": "drag-end"})
    assert len(img.overlay) == 2 and img.overlay[1]["points"] == [[44.0, 52.0]] and "2 spot" in out["message"] and row != session.results.rows[-1]
    assert session.run_op("tools.spot_counter", {"x": 1, "y": 1, "event": "drag-start"}).get("message") is None and len(img.overlay) == 2
    # ⇧-click only reports
    out = session.run_op("tools.spot_counter", {"x": 5, "y": 5, "shift": True})
    assert len(img.overlay) == 2 and out.get("results_added") is None and "2 spot" in out["message"]
    # ⌥-click removes the nearest spot (within reach), ids stay unique afterwards
    out = session.run_op("tools.spot_counter", {"x": 11, "y": 11, "alt": True})
    assert len(img.overlay) == 1 and img.overlay[0]["id"] == "spot-2" and "Removed Spot 1" in out["message"]
    assert "No spot near" in session.run_op("tools.spot_counter", {"x": 200, "y": 200, "alt": True})["message"] and len(img.overlay) == 1
    session.run_op("tools.spot_counter", {"x": 70, "y": 70})
    assert [o["id"] for o in img.overlay] == ["spot-2", "spot-3"] and session.results.rows[-1]["Count"] == 2
    # analysis kind: pixels untouched, nothing to undo; other overlay items survive
    assert session.histories[img.id].labels()["undo"] == []
    img.overlay.append({"type": "text", "id": "note", "x": 1, "y": 1, "text": "hi"})
    session.run_op("tools.spot_counter", {"x": 75, "y": 70, "alt": True})
    assert sorted(o["id"] for o in img.overlay) == ["note", "spot-2"]


def test_spot_counter_on_stack_16bit_and_rgb():
    s = Session()
    stack = s.add_image(LumenImage((np.arange(4 * 2 * 8 * 8).reshape(4, 2, 8, 8) * 100).astype(np.uint16), name="stack", axes="ZCYX"))
    s.update_display(stack.id, {"z": 2, "c": 1})
    s.run_op("tools.spot_counter", {"x": 3, "y": 4})
    sp = stack.overlay[0]
    assert sp["position"] == {"t": 0, "z": 2, "c": 1} and s.results.rows[-1]["Value"] == float(stack.data[0, 2, 1, 4, 3]) and s.results.rows[-1]["Z"] == 2
    s.run_op("tools.spot_counter", {"x": 3, "y": 4, "all_planes": True})
    assert "position" not in stack.overlay[1]
    # a spot on another plane is out of reach for ⌥-click on this one
    s.update_display(stack.id, {"z": 0})
    assert "No spot to remove" in s.run_op("tools.spot_counter", {"x": 3, "y": 4, "alt": True, "all_planes": False})["message"] or len(stack.overlay) == 1
    rgb = s.add_image(LumenImage(np.random.default_rng(0).integers(0, 255, (16, 16, 3), np.uint8), name="rgb", rgb=True))
    assert rgb.rgb
    s.run_op("tools.spot_counter", {"x": 2, "y": 3})
    assert s.results.rows[-1]["Value"] == ",".join(f"{v:g}" for v in rgb.data[0, 0, :, 3, 2])
    flt = s.add_image(LumenImage(np.linspace(0, 1, 64, dtype=np.float32).reshape(8, 8), name="f"))
    s.run_op("tools.spot_counter", {"x": 7, "y": 7})
    assert s.results.rows[-1]["Value"] == pytest.approx(1.0)
    assert s.run_op("tools.spot_counter", {"x": 50, "y": 50})["results_added"] == 1 and s.results.rows[-1]["Value"] is None  # outside


# ---------------------------------------------------------------- Measure Distance Tool
def test_measure_distance_tool_reports_calibrated_length(session):
    img = session.active
    img.calibration.pixel_width = img.calibration.pixel_height = 0.5
    img.calibration.unit = "um"
    out = session.run_op("tools.measure_distance", {"x": 10, "y": 10, "x2": 40, "y2": 50, "event": "drag-end"})
    row = session.results.rows[-1]
    assert row["Length"] == pytest.approx(25.0) and row["Unit"] == "um" and row["X2"] == 40 and row["Label"] == "Distance 1"
    assert row["Angle"] == pytest.approx(-53.13, abs=0.01)  # ImageJ convention: counter-clockwise, y up; down-right is negative
    assert out["roi"]["type"] == "line" and out["roi"]["x2"] == 40 and "25.00 um" in out["message"]
    line = img.overlay[-1]
    assert line["type"] == "line" and line["label"] == "25.00 um" and line["position"] == {"t": 0, "z": 0, "c": 0} and line["id"] == "distance-1"
    # a plain click reports position and value, a drag-start nothing
    out = session.run_op("tools.measure_distance", {"x": 3, "y": 4})
    assert out.get("results_added") is None and "drag" in out["message"] and f"= {float(img.data[0, 0, 0, 4, 3])}" in out["message"]
    assert session.run_op("tools.measure_distance", {"x": 3, "y": 4, "event": "drag-start"}).get("message") is None
    # without the overlay line; decimals honoured; numbering continues
    n = len(img.overlay)
    out = session.run_op("tools.measure_distance", {"x": 0, "y": 0, "x2": 3, "y2": 4, "event": "drag-end", "add_to_overlay": False, "decimals": 1})
    assert len(img.overlay) == n and session.results.rows[-1]["Length"] == pytest.approx(2.5) and session.results.rows[-1]["Label"] == "Distance 2"
    assert "2.5 um" in out["message"]
    # uncalibrated image: pixel units
    s2 = Session()
    s2.add_image(LumenImage(np.zeros((8, 8), np.uint8), name="px"))
    s2.run_op("tools.measure_distance", {"x": 0, "y": 0, "x2": 6, "y2": 8, "event": "drag-end"})
    assert s2.results.rows[-1]["Length"] == 10 and s2.results.rows[-1]["Unit"] in ("pixel", "pixels", "px")


# ---------------------------------------------------------------- text files (script editor Open… / Save…)
def test_textfile_router_reads_and_writes(tmp_path):
    c = TestClient(create_app(Session()), raise_server_exceptions=False)
    p = tmp_path / "sub" / "script.py"
    r = c.post("/api/textfile", json={"path": str(p), "text": "print('hé')\n"})
    assert r.status_code == 200, r.text
    assert r.json()["path"] == str(p) and r.json()["bytes"] == len("print('hé')\n".encode("utf-8"))
    assert p.read_text(encoding="utf-8") == "print('hé')\n"
    r = c.get("/api/textfile", params={"path": str(p)})
    assert r.status_code == 200 and r.json()["text"] == "print('hé')\n" and r.json()["name"] == "script.py"
    (tmp_path / "bom.txt").write_bytes("﻿line1\r\nline2".encode("utf-8"))
    assert c.get("/api/textfile", params={"path": str(tmp_path / "bom.txt")}).json()["text"] == "line1\nline2"  # BOM dropped, newlines normalised
    assert c.get("/api/textfile", params={"path": str(tmp_path / "missing.txt")}).status_code == 404
    assert c.post("/api/textfile", json={"path": "  ", "text": "x"}).status_code == 400
    assert c.post("/api/textfile", json={"path": str(tmp_path), "text": "x"}).status_code == 400
