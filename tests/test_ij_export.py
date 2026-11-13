"""Recorder → ImageJ macro export (lumen/core/ij_export.py): exact conversions, honest skips, and a round trip that runs
the exported macro through the macro layer and compares the pixels with the recorded session."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core.ij_compat import IJ
from lumen.core.ij_export import recorder_to_ijm, roi_lines
from lumen.core.session import Session
from lumen.server.app import create_app


def test_lines_convert_exactly():
    lines = [
        "# Lumen recorder – session started 2026-11-08 04:00:01",
        'lumen.open("/data/a.tif")  # → "a.tif"',
        'lumen.run("filters.gaussian", sigma=3.0)',
        'lumen.run("filters.median", radius=2.0, scope=\'slice\')',
        'lumen.run("process.subtract_background", radius=40.0, light_background=False, create_background=False, smooth=True, method=\'Rolling ball\')',
        'lumen.select("b.tif")',
        'lumen.run("analyze.measure", roi={\'type\': \'rect\', \'x\': 10, \'y\': 20, \'w\': 30, \'h\': 40})',
        'lumen.run("stacks.z_project", method=\'Max Intensity\', start=1, stop=0, axis=\'Z\')  # → "MAX_b.tif"',
        'lumen.run("threshold.compute", method=\'Otsu\', dark_background=True, channel=2)',
        'lumen.run("edit.invert")',
        'lumen.undo("MAX_b.tif")',  # cancels the invert (a display edit would be an undo step of its own, so the undo comes before them)
        'lumen.display("MAX_b.tif", channel=2, min=100, max=1500, lut=\'Fire\')',
        'lumen.display("MAX_b.tif", mode=\'single\')',
        'lumen.display("MAX_b.tif", z=4)',
        'lumen.close("MAX_b.tif")',
        'lumen.run("file.new", title=\'Untitled\', type=\'16-bit\', width=64, height=32, slices=3, channels=1, frames=1, fill=\'Ramp\', period=8)  # → "Untitled"',
        'lumen.run("rois.delete", indices=\'1,3-4\')',
        'lumen.run("rois.clear")',
        'lumen.save("/out/x.png")',
    ]
    out = recorder_to_ijm(lines)
    assert out["skipped"] == [] and out["count"] == len(lines) - 1
    # the undo cancelled the invert: both lines are left out (never run("Undo"), E8-3), and the script says so
    assert out["undone"] == ['lumen.run("edit.invert")'] and out["removed"] == 2 and out["converted"] == out["count"] - 2
    assert out["lines"] == [
        "// Lumen recorder – session started 2026-11-08 04:00:01",
        '// note: 1 recorded step the session undid was left out together with the undo / redo lines (ImageJ\'s Undo is single-level, so run("Undo") is never written): lumen.run("edit.invert")',
        'open("/data/a.tif"); // → "a.tif"',
        'run("Gaussian Blur...", "sigma=3 stack");',
        'run("Median...", "radius=2");',
        'run("Subtract Background...", "rolling=40 stack");',
        'selectWindow("b.tif");',
        "makeRectangle(10, 20, 30, 40);",
        'run("Measure");',
        'run("Z Project...", "projection=[Max Intensity] all"); // → "MAX_b.tif"',
        "Stack.setChannel(2);",
        'setAutoThreshold("Otsu dark");',
        "Stack.setChannel(2);",
        "setMinAndMax(100, 1500);",
        'run("Fire");',
        'Stack.setDisplayMode("color");',
        "Stack.setSlice(5);",
        'close("MAX_b.tif");',
        'newImage("Untitled", "16-bit ramp", 64, 32, 3); // → "Untitled"',
        'roiManager("select", newArray(1, 3, 4));',
        'roiManager("delete");',
        'roiManager("reset");',
        'saveAs("PNG", "/out/x.png");',
    ]
    assert out["notes"] == [out["lines"][1][len("// note: "):]] and not any(ln.startswith('run("Undo")') for ln in out["lines"])


def test_undo_lines_are_resolved_instead_of_exported():
    marker = "# Lumen recorder – session started 2026-11-08 04:00:01"
    gauss, median = 'lumen.run("filters.gaussian", sigma=3.0)', 'lumen.run("filters.median", radius=2.0)'
    lines = [marker, 'lumen.open("/data/a.tif")  # → "a.tif"', gauss, median, 'lumen.undo("a.tif")', 'lumen.undo("a.tif")', 'lumen.redo("a.tif")',
             'lumen.run("analyze.measure")', 'lumen.run("edit.invert")', 'lumen.undo("a.tif")']
    out = recorder_to_ijm(lines)
    assert out["undone"] == [median, 'lumen.run("edit.invert")'] and out["removed"] == 6 and out["skipped"] == []
    assert out["count"] == 9 and out["converted"] == 3
    body = [ln for ln in out["lines"] if not ln.startswith("//")]
    assert body == ['open("/data/a.tif"); // → "a.tif"', 'run("Gaussian Blur...", "sigma=3 stack");', 'run("Measure");']
    assert out["lines"][0] == "// " + marker[2:] and out["lines"][1].startswith("// note: 2 recorded steps the session undid were left out")
    assert not any(ln.startswith('run("Undo")') for ln in out["lines"])
    # an undo that cancels nothing recorded (its image has no step) is kept as a comment that says why
    out = recorder_to_ijm(['lumen.open("/data/a.tif")  # → "a.tif"', 'lumen.run("analyze.measure")', 'lumen.undo("a.tif")', 'lumen.redo("a.tif")'])
    assert [s["line"] for s in out["skipped"]] == ['lumen.undo("a.tif")', 'lumen.redo("a.tif")'] and out["undone"] == [] and out["removed"] == 0
    assert out["lines"][2].startswith("// not converted (this undo cancels no recorded step") and "no Redo" in out["lines"][3]
    assert out["notes"] == [] and not any(ln.startswith('run("Undo")') for ln in out["lines"])


def test_notes_are_written_into_the_script_header():
    lines = ["# Lumen recorder – session started 2026-11-08 04:00:01", 'lumen.open("/data/a.tif")  # → "a.tif"',
             'lumen.run("color.lut", lut=\'Fire\', all_channels=True)', 'lumen.run("color.lut", lut=\'Grays\', all_channels=True)']
    out = recorder_to_ijm(lines)
    assert out["notes"] == ["Set LUT on all channels: ImageJ's run(\"<LUT>\") changes the current channel only"]  # once, not per line
    assert out["lines"][:3] == ["// Lumen recorder – session started 2026-11-08 04:00:01", "// note: " + out["notes"][0], 'open("/data/a.tif"); // → "a.tif"']
    assert out["script"].splitlines()[1] == "// note: " + out["notes"][0]
    # without a session marker the notes come first
    out = recorder_to_ijm(lines[1:])
    assert out["lines"][0] == "// note: " + out["notes"][0] and out["lines"][1].startswith("open(")
    assert recorder_to_ijm(['lumen.run("filters.median", radius=2.0)'])["lines"] == ['run("Median...", "radius=2 stack");']  # no notes: no header


def test_display_with_only_min_or_only_max_keeps_the_other_end():
    out = recorder_to_ijm(['lumen.display("a.tif", channel=1, max=1500)', 'lumen.display("a.tif", min=100)'])
    assert out["skipped"] == [] and out["lines"] == ['selectWindow("a.tif");', "Stack.setChannel(1);", "getMinAndMax(__lo, __hi);", "setMinAndMax(__lo, 1500);",
                                                      "getMinAndMax(__lo, __hi);", "setMinAndMax(100, __hi);"]
    # round trip through the macro layer: the other end of the range survives
    a = Session()
    a.run_op("file.new", {"title": "rt", "type": "16-bit", "width": 16, "height": 16, "fill": "Ramp"})
    a.update_display(a.active_id, {"channels": [{"min": 100, "max": 4000}]}, record=True)
    a.run_op("edit.invert", {})  # successive edits of one channel collapse into one display line: a command in between keeps them apart
    a.update_display(a.active_id, {"channels": [{"max": 1500}]}, record=True)
    exported = recorder_to_ijm(list(a.macro))
    assert exported["skipped"] == [] and exported["lines"][-2:] == ["getMinAndMax(__lo, __hi);", "setMinAndMax(__lo, 1500);"]
    b = Session()
    IJ(b).macro(exported["script"])
    ch = b.active.display.channels[0]
    assert (ch.min, ch.max) == (100.0, 1500.0)


def test_unconvertible_lines_are_kept_as_comments_with_reasons():
    lines = [
        'lumen.run("file.sample", name=\'Blobs\')  # → "blobs.tif"',
        'lumen.redo("blobs.tif")',
        'lumen.run("batch.process_folder", input_folder=\'/x\', script=\'pass\')',
        'lumen.run("process.clahe", block_size=127, bins=256, clip_limit=0.01)',
        'lumen.run("results.restore_cleared")',
        'lumen.display("blobs.tif", channel=1, visible=False)',
        'lumen.run("stacks.z_project", method=\'Max Intensity\', start=1, stop=0, axis=\'T\')',
    ]
    out = recorder_to_ijm(lines)
    reasons = [s["reason"] for s in out["skipped"]]
    assert [s["line"] for s in out["skipped"]] == [lines[0], lines[1], lines[2], lines[3], lines[4], lines[6]]
    assert "not an ImageJ sample" in reasons[0] and "no Redo" in reasons[1] and "getFileList" in reasons[2]
    assert "clip_limit=0.01 has no direct option" in reasons[3]  # ImageJ's maximum=3 would give 0.03: not silently dropped
    assert "cannot restore" in reasons[4] and "single slice" in reasons[5]
    assert out["converted"] == 1 and out["count"] == 7
    # the skipped lines stay in the script, as comments that say why (after the note header: no session marker here)
    assert out["lines"][0].startswith("// note: channel visibility (hidden)") and len(out["notes"]) == 1
    assert out["lines"][1].startswith("// not converted (the bundled sample 'Blobs' is not an ImageJ sample): lumen.run(\"file.sample\"")
    # the visibility part of a display line is a comment with the real ImageJ function named
    assert out["lines"][-3] == 'Stack.setChannel(1);' and 'Stack.setActiveChannels' in out["lines"][-2]


def test_roi_constructors():
    assert roi_lines({"type": "oval", "x": 1.5, "y": 2, "w": 3, "h": 4}) == ["makeOval(1.5, 2, 3, 4);"]
    assert roi_lines({"type": "polygon", "points": [[0, 0], [10, 0], [5, 8]]}) == ["makePolygon(0, 0, 10, 0, 5, 8);"]
    assert roi_lines({"type": "line", "x1": 0, "y1": 0, "x2": 9, "y2": 9, "width": 3}) == ["makeLine(0, 0, 9, 9, 3);"]
    assert roi_lines({"type": "freehand", "points": [[1, 2], [3, 4]]}) == ['makeSelection("freehand", newArray(1, 3), newArray(2, 4));']
    assert roi_lines(None) == []


def _pixels(s: Session, name: str) -> np.ndarray:
    img = next(i for i in s.images.values() if i.name == name)
    return np.asarray(img.data).copy()


def test_round_trip_through_the_macro_layer_reproduces_the_pixels():
    """A session is recorded, exported as a macro, the macro is run in a fresh session: same images, same bytes."""
    a = Session()
    a.run_op("file.new", {"title": "rt", "type": "8-bit", "width": 64, "height": 48, "fill": "Ramp"})
    a.run_op("filters.gaussian", {"sigma": 3.0})
    a.run_op("filters.median", {"radius": 2.0}, scope="slice")
    a.run_op("process.subtract_background", {"radius": 20.0})
    a.run_op("edit.invert", {})
    a.run_op("image.duplicate", {"title": "copy", "what": "Whole stack"})
    a.run_op("threshold.apply", {"method": "Otsu", "dark_background": True})
    a.run_op("analyze.particles", {"min_size": 0.0, "max_size": 0.0, "show": "Nothing"})
    a.run_op("analyze.measure", {}, roi={"type": "rect", "x": 4, "y": 5, "w": 20, "h": 10}, image_id=[i for i in a.images.values() if i.name == "rt"][0].id)
    a.run_op("file.new", {"title": "st", "type": "16-bit", "width": 32, "height": 24, "slices": 5, "fill": "Ramp"})
    a.run_op("stacks.z_project", {"method": "Average Intensity"})
    exported = recorder_to_ijm(list(a.macro), titles={i.id: i.name for i in a.images.values()})
    assert exported["skipped"] == [], exported["skipped"]
    assert 'run("Gaussian Blur...", "sigma=3 stack");' in exported["lines"]
    assert 'setOption("BlackBackground", true);' in exported["lines"]

    b = Session()
    IJ(b).macro(exported["script"])
    for name in ("rt", "copy", "AVG_st"):
        np.testing.assert_array_equal(_pixels(b, name), _pixels(a, name), err_msg=name)
    assert len(b.results.rows) == len(a.results.rows) > 0
    assert len(b.roi_manager.to_list()) == len(a.roi_manager.to_list()) > 0


def test_ijm_endpoint_returns_this_clients_recording():
    s = Session()
    c = TestClient(create_app(s))
    h = {"X-Lumen-Client": "tab-1"}
    assert c.post("/api/ops/file.new/run", json={"params": {"title": "e", "type": "8-bit", "width": 16, "height": 16, "fill": "Ramp"}}, headers=h).status_code == 200
    assert c.post("/api/ops/filters.median/run", json={"params": {"radius": 1.0}}, headers=h).status_code == 200
    r = c.get("/api/recorder/ijm", headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["converted"] == 2 and body["count"] == 2 and body["skipped"] == []
    assert body["lines"][0].startswith("// Lumen recorder – session started")
    assert body["lines"][1].startswith('newImage("e", "8-bit ramp", 16, 16, 1);') and body["lines"][2] == 'run("Median...", "radius=1 stack");'
    # another client sees only its own (empty) recording
    other = c.get("/api/recorder/ijm", headers={"X-Lumen-Client": "tab-2"}).json()
    assert other["count"] == 0 and other["converted"] == 0
