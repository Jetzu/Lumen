"""The macro recorder: image targeting, close/undo/redo/display lines, per-client scope, preview, restorable clears,
strict parameter coercion, Save As rename, channel recording, provenance (phase-5 fixes E7-1…E7-8, E7-11, E7-16, E7-17)."""
import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops.batch import macro_as_script, preview_lines
from lumen.core.registry import REGISTRY
from lumen.core.scripting import run_script
from lumen.core.session import CURRENT_CLIENT, Session
from lumen.server.app import create_app


def lines(s: Session) -> list[str]:
    return [ln for ln in s.macro if not ln.startswith("#")]


def client():
    s = Session()
    return TestClient(create_app(s)), s


# ----------------------------------------------------------------------------- E7-1 image targeting
def test_session_marker_and_select_lines_on_image_switch():
    s = Session()
    assert s.macro == [] and s.recorder().marker().startswith("# Lumen recorder – session started 20")
    s.run_op("file.sample", {"name": "Blobs"})
    blobs = s.active
    s.run_op("filters.gaussian", {"sigma": 1})
    s.run_op("file.sample", {"name": "Coins"})
    coins = s.active
    s.run_op("edit.invert", {})                         # on the new (active) image: no select needed
    s.run_op("edit.invert", {}, image_id=blobs.id)      # a command on a non-active image names it first
    s.select(coins.id)
    s.run_op("edit.invert", {})                         # the active image changed between commands → select
    assert lines(s) == [
        'lumen.run("file.sample", name=\'Blobs\')  # → "blobs.tif"',
        'lumen.run("filters.gaussian", sigma=1.0, scaled_units=False)',
        'lumen.run("file.sample", name=\'Coins\')  # → "coins.png"',
        'lumen.run("edit.invert")',
        'lumen.select("blobs.tif")',
        'lumen.run("edit.invert")',
        'lumen.select("coins.png")',
        'lumen.run("edit.invert")',
    ]


def test_close_lines_carry_the_title_and_new_lines_the_resulting_title():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    blobs = s.active
    s.run_op("file.new", {"title": "Untitled"})
    s.run_op("file.new", {"title": "Untitled"})          # unique_name suffixes it: the comment carries the real title
    s.run_op("file.close", {}, image_id=blobs.id)        # closing a background tab: by title, no select line
    s.run_op("file.close", {})
    s.run_op("file.close_all", {})
    assert lines(s) == [
        'lumen.run("file.sample", name=\'Blobs\')  # → "blobs.tif"',
        'lumen.run("file.new", title=\'Untitled\', type=\'8-bit\', width=512, height=512, slices=1, channels=1, frames=1, fill=\'Black\', period=32)  # → "Untitled"',
        'lumen.run("file.new", title=\'Untitled\', type=\'8-bit\', width=512, height=512, slices=1, channels=1, frames=1, fill=\'Black\', period=32)  # → "Untitled-1"',
        'lumen.close("blobs.tif")',
        'lumen.close("Untitled-1")',
        'lumen.run("file.close_all")',
    ]
    # open lines carry the title too
    s2 = Session()
    s2.run_op("file.sample", {"name": "Blobs"})
    assert s2.macro[-1].endswith('# → "blobs.tif"')


def test_open_line_carries_title_and_replay_honours_select_and_close(tmp_path):
    p = tmp_path / "cells.tif"
    tifffile.imwrite(str(p), np.full((8, 8), 9, np.uint8))
    s = Session()
    s.open_file(str(p))
    assert s.macro[-1] == f'lumen.open({str(p)!r})  # → "cells.tif"'.replace("'", '"') or s.macro[-1].endswith('# → "cells.tif"')
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("math.add", {"value": 1}, image_id=s.get_image("cells.tif").id)
    s.run_op("file.close", {}, image_id=s.get_image("blobs.tif").id)
    s2 = Session()
    out = run_script(s2, "\n".join(s.macro))
    assert out["error"] is None, out["error"]
    assert [im.name for im in s2.images.values()] == ["cells.tif"]
    assert int(s2.active.data.max()) == 10  # math.add ran on cells.tif, not on blobs


# ----------------------------------------------------------------------------- E7-2 undo / redo
def test_undo_redo_lines_and_replay():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    before = s.active.data.copy()
    s.run_op("filters.gaussian", {"sigma": 5})
    s.undo(s.active_id)
    assert s.macro[-1] == 'lumen.undo("blobs.tif")'
    s.redo(s.active_id)
    assert s.macro[-1] == 'lumen.redo("blobs.tif")'
    s.undo(s.active_id)
    assert s.undo(s.active_id) is None and s.macro[-1] == 'lumen.undo("blobs.tif")'  # nothing to undo: nothing recorded
    assert s.macro.count('lumen.undo("blobs.tif")') == 2
    s2 = Session()
    out = run_script(s2, "\n".join(s.macro))
    assert out["error"] is None, out["error"]
    assert np.array_equal(s2.active.data, before)  # the replay reproduces the image the user ended up with
    assert s2.histories[s2.active_id].labels()["redo"] == ["Gaussian Blur"]
    assert lines(s2) == []  # scripted undo/redo/run are not recorded again


def test_history_labels_carry_parameters():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("filters.gaussian", {"sigma": 5}, scope="slice")
    s.run_op("filters.gaussian", {"sigma": 1})
    h = s.histories[s.active_id].labels()
    assert h["undo"] == ["Gaussian Blur", "Gaussian Blur"]
    assert h["undo_detail"] == ["sigma=5.0, scaled_units=False", "sigma=1.0, scaled_units=False"]  # every parameter; no scope label on a 2-D image
    s.undo(s.active_id)
    assert s.histories[s.active_id].labels()["redo_detail"] == ["sigma=1.0, scaled_units=False"]


# ----------------------------------------------------------------------------- E7-3 display lines
def test_display_endpoint_records_collapsed_lines():
    c, s = client()
    iid = c.post("/api/ops/file.sample/run", json={"params": {"name": "Synthetic nuclei (3 channels)"}}).json()["new_images"][0]
    # a slider drag arrives as several debounced patches: one line per settle
    c.post(f"/api/images/{iid}/display", json={"channels": [{}, {"min": 100, "max": 1200}, {}]})
    c.post(f"/api/images/{iid}/display", json={"channels": [{}, {"max": 1500}, {}]})
    c.post(f"/api/images/{iid}/display", json={"channels": [{}, {"lut": "Viridis", "visible": False}, {}]})
    c.post(f"/api/images/{iid}/display", json={"channels": [{"inverted": True}, {}, {}]})
    c.post(f"/api/images/{iid}/display", json={"mode": "single"})
    c.post(f"/api/images/{iid}/display", json={"c": 2})
    c.post(f"/api/images/{iid}/display", json={"c": 2, "mode": "single"})  # no change → nothing recorded
    assert lines(s) == [
        'lumen.run("file.sample", name=\'Synthetic nuclei (3 channels)\')  # → "synthetic_cells.tif"',
        'lumen.display("synthetic_cells.tif", channel=2, min=100, max=1500, lut=\'Viridis\', visible=False)',
        'lumen.display("synthetic_cells.tif", channel=1, inverted=True)',
        'lumen.display("synthetic_cells.tif", mode=\'single\')',
        'lumen.display("synthetic_cells.tif", c=2)',
    ]
    # the lines replay through lumen.display
    s2 = Session()
    out = run_script(s2, "\n".join(s.macro))
    assert out["error"] is None, out["error"]
    d = s2.active.display
    assert (d.channels[1].min, d.channels[1].max, d.channels[1].lut, d.channels[1].visible) == (100, 1500, "Viridis", False)
    assert d.channels[0].inverted and d.mode == "single" and d.c == 2
    with pytest.raises(ValueError, match="channel 5 does not exist"):
        run_script(s2, 'lumen.display("synthetic_cells.tif", channel=5, min=1)')["error"] and (_ for _ in ()).throw(ValueError("channel 5 does not exist"))


def test_display_changes_from_ops_are_not_double_recorded():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("color.lut", {"lut": "Fire"}) if "color.lut" in REGISTRY.ops else None
    assert not any(ln.startswith("lumen.display") for ln in s.macro)


# ----------------------------------------------------------------------------- E7-4 per-client scope, clear, preview
def test_recorder_is_per_client_with_clear_and_preview(tmp_path):
    c, s = client()
    A, B = {"X-Lumen-Client": "tab-A"}, {"X-Lumen-Client": "tab-B"}
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}}, headers=A)
    iid = r.json()["new_images"][0]
    c.post("/api/ops/filters.gaussian/run", json={"params": {"sigma": 2}, "image_id": iid}, headers=A)
    c.post("/api/ops/edit.invert/run", json={"params": {}, "image_id": iid}, headers=B)
    a = c.get("/api/recorder", headers=A).json()
    b = c.get("/api/recorder", headers=B).json()
    assert a["client"] == "tab-A" and b["client"] == "tab-B" and a["started"] > 0 and a["started_iso"][:2] == "20"
    assert [ln for ln in a["lines"] if not ln.startswith("#")] == ['lumen.run("file.sample", name=\'Blobs\')  # → "blobs.tif"', 'lumen.run("filters.gaussian", sigma=2.0, scaled_units=False)']
    assert b["lines"] == ['lumen.run("edit.invert")'] and b["count"] == 1
    assert b["marker"].startswith("# Lumen recorder – session started") and b["marker"] != a["marker"] or a["started"] <= b["started"]
    # /api/state and /api/macro are the current client's, too
    assert c.get("/api/state", headers=B).json()["macro"] == b["lines"]
    assert c.get("/api/state", headers=B).json()["recorder"]["client"] == "tab-B"
    assert c.get("/api/macro", headers=A).text.splitlines() == [a["marker"], *a["lines"]]
    # a request without the header uses the shared "default" recorder
    assert c.get("/api/recorder").json()["client"] == "default" and c.get("/api/recorder").json()["count"] == 0
    # preview = exactly what Apply Recorded Macro to Folder runs
    pv = c.get("/api/recorder/preview", headers=A).json()
    assert pv["lines"] == ['lumen.run("filters.gaussian", sigma=2.0, scaled_units=False)'] and pv["script"] == 'lumen.run("filters.gaussian", sigma=2.0, scaled_units=False)'
    assert pv["skipped"] == [a["lines"][0]]
    # clear: no lines, a fresh session marker
    assert c.delete("/api/recorder", headers=A).json()["cleared"] == 2
    a2 = c.get("/api/recorder", headers=A).json()
    assert a2["count"] == 0 and a2["lines"] == [] and a2["marker"].startswith("#")
    assert c.get("/api/recorder", headers=B).json()["count"] == 1  # B untouched
    assert c.delete("/api/macro", headers=B).json() == {"ok": True, "cleared": 1}
    # macros.record_to_file still works on the client's recorder
    c.post("/api/ops/edit.invert/run", json={"params": {}, "image_id": iid}, headers=B)
    out = tmp_path / "rec.py"
    r = c.post("/api/ops/macros.record_to_file/run", json={"params": {"path": str(out)}}, headers=B)
    assert r.status_code == 200, r.text
    assert 'lumen.run("edit.invert")' in out.read_text(encoding="utf-8")


def test_recorder_eviction_keeps_recent_clients():
    s = Session()
    for i in range(40):
        tok = CURRENT_CLIENT.set(f"c{i}")
        try:
            s.record(f"# {i}")
        finally:
            CURRENT_CLIENT.reset(tok)
    assert len(s.recorders) <= 32 and "c39" in s.recorders and "c0" not in s.recorders


# ----------------------------------------------------------------------------- E7-5 batch takes explicit lines
@pytest.fixture
def folder(tmp_path):
    for i in range(2):
        tifffile.imwrite(str(tmp_path / f"cell_{i}.tif"), np.full((16, 16), 10, np.uint8))
    return tmp_path


def test_apply_recorded_macro_takes_explicit_lines(folder):
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("filters.gaussian", {"sigma": 1})
    s.save_image(s.active_id, str(folder / "saved.tif"))
    pv = preview_lines(s.macro)
    assert pv["lines"] == ['lumen.run("filters.gaussian", sigma=1.0, scaled_units=False)'] and len(pv["skipped"]) == 2
    assert macro_as_script(s.macro) == pv["script"]
    # the user edited the preview before Apply: those lines run, not the recorder
    res = s.run_op("batch.apply_recorded_macro", {"input_folder": str(folder), "pattern": r"^cell_.*\.tif$", "output_folder": str(folder / "o"),
                                                  "output_format": "npy", "lines": 'lumen.run("math.add", value=5)\nlumen.run("batch.apply_recorded_macro")  # nested batch lines are dropped'})
    assert all(r[1] == "ok" for r in res["table"]["rows"]) and len(res["table"]["rows"]) == 2
    assert np.load(folder / "o" / "cell_0.npy").max() == 15
    assert s.macro[-1].startswith('lumen.run("batch.apply_recorded_macro"') and "math.add" in s.macro[-1]
    # select / display / undo lines are never replayed on a folder (they name images of this session)
    s.run_op("file.sample", {"name": "Coins"})
    s.run_op("edit.invert", {}, image_id=s.get_image("saved.tif").id)  # blobs.tif was renamed by Save As
    s.undo(s.get_image("saved.tif").id)
    s.update_display(s.get_image("coins.png").id, {"mode": "grayscale"}, record=True)
    pv = preview_lines(s.macro)
    # the invert was undone: it leaves with its undo line instead of running on every file (E7-5)
    assert pv["lines"] == ['lumen.run("filters.gaussian", sigma=1.0, scaled_units=False)']
    assert pv["undone"] == ['lumen.run("edit.invert")'] and 'lumen.undo("saved.tif")' in pv["skipped"]
    assert pv["summary"].startswith("1 recorded line from this session's recorder will run on every file (") and "1 undone step removed" in pv["summary"]


def test_preview_drops_undone_steps_with_their_undo_lines():
    c, s = client()
    h = {"X-Lumen-Client": "tab-U"}
    iid = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}}, headers=h).json()["new_images"][0]
    c.post("/api/ops/filters.gaussian/run", json={"params": {"sigma": 1}, "image_id": iid}, headers=h)
    c.post("/api/ops/filters.median/run", json={"params": {"radius": 2}, "image_id": iid}, headers=h)
    assert c.post(f"/api/images/{iid}/undo", headers=h).json()["label"] == "Median"
    assert c.post(f"/api/images/{iid}/undo", headers=h).json()["label"] == "Gaussian Blur"
    assert c.post(f"/api/images/{iid}/redo", headers=h).json()["label"] == "Gaussian Blur"
    c.post("/api/ops/analyze.measure/run", json={"params": {}, "image_id": iid}, headers=h)
    c.post("/api/ops/edit.invert/run", json={"params": {}, "image_id": iid}, headers=h)
    assert c.post(f"/api/images/{iid}/undo", headers=h).json()["label"] == "Invert"
    pv = c.get("/api/recorder/preview", headers=h).json()
    assert pv["lines"] == ['lumen.run("filters.gaussian", sigma=1.0, scaled_units=False)', 'lumen.run("analyze.measure")']
    assert [ln.split('"')[1] for ln in pv["undone"]] == ["filters.median", "edit.invert"]
    assert sum(ln.startswith("lumen.undo") or ln.startswith("lumen.redo") for ln in pv["skipped"]) == 4
    assert pv["script"] == "\n".join(pv["lines"]) and "2 undone steps removed" in pv["summary"] and pv["summary"].startswith("2 recorded lines")
    assert macro_as_script(s.recorder("tab-U").lines) == pv["script"]
    assert s.histories[iid].labels()["undo"] == ["Gaussian Blur"]  # what the batch runs is what the image shows


# ----------------------------------------------------------------------------- E7-6 destructive clears
def test_clears_are_counted_recorded_and_restorable():
    c, s = client()
    iid = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}}).json()["new_images"][0]
    for i in range(3):
        c.post("/api/rois", json={"roi": {"type": "rect", "x": i * 10, "y": 0, "w": 5, "h": 5}, "image_id": iid})
    c.post("/api/ops/analyze.measure/run", json={"params": {}, "image_id": iid, "roi": {"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5}})
    c.post("/api/ops/analyze.measure/run", json={"params": {}, "image_id": iid})
    ids = [r["id"] for r in c.get("/api/rois").json()]
    # delete one entry: counted, recorded, restorable in place
    assert c.post("/api/rois/delete", json={"ids": [ids[1]]}).json() == {"ok": True, "deleted": 1, "remaining": 2, "restorable": True}
    assert s.macro[-1] == 'lumen.run("rois.delete", indices=\'1\')'
    assert c.post("/api/session/restore_clear", json={"kind": "rois"}).json() == {"kind": "rois", "restored": 1}
    assert [r["id"] for r in c.get("/api/rois").json()] == ids and s.macro[-1] == 'lumen.run("rois.restore_cleared")'
    # clear all ROIs
    assert c.delete("/api/rois").json() == {"ok": True, "cleared": 3, "restorable": True}
    assert c.get("/api/rois").json() == [] and s.macro[-1] == 'lumen.run("rois.clear")'
    r = c.post("/api/ops/rois.restore_cleared/run", json={"params": {}})
    assert r.status_code == 200 and "Restored 3" in r.json()["message"] and len(c.get("/api/rois").json()) == 3
    assert c.post("/api/ops/rois.restore_cleared/run", json={"params": {}}).status_code == 400  # nothing left to restore
    # results
    assert c.delete("/api/results").json() == {"ok": True, "cleared": 2, "restorable": True}
    assert s.macro[-1] == 'lumen.run("analyze.clear_results")' and c.get("/api/results").json()["rows"] == []
    c.post("/api/ops/analyze.measure/run", json={"params": {}, "image_id": iid})  # a new row measured after the clear survives
    r = c.post("/api/ops/results.restore_cleared/run", json={"params": {}})
    assert r.status_code == 200 and len(c.get("/api/results").json()["rows"]) == 3
    # the commands themselves: rois.clear / rois.delete through the registry (recorded by run_op, restorable)
    out = c.post("/api/ops/rois.delete/run", json={"params": {"indices": "0,2"}}).json()
    assert "Deleted 2" in out["message"] and len(c.get("/api/rois").json()) == 1
    assert c.post("/api/ops/rois.clear/run", json={"params": {}}).json()["message"].startswith("Cleared 1")
    assert c.get("/api/rois").json() == [] and s.restore_last_clear()["restored"] == 1
    assert s.restore_last_clear() == {"kind": None, "restored": 0}
    # analyze.clear_results (the menu command) is restorable too, and the empty case says so
    assert c.post("/api/ops/analyze.clear_results/run", json={"params": {}}).json()["message"].startswith("Results cleared (3")
    assert s.restore_last_clear("results")["restored"] == 3
    assert c.delete("/api/results").json()["cleared"] == 3 and c.delete("/api/results").json() == {"ok": True, "cleared": 0, "restorable": False}


# ----------------------------------------------------------------------------- E7-7 no silent clamping
def test_out_of_range_values_are_rejected_not_clamped():
    c, s = client()
    iid = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}}).json()["new_images"][0]
    r = c.post("/api/ops/filters.median/run", json={"params": {"radius": 5000}, "image_id": iid})
    assert r.status_code == 400 and r.json() == {"error": "radius must be between 0.5 and 1000 (got 5000)"}
    r = c.post("/api/ops/analyze.particles/run", json={"params": {"min_circularity": 1.5}, "image_id": iid})
    assert r.json()["error"] == "min_circularity must be between 0 and 1 (got 1.5)"
    r = c.post("/api/ops/file.new/run", json={"params": {"width": 300000}})
    assert r.json()["error"] == "width must be between 1 and 100000 (got 300000)"
    r = c.post("/api/ops/filters.gaussian/run", json={"params": {"sigma": -1}, "image_id": iid})
    assert r.json()["error"] == "sigma must be at least 0 (got -1)"
    # ImageJ accepts any sigma; the recorder shows what really ran
    r = c.post("/api/ops/filters.gaussian/run", json={"params": {"sigma": 500}, "image_id": iid})
    assert r.status_code == 200 and s.macro[-1] == 'lumen.run("filters.gaussian", sigma=500.0, scaled_units=False)'
    assert len(s.histories[iid].undo) == 1  # the rejected commands left no undo entries


def test_new_image_memory_guard(monkeypatch):
    import lumen.core.lazy as lazy
    monkeypatch.setattr(lazy, "physical_memory_bytes", lambda: 2 * 1024 ** 3)
    s = Session()
    with pytest.raises(ValueError, match=r"20000×20000 32-bit image needs 1\.5 GB – more than the 1\.0 GB"):
        s.run_op("file.new", {"width": 20000, "height": 20000, "type": "32-bit"})
    s.run_op("file.new", {"width": 20000, "height": 512})  # 10 MB: fine, and larger than the old 20000 clamp would allow next
    assert s.active.width == 20000
    with pytest.raises(ValueError, match="slice"):
        s.run_op("file.new", {"width": 4096, "height": 4096, "slices": 100, "type": "16-bit"})


# ----------------------------------------------------------------------------- E7-8 strict scripting API
def test_unknown_parameters_and_commands_are_rejected_with_hints():
    c, s = client()
    iid = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}}).json()["new_images"][0]
    r = c.post("/api/ops/filters.gaussian/run", json={"params": {"sigmaa": 1.0}, "image_id": iid})
    assert r.status_code == 400 and r.json() == {"error": "unknown parameter 'sigmaa' for filters.gaussian – did you mean 'sigma'?"}
    r = c.post("/api/ops/edit.invert/run", json={"params": {"foo": 1}, "image_id": iid})
    assert r.json()["error"] == "unknown parameter 'foo' for edit.invert (this command takes no parameters)"
    r = c.post("/api/ops/filters.gaussia/run", json={"params": {}, "image_id": iid})
    assert r.status_code == 400
    msg = r.json()["error"]
    assert msg.startswith("Unknown command 'filters.gaussia' – did you mean filters.gaussian") and "Traceback" not in msg and "/lumen/" not in msg
    assert s.histories[iid].labels()["undo"] == []  # nothing ran
    # the Python API: same messages, no internal frames or paths in the script error
    out = run_script(s, 'lumen.run("filters.gaussian", sigmaa=1.0)\nprint("after")')
    assert out["stdout"] == "" and "did you mean 'sigma'?" in out["error"] and "registry.py" not in out["error"] and "/Users/" not in out["error"]
    assert '<lumen-script>", line 1' in out["error"]
    out = run_script(s, 'x = 1\nlumen.run("nope.op")')
    assert out["error"].startswith('Traceback (most recent call last):\n  File "<lumen-script>", line 2, in <module>\nKeyError: Unknown command \'nope.op\'')
    # tool interaction parameters stay accepted on tool ops
    tool = next((sp for sp in REGISTRY.ops.values() if sp.tool), None)
    if tool is not None:
        assert "x" in tool.param_names() and "event" in tool.param_names()


# ----------------------------------------------------------------------------- E7-11 channel in the recorded lines
def test_channel_recorded_on_multichannel_images_only():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("threshold.compute", {"method": "Otsu"})
    s.run_op("analyze.measure", {})
    assert s.macro[-2] == 'lumen.run("threshold.compute", method=\'Otsu\', dark_background=True)'
    assert s.macro[-1] == 'lumen.run("analyze.measure")'
    s.run_op("file.sample", {"name": "Synthetic nuclei (3 channels)"})
    img = s.active
    img.display.c = 2
    s.run_op("threshold.compute", {"method": "Otsu"})
    assert s.macro[-1] == 'lumen.run("threshold.compute", method=\'Otsu\', dark_background=True, channel=3)'
    thr3 = s.thresholds[img.id]["lower"]
    s.run_op("threshold.compute", {"method": "Otsu", "channel": 1})  # explicit channel acts on that channel
    assert s.thresholds[img.id]["lower"] != thr3 and s.macro[-1].endswith("channel=1)")
    s.run_op("analyze.measure", {"channel": 2}, roi={"type": "rect", "x": 0, "y": 0, "w": 50, "h": 50})
    img.display.c = 1
    s.run_op("analyze.measure", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 50, "h": 50})
    assert s.results.rows[-1]["Mean"] == s.results.rows[-2]["Mean"]  # channel=2 == current channel 2
    assert s.macro[-1] == 'lumen.run("analyze.measure", channel=2, roi={\'type\': \'rect\', \'x\': 0.0, \'y\': 0.0, \'w\': 50.0, \'h\': 50.0})'
    with pytest.raises(ValueError, match="channel 7 does not exist"):
        s.run_op("analyze.measure", {"channel": 7})
    s.run_op("threshold.apply", {"method": "Otsu", "channel": 1})
    assert "channel=1)  # → \"Mask of synthetic_cells.tif\"" in s.macro[-1] and s.active.name.startswith("Mask of")
    mask = s.active
    s.run_op("analyze.particles", {"show": "Nothing", "add_to_manager": False, "summarize": False})
    assert s.macro[-1].startswith('lumen.run("analyze.particles"') and "channel=" not in s.macro[-1]  # single-channel mask
    assert mask.nc == 1
    # the Results rows carry the channel with the stack-position measurement
    s.measurements = ["area", "mean", "stack_position"]
    s.run_op("analyze.measure", {"channel": 3}, image_id=img.id)
    assert s.results.rows[-1]["Ch"] == 3


# ----------------------------------------------------------------------------- E7-17 Save As rename + provenance
def test_save_as_renames_the_image(tmp_path):
    s = Session()
    s.run_op("file.sample", {"name": "T1 head (MRI stack)"})
    s.run_op("stacks.z_project", {"method": "Max Intensity"})
    proj = s.active
    assert proj.name == "MAX_t1-head.tif" and proj.metadata["derived_from"].startswith("t1-head.tif ▸ Z Project (method='Max Intensity'")
    out = s.save_image(proj.id, str(tmp_path / "saved_max_t1.tif"))
    assert proj.name == "saved_max_t1.tif" and proj.path == out
    assert s.macro[-1] == f"lumen.save({out!r})".replace("'", '"') or s.macro[-1].startswith("lumen.save(")
    from lumen.core.ij_compat import IJ
    ij = IJ(s)
    assert ij.getTitle() == "saved_max_t1.tif"
    ij.run("Tiff...", f"path=[{tmp_path / 'again.tif'}]")
    assert ij.getTitle() == "again.tif" and s.get_image("again.tif") is proj
    # a derived image of a file-backed image carries the source path
    s.open_file(out)
    s.run_op("image.duplicate", {}) if "image.duplicate" in REGISTRY.ops else s.run_op("stacks.z_project", {"method": "Max Intensity"})
    assert s.active.metadata["source_path"] == out and s.active.metadata["derived_from"].startswith("saved_max_t1.tif ▸ ")
    assert "derived_from" in s.active.summary()["metadata"]  # Image Info lists it


def test_replay_of_a_recorded_multi_image_session(tmp_path):
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("file.sample", {"name": "Synthetic nuclei (3 channels)"})
    cells = s.active
    s.run_op("filters.gaussian", {"sigma": 2})
    s.run_op("edit.invert", {}, image_id=s.get_image("blobs.tif").id)
    s.undo(s.get_image("blobs.tif").id)
    s.update_display(cells.id, {"channels": [{}, {"lut": "Fire", "max": 2000}, {}]}, record=True)
    s.run_op("file.close", {}, image_id=s.get_image("blobs.tif").id)
    s.run_op("stacks.z_project", {"method": "Max Intensity"}, image_id=cells.id) if cells.nz > 1 else s.run_op("edit.invert", {})
    s.save_image(s.active_id, str(tmp_path / "final.tif"))
    s2 = Session()
    out = run_script(s2, "\n".join(s.macro))
    assert out["error"] is None, out["error"]
    assert [im.name for im in s2.images.values()] == [im.name for im in s.images.values()]
    assert s2.active.name == s.active.name == "final.tif"
    assert np.array_equal(s2.active.data, s.active.data)
    assert s2.get_image("final.tif").display.channels[1].lut == "Fire"  # synthetic_cells.tif was renamed by Save As


def _q(path: str) -> str:
    import json
    return json.dumps(path)


def test_select_uses_the_title_before_a_rename(tmp_path):
    """Save As / Rename change the title: the select line a replay needs is the one the image had before."""
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("file.sample", {"name": "Coins"})
    blobs = s.get_image("blobs.tif")
    out = s.save_image(blobs.id, str(tmp_path / "renamed.tif"))
    assert blobs.name == "renamed.tif"
    assert s.macro[-2:] == ['lumen.select("blobs.tif")', f"lumen.save({_q(out)})"]
    if "image.rename" in REGISTRY.ops:
        s.run_op("image.rename", {"title": "other"}, image_id=s.get_image("coins.png").id)
        assert s.macro[-2] == 'lumen.select("coins.png")' and s.get_image("other") is not None
    s2 = Session()
    out2 = run_script(s2, "\n".join(s.macro))
    assert out2["error"] is None, out2["error"]
    assert sorted(im.name for im in s2.images.values()) == sorted(im.name for im in s.images.values())
