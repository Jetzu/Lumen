"""Guided workflows: definitions (lumen/core/workflows.py) and the /api/workflows router."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core import ops  # noqa: F401  (registers commands)
from lumen.core import workflows as W
from lumen.core.registry import REGISTRY
from lumen.core.scripting import run_script
from lumen.core.session import Session
from lumen.server.app import create_app

EXPECTED = {"count_nuclei", "intensity_per_cell", "track_cells", "pixel_classifier", "colocalization", "stitch_tiles",
            "zstack_figure", "batch_folder", "register_timelapse", "edit_labels"}


def client():
    s = Session()
    return TestClient(create_app(s), raise_server_exceptions=False), s


# ------------------------------------------------------------------ definitions
def test_all_expected_workflows_defined_and_valid():
    ids = {w["id"] for w in W.WORKFLOWS}
    assert EXPECTED <= ids
    assert W.validate() == []


def test_every_step_op_exists_and_defaults_match_params():
    for wf in W.all_workflows():
        assert wf["title"] and wf["description"] and wf["steps"]
        for st in wf["steps"]:
            assert st["title"] and st["help"], (wf["id"], st["id"])
            for r in st["requires"]:
                assert r in W.REQUIREMENTS
            if st["op"] is None:
                continue
            spec = REGISTRY.get(st["op"])
            assert spec.id == st["op"]
            names = {p.name: p for p in spec.params}
            for k, v in st["params"].items():
                assert k in names, f"{wf['id']}/{st['id']}: {st['op']} has no param {k}"
                p = names[k]
                if p.type == "image":
                    continue
                coerced = p.coerce(v)
                if p.type in ("int", "float"):
                    assert coerced == v, f"{wf['id']}/{st['id']}: {k}={v} clamped to {coerced}"
                elif p.type == "choice":
                    assert v in (p.choices or [v])
                elif p.type == "bool":
                    assert isinstance(v, bool)
            # the merged defaults must coerce without error
            W.step_params(st)


def test_validate_reports_bad_definitions():
    bad = [{"id": "x", "title": "X", "description": "", "steps": [
        {"id": "a", "title": "A", "help": "h", "op": "filters.nope", "params": {}},
        {"id": "b", "title": "B", "help": "h", "op": "filters.gaussian", "params": {"sigma": 2.0, "radius": 1}},
        {"id": "c", "title": "C", "help": "h", "op": "filters.median", "params": {"radius": 5000.0}},
        {"id": "d", "title": "D", "help": "h", "op": "threshold.compute", "params": {"method": "Nope"}},
        {"id": "d", "title": "D", "help": "h", "op": "analyze.particles", "params": {}, "preview": True, "requires": "magic"},
        {"id": "e", "title": "E", "help": "", "op": None, "params": {"x": 1}},
    ]}]
    problems = "\n".join(W.validate(bad))
    assert "unknown op 'filters.nope'" in problems
    assert "no parameter 'radius'" in problems
    assert "x/c: radius must be between 0.5 and" in problems and "got 5000" in problems  # rejected, not clamped
    assert "'Nope' is not one of" in problems
    assert "duplicate step id" in problems
    assert "cannot preview" in problems
    assert "unknown requirement 'magic'" in problems
    assert "empty help" in problems
    assert "manual step has params" in problems


def test_tokens_and_manual_steps():
    wf = W.get_workflow("intensity_per_cell")
    measure = next(s for s in wf["steps"] if s["id"] == "measure")
    assert measure["params"]["intensity_image"] == "$original"
    assert W.step_params(measure)["intensity_image"] == "$original"
    wf = W.get_workflow("pixel_classifier")
    assert wf["steps"][0]["op"] is None and wf["steps"][0]["requires"] == ["image"]
    assert W.get_workflow("track_cells")["requires"] == ["timelapse"]
    with pytest.raises(KeyError):
        W.get_workflow("nope")


# ------------------------------------------------------------------ scripts
def test_nuclei_script_runs_on_blobs_and_counts_particles():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    r = W.build_script("count_nuclei")
    code = r["script"]
    assert code.count("lumen.run(") == 9  # every step incl. optional ones
    assert 'lumen.run("analyze.particles"' in code and 'min_size=20.0' in code
    out = run_script(s, code)
    assert out["error"] is None, out["error"]
    names = [im.name for im in s.images.values()]
    assert "blobs.tif" in names and "nuclei" in names  # original untouched, copy segmented
    assert s.get_image("blobs.tif").data.dtype == np.uint8 and s.get_image("blobs.tif").data.max() > 0
    count = s.tables["Particle Summary"]["rows"][0][1]
    assert count > 10
    assert len(s.results.rows) == count
    assert any(l["msg"] == f"{count} particles" for l in s.log)
    assert "Summary" in s.tables


def test_script_skips_optional_steps_and_applies_overrides():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    r = W.build_script("count_nuclei", params={"count": {"min_size": 5, "exclude_on_edges": False}, "smooth": {"sigma": 1}},
                       skip=["copy", "background", "summary"])
    code = r["script"]
    assert code.count("lumen.run(") == 6
    assert "image.duplicate" not in code and "subtract_background" not in code and "analyze.summarize" not in code
    assert "— skipped" in code
    assert "sigma=1.0" in code and "min_size=5.0" in code and "exclude_on_edges=False" in code
    included = [st["id"] for st in r["steps"] if st["included"]]
    assert included == ["smooth", "threshold", "mask", "fill", "split", "count"]
    out = run_script(s, code)
    assert out["error"] is None, out["error"]
    assert [im.name for im in s.images.values()] == ["blobs.tif", "Outlines of blobs.tif"]  # in place, no copy
    assert s.tables["Particle Summary"]["rows"][0][1] > 10
    # a required step cannot be skipped; include_optional=False drops every optional step without overrides
    r2 = W.build_script("count_nuclei", skip=["count"], include_optional=False)
    assert 'lumen.run("analyze.particles"' in r2["script"] and "image.duplicate" not in r2["script"]


def test_intensity_script_measures_original_image():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    r = W.build_script("intensity_per_cell", skip=["background"])
    assert "intensity_image=source.id" in r["script"]
    out = run_script(s, r["script"])
    assert out["error"] is None, out["error"]
    rows = s.results.rows
    assert len(rows) > 10
    # intensities come from the original 8-bit image, not from the label image
    assert all(0 < row["Mean"] <= 255 for row in rows)
    assert max(row["Mean"] for row in rows) > 100


def test_tracking_script_on_synthetic_timelapse(tmp_path):
    s = Session()
    s.run_op("file.sample", {"name": "Nuclei timelapse (synthetic)"})
    r = W.build_script("track_cells", params={"detect": {"radius": 9, "add_to_manager": False}, "link": {"max_displacement": 15},
                                              "export": {"path": str(tmp_path / "tracks.csv")}})
    out = run_script(s, r["script"])
    assert out["error"] is None, out["error"]
    assert "Tracks" in s.tables and len(s.tables["Tracks"]["rows"]) >= 5
    header = (tmp_path / "tracks.csv").read_text().splitlines()[0]
    assert "TRACK_ID" in header and "POSITION_X" in header


def test_edit_labels_workflow_script_edits_a_label_image():
    from skimage.draw import disk
    from lumen.core.image import LumenImage
    wf = W.get_workflow("edit_labels")
    assert wf["requires"] == ["labels"] and [s["id"] for s in wf["steps"]] == ["pick", "select", "remove", "merge", "grow", "fill", "size", "border", "largest", "undo", "rois"]
    assert wf["steps"][0]["op"] is None and wf["steps"][9]["action"] == "undo" and sum(1 for s in wf["steps"] if s["optional"]) == 9
    lab = np.zeros((80, 100), np.uint16)
    rr, cc = disk((25, 25), 12)
    lab[rr, cc] = 1          # the label we pick
    lab[50:60, 40:60] = 2    # 200 px, gets a hole
    lab[52:55, 45:48] = 0
    lab[0:5, 90:100] = 3     # touches the border
    lab[70:72, 10:12] = 4    # 4 px speck
    s = Session()
    im = s.add_image(LumenImage(lab, name="lab"))
    s.run_op("edit.set_foreground", {"value": 1})  # the value picker's click
    r = W.build_script("edit_labels", params={"remove": {"labels": "", "new_value": 0}, "size": {"min_size": 10}}, skip=["merge", "grow", "largest"])
    code = r["script"]
    assert "# 10. Undo a step" in code and "# undo (⌘Z) is interactive" in code and code.count("lumen.run(") == 6
    assert "morpho.merge_labels" not in code and 'lumen.run("morpho.replace_label", labels=\'\', new_value=0)' in code
    out = run_script(s, code)
    assert out["error"] is None, out["error"]
    p = im.data[0, 0, 0]
    assert (p == 1).sum() == 0            # the picked label was removed
    assert (p == 2).sum() == 200          # its hole was filled
    assert (p == 3).sum() == 0            # border label dropped
    assert (p == 4).sum() == 0            # speck dropped
    assert [x["name"] for x in s.roi_manager.rois] == ["L2"]
    assert s.histories[im.id].labels()["undo"][-3:] == ["Fill Label Holes", "Label Size Opening", "Remove Border Labels"]


def test_manual_steps_become_comments_or_script_lines():
    r = W.build_script("zstack_figure")
    assert 'lumen.save("figure.png")' in r["script"]
    r = W.build_script("pixel_classifier")
    assert "# 1. Draw training regions" in r["script"] and "#    " in r["script"]
    assert r["steps"][0]["manual"] is True
    r = W.build_script("stitch_tiles")
    assert "source = lumen.image" not in r["script"]  # needs no open image


# ------------------------------------------------------------------ router
def test_api_list_and_get():
    c, _ = client()
    r = c.get("/api/workflows")
    assert r.status_code == 200, r.text
    lst = r.json()
    assert isinstance(lst, list) and {w["id"] for w in lst} >= EXPECTED
    w = next(w for w in lst if w["id"] == "count_nuclei")
    assert w["n_steps"] == 9 and w["n_required"] == 6 and "analyze.particles" in w["ops"] and w["requires"] == ["image"]
    r = c.get("/api/workflows/count_nuclei")
    assert r.status_code == 200
    full = r.json()
    steps = full["steps"]
    assert [st["id"] for st in steps][:3] == ["copy", "smooth", "background"]
    smooth = steps[1]
    assert smooth["op_name"] == "Gaussian Blur" and smooth["op_kind"] == "plane" and smooth["op_preview"] is True and smooth["preview"] is True
    assert steps[0]["optional"] is True and steps[-2]["requires"] == ["binary"]
    r = c.get("/api/workflows/pixel_classifier")
    assert r.json()["steps"][0]["op"] is None and r.json()["steps"][0]["op_name"] is None
    r = c.get("/api/workflows/nope")
    assert r.status_code == 404 and "Unknown workflow" in r.json()["error"]


def test_api_script_endpoint_returns_runnable_code():
    c, s = client()
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}})
    assert r.status_code == 200, r.text
    r = c.post("/api/workflows/count_nuclei/script", json={"params": {"count": {"min_size": 30}}, "skip": ["summary"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == "count_nuclei" and "min_size=30.0" in body["script"]
    r = c.post("/api/script/run", json={"code": body["script"]})
    assert r.status_code == 200 and r.json()["error"] is None, r.text
    st = c.get("/api/state").json()
    assert st["tables"]["Particle Summary"]["rows"][0][1] > 10
    assert len(st["results"]["rows"]) == st["tables"]["Particle Summary"]["rows"][0][1]
    # GET variant and empty body
    r = c.get("/api/workflows/count_nuclei/script", params={"include_optional": "false"})
    assert r.status_code == 200 and "image.duplicate" not in r.json()["script"]
    r = c.post("/api/workflows/count_nuclei/script")
    assert r.status_code == 200 and "image.duplicate" in r.json()["script"]


def test_api_script_rejects_unknown_step_or_bad_value():
    c, _ = client()
    r = c.post("/api/workflows/count_nuclei/script", json={"params": {"nope": {"a": 1}}})
    assert r.status_code == 400 and "no step 'nope'" in r.json()["error"]
    r = c.post("/api/workflows/count_nuclei/script", json={"params": {"threshold": {"method": "Bogus"}}})
    assert r.status_code == 400 and "Bogus" in r.json()["error"]
    r = c.post("/api/workflows/nope/script", json={})
    assert r.status_code == 404
