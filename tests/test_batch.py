"""Plugins ▸ Batch commands."""
import numpy as np
import pytest
import tifffile

from lumen.core import ops  # noqa: F401
from lumen.core.ops.batch import macro_as_script
from lumen.core.session import Session


def _undo_redo(history):
    """The undo / redo labels only (labels() also carries the per-step parameter details since phase 5)."""
    return {k: history.labels()[k] for k in ("undo", "redo")}


@pytest.fixture
def folder(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        tifffile.imwrite(str(tmp_path / f"cell_{i}.tif"), rng.integers(0, 255, (32, 40)).astype(np.uint8))
    (tmp_path / "readme.txt").write_text("not an image")
    (tmp_path / "other.png").write_bytes(b"")  # matches the default pattern but is broken on purpose – excluded below by pattern
    return tmp_path


def test_process_folder_gaussian_png(folder):
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})  # an image is open before the batch and must stay active afterwards
    active_before = s.active_id
    macro_before = list(s.macro)
    out_dir = folder / "out"
    script = 'lumen.run("filters.gaussian", sigma=2)\nlumen.run("analyze.measure", roi={"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10})\nprint("done", lumen.image.name)'
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"\.tif$", "output_folder": str(out_dir), "output_format": "png", "script": script})
    table = res["table"]
    assert table["name"] == "Batch"
    assert [r[0] for r in table["rows"]] == ["cell_0.tif", "cell_1.tif", "cell_2.tif"]
    assert all(r[1] == "ok" for r in table["rows"]), table["rows"]
    assert sorted(p.name for p in out_dir.glob("*.png")) == ["cell_0.png", "cell_1.png", "cell_2.png"]
    lines = (out_dir / "Results.csv").read_text().splitlines()
    assert lines[0].startswith("File,") and len(lines) == 4 and lines[1].startswith("cell_0.tif,")
    assert "3 ok, 0 failed" in res["message"]
    # the session is intact: batch images closed, previous image still active, recorder only grew by the batch call itself
    assert len(s.images) == 1 and s.active_id == active_before
    assert s.macro[:-1] == macro_before and s.macro[-1].startswith('lumen.run("batch.process_folder"')
    assert "Batch" in s.tables


def test_process_folder_errors_and_options(folder):
    s = Session()
    out_dir = folder / "out2"
    bad = 'lumen.run("filters.gaussian", sigma=2)\nif lumen.image.name.startswith("cell_1"): raise ValueError("boom")'
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"\.tif$", "output_folder": str(out_dir), "output_format": "tif", "script": bad, "save_results": False})
    rows = res["table"]["rows"]
    assert [r[1] for r in rows] == ["ok", "error", "ok"] and "boom" in rows[1][4]
    assert sorted(p.name for p in out_dir.glob("*.tif")) == ["cell_0.tif", "cell_2.tif"]
    assert not (out_dir / "Results.csv").exists()
    assert len(s.images) == 0
    # stop at the first error
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"\.tif$", "output_folder": str(out_dir), "output_format": "none", "script": bad, "continue_on_error": False})
    assert [r[1] for r in res["table"]["rows"]] == ["ok", "error"]
    # keep images open, limit the number of files, no output
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"\.tif$", "output_format": "none", "script": "", "close_after": False, "max_files": 2, "save_results": False})
    assert len(res["table"]["rows"]) == 2 and len(s.images) == 2
    assert not (folder / "lumen_output").exists()
    with pytest.raises(FileNotFoundError):
        s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": "nomatch", "script": ""})
    with pytest.raises(FileNotFoundError):
        s.run_op("batch.process_folder", {"input_folder": str(folder / "missing"), "script": ""})


def test_process_folder_recursive_and_ome(folder):
    sub = folder / "sub"
    sub.mkdir()
    tifffile.imwrite(str(sub / "deep.tif"), np.zeros((8, 8), np.uint8))
    s = Session()
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"\.tif$", "recursive": True, "output_folder": str(folder / "o"), "output_format": "ome.tif", "script": 'lumen.run("math.add", value=5)'})
    names = [r[0] for r in res["table"]["rows"]]
    assert "deep.tif" in names and len(names) == 4
    assert (folder / "o" / "sub" / "deep.ome.tif").exists()
    from lumen.core.io import read_image
    assert read_image(folder / "o" / "sub" / "deep.ome.tif").data.min() == 5


def test_apply_recorded_macro(folder):
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("filters.gaussian", {"sigma": 1})
    s.run_op("math.multiply", {"value": 0})
    s.save_image(s.active_id, str(folder / "saved.tif"))
    script = macro_as_script(s.macro)
    assert "file.sample" not in script and "lumen.save" not in script and "lumen.open" not in script
    assert 'lumen.run("filters.gaussian"' in script and 'lumen.run("math.multiply"' in script
    res = s.run_op("batch.apply_recorded_macro", {"input_folder": str(folder), "pattern": r"^cell_.*\.tif$", "output_folder": str(folder / "macro_out"), "output_format": "npy"})
    assert all(r[1] == "ok" for r in res["table"]["rows"]) and len(res["table"]["rows"]) == 3
    assert np.load(folder / "macro_out" / "cell_0.npy").max() == 0
    assert s.active.name == "saved.tif"  # Save As renamed blobs.tif (ImageJ semantics); the batch restored it as the active image
    s2 = Session()
    with pytest.raises(ValueError):
        s2.run_op("batch.apply_recorded_macro", {"input_folder": str(folder)})


def test_run_script_on_open_images():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("file.sample", {"name": "Coins"})
    s.run_op("file.sample", {"name": "Moon"})
    first = list(s.images)[0]
    s.select(first)
    before = {iid: im.data.copy() for iid, im in s.images.items()}
    res = s.run_op("batch.run_script_on_open_images", {"script": 'lumen.run("math.set", value=0)\nprint(lumen.image.name)'})
    rows = res["table"]["rows"]
    assert len(rows) == 3 and all(r[1] == "ok" for r in rows)
    assert all(im.data.max() == 0 for im in s.images.values())
    assert s.active_id == first
    for iid in s.images:
        assert s.histories[iid].can_undo()
        s.undo(iid)
        assert np.array_equal(s.images[iid].data, before[iid])
    res = s.run_op("batch.run_script_on_open_images", {"script": "raise RuntimeError('x')", "continue_on_error": False})
    assert len(res["table"]["rows"]) == 1 and res["table"]["rows"][0][1] == "error"
    with pytest.raises(ValueError):
        Session().run_op("batch.run_script_on_open_images", {"script": "pass"})


# ----------------------------------------------------------------- regressions (review round 1)

def test_process_folder_keeps_dotted_names(tmp_path):
    """Output names keep dotted tails (exp1.ch0 / cells_2024.03.12) instead of collapsing onto each other."""
    for i, n in enumerate(["exp1.ch0.tif", "exp1.ch1.tif", "cells_2024.03.12.tif"]):
        tifffile.imwrite(str(tmp_path / n), np.full((8, 8), i * 50, np.uint8))
    for fmt, ext in [("tif", ".tif"), ("png", ".png"), ("npy", ".npy"), ("ome.tif", ".ome.tif")]:
        out = tmp_path / f"out_{fmt.replace('.', '_')}"
        res = Session().run_op("batch.process_folder", {"input_folder": str(tmp_path), "pattern": r"\.tif$", "output_folder": str(out),
                                                        "output_format": fmt, "script": "", "save_results": False})
        expect = ["cells_2024.03.12" + ext, "exp1.ch0" + ext, "exp1.ch1" + ext]
        assert sorted(p.name for p in out.iterdir()) == expect
        assert [r[3] for r in res["table"]["rows"]] == expect
    from lumen.core.io import read_image
    assert read_image(tmp_path / "out_tif" / "exp1.ch0.tif").data.max() == 0 and read_image(tmp_path / "out_tif" / "exp1.ch1.tif").data.max() == 50


def test_results_csv_survives_clear_results(folder):
    """A per-file `Clear Results` (or results_replace) must not drop that file's rows from Results.csv."""
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("analyze.measure", {})  # the session table is non-empty before the batch
    out = folder / "o_clear"
    script = 'lumen.clear_results()\nlumen.run("analyze.measure", roi={"type": "rect", "x": 0, "y": 0, "w": 4, "h": 4})\nassert len(lumen.results) == 1'
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"\.tif$", "output_folder": str(out), "output_format": "none", "script": script})
    assert [r[1] for r in res["table"]["rows"]] == ["ok", "ok", "ok"], res["table"]["rows"]
    lines = (out / "Results.csv").read_text().splitlines()
    assert lines[0].startswith("File,") and [ln.split(",")[0] for ln in lines[1:]] == ["cell_0.tif", "cell_1.tif", "cell_2.tif"]
    assert res["table"]["summary"]["results_csv"] == str(out / "Results.csv")
    # the session table keeps the pre-existing row plus one row per file
    assert len(s.results.rows) == 4
    # analyze.particles with clear_results=True on masks with different particle counts
    masks = folder / "masks"
    masks.mkdir()
    for i in range(3):
        m = np.zeros((32, 32), np.uint8)
        for k in range(i + 1):
            m[2 + k * 8:6 + k * 8, 2:6] = 255
        tifffile.imwrite(str(masks / f"p_{i}.tif"), m)
    out2 = folder / "o_particles"
    script = 'lumen.run("threshold.set", lower=128, upper=255)\nlumen.run("analyze.particles", clear_results=True, summarize=False, add_to_manager=False)'
    res = Session().run_op("batch.process_folder", {"input_folder": str(masks), "pattern": r"\.tif$", "output_folder": str(out2), "output_format": "none", "script": script})
    assert all(r[1] == "ok" for r in res["table"]["rows"]), res["table"]["rows"]
    files = [ln.split(",")[0] for ln in (out2 / "Results.csv").read_text().splitlines()[1:]]
    assert files == ["p_0.tif"] + ["p_1.tif"] * 2 + ["p_2.tif"] * 3


def test_process_folder_saves_image_the_script_produced(tmp_path):
    """A script whose result is a NEW image (Z Project, Duplicate …) saves that image, not the raw input."""
    tifffile.imwrite(str(tmp_path / "stack.tif"), np.stack([np.full((8, 8), i, np.uint8) for i in range(5)]))
    from lumen.core.io import read_image
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})  # a pre-existing image must never be saved under the batch file's name
    out = tmp_path / "proj"
    res = s.run_op("batch.process_folder", {"input_folder": str(tmp_path), "pattern": r"\.tif$", "output_folder": str(out), "output_format": "tif",
                                            "script": 'lumen.run("stacks.z_project")', "save_results": False})
    assert res["table"]["rows"][0][1] == "ok" and res["table"]["rows"][0][3] == "stack.tif"
    data = read_image(out / "stack.tif").data
    assert data.shape[1] == 1 and data.max() == 4
    assert len(s.images) == 1 and s.active.name == "blobs.tif"
    # duplicate + edit saves the edited duplicate; in-place edits still save the opened image
    out2 = tmp_path / "dup"
    s.run_op("batch.process_folder", {"input_folder": str(tmp_path), "pattern": r"\.tif$", "output_folder": str(out2), "output_format": "tif",
                                      "script": 'lumen.run("image.duplicate")\nlumen.run("math.set", value=99)', "save_results": False})
    assert read_image(out2 / "stack.tif").data.max() == 99
    out3 = tmp_path / "inplace"
    s.run_op("batch.process_folder", {"input_folder": str(tmp_path), "pattern": r"\.tif$", "output_folder": str(out3), "output_format": "tif",
                                      "script": 'lumen.run("math.set", value=7)', "save_results": False})
    assert read_image(out3 / "stack.tif").data.max() == 7
    # a script that closes the opened image saves nothing instead of the unrelated pre-existing image
    out4 = tmp_path / "closed"
    res = s.run_op("batch.process_folder", {"input_folder": str(tmp_path), "pattern": r"\.tif$", "output_folder": str(out4), "output_format": "tif",
                                            "script": "lumen.close()", "save_results": False})
    assert res["table"]["rows"][0][1] == "ok" and res["table"]["rows"][0][3] == "" and not list(out4.glob("*.tif"))


def test_script_exit_ends_the_script_only(folder):
    """exit()/quit()/sys.exit() end the script (ImageJ `exit` semantics) – never the request or the batch."""
    import sys
    from lumen.core.scripting import run_script
    s = Session()
    assert run_script(s, "print('a'); exit(); print('b')") == {"stdout": "a\n", "error": None}
    assert run_script(s, "quit()")["error"] is None
    assert run_script(s, "import sys\nsys.exit(0)")["error"] is None
    assert run_script(s, "import sys\nsys.exit('boom')")["error"] == "boom"
    assert "2" in run_script(s, "import sys\nsys.exit(2)")["error"]
    assert not sys.stdin.closed  # the builtin site Quitter would have closed it
    s.run_op("file.sample", {"name": "Blobs"})
    active_before, macro_before = s.active_id, list(s.macro)
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"^cell_.*\.tif$", "output_format": "none", "save_results": False,
                                            "script": 'lumen.run("math.add", value=1)\nexit()'})
    assert [r[1] for r in res["table"]["rows"]] == ["ok", "ok", "ok"]
    assert s.active_id == active_before and len(s.images) == 1
    assert s.macro[:-1] == macro_before and s.macro[-1].startswith('lumen.run("batch.process_folder"')
    # exit("message") is the script's error and honours continue_on_error
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"^cell_.*\.tif$", "output_format": "none", "save_results": False,
                                            "script": 'exit("stop here")', "continue_on_error": False})
    assert [r[1] for r in res["table"]["rows"]] == ["error"] and "stop here" in res["table"]["rows"][0][4]
    assert len(s.macro) == len(macro_before) + 2 and s.macro[-1].startswith('lumen.run("batch.process_folder"')
    # run on open images: same guarantees
    res = s.run_op("batch.run_script_on_open_images", {"script": "exit()"})
    assert res["table"]["rows"][0][1] == "ok" and s.active_id == active_before


def test_failed_or_non_modifying_op_leaves_history_untouched():
    """A failing op leaves no phantom undo entry; Duplicate / Z Project keep the redo stack."""
    from lumen.core.image import LumenImage
    s = Session()
    img = s.add_image(LumenImage(np.zeros((100, 100), np.uint8), name="blank"))
    s.run_op("edit.flood_fill", {"x": 5, "y": 5})
    s.undo(img.id)
    assert _undo_redo(s.histories[img.id]) == {"undo": [], "redo": ["Flood fill"]}
    with pytest.raises(ValueError):
        s.run_op("edit.flood_fill", {"x": 500, "y": 5})
    assert _undo_redo(s.histories[img.id]) == {"undo": [], "redo": ["Flood fill"]}
    with pytest.raises(ValueError):
        s.run_op("color.arrange", {"order": "9"})
    assert _undo_redo(s.histories[img.id]) == {"undo": [], "redo": ["Flood fill"]}
    assert img.data.max() == 0
    s.run_op("image.duplicate", {}, image_id=img.id)
    s.run_op("image.duplicate", {"what": "Current plane"}, image_id=img.id)  # another non-modifying op
    assert _undo_redo(s.histories[img.id]) == {"undo": [], "redo": ["Flood fill"]}
    assert s.redo(img.id) == "Flood fill" and img.data.max() == 255
    # a modifying op after an undo still clears redo, and undo still restores the pixels
    s.undo(img.id)
    s.run_op("math.set", {"value": 3}, image_id=img.id)
    assert _undo_redo(s.histories[img.id]) == {"undo": ["Set"], "redo": []} or s.histories[img.id].labels()["redo"] == []
    assert s.undo(img.id) and img.data.max() == 0


def test_readding_manager_roi_gets_fresh_id():
    """Re-adding a selection picked from the ROI Manager creates an independent entry (ImageJ 'Add')."""
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("selection.add_to_manager", {}, roi={"type": "rect", "x": 10, "y": 10, "w": 20, "h": 20})
    first = dict(s.roi_manager.rois[0])
    s.run_op("selection.add_to_manager", {}, roi=first)
    s.run_op("selection.add_to_manager", {}, roi={**first, "x": 30})  # picked, moved, added again
    ids = [r["id"] for r in s.roi_manager.rois]
    assert len(ids) == 3 and len(set(ids)) == 3
    s.roi_manager.remove([first["id"]])
    assert len(s.roi_manager.rois) == 2
    from lumen.core.scripting import run_script
    assert run_script(s, "lumen.add_roi(lumen.rois[0]); lumen.add_roi(lumen.rois[0])")["error"] is None
    assert len({r["id"] for r in s.roi_manager.rois}) == len(s.roi_manager.rois) == 4


def test_default_pattern_is_case_insensitive(tmp_path):
    from lumen.core.ops.batch import DEFAULT_PATTERN
    import re
    tifffile.imwrite(str(tmp_path / "A01.TIF"), np.zeros((8, 8), np.uint8))
    tifffile.imwrite(str(tmp_path / "b02.Tiff"), np.zeros((8, 8), np.uint8))
    res = Session().run_op("batch.process_folder", {"input_folder": str(tmp_path), "script": "", "output_format": "none", "save_results": False})
    assert [r[0] for r in res["table"]["rows"]] == ["A01.TIF", "b02.Tiff"] and res["table"]["summary"]["total"] == 2
    assert re.search(DEFAULT_PATTERN, "scan.CZI") and re.search(DEFAULT_PATTERN, "x.ND2") and not re.search(DEFAULT_PATTERN, "notes.txt")
    re.compile("^cell_.*" + DEFAULT_PATTERN)  # a user may prepend to the pre-filled dialog value


def test_invalid_file_pattern_is_reported(folder):
    s = Session()
    with pytest.raises(ValueError, match="Invalid file pattern"):
        s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": "[", "script": ""})
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("math.add", {"value": 1})
    with pytest.raises(ValueError, match="Invalid file pattern"):
        s.run_op("batch.apply_recorded_macro", {"input_folder": str(folder), "pattern": "("})
    # the glob habit is understood instead of crashing
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": "*.tif", "script": "", "output_format": "none", "save_results": False})
    assert [r[0] for r in res["table"]["rows"]] == ["cell_0.tif", "cell_1.tif", "cell_2.tif"]


def test_apply_recorded_macro_leaves_out_undone_steps(tmp_path):
    """E7-5: a σ=500 blur the user undid must not run on every file; the result says how many steps were removed."""
    for i in range(2):
        tifffile.imwrite(str(tmp_path / f"flat_{i}.tif"), np.full((16, 16), 10, np.uint8))
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    iid = s.active_id
    s.run_op("filters.gaussian", {"sigma": 500})
    s.undo(iid)
    s.run_op("edit.invert", {})
    s.run_op("math.add", {"value": 5})
    s.undo(iid)
    s.undo(iid)
    s.redo(iid)  # invert is back, the add stays undone
    res = s.run_op("batch.apply_recorded_macro", {"input_folder": str(tmp_path), "pattern": r"^flat_.*\.tif$", "output_folder": str(tmp_path / "o"),
                                                  "output_format": "npy", "save_results": False})
    assert [r[1] for r in res["table"]["rows"]] == ["ok", "ok"]
    assert np.load(tmp_path / "o" / "flat_0.npy").max() == 245  # inverted only: no blur, no +5
    assert "2 ok, 0 failed" in res["message"] and "2 undone steps removed from the recording before the run" in res["message"]
    assert [ln.split('"')[1] for ln in res["table"]["summary"]["undone"]] == ["filters.gaussian", "math.add"]
    assert "filters.gaussian" in res["log"] and "math.add" in res["log"]
    # a recording whose only commands were undone is reported as empty, with the reason
    s2 = Session()
    s2.run_op("file.sample", {"name": "Blobs"})
    s2.run_op("edit.invert", {})
    s2.undo(s2.active_id)
    with pytest.raises(ValueError, match=r"recorder is empty.*1 recorded step undone and left out"):
        s2.run_op("batch.apply_recorded_macro", {"input_folder": str(tmp_path), "pattern": r"^flat_.*\.tif$"})
