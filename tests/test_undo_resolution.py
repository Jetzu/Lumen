"""Undo-aware reading of recorder lines (lumen/core/recorder_lines.py, E7-5 / E8-3) and the ImageJ command names on the
op JSON (E7-20)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from lumen.core import ops  # noqa: F401
from lumen.core.ij_compat import COMMANDS
from lumen.core.recorder_lines import created_titles, creates_undo_entry, parse_line, resolve_undo, undone_lines
from lumen.core.registry import REGISTRY, ij_commands_for
from lumen.core.session import Session
from lumen.server.app import create_app

MARKER = "# Lumen recorder – session started 2026-11-08 10:00:00"
OPEN = 'lumen.run("file.sample", name=\'Blobs\')  # → "blobs.tif"'
GAUSS = 'lumen.run("filters.gaussian", sigma=1.0, scaled_units=False)'
MEDIAN = 'lumen.run("filters.median", radius=2.0, shape=\'disk\')'
INVERT = 'lumen.run("edit.invert")'
MEASURE = 'lumen.run("analyze.measure")'
UNDO = 'lumen.undo("blobs.tif")'
REDO = 'lumen.redo("blobs.tif")'


def test_parse_and_created_titles():
    assert parse_line(OPEN) == ("run", ["file.sample"], {"name": "Blobs"})
    assert created_titles(OPEN) == ["blobs.tif"] and created_titles(GAUSS) == []
    assert created_titles('lumen.run("color.split")  # → "C1-a.tif", "C2-a.tif"') == ["C1-a.tif", "C2-a.tif"]
    assert parse_line(MARKER) is None and parse_line("x = 1") is None
    assert parse_line('lumen.display("a", channel=1, min=10)') == ("display", ["a"], {"channel": 1, "min": 10})


def test_which_commands_leave_an_undo_entry():
    assert creates_undo_entry("filters.gaussian") is True and creates_undo_entry("edit.invert") is True
    assert creates_undo_entry("image.crop") is True
    assert creates_undo_entry("analyze.measure") is False and creates_undo_entry("threshold.compute") is False
    assert creates_undo_entry("rois.clear") is False and creates_undo_entry("analyze.clear_results") is False
    assert creates_undo_entry("results.restore_cleared") is False and creates_undo_entry("batch.process_folder") is False
    assert creates_undo_entry("file.new") is False  # creates an image, changes none
    assert creates_undo_entry("plugin.no_such_op") is None


def test_gaussian_median_undo_undo_redo_keeps_only_the_gaussian():
    lines = [MARKER, OPEN, GAUSS, MEDIAN, UNDO, UNDO, REDO]
    out, dropped = resolve_undo(lines)
    assert out == [MARKER, OPEN, GAUSS]
    assert [(d["index"], d["kind"]) for d in dropped] == [(3, "undone"), (4, "undo"), (5, "undo"), (6, "redo")]
    assert undone_lines(dropped) == [MEDIAN]
    assert all(d["target"] == "blobs.tif" for d in dropped)


def test_undo_after_a_measurement_cancels_the_last_pixel_command():
    lines = [OPEN, GAUSS, MEASURE, UNDO]
    out, dropped = resolve_undo(lines)
    assert out == [OPEN, MEASURE] and undone_lines(dropped) == [GAUSS]
    # a threshold computed, ROI Manager and Results commands are not undo steps either
    lines = [OPEN, INVERT, 'lumen.run("threshold.compute", method=\'Otsu\', dark_background=True)', 'lumen.run("rois.clear")', 'lumen.run("analyze.clear_results")', UNDO]
    out, dropped = resolve_undo(lines)
    assert undone_lines(dropped) == [INVERT] and len(out) == 4


def test_a_new_step_after_an_undo_clears_the_redo_stack():
    lines = [OPEN, GAUSS, UNDO, INVERT, REDO]
    out, dropped = resolve_undo(lines)
    assert out == [OPEN, INVERT, REDO]  # the redo had nothing to restore: it stays for the caller to report
    assert undone_lines(dropped) == [GAUSS]


def test_unmatched_undo_stays_and_unknown_commands_block_the_match():
    out, dropped = resolve_undo([OPEN, MEASURE, UNDO])
    assert out == [OPEN, MEASURE, UNDO] and dropped == []
    out, dropped = resolve_undo([OPEN, GAUSS, 'lumen.run("plugin.no_such_op", k=1)', UNDO])
    assert dropped == [] and len(out) == 4  # the unknown command may or may not be a step: nothing is cancelled
    # an undo of an image whose stack is empty does not touch another image's steps
    out, dropped = resolve_undo([OPEN, GAUSS, 'lumen.run("file.sample", name=\'Coins\')  # → "coins.png"', 'lumen.undo("coins.png")'])
    assert dropped == [] and GAUSS in out


def test_stacks_are_per_image_and_follow_select_close_rename_and_save_as():
    coins = 'lumen.run("file.sample", name=\'Coins\')  # → "coins.png"'
    lines = [OPEN, GAUSS, coins, INVERT, 'lumen.select("blobs.tif")', MEDIAN, 'lumen.undo("coins.png")', UNDO]
    out, dropped = resolve_undo(lines)
    assert undone_lines(dropped) == [INVERT, MEDIAN] and GAUSS in out
    # Save As renames the window: later undo lines name the file name
    lines = [OPEN, GAUSS, 'lumen.save("/tmp/out/saved.tif")', 'lumen.undo("saved.tif")']
    out, dropped = resolve_undo(lines)
    assert undone_lines(dropped) == [GAUSS]
    lines = [OPEN, GAUSS, 'lumen.run("image.rename", title=\'other\')', 'lumen.undo("other")']
    assert undone_lines(resolve_undo(lines)[1]) == [GAUSS]
    # a closed image forgets its stack; display lines name their image
    lines = [OPEN, GAUSS, 'lumen.close("blobs.tif")', coins, 'lumen.display("coins.png", mode=\'single\')', INVERT, UNDO]
    out, dropped = resolve_undo(lines)
    assert dropped == []  # blobs.tif is closed: its undo cancels nothing
    lines = [OPEN, GAUSS, coins, 'lumen.display("blobs.tif", channel=1, min=5, max=200)', INVERT, UNDO]
    assert undone_lines(resolve_undo(lines)[1]) == [INVERT]


def test_resolution_matches_a_real_session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    iid = s.active_id
    s.run_op("filters.gaussian", {"sigma": 1})
    s.run_op("filters.median", {"radius": 2})
    s.undo(iid)
    s.undo(iid)
    s.redo(iid)
    s.run_op("analyze.measure", {})
    s.run_op("edit.invert", {})
    s.undo(iid)
    out, dropped = resolve_undo(list(s.macro))
    assert [ln.split("(")[0] for ln in out] == ["lumen.run", "lumen.run", "lumen.run"]
    assert any("filters.gaussian" in ln for ln in out) and any("analyze.measure" in ln for ln in out)
    assert [ln.split('"')[1] for ln in undone_lines(dropped)] == ["filters.median", "edit.invert"]
    assert s.histories[iid].labels()["undo"] == ["Gaussian Blur"]  # the session agrees: only the gaussian is applied


# ----------------------------------------------------------------------------- E7-20 ImageJ names on the op JSON
def test_ops_json_lists_the_imagej_command_names():
    assert ij_commands_for("filters.gaussian") == ["Gaussian Blur..."]
    assert ij_commands_for("filters.median") == ["Median..."]
    assert "Enhance Local Contrast (CLAHE)" in ij_commands_for("process.clahe") and "CLAHE" in ij_commands_for("process.clahe")
    assert ij_commands_for("adjust.bleach_correction") == ["Bleach Correction"]
    assert ij_commands_for("batch.process_folder") == [] and ij_commands_for("no.such.op") == []
    assert ij_commands_for("stacks.reslice") and ij_commands_for("stacks.reslice_line")  # one handler serving two ops lists both
    assert REGISTRY.ops["filters.gaussian"].to_dict()["ij_commands"] == ["Gaussian Blur..."]
    ops_json = {o["id"]: o for o in REGISTRY.list()}
    assert all(isinstance(o["ij_commands"], list) for o in ops_json.values())
    mapped = {o for o in ops_json if ops_json[o]["ij_commands"]}
    assert len(mapped) > 200
    registered = {lumen.strip() for _fn, lumen, _n in COMMANDS.values() for lumen in str(lumen).split(" / ")} & set(ops_json)
    assert mapped == registered
    c = TestClient(create_app(Session()))
    api = {o["id"]: o for o in c.get("/api/ops").json()}
    assert api["filters.gaussian"]["ij_commands"] == ["Gaussian Blur..."] and api["filters.unsharp"]["ij_commands"] == ["Unsharp Mask..."]
    assert {o["id"]: o["ij_commands"] for o in c.get("/api/meta").json()["ops"]}["filters.median"] == ["Median..."]
