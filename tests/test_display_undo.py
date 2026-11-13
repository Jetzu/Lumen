"""Display edits as undo steps (E5-8 / E3-9), full parameters in History, Set LUT's channel, and the channel in the
particles message (E9-2)."""
import numpy as np

import lumen.core.ops  # noqa: F401
from lumen.core.recorder_lines import resolve_undo
from lumen.core.session import Session


def _three_channels() -> tuple[Session, str]:
    s = Session()
    s.run_op("file.new", {"title": "mc", "type": "16-bit", "width": 32, "height": 24, "channels": 3, "fill": "Ramp"})
    return s, s.active.id


def _labels(s: Session, iid: str) -> dict:
    return s.histories[iid].labels()


def test_display_edit_is_an_undo_step_and_a_plane_change_is_not():
    s, iid = _three_channels()
    img = s.active
    assert _labels(s, iid)["undo"] == []
    s.update_display(iid, {"channels": [{}, {"min": 100, "max": 1500}, {}]}, record=True)
    lab = _labels(s, iid)
    assert lab["undo"] == ["Display"] and lab["undo_detail"][0].startswith("channel 2") and "max=1500" in lab["undo_detail"][0]
    assert img.display.channels[1].max == 1500
    # a slider drag: more patches of the same channel merge into the same step
    s.update_display(iid, {"channels": [{}, {"max": 1400}, {}]}, record=True)
    s.update_display(iid, {"channels": [{}, {"min": 120}, {}]}, record=True)
    assert _labels(s, iid)["undo"] == ["Display"]
    # another channel, the mode and the LUT are their own steps; z / c changes are none
    s.update_display(iid, {"channels": [{"lut": "Fire"}, {}, {}]}, record=True)
    s.update_display(iid, {"mode": "single"}, record=True)
    s.update_display(iid, {"c": 2}, record=True)
    lab = _labels(s, iid)
    assert lab["undo"] == ["Display", "Display", "Display mode"] and lab["undo_detail"][2] == "composite → single"
    assert s.histories[iid].can_undo()
    # undo brings the display back step by step, pixels untouched
    before = np.asarray(img.data).copy()
    assert s.undo(iid) == "Display mode" and img.display.mode == "composite"
    assert s.undo(iid) == "Display" and img.display.channels[0].lut != "Fire"
    assert s.undo(iid) == "Display" and img.display.channels[1].max != 1400 and img.display.channels[1].min != 120
    np.testing.assert_array_equal(np.asarray(img.data), before)
    assert s.redo(iid) == "Display" and img.display.channels[1].max == 1400 and img.display.channels[1].min == 120
    # the recorder wrote the display lines and the undo / redo lines
    lines = [ln for ln in s.macro if not ln.startswith("#")]
    assert any(ln.startswith('lumen.display("mc", channel=2') for ln in lines)
    assert sum(ln.startswith('lumen.undo("mc")') for ln in lines) == 3 and sum(ln.startswith('lumen.redo("mc")') for ln in lines) == 1


def test_resolve_undo_treats_display_edits_as_steps():
    lines = [
        'lumen.run("file.new", title=\'a\', type=\'8-bit\', width=8, height=8, slices=1, channels=2, frames=1, fill=\'Black\', period=32)  # → "a"',
        'lumen.run("filters.gaussian", sigma=2.0, scaled_units=False)',
        'lumen.display("a", channel=2, min=10, max=200)',
        'lumen.undo("a")',  # cancels the display edit, not the gaussian
    ]
    out, dropped = resolve_undo(lines)
    assert out == lines[:2] and {d["kind"] for d in dropped} == {"undone", "undo"} and dropped[0]["line"] == lines[2]
    out, dropped = resolve_undo(lines[:2] + ['lumen.display("a", z=3)', 'lumen.undo("a")'])
    assert out == [lines[0], 'lumen.display("a", z=3)']  # a plane change is no step: the undo cancels the gaussian


def test_history_detail_lists_every_parameter_and_the_scope_on_stacks():
    s = Session()
    s.run_op("file.new", {"title": "st", "type": "8-bit", "width": 16, "height": 16, "slices": 3, "fill": "Ramp"})
    iid = s.active.id
    s.run_op("filters.gaussian", {"sigma": 2.0})
    s.run_op("filters.median", {"radius": 2.0}, scope="slice")
    lab = _labels(s, iid)
    assert lab["undo_detail"][0] == "sigma=2.0, scaled_units=False, scope=stack (all planes)"  # defaults are shown too
    assert lab["undo_detail"][1] == "radius=2.0, shape='disk', scope=slice (current plane)"
    s2 = Session()
    s2.run_op("file.new", {"title": "flat", "type": "8-bit", "width": 16, "height": 16, "fill": "Ramp"})
    s2.run_op("filters.gaussian", {"sigma": 3.0})
    assert _labels(s2, s2.active.id)["undo_detail"][0] == "sigma=3.0, scaled_units=False"  # no scope on a single plane


def test_set_lut_names_its_channel_in_the_recording():
    s, iid = _three_channels()
    s.active.display.c = 1
    s.run_op("color.lut", {"lut": "Fire"})
    assert s.active.display.channels[1].lut == "Fire" and s.active.display.channels[0].lut != "Fire"
    assert any(ln.startswith('lumen.run("color.lut", lut=\'Fire\', all_channels=False, channel=2)') for ln in s.macro)
    s.run_op("color.lut", {"lut": "Ice", "channel": 3})
    assert s.active.display.channels[2].lut == "Ice"
    s1 = Session()
    s1.run_op("file.new", {"title": "one", "type": "8-bit", "width": 8, "height": 8, "fill": "Ramp"})
    s1.run_op("color.lut", {"lut": "Fire"})
    assert any(ln.startswith('lumen.run("color.lut", lut=\'Fire\', all_channels=False)') for ln in s1.macro)  # no channel on 1-channel images


def test_particles_message_names_the_channel():
    s, iid = _three_channels()
    s.active.display.channels[0].name = "DAPI"
    out = s.run_op("analyze.particles", {"auto_threshold": True, "channel": 1})
    assert "on channel 1 (DAPI)" in (out.get("message") or out.get("log") or "")
    s1 = Session()
    s1.run_op("file.sample", {"name": "Blobs"})
    out = s1.run_op("analyze.particles", {"auto_threshold": True})
    assert "on channel" not in (out.get("message") or out.get("log") or "")


def test_restore_cancels_the_clear_in_replays():
    """"Undo clear" took a clear back: neither the clear nor the restore belongs in a batch or a macro (E7-5, E10)."""
    lines = [
        'lumen.run("file.sample", name=\'Blobs\')  # → "blobs.tif"',
        'lumen.run("analyze.particles", min_size=0.0, max_size=0.0, show=\'Nothing\', auto_threshold=True)',
        'lumen.run("rois.clear")',
        'lumen.run("rois.restore_cleared")',
        'lumen.run("analyze.clear_results")',
        'lumen.run("results.restore_cleared")',
        'lumen.run("rois.delete", indices=\'1\')',
    ]
    out, dropped = resolve_undo(lines)
    assert out == [lines[0], lines[1], lines[6]]
    assert [d["line"] for d in dropped] == lines[2:6]
    # a restore with nothing to take back stays (the caller reports it)
    out, dropped = resolve_undo([lines[0], 'lumen.run("rois.restore_cleared")'])
    assert out == [lines[0], 'lumen.run("rois.restore_cleared")'] and dropped == []
