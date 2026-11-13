"""Tests for Edit / Options / 8-bit Color / LUT viewers / Help (lumen/core/ops/edit_extra.py)."""
import math

import numpy as np
import pytest

from lumen.core import luts
from lumen.core import ops  # noqa: F401
from lumen.core import roi as roimod
from lumen.core.image import LumenImage
from lumen.core.ops import edit_extra
from lumen.core.registry import REGISTRY
from lumen.core.session import Session


@pytest.fixture(autouse=True)
def _restore_luts():
    """Runtime LUT registration is global: undo it after every test so other test modules see the built-ins only."""
    saved_tables, saved_names = dict(luts._LUTS), list(luts.LUT_NAMES)
    yield
    luts._LUTS.clear()
    luts._LUTS.update(saved_tables)
    luts.LUT_NAMES[:] = saved_names
    luts._invalidate_caches()


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


def _ramp(session, h=40, w=60, name="ramp.tif"):
    arr = (np.arange(h * w, dtype=np.float32).reshape(h, w) % 200).astype(np.uint8)
    return session.add_image(LumenImage(arr, name=name))


# ---------------------------------------------------------------- Cut / Paste Control / paste modes
def test_cut_copies_and_clears(session):
    img = session.active
    before = img.data.copy()
    roi = {"type": "rect", "x": 10, "y": 20, "w": 30, "h": 15}
    session.run_op("edit.cut", {}, roi=roi)
    clip = session.settings["clipboard"]
    assert clip["data"].shape == (1, 15, 30) and np.array_equal(clip["data"][0], before[0, 0, 0, 20:35, 10:40]) and clip["mask"] is not None
    assert img.data[0, 0, 0, 20:35, 10:40].max() == 0
    assert np.array_equal(img.data[0, 0, 0, 40:, 50:], before[0, 0, 0, 40:, 50:])
    session.undo(img.id)
    assert np.array_equal(img.data, before)
    session.settings["background"] = 7
    session.run_op("edit.cut", {}, roi={"type": "oval", "x": 0, "y": 0, "w": 20, "h": 20})
    assert img.data[0, 0, 0, 10, 10] == 7 and img.data[0, 0, 0, 0, 0] == before[0, 0, 0, 0, 0]  # outside the oval untouched


def test_paste_control_and_modes():
    s = Session()
    base = np.full((10, 10), 100, np.uint8)
    img = s.add_image(LumenImage(base.copy(), name="base.tif"))
    src = s.add_image(LumenImage(np.full((4, 4), 50, np.uint8), name="src.tif"))
    src.data[0, 0, 0, 0, 0] = 0
    s.run_op("edit.copy", {}, image_id=src.id)
    assert s.run_op("edit.paste_control", {"mode": "Add"})["message"] == "Paste mode: Add"
    assert s.settings["paste_mode"] == "Add"
    at = {"type": "rect", "x": 3, "y": 3, "w": 4, "h": 4}
    out = s.run_op("edit.paste_with_mode", {}, image_id=img.id, roi=at)
    assert out["roi"] == dict(out["roi"], x=3, y=3, w=4, h=4)
    p = img.data[0, 0, 0]
    assert p[4, 4] == 150 and p[3, 3] == 100 + 0 and p[0, 0] == 100
    s.undo(img.id)
    expect = {"Copy": 50, "Blend": 75, "Average": 75, "Difference": 50, "Transparent-white": 50, "Transparent-zero": 50, "AND": 100 & 50, "OR": 100 | 50,
              "XOR": 100 ^ 50, "Subtract": 50, "Multiply": 255, "Divide": 2, "Min": 50, "Max": 100}
    for mode, val in expect.items():
        s.run_op("edit.paste_with_mode", {"mode": mode}, image_id=img.id, roi=at)
        p = img.data[0, 0, 0]
        assert p[5, 5] == val, mode
        if mode == "Transparent-zero":
            assert p[3, 3] == 100  # zero source pixels are transparent
        if mode == "Divide":
            assert p[3, 3] == 100  # division by zero keeps the target
        s.undo(img.id)
    with pytest.raises(ValueError):
        s.run_op("edit.paste_control", {"mode": "Bogus"})


def test_blend_paste_float_and_uint16():
    d = np.array([[1.0, 2.0]], np.float32)
    s = np.array([[3.0, 0.0]], np.float32)
    assert edit_extra.blend_paste(d, s, "Multiply").tolist() == [[3.0, 0.0]]
    assert np.allclose(edit_extra.blend_paste(d, s, "Divide"), [[1 / 3, 2.0]])
    d16 = np.array([[60000]], np.uint16)
    assert edit_extra.blend_paste(d16, np.array([[10000]], np.uint16), "Add").tolist() == [[65535]]


# ---------------------------------------------------------------- Selection
def test_fit_circle_points_lines_and_areas(session):
    th = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    rng = np.random.default_rng(0)
    pts = np.stack([100 + 30 * np.cos(th) + rng.normal(0, 0.2, 24), 80 + 30 * np.sin(th) + rng.normal(0, 0.2, 24)], 1)
    out = session.run_op("selection.fit_circle", {}, roi={"type": "point", "points": pts.tolist()})
    r = out["roi"]
    assert r["type"] == "oval" and r["w"] == pytest.approx(60, abs=0.5) and r["x"] + r["w"] / 2 == pytest.approx(100, abs=0.3) and r["y"] + r["h"] / 2 == pytest.approx(80, abs=0.3)
    out = session.run_op("selection.fit_circle", {}, roi={"type": "line", "x1": 10, "y1": 10, "x2": 40, "y2": 50})
    r = out["roi"]
    assert r["x"] + r["w"] / 2 == pytest.approx(25) and r["w"] == pytest.approx(50)
    out = session.run_op("selection.fit_circle", {}, roi={"type": "rect", "x": 20, "y": 30, "w": 40, "h": 10})
    r = out["roi"]
    assert r["w"] == pytest.approx(2 * math.sqrt(400 / math.pi), abs=0.05) and r["x"] + r["w"] / 2 == pytest.approx(40, abs=0.01)
    with pytest.raises(ValueError, match="collinear"):
        session.run_op("selection.fit_circle", {}, roi={"type": "point", "points": [[0, 0], [1, 1], [2, 2], [3, 3]]})


def test_fit_rectangle_rotated(session):
    a = math.radians(30)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    rect = np.array([[-40, -10], [40, -10], [40, 10], [-40, 10], [0, 0], [10, 5]], float)
    pts = rect @ R.T + [120, 100]
    out = session.run_op("selection.fit_rectangle", {}, roi={"type": "polygon", "points": pts.tolist()})
    r = out["roi"]
    assert r["type"] == "polygon" and len(r["points"]) == 4
    c = np.asarray(r["points"])
    sides = sorted([np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[1]))])
    assert sides[0] == pytest.approx(20, abs=2e-3) and sides[1] == pytest.approx(80, abs=2e-3)
    assert c.mean(axis=0) == pytest.approx([120, 100], abs=2e-3)
    with pytest.raises(ValueError):
        session.run_op("selection.fit_rectangle", {}, roi={"type": "line", "x1": 0, "y1": 0, "x2": 5, "y2": 5})


def test_fit_spline_closed_and_open(session):
    sq = [[10, 10], [60, 12], [58, 50], [12, 48]]
    out = session.run_op("selection.fit_spline", {"points": 1200}, roi={"type": "polygon", "points": sq})
    r = out["roi"]
    assert r["type"] == "polygon" and len(r["points"]) == 1200
    pts = np.asarray(r["points"])
    for v in sq:  # interpolating: passes through every vertex (samples are ~0.15 px apart)
        assert np.min(np.hypot(*(pts - v).T)) < 0.2
    assert len(session.run_op("selection.fit_spline", {}, roi={"type": "polygon", "points": sq})["roi"]["points"]) == 86  # ~perimeter / 2
    out = session.run_op("selection.fit_spline", {}, roi={"type": "polyline", "points": [[0, 0], [10, 20], [20, 0], [30, 20]], "width": 3})
    r = out["roi"]
    assert r["type"] == "polyline" and r["width"] == 3 and len(r["points"]) >= 50
    assert r["points"][0] == pytest.approx([0, 0], abs=1e-6) and r["points"][-1] == pytest.approx([30, 20], abs=1e-6)
    with pytest.raises(ValueError):
        session.run_op("selection.fit_spline", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5})


def test_image_to_selection(session):
    img = session.active
    m = np.zeros((img.height, img.width), np.uint8)
    m[10:30, 20:50] = 255
    mask_img = session.add_image(LumenImage(m, name="mask.tif"), activate=False)
    out = session.run_op("selection.image_to_selection", {"image": "mask.tif", "opacity": 40}, image_id=img.id)
    r = out["roi"]
    assert r["type"] == "mask" and (r["x"], r["y"], r["w"], r["h"]) == (20, 10, 30, 20) and r["opacity"] == 40
    assert roimod.to_mask(r, m.shape).sum() == 600
    small = session.add_image(LumenImage(np.full((5, 5), 1, np.uint8), name="small.tif"), activate=False)
    out = session.run_op("selection.image_to_selection", {"image": small.id}, image_id=img.id)
    assert (out["roi"]["w"], out["roi"]["h"]) == (5, 5)
    with pytest.raises(ValueError):
        session.run_op("selection.image_to_selection", {}, image_id=img.id)


def test_select_bounding_box_guess_background():
    s = Session()
    a = np.full((50, 70), 10, np.uint8)
    a[20:40, 30:55] = 200
    a[0, 0] = 11  # a noisy border pixel does not change the mode
    s.add_image(LumenImage(a, name="bb.tif"))
    out = s.run_op("selection.select_bounding_box", {"tolerance": 2})
    r = out["roi"]
    assert (r["x"], r["y"], r["w"], r["h"]) == (30, 20, 25, 20) and "Background 10" in out["message"]
    out = s.run_op("selection.select_bounding_box", {"tolerance": 0})
    assert (out["roi"]["x"], out["roi"]["y"]) == (0, 0)  # the noisy pixel now counts
    s.add_image(LumenImage(np.zeros((5, 5), np.uint8), name="flat.tif"))
    with pytest.raises(ValueError, match="background"):
        s.run_op("selection.select_bounding_box", {})
    rgb = np.zeros((20, 20, 3), np.uint8)
    rgb[5:8, 6:10, 2] = 90
    s.add_image(LumenImage(rgb, name="rgb.png", axes="YXS", rgb=True))
    out = s.run_op("selection.select_bounding_box", {})
    assert (out["roi"]["x"], out["roi"]["y"], out["roi"]["w"], out["roi"]["h"]) == (6, 5, 4, 3)


def test_selection_properties(session):
    roi = {"type": "rect", "x": 1, "y": 2, "w": 3, "h": 4, "name": "old"}
    out = session.run_op("selection.properties", {"name": "cell", "color": "red", "stroke_width": 3, "fill": "#00ff00", "position": "1,2,3"}, roi=roi)
    r = out["roi"]
    assert r["name"] == "cell" and r["color"] == "#ff0000" and r["stroke_width"] == 3 and r["fill"] == "#00ff00" and r["position"] == {"t": 0, "z": 1, "c": 2}
    out = session.run_op("selection.properties", {"fill": "none", "position": "none"}, roi=r)
    assert "fill" not in out["roi"] and "position" not in out["roi"] and out["roi"]["name"] == "cell" and out["roi"]["color"] == "#ff0000"
    with pytest.raises(ValueError):
        session.run_op("selection.properties", {"color": "not-a-colour"}, roi=roi)


def test_rounded_rect(session):
    out = session.run_op("selection.rounded_rect", {"x": 10, "y": 20, "w": 100, "h": 60, "corner_radius": 15})
    r = out["roi"]
    pts = np.asarray(r["points"])
    assert r["type"] == "polygon" and len(pts) == 36 and r["corner_radius"] == 15
    assert pts[:, 0].min() == pytest.approx(10) and pts[:, 0].max() == pytest.approx(110) and pts[:, 1].min() == pytest.approx(20) and pts[:, 1].max() == pytest.approx(80)
    m = roimod.to_mask(r, (120, 140))
    assert not m[20, 10] and m[50, 60] and m[21, 60]  # corner cut away, centre and edge midpoints inside
    sq = session.run_op("selection.rounded_rect", {"x": 0, "y": 0, "w": 10, "h": 10, "corner_radius": 0})["roi"]
    assert len(sq["points"]) == 4


# ---------------------------------------------------------------- Options / preferences
def test_options_write_settings():
    s = Session()
    out = s.run_op("edit.options", {"roi_color": "#ff0000", "roi_stroke_width": 2.5, "font_size": 18, "jpeg_quality": 60, "default_save_format": "png",
                                    "point_size": "8", "interpolate_zoom": True, "antialiased_text": False, "plot_y_min": -1, "plot_y_max": 5})
    st = s.settings
    assert st["roi_color"] == "#ff0000" and st["roi_stroke_width"] == 2.5 and st["font_size"] == 18 and st["jpeg_quality"] == 60
    assert st["default_save_format"] == "png" and st["point_size"] == 8 and st["interpolate_zoom"] is True and st["antialiased_text"] is False
    assert st["plot_y_min"] == -1 and st["plot_y_max"] == 5 and st["font_family"] == "sans-serif" and st["foreground"] == 255
    assert "roi_color" in out["message"]
    for key in ("roi_color", "roi_stroke_width", "font_size", "font_family", "foreground", "background", "jpeg_quality", "default_save_format", "point_size",
                "point_color", "arrow_style", "arrow_head_size", "arrow_width", "plot_y_min", "plot_y_max", "plot_interpolate", "interpolate_zoom", "antialiased_text"):
        assert key in edit_extra.PREFS and key in st


def test_option_sub_dialogs_and_reset():
    s = Session()
    s.run_op("edit.options_io", {"jpeg_quality": 42, "default_save_format": "zip"})
    s.run_op("edit.options_point_tool", {"point_size": "12", "point_color": "0,255,0"})
    s.run_op("edit.options_arrow_tool", {"arrow_style": "Open", "arrow_head_size": 15, "arrow_width": 3})
    s.run_op("edit.options_fonts", {"font_family": "monospace", "font_size": 11})
    s.run_op("edit.options_appearance", {"interpolate_zoom": True})
    s.run_op("edit.options_plots", {"plot_y_min": 0, "plot_y_max": 100, "plot_interpolate": False})
    s.run_op("edit.options_roi_defaults", {"roi_color": "#00f", "roi_stroke_width": 4})
    s.run_op("edit.options_rounded_rect", {"rounded_rect_radius": 33})
    s.run_op("edit.options_misc", {"paste_mode": "Max", "divide_by_zero": "NaN"})
    st = s.settings
    assert st["jpeg_quality"] == 42 and st["default_save_format"] == "zip" and st["point_size"] == 12 and st["point_color"] == "#00ff00"
    assert st["arrow_style"] == "Open" and st["arrow_head_size"] == 15 and st["arrow_width"] == 3 and st["font_family"] == "monospace" and st["font_size"] == 11
    assert st["interpolate_zoom"] is True and st["plot_y_max"] == 100 and st["plot_interpolate"] is False and st["roi_color"] == "#0000ff" and st["roi_stroke_width"] == 4
    assert st["rounded_rect_radius"] == 33 and st["paste_mode"] == "Max" and st["divide_by_zero"] == "NaN"
    s.run_op("edit.options_reset", {})
    assert st["jpeg_quality"] == 85 and st["roi_color"] == "#ffd54a" and st["paste_mode"] == "Copy" and st["foreground"] == 255
    with pytest.raises(ValueError):
        s.run_op("edit.options_io", {"default_save_format": "exe"})
    ids = {"edit.options", "edit.options_io", "edit.options_point_tool", "edit.options_arrow_tool", "edit.options_fonts", "edit.options_appearance",
           "edit.options_plots", "edit.options_roi_defaults", "edit.options_rounded_rect", "edit.options_misc"}
    assert ids <= set(REGISTRY.ops) and all((REGISTRY.ops[i].menu or REGISTRY.ops[i].category) == "Edit/Options" for i in ids)


# ---------------------------------------------------------------- Image ▸ Type ▸ 8-bit Color
def test_8bit_color_quantise():
    from lumen.core.render import render_rgb
    s = Session()
    s.run_op("file.sample", {"name": "Astronaut (RGB)"})
    img = s.active
    before = render_rgb(img).astype(np.float32)
    out = s.run_op("type.8bit_color", {"colors": 64})
    assert img.dtype == np.uint8 and img.nc == 1 and not img.rgb and img.bit_depth == "8-bit"
    lut_name = img.display.channels[0].lut
    assert lut_name == "Indexed astronaut.png" and lut_name in luts.LUT_NAMES and "LUT" in out["message"]
    assert len(np.unique(img.data)) <= 64
    after = render_rgb(img).astype(np.float32)
    assert np.abs(after - before).mean() < 12  # the palette reproduces the picture
    s.undo(img.id)
    assert img.rgb and img.nc == 3
    s.add_image(LumenImage(np.zeros((4, 4), np.uint8), name="gray.tif"))
    with pytest.raises(ValueError, match="RGB"):
        s.run_op("type.8bit_color", {})


def test_8bit_color_stack_shares_palette():
    s = Session()
    rgb = np.zeros((2, 8, 8, 3), np.uint8)
    rgb[0, :, :4] = (255, 0, 0)
    rgb[1, :, 4:] = (0, 0, 255)
    img = s.add_image(LumenImage(rgb, name="st.tif", axes="ZYXS", rgb=True))
    s.run_op("type.8bit_color", {"colors": 8})
    assert img.shape == (1, 2, 1, 8, 8)
    table = luts.get_lut(img.display.channels[0].lut)
    assert tuple(table[int(img.data[0, 0, 0, 0, 0])]) == (255, 0, 0) and tuple(table[int(img.data[0, 1, 0, 0, 7])]) == (0, 0, 255)


# ---------------------------------------------------------------- LUT viewers / editor
def test_show_lut_and_display_luts(session):
    img = session.active
    session.run_op("color.lut", {"lut": "Fire"})
    img.display.channels[0].min, img.display.channels[0].max = 0, 100
    out = session.run_op("color.show_lut", {})
    card = session.images[out["new_images"][0]]
    assert card.rgb and card.name == "LUT of blobs.tif" and card.metadata["lut"] == "Fire" and card.width == 256 + 48
    bar_row = card.data[0, 0, :, 22 + 16, 24:24 + 256].T  # a row inside the gradient bar
    assert np.abs(bar_row.astype(int) - luts.get_lut("Fire").astype(int)).max() <= 1
    assert "0 – 100" in out["message"]
    out = session.run_op("color.display_luts", {"row_height": 10})
    montage = session.images[out["new_images"][0]]
    assert montage.rgb and montage.height == len(luts.LUT_NAMES) * 14 + 4 and f"{len(luts.LUT_NAMES)} lookup tables" in out["message"]
    assert tuple(montage.data[0, 0, :, 4 + 5, 110 + 255]) == tuple(luts.get_lut(luts.LUT_NAMES[0])[255])


def test_edit_lut_gradient_and_table(session):
    out = session.run_op("color.edit_lut", {"name": "Teal ramp", "start": "#000000", "end": "#00ffff"})
    assert "Teal ramp" in luts.LUT_NAMES and session.active.display.channels[0].lut == "Teal ramp" and "applied" in out["message"]
    assert luts.get_lut("Teal ramp")[255].tolist() == [0, 255, 255] and luts.get_lut("Teal ramp")[128].tolist() == [0, 128, 128]
    table = "\n".join(f"{i} {255 - i} 128" for i in range(0, 256, 5))  # 52 lines → interpolated
    session.run_op("color.edit_lut", {"name": "Tabled", "table": table, "apply": False})
    t = luts.get_lut("Tabled")
    assert t.shape == (256, 3) and t[0].tolist() == [0, 255, 128] and t[255].tolist() == [255, 0, 128] and session.active.display.channels[0].lut == "Teal ramp"
    s2 = Session()
    s2.run_op("color.edit_lut", {"name": "NoImage", "start": "red", "end": "blue"})  # no image open is fine
    assert luts.get_lut("NoImage")[0].tolist() == [255, 0, 0]


# ---------------------------------------------------------------- Help
def test_help_links_and_about_plugins():
    s = Session()
    out = s.run_op("help.links", {})
    rows = out["table"]["rows"]
    assert out["table"]["columns"] == ["Group", "Title", "URL"] and any(r[2].startswith("https://imagej.net") for r in rows) and any(r[0] == "Fiji" for r in rows)
    out = s.run_op("help.about_plugins", {})
    rows = {r[0]: r for r in out["table"]["rows"]}
    assert "edit_extra" in rows and rows["edit_extra"][1] == "built-in" and rows["edit_extra"][2] >= 25 and rows["edit_extra"][5] == "loaded"
    assert "file_extra" in rows and "Edit" in rows["edit_extra"][3] and "Help" in rows["edit_extra"][3]
    assert sum(r[2] for r in rows.values()) == len(REGISTRY.ops)
    assert "commands in" in out["message"]


def test_help_command():
    s = Session()
    out = s.run_op("help.command", {"command": "filters.gaussian"})
    assert out["table"]["columns"][0] == "Parameter" and any(r[0] == "sigma" for r in out["table"]["rows"])
    assert "Gaussian" in out["message"] and "Menu:" in out["message"]
    out = s.run_op("help.command", {"command": "Fit Circle"})  # by display name
    assert "selection.fit_circle" in out["message"] and "needs any selection" in out["message"]
    out = s.run_op("help.command", {"command": "options_"})  # several matches → list
    assert out["table"]["columns"][0] == "Id" and len(out["table"]["rows"]) >= 10
    with pytest.raises(ValueError, match="No command matches"):
        s.run_op("help.command", {"command": "zzz-not-a-command"})
    with pytest.raises(ValueError):
        s.run_op("help.command", {"command": ""})


# ---------------------------------------------------------------- regressions from the code review
def test_help_command_one_sided_ranges():
    s = Session()
    out = s.run_op("help.command", {"command": "selection.rounded_rect"})  # w/h/corner_radius have a min but no max
    rng = {r[0]: r[4] for r in out["table"]["rows"]}
    assert rng["w"] == "1–" and rng["corner_radius"] == "0–" and rng["arc_points"] == "1–64" and rng["x"] == ""
    out = s.run_op("help.command", {"command": "file.open_recent"})
    assert any(r[4].endswith("–") for r in out["table"]["rows"])
    radius = next(r for r in s.run_op("help.command", {"command": "filters.median"})["table"]["rows"] if r[0] == "radius")
    assert radius[4] == "0.5–1000"  # two-sided ranges render as before (Gaussian's sigma is unbounded now, like ImageJ's)
    for oid in list(REGISTRY.ops):  # every registered command can be described
        s.run_op("help.command", {"command": oid}, record=False)


def test_cut_requires_area_selection(session):
    img = session.active
    before = img.data.copy()
    assert REGISTRY.ops["edit.cut"].requires_roi == "area"
    for roi in (None, {"type": "line", "x1": 0, "y1": 0, "x2": 50, "y2": 50}, {"type": "point", "points": [[10, 10]]}):
        with pytest.raises(ValueError, match="area selection"):
            session.run_op("edit.cut", {}, roi=roi)
    assert np.array_equal(img.data, before) and "clipboard" not in session.settings and len(session.histories[img.id].undo) == 0
    session.run_op("edit.cut", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 8, "h": 8})
    assert img.data[0, 0, 0, :8, :8].max() == 0 and img.data[0, 0, 0, 8:, 8:].max() == before[0, 0, 0, 8:, 8:].max()
    assert session.macro[-1].startswith('lumen.run("edit.cut", roi=')  # the recorder keeps the selection, so a replay cannot wipe the plane


def test_options_partial_call_keeps_sibling_preferences():
    s = Session()
    assert all(p.default is None for p in edit_extra.PREFS.values()) and edit_extra.DEFAULT_SETTINGS == edit_extra.PREF_DEFAULTS
    assert set(edit_extra.PREF_DEFAULTS) == set(edit_extra.PREFS) and edit_extra.PREF_DEFAULTS["paste_mode"] == "Copy"
    s.run_op("edit.options_misc", {"paste_mode": "Max"})
    out = s.run_op("edit.options_misc", {"divide_by_zero": "NaN"})
    assert s.settings["paste_mode"] == "Max" and s.settings["divide_by_zero"] == "NaN" and out["message"] == "Misc: divide_by_zero='NaN'"
    s.settings["roi_color"] = "#ff0000"
    s.run_op("edit.options", {"font_size": 18})
    assert s.settings["roi_color"] == "#ff0000" and s.settings["font_size"] == 18
    assert s.settings["font_family"] == "sans-serif" and s.settings["jpeg_quality"] == 85  # keys never set get their default …
    s.settings["jpeg_quality"] = 42
    s.run_op("edit.options_io", {"default_save_format": "png"})
    assert s.settings["jpeg_quality"] == 42 and s.settings["default_save_format"] == "png"  # … an existing value is never reset
    assert s.run_op("edit.options_io", {})["message"] == "Input/Output: nothing changed"
    assert s.run_op("edit.options_roi_defaults", {"roi_color": ""})["message"].endswith("nothing changed")  # empty = keep
    assert s.settings["roi_color"] == "#ff0000"
    s.run_op("edit.options_reset", {})
    assert s.settings["jpeg_quality"] == 85 and s.settings["roi_color"] == "#ffd54a" and s.settings["paste_mode"] == "Copy"
    # the ImageJ-macro path pre-seeds every parameter from the current settings and keeps working with None defaults
    from lumen.core.ij_compat import IJ
    ij = IJ(s)
    ij.run("Paste Control...", "transfer=Max")
    ij.run("Misc...", "divide=NaN")
    assert s.settings["paste_mode"] == "Max" and s.settings["divide_by_zero"] == "NaN"


def test_edit_lut_never_overwrites_builtin(session):
    out = session.run_op("color.edit_lut", {"name": "Grays", "start": "red", "end": "blue"})
    assert "'Grays 2'" in out["message"] and session.active.display.channels[0].lut == "Grays 2"
    assert luts.get_lut("Grays")[255].tolist() == [255, 255, 255] and luts.get_lut("Grays 2")[255].tolist() == [0, 0, 255]
    assert luts.is_builtin("Grays") and not luts.is_builtin("Grays 2") and not luts.unregister_lut("Grays")
    out = session.run_op("color.edit_lut", {"name": " Fire ", "start": "red", "end": "blue", "apply": False})  # padded name
    assert "'Fire 2'" in out["message"] and luts.get_lut("Fire")[0].tolist() == [0, 0, 0]
    # a custom name is still updated in place
    session.run_op("color.edit_lut", {"name": "Mine", "start": "red", "end": "blue", "apply": False})
    session.run_op("color.edit_lut", {"name": "Mine", "start": "black", "end": "green", "apply": False})
    assert "Mine 2" not in luts.LUT_NAMES and luts.get_lut("Mine")[255].tolist() == [0, 255, 0]


def test_lut_version_changes_on_in_place_replacement(session):
    # the browser refetches /api/luts when state.lut_version changes; the name count alone misses a replacement
    v0 = session.state()["lut_version"]
    session.run_op("color.edit_lut", {"name": "Custom", "start": "black", "end": "red"})
    v1, n1 = session.state()["lut_version"], len(luts.LUT_NAMES)
    session.run_op("color.edit_lut", {"name": "Custom", "start": "black", "end": "blue"})
    v2 = session.state()["lut_version"]
    assert v0 < v1 < v2 and len(luts.LUT_NAMES) == n1 and luts.get_lut("Custom")[255].tolist() == [0, 0, 255]
    assert luts.unregister_lut("Custom") and session.state()["lut_version"] > v2
    # 8-bit Color re-run on the same image name after an undo: same LUT name, different palette
    s = Session()
    s.run_op("file.sample", {"name": "Astronaut (RGB)"})
    img = s.active
    s.run_op("type.8bit_color", {"colors": 256})
    va, na = s.state()["lut_version"], len(luts.LUT_NAMES)
    s.undo(img.id)
    s.run_op("type.8bit_color", {"colors": 4})
    assert len(luts.LUT_NAMES) == na and s.state()["lut_version"] > va


def test_paste_current_setting_honours_paste_control_and_tolerates_stale_values():
    s = Session()
    img = s.add_image(LumenImage(np.full((10, 10), 100, np.uint8), name="base.tif"))
    src = s.add_image(LumenImage(np.full((4, 4), 50, np.uint8), name="src.tif"))
    s.run_op("edit.copy", {}, image_id=src.id)
    at = {"type": "rect", "x": 3, "y": 3, "w": 4, "h": 4}
    s.run_op("edit.paste_control", {"mode": "Add"})
    assert s.run_op("edit.paste_with_mode", {}, image_id=img.id, roi=at)["message"] == "Pasted (Add)" and img.data[0, 0, 0, 4, 4] == 150
    s.undo(img.id)
    # the Preferences sheet PUTs the mode as a plain string: stale or ImageJ spellings must not break Paste
    for stored, expect in (("Transparent-white", "Transparent-white"), ("transparent", "Transparent-white"), ("Transparent zero", "Transparent-zero"),
                           ("max", "Max"), ("Bogus", "Copy"), ("", "Copy"), (None, "Copy")):
        s.settings["paste_mode"] = stored
        assert edit_extra.current_paste_mode(s) == expect, stored
        assert s.run_op("edit.paste_with_mode", {}, image_id=img.id, roi=at)["message"] == f"Pasted ({expect})"
        s.undo(img.id)
    assert s.run_op("edit.paste_control", {"mode": "Transparent-white"})["message"] == "Paste mode: Transparent-white"


def test_edit_paste_honours_paste_control():
    s = Session()
    img = s.add_image(LumenImage(np.full((10, 10), 100, np.uint8), name="base.tif"))
    src = s.add_image(LumenImage(np.full((4, 4), 50, np.uint8), name="src.tif"))
    s.run_op("edit.copy", {}, image_id=src.id)
    at = {"type": "rect", "x": 3, "y": 3, "w": 4, "h": 4}
    s.run_op("edit.paste_control", {"mode": "Add"})
    out = s.run_op("edit.paste", {}, image_id=img.id, roi=at)
    assert img.data[0, 0, 0, 4, 4] == 150 and img.data[0, 0, 0, 0, 0] == 100
    assert out["roi"]["type"] == "rect" and out["roi"]["x"] == 3  # the pasted rectangle stays selected
    s.undo(img.id)
    s.run_op("edit.paste_control", {"mode": "Copy"})
    s.run_op("edit.paste", {}, image_id=img.id, roi=at)
    assert img.data[0, 0, 0, 4, 4] == 50


def test_transparent_white_paste_mode():
    assert edit_extra.PASTE_MODES.index("Transparent-white") == edit_extra.PASTE_MODES.index("Transparent-zero") - 1  # ImageJ's dialog order
    s = Session()
    img = s.add_image(LumenImage(np.full((10, 10), 100, np.uint8), name="base.tif"))
    src = s.add_image(LumenImage(np.full((4, 4), 50, np.uint8), name="src.tif"))
    src.data[0, 0, 0, 0, 0] = 0
    src.data[0, 0, 0, 0, 1] = 255
    s.run_op("edit.copy", {}, image_id=src.id)
    s.run_op("edit.paste_with_mode", {"mode": "Transparent-white"}, image_id=img.id, roi={"type": "rect", "x": 3, "y": 3, "w": 4, "h": 4})
    p = img.data[0, 0, 0]
    assert p[3, 4] == 100 and p[3, 3] == 0 and p[5, 5] == 50  # 255 is transparent for 8-bit (ImageJ ByteBlitter); 0 is pasted
    # 16-bit and float: ShortBlitter / FloatBlitter treat COPY_TRANSPARENT as a plain copy (no 65535 sentinel)
    assert edit_extra.blend_paste(np.array([[1, 2]], np.uint16), np.array([[65535, 255]], np.uint16), "Transparent-white").tolist() == [[65535, 255]]
    assert edit_extra.blend_paste(np.array([[1.0, 2.0]], np.float32), np.array([[255.0, 0.0]], np.float32), "Transparent-white").tolist() == [[255.0, 0.0]]
    # RGB: only a pixel that is white (zero) in every channel is transparent – ColorBlitter tests the packed pixel, not each channel
    s = Session()
    rgb = s.add_image(LumenImage(np.full((6, 6, 3), 100, np.uint8), name="rgb.png", axes="YXS", rgb=True))
    patch = np.full((2, 2, 3), 255, np.uint8)
    patch[0, 0] = (255, 0, 0)
    patch[1, 1] = (0, 0, 0)
    src = s.add_image(LumenImage(patch, name="patch.png", axes="YXS", rgb=True))
    s.run_op("edit.copy", {}, image_id=src.id)
    at = {"type": "rect", "x": 2, "y": 2, "w": 2, "h": 2}
    s.run_op("edit.paste_with_mode", {"mode": "Transparent-white"}, image_id=rgb.id, roi=at)
    p = rgb.data[0, 0]
    assert p[:, 2, 2].tolist() == [255, 0, 0] and p[:, 2, 3].tolist() == [100, 100, 100] and p[:, 3, 3].tolist() == [0, 0, 0]
    s.undo(rgb.id)
    s.run_op("edit.paste_with_mode", {"mode": "Transparent-zero"}, image_id=rgb.id, roi=at)
    p = rgb.data[0, 0]
    assert p[:, 2, 2].tolist() == [255, 0, 0] and p[:, 2, 3].tolist() == [255, 255, 255] and p[:, 3, 3].tolist() == [100, 100, 100]


def test_selection_properties_fill_is_stored_for_renderers(session):
    img = session.active
    rect = {"type": "rect", "x": 10, "y": 10, "w": 20, "h": 20}
    out = session.run_op("selection.properties", {"fill": "#00ff00"}, roi=rect)
    r = out["roi"]
    assert r["fill"] == "#00ff00" and r["fill_color"] == "#00ff00" and "fill #00ff00" in out["message"]  # 'fill_color' is the .roi writer's key
    session.run_op("overlay.add", {}, roi=r)
    assert img.overlay[-1]["fill"] == "#00ff00"  # the overlay item carries the fill for the viewport / Flatten to honour
    out = session.run_op("selection.properties", {"fill": "none"}, roi=r)
    assert "fill" not in out["roi"] and "fill_color" not in out["roi"] and "fill none" in out["message"]
    # a fill colour read from an ImageJ .roi file ('fill_color') becomes visible under the canonical key, and 'none' clears it too
    assert session.run_op("selection.properties", {"name": "imported"}, roi=dict(rect, fill_color="#0000ff"))["roi"]["fill"] == "#0000ff"
    assert "fill_color" not in session.run_op("selection.properties", {"fill": "none"}, roi=dict(rect, fill_color="#0000ff"))["roi"]


# ---------------------------------------------------------------- Fit Circle to Image / Points from Mask
def _disk(h, w, cx, cy, r, dtype=np.uint8, fg=255, bg=0):
    yy, xx = np.mgrid[0:h, 0:w]
    m = (xx + 0.5 - cx) ** 2 + (yy + 0.5 - cy) ** 2 <= r * r
    return np.where(m, fg, bg).astype(dtype)


def test_fit_circle_to_image_binary_threshold_and_selection():
    s = Session()
    s.add_image(LumenImage(_disk(50, 60, 30.5, 25.5, 15), name="disk.tif"))
    for method in edit_extra.CIRCLE_FIT_METHODS:
        out = s.run_op("selection.fit_circle_to_image", {"method": method})
        r = out["roi"]
        assert r["type"] == "oval" and r["name"] == "Fitted circle", method
        assert r["x"] + r["w"] / 2 == pytest.approx(30.5, abs=0.3) and r["y"] + r["h"] / 2 == pytest.approx(25.5, abs=0.3), method
        assert "non-zero" in out["message"], method
        if method.startswith("All pixels"):
            assert r["w"] / 2 == pytest.approx(15 / math.sqrt(2), abs=0.6)  # the algebraic fit of a *filled* disk: r² = mean d² = R²/2 (use it for rings)
        else:
            assert r["w"] / 2 == pytest.approx(15, abs=0.6), method
    assert s.macro[-1].startswith('lumen.run("selection.fit_circle_to_image"')
    ring = _disk(50, 60, 30.5, 25.5, 15) & ~_disk(50, 60, 30.5, 25.5, 12)
    s.add_image(LumenImage(ring, name="ring.tif"))
    r = s.run_op("selection.fit_circle_to_image", {"method": "All pixels (least squares)"})["roi"]
    assert r["x"] + r["w"] / 2 == pytest.approx(30.5, abs=0.3) and 12 <= r["w"] / 2 <= 15
    # a thresholded 16-bit image: the foreground is the threshold range, not the non-zero pixels
    img16 = s.add_image(LumenImage(_disk(50, 60, 20.5, 20.5, 10, np.uint16, 500, 10), name="disk16.tif"))
    out = s.run_op("selection.fit_circle_to_image", {"method": "Area and centroid"})  # no threshold: every non-zero pixel is foreground → the whole plane
    assert "non-zero" in out["message"] and out["roi"]["x"] + out["roi"]["w"] / 2 == pytest.approx(30, abs=0.01) and out["roi"]["y"] + out["roi"]["h"] / 2 == pytest.approx(25, abs=0.01)
    s.thresholds[img16.id] = {"lower": 100, "upper": 65535}
    out = s.run_op("selection.fit_circle_to_image", {})
    r = out["roi"]
    assert r["x"] + r["w"] / 2 == pytest.approx(20.5, abs=0.3) and r["w"] / 2 == pytest.approx(10, abs=0.6) and "threshold 100" in out["message"]
    # two blobs: an area selection restricts the fit to the blob inside it
    two = _disk(60, 100, 25.5, 30.5, 12) | _disk(60, 100, 75.5, 30.5, 8)
    s.add_image(LumenImage(two, name="two.tif"))
    out = s.run_op("selection.fit_circle_to_image", {}, roi={"type": "rect", "x": 55, "y": 0, "w": 45, "h": 60})
    r = out["roi"]
    assert r["x"] + r["w"] / 2 == pytest.approx(75.5, abs=0.3) and r["w"] / 2 == pytest.approx(8, abs=0.6) and "inside the selection" in out["message"]
    s.add_image(LumenImage(np.zeros((10, 10), np.uint8), name="empty.tif"))
    with pytest.raises(ValueError, match="No foreground"):
        s.run_op("selection.fit_circle_to_image", {})


def test_fit_circle_to_image_rgb_and_calibrated_message():
    s = Session()
    rgb = np.zeros((40, 40, 3), np.uint8)
    rgb[..., 2] = _disk(40, 40, 20.5, 20.5, 9)  # blue disk: non-zero in any channel counts
    img = s.add_image(LumenImage(rgb, name="rgb.png", axes="YXS", rgb=True))
    img.calibration.pixel_width = img.calibration.pixel_height = 0.5
    img.calibration.unit = "µm"
    out = s.run_op("selection.fit_circle_to_image", {})
    assert out["roi"]["w"] / 2 == pytest.approx(9, abs=0.6) and "any channel" in out["message"] and "µm" in out["message"]


def test_points_from_mask_subsampling_thinning_and_manager():
    s = Session()
    a = np.zeros((10, 12), np.uint8)
    a[2:5, 3:6] = 255
    img = s.add_image(LumenImage(a, name="sq.tif"))
    out = s.run_op("selection.points_from_mask", {})
    pts = out["roi"]["points"]
    assert out["roi"]["type"] == "point" and len(pts) == 9 and [3.5, 2.5] in pts and [5.5, 4.5] in pts and "9 points" in out["message"]
    assert roimod.to_mask(out["roi"], a.shape).sum() == 9 and np.array_equal(roimod.to_mask(out["roi"], a.shape), a > 0)  # pixel centres map back to the mask
    out = s.run_op("selection.points_from_mask", {"step": 2})
    assert sorted(out["roi"]["points"]) == [[4.5, 2.5], [4.5, 4.5]] and "every 2. pixel" in out["message"]
    out = s.run_op("selection.points_from_mask", {"max_points": 4})
    assert len(out["roi"]["points"]) == 4 and "thinned from 9" in out["message"]
    out = s.run_op("selection.points_from_mask", {}, roi={"type": "rect", "x": 3, "y": 2, "w": 1, "h": 3})
    assert len(out["roi"]["points"]) == 3 and "inside the selection" in out["message"]
    out = s.run_op("selection.points_from_mask", {"output": "ROI Manager"})
    assert out["rois_added"] == 1 and s.roi_manager.rois[-1]["type"] == "point" and len(s.roi_manager.rois[-1]["points"]) == 9 and "roi" not in out
    s.thresholds[img.id] = {"lower": 0, "upper": 0}  # thresholded: the background becomes the foreground
    assert len(s.run_op("selection.points_from_mask", {})["roi"]["points"]) == 120 - 9
    s.add_image(LumenImage(np.zeros((4, 4), np.uint8), name="empty.tif"))
    with pytest.raises(ValueError, match="No foreground"):
        s.run_op("selection.points_from_mask", {})


# ---------------------------------------------------------------- Image ▸ Drawing ▸ gradients
def test_linear_gradient_angles_types_and_undo():
    s = Session()
    img = s.add_image(LumenImage(np.full((10, 20), 7, np.uint8), name="g8.tif"))
    s.run_op("edit.linear_gradient", {})
    p = img.data[0, 0, 0]
    assert p[:, 0].min() == p[:, 0].max() and p[0, 0] >= 245 and p[0, -1] <= 10 and np.all(np.diff(p[0].astype(int)) <= 0)  # fg left → bg right
    s.run_op("edit.linear_gradient", {"reverse": True})
    assert img.data[0, 0, 0, 0, 0] <= 10 and img.data[0, 0, 0, 0, -1] >= 245
    s.run_op("edit.linear_gradient", {"angle": 90})
    p = img.data[0, 0, 0]
    assert p[0].min() == p[0].max() and p[-1, 0] >= 240 and p[0, 0] <= 15  # 90° = foreground at the bottom
    s.run_op("edit.linear_gradient", {"angle": 180})
    assert img.data[0, 0, 0, 0, -1] >= 245 and img.data[0, 0, 0, 0, 0] <= 10
    assert s.undo(img.id) is not None and img.data[0, 0, 0, 0, 0] <= 15  # back to the 90° gradient
    s.run_op("edit.linear_gradient", {"fg_value": 100, "bg_value": 50})
    assert img.data[0, 0, 0, 0, 0] == 100 and img.data[0, 0, 0, 0, -1] == 50
    img16 = s.add_image(LumenImage(np.zeros((4, 8), np.uint16), name="g16.tif"))
    s.run_op("edit.linear_gradient", {})
    assert img16.data[0, 0, 0, 0, 0] == 65535 and img16.data[0, 0, 0, 0, -1] == 0  # the 255 foreground setting means full scale on 16-bit
    img32 = s.add_image(LumenImage(np.zeros((4, 8), np.float32), name="g32.tif"))
    s.settings.update(foreground=1.0, background=-1.0)
    s.run_op("edit.linear_gradient", {})
    assert img32.data[0, 0, 0, 0, 0] == pytest.approx(1.0, abs=0.07) and img32.data[0, 0, 0, 0, -1] == pytest.approx(-1.0, abs=0.07)


def test_linear_gradient_selection_line_rgb_and_stack():
    s = Session()
    img = s.add_image(LumenImage(np.full((20, 20), 7, np.uint8), name="sel.tif"))
    s.run_op("edit.linear_gradient", {}, roi={"type": "rect", "x": 5, "y": 5, "w": 10, "h": 10})
    p = img.data[0, 0, 0]
    assert p[0, 0] == 7 and p[19, 19] == 7 and p[10, 5] >= 240 and p[10, 14] <= 30 and np.all(np.diff(p[10, 5:15].astype(int)) <= 0)  # spans the box
    s.undo(img.id)
    s.run_op("edit.linear_gradient", {}, roi={"type": "line", "x1": 5, "y1": 10, "x2": 15, "y2": 10})
    p = img.data[0, 0, 0]
    assert p[10, 0] == 255 and p[10, 5] >= 240 and p[10, 15] <= 15 and p[10, 19] == 0 and p[0, 10] == p[19, 10]  # along the line, whole plane
    rgb = s.add_image(LumenImage(np.zeros((6, 10, 3), np.uint8), name="rgb.png", axes="YXS", rgb=True))
    s.run_op("edit.linear_gradient", {"fg_color": "#ff0000", "bg_color": "#0000ff"})
    assert rgb.data[0, 0, :, 0, 0].tolist() == [255, 0, 0] and rgb.data[0, 0, :, 0, -1].tolist() == [0, 0, 255]
    stack = s.add_image(LumenImage(np.zeros((3, 4, 6), np.uint8), name="stack.tif", axes="ZYX"))
    stack.display.z = 1
    s.run_op("edit.linear_gradient", {})
    assert stack.data[0, 1, 0, 0, 0] == 255 and stack.data[0, 0, 0, 0, 0] == 0  # current plane only by default
    s.run_op("edit.linear_gradient", {"apply_to": "Whole stack"})
    assert stack.data[0, 0, 0, 0, 0] == 255 and stack.data[0, 2, 0, 0, 0] == 255


def test_radial_gradient_centre_radius_and_oval_selection():
    s = Session()
    img = s.add_image(LumenImage(np.zeros((21, 21), np.uint8), name="r.tif"))
    s.run_op("edit.radial_gradient", {})
    p = img.data[0, 0, 0].astype(int)
    assert p[10, 10] == 255 and p[0, 0] == 0 and p[10, 0] <= 15 and p[10, 0] == p[0, 10] == p[20, 10]  # centre (10.5, 10.5), radius 10.5
    assert p[10, 10] > p[10, 7] > p[10, 4] > p[10, 1]
    s.run_op("edit.radial_gradient", {"center_x": 0, "center_y": 0, "radius": 10, "reverse": True})
    p = img.data[0, 0, 0].astype(int)
    assert p[0, 0] <= 20 and p[0, 15] == 255 and p[20, 20] == 255
    big = s.add_image(LumenImage(np.full((40, 60), 9, np.uint8), name="big.tif"))
    s.run_op("edit.radial_gradient", {}, roi={"type": "oval", "x": 10, "y": 5, "w": 30, "h": 30})
    p = big.data[0, 0, 0].astype(int)
    assert p[20, 25] >= 240 and p[0, 0] == 9 and p[39, 59] == 9 and p[20, 11] < 30 and p[20, 11] > 0  # foreground at the centre (25, 20: a pixel corner), background at the rim
    assert s.macro[-1].startswith('lumen.run("edit.radial_gradient"') and "roi=" in s.macro[-1]


def test_gradient_specs():
    for oid in ("edit.linear_gradient", "edit.radial_gradient"):
        d = REGISTRY.ops[oid].to_dict()
        assert d["menu"] == "Image/Drawing" and d["kind"] == "image" and d["preview"] is True and d["supports_rgb"] is True


# ---------------------------------------------------------------- Overlay Options…
def test_options_overlay_writes_only_passed_keys_and_normalises_fill():
    s = Session()
    out = s.run_op("edit.options_overlay", {"overlay_show_labels": True, "overlay_label_font_size": 18, "overlay_default_fill": "red"})
    st = s.settings
    assert st["overlay_show_labels"] is True and st["overlay_label_font_size"] == 18 and st["overlay_default_fill"] == "#ff0000"
    assert st["overlay_use_names"] is False and st["overlay_label_color"] == "#ffff00" and st["overlay_stroke_width"] == 1.5 and st["overlay_label_background"] is False  # seeded defaults
    assert "overlay_show_labels=True" in out["message"]
    s.run_op("edit.options_overlay", {"overlay_use_names": True})
    assert st["overlay_show_labels"] is True and st["overlay_label_font_size"] == 18 and st["overlay_use_names"] is True  # siblings untouched
    s.run_op("edit.options_overlay", {"overlay_default_fill": "none", "overlay_label_color": "0,0,255", "overlay_stroke_width": 3})
    assert st["overlay_default_fill"] == "" and st["overlay_label_color"] == "#0000ff" and st["overlay_stroke_width"] == 3.0
    assert s.run_op("edit.options_overlay", {})["message"] == "Overlay Options: nothing changed"
    for k in ("overlay_stroke_width", "overlay_default_fill", "overlay_show_labels", "overlay_use_names", "overlay_label_font_size", "overlay_label_color", "overlay_label_background"):
        assert k in edit_extra.PREFS and k in edit_extra.DEFAULT_SETTINGS
    from lumen.server.extra.settings import DEFAULTS, PREFERENCES
    for k in edit_extra._OVERLAY_KEYS:
        assert k in PREFERENCES and DEFAULTS[k] == edit_extra.DEFAULT_SETTINGS[k], k
    s.run_op("edit.options_reset")
    assert st["overlay_label_font_size"] == 12 and st["overlay_default_fill"] == ""
