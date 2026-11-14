"""Tests for the ImageJ-1 compatibility layer (option parser, IJ.run mappings, macro transpiler, scripting hook)."""
import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ij_compat import COMMANDS, IJ, Options, command_table, parse_options, transpile
from lumen.core.scripting import run_script
from lumen.core.session import Session


@pytest.fixture
def ij():
    s = Session()
    ij = IJ(s)
    ij.run("Blobs (25K)")
    return ij


def _stack(session, nz=5):
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 255, size=(nz, 32, 32), dtype=np.uint8)
    return session.add_image(LumenImage(vol, name="stack.tif", axes="ZYX"))


# ---------------------------------------------------------------- option strings
def test_parse_options_basic():
    o = parse_options("sigma=2 stack")
    assert o == {"sigma": "2", "stack": True}  # values stay raw text (ImageJ Macro.getValue); num() parses on demand
    assert o.num("sigma") == 2 and isinstance(o.num("sigma"), int)
    assert parse_options("radius=2.5").num("radius") == 2.5
    assert parse_options("") == {}


def test_parse_options_brackets_quotes_and_ranges():
    o = parse_options('title=[my image] size=20-Infinity text1="a b" code=[v=v*2] display clear circularity=0.50-1.00')
    assert o["title"] == "my image" and o["text1"] == "a b" and o["code"] == "v=v*2"
    assert o["display"] is True and o["clear"] is True
    assert o.rng("size") == (20.0, float("inf"))
    assert o.rng("circularity") == (0.5, 1.0)
    assert o.flag("stack") is False and o.num("missing", 7) == 7 and o.str("title") == "my image"


def test_parse_options_nested_brackets_and_paths():
    o = parse_options("open=[/tmp/some dir/file.tif] kernel=[[1 2 1] [2 4 2]] x=-3 y=1e3")
    assert o["open"] == "/tmp/some dir/file.tif" and o["kernel"] == "[1 2 1] [2 4 2]"
    assert o["x"] == "-3" and o["y"] == "1e3"
    assert o.num("x") == -3 and o.num("y") == 1000.0


# ---------------------------------------------------------------- IJ.run mappings
def test_run_filter_changes_pixels(ij):
    before = ij.img.current_plane().copy()
    ij.run("Gaussian Blur...", "sigma=2")
    assert not np.array_equal(before, ij.img.current_plane())
    ij.run("Invert")
    ij.run("Median...", "radius=1")
    ij.run("Add...", "value=10")
    assert ij.img.current_plane().shape == before.shape


def test_run_duplicate_type_and_stack_commands(ij):
    ij.run("Duplicate...", "title=copy")
    assert ij.nImages() == 2 and ij.getTitle() == "copy"
    ij.run("16-bit")
    assert ij.img.data.dtype == np.uint16
    ij.run("8-bit")
    assert ij.img.data.dtype == np.uint8
    st = _stack(ij.session)
    ij.selectWindow("stack.tif")
    assert ij.nSlices() == 5
    ij.run("Z Project...", "projection=[Max Intensity]")
    assert ij.img.nz == 1 and ij.img.id != st.id
    ij.selectWindow("stack.tif")
    ij.run("Make Substack...", "slices=1-3")
    assert ij.img.nz == 3


def test_run_threshold_and_particles(ij):
    ij.run("Gaussian Blur...", "sigma=2")
    ij.setAutoThreshold("Otsu dark")
    ij.run("Convert to Mask")
    assert set(np.unique(ij.img.current_plane())).issubset({0, 255})
    ij.run("Watershed")
    ij.run("Set Measurements...", "area mean centroid redirect=None decimal=3")
    ij.run("Analyze Particles...", "size=20-Infinity display clear")
    assert ij.nResults() > 10
    assert ij.getResult("Area", 0) > 0


def test_run_unknown_and_unsupported_commands(ij):
    with pytest.raises(ValueError):
        ij.run("No Such Command...")
    with pytest.raises(ValueError, match="no file given"):
        ij.run("Bio-Formats Importer")  # mapped now (file.bioformats_import): it still needs open=[path]
    assert COMMANDS["Bio-Formats Importer"][1] == "file.bioformats_import"
    assert COMMANDS["Montage to Stack..."][1] == "stacks.montage_to_stack"


def test_run_fit_spline(ij):
    ij.roi = {"type": "oval", "x": 20, "y": 20, "w": 60, "h": 40}
    ij.run("Fit Spline")
    assert ij.roi["type"] in ("polygon", "freehand") and len(ij.roi["points"]) >= 4   # ImageJ: an oval interpolated at 1 px is a FREEROI
    assert COMMANDS["Fit Spline"][1] == "selection.interpolate"


def test_run_color_threshold_on_rgb():
    s = Session()
    ij = IJ(s)
    s.run_op("file.sample", {"name": "Astronaut (RGB)"})
    n = len(s.images)
    ij.run("Color Threshold...", "space=HSB hue_min=0 hue_max=255 saturation_min=0 saturation_max=255 brightness_min=128 brightness_max=255 output=Mask")
    assert len(s.images) == n + 1
    mask = list(s.images.values())[-1]
    assert mask.name.startswith("Mask") and set(np.unique(mask.current_plane())).issubset({0, 255})


def test_run_image_sequence(tmp_path):
    from PIL import Image
    for i in range(3):
        Image.fromarray(np.full((16, 20), 10 * i, np.uint8)).save(tmp_path / f"img_{i:02d}.png")
    s = Session()
    ij = IJ(s)
    ij.run("Image Sequence...", f"open=[{tmp_path}] sort")
    assert ij.nSlices() == 3 and ij.img.width == 20
    ij.run("Image Sequence...", f"dir=[{tmp_path}] file=_01 sort")
    assert ij.nSlices() == 1 and int(ij.img.current_plane()[0, 0]) == 10
    with pytest.raises(ValueError):
        ij.run("Image Sequence...", "")


def test_command_table_lists_mappings():
    rows = command_table()
    cmds = " ".join(r["command"] for r in rows)
    assert '"Gaussian Blur..."' in cmds and '"Analyze Particles..."' in cmds and '"Color Threshold..."' in cmds
    assert len(rows) > 100


# ---------------------------------------------------------------- macros
TARGET_MACRO = '''
run("Blobs (25K)");
run("Gaussian Blur...", "sigma=2");
setAutoThreshold("Otsu dark");
run("Convert to Mask");
run("Watershed");
run("Set Measurements...", "area mean centroid redirect=None decimal=3");
run("Analyze Particles...", "size=20-Infinity display clear");
n = nResults;
print("Particles: " + n);
for (i = 0; i < n; i++) {
    print(i + ": " + getResult("Area", i));
}
'''


def test_target_macro_runs_end_to_end():
    s = Session()
    ij = IJ(s)
    out = ij.macro(TARGET_MACRO)
    assert ij.nResults() > 10
    lines = out.strip().splitlines()
    assert lines[0].startswith("Particles: ") and int(lines[0].split()[1]) == ij.nResults()
    assert len(lines) == ij.nResults() + 1
    assert s.results.rows[0]["Area"] > 0


def test_macro_language_features(ij):
    macro = '''
var total = 0;
function area(w, h) { total += w * h; return w * h; }
getDimensions(w, h, c, z, t);
a = newArray(1, 2, 3);
a[1] = 10;
sum = 0;
for (i = 0; i < a.length; i++) sum += a[i];
i = 0; while (i < 3) { i++; } do { i--; } while (i > 1);
s = "abc";
print("sum=" + sum + " area=" + area(w, h) + " i=" + i + " " + (i == 1 ? "one" : "not"));
print(toUpperCase(s) + " " + lengthOf(s) + " " + substring(s, 1) + " " + d2s(3.14159, 2));
if (indexOf(getTitle(), "blobs") >= 0 && !endsWith(getTitle(), ".xyz")) print("blobs"); else print("other");
setAutoThreshold("Otsu dark");
run("Convert to Mask");
print("binary=" + is("binary"));
'''
    out = ij.macro(macro)
    assert "sum=14 area=65536 i=1 one" in out
    assert "ABC 3 bc 3.14" in out
    assert "blobs" in out and "binary=1" in out


def test_macro_unsupported_constructs_raise(ij):
    with pytest.raises(ValueError, match="Unsupported macro construct"):
        ij.macro("x = 1 @ 2;")
    with pytest.raises(ValueError, match="Unsupported macro"):
        ij.macro("noSuchFunction(1);")
    with pytest.raises(ValueError):
        transpile("if (x { }")


def test_macro_roi_manager_and_results(ij):
    out = ij.macro('''
makeRectangle(10, 10, 40, 30);
roiManager("Add");
makeOval(60, 60, 30, 30);
roiManager("Add");
print(roiManager("count"));
roiManager("Select", 0);
run("Measure");
print(nResults + " " + getResult("Area", 0));
''')
    assert out.splitlines()[0] == "2"
    assert out.splitlines()[1].startswith("1 ") and ij.nResults() == 1
    assert ij.session.results.rows[0]["Area"] == pytest.approx(1200)


def test_macro_exit_and_transpile(ij):
    py = transpile('print("a"); exit("stop"); print("b");')
    assert "_MacroExit" in py or "exit" in py
    out = ij.macro('print("a"); exit("stop"); print("b");')
    assert "a" in out and "stop" in out and "b" not in out.replace("stop", "")


# ---------------------------------------------------------------- Lumen ops & scripting hook
def test_run_ij_macro_op_and_command_list():
    s = Session()
    out = s.run_op("script.run_ij_macro", {"macro": TARGET_MACRO})
    assert "Particles:" in out["log"]
    assert len(s.results.rows) > 10
    out = s.run_op("script.ij_command_list", {})
    assert out["table"]["columns"] == ["IJ command", "Lumen command", "Notes"] and len(out["table"]["rows"]) > 100


def test_scripting_hook_exposes_ij():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    before = s.active.current_plane().copy()
    res = run_script(s, '''
print(IJ is lumen.ij)
IJ.run("Gaussian Blur...", "sigma=2")
run("Invert")
makeRectangle(5, 5, 20, 20)
run("Measure")
print(nResults(), getResult("Area", 0))
''')
    assert res["error"] is None, res["error"]
    lines = res["stdout"].strip().splitlines()
    assert lines[0] == "True" and lines[1].startswith("1 ")
    assert not np.array_equal(before, s.active.current_plane())
    # builtins are never shadowed by macro helpers
    res = run_script(s, "print(open, round(2.5), abs(-1))")
    assert res["error"] is None and "built-in" in res["stdout"]


# ---------------------------------------------------------------- regression tests (review findings)
def _tzc(session, shape, name="hyper", seed=0, dtype=np.uint8):
    rng = np.random.default_rng(seed)
    vol = rng.integers(0, 255 if dtype == np.uint8 else 4000, size=shape, dtype=dtype)
    return session.add_image(LumenImage(vol, name=name, axes="TZCYX")), vol


def test_options_keep_numeric_looking_text(ij):
    o = parse_options("title=007 label=[0012] file=0.50 t=1e3")
    assert o.str("title") == "007" and o.str("label") == "0012" and o.str("file") == "0.50" and o.str("t") == "1e3"
    assert o.num("t") == 1000.0 and o.num("file") == 0.5
    ij.run("Rename...", "title=007")
    assert ij.getTitle() == "007"
    ij.run("image.rename", "title=0012")  # generic op fallback hands the string Param a string
    assert ij.getTitle() == "0012"
    ij.run("Scale...", "x=- y=- width=64 height=48 interpolation=Bilinear average create title=small")
    assert (ij.img.width, ij.img.height, ij.getTitle()) == (64, 48, "small")


def test_gaussian_blur_legacy_radius(ij):
    ij.run("Gaussian Blur...", "radius=10")
    a = ij.img.current_plane().copy()
    ij.run("Blobs (25K)")
    ij.run("Gaussian Blur...", "sigma=4")
    assert np.array_equal(a, ij.img.current_plane())  # ImageJ: sigma = radius / 2.5


def test_run_reports_the_real_command_and_error(ij):
    with pytest.raises(ValueError, match="Fit Circle"):
        ij.run("Fit Circle")  # needs a selection: the op's own error names the command
    with pytest.raises(ValueError, match="TIFF Virtual Stack"):
        ij.run("TIFF Virtual Stack...")  # no open=[path]: the error names the command
    with pytest.raises(FileNotFoundError, match="nowhere"):
        ij.run("Bio-Formats Importer", "open=[/nowhere.nd2]")


def test_image_calculator_replaces_image1_not_the_active_image(ij):
    ij.run("Rename...", "title=A")
    a0 = ij.img.data.copy()
    ij.run("Blobs (25K)")
    ij.run("Rename...", "title=B")
    ij.run("Divide...", "value=2")
    b0 = ij.img.data.copy()
    ij.run("Image Calculator...", "image1=A operation=Subtract image2=B")
    A, B = ij._image("A"), ij._image("B")
    assert np.array_equal(A.data, a0.astype(np.int32) - b0.astype(np.int32)) and np.array_equal(B.data, b0)
    assert ij.img is A and ij.session.histories[A.id].can_undo()
    ij.session.undo(A.id)
    assert np.array_equal(A.data, a0)
    ij.run("Image Calculator...", "image1=A operation=Subtract image2=B create 32-bit")
    assert ij.img.name.startswith("Result of A") and ij.img.data.dtype == np.float32


def test_duplicate_range_on_plain_stacks(ij):
    vol = np.arange(10, dtype=np.uint8)[:, None, None] * np.ones((1, 8, 8), np.uint8)
    ij.session.add_image(LumenImage(vol, name="st", axes="ZYX"))
    ij.run("Duplicate...", "title=sub duplicate range=2-4")
    assert ij.getTitle() == "sub" and ij.img.nz == 3 and list(ij.img.data[0, :, 0, 0, 0]) == [1, 2, 3]
    ij.selectWindow("st")
    ij.run("Duplicate...", "title=tail duplicate range=4")
    assert ij.img.nz == 7  # a single number means first..end
    ij.selectWindow("st")
    ij.run("Duplicate...", "title=all duplicate range=5-2")
    assert ij.img.nz == 10  # first > last → whole stack
    ij.selectWindow("st")
    ij.run("Duplicate...", "title=one range=2-4")
    assert ij.img.nz == 1  # without 'duplicate' only the current plane is copied
    ij.session.add_image(LumenImage(np.zeros((6, 1, 1, 8, 8), np.uint8), name="tl", axes="TZCYX"))
    ij.run("Duplicate...", "title=frames duplicate range=2-4")
    assert ij.img.data.shape == (3, 1, 1, 8, 8)  # time-lapse: the "slices" are the frames


def test_duplicate_composite_hyperstack_keeps_all_channels():
    s = Session()
    ij = IJ(s)
    img, _ = _tzc(s, (1, 5, 2, 8, 8), name="comp")
    img.display.mode = "composite"
    ij.setSlice(2)
    ij.run("Duplicate...", "title=dup")
    assert ij.img.data.shape == (1, 1, 2, 8, 8) and ij.img.display.mode == "composite"
    img.display.mode = "single"
    ij.selectWindow("comp")
    ij.run("Duplicate...", "title=dup2")
    assert ij.img.data.shape == (1, 1, 1, 8, 8)
    flat, _ = _tzc(s, (1, 1, 3, 8, 8), name="flat")  # "single composite" (nz == nt == 1): only the current channel
    flat.display.mode = "composite"
    ij.run("Duplicate...", "title=dup3")
    assert ij.img.data.shape == (1, 1, 1, 8, 8)


def test_make_substack_applies_all_ranges():
    s = Session()
    ij = IJ(s)
    src, vol = _tzc(s, (4, 6, 3, 8, 8))
    ij.run("Make Substack...", "channels=1-2 slices=2-4 frames=1-2")
    assert ij.img.data.shape == (2, 3, 2, 8, 8) and np.array_equal(ij.img.data, vol[0:2, 1:4, 0:2])
    assert len(s.images) == 2  # no intermediate images
    ij.selectWindow("hyper")
    ij.run("Make Substack...", "slices=2-4 delete")
    assert ij.img.nz == 3 and src.nz == 3  # single range: delete still shrinks the original
    ij.selectWindow("hyper")
    ij.run("Make Substack...", "channels=1 slices=1-2 delete")
    assert (src.nz, src.nc) == (3, 3)  # multi-axis selection never deletes (ImageJ's hyperstack path has no delete)
    s.add_image(LumenImage(np.zeros((12, 1, 1, 8, 8), np.uint8), name="tl", axes="TZCYX"))
    ij.run("Make Substack...", "slices=1-3")
    assert ij.img.data.shape == (3, 1, 1, 8, 8)


def test_auto_threshold_without_stack_flag_only_touches_the_current_slice():
    s = Session()
    ij = IJ(s)
    vol = np.random.default_rng(0).integers(0, 255, size=(10, 16, 16), dtype=np.uint8)
    img = s.add_image(LumenImage(vol, name="u8", axes="ZYX"))
    ij.Stack.setSlice(6)
    ij.run("Auto Threshold", "method=Otsu white")
    assert [z for z in range(10) if not np.array_equal(img.data[0, z, 0], vol[z])] == [5]
    assert set(np.unique(img.data[0, 5, 0])) <= {0, 255}
    ij.run("Auto Threshold", "method=Otsu white stack")
    assert all(set(np.unique(img.data[0, z, 0])) <= {0, 255} for z in range(10))
    vol16 = np.random.default_rng(1).integers(0, 4000, size=(5, 16, 16), dtype=np.uint16)
    im16 = s.add_image(LumenImage(vol16, name="u16", axes="ZYX"))
    ij.Stack.setSlice(3)
    ij.run("Auto Threshold", "method=Otsu white")
    assert im16.data.dtype == np.uint16 and set(np.unique(im16.data[0, 2, 0])) <= {0, 65535}
    assert [z for z in range(5) if not np.array_equal(im16.data[0, z, 0], vol16[z])] == [2]
    s.undo(im16.id)
    assert np.array_equal(im16.data[0, :, 0], vol16)


def test_z_project_time_lapse_and_all_flag():
    s = Session()
    ij = IJ(s)
    tl = np.random.default_rng(0).integers(0, 255, size=(12, 16, 16), dtype=np.uint8)
    s.add_image(LumenImage(tl, name="tl", axes="TYX"))
    ij.run("Z Project...", "projection=[Max Intensity]")
    assert ij.img.data.shape == (1, 1, 1, 16, 16) and np.array_equal(ij.img.data[0, 0, 0], tl.max(axis=0))
    src, vol = _tzc(s, (4, 5, 2, 8, 8))
    ij.Stack.setFrame(3)
    ij.run("Z Project...", "projection=[Max Intensity]")
    assert ij.img.data.shape == (1, 1, 2, 8, 8) and np.array_equal(ij.img.data[0, 0], vol[2].max(axis=0))
    ij.selectWindow("hyper")
    ij.run("Z Project...", "projection=[Max Intensity] all")
    assert ij.img.data.shape == (4, 1, 2, 8, 8)


def test_add_slice_channel():
    s = Session()
    ij = IJ(s)
    img, vol = _tzc(s, (1, 4, 3, 8, 8))
    names = [c.name for c in img.display.channels]
    img.display.c = 1
    ij.run("Add Slice", "add=channel")
    assert img.data.shape == (1, 4, 4, 8, 8) and len(img.display.channels) == 4
    assert not img.data[:, :, 2].any() and np.array_equal(img.data[:, :, [0, 1, 3]], vol)
    assert [c.name for c in img.display.channels][:2] == names[:2] and img.display.channels[3].name == names[2]
    s.undo(img.id)
    assert img.data.shape == (1, 4, 3, 8, 8)
    ij.run("Add Slice", "add=slice")
    ij.run("Add Slice", "add=frame")
    ij.run("Add Slice")
    assert img.data.shape == (2, 6, 3, 8, 8)


def test_enlarge_specify_band_calibrated_units(ij):
    ij.makeRectangle(50, 50, 20, 20)
    ij.run("Enlarge...", "enlarge=5")
    assert ij.getSelectionBounds() == (45, 45, 30, 30)  # uncalibrated: pixels
    ij.run("Properties...", "unit=um pixel_width=0.5 pixel_height=0.25 voxel_depth=1")
    ij.makeRectangle(50, 50, 20, 20)
    ij.run("Enlarge...", "enlarge=5")
    assert ij.getSelectionBounds() == (40, 40, 40, 40)  # 5 µm = 10 px
    ij.makeRectangle(50, 50, 20, 20)
    ij.run("Enlarge...", "enlarge=5 pixel")
    assert ij.getSelectionBounds() == (45, 45, 30, 30)
    ij.run("Specify...", "width=10 height=10 x=1 y=1 scaled")
    assert ij.getSelectionBounds() == (2, 4, 20, 40)
    ij.run("Specify...", "width=10 height=10 x=1 y=1")
    assert ij.getSelectionBounds() == (1, 1, 10, 10)
    ij.makeRectangle(50, 50, 20, 20)
    ij.run("Make Band...", "band=5")
    assert ij.roi is not None and ij.getSelectionBounds()[2] == 40  # 5 µm band = 10 px on each side


def test_label_numbers_z_slices_and_time_stamper_never_uses_a_frame_index():
    from lumen.core.overlay_render import format_time_stamp
    from lumen.core.registry import REGISTRY
    s = Session()
    ij = IJ(s)
    img = s.add_image(LumenImage(np.zeros((5, 16, 16), np.uint8), name="z", axes="ZYX"))
    ij.run("Label...", "format=0 starting=1 interval=1 x=5 y=20 font=18")
    item = [o for o in img.overlay if o.get("type") == "time_stamp"][0]
    assert [format_time_stamp(item["format"], img, 0, z, item.get("start_frame", 0)) for z in range(5)] == ["1", "2", "3", "4", "5"]
    calls = []
    orig = ij._run
    ij._run = lambda op_id, params=None, **kw: calls.append((op_id, dict(params or {}))) or orig(op_id, params, **kw)
    ij.run("Label...", "format=0 starting=10 interval=5 x=5 y=20 font=18 text=lab")
    op, p = calls[-1]
    assert op == "stacks.label_slices" and p["format"].endswith(" lab") and "start_frame" not in p
    if {"start", "interval"} <= {q.name for q in REGISTRY.get(op).params}:  # label = starting + (n-1)*interval once the op supports it
        assert (p["start"], p["interval"]) == (10, 5) and p["format"].startswith("{label}")
    mov = s.add_image(LumenImage(np.zeros((6, 16, 16), np.uint8), name="t", axes="TYX"))
    ij.run("Time Stamper", "starting=10 interval=2 x=2 y=15 font=12 decimal=0 or=sec")
    op, p = calls[-1]
    item = [o for o in mov.overlay if o.get("type") == "time_stamp"][0]
    assert op == "overlay.time_stamp" and p["format"] == "{time:.0f} sec" and "start_frame" not in p  # 'starting' is a time, not a frame index
    stamps = [format_time_stamp(item["format"], mov, t, 0, item.get("start_frame", 0)) for t in range(6)]
    assert not any(st.startswith("-") for st in stamps)
    if {"start_time", "interval"} <= {q.name for q in REGISTRY.get(op).params}:
        assert (p["start_time"], p["interval"]) == (10, 2)


def test_for_loop_keeps_c_semantics(ij):
    assert ij.macro("for (i = 0; i < 3; i++) {}\nprint(i);") == "3\n"
    assert ij.macro("n = 0; for (i = 0; i < n; i++) {}\nprint(i);") == "0\n"
    assert ij.macro('a = newArray(5,7,9); n = a.length; for (i = 0; i < n; i++) { if (a[i] == 100) break; } if (i == n) print("not found"); else print("found at " + i);') == "not found\n"
    assert ij.macro("for (i = 0; i < 10; i++) { if (i == 3) i = 7; print(i); }").split() == ["0", "1", "2", "7", "8", "9"]
    assert ij.macro("for (i = 0; i < 5; i++) { if (i == 2) continue; print(i); }").split() == ["0", "1", "3", "4"]
    out = ij.macro('run("Clear Results"); for (k = 0; k < 6; k++) setResult("Area", k, k*10);'
                   'for (i = 0; i < nResults; i++) { if (getResult("Area", i) < 35) { IJ.deleteRows(i, i); i--; } }'
                   'for (i = 0; i < nResults; i++) print(getResult("Area", i));')
    assert out.split() == ["40", "50"]


def test_do_while_continue_retests_the_condition(ij):
    # standard C/Java semantics (real ImageJ 1.54 ignores continue in a do-while, which is an interpreter quirk)
    assert ij.macro('i = 0; do { i++; if (i >= 3) continue; print(i); } while (i < 3); print("i=" + i);') == "1\n2\ni=3\n"
    assert ij.macro('i = 0; do { i++; if (i % 2 == 0) continue; print(i); } while (i < 6); print("i=" + i);') == "1\n3\n5\ni=6\n"
    assert ij.macro('for (k=0;k<2;k++){ j=0; do { j++; if (j==1) continue; print(k+":"+j);} while (j<2);} print("done");') == "0:2\n1:2\ndone\n"
    assert ij.macro("i=0; do i++; while (i<3); print(i);") == "3\n"


def test_empty_statement_body(ij):
    assert transpile('if (false);\nprint("hello");') == "if False:\n    pass\nprint('hello')\n"
    assert ij.macro('if (false);\nprint("hello");') == "hello\n"
    assert ij.macro('if (false); else print("x");') == "x\n"
    assert ij.macro('for (i=0;i<3;i++);\nprint("done " + i);') == "done 3\n"
    assert ij.macro("{ ; ; } x = 1;;; print(x);") == "1\n"


def test_split_regex_form_matches_java(ij):
    from lumen.core.ij_compat import _split
    assert ij.macro('a = split("a,b,,c", "(,)"); print(a.length); Array.print(a);') == "4\na,b,,c\n"
    assert ij.macro('a = split("l1\\nl2\\nl3", "(\\n)"); print(a.length);') == "3\n"
    assert list(_split("a,b,,c,,", "(,)")) == ["a", "b", "", "c", "", ""]  # limit -1 keeps trailing empties
    assert list(_split("a1b22c", "(([0-9])+)")) == ["a", "b", "c"]  # inner groups never leak
    assert list(_split("a-b", "()")) == ["a-b"] and list(_split("", "(,)")) == []


def test_eval_and_runmacro_share_the_log(ij, tmp_path):
    assert ij.macro('print("a"); eval("print(\\"b\\");"); print("c");') == "a\nb\nc\n"
    p = tmp_path / "inner.ijm"
    p.write_text('print("inner");\n')
    assert ij.macro('print("a"); runMacro("%s"); print("c");' % p.as_posix()) == "a\ninner\nc\n"
    assert ij.macro('print("a"); eval("print(\\"b\\");"); print(getInfo("log"));').startswith("a\nb\na\nb\n")
    with pytest.raises(ValueError):
        ij.macro('eval("bogusFn();");')
    assert ij.macro('print("z");') == "z\n"  # depth restored after an error inside eval


def test_modulo_uses_java_remainder(ij):
    assert ij.macro("print(-5 % 3); print(5 % -3); print(-5.0 % 3); i = -1; n = 4; print(i % n); print(7 % 0); print(-7.5 % 2);") == "-2\n2\n-2\n-1\nNaN\n-1.5\n"


def test_replace_expands_java_group_references(ij):
    from lumen.core.ij_compat import _replace
    assert ij.macro('print(replace("img_12_34.tif", "img_(\\\\d+)_(\\\\d+).tif", "$2-$1"));') == "34-12\n"
    assert ij.macro('print(replace("a.b.c", ".", ""));') == "abc\n"  # single-char old → literal (ImageJ ≥ 1.53d)
    assert _replace("a.b", "\\.", "\\$") == "a$b" and _replace("y", "(x)?y", "[$1]") == "[]"
    assert _replace("ab", "(a)(b)", "$10") == "a0" and _replace("abcabc", "b", "$0$0") == "a$0$0ca$0$0c"


def test_len_variable_does_not_break_length(ij):
    assert ij.macro("a = newArray(1,2,3); len = 3; print(a.length);") == "3\n"
    assert ij.macro("function f(len) { b = newArray(1,2); return b.length + len; } print(f(10));") == "12\n"


def test_number_to_string_matches_imagej(ij):
    from lumen.core.ij_compat import _str
    assert [_str(x) for x in (3e-5, 1.234e-5, 5e-5, 9e-6, 0.5, 1 / 3, 123.456, 1e9, 1.5e9, 3e9, 1.5e10, -3e-5)] == \
        ["0.00003", "0.00001234", "0.00005", "9.0000E-6", "0.5", "0.3333", "123.456", "1000000000", "1500000000", "3.0000E9", "1.5000E10", "-0.00003"]
    assert ij.macro('print("p=" + 0.00003); print("" + 0.5); print("" + 1/3); print("" + 3000000000); print(d2s(3e-5, -4)); print(d2s(3.14159, 2));') == \
        "p=0.00003\n0.5\n0.3333\n3.0000E9\n3.0000E-5\n3.14\n"


# ---------------------------------------------------------------- review follow-ups (verified findings)
def _planes_changed(before, img):
    return [z for z in range(img.nz) if not np.array_equal(before[0, z, 0], img.data[0, z, 0])]


def test_transform_commands_honour_the_stack_keyword():
    """Flip / Rotate / Translate go through IJ.setupDialog in ImageJ: without 'stack' only the current slice is transformed."""
    for command, options in [("Flip Horizontally", ""), ("Flip Vertically", ""), ("Translate...", "x=3 y=0 interpolation=None"), ("Rotate... ", "angle=15 grid=1 interpolation=Bilinear")]:
        s = Session(); ij = IJ(s)
        img = _stack(s); img.display.z = 2
        before = img.data.copy()
        ij.run(command, options)
        assert _planes_changed(before, img) == [2], command
        assert s.undo(img.id) and np.array_equal(img.data, before), command  # one undo step restores the untouched stack
        ij.run(command, (options + " stack").strip())
        assert _planes_changed(before, img) == [0, 1, 2, 3, 4], command
    # a flip inside a selection mirrors the bounding box of the current plane only
    s = Session(); ij = IJ(s); img = _stack(s); img.display.z = 1; before = img.data.copy()
    ij.makeRectangle(4, 4, 10, 8)
    ij.run("Flip Horizontally", "")
    assert _planes_changed(before, img) == [1]
    assert np.array_equal(img.data[0, 1, 0, 4:12, 4:14], before[0, 1, 0, 4:12, 4:14][:, ::-1]) and np.array_equal(img.data[0, 1, 0, 20:, 20:], before[0, 1, 0, 20:, 20:])
    # Rotate + enlarge: the canvas grows for every slice (Rotator.enlargeCanvas), only the current slice is rotated
    s = Session(); ij = IJ(s); img = _stack(s); img.display.z = 2; before = img.data.copy()
    ij.run("Rotate... ", "angle=45 grid=1 interpolation=Bilinear enlarge")
    H, W = before.shape[-2:]
    nh, nw = img.data.shape[-2:]
    assert nh > H and nw > W
    oy, ox = (nh - H) // 2, (nw - W) // 2
    assert all(np.array_equal(img.data[0, z, 0, oy:oy + H, ox:ox + W], before[0, z, 0]) for z in (0, 1, 3, 4))
    assert not np.array_equal(img.data[0, 2, 0, oy:oy + H, ox:ox + W], before[0, 2, 0])
    # Rotate 90 / Flip Z are whole-stack commands in ImageJ too (no setupDialog)
    s = Session(); ij = IJ(s); img = _stack(s); img.display.z = 2; before = img.data.copy()
    ij.run("Flip Z", "")
    assert _planes_changed(before, img) == [0, 1, 3, 4]
    ij.run("Rotate 90 Degrees Right", "")
    assert all(np.array_equal(img.data[0, z, 0], np.rot90(before[0, 4 - z, 0], k=-1)) for z in range(5))


def test_roi_manager_add_gives_every_entry_its_own_id(ij):
    rm = ij.session.roi_manager
    assert ij.macro('makeRectangle(0,0,10,10); roiManager("Add"); roiManager("Add"); print(roiManager("count")); roiManager("select", 0); roiManager("delete"); print(roiManager("count"));') == "2\n1\n"
    ij.roiManager("reset")
    assert ij.macro('makeRectangle(20,20,30,30); roiManager("Add"); roiManager("select", 0); run("Enlarge...", "enlarge=5"); roiManager("Add"); print(roiManager("count")); '
                    'roiManager("select", 1); roiManager("delete"); print(roiManager("count"));') == "2\n1\n"
    assert len(rm.rois) == 1 and rm.rois[0]["w"] == pytest.approx(30)  # the enlarged copy went, the original stays
    ij.roiManager("reset")
    ij.macro('makeRectangle(0,0,10,10); roiManager("Add"); roiManager("Add"); roiManager("select", 1); makeRectangle(5,5,20,20); roiManager("Update");')
    assert [(r["x"], r["w"]) for r in rm.rois] == [(0, 10), (5, 20)] and len({r["id"] for r in rm.rois}) == 2  # Update patches the selected entry only


def test_list_set_measurements(ij):
    import os
    out = ij.macro('makeRectangle(10,10,20,20); List.setMeasurements; print(List.getValue("Mean")); print(List.getValue("Area")); print(List.getValue("Nope")); print(List.get("Area"));')
    assert out == "39.8225\n400\nNaN\n400.0\n" and ij.nResults() == 0  # bare ImageJ spelling; the Results table is not touched
    assert ij.macro('setThreshold(0, 40); makeRectangle(10,10,20,20); List.setMeasurements("limit"); a = List.getValue("Area"); print(a > 0 && a < 400); List.setMeasurements(); print(List.getValue("Area"));') == "1\n400\n"
    assert ij.macro('List.clear; List.set("a", 1); print(List.size); List.clear; print(List.size); List.set("s", "abc"); print(List.getValue("s")); print(File.separator);') == "1\n0\nNaN\n%s\n" % os.sep


def test_macro_bare_exit_aborts(ij):
    assert ij.macro('print("a"); exit; print("b");') == "a\n"
    assert "exit()" in transpile("if (nImages == 0) exit;")
    assert ij.macro('function f() { exit; } f(); print("no");') == ""
    assert ij.macro('print("a"); exit("msg"); print("b");') == "a\nmsg\n"
    assert ij.macro('beep; print("ok");') == "ok\n"
    empty = IJ(Session())
    assert empty.macro('if (nImages == 0) exit; print("b");') == "" and empty.macro('if (nImages == 0) exit\nprint("b");') == ""


def test_run_macro_passes_the_argument(ij, tmp_path):
    inner = tmp_path / "inner.ijm"
    inner.write_text('print("arg=" + getArgument());\n')
    outer = tmp_path / "outer.ijm"
    outer.write_text('print("outer=" + getArgument()); runMacro("%s", "inner"); print("outer again=" + getArgument());\n' % inner.as_posix())
    assert ij.macro('print("before"); runMacro("%s", "hello"); print("after");' % inner.as_posix()) == "before\narg=hello\nafter\n"
    assert ij.macro('runMacro("%s", "o"); print("top=" + getArgument());' % outer.as_posix()) == "outer=o\narg=inner\nouter again=o\ntop=\n"
    assert ij.macro('runMacro("%s", 4/2); runMacro("%s");' % (inner.as_posix(), inner.as_posix())) == "arg=2\narg=\n"  # numeric arguments read like ImageJ strings
    assert ij.macro('eval("print(\\"e=\\" + getArgument());", "ev"); print("<" + getArgument() + ">");') == "e=ev\n<>\n"
    bad = tmp_path / "bad.ijm"
    bad.write_text("bogusFn();\n")
    with pytest.raises(ValueError):
        ij.macro('runMacro("%s", "x");' % bad.as_posix())
    assert ij._args == [] and ij.macro('print("<" + getArgument() + ">");') == "<>\n"  # no stale argument after an error


def test_macro_leading_zero_integer_literals(ij):
    out = ij.macro("print(007); print(010); print(08); x = 007; print(x + 1); a = newArray(1, 2, 3); print(a[01]); for (i = 00; i < 02; i++) print(i); print(-08);")
    assert out.splitlines() == ["7", "10", "8", "8", "2", "0", "1", "-8"]
    assert ij.macro("print(007.5); print(1e400); print(0x1F); print(.5); print(1e3);") == "7.5\nInfinity\n31\n0.5\n1000\n"  # float / hex / exponent forms untouched
    assert transpile("x = 0x1F; y = .5; z = 1e3;") == "x = 31\ny = 0.5\nz = 1e3\n"
    with pytest.raises(ValueError, match="Unsupported macro construct"):  # generated Python that does not compile is reported like any other unsupported construct
        ij.macro("return 1;")


def test_label_and_time_stamper_render_like_imagej():
    from lumen.core.overlay_render import time_stamp_text
    s = Session(); ij = IJ(s)
    img = s.add_image(LumenImage(np.zeros((5, 16, 16), np.uint8), name="z", axes="ZYX"))
    ij.run("Label...", "format=0000 starting=10 interval=5 x=5 y=20 font=18 text=slice")
    item = [o for o in img.overlay if o.get("type") == "time_stamp"][0]
    assert (item["label_start"], item["label_interval"]) == (10, 5)
    assert [time_stamp_text(item, img, 0, z) for z in range(5)] == ["slice 0010", "slice 0015", "slice 0020", "slice 0025", "slice 0030"]  # StackLabeler: text + " " + zeroFill(n)
    ij.run("Label...", "format=0 starting=10 interval=5 text=slice")
    item = [o for o in img.overlay if o.get("type") == "time_stamp"][0]
    assert [time_stamp_text(item, img, 0, z) for z in range(5)] == ["10 slice", "15 slice", "20 slice", "25 slice", "30 slice"]  # number first for format 0
    mov = s.add_image(LumenImage(np.zeros((6, 16, 16), np.uint8), name="t", axes="TYX"))
    ij.run("Time Stamper", "starting=0 interval=2 x=2 y=15 font=14 decimal=1 anti-aliased or=sec")
    item = [o for o in mov.overlay if o.get("type") == "time_stamp"][0]
    assert (item["start_time"], item["interval"], item["format"]) == (0.0, 2.0, "{time:.1f} sec")
    assert time_stamp_text(item, mov, 3, 0) == "6.0 sec" and [time_stamp_text(item, mov, t, 0) for t in range(3)] == ["0.0 sec", "2.0 sec", "4.0 sec"]
    mov.calibration.frame_interval = 7.0
    ij.run("Time Stamper", "starting=10 interval=0.5 decimal=2 or=min")
    item = [o for o in mov.overlay if o.get("type") == "time_stamp"][0]
    assert [time_stamp_text(item, mov, t, 0) for t in range(3)] == ["10.00 min", "10.50 min", "11.00 min"]  # Time_Stamper never uses the frame interval
    s.run_op("overlay.burn_in", {}, image_id=mov.id)  # the rasteriser understands the new item keys


# ---------------------------------------------------------------- mappings added with the 2026-11 command modules
def _ids(session):
    return [im.name for im in session.images.values()]


def _capture(ij):
    """Replace ij._run with a recorder: returns the list of (op id, params) the handlers would run."""
    calls = []
    ij._run = lambda op, p=None, **k: calls.append((op, p or {})) or {"new_images": []}
    return calls


def test_edit_clipboard_commands(ij):
    ij.makeRectangle(10, 10, 40, 30)
    ij.run("Copy")
    assert ij.session.settings["clipboard"]["data"].shape == (1, 30, 40)
    ij.run("Paste Control...", "transfer=Blend")
    assert ij.session.settings["paste_mode"] == "Blend"
    ij.setPasteMode("Transparent-white")
    assert ij.session.settings["paste_mode"] == "Transparent-white"
    ij.setPasteMode("transparent zero")
    assert ij.session.settings["paste_mode"] == "Transparent-zero"
    ij.setPasteMode("Copy")
    ij.makeRectangle(100, 100, 40, 30)
    ij.run("Paste")
    assert np.array_equal(ij.img.current_plane()[100:130, 100:140], ij.session.settings["clipboard"]["data"][0])
    assert ij.roi["type"] == "rect" and ij.roi["x"] == 100  # Paste leaves the pasted rectangle selected
    ij.makeRectangle(10, 10, 40, 30)
    ij.run("Cut")
    assert int(ij.img.current_plane()[10:40, 10:50].max()) == 0  # cleared to the background value
    n = ij.nImages()
    ij.run("Internal Clipboard")
    assert ij.nImages() == n + 1 and ij.getTitle() == "Clipboard" and (ij.getWidth(), ij.getHeight()) == (40, 30)
    with pytest.raises(ValueError, match="transfer mode"):
        ij.run("Paste Control...", "transfer=Bogus")


def test_fit_circle_fit_rectangle_and_bounding_box(ij):
    ij.makeOval(20, 20, 60, 40)
    ij.run("Fit Circle")
    assert ij.roi["type"] == "oval" and abs(ij.roi["w"] - ij.roi["h"]) < 1e-6
    ij.makePolygon(10, 10, 60, 20, 50, 70, 15, 60)
    ij.run("Fit Rectangle")
    assert ij.roi["type"] == "polygon" and len(ij.roi["points"]) == 4
    ij.run("Select None")
    img = ij.newImage("box", "8-bit black", 50, 40, 1)
    img.data[0, 0, 0, 5:15, 7:30] = 200
    ij.run("Select Bounding Box")
    assert (ij.roi["x"], ij.roi["y"], ij.roi["w"], ij.roi["h"]) == (7, 5, 23, 10)


def test_roi_properties_dialog_both_spellings(ij):
    ij.makeRectangle(5, 5, 20, 20)
    ij.run("Properties... ", "name=lane1 stroke=red width=3 fill=none position=none")  # ImageJ: trailing space = selection properties
    assert ij.roi["name"] == "lane1" and ij.roi["color"] == "#ff0000" and ij.roi["stroke_width"] == 3.0 and "fill" not in ij.roi
    ij.run("Properties...", "name=lane2 stroke=blue fill=yellow group=2")  # same keys without the space → still the ROI dialog
    assert ij.roi["name"] == "lane2" and ij.roi["color"] == "#0000ff" and ij.roi["fill"] == "#ffff00" and ij.roi["group"] == 2
    st = _stack(ij.session, nz=4)
    ij.selectWindow("stack.tif")
    ij.makeRectangle(1, 1, 5, 5)
    ij.run("Properties... ", "position=3")  # a stack index → t,z,c of that plane (0-based inside Lumen)
    assert ij.roi["position"] == {"t": 0, "z": 2, "c": 0}
    ij.run("Properties...", "unit=um pixel_width=0.5 pixel_height=0.5 voxel_depth=1")  # image keys → Image ▸ Properties
    assert st.calibration.unit == "um" and st.calibration.pixel_width == 0.5


def test_restore_selection(ij):
    ij.makeRectangle(5, 5, 20, 20)
    ij.run("Select None")
    assert ij.roi is None
    ij.run("Restore Selection")
    assert ij.roi is not None and (ij.roi["x"], ij.roi["w"]) == (5, 20)
    ij.makeOval(0, 0, 10, 10)
    ij.restoreSelection()
    assert ij.roi["type"] == "rect"  # the oval replaced the rectangle, so the rectangle is what gets restored


def test_option_dialogs_map_to_preferences(ij):
    s = ij.session.settings
    ij.run("Colors...", "foreground=white background=black selection=cyan")
    assert s["roi_color"] == "#00ffff" and ij._fg == (255, 255, 255) and ij._bg == (0, 0, 0)
    ij.run("Input/Output...", "jpeg=70 gif=-1 file=.csv use_file copy_column save_column")
    assert s["jpeg_quality"] == 70 and s["default_save_format"] == "tif"  # untouched keys keep their value
    ij.run("Point Tool...", "type=Dot color=Red size=Large label show counter=0")
    assert (s["point_type"], s["point_size"], s["point_color"]) == ("Dot", 6, "#ff0000")
    ij.run("Arrow Tool...", "width=3 size=12 color=Yellow style=Open double")
    assert (s["arrow_style"], s["arrow_head_size"], s["arrow_width"], s["arrow_double"]) == ("Open", 12.0, 3.0, True)
    ij.run("Arrow Tool...", "width=3 size=12 style=Filled")
    assert s["arrow_double"] is False  # an absent check-box keyword means unchecked
    ij.run("Fonts...", "font=Serif size=18 style=[bold antialiased]")
    assert (s["font_family"], s["font_size"], s["antialiased_text"], ij.font_size) == ("Serif", 18, True, 18)
    ij.run("Appearance...", "interpolate auto menu=15 gui=1.0")
    assert s["interpolate_zoom"] is True and s["auto_contrast_stacks"] is True
    ij.run("Plots...", "width=800 height=400 minimum=0 maximum=255 interpolate")
    assert (s["plot_width"], s["plot_height"], s["plot_y_max"], s["plot_interpolate"]) == (800, 400, 255.0, True)
    ij.run("Rounded Rect Tool...", "corner=30 width=3")
    assert s["rounded_rect_radius"] == 15.0 and s["rounded_rect_stroke_width"] == 3.0  # ImageJ's corner= is a diameter
    ij.run("Misc...", "divide=NaN require")
    assert s["divide_by_zero"] == "NaN" and s["require_control_key"] is True
    for name in ("Arrow Tool...", "Appearance...", "Plots...", "Misc...", "Edit LUT..."):
        assert ij.run(name) is None  # the bare dialog is interactive → nothing changes


def test_8bit_color_rgb_and_grayscale(ij):
    ij.run("8-bit Color", "number=16")
    assert ij.img.data.dtype == np.uint8 and not ij.img.rgb
    ij.session.run_op("file.sample", {"name": "Astronaut (RGB)"})
    assert ij.img.rgb
    ij.run("8-bit Color", "number=8")
    assert not ij.img.rgb and ij.img.nc == 1 and len(np.unique(ij.img.current_plane())) <= 8


def test_lut_commands(ij):
    n = ij.nImages()
    ij.run("Show LUT")
    assert ij.nImages() == n + 1 and ij.img.rgb and ij.getTitle().startswith("LUT of")
    ij.selectWindow("blobs.tif")
    ij.run("Display LUTs")
    assert ij.nImages() == n + 2
    ij.selectWindow("blobs.tif")
    ij.run("Edit LUT...", "name=MyLut start=black end=red")
    assert ij.img.display.channels[0].lut == "MyLut"


def test_slice_labels_use_the_per_plane_list(ij):
    _stack(ij.session, nz=4)
    ij.selectWindow("stack.tif")
    ij.setSlice(3)
    ij.run("Set Label...", "label=third")
    assert ij.img.metadata["slice_labels"] == ["", "", "third", ""]
    assert ij.Stack.getSliceLabel() == "third" and ij.getMetadata("Label") == "third" and ij.getInfo("slice.label") == "third"
    ij.Stack.setSliceLabel("first", 1)
    assert ij.Stack.getSliceLabel(1) == "first" and ij.Property.getSliceLabel() == "third"
    ij.run("Remove Slice Labels")
    assert "slice_labels" not in ij.img.metadata and ij.Stack.getSliceLabel() == ""


def test_image_to_results_and_results_to_image(ij):
    ij.makeRectangle(10, 10, 8, 6)
    ij.run("Image to Results")
    assert ij.nResults() == 6 and ij.session.results.columns[:2] == ["X10", "X11"]
    assert ij.getResult("X10", 0) == float(ij.img.current_plane()[10, 10])
    ij.run("Results to Image")
    assert ij.getTitle() == "Results Table" and (ij.getWidth(), ij.getHeight()) == (8, 6) and ij.img.data.dtype == np.float32
    assert np.array_equal(ij.img.current_plane(), list(ij.session.images.values())[0].current_plane()[10:16, 10:18].astype(np.float32))


def test_montage_to_stack_inverts_make_montage():
    s = Session()
    ij = IJ(s)
    st = _stack(s, nz=4)
    ij.run("Make Montage...", "columns=2 rows=2 scale=1 border=0")
    assert ij.getWidth() == 64
    ij.run("Montage to Stack...", "columns=2 rows=2 border=0")
    assert ij.nSlices() == 4 and ij.getWidth() == 32
    assert np.array_equal(ij.img.data[0, :, 0], st.data[0, :, 0])
    ij.selectWindow("stack.tif")
    ij.run("Make Montage...", "columns=4 rows=1 scale=1 border=2")
    ij.run("Montage to Stack...", "images_per_row=4 images_per_column=1 border=2")  # ImageJ StackMaker's own keys
    assert ij.nSlices() == 4 and np.array_equal(ij.img.data[0, 3, 0], st.data[0, 3, 0])


def test_auto_crop_statistics_and_grid(ij):
    img = ij.newImage("pad", "8-bit black", 60, 50, 1)
    img.data[0, 0, 0, 10:30, 5:45] = 77
    ij.run("Auto Crop")
    assert (ij.getWidth(), ij.getHeight()) == (40, 20)
    ij.run("Statistics")
    assert ij.session.tables["Stack Statistics"]["rows"][0][1] == 800 and ij.getResult("Mean", ij.nResults() - 1) == 77.0
    ij.selectWindow("blobs.tif")
    ij.run("Grid...", "grid=Lines area=2000 color=Cyan bold")
    lines = [o for o in ij.img.overlay if o.get("grid")]
    assert lines and all(o["color"] == "#00ffff" and o["stroke_width"] == 2.0 for o in lines)
    ij.run("Grid...", "grid=Points area=2000")
    assert any(o["type"] == "point" for o in ij.img.overlay)
    ij.run("Grid...", "grid=None")
    assert not any(o.get("grid") for o in ij.img.overlay)


def test_interactive_transform_and_polar_transformer(ij):
    n = ij.nImages()
    ij.run("Interactive Affine", "source=[10,10; 200,10; 10,200] target=[12,12; 198,14; 10,205]")
    assert ij.nImages() == n + 1 and ij.getTitle().endswith("(affine)") and ij.getResultString("Model", ij.nResults() - 1) == "Affine"
    ij.selectWindow("blobs.tif")
    ij.makePoint(30, 30)
    ij.run("Polar Transformer", "method=Polar number=180 default_center")
    assert ij.getHeight() == 180 and ij.img.metadata["center"] == [128.0, 128.0]  # default_center ignores the point selection
    ij.selectWindow("blobs.tif")
    ij.run("Polar Transformer", "method=Polar number=90 polar=[40 50]")
    assert ij.getHeight() == 90 and ij.img.metadata["center"] == [40.0, 50.0]
    ij.selectWindow("blobs.tif")
    with pytest.raises(ValueError, match="polar"):
        ij.run("Polar Transformer", "method=Cartesian")


def test_add_image_overlay(ij):
    ij.run("Duplicate...", "title=inset")
    ij.run("Scale...", "x=0.25 y=0.25 interpolation=Bilinear create title=small")
    ij.selectWindow("blobs.tif")
    ij.run("Add Image...", "image=small x=10 y=20 opacity=50 zero")
    item = ij.img.overlay[-1]
    assert item["type"] == "image" and item["x"] == 10 and item["opacity"] == 0.5 and item["name"] == "small"


def test_fft_commands_round_trip(ij):
    plane = ij.img.current_plane().copy()
    ij.run("FFT Options...", "fft")
    assert ij.session.settings["fft_options"]["do_forward"] is True and ij.session.settings["fft_options"]["raw_power_spectrum"] is False
    ij.run("FFT")
    fft_title = ij.getTitle()
    assert fft_title.startswith("FFT of") and ij.img.data.dtype == np.uint8
    ps = ij.img.current_plane().copy()
    ij.run("Swap Quadrants")
    assert not np.array_equal(ps, ij.img.current_plane())
    ij.run("Swap Quadrants")
    assert np.array_equal(ps, ij.img.current_plane())
    ij.makeRectangle(0, 0, 20, 20)
    ij.run("Fill")  # paint into the spectrum …
    ij.run("Select None")
    ij.run("Redisplay Power Spectrum")  # … and restore it from the stored transform
    assert np.array_equal(ps, ij.img.current_plane())
    ij.run("Inverse FFT")
    assert ij.getTitle().startswith("Inverse FFT") and ij.img.data.dtype == np.uint8
    assert np.abs(ij.img.current_plane().astype(float) - plane).max() <= 1
    ij.selectWindow("blobs.tif")
    ij.run("FD Math...", "image1=blobs.tif operation=Correlate image2=blobs.tif result=Corr do")
    assert ij.getTitle() == "Corr" and ij.img.data.dtype == np.float32
    h, w = ij.img.current_plane().shape
    assert np.unravel_index(np.argmax(ij.img.current_plane()), (h, w)) == (h // 2, w // 2)  # auto-correlation peaks at the origin (centred)
    ij.selectWindow("blobs.tif")
    ij.run("FD Math...", "image1=blobs.tif operation=Convolve image2=blobs.tif result=Freq")  # no 'do' → frequency-domain result
    assert ij.getTitle() == "Freq" and ij.img.metadata.get("fft") is True
    ij.run("Duplicate...", "title=filt")
    ij.run("Select All"); ij.run("Fill"); ij.run("Select None")  # an all-pass filter
    ij.selectWindow("blobs.tif")
    ij.run("Custom Filter...", "filter=filt")
    assert np.abs(ij.img.current_plane().astype(float) - plane).max() <= 1
    ij.run("FFT Options...", "fft raw do")
    assert ij.getTitle().startswith(("PS of", "FFT of"))  # 'do' runs the forward transform right away ('raw' adds the 32-bit power spectrum)


def test_process_filter_mappings(ij):
    before = ij.img.current_plane().copy()
    calls = _capture(ij)
    ij.run("Normalize Local Contrast", "block_radius_x=20 block_radius_y=25 standard_deviations=3 center stretch")
    ij.run("ROF Denoise", "theta=30")
    ij.run("Retinex", "level=Low scale=100 scale_division=2 dynamic=1.20")
    ij.run("Shape Index Map", "gaussian=3")
    ij.run("Subtract Background...", "rolling=20 sliding light")
    ij.run("Auto Local Threshold", "method=midgrey radius=15 parameter_1=0 parameter_2=0 white")
    ij.run("Ultimate Points", "output=Image")
    ij.run("Watershed", "method=[ImageJ EDM/UEP]")
    ij.run("Skeletonize", "method=Lee")
    ij.run("Macro...", "code=[v=v+50*sin(d/10)] stack")
    assert calls == [
        ("filters.normalize_local_contrast", {"block_radius_x": 20, "block_radius_y": 25, "standard_deviations": 3, "center": True, "stretch": True}),
        ("filters.rof_denoise", {"theta": 30}),
        ("process.retinex", {"level": "Low", "scale": 100, "scale_division": 2, "dynamic": 1.2}),
        ("filters.shape_index", {"sigma": 3}),
        ("process.subtract_background", {"radius": 20, "light_background": True, "create_background": False, "smooth": True, "method": "Sliding paraboloid"}),
        ("threshold.local", {"method": "MidGrey", "radius": 15, "parameter_1": 0, "white_objects": True}),
        ("binary.ultimate_points", {"output": "Image"}),
        ("binary.watershed", {"method": "ImageJ EDM/UEP"}),
        ("binary.skeletonize", {"method": "Lee"}),
        ("math.macro", {"code": "v=v+50*sin(d/10)"}),
    ]
    del ij._run  # back to the real thing
    ij.run("Dither")
    assert set(np.unique(ij.img.current_plane())) <= {0, 255}
    ij.run("Macro...", "code=[v=255-v]")
    assert np.array_equal(ij.img.current_plane(), 255 - np.where(before > 127, 255, 0).astype(np.uint8)) or set(np.unique(ij.img.current_plane())) <= {0, 255}
    n = ij.nImages()
    ij.run("Show Circular Masks")
    assert ij.nImages() == n + 1
    ij.selectWindow("blobs.tif")
    ij.run("Ultimate Points", "output=Image")
    assert ij.getTitle().startswith("UEP of") and ij.img.data.dtype == np.float32
    ij.selectWindow("blobs.tif")
    ij.run("Exact Euclidean Distance Transform (3D)")
    assert ij.getTitle().startswith("EDM 3D") and ij.img.data.dtype == np.float32


def test_analyze_line_graph_directionality_and_skeleton(ij):
    ij.run("Analyze Line Graph")
    assert ij.nResults() > 10 and "Curve" in ij.session.results.columns
    ij.run("Clear Results")
    ij.run("Directionality", "method=[Fourier components] nbins=45 histogram_start=-90 histogram_end=90 display_table")
    assert ij.nResults() == 1 and "Direction (°)" in ij.session.results.columns
    assert len(ij.session.tables["Directionality histograms"]["rows"]) == 45
    ij.run("Clear Results")
    ij.run("Auto Threshold", "method=Default white")
    ij.run("Skeletonize")
    ij.run("Analyze Skeleton (2D/3D)", "prune=none calculate show display")
    assert ij.nResults() >= 1 and "# Branches" in ij.session.results.columns and ij.isOpen("Tagged skeleton of blobs.tif")
    assert "Branches" in ij.session.tables


def test_coloc2_two_images_and_one_hyperstack():
    s = Session()
    ij = IJ(s)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, size=(40, 40), dtype=np.uint8)
    b = np.clip(a.astype(int) + rng.integers(-20, 20, size=(40, 40)), 0, 255).astype(np.uint8)
    s.add_image(LumenImage(np.stack([a, b])[None, None], name="two", axes="TZCYX"))
    ij.run("Coloc 2", "channel_1=two channel_2=two psf=3 costes_randomisations=5")
    assert ij.nResults() == 1 and ij.getResult("Pearson", 0) > 0.9 and ij.getResult("Channel B", 0) == 2
    s.add_image(LumenImage(a, name="A", axes="YX"))
    s.add_image(LumenImage(b, name="B", axes="YX"))
    n = len(s.images)
    ij.run("Coloc 2", "channel_1=A channel_2=B roi_or_mask=[<None>] threshold_regression=Costes psf=3 costes_randomisations=5")
    assert ij.nResults() == 2 and ij.getResult("Pearson", 1) > 0.9
    assert len(s.images) == n  # the temporary 2-channel image is closed again (the scatter plot is off)


def test_gel_commands(ij):
    ij.makeRectangle(10, 0, 20, 100)
    ij.run("Select First Lane")
    ij.makeRectangle(40, 0, 20, 100)
    ij.run("Select Next Lane")
    assert [r["name"] for r in ij.session.roi_manager.rois] == ["Lane 1", "Lane 2"]
    ij.run("Gel Analyzer Options...", "vertical label")
    assert ij.gel_options == {"orientation": "Vertical", "dark_bands": True, "label": "Percent"}
    n = ij.nImages()
    ij.run("Plot Lanes")
    assert ij.nImages() == n + 1 and ij.getTitle().startswith("Plots of") and len(ij.session.last_plot["series"]) == 2
    ij.selectWindow("blobs.tif")
    ij.run("Label Peaks")
    assert ij.nResults() >= 1 and "Percent" in ij.session.results.columns and any(o["type"] == "text" and o["text"].endswith("%") for o in ij.img.overlay)
    ij.makeRectangle(70, 0, 20, 100)
    ij.run("Select First Lane")  # starts a new gel: the old lanes are dropped
    assert [r["name"] for r in ij.session.roi_manager.rois] == ["Lane 1"]


def test_hyperstack_commands():
    s = Session()
    ij = IJ(s)
    vol = np.random.default_rng(2).integers(0, 255, size=(3, 4, 2, 24, 24), dtype=np.uint8)
    s.add_image(LumenImage(vol, name="hyper", axes="TZCYX"))
    ij.Stack.setPosition(2, 3, 2)
    ij.run("Reduce Dimensionality...", "channels slices keep")
    assert ij.getDimensions() == (24, 24, 2, 4, 1) and np.array_equal(ij.img.data[0], vol[1])
    assert ij.isOpen("hyper")
    ij.selectWindow("hyper")
    ij.run("Make Subset...", "channels=1 slices=1-2 frames=1,3")
    assert ij.getDimensions() == (24, 24, 1, 2, 2) and np.array_equal(ij.img.data[1, 1, 0], vol[2, 1, 0])
    ij.selectWindow("hyper")
    ij.run("Hyperstack to Stack")
    assert ij.getDimensions() == (24, 24, 1, 24, 1) and ij.Stack.getSliceLabel(2) == "c:2/2 z:1/4 t:1/3"
    _stack(s, nz=3)
    ij.selectWindow("stack.tif")
    assert ij.run("Hyperstack to Stack") is None  # a plain stack is left alone


def test_reslice_variants():
    s = Session()
    ij = IJ(s)
    vol = np.random.default_rng(3).integers(0, 255, size=(1, 4, 1, 24, 30), dtype=np.uint8)
    s.add_image(LumenImage(vol, name="vol", axes="TZCYX"))
    ij.run("Reslice [/]...", "output=1.000 start=Top avoid")
    assert ij.getDimensions() == (30, 4, 1, 24, 1)
    ij.selectWindow("vol")
    ij.run("Reslice [/]...", "output=1.000 start=Left avoid")
    assert ij.getDimensions() == (24, 4, 1, 30, 1)
    ij.selectWindow("vol")
    ij.makeLine(2, 2, 20, 20)
    ij.run("Reslice [/]...", "output=1.000 slice_count=3 avoid")
    assert ij.nSlices() == 3 and ij.getHeight() == 4
    ij.selectWindow("vol")
    ij.run("Select None")
    ij.run("Radial Reslice", "angle=180 degrees_per_slice=10 direction=Clockwise rotate")
    assert ij.nSlices() == 18 and ij.getHeight() == 4


def test_registration_plugin_mappings():
    s = Session()
    ij = IJ(s)
    rng = np.random.default_rng(4)
    base = (rng.random((64, 64)) * 255).astype(np.uint8)
    s.add_image(LumenImage(base, name="im1", axes="YX"))
    s.add_image(LumenImage(np.roll(base, (3, -2), axis=(0, 1)), name="im2", axes="YX"))
    ij.selectWindow("im1")
    ij.run("Phase Correlation", "first_image=im1 second_image=im2 subpixel_accuracy")
    assert (ij.getResult("dx", 0), ij.getResult("dy", 0)) == (-2.0, 3.0)
    ij.run("Pairwise stitching", "first_image=im1 second_image=im2 fusion_method=[Linear Blending] fused_image=[fused] check_peaks=5")
    assert ij.getTitle() == "fused"
    from skimage import data as skdata
    cam = skdata.camera()[100:228, 100:228]  # SIFT needs real structure, not 64×64 noise
    s.add_image(LumenImage(cam, name="f1", axes="YX"))
    s.add_image(LumenImage(np.roll(cam, (5, -7), axis=(0, 1)), name="f2", axes="YX"))
    ij.run("Extract SIFT Correspondences", "source_image=f1 target_image=f2 initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 "
                                           "feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 "
                                           "expected_transformation=Rigid")
    assert "Correspondences" in s.tables and ij.getResultString("Model", ij.nResults() - 1) == "Rigid body" and ij.getResultString("Method", ij.nResults() - 1) == "SIFT"
    ij.selectWindow("im1")
    ij.run("bUnwarpJ", "source_image=im1 target_image=im2 registration=Fast image_subsampling_factor=1 final_deformation=Fine")
    assert ij.getTitle() == "im1 on im2 (bUnwarpJ)"
    st = _stack(s, nz=3)
    ij.selectWindow("stack.tif")
    ij.run("Elastic Stack Alignment", "")
    assert "Alignment" in s.tables and ij.img.id == st.id
    ij.run("Color Clustering", "number_of_clusters=3")
    assert ij.getTitle().startswith("Clusters of") and len(s.tables["Clusters"]["rows"]) == 3


def test_about_plugins_and_plugin_noops(ij):
    ij.run("About Plugins")
    assert "Plugins" in ij.session.tables and ij.session.tables["Plugins"]["rows"]
    for name in ("Manual Tracking", "3D Viewer", "Capture Screen", "Help on Menu Item"):
        assert ij.run(name) is None and COMMANDS[name][1] == "(no-op)"


def test_file_import_export_commands(ij, tmp_path):
    ij.run("Measure")
    csv = tmp_path / "res.csv"
    ij.run("Results...", f"save=[{csv}]")
    assert csv.exists()
    ij.run("Clear Results")
    ij.run("Results... ", f"open=[{csv}]")  # ImageJ's import spelling has a trailing space
    assert ij.nResults() == 1 and ij.getResult("Mean", 0) > 0
    ij.run("Table...", f"open=[{csv}]")
    assert "res" in ij.session.tables
    txt = tmp_path / "notes.txt"
    txt.write_text("a\nb\n")
    ij.run("Text File...", f"open=[{txt}]")
    assert len(ij.session.tables["notes.txt"]["rows"]) == 2
    ij.run("ZIP...", f"save=[{tmp_path / 'img.zip'}]")
    assert (tmp_path / "img.zip").exists()
    lut = tmp_path / "mylut.lut"
    ij.run("LUT...", f"save=[{lut}]")
    assert lut.stat().st_size == 768
    ij.run("LUT... ", f"open=[{lut}]")
    assert ij.img.display.channels[0].lut == "mylut"
    xy = tmp_path / "xy.txt"
    xy.write_text("10 10\n50 10\n50 50\n10 50\n")
    ij.run("XY Coordinates... ", f"open=[{xy}]")
    assert ij.roi["type"] == "polygon" and len(ij.roi["points"]) == 4
    ij.run("XY Coordinates...", f"save=[{tmp_path / 'out.csv'}]")
    assert (tmp_path / "out.csv").read_text().count("\n") == 5
    ij.run("Text...", f"save=[{tmp_path / 'results.txt'}]")  # Results has rows → they are saved
    assert "Mean" in (tmp_path / "results.txt").read_text()
    with pytest.raises(ValueError, match="open="):
        ij.run("Stack From List...", "")


def test_open_next_stack_from_list_and_batch_measure(ij, tmp_path):
    ij.run("Tiff...", f"path=[{tmp_path / 'a.tif'}]")
    ij.run("Duplicate...", "title=b")
    ij.run("Invert")
    ij.run("Tiff...", f"path=[{tmp_path / 'b.tif'}]")
    ij.run("Open Next")
    # Open Next closes the current image, like ImageJ; the first image is still open as "a.tif" (Save As renamed it,
    # ImageJ semantics), so the re-opened file gets Lumen's unique title "a.tif-1"
    assert ij.getTitle() == "a.tif-1" and not ij.isOpen("b")
    lst = tmp_path / "list.txt"
    lst.write_text("a.tif\nb.tif\n")
    ij.run("Stack From List...", f"open=[{lst}]")
    assert ij.nSlices() == 2 and ij.getTitle().startswith("list")
    ij.run("Image Sequence... ", f"dir=[{tmp_path / 'seq'}] format=PNG name=img digits=3")
    assert sorted(p.name for p in (tmp_path / "seq").iterdir()) == ["img001.png", "img002.png"]
    ij.run("Clear Results")
    ij.run("Measure...", f"input=[{tmp_path}] file=.tif")
    assert ij.nResults() == 2 and "File" in ij.session.results.columns
    raw = tmp_path / "raw.bin"
    np.arange(16 * 8, dtype=np.uint16).tofile(raw)
    ij.run("Raw...", f"open=[{raw}] image=[16-bit Unsigned] width=16 height=8 offset=0 number=1 gap=0 little-endian")
    assert (ij.getWidth(), ij.getHeight()) == (16, 8) and ij.img.data.dtype == np.uint16 and int(ij.getPixel(15, 7)) == 127


def test_save_as_formats(ij, tmp_path):
    assert ij.saveAs("ZIP", str(tmp_path / "z.zip")).endswith("z.zip") and (tmp_path / "z.zip").exists()
    ij.saveAs("LUT", str(tmp_path / "l.lut"))
    assert (tmp_path / "l.lut").stat().st_size == 768
    ij.makeRectangle(1, 1, 5, 5)
    ij.saveAs("XY Coordinates", str(tmp_path / "xy.txt"))
    assert (tmp_path / "xy.txt").read_text().startswith("X\tY")
    ij.saveAs("Selection", str(tmp_path / "sel.roi"))
    assert (tmp_path / "sel.roi").stat().st_size > 0
    ij.run("Measure")
    import importlib.util
    if importlib.util.find_spec("openpyxl"):  # .xlsx needs the optional [formats] extra
        ij.saveAs("Results", str(tmp_path / "r.xlsx"))
        assert (tmp_path / "r.xlsx").stat().st_size > 0
    ij.saveAs("Results", str(tmp_path / "r.csv"))
    assert "Mean" in (tmp_path / "r.csv").read_text()
    ij.saveAs("Text", str(tmp_path / "t.txt"))  # nothing printed → the Results table
    assert "Mean" in (tmp_path / "t.txt").read_text()
    ij.print("hello log")
    ij.saveAs("Text", str(tmp_path / "log.txt"))
    assert (tmp_path / "log.txt").read_text() == "hello log\n"


def test_imagej_sample_names_map_to_builtin_or_download(ij):
    calls = _capture(ij)
    for name in ("Blobs (25K)", "T1 Head (2.4M, 16-bits)", "Clown", "AuPbSn 40 (56K)", "Fly Brain (1MB)", "Cardio (768K, RGB DICOM)", "Neuron (1.6M, 5 channels)", "Boats"):
        ij.run(name)
    ij.run("Clown (14K)", "download")
    assert calls == [("file.sample", {"name": "Blobs"}), ("file.sample", {"name": "T1 head (MRI stack)"}), ("file.sample", {"name": "Astronaut (RGB)"}),
                     ("file.open_imagej_sample", {"name": "AuPbSn 40", "use_cache": True}), ("file.open_imagej_sample", {"name": "Flybrain", "use_cache": True}),
                     ("file.open_imagej_sample", {"name": "Cardio (RGB DICOM)", "use_cache": True}), ("file.open_imagej_sample", {"name": "Neuron", "use_cache": True}),
                     ("file.open_imagej_sample", {"name": "Boats", "use_cache": True}), ("file.open_imagej_sample", {"name": "Clown", "use_cache": True})]
    from lumen.core.ops.file_extra import IJ_SAMPLE_NAMES
    assert len(IJ_SAMPLE_NAMES) == 29 and all(n in COMMANDS for n in IJ_SAMPLE_NAMES)  # every ImageJ sample name resolves


def test_command_table_counts_after_the_new_mappings():
    rows = command_table()
    by = {r["command"]: r for r in rows}
    assert len(COMMANDS) >= 390 and len(rows) >= 270
    noop = [r for r in rows if r["lumen"] == "(no-op)"]
    unsupported = [r for r in rows if r["lumen"] == "(unsupported)"]
    assert len(noop) == 1 and '"Monitor Memory..."' in noop[0]["command"] and '"Misc..."' not in noop[0]["command"]
    assert not unsupported  # the Bio-Formats commands, TIFF Virtual Stack... and AVI... are mapped since the Bio-Formats parity work
    assert by['"Bio-Formats Importer" / "Bio-Formats" / "Bio-Formats (Windowless)" / "Bio-Formats Windowless Importer"']["lumen"] == "file.bioformats_import"
    assert by['"Bio-Formats Exporter"']["lumen"] == "file.bioformats_export" and by['"TIFF Virtual Stack..."']["lumen"] == "file.import_tiff_virtual"
    assert by['"Bio-Formats Macro Extensions"']["lumen"] == "Ext.*" and by['"Stack Slicer"']["lumen"] == "file.stack_slicer"
    for name, op in (("Fit Circle", "selection.fit_circle"), ("Inverse FFT", "process.inverse_fft"), ("FD Math...", "process.fd_math"), ("Add Image...", "overlay.add_image"),
                     ("Image to Results", "image.to_results"), ("Results to Image", "image.from_results"), ("Custom Filter...", "process.fft_custom_filter"),
                     ("Swap Quadrants", "process.fft_swap_quadrants"), ("Montage to Stack...", "stacks.montage_to_stack"), ("Reslice [/]...", "stacks.reslice / stacks.reslice_line"),
                     ("Open Next", "file.open_next"), ("Internal Clipboard", "file.new_from_clipboard"), ("Cut", "edit.cut"), ("Paste Control...", "edit.paste_control"),
                     ("Hyperstack to Stack", "stacks.hyperstack_to_stack"), ("Set Label...", "stacks.set_slice_label"), ("Macro...", "math.macro"), ("Coloc 2", "analyze.coloc_costes"),
                     ("bUnwarpJ", "registration.bunwarpj"), ("Load a TrackMate file", "tracking.import_trackmate_xml"), ("About Plugins", "help.about_plugins")):
        assert COMMANDS[name][1] == op, name
    assert "diameter" in by['"Rounded Rect Tool..."']["notes"] and "trailing space" in by['"LUT... " / "LUT..."']["notes"]


# ---------------------------------------------------------------- Make Substack: every range re-indexes the slice labels
def test_make_substack_multi_axis_keeps_slice_labels():
    s = Session()
    vol = np.zeros((3, 2, 8, 8), np.uint8)
    for z in range(3):
        for c in range(2):
            vol[z, c] = 10 * z + c
    img = s.add_image(LumenImage(vol, name="hs.tif", axes="ZCYX"))
    img.metadata["slice_labels"] = [f"z{z + 1}c{c + 1}" for z in range(3) for c in range(2)]
    IJ(s).macro('run("Make Substack...", "channels=2 slices=1-2");')
    sub = s.active
    assert sub.name.startswith("Substack") and sub.shape[:3] == (1, 2, 1)
    assert [int(sub.data[0, z, 0, 0, 0]) for z in range(2)] == [1, 11]
    assert sub.metadata["slice_labels"] == ["z1c2", "z2c2"]  # the 2nd range (slices) re-indexed the labels too
    assert img.metadata["slice_labels"] == [f"z{z + 1}c{c + 1}" for z in range(3) for c in range(2)]  # the source is untouched


# ---------------------------------------------------------------- roiManager("Measure") honours Set Measurements
def test_roi_manager_measure_honours_set_measurements():
    s = Session()
    ij = IJ(s)
    ij.macro('run("Blobs (25K)"); makeRectangle(10, 20, 40, 30); roiManager("Add"); '
             'run("Set Measurements...", "area centroid invert redirect=None decimal=3"); run("Measure"); roiManager("Measure");')
    rows = s.results.rows
    assert len(rows) == 2 and rows[0] == rows[1]  # the manager row equals the Measure row for the same ROI
    assert "Label" not in rows[1]  # 'display' (Display label) was not ticked
    assert rows[1]["Y"] == pytest.approx(ij.img.height - (20 + 30 / 2) - 1)  # 'invert' (Invert Y coordinates) applied: ImageJ's height − y − 1
    ij.macro('run("Set Measurements...", "area centroid display redirect=None decimal=3"); roiManager("Measure");')
    assert s.results.rows[2]["Label"].startswith(ij.img.name + ":") and s.results.rows[2]["Y"] == pytest.approx(20 + 30 / 2)


# ---------------------------------------------------------------- phase 5: macro names → op ids / params (G1, MorphoLibJ, registration estimators, Weka, E2)
_P5_CASES = [
    ("Fit Circle to Image", "", "selection.fit_circle_to_image", {}),
    ("Points from Mask", "step=2 max_points=10", "selection.points_from_mask", {"step": "2", "max_points": "10"}),
    ("Labels...", "color=white font=12 show use bold draw", "analyze.label", {"what": "Index", "use_names": True, "font_size": 12, "color": "white", "bold": True, "background": True, "source": "Overlay"}),
    ("Measure Overlay", "", "overlay.measure", {}),
    ("Overlay Options...", "width=2 fill=none show use font=14 color=yellow background", "edit.options_overlay",
     {"overlay_stroke_width": 2, "overlay_default_fill": "none", "overlay_show_labels": True, "overlay_use_names": True, "overlay_label_font_size": 14, "overlay_label_color": "yellow",
      "overlay_label_background": True}),
    ("Search...", "search=blobs case", "help.search_files", {"query": "blobs", "case_sensitive": True}),
    ("Linear Gradient", "angle=45", "edit.linear_gradient", {"angle": "45"}),
    ("Radial Gradient", "radius=20", "edit.radial_gradient", {"radius": "20"}),
    ("Update ImageJ...", "check", "help.update", {"check_only": True, "dry_run": False}),
    ("Update Fiji...", "dry", "help.update", {"check_only": False, "dry_run": True}),
    ("Text Window", "name=[Notes.txt] text=hello", "file.new_text", {"name": "Notes.txt", "text": "hello"}),
    ("Table...", "name=[My Table] headings=[A,B]", "table.new", {"name": "My Table", "columns": "A,B"}),
    # MorphoLibJ
    ("Labels To RGB", "colormap=[Golden angle] background=White shuffle", "morpho.labels_to_rgb", {"lut": "Golden angle", "background": "white", "shuffle": True}),
    ("Labels To RGB", "colormap=[Glasbey Dark] background=Black", "morpho.labels_to_rgb", {"lut": "Glasbey on dark", "shuffle": False}),
    ("Assign Measure to Label", "results=Results column=Mean", "morpho.assign_measure_to_label", {"table": "Results", "column": "Mean"}),
    ("Microstructure Analysis", "area perimeter directions=2 connectivity=4", "morpho.microstructure_analysis",
     {"mode": "2D (current plane)", "features": ["Area / volume density", "Perimeter / surface density"], "directions_2d": "2", "connectivity": "4"}),
    ("Microstructure Analysis 3D", "volume mean_breadth euler_number directions=3 connectivity=26", "morpho.microstructure_analysis",
     {"mode": "3D (whole stack)", "features": ["Area / volume density", "Mean breadth density", "Euler density"], "directions_3d": "3", "connectivity": "8"}),
    ("Analyze Regions 3D", "volume surface_area sphericity euler_number max._inscribed surface_area_method=[Crofton (3 dirs.)] euler_connectivity=C6", "morpho.analyze_regions_3d",
     {"features": ["Volume", "Surface area", "Sphericity", "Euler number", "Inscribed sphere"], "directions": "3", "connectivity": "6"}),
    ("Morphological Filters", "operation=[White Top Hat] element=[Line 45 degrees] radius=3", "morpho.filters", {"operation": "White Top Hat", "element": "Line", "line_angle": 45.0, "radius": 3}),
    ("Morphological Filters", "operation=Erosion element=[Vertical Line] radius=2 stack", "morpho.filters", {"operation": "Erosion", "element": "Line", "line_angle": 90.0}),
    ("Morphological Filters (3D)", "operation=Closing element=Cube x-radius=2 y-radius=2 z-radius=1", "morpho.filters_3d",
     {"operation": "Closing", "element": "Cube", "radius_x": 2, "radius_y": 2, "radius_z": 1}),
    ("Regional Min & Max 3D", "operation=[Regional Maxima] connectivity=26", "morpho.regional_min_max", {"operation": "Regional Maxima", "connectivity": "8", "mode": "3D (whole stack)"}),
    ("Extended Min & Max", "operation=[Extended Minima] dynamic=15 connectivity=8", "morpho.extended_min_max", {"operation": "Extended Minima", "dynamic": 15, "connectivity": "8", "mode": "2D (current plane)"}),
    ("Impose Min & Max", "original=blobs.gif marker=im2 operation=[Impose Minima] connectivity=4", "morpho.impose_min_max", {"operation": "Impose Minima", "marker_image": "@im2", "connectivity": "4"}),
    ("Chamfer Distance Map", "distances=[Chessknight (5,7,11)] output=[16 bits] normalize", "morpho.chamfer_distance_map", {"weights": "Chessknight", "output": "16-bit", "normalize": True}),
    ("Chamfer Distance Map 3D", "distances=[Svensson (3,4,5,7)] output=[32 bits]", "morpho.chamfer_distance_map", {"weights": "Borgefors", "output": "32-bit", "normalize": False, "mode": "3D (whole stack)"}),
    ("Geodesic Distance Map", "marker=im2 mask=blobs.gif distances=[Borgefors (3,4)] normalize", "morpho.geodesic_distance_map", {"marker_image": "@im2", "weights": "Borgefors", "normalize": True}),
    ("Marker-controlled Watershed", "input=blobs.gif marker=im2 mask=None compactness=0 calculate use", "morpho.marker_watershed",
     {"markers": "Marker image", "marker_image": "@im2", "calculate_dams": True, "connectivity": "8", "mode": "2D (current plane)"}),
    ("Morphological Reconstruction", "marker=im2 mask=blobs.gif type=[By Erosion] connectivity=8", "morpho.reconstruct", {"operation": "By Erosion", "marker_image": "@im2", "connectivity": "8"}),
    ("Kill Borders", "", "morpho.kill_borders", {}),
    ("Fill Holes (Binary/Gray)", "", "morpho.fill_holes", {}),
    ("Area Opening", "pixel=50", "morpho.area_opening", {"min_area": "50"}),
    ("Directional Filtering", "type=Max operation=Opening line=20 direction=32", "morpho.directional_filter", {"type": "Max", "operation": "Opening", "line_length": "20", "n_directions": "32"}),
    ("Binary Overlay", "reference=blobs.gif binary=im2 overlay=Green", "morpho.binary_overlay", {"mask_image": "@im2", "color": "#00ff00"}),
    ("Label Overlay", "reference=blobs.gif labels=im2 opacity=40", "morpho.label_overlay", {"label_image": "@im2", "opacity": 0.4}),
    ("Merge Labels", "label(s)=1,2", "morpho.merge_labels", {"labels": "1,2"}),
    ("Label Size Opening", "min_size=30", "morpho.label_size_opening", {"min_size": "30"}),
    ("Select Label(s)", "label(s)=3", "morpho.select_label", {"label": 3, "add_to_manager": False}),
    ("Remove Label(s)", "label(s)=1,2", "morpho.replace_label", {"labels": "1,2", "new_value": 0}),
    # registration estimators / time lapse
    ("Linear Stack Alignment with SIFT", "initial_gaussian_blur=1.60 steps_per_scale_octave=3 minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 "
     "feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Affine interpolate", "registration.align_stack",
     {"axis": "Z", "estimator": "SIFT feature matching", "mode": "Affine", "reference": "Previous slice", "initial_blur": 1.6, "descriptor_size": 4, "max_ratio": 0.92, "interpolation": "Linear",
      "show_info": False, "create_new": True}),
    ("StackReg", "transformation=[Scaled Rotation]", "registration.align_stack", {"estimator": "Intensity (pyramid optimisation)", "mode": "Scaled rotation", "reference": "Current slice", "create_new": False}),
    ("Descriptor-based registration (2d/3d)", "first_image=blobs.gif second_image=im2 brightness_of_objects=Low approximate_size=[5 px] type_of_detections=[Minima & Maxima] "
     "transformation_model=[Affine (2d)] number_of_neighbors=3 redundancy=1 significance=3 allowed_error_for_ransac=5 choose_registration_channel_for_image_1=1", "registration.descriptor_registration",
     {"target": "@im2", "threshold": 0.1, "size": 5.0, "objects": "Bright and dark", "model": "Affine", "n_neighbors": 3, "ransac_threshold": 5, "channel": 1, "create_new": True}),
    ("Descriptor-based series registration (2d/3d + t)", "brightness_of_objects=[Very strong] approximate_size=[3 px] type_of_detections=[Maxima only] transformation_model=[Rigid (2d)] "
     "register=[Register to first frame] choose_registration_channel=2", "registration.descriptor_series",
     {"threshold": 0.01, "size": 3.0, "objects": "Bright", "model": "Rigid body", "reference": "First frame", "channel": 2, "create_new": False}),
    ("Extract MOPS Correspondences", "source_image=blobs.gif target_image=im2 expected_transformation=Perspective feature_descriptor_size=16", "registration.extract_correspondences",
     {"target": "@im2", "method": "MOPS", "model": "Projective", "descriptor_size": 16, "add_rois": True}),
    ("Phase Map", "min_period=3 max_period=20 detrend=Linear phase=Radians output=[Frequency (Hz)] reference=[Mean signal]", "timelapse.phase_map",
     {"min_period": 3, "max_period": 20, "detrend": "Linear", "window": True, "phase_unit": "Radians", "frequency_output": "Frequency (Hz)", "reference": "Mean signal"}),  # window is on unless…
    ("Phase Map", "min_period=2 window", "timelapse.phase_map", {"window": True}),
    # Weka
    ("Trainable Weka Segmentation", "classes=[bg,fg] features=[Gaussian_blur Sobel_filter Hessian] max_sigma=4 classifier=[weka.classifiers.trees.FastRandomForest] n_trees=50", "segment.train_classifier",
     {"classes": "bg,fg", "features": "Gaussian blur,Sobel filter,Hessian", "max_sigma": 4, "classifier": "Random forest", "n_trees": 50}),
    ("Trainable Weka Segmentation 3D", "classes=[a,b] max_sigma=2 membrane_thickness=3", "segment3d.train_classifier", {"classes": "a,b", "max_sigma": 2}),
    # E2
    ("bUnwarpJ", "source_image=blobs.gif target_image=im2 registration=Accurate image_subsampling_factor=1 initial_deformation=[Very Coarse] final_deformation=[Super Fine] divergence_weight=0.1 "
     "curl_weight=0.2 landmark_weight=0 image_weight=1 consistency_weight=10 stop_threshold=0.01 save_transformations", "registration.bunwarpj",
     {"target": "@im2", "mode": "Accurate", "subsampling": 1, "initial_deformation": "Very Coarse", "final_deformation": "Super Fine", "divergence_weight": 0.1, "curl_weight": 0.2, "landmark_weight": 0,
      "consistency_weight": 10, "stop_threshold": 0.01, "output_field": True}),
    ("Grid/Collection stitching", "type=[Grid: snake by rows] order=[Right & Down] grid_size_x=3 grid_size_y=2 tile_overlap=20 first_file_index_i=1 directory=[/data/tiles] file_names=tile_{ii}.tif "
     "output_textfile_name=TileConfiguration.txt fusion_method=[Max. Intensity] regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 compute_overlap "
     "subpixel_accuracy computation_parameters=[Save computation time (but use more RAM)] image_output=[Write to disk] output_directory=[/data/out]", "registration.stitch_collection",
     {"layout": "Grid (folder)", "order": "Snake by rows", "grid_x": 3, "grid_y": 2, "overlap_percent": 20, "folder": "/data/tiles", "pattern": "tile_*.tif", "fusion": "Max intensity",
      "regression_threshold": 0.3, "max_avg_displacement": 2.5, "abs_displacement": 3.5, "compute_overlap": True, "subpixel": True, "output": "Write to file"}),
    ("Grid/Collection stitching", "type=[Positions from file] order=[Defined by TileConfiguration] directory=[/data/tiles] layout_file=TileConfiguration.txt fusion_method=[Linear Blending] "
     "image_output=[Fuse and display]", "registration.stitch_collection", {"layout": "Positions from file", "folder": "/data/tiles", "layout_file": "TileConfiguration.txt", "fusion": "Linear blending", "output": "Open image"}),
    ("Elastic Montaging", "layer_scale=0.5 resolution=32 minimal_pmcc_r=0.5 stiffness=0.8", "registration.elastic_montage", {"layout": "Open images", "downscale": 2, "min_correlation": 0.5, "stiffness": 0.8}),
    ("Landmark Correspondences", "source_image=blobs.gif template_image=im2 transformation_method=[Moving Least Squares (non-linear)] transformation_class=Rigid alpha=2 interpolate",
     "image.transform_landmark_correspondences", {"target": "@im2", "model": "Moving least squares (rigid)", "alpha": 2, "interpolation": "Linear"}),
    ("Landmark Correspondences", "source_image=blobs.gif template_image=im2 transformation_method=Perspective", "image.transform_landmark_correspondences", {"model": "Projective", "interpolation": "Nearest"}),
    ("Moving Least Squares", "source=[0,0; 10,0; 0,10] target=[1,1; 11,1; 1,11] model=Similarity alpha=1.5 create", "image.transform_mls",
     {"model": "Similarity", "source_points": "0,0; 10,0; 0,10", "target_points": "1,1; 11,1; 1,11", "alpha": 1.5, "create_new": True}),
]


@pytest.mark.parametrize("command,options,op_id,expected", _P5_CASES, ids=[f"{c[0]}#{i}" for i, c in enumerate(_P5_CASES)])
def test_phase5_macro_mappings(ij, command, options, op_id, expected):
    s = ij.session
    im2 = s.add_image(LumenImage(np.zeros((64, 64), np.uint8), name="im2"), activate=False)
    calls: list[tuple[str, dict]] = []
    ij._run = lambda op, params=None, **kw: (calls.append((op, dict(params or {}))) or {})  # resolve only: no op is executed
    ij.run(command, options)
    assert calls and calls[-1][0] == op_id, calls
    assert COMMANDS[command][1].split(" / ")[0].startswith(op_id.split(".")[0]) or op_id in COMMANDS[command][1]
    got = calls[-1][1]
    for k, v in expected.items():
        v = im2.id if v == "@im2" else v
        assert k in got, (k, got)
        assert str(got[k]) == str(v), (k, got[k], v)


def test_phase5_commands_run_on_synthetic_images(ij):
    s = ij.session
    assert "Overlay Options..." not in [n for n, (fn, lumen, _) in COMMANDS.items() if lumen == "(no-op)"] and COMMANDS["Labels..."][1] == "analyze.label"
    # --- MorphoLibJ label commands on a small label image
    lab = np.zeros((40, 40), np.uint8)
    lab[5:15, 5:15] = 1
    lab[20:35, 20:35] = 2
    lab[0:3, 30:40] = 3  # touches the border
    s.add_image(LumenImage(lab, name="labels.tif"))
    ij.run("Kill Borders")
    assert ij.img.current_plane()[1, 35] == 0 and ij.img.current_plane()[25, 25] == 2
    ij.run("Select Label(s)", "label(s)=2")
    assert ij.roi is not None and ij.roi.get("name") == "L2"
    ij.run("Select Label(s)", "label(s)=1,2")
    assert len(s.roi_manager.to_list()) >= 2
    n_before = len(s.images)
    ij.run("Labels To RGB", "colormap=Glasbey background=Black")
    assert len(s.images) == n_before + 1 and ij.img.rgb
    ij.selectWindow("labels.tif")
    ij.run("Select None")  # the Select Label(s) outline would confine the filters below
    ij.run("Remove Label(s)", "label(s)=1")
    assert not (ij.img.current_plane() == 1).any() and (ij.img.current_plane() == 2).sum() == 225
    ij.run("Morphological Filters", "operation=Dilation element=Square radius=1")
    assert (ij.img.current_plane() > 0).sum() > 225
    ij.run("Label Size Opening", "min_size=500")
    assert not ij.img.current_plane().any()
    ij.run("Fill Holes (Binary/Gray)")
    ij.run("Area Opening", "pixel=10")
    ij.run("Directional Filtering", "type=Max operation=Opening line=5 direction=4")
    # --- grey-level MorphoLibJ commands on blobs
    ij.selectWindow("blobs.gif")
    for cmd, opts in (("Regional Min & Max", "operation=[Regional Maxima] connectivity=8"), ("Extended Min & Max", "operation=[Extended Maxima] dynamic=20 connectivity=8"),
                      ("Chamfer Distance Map", "distances=[Borgefors (3,4)] output=[32 bits] normalize")):
        ij.selectWindow("blobs.gif")
        ij.run(cmd, opts)
    # --- G1: selection / overlay / drawing / text windows
    mask = np.zeros((32, 32), np.uint8)
    mask[8:16, 8:16] = 255
    s.add_image(LumenImage(mask, name="mask.tif"))
    ij.run("Points from Mask", "step=4 max_points=10")
    assert ij.roi is not None and ij.roi.get("type") in ("point", "multipoint")
    ij.run("Fit Circle to Image")
    assert ij.roi is not None and ij.roi.get("type") in ("oval", "ellipse", "circle")
    ij.roi = None
    before = ij.img.current_plane().copy()
    ij.run("Linear Gradient", "angle=0")
    assert not np.array_equal(before, ij.img.current_plane())
    ij.run("Radial Gradient", "radius=10")
    ij.selectWindow("blobs.gif")
    ij.makeRectangle(2, 2, 10, 10)
    ij.run("Add Selection...")
    ij.run("Labels...", "color=white font=12 show use")
    ij.run("Measure Overlay")
    assert s.results.rows and s.results.rows[-1]["Area"] == 100
    ij.run("Overlay Options...", "width=3 show use font=10 color=cyan")
    assert s.settings.get("overlay_stroke_width") == 3 and s.settings.get("overlay_use_names") is True
    ij.run("Text Window", "name=[Notes.txt] text=[first line]")
    assert s.tables["Notes.txt"]["rows"] == [["first line"]]
    ij.macro('print("[Notes.txt]", "second " + 2);')  # print("[window]", …) appends to an open text window
    assert s.tables["Notes.txt"]["rows"] == [["first line"], ["second 2"]]
    ij.run("Table...", "name=[T1] headings=[A,B,C]")
    assert s.tables["T1"]["columns"] == ["A", "B", "C"]
    # --- time lapse / StackReg on a small T stack
    t = np.arange(16, dtype=np.float32)
    frames = (128 + 100 * np.sin(2 * np.pi * t / 4.0))[:, None, None] * np.ones((1, 8, 8), np.float32)
    s.add_image(LumenImage(frames.astype(np.uint8), name="osc.tif", axes="TYX"))
    ij.run("Phase Map", "min_period=2 max_period=8")
    base = np.zeros((32, 32), np.uint8)
    base[8:20, 10:22] = 200
    st = np.stack([base, np.roll(base, 2, axis=1), np.roll(base, -1, axis=0)])
    stack = s.add_image(LumenImage(st, name="shift.tif", axes="TYX"))
    ij.run("StackReg", "transformation=Translation")
    assert ij.img.id == stack.id and ij.img.shape[-2:] == (32, 32)


def test_weka_statics_via_call(ij):
    s = ij.session
    assert ij.run("Trainable Weka Segmentation") is None  # no options: opens the Weka session for the call() statics
    ij.macro('call("trainableSegmentation.Weka_Segmentation.setMaximumSigma", "2.0");')  # call() goes through the transpiler
    ij.call("trainableSegmentation.Weka_Segmentation.changeClassName", "0", "background")
    ij.call("trainableSegmentation.Weka_Segmentation.changeClassName", "1", "blob")
    ij.call("trainableSegmentation.Weka_Segmentation.setFeature", "Gaussian_blur=true")
    ij.call("trainableSegmentation.Weka_Segmentation.setFeature", "Sobel_filter=true")
    ij.call("trainableSegmentation.Weka_Segmentation.setFeature", "Hessian=false")
    ij.call("trainableSegmentation.Weka_Segmentation.setClassifier", "weka.classifiers.trees.FastRandomForest", "-I 10 -K 0 -S 1")
    st = ij._weka_state()
    assert st["params"] == {"max_sigma": 2.0, "classifier": "Random forest", "n_trees": 10, "features_per_node": 0} and st["classes"] == ["background", "blob"]
    plane = ij.img.current_plane()
    ys, xs = np.nonzero(plane < np.percentile(plane, 10))
    ij.makeRectangle(int(xs[0]), int(ys[0]), 3, 3)
    ij.call("trainableSegmentation.Weka_Segmentation.addTrace", "0", "1")
    ys, xs = np.nonzero(plane > np.percentile(plane, 95))
    ij.makeRectangle(int(xs[0]), int(ys[0]), 3, 3)
    ij.call("trainableSegmentation.Weka_Segmentation.addTrace", "1", "1")
    names = [r["name"] for r in s.roi_manager.to_list()]
    assert names == ["background 1", "blob 2"]
    n = len(s.images)
    ij.call("trainableSegmentation.Weka_Segmentation.trainClassifier")
    assert len(s.images) == n  # apply_now=False
    ij.call("trainableSegmentation.Weka_Segmentation.getResult")
    assert len(s.images) == n + 1 and ij.img.name != "blobs.gif"
    with pytest.raises(ValueError, match="Unsupported macro function: call"):
        ij.call("ij.IJ.freeMemory")


def test_plugin_tools_run_from_imagej_macro_with_interaction_params():
    # run("Spot Click Counter", "x=10 y=12") promised by tools_ops: the registry fallback must pass the TOOL_PARAMS
    # (x, y, x2, y2, button, modifiers, event) through, not only spec.params (the spot used to land at (0, 0))
    s = Session()
    im = LumenImage(np.zeros((32, 32), np.uint8), name="a.tif")
    im.data[0, 0, 0, 12, 10] = 200
    s.add_image(im)
    ij = IJ(s)
    out = ij.run("Spot Click Counter", "x=10 y=12")
    assert "1 spot" in out["message"] and im.overlay[-1]["points"] == [[10.0, 12.0]]
    row = s.results.rows[-1]
    assert (row["X"], row["Y"], row["Value"]) == (10, 12, 200.0)
    # bare flags (shift / alt) arrive as True; a drag with event=drag-end measures through the macro runner too
    assert "1 spot" in ij.run("Spot Click Counter", "x=20 y=20 shift")["message"] and len(im.overlay) == 1
    ij.run("Spot Click Counter", "x=11 y=13 alt")
    assert im.overlay == []
    n = len(s.results.rows)
    ij.macro('run("Measure Distance Tool", "x=0 y=0 x2=3 y2=4 event=drag-end");')
    assert len(s.results.rows) == n + 1 and s.results.rows[-1]["Length"] == pytest.approx(5.0) and s.results.rows[-1]["Label"] == "Distance 1"
    # a tool's own params still work alongside, and a non-tool op fallback is unchanged
    ij.run("Spot Click Counter", "x=5 y=5 marker_size=9 color=#00ff00")
    assert im.overlay[-1]["point_size"] == 9 and im.overlay[-1]["points"] == [[5.0, 5.0]]
    before = im.data.copy()
    ij.run("filters.gaussian", "sigma=1 x=99")  # a non-tool op ignores stray interaction keys as before
    assert not np.array_equal(im.data, before)


# ---------------------------------------------------------------- macro tools: tool functions, precedence, Plugins ▸ New
def test_tool_macro_functions_precedence_and_cursor(ij):
    # ImageJ precedence: & | ^ << >> bind like * / %, so the tool-macro idiom `flags&16!=0` is (flags&16) != 0
    assert ij.macro("flags = 17; print(flags&16!=0); print(flags&2!=0); print(1|2==3); print(2*3&2); print(1<<2+1); print(3&1+1);") == "1\n0\n1\n2\n5\n2\n"
    # outside a tool interaction the cursor is unknown; IJ.set_cursor feeds the states a tool run hands out (the last repeats)
    assert ij.macro('getCursorLoc(x, y, z, f); print(x + " " + y + " " + z + " " + f);') == "-1 -1 0 0\n"
    ij.set_cursor([(3, 4, 1, 17), (8, 9, 1, 17), (8, 9, 1, 1)], {"shift"})
    out = ij.macro('n = 0; getCursorLoc(x, y, z, f); x0 = x; while (f&16!=0) { n++; getCursorLoc(x, y, z, f); } print(x0 + "->" + x + "," + y + " z=" + z + " n=" + n + " shift=" + isKeyDown("shift") + " f=" + f);')
    assert out == "3->8,9 z=1 n=2 shift=1 f=1\n"
    assert ij.getCursorLoc() == (8.0, 9.0, 1, 1)  # the last state stays
    ij.set_cursor([], set())
    assert ij.getCursorLoc() == (-1, -1, 0, 0) and not ij.isKeyDown("shift")
    # setKeyDown / isKeyDown, setTool / toolID / IJ.getToolName / call("ij.gui.Toolbar.…"), ij.Prefs.get/set, showStatus
    assert ij.macro('setKeyDown("shift alt"); print(isKeyDown("shift") + "" + isKeyDown("alt") + "" + isKeyDown("ctrl")); setKeyDown("none"); print(isKeyDown("alt"));') == "110\n0\n"
    assert ij.macro('print(toolID + " " + IJ.getToolName()); setTool("oval"); print(toolID + " " + IJ.getToolName() + " " + call("ij.gui.Toolbar.getToolId") + " " + call("ij.gui.Toolbar.getToolName"));'
                    'setTool(12); print(IJ.getToolName()); call("ij.gui.Toolbar.setTool", "line"); print(toolID);') == "0 rectangle\n1 oval 1 oval\nhand\n4\n"
    assert ij.macro('call("ij.Prefs.set", "my.key", 2.5); print(call("ij.Prefs.get", "my.key", 0)); print(call("ij.Prefs.get", "missing", "dflt")); showStatus("busy", 3);') == "2.5\ndflt\n"
    assert ij.session.settings["ij.prefs"] == {"my.key": "2.5"} and ij.status == "busy 3"
    with pytest.raises(ValueError, match="Unsupported macro function: call"):
        ij.macro('call("ij.IJ.freeMemory");')
    # a whole-file run skips the tool definitions (they only run from the tool rail) but runs the rest
    assert ij.macro('macro "X Tool - C000" { print("no"); } macro "X Tool Options" { print("no"); } macro "Y Action Tool" { print("no"); } print("yes"); macro "Plain" { print("plain"); }') == "yes\nplain\n"
    # a persistent namespace keeps var globals and functions between runs, rebinding the helpers to the running IJ
    env: dict = {}
    ij.macro("var hits = 0; function hit() { hits++; }", env=env)
    assert ij.macro("hit(); hit(); print(hits);", env=env) == "2\n" and ij.macro("print(hits + 1);", env=env) == "3\n" and env["IJ"] is ij


def test_plugins_new_commands_map_to_templates(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path / "home"))
    from lumen.core.ops import macro_tools as M
    from lumen.core.registry import REGISTRY
    saved = dict(REGISTRY.ops)
    try:
        s = Session()
        ij = IJ(s)
        assert "Macro Tool template" in ij.run("Macro Tool", "name=[Ring]")["message"] and REGISTRY.ops["macrotool.ring_tool"].tool
        assert "Plugin Frame template" in ij.run("New... ", "type=[Plugin Frame] name=Frm")["message"] and REGISTRY.ops["user.frm"].frame
        assert "Plugin Tool template" in ij.run("Plugin Tool", "name=Pt")["message"] and REGISTRY.ops["user.pt"].tool
        assert (tmp_path / "home" / "plugins" / "ring.ijm").is_file() and (tmp_path / "home" / "plugins" / "frm.py").is_file()
        assert ij.run("New...", "type=Table name=[T1] headings=[A,B]")["table"]["columns"] == ["A", "B"]
        with pytest.raises(ValueError, match="unknown template type"):
            ij.run("New...", "type=[Flux Capacitor]")
        rows = command_table()
        assert any('"Plugin Frame"' in r["command"] and r["lumen"] == "plugins.new_template" for r in rows)
    finally:
        for key in [k for k in M.MACRO_TOOL_FILES if str(tmp_path) in k]:
            M.unregister_file(key)
        REGISTRY.ops.clear()
        REGISTRY.ops.update(saved)


def test_run_particles_without_threshold_fails_like_imagej(ij):
    """run("Analyze Particles...") keeps ImageJ's rule: a thresholded or 8-bit binary image is required."""
    assert not ij.img.is_binary
    with pytest.raises(ValueError, match="thresholded or binary image"):
        ij.run("Analyze Particles...", "size=20-Infinity display clear")
    assert ij.nResults() == 0
    ij.setAutoThreshold("Otsu dark")
    ij.run("Analyze Particles...", "size=20-Infinity display clear")  # a threshold (without Convert to Mask) is enough
    assert ij.nResults() > 10
