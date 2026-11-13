"""Plugins ▸ Macros / Plugins ▸ Utilities (lumen/core/ops/plugins_ops.py): startup scripts, macro install / run / record,
plugin templates, hot reload, DICOM metadata, the drawing fills of File ▸ New and the command list / finder."""
import sys
from pathlib import Path

import numpy as np
import pytest

from lumen.core import ops as ops_pkg
from lumen.core.ops import plugins_ops as P
from lumen.core.ops.file_ops import FILLS, fill_pattern
from lumen.core.registry import REGISTRY
from lumen.core.session import Session


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated Lumen home ($LUMEN_HOME): nothing touches ~/.lumen, and the registry is put back afterwards."""
    h = tmp_path / "lumen_home"
    monkeypatch.setenv("LUMEN_HOME", str(h))
    monkeypatch.delenv("LUMEN_PLUGINS", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    saved = dict(REGISTRY.ops)
    saved_ops = {k: list(v) for k, v in ops_pkg.PLUGIN_OPS.items()}
    saved_err = dict(ops_pkg.PLUGIN_ERRORS)
    saved_mods = set(sys.modules)
    yield h
    REGISTRY.ops.clear()
    REGISTRY.ops.update(saved)
    ops_pkg.PLUGIN_OPS.clear()
    ops_pkg.PLUGIN_OPS.update(saved_ops)
    ops_pkg.PLUGIN_ERRORS.clear()
    ops_pkg.PLUGIN_ERRORS.update(saved_err)
    for m in [m for m in sys.modules if m.startswith("lumen_plugin_") and m not in saved_mods]:
        sys.modules.pop(m, None)


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


def _table(res, name):
    assert res["table"]["name"] == name
    return res["table"]


PLUGIN_V1 = '''from lumen.core.registry import op

@op("testreload.op", "Version 1", "Plugins/Test")
def version(plane, ctx):
    return plane
'''
PLUGIN_V2 = PLUGIN_V1.replace("Version 1", "Version 2").replace("return plane", "return 255 - plane")


# ---------------------------------------------------------------- menu placement
def test_menu_placement_and_kinds():
    menus = {oid: REGISTRY.ops[oid].to_dict()["menu"] for oid in REGISTRY.ops}
    for oid in ("macros.edit_startup", "macros.install", "macros.run_file", "macros.record_to_file"):
        assert menus[oid] == "Plugins/Macros", oid
    for oid in ("plugins.new_template", "plugins.open_folder", "plugins.list", "plugins.reload", "utilities.dicom_metadata",
                "utilities.list_commands", "utilities.command_finder"):
        assert menus[oid] == "Plugins/Utilities", oid
        assert REGISTRY.ops[oid].kind == "analysis"
    assert REGISTRY.ops["utilities.dicom_metadata"].requires_image is True
    assert all(REGISTRY.ops[o].requires_image is False for o in ("macros.install", "plugins.reload", "utilities.command_finder"))
    assert menus["frames.image_statistics"] == "Plugins/Frames" and REGISTRY.ops["frames.image_statistics"].frame
    assert P.TEMPLATE_KINDS[-2:] == ["Macro Tool", "Plugin Frame"]


# ---------------------------------------------------------------- startup scripts
def test_startup_scripts_run_on_a_fresh_session(home):
    assert P.lumen_home() == home and P.startup_paths() == [home / "startup.py", home / "startup.ijm"]
    assert P.run_startup_scripts(Session()) == []  # nothing to run
    home.mkdir()
    (home / "startup.py").write_text('lumen.run("file.sample", name="Blobs")\nlumen.log("py-start")\nprint("printed")\n')
    (home / "startup.ijm").write_text('newImage("fromijm", "8-bit black", 32, 16, 1);\nprint("ijm-start");\n')
    s = Session()
    res = P.run_startup_scripts(s)
    assert [r["ok"] for r in res] == [True, True] and [Path(r["path"]).name for r in res] == ["startup.py", "startup.ijm"]
    assert res[0]["output"] == "printed\n" and res[1]["output"].strip() == "ijm-start"
    assert [im.name for im in s.images.values()] == ["blobs.tif", "fromijm"]
    msgs = [e["msg"] for e in s.log]
    assert "py-start" in msgs and any(m.startswith("Startup script startup.py ran") for m in msgs)
    assert "printed" in msgs and "ijm-start" in msgs and any(m.startswith("Startup script startup.ijm ran") for m in msgs)


def test_startup_script_errors_are_logged_never_raised(home, capsys):
    home.mkdir()
    (home / "startup.py").write_text('lumen.log("before")\nraise RuntimeError("boom")\n')
    (home / "startup.ijm").write_text('noSuchMacroFunction(1);\n')
    s = Session()
    res = P.run_startup_scripts(s)
    assert [r["ok"] for r in res] == [False, False]
    assert "boom" in res[0]["error"] and "Unsupported" in res[1]["error"]
    errors = [e for e in s.log if e["level"] == "error"]
    assert len(errors) == 2 and "startup.py failed" in errors[0]["msg"] and "boom" in errors[0]["msg"]
    assert "startup.ijm failed" in errors[1]["msg"]
    assert "before" in [e["msg"] for e in s.log]  # the lines before the error ran
    # the launcher hook reports the failures on stderr and never raises
    from lumen.__main__ import _run_startup_scripts
    _run_startup_scripts(Session())
    err = capsys.readouterr().err
    assert "startup.py failed" in err and "boom" in err and "startup.ijm failed" in err


def test_edit_startup_creates_a_documented_file_and_lists_it(home, session):
    res = session.run_op("macros.edit_startup")
    py = home / "startup.py"
    assert py.is_file() and str(py) in res["message"] and "0 code line(s)" in res["message"]
    t = _table(res, "startup.py")
    assert t["columns"] == ["Line", "Text"] and [r[1] for r in t["rows"]] == py.read_text().splitlines()
    assert all(ln.startswith("#") for ln in py.read_text().splitlines() if ln.strip())
    py.write_text('lumen.log("x")\n# comment\nlumen.log("y")\n')
    res = session.run_op("macros.edit_startup", {"create": False})
    assert "2 code line(s)" in res["message"] and len(res["table"]["rows"]) == 3 and 'lumen.log("y")' in res["log"]
    res = session.run_op("macros.edit_startup", {"kind": "ImageJ macro (startup.ijm)"})
    ijm = home / "startup.ijm"
    assert ijm.is_file() and all(ln.startswith("//") for ln in ijm.read_text().splitlines() if ln.strip())
    assert res["table"]["name"] == "startup.ijm" and str(ijm) in res["message"]
    # the created templates are inert: running them does nothing and fails nothing
    assert all(r["ok"] for r in P.run_startup_scripts(Session())) and len(P.run_startup_scripts(Session())) == 2


# ---------------------------------------------------------------- templates
@pytest.mark.parametrize("kind", P.TEMPLATE_KINDS)
def test_new_template_is_written_importable_and_runs(home, session, kind):
    events = []
    session.listeners.append(events.append)
    res = session.run_op("plugins.new_template", {"kind": kind, "name": "My Thing"})
    pdir = home / "plugins"
    t = res["table"]
    assert t["columns"] == ["Line", "Text"] and len(t["rows"]) > 10 and str(pdir) in res["message"]
    assert "command registered" in res["message"] and session.settings["ops_version"] == 1
    assert any(e["type"] == "ops" and e["ops_version"] == 1 for e in events)
    if kind == "Macro Tool":
        # an ImageJ tool macro set: no wrapper module, the "… Tool" block is a tool-rail tool run through the registry
        assert (pdir / "my_thing.ijm").is_file() and not (pdir / "my_thing_macro.py").exists() and t["name"] == "my_thing.ijm"
        oid = "macrotool.my_thing_tool"
        spec = REGISTRY.ops[oid]
        assert spec.name == "My Thing Tool" and spec.tool and spec.to_dict()["menu"] == "Plugins/Tools" and spec.tool_icon == "◯" and spec.tool_color == "#00cc00"
        assert [p.name for p in spec.params] == ["options"] and ops_pkg.PLUGIN_OPS.get(str(pdir / "my_thing.ijm")) is None
        out = session.run_op(oid, {"x": 20, "y": 30, "event": "click"})
        assert out["message"] == "My Thing: circle 1 at (20, 30) r=12" and session.results.rows[-1]["Radius"] == 12
        assert session.run_op(oid, {"x": 20, "y": 30, "x2": 24, "y2": 33, "event": "drag-end"})["message"] == "My Thing: circle 2 at (20, 30) r=5"
        assert session.run_op(oid, {"options": True})["message"] == "My Thing: radius 12, counter reset"
        with pytest.raises(FileExistsError):
            session.run_op("plugins.new_template", {"kind": kind, "name": "My Thing"})
        session.run_op("plugins.new_template", {"kind": kind, "name": "My Thing", "overwrite": True})
        assert oid in REGISTRY.ops
        return
    if kind == "ImageJ macro":
        assert (pdir / "my_thing.ijm").is_file() and (pdir / "my_thing_macro.py").is_file()
        assert t["name"] == "my_thing.ijm" and [r[1] for r in t["rows"]] == (pdir / "my_thing.ijm").read_text().splitlines()
        oid = "macros.my_thing"
        spec = REGISTRY.ops[oid]
        assert spec.name == "My Thing" and spec.to_dict()["menu"] == "Plugins/Macros" and spec.func.__module__ == "lumen_plugin_my_thing_macro"
        out = session.run_op(oid)
        assert "My Thing:" in out["log"] and "particles" in out["log"] and len(session.results.rows) > 0
        assert ops_pkg.PLUGIN_OPS[str(pdir / "my_thing_macro.py")] == [oid]
        return
    path = pdir / "my_thing.py"
    assert path.is_file() and t["name"] == "my_thing.py" and [r[1] for r in t["rows"]] == path.read_text().splitlines()
    oid = "user.my_thing"
    spec = REGISTRY.ops[oid]
    assert spec.name == "My Thing" and spec.func.__module__ == "lumen_plugin_my_thing" and ops_pkg.PLUGIN_OPS[str(path)] == [oid]
    assert path.read_text().startswith('"""My Thing') and "Reload Plugins" in path.read_text()
    before = session.active.data.copy()
    if kind == "Python command":
        assert spec.kind == "image"
        out = session.run_op(oid, {"offset": 5})
        assert len(out["new_images"]) == 1 and session.active.name == "blobs.tif+5"
        np.testing.assert_array_equal(session.active.data, np.clip(before.astype(np.int32) + 5, 0, 255).astype(np.uint8))
        session.select(next(iter(session.images)))
        session.run_op(oid, {"offset": 5, "as_new": False})
        assert session.active.dtype == np.uint8 and int(session.active.data.max()) == 255 and (session.active.data != before).any()
    elif kind == "Python filter (plane)":
        assert spec.kind == "plane" and spec.preview
        session.run_op(oid, {"sigma": 1.5})
        assert session.active.dtype == np.uint8 and (session.active.data != before).any()
        assert int(session.active.data.min()) == 0 and int(session.active.data.max()) == 255  # stretched
    elif kind == "Plugin Tool":
        # a tool-rail tool: tool fields in the registry / API, the interaction params reach the function, clicks mark + count
        assert spec.kind == "analysis" and spec.tool and spec.to_dict()["tool"] is True and spec.to_dict()["tool_icon"] == "✚" and spec.tool_cursor == "crosshair"
        assert [p.name for p in spec.params] == ["radius", "color"]
        img = session.active
        out = session.run_op(oid, {"x": 20, "y": 30, "event": "click"})
        assert out["results_added"] == 1 and "1 mark" in out["message"] and len(img.overlay) == 1
        mark = img.overlay[0]
        assert mark["type"] == "oval" and mark["id"] == "my_thing-1" and mark["x"] == 12 and mark["w"] == 16 and mark["position"] == {"t": 0, "z": 0, "c": 0}
        assert session.results.rows[-1]["Value"] == float(before[0, 0, 0, 30, 20]) and session.results.rows[-1]["Label"] == "Mark 1"
        assert session.run_op(oid, {"x": 5, "y": 5, "event": "drag-start"}).get("message") is None and len(img.overlay) == 1
        session.run_op(oid, {"x": 10, "y": 10, "x2": 30, "y2": 50, "event": "drag-end", "radius": 4})  # the midpoint of the drag
        assert len(img.overlay) == 2 and img.overlay[1]["x"] == 16 and img.overlay[1]["y"] == 26 and img.overlay[1]["w"] == 8
        out = session.run_op(oid, {"x": 21, "y": 31, "alt": True})
        assert "Removed Mark 1" in out["message"] and [o["id"] for o in img.overlay] == ["my_thing-2"]
        assert session.macro[-1].startswith(f'lumen.run("{oid}", radius=8.0, color=\'#4ade80\', x=21.0, y=31.0, button=0, shift=False, alt=True')
        np.testing.assert_array_equal(session.active.data, before)  # analysis never changes pixels
    elif kind == "Plugin Frame":
        # a plugin frame: kind="frame" → an analysis op with the frame flag (own sub-panel in the Plugin Frames inspector section)
        assert spec.kind == "analysis" and spec.frame and spec.to_dict()["frame"] is True and spec.to_dict()["preview"] is False and spec.requires_image is False
        assert spec.to_dict()["menu"] == "Plugins/Frames" and [p.name for p in spec.params] == ["decimals", "plot_means"]
        out = session.run_op(oid, {"decimals": 1})
        assert out["table"]["name"] == "My Thing" and out["table"]["rows"][0][0] == "blobs.tif" and out["table"]["rows"][0][4] == round(float(before.mean()), 1)
        assert out["plot"]["series"][0]["values"] == [out["table"]["rows"][0][4]] and out["message"] == "1 open image(s)"
        assert session.run_op(oid, {"plot_means": False}).get("plot") is None
        np.testing.assert_array_equal(session.active.data, before)  # analysis never changes pixels
    else:
        assert spec.kind == "analysis" and {p.type for p in spec.params} == {"string", "float", "int", "choice", "bool", "text"}
        out = session.run_op(oid, {"method": "Max", "label": "blobs"})
        assert out["results_added"] == 1 and session.results.rows[-1]["Max"] == float(before.max()) and session.results.rows[-1]["Label"] == "blobs"
        assert "My Thing: 1 row(s), 3 iteration(s)" in out["log"]
        np.testing.assert_array_equal(session.active.data, before)  # analysis never changes pixels
    # names must stay unique, unless overwriting is explicitly allowed
    with pytest.raises(FileExistsError):
        session.run_op("plugins.new_template", {"kind": kind, "name": "My Thing"})
    session.run_op("plugins.new_template", {"kind": kind, "name": "My Thing", "overwrite": True})
    assert oid in REGISTRY.ops


def test_slug_avoids_keywords_builtins_and_template_names():
    assert P._slug("My Thing") == "my_thing" and P._slug("3D view") == "m_3d_view" and P._slug("") == "macro" and P._slug("***") == "macro"
    for name, slug in (("Pass", "pass_"), ("Class", "class_"), ("Import", "import_"), ("Return", "return_"), ("Lambda", "lambda_"),
                       ("Yield", "yield_"), ("Global", "global_"), ("With", "with_"), ("Float", "float_"), ("Int", "int_"), ("Len", "len_"),
                       ("Str", "str_"), ("Min", "min_"), ("List", "list_"), ("Type", "type_"), ("Open", "open_"), ("Filter", "filter_"),
                       ("np", "np_"), ("NDI", "ndi_"), ("math", "math_"), ("op", "op_"), ("cast_to", "cast_to_")):
        assert P._slug(name) == slug, name
        assert slug.isidentifier() and not __import__("keyword").iskeyword(slug)


@pytest.mark.parametrize("name,kind,oid", [("Pass", "Python command", "user.pass_"), ("np", "Python command", "user.np_"),
                                           ("Class", "Python filter (plane)", "user.class_"), ("Float", "Script with dialog", "user.float_"),
                                           ("Lambda", "ImageJ macro", "macros.lambda_"), ("Return", "Plugin Tool", "user.return_")])
def test_new_template_with_keyword_or_builtin_name_loads_and_runs(home, session, name, kind, oid):
    """``def pass(ctx)`` is a SyntaxError that used to leave an unloadable file in the plugins folder (failing on every
    reload); builtins / template imports as names shadowed what the generated body calls."""
    res = session.run_op("plugins.new_template", {"kind": kind, "name": name})
    assert "command registered" in res["message"] and oid in REGISTRY.ops and REGISTRY.ops[oid].name == name
    assert not ops_pkg.PLUGIN_ERRORS
    pdir = home / "plugins"
    slug = oid.split(".", 1)[1]
    assert (pdir / (f"{slug}.ijm" if kind == "ImageJ macro" else f"{slug}.py")).is_file()
    before = session.active.data.copy()
    if kind == "Python command":
        session.run_op(oid, {"offset": 5})
        assert session.active.name == "blobs.tif+5"
    elif kind == "Python filter (plane)":
        session.run_op(oid, {"sigma": 1.5})
        assert (session.active.data != before).any()
    elif kind == "Script with dialog":
        assert session.run_op(oid, {"method": "Max", "label": "blobs"})["results_added"] == 1
    elif kind == "ImageJ macro":
        assert "particles" in session.run_op(oid)["log"]
    else:
        assert session.run_op(oid, {"x": 20, "y": 30, "event": "click"})["results_added"] == 1
        assert session.active.overlay[0]["id"] == f"{slug}-1"
    out = session.run_op("plugins.reload")
    assert oid in REGISTRY.ops and not ops_pkg.PLUGIN_ERRORS and "failed" not in out["message"]


def test_install_keyword_named_macro_and_script(home, session, tmp_path):
    ijm = tmp_path / "pass.ijm"
    ijm.write_text('print("macro ok");\n')
    res = session.run_op("macros.install", {"path": str(ijm)})
    assert res["table"]["rows"] == [["macros.pass_", "pass", "Plugins/Macros"]] and (home / "plugins" / "pass_macro.py").is_file()
    assert session.run_op("macros.pass_")["log"].strip() == "macro ok"
    script = tmp_path / "class.py"
    script.write_text('print("script ok")\n')
    res = session.run_op("macros.install", {"path": str(script)})
    assert res["table"]["rows"] == [["macros.class_", "class", "Plugins/Macros"]] and (home / "plugins" / "class_script.py").is_file()
    assert session.run_op("macros.class_")["log"].strip() == "script ok"
    session.run_op("plugins.reload")
    assert not ops_pkg.PLUGIN_ERRORS and "macros.pass_" in REGISTRY.ops and "macros.class_" in REGISTRY.ops


def test_new_template_without_loading_registers_on_reload(home, session):
    res = session.run_op("plugins.new_template", {"kind": "Python filter (plane)", "name": "Later", "load": False})
    assert "reload plugins" in res["message"] and (home / "plugins" / "later.py").is_file() and "user.later" not in REGISTRY.ops
    session.run_op("plugins.reload")
    assert "user.later" in REGISTRY.ops


# ---------------------------------------------------------------- install
def test_install_ijm_creates_a_working_command_on_blobs(home, session, tmp_path):
    src = tmp_path / "src" / "my blur.ijm"
    src.parent.mkdir()
    src.write_text('run("Gaussian Blur...", "sigma=3");\nprint("blurred " + getTitle());\n')
    events = []
    session.listeners.append(events.append)
    res = session.run_op("macros.install", {"path": str(src)})
    pdir = home / "plugins"
    assert (pdir / "my blur.ijm").read_text() == src.read_text() and (pdir / "my blur_macro.py").is_file()
    assert _table(res, "Installed")["rows"] == [["macros.my_blur", "my blur", "Plugins/Macros"]]
    assert "Installed ImageJ macro my blur.ijm" in res["message"]
    assert any(e["type"] == "ops" for e in events) and session.settings["ops_version"] == 1
    spec = REGISTRY.ops["macros.my_blur"]
    assert spec.kind == "analysis" and spec.requires_image is False and spec.func.__module__ == "lumen_plugin_my blur_macro"
    before = session.active.data.copy()
    out = session.run_op("macros.my_blur")
    assert out["log"].strip() == "blurred blobs.tif" and (session.active.data != before).any()
    assert session.macro[-1] == 'lumen.run("macros.my_blur")'
    # the installed .ijm is re-read on every run (edit it without reloading)
    (pdir / "my blur.ijm").write_text('makeRectangle(2, 3, 10, 12);\nprint("edited");\n')
    out = session.run_op("macros.my_blur")
    assert out["log"].strip() == "edited" and out["roi"]["type"] == "rect" and out["roi"]["x"] == 2
    # the plugin list knows both files and the command
    rows = {r[0]: r for r in session.run_op("plugins.list")["table"]["rows"]}
    assert rows["my blur.ijm"][2] == "ImageJ macro" and rows["my blur.ijm"][5] == "macros.my_blur"
    assert rows["my blur_macro.py"][5] == "macros.my_blur" and rows["my blur_macro.py"][6].startswith("loaded")
    # custom name + menu, and the registry survives a full reload
    res = session.run_op("macros.install", {"path": str(src), "name": "Blur It", "menu": "Plugins/Mine"})
    assert REGISTRY.ops["macros.my_blur"].name == "Blur It" and REGISTRY.ops["macros.my_blur"].to_dict()["menu"] == "Plugins/Mine"
    session.run_op("plugins.reload")
    assert REGISTRY.ops["macros.my_blur"].name == "Blur It"


def test_install_python_plugin_module_and_plain_script(home, session, tmp_path):
    mod = tmp_path / "inverter.py"
    mod.write_text('from lumen.core.registry import op\n\n@op("testinstall.invert", "Test Invert", "Plugins/Test")\ndef inv(plane, ctx):\n    return 255 - plane\n')
    res = session.run_op("macros.install", {"path": str(mod)})
    assert (home / "plugins" / "inverter.py").read_text() == mod.read_text() and "plugin module" in res["message"]
    assert res["table"]["rows"] == [["testinstall.invert", "Test Invert", "Plugins/Test"]]
    before = session.active.data.copy()
    session.run_op("testinstall.invert")
    np.testing.assert_array_equal(session.active.data, 255 - before)
    # a plain script (no @op) is wrapped into a command and kept under <home>/macros
    script = tmp_path / "smooth_it.py"
    script.write_text('lumen.run("filters.gaussian", sigma=2)\nprint("script ok", lumen.image.name)\n')
    res = session.run_op("macros.install", {"path": str(script)})
    assert (home / "macros" / "smooth_it.py").is_file() and (home / "plugins" / "smooth_it_script.py").is_file()
    assert res["table"]["rows"] == [["macros.smooth_it", "smooth it", "Plugins/Macros"]] and "Python script" in res["message"]
    before = session.active.data.copy()
    out = session.run_op("macros.smooth_it")
    assert out["log"].strip() == "script ok blobs.tif" and (session.active.data != before).any()
    # errors
    with pytest.raises(FileNotFoundError):
        session.run_op("macros.install", {"path": str(tmp_path / "missing.ijm")})
    (tmp_path / "x.java").write_text("class X {}")
    with pytest.raises(ValueError, match="only .py and .ijm"):
        session.run_op("macros.install", {"path": str(tmp_path / "x.java")})
    with pytest.raises(ValueError, match="Choose"):
        session.run_op("macros.install", {"path": ""})
    (tmp_path / "broken.py").write_text("from lumen.core.registry import op\nthis is not python\n")
    with pytest.raises(ValueError, match="broken.py failed to load"):
        session.run_op("macros.install", {"path": str(tmp_path / "broken.py")})


# ---------------------------------------------------------------- hot reload
def test_reload_picks_up_changed_added_and_deleted_plugins(home, session):
    pdir = home / "plugins"
    pdir.mkdir(parents=True)
    f = pdir / "reloadme.py"
    f.write_text(PLUGIN_V1)
    events = []
    session.listeners.append(events.append)
    res = session.run_op("plugins.reload")
    t = _table(res, "Reloaded plugins")
    assert t["columns"] == ["File", "Folder", "Status", "Commands"] and t["rows"] == [["reloadme.py", str(pdir), "loaded", "testreload.op"]]
    assert "Reloaded 1 plugin file(s): 1 command(s) now, 0 replaced/removed" in res["message"]
    assert REGISTRY.ops["testreload.op"].name == "Version 1" and ops_pkg.PLUGIN_OPS[str(f)] == ["testreload.op"]
    assert session.settings["ops_version"] == 1 and [e for e in events if e["type"] == "ops"][-1]["ops_version"] == 1
    before = session.active.data.copy()
    session.run_op("testreload.op")
    np.testing.assert_array_equal(session.active.data, before)
    # edit the file: the command is replaced (name and behaviour), no duplicate-id error
    f.write_text(PLUGIN_V2)
    res = session.run_op("plugins.reload")
    assert "1 replaced/removed" in res["message"] and REGISTRY.ops["testreload.op"].name == "Version 2"
    session.run_op("testreload.op")
    np.testing.assert_array_equal(session.active.data, 255 - before)
    assert session.settings["ops_version"] == 2
    # a second, broken file is reported without hurting the good one
    (pdir / "broken.py").write_text("from lumen.core.registry import op\n@op('testreload.broken', 'Broken', 'Plugins/Test')\ndef b(plane, ctx): return plane\nraise ImportError('nope')\n")
    res = session.run_op("plugins.reload")
    rows = {r[0]: r for r in res["table"]["rows"]}
    assert rows["reloadme.py"][2] == "loaded" and rows["broken.py"][2].startswith("error: ImportError: nope")
    assert "1 file(s) failed" in res["message"] and "testreload.op" in REGISTRY.ops
    assert ops_pkg.PLUGIN_ERRORS[str(pdir / "broken.py")].startswith("ImportError")
    assert ops_pkg.PLUGIN_OPS[str(pdir / "broken.py")] == ["testreload.broken"]  # registered before the failure → tracked
    # deleting files removes their commands
    f.unlink()
    (pdir / "broken.py").unlink()
    res = session.run_op("plugins.reload")
    assert "testreload.op" not in REGISTRY.ops and "testreload.broken" not in REGISTRY.ops and res["table"]["rows"] == []
    assert "0 command(s) now, 2 replaced/removed" in res["message"]
    # direct API
    f.write_text(PLUGIN_V1)
    info = ops_pkg.reload_user_plugins()
    assert info["files"] == [str(f)] and info["ops"] == {str(f): ["testreload.op"]} and info["errors"] == {} and info["removed"] == []
    assert ops_pkg.load_plugin_file(f) == ["testreload.op"]  # re-importing one file replaces its own commands


def test_reload_honours_lumen_plugins_env(home, session, tmp_path, monkeypatch):
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "fromenv.py").write_text(PLUGIN_V1.replace("testreload.op", "testreload.env"))
    monkeypatch.setenv("LUMEN_PLUGINS", str(extra))
    assert ops_pkg.plugin_dirs() == [home / "plugins", extra]
    session.run_op("plugins.reload")
    assert "testreload.env" in REGISTRY.ops
    rows = {r[0]: r for r in session.run_op("plugins.list")["table"]["rows"]}
    assert rows["fromenv.py"][1] == str(extra) and rows["fromenv.py"][5] == "testreload.env"


# ---------------------------------------------------------------- run file / record
def test_run_file_python_and_macro(home, session, tmp_path):
    py = tmp_path / "s.py"
    py.write_text('lumen.run("filters.gaussian", sigma=1)\nprint("hi", len(lumen.images))\n')
    before = session.active.data.copy()
    out = session.run_op("macros.run_file", {"path": str(py)})
    assert out["log"] == "hi 1\n" and "s.py finished" in out["message"] and (session.active.data != before).any()
    ijm = tmp_path / "m.ijm"
    ijm.write_text('makeRectangle(1, 2, 5, 6);\nrun("Measure");\nprint("n=" + nResults);\n')
    out = session.run_op("macros.run_file", {"path": str(ijm)})
    assert out["log"].strip() == "n=1" and out["roi"]["w"] == 5 and len(session.results.rows) == 1
    txt = tmp_path / "m.txt"
    txt.write_text('print("txt macro");\n')
    assert session.run_op("macros.run_file", {"path": str(txt)})["log"].strip() == "txt macro"
    bad = tmp_path / "bad.py"
    bad.write_text('print("start")\nraise ValueError("script boom")\n')
    with pytest.raises(ValueError, match="bad.py: ValueError: script boom"):
        session.run_op("macros.run_file", {"path": str(bad)})
    assert any("script boom" in e["msg"] and e["level"] == "error" for e in session.log)  # traceback in the Log
    with pytest.raises(ValueError, match="only .py"):
        session.run_op("macros.run_file", {"path": str(tmp_path / "x.java") if (tmp_path / "x.java").write_text("x") else ""})
    with pytest.raises(FileNotFoundError):
        session.run_op("macros.run_file", {"path": str(tmp_path / "nope.py")})


def test_record_to_file(home, session, tmp_path):
    session.run_op("filters.gaussian", {"sigma": 2})
    res = session.run_op("macros.record_to_file", {"title": "My recording"})
    p = Path(res["message"].split(" to ", 1)[1])
    assert p.parent == home / "macros" and p.name.startswith("Recorded_") and p.suffix == ".py"
    text = p.read_text()
    assert text.startswith("# My recording\n# Recorded by Lumen") and 'lumen.run("file.sample", name=\'Blobs\')' in text
    assert 'lumen.run("filters.gaussian"' in text and "record_to_file" not in text
    assert _table(res, p.name)["rows"][-1][1] == text.splitlines()[-1]
    # explicit path without a suffix, clearing the recorder; the file replays
    res = session.run_op("macros.record_to_file", {"path": str(tmp_path / "again"), "clear_recorder": True})
    assert (tmp_path / "again.py").is_file() and len(session.macro) == 1 and "record_to_file" in session.macro[0]  # only its own line is left
    s2 = Session()
    P.run_python_file(s2, tmp_path / "again.py")
    assert [im.name for im in s2.images.values()] == ["blobs.tif"]
    with pytest.raises(ValueError, match="recorder is empty"):
        session.run_op("macros.record_to_file")


# ---------------------------------------------------------------- File ▸ New drawing fills
@pytest.mark.parametrize("type_,full", [("8-bit", 255), ("16-bit", 65535), ("32-bit", 1.0)])
def test_file_new_gradient_fills(type_, full):
    s = Session()
    assert FILLS == ["Black", "White", "Ramp", "Linear gradient", "Radial gradient", "Checkerboard", "Sine grating", "Noise"]
    assert next(p for p in REGISTRY.ops["file.new"].params if p.name == "fill").choices == FILLS
    W, H = 64, 32

    def new(fill, period=8):
        s.run_op("file.new", {"type": type_, "width": W, "height": H, "fill": fill, "period": period})
        a = s.active.data[0, 0, 0]
        assert a.shape == (H, W) and s.active.dtype == {"8-bit": np.uint8, "16-bit": np.uint16, "32-bit": np.float32}[type_]
        return a.astype(np.float64)

    lin = new("Linear gradient")
    assert lin[0, 0] == 0 and lin[-1, -1] == full and np.all(np.diff(lin, axis=1) >= 0) and np.all(np.diff(lin, axis=0) >= 0)
    assert abs(lin[H // 2, W // 2] - full / 2) <= full * 0.03
    rad = new("Radial gradient")
    assert rad[0, 0] == 0 and rad[-1, -1] == 0 and rad[H // 2, W // 2] == rad.max() >= 0.97 * full
    assert rad[H // 2, W // 2] > rad[0, W // 2] > rad[0, 0]
    chk = new("Checkerboard", 8)
    assert set(np.unique(chk)) == {0.0, float(full)}
    assert chk[0, 0] == 0 and chk[0, 8] == full and chk[8, 0] == full and chk[8, 8] == 0 and np.all(chk[:8, :8] == 0)
    sin = new("Sine grating", 16)
    np.testing.assert_allclose(sin[:, 0], full / 2, atol=max(1, full * 0.005))
    assert sin[0, 4] == full and sin[0, 12] == 0 and np.array_equal(sin[:, 0], sin[:, 16]) and np.array_equal(sin[:, 4], sin[:, 20])
    assert abs(sin.mean() - full / 2) <= full * 0.01
    ramp = new("Ramp")
    assert ramp[0, 0] == 0 and ramp[0, -1] == full and np.array_equal(ramp[0], ramp[-1])
    line = f'lumen.run("file.new", title=\'Untitled\', type={type_!r}, width=64, height=32, slices=1, channels=1, frames=1, fill=\'Ramp\', period=8)'
    assert s.macro[-1].startswith(line) and s.macro[-1].endswith('  # → "Untitled-4"')  # the recorder names the image it created


def test_fill_pattern_edge_cases_and_rgb():
    assert fill_pattern("Checkerboard", 1, 1, 255, 4).shape == (1, 1)
    assert fill_pattern("Radial gradient", 1, 5, 255).max() == 255
    assert fill_pattern("Sine grating", 3, 2, 255, 0).shape == (2, 3)  # period clamps to 1
    with pytest.raises(ValueError):
        fill_pattern("Plaid", 4, 4)
    s = Session()
    s.run_op("file.new", {"type": "RGB", "width": 16, "height": 8, "fill": "Checkerboard", "period": 4})
    img = s.active
    assert img.rgb and img.nc == 3 and np.array_equal(img.data[0, 0, 0], img.data[0, 0, 2]) and int(img.data.max()) == 255
    # the ImageJ newImage() mapping is unaffected
    from lumen.core.ij_compat import IJ
    IJ(s).macro('newImage("x", "8-bit ramp", 10, 4, 1);')
    assert s.active.name == "x" and int(s.active.data[0, 0, 0, 0, -1]) == 255


# ---------------------------------------------------------------- command list / finder
def test_list_commands_and_command_finder(session):
    res = session.run_op("utilities.list_commands")
    t = _table(res, "Commands")
    assert t["columns"] == ["Id", "Name", "Menu", "Kind", "Shortcut", "Parameters", "Description"] and len(t["rows"]) == len(REGISTRY.ops)
    row = next(r for r in t["rows"] if r[0] == "filters.gaussian")
    assert row[1] == "Gaussian Blur" and row[2] == "Process/Filters" and row[3] == "plane" and "sigma: float = " in row[5]
    menus = [r[2].lower() for r in t["rows"]]
    assert menus == sorted(menus)
    sub = session.run_op("utilities.list_commands", {"menu": "process/filters", "kind": "plane"})["table"]["rows"]
    assert sub and all(r[2] == "Process/Filters" and r[3] == "plane" for r in sub) and f"{len(sub)} of {len(REGISTRY.ops)}" in session.run_op("utilities.list_commands", {"menu": "process/filters", "kind": "plane"})["message"]
    res = session.run_op("utilities.command_finder", {"query": "gaussian blur"})
    t = _table(res, "Command Finder")
    assert t["columns"] == ["Id", "Name", "Menu", "Shortcut", "Score", "Description"] and t["rows"][0][0] == "filters.gaussian"
    assert "match 'gaussian blur'" in res["message"]
    for q in ("gausian", "gausblur", "gaus blr", "GAUSSIAN"):  # typos, character subsequences, partial words, case
        ids = [r[0] for r in session.run_op("utilities.command_finder", {"query": q, "max_results": 5})["table"]["rows"]]
        assert "filters.gaussian" in ids, q
    assert [r[0] for r in session.run_op("utilities.command_finder", {"query": "particles"})["table"]["rows"]][0].startswith("analyze.")
    res = session.run_op("utilities.command_finder", {"query": "zzqqxxyy"})
    assert res["table"]["rows"] == [] and res["message"].startswith("No command")
    assert len(session.run_op("utilities.command_finder", {"query": "image", "max_results": 3})["table"]["rows"]) == 3
    with pytest.raises(ValueError):
        session.run_op("utilities.command_finder", {"query": "  "})
    assert P.fuzzy_score(REGISTRY.ops["filters.gaussian"], "gaussian") > P.fuzzy_score(REGISTRY.ops["filters.gaussian3d"], "gaussian")  # shorter name wins the tie
    assert P.fuzzy_score(REGISTRY.ops["filters.gaussian"], "gaussian zzz") == 0  # every word must match


# ---------------------------------------------------------------- plugins folder
def test_open_folder_and_list(home, session):
    res = session.run_op("plugins.open_folder")
    pdir = home / "plugins"
    assert pdir.is_dir() and str(pdir) in res["message"] and res["table"]["rows"] == []
    session.run_op("plugins.new_template", {"kind": "Python filter (plane)", "name": "Listed"})
    res = session.run_op("plugins.open_folder")
    rows = _table(res, "Plugins folder")["rows"]
    assert [r[0] for r in rows] == ["listed.py"] and rows[0][2] == "plugin" and rows[0][5] == "user.listed" and rows[0][6] == "loaded (1 command(s))"
    res = session.run_op("plugins.list")
    assert "1 plugin file(s) in 1 folder(s), 1 plugin command(s)" in res["message"]


# ---------------------------------------------------------------- DICOM metadata
def _write_dicom(path, rows=8, cols=10, value=100):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.PatientName, ds.StudyDescription = "CT", "Test^Patient", "Phantom study"
    ds.Rows, ds.Columns = rows, cols
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelSpacing, ds.SliceThickness = [0.7, 0.8], 2.5
    seq_item = Dataset()
    seq_item.CodeValue, seq_item.CodeMeaning = "T-A0100", "Brain"
    ds.AnatomicRegionSequence = [seq_item]
    ds.add_new((0x0009, 0x0010), "LO", "PRIVATE CREATOR")
    ds.PixelData = np.full((rows, cols), value, np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)


def test_dicom_metadata_table(session, tmp_path):
    with pytest.raises(ValueError, match="not opened from a DICOM"):
        session.run_op("utilities.dicom_metadata")
    f = tmp_path / "img.dcm"
    _write_dicom(f)
    session.open_file(str(f))
    res = session.run_op("utilities.dicom_metadata")
    t = _table(res, "DICOM")
    assert t["columns"] == ["Tag", "Name", "VR", "Value", "Keyword"]
    by_kw = {r[4]: r for r in t["rows"] if r[4]}
    assert by_kw["PatientName"][3] == "Test^Patient" and by_kw["PatientName"][0] == "0010,0010" and by_kw["PatientName"][2] == "PN"
    assert by_kw["Modality"][3] == "CT" and by_kw["PixelSpacing"][3] == "0.7, 0.8" and by_kw["TransferSyntaxUID"][3].startswith("1.2.840")
    assert "PixelData" not in by_kw
    assert by_kw["AnatomicRegionSequence"][2] == "SQ" and by_kw["CodeMeaning"][1] == "    Code Meaning" and by_kw["CodeMeaning"][3] == "Brain"
    assert any(r[0] == "0009,0010" and r[3] == "PRIVATE CREATOR" for r in t["rows"])
    assert "img.dcm" in res["message"] and "CT" in res["message"] and "Test^Patient" in res["message"]
    assert not any(r[0] == "0009,0010" for r in session.run_op("utilities.dicom_metadata", {"include_private": False})["table"]["rows"])
    sub = session.run_op("utilities.dicom_metadata", {"filter": "patient"})
    assert all("patient" in " ".join(map(str, r)).lower() for r in sub["table"]["rows"]) and len(sub["table"]["rows"]) < len(t["rows"])
    assert "matching 'patient'" in sub["message"]
    # a DICOM folder is read from its first file; metadata["dicom_path"] takes precedence over the path
    d = tmp_path / "series"
    d.mkdir()
    for i in range(2):
        _write_dicom(d / f"s{i}.dcm", value=i)
    session.open_file(str(d))
    assert session.active.path == str(d)
    assert {r[4] for r in session.run_op("utilities.dicom_metadata")["table"]["rows"]} >= {"PatientName", "Modality"}
    session.run_op("file.sample", {"name": "Blobs"})
    session.active.metadata["dicom_path"] = str(f)
    assert len(session.run_op("utilities.dicom_metadata")["table"]["rows"]) == len(t["rows"])
    (tmp_path / "plain.tif").write_bytes(b"II*\x00" + b"\x00" * 64)
    session.active.metadata["dicom_path"] = str(tmp_path / "plain.tif")
    with pytest.raises(ValueError, match="not a DICOM image"):
        session.run_op("utilities.dicom_metadata")


def test_dicom_metadata_needs_pydicom(session, tmp_path, monkeypatch):
    f = tmp_path / "x.dcm"
    _write_dicom(f)
    session.open_file(str(f))
    monkeypatch.setitem(sys.modules, "pydicom", None)
    with pytest.raises(RuntimeError, match="pip install pydicom"):
        session.run_op("utilities.dicom_metadata")
