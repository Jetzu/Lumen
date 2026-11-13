"""ImageJ macro-language tools (lumen/core/ops/macro_tools.py): parsing of ``macro "Name Tool - icon" { … }`` sets, icon decoding,
registration as tool ops, clicks / drags / modifiers through the registry (getCursorLoc, isKeyDown, toolID …), Tool Options /
Tool Selected macros with persistent ``var`` state, Install… / reload / startup.ijm, the Macro Tool and Plugin Frame templates,
``run("New...", "type=[Plugin Frame]")`` and the built-in Image Statistics frame."""
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core import ops as ops_pkg
from lumen.core.ij_compat import IJ, IJ_CUSTOM_TOOL_ID
from lumen.core.image import LumenImage
from lumen.core.ops import macro_tools as M
from lumen.core.ops import plugins_ops as P
from lumen.core.registry import REGISTRY
from lumen.core.session import Session
from lumen.server.app import create_app

CIRCLE = '''// ImageJ's "Circle Tool" example: draws a circle of the given radius where you click
var radius = 20;
macro "Circle Tool - C00cO11cc" {
    getCursorLoc(x, y, z, flags);
    makeOval(x-radius, y-radius, radius*2, radius*2);
    print("circle at " + x + "," + y + " r=" + radius + " flags=" + flags + " z=" + z);
}
macro "Circle Tool Options" {
    radius = getNumber("Radius: ", 7);
    print("radius now " + radius);
}
'''

SPRAY = '''
macro "Spray Can Tool - C0a0D21D31D52D45D62D71D75D96D82D99Dc4Db5Da6Db6Dc7Dd8Dd9De9Df8" {
    leftButton=16; rightButton=4;
    getCursorLoc(x, y, z, flags);
    n = 0;
    while (flags&leftButton!=0) {        // ImageJ idiom: loop while the button is down
        setPixel(x, y, 255);
        n++;
        getCursorLoc(x, y, z, flags);
        wait(10);
    }
    if (flags&rightButton!=0) print("right button");
    print("sprayed " + n);
}
'''

ARROW = '''
macro "Arrow Tool - C037L1ee1L65e1Lb8e1" {
    leftButton=16;
    getCursorLoc(x, y, z, flags);
    xstart = x; ystart = y;
    while (true) {
        getCursorLoc(x, y, z, flags);
        if (flags&leftButton==0) break;
    }
    if (isKeyDown("shift")) print("shift held");
    setKeyDown("none");
    makeLine(xstart, ystart, x, y);
    print("arrow " + xstart + "," + ystart + " -> " + x + "," + y);
}
macro "Arrow Tool Selected" {
    print("selected " + IJ.getToolName() + " id=" + toolID + " call=" + call("ij.gui.Toolbar.getToolId") + " name=" + call("ij.gui.Toolbar.getToolName"));
}
macro "Count Arrows [c]" { print("count macro: " + nResults); }
macro "Stamp Action Tool - Cf00F2288" { print("stamped at " + getTitle()); }
macro "Choices Menu Tool - C000" { print("menus are not supported"); }
'''


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated Lumen home ($LUMEN_HOME) with the registry and the macro-tool bookkeeping put back afterwards."""
    h = tmp_path / "lumen_home"
    monkeypatch.setenv("LUMEN_HOME", str(h))
    monkeypatch.delenv("LUMEN_PLUGINS", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    saved = dict(REGISTRY.ops)
    saved_files = {k: list(v) for k, v in M.MACRO_TOOL_FILES.items()}
    saved_err = dict(M.MACRO_TOOL_ERRORS)
    saved_sets = dict(M._SETS)
    saved_ops = {k: list(v) for k, v in ops_pkg.PLUGIN_OPS.items()}
    saved_mods = set(sys.modules)
    (h / "plugins").mkdir(parents=True)
    yield h
    REGISTRY.ops.clear()
    REGISTRY.ops.update(saved)
    M.MACRO_TOOL_FILES.clear()
    M.MACRO_TOOL_FILES.update(saved_files)
    M.MACRO_TOOL_ERRORS.clear()
    M.MACRO_TOOL_ERRORS.update(saved_err)
    M._SETS.clear()
    M._SETS.update(saved_sets)
    ops_pkg.PLUGIN_OPS.clear()
    ops_pkg.PLUGIN_OPS.update(saved_ops)
    for m in [m for m in sys.modules if m.startswith("lumen_plugin_") and m not in saved_mods]:
        sys.modules.pop(m, None)


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


def _write(home: Path, name: str, text: str, folder: str = "plugins") -> Path:
    p = home / folder / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


# ---------------------------------------------------------------- parsing
def test_parse_macro_set_blocks_install_code_and_grouping():
    text = ('var n = 0;  // a "{" in a comment\nfunction f(a) { return "}" + a; }\n' + CIRCLE + ARROW + "print('auto-run');\n")
    ms = M.parse_macro_set(text)
    names = [b.name for b in ms.blocks]
    assert names == ["Circle Tool - C00cO11cc", "Circle Tool Options", "Arrow Tool - C037L1ee1L65e1Lb8e1", "Arrow Tool Selected", "Count Arrows [c]",
                     "Stamp Action Tool - Cf00F2288", "Choices Menu Tool - C000"]
    assert ms.blocks[0].base == "Circle Tool" and ms.blocks[0].icon == "C00cO11cc" and ms.blocks[1].icon == "" and ms.blocks[0].line == 5
    assert "makeOval" in ms.blocks[0].body and "macro" not in ms.install_code.replace("// a", "")
    assert 'function f(a) { return "}" + a; }' in ms.install_code and "var n = 0;" in ms.install_code and "print('auto-run');" in ms.install_code
    assert "var radius = 20;" in ms.install_code  # the var of the circle set is install code too
    tools = {t.name: t for t in ms.tools}
    assert set(tools) == {"Circle Tool", "Arrow Tool", "Stamp Action Tool"}
    assert tools["Circle Tool"].options is not None and "getNumber" in tools["Circle Tool"].options and tools["Circle Tool"].selected is None
    assert tools["Arrow Tool"].selected is not None and tools["Arrow Tool"].options is None and tools["Arrow Tool"].icon == "C037L1ee1L65e1Lb8e1"
    assert tools["Stamp Action Tool"].action and not tools["Arrow Tool"].action
    assert [b.base for b in ms.macros] == ["Count Arrows [c]"]  # Menu Tool blocks are neither tools nor macros
    # single-quoted names, an Options block before its tool, nested braces in the body
    ms2 = M.parse_macro_set("macro 'Dot Tool Options' { if (true) { x = 1; } }\nmacro 'Dot Tool - C000D55' { if (x) { print('{'); } }")
    assert [t.name for t in ms2.tools] == ["Dot Tool"] and ms2.tools[0].options.strip() == "if (true) { x = 1; }" and "print('{')" in ms2.tools[0].body
    assert M.parse_macro_set("print(1);").tools == [] and M.parse_macro_set("").install_code == ""
    assert M.has_tool_macros(CIRCLE) and not M.has_tool_macros("macro 'Plain' { print(1); }") and not M.has_tool_macros("macro 'Broken Tool' { ")
    with pytest.raises(ValueError, match="missing"):
        M.parse_macro_set('macro "Broken Tool" { print(1);')


def test_decode_icon():
    assert M.decode_icon("C00cO11cc") == ("◯", "#0000cc")
    assert M.decode_icon("Cf00T0b12C") == ("C", "#ff0000")
    assert M.decode_icon("T0b11CT7b11LTab11I") == ("CL", "")
    assert M.decode_icon("C037L1ee1L65e1Lb8e1") == ("╱", "#003377")
    assert M.decode_icon("C0a0D21D31D52") == ("⁘", "#00aa00")
    assert M.decode_icon("Cf00F2288") == ("■", "#ff0000")
    assert M.decode_icon("C000V2288") == ("●", "#000000")
    assert M.decode_icon("P11f1ff1f10") == ("⬠", "")
    assert M.decode_icon("") == ("⚙", "") and M.decode_icon("B03C037R1177") == ("▭", "#003377")
    assert M.decode_icon("C0f0C00f") == ("⚙", "#00ff00")  # the first colour wins


# ---------------------------------------------------------------- registration and runs
def test_register_file_creates_tool_ops(home, session):
    p = _write(home, "arrow.ijm", ARROW)
    ids = M.register_file(p, session)
    assert ids == ["macrotool.arrow_tool", "macrotool.stamp_action_tool", "macros.count_arrows"] and M.MACRO_TOOL_FILES[str(p)] == ids
    assert session.settings["ops_version"] == 1
    sp = REGISTRY.ops["macrotool.arrow_tool"]
    assert sp.tool and sp.kind == "analysis" and sp.tool_icon == "╱" and sp.tool_color == "#003377" and sp.tool_cursor == "crosshair"
    assert sp.to_dict()["menu"] == "Plugins/Tools" and sp.requires_image is False and [q.name for q in sp.params] == ["selected"]
    assert "arrow.ijm" in sp.description and '"Arrow Tool Selected"' in sp.description and "macro tool" in sp.keywords
    d = sp.to_dict()
    assert d["tool"] is True and d["tool_color"] == "#003377" and d["frame"] is False
    stamp = REGISTRY.ops["macrotool.stamp_action_tool"]
    assert not stamp.tool and stamp.icon == "■" and stamp.to_dict()["menu"] == "Plugins/Tools"
    assert session.run_op("macrotool.stamp_action_tool")["log"] == "stamped at blobs.tif\n"
    cnt = REGISTRY.ops["macros.count_arrows"]
    assert cnt.name == "Count Arrows" and cnt.to_dict()["menu"] == "Plugins/Macros" and session.run_op("macros.count_arrows")["log"] == "count macro: 0\n"
    # registering again replaces (same ids); a file without tools registers nothing
    assert M.register_file(p) == ids and len([o for o in REGISTRY.ops if o.startswith("macrotool.")]) == 2
    assert M.register_file(_write(home, "plain.ijm", "print(1);")) == [] and M.MACRO_TOOL_FILES[str(home / "plugins" / "plain.ijm")] == []
    assert M.unregister_file(p) == ids and "macrotool.arrow_tool" not in REGISTRY.ops and str(p) not in M.MACRO_TOOL_FILES


def test_click_runs_macro_with_cursor_and_flags(home, session):
    M.register_file(_write(home, "circle.ijm", CIRCLE))
    img = session.active
    out = session.run_op("macrotool.circle_tool", {"x": 30, "y": 40})
    assert out["log"] == "circle at 30,40 r=20 flags=16 z=0\n" and out["message"] == "circle at 30,40 r=20 flags=16 z=0"
    assert out["roi"]["type"] == "oval" and out["roi"]["x"] == 10 and out["roi"]["w"] == 40
    np.testing.assert_array_equal(img.data, session.images[img.id].data)
    # modifiers and buttons use ImageJ's flag bits: 1 shift, 2 ctrl, 8 alt, 16 left, 4 right, 8 middle
    assert "flags=17" in session.run_op("macrotool.circle_tool", {"x": 1, "y": 2, "shift": True})["log"]
    assert "flags=27" in session.run_op("macrotool.circle_tool", {"x": 1, "y": 2, "shift": True, "ctrl": True, "alt": True})["log"]
    assert "flags=4 " in session.run_op("macrotool.circle_tool", {"x": 1, "y": 2, "button": 2})["log"]
    assert "flags=8 " in session.run_op("macrotool.circle_tool", {"x": 1, "y": 2, "button": 1})["log"]
    # every interaction is recorded like any plugin tool
    assert session.macro[-1].startswith('lumen.run("macrotool.circle_tool", options=False, x=1.0, y=2.0, button=1')
    # drag-start does nothing; a drag of a non-looping macro runs once, at the press position (ImageJ: one run per press)
    assert session.run_op("macrotool.circle_tool", {"x": 5, "y": 6, "event": "drag-start"}).get("log") is None
    assert "circle at 5,6 " in session.run_op("macrotool.circle_tool", {"x": 5, "y": 6, "x2": 50, "y2": 60, "event": "drag-end"})["log"]
    # the selection follows the macro: no selection left → clear_roi
    M.register_file(_write(home, "clear.ijm", 'macro "Clear Tool - C000" { run("Select None"); }'))
    assert session.run_op("macrotool.clear_tool", {"x": 1, "y": 1}, roi={"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5}).get("clear_roi") is True


def test_drag_loops_see_press_and_release(home, session):
    M.register_file(_write(home, "spray.ijm", SPRAY))
    M.register_file(_write(home, "arrow.ijm", ARROW))
    img = session.active
    img.data[...] = 0
    assert session.run_op("macrotool.spray_can_tool", {"x": 1, "y": 2, "event": "drag-start"}) == {"op": "macrotool.spray_can_tool", "new_images": [], "elapsed_ms": pytest.approx(0, abs=1000)}
    out = session.run_op("macrotool.spray_can_tool", {"x": 1, "y": 2, "x2": 9, "y2": 8, "event": "drag-end"})
    assert out["log"] == "sprayed 2\n" and img.data[0, 0, 0, 2, 1] == 255 and img.data[0, 0, 0, 8, 9] == 255 and int(img.data.sum()) == 510
    assert session.run_op("macrotool.spray_can_tool", {"x": 3, "y": 3})["log"] == "sprayed 1\n" and img.data[0, 0, 0, 3, 3] == 255
    assert session.run_op("macrotool.spray_can_tool", {"x": 4, "y": 4, "button": 2})["log"] == "right button\nsprayed 0\n"
    # the arrow example: start at the press, end at the release; the roi is the line; shift reaches isKeyDown
    out = session.run_op("macrotool.arrow_tool", {"x": 1, "y": 2, "x2": 9, "y2": 8, "event": "drag-end", "shift": True})
    assert out["log"] == "shift held\narrow 1,2 -> 9,8\n" and out["roi"] == {"type": "line", "x1": 1.0, "y1": 2.0, "x2": 9.0, "y2": 8.0, "width": 1, "id": out["roi"]["id"]}
    assert session.run_op("macrotool.arrow_tool", {"x": 3, "y": 4})["log"] == "arrow 3,4 -> 3,4\n"


def test_options_and_selected_macros_and_persistent_vars(home, session):
    M.register_file(_write(home, "circle.ijm", CIRCLE))
    M.register_file(_write(home, "arrow.ijm", ARROW))
    sp = REGISTRY.ops["macrotool.circle_tool"]
    assert [q.name for q in sp.params] == ["options"] and sp.params[0].type == "bool" and "Options" in sp.params[0].help
    out = session.run_op("macrotool.circle_tool", {"options": True})
    assert out["log"] == "radius now 7\n" and out["message"] == "radius now 7"
    # the var set by the Options macro is what the tool reads next (ImageJ keeps the installed set's globals)
    assert "r=7 " in session.run_op("macrotool.circle_tool", {"x": 30, "y": 40})["log"]
    # another session has its own state
    s2 = Session()
    s2.run_op("file.sample", {"name": "Blobs"})
    assert "r=20 " in s2.run_op("macrotool.circle_tool", {"x": 30, "y": 40})["log"]
    # Tool Selected: tool name and ImageJ-style custom tool id (CUSTOM1 = 15 for the first macro tool)
    out = session.run_op("macrotool.arrow_tool", {"selected": True})
    assert out["log"] == f"selected Arrow Tool id={IJ_CUSTOM_TOOL_ID + 1} call={IJ_CUSTOM_TOOL_ID + 1} name=Arrow Tool\n"
    # the parameters only exist for the blocks a set defines: an unknown one is rejected by coerce_params (E7-8: no silent drops)
    assert [q.name for q in REGISTRY.ops["macrotool.arrow_tool"].params] == ["selected"]
    with pytest.raises(ValueError, match="unknown parameter 'selected' for macrotool.circle_tool"):
        session.run_op("macrotool.circle_tool", {"selected": True})
    with pytest.raises(ValueError, match="unknown parameter 'options' for macrotool.arrow_tool"):
        session.run_op("macrotool.arrow_tool", {"options": True})
    assert "circle at 0,0 " in session.run_op("macrotool.circle_tool", {})["log"]  # a plain click
    with pytest.raises(ValueError, match="has no"):
        M.run_tool(type("Ctx", (), {"session": session, "roi": None})(), home / "plugins" / "arrow.ijm", "Arrow Tool", options=True)
    # a changed file starts afresh: the var initialiser runs again
    _write(home, "circle.ijm", CIRCLE.replace("var radius = 20;", "var radius = 3;"))
    assert "r=3 " in session.run_op("macrotool.circle_tool", {"x": 30, "y": 40})["log"]
    # a tool removed from its file reports it
    _write(home, "circle.ijm", 'macro "Other Tool - C000" { print(1); }')
    with pytest.raises(ValueError, match="no longer defined"):
        session.run_op("macrotool.circle_tool", {"x": 1, "y": 1})


def test_tool_on_a_stack_sees_the_slice_and_pixels(home):
    M.register_file(_write(home, "probe.ijm", 'macro "Probe Tool - C000" { getCursorLoc(x, y, z, flags); print(x + "," + y + " z=" + z + " v=" + getPixel(x, y)); }'))
    s = Session()
    stack = s.add_image(LumenImage((np.arange(4 * 8 * 8).reshape(4, 8, 8) * 100).astype(np.uint16), name="stack", axes="ZYX"))
    s.update_display(stack.id, {"z": 2})
    assert s.run_op("macrotool.probe_tool", {"x": 3, "y": 4})["log"] == f"3,4 z=2 v={int(stack.data[0, 2, 0, 4, 3])}\n"
    rgb = s.add_image(LumenImage(np.full((4, 4, 3), 200, np.uint8), name="rgb", rgb=True))
    assert rgb.rgb and s.run_op("macrotool.probe_tool", {"x": 1, "y": 1})["log"] == f"1,1 z=0 v={(200 << 16) | (200 << 8) | 200}\n"
    s.add_image(LumenImage(np.full((4, 4), 0.5, np.float32), name="f"))
    assert s.run_op("macrotool.probe_tool", {"x": 1, "y": 1})["log"] == "1,1 z=0 v=0.5\n"


# ---------------------------------------------------------------- install, scan, reload, startup
def test_install_registers_a_tool_set_without_a_wrapper(home, session, tmp_path):
    src = tmp_path / "Circle_Tool.ijm"
    src.write_text(CIRCLE, encoding="utf-8")
    res = session.run_op("macros.install", {"path": str(src)})
    pdir = home / "plugins"
    assert (pdir / "Circle_Tool.ijm").is_file() and not (pdir / "Circle_Tool_macro.py").exists()
    assert res["table"]["rows"] == [["macrotool.circle_tool", "Circle Tool", "Plugins/Tools"]] and "Circle Tool" in res["message"]
    assert "macros.circle_tool" not in REGISTRY.ops and REGISTRY.ops["macrotool.circle_tool"].tool
    assert "r=20" in session.run_op("macrotool.circle_tool", {"x": 30, "y": 40})["log"]
    rows = {r[0]: r for r in session.run_op("plugins.list")["table"]["rows"]}
    assert rows["Circle_Tool.ijm"][2] == "ImageJ macro tool set" and rows["Circle_Tool.ijm"][5] == "macrotool.circle_tool" and rows["Circle_Tool.ijm"][6] == "loaded (1 tool(s), 0 macro command(s))"
    # a plain macro is still wrapped into a command, and a wrapped file that becomes a tool set loses its wrapper
    plain = tmp_path / "plain.ijm"
    plain.write_text('print("plain ran");', encoding="utf-8")
    assert session.run_op("macros.install", {"path": str(plain)})["table"]["rows"] == [["macros.plain", "plain", "Plugins/Macros"]]
    assert (pdir / "plain_macro.py").is_file() and session.run_op("macros.plain")["log"] == "plain ran\n"
    plain.write_text(SPRAY, encoding="utf-8")
    assert session.run_op("macros.install", {"path": str(plain)})["table"]["rows"] == [["macrotool.spray_can_tool", "Spray Can Tool", "Plugins/Tools"]]
    assert not (pdir / "plain_macro.py").exists() and "macros.plain" not in REGISTRY.ops


def test_scan_reload_errors_and_startup(home, session):
    _write(home, "circle.ijm", CIRCLE)
    _write(home, "spray.ijm", SPRAY, folder="macros")
    _write(home, "broken.ijm", 'macro "Broken Tool - C000" { print(1); ', folder="plugins")
    _write(home, "startup.ijm", 'print("startup ran");\nmacro "Startup Tool - C000T0b12S" { getCursorLoc(x, y, z, f); print("startup tool " + x); }', folder=".")
    info = M.load_macro_tools()
    assert sorted(Path(f).name for f in info["files"]) == ["circle.ijm", "spray.ijm", "startup.ijm"]
    assert list(info["errors"]) == [str(home / "plugins" / "broken.ijm")] and "missing" in info["errors"][str(home / "plugins" / "broken.ijm")]
    assert {"macrotool.circle_tool", "macrotool.spray_can_tool", "macrotool.startup_tool"} <= set(REGISTRY.ops)
    assert REGISTRY.ops["macrotool.startup_tool"].tool_icon == "S"
    # the startup macro runs its auto-run code only: the tool body is a definition
    out = P.run_startup_scripts(session)
    assert out[0]["ok"] and out[0]["output"] == "startup ran\n"
    assert session.run_op("macrotool.startup_tool", {"x": 7, "y": 7})["log"] == "startup tool 7\n"
    # plugins.reload rescans: a removed file drops its tools, a new tool appears, errors are listed
    (home / "plugins" / "circle.ijm").unlink()
    _write(home, "second.ijm", 'macro "Second Tool - C000" { print("second"); }')
    res = session.run_op("plugins.reload")
    assert "macrotool.circle_tool" not in REGISTRY.ops and "macrotool.second_tool" in REGISTRY.ops and "macrotool.spray_can_tool" in REGISTRY.ops
    rows = {r[0]: r for r in res["table"]["rows"]}
    assert rows["spray.ijm"][2] == "loaded (macro tools)" and rows["broken.ijm"][2].startswith("error:") and "macro tool command(s)" in res["message"]
    assert rows["broken.ijm"][3] == "" and rows["second.ijm"][3] == "macrotool.second_tool" and rows["spray.ijm"][3] == "macrotool.spray_can_tool"
    listed = {r[0]: r for r in session.run_op("plugins.list")["table"]["rows"]}  # the plugins folder only
    assert listed["broken.ijm"][6].startswith("error:") and listed["second.ijm"][2] == "ImageJ macro tool set" and "spray.ijm" not in listed
    # tool_id_of follows the registration order of the tool ops
    tool_ids = [sp.name for sp in REGISTRY.ops.values() if sp.tool and sp.id.startswith("macrotool.")]
    assert M.tool_id_of(tool_ids[0]) == IJ_CUSTOM_TOOL_ID and M.tool_id_of(tool_ids[-1]) == IJ_CUSTOM_TOOL_ID + len(tool_ids) - 1 and M.tool_id_of("nope") == IJ_CUSTOM_TOOL_ID


def test_whole_file_runs_skip_tool_blocks(session):
    ij = IJ(session)
    out = ij.macro('var r = 2;\nmacro "Foo Tool - C000" { print("tool body"); }\nprint("install " + r);\nmacro "Bar" { print("bar runs"); }\n'
                   'macro "Foo Tool Options" { print("options"); }\nmacro "Act Action Tool" { print("act"); }\nmacro "M Menu Tool" { print("menu"); }\nfunction g() { return 1; }')
    assert out == "install 2\nbar runs\n"
    # getCursorLoc outside a tool interaction is unknown; the tool helpers have ImageJ defaults
    assert ij.macro("getCursorLoc(x, y, z, f); print(x + \" \" + y + \" \" + z + \" \" + f); print(toolID); print(IJ.getToolName()); print(isKeyDown(\"shift\"));") == "-1 -1 0 0\n0\nrectangle\n0\n"
    assert ij.macro('setTool("oval"); print(toolID + " " + IJ.getToolName()); setTool(7); print(IJ.getToolName()); setTool("Circle Tool"); print(toolID >= 15);') == "1 oval\npoint\n1\n"
    assert ij.macro('setKeyDown("shift alt"); print(isKeyDown("shift") + " " + isKeyDown("alt") + " " + isKeyDown("ctrl")); setKeyDown("none"); print(isKeyDown("shift"));') == "1 1 0\n0\n"
    # ImageJ precedence: & | ^ << >> bind like * (the tool-macro idiom `flags&16!=0`)
    assert ij.macro("flags = 17; print(flags&16!=0); print(flags&2!=0); print(flags&16==16); print(1|2==3); print(2*3&2);") == "1\n0\n1\n1\n2\n"
    assert ij.macro('call("ij.Prefs.set", "tool.radius", 12); print(call("ij.Prefs.get", "tool.radius", 5)); print(call("ij.Prefs.get", "nope", 5));') == "12\n5\n"
    assert session.settings["ij.prefs"] == {"tool.radius": "12"}


def test_macro_env_persists_vars_and_keeps_user_functions(session):
    ij = IJ(session)
    env: dict = {}
    assert ij.macro("var n = 1;\nfunction wait(ms) { print(\"my wait \" + ms); }\nfunction bump() { n++; }", env=env) == ""
    assert ij.macro("bump(); bump(); print(n); wait(3);", env=env) == "3\nmy wait 3\n"
    ij2 = IJ(session)
    assert ij2.macro("bump(); print(n); wait(4); print(getTitle());", env=env) == "4\nmy wait 4\nblobs.tif\n"  # rebound to ij2, user wait() kept
    assert env["IJ"] is ij2 and "bump" in env and "wait" not in env["__ij_names__"] and "print" in env["__ij_names__"]


# ---------------------------------------------------------------- templates, run("New…") and the API
def test_macro_tool_template_runs(home, session):
    res = session.run_op("plugins.new_template", {"kind": "Macro Tool", "name": "Dots"})
    p = home / "plugins" / "dots.ijm"
    assert p.is_file() and res["table"]["name"] == "dots.ijm" and "macrotool.dots_tool" in res["message"]
    sp = REGISTRY.ops["macrotool.dots_tool"]
    assert sp.tool and sp.name == "Dots Tool" and sp.tool_icon == "◯" and sp.tool_color == "#00cc00" and [q.name for q in sp.params] == ["options"]
    img = session.active
    out = session.run_op("macrotool.dots_tool", {"x": 20, "y": 30})
    assert out["message"] == "Dots: circle 1 at (20, 30) r=12" and session.results.rows[-1]["Label"] == "Dots 1" and session.results.rows[-1]["Radius"] == 12
    out = session.run_op("macrotool.dots_tool", {"x": 20, "y": 30, "x2": 23, "y2": 34, "event": "drag-end"})
    assert out["message"] == "Dots: circle 2 at (20, 30) r=5"  # the drag length sets the radius
    assert session.run_op("macrotool.dots_tool", {"x": 20, "y": 30, "alt": True})["message"] == f"Dots: (20, 30) = {int(img.data[0, 0, 0, 30, 20])}"
    assert session.run_op("macrotool.dots_tool", {"options": True})["message"] == "Dots: radius 12, counter reset"
    assert session.run_op("macrotool.dots_tool", {"x": 1, "y": 1})["message"] == "Dots: circle 1 at (1, 1) r=12"
    # "Dots Tool" as the name does not double the suffix; existing files are protected
    with pytest.raises(FileExistsError):
        session.run_op("plugins.new_template", {"kind": "Macro Tool", "name": "Dots Tool"})
    session.run_op("plugins.new_template", {"kind": "Macro Tool", "name": "Dots Tool", "overwrite": True})
    assert REGISTRY.ops["macrotool.dots_tool"].name == "Dots Tool"
    assert session.run_op("plugins.new_template", {"kind": "Macro Tool", "name": "Later", "load": False})["message"].endswith("reload plugins to register it")
    assert "macrotool.later_tool" not in REGISTRY.ops and (home / "plugins" / "later.ijm").is_file()


def test_run_new_commands_write_templates(home, session):
    ij = IJ(session)
    out = ij.run("New...", "type=[Plugin Frame] name=[Stats Frame]")
    assert "Plugin Frame template" in out["message"] and REGISTRY.ops["user.stats_frame"].frame and REGISTRY.ops["user.stats_frame"].to_dict()["frame"] is True
    res = session.run_op("user.stats_frame", {"decimals": 1})
    assert res["table"]["columns"][:2] == ["Image", "Size"] and res["table"]["rows"][0][0] == "blobs.tif" and res["plot"]["series"][0]["name"] == "Mean"
    assert "Macro Tool template" in ij.run("Macro Tool", "name=Zed")["message"] and REGISTRY.ops["macrotool.zed_tool"].tool
    assert "Plugin Tool template" in ij.run("Plugin Tool", "name=[Pt]")["message"] and REGISTRY.ops["user.pt"].tool
    assert "Python command template" in ij.run("Plugin", "name=Cmd")["message"] and REGISTRY.ops["user.cmd"].kind == "image"
    assert "Python filter (plane) template" in ij.run("Plugin Filter", "name=Flt")["message"] and REGISTRY.ops["user.flt"].kind == "plane"
    assert "ImageJ macro template" in ij.run("Macro", "name=Mac")["message"] and "macros.mac" in REGISTRY.ops
    assert ij.run("New...", "type=[Text Window] name=Notes.txt text=hello")["table"]["name"] == "Notes.txt"
    with pytest.raises(ValueError, match="unknown template type"):
        ij.run("New...", "name=x")
    with pytest.raises(FileExistsError):
        ij.run("Plugin Frame", "name=[Stats Frame]")
    ij.run("Plugin Frame", "name=[Stats Frame] overwrite")


def test_image_statistics_frame_and_frame_flag():
    s = Session()
    sp = REGISTRY.ops["frames.image_statistics"]
    assert sp.frame and sp.kind == "analysis" and sp.requires_image is False and sp.to_dict()["frame"] is True and sp.to_dict()["preview"] is False
    assert sp.to_dict()["menu"] == "Plugins/Frames"
    assert s.run_op("frames.image_statistics")["table"]["rows"] == [] and "open an image" in s.run_op("frames.image_statistics")["message"]
    s.run_op("file.sample", {"name": "Blobs"})
    blobs = s.active
    stack = s.add_image(LumenImage((np.arange(3 * 8 * 8).reshape(3, 8, 8) * 100).astype(np.uint16), name="stack", axes="ZYX"))
    s.update_display(stack.id, {"z": 1})
    rgb = s.add_image(LumenImage(np.full((4, 4, 3), 200, np.uint8), name="rgb", rgb=True))
    flt = s.add_image(LumenImage(np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4), name="f"))
    res = s.run_op("frames.image_statistics", {"decimals": 3})
    rows = {r[0]: r for r in res["table"]["rows"]}
    assert res["table"]["name"] == "Image Statistics" and res["message"] == "4 open image(s)" and s.tables["Image Statistics"] == res["table"]
    assert rows["blobs.tif"][1:4] == ["256×256", "1/1/1", "uint8"] and rows["blobs.tif"][4] == pytest.approx(float(blobs.data.mean()), abs=1e-3)
    assert rows["stack"][4] == pytest.approx(float(stack.data[0, 1].mean()), abs=1e-3) and rows["stack"][8] == 64 and rows["stack"][2] == "3/1/1"
    assert rows["rgb"][3] == "RGB" and rows["rgb"][4] == 200 and rows["rgb"][5] == 0 and rows["rgb"][8] == 48
    assert rows["f"][4] == pytest.approx(0.5, abs=1e-3) and rows["f"][6] == 0 and rows["f"][7] == 1
    rows = {r[0]: r for r in s.run_op("frames.image_statistics", {"plane": "Whole stack"})["table"]["rows"]}
    assert rows["stack"][4] == pytest.approx(float(stack.data.mean()), abs=1e-2) and rows["stack"][8] == 192
    s.select(blobs.id)
    rows = {r[0]: r for r in s.run_op("frames.image_statistics", {"use_selection": True}, roi={"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10})["table"]["rows"]}
    assert rows["blobs.tif"][8] == 100 and rows["blobs.tif"][4] == pytest.approx(float(blobs.data[0, 0, 0, :10, :10].mean()), abs=1e-2) and rows["stack"][8] == 64
    # the API exposes the flag for every op
    c = TestClient(create_app(Session()))
    api_ops = {o["id"]: o for o in c.get("/api/ops").json()}
    assert api_ops["frames.image_statistics"]["frame"] is True and api_ops["filters.gaussian"]["frame"] is False
    assert all("frame" in o and "tool_color" in o for o in api_ops.values())
    assert c.post("/api/ops/frames.image_statistics/run", json={"params": {}}).json()["table"]["rows"] == []
