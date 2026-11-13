"""System / housekeeping commands (lumen/core/ops/system_ops.py): Plugins ▸ Utilities ▸ Properties / Threads /
Benchmark / Reset / Monitor Events / Search, Edit ▸ Options ▸ Memory & Threads / Proxy / Compiler / DICOM,
Process ▸ Batch ▸ Virtual Stack and Help ▸ Check for Updates."""
import os
import platform
import threading
import urllib.error
from pathlib import Path

import numpy as np
import pytest
import tifffile

import lumen
from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops import system_ops
from lumen.core.registry import REGISTRY
from lumen.core.session import Session
from lumen.core.undo import History

_ENV_VARS = system_ops._PROXY_VARS + system_ops._NO_PROXY_VARS


@pytest.fixture(autouse=True)
def _restore_process_state():
    """Undo budget, BLAS thread limits and the proxy variables are process-wide: put them back after every test."""
    saved_budget = History.default_budget_bytes
    saved_env = {v: os.environ.get(v) for v in _ENV_VARS}
    yield
    History.set_default_budget(saved_budget)
    system_ops.apply_thread_limit(0)
    for v, val in saved_env.items():
        if val is None:
            os.environ.pop(v, None)
        else:
            os.environ[v] = val


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


def _table(res, name):
    assert res["table"]["name"] == name
    return res["table"]


# ---------------------------------------------------------------- menu placement
def test_menu_placement_and_kinds():
    menus = {oid: REGISTRY.ops[oid].to_dict()["menu"] for oid in REGISTRY.ops}
    for oid in ("help.properties", "help.threads", "help.benchmark", "help.reset", "help.monitor_events", "help.search_commands", "help.reload_plugins"):
        assert menus[oid] == "Plugins/Utilities", oid
    for oid in ("edit.options_memory", "edit.options_proxy", "edit.options_dicom", "help.compiler_info"):
        assert menus[oid] == "Edit/Options", oid
    assert menus["batch.virtual_stack"] == "Process/Batch"
    assert menus["help.check_updates"] == "Help"
    for oid in ("help.properties", "help.benchmark", "edit.options_memory", "help.check_updates", "batch.virtual_stack"):
        assert REGISTRY.ops[oid].requires_image is False


# ---------------------------------------------------------------- Properties / Threads / Benchmark / Reset / Monitor Events
def test_properties_table(session):
    res = session.run_op("help.properties")
    t = _table(res, "Properties")
    assert t["columns"] == ["Property", "Value"]
    props = {r[0]: r[1] for r in t["rows"]}
    assert props["Lumen version"] == lumen.__version__
    assert props["Python"].startswith(platform.python_version())
    assert props["numpy"] == np.__version__
    for key in ("scipy", "scikit-image", "scikit-learn", "tifffile", "Platform", "CPU count", "Process RSS", "Environment (sys.prefix)",
                "Plugin folders", "Image memory", "Undo memory", "Undo budget", "Max threads (BLAS/OpenMP)", "Proxy"):
        assert key in props, key
    assert props["Open images"] == 1 and props["Commands"] == len(REGISTRY.ops)
    assert props["Image memory"].endswith("MB")
    assert "Lumen" in res["message"] and "Properties" in session.tables


def test_threads_table(session):
    done = threading.Event()
    t = threading.Thread(target=done.wait, name="lumen-test-worker", daemon=True)
    t.start()
    try:
        res = session.run_op("help.threads")
        tb = _table(res, "Threads")
        assert tb["columns"][:2] == ["Name", "Id"]
        names = [r[0] for r in tb["rows"]]
        assert tb["rows"][0][5] == "main" and "lumen-test-worker" in names
        row = tb["rows"][names.index("lumen-test-worker")]
        assert row[3] is True and row[4] is True and ".py:" in row[6]  # daemon, alive, "file.py:line func"
        assert "thread" in res["message"]
    finally:
        done.set()
        t.join(1)


def test_benchmark_table(session):
    res = session.run_op("help.benchmark", {"size": 128, "repeats": 2})
    tb = _table(res, "Benchmark")
    assert tb["columns"] == ["Operation", "Time (ms)", "Result"]
    names = [r[0] for r in tb["rows"]]
    assert names[-1] == "Total" and len(names) == 14
    for key in ("Gaussian", "Median", "FFT", "Threshold"):
        assert any(key in n for n in names), key
    assert all(isinstance(r[1], float) and r[1] >= 0 for r in tb["rows"])
    assert tb["rows"][-1][1] == pytest.approx(sum(r[1] for r in tb["rows"][:-1]), abs=0.1)
    assert "particles" in tb["rows"][-2][2]
    assert "128×128" in res["message"] and "best of 2" in res["message"]
    assert len(session.images) == 1  # the synthetic image never enters the session


def test_reset_clears_session_state(session):
    s = session
    img = s.active
    s.run_op("analyze.measure", roi={"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10})
    s.roi_manager.add({"type": "rect", "x": 1, "y": 1, "w": 5, "h": 5})
    s.thresholds[img.id] = {"lower": 10, "upper": 200}
    s.tables["Custom"] = {"name": "Custom", "columns": ["a"], "rows": [[1]]}
    s.last_plot = {"title": "p", "x": [0], "series": []}
    assert s.results.rows and s.roi_manager.rois and s.macro
    res = s.run_op("help.reset")
    assert s.results.rows == [] and s.results.columns == []
    assert s.roi_manager.rois == [] and s.thresholds == {} and s.tables == {} and s.last_plot is None
    assert len(s.macro) == 1 and s.macro[0].startswith('lumen.run("help.reset"')  # only the reset call itself is recorded
    assert len(s.images) == 1 and s.log  # images and the Log are kept by default
    for word in ("Results", "ROI Manager", "tables", "thresholds", "macro"):
        assert word in res["message"], word
    res = s.run_op("help.reset", {"close_images": True, "clear_log": True})
    assert len(s.images) == 0 and s.active_id is None and "1 image(s)" in res["message"]
    # only the reset's own summary line and run_op's "Reset… (n ms)" timing line remain
    assert len(s.log) == 2 and s.log[0]["msg"].startswith("Reset: cleared") and s.log[1]["msg"].startswith("Reset…")
    assert s.run_op("help.reset", {"clear_macro": False})["message"] == "Reset: nothing to clear"  # the macro holds the previous reset call


def test_monitor_events_table(session):
    s = session
    for i in range(5):
        s.add_log(f"event {i}", level="error" if i % 2 else "info")
    res = s.run_op("help.monitor_events", {"n": 3})
    tb = _table(res, "Events")
    assert tb["columns"] == ["#", "Time", "Level", "Message"]
    assert [r[3] for r in tb["rows"]] == ["event 2", "event 3", "event 4"]
    assert [r[2] for r in tb["rows"]] == ["info", "error", "info"]
    assert all(len(r[1].split(":")) == 3 for r in tb["rows"])
    assert tb["rows"][0][0] == len(s.log) - 3  # run_op logged the monitor call afterwards
    assert f"of {len(s.log) - 1} log entries" in res["message"]
    res = s.run_op("help.monitor_events", {"n": 50, "level": "error"})
    assert [r[3] for r in res["table"]["rows"]] == ["event 1", "event 3"]


# ---------------------------------------------------------------- Search
def test_search_commands(session):
    res = session.run_op("help.search_commands", {"query": "gaussian blur"})
    tb = _table(res, "Search")
    assert tb["columns"] == ["Id", "Name", "Menu", "Shortcut", "Description", "Score"]
    assert tb["rows"][0][0] == "filters.gaussian"
    scores = [r[5] for r in tb["rows"]]
    assert scores == sorted(scores, reverse=True) and all(sc > 0 for sc in scores)
    assert "match 'gaussian blur'" in res["message"]
    ids = [r[0] for r in session.run_op("help.search_commands", {"query": "help.thr"})["table"]["rows"]]
    assert "help.threads" in ids
    res = session.run_op("help.search_commands", {"query": "zzzqqqxx"})
    assert res["table"]["rows"] == [] and "No command" in res["message"]
    assert len(session.run_op("help.search_commands", {"query": "e", "max_results": 5})["table"]["rows"]) == 5
    with pytest.raises(ValueError):
        session.run_op("help.search_commands", {"query": "   "})


# ---------------------------------------------------------------- Memory & Threads
def test_options_memory_updates_existing_and_new_histories(session):
    s = session
    img = s.active
    hist = s.histories[img.id]
    for i in range(4):
        hist.push(img, f"step {i}")
    assert len(hist.undo) == 4 and hist.budget == History.default_budget_bytes
    res = s.run_op("edit.options_memory", {"undo_budget_mb": 0, "max_threads": 2})
    assert hist.budget == 0 and len(hist.undo) == 1 and hist.undo[0].label == "step 3"  # trimmed to the newest snapshot
    assert History.default_budget_bytes == 0 and s.settings["undo_budget_mb"] == 0 and s.settings["max_threads"] == 2
    assert "Undo budget 0.0 MB" in res["message"] and "thread" in res["message"]
    try:
        import threadpoolctl
        assert all(info["num_threads"] <= 2 for info in threadpoolctl.threadpool_info())
        assert "limited to 2 thread(s)" in res["message"]
    except ImportError:
        assert "stored only" in res["message"]
    # histories created afterwards (in this and in new sessions) use the new budget
    res = s.run_op("edit.options_memory", {"undo_budget_mb": 7, "max_threads": 0})
    assert hist.budget == 7_000_000
    new = s.add_image(LumenImage(np.zeros((8, 8), np.uint8), name="new"))
    assert s.histories[new.id].budget == 7_000_000
    other = Session()
    im2 = other.add_image(LumenImage(np.zeros((8, 8), np.uint8), name="other"))
    assert other.histories[im2.id].budget == 7_000_000
    assert "no limit" in res["message"] or "stored only" in res["message"]
    assert s.run_op("edit.options_memory", {"undo_budget_mb": 1500, "collect_garbage": True})["message"].count("collected") == 1


# ---------------------------------------------------------------- Proxy
def test_options_proxy_sets_and_unsets_env(session):
    s = session
    res = s.run_op("edit.options_proxy", {"proxy": "proxy.lab:3128", "no_proxy": "localhost, .lab"})
    assert s.settings["proxy"] == "http://proxy.lab:3128" and s.settings["no_proxy"] == "localhost,.lab"
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        assert os.environ[var] == "http://proxy.lab:3128"
    assert os.environ["NO_PROXY"] == "localhost,.lab" and os.environ["no_proxy"] == "localhost,.lab"
    assert "proxy.lab" in res["message"]
    res = s.run_op("edit.options_proxy", {"proxy": "https://user:pw@gw.example.org:8080"})
    assert os.environ["HTTPS_PROXY"] == "https://user:pw@gw.example.org:8080" and "NO_PROXY" not in os.environ
    res = s.run_op("edit.options_proxy", {"proxy": ""})
    assert s.settings["proxy"] == "" and "disabled" in res["message"]
    for var in _ENV_VARS:
        assert var not in os.environ
    with pytest.raises(ValueError):
        s.run_op("edit.options_proxy", {"proxy": "http://bad proxy"})


# ---------------------------------------------------------------- DICOM / Compiler / Reload plugins
def test_options_dicom_stores_settings(session):
    res = session.run_op("edit.options_dicom", {"open_as_32bit": True, "ignore_rescale": True, "calibrate_images": False, "window_level_on_open": False})
    st = session.settings
    assert st["dicom_open_as_32bit"] is True and st["dicom_ignore_rescale"] is True
    assert st["dicom_calibrate_images"] is False and st["dicom_window_level_on_open"] is False
    assert "open_as_32bit=on" in res["message"] and "calibrate_images=off" in res["message"]
    session.run_op("edit.options_dicom")
    assert st["dicom_open_as_32bit"] is False and st["dicom_calibrate_images"] is True


def test_compiler_info_and_reload_plugins(session, tmp_path, monkeypatch):
    plug = tmp_path / "plugins"
    plug.mkdir()
    (plug / "my_plugin.py").write_text(
        'from lumen.core.registry import op\n\n@op("testplugin.invert_x", "Invert X", "Plugins/Test")\ndef invert_x(plane, ctx):\n    return 255 - plane\n')
    monkeypatch.setenv("LUMEN_PLUGINS", str(plug))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))  # never touch ~/.lumen/plugins
    user_ops = {oid: sp for oid, sp in REGISTRY.ops.items() if (getattr(sp.func, "__module__", "") or "").startswith("lumen_plugin_")}
    try:
        res = session.run_op("help.compiler_info")
        props = {r[0]: r[1] for r in _table(res, "Compiler")["rows"]}
        assert "Python" in props["Language"] and str(plug) in props["Plugin folders"] and props["Plugin files"] == "1"
        assert "not loaded" in props["plugin my_plugin.py"] and "@op" in props["Minimal plugin"] and "reload_plugins" in props["Reload"]
        res = session.run_op("help.reload_plugins")
        assert "testplugin.invert_x" in REGISTRY.ops and "1 plugin command(s)" in res["message"]
        assert REGISTRY.ops["testplugin.invert_x"].func.__module__ == "lumen_plugin_my_plugin"
        assert res["table"]["rows"] == [["testplugin.invert_x", "Invert X", "Plugins/Test"]]
        props = {r[0]: r[1] for r in session.run_op("help.compiler_info")["table"]["rows"]}
        assert "testplugin.invert_x" in props["plugin my_plugin.py"]
        # the plugin works and a second reload replaces it without a duplicate-id error
        session.run_op("testplugin.invert_x")
        session.run_op("help.reload_plugins")
        assert "testplugin.invert_x" in REGISTRY.ops
    finally:
        REGISTRY.ops.pop("testplugin.invert_x", None)
        for oid, sp in user_ops.items():
            REGISTRY.ops.setdefault(oid, sp)


# ---------------------------------------------------------------- Virtual stack
def test_virtual_stack_processes_folder_plane_by_plane(tmp_path):
    rng = np.random.default_rng(0)
    for i in range(3):
        tifffile.imwrite(str(tmp_path / f"plane_{i}.tif"), rng.integers(0, 255, (24, 32)).astype(np.uint8))
    s = Session()
    out = tmp_path / "out"
    script = 'lumen.run("filters.gaussian", sigma=1)\nlumen.run("analyze.measure")'
    res = s.run_op("batch.virtual_stack", {"input_folder": str(tmp_path), "pattern": r"\.tif$", "output_folder": str(out), "format": "png", "script": script})
    tb = _table(res, "Batch")
    assert [r[0] for r in tb["rows"]] == ["plane_0.tif", "plane_1.tif", "plane_2.tif"]
    assert all(r[1] == "ok" for r in tb["rows"]), tb["rows"]
    assert sorted(p.name for p in out.glob("*.png")) == ["plane_0.png", "plane_1.png", "plane_2.png"]
    assert len(s.images) == 0  # each plane is closed before the next one is read: the stack is never held
    assert len(s.results.rows) == 3 and (out / "Results.csv").read_text().count("\n") == 4
    assert "Virtual stack: 3 ok, 0 failed of 3 file(s)" in res["message"]
    assert len(s.macro) == 1 and s.macro[0].startswith('lumen.run("batch.virtual_stack"')
    spec = REGISTRY.ops["batch.virtual_stack"]
    assert "plane by plane" in spec.description and "never held" in spec.description
    with pytest.raises(FileNotFoundError):
        s.run_op("batch.virtual_stack", {"input_folder": str(tmp_path / "missing"), "script": ""})


# ---------------------------------------------------------------- Check for updates
class _Response:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self.payload


def test_check_updates_offline_and_newer_release(session, monkeypatch):
    calls = []

    def offline(url, timeout=None):
        calls.append((url, timeout))
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(system_ops.urllib.request, "urlopen", offline)
    res = session.run_op("help.check_updates")
    rows = {r[0]: (r[1], r[2]) for r in _table(res, "Updates")["rows"]}
    assert rows["Installed (lumen.__version__)"][0] == lumen.__version__
    assert rows["Source tree (pyproject.toml)"][0] == lumen.__version__ and "same as installed" in rows["Source tree (pyproject.toml)"][1]
    assert rows["PyPI lumen-imaging"] == ("unknown", "no network")
    assert calls == [("https://pypi.org/pypi/lumen-imaging/json", 3.0)]
    assert "no network" in res["message"] and "nothing was installed" in res["message"]
    # a newer release is reported with the pip command but never installed
    monkeypatch.setattr(system_ops.urllib.request, "urlopen", lambda url, timeout=None: _Response(b'{"info": {"version": "99.0"}}'))
    res = session.run_op("help.check_updates")
    rows = {r[0]: (r[1], r[2]) for r in res["table"]["rows"]}
    assert rows["PyPI lumen-imaging"][0] == "99.0" and "pip install -U lumen-imaging" in rows["PyPI lumen-imaging"][1]
    assert "99.0 is available" in res["message"] and lumen.__version__ == rows["Installed (lumen.__version__)"][0]
    # same release = up to date; 404 = not on PyPI; offline-only check never touches the network
    monkeypatch.setattr(system_ops.urllib.request, "urlopen", lambda url, timeout=None: _Response(('{"info": {"version": "%s"}}' % lumen.__version__).encode()))
    assert "up to date" in session.run_op("help.check_updates")["message"]
    monkeypatch.setattr(system_ops.urllib.request, "urlopen", lambda url, timeout=None: (_ for _ in ()).throw(urllib.error.HTTPError(url, 404, "nf", {}, None)))
    assert ("unknown", "not on PyPI") == {r[0]: (r[1], r[2]) for r in session.run_op("help.check_updates")["table"]["rows"]}["PyPI lumen-imaging"]
    calls.clear()
    monkeypatch.setattr(system_ops.urllib.request, "urlopen", offline)
    res = session.run_op("help.check_updates", {"check_pypi": False})
    assert calls == [] and "not checked" in [r[1] for r in res["table"]["rows"]]


def test_compare_versions():
    assert system_ops.compare_versions("0.1.0", "0.1.0") == 0
    assert system_ops.compare_versions("0.2.0", "0.1.9") == 1
    assert system_ops.compare_versions("0.1.0rc1", "0.1.0") == -1
    assert system_ops.compare_versions("1.10", "1.9") == 1


# ---------------------------------------------------------------- Search… (files, images, Log, script)
def test_search_files_greps_folders_images_log_and_script(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path))
    monkeypatch.delenv("LUMEN_PLUGINS", raising=False)
    (tmp_path / "plugins").mkdir()
    (tmp_path / "macros" / "sub").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "plugins" / "my_plugin.py").write_text("import numpy\n# the Needle is here\nprint('needle')\n", encoding="utf-8")
    (tmp_path / "macros" / "sub" / "demo.ijm").write_text('run("Measure");\n// needle in a macro\n', encoding="utf-8")
    (tmp_path / "scripts" / "notes.txt").write_text("nothing to see\n", encoding="utf-8")
    (tmp_path / "scripts" / "image.tif").write_bytes(b"needle")  # not a text file
    (tmp_path / "plugins" / ".hidden.py").write_text("needle\n", encoding="utf-8")
    (tmp_path / "plugins" / "__pycache__").mkdir()
    (tmp_path / "plugins" / "__pycache__" / "x.py").write_text("needle\n", encoding="utf-8")
    s = Session()
    img = s.add_image(LumenImage(np.zeros((4, 4), np.uint8), name="needle.tif"))
    img.metadata["info"] = {"note": "a Needle in the metadata", "list": ["no", "needle"]}
    s.add_log("log needle entry")
    s.macro.append('lumen.run("filters.gaussian", sigma=2)  # needle')
    res = s.run_op("help.search_files", {"query": "needle"})
    t = _table(res, "Search Files")
    assert t["columns"] == ["Where", "File", "Line", "Text"]
    rows = t["rows"]
    files = [(r[0], Path(r[1]).name, r[2]) for r in rows if r[0] in ("plugins", "macros", "scripts")]
    assert ("plugins", "my_plugin.py", 2) in files and ("plugins", "my_plugin.py", 3) in files and ("macros", "demo.ijm", 2) in files
    assert all(name not in (".hidden.py", "x.py", "image.tif", "notes.txt") for _w, name, _l in files)
    assert ["image", "needle.tif", "", "needle.tif"] in rows
    meta = [r for r in rows if r[0] == "metadata"]
    assert ["metadata", "needle.tif", "info.note", "a Needle in the metadata"] in meta and ["metadata", "needle.tif", "info.list[1]", "needle"] in meta
    assert any(r[0] == "Log" and r[3] == "log needle entry" for r in rows)
    assert any(r[0] == "script" and r[1] == "recorded script" and r[2] == 1 for r in rows)
    assert "3 file(s) searched" in res["message"] and "'needle'" in res["message"]
    # case-sensitive: the capitalised hits disappear; the sources can be switched off; the result count is capped
    res = s.run_op("help.search_files", {"query": "needle", "case_sensitive": True})
    assert not any("Needle" in r[3] for r in res["table"]["rows"]) and any("needle" in r[3] for r in res["table"]["rows"])
    res = s.run_op("help.search_files", {"query": "needle", "files": False, "images": False, "log": False})
    rows = res["table"]["rows"]  # the recorder also logged the search calls above, so several script lines match
    assert rows and all(r[0] == "script" for r in rows) and rows[0][2] == 1 and rows[0][3].startswith('lumen.run("filters.gaussian"') and "file(s) searched" not in res["message"]
    res = s.run_op("help.search_files", {"query": "needle", "max_results": 2})
    assert len(res["table"]["rows"]) == 2 and "stopped after 2" in res["message"]
    assert not s.run_op("help.search_files", {"query": "zzz-nothing"})["table"]["rows"]
    with pytest.raises(ValueError, match="text to search"):
        s.run_op("help.search_files", {"query": "  "})
    assert REGISTRY.ops["help.search_files"].to_dict()["menu"] == "Plugins/Utilities"


# ---------------------------------------------------------------- Update Lumen (help.update)
class _Proc:
    """A finished subprocess.Popen stand-in: communicate() hands back the canned output."""

    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr, self.pid = returncode, stdout, stderr, 4242

    def communicate(self, timeout=None):
        return self.stdout, self.stderr

    def kill(self):
        pass


def _fake_run(calls, results=None):
    """Fake subprocess.Popen: records argv / cwd / the non-interactive kwargs run_command must pass, returns the next _Proc."""
    results = list(results or [])

    def popen(argv, cwd=None, **kwargs):
        calls.append({"argv": list(argv), "cwd": cwd})
        # never interactive: stdin closed, no controlling terminal (POSIX), git / GCM / ssh / pip told not to prompt
        assert kwargs["stdin"] is system_ops.subprocess.DEVNULL and kwargs["start_new_session"] is (os.name != "nt")
        assert kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0" and kwargs["env"]["PIP_NO_INPUT"] == "1" and kwargs["env"]["GCM_INTERACTIVE"] == "never"
        assert kwargs["env"]["SSH_ASKPASS_REQUIRE"] == "never" and "GIT_ASKPASS" not in kwargs["env"] and "GIT_SSH_COMMAND" not in kwargs["env"]
        assert kwargs["stdout"] is system_ops.subprocess.PIPE and kwargs["stderr"] is system_ops.subprocess.PIPE and kwargs["encoding"] == "utf-8"
        return results.pop(0) if results else _Proc(0, "Already up to date.")

    return popen


def test_update_git_checkout_pull_dry_run_and_check(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(system_ops, "package_root", lambda: tmp_path)
    monkeypatch.setattr(system_ops.shutil, "which", lambda name: "/usr/bin/git" if name == "git" else None)
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    monkeypatch.delenv("GIT_ASKPASS", raising=False)
    calls = []
    monkeypatch.setattr(system_ops.subprocess, "Popen", _fake_run(calls, [_Proc(0, "Updating 1a2b..3c4d\nFast-forward\n 2 files changed", "")]))
    s = Session()
    assert system_ops.install_kind(tmp_path) == "git"
    res = s.run_op("help.update")
    assert calls == [{"argv": ["/usr/bin/git", "pull", "--ff-only"], "cwd": str(tmp_path)}]
    t = _table(res, "Update")
    assert t["columns"] == ["Step", "Status", "Output"] and t["rows"][0][0] == "Installed" and t["rows"][0][1] == lumen.__version__ and "git install" in t["rows"][0][2]
    assert t["rows"][1] == ["/usr/bin/git pull --ff-only", "exit 0", "Updating 1a2b..3c4d\nFast-forward\n 2 files changed"]
    assert "Restart Lumen" in res["message"] and "nothing was restarted" in res["message"] and "Fast-forward" in res["log"]
    # dry run: nothing executed
    calls.clear()
    res = s.run_op("help.update", {"dry_run": True})
    assert calls == [] and res["table"]["rows"][1][1] == "dry run" and "would run `/usr/bin/git pull --ff-only`" in res["message"] and "nothing was changed" in res["message"]
    # check only: fetch + count the commits behind
    monkeypatch.setattr(system_ops.subprocess, "Popen", _fake_run(calls, [_Proc(0, ""), _Proc(0, "3\n")]))
    res = s.run_op("help.update", {"check_only": True, "timeout": 30})
    assert [c["argv"] for c in calls] == [["/usr/bin/git", "fetch", "--quiet"], ["/usr/bin/git", "rev-list", "--count", "HEAD..@{upstream}"]]
    assert "3 commit(s) behind" in res["message"] and "nothing was installed" in res["message"]
    calls.clear()
    monkeypatch.setattr(system_ops.subprocess, "Popen", _fake_run(calls, [_Proc(0, ""), _Proc(0, "0")]))
    assert "up to date" in s.run_op("help.update", {"check_only": True})["message"]
    monkeypatch.setattr(system_ops.subprocess, "Popen", _fake_run(calls, [_Proc(0, ""), _Proc(128, "", "fatal: no upstream configured for branch 'main'")]))
    assert "could not compare" in s.run_op("help.update", {"check_only": True})["message"]
    monkeypatch.setattr(system_ops.subprocess, "Popen", _fake_run(calls, [_Proc(1, "", "fatal: unable to access")]))
    assert "git fetch failed" in s.run_op("help.update", {"check_only": True})["message"]
    # git told not to prompt: the credential error is reported at once instead of a terminal prompt stalling the server
    monkeypatch.setattr(system_ops.subprocess, "Popen", _fake_run(calls, [_Proc(128, "", "fatal: could not read Username for 'https://github.com': terminal prompts disabled")]))
    assert "terminal prompts disabled" in s.run_op("help.update", {"check_only": True})["message"]
    # a failing pull is reported, never raised
    monkeypatch.setattr(system_ops.subprocess, "Popen", _fake_run(calls, [_Proc(1, "", "error: Your local changes would be overwritten")]))
    res = s.run_op("help.update")
    assert res["message"].startswith("Update failed (exit 1: error: Your local changes") and res["table"]["rows"][1][1] == "exit 1"
    # git missing from PATH: no command runs
    calls.clear()
    monkeypatch.setattr(system_ops.shutil, "which", lambda name: None)
    res = s.run_op("help.update")
    assert calls == [] and "git was not found" in res["message"] and res["table"]["rows"][1][:2] == ["git", "not found"]


def test_update_pip_install_source_folder_and_run_command(tmp_path, monkeypatch):
    monkeypatch.setattr(system_ops, "package_root", lambda: tmp_path)
    calls = []
    monkeypatch.delenv("GIT_SSH_COMMAND", raising=False)
    monkeypatch.delenv("GIT_ASKPASS", raising=False)
    monkeypatch.setattr(system_ops.subprocess, "Popen", _fake_run(calls, [_Proc(0, "Successfully installed lumen-imaging-9.9")]))
    s = Session()
    assert system_ops.install_kind(tmp_path) == "pip"
    res = s.run_op("help.update")
    assert calls[0]["argv"] == [system_ops.sys.executable, "-m", "pip", "install", "--upgrade", "--no-input", "lumen-imaging"] and calls[0]["cwd"] is None
    assert "Restart Lumen" in res["message"] and "Successfully installed" in res["log"]
    calls.clear()
    assert "would run" in s.run_op("help.update", {"dry_run": True})["message"] and calls == []
    monkeypatch.setattr(system_ops, "pypi_version", lambda name=system_ops.PYPI_NAME, timeout=3.0: ("99.0", "ok"))
    res = s.run_op("help.update", {"check_only": True})
    assert calls == [] and "99.0 is available on PyPI" in res["message"] and res["table"]["rows"][1][:2] == ["PyPI lumen-imaging", "99.0"]
    monkeypatch.setattr(system_ops, "pypi_version", lambda name=system_ops.PYPI_NAME, timeout=3.0: (None, "no network"))
    assert "no network" in s.run_op("help.update", {"check_only": True})["message"]
    # a source folder without git is never touched
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "lumen-imaging"\n', encoding="utf-8")
    assert system_ops.install_kind(tmp_path) == "source"
    res = s.run_op("help.update")
    assert calls == [] and "without git" in res["message"] and "nothing was changed" in res["message"] and res["table"]["rows"][1][1] == "cannot update automatically"
    with pytest.raises(ValueError):
        system_ops.update_argv("source")
    assert REGISTRY.ops["help.update"].to_dict()["menu"] == "Help" and REGISTRY.ops["help.update"].requires_image is False


def test_run_command_real_subprocess(tmp_path):
    # the real runner (no monkeypatch here): list argv, no shell, UTF-8 output, errors reported instead of raised
    code, out = system_ops.run_command([system_ops.sys.executable, "-c", "import sys; print('héllo'); sys.stderr.write('warn'); sys.exit(3)"], tmp_path, 60)
    assert code == 3 and out == "héllo\nwarn"
    code, out = system_ops.run_command(["definitely-not-a-command-xyz"], tmp_path, 5)
    assert code is None and "not found" in out
    # stdin is closed, so a child that waits for input ends immediately instead of blocking the server
    code, out = system_ops.run_command([system_ops.sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"], tmp_path, 10)
    assert code == 0 and out == "''"
    # the non-interactive environment reaches the child; a timeout kills it and is reported, not raised
    code, out = system_ops.run_command([system_ops.sys.executable, "-c", "import os; print(os.environ['GIT_TERMINAL_PROMPT'], os.environ['PIP_NO_INPUT'])"], tmp_path, 10)
    assert (code, out) == (0, "0 1")
    t0 = system_ops.time.monotonic()
    code, out = system_ops.run_command([system_ops.sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 1)
    assert code is None and out == "timed out after 1 s" and system_ops.time.monotonic() - t0 < 10
    # the job's cancel flag kills a running command
    cancel = threading.Event()
    cancel.set()
    t0 = system_ops.time.monotonic()
    code, out = system_ops.run_command([system_ops.sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 60, cancel)
    assert (code, out) == (None, "cancelled") and system_ops.time.monotonic() - t0 < 10


def test_update_cancel_raises_operation_cancelled(tmp_path, monkeypatch):
    from lumen.core.registry import OperationCancelled
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(system_ops, "package_root", lambda: tmp_path)
    monkeypatch.setattr(system_ops.shutil, "which", lambda name: "/usr/bin/git" if name == "git" else None)
    monkeypatch.setattr(system_ops, "run_command", lambda argv, cwd=None, timeout=600.0, cancel=None: (None, "cancelled"))
    from lumen.core.registry import OpContext
    job = {"cancel": threading.Event()}  # the job record run_op hands to ctx; Cancel in the jobs panel sets the flag
    job["cancel"].set()
    with pytest.raises(OperationCancelled):
        system_ops.update(OpContext(session=Session(), image=None, job=job))
    with pytest.raises(OperationCancelled):
        system_ops.update(OpContext(session=Session(), image=None, job=job), check_only=True)


def test_options_proxy_rebuilds_urllib_opener(session, monkeypatch):
    import urllib.request
    # freeze urllib's global opener with no proxy configured (any earlier urlopen in the process does this)
    monkeypatch.setattr(urllib.request, "_opener", None)
    for v in _ENV_VARS:
        monkeypatch.delenv(v, raising=False)
    urllib.request.urlopen("data:,x").close()
    frozen = urllib.request._opener
    assert frozen is not None and not any(isinstance(h, urllib.request.ProxyHandler) for h in frozen.handle_open.get("https", []))
    # Proxy Settings… → later urlopen() calls (PyPI check, sample / URL downloads) route through the proxy
    session.run_op("edit.options_proxy", {"proxy": "proxy.lab:3128"})
    opener = urllib.request._opener
    assert opener is not frozen
    proxies = [h for h in opener.handle_open["https"] if isinstance(h, urllib.request.ProxyHandler)]
    assert proxies and proxies[0].proxies["https"] == "http://proxy.lab:3128"
    # … and directly again once it is cleared
    session.run_op("edit.options_proxy", {"proxy": ""})
    assert not any(isinstance(h, urllib.request.ProxyHandler) for h in urllib.request._opener.handle_open.get("https", []))


# ---------------------------------------------------------------- text windows and tables (lumen/core/ops/text_ops.py)
def test_text_window_lifecycle(tmp_path):
    from lumen.core.ops import text_ops
    s = Session()
    with pytest.raises(ValueError, match="No text window"):
        s.run_op("text.append", {"text": "x"})
    res = s.run_op("file.new_text", {"text": "first line\nsecond"})
    assert res["table"] == {"name": "Untitled.txt", "columns": ["Text"], "rows": [["first line"], ["second"]]} and "Untitled.txt" in s.tables
    assert s.run_op("file.new_text", {})["table"]["name"] == "Untitled-2.txt"  # a new window never replaces an open one
    assert s.run_op("file.new_text", {"name": "notes"})["table"]["name"] == "notes"
    assert s.run_op("file.new_text", {"name": "notes"})["table"]["name"] == "notes-2"
    res = s.run_op("text.append", {"name": "Untitled.txt", "text": "third\nfourth"})
    assert s.tables["Untitled.txt"]["rows"] == [["first line"], ["second"], ["third"], ["fourth"]] and "2 line(s) appended" in res["message"]
    s.run_op("text.append", {"text": "last"})  # empty name = the most recent text window
    assert s.tables["notes-2"]["rows"] == [["last"]]
    s.run_op("text.set", {"name": "Untitled.txt", "text": "ünïcode ✓\n"})
    assert s.tables["Untitled.txt"]["rows"] == [["ünïcode ✓"]]
    out = tmp_path / "out" / "text"
    res = s.run_op("file.save_text", {"name": "Untitled.txt", "path": str(out)})
    assert (tmp_path / "out" / "text.txt").read_text(encoding="utf-8") == "ünïcode ✓\n" and "Saved 'Untitled.txt'" in res["message"]
    res = s.run_op("text.to_log", {"name": "Untitled.txt"})
    assert s.log[-2]["msg"] == "--- Untitled.txt ---\nünïcode ✓" and "1 line(s)" in res["message"]  # [-1] is run_op's own "Text to Log (n ms)" entry
    with pytest.raises(ValueError, match="Give an output path"):
        s.run_op("file.save_text", {"name": "notes"})
    with pytest.raises(ValueError, match="No table named"):
        s.run_op("text.append", {"name": "nope", "text": "x"})
    assert text_ops.is_text_table(s.tables["notes"]) and text_ops.table_text(s.tables["Untitled.txt"]) == "ünïcode ✓"


def test_new_table_columns_and_text_helpers_on_plain_tables(tmp_path):
    from lumen.core.ops import text_ops
    s = Session()
    res = s.run_op("table.new", {"name": "Counts", "columns": "A, B,B\tC"})
    assert res["table"] == {"name": "Counts", "columns": ["A", "B", "B 2", "C"], "rows": []} and s.tables["Counts"]["rows"] == []
    assert s.run_op("table.new", {"name": "Counts"})["table"] == {"name": "Counts-2", "columns": [], "rows": []}
    s.tables["Counts"]["rows"].append([1, 2.5, None, "x"])
    with pytest.raises(ValueError, match="not a text window"):
        s.run_op("text.append", {"name": "Counts", "text": "x"})
    res = s.run_op("file.save_text", {"name": "Counts", "path": str(tmp_path / "counts.tsv")})
    assert (tmp_path / "counts.tsv").read_text(encoding="utf-8") == "A\tB\tB 2\tC\n1\t2.5\t\tx\n"
    assert text_ops.parse_columns(["x", "", "x"]) == ["x", "x 2"] and text_ops.parse_columns("") == []
    for oid, menu in (("file.new_text", "File/New"), ("table.new", "File/New"), ("text.append", "File/Text"), ("text.set", "File/Text"), ("file.save_text", "File/Text"), ("text.to_log", "File/Text")):
        assert REGISTRY.ops[oid].to_dict()["menu"] == menu and REGISTRY.ops[oid].requires_image is False, oid
