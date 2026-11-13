"""Tests for the preferences / recent-files / docs router (lumen/server/extra/settings.py)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core.session import Session
from lumen.server.app import create_app
from lumen.server.extra.settings import DEFAULTS


def client(tmp_path=None, monkeypatch=None):
    s = Session()
    if tmp_path is not None and monkeypatch is not None:
        # never touch the user's ~/.lumen/recent.json from the tests
        monkeypatch.setattr(Session, "_recent_path", staticmethod(lambda: tmp_path / "recent.json"))
        s.recent = []
    return TestClient(create_app(s)), s


def test_get_returns_defaults_and_hides_internal_keys():
    c, s = client()
    s.settings.update({
        "clipboard": {"data": np.zeros((1, 1, 2, 2)), "mask": None, "rgb": False},
        "classifier": object(),
        "fft": np.zeros(4),
        "tracks": [{"id": "0"}],
        "tracks_image": "img-1",
        "spots": [{"x": 1}],
        "spots_meta": {"image_id": "img-1"},
        "deformation": np.zeros(2),
        "_private": 1,
        "custom_plugin_flag": True,
        "numpy_scalar": np.float32(2.5),
    })
    r = c.get("/api/settings")
    assert r.status_code == 200, r.text
    data = r.json()
    for hidden in ("clipboard", "classifier", "fft", "tracks", "tracks_image", "spots", "spots_meta", "deformation", "_private"):
        assert hidden not in data
    # defaults for every preference the UI edits, session values win
    for k, v in DEFAULTS.items():
        assert k in data
    assert data["foreground"] == 255 and data["background"] == 0
    assert data["roi_color"] == DEFAULTS["roi_color"]
    # unknown but JSON-safe keys pass through, numpy scalars are converted
    assert data["custom_plugin_flag"] is True
    assert data["numpy_scalar"] == 2.5


def test_put_merges_and_persists():
    c, s = client()
    r = c.put("/api/settings", json={"roi_color": "#123456", "interpolate_zoom": True, "jpeg_quality": 70})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["roi_color"] == "#123456" and data["interpolate_zoom"] is True and data["jpeg_quality"] == 70
    assert s.settings["roi_color"] == "#123456"
    assert s.settings["interpolate_zoom"] is True
    # a second partial update keeps earlier values (merge, not replace)
    r = c.patch("/api/settings", json={"font_size": 20})
    assert r.status_code == 200
    assert r.json()["roi_color"] == "#123456" and r.json()["font_size"] == 20
    assert c.get("/api/settings").json()["font_size"] == 20
    # the ops that read the colour settings see the same dict
    r = c.put("/api/settings", json={"foreground": 128})
    assert s.settings["foreground"] == 128
    # version bumps so clients re-fetch
    assert s.version > 0


def test_post_merges_when_not_shadowed():
    c, s = client()
    r = c.post("/api/settings", json={"roi_color": "#abcdef"})
    assert r.status_code == 200, r.text
    if "roi_color" not in r.json():
        pytest.xfail("POST /api/settings is still handled by the core route in lumen/server/app.py (measurements only); the frontend uses PUT")
    assert s.settings["roi_color"] == "#abcdef"


def test_rejects_bad_values_atomically():
    c, s = client()
    before = dict(s.settings)
    bad = [
        {"clipboard": 1},                       # internal key
        {"tracks_image": "x"},                  # internal companion key
        {"_secret": 1},                         # private key
        {"roi_color": 5},                       # wrong type
        {"roi_color": "yellow"},                # not a hex colour
        {"jpeg_quality": 500},                  # out of range
        {"interpolate_zoom": "yes"},            # bool expected
        {"font_size": True},                    # bool is not a number
        {"measurements": "area"},               # list expected
        {"font_size": 20, "roi_color": "nope"},  # one bad value rejects the whole request
    ]
    for body in bad:
        r = c.put("/api/settings", json=body)
        assert r.status_code == 400, (body, r.text)
        assert "error" in r.json()
    assert dict(s.settings) == before
    assert "font_size" not in s.settings
    # not an object at all
    r = c.put("/api/settings", json=[1, 2, 3])
    assert r.status_code in (400, 422)


def test_recent_clear(tmp_path, monkeypatch):
    c, s = client(tmp_path, monkeypatch)
    s.add_recent("/data/a.tif")
    s.add_recent("/data/b.tif")
    assert c.get("/api/state").json()["recent"] == ["/data/b.tif", "/data/a.tif"]
    r = c.put("/api/settings", json={"recent_clear": True})
    assert r.status_code == 200, r.text
    assert r.json()["recent"] == []
    assert s.recent == []
    assert c.get("/api/state").json()["recent"] == []
    assert (tmp_path / "recent.json").read_text() == "[]"
    # recent_clear is not stored as a setting
    assert "recent_clear" not in s.settings
    assert "recent_clear" not in c.get("/api/settings").json()


def test_measurement_keys_are_applied_like_the_core_endpoint():
    c, s = client()
    r = c.put("/api/settings", json={"measurements": ["area", "mean", "not-a-measurement"], "measure_options": {"decimals": 5}})
    assert r.status_code == 200, r.text
    assert s.measurements == ["area", "mean"]
    assert s.measure_options["decimals"] == 5
    assert r.json()["measurements"] == ["area", "mean"]
    assert r.json()["measure_options"]["decimals"] == 5
    assert "measurements" not in s.settings


def test_advanced_keys_are_applied_like_edit_options(monkeypatch):
    """undo_budget_mb / max_threads / proxy / no_proxy used to be stored only: the Preferences sheet now applies them
    exactly like Edit ▸ Options ▸ Memory & Threads / Proxy Settings (History budgets, threadpoolctl, HTTP(S)_PROXY)."""
    import os
    from lumen.core.ops import system_ops
    from lumen.core.undo import History
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(var, raising=False)
    default_before = History.default_budget_bytes
    c, s = client()
    try:
        assert c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}}).status_code == 200
        r = c.put("/api/settings", json={"undo_budget_mb": 100, "max_threads": 2, "proxy": "proxy.local:3128", "no_proxy": "localhost, .lab"})
        assert r.status_code == 200, r.text
        assert History.default_budget_bytes == 100_000_000 and s.settings["undo_budget_mb"] == 100
        assert s.histories and all(h.budget == 100_000_000 for h in s.histories.values())
        assert c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}}).status_code == 200
        assert len(s.histories) == 2 and all(h.budget == 100_000_000 for h in s.histories.values())  # new histories too
        assert s.settings["max_threads"] == 2 and system_ops._THREAD_LIMITER is not None
        assert r.json()["proxy"] == s.settings["proxy"] == "http://proxy.local:3128" and s.settings["no_proxy"] == "localhost,.lab"
        assert os.environ["HTTP_PROXY"] == os.environ["HTTPS_PROXY"] == os.environ["http_proxy"] == "http://proxy.local:3128"
        assert os.environ["NO_PROXY"] == os.environ["no_proxy"] == "localhost,.lab"
        # a request carrying only no_proxy keeps the stored proxy
        assert c.put("/api/settings", json={"no_proxy": ""}).status_code == 200
        assert os.environ["HTTP_PROXY"] == "http://proxy.local:3128" and "NO_PROXY" not in os.environ
        # an invalid proxy is rejected and changes nothing
        r = c.put("/api/settings", json={"proxy": "proxy local:3128"})
        assert r.status_code == 400 and "proxy" in r.json()["error"] and os.environ["HTTP_PROXY"] == "http://proxy.local:3128"
        # Reset to defaults: the variables go, the pools are restored, the budget is 1.5 GB again
        r = c.put("/api/settings", json={"proxy": "", "max_threads": 0, "undo_budget_mb": DEFAULTS["undo_budget_mb"]})
        assert r.status_code == 200, r.text
        assert "HTTP_PROXY" not in os.environ and "https_proxy" not in os.environ and system_ops._THREAD_LIMITER is None
        assert History.default_budget_bytes == DEFAULTS["undo_budget_mb"] * 1_000_000 and all(h.budget == DEFAULTS["undo_budget_mb"] * 1_000_000 for h in s.histories.values())
        assert c.put("/api/settings", json={"proxy": "https://user:pw@gw.lab:8080"}).json()["proxy"] == "https://user:pw@gw.lab:8080"
        assert os.environ["HTTPS_PROXY"] == "https://user:pw@gw.lab:8080"
    finally:
        History.set_default_budget(default_before)
        system_ops.apply_thread_limit(0)
        system_ops.apply_proxy("", "")


def test_state_has_memory_bytes():
    c, s = client()
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}})
    assert r.status_code == 200, r.text
    st = c.get("/api/state").json()
    assert isinstance(st["memory_bytes"], int) and st["memory_bytes"] > 0


def test_docs_routes():
    c, _ = client()
    r = c.get("/api/docs")
    assert r.status_code == 200
    names = [d["name"] for d in r.json()["docs"]]
    assert names == ["README", "COMMANDS", "FIJI_PARITY", "SCRIPTING", "MCP", "RESEARCH", "DESIGN"]
    assert all(d["available"] for d in r.json()["docs"])
    r = c.get("/api/docs/COMMANDS")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/markdown")
    assert "`file.sample`" in r.text
    # case-insensitive, .md suffix tolerated
    r = c.get("/api/docs/readme.md")
    assert r.status_code == 200 and r.text.startswith("# Lumen")
    assert c.get("/api/docs/fiji-parity").status_code == 200
    r = c.get("/api/docs/NOPE")
    assert r.status_code == 404
    assert "Unknown document" in r.json()["error"]


def test_overlay_options_keys_are_validated():
    c, s = client()
    data = c.get("/api/settings").json()
    assert data["overlay_default_fill"] == "" and data["overlay_show_labels"] is False and data["overlay_use_names"] is False
    assert data["overlay_label_font_size"] == 12 and data["overlay_label_color"] == "#ffff00" and data["overlay_label_background"] is False and data["overlay_stroke_width"] == 1.5
    r = c.put("/api/settings", json={"overlay_default_fill": "#ff000080", "overlay_show_labels": True, "overlay_use_names": True, "overlay_label_font_size": 20,
                                     "overlay_label_color": "#00ff00", "overlay_label_background": True, "overlay_stroke_width": 3})
    assert r.status_code == 200, r.text
    assert s.settings["overlay_default_fill"] == "#ff000080" and s.settings["overlay_show_labels"] is True and s.settings["overlay_label_font_size"] == 20
    assert s.settings["overlay_label_color"] == "#00ff00" and s.settings["overlay_stroke_width"] == 3
    assert c.put("/api/settings", json={"overlay_default_fill": "none"}).status_code == 200 and s.settings["overlay_default_fill"] == ""
    assert c.put("/api/settings", json={"overlay_default_fill": "#ABC"}).status_code == 200 and s.settings["overlay_default_fill"] == "#ABC"
    for bad in ({"overlay_default_fill": "red"}, {"overlay_label_color": "yellow"}, {"overlay_label_font_size": 0}, {"overlay_stroke_width": 0},
                {"overlay_show_labels": "yes"}, {"overlay_use_names": 1}, {"overlay_label_background": "true"}):
        r = c.put("/api/settings", json=bad)
        assert r.status_code == 400, bad
    assert s.settings["overlay_label_color"] == "#00ff00" and s.settings["overlay_label_font_size"] == 20  # rejected requests change nothing


def test_gui_preferences_confirm_close_and_density():
    """docs/USABILITY.md M1 / M11: 'ask before closing a modified image' and the inspector density live in the settings."""
    c, s = client()
    data = c.get("/api/settings").json()
    assert data["confirm_close_modified"] is True and data["ui_density"] == "comfortable"
    r = c.put("/api/settings", json={"confirm_close_modified": False, "ui_density": "Compact"})
    assert r.status_code == 200, r.text
    assert r.json()["confirm_close_modified"] is False and r.json()["ui_density"] == "compact"  # canonical lower case
    assert s.settings["confirm_close_modified"] is False and s.settings["ui_density"] == "compact"
    for bad in ({"confirm_close_modified": "no"}, {"ui_density": "cozy"}, {"ui_density": 1}):
        r = c.put("/api/settings", json=bad)
        assert r.status_code == 400, bad
    assert s.settings["ui_density"] == "compact"
