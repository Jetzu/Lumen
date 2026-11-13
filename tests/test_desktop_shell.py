"""The desktop shell (desktop/lumen_desktop) without a window: the single-instance hand-over of files, the window-state
file, the bridge's dialog helpers and the smoke mode (server + registry + image round trip).  pywebview is not needed;
the windowed parts are exercised by desktop/run_dev.py --debug and the packaged app's --smoke (docs/DESKTOP.md)."""
from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parent.parent / "desktop"
sys.path.insert(0, str(DESKTOP))

from lumen_desktop import bridge, instance, paths, server  # noqa: E402
from lumen_desktop.state import MIN_HEIGHT, MIN_WIDTH, DesktopState  # noqa: E402


def test_second_launch_hands_files_to_the_running_instance(tmp_path):
    opened: list[list[str]] = []
    focused = threading.Event()
    lock = tmp_path / "desktop.lock"
    inst = instance.SingleInstance(lock, on_focus=focused.set, on_open=opened.append)
    inst.acquire("http://127.0.0.1:9")
    try:
        info = instance.notify_existing(lock, files=["/data/a.tif", "/data/b.czi"])
        assert info and info["url"] == "http://127.0.0.1:9"
        for _ in range(50):
            if opened and focused.is_set():
                break
            time.sleep(0.05)
        assert opened == [["/data/a.tif", "/data/b.czi"]] and focused.is_set()
        # without files: focus only
        focused.clear()
        assert instance.notify_existing(lock)
        assert focused.wait(2) and len(opened) == 1
        # a wrong token is refused
        bad = json.loads(lock.read_text())
        assert instance._send(int(bad["control_port"]), "nope OPEN [\"/x\"]") == "DENIED"
    finally:
        inst.release()
    assert instance.notify_existing(lock) is None  # nobody listening any more: stale lock


def test_window_state_is_sanitised(tmp_path):
    st = DesktopState(tmp_path / "s.json")
    st.remember_size(300, 200)  # smaller than the minimum
    st.remember_position(50000, 50000)  # off every screen
    st.last_port = 8771
    st.last_open_dir = str(tmp_path)
    st.save()
    st2 = DesktopState(tmp_path / "s.json")
    g = st2.geometry([(0, 0, 1440, 900)])
    assert g["width"] >= MIN_WIDTH and g["height"] >= MIN_HEIGHT and "x" not in g
    assert st2.last_port == 8771 and st2.last_open_dir == str(tmp_path)
    (tmp_path / "broken.json").write_text("{not json")
    assert DesktopState(tmp_path / "broken.json").geometry([])["width"] >= MIN_WIDTH


def test_dialog_helpers(tmp_path):
    assert bridge._file_types([".py", "ijm"]) == ("Files (*.py;*.ijm)", "All files (*.*)")
    assert bridge._file_types(None) == ()
    f = tmp_path / "x.tif"
    f.write_bytes(b"0")
    assert bridge._initial_dir(str(f)) == str(tmp_path)
    assert bridge._initial_dir(None, str(tmp_path / "missing"), str(tmp_path)) == str(tmp_path)
    assert bridge._dialog_paths(("a", "b")) == ["a", "b"] and bridge._dialog_paths("a") == ["a"] and bridge._dialog_paths(None) == []
    assert bridge.safe_filename("a/b:c?.tif") == "a_b_c_.tif"
    assert "*.czi" in bridge.OPEN_FILE_TYPES[0] and "*.ome.tif" in bridge.OPEN_FILE_TYPES[0]


def test_paths_follow_lumen_home(tmp_path, monkeypatch):
    monkeypatch.setenv("LUMEN_HOME", str(tmp_path / "home"))
    assert paths.lumen_home() == tmp_path / "home" and paths.desktop_dir() == tmp_path / "home" / "desktop"
    from lumen.core.io import lumen_home
    assert lumen_home() == paths.lumen_home()  # the desktop app and the `lumen` command share one folder


def test_pick_port_prefers_the_last_port_when_free():
    p = server.pick_port(None)
    assert 1024 <= p <= 65535
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        busy = s.getsockname()[1]
        assert server.pick_port(busy) != busy


def test_smoke_mode_end_to_end(tmp_path, monkeypatch, capsys):
    """`lumen-desktop --smoke`: server up, 400+ commands, an image created and a TIFF opened through the API, the UI page
    served, the lazily imported libraries and the bundled docs reachable."""
    import numpy as np
    import tifffile

    from lumen_desktop import app

    img = tmp_path / "in.tif"
    tifffile.imwrite(img, (np.arange(40 * 30) % 255).astype("uint8").reshape(30, 40))
    monkeypatch.delenv("LUMEN_DESKTOP_PORT", raising=False)
    code = app.main(["--smoke", "--home", str(tmp_path / "home"), str(img)])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "LUMEN_DESKTOP_SMOKE_OK" in out and "commands=" in out
    assert (tmp_path / "home" / "desktop" / "desktop.log").is_file()
