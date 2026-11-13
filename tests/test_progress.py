"""Progress reporting and cancellation of long operations.

Session.run_op keeps a job record per running command (``session.jobs``), broadcasts throttled
``{"type": "progress", "job": {...}}`` events to the listeners and honours ``POST /api/jobs/{id}/cancel``
through ``OpContext.check_cancel()``; a cancelled op leaves the image unchanged and answers 409.
"""
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core import ops  # noqa: F401  (registers the real commands)
from lumen.core.image import LumenImage
from lumen.core.registry import REGISTRY, OpContext, OperationCancelled, Param, op
from lumen.core.session import Session
from lumen.server.app import create_app

# A plane op that sleeps per plane: slow enough for the throttled progress stream to carry several
# events and for a cancel request to land mid-way (registered once; the registry is process-global).
if "test.slow_invert" not in REGISTRY.ops:
    @op("test.slow_invert", "Slow Invert (test)", "Test", kind="plane", preview=False, params=[Param("delay", "float", 0.02, min=0)])
    def _slow_invert(plane, ctx, delay=0.02):
        time.sleep(delay)
        return 255 - plane


def _stack(nz: int = 20) -> LumenImage:
    rng = np.random.default_rng(0)
    return LumenImage(rng.integers(0, 200, (1, nz, 1, 16, 16)).astype(np.uint8), name="stack", axes="TZCYX")


def _progress_listener(s: Session) -> list[dict]:
    jobs: list[dict] = []
    s.listeners.append(lambda e: jobs.append(e["job"]) if e.get("type") == "progress" else None)
    return jobs


def _cancel_when_running(s: Session) -> threading.Thread:
    """A thread that cancels the first job as soon as it appears (the op holds the session lock meanwhile)."""
    def run() -> None:
        for _ in range(400):
            with s._jobs_lock:
                ids = list(s.jobs)
            if ids:
                assert s.cancel_job(ids[0])
                return
            time.sleep(0.005)
    th = threading.Thread(target=run)
    th.start()
    return th


def test_plane_op_emits_increasing_progress():
    s = Session()
    img = s.add_image(_stack(20))
    jobs = _progress_listener(s)
    t0 = time.time()
    res = s.run_op("test.slow_invert", {"delay": 0.02}, image_id=img.id)
    dt = time.time() - t0
    assert "elapsed_ms" in res and img.data[0, 0, 0, 0, 0] == 255 - _stack(20).data[0, 0, 0, 0, 0]
    assert len(jobs) >= 3
    first, last = jobs[0], jobs[-1]
    assert first["op"] == "test.slow_invert" and first["name"] == "Slow Invert (test)" and first["fraction"] == 0.0
    assert first["cancelled"] is False and "done" not in first and isinstance(first["started"], float)
    assert last["done"] is True and last["status"] == "ok" and last["cancelled"] is False and last["fraction"] == 1.0
    assert {j["id"] for j in jobs} == {first["id"]}
    fractions = [j["fraction"] for j in jobs]
    assert fractions == sorted(fractions)
    assert len(set(fractions)) >= 5  # 20 planes x 20 ms = 400 ms: several throttled updates between start and end
    assert any(j["message"].startswith("Plane ") for j in jobs)
    # throttled to <= 20 events/s (start and end events excluded; the first update is always let through)
    assert len(jobs) - 2 <= dt * 20 + 1
    assert jobs[1]["message"] == "Plane 1/20"
    assert s.jobs == {}  # cleared at the end
    assert s.histories[img.id].labels()["undo"] == ["Slow Invert (test)"]


def test_progress_is_noop_outside_a_run_and_in_previews():
    s = Session()
    img = s.add_image(_stack(4))
    ctx = OpContext(session=s, image=img)
    ctx.progress(0.5, "nothing happens")
    ctx.check_cancel()
    jobs = _progress_listener(s)
    s.run_op("filters.gaussian", {"sigma": 1}, image_id=img.id, preview=True, record=False)
    assert jobs == [] and s.jobs == {}


def test_cancel_from_thread_restores_image():
    s = Session()
    img = s.add_image(_stack(20))
    before = img.data.copy()
    rev0 = img.rev
    jobs = _progress_listener(s)
    th = _cancel_when_running(s)
    with pytest.raises(OperationCancelled):
        s.run_op("test.slow_invert", {"delay": 0.02}, image_id=img.id)
    th.join(timeout=5)
    assert np.array_equal(img.data, before) and img.rev == rev0
    assert {k: s.histories[img.id].labels()[k] for k in ("undo", "redo")} == {"undo": [], "redo": []}
    assert s.jobs == {}
    assert jobs[-1]["done"] is True and jobs[-1]["status"] == "cancelled" and jobs[-1]["cancelled"] is True
    assert jobs[-1]["fraction"] < 1.0
    assert not any("test.slow_invert" in line for line in s.macro)
    assert s.cancel_job("nope") is False


def test_cancel_endpoint_409_and_jobs_listing():
    s = Session()
    img = s.add_image(_stack(30))
    before = img.data.copy()
    c = TestClient(create_app(s), raise_server_exceptions=False)
    assert c.get("/api/jobs").json() == []
    result: dict = {}

    def runner() -> None:
        result["r"] = c.post("/api/ops/test.slow_invert/run", json={"params": {"delay": 0.03}, "image_id": img.id})

    th = threading.Thread(target=runner)
    th.start()
    job = None
    for _ in range(400):  # GET /api/jobs answers while the op holds the session lock
        listed = c.get("/api/jobs").json()
        if listed:
            job = listed[0]
            break
        time.sleep(0.005)
    assert job is not None, "the running job was not listed"
    assert job["op"] == "test.slow_invert" and job["name"] == "Slow Invert (test)" and job["cancelled"] is False
    assert 0.0 <= job["fraction"] < 1.0 and job["elapsed"] >= 0 and set(job) >= {"id", "op", "name", "started", "fraction", "message"}
    r = c.post(f"/api/jobs/{job['id']}/cancel")
    assert r.status_code == 200 and r.json() == {"ok": True, "id": job["id"]}
    th.join(timeout=15)
    r = result["r"]
    assert r.status_code == 409 and r.json() == {"error": "Cancelled"}
    assert np.array_equal(img.data, before)
    assert c.get("/api/jobs").json() == []
    assert c.post(f"/api/jobs/{job['id']}/cancel").status_code == 404
    assert {k: v for k, v in c.get("/api/state").json()["history"][img.id].items() if k in ("undo", "redo")} == {"undo": [], "redo": []}
    # the session is usable afterwards and an ordinary error is still a 400
    assert c.post("/api/ops/test.slow_invert/run", json={"params": {"delay": 0.0}, "image_id": img.id}).status_code == 200
    assert c.post("/api/ops/filters.gaussian/run", json={"params": {"sigma": 3}, "image_id": "missing"}).status_code == 400


def _folder(tmp_path, n: int):
    import tifffile
    for i in range(n):
        tifffile.imwrite(str(tmp_path / f"im_{i}.tif"), np.full((8, 8), 10 * i, np.uint8))
    return tmp_path


def test_batch_reports_per_file_progress(tmp_path):
    folder = _folder(tmp_path, 3)
    s = Session()
    jobs = _progress_listener(s)
    script = 'import time\ntime.sleep(0.06)\nlumen.run("math.add", value=1)'
    res = s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"\.tif$", "output_format": "none", "script": script, "save_results": False})
    assert "3 ok, 0 failed" in res["message"]
    # the commands the script runs are nested: they report into the batch job, never start their own
    assert {j["op"] for j in jobs} == {"batch.process_folder"} and len({j["id"] for j in jobs}) == 1
    msgs = [j["message"] for j in jobs]
    for i in range(3):
        assert any(m == f"im_{i}.tif ({i + 1}/3)" for m in msgs), (i, msgs)
    fr = [j["fraction"] for j in jobs]
    assert fr == sorted(fr) and jobs[-1]["done"] is True and jobs[-1]["status"] == "ok"
    assert s.jobs == {}


def test_batch_cancel_restores_session(tmp_path):
    folder = _folder(tmp_path, 8)
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    active_before = s.active_id
    jobs = _progress_listener(s)
    th = _cancel_when_running(s)
    with pytest.raises(OperationCancelled):
        s.run_op("batch.process_folder", {"input_folder": str(folder), "pattern": r"\.tif$", "output_format": "none",
                                          "script": 'import time\ntime.sleep(0.05)\nlumen.run("math.add", value=1)', "save_results": False})
    th.join(timeout=5)
    assert len(s.images) == 1 and s.active_id == active_before  # per-file images closed, previous image re-selected
    assert s.jobs == {} and jobs[-1]["status"] == "cancelled"
    assert not any("batch.process_folder" in line for line in s.macro)


def test_long_ops_report_progress():
    """The instrumented ops run (and report) on small inputs: align_stack / elastic_align / drift / project3d / skeleton / tracking."""
    s = Session()
    rng = np.random.default_rng(1)
    base = np.zeros((40, 40), np.uint8)
    base[10:20, 12:24] = 200
    zstack = np.stack([np.roll(base, k, axis=1) for k in range(4)])[None, :, None]  # (T=1, Z=4, C=1)
    img = s.add_image(LumenImage(zstack.astype(np.uint8), name="z", axes="TZCYX"))
    tseries = np.stack([np.roll(base, k, axis=0) for k in range(3)])[:, None, None]  # (T=3, Z=1, C=1)
    img_t = s.add_image(LumenImage(tseries.astype(np.uint8), name="t", axes="TZCYX"))
    jobs = _progress_listener(s)

    def ran(op_id: str) -> list[dict]:
        mine = [j for j in jobs if j["op"] == op_id]
        assert mine and mine[-1]["done"] is True and mine[-1]["status"] == "ok", (op_id, mine[-1:])
        return mine

    s.run_op("registration.align_stack", {"axis": "Z", "mode": "Translation"}, image_id=img.id)
    ran("registration.align_stack")
    s.run_op("registration.elastic_align", {"axis": "Z", "num_warp": 1, "num_iter": 2}, image_id=img.id)
    ran("registration.elastic_align")
    s.run_op("registration.drift_correct", {}, image_id=img_t.id)
    ran("registration.drift_correct")
    s.run_op("stacks.project3d", {"angle_step": 90, "total_rotation": 180, "max_size": 32}, image_id=img.id)
    ran("stacks.project3d")
    s.run_op("analyze.skeleton", {"all_planes": True, "show": "Nothing"}, image_id=img.id)
    ran("analyze.skeleton")
    spots = rng.integers(0, 2, (3, 1, 1, 40, 40)).astype(np.uint8) * 50
    for t in range(3):
        spots[t, 0, 0, 20, 10 + t] = 255
    img_s = s.add_image(LumenImage(spots, name="spots", axes="TZCYX"))
    s.run_op("tracking.track", {"radius": 1.5, "threshold": 50.0}, image_id=img_s.id)
    mine = ran("tracking.track")
    # three tiny frames detect faster than the throttle window: the first update and the final event carry the messages
    assert mine[1]["message"] == "Detecting spots in frame 1/3"
    assert mine[-1]["message"].startswith(("Detecting spots", "Linking frame", "Linking track segments"))
