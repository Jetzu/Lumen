import csv

import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.session import Session

STARTS = np.array([[20, 20], [100, 30], [30, 100], [90, 90], [64, 64]], float)  # x, y
VEL = np.array([[3, 0], [-2, 2], [0, 3], [-3, -1], [1, 1]], float)  # px / frame


def moving_spots(nt=10, size=128, amp=200.0, sigma=2.0, noise=5.0, seed=0, drop=None):
    """Gaussian spots moving linearly; ``drop`` = {(spot_index, frame), ...} removes single spots."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    data = np.zeros((nt, size, size), np.float32)
    for t in range(nt):
        f = np.full((size, size), 10.0, np.float32)
        for k, (s, v) in enumerate(zip(STARTS, VEL)):
            if drop and (k, t) in drop:
                continue
            x, y = s + v * t
            f += amp * np.exp(-(((xx + 0.5) - x) ** 2 + ((yy + 0.5) - y) ** 2) / (2 * sigma ** 2))
        data[t] = f + rng.normal(0, noise, (size, size))
    return data


@pytest.fixture
def session():
    s = Session()
    img = LumenImage(moving_spots(), name="moving", axes="TYX")
    img.calibration.frame_interval = 2.0
    img.calibration.time_unit = "s"
    s.add_image(img)
    return s


def test_track_recovers_tracks_and_speeds(session):
    img = session.active
    out = session.run_op("tracking.track", {"radius": 2.5, "threshold": 0, "max_displacement": 6, "gap_closing_frames": 2})
    assert "5 tracks" in out["message"]
    tracks = session.settings["tracks"]
    assert len(tracks) == 5 and all(len(t["spots"]) == 10 for t in tracks)
    # every track should match one of the ground-truth velocities (speed in px/s with a 2 s interval)
    expected = sorted(np.hypot(VEL[:, 0], VEL[:, 1]) / 2.0)
    got = sorted(t["stats"]["Mean speed"] for t in tracks)
    assert np.allclose(got, expected, atol=0.1)
    assert all(t["stats"]["Confinement ratio"] > 0.98 for t in tracks)
    assert all(t["stats"]["Duration (s)"] == 18.0 for t in tracks)
    # overlay format
    tr = [o for o in img.overlay if o["type"] == "track"]
    assert len(tr) == 5
    assert tr[0]["name"] == "Track 0" and tr[0]["color"].startswith("#") and len(tr[0]["points"]) == 10
    assert all(len(p) == 3 for p in tr[0]["points"])
    assert [p[2] for p in tr[0]["points"]] == list(range(10))
    assert len({o["color"] for o in tr}) == 5
    # the first track starts at (20, 20) and moves +3 px/frame in x (sub-pixel localisation)
    first = min(tr, key=lambda o: o["points"][0][0])
    assert abs(first["points"][0][0] - 20) < 0.3 and abs(first["points"][0][1] - 20) < 0.3
    assert abs(first["points"][-1][0] - 47) < 0.3
    # track summaries go to the Tracks table only (with the overlay colour as a hex Color column);
    # the shared Results table stays untouched
    assert not session.results.rows
    tab = session.tables["Tracks"]
    assert tab["columns"][0] == "Label" and "Color" in tab["columns"] and len(tab["rows"]) == 5
    row = dict(zip(tab["columns"], tab["rows"][0]))
    assert row["Label"] == "Track 0" and row["N spots"] == 10 and row["Color"] == tr[0]["color"] and row["Color"].startswith("#")


def test_detect_then_link(session):
    img = session.active
    out = session.run_op("tracking.detect_spots", {"radius": 2.5, "threshold": 30})
    assert "50 spots" in out["message"]
    spots = session.settings["spots"]
    assert len(spots) == 50 and {"t", "x", "y", "z", "r", "quality", "intensity"} <= set(spots[0])
    assert all(sp["quality"] >= 30 for sp in spots)
    # spot rows: Label first, calibrated and pixel coordinates
    row = session.results.rows[0]
    assert list(row)[0] == "Label" and {"Frame", "X", "Y", "X (px)", "Y (px)", "Radius", "Quality", "Mean intensity"} <= set(row)
    # one point ROI per frame with the frame position kept
    rois = session.roi_manager.rois
    assert len(rois) == 10
    assert rois[3]["name"] == "Spots t3" and rois[3]["position"] == {"t": 3, "z": 0, "c": 0} and len(rois[3]["points"]) == 5
    assert rois[3]["image_id"] == img.id
    session.run_op("tracking.link_spots", {"max_displacement": 6, "gap_closing_frames": 0})
    assert len(session.settings["tracks"]) == 5
    assert len(session.results.rows) == 50  # linking adds no rows to the Results table
    assert len(session.tables["Tracks"]["rows"]) == 5
    # clear tracks removes the overlays only
    session.run_op("tracking.clear_tracks", {})
    assert not [o for o in img.overlay if o["type"] == "track"]
    assert "tracks" not in session.settings and session.settings["spots"]


def test_link_requires_detection():
    s = Session()
    s.add_image(LumenImage(moving_spots(nt=3), name="m", axes="TYX"))
    with pytest.raises(ValueError, match="Detect Spots first"):
        s.run_op("tracking.link_spots", {})
    with pytest.raises(ValueError, match="Link Spots"):
        s.run_op("tracking.filter_tracks", {})


def test_track_spot_rows_only_on_request(session):
    session.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6})
    assert not session.results.rows
    session.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6, "add_to_results": True})
    rows = session.results.rows
    assert len(rows) == 50 and all("Frame" in r and "N spots" not in r for r in rows)


def test_filter_tracks(session):
    img = session.active
    session.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6})
    assert len(session.settings["tracks"]) == 5
    # speeds in px/s (2 s interval): 1.5, 1.41, 1.5, 1.58, 0.71 → only the slow diagonal track passes max_speed 1
    out = session.run_op("tracking.filter_tracks", {"max_speed": 1.0})
    assert "Kept 1 of 5" in out["message"] and "4 removed" in out["message"]
    assert len(session.settings["tracks"]) == 1 and len([o for o in img.overlay if o["type"] == "track"]) == 1
    tab = session.tables["Tracks"]
    assert len(tab["rows"]) == 1
    row = dict(zip(tab["columns"], tab["rows"][0]))
    assert row["Max speed"] < 1.0 and row["Color"] == session.settings["tracks"][0]["color"]
    # net displacement over 9 frames: |v| · 9 = 27, 25.5, 27, 28.5, 12.7 px
    session.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6})
    out = session.run_op("tracking.filter_tracks", {"min_displacement": 20})
    assert "Kept 4 of 5" in out["message"]
    kept = session.settings["tracks"]
    assert len(kept) == 4 and all(t["stats"]["Net displacement"] >= 20 for t in kept)
    assert {o["id"] for o in img.overlay if o["type"] == "track"} == {f"track-{t['id']}" for t in kept}
    assert [r[0] for r in session.tables["Tracks"]["rows"]] == [t["name"] for t in kept]
    # filters compose on the remaining tracks; durations are 9 frames
    out = session.run_op("tracking.filter_tracks", {"min_duration": 9})
    assert "Kept 4 of 4" in out["message"]
    out = session.run_op("tracking.filter_tracks", {"min_duration": 10})
    assert "Kept 0 of 4" in out["message"] and not [o for o in img.overlay if o["type"] == "track"]
    assert session.tables["Tracks"]["rows"] == [] and "Color" in session.tables["Tracks"]["columns"]
    session.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6})
    assert "Kept 5 of 5" in session.run_op("tracking.filter_tracks", {"min_length": 10})["message"]
    assert "Kept 0 of 5" in session.run_op("tracking.filter_tracks", {"min_length": 11})["message"]
    # no filter: everything is kept and the Results table was never touched
    session.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6})
    assert "Kept 5 of 5" in session.run_op("tracking.filter_tracks", {})["message"]
    assert not session.results.rows


def test_dense_spots_hint():
    img, n = grid_spots(256, 8, noise=20)  # 1024 spots in a single frame
    assert n > 400
    s = Session()
    s.add_image(LumenImage(img, name="dense"))
    out = s.run_op("tracking.detect_spots", {"radius": 2.5, "threshold": 0, "add_to_manager": False, "add_to_results": False})
    assert "consider a larger radius or threshold" in out["message"]
    out = s.run_op("tracking.detect_spots", {"radius": 2.5, "threshold": 50, "add_to_manager": False, "add_to_results": False})
    assert "consider" not in out["message"]  # explicit threshold: the user chose it
    out = s.run_op("tracking.track", {"radius": 2.5, "threshold": 0})
    assert "consider a larger radius or threshold" in out["message"]
    sparse = Session()
    sparse.add_image(LumenImage(moving_spots(nt=3), name="sparse", axes="TYX"))
    out = sparse.run_op("tracking.track", {"radius": 2.5, "threshold": 0})
    assert "consider" not in out["message"]


def test_detector_variants_and_single_frame():
    s = Session()
    s.add_image(LumenImage(moving_spots(nt=1), name="single", axes="TYX"))
    for det in ("LoG", "DoG", "Find maxima"):
        s.roi_manager.clear()
        out = s.run_op("tracking.detect_spots", {"detector": det, "radius": 2.5, "threshold": 0, "add_to_results": False})
        assert "5 spots" in out["message"], det
        assert len(s.roi_manager.rois) == 1 and s.roi_manager.rois[0]["name"] == "Spots t0"
    # calibrated units: radius 0.5 µm at 0.2 µm/px = 2.5 px
    img = s.active
    img.calibration.pixel_width = img.calibration.pixel_height = 0.2
    img.calibration.unit = "µm"
    out = s.run_op("tracking.detect_spots", {"radius": 0.5, "units": "calibrated", "threshold": 0, "add_to_manager": False})
    assert "5 spots" in out["message"]
    assert abs(s.results.rows[-1]["X (px)"] * 0.2 - s.results.rows[-1]["X"]) < 1e-9


def test_gap_closing():
    s = Session()
    s.add_image(LumenImage(moving_spots(drop={(0, 5)}), name="gap", axes="TYX"))
    s.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6, "gap_closing_frames": 0, "min_track_length": 1})
    assert len(s.settings["tracks"]) == 6  # the broken track is split in two
    s.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6, "gap_closing_frames": 2, "max_gap_distance": 15})
    tracks = s.settings["tracks"]
    assert len(tracks) == 5
    lens = sorted(len(t["spots"]) for t in tracks)
    assert lens == [9, 10, 10, 10, 10]
    gapped = next(t for t in tracks if len(t["spots"]) == 9)
    assert [sp["t"] for sp in gapped["spots"]] == [0, 1, 2, 3, 4, 6, 7, 8, 9]


def test_export_tracks(session, tmp_path):
    with pytest.raises(ValueError):
        session.run_op("tracking.export_tracks", {"path": str(tmp_path / "x.csv")})
    session.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6})
    p = tmp_path / "tracks.csv"
    out = session.run_op("tracking.export_tracks", {"path": str(p)})
    assert "50 spots" in out["message"]
    with open(p) as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == ["ID", "TRACK_ID", "FRAME", "POSITION_X", "POSITION_Y", "POSITION_Z", "POSITION_T", "RADIUS", "QUALITY", "MEAN_INTENSITY"]
    assert len(rows) == 51
    assert {r[1] for r in rows[1:]} == {"0", "1", "2", "3", "4"}
    # POSITION_T uses the frame interval (2 s)
    assert float(rows[1][6]) == float(rows[1][2]) * 2.0
    p2 = tmp_path / "lumen.csv"
    session.run_op("tracking.export_tracks", {"path": str(p2), "layout": "Lumen"})
    with open(p2) as fh:
        hdr = next(csv.reader(fh))
    assert hdr[:4] == ["Spot", "Track", "Frame", "X (px)"]


def test_kymograph(session):
    img = session.active
    # a horizontal line through the path of the first spot (y = 20)
    roi = {"type": "line", "x1": 10, "y1": 20, "x2": 60, "y2": 20}
    out = session.run_op("tracking.kymograph", {}, roi=roi)
    kymo = session.images[out["new_images"][0]]
    assert kymo.height == 10 and kymo.width == 51 and kymo.dtype == np.float32
    assert kymo.calibration.pixel_height == 2.0  # frame interval
    # the bright ridge moves 3 px per frame
    peaks = np.argmax(kymo.data[0, 0, 0], axis=1)
    assert np.allclose(np.diff(peaks), 3, atol=1)
    with pytest.raises(ValueError, match="line"):
        session.run_op("tracking.kymograph", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10})
    single = Session()
    single.add_image(LumenImage(moving_spots(nt=1), name="single", axes="TYX"))
    with pytest.raises(ValueError, match="T > 1"):
        single.run_op("tracking.kymograph", {}, roi=roi)


# ---------------------------------------------------------------- regressions (review fixes)
def grid_spots(size, pitch, noise, amp=200.0, sigma=2.0, jitter=0.3, seed=0):
    """Dense uint16 grid of Gaussian spots (amplitude ± jitter) on a background of 100. Returns (image, n_spots)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    f = np.full((size, size), 100.0)
    n = 0
    for y in range(pitch // 2, size, pitch):
        for x in range(pitch // 2, size, pitch):
            a = amp * (1 + rng.uniform(-jitter, jitter))
            f += a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
            n += 1
    if noise > 0:
        f += rng.normal(0, noise, f.shape)
    return np.clip(f, 0, 65535).astype(np.uint16), n


def _detect_count(data, **params):
    s = Session()
    s.add_image(LumenImage(data, name="g", axes="TYX" if data.ndim == 3 else None))
    s.run_op("tracking.detect_spots", {"radius": 2.5, "threshold": 0, "add_to_manager": False, "add_to_results": False, **params})
    return len(s.settings["spots"])


def test_auto_threshold_keeps_dense_and_clean_spots():
    # the automatic threshold used to floor the Otsu split at median + 3 MAD of the candidate maxima, which
    # on dense fields (most candidates are spots) landed above every real spot: 0 of 441 detected
    img, n = grid_spots(256, 12, noise=5)
    assert _detect_count(img) == n
    img, n = grid_spots(256, 8, noise=20)
    assert _detect_count(img) == n
    # noiseless grids, identical and jittered amplitudes
    img, n = grid_spots(128, 16, noise=0, jitter=0.0)
    assert _detect_count(img) == n
    img, n = grid_spots(128, 16, noise=0)
    assert _detect_count(img) == n


@pytest.mark.parametrize("detector", ["LoG", "DoG", "Find maxima"])
def test_auto_threshold_rejects_pure_noise(detector):
    rng = np.random.default_rng(3)
    for sigma in (5, 20):
        noise = np.clip(100 + rng.normal(0, sigma, (256, 256)), 0, 65535).astype(np.uint16)
        assert _detect_count(noise, detector=detector) <= 2
    stack = np.clip(100 + rng.normal(0, 10, (10, 256, 256)), 0, 65535).astype(np.uint16)
    assert _detect_count(stack, detector=detector) <= 5


def test_calibrated_radius_below_half_pixel_is_not_clamped():
    from lumen.core.registry import REGISTRY
    # the shared radius Param had min=0.5, a pixel-unit floor silently applied to calibrated values
    assert REGISTRY.get("tracking.detect_spots").coerce_params({"radius": 0.25, "units": "calibrated"})["radius"] == 0.25
    assert REGISTRY.get("tracking.track").coerce_params({"radius": 0.15, "units": "calibrated"})["radius"] == 0.15
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64]
    f = np.full((64, 64), 10.0)
    for x, y in [(20, 20), (24, 20), (40, 40), (44, 40), (20, 44), (24, 44)]:  # pairs 4 px apart, sigma 1 px
        f += 200 * np.exp(-((xx + 0.5 - x) ** 2 + (yy + 0.5 - y) ** 2) / 2.0)
    f += rng.normal(0, 2, f.shape)
    s = Session()
    img = s.add_image(LumenImage(f.astype(np.float32), name="vesicles"))
    img.calibration.pixel_width = img.calibration.pixel_height = 0.1
    img.calibration.unit = "um"
    s.run_op("tracking.detect_spots", {"radius": 0.15, "units": "calibrated", "threshold": 0})
    assert s.settings["spots_meta"]["radius_px"] == pytest.approx(1.5)
    assert s.results.rows[-1]["Radius"] == pytest.approx(0.15)
    assert len(s.settings["spots"]) == 6  # at the clamped 5 px the pairs merged into 3
    with pytest.raises(ValueError, match="positive"):
        s.run_op("tracking.detect_spots", {"radius": 0, "threshold": 0})


def test_lap_linking_is_sparse_and_keeps_zero_cost_links():
    import tracemalloc
    from lumen.core.ops.tracking import _link
    # stationary spots (cost exactly 0) must still be linked, frame to frame and across a gap
    spots = [{"t": 0, "x": 10.0, "y": 10.0}, {"t": 1, "x": 10.0, "y": 10.0}, {"t": 0, "x": 30.0, "y": 30.0}, {"t": 1, "x": 30.0, "y": 30.0}]
    assert _link(spots, 5.0, 0, 0.0, 2) == [[0, 1], [2, 3]]
    assert _link([{"t": 0, "x": 10.0, "y": 10.0}, {"t": 2, "x": 10.0, "y": 10.0}], 5.0, 2, 5.0, 2) == [[0, 1]]
    # 20 000 single-spot segments: the dense (2 n)^2 augmented matrix would need 12.8 GB
    rng = np.random.default_rng(0)
    spots = [{"t": t, "x": float(rng.uniform(0, 4000)), "y": float(rng.uniform(0, 4000))} for t in range(500) for _ in range(40)]
    tracemalloc.start()
    tracks = _link(spots, 0.5, 2, 15.0, 2)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert peak < 100e6
    assert tracks and all(1 <= spots[b]["t"] - spots[a]["t"] <= 3 for tr in tracks for a, b in zip(tr, tr[1:]))
    assert all(np.hypot(spots[a]["x"] - spots[b]["x"], spots[a]["y"] - spots[b]["y"]) <= 15.0 for tr in tracks for a, b in zip(tr, tr[1:]))


def test_lap_matches_dense_optimum():
    from scipy.optimize import linear_sum_assignment
    from lumen.core.ops.tracking import _lap

    def dense(cost, alt):  # the former dense Jaqaman matrix
        n, m = cost.shape
        finite = np.isfinite(cost)
        eps = float(cost[finite].min())
        full = np.full((n + m, n + m), np.inf)
        full[:n, :m] = cost
        full[:n, m:] = np.where(np.eye(n, dtype=bool), alt, np.inf)
        full[n:, :m] = np.where(np.eye(m, dtype=bool), alt, np.inf)
        full[n:, m:] = np.where(finite.T, eps, np.inf)
        rows, cols = linear_sum_assignment(full)
        return [(int(i), int(j)) for i, j in zip(rows, cols) if i < n and j < m]

    def objective(cost, links, alt):
        n, m = cost.shape
        return sum(cost[i, j] for i, j in links) + alt * (n - len(links) + m - len(links))

    rng = np.random.default_rng(0)
    for trial in range(60):
        n, m = int(rng.integers(1, 30)), int(rng.integers(1, 30))
        cost = rng.uniform(0, 100, (n, m))
        cost[rng.random((n, m)) < 0.8] = np.inf
        if trial % 3 == 0:
            cost[np.isfinite(cost) & (rng.random((n, m)) < 0.3)] = 0.0  # coincident spots
        if not np.isfinite(cost).any():
            continue
        ii, jj = np.nonzero(np.isfinite(cost))
        links = _lap(n, m, ii, jj, cost[ii, jj], 105.0)
        assert all(np.isfinite(cost[i, j]) for i, j in links)
        assert objective(cost, links, 105.0) == pytest.approx(objective(cost, dense(cost, 105.0), 105.0))


def test_detect_does_not_retain_response_planes(monkeypatch):
    import tracemalloc
    from lumen.core.ops import tracking as tk
    nt, size = 20, 512
    data = (moving_spots(nt=nt, size=size) + 100).astype(np.uint16)  # offset: no clipped plateaus
    held = {}
    orig = tk._auto_threshold

    def spy(q, floor=0.0):  # called once all frames have been processed
        held["bytes"] = tracemalloc.get_traced_memory()[0]
        return orig(q, floor)

    monkeypatch.setattr(tk, "_auto_threshold", spy)
    s = Session()
    s.add_image(LumenImage(data, name="movie", axes="TYX"))
    tracemalloc.start()
    out = s.run_op("tracking.detect_spots", {"radius": 2.5, "threshold": 0, "add_to_manager": False, "add_to_results": False})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    planes = 4 * size * size * nt  # one float32 response per frame: what the op used to hold until the end
    assert held["bytes"] < 0.3 * planes  # per-candidate arrays only (a few thousand noise maxima per frame)
    assert peak < planes  # the old code held all the planes plus the same per-frame transients (~1.2x)
    assert "100 spots" in out["message"]


# ---------------------------------------------------------------- registration regressions (same review batch)
def test_register_to_image_beyond_half_canvas():
    from skimage.data import camera
    cam = camera().astype(np.uint8)

    def register(mov, tgt, mode="Translation"):
        s = Session()
        a = s.add_image(LumenImage(tgt, name="target"))
        b = s.add_image(LumenImage(mov, name="moving"))
        s.select(b.id)
        out = s.run_op("registration.register_to_image", {"target": a.id, "mode": mode})
        return s.results.rows[-1], s.images[out["new_images"][0]]

    # circular phase correlation wrapped +300 to -212 and the warped image was all zeros
    r, new = register(cam[300:400, 300:400], cam)
    assert abs(r["dx"] - 300) < 0.5 and abs(r["dy"] - 300) < 0.5
    assert np.array_equal(new.data[0, 0, 0, 300:400, 300:400], cam[300:400, 300:400])
    assert np.count_nonzero(new.data) == 10000
    r, _ = register(cam[450:, 10:110], cam)  # clipped at the bottom edge
    assert abs(r["dx"] - 10) < 0.5 and abs(r["dy"] - 450) < 0.5
    r, new = register(cam, cam[300:400, 300:400])  # moving image larger than the target
    assert abs(r["dx"] + 300) < 0.5 and abs(r["dy"] + 300) < 0.5
    assert np.array_equal(new.data[0, 0, 0], cam[300:400, 300:400])
    r, _ = register(cam[300:500, 300:500], cam, mode="Rigid body")  # translation fallback of the feature modes
    assert abs(r["dx"] - 300) < 3 and abs(r["dy"] - 300) < 3


def test_stitch_overlay_opens_in_composite_mode(tmp_path):
    import tifffile
    from skimage.data import camera
    cam = camera().astype(np.uint8)
    s = Session()
    c1 = s.add_image(LumenImage(cam[0:300, 0:300], name="c1"))
    c2 = s.add_image(LumenImage(cam[20:320, 200:500], name="c2"))
    out = s.run_op("registration.pairwise_stitch", {"image1": c1.id, "image2": c2.id, "fusion": "Overlay (channels)"})
    st = s.images[out["new_images"][0]]
    assert st.nc == 2 and st.display.mode == "composite" and st.display.c == 0
    for k, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        tifffile.imwrite(str(tmp_path / f"tile_{k:02d}.tif"), cam[r * 160:r * 160 + 200, c * 160:c * 160 + 200])
    out = s.run_op("registration.grid_stitch", {"folder": str(tmp_path), "pattern": "*.tif", "grid_x": 2, "grid_y": 2, "overlap_percent": 20, "fusion": "Overlay (channels)"})
    st = s.images[out["new_images"][0]]
    assert st.nc == 4 and st.display.mode == "composite"


def test_grid_stitch_skips_hidden_files_and_names_bad_tiles(tmp_path):
    import tifffile
    from skimage.data import camera
    cam = camera().astype(np.uint8)
    for k, (r, c) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        tifffile.imwrite(str(tmp_path / f"tile_{k:02d}.tif"), cam[r * 160:r * 160 + 200, c * 160:c * 160 + 200])
        (tmp_path / f"._tile_{k:02d}.tif").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 100)  # macOS AppleDouble stub
    (tmp_path / ".DS_Store").write_bytes(b"x")
    s = Session()
    for pattern in ("*.tif", r"tile_\d+\.tif"):  # glob and (unanchored) regex path
        out = s.run_op("registration.grid_stitch", {"folder": str(tmp_path), "pattern": pattern, "grid_x": 2, "grid_y": 2, "overlap_percent": 20})
        assert [row[1] for row in s.tables["Tile positions"]["rows"]] == [f"tile_{k:02d}.tif" for k in range(4)]
        assert [(row[4], row[5]) for row in s.tables["Tile positions"]["rows"]] == [(0, 0), (160, 0), (0, 160), (160, 160)]
        assert "4/4 neighbour pairs" in out["message"]
    # a stray matching file is reported rather than silently truncated away
    tifffile.imwrite(str(tmp_path / "overview.tif"), cam[:200, :200])
    out = s.run_op("registration.grid_stitch", {"folder": str(tmp_path), "pattern": "*.tif", "grid_x": 2, "grid_y": 2, "overlap_percent": 20})
    assert "1 further file" in out["message"] and "tile_03.tif" in out["message"]
    (tmp_path / "overview.tif").unlink()
    # an unreadable visible tile fails loudly with its name (grid position is assigned by index)
    (tmp_path / "tile_01.tif").write_bytes(b"junk" * 50)
    with pytest.raises(ValueError, match="tile_01.tif"):
        s.run_op("registration.grid_stitch", {"folder": str(tmp_path), "pattern": "*.tif", "grid_x": 2, "grid_y": 2, "overlap_percent": 20})


# ---------------------------------------------------------------- manual tracking, TrackMate XML, tracks → ROIs
def test_manual_tracking_round_trip(session):
    img = session.active
    out = session.run_op("tracking.manual_new_track", {})
    assert "track_id = 0" in out["message"] and session.settings["tracks_image"] == img.id
    tr = [o for o in img.overlay if o["type"] == "track"]
    assert tr == [{"type": "track", "id": "manual-0", "name": "Manual 0", "color": tr[0]["color"], "track_id": "0", "points": []}] and tr[0]["color"].startswith("#")
    assert session.tables["Tracks"]["rows"][0][0] == "Manual 0"
    for t in range(4):
        out = session.run_op("tracking.manual_add_point", {"track_id": 0, "x": 20 + 3 * t, "y": 20, "t": t})
    assert "added on frame 3 (4 spots)" in out["message"]
    # replace the point of a frame; add from the point selection on the current frame (display advances)
    session.run_op("tracking.manual_add_point", {"track_id": 0, "x": 29.5, "y": 20.5, "t": 3})
    img.display.t = 4
    out = session.run_op("tracking.manual_add_point", {}, roi={"type": "point", "points": [[32, 20]]})
    assert "frame 4 (5 spots)" in out["message"] and img.display.t == 5
    tr = next(o for o in img.overlay if o["type"] == "track")
    assert tr["id"] == "manual-0" and tr["points"] == [[20, 20, 0], [23, 20, 1], [26, 20, 2], [29.5, 20.5, 3], [32, 20, 4]]
    track = session.settings["tracks"][0]
    assert track["overlay_id"] == "manual-0" and track["manual"] and len(track["spots"]) == 5 and track["spots"][0]["intensity"] > 100
    assert track["stats"]["N spots"] == 5 and track["stats"]["Start frame"] == 0 and track["stats"]["End frame"] == 4
    assert abs(track["stats"]["Net displacement"] - 12) < 1e-6 and abs(track["stats"]["Mean speed"] - 1.5) < 0.1  # 3 px / 2 s
    row = dict(zip(session.tables["Tracks"]["columns"], session.tables["Tracks"]["rows"][0]))
    assert row["Label"] == "Manual 0" and row["N spots"] == 5 and row["Color"] == tr["color"]
    # a second, named track; delete a point; delete a track
    out = session.run_op("tracking.manual_new_track", {"name": "cell B", "color": "#123456"})
    assert "track_id = 1" in out["message"]
    session.run_op("tracking.manual_add_point", {"x": 100, "y": 30, "t": 0})  # −1 = the last manual track
    assert len(session.settings["tracks"][1]["spots"]) == 1 and session.settings["tracks"][1]["color"] == "#123456"
    out = session.run_op("tracking.manual_delete_point", {"track_id": 0, "t": 2})
    assert "4 spots left" in out["message"] and [p[2] for p in next(o for o in img.overlay if o["id"] == "manual-0")["points"]] == [0, 1, 3, 4]
    with pytest.raises(ValueError, match="no point on frame 2"):
        session.run_op("tracking.manual_delete_point", {"track_id": 0, "t": 2})
    with pytest.raises(ValueError, match="No track with id"):
        session.run_op("tracking.manual_add_point", {"track_id": 7, "x": 1, "y": 1})
    with pytest.raises(ValueError, match="point selection"):
        session.run_op("tracking.manual_add_point", {"track_id": 0})
    # filtering keeps manual overlays by their own ids
    out = session.run_op("tracking.filter_tracks", {"min_length": 2})
    assert "Kept 1 of 2" in out["message"] and [o["id"] for o in img.overlay if o["type"] == "track"] == ["manual-0"]
    session.run_op("tracking.manual_delete_point", {"track_id": 0, "delete_track": True})
    assert session.settings["tracks"] == [] and not [o for o in img.overlay if o["type"] == "track"]
    assert not session.results.rows


def test_tracks_to_rois(session):
    img = session.active
    session.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6})
    session.run_op("tracking.manual_new_track", {})  # empty: skipped
    session.run_op("tracking.manual_new_track", {})
    session.run_op("tracking.manual_add_point", {"x": 5, "y": 5, "t": 2})
    out = session.run_op("tracking.tracks_to_rois", {"width": 2})
    assert "6 track ROIs" in out["message"]
    rois = session.roi_manager.rois
    assert len(rois) == 6 and all(r["image_id"] == img.id for r in rois)
    first = rois[0]
    assert first["type"] == "polyline" and first["name"] == "Track 0" and first["width"] == 2 and len(first["points"]) == 10 and first["position"] == {"t": 0, "z": 0, "c": 0}
    assert first["color"] == session.settings["tracks"][0]["color"]
    assert rois[-1]["type"] == "point" and rois[-1]["points"] == [[5, 5]] and rois[-1]["position"]["t"] == 2
    fresh = Session()
    fresh.add_image(LumenImage(moving_spots(nt=2), name="m", axes="TYX"))
    with pytest.raises(ValueError, match="No tracks"):
        fresh.run_op("tracking.tracks_to_rois", {})


def test_trackmate_xml_round_trip(session, tmp_path):
    import xml.etree.ElementTree as ET
    img = session.active
    img.calibration.pixel_width = img.calibration.pixel_height = 0.5
    img.calibration.unit = "µm"
    with pytest.raises(ValueError, match="No tracks"):
        session.run_op("tracking.export_trackmate_xml", {"path": str(tmp_path / "x.xml")})
    session.run_op("tracking.track", {"radius": 2.5, "threshold": 30, "max_displacement": 6})
    before = session.settings["tracks"]
    pts_before = {t["name"]: [(round(sp["x"], 3), round(sp["y"], 3), sp["t"]) for sp in t["spots"]] for t in before}
    p = tmp_path / "tracks"
    out = session.run_op("tracking.export_trackmate_xml", {"path": str(p)})
    p = tmp_path / "tracks.xml"
    assert p.is_file() and "50 spots of 5 tracks" in out["message"]
    root = ET.parse(str(p)).getroot()
    assert root.tag == "TrackMate" and root.get("version")
    model = root.find("Model")
    assert model.get("spatialunits") == "µm" and model.get("timeunits") == "s"
    assert [c.tag for c in model.find("FeatureDeclarations")] == ["SpotFeatures", "EdgeFeatures", "TrackFeatures"]
    assert model.find("AllSpots").get("nspots") == "50" and len(model.findall("AllSpots/SpotsInFrame")) == 10
    spot = model.find("AllSpots/SpotsInFrame/Spot")
    for key in ("ID", "name", "VISIBILITY", "QUALITY", "POSITION_X", "POSITION_Y", "POSITION_Z", "POSITION_T", "FRAME", "RADIUS", "MEAN_INTENSITY_CH1"):
        assert spot.get(key) is not None, key
    assert float(spot.get("POSITION_T")) == float(spot.get("FRAME")) * 2.0  # calibrated time
    tracks = model.findall("AllTracks/Track")
    assert len(tracks) == 5 and all(len(t.findall("Edge")) == 9 and t.get("NUMBER_SPOTS") == "10" for t in tracks)
    assert {t.get("TRACK_ID") for t in tracks} == {e.get("TRACK_ID") for e in model.findall("FilteredTracks/TrackID")}
    edge = tracks[0].find("Edge")
    assert edge.get("SPOT_SOURCE_ID") and edge.get("SPOT_TARGET_ID") and edge.get("LINK_COST")
    imd = root.find("Settings/ImageData")
    assert imd.get("pixelwidth") == "0.5" and imd.get("nframes") == "10" and imd.get("timeinterval") == "2.0"
    assert root.find("Settings/DetectorSettings").get("DETECTOR_NAME") == "MANUAL_DETECTOR" and root.find("Settings/TrackerSettings").get("TRACKER_NAME") == "MANUAL_TRACKER"
    # positions are calibrated in TrackMate's pixel-centre convention: a spot at x = 20 px (Lumen) is written as (20 - 0.5) * 0.5 = 9.75 µm
    xs = sorted(float(sp.get("POSITION_X")) for sp in model.find("AllSpots/SpotsInFrame").findall("Spot"))
    assert abs(xs[0] - 9.75) < 0.2
    # import back: same tracks, points, statistics, overlays and table
    session.run_op("tracking.clear_tracks", {"forget_spots": True})
    out = session.run_op("tracking.import_trackmate_xml", {"path": str(p), "add_to_manager": True})
    assert "Imported 5 tracks (50 linked of 50 spots)" in out["message"]
    after = session.settings["tracks"]
    assert {t["name"]: [(round(sp["x"], 3), round(sp["y"], 3), sp["t"]) for sp in t["spots"]] for t in after} == pts_before
    assert [t["id"] for t in after] == [t["id"] for t in before]
    for a, b in zip(before, after):
        assert abs(a["stats"]["Mean speed"] - b["stats"]["Mean speed"]) < 1e-6 and abs(a["spots"][0]["intensity"] - b["spots"][0]["intensity"]) < 1e-6
    assert len([o for o in img.overlay if o["type"] == "track"]) == 5 and len(session.tables["Tracks"]["rows"]) == 5
    assert len(session.settings["spots"]) == 50 and len(session.roi_manager.rois) == 10 and session.settings["spots_meta"]["image_id"] == img.id
    assert not session.results.rows
    # manual tracks export too, and re-export after import works
    session.run_op("tracking.manual_new_track", {})
    session.run_op("tracking.manual_add_point", {"x": 3, "y": 4, "t": 0})
    out = session.run_op("tracking.export_trackmate_xml", {"path": str(tmp_path / "again.xml")})
    assert "51 spots of 6 tracks" in out["message"]


def test_import_trackmate_xml_branching_and_filtered(tmp_path):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<TrackMate version="7.10.2">
  <Model spatialunits="µm" timeunits="sec">
    <FeatureDeclarations><SpotFeatures/><EdgeFeatures/><TrackFeatures/></FeatureDeclarations>
    <AllSpots nspots="7">
      <SpotsInFrame frame="0">
        <Spot ID="10" name="ID10" VISIBILITY="1" QUALITY="5.0" POSITION_X="2.0" POSITION_Y="4.0" POSITION_Z="0.0" POSITION_T="0.0" FRAME="0" RADIUS="0.5" MEAN_INTENSITY_CH1="100.0"/>
        <Spot ID="20" name="ID20" VISIBILITY="1" QUALITY="5.0" POSITION_X="10.0" POSITION_Y="10.0" POSITION_Z="0.0" POSITION_T="0.0" FRAME="0" RADIUS="0.5"/>
      </SpotsInFrame>
      <SpotsInFrame frame="1">
        <Spot ID="11" name="ID11" VISIBILITY="1" QUALITY="5.0" POSITION_X="2.4" POSITION_Y="4.0" POSITION_Z="0.0" POSITION_T="1.5" FRAME="1" RADIUS="0.5"/>
        <Spot ID="21" name="ID21" VISIBILITY="1" QUALITY="5.0" POSITION_X="10.4" POSITION_Y="10.0" POSITION_Z="0.0" POSITION_T="1.5" FRAME="1" RADIUS="0.5"/>
      </SpotsInFrame>
      <SpotsInFrame frame="2">
        <Spot ID="12" name="ID12" VISIBILITY="1" QUALITY="5.0" POSITION_X="2.8" POSITION_Y="4.0" POSITION_Z="0.0" POSITION_T="3.0" FRAME="2" RADIUS="0.5"/>
        <Spot ID="13" name="ID13" VISIBILITY="1" QUALITY="5.0" POSITION_X="2.8" POSITION_Y="5.0" POSITION_Z="0.0" POSITION_T="3.0" FRAME="2" RADIUS="0.5"/>
        <Spot ID="22" name="ID22" VISIBILITY="1" QUALITY="5.0" POSITION_X="10.8" POSITION_Y="10.0" POSITION_Z="0.0" POSITION_T="3.0" FRAME="2" RADIUS="0.5"/>
      </SpotsInFrame>
    </AllSpots>
    <AllTracks>
      <Track name="Track_0" TRACK_ID="0" NUMBER_SPOTS="4">
        <Edge SPOT_SOURCE_ID="10" SPOT_TARGET_ID="11" LINK_COST="0.16"/>
        <Edge SPOT_SOURCE_ID="11" SPOT_TARGET_ID="12" LINK_COST="0.16"/>
        <Edge SPOT_SOURCE_ID="11" SPOT_TARGET_ID="13" LINK_COST="1.16"/>
      </Track>
      <Track name="Track_1" TRACK_ID="1" NUMBER_SPOTS="3">
        <Edge SPOT_SOURCE_ID="20" SPOT_TARGET_ID="21" LINK_COST="0.16"/>
        <Edge SPOT_SOURCE_ID="21" SPOT_TARGET_ID="22" LINK_COST="0.16"/>
      </Track>
    </AllTracks>
    <FilteredTracks><TrackID TRACK_ID="0"/></FilteredTracks>
  </Model>
  <Settings>
    <ImageData filename="x.tif" folder="/tmp" width="64" height="64" nslices="1" nframes="3" pixelwidth="0.2" pixelheight="0.2" voxeldepth="1.0" timeinterval="1.5"/>
  </Settings>
</TrackMate>"""
    p = tmp_path / "tm.xml"
    p.write_text(xml, encoding="utf-8")
    s = Session()
    s.add_image(LumenImage(np.zeros((3, 64, 64), np.uint8), name="z", axes="TYX"))
    out = s.run_op("tracking.import_trackmate_xml", {"path": str(p)})
    assert "Imported 1 tracks (4 linked of 7 spots)" in out["message"] and "1 track with splits / merges" in out["message"]
    tr = s.settings["tracks"]
    # positions divided by the file's pixel size (0.2 µm) plus half a pixel (TrackMate's pixel centre i is Lumen's i + 0.5); the split is kept as a lineage
    assert tr[0]["name"] == "Track_0" and [(round(sp["x"], 2), round(sp["y"], 2), sp["t"]) for sp in tr[0]["spots"]] == [(10.5, 20.5, 0), (12.5, 20.5, 1), (14.5, 20.5, 2), (14.5, 25.5, 2)]
    assert tr[0]["edges"] == [[10, 11], [11, 12], [11, 13]] and tr[0]["stats"]["N splits"] == 1 and tr[0]["stats"]["N merges"] == 0
    assert tr[0]["spots"][0]["r"] == 2.5 and len(s.settings["spots"]) == 7
    # one overlay polyline per branch: the daughter branch starts at the mother spot so the lines meet
    items = [o for o in s.active.overlay if o["type"] == "track"]
    assert [o["id"] for o in items] == ["track-0", "track-0-b1"] and all(o["track_id"] == "0" for o in items)
    assert [p[2] for p in items[0]["points"]] == [0, 1, 2] and [p[2] for p in items[1]["points"]] == [1, 2] and items[1]["points"][-1][1] == 25.5
    out = s.run_op("tracking.import_trackmate_xml", {"path": str(p), "only_filtered": False})
    assert [t["name"] for t in s.settings["tracks"]] == ["Track_0", "Track_1"]
    s.run_op("tracking.import_trackmate_xml", {"path": str(p), "units": "Pixels"})
    assert [round(sp["x"], 2) for sp in s.settings["tracks"][0]["spots"]] == [2.5, 2.9, 3.3, 3.3]
    with pytest.raises(ValueError, match="does not exist"):
        s.run_op("tracking.import_trackmate_xml", {"path": str(tmp_path / "nope.xml")})
    (tmp_path / "junk.xml").write_text("<a><b></a>")
    with pytest.raises(ValueError, match="not valid XML"):
        s.run_op("tracking.import_trackmate_xml", {"path": str(tmp_path / "junk.xml")})


def test_track_models_are_per_image():
    """Tracks of one image are parked when another image is tracked and restored on return (TrackMate: one model per image)."""
    from lumen.core.session import Session
    s = Session()
    s.run_op("file.sample", {"name": "Nuclei timelapse (synthetic)"})
    a = s.active
    s.run_op("tracking.track", {"radius": 8})
    n_a = len(s.settings["tracks"])
    s.run_op("file.sample", {"name": "Nuclei timelapse (synthetic)"})
    b = s.active
    s.run_op("tracking.manual_new_track", {})
    s.run_op("tracking.manual_add_point", {"track_id": -1, "x": 10, "y": 10, "t": 0})
    assert s.settings["tracks_image"] == b.id and len(s.settings["tracks"]) == 1
    s.select(a.id)
    out = s.run_op("tracking.filter_tracks", {"min_length": 1})
    assert s.settings["tracks_image"] == a.id and len(s.settings["tracks"]) == n_a
    assert b.id in s.settings["tracks_archive"] and len(s.settings["tracks_archive"][b.id]) == 1
    assert "Kept" in out["message"]


TRACK = {"radius": 2.5, "threshold": 30, "max_displacement": 6}


def _two_images():
    """One session with the moving-spots movie A (5 tracks when tracked) and a blank 3-frame movie B."""
    s = Session()
    a = s.add_image(LumenImage(moving_spots(), name="A", axes="TYX"))
    b = s.add_image(LumenImage(np.zeros((3, 32, 32), np.float32), name="B", axes="TYX"))
    return s, a, b


def test_failing_op_on_untracked_image_leaves_active_model_untouched(tmp_path):
    """An op that needs tracks, run on an image without any while another image's model is active, raises without
    parking that model: tracks / tracks_image / tracks_archive stay as they were, the model endpoint still serves the
    tracked image and the exporters still work (a failed command changes nothing)."""
    from lumen.server.extra.tracking import _tracks_of
    s, a, b = _two_images()
    s.run_op("tracking.track", TRACK, image_id=a.id)
    for op_id, params in (("tracking.filter_tracks", {"min_length": 3}), ("tracking.feature_tables", {}), ("tracking.tracks_to_rois", {}),
                          ("tracking.manual_add_point", {"track_id": -1, "x": 10, "y": 10, "t": 0})):
        with pytest.raises(ValueError):
            s.run_op(op_id, params, image_id=b.id)
        assert s.settings["tracks_image"] == a.id and len(s.settings["tracks"]) == 5, op_id
        assert a.id not in s.settings["tracks_archive"] and b.id not in s.settings["tracks_archive"], op_id
        assert len(_tracks_of(s, a.id)) == 5 and _tracks_of(s, b.id) == [], op_id
    out = s.run_op("tracking.export_tracks", {"path": str(tmp_path / "a.csv")}, image_id=a.id)
    assert "50 spots of 5 tracks" in out["message"]


def test_detect_spots_keeps_other_images_model():
    """Detect Spots invalidates the tracks of its own image only: another image's active model is parked in the archive
    (not deleted), and a re-detection on the tracked image drops that image's model and undo steps."""
    from lumen.server.extra.tracking import _tracks_of
    s, a, b = _two_images()
    s.run_op("tracking.track", TRACK, image_id=a.id)
    s.run_op("tracking.detect_spots", {"radius": 2.5, "threshold": 30}, image_id=b.id)
    assert a.id in s.settings["tracks_archive"] and len(s.settings["tracks_archive"][a.id]) == 5
    assert "tracks" not in s.settings and "tracks_image" not in s.settings
    assert len(_tracks_of(s, a.id)) == 5 and _tracks_of(s, b.id) == [] and len([o for o in a.overlay if o["type"] == "track"]) == 5
    with pytest.raises(ValueError, match="No tracks for this image"):
        s.run_op("tracking.feature_tables", {}, image_id=b.id)
    out = s.run_op("tracking.feature_tables", {}, image_id=a.id)
    assert "5 tracks" in out["message"] and s.settings["tracks_image"] == a.id
    # an undo step for A, then a fresh detection on A itself: A's model and its undo steps are gone, nothing is parked
    s.run_op("tracking.delete_spot", {"spot_id": s.settings["tracks"][0]["spots"][0]["id"]}, image_id=a.id)
    assert any(e["image"] == a.id for e in s.settings["tracks_undo"])
    s.run_op("tracking.detect_spots", {"radius": 2.5, "threshold": 30}, image_id=a.id)
    assert "tracks" not in s.settings and a.id not in s.settings["tracks_archive"] and s.settings["spots_meta"]["image_id"] == a.id
    assert not [e for e in s.settings.get("tracks_undo") or [] if e["image"] == a.id]
    s.run_op("tracking.link_spots", {"max_displacement": 6, "gap_closing_frames": 0}, image_id=a.id)
    assert len(s.settings["tracks"]) == 5 and s.settings["tracks_image"] == a.id


def test_clear_tracks_is_image_scoped():
    """Clear Tracks removes the current image's model wherever it lives (active or parked) and its undo steps, and never
    another image's model – so cleared tracks cannot come back from the archive or through Undo Track Edit."""
    from lumen.server.extra.tracking import _tracks_of
    s, a, b = _two_images()
    s.run_op("tracking.track", TRACK, image_id=a.id)
    out = s.run_op("tracking.clear_tracks", {}, image_id=b.id)  # B has no tracks: A's model is not touched
    assert "Removed 0 track overlays" in out["message"]
    assert len(_tracks_of(s, a.id)) == 5 and "5 tracks" in s.run_op("tracking.feature_tables", {}, image_id=a.id)["message"]
    # an undo step for A, then park A's model behind a manual track on B and clear A
    s.run_op("tracking.delete_spot", {"spot_id": s.settings["tracks"][0]["spots"][0]["id"]}, image_id=a.id)
    s.run_op("tracking.manual_new_track", {}, image_id=b.id)
    s.run_op("tracking.manual_add_point", {"track_id": -1, "x": 10, "y": 10, "t": 0}, image_id=b.id)
    assert s.settings["tracks_image"] == b.id and a.id in s.settings["tracks_archive"] and any(e["image"] == a.id for e in s.settings["tracks_undo"])
    out = s.run_op("tracking.clear_tracks", {}, image_id=a.id)
    assert "Removed 5 track overlays" in out["message"] and not [o for o in a.overlay if o["type"] == "track"]
    assert a.id not in s.settings["tracks_archive"] and _tracks_of(s, a.id) == []
    with pytest.raises(ValueError, match="No tracks for this image"):
        s.run_op("tracking.feature_tables", {}, image_id=a.id)
    with pytest.raises(ValueError, match="Nothing to undo"):
        s.run_op("tracking.undo_edit", {}, image_id=a.id)
    # B's manual track survived (parked while A was cleared) and B's spots are still there for B
    assert len(_tracks_of(s, b.id)) == 1 and len([o for o in b.overlay if o["type"] == "track"]) == 1
    assert "1 tracks" in s.run_op("tracking.feature_tables", {}, image_id=b.id)["message"]


def test_exporters_write_the_current_images_model(tmp_path):
    """Export Tracks (CSV / TrackMate XML) write the tracks, calibration and name of the image they run on – also when
    that image's model is parked because another image's is active – without swapping the active model, and never
    another image's detection as 'unlinked' spots."""
    import xml.etree.ElementTree as ET
    s, a, b = _two_images()
    a.calibration.pixel_width = a.calibration.pixel_height = 0.5
    a.calibration.unit = "µm"
    s.run_op("tracking.track", TRACK, image_id=a.id)
    s.run_op("tracking.manual_new_track", {}, image_id=b.id)
    s.run_op("tracking.manual_add_point", {"track_id": -1, "x": 10, "y": 10, "t": 0}, image_id=b.id)
    assert s.settings["tracks_image"] == b.id and a.id in s.settings["tracks_archive"]
    out = s.run_op("tracking.export_tracks", {"path": str(tmp_path / "a.csv")}, image_id=a.id)
    assert "50 spots of 5 tracks" in out["message"]
    with open(tmp_path / "a.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    assert len(rows) == 51 and {r[1] for r in rows[1:]} == {"0", "1", "2", "3", "4"}
    first = s.settings["tracks_archive"][a.id][0]["spots"][0]
    assert float(rows[1][3]) == pytest.approx((first["x"] - 0.5) * 0.5, abs=1e-3)  # A's calibration, TrackMate's pixel-centre shift
    assert s.settings["tracks_image"] == b.id and a.id in s.settings["tracks_archive"]  # an export never swaps the active model
    s.run_op("tracking.export_trackmate_xml", {"path": str(tmp_path / "a.xml")}, image_id=a.id)
    root = ET.parse(str(tmp_path / "a.xml")).getroot()
    assert root.find("Model/AllSpots").get("nspots") == "50" and root.find("Model").get("spatialunits") == "µm"
    imd = root.find("Settings/ImageData")
    assert imd.get("filename") == "A.tif" and imd.get("pixelwidth") == "0.5"
    # B's exports carry B's single manual spot only: A's detection is not B's unlinked spots
    out = s.run_op("tracking.export_tracks", {"path": str(tmp_path / "b.csv"), "include_unlinked": True}, image_id=b.id)
    assert out["message"].startswith("Wrote 1 spots of 1 tracks")
    s.run_op("tracking.export_trackmate_xml", {"path": str(tmp_path / "b.xml"), "include_unlinked": True}, image_id=b.id)
    assert ET.parse(str(tmp_path / "b.xml")).getroot().find("Model/AllSpots").get("nspots") == "1"
    # the same exports from A with the unlinked spots include A's whole detection
    out = s.run_op("tracking.export_tracks", {"path": str(tmp_path / "a2.csv"), "include_unlinked": True}, image_id=a.id)
    assert "50 spots of 5 tracks" in out["message"]


def test_trackmate_pixel_centre_convention(tmp_path):
    """TrackMate puts the centre of pixel i at i * pixel size, Lumen at i + 0.5 px: the XML / CSV exporters shift by
    half a pixel (a spot on pixel 10 is written as 5.0 µm at 0.5 µm/px, not 5.25) and the XML importer shifts back."""
    import xml.etree.ElementTree as ET
    s = Session()
    data = np.zeros((3, 40, 40), np.float32)
    data[:, 20, 10] = 1000.0  # one bright pixel at index (x=10, y=20) on every frame
    img = s.add_image(LumenImage(data, name="px", axes="TYX"))
    img.calibration.pixel_width = img.calibration.pixel_height = 0.5
    img.calibration.unit = "µm"
    s.run_op("tracking.track", {"radius": 1.0, "threshold": 10, "subpixel": False, "max_displacement": 3})
    assert [(sp["x"], sp["y"]) for sp in s.settings["tracks"][0]["spots"]] == [(10.5, 20.5)] * 3
    s.run_op("tracking.export_trackmate_xml", {"path": str(tmp_path / "px.xml")})
    root = ET.parse(str(tmp_path / "px.xml")).getroot()
    spots = root.findall("Model/AllSpots/SpotsInFrame/Spot")
    assert len(spots) == 3 and {(sp.get("POSITION_X"), sp.get("POSITION_Y"), sp.get("POSITION_Z")) for sp in spots} == {("5.0", "10.0", "0.0")}
    track = root.find("Model/AllTracks/Track")
    assert (track.get("TRACK_X_LOCATION"), track.get("TRACK_Y_LOCATION"), track.get("TRACK_DISPLACEMENT")) == ("5.0", "10.0", "0.0")
    edge = track.find("Edge")
    assert (edge.get("EDGE_X_LOCATION"), edge.get("EDGE_Y_LOCATION"), edge.get("DISPLACEMENT")) == ("5.0", "10.0", "0.0")
    # the TrackMate CSV layout shifts too; the Lumen layout keeps Lumen's convention next to its pixel columns
    s.run_op("tracking.export_tracks", {"path": str(tmp_path / "tm.csv"), "layout": "TrackMate"})
    with open(tmp_path / "tm.csv") as fh:
        rows = list(csv.reader(fh))
    assert rows[1][3:6] == ["5.0", "10.0", "0.0"]
    s.run_op("tracking.export_tracks", {"path": str(tmp_path / "lumen.csv"), "layout": "Lumen"})
    with open(tmp_path / "lumen.csv") as fh:
        rows = list(csv.reader(fh))
    assert rows[1][3:7] == ["10.5", "20.5", "5.25", "10.25"]
    # a TrackMate file with the spot on pixel 10 / 20 imports to Lumen's pixel centre (10.5, 20.5), in both units modes
    tm = tmp_path / "tm.xml"
    tm.write_text("""<TrackMate version="7.11.1"><Model spatialunits="µm" timeunits="sec"><AllSpots nspots="2">
<SpotsInFrame frame="0"><Spot ID="1" POSITION_X="5.0" POSITION_Y="10.0" POSITION_Z="0.0" FRAME="0" RADIUS="0.5"/></SpotsInFrame>
<SpotsInFrame frame="1"><Spot ID="2" POSITION_X="5.0" POSITION_Y="10.0" POSITION_Z="0.0" FRAME="1" RADIUS="0.5"/></SpotsInFrame></AllSpots>
<AllTracks><Track name="T" TRACK_ID="0"><Edge SPOT_SOURCE_ID="1" SPOT_TARGET_ID="2"/></Track></AllTracks>
<FilteredTracks><TrackID TRACK_ID="0"/></FilteredTracks></Model>
<Settings><ImageData width="40" height="40" nslices="1" nframes="3" pixelwidth="0.5" pixelheight="0.5" voxeldepth="1.0" timeinterval="1.0"/></Settings></TrackMate>""", encoding="utf-8")
    s.run_op("tracking.clear_tracks", {"forget_spots": True})
    s.run_op("tracking.import_trackmate_xml", {"path": str(tm), "add_to_manager": True})
    assert [(sp["x"], sp["y"], sp["z"]) for sp in s.settings["tracks"][0]["spots"]] == [(10.5, 20.5, 0), (10.5, 20.5, 0)]
    assert next(o for o in img.overlay if o["type"] == "track")["points"] == [[10.5, 20.5, 0], [10.5, 20.5, 1]]
    assert s.roi_manager.rois[0]["points"] == [[10.5, 20.5]]
    s.run_op("tracking.import_trackmate_xml", {"path": str(tm), "units": "Pixels"})
    assert [(sp["x"], sp["y"]) for sp in s.settings["tracks"][0]["spots"]] == [(5.5, 10.5), (5.5, 10.5)]
    # re-exporting the imported model reproduces the file's coordinates
    s.run_op("tracking.import_trackmate_xml", {"path": str(tm)})
    s.run_op("tracking.export_trackmate_xml", {"path": str(tmp_path / "again.xml")})
    again = ET.parse(str(tmp_path / "again.xml")).getroot().find("Model/AllSpots/SpotsInFrame/Spot")
    assert (again.get("POSITION_X"), again.get("POSITION_Y")) == ("5.0", "10.0")


def test_tracks_table_columns_do_not_depend_on_first_track(session):
    """An empty manual track listed first must not hide the calibrated Duration column of the populated tracks
    (it gets 0.0 there, not None), and the layout stays put as tracks are added or emptied."""
    session.run_op("tracking.manual_new_track", {})
    session.run_op("tracking.manual_new_track", {})
    session.run_op("tracking.manual_add_point", {"track_id": 1, "x": 20, "y": 20, "t": 0})
    session.run_op("tracking.manual_add_point", {"track_id": 1, "x": 23, "y": 20, "t": 1})
    tab = session.tables["Tracks"]
    cols = tab["columns"]
    assert cols[0] == "Label" and cols[-1] == "Color" and "Duration (s)" in cols
    assert cols.index("Duration (s)") == cols.index("Duration (frames)") + 1
    rows = [dict(zip(cols, r)) for r in tab["rows"]]
    assert rows[0]["Label"] == "Manual 0" and rows[0]["N spots"] == 0 and rows[0]["Duration (s)"] == 0.0
    assert rows[1]["Label"] == "Manual 1" and rows[1]["Duration (frames)"] == 1 and rows[1]["Duration (s)"] == 2.0
    assert list(session.settings["tracks"][0]["stats"]) == list(session.settings["tracks"][1]["stats"])
    session.run_op("tracking.manual_new_track", {})
    assert session.tables["Tracks"]["columns"] == cols and session.tables["Tracks"]["rows"][2][cols.index("Duration (s)")] == 0.0
    session.run_op("tracking.manual_delete_point", {"track_id": 1, "t": 0})
    session.run_op("tracking.manual_delete_point", {"track_id": 1, "t": 1})
    assert session.tables["Tracks"]["columns"] == cols and session.tables["Tracks"]["rows"][1][cols.index("N spots")] == 0
    # defence in depth: rows with different key sets still share one layout, Label first and Color last
    from lumen.core.ops.tracking import _tracks_table
    tab = _tracks_table([{"name": "a", "color": "#000000", "stats": {"N spots": 0}}, {"name": "b", "color": "#ffffff", "stats": {"N spots": 1, "Duration (s)": 2.0}}])
    assert tab["columns"] == ["Label", "N spots", "Duration (s)", "Color"] and tab["rows"] == [["a", 0, None, "#000000"], ["b", 1, 2.0, "#ffffff"]]
    assert _tracks_table([])["columns"] == ["Label", "Color"]
