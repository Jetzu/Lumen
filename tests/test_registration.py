import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.data import camera
from skimage.transform import rotate

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.session import Session

CAM = camera().astype(np.float32)
SHIFTS = [(0, 0), (3, -2), (5, 4), (-4, 6), (2, 2), (-6, -3), (7, 1), (1, -5)]  # (dy, dx) of each slice


def alignment_rows(s):
    """The 'Alignment' table (per-slice / per-channel / per-frame shifts) as a list of dicts."""
    tab = s.tables["Alignment"]
    return [dict(zip(tab["columns"], r)) for r in tab["rows"]]


def shifted_stack(shifts=SHIFTS, size=200, origin=150, channels=2):
    """A (Z, C, Y, X) uint8 stack whose slices are crops of camera() at known offsets.

    Slice k shows the scene displaced by (-dy, -dx), so aligning it onto slice 0 needs the
    shift (+dy, +dx).
    """
    planes = []
    for dy, dx in shifts:
        a = CAM[origin + dy:origin + dy + size, origin + dx:origin + dx + size]
        chans = [a] + [np.roll(a, 7 * (c + 1), axis=0) for c in range(channels - 1)]
        planes.append(np.stack(chans))
    return np.stack(planes).astype(np.uint8)


def test_align_stack_translation():
    s = Session()
    img = s.add_image(LumenImage(shifted_stack(), name="shifted", axes="ZCYX"))
    assert img.nz == 8 and img.nc == 2
    before = img.data.copy()
    out = s.run_op("registration.align_stack", {"axis": "Z", "mode": "Translation", "reference": "First slice", "create_new": True})
    assert "Aligned 8 slices" in out["message"]
    rows = alignment_rows(s)
    assert len(rows) == 8 and not s.results.rows  # shifts go to the Alignment table, not to Results
    assert s.tables["Alignment"]["columns"] == ["Label", "Slice", "dx", "dy", "Angle", "Scale"]
    assert rows[0]["Label"].startswith("shifted") and rows[0]["Slice"] == 1
    for r, (dy, dx) in zip(rows, SHIFTS):
        assert abs(r["dx"] - dx) < 0.5 and abs(r["dy"] - dy) < 0.5 and abs(r["Angle"]) < 1e-6
    new = s.images[out["new_images"][0]]
    assert new.shape == img.shape and new.dtype == np.uint8
    assert np.array_equal(img.data, before)  # create_new leaves the source untouched
    # after alignment every slice matches slice 0 in the central region, in both channels
    for c in range(2):
        ctr = new.data[0, :, c, 40:160, 40:160].astype(float)
        assert np.abs(ctr - ctr[:1]).mean(axis=(1, 2)).max() < 2.0
    # in-place with the chain reference, undoable
    s.select(img.id)
    s.run_op("registration.align_stack", {"axis": "Z", "mode": "Translation", "reference": "Previous slice"})
    ctr = img.data[0, :, 0, 40:160, 40:160].astype(float)
    assert np.abs(ctr - ctr[:1]).mean(axis=(1, 2)).max() < 2.0
    s.undo(img.id)
    assert np.array_equal(img.data, before)


@pytest.mark.parametrize("mode", ["Rigid body", "Affine"])
def test_align_stack_feature_modes(mode):
    s = Session()
    s.add_image(LumenImage(shifted_stack(channels=1), name="shifted", axes="ZCYX"))
    out = s.run_op("registration.align_stack", {"axis": "Z", "mode": mode, "reference": "First slice", "create_new": True})
    for r, (dy, dx) in zip(alignment_rows(s), SHIFTS):
        assert abs(r["dx"] - dx) < 0.5 and abs(r["dy"] - dy) < 0.5 and abs(r["Angle"]) < 0.2
    assert "feature-based" in out["message"]


def test_align_stack_rigid_rotation():
    s = Session()
    base = CAM[100:400, 100:400]
    rot = rotate(base, 5.0, preserve_range=True)
    s.add_image(LumenImage(np.stack([base, rot]).astype(np.uint8), name="rot", axes="ZYX"))
    s.run_op("registration.align_stack", {"axis": "Z", "mode": "Rigid body", "reference": "First slice"})
    r = alignment_rows(s)[-1]
    assert abs(r["Angle"] - 5.0) < 0.3
    img = s.active
    ctr = img.data[0, :, 0, 100:200, 100:200].astype(float)
    assert np.abs(ctr[1] - ctr[0]).mean() < 3.0


def test_align_stack_time_axis_and_errors():
    s = Session()
    data = shifted_stack(shifts=SHIFTS[:4], channels=1)[:, 0]  # (T, Y, X)
    s.add_image(LumenImage(data, name="tl", axes="TYX"))
    s.run_op("registration.align_stack", {"axis": "T", "mode": "Translation", "reference": "First slice"})
    rows = alignment_rows(s)
    assert len(rows) == 4 and not s.results.rows
    for r, (dy, dx) in zip(rows, SHIFTS[:4]):
        assert abs(r["dx"] - dx) < 0.5 and abs(r["dy"] - dy) < 0.5
    with pytest.raises(ValueError, match="Z > 1"):
        s.run_op("registration.align_stack", {"axis": "Z"})
    s.add_image(LumenImage(CAM[:64, :64].astype(np.uint8), name="flat"))
    with pytest.raises(ValueError, match="T > 1"):
        s.run_op("registration.align_stack", {"axis": "T"})


def test_align_channels():
    s = Session()
    ch2 = np.roll(np.roll(CAM, 3, axis=0), -2, axis=1)
    img = s.add_image(LumenImage(np.stack([CAM, ch2]).astype(np.uint8), name="chan", axes="CYX"))
    out = s.run_op("registration.align_channels", {"reference_channel": 1})
    rows = alignment_rows(s)
    assert len(rows) == 2 and not s.results.rows and s.tables["Alignment"]["columns"] == ["Label", "Channel", "dx", "dy"]
    assert rows[0]["dx"] == 0 and rows[0]["dy"] == 0
    assert abs(rows[1]["dx"] - 2) < 0.5 and abs(rows[1]["dy"] + 3) < 0.5
    assert "C2" in out["message"]
    inner = (slice(10, -10), slice(10, -10))
    assert np.abs(img.data[0, 0, 0][inner].astype(float) - img.data[0, 0, 1][inner]).mean() < 1.0
    s.add_image(LumenImage(CAM.astype(np.uint8), name="one"))
    with pytest.raises(ValueError, match="two channels"):
        s.run_op("registration.align_channels", {})


def test_register_to_image():
    s = Session()
    a = s.add_image(LumenImage(CAM[50:250, 50:250].astype(np.uint8), name="A"))
    b = s.add_image(LumenImage(CAM[55:255, 47:247].astype(np.uint8), name="B"))
    s.select(b.id)
    out = s.run_op("registration.register_to_image", {"target": a.id})
    r = s.results.rows[-1]
    assert abs(r["dx"] + 3) < 0.5 and abs(r["dy"] - 5) < 0.5
    new = s.images[out["new_images"][0]]
    assert new.shape == a.shape
    inner = (slice(10, -10), slice(10, -10))
    assert np.abs(new.data[0, 0, 0][inner].astype(float) - a.data[0, 0, 0][inner]).mean() < 1.0
    s.select(b.id)
    with pytest.raises(ValueError, match="different image"):
        s.run_op("registration.register_to_image", {"target": b.id})
    with pytest.raises(ValueError, match="target"):
        s.run_op("registration.register_to_image", {})


@pytest.mark.parametrize("fusion", ["Linear blend", "Max", "Average", "Overlay (channels)"])
def test_pairwise_stitch(fusion):
    s = Session()
    c1 = s.add_image(LumenImage(CAM[0:300, 0:300].astype(np.uint8), name="c1"))
    c2 = s.add_image(LumenImage(CAM[20:320, 200:500].astype(np.uint8), name="c2"))
    out = s.run_op("registration.pairwise_stitch", {"image1": c1.id, "image2": c2.id, "fusion": fusion})
    st = s.images[out["new_images"][0]]
    r = s.results.rows[-1]
    assert r["dx"] == 200 and r["dy"] == 20 and r["Correlation"] > 0.99
    assert (st.width, st.height) == (500, 320)
    if fusion == "Overlay (channels)":
        assert st.nc == 2
        assert np.array_equal(st.data[0, 0, 0, :300, :300], CAM[0:300, 0:300].astype(np.uint8))
        assert np.array_equal(st.data[0, 0, 1, 20:320, 200:500], CAM[20:320, 200:500].astype(np.uint8))
    else:
        assert st.nc == 1 and st.dtype == np.uint8
        # covered area equals the original scene (identical content in the overlap)
        assert np.abs(st.data[0, 0, 0, 20:300, :].astype(float) - CAM[20:300, 0:500]).max() <= 1
        assert np.abs(st.data[0, 0, 0, :20, :300].astype(float) - CAM[:20, 0:300]).max() <= 1


def _write_grid(folder, tile=200, overlap=0.2, jitter=True, writer="tifffile"):
    import tifffile
    from lumen.core.io import write_image
    step = int(tile * (1 - overlap))
    truth = []
    k = 0
    for r in range(2):
        for c in range(2):
            y = r * step + (3 if (jitter and r) else 0)
            x = c * step + (-4 if (jitter and c) else 0)
            t = CAM[y:y + tile, x:x + tile].astype(np.uint8)
            if writer == "tifffile":
                tifffile.imwrite(str(folder / f"tile_{k:02d}.tif"), t)
            else:
                write_image(LumenImage(t, name=f"tile_{k:02d}"), str(folder / f"tile_{k:02d}.tif"))
            truth.append((x, y))
            k += 1
    return truth


def test_find_offset_on_smooth_images_and_noisy_slivers():
    """Whitened phase correlation loses the true peak on smooth / blurred images (it amplifies their empty high
    frequencies), so Pairwise Stitching / Register to Image placed them at a wrong offset with a small positive NCC.
    Candidates from the plain cross-correlation (credible overlaps only) and Fisher-z ranking fix that without
    trading the noisy case (a 100 px crop in N(0, 60) noise, NCC ~0.18) for a chance sliver overlap."""
    from lumen.core.ops.registration import _find_offset
    half = CAM[::2, ::2]
    for sigma in (1.5, 3.0, 5.0):
        sm = ndi.gaussian_filter(half, sigma)
        dy, dx, ncc = _find_offset(sm, sm[20:90, 30:120])
        assert (dy, dx) == (20, 30) and ncc > 0.99, (sigma, dy, dx, ncc)
    rng = np.random.default_rng(0)
    tex = ndi.gaussian_filter(rng.random((200, 240)), 2.0).astype(np.float32)
    assert _find_offset(tex, tex[20:90, 30:120])[:2] == (20, 30)
    noisy = CAM + rng.normal(0, 60, CAM.shape).astype(np.float32)
    crop = CAM[300:400, 300:400] + rng.normal(0, 60, (100, 100)).astype(np.float32)
    dy, dx, ncc = _find_offset(noisy, crop)
    assert (dy, dx) == (300, 300) and 0.1 < ncc < 0.4
    assert _find_offset(half, half[20:90, 30:120])[:2] == (20, 30)  # textured control
    # the expected / max_dev neighbourhood is still honoured; no overlap at all gives NCC -1
    assert _find_offset(half, half[20:90, 30:120], expected=(20, 30), max_dev=(4, 4))[:2] == (20, 30)
    assert _find_offset(half, half[20:90, 30:120], expected=(200, 100), max_dev=(2, 2))[2] < 0
    # op level: Register to Image (Translation) and Pairwise Stitching place the blurred crop correctly
    sm = ndi.gaussian_filter(half, 1.5)
    s = Session()
    a = s.add_image(LumenImage(sm, name="cam"))
    b = s.add_image(LumenImage(sm[20:90, 30:120].copy(), name="crop"))
    s.select(b.id)
    s.run_op("registration.register_to_image", {"target": a.id, "mode": "Translation"})
    r = s.results.rows[-1]
    assert abs(r["dx"] - 30) < 0.5 and abs(r["dy"] - 20) < 0.5
    out = s.run_op("registration.pairwise_stitch", {"image1": a.id, "image2": b.id, "fusion": "Average"})
    r = s.results.rows[-1]
    assert r["dx"] == 30 and r["dy"] == 20 and r["Correlation"] > 0.99
    assert np.abs(s.images[out["new_images"][0]].data[0, 0, 0] - sm).max() < 1e-3


def test_grid_stitch(tmp_path):
    truth = _write_grid(tmp_path)
    (tmp_path / "notes.txt").write_text("ignore me")
    s = Session()
    out = s.run_op("registration.grid_stitch", {"folder": str(tmp_path), "pattern": "tile_*.tif", "grid_x": 2, "grid_y": 2, "overlap_percent": 20, "refine": True})
    st = s.images[out["new_images"][0]]
    tab = s.tables["Tile positions"]
    assert tab["columns"] == ["Tile", "File", "Row", "Col", "X", "Y"]
    pos = [(row[4], row[5]) for row in tab["rows"]]
    assert pos == truth
    assert (st.width, st.height) == (356, 363)
    assert "4/4 neighbour pairs" in out["message"]
    assert np.abs(st.data[0, 0, 0].astype(float) - CAM[:363, :356]).max() <= 1
    # pure grid placement ignores the jitter
    out = s.run_op("registration.grid_stitch", {"folder": str(tmp_path), "pattern": "tile_*.tif", "grid_x": 2, "grid_y": 2, "overlap_percent": 20, "refine": False})
    st2 = s.images[out["new_images"][0]]
    assert (st2.width, st2.height) == (360, 360)
    assert [(row[4], row[5]) for row in s.tables["Tile positions"]["rows"]] == [(0, 0), (160, 0), (0, 160), (160, 160)]
    # regex pattern + snake order (same files, different grid assignment) still stitch
    out = s.run_op("registration.grid_stitch", {"folder": str(tmp_path), "pattern": r"tile_\d+\.tif", "grid_x": 2, "grid_y": 2, "overlap_percent": 20, "order": "Snake by rows", "refine": True})
    assert out["new_images"]
    with pytest.raises(ValueError, match="need 9"):
        s.run_op("registration.grid_stitch", {"folder": str(tmp_path), "pattern": "tile_*.tif", "grid_x": 3, "grid_y": 3})


def test_grid_stitch_reads_lumen_tiffs(tmp_path):
    truth = _write_grid(tmp_path, jitter=False, writer="lumen")
    s = Session()
    out = s.run_op("registration.grid_stitch", {"folder": str(tmp_path), "pattern": "*.tif", "grid_x": 2, "grid_y": 2, "overlap_percent": 20, "refine": True})
    st = s.images[out["new_images"][0]]
    assert [(row[4], row[5]) for row in s.tables["Tile positions"]["rows"]] == truth
    assert (st.width, st.height) == (360, 360)


def test_drift_correct_3d():
    rng = np.random.default_rng(1)
    vol = ndi.gaussian_filter(rng.random((6, 64, 64)).astype(np.float32), 1.5) * 1000
    drift = [(0, 0, 0), (1, 2, -3), (1, 4, -5), (2, 5, -6)]  # (dz, dy, dx) applied to each frame
    frames = [np.roll(np.roll(np.roll(vol, dz, axis=0), dy, axis=1), dx, axis=2) for dz, dy, dx in drift]
    s = Session()
    img = s.add_image(LumenImage(np.stack(frames)[:, :, None], name="drift", axes="TZCYX"))
    out = s.run_op("registration.drift_correct", {"correct_z": True, "reference": "First frame", "create_new": True})
    rows = alignment_rows(s)
    assert len(rows) == 4 and not s.results.rows and s.tables["Alignment"]["columns"] == ["Label", "Frame", "dx", "dy", "dz"]
    for r, (dz, dy, dx) in zip(rows, drift):
        assert abs(r["dx"] + dx) < 0.5 and abs(r["dy"] + dy) < 0.5 and abs(r["dz"] + dz) < 0.5
    new = s.images[out["new_images"][0]]
    assert new.shape == img.shape
    inner = (slice(2, 4), slice(8, -8), slice(8, -8))
    for t in range(1, 4):
        assert np.abs(new.data[t, :, 0][inner] - new.data[0, :, 0][inner]).mean() < 0.05 * vol.std()
    s.run_op("registration.drift_correct", {"reference": "Previous frame"})
    s.add_image(LumenImage(vol, name="z-only", axes="ZYX"))
    with pytest.raises(ValueError, match="T > 1"):
        s.run_op("registration.drift_correct", {})


def test_registration_extra_ops_registered():
    from lumen.core.registry import REGISTRY
    for op_id in ("registration.elastic_align", "registration.elastic_pair", "registration.apply_deformation", "registration.align_folder",
                  "registration.phase_correlation", "registration.feature_correspondences"):
        spec = REGISTRY.get(op_id)
        assert spec.to_dict()["menu"] == "Plugins/Registration" and spec.params
    assert REGISTRY.get("registration.align_folder").to_dict()["requires_image"] is False
    assert REGISTRY.get("registration.elastic_align").kind == "image" and REGISTRY.get("registration.elastic_pair").kind == "analysis"
