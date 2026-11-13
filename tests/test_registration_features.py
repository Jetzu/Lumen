"""Registration estimators and feature extraction (lumen/core/ops/registration_features.py, timelapse_ops.py,
the estimator choice of registration.align_stack and detect_dog_nd of registration3d.py)."""
import math

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.data import camera
from skimage.transform import SimilarityTransform, warp

from lumen.core import ops  # noqa: F401
from lumen.core.image import Calibration, LumenImage
from lumen.core.ops import registration_features as rf
from lumen.core.ops.registration import ALIGN_MODES, ESTIMATORS, REFS, _decompose, _warp
from lumen.core.ops.registration3d import detect_dog_nd
from lumen.core.ops.timelapse_ops import phase_map
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

CAM = camera().astype(np.float32)
BASE = CAM[100:356, 100:356]
SHIFTS = [(0, 0), (3, -2), (5, 4), (-4, 6), (2, 2)]  # (dy, dx) per slice
NEW_OPS = ["registration.extract_correspondences", "registration.descriptor_registration", "registration.descriptor_series", "registration.landmark_pairs", "timelapse.phase_map"]


def alignment_rows(s):
    tab = s.tables["Alignment"]
    return [dict(zip(tab["columns"], r)) for r in tab["rows"]]


def shifted_stack(shifts=SHIFTS, size=200, origin=150, channels=1):
    """Slice k shows the camera scene displaced by (−dy, −dx): aligning it onto slice 0 needs (+dy, +dx)."""
    planes = []
    for dy, dx in shifts:
        a = CAM[origin + dy:origin + dy + size, origin + dx:origin + dx + size]
        planes.append(np.stack([a] + [np.roll(a, 7 * (c + 1), axis=0) for c in range(channels - 1)]))
    return np.stack(planes).astype(np.uint8)


def similarity_matrix(shape, angle, scale, dx, dy):
    """3×3 (x, y) matrix: rotation by angle (deg) and scale about the image centre, then a shift."""
    h, w = shape
    c = np.array([(w - 1) / 2, (h - 1) / 2])
    m = SimilarityTransform(scale=scale, rotation=np.deg2rad(angle)).params.copy()
    m[:2, 2] = c - m[:2, :2] @ c + np.array([dx, dy])
    return m


def moved(img, m):
    """The scene seen through the inverse of m, so that warping it with m (moving → reference) gives back ``img``."""
    return warp(img, m, order=1, preserve_range=True).astype(np.float32)


def beads_2d(pts, shape=(256, 256), sigma=1.5, amp=1000.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    img = np.zeros(shape, np.float32)
    for x, y in pts:
        img += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return img


def beads_3d(pts, shape=(24, 96, 96), sigma=(1.0, 1.5, 1.5)):
    vol = np.zeros(shape, np.float32)
    for x, y, z in pts:
        vol[int(round(z)), int(round(y)), int(round(x))] = 1000
    return ndi.gaussian_filter(vol, sigma)


# ---------------------------------------------------------------- registry
def test_new_ops_registered():
    for op_id in NEW_OPS:
        spec = REGISTRY.get(op_id)
        assert spec.params and spec.description and spec.keywords, op_id
    assert REGISTRY.get("registration.extract_correspondences").to_dict()["menu"] == "Plugins/Feature Extraction"
    assert REGISTRY.get("timelapse.phase_map").to_dict()["menu"] == "Plugins/Time Lapse"
    assert REGISTRY.get("registration.descriptor_registration").kind == "image" and REGISTRY.get("registration.landmark_pairs").kind == "image"
    spec = REGISTRY.get("registration.align_stack")
    names = {p.name: p for p in spec.params}
    assert names["estimator"].choices == ESTIMATORS and names["mode"].choices == ALIGN_MODES and names["reference"].choices == REFS
    assert "Current slice" in REFS and "Bilinear" in ALIGN_MODES and "Scaled rotation" in ALIGN_MODES
    for k in ("initial_blur", "steps_per_octave", "min_image_size", "max_image_size", "descriptor_size", "orientation_bins", "max_ratio", "max_alignment_error", "min_inlier_ratio"):
        assert k in names and names[k].advanced
    # the old correspondences command stays
    assert REGISTRY.get("registration.feature_correspondences").params


# ---------------------------------------------------------------- DoG detector
def test_detect_dog_nd_bright_dark_subpixel():
    img = beads_2d([(40.3, 60.6), (120.0, 30.0), (200.7, 180.2)], amp=100.0)
    pts, resp = detect_dog_nd(img, 1.8, "Bright", 0.05, 10)
    assert pts.shape == (3, 2) and resp[0] >= resp[-1]
    for x, y in [(40.3, 60.6), (120.0, 30.0), (200.7, 180.2)]:
        assert np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y)) < 0.3
    dark = 100 - img
    pts_d, _ = detect_dog_nd(dark, 1.8, "Dark", 0.05, 10)
    assert len(pts_d) == 3
    both, _ = detect_dog_nd(dark, 1.8, "Bright and dark", 0.05, 10)
    assert len(both) >= 3
    assert detect_dog_nd(np.zeros((32, 32), np.float32), 1.5)[0].shape == (0, 2)
    vol = beads_3d([(20, 30, 8), (60, 70, 15)])
    p3, _ = detect_dog_nd(vol, (0.7, 1.2, 1.2), "Bright", 0.1, 5)
    assert p3.shape[1] == 3 and np.min(np.hypot(p3[:, 0] - 20, p3[:, 1] - 30)) < 0.5 and np.min(np.abs(p3[:, 2] - 8)) < 0.5


# ---------------------------------------------------------------- Align Stack: SIFT estimator
def test_align_stack_sift_translation_and_rotation():
    s = Session()
    img = s.add_image(LumenImage(shifted_stack(channels=2), name="shifted", axes="ZCYX"))
    out = s.run_op("registration.align_stack", {"axis": "Z", "mode": "Translation", "estimator": ESTIMATORS[1], "reference": "First slice", "create_new": True, "show_info": True})
    assert "SIFT" in out["message"] and f"{len(SHIFTS) - 1} slices by SIFT matching" in out["message"]
    rows = alignment_rows(s)
    assert s.tables["Alignment"]["columns"] == ["Label", "Slice", "dx", "dy", "Angle", "Scale", "Features", "Matches", "Inliers", "SIFT"]
    for r, (dy, dx) in zip(rows, SHIFTS):
        assert abs(r["dx"] - dx) < 0.5 and abs(r["dy"] - dy) < 0.5 and abs(r["Angle"]) < 1e-6
        if r["Slice"] > 1:
            assert r["SIFT"] and r["Inliers"] >= 10 and r["Matches"] >= r["Inliers"]
    new = s.images[out["new_images"][0]]
    assert new.shape == img.shape and new.dtype == np.uint8
    for c in range(2):  # all channels transformed by the driving channel's transform
        ctr = new.data[0, :, c, 40:160, 40:160].astype(float)
        assert np.abs(ctr - ctr[:1]).mean(axis=(1, 2)).max() < 2.0
    assert "Matches:" in out["log"]
    # rigid body: a rotated second slice, matrices logged
    m = similarity_matrix(BASE.shape, 5.0, 1.0, 3.0, -2.0)
    s.add_image(LumenImage(np.stack([BASE, moved(BASE, m)]).astype(np.uint8), name="rot", axes="ZYX"))
    out = s.run_op("registration.align_stack", {"axis": "Z", "mode": "Rigid body", "estimator": ESTIMATORS[1], "reference": "First slice", "show_matrix": True})
    r = alignment_rows(s)[-1]
    assert abs(r["Angle"] - 5.0) < 0.3 and abs(r["Scale"] - 1) < 0.01
    assert "rot:Z2:" in out["log"]
    img = s.active
    ctr = img.data[0, :, 0, 60:200, 60:200].astype(float)
    assert np.abs(ctr[1] - ctr[0]).mean() < 4.0


# ---------------------------------------------------------------- Align Stack: intensity (StackReg) estimator
@pytest.mark.parametrize("mode,angle,scale,dx,dy", [("Translation", 0, 1, 4.3, -2.7), ("Rigid body", 5, 1, 3, -2), ("Scaled rotation", -4, 1.06, -3, 5), ("Affine", 3, 1.03, 2, 2), ("Bilinear", 2, 1, 1, -1)])
def test_pyramid_transform_recovers_models(mode, angle, scale, dx, dy):
    m = similarity_matrix(BASE.shape, angle, scale, dx, dy)
    mov = moved(BASE, m)
    tf, info = rf.pyramid_transform(BASE, mov, mode)
    d = _decompose(tf)
    assert abs(d["Angle"] - angle) < 0.1 and abs(d["Scale"] - scale) < 0.005
    truth = _decompose(m)
    assert abs(d["dx"] - truth["dx"]) < 0.3 and abs(d["dy"] - truth["dy"]) < 0.3
    assert info["levels"] >= 3 and info["correlation"] > 0.98
    w = _warp(mov, tf, 1)
    assert np.abs(w[40:-40, 40:-40] - BASE[40:-40, 40:-40]).mean() < 4.0
    if mode == "Bilinear":
        assert isinstance(tf, rf.BilinearTransform) and tf.matrix.shape == (3, 3)


def test_align_stack_intensity_estimator_and_current_slice_chain():
    s = Session()
    img = s.add_image(LumenImage(shifted_stack(channels=2), name="shifted", axes="ZCYX"))
    img.display.z = 2
    out = s.run_op("registration.align_stack", {"axis": "Z", "mode": "Translation", "estimator": ESTIMATORS[2], "reference": "Current slice", "create_new": True, "show_info": True})
    assert "intensity pyramid" in out["message"] and "ref: current slice" in out["message"]
    rows = alignment_rows(s)
    assert s.tables["Alignment"]["columns"] == ["Label", "Slice", "dx", "dy", "Angle", "Scale", "Correlation", "Levels"]
    ref_dy, ref_dx = SHIFTS[2]
    for r, (dy, dx) in zip(rows, SHIFTS):  # everything is expressed relative to slice 3 (the displayed one)
        assert abs(r["dx"] - (dx - ref_dx)) < 0.5 and abs(r["dy"] - (dy - ref_dy)) < 0.5
        if r["Slice"] != 3:
            assert r["Correlation"] > 0.9 and r["Levels"] >= 2
    new = s.images[out["new_images"][0]]
    for c in range(2):
        ctr = new.data[0, :, c, 40:160, 40:160].astype(float)
        assert np.abs(ctr - ctr[2:3]).mean(axis=(1, 2)).max() < 2.0
    # scaled rotation and bilinear through the op (previous-slice chain; bilinear is registered onto the aligned previous slice)
    m1 = similarity_matrix(BASE.shape, 4.0, 1.05, 2.0, -3.0)
    stack = np.stack([BASE, moved(BASE, m1), moved(moved(BASE, m1), m1)]).astype(np.uint8)
    for mode in ("Scaled rotation", "Bilinear"):
        s.add_image(LumenImage(stack, name=mode, axes="ZYX"))
        s.run_op("registration.align_stack", {"axis": "Z", "mode": mode, "estimator": ESTIMATORS[2], "reference": "Previous slice"})
        rows = alignment_rows(s)
        assert abs(rows[1]["Angle"] - 4.0) < 0.3 and abs(rows[2]["Angle"] - 8.0) < 0.5
        if mode == "Scaled rotation":
            assert abs(rows[1]["Scale"] - 1.05) < 0.01 and abs(rows[2]["Scale"] - 1.05 ** 2) < 0.02
        img = s.active
        ctr = img.data[0, :, 0, 70:190, 70:190].astype(float)
        assert np.abs(ctr - ctr[:1]).mean(axis=(1, 2)).max() < 6.0
    # feature estimators have no bilinear model: affine + note; the default estimator knows scaled rotations
    s.add_image(LumenImage(stack, name="feat", axes="ZYX"))
    out = s.run_op("registration.align_stack", {"axis": "Z", "mode": "Bilinear", "estimator": ESTIMATORS[0], "reference": "First slice", "create_new": True})
    assert "bilinear → affine" in out["message"] and abs(alignment_rows(s)[1]["Scale"] - 1.05) < 0.03
    s.add_image(LumenImage(stack, name="feat2", axes="ZYX"))
    out = s.run_op("registration.align_stack", {"axis": "Z", "mode": "Scaled rotation", "estimator": ESTIMATORS[0], "reference": "First slice"})
    assert "feature-based" in out["message"] and abs(alignment_rows(s)[1]["Scale"] - 1.05) < 0.01 and abs(alignment_rows(s)[1]["Angle"] - 4.0) < 0.2


# ---------------------------------------------------------------- Extract Correspondences
@pytest.mark.parametrize("method", rf.CORRESPONDENCE_METHODS)
def test_extract_correspondences_methods(method):
    s = Session()
    a = s.add_image(LumenImage(BASE.astype(np.uint8), name="a"))
    b = s.add_image(LumenImage(np.roll(np.roll(BASE, 5, axis=0), -7, axis=1).astype(np.uint8), name="b"))
    s.select(a.id)
    n_rois = len(s.roi_manager.rois)
    out = s.run_op("registration.extract_correspondences", {"target": b.id, "method": method, "model": "Translation", "min_correlation": 0.7})
    row = s.results.rows[-1]
    assert row["Method"] == method and row["Model"] == "Translation" and row["N inliers"] >= 8 and row["N inliers"] <= row["N matches"]
    assert abs(row["dx"] + 7) < 0.6 and abs(row["dy"] - 5) < 0.6 and row["RMS residual"] < 2.0
    tab = s.tables["Correspondences"]
    assert tab["columns"] == ["Match", "X1", "Y1", "X2", "Y2", "Residual"] and len(tab["rows"]) == row["N inliers"]
    assert out["rois_added"] == 2 and len(s.roi_manager.rois) == n_rois + 2
    ra, rb = s.roi_manager.rois[-2:]
    assert ra["name"] == "Matches" and ra["image_id"] == a.id and rb["image_id"] == b.id and len(ra["points"]) == row["N inliers"]
    corr = s.settings["correspondences"]
    assert corr["source"] == a.id and corr["target"] == b.id and corr["method"] == method and len(corr["src"]) == row["N inliers"]
    assert f"{row['N inliers']} inliers" in out["message"]
    # the stored correspondences drive Landmark Transform
    out = s.run_op("image.transform_landmarks", {"model": "Translation"})
    assert abs(s.results.rows[-1]["dx"] + 7) < 0.6


def test_extract_correspondences_rigid_and_errors():
    s = Session()
    m = similarity_matrix(BASE.shape, 6.0, 1.0, 2.0, 1.0)
    a = s.add_image(LumenImage(BASE.astype(np.uint8), name="a"))
    b = s.add_image(LumenImage(moved(BASE, m).astype(np.uint8), name="b"))
    s.select(a.id)
    s.run_op("registration.extract_correspondences", {"target": b.id, "method": "SIFT", "model": "Rigid body", "add_rois": False})
    row = s.results.rows[-1]
    # a maps onto b with the inverse of m (b = a seen through m⁻¹)
    assert abs(row["Angle"] + 6.0) < 0.3 and abs(row["Scale"] - 1) < 0.01
    s.run_op("registration.extract_correspondences", {"target": b.id, "method": "MOPS", "model": "Similarity", "descriptor_size": 12})
    assert abs(s.results.rows[-1]["Angle"] + 6.0) < 0.5
    with pytest.raises(ValueError, match="different image"):
        s.run_op("registration.extract_correspondences", {"target": a.id})
    with pytest.raises(ValueError, match="target"):
        s.run_op("registration.extract_correspondences", {})
    flat = s.add_image(LumenImage(np.zeros((128, 128), np.uint8), name="flat"))
    with pytest.raises(ValueError, match="matches"):
        s.run_op("registration.extract_correspondences", {"target": a.id, "method": "SIFT"})
    assert flat.id


def test_mops_descriptors_match_known_shift():
    mov = np.roll(np.roll(BASE, 5, axis=0), -7, axis=1)
    ka, da, sa = rf.mops_features(BASE)
    kb, db, sb = rf.mops_features(mov)
    assert da.shape[1] == 256 and len(ka) == len(sa) and np.all(sa > 0)
    assert np.allclose(da.mean(axis=1), 0, atol=1e-4) and np.allclose(da.std(axis=1), 1, atol=1e-3)
    mt = rf.match_features(da, db, 0.92)
    d = kb[mt[:, 1]] - ka[mt[:, 0]]
    good = np.sum((np.abs(d[:, 0] - 5) < 1) & (np.abs(d[:, 1] + 7) < 1))
    assert len(mt) >= 30 and good / len(mt) > 0.7
    m, info = rf.feature_transform(BASE, mov, "Translation", "MOPS")  # the matrix maps the moving image onto the reference
    assert info["features_used"] and abs(m[0, 2] - 7) < 0.5 and abs(m[1, 2] + 5) < 0.5
    # a flat image has no features and no transform
    m, info = rf.feature_transform(np.zeros((64, 64), np.float32), np.zeros((64, 64), np.float32), "Translation", "MOPS")
    assert m is None and info["inliers"] == 0


# ---------------------------------------------------------------- Descriptor-based registration (2-D)
def test_descriptor_registration_2d_rigid():
    rng = np.random.default_rng(0)
    pts = rng.random((120, 2)) * 200 + 28
    m = similarity_matrix((256, 256), 7.0, 1.0, 6.0, -4.0)  # a (pixel-centre) → b
    pts_b = pts @ m[:2, :2].T + m[:2, 2]
    s = Session()
    a = s.add_image(LumenImage(np.stack([beads_2d(pts), beads_2d(pts) * 0.5]), name="a", axes="CYX"))
    b = s.add_image(LumenImage(beads_2d(pts_b), name="b"))
    other = s.add_image(LumenImage(beads_2d(pts) * 2, name="other"))
    s.select(a.id)
    out = s.run_op("registration.descriptor_registration", {"target": b.id, "model": "Rigid body", "size": 5, "also_apply_to": other.id})
    row = s.results.rows[-1]
    assert row["Label"] == "a → b" and row["Model"] == "Rigid body" and row["N inliers"] >= 80 and row["N inliers"] <= row["N candidates"] <= row["N detections"]
    assert abs(row["Angle"] - 7.0) < 0.1 and abs(row["Scale"] - 1) < 1e-3 and abs(row["dx"] - m[0, 2]) < 0.3 and abs(row["dy"] - m[1, 2]) < 0.3 and row["RMS residual"] < 0.3
    assert len(out["new_images"]) == 2
    res_other, res = (s.images[i] for i in out["new_images"])
    assert res.name == "a on b" and res.shape == (1, 1, 2, 256, 256) and res.dtype == np.float32 and res_other.name == "other on b"
    inner = (slice(40, -40), slice(40, -40))
    # sharp synthetic beads lose a few per cent of their peaks to the bilinear resampling; the transform itself is checked above
    assert np.abs(res.data[0, 0, 0][inner] - b.data[0, 0, 0][inner]).mean() < 0.1 * b.data.mean() + 0.5
    assert np.abs(res.data[0, 0, 1][inner] - 0.5 * b.data[0, 0, 0][inner]).mean() < 0.1 * b.data.mean() + 0.5
    assert np.abs(res_other.data[0, 0, 0][inner] - 2 * b.data[0, 0, 0][inner]).mean() < 0.2 * b.data.mean() + 1
    tab = s.tables["Correspondences"]
    assert tab["columns"] == ["Match", "X1", "Y1", "X2", "Y2", "Residual"] and len(tab["rows"]) == row["N inliers"]
    names = [r["name"] for r in s.roi_manager.rois]
    assert names.count("Matches") == 2
    assert s.settings["correspondences"]["method"] == "Descriptor-based"
    # similarity / affine models, in place (undoable), and the translation model fails on a rotated scene
    for model in ("Similarity", "Affine"):
        s.select(a.id)
        s.run_op("registration.descriptor_registration", {"target": b.id, "model": model, "size": 5, "create_new": False, "add_rois": False})
        assert abs(s.results.rows[-1]["Angle"] - 7.0) < 0.2 and abs(s.results.rows[-1]["Scale"] - 1) < 0.01
        s.undo(a.id)
    with pytest.raises(ValueError, match="matching failed"):
        s.run_op("registration.descriptor_registration", {"target": b.id, "model": "Translation", "size": 5, "min_inlier_ratio": 0.5})
    with pytest.raises(ValueError, match="different image"):
        s.run_op("registration.descriptor_registration", {"target": a.id})


# ---------------------------------------------------------------- Descriptor-based registration (3-D)
def test_descriptor_registration_3d_translation_calibrated():
    rng = np.random.default_rng(1)
    pts = rng.random((150, 3)) * np.array([76, 76, 18]) + np.array([10, 10, 3])
    vol = beads_3d(pts)
    shift = (1.5, -2.25, 3.5)  # (dz, dy, dx) applied to the moving volume
    mov = ndi.shift(vol, shift, order=1)
    cal = Calibration(pixel_width=0.5, pixel_height=0.5, pixel_depth=1.0, unit="µm")
    s = Session()
    a = s.add_image(LumenImage(vol[:, None], name="ref", axes="ZCYX"))
    b = s.add_image(LumenImage(np.stack([mov, mov * 0.5], axis=1), name="mov", axes="ZCYX"))
    a.calibration, b.calibration = cal, Calibration(**cal.to_dict())
    s.select(b.id)
    out = s.run_op("registration.descriptor_registration", {"target": a.id, "model": "Translation", "size": 4, "threshold": 0.05})
    row = s.results.rows[-1]
    assert row["Label"] == "mov → ref" and row["N inliers"] >= 80
    assert abs(row["dx"] + shift[2]) < 0.3 and abs(row["dy"] + shift[1]) < 0.3 and abs(row["dz"] + shift[0]) < 0.3 and abs(row["Angle"]) < 1e-6
    assert abs(row["dx [µm]"] - row["dx"] * 0.5) < 1e-9 and abs(row["dz [µm]"] - row["dz"] * 1.0) < 1e-9
    assert s.tables["Correspondences"]["columns"] == ["Match", "X1", "Y1", "Z1", "X2", "Y2", "Z2", "Residual"]
    new = s.images[out["new_images"][0]]
    assert new.shape == (1, 24, 2, 96, 96) and new.calibration.unit == "µm"
    inner = (slice(3, -3), slice(8, -8), slice(8, -8))
    ideal = np.abs(ndi.shift(mov, tuple(-v for v in shift), order=1)[inner] - vol[inner]).mean()  # the unavoidable double-interpolation error
    for c, f in ((0, 1.0), (1, 0.5)):
        assert np.abs(new.data[0, :, c][inner] - f * vol[inner]).mean() < 1.1 * f * ideal + 0.05 * vol.std()
    assert np.array(s.settings["registration3d"]["matrix"]).shape == (4, 4)
    rois = [r for r in s.roi_manager.rois if r["name"].startswith("Matches z")]
    assert rois and all("z" in r["position"] for r in rois)
    # rigid 3-D with a small in-plane rotation
    ang = 4.0
    m2 = similarity_matrix((96, 96), ang, 1.0, 0.0, 0.0)
    rot = np.stack([moved(sl, m2) for sl in vol]).astype(np.float32)
    c = s.add_image(LumenImage(rot[:, None], name="rot", axes="ZCYX"))
    c.calibration = Calibration(**cal.to_dict())
    s.run_op("registration.descriptor_registration", {"target": a.id, "model": "Rigid body", "size": 4, "ransac_threshold": 2.0})
    row = s.results.rows[-1]
    assert abs(row["Angle"] - ang) < 0.3 and abs(row["Scale"] - 1) < 0.01 and row["N inliers"] >= 60  # rot (seen through m2⁻¹) → ref is m2
    assert c.id


def test_descriptor_series_2d_and_3d():
    rng = np.random.default_rng(2)
    pts = rng.random((100, 2)) * 180 + 38
    img = beads_2d(pts)
    frames = [ndi.shift(img, (-dy, -dx), order=1) for dy, dx in SHIFTS[:4]]  # aligning frame k onto frame 0 needs (+dy, +dx)
    s = Session()
    im = s.add_image(LumenImage(np.stack([np.stack([f, f * 0.5]) for f in frames]), name="series", axes="TCYX"))
    out = s.run_op("registration.descriptor_series", {"model": "Rigid body", "reference": "First frame", "size": 5, "create_new": True})
    rows = alignment_rows(s)
    assert s.tables["Alignment"]["columns"] == ["Label", "Frame", "dx", "dy", "Angle", "Scale", "Detections", "Candidates", "Inliers"]
    for r, (dy, dx) in zip(rows, SHIFTS[:4]):
        assert abs(r["dx"] - dx) < 0.3 and abs(r["dy"] - dy) < 0.3 and abs(r["Angle"]) < 0.2
        if r["Frame"] > 1:
            assert r["Inliers"] >= 50 and r["Detections"] >= 80
    new = s.images[out["new_images"][0]]
    inner = (slice(40, -40), slice(40, -40))
    for c in range(2):
        for t in range(1, 4):
            assert np.abs(new.data[t, 0, c][inner] - new.data[0, 0, c][inner]).mean() < 0.05 * img.mean() + 0.5
    assert "Descriptor-based series registration over 4 frames" in out["message"] and "without a consistent match" not in out["message"]
    # chained reference, in place and undoable; a frame without beads keeps the previous transform
    s.select(im.id)
    before = im.data.copy()
    s.run_op("registration.descriptor_series", {"model": "Translation", "reference": "Previous frame", "size": 5})
    assert all(abs(r["dx"] - dx) < 0.3 for r, (dy, dx) in zip(alignment_rows(s), SHIFTS[:4]))
    s.undo(im.id)
    assert np.array_equal(im.data, before)
    blank = np.stack(frames[:2] + [np.zeros_like(img)])
    s.add_image(LumenImage(blank, name="gap", axes="TYX"))
    out = s.run_op("registration.descriptor_series", {"model": "Translation", "reference": "Previous frame", "size": 5})
    rows = alignment_rows(s)
    assert "1 frame without a consistent match" in out["message"] and rows[2]["dx"] == rows[1]["dx"] and rows[2]["Inliers"] == 0
    with pytest.raises(ValueError, match="T > 1"):
        s.run_op("registration.descriptor_series", {}, image_id=s.add_image(LumenImage(img, name="single")).id)
    # 3-D + t
    pts3 = rng.random((120, 3)) * np.array([76, 76, 18]) + np.array([10, 10, 3])
    vol = beads_3d(pts3)
    drift = [(0, 0, 0), (1, 2, -3), (2, 3, -5)]  # (dz, dy, dx) applied to each frame
    vols = np.stack([ndi.shift(vol, d, order=1) for d in drift])
    s.add_image(LumenImage(vols[:, :, None], name="drift3d", axes="TZCYX"))
    out = s.run_op("registration.descriptor_series", {"model": "Translation", "reference": "First frame", "size": 4, "create_new": True})
    rows = alignment_rows(s)
    assert s.tables["Alignment"]["columns"][:6] == ["Label", "Frame", "dx", "dy", "dz", "Angle"]
    for r, (dz, dy, dx) in zip(rows, drift):
        assert abs(r["dx"] + dx) < 0.3 and abs(r["dy"] + dy) < 0.3 and abs(r["dz"] + dz) < 0.3
    new = s.images[out["new_images"][0]]
    inner3 = (slice(3, -3), slice(8, -8), slice(8, -8))
    for t in range(1, 3):
        assert np.abs(new.data[t, :, 0][inner3] - new.data[0, :, 0][inner3]).mean() < 0.1 * vol.std()
    assert "3-D" in out["message"]


# ---------------------------------------------------------------- manual landmark pairs
def test_landmark_pairs():
    m = similarity_matrix(BASE.shape, 8.0, 1.0, 5.0, -3.0)  # a (pixel-centre) → b
    s = Session()
    a = s.add_image(LumenImage(BASE.astype(np.uint8), name="a"))
    b = s.add_image(LumenImage(warp(BASE, np.linalg.inv(m), order=1, preserve_range=True).astype(np.uint8), name="b"))
    src = np.array([[40.0, 50.0], [200.0, 60.0], [120.0, 210.0], [60.0, 160.0]])
    dst = src @ m[:2, :2].T + m[:2, 2]
    s.roi_manager.add({"type": "point", "points": (src + 0.5).tolist(), "name": "landmarks:A"}, image_id=a.id, position={"t": 0, "z": 0, "c": 0})
    s.roi_manager.add({"type": "point", "points": (dst + 0.5).tolist(), "name": "landmarks:B"}, image_id=b.id, position={"t": 0, "z": 0, "c": 0})
    s.select(a.id)
    out = s.run_op("registration.landmark_pairs", {"model": "Rigid body"})
    row = s.results.rows[-1]
    assert row["Label"] == "a → b" and row["Source ROI"] == "landmarks:A" and row["Target ROI"] == "landmarks:B" and row["N landmarks"] == 4
    assert abs(row["Angle"] - 8.0) < 1e-6 and abs(row["dx"] - m[0, 2]) < 1e-6 and abs(row["dy"] - m[1, 2]) < 1e-6 and row["RMS residual"] < 1e-6
    new = s.images[out["new_images"][0]]
    assert new.shape == b.shape and new.name.startswith("a (rigid body")
    inner = (slice(40, -40), slice(40, -40))
    assert np.abs(new.data[0, 0, 0][inner].astype(float) - b.data[0, 0, 0][inner]).mean() < 3.0
    assert s.tables["Correspondences"]["columns"] == ["Match", "X1", "Y1", "X2", "Y2", "Residual"] and len(s.tables["Correspondences"]["rows"]) == 4
    assert s.settings["landmark_transform"]["model"] == "Rigid body" and s.settings["correspondences"]["method"] == "Manual landmarks"
    # compute only (explicit names, similarity), then 'Landmark Transform' reuses the stored pairs
    s.select(a.id)
    out = s.run_op("registration.landmark_pairs", {"model": "Similarity", "source_roi": "landmarks:A", "target_roi": "landmarks:B", "apply": False})
    assert not out["new_images"] and abs(s.results.rows[-1]["Scale"] - 1) < 1e-6
    s.run_op("image.transform_landmarks", {"model": "Affine"})
    assert abs(s.results.rows[-1]["Angle"] - 8.0) < 1e-6
    # errors
    s.roi_manager.add({"type": "point", "points": [[1, 1]], "name": "landmarks:C"}, image_id=b.id)
    with pytest.raises(ValueError, match="one to one"):
        s.run_op("registration.landmark_pairs", {"source_roi": "landmarks:A", "target_roi": "landmarks:C"})
    with pytest.raises(ValueError, match="No ROI named"):
        s.run_op("registration.landmark_pairs", {"source_roi": "nope", "target_roi": "landmarks:B"})
    s.roi_manager.clear()
    with pytest.raises(ValueError, match="two multi-point ROIs"):
        s.run_op("registration.landmark_pairs", {})


# ---------------------------------------------------------------- Phase Map
def test_phase_map_function_and_op():
    T = 48
    t = np.arange(T, dtype=np.float32)
    H, W = 16, 24
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    phi = np.deg2rad(yy * 10.0)  # phase gradient along y
    left = 5.0 * np.cos(2 * np.pi * t[:, None, None] / 8 + phi[None]) * (xx < 12)[None]
    right = 2.0 * np.cos(2 * np.pi * t[:, None, None] / 12 + 0.3) * (xx >= 12)[None]
    sig = (100 + 0.5 * t[:, None, None] + left + right).astype(np.float32)
    amp, ph, fr, power, freqs = phase_map(sig, 2, 0, "Linear", True)
    assert amp.shape == (H, W) and abs(amp[3, 2] - 5.0) < 0.3 and abs(amp[3, 20] - 2.0) < 0.2
    assert np.allclose(fr[:, :12], 1 / 8) and np.allclose(fr[:, 12:], 1 / 12)
    assert abs(math.degrees(float(ph[3, 2])) - 30.0) < 3.0 and abs(math.degrees(float(ph[9, 5])) - 90.0) < 3.0 and abs(float(ph[2, 20]) - 0.3) < 0.1
    assert len(power) == len(freqs) == T // 2 + 1
    with pytest.raises(ValueError, match="period range"):
        phase_map(sig, 30, 40)
    with pytest.raises(ValueError, match="3 frames"):
        phase_map(sig[:2])
    s = Session()
    img = s.add_image(LumenImage(np.stack([sig, sig * 0.0 + 7]), name="osc", axes="CTYX"))
    img.display.c = 0
    out = s.run_op("timelapse.phase_map", {"detrend": "Linear"})
    res = s.images[out["new_images"][0]]
    assert res.shape == (1, 1, 3, H, W) and res.dtype == np.float32 and res.name == "Phase map osc"
    assert [c.name for c in res.display.channels] == ["Amplitude", "Phase (°)", "Period"]
    assert abs(res.data[0, 0, 0, 3, 2] - 5) < 0.3 and abs(res.data[0, 0, 1, 3, 2] - 30) < 3 and abs(res.data[0, 0, 2, 3, 2] - 8) < 1e-4 and abs(res.data[0, 0, 2, 3, 20] - 12) < 1e-4
    row = s.results.rows[-1]
    assert row["Frames"] == T and abs(row["Dominant period (frames)"] - 8) < 1e-6 and row["Mean amplitude"] > 1 and "Frequency (cycles/frame)" in row
    assert s.last_plot["x_label"].startswith("Frequency") and len(s.last_plot["series"][0]["values"]) == T // 2
    assert "dominant period of the mean signal 8.00 frames" in out["message"]
    # relative phase, radians, Hz output with a frame interval, and a selection as the reference region
    s.select(img.id)
    with pytest.raises(ValueError, match="frame interval"):
        s.run_op("timelapse.phase_map", {"frequency_output": "Frequency (Hz)"})
    img.calibration.frame_interval = 0.5
    out = s.run_op("timelapse.phase_map", {"detrend": "Linear", "phase_unit": "Radians", "frequency_output": "Frequency (Hz)", "reference": "Selection"}, roi={"type": "rect", "x": 0, "y": 0, "w": 12, "h": 1})
    res = s.images[out["new_images"][0]]
    assert abs(res.data[0, 0, 2, 3, 2] - 0.25) < 1e-4 and abs(res.data[0, 0, 1, 0, 2]) < 0.05 and abs(res.data[0, 0, 1, 9, 5] - np.deg2rad(90)) < 0.06
    assert abs(s.results.rows[-1]["Frequency (Hz)"] - 0.25) < 1e-6 and "Hz" in s.last_plot["x_label"]
    # Z stacks: current slice only unless all_slices
    stack = s.add_image(LumenImage(np.stack([sig, sig * 2], axis=1), name="zt", axes="TZYX"))
    assert stack.nz == 2
    assert s.images[s.run_op("timelapse.phase_map", {}, image_id=stack.id)["new_images"][0]].nz == 1
    res = s.images[s.run_op("timelapse.phase_map", {"all_slices": True, "detrend": "Linear"}, image_id=stack.id)["new_images"][0]]
    assert res.nz == 2 and abs(res.data[0, 1, 0, 3, 2] - 10) < 0.6 and len(s.results.rows) >= 2
    with pytest.raises(ValueError, match="T > 2"):
        s.run_op("timelapse.phase_map", {}, image_id=s.add_image(LumenImage(sig[:2], name="short", axes="TYX")).id)


def test_bilinear_chain_survives_long_drift_and_jitter():
    """Bilinear + intensity with 'Previous slice' / 'Current slice' registers every slice onto the *aligned* previous
    slice. The chained estimate is seeded with the previous transform plus the raw inter-slice step and fitted on
    the valid window of the warped reference, so the zero border of the accumulated shift no longer locks the fit
    onto a distorted solution (the old chain collapsed beyond ~10 px: residual ~10 grey levels, Scale 1.1-1.2)."""
    from lumen.core.ops import registration as R
    rng = np.random.default_rng(0)
    big = ndi.gaussian_filter(rng.random((260, 280)) * 200 + 20, 2.0).astype(np.float32)

    def crops(shifts):  # the scene seen through a window shifted per slice: no synthetic border
        return np.stack([big[50 - sy:50 - sy + 160, 50 - sx:50 - sx + 180] for sy, sx in shifts])

    drift = [(round(2.5 * k), round(-2.3 * k)) for k in range(10)]  # up to (23, -21) px
    jitter = [(0, 0), (8, -6), (-5, 7), (9, 3), (-7, -8), (4, 9), (-9, 2)]
    for shifts, reference, anchor in ((drift, "Previous slice", 0), (drift, "Current slice", 3), (jitter, "Previous slice", 0), (jitter, "Current slice", 3)):
        s = Session()
        img = s.add_image(LumenImage(crops(shifts), name="drift", axes="ZYX"))
        img.display.z = anchor
        out = s.run_op("registration.align_stack", {"mode": "Bilinear", "estimator": ESTIMATORS[2], "reference": reference, "create_new": True, "show_info": True})
        al = s.images[out["new_images"][0]].data[0, :, 0]
        rows = alignment_rows(s)
        for k in range(len(shifts)):
            resid = float(np.abs(al[k, 40:120, 40:140] - al[anchor, 40:120, 40:140]).mean())
            assert resid < 1.0 and abs(rows[k]["Scale"] - 1.0) < 0.005, (reference, k, resid, rows[k])
            if k != anchor:
                assert rows[k]["Correlation"] > 0.99, (reference, k, rows[k])
    # re-expressing a transform on a window and back is exact, for matrices and bilinear models (x·y term included)
    tf = rf.BilinearTransform(np.array([[3.0, 1.02, 0.03, 2e-4], [-2.0, -0.01, 0.98, -1e-4]]), (160, 180))
    win = R._rebase_transform(tf, (20, 30), (25, 10), (160, 180), (100, 120))
    back = R._rebase_transform(win, (-20, -30), (-25, -10), (100, 120), (160, 180))
    assert np.allclose(back.coef, tf.coef, atol=1e-9) and back.shape == (160, 180)
    xm, ym = tf.coords()
    xw, yw = win.coords()
    assert np.allclose(xw + 10, xm[20:120, 30:150]) and np.allclose(yw + 25, ym[20:120, 30:150])
    m = np.array([[1.0, 0.1, 5.0], [-0.05, 1.0, -3.0], [0, 0, 1.0]])
    mw = R._rebase_transform(m, (20, 30), (25, 10), (160, 180), (100, 120))
    assert np.allclose(R._rebase_transform(mw, (-20, -30), (-25, -10), (100, 120), (160, 180)), m)
    assert np.allclose(mw @ np.array([7.0, 9.0, 1.0]), m @ np.array([17.0, 34.0, 1.0]) - np.array([30.0, 20.0, 0.0]))
