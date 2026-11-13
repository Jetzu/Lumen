"""Elastic / landmark / folder registration, phase correlation, SIFT correspondences, interactive-style
transforms, keyframe ROI interpolation, dense optic flow and colour clustering."""
import csv

import numpy as np
import pytest
from skimage.data import camera
from skimage.registration import phase_cross_correlation
from skimage.transform import AffineTransform, rotate, warp

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.io import read_image, write_image
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

CAM = camera().astype(np.float32)
BASE = CAM[100:300, 100:300]
INNER = (slice(20, -20), slice(20, -20))

NEW_OPS = {
    "registration.elastic_align": "Plugins/Registration", "registration.elastic_pair": "Plugins/Registration", "registration.apply_deformation": "Plugins/Registration",
    "registration.align_folder": "Plugins/Registration", "registration.phase_correlation": "Plugins/Registration", "registration.feature_correspondences": "Plugins/Registration",
    "process.optic_flow": "Plugins/Tracking", "image.transform_landmarks": "Image/Transform", "image.perspective": "Image/Transform", "image.shear": "Image/Transform",
    "image.warp_polar": "Image/Transform", "stacks.interpolate_rois": "Image/Stacks/Tools", "segment.color_clustering": "Plugins/Segmentation",
}


def deform(plane, amp=4.0, period=60.0):
    """Smooth sinusoidal non-rigid deformation of a plane."""
    H, W = plane.shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float32)
    v = amp * np.sin(2 * np.pi * cc / period)
    u = amp * np.cos(2 * np.pi * rr / period)
    return warp(plane, np.array([rr + v, cc + u]), order=1, mode="constant", preserve_range=True).astype(np.float32)


def mse(a, b):
    return float(np.mean((a[INNER].astype(float) - b[INNER].astype(float)) ** 2))


def table_rows(s, name):
    tab = s.tables[name]
    return [dict(zip(tab["columns"], r)) for r in tab["rows"]]


def test_new_ops_registered():
    for op_id, menu in NEW_OPS.items():
        spec = REGISTRY.get(op_id)
        assert spec.to_dict()["menu"] == menu, op_id
        assert spec.params and spec.description


# ---------------------------------------------------------------- elastic alignment
@pytest.mark.parametrize("method", ["TV-L1", "ILK"])
def test_elastic_align_reduces_residual(method):
    d1, d2 = deform(BASE, 4, 60), deform(BASE, 3, 80)
    s = Session()
    img = s.add_image(LumenImage(np.stack([BASE, d1, d2]).astype(np.uint8), name="def", axes="ZYX"))
    before = [mse(img.data[0, k, 0], BASE) for k in (1, 2)]
    assert min(before) > 500
    out = s.run_op("registration.elastic_align", {"axis": "Z", "reference": "First slice", "method": method, "create_new": True, "output_displacement": True})
    assert "Elastically aligned 3 slices" in out["message"]
    new = s.images[out["new_images"][-1]]
    assert new.name.startswith("Elastic") and s.active_id == new.id and new.dtype == np.uint8 and new.shape == img.shape
    after = [mse(new.data[0, k, 0], BASE) for k in (1, 2)]
    assert all(a < b / 4 for a, b in zip(after, before))
    assert np.array_equal(new.data[0, 0, 0], img.data[0, 0, 0])  # the reference slice is untouched
    disp = s.images[out["new_images"][0]]
    assert disp.shape == (1, 3, 1, 200, 200) and disp.dtype == np.float32
    assert float(disp.data[0, 0, 0].max()) == 0 and 1.0 < float(disp.data[0, 1, 0].mean()) < 6.0
    rows = table_rows(s, "Alignment")
    assert s.tables["Alignment"]["columns"] == ["Label", "Slice", "Mean displacement", "Max displacement", "Mean dx", "Mean dy"]
    assert rows[0]["Mean displacement"] == 0 and rows[1]["Max displacement"] > 2 and not s.results.rows
    # in place along T with the chain reference + downscale, undoable
    s2 = Session()
    img2 = s2.add_image(LumenImage(np.stack([BASE, d1, d2]).astype(np.uint8), name="tl", axes="TYX"))
    s2.run_op("registration.elastic_align", {"axis": "T", "reference": "Previous slice", "method": method, "downscale": 2})
    assert s2.active_id == img2.id
    assert all(mse(img2.data[k, 0, 0], BASE) < b / 3 for k, b in zip((1, 2), before))
    s2.undo(img2.id)
    assert np.array_equal(img2.data[1, 0, 0], d1.astype(np.uint8))
    with pytest.raises(ValueError, match="Z > 1"):
        s2.run_op("registration.elastic_align", {"axis": "Z"})


def test_elastic_pair_and_apply_deformation():
    d1 = deform(BASE, 4, 60)
    s = Session()
    a = s.add_image(LumenImage(BASE.astype(np.uint8), name="target"))
    b = s.add_image(LumenImage(d1.astype(np.uint8), name="moving"))
    s.select(b.id)
    out = s.run_op("registration.elastic_pair", {"target": a.id, "method": "ILK", "output_field": True})
    reg, field = (s.images[i] for i in out["new_images"])
    assert reg.shape == a.shape and mse(reg.data[0, 0, 0], BASE) < mse(d1, BASE) / 4
    assert field.nc == 2 and field.dtype == np.float32 and field.metadata.get("deformation_field")
    assert [c.name for c in field.display.channels] == ["dx", "dy"]
    row = s.results.rows[-1]
    assert row["Label"] == "moving → target" and row["Max displacement"] > row["Mean displacement"] > 1.0
    stored = s.settings["deformation"]
    assert stored["flow"].shape == (2, 200, 200) and stored["target"] == "target"
    # a third image warped with the stored field, and with the field image: identical results
    c = s.add_image(LumenImage(d1.astype(np.uint8), name="third"))
    out = s.run_op("registration.apply_deformation", {})
    third = s.images[out["new_images"][0]]
    assert third.name == "third (deformed)" and mse(third.data[0, 0, 0], BASE) < mse(d1, BASE) / 4
    s.select(c.id)
    out = s.run_op("registration.apply_deformation", {"field": field.id})
    assert np.array_equal(s.images[out["new_images"][0]].data, third.data)
    s.select(c.id)
    with pytest.raises(ValueError, match="different image"):
        s.run_op("registration.elastic_pair", {"target": c.id})
    fresh = Session()
    fresh.add_image(LumenImage(BASE.astype(np.uint8), name="x"))
    with pytest.raises(ValueError, match="No stored deformation"):
        fresh.run_op("registration.apply_deformation", {})


@pytest.mark.parametrize("method", ["TV-L1", "ILK"])
def test_apply_deformation_reproduces_elastic_pair_for_a_larger_moving_image(method):
    """elastic_pair estimates and warps on the max(target, moving) canvas and crops afterwards, but stores the field
    cropped to the target: re-applying it must warp on the same canvas (not crop the source first), so flow vectors
    that point beyond the target sample real pixels instead of zero padding – via the stored field and the field image."""
    s = Session()
    tgt = s.add_image(LumenImage(CAM[100:250, 100:250].astype(np.uint8), name="target"))  # 150 × 150
    mov = s.add_image(LumenImage(CAM[100:300, 90:290].astype(np.uint8), name="moving"))  # 200 × 200, the target's content 10 px to the right
    s.select(mov.id)
    out = s.run_op("registration.elastic_pair", {"target": tgt.id, "method": method, "output_field": True})
    reg, field = (s.images[i] for i in out["new_images"])
    assert reg.shape == (1, 1, 1, 150, 150) and s.settings["deformation"]["flow"].shape == (2, 150, 150) and s.settings["deformation"]["shape"] == [150, 150]
    assert float((reg.data[..., -10:] == 0).mean()) < 0.1  # the right edge holds content pulled in from beyond the target
    for params in ({}, {"field": field.id}):
        s.select(mov.id)
        out = s.run_op("registration.apply_deformation", params)
        applied = s.images[out["new_images"][0]]
        assert applied.shape == reg.shape and np.array_equal(applied.data, reg.data), params
    # the approximate inverse still works and keeps the field size
    s.select(mov.id)
    out = s.run_op("registration.apply_deformation", {"invert": True})
    assert s.images[out["new_images"][0]].shape == (1, 1, 1, 150, 150)


def test_apply_deformation_reproduces_block_matching_warp():
    """Block matching stores its dense field the same way (cropped to the target after warping on the union canvas)."""
    s = Session()
    tgt = s.add_image(LumenImage(CAM[100:250, 100:250].astype(np.uint8), name="target"))
    mov = s.add_image(LumenImage(CAM[100:300, 90:290].astype(np.uint8), name="moving"))
    s.select(mov.id)
    out = s.run_op("registration.block_matching_2d", {"target": tgt.id, "warp": True, "store": True, "arrows": False, "min_correlation": 0.3})
    warped = s.images[out["new_images"][0]]
    assert warped.shape == (1, 1, 1, 150, 150) and s.settings["deformation"]["method"] == "Block matching"
    s.select(mov.id)
    out = s.run_op("registration.apply_deformation", {})
    assert np.array_equal(s.images[out["new_images"][0]].data, warped.data)


# ---------------------------------------------------------------- folder registration
SHIFTS = [(0, 0), (3, -2), (5, 4), (-4, 6)]  # (dy, dx) of the scene in each file


def _write_shifted(folder):
    for k, (dy, dx) in enumerate(SHIFTS):
        a = CAM[150 + dy:350 + dy, 150 + dx:350 + dx].astype(np.uint8)
        write_image(LumenImage(a, name=f"img_{k:02d}"), str(folder / f"img_{k:02d}.tif"))
    (folder / "notes.txt").write_text("ignore me")


@pytest.mark.parametrize("mode,reference", [("Translation", "First image"), ("Affine", "Previous image"), ("Elastic", "First image")])
def test_align_folder(tmp_path, mode, reference):
    _write_shifted(tmp_path)
    out_dir = tmp_path / "out"
    s = Session()
    out = s.run_op("registration.align_folder", {"folder": str(tmp_path), "mode": mode, "reference": reference, "output_folder": str(out_dir)})
    assert "Registered 4 images" in out["message"] and not s.results.rows
    files = sorted(f.name for f in out_dir.iterdir())
    assert files == ["img_00.tif", "img_01.tif", "img_02.tif", "img_03.tif", "transforms.csv"]
    rows = table_rows(s, "Alignment")
    assert len(rows) == 4 and [r["Label"] for r in rows] == files[:4]
    if mode == "Elastic":
        assert s.tables["Alignment"]["columns"] == ["Label", "Image", "Mean displacement", "Max displacement", "Mean dx", "Mean dy"]
        for r, (dy, dx) in zip(rows, SHIFTS):  # same convention as align_stack: the shift applied to the image
            assert abs(r["Mean dx"] - dx) < 0.5 and abs(r["Mean dy"] - dy) < 0.5
    else:
        assert s.tables["Alignment"]["columns"] == ["Label", "Image", "dx", "dy", "Angle", "Scale"]
        for r, (dy, dx) in zip(rows, SHIFTS):
            assert abs(r["dx"] - dx) < 0.5 and abs(r["dy"] - dy) < 0.5 and abs(r["Angle"]) < 0.2
    with open(out_dir / "transforms.csv") as fh:
        csv_rows = list(csv.reader(fh))
    assert csv_rows[0] == s.tables["Alignment"]["columns"] and len(csv_rows) == 5
    imgs = [read_image(out_dir / f) for f in files[:4]]
    ref = imgs[0].data[0, 0, 0, 40:160, 40:160].astype(float)
    for im in imgs[1:]:
        assert np.abs(im.data[0, 0, 0, 40:160, 40:160].astype(float) - ref).mean() < 1.0
    # default output folder + pattern; the output folder itself is skipped on a re-run
    out = s.run_op("registration.align_folder", {"folder": str(tmp_path), "pattern": r"img_0[01]"})
    assert sorted(f.name for f in (tmp_path / "registered").iterdir()) == ["img_00.tif", "img_01.tif", "transforms.csv"]
    with pytest.raises(ValueError, match="No images"):
        s.run_op("registration.align_folder", {"folder": str(tmp_path), "pattern": "nomatch"})
    with pytest.raises(ValueError, match="folder"):
        s.run_op("registration.align_folder", {})


def test_align_folder_rejects_bad_pattern_and_same_output_folder(tmp_path):
    """A malformed pattern is a user error (ValueError → HTTP 400 with a message, not a bare 500); the '*.tif' glob habit
    is translated; an output folder equal to or containing the input folder is refused instead of 'No images'."""
    from fastapi.testclient import TestClient
    from lumen.server.app import create_app
    _write_shifted(tmp_path)
    before = {f.name: f.read_bytes() for f in tmp_path.iterdir() if f.is_file()}
    s = Session()
    with pytest.raises(ValueError, match="Invalid file pattern"):
        s.run_op("registration.align_folder", {"folder": str(tmp_path), "pattern": "(", "output_folder": str(tmp_path / "out")})
    c = TestClient(create_app(s), raise_server_exceptions=False)
    r = c.post("/api/ops/registration.align_folder/run", json={"params": {"folder": str(tmp_path), "pattern": "[", "output_folder": str(tmp_path / "out")}})
    assert r.status_code == 400 and "Invalid file pattern" in r.json()["error"]
    out = s.run_op("registration.align_folder", {"folder": str(tmp_path), "pattern": "*_0[01].tif", "output_folder": str(tmp_path / "out")})
    assert "Registered 2 images" in out["message"] and sorted(f.name for f in (tmp_path / "out").iterdir()) == ["img_00.tif", "img_01.tif", "transforms.csv"]
    for od in (str(tmp_path), str(tmp_path) + "/", str(tmp_path.parent), str(tmp_path / ".." / tmp_path.name)):
        with pytest.raises(ValueError, match="output folder"):
            s.run_op("registration.align_folder", {"folder": str(tmp_path), "output_folder": od})
    assert {f.name: f.read_bytes() for f in tmp_path.iterdir() if f.is_file()} == before  # the originals were never touched


# ---------------------------------------------------------------- phase correlation / correspondences
def test_phase_correlation():
    s = Session()
    a = s.add_image(LumenImage(CAM[50:250, 50:250].astype(np.uint8), name="A"))
    b = s.add_image(LumenImage(CAM[55:255, 47:247].astype(np.uint8), name="B"))
    s.select(b.id)
    out = s.run_op("registration.phase_correlation", {"target": a.id})
    r = s.results.rows[-1]
    assert r["Label"] == "B → A" and abs(r["dx"] + 3) < 0.3 and abs(r["dy"] - 5) < 0.3
    assert r["Peak"] > 0.5 and r["Correlation"] > 0.99 and "(-3.00, +5.00)" in out["message"]
    # large offsets / different sizes go through the zero-padded correlator
    c = s.add_image(LumenImage(CAM[300:400, 300:400].astype(np.uint8), name="C"))
    full = s.add_image(LumenImage(CAM.astype(np.uint8), name="full"))
    s.select(c.id)
    s.run_op("registration.phase_correlation", {"target": full.id})
    r = s.results.rows[-1]
    assert abs(r["dx"] - 300) < 0.5 and abs(r["dy"] - 300) < 0.5
    with pytest.raises(ValueError, match="target"):
        s.run_op("registration.phase_correlation", {})


@pytest.mark.parametrize("detector", ["SIFT", "ORB"])
def test_feature_correspondences(detector):
    base = CAM[100:356, 100:356]
    tf = AffineTransform(scale=1.1, rotation=np.deg2rad(8), translation=(12, -7))
    moved = warp(base, tf.inverse, order=1, preserve_range=True).astype(np.uint8)  # moved = tf applied to base
    s = Session()
    t = s.add_image(LumenImage(base.astype(np.uint8), name="base"))
    m = s.add_image(LumenImage(moved, name="moved"))
    s.select(m.id)
    out = s.run_op("registration.feature_correspondences", {"target": t.id, "detector": detector, "model": "Affine"})
    r = s.results.rows[-1]
    assert r["Detector"] == detector and r["N inliers"] >= 10 and r["N inliers"] <= r["N matches"]
    # moved → base is the inverse of tf: angle −8°, scale 1/1.1
    assert abs(r["Angle"] + 8) < 0.5 and abs(r["Scale"] - 1 / 1.1) < 0.02 and r["RMS residual"] < 1.5
    inv = np.linalg.inv(tf.params)
    assert abs(r["dx"] - inv[0, 2]) < 1.5 and abs(r["dy"] - inv[1, 2]) < 1.5
    assert "inliers" in out["message"]
    tab = s.tables["Correspondences"]
    assert tab["columns"] == ["Match", "X1", "Y1", "X2", "Y2", "Residual"] and len(tab["rows"]) == r["N inliers"]
    rois = s.roi_manager.rois
    assert len(rois) == 2 and {x["name"] for x in rois} == {"Matches"} and {x["image_id"] for x in rois} == {m.id, t.id}
    assert all(len(x["points"]) == r["N inliers"] for x in rois)
    # the inlier pairs map onto each other with the estimated matrix
    corr = s.settings["correspondences"]
    src, dst = np.array(corr["src"]) - 0.5, np.array(corr["dst"]) - 0.5
    M = np.array(corr["matrix"])
    proj = (np.c_[src, np.ones(len(src))] @ M.T)[:, :2]
    assert np.hypot(*(proj - dst).T).max() < 3.5
    # the stored correspondences drive the landmark transform
    out = s.run_op("image.transform_landmarks", {"model": "Affine"})
    new = s.images[out["new_images"][0]]
    inner = (slice(30, -30), slice(30, -30))
    assert np.mean((new.data[0, 0, 0][inner].astype(float) - base[inner]) ** 2) < np.mean((moved[inner].astype(float) - base[inner]) ** 2) / 4


# ---------------------------------------------------------------- interactive-style transforms
def test_transform_landmarks_recovers_affine():
    s = Session()
    img = s.add_image(LumenImage(BASE.astype(np.uint8), name="base"))
    A = AffineTransform(scale=(1.2, 0.9), rotation=np.deg2rad(10), shear=0.1, translation=(5, -3))
    src = np.array([[30, 40], [150, 50], [160, 170], [40, 150], [100, 100]], float)  # ROI coordinates
    dst = A(src - 0.5) + 0.5
    text = "; ".join(f"{x:.4f},{y:.4f}" for x, y in dst)
    out = s.run_op("image.transform_landmarks", {"model": "Affine", "target_points": text}, roi={"type": "point", "points": src.tolist()})
    new = s.images[out["new_images"][0]]
    expected = warp(BASE, A.inverse, order=1, preserve_range=True)
    assert np.abs(new.data[0, 0, 0].astype(float) - expected).max() <= 1.0
    r = s.results.rows[-1]
    assert r["Model"] == "Affine" and r["N landmarks"] == 5 and r["RMS residual"] < 1e-3
    assert abs(r["dx"] - 5) < 1e-3 and abs(r["dy"] + 3) < 1e-3 and abs(r["Angle"] - 10) < 1e-3
    assert np.allclose(np.array(s.settings["landmark_transform"]["matrix"]), A.params, atol=1e-4)
    # projective with 4 points, rigid with 2, output size, in place
    out = s.run_op("image.transform_landmarks", {"model": "Projective", "source_points": "10,10; 190,10; 190,190; 10,190", "target_points": "20,10; 180,20; 190,180; 10,190", "width": 100, "height": 80})
    assert s.images[out["new_images"][0]].shape == (1, 1, 1, 80, 100)
    s.select(img.id)
    out = s.run_op("image.transform_landmarks", {"model": "Rigid body", "source_points": "30,40; 150,50", "target_points": "40,30; 160,60", "create_new": False})
    assert not out["new_images"] and abs(s.results.rows[-1]["Scale"] - 1) < 1e-6
    s.undo(img.id)
    assert np.array_equal(img.data[0, 0, 0], BASE.astype(np.uint8))
    # ROI manager points as target
    s.roi_manager.add({"type": "point", "points": [[31, 41], [151, 51], [161, 171]], "name": "tgt"})
    out = s.run_op("image.transform_landmarks", {"model": "Affine", "source_points": "30,40; 150,50; 160,170", "target_roi": "tgt"})
    assert abs(s.results.rows[-1]["dx"] - 1) < 1e-6 and abs(s.results.rows[-1]["dy"] - 1) < 1e-6
    with pytest.raises(ValueError, match="at least 3"):
        s.run_op("image.transform_landmarks", {"model": "Affine", "source_points": "1,2; 3,4", "target_points": "1,2; 3,4"})
    with pytest.raises(ValueError, match="one to one"):
        s.run_op("image.transform_landmarks", {"model": "Affine", "source_points": "1,2", "target_points": "1,2; 3,4; 5,6"})
    with pytest.raises(ValueError, match="source landmarks"):
        s.run_op("image.transform_landmarks", {"model": "Affine", "target_points": "1,2; 3,4; 5,6"})


def test_perspective_shear_and_polar():
    s = Session()
    img = s.add_image(LumenImage(BASE.astype(np.uint8), name="base"))
    # an axis-aligned quadrilateral is a plain crop (edges = region borders)
    out = s.run_op("image.perspective", {"corners": "20,30; 120,30; 120,80; 20,80"})
    rect = s.images[out["new_images"][0]]
    assert rect.shape == (1, 1, 1, 50, 100) and np.array_equal(rect.data[0, 0, 0], BASE[30:80, 20:120].astype(np.uint8))
    # any corner order, explicit size, from a polygon selection
    out = s.run_op("image.perspective", {"width": 60, "height": 40}, roi={"type": "polygon", "points": [[170, 30], [20, 170], [180, 160], [30, 40]]})
    assert s.images[out["new_images"][0]].shape == (1, 1, 1, 40, 60)
    with pytest.raises(ValueError, match="4 corner"):
        s.run_op("image.perspective", {"corners": "1,1; 2,2; 3,3"})
    # shear: the centre row stays, a row 50 px below is shifted by 15 px
    s.select(img.id)
    s.run_op("image.shear", {"shear_x": 0.3, "shear_y": 0.0})
    assert img.shape == (1, 1, 1, 200, 200)
    cy = 99.5
    y = int(cy + 50.5)
    assert np.abs(img.data[0, 0, 0, y, 40:150].astype(float) - BASE[y, 25:135]).mean() < 2.0
    s.undo(img.id)
    s.run_op("image.shear", {"shear_x": 0.3, "enlarge": True})
    assert img.height == 200 and img.width > 250
    s.undo(img.id)
    # polar: shape, calibration, and a 30° rotation becomes a 30-row shift along the angle axis
    rot = rotate(BASE, 30, preserve_range=True)
    s.add_image(LumenImage(np.stack([BASE, rot]).astype(np.uint8), name="rot", axes="ZYX"))
    out = s.run_op("image.warp_polar", {"radius": 80})
    pol = s.images[out["new_images"][0]]
    assert pol.shape == (1, 2, 1, 360, 80) and pol.calibration.pixel_height == 1.0 and pol.metadata["polar"] == "linear"
    shift = phase_cross_correlation(pol.data[0, 0, 0].astype(float), pol.data[0, 1, 0].astype(float))[0]
    assert abs(abs(shift[0]) - 30) < 1 and abs(shift[1]) < 1
    out = s.run_op("image.warp_polar", {"scaling": "log", "angles": 180, "radii": 64}, roi={"type": "rect", "x": 40, "y": 60, "w": 20, "h": 20})
    pol = s.images[out["new_images"][0]]
    assert pol.shape == (1, 2, 1, 180, 64) and pol.metadata["center"] == [50.0, 70.0] and pol.metadata["polar"] == "log"


# ---------------------------------------------------------------- keyframe ROI interpolation
def test_interpolate_rois():
    s = Session()
    img = s.add_image(LumenImage(np.zeros((5, 64, 64), np.uint8), name="tl", axes="TYX"))
    s.roi_manager.add({"type": "rect", "x": 10, "y": 10, "w": 10, "h": 10, "name": "cell"}, image_id=img.id, position={"t": 0, "z": 0, "c": 0})
    s.roi_manager.add({"type": "rect", "x": 30, "y": 20, "w": 20, "h": 14, "name": "cell"}, image_id=img.id, position={"t": 4, "z": 0, "c": 0})
    out = s.run_op("stacks.interpolate_rois", {"axis": "T"})
    assert "Interpolated 3 ROIs" in out["message"] and out["rois_added"] == 3
    rois = s.roi_manager.rois
    assert len(rois) == 5
    mid = next(r for r in rois if r["position"]["t"] == 2)
    assert mid["type"] == "rect" and (mid["x"], mid["y"], mid["w"], mid["h"]) == (20, 15, 15, 12) and mid["name"] == "cell t3" and mid["image_id"] == img.id
    assert [r["position"]["t"] for r in rois[2:]] == [1, 2, 3]
    # polygons with different vertex counts (resampled), ovals, lines, points; the name filter picks a series
    s = Session()
    img = s.add_image(LumenImage(np.zeros((3, 64, 64), np.uint8), name="tl", axes="TYX"))
    add = lambda roi, t: s.roi_manager.add(roi, image_id=img.id, position={"t": t, "z": 0, "c": 0})  # noqa: E731
    add({"type": "polygon", "points": [[10, 10], [30, 10], [30, 30], [10, 30]], "name": "p"}, 0)
    add({"type": "polygon", "points": [[50, 50], [50, 30], [30, 30], [30, 50], [40, 52]], "name": "p"}, 2)  # reversed orientation, 5 vertices
    add({"type": "oval", "x": 0, "y": 0, "w": 10, "h": 10, "name": "o"}, 0)
    add({"type": "oval", "x": 20, "y": 20, "w": 30, "h": 10, "name": "o"}, 2)
    add({"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 10, "name": "l"}, 0)
    add({"type": "line", "x1": 10, "y1": 0, "x2": 30, "y2": 10, "name": "l"}, 2)
    add({"type": "point", "points": [[1, 1], [5, 5]], "name": "dots"}, 0)
    add({"type": "point", "points": [[3, 1], [5, 9]], "name": "dots"}, 2)
    with pytest.raises(ValueError, match="name filter"):
        s.run_op("stacks.interpolate_rois", {"axis": "T"})
    s.run_op("stacks.interpolate_rois", {"axis": "T", "name_filter": "p", "n_vertices": 8})
    poly = s.roi_manager.rois[-1]
    assert poly["type"] == "polygon" and len(poly["points"]) == 8 and poly["name"] == "p t2"
    pts = np.array(poly["points"])
    assert np.allclose(pts.mean(axis=0), [30, 30], atol=2)  # between the squares centred at (20, 20) and (40, 40)
    s.run_op("stacks.interpolate_rois", {"axis": "T", "name_filter": "o"})
    ov = s.roi_manager.rois[-1]
    assert ov["type"] == "oval" and (ov["x"], ov["y"], ov["w"], ov["h"]) == (10, 10, 20, 10)
    s.run_op("stacks.interpolate_rois", {"axis": "T", "name_filter": "l", "prefix": "mid "})
    ln = s.roi_manager.rois[-1]
    assert ln["type"] == "line" and (ln["x1"], ln["y1"], ln["x2"], ln["y2"]) == (5, 0, 20, 10) and ln["name"] == "mid t2"
    s.run_op("stacks.interpolate_rois", {"axis": "T", "name_filter": "dots"})
    assert s.roi_manager.rois[-1]["points"] == [[2, 1], [5, 7]]
    with pytest.raises(ValueError, match="at least two"):
        s.run_op("stacks.interpolate_rois", {"axis": "T", "name_filter": "zzz"})


# ---------------------------------------------------------------- dense optic flow
@pytest.mark.parametrize("method", ["TV-L1", "ILK"])
def test_optic_flow_recovers_shift(method):
    base = CAM[100:228, 100:228]
    frames = np.stack([np.roll(base, 3 * k, axis=1) for k in range(4)])
    s = Session()
    img = s.add_image(LumenImage(frames.astype(np.uint8), name="flow", axes="TYX"))
    out = s.run_op("process.optic_flow", {"method": method, "vector_overlay": True, "components": True, "grid": 32})
    mag, direction, comp = (s.images[i] for i in out["new_images"])
    inner = (slice(16, -16), slice(16, -16))
    assert mag.shape == (3, 1, 1, 128, 128) and mag.dtype == np.float32 and mag.metadata["optic_flow"] == "magnitude"
    assert abs(float(mag.data[0, 0, 0][inner].mean()) - 3.0) < 0.2 and abs(float(mag.data[2, 0, 0][inner].mean()) - 3.0) < 0.2
    assert direction.shape == mag.shape and abs(float(direction.data[0, 0, 0][inner].mean())) < 3.0  # motion along +x = 0°
    assert comp.nc == 2 and abs(float(comp.data[0, 0, 0][inner].mean()) - 3.0) < 0.2 and abs(float(comp.data[0, 0, 1][inner].mean())) < 0.2
    rows = table_rows(s, "Optic flow")
    assert s.tables["Optic flow"]["columns"] == ["Label", "Frame", "Mean dx", "Mean dy", "Mean magnitude", "Max magnitude", "Mean direction"]
    assert len(rows) == 3 and all(abs(r["Mean dx"] - 3) < 0.3 and abs(r["Mean dy"]) < 0.3 for r in rows) and not s.results.rows
    arrows = [o for o in img.overlay if o["type"] == "arrow"]
    assert len(arrows) == 3 * 16 and {a["position"]["t"] for a in arrows} == {0, 1, 2}
    a = next(x for x in arrows if x["position"]["t"] == 1 and x["x1"] == 48 and x["y1"] == 48)
    assert abs(a["x2"] - 51) < 0.5 and abs(a["y2"] - 48) < 0.5 and a["flow"]
    assert "3 frame pairs" in out["message"] and "48 arrows" in out["message"]
    # re-running replaces the flow arrows instead of piling them up
    s.select(img.id)
    s.run_op("process.optic_flow", {"method": method, "vector_overlay": True, "grid": 64, "magnitude": False, "direction": False})
    assert len([o for o in img.overlay if o["type"] == "arrow"]) == 3 * 4
    single = Session()
    single.add_image(LumenImage(base.astype(np.uint8), name="one"))
    with pytest.raises(ValueError, match="T > 1"):
        single.run_op("process.optic_flow", {})


def test_optic_flow_arrows_are_pinned_to_their_frame():
    """Each frame pair's arrows carry position.t = first frame of the pair – the per-plane overlay key shared with the
    other ops (morphology, block matching, spot ROIs) and honoured by burn_overlay: drawing frame k burns only frame k's
    arrows and the last frame gets none, instead of every frame's arrows on every frame."""
    from lumen.core.overlay_render import burn_overlay
    from lumen.core.render import render_rgb
    base = CAM[100:228, 100:228]
    frames = np.stack([np.roll(base, 3 * k, axis=1) for k in range(4)])
    s = Session()
    img = s.add_image(LumenImage(frames.astype(np.uint8), name="flow", axes="TYX"))
    s.run_op("process.optic_flow", {"method": "ILK", "vector_overlay": True, "grid": 32, "magnitude": False, "direction": False})
    arrows = [o for o in img.overlay if o["type"] == "arrow"]
    assert len(arrows) == 48 and all(o["position"] == {"t": o["position"]["t"], "z": 0, "c": 0} and 0 <= o["position"]["t"] < 3 for o in arrows)
    assert [o["position"] for o in img.summary()["overlay"] if o["type"] == "arrow"] == [o["position"] for o in arrows]  # the key reaches the client
    for t in range(4):
        rgb = render_rgb(img, t=t, z=0)
        burned = burn_overlay(rgb, img.overlay, img, t=t, z=0)
        own = burn_overlay(rgb, [o for o in arrows if o["position"]["t"] == t], img, t=t, z=0)
        assert np.array_equal(burned, own), t
        assert ((burned != rgb).any(axis=-1).sum() > 0) == (t < 3), t


# ---------------------------------------------------------------- colour clustering
def test_color_clustering_three_flat_colours():
    rgb = np.zeros((60, 90, 3), np.uint8)
    rgb[:, :30] = (200, 30, 30)
    rgb[:, 30:60] = (30, 200, 30)
    rgb[:, 60:] = (30, 30, 200)
    rng = np.random.default_rng(0)
    rgb = np.clip(rgb.astype(int) + rng.integers(-10, 10, rgb.shape), 0, 255).astype(np.uint8)
    for cs in ("RGB", "Lab", "HSB"):
        s = Session()
        s.add_image(LumenImage(rgb, name="flags", rgb=True))
        out = s.run_op("segment.color_clustering", {"n_clusters": 3, "color_space": cs, "color_image": True})
        lab = s.images[out["new_images"][0]]
        assert lab.name == "Clusters of flags" and lab.dtype == np.uint8 and lab.display.channels[0].lut == "Glasbey on dark"
        assert np.unique(lab.data).tolist() == [1, 2, 3]
        blocks = [np.unique(lab.data[0, 0, 0][:, a:a + 30]).tolist() for a in (0, 30, 60)]
        assert all(len(b) == 1 for b in blocks) and len({b[0] for b in blocks}) == 3
        rows = table_rows(s, "Clusters")
        assert len(rows) == 3 and all(r["Pixels"] == 1800 and abs(r["Fraction (%)"] - 100 / 3) < 1e-6 for r in rows)
        assert {r["Color"] for r in rows} == {"#c71e1d", "#1ec71e", "#1d1dc7"}
        cim = s.images[out["new_images"][1]]
        assert cim.rgb and cim.shape == (1, 1, 3, 60, 90) and np.abs(cim.data[0, 0].astype(int) - rgb.transpose(2, 0, 1)).max() <= 12
    # multi-channel grey image, ROI restriction and spatial features
    s = Session()
    g = np.zeros((2, 60, 90), np.float32)
    g[0, :, :45] = 100
    g[1, :30] = 50
    img = s.add_image(LumenImage(g, name="two", axes="CYX"))
    out = s.run_op("segment.color_clustering", {"n_clusters": 4})
    lab = s.images[out["new_images"][0]]
    assert np.unique(lab.data).tolist() == [1, 2, 3, 4] and s.tables["Clusters"]["columns"] == ["Cluster", "Pixels", "Fraction (%)", "Mean C1", "Mean C2"]
    s.select(img.id)
    out = s.run_op("segment.color_clustering", {"n_clusters": 2, "spatial_weight": 1.0}, roi={"type": "rect", "x": 0, "y": 0, "w": 45, "h": 60})
    lab = s.images[out["new_images"][0]]
    assert np.unique(lab.data).tolist() == [0, 1, 2] and not lab.data[0, 0, 0][:, 45:].any()
    with pytest.raises(ValueError, match="clusters requested"):
        s.run_op("segment.color_clustering", {"n_clusters": 5}, roi={"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2})
