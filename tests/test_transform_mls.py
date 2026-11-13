"""Moving least squares deformation and Landmark Correspondences (lumen/core/ops/transform_mls.py)."""
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.data import camera
from skimage.transform import AffineTransform, SimilarityTransform, EuclideanTransform

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops.transform_mls import mls_map
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

CAM = camera().astype(np.float32)
BASE = CAM[100:260, 120:300]  # 160 × 180
SRC = np.array([[20.0, 25.0], [150.0, 30.0], [40.0, 130.0], [160.0, 140.0]])


def test_ops_registered():
    for op_id in ("image.transform_mls", "image.transform_landmark_correspondences"):
        spec = REGISTRY.get(op_id)
        assert spec.to_dict()["menu"] == "Image/Transform" and spec.params and spec.description


@pytest.mark.parametrize("model, tf", [("Affine", AffineTransform(scale=(1.05, 0.95), rotation=0.1, shear=0.05, translation=(3, -2))),
                                       ("Similarity", SimilarityTransform(scale=1.1, rotation=-0.2, translation=(-4, 5))),
                                       ("Rigid", EuclideanTransform(rotation=0.3, translation=(2, 7)))])
def test_mls_reproduces_a_global_transform_from_four_landmarks(model, tf):
    """When the landmark pairs follow one global transform of the model's class, MLS reproduces it everywhere, exactly."""
    dst = tf(SRC)
    yy, xx = np.mgrid[0:60, 0:70].astype(float) * 2.5
    fx, fy = mls_map(SRC, dst, xx, yy, alpha=1.0, model=model)
    truth = tf(np.c_[xx.ravel(), yy.ravel()])
    assert np.abs(fx.ravel() - truth[:, 0]).max() < 1e-6 and np.abs(fy.ravel() - truth[:, 1]).max() < 1e-6
    # control points are interpolated exactly, including evaluation right on them
    px, py = mls_map(SRC, dst, SRC[:, 0], SRC[:, 1], model=model)
    assert np.allclose(np.c_[px, py], dst, atol=1e-9)
    # a single pair is a translation; mismatched counts are refused
    fx1, fy1 = mls_map(SRC[:1], dst[:1], xx, yy, model=model)
    assert np.allclose(fx1 - xx, dst[0, 0] - SRC[0, 0]) and np.allclose(fy1 - yy, dst[0, 1] - SRC[0, 1])
    with pytest.raises(ValueError, match="same number"):
        mls_map(SRC, dst[:3], xx, yy)


def test_transform_mls_op_matches_affine_landmark_transform_and_interpolates_landmarks():
    tf = AffineTransform(scale=(1.04, 0.97), rotation=0.08, translation=(4, -3))
    dst = tf(SRC - 0.5) + 0.5
    s = Session()
    img = s.add_image(LumenImage(BASE.astype(np.uint8), name="cam"))
    pts = "; ".join(f"{x},{y}" for x, y in SRC)
    tpts = "; ".join(f"{x},{y}" for x, y in dst)
    out = s.run_op("image.transform_mls", {"model": "Affine", "source_points": pts, "target_points": tpts})
    mls = s.images[out["new_images"][0]]
    assert mls.name == "cam (mls)" and mls.shape == img.shape and mls.dtype == np.uint8
    row = s.results.rows[-1]
    assert row["N landmarks"] == 4 and row["Model"] == "Moving least squares (affine)" and row["Mean displacement"] > 1
    s.select(img.id)
    out2 = s.run_op("image.transform_landmarks", {"model": "Affine", "source_points": pts, "target_points": tpts})
    lin = s.images[out2["new_images"][0]]
    diff = np.abs(mls.data.astype(float) - lin.data.astype(float))
    # the same affine warp up to rounding; skimage.warp (transform_landmarks) and map_coordinates (MLS) only differ for
    # output pixels whose source position lies within half a pixel outside the image (edge value vs 0)
    assert float((diff > 1).mean()) < 0.01 and np.median(diff) == 0
    # a genuinely local deformation: bright dots at the source landmarks end up at the target landmarks
    dots = np.zeros((160, 180), np.float32)
    src5 = np.array([[30, 30], [150, 30], [30, 130], [150, 130], [90, 80]], float)
    dst5 = src5.copy()
    dst5[4] += (12, -9)
    for x, y in src5:
        dots[int(y), int(x)] = 255
    dots = ndi.gaussian_filter(dots, 1.5) * 10
    s2 = Session()
    d = s2.add_image(LumenImage(dots, name="dots"))
    s2.roi_manager.add({"type": "point", "points": (src5 + 0.5).tolist(), "name": "src"}, image_id=d.id)
    s2.roi_manager.add({"type": "point", "points": (dst5 + 0.5).tolist(), "name": "dst"}, image_id=d.id)
    for model in ("Affine", "Similarity", "Rigid"):
        s2.select(d.id)
        out = s2.run_op("image.transform_mls", {"model": model, "source_roi": "src", "target_roi": "dst", "alpha": 1.5, "create_new": True})
        w = s2.images[out["new_images"][0]].data[0, 0, 0]
        for x, y in dst5:
            win = w[int(y) - 3:int(y) + 4, int(x) - 3:int(x) + 4]
            py, px = np.unravel_index(int(np.argmax(win)), win.shape)
            assert abs(py - 3) <= 1 and abs(px - 3) <= 1, (model, x, y)
    # in place (undoable) with a point selection as the source and a size change
    s2.select(d.id)
    s2.run_op("image.transform_mls", {"model": "Affine", "target_points": "; ".join(f"{x},{y}" for x, y in dst5 + 0.5), "create_new": False, "width": 200, "height": 100},
              roi={"type": "point", "points": (src5 + 0.5).tolist()})
    assert d.shape == (1, 1, 1, 100, 200)
    s2.undo(d.id)
    assert d.shape == (1, 1, 1, 160, 180)
    with pytest.raises(ValueError, match="correspond one to one"):  # the image's 10 attached points vs 2 targets
        s2.run_op("image.transform_mls", {"target_points": "1,1; 2,2"})
    plain = s2.add_image(LumenImage(dots, name="plain"))
    with pytest.raises(ValueError, match="source landmarks"):
        s2.run_op("image.transform_mls", {"target_points": "1,1; 2,2"}, image_id=plain.id)
    with pytest.raises(ValueError, match="correspond one to one"):
        s2.run_op("image.transform_mls", {"source_points": "1,1; 2,2; 3,3", "target_points": "1,1; 2,2"})


def test_landmark_correspondences_between_two_images_and_table_sources():
    tf = SimilarityTransform(scale=1.0, rotation=0.0, translation=(10, 6))
    src_pts = SRC
    dst_pts = tf(SRC)
    s = Session()
    source = s.add_image(LumenImage(BASE.astype(np.uint8), name="source"))
    target = s.add_image(LumenImage(np.zeros((200, 220), np.uint8), name="target"))
    s.roi_manager.add({"type": "point", "points": src_pts.tolist(), "name": "src"}, image_id=source.id)
    s.roi_manager.add({"type": "point", "points": dst_pts.tolist(), "name": "dst"}, image_id=target.id)
    s.select(source.id)
    out = s.run_op("image.transform_landmark_correspondences", {"target": target.id, "model": "Rigid body"})
    res = s.images[out["new_images"][0]]
    assert res.name == "source on target" and res.shape == target.shape
    row = s.results.rows[-1]
    assert row["Model"] == "Rigid body" and row["N landmarks"] == 4 and abs(row["dx"] - 10) < 1e-6 and abs(row["dy"] - 6) < 1e-6 and row["RMS residual"] < 1e-6
    assert np.array_equal(res.data[0, 0, 0, 6:166, 10:190], source.data[0, 0, 0])
    assert s.settings["landmark_transform"]["target"] == target.id
    # moving least squares between two images (target size), no Results row
    s.select(source.id)
    out = s.run_op("image.transform_landmark_correspondences", {"target": target.id, "model": "Moving least squares (similarity)", "show_matrix": False})
    res2 = s.images[out["new_images"][0]]
    assert res2.shape == target.shape and "results_added" not in out
    assert np.abs(res2.data[0, 0, 0, 6:166, 10:190].astype(float) - source.data[0, 0, 0]).max() <= 1
    # landmarks from a 'Correspondences' table (Extract SIFT Correspondences) when no points are given
    s3 = Session()
    a = s3.add_image(LumenImage(BASE.astype(np.uint8), name="a"))
    s3.tables["Correspondences"] = {"name": "Correspondences", "columns": ["Match", "X1", "Y1", "X2", "Y2", "Residual"],
                                   "rows": [[i + 1, float(p[0]), float(p[1]), float(q[0]), float(q[1]), 0.0] for i, (p, q) in enumerate(zip(src_pts, dst_pts))]}
    out = s3.run_op("image.transform_mls", {"model": "Rigid"})
    assert s3.results.rows[-1]["N landmarks"] == 4 and s3.images[out["new_images"][0]].shape == a.shape
    s3.select(a.id)
    with pytest.raises(ValueError, match="different image"):
        s3.run_op("image.transform_landmark_correspondences", {"target": a.id})
    with pytest.raises(ValueError, match="Choose a target"):
        s3.run_op("image.transform_landmark_correspondences", {})


def test_mls_affine_refuses_two_or_collinear_landmarks_and_weights_are_scale_free():
    """Fiji's AffineModel2D needs >= 3 non-collinear matches: Lumen raises instead of returning a garbage warp, while
    Similarity / Rigid are exact with two landmarks; the per-pixel weights are normalised so large alpha on a large
    grid cannot push the moment sums below an absolute tolerance (the old 1e-12 clamp collapsed valid landmark sets)."""
    yy, xx = np.mgrid[0:20, 0:20].astype(float)
    p2 = np.array([[5.0, 5.0], [15.0, 12.0]])
    with pytest.raises(ValueError, match="non-collinear"):
        mls_map(p2, p2 + [2, 1], xx, yy, model="Affine")
    p3 = np.array([[2.0, 2.0], [8.0, 8.0], [14.0, 14.0]])
    with pytest.raises(ValueError, match="non-collinear"):
        mls_map(p3, p3 + 1, xx, yy, model="Affine")
    with pytest.raises(ValueError, match="ill-defined"):
        mls_map(np.array([[3.0, 3.0], [3.0, 3.0]]), np.array([[4.0, 3.0], [3.0, 3.0]]), xx, yy, model="Rigid")
    for model in ("Similarity", "Rigid"):
        fx, fy = mls_map(p2, p2 + [2, 1], xx, yy, model=model)
        assert np.abs(fx - xx - 2).max() < 1e-9 and np.abs(fy - yy - 1).max() < 1e-9
    # three valid landmarks following a pure translation on a 300 px grid: exact for every alpha the Param allows
    yy, xx = np.mgrid[0:300, 0:300].astype(float)
    p = np.array([[50.0, 60.0], [250.0, 40.0], [120.0, 260.0]])
    for model in ("Affine", "Similarity", "Rigid"):
        for alpha in (1, 2, 3, 5):
            fx, fy = mls_map(p, p + 1, xx, yy, alpha=alpha, model=model)
            assert np.abs(fx - xx - 1).max() < 1e-6 and np.abs(fy - yy - 1).max() < 1e-6, (model, alpha)
    # op level: both MLS commands refuse, the sibling models reproduce the translation
    s = Session()
    img = s.add_image(LumenImage(BASE.astype(np.uint8), name="cam"))
    tgt = s.add_image(LumenImage(np.zeros((160, 180), np.uint8), name="tgt"))
    two = {"source_points": "20,30; 100,40", "target_points": "25,27; 105,37"}
    s.select(img.id)
    with pytest.raises(ValueError, match="non-collinear"):
        s.run_op("image.transform_mls", {"model": "Affine", **two})
    s.select(img.id)
    with pytest.raises(ValueError, match="non-collinear"):
        s.run_op("image.transform_landmark_correspondences", {"target": tgt.id, "model": "Moving least squares (affine)", **two})
    expect = ndi.shift(BASE, (-3, 5), order=1, mode="constant", cval=0.0)
    for model in ("Similarity", "Rigid"):
        s.select(img.id)
        out = s.run_op("image.transform_mls", {"model": model, **two})
        got = s.images[out["new_images"][0]].data[0, 0, 0].astype(float)
        assert np.abs(got[5:-5, 10:-10] - np.round(expect)[5:-5, 10:-10]).max() <= 1
