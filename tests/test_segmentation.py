"""Plugins ▸ Segmentation: pixel classifier, random walker, active contours, superpixels, label utilities."""
import importlib.util

import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage import measure as skmeas, morphology as skm

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops._util import label_image
from lumen.core.ops.file_ops import _blobs
from lumen.core.registry import REGISTRY
from lumen.core.session import Session
from lumen.core.threshold import auto_threshold

SEGMENT_OPS = ["segment.train_classifier", "segment.apply_classifier", "segment.save_classifier", "segment.load_classifier",
               "segment.random_walker", "segment.active_contour", "segment.morph_snakes", "segment.superpixels", "segment.multi_otsu",
               "segment.cellpose", "segment.stardist", "segment.labels_to_rois", "segment.rois_to_labels", "segment.label_filter",
               "segment.expand_labels", "segment.shrink_labels", "segment.remove_border_labels", "segment.merge_touching_labels",
               "segment.seeded_watershed"]


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1)


def _otsu_mask(img):
    p = img.current_plane()
    lo, _ = auto_threshold(p, "Otsu")
    return p >= lo


def _add_class_rects(session, img, n=4):
    """A few 7x7 rectangles well inside / outside the blobs, named like a user would ("foreground 1", ...)."""
    otsu = _otsu_mask(img)
    rng = np.random.default_rng(1)
    for name, m in (("foreground", ndi.binary_erosion(otsu, skm.disk(5))), ("background", ndi.binary_erosion(~otsu, skm.disk(5)))):
        ys, xs = np.nonzero(m)
        k = 0
        for i in rng.permutation(len(ys)):
            y, x = ys[i], xs[i]
            if y >= 3 and x >= 3 and m[y - 3:y + 4, x - 3:x + 4].all():
                session.roi_manager.add({"type": "rect", "x": int(x - 3), "y": int(y - 3), "w": 7, "h": 7, "name": f"{name} {k + 1}"})
                k += 1
                if k == n:
                    break


def _blob_seed_points(img):
    """One point per blob (the most interior pixel) and a grid of background points."""
    otsu = _otsu_mask(img)
    lab = skmeas.label(otsu, connectivity=2)
    edt = ndi.distance_transform_edt(otsu)
    fg = []
    for rp in skmeas.regionprops(lab):
        c = rp.coords
        i = int(np.argmax(edt[c[:, 0], c[:, 1]]))
        fg.append([float(c[i, 1]) + 0.5, float(c[i, 0]) + 0.5])
    outside = ~ndi.binary_dilation(otsu, skm.disk(4))
    yy, xx = np.mgrid[8:img.height:16, 8:img.width:16]
    bg = [[float(x) + 0.5, float(y) + 0.5] for y, x in zip(yy.ravel(), xx.ravel()) if outside[y, x]]
    return fg, bg


def _new_images(session, out):
    return [session.images[i] for i in out["new_images"]]


# ------------------------------------------------------------------ registry
def test_segmentation_ops_registered():
    for op_id in SEGMENT_OPS:
        spec = REGISTRY.get(op_id)
        assert spec.category == "Plugins/Segmentation", op_id
        assert spec.to_dict()["menu"] == "Plugins/Segmentation"


# ------------------------------------------------------------ pixel classifier
def test_train_classifier_separates_blobs(session):
    img = session.active
    _add_class_rects(session, img)
    out = session.run_op("segment.train_classifier", {"n_trees": 20, "max_sigma": 4})
    row = session.results.rows[-1]
    assert row["Label"].startswith("Classifier") and row["Classes"] == 2 and "OOB accuracy" in row
    assert row["Pixels 'foreground'"] > 0 and row["Pixels 'background'"] > 0
    assert "classifier" in session.settings and session.settings["classifier"]["classes"] == ["background", "foreground"]
    probs, labels = _new_images(session, out)
    assert probs.name.startswith("Probabilities of") and probs.dtype == np.float32 and probs.nc == 2
    assert all(ch.lut == "Grays" and ch.min == 0.0 and ch.max == 1.0 for ch in probs.display.channels)
    assert np.allclose(probs.data.sum(axis=2), 1.0, atol=1e-4)
    assert labels.dtype == np.uint8 and labels.display.channels[0].lut == "Glasbey on dark"
    assert set(np.unique(labels.current_plane()).tolist()) <= {0, 1}
    assert _iou(labels.current_plane() == 1, _otsu_mask(img)) > 0.8


def test_apply_classifier_on_second_image(session):
    img = session.active
    _add_class_rects(session, img)
    session.run_op("segment.train_classifier", {"n_trees": 20, "max_sigma": 4, "apply_now": False})
    assert len(session.images) == 1
    img2 = session.add_image(LumenImage(_blobs(5), name="blobs2"))
    out = session.run_op("segment.apply_classifier", {"output": "Labels", "mask_class": "foreground", "probability_threshold": 0.5})
    labels, mask = _new_images(session, out)
    assert labels.name == "Classification of blobs2" and mask.name == "foreground mask of blobs2"
    assert _iou(labels.current_plane() == 1, _otsu_mask(img2)) > 0.8
    assert mask.is_binary and _iou(mask.current_plane() > 0, labels.current_plane() == 1) > 0.95
    with pytest.raises(ValueError, match="Unknown class"):
        session.run_op("segment.apply_classifier", {"mask_class": "nucleus"})


def test_apply_classifier_without_model_raises(session):
    with pytest.raises(ValueError, match="No trained pixel classifier"):
        session.run_op("segment.apply_classifier", {})
    with pytest.raises(ValueError, match="No training ROIs"):
        session.run_op("segment.train_classifier", {"n_trees": 5})


def test_save_load_classifier_roundtrip(session, tmp_path):
    img = session.active
    _add_class_rects(session, img)
    out = session.run_op("segment.train_classifier", {"n_trees": 10, "max_sigma": 2, "output": "Labels"})
    first = _new_images(session, out)[0].current_plane().copy()
    path = tmp_path / "blobs.joblib"
    session.run_op("segment.save_classifier", {"path": str(path)})
    assert path.exists()
    session.settings.pop("classifier")
    s2 = Session()
    s2.run_op("file.sample", {"name": "Blobs"})
    out = s2.run_op("segment.load_classifier", {"path": str(path)})
    assert "background, foreground" in out["message"]
    out = s2.run_op("segment.apply_classifier", {"output": "Labels"})
    assert np.array_equal(_new_images(s2, out)[0].current_plane(), first)
    with pytest.raises(ValueError, match="No such file"):
        s2.run_op("segment.load_classifier", {"path": str(tmp_path / "missing.joblib")})


# ---------------------------------------------------------------- random walker
def test_random_walker(session):
    img = session.active
    fg, bg = _blob_seed_points(img)
    session.roi_manager.add({"type": "point", "points": fg, "name": "foreground"})
    session.roi_manager.add({"type": "point", "points": bg, "name": "background"})
    out = session.run_op("segment.random_walker", {"beta": 2000, "output": "Labels and probabilities"})
    labels, probs = _new_images(session, out)
    assert labels.display.channels[0].lut == "Glasbey on dark" and set(np.unique(labels.current_plane()).tolist()) == {0, 1}
    assert probs.nc == 2 and probs.dtype == np.float32
    assert _iou(labels.current_plane() == 1, _otsu_mask(img)) > 0.8
    session.roi_manager.clear()
    with pytest.raises(ValueError, match="at least two classes"):
        session.run_op("segment.random_walker", {})


# ------------------------------------------------------------- multi-Otsu
def test_multi_otsu(session):
    img = session.active
    out = session.run_op("segment.multi_otsu", {"classes": 3})
    labels = _new_images(session, out)[0]
    assert set(np.unique(labels.current_plane()).tolist()) == {0, 1, 2}
    row = session.results.rows[-1]
    assert row["Label"] == img.name and 0 < row["Threshold 1"] < row["Threshold 2"] < 255
    assert _iou(labels.current_plane() == 2, _otsu_mask(img)) > 0.5


# ------------------------------------------------------------ superpixels
def test_superpixels(session):
    img = session.active
    out = session.run_op("segment.superpixels", {"method": "SLIC", "n_segments": 100, "output": "Both"})
    labels, boundaries = _new_images(session, out)
    n = int(labels.current_plane().max())
    assert 60 <= n <= 140 and labels.display.channels[0].lut == "Glasbey on dark"
    assert boundaries.is_binary and 0 < (boundaries.current_plane() > 0).mean() < 0.3
    session.select(img.id)
    out = session.run_op("segment.superpixels", {"method": "Felzenszwalb", "scale": 50, "min_size": 20})
    assert int(_new_images(session, out)[0].current_plane().max()) > 10
    session.select(img.id)
    out = session.run_op("segment.superpixels", {"method": "Compact watershed", "n_segments": 50})
    assert 30 <= int(_new_images(session, out)[0].current_plane().max()) <= 60


# ---------------------------------------------------------- active contours
def test_active_contours(session):
    img = session.active
    otsu = _otsu_mask(img)
    out = session.run_op("segment.active_contour", {"iterations": 100, "to_selection": True})
    mask = _new_images(session, out)[0]
    assert mask.is_binary and _iou(mask.current_plane() > 0, otsu) > 0.8
    assert out["roi"]["type"] == "mask"
    session.select(img.id)
    out = session.run_op("segment.morph_snakes", {"iterations": 30, "init": "checkerboard", "to_selection": False})
    assert _iou(_new_images(session, out)[0].current_plane() > 0, otsu) > 0.8
    session.select(img.id)
    with pytest.raises(ValueError, match="selection"):
        session.run_op("segment.morph_snakes", {"iterations": 5, "init": "selection"})
    out = session.run_op("segment.morph_snakes", {"method": "Geodesic active contour", "iterations": 20, "init": "selection", "balloon": 1},
                         roi={"type": "rect", "x": 100, "y": 100, "w": 30, "h": 30})
    assert out["roi"]["type"] == "mask" and _new_images(session, out)[0].is_binary


# ------------------------------------------------------------ label utilities
def _label_image_of_blobs(session):
    img = session.active
    lab = skmeas.label(_otsu_mask(img), connectivity=2)
    return session.add_image(label_image(lab, "labels", like=img)), lab


def test_expand_and_shrink_labels(session):
    li, lab = _label_image_of_blobs(session)
    n, area = int(lab.max()), int((lab > 0).sum())
    session.run_op("segment.expand_labels", {"distance": 2})
    expanded = li.current_plane()
    assert li.dtype == np.uint8 and int(expanded.max()) == n and (expanded > 0).sum() > area
    assert len(np.unique(expanded)) == n + 1
    session.run_op("segment.shrink_labels", {"distance": 2})
    shrunk = li.current_plane()
    assert area * 0.9 <= (shrunk > 0).sum() <= (expanded > 0).sum()
    assert np.all(shrunk[shrunk > 0] == expanded[shrunk > 0])


def test_labels_to_rois_roundtrip(session):
    li, lab = _label_image_of_blobs(session)
    n = int(lab.max())
    out = session.run_op("segment.labels_to_rois", {})
    rois = session.roi_manager.to_list()
    assert len(rois) == n and rois[0]["name"] == "L1" and rois[-1]["name"] == f"L{n}"
    assert all(r["position"] == {"t": 0, "z": 0, "c": 0} and r["image_id"] == li.id for r in rois)
    out = session.run_op("segment.rois_to_labels", {})
    back = _new_images(session, out)[0]
    assert int(back.current_plane().max()) == n
    assert _iou(back.current_plane() > 0, lab > 0) > 0.95
    session.select(li.id)
    out = session.run_op("segment.labels_to_rois", {"min_size": 100, "clear_manager": True})
    assert 0 < len(session.roi_manager.rois) < n


def test_label_filter_border_and_merge(session):
    img = session.active
    li, lab = _label_image_of_blobs(session)
    n = int(lab.max())
    out = session.run_op("segment.label_filter", {"min_area": 150, "intensity_image": img.id, "min_mean": 100})
    filtered = _new_images(session, out)[0].current_plane()
    kept = int(filtered.max())
    assert 0 < kept < n and len(np.unique(filtered)) == kept + 1
    assert all(rp.area >= 150 for rp in skmeas.regionprops(filtered.astype(np.int32)))
    session.select(li.id)
    session.run_op("segment.remove_border_labels", {})
    inner = li.current_plane()
    assert 0 < len(np.unique(inner)) - 1 < n
    assert inner[0, :].max() == 0 and inner[:, -1].max() == 0
    session.run_op("segment.expand_labels", {"distance": 6})
    session.run_op("segment.merge_touching_labels", {})
    merged = li.current_plane()
    assert 0 < int(merged.max()) < len(np.unique(inner)) - 1
    assert skmeas.label(merged > 0, connectivity=2).max() == int(merged.max())


def test_seeded_watershed(session):
    img = session.active
    fg, _ = _blob_seed_points(img)
    session.run_op("threshold.compute", {"method": "Otsu"})
    with pytest.raises(ValueError, match="point selection"):
        session.run_op("segment.seeded_watershed", {})
    out = session.run_op("segment.seeded_watershed", {}, roi={"type": "point", "points": fg})
    labels = _new_images(session, out)[0]
    assert int(labels.current_plane().max()) == len(fg)
    assert len(session.roi_manager.rois) == len(fg)
    assert _iou(labels.current_plane() > 0, _otsu_mask(img)) > 0.95
    assert session.results.rows[-1]["Count"] == len(fg)


# ------------------------------------------------------- optional deep learning
@pytest.mark.parametrize("op_id,package", [("segment.cellpose", "cellpose"), ("segment.stardist", "stardist")])
def test_deep_learning_wrappers_need_their_package(session, op_id, package):
    if importlib.util.find_spec(package) is not None:
        pytest.skip(f"{package} is installed")
    with pytest.raises(ValueError, match=f"pip install {package}"):
        session.run_op(op_id, {})


# --------------------------------------------------------------- regressions
def _rw_three_classes(session, img, extra=None):
    """Seeds for 'background' and 'nucleus' only (the middle class 'cell' has none unless `extra` adds it)."""
    fg, bg = _blob_seed_points(img)
    session.roi_manager.add({"type": "point", "points": fg, "name": "nucleus"})
    session.roi_manager.add({"type": "point", "points": bg, "name": "background"})
    if extra:
        session.roi_manager.add(extra)


def test_random_walker_labels_keep_class_values_when_a_class_has_no_seeds(session):
    """skimage renumbers the seeds to 1..n; the 'Labels' output must map them back to the class indices like the
    probabilities output does (label 2 == 'nucleus' per metadata['classes'])."""
    img = session.active
    _rw_three_classes(session, img)
    params = {"classes": "background,cell,nucleus", "beta": 2000}
    out = session.run_op("segment.random_walker", {**params, "output": "Labels"})
    lab_only = _new_images(session, out)[0]
    session.select(img.id)
    out = session.run_op("segment.random_walker", {**params, "output": "Labels and probabilities"})
    lab_both, probs = _new_images(session, out)
    assert set(np.unique(lab_only.current_plane()).tolist()) == {0, 2}
    assert np.array_equal(lab_only.current_plane(), lab_both.current_plane())
    assert lab_only.metadata["classes"] == "background, cell, nucleus"
    assert [c.name for c in probs.display.channels] == ["background", "nucleus"]
    assert _iou(lab_only.current_plane() == 2, _otsu_mask(img)) > 0.8
    assert out["message"] == "Random walker: 2 classes"


def test_random_walker_markers_image_keeps_sparse_label_values(session):
    img = session.active
    fg, bg = _blob_seed_points(img)
    mk = np.zeros((img.height, img.width), np.uint16)
    for x, y in bg:
        mk[int(y), int(x)] = 1
    for i, (x, y) in enumerate(fg):
        mk[int(y), int(x)] = 3 if i % 2 == 0 else 7
    m = session.add_image(LumenImage(mk, name="markers"))
    results = []
    for output in ("Labels", "Labels and probabilities"):
        session.select(img.id)
        out = session.run_op("segment.random_walker", {"markers_image": m.id, "beta": 2000, "output": output})
        results.append(_new_images(session, out)[0].current_plane())
    assert set(np.unique(results[0]).tolist()) == {1, 3, 7}
    assert np.array_equal(results[0], results[1])


def test_random_walker_selection_dropping_seeds(session):
    img = session.active
    fg, bg = _blob_seed_points(img)
    session.roi_manager.add({"type": "point", "points": fg, "name": "foreground"})
    session.roi_manager.add({"type": "point", "points": [[2.5, 2.5], [2.5, 60.5]], "name": "background"})
    inner = {"type": "rect", "x": 10, "y": 10, "w": img.width - 20, "h": img.height - 20}
    # every background seed lies outside the selection -> a clear error instead of an all-zero result
    with pytest.raises(ValueError, match="inside the selection.*only 'foreground' has seeds"):
        session.run_op("segment.random_walker", {"beta": 2000}, roi=inner)
    # three classes, only the 'cell' seed outside the selection -> nucleus stays label 2 in both output modes
    session.roi_manager.clear()
    _rw_three_classes(session, img, extra={"type": "point", "points": [[5.5, 5.5]], "name": "cell"})
    params = {"classes": "background,cell,nucleus", "beta": 2000}
    planes = []
    for output in ("Labels", "Labels and probabilities"):
        session.select(img.id)
        out = session.run_op("segment.random_walker", {**params, "output": output}, roi=inner)
        ims = _new_images(session, out)
        planes.append(ims[0].current_plane())
        assert out["message"] == "Random walker: 2 classes"
    assert set(np.unique(planes[0]).tolist()) == {0, 2} and np.array_equal(planes[0], planes[1])
    assert [c.name for c in ims[1].display.channels] == ["background", "nucleus"] and ims[1].nc == 2
    assert planes[0][:10, :].max() == 0 and planes[0][:, :10].max() == 0  # outside the selection = 0
    assert _iou(planes[0] == 2, _otsu_mask(img) & (planes[0] >= 0)) > 0.6


def test_random_walker_gap_free_markers_raise(session):
    """A markers image without unlabelled pixels made skimage return an (H, W, n) array -> IndexError."""
    img = session.active
    out = session.run_op("segment.superpixels", {"method": "SLIC", "n_segments": 20})
    sp = _new_images(session, out)[0]
    assert not (sp.current_plane() == 0).any()
    for output in ("Labels", "Labels and probabilities"):
        session.select(img.id)
        with pytest.raises(ValueError, match="nothing to segment"):
            session.run_op("segment.random_walker", {"markers_image": sp.id, "output": output})


def test_merge_touching_labels_refuses_to_clip_labels(session):
    mask = np.zeros((100, 100), np.uint8)
    for i in range(300):  # 300 isolated 2x2 squares
        y, x = 5 * (i // 20), 5 * (i % 20)
        mask[y:y + 2, x:x + 2] = 255
    li = session.add_image(LumenImage(mask, name="mask8"))
    before = li.current_plane().copy()
    with pytest.raises(ValueError, match="more than a 8-bit image can hold.*16-bit"):
        session.run_op("segment.merge_touching_labels", {})
    assert np.array_equal(li.current_plane(), before)
    session.run_op("type.16bit", {})
    session.run_op("segment.merge_touching_labels", {})
    merged = li.current_plane()
    assert li.dtype == np.uint16 and int(merged.max()) == 300 and len(np.unique(merged)) == 301


def test_apply_classifier_across_bit_depths(session):
    """A classifier trained on 8-bit must give the same result on 32-bit / 16-bit copies with the same values."""
    from lumen.core.ops.segmentation import _plane_features
    img = session.active
    _add_class_rects(session, img)
    out = session.run_op("segment.train_classifier", {"n_trees": 20, "max_sigma": 4, "output": "Labels"})
    ref = _new_images(session, out)[0].current_plane()
    model = session.settings["classifier"]
    assert model["version"] >= 2 and model["trained_dtype"] == "uint8" and model["settings"]["intensity_scale"] == 255.0
    for conv in ("type.32bit", "type.16bit"):
        copy = session.add_image(LumenImage(img.data.copy(), name=f"copy {conv}"))
        session.run_op(conv, {})
        assert copy.dtype != np.uint8 and np.array_equal(copy.current_plane(), img.current_plane())
        out = session.run_op("segment.apply_classifier", {"output": "Labels"})
        lab = _new_images(session, out)[0].current_plane()
        assert np.array_equal(lab, ref), conv
        assert "0.0%" not in out["message"] and "trained on a uint8 image" in out["message"]
    # classifiers saved before version 2 (no intensity_scale) keep the per-dtype scaling they were fitted with
    legacy = {"features": ["Gaussian"], "sigmas": [1.0]}
    plane = img.current_plane()
    assert np.allclose(_plane_features(plane, legacy)[0][..., 0], plane / 255.0)
    assert np.allclose(_plane_features(plane.astype(np.uint16), legacy)[0][..., 0], plane / 65535.0)
    assert np.allclose(_plane_features(plane, {**legacy, "intensity_scale": 255.0})[0][..., 0], plane / 255.0)
    assert np.allclose(_plane_features(plane.astype(np.float32), {**legacy, "intensity_scale": 255.0})[0][..., 0], plane / 255.0)


def test_superpixels_more_than_65535_segments_do_not_wrap():
    from lumen.core.ops.segmentation import _any_count_label_image
    s = Session()
    rng = np.random.default_rng(0)
    noise = s.add_image(LumenImage(rng.integers(0, 256, (300, 300), dtype=np.uint8), name="noise"))
    out = s.run_op("segment.superpixels", {"method": "SLIC", "n_segments": 100000})
    lab = _new_images(s, out)[0]
    n = int(out["message"].split()[0])
    assert n > 65535
    assert lab.dtype == np.float32 and lab.display.channels[0].lut == "Glasbey on dark" and lab.display.channels[0].max == float(n)
    assert int(lab.current_plane().max()) == n and len(np.unique(lab.current_plane())) == n
    small = _any_count_label_image(np.arange(300, dtype=np.int32).reshape(10, 30), "small", like=noise)
    assert small.dtype == np.uint16 and int(small.current_plane().max()) == 299


def test_cellpose_channel_out_of_range_is_a_clear_error(session, monkeypatch):
    import sys
    import types
    calls = []

    class _Net:
        def __init__(self, **kw):
            pass

        def eval(self, x, channels=None, **kw):
            calls.append((x.shape, channels, float(x[..., 0].mean()), float(x[..., 1].mean())))
            return np.zeros(x.shape[:2], np.int32), None, None, None

    cp, models = types.ModuleType("cellpose"), types.ModuleType("cellpose.models")
    models.Cellpose = _Net
    cp.models = models
    monkeypatch.setitem(sys.modules, "cellpose", cp)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    session.run_op("file.sample", {"name": "Cells (3-D, 2 channels)"})
    img = session.active
    assert img.nc == 2
    img.data[:, :, 0] = 10
    img.data[:, :, 1] = 200
    with pytest.raises(ValueError, match="Cytoplasm channel 3 does not exist \\(image has 2\\)"):
        session.run_op("segment.cellpose", {"cyto_channel": 3})
    with pytest.raises(ValueError, match="Nuclei channel 3 does not exist"):
        session.run_op("segment.cellpose", {"cyto_channel": 1, "nuclei_channel": 3})
    assert calls == []
    session.run_op("segment.cellpose", {"cyto_channel": 1, "nuclei_channel": 2, "add_to_manager": False})
    assert calls[-1][1:] == ([1, 2], 10.0, 200.0)


def test_label_image_and_count_masks_above_65535_labels_do_not_wrap():
    from skimage import measure as skmeas
    from lumen.core.ops._util import label_image
    m = np.zeros((512, 512), bool)
    m[::2, ::2] = True  # 256 × 256 isolated pixels = 65536 objects
    lab = skmeas.label(m, connectivity=2)
    n = int(lab.max())
    assert n == 65536
    li = label_image(lab, "labels")
    assert li.dtype == np.float32 and int(li.current_plane().max()) == n and li.display.channels[0].max == float(n)
    assert label_image(np.arange(300, dtype=np.int64).reshape(10, 30), "small").dtype == np.uint16
    assert label_image(np.arange(200, dtype=np.int64).reshape(10, 20), "tiny").dtype == np.uint8
    s = Session()
    s.add_image(LumenImage(m.astype(np.uint8) * 255, name="dots"))
    out = s.run_op("analyze.particles", {"show": "Count Masks", "display_results": False, "summarize": False, "add_to_manager": False})
    assert out["message"].startswith("65536 particles")
    cm = s.images[out["new_images"][0]]
    plane = cm.current_plane()
    assert cm.dtype == np.float32 and int(plane.max()) == 65536 and len(np.unique(plane)) == 65537
