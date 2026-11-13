"""Trainable Weka Segmentation parity: classifier choice, the extended feature bank, model files, apply targets
(other image, folder), Overlay output + toggle, classifier info tables and the `class:<name>` trace convention."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy import ndimage as ndi
from skimage import morphology as skm

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops import classifier_ops as clf
from lumen.core.ops import segmentation as seg
from lumen.core.ops.file_ops import _blobs
from lumen.core.registry import REGISTRY
from lumen.core.session import Session
from lumen.core.threshold import auto_threshold
from lumen.server.app import create_app

FAST = {"n_trees": 10, "max_sigma": 2.0, "max_pixels_per_class": 1500}


# ------------------------------------------------------------------ helpers
def _otsu_mask(img):
    p = img.current_plane()
    lo, _ = auto_threshold(p, "Otsu")
    return p >= lo


def _iou(a, b):
    a, b = a.astype(bool), b.astype(bool)
    return (a & b).sum() / max((a | b).sum(), 1)


def _add_traces(session, img, n=8, naming="class", erode=1):
    """7×7 rectangles inside / outside the blobs (``erode`` px away from the Otsu edge, so the classes cover the edge
    zone too), named 'class:<name>' (the panel's convention) or '<name> i'."""
    otsu = _otsu_mask(img)
    rng = np.random.default_rng(1)
    for name, m in (("foreground", ndi.binary_erosion(otsu, skm.disk(erode))), ("background", ndi.binary_erosion(~otsu, skm.disk(erode)))):
        ys, xs = np.nonzero(m)
        k = 0
        for i in rng.permutation(len(ys)):
            y, x = ys[i], xs[i]
            if y >= 3 and x >= 3 and m[y - 3:y + 4, x - 3:x + 4].all():
                roi_name = f"class:{name}" if naming == "class" else f"{name} {k + 1}"
                session.roi_manager.add({"type": "rect", "x": int(x - 3), "y": int(y - 3), "w": 7, "h": 7, "name": roi_name, "color": "#ff0000"},
                                        image_id=img.id, position={"t": 0, "z": 0, "c": 0})
                k += 1
                if k >= n:
                    break


def _blobs_session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    img = s.active
    _add_traces(s, img)
    return s, img


def _new_images(session, out):
    return [session.images[i] for i in out["new_images"]]


def _bright_sphere(seed=0, shape=(16, 40, 40), radius=9):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.indices(shape)
    c = tuple(v // 2 for v in shape)
    sph = (zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2 <= radius * radius
    vol = ndi.gaussian_filter(np.where(sph, 200.0, 50.0) + rng.normal(0, 20, shape), 0.5)
    return np.clip(vol, 0, 255).astype(np.uint8), sph


def _sphere_session():
    data, sph = _bright_sphere()
    s = Session()
    img = s.add_image(LumenImage(data, name="sphere"))
    s.roi_manager.add({"type": "rect", "x": 16, "y": 16, "w": 8, "h": 8, "name": "class:foreground"}, image_id=img.id, position={"t": 0, "z": 8, "c": 0})
    s.roi_manager.add({"type": "rect", "x": 1, "y": 1, "w": 8, "h": 8, "name": "class:background"}, image_id=img.id, position={"t": 0, "z": 2, "c": 0})
    return s, img, sph


# ------------------------------------------------------------------ registry / helpers
def test_new_ops_registered():
    for op_id in ("segment.classifier_info", "segment.classifier_overlay", "segment.apply_classifier_folder"):
        spec = REGISTRY.get(op_id)
        assert spec.category == "Plugins/Segmentation" and spec.kind == "analysis", op_id
    train = REGISTRY.get("segment.train_classifier")
    names = {p.name for p in train.params}
    assert {"classifier", "balance_classes", "features_per_node", "min_sigma", "max_sigma", "membrane_thickness", "membrane_patch_size", "class_colors"} <= names
    assert next(p for p in train.params if p.name == "classifier").choices == clf.CLASSIFIER_CHOICES
    assert next(p for p in train.params if p.name == "features").choices == clf.FEATURE_CHOICES and len(clf.FEATURE_CHOICES) == 20
    assert next(p for p in train.params if p.name == "output").choices == ["Probabilities", "Labels", "Both", "Overlay"]
    apply = REGISTRY.get("segment.apply_classifier")
    assert next(p for p in apply.params if p.name == "image").type == "image"
    train3 = REGISTRY.get("segment3d.train_classifier")
    assert {"classifier", "min_sigma", "balance_classes"} <= {p.name for p in train3.params}
    assert {"Laplacian", "Mean", "Median", "Structure"} <= set(next(p for p in train3.params if p.name == "features").choices)


def test_feature_normalisation_sigmas_and_colours():
    assert clf.normalize_features("Gaussian blur, Sobel filter, Hessian, Difference of gaussians, Neighbours") == \
        ["Gaussian", "Difference of Gaussians", "Sobel", "Hessian", "Neighbors"]
    assert clf.normalize_features([]) == ["Gaussian"]
    assert clf.sigma_list(1.0, 8.0) == [1.0, 2.0, 4.0, 8.0] and clf.sigma_list(0.5, 2.0) == [0.5, 1.0, 2.0] and clf.sigma_list(4.0, 2.0) == [2.0]
    assert seg._sigmas(8.0) == [1.0, 2.0, 4.0, 8.0]  # the historical ladder is unchanged
    assert clf.parse_colors("#112233, 445566, bad", 4) == ["#112233", "#445566", "#0000ff", "#00ffff"]
    assert clf.hex_to_rgb("#ff8000") == (255, 128, 0)
    assert clf.normalize_classifier("randomForest") == "Random forest" and clf.normalize_classifier("knn") == "k-NN"
    with pytest.raises(ValueError, match="Unknown classifier"):
        clf.normalize_classifier("perceptron tree")


def test_class_of_accepts_trace_naming():
    classes = ["background", "cell", "cell nucleus"]
    assert seg._class_of("class:cell", classes) == 1 and seg._class_of("Class: cell nucleus", classes) == 2
    assert seg._class_of("cell 3", classes) == 1 and seg._class_of("background-1", classes) == 0 and seg._class_of("other", classes) is None


def test_trace_tool_adds_and_removes_traces():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    img = s.active
    spec = REGISTRY.get("segment.trace_tool")
    assert spec.tool and spec.tool_icon and {p.name for p in spec.params} == {"class_name", "width", "color"}
    assert s.run_op("segment.trace_tool", {"class_name": "foreground", "event": "drag-start", "x": 1, "y": 1}) .get("message") is None
    out = s.run_op("segment.trace_tool", {"class_name": "foreground", "width": 5, "color": "#00ff00", "x": 10, "y": 12, "x2": 40, "y2": 12, "event": "drag-end"})
    assert "trace added (1 trace" in out["message"]
    s.run_op("segment.trace_tool", {"class_name": "background", "width": 9, "x": 100.4, "y": 100.6, "event": "click"})
    rois = s.roi_manager.to_list()
    assert [r["name"] for r in rois] == ["class:foreground", "class:background"] and rois[0]["type"] == "polyline" and rois[0]["width"] == 5
    assert rois[0]["color"] == "#00ff00" and rois[0]["position"] == {"t": 0, "z": 0, "c": 0} and rois[1]["type"] == "oval" and rois[1]["w"] == 9
    from lumen.core import roi as roimod
    m = roimod.to_mask(rois[0], (img.height, img.width))
    assert m[12, 25] and m[10, 25] and not m[20, 25] and 100 < m.sum() < 220  # a 5 px wide, 30 px long stroke
    assert len(seg._trace_rois(s, img)) == 2 and len(seg._trace_rois(s, img, "foreground")) == 1
    # the classes train from these traces; ⌥-click removes the nearest trace of a class
    s.run_op("segment.train_classifier", {**FAST, "apply_now": False, "classes": "background,foreground"})
    assert all(n > 0 for n in s.settings["classifier"]["n_samples"])
    out = s.run_op("segment.trace_tool", {"class_name": "foreground", "x": 12, "y": 12, "alt": True, "event": "click"})
    assert "Removed" in out["message"] and [r["name"] for r in s.roi_manager.to_list()] == ["class:background"]
    with pytest.raises(ValueError, match="No 'foreground' trace"):
        s.run_op("segment.trace_tool", {"class_name": "foreground", "x": 12, "y": 12, "alt": True, "event": "click"})


def test_rank_features_keep_their_resolution_on_narrow_range_16bit_data():
    """The 8-bit copy behind Median / Bilateral / Entropy (and the Lipschitz slopes) spans the plane's own range like
    Weka's convertToByte(true): 12-bit-in-16-bit or low-count fluorescence data no longer collapses to a handful of
    grey levels (version 3 gave 1-3 distinct Median / Bilateral values, i.e. constant columns)."""
    rng = np.random.default_rng(0)
    feats = ["Median", "Bilateral", "Entropy", "Lipschitz"]
    for lo, hi in ((100, 600), (0, 4095)):
        p16 = (rng.random((64, 64)) * (hi - lo) + lo).astype(np.uint16)
        settings = clf.feature_settings("Current channel", feats, 1.0, 2.0, dtype=np.uint16)
        assert settings["feature_version"] == clf.FEATURE_VERSION == 4 and settings["intensity_scale"] == 65535.0
        F, names = clf.plane_features(p16, settings)
        assert np.isfinite(F).all() and np.allclose(F[..., 0], p16 / 65535.0)
        for i, n in enumerate(names):
            if n.startswith(("Median", "Bilateral", "Lipschitz")):
                assert len(np.unique(F[..., i])) > 50 and F[..., i].std() > 0, (lo, hi, n)
            elif n.startswith("Entropy"):
                assert F[..., i].std() > 0 and F[..., i].max() > 1.0, n
        # rank outputs are mapped back onto the scaled plane's units: the median stays comparable with Original
        m = F[..., names.index("Median_1")]
        direct = ndi.median_filter(p16 / 65535.0, footprint=clf._disk(1), mode="reflect")
        assert np.abs(m - direct)[2:-2, 2:-2].max() <= (hi - lo) / 65535.0 / 255.0 + 1e-7
        # a classifier trained on uint16 transfers to a float32 image holding the same values
        Ff, _ = clf.plane_features(p16.astype(np.float32), settings)
        assert np.array_equal(F, Ff)
        # a saved version-3 model keeps the exact features it was trained with (the fixed 0..1 copy)
        F3, n3 = clf.plane_features(p16, {**settings, "feature_version": 3})
        from skimage.filters.rank import median
        q3 = np.clip(p16 / 65535.0 * 255.0 + 0.5, 0, 255).astype(np.uint8)
        assert n3 == names and np.allclose(F3[..., names.index("Median_1")], median(q3, clf._disk(1)) / 255.0)
        assert len(np.unique(F3[..., names.index("Bilateral_5_50")])) <= 5
    # an 8-bit image using its full range gives the same copy as before (version 3 and 4 agree)
    p8 = _blobs(5).copy()
    p8[0, 0], p8[-1, -1] = 0, 255
    s8 = clf.feature_settings("Current channel", ["Median", "Bilateral", "Lipschitz"], 1.0, 1.0, dtype=np.uint8)
    F4, n4 = clf.plane_features(p8, s8)
    F3, _ = clf.plane_features(p8, {**s8, "feature_version": 3})
    assert p8.min() == 0 and p8.max() == 255 and np.allclose(F4, F3, atol=1e-6)
    # NaN / infinite pixels and a flat plane do not break the copy
    q, lo, hi = clf._to_uint8(np.array([[np.nan, 0.2], [0.4, np.inf]], np.float32))
    assert (lo, hi) == pytest.approx((0.2, 0.4)) and q[0, 1] == 0 and q[1, 0] == 255
    assert clf._to_uint8(np.full((4, 4), 0.5, np.float32))[1:] == (0.5, 1.5)


def test_full_feature_bank_and_legacy_features():
    p = _blobs(5)
    settings = clf.feature_settings("Current channel", clf.FEATURE_CHOICES, 1.0, 2.0, dtype=p.dtype, membrane_thickness=2, membrane_patch_size=15)
    F, names = clf.plane_features(p, settings)
    assert F.shape == (p.shape[0], p.shape[1], len(names)) and F.dtype == np.float32 and np.isfinite(F).all()
    assert len(set(names)) == len(names) and names[0] == "Original"
    assert "Hessian_det_2" in names and "Membrane_median" in names and "Kuwahara_var_mean2" in names and "Lipschitz_20" in names
    assert "Gabor_max_1" in names and "Derivatives_5_2" in names and "Structure_largest_2" in names and "Neighbor_1_1_2" in names
    assert "Bilateral_10_100" in names and "Anisotropic_2" in names and "Median_2" in names
    # a version-2 model (no feature_version) still gets its two Hessian eigenvalues and 12-angle membrane projections
    legacy = {"features": ["Hessian", "Membrane projections"], "sigmas": [1.0], "intensity_scale": 255.0}
    Fl, nl = clf.plane_features(p, legacy)
    assert nl == ["Original", "Hessian_max_1", "Hessian_min_1", "Membrane_mean", "Membrane_max", "Membrane_min", "Membrane_std"]
    assert np.allclose(Fl[..., 0], p / 255.0)
    # the Lipschitz top-hat is non-negative and zero on a flat image; anisotropic diffusion keeps the mean
    flat = np.full((20, 20), 0.3, np.float32)
    assert np.allclose(clf.lipschitz_tophat(flat, 0.02), 0.0) and (clf.lipschitz_tophat(p / 255.0, 0.02) >= -1e-6).all()
    assert abs(float(clf.anisotropic_diffusion(p.astype(np.float32) / 255.0, 1.0).mean()) - float(p.mean() / 255.0)) < 0.01


# ------------------------------------------------------------------ training with every classifier
@pytest.mark.parametrize("classifier", clf.CLASSIFIER_CHOICES)
def test_train_every_classifier_on_two_class_image(classifier):
    s, img = _blobs_session()
    out = s.run_op("segment.train_classifier", {**FAST, "classifier": classifier, "hidden_units": 16, "k_neighbors": 3, "svm_max_samples": 800, "output": "Labels"})
    model = s.settings["classifier"]
    assert model["classifier"] == classifier and model["kind"] == "classifier" and model["version"] == clf.MODEL_VERSION
    assert model["classes"] == ["background", "foreground"] and model["colors"] == ["#ff0000", "#00ff00"]
    assert sum(model["n_samples"]) > 0 and model["train_time"] >= 0.0 and model["n_fitted"] <= sum(model["n_samples"])
    labels = _new_images(s, out)[0]
    assert labels.name == f"Classification of {img.name}" and labels.metadata["class_colors"] == "#ff0000, #00ff00"
    assert _iou(labels.current_plane() == 1, _otsu_mask(img)) > 0.75, classifier
    row = s.results.rows[-1]
    assert row["Classifier"] == classifier and row["Pixels 'foreground'"] > 0 and "Training time (s)" in row
    if classifier in ("Random forest", "Extra trees"):
        assert model["accuracy_kind"] == "out-of-bag" and 0.5 < row["OOB accuracy"] <= 1.0 and model["importances"] is not None
    else:
        assert model["accuracy_kind"] == "hold-out" and 0.5 < row["Holdout accuracy"] <= 1.0
    if classifier == "SVM":
        assert model["n_fitted"] <= 800
    assert "Trained " + classifier in out["message"]
    # the summary tables are published in the session state
    assert set(s.tables) >= {"Classifier", "Classifier classes", "Classifier summary"}
    classes = s.tables["Classifier classes"]
    assert classes["columns"] == ["Index", "Class", "Colour", "Training samples", "Share %"] and [r[1] for r in classes["rows"]] == ["background", "foreground"]
    summary = dict(s.tables["Classifier summary"]["rows"])
    assert summary["Classifier"] == classifier and summary["Kind"].startswith("2-D") and summary["Training samples"] == sum(model["n_samples"])


def test_balance_and_sample_cap():
    s, img = _blobs_session()
    s.run_op("segment.train_classifier", {**FAST, "classifier": "k-NN", "balance_classes": True, "max_pixels_per_class": 120, "apply_now": False})
    m = s.settings["classifier"]
    assert m["n_samples"][0] == m["n_samples"][1] == 120  # k-NN balances by subsampling to the smallest (capped) class
    s.run_op("segment.train_classifier", {**FAST, "classifier": "Random forest", "balance_classes": False, "max_pixels_per_class": 100, "apply_now": False})
    assert s.settings["classifier"]["n_samples"] == [100, 100] and s.settings["classifier"]["params"]["balance_classes"] is False
    with pytest.raises(ValueError, match="No training pixels for class"):
        s.run_op("segment.train_classifier", {**FAST, "classes": "background,foreground,nucleus", "apply_now": False})


def test_trace_names_and_prefix_names_both_train():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    img = s.active
    _add_traces(s, img, naming="prefix")
    s.run_op("segment.train_classifier", {**FAST, "apply_now": False})
    counts = s.settings["classifier"]["n_samples"]
    assert all(0 < n <= 8 * 49 for n in counts) and len(counts) == 2  # rectangles may overlap, so at most 8 × 49 per class


# ------------------------------------------------------------------ outputs: probabilities, other image, overlay
def test_probability_output_shape_and_apply_to_other_image():
    s, img = _blobs_session()
    s.run_op("segment.train_classifier", {**FAST, "apply_now": False})
    other = s.add_image(LumenImage(np.stack([_blobs(7), _blobs(8)])[:, None], name="pair"))  # (Z=2, C=1) stack
    s.select(img.id)
    out = s.run_op("segment.apply_classifier", {"image": other.id, "all_planes": True, "output": "Probabilities"})
    probs = _new_images(s, out)[0]
    assert probs.name == "Probabilities of pair" and probs.shape == (1, 2, 2, 256, 256) and probs.dtype == np.float32
    assert np.allclose(probs.data.sum(axis=2), 1.0, atol=1e-4) and [c.name for c in probs.display.channels] == ["background", "foreground"]
    out = s.run_op("segment.apply_classifier", {"image": "pair", "all_planes": False, "output": "Labels"})
    lab = _new_images(s, out)[0]
    assert lab.shape == (1, 1, 1, 256, 256) and _iou(lab.current_plane() == 1, _otsu_mask(other)) > 0.8
    with pytest.raises((KeyError, ValueError)):
        s.run_op("segment.apply_classifier", {"image": "no such image"})


def test_overlay_output_and_toggle():
    s, img = _blobs_session()
    out = s.run_op("segment.train_classifier", {**FAST, "output": "Overlay", "overlay_opacity": 0.5, "class_colors": "#ff0000,#00ff00"})
    assert out["new_images"] == [] and s.active_id == img.id  # the overlay image is not activated
    ov = seg._find_overlay_image(s, img)
    assert ov is not None and ov.rgb and ov.shape == (1, 1, 3, 256, 256) and ov.name == f"Classifier overlay of {img.name}"
    assert ov.metadata["classifier_overlay_of"] == img.id and ov.metadata["classifier_overlay_z"] == 0
    item = next(o for o in img.overlay if o.get("source") == "classifier")
    assert item["type"] == "image" and item["image_id"] == ov.id and item["opacity"] == 0.5 and item["position"] == {"t": 0, "z": 0, "c": None}
    fg = _otsu_mask(img)
    green = (ov.data[0, 0, 1] == 255) & (ov.data[0, 0, 0] == 0)
    assert _iou(green, fg) > 0.8
    # toggle hides, shows (with a new opacity) and removes
    s.run_op("segment.classifier_overlay", {"action": "Toggle"})
    assert not any(o.get("source") == "classifier" for o in img.overlay)
    s.run_op("segment.classifier_overlay", {"action": "Toggle", "opacity": 0.8})
    assert next(o for o in img.overlay if o.get("source") == "classifier")["opacity"] == 0.8
    # a second apply replaces the overlay image (the stale one is closed) and a whole-stack overlay has no plane pin
    s.run_op("segment.apply_classifier", {"output": "Overlay", "all_planes": True})
    ov2 = seg._find_overlay_image(s, img)
    assert ov2 is not None and ov2.id != ov.id and ov.id not in s.images and "position" not in next(o for o in img.overlay if o.get("source") == "classifier")
    s.run_op("segment.classifier_overlay", {"action": "Remove"})
    assert ov2.id not in s.images and not any(o.get("source") == "classifier" for o in img.overlay)
    with pytest.raises(ValueError, match="apply the classifier with output 'Overlay'"):
        s.run_op("segment.classifier_overlay", {"action": "Show"})


# ------------------------------------------------------------------ persistence + info
def test_save_load_round_trip_every_classifier(tmp_path):
    for classifier in ("Random forest", "Gradient boosting", "SVM", "MLP"):
        s, img = _blobs_session()
        out = s.run_op("segment.train_classifier", {**FAST, "classifier": classifier, "hidden_units": 8, "output": "Probabilities", "class_colors": "#123456,#abcdef"})
        ref = _new_images(s, out)[0].data.copy()
        path = tmp_path / f"{classifier.replace(' ', '_')}"  # no extension: .joblib is added
        out = s.run_op("segment.save_classifier", {"path": str(path)})
        saved = tmp_path / f"{classifier.replace(' ', '_')}.joblib"
        assert saved.exists() and str(saved) in out["message"] and s.settings["classifier"]["path"] == str(saved)
        s2 = Session()
        s2.run_op("file.sample", {"name": "Blobs"})
        out = s2.run_op("segment.load_classifier", {"path": str(saved)})
        m = s2.settings["classifier"]
        assert m["classifier"] == classifier and m["colors"] == ["#123456", "#abcdef"] and m["classes"] == ["background", "foreground"]
        assert "sklearn_version" in m and m["saved"] and out["table"]["name"] == "Classifier" and "Classifier summary" in s2.tables
        out = s2.run_op("segment.apply_classifier", {"output": "Probabilities"})
        assert np.array_equal(_new_images(s2, out)[0].data, ref), classifier
    # a directory gets the default name; 3-D files are refused by the 2-D loader
    s.run_op("segment.save_classifier", {"path": str(tmp_path)})
    assert (tmp_path / "classifier.joblib").exists()
    import joblib
    joblib.dump({"model": None, "classes": ["a", "b"], "settings": {}, "kind": "classifier3d"}, tmp_path / "vol.joblib")
    with pytest.raises(ValueError, match="3-D pixel classifier"):
        s.run_op("segment.load_classifier", {"path": str(tmp_path / "vol.joblib")})
    (tmp_path / "junk.joblib").write_bytes(b"not a model")
    with pytest.raises(ValueError, match="not a Lumen pixel classifier"):
        s.run_op("segment.load_classifier", {"path": str(tmp_path / "junk.joblib")})


def test_classifier_info_tables():
    s, img = _blobs_session()
    with pytest.raises(ValueError, match="No trained pixel classifier"):
        s.run_op("segment.classifier_info", {})
    s.run_op("segment.train_classifier", {**FAST, "apply_now": False, "features": ["Gaussian", "Sobel"]})
    out = s.run_op("segment.classifier_info", {})
    table = out["table"]
    assert table["name"] == "Classifier" and table["columns"] == ["Rank", "Feature", "Importance"]
    assert [r[0] for r in table["rows"]] == list(range(1, 6)) and {r[1] for r in table["rows"]} == {"Original", "Gaussian_1", "Gaussian_2", "Sobel_1", "Sobel_2"}
    imps = [r[2] for r in table["rows"]]
    assert imps == sorted(imps, reverse=True) and abs(sum(imps) - 1.0) < 1e-6
    classes = s.tables["Classifier classes"]["rows"]
    assert classes[1][1:3] == ["foreground", "#00ff00"] and classes[0][3] == s.settings["classifier"]["n_samples"][0]
    assert "Random forest" in out["message"] and "training samples" in out["message"]
    # without importances (SVM) the column is empty, never NaN
    s.run_op("segment.train_classifier", {**FAST, "classifier": "SVM", "apply_now": False, "features": ["Gaussian"]})
    out = s.run_op("segment.classifier_info", {})
    assert all(r[2] is None for r in out["table"]["rows"])


# ------------------------------------------------------------------ 3-D variant
def test_train_classifier3d_choice_features_and_round_trip(tmp_path):
    s, img, sph = _sphere_session()
    out = s.run_op("segment3d.train_classifier", {"classifier": "Extra trees", "n_trees": 15, "max_sigma": 2, "features": ["Gaussian", "Laplacian", "Mean", "Median", "Structure"],
                                                  "output": "Labels", "class_colors": "#000000,#ffffff"})
    m = s.settings["classifier3d"]
    assert m["kind"] == "classifier3d" and m["classifier"] == "Extra trees" and m["colors"] == ["#000000", "#ffffff"]
    assert m["settings"]["features"] == ["Gaussian", "Laplacian", "Mean", "Median", "Structure"] and m["settings"]["sigmas"] == [1.0, 2.0]
    assert m["n_features"] == 1 + 2 * (1 + 1 + 1 + 1 + 3) and "Structure3D_largest_2" in m["feature_names"] and "Median3D_1" in m["feature_names"]
    labels = _new_images(s, out)[0]
    assert _iou(labels.data[0, :, 0] == 1, sph) > 0.8 and "Classifier 3D classes" in s.tables
    assert dict(s.tables["Classifier 3D summary"]["rows"])["Kind"].startswith("3-D")
    path = s.run_op("segment3d.save_classifier", {"path": str(tmp_path / "vol.joblib")})
    s2 = Session()
    s2.add_image(LumenImage(_bright_sphere()[0], name="sphere"))
    s2.run_op("segment3d.load_classifier", {"path": str(tmp_path / "vol.joblib")})
    out2 = s2.run_op("segment3d.apply_classifier", {"output": "Labels"})
    assert np.array_equal(_new_images(s2, out2)[0].data, labels.data)
    # Overlay output on a stack covers every slice; apply to another open image through the `image` param
    src = s2.get_image("sphere")
    s2.select(src.id)  # the label stack became active
    s2.run_op("segment3d.apply_classifier", {"output": "Overlay"})
    ov = seg._find_overlay_image(s2, src)
    assert ov is not None and ov.shape == (1, 16, 3, 40, 40) and "position" not in next(o for o in src.overlay if o.get("source") == "classifier")
    other = s2.add_image(LumenImage(_bright_sphere(3)[0], name="sphere 2"))
    s2.select(src.id)
    out3 = s2.run_op("segment3d.apply_classifier", {"image": other.id, "output": "Probabilities"})
    probs = _new_images(s2, out3)[0]
    assert probs.name == "Probabilities of sphere 2" and probs.shape == (1, 16, 2, 40, 40) and np.allclose(probs.data.sum(axis=2), 1.0, atol=1e-4)
    assert "path" in path["message"] or "Saved" in path["message"]


# ------------------------------------------------------------------ folder batch
def test_apply_classifier_folder(tmp_path):
    from lumen.core.io import read_image, write_image
    s, img = _blobs_session()
    s.run_op("segment.train_classifier", {**FAST, "apply_now": False})
    src = tmp_path / "in"
    src.mkdir()
    for i in (3, 4):
        write_image(LumenImage(_blobs(i), name=f"b{i}"), str(src / f"blobs_{i}.tif"), "tif")
    (src / "notes.txt").write_text("skip me", encoding="utf-8")
    with pytest.raises(ValueError, match="Input folder not found"):
        s.run_op("segment.apply_classifier_folder", {"input_folder": str(tmp_path / "missing")})
    out = s.run_op("segment.apply_classifier_folder", {"input_folder": str(src), "output": "Both"})
    out_dir = src / "lumen_classified"
    assert sorted(p.name for p in out_dir.iterdir()) == ["blobs_3_labels.tif", "blobs_3_probabilities.tif", "blobs_4_labels.tif", "blobs_4_probabilities.tif"]
    lab = read_image(str(out_dir / "blobs_3_labels.tif"))
    assert set(np.unique(lab.data).tolist()) <= {0, 1} and _iou(lab.data[0, 0, 0] == 1, _otsu_mask(s.add_image(LumenImage(_blobs(3), name="ref")))) > 0.8
    probs = read_image(str(out_dir / "blobs_3_probabilities.tif"))
    assert probs.nc == 2 and probs.dtype == np.float32
    assert "Classified 2 of 2" in out["message"] and [r["Status"] for r in s.results.rows[-2:]] == ["ok", "ok"] and "foreground" in s.results.rows[-1]["Classes"]
    out = s.run_op("segment.apply_classifier_folder", {"input_folder": str(src), "output": "Overlay", "output_format": "png", "output_folder": str(tmp_path / "ov"), "max_files": 1})
    files = sorted(p.name for p in (tmp_path / "ov").iterdir())
    assert files == ["blobs_3_overlay.png"] and read_image(str(tmp_path / "ov" / files[0])).rgb


# ------------------------------------------------------------------ REST
def test_api_train_overlay_and_state_tables():
    s = Session()
    c = TestClient(create_app(s))
    r = c.post("/api/ops/file.sample/run", json={"params": {"name": "Blobs"}})
    iid = r.json()["new_images"][0]
    _add_traces(s, s.images[iid])
    r = c.post("/api/ops/segment.train_classifier/run", json={"params": {**FAST, "classifier": "Logistic regression", "output": "Overlay"}, "image_id": iid})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["table"]["name"] == "Classifier" and "Trained Logistic regression" in body["message"]
    st = c.get("/api/state").json()  # NaN-free JSON with the published tables and the overlay item
    assert {"Classifier", "Classifier classes", "Classifier summary"} <= set(st["tables"])
    me = next(im for im in st["images"] if im["id"] == iid)
    item = next(o for o in me["overlay"] if o.get("source") == "classifier")
    ov = next(im for im in st["images"] if im["id"] == item["image_id"])
    assert ov["rgb"] and ov["metadata"]["classifier_overlay_of"] == iid and st["active"] == iid
    r = c.post("/api/ops/segment.classifier_overlay/run", json={"params": {"action": "Hide"}, "image_id": iid})
    assert r.status_code == 200 and not any(o.get("source") == "classifier" for o in c.get("/api/state").json()["images"][0]["overlay"])
    r = c.post("/api/ops/segment.classifier_info/run", json={"params": {}})
    assert r.status_code == 200 and r.json()["table"]["columns"] == ["Rank", "Feature", "Importance"]
    r = c.get(f"/api/images/{item['image_id']}/rgb.png")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/png")
