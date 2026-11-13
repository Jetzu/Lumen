"""bUnwarpJ-style cubic B-spline registration and transform files (lumen/core/ops/registration_bspline.py)."""
import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.data import camera
from skimage.transform import warp

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.ops.registration_bspline import (BGrid, _Problem, coeffs_to_field, field_to_coeffs, invert_field, load_transform, register_bspline,
                                                 save_elastic_transform, save_raw_transform, warp_field, warping_index)
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

CAM = camera().astype(np.float32)
BASE = CAM[100:228, 100:228]  # 128 × 128
INNER = (slice(16, -16), slice(16, -16))
NEW_OPS = ["registration.bunwarpj", "registration.bunwarpj_apply", "registration.bunwarpj_load", "registration.bunwarpj_save",
           "registration.bunwarpj_invert", "registration.bunwarpj_compare", "registration.bunwarpj_convert"]


def deform(plane, amp=3.0, period=50.0):
    """Smooth sinusoidal non-rigid deformation of a plane."""
    H, W = plane.shape
    rr, cc = np.mgrid[0:H, 0:W].astype(np.float32)
    v = amp * np.sin(2 * np.pi * cc / period)
    u = amp * np.cos(2 * np.pi * rr / period)
    return warp(plane, np.array([rr + v, cc + u]), order=1, mode="constant", preserve_range=True).astype(np.float32)


def mse(a, b):
    return float(np.mean((np.asarray(a, float)[INNER] - np.asarray(b, float)[INNER]) ** 2))


def test_ops_registered():
    for op_id in NEW_OPS:
        spec = REGISTRY.get(op_id)
        assert spec.to_dict()["menu"] == "Plugins/Registration", op_id
        assert spec.params and spec.description and spec.kind == "analysis"


def test_bspline_basis_identity_fit_refinement_and_file_coefficients():
    g = BGrid(4, 64, 48)
    cx, cy = g.identity()
    ux, uy = coeffs_to_field(4, cx, cy, 64, 48, absolute=True)
    assert np.abs(ux).max() < 1e-9 and np.abs(uy).max() < 1e-9  # identity coefficients → zero displacement (linear precision)
    rng = np.random.default_rng(0)
    dx, dy = rng.normal(size=(g.n, g.n)) * 3, rng.normal(size=(g.n, g.n)) * 3
    fx, fy = g.field(dx, dy)
    rx, ry = g.fit(fx, fy)
    assert np.allclose(rx, dx, atol=1e-9) and np.allclose(ry, dy, atol=1e-9)
    fx2, fy2 = BGrid(8, 64, 48).field(*BGrid(8, 64, 48).fit(fx, fy))  # a coarser field is exactly representable on the refined grid
    assert np.abs(fx2 - fx).max() < 1e-9 and np.abs(fy2 - fy).max() < 1e-9
    acx, acy = field_to_coeffs(fx, fy, 4)
    assert np.allclose(acx - cx, dx, atol=1e-8) and np.allclose(acy - cy, dy, atol=1e-8)
    with pytest.raises(ValueError, match="must be 7 × 7"):
        coeffs_to_field(4, np.zeros((5, 5)), np.zeros((5, 5)), 64, 48)


@pytest.mark.parametrize("bidirectional", [True, False])
def test_energy_gradient_matches_finite_differences(bidirectional):
    """The divergence / curl / landmark / consistency gradients are analytic (exact); the SSD gradient samples the
    spline-interpolated image gradient, which is exact on the pixel grid and approximate in between (a few per cent)."""
    rng = np.random.default_rng(1)
    T = ndi.gaussian_filter(rng.random((40, 50)), 2) * 255
    gd = BGrid(2, 50, 40)
    gi = BGrid(2, 46, 42) if bidirectional else None
    S = ndi.gaussian_filter(rng.random((42, 46)), 2) * 255 if gi else np.roll(T, 2, axis=1)
    lm_t = np.array([[10.0, 12.0], [30.0, 20.0], [40.0, 35.0]])
    lm_s = lm_t + np.array([2.0, -1.0])
    n = 2 * gd.n * gd.n + (2 * gi.n * gi.n if gi else 0)
    theta = rng.normal(size=n) * 0.7
    eps = 1e-4
    # analytic terms: every partial derivative
    p = _Problem(T, S, gd, gi, {"image": 0.0, "div": 0.5, "curl": 0.5, "landmark": 1.0, "consistency": 10.0}, lm_t, lm_s)
    e, grad = p.energy(theta)
    assert np.isfinite(e) and grad.shape == (n,)
    num = np.zeros(n)
    for i in range(n):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        num[i] = (p.energy(tp)[0] - p.energy(tm)[0]) / (2 * eps)
    assert np.abs(num - grad).max() < 1e-6 * np.abs(num).max()
    # image term: directional derivatives along random directions (the sampled gradient is exact on the pixel grid only)
    p = _Problem(T, S, gd, gi, {"image": 1.0, "div": 0.0, "curl": 0.0, "landmark": 0.0, "consistency": 0.0}, lm_t, lm_s)
    e, grad = p.energy(theta)
    for _ in range(5):
        d = rng.normal(size=n)
        d /= np.linalg.norm(d)
        fd = (p.energy(theta + eps * d)[0] - p.energy(theta - eps * d)[0]) / (2 * eps)
        assert abs(fd - grad @ d) < 0.03 * np.linalg.norm(grad) + 1e-6  # within 3 % of the gradient norm


@pytest.mark.parametrize("mode", ["Mono", "Fast"])
def test_bunwarpj_recovers_a_synthetic_warp_and_round_trips(mode, tmp_path):
    mov_plane = deform(BASE)
    s = Session()
    tgt = s.add_image(LumenImage(BASE.astype(np.uint8), name="target"))
    mov = s.add_image(LumenImage(mov_plane.astype(np.uint8), name="moving"))
    s.select(mov.id)
    params = {"target": tgt.id, "mode": mode, "final_deformation": "Very Fine", "output_field": True, "save_direct": str(tmp_path / "direct.txt")}
    if mode != "Mono":
        params.update(save_inverse=str(tmp_path / "inverse.txt"), output_target=True)
    out = s.run_op("registration.bunwarpj", params)
    imgs = [s.images[i] for i in out["new_images"]]
    reg, field = imgs[0], imgs[-1]
    assert reg.name == "moving on target (bUnwarpJ)" and reg.shape == tgt.shape and reg.dtype == np.uint8
    before, after = mse(mov_plane, BASE), mse(reg.data[0, 0, 0], BASE)
    assert after < before / 5, (before, after)
    assert field.nc == 2 and field.dtype == np.float32 and field.metadata.get("deformation_field") and [c.name for c in field.display.channels] == ["dx", "dy"]
    row = s.results.rows[-1]
    assert row["Label"] == "moving → target" and row["Mode"] == mode and row["Intervals"] == 8
    assert row["Max displacement"] > row["Mean displacement"] > 0.5 and row["Iterations"] > 3
    if mode != "Mono":
        assert len(imgs) == 3 and imgs[1].name == "target on moving (bUnwarpJ inverse)" and imgs[1].shape == mov.shape
        assert 0 <= row["Warping index"] < 1.5
        assert mse(imgs[1].data[0, 0, 0], mov_plane) < mse(BASE, mov_plane) / 3  # the inverse transform registers the target onto the source
        assert (tmp_path / "inverse.txt").is_file()
    # the stored field re-applied with 'Apply Deformation' reproduces the registered image on the other image
    s.select(mov.id)
    out2 = s.run_op("registration.apply_deformation", {})
    applied0 = s.images[out2["new_images"][0]].data.astype(float)
    # skimage.warp (Apply Deformation) and map_coordinates (bUnwarpJ) only differ for source coordinates within half a
    # pixel outside the image (edge value vs 0): the same field, up to rounding, everywhere else
    assert float((np.abs(applied0 - reg.data.astype(float)) > 1).mean()) < 0.01
    # the elastic transform file: bUnwarpJ format, same field as the session transform, applies through bunwarpj_apply
    text = (tmp_path / "direct.txt").read_text(encoding="utf-8")
    assert text.startswith("Intervals=8\n") and "X Coeffs" in text and "Y Coeffs" in text
    tr = load_transform(tmp_path / "direct.txt")
    assert tr["kind"] == "elastic" and tr["intervals"] == 8 and tr["cx"].shape == (11, 11)
    st = s.settings["bunwarpj"]["direct"]
    ux, uy = coeffs_to_field(8, tr["cx"], tr["cy"], 128, 128)
    sx, sy = coeffs_to_field(st["intervals"], st["cx"], st["cy"], st["width"], st["height"])
    assert np.allclose(ux, sx, atol=1e-4) and np.allclose(uy, sy, atol=1e-4)
    s.select(mov.id)
    out3 = s.run_op("registration.bunwarpj_apply", {"file": str(tmp_path / "direct.txt")})
    applied = s.images[out3["new_images"][0]]
    assert applied.shape == reg.shape and np.abs(applied.data.astype(float) - reg.data.astype(float)).max() <= 2
    # the numerically inverted transform maps the registered image back onto the moving one
    s.select(applied.id)
    out4 = s.run_op("registration.bunwarpj_apply", {"direction": "Direct", "invert": True})
    back = s.images[out4["new_images"][0]]
    assert mse(back.data[0, 0, 0], mov_plane) < mse(BASE, mov_plane) / 3
    s.select(applied.id)
    with pytest.raises(ValueError, match="different image"):
        s.run_op("registration.bunwarpj", {"target": applied.id})


def test_bunwarpj_landmarks_only_and_landmark_sources():
    s = Session()
    tgt = s.add_image(LumenImage(BASE.astype(np.uint8), name="target"))
    mov = s.add_image(LumenImage(BASE.astype(np.uint8), name="moving"))
    src_pts = [[20, 20], [100, 24], [30, 100], [96, 96], [60, 64]]
    tgt_pts = [[x - 3, y + 2] for x, y in src_pts]  # the source content sits 3 px right / 2 px up of the target's
    s.roi_manager.add({"type": "point", "points": src_pts, "name": "src"}, image_id=mov.id)
    s.roi_manager.add({"type": "point", "points": tgt_pts, "name": "tgt"}, image_id=tgt.id)
    s.select(mov.id)
    out = s.run_op("registration.bunwarpj", {"target": tgt.id, "mode": "Mono", "image_weight": 0.0, "landmark_weight": 1.0, "final_deformation": "Coarse"})
    row = s.results.rows[-1]
    assert row["N landmarks"] == 5 and row["Landmark RMS"] < 0.3
    st = s.settings["bunwarpj"]["direct"]
    ux, uy = coeffs_to_field(st["intervals"], st["cx"], st["cy"], st["width"], st["height"])
    assert abs(ux[64, 64] - 3) < 0.5 and abs(uy[64, 64] + 2) < 0.5  # target pixel samples the source 3 px right, 2 px up
    reg = s.images[out["new_images"][0]]
    assert np.abs(reg.data[0, 0, 0, 20:-20, 20:-20].astype(float) - BASE[18:-22, 23:-17]).mean() < 3
    # explicit point lists and ROI-manager names work too; mismatched counts are refused
    s.select(mov.id)
    s.run_op("registration.bunwarpj", {"target": tgt.id, "mode": "Mono", "image_weight": 0.0, "final_deformation": "Very Coarse",
                                        "source_landmarks": "src", "target_landmarks": "; ".join(f"{x},{y}" for x, y in tgt_pts)})
    assert s.results.rows[-1]["Landmark RMS"] < 0.5
    with pytest.raises(ValueError, match="correspond one to one"):
        s.run_op("registration.bunwarpj", {"target": tgt.id, "source_landmarks": "1,1; 2,2", "target_landmarks": "1,1"})
    with pytest.raises(ValueError, match="on both images"):  # the active (registered) image has no point ROIs, the target has
        s.run_op("registration.bunwarpj", {"target": tgt.id, "source_landmarks": "", "target_landmarks": ""})
    plain = Session()
    p1 = plain.add_image(LumenImage(BASE.astype(np.uint8), name="a"))
    plain.add_image(LumenImage(BASE.astype(np.uint8), name="b"))
    with pytest.raises(ValueError, match="needs landmarks"):
        plain.run_op("registration.bunwarpj", {"target": p1.id, "image_weight": 0.0})


def test_transform_file_ops_save_load_invert_compare_convert(tmp_path):
    s = Session()
    tgt = s.add_image(LumenImage(BASE.astype(np.uint8), name="target"))
    mov = s.add_image(LumenImage(deform(BASE, 2.0, 60.0).astype(np.uint8), name="moving"))
    s.select(mov.id)
    s.run_op("registration.bunwarpj", {"target": tgt.id, "mode": "Fast", "final_deformation": "Coarse", "max_iterations": 15})
    st = s.settings["bunwarpj"]
    assert st["direct"]["intervals"] == 2 and st["inverse"] is not None
    out = s.run_op("registration.bunwarpj_save", {"file": str(tmp_path / "d.txt"), "inverse_file": str(tmp_path / "i.txt")})
    assert "d.txt" in out["message"] and load_transform(tmp_path / "i.txt")["intervals"] == 2
    s.run_op("registration.bunwarpj_save", {"file": str(tmp_path / "d_raw.txt"), "format": "Raw (pixel coordinates)"})
    raw = load_transform(tmp_path / "d_raw.txt")
    assert raw["kind"] == "raw" and raw["x"].shape == (128, 128) and abs(raw["x"][0, 5] - 5) < 10
    # compare: the session's direct / inverse pair has a small warping index; same-direction distance of a file with itself is 0
    s.run_op("registration.bunwarpj_compare", {})
    row = s.results.rows[-1]
    assert row["Mode"].startswith("Opposite") and 0 <= row["Warping index"] < 1.0
    s.run_op("registration.bunwarpj_compare", {"file1": str(tmp_path / "d.txt"), "file2": str(tmp_path / "d_raw.txt"), "mode": "Same direction (mean distance)"})
    assert s.results.rows[-1]["Mean distance"] < 1e-3
    # convert: elastic → raw and raw → elastic (fit)
    s.run_op("registration.bunwarpj_convert", {"file": str(tmp_path / "d.txt"), "output_file": str(tmp_path / "conv_raw.txt")})
    conv = load_transform(tmp_path / "conv_raw.txt")
    assert conv["kind"] == "raw" and np.allclose(conv["x"], raw["x"], atol=1e-3)
    out = s.run_op("registration.bunwarpj_convert", {"file": str(tmp_path / "conv_raw.txt"), "output_file": str(tmp_path / "fit.txt"), "mode": "Raw → elastic (B-spline fit)", "intervals": 2})
    assert "mean fit error 0.0" in out["message"] and load_transform(tmp_path / "fit.txt")["intervals"] == 2
    # invert (numerical) with swap, then load a file as the inverse
    out = s.run_op("registration.bunwarpj_invert", {"output_file": str(tmp_path / "inv.txt"), "swap": True})
    assert "swapped" in out["message"] and s.settings["bunwarpj"]["source"] == "target" and load_transform(tmp_path / "inv.txt")["intervals"] == 2
    s.select(mov.id)
    out = s.run_op("registration.bunwarpj_load", {"file": str(tmp_path / "d.txt"), "output_field": True})
    fimg = s.images[out["new_images"][0]]
    assert fimg.nc == 2 and s.settings["bunwarpj"]["direct"]["intervals"] == 2 and s.settings["deformation"]["flow"].shape == (2, 128, 128)
    s.run_op("registration.bunwarpj_load", {"file": str(tmp_path / "i.txt"), "as_inverse": True})
    assert s.settings["bunwarpj"]["inverse"]["intervals"] == 2
    # errors
    with pytest.raises(ValueError, match="not found"):
        load_transform(tmp_path / "missing.txt")
    (tmp_path / "junk.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="neither"):
        load_transform(tmp_path / "junk.txt")
    fresh = Session()
    fresh.add_image(LumenImage(BASE.astype(np.uint8), name="x"))
    with pytest.raises(ValueError, match="No bUnwarpJ transform"):
        fresh.run_op("registration.bunwarpj_apply", {})
    with pytest.raises(ValueError, match="Give the transform file"):
        fresh.run_op("registration.bunwarpj_load", {})


def test_invert_field_and_warping_index_helpers():
    H, W = 60, 70
    rr, cc = np.mgrid[0:H, 0:W].astype(float)
    ux = 2.0 * np.sin(2 * np.pi * rr / 60)
    uy = 1.5 * np.cos(2 * np.pi * cc / 70)
    vx, vy = invert_field(ux, uy)
    assert warping_index(ux, uy, vx, vy) < 0.05
    plane = ndi.gaussian_filter(np.random.default_rng(3).random((H, W)), 2)
    once = warp_field(plane, ux, uy, 1)
    back = warp_field(once, vx, vy, 1)
    assert np.abs(back - plane)[8:-8, 8:-8].max() < 0.05
    # register_bspline on raw arrays (no session), sizes differing between source and target
    res = register_bspline(plane * 255, np.pad(plane, ((0, 4), (0, 6)))[:, :] * 255, mode="Mono", initial_level=0, final_level=1, max_iter=10)
    assert res["direct"][0] == 2 and res["direct"][1].shape == (5, 5) and res["inverse"] is None
    with pytest.raises(ValueError, match="at least 8"):
        register_bspline(np.zeros((4, 4)), np.zeros((4, 4)))
    with pytest.raises(ValueError, match="Unknown deformation level"):
        from lumen.core.ops.registration_bspline import _level_index
        _level_index("Ultra")


def test_raw_transform_file_apply(tmp_path):
    """A raw transform file (absolute coordinates per pixel) warps the current image on its own canvas size."""
    H, W = 40, 50
    yy, xx = np.mgrid[0:H, 0:W]
    save_raw_transform(tmp_path / "raw.txt", xx + 3.0, yy + 0.0)  # every target pixel samples 3 px to the right
    s = Session()
    img = s.add_image(LumenImage(CAM[:H, :W + 10].astype(np.uint8), name="src"))
    out = s.run_op("registration.bunwarpj_apply", {"file": str(tmp_path / "raw.txt")})
    res = s.images[out["new_images"][0]]
    assert res.shape == (1, 1, 1, H, W) and np.array_equal(res.data[0, 0, 0], img.data[0, 0, 0, :, 3:W + 3])
    save_elastic_transform(tmp_path / "e.txt", 1, *BGrid(1, 10, 10).identity())
    assert load_transform(tmp_path / "e.txt")["cx"].shape == (4, 4)
