"""Image / Process gap fillers: auto crop, image ⇄ results, image overlays, FFT tool set, binary extras,
sliding paraboloid, local thresholds, NLC / ROF / shape index / circular masks, retinex, Math ▸ Macro,
stack tools (slice labels, montage ⇄ stack, hyperstack tools, statistics, line / radial reslice), batch measure."""
import numpy as np
import pytest
import tifffile
from skimage.measure import label as sk_label

from lumen.core import ops  # noqa: F401
from lumen.core.image import LumenImage
from lumen.core.session import Session


@pytest.fixture
def session():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    return s


def _two_disks():
    yy, xx = np.mgrid[:60, :100]
    disks = np.zeros((60, 100), np.uint8)
    disks[((xx - 35) ** 2 + (yy - 30) ** 2 < 400) | ((xx - 65) ** 2 + (yy - 30) ** 2 < 400)] = 255
    return disks


# ====================================================================== Image ▸ Adjust ▸ Auto Crop
def test_auto_crop_and_selection():
    s = Session()
    a = np.full((30, 40), 7, np.uint8)
    a[5:15, 10:25] = 200
    img = s.add_image(LumenImage(a, name="ac"))
    out = s.run_op("image.auto_crop_selection")
    roi = out["roi"]
    assert (roi["x"], roi["y"], roi["w"], roi["h"]) == (10, 5, 15, 10)
    s.run_op("image.auto_crop")
    assert img.shape[3:] == (10, 15) and img.data.min() == 200
    # tolerance: a faint halo close to the background does not count
    b = np.full((20, 20), 100, np.uint8)
    b[8:12, 8:12] = 250
    b[5:15, 5:15] = np.where(b[5:15, 5:15] == 250, 250, 103)
    img2 = s.add_image(LumenImage(b, name="tol"))
    s.run_op("image.auto_crop", {"tolerance": 5})
    assert img2.shape[3:] == (4, 4)
    s.add_image(LumenImage(np.full((10, 10), 3, np.uint8), name="flat"))
    assert "Nothing to crop" in s.run_op("image.auto_crop")["message"]


# ====================================================================== Image ⇄ Results
def test_image_to_results_and_back():
    s = Session()
    a = np.arange(12, dtype=np.uint8).reshape(3, 4)
    s.add_image(LumenImage(a, name="small"))
    out = s.run_op("image.to_results")
    assert out["results_added"] == 3 and s.results.columns == ["X0", "X1", "X2", "X3"]
    assert s.results.rows[1]["X2"] == 6
    s.run_op("image.from_results", {"title": "back"})
    back = s.active
    assert back.name == "back" and back.dtype == np.float32 and np.array_equal(back.current_plane(), a.astype(np.float32))
    # with a selection only the bounding box is exported
    s.run_op("image.to_results", {}, image_id=s.get_image("small").id, roi={"type": "rect", "x": 1, "y": 1, "w": 2, "h": 2})
    assert s.results.columns == ["X1", "X2"] and s.results.rows[0]["X1"] == 5


# ====================================================================== image overlays
def test_overlay_add_image_and_burn(session):
    blobs = session.active
    small = session.add_image(LumenImage(np.full((20, 20), 255, np.uint8), name="sq"))
    session.select(blobs.id)
    session.run_op("overlay.add_image", {"image": small.id, "x": 10, "y": 10, "opacity": 50, "scale": 1.0})
    item = blobs.overlay[-1]
    assert item["type"] == "image" and item["image_id"] == small.id and item["opacity"] == pytest.approx(0.5) and item["x"] == 10
    session.run_op("overlay.burn_image")
    burned = session.active
    assert burned.rgb and burned.shape[3:] == (blobs.height, blobs.width)
    v = int(blobs.data[0, 0, 0, 15, 15])
    assert burned.data[0, 0, 0, 15, 15] == pytest.approx((v + 255) / 2, abs=1)   # blended 50 %
    assert burned.data[0, 0, 0, 100, 100] == blobs.data[0, 0, 0, 100, 100]        # untouched outside the inset
    with pytest.raises(ValueError):
        session.run_op("overlay.burn_image", {}, image_id=small.id)


# ====================================================================== FFT tool set
def test_fft_stores_transform_and_inverse_roundtrip(session):
    blobs = session.active
    session.run_op("process.fft")
    fimg = session.active
    assert fimg.dtype == np.uint8 and fimg.metadata["fft_source"] == blobs.id
    from lumen.core.ops._imagej import fft_pad_size
    n = fft_pad_size(blobs.width, blobs.height)        # ImageJ pads to a square power of two (FFT.pad)
    entry = session.settings["fft"][fimg.id]
    assert fimg.id in session.settings["fft"] and entry["F"].shape == (n, n) and entry["fht"].shape == (n, n)
    assert entry["orig_shape"] == (blobs.height, blobs.width) and (fimg.height, fimg.width) == (n, n)
    session.run_op("process.inverse_fft")
    inv = session.active
    assert inv.dtype == np.uint8 and np.array_equal(inv.current_plane(), blobs.current_plane())
    with pytest.raises(ValueError):
        session.run_op("process.inverse_fft", {}, image_id=blobs.id)  # not a frequency-domain image


def test_fft_masking_redisplay_and_swap(session):
    blobs = session.active
    session.run_op("process.fft")
    fimg = session.active
    ps = fimg.current_plane().copy()
    yy, xx = np.mgrid[: ps.shape[0], : ps.shape[1]]
    cy, cx = ps.shape[0] // 2, ps.shape[1] // 2
    ps[(yy - cy) ** 2 + (xx - cx) ** 2 < 25] = 0  # block the low frequencies
    fimg.replace_data(ps[None, None, None])
    session.run_op("process.inverse_fft", {"output": "32-bit"}, image_id=fimg.id)
    hp = session.active
    assert hp.dtype == np.float32 and abs(float(hp.data.mean())) < 1e-3  # DC removed
    assert hp.data.std() > 1
    # pass mode: only the white region (the DC term) passes → nearly constant image
    ps2 = session.settings["fft"][fimg.id]["F"]
    ps8 = np.zeros(ps.shape, np.uint8) + 128
    ps8[(yy - cy) ** 2 + (xx - cx) ** 2 <= 36] = 255  # (a region: the mask is smoothed three times like ImageJ's)
    fimg.replace_data(ps8[None, None, None])
    session.run_op("process.inverse_fft", {"output": "32-bit"}, image_id=fimg.id)
    lp = session.active
    assert lp.data.std() < hp.data.std() and abs(float(lp.data.mean()) - float(blobs.data.mean())) < 1.0
    session.run_op("process.fft_redisplay_power_spectrum", {}, image_id=fimg.id)
    assert fimg.current_plane().min() >= 1 and fimg.current_plane()[cy, cx] == 254   # ImageJ: log-scaled to 1…254
    before = fimg.current_plane().copy()
    session.run_op("process.fft_swap_quadrants", {}, image_id=fimg.id)
    assert np.array_equal(fimg.current_plane(), np.fft.fftshift(before))


def test_swap_quadrants_is_an_involution_on_odd_sizes():
    """ImageJ's swapQuadrants swaps the four half-blocks: applying it twice restores the plane for every size
    (np.fft.fftshift twice shifts an odd-sized plane by one pixel)."""
    s = Session()
    for shape in [(9, 9), (9, 11), (7, 8), (8, 8), (1, 5)]:
        img = s.add_image(LumenImage(np.arange(np.prod(shape), dtype=np.float32).reshape(shape), name=f"odd{shape}"))
        before = img.data.copy()
        s.run_op("process.fft_swap_quadrants")
        if shape[0] % 2 == 0 and shape[1] % 2 == 0:
            assert np.array_equal(img.data[0, 0, 0], np.fft.fftshift(before[0, 0, 0]))  # even sizes: identical to fftshift
        s.run_op("process.fft_swap_quadrants")
        assert np.array_equal(img.data, before), shape
    # one swap of a centred power spectrum moves the DC term to the FFT-origin corner (0, 0); like ImageJ the
    # 9×11 image is padded to 16×16 before the transform
    odd = s.add_image(LumenImage(np.full((9, 11), 50, np.uint8), name="odd-src"))
    odd.data[0, 0, 0, 4, 5] = 255
    out = s.run_op("process.fft", {}, image_id=odd.id)
    ps = s.images[out["new_images"][0]]
    assert ps.current_plane().shape == (16, 16)
    assert np.unravel_index(int(np.argmax(ps.current_plane())), (16, 16)) == (8, 8)
    s.run_op("process.fft_swap_quadrants", {}, image_id=ps.id)
    assert np.unravel_index(int(np.argmax(ps.current_plane())), (16, 16)) == (0, 0)
    s.run_op("process.inverse_fft", {}, image_id=ps.id)   # … and the inverse crops the padding again
    assert s.active.current_plane().shape == (9, 11) and np.array_equal(s.active.current_plane(), odd.current_plane())


def test_fft_options_outputs(session):
    blobs = session.active
    session.run_op("process.fft_options", {"fft_window": True, "raw_power_spectrum": True, "complex_output": True})
    assert session.settings["fft_options"]["complex_output"] is True
    out = session.run_op("process.fft", {}, image_id=blobs.id)
    names = [session.images[i].name for i in out["new_images"]]
    assert names == ["FFT of blobs.tif", "PS of blobs.tif", "Complex of blobs.tif"]
    comp = session.images[out["new_images"][2]]
    assert comp.nz == 2 and comp.dtype == np.float32 and comp.metadata["slice_labels"] == ["Real", "Imaginary"]
    raw = session.images[out["new_images"][1]]
    assert raw.dtype == np.float32 and raw.data.max() > 0
    session.run_op("process.fft_options", {"do_forward": False})
    with pytest.raises(ValueError):
        session.run_op("process.fft", {}, image_id=blobs.id)


def test_fd_math_and_custom_filter(session):
    blobs = session.active
    session.run_op("process.fd_math", {"image1": blobs.id, "operation": "Correlate", "image2": blobs.id, "do_inverse": True, "result": "AC"})
    ac = session.active
    assert ac.name == "AC" and ac.dtype == np.float32
    H, W = ac.height, ac.width
    assert np.unravel_index(int(np.argmax(ac.current_plane())), (H, W)) == (H // 2, W // 2)  # autocorrelation peaks at the origin (centred)
    out = session.run_op("process.fd_math", {"image1": blobs.id, "operation": "Convolve", "image2": blobs.id, "do_inverse": False, "result": "Conv"})
    conv = session.images[out["new_images"][0]]
    assert conv.dtype == np.uint8 and conv.id in session.settings["fft"]
    # deconvolving the convolution with the same image recovers the original (up to the centring shift)
    session.run_op("process.fd_math", {"image1": conv.id, "operation": "Deconvolve", "image2": blobs.id, "do_inverse": True, "regularization": 0.0, "result": "Dec"})
    dec = np.fft.ifftshift(session.active.current_plane())
    assert np.corrcoef(dec.ravel(), blobs.current_plane().astype(float).ravel())[0, 1] > 0.99
    # custom filter: an all-pass filter leaves the image unchanged, an all-block filter zeroes it
    passall = session.add_image(LumenImage(np.full((blobs.height, blobs.width), 255, np.uint8), name="pass"))
    before = blobs.data.copy()
    session.run_op("process.fft_custom_filter", {"filter": passall.id}, image_id=blobs.id)
    assert np.array_equal(blobs.data, before)
    block = session.add_image(LumenImage(np.zeros((blobs.height, blobs.width), np.uint8), name="block"))
    session.run_op("process.fft_custom_filter", {"filter": block.id}, image_id=blobs.id)
    assert blobs.data.max() == 0
    small = session.add_image(LumenImage(np.full((16, 16), 255, np.uint8), name="small"))  # resized to the image size
    blobs.replace_data(before)
    session.run_op("process.fft_custom_filter", {"filter": small.id}, image_id=blobs.id)
    assert np.array_equal(blobs.data, before)


def test_fd_math_correlate_orientation_matches_imagej():
    """FD Math Correlate = F1·conj(F2) like ImageJ's FHT.conjugateMultiply: with image2 = image1 shifted right by 5 and
    down by 3, the correlation peak sits at centre − shift (ImageJ 1.54p: x=27, y=29 on 64×64). conj(F1)·F2 would
    give the point-reflected image (peak at +5, +3), i.e. the opposite sign for every drift read off the peak."""
    s = Session()
    rng = np.random.default_rng(0)
    for shape in [(64, 64), (63, 61)]:
        f1 = rng.random(shape).astype(np.float32)
        f2 = np.roll(f1, (3, 5), axis=(0, 1))
        a = s.add_image(LumenImage(f1, name="f1"))
        b = s.add_image(LumenImage(f2, name="f2"))
        out = s.run_op("process.fd_math", {"image1": a.id, "operation": "Correlate", "image2": b.id, "do_inverse": True, "result": "corr"})
        r = s.images[out["new_images"][0]].current_plane()
        y, x = np.unravel_index(int(np.argmax(r)), r.shape)
        assert (int(x) - shape[1] // 2, int(y) - shape[0] // 2) == (-5, -3), shape
    # the frequency-domain output keeps ImageJ's orientation for a later Inverse FFT; the power spectrum is unaffected
    out = s.run_op("process.fd_math", {"image1": a.id, "operation": "Correlate", "image2": b.id, "do_inverse": False, "result": "corrF"})
    F = s.settings["fft"][out["new_images"][0]]["F"]
    F1 = np.fft.fftshift(np.fft.fft2(f1)); F2 = np.fft.fftshift(np.fft.fft2(f2))
    assert np.allclose(F, F1 * np.conj(F2), rtol=1e-4, atol=1e-3)


def test_show_circular_masks():
    s = Session()
    out = s.run_op("filters.show_circular_masks", {"max_radius": 3})
    img = s.images[out["new_images"][0]]
    assert img.nz == 6 and img.metadata["slice_labels"] == ["radius=0.5", "radius=1", "radius=1.5", "radius=2", "radius=2.5", "radius=3"]
    # ImageJ kernel rule dx² + dy² ≤ r² + 1: radius 0.5 → 3×3 cross without corners... (r²+1 = 1.25 → the 4-neighbours), radius 2 → 21 pixels
    counts = [int((img.data[0, z, 0] > 0).sum()) for z in range(6)]
    assert counts[0] == 5 and counts[3] == 21 and counts == sorted(counts)


# ====================================================================== Process ▸ Binary extras
def test_binary_dither_and_median():
    s = Session()
    ramp = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (64, 1))
    img = s.add_image(LumenImage(ramp, name="ramp"))
    s.run_op("binary.dither")
    plane = img.current_plane()
    assert set(np.unique(plane)) == {0, 255}
    assert 0.45 < (plane > 0).mean() < 0.55                      # error diffusion keeps the mean gray level
    assert (plane[:, :8] > 0).mean() < 0.1 and (plane[:, -8:] > 0).mean() > 0.9
    disks = _two_disks()
    noisy = disks.copy()
    rng = np.random.default_rng(0)
    flip = rng.random(disks.shape) < 0.05
    noisy[flip] = 255 - noisy[flip]
    nz = s.add_image(LumenImage(noisy, name="noisy"))
    s.run_op("binary.median", {"radius": 2})
    assert (nz.current_plane() != disks).sum() < 0.02 * disks.size and (noisy != disks).sum() > 0.04 * disks.size


def test_label_connected_components_does_not_wrap_above_65535():
    """Label images are uint8 up to 255 labels, uint16 up to 65535 and 32-bit beyond (like ImageJ's 32-bit option);
    a uint16 cast would wrap object 65536 to background and make later objects share labels 1, 2, ..."""
    s = Session()
    big = np.zeros((520, 520), np.uint8)
    big[::2, ::2] = 255  # 67600 isolated pixels
    s.add_image(LumenImage(big, name="big"))
    out = s.run_op("binary.label")
    lab = s.images[out["new_images"][0]]
    assert out["message"] == "67600 objects" and lab.dtype == np.float32
    plane = lab.current_plane()
    assert int(plane.max()) == 67600 and len(np.unique(plane)) == 67601 and not (plane[big > 0] == 0).any()
    assert np.array_equal(plane.astype(np.int64), sk_label(big > 0, connectivity=2))
    assert lab.display.channels[0].max == 67600.0 and lab.display.channels[0].lut == "Glasbey on dark"
    small = np.zeros((20, 20), np.uint8)
    small[::2, ::2] = 255  # 100 objects → uint8
    s.add_image(LumenImage(small, name="small"))
    assert s.images[s.run_op("binary.label")["new_images"][0]].dtype == np.uint8
    mid = np.zeros((40, 40), np.uint8)
    mid[::2, ::2] = 255  # 400 objects → uint16
    s.add_image(LumenImage(mid, name="mid"))
    assert s.images[s.run_op("binary.label")["new_images"][0]].dtype == np.uint16


def test_binary_skeletonize_methods(session):
    session.run_op("threshold.apply", {"method": "Default"})
    img = session.active
    binary = img.data.copy()
    sizes = {}
    for method in ["Zhang-Suen", "Lee", "Thin (Guo-Hall)", "Medial axis"]:
        img.replace_data(binary.copy())
        session.run_op("binary.skeletonize", {"method": method})
        sk = img.current_plane() > 0
        assert set(np.unique(img.current_plane())) <= {0, 255}
        assert sk_label(sk, connectivity=2).max() == sk_label(binary[0, 0, 0] > 0, connectivity=2).max()  # topology preserved
        sizes[method] = int(sk.sum())
    assert sizes["Medial axis"] > sizes["Zhang-Suen"] and 0 < sizes["Lee"] < (binary > 0).sum()


def test_watershed_methods_and_ultimate_points():
    s = Session()
    disks = _two_disks()
    for method in ["ImageJ EDM/UEP", "Distance transform (h-maxima)"]:
        img = s.add_image(LumenImage(disks, name="disks"))
        s.run_op("binary.watershed", {"method": method})
        assert sk_label(img.current_plane() > 0, connectivity=2).max() == 2, method
        assert (img.current_plane() > 0).sum() > 0.9 * (disks > 0).sum()
    img = s.add_image(LumenImage(disks, name="d"))
    out = s.run_op("binary.ultimate_points", {"output": "Point selection"})
    pts = out["roi"]["points"]
    assert len(pts) == 2 and sorted(round(p[0]) for p in pts) == [36, 66]
    out = s.run_op("binary.ultimate_points", {"output": "Image"})
    uep = s.images[out["new_images"][0]]
    assert uep.dtype == np.float32 and int((uep.data > 0).sum()) == 2 and uep.data.max() == pytest.approx(20.0, abs=1.0)


def test_ultimate_points_one_per_plateau_and_border_maxima_kept():
    """ImageJ's UEPs (MaximumFinder SINGLE_POINTS, excludeOnEdges=false): a flat EDM ridge gives one point (the plateau
    pixel nearest its centroid), and a maximum on the image border is reported rather than a lower neighbour."""
    from lumen.core.ops.binary import ultimate_eroded_points
    s = Session()
    rect = np.zeros((70, 90), np.uint8)
    rect[20:40, 20:60] = 255  # 20×40 rectangle: the EDM ridge is a 20-pixel plateau at value 10
    s.add_image(LumenImage(rect, name="rect"))
    out = s.run_op("binary.ultimate_points", {"output": "Point selection"})
    assert out["message"] == "1 ultimate points" and len(out["roi"]["points"]) == 1
    (px, py), = out["roi"]["points"]
    assert 20 <= px <= 60 and 29 <= py <= 31
    out = s.run_op("binary.ultimate_points", {"output": "Image"})
    uep = s.images[out["new_images"][0]].current_plane()
    assert int((uep > 0).sum()) == 1 and uep.max() == pytest.approx(10.0)
    for shape in [(20, 40), (3, 40), (20, 20), (2, 2)]:  # bars, squares and a 2×2 block: still one point each
        b = np.zeros((shape[0] + 10, shape[1] + 10), bool)
        b[5:5 + shape[0], 5:5 + shape[1]] = True
        assert len(ultimate_eroded_points(b)[0]) == 1, shape
    # a half disk cut by the top edge: the EDM maximum (15.0) lies on the border and must not be dropped
    yy, xx = np.mgrid[:40, :80]
    half = (xx - 40) ** 2 + yy ** 2 < 15 ** 2
    pts, edm = ultimate_eroded_points(half)
    assert pts.tolist() == [[0, 40]] and edm[0, 40] == pytest.approx(15.0)
    s.add_image(LumenImage(half.astype(np.uint8) * 255, name="half"))
    out = s.run_op("binary.ultimate_points", {"output": "Image"})
    assert s.images[out["new_images"][0]].current_plane().max() == pytest.approx(15.0)
    assert ultimate_eroded_points(np.zeros((5, 5), bool))[0].shape == (0, 2)


def test_signed_and_3d_distance_maps():
    s = Session()
    disks = _two_disks()
    s.add_image(LumenImage(disks, name="d"))
    out = s.run_op("binary.signed_distance_map")
    sd = s.images[out["new_images"][0]]
    assert sd.dtype == np.float32
    plane = sd.current_plane()
    assert plane[30, 35] == pytest.approx(20.0, abs=1.0) and plane[0, 0] < 0 and plane[30, 50] >= 0
    vol = np.zeros((5, 30, 30), np.uint8)
    vol[1:4, 10:20, 10:20] = 255
    v = s.add_image(LumenImage(vol, name="v", axes="ZYX"))
    v.calibration.pixel_depth = 3.0
    v.calibration.unit = "um"
    out = s.run_op("binary.distance_map_3d")
    edt = s.images[out["new_images"][0]]
    assert edt.dtype == np.float32 and edt.data.max() == pytest.approx(5.0)  # centre voxel: 5 px to the XY border < 3 µm·… in Z (anisotropic)
    out = s.run_op("binary.distance_map_3d", {"calibrated": False}, image_id=v.id)
    assert s.images[out["new_images"][0]].data.max() == pytest.approx(2.0)  # isotropic: 2 voxels to the Z border


# ====================================================================== Subtract background: sliding paraboloid
def test_sliding_paraboloid_background():
    from lumen.core.ops.enhance import parabola_envelope_1d, sliding_paraboloid_background
    flat = np.full(50, 10.0)
    flat[25] = 100
    env = parabola_envelope_1d(flat, 0.05)
    assert np.all(env <= flat + 1e-9) and env.min() == pytest.approx(10.0) and env[25] < 20
    ramp = np.tile(np.linspace(0, 100, 40), (40, 1))
    assert np.allclose(sliding_paraboloid_background(ramp, 10), ramp)   # a plane is its own background
    img = np.full((40, 40), 10.0)
    img[20, 20] = 200
    bg = sliding_paraboloid_background(img, 10)
    assert bg.min() == pytest.approx(10.0) and bg[20, 20] < 15
    s = Session()
    a = (np.linspace(20, 120, 64)[None, :] + np.zeros((64, 1))).astype(np.uint8)
    a[30:34, 30:34] = 250
    im = s.add_image(LumenImage(a, name="bg"))
    s.run_op("process.subtract_background", {"radius": 10, "method": "Sliding paraboloid", "smooth": False})
    out = im.current_plane()
    assert out[32, 32] > 100 and out[10, 10] < 10 and out[50, 60] < 10


# ====================================================================== Auto Local Threshold additions
def test_local_threshold_new_methods():
    s = Session()
    yy, xx = np.mgrid[:64, :64]
    base = (xx * 2).astype(np.uint8)   # slow horizontal gradient
    base[20:44, 20:44] += 60
    for method in ["Contrast", "MidGrey", "Otsu"]:
        img = s.add_image(LumenImage(base, name=method))
        s.run_op("threshold.local", {"method": method, "radius": 10})
        plane = img.current_plane()
        assert set(np.unique(plane)) <= {0, 255}
        # (32, 24) lies inside the bright square with a window reaching outside it; (5, 5) is plain gradient
        assert plane[32, 24] == 255 and plane[5, 5] == 0, method
    img = s.add_image(LumenImage(base, name="inv"))
    s.run_op("threshold.local", {"method": "MidGrey", "radius": 10, "white_objects": False})
    assert img.current_plane()[32, 24] == 0 and img.current_plane()[5, 5] == 255


def test_local_threshold_bernsen_black_objects():
    """Fiji's Bernsen for black objects only swaps the output values: a flat bright background is background (0), a
    flat dark object interior is object, and the mask is the bitwise inverse of the white-objects mask."""
    s = Session()
    a = np.full((80, 80), 200, np.uint8)
    a[10:25, 10:25] = 40
    a[50:70, 45:70] = 40
    img = s.add_image(LumenImage(a.copy(), name="bernsen-black"))
    s.run_op("threshold.local", {"method": "Bernsen", "radius": 5, "white_objects": False})
    p = img.current_plane()
    assert p[75, 5] == 0 and p[17, 17] == 255 and p[60, 55] == 255
    assert abs(float(np.mean(p == 255)) - float(np.mean(a == 40))) < 0.01
    img2 = s.add_image(LumenImage(a.copy(), name="bernsen-white"))
    s.run_op("threshold.local", {"method": "Bernsen", "radius": 5, "white_objects": True})
    assert np.array_equal(img2.current_plane(), 255 - p)
    u = s.add_image(LumenImage(np.full((40, 40), 200, np.uint8), name="uniform-bright"))
    s.run_op("threshold.local", {"method": "Bernsen", "radius": 5, "white_objects": False})
    assert not u.current_plane().any()
    # 8-bit midpoint is integer like Fiji's (min + max) / 2: a pixel equal to floor(mid) is object (>=)
    e = s.add_image(LumenImage(np.array([[0, 101, 101, 101, 101, 203]] * 6, np.uint8), name="edge"))
    s.run_op("threshold.local", {"method": "Bernsen", "radius": 5, "white_objects": True})
    assert e.current_plane()[0, 2] == 255


def test_threshold_apply_preview_has_no_side_effects():
    """Previewing Apply Threshold must neither flag the image as a binary mask nor drop the session threshold
    (otherwise the final Apply with method "Current" silently falls back to the Default auto-threshold)."""
    s = Session()
    g = s.add_image(LumenImage(np.arange(64, dtype=np.uint8).reshape(8, 8), name="ramp"))
    s.run_op("threshold.set", {"lower": 10, "upper": 40})
    pv1 = s.run_op("threshold.apply", {"method": "Current"}, preview=True)["preview"]
    assert g.lut_is_binary is False and g.summary()["binary_hint"] is False
    assert s.thresholds[g.id]["lower"] == 10 and np.array_equal(g.data[0, 0, 0], np.arange(64, dtype=np.uint8).reshape(8, 8))
    pv2 = s.run_op("threshold.apply", {"method": "Current"}, preview=True)["preview"]
    assert np.array_equal(pv1, pv2) and int((pv1 == 255).sum()) == 31
    s.run_op("threshold.apply", {"method": "Current"})
    assert g.lut_is_binary is True and int((g.data == 255).sum()) == 31 and g.id not in s.thresholds
    # Make Binary shares the implementation
    h = s.add_image(LumenImage(np.arange(64, dtype=np.uint8).reshape(8, 8), name="ramp2"))
    s.run_op("binary.make", {"method": "Otsu"}, preview=True)
    assert h.lut_is_binary is False


# ====================================================================== Filters
def test_normalize_local_contrast_rof_shape_index(session):
    img = session.active
    orig = img.data.copy()
    session.run_op("filters.normalize_local_contrast", {"block_radius_x": 20, "block_radius_y": 20, "standard_deviations": 3})
    nlc = img.current_plane().astype(float)
    assert 100 < nlc.mean() < 155 and nlc.std() > 20
    img.replace_data(orig.copy())
    session.run_op("filters.rof_denoise", {"theta": 25})
    assert img.dtype == np.uint8 and img.current_plane().std() < orig[0, 0, 0].std()
    assert np.corrcoef(img.current_plane().ravel().astype(float), orig[0, 0, 0].ravel().astype(float))[0, 1] > 0.9
    img.replace_data(orig.copy())
    session.run_op("filters.shape_index", {"sigma": 2})
    assert img.dtype == np.float32 and float(np.nanmin(img.data)) >= -1 and float(np.nanmax(img.data)) <= 1


def test_retinex_rgb_and_gray():
    s = Session()
    s.run_op("file.sample", {"name": "Astronaut (RGB)"})
    rgb = s.active
    s.run_op("process.retinex", {"scale": 60, "scale_division": 2})
    assert rgb.rgb and rgb.dtype == np.uint8 and 80 < rgb.data.mean() < 180
    s.run_op("file.sample", {"name": "Blobs"})
    g = s.active
    before = g.data.copy()
    s.run_op("process.retinex", {"scale": 30})
    assert g.dtype == np.uint8 and not np.array_equal(g.data, before)
    assert np.corrcoef(g.data.ravel().astype(float), before.ravel().astype(float))[0, 1] > 0.5


# ====================================================================== Math ▸ Macro
def test_math_macro(session):
    from lumen.core.ops.process_extra import macro_to_python
    assert macro_to_python("v = 255 * (x < 10 && y < 10 || d < 5)") == [("v", "255 * ((x < 10) & (y < 10) | (d < 5))")]
    assert macro_to_python("v += 10; v = pow(v, 0.5)") == [("v", "(v) + (10)"), ("v", "pow(v, 0.5)")]
    img = session.active
    orig = img.data.copy()
    session.run_op("math.macro", {"code": "v = v*2"})
    assert np.array_equal(img.data, np.minimum(orig.astype(int) * 2, 255).astype(np.uint8))
    img.replace_data(orig.copy())
    session.run_op("math.macro", {"code": "v = 255 * (x < 10 && y < 10 || d < 5)"})
    p = img.current_plane()
    assert p[0, 0] == 255 and p[50, 50] == 0 and p[img.height // 2, img.width // 2] == 255
    img.replace_data(orig.copy())
    session.run_op("math.macro", {"code": "v = v + 50*sin(d/10)"})
    assert img.dtype == np.uint8 and not np.array_equal(img.data, orig)
    img.replace_data(orig.copy())
    session.run_op("math.macro", {"code": "v = !(v > 100) * 255"})
    assert np.array_equal(img.current_plane() > 0, orig[0, 0, 0] <= 100)
    # ROI aware (plane op) and error reporting
    img.replace_data(orig.copy())
    session.run_op("math.macro", {"code": "v = 0"}, roi={"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10})
    assert img.current_plane()[:10, :10].max() == 0 and np.array_equal(img.current_plane()[20:, 20:], orig[0, 0, 0, 20:, 20:])
    with pytest.raises(ValueError, match="Unknown variable"):
        session.run_op("math.macro", {"code": "v = foo(v)"})
    with pytest.raises(ValueError, match="assign"):
        session.run_op("math.macro", {"code": "  "})


def test_math_macro_exponent_literals_and_ast_whitelist(session):
    """Numeric literals that ast.unparse prints in exponent form (2.5e-05, 1e+16) are Constants, not a name 'e';
    attribute access, subscripts and keyword arguments are rejected with a clear message."""
    img = session.add_image(LumenImage(np.full((4, 4), 2.0, np.float32), name="f"))
    for code, expect in (("v = v * 2.5e-5", 5e-5), ("v = v * 1e-7", 2e-7), ("v = v * 1e16", 2e16), ("v = v + 1E20", 1e20 + 2), ("v = v * 1e-3", 2e-3)):
        img.replace_data(np.full((1, 1, 1, 4, 4), 2.0, np.float32))
        session.run_op("math.macro", {"code": code}, image_id=img.id)
        assert float(img.data[0, 0, 0, 0, 0]) == pytest.approx(expect, rel=1e-6), code
    for code, msg in (("v = v.T", "Unsupported syntax"), ("v = v[0]", "Unsupported syntax"), ("v = pow(v, y=2)", "plain function calls"),
                      ("v = 'a'", "Unsupported literal"), ("v = foo(v)", "Unknown variable")):
        with pytest.raises(ValueError, match=msg):
            session.run_op("math.macro", {"code": code}, image_id=img.id)


def test_math_macro_z_is_the_plane_being_processed():
    """A stack run evaluates every plane with its own 0-based z (ImageJ sets z per slice), not the displayed slice;
    slice scope and preview use the displayed plane's index; hyperstacks also expose t and c."""
    s = Session()
    st = s.add_image(LumenImage(np.zeros((4, 8, 8), np.float32), axes="ZYX"))
    st.display.z = 2
    s.run_op("math.macro", {"code": "v = z"}, scope="stack")
    assert [float(st.data[0, z, 0, 0, 0]) for z in range(4)] == [0.0, 1.0, 2.0, 3.0]
    s.run_op("math.macro", {"code": "v = z + 10"}, scope="slice")  # only the displayed slice, with its own z
    assert [float(st.data[0, z, 0, 0, 0]) for z in range(4)] == [0.0, 1.0, 12.0, 3.0]
    pv = s.run_op("math.macro", {"code": "v = z * 100"}, preview=True)["preview"]
    assert float(pv[0, 0, 0]) == 200.0 and [float(st.data[0, z, 0, 0, 0]) for z in range(4)] == [0.0, 1.0, 12.0, 3.0]
    # ROI masking still applies per plane
    s.run_op("math.macro", {"code": "v = -1 - z"}, scope="stack", roi={"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2})
    assert [float(st.data[0, z, 0, 0, 0]) for z in range(4)] == [-1.0, -2.0, -3.0, -4.0]
    assert [float(st.data[0, z, 0, 5, 5]) for z in range(4)] == [0.0, 1.0, 12.0, 3.0]
    hs = s.add_image(LumenImage(np.zeros((2, 3, 2, 4, 4), np.float32), axes="TZCYX"))
    s.run_op("math.macro", {"code": "v = t * 100 + z * 10 + c"}, scope="stack")
    for t in range(2):
        for z in range(3):
            for c in range(2):
                assert float(hs.data[t, z, c, 0, 0]) == t * 100 + z * 10 + c
    # a direct call with a copied plane falls back to the displayed plane
    from lumen.core.ops.process_extra import math_macro
    from lumen.core.registry import OpContext
    st.display.z = 3
    out = math_macro(st.data[0, 1, 0].copy(), OpContext(session=s, image=st), code="v = z")
    assert float(out[0, 0]) == 3.0


# ====================================================================== Stacks
def _hyperstack(session):
    vol = np.zeros((2, 3, 2, 20, 30), np.uint16)
    for t in range(2):
        for z in range(3):
            for c in range(2):
                vol[t, z, c] = (t * 6 + z * 2 + c) * 10
    return session.add_image(LumenImage(vol, name="hs", axes="TZCYX"))


def test_slice_labels():
    s = Session()
    img = _hyperstack(s)
    img.display.t, img.display.z, img.display.c = 1, 2, 1
    s.run_op("stacks.set_slice_label", {"label": "last"})
    labels = img.metadata["slice_labels"]
    assert len(labels) == 12 and labels[11] == "last" and labels[0] == ""
    s.run_op("stacks.slice_labels_to_results")
    assert s.results.rows[-1] == {"Label": "hs", "Index": 12, "Ch": 2, "Slice": 3, "Frame": 2, "Slice Label": "last"}
    s.run_op("stacks.remove_slice_labels")
    assert "slice_labels" not in img.metadata


def test_hyperstack_to_stack_reduce_subset():
    s = Session()
    img = _hyperstack(s)
    out = s.run_op("stacks.hyperstack_to_stack")
    flat = s.images[out["new_images"][0]]
    assert flat.shape == (1, 12, 1, 20, 30)
    assert [int(flat.data[0, i, 0].mean()) for i in range(12)] == [i * 10 for i in range(12)]  # c fastest, then z, then t
    assert flat.metadata["slice_labels"][1] == "c:2/2 z:1/3 t:1/2"
    img.display.t, img.display.z, img.display.c = 1, 2, 1
    out = s.run_op("stacks.reduce_dimensionality", {"keep_channels": True, "keep_slices": False, "keep_frames": False}, image_id=img.id)
    red = s.images[out["new_images"][0]]
    assert red.shape == (1, 1, 2, 20, 30) and [int(red.data[0, 0, c].mean()) for c in range(2)] == [100, 110]
    with pytest.raises(ValueError):
        s.run_op("stacks.reduce_dimensionality", {}, image_id=img.id)
    out = s.run_op("stacks.make_subset", {"channels": "2", "slices": "1-3-2", "frames": ""}, image_id=img.id)
    sub = s.images[out["new_images"][0]]
    assert sub.shape == (2, 2, 1, 20, 30) and [int(sub.data[t, z, 0].mean()) for t in range(2) for z in range(2)] == [10, 50, 70, 110]
    assert sub.display.channels[0].name == img.display.channels[1].name


def test_stack_statistics():
    s = Session()
    img = _hyperstack(s)
    img.calibration.pixel_width = img.calibration.pixel_height = 0.5
    img.calibration.pixel_depth = 2.0
    img.calibration.unit = "um"
    out = s.run_op("stacks.statistics")
    table = out["table"]
    assert table["name"] == "Stack Statistics" and "Stack Statistics" in s.tables and table["columns"][2] == "Volume (um³)"
    row = table["rows"][0]
    assert row[0] == "hs:C1" and row[1] == 3600 and row[2] == pytest.approx(3600 * 0.5)
    assert row[3] == pytest.approx(50.0) and row[5] == 0 and row[6] == 100 and row[8] == pytest.approx(60.0)  # ImageJ histogram median: the upper middle value
    assert s.results.rows[-1]["Mean"] == pytest.approx(60.0)  # channel 2
    out = s.run_op("stacks.statistics", {}, image_id=img.id, roi={"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10})
    assert out["table"]["rows"][0][1] == 600


def test_montage_to_stack_roundtrip():
    s = Session()
    st = s.add_image(LumenImage(np.stack([np.full((10, 12), i, np.uint8) for i in range(6)]), name="st", axes="ZYX"))
    s.run_op("stacks.montage", {"columns": 3, "rows": 2, "border": 2})
    assert s.active.shape[3:] == (22, 40)
    out = s.run_op("stacks.montage_to_stack", {"columns": 3, "rows": 2, "border": 2})
    back = s.images[out["new_images"][0]]
    assert back.shape == (1, 6, 1, 10, 12) and [int(back.data[0, z, 0].mean()) for z in range(6)] == [0, 1, 2, 3, 4, 5]
    assert np.array_equal(back.data, st.data)
    with pytest.raises(ValueError):
        s.run_op("stacks.montage_to_stack", {"columns": 100, "rows": 1})


def test_reslice_line_mode_and_radial():
    s = Session()
    vol = np.zeros((5, 40, 40), np.uint8)
    for z in range(5):
        vol[z] = z * 50
    img = s.add_image(LumenImage(vol, name="v", axes="ZYX"))
    line = {"type": "line", "x1": 5, "y1": 20, "x2": 35, "y2": 20, "width": 1}
    out = s.run_op("stacks.reslice", {}, roi=line)
    rs = s.images[out["new_images"][0]]
    assert rs.shape == (1, 1, 1, 5, 31) and list(rs.data[0, 0, 0, :, 5]) == [0, 50, 100, 150, 200]
    out = s.run_op("stacks.reslice", {"slice_count": 3, "output_spacing": 2, "rotate": True}, image_id=img.id, roi=line)
    rs3 = s.images[out["new_images"][0]]
    assert rs3.shape == (1, 3, 1, 31, 5) and rs3.calibration.pixel_depth == pytest.approx(2.0)
    out = s.run_op("stacks.reslice_line", {}, image_id=img.id, roi={"type": "polyline", "points": [[5, 5], [20, 5], [20, 30]], "width": 1})
    assert s.images[out["new_images"][0]].shape[3] == 5
    out = s.run_op("stacks.reslice", {}, image_id=img.id)  # no selection → orthogonal as before
    assert s.images[out["new_images"][0]].shape == (1, 40, 1, 5, 40)
    out = s.run_op("stacks.radial_reslice", {"n_angles": 4}, image_id=img.id)
    rad = s.images[out["new_images"][0]]
    assert rad.shape == (1, 4, 1, 5, 20) and rad.metadata["slice_labels"] == ["0.0°", "90.0°", "180.0°", "270.0°"]
    assert list(rad.data[0, 1, 0, :, 3]) == [0, 50, 100, 150, 200]
    out = s.run_op("stacks.radial_reslice", {"n_angles": 2, "length": 10}, image_id=img.id, roi={"type": "point", "points": [[20, 20]]})
    assert s.images[out["new_images"][0]].shape == (1, 2, 1, 5, 10)
    with pytest.raises(ValueError):
        s.run_op("stacks.reslice_line", {}, image_id=img.id, roi={"type": "rect", "x": 0, "y": 0, "w": 5, "h": 5})


# ====================================================================== Process ▸ Batch ▸ Measure
def test_batch_measure(tmp_path):
    for i in range(3):
        a = np.zeros((20, 20), np.uint8)
        a[5:15, 5:15] = 100 + i
        tifffile.imwrite(str(tmp_path / f"im{i}.tif"), a)
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    active = s.active_id
    s.measurements = ["area", "mean", "min", "max"]
    out = s.run_op("batch.measure", {"input_folder": str(tmp_path), "threshold": "Otsu"})
    assert out["results_added"] == 3 and "3 ok, 0 failed" in out["message"]
    rows = s.results.rows
    assert [r["File"] for r in rows] == ["im0.tif", "im1.tif", "im2.tif"]
    assert rows[0]["Area"] == 100 and rows[2]["Mean"] == 102 and rows[0]["Min"] == 100   # limited to the thresholded square
    assert len(s.images) == 1 and s.active_id == active and s.macro[-1].startswith('lumen.run("batch.measure"')
    s.results.clear()
    out = s.run_op("batch.measure", {"input_folder": str(tmp_path), "threshold": "None", "max_files": 1})
    assert len(s.results.rows) == 1 and s.results.rows[0]["Area"] == 400 and s.results.rows[0]["Mean"] == pytest.approx(25.0)
    with pytest.raises(FileNotFoundError):
        s.run_op("batch.measure", {"input_folder": str(tmp_path / "missing")})


# ====================================================================== regression tests (code review, fix round 4)
def test_apply_threshold_current_recomputes_per_plane_with_the_stored_method():
    # 'Calculate threshold for each plane' with method Current used to apply the one stored (lower, upper) pair to
    # every plane; ImageJ recomputes per slice with the Threshold dialog's method (its Convert Stack to Binary selects
    # from the level itself, without setAutoThreshold's +1; the legacy ISODATA auto threshold for a manual range)
    from lumen.core.threshold import auto_threshold, legacy_isodata_threshold
    rng = np.random.default_rng(0)
    planes = [rng.integers(0, 101, (32, 32)).astype(np.uint8), rng.integers(150, 251, (32, 32)).astype(np.uint8)]

    def expect(method, dark=True):
        if method == "legacy":
            return [((p >= lo) & (p <= hi)).astype(np.uint8) * 255 for p in planes for (lo, hi) in [legacy_isodata_threshold(p)]]
        return [((p >= lo) & (p <= hi)).astype(np.uint8) * 255 for p in planes for (lo, hi) in [auto_threshold(p, method, None, dark, add_one=False)]]

    s = Session()
    img = s.add_image(LumenImage(np.stack(planes), name="t", axes="ZYX"))
    s.run_op("threshold.compute", {"method": "Otsu"})
    stored = dict(s.thresholds[img.id])
    s.run_op("threshold.apply", {"method": "Current", "per_plane": True}, preview=True)
    assert s.thresholds[img.id] == stored and not img.lut_is_binary  # a preview leaves the session threshold alone
    s.run_op("threshold.apply", {"method": "Current", "per_plane": True})
    assert all(np.array_equal(img.data[0, z, 0], e) for z, e in enumerate(expect("Otsu")))
    assert 0.3 < (img.data[0, 1, 0] == 255).mean() < 0.7  # plane 1 used to come out 100 % white
    # the panel's background sense is kept for the recompute
    s.undo(img.id)
    s.run_op("threshold.compute", {"method": "Otsu", "dark_background": False})
    s.run_op("threshold.apply", {"method": "Current", "per_plane": True})
    assert all(np.array_equal(img.data[0, z, 0], e) for z, e in enumerate(expect("Otsu", dark=False)))
    # a manual range carries no method: per plane falls back to ImageJ's legacy ISODATA auto threshold instead of raising
    s.undo(img.id)
    s.run_op("threshold.set", {"lower": 50, "upper": 255})
    s.run_op("threshold.apply", {"method": "Current", "per_plane": True})
    assert all(np.array_equal(img.data[0, z, 0], e) for z, e in enumerate(expect("legacy")))
    # without per_plane the stored range still applies as one pair to every plane
    s.undo(img.id)
    s.run_op("threshold.set", {"lower": 50, "upper": 255})
    s.run_op("threshold.apply", {"method": "Current"})
    assert (img.data[0, 1, 0] == 255).all() and np.array_equal(img.data[0, 0, 0], (planes[0] >= 50).astype(np.uint8) * 255)


def test_apply_threshold_on_a_multichannel_z_stack_makes_a_grays_mask():
    # the 4-D (T, Z, Y, X) slab handed to binary_image was read as ZCYX, so for Z > 1 the mask was born with Z channels
    # and a Cyan LUT that the reshape + sync_channels(keep_luts) then preserved: the mask rendered cyan instead of white
    from lumen.core.render import render_rgb
    from lumen.core.threshold import auto_threshold
    rng = np.random.default_rng(2)
    s = Session()
    img = s.add_image(LumenImage(rng.integers(0, 1000, (2, 3, 2, 8, 8)).astype(np.uint16), name="hs", axes="TZCYX"))
    img.display.c = 1
    out = s.run_op("threshold.apply", {"method": "Otsu"})
    m = s.images[out["new_images"][0]]
    assert m.shape == (2, 3, 1, 8, 8) and m.nc == 1 and m.lut_is_binary and m.display.mode == "single"
    assert [c.lut for c in m.display.channels] == ["Grays"]
    lo, hi = auto_threshold(img.data[0, 0, 1], "Otsu")
    assert np.array_equal(m.data[:, :, 0], ((img.data[:, :, 1] >= lo) & (img.data[:, :, 1] <= hi)).astype(np.uint8) * 255)
    rgb = render_rgb(m)
    assert set(map(tuple, rgb.reshape(-1, 3))) <= {(0, 0, 0), (255, 255, 255)}  # black / white, no cyan foreground


# ====================================================================== ImageJ-exact filter ports (lumen.core.ops._imagej)
def _crossval_image(name):
    """The deterministic synthetic image of the ImageJ cross-validation harness (tests/crossval/harness.py)."""
    import importlib
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("tests.crossval.harness").IMAGES[name]()[0]


_IJ_POS = [(0, 0), (0, 159), (119, 0), (119, 159), (7, 11), (23, 77), (41, 130), (60, 80), (64, 33), (88, 141), (101, 5), (113, 99)]
#: pixel values ImageJ 1.54p produced at _IJ_POS (copied from the result.tif files of the cross-validation harness)
_IJ_REF = [
    ("blobs8", "filters.median", {"radius": 2},
     [29, 29, 34, 25, 33, 43, 35, 32, 28, 99, 28, 31]),
    ("blobs8", "filters.variance", {"radius": 2},
     [7, 15, 31, 27, 31, 68, 45, 51, 31, 241, 28, 20]),
    ("blobs8", "filters.tophat", {"radius": 10},
     [15, 20, 30, 8, 18, 0, 0, 14, 1, 49, 7, 6]),
    ("blobs8", "filters.gaussian", {"sigma": 6},
     [30, 30, 32, 30, 30, 62, 46, 41, 40, 99, 30, 37]),
    ("blobs16", "filters.gaussian", {"sigma": 2},
     [472, 406, 484, 375, 651, 519, 504, 1393, 495, 1590, 498, 476]),
    ("blobs8", "process.subtract_background", {"radius": 15},
     [4, 0, 6, 0, 7, 0, 0, 4, 0, 57, 0, 0]),
    ("blobs16", "process.subtract_background", {"radius": 15},
     [65, 0, 0, 0, 205, 51, 199, 823, 105, 972, 27, 0]),
    ("blobs8", "process.subtract_background", {"radius": 15, "method": "Sliding paraboloid"},
     [14, 7, 17, 4, 12, 0, 0, 8, 0, 62, 2, 5]),
    ("blobs32", "process.subtract_background", {"radius": 15, "method": "Sliding paraboloid"},
     [-0.05347803235054016, 0.13542327284812927, 0.13182687759399414, 0.1452781707048416, 0.07134366035461426, 0.026163548231124878,
         0.10759963095188141, 0.5401016473770142, 0.13340523838996887, 0.06159326434135437, -0.007134698331356049, 0.04985120892524719]),
    ("blobs8", "process.subtract_background", {"radius": 15, "create_background": True},
     [27, 32, 33, 27, 27, 33, 29, 27, 27, 35, 26, 28]),
    ("blobs16", "process.enhance_contrast", {"saturated": 0.35, "normalize": True},
     [4977, 1163, 3023, 992, 7164, 3814, 6156, 20330, 5164, 20082, 3535, 1922]),
    ("blobs32", "process.enhance_contrast", {"saturated": 0.35, "normalize": True},
     [0.020161645486950874, 0.09350445866584778, 0.07571077346801758, 0.1042691022157669, 0.05750493332743645, 0.04799818992614746,
         0.22262130677700043, 0.83846515417099, 0.11668818444013596, 0.08424197882413864, 0.139637291431427, 0.07191048562526703]),
    ("blobs8", "process.sharpen", {},
     [37, 26, 52, 20, 36, 0, 0, 36, 4, 87, 12, 22]),
    ("blobs16", "process.find_edges", {},
     [527, 685, 491, 412, 682, 278, 301, 1623, 360, 1147, 713, 78]),
    ("blobs8", "filters.convolve", {"kernel": "1 2 1\n2 4 2\n1 2 1", "normalize": True},
     [29, 30, 35, 27, 33, 40, 29, 30, 27, 94, 29, 29]),
    ("blobs8", "filters.unsharp", {"radius": 2, "weight": 0.6},
     [34, 27, 47, 19, 38, 1, 0, 32, 7, 71, 16, 21]),
    ("blobs8", "process.fft_bandpass", {"large": 40, "small": 3, "suppress": "None", "tolerance": 5, "autoscale": True, "saturate": True},
     [59, 62, 48, 45, 63, 39, 36, 16, 28, 103, 40, 14]),
]


@pytest.mark.parametrize("case", _IJ_REF, ids=[f"{c[0]}-{c[1]}" for c in _IJ_REF])
def test_imagej_exact_ports_match_imagej_pixels(case):
    name, op_id, params, expected = case
    s = Session()
    img = s.add_image(LumenImage(_crossval_image(name), name=name))
    s.run_op(op_id, dict(params))
    out = img.current_plane()
    got = [float(out[y, x]) for (y, x) in _IJ_POS]
    assert got == pytest.approx(expected, abs=0.0), (name, op_id, params)


def test_imagej_light_background_subtraction_and_power_spectrum():
    s = Session()
    img = s.add_image(LumenImage((255 - _crossval_image("blobs8")).astype(np.uint8), name="inv"))
    s.run_op("process.subtract_background", {"radius": 30, "light_background": True})
    out = img.current_plane()
    assert [int(out[y, x]) for (y, x) in _IJ_POS] == [251, 255, 249, 255, 248, 255, 255, 250, 255, 192, 255, 255]
    # Process ▸ FFT: 160×120 padded to 256×256 with the (truncated) mean, float32 Hartley transform, 1…254 log scale
    src = s.add_image(LumenImage(_crossval_image("blobs8"), name="blobs8"))
    out = s.run_op("process.fft", {}, image_id=src.id)
    ps = s.images[out["new_images"][0]].current_plane()
    assert ps.shape == (256, 256) and ps.dtype == np.uint8
    pos = [(128, 128), (0, 0), (128, 129), (100, 140), (64, 64), (200, 30), (255, 255), (128, 0), (0, 128), (130, 126), (90, 90), (250, 12)]
    assert [int(ps[y, x]) for (y, x) in pos] == [254, 77, 186, 128, 78, 80, 77, 45, 94, 180, 78, 80]
    s.run_op("process.inverse_fft", {}, image_id=out["new_images"][0])
    assert np.array_equal(s.active.current_plane(), src.current_plane())


def test_imagej_port_details():
    from lumen.core.ops import _imagej as IJ
    # RankFilters.makeLineRadii: (int)(r²)+1, with the legacy radius remapping
    assert [int(IJ.rank_kernel(r).sum()) for r in (0.5, 1, 1.5, 2, 2.5, 3, 4)] == [5, 9, 13, 21, 29, 37, 57]
    # 3×3 filters: ImageJ's integer arithmetic ((sum+4)/9, (sum+2)/4 truncating toward zero, truncated Sobel)
    a = np.array([[0, 0, 0], [0, 9, 0], [0, 0, 0]], np.uint8)
    assert IJ.smooth3x3(a)[1, 1] == (9 + 4) // 9 and IJ.smooth3x3(np.full((3, 3), 255, np.uint8))[0, 0] == 255
    b = np.array([[10, 10, 10], [10, 20, 10], [10, 10, 10]], np.uint8)
    assert IJ.sharpen3x3(b)[1, 1] == (12 * 20 - 8 * 10 + 2) // 4 and IJ.sharpen3x3(np.array([[50, 0, 50]] * 3, np.uint8))[1, 1] == 0
    e = np.array([[0, 0, 0], [0, 0, 0], [3, 3, 3]], np.uint8)
    assert IJ.find_edges3x3(e)[1, 1] == int(np.sqrt((3 + 6 + 3) ** 2))
    f = e.astype(np.float32)
    assert IJ.find_edges3x3(f).dtype == np.float32 and IJ.find_edges3x3(f)[1, 1] == np.float32(12.0)
    # Convolver: products in float32, double sum, (int)(v + 0.5f) for 8-bit (half up at exact .5)
    c = np.zeros((3, 3), np.uint8)
    c[1, 1] = 8                                             # 8·4/16 = 2.0, 8·2/16 = 1.0, 8·1/16 = 0.5 → 1 (half up)
    out = IJ.convolve(c, np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32), True)
    assert out[1, 1] == 2 and out[0, 1] == 1 and out[0, 0] == 1
    # GaussianBlur.makeGaussianKernel: radius ceil(σ·sqrt(−2 ln acc)) + 1, normalised, running sums end near 0
    k, ks = IJ.make_gaussian_kernel(2.0, 0.002, 160)
    assert k.shape[0] == 9 and abs(float(k[0]) + 2 * float(k[1:].sum()) - 1.0) < 1e-6 and abs(float(ks[-1])) < 1e-6
    assert IJ.gaussian_blur(np.full((20, 30), 77, np.uint8), 2.0, 2.0).tolist() == np.full((20, 30), 77).tolist()
    big = IJ.gaussian_blur(np.full((40, 50), 1000, np.uint16), 12.0, 12.0)          # downscaled path (σ > 8.5)
    assert big.dtype == np.uint16 and np.abs(big.astype(int) - 1000).max() <= 1
    # ContrastEnhancer: 16-bit limits from the full histogram (integers), 32-bit normalize clipped to 0…1
    v16 = np.arange(1000, 1000 + 4096, dtype=np.uint16).reshape(64, 64)
    lo, hi = IJ.contrast_limits(v16, np.uint16, 0.35)
    assert lo == 1000 + int(4096 * 0.35 / 200) and hi == 1000 + 4095 - int(4096 * 0.35 / 200)
    nf = IJ.normalize_plane(np.linspace(-1, 2, 100, dtype=np.float32).reshape(10, 10), 0.0, 1.0)
    assert nf.min() == 0.0 and nf.max() == 1.0 and nf.dtype == np.float32
    assert IJ.contrast_limits(np.full(50, 7, np.uint8), np.uint8, 0.35) is None       # hmax ≤ hmin → ImageJ does nothing
    # BackgroundSubtracter: ball shrink factors / patch widths and the FFT padding rule
    assert [(IJ.RollingBall(r).shrink, IJ.RollingBall(r).width) for r in (5, 15, 50, 150)] == [(1, 9), (2, 15), (4, 19), (8, 25)]
    assert IJ.fft_pad_size(160, 120) == 256 and IJ.fft_pad_size(256, 256) == 256 and IJ.fft_pad_size(257, 10) == 512
    padded = IJ.fft_pad(np.array([[1, 2], [3, 4]], np.uint8).repeat(2, 0).repeat(3, 1))   # 4×6 → 8×8 filled with (int) mean
    assert padded.shape == (8, 8) and padded[7, 7] == 2 and padded[0, 0] == 1
