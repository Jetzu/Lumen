"""Image ▸ Adjust ▸ Bleach Correction (lumen/core/ops/bleach.py): the three CorrectBleach methods, the macro mapping and
the export (E7-10)."""
from __future__ import annotations

import numpy as np
import pytest

from lumen.core import ops  # noqa: F401
from lumen.core.ij_compat import IJ
from lumen.core.ij_export import recorder_to_ijm
from lumen.core.image import LumenImage
from lumen.core.ops.bleach import METHODS, NAME_PREFIX, TABLE_NAME, fit_exponential
from lumen.core.registry import REGISTRY
from lumen.core.session import Session

T = 10


def _series(seed: int = 0, size: int = 64, nc: int = 1, nz: int = 1) -> np.ndarray:
    """``I_t = 100·exp(−0.1 t) + 10`` plus noise, (T, Z, C, Y, X) uint16."""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float64)
    base = 100.0 * np.exp(-0.1 * t) + 10.0
    data = np.empty((T, nz, nc, size, size), np.float64)
    for c in range(nc):
        for z in range(nz):
            data[:, z, c] = (base * (1 + 0.5 * c))[:, None, None] + rng.normal(0, 2.0, (T, size, size))
    return np.clip(data, 0, None).astype(np.uint16)


def _means(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(arr.shape[0], -1).mean(axis=1)


@pytest.fixture
def session():
    s = Session()
    s.add_image(LumenImage(_series(), name="series.tif", axes="TZCYX"))
    return s


def test_simple_ratio_and_exponential_fit_flatten_the_frame_means(session):
    s = session
    src = s.active
    before = _means(src.data[:, 0, 0])
    assert before[0] > 1.9 * before[-1]  # the series really bleaches
    for method in ("Simple Ratio", "Exponential Fit"):
        out = s.run_op("adjust.bleach_correction", {"method": method}, image_id=src.id)
        res = s.active
        assert res.name.startswith(NAME_PREFIX + "series.tif") and res.id in out["new_images"] and res.data.dtype == np.float32
        assert res.shape == src.shape and res.metadata["derived_from"].startswith("series.tif ▸ Bleach Correction... (")
        after = _means(res.data[:, 0, 0])
        assert np.all(np.abs(after - before[0]) <= 0.02 * before[0]), (method, after)
        table = out["table"]
        assert table["name"] == TABLE_NAME and table["columns"] == ["Frame", "Mean before", "Mean after", "Ratio"]
        assert [r[0] for r in table["rows"]] == list(range(1, T + 1))
        assert table["rows"][0][1] == pytest.approx(before[0]) and table["rows"][-1][1] == pytest.approx(before[-1])
        assert table["rows"][0][3] == pytest.approx(1.0) and table["rows"][-1][3] > 1.9
        assert table["summary"]["method"] == method and table["summary"]["time_axis"] == "T" and table["summary"]["frames"] == T
        plot = out["plot"]
        assert plot["x"] == list(range(1, T + 1)) and [p["name"] for p in plot["series"]][:2] == ["Mean before", "Mean after"]
        assert plot["series"][0]["values"] == pytest.approx(before.tolist())
        assert "Bleach Correction (" + method in out["message"]
        if method == "Exponential Fit":
            fit = table["summary"]["fit"]["C1"]
            assert fit["a"] == pytest.approx(100.0, rel=0.05) and fit["b"] == pytest.approx(0.1, rel=0.1) and fit["c"] == pytest.approx(10.0, abs=2.0)
            assert fit["half_life_frames"] == pytest.approx(np.log(2) / fit["b"])
            assert [p["name"] for p in plot["series"]] == ["Mean before", "Mean after", "Fit"]
            assert "exp(" in out["message"] and "half-life" in out["message"]
    assert len(s.images) == 3 and src.data.dtype == np.uint16  # the source is untouched (the plugin corrects a duplicate)


def test_fit_exponential_recovers_the_decay():
    t = np.arange(T)
    a, b, c = fit_exponential(t, 100.0 * np.exp(-0.1 * t) + 10.0)
    assert a == pytest.approx(100.0, rel=1e-3) and b == pytest.approx(0.1, rel=1e-3) and c == pytest.approx(10.0, abs=0.05)


def test_background_keep_type_roi_and_channels(session):
    s = session
    src = s.active
    before = _means(src.data[:, 0, 0])
    # the background enters the ratios: (m0 − bg) / (mt − bg) with bg = 10 is exactly the decay term
    out = s.run_op("adjust.bleach_correction", {"method": "Simple Ratio", "background": 10.0}, image_id=src.id)
    ratios = [r[3] for r in out["table"]["rows"]]
    assert ratios[-1] == pytest.approx((before[0] - 10) / (before[-1] - 10)) and ratios[-1] > (before[0] / before[-1])
    assert s.active.metadata["bleach_correction"] == "Simple Ratio (background 10)"
    # keep_type: rounded and clipped back to 16-bit, means still flat
    out = s.run_op("adjust.bleach_correction", {"method": "Exponential Fit", "keep_type": True}, image_id=src.id)
    res = s.active
    assert res.data.dtype == np.uint16 and np.all(np.abs(_means(res.data[:, 0, 0]) - before[0]) <= 0.02 * before[0])
    # an area selection: the means are taken inside it (the plugin's ROI behaviour) and the whole frame is scaled
    roi = {"type": "rect", "x": 0, "y": 0, "w": 16, "h": 16}
    out = s.run_op("adjust.bleach_correction", {"method": "Simple Ratio"}, image_id=src.id, roi=roi)
    inside = _means(src.data[:, 0, 0, :16, :16])
    assert out["table"]["rows"][-1][1] == pytest.approx(inside[-1]) and out["table"]["rows"][-1][3] == pytest.approx(inside[0] / inside[-1])
    # channels are corrected independently (the table gets a Channel column)
    s.add_image(LumenImage(_series(nc=2), name="two.tif", axes="TZCYX"))
    two = s.active
    out = s.run_op("adjust.bleach_correction", {"method": "Simple Ratio"}, image_id=two.id)
    res = s.active
    assert out["table"]["columns"] == ["Frame", "Channel", "Mean before", "Mean after", "Ratio"] and len(out["table"]["rows"]) == 2 * T
    for c in range(2):
        m = _means(res.data[:, 0, c])
        assert np.all(np.abs(m - m[0]) <= 0.02 * m[0])
    assert _means(res.data[:, 0, 1])[0] == pytest.approx(1.5 * _means(res.data[:, 0, 0])[0], rel=0.02)
    assert [p["name"] for p in out["plot"]["series"]] == ["C1 Mean before", "C1 Mean after", "C2 Mean before", "C2 Mean after"]


def test_histogram_matching_matches_frame_0(session):
    s = session
    src = s.active
    out = s.run_op("adjust.bleach_correction", {"method": "Histogram Matching"}, image_id=src.id)
    res = s.active
    assert res.data.dtype == np.uint16  # the type is kept
    f0, f5 = res.data[0, 0, 0].astype(np.float64), res.data[5, 0, 0].astype(np.float64)
    before5 = src.data[5, 0, 0].astype(np.float64)
    assert before5.mean() < 0.7 * f0.mean()  # frame 6 was dim before
    # equal-size images with matched histograms have the same sorted pixel values (up to the rounding to 16-bit)
    assert np.abs(np.sort(f5.ravel()) - np.sort(f0.ravel())).max() <= 1
    assert f5.mean() == pytest.approx(f0.mean(), rel=0.01) and f5.std() == pytest.approx(f0.std(), rel=0.05)
    assert np.percentile(f5, [5, 50, 95]) == pytest.approx(np.percentile(f0, [5, 50, 95]), abs=1.0)
    assert (f5.min(), f5.max()) == pytest.approx((f0.min(), f0.max()), abs=1.0)
    assert np.array_equal(res.data[0], src.data[0])  # frame 1 is the reference and stays as it is
    assert all(r[3] == "" for r in out["table"]["rows"]) and "Histogram Matching" in out["message"]


def test_plain_stack_uses_slices_as_time_and_errors_are_clear():
    s = Session()
    zstack = np.moveaxis(_series(), 0, 1)  # (1, T, 1, Y, X): no frames, T slices
    s.add_image(LumenImage(zstack, name="stack.tif", axes="TZCYX"))
    src = s.active
    out = s.run_op("adjust.bleach_correction", {"method": "Simple Ratio"})
    res = s.active
    assert res.nz == T and res.nt == 1 and "slices treated as time points" in out["message"] and out["table"]["summary"]["time_axis"] == "Z"
    m = _means(np.moveaxis(res.data, 1, 0)[:, 0, 0])
    assert np.all(np.abs(m - m[0]) <= 0.02 * m[0])
    s.add_image(LumenImage(np.zeros((8, 8), np.uint8), name="single"))
    with pytest.raises(ValueError, match="needs a time series"):
        s.run_op("adjust.bleach_correction", {})
    s.add_image(LumenImage(np.zeros((8, 8, 3), np.uint8), name="rgb", rgb=True))
    with pytest.raises(ValueError, match="RGB images"):  # supports_rgb=False: the registry refuses it before the op runs
        s.run_op("adjust.bleach_correction", {})
    s.add_image(LumenImage(np.zeros((3, 8, 8), np.uint8), name="dark", axes="ZYX"))
    with pytest.raises(ValueError, match="not above the background"):
        s.run_op("adjust.bleach_correction", {"method": "Simple Ratio"})
    with pytest.raises(ValueError, match="is not one of"):
        s.run_op("adjust.bleach_correction", {"method": "Linear"}, image_id=src.id)


def test_palette_keywords_macro_mapping_and_export(session):
    spec = REGISTRY.ops["adjust.bleach_correction"]
    assert spec.name == "Bleach Correction..." and spec.category == "Image/Adjust" and spec.kind == "image"
    for kw in ("bleach", "photobleaching", "decay"):
        assert kw in spec.keywords
    assert [p.name for p in spec.params] == ["method", "background", "keep_type"] and spec.params[0].choices == METHODS
    assert spec.to_dict()["ij_commands"] == ["Bleach Correction"]
    s = session
    src = s.active
    ij = IJ(s)
    ij.run("Bleach Correction", "correction=[Exponential Fit] background=0")
    res = s.active
    assert res.name == NAME_PREFIX + "series.tif" and res.data.dtype == np.float32
    m = _means(res.data[:, 0, 0])
    assert np.all(np.abs(m - m[0]) <= 0.02 * m[0])
    ij.run("Bleach Correction", "correction=[Histogram Matching]")  # on the corrected float image: histogram matching keeps float
    assert s.active.name == NAME_PREFIX + NAME_PREFIX + "series.tif" and s.active.data.dtype == np.float32
    # recorded lines export as the plugin's own option string
    s.run_op("adjust.bleach_correction", {"method": "Simple Ratio", "background": 10.0}, image_id=src.id)
    s.run_op("adjust.bleach_correction", {"method": "Histogram Matching"}, image_id=src.id)
    exported = recorder_to_ijm(list(s.macro)[-4:], titles={i.id: i.name for i in s.images.values()})  # select + run, twice
    assert exported["skipped"] == []
    runs = [ln for ln in exported["lines"] if ln.startswith("run(")]
    assert len(runs) == 2 and exported["lines"][0] == 'selectWindow("series.tif");'
    assert runs[0].startswith('run("Bleach Correction", "correction=[Simple Ratio] background=10"); // → "Bleach_corrected_series.tif')
    assert runs[1].startswith('run("Bleach Correction", "correction=[Histogram Matching] background=0"); // → "Bleach_corrected_series.tif')
