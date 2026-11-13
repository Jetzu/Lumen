"""Gaussian Blur 'Scaled units' (ImageJ GaussianBlur: sigmaX = sigma / pixelWidth, sigmaY = sigma / pixelHeight)."""
import numpy as np

import lumen.core.ops  # noqa: F401  (registers the commands)

from lumen.core.ij_compat import IJ
from lumen.core.ij_export import recorder_to_ijm
from lumen.core.session import Session


def _session(pw=0.5, ph=0.25):
    s = Session()
    s.run_op("file.new", {"title": "g", "type": "8-bit", "width": 48, "height": 40, "fill": "Ramp"})
    s.run_op("image.properties", {"pixel_width": pw, "pixel_height": ph, "unit": "um"})
    return s


def test_scaled_sigma_divides_by_the_pixel_size():
    a = _session()
    a.run_op("filters.gaussian", {"sigma": 1.0, "scaled_units": True})
    b = Session()
    b.run_op("file.new", {"title": "g", "type": "8-bit", "width": 48, "height": 40, "fill": "Ramp"})
    # the same blur in pixels: sigma 2 along X (1 / 0.5) and 4 along Y (1 / 0.25) – anisotropic, so neither plain sigma matches
    from lumen.core.ops._imagej import gaussian_blur as ij_gaussian_blur
    expected = ij_gaussian_blur(b.active.data[0, 0, 0].copy(), 2.0, 4.0)
    np.testing.assert_array_equal(a.active.data[0, 0, 0], expected)
    c = _session()
    c.run_op("filters.gaussian", {"sigma": 1.0})
    assert not np.array_equal(c.active.data[0, 0, 0], a.active.data[0, 0, 0])  # unscaled is a different (isotropic) blur


def test_macro_flag_and_export_round_trip():
    a = _session()
    IJ(a).macro('run("Gaussian Blur...", "sigma=1 scaled");')
    b = _session()
    b.run_op("filters.gaussian", {"sigma": 1.0, "scaled_units": True})
    np.testing.assert_array_equal(a.active.data, b.active.data)
    out = recorder_to_ijm([ln for ln in b.macro if 'filters.gaussian' in ln])
    assert out["lines"] == ['run("Gaussian Blur...", "sigma=1 scaled stack");']
    plain = recorder_to_ijm(['lumen.run("filters.gaussian", sigma=3.0, scaled_units=False)'])
    assert plain["lines"] == ['run("Gaussian Blur...", "sigma=3 stack");']
