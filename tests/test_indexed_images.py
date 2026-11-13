"""Palette (indexed) GIF / PNG images open like ImageJ opens them: 8-bit indices + the palette as the LUT (E11-1)."""
import numpy as np
from PIL import Image

import lumen.core.ops  # noqa: F401
from lumen.core.io import read_image
from lumen.core.session import Session


def _indexed(path, idx, palette):
    im = Image.fromarray(idx.astype(np.uint8), "P")
    im.putpalette(palette.astype(np.uint8).ravel().tolist())
    im.save(path, optimize=False)  # keep the palette as written (PIL would compact unused entries and shift the indices)


def test_inverted_grey_palette_keeps_the_indices_like_imagej(tmp_path):
    """blobs.gif is stored with a reversed grey palette: ImageJ measures the raw indices (mean 103.27 for Blobs)."""
    rng = np.random.default_rng(0)
    idx = rng.integers(8, 248, size=(40, 50)).astype(np.uint8)
    pal = np.repeat((255 - np.arange(256))[:, None], 3, axis=1)
    _indexed(tmp_path / "blobs_like.gif", idx, pal)
    img = read_image(str(tmp_path / "blobs_like.gif"))
    assert not img.rgb and img.dtype == np.uint8 and img.data.shape[-2:] == (40, 50)
    np.testing.assert_array_equal(img.data[0, 0, 0], idx)
    assert img.display.channels[0].inverted and img.metadata.get("inverting_lut")
    s = Session()
    s.add_image(img)
    s.run_op("analyze.measure", {})
    assert abs(s.results.rows[-1]["Mean"] - float(idx.mean())) < 1e-3  # the indices, not the displayed (inverted) grey


def test_plain_grey_and_colour_palettes(tmp_path):
    idx = np.tile(np.arange(64, dtype=np.uint8), (10, 1))
    _indexed(tmp_path / "grey.png", idx, np.repeat(np.arange(256)[:, None], 3, axis=1))
    g = read_image(str(tmp_path / "grey.png"))
    assert not g.rgb and not g.display.channels[0].inverted and g.display.channels[0].lut == "Grays"
    np.testing.assert_array_equal(g.data[0, 0, 0], idx)
    pal = np.zeros((256, 3), np.uint8)
    pal[:, 0] = np.arange(256)  # a red ramp
    _indexed(tmp_path / "red.gif", idx, pal)
    r = read_image(str(tmp_path / "red.gif"))
    assert not r.rgb and r.display.channels[0].lut.startswith("red palette")
    from lumen.core import luts
    np.testing.assert_array_equal(luts.get_lut(r.display.channels[0].lut)[:, 0], np.arange(256))


def test_rgb_and_animated_gifs_are_unchanged(tmp_path):
    rgb = np.zeros((12, 12, 3), np.uint8)
    rgb[..., 1] = 200
    Image.fromarray(rgb, "RGB").save(tmp_path / "rgb.png")
    assert read_image(str(tmp_path / "rgb.png")).rgb
    frames = [Image.fromarray(np.full((8, 8, 3), v, np.uint8), "RGB") for v in (10, 120, 240)]
    frames[0].save(tmp_path / "anim.gif", save_all=True, append_images=frames[1:])
    a = read_image(str(tmp_path / "anim.gif"))
    assert a.nt == 3
