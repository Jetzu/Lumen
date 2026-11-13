"""Tests for the built-in LUT set (lumen/core/luts.py): parity with Fiji's luts/ folder and ImageJ's LutLoader."""
import colorsys
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lumen.core import luts
from lumen.core.session import Session
from lumen.server.app import create_app

# Fiji luts/ folder, as Image ▸ Lookup Tables names them (file name without extension)
FIJI_NAMES = ["Cyan Hot", "Green Fire Blue", "Magenta Hot", "Orange Hot", "Red Hot", "Yellow Hot", "Rainbow RGB", "Royal", "Sepia",
              "Thal", "Thallium", "Union Jack", "ICA", "ICA2", "ICA3", "brgbcmyw", "5 ramps", "6 shades", "blue orange icb", "cool",
              "edges", "gem", "glow", "physics", "smart", "mpl-inferno", "mpl-magma", "mpl-plasma", "mpl-viridis",
              "16 colors", "Thermal", "Phase", "Glasbey", "Glasbey on dark", "Glasbey inverted", "Random"]
# ImageJ core (LutLoader) / matplotlib tables
CORE_NAMES = ["Grays", "Fire", "Ice", "Spectrum", "3-3-2 RGB", "Red/Green", "HiLo",
              "Cividis", "Turbo", "Jet", "Viridis", "Magma", "Inferno", "Plasma"]
# Fiji luts/ file name → Lumen name, for the byte-for-byte comparison against a local Fiji install
FIJI_DIR = Path("/Applications/Fiji/luts")
FIJI_FILES = {"16_colors": "16 colors", "5_ramps": "5 ramps", "6_shades": "6 shades", "blue_orange_icb": "blue orange icb",
              "glasbey": "Glasbey", "glasbey_inverted": "Glasbey inverted", "glasbey_on_dark": "Glasbey on dark", "phase": "Phase",
              "royal": "Royal", "sepia": "Sepia", "thal": "Thal", "thallium": "Thallium", "unionjack": "Union Jack"}
HOT = ["Red Hot", "Orange Hot", "Yellow Hot", "Cyan Hot", "Magenta Hot"]
SEQUENTIAL = HOT + ["Green Fire Blue", "Sepia", "Thallium", "Grays"]  # luminance never decreases
SEQUENTIAL_TOL = ["mpl-inferno", "mpl-magma", "mpl-plasma", "mpl-viridis", "gem", "cool", "blue orange icb", "Rainbow RGB", "Fire"]  # up to 8-bit rounding
LABELS = ["Glasbey", "Glasbey on dark", "Glasbey inverted", "Random"]


def _luminance(table: np.ndarray) -> np.ndarray:
    t = table.astype(np.float64)
    return 0.299 * t[:, 0] + 0.587 * t[:, 1] + 0.114 * t[:, 2]


# ---------------------------------------------------------------- registry
def test_fiji_names_are_built_in():
    for name in FIJI_NAMES + CORE_NAMES:
        assert name in luts.LUT_NAMES, name
        assert name in luts.BUILTIN_NAMES and luts.is_builtin(name), name
        assert not luts.unregister_lut(name), name  # built-ins cannot be removed
    assert len(luts.LUT_NAMES) == len(set(luts.LUT_NAMES))
    assert luts.LUT_NAMES[0] == "Grays"  # the existing order is kept, Fiji's tables are appended
    assert luts.LUT_NAMES.index("Red Hot") > luts.LUT_NAMES.index("Jet")
    with pytest.raises(KeyError):
        luts.get_lut("Union_Jack")


def test_every_table_is_256x3_uint8_and_exports_as_imagej_lut(tmp_path):
    serialised = luts.all_luts()
    for name in luts.LUT_NAMES:
        t = luts.get_lut(name)
        assert t.shape == (256, 3) and t.dtype == np.uint8, name
        assert serialised[name] == t.tolist(), name
        raw = luts.lut_bytes(t)
        assert len(raw) == 768 and np.array_equal(luts.parse_lut_bytes(raw), t), name
        assert raw[:256] == t[:, 0].tobytes() and raw[512:] == t[:, 2].tobytes(), name  # R plane, G plane, B plane
    p = luts.write_lut_file(luts.get_lut("Green Fire Blue"), tmp_path / "gfb")
    assert p.endswith("gfb.lut") and np.array_equal(luts.read_lut_file(p), luts.get_lut("Green Fire Blue"))


# ---------------------------------------------------------------- fidelity to Fiji's files
def test_hot_family_runs_from_black_to_white():
    for name in HOT + ["Green Fire Blue", "Sepia"]:
        t = luts.get_lut(name)
        assert t[0].tolist() == [0, 0, 0], name
        assert t[255].tolist() == [255, 255, 255], name
    # the colour that gives the family its name saturates first: red before green, cyan (g+b) before red, …
    assert luts.get_lut("Red Hot")[96].tolist() == [255, 29, 0] and luts.get_lut("Red Hot")[176].tolist() == [255, 255, 32]
    assert luts.get_lut("Cyan Hot")[96].tolist() == [0, 140, 255] and luts.get_lut("Cyan Hot")[176].tolist() == [17, 255, 255]
    assert luts.get_lut("Magenta Hot")[128][1] < luts.get_lut("Magenta Hot")[128][0] == 255
    assert luts.get_lut("Yellow Hot")[128][2] < luts.get_lut("Yellow Hot")[128][1]
    assert luts.get_lut("Orange Hot")[128][0] == 255 and luts.get_lut("Orange Hot")[128][2] == 0


def test_exact_fiji_values():
    """Spot checks against the entries of the original .lut files."""
    g = luts.get_lut
    assert g("Rainbow RGB")[0].tolist() == [0, 0, 0] and g("Rainbow RGB")[255].tolist() == [151, 143, 146]
    assert g("Union Jack")[63].tolist() == [0, 0, 190] and g("Union Jack")[127].tolist() == [255, 255, 255] and g("Union Jack")[191].tolist() == [190, 0, 0]
    assert g("5 ramps")[50].tolist() == [0, 0, 250] and g("5 ramps")[101].tolist() == [0, 250, 0] and g("5 ramps")[152].tolist() == [250, 250, 0]
    assert g("5 ramps")[203].tolist() == [250, 0, 0] and g("5 ramps")[255].tolist() == [255, 255, 255]
    assert g("6 shades")[64].tolist() == [255, 0, 0] and g("6 shades")[128].tolist() == [0, 0, 255] and g("6 shades")[255].tolist() == [255, 255, 247]
    assert [g("brgbcmyw")[i].tolist() for i in (0, 32, 64, 96, 128, 160, 192, 224)] == [
        [0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 255, 255], [248, 0, 255], [255, 255, 0], [255, 255, 255]]
    assert g("edges")[0].tolist() == [1, 1, 1] and g("edges")[128].tolist() == [253, 253, 253] and g("edges")[255].tolist() == [1, 1, 1]
    assert g("ICA")[0].tolist() == [204, 204, 255] and g("ICA")[255].tolist() == [255, 255, 204]
    assert g("ICA2")[0].tolist() == [255, 255, 0] and g("ICA3")[0].tolist() == [255, 204, 255]
    assert g("physics")[0].tolist() == [47, 2, 119] and g("physics")[255].tolist() == [233, 34, 16]
    assert g("Thallium")[0].tolist() == [1, 1, 1] and g("Thallium")[255].tolist() == [253, 253, 253]
    assert g("Thal")[0].tolist() == [1, 1, 9] and g("Thal")[255].tolist() == [254, 254, 254]
    assert g("Royal")[0].tolist() == [1, 1, 7] and g("Royal")[255].tolist() == [254, 254, 254]
    assert g("gem")[0].tolist() == [1, 2, 2] and g("gem")[255].tolist() == [254, 252, 250]
    assert g("glow")[0].tolist() == [0, 138, 0] and g("glow")[255].tolist() == [0, 0, 255]
    assert g("smart")[0].tolist() == [2, 2, 2] and g("smart")[255].tolist() == [254, 253, 252]
    assert g("cool")[0].tolist() == [1, 3, 2] and g("cool")[128].tolist() == [127, 1, 254] and g("cool")[255].tolist() == [254, 250, 246]
    assert g("blue orange icb")[0].tolist() == [1, 2, 1] and g("blue orange icb")[127].tolist() == [127, 1, 127] and g("blue orange icb")[255].tolist() == [254, 251, 250]
    assert g("Sepia")[1].tolist() == [43, 28, 4] and g("Sepia")[128].tolist() == [187, 157, 128]
    assert g("Glasbey on dark")[255].tolist() == [17, 101, 82] and g("Glasbey inverted")[255].tolist() == [248, 248, 232]
    # 16_colors.lut: 16 bands of 16 – band 13 (208 … 223) is a magenta → pink ramp, band 14 is (222, 180, 222)
    assert g("16 colors")[0].tolist() == [0, 0, 0] and g("16 colors")[16].tolist() == [1, 1, 171] and g("16 colors")[207].tolist() == [245, 0, 0]
    assert g("16 colors")[208].tolist() == [245, 0, 135] and g("16 colors")[223].tolist() == [245, 0, 228]
    assert g("16 colors")[224].tolist() == [222, 180, 222] and g("16 colors")[239].tolist() == [222, 180, 222] and g("16 colors")[255].tolist() == [255, 255, 255]
    assert all(len({tuple(r) for r in g("16 colors")[k:k + 16].tolist()}) == 1 for k in range(0, 256, 16) if k != 208)
    # Thermal.lut / phase.lut / glasbey.lut (not gnuplot2, an HSV sweep or random hues)
    assert g("Thermal")[0].tolist() == [70, 0, 115] and g("Thermal")[128].tolist() == [251, 254, 243] and g("Thermal")[255].tolist() == [255, 2, 2]
    assert g("Phase")[0].tolist() == [0, 0, 252] and g("Phase")[128].tolist() == [200, 200, 200] and g("Phase")[255].tolist() == [252, 0, 0]
    assert g("Glasbey")[0].tolist() == [255, 255, 255] and g("Glasbey")[1].tolist() == [0, 0, 255] and g("Glasbey")[2].tolist() == [255, 0, 0]
    assert g("Glasbey")[3].tolist() == [0, 255, 0] and g("Glasbey")[128].tolist() == [209, 255, 141] and g("Glasbey")[255].tolist() == [189, 117, 101]
    # Fiji's "cool" (NIH Image) is not matplotlib's cool
    assert not np.array_equal(g("cool"), g("Cool"))


def test_mpl_names_alias_the_matplotlib_tables():
    for fiji, ours in (("mpl-inferno", "Inferno"), ("mpl-magma", "Magma"), ("mpl-plasma", "Plasma"), ("mpl-viridis", "Viridis")):
        # the same 256 matplotlib samples: Fiji's files truncate, Lumen's own names round – never more than one level apart
        d = np.abs(luts.get_lut(fiji).astype(int) - luts.get_lut(ours).astype(int))
        assert d.max() <= 1 and (luts.get_lut(fiji).astype(int) <= luts.get_lut(ours).astype(int)).all(), fiji
    assert luts.get_lut("mpl-viridis")[0].tolist() == [68, 1, 84] and luts.get_lut("mpl-viridis")[255].tolist() == [253, 231, 36]
    assert luts.get_lut("mpl-inferno")[128].tolist() == [187, 55, 84] and luts.get_lut("mpl-magma")[128].tolist() == [182, 54, 121]
    assert luts.get_lut("mpl-plasma")[128].tolist() == [203, 71, 119]


@pytest.mark.skipif(not FIJI_DIR.is_dir(), reason="no local Fiji install to compare with")
def test_every_fiji_lut_file_is_reproduced_byte_for_byte():
    """Every file of Fiji's luts/ folder (36 in Fiji 2.x) decodes – through Lumen's own .lut reader, i.e. exactly as
    ij.plugin.LutLoader reads it – to the table Lumen registers under the Image ▸ Lookup Tables name."""
    files = sorted(FIJI_DIR.glob("*.lut"))
    assert len(files) >= 30
    for path in files:
        name = FIJI_FILES.get(path.stem, path.stem)
        assert name in luts.BUILTIN_NAMES, path.name
        assert np.array_equal(luts.read_lut_file(path), luts.get_lut(name)), path.name


def _lutloader_interpolate(r, g, b):
    """ij.plugin.LutLoader.interpolate(reds, greens, blues, nColors): scale = nColors / 256, i1 = (int)(i * scale), i2 = i1 + 1
    clamped to the last knot, (byte) truncation – transcribed scalar for scalar from the Java source."""
    n = len(r)
    scale = n / 256.0
    out = np.zeros((256, 3), np.uint8)
    for i in range(256):
        i1 = int(i * scale)
        i2 = min(i1 + 1, n - 1)
        frac = i * scale - i1
        for c, src in enumerate((r, g, b)):
            out[i, c] = int((1 - frac) * src[i1] + frac * src[i2])
    return out


def _java_hsb_to_rgb(hue: float) -> list[int]:
    """java.awt.Color.HSBtoRGB(hue, 1f, 1f) in float32 arithmetic, as LutLoader.spectrum() calls it (Color.getHSBColor)."""
    f32 = np.float32
    h = (f32(hue) - f32(np.floor(f32(hue)))) * f32(6)
    f = h - f32(np.floor(h))
    p, q, t, v = f32(0), f32(1) - f, f, f32(1)
    tri = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][int(h)]
    return [int(c * f32(255) + f32(0.5)) for c in tri]


def test_imagej_core_tables_match_lutloader():
    """Fire / Ice / Spectrum / Red/Green / 3-3-2 RGB / Grays are byte-identical to ij.plugin.LutLoader (ImageJ 1.54p):
    the algorithms are re-implemented here from the Java source, the spot values come from a dump of the real jar."""
    g = luts.get_lut
    fire_r = [0, 0, 1, 25, 49, 73, 98, 122, 146, 162, 173, 184, 195, 207, 217, 229, 240, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
    fire_g = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 35, 57, 79, 101, 117, 133, 147, 161, 175, 190, 205, 219, 234, 248, 255, 255, 255, 255]
    fire_b = [0, 61, 96, 130, 165, 192, 220, 227, 210, 181, 151, 122, 93, 64, 35, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 98, 160, 223, 255]
    ice_r = [0, 0, 0, 0, 0, 0, 19, 29, 50, 48, 79, 112, 134, 158, 186, 201, 217, 229, 242, 250, 250, 250, 250, 251, 250, 250, 250, 250, 251, 251, 243, 230]
    ice_g = [156, 165, 176, 184, 190, 196, 193, 184, 171, 162, 146, 125, 107, 93, 81, 87, 92, 97, 95, 93, 93, 90, 85, 69, 64, 54, 47, 35, 19, 0, 4, 0]
    ice_b = [140, 147, 158, 166, 170, 176, 209, 220, 234, 225, 236, 246, 250, 251, 250, 250, 245, 230, 230, 222, 202, 180, 163, 142, 123, 114, 106, 94, 84, 64, 26, 27]
    assert np.array_equal(g("Fire"), _lutloader_interpolate(fire_r, fire_g, fire_b))
    assert np.array_equal(g("Ice"), _lutloader_interpolate(ice_r, ice_g, ice_b))
    # knot k sits exactly at index 8k, the last knot is held flat over 248 … 255 (values from the jar)
    assert g("Fire")[8].tolist() == [0, 0, 61] and g("Fire")[239].tolist() == [255, 255, 215] and g("Fire")[255].tolist() == [255, 255, 255]
    assert len({tuple(r) for r in g("Fire")[248:].tolist()}) == 1 and len({tuple(r) for r in g("Ice")[248:].tolist()}) == 1
    assert g("Ice")[0].tolist() == [0, 156, 140] and g("Ice")[239].tolist() == [244, 3, 30] and g("Ice")[255].tolist() == [230, 0, 27]
    # spectrum(): Color.getHSBColor(i/255f, 1f, 1f) rounds
    assert g("Spectrum").tolist() == [_java_hsb_to_rgb(i / 255.0) for i in range(256)]
    assert g("Spectrum").tolist() == [[int(v * 255 + 0.5) for v in colorsys.hsv_to_rgb(i / 255.0, 1, 1)] for i in range(256)]
    assert g("Spectrum")[3].tolist() == [255, 18, 0] and g("Spectrum")[128].tolist() == [0, 252, 255] and g("Spectrum")[254].tolist() == [255, 0, 6]
    assert g("Spectrum")[0].tolist() == [255, 0, 0] and g("Spectrum")[255].tolist() == [255, 0, 0]  # hue 1.0 wraps back to red
    # redGreen(): reds[i] = i*2 for i < 128, greens[i] = (byte)(i*2) for i >= 128 – black → red, then black → green
    assert g("Red/Green")[0].tolist() == [0, 0, 0] and g("Red/Green")[1].tolist() == [2, 0, 0] and g("Red/Green")[8].tolist() == [16, 0, 0]
    assert g("Red/Green")[64].tolist() == [128, 0, 0] and g("Red/Green")[127].tolist() == [254, 0, 0]
    assert g("Red/Green")[128].tolist() == [0, 0, 0] and g("Red/Green")[192].tolist() == [0, 128, 0] and g("Red/Green")[255].tolist() == [0, 254, 0]
    assert g("Red/Green")[:128, 1:].max() == 0 and g("Red/Green")[128:, [0, 2]].max() == 0
    # grays / 3-3-2 RGB
    assert g("Grays").tolist() == [[i, i, i] for i in range(256)]
    assert g("3-3-2 RGB").tolist() == [[i & 0xE0, (i << 3) & 0xE0, (i << 6) & 0xC0] for i in range(256)]
    # the ImageJ interpolation helper differs from the linspace one (which Orange keeps using)
    assert np.array_equal(luts._interp([(0, 0, 0), (255, 128, 0)]), g("Orange")) and g("Orange")[255].tolist() == [255, 128, 0]
    assert luts._ij_interp([(0, 0, 0), (255, 255, 255)])[128:].tolist() == [[255, 255, 255]] * 128  # n = 2: knot 1 at 128, then flat


def test_sequential_maps_have_monotonic_luminance():
    for name in SEQUENTIAL:
        lum = _luminance(luts.get_lut(name))
        assert np.all(np.diff(lum) >= 0), name
        assert lum[-1] - lum[0] > 200, name
    for name in SEQUENTIAL_TOL:
        lum = _luminance(luts.get_lut(name))
        assert np.all(np.diff(lum) >= -1.0), name
        assert lum[-1] > lum[0] + 100, name


def test_label_maps_are_distinct():
    for name in LABELS:
        t = luts.get_lut(name)
        assert len({tuple(row) for row in t.tolist()}) == 256, name
    for name in ("Glasbey on dark", "Glasbey inverted", "Random"):
        assert luts.get_lut(name)[0].tolist() == [0, 0, 0], name
    # Fiji's glasbey.lut starts with white (background labels render white, as in Fiji); label images therefore use
    # LABEL_LUT, a Fiji Glasbey table with black at 0
    assert luts.get_lut("Glasbey")[0].tolist() == [255, 255, 255]
    assert luts.LABEL_LUT in luts.BUILTIN_NAMES and luts.LABEL_LUT.startswith("Glasbey") and luts.get_lut(luts.LABEL_LUT)[0].tolist() == [0, 0, 0]
    assert not np.array_equal(luts.get_lut("Glasbey on dark"), luts.get_lut("Glasbey inverted"))
    assert not np.array_equal(luts.get_lut("Glasbey on dark"), luts.get_lut("Glasbey"))
    # "Random" is reproducible and every colour shows on a black background
    r = luts.get_lut("Random")
    assert np.array_equal(r, luts._random_lut()) and not np.array_equal(r, luts._random_lut(seed=1))
    assert r[1:].max(axis=1).min() >= 96


# ---------------------------------------------------------------- helpers
def test_piecewise_and_blob_helpers():
    assert np.array_equal(luts._piecewise([(0, 0, 0, 0), (255, 255, 255, 255)]), luts.get_lut("Grays"))
    assert luts._piecewise([(0, 0, 0, 0), (127, 0, 0, 0), (128, 255, 255, 255), (255, 255, 255, 255)])[127:129].tolist() == [[0, 0, 0], [255, 255, 255]]
    import base64
    import zlib
    for name in ("HiLo", "16 colors", "Spectrum"):  # large negative steps exercise the modulo-256 deltas
        t = luts.get_lut(name)
        deltas = np.diff(t.T.astype(np.int16), axis=1, prepend=0).astype(np.uint8).tobytes()
        blob = base64.b64encode(zlib.compress(deltas)).decode()
        assert np.array_equal(luts._unpack_blob(blob), t), name


def test_builtin_registration_stays_cheap():
    t0 = time.perf_counter()
    luts._register_builtin()
    assert time.perf_counter() - t0 < 1.0  # decoding the Fiji set takes a few milliseconds; never precompute anything slow here
    assert all(name in luts._LUTS for name in FIJI_NAMES)


# ---------------------------------------------------------------- through the app
def test_api_luts_serves_the_fiji_set():
    c = TestClient(create_app(Session()))
    r = c.get("/api/luts")
    assert r.status_code == 200, r.text
    tables = r.json()
    for name in FIJI_NAMES + CORE_NAMES:
        assert name in tables, name
        t = tables[name]
        assert len(t) == 256 and all(len(row) == 3 and all(isinstance(v, int) and 0 <= v <= 255 for v in row) for row in t), name
    assert tables["Red Hot"][0] == [0, 0, 0] and tables["Union Jack"][127] == [255, 255, 255]
    assert c.get("/api/meta").json()["luts"] == luts.LUT_NAMES


def test_fiji_lut_applies_through_the_lut_op():
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.run_op("color.lut", {"lut": "Green Fire Blue"})
    assert s.active.display.channels[0].lut == "Green Fire Blue"
    rgb = luts.apply_lut(np.array([[0, 255]], np.uint8), "Green Fire Blue", 0, 255)
    assert rgb.shape == (1, 2, 3) and rgb[0, 0].tolist() == [0, 0, 0] and rgb[0, 1].tolist() == [255, 255, 255]


def test_fiji_lut_names_work_as_imagej_macro_commands():
    from lumen.core.ij_compat import IJ
    ij = IJ(Session())
    ij.run("Blobs (25K)")
    ij.run("Cyan Hot")  # run("<LUT name>") as in Fiji macros
    assert ij.img.display.channels[0].lut == "Cyan Hot"
    ij.run("glasbey on dark")  # case-insensitive like ImageJ's menu lookup
    assert ij.img.display.channels[0].lut == "Glasbey on dark"
    r, g, b = ij.getLut()
    assert (int(r[255]), int(g[255]), int(b[255])) == (17, 101, 82)
    # run("Thermal") / run("phase") / run("glasbey") / run("Red/Green") give Fiji's tables, not look-alikes
    for cmd, name, idx, rgb in (("Thermal", "Thermal", 1, (70, 0, 115)), ("phase", "Phase", 1, (0, 0, 252)), ("glasbey", "Glasbey", 1, (0, 0, 255)),
                                ("Red/Green", "Red/Green", 127, (254, 0, 0)), ("Fire", "Fire", 239, (255, 255, 215)), ("16 colors", "16 colors", 208, (245, 0, 135))):
        ij.run(cmd)
        assert ij.img.display.channels[0].lut == name, cmd
        r, g, b = ij.getLut()
        assert (int(r[idx]), int(g[idx]), int(b[idx])) == rgb, cmd


# ---------------------------------------------------------------- runtime registration guards / LUT_VERSION
@pytest.fixture
def _runtime_luts():
    """Undo runtime registrations after the test (the table is module-global and shared by every test module)."""
    saved_tables, saved_names = dict(luts._LUTS), list(luts.LUT_NAMES)
    yield
    luts._LUTS.clear()
    luts._LUTS.update(saved_tables)
    luts.LUT_NAMES[:] = saved_names
    luts._invalidate_caches()


def test_register_lut_never_overwrites_builtins_even_with_replace(_runtime_luts):
    fire = luts.get_lut("Fire").copy()
    black = np.zeros((256, 3), np.uint8)
    assert luts.register_lut("Fire", black, replace=True) == "Fire 2"  # replace=True only updates runtime LUTs
    assert np.array_equal(luts.get_lut("Fire"), fire) and np.array_equal(luts.get_lut("Fire 2"), black)
    assert luts.register_lut(" Grays ", black, replace=True) == "Grays 2"  # the name is stripped before the check
    assert luts.get_lut("Grays")[255].tolist() == [255, 255, 255]
    # runtime LUTs are still replaced in place
    other = np.full((256, 3), 7, np.uint8)
    assert luts.register_lut("Fire 2", other, replace=True) == "Fire 2" and np.array_equal(luts.get_lut("Fire 2"), other)
    assert luts.unregister_lut("Fire 2") and luts.unregister_lut("Grays 2") and not luts.unregister_lut("Fire")


def test_lut_version_tracks_every_table_mutation(_runtime_luts):
    v = luts.lut_version()
    assert isinstance(v, int) and v == luts.LUT_VERSION
    ramp = luts.get_lut("Grays").copy()
    assert luts.register_lut("Vtest", ramp) == "Vtest" and luts.lut_version() == v + 1          # added
    assert luts.register_lut("Vtest", ramp) == "Vtest" and luts.lut_version() == v + 1          # identical re-import: no-op
    assert luts.register_lut("Vtest", ramp[::-1], replace=True) == "Vtest"                      # replaced in place …
    assert luts.lut_version() == v + 2 and len(luts.LUT_NAMES) == len(set(luts.LUT_NAMES))      # … same names, new version
    assert not luts.unregister_lut("Grays") and luts.lut_version() == v + 2                     # refused: no bump
    assert luts.unregister_lut("Vtest") and luts.lut_version() == v + 3                         # removed
    # exposed next to lut_names so the browser can refetch /api/luts after an in-place replacement
    s = Session()
    st = s.state()
    assert st["lut_version"] == luts.lut_version() and st["lut_names"] is luts.LUT_NAMES
    c = TestClient(create_app(s))
    assert c.get("/api/state").json()["lut_version"] == luts.lut_version()
    luts.register_lut("Vtest", ramp)
    assert c.get("/api/state").json()["lut_version"] == luts.lut_version() == v + 4


# ---------------------------------------------------------------- LUT groups (Image ▸ Lookup Tables, Display select – GUI fix round 2)
def test_lut_groups_cover_every_builtin_exactly_once():
    grouped = [n for _, members in luts.LUT_GROUPS for n in members]
    assert len(grouped) == len(set(grouped)), "a LUT is listed in two groups"
    assert set(grouped) == set(luts.BUILTIN_NAMES), set(grouped) ^ set(luts.BUILTIN_NAMES)
    names = [g["name"] for g in luts.lut_groups()]
    assert names[:4] == ["ImageJ", "Fiji", "Perceptual (matplotlib)", "Labels"]
    assert "Fire" in luts.lut_groups()[0]["luts"] and "Viridis" in luts.lut_groups()[2]["luts"]
    for alias in luts.LUT_ALIASES:
        assert alias in luts.LUT_NAMES, alias


def test_lut_groups_list_runtime_tables_under_imported(_runtime_luts):
    name = luts.register_lut("Custom Ramp", np.linspace(0, 255, 256).astype(np.uint8)[:, None].repeat(3, axis=1))
    groups = luts.lut_groups()
    # other test modules may have left runtime tables behind: the newest registration is last in "Imported"
    assert groups[-1]["name"] == "Imported" and groups[-1]["luts"][-1] == name
    assert name not in [n for g in groups[:-1] for n in g["luts"]]


def test_api_luts_groups_endpoint():
    client = TestClient(create_app(Session()))
    r = client.get("/api/luts/groups")
    assert r.status_code == 200
    body = r.json()
    assert [g["name"] for g in body["groups"]][:2] == ["ImageJ", "Fiji"]
    assert "mpl-viridis" in body["aliases"]
    assert sum(len(g["luts"]) for g in body["groups"]) == len(client.get("/api/meta").json()["luts"])


def test_set_lut_op_is_named_set_lut_and_keeps_its_id():
    from lumen.core.registry import REGISTRY
    Session()  # loads the op modules
    spec = REGISTRY.get("color.lut")
    assert spec.name == "Set LUT"
    assert REGISTRY.get("Set LUT").id == "color.lut"
    # ImageJ's destructive "Apply LUT" resolves to the bake-into-pixels command alone now
    assert REGISTRY.get("Apply LUT").id == "type.apply_lut"
