"""File-format readers/writers, ImageJ ROI files and the import/export commands."""
import json
import os
import zipfile

import numpy as np
import pytest

from lumen.core import ops  # noqa: F401  (registers the commands)
from lumen.core import io
from lumen.core.image import Calibration, LumenImage
from lumen.core.session import Session


@pytest.fixture
def stack():
    rng = np.random.default_rng(0)
    data = rng.integers(0, 4000, (2, 3, 2, 17, 23)).astype(np.uint16)  # T Z C Y X
    return LumenImage(data, name="stack", axes="TZCYX", calibration=Calibration(0.5, 0.5, 2.0, "µm", frame_interval=1.5),
                      channel_names=["DAPI", "GFP"])


@pytest.fixture
def volume():
    rng = np.random.default_rng(1)
    return LumenImage(rng.integers(0, 255, (5, 10, 12)).astype(np.uint8), name="vol", axes="ZYX", calibration=Calibration(1.0, 1.0, 3.0, "mm"))


def _same(a: LumenImage, b: LumenImage) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a.data, b.data)


# ----------------------------------------------------------------------------- TIFF family
def test_imagej_tiff_hyperstack(tmp_path, stack):
    out = io.write_image(stack, tmp_path / "s.tif")
    back = io.read_image(out)
    assert _same(stack, back)
    assert back.calibration.unit == "µm" and back.calibration.pixel_width == pytest.approx(0.5)
    assert back.calibration.pixel_depth == pytest.approx(2.0) and back.calibration.frame_interval == pytest.approx(1.5)


def test_single_channel_imagej_tiff_roundtrip(tmp_path):
    """Regression: tifffile returns one ImageJ LUT as a (3, 256) ndarray, which used to break _apply_ij_display."""
    img = LumenImage(np.arange(12, dtype=np.uint8).reshape(3, 4), name="one")
    img.display.channels[0].lut = "Green"
    img.display.channels[0].min, img.display.channels[0].max = 10.0, 200.0
    back = io.read_image(io.write_image(img, tmp_path / "one.tif"))
    assert _same(img, back) and back.nc == 1
    assert back.display.channels[0].lut == "Green"
    assert (back.display.channels[0].min, back.display.channels[0].max) == (10.0, 200.0)
    img16 = LumenImage(np.arange(12, dtype=np.uint16).reshape(3, 4), name="one16")
    assert _same(img16, io.read_image(io.write_image(img16, tmp_path / "one16.tif")))


def test_ome_tiff_axes_and_channels(tmp_path, stack):
    out = io.write_image(stack, tmp_path / "s.ome.tif")
    assert out.endswith(".ome.tif")
    back = io.read_image(out)
    assert _same(stack, back)
    assert [c.name for c in back.display.channels] == ["DAPI", "GFP"]
    assert back.calibration.pixel_depth == pytest.approx(2.0)


def _display_image() -> LumenImage:
    rng = np.random.default_rng(5)
    img = LumenImage(rng.integers(0, 4000, (2, 2, 3, 9, 11)).astype(np.uint16), name="cells", axes="TZCYX", channel_names=["DAPI", "GFP", "RFP"])
    for ch, lut, lo, hi in zip(img.display.channels, ["Blue", "Green", "Magenta"], [10.0, 20.5, 30.0], [1000.0, 2000.25, 3500.0]):
        ch.lut, ch.min, ch.max = lut, lo, hi
    img.display.channels[2].inverted = True
    img.display.mode = "composite"
    return img


def _display_state(img: LumenImage):
    return [(c.name, c.lut, c.inverted) for c in img.display.channels], img.display.mode


def _assert_ranges_equal(a: LumenImage, b: LumenImage) -> None:
    for x, y in zip(a.display.channels, b.display.channels):
        assert x.min == pytest.approx(y.min) and x.max == pytest.approx(y.max)


def test_imagej_tiff_display_roundtrip(tmp_path):
    """Channel names (Labels), LUT tables, ranges, mode and inverted LUTs survive write -> read of an ImageJ TIFF."""
    import tifffile
    img = _display_image()
    img.display.channels[1].name = "α-tubulin"  # non-ASCII: Labels are UTF-16 (ImageJ's own encoding); the description stays ASCII
    out = io.write_image(img, tmp_path / "d.tif")
    back = io.read_image(out)
    assert _same(img, back)
    assert _display_state(back) == _display_state(img)
    _assert_ranges_equal(back, img)
    with tifffile.TiffFile(out) as tf:
        ij = tf.imagej_metadata
        assert tf.pages[0].description.isascii()
        assert list(ij["Labels"]) == ["DAPI", "α-tubulin", "RFP"] * (img.nt * img.nz) and ij["mode"] == "composite"
        assert list(ij["Ranges"]) == pytest.approx([10.0, 1000.0, 20.5, 2000.25, 30.0, 3500.0])
        luts = np.asarray(ij["LUTs"])
        assert luts.shape == (3, 3, 256)
        assert tuple(luts[2][:, 0]) == (255, 0, 255) and tuple(luts[2][:, -1]) == (0, 0, 0)  # inverted Magenta: reversed table
    # per-slice labels that differ within a channel are slice labels, not channel names
    tifffile.imwrite(str(tmp_path / "slices.tif"), img.data[0], imagej=True, metadata={"axes": "ZCYX", "Labels": [f"slice {i}" for i in range(6)]})
    assert [c.name for c in io.read_image(tmp_path / "slices.tif").display.channels] == ["C1", "C2", "C3"]


def test_ome_tiff_display_roundtrip(tmp_path):
    """OME-TIFF: standard Channel Name/Color plus Lumen's MapAnnotation (LUT names, ranges, mode, inverted, visible)."""
    import tifffile
    img = _display_image()
    img.display.channels[1].visible = False
    out = io.write_image(img, tmp_path / "d.ome.tif")
    back = io.read_image(out)
    assert _same(img, back)
    assert _display_state(back) == _display_state(img)
    assert [c.visible for c in back.display.channels] == [True, False, True]
    _assert_ranges_equal(back, img)
    with tifffile.TiffFile(out) as tf:
        xml = tf.ome_metadata
    # Color is the signed 32-bit RGBA of the LUT's top colour (blue, green, magenta)
    assert 'Name="DAPI" Color="65535"' in xml and 'Name="GFP" Color="16711935"' in xml and 'Name="RFP" Color="-16711681"' in xml
    assert 'Namespace="lumen.display"' in xml and '<M K="mode">composite</M>' in xml
    assert '<M K="channel.2.inverted">true</M>' in xml and '<M K="channel.1.visible">false</M>' in xml and '<M K="channel.1.max">2000.25</M>' in xml
    # a foreign OME-TIFF (no Lumen annotation): Color alone picks the nearest primary LUT, the auto range stays
    md = {"axes": "ZCYX", "Channel": {"Name": ["a", "b", "c"], "Color": [-16776961, -65281, -1]}}  # red, yellow, white
    tifffile.imwrite(str(tmp_path / "f.ome.tif"), img.data[0], ome=True, metadata=md)
    foreign = io.read_image(tmp_path / "f.ome.tif")
    assert [c.lut for c in foreign.display.channels] == ["Red", "Yellow", "Grays"]
    assert [c.name for c in foreign.display.channels] == ["a", "b", "c"] and foreign.display.channels[0].max != 1000.0


def test_lut_table_match_roundtrip(tmp_path):
    """A non-primary LUT (Fire) comes back by colour-table match; inverted tables come back as inverted=True."""
    from lumen.core.luts import get_lut
    f = LumenImage(np.arange(64, dtype=np.uint16).reshape(8, 8), name="fire")
    f.display.channels[0].lut = "Fire"
    assert io.read_image(io.write_image(f, tmp_path / "fire.tif")).display.channels[0].lut == "Fire"
    assert io.read_image(io.write_image(f, tmp_path / "fire.ome.tif")).display.channels[0].lut == "Fire"
    g = LumenImage(np.arange(64, dtype=np.uint8).reshape(8, 8), name="inv")
    g.display.channels[0].inverted = True
    b = io.read_image(io.write_image(g, tmp_path / "inv.tif"))
    assert (b.display.channels[0].lut, b.display.channels[0].inverted) == ("Grays", True)
    # Grays and HiLo differ in two entries only: both still come back exactly
    assert io._lut_from_ij_table(get_lut("HiLo").T) == ("HiLo", False) and io._lut_from_ij_table(get_lut("Grays").T) == ("Grays", False)
    # Lumen's Fire reproduces Fiji's LutLoader.fire() byte for byte; a perturbed table still matches by nearest table
    fire_r = [0, 0, 1, 25, 49, 73, 98, 122, 146, 162, 173, 184, 195, 207, 217, 229, 240, 252, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
    fire_g = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 35, 57, 79, 101, 117, 133, 147, 161, 175, 190, 205, 219, 234, 248, 255, 255, 255, 255]
    fire_b = [0, 61, 96, 130, 165, 192, 220, 227, 210, 181, 151, 122, 93, 64, 35, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35, 98, 160, 223, 255]
    ij_fire = np.zeros((3, 256), np.uint8)
    for i in range(256):
        i1, frac = int(i * 32 / 256), i * 32 / 256 - int(i * 32 / 256)
        i2 = min(i1 + 1, 31)
        for k, src in enumerate((fire_r, fire_g, fire_b)):
            ij_fire[k, i] = int((1 - frac) * src[i1] + frac * src[i2])
    assert np.array_equal(ij_fire.T, get_lut("Fire"))  # byte-identical to LutLoader.fire() now
    assert io._lut_from_ij_table(ij_fire) == ("Fire", False)
    assert io._lut_from_ij_table(ij_fire[:, ::-1]) == ("Fire", True)
    near = ij_fire.astype(np.int16)
    near[2, 100:110] += 3  # a slightly different table still comes back by nearest match
    assert io._lut_from_ij_table(near.clip(0, 255).astype(np.uint8)) == ("Fire", False)
    assert io._lut_from_ij_table(near.clip(0, 255).astype(np.uint8)[:, ::-1]) == ("Fire", True)
    red_inv = np.zeros((3, 256), np.uint8)  # Fiji "Invert LUT" on a Red channel: top-colour heuristic on the reversed table
    red_inv[0] = np.arange(255, -1, -1)
    assert io._lut_from_ij_table(red_inv) == ("Red", True)


def test_lsm_like_multipage_tiff_axes(tmp_path):
    """A plain multi-page TIFF with explicit axes (what an LSM/STK series looks like through tifffile)."""
    import tifffile
    data = np.random.default_rng(2).integers(0, 255, (4, 3, 16, 20)).astype(np.uint8)  # Z C Y X
    p = tmp_path / "multi.tif"
    tifffile.imwrite(str(p), data, photometric="minisblack", metadata={"axes": "ZCYX"})
    img = io.read_image(p)
    assert img.shape == (1, 4, 3, 16, 20)
    assert np.array_equal(img.data[0], data)
    # a TIFF with a 'Q'/'I' (unknown / sequence) axis is folded into Z
    tifffile.imwrite(str(tmp_path / "seq.tif"), data[:, 0], photometric="minisblack")
    img2 = io.read_image(tmp_path / "seq.tif")
    assert img2.nz == 4 and img2.nc == 1


def test_lsm_stk_metadata_hooks():
    """LSM voxel sizes (metres) and STK calibration are honoured when tifffile exposes them."""
    class FakeTag:
        def __init__(self, v):
            self.value = v

    class Page:
        tags = {}

    class TF:
        pages = [Page()]
        imagej_metadata = None
        ome_metadata = None
        lsm_metadata = {"VoxelSizeX": 0.2e-6, "VoxelSizeY": 0.2e-6, "VoxelSizeZ": 1e-6, "TimeIntervall": 2.0, "ChannelColors": {"ColorNames": ["Ch1", "Ch2"]}}
        stk_metadata = None
        metaseries_metadata = None
        micromanager_metadata = None
    cal, meta, names, _ = io._ij_metadata_to_calibration(TF())
    assert cal.unit == "µm" and cal.pixel_width == pytest.approx(0.2) and cal.pixel_depth == pytest.approx(1.0)
    assert cal.frame_interval == 2.0 and names == ["Ch1", "Ch2"]

    class TF2(TF):
        lsm_metadata = None
        stk_metadata = {"XCalibration": 0.65, "YCalibration": 0.65, "CalibrationUnits": "um", "ZDistance": np.array([0.0, 0.5, 0.5, 0.5])}
    cal, meta, names, _ = io._ij_metadata_to_calibration(TF2())
    assert cal.unit == "µm" and cal.pixel_width == pytest.approx(0.65) and cal.pixel_depth == pytest.approx(0.5)


# ----------------------------------------------------------------------------- numpy raw / text
def test_raw_import_variants(tmp_path):
    rng = np.random.default_rng(3)
    d16 = rng.integers(0, 60000, (3, 7, 9)).astype(np.uint16)
    (tmp_path / "be.raw").write_bytes(b"\x00" * 16 + d16.astype(">u2").tobytes())
    img = io.read_raw(tmp_path / "be.raw", 9, 7, "16-bit unsigned", offset=16, n_images=0)
    assert img.shape == (1, 3, 1, 7, 9) and np.array_equal(img.data[0, :, 0], d16)
    (tmp_path / "le.raw").write_bytes(d16[:1].astype("<u2").tobytes() + b"\xff" * 4 + d16[1:2].astype("<u2").tobytes())
    img = io.read_raw(tmp_path / "le.raw", 9, 7, "16-bit unsigned", n_images=2, gap=4, little_endian=True)
    assert np.array_equal(img.data[0, :, 0], d16[:2])
    rgb = rng.integers(0, 255, (4, 5, 3)).astype(np.uint8)
    rgb.tofile(str(tmp_path / "rgb.raw"))
    img = io.read_raw(tmp_path / "rgb.raw", 5, 4, "24-bit RGB")
    assert img.rgb and np.array_equal(np.transpose(img.data[0, 0], (1, 2, 0)), rgb)
    d8 = np.arange(12, dtype=np.uint8).reshape(3, 4)
    d8.tofile(str(tmp_path / "w.raw"))
    img = io.read_raw(tmp_path / "w.raw", 4, 3, "8-bit", white_is_zero=True)
    assert np.array_equal(img.data[0, 0, 0], 255 - d8)
    f32 = rng.random((6, 5)).astype(np.float32)
    f32.astype("<f4").tofile(str(tmp_path / "f.raw"))
    img = io.read_raw(tmp_path / "f.raw", 5, 6, "32-bit real", little_endian=True)
    assert img.dtype == np.float32 and np.array_equal(img.data[0, 0, 0], f32)
    s = Session()
    out = s.run_op("file.import_raw", {"path": str(tmp_path / "be.raw"), "image_type": "16-bit unsigned", "width": 9, "height": 7, "offset": 16, "n_images": 0})
    assert s.images[out["new_images"][0]].nz == 3


def test_text_image(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text("x,y,z\n1,2,3\n4,5,6.5\n")
    img = io.read_text_image(p)
    assert img.dtype == np.float32 and img.shape == (1, 1, 1, 2, 3) and img.data[0, 0, 0, 1, 2] == 6.5
    s = Session()
    out = s.run_op("file.import_text_image", {"path": str(p)})
    assert s.images[out["new_images"][0]].width == 3
    out = io.write_image(img, tmp_path / "m2.txt")
    assert np.array_equal(io.read_text_image(out).data, img.data)


# ----------------------------------------------------------------------------- optional-library formats
def test_dicom(tmp_path):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    d = tmp_path / "series"
    d.mkdir()
    for i in range(4):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SOPClassUID = pydicom.uid.CTImageStorage
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.Modality, ds.PatientName = "CT", "Test"
        ds.InstanceNumber = 4 - i  # files are written in reverse order on purpose
        ds.Rows, ds.Columns = 8, 10
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.PixelSpacing, ds.SliceThickness = [0.7, 0.8], 2.5
        ds.ImagePositionPatient = [0, 0, (4 - i) * 2.5]
        ds.PixelData = np.full((8, 10), (4 - i) * 100, np.uint16).tobytes()
        ds.save_as(str(d / f"img{i}.dcm"), enforce_file_format=True)
    one = io.read_image(d / "img0.dcm")
    assert one.shape == (1, 1, 1, 8, 10) and one.dtype == np.uint16
    assert one.calibration.unit == "mm" and one.calibration.pixel_width == 0.8 and one.calibration.pixel_height == 0.7
    vol = io.read_image(d)
    assert vol.nz == 4
    assert [int(vol.data[0, z, 0, 0, 0]) for z in range(4)] == [100, 200, 300, 400]  # sorted by InstanceNumber
    assert vol.calibration.pixel_depth == 2.5


def test_nifti(tmp_path, stack, volume):
    pytest.importorskip("nibabel")
    out = io.write_image(volume, tmp_path / "v.nii")
    back = io.read_image(out)
    assert _same(volume, back) and back.calibration.unit == "mm" and back.calibration.pixel_depth == 3.0
    out = io.write_image(stack, tmp_path / "s.nii.gz")
    assert out.endswith(".nii.gz")
    back = io.read_image(out)
    assert _same(stack, back) and back.calibration.unit == "µm" and back.calibration.pixel_width == pytest.approx(0.5)
    # a NIfTI written by nibabel directly (X, Y, Z order) comes back as (Z, Y, X) with voxel sizes from the affine
    import nibabel as nib
    arr = np.arange(2 * 3 * 4, dtype=np.int16).reshape(2, 3, 4)  # X Y Z
    nib.save(nib.Nifti1Image(arr, np.diag([1.5, 2.5, 3.5, 1.0])), str(tmp_path / "raw.nii"))
    img = io.read_image(tmp_path / "raw.nii")
    assert img.shape == (1, 4, 1, 3, 2)
    assert (img.calibration.pixel_width, img.calibration.pixel_height, img.calibration.pixel_depth) == (1.5, 2.5, 3.5)


def test_mrc(tmp_path, volume):
    pytest.importorskip("mrcfile")
    out = io.write_image(volume, tmp_path / "v.mrc")
    back = io.read_image(out)
    assert back.shape == volume.shape and np.array_equal(back.data, volume.data)
    assert back.calibration.unit == "Å" and back.calibration.pixel_depth == pytest.approx(3e7)
    f = LumenImage(np.random.default_rng(4).random((4, 8, 9)).astype(np.float32), name="f", axes="ZYX")
    back = io.read_image(io.write_image(f, tmp_path / "f.rec", "mrc"))
    assert _same(f, back)


def test_hdf5(tmp_path, stack):
    h5py = pytest.importorskip("h5py")
    out = io.write_image(stack, tmp_path / "s.h5")
    back = io.read_image(out)
    assert _same(stack, back) and back.calibration.unit == "µm" and [c.name for c in back.display.channels] == ["DAPI", "GFP"]
    # generic file: largest dataset is picked, element_size_um honoured, explicit dataset selectable
    with h5py.File(tmp_path / "g.h5", "w") as f:
        f.create_dataset("small", data=np.zeros((4, 4), np.uint8))
        ds = f.create_dataset("grp/big", data=np.arange(3 * 6 * 7, dtype=np.uint16).reshape(3, 6, 7))
        ds.attrs["element_size_um"] = np.array([2.0, 0.5, 0.5])
    img = io.read_image(tmp_path / "g.h5")
    assert img.shape == (1, 3, 1, 6, 7) and img.calibration.pixel_depth == 2.0 and img.calibration.unit == "µm"
    assert io.read_image(tmp_path / "g.h5", dataset="small").width == 4


def test_imaris_layout(tmp_path):
    h5py = pytest.importorskip("h5py")
    p = tmp_path / "cells.ims"
    with h5py.File(p, "w") as f:
        info = f.create_group("DataSetInfo/Image")
        for k, v in (("X", "7"), ("Y", "6"), ("Z", "3"), ("ExtMin0", "0"), ("ExtMin1", "0"), ("ExtMin2", "0"), ("ExtMax0", "3.5"), ("ExtMax1", "3"), ("ExtMax2", "6"), ("Unit", "um")):
            info.attrs[k] = np.frombuffer(v.encode(), dtype="S1")  # Imaris stores strings as char arrays
        for c in range(2):
            f.create_group(f"DataSetInfo/Channel {c}").attrs["Name"] = np.frombuffer(f"ch{c}".encode(), dtype="S1")
            # Imaris pads datasets to the chunk size – the reader must crop to the declared size
            f.create_dataset(f"DataSet/ResolutionLevel 0/TimePoint 0/Channel {c}/Data", data=np.full((4, 8, 8), c + 1, np.uint8))
    img = io.read_image(p)
    assert img.shape == (1, 3, 2, 6, 7)
    assert [c.name for c in img.display.channels] == ["ch0", "ch1"]
    assert img.calibration.unit == "µm" and img.calibration.pixel_width == pytest.approx(0.5) and img.calibration.pixel_depth == pytest.approx(2.0)


def test_zarr_ome(tmp_path, stack):
    zarr = pytest.importorskip("zarr")
    out = io.write_image(stack, tmp_path / "s.zarr")
    assert os.path.isdir(out)
    back = io.read_image(out)
    assert _same(stack, back)
    assert back.calibration.unit == "µm" and back.calibration.pixel_depth == pytest.approx(2.0) and back.calibration.pixel_width == pytest.approx(0.5)
    assert [c.name for c in back.display.channels] == ["DAPI", "GFP"]
    meta = json.loads((tmp_path / "s.zarr" / ".zattrs").read_text())
    assert meta["multiscales"][0]["version"] == "0.4"
    assert [a["name"] for a in meta["multiscales"][0]["axes"]] == ["t", "c", "z", "y", "x"]
    # a bare array (no multiscales) is opened too
    z = zarr.open(str(tmp_path / "plain.zarr"), mode="w", shape=(5, 6), dtype="uint8")
    z[:] = 7
    img = io.read_image(tmp_path / "plain.zarr")
    assert img.shape == (1, 1, 1, 5, 6) and img.data.max() == 7


def test_video_roundtrip(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    rng = np.random.default_rng(5)
    rgb = LumenImage(rng.integers(0, 255, (6, 21, 33, 3)).astype(np.uint8), name="mov", axes="TYXS", rgb=True)
    for fmt in ("avi", "mp4"):
        out = io.write_image(rgb, tmp_path / f"m.{fmt}", fps=5)
        back = io.read_image(out)
        assert back.nt == 6 and back.rgb and back.height in (21, 22) and back.width in (33, 34)
        assert back.calibration.frame_interval == pytest.approx(0.2)
    grey = LumenImage((np.arange(5 * 20 * 30) % 256).astype(np.uint8).reshape(5, 20, 30), name="g", axes="TYX")
    back = io.read_image(io.write_image(grey, tmp_path / "g.avi", fps=10))
    assert back.nt == 5 and not back.rgb


def test_pgm_ppm_and_imageio_formats(tmp_path, volume):
    out = io.write_image(volume, tmp_path / "v.pgm")
    back = io.read_image(out)
    assert np.array_equal(back.data[0, 0, 0], volume.data[0, 0, 0])
    rgb = LumenImage(np.random.default_rng(6).integers(0, 255, (9, 11, 3)).astype(np.uint8), name="c", axes="YXS", rgb=True)
    for fmt in ("ppm", "png", "webp", "tga"):
        back = io.read_image(io.write_image(rgb, tmp_path / f"c.{fmt}"))
        assert back.rgb and back.width == 11 and back.height == 9, fmt


def test_metaimage_pure_numpy(tmp_path, stack, volume):
    # MetaIO ElementSpacing is in millimetres: the µm stack is converted on write (time spacing untouched)
    # and the original unit is restored from the Comment tag on read
    out = io.write_image(stack, tmp_path / "s.mha")
    header = (tmp_path / "s.mha").read_bytes()[:600].decode("latin-1")
    assert "ElementSpacing = 0.0005 0.0005 0.002 1.5" in header and "Comment = Lumen unit=um" in header
    back = io.read_image(out)
    assert _same(stack, back) and back.calibration.unit == "µm"
    assert back.calibration.pixel_width == pytest.approx(0.5) and back.calibration.pixel_depth == pytest.approx(2.0)
    out = io.write_image(volume, tmp_path / "v.mhd")
    assert (tmp_path / "v.raw").exists()
    hdr = (tmp_path / "v.mhd").read_text()
    assert "ElementSpacing = 1 1 3" in hdr and "Comment" not in hdr  # mm files are written verbatim, no marker
    back = io.read_image(out)
    assert _same(volume, back) and back.calibration.unit == "mm" and back.calibration.pixel_depth == 3.0
    px = LumenImage(np.zeros((4, 5), np.uint8), name="px")  # uncalibrated: spacing 1 1, unit restored as 'pixel'
    back = io.read_image(io.write_image(px, tmp_path / "px.mha"))
    assert back.calibration.unit == "pixel" and back.calibration.pixel_width == 1.0
    # a foreign file without the marker keeps the mm convention
    (tmp_path / "f.mhd").write_text("ObjectType = Image\nNDims = 2\nDimSize = 3 2\nElementSpacing = 0.5 0.5\nElementType = MET_UCHAR\nElementDataFile = f.raw\n")
    (tmp_path / "f.raw").write_bytes(bytes(6))
    assert io.read_image(tmp_path / "f.mhd").calibration.unit == "mm"


def test_missing_optional_library_message(tmp_path, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "mrcfile", None)  # makes `import mrcfile` raise ImportError
    monkeypatch.setitem(sys.modules, "pydicom", None)
    (tmp_path / "x.mrc").write_bytes(b"\x00" * 1024)
    with pytest.raises(RuntimeError, match="pip install mrcfile"):
        io.read_image(tmp_path / "x.mrc")
    (tmp_path / "x.dcm").write_bytes(b"\x00" * 1024)
    with pytest.raises(RuntimeError, match="pip install pydicom"):
        io.read_image(tmp_path / "x.dcm")
    vol = LumenImage(np.zeros((2, 3, 4), np.uint8), name="v", axes="ZYX")
    with pytest.raises(RuntimeError, match="pip install mrcfile"):
        io.write_image(vol, tmp_path / "y.mrc")


def test_image_exts_and_list_dir(tmp_path):
    for ext in (".nii.gz", ".dcm", ".zarr", ".mrc", ".h5", ".mp4", ".pgm", ".lsm", ".stk", ".mha"):
        assert ext in io.IMAGE_EXTS, ext
    (tmp_path / "a.nii.gz").write_bytes(b"")
    (tmp_path / "notes.md").write_bytes(b"")
    (tmp_path / "x.zarr").mkdir()
    entries = {e["name"]: e for e in io.list_dir(str(tmp_path))["entries"]}
    assert entries["a.nii.gz"]["image"] and not entries["notes.md"]["image"]
    assert entries["x.zarr"]["dir"] and entries["x.zarr"]["image"]


# ----------------------------------------------------------------------------- ImageJ ROIs
ROIS = [
    {"type": "rect", "x": 10, "y": 20, "w": 30, "h": 40, "name": "box", "position": {"t": 0, "z": 3, "c": 1}},
    {"type": "oval", "x": 1.5, "y": 2.25, "w": 10, "h": 5, "name": "ellipse ü"},
    {"type": "polygon", "points": [[1, 1], [20, 3], [15, 18]], "name": "poly", "color": "#ff0000"},
    {"type": "freehand", "points": [[1.5, 1.25], [20.5, 3], [15, 18.75]], "name": "free"},
    {"type": "line", "x1": 3, "y1": 4, "x2": 30, "y2": 5.5, "width": 3, "name": "ln", "position": {"t": 2, "z": 0, "c": 0}},
    {"type": "polyline", "points": [[0, 0], [5, 5], [10, 0]], "name": "pl"},
    {"type": "point", "points": [[3, 4], [7, 8]], "name": "pts"},
]


def _strip(r):
    return {k: v for k, v in r.items() if k not in ("id", "image_id", "stroke_width")}


def test_ij_roi_zip_roundtrip(tmp_path):
    out = io.write_ij_rois(ROIS, tmp_path / "rois.zip")
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == sorted(f"{r['name']}.roi" for r in ROIS)
    back = io.read_ij_rois(out)
    assert len(back) == len(ROIS)
    for a, b in zip(ROIS, back):
        assert b["type"] == a["type"] and b["name"] == a["name"]
        if a["type"] in ("rect", "oval"):
            assert (b["x"], b["y"], b["w"], b["h"]) == (a["x"], a["y"], a["w"], a["h"])
        elif a["type"] == "line":
            assert (b["x1"], b["y1"], b["x2"], b["y2"], b["width"]) == (3.0, 4.0, 30.0, 5.5, 3.0)
        else:
            assert b["points"] == [[float(x), float(y)] for x, y in a["points"]]  # exact, incl. sub-pixel
        if "position" in a:
            assert b["position"] == a["position"]
        if "color" in a:
            assert b["color"] == a["color"]


def test_ij_roi_single_file_and_header(tmp_path):
    out = io.write_ij_rois([ROIS[0]], tmp_path / "box.roi")
    raw = (tmp_path / "box.roi").read_bytes()
    assert raw[:4] == b"Iout" and raw[6] == 1  # magic, rect type code
    assert int.from_bytes(raw[8:10], "big") == 20 and int.from_bytes(raw[10:12], "big") == 10  # top, left
    assert int.from_bytes(raw[12:14], "big") == 60 and int.from_bytes(raw[14:16], "big") == 40  # bottom, right
    hdr2 = int.from_bytes(raw[60:64], "big")
    assert int.from_bytes(raw[hdr2 + 4:hdr2 + 8], "big") == 2  # c position, 1-based
    assert int.from_bytes(raw[hdr2 + 8:hdr2 + 12], "big") == 4  # z position
    assert raw.endswith("box".encode("utf-16-be"))
    assert _strip(io.read_ij_rois(out)[0]) == _strip({**ROIS[0], "x": 10.0, "y": 20.0, "w": 30.0, "h": 40.0})
    # a polygon written with integer coordinates has no sub-pixel flag; one with fractions does
    poly = io.encode_ij_roi(ROIS[2])
    assert int.from_bytes(poly[50:52], "big") & 128 == 0 and int.from_bytes(poly[16:18], "big") == 3
    free = io.encode_ij_roi(ROIS[3])
    assert int.from_bytes(free[50:52], "big") & 128 == 128 and free[6] == 7


def test_ij_roi_mask_becomes_outline(tmp_path):
    from lumen.core import roi as roimod
    m = np.zeros((20, 20), bool)
    m[5:15, 5:15] = True
    r = roimod.from_mask(m, "blob")
    back = io.read_ij_rois(io.write_ij_rois([r], tmp_path / "m.zip"))[0]
    assert back["type"] == "freehand" and back["name"] == "blob"
    mask = roimod.to_mask(back, (20, 20))
    assert abs(int(mask.sum()) - 100) <= 20


def test_ij_roi_import_export_ops(tmp_path):
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    out = io.write_ij_rois(ROIS, tmp_path / "in.zip")
    res = s.run_op("file.import_ij_rois", {"path": out})
    assert "7" in res["message"] and len(s.roi_manager.rois) == 7
    assert s.roi_manager.rois[0]["position"] == {"t": 0, "z": 3, "c": 1}
    res = s.run_op("file.export_ij_rois", {"path": str(tmp_path / "out.zip")})
    assert (tmp_path / "out.zip").exists()
    names = [r["name"] for r in io.read_ij_rois(tmp_path / "out.zip")]
    assert names == [r["name"] for r in ROIS]
    s.run_op("file.export_ij_rois", {"path": str(tmp_path / "sel"), "which": "Current selection"}, roi={"type": "rect", "x": 1, "y": 2, "w": 3, "h": 4})
    assert io.read_ij_rois(tmp_path / "sel.roi")[0]["w"] == 3.0
    s.run_op("file.export_rois_json", {"path": str(tmp_path / "rois.json")})
    data = json.loads((tmp_path / "rois.json").read_text())
    assert len(data["rois"]) == 7
    s.roi_manager.clear()
    s.run_op("file.import_rois_json", {"path": str(tmp_path / "rois.json")})
    assert len(s.roi_manager.rois) == 7


# ----------------------------------------------------------------------------- sequences & exports
def test_import_sequence_with_regex(tmp_path):
    import tifffile
    rng = np.random.default_rng(7)
    frames = {}
    for i in (1, 2, 10):  # natural order differs from lexical order
        for ch in ("ch1", "ch2"):
            arr = rng.integers(0, 255, (8, 9)).astype(np.uint8)
            frames[(i, ch)] = arr
            tifffile.imwrite(str(tmp_path / f"img{i}_{ch}.tif"), arr)
    (tmp_path / "ignore.txt").write_text("x")
    s = Session()
    out = s.run_op("file.import_sequence", {"folder": str(tmp_path), "pattern": r"_ch1\.tif$", "sort": "natural"})
    img = s.images[out["new_images"][0]]
    assert img.shape == (1, 3, 1, 8, 9)
    assert np.array_equal(img.data[0, :, 0], np.stack([frames[(1, "ch1")], frames[(2, "ch1")], frames[(10, "ch1")]]))
    out = s.run_op("file.import_sequence", {"folder": str(tmp_path), "pattern": r"^img1_", "stack_as": "Channels"})
    img = s.images[out["new_images"][0]]
    assert img.nc == 2 and img.nz == 1 and [c.name for c in img.display.channels] == ["img1_ch1", "img1_ch2"]
    out = s.run_op("file.import_sequence", {"folder": str(tmp_path), "pattern": r"_ch2", "stack_as": "T series", "start": 2, "count": 2, "scale": 50})
    img = s.images[out["new_images"][0]]
    assert img.nt == 2 and img.height == 4
    with pytest.raises(FileNotFoundError):
        s.run_op("file.import_sequence", {"folder": str(tmp_path), "pattern": "nothing"})
    assert io.read_sequence(tmp_path).nz == 6  # plain folder → every image, sorted by name


def test_export_sequence_and_video(tmp_path, stack):
    s = Session()
    s.add_image(stack)
    out_dir = tmp_path / "seq"
    res = s.run_op("file.export_sequence", {"folder": str(out_dir), "format": "tif", "split": "Every slice (all channels)"})
    files = sorted(out_dir.iterdir())
    assert len(files) == 6 and files[0].name == "stack_z0001_t0001.tif"
    back = io.read_image(files[0])
    assert back.nc == 2 and np.array_equal(back.data[0, 0], stack.data[0, 0])
    assert res["table"]["name"] == "Export" and len(res["table"]["rows"]) == 6
    res = s.run_op("file.export_sequence", {"folder": str(tmp_path / "planes"), "format": "png", "split": "Every plane", "name_pattern": "{name}_{i:03d}"})
    assert len(list((tmp_path / "planes").glob("*.png"))) == 12
    res = s.run_op("file.export_sequence", {"folder": str(tmp_path / "frames"), "format": "npy", "split": "Every frame (whole Z stack)"})
    assert len(list((tmp_path / "frames").glob("*.npy"))) == 2
    res = s.run_op("file.export_video", {"path": str(tmp_path / "anim"), "format": "gif", "fps": 4, "axis": "Z"})
    assert (tmp_path / "anim.gif").exists() and "3 frame" in res["message"]
    if pytest.importorskip("imageio_ffmpeg", reason="no ffmpeg") is not None:
        s.run_op("file.export_video", {"path": str(tmp_path / "anim"), "format": "mp4", "fps": 4, "axis": "T"})
        assert io.read_image(tmp_path / "anim.mp4").nt == 2


def test_save_xy_coordinates_and_results(tmp_path):
    s = Session()
    s.run_op("file.sample", {"name": "Blobs"})
    s.active.calibration = Calibration(0.5, 0.5, 1.0, "µm")
    s.run_op("file.save_xy_coordinates", {"path": str(tmp_path / "xy")}, roi={"type": "polygon", "points": [[1, 2], [10, 2], [10, 12.5]]})
    lines = (tmp_path / "xy.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "X,Y,X (µm),Y (µm)" and lines[1] == "1,2,0.5,1" and lines[3] == "10,12.5,5,6.25"
    s.run_op("file.save_xy_coordinates", {"path": str(tmp_path / "line.txt")}, roi={"type": "line", "x1": 0, "y1": 0, "x2": 3, "y2": 4})
    assert len((tmp_path / "line.txt").read_text().splitlines()) == 3
    with pytest.raises(ValueError):
        s.run_op("file.save_xy_coordinates", {"path": str(tmp_path / "none.csv")})
    s.run_op("analyze.measure", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 10, "h": 10})
    s.run_op("analyze.measure", {}, roi={"type": "rect", "x": 5, "y": 5, "w": 10, "h": 10})
    s.run_op("file.export_results", {"path": str(tmp_path / "res"), "format": "csv"})
    rows = (tmp_path / "res.csv").read_text().splitlines()
    assert rows[0].startswith(" ,") and "Area" in rows[0] and len(rows) == 3
    s.run_op("file.export_results", {"path": str(tmp_path / "res.tsv"), "format": "tsv", "row_numbers": False})
    assert "\t" in (tmp_path / "res.tsv").read_text().splitlines()[0]
    if pytest.importorskip("openpyxl", reason="no openpyxl"):
        import openpyxl
        s.run_op("file.export_results", {"path": str(tmp_path / "res.xlsx"), "format": "xlsx"})
        ws = openpyxl.load_workbook(tmp_path / "res.xlsx").active
        assert ws.max_row == 3
    s.run_op("file.export_results", {"path": str(tmp_path / "summary.csv"), "table": "Export"}) if "Export" in s.tables else None
    with pytest.raises(ValueError):
        s.run_op("file.export_results", {"path": str(tmp_path / "x.csv"), "table": "NoSuchTable"})


def test_open_url_rejects_non_http():
    s = Session()
    with pytest.raises(ValueError):
        s.run_op("file.open_url", {"url": "file:///etc/passwd"})


# ----------------------------------------------------------------------------- regressions (review round 1)
def test_write_image_keeps_dotted_stems(tmp_path):
    """Path.with_suffix() used to eat everything after the last dot ('cells.v2' -> 'cells.tif')."""
    img = LumenImage(np.zeros((2, 3, 1, 8, 8), np.uint8), name="cells.v2.tif", axes="TZCYX")
    assert io.write_image(img, tmp_path / "cells.v2", "tif").replace("\\", "/").endswith("/cells.v2.tif")
    assert io.write_image(img, tmp_path / "exp-1.5x", "png").replace("\\", "/").endswith("/exp-1.5x.png")
    assert io.write_image(img, tmp_path / "exp-1.5x", "npy").replace("\\", "/").endswith("/exp-1.5x.npy")
    assert io.write_image(img, tmp_path / "cells.tiff", "tif").replace("\\", "/").endswith("/cells.tiff")  # alias kept
    assert io.write_image(img, tmp_path / "cells.png", "tif").replace("\\", "/").endswith("/cells.tif")  # known extension replaced
    assert io.write_text_image(img, tmp_path / "m.v2").replace("\\", "/").endswith("/m.v2.txt")
    assert io.write_text_image(img, tmp_path / "m2.csv").replace("\\", "/").endswith("/m2.csv")
    assert io.write_video([np.zeros((8, 8, 3), np.uint8)] * 2, tmp_path / "anim.v2", "gif").replace("\\", "/").endswith("/anim.v2.gif")
    out = io.write_image(img, tmp_path / "vol.v2", "mhd")
    assert out.replace("\\", "/").endswith("/vol.v2.mhd") and (tmp_path / "vol.v2.raw").exists()
    assert io.write_ij_rois(ROIS[:2], tmp_path / "rois.v2").replace("\\", "/").endswith("/rois.v2.zip")
    assert io.strip_image_ext("sample.ome.tif") == "sample" and io.strip_image_ext("cells.v2.tif") == "cells.v2"
    assert io.strip_image_ext("cells.v2") == "cells.v2"
    s = Session()
    s.add_image(img)
    out_dir = tmp_path / "seq"
    res = s.run_op("file.export_sequence", {"folder": str(out_dir), "format": "tif", "split": "Every slice (all channels)"})
    names = sorted(f.name for f in out_dir.iterdir())
    assert len(names) == 6 and names[0] == "cells.v2_z0001_t0001.tif" and len(res["table"]["rows"]) == 6
    for z in range(3):
        back = io.read_image(out_dir / f"cells.v2_z{z + 1:04d}_t0002.tif")
        assert np.array_equal(back.data[0, 0], img.data[1, z])


def test_ome_tiff_rgb_roundtrip(tmp_path):
    import skimage.data
    from lumen.core.render import render_rgb
    img = LumenImage(skimage.data.astronaut(), name="a", axes="YXS", rgb=True)
    back = io.read_image(io.write_image(img, tmp_path / "a.ome.tif"))
    assert back.rgb and _same(img, back)
    assert [c.lut for c in back.display.channels] == ["Red", "Green", "Blue"]
    assert np.array_equal(render_rgb(img), render_rgb(back))
    z = LumenImage(np.stack([skimage.data.astronaut()[::4, ::4]] * 3), name="z", axes="ZYXS", rgb=True)
    backz = io.read_image(io.write_image(z, tmp_path / "z.ome.tif", compress=True))
    assert backz.rgb and _same(z, backz)


def test_ome_tiff_narrow_images_are_not_rgb(tmp_path):
    """Images 3 or 4 pixels wide (kymograph columns, line profiles, small crops) write as OME-TIFF: tifffile must not take
    the trailing X of 3 / 4 for RGB samples ('axes do not match stored shape')."""
    for shape in ((1, 1, 1, 10, 4), (1, 1, 1, 10, 3), (1, 1, 3, 10, 3), (1, 1, 4, 10, 4), (2, 2, 2, 3, 3)):
        for dt in (np.uint8, np.uint16, np.float32):
            img = LumenImage(np.arange(int(np.prod(shape)), dtype=dt).reshape(shape), name="n", axes="TZCYX")
            img.display.channels[0].name = "DAPI"
            back = io.read_image(io.write_image(img, tmp_path / "n.ome.tif", "ome.tif", compression="LZW"))
            assert _same(img, back) and not back.rgb and back.display.channels[0].name == "DAPI", (shape, dt)
    rgb = LumenImage(np.zeros((1, 1, 3, 10, 4), np.uint8), name="rgb", axes="TZCYX", rgb=True)
    back = io.read_image(io.write_image(rgb, tmp_path / "rgb.ome.tif", "ome.tif"))
    assert back.rgb and back.shape == (1, 1, 3, 10, 4)


def test_imagej_slice_labels_roundtrip(tmp_path):
    """ImageJ per-slice Labels become metadata['slice_labels'] (all of them, in T/Z/C order) and survive a plain Open + Save;
    per-channel-consistent Labels stay channel names so a channel rename still survives save + reopen."""
    import tifffile
    from lumen.core.ops.stacks import plane_index, slice_labels
    labels = [f"frame_{i:03d}.png" for i in range(5)]
    tifffile.imwrite(str(tmp_path / "labelled.tif"), np.zeros((5, 8, 8), np.uint8), imagej=True, metadata={"axes": "ZYX", "Labels": labels})
    img = io.read_image(tmp_path / "labelled.tif")
    assert img.metadata["slice_labels"] == labels and slice_labels(img) == labels and "labels" not in img.metadata
    out = io.write_image(img, tmp_path / "resaved.tif")
    with tifffile.TiffFile(out) as tf:
        assert list(tf.imagej_metadata["Labels"]) == labels
    assert slice_labels(io.read_image(out)) == labels
    assert slice_labels(io.read_image(tmp_path / "labelled.tif", virtual=True)) == labels  # the virtual path too
    many = [f"s{i}" for i in range(60)]
    tifffile.imwrite(str(tmp_path / "l60.tif"), np.zeros((60, 4, 4), np.uint8), imagej=True, metadata={"axes": "ZYX", "Labels": many})
    assert slice_labels(io.read_image(tmp_path / "l60.tif")) == many  # no 50-entry cap
    lab = [f"t{t}z{z}c{c}" for t in range(2) for z in range(3) for c in range(2)]
    tifffile.imwrite(str(tmp_path / "hs.tif"), np.zeros((2, 3, 2, 4, 4), np.uint8), imagej=True, metadata={"axes": "TZCYX", "Labels": lab})
    hs = io.read_image(tmp_path / "hs.tif")
    assert all(slice_labels(hs)[plane_index(hs, t, z, c)] == f"t{t}z{z}c{c}" for t in range(2) for z in range(3) for c in range(2))
    assert [c.name for c in hs.display.channels] == ["C1", "C2"]
    # labels that are the same on every plane of a channel are channel names, not slice labels
    ch = LumenImage(np.zeros((2, 2, 2, 4, 4), np.uint16), name="ch", axes="TZCYX", channel_names=["DAPI", "GFP"])
    back = io.read_image(io.write_image(ch, tmp_path / "ch.tif"))
    assert [c.name for c in back.display.channels] == ["DAPI", "GFP"] and "slice_labels" not in back.metadata
    back.display.channels[1].name = "mCherry"
    assert [c.name for c in io.read_image(io.write_image(back, tmp_path / "ch2.tif")).display.channels] == ["DAPI", "mCherry"]
    # partial labels (fewer than planes) are padded, empties are written verbatim; a stack without labels writes channel names
    img.metadata["slice_labels"] = ["first", "", "third"]
    with tifffile.TiffFile(io.write_image(img, tmp_path / "partial.tif")) as tf:
        assert list(tf.imagej_metadata["Labels"]) == ["first", "", "third", "", ""]
    assert slice_labels(io.read_image(tmp_path / "partial.tif")) == ["first", "", "third", "", ""]
    plain = LumenImage(np.zeros((1, 3, 1, 4, 4), np.uint8), name="plain", axes="TZCYX")
    with tifffile.TiffFile(io.write_image(plain, tmp_path / "plain.tif")) as tf:
        assert list(tf.imagej_metadata["Labels"]) == ["C1", "C1", "C1"]


def test_ome_zarr_without_axes_is_tczyx(tmp_path):
    """OME-Zarr 0.1 / 0.2 multiscales have no 'axes' key: the layout is the fixed T C Z Y X, and the Series table and the
    opened image must agree (Z and C used to be swapped on open)."""
    zarr = pytest.importorskip("zarr")
    data = np.zeros((1, 2, 10, 16, 16), np.uint16)
    for c in range(2):
        for z in range(10):
            data[0, c, z] = 1000 * c + z

    def write(path, arr, version="0.2", axes=None):
        try:
            g = zarr.open_group(str(path), mode="w", zarr_format=2)
        except TypeError:
            g = zarr.open_group(str(path), mode="w")
        if hasattr(g, "create_array"):
            a = g.create_array("0", shape=arr.shape, chunks=arr.shape, dtype=arr.dtype)
            a[...] = arr
        else:
            g.create_dataset("0", data=arr, chunks=arr.shape)
        ms = {"version": version, "datasets": [{"path": "0"}]}
        if axes is not None:
            ms["axes"] = axes
        g.attrs["multiscales"] = [ms]
    write(tmp_path / "old5.zarr", data)
    info = io.list_series(tmp_path / "old5.zarr")[0]
    img = io.read_image(tmp_path / "old5.zarr")
    assert (info.size_z, info.size_c, info.axes) == (10, 2, "TCZYX") and (img.nz, img.nc) == (10, 2)
    assert all(int(img.data[0, z, c, 0, 0]) == 1000 * c + z for c in range(2) for z in range(10))
    write(tmp_path / "old4.zarr", data[0])
    info = io.list_series(tmp_path / "old4.zarr")[0]
    img = io.read_image(tmp_path / "old4.zarr")
    assert (info.size_z, info.size_c, info.axes) == (10, 2, "CZYX") and (img.nz, img.nc) == (10, 2) and int(img.data[0, 3, 1, 0, 0]) == 1003
    write(tmp_path / "bad.zarr", data, version="0.4", axes=[{"name": "c"}, {"name": "y"}, {"name": "x"}])  # wrong length: both sides default
    assert (io.list_series(tmp_path / "bad.zarr")[0].size_z, io.read_image(tmp_path / "bad.zarr").nz) == (10, 10)
    write(tmp_path / "ok.zarr", np.transpose(data, (0, 2, 1, 3, 4)), version="0.4", axes=[{"name": n} for n in "tzcyx"])  # explicit axes still win
    ok = io.read_image(tmp_path / "ok.zarr")
    assert (ok.nz, ok.nc) == (10, 2) and int(ok.data[0, 3, 1, 0, 0]) == 1003


def test_npy_keeps_full_hyperstack(tmp_path):
    for shape, dtype in (((2, 1, 2, 6, 7), np.uint8), ((1, 1, 3, 6, 7), np.uint8), ((1, 1, 1, 1, 100), np.float32),
                         ((3, 4, 1, 6, 7), np.uint16), ((1, 1, 1, 1, 1), np.uint8), ((1, 1, 3, 6, 3), np.uint8)):
        img = LumenImage(np.random.default_rng(1).integers(0, 200, shape).astype(dtype), name="n", axes="TZCYX")
        out = io.write_image(img, tmp_path / "n.npy")
        assert np.load(out).shape == shape
        assert _same(img, io.read_image(out)), shape
    np.save(tmp_path / "legacy.npy", np.zeros((3, 6, 7), np.uint8))  # a foreign 3-D array still opens as a Z stack
    assert io.read_image(tmp_path / "legacy.npy").shape == (1, 3, 1, 6, 7)


def test_ij_roi_large_point_counts(tmp_path):
    import struct
    th = np.linspace(0, 2 * np.pi, 40000, endpoint=False)
    sub = np.stack([5000 + 1000 * np.cos(th), 5000 + 1000 * np.sin(th)], 1)
    ints = np.round(sub).astype(int)
    for roi in ({"type": "polygon", "points": ints.tolist()}, {"type": "freehand", "points": sub.tolist()},
                {"type": "polygon", "points": ints[:1000].tolist() * 65}):  # 65000 > 32767 (signed) but <= 65535
        raw = io.encode_ij_roi(roi)
        n = len(roi["points"])
        assert struct.unpack_from(">H", raw, 16)[0] == n  # unsigned count, not clipped to 32767
        pts = np.asarray(io.decode_ij_roi(raw)["points"])
        assert len(pts) == n and np.allclose(pts, np.asarray(roi["points"]), atol=1e-3)
    # > 65535 points: a point ROI goes through ImageJ's count-0 + SIZE field…
    many = np.stack([np.arange(70000) % 300, np.arange(70000) // 300], 1).tolist()
    raw = io.encode_ij_roi({"type": "point", "points": many})
    assert struct.unpack_from(">H", raw, 16)[0] == 0 and struct.unpack_from(">i", raw, 18)[0] == 70000
    assert np.array_equal(io.decode_ij_roi(raw)["points"], many)
    # …a polygon outline is saved as a composite ShapeRoi (type rect + shape size), as ImageJ does
    raw = io.encode_ij_roi({"type": "polygon", "points": many, "name": "big"})
    assert raw[6] == 1 and struct.unpack_from(">i", raw, 36)[0] == 3 * 70000 + 1
    back = io.decode_ij_roi(raw)
    assert back["type"] == "polygon" and back["name"] == "big" and np.array_equal(back["points"], many)
    back = io.read_ij_rois(io.write_ij_rois([{"type": "freehand", "points": many}], tmp_path / "big.zip"))[0]
    assert len(back["points"]) == 70000
    with pytest.raises(ValueError, match="65535"):
        io.encode_ij_roi({"type": "polyline", "points": many})
    corrupt = bytearray(64)
    corrupt[0:4] = b"Iout"
    struct.pack_into(">H", corrupt, 16, 1000)  # polygon (type 0) claiming 1000 points in a 64-byte file
    with pytest.raises(ValueError, match="Corrupt"):
        io.decode_ij_roi(bytes(corrupt))


def test_ij_roi_name_with_non_bmp_characters(tmp_path):
    for name in ("cell 🙂 A", "𝔸𝔹ℂ", "a😀😀", "日本語 A", "ümlaut é"):
        raw = io.encode_ij_roi({"type": "rect", "x": 1, "y": 2, "w": 3, "h": 4, "name": name})
        assert io.decode_ij_roi(raw)["name"] == name
    back = io.read_ij_rois(io.write_ij_rois([{**ROIS[0], "name": "cell 🙂 A"}], tmp_path / "emoji.zip"))
    assert back[0]["name"] == "cell 🙂 A"


def test_grey_alpha_png_reads_as_greyscale(tmp_path):
    import imageio.v3 as iio
    la = np.zeros((20, 30, 2), np.uint8)
    la[..., 0] = (np.arange(30, dtype=np.uint8) * 8)[None, :]
    la[..., 1] = 255
    for ext in ("png", "tga"):
        iio.imwrite(tmp_path / f"la.{ext}", la)
        back = io.read_image(tmp_path / f"la.{ext}")
        assert back.shape == (1, 1, 1, 20, 30) and not back.rgb, ext
        assert np.array_equal(back.data[0, 0, 0], la[..., 0])
    grey = np.zeros((5, 16, 24), np.uint8)  # a multi-frame greyscale stack must still be read as (N, H, W)
    grey[:] = np.arange(5, dtype=np.uint8)[:, None, None]
    iio.imwrite(tmp_path / "stack.tif", grey)
    assert io.read_image(tmp_path / "stack.tif").shape == (1, 5, 1, 16, 24)


def test_hdf5_non_ascii_channel_names(tmp_path):
    pytest.importorskip("h5py")
    img = LumenImage(np.zeros((1, 1, 2, 6, 7), np.uint16), name="x", axes="TZCYX", channel_names=["DAPI", "GFP – 488 (µ)"])
    back = io.read_image(io.write_image(img, tmp_path / "x.h5"))
    assert [c.name for c in back.display.channels] == ["DAPI", "GFP – 488 (µ)"]


def test_time_unit_roundtrip(tmp_path):
    import tifffile
    img = LumenImage(np.zeros((4, 1, 1, 6, 7), np.uint8), name="tl", axes="TZCYX",
                     calibration=Calibration(0.65, 0.65, 1.0, "µm", frame_interval=5.0, time_unit="min"))
    for fmt in ("tif", "ome.tif", "h5", "zarr"):
        if fmt == "h5":
            pytest.importorskip("h5py")
        if fmt == "zarr":
            pytest.importorskip("zarr")
        cal = io.read_image(io.write_image(img, tmp_path / f"tl.{fmt}")).calibration
        assert (cal.frame_interval, cal.time_unit) == (5.0, "min"), fmt
    assert "tunit=min" in tifffile.TiffFile(tmp_path / "tl.tif").pages[0].description
    if pytest.importorskip("nibabel", reason="no nibabel"):
        cal = io.read_image(io.write_image(img, tmp_path / "tl.nii")).calibration  # NIfTI has no minutes: converted
        assert cal.frame_interval == pytest.approx(300.0) and cal.time_unit == "sec"
    # a Fiji-authored file carries tunit= in the description
    tifffile.imwrite(tmp_path / "fiji.tif", np.zeros((4, 6, 7), np.uint8), imagej=True, metadata={"axes": "TYX", "finterval": 5.0, "tunit": "min"})
    cal = io.read_image(tmp_path / "fiji.tif").calibration
    assert (cal.frame_interval, cal.time_unit) == (5.0, "min")
    secs = LumenImage(np.zeros((2, 1, 1, 6, 7), np.uint8), name="s", axes="TZCYX", calibration=Calibration(frame_interval=2.0))
    assert "tunit" not in tifffile.TiffFile(io.write_image(secs, tmp_path / "s.tif")).pages[0].description  # ImageJ omits seconds


def test_imagej_min_max_display_range(tmp_path):
    import tifffile
    # Fiji saves the window of a non-composite image as min=/max= in the description (Ranges= is for composites)
    tifffile.imwrite(tmp_path / "ij.tif", np.full((5, 5), 1000, np.uint16), description="ImageJ=1.54f\nimages=1\nunit=\\u00B5m\nmin=100.0\nmax=3000.0\n")
    ch = io.read_image(tmp_path / "ij.tif").display.channels[0]
    assert (ch.min, ch.max) == (100.0, 3000.0)
    tifffile.imwrite(tmp_path / "unset.tif", np.full((5, 5), 1000, np.uint16), description="ImageJ=1.54f\nimages=1\nmin=0.0\nmax=0.0\n")
    ch = io.read_image(tmp_path / "unset.tif").display.channels[0]
    assert (ch.min, ch.max) != (0.0, 0.0)  # 0/0 means unset: auto range kept
    tifffile.imwrite(tmp_path / "two.tif", np.zeros((2, 5, 5), np.uint16), imagej=True, metadata={"axes": "CYX", "min": 100.0, "max": 3000.0})
    back = io.read_image(tmp_path / "two.tif")
    assert back.nc == 2 and all((c.min, c.max) == (100.0, 3000.0) for c in back.display.channels)
    tifffile.imwrite(tmp_path / "both.tif", np.zeros((2, 5, 5), np.uint16), imagej=True,
                     metadata={"axes": "CYX", "min": 100.0, "max": 3000.0, "Ranges": (10.0, 20.0, 30.0, 40.0)})
    back = io.read_image(tmp_path / "both.tif")
    assert [(c.min, c.max) for c in back.display.channels] == [(10.0, 20.0), (30.0, 40.0)]  # per-channel Ranges win
    img = LumenImage(np.arange(12, dtype=np.uint16).reshape(3, 4), name="w")
    img.display.channels[0].min, img.display.channels[0].max = 2.0, 9.0
    md = tifffile.TiffFile(io.write_image(img, tmp_path / "w.tif")).imagej_metadata
    assert (md["min"], md["max"]) == (2.0, 9.0)


def test_tiff_resolution_unit(tmp_path):
    import tifffile
    z = np.zeros((10, 10), np.uint8)
    tifffile.imwrite(tmp_path / "inch.tif", z, resolution=(300, 300), resolutionunit="INCH")
    tifffile.imwrite(tmp_path / "cm.tif", z, resolution=(100, 100), resolutionunit="CENTIMETER")
    tifffile.imwrite(tmp_path / "none.tif", z, resolution=(300, 300), resolutionunit="NONE")
    tifffile.imwrite(tmp_path / "dpi72.tif", z, resolution=(72, 72), resolutionunit="INCH")
    tifffile.imwrite(tmp_path / "micro_cm.tif", z, resolution=(20000, 20000), resolutionunit="CENTIMETER")
    c = io.read_image(tmp_path / "inch.tif").calibration
    assert c.unit == "inch" and c.pixel_width == pytest.approx(1 / 300)
    c = io.read_image(tmp_path / "cm.tif").calibration
    assert c.unit == "cm" and c.pixel_width == pytest.approx(0.01)
    c = io.read_image(tmp_path / "none.tif").calibration
    assert c.unit == "unit" and c.pixel_width == pytest.approx(1 / 300)
    c = io.read_image(tmp_path / "dpi72.tif").calibration
    assert c.unit == "pixel" and c.pixel_width == 1.0  # screen default: uncalibrated, like ImageJ
    c = io.read_image(tmp_path / "micro_cm.tif").calibration
    assert c.unit == "µm" and c.pixel_width == pytest.approx(0.5)  # ImageJ converts <= 0.001 cm to µm
    lumen = LumenImage(z, name="l", calibration=Calibration(0.5, 0.5, 1.0, "µm"))
    assert io.read_image(io.write_image(lumen, tmp_path / "lumen.tif")).calibration.unit == "µm"


def test_revert_reuses_import_parameters(tmp_path):
    import tifffile
    f = tmp_path / "seq"
    f.mkdir()
    for i in range(3):
        for ch in (1, 2):
            tifffile.imwrite(f / f"img{i}_ch{ch}.tif", np.full((4, 5), i * 10 + ch, np.uint8))
    s = Session()
    s.run_op("file.import_sequence", {"folder": str(f), "pattern": r"_ch1\.tif$", "stack_as": "T series"})
    img = s.active
    assert img.shape == (3, 1, 1, 4, 5)
    original = img.data.copy()
    s.run_op("image.crop", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2})
    assert img.shape == (3, 1, 1, 2, 2)
    s.run_op("file.revert", {})
    assert img.shape == (3, 1, 1, 4, 5) and np.array_equal(img.data, original)
    # after Save As the image is backed by the new file, which Revert re-reads instead of the folder
    s.save_image(img.id, str(tmp_path / "saved.v1"), "tif")
    s.run_op("image.crop", {}, roi={"type": "rect", "x": 0, "y": 0, "w": 2, "h": 2})
    s.run_op("file.revert", {})
    assert img.path.endswith("saved.v1.tif") and np.array_equal(img.data, original)
    # raw import (single file and "open all files in folder")
    rawdir = tmp_path / "raw"
    rawdir.mkdir()
    for k in range(2):
        (rawdir / f"v{k}.raw").write_bytes(bytes(range(k * 40, k * 40 + 40)))
    s.run_op("file.import_raw", {"path": str(rawdir / "v0.raw"), "width": 5, "height": 4, "n_images": 2})
    raw = s.active
    assert raw.shape == (1, 2, 1, 4, 5)
    s.run_op("image.crop", {}, roi={"type": "rect", "x": 1, "y": 1, "w": 2, "h": 2})
    s.run_op("file.revert", {})
    assert raw.shape == (1, 2, 1, 4, 5) and raw.data[0, 1, 0, 0, 0] == 20
    s.run_op("file.import_raw", {"path": str(rawdir / "v0.raw"), "width": 5, "height": 4, "n_images": 2, "open_all": True})
    allraw = s.active
    assert allraw.shape == (1, 4, 1, 4, 5)
    s.run_op("file.revert", {})
    assert allraw.shape == (1, 4, 1, 4, 5) and allraw.data[0, 3, 0, 0, 0] == 60
    # multi-series TIFF and explicit HDF5 dataset
    with tifffile.TiffWriter(tmp_path / "multi.tif") as tw:
        tw.write(np.zeros((8, 8), np.uint8))
        tw.write(np.ones((4, 5), np.uint8))
    ms = s.add_image(io.read_image(tmp_path / "multi.tif", series=1))
    assert ms.shape == (1, 1, 1, 4, 5)
    s.run_op("file.revert", {}, image_id=ms.id)
    assert ms.shape == (1, 1, 1, 4, 5) and ms.data.max() == 1
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "g.h5", "w") as hf:
        hf.create_dataset("small", data=np.full((3, 3), 7, np.uint8))
        hf.create_dataset("big", data=np.zeros((10, 10), np.uint8))
    hi = s.add_image(io.read_image(tmp_path / "g.h5", dataset="small"))
    s.run_op("file.revert", {}, image_id=hi.id)
    assert hi.shape == (1, 1, 1, 3, 3) and hi.data.max() == 7


def _stub_module(monkeypatch, name, **attrs):
    import sys
    import types
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def test_nd2_positions_and_time(tmp_path, monkeypatch):
    class FakeND2:
        def __init__(self, path, sizes):
            self.sizes = sizes

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def asarray(self):
            shape = tuple(self.sizes.values())
            arr = np.zeros(shape, np.uint16)
            if "T" not in self.sizes:
                return arr
            # tag every frame with 10 * position + time so the folded order can be checked
            pi, ti = list(self.sizes).index("P"), list(self.sizes).index("T")
            for p in range(self.sizes["P"]):
                for t in range(self.sizes["T"]):
                    idx = [slice(None)] * arr.ndim
                    idx[pi], idx[ti] = p, t
                    arr[tuple(idx)] = 10 * p + t
            return arr

        def voxel_size(self):
            raise RuntimeError("no calibration")

    (tmp_path / "x.nd2").write_bytes(b"\x00")
    for order in ({"P": 2, "T": 3, "Z": 4, "C": 2, "Y": 4, "X": 5}, {"T": 3, "P": 2, "Z": 4, "C": 2, "Y": 4, "X": 5}):
        sizes = dict(order)
        _stub_module(monkeypatch, "nd2", ND2File=lambda path, sizes=sizes: FakeND2(path, sizes))
        img = io.read_image(tmp_path / "x.nd2")
        assert img.shape == (6, 4, 2, 4, 5) and img.metadata["positions"] == 2
        assert [int(img.data[t, 0, 0, 0, 0]) for t in range(6)] == [0, 1, 2, 10, 11, 12]
    _stub_module(monkeypatch, "nd2", ND2File=lambda path: FakeND2(path, {"P": 2, "C": 2, "Y": 4, "X": 5}))
    assert io.read_image(tmp_path / "x.nd2").shape == (2, 1, 2, 4, 5)  # positions only: frames, unchanged


def test_czi_scenes_and_samples(tmp_path, monkeypatch):
    class ClassicCzi:  # czifile <= 2019.7: '0' = sample, 'S' = scene
        def __init__(self, path, axes, shape, dtype=np.uint16):
            self.axes, self.shape, self.dtype = axes, shape, dtype

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def asarray(self):
            arr = np.zeros(self.shape, self.dtype)
            if "S" in self.axes:
                i = self.axes.index("S")
                for k in range(self.shape[i]):
                    np.moveaxis(arr, i, 0)[k] = k + 1
            return arr

    (tmp_path / "x.czi").write_bytes(b"\x00")
    _stub_module(monkeypatch, "czifile", CziFile=lambda p: ClassicCzi(p, "SCZYX0", (4, 2, 5, 6, 7, 1)))
    img = io.read_image(tmp_path / "x.czi")  # 4 scenes, no time axis -> scenes become frames, nothing dropped
    assert img.shape == (4, 5, 2, 6, 7) and not img.rgb and img.metadata["positions"] == 4
    assert [int(img.data[t, 0, 0, 0, 0]) for t in range(4)] == [1, 2, 3, 4]
    _stub_module(monkeypatch, "czifile", CziFile=lambda p: ClassicCzi(p, "STCZYX0", (3, 2, 1, 5, 6, 7, 1)))
    img = io.read_image(tmp_path / "x.czi", series=2)  # scenes + time: `series` picks the scene
    assert img.shape == (2, 5, 1, 6, 7) and img.metadata["series_count"] == 3 and int(img.data.max()) == 3
    _stub_module(monkeypatch, "czifile", CziFile=lambda p: ClassicCzi(p, "SZYX0", (1, 5, 6, 7, 3), np.uint8))
    img = io.read_image(tmp_path / "x.czi")  # RGB Z stack: the sample axis is RGB, not channels or garbage
    assert img.shape == (1, 5, 3, 6, 7) and img.rgb and img.width == 7

    class NewScene:  # czifile >= 2026: scenes are separate objects whose axes use 'S' for samples
        def __init__(self, v):
            self.axes, self.v = "CZYX", v

        def asarray(self):
            return np.full((2, 3, 6, 7), self.v, np.uint16)

    class NewCzi:
        scenes = {"a": NewScene(1), "b": NewScene(2)}

        def __init__(self, p):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    _stub_module(monkeypatch, "czifile", CziFile=NewCzi)
    img = io.read_image(tmp_path / "x.czi", series=1)
    assert img.shape == (1, 3, 2, 6, 7) and img.metadata["series_count"] == 2 and int(img.data.max()) == 2


def test_lif_reads_every_time_point(tmp_path, monkeypatch):
    from collections import namedtuple
    Dims = namedtuple("Dims", "x y z t m")
    calls = []

    class FakeLifImage:
        def __init__(self, name, nt, nz, nc, h, w):
            self.name, self.dims, self.channels = name, Dims(w, h, nz, nt, 1), nc
            self.scale = (2.0, 2.0, 0.5, 4.0)  # px/µm for x, y, z; frames/s for t

        def get_frame(self, z=0, t=0, c=0, m=0):
            calls.append((t, z, c))
            return np.full((self.dims.y, self.dims.x), 100 * t + 10 * z + c, np.uint16)

    class FakeLif:
        def __init__(self, path):
            self.images = [FakeLifImage("first", 5, 2, 2, 6, 8), FakeLifImage("second", 1, 1, 1, 4, 4)]
            self.num_images = 2
            self.image_list = [{"name": im.name} for im in self.images]

        def get_image(self, i):
            return self.images[i]

    _stub_module(monkeypatch, "readlif")
    _stub_module(monkeypatch, "readlif.reader", LifFile=FakeLif)
    (tmp_path / "x.lif").write_bytes(b"\x00")
    img = io.read_image(tmp_path / "x.lif")
    assert img.shape == (5, 2, 2, 6, 8) and img.nt == 5
    assert len(set(calls)) == 20 and max(t for t, _, _ in calls) == 4
    for t, z, c in ((0, 0, 0), (4, 1, 1), (2, 0, 1)):
        assert int(img.data[t, z, c, 0, 0]) == 100 * t + 10 * z + c
    cal = img.calibration
    assert cal.pixel_width == 0.5 and cal.pixel_depth == 2.0 and cal.frame_interval == 0.25 and cal.time_unit == "sec"
    assert img.metadata["series_count"] == 2 and img.metadata["lif_image_name"] == "first"
    second = io.read_image(tmp_path / "x.lif", series=1)
    assert second.shape == (1, 1, 1, 4, 4) and second.metadata["lif_image_name"] == "second"
    assert io.read_image(tmp_path / "x.lif", series=99).metadata["lif_image_index"] == 1  # clamped like TIFF series
    infos = io.list_series(tmp_path / "x.lif")  # the Bio-Formats series list without reading the pixels
    assert [(i.name, i.size_x, i.size_y, i.size_z, i.size_c, i.size_t) for i in infos] == [("first", 8, 6, 2, 2, 5), ("second", 4, 4, 1, 1, 1)]
    assert infos[0].pixel_width == 0.5 and infos[0].time_increment == 0.25 and infos[0].format == "Leica LIF" and infos[0].dtype == "uint16"


# ----------------------------------------------------------------------------- series + resolution levels (HDF5 / Imaris / OME-Zarr)
def test_hdf5_datasets_and_imaris_levels_as_series(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "g.h5", "w") as hf:
        hf.create_dataset("small", data=np.full((3, 3), 7, np.uint8))
        hf.create_dataset("big", data=np.zeros((10, 10), np.uint8))
    infos = io.list_series(tmp_path / "g.h5")
    assert [(i.name, i.size_x) for i in infos] == [("big", 10), ("small", 3)] and infos[0].format == "HDF5"
    img = io.read_image(tmp_path / "g.h5", series=1)  # the k-th listed dataset
    assert img.shape == (1, 1, 1, 3, 3) and int(img.data.max()) == 7 and img.metadata["dataset"] == "small"
    # an Imaris-style file with two resolution levels
    rng = np.random.default_rng(5)
    full = rng.integers(0, 4000, (4, 8, 8)).astype(np.uint16)

    def sattr(g, k, v):
        g.attrs[k] = np.array([c.encode() for c in str(v)], dtype="S1")
    with h5py.File(tmp_path / "v.ims", "w") as hf:
        info = hf.create_group("DataSetInfo/Image")
        for k, v in (("X", 8), ("Y", 8), ("Z", 4), ("ExtMin0", 0), ("ExtMin1", 0), ("ExtMin2", 0), ("ExtMax0", 4.0), ("ExtMax1", 4.0), ("ExtMax2", 8.0), ("Unit", "um")):
            sattr(info, k, v)
        sattr(hf.create_group("DataSetInfo/Channel 0"), "Name", "Nuclei")
        for lvl, arr in ((0, full), (1, full[::2, ::2, ::2])):
            g = hf.create_group(f"DataSet/ResolutionLevel {lvl}/TimePoint 0/Channel 0")
            padded = np.zeros((arr.shape[0], arr.shape[1] + 2, arr.shape[2] + 3), arr.dtype)  # Imaris pads Data to chunk multiples
            padded[:, :arr.shape[1], :arr.shape[2]] = arr
            g.create_dataset("Data", data=padded)
            for k, v in (("ImageSizeX", arr.shape[2]), ("ImageSizeY", arr.shape[1]), ("ImageSizeZ", arr.shape[0])):
                sattr(g, k, v)
    infos = io.list_series(tmp_path / "v.ims")
    assert len(infos) == 1 and infos[0].levels == [(8, 8), (4, 4)] and infos[0].channel_names == ["Nuclei"] and infos[0].format.startswith("Imaris")
    assert infos[0].pixel_width == pytest.approx(0.5) and infos[0].pixel_depth == pytest.approx(2.0) and infos[0].unit == "µm"
    l0 = io.read_image(tmp_path / "v.ims")
    assert l0.shape == (1, 4, 1, 8, 8) and np.array_equal(l0.data[0, :, 0], full) and l0.metadata["levels"] == 2
    l1 = io.read_image(tmp_path / "v.ims", level=1)
    assert l1.shape == (1, 2, 1, 4, 4) and np.array_equal(l1.data[0, :, 0], full[::2, ::2, ::2]) and l1.calibration.pixel_width == pytest.approx(1.0)
    assert l1.calibration.pixel_depth == pytest.approx(4.0) and l1.metadata["level"] == 1


def test_ome_zarr_multiscale_levels_and_collections(tmp_path):
    zarr = pytest.importorskip("zarr")
    full = np.random.default_rng(6).integers(0, 255, (2, 16, 16)).astype(np.uint8)  # C Y X

    def write_image(path, data, name):
        try:
            g = zarr.open_group(str(path), mode="w", zarr_format=2)
        except TypeError:
            g = zarr.open_group(str(path), mode="w")
        for lvl, arr in ((0, data), (1, data[:, ::2, ::2])):
            if hasattr(g, "create_array"):
                a = g.create_array(str(lvl), shape=arr.shape, chunks=arr.shape, dtype=arr.dtype)
                a[...] = arr
            else:
                g.create_dataset(str(lvl), data=arr, chunks=arr.shape)
        g.attrs["multiscales"] = [{"version": "0.4", "name": name, "axes": [{"name": "c", "type": "channel"}, {"name": "y", "type": "space", "unit": "micrometer"},
                                                                            {"name": "x", "type": "space", "unit": "micrometer"}],
                                   "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 0.5, 0.5]}]},
                                                {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0]}]}]}]
        g.attrs["omero"] = {"channels": [{"label": "A", "color": "FF0000"}, {"label": "B", "color": "00FF00"}]}
    write_image(tmp_path / "img.zarr", full, "img")
    infos = io.list_series(tmp_path / "img.zarr")
    assert len(infos) == 1 and infos[0].levels == [(16, 16), (8, 8)] and infos[0].channel_names == ["A", "B"] and infos[0].format == "OME-Zarr"
    assert infos[0].pixel_width == pytest.approx(0.5) and infos[0].unit == "µm" and infos[0].size_c == 2
    l1 = io.read_image(tmp_path / "img.zarr", level=1)
    assert l1.shape == (1, 1, 2, 8, 8) and np.array_equal(l1.data[0, 0], full[:, ::2, ::2]) and l1.calibration.pixel_width == pytest.approx(1.0)
    assert l1.metadata["level"] == 1 and io.read_image(tmp_path / "img.zarr").calibration.pixel_width == pytest.approx(0.5)
    # a bioformats2raw collection: numbered sub-groups, each an image = one series
    coll = tmp_path / "coll.zarr"
    try:
        zarr.open_group(str(coll), mode="w", zarr_format=2)
    except TypeError:
        zarr.open_group(str(coll), mode="w")
    write_image(coll / "0", full, "first")
    write_image(coll / "1", full[:, :8, :12], "second")
    infos = io.list_series(coll)
    assert [(i.index, i.name, i.size_x, i.size_y) for i in infos] == [(0, "coll #0", 16, 16), (1, "coll #1", 12, 8)]
    second = io.read_image(coll, series=1)
    assert second.shape == (1, 1, 2, 8, 12) and second.metadata["series_count"] == 2 and second.metadata["series"] == 1
    assert io.read_image(coll, series=1, level=1).shape == (1, 1, 2, 4, 6)
