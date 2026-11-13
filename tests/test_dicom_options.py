"""The DICOM reader honours the Edit ▸ Options ▸ DICOM preferences (edit.options_dicom → session settings)."""
import numpy as np
import pytest

from lumen.core import io
from lumen.core import ops  # noqa: F401
from lumen.core.session import SESSION, Session

pydicom = pytest.importorskip("pydicom")

DICOM_KEYS = ("dicom_open_as_32bit", "dicom_ignore_rescale", "dicom_calibrate_images", "dicom_window_level_on_open")


def _write_ct(path, value=1500, slope=1.0, intercept=-1024.0, window=(40.0, 400.0), instance=1, z=0.0):
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = pydicom.uid.CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = pydicom.uid.CTImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality, ds.PatientName = "CT", "Test"
    ds.InstanceNumber = instance
    ds.Rows, ds.Columns = 6, 8
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.PixelSpacing, ds.SliceThickness = [0.5, 0.75], 2.0
    ds.ImagePositionPatient = [0, 0, z]
    ds.RescaleSlope, ds.RescaleIntercept = slope, intercept
    ds.WindowCenter, ds.WindowWidth = window
    ds.PixelData = np.full((6, 8), value, np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return path


@pytest.fixture
def ct(tmp_path):
    return _write_ct(tmp_path / "ct.dcm")


@pytest.fixture
def clean_session_settings():
    saved = {k: SESSION.settings[k] for k in DICOM_KEYS if k in SESSION.settings}
    for k in DICOM_KEYS:
        SESSION.settings.pop(k, None)
    yield SESSION.settings
    for k in DICOM_KEYS:
        SESSION.settings.pop(k, None)
    SESSION.settings.update(saved)


def test_defaults_keep_native_dtype_and_calibrate(ct, clean_session_settings):
    img = io.read_image(ct)
    assert img.dtype == np.uint16 and int(img.data.max()) == 1500          # raw stored values
    assert img.metadata["RescaleIntercept"] == -1024.0 and img.metadata["RescaleSlope"] == 1.0
    assert img.calibration.unit == "mm" and (img.calibration.pixel_width, img.calibration.pixel_height, img.calibration.pixel_depth) == (0.75, 0.5, 2.0)
    ch = img.display.channels[0]
    # the window (40 ± 200 HU) is mapped back to stored values for a raw image
    assert (ch.min, ch.max) == (40 - 200 + 1024, 40 + 200 + 1024)


def test_open_as_32bit_applies_rescale(ct, clean_session_settings):
    img = io.read_image(ct, dicom_options={"open_as_32bit": True})
    assert img.dtype == np.float32 and float(img.data.max()) == 1500 - 1024
    ch = img.display.channels[0]
    assert (ch.min, ch.max) == (-160.0, 240.0)                                   # window in HU
    img = io.read_image(ct, dicom_options={"open_as_32bit": True, "ignore_rescale": True})
    assert img.dtype == np.float32 and float(img.data.max()) == 1500
    assert (img.display.channels[0].min, img.display.channels[0].max) == (864.0, 1264.0)
    # the prefixed key spelling works too
    img = io.read_image(ct, dicom_options={"dicom_open_as_32bit": True})
    assert img.dtype == np.float32


def test_calibrate_and_window_level_switches(ct, clean_session_settings):
    img = io.read_image(ct, dicom_options={"calibrate_images": False})
    assert img.calibration.unit == "pixel" and img.calibration.pixel_width == 1.0 and img.calibration.pixel_depth == 1.0
    img = io.read_image(ct, dicom_options={"window_level_on_open": False})
    ch = img.display.channels[0]
    assert (ch.min, ch.max) != (864.0, 1264.0)


def test_session_settings_via_options_dicom(ct, clean_session_settings, tmp_path):
    s = Session()
    s.run_op("edit.options_dicom", {"open_as_32bit": True, "calibrate_images": False, "window_level_on_open": False})
    # a private Session does not affect the reader – only the global SESSION preferences do
    assert io.read_image(ct).dtype == np.uint16
    SESSION.settings.update({k: s.settings[k] for k in DICOM_KEYS})
    img = io.read_image(ct)
    assert img.dtype == np.float32 and float(img.data.max()) == 476.0 and img.calibration.unit == "pixel"
    # explicit overrides win over the session settings
    assert io.read_image(ct, dicom_options={"open_as_32bit": False}).dtype == np.uint16
    # directories honour the same preferences
    d = tmp_path / "series"
    d.mkdir()
    for i in range(3):
        _write_ct(d / f"s{i}.dcm", value=1000 + i, instance=i + 1, z=2.0 * i)
    vol = io.read_image(d)
    assert vol.dtype == np.float32 and vol.nz == 3 and [float(vol.data[0, z, 0, 0, 0]) for z in range(3)] == [-24.0, -23.0, -22.0]
    assert vol.calibration.unit == "pixel"
    SESSION.settings.update({"dicom_open_as_32bit": False, "dicom_calibrate_images": True})
    vol = io.read_image(d)
    assert vol.dtype == np.uint16 and vol.calibration.pixel_depth == 2.0 and vol.calibration.unit == "mm"


def test_dicom_reader_options_helper(clean_session_settings):
    assert io.dicom_reader_options() == io.DICOM_OPTION_DEFAULTS
    SESSION.settings["dicom_ignore_rescale"] = True
    assert io.dicom_reader_options()["dicom_ignore_rescale"] is True
    assert io.dicom_reader_options({"ignore_rescale": False, "bogus": True})["dicom_ignore_rescale"] is False
