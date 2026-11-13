"""``ij.io.FileSaver`` / ``Opener`` and the Bio-Formats entry points ``loci.plugins.BF`` /
``loci.plugins.in.ImporterOptions`` – all over Lumen's readers and writers."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ._context import current
from ._images import ImagePlus, in_session, temporarily_in_session


class FileSaver:
    """``FileSaver(imp).saveAsTiff(path)`` … – every format Lumen writes (tif, png, jpg, gif, bmp, zip, raw, text)."""

    _jpeg_quality = 85

    def __init__(self, imp: ImagePlus) -> None:
        self._imp = imp

    def _save(self, path: Any, fmt: str | None) -> bool:
        ctx = current()
        img = self._imp.img
        p = os.fspath(path)
        with temporarily_in_session(ctx, img):
            ctx.session.save_image(img.id, p, fmt)
        self._imp.changes = False
        return True

    def save(self) -> bool:
        path = self._imp.img.path
        if not path:
            raise ValueError("FileSaver.save(): the image has no file path (use saveAsTiff(path))")
        return self._save(path, None)

    def saveAsTiff(self, path: Any = None) -> bool:
        return self._save(path or self._imp.img.path, "tif")

    saveAsTiffStack = saveAsTiff

    def saveAsPng(self, path: Any) -> bool:
        return self._save(path, "png")

    def saveAsJpeg(self, path: Any) -> bool:
        return self._save(path, "jpg")

    def saveAsGif(self, path: Any) -> bool:
        return self._save(path, "gif")

    def saveAsBmp(self, path: Any) -> bool:
        return self._save(path, "bmp")

    def saveAsPgm(self, path: Any) -> bool:
        return self._save(path, "pgm")

    def saveAsZip(self, path: Any) -> bool:
        ctx = current()
        with temporarily_in_session(ctx, self._imp.img):
            ctx.ij._run("file.export_zip", {"path": os.fspath(path)}, image_id=self._imp.img.id)
        return True

    def saveAsRaw(self, path: Any) -> bool:
        img = self._imp.img
        with open(os.fspath(path), "wb") as fh:
            fh.write(np.ascontiguousarray(img.data).tobytes())
        return True

    saveAsRawStack = saveAsRaw

    def saveAsText(self, path: Any) -> bool:
        img = self._imp.img
        d = img.display
        plane = img.data[d.t, d.z, d.c] if not img.rgb else img.data[d.t, d.z].astype(np.float32).mean(axis=0)
        np.savetxt(os.fspath(path), plane, fmt="%g", delimiter="\t")
        return True

    def saveAsFits(self, path: Any) -> bool:
        return self._save(path, "fits")

    def saveAsLut(self, path: Any) -> bool:
        ctx = current()
        with temporarily_in_session(ctx, self._imp.img):
            ctx.ij._run("file.export_lut", {"path": os.fspath(path)}, image_id=self._imp.img.id)
        return True

    @staticmethod
    def setJpegQuality(q: Any) -> None:
        FileSaver._jpeg_quality = int(q)

    @staticmethod
    def getJpegQuality() -> int:
        return FileSaver._jpeg_quality

    @staticmethod
    def okForGif(imp: Any) -> bool:
        return True

    @staticmethod
    def okForJpeg(imp: Any) -> bool:
        return True


class Opener:
    """``Opener().openImage(path)`` (no tab until ``show()``) / ``Opener().open(path)`` (opens and shows)."""

    UNKNOWN, TIFF, DICOM, FITS, PGM, JPEG, GIF, LUT, BMP, ZIP, JAVA_OR_TEXT, ROI, TEXT, PNG, TIFF_AND_DICOM, CUSTOM, AVI, OJJ, TABLE, RAW = range(20)

    def __init__(self) -> None:
        self._silent = False

    def openImage(self, *args: Any) -> ImagePlus | None:
        if len(args) == 2:
            path = os.path.join(str(args[0]), str(args[1]))
        elif args:
            path = os.fspath(args[0])
        else:
            raise ValueError("Opener.openImage(path) needs a path")
        from ..io import read_image
        try:
            return ImagePlus(_img=read_image(path))
        except FileNotFoundError:
            return None

    openTiff = openImage
    openZip = openImage
    openUsingBioFormats = openImage

    def open(self, *args: Any) -> None:  # noqa: A003
        if not args:
            raise ValueError("Opener.open(path) needs a path (no file dialog in scripts)")
        current().ij.open(os.fspath(args[0]))

    def openTiffStack(self, *a: Any) -> ImagePlus | None:
        return self.openImage(*a)

    def openRoi(self, path: Any) -> Any:
        ctx = current()
        before = len(ctx.session.roi_manager.rois)
        ctx.ij._run("file.import_ij_rois", {"path": os.fspath(path)})
        rois = ctx.session.roi_manager.rois[before:]
        from ._rois import Roi
        return Roi._wrap(rois[0]) if rois else None

    def setSilentMode(self, b: Any) -> None:
        self._silent = bool(b)

    def getFileType(self, path: Any) -> int:
        ext = os.path.splitext(os.fspath(path))[1].lower()
        return {".tif": self.TIFF, ".tiff": self.TIFF, ".dcm": self.DICOM, ".fits": self.FITS, ".fit": self.FITS, ".pgm": self.PGM, ".jpg": self.JPEG,
                ".jpeg": self.JPEG, ".gif": self.GIF, ".lut": self.LUT, ".bmp": self.BMP, ".zip": self.ZIP, ".roi": self.ROI, ".txt": self.TEXT,
                ".csv": self.TABLE, ".png": self.PNG, ".avi": self.AVI, ".raw": self.RAW}.get(ext, self.UNKNOWN)

    @staticmethod
    def getDefaultDirectory() -> str:
        return os.getcwd() + os.sep

    @staticmethod
    def setDefaultDirectory(d: Any) -> None:
        pass

    @staticmethod
    def openResultsTable(path: Any) -> Any:
        from ._tables import ResultsTable
        return ResultsTable.open(path)

    @staticmethod
    def openTable(path: Any) -> Any:
        from ._tables import ResultsTable
        return ResultsTable.open(path)


class ImporterOptions:
    """``loci.plugins.in.ImporterOptions``: ``setId`` / ``setSeriesOn`` / ``setOpenAllSeries`` / ``setVirtual`` …"""

    COLOR_MODE_DEFAULT, COLOR_MODE_COMPOSITE, COLOR_MODE_COLORIZED, COLOR_MODE_GRAYSCALE, COLOR_MODE_CUSTOM = "Default", "Composite", "Colorized", "Grayscale", "Custom"
    VIEW_HYPERSTACK, VIEW_STANDARD, VIEW_NONE = "Hyperstack", "Standard ImageJ", "Metadata only"

    def __init__(self) -> None:
        self.id: str | None = None
        self.series: dict[int, bool] = {}
        self.open_all = False
        self.virtual = False
        self.autoscale = True
        self.color_mode = self.COLOR_MODE_DEFAULT
        self.crop: tuple[int, int, int, int] | None = None
        self.split_channels = False

    def setId(self, path: Any) -> None:  # noqa: A003
        self.id = os.fspath(path)

    def getId(self) -> str | None:
        return self.id

    def setSeriesOn(self, i: Any, on: Any) -> None:
        self.series[int(i)] = bool(on)

    def isSeriesOn(self, i: Any) -> bool:
        return self.series.get(int(i), False)

    def setOpenAllSeries(self, b: Any) -> None:
        self.open_all = bool(b)

    def setVirtual(self, b: Any) -> None:
        self.virtual = bool(b)

    def setAutoscale(self, b: Any) -> None:
        self.autoscale = bool(b)

    def setColorMode(self, m: Any) -> None:
        self.color_mode = str(m)

    def setSplitChannels(self, b: Any) -> None:
        self.split_channels = bool(b)

    def setCrop(self, b: Any) -> None:
        pass

    def setCropRegion(self, i: Any, region: Any) -> None:
        self.crop = (int(region.x), int(region.y), int(region.width), int(region.height))

    def setStackFormat(self, f: Any) -> None:
        pass

    def setWindowless(self, b: Any) -> None:
        pass

    def setQuiet(self, b: Any) -> None:
        pass

    def setGroupFiles(self, b: Any) -> None:
        pass

    def setConcatenate(self, b: Any) -> None:
        pass

    def setShowMetadata(self, b: Any) -> None:
        pass

    def setShowOMEXML(self, b: Any) -> None:
        pass

    def setStitchTiles(self, b: Any) -> None:
        pass

    def setSwapDimensions(self, b: Any) -> None:
        pass

    def setSpecifyRanges(self, b: Any) -> None:
        pass

    def setCBegin(self, *a: Any) -> None:
        pass

    def setCEnd(self, *a: Any) -> None:
        pass

    def setCStep(self, *a: Any) -> None:
        pass

    def setZBegin(self, *a: Any) -> None:
        pass

    def setZEnd(self, *a: Any) -> None:
        pass

    def setTBegin(self, *a: Any) -> None:
        pass

    def setTEnd(self, *a: Any) -> None:
        pass


class BF:
    """``loci.plugins.BF.openImagePlus(path | ImporterOptions)`` → ``[ImagePlus, …]`` (one per selected series)."""

    @staticmethod
    def openImagePlus(arg: Any) -> list[ImagePlus]:
        from ..io import read_image
        if isinstance(arg, ImporterOptions):
            if not arg.id:
                raise ValueError("BF.openImagePlus: call options.setId(path) first")
            path = arg.id
            first = read_image(path, virtual=arg.virtual)
            names = first.metadata.get("series_names") or []
            n = max(len(names), 1)
            wanted = [i for i in range(n) if arg.open_all or arg.series.get(i, i == 0 and not arg.series)]
            imps: list[ImagePlus] = []
            for i in wanted:
                img = first if i == int(first.metadata.get("series", 0)) else read_image(path, series=i, virtual=arg.virtual)
                if arg.crop is not None:
                    x, y, w, h = arg.crop
                    img = img.copy(data=np.ascontiguousarray(img.data[:, :, :, y:y + h, x:x + w]))
                imps.append(ImagePlus(_img=img))
            return imps
        return [ImagePlus(_img=read_image(os.fspath(arg)))]

    @staticmethod
    def openThumbImagePlus(arg: Any) -> list[ImagePlus]:
        return BF.openImagePlus(arg)


class ImageReader:
    """``loci.formats.ImageReader``: ``setId`` + the size accessors through the Bio-Formats macro extensions."""

    def __init__(self) -> None:
        self._ext = None

    def _e(self) -> Any:
        return current().ij.Ext

    def setId(self, path: Any) -> None:  # noqa: A003
        self._e().setId(os.fspath(path))

    def getSeriesCount(self) -> int:
        return self._e().getSeriesCount()

    def setSeries(self, i: Any) -> None:
        self._e().setSeries(i)

    def getSeries(self) -> int:
        return self._e().getSeries()

    def getSizeX(self) -> int:
        return self._e().getSizeX()

    def getSizeY(self) -> int:
        return self._e().getSizeY()

    def getSizeZ(self) -> int:
        return self._e().getSizeZ()

    def getSizeC(self) -> int:
        return self._e().getSizeC()

    def getSizeT(self) -> int:
        return self._e().getSizeT()

    def getImageCount(self) -> int:
        return self._e().getImageCount()

    def getPixelType(self) -> str:
        return self._e().getPixelType()

    def getDimensionOrder(self) -> str:
        return self._e().getDimensionOrder()

    def isRGB(self) -> bool:
        return bool(self._e().isInterleaved())

    def getFormat(self) -> str:
        return self._e().getFormat(self._e().getCurrentFile())

    def close(self) -> None:
        self._e().close()

    def getMetadataValue(self, key: Any) -> str:
        return self._e().getMetadataValue(key)

    def getSeriesMetadataValue(self, key: Any) -> str:
        return self._e().getSeriesMetadataValue(key)


__all__ = ["FileSaver", "Opener", "ImporterOptions", "BF", "ImageReader", "in_session"]
