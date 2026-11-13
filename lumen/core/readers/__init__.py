"""Native (pure-Python) readers for proprietary microscopy formats that otherwise need Java Bio-Formats, plus the
optional Java Bio-Formats bridge.

* :mod:`.dm3` – Gatan DigitalMicrograph DM3 / DM4 (tag-directory parser: data type, dimensions, calibration, stacks)
* :mod:`.zvi` – Zeiss AxioVision ZVI (OLE compound file: image items, pixel format, scaling, Z / C / T stacks)
* :mod:`.lei` – Leica LEI (TCS / LCS header listing the TIFF planes with dimensions and calibration)
* :mod:`.bioformats_bridge` – everything else through ``bioio-bioformats`` + a JDK (the ``[bioformats]`` extra)

Every reader module exposes ``list_series(path) -> list[NativeSeries]`` and ``read(path, series=0) ->
(ndarray, NativeSeries)``; :mod:`lumen.core.io` turns the results into :class:`~lumen.core.io.SeriesInfo` /
:class:`~lumen.core.image.LumenImage`.  The modules are imported lazily by ``io`` so that importing Lumen never
pays for them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class NativeSeries:
    """What a native reader knows about one image series before (or after) reading its pixels."""
    index: int = 0
    name: str = ""
    #: storage shape in ``axes`` order
    shape: tuple[int, ...] = ()
    #: axes letters of ``shape`` (subset / permutation of ``TZCYX``, ``S`` for interleaved RGB samples)
    axes: str = "YX"
    dtype: Any = np.uint8
    rgb: bool = False
    pixel_width: float | None = None
    pixel_height: float | None = None
    pixel_depth: float | None = None
    unit: str = ""
    frame_interval: float | None = None
    time_unit: str = ""
    channel_names: list[str] = field(default_factory=list)
    format: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def size(self, axis: str) -> int:
        return int(self.shape[self.axes.index(axis)]) if axis in self.axes else 1


#: (format name, extensions, module) of the native readers – for the Bio-Formats Plugins Configuration table
NATIVE_READERS: list[tuple[str, str, str]] = [
    ("Gatan DigitalMicrograph DM3 / DM4", "dm3 dm4", "lumen.core.readers.dm3"),
    ("Zeiss AxioVision ZVI", "zvi", "lumen.core.readers.zvi"),
    ("Leica LEI (TCS / LCS TIFF series)", "lei", "lumen.core.readers.lei"),
]

#: file extension -> reader module name (relative to this package)
READER_BY_EXT: dict[str, str] = {".dm3": "dm3", ".dm4": "dm3", ".zvi": "zvi", ".lei": "lei"}
