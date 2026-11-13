"""The in-memory image model.

Every image in Lumen is a 5-D numpy array with axes (T, Z, C, Y, X) — the ImageJ
hyperstack layout — plus calibration, per-channel display settings and free-form
metadata.  RGB images are stored as C = 3 uint8 with ``rgb=True``.
"""
from __future__ import annotations

import itertools
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Iterator

import numpy as np

AXES = "TZCYX"
SUPPORTED_DTYPES = (np.uint8, np.uint16, np.float32)

DEFAULT_CHANNEL_LUTS = ["Grays", "Green", "Magenta", "Cyan", "Yellow", "Red", "Blue"]
MULTI_CHANNEL_LUTS = ["Cyan", "Magenta", "Yellow", "Red", "Green", "Blue", "Grays"]
RGB_LUTS = ["Red", "Green", "Blue"]


def _default_lut_for(channel: int, nchannels: int, rgb: bool) -> str:
    if rgb:
        return RGB_LUTS[channel % 3]
    if nchannels == 1:
        return "Grays"
    return MULTI_CHANNEL_LUTS[channel % len(MULTI_CHANNEL_LUTS)]


@dataclass
class Calibration:
    pixel_width: float = 1.0
    pixel_height: float = 1.0
    pixel_depth: float = 1.0
    unit: str = "pixel"
    frame_interval: float = 0.0
    time_unit: str = "sec"
    value_unit: str = ""

    @property
    def scaled(self) -> bool:
        return self.unit not in ("pixel", "pixels", "", None) and self.pixel_width > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChannelDisplay:
    name: str = ""
    lut: str = "Grays"
    min: float = 0.0
    max: float = 255.0
    visible: bool = True
    inverted: bool = False
    gamma: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Display:
    channels: list[ChannelDisplay] = field(default_factory=list)
    mode: str = "composite"  # composite | single | grayscale
    t: int = 0
    z: int = 0
    c: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "channels": [c.to_dict() for c in self.channels],
            "mode": self.mode,
            "t": self.t,
            "z": self.z,
            "c": self.c,
        }


class LumenImage:
    """A hyperstack: (T, Z, C, Y, X) array plus calibration and display settings."""

    def __init__(
        self,
        data: np.ndarray,
        name: str = "Untitled",
        axes: str | None = None,
        rgb: bool = False,
        calibration: Calibration | None = None,
        metadata: dict[str, Any] | None = None,
        path: str | None = None,
        channel_names: list[str] | None = None,
    ):
        self.id = uuid.uuid4().hex[:10]
        self.name = name
        self.path = path
        self.rgb = rgb
        self.metadata: dict[str, Any] = metadata or {}
        self.calibration = calibration or Calibration()
        self.data = self.to_canonical(data, axes, rgb=rgb)
        self.display = Display()
        self.lut_is_binary = False
        self.overlay: list[dict[str, Any]] = []  # non-destructive drawn shapes
        self.rev = 0  # bumped whenever pixel data changes (client cache key)
        self.reset_display(channel_names)

    # ------------------------------------------------------------------ layout
    @staticmethod
    def to_canonical(data: np.ndarray, axes: str | None = None, rgb: bool = False) -> np.ndarray:
        """Return a contiguous (T, Z, C, Y, X) array of a supported dtype."""
        data = np.asarray(data)
        if axes is None:
            if data.ndim == 2:
                axes = "YX"
            elif data.ndim == 3:
                axes = "YXS" if (rgb or (data.shape[-1] in (3, 4) and data.dtype == np.uint8)) else "ZYX"
            elif data.ndim == 4:
                axes = "ZCYX"
            elif data.ndim == 5:
                axes = "TZCYX"
            else:
                raise ValueError(f"Unsupported image dimensionality: {data.shape}")
        axes = axes.upper()
        if len(axes) != data.ndim:
            raise ValueError(f"axes {axes!r} do not match shape {data.shape}")
        # RGB sample axis → treat as channel axis
        if "S" in axes:
            if data.shape[axes.index("S")] == 4:  # drop alpha
                data = np.take(data, [0, 1, 2], axis=axes.index("S"))
            axes = axes.replace("S", "C")
        # Unknown axes (e.g. 'Q', 'I') – merge into Z
        for ch in axes:
            if ch not in AXES:
                axes = axes.replace(ch, "Z" if "Z" not in axes else "T")
        # add missing axes
        for ch in AXES:
            if ch not in axes:
                data = np.expand_dims(data, 0)
                axes = ch + axes
        order = [axes.index(ch) for ch in AXES]
        data = np.transpose(data, order)
        data = LumenImage.coerce_dtype(data)
        return np.ascontiguousarray(data)

    @staticmethod
    def coerce_dtype(data: np.ndarray) -> np.ndarray:
        if not data.dtype.isnative:
            # '>u2' / '>f4' … (big-endian TIFF memmaps, FITS): byte-swap to the native order (a copy) instead of
            # silently promoting to float32 – a .view(newbyteorder) would reinterpret the bytes, so astype it is
            data = data.astype(data.dtype.newbyteorder("="))
        if data.dtype == np.bool_:
            return data.astype(np.uint8) * 255
        if data.dtype in (np.uint8, np.uint16, np.float32):
            return data
        if data.dtype == np.int8:
            return (data.astype(np.int16) + 128).astype(np.uint8)
        if data.dtype in (np.int16,):
            return data.astype(np.float32)
        if data.dtype in (np.int32, np.int64, np.uint32, np.uint64, np.float64, np.float16):
            return data.astype(np.float32)
        return data.astype(np.float32)

    # ------------------------------------------------------------- properties
    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return tuple(self.data.shape)  # type: ignore[return-value]

    @property
    def nt(self) -> int:
        return self.data.shape[0]

    @property
    def nz(self) -> int:
        return self.data.shape[1]

    @property
    def nc(self) -> int:
        return self.data.shape[2]

    @property
    def height(self) -> int:
        return self.data.shape[3]

    @property
    def width(self) -> int:
        return self.data.shape[4]

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def bit_depth(self) -> str:
        if self.rgb:
            return "RGB"
        # keyed on the native-order dtype so a stray non-native array still answers instead of raising KeyError
        return {np.dtype(np.uint8): "8-bit", np.dtype(np.uint16): "16-bit", np.dtype(np.float32): "32-bit"}[np.dtype(self.data.dtype).newbyteorder("=")]

    @property
    def is_stack(self) -> bool:
        return self.nz > 1 or self.nt > 1

    @property
    def is_binary(self) -> bool:
        if self.data.dtype != np.uint8:
            return False
        sy, sx = max(1, self.height // 64), max(1, self.width // 64)
        if self.is_virtual:  # a virtual stack: judge the current plane instead of reading every plane of the file
            u = np.unique(self.current_plane()[::sy, ::sx])
        else:
            u = np.unique(self.data[:, :, :, ::sy, ::sx])
        return bool(set(u.tolist()) <= {0, 255})

    @property
    def nbytes(self) -> int:
        """Bytes the pixels hold in process memory: the array size, or – for a virtual stack – only its cached planes
        (a file-backed memmap counts 0: its pages belong to the OS page cache).  ``full_bytes`` is the decoded size."""
        from .lazy import resident_bytes
        return resident_bytes(self.data)

    @property
    def full_bytes(self) -> int:
        """Decoded size of the whole (T, Z, C, Y, X) stack – what materialising a virtual stack costs."""
        from .lazy import full_bytes
        return full_bytes(self.data)

    @property
    def is_virtual(self) -> bool:
        """True while the pixels still live in the file (a lazy array or memmap, see :mod:`lumen.core.lazy`)."""
        from .lazy import is_virtual
        return is_virtual(self.data)

    # ----------------------------------------------------------------- planes
    def plane(self, t: int = 0, z: int = 0, c: int = 0) -> np.ndarray:
        return self.data[t, z, c]

    def set_plane(self, arr: np.ndarray, t: int = 0, z: int = 0, c: int = 0) -> None:
        self.data[t, z, c] = arr

    def iter_planes(self) -> Iterator[tuple[int, int, int]]:
        return itertools.product(range(self.nt), range(self.nz), range(self.nc))

    def current_plane(self) -> np.ndarray:
        d = self.display
        return self.data[d.t, d.z, d.c]

    def clamp_position(self) -> None:
        d = self.display
        d.t = min(max(d.t, 0), self.nt - 1)
        d.z = min(max(d.z, 0), self.nz - 1)
        d.c = min(max(d.c, 0), self.nc - 1)

    # ---------------------------------------------------------------- display
    def value_range(self) -> tuple[float, float]:
        if self.data.dtype == np.uint8:
            return 0.0, 255.0
        if self.data.dtype == np.uint16:
            return 0.0, 65535.0
        # a virtual float stack: the range of the current plane (scanning every plane would read the whole file)
        src = self.current_plane() if self.is_virtual else self.data
        finite = src[np.isfinite(src)]
        if finite.size == 0:
            return 0.0, 1.0
        return float(finite.min()), float(finite.max())

    def auto_range(self, c: int, saturated: float = 0.35) -> tuple[float, float]:
        """ImageJ-style auto contrast: clip `saturated` percent of pixels (total)."""
        sub = self.data[self.display.t, self.display.z, c]
        if sub.size > 2_000_000:
            step = int(np.sqrt(sub.size / 2_000_000)) + 1
            sub = sub[::step, ::step]
        finite = sub[np.isfinite(sub)] if sub.dtype == np.float32 else sub.ravel()
        if finite.size == 0:
            return 0.0, 1.0
        lo = float(np.percentile(finite, saturated / 2))
        hi = float(np.percentile(finite, 100 - saturated / 2))
        if hi <= lo:
            lo, hi = float(finite.min()), float(finite.max())
            if hi <= lo:
                hi = lo + 1
        return lo, hi

    def reset_display(self, channel_names: list[str] | None = None, keep_luts: bool = False) -> None:
        self.clamp_position()  # the data shape may have changed; auto_range() indexes the current position
        old = {i: ch for i, ch in enumerate(self.display.channels)}
        chans = []
        for c in range(self.nc):
            lut = old[c].lut if (keep_luts and c in old) else _default_lut_for(c, self.nc, self.rgb)
            if self.data.dtype == np.uint8:
                lo, hi = 0.0, 255.0
            elif self.data.dtype == np.uint16:
                lo, hi = self.auto_range(c)
            else:
                lo, hi = self.auto_range(c, saturated=0.0)
            name = (channel_names[c] if channel_names and c < len(channel_names) else (old[c].name if c in old else f"C{c + 1}"))
            chans.append(ChannelDisplay(name=name, lut=lut, min=lo, max=hi, visible=True))
        self.display.channels = chans
        if self.nc == 1:
            self.display.mode = "single"
        self.clamp_position()

    def sync_channels(self) -> None:
        """Keep the channel display list in step with the data after a data change."""
        self.clamp_position()  # must run before auto_range() reads t/z/c
        if len(self.display.channels) != self.nc:
            self.reset_display(keep_luts=True)
        else:
            lo, hi = self.value_range()
            for c, ch in enumerate(self.display.channels):
                if self.data.dtype == np.uint8:
                    ch.min, ch.max = 0.0, 255.0
                elif ch.min < lo or ch.max > hi or ch.max <= ch.min:
                    ch.min, ch.max = self.auto_range(c)
        self.clamp_position()

    # --------------------------------------------------------------- helpers
    def copy(self, name: str | None = None, data: np.ndarray | None = None) -> "LumenImage":
        img = LumenImage.__new__(LumenImage)
        img.id = uuid.uuid4().hex[:10]
        img.name = name or self.name
        img.path = None
        img.rgb = self.rgb
        img.metadata = dict(self.metadata)
        img.calibration = Calibration(**self.calibration.to_dict())
        img.data = np.ascontiguousarray(data if data is not None else self.data.copy())
        img.display = Display(
            channels=[ChannelDisplay(**c.to_dict()) for c in self.display.channels],
            mode=self.display.mode, t=self.display.t, z=self.display.z, c=self.display.c,
        )
        img.lut_is_binary = self.lut_is_binary
        img.overlay = [dict(o) for o in self.overlay]
        img.rev = 0
        if data is not None:
            img.rgb = self.rgb and data.shape[2] == 3 and data.dtype == np.uint8
            img.sync_channels()
        return img

    def replace_data(self, data: np.ndarray, rgb: bool | None = None) -> None:
        self.data = np.ascontiguousarray(self.coerce_dtype(data))
        self.rev += 1
        if rgb is not None:
            self.rgb = rgb
        elif not (self.data.shape[2] == 3 and self.data.dtype == np.uint8):
            self.rgb = False
        self.sync_channels()

    def summary(self) -> dict[str, Any]:
        cal = self.calibration
        return {
            "id": self.id,
            "rev": self.rev,
            "name": self.name,
            "path": self.path,
            "shape": {"t": self.nt, "z": self.nz, "c": self.nc, "y": self.height, "x": self.width},
            "dtype": str(self.data.dtype),
            "bit_depth": self.bit_depth,
            "rgb": self.rgb,
            "calibration": cal.to_dict(),
            "display": self.display.to_dict(),
            "value_range": list(self.value_range()),
            "nbytes": self.nbytes,
            # data larger than RAM: a virtual stack reports what it holds in memory and what the whole stack would cost
            "virtual": self.is_virtual,
            "resident_bytes": self.nbytes,
            "full_bytes": self.full_bytes,
            "virtual_source": __import__("lumen.core.lazy", fromlist=["describe"]).describe(self.data) if self.is_virtual else None,
            "metadata": {k: v for k, v in self.metadata.items() if isinstance(v, (str, int, float, bool))},
            "overlay": self.overlay,
            "binary_hint": self.data.dtype == np.uint8 and self.lut_is_binary,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<LumenImage {self.name!r} {self.shape} {self.data.dtype}>"
