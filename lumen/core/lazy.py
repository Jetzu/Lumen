"""Lazy (out-of-core) 5-D image arrays: data larger than RAM.

A :class:`LazyArray` behaves like a read-mostly ``(T, Z, C, Y, X)`` numpy array whose planes stay in the file until
they are needed.  Planes come from a :class:`PlaneSource` – a chunked store with numpy-style indexing (an HDF5
dataset, a Zarr / OME-Zarr array, a NumPy ``.npy`` memmap, a 5-D memmap view of an uncompressed TIFF), the per-channel
datasets of an Imaris resolution level, or the page reader of a TIFF series – and pass through one process-wide LRU
plane cache whose size is the preference ``virtual_cache_mb`` (default 512 MB).

Contract (shared with ``lumen.core.io.VirtualStack``, the TIFF specialisation, and :func:`is_virtual`):

* ``arr[t, z, c]`` reads one plane; slices read only the planes (and, on chunked stores, only the region) they cover;
  the viewer, Z/T profiles, projections, statistics and exports therefore never hold the whole stack.
* anything that needs the whole array (``np.asarray``, reductions such as ``.max()``, arithmetic, in-place writes)
  *materialises* it once – copy-on-write: the first pixel-modifying command turns the image into an ordinary
  in-memory array, undo and Save then work as usual.
* ``nbytes`` is the *resident* size (cached planes, or the materialised array), ``full_bytes`` the decoded size.
* ``fresh()`` re-opens the same source (a handle for undo snapshots that never shares the cache or a materialised copy).

``read_image(path, virtual=True)`` builds these for TIFF, HDF5 / Imaris, Zarr / OME-Zarr and ``.npy`` files, and
``read_image(path, virtual="auto")`` does so automatically when the decoded size exceeds the preference
``auto_virtual_mb`` (default: half of the physical RAM, see :func:`auto_virtual_bytes`).
"""
from __future__ import annotations

import itertools
import os
import sys
import threading
from collections import OrderedDict
from typing import Any, Callable, Iterator

import numpy as np

from .image import LumenImage

#: default size of the shared plane cache (the ``virtual_cache_mb`` preference)
DEFAULT_CACHE_MB = 512
#: fallback physical memory when the platform does not report it
_FALLBACK_RAM = 8 * 1024 ** 3

_PLANE_AXES = "TZC"


# ============================================================================ memory
def physical_memory_bytes() -> int:
    """Installed physical RAM in bytes (``os.sysconf`` on POSIX / macOS, ``GlobalMemoryStatusEx`` on Windows)."""
    try:
        if hasattr(os, "sysconf"):
            names = os.sysconf_names
            if "SC_PAGE_SIZE" in names and "SC_PHYS_PAGES" in names:
                v = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
                if v > 0:
                    return v
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):  # type: ignore[attr-defined]
                return int(st.ullTotalPhys)
        except Exception:  # noqa: BLE001
            pass
    return _FALLBACK_RAM


def default_auto_virtual_mb() -> int:
    """Default of the ``auto_virtual_mb`` preference: half of the physical RAM, in MB."""
    return max(1, int(physical_memory_bytes() // 2 // (1024 ** 2)))


def _setting(key: str) -> Any:
    try:
        from .session import SESSION  # imported here: session imports io which imports this module
        return SESSION.settings.get(key)
    except Exception:  # noqa: BLE001
        return None


def auto_virtual_bytes(settings: dict[str, Any] | None = None) -> int:
    """Decoded size above which ``read_image(virtual="auto")`` opens a file lazily: the preference ``auto_virtual_mb``
    (0 = never) of ``settings`` or of the global session, else half of the physical RAM."""
    v = settings.get("auto_virtual_mb") if settings is not None else _setting("auto_virtual_mb")
    if v is None:
        return physical_memory_bytes() // 2
    try:
        mb = float(v)
    except (TypeError, ValueError):
        return physical_memory_bytes() // 2
    return int(max(mb, 0) * 1024 ** 2) if mb > 0 else sys.maxsize


# ============================================================================ the shared plane cache
class PlaneCache:
    """Process-wide LRU cache of decoded planes, bounded in bytes; entries belong to an owner (one LazyArray)."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget = int(budget_bytes)
        self._lock = threading.Lock()
        self._entries: "OrderedDict[tuple[int, tuple[int, int, int]], np.ndarray]" = OrderedDict()
        self._owner_bytes: dict[int, int] = {}
        self.total = 0
        self._ids = itertools.count(1)

    def new_owner(self) -> int:
        return next(self._ids)

    def set_budget(self, budget_bytes: int) -> None:
        with self._lock:
            self.budget = max(0, int(budget_bytes))
            self._evict()

    def get(self, owner: int, key: tuple[int, int, int]) -> np.ndarray | None:
        k = (owner, key)
        with self._lock:
            hit = self._entries.get(k)
            if hit is not None:
                self._entries.move_to_end(k)
            return hit

    def put(self, owner: int, key: tuple[int, int, int], plane: np.ndarray) -> None:
        n = int(plane.nbytes)
        if n > self.budget:  # a plane bigger than the whole budget would only evict everything else
            return
        k = (owner, key)
        with self._lock:
            old = self._entries.pop(k, None)
            if old is not None:
                self.total -= int(old.nbytes)
                self._owner_bytes[owner] = self._owner_bytes.get(owner, 0) - int(old.nbytes)
            self._entries[k] = plane
            self.total += n
            self._owner_bytes[owner] = self._owner_bytes.get(owner, 0) + n
            self._evict()

    def _evict(self) -> None:
        while self.total > self.budget and self._entries:
            (owner, _key), plane = self._entries.popitem(last=False)
            self.total -= int(plane.nbytes)
            self._owner_bytes[owner] = self._owner_bytes.get(owner, 0) - int(plane.nbytes)

    def drop(self, owner: int) -> None:
        with self._lock:
            for k in [k for k in self._entries if k[0] == owner]:
                plane = self._entries.pop(k)
                self.total -= int(plane.nbytes)
            self._owner_bytes.pop(owner, None)

    def owner_bytes(self, owner: int) -> int:
        with self._lock:
            return max(0, int(self._owner_bytes.get(owner, 0)))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._owner_bytes.clear()
            self.total = 0


PLANE_CACHE = PlaneCache(DEFAULT_CACHE_MB * 1024 ** 2)


def cache_budget_bytes() -> int:
    return PLANE_CACHE.budget


def set_cache_budget(mb: float) -> int:
    """Apply the ``virtual_cache_mb`` preference; returns the budget in bytes."""
    try:
        mb = float(mb)
    except (TypeError, ValueError):
        mb = DEFAULT_CACHE_MB
    PLANE_CACHE.set_budget(int(max(mb, 1) * 1024 ** 2))
    return PLANE_CACHE.budget


# ============================================================================ axis mapping
def map_axes(axes: str, shape: tuple[int, ...], rgb: bool) -> tuple[list[tuple[int, str, int]], tuple[int, int], int, tuple[int, int, int, int, int]]:
    """Map a reader's axes string onto Lumen's (T, Z, C, Y, X) without copying: returns ``(lead, (Y, X), samples, shape5)``
    where ``lead`` lists ``(source axis index, T|Z|C, size)`` for every non-spatial axis (unknown axes become Z, then T,
    like ``LumenImage.to_canonical``) and ``samples`` is the size of an RGB(A) sample axis ``S`` (0 = none)."""
    axes_u = str(axes).upper()
    shp = tuple(int(v) for v in shape)
    if "Y" not in axes_u or "X" not in axes_u or len(axes_u) != len(shp):
        raise ValueError(f"virtual stack: unsupported axes {axes_u!r} for shape {shp}")
    yx = (shp[axes_u.index("Y")], shp[axes_u.index("X")])
    s_i = axes_u.index("S") if "S" in axes_u else None
    samples = shp[s_i] if s_i is not None else 0
    lead = [(i, ch) for i, ch in enumerate(axes_u) if ch not in "YXS"]
    present = {ch for ch in axes_u if ch in _PLANE_AXES} | ({"C"} if s_i is not None else set())
    letters: list[str] = []
    for _i, ch in lead:
        if ch in _PLANE_AXES:
            letters.append(ch)
        else:
            lt = "Z" if "Z" not in present else "T"
            present.add(lt)
            letters.append(lt)
    if len(set(letters)) != len(letters) or (s_i is not None and "C" in letters):
        raise ValueError(f"virtual stack: axes {axes_u!r} cannot be mapped onto T/Z/C without merging")
    lead_sizes = [(i, lt, shp[i]) for (i, _ch), lt in zip(lead, letters)]
    sizes = {"T": 1, "Z": 1, "C": 1}
    for _i, lt, n in lead_sizes:
        sizes[lt] = n
    if s_i is not None:
        sizes["C"] = 3 if rgb else samples
    return lead_sizes, yx, samples, (sizes["T"], sizes["Z"], sizes["C"], yx[0], yx[1])


def _native(plane: Any) -> np.ndarray:
    """A plane in a supported native dtype (``LumenImage.coerce_dtype``), contiguous when 2-D; 0-d results stay 0-d."""
    arr = np.asarray(plane)
    if arr.ndim >= 1:
        arr = np.ascontiguousarray(arr)
    return LumenImage.coerce_dtype(arr)


# ============================================================================ plane sources
class PlaneSource:
    """Where the planes of a :class:`LazyArray` come from.  Subclasses read one ``(Y, X)`` plane (in the file's own
    dtype) and may read sub-regions directly (``region_reads``); ``reopen()`` returns an independent handle."""
    #: (T, Z, C, Y, X)
    shape: tuple[int, int, int, int, int]
    src_dtype: np.dtype
    rgb: bool = False
    path: str | None = None
    description: str = "planes on demand"
    region_reads: bool = False

    def read_plane(self, t: int, z: int, c: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def read_region(self, t: int, z: int, c: int, ys: Any, xs: Any) -> np.ndarray:
        return np.asarray(self.read_plane(t, z, c))[ys, xs]

    def close(self) -> None:
        pass

    def reopen(self) -> "PlaneSource":  # pragma: no cover - abstract
        raise NotImplementedError


class ArrayPlaneSource(PlaneSource):
    """Planes of any array with numpy-style indexing – an h5py ``Dataset``, a ``zarr.Array``, a ``numpy.memmap`` or a
    plain ndarray – with the reader's ``axes`` (``TCZYX``, ``ZYX``, ``YXS`` …).  ``opener`` returns the array (called
    lazily, so a fresh handle holds no file until it is read), ``closer`` releases it (``closer(array)``)."""
    region_reads = True

    def __init__(self, opener: Callable[[], Any], axes: str, shape: tuple[int, ...], dtype: Any, rgb: bool = False, *,
                 description: str = "planes on demand", path: str | None = None, closer: Callable[[Any], None] | None = None,
                 region_reads: bool = True) -> None:
        self._opener = opener
        self._closer = closer
        self._axes = str(axes).upper()
        self._src_shape = tuple(int(v) for v in shape)
        self.src_dtype = np.dtype(dtype)
        self.rgb = bool(rgb)
        self.description = description
        self.path = path
        self.region_reads = bool(region_reads)
        lead, self._yx, self._samples, self.shape = map_axes(self._axes, self._src_shape, self.rgb)
        self._lead = {i: lt for i, lt, _n in lead}
        self._y_i, self._x_i = self._axes.index("Y"), self._axes.index("X")
        self._s_i = self._axes.index("S") if "S" in self._axes else None
        self._arr: Any = None
        self._lock = threading.RLock()

    def _array(self) -> Any:
        if self._arr is None:
            self._arr = self._opener()
        return self._arr

    def _index(self, t: int, z: int, c: int, ys: Any = slice(None), xs: Any = slice(None)) -> tuple:
        coords = {"T": t, "Z": z, "C": c}
        idx: list[Any] = []
        for i in range(len(self._axes)):
            if i in self._lead:
                idx.append(coords[self._lead[i]])
            elif i == self._y_i:
                idx.append(ys)
            elif i == self._x_i:
                idx.append(xs)
            else:  # S
                idx.append(slice(None))
        return tuple(idx)

    def _finish(self, sub: Any, c: int, ys: Any, xs: Any) -> np.ndarray:
        arr = np.asarray(sub)
        y_int, x_int = not isinstance(ys, slice), not isinstance(xs, slice)
        if self._s_i is not None:
            # the sample axis is the last remaining axis after Y / X (whatever its position in the file) unless an
            # integer y / x index removed those axes: then it is the only remaining axis
            spatial_before_s = sum(1 for i in (self._y_i, self._x_i) if i < self._s_i and not (y_int if i == self._y_i else x_int))
            s_pos = spatial_before_s
            arr = np.take(arr, min(c, arr.shape[s_pos] - 1), axis=s_pos)
        if not y_int and not x_int and self._x_i < self._y_i:
            arr = arr.T
        return arr

    def read_plane(self, t: int, z: int, c: int) -> np.ndarray:
        with self._lock:
            return self._finish(self._array()[self._index(t, z, c)], c, slice(None), slice(None))

    def read_region(self, t: int, z: int, c: int, ys: Any, xs: Any) -> np.ndarray:
        with self._lock:
            return self._finish(self._array()[self._index(t, z, c, ys, xs)], c, ys, xs)

    def close(self) -> None:
        with self._lock:
            arr, self._arr = self._arr, None
        if arr is not None and self._closer is not None:
            try:
                self._closer(arr)
            except Exception:  # noqa: BLE001
                pass

    def reopen(self) -> "ArrayPlaneSource":
        return ArrayPlaneSource(self._opener, self._axes, self._src_shape, self.src_dtype, self.rgb, description=self.description,
                                path=self.path, closer=self._closer, region_reads=self.region_reads)


def hdf5_source(path: str | os.PathLike, key: str, axes: str, shape: tuple[int, ...], dtype: Any, rgb: bool = False) -> ArrayPlaneSource:
    """Planes of the dataset ``key`` of an HDF5 file (opened read-only when first read)."""
    p = str(path)

    def opener():
        import h5py
        return h5py.File(p, "r")[key]

    def closer(ds):
        try:
            ds.file.close()
        except Exception:  # noqa: BLE001
            pass

    return ArrayPlaneSource(opener, axes, shape, dtype, rgb, description=f"HDF5 dataset {key}", path=p, closer=closer)


def zarr_source(path: str | os.PathLike, key: str | None, axes: str, shape: tuple[int, ...], dtype: Any, rgb: bool = False,
                description: str | None = None) -> ArrayPlaneSource:
    """Planes of a Zarr array (``key`` inside the group at ``path``, None = the array itself)."""
    p = str(path)

    def opener():
        import zarr
        node = zarr.open(p, mode="r")
        return node[key] if key is not None else node

    return ArrayPlaneSource(opener, axes, shape, dtype, rgb, description=description or f"Zarr array {key or ''}".rstrip(), path=p)


def npy_source(path: str | os.PathLike, axes: str, shape: tuple[int, ...], dtype: Any, rgb: bool = False) -> ArrayPlaneSource:
    """Planes of a ``.npy`` file through a read-only memory map (planes are coerced to a native dtype on read)."""
    p = str(path)
    return ArrayPlaneSource(lambda: np.load(p, mmap_mode="r"), axes, shape, dtype, rgb, description="NumPy memmap", path=p)


def view_source(view: Any, rgb: bool = False, description: str = "memory-mapped, planes byte-swapped on demand", path: str | None = None) -> ArrayPlaneSource:
    """Planes of an in-memory / memory-mapped (T, Z, C, Y, X) view (e.g. a big-endian TIFF memmap)."""
    return ArrayPlaneSource(lambda: view, "TZCYX", tuple(view.shape), view.dtype, rgb, description=description, path=path)


class ImarisPlaneSource(PlaneSource):
    """One resolution level of an Imaris (.ims) file: ``DataSet/ResolutionLevel k/TimePoint t/Channel c/Data`` per plane."""

    def __init__(self, path: str | os.PathLike, level: str, timepoints: list[str], channels: list[str], sizes: tuple[int, int, int], dtype: Any) -> None:
        self.path = str(path)
        self.level = level
        self.timepoints = list(timepoints)
        self.channels = list(channels)
        self.sizes = tuple(int(v) for v in sizes)  # (Z, Y, X)
        self.src_dtype = np.dtype(dtype)
        self.rgb = False
        self.description = f"Imaris {level}"
        self.region_reads = True
        self.shape = (len(self.timepoints), self.sizes[0], len(self.channels), self.sizes[1], self.sizes[2])
        self._f: Any = None
        self._lock = threading.RLock()

    def _dataset(self, t: int, c: int) -> Any:
        if self._f is None:
            import h5py
            self._f = h5py.File(self.path, "r")
        return self._f["DataSet"][self.level][self.timepoints[t]][self.channels[c]]["Data"]

    def read_plane(self, t: int, z: int, c: int) -> np.ndarray:
        with self._lock:
            return np.asarray(self._dataset(t, c)[z, :self.sizes[1], :self.sizes[2]])

    def read_region(self, t: int, z: int, c: int, ys: Any, xs: Any) -> np.ndarray:
        with self._lock:
            return np.asarray(self._dataset(t, c)[z, :self.sizes[1], :self.sizes[2]][ys, xs]) if not (isinstance(ys, slice) and isinstance(xs, slice)) \
                else np.asarray(self._dataset(t, c)[(z,) + _clip_slices((ys, xs), self.sizes[1:])])

    def close(self) -> None:
        with self._lock:
            f, self._f = self._f, None
        if f is not None:
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass

    def reopen(self) -> "ImarisPlaneSource":
        return ImarisPlaneSource(self.path, self.level, self.timepoints, self.channels, self.sizes, self.src_dtype)


def _clip_slices(sl: tuple[slice, slice], sizes: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(*s.indices(n)) for s, n in zip(sl, sizes))


class TiffPageSource(PlaneSource):
    """Planes of one TIFF series / resolution level read page by page with tifffile (compressed or tiled files)."""

    def __init__(self, path: str | os.PathLike, series: int, level: int, axes: str, shape: tuple[int, ...], dtype: Any, rgb: bool) -> None:
        self.path = str(path)
        self.series_index = int(series)
        self.level = int(level)
        self._src_axes = str(axes).upper()
        self._src_shape = tuple(int(v) for v in shape)
        self.src_dtype = np.dtype(dtype)
        self.rgb = bool(rgb)
        self.description = "planes on demand"
        lead, self._yx, self._samples, self.shape = map_axes(self._src_axes, self._src_shape, self.rgb)
        self._lead = [(lt, n) for _i, lt, n in lead]
        self._tf: Any = None
        self._pages: Any = None
        self._ppp = 1
        self._lock = threading.RLock()

    def _open(self) -> None:
        if self._tf is None:
            import tifffile
            self._tf = tifffile.TiffFile(self.path)
            series = self._tf.series[self.series_index]
            levels = series.levels or [series]
            lvl = levels[min(self.level, len(levels) - 1)]
            self._pages = lvl.pages
            kf = lvl.keyframe
            kaxes = str(getattr(kf, "axes", "YX"))
            samples = kf.shape[kaxes.index("S")] if "S" in kaxes else 1
            self._ppp = max(1, int(np.prod(kf.shape)) // max(1, self._yx[0] * self._yx[1] * samples))

    def _page_planes(self, i: int) -> np.ndarray:
        """Page ``i`` as (planes, Y, X) or (planes, Y, X, S)."""
        self._open()
        page = self._pages[i] if i < len(self._pages) else None
        if page is None:
            shape = (self._ppp,) + self._yx + ((self._samples,) if self._samples else ())
            return np.zeros(shape, self.src_dtype)
        arr = np.asarray(page.asarray())
        paxes = str(getattr(page, "axes", "") or "")
        if "S" in paxes and len(paxes) == arr.ndim:
            arr = np.moveaxis(arr, paxes.index("S"), -1)
            return arr.reshape((-1,) + self._yx + (arr.shape[-1],))
        if self._samples and arr.ndim >= 3 and arr.shape[-1] == self._samples:
            return arr.reshape((-1,) + self._yx + (self._samples,))
        return arr.reshape((-1,) + self._yx)

    def read_plane(self, t: int, z: int, c: int) -> np.ndarray:
        coords = {"T": t, "Z": z, "C": c}
        k = 0
        for lt, n in self._lead:
            k = k * n + coords[lt]
        page_i, sub = divmod(k, self._ppp)
        with self._lock:
            planes = self._page_planes(page_i)
        plane = planes[min(sub, planes.shape[0] - 1)]
        if self._samples:
            plane = plane[..., min(c, plane.shape[-1] - 1)]
        return plane

    def close(self) -> None:
        with self._lock:
            tf, self._tf, self._pages = self._tf, None, None
        if tf is not None:
            try:
                tf.close()
            except Exception:  # noqa: BLE001
                pass

    def reopen(self) -> "TiffPageSource":
        return TiffPageSource(self.path, self.series_index, self.level, self._src_axes, self._src_shape, self.src_dtype, self.rgb)


# ============================================================================ the lazy array
class LazyArray:
    """A (T, Z, C, Y, X) array whose planes are read on demand from a :class:`PlaneSource` (see the module docstring)."""
    ndim = 5

    def __init__(self, source: PlaneSource, cache: PlaneCache | None = None) -> None:
        self._source: PlaneSource = source
        self._cache = cache if cache is not None else PLANE_CACHE
        self._owner = self._cache.new_owner()
        self.shape: tuple[int, int, int, int, int] = tuple(int(v) for v in source.shape)  # type: ignore[assignment]
        self.rgb = bool(source.rgb)
        self._dtype = LumenImage.coerce_dtype(np.zeros(1, source.src_dtype)).dtype
        self._lock = threading.RLock()
        self._mat: np.ndarray | None = None
        #: planes read from the source so far (tests and the Log use it to prove laziness)
        self.planes_read = 0

    # ---------------------------------------------------------------- properties
    @property
    def source(self) -> PlaneSource:
        return self._source

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def full_bytes(self) -> int:
        """Decoded size of the whole stack (what materialising it costs)."""
        return int(self.size * self._dtype.itemsize)

    @property
    def nbytes(self) -> int:
        """Bytes held in memory (the materialised array or this array's cached planes), not the size on disk."""
        if self._mat is not None:
            return int(self._mat.nbytes)
        return self._cache.owner_bytes(self._owner)

    @property
    def materialized(self) -> bool:
        return self._mat is not None

    @property
    def description(self) -> str:
        return self._source.description

    def __len__(self) -> int:
        return self.shape[0]

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.shape} {self._dtype} {'in memory' if self._mat is not None else 'lazy'} {self._source.description!r}>"

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    # ---------------------------------------------------------------- array protocol
    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        arr = self.materialize()
        return arr if dtype is None or np.dtype(dtype) == arr.dtype else arr.astype(dtype)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.materialize(), name)  # min / max / mean / ravel / reshape / tobytes … read the whole stack

    def __setitem__(self, key: Any, value: Any) -> None:
        self.materialize()[key] = value

    def copy(self) -> np.ndarray:
        """An in-memory copy (reads every plane once; the lazy array itself stays lazy)."""
        if self._mat is not None:
            return self._mat.copy()
        return self._read_all()

    def astype(self, dtype: Any, **kw: Any) -> np.ndarray:
        return self.materialize().astype(dtype, **kw)

    def materialize(self) -> np.ndarray:
        """Read every plane into memory (once) and return the in-memory (T, Z, C, Y, X) array – copy-on-write."""
        with self._lock:
            if self._mat is None:
                self._mat = self._read_all()
                self._cache.drop(self._owner)
                self.close()
            return self._mat

    def close(self) -> None:
        src = self.__dict__.get("_source")
        if src is not None:
            src.close()

    def fresh(self) -> "LazyArray":
        """An independent lazy handle over the same source (for undo snapshots – never shares the materialised copy)."""
        return type(self)._from_source(self._source.reopen())

    @classmethod
    def _from_source(cls, source: PlaneSource) -> "LazyArray":
        return LazyArray(source)

    def __del__(self) -> None:  # pragma: no cover
        try:
            cache, owner = self.__dict__.get("_cache"), self.__dict__.get("_owner")
            if cache is not None and owner is not None:
                cache.drop(owner)
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # ---------------------------------------------------------------- reading
    def plane(self, t: int, z: int, c: int, cache: bool = True) -> np.ndarray:
        """One native-dtype (Y, X) plane; ``cache=False`` for one-pass streaming reads that should not churn the cache."""
        if self._mat is not None:
            return self._mat[t, z, c]
        key = (t, z, c)
        hit = self._cache.get(self._owner, key)
        if hit is not None:
            return hit
        with self._lock:
            plane = _native(self._source.read_plane(t, z, c))
            self.planes_read += 1
        if cache:
            self._cache.put(self._owner, key, plane)
        return plane

    def _region(self, t: int, z: int, c: int, ys: Any, xs: Any) -> np.ndarray:
        if self._mat is not None:
            return self._mat[t, z, c][ys, xs]
        hit = self._cache.get(self._owner, (t, z, c))
        if hit is not None:
            return hit[ys, xs]
        full = isinstance(ys, slice) and isinstance(xs, slice) and ys == slice(None) and xs == slice(None)
        if full or not self._source.region_reads:
            return self.plane(t, z, c)[ys, xs]
        with self._lock:
            sub = _native(self._source.read_region(t, z, c, ys, xs))
            self.planes_read += 1
        return sub

    def _read_all(self) -> np.ndarray:
        nt, nz, nc = self.shape[:3]
        out = np.empty(self.shape, self._dtype)
        for t in range(nt):
            for z in range(nz):
                for c in range(nc):
                    out[t, z, c] = self.plane(t, z, c, cache=False)
        return out

    def iter_planes(self, order: str = "tzc") -> Iterator[tuple[tuple[int, int, int], np.ndarray]]:
        """``((t, z, c), plane)`` for every plane, read one at a time (cache-free) – for streaming commands."""
        yield from iter_planes(self, order)

    def __getitem__(self, key: Any) -> Any:
        if self._mat is not None:
            return self._mat[key]
        if not isinstance(key, tuple):
            key = (key,)
        if len(key) > 5 or any(k is None or k is Ellipsis or isinstance(k, (list, np.ndarray)) for k in key):
            return self.materialize()[key]
        key = key + (slice(None),) * (5 - len(key))
        sel: list[list[int]] = []
        squeeze: list[bool] = []
        for ax in range(3):
            k, n = key[ax], self.shape[ax]
            if isinstance(k, slice):
                sel.append(list(range(*k.indices(n))))
                squeeze.append(False)
            else:
                i = int(k)
                if i < 0:
                    i += n
                if not 0 <= i < n:
                    raise IndexError(f"index {k} is out of bounds for axis {ax} with size {n}")
                sel.append([i])
                squeeze.append(True)
        ys, xs = key[3], key[4]
        planes = [self._region(t, z, c, ys, xs) for t in sel[0] for z in sel[1] for c in sel[2]]
        if not planes:
            sub_shape = np.empty(self.shape[3:], self._dtype)[ys, xs].shape
            out = np.empty((len(sel[0]), len(sel[1]), len(sel[2])) + sub_shape, self._dtype)
        else:
            out = np.stack(planes).reshape((len(sel[0]), len(sel[1]), len(sel[2])) + planes[0].shape)
        for ax in (2, 1, 0):
            if squeeze[ax]:
                out = out[(slice(None),) * ax + (0,)]
        return out


def _binop(name: str):
    def fn(self, *args):
        return getattr(self.materialize(), name)(*args)
    fn.__name__ = name
    return fn


for _name in ("__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__", "__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__",
              "__mod__", "__pow__", "__neg__", "__pos__", "__abs__", "__invert__", "__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__",
              "__and__", "__rand__", "__or__", "__ror__", "__xor__", "__rxor__", "__matmul__"):
    setattr(LazyArray, _name, _binop(_name))
LazyArray.__hash__ = object.__hash__  # type: ignore[assignment]


# ============================================================================ helpers for ops
def is_virtual(data: Any) -> bool:
    """True for pixel data that still lives in the file: a lazy array that was not materialised, or a memmap view."""
    if isinstance(data, LazyArray):
        return not data.materialized
    return isinstance(data, np.memmap) and getattr(data, "_mmap", None) is not None


def resident_bytes(data: Any) -> int:
    """Bytes the pixel data holds in process memory: cached planes of a lazy array, 0 for a file-backed memmap (its pages
    belong to the OS page cache and are reclaimable), the array size otherwise."""
    if isinstance(data, LazyArray):
        return int(data.nbytes)
    if isinstance(data, np.memmap) and getattr(data, "_mmap", None) is not None:
        return 0
    return int(getattr(data, "nbytes", 0))


def full_bytes(data: Any) -> int:
    """Decoded size of the whole (T, Z, C, Y, X) array."""
    if isinstance(data, LazyArray):
        return data.full_bytes
    return int(np.prod(data.shape)) * int(np.dtype(data.dtype).itemsize)


def describe(data: Any) -> str:
    """How a virtual image is read ("memory-mapped", "HDF5 dataset /data", …)."""
    if isinstance(data, LazyArray):
        return data.description
    if isinstance(data, np.memmap):
        return "memory-mapped"
    return "in memory"


def get_plane(data: Any, t: int, z: int, c: int) -> np.ndarray:
    """One plane of lazy or in-memory data without caching (streaming reads)."""
    if isinstance(data, LazyArray):
        return data.plane(t, z, c, cache=False)
    return data[t, z, c]


def iter_planes(data: Any, order: str = "tzc") -> Iterator[tuple[tuple[int, int, int], np.ndarray]]:
    """``((t, z, c), plane)`` over every plane in ``order`` (a permutation of "tzc", first = slowest)."""
    nt, nz, nc = data.shape[:3]
    sizes = {"t": nt, "z": nz, "c": nc}
    axes = [ch for ch in order.lower() if ch in sizes]
    for combo in itertools.product(*(range(sizes[a]) for a in axes)):
        pos = dict(zip(axes, combo))
        t, z, c = pos.get("t", 0), pos.get("z", 0), pos.get("c", 0)
        yield (t, z, c), get_plane(data, t, z, c)


def take_planes(data: Any, t_idx: list[int] | None = None, z_idx: list[int] | None = None, c_idx: list[int] | None = None,
                progress: Callable[[float, str], None] | None = None, check_cancel: Callable[[], None] | None = None) -> np.ndarray:
    """A sub-stack (the given T / Z / C indices, None = all) read plane by plane into a new in-memory array – a substack
    of a 2 GB virtual stack costs only the substack's memory."""
    nt, nz, nc, h, w = data.shape
    ts = list(range(nt)) if t_idx is None else [int(i) for i in t_idx]
    zs = list(range(nz)) if z_idx is None else [int(i) for i in z_idx]
    cs = list(range(nc)) if c_idx is None else [int(i) for i in c_idx]
    out = np.empty((len(ts), len(zs), len(cs), h, w), data.dtype)
    n = len(ts) * len(zs) * len(cs)
    k = 0
    for i, t in enumerate(ts):
        for j, z in enumerate(zs):
            for m, c in enumerate(cs):
                if check_cancel is not None:
                    check_cancel()
                out[i, j, m] = get_plane(data, t, z, c)
                k += 1
                if progress is not None and (k % 8 == 0 or k == n):
                    progress(k / n, f"Plane {k}/{n}")
    return out


#: memory given to one tile of a streamed median (planes × tile rows × width × 4 bytes)
MEDIAN_TILE_BYTES = 256 * 1024 ** 2


def stream_project(data: Any, axis: int, method: str, start: int = 0, stop: int | None = None,
                   progress: Callable[[float, str], None] | None = None, check_cancel: Callable[[], None] | None = None) -> np.ndarray:
    """Project ``data`` along ``axis`` (0 = T, 1 = Z) over ``[start, stop)`` plane by plane, matching
    ``lumen.core.ops.stacks._project`` (nan-aware max / min / mean / sum / SD (ddof=1) / median) without holding more
    than a few planes: one pass for max / min / mean / sum, two passes for the SD, row tiles for the median.
    Returns float32 of the input shape with the projected axis collapsed to 1."""
    nt, nz, nc, h, w = data.shape
    n_axis = data.shape[axis]
    a, b = max(int(start), 0), n_axis if stop is None else min(int(stop), n_axis)
    if a >= b:
        raise ValueError("empty projection range")
    other = nz if axis == 0 else nt
    out_shape = (1, nz, nc, h, w) if axis == 0 else (nt, 1, nc, h, w)
    out = np.empty(out_shape, np.float32)
    nk = b - a
    passes = 2 if method == "Standard Deviation" else 1
    total = other * nc * nk * passes
    done = 0

    def tz(o: int, k: int) -> tuple[int, int]:
        return (k, o) if axis == 0 else (o, k)

    def tick(msg: str) -> None:
        nonlocal done
        done += 1
        if progress is not None and (done % 4 == 0 or done == total):
            progress(done / total, msg)

    for o in range(other):
        for c in range(nc):
            if method == "Median":
                res = _stream_median(data, axis, o, c, a, b, h, w, check_cancel)
                done += nk
                if progress is not None:
                    progress(done / total, f"Median {o * nc + c + 1}/{other * nc}")
            else:
                acc: np.ndarray | None = None
                cnt = np.zeros((h, w), np.int32) if method in ("Average Intensity", "Standard Deviation") else None
                for k in range(a, b):
                    if check_cancel is not None:
                        check_cancel()
                    t, z = tz(o, k)
                    f = get_plane(data, t, z, c).astype(np.float32, copy=False)
                    if method == "Max Intensity":
                        acc = f.copy() if acc is None else np.fmax(acc, f)
                    elif method == "Min Intensity":
                        acc = f.copy() if acc is None else np.fmin(acc, f)
                    else:  # sum / mean / SD pass 1
                        fin = np.isfinite(f)
                        contrib = np.where(fin, f, 0).astype(np.float64)
                        acc = contrib if acc is None else acc + contrib
                        if cnt is not None:
                            cnt += fin
                    tick(f"Plane {done + 1}/{total}")
                assert acc is not None
                if method in ("Max Intensity", "Min Intensity"):
                    res = acc.astype(np.float32)
                elif method == "Sum Slices":
                    res = acc.astype(np.float32)
                elif method == "Average Intensity":
                    with np.errstate(invalid="ignore", divide="ignore"):
                        res = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)
                else:  # Standard Deviation (ddof=1), pass 2 over the deviations from the mean
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
                    sq = np.zeros((h, w), np.float64)
                    for k in range(a, b):
                        if check_cancel is not None:
                            check_cancel()
                        t, z = tz(o, k)
                        f = get_plane(data, t, z, c).astype(np.float64, copy=False)
                        d = np.where(np.isfinite(f), f - mean, 0)
                        sq += d * d
                        tick(f"Plane {done + 1}/{total}")
                    with np.errstate(invalid="ignore", divide="ignore"):
                        res = np.where(cnt > 1, np.sqrt(sq / np.maximum(cnt - 1, 1)), np.nan if nk > 1 else 0.0).astype(np.float32)
                    if nk <= 1:
                        res = np.zeros((h, w), np.float32)
            if axis == 0:
                out[0, o, c] = res
            else:
                out[o, 0, c] = res
    return out


def _stream_median(data: Any, axis: int, o: int, c: int, a: int, b: int, h: int, w: int, check_cancel: Callable[[], None] | None) -> np.ndarray:
    nk = b - a
    rows = max(1, min(h, MEDIAN_TILE_BYTES // max(1, nk * w * 4)))
    res = np.empty((h, w), np.float32)
    for y0 in range(0, h, rows):
        if check_cancel is not None:
            check_cancel()
        y1 = min(h, y0 + rows)
        tile = np.empty((nk, y1 - y0, w), np.float32)
        for i, k in enumerate(range(a, b)):
            t, z = (k, o) if axis == 0 else (o, k)
            tile[i] = np.asarray(data[t, z, c, y0:y1, :], dtype=np.float32)
        res[y0:y1] = np.nanmedian(tile, axis=0)
    return res


def stream_statistics(data: Any, c: int, mask: np.ndarray | None = None, progress: Callable[[float, str], None] | None = None,
                      check_cancel: Callable[[], None] | None = None) -> dict[str, Any]:
    """Voxel count, mean, SD (ddof=1), min, max, mode and median of channel ``c`` over every (t, z) plane (inside
    ``mask``) in one pass for integer data (exact histogram) and two passes for float data (min/max, then a 65536-bin
    histogram: the mode is the 256-bin mode as in Stack Statistics, the median comes from the fine histogram)."""
    nt, nz, nc, h, w = data.shape
    dtype = np.dtype(data.dtype)
    n_planes = nt * nz
    integer = dtype.kind in "ui"
    total = n_planes * (1 if integer else 2)
    done = 0

    def tick() -> None:
        nonlocal done
        done += 1
        if progress is not None and (done % 4 == 0 or done == total):
            progress(done / total, f"Plane {done}/{total}")

    nmax = int(np.iinfo(dtype).max) if integer else 0
    hist = np.zeros(nmax + 1, np.int64) if integer else None
    count = 0
    s = 0.0
    ss = 0.0
    mn, mx = np.inf, -np.inf
    for t in range(nt):
        for z in range(nz):
            if check_cancel is not None:
                check_cancel()
            p = get_plane(data, t, z, c)
            v = p[mask] if mask is not None else p.ravel()
            if integer:
                if v.size:
                    hist += np.bincount(v.astype(np.int64), minlength=nmax + 1)
                    v64 = v.astype(np.float64)
                    count += int(v.size)
                    s += float(v64.sum())
                    ss += float((v64 * v64).sum())
                    mn, mx = min(mn, float(v.min())), max(mx, float(v.max()))
            else:
                v = v.astype(np.float64)
                v = v[np.isfinite(v)]
                if v.size:
                    count += int(v.size)
                    s += float(v.sum())
                    ss += float((v * v).sum())
                    mn, mx = min(mn, float(v.min())), max(mx, float(v.max()))
            tick()
    if count == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "min": float("nan"), "max": float("nan"), "mode": float("nan"), "median": float("nan")}
    mean = s / count
    var = max((ss - count * mean * mean) / (count - 1), 0.0) if count > 1 else 0.0
    sd = float(np.sqrt(var))
    if integer:
        assert hist is not None
        mode = float(int(np.argmax(hist)))
        # ImageJ's ImageStatistics.calculateMedian: the first value whose cumulative count exceeds half the pixels
        # (never the mean of the two middle values) – the same rule as the in-memory Stack Statistics
        cum = np.cumsum(hist)
        above = np.nonzero(cum > count / 2.0)[0]
        median = float(above[0]) if above.size else float(hist.size - 1)
        return {"n": count, "mean": mean, "sd": sd, "min": mn, "max": mx, "mode": mode, "median": median, "median_exact": True}
    # float: second pass for a fine histogram over [min, max]
    bins = 65536
    span = (mx - mn) or 1.0
    fine = np.zeros(bins, np.int64)
    for t in range(nt):
        for z in range(nz):
            if check_cancel is not None:
                check_cancel()
            p = get_plane(data, t, z, c)
            v = p[mask] if mask is not None else p.ravel()
            v = v.astype(np.float64)
            v = v[np.isfinite(v)]
            if v.size:
                idx = np.clip(((v - mn) / span * bins).astype(np.int64), 0, bins - 1)
                fine += np.bincount(idx, minlength=bins)
            tick()
    coarse = fine.reshape(256, bins // 256).sum(axis=1)
    k = int(np.argmax(coarse))
    mode = mn + (k + 0.5) * span / 256
    cum = np.cumsum(fine)
    # like np.median: the middle sample, or the mean of the two middle samples for an even count (bin centres)
    lo = int(np.searchsorted(cum, (count - 1) // 2 + 1))
    hi = int(np.searchsorted(cum, count // 2 + 1))
    median = mn + ((lo + hi) / 2.0 + 0.5) * span / bins
    if mx == mn:
        median = mode = mn
    return {"n": count, "mean": mean, "sd": sd, "min": mn, "max": mx, "mode": mode, "median": median, "median_exact": False}


def page_iterator(data: Any, rgb: bool, progress: Callable[[float, str], None] | None = None,
                  check_cancel: Callable[[], None] | None = None) -> tuple[Iterator[np.ndarray], tuple[int, ...]]:
    """TIFF pages of a (T, Z, C, Y, X) array one at a time – (Y, X) planes in T, Z, C order, or (Y, X, S) pages per
    (T, Z) for RGB – plus the page-stack shape tifffile needs (``TZCYX`` or ``TZYXS``)."""
    nt, nz, nc, h, w = data.shape
    if rgb:
        shape: tuple[int, ...] = (nt, nz, h, w, nc)
        n = nt * nz
    else:
        shape = (nt, nz, nc, h, w)
        n = nt * nz * nc

    def gen() -> Iterator[np.ndarray]:
        k = 0
        for t in range(nt):
            for z in range(nz):
                if rgb:
                    if check_cancel is not None:
                        check_cancel()
                    page = np.stack([get_plane(data, t, z, c) for c in range(nc)], axis=-1)
                    yield np.ascontiguousarray(page)
                    k += 1
                    if progress is not None and (k % 4 == 0 or k == n):
                        progress(k / n, f"Page {k}/{n}")
                else:
                    for c in range(nc):
                        if check_cancel is not None:
                            check_cancel()
                        yield np.ascontiguousarray(get_plane(data, t, z, c))
                        k += 1
                        if progress is not None and (k % 4 == 0 or k == n):
                            progress(k / n, f"Page {k}/{n}")

    return gen(), shape
