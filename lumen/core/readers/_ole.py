"""A small reader for OLE2 / Compound File Binary (CFB) containers – the wrapper around Zeiss ZVI files.

``olefile`` is used when it is installed; otherwise the built-in parser below handles the format (header, FAT
with DIFAT chaining, mini FAT / mini stream, directory tree) with no dependency at all.  Only reading is
supported: :meth:`OleContainer.listdir`, :meth:`OleContainer.exists` and :meth:`OleContainer.read` (whole
stream as ``bytes``).  Stream paths use ``/`` as separator (``Image/Item(0)/Contents``).
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any

MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_ENDOFCHAIN = 0xFFFFFFFE
_FREESECT = 0xFFFFFFFF
_NOSTREAM = 0xFFFFFFFF
_MAXREGSECT = 0xFFFFFFFA
_TYPE_STORAGE, _TYPE_STREAM, _TYPE_ROOT = 1, 2, 5


class OleError(ValueError):
    pass


def is_ole(path: str | os.PathLike) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == MAGIC
    except OSError:
        return False


class _Entry:
    __slots__ = ("name", "kind", "left", "right", "child", "start", "size", "sid", "path")

    def __init__(self, sid: int, raw: bytes):
        self.sid = sid
        name_len = struct.unpack_from("<H", raw, 64)[0]
        name_len = max(0, min(name_len, 64))
        self.name = raw[:name_len].decode("utf-16-le", "replace").rstrip("\x00")
        self.kind = raw[66]
        self.left, self.right, self.child = struct.unpack_from("<III", raw, 68)
        self.start = struct.unpack_from("<I", raw, 116)[0]
        self.size = struct.unpack_from("<Q", raw, 120)[0]
        self.path = ""


class _BuiltinOle:
    """CFB reader: parses the directory on open, reads streams on demand."""

    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "rb")
        try:
            self._parse()
        except Exception:
            self.f.close()
            raise

    # ------------------------------------------------------------------ parsing
    def _parse(self) -> None:
        f = self.f
        header = f.read(512)
        if len(header) < 512 or header[:8] != MAGIC:
            raise OleError(f"{self.path.name} is not an OLE compound file")
        major = struct.unpack_from("<H", header, 26)[0]
        sector_shift, mini_shift = struct.unpack_from("<HH", header, 30)
        if sector_shift not in (9, 12) or mini_shift != 6:
            raise OleError(f"{self.path.name}: unsupported OLE sector size")
        self.sector_size = 1 << sector_shift
        self.mini_size = 1 << mini_shift
        (n_fat, first_dir, _sig, self.cutoff, first_minifat, n_minifat, first_difat, n_difat) = struct.unpack_from("<IIIIIIII", header, 44)
        self.size = path_size = os.fstat(f.fileno()).st_size
        self.n_sectors = max(0, (path_size - 512 + self.sector_size - 1) // self.sector_size)
        per_sector = self.sector_size // 4
        # DIFAT: 109 entries in the header, then chained DIFAT sectors (last entry = next DIFAT sector)
        difat = list(struct.unpack_from("<109I", header, 76))
        sect = first_difat
        for _ in range(n_difat):
            if sect > _MAXREGSECT:
                break
            raw = self._sector(sect)
            vals = struct.unpack("<%dI" % per_sector, raw)
            difat.extend(vals[:-1])
            sect = vals[-1]
        fat: list[int] = []
        for s in difat[:max(n_fat, 0)]:
            if s > _MAXREGSECT:
                break
            fat.extend(struct.unpack("<%dI" % per_sector, self._sector(s)))
        self.fat = fat
        # directory
        dir_raw = self._chain(first_dir, None, fat, self._sector, self.n_sectors)
        self.entries: list[_Entry] = [_Entry(i, dir_raw[o:o + 128]) for i, o in enumerate(range(0, len(dir_raw) - 127, 128))]
        if not self.entries or self.entries[0].kind != _TYPE_ROOT:
            raise OleError(f"{self.path.name}: OLE directory has no root entry")
        root = self.entries[0]
        # mini FAT + mini stream (lives in regular sectors, addressed from the root entry)
        self.minifat: list[int] = []
        if n_minifat and first_minifat <= _MAXREGSECT:
            raw = self._chain(first_minifat, None, fat, self._sector, self.n_sectors)
            self.minifat = list(struct.unpack("<%dI" % (len(raw) // 4), raw[:len(raw) // 4 * 4]))
        self.ministream = self._chain(root.start, root.size, fat, self._sector, self.n_sectors) if root.size else b""
        self.by_path: dict[str, _Entry] = {}
        self._walk(root.child, "", 0)

    def _walk(self, sid: int, prefix: str, depth: int) -> None:
        seen: set[int] = set()
        stack = [sid]
        while stack:
            sid = stack.pop()
            if sid > _MAXREGSECT or sid >= len(self.entries) or sid in seen or depth > 64:
                continue
            seen.add(sid)
            e = self.entries[sid]
            e.path = prefix + e.name
            self.by_path[e.path.lower()] = e
            stack.extend([e.left, e.right])
            if e.kind == _TYPE_STORAGE and e.child <= _MAXREGSECT:
                self._walk(e.child, e.path + "/", depth + 1)

    def _sector(self, sect: int) -> bytes:
        self.f.seek((sect + 1) * self.sector_size)
        raw = self.f.read(self.sector_size)
        return raw.ljust(self.sector_size, b"\x00")

    def _mini_sector(self, sect: int) -> bytes:
        o = sect * self.mini_size
        return self.ministream[o:o + self.mini_size].ljust(self.mini_size, b"\x00")

    @staticmethod
    def _chain(start: int, size: int | None, fat: list[int], get: Any, limit: int) -> bytes:
        parts: list[bytes] = []
        sect, n = start, 0
        total = 0
        while sect <= _MAXREGSECT:
            if n > limit + 1 or n > 1 << 24:
                raise OleError("corrupt OLE sector chain")
            chunk = get(sect)
            parts.append(chunk)
            total += len(chunk)
            if size is not None and total >= size:
                break
            sect = fat[sect] if sect < len(fat) else _ENDOFCHAIN
            n += 1
        data = b"".join(parts)
        return data[:size] if size is not None else data

    # ------------------------------------------------------------------ API
    def listdir(self) -> list[str]:
        return sorted(e.path for e in self.entries if e.kind == _TYPE_STREAM and e.path)

    def exists(self, name: str) -> bool:
        return name.lower() in self.by_path

    def read(self, name: str) -> bytes:
        e = self.by_path.get(name.lower())
        if e is None or e.kind != _TYPE_STREAM:
            raise KeyError(name)
        if e.size == 0:
            return b""
        if e.size < self.cutoff:
            return self._chain(e.start, e.size, self.minifat, self._mini_sector, len(self.ministream) // self.mini_size)
        return self._chain(e.start, e.size, self.fat, self._sector, self.n_sectors)

    def close(self) -> None:
        self.f.close()


class _OlefileWrapper:
    def __init__(self, path: Path, olefile: Any):
        self.ole = olefile.OleFileIO(str(path))

    def listdir(self) -> list[str]:
        return sorted("/".join(parts) for parts in self.ole.listdir(streams=True, storages=False))

    def exists(self, name: str) -> bool:
        return bool(self.ole.exists(name))

    def read(self, name: str) -> bytes:
        with self.ole.openstream(name) as s:
            return s.read()

    def close(self) -> None:
        self.ole.close()


class OleContainer:
    """``with OleContainer(path) as ole: ole.read("Image/Contents")`` – olefile when available, else the built-in parser."""

    def __init__(self, path: str | os.PathLike, prefer_olefile: bool = True):
        self.path = Path(path)
        self.backend = "built-in"
        impl: Any = None
        if prefer_olefile:
            try:
                import olefile  # type: ignore
                impl = _OlefileWrapper(self.path, olefile)
                self.backend = "olefile"
            except ImportError:
                impl = None
            except Exception:
                impl = None  # a file olefile rejects: let the built-in parser have a go (its error is clearer)
        self._impl = impl if impl is not None else _BuiltinOle(self.path)

    def listdir(self) -> list[str]:
        return self._impl.listdir()

    def exists(self, name: str) -> bool:
        return self._impl.exists(name)

    def read(self, name: str) -> bytes:
        return self._impl.read(name)

    def close(self) -> None:
        self._impl.close()

    def __enter__(self) -> "OleContainer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
