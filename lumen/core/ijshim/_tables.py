"""``ij.measure.ResultsTable``, ``ij.measure.Measurements``, ``ij.plugin.frame.RoiManager`` and
``ij.plugin.filter.Analyzer``.

``ResultsTable.getResultsTable()`` is a live view of the session's Results table (the inspector's Results
section); ``ResultsTable()`` is a private table that ``show(title)`` publishes as a named table (or as the
Results table when the title is "Results").  ``RoiManager.getInstance()`` is the session's ROI Manager."""
from __future__ import annotations

import csv
import io
import math
import os
from typing import Any

import numpy as np

from .. import roi as roimod
from ..measure import ALL_MEASUREMENTS, measure_roi
from ._context import Context, current
from ._images import ImagePlus, as_current, ensure_in_session
from ._rois import Roi

# the predefined column indices of ij.measure.ResultsTable
_STD_COLUMNS = ["Area", "Mean", "StdDev", "Mode", "Min", "Max", "X", "Y", "XM", "YM", "Perim.", "BX", "BY", "Width", "Height", "Major", "Minor", "Angle",
                "Circ.", "Feret", "IntDen", "Median", "Skew", "Kurt", "%Area", "RawIntDen", "Ch", "Slice", "Frame", "FeretX", "FeretY", "FeretAngle",
                "MinFeret", "AR", "Round", "Solidity", "MinThr", "MaxThr"]
_LAST_HEADING = len(_STD_COLUMNS) - 1


class Measurements:
    """The ``ij.measure.Measurements`` bit flags (``Analyzer.setMeasurements`` / ``ParticleAnalyzer``)."""

    AREA, MEAN, STD_DEV, MODE, MIN_MAX, CENTROID, CENTER_OF_MASS, PERIMETER, LIMIT, RECT, LABELS, ELLIPSE, INVERT_Y = (
        1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    CIRCULARITY = SHAPE_DESCRIPTORS = 8192
    FERET, INTEGRATED_DENSITY, MEDIAN, SKEWNESS, KURTOSIS, AREA_FRACTION, SLICE = 16384, 0x8000, 0x10000, 0x20000, 0x40000, 0x80000, 0x100000
    STACK_POSITION = SLICE
    SCIENTIFIC_NOTATION, ADD_TO_OVERLAY, NaN_EMPTY_CELLS = 0x200000, 0x400000, 0x800000
    ALL_STATS = (AREA + MEAN + STD_DEV + MODE + MIN_MAX + CENTROID + CENTER_OF_MASS + PERIMETER + RECT + ELLIPSE + SHAPE_DESCRIPTORS + FERET
                 + INTEGRATED_DENSITY + MEDIAN + SKEWNESS + KURTOSIS + AREA_FRACTION)


_BIT_TO_KEY = [(Measurements.AREA, "area"), (Measurements.MEAN, "mean"), (Measurements.STD_DEV, "std"), (Measurements.MODE, "mode"), (Measurements.MIN_MAX, "min"),
               (Measurements.MIN_MAX, "max"), (Measurements.CENTROID, "centroid"), (Measurements.CENTER_OF_MASS, "center_of_mass"), (Measurements.PERIMETER, "perimeter"),
               (Measurements.RECT, "bounding_rect"), (Measurements.ELLIPSE, "fit_ellipse"), (Measurements.SHAPE_DESCRIPTORS, "shape_descriptors"), (Measurements.FERET, "feret"),
               (Measurements.INTEGRATED_DENSITY, "integrated_density"), (Measurements.MEDIAN, "median"), (Measurements.SKEWNESS, "skewness"), (Measurements.KURTOSIS, "kurtosis"),
               (Measurements.AREA_FRACTION, "area_fraction"), (Measurements.SLICE, "stack_position")]


def measurement_keys(bits: int) -> list[str]:
    keys: list[str] = []
    for bit, key in _BIT_TO_KEY:
        if bits & bit and key not in keys:
            keys.append(key)
    return keys


def measurement_bits(keys: list[str]) -> int:
    bits = 0
    for bit, key in _BIT_TO_KEY:
        if key in keys:
            bits |= bit
    return bits


def _num(v: Any) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    try:
        return float(str(v))
    except ValueError:
        return float("nan")


class ResultsTable:
    """``ij.measure.ResultsTable`` (see the module docstring for live vs. private tables)."""

    COLUMN_NOT_FOUND = -1
    COLUMN_IN_USE = -2
    TABLE_FULL = -3
    AUTO_FORMAT = 32767
    _js_props = ()

    def __init__(self, n_rows: Any = None, *, _live: bool = False, _title: str | None = None) -> None:
        self._live = _live
        self._rows: list[dict[str, Any]] = []
        self._cols: list[str] = []
        self._custom: list[str] = []  # custom columns in creation order → internal index LAST_HEADING + 1 + i
        self._title = _title or ("Results" if _live else None)
        self._precision = 3
        self._decimals: dict[str, int] = {}
        self._nan_empty = False
        self._show_row_numbers = True
        if n_rows and not _live:
            for _ in range(int(n_rows)):
                self._rows.append({})

    # --- storage
    def _table(self) -> Any:
        return current().session.results if self._live else None

    def _rows_list(self) -> list[dict[str, Any]]:
        return self._table().rows if self._live else self._rows

    def _cols_list(self) -> list[str]:
        return self._table().columns if self._live else self._cols

    def _col_name(self, col: Any) -> str:
        if isinstance(col, (int, np.integer)) and not isinstance(col, bool):
            i = int(col)
            if 0 <= i <= _LAST_HEADING:
                return _STD_COLUMNS[i]
            j = i - _LAST_HEADING - 1
            if 0 <= j < len(self._custom_cols()):
                return self._custom_cols()[j]
            raise ValueError(f"ResultsTable: column index {i} not found")
        return str(col)

    def _custom_cols(self) -> list[str]:
        cols = [c for c in self._cols_list() if c not in _STD_COLUMNS and c != "Label"]
        for c in cols:
            if c not in self._custom:
                self._custom.append(c)
        return [c for c in self._custom if c in cols] + [c for c in cols if c not in self._custom]

    def _ensure_col(self, name: str) -> None:
        cols = self._cols_list()
        if name not in cols:
            cols.append(name)
            if name not in _STD_COLUMNS and name != "Label":
                self._custom.append(name)

    def _changed(self) -> None:
        if self._live:
            current().session._changed("results")

    # --- static access
    @staticmethod
    def getResultsTable(title: Any = None) -> "ResultsTable":
        if title is not None and str(title) != "Results":
            return ResultsTable._from_session_table(str(title))
        return ResultsTable(_live=True)

    @staticmethod
    def getActiveTable() -> "ResultsTable":
        return ResultsTable(_live=True)

    @staticmethod
    def getResultsWindow() -> None:
        return None

    @staticmethod
    def _from_session_table(title: str) -> "ResultsTable":
        ctx = current()
        tb = ctx.session.tables.get(title)
        rt = ResultsTable(_title=title)
        if tb:
            cols = [str(c) for c in tb.get("columns", [])]
            rt._cols = list(cols)
            rt._rows = [dict(zip(cols, r)) for r in tb.get("rows", [])]
        return rt

    @staticmethod
    def open(path: Any) -> "ResultsTable":  # noqa: A003
        """Read a CSV / TSV / tab-separated text file (ImageJ's ``ResultsTable.open``)."""
        rt = ResultsTable(_title=os.path.basename(os.fspath(path)))
        with open(os.fspath(path), encoding="utf-8-sig", newline="") as fh:
            text = fh.read()
        if not text.strip():
            return rt
        delim = "\t" if "\t" in text.splitlines()[0] else ","
        reader = csv.reader(io.StringIO(text), delimiter=delim)
        header = next(reader)
        skip_first = bool(header) and header[0].strip() in ("", " ")
        cols = header[1:] if skip_first else header
        rt._cols = [c.strip() for c in cols]
        for row in reader:
            if not row:
                continue
            vals = row[1:] if skip_first else row
            d: dict[str, Any] = {}
            for c, v in zip(rt._cols, vals):
                try:
                    d[c] = float(v) if v.strip() not in ("", "NaN") else (float("nan") if v.strip() == "NaN" else "")
                    if isinstance(d[c], float) and d[c].is_integer() and "." not in v:
                        d[c] = int(d[c])
                except ValueError:
                    d[c] = v
            rt._rows.append(d)
        return rt

    open2 = open

    # --- size
    def size(self) -> int:
        return len(self._rows_list())

    getCounter = size
    __len__ = size

    def incrementCounter(self) -> None:
        if self._live:
            self._table().append([{}])
        else:
            self._rows.append({})

    addRow = incrementCounter

    def reset(self) -> None:
        if self._live:
            self._table().clear()
            self._changed()
        else:
            self._rows, self._cols = [], []

    def deleteRow(self, i: Any) -> None:
        rows = self._rows_list()
        i = int(i)
        if 0 <= i < len(rows):
            del rows[i]
            self._changed()

    def deleteRows(self, first: Any, last: Any) -> None:
        rows = self._rows_list()
        rows[:] = [r for i, r in enumerate(rows) if not int(first) <= i <= int(last)]
        self._changed()

    def deleteColumn(self, name: Any) -> None:
        n = str(name)
        cols = self._cols_list()
        if n in cols:
            cols.remove(n)
        for r in self._rows_list():
            r.pop(n, None)
        self._changed()

    # --- values
    def addValue(self, col: Any, value: Any) -> None:
        rows = self._rows_list()
        if not rows:
            raise ValueError("ResultsTable.addValue: call incrementCounter() first (Counter==0)")
        name = self._col_name(col)
        self._ensure_col(name)
        rows[-1][name] = value

    def addLabel(self, *args: Any) -> None:
        rows = self._rows_list()
        if not rows:
            raise ValueError("ResultsTable.addLabel: call incrementCounter() first")
        label = args[-1]
        self._ensure_col("Label")
        rows[-1]["Label"] = str(label)

    def setLabel(self, label: Any, row: Any) -> None:
        rows = self._rows_list()
        i = int(row)
        if i == len(rows):
            self.incrementCounter()
        self._ensure_col("Label")
        self._rows_list()[i]["Label"] = str(label)

    def addStringValue(self, col: Any, value: Any) -> None:
        self.addValue(col, str(value))

    def setValue(self, col: Any, row: Any, value: Any) -> None:
        rows = self._rows_list()
        i = int(row)
        if i == len(rows):
            self.incrementCounter()
            rows = self._rows_list()
        if not 0 <= i < len(rows):
            raise ValueError(f"ResultsTable.setValue: row {i} out of range (0–{len(rows)})")
        name = self._col_name(col)
        self._ensure_col(name)
        rows[i][name] = value

    def getValue(self, col: Any, row: Any) -> float:
        rows = self._rows_list()
        name = self._col_name(col)
        if name not in self._cols_list():
            raise ValueError(f'"{name}" column not found')
        i = int(row)
        if not 0 <= i < len(rows):
            raise ValueError(f"ResultsTable.getValue: row {i} out of range (0–{len(rows) - 1})")
        return _num(rows[i].get(name))

    def getValueAsDouble(self, col: Any, row: Any) -> float:
        return self.getValue(col, row)

    def getStringValue(self, col: Any, row: Any) -> str:
        rows = self._rows_list()
        name = self._col_name(col)
        i = int(row)
        if name not in self._cols_list() or not 0 <= i < len(rows):
            return "null"
        v = rows[i].get(name)
        if v is None:
            return "null" if name != "Label" else ""
        if isinstance(v, float):
            return self._fmt(name, v)
        return str(v)

    def _fmt(self, col: str, v: float) -> str:
        if math.isnan(v):
            return "NaN"
        d = self._decimals.get(col, self._precision)
        if d == self.AUTO_FORMAT:
            return f"{v:g}"
        return f"{v:.{d}f}"

    def getLabel(self, row: Any) -> str | None:
        rows = self._rows_list()
        i = int(row)
        if "Label" not in self._cols_list() or not 0 <= i < len(rows):
            return None
        v = rows[i].get("Label")
        return None if v is None else str(v)

    def getRowAsString(self, row: Any) -> str:
        rows = self._rows_list()
        i = int(row)
        cells = [str(i + 1)] if self._show_row_numbers else []
        for c in self._cols_list():
            v = rows[i].get(c)
            cells.append("" if v is None else (self._fmt(c, v) if isinstance(v, float) else str(v)))
        return "\t".join(cells)

    def getRowIndex(self, label: Any) -> int:
        for i, r in enumerate(self._rows_list()):
            if str(r.get("Label", "")) == str(label):
                return i
        return -1

    def getColumn(self, col: Any) -> list[float] | None:
        name = self._col_name(col)
        if name not in self._cols_list():
            return None
        return [_num(r.get(name)) for r in self._rows_list()]

    getColumnAsDoubles = getColumn

    def getColumnAsStrings(self, col: Any) -> list[str] | None:
        name = self._col_name(col)
        if name not in self._cols_list():
            return None
        return [self.getStringValue(name, i) for i in range(self.size())]

    getStringColumn = getColumnAsStrings

    def setColumn(self, col: Any, values: Any) -> None:
        name = self._col_name(col)
        self._ensure_col(name)
        vals = list(values)
        while self.size() < len(vals):
            self.incrementCounter()
        for i, v in enumerate(vals):
            self._rows_list()[i][name] = v

    def columnExists(self, col: Any) -> bool:
        try:
            return self._col_name(col) in self._cols_list()
        except ValueError:
            return False

    def getColumnIndex(self, name: Any) -> int:
        n = str(name)
        if n not in self._cols_list():
            return self.COLUMN_NOT_FOUND
        if n in _STD_COLUMNS:
            return _STD_COLUMNS.index(n)
        return _LAST_HEADING + 1 + self._custom_cols().index(n)

    def getFreeColumn(self, name: Any) -> int:
        n = str(name)
        if n in self._cols_list():
            return self.COLUMN_IN_USE
        self._ensure_col(n)
        return self.getColumnIndex(n)

    def getLastColumn(self) -> int:
        cols = [c for c in self._cols_list() if c != "Label"]
        return max((self.getColumnIndex(c) for c in cols), default=-1)

    def getHeadings(self) -> list[str]:
        return list(self._cols_list())

    def getColumnHeadings(self) -> str:
        cols = self._cols_list()
        return (" \t" if self._show_row_numbers else "") + "\t".join(cols)

    def getColumnHeading(self, col: Any) -> str | None:
        try:
            n = self._col_name(col)
        except ValueError:
            return None
        return n if n in self._cols_list() else None

    def getTitle(self) -> str:
        return self._title or "Results"

    def setDecimalPlaces(self, col: Any, digits: Any) -> None:
        try:
            self._decimals[self._col_name(col)] = int(digits)
        except ValueError:
            pass

    def setPrecision(self, digits: Any) -> None:
        self._precision = int(digits)

    def setNaNEmptyCells(self, b: Any) -> None:
        self._nan_empty = bool(b)

    def showRowNumbers(self, b: Any) -> None:
        self._show_row_numbers = bool(b)

    showRowIndexes = showRowNumbers

    def disableRowLabels(self) -> None:
        self.deleteColumn("Label")

    def setDefaultHeadings(self) -> None:
        pass

    def saveColumnHeaders(self, b: Any) -> None:
        pass

    def sort(self, col: Any) -> None:
        name = self._col_name(col)
        rows = self._rows_list()
        rows.sort(key=lambda r: _num(r.get(name)))
        self._changed()

    def applyMacro(self, code: Any) -> None:
        raise ValueError("ResultsTable.applyMacro is not supported")

    def getTableAsImage(self) -> None:
        return None

    def update(self, *a: Any) -> None:
        self._changed()

    updateResults = update

    # --- publish / save
    def show(self, title: Any = None) -> None:
        ctx = current()
        s = ctx.session
        name = str(title) if title is not None else self.getTitle()
        if self._live:
            s._changed("results")
            return
        self._title = name
        if name == "Results":
            s.results.clear()
            s.results.append([dict(r) for r in self._rows])
            for c in self._cols:
                if c not in s.results.columns:
                    s.results.columns.append(c)
            s._changed("results")
            return
        s.tables[name] = {"name": name, "columns": list(self._cols), "rows": [[_json(r.get(c)) for c in self._cols] for r in self._rows]}
        s._changed("state")

    def save(self, path: Any) -> bool:
        self.saveAs(path)
        return True

    def saveAs(self, path: Any) -> None:
        p = os.fspath(path)
        delim = "\t" if p.lower().endswith((".tsv", ".txt", ".xls")) else ","
        cols = self._cols_list()
        with open(p, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter=delim, lineterminator="\n")
            w.writerow(([" "] if self._show_row_numbers else []) + cols)
            for i, r in enumerate(self._rows_list(), 1):
                w.writerow(([i] if self._show_row_numbers else []) + [self._fmt(c, r[c]) if isinstance(r.get(c), float) else ("" if r.get(c) is None else r.get(c)) for c in cols])

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"ResultsTable[{self.getTitle()}: {self.size()} rows, {len(self._cols_list())} columns]"

    def __iter__(self) -> Any:
        return iter(range(self.size()))


def _json(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# ====================================================================== RoiManager
class RoiManager:
    """``ij.plugin.frame.RoiManager`` – the session's ROI Manager (``getInstance()`` / ``RoiManager()`` give the same
    object).  Commands go through the macro layer's ``roiManager(…)``."""

    _js_props = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> "RoiManager":
        ctx = current()
        rm = ctx.roi_manager
        if rm is None:
            rm = object.__new__(cls)
            rm._ctx = ctx
            ctx.roi_manager = rm
        return rm

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @staticmethod
    def getInstance() -> "RoiManager":
        return RoiManager()

    getInstance2 = getInstance
    getRoiManager = getInstance

    def _rm(self) -> roimod.RoiManager:
        return self._ctx.session.roi_manager

    def _ij(self) -> Any:
        return self._ctx.ij

    # --- contents
    def getCount(self) -> int:
        return len(self._rm().rois)

    size = getCount

    def __len__(self) -> int:
        return self.getCount()

    def getRoisAsArray(self) -> list[Roi]:
        out = []
        for d in self._rm().rois:
            r = Roi._wrap({k: v for k, v in d.items() if k != "image_id"})
            out.append(r)
        return out

    def getRoi(self, i: Any) -> Roi | None:
        rois = self._rm().rois
        i = int(i)
        if not 0 <= i < len(rois):
            return None
        return Roi._wrap({k: v for k, v in rois[i].items() if k != "image_id"})

    def getName(self, i: Any) -> str | None:
        rois = self._rm().rois
        i = int(i)
        return str(rois[i].get("name", "")) if 0 <= i < len(rois) else None

    def getRoiIndex(self, roi: Roi) -> int:
        for i, d in enumerate(self._rm().rois):
            if d.get("id") == roi._roi.get("id"):
                return i
        for i, d in enumerate(self._rm().rois):
            if Roi._wrap(d) == roi:
                return i
        return -1

    def getIndex(self, name: Any) -> int:
        for i, d in enumerate(self._rm().rois):
            if d.get("name") == str(name):
                return i
        return -1

    def contains(self, roi: Roi) -> bool:
        return self.getRoiIndex(roi) >= 0

    def rename(self, i: Any, name: Any) -> bool:
        rois = self._rm().rois
        i = int(i)
        if 0 <= i < len(rois):
            rois[i]["name"] = str(name)
            self._ctx.session._changed("rois")
            return True
        return False

    def setName(self, name: Any, i: Any) -> None:
        self.rename(i, name)

    def addRoi(self, roi: Roi) -> None:
        self.add(roi, -1)

    def add(self, *args: Any) -> None:
        """``add(roi, n)`` / ``add(imp, roi, n)`` – ``n`` names the entry when ≥ 0 (ImageJ: a suffix)."""
        imp: ImagePlus | None = None
        if args and isinstance(args[0], ImagePlus):
            imp, args = args[0], args[1:]
        roi = args[0]
        n = int(args[1]) if len(args) > 1 else -1
        s = self._ctx.session
        img = imp._img if imp is not None else s.active
        if imp is not None:
            ensure_in_session(self._ctx, img)
        d = {k: v for k, v in roi._roi.items() if k not in ("id", "image_id")}
        pos = d.pop("position", None) or ({"t": img.display.t, "z": img.display.z, "c": img.display.c} if img is not None else None)
        entry = self._rm().add(d, image_id=img.id if img is not None else None, position=pos)
        if n >= 0 and not roi._roi.get("name"):
            entry["name"] = f"{entry['name']}-{n}" if n else entry["name"]
        s._changed("rois")

    def setRoi(self, roi: Roi, i: Any) -> None:
        rois = self._rm().rois
        i = int(i)
        if 0 <= i < len(rois):
            keep = {k: rois[i][k] for k in ("id", "image_id", "position", "name") if k in rois[i]}
            rois[i] = roimod.normalize({**{k: v for k, v in roi._roi.items() if k not in ("id", "image_id")}, **keep})
            self._ctx.session._changed("rois")

    # --- selection
    def select(self, *args: Any) -> None:
        """``select(i)`` / ``select(imp, i)`` / ``select(i, additive)``."""
        imp: ImagePlus | None = None
        if args and isinstance(args[0], ImagePlus):
            imp, args = args[0], args[1:]
        i = int(args[0])
        additive = bool(args[1]) if len(args) > 1 else False
        with as_current(self._ctx, imp):
            if additive and self._ij().selected_rois:
                sel = list(self._ij().selected_rois) + [i]
                self._ij().roiManager("select", sel)
            else:
                self._ij().roiManager("select", i)

    def selectAndMakeVisible(self, imp: Any, i: Any) -> None:
        self.select(imp, i)

    def deselect(self, roi: Any = None) -> None:
        self._ij().roiManager("deselect")

    def setSelectedIndexes(self, indexes: Any) -> None:
        idx = [int(i) for i in indexes]
        if idx:
            self._ij().roiManager("select", idx)
        else:
            self._ij().roiManager("deselect")

    def getSelectedIndexes(self) -> list[int]:
        return list(self._ij().selected_rois)

    def getIndexes(self) -> list[int]:
        sel = self.getSelectedIndexes()
        return sel if sel else list(range(self.getCount()))

    def getSelectedIndex(self) -> int:
        sel = self.getSelectedIndexes()
        return sel[0] if sel else -1

    def isSelected(self, i: Any) -> bool:
        return int(i) in self.getSelectedIndexes()

    def getSelectedRoisAsArray(self) -> list[Roi]:
        rois = self.getRoisAsArray()
        sel = self.getSelectedIndexes()
        return [rois[i] for i in sel if i < len(rois)] if sel else rois

    # --- commands
    def runCommand(self, *args: Any) -> bool:
        """``runCommand(cmd)``, ``runCommand(cmd, arg)``, ``runCommand(imp, cmd)``, ``runCommand(imp, cmd, arg)``."""
        imp: ImagePlus | None = None
        if args and isinstance(args[0], ImagePlus):
            imp, args = args[0], args[1:]
        if not args:
            raise TypeError("runCommand(command)")
        cmd = str(args[0])
        extra = [a for a in args[1:] if a is not None and a != ""]
        low = cmd.strip().lower()
        with as_current(self._ctx, imp):
            if low in ("select all",):
                self._ij().roiManager("select", list(range(self.getCount())))
                return True
            if low in ("show all", "show none", "show all with labels", "show all without labels", "usenames", "associate", "centered", "remove slice info",
                       "remove channel info", "remove frame info", "list", "sort by name"):
                self._ij().roiManager(cmd)
                return True
            if low == "delete" and not self._ij().selected_rois and not extra:
                self._ij().roiManager("delete")
                return True
            self._ij().roiManager(cmd, *extra)
        return True

    def reset(self) -> None:
        self._ij().roiManager("reset")

    def remove(self, i: Any) -> None:
        rois = self._rm().rois
        i = int(i)
        if 0 <= i < len(rois):
            self._rm().remove([rois[i]["id"]])
            self._ctx.session._changed("rois")

    def multiMeasure(self, imp: Any = None) -> ResultsTable:
        with as_current(self._ctx, imp):
            out = self._ij()._run("analyze.multi_measure", {"one_row_per_slice": True})
        rt = ResultsTable(_title="Results")
        tb = out.get("table")
        if tb:
            cols = [str(c) for c in tb.get("columns", [])]
            rt._cols = list(cols)
            rt._rows = [dict(zip(cols, r)) for r in tb.get("rows", [])]
        return rt

    def measure(self, imp: Any = None) -> bool:
        with as_current(self._ctx, imp):
            self._ij().roiManager("measure")
        return True

    def moveRoisToOverlay(self, imp: Any) -> None:
        with as_current(self._ctx, imp):
            self._ij()._run("overlay.from_manager", {})

    def translate(self, dx: Any, dy: Any) -> None:
        self._ij().roiManager("translate", float(dx), float(dy))

    def sort(self) -> None:
        self._ij().roiManager("sort")

    def setGroup(self, g: Any) -> None:
        for i in self.getIndexes():
            self._rm().rois[i]["group"] = int(g)

    def selectGroup(self, g: Any) -> None:
        idx = [i for i, d in enumerate(self._rm().rois) if int(d.get("group", 0)) == int(g)]
        self.setSelectedIndexes(idx)

    def setEditMode(self, imp: Any, b: Any) -> None:
        pass

    def setVisible(self, b: Any) -> None:
        pass

    def isVisible(self) -> bool:
        return True

    def show(self) -> None:
        pass

    def close(self) -> None:
        pass

    def toFront(self) -> None:
        pass

    def getTitle(self) -> str:
        return "ROI Manager"

    def toString(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return f"RoiManager[{self.getCount()} ROIs]"

    @staticmethod
    def restoreCentered(b: Any) -> None:
        pass


# ====================================================================== Analyzer
class Analyzer:
    """``ij.plugin.filter.Analyzer``: ``Analyzer(imp[, measurements[, rt]]).measure()`` and the static
    measurement settings (``setMeasurements`` ↔ Analyze ▸ Set Measurements)."""

    _precision = 3

    def __init__(self, imp: Any = None, measurements: Any = None, rt: ResultsTable | None = None) -> None:
        self._imp = imp
        self._bits = int(measurements) if measurements is not None else None
        self._rt = rt

    def measure(self) -> bool:
        ctx = current()
        imp = self._imp or ctx.session.active and ImagePlus(_img=ctx.session.active)
        if imp is None:
            raise ValueError("Analyzer.measure: no image")
        img = imp._img
        keys = measurement_keys(self._bits) if self._bits is not None else list(ctx.session.measurements)
        d = img.display
        thr = None
        if self._bits is not None and self._bits & Measurements.LIMIT or (self._bits is None and ctx.session.measure_options.get("limit_to_threshold")):
            rec = ctx.session.thresholds.get(img.id)
            if rec is not None:
                thr = (float(rec["lower"]), float(rec["upper"]))
        rows = measure_roi(img, ctx.roi, keys or ["area", "mean", "min", "max"], t=d.t, z=d.z, c=d.c, limit_to_threshold=thr,
                           label=img.name, options=ctx.session.measure_options)
        rt = self._rt if self._rt is not None else ResultsTable(_live=True)
        for r in rows:
            rt.incrementCounter()
            for k, v in r.items():
                rt.addValue(k, v)
        rt.update()
        return True

    def displayResults(self) -> None:
        if self._rt is not None:
            self._rt.show("Results")
        else:
            current().session._changed("results")

    def updateHeadings(self) -> None:
        pass

    def getResultsTable(self) -> ResultsTable:  # noqa: D102  (ImageJ has this both static and as an instance method)
        return self._rt if self._rt is not None else ResultsTable(_live=True)

    @staticmethod
    def setMeasurements(bits: Any) -> None:
        ctx = current()
        b = int(bits)
        ctx.session.measurements = measurement_keys(b) or ["area"]
        opts = ctx.session.measure_options
        opts["limit_to_threshold"] = bool(b & Measurements.LIMIT)
        opts["display_label"] = bool(b & Measurements.LABELS)
        opts["invert_y"] = bool(b & Measurements.INVERT_Y)
        opts["nan_empty_cells"] = bool(b & Measurements.NaN_EMPTY_CELLS)
        opts["scientific_notation"] = bool(b & Measurements.SCIENTIFIC_NOTATION)
        opts["add_to_overlay"] = bool(b & Measurements.ADD_TO_OVERLAY)
        ctx.session._changed("state")

    @staticmethod
    def getMeasurements() -> int:
        ctx = current()
        b = measurement_bits(list(ctx.session.measurements))
        opts = ctx.session.measure_options
        if opts.get("limit_to_threshold"):
            b |= Measurements.LIMIT
        if opts.get("display_label", True):
            b |= Measurements.LABELS
        if opts.get("invert_y"):
            b |= Measurements.INVERT_Y
        if opts.get("nan_empty_cells"):
            b |= Measurements.NaN_EMPTY_CELLS
        return b

    @staticmethod
    def setMeasurement(bit: Any, state: Any) -> None:
        b = Analyzer.getMeasurements()
        b = (b | int(bit)) if state else (b & ~int(bit))
        Analyzer.setMeasurements(b)

    @staticmethod
    def setPrecision(n: Any) -> None:
        Analyzer._precision = int(n)
        current().session.measure_options["decimals"] = int(n)

    @staticmethod
    def getPrecision() -> int:
        return Analyzer._precision

    @staticmethod
    def setRedirectImage(imp: Any) -> None:
        pass

    @staticmethod
    def getRedirectImage(imp: Any = None) -> None:
        return None

    @staticmethod
    def setOption(name: Any, value: Any) -> None:
        pass

    @staticmethod
    def resetCounter() -> bool:
        ResultsTable(_live=True).reset()
        return True

    @staticmethod
    def getCounter() -> int:
        return ResultsTable(_live=True).size()

    @staticmethod
    def setDefaultHeadings() -> None:
        pass


__all__ = ["ResultsTable", "RoiManager", "Measurements", "Analyzer", "measurement_keys", "measurement_bits", "ALL_MEASUREMENTS", "Context"]
