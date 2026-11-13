"""Dialogs without a user: ``GenericDialog`` (and ``NonBlockingGenericDialog`` / ``GenericDialogPlus``) collects its
fields and answers ``getNext…`` from the script parameters handed to the runner (the Script panel's parameter
prompt, keyed by field label) or from the defaults; ``WaitForUserDialog`` / ``YesNoCancelDialog`` /
``MessageDialog`` log and continue; ``OpenDialog`` / ``SaveDialog`` / ``DirectoryChooser`` answer from the
parameters (``file`` / ``directory`` / the dialog title) or return ``None`` like a cancelled dialog.
``TextWindow`` is a text window (a named table) and ``Plot`` becomes the inspector's Plot section."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from ._context import current
from ._images import ImagePlus, ensure_in_session
from ._java import color_hex


def _as_bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in ("true", "1", "yes", "y", "on")


class _Field:
    """A dialog component as returned by ``getNumericFields()`` …: ``getText`` / ``setText`` / ``getState`` …"""

    def __init__(self, value: Any) -> None:
        self._v = value

    def getText(self) -> str:
        return str(self._v)

    def setText(self, t: Any) -> None:
        self._v = t

    def getState(self) -> bool:
        return bool(self._v)

    def setState(self, b: Any) -> None:
        self._v = bool(b)

    def getSelectedItem(self) -> Any:
        return self._v

    def getSelectedIndex(self) -> int:
        return 0

    def select(self, v: Any) -> None:
        self._v = v

    def addTextListener(self, *a: Any) -> None:
        pass

    def addItemListener(self, *a: Any) -> None:
        pass

    def addKeyListener(self, *a: Any) -> None:
        pass

    def setEnabled(self, *a: Any) -> None:
        pass

    def setVisible(self, *a: Any) -> None:
        pass


class GenericDialog:
    """``ij.gui.GenericDialog`` – non-interactive (see the module docstring)."""

    _js_props = ()

    def __init__(self, title: Any = "", parent: Any = None) -> None:
        self.title = str(title)
        self._fields: list[dict[str, Any]] = []
        self._cursor: dict[str, int] = {}
        self._shown = False
        self._labels: list[str] = []

    # --- adding fields
    def _add(self, kind: str, label: Any, default: Any, **extra: Any) -> None:
        self._fields.append({"kind": kind, "label": str(label), "default": default, **extra})

    def addNumericField(self, label: Any, default: Any, digits: Any = 0, columns: Any = None, units: Any = None) -> None:
        self._add("number", label, float(default), digits=int(digits or 0), units=units)

    def addSlider(self, label: Any, lo: Any, hi: Any, default: Any, step: Any = None) -> None:
        self._add("number", label, float(default), min=float(lo), max=float(hi), step=float(step) if step else None)

    def addStringField(self, label: Any, default: Any, columns: Any = None) -> None:
        self._add("string", label, "" if default is None else str(default))

    def addCheckbox(self, label: Any, default: Any) -> None:
        self._add("bool", label, bool(default))

    def addCheckboxGroup(self, rows: Any, cols: Any, labels: Any, defaults: Any, headings: Any = None) -> None:
        for lab, d in zip(labels, defaults):
            self._add("bool", lab, bool(d))

    def addChoice(self, label: Any, items: Any, default: Any = None) -> None:
        items = [str(i) for i in items]
        self._add("choice", label, str(default) if default is not None else (items[0] if items else ""), choices=items)

    def addRadioButtonGroup(self, label: Any, items: Any, rows: Any = 1, cols: Any = 1, default: Any = None) -> None:
        items = [str(i) for i in items]
        self._add("radio", label, str(default) if default is not None else (items[0] if items else ""), choices=items)

    def addEnumChoice(self, label: Any, default: Any) -> None:
        self._add("choice", label, str(default), choices=[str(default)])

    def addDirectoryField(self, label: Any, default: Any = "", columns: Any = None) -> None:
        self._add("directory", label, "" if default is None else str(default))

    def addFileField(self, label: Any, default: Any = "", columns: Any = None) -> None:
        self._add("file", label, "" if default is None else str(default))

    addDirectoryOrFileField = addFileField

    def addImageChoice(self, label: Any, default: Any = None) -> None:
        ctx = current()
        titles = [im.name for im in ctx.session.images.values()]
        self._add("image", label, str(default) if default else (ctx.session.active.name if ctx.session.active is not None else ""), choices=titles)

    def addTextAreas(self, text1: Any, text2: Any = None, rows: Any = 8, cols: Any = 40) -> None:
        self._add("text", "Text 1", "" if text1 is None else str(text1))
        self._add("text", "Text 2", "" if text2 is None else str(text2))

    def addMessage(self, text: Any, *rest: Any) -> None:
        self._fields.append({"kind": "message", "label": str(text), "default": None})

    def addHelp(self, url: Any) -> None:
        pass

    def addPanel(self, *a: Any) -> None:
        pass

    def addImage(self, *a: Any) -> None:
        pass

    def addButton(self, *a: Any) -> None:
        pass

    def addToSameRow(self) -> None:
        pass

    def addDialogListener(self, *a: Any) -> None:
        pass

    def setInsets(self, *a: Any) -> None:
        pass

    def setOKLabel(self, *a: Any) -> None:
        pass

    def setCancelLabel(self, *a: Any) -> None:
        pass

    def hideCancelButton(self) -> None:
        pass

    def enableYesNoCancel(self, *a: Any) -> None:
        pass

    def centerDialog(self, *a: Any) -> None:
        pass

    def setLocation(self, *a: Any) -> None:
        pass

    def setSmartRecording(self, *a: Any) -> None:
        pass

    def setResizable(self, *a: Any) -> None:
        pass

    def setEchoChar(self, *a: Any) -> None:
        pass

    def setDefaultFormat(self, *a: Any) -> None:
        pass

    def pack(self) -> None:
        pass

    def dispose(self) -> None:
        pass

    def setVisible(self, *a: Any) -> None:
        pass

    def toFront(self) -> None:
        pass

    # --- showing / reading
    def showDialog(self) -> None:
        self._shown = True
        self._cursor = {}

    def wasCanceled(self) -> bool:
        return False

    def wasOKed(self) -> bool:
        return True

    def invalidNumber(self) -> bool:
        return False

    def getErrorMessage(self) -> None:
        return None

    def getTitle(self) -> str:
        return self.title

    def _value(self, f: dict[str, Any]) -> Any:
        ctx = current()
        v = ctx.param(f["label"], default=None)
        return f["default"] if v is None else v

    def _next(self, *kinds: str) -> dict[str, Any]:
        key = kinds[0]
        start = self._cursor.get(key, 0)
        matching = [f for f in self._fields if f["kind"] in kinds]
        if start >= len(matching):
            raise ValueError(f"GenericDialog.getNext{key.capitalize()}: no more {key} fields")
        self._cursor[key] = start + 1
        return matching[start]

    def getNextNumber(self) -> float:
        f = self._next("number")
        try:
            return float(self._value(f))
        except (TypeError, ValueError):
            return float(f["default"])

    def getNextString(self) -> str:
        f = self._next("string", "file", "directory")
        return str(self._value(f))

    def getNextText(self) -> str:
        f = self._next("text")
        return str(self._value(f))

    def getNextBoolean(self) -> bool:
        f = self._next("bool")
        return _as_bool(self._value(f), bool(f["default"]))

    def getNextChoice(self) -> str:
        f = self._next("choice")
        v = self._value(f)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and f.get("choices") and 0 <= int(v) < len(f["choices"]):
            return f["choices"][int(v)]
        return str(v)

    def getNextChoiceIndex(self) -> int:
        f = self._next("choice")
        v = self._value(f)
        choices = f.get("choices") or []
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
        return choices.index(str(v)) if str(v) in choices else 0

    def getNextEnumChoice(self, *a: Any) -> str:
        return self.getNextChoice()

    def getNextRadioButton(self) -> str:
        f = self._next("radio")
        return str(self._value(f))

    def getNextImage(self) -> ImagePlus | None:
        f = self._next("image")
        ctx = current()
        try:
            return ImagePlus(_img=ctx.ij._image(str(self._value(f))))
        except ValueError:
            return None

    def getNumericFields(self) -> list[_Field]:
        return [_Field(self._value(f)) for f in self._fields if f["kind"] == "number"]

    def getStringFields(self) -> list[_Field]:
        return [_Field(self._value(f)) for f in self._fields if f["kind"] in ("string", "file", "directory")]

    def getCheckboxes(self) -> list[_Field]:
        return [_Field(_as_bool(self._value(f), bool(f["default"]))) for f in self._fields if f["kind"] == "bool"]

    def getChoices(self) -> list[_Field]:
        return [_Field(self._value(f)) for f in self._fields if f["kind"] == "choice"]

    def getSliders(self) -> list[_Field]:
        return [_Field(self._value(f)) for f in self._fields if f["kind"] == "number" and "min" in f]

    def getRadioButtonGroups(self) -> list[_Field]:
        return [_Field(self._value(f)) for f in self._fields if f["kind"] == "radio"]

    def getTextArea1(self) -> _Field | None:
        t = [f for f in self._fields if f["kind"] == "text"]
        return _Field(self._value(t[0])) if t else None

    def getTextArea2(self) -> _Field | None:
        t = [f for f in self._fields if f["kind"] == "text"]
        return _Field(self._value(t[1])) if len(t) > 1 else None

    def getLabel(self) -> Any:
        return None

    def getButtons(self) -> list[Any]:
        return []

    def getMessage(self) -> Any:
        return None

    def getInsets(self) -> Any:
        return None

    def getParameters(self) -> list[dict[str, Any]]:
        """Lumen extension: the fields as parameter descriptors (what the Script panel could prompt for)."""
        out = []
        for f in self._fields:
            if f["kind"] == "message":
                continue
            d = {"name": f["label"], "label": f["label"], "type": {"number": "float", "bool": "bool", "choice": "choice", "radio": "choice", "image": "image"}.get(f["kind"], "string"),
                 "default": f["default"]}
            for k in ("min", "max", "step", "choices"):
                if f.get(k) is not None:
                    d[k] = f[k]
            if f["kind"] in ("file", "directory"):
                d["type"] = "file"
                d["style"] = f["kind"]
            out.append(d)
        return out


class NonBlockingGenericDialog(GenericDialog):
    pass


class GenericDialogPlus(GenericDialog):
    """``fiji.util.gui.GenericDialogPlus``: ``addFileField`` / ``addDirectoryField`` / ``addImageChoice`` …"""

    def addButton(self, label: Any, listener: Any = None) -> None:
        pass

    def addComponent(self, *a: Any) -> None:
        pass


class WaitForUserDialog:
    def __init__(self, title: Any, message: Any = None) -> None:
        self.title = str(title)
        self.message = "" if message is None else str(message)

    def show(self) -> None:
        current().emit(f"WaitForUser{': ' + self.title if self.message else ''}: {self.message or self.title} – continuing (a script cannot wait for the user)")

    def escPressed(self) -> bool:
        return False

    def setVisible(self, b: Any) -> None:
        if b:
            self.show()


class YesNoCancelDialog:
    def __init__(self, parent: Any, title: Any, message: Any, yes: Any = "Yes", no: Any = "No") -> None:
        current().emit(f"{title}: {message} → Yes")

    def yesPressed(self) -> bool:
        return True

    def cancelPressed(self) -> bool:
        return False

    def noPressed(self) -> bool:
        return False


class MessageDialog:
    def __init__(self, parent: Any, title: Any, message: Any) -> None:
        current().emit(f"{title}: {message}")

    def escapePressed(self) -> bool:
        return False


class HTMLDialog:
    def __init__(self, title: Any, html: Any, modal: Any = True) -> None:
        import re
        text = re.sub(r"<[^>]+>", "", str(html))
        current().emit(f"{title}: {text.strip()}")

    def escapePressed(self) -> bool:
        return False


# ====================================================================== file dialogs
_LAST_DIR: str | None = None


class OpenDialog:
    """``OpenDialog(title[, default])`` / ``OpenDialog(title, dir, name)``: answered from the parameters."""

    def __init__(self, title: Any = "Open", *args: Any) -> None:
        ctx = current()
        self.title = str(title)
        default = None
        if len(args) == 1:
            default = args[0]
        elif len(args) >= 2:
            default = os.path.join(str(args[0]), str(args[1])) if args[0] else args[1]
        v = ctx.param(self.title, "file", "path", "open", default=None)
        self._path = os.fspath(v) if v is not None else (str(default) if default and os.path.exists(str(default)) else None)

    def getPath(self) -> str | None:
        return self._path

    def getDirectory(self) -> str | None:
        if self._path is None:
            return None
        return os.path.dirname(self._path) + os.sep

    def getFileName(self) -> str | None:
        return os.path.basename(self._path) if self._path else None

    @staticmethod
    def getLastDirectory() -> str | None:
        return _LAST_DIR

    @staticmethod
    def setLastDirectory(d: Any) -> None:
        global _LAST_DIR
        _LAST_DIR = str(d)

    @staticmethod
    def getDefaultDirectory() -> str:
        return _LAST_DIR or os.getcwd() + os.sep

    @staticmethod
    def setDefaultDirectory(d: Any) -> None:
        global _LAST_DIR
        _LAST_DIR = str(d)

    @staticmethod
    def getLastName() -> str | None:
        return None


class SaveDialog:
    """``SaveDialog(title, name, ext)`` / ``SaveDialog(title, dir, name, ext)``."""

    def __init__(self, title: Any, *args: Any) -> None:
        ctx = current()
        self.title = str(title)
        if len(args) >= 3:
            d, name, ext = str(args[0]), str(args[1]), str(args[2])
        elif len(args) == 2:
            d, name, ext = "", str(args[0]), str(args[1])
        else:
            d, name, ext = "", str(args[0]) if args else "", ""
        v = ctx.param(self.title, "save", "path", "file", default=None)
        if v is not None:
            self._path: str | None = os.fspath(v)
        elif d:
            base = os.path.splitext(name)[0] if ext else name
            self._path = os.path.join(d, base + ext)
        else:
            self._path = None

    def getDirectory(self) -> str | None:
        return (os.path.dirname(self._path) + os.sep) if self._path else None

    def getFileName(self) -> str | None:
        return os.path.basename(self._path) if self._path else None

    def getPath(self) -> str | None:
        return self._path

    @staticmethod
    def setExtension(name: Any, ext: Any) -> str:
        return os.path.splitext(str(name))[0] + str(ext)


class DirectoryChooser:
    """``DirectoryChooser(title).getDirectory()``: the ``directory`` parameter (or the title's), else ``None``."""

    _default: str | None = None

    def __init__(self, title: Any = "Choose a directory") -> None:
        ctx = current()
        self.title = str(title)
        v = ctx.param(self.title, "directory", "dir", "folder", default=None)
        if v is None:
            self._dir: str | None = None
        else:
            p = os.fspath(v)
            self._dir = p if p.endswith(os.sep) else p + os.sep

    def getDirectory(self) -> str | None:
        return self._dir

    @staticmethod
    def setDefaultDirectory(d: Any) -> None:
        DirectoryChooser._default = str(d)


# ====================================================================== text windows / plots
class _TextPanel:
    def __init__(self, win: "TextWindow") -> None:
        self._w = win

    def appendLine(self, s: Any) -> None:
        self._w.append(s)

    def append(self, s: Any) -> None:
        self._w.append(s)

    def setText(self, s: Any) -> None:
        self._w._set(str(s))

    def getText(self) -> str:
        return self._w.getText()

    def clear(self) -> None:
        self._w._set("")

    def getLineCount(self) -> int:
        t = self._w.getText()
        return len(t.splitlines()) if t else 0

    def getLine(self, i: Any) -> str:
        lines = self._w.getText().splitlines()
        return lines[int(i)] if 0 <= int(i) < len(lines) else ""

    def setColumnHeadings(self, h: Any) -> None:
        pass

    def saveAs(self, path: Any) -> bool:
        with open(os.fspath(path), "w", encoding="utf-8") as fh:
            fh.write(self._w.getText())
        return True


class TextWindow:
    """``TextWindow(title, text, w, h)`` / ``TextWindow(title, headings, text, w, h)``: a Lumen text window."""

    def __init__(self, title: Any, *args: Any) -> None:
        ctx = current()
        self.title = str(title)
        text = ""
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
            text = (args[0] + "\n" if args[0] else "") + args[1]
        elif args and isinstance(args[0], str):
            text = args[0]
        ctx.ij._run("file.new_text", {"name": self.title, "text": text})
        self.title = next((k for k in reversed(list(ctx.session.tables)) if k == self.title or k.startswith(self.title)), self.title)

    def _set(self, text: str) -> None:
        current().ij._run("text.set", {"name": self.title, "text": text})

    def append(self, s: Any) -> None:
        current().ij._run("text.append", {"name": self.title, "text": str(s)})

    def getText(self) -> str:
        ctx = current()
        tb = ctx.session.tables.get(self.title)
        if not tb:
            return ""
        from ..ops.text_ops import table_text
        return table_text(tb)

    def getTextPanel(self) -> _TextPanel:
        return _TextPanel(self)

    def getTitle(self) -> str:
        return self.title

    def setVisible(self, b: Any) -> None:
        pass

    def show(self) -> None:
        pass

    def close(self) -> None:
        ctx = current()
        ctx.session.tables.pop(self.title, None)
        ctx.session._changed("state")

    def setLocation(self, *a: Any) -> None:
        pass

    def setSize(self, *a: Any) -> None:
        pass


class Plot:
    """``ij.gui.Plot(title, xLabel, yLabel[, xs, ys])``: series added with ``add`` / ``addPoints`` / ``drawLine``;
    ``show()`` (or ``update()``) puts the plot into the inspector's Plot section."""

    CIRCLE, X, LINE, BOX, TRIANGLE, CROSS, DOT, CONNECTED_CIRCLES, DIAMOND, BAR, SEPARATED_BAR, FILLED, CUSTOM = range(13)
    X_NUMBERS, Y_NUMBERS, X_TICKS, Y_TICKS, X_GRID, Y_GRID, X_FORCE2GRID, Y_FORCE2GRID = 1, 2, 4, 8, 16, 32, 64, 128
    X_LOG_NUMBERS, Y_LOG_NUMBERS, X_MINOR_TICKS, Y_MINOR_TICKS = 256, 512, 1024, 2048
    DEFAULT_FLAGS = X_NUMBERS + Y_NUMBERS + X_TICKS + Y_TICKS + X_GRID + Y_GRID

    def __init__(self, title: Any = "Plot", x_label: Any = "X", y_label: Any = "Y", xs: Any = None, ys: Any = None, flags: Any = None) -> None:
        self.title = str(title)
        self.x_label = str(x_label)
        self.y_label = str(y_label)
        self._series: list[dict[str, Any]] = []
        self._labels: list[dict[str, Any]] = []
        self._color = "#ff8800"
        self._line_width = 1.0
        self._limits: tuple[float, float, float, float] | None = None
        self._legend: list[str] = []
        self._log_x = self._log_y = False
        self._frame = (600, 300)
        if ys is not None and xs is not None:
            self.addPoints(xs, ys, self.LINE)
        elif xs is not None and ys is None and isinstance(xs, (list, tuple, np.ndarray)):  # Plot(title, x, y, ys) legacy form
            self.addPoints(list(range(len(xs))), xs, self.LINE)

    # --- data
    def _add(self, xs: Any, ys: Any, shape: Any, label: str | None = None) -> None:
        y = [float(v) for v in ys]
        x = [float(v) for v in xs] if xs is not None else [float(i) for i in range(len(y))]
        name = label or (self._legend[len(self._series)] if len(self._legend) > len(self._series) else f"Series {len(self._series) + 1}")
        self._series.append({"name": name, "values": y, "x": x, "color": self._color, "shape": shape, "width": self._line_width})

    def add(self, shape: Any, *args: Any) -> None:
        s = str(shape).lower() if isinstance(shape, str) else shape
        if len(args) == 1:
            self._add(None, args[0], s)
        else:
            self._add(args[0], args[1], s)

    def addPoints(self, *args: Any) -> None:
        """``addPoints(xs, ys, shape)`` / ``addPoints(xs, ys, yErrors, shape)`` / ``addPoints(xs, ys, shape, label)``."""
        if len(args) >= 4 and isinstance(args[2], (list, tuple, np.ndarray)):
            xs, ys, errs, shape = args[0], args[1], args[2], args[3]
            self._add(xs, ys, shape)
            self._series[-1]["errors"] = [float(v) for v in errs]
            return
        xs, ys = args[0], args[1]
        shape = args[2] if len(args) > 2 else self.LINE
        label = str(args[3]) if len(args) > 3 else None
        self._add(xs, ys, shape, label)

    def drawLine(self, x1: Any, y1: Any, x2: Any, y2: Any) -> None:
        self._add([x1, x2], [y1, y2], self.LINE, "line")

    def drawVectors(self, x1: Any, y1: Any, x2: Any, y2: Any) -> None:
        for a, b, c, d in zip(x1, y1, x2, y2):
            self.drawLine(a, b, c, d)

    def addErrorBars(self, errs: Any) -> None:
        if self._series:
            self._series[-1]["errors"] = [float(v) for v in errs]

    def addHorizontalErrorBars(self, errs: Any) -> None:
        pass

    def addLabel(self, x: Any, y: Any, text: Any) -> None:
        self._labels.append({"x": float(x), "y": float(y), "text": str(text)})

    addText = addLabel

    def addLegend(self, labels: Any, options: Any = None) -> None:
        self._legend = [s.strip() for s in str(labels).replace("\t", "\n").split("\n") if s.strip()]
        for s, name in zip(self._series, self._legend):
            s["name"] = name

    setLegend = addLegend

    # --- style
    def setColor(self, c: Any, c2: Any = None) -> None:
        self._color = color_hex(c, "#ff8800")

    def setLineWidth(self, w: Any) -> None:
        self._line_width = float(w)

    def setLimits(self, xmin: Any, xmax: Any, ymin: Any, ymax: Any) -> None:
        self._limits = (float(xmin), float(xmax), float(ymin), float(ymax))

    def setLimitsToFit(self, b: Any = True) -> None:
        self._limits = None

    def getLimits(self) -> list[float]:
        if self._limits:
            return list(self._limits)
        xs = [v for s in self._series for v in s["x"]]
        ys = [v for s in self._series for v in s["values"]]
        return [min(xs) if xs else 0.0, max(xs) if xs else 1.0, min(ys) if ys else 0.0, max(ys) if ys else 1.0]

    def setAxisXLog(self, b: Any) -> None:
        self._log_x = bool(b)

    def setAxisYLog(self, b: Any) -> None:
        self._log_y = bool(b)

    def setLogScaleX(self, b: Any = True) -> None:
        self._log_x = bool(b)

    def setLogScaleY(self, b: Any = True) -> None:
        self._log_y = bool(b)

    def setXYLabels(self, x: Any, y: Any) -> None:
        self.x_label, self.y_label = str(x), str(y)

    def setFrameSize(self, w: Any, h: Any) -> None:
        self._frame = (int(w), int(h))

    def setSize(self, w: Any, h: Any) -> None:
        self._frame = (int(w), int(h))

    def setFont(self, *a: Any) -> None:
        pass

    def setFormatFlags(self, *a: Any) -> None:
        pass

    def setJustification(self, *a: Any) -> None:
        pass

    def setBackgroundColor(self, *a: Any) -> None:
        pass

    def setStyle(self, *a: Any) -> None:
        pass

    def setOptions(self, *a: Any) -> None:
        pass

    def useTemplate(self, *a: Any) -> None:
        pass

    # --- output
    def _dict(self) -> dict[str, Any]:
        x = self._series[0]["x"] if self._series else []
        series = [{"name": s["name"], "values": s["values"], "x": s["x"], "color": s["color"]} for s in self._series]
        d: dict[str, Any] = {"title": self.title, "x": x, "series": series, "x_label": self.x_label, "y_label": self.y_label}
        if self._limits:
            d["limits"] = list(self._limits)
        if self._labels:
            d["labels"] = list(self._labels)
        if self._log_x or self._log_y:
            d["log_x"], d["log_y"] = self._log_x, self._log_y
        return d

    def show(self) -> None:
        ctx = current()
        ctx.session.last_plot = self._dict()
        ctx.session._changed("plot")

    update = show
    draw = show

    def getImagePlus(self) -> ImagePlus | None:
        """The plot rendered as an RGB image (shown as a tab, like ImageJ's plot window image)."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None
        w, h = self._frame
        fig, ax = plt.subplots(figsize=(max(w, 200) / 100, max(h, 150) / 100), dpi=100)
        for s in self._series:
            ax.plot(s["x"], s["values"], color=s["color"], linewidth=s["width"], marker="o" if s["shape"] in (self.CIRCLE, "circle", self.CONNECTED_CIRCLES) else None,
                    linestyle="none" if s["shape"] in (self.CIRCLE, "circle", self.DOT, "dot", self.X, "x", self.CROSS, "cross", self.BOX, "box") else "-", label=s["name"])
        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        if self._limits:
            ax.set_xlim(self._limits[0], self._limits[1])
            ax.set_ylim(self._limits[2], self._limits[3])
        if self._log_x:
            ax.set_xscale("log")
        if self._log_y:
            ax.set_yscale("log")
        fig.tight_layout()
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        from ..image import LumenImage
        img = LumenImage(np.ascontiguousarray(rgba[:, :, :3]), name=self.title, rgb=True)
        return ImagePlus(_img=img)

    def getProcessor(self) -> Any:
        imp = self.getImagePlus()
        return imp.getProcessor() if imp is not None else None

    def getResultsTable(self, *a: Any) -> Any:
        from ._tables import ResultsTable
        rt = ResultsTable(_title=self.title)
        n = max((len(s["values"]) for s in self._series), default=0)
        for i in range(n):
            rt.incrementCounter()
            for k, s in enumerate(self._series):
                if i < len(s["values"]):
                    rt.addValue("X" if k == 0 else f"X{k}", s["x"][i])
                    rt.addValue("Y" if k == 0 else f"Y{k}", s["values"][i])
        return rt

    def getPlotWindow(self) -> None:
        return None

    def getTitle(self) -> str:
        return self.title

    def setTitle(self, t: Any) -> None:
        self.title = str(t)

    def appendToStack(self) -> None:
        pass

    def dispose(self) -> None:
        pass


PlotWindow = Plot


class Toolbar:
    RECTANGLE, OVAL, POLYGON, FREEROI, LINE, POLYLINE, FREELINE, POINT, WAND, TEXT, MAGNIFIER, HAND, DROPPER, ANGLE = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14

    @staticmethod
    def getInstance() -> "Toolbar":
        return Toolbar()

    @staticmethod
    def getToolId(*a: Any) -> int:
        return 0

    @staticmethod
    def getToolName() -> str:
        return current().ij.getToolName()

    @staticmethod
    def setTool(name: Any) -> bool:
        current().ij.setTool(name)
        return True

    @staticmethod
    def getForegroundColor() -> Any:
        from ._java import Color
        return Color(current().ij._fg_hex)

    @staticmethod
    def getBackgroundColor() -> Any:
        from ._java import Color
        return Color(current().ij._bg_hex)

    @staticmethod
    def setForegroundColor(c: Any) -> None:
        current().ij.setForegroundColor(color_hex(c))

    @staticmethod
    def setBackgroundColor(c: Any) -> None:
        current().ij.setBackgroundColor(color_hex(c))

    @staticmethod
    def addPlugInTool(*a: Any) -> None:
        pass

    @staticmethod
    def restorePreviousTool() -> None:
        pass


__all__ = ["GenericDialog", "NonBlockingGenericDialog", "GenericDialogPlus", "WaitForUserDialog", "YesNoCancelDialog", "MessageDialog", "HTMLDialog",
           "OpenDialog", "SaveDialog", "DirectoryChooser", "TextWindow", "Plot", "PlotWindow", "Toolbar", "ensure_in_session"]
