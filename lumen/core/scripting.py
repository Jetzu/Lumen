"""The `lumen` object exposed to the built-in Python console and macros."""
from __future__ import annotations

import contextlib
import io
import traceback
from typing import Any

import numpy as np

from .image import LumenImage
from .registry import REGISTRY
from .session import Session


class LumenAPI:
    """Scripting façade.  Example::

        img = lumen.open("/data/cells.tif")
        lumen.run("filters.gaussian", sigma=2)
        lumen.run("threshold.auto", method="Otsu")
        rows = lumen.run("analyze.particles", min_size=50)
        print(lumen.results)
    """

    def __init__(self, session: Session) -> None:
        self._s = session
        self.roi: dict[str, Any] | None = None

    # images
    @property
    def image(self) -> LumenImage | None:
        return self._s.active

    @property
    def images(self) -> list[LumenImage]:
        return list(self._s.images.values())

    @property
    def data(self) -> np.ndarray:
        """The active image as a (T, Z, C, Y, X) numpy array (a view – edit then call update())."""
        return self._s.get_image().data

    @property
    def plane(self) -> np.ndarray:
        return self._s.get_image().current_plane()

    def open(self, path: str) -> LumenImage:
        return self._s.open_file(path)

    def save(self, path: str, fmt: str | None = None) -> str:
        return self._s.save_image(self._s.get_image().id, path, fmt)

    def select(self, name_or_id: str) -> LumenImage:
        img = self._s.get_image(name_or_id)
        self._s.select(img.id)
        return img

    def close(self, name_or_id: str | None = None) -> None:
        img = self._s.get_image(name_or_id)
        self._s.close_image(img.id)

    def undo(self, name_or_id: str | None = None) -> str | None:
        """Undo the last pixel-modifying command of the image (default: the active one); returns its label."""
        return self._s.undo(self._s.get_image(name_or_id).id, record=False)

    def redo(self, name_or_id: str | None = None) -> str | None:
        return self._s.redo(self._s.get_image(name_or_id).id, record=False)

    def display(self, name_or_id: str | None = None, channel: int | None = None, **settings: Any) -> dict[str, Any]:
        """Change how an image is shown (what the recorder writes for inspector edits)::

            lumen.display("cells.tif", channel=1, min=0, max=1500, lut="Fire", visible=True, inverted=False, gamma=1.0)
            lumen.display("cells.tif", mode="composite")      # composite | single | grayscale
            lumen.display("cells.tif", t=0, z=12, c=1)          # the displayed plane (0-based)

        ``channel`` is 1-based (like the Results "Ch" column); without it the per-channel settings apply to the
        current channel.  Returns the image's display dict.
        """
        img = self._s.get_image(name_or_id)
        allowed = {"min", "max", "lut", "visible", "inverted", "gamma", "mode", "t", "z", "c"}
        unknown = [k for k in settings if k not in allowed]
        if unknown:
            raise ValueError(f"lumen.display: unknown setting {unknown[0]!r} (allowed: {', '.join(sorted(allowed))})")
        patch: dict[str, Any] = {k: settings[k] for k in ("mode", "t", "z", "c") if k in settings}
        chp = {k: settings[k] for k in ("min", "max", "lut", "visible", "inverted", "gamma") if k in settings}
        if chp:
            idx = img.display.c if channel is None else int(channel) - 1
            if not 0 <= idx < img.nc:
                raise ValueError(f"lumen.display: {img.name} has {img.nc} channel(s), channel {channel} does not exist")
            patch["channels"] = [chp if i == idx else {} for i in range(img.nc)]
        self._s.update_display(img.id, patch)
        return img.display.to_dict()

    def new_image(self, array: np.ndarray, name: str = "Untitled", axes: str | None = None, like: LumenImage | None = None) -> LumenImage:
        img = LumenImage(np.asarray(array), name=name, axes=axes)
        if like is not None:
            img.calibration = type(like.calibration)(**like.calibration.to_dict())
        return self._s.add_image(img)

    def update(self, array: np.ndarray | None = None, label: str = "Script edit") -> None:
        """Commit a numpy edit to the active image (records an undo step)."""
        img = self._s.get_image()
        self._s.histories[img.id].push(img, label)
        if array is not None:
            img.replace_data(LumenImage.to_canonical(np.asarray(array), None, img.rgb))
        else:
            img.sync_channels()
        self._s._changed("images")

    # commands
    def run(self, command: str, scope: str = "stack", roi: dict[str, Any] | None = None, image: str | None = None, **params: Any) -> Any:
        img_id = self._s.get_image(image).id if image else None
        out = self._s.run_op(command, params, image_id=img_id, roi=roi or self.roi, scope=scope, record=False)
        if "plot" in out:
            return out["plot"]
        if out.get("results_added"):
            return self._s.results.rows[-out["results_added"]:]
        if out.get("new_images"):
            return [self._s.images[i] for i in out["new_images"] if i in self._s.images]
        return out

    def commands(self, search: str = "") -> list[str]:
        s = search.lower()
        return [f"{spec.id:32s} {spec.name}" for spec in REGISTRY.ops.values() if s in spec.id.lower() or s in spec.name.lower()]

    def help(self, command: str) -> str:
        spec = REGISTRY.get(command)
        lines = [f"{spec.name}  [{spec.id}]  – {spec.description}"]
        for p in spec.params:
            extra = f" choices={p.choices}" if p.choices else ""
            lines.append(f"  {p.name}: {p.type} = {p.default!r}{extra}  {p.help}")
        return "\n".join(lines)

    # results & rois
    @property
    def results(self) -> list[dict[str, Any]]:
        return self._s.results.rows

    def clear_results(self) -> None:
        self._s.results.clear()
        self._s._changed("results")

    @property
    def rois(self) -> list[dict[str, Any]]:
        return self._s.roi_manager.to_list()

    def add_roi(self, roi: dict[str, Any]) -> dict[str, Any]:
        if roi.get("id") and any(x.get("id") == roi["id"] for x in self._s.roi_manager.rois):
            roi = {k: v for k, v in roi.items() if k != "id"}  # re-adding a manager entry makes a new, independent one
        r = self._s.roi_manager.add(roi, image_id=self._s.active_id)
        self._s._changed("rois")
        return r

    def set_roi(self, roi: dict[str, Any] | None) -> None:
        """Set the selection used by subsequent run() calls."""
        self.roi = roi

    def rect(self, x: float, y: float, w: float, h: float) -> dict[str, Any]:
        self.roi = {"type": "rect", "x": x, "y": y, "w": w, "h": h}
        return self.roi

    def log(self, *msg: Any) -> None:
        self._s.add_log(" ".join(str(m) for m in msg))

    @property
    def session(self) -> Session:
        return self._s

    # ImageJ compatibility
    @property
    def ij(self) -> Any:
        """The ImageJ façade: ``lumen.ij.run("Gaussian Blur...", "sigma=2")``, ``lumen.ij.macro(text)``.
        Its macro helpers (run, selectWindow, makeRectangle, nResults, getResult …) are also injected into scripts."""
        if getattr(self, "_ij", None) is None:
            from .ij_compat import IJ
            self._ij = IJ(self._s, api=self)
        return self._ij


def format_script_error(exc: BaseException) -> str:
    """The traceback of a failed script reduced to what concerns its author: the frames inside the script
    (``<lumen-script>``, with line numbers) and the exception line – Lumen's own frames and file paths are left out
    (they go to the server log).  ``KeyError`` messages lose their repr quotes."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<lumen-script>"]
    head = "Traceback (most recent call last):\n" + "".join(traceback.format_list(frames)) if frames else ""
    if isinstance(exc, KeyError) and exc.args:
        tail = f"KeyError: {exc.args[0]}\n"
    else:
        tail = "".join(traceback.format_exception_only(type(exc), exc))
    return head + tail


def _script_exit(code: Any = None) -> None:
    """exit()/quit() inside a script end the script only (ImageJ ``exit`` semantics).  The builtin
    site Quitter would additionally close the server's sys.stdin before raising."""
    raise SystemExit(code)


def run_script(session: Session, code: str) -> dict[str, Any]:
    api = LumenAPI(session)
    g: dict[str, Any] = {"lumen": api, "np": np, "numpy": np, "__name__": "__lumen__",
                         "exit": _script_exit, "quit": _script_exit}
    try:
        import skimage
        import scipy
        g["skimage"] = skimage
        g["scipy"] = scipy
    except Exception:  # pragma: no cover
        pass
    try:  # ImageJ macro functions (run("…"), selectWindow, nResults, getResult …) and the IJ object
        for k, v in api.ij.script_globals().items():
            g.setdefault(k, v)
    except Exception:  # pragma: no cover
        pass
    buf = io.StringIO()
    err = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            try:
                exec(compile(code, "<lumen-script>", "exec"), g)
            except SystemExit as e:
                # exit()/quit()/sys.exit(): an early end of the *script*, never of the request, the batch
                # or the server.  None/0 = clean end; a message or non-zero status is the script's error.
                if e.code not in (None, 0):
                    err = e.code if isinstance(e.code, str) else f"SystemExit: {e.code}"
            except Exception as e:
                err = format_script_error(e)
                try:
                    import logging
                    logging.getLogger("lumen.script").debug("script failed", exc_info=e)
                except Exception:  # pragma: no cover
                    pass
    finally:
        session._changed("images")
    return {"stdout": buf.getvalue(), "error": err}
