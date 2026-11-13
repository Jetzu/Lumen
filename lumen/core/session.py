"""The session: open images, results table, ROI manager, undo history, recorder."""
from __future__ import annotations

import contextvars
import json
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import roi as roimod
from .image import LumenImage, Calibration
from .registry import REGISTRY, OpContext, OperationCancelled, OpResult, OpSpec, apply_plane_op
from .undo import History


def _same_metadata(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Equality of two image metadata dicts that never raises: a value without plain equality (an ndarray of fit
    coefficients, say) makes ``==`` ambiguous, and inside run_op's failure handler a secondary error would mask the
    real one – such metadata counts as changed, and restoring the pre-op snapshot is always safe."""
    try:
        return bool(a == b)
    except Exception:
        return False


class ResultsTable:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.columns: list[str] = []

    def append(self, rows: list[dict[str, Any]]) -> None:
        for r in rows:
            for k in r.keys():
                if k not in self.columns:
                    self.columns.append(k)
            self.rows.append(r)

    def clear(self) -> None:
        self.rows = []
        self.columns = []

    def to_dict(self) -> dict[str, Any]:
        cols = [c for c in self.columns]
        return {"columns": cols, "rows": [[_json_safe(r.get(c)) for c in cols] for r in self.rows]}

    def to_csv(self) -> str:
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow([" "] + self.columns)
        for i, r in enumerate(self.rows, 1):
            w.writerow([i] + [_fmt(r.get(c)) for c in self.columns])
        return buf.getvalue()


def _fmt(v: Any) -> Any:
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if np.isfinite(v) else "NaN"
    return v


def _json_safe(v: Any) -> Any:
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, (np.integer,)):
        v = int(v)
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


#: the client (browser tab / API user) of the request being served: the server's middleware sets it from the
#: ``X-Lumen-Client`` header, so every recorder line lands in that client's own recorder (E7-4). Unset (direct Session
#: use, tests, scripts run outside a request) = the ``default`` recorder.
CURRENT_CLIENT: "contextvars.ContextVar[str | None]" = contextvars.ContextVar("lumen_client", default=None)
DEFAULT_CLIENT = "default"
#: recorders of clients not heard from for the longest time are dropped beyond this many (each page load is a new client)
MAX_RECORDERS = 32
#: ops whose ``channel`` parameter (0 = the current channel) is recorded as the explicit 1-based channel on multi-channel
#: images, so the replayed line does not depend on the hidden channel slider (E7-11)
CHANNEL_RECORDED_OPS = {"threshold.compute", "threshold.apply", "analyze.measure", "analyze.particles", "color.lut"}


class Recorder:
    """The macro recorder of one client: its Python lines, when it started and which image its last line targeted.

    Lines are Python for the scripting API (``lumen.run(...)``, ``lumen.open(...)``, ``lumen.select("title")``,
    ``lumen.close("title")``, ``lumen.undo("title")``, ``lumen.display("title", channel=1, min=…)``); ``marker()`` is
    the ``#`` comment naming the session start (shown by the recorder pane and written at the top of exports; not one
    of the lines, so ``len(lines)`` counts commands).  ``active_image`` is the id of the image the last recorded command
    ran on: a command on another image is preceded by ``lumen.select("<title>")`` (Fiji's selectWindow), so a replay
    runs every step on the image it was recorded on (E7-1).
    """

    def __init__(self, client: str, active_image: str | None = None) -> None:
        self.client = client
        self.started = 0.0
        self.touched = 0.0
        self.lines: list[str] = []
        self.active_image: str | None = None
        self._display: tuple[str, Any, int, dict[str, Any]] | None = None  # last display line: image id, channel key, index, kwargs
        self.reset(active_image)

    def reset(self, active_image: str | None = None) -> None:
        self.started = self.touched = time.time()
        self.lines = []
        self.active_image = active_image
        self._display = None

    def marker(self) -> str:
        return f"# Lumen recorder – session started {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.started))}"

    def add(self, line: str) -> None:
        self.lines.append(line)
        if len(self.lines) > 2000:
            del self.lines[:len(self.lines) - 2000]
        self.touched = time.time()
        self._display = None

    def target(self, image_id: str, name: str) -> None:
        """Make the image the next line runs on: records ``lumen.select("<title>")`` when it is not already.  ``name``
        is the title the image had *before* the command (Save As / Rename change it; the replay selects the old one)."""
        if image_id != self.active_image:
            self.add(f"lumen.select({json.dumps(name)})")
            self.active_image = image_id

    def add_display(self, image: "LumenImage", key: Any, kwargs: dict[str, Any]) -> None:
        """Record a ``lumen.display`` line; successive edits of the same image and channel (``key``) collapse into one
        line (a slider drag arrives as several debounced patches)."""
        last = self._display
        if last is not None and last[0] == image.id and last[1] == key and last[2] == len(self.lines) - 1:
            merged = {**last[3], **kwargs}
        else:
            self.target(image.id, image.name)
            merged = dict(kwargs)
        ordered = {k: merged[k] for k in ("channel", "min", "max", "lut", "visible", "inverted", "gamma", "mode", "t", "z", "c") if k in merged}
        line = f"lumen.display({json.dumps(image.name)}, " + ", ".join(f"{k}={_py(v)}" for k, v in ordered.items()) + ")"
        if last is not None and last[0] == image.id and last[1] == key and last[2] == len(self.lines) - 1:
            self.lines[-1] = line
        else:
            self.add(line)
        self._display = (image.id, key, len(self.lines) - 1, merged)
        self.touched = time.time()

    def info(self) -> dict[str, Any]:
        return {"client": self.client, "started": self.started, "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.started)),
                "marker": self.marker(), "active_image": self.active_image, "lines": list(self.lines), "count": len(self.lines)}


class Session:
    def __init__(self) -> None:
        self.images: "OrderedDict[str, LumenImage]" = OrderedDict()
        self.active_id: str | None = None
        self.results = ResultsTable()
        self.roi_manager = roimod.RoiManager()
        self.histories: dict[str, History] = {}
        self.recorders: dict[str, Recorder] = {}
        #: the last cleared ROI Manager / Results contents (one-step "Undo clear", E7-6): see restore_last_clear
        self.last_clear: dict[str, Any] | None = None
        self.log: list[dict[str, Any]] = []
        self.version = 0
        self.lock = threading.RLock()
        self.listeners: list[Callable[[dict[str, Any]], None]] = []
        self.measurements: list[str] = ["area", "mean", "min", "max"]
        self.measure_options: dict[str, Any] = {"limit_to_threshold": False, "decimals": 3, "display_label": True}
        self.thresholds: dict[str, dict[str, Any]] = {}  # image_id -> {lower, upper}
        self.settings: dict[str, Any] = {"foreground": 255, "background": 0}
        self.last_plot: dict[str, Any] | None = None
        self.tables: dict[str, dict[str, Any]] = {}
        self.recent: list[str] = self._load_recent()
        # long operations: job records by id (see run_op). Guarded by their own small lock, never by
        # self.lock, so that GET /api/jobs and POST /api/jobs/{id}/cancel work while an op holds the session lock.
        self.jobs: dict[str, dict[str, Any]] = {}
        self._jobs_lock = threading.Lock()
        self._job_local = threading.local()  # the job of the op running in this thread (nested ops share it)

    # ------------------------------------------------------------- events
    def _emit(self, evt: dict[str, Any]) -> None:
        for cb in list(self.listeners):
            try:
                cb(evt)
            except Exception:
                pass

    def _changed(self, what: str = "state", **extra: Any) -> None:
        self.version += 1
        self._emit({"type": what, "version": self.version, **extra})

    # ------------------------------------------------------------- jobs (progress + cancellation)
    #: minimum spacing of progress events per job (s): at most 20 per second
    PROGRESS_INTERVAL = 0.05

    @staticmethod
    def job_info(job: dict[str, Any]) -> dict[str, Any]:
        """The JSON view of a job record (the ``cancel`` Event becomes the ``cancelled`` flag)."""
        return {"id": job["id"], "op": job["op"], "name": job["name"], "started": job["started"],
                "fraction": job["fraction"], "message": job["message"], "cancelled": job["cancel"].is_set(),
                "elapsed": round(time.time() - job["started"], 3)}

    def list_jobs(self) -> list[dict[str, Any]]:
        """The operations running right now (lock-free with respect to the session lock)."""
        with self._jobs_lock:
            return [self.job_info(j) for j in self.jobs.values()]

    def cancel_job(self, job_id: str) -> bool:
        """Ask the running op ``job_id`` to stop; it raises OperationCancelled at its next check_cancel()."""
        with self._jobs_lock:
            job = self.jobs.get(job_id)
        if job is None:
            return False
        job["cancel"].set()
        return True

    def _job_start(self, spec: OpSpec) -> dict[str, Any]:
        job = {"id": uuid.uuid4().hex[:12], "op": spec.id, "name": spec.name, "started": time.time(),
               "fraction": 0.0, "message": "", "cancel": threading.Event(), "_emitted": 0.0}
        with self._jobs_lock:
            self.jobs[job["id"]] = job
        # _emitted stays 0 so the op's first progress() (the one that carries its first message) is never
        # throttled away behind the start event; from then on updates are spaced >= PROGRESS_INTERVAL
        self._emit({"type": "progress", "job": self.job_info(job)})
        return job

    def _job_progress(self, job: dict[str, Any], fraction: float, message: str = "") -> None:
        try:
            fraction = float(fraction)
        except (TypeError, ValueError):
            fraction = 0.0
        if fraction != fraction:  # NaN
            fraction = 0.0
        fraction = min(max(fraction, 0.0), 1.0)
        now = time.time()
        with self._jobs_lock:
            job["fraction"] = fraction
            if message:
                job["message"] = str(message)
            emit = now - job["_emitted"] >= self.PROGRESS_INTERVAL
            if emit:
                job["_emitted"] = now
        if emit:
            self._emit({"type": "progress", "job": self.job_info(job)})

    def _job_end(self, job: dict[str, Any], status: str) -> None:
        """Drop the job and broadcast its final event (``done: true``, ``status`` ok | error | cancelled)."""
        with self._jobs_lock:
            self.jobs.pop(job["id"], None)
        info = self.job_info(job)
        info.update(done=True, status=status, cancelled=status == "cancelled")
        if status == "ok":
            info["fraction"] = 1.0
        self._emit({"type": "progress", "job": info})

    def add_log(self, msg: str, level: str = "info") -> None:
        self.log.append({"t": time.time(), "level": level, "msg": str(msg)})
        self.log = self.log[-500:]

    # ------------------------------------------------------------- images
    def add_image(self, img: LumenImage, activate: bool = True) -> LumenImage:
        with self.lock:
            img.name = self.unique_name(img.name)
            self.images[img.id] = img
            self.histories[img.id] = History()
            if activate or self.active_id is None:
                self.active_id = img.id
            self._changed("images")
        return img

    def unique_name(self, name: str) -> str:
        names = {im.name for im in self.images.values()}
        if name not in names:
            return name
        base = name
        k = 1
        while f"{base}-{k}" in names:
            k += 1
        return f"{base}-{k}"

    def close_image(self, image_id: str) -> None:
        with self.lock:
            self.images.pop(image_id, None)
            self.histories.pop(image_id, None)
            self.thresholds.pop(image_id, None)
            if self.active_id == image_id:
                self.active_id = next(reversed(self.images)) if self.images else None
            for rec in self.recorders.values():  # the next recorded command names its image again
                if rec.active_image == image_id:
                    rec.active_image = None
            self._changed("images")

    def get_image(self, key: str | None = None) -> LumenImage:
        if key is None:
            if self.active_id is None:
                raise ValueError("No image is open")
            return self.images[self.active_id]
        if key in self.images:
            return self.images[key]
        for im in self.images.values():
            if im.name == key:
                return im
        raise KeyError(f"No image {key!r}")

    @property
    def active(self) -> LumenImage | None:
        return self.images.get(self.active_id) if self.active_id else None

    def select(self, image_id: str) -> None:
        if image_id in self.images:
            self.active_id = image_id
            self._changed("active")

    # ------------------------------------------------------------- recent files
    @staticmethod
    def _recent_path():
        from .io import lumen_home
        return lumen_home() / "recent.json"

    def _load_recent(self) -> list[str]:
        try:
            data = json.loads(self._recent_path().read_text(encoding="utf-8"))
            return [str(p) for p in data][:20] if isinstance(data, list) else []
        except Exception:
            return []

    def add_recent(self, path: str) -> None:
        path = str(path)
        self.recent = [path] + [p for p in self.recent if p != path]
        self.recent = self.recent[:20]
        try:
            rp = self._recent_path()
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(json.dumps(self.recent), encoding="utf-8")
        except Exception:
            pass

    def clear_recent(self) -> None:
        self.recent = []
        try:
            self._recent_path().write_text("[]", encoding="utf-8")
        except Exception:
            pass

    def open_file(self, path: str, series: int | None = None, level: int = 0, virtual: bool | str = "auto") -> LumenImage:
        from .io import read_image
        img = read_image(path, series=series, level=level, virtual=virtual)
        self.add_image(img)
        self.add_recent(path)
        self.record(f'lumen.open({json.dumps(path)})  # → {json.dumps(img.name)}')
        self.recorder().active_image = img.id
        self.add_log(f"Opened {img.name} {img.shape} {img.bit_depth}")
        names = img.metadata.get("series_names")
        if names and len(names) > 1:
            self.add_log(f"{img.name}: series {img.metadata.get('series', 0)} of {len(names)} opened – others: "
                         + ", ".join(f"{i}: {n}" for i, n in enumerate(names)) + " (File ▸ Import ▸ Bio-Formats… opens them)")
        return img

    def save_image(self, image_id: str, path: str, fmt: str | None = None) -> str:
        from .io import write_image
        img = self.get_image(image_id)
        if img.path and not img.metadata.get("source_path") and not img.metadata.get("derived_from"):
            img.metadata["source_path"] = img.path  # the file this image was opened from stays the source after a Save As (E10-4)
        out = write_image(img, path, fmt, history=self.histories.get(img.id))  # the processing log goes into the ImageJ Info tag (E7-16)
        img.path = out
        # ImageJ semantics (E7-17): Save As renames the window to the file name, so getTitle() / the tab / Image Info
        # agree with the path afterwards and later selectWindow("saved.tif") lines find it
        self.recorder().target(img.id, img.name)  # by the title it has now: the replay saves (and renames) it too
        new_name = Path(out).name
        if new_name and new_name != img.name:
            old = img.name
            img.name = self.unique_name(new_name)
            self.add_log(f"{old} renamed to {img.name} (Save As)")
        self.record(f'lumen.save({json.dumps(out)})')
        self.add_log(f"Saved {out}")
        self._changed("images")
        return out

    # ------------------------------------------------------------- display
    def update_display(self, image_id: str, patch: dict[str, Any], record: bool = False) -> None:
        """Apply a display patch (position, mode, per-channel min/max/lut/visible/inverted/gamma/name).

        ``record=True`` (the display endpoint) writes the *effective* change to the recorder as
        ``lumen.display("<title>", channel=1, min=…, max=…, lut=…, visible=…, inverted=…)`` / ``mode=…`` / ``t=… z=… c=…``
        lines, so figure appearance is reproducible like Fiji's setMinAndMax() / run("Fire") records (E7-3).
        """
        img = self.get_image(image_id)
        d = img.display
        before = d.to_dict() if record else None
        if record:  # a display edit (not a plane change) is an undo step of its own (E5-8); a slider drag merges into one
            label, detail, group = _display_step(d, patch)
            if label:
                self.histories[image_id].push_display(img, label, detail, group)
        for k in ("t", "z", "c"):
            if k in patch and patch[k] is not None:
                setattr(d, k, int(patch[k]))
        if "mode" in patch and patch["mode"] in ("composite", "single", "grayscale"):
            d.mode = patch["mode"]
        if "channels" in patch:
            for i, chp in enumerate(patch["channels"]):
                if i < len(d.channels) and chp:
                    ch = d.channels[i]
                    for k in ("name", "lut", "min", "max", "visible", "inverted", "gamma"):
                        if k in chp and chp[k] is not None:
                            setattr(ch, k, chp[k])
        img.clamp_position()
        self.version += 1  # display changes don't broadcast (client already knows)
        if before is not None:
            self._record_display(img, before, d.to_dict())

    def _record_display(self, img: LumenImage, before: dict[str, Any], after: dict[str, Any]) -> None:
        rec = self.recorder()
        for i, (b, a) in enumerate(zip(before["channels"], after["channels"])):
            changed = {k: a[k] for k in ("min", "max", "lut", "visible", "inverted", "gamma") if a.get(k) != b.get(k)}
            if changed:
                rec.add_display(img, i + 1, {"channel": i + 1, **changed})
        if after["mode"] != before["mode"]:
            rec.add_display(img, "mode", {"mode": after["mode"]})
        pos = {k: after[k] for k in ("t", "z", "c") if after[k] != before[k]}
        if pos:
            rec.add_display(img, "position", pos)

    # ------------------------------------------------------------- undo
    def undo(self, image_id: str, record: bool = True) -> str | None:
        img = self.get_image(image_id)
        label = self.histories[image_id].do_undo(img)
        if label:
            self.add_log(f"Undo {label}")
            if record:  # the recorded script reproduces the image the user ends up with (E7-2)
                self.record(f"lumen.undo({json.dumps(img.name)})")
            self._changed("images")
        return label

    def redo(self, image_id: str, record: bool = True) -> str | None:
        img = self.get_image(image_id)
        label = self.histories[image_id].do_redo(img)
        if label:
            self.add_log(f"Redo {label}")
            if record:
                self.record(f"lumen.redo({json.dumps(img.name)})")
            self._changed("images")
        return label

    # ------------------------------------------------------------- recorder (one per client, see Recorder)
    def recorder(self, client: str | None = None) -> Recorder:
        """The recorder of ``client`` (default: the client of the request being served, else the shared ``default``)."""
        key = client or CURRENT_CLIENT.get() or DEFAULT_CLIENT
        rec = self.recorders.get(key)
        if rec is None:
            rec = self.recorders[key] = Recorder(key, self.active_id)
            if len(self.recorders) > MAX_RECORDERS:
                stale = sorted((r for k, r in self.recorders.items() if k != DEFAULT_CLIENT), key=lambda r: r.touched)
                for r in stale[:len(self.recorders) - MAX_RECORDERS]:
                    self.recorders.pop(r.client, None)
        rec.touched = time.time()
        return rec

    @property
    def macro(self) -> list[str]:
        """The current client's recorded lines (the list itself: ``session.macro.clear()`` empties the recorder)."""
        return self.recorder().lines

    @macro.setter
    def macro(self, lines: list[str]) -> None:
        self.recorder().lines = list(lines)

    def record(self, line: str) -> None:
        self.recorder().add(line)

    def clear_recorder(self, client: str | None = None) -> int:
        """Empty the client's recorder (its session marker restarts); returns the number of lines dropped."""
        rec = self.recorder(client)
        n = len(rec.lines)
        rec.reset(self.active_id)
        return n

    def _recorded_params(self, spec: OpSpec, params: dict[str, Any], image: LumenImage | None) -> dict[str, Any]:
        """The parameters as written to the recorder: the channel-aware analysis ops get the explicit 1-based channel
        on multi-channel images (channel=0 means "the current one", which a replay cannot know) and no channel at all
        on single-channel / RGB images."""
        if spec.id in CHANNEL_RECORDED_OPS and "channel" in params and image is not None:
            params = dict(params)
            if image.nc > 1 and not image.rgb:
                params["channel"] = int(params["channel"]) or image.display.c + 1
            else:
                params["channel"] = None
        return params

    def _record_op(self, spec: OpSpec, params: dict[str, Any], scope: str, roi: dict[str, Any] | None, image: LumenImage | None,
                   result: OpResult, payload: dict[str, Any], name_before: str | None = None) -> None:
        rec = self.recorder()
        name = name_before if (name_before is not None and image is not None) else (image.name if image is not None else "")
        if spec.id == "file.close" and image is not None:
            rec.add(f"lumen.close({json.dumps(name)})")  # closes by title (Fiji's close("title"))
            return  # close_image() already forgot it as the recorder's target
        if image is not None:
            rec.target(image.id, name)  # the title before the command: Rename / Save As change it
        line = _macro_line(spec, self._recorded_params(spec, params, image), scope, roi)
        new_ids = payload.get("new_images") or []
        names = [self.images[i].name for i in new_ids if i in self.images]
        if names:  # the title the new image got (unique_name may have suffixed it), the name later select lines use
            line += "  # → " + ", ".join(json.dumps(n) for n in names)
        rec.add(line)
        if new_ids:
            rec.active_image = new_ids[-1] if new_ids[-1] in self.images else None
        elif spec.id == "file.close_all":
            rec.active_image = None
        if result.select_image:
            rec.active_image = result.select_image

    # ------------------------------------------------------------- destructive clears (restorable, E7-6)
    def clear_rois(self, record: bool = True) -> int:
        """Empty the ROI Manager; the entries are kept for restore_last_clear(). Returns the number cleared."""
        with self.lock:
            rois = [dict(r) for r in self.roi_manager.to_list()]
            if rois:
                self.last_clear = {"kind": "rois", "rois": rois, "count": len(rois), "t": time.time()}
            self.roi_manager.clear()
            if record:
                self.record('lumen.run("rois.clear")')
            self._changed("rois")
        return len(rois)

    def delete_rois(self, ids: list[str], record: bool = True) -> int:
        """Delete the ROI Manager entries ``ids`` (restorable); returns the number deleted."""
        with self.lock:
            all_rois = self.roi_manager.to_list()
            idx = [i for i, r in enumerate(all_rois) if r.get("id") in set(ids)]
            gone = [(i, dict(all_rois[i])) for i in idx]
            if gone:
                self.last_clear = {"kind": "rois", "rois": [r for _, r in gone], "indices": [i for i, _ in gone], "count": len(gone), "t": time.time()}
                self.roi_manager.remove([r["id"] for _, r in gone])
                if record:
                    self.record(f'lumen.run("rois.delete", indices={_py(",".join(str(i) for i, _ in gone))})')
            self._changed("rois")
        return len(gone)

    def clear_results(self, record: bool = True) -> int:
        """Empty the Results table (restorable); returns the number of rows cleared."""
        with self.lock:
            n = len(self.results.rows)
            if n:
                self.last_clear = {"kind": "results", "rows": [dict(r) for r in self.results.rows], "columns": list(self.results.columns), "count": n, "t": time.time()}
            self.results.clear()
            if record:
                self.record('lumen.run("analyze.clear_results")')
            self._changed("results")
        return n

    def restore_last_clear(self, kind: str | None = None) -> dict[str, Any]:
        """Put back what the last clear_rois / delete_rois / clear_results removed (one step). ``kind`` = "rois" |
        "results" restores only that kind; returns ``{"kind", "restored"}`` (restored 0 when there is nothing)."""
        with self.lock:
            lc = self.last_clear
            if lc is None or (kind and lc["kind"] != kind):
                return {"kind": kind or (lc or {}).get("kind"), "restored": 0}
            self.last_clear = None
            if lc["kind"] == "rois":
                current = list(self.roi_manager.rois)
                indices = lc.get("indices")
                if indices:
                    for i, r in zip(indices, lc["rois"]):  # deleted entries go back to their old positions
                        current.insert(min(max(int(i), 0), len(current)), r)
                else:
                    current = lc["rois"] + current
                self.roi_manager.rois = current
                self._changed("rois")
            else:
                rows = lc["rows"] + list(self.results.rows)
                self.results.clear()
                self.results.append(rows)
                for c in lc["columns"]:
                    if c not in self.results.columns:
                        self.results.columns.append(c)
                self._changed("results")
            return {"kind": lc["kind"], "restored": lc["count"]}

    # ------------------------------------------------------------- ops
    def run_op(self, op_id: str, params: dict[str, Any] | None = None, image_id: str | None = None,
               roi: dict[str, Any] | None = None, scope: str = "stack", preview: bool = False,
               record: bool = True) -> dict[str, Any]:
        spec = REGISTRY.get(op_id)
        params = spec.coerce_params(params)
        with self.lock:
            image = None
            if spec.requires_image:
                image = self.get_image(image_id) if image_id else self.active
                if image is None:
                    raise ValueError("Open an image first")
                if image.rgb and not spec.supports_rgb:
                    raise ValueError(f"{spec.name} does not work on RGB images – convert to 8-bit or split channels first")
            if spec.requires_roi:
                if roi is None:
                    kind = {"area": "an area", "line": "a line", "line_or_rect": "a line or rectangle", "point": "a point"}.get(spec.requires_roi, "a")
                    raise ValueError(f"{spec.name} needs {kind} selection")
                if spec.requires_roi == "line_or_rect" and not (roimod.is_line(roi) or roi.get("type") == "rect"):
                    raise ValueError(f"{spec.name} needs a line or rectangle selection")
                if spec.requires_roi == "area" and not roimod.is_area(roi):
                    raise ValueError(f"{spec.name} needs an area selection")
                if spec.requires_roi == "line" and not roimod.is_line(roi):
                    raise ValueError(f"{spec.name} needs a line selection")
            if roi is not None:
                roi = roimod.normalize(roi)
            ctx = OpContext(session=self, image=image, roi=roi, scope=scope if spec.kind == "plane" else "stack", preview=preview)
            if preview:
                return self._preview(spec, ctx, params)
            # the job record: progress events + cancel flag. A command run by a script or batch from inside
            # another command shares the outer job (its progress bar, its cancel flag) instead of starting its own.
            outer = getattr(self._job_local, "job", None)
            if outer is not None:
                ctx.job, ctx.nested, job = outer, True, None
            else:
                job = ctx.job = self._job_start(spec)
                self._job_local.job = job
            status = "error"
            t0 = time.time()
            # snapshot before the op runs (some image ops edit img.display/calibration in place) but
            # commit it to the history only once the op has succeeded and actually changed the image:
            # a failed or non-modifying command (Duplicate, Z Project …) leaves undo AND redo untouched
            hist = self.histories[image.id] if (image is not None and spec.kind in ("plane", "image")) else None
            snap = History.snapshot(image, spec.undo_label or spec.name, _param_summary(spec, params, scope, image.nt * image.nz * image.nc)) if hist is not None else None
            rev0 = image.rev if hist is not None else None
            name_before = image.name if image is not None else None
            changed = False
            try:
                try:
                    if spec.kind == "plane":
                        new = apply_plane_op(spec, ctx, params)
                        image.replace_data(new)
                        result = OpResult()
                        changed = True
                    elif spec.kind == "image":
                        result = spec.func(ctx, **params) or OpResult()
                        if image is not None:
                            changed = (result.data is not None or bool(result.close_image) or bool(result.rename)
                                       or result.calibration is not None or result.display is not None
                                       or result.overlay is not None
                                       or (rev0 is not None and image.rev != rev0))  # in-place edits (slice labels …)
                            if result.data is not None:
                                image.replace_data(result.data, rgb=result.rgb)
                    else:
                        result = spec.func(ctx, **params) or OpResult()
                except BaseException as exc:
                    if isinstance(exc, OperationCancelled):
                        status = "cancelled"
                    if snap is not None and (
                            image.rev != rev0 or image.name != snap.name or image.overlay != snap.overlay
                            or image.calibration.to_dict() != snap.calibration or image.display.to_dict() != snap.display
                            # in-place metadata edits (the stack editors rewrite the slice labels before the new data
                            # is installed) and the binary-mask flag are part of the snapshot too
                            or bool(image.lut_is_binary) != snap.lut_is_binary
                            or not _same_metadata(image.metadata, snap.metadata)):
                        # the op failed half-way (or was cancelled): roll back its in-place edits; no undo entry is left behind
                        History.restore(image, snap)
                    raise
                if hist is not None and changed:
                    hist.undo.append(snap)
                    hist.redo.clear()
                    hist._trim()
                payload = self._apply_result(spec, image, result, _param_summary(spec, params, scope, image.nt * image.nz * image.nc) if image is not None else "")
                dt = time.time() - t0
                payload["elapsed_ms"] = int(dt * 1000)
                if record:
                    self._record_op(spec, params, scope, roi, image, result, payload, name_before)
                self.add_log(f"{spec.name} ({dt * 1000:.0f} ms)")
                self._changed("images")
                status = "ok"
                return payload
            finally:
                if job is not None:
                    self._job_local.job = None
                    self._job_end(job, status)

    def _preview(self, spec: OpSpec, ctx: OpContext, params: dict[str, Any]) -> dict[str, Any]:
        image = ctx.img
        d = image.display
        if spec.kind == "plane":
            prev_scope = ctx.scope
            ctx.scope = "slice" if prev_scope in ("stack", "slice", "channel") else "plane"
            if ctx.scope == "slice" and prev_scope == "channel":
                ctx.scope = "plane"
            data = apply_plane_op(spec, ctx, params)
            planes = data[d.t, d.z]
        elif spec.kind == "image" and spec.preview:
            saved = image.data
            try:
                res = spec.func(ctx, **params) or OpResult()
            finally:
                image.data = saved
            if res.data is None:
                planes = image.data[d.t, d.z]
            else:
                dd = res.data
                tt = min(d.t, dd.shape[0] - 1)
                zz = min(d.z, dd.shape[1] - 1)
                planes = dd[tt, zz]
        else:
            raise ValueError("preview not supported")
        return {"preview": planes}

    def _apply_result(self, spec: OpSpec, image: LumenImage | None, res: OpResult, detail: str = "") -> dict[str, Any]:
        payload: dict[str, Any] = {"op": spec.id}
        if res.rename and image is not None:
            image.name = self.unique_name(res.rename)
        if res.calibration and image is not None:
            image.calibration = Calibration(**{**image.calibration.to_dict(), **res.calibration})
        if res.display and image is not None:
            self.update_display(image.id, res.display)
        if res.overlay is not None and image is not None:
            image.overlay = res.overlay
        for im in res.new_images:
            if image is not None and im is not image:
                # provenance (E7-17): where a derived image (Z Project, Duplicate, mask …) came from – scalar metadata,
                # so Image Info lists it and OME-TIFF export embeds it
                im.metadata["derived_from"] = f"{image.name} ▸ {spec.name}" + (f" ({detail})" if detail else "")  # with its parameters (E10-4)
                src = image.path or image.metadata.get("source_path")
                if src:
                    im.metadata["source_path"] = str(src)
            self.add_image(im, activate=True)
        payload["new_images"] = [im.id for im in res.new_images]
        if res.close_image and image is not None:
            self.close_image(image.id)
        if res.select_image:
            self.select(res.select_image)
        if res.results is not None:
            if res.results_replace:
                self.results.clear()
            self.results.append(res.results)
            payload["results_added"] = len(res.results)
            self._changed("results")
        if res.plot is not None:
            self.last_plot = res.plot
            payload["plot"] = res.plot
        if res.histogram is not None:
            payload["histogram"] = res.histogram
        if res.table is not None:
            self.tables[res.table.get("name", "Table")] = res.table
            payload["table"] = res.table
        if res.roi is not None:
            payload["roi"] = roimod.normalize(res.roi)
        if res.clear_roi:
            payload["clear_roi"] = True
        if res.rois_add:
            for r in res.rois_add:
                # a ROI may carry its own plane position (e.g. per-frame spots); otherwise use the current plane
                pos = r.get("position") or ({"t": image.display.t, "z": image.display.z, "c": image.display.c} if image else None)
                if r.get("id") and any(x.get("id") == r["id"] for x in self.roi_manager.rois):
                    # a selection picked from the manager (or an overlay item) still carries the manager's id:
                    # "Add" always creates a new independent entry, so drop it and let the manager mint a fresh one
                    r = {k: v for k, v in r.items() if k != "id"}
                self.roi_manager.add(r, image_id=r.get("image_id") or (image.id if image else None), position=pos)
            payload["rois_added"] = len(res.rois_add)
            self._changed("rois")
        if res.threshold is not None and image is not None:
            self.thresholds[image.id] = res.threshold
            payload["threshold"] = res.threshold
        if res.message:
            payload["message"] = res.message
        if res.log:
            self.add_log(res.log)
            payload["log"] = res.log
        return payload

    # ------------------------------------------------------------- state
    def state(self) -> dict[str, Any]:
        with self.lock:
            return {
                "version": self.version,
                "images": [im.summary() for im in self.images.values()],
                "active": self.active_id,
                "results": self.results.to_dict(),
                "rois": self.roi_manager.to_list(),
                "history": {iid: h.labels() for iid, h in self.histories.items()},
                "macro": self.macro[-200:],
                "recorder": {k: v for k, v in self.recorder().info().items() if k != "lines"},
                "log": self.log[-50:],
                "measurements": self.measurements,
                "measure_options": self.measure_options,
                "thresholds": self.thresholds,
                "last_plot": self.last_plot,
                "tables": self.tables,
                "recent": self.recent,
                "lut_names": __import__("lumen.core.luts", fromlist=["LUT_NAMES"]).LUT_NAMES,
                "lut_version": __import__("lumen.core.luts", fromlist=["lut_version"]).lut_version(),
                "memory_bytes": int(sum(im.nbytes for im in self.images.values()) + sum(sum(s.nbytes for s in h.undo) + sum(s.nbytes for s in h.redo) for h in self.histories.values())),
            }


def _param_summary(spec: OpSpec, params: dict[str, Any], scope: str, nplanes: int = 1) -> str:
    """"sigma=2.0, scope=stack (all planes)": every parameter of a run (defaults included – a row must not hide the sigma
    it ran with, E5-8 / E9) for the History panel, plus the scope of a plane command on a multi-plane image."""
    parts = [f"{k}={_py(v)}" for k, v in params.items() if v is not None and not (isinstance(v, str) and len(v) > 40)]
    if spec.kind == "plane" and nplanes > 1:
        parts.append("scope=slice (current plane)" if scope in ("slice", "plane") else f"scope={scope}" + (" (all planes)" if scope == "stack" else ""))
    elif spec.kind == "plane" and scope not in ("stack", "slice", "plane"):
        parts.append(f"scope={scope}")
    return ", ".join(parts)


def _display_step(d: Any, patch: dict[str, Any]) -> tuple[str, str, str]:
    """(label, detail, group) of the undo step a display patch makes – ("", "", "") for a plane change (t / z / c) or a
    patch that changes nothing. Channel edits are grouped per channel ("c2"), the composite mode as "mode"."""
    if "mode" in patch and patch["mode"] in ("composite", "single", "grayscale") and patch["mode"] != d.mode:
        return "Display mode", f"{d.mode} → {patch['mode']}", "mode"
    for i, chp in enumerate(patch.get("channels") or []):
        if i >= len(d.channels) or not chp:
            continue
        ch = d.channels[i]
        changed = {k: chp[k] for k in ("lut", "min", "max", "visible", "inverted", "gamma", "name") if k in chp and chp[k] is not None and getattr(ch, k) != chp[k]}
        if changed:
            what = ", ".join(f"{k}={_py(v)}" for k, v in changed.items())
            return "Display", f"channel {i + 1}" + (f" ({ch.name})" if getattr(ch, "name", "") else "") + f": {what}", f"c{i}"
    return "", "", ""


def _macro_line(spec: OpSpec, params: dict[str, Any], scope: str, roi: dict[str, Any] | None) -> str:
    # None-valued params mean "leave unset" (Edit ▸ Options dialogs): the replayed line keeps the same meaning without them
    args = ", ".join(f"{k}={_py(v)}" for k, v in params.items() if v is not None)
    extra = ""
    if spec.kind == "plane" and scope != "stack":
        extra = f", scope={_py(scope)}"
    if roi is not None and (spec.roi_aware or spec.requires_roi):
        r = {k: v for k, v in roi.items() if k not in ("id", "name", "image_id")}
        extra += f", roi={_py(r)}"
    sep = ", " if args and extra else ""
    return f'lumen.run("{spec.id}"{", " if args else ""}{args}{extra})'


def _py(v: Any) -> str:
    """Python literal for the macro recorder (round-trips through exec)."""
    if isinstance(v, float):
        return repr(round(v, 6))
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_py(k)}: {_py(x)}" for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_py(x) for x in v) + "]"
    return repr(v)


SESSION = Session()
