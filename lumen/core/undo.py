"""Unlimited (memory-bounded) undo/redo per image."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .image import LumenImage, Calibration


@dataclass
class Snapshot:
    label: str
    data: np.ndarray
    rgb: bool
    calibration: dict[str, Any]
    display: dict[str, Any]
    overlay: list[dict[str, Any]]
    name: str
    #: deep copy of ``img.metadata`` – stack editors (Delete Slice, Add Slice, Reverse, Re-order, Make Substack, Reduce …)
    #: rewrite the per-plane slice-label list together with the pixels, so an undo has to bring the matching list back
    metadata: dict[str, Any] = field(default_factory=dict)
    #: the binary-mask flag (Apply Threshold / Make Binary set it; undoing them must clear it again)
    lut_is_binary: bool = False
    #: the parameters of the command this snapshot precedes ("sigma=5, scope=slice"): two "Gaussian Blur" entries
    #: in the History panel are told apart by it (provenance, E7-16)
    detail: str = ""
    #: a display-only step (E5-8): ``data`` is an empty placeholder and ``restore`` touches only ``img.display``, so
    #: changing a channel's range, LUT, visibility or the composite mode is undoable like a pixel edit
    display_only: bool = False
    #: groups consecutive display edits of one control ("c2" for channel 2, "mode"): a slider drag becomes one step
    group: str = ""

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)  # metadata / display / overlay are tiny next to the pixels: only those count against the budget


class History:
    #: Memory budget (bytes) given to every *new* history.  Edit ▸ Options ▸ Memory & Threads
    #: (``edit.options_memory``, ``session.settings["undo_budget_mb"]``) changes it and calls
    #: :meth:`set_budget` on the histories that already exist.
    default_budget_bytes: int = 1_500_000_000

    def __init__(self, budget_bytes: int | None = None) -> None:
        self.undo: list[Snapshot] = []
        self.redo: list[Snapshot] = []
        self.budget = int(budget_bytes if budget_bytes is not None else History.default_budget_bytes)

    @classmethod
    def set_default_budget(cls, budget_bytes: int) -> None:
        """Budget for histories created from now on (``session.add_image`` creates one per image)."""
        cls.default_budget_bytes = max(0, int(budget_bytes))

    def set_budget(self, budget_bytes: int) -> None:
        """Change this history's budget and drop the oldest snapshots that no longer fit."""
        self.budget = max(0, int(budget_bytes))
        self._trim()

    @staticmethod
    def snapshot(img: LumenImage, label: str, detail: str = "") -> Snapshot:
        data = img.data
        try:
            from .io import is_virtual, read_image
            if is_virtual(data) and hasattr(data, "fresh"):
                # a virtual (lazy) stack has never been written: snapshot an independent lazy handle over the same
                # source instead of reading the whole stack into memory (never share the image's own lazy object –
                # its copy-on-write writes would modify the snapshot too)
                data = data.fresh()
            elif is_virtual(data) and img.path:
                src = img.metadata.get("source") or {}
                data = read_image(img.path, series=src.get("series"), level=int(src.get("level", 0) or 0), virtual=True).data
            else:
                data = data.copy()
        except Exception:  # noqa: BLE001 – a failed lazy re-open falls back to the ordinary in-memory copy
            data = img.data.copy()
        return Snapshot(label=label, data=data, rgb=img.rgb, calibration=img.calibration.to_dict(),
                        display=img.display.to_dict(), overlay=[dict(o) for o in img.overlay], name=img.name,
                        metadata=copy.deepcopy(img.metadata), lut_is_binary=bool(img.lut_is_binary), detail=detail)

    @staticmethod
    def snapshot_display(img: LumenImage, label: str, detail: str = "", group: str = "") -> Snapshot:
        """A display-only snapshot (no pixel copy): undoing it restores ``img.display`` and nothing else."""
        return Snapshot(label=label, data=np.empty(0, dtype=np.uint8), rgb=img.rgb, calibration=img.calibration.to_dict(),
                        display=img.display.to_dict(), overlay=[], name=img.name, metadata={}, lut_is_binary=bool(img.lut_is_binary),
                        detail=detail, display_only=True, group=group)

    def push_display(self, img: LumenImage, label: str, detail: str = "", group: str = "") -> bool:
        """Record the display state *before* a display edit. Consecutive edits of the same ``group`` (a slider drag, several
        patches of one channel) merge into the first step; returns True when a new step was added."""
        if self.undo and self.undo[-1].display_only and group and self.undo[-1].group == group:
            self.undo[-1].detail = detail or self.undo[-1].detail
            self.redo.clear()
            return False
        self.undo.append(self.snapshot_display(img, label, detail, group))
        self.redo.clear()
        self._trim()
        return True

    @staticmethod
    def restore(img: LumenImage, snap: Snapshot) -> None:
        from .image import ChannelDisplay, Display
        if snap.display_only:
            d = snap.display
            img.display = Display(channels=[ChannelDisplay(**c) for c in d["channels"]], mode=d["mode"], t=d["t"], z=d["z"], c=d["c"])
            img.clamp_position()
            return
        try:
            from .io import is_virtual
            lazy = is_virtual(snap.data)
        except Exception:  # noqa: BLE001
            lazy = False
        img.data = snap.data if lazy else snap.data.copy()  # a lazy snapshot is installed as is (it is never written to)
        img.rev += 1
        img.rgb = snap.rgb
        img.calibration = Calibration(**snap.calibration)
        d = snap.display
        img.display = Display(channels=[ChannelDisplay(**c) for c in d["channels"]], mode=d["mode"], t=d["t"], z=d["z"], c=d["c"])
        img.overlay = [dict(o) for o in snap.overlay]
        img.name = snap.name
        img.metadata = copy.deepcopy(snap.metadata)
        img.lut_is_binary = bool(snap.lut_is_binary)
        img.clamp_position()

    def push(self, img: LumenImage, label: str) -> None:
        self.undo.append(self.snapshot(img, label))
        self.redo.clear()
        self._trim()

    def _trim(self) -> None:
        total = sum(s.nbytes for s in self.undo)
        while len(self.undo) > 1 and total > self.budget:
            total -= self.undo.pop(0).nbytes

    def can_undo(self) -> bool:
        return bool(self.undo)

    def can_redo(self) -> bool:
        return bool(self.redo)

    def do_undo(self, img: LumenImage) -> str | None:
        if not self.undo:
            return None
        snap = self.undo.pop()
        self.redo.append(self.snapshot_display(img, snap.label, snap.detail, snap.group) if snap.display_only else self.snapshot(img, snap.label, snap.detail))
        self.restore(img, snap)
        return snap.label

    def do_redo(self, img: LumenImage) -> str | None:
        if not self.redo:
            return None
        snap = self.redo.pop()
        self.undo.append(self.snapshot_display(img, snap.label, snap.detail, snap.group) if snap.display_only else self.snapshot(img, snap.label, snap.detail))
        self.restore(img, snap)
        return snap.label

    def labels(self) -> dict[str, list[str]]:
        """``undo`` / ``redo``: the step labels (unchanged); ``undo_detail`` / ``redo_detail``: the parameters of each step
        ("sigma=5, scope=slice", "" when the command has none) for the History panel's tooltips."""
        return {"undo": [s.label for s in self.undo], "redo": [s.label for s in self.redo],
                "undo_detail": [s.detail for s in self.undo], "redo_detail": [s.detail for s in self.redo]}
