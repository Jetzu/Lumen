"""ROI Manager / Results clears as commands – recorded, counted and restorable (E7-6).

The inspector's "Clear" buttons used to empty the ROI Manager (41 ROIs) or the Results table (43 rows) through bare REST
calls: unrecorded and irreversible.  They now go through ``Session.clear_rois`` / ``delete_rois`` / ``clear_results``,
which keep what they removed, and these commands put it back (one step, like Fiji's confirmation offering to save).
The replayable lines are ``lumen.run("rois.clear")``, ``lumen.run("rois.delete", indices="3,5")`` and
``lumen.run("analyze.clear_results")`` (ImageJ's roiManager("reset") / roiManager("delete") / run("Clear Results")).
"""
from __future__ import annotations

from ..registry import Param, OpResult, op

CAT = "Analyze/Tools"
_NOIMG = dict(kind="analysis", requires_image=False, preview=False, roi_aware=False, scope_param=False)


def parse_indices(text: str, n: int) -> list[int]:
    """"3,5,8-10" → sorted 0-based indices within ``range(n)`` (ranges inclusive); negative = from the end."""
    out: set[int] = set()
    for part in str(text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part[1:]:
            a, b = part[1:].split("-", 1) if part.startswith("-") else part.split("-", 1)
            lo, hi = int(a) * (-1 if part.startswith("-") else 1), int(b)
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    idx = sorted(i if i >= 0 else n + i for i in out)
    bad = [i for i in idx if not 0 <= i < n]
    if bad:
        raise ValueError(f"ROI index {bad[0]} is out of range (the manager has {n} entries)")
    return idx


@op("rois.clear", "ROI Manager ▸ Delete All", CAT, **_NOIMG, keywords=["roi manager", "clear", "reset", "delete all"],
    description="Empty the ROI Manager (ImageJ's roiManager('reset')); one step restorable with rois.restore_cleared.")
def rois_clear(ctx):
    n = ctx.session.clear_rois(record=False)  # run_op records this command's own line
    msg = f"Cleared {n} ROI(s)" + (" – ROI Manager ▸ Restore Cleared puts them back" if n else "")
    return OpResult(message=msg, log=msg)


@op("rois.delete", "ROI Manager ▸ Delete", CAT, **_NOIMG, keywords=["roi manager", "delete", "remove"],
    params=[Param("indices", "string", "", label="Entries (0-based)", help="Comma-separated 0-based indices or ranges, e.g. 0,2,5-7; empty = none")],
    description="Delete the given ROI Manager entries (ImageJ's roiManager('select', i); roiManager('delete')); restorable with rois.restore_cleared.")
def rois_delete(ctx, indices=""):
    s = ctx.session
    rois = s.roi_manager.to_list()
    idx = parse_indices(indices, len(rois))
    if not idx:
        raise ValueError("No ROI Manager entries given (indices=\"0,2,5-7\")")
    n = s.delete_rois([rois[i]["id"] for i in idx], record=False)
    msg = f"Deleted {n} ROI(s), {len(s.roi_manager.to_list())} left"
    return OpResult(message=msg, log=msg)


@op("rois.restore_cleared", "ROI Manager ▸ Restore Cleared", CAT, **_NOIMG, keywords=["roi manager", "undo clear", "restore", "undelete"],
    description="Put back the ROIs the last ROI Manager Clear / Delete removed (one step).")
def rois_restore(ctx):
    out = ctx.session.restore_last_clear("rois")
    if not out["restored"]:
        raise ValueError("Nothing to restore: no ROI Manager clear or delete since the last restore")
    msg = f"Restored {out['restored']} ROI(s)"
    return OpResult(message=msg, log=msg)


@op("results.restore_cleared", "Results ▸ Restore Cleared", CAT, **_NOIMG, keywords=["results", "undo clear", "restore"],
    description="Put back the rows the last Clear Results removed (one step; they are prepended to any rows measured since).")
def results_restore(ctx):
    out = ctx.session.restore_last_clear("results")
    if not out["restored"]:
        raise ValueError("Nothing to restore: the Results table was not cleared since the last restore")
    msg = f"Restored {out['restored']} result row(s)"
    return OpResult(message=msg, log=msg)
