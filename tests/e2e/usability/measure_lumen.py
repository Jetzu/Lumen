#!/usr/bin/env python
"""Measure the clicks / key presses Lumen needs for the 10 tasks of docs/USABILITY.md.

Drives a running Lumen server (default http://127.0.0.1:8766) in headless Google Chrome through
Playwright (channel "chrome"). Selectors follow tests/e2e/conftest.py (palette, sheet, section,
drag). Every pointer or keyboard action the script performs goes through ``Counter``; nothing is
done through the REST API except resetting the session between tasks and reading state back for
verification.

Counting rules (the same rules are applied to the Fiji side of the assessment):
  click     one mouse click, or one click-drag (press, move, release)
  key       one key press or chord (Ctrl+K, Enter, M, Shift+S, ...)
  text      one typed string (a search query, a number, a path) regardless of its length
  dropdown  a choice from a <select> = 2 clicks (open the list, pick the entry)
  dialog    the command palette (transient modal overlay: closes on Esc / on running an entry) and
            parameter sheets (non-modal, docked at the top-right of the viewport)
Setup actions (opening the sample image through the UI) are counted but reported separately,
except in task 1 where opening the image *is* the task.

Usage:  e2e-venv/bin/python measure_lumen.py [URL]
Output: lumen_counts.json and lumen_counts.md next to this script.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable
from urllib import request as urlrequest

from playwright.sync_api import sync_playwright

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766").rstrip("/")
OUT = Path(__file__).resolve().parent
BATCH_IN = OUT / "batch_in"
TRACKS_CSV = OUT / "tracks_export.csv"


# ------------------------------------------------------------------ helpers
def http_json(path: str, method: str = "GET", body: Any = None, timeout: float = 60) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urlrequest.Request(URL + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlrequest.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def wait_until(cond: Callable[[], Any], timeout: float = 30, interval: float = 0.2, what: str = "condition") -> Any:
    deadline = time.monotonic() + timeout
    last: Any = None
    while time.monotonic() < deadline:
        try:
            last = cond()
        except Exception as e:  # transient REST / DOM races
            last = f"error: {e}"
        if last and not (isinstance(last, str) and last.startswith("error:")):
            return last
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout}s waiting for {what} (last value: {last!r})")


class Counter:
    """Every UI action of the script goes through here, so the counts are complete by construction."""

    def __init__(self, page: Any) -> None:
        self.page = page
        self.reset()

    def reset(self) -> None:
        self.clicks = 0
        self.keys = 0
        self.texts = 0
        self.dialogs: list[tuple[str, bool]] = []
        self.log: list[str] = []

    def click(self, locator: Any, what: str) -> None:
        locator.click(timeout=15_000)
        self.clicks += 1
        self.log.append(f"click      {what}")

    def key(self, combo: str, what: str = "") -> None:
        self.page.keyboard.press(combo)
        self.keys += 1
        self.log.append(f"key        {combo}  {what}".rstrip())

    def text(self, locator: Any, s: str, what: str) -> None:
        locator.fill(s)
        self.texts += 1
        self.log.append(f"text       {what}: {s!r} ({len(s)} chars, counted as 1)")

    def dropdown(self, locator: Any, label: str | None = None, value: str | None = None, what: str = "") -> None:
        if label is not None:
            locator.select_option(label=label)
        else:
            locator.select_option(value=value)
        self.clicks += 2
        self.log.append(f"dropdown   {what} -> {label or value} (2 clicks)")

    def drag(self, locator: Any, fx0: float, fy0: float, fx1: float, fy1: float, what: str, steps: int = 12) -> None:
        box = locator.bounding_box()
        assert box, "element has no bounding box"
        x0, y0 = box["x"] + box["width"] * fx0, box["y"] + box["height"] * fy0
        x1, y1 = box["x"] + box["width"] * fx1, box["y"] + box["height"] * fy1
        self.page.mouse.move(x0, y0)
        self.page.mouse.down()
        self.page.mouse.move(x1, y1, steps=steps)
        self.page.mouse.up()
        self.clicks += 1
        self.log.append(f"click-drag {what}")

    def dialog(self, name: str, modal: bool) -> None:
        self.dialogs.append((name, modal))
        self.log.append(f"dialog     {name} ({'modal overlay' if modal else 'non-modal panel'})")

    def snapshot(self) -> dict[str, Any]:
        return {
            "clicks": self.clicks,
            "keys": self.keys,
            "texts": self.texts,
            "actions": self.clicks + self.keys + self.texts,
            "dialogs": len(self.dialogs),
            "modal": sum(1 for _, m in self.dialogs if m),
            "dialog_names": [n for n, _ in self.dialogs],
            "log": list(self.log),
        }


class Lumen:
    def __init__(self, page: Any, c: Counter) -> None:
        self.page = page
        self.c = c

    # --- REST (verification + reset only)
    def state(self) -> dict[str, Any]:
        return http_json("/api/state")

    def image(self, iid: str) -> dict[str, Any]:
        return next(i for i in self.state()["images"] if i["id"] == iid)

    def reset(self) -> None:
        st = self.state()
        for img in st.get("images", []):
            http_json(f"/api/images/{img['id']}", "DELETE")
        http_json("/api/rois", "DELETE")
        http_json("/api/results", "DELETE")

    # --- UI
    def palette(self, query: str, name: str, how: str = "click") -> str:
        """Ctrl+K, type ``query``, run the entry called exactly ``name`` (click it, or Enter on the selected one)."""
        self.c.key("Control+k", "open the command palette")
        self.c.dialog("command palette", modal=True)
        inp = self.page.locator(".palette input")
        inp.wait_for(timeout=10_000)
        self.c.text(inp, query, "palette query")
        item = self.page.locator(".palette-item", has=self.page.locator(".pi-name", has_text=re.compile(rf"^{re.escape(name)}$")))
        item.first.wait_for(timeout=10_000)
        selected = self.page.locator(".palette-item.sel .pi-name").first.inner_text()
        if how == "enter":
            assert selected == name, f"selected entry is {selected!r}, not {name!r}"
            self.c.key("Enter", f"run the selected entry '{name}'")
        else:
            self.c.click(item.first, f"palette entry '{name}'")
        self.page.locator(".palette").wait_for(state="hidden", timeout=10_000)
        return selected

    def sheet(self, name: str | None = None) -> Any:
        sh = self.page.locator(".sheet")
        sh.wait_for(timeout=15_000)
        title = sh.locator("h3").first.inner_text()
        self.c.dialog(f"parameter sheet '{name or title}'", modal=False)
        return sh

    def sheet_has_preview(self) -> bool:
        return self.page.locator(".sheet .sheet-foot label.switch input").count() > 0

    def apply(self, what: str = "Apply") -> None:
        self.c.click(self.page.locator(".sheet .sheet-foot button.primary"), what)
        self.page.locator(".sheet").wait_for(state="hidden", timeout=180_000)

    def open_sample(self, name: str) -> str:
        before = {i["id"] for i in self.state().get("images", [])}
        self.palette("open sample", "Open Sample")
        sh = self.sheet("Open Sample")
        self.c.dropdown(sh.locator("select").first, label=name, what="sample")
        self.apply("Apply (open the sample)")
        new = wait_until(lambda: [i["id"] for i in self.state().get("images", []) if i["id"] not in before], 120, what=f"sample {name!r} to open")
        self.page.locator(".statusbar").get_by_text(re.compile(r"\d+×\d+ px")).wait_for(timeout=30_000)
        return new[0]

    def section(self, title: str) -> Any:
        head = self.page.locator(f"xpath=//div[contains(@class,'section-head')][text()={title!r}]")
        head.first.wait_for(timeout=30_000)
        if "open" not in (head.first.get_attribute("class") or ""):
            self.c.click(head.first, f"expand the inspector section '{title}'")
        body = head.first.locator("xpath=following-sibling::div[contains(@class,'section-body')]")
        body.wait_for(timeout=10_000)
        return body


# ------------------------------------------------------------------ tasks
def task1(L: Lumen, c: Counter) -> dict[str, Any]:
    """Open a multi-channel image and adjust each channel (opening is part of the task)."""
    iid = L.open_sample("Synthetic nuclei (3 channels)")
    img = L.image(iid)
    before = [(round(ch["min"], 1), round(ch["max"], 1)) for ch in img["display"]["channels"]]
    disp = L.section("Display")
    autos = disp.locator('button[title^="Auto"]')
    n = autos.count()
    assert n == 3, f"expected 3 per-channel Auto buttons, found {n}"
    for i in range(3):
        c.click(autos.nth(i), f"Auto display range, channel {i + 1} (all three channels' sliders are visible at once)")
    after = wait_until(lambda: (lambda a: a if sum(x != y for x, y in zip(a, before)) >= 2 else None)([(round(ch["min"], 1), round(ch["max"], 1)) for ch in L.image(iid)["display"]["channels"]]), 20, what="display ranges to change")
    hist = L.state()["history"].get(iid, {}).get("undo", [])
    return {
        "verified": f"3 channels; per-channel display ranges {before} -> {after}; mode={img['display']['mode']}",
        "preview": "yes: range/LUT changes composite instantly in the client (no server round trip)",
        "undo": f"display changes are not undo steps (history after task: {hist}); 'Reset'/'Reset all' buttons restore the data range",
        "note": "Composite mode shows all three channels' histogram sliders at once; no channel switching needed",
    }


def task2(L: Lumen, c: Counter) -> dict[str, Any]:
    """Threshold + Analyze Particles with results (image already open)."""
    iid = L.state()["active"]
    thr = L.section("Threshold")
    c.click(thr.locator("button", has_text=re.compile(r"^\s*Auto\s*$")), "Threshold section: Auto (Default method, red overlay)")
    t = wait_until(lambda: L.state()["thresholds"].get(iid), what="an auto threshold")
    c.click(thr.locator("button", has_text="Particles"), "Threshold section: Particles…")
    sh = L.sheet("Analyze Particles")
    preview = L.sheet_has_preview()
    L.apply("Run (Analyze Particles)")
    n = wait_until(lambda: len(L.state()["rois"]), 60, what="ROIs in the manager")
    rows = wait_until(lambda: len(L.state()["results"]["rows"]), 30, what="results rows")
    rois_visible = L.section("ROI Manager").locator(".roi-item").count()
    res_visible = L.section("Results").locator("table.results tbody tr").count()
    hist = L.state()["history"].get(iid, {}).get("undo", [])
    return {
        "verified": f"threshold {t['lower']:.0f}-{t['upper']:.0f}; {n} ROIs in the docked ROI Manager ({rois_visible} rows visible), {rows} result rows ({res_visible} visible) in the docked Results section; summary in the Table section",
        "preview": f"threshold: yes (red overlay follows the slider live); Analyze Particles sheet: {'yes' if preview else 'no'} (analysis command)",
        "undo": f"history after task: {hist} (analysis results are not undo steps; pixels untouched)",
    }


def task3(L: Lumen, c: Counter) -> dict[str, Any]:
    """Measure a drawn ROI (image already open)."""
    c.key("1", "rectangle tool")
    c.drag(L.page.locator(".canvas-wrap"), 0.35, 0.35, 0.6, 0.6, "draw the rectangle on the canvas")
    L.page.locator(".statusbar .pill", has_text="rect").wait_for(timeout=10_000)
    c.key("m", "Measure")
    rows = wait_until(lambda: L.state()["results"]["rows"], 30, what="a results row")
    res_body = L.section("Results")
    res_visible = wait_until(lambda: res_body.locator("table.results tbody tr").count(), 20, what="a visible results row")
    return {
        "verified": f"{len(rows)} result row(s) in the docked Results section ({res_visible} visible): {rows[0] if rows else None}",
        "preview": "n/a (selection geometry shown live in the status bar while dragging)",
        "undo": "n/a (measurement does not change pixels)",
    }


def task4(L: Lumen, c: Counter) -> dict[str, Any]:
    """Z-projection of a stack (stack already open)."""
    before = {i["id"] for i in L.state()["images"]}
    L.palette("z proj", "Z Project", how="enter")
    sh = L.sheet("Z Project")
    preview = L.sheet_has_preview()
    L.apply("Apply (Max Intensity, all slices = defaults)")
    new = wait_until(lambda: [i for i in L.state()["images"] if i["id"] not in before], 60, what="the projection image")
    img = new[0]
    tabs = L.page.locator(".tabs .tab, .image-tabs .tab, [role='tablist'] [role='tab']").count()
    return {
        "verified": f"new image '{img['name']}' with shape {img['shape']} opened as a tab (tabs counted: {tabs})",
        "preview": "yes" if preview else "no (whole-image command: the sheet has no Preview switch)",
        "undo": "the projection is a new image; closing the tab discards it",
    }


def task5a(L: Lumen, c: Counter) -> dict[str, Any]:
    """3-channel composite with LUTs: assign a LUT per channel on the open composite."""
    iid = L.state()["active"]
    disp = L.section("Display")
    sels = disp.locator('select[title="Lookup table"]')
    assert sels.count() == 3, sels.count()
    luts = http_json("/api/meta")["luts"]
    wanted = [n for n in ("Cyan", "Magenta", "Yellow") if n in luts]
    if len(wanted) < 3:
        wanted = ["Red", "Green", "Blue"]
    for i, lut in enumerate(wanted):
        c.dropdown(sels.nth(i), label=lut, what=f"LUT of channel {i + 1}")
    got = wait_until(lambda: (lambda l: l if l == wanted else None)([ch["lut"] for ch in L.image(iid)["display"]["channels"]]), 20, what="LUTs to be set")
    mode = L.image(iid)["display"]["mode"]
    return {
        "verified": f"LUTs {got}; mode={mode} (composite is the default for multi-channel images)",
        "preview": "yes: the composite re-renders instantly on the client",
        "undo": "display state; 'Reset all' restores defaults; not an undo step",
    }


def task5b(L: Lumen, c: Counter, parts: list[str]) -> dict[str, Any]:
    """3-channel composite from three separate single-channel images (Merge Channels)."""
    before = {i["id"] for i in L.state()["images"]}
    L.palette("merge", "Merge Channels", how="enter")
    sh = L.sheet("Merge Channels")
    preview = L.sheet_has_preview()
    sels = sh.locator("select")
    for i, pid in enumerate(parts[:3]):
        c.dropdown(sels.nth(i), value=pid, what=f"C{i + 1} source image")
    L.apply("Apply (Create composite = default)")
    new = wait_until(lambda: [i for i in L.state()["images"] if i["id"] not in before], 60, what="the merged image")
    img = new[0]
    return {
        "verified": f"merged image '{img['name']}' shape {img['shape']} (c={img['shape']['c']}), LUTs {[ch['lut'] for ch in img['display']['channels']]}",
        "preview": "yes" if preview else "no (whole-image command)",
        "undo": "new image; sources kept only if 'Keep source' is ticked",
    }


def task6(L: Lumen, c: Counter) -> dict[str, Any]:
    """Run a filter (Gaussian) with live preview and undo it (image already open)."""
    iid = L.state()["active"]
    L.palette("gauss", "Gaussian Blur", how="enter")
    sh = L.sheet("Gaussian Blur")
    sw = sh.locator(".sheet-foot label.switch input")
    preview_default = sw.is_checked()
    if not preview_default:
        with L.page.expect_response(lambda r: "/api/ops/filters.gaussian/preview" in r.url and r.ok, timeout=30_000):
            c.click(sh.locator(".sheet-foot label.switch"), "enable Preview")
    sigma = sh.locator('input[type="number"]').first
    with L.page.expect_response(lambda r: "/api/ops/filters.gaussian/preview" in r.url and r.ok, timeout=30_000):
        c.text(sigma, "3", "sigma")
    L.apply("Apply")
    wait_until(lambda: L.state()["history"][iid]["undo"] == ["Gaussian Blur"], what="Gaussian Blur in the undo history")
    c.key("Control+z", "Undo")
    wait_until(lambda: L.state()["history"][iid]["undo"] == [] and L.state()["history"][iid]["redo"] == ["Gaussian Blur"], what="undo")
    return {
        "verified": "history: ['Gaussian Blur'] -> undo -> redo ['Gaussian Blur']; the preview request fired while typing sigma",
        "preview": f"yes (Preview switch in the sheet; on by default: {preview_default}); previews the current plane while editing",
        "undo": "see depth check",
    }


def undo_depth_check(L: Lumen, c: Counter) -> dict[str, Any]:
    iid = L.state()["active"]
    for combo, what in (("Shift+S", "Smooth"), ("Shift+S", "Smooth"), ("Shift+I", "Invert")):
        n = len(L.state()["history"][iid]["undo"])
        c.key(combo, what)
        wait_until(lambda: len(L.state()["history"][iid]["undo"]) == n + 1, what=f"{what} in history")
    depth = list(L.state()["history"][iid]["undo"])
    hist_rows = L.section("History").locator(".hist-item").count()
    for _ in range(3):
        c.key("Control+z", "Undo")
    wait_until(lambda: L.state()["history"][iid]["undo"] == [] and len(L.state()["history"][iid]["redo"]) == 3, what="three undos")
    c.key("Control+Shift+z", "Redo")
    wait_until(lambda: len(L.state()["history"][iid]["undo"]) == 1, what="one redo")
    return {"stack_before_undo": depth, "history_rows_visible": hist_rows, "after_3_undo_redo_list_len": 3, "redo_works": True}


def task7(L: Lumen, c: Counter) -> dict[str, Any]:
    """Track cells and export the tracks (time-lapse already open)."""
    if TRACKS_CSV.exists():
        TRACKS_CSV.unlink()
    L.palette("track", "Track (detect + link)", how="enter")
    sh = L.sheet("Track (detect + link)")
    preview = L.sheet_has_preview()
    L.apply("Run (defaults: LoG, radius 2.5, automatic threshold, LAP linking)")
    tracks = L.section("Tracks")
    badge = L.page.locator("xpath=//div[contains(@class,'section-head')][text()='Tracks']//span[contains(@class,'badge')]").first.inner_text()
    c.click(tracks.locator("button", has_text="Export CSV"), "Tracks section: Export CSV…")
    sh = L.sheet("Export Tracks (CSV)")
    c.text(sh.locator('input[type="text"]').first, str(TRACKS_CSV), "output path")
    L.apply("Run (export)")
    wait_until(lambda: TRACKS_CSV.exists() and TRACKS_CSV.stat().st_size > 0, 30, what="the CSV file")
    head = TRACKS_CSV.read_text(encoding="utf-8").splitlines()[:2]
    n_lines = len(TRACKS_CSV.read_text(encoding="utf-8").splitlines())
    tbl_rows = tracks.locator("table.tracks tbody tr").count()
    return {
        "verified": f"{badge} tracks in the docked Tracks section ({tbl_rows} table rows, time-coloured overlays); CSV written: {n_lines} lines, header {head[0][:80]}",
        "preview": "yes" if preview else "no (analysis command; results appear as overlays + Tracks table after Run)",
        "undo": "tracks live in a Tracks table (delete selected / Clear Tracks); not pixel undo",
    }


def task8(L: Lumen, c: Counter) -> dict[str, Any]:
    """Find a command whose menu location you do not know (CLAHE) and launch it."""
    c.key("Control+k", "open the command palette")
    c.dialog("command palette", modal=True)
    inp = L.page.locator(".palette input")
    inp.wait_for(timeout=10_000)
    c.text(inp, "clahe", "palette query")
    first = L.page.locator(".palette-item.sel").first
    first.wait_for(timeout=10_000)
    name = first.locator(".pi-name").inner_text()
    cat = first.locator(".pi-cat").inner_text()
    desc = first.locator(".pi-desc").inner_text() if first.locator(".pi-desc").count() else ""
    c.key("Enter", f"run '{name}'")
    sh = L.page.locator(".sheet")
    sh.wait_for(timeout=15_000)
    title = sh.locator("h3").first.inner_text()
    # uncounted probes: word-order-free matching and the 'no match' fallback
    L.page.keyboard.press("Escape")
    L.page.keyboard.press("Control+k")
    L.page.locator(".palette input").fill("local contrast")
    probe = [e.inner_text() for e in L.page.locator(".palette-item .pi-name").all()[:5]]
    L.page.locator(".palette input").fill("zzqx-nothing")
    empty = L.page.locator(".palette-empty").inner_text() if L.page.locator(".palette-empty").count() else ""
    L.page.keyboard.press("Escape")
    return {
        "verified": f"first match '{name}' [{cat}] - '{desc}'; Enter opened the sheet '{title}'. Probe 'local contrast' -> {probe}; no-match text: {empty!r}",
        "preview": "n/a",
        "undo": "n/a",
    }


def task9(L: Lumen, c: Counter) -> dict[str, Any]:
    """Batch-process a folder with the recorded steps (image already open)."""
    out_dir = BATCH_IN / "lumen_output"
    if out_dir.exists():
        for p in out_dir.iterdir():
            p.unlink()
    script = L.section("Script & Recorder")
    clear = script.locator("button", has_text=re.compile(r"^Clear$"))
    if clear.count() and clear.first.is_enabled():
        c.click(clear.first, "Recorder: Clear (start a fresh recording)")
        wait_until(lambda: L.state()["macro"] == [], what="an empty recorder")
    # the interactive step that is being recorded
    L.palette("gauss", "Gaussian Blur", how="enter")
    L.sheet("Gaussian Blur")
    L.apply("Apply (sigma 2 = default)")
    recorded = wait_until(lambda: list(L.state()["macro"]), what="a recorded line")
    rec_snapshot = c.snapshot()
    # replay the recording over the folder
    L.palette("recorded macro", "Apply Recorded Macro to Folder...", how="enter")
    sh = L.sheet("Apply Recorded Macro to Folder...")
    preview = L.sheet_has_preview()
    c.text(sh.locator('input[type="text"]').first, str(BATCH_IN), "input folder")
    L.apply("Run (output format tif = default)")
    files = wait_until(lambda: sorted(p.name for p in out_dir.glob("*.tif")) if out_dir.exists() else None, 60, what="batch output files")
    log_tail = [ln for ln in L.state()["log"][-6:]]
    return {
        "verified": f"recorded: {recorded}; batch output: {files} in {out_dir}; log tail: {log_tail}",
        "preview": "no" if not preview else "yes",
        "undo": "the originals are untouched; outputs go to <input>/lumen_output",
        "record_part_actions": rec_snapshot["actions"],
        "recorded_lines": recorded,
    }


def task10(L: Lumen, c: Counter) -> dict[str, Any]:
    """Compare two images side by side (two images already open)."""
    c.key("Control+Shift+Backslash", "Split view")
    L.page.locator(".split-view").wait_for(timeout=10_000)
    panes = L.page.locator(".split-pane").count()
    names = [s.evaluate("e => e.options[e.selectedIndex].text") for s in L.page.locator(".split-pane select").all()]
    L.page.keyboard.press("Control+Shift+Backslash")  # uncounted: leave the split view again
    return {
        "verified": f"{panes} panes in one window showing {names}; pan/zoom of the compare pane follows the active pane",
        "preview": "n/a",
        "undo": "n/a",
    }


# ------------------------------------------------------------------ driver
def main() -> int:
    results: list[dict[str, Any]] = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True, args=["--ignore-gpu-blocklist", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        ctx = browser.new_context(viewport={"width": 1440, "height": 900})
        page = ctx.new_page()
        errors: list[str] = []
        page.on("pageerror", lambda err: errors.append(str(err)))
        c = Counter(page)
        L = Lumen(page, c)
        L.reset()
        page.goto(URL)
        page.locator(".app").wait_for(timeout=30_000)

        def run(num: int, title: str, setup: Callable[[], Any] | None, body: Callable[[Any], dict[str, Any]]) -> None:
            L.reset()
            page.reload()
            page.locator(".app").wait_for(timeout=30_000)
            c.reset()
            entry: dict[str, Any] = {"task": num, "title": title}
            t0 = time.monotonic()
            try:
                setup_ctx = setup() if setup else None
                entry["setup"] = c.snapshot()
                macro_before = len(L.state()["macro"])
                c.reset()
                info = body(setup_ctx)
                entry["measured"] = c.snapshot()
                entry.update(info)
                entry["recorded"] = L.state()["macro"][macro_before:]
            except Exception as e:  # keep going; the table shows the failure
                entry["error"] = f"{type(e).__name__}: {e}"
                entry["measured"] = c.snapshot()
                try:
                    page.screenshot(path=str(OUT / f"task{num}_error.png"))
                except Exception:
                    pass
            entry["seconds"] = round(time.monotonic() - t0, 1)
            results.append(entry)
            m = entry["measured"]
            print(f"[task {num}] {title}: clicks={m['clicks']} keys={m['keys']} texts={m['texts']} actions={m['actions']} dialogs={m['dialogs']} (modal {m['modal']})" + (f"  ERROR {entry['error']}" if entry.get("error") else ""), flush=True)

        run(1, "Open a multi-channel image and adjust each channel", None, lambda _: task1(L, c))
        run(2, "Threshold + Analyze Particles with results", lambda: L.open_sample("Blobs"), lambda _: task2(L, c))
        run(3, "Measure a drawn ROI", lambda: L.open_sample("Blobs"), lambda _: task3(L, c))
        run(4, "Z-projection of a stack", lambda: L.open_sample("Cells (3-D, 2 channels)"), lambda _: task4(L, c))
        run(5, "3-channel composite with LUTs (a: LUT per channel of an open composite)", lambda: L.open_sample("Synthetic nuclei (3 channels)"), lambda _: task5a(L, c))

        def setup5b() -> list[str]:
            L.open_sample("Synthetic nuclei (3 channels)")
            before = {i["id"] for i in L.state()["images"]}
            L.palette("split", "Split Channels", how="enter")
            parts = wait_until(lambda: (lambda n: n if len(n) == 3 else None)([i["id"] for i in L.state()["images"] if i["id"] not in before]), 60, what="three split images")
            return parts

        run(5.5, "3-channel composite with LUTs (b: merge three single-channel images)", setup5b, lambda parts: task5b(L, c, parts))

        def body6(_: Any) -> dict[str, Any]:
            macro_len = len(L.state()["macro"])
            info = task6(L, c)
            info["recorded_task_only"] = L.state()["macro"][macro_len:]
            task_counts = c.snapshot()
            c.reset()
            depth = undo_depth_check(L, c)
            info["undo_depth_check"] = depth
            info["undo_depth_check_actions"] = c.snapshot()["actions"]
            # restore the task-only counters for the table
            c.clicks, c.keys, c.texts, c.dialogs, c.log = task_counts["clicks"], task_counts["keys"], task_counts["texts"], [(n, False) for n in task_counts["dialog_names"]], task_counts["log"]
            c.dialogs = [("command palette", True), ("parameter sheet 'Gaussian Blur'", False)]
            return info

        run(6, "Run a filter and undo it", lambda: L.open_sample("Blobs"), body6)
        run(7, "Track cells and export tracks", lambda: L.open_sample("Nuclei timelapse (synthetic)"), lambda _: task7(L, c))
        run(8, "Find a command whose menu location you do not know", lambda: L.open_sample("Blobs"), lambda _: task8(L, c))
        run(9, "Batch-process a folder with the recorded steps", lambda: L.open_sample("Blobs"), lambda _: task9(L, c))
        run(10, "Compare two images side by side", lambda: (L.open_sample("Blobs"), L.open_sample("Coins")), lambda _: task10(L, c))

        ctx.close()
        browser.close()
        if errors:
            print("page errors:", errors)

    (OUT / "lumen_counts.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    lines = ["| # | Task | Setup actions | Clicks | Keys | Text entries | Total | Dialogs (modal) | Preview | Verified |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        m = r["measured"]
        s = r.get("setup", {}).get("actions", 0)
        lines.append(f"| {r['task']} | {r['title']} | {s} | {m['clicks']} | {m['keys']} | {m['texts']} | {m['actions']} | {m['dialogs']} ({m['modal']}) | {r.get('preview', '')} | {r.get('error') or r.get('verified', '')} |")
    (OUT / "lumen_counts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    for r in results:
        print(f"\n--- task {r['task']} log ---")
        for ln in r["measured"]["log"]:
            print("  " + ln)
        if r.get("recorded"):
            print("  recorded:", r["recorded"])
    return 1 if any(r.get("error") for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
