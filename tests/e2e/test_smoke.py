"""Smoke tests of the Lumen UI in headless Google Chrome.

Skipped as a whole when Playwright is not importable; individual tests skip when the served build
(``lumen/static``) predates the feature they exercise. Each test ends by asserting that the browser
console stayed free of errors (GL/WebGL driver chatter is ignored, see ``conftest.IGNORED_CONSOLE``).
"""
from __future__ import annotations

import re

import pytest

pytest.importorskip("playwright.sync_api", reason="playwright is not installed (pip install -e '.[e2e]')")

from playwright.sync_api import Page, expect  # noqa: E402

from .conftest import Console, LumenApp, wait_until  # noqa: E402

SAMPLE_3CH = "Synthetic nuclei (3 channels)"
SAMPLE_3D = "Cells (3-D, 2 channels)"


def no_console_errors(console: Console) -> None:
    assert console.unexpected() == [], "unexpected browser console errors:\n" + "\n".join(console.unexpected())


# ------------------------------------------------------------------ loading
def test_app_loads_without_console_errors(page: Page, app: LumenApp, console: Console) -> None:
    expect(page).to_have_title("Lumen")
    expect(page.locator(".topbar")).to_be_visible()
    expect(page.locator(".rail")).to_be_visible()
    expect(page.locator(".statusbar")).to_contain_text("Open an image to begin")
    meta = app.meta()
    assert len(meta["ops"]) > 100 and "Fire" in meta["luts"]
    no_console_errors(console)


def test_open_three_channel_sample(page: Page, app: LumenApp, console: Console) -> None:
    iid = app.open_sample(SAMPLE_3CH)
    img = next(i for i in app.state()["images"] if i["id"] == iid)
    assert img["shape"]["c"] == 3 and img["shape"]["x"] == 512
    expect(page.locator(".statusbar")).to_contain_text("512×512 px")
    # the three channel chips of the dimension bar
    expect(page.locator(".dimbar .chip")).to_have_count(3)
    no_console_errors(console)


# ------------------------------------------------------------------ filters, preview, undo
def test_gaussian_preview_apply_undo(page: Page, app: LumenApp, console: Console) -> None:
    iid = app.open_sample(SAMPLE_3CH)
    app.palette("gaussian", "Gaussian Blur")
    sheet = app.sheet()
    expect(sheet.locator("h3")).to_have_text("Gaussian Blur")
    preview = sheet.locator(".sheet-foot label.switch input")
    if preview.is_checked():
        preview.uncheck()
    with page.expect_response(lambda r: "/api/ops/filters.gaussian/preview" in r.url and r.ok, timeout=30_000):
        preview.check()
    app.apply_sheet()
    wait_until(lambda: app.state()["history"][iid]["undo"] == ["Gaussian Blur"], what="Gaussian Blur in the undo history")
    undo = page.locator('.topbar button[title^="Undo"]')
    expect(undo).to_be_enabled()
    undo.click()
    wait_until(lambda: app.state()["history"][iid]["undo"] == [] and app.state()["history"][iid]["redo"] == ["Gaussian Blur"], what="undo to clear the history")
    expect(undo).to_be_disabled()
    expect(page.locator('.topbar button[title^="Redo"]')).to_be_enabled()
    no_console_errors(console)


# ------------------------------------------------------------------ threshold, mask, particles
def test_threshold_mask_and_analyze_particles(page: Page, app: LumenApp, console: Console) -> None:
    iid = app.open_sample(SAMPLE_3CH)
    thr = app.section("Threshold")
    thr.locator("button", has_text=re.compile(r"^\s*Auto\s*$")).click()
    wait_until(lambda: app.state()["thresholds"].get(iid), what="an auto threshold")
    expect(thr).to_contain_text("% selected")
    thr.locator("button", has_text="Apply → mask").click()
    expect(app.sheet().locator("h3")).to_have_text("Apply Threshold → Mask")
    app.apply_sheet()
    # a multi-channel image thresholds into a new single-channel 8-bit mask image, which becomes active
    mask = wait_until(lambda: next((i for i in app.state()["images"] if i["id"] != iid and i["id"] == app.state()["active"]), None), what="the mask image to become active")
    assert mask["bit_depth"].startswith("8") and mask["shape"]["c"] == 1, mask
    app.palette("analyze particles", "Analyze Particles")
    expect(app.sheet().locator("h3")).to_have_text("Analyze Particles")
    app.apply_sheet()
    rois = app.section("ROI Manager")
    expect(rois.locator(".roi-item").first).to_be_visible(timeout=30_000)
    n = wait_until(lambda: len(app.state()["rois"]), what="ROIs in the manager")
    assert n > 10, n
    expect(rois.locator(".roi-item")).to_have_count(n)
    expect(app.section("Results").locator("table.results tbody tr").first).to_be_visible()
    no_console_errors(console)


# ------------------------------------------------------------------ selection + measure
def test_rectangle_selection_and_measure(page: Page, app: LumenApp, console: Console) -> None:
    app.open_sample(SAMPLE_3CH)
    page.locator('.rail button[title="Rectangle (1)"]').click()
    app.drag(page.locator(".canvas-wrap"), 0.35, 0.35, 0.6, 0.6)
    expect(page.locator(".statusbar .pill", has_text="rect")).to_be_visible()
    page.keyboard.press("m")
    results = app.section("Results")
    expect(results.locator("table.results tbody tr")).to_have_count(1, timeout=20_000)
    rows = app.state()["results"]["rows"]
    assert len(rows) == 1
    no_console_errors(console)


# ------------------------------------------------------------------ 3-D view (WebGL)
def test_volume_view_renders(page: Page, app: LumenApp, console: Console) -> None:
    app.open_sample(SAMPLE_3D)
    body = app.section("3-D View")
    if body.get_by_text("WebGL2 is not available").count():
        pytest.skip("WebGL2 is not available in this browser")
    canvas = body.locator(".vol-canvas")
    expect(canvas).to_be_visible()
    lit_fraction = """() => {
      const c = document.querySelector('.vol-canvas');
      if (!c || !c.width || !c.height) return 0;
      const o = document.createElement('canvas'); o.width = c.width; o.height = c.height;
      const g = o.getContext('2d'); g.drawImage(c, 0, 0);
      const d = g.getImageData(0, 0, o.width, o.height).data;
      let lit = 0, n = 0;
      for (let i = 0; i < d.length; i += 4 * 7) { n++; if (d[i] + d[i + 1] + d[i + 2] > 30) lit++; }
      return lit / n;
    }"""
    wait_until(lambda: page.evaluate(lit_fraction) > 0.01, timeout=90, interval=1, what="the volume canvas to show something")
    assert page.evaluate(lit_fraction) > 0.01
    no_console_errors(console)


# ------------------------------------------------------------------ chrome: theme, help
def test_theme_toggle(page: Page, console: Console) -> None:
    theme = lambda: page.evaluate("document.documentElement.dataset.theme")  # noqa: E731
    assert theme() == "dark"
    page.locator('button[title="Toggle theme"]').click()
    wait_until(lambda: theme() == "light", what="light theme")
    assert page.evaluate("localStorage.getItem('lumen.theme')") == "light"
    page.locator('button[title="Toggle theme"]').click()
    wait_until(lambda: theme() == "dark", what="dark theme")
    no_console_errors(console)


def test_help_overlay(page: Page, console: Console) -> None:
    page.keyboard.press("?")
    overlay = page.locator(".help-overlay")
    expect(overlay).to_be_visible()
    expect(overlay.locator("h2")).to_have_text("Keyboard shortcuts")
    page.keyboard.press("Escape")
    expect(overlay).to_be_hidden()
    page.locator('button[title="Keyboard shortcuts (?)"]').click()
    expect(overlay).to_be_visible()
    overlay.locator("button", has_text="Close").click()
    expect(overlay).to_be_hidden()
    no_console_errors(console)


# ------------------------------------------------------------------ accessibility (needs a build with the aria attributes)
def _needs_aria_build(page: Page) -> None:
    if page.locator('.rail[role="toolbar"]').count() == 0:
        pytest.skip("the served lumen/static build predates the accessibility attributes (rebuild the frontend)")


def test_tool_rail_aria_pressed(page: Page, app: LumenApp, console: Console) -> None:
    _needs_aria_build(page)
    rail = page.locator('.rail[role="toolbar"]')
    pointer = rail.locator('button[aria-label^="Select / move"]')
    rect = rail.locator('button[aria-label^="Rectangle"]')
    expect(pointer).to_have_attribute("aria-pressed", "true")
    rect.click()
    expect(rect).to_have_attribute("aria-pressed", "true")
    expect(pointer).to_have_attribute("aria-pressed", "false")
    # the status bar is a landmark, not a live region: its cursor readout changes on every mouse move (E7-19)
    expect(page.locator(".statusbar")).to_have_attribute("role", "contentinfo")
    for label in ("Toggle theme", "Keyboard shortcuts (?)"):
        expect(page.locator(f'.topbar button[aria-label="{label}"]')).to_have_count(1)
    no_console_errors(console)


def test_range_slider_keyboard(page: Page, app: LumenApp, console: Console) -> None:
    _needs_aria_build(page)
    iid = app.open_sample(SAMPLE_3CH)
    thr = app.section("Threshold")
    thr.locator("button", has_text=re.compile(r"^\s*Auto\s*$")).click()
    wait_until(lambda: app.state()["thresholds"].get(iid), what="an auto threshold")
    lo_handle = thr.locator('.range-slider [role="slider"]').first
    lo_handle.focus()
    before = float(lo_handle.get_attribute("aria-valuenow") or 0)
    vmin, vmax = (float(lo_handle.get_attribute(a) or 0) for a in ("aria-valuemin", "aria-valuemax"))
    span = vmax - vmin
    now = lambda: float(lo_handle.get_attribute("aria-valuenow") or 0)  # noqa: E731
    page.keyboard.press("ArrowRight")  # +1 % of the range
    wait_until(lambda: now() != before, what="the handle to move")
    after = now()
    assert abs((after - before) - span * 0.01) < 1e-6 * max(1.0, span), (before, after, span)
    page.keyboard.press("Shift+ArrowLeft")  # -10 % (clamped at the minimum)
    wait_until(lambda: now() < after, what="a 10 % step back")
    final = now()
    assert abs(final - max(vmin, after - span * 0.1)) < 1e-6 * max(1.0, span), (after, final, vmin, span)
    # the keyboard steps are committed to the server-side threshold (integer images round the value)
    wait_until(lambda: abs(app.state()["thresholds"][iid]["lower"] - final) <= 1.0, what="the threshold to follow the keyboard")
    no_console_errors(console)
