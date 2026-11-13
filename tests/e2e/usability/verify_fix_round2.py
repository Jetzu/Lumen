"""Browser check of the GUI fix round 2 (the E5–E7 items of docs/USABILITY.md §5.4) against a running Lumen server: installed Chrome, headless. Run from a venv with Playwright: `python tests/e2e/usability/verify_fix_round2.py http://127.0.0.1:8766`. Prints one line per check (17 checks); "ok" everywhere on the build of 8 Nov 2026.

Usage: verify_fix_round2.py <url>
"""
from __future__ import annotations

import re
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"
IGNORED = re.compile(r"GL ReadPixels|GL Driver Message|SwiftShader|WebGL|Automatic fallback", re.I)
results: dict[str, str] = {}


def open_palette(page, query):
    page.keyboard.press("Escape")
    page.keyboard.press("Meta+k")
    inp = page.locator(".palette input")
    inp.wait_for(timeout=10_000)
    inp.fill(query)
    page.wait_for_timeout(350)


def palette_names(page, query):
    open_palette(page, query)
    names = page.locator(".palette-item .pi-name").all_text_contents()
    page.keyboard.press("Escape")
    page.wait_for_timeout(150)
    return [n.strip() for n in names]


def run_palette(page, query, name):
    open_palette(page, query)
    item = page.locator(".palette-item", has=page.locator(".pi-name", has_text=re.compile(rf"^{re.escape(name)}$")))
    item.first.click(timeout=10_000)
    page.locator(".palette").wait_for(state="hidden", timeout=10_000)


def open_sample(page, contains):
    run_palette(page, "open sample", "Open Sample")
    sheet = page.locator(".sheet")
    sheet.wait_for(timeout=10_000)
    sel = sheet.locator("select").first
    labels = sel.locator("option").all_text_contents()
    pick = next((l for l in labels if contains.lower() in l.lower()), labels[0])
    sel.select_option(label=pick)
    sheet.locator(".sheet-foot button.primary").click()
    page.locator(".statusbar").get_by_text(re.compile(r"\d+×\d+ px")).wait_for(timeout=60_000)
    page.wait_for_timeout(800)
    return pick


def section(page, title):
    """The section toggle button whose text contains `title`; expands it when collapsed."""
    btn = page.locator("button.section-toggle", has_text=re.compile(re.escape(title), re.I)).first
    btn.wait_for(timeout=10_000)
    if btn.get_attribute("aria-expanded") == "false":
        btn.click()
        page.wait_for_timeout(300)
    return btn


def section_titles(page):
    return [t.strip() for t in page.locator("button.section-toggle").all_text_contents()]


CONTRAST_JS = """
(el) => {
  const lum = (c) => { const [r,g,b] = c.match(/\\d+(\\.\\d+)?/g).slice(0,3).map(Number).map(v => { v/=255; return v<=0.03928? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); }); return 0.2126*r+0.7152*g+0.0722*b; };
  const fg = getComputedStyle(el).color;
  let node = el, bg = null;
  while (node) { const b = getComputedStyle(node).backgroundColor; if (b && !/rgba\\(\\d+, \\d+, \\d+, 0\\)/.test(b) && b !== 'transparent') { bg = b; break; } node = node.parentElement; }
  if (!bg) bg = 'rgb(255, 255, 255)';
  const l1 = lum(fg), l2 = lum(bg);
  return { fg, bg, ratio: (Math.max(l1,l2)+0.05)/(Math.min(l1,l2)+0.05), weight: getComputedStyle(el).fontWeight };
}
"""

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    ctx = browser.new_context(viewport={"width": 1500, "height": 950})
    page = ctx.new_page()
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and not IGNORED.search(m.text) else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(URL, wait_until="networkidle")
    page.wait_for_timeout(800)

    # palette ranking and LUT entries (N1, N3, N4)
    try:
        fire = palette_names(page, "fire")
        results["palette_fire"] = "ok" if fire and "fire" in fire[0].lower() and fire[0].lower().startswith("lut") else f"first hits: {fire[:3]}"
        vir = palette_names(page, "viridis")
        results["palette_viridis"] = "ok" if any("viridis" in n.lower() for n in vir) else f"hits: {vir[:3]}"
        clahe = palette_names(page, "enhance local contrast")
        results["palette_fiji_alias"] = "ok" if any("clahe" in n.lower() for n in clahe) else f"hits: {clahe[:3]}"
        osample = palette_names(page, "open sample")
        results["palette_open_sample_first"] = "ok" if osample and osample[0] == "Open Sample" else f"first: {osample[:2]}"
        applylut = palette_names(page, "apply lut")
        results["palette_apply_lut_names"] = "ok" if applylut and len(set(applylut)) == len(applylut) and any("Display Range" in n for n in applylut) else f"hits: {applylut[:4]}"
    except Exception as e:  # noqa: BLE001
        results["palette"] = f"error: {str(e)[:160]}"

    # menubar contrast and the Lookup Tables submenu (N1, N26)
    try:
        trig = page.locator(".menubar .menu-trigger", has_text=re.compile(r"^Image$")).first
        c = trig.evaluate(CONTRAST_JS)
        results["menubar_contrast"] = f"ok ({c['ratio']:.2f}:1, weight {c['weight']})" if c["ratio"] >= 4.5 else f"ratio {c['ratio']:.2f} fg {c['fg']} bg {c['bg']}"
        trig.click()
        page.wait_for_timeout(300)
        lt = page.locator('[role="menuitem"]', has_text=re.compile(r"Lookup Tables")).first
        lt.hover()
        page.wait_for_timeout(400)
        items = [t.strip() for t in page.locator('[role="menuitem"]').all_text_contents()]
        has_fire = any(t == "Fire" or t.startswith("Fire") for t in items)
        has_set = any("Set LUT" in t for t in items)
        has_apply = any("Apply Display Range" in t for t in items)
        results["lut_submenu"] = "ok" if has_fire and has_set and has_apply else f"fire={has_fire} set={has_set} apply={has_apply} items={[t for t in items if 'LUT' in t or t in ('Fire','Grays')][:6]}"
        page.keyboard.press("Escape")
        page.keyboard.press("Escape")
    except Exception as e:  # noqa: BLE001
        results["lut_submenu"] = f"error: {str(e)[:160]}"

    # a 3-channel sample: the Threshold section names its channel (N6); ROI Manager has 'This image only' (N19)
    try:
        page.keyboard.press("Escape")
        pick = open_sample(page, "3 channels")
        section(page, "Threshold")
        thr = page.locator(".section", has=page.locator("button.section-toggle", has_text=re.compile("Threshold", re.I))).first
        txt = thr.inner_text().lower() if thr.count() else page.inner_text("body").lower()
        results["threshold_names_channel"] = "ok" if ("channel" in txt and ("dapi" in txt or "nuclei" in txt or "c1" in txt or "1" in txt)) else f"text: {txt[:160]!r}"
        section(page, "ROI Manager")
        body = page.inner_text("body").lower()
        results["roi_this_image_only"] = "ok" if "this image only" in body else "switch text not found"
    except Exception as e:  # noqa: BLE001
        results["threshold_names_channel"] = f"error: {str(e)[:160]}"

    # Gaussian via the sheet → recorder marker + line, history parameters, Edit ▸ Undo names the step, → IJM (N10, N16, N23, N33)
    try:
        run_palette(page, "gaussian blur", "Gaussian Blur")
        sheet = page.locator(".sheet")
        sheet.wait_for(timeout=10_000)
        sigma = sheet.locator('input[inputmode="decimal"], input[type="number"]').first
        sigma.fill("3")
        sheet.locator(".sheet-foot button.primary").click()
        page.locator(".sheet").wait_for(state="hidden", timeout=30_000)
        page.wait_for_timeout(800)
        section(page, "History")
        hist = page.inner_text("body")
        results["history_shows_params"] = "ok" if re.search(r"sigma\s*=\s*3", hist) else "no 'sigma=3' in the page"
        page.locator(".menubar .menu-trigger", has_text=re.compile(r"^Edit$")).first.click()
        page.wait_for_timeout(300)
        items = [t.strip() for t in page.locator('[role="menuitem"]').all_text_contents()]
        results["edit_undo_named"] = "ok" if any(t.startswith("Undo Gaussian") for t in items) else f"undo items: {[t for t in items if t.startswith('Undo')]}"
        page.keyboard.press("Escape")
        page.keyboard.press("Escape")
        section(page, "Script & Recorder")
        lines = [t.strip() for t in page.locator(".macro-lines").first.inner_text().splitlines() if t.strip()]
        marker_ok = bool(lines) and lines[0].startswith("# Lumen recorder")
        gauss_ok = any('lumen.run("filters.gaussian"' in l and "sigma=3" in l for l in lines)
        results["recorder_marker_and_line"] = "ok" if marker_ok and gauss_ok else f"lines: {lines[:3]}"
        page.get_by_role("button", name=re.compile(r"→ IJM")).first.click()
        page.wait_for_timeout(600)
        # the editor may ask Replace / Append when it holds text
        for label in ("Replace", "Append"):
            b = page.get_by_role("button", name=re.compile(rf"^{label}")).first
            if b.count() and b.is_visible():
                b.click()
                page.wait_for_timeout(400)
                break
        editor = page.locator("textarea").filter(has_text=re.compile(r"run\(")).first
        code = editor.input_value() if editor.count() else "\n".join(t.input_value() for t in page.locator("textarea").all())
        results["ijm_export_in_editor"] = "ok" if 'run("Gaussian Blur...", "sigma=3' in code else f"editor: {code[:160]!r}"
    except Exception as e:  # noqa: BLE001
        results["recorder"] = f"error: {str(e)[:160]}"

    # a time series via New Image: Tracks section present and auto-opened with the wizard (N21, N29)
    try:
        page.keyboard.press("Escape")
        run_palette(page, "new image", "New Image")
        sheet = page.locator(".sheet")
        sheet.wait_for(timeout=10_000)
        frames = sheet.get_by_label(re.compile(r"^Frames")).first
        frames.fill("6")
        sheet.locator(".sheet-foot button.primary").click()
        page.locator(".sheet").wait_for(state="hidden", timeout=30_000)
        page.wait_for_timeout(1000)
        titles = section_titles(page)
        results["tracks_section_timeseries"] = "ok" if any("tracks" in t.lower() for t in titles) else f"titles: {titles}"
        body = page.inner_text("body").lower()
        results["track_wizard_text"] = "ok" if "wizard" in body else "no wizard text"
        section(page, "Pixel Classifier")
        body = page.inner_text("body").lower()
        results["pixel_classifier_panel"] = "ok" if ("add class" in body or "train" in body) else "panel text not found"
    except Exception as e:  # noqa: BLE001
        results["tracks_section_timeseries"] = f"error: {str(e)[:160]}"

    # out-of-range value: inline message and disabled Apply (N14)
    try:
        page.keyboard.press("Escape")
        run_palette(page, "median", "Median")
        sheet = page.locator(".sheet")
        sheet.wait_for(timeout=10_000)
        radius = sheet.locator('input[inputmode="decimal"], input[type="number"]').first
        radius.fill("5000")
        page.wait_for_timeout(400)
        apply_btn = sheet.locator(".sheet-foot button.primary")
        disabled = apply_btn.is_disabled()
        txt = sheet.inner_text().lower()
        results["out_of_range_inline"] = "ok" if disabled and ("between" in txt or "at most" in txt or "≤" in txt or "max" in txt) else f"disabled={disabled} text={txt[-160:]!r}"
        sheet.locator(".sheet-foot button", has_text=re.compile("Cancel")).first.click()
    except Exception as e:  # noqa: BLE001
        results["out_of_range_inline"] = f"error: {str(e)[:160]}"

    results["console_errors"] = "none" if not errors else f"{len(errors)}: {errors[:3]}"
    ctx.close()
    browser.close()

for k, v in results.items():
    print(f"{k:30s} {v}")
