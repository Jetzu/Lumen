#!/usr/bin/env python
"""Evidence collector for the heuristic evaluations of docs/USABILITY.md section 3.

Drives a running Lumen server in headless Google Chrome (Playwright, channel "chrome") through the
UI locations an evaluator visits - menus, command palette, parameter sheets, every inspector
section, preferences, the help overlay, split view, the script editor, the pixel classifier, the
tracks panel and the 3-D view - and records *evidence*: screenshots, DOM facts (roles, labels,
contrast ratios, focus order, target sizes, text of hints and errors) and the server-side outcome
of each step read back through the REST API.

Three probe sets correspond to the three evaluator personas of USABILITY.md section 1.1:

  power      a Fiji power user who scripts everything (recorder, script editor, macros, repeat,
             custom shortcuts, batch, stack scope, API docs)
  biologist  a wet-lab biologist who uses Fiji twice a month (menus rather than the palette,
             labels and hints, errors, closing unsaved images, file dialog, workflows, laptop
             viewport 1366x768)
  a11y       an accessibility / HCI reviewer (landmarks, focus order, keyboard reachability,
             ARIA of palette / menus / sheets / dialogs, contrast in both themes, target sizes,
             unlabeled controls, reduced motion, reflow at a narrow viewport)

The script collects facts; the severity judgements in USABILITY.md are the evaluator's, made from
this evidence and the screenshots. Nothing is asserted here: a probe that fails is recorded as a
failed probe and the walk continues.

Usage:   python heuristic_walkthrough.py [--url http://127.0.0.1:8766] [--out DIR] [--persona power|biologist|a11y|all]
Output:  <out>/<persona>/NN_<name>.png and <out>/<persona>/evidence.json (observations + facts)

Only the images the walk opened are closed again at the end; the rest of the session's state is
left as it was. Selectors follow tests/e2e/conftest.py and tests/e2e/usability/measure_lumen.py.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable
from urllib import request as urlrequest

from playwright.sync_api import sync_playwright

SAMPLE_3CH = "Synthetic nuclei (3 channels)"
SAMPLE_BLOBS = "Blobs"
SAMPLE_3D = "Cells (3-D, 2 channels)"
SAMPLE_TL = "Nuclei timelapse (synthetic)"
SAMPLE_COINS = "Coins"

SWIFTSHADER = ["--ignore-gpu-blocklist", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]


# ------------------------------------------------------------------ small helpers
def http_json(url: str, path: str, method: str = "GET", body: Any = None, timeout: float = 60) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = urlrequest.Request(url + path, data=data, method=method, headers={"Content-Type": "application/json"})
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


JS_ACTIVE = """() => {
  const e = document.activeElement; if (!e) return {tag: 'none'};
  const r = e.getBoundingClientRect();
  const region = e.closest('.help-overlay') ? 'help' : e.closest('.palette') ? 'palette' : e.closest('.sheet') ? 'sheet'
    : e.closest('.modal-backdrop') ? 'modal' : e.closest('.topbar') ? 'topbar' : e.closest('.rail') ? 'rail'
    : e.closest('.tabs') ? 'tabs' : e.closest('.section') ? 'inspector' : e.closest('.statusbar') ? 'statusbar'
    : e.closest('.viewport') ? 'viewport' : e === document.body ? 'body' : 'other';
  return {tag: e.tagName.toLowerCase(), cls: String(e.className).slice(0, 40), role: e.getAttribute('role'),
          name: (e.getAttribute('aria-label') || e.getAttribute('title') || e.textContent || '').trim().slice(0, 40),
          region, outline: getComputedStyle(e).outlineStyle, visible: !!(r.width && r.height)};
}"""

JS_CONTRAST = """(sels) => {
  const lin = v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const lum = c => 0.2126 * lin(c.r) + 0.7152 * lin(c.g) + 0.0722 * lin(c.b);
  const parse = s => { const m = (s || '').match(/rgba?\\(([^)]+)\\)/); if (!m) return null; const p = m[1].split(',').map(x => parseFloat(x)); return {r: p[0], g: p[1], b: p[2], a: p.length > 3 ? p[3] : 1}; };
  const blend = (c, bg) => ({r: c.r * c.a + bg.r * (1 - c.a), g: c.g * c.a + bg.g * (1 - c.a), b: c.b * c.a + bg.b * (1 - c.a), a: 1});
  const bgOf = el => { let e = el, acc = null; while (e) { const c = parse(getComputedStyle(e).backgroundColor); if (c && c.a > 0) { if (c.a >= 0.99) return acc ? blend(acc, c) : c; acc = acc ? blend(acc, c) : c; } e = e.parentElement; } const base = {r: 255, g: 255, b: 255, a: 1}; return acc ? blend(acc, base) : base; };
  const out = [];
  for (const spec of sels) {
    const [sel, pseudo] = spec.split('::');
    const els = Array.from(document.querySelectorAll(sel)).filter(e => e.getClientRects().length && getComputedStyle(e).visibility !== 'hidden');
    const el = els[0]; if (!el) { out.push({sel: spec, missing: true}); continue; }
    const cs = pseudo ? getComputedStyle(el, '::' + pseudo) : getComputedStyle(el);
    let fg = parse(cs.color); const bg = bgOf(el);
    if (!fg) { out.push({sel: spec, missing: true}); continue; }
    if (fg.a < 1) fg = blend(fg, bg);
    const L1 = lum(fg), L2 = lum(bg);
    const ratio = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
    const size = parseFloat(cs.fontSize), weight = parseInt(cs.fontWeight) || 400;
    const large = size >= 24 || (size >= 18.66 && weight >= 700);
    out.push({sel: spec, text: (el.textContent || '').trim().slice(0, 30), fg: `rgb(${Math.round(fg.r)},${Math.round(fg.g)},${Math.round(fg.b)})`,
              bg: `rgb(${Math.round(bg.r)},${Math.round(bg.g)},${Math.round(bg.b)})`, ratio: Math.round(ratio * 100) / 100,
              px: size, weight, large, passAA: ratio >= (large ? 3 : 4.5), passUI: ratio >= 3});
  }
  return out;
}"""

JS_TARGETS = """() => {
  const sel = 'button, [role=button], [role=slider], [role=tab], input:not([type=hidden]), select, a[href], .tab, .section-head, .menu-trigger, .roi-item, .palette-item, .chip';
  const res = [];
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue;
    if (getComputedStyle(el).visibility === 'hidden') continue;
    res.push({tag: el.tagName.toLowerCase(), cls: String(el.className).slice(0, 36),
              name: (el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '').trim().slice(0, 36),
              w: Math.round(r.width), h: Math.round(r.height)});
  }
  return res;
}"""

JS_UNLABELED = """() => {
  const bad = [];
  for (const el of document.querySelectorAll('button, input:not([type=hidden]), select, textarea, [role=slider]')) {
    const r = el.getBoundingClientRect(); if (!r.width || !r.height) continue;
    let name = el.getAttribute('aria-label') || el.getAttribute('aria-labelledby') || el.getAttribute('title') || el.getAttribute('placeholder') || '';
    if (!name && el.id) { const l = document.querySelector(`label[for="${CSS.escape(el.id)}"]`); if (l) name = l.textContent; }
    if (!name) { const l = el.closest('label'); if (l) name = l.textContent; }
    if (!name && el.tagName === 'BUTTON') name = el.textContent;
    if (!name || !name.trim()) bad.push({tag: el.tagName.toLowerCase(), type: el.getAttribute('type'), cls: String(el.className).slice(0, 40), where: (el.closest('.section') ? 'inspector ' + (el.closest('.section').getAttribute('data-section') || '') : el.closest('.sheet') ? 'sheet' : el.closest('.topbar') ? 'topbar' : el.closest('.tabs') ? 'tabs' : el.closest('.statusbar') ? 'statusbar' : el.closest('.modal-backdrop') ? 'modal' : 'other'), html: el.outerHTML.slice(0, 110)});
  }
  return bad;
}"""

JS_LANDMARKS = """() => {
  const roles = ['main','banner','navigation','complementary','contentinfo','region','toolbar','menubar','menu','menuitem','status','dialog','alert','tablist','tab','listbox','option','combobox','slider','separator','group','log'];
  const native = {main: 'main', banner: 'header', navigation: 'nav', complementary: 'aside', contentinfo: 'footer'};
  const counts = {};
  for (const r of roles) { const n = document.querySelectorAll(`[role="${r}"]`).length + (native[r] ? document.querySelectorAll(native[r]).length : 0); if (n) counts[r] = n; }
  return {lang: document.documentElement.lang, title: document.title, landmarks: counts,
          headings: Array.from(document.querySelectorAll('h1,h2,h3,h4')).map(h => h.tagName + ':' + h.textContent.trim().slice(0, 30)).slice(0, 25),
          skipLink: !!document.querySelector('a.skip-link, a[href="#main"], .skip'),
          canvases: Array.from(document.querySelectorAll('canvas')).map(c => ({role: c.getAttribute('role'), label: c.getAttribute('aria-label'), w: c.width, h: c.height, tabindex: c.getAttribute('tabindex')})),
          live: Array.from(document.querySelectorAll('[aria-live]')).map(e => String(e.className).slice(0, 20) + ':' + e.getAttribute('aria-live')),
          sectionHeads: Array.from(document.querySelectorAll('.section-head')).map(h => ({text: h.firstChild && h.childNodes[1] ? h.childNodes[1].textContent : h.textContent.trim().slice(0, 20), tag: h.tagName.toLowerCase(), role: h.getAttribute('role'), tabindex: h.getAttribute('tabindex'), expanded: h.getAttribute('aria-expanded')})),
          tabs: Array.from(document.querySelectorAll('.tabs .tab')).map(t => ({tag: t.tagName.toLowerCase(), role: t.getAttribute('role'), tabindex: t.getAttribute('tabindex'), selected: t.getAttribute('aria-selected')})),
          reducedMotionRule: Array.from(document.styleSheets).some(ss => { try { return Array.from(ss.cssRules).some(r => r.media && /prefers-reduced-motion/.test(r.media.mediaText)); } catch (e) { return false; } }),
          animated: Array.from(document.querySelectorAll('*')).filter(e => { const cs = getComputedStyle(e); return cs.animationName && cs.animationName !== 'none' && e.getClientRects().length; }).map(e => String(e.className).slice(0, 30) + ':' + getComputedStyle(e).animationName).slice(0, 10)};
}"""


# ------------------------------------------------------------------ the walk
class Walk:
    def __init__(self, page: Any, url: str, persona: str, out: Path) -> None:
        self.page = page
        self.url = url
        self.persona = persona
        self.out = out
        self.out.mkdir(parents=True, exist_ok=True)
        self.n = 0
        self.obs: list[dict[str, Any]] = []
        self.facts: dict[str, Any] = {}
        self.opened: list[str] = []
        self.console: list[str] = []
        page.on("console", lambda m: self.console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.on("pageerror", lambda e: self.console.append(f"pageerror: {e}"))

    # --- REST (verification + cleanup only)
    def api(self, path: str, method: str = "GET", body: Any = None) -> Any:
        return http_json(self.url, path, method, body)

    def state(self) -> dict[str, Any]:
        return self.api("/api/state")

    def image(self, iid: str) -> dict[str, Any] | None:
        return next((i for i in self.state()["images"] if i["id"] == iid), None)

    # --- evidence
    def shot(self, name: str, locator: Any = None, full: bool = False) -> str:
        self.n += 1
        p = self.out / f"{self.n:02d}_{name}.png"
        try:
            if locator is not None:
                locator.screenshot(path=str(p), timeout=10_000)
            else:
                self.page.screenshot(path=str(p), full_page=full)
        except Exception as e:
            self.page.screenshot(path=str(p))
            self.note("screenshot", f"locator screenshot failed for {name}: {str(e).splitlines()[0]}")
        return p.name

    def note(self, location: str, text: str, shot: str | None = None, **data: Any) -> None:
        entry: dict[str, Any] = {"location": location, "text": text}
        if shot:
            entry["shot"] = shot
        entry.update(data)
        self.obs.append(entry)
        print(f"  [{location}] {text}" + (f"  ({shot})" if shot else ""))

    def fact(self, key: str, value: Any) -> None:
        self.facts[key] = value

    def probe(self, name: str, fn: Callable[[], None]) -> None:
        print(f"-- {self.persona}: {name}")
        try:
            fn()
        except Exception as e:
            self.note(name, f"PROBE FAILED: {str(e).splitlines()[0][:200]}", shot=self.shot(f"failed_{name}"))
            traceback.print_exc(limit=2)
        finally:
            self.settle()

    def settle(self) -> None:
        """Close whatever overlay a probe left open (Esc in Lumen's fixed order), so the next probe starts clean."""
        for i in range(6):
            if self.page.locator(".palette, .sheet, .help-overlay, .modal-backdrop, .menu-popover:visible").count() == 0:
                break
            if i >= 2:
                # an input that swallows Esc (the shortcut-capture field): drop focus first, then click the backdrop corner
                self.page.evaluate("() => document.activeElement && document.activeElement.blur()")
            if i >= 4 and self.page.locator(".modal-backdrop").count():
                self.page.locator(".modal-backdrop").first.click(position={"x": 3, "y": 3}, force=True)
            else:
                self.page.keyboard.press("Escape")
            self.page.wait_for_timeout(150)

    # --- UI drivers (same selectors as tests/e2e/conftest.py)
    def palette_open(self, query: str) -> dict[str, Any]:
        self.page.keyboard.press("Control+k")
        inp = self.page.locator(".palette input")
        inp.wait_for(timeout=10_000)
        inp.fill(query)
        self.page.wait_for_timeout(250)
        items = self.page.locator(".palette-item")
        entries = []
        for i in range(min(items.count(), 8)):
            it = items.nth(i)
            entries.append({
                "name": it.locator(".pi-name").inner_text() if it.locator(".pi-name").count() else it.inner_text()[:40],
                "cat": it.locator(".pi-cat").inner_text() if it.locator(".pi-cat").count() else "",
                "desc": it.locator(".pi-desc").inner_text() if it.locator(".pi-desc").count() else "",
            })
        sel = self.page.locator(".palette-item.sel .pi-name")
        return {"query": query, "count": items.count(), "selected": sel.first.inner_text() if sel.count() else None,
                "entries": entries, "empty": self.page.locator(".palette-empty").inner_text() if self.page.locator(".palette-empty").count() else ""}

    def palette_run(self, query: str, name: str | None = None) -> str:
        info = self.palette_open(query)
        if name is None or info["selected"] == name:
            self.page.keyboard.press("Enter")
        else:
            item = self.page.locator(".palette-item", has=self.page.locator(".pi-name", has_text=re.compile(rf"^{re.escape(name)}$")))
            item.first.click(timeout=10_000)
        self.page.locator(".palette").wait_for(state="hidden", timeout=10_000)
        return info["selected"] or ""

    def sheet(self, timeout: int = 15_000) -> Any:
        sh = self.page.locator(".sheet")
        sh.wait_for(timeout=timeout)
        return sh

    def sheet_info(self) -> dict[str, Any]:
        sh = self.page.locator(".sheet")
        if not sh.count():
            return {"present": False}
        fields = [e.inner_text().strip() for e in sh.locator("label, .field > span:first-child, .field-label").all()]
        buttons = [b.inner_text().strip() or (b.get_attribute("aria-label") or b.get_attribute("title") or "") for b in sh.locator("button").all()]
        return {"present": True, "title": sh.locator("h3").first.inner_text(), "fields": [f for f in fields if f][:20],
                "buttons": [b for b in buttons if b][:12], "preview_switch": sh.locator(".sheet-foot label.switch input").count() > 0,
                "preview_on": sh.locator(".sheet-foot label.switch input").is_checked() if sh.locator(".sheet-foot label.switch input").count() else None,
                "description": sh.locator(".sheet-desc, .desc, p").first.inner_text()[:160] if sh.locator(".sheet-desc, .desc, p").count() else "",
                "scope_select": sh.locator('select[aria-label="Apply to"]').count() > 0,
                "scope_options": [o.inner_text() for o in sh.locator('select[aria-label="Apply to"] option').all()] if sh.locator('select[aria-label="Apply to"]').count() else [],
                "advanced": sh.locator(".advanced-toggle").count() > 0,
                "error": sh.locator(".sheet-error").inner_text() if sh.locator(".sheet-error").count() else ""}

    def apply_sheet(self, timeout: int = 180_000) -> None:
        self.page.locator(".sheet .sheet-foot button.primary").click()
        self.page.locator(".sheet").wait_for(state="hidden", timeout=timeout)

    def open_sample(self, name: str) -> str:
        before = {i["id"] for i in self.state().get("images", [])}
        self.palette_run("open sample", "Open Sample")
        self.sheet().locator("select").first.select_option(label=name)
        self.apply_sheet()
        new = wait_until(lambda: [i["id"] for i in self.state().get("images", []) if i["id"] not in before], 120, what=f"sample {name!r}")
        self.page.locator(".statusbar").get_by_text(re.compile(r"\d+×\d+ px")).wait_for(timeout=30_000)
        self.opened.append(new[0])
        return new[0]

    def section_head(self, title: str) -> Any:
        return self.page.locator(f"xpath=//button[contains(@class,'section-toggle')][text()={title!r}]").first

    def section(self, title: str, open: bool = True) -> Any:
        head = self.section_head(title)
        if not head.count():
            return None
        is_open = "open" in (head.get_attribute("class") or "")
        if is_open != open:
            head.click()
        body = head.locator("xpath=following-sibling::div[contains(@class,'section-body')]")
        if open:
            body.wait_for(timeout=10_000)
        return body

    def menu_open(self, label: str) -> list[dict[str, Any]]:
        self.settle()
        trig = self.page.locator(".menubar .menu-trigger", has_text=re.compile(rf"^{re.escape(label)}$")).first
        trig.click()
        pop = self.page.locator(".menu-popover:not(.submenu)").first
        pop.wait_for(timeout=5_000)
        items = []
        for it in pop.locator(":scope > .menu-item, :scope > .menu-sep").all():
            if "menu-sep" in (it.get_attribute("class") or ""):
                items.append({"sep": True})
                continue
            items.append({"label": it.locator(".label").first.inner_text() if it.locator(".label").count() else it.inner_text()[:40],
                          "kbd": it.locator(":scope > .kbd").first.inner_text() if it.locator(":scope > .kbd").count() else "",
                          "submenu": it.get_attribute("aria-haspopup") == "menu",
                          "disabled": it.get_attribute("aria-disabled") == "true", "title": it.get_attribute("title") or ""})
        return items

    def menu_click(self, path: list[str]) -> None:
        """Click through a menu path, hovering sub-menus: ['Image', 'Adjust', 'Threshold…']."""
        self.menu_open(path[0])
        scope = self.page.locator(".menu-popover:not(.submenu)").first
        for i, label in enumerate(path[1:]):
            item = scope.locator(".menu-item", has=self.page.locator(":scope > .label", has_text=re.compile(rf"^{re.escape(label)}"))).first
            item.wait_for(timeout=5_000)
            if i < len(path) - 2:
                item.hover()
                scope = item.locator(".menu-popover.submenu").first
                scope.wait_for(timeout=5_000)
            else:
                item.click()
        self.page.wait_for_timeout(300)

    def cleanup(self) -> None:
        for iid in self.opened:
            try:
                self.api(f"/api/images/{iid}", "DELETE")
            except Exception:
                pass
        self.opened.clear()

    def save(self) -> None:
        errs = [c for c in self.console if not any(k in c for k in ("ReadPixels", "GL Driver", "GL_INVALID", "SwiftShader", "WebGL", "GPU stall", "Automatic fallback"))]
        (self.out / "evidence.json").write_text(json.dumps({"persona": self.persona, "url": self.url, "observations": self.obs, "facts": self.facts,
                                                            "console": errs[:40]}, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"saved {self.out / 'evidence.json'} ({len(self.obs)} observations, {len(self.facts)} facts, {len(errs)} console errors/warnings)")


# ------------------------------------------------------------------ shared steps
def common_tour(W: Walk) -> None:
    """Menus, inspector sections, preferences, help overlay: every persona looks at these."""
    def menus() -> None:
        all_menus: dict[str, Any] = {}
        for label in ["File", "Edit", "Image", "Process", "Analyze", "Plugins", "View", "Help"]:
            items = W.menu_open(label)
            all_menus[label] = items
            if label in ("File", "Image", "Plugins", "Help"):
                W.shot(f"menu_{label.lower()}")
            W.settle()
        W.fact("menus", all_menus)
        counts = {k: len([i for i in v if not i.get("sep")]) for k, v in all_menus.items()}
        subs = {k: [i["label"] for i in v if i.get("submenu")] for k, v in all_menus.items()}
        W.note("menu bar", f"top-level item counts {counts}; submenus {subs}")
        close = next((i for i in all_menus["File"] if i.get("label") == "Close"), None)
        W.note("File menu", f"'Close' item shows shortcut {close['kbd'] if close else '?'}; the global shortcut handler has no ⌘W binding, so in a browser tab this chord closes the tab")

    def sections() -> None:
        heads = W.page.locator(".section-head")
        titles = [h.evaluate("h => h.childNodes[1] ? h.childNodes[1].textContent : h.textContent") for h in heads.all()]
        open_now = [t for t, h in zip(titles, heads.all()) if "open" in (h.get_attribute("class") or "")]
        W.fact("sections", {"titles": titles, "open": open_now})
        W.note("inspector", f"{len(titles)} sections present for this image: {titles}; open by default: {open_now}")
        for t in titles:
            try:
                body = W.section(t, True)
                if body is not None:
                    W.shot(f"section_{re.sub(r'[^a-z0-9]+', '_', t.lower())}", locator=body.locator("xpath=.."))
                    hints = [h.inner_text()[:140] for h in body.locator(".hint-text, .hint").all()[:3]]
                    if hints:
                        W.fact(f"hint:{t}", hints)
            except Exception as e:
                W.note(f"inspector ▸ {t}", f"could not expand: {str(e).splitlines()[0][:120]}")
        insp = W.page.locator(".section").first.locator("xpath=..")
        sh, ch = insp.evaluate("e => [e.scrollHeight, e.clientHeight]")
        W.fact("inspector_scroll", {"scrollHeight": sh, "clientHeight": ch, "viewport": W.page.viewport_size})
        W.note("inspector", f"with every section expanded the inspector is {sh} px tall in a {ch} px column ({sh / max(ch, 1):.1f} screens)", shot=W.shot("inspector_all_open", full=False))
        for t in titles:
            if t not in open_now:
                W.section(t, False)

    def prefs() -> None:
        W.page.keyboard.press("Control+,")
        W.page.locator(".modal-backdrop").wait_for(timeout=5_000)
        groups = [g.inner_text()[:40] for g in W.page.locator(".prefs-group > h4, .prefs-group > .title, .prefs-group h4, .prefs-group strong").all()][:20]
        rows = W.page.locator(".prefs-row").count()
        W.fact("prefs", {"groups": groups, "rows": rows, "active": W.page.evaluate(JS_ACTIVE)})
        W.note("Preferences (⌘,)", f"{rows} rows in groups {groups}; focus after opening: {W.facts['prefs']['active']['region']}", shot=W.shot("preferences"))
        W.page.keyboard.press("Escape")

    def help_overlay() -> None:
        W.page.keyboard.press("?")
        ov = W.page.locator(".help-overlay")
        ov.wait_for(timeout=5_000)
        text = ov.inner_text()
        W.fact("help_overlay", {"chars": len(text), "has_cmdk": "⌘K" in text or "Ctrl+K" in text, "has_cmdw": "⌘W" in text or "Ctrl+W" in text, "head": text[:300]})
        W.note("Help overlay (?)", f"{len(text)} characters; lists ⌘K: {W.facts['help_overlay']['has_cmdk']}; lists ⌘W: {W.facts['help_overlay']['has_cmdw']}", shot=W.shot("help_overlay"))
        W.page.keyboard.press("Escape")

    W.probe("menus", menus)
    W.probe("sections", sections)
    W.probe("prefs", prefs)
    W.probe("help", help_overlay)


def empty_state(W: Walk) -> None:
    W.page.locator(".app").wait_for(timeout=30_000)
    W.page.wait_for_timeout(500)
    status = W.page.locator(".statusbar").inner_text()
    empty = W.page.locator(".empty-viewport").inner_text()[:400] if W.page.locator(".empty-viewport").count() else "(no .empty-viewport)"
    W.fact("empty_state", {"status": status, "viewport_text": empty})
    W.note("empty viewport", f"status bar: {status!r}; viewport text: {empty!r}", shot=W.shot("empty"))


# ------------------------------------------------------------------ persona A: power user
def walk_power(W: Walk) -> None:
    W.probe("empty", lambda: empty_state(W))

    def palette_queries() -> None:
        res = {}
        for q in ["filters.gaussian", "Subtract Background", "Gaussian Blur...", "bandpass", "enhance contrast", "duplicate", "crop", "bin", "set scale", "macro"]:
            info = W.palette_open(q)
            res[q] = {"count": info["count"], "selected": info["selected"], "top": [e["name"] for e in info["entries"][:4]], "empty": info["empty"]}
            if q == "Subtract Background":
                W.shot("palette_subtract_background")
            W.page.keyboard.press("Escape")
            W.page.locator(".palette").wait_for(state="hidden", timeout=5_000)
        W.fact("palette_queries", res)
        for q, r in res.items():
            W.note("palette", f"query {q!r}: {r['count']} matches, selected {r['selected']!r}, top {r['top']}")
        foot = W.page.evaluate("() => { const b = document.querySelector('.palette'); return b ? 1 : 0; }")
        W.fact("palette_closed", foot == 0)

    def recorder_and_editor() -> None:
        iid = W.open_sample(SAMPLE_BLOBS)
        W.palette_run("gauss", "Gaussian Blur")
        sh = W.sheet()
        info = W.sheet_info()
        W.note("Gaussian Blur sheet", f"fields {info['fields']}, buttons {info['buttons']}, preview switch {info['preview_switch']} (on: {info['preview_on']}), help button present: {'Help on this command (command reference)' in info['buttons']}", shot=W.shot("sheet_gaussian"))
        sh.locator('input[inputmode="decimal"], input[type="number"]').first.fill("2")
        # typing guard: a letter typed into the number field must not fire the single-letter command shortcut
        rows_before = len(W.state()["results"]["rows"])
        sh.locator('input[inputmode="decimal"], input[type="number"]').first.press("m")
        W.page.wait_for_timeout(600)
        rows_after = len(W.state()["results"]["rows"])
        W.note("sheet, sigma field", f"pressing 'm' while typing in the number field: results rows {rows_before} -> {rows_after} (unchanged = the typing guard works)")
        W.apply_sheet()
        wait_until(lambda: W.state()["history"].get(iid, {}).get("undo"), what="history")
        body = W.section("Script & Recorder")
        lang = body.locator('select[aria-label="Language"]')
        langs = [o.inner_text() for o in lang.locator("option").all()]
        btns = [b.get_attribute("title") or b.inner_text().strip() for b in body.locator("button").all()]
        rec_text = body.locator(".macro-lines").inner_text() if body.locator(".macro-lines").count() else ""
        editor = body.locator(".script-editor textarea")
        is_textarea = editor.evaluate("e => e.tagName + (e.getAttribute('aria-label') ? ' aria-label=' + e.getAttribute('aria-label') : '')") if editor.count() else None
        gutter = body.locator(".editor-gutter").count()
        W.fact("script_panel", {"languages": langs, "buttons": btns, "recorded_tail": rec_text[-300:], "editor_tag": is_textarea, "gutter": gutter})
        W.note("Script & Recorder", f"languages {langs}; buttons {btns}; editor element {is_textarea} with line-number gutter: {bool(gutter)}; recorded tail: {rec_text[-160:]!r}", shot=W.shot("script_panel", locator=body))
        # run an ImageJ macro line in the editor
        lang.select_option(label=next((l for l in langs if "macro" in l.lower() or "ImageJ" in l), langs[0]))
        n0 = len(W.state()["history"].get(iid, {}).get("undo", []))
        editor.fill('run("Gaussian Blur...", "sigma=2");')
        run_btn = body.locator("button", has_text=re.compile(r"^\s*Run")).first
        run_btn.click()
        try:
            wait_until(lambda: len(W.state()["history"].get(iid, {}).get("undo", [])) > n0, 30, what="macro to run")
            W.note("Script editor, ImageJ macro", f"run(\"Gaussian Blur...\", \"sigma=2\") executed: history {n0} -> {len(W.state()['history'][iid]['undo'])}")
        except AssertionError as e:
            W.note("Script editor, ImageJ macro", f"macro did not change the history: {e}", shot=W.shot("macro_fail"))
        out = body.locator(".console-out").inner_text()[-300:] if body.locator(".console-out").count() else ""
        W.fact("macro_output", out)
        # a failing Python line: how is the error shown?
        lang.select_option(label=next((l for l in langs if "python" in l.lower()), langs[0]))
        editor.fill('lumen.run("filters.nope", sigma=2)')
        run_btn.click()
        W.page.wait_for_timeout(1500)
        out2 = body.locator(".console-out").inner_text()[-400:] if body.locator(".console-out").count() else ""
        W.note("Script editor, Python error", f"console output after a wrong op id: {out2!r}", shot=W.shot("script_error", locator=body))
        # ⌘R inside the editor runs the script (and must not reload the page)
        editor.fill('print("hello from cmd-r")')
        editor.focus()
        W.page.keyboard.press("Control+r")
        W.page.wait_for_timeout(1200)
        out3 = body.locator(".console-out").inner_text()[-200:] if body.locator(".console-out").count() else ""
        W.note("Script editor, ⌘R", f"output {out3!r}; page still loaded: {W.page.locator('.app').count() == 1}")
        # repeat last (⇧R)
        W.page.locator(".canvas-wrap").click(position={"x": 5, "y": 5})
        n1 = len(W.state()["history"].get(iid, {}).get("undo", []))
        W.page.keyboard.press("Shift+R")
        try:
            wait_until(lambda: len(W.state()["history"].get(iid, {}).get("undo", [])) > n1, 20, what="repeat last")
            W.note("⇧R Repeat Last", f"repeated the last command: history {n1} -> {len(W.state()['history'][iid]['undo'])} ({W.state()['history'][iid]['undo'][-1]})")
        except AssertionError:
            W.note("⇧R Repeat Last", "did not add a history entry (the last command may have been the script run)", shot=W.shot("repeat_last"))
        hist = W.section("History")
        items = [h.inner_text()[:40] for h in hist.locator(".hist-item").all()]
        W.note("History section", f"entries {items}", shot=W.shot("history", locator=hist))
        # display changes recorded?
        disp = W.section("Display")
        body = W.section("Script & Recorder")
        rec_before = body.locator(".macro-lines").inner_text()
        disp = W.section("Display")
        disp.locator('select[title="Lookup table"]').first.select_option(label="Fire")
        W.page.wait_for_timeout(500)
        rec_after = W.section("Script & Recorder").locator(".macro-lines").inner_text()
        W.fact("lut_recorded", rec_after != rec_before)
        W.note("Display ▸ LUT dropdown + recorder", f"after changing the LUT to Fire the recorder text {'mentions' if 'lut' in rec_after.lower() or 'Fire' in rec_after else 'does not mention'} the LUT (display state is not recorded)")

    def custom_shortcuts() -> None:
        items = W.menu_open("Plugins")
        sc = [i for i in items if i.get("label", "").startswith("Shortcuts")]
        W.note("Plugins menu", f"Shortcuts submenu present: {bool(sc)}")
        W.settle()
        info = W.palette_open("shortcut")
        W.note("palette", f"'shortcut' -> {[e['name'] for e in info['entries'][:5]]}")
        W.page.keyboard.press("Escape")
        try:
            W.menu_click(["Plugins", "Shortcuts", "Add Shortcut"])
            W.page.locator(".modal-backdrop, .sheet").first.wait_for(timeout=5_000)
            a = W.page.evaluate(JS_ACTIVE)
            W.page.keyboard.press("Escape")
            W.page.wait_for_timeout(250)
            still = W.page.locator(".modal-backdrop").count() > 0
            W.note("Plugins ▸ Shortcuts ▸ Add Shortcut", f"dialog opened with focus on {a['tag']}/{a['name'][:30]!r}; Esc closes it from there: {not still}", shot=W.shot("shortcuts_dialog"))
        except Exception as e:
            W.note("Plugins ▸ Shortcuts", f"Add Shortcut not reachable this way: {str(e).splitlines()[0][:120]}", shot=W.shot("shortcuts_menu"))

    def stack_scope_and_batch() -> None:
        iid = W.open_sample(SAMPLE_3D)
        W.palette_run("gauss", "Gaussian Blur")
        info = W.sheet_info()
        W.note("Gaussian Blur sheet on a z-stack", f"'Apply to' scope select present: {info['scope_select']} with options {info['scope_options']}; preview on: {info['preview_on']}", shot=W.shot("sheet_gaussian_stack"))
        W.settle()
        W.palette_run("z proj", "Z Project")
        info = W.sheet_info()
        W.note("Z Project sheet", f"fields {info['fields']}; preview switch: {info['preview_switch']}; advanced: {info['advanced']}", shot=W.shot("sheet_zproject"))
        W.settle()
        info = W.palette_open("process folder")
        W.note("palette", f"'process folder' -> {[e['name'] for e in info['entries'][:5]]}")
        W.page.keyboard.press("Enter")
        W.sheet()
        info = W.sheet_info()
        browse = W.page.locator(".sheet button", has_text=re.compile("Browse|…|Choose")).count()
        W.note("batch sheet", f"title {info['title']!r}; fields {info['fields']}; browse buttons: {browse}; preview: {info['preview_switch']}", shot=W.shot("sheet_batch"))
        W.settle()
        info = W.palette_open("run script")
        W.note("palette", f"'run script' -> {[e['name'] for e in info['entries'][:5]]}")
        W.page.keyboard.press("Escape")

    def api_docs() -> None:
        codes = {}
        for p in ["/docs", "/openapi.json", "/api/meta"]:
            try:
                urlrequest.urlopen(W.url + p, timeout=10)
                codes[p] = 200
            except Exception as e:
                codes[p] = str(e)[:40]
        help_items = [i.get("label") for i in W.menu_open("Help") if not i.get("sep")]
        W.settle()
        W.note("Help menu / REST", f"Help items {help_items}; HTTP {codes}; the Help menu {'does' if any('API' in (h or '') for h in help_items) else 'does not'} mention the REST API")
        try:
            W.menu_click(["Help", "Scripting"])
            W.page.locator(".modal-backdrop").wait_for(timeout=5_000)
            W.note("Help ▸ Scripting", "in-app docs modal opened", shot=W.shot("docs_scripting"))
        except Exception:
            try:
                W.settle()
                W.menu_click(["Help", "Command Reference"])
                W.page.locator(".modal-backdrop").wait_for(timeout=5_000)
                W.note("Help ▸ Command Reference", "in-app docs modal opened", shot=W.shot("docs_commands"))
            except Exception as e:
                W.note("Help menu docs", f"could not open a docs entry: {str(e).splitlines()[0][:100]}")

    def tracks_and_classifier() -> None:
        iid = W.open_sample(SAMPLE_TL)
        W.palette_run("track", "Track (detect + link)")
        info = W.sheet_info()
        W.note("Track (detect + link) sheet", f"fields {info['fields'][:14]}; preview switch: {info['preview_switch']}; advanced: {info['advanced']}", shot=W.shot("sheet_track"))
        W.apply_sheet()
        body = wait_until(lambda: W.section("Tracks"), 30, what="Tracks section")
        W.page.wait_for_timeout(800)
        btns = [b.get_attribute("title") or b.inner_text().strip() for b in body.locator("button").all()][:16]
        W.note("Tracks section", f"buttons {btns}", shot=W.shot("tracks_panel", locator=body))
        W.fact("tracks_recorded", "tracking.track" in W.section("Script & Recorder").locator(".macro-lines").inner_text())
        W.note("Recorder after tracking", f"tracking.track line recorded: {W.facts['tracks_recorded']}")
        cls = W.section("Pixel Classifier")
        btns = [b.get_attribute("title") or b.inner_text().strip() for b in cls.locator("button").all()][:20]
        W.note("Pixel Classifier section", f"buttons {btns}", shot=W.shot("classifier_panel", locator=cls))

    def split_and_3d() -> None:
        W.open_sample(SAMPLE_COINS)
        W.page.keyboard.press("Control+Shift+Backslash")
        W.page.locator(".split-view").wait_for(timeout=10_000)
        names = [s.evaluate("e => e.options[e.selectedIndex].text") for s in W.page.locator(".split-pane select").all()]
        W.note("Split view (⌘⇧\\)", f"two panes {names}", shot=W.shot("split_view"))
        W.page.keyboard.press("Control+Shift+Backslash")
        if not any((W.image(i) or {"shape": {"z": 1}})["shape"]["z"] > 1 for i in W.opened):
            W.open_sample(SAMPLE_3D)
        if W.section_head("3-D View").count() == 0:
            W.page.locator(".tabs .tab", has_text=re.compile("cells3d")).first.click()
            W.page.wait_for_timeout(500)
        vol = W.section("3-D View")
        W.page.wait_for_timeout(2500)
        ctl = [b.get_attribute("title") or b.inner_text().strip() for b in vol.locator("button, select").all()][:16]
        auto = vol.locator('[title="Auto-rotate"]').first
        auto_state = auto.evaluate("e => e.getAttribute('aria-pressed') || (e.checked !== undefined ? String(e.checked) : e.className)") if auto.count() else "n/a"
        W.note("3-D View section", f"controls {ctl}; auto-rotate state {auto_state}", shot=W.shot("volume_section", locator=vol))
        big = vol.locator('[title="Open in a large view"]')
        if big.count():
            big.first.click()
            W.page.wait_for_timeout(2000)
            W.note("3-D large view", "opened", shot=W.shot("volume_large"))
            W.page.keyboard.press("Escape")

    W.probe("palette_queries", palette_queries)
    W.probe("recorder_and_editor", recorder_and_editor)
    W.probe("custom_shortcuts", custom_shortcuts)
    W.probe("stack_scope_and_batch", stack_scope_and_batch)
    W.probe("api_docs", api_docs)
    W.probe("tracks_and_classifier", tracks_and_classifier)
    W.probe("split_and_3d", split_and_3d)
    common_tour(W)


# ------------------------------------------------------------------ persona B: occasional biologist
def walk_biologist(W: Walk) -> None:
    W.probe("empty", lambda: empty_state(W))

    def open_via_menu() -> None:
        before = {i["id"] for i in W.state()["images"]}
        W.menu_open("File")
        W.shot("menu_file_open")
        W.settle()
        W.menu_click(["File", "Open Sample", SAMPLE_3CH])
        new = wait_until(lambda: [i["id"] for i in W.state()["images"] if i["id"] not in before], 60, what="sample via menu")
        W.opened.append(new[0])
        W.page.locator(".statusbar").get_by_text(re.compile(r"\d+×\d+ px")).wait_for(timeout=30_000)
        W.page.wait_for_timeout(500)
        tabs = [t.get_attribute("title") or t.inner_text() for t in W.page.locator(".tabs .tab").all()]
        W.note("File ▸ Open Sample ▸ …", f"image opened from the menu; status {W.page.locator('.statusbar').inner_text()[:120]!r}; tab tooltip {tabs[-1][:100]!r}", shot=W.shot("after_open_menu"))

    def adjust_via_menu() -> None:
        items = W.menu_open("Image")
        adjust = next((i for i in items if i.get("label", "").startswith("Adjust")), None)
        W.settle()
        W.menu_open("Image")
        it = W.page.locator(".menu-popover:not(.submenu) .menu-item", has=W.page.locator(":scope > .label", has_text=re.compile("^Adjust"))).first
        it.hover()
        sub = it.locator(".menu-popover.submenu")
        sub.wait_for(timeout=5_000)
        sub_items = [x.inner_text().replace("\n", " ")[:40] for x in sub.locator(".menu-item").all()]
        W.note("Image ▸ Adjust", f"submenu entries {sub_items}", shot=W.shot("menu_image_adjust"))
        W.settle()
        has_bc = any("Brightness" in x for x in sub_items)
        W.note("Image ▸ Adjust", f"contains a Brightness/Contrast entry: {has_bc}")
        W.settle()
        info = W.palette_open("brightness")
        W.note("palette", f"'brightness' -> {[e['name'] for e in info['entries'][:5]]}")
        W.page.keyboard.press("Escape")
        W.section("Display", False)
        W.page.keyboard.press("Control+Shift+c")
        W.page.wait_for_timeout(500)
        disp_open = "open" in (W.section_head("Display").get_attribute("class") or "")
        toast = " | ".join(W.page.locator(".toast").all_inner_texts())[:160]
        sheet = W.sheet_info()
        W.note("⌘⇧C (Fiji's Brightness/Contrast chord)", f"Display section open: {disp_open}; sheet: {sheet.get('title') if sheet.get('present') else 'none'}; toast: {toast!r}", shot=W.shot("after_bc_menu"))
        W.settle()
        disp = W.section("Display")
        labels = [t for t in disp.locator("label, .k, .row-label, span[title]").all_inner_texts()][:24]
        hint = [h.inner_text()[:160] for h in disp.locator(".hint-text, .hint").all()][:2]
        W.note("Display section", f"visible labels {labels}; hint {hint}", shot=W.shot("display_panel", locator=disp))
        # is a display change undoable through Edit ▸ Undo?
        undo_before = W.page.locator('.topbar button[title^="Undo"]').is_enabled()
        disp.locator('button[title^="Auto"]').first.click()
        W.page.wait_for_timeout(400)
        undo_after = W.page.locator('.topbar button[title^="Undo"]').is_enabled()
        W.note("Display ▸ Auto, then Undo button", f"Undo enabled before {undo_before} / after a display change {undo_after} (display changes are not undo steps; 'Reset' is the way back)")

    def threshold_particles() -> None:
        thr = W.section("Threshold")
        part = thr.locator("button", has_text="Particles")
        W.note("Threshold section", f"'Particles…' enabled before a threshold exists: {part.first.is_enabled()}; buttons {[b.inner_text().strip() or b.get_attribute('title') for b in thr.locator('button').all()][:10]}", shot=W.shot("threshold_panel", locator=thr))
        W.menu_click(["Analyze", "Analyze Particles"])
        sh = W.sheet()
        info = W.sheet_info()
        W.note("Analyze ▸ Analyze Particles… without a threshold", f"sheet opened with button {[b for b in info['buttons'] if b in ('Run', 'Apply')]}; error before running: {info['error']!r}", shot=W.shot("particles_no_threshold"))
        W.page.locator(".sheet .sheet-foot button.primary").click()
        W.page.wait_for_timeout(2500)
        err = W.page.locator(".sheet-error").inner_text() if W.page.locator(".sheet-error").count() else ""
        toast = " | ".join(W.page.locator(".toast").all_inner_texts())[:200]
        W.note("Analyze Particles ▸ Run without a threshold", f"sheet error {err!r}; toast {toast!r}; sheet still open: {W.page.locator('.sheet').count() > 0}", shot=W.shot("particles_error"))
        W.settle()
        thr = W.section("Threshold")
        thr.locator("button", has_text=re.compile(r"^\s*Auto\s*$")).click()
        W.page.wait_for_timeout(600)
        thr.locator("button", has_text="Particles").click()
        info = W.sheet_info()
        W.note("Analyze Particles sheet", f"fields {info['fields']}; buttons {info['buttons']}; description {info['description']!r}", shot=W.shot("particles_sheet"))
        W.apply_sheet()
        n = wait_until(lambda: len(W.state()["rois"]), 60, what="ROIs")
        W.page.wait_for_timeout(800)
        W.note("after Analyze Particles", f"{n} ROIs; sections with badges: {[h.inner_text()[:30] for h in W.page.locator('.section-head', has=W.page.locator('.badge')).all()]}", shot=W.shot("after_particles"))

    def draw_and_measure() -> None:
        tool_titles = [b.get_attribute("title") for b in W.page.locator(".rail button").all()]
        pressed = [b.get_attribute("title") for b in W.page.locator('.rail button[aria-pressed="true"]').all()]
        W.note("tool rail", f"tools {tool_titles}; active by default {pressed}", shot=W.shot("tool_rail", locator=W.page.locator(".rail")))
        canvas = W.page.locator(".canvas-wrap")
        box = canvas.bounding_box()
        W.page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.3)
        W.page.mouse.down()
        W.page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5, steps=8)
        W.page.mouse.up()
        W.page.wait_for_timeout(300)
        status = W.page.locator(".statusbar").inner_text()
        W.note("canvas drag with the default tool", f"status bar after the drag: {status[:160]!r} (no selection pill = the drag panned or did nothing)", shot=W.shot("drag_default_tool"))
        W.page.locator('.rail button[title="Rectangle (1)"]').click()
        W.page.mouse.move(box["x"] + box["width"] * 0.3, box["y"] + box["height"] * 0.3)
        W.page.mouse.down()
        W.page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5, steps=8)
        W.page.mouse.up()
        W.page.locator(".statusbar .pill", has_text="rect").wait_for(timeout=10_000)
        rows0 = len(W.state()["results"]["rows"])
        W.menu_click(["Analyze", "Measure"])
        wait_until(lambda: len(W.state()["results"]["rows"]) > rows0, 20, what="a results row")
        W.note("Analyze ▸ Measure", "one row added to Results", shot=W.shot("after_measure"))

    def filter_undo_close() -> None:
        iid = W.state()["active"]
        W.menu_click(["Process", "Filters", "Gaussian Blur"])
        sh = W.sheet()
        info = W.sheet_info()
        W.note("Process ▸ Filters ▸ Gaussian Blur…", f"fields {info['fields']}; buttons {info['buttons']}; preview on by default: {info['preview_on']}", shot=W.shot("gaussian_sheet_menu"))
        sh.locator('input[inputmode="decimal"], input[type="number"]').first.fill("3")
        W.page.wait_for_timeout(800)
        hud = [c.inner_text() for c in W.page.locator(".viewport-hud .hud-chip, .hud-chip").all()]
        W.note("viewport HUD during preview", f"chips {hud}", shot=W.shot("preview_hud"))
        W.apply_sheet()
        wait_until(lambda: W.state()["history"].get(iid, {}).get("undo"), what="history")
        W.menu_click(["Edit", "Undo"])
        wait_until(lambda: not W.state()["history"].get(iid, {}).get("undo"), 20, what="undo")
        W.note("Edit ▸ Undo", "undo from the menu worked")
        W.menu_click(["Edit", "Redo"])
        wait_until(lambda: W.state()["history"].get(iid, {}).get("undo"), 20, what="redo")
        tab = W.page.locator(".tabs .tab", has_text=re.compile("synthetic")).first
        W.note("image tab", f"tab title after a modification: {(tab.get_attribute('title') or '')[:160]!r}; tab text {tab.inner_text()!r}")
        # Close a modified image from the menu: is there a confirmation?
        W.menu_click(["File", "Close"])
        W.page.wait_for_timeout(800)
        dialog = W.page.locator(".modal-backdrop, [role=alertdialog], .confirm").count()
        still = W.image(iid) is not None
        toast = " | ".join(W.page.locator(".toast").all_inner_texts())[:160]
        W.note("File ▸ Close on a modified image", f"confirmation dialog: {dialog > 0}; image still open: {still}; toast {toast!r}", shot=W.shot("close_modified"))
        if not still and iid in W.opened:
            W.opened.remove(iid)
        W.settle()

    def save_as_dialog() -> None:
        if not W.state()["images"] or W.state().get("active") not in W.opened:
            W.open_sample(SAMPLE_BLOBS)
        W.menu_click(["File", "Save As"])
        W.page.locator(".modal-backdrop").wait_for(timeout=5_000)
        text = W.page.locator(".modal-backdrop").inner_text()[:300]
        quick = [q.inner_text()[:20] for q in W.page.locator(".quick-links button, .quick-links a").all()][:8]
        W.note("File ▸ Save As…", f"server-side file dialog; quick links {quick}; text head {text[:140]!r}", shot=W.shot("save_as_dialog"))
        W.page.keyboard.press("Escape")

    def workflows_help() -> None:
        wf = W.section("Workflows")
        W.page.wait_for_timeout(600)
        cards = [c.inner_text().split("\n")[0][:40] for c in wf.locator(".wf-card").all()]
        W.note("Workflows section", f"{len(cards)} guided workflows: {cards}", shot=W.shot("workflows", locator=wf))
        if wf.locator(".wf-card").count():
            wf.locator(".wf-card").first.click()
            W.page.wait_for_timeout(500)
            steps = [s.inner_text()[:60] for s in wf.locator(".wf-step-head").all()][:6]
            W.note("Workflows ▸ first card", f"steps {steps}", shot=W.shot("workflow_open", locator=wf))
        help_items = [i.get("label") for i in W.menu_open("Help") if not i.get("sep")]
        W.settle()
        W.note("Help menu", f"items {help_items}")
        try:
            W.menu_click(["Help", "Lumen README"])
            W.page.locator(".modal-backdrop").wait_for(timeout=5_000)
            W.note("Help ▸ README", "docs modal", shot=W.shot("docs_readme"))
        except Exception as e:
            W.note("Help ▸ README", f"not opened: {str(e).splitlines()[0][:100]}")

    def timelapse_tracks() -> None:
        iid = W.open_sample(SAMPLE_TL)
        items = W.menu_open("Plugins")
        W.settle()
        W.menu_open("Plugins")
        it = W.page.locator(".menu-popover:not(.submenu) .menu-item", has=W.page.locator(":scope > .label", has_text=re.compile("^Tracking"))).first
        it.hover()
        sub = it.locator(".menu-popover.submenu")
        sub.wait_for(timeout=5_000)
        entries = [x.inner_text().replace("\n", " ")[:50] for x in sub.locator(".menu-item").all()]
        W.note("Plugins ▸ Tracking", f"entries {entries}", shot=W.shot("menu_tracking"))
        W.settle()
        tr = W.section("Tracks")
        btns = [b.get_attribute("title") or b.inner_text().strip() for b in tr.locator("button").all()][:12]
        W.note("Tracks section before tracking", f"buttons {btns}", shot=W.shot("tracks_empty", locator=tr))
        wiz = tr.locator('button[title^="TrackMate wizard"]')
        if wiz.count():
            wiz.first.click()
            W.page.wait_for_timeout(800)
            W.note("Tracks ▸ TrackMate wizard", f"step text {W.page.locator('.tw-form, .sheet, .modal-backdrop').first.inner_text()[:200]!r}", shot=W.shot("track_wizard"))
            W.settle()

    def stack_3d() -> None:
        W.open_sample(SAMPLE_3D)
        dim = [c.inner_text() for c in W.page.locator(".dimbar .chip").all()][:6]
        W.note("dimension bar", f"chips {dim}")
        vol = W.section("3-D View")
        W.page.wait_for_timeout(2500)
        labels = [l.inner_text()[:30] for l in vol.locator("label, .k").all()][:14]
        W.note("3-D View section", f"labels {labels}", shot=W.shot("volume_section", locator=vol))
        ortho = W.section("Orthogonal Views")
        W.page.wait_for_timeout(800)
        W.note("Orthogonal Views section", "opened", shot=W.shot("ortho_section", locator=ortho))
        cls = W.section("Pixel Classifier")
        txt = cls.inner_text()[:400]
        W.note("Pixel Classifier section", f"text head {txt[:200]!r}", shot=W.shot("classifier_panel", locator=cls))

    W.probe("open_via_menu", open_via_menu)
    W.probe("adjust_via_menu", adjust_via_menu)
    W.probe("threshold_particles", threshold_particles)
    W.probe("draw_and_measure", draw_and_measure)
    W.probe("filter_undo_close", filter_undo_close)
    W.probe("save_as_dialog", save_as_dialog)
    W.probe("workflows_help", workflows_help)
    W.probe("timelapse_tracks", timelapse_tracks)
    W.probe("stack_3d", stack_3d)
    common_tour(W)


# ------------------------------------------------------------------ persona C: accessibility / HCI reviewer
CONTRAST_SELECTORS = [
    ".statusbar", ".statusbar .pill", ".section-head", ".section-head .badge", ".hint-text", ".menu-trigger", ".tab", ".tab.active",
    ".btn", ".btn.primary", "button.primary", ".icon-btn", ".kbd", ".k", ".v", ".dimbar .chip", ".hud-chip", ".empty-viewport",
    ".palette input::placeholder", ".pi-desc", ".pi-cat", ".palette-foot", ".palette-group", ".sheet h3", ".sheet label", ".sheet-desc", ".advanced-toggle",
    ".sheet-foot .switch", "input[type=number]", "select", ".toast", ".help-overlay h2", ".help-overlay .kbd", ".results td", ".roi-item", ".hist-item", ".console-out",
]


def walk_a11y(W: Walk) -> None:
    W.probe("empty", lambda: empty_state(W))

    def landmarks() -> None:
        lm = W.page.evaluate(JS_LANDMARKS)
        W.fact("landmarks_empty", lm)
        W.note("document", f"lang={lm['lang']!r} title={lm['title']!r}; landmarks {lm['landmarks']}; headings {lm['headings'][:8]}; skip link: {lm['skipLink']}; reduced-motion rule in CSS: {lm['reducedMotionRule']}")

    def tab_walk() -> None:
        W.page.locator("body").click(position={"x": 2, "y": 2})
        W.page.evaluate("() => { if (document.activeElement) document.activeElement.blur(); }")
        stops = []
        for i in range(70):
            W.page.keyboard.press("Tab")
            stops.append(W.page.evaluate(JS_ACTIVE))
            if i == 6:
                W.shot("focus_ring_topbar")
        W.fact("tab_walk_empty", stops)
        regions = [s["region"] for s in stops]
        order = []
        for r in regions:
            if not order or order[-1] != r:
                order.append(r)
        W.note("keyboard, Tab from the top (no image)", f"70 Tab stops pass through regions {order}; stops without a visible outline: {sum(1 for s in stops if s['outline'] == 'none')}; first 12: {[(s['region'], s['name'][:18]) for s in stops[:12]]}")

    def menubar_keyboard() -> None:
        W.page.locator(".menubar .menu-trigger").first.focus()
        W.page.keyboard.press("Enter")
        W.page.wait_for_timeout(300)
        opened = W.page.locator(".menu-popover:visible").count() > 0
        W.page.keyboard.press("ArrowDown")
        W.page.wait_for_timeout(200)
        a1 = W.page.evaluate(JS_ACTIVE)
        W.page.keyboard.press("ArrowRight")
        W.page.wait_for_timeout(200)
        a2 = W.page.evaluate(JS_ACTIVE)
        items_focusable = W.page.evaluate("() => Array.from(document.querySelectorAll('.menu-item')).filter(e => e.tabIndex >= 0).length")
        W.note("menu bar, keyboard", f"Enter on the focused 'File' trigger opens the menu: {opened}; after ArrowDown focus is on {a1['region']}/{a1['name'][:20]!r} (role {a1['role']}); after ArrowRight {a2['region']}/{a2['name'][:20]!r}; focusable menu items: {items_focusable}", shot=W.shot("menu_keyboard"))
        W.settle()

    def palette_aria() -> None:
        W.page.keyboard.press("Control+k")
        W.page.locator(".palette input").wait_for(timeout=10_000)
        W.page.locator(".palette input").fill("thres")
        W.page.wait_for_timeout(300)
        attrs = W.page.evaluate("""() => { const p = document.querySelector('.palette'); const i = p.querySelector('input'); const sel = p.querySelector('.palette-item.sel');
            return {palette_role: p.getAttribute('role'), modal: p.getAttribute('aria-modal'), input_role: i.getAttribute('role'), input_aria: Array.from(i.attributes).filter(a => a.name.startsWith('aria-')).map(a => a.name + '=' + a.value),
                    list_role: p.querySelector('.palette-list').getAttribute('role'), item_role: sel && sel.getAttribute('role'), item_selected: sel && sel.getAttribute('aria-selected'), items: p.querySelectorAll('.palette-item').length,
                    backdrop_inert: !!document.querySelector('.app[inert], .app[aria-hidden="true"]')}; }""")
        W.page.keyboard.press("ArrowDown")
        W.page.wait_for_timeout(150)
        after = W.page.evaluate("() => { const i = document.querySelector('.palette input'); return {active: document.activeElement === i, activedescendant: i.getAttribute('aria-activedescendant'), sel: document.querySelector('.palette-item.sel .pi-name')?.textContent}; }")
        W.fact("palette_aria", {**attrs, **after})
        W.note("command palette (⌘K)", f"roles: palette={attrs['palette_role']} input={attrs['input_role']} list={attrs['list_role']} item={attrs['item_role']} aria-selected={attrs['item_selected']}; input aria attrs {attrs['input_aria']}; after ArrowDown the input keeps focus ({after['active']}) and aria-activedescendant={after['activedescendant']}; background inert: {attrs['backdrop_inert']}", shot=W.shot("palette_aria"))
        W.page.keyboard.press("Escape")
        W.page.wait_for_timeout(200)
        back = W.page.evaluate(JS_ACTIVE)
        W.note("command palette, Esc", f"focus after closing: {back['region']}/{back['tag']}")

    def image_and_sliders() -> None:
        iid = W.open_sample(SAMPLE_3CH)
        lm = W.page.evaluate(JS_LANDMARKS)
        W.fact("landmarks_image", lm)
        W.note("document with an image", f"landmarks {lm['landmarks']}; canvases {lm['canvases'][:3]}; live regions {lm['live']}; section heads: tag={lm['sectionHeads'][0]['tag'] if lm['sectionHeads'] else None} role={lm['sectionHeads'][0]['role'] if lm['sectionHeads'] else None} tabindex={lm['sectionHeads'][0]['tabindex'] if lm['sectionHeads'] else None} aria-expanded={lm['sectionHeads'][0]['expanded'] if lm['sectionHeads'] else None}; image tabs: {lm['tabs'][:2]}; animated now: {lm['animated']}")
        sliders = W.page.evaluate("""() => Array.from(document.querySelectorAll('[role=slider]')).slice(0, 4).map(s => ({label: s.getAttribute('aria-label'), now: s.getAttribute('aria-valuenow'), text: s.getAttribute('aria-valuetext'), min: s.getAttribute('aria-valuemin'), max: s.getAttribute('aria-valuemax'), tabindex: s.getAttribute('tabindex')}))""")
        W.note("Display section sliders", f"first sliders {sliders}")
        if sliders:
            h = W.page.locator('.range-slider [role="slider"]').first
            h.focus()
            v0 = h.get_attribute("aria-valuenow")
            W.page.keyboard.press("ArrowRight")
            W.page.wait_for_timeout(200)
            v1 = h.get_attribute("aria-valuenow")
            W.note("slider keyboard", f"ArrowRight: aria-valuenow {v0} -> {v1}", shot=W.shot("slider_focus", locator=W.section("Display")))
        # keyboard reachability of the inspector section heads
        W.page.evaluate("() => document.activeElement && document.activeElement.blur()")
        W.page.locator(".rail button").first.focus()
        reached = set()
        for _ in range(140):
            W.page.keyboard.press("Tab")
            a = W.page.evaluate(JS_ACTIVE)
            reached.add((a["region"], a["cls"][:14]))
            if a["region"] == "body":
                break
        heads = any(c.startswith("section-head") for _, c in reached)
        tabs_reached = any(c.startswith("tab") and r == "tabs" for r, c in reached)
        W.fact("tab_walk_image_regions", sorted(reached))
        W.note("keyboard, Tab with an image open", f"section heads reachable by Tab: {heads}; image tabs reachable: {tabs_reached}; regions reached {sorted({r for r, _ in reached})}")
        unl = W.page.evaluate(JS_UNLABELED)
        W.fact("unlabeled_image", unl)
        W.note("unlabeled controls (image open, default sections)", f"{len(unl)}: {[(u['where'], u['tag'], u['cls'][:20]) for u in unl[:12]]}")
        tg = W.page.evaluate(JS_TARGETS)
        small = [t for t in tg if t["w"] < 24 or t["h"] < 24]
        W.fact("targets_image", {"total": len(tg), "small": small[:40]})
        W.note("target sizes (image open)", f"{len(small)} of {len(tg)} interactive elements are smaller than 24×24 CSS px: {[(t['cls'][:16] or t['tag'], t['name'][:16], t['w'], t['h']) for t in small[:14]]}")

    def contrast_both_themes() -> None:
        W.page.keyboard.press("Control+k")
        W.page.locator(".palette input").wait_for(timeout=5_000)
        W.page.locator(".palette input").fill("thres")
        W.page.wait_for_timeout(200)
        res_pal = W.page.evaluate(JS_CONTRAST, [s for s in CONTRAST_SELECTORS if s.startswith(".palette") or s.startswith(".pi-")])
        W.page.keyboard.press("Escape")
        W.palette_run("gauss", "Gaussian Blur")
        W.sheet()
        res_sheet = W.page.evaluate(JS_CONTRAST, [s for s in CONTRAST_SELECTORS if s.startswith(".sheet") or s.startswith(".advanced") or s.startswith("input") or s.startswith("select")])
        W.shot("sheet_dark")
        W.settle()
        theme = W.page.evaluate("() => document.documentElement.getAttribute('data-theme') || document.body.className")
        res_dark = W.page.evaluate(JS_CONTRAST, [s for s in CONTRAST_SELECTORS if not (s.startswith(".palette") or s.startswith(".pi-") or s.startswith(".sheet") or s.startswith(".advanced") or s.startswith("input") or s.startswith("select") or s.startswith(".help"))])
        W.page.keyboard.press("?")
        W.page.locator(".help-overlay").wait_for(timeout=5_000)
        res_help = W.page.evaluate(JS_CONTRAST, [s for s in CONTRAST_SELECTORS if s.startswith(".help")])
        W.page.keyboard.press("Escape")
        dark = res_dark + res_pal + res_sheet + res_help
        W.page.locator('button[title="Toggle theme"]').click()
        W.page.wait_for_timeout(400)
        W.shot("light_theme")
        res_light = W.page.evaluate(JS_CONTRAST, [s for s in CONTRAST_SELECTORS if not (s.startswith(".palette") or s.startswith(".pi-") or s.startswith(".sheet") or s.startswith(".advanced") or s.startswith("input") or s.startswith("select") or s.startswith(".help"))])
        W.page.keyboard.press("Control+k")
        W.page.locator(".palette input").wait_for(timeout=5_000)
        W.page.locator(".palette input").fill("thres")
        W.page.wait_for_timeout(200)
        res_light += W.page.evaluate(JS_CONTRAST, [s for s in CONTRAST_SELECTORS if s.startswith(".palette") or s.startswith(".pi-")])
        W.shot("palette_light")
        W.page.keyboard.press("Escape")
        W.palette_run("gauss", "Gaussian Blur")
        W.sheet()
        res_light += W.page.evaluate(JS_CONTRAST, [s for s in CONTRAST_SELECTORS if s.startswith(".sheet") or s.startswith(".advanced") or s.startswith("input") or s.startswith("select")])
        W.shot("sheet_light")
        W.settle()
        W.page.locator('button[title="Toggle theme"]').click()
        W.page.wait_for_timeout(300)
        W.fact("contrast", {"dark_theme_attr": theme, "dark": dark, "light": res_light})
        for name, res in (("dark", dark), ("light", res_light)):
            fails = [r for r in res if not r.get("missing") and not r["passAA"]]
            W.note(f"contrast, {name} theme", f"{len(fails)} of {len([r for r in res if not r.get('missing')])} sampled text styles below WCAG AA (4.5:1, 3:1 large): {[(r['sel'], r['ratio'], r['px']) for r in fails]}")

    def dialogs_focus() -> None:
        W.palette_run("gauss", "Gaussian Blur")
        W.sheet()
        a = W.page.evaluate(JS_ACTIVE)
        attrs = W.page.evaluate("() => { const s = document.querySelector('.sheet'); return {role: s.getAttribute('role'), labelledby: s.getAttribute('aria-labelledby'), modal: s.getAttribute('aria-modal'), title_id: !!document.getElementById(s.getAttribute('aria-labelledby') || '')}; }")
        W.note("parameter sheet", f"attrs {attrs}; focus after opening: {a['region']}/{a['tag']}/{a['name'][:20]!r}")
        W.page.keyboard.press("Escape")
        W.page.wait_for_timeout(200)
        b = W.page.evaluate(JS_ACTIVE)
        W.note("parameter sheet, Esc", f"focus after closing: {b['region']}/{b['tag']}")
        W.page.keyboard.press("?")
        W.page.locator(".help-overlay").wait_for(timeout=5_000)
        escaped = False
        for _ in range(40):
            W.page.keyboard.press("Tab")
            a = W.page.evaluate(JS_ACTIVE)
            if a["region"] not in ("help",):
                escaped = True
                break
        W.note("help overlay (role=dialog aria-modal)", f"focus leaves the overlay within 40 Tabs: {escaped} (to {a['region']}/{a['name'][:20]!r})", shot=W.shot("help_focus"))
        W.page.keyboard.press("Escape")
        W.page.keyboard.press("Control+,")
        W.page.locator(".modal-backdrop").wait_for(timeout=5_000)
        attrs = W.page.evaluate("() => { const d = document.querySelector('.modal-backdrop > div, .modal-backdrop .sheet, .modal-backdrop [role=dialog]') || document.querySelector('.modal-backdrop'); return {role: d.getAttribute('role'), modal: d.getAttribute('aria-modal'), labelledby: d.getAttribute('aria-labelledby'), tabindex: d.getAttribute('tabindex')}; }")
        a = W.page.evaluate(JS_ACTIVE)
        unl = W.page.evaluate(JS_UNLABELED)
        W.note("Preferences dialog", f"attrs {attrs}; focus on open: {a['region']}/{a['tag']}; unlabeled controls inside modals: {[(u['tag'], u['cls'][:16]) for u in unl if u['where'] == 'modal'][:8]}", shot=W.shot("prefs_a11y"))
        W.page.keyboard.press("Escape")
        W.page.keyboard.press("Control+o")
        W.page.locator(".modal-backdrop").wait_for(timeout=5_000)
        attrs = W.page.evaluate("() => { const d = document.querySelector('.modal-backdrop [role=dialog]') || document.querySelector('.modal-backdrop > div'); return d ? {role: d.getAttribute('role'), modal: d.getAttribute('aria-modal'), labelledby: d.getAttribute('aria-labelledby')} : null; }")
        a = W.page.evaluate(JS_ACTIVE)
        files_focusable = W.page.evaluate("() => Array.from(document.querySelectorAll('.file-item')).filter(e => e.tabIndex >= 0).length")
        W.note("File dialog (⌘O)", f"attrs {attrs}; focus on open: {a['region']}/{a['tag']}/{a['name'][:20]!r}; focusable file items: {files_focusable}", shot=W.shot("file_dialog_a11y"))
        W.page.keyboard.press("Escape")

    def toolbar_and_tabs() -> None:
        rail = W.page.locator('.rail[role="toolbar"]')
        W.page.locator(".rail button").first.focus()
        W.page.keyboard.press("ArrowDown")
        W.page.wait_for_timeout(150)
        a = W.page.evaluate(JS_ACTIVE)
        W.page.keyboard.press("Tab")
        b = W.page.evaluate(JS_ACTIVE)
        W.note("tool rail (role=toolbar)", f"ArrowDown from the first tool moves focus to {a['region']}/{a['name'][:20]!r}; Tab moves to {b['region']}/{b['name'][:20]!r} (WAI-ARIA toolbar pattern expects arrow keys to move within the toolbar)")
        W.open_sample(SAMPLE_BLOBS)
        tabs = W.page.evaluate("() => Array.from(document.querySelectorAll('.tabs .tab')).map(t => ({tag: t.tagName, role: t.getAttribute('role'), tabindex: t.tabIndex, selected: t.getAttribute('aria-selected'), close: !!t.querySelector('button.close'), closeLabel: t.querySelector('button.close')?.getAttribute('aria-label') || t.querySelector('button.close')?.getAttribute('title')}))")
        W.note("image tabs", f"{tabs[:3]}")
        W.page.keyboard.press("Control+Shift+Backslash")
        W.page.locator(".split-view").wait_for(timeout=10_000)
        sp = W.page.evaluate("() => { const d = document.querySelector('.split-divider'); return {divider_role: d && d.getAttribute('role'), divider_tabindex: d && d.tabIndex, heads: Array.from(document.querySelectorAll('.split-head')).map(h => h.textContent.trim().slice(0, 40))}; }")
        W.note("split view", f"{sp}", shot=W.shot("split_view_a11y"))
        W.page.keyboard.press("Control+Shift+Backslash")
        sr = W.page.evaluate("() => { const s = document.querySelector('.splitter'); return s ? {role: s.getAttribute('role'), tabindex: s.tabIndex, orientation: s.getAttribute('aria-orientation'), valuenow: s.getAttribute('aria-valuenow')} : null; }")
        W.note("inspector splitter", f"{sr}")

    def motion_and_reflow() -> None:
        iid = W.open_sample(SAMPLE_3D)
        vol = W.section("3-D View")
        W.page.wait_for_timeout(2500)
        auto = vol.locator('[title="Auto-rotate"]').first
        st = auto.evaluate("e => ({tag: e.tagName, pressed: e.getAttribute('aria-pressed'), checked: e.checked, cls: e.className})") if auto.count() else None
        canvas = vol.locator(".vol-canvas").first
        cattrs = canvas.evaluate("c => ({tabindex: c.tabIndex, role: c.getAttribute('role'), label: c.getAttribute('aria-label')})") if canvas.count() else None
        W.note("3-D View", f"auto-rotate control {st}; canvas {cattrs} (keyboard orbit/zoom?)", shot=W.shot("volume_a11y", locator=vol))
        lm = W.page.evaluate(JS_LANDMARKS)
        W.note("animations", f"elements animating now: {lm['animated']}; CSS has a prefers-reduced-motion rule: {lm['reducedMotionRule']}")
        tr = W.section("Tracks", True) if W.section_head("Tracks").count() else None
        # the busy spinner / progress chip during a long op
        W.palette_run("gauss", "Gaussian Blur")
        W.sheet().locator('input[type="number"]').first.fill("8")
        W.page.wait_for_timeout(100)
        spin = W.page.evaluate("() => Array.from(document.querySelectorAll('.spin, .busy-dot, .progress')).map(e => ({cls: String(e.className).slice(0, 20), role: e.getAttribute('role'), label: e.getAttribute('aria-label'), anim: getComputedStyle(e).animationName}))")
        W.note("busy indicators", f"{spin[:4]}")
        W.settle()

    def narrow_viewport() -> None:
        ctx = W.page.context
        pg = ctx.new_page()
        pg.set_viewport_size({"width": 640, "height": 400})
        pg.goto(W.url)
        pg.locator(".app").wait_for(timeout=30_000)
        pg.wait_for_timeout(800)
        m = pg.evaluate("() => ({scrollW: document.documentElement.scrollWidth, clientW: document.documentElement.clientWidth, inspector: !!document.querySelector('.section'), inspectorWidth: (document.querySelector('.section')||{}).clientWidth || 0, canvasW: (document.querySelector('.viewport')||{}).clientWidth || 0, rail: !!document.querySelector('.rail'), menubar: (document.querySelector('.menubar')||{}).clientWidth || 0})")
        p = W.out / f"{W.n + 1:02d}_narrow_640.png"
        W.n += 1
        pg.screenshot(path=str(p))
        W.note("viewport 640×400 (≈ 200 % zoom on a 1280 px laptop)", f"{m}", shot=p.name)
        pg.close()

    W.probe("landmarks", landmarks)
    W.probe("tab_walk", tab_walk)
    W.probe("menubar_keyboard", menubar_keyboard)
    W.probe("palette_aria", palette_aria)
    W.probe("image_and_sliders", image_and_sliders)
    W.probe("contrast_both_themes", contrast_both_themes)
    W.probe("dialogs_focus", dialogs_focus)
    W.probe("toolbar_and_tabs", toolbar_and_tabs)
    W.probe("motion_and_reflow", motion_and_reflow)
    W.probe("narrow_viewport", narrow_viewport)
    common_tour(W)


PERSONAS: dict[str, tuple[Callable[[Walk], None], dict[str, int]]] = {
    "power": (walk_power, {"width": 1440, "height": 900}),
    "biologist": (walk_biologist, {"width": 1366, "height": 768}),
    "a11y": (walk_a11y, {"width": 1440, "height": 900}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8766")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "walkthrough"))
    ap.add_argument("--persona", default="all", choices=["all", *PERSONAS])
    ap.add_argument("--channel", default="chrome")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    http_json(url, "/api/meta")  # fail early when the server is not there
    names = list(PERSONAS) if args.persona == "all" else [args.persona]
    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.channel or None, headless=True, args=SWIFTSHADER)
        try:
            for name in names:
                fn, vp = PERSONAS[name]
                ctx = browser.new_context(viewport=vp)
                page = ctx.new_page()
                W = Walk(page, url, name, Path(args.out) / name)
                try:
                    page.goto(url)
                    fn(W)
                finally:
                    W.cleanup()
                    W.save()
                    ctx.close()
        finally:
            browser.close()


if __name__ == "__main__":
    main()
