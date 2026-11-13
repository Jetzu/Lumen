"""Browser check that the menubar dropdowns are really visible and clickable (E8-1 regression guard) against a
running Lumen server: installed Chrome, headless. `python tests/e2e/usability/verify_menus_visible.py http://127.0.0.1:8766`

A DOM read is not enough (the items were in the DOM while the popover was clipped): this script opens every top-level
menu with the mouse, asks the browser which element sits at the centre of the popover and of its first item
(`document.elementFromPoint`), checks that it is a menu item, that the popover's box lies inside the viewport, hovers
the first submenu and checks the same for it, then clicks an item and checks that the menu closed. It also repeats
the first-menu check at 1280, 1440 and 1920 px wide, and with the keyboard (F10 → ↓).
"""
from __future__ import annotations

import re
import sys

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8766"
results: dict[str, str] = {}

JS_AT = """
(sel) => {
  const el = document.querySelector(sel);
  if (!el) return { ok: false, why: 'no element' };
  const r = el.getBoundingClientRect();
  const cx = r.left + r.width / 2, cy = r.top + Math.min(r.height / 2, 60);
  const hit = document.elementFromPoint(cx, cy);
  const inside = r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight;
  return { ok: !!hit && (el === hit || el.contains(hit)), inside, box: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)],
           hit: hit ? (hit.tagName + '.' + hit.className + (hit.getAttribute('role') ? '[' + hit.getAttribute('role') + ']' : '')) : null };
}
"""


def check_menu(page, label: str, key: str) -> None:
    trig = page.locator(".menubar .menu-trigger", has_text=re.compile(rf"^{re.escape(label)}$")).first
    trig.click()
    page.wait_for_timeout(250)
    pop = page.locator('.menu-popover[data-menubar-popover], .menu-popover').first
    pop.wait_for(state="visible", timeout=5000)
    a = page.evaluate(JS_AT, ".menu-popover[data-menubar-popover], .menu-popover")
    # the first *enabled* item: a disabled one (Crop without a selection) has pointer-events: none by design
    first = page.locator('.menu-popover [role="menuitem"]:not([aria-disabled="true"])').first
    fb = first.bounding_box()
    hit = page.evaluate("([x, y]) => { const h = document.elementFromPoint(x, y); return h ? (h.closest('[role=menuitem]') ? 'menuitem' : h.tagName + '.' + h.className) : null; }", [fb["x"] + fb["width"] / 2, fb["y"] + fb["height"] / 2]) if fb else None
    results[f"menu_{key}_visible"] = ("ok" if a["ok"] and a["inside"] and hit == "menuitem" else f"popover hit={a.get('hit')} inside={a.get('inside')} box={a.get('box')} first-item hit={hit}")
    # first submenu, if any
    sub = page.locator('.menu-popover [role="menuitem"][aria-haspopup="menu"], .menu-popover [role="menuitem"][aria-haspopup="true"]').first
    if sub.count():
        sub.hover()
        page.wait_for_timeout(350)
        subpops = page.locator(".menu-popover .menu-popover, .submenu, [data-submenu]")
        if subpops.count():
            sb = subpops.first.bounding_box()
            sh = page.evaluate("([x, y]) => { const h = document.elementFromPoint(x, y); return h ? (h.closest('[role=menuitem]') ? 'menuitem' : h.tagName + '.' + h.className) : null; }", [sb["x"] + 20, sb["y"] + 14]) if sb else None
            results[f"menu_{key}_submenu"] = "ok" if sh == "menuitem" else f"submenu hit={sh} box={sb}"
        else:
            results[f"menu_{key}_submenu"] = "no submenu popover found after hover"
    page.keyboard.press("Escape")
    page.keyboard.press("Escape")
    page.wait_for_timeout(150)


with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True, args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader"])
    for width in (1440, 1280, 1920):
        ctx = browser.new_context(viewport={"width": width, "height": 900})
        page = ctx.new_page()
        page.goto(URL, wait_until="networkidle")
        page.wait_for_timeout(600)
        try:
            if width == 1440:
                for label in ("File", "Edit", "Image", "Process", "Analyze", "Plugins", "Window", "Help"):
                    if page.locator(".menubar .menu-trigger", has_text=re.compile(rf"^{label}$")).count():
                        check_menu(page, label, label.lower())
                # keyboard: F10 then ArrowDown opens the first menu; the first item must be visible where it says it is
                page.keyboard.press("Escape")
                page.locator("body").click(position={"x": 700, "y": 500})
                page.keyboard.press("F10")
                page.keyboard.press("ArrowDown")
                page.wait_for_timeout(300)
                a = page.evaluate(JS_AT, ".menu-popover")
                results["keyboard_menu_visible"] = "ok" if a["ok"] and a["inside"] else f"{a}"
                # E10-1: the keyboard focus must be on a menu item, ↓ moves to the next one, → opens the next menu with its first
                # item focused, Esc returns to the trigger
                focused = page.evaluate("() => { const a = document.activeElement; return a ? (a.getAttribute('role') || a.tagName) + ':' + (a.textContent || '').trim().slice(0, 20) : null; }")
                first_menu = page.locator('.menubar .menu-trigger[aria-expanded="true"]').first.inner_text().strip() if page.locator('.menubar .menu-trigger[aria-expanded="true"]').count() else None
                page.keyboard.press("ArrowDown")
                page.wait_for_timeout(120)
                focused2 = page.evaluate("() => { const a = document.activeElement; return a ? (a.getAttribute('role') || a.tagName) + ':' + (a.textContent || '').trim().slice(0, 20) : null; }")
                page.keyboard.press("ArrowRight")
                page.wait_for_timeout(300)
                focused3 = page.evaluate("() => { const a = document.activeElement; return a ? (a.getAttribute('role') || a.tagName) + ':' + (a.textContent || '').trim().slice(0, 20) : null; }")
                open_label = page.locator('.menubar .menu-trigger[aria-expanded="true"]').first.inner_text().strip() if page.locator('.menubar .menu-trigger[aria-expanded="true"]').count() else None
                page.keyboard.press("Escape")
                page.wait_for_timeout(120)
                focused4 = page.evaluate("() => { const a = document.activeElement; return a ? (a.className || a.tagName) : null; }")
                ok = bool(focused and focused.startswith("menuitem")) and bool(focused2 and focused2.startswith("menuitem")) and focused != focused2 \
                    and bool(focused3 and focused3.startswith("menuitem")) and open_label is not None and open_label != first_menu and "menu-trigger" in str(focused4)
                results["keyboard_focus_moves"] = "ok" if ok else f"after ↓: {focused} | ↓ again: {focused2} | → : {focused3} (open menu {open_label}) | Esc: {focused4}"
                page.keyboard.press("Escape")
                # a click on a real item runs it and closes the menu (Edit ▸ Select None is harmless)
                page.locator(".menubar .menu-trigger", has_text=re.compile(r"^Help$")).first.click()
                page.wait_for_timeout(250)
                item = page.locator('.menu-popover [role="menuitem"]').first
                name = item.inner_text().strip()
                item.click()
                page.wait_for_timeout(500)
                results["click_closes_menu"] = "ok" if page.locator(".menu-popover").count() == 0 or not page.locator(".menu-popover").first.is_visible() else f"menu still open after clicking {name!r}"
                page.keyboard.press("Escape")
            else:
                check_menu(page, "Image", f"image_{width}")
        except Exception as e:  # noqa: BLE001
            results[f"error_{width}"] = str(e)[:200]
        ctx.close()
    browser.close()

for k, v in results.items():
    print(f"{k:28s} {v}")
