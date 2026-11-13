import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ChevronRight, Moon, PanelRight, Redo2, Search, Sun, Undo2, HelpCircle, Settings } from "lucide-react";
import { useStore } from "../store";
import { buildMenus, type MenuItem } from "../menus";
import { doAction } from "../actions";
import { kbd } from "../platform";
import { leaveMenubar } from "../shortcuts";
import ProgressBar from "./ProgressBar";

/**
 * E8-1: the menubar's dropdowns are rendered through a portal into document.body with `position: fixed` coordinates
 * taken from their trigger, so no ancestor (the top bar, the menubar, any future overflow / transform on them) can
 * clip them.  This attribute marks such a popover: the bar's outside-click and blur checks treat it as part of the bar,
 * which DOM containment no longer says.
 */
const POPOVER_ATTR = "data-menubar-popover";

/** Whether `node` lies in the menubar itself or in one of its portaled popovers. */
function withinMenubar(bar: HTMLElement | null, node: EventTarget | null): boolean {
  if (!(node instanceof Node)) return false;
  if (bar && bar.contains(node)) return true;
  const el = node instanceof Element ? node : node.parentElement;
  return !!el && !!el.closest(`[${POPOVER_ATTR}]`);
}

const HIDDEN_UNTIL_PLACED: React.CSSProperties = { position: "fixed", top: 0, left: 0, visibility: "hidden" };

/**
 * Fixed-position coordinates of a portaled menu: measured from the trigger's getBoundingClientRect() when the menu
 * opens (before paint) and again on every window resize and scroll.  A menu that would leave the viewport on the
 * right is flipped to end at the trigger's right edge; one that does not fit below opens upwards when there is more
 * room above; one taller than the room gets a max-height with its own scrollbar (Image ▸ Lookup Tables).
 */
function usePopoverPosition(anchor: HTMLElement | null, ref: React.RefObject<HTMLDivElement>): React.CSSProperties {
  const [pos, setPos] = useState<React.CSSProperties | null>(null);
  useLayoutEffect(() => {
    if (!anchor) return;
    const place = (e?: Event) => {
      const el = ref.current;
      if (!el) return;
      if (e && e.type === "scroll" && e.target instanceof Node && el.contains(e.target)) return; // the menu's own scrollbar
      const a = anchor.getBoundingClientRect();
      const vw = window.innerWidth;
      const vh = window.innerHeight;
      const M = 6; // margin to the viewport edges
      // natural size, then put the limit back (React only writes a style it sees changing)
      const prevMax = el.style.maxHeight;
      el.style.maxHeight = "";
      const w = el.offsetWidth;
      const h = el.offsetHeight;
      el.style.maxHeight = prevMax;
      let left = a.left;
      if (left + w > vw - M) left = Math.max(M, Math.min(a.right, vw - M) - w);
      const below = vh - a.bottom - 2 - M;
      const above = a.top - 2 - M;
      const down = h <= below || below >= above;
      const room = Math.max(60, Math.floor(down ? below : above));
      const top = down ? a.bottom + 2 : Math.max(M, a.top - 2 - Math.min(h, room));
      setPos({ position: "fixed", top, left, maxHeight: room, overflowY: h > room ? "auto" : undefined });
    };
    place();
    window.addEventListener("resize", place);
    window.addEventListener("scroll", place, true);
    return () => {
      window.removeEventListener("resize", place);
      window.removeEventListener("scroll", place, true);
    };
  }, [anchor, ref]);
  return pos ?? HIDDEN_UNTIL_PLACED;
}

export default function TopBar() {
  const ops = useStore((s) => s.meta?.ops || []);
  const menuOpen = useStore((s) => s.menuOpen);
  const setMenuOpen = useStore((s) => s.setMenuOpen);
  const theme = useStore((s) => s.theme);
  const setTheme = useStore((s) => s.setTheme);
  const busy = useStore((s) => s.busy);
  const state = useStore((s) => s.state);
  // joined so the menus are only rebuilt when the recent-files list really changes
  const recentKey = useStore((s) => (s.state?.recent || []).join("\n"));
  const active = state?.active;
  const hist = active ? state?.history[active] : undefined;
  const undoLabel = hist?.undo.length ? hist.undo[hist.undo.length - 1] : undefined;
  const redoLabel = hist?.redo.length ? hist.redo[hist.redo.length - 1] : undefined;
  const lutGroups = useStore((s) => s.lutGroups);
  const lutGrouping = useStore((s) => s.lutGrouping);
  const lutNames = useStore((s) => s.meta?.luts);
  const menus = useMemo(() => buildMenus(ops, { recent: recentKey ? recentKey.split("\n") : [], undoLabel, redoLabel, lutGroups, lutGrouping, lutNames }), [ops, recentKey, undoLabel, redoLabel, lutGroups, lutGrouping, lutNames]);
  const barRef = useRef<HTMLDivElement>(null);
  // Keyboard menubar (WAI-ARIA menubar pattern): the triggers share one Tab stop (roving tabindex); ←/→ move
  // between them, ↓ / ↑ / Enter / Space open a menu and focus its first (last) item, Esc closes and returns
  // the focus to the trigger.  F10 or a tap on Alt focus the bar (shortcuts.ts).  `focusItems` says which
  // item a freshly opened menu should focus: only menus opened from the keyboard take the focus.
  const [focusIdx, setFocusIdx] = useState(0);
  const [focusItems, setFocusItems] = useState<"first" | "last" | null>(null);
  const triggerRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const indexOf = (label: string) => menus.findIndex((m) => m.label === label);
  const openMenu = (i: number, focus: "first" | "last" | null) => {
    setFocusIdx(i);
    setFocusItems(focus);
    setMenuOpen(menus[i].label);
  };
  /** Close the open menu; `refocus` puts the keyboard focus back on its trigger (keyboard activation / Esc). */
  const closeMenu = (refocus: boolean) => {
    const i = menuOpen ? indexOf(menuOpen) : -1;
    setMenuOpen(null);
    setFocusItems(null);
    if (refocus && i >= 0) triggerRefs.current[i]?.focus();
  };
  const switchMenu = (dir: 1 | -1) => {
    const n = menus.length;
    const cur = menuOpen ? indexOf(menuOpen) : focusIdx;
    const j = (cur + dir + n) % n;
    if (menuOpen) openMenu(j, "first");
    else {
      setFocusIdx(j);
      triggerRefs.current[j]?.focus();
    }
  };
  const onTriggerKey = (e: React.KeyboardEvent, i: number) => {
    switch (e.key) {
      case "ArrowRight":
        switchMenu(1);
        break;
      case "ArrowLeft":
        switchMenu(-1);
        break;
      case "ArrowDown":
      case "Enter":
      case " ":
        openMenu(i, "first");
        break;
      case "ArrowUp":
        openMenu(i, "last");
        break;
      case "Home":
        setFocusIdx(0);
        triggerRefs.current[0]?.focus();
        if (menuOpen) openMenu(0, "first");
        break;
      case "End": {
        const j = menus.length - 1;
        setFocusIdx(j);
        triggerRefs.current[j]?.focus();
        if (menuOpen) openMenu(j, "first");
        break;
      }
      case "Escape":
        // a second Esc (no menu open) leaves the bar: back to the element F10 came from, else the image (E5-17)
        if (!menuOpen) leaveMenubar();
        else closeMenu(true);
        break;
      case "Tab":
        if (menuOpen) setMenuOpen(null); // the focus leaves the bar; never trap it
        return;
      default:
        return;
    }
    e.preventDefault();
    e.stopPropagation();
  };

  useEffect(() => {
    if (!menuOpen) return;
    // a press outside the bar and its (portaled) popovers closes the menu; one inside a popover reaches the item
    const close = (e: MouseEvent) => {
      if (!withinMenubar(barRef.current, e.target)) setMenuOpen(null);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [menuOpen, setMenuOpen]);
  // the focus left the menubar and its popovers (Tab, a click elsewhere that moves focus): close the open menu.
  // React's onBlur bubbles from the portaled items through the React tree, so this sees them too.
  const onBarBlur = (e: React.FocusEvent) => {
    if (menuOpen && e.relatedTarget && !withinMenubar(barRef.current, e.relatedTarget)) setMenuOpen(null);
  };

  const undoTitle = `Undo${undoLabel ? " " + undoLabel : ""} (${kbd("⌘Z")})`;
  const redoTitle = `Redo${redoLabel ? " " + redoLabel : ""} (${kbd("⇧⌘Z")})`;
  const inspectorLabel = `Toggle inspector (${kbd("⌘\\")})`;
  const prefsLabel = `Preferences (${kbd("⌘,")})`;
  return (
    <div className="topbar">
      <div className="logo"><span className="dot" />Lumen</div>
      <div className="menubar" role="menubar" aria-label="Main menu" ref={barRef} onBlur={onBarBlur}>
        {menus.map((m, i) => (
          <div key={m.label} style={{ position: "relative" }}>
            <button
              className={"menu-trigger" + (menuOpen === m.label ? " open" : "")}
              role="menuitem"
              aria-haspopup="menu"
              aria-expanded={menuOpen === m.label}
              tabIndex={i === focusIdx ? 0 : -1}
              ref={(el) => {
                triggerRefs.current[i] = el;
              }}
              onFocus={() => setFocusIdx(i)}
              onKeyDown={(e) => onTriggerKey(e, i)}
              onClick={() => {
                if (menuOpen === m.label) closeMenu(false);
                else openMenu(i, null);
              }}
              onMouseEnter={() => menuOpen && menuOpen !== m.label && openMenu(i, null)}
            >
              {m.label}
            </button>
            {menuOpen === m.label && (
              <MenuList items={m.items} anchor={triggerRefs.current[i]} autoFocus={focusItems} onDone={(fromKeyboard) => closeMenu(fromKeyboard)} onLeft={() => switchMenu(-1)} onRight={() => switchMenu(1)} onEscape={() => closeMenu(true)} />
            )}
          </div>
        ))}
      </div>
      <button className="search-btn" onClick={() => useStore.getState().setPaletteOpen(true)}>
        <Search size={14} /> <span className="search-label">Search commands…</span> <span className="kbd">{kbd("⌘K")}</span>
      </button>
      <ProgressBar />
      {busy > 0 && <span className="busy-dot" role="status" aria-label="Working…" title="Working…" />}
      <button className="icon-btn" title={undoTitle} aria-label={undoTitle} disabled={!hist?.undo.length} onClick={() => void useStore.getState().undo()}><Undo2 size={16} /></button>
      <button className="icon-btn" title={redoTitle} aria-label={redoTitle} disabled={!hist?.redo.length} onClick={() => void useStore.getState().redo()}><Redo2 size={16} /></button>
      <button className="icon-btn" title={inspectorLabel} aria-label={inspectorLabel} onClick={() => doAction("inspector")}><PanelRight size={16} /></button>
      {/* "secondary": hidden on narrow windows (styles.css, < 1000 px) so the bar never overflows – all three stay reachable from the View / Edit / Help menus */}
      <button className="icon-btn secondary" title="Toggle theme" aria-label="Toggle theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}</button>
      <button className="icon-btn secondary" title={prefsLabel} aria-label={prefsLabel} onClick={() => doAction("preferences")}><Settings size={16} /></button>
      <button className="icon-btn secondary" title="Keyboard shortcuts (?)" aria-label="Keyboard shortcuts (?)" onClick={() => doAction("help")}><HelpCircle size={16} /></button>
    </div>
  );
}

interface MenuListProps {
  items: MenuItem[];
  /** Called after an item ran; `fromKeyboard` = activated with Enter / Space (the trigger takes the focus back). */
  onDone: (fromKeyboard: boolean) => void;
  sub?: boolean;
  /** Focus the first / last item once the list is shown (keyboard-opened menus and submenus). */
  autoFocus?: "first" | "last" | null;
  /** ← : top-level menu → previous menubar menu; submenu → close it and focus the parent item. */
  onLeft?: () => void;
  /** → on an item without submenu: next menubar menu (also from inside submenus, as in the ARIA pattern). */
  onRight?: () => void;
  /** Esc: top-level → close the menu and refocus the trigger; submenu → close the submenu. */
  onEscape?: () => void;
  /** Top-level menus: the trigger the popover is positioned under; the list is then rendered through a portal into document.body (E8-1). */
  anchor?: HTMLElement | null;
  /** Submenus: position adjustments measured by SubMenu (flip left / shift up at the viewport edges). */
  style?: React.CSSProperties;
}

function MenuList({ items, onDone, sub, autoFocus, onLeft, onRight, onEscape, anchor, style }: MenuListProps) {
  const ops = useStore((s) => s.ops);
  const st = useStore();
  const img = st.activeImage();
  const ref = useRef<HTMLDivElement>(null);
  const portal = !sub && !!anchor;
  const fixedStyle = usePopoverPosition(portal ? anchor! : null, ref);
  // the submenu that is open (hover or keyboard) and whether its first/last item takes the focus
  const [openSub, setOpenSub] = useState<{ i: number; focus: "first" | "last" | null } | null>(null);
  const itemsOf = () => Array.from(ref.current?.querySelectorAll<HTMLElement>(':scope > [role^="menuitem"]:not([aria-disabled="true"])') ?? []);
  // a portaled popover is visibility:hidden until usePopoverPosition has measured it (one layout pass), and a hidden
  // element cannot take focus – so the keyboard focus is placed once the popover is visible (E10-1), and only once per
  // opening (the position changes again on scroll / resize, which must not steal the focus back to the first item)
  const focusedFor = useRef<unknown>(null);
  useEffect(() => {
    if (!autoFocus) {
      focusedFor.current = null;
      return;
    }
    if (focusedFor.current === autoFocus) return;
    let raf = 0;
    const tryFocus = (n: number) => {
      const el = ref.current;
      const list = itemsOf();
      const target = autoFocus === "last" ? list[list.length - 1] : list[0];
      if (el && target && getComputedStyle(el).visibility !== "hidden") {
        focusedFor.current = autoFocus;
        target.focus();
        return;
      }
      if (n < 20) raf = requestAnimationFrame(() => tryFocus(n + 1));
    };
    tryFocus(0);
    return () => cancelAnimationFrame(raf);
  }, [autoFocus, fixedStyle]); // eslint-disable-line react-hooks/exhaustive-deps
  const click = (it: MenuItem, fromKeyboard: boolean) => {
    if (it.disabled) return;
    if (it.op && it.params) void st.runOp(it.op, it.params);
    else if (it.op) st.invoke(it.op);
    else if (it.action) doAction(it.action, it.arg);
    onDone(fromKeyboard);
  };
  const subItemOf = (i: number) => ref.current?.querySelector<HTMLElement>(`:scope > [data-sub="${i}"]`) ?? null;
  const closeSub = () => {
    const i = openSub?.i;
    setOpenSub(null);
    if (i !== undefined) subItemOf(i)?.focus();
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    const list = itemsOf();
    const cur = list.indexOf(document.activeElement as HTMLElement);
    const focusAt = (i: number) => {
      if (list.length) list[((i % list.length) + list.length) % list.length].focus();
    };
    switch (e.key) {
      case "ArrowDown":
        focusAt(cur + 1);
        break;
      case "ArrowUp":
        focusAt(cur - 1);
        break;
      case "Home":
        focusAt(0);
        break;
      case "End":
        focusAt(list.length - 1);
        break;
      case "ArrowRight": {
        const el = list[cur];
        if (el?.dataset.sub !== undefined) setOpenSub({ i: Number(el.dataset.sub), focus: "first" });
        else if (onRight) onRight();
        else return;
        break;
      }
      case "ArrowLeft":
        if (!onLeft) return;
        onLeft();
        break;
      case "Enter":
      case " ": {
        const el = list[cur];
        if (!el) return;
        if (el.dataset.sub !== undefined) setOpenSub({ i: Number(el.dataset.sub), focus: "first" });
        else el.click(); // detail 0 → treated as keyboard activation in the click handler
        break;
      }
      case "Escape":
        if (!onEscape) return;
        onEscape();
        break;
      case "Tab":
        return; // the menubar closes the menu on blur
      default: {
        // type-ahead: jump to the next item whose label starts with the typed character
        if (e.key.length !== 1 || e.metaKey || e.ctrlKey || e.altKey) return;
        const ch = e.key.toLowerCase();
        for (let k = 1; k <= list.length; k++) {
          const el = list[(cur + k + list.length) % list.length];
          if ((el.querySelector(".label")?.textContent || "").trim().toLowerCase().startsWith(ch)) {
            el.focus();
            break;
          }
        }
      }
    }
    e.preventDefault();
    e.stopPropagation(); // reaches neither the parent menu nor the global shortcuts
  };
  const list = (
    <div className={"menu-popover" + (sub ? " submenu" : "")} role="menu" ref={ref} style={sub ? { display: "block", ...style } : portal ? fixedStyle : undefined} onKeyDown={onKeyDown} {...(portal ? { [POPOVER_ATTR]: "" } : {})}>
      {items.map((it, i) => {
        if (it.sep) return <div key={i} className="menu-sep" role="separator" />;
        const spec = it.op ? ops[it.op] : undefined;
        const disabled = it.disabled || (spec?.requires_image && !img) || (!!spec?.requires_roi && img && !st.selections[img.id]);
        // a disabled group header inside a menu ("ImageJ" above the LUT names): plain text, not a menuitem
        if (it.disabled && !it.op && !it.action && !it.children && !it.kbd) return <div key={i} className="menu-heading" role="presentation">{it.label}</div>;
        if (it.children) {
          const isOpen = openSub?.i === i;
          return (
            <div
              key={i}
              className={"menu-item has-sub" + (isOpen ? " kb-open" : "")}
              role="menuitem"
              aria-haspopup="menu"
              aria-expanded={isOpen}
              tabIndex={-1}
              data-sub={i}
              onMouseEnter={() => setOpenSub({ i, focus: null })}
              onClick={(e) => {
                e.stopPropagation();
                setOpenSub({ i, focus: e.detail === 0 ? "first" : null });
              }}
            >
              <span className="label">{it.label}</span>
              <ChevronRight size={13} className="chev" />
              <SubMenu items={it.children} open={isOpen} onDone={onDone} autoFocus={isOpen ? openSub.focus : null} onLeft={closeSub} onRight={onRight} onEscape={closeSub} />
            </div>
          );
        }
        const checked = it.checked ? !!(st as unknown as Record<string, unknown>)[it.checked] : false;
        return (
          <div
            key={i}
            className={"menu-item" + (disabled ? " disabled" : "")}
            role={it.checked ? "menuitemcheckbox" : "menuitem"}
            aria-checked={it.checked ? checked : undefined}
            aria-disabled={disabled || undefined}
            tabIndex={-1}
            onMouseEnter={() => setOpenSub(null)}
            onClick={(e) => click(it, e.detail === 0)}
            title={it.title || spec?.description}
          >
            {it.checked && <span className="check" aria-hidden="true">{checked ? "✓" : ""}</span>}
            <span className="label">{it.label}</span>
            {it.kbd && <span className="kbd">{kbd(it.kbd)}</span>}
          </div>
        );
      })}
    </div>
  );
  // the popover lives in document.body: nothing above the menubar can clip it (E8-1); React events (keys, blur,
  // clicks) still bubble through the React tree to the menubar's handlers
  return portal ? createPortal(list, document.body) : list;
}

/**
 * A submenu: absolutely positioned at the right of its item (CSS), and – once it is open (hover or keyboard) –
 * measured: one that would leave the viewport on the right is flipped to the item's left, one that would leave it
 * at the bottom is shifted up, one taller than the viewport scrolls.  Inside a portaled top-level menu the only
 * ancestor is body, so the viewport edges are the only limits.
 */
function SubMenu({ open, ...props }: MenuListProps & { open: boolean }) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const [adjust, setAdjust] = useState<{ wrap: React.CSSProperties; list: React.CSSProperties } | null>(null);
  useLayoutEffect(() => {
    if (!open) {
      setAdjust(null);
      return;
    }
    const wrap = wrapRef.current;
    const list = wrap?.querySelector<HTMLElement>(":scope > .menu-popover");
    const item = wrap?.parentElement;
    if (!wrap || !list || !item) return;
    // measure at the natural place (right of the item, top aligned)
    wrap.style.left = "";
    wrap.style.right = "";
    wrap.style.top = "";
    list.style.left = "";
    list.style.right = "";
    list.style.maxHeight = "";
    list.style.overflowY = "";
    const r = list.getBoundingClientRect();
    const ir = item.getBoundingClientRect();
    const M = 6;
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const wrapStyle: React.CSSProperties = {};
    const listStyle: React.CSSProperties = {};
    if (r.right > vw - M && ir.left - r.width >= M) {
      // flip: the wrapper sits at the item's left edge, the list extends leftwards from it
      wrapStyle.left = "auto";
      wrapStyle.right = "100%";
      listStyle.left = "auto";
      listStyle.right = 0;
    }
    if (r.height > vh - 2 * M) {
      listStyle.maxHeight = vh - 2 * M;
      listStyle.overflowY = "auto";
      wrapStyle.top = -5 - Math.max(0, r.top - M);
    } else if (r.bottom > vh - M) wrapStyle.top = -5 - Math.min(r.bottom - (vh - M), Math.max(0, r.top - M));
    setAdjust({ wrap: wrapStyle, list: listStyle });
  }, [open]);
  return (
    <div className="submenu" ref={wrapRef} style={adjust?.wrap}>
      <MenuList {...props} sub style={adjust?.list} />
    </div>
  );
}
