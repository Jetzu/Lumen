import React, { useCallback, useEffect, useRef, useState } from "react";
import { X, Plus, ChevronDown, Columns2 } from "lucide-react";
import { useStore, useOrderedImages } from "../store";
import { doAction } from "../actions";
import { kbd } from "../platform";
import type { ImageSummary } from "../types";

const DRAG_THRESHOLD = 6; // px of pointer travel before a press becomes a tab drag

/** Close several images (Close Others / Close All): one confirmation when any of them has unsaved changes. */
async function closeMany(ids: string[]): Promise<void> {
  await useStore.getState().closeImages(ids);
}

function dimsLabel(im: ImageSummary): string {
  const sh = im.shape;
  const dims = [sh.c > 1 ? `${sh.c}c` : "", sh.z > 1 ? `${sh.z}z` : "", sh.t > 1 ? `${sh.t}t` : ""].filter(Boolean).join(" ");
  return `${sh.x}×${sh.y}${dims ? " · " + dims : ""} · ${im.bit_depth}`;
}

export default function ImageTabs() {
  const images = useOrderedImages();
  const active = useStore((s) => s.state?.active);
  const history = useStore((s) => s.state?.history);
  const savedRevs = useStore((s) => s.savedRevs);
  const isModified = useStore((s) => s.isModified);
  void savedRevs; // subscribed so a Save re-renders the modified dots (isModified reads the store)
  const select = useStore((s) => s.selectImage);
  const close = useStore((s) => s.closeImage);
  const setImageOrder = useStore((s) => s.setImageOrder);
  const splitOn = useStore((s) => s.split.on);
  const toggleSplit = useStore((s) => s.toggleSplitView);

  const scrollRef = useRef<HTMLDivElement>(null);
  const [fade, setFade] = useState({ l: false, r: false });
  const [menuOpen, setMenuOpen] = useState(false);
  const [ctx, setCtx] = useState<{ x: number; y: number; id: string } | null>(null);
  const [drag, setDrag] = useState<{ id: string; over: string | null; after: boolean } | null>(null);
  const dragRef = useRef<{ id: string; sx: number; sy: number; started: boolean; pointerId: number; target: Element } | null>(null);
  const dropRef = useRef<{ over: string | null; after: boolean }>({ over: null, after: false });
  const suppressClick = useRef(false);

  // ---------------------------------------------------------- overflow fades
  const updateFade = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const l = el.scrollLeft > 2;
    const r = el.scrollLeft + el.clientWidth < el.scrollWidth - 2;
    setFade((f) => (f.l === l && f.r === r ? f : { l, r }));
  }, []);
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    updateFade();
    const ro = new ResizeObserver(updateFade);
    ro.observe(el);
    el.addEventListener("scroll", updateFade, { passive: true });
    return () => {
      ro.disconnect();
      el.removeEventListener("scroll", updateFade);
    };
  }, [updateFade]);
  useEffect(() => {
    updateFade();
  }, [images, updateFade]);

  // keep the active tab in view; when the tab strip has the keyboard focus (←/→, ⌘1–9), the focus follows the selection
  useEffect(() => {
    if (!active || !scrollRef.current) return;
    const el = scrollRef.current.querySelector<HTMLElement>(`[data-tab-id="${CSS.escape(active)}"]`);
    el?.scrollIntoView({ inline: "nearest", block: "nearest" });
    if (el && scrollRef.current.contains(document.activeElement) && document.activeElement !== el) el.focus();
  }, [active]);

  // WAI-ARIA tabs pattern: one Tab stop (the active tab), ←/→ Home/End select, Delete closes
  const onTabKey = (e: React.KeyboardEvent) => {
    if (!images.length) return;
    const cur = images.findIndex((im) => im.id === active);
    let next = -1;
    switch (e.key) {
      case "ArrowRight":
        next = (cur + 1) % images.length;
        break;
      case "ArrowLeft":
        next = (cur - 1 + images.length) % images.length;
        break;
      case "Home":
        next = 0;
        break;
      case "End":
        next = images.length - 1;
        break;
      case "Delete":
      case "Backspace":
        if (cur >= 0) void close(images[cur].id);
        e.preventDefault();
        return;
      default:
        return;
    }
    e.preventDefault();
    e.stopPropagation();
    select(images[next].id);
  };

  // vertical wheel scrolls the strip horizontally
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (Math.abs(e.deltaY) > Math.abs(e.deltaX) && el.scrollWidth > el.clientWidth) {
        el.scrollLeft += e.deltaY;
        e.preventDefault();
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  // ⌘⇧\ (Ctrl+Shift+\) toggles the split view – handled centrally in shortcuts.ts so custom shortcuts take precedence

  // ---------------------------------------------------------- drag to reorder
  const onPointerDown = (e: React.PointerEvent, id: string) => {
    if (e.button !== 0 || (e.target as Element).closest("button")) return;
    dragRef.current = { id, sx: e.clientX, sy: e.clientY, started: false, pointerId: e.pointerId, target: e.currentTarget };
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    if (!d.started) {
      if (Math.hypot(e.clientX - d.sx, e.clientY - d.sy) < DRAG_THRESHOLD) return;
      d.started = true;
      d.target.setPointerCapture?.(d.pointerId);
    }
    const el = scrollRef.current;
    const tabs = Array.from(el?.querySelectorAll<HTMLElement>(".tab[data-tab-id]") || []);
    let over: string | null = null;
    let after = false;
    for (const t of tabs) {
      const r = t.getBoundingClientRect();
      if (e.clientX >= r.left && e.clientX <= r.right) {
        over = t.dataset.tabId!;
        after = e.clientX > r.left + r.width / 2;
        break;
      }
    }
    if (!over && tabs.length) {
      const first = tabs[0].getBoundingClientRect();
      const last = tabs[tabs.length - 1].getBoundingClientRect();
      if (e.clientX < first.left) over = tabs[0].dataset.tabId!;
      else if (e.clientX > last.right) {
        over = tabs[tabs.length - 1].dataset.tabId!;
        after = true;
      }
    }
    dropRef.current = { over, after };
    setDrag({ id: d.id, over, after });
    if (el) {
      const r = el.getBoundingClientRect();
      if (e.clientX < r.left + 24) el.scrollLeft -= 8;
      else if (e.clientX > r.right - 24) el.scrollLeft += 8;
    }
  };
  const endDrag = (commit: boolean) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (!d || !d.started) return;
    d.target.releasePointerCapture?.(d.pointerId);
    suppressClick.current = true; // the click that follows the pointerup must not select the dragged tab
    window.setTimeout(() => (suppressClick.current = false), 0);
    setDrag(null);
    const { over, after } = dropRef.current;
    if (!commit || !over || over === d.id) return;
    const ids = images.map((im) => im.id).filter((id) => id !== d.id);
    const at = ids.indexOf(over) + (after ? 1 : 0);
    ids.splice(at, 0, d.id);
    setImageOrder(ids);
  };

  // ---------------------------------------------------------- context menu
  const ctxImage = ctx ? images.find((im) => im.id === ctx.id) : undefined;
  const idx = ctx ? images.findIndex((im) => im.id === ctx.id) : -1;
  const closeCtx = () => setCtx(null);
  const ctxItems: { label: string; run: () => void; disabled?: boolean; sep?: boolean }[] = ctxImage
    ? [
        { label: "Close", run: () => void close(ctxImage.id) },
        { label: "Close Others", run: () => void closeMany(images.filter((im) => im.id !== ctxImage.id).map((im) => im.id)), disabled: images.length < 2 },
        { label: "Close to the Right", run: () => void closeMany(images.slice(idx + 1).map((im) => im.id)), disabled: idx >= images.length - 1 },
        { label: "Close All", run: () => void closeMany(images.map((im) => im.id)) },
        { label: "Compare Side by Side", sep: true, disabled: images.length < 2, run: () => {
          const st = useStore.getState();
          const act = st.state?.active ?? null;
          if (ctxImage.id === act) st.setSplit({ on: true, other: st.split.other && st.split.other !== act && images.some((im) => im.id === st.split.other) ? st.split.other : images.find((im) => im.id !== act)?.id ?? null });
          else st.setSplit({ on: true, other: ctxImage.id });
        } },
      ]
    : [];

  return (
    <div className="tabs">
      <div className={"tab-scroll-wrap" + (fade.l ? " fade-l" : "") + (fade.r ? " fade-r" : "")}>
        <div className="tab-scroll" ref={scrollRef} role="tablist" aria-label="Open images" aria-orientation="horizontal" onKeyDown={onTabKey}>
          {images.length === 0 && <div className="empty">No images open — press {kbd("⌘O")} or drop files here</div>}
          {images.map((im, i) => {
            const modified = isModified(im.id);
            const undoSteps = history?.[im.id]?.undo.length ?? 0;
            const cls = "tab" + (im.id === active ? " active" : "") + (modified ? " modified" : "") + (drag?.id === im.id ? " dragging" : "") + (drag && drag.over === im.id && drag.id !== im.id ? (drag.after ? " drop-after" : " drop-before") : "");
            const shortcut = i < 8 ? kbd(`⌘${i + 1}`) : i === images.length - 1 ? kbd("⌘9") : "";
            return (
              <div key={im.id} data-tab-id={im.id} className={cls}
                role="tab" aria-selected={im.id === active} tabIndex={im.id === active ? 0 : -1}
                aria-label={`${im.name}, ${dimsLabel(im)}${modified ? ", modified" : ""}`}
                onClick={() => { if (suppressClick.current) return; select(im.id); }}
                onAuxClick={(e) => { if (e.button === 1) void close(im.id); }}
                onPointerDown={(e) => onPointerDown(e, im.id)} onPointerMove={onPointerMove} onPointerUp={() => endDrag(true)} onPointerCancel={() => endDrag(false)}
                onContextMenu={(e) => { e.preventDefault(); setMenuOpen(false); setCtx({ x: e.clientX, y: e.clientY, id: im.id }); }}
                title={`${im.name}\n${dimsLabel(im)}${modified ? "\n● modified (" + undoSteps + " undoable step" + (undoSteps === 1 ? "" : "s") + ")" : ""}${shortcut ? "\n" + shortcut + " selects this tab" : ""}\nDrag to reorder · middle-click or Delete to close · right-click for more`}>
                <span className="name">{im.name}</span>
                <span className="dims">{dimsLabel(im)}</span>
                <span className="mod-dot" aria-hidden="true" />
                <button className="close" tabIndex={-1} onClick={(e) => { e.stopPropagation(); void close(im.id); }} title={`Close ${im.name}${modified ? " (unsaved changes)" : ""}`} aria-label={`Close ${im.name}`}><X size={12} /></button>
              </div>
            );
          })}
        </div>
      </div>
      <div className="tab-tools">
        {images.length > 1 && (
          <button className={"icon-btn" + (menuOpen ? " on" : "")} title={`All open images (${images.length})`} onClick={() => setMenuOpen((v) => !v)}><ChevronDown size={14} /></button>
        )}
        <button className={"icon-btn" + (splitOn ? " on" : "")} title={`Split view – compare two images side by side (${kbd("⌘⇧\\")})`} disabled={images.length < 2 && !splitOn} onClick={toggleSplit}><Columns2 size={15} /></button>
        <button className="icon-btn" title={`Open image (${kbd("⌘O")})`} onClick={() => doAction("open")}><Plus size={14} /></button>
        {menuOpen && (
          <>
            <div className="ctx-backdrop" onPointerDown={() => setMenuOpen(false)} onContextMenu={(e) => { e.preventDefault(); setMenuOpen(false); }} />
            <div className="menu-popover tab-menu" role="menu" aria-label="Open images">
              {images.map((im, i) => (
                <div key={im.id} role="menuitemradio" aria-checked={im.id === active} tabIndex={-1} className={"menu-item" + (im.id === active ? " current" : "")} onClick={() => { select(im.id); setMenuOpen(false); }}>
                  <span className="check">{im.id === active ? "●" : ""}</span>
                  <span className="label" style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{i + 1}. {im.name}</span>
                  <span className="dims">{dimsLabel(im)}</span>
                </div>
              ))}
              <div className="menu-sep" />
              <div className="menu-item" role="menuitem" tabIndex={-1} onClick={() => { setMenuOpen(false); void closeMany(images.map((im) => im.id)); }}><span className="check" /><span className="label">Close All</span></div>
            </div>
          </>
        )}
      </div>
      {ctx && ctxImage && (
        <>
          <div className="ctx-backdrop" onPointerDown={closeCtx} onContextMenu={(e) => { e.preventDefault(); closeCtx(); }} />
          <div className="ctx-menu" style={{ left: Math.min(ctx.x, window.innerWidth - 210), top: Math.min(ctx.y, window.innerHeight - 40 * (ctxItems.length + 1)) }}>
            <div className="menu-item disabled" style={{ opacity: 0.7 }}><span className="label" style={{ overflow: "hidden", textOverflow: "ellipsis", maxWidth: 240 }}>{ctxImage.name}</span></div>
            <div className="menu-sep" />
            {ctxItems.map((it) => (
              <React.Fragment key={it.label}>
                {it.sep && <div className="menu-sep" />}
                <div className={"menu-item" + (it.disabled ? " disabled" : "")} onClick={() => { closeCtx(); it.run(); }}><span className="label">{it.label}</span></div>
              </React.Fragment>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

export { closeMany };
