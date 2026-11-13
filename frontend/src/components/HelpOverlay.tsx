import { useRef } from "react";
import { useStore } from "../store";
import { kbd } from "../platform";
import { prettyCombo } from "../customShortcuts";
import { useFocusTrap } from "../focusTrap";

const GROUPS: { title: string; items: [string, string][] }[] = [
  { title: "General", items: [["Command palette", "⌘K"], ["Open file", "⌘O"], ["Save / Save as", "⌘S / ⇧⌘S"], ["Close image / Close all", "⌘W / ⇧⌘W"], ["Undo / Redo", "⌘Z / ⇧⌘Z"], ["Repeat last command", "⇧R"], ["Preferences", "⌘,"], ["Toggle inspector", "⌘\\"], ["Split view", "⇧⌘\\"], ["Print", "⌘P"], ["Script editor (brush size while the brush tool is active)", "["], ["Leave the script editor", "Esc"], ["Run script / selection (editor)", "⌘R / ⇧⌘R"], ["New text window", "⇧N"], ["Menu bar (then ← → ↑ ↓, ↵)", "F10 / ⌥ tap"], ["Cancel / deselect", "Esc"], ["This help", "?"]] },
  { title: "Tools", items: [["Select / move", "V"], ["Pan (hold)", "Space"], ["Rectangle / Oval", "1 / 2"], ["Polygon / Freehand", "3 / 4"], ["Line / Segmented line", "5 / 6"], ["Points / Wand", "7 / 8"], ["Angle", "Q"], ["Manual tracking", "X"], ["Brush / Bucket fill", "9 / U"], ["Text / Arrow", "Y / J"], ["Zoom / Value picker", "Z / I"]] },
  { title: "View", items: [["Zoom in / out", "+ / −"], ["Fit to window", "0"], ["Zoom at cursor", "wheel"], ["Image tab 1–8 / last", "⌘1–8 / ⌘9"], ["Next / previous image", "⌃Tab / ⌃⇧Tab · ← → on a tab"], ["Previous / next slice", ", / ."], ["Previous / next frame", "< / >"], ["Scroll slices / frames", "⇧ wheel / ⌥ wheel"], ["Nudge selection", "arrows (⇧ ×10)"], ["Show / hide overlay", "⇧O"], ["Zoom to selection · Sync views", "View menu"]] },
  { title: "Commands (ImageJ-compatible)", items: [["Measure", "M"], ["Add to ROI manager", "T"], ["Histogram", "H"], ["Plot profile", "K"], ["Select all / none", "A / ⇧A"], ["Brightness & contrast", "⇧⌘C"], ["Threshold", "⇧⌘T"], ["Crop / Duplicate", "⇧X / ⇧D"], ["Fill / Invert", "F / ⇧I"], ["Smooth", "⇧S"], ["Cut / Copy / Paste pixels", "⌘X / ⌘C / ⌘V"], ["Copy to System", "Edit menu"], ["Restore selection", "⇧E"], ["Add to overlay", "B"], ["New image", "N"], ["Color picker", "⇧K"], ["Capture screen / image", "⇧G / ⇧H"]] },
  { title: "Drawing & painting", items: [["Constrain square / circle / 45°", "⇧ drag"], ["Draw from centre", "⌥ drag"], ["Close polygon", "double-click / ↵"], ["Remove last vertex", "⌫"], ["Add point", "⇧ click"], ["Brush size (brush tool active; otherwise [ opens the script editor)", "[ / ]"], ["Erase with brush", "⌥ drag"], ["Commit / cancel text", "↵ / Esc"], ["Angle: arm end, vertex, arm end", "3 clicks"], ["Track: add point (next frame)", "click"], ["Track: delete this frame's point", "⌥ click"]] },
];

export default function HelpOverlay() {
  const close = () => useStore.getState().setHelpOpen(false);
  const custom = useStore((s) => s.customShortcuts);
  // M25: focus moves into the overlay once on open (the container itself: long reading content, so no scroll
  // jump; ↑↓ scroll it), Tab / Shift-Tab cycle inside (Manage…, Close), Esc closes and the "?" button or menu
  // trigger that opened it gets the focus back.  The hook runs once, not on every render (customShortcuts
  // changes re-render this).
  const dialogRef = useRef<HTMLDivElement>(null);
  const trap = useFocusTrap(dialogRef, { onEscape: close, initialFocus: "container" });
  const manage = () => {
    close();
    useStore.getState().setShortcutsOpen(true);
  };
  return (
    <div className="modal-backdrop" onMouseDown={close}>
      <div className="help-overlay" role="dialog" aria-modal="true" aria-labelledby="help-title" tabIndex={-1} ref={dialogRef} onMouseDown={(e) => e.stopPropagation()} onKeyDown={trap.onKeyDown}>
        <h2 id="help-title">Keyboard shortcuts</h2>
        <div className="muted small">Lumen keeps ImageJ's single-letter command shortcuts and adds a command palette for everything else. Esc closes this list; ↑ ↓ scroll it.</div>
        <div className="cols">
          {GROUPS.map((g) => (
            <div key={g.title}>
              <h4>{g.title}</h4>
              {g.items.map(([l, k]) => <div key={l} className="sc"><span>{l}</span><span className="kbd">{kbd(k)}</span></div>)}
            </div>
          ))}
          <div>
            <h4>Custom (Plugins ▸ Shortcuts)</h4>
            {custom.length === 0 && <div className="hint-text">No custom shortcuts yet.</div>}
            {custom.map((c) => <div key={c.combo} className="sc"><span>{c.label}</span><span className="kbd">{prettyCombo(c.combo)}</span></div>)}
            <button className="btn sm manage" onClick={manage}>Manage…</button>
          </div>
        </div>
        <div className="btn-row" style={{ justifyContent: "flex-end", marginTop: 14 }}><button className="btn primary" onClick={close}>Close</button></div>
      </div>
    </div>
  );
}
