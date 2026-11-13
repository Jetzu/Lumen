import { useMemo, useRef, useState } from "react";
import { X, Keyboard, Trash2, Plus } from "lucide-react";
import { useStore } from "../store";
import { CLIENT_ACTIONS } from "../actions";
import { prettyShortcut } from "../menus";
import { BUILTIN_SHORTCUTS, comboFromEvent, prettyCombo, type CustomShortcut } from "../customShortcuts";
import { useFocusTrap } from "../focusTrap";

interface Entry { id: string; name: string; category: string; op?: string; action?: string; arg?: unknown; kbd?: string }

/** Plugins ▸ Shortcuts ▸ Add / Remove Shortcut…: custom key bindings kept in localStorage "lumen.shortcuts". */
export default function ShortcutsDialog() {
  const ops = useStore((s) => s.meta?.ops || []);
  const list = useStore((s) => s.customShortcuts);
  const setList = useStore((s) => s.setCustomShortcuts);
  const toast = useStore((s) => s.toast);
  const close = () => useStore.getState().setShortcutsOpen(false);
  const [q, setQ] = useState("");
  const [target, setTarget] = useState<Entry | null>(null);
  const [combo, setCombo] = useState<string | null>(null);
  const [capturing, setCapturing] = useState(false);
  // focus starts in the command search, Tab cycles inside, Esc closes, the opener (menu trigger or the help
  // overlay's Manage… button's own opener) gets the focus back
  const dialogRef = useRef<HTMLDivElement>(null);
  const trap = useFocusTrap(dialogRef, { onEscape: close, initialFocus: "#sc-command" });

  const entries = useMemo<Entry[]>(() => [
    ...CLIENT_ACTIONS.map((a) => ({ id: a.id, name: a.name, category: a.category, action: a.action, arg: a.arg, kbd: a.kbd })),
    ...ops.map((o) => ({ id: o.id, name: o.name, category: o.menu, op: o.id, kbd: prettyShortcut(o.shortcut) })),
  ], [ops]);
  const matches = useMemo(() => {
    const w = q.trim().toLowerCase();
    const all = w ? entries.filter((e) => e.name.toLowerCase().includes(w) || e.id.toLowerCase().includes(w) || e.category.toLowerCase().includes(w)) : entries;
    return all.slice(0, 60);
  }, [q, entries]);

  const builtin = combo ? BUILTIN_SHORTCUTS.find((b) => b.combo === combo) : undefined;
  const existing = combo ? list.find((c) => c.combo === combo) : undefined;
  const labelOf = (c: CustomShortcut) => c.label || c.op || c.action || "";

  const add = () => {
    if (!target || !combo) return;
    // a client action keeps its argument: the Help ▸ docs entries share action "docs" and differ only by arg
    const entry: CustomShortcut = { combo, label: target.name, ...(target.op ? { op: target.op } : { action: target.action, ...(target.arg !== undefined ? { arg: target.arg } : {}) }) };
    setList([...list.filter((c) => c.combo !== combo), entry]);
    toast(`${prettyCombo(combo)} → ${target.name}`, "success");
    setCombo(null);
    setTarget(null);
    setQ("");
  };
  const remove = (c: CustomShortcut) => setList(list.filter((x) => x !== c));

  const onCapture = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Tab") return; // never trap the keyboard in the capture field: Tab moves on (dialog trap cycles)
    e.preventDefault();
    e.stopPropagation();
    if (e.key === "Escape") {
      setCombo(null);
      (e.target as HTMLInputElement).blur();
      return;
    }
    const c = comboFromEvent(e);
    if (c) setCombo(c);
  };

  return (
    <div className="modal-backdrop" onMouseDown={close}>
      <div className="dialog shortcuts" role="dialog" aria-modal="true" aria-labelledby="shortcuts-title" tabIndex={-1} ref={dialogRef} onMouseDown={(e) => e.stopPropagation()} onKeyDown={trap.onKeyDown}>
        <div className="dialog-head">
          <Keyboard size={16} aria-hidden="true" />
          <h3 id="shortcuts-title">Custom shortcuts</h3>
          <span className="dim small">stored in this browser · override the built-in bindings</span>
          <span style={{ flex: 1 }} />
          <button className="icon-btn" onClick={close} title="Close (Esc)" aria-label="Close (Esc)"><X size={15} /></button>
        </div>
        <div className="dialog-body">
          <div className="sc-add">
            <div className="sc-col">
              <label className="small muted" htmlFor="sc-command">1 · Command</label>
              <input id="sc-command" type="text" placeholder="Search commands…" value={target ? target.name : q} onChange={(e) => { setTarget(null); setQ(e.target.value); }} />
              {!target && (
                <div className="sc-list" role="listbox" aria-label="Matching commands">
                  {matches.map((e) => (
                    <div key={e.id} className="sc-item" role="option" aria-selected={false} tabIndex={0} onClick={() => setTarget(e)} onKeyDown={(ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); setTarget(e); } }}>
                      <span className="nm">{e.name}</span>
                      <span className="cat">{e.category}</span>
                      {e.kbd && <span className="kbd">{e.kbd}</span>}
                    </div>
                  ))}
                  {!matches.length && <div className="hint-text" style={{ padding: 8 }}>No command matches “{q}”.</div>}
                </div>
              )}
              {target && <div className="hint-text">{target.category} · <code>{target.op || target.action}</code> <button className="btn sm" style={{ marginLeft: 6 }} onClick={() => setTarget(null)}>change</button></div>}
            </div>
            <div className="sc-col">
              <label className="small muted" htmlFor="sc-keys">2 · Keys</label>
              <input
                id="sc-keys"
                type="text"
                readOnly
                className={"sc-capture" + (capturing ? " on" : "")}
                value={combo ? prettyCombo(combo) : ""}
                placeholder={capturing ? "Press the key combination…" : "Click, then press keys"}
                onFocus={() => setCapturing(true)}
                onBlur={() => setCapturing(false)}
                onKeyDown={onCapture}
              />
              {builtin && <div className="hint-text" style={{ color: "var(--warn)" }}>Overrides the built-in “{builtin.label}”.</div>}
              {existing && <div className="hint-text" style={{ color: "var(--warn)" }}>Replaces your shortcut for “{labelOf(existing)}”.</div>}
              {!builtin && !existing && combo && <div className="hint-text">Free.</div>}
              <button className="btn primary" disabled={!target || !combo} onClick={add} style={{ marginTop: 8 }}><Plus size={13} /> Add shortcut</button>
            </div>
          </div>
          <h4 className="sc-h">Your shortcuts {list.length ? `(${list.length})` : ""}</h4>
          {!list.length && <div className="hint-text">None yet. Shortcuts are checked before the built-in ones, so you can rebind any key. Single keys only fire outside text fields.</div>}
          <div className="sc-current">
            {list.map((c) => (
              <div key={c.combo} className="sc-row">
                <span className="kbd">{prettyCombo(c.combo)}</span>
                <span className="nm">{labelOf(c)}</span>
                <span className="cat">{c.op || c.action}</span>
                <button className="icon-btn" title="Remove shortcut" aria-label={`Remove shortcut ${prettyCombo(c.combo)} (${labelOf(c)})`} onClick={() => remove(c)}><Trash2 size={13} /></button>
              </div>
            ))}
          </div>
        </div>
        <div className="dialog-foot">
          <span className="dim small">Listed under Help ▸ Keyboard Shortcuts.</span>
          <span className="grow" />
          {list.length > 0 && <button className="btn sm danger" onClick={() => { if (window.confirm("Remove all custom shortcuts?")) setList([]); }}>Remove all</button>}
          <button className="btn primary" onClick={close}>Done</button>
        </div>
      </div>
    </div>
  );
}
