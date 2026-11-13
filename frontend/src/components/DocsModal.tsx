import { useEffect, useMemo, useRef, useState } from "react";
import { X, Search, ExternalLink } from "lucide-react";
import { api } from "../api";
import { useStore, type DocName } from "../store";
import { renderMarkdown } from "./markdown";
import { DOC_LINKS } from "../menus";
import { useFocusTrap } from "../focusTrap";

const DOCS: { name: DocName; title: string }[] = [
  { name: "README", title: "README" },
  { name: "COMMANDS", title: "Commands" },
  { name: "FIJI_PARITY", title: "Fiji parity" },
  { name: "SCRIPTING", title: "Scripting" },
  { name: "MCP", title: "AI assistants" },
  { name: "DESIGN", title: "Design" },
  { name: "RESEARCH", title: "Research" },
];

/** Help ▸ Lumen docs and "?" in the operation sheet: the bundled markdown, fetched from /api/docs/{name}. */
export default function DocsModal() {
  const open = useStore((s) => s.docsOpen)!;
  const setDocsOpen = useStore((s) => s.setDocsOpen);
  const close = () => setDocsOpen(null);
  // from a sheet's "?" (compact): the user documents only – the Design / Research notes stay in the Help menu (E7-20)
  const docs = open.compact ? DOCS.filter((d) => d.name !== "DESIGN" && d.name !== "RESEARCH") : DOCS;
  const [md, setMd] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [missing, setMissing] = useState(false);
  const bodyRef = useRef<HTMLDivElement>(null);
  const cache = useRef<Partial<Record<DocName, string>>>({});
  // Focus management (M24/M25): the filter box takes focus once on open (typing filters the table at once),
  // Tab cycles inside, Esc closes (a non-empty filter clears itself first, see the input) and the opener
  // (menu trigger, the sheet's "?" button) gets the focus back on close.  The hook runs once, not on every
  // render, so a keystroke in the filter box never pulls the focus off the input.
  const dialogRef = useRef<HTMLDivElement>(null);
  const trap = useFocusTrap(dialogRef, { onEscape: close, initialFocus: ".doc-search input" });

  useEffect(() => {
    let alive = true;
    setErr(null);
    setMissing(false);
    setQuery("");
    const hit = cache.current[open.name];
    if (hit) {
      setMd(hit);
      return;
    }
    setMd(null);
    api.doc(open.name).then((t) => {
      cache.current[open.name] = t;
      if (alive) setMd(t);
    }).catch((e) => alive && setErr(e.message || String(e)));
    return () => {
      alive = false;
    };
  }, [open.name]);

  // the ImageJ-macro equivalents the server sends per command (ij_commands, newer servers): shown under the reference rows (E7-20)
  const ops = useStore((s) => s.ops);
  const ijCommands = useMemo(() => {
    const m: Record<string, string[]> = {};
    for (const o of Object.values(ops)) if (o.ij_commands && o.ij_commands.length) m[o.id] = o.ij_commands;
    return m;
  }, [ops]);
  const nodes = useMemo(() => (md ? renderMarkdown(md, { filter: query, onDoc: (name) => setDocsOpen({ name }), ijCommands }) : null), [md, query, setDocsOpen, ijCommands]);

  // scroll to the command's row ("Help on this command")
  useEffect(() => {
    if (!md || !open.anchor || query) return;
    const el = bodyRef.current?.querySelector(`[data-op="${CSS.escape(open.anchor)}"]`) as HTMLElement | null;
    if (!el) {
      setMissing(true);
      return;
    }
    setMissing(false);
    el.classList.add("hl");
    window.setTimeout(() => el.scrollIntoView({ block: "center" }), 0);
    return () => el.classList.remove("hl");
  }, [md, nodes, open.anchor, query]);

  return (
    <div className="modal-backdrop" onMouseDown={close}>
      <div className="dialog docs" role="dialog" aria-modal="true" aria-labelledby={"doc-tab-" + open.name} tabIndex={-1} ref={dialogRef} onMouseDown={(e) => e.stopPropagation()} onKeyDown={trap.onKeyDown}>
        <div className="dialog-head">
          <div className="doc-tabs" role="group" aria-label="Document">
            {docs.map((d) => (
              <button key={d.name} id={"doc-tab-" + d.name} className={"doc-tab" + (d.name === open.name ? " on" : "")} aria-pressed={d.name === open.name} onClick={() => setDocsOpen({ name: d.name, compact: open.compact })}>{d.title}</button>
            ))}
          </div>
          <span style={{ flex: 1 }} />
          <label className="doc-search">
            <Search size={13} aria-hidden="true" />
            <input placeholder="Filter table rows…" aria-label="Filter table rows" value={query} onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Escape" && query) { e.stopPropagation(); setQuery(""); } }} />
          </label>
          <button className="icon-btn" onClick={close} title="Close (Esc)" aria-label="Close (Esc)"><X size={15} /></button>
        </div>
        <div className="dialog-body doc-body" ref={bodyRef}>
          {open.anchor && missing && <div className="sheet-error" style={{ margin: "0 0 10px" }}>No entry for <code>{open.anchor}</code> in the command reference (the file may predate this command).</div>}
          {err && <div className="sheet-error" style={{ margin: 0 }}>{err}</div>}
          {!err && !md && <div className="hint-text" style={{ padding: 20 }}>Loading…</div>}
          {nodes && <div className="md">{nodes}</div>}
        </div>
        <div className="dialog-foot">
          {DOC_LINKS.map((l) => (
            <a key={l.url} className="btn sm" href={l.url} target="_blank" rel="noopener noreferrer"><ExternalLink size={11} /> {l.label}</a>
          ))}
          <span className="grow" />
          <button className="btn" onClick={close}>Close</button>
        </div>
      </div>
    </div>
  );
}
