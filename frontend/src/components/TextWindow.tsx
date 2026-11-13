/**
 * File ▸ New ▸ Text Window (⇧N) and text files opened as tables: a single-column "Text" table (file.new_text,
 * file.import_text's "Line" / "Text" table too) is shown in the inspector's table section as an editable text
 * area with Save (text.set – the server-side window follows the edits), Append (text.append, ImageJ's
 * print("[window]", …)), Save to file (file.save_text, after saving the edits), Download, "→ Editor" and
 * Run as Python / ImageJ macro / JavaScript through the script editor.  Every op is optional: without it the
 * window keeps working locally (Save to file then goes through POST /api/textfile) and a toast says so.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { CornerDownLeft, Download, FileText, Play, Save } from "lucide-react";
import { api } from "../api";
import { useStore } from "../store";
import { guessLang, useScriptStore, type ScriptLang } from "../scriptStore";
import type { TableData } from "../types";

/** A table that is really a text window: one "Text" column, or "Line" + "Text" (file.import_text). */
export function isTextTable(t: TableData | null | undefined): boolean {
  if (!t || !Array.isArray(t.columns)) return false;
  const cols = t.columns.map((c) => String(c).toLowerCase());
  return (cols.length === 1 && cols[0] === "text") || (cols.length === 2 && cols[0] === "line" && cols[1] === "text");
}

export function tableText(t: TableData): string {
  const col = t.columns.length === 1 ? 0 : 1;
  return t.rows.map((r) => String(r?.[col] ?? "")).join("\n");
}

function textTable(name: string, text: string): TableData {
  return { name, columns: ["Text"], rows: text ? text.split("\n").map((ln) => [ln]) : [] };
}

function withExt(name: string): string {
  return /\.[a-z0-9]+$/i.test(name) ? name : name + ".txt";
}

export default function TextWindow({ table }: { table: TableData }) {
  const toast = useStore((s) => s.toast);
  const runOp = useStore((s) => s.runOp);
  const setFileDialog = useStore((s) => s.setFileDialog);
  const hasSet = useStore((s) => !!s.ops["text.set"]);
  const hasAppend = useStore((s) => !!s.ops["text.append"]);
  const hasSaveFile = useStore((s) => !!s.ops["file.save_text"]);
  // the window lives on the server (session.tables) when it came from an op; a local fallback window does not
  const onServer = useStore((s) => !!s.state?.tables?.[table.name]);
  const initial = useMemo(() => tableText(table), [table]);
  const [text, setText] = useState(initial);
  const [dirty, setDirty] = useState(false);
  const [line, setLine] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    setText(initial);
    setDirty(false);
  }, [initial]);
  const lineCount = useMemo(() => (text ? text.split("\n").length : 0), [text]);
  const name = table.name || "Untitled.txt";
  // text.set / text.append / file.save_text work on single-column "Text" windows; a "Line" / "Text" table
  // (file.import_text) is edited locally and written out through the text-file endpoint
  const pureText = table.columns.length === 1;
  const serverSave = hasSet && onServer && pureText;

  /** Save the edits: text.set on the server window, else keep them in the local table. */
  const save = async (silent = false): Promise<boolean> => {
    if (serverSave) {
      const res = await runOp("text.set", { name, text }, { silent: true });
      if (!res) return false;
      if (!silent) toast(`Saved ${name} (${lineCount} line${lineCount === 1 ? "" : "s"})`, "success");
    } else {
      useStore.setState({ table: textTable(name, text) });
      if (!silent) toast(hasSet ? `${name} is a local window – kept the edits here (Save to file writes them out)` : "text.set is not registered on this server – edits kept locally (Save to file writes them out)", "info");
    }
    setDirty(false);
    return true;
  };
  const append = async () => {
    const add = line;
    setLine("");
    if (hasAppend && onServer && pureText) {
      if (dirty && !(await save(true))) return;
      await runOp("text.append", { name, text: add }, { silent: true });
      return;
    }
    const next = text ? text + "\n" + add : add;
    setText(next);
    useStore.setState({ table: textTable(name, next) });
    if (!hasAppend) toast("text.append is not registered on this server – appended locally", "info");
  };
  const saveToFile = () =>
    setFileDialog({
      mode: "pick-save", title: `Save “${name}” to a file`, filename: withExt(name),
      onPick: (p) => {
        void (async () => {
          try {
            if (hasSaveFile && onServer && pureText) {
              if (dirty && !(await save(true))) return;
              const res = await runOp("file.save_text", { name, path: p }, { silent: true });
              if (res) toast(res.message || `Saved ${p}`, "success");
            } else {
              const r = await api.writeTextFile(p, text + (text ? "\n" : ""));
              setDirty(false);
              toast(`Saved ${r.path} (${r.bytes} bytes)${hasSaveFile ? "" : " – file.save_text is not registered, wrote the file directly"}`, "success");
            }
          } catch (e: any) {
            toast(e.message || String(e), "error");
          }
        })();
      },
    });
  const download = () => {
    const url = URL.createObjectURL(new Blob([text + (text ? "\n" : "")], { type: "text/plain;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = withExt(name);
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  const runAs = (lang: ScriptLang) => useScriptStore.getState().load(text, lang, { run: true, path: null });
  const toEditor = () => useScriptStore.getState().load(text, guessLang(name, useScriptStore.getState().lang), { path: null });

  return (
    <div className="text-window">
      <div className="btn-row" style={{ marginTop: 0, marginBottom: 6 }}>
        <button className="btn sm primary" onClick={() => void save()} disabled={!dirty} title={serverSave ? "Save the edits into the text window (text.set)" : "Keep the edits in this window"}><Save size={12} /> Save</button>
        <button className="btn sm" onClick={saveToFile} title="Write the text to a file on the server (file.save_text)"><FileText size={12} /> Save to file…</button>
        <button className="btn sm" onClick={download} title="Download to this computer"><Download size={12} /></button>
        <span className="grow" />
        <select aria-label="Run as" title="Run the text as a script" value="" onChange={(e) => { const v = e.target.value as ScriptLang | ""; if (v) runAs(v); }}>
          <option value="">Run as…</option>
          <option value="python">Python</option>
          <option value="jython">Jython (ImageJ 1.x API)</option>
          <option value="groovy">Groovy (Java bridge)</option>
          <option value="beanshell">BeanShell (Java bridge)</option>
          <option value="rhino">Rhino JavaScript (Java bridge)</option>
          <option value="ijm">ImageJ macro</option>
          <option value="js">JavaScript</option>
        </select>
        <button className="btn sm" onClick={toEditor} title="Open the text in the script editor"><Play size={12} /> → Editor</button>
      </div>
      <textarea ref={taRef} className="text-window-area" rows={10} value={text} spellCheck={false} aria-label={`Text window ${name}`}
        onChange={(e) => { setText(e.target.value); setDirty(true); }}
        onKeyDown={(e) => { e.stopPropagation(); if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") { e.preventDefault(); void save(); } }} />
      <div className="row" style={{ marginTop: 6 }}>
        <input type="text" value={line} placeholder="Append a line…" aria-label="Line to append" style={{ flex: 1 }} spellCheck={false}
          onChange={(e) => setLine(e.target.value)} onKeyDown={(e) => { e.stopPropagation(); if (e.key === "Enter") void append(); }} />
        <button className="btn sm" onClick={() => void append()} title={'Append the line to the window (text.append, like print("[window]", …))'}><CornerDownLeft size={12} /> Append</button>
      </div>
      <div className="hint-text">{name}{dirty ? " • edited" : ""} · {lineCount} line{lineCount === 1 ? "" : "s"}{onServer ? "" : " · local window"} · {serverSave ? "Save (⌘S in the text) stores the edits in the window" : "edits stay in this window until you save them to a file"}</div>
    </div>
  );
}

/** File ▸ New ▸ Text Window: through `file.new_text` when registered, else a local (unsaved) text window. */
export function newTextWindow(): void {
  const st = useStore.getState();
  st.setMenuOpen(null);
  st.setInspectorOpen(true);
  const spec = st.ops["file.new_text"];
  if (spec) {
    const params: Record<string, unknown> = {};
    for (const p of spec.params) params[p.name] = p.type === "image" ? null : p.default;
    void st.runOp("file.new_text", params, { silent: true }).then((res) => {
      if (res && !res.table) {
        useStore.setState({ table: textTable("Untitled.txt", "") });
        useStore.getState().togglePanel("table", true);
      }
    });
    return;
  }
  st.toast("file.new_text is not registered on this server – opened a local text window (Save to file writes it out)", "info");
  useStore.setState({ table: textTable("Untitled.txt", "") });
  st.togglePanel("table", true);
}

/** File ▸ New ▸ Table…: through `table.new` when registered, else an empty local table. */
export function newTable(): void {
  const st = useStore.getState();
  st.setMenuOpen(null);
  st.setInspectorOpen(true);
  if (st.ops["table.new"]) {
    st.invoke("table.new");
    return;
  }
  st.toast("table.new is not registered on this server – showing an empty local table", "info");
  useStore.setState({ table: { name: "Table", columns: ["A", "B", "C"], rows: [] } });
  st.togglePanel("table", true);
}
