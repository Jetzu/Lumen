/**
 * Inspector ▸ Script & Recorder (Plugins ▸ Scripting ▸ Script Editor, `[`): a Python / Jython / ImageJ-macro / JavaScript
 * editor – plus Fiji's JVM languages (Groovy, BeanShell, Rhino; opt-in JRuby / Clojure) through the server's optional
 * Java bridge – with line numbers, Open… / Save… (server paths through the file dialog, plus browser upload / download),
 * Run (⌘R or ⌘↵) / Run selection / Stop / Clear, an output console and the macro recorder (copy, or hand the
 * recorded steps to the editor as Python or as JavaScript).
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { highlight } from "../highlight";
import { Copy, Download, FolderOpen, Maximize2, Play, Save, Square, Upload, X } from "lucide-react";
import { api } from "../api";
import { useStore } from "../store";
import { kbd } from "../platform";
import { useFocusTrap } from "../focusTrap";
import { recorderToJs, runJs, type JsRunHandle } from "../jsRuntime";
import { DEFAULT_CODE, SCRIPT_LANGS, guessLang, isJvmLang, langExt, runJvmScript, useScriptStore, type ScriptLang } from "../scriptStore";
import { JS_IJ_EXAMPLE, JYTHON_EXAMPLE, PROMPT_TYPES, hasScriptParams, runPythonScript, scriptParams, type ScriptParamDescriptor } from "../ijScript";
import ScriptParamsDialog from "./ScriptParamsDialog";

const KEY_PARAMS = "lumen.script.params";

/** The values entered in the #@ parameter prompt, kept per parameter name for the next run. */
function loadParamMemory(): Record<string, unknown> {
  try {
    return JSON.parse(localStorage.getItem(KEY_PARAMS) || "{}") as Record<string, unknown>;
  } catch {
    return {};
  }
}

function baseName(p: string): string {
  return p.split(/[\\/]/).pop() || p;
}

function downloadText(name: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * "Open in a large view" (E7-14): the same editor and recorder as a dialog over the image, like the 3-D view.
 * Mounted from the Inspector so it is available while the Script section is collapsed; Esc closes it.
 */
export function ScriptModalHost() {
  const open = useScriptStore((s) => s.largeOpen);
  const close = () => useScriptStore.getState().setLargeOpen(false);
  const dialogRef = useRef<HTMLDivElement>(null);
  // keyed on `open`: the host stays mounted while closed, so the trap must take focus when the dialog appears (the
  // textarea, not <body>) and hand it back to the opener when it goes (E7-14 / E7-19)
  const trap = useFocusTrap(dialogRef, { onEscape: close, initialFocus: "textarea", isolate: true, resetKey: open });
  if (!open) return null;
  return (
    <div className="modal-backdrop vol-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}>
      <div className="dialog script-modal" role="dialog" aria-modal="true" aria-labelledby="script-modal-title" tabIndex={-1} ref={dialogRef} onKeyDown={trap.onKeyDown}>
        <div className="dialog-head">
          <h3 id="script-modal-title">Script &amp; Recorder</h3>
          <span className="grow" />
          <button className="icon-btn" onClick={close} title="Back to the inspector (Esc)" aria-label="Back to the inspector (Esc)"><X size={16} /></button>
        </div>
        <div className="dialog-body script-modal-body">
          <ScriptPanel large />
        </div>
      </div>
    </div>
  );
}

/** The recorder's lines without the leading session marker (a `#` comment line written by the server). */
function recorderLines(macro: string[], marker?: string): { marker: string | null; lines: string[] } {
  const first = macro[0];
  if (first && first.startsWith("#") && (!marker || first === marker || /recorder/i.test(first))) return { marker: first, lines: macro.slice(1) };
  return { marker: marker || null, lines: macro };
}

export default function ScriptPanel({ large = false }: { large?: boolean }) {
  const macro = useStore((s) => s.state!.macro);
  const recorderInfo = useStore((s) => s.state?.recorder);
  const refresh = useStore((s) => s.refresh);
  const toast = useStore((s) => s.toast);
  const setFileDialog = useStore((s) => s.setFileDialog);
  const code = useScriptStore((s) => s.code);
  const lang = useScriptStore((s) => s.lang);
  const path = useScriptStore((s) => s.path);
  const out = useScriptStore((s) => s.output);
  const running = useScriptStore((s) => s.running);
  const focusTick = useScriptStore((s) => s.focusTick);
  const runTick = useScriptStore((s) => s.runTick);
  const dirty = useScriptStore((s) => s.dirty);
  const largeOpen = useScriptStore((s) => s.largeOpen);
  const { setCode, setLang, setPath, setOutput, appendOutput, setRunning, load, setDirty } = useScriptStore.getState();
  const taRef = useRef<HTMLTextAreaElement>(null);
  const gutterRef = useRef<HTMLDivElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const macroRef = useRef<HTMLDivElement>(null);
  const jsRef = useRef<JsRunHandle | null>(null);
  const paramMemory = useRef<Record<string, unknown>>(loadParamMemory());
  const [paramPrompt, setParamPrompt] = useState<{ params: ScriptParamDescriptor[]; resolve: (v: Record<string, unknown> | null) => void } | null>(null);
  const lineCount = useMemo(() => code.split("\n").length, [code]);
  const lines = useMemo(() => Array.from({ length: lineCount }, (_, i) => i + 1).join("\n"), [lineCount]);
  // the editor grows with its content: at least 10 rows, up to 40 in the inspector (resizable by hand too, E7-14)
  const rows = large ? 24 : Math.max(10, Math.min(40, lineCount + 1));
  const rec = recorderLines(macro, recorderInfo?.marker);
  const nRecorded = recorderInfo?.count ?? rec.lines.length;
  // the recorder pane follows the newest line (E7-4)
  useEffect(() => {
    const el = macroRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [macro.length, large]);

  /**
   * Put new text into the editor; unsaved edits are never overwritten silently (E7-15): the user chooses between
   * replacing them and appending below.
   */
  const replaceEditor = async (text: string, newLang?: ScriptLang) => {
    const cur = useScriptStore.getState();
    const untouched = !cur.dirty || !cur.code.trim() || Object.values(DEFAULT_CODE).includes(cur.code);
    if (!untouched) {
      const n = cur.code.split("\n").length;
      const answer = await useStore.getState().askConfirm({ title: "Replace the editor text?", message: `The editor holds ${n} unsaved line${n === 1 ? "" : "s"}. Replace them with the new text, or append it below?`, confirmLabel: "Replace", altLabel: "Append below" });
      if (answer === "cancel") return;
      if (answer === "alt") {
        load(cur.code.replace(/\s*$/, "\n\n") + text, newLang);
        setDirty(true);
        return;
      }
    }
    load(text, newLang);
    setDirty(true);
  };
  /** Esc (or ⌘[ / Ctrl+[) hands the keyboard back to the image so the single-key shortcuts work again (E5-15). */
  const leaveEditor = () => {
    taRef.current?.blur();
    if (large) useScriptStore.getState().setLargeOpen(false);
    (document.querySelector(".canvas-wrap") as HTMLElement | null)?.focus();
  };

  // focusTick / runTick are one-shot REQUESTS kept in the (persistent) script store; an effect also runs on every
  // mount, and the panel is unmounted whenever its section is collapsed or the inspector closed (⌘\), so each request
  // is consumed (reset to 0) as soon as it is acted on - otherwise every later remount replayed the last "Run as…",
  // re-executing the script on the server.  The store is read directly so a pending request is still honoured when
  // the panel mounts because of it (load() opens the section and bumps the tick in the same batched render).
  useEffect(() => {
    if (!useScriptStore.getState().focusTick) return;
    useScriptStore.setState({ focusTick: 0 });
    window.setTimeout(() => taRef.current?.focus(), 30);
  }, [focusTick]);
  useEffect(() => {
    if (!useScriptStore.getState().runTick) return;
    useScriptStore.setState({ runTick: 0 }); // consume before running: a remount must not replay the request
    void run(false);
  }, [runTick]); // eslint-disable-line react-hooks/exhaustive-deps

  const run = async (selectionOnly: boolean) => {
    if (useScriptStore.getState().running) return;
    const ta = taRef.current;
    let src = useScriptStore.getState().code;
    const curLang = useScriptStore.getState().lang;
    if (selectionOnly && ta && ta.selectionStart !== ta.selectionEnd) src = ta.value.slice(ta.selectionStart, ta.selectionEnd);
    else if (selectionOnly) {
      // no selection: the current line
      const pos = ta?.selectionStart ?? 0;
      const start = src.lastIndexOf("\n", pos - 1) + 1;
      const end = src.indexOf("\n", pos);
      src = src.slice(start, end < 0 ? undefined : end);
    }
    if (!src.trim()) return toast("Nothing to run", "info");
    // #@ script parameters (Python / Jython and the JVM languages): ask for the values first, outside the running state
    let params: Record<string, unknown> | undefined;
    if ((curLang === "python" || curLang === "jython" || isJvmLang(curLang)) && hasScriptParams(src)) {
      let descr: ScriptParamDescriptor[] = [];
      try {
        descr = await scriptParams(src);
      } catch (e: any) {
        toast(`Could not read the #@ parameters: ${e?.message || e}`, "error");
      }
      if (descr.some((p) => PROMPT_TYPES.has(p.type) && p.io !== "output" && p.visibility?.toUpperCase() !== "MESSAGE")) {
        const values = await new Promise<Record<string, unknown> | null>((resolve) => setParamPrompt({ params: descr, resolve }));
        setParamPrompt(null);
        if (values === null) return;
        params = values;
        paramMemory.current = { ...paramMemory.current, ...values };
        try {
          localStorage.setItem(KEY_PARAMS, JSON.stringify(paramMemory.current));
        } catch {
          /* ignore */
        }
      }
    }
    setRunning(true);
    setOutput({ stdout: "", error: null });
    const st = useStore.getState();
    try {
      if (curLang === "python" || curLang === "jython") {
        const img = st.activeImage();
        const r = await runPythonScript(src, { params, roi: img ? st.selections[img.id] ?? null : null, language: curLang });
        setOutput({ stdout: r.stdout, error: r.error });
        if (r.roi !== undefined) {
          // the script used the ImageJ API: its final selection (or none) becomes the viewport's
          const after = useStore.getState();
          const target = after.activeImage();
          if (target) after.setSelection(target.id, r.roi ?? null);
        }
        if (r.error) toast("Script error – see the output", "error");
      } else if (isJvmLang(curLang)) {
        // Groovy / BeanShell / Rhino / JRuby / Clojure: the server's Java bridge (lumen.core.ijshim.jvm_bridge) binds the
        // active image as imp, converts it back with undo, adds the images the script showed, copies output / Results
        // table / ROI Manager; a missing bridge comes back as an error text with the install steps
        const img = st.activeImage();
        await st.flushDisplay();
        const r = await runJvmScript(src, { language: curLang, params, roi: img ? st.selections[img.id] ?? null : null });
        const note = r.note && !r.error && /RoiManager/.test(src) ? `(${r.note})\n` : "";
        setOutput({ stdout: r.stdout + (note && r.stdout && !r.stdout.endsWith("\n") ? "\n" : "") + note, error: r.error });
        if (r.roi !== undefined) {
          const after = useStore.getState();
          const target = after.activeImage();
          if (target) after.setSelection(target.id, r.roi ?? null);
        }
        if (r.error) toast(/lumen-imaging\[jvm\]/.test(r.error) ? "The Java bridge is not available – the output lists the install steps" : "Script error – see the output", "error");
        else if (r.new_images?.length) toast(`${r.new_images.length} image(s) created by the script`, "info");
      } else if (curLang === "ijm") {
        if (!st.ops["script.run_ij_macro"]) throw new Error("The ImageJ macro interpreter (script.run_ij_macro) is not available on this server");
        const img = st.activeImage();
        await st.flushDisplay();
        const res = await api.run("script.run_ij_macro", { macro: src }, img?.id ?? null, img ? st.selections[img.id] ?? null : null, "stack");
        const after = useStore.getState();
        const target = after.activeImage();
        if (res.roi && target) after.setSelection(target.id, res.roi);
        if (res.clear_roi && target) after.setSelection(target.id, null);
        if (res.table) {
          useStore.setState({ table: res.table });
          after.togglePanel("table", true);
        }
        setOutput({ stdout: res.log || "", error: null });
        if (res.message) toast(res.message, "info");
      } else {
        const handle = runJs(src, { onLog: (t) => appendOutput(t) });
        jsRef.current = handle;
        const r = await handle.done;
        jsRef.current = null;
        const cur = useScriptStore.getState().output || { stdout: "", error: null };
        setOutput({ stdout: cur.stdout + (r.result !== undefined ? `→ ${r.result}\n` : ""), error: r.error });
        if (r.error) toast(r.error === "Stopped" ? "Script stopped" : "JavaScript error – see the output", r.error === "Stopped" ? "info" : "error");
      }
      await refresh();
    } catch (e: any) {
      const cur = useScriptStore.getState().output || { stdout: "", error: null };
      setOutput({ stdout: cur.stdout, error: e?.message || String(e) });
      toast(e?.message || String(e), "error");
    } finally {
      setRunning(false);
    }
  };

  const stop = () => {
    if (jsRef.current) jsRef.current.stop();
    else toast("Only JavaScript scripts can be stopped from here; Python / Jython scripts, macros and JVM scripts run on the server (cancel them in the progress bar)", "info");
  };

  const changeLang = (l: ScriptLang) => {
    const cur = useScriptStore.getState();
    // an untouched default snippet follows the language
    if (!dirty && Object.values(DEFAULT_CODE).includes(cur.code)) setCode(DEFAULT_CODE[l]);
    setLang(l);
  };

  const openServer = () =>
    setFileDialog({
      mode: "pick-open", title: "Open script", extensions: [".py", ".jy", ".ijm", ".txt", ".js", ".mjs", ".groovy", ".gvy", ".bsh", ".rb", ".clj"],
      onPick: (p) => {
        void api.readTextFile(p).then((r) => {
          load(r.text, guessLang(r.name, useScriptStore.getState().lang, r.text), { path: r.path });
          setDirty(false);
          toast(`Opened ${r.name}`, "success");
        }).catch((e) => toast(e.message, "error"));
      },
    });
  const saveServer = () => {
    const cur = useScriptStore.getState();
    setFileDialog({
      mode: "pick-save", title: "Save script", filename: cur.path ? baseName(cur.path) : `script${langExt(cur.lang)}`,
      onPick: (p) => {
        void api.writeTextFile(p, useScriptStore.getState().code).then((r) => {
          setPath(r.path);
          setDirty(false);
          toast(`Saved ${baseName(r.path)} (${r.bytes} bytes)`, "success");
        }).catch((e) => toast(e.message, "error"));
      },
    });
  };
  const onUpload = (f: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result ?? "");
      load(text, guessLang(f.name, useScriptStore.getState().lang, text), { path: null });
      setDirty(false);
      toast(`Loaded ${f.name}`, "success");
    };
    reader.onerror = () => toast(`Could not read ${f.name}`, "error");
    reader.readAsText(f);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    e.stopPropagation();
    const mod = e.metaKey || e.ctrlKey;
    if (e.key === "Escape" || (mod && e.key === "[")) {
      e.preventDefault();
      leaveEditor();
      return;
    }
    if (mod && (e.key === "Enter" || e.key.toLowerCase() === "r")) {
      e.preventDefault();
      void run(e.shiftKey);
      return;
    }
    if (e.key === "Tab" && !mod) {
      // insert two spaces instead of leaving the editor
      e.preventDefault();
      const ta = e.currentTarget;
      const s = ta.selectionStart;
      const next = ta.value.slice(0, s) + "  " + ta.value.slice(ta.selectionEnd);
      setCode(next);
      setDirty(true);
      window.requestAnimationFrame(() => ta.setSelectionRange(s + 2, s + 2));
    }
  };
  const hlRef = useRef<HTMLPreElement>(null);
  const syncScroll = () => {
    const ta = taRef.current;
    if (!ta) return;
    if (gutterRef.current) gutterRef.current.scrollTop = ta.scrollTop;
    if (hlRef.current) {
      hlRef.current.scrollTop = ta.scrollTop;
      hlRef.current.scrollLeft = ta.scrollLeft;
    }
  };
  // syntax colours: a <pre> under the transparent textarea (highlight.ts); recomputed only when the text or language changes
  const highlighted = useMemo(() => highlight(code, lang), [code, lang]);
  const title = (path ? baseName(path) : "untitled" + langExt(lang)) + (dirty ? " •" : "");

  if (!large && largeOpen) {
    return (
      <div className="script-editor">
        <div className="hint-text">The editor is open in the large view.</div>
        <button className="btn sm" onClick={() => useScriptStore.getState().setLargeOpen(false)}>Back to the inspector</button>
      </div>
    );
  }
  return (
    <div className={"script-editor" + (large ? " large" : "")}>
      <div className="script-top">
        <select className="script-lang" value={lang} onChange={(e) => changeLang(e.target.value as ScriptLang)} aria-label="Language" title="Language">
          {SCRIPT_LANGS.map((l) => <option key={l.id} value={l.id}>{l.label}</option>)}
        </select>
        <button className="btn sm" onClick={openServer} title="Open a script from the server's file system"><FolderOpen size={12} /> Open…</button>
        <button className="icon-btn" style={{ width: 24, height: 24 }} onClick={() => fileRef.current?.click()} title="Load a file from this computer" aria-label="Upload script"><Upload size={13} /></button>
        <button className="btn sm" onClick={saveServer} title="Save on the server (choose a folder and a name)"><Save size={12} /> Save…</button>
        <button className="icon-btn" style={{ width: 24, height: 24 }} onClick={() => downloadText(path ? baseName(path) : "script" + langExt(lang), code)} title="Download to this computer" aria-label="Download script"><Download size={13} /></button>
        {!large && <button className="icon-btn" style={{ width: 24, height: 24 }} onClick={() => useScriptStore.getState().setLargeOpen(true)} title="Open the editor in a large view" aria-label="Open the editor in a large view"><Maximize2 size={13} /></button>}
        <input ref={fileRef} type="file" hidden accept=".py,.jy,.ijm,.txt,.js,.mjs,.groovy,.gvy,.bsh,.rb,.clj" onChange={(e) => { const f = e.target.files?.[0]; if (f) onUpload(f); e.target.value = ""; }} />
      </div>
      <div className="mono small dim script-title" title={path || "not saved on the server"}>{title}{dirty ? " – unsaved changes" : ""}</div>
      <div className="editor-wrap">
        <div className="editor-gutter" ref={gutterRef} aria-hidden="true">{lines}</div>
        <div className="code-layer">
          <pre className="code-hl" ref={hlRef} aria-hidden="true">
            {highlighted.map((toks, k) => (
              <div key={k} className="hl-line">{toks.length ? toks.map((tk, j) => tk.kind === "txt" ? tk.text : <span key={j} className={"hl-" + tk.kind}>{tk.text}</span>) : "\u200b"}</div>
            ))}
          </pre>
          <textarea ref={taRef} rows={rows} value={code} spellCheck={false} wrap="off" aria-label={`${SCRIPT_LANGS.find((l) => l.id === lang)?.label} code`} aria-keyshortcuts="Escape" title="Esc returns to the image (shortcuts), ⌘R runs"
            onChange={(e) => { setCode(e.target.value); setDirty(true); }} onKeyDown={onKeyDown} onScroll={syncScroll} />
        </div>
      </div>
      <div className="btn-row" style={{ marginTop: 6 }}>
        <button className="btn sm primary" onClick={() => void run(false)} disabled={running} title={`Run (${kbd("⌘R")} or ${kbd("⌘↵")} in the editor)`}><Play size={12} /> Run</button>
        <button className="btn sm" onClick={() => void run(true)} disabled={running} title={`Run the selected lines, or the current line (${kbd("⇧⌘R")})`}>Run selection</button>
        {running && <button className="btn sm danger" onClick={stop} title="Stop the running JavaScript"><Square size={11} /> Stop</button>}
        <button className="btn sm" onClick={() => setOutput(null)} disabled={!out}>Clear output</button>
        <span className="grow" />
        {lang === "python" && <button className="btn sm" onClick={() => void replaceEditor('print(lumen.commands("thresh"))\nprint(lumen.help("analyze.particles"))')} title="Snippet: list commands and show a command's parameters">API help</button>}
        {lang === "jython" && <button className="btn sm" onClick={() => void replaceEditor(JYTHON_EXAMPLE, "jython")} title="Example: a Fiji Jython script (ij.IJ, RoiManager, ResultsTable, a #@ parameter)">Example</button>}
        {lang === "js" && <button className="btn sm" onClick={() => void replaceEditor('lumen.log(await lumen.commands("thresh"));\nlumen.log(await lumen.help("analyze.particles"));')} title="Snippet: list commands and show a command's parameters">API help</button>}
        {lang === "js" && <button className="btn sm" onClick={() => void replaceEditor(JS_IJ_EXAMPLE, "js")} title="Example: a Fiji JavaScript script (importClass(Packages.ij.IJ) …)">ImageJ example</button>}
        {(lang === "ijm" || lang === "jython") && <button className="btn sm" onClick={() => useStore.getState().invoke("script.ij_command_list")} title="Table of the supported run(…) commands and macro functions">Command list</button>}
        {isJvmLang(lang) && <button className="btn sm" onClick={() => void replaceEditor(DEFAULT_CODE[lang], lang)} title={`Example: a Fiji ${SCRIPT_LANGS.find((l) => l.id === lang)?.label.replace(/ \(.*\)$/, "")} script (imp, IJ.run, IJ.log, a #@ parameter)`}>Example</button>}
        {isJvmLang(lang) && <button className="btn sm" onClick={() => useStore.getState().invoke("script.jvm_bridge_status")} title="What the Java bridge needs and has: the [jvm] extra, a JDK, ij.jar and the language jars, whether the JVM is running">Bridge status</button>}
      </div>
      {lang === "jython" && (
        <div className="small dim script-hint">
          Python with the ImageJ 1.x classes – <code>from ij import IJ, ImagePlus</code>, <code>from ij.plugin.frame import RoiManager</code> … – run by Lumen's Python,
          not a JVM (Groovy, BeanShell and Rhino are in the language list: they need the Java bridge); <code>#@</code> parameters are asked for before the run (see Scripting ▸ Running Fiji scripts).
        </div>
      )}
      {isJvmLang(lang) && (
        <div className="small dim script-hint">
          Runs on the server's JVM with the real ImageJ 1.x – needs the Java bridge: <code>pip install "lumen-imaging[jvm]"</code> and a JDK 11+ (<code>JAVA_HOME</code>,
          <code>java</code> on PATH or Homebrew OpenJDK); the first run downloads <code>ij.jar</code> and the language jars through Maven (or set <code>LUMEN_IJ_JAR</code> /
          <code>LUMEN_JVM_CLASSPATH</code> to Fiji's jars). The active image is <code>imp</code> (its changes come back with undo), <code>show()</code>-ed images become new images,
          output, <code>IJ.log</code> and the Results table are copied back; the ROI Manager needs <code>LUMEN_JVM_HEADLESS=0</code>
          {(lang === "jruby" || lang === "clojure") && <> and this language <code>LUMEN_JVM_LANGUAGES={lang}</code></>}. See Scripting ▸ JVM languages.
        </div>
      )}
      {out && <div className="console-out" role="log">{out.stdout}{out.error && <span className="err">{out.error}</span>}{!out.stdout && !out.error && <span className="dim">(no output)</span>}</div>}
      {paramPrompt && (
        <ScriptParamsDialog title={`Script parameters – ${path ? baseName(path) : "untitled" + langExt(lang)}`} params={paramPrompt.params} initial={paramMemory.current}
          onSubmit={(v) => paramPrompt.resolve(v)} onCancel={() => paramPrompt.resolve(null)} />
      )}
      {/* E7-14: title line (count, copy), then the export buttons wrapping below it – nothing is cut off at the default inspector width */}
      <div className="recorder-head">
        <div className="recorder-title">
          <label title={recorderInfo ? `This browser tab's recorder, started ${recorderInfo.started_iso.replace("T", " ")} – it survives a reload (the tab keeps its client id)` : "The macro recorder"}>Recorder <span className="dim">({nRecorded} line{nRecorded === 1 ? "" : "s"})</span></label>
          <button className="btn sm" onClick={() => { navigator.clipboard.writeText(macro.join("\n")); toast("Recorded commands copied", "success"); }} disabled={!nRecorded} title="Copy the recorded Python lines" aria-label="Copy the recorded lines"><Copy size={12} /></button>
        </div>
        <div className="recorder-actions">
        <button className="btn sm" onClick={() => void replaceEditor((rec.marker ? [rec.marker, ...rec.lines] : rec.lines).join("\n") + "\n", "python")} disabled={!nRecorded} title="Put the recorded steps into the editor as Python">→ Python</button>
        <button className="btn sm" onClick={() => void replaceEditor(recorderToJs(rec.marker ? [rec.marker, ...rec.lines] : rec.lines), "js")} disabled={!nRecorded} title="Put the recorded steps into the editor as JavaScript (async lumen calls)">→ JavaScript</button>
        <button className="btn sm" onClick={() => void api.recorderIjm().then((r) => {
          const text = r.script ?? r.macro ?? (r.lines || []).join("\n");
          if (!text) throw new Error("empty");
          const skipped = r.skipped?.length ?? 0;
          const removed = r.removed ?? 0;
          if (skipped || r.notes?.length || removed) toast(`${r.converted ?? "?"} of ${r.count ?? "?"} recorded lines converted to ImageJ macro${removed ? `; ${removed} undone step${removed === 1 ? "" : "s"} left out` : ""}${skipped ? `; ${skipped} kept as // comments (no ImageJ equivalent)` : ""}${r.notes?.length ? `; ${r.notes.length} note${r.notes.length === 1 ? "" : "s"} in the script` : ""}`, "info");
          return replaceEditor(text.replace(/\n?$/, "\n"), "ijm");
        }).catch((e) => toast(`ImageJ-macro export of the recorder is not available on this server${e?.message && e.message !== "Not Found" && e.message !== "empty" ? ` (${e.message})` : ""} – use → Python, or the Command list for the run(\"…\") names`, "info"))} disabled={!nRecorded} title="Put the recorded steps into the editor as an ImageJ macro; lines without an ImageJ equivalent become // comments that say why">→ IJM</button>
        <button className="btn sm" onClick={() => void api.clearRecorder().then((r) => { toast(`Recorder cleared${typeof r?.cleared === "number" ? ` (${r.cleared} line${r.cleared === 1 ? "" : "s"})` : ""} – a new session marker starts it`, "info"); return refresh(); }).catch((e) => toast(`Could not clear the recorder: ${e?.message || e}`, "error"))} disabled={!nRecorded} title="Empty this tab's recorder (the editor text stays)">Clear</button>
        </div>
      </div>
      <div className={"macro-lines" + (large ? " large" : "")} ref={macroRef} role="log" aria-label="Recorded commands" aria-live="off">
        {rec.marker && <span className="macro-marker">{rec.marker}{"\n"}</span>}
        {rec.lines.length ? (large ? rec.lines : rec.lines.slice(-60)).join("\n") : "Every command you apply in this tab is recorded here as Python (display changes, undo and the image each command ran on included)."}
      </div>
    </div>
  );
}
