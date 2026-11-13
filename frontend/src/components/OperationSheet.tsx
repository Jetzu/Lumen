import { useEffect, useMemo, useRef, useState } from "react";
import { X, Sparkles, Loader2, CircleHelp } from "lucide-react";
import { api } from "../api";
import { useStore } from "../store";
import { chooseStride } from "../render/stride";
import type { ParamSpec } from "../types";
import { fmt } from "./RangeSlider";
import { doAction } from "../actions";
import { useFocusTrap, wasKeyboard } from "../focusTrap";
import { displayName, ijMacroCall } from "../menus";
import { fmtMB, newImageBytes } from "../store";
import { groupLuts, lutLabel } from "../lutGroups";
import { lutGradientCss } from "../render/compositor";
import NumberInput from "./NumberInput";

/** "between 0 and 200" / "at least 1" / "at most 20000" for an int / float parameter's limits. */
function rangeText(spec: ParamSpec): string {
  if (spec.min !== undefined && spec.max !== undefined) return `between ${fmt(spec.min)} and ${fmt(spec.max)}`;
  if (spec.min !== undefined) return `at least ${fmt(spec.min)}`;
  if (spec.max !== undefined) return `at most ${fmt(spec.max)}`;
  return "";
}

/** The parameter's value is outside its declared range (the server rejects it too, E7-7). */
function outOfRange(spec: ParamSpec, v: unknown): boolean {
  if ((spec.type !== "int" && spec.type !== "float") || typeof v !== "number" || !Number.isFinite(v)) return false;
  return (spec.min !== undefined && v < spec.min) || (spec.max !== undefined && v > spec.max);
}

export default function OperationSheet() {
  const sheet = useStore((s) => s.sheet)!;
  const setSheet = useStore((s) => s.setSheet);
  const closeSheet = useStore((s) => s.closeSheet);
  const applySheet = useStore((s) => s.applySheet);
  const setPreview = useStore((s) => s.setPreview);
  const img = useStore((s) => s.activeImage());
  const images = useStore((s) => s.state?.images || []);
  const selection = useStore((s) => (img ? s.selections[img.id] : null));
  const view = useStore((s) => (img ? s.views[img.id] : undefined));
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [previewing, setPreviewing] = useState(false);
  const timer = useRef<number | null>(null);
  const reqId = useRef(0);
  const op = sheet.op;
  const canPreview = op.preview && !!img;
  const isStack = img ? img.shape.z > 1 || img.shape.t > 1 : false;
  const multiChannel = img ? img.shape.c > 1 && !img.rgb : false;

  // Focus management (docs/USABILITY.md M24).  Opened from the keyboard (palette ⏎, a menu ⏎, a shortcut) the
  // sheet takes focus on its first control and hands it back to the opener (the element focused before the
  // palette, the menu trigger, the canvas) on close; opened with the mouse it leaves the focus where it is,
  // so a pointer user keeps drawing selections and panning.  Tab / Shift-Tab cycle inside either way and Esc
  // closes.  The sheet is NOT modal (preview on the canvas, a selection drawn while it is open narrows the
  // command), hence no aria-modal.  Re-evaluated per command: the component stays mounted when the palette
  // replaces one sheet with another.
  const sheetRef = useRef<HTMLDivElement>(null);
  const openedByKeyboard = useMemo(() => wasKeyboard(), [op.id]);
  const trap = useFocusTrap(sheetRef, { onEscape: closeSheet, initialFocus: openedByKeyboard ? "first" : false, returnFocus: openedByKeyboard, resetKey: op.id });

  // live preview
  useEffect(() => {
    if (!canPreview || !sheet.preview || !img) {
      ++reqId.current; // a request still in flight must not land after the toggle (ghost preview pixels)
      setPreviewing(false); // its `finally` is skipped now, so reset the spinner here
      setPreview(null, null);
      return;
    }
    if (sheet.busy) {
      // Apply is running: a pending preview would queue behind it on the session lock and be computed
      // on the already-modified image.  Keep what is on screen; closeSheet() clears it on success.
      ++reqId.current;
      setPreviewing(false);
      return;
    }
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      const id = ++reqId.current;
      setPreviewing(true);
      try {
        const stride = chooseStride(img, view?.scale ?? 1);
        const planes = await api.preview(op.id, sheet.params, img.id, selection ?? null, sheet.scope, stride);
        if (id === reqId.current) {
          setPreview(planes, img.id);
          setSheet({ error: null });
        }
      } catch (e: any) {
        if (id === reqId.current) setSheet({ error: e.message });
      } finally {
        if (id === reqId.current) setPreviewing(false);
      }
    }, 140);
    return () => {
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [sheet.params, sheet.preview, sheet.scope, sheet.busy, op.id, img?.id, img?.rev, img?.display.t, img?.display.z, img?.display.c, selection, canPreview]); // eslint-disable-line react-hooks/exhaustive-deps

  // unmount (Cancel / Esc / Apply): drop the in-flight request too, otherwise its response would
  // re-populate previewPlanes after closeSheet() cleared them and the viewport keep drawing the
  // cancelled preview until another sheet is opened and closed
  useEffect(() => () => { ++reqId.current; setPreview(null, null); }, [setPreview]);

  const set = (name: string, value: any) => setSheet({ params: { ...sheet.params, [name]: value } });
  const visible = op.params.filter((p) => !p.advanced || showAdvanced);
  const hasAdvanced = op.params.some((p) => p.advanced);
  const invalid = op.params.filter((p) => outOfRange(p, sheet.params[p.name]));
  // fields whose typed text is not a number ("abc", "2,5,1"): Apply stays disabled until it parses (E10-3)
  const [badText, setBadText] = useState<Record<string, boolean>>({});
  useEffect(() => setBadText({}), [op.id]);
  const badTextNames = op.params.filter((p) => badText[p.name]).map((p) => p.label);
  const blocked = invalid.length > 0 || badTextNames.length > 0;
  // Analyze Particles without a threshold: the fix is a button in the error, not a hidden advanced option (E5-5 / E6-2)
  const particlesFix = op.id === "analyze.particles" && !!sheet.error && /threshold/i.test(sheet.error) && !sheet.params.auto_threshold && op.params.some((p) => p.name === "auto_threshold");
  // a 400 "sigma must be between 0.5 and 1000 (got 5000)" names the parameter first: the message also sits under that field
  const errorParam = sheet.error ? op.params.find((p) => sheet.error!.toLowerCase().startsWith(p.name.toLowerCase() + " ") || sheet.error!.toLowerCase().startsWith(p.label.toLowerCase() + " ")) : undefined;
  const title = displayName(op);
  // File ▸ New: the size the image will take, before Apply (E5-18); the server refuses more than half of the RAM
  const sizeNote = op.id === "file.new" ? (() => {
    const bytes = newImageBytes(sheet.params);
    const ram = useStore.getState().state?.memory_limit_bytes;
    const big = bytes > Math.min(512 * 1048576, ram ? ram / 4 : Infinity);
    return { text: `Size: ${fmtMB(bytes)} in memory${ram ? ` (${(100 * bytes / ram).toFixed(bytes / ram < 0.01 ? 2 : 0)}% of ${fmtMB(ram)} RAM)` : ""}${big ? " – large: a confirmation follows, and every undo step costs the same again" : ""}`, big };
  })() : null;
  const nPlanes = img ? img.shape.z * img.shape.t * (multiChannel ? img.shape.c : 1) : 0;

  return (
    <div className="sheet" role="dialog" aria-labelledby="sheet-title" aria-describedby={op.description ? "sheet-desc" : undefined} tabIndex={-1} ref={sheetRef} onKeyDown={(e) => {
      if (e.key === "Enter" && !(e.target instanceof HTMLTextAreaElement)) {
        if (e.target instanceof HTMLButtonElement) {
          // a focused button answers for itself (Enter on Cancel must not apply); while a run is in progress
          // nothing may fire, so a repeated / busy Enter cannot fall through to the Apply button
          if (sheet.busy) e.preventDefault();
        } else {
          // preventDefault unconditionally so a repeated / busy Enter cannot fall through;
          // one run per sheet: ignore OS auto-repeat, a run in progress and an IME composition commit
          e.preventDefault();
          // Enter obeys the same gate as the Apply button: an out-of-range value or text that is not a number ("abc") never runs (E11)
          if (!sheet.busy && !e.repeat && !e.nativeEvent.isComposing && !blocked) void applySheet();
        }
      }
      trap.onKeyDown(e); // Tab / Shift-Tab cycle inside, Esc closes (consumed: the global Esc chain must not act twice)
    }}>
      <div className="sheet-head">
        <h3 id="sheet-title">{title}</h3>
        {previewing && <Loader2 size={14} className="spin" role="status" aria-label="Previewing…" style={{ color: "var(--accent)" }} />}
        <button className={"icon-btn" + (showHelp ? " active" : "")} onClick={() => setShowHelp(!showHelp)} aria-expanded={showHelp} aria-controls="sheet-help" title="What the parameters mean (defaults, units, ranges)" aria-label="Help on this command"><CircleHelp size={15} /></button>
        <button className="icon-btn" onClick={closeSheet} title="Cancel (Esc)" aria-label="Cancel (Esc)"><X size={15} /></button>
      </div>
      {op.description && <div className="sheet-desc" id="sheet-desc">{op.description}</div>}
      {showHelp && (
        <div className="sheet-help" id="sheet-help" role="region" aria-label="Parameter help">
          {op.params.length === 0 && <div className="hint-text">This command takes no parameters.</div>}
          {op.params.map((p) => (
            <div key={p.name} className="sheet-help-row">
              <b>{p.label}</b>{p.advanced ? <span className="dim"> (advanced)</span> : null}
              <span className="sheet-help-meta">
                {p.default !== undefined && p.default !== null && p.default !== "" && <>default <code>{String(p.default)}</code></>}
                {p.unit && <> · {p.unit}</>}
                {(p.type === "int" || p.type === "float") && rangeText(p) && <> · {rangeText(p)}</>}
                {p.choices && p.choices.length > 0 && p.choices.length <= 8 && <> · {p.choices.join(" / ")}</>}
              </span>
              {p.help && <div className="sheet-help-text">{p.help}</div>}
            </div>
          ))}
          {/* the ImageJ-macro equivalent, when the server sends it (E7-20); older servers have no ij_commands */}
          {op.ij_commands && op.ij_commands.length > 0 && (
            <div className="sheet-help-row sheet-help-ij">
              <b>ImageJ</b>
              <span className="sheet-help-meta">{op.ij_commands.map((c, i) => <code key={i} title="The equivalent call in an ImageJ macro (→ IJM uses it)">{ijMacroCall(c)}</code>)}</span>
            </div>
          )}
          <div className="sheet-help-foot">
            <span className="mono dim small">{op.id}</span>
            <span className="grow" />
            <button className="btn sm" onClick={() => doAction("helpOnCommand", op.id)}>Command reference…</button>
          </div>
        </div>
      )}
      <div className="sheet-body">
        {visible.map((p) => <ParamField key={op.id + ":" + p.name} spec={p} value={sheet.params[p.name]} onChange={(v) => set(p.name, v)} images={images} opId={op.id} serverError={errorParam === p ? sheet.error! : undefined} onTextInvalid={(bad) => setBadText((b) => (!!b[p.name] === bad ? b : { ...b, [p.name]: bad }))} />)}
        {errorParam && errorParam.advanced && !showAdvanced && <div className="help">The rejected value is under Advanced options.</div>}
        {sizeNote && <div className={"hint-text sheet-size" + (sizeNote.big ? " big" : "")} role="status">{sizeNote.text}</div>}
        {hasAdvanced && <button className="advanced-toggle" aria-expanded={showAdvanced} onClick={() => setShowAdvanced(!showAdvanced)}>{showAdvanced ? "▾ Hide advanced" : "▸ Advanced options"}</button>}
        {op.scope_param && img && (isStack || multiChannel) && (
          <div className="field">
            <label htmlFor="sheet-scope">Apply to</label>
            <div className="ctrl">
              <select id="sheet-scope" value={sheet.scope} onChange={(e) => setSheet({ scope: e.target.value })}>
                <option value="plane">Current plane only{multiChannel ? ` (channel ${img.display.c + 1})` : ""}</option>
                {multiChannel && <option value="slice">All channels of this {isStack ? "slice" : "image"}</option>}
                {isStack && multiChannel && <option value="channel">Channel {img.display.c + 1}, all slices/frames</option>}
                {isStack && <option value="stack">Entire stack ({nPlanes} planes)</option>}
              </select>
            </div>
            {isStack && <div className="help">{sheet.scope === "stack" ? `Processes all ${nPlanes} planes (ImageJ's “Process all images?” = Yes).` : `Only the plane shown now – choose “Entire stack” to process all ${nPlanes} planes.`}</div>}
          </div>
        )}
        {sheet.note && <div className="hint-text sheet-note">{sheet.note}</div>}
        {op.roi_aware && op.kind === "plane" && selection && (selection.type === "rect" || selection.type === "oval" || selection.type === "polygon" || selection.type === "freehand" || selection.type === "mask") && (
          <div className="hint-text">Applied inside the current selection only.</div>
        )}
      </div>
      {(invalid.length > 0 || badTextNames.length > 0) && !sheet.error && (
        <div className="sheet-error" role="alert">{[...invalid.map((p) => `${p.label} must be ${rangeText(p)} (got ${fmt(sheet.params[p.name])})`), ...badTextNames.map((l) => `${l} is not a number`)].join("; ")}</div>
      )}
      {sheet.error && (
        <div className="sheet-error" role="alert">
          {sheet.error}
          {particlesFix && (
            <div className="sheet-error-actions">
              <button className="btn sm primary" onClick={() => { setSheet({ params: { ...sheet.params, auto_threshold: true }, error: null }); window.setTimeout(() => void applySheet(), 0); }}>Auto-threshold and run</button>
              <button className="btn sm" onClick={() => { closeSheet(); doAction("threshold"); }}>Open the Threshold panel</button>
            </div>
          )}
        </div>
      )}
      <div className="sheet-foot">
        {canPreview && (
          <label className="switch" title="Preview on the current plane without changing the image">
            <input type="checkbox" checked={sheet.preview} onChange={(e) => setSheet({ preview: e.target.checked })} /> <Sparkles size={12} /> Preview
          </label>
        )}
        <span className="grow" />
        <button className="btn" onClick={closeSheet}>Cancel</button>
        <button className="btn primary" disabled={sheet.busy || blocked} title={blocked ? (badTextNames.length ? `${badTextNames.join(", ")}: type a number first` : "Fix the out-of-range value first") : undefined} onClick={() => void applySheet()}>{sheet.busy ? "Working…" : op.kind === "analysis" ? "Run" : "Apply"}</button>
      </div>
    </div>
  );
}

/**
 * Path-like string parameters get a Browse… button (docs/USABILITY.md M12).  Detected from the parameter
 * name (`path`, `folder`, `input_folder`, `*_dir`, `*_file` …) or a label ending in "folder" / "path" / "file";
 * "output" / "save" / "export" wording means the user names a new file (save picker).
 */
function pathParamKind(spec: ParamSpec, opId: string): "file" | "file-out" | "folder" | null {
  if (spec.type !== "string") return null;
  const name = spec.name.toLowerCase();
  const label = (spec.label || "").trim().toLowerCase();
  const isFolder = /(^|_)(folder|dir|directory)(_|$)/.test(name) || /(^|\s)(folder|directory)$/.test(label);
  const isFile = /(^|_)(path|file|filename)(_|$)/.test(name) || /(^|\s)(path|file)$/.test(label);
  if (!isFolder && !isFile) return null;
  if (isFolder) return "folder";
  const out = /(^|_)(output|out|save|export|target|dest)(_|$)/.test(name) || /\b(output|save|export|written|destination)\b/i.test(spec.help || "") || /(^|\.)(save|export|write)/.test(opId.replace(/^.*\./, "")) || /export|save/.test(opId);
  return out ? "file-out" : "file";
}

/** Split a server path into its folder and base name ("" when there is no separator). */
function splitPath(p: string): { dir: string; base: string } {
  const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return i < 0 ? { dir: "", base: p } : { dir: p.slice(0, i) || p.slice(0, i + 1), base: p.slice(i + 1) };
}

function BrowseButton({ spec, value, kind, onPick }: { spec: ParamSpec; value: any; kind: "file" | "file-out" | "folder"; onPick: (path: string) => void }) {
  const browse = () => {
    const cur = typeof value === "string" ? value.trim() : "";
    const st = useStore.getState();
    if (kind === "folder") {
      st.setFileDialog({ mode: "pick-folder", title: `${spec.label} – choose a folder`, initialPath: cur || undefined, onPick });
      return;
    }
    const { dir, base } = splitPath(cur);
    if (kind === "file-out") st.setFileDialog({ mode: "pick-save", title: `${spec.label} – choose where to save`, initialPath: dir || undefined, filename: base || undefined, onPick });
    else st.setFileDialog({ mode: "pick-open", title: `${spec.label} – choose a file`, initialPath: dir || undefined, onPick });
  };
  const what = kind === "folder" ? "folder" : "file";
  return (
    <button type="button" className="btn sm browse" onClick={browse} title={`Choose ${kind === "folder" ? "a folder" : "a file"} on the server for “${spec.label}”`} aria-label={`Browse for ${spec.label} ${what}`}>
      Browse…
    </button>
  );
}

/**
 * One parameter row (M26, label association): the visible label names the control (`aria-labelledby` on the
 * row label's text span, `htmlFor` for click-to-focus), the help text describes it (`aria-describedby`), and a
 * range slider carries aria-valuemin / max / now / text.  Ids are "p-<param name>"; the sheet shows one
 * command at a time, so they are unique on the page.
 */
function ParamField({ spec, value, onChange, images, opId, serverError, onTextInvalid }: { spec: ParamSpec; value: any; onChange: (v: any) => void; images: { id: string; name: string }[]; opId: string; serverError?: string; onTextInvalid?: (bad: boolean) => void }) {
  const t = spec.type;
  const id = "p-" + spec.name;
  const labelId = id + "-label";
  const hintId = spec.help ? id + "-help" : undefined;
  const a11y = { "aria-labelledby": labelId, "aria-describedby": hintId };
  const fieldError = serverError ? <div className="field-error" role="alert">{serverError}</div> : null;
  if (t === "bool") {
    return (
      <div className="field check" title={spec.help}>
        <input type="checkbox" id={id} {...a11y} checked={!!value} onChange={(e) => onChange(e.target.checked)} />
        <label htmlFor={id}><span id={labelId}>{spec.label}</span>{spec.help && <span id={hintId} className="help" style={{ marginLeft: 6 }}>{spec.help}</span>}</label>
      </div>
    );
  }
  let ctrl: React.ReactNode;
  if (t === "int" || t === "float") {
    const hasRange = spec.min !== undefined && spec.max !== undefined && spec.max - spec.min <= 100000;
    const step = spec.step ?? (t === "int" ? 1 : 0.1);
    ctrl = <NumberField spec={spec} value={value} onChange={onChange} hasRange={hasRange} step={step} id={id} labelId={labelId} hintId={hintId} onTextInvalid={onTextInvalid} />;
  } else if (t === "choice") {
    const st = useStore.getState();
    const isLut = spec.name === "lut" && (spec.choices || []).filter((c) => st.meta?.luts.includes(c)).length >= 8;
    if (isLut) {
      // LUT choices come grouped (ImageJ / Fiji / perceptual / labels …) with the near-duplicates labelled, and the
      // chosen table is previewed as a gradient (E6-9 / E7-9 / E7-13)
      const names = spec.choices || [];
      const extra = names.filter((c) => !st.meta?.luts.includes(c)); // "Current" …
      const groups = groupLuts(names.filter((c) => st.meta?.luts.includes(c)), st.lutGrouping);
      const cur = typeof value === "string" ? value : "";
      ctrl = (
        <div className="lut-choice">
          <select id={id} {...a11y} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>
            {extra.map((c) => <option key={c} value={c}>{c}</option>)}
            {groups.map((g) => (
              <optgroup key={g.name} label={g.name}>{g.luts.map((c) => <option key={c} value={c}>{lutLabel(c, st.lutGrouping)}</option>)}</optgroup>
            ))}
          </select>
          {st.meta?.luts.includes(cur) && <span className="lut-swatch" aria-hidden="true" style={{ background: lutGradientCss(st.luts, cur) }} title={cur} />}
        </div>
      );
    } else ctrl = <select id={id} {...a11y} value={value ?? ""} onChange={(e) => onChange(e.target.value)}>{(spec.choices || []).map((c) => <option key={c} value={c}>{c}</option>)}</select>;
  } else if (t === "multichoice") {
    const arr: string[] = Array.isArray(value) ? value : [];
    const meta = useStore.getState().meta;
    const labelOf = (c: string) => meta?.measurements.find((m) => m.id === c)?.label || c;
    ctrl = (
      <div className="multi" role="group" id={id} {...a11y}>
        {(spec.choices || []).map((c) => (
          <label key={c}><input type="checkbox" checked={arr.includes(c)} onChange={(e) => onChange(e.target.checked ? [...arr, c] : arr.filter((x) => x !== c))} />{labelOf(c)}</label>
        ))}
      </div>
    );
  } else if (t === "image") {
    ctrl = (
      <select id={id} {...a11y} value={value ?? ""} onChange={(e) => onChange(e.target.value || null)}>
        <option value="">— none —</option>
        {images.map((im) => <option key={im.id} value={im.id}>{im.name}</option>)}
      </select>
    );
  } else if (t === "text") {
    const code = /lines|script|macro|code/.test(spec.name);
    ctrl = <textarea id={id} {...a11y} rows={code ? 8 : 4} spellCheck={code ? false : undefined} wrap={code ? "off" : undefined} value={value ?? ""} onChange={(e) => onChange(e.target.value)} style={{ flex: 1 }} />;
  } else if (t === "color") {
    ctrl = <input type="color" id={id} {...a11y} value={value ?? "#ffd700"} onChange={(e) => onChange(e.target.value)} />;
  } else {
    const pathKind = pathParamKind(spec, opId);
    ctrl = (
      <>
        <input type="text" id={id} {...a11y} value={value ?? ""} onChange={(e) => onChange(e.target.value)} spellCheck={pathKind ? false : undefined} />
        {pathKind && <BrowseButton spec={spec} value={value} kind={pathKind} onPick={onChange} />}
      </>
    );
  }
  return (
    <div className={"field" + (serverError ? " has-error" : "")}>
      {/* a group (multichoice) is not a labelable element: no htmlFor there */}
      <label htmlFor={t === "multichoice" ? undefined : id}><span id={labelId}>{spec.label}</span>{spec.unit && <span className="unit">{spec.unit}</span>}{(t === "int" || t === "float") && (spec.min !== undefined || spec.max !== undefined) && <span className="unit">{spec.min !== undefined && spec.max !== undefined ? `${fmt(spec.min)}–${fmt(spec.max)}` : spec.min !== undefined ? `≥ ${fmt(spec.min)}` : `≤ ${fmt(spec.max!)}`}</span>}</label>
      <div className="ctrl">{ctrl}</div>
      {fieldError}
      {spec.help && <div className="help" id={hintId}>{spec.help}</div>}
    </div>
  );
}

/**
 * int / float control.  The raw text is kept while editing: a type=number input reports "" for any
 * intermediate value ("-", ".", "1e", cleared) and a controlled numeric value makes React rewrite the
 * DOM from props after every keystroke, which swallowed the minus sign / decimal point and reset the
 * field to the default.  Only finite numbers are committed to sheet.params; an empty field falls back
 * to the last committed value (or the default) on blur.
 */
function NumberField({ spec, value, onChange, hasRange, step, id, labelId, hintId, onTextInvalid }: { spec: ParamSpec; value: any; onChange: (v: any) => void; hasRange: boolean; step: number; id: string; labelId: string; hintId?: string; onTextInvalid?: (bad: boolean) => void }) {
  const parse = (s: string) => (spec.type === "int" ? parseInt(s, 10) : parseFloat(s));
  const committed = typeof value === "number" && Number.isFinite(value) ? value : undefined;
  const sliderValue = Number(value ?? spec.min ?? 0);
  const valueNow = Number.isFinite(sliderValue) ? sliderValue : undefined;
  const bad = outOfRange(spec, value);
  return (
    <>
      {hasRange && (
        <input type="range" aria-labelledby={labelId} aria-describedby={hintId} aria-valuemin={spec.min} aria-valuemax={spec.max} aria-valuenow={valueNow} aria-valuetext={valueNow === undefined ? undefined : `${fmt(valueNow)}${spec.unit ? " " + spec.unit : ""}`}
          min={spec.min} max={spec.max} step={step} value={sliderValue} onChange={(e) => onChange(parse(e.target.value))} />
      )}
      {/* with a slider the number box is the "<label> value" field; the row label (htmlFor) names it otherwise */}
      {/* a text field with a decimal point whatever the browser locale (E7-26): NumberInput accepts "," too and keeps min / max / step for ↑ ↓ */}
      <NumberInput id={id} aria-label={hasRange ? `${spec.label} value` : spec.label} aria-labelledby={hasRange ? undefined : labelId} aria-describedby={hintId} invalid={bad} title={bad ? `${spec.label} must be ${rangeText(spec)}` : undefined}
        min={spec.min} max={spec.max} step={step} integer={spec.type === "int"} value={committed}
        onChange={onChange}
        onValidity={(ok) => onTextInvalid?.(!ok)}
        onInvalidBlur={() => {
          // an emptied field falls back to the last committed value, else the default
          const v = committed ?? spec.default;
          if (v !== value) onChange(v);
        }} />
    </>
  );
}
