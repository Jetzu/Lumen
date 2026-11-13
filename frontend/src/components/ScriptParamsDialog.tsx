/**
 * The prompt for a script's `#@` parameters (SciJava script parameters, see lumen/core/ijshim/params.py): one
 * field per promptable parameter (numbers, booleans, strings, choices, files and folders); images, tables and
 * services are supplied by the runner and only listed. Enter runs, Escape cancels.
 */
import { useRef, useState } from "react";
import { FolderOpen, X } from "lucide-react";
import { useStore } from "../store";
import { PROMPT_TYPES, type ScriptParamDescriptor } from "../ijScript";
import { useFocusTrap } from "../focusTrap";

interface Props {
  title: string;
  params: ScriptParamDescriptor[];
  /** Values from the previous run (prefill). */
  initial?: Record<string, unknown>;
  onSubmit: (values: Record<string, unknown>) => void;
  onCancel: () => void;
}

const promptable = (p: ScriptParamDescriptor) => PROMPT_TYPES.has(p.type) && p.io !== "output" && p.visibility?.toUpperCase() !== "MESSAGE";

function initialValue(p: ScriptParamDescriptor, prev: unknown): unknown {
  if (prev !== undefined && prev !== null) return prev;
  if (p.default !== undefined && p.default !== null) return p.default;
  if (p.type === "bool") return false;
  if (p.type === "choice") return p.choices?.[0] ?? "";
  return "";
}

export default function ScriptParamsDialog({ title, params, initial, onSubmit, onCancel }: Props) {
  const setFileDialog = useStore((s) => s.setFileDialog);
  const fields = params.filter(promptable);
  const auto = params.filter((p) => !promptable(p) && p.io !== "output" && p.visibility?.toUpperCase() !== "MESSAGE");
  const messages = params.filter((p) => p.visibility?.toUpperCase() === "MESSAGE" || p.type === "message");
  const [values, setValues] = useState<Record<string, unknown>>(() => Object.fromEntries(fields.map((p) => [p.name, initialValue(p, initial?.[p.name])])));
  const firstRef = useRef<HTMLInputElement | HTMLSelectElement | null>(null);
  // the first parameter field takes focus on open, Tab cycles inside, Esc cancels, the Run button / editor that
  // opened the prompt gets the focus back; isolate keeps the editor's and the global shortcuts out of it
  const dialogRef = useRef<HTMLDivElement>(null);
  const trap = useFocusTrap(dialogRef, { onEscape: onCancel, initialFocus: () => firstRef.current, isolate: true });

  const set = (name: string, v: unknown) => setValues((cur) => ({ ...cur, [name]: v }));
  const submit = () => {
    const out: Record<string, unknown> = {};
    for (const p of fields) {
      const v = values[p.name];
      if (p.type === "int" || p.type === "float") {
        if (v === "" || v === undefined || v === null) continue; // the runner uses the declared default
        const n = Number(v);
        if (!Number.isFinite(n)) continue;
        out[p.name] = p.type === "int" ? Math.round(n) : n;
      } else out[p.name] = v;
    }
    onSubmit(out);
  };
  const onKeyDown = (e: React.KeyboardEvent) => {
    trap.onKeyDown(e); // Escape → onCancel, Tab cycles, everything else stays inside the dialog
    // Enter in a field runs; on a focused button it activates that button (Cancel must not run the script)
    if (e.key === "Enter" && !e.defaultPrevented && !(e.target instanceof HTMLTextAreaElement) && !(e.target instanceof HTMLButtonElement)) {
      e.preventDefault();
      submit();
    }
  };
  const browse = (p: ScriptParamDescriptor) =>
    setFileDialog({
      mode: "pick-open",
      title: `Choose ${p.label}`,
      onPick: (path) => set(p.name, p.type === "directory" ? path.replace(/[\\/][^\\/]*$/, "") : path),
    });

  return (
    <div className="modal-backdrop" onMouseDown={onCancel}>
      <div className="dialog script-params" role="dialog" aria-modal="true" aria-labelledby="script-params-title" tabIndex={-1} ref={dialogRef} onMouseDown={(e) => e.stopPropagation()} onKeyDown={onKeyDown}>
        <div className="dialog-head">
          <h3 id="script-params-title">{title}</h3>
          <span className="grow" />
          <button className="icon-btn" title="Cancel" aria-label="Cancel" onClick={onCancel}><X size={15} /></button>
        </div>
        <div className="dialog-body">
          {messages.map((p) => <div key={p.name} className="small">{String(p.default ?? p.label)}</div>)}
          {fields.map((p, i) => {
            const v = values[p.name];
            const id = `sp-${p.name}`;
            const ref = i === 0 ? (el: HTMLInputElement | HTMLSelectElement | null) => { firstRef.current = el; } : undefined;
            let input: React.ReactNode;
            if (p.type === "bool") {
              input = <input id={id} ref={ref as React.Ref<HTMLInputElement>} type="checkbox" checked={Boolean(v)} onChange={(e) => set(p.name, e.target.checked)} />;
            } else if (p.type === "choice") {
              input = (
                <select id={id} ref={ref as React.Ref<HTMLSelectElement>} value={String(v ?? "")} onChange={(e) => set(p.name, e.target.value)}>
                  {(p.choices ?? []).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              );
            } else if (p.type === "int" || p.type === "float") {
              input = (
                <input id={id} ref={ref as React.Ref<HTMLInputElement>} type="number" value={v === undefined || v === null ? "" : String(v)} min={p.min} max={p.max}
                  step={p.step ?? (p.type === "int" ? 1 : "any")} onChange={(e) => set(p.name, e.target.value)} />
              );
            } else {
              input = (
                <span className="sp-file">
                  <input id={id} ref={ref as React.Ref<HTMLInputElement>} type="text" value={String(v ?? "")} onChange={(e) => set(p.name, e.target.value)}
                    placeholder={p.type === "directory" ? "folder path" : p.type === "file" ? "file path" : ""} />
                  {(p.type === "file" || p.type === "directory") && (
                    <button type="button" className="btn sm" onClick={() => browse(p)} title={p.type === "directory" ? "Pick a file inside the folder" : "Pick a file on the server"} aria-label={`Browse for ${p.label}`}>
                      <FolderOpen size={12} />
                    </button>
                  )}
                </span>
              );
            }
            return (
              <label key={p.name} className="sp-row" htmlFor={id} title={p.description || `${p.name} (${p.java_type ?? p.type})`}>
                <span className="sp-label">{p.label}</span>
                {input}
              </label>
            );
          })}
          {auto.length > 0 && (
            <div className="small dim">Supplied by Lumen: {auto.map((p) => `${p.name} (${p.java_type ?? p.type})`).join(", ")}</div>
          )}
        </div>
        <div className="dialog-foot">
          <span className="small dim">The script's #@ parameters; the values are kept for the next run.</span>
          <span className="grow" />
          <button className="btn" onClick={onCancel}>Cancel</button>
          <button className="btn primary" onClick={submit}>Run</button>
        </div>
      </div>
    </div>
  );
}
