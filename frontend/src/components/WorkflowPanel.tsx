import { useEffect, useMemo, useRef, useState } from "react";
import { Check, Copy, FastForward, Play, RotateCcw, SkipForward, Undo2 } from "lucide-react";
import { useStore } from "../store";
import { doAction } from "../actions";
import type { ImageSummary, Roi, SessionState } from "../types";

/**
 * Guided, step-by-step workflows (the wizard GUIs of TrackMate / Weka / CellProfiler), served by
 * GET /api/workflows (lumen/core/workflows.py).  Each step points at one registered command; "Run…"
 * opens the ordinary operation sheet with the workflow's defaults, and the step is ticked off when the
 * command shows up in the recorder.  Progress lives in localStorage "lumen.workflow".
 */

export interface WorkflowSummary {
  id: string;
  title: string;
  description: string;
  requires: string[];
  tags: string[];
  n_steps: number;
  n_required: number;
  ops: string[];
}

export interface WorkflowStep {
  id: string;
  title: string;
  help: string;
  op: string | null;
  params: Record<string, any>;
  preview: boolean;
  optional: boolean;
  requires: string[];
  outputs_hint: string;
  action: string | null;
  script: string | null;
  op_name: string | null;
  op_kind: string | null;
  op_preview: boolean;
}

export interface Workflow {
  id: string;
  title: string;
  description: string;
  requires: string[];
  tags: string[];
  steps: WorkflowStep[];
}

interface Progress {
  id: string | null;
  done: string[];
  skipped: string[];
  /** Parameters the user actually ran a step with (from the sheet) – reused on "Run again" and in the script. */
  params: Record<string, Record<string, any>>;
  /** The image that was active when the workflow was started ("$original" in step params); captured at the
   *  first step instead when the workflow was picked before an image was open. */
  sourceImage: string | null;
  /** Its name: image ids are per session, so after a reload / server restart the reopened original is re-matched by name. */
  sourceName: string | null;
}

const LS_KEY = "lumen.workflow";
const EMPTY: Progress = { id: null, done: [], skipped: [], params: {}, sourceImage: null, sourceName: null };
const ACTION_LABELS: Record<string, string> = { save: "Save", saveAs: "Save As…", open: "Open…", undo: "Undo (⌘Z)", redo: "Redo (⇧⌘Z)" };
/** Requirements that block "Run" when unmet; the others (binary, labels, spots, classifier) can only be guessed, so they just warn. */
const HARD = new Set(["image", "selection", "rois", "threshold", "stack", "timelapse", "multichannel", "tracks"]);

function loadProgress(): Progress {
  try {
    const raw = JSON.parse(localStorage.getItem(LS_KEY) || "null");
    if (raw && typeof raw === "object" && (raw.id === null || typeof raw.id === "string")) {
      return {
        id: raw.id,
        done: Array.isArray(raw.done) ? raw.done.filter((x: unknown) => typeof x === "string") : [],
        skipped: Array.isArray(raw.skipped) ? raw.skipped.filter((x: unknown) => typeof x === "string") : [],
        params: raw.params && typeof raw.params === "object" ? raw.params : {},
        sourceImage: typeof raw.sourceImage === "string" ? raw.sourceImage : null,
        sourceName: typeof raw.sourceName === "string" ? raw.sourceName : null,
      };
    }
  } catch {
    /* corrupt entry */
  }
  return EMPTY;
}

/** Saved step parameters with every occurrence of image id `from` replaced by `to` (the source was re-matched after a reload). */
function remapId(params: Record<string, Record<string, any>>, from: string | null, to: string): Record<string, Record<string, any>> {
  if (!from || from === to) return params;
  const out: Record<string, Record<string, any>> = {};
  for (const [sid, p] of Object.entries(params)) {
    const q: Record<string, any> = {};
    for (const [k, v] of Object.entries(p)) q[k] = v === from ? to : v;
    out[sid] = q;
  }
  return out;
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = r.statusText;
    try {
      const j = await r.json();
      msg = j.error || j.detail || msg;
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return (await r.json()) as T;
}

/** null when the requirement holds, otherwise a short hint of what is missing. */
function unmet(req: string, img: ImageSummary | undefined, selection: Roi | null, state: SessionState | null, doneOps: Set<string>): string | null {
  const needImg = "open an image first";
  switch (req) {
    case "none":
      return null;
    case "image":
      return img ? null : needImg;
    case "selection":
      return !img ? needImg : selection ? null : "draw a selection on the image";
    case "rois":
      return state?.rois.length ? null : "add ROIs to the ROI manager";
    case "threshold":
      return !img ? needImg : state?.thresholds[img.id] ? null : "set a threshold first (previous step)";
    case "binary":
      return !img ? needImg : img.binary_hint ? null : "needs a binary (0/255) mask";
    case "labels":
      return !img ? needImg : null;
    case "stack":
      return !img ? needImg : img.shape.z > 1 ? null : "needs a Z stack";
    case "timelapse":
      return !img ? needImg : img.shape.t > 1 ? null : "needs a time series (T > 1)";
    case "multichannel":
      return !img ? needImg : img.shape.c > 1 && !img.rgb ? null : "needs two or more channels";
    case "spots":
      return doneOps.has("tracking.detect_spots") || state?.rois.length ? null : "detect spots first";
    case "tracks":
      return state?.tables && state.tables["Tracks"] ? null : "link spots into tracks first";
    case "classifier":
      return doneOps.has("segment.train_classifier") || doneOps.has("segment.load_classifier") ? null : "train (or load) a classifier first";
    default:
      return img ? null : needImg;
  }
}

export default function WorkflowPanel() {
  const img = useStore((s) => s.activeImage());
  const state = useStore((s) => s.state);
  const selection = useStore((s) => (img ? s.selections[img.id] ?? null : null));
  const ops = useStore((s) => s.ops);
  const toast = useStore((s) => s.toast);
  const setWorkflowProgress = useStore((s) => s.setWorkflowProgress);
  const macro = state?.macro;

  const [list, setList] = useState<WorkflowSummary[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [wf, setWf] = useState<Workflow | null>(null);
  const [progress, setProgress] = useState<Progress>(loadProgress);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [armed, setArmed] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const cache = useRef(new Map<string, Workflow>());

  // ---- data
  useEffect(() => {
    let alive = true;
    getJson<WorkflowSummary[]>("/api/workflows")
      .then((l) => alive && setList(l))
      .catch((e: any) => alive && setListError(e.message || String(e)));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    const id = progress.id;
    if (!id) {
      setWf(null);
      return;
    }
    const hit = cache.current.get(id);
    if (hit) {
      setWf(hit);
      return;
    }
    let alive = true;
    getJson<Workflow>(`/api/workflows/${encodeURIComponent(id)}`)
      .then((w) => {
        cache.current.set(id, w);
        if (alive) setWf(w);
      })
      .catch((e: any) => {
        if (!alive) return;
        toast(`Workflow unavailable: ${e.message || e}`, "error");
        setProgress((p) => ({ ...p, id: null }));
      });
    return () => {
      alive = false;
    };
  }, [progress.id]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(progress));
    } catch {
      /* quota / private mode */
    }
  }, [progress]);

  // ---- derived
  const doneSet = useMemo(() => new Set(progress.done), [progress.done]);
  const skipSet = useMemo(() => new Set(progress.skipped), [progress.skipped]);
  const doneOps = useMemo(() => new Set(wf ? wf.steps.filter((s) => s.op && doneSet.has(s.id)).map((s) => s.op as string) : []), [wf, doneSet]);
  const current = useMemo(() => wf?.steps.find((s) => !doneSet.has(s.id) && !skipSet.has(s.id)) ?? null, [wf, doneSet, skipSet]);
  const total = wf ? wf.steps.filter((s) => !skipSet.has(s.id)).length : 0;
  const nDone = wf ? wf.steps.filter((s) => doneSet.has(s.id)).length : 0;
  const allDone = !!wf && current === null;

  useEffect(() => {
    // the badge of the (possibly collapsed) section header; kept when the panel unmounts since progress persists
    setWorkflowProgress(wf ? { title: wf.title, done: nDone, total } : null);
  }, [wf, nDone, total, setWorkflowProgress]);

  const unmetFor = (step: WorkflowStep, live = false, liveDoneOps?: Set<string>): { hint: string; hard: boolean } | null => {
    // live = read the store directly (inside "Run remaining", where the render-time snapshot is stale)
    const st = live ? useStore.getState() : null;
    const im = st ? st.activeImage() : img;
    const sel = st ? (im ? st.selections[im.id] ?? null : null) : selection;
    const ss = st ? st.state : state;
    for (const r of step.requires) {
      const u = unmet(r, im, sel, ss, liveDoneOps ?? doneOps);
      if (u) return { hint: u, hard: HARD.has(r) };
    }
    return null;
  };
  const workflowUnmet = wf ? wf.requires.map((r) => unmet(r, img, selection, state, doneOps)).find((u) => u) ?? null : null;

  // ---- progress edits
  const start = (id: string | null) => {
    setProgress({ id, done: [], skipped: [], params: {}, sourceImage: img?.id ?? null, sourceName: img?.name ?? null });
    setExpanded(null);
    setArmed(null);
  };
  const markDone = (sid: string, params?: Record<string, any>) => {
    setProgress((p) => ({
      ...p,
      done: p.done.includes(sid) ? p.done : [...p.done, sid],
      skipped: p.skipped.filter((x) => x !== sid),
      params: params ? { ...p.params, [sid]: params } : p.params,
    }));
    setExpanded(null);
  };
  const skip = (sid: string) => {
    setProgress((p) => ({ ...p, skipped: [...p.skipped.filter((x) => x !== sid), sid], done: p.done.filter((x) => x !== sid) }));
    setExpanded(null);
  };
  const reopen = (sid: string) => {
    setProgress((p) => ({ ...p, done: p.done.filter((x) => x !== sid), skipped: p.skipped.filter((x) => x !== sid) }));
    setExpanded(sid);
  };

  /** The open image "$original" stands for: by id, else by name (ids change on reload / server restart). */
  const findSource = (p: Progress): ImageSummary | undefined => {
    const imgs = useStore.getState().state?.images ?? [];
    return imgs.find((i) => i.id === p.sourceImage) ?? (p.sourceName ? imgs.find((i) => i.name === p.sourceName) : undefined);
  };
  /**
   * The source image, re-synced to its current id when it was re-matched by name.  A workflow picked before
   * an image was open adopts the active image at the FIRST step (nothing done yet) – never later: by the
   * measure step the active image is the label image, and intensities would silently be the label ids.
   */
  const ensureSource = (): ImageSummary | undefined => {
    const found = findSource(progress);
    if (found) {
      if (found.id !== progress.sourceImage) setProgress((p) => ({ ...p, sourceImage: found.id, sourceName: found.name, params: remapId(p.params, p.sourceImage, found.id) }));
      return found;
    }
    if (!progress.sourceImage && progress.done.length === 0) {
      const im = useStore.getState().activeImage();
      if (im) {
        setProgress((p) => ({ ...p, sourceImage: im.id, sourceName: im.name }));
        return im;
      }
    }
    return undefined;
  };
  /**
   * Workflow defaults ← the user's last values for this step, with $tokens resolved to image ids.
   * "$original" (and a saved source id) resolves to `source`; without one it becomes null – never the
   * active image – and `missing` tells the caller to warn or stop.
   */
  const resolveParams = (step: WorkflowStep, source: ImageSummary | undefined): { params: Record<string, any>; missing: boolean } => {
    const out: Record<string, any> = { ...step.params, ...(progress.params[step.id] || {}) };
    const active = useStore.getState().activeImage();
    let missing = false;
    for (const [k, v] of Object.entries(out)) {
      if (v === "$original" || (progress.sourceImage && v === progress.sourceImage)) {
        out[k] = source?.id ?? null;
        if (!source) missing = true;
      } else if (v === "$active") out[k] = active?.id ?? null;
    }
    return { params: out, missing };
  };
  const sourceGone = (step: WorkflowStep) => `${step.title}: the image this workflow started from is no longer open – reopen it, or press Start over with it active`;
  /** The reverse for the script: the source image id becomes "$original" again so the script stays portable. */
  const scriptParams = (): Record<string, Record<string, any>> => {
    const out: Record<string, Record<string, any>> = {};
    for (const [sid, params] of Object.entries(progress.params)) {
      const p: Record<string, any> = {};
      for (const [k, v] of Object.entries(params)) p[k] = progress.sourceImage && v === progress.sourceImage ? "$original" : v;
      out[sid] = p;
    }
    return out;
  };

  // ---- tick steps off when their command shows up in the recorder (sheet Apply, menu, palette, script)
  const latest = useRef({ wf, armed, current, markDone });
  latest.current = { wf, armed, current, markDone };
  const macroRef = useRef<{ len: number; last: string | undefined }>({ len: macro?.length ?? 0, last: macro?.[(macro?.length ?? 1) - 1] });
  const usedRun = useRef(0); // store.lastRun.seq already used to tick a step
  useEffect(() => {
    if (!macro) return;
    const last = macro[macro.length - 1];
    const prev = macroRef.current;
    if (macro.length === prev.len && last === prev.last) return;
    macroRef.current = { len: macro.length, last };
    const { wf: w, armed: a, current: cur, markDone: done } = latest.current;
    if (!w || !last) return;
    const m = /^lumen\.run\("([^"]+)"/.exec(last);
    if (!m) return;
    const opId = m[1];
    const armedStep = a ? w.steps.find((s) => s.id === a) : undefined;
    const target = armedStep && armedStep.op === opId ? armedStep : cur && cur.op === opId ? cur : undefined;
    if (!target) return;
    // Which values were used?  The sheet stays open (busy) until runOp resolves, i.e. after this render, so
    // on the Run… / Apply path its params are exactly what was sent.  Otherwise (menu, palette, Repeat) use
    // store.lastRun, which runOp writes before the refresh that carries the macro line – but never a record
    // that was already consumed, so a line produced by a script cannot be ticked with an older run's values.
    const st = useStore.getState();
    let params: Record<string, any> | undefined;
    if (st.sheet && st.sheet.busy && st.sheet.op.id === opId) params = st.sheet.params;
    else if (st.lastRun && st.lastRun.opId === opId && st.lastRun.seq !== usedRun.current) {
      params = st.lastRun.params;
      usedRun.current = st.lastRun.seq;
    }
    done(target.id, params);
    if (a === target.id) setArmed(null);
  }, [macro]);

  // ---- running
  const runStep = (step: WorkflowStep) => {
    if (!step.op) {
      if (step.action) doAction(step.action);
      return;
    }
    const op = ops[step.op];
    if (!op) return toast(`Unknown command ${step.op}`, "error");
    const u = unmetFor(step);
    if (u && u.hard) return toast(`${step.title}: ${u.hint}`, "error");
    const { params, missing } = resolveParams(step, ensureSource());
    // the sheet still opens (its image field shows "— none —"), so the user can pick the image there
    if (missing) toast(`${sourceGone(step)} – or pick the image in the sheet`, "error");
    setArmed(step.id);
    setExpanded(step.id);
    const st = useStore.getState();
    st.openSheet(op, params);
    if (op.preview) st.setSheet({ preview: !!step.preview });
  };

  const runRemaining = async () => {
    if (!wf || running) return;
    setRunning(true);
    const liveDoneOps = new Set(doneOps);
    const source = ensureSource();
    try {
      for (const step of wf.steps) {
        if (doneSet.has(step.id) || skipSet.has(step.id)) continue;
        if (!step.op) {
          toast(`"${step.title}" is a manual step – do it, mark it done, then continue`, "info");
          setExpanded(step.id);
          break;
        }
        const u = unmetFor(step, true, liveDoneOps);
        if (u && u.hard) {
          toast(`${step.title}: ${u.hint}`, "error");
          setExpanded(step.id);
          break;
        }
        const { params, missing } = resolveParams(step, source);
        if (missing) {
          // hard stop: without the original the measure step would silently read the label image
          toast(sourceGone(step), "error");
          setExpanded(step.id);
          break;
        }
        const res = await useStore.getState().runOp(step.op, params, { scope: "stack" });
        if (!res) {
          setExpanded(step.id);
          break; // the error was toasted by runOp
        }
        liveDoneOps.add(step.op);
        markDone(step.id, params);
      }
    } finally {
      setRunning(false);
    }
  };

  const copyScript = async () => {
    if (!wf) return;
    try {
      const body = { params: scriptParams(), skip: progress.skipped, include_optional: true };
      const r = await getJson<{ script: string }>(`/api/workflows/${encodeURIComponent(wf.id)}/script`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      await navigator.clipboard.writeText(r.script);
      toast("Workflow script copied – paste it into the Script panel or Process Folder…", "success");
    } catch (e: any) {
      toast(e.message || String(e), "error");
    }
  };

  // ---- render
  if (listError) return <div className="hint-text">Workflows are unavailable ({listError}).</div>;
  if (!list) return <div className="hint-text">Loading workflows…</div>;

  return (
    <div className="wf">
      <div className="wf-head">
        <select value={progress.id ?? ""} onChange={(e) => start(e.target.value || null)} title="Pick a guided workflow">
          <option value="">Choose a workflow…</option>
          {list.map((w) => <option key={w.id} value={w.id}>{w.title}</option>)}
        </select>
        {wf && <button className="icon-btn" title="Start over (clears the ticks)" onClick={() => start(progress.id)}><RotateCcw size={14} /></button>}
      </div>

      {!wf && (
        <div className="wf-cards">
          {list.map((w) => {
            const u = w.requires.map((r) => unmet(r, img, selection, state, doneOps)).find((x) => x) ?? null;
            return (
              <button key={w.id} className={"wf-card" + (u ? " blocked" : "")} onClick={() => start(w.id)} title={u ? `Requires: ${u}` : `${w.n_steps} steps`}>
                <span className="wf-card-title">{w.title}<span className="wf-opt">{w.n_steps} steps</span></span>
                <span className="wf-card-desc">{w.description}</span>
                {u && <span className="wf-req">Requires: {u}</span>}
              </button>
            );
          })}
        </div>
      )}

      {wf && (
        <>
          <div className="wf-desc">{wf.description}</div>
          {workflowUnmet && <div className="wf-req">This workflow needs: {workflowUnmet}</div>}
          <div className="wf-progress" title={`${nDone} of ${total} steps done`}><div style={{ width: `${total ? (100 * nDone) / total : 0}%` }} /></div>
          <div className="wf-steps">
            {wf.steps.map((step, i) => {
              const isDone = doneSet.has(step.id);
              const isSkipped = skipSet.has(step.id);
              const isCurrent = current?.id === step.id;
              // null = default (the current step is open); "" = the user collapsed the current step; else the open step's id
              const open = expanded !== null ? expanded === step.id : isCurrent;
              const u = unmetFor(step);
              const blocked = !!u && !isDone && !isSkipped;
              const cls = "wf-step" + (isDone ? " done" : "") + (isSkipped ? " skipped" : "") + (isCurrent ? " current" : "") + (blocked ? " blocked" : "") + (open ? " open" : "");
              return (
                <div key={step.id} className={cls}>
                  <div className="wf-step-head" onClick={() => setExpanded(open ? (isCurrent ? "" : null) : step.id)} title={step.op_name ? `${step.op_name}  [${step.op}]` : "Manual step"}>
                    <span className="wf-mark">{isDone ? <Check size={11} /> : isSkipped ? "–" : i + 1}</span>
                    <span className="wf-title">{step.title}</span>
                    {step.optional && !isDone && !isSkipped && <span className="wf-opt">optional</span>}
                    {!step.op && <span className="wf-opt">manual</span>}
                  </div>
                  {open && (
                    <div className="wf-body">
                      <div className="wf-help">{step.help}</div>
                      {u && <div className="wf-req">Requires: {u.hint}</div>}
                      {step.outputs_hint && <div className="wf-out">{step.outputs_hint}</div>}
                      <div className="wf-actions">
                        {step.op ? (
                          <button className="btn sm primary" disabled={running || !!(u && u.hard)} onClick={() => runStep(step)} title={`Open ${step.op_name} with the workflow's defaults`}>
                            <Play size={11} /> {isDone ? "Run again…" : "Run…"}
                          </button>
                        ) : (
                          step.action && <button className="btn sm primary" onClick={() => doAction(step.action!)}>{ACTION_LABELS[step.action] || step.action}</button>
                        )}
                        {!isDone && (
                          <button className="btn sm" onClick={() => { ensureSource(); markDone(step.id); }} title={step.op ? "Tick off without running (already done by hand)" : "Tick this step off"}>
                            <Check size={11} /> {step.op ? "Done" : "Mark done"}
                          </button>
                        )}
                        {step.optional && !isSkipped && !isDone && <button className="btn sm" onClick={() => skip(step.id)}><SkipForward size={11} /> Skip</button>}
                        {(isDone || isSkipped) && <button className="btn sm" onClick={() => reopen(step.id)} title="Untick this step"><Undo2 size={11} /> Reopen</button>}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <div className="wf-foot">
            <button className="btn sm" disabled={running || allDone || !!workflowUnmet} onClick={() => void runRemaining()} title="Run every remaining step with its defaults, in order (stops at manual steps and errors)">
              <FastForward size={12} /> {running ? "Running…" : "Run remaining with defaults"}
            </button>
            <button className="btn sm" onClick={() => void copyScript()} title="Copy a Python script that replays this workflow (with the values you used)">
              <Copy size={12} /> Copy as script
            </button>
          </div>
          {allDone && <div className="hint-text">All steps done. Copy the script to repeat the analysis, or run it on a folder with Process ▸ Batch ▸ Process Folder…</div>}
        </>
      )}
    </div>
  );
}
