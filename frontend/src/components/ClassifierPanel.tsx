/**
 * Inspector ▸ Pixel Classifier – the Trainable Weka Segmentation GUI.
 *
 * Classes (add / rename / colour / delete, the active one takes new traces), traces drawn with the freehand /
 * rectangle / oval / line tools in trace mode, the brush (viewport hook) or the "Classifier Trace Tool" pen on
 * the tool rail – stored as ROI-manager entries `class:<name>` –, Train / Apply (plane, stack, other image,
 * folder) / Toggle overlay / Get probability / Create result / Save / Load / Settings / Info, and a status line
 * from the `Classifier classes` / `Classifier summary` tables the ops publish in the session state.
 * Everything runs through the registry ops (`segment.*`, `segment3d.*`) and the ROI-manager REST calls.
 */
import React, { useEffect, useMemo, useState } from "react";
import { Eye, FolderOpen, Info, Layers, Pencil, Play, Plus, Save, Settings2, Trash2, Wand2 } from "lucide-react";
import { api } from "../api";
import { useStore } from "../store";
import type { ImageSummary, ParamSpec, Roi, TableData } from "../types";
import {
  addTrace, applyParams, CAPTURE_TYPES, DEFAULT_SETTINGS, OPS, runQuiet, startOverlaySync, TRACE_TOOL, TRACE_TOOL_OP, traceClass, traceName, tracesOf,
  trainParams, useClassifierStore, type ClassifierMode, type ClassifierSettings,
} from "../classifierStore";
import NumberInput from "./NumberInput";

type Scope = "plane" | "stack";

function kv(table?: TableData): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (table) for (const r of table.rows) out[String(r[0])] = r[1];
  return out;
}

function choicesOf(params: ParamSpec[] | undefined, name: string): string[] {
  return params?.find((p) => p.name === name)?.choices || [];
}

function num(v: string, fallback: number): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

export default function ClassifierPanel({ img }: { img: ImageSummary }) {
  const state = useStore((s) => s.state);
  const ops = useStore((s) => s.ops);
  const meta = useStore((s) => s.meta);
  const tool = useStore((s) => s.tool);
  const showManagerRois = useStore((s) => s.showManagerRois);
  const selection = useStore((s) => s.selections[img.id] ?? null);
  const cs = useClassifierStore();
  const [scope, setScope] = useState<Scope>("plane");
  const [editing, setEditing] = useState<number | null>(null);
  const [otherId, setOtherId] = useState<string>("");

  const isStack = img.shape.z > 1;
  const mode: ClassifierMode = cs.mode === "3d" && isStack ? "3d" : "2d";
  const ids = OPS[mode];
  const trainSpec = ops[ids.train];
  const haveOps = !!trainSpec && !!ops[ids.apply];
  const rois = state?.rois;
  const traces = useMemo(() => tracesOf(rois || [], img), [rois, img]);
  const counts = useMemo(() => {
    const m = new Map<string, Roi[]>();
    for (const r of traces) {
      const k = (traceClass(r) || "").toLowerCase();
      m.set(k, [...(m.get(k) || []), r]);
    }
    return m;
  }, [traces]);
  const classesTable = state?.tables?.[`${ids.prefix} classes`];
  const summary = kv(state?.tables?.[`${ids.prefix} summary`]);
  const trained = !!classesTable && !!state?.tables?.[`${ids.prefix} summary`];
  const hasOverlay = img.overlay.some((o) => (o as unknown as { source?: string }).source === "classifier");
  const others = useMemo(() => (state?.images || []).filter((im) => im.id !== img.id && !("classifier_overlay_of" in (im.metadata || {}))), [state?.images, img.id]);
  const active = cs.classes[cs.active];
  const busy = cs.running !== null;
  const penActive = tool === TRACE_TOOL;
  const features = mode === "3d" ? cs.settings.features3d : cs.settings.features;

  useEffect(() => startOverlaySync(), []);

  // the pen tool follows the active class, its colour and the trace width
  useEffect(() => {
    if (active) useStore.getState().setToolParams(TRACE_TOOL_OP, { class_name: active.name, width: cs.traceWidth, color: active.color });
  }, [active, cs.traceWidth]);

  // trace mode: a selection drawn with the freehand / rect / oval / line tools becomes a trace of the active class
  useEffect(() => {
    if (!cs.traceMode) return;
    let last: Roi | null = useStore.getState().selections[img.id] ?? null;
    return useStore.subscribe((s) => {
      const sel = s.selections[img.id] ?? null;
      if (sel === last) return;
      last = sel;
      if (!sel || !CAPTURE_TYPES.has(sel.type)) return;
      if (sel.id && s.state?.rois.some((r) => r.id === sel.id)) return; // picked from the ROI manager, not drawn
      const c = useClassifierStore.getState();
      const cls = c.classes[c.active];
      const im = s.image(img.id);
      if (!cls || !im) return;
      void addTrace(im, sel, cls, c.traceStyle, c.traceWidth).then((ok) => {
        if (ok) useStore.getState().setSelection(img.id, null);
      });
    });
  }, [cs.traceMode, img.id]);

  const toast = (msg: string, kind: "info" | "error" | "success" = "info") => useStore.getState().toast(msg, kind);
  const run = async (what: string, fn: () => Promise<unknown>) => {
    if (busy) return;
    cs.setRunning(what);
    try {
      await fn();
    } finally {
      useClassifierStore.getState().setRunning(null);
    }
  };

  // ---------------------------------------------------------------- classes
  const rename = async (i: number, name: string) => {
    setEditing(null);
    const old = cs.classes[i];
    const nm = name.trim();
    if (!old || !nm || nm.toLowerCase() === old.name.toLowerCase()) {
      if (old && nm && nm !== old.name) cs.renameClass(i, nm);
      return;
    }
    cs.renameClass(i, nm);
    const mine = counts.get(old.name.toLowerCase()) || [];
    for (const r of mine) await api.patchRoi(r.id!, { name: traceName(nm) });
    if (mine.length) await useStore.getState().refresh();
  };
  const recolor = async (i: number, color: string) => {
    const old = cs.classes[i];
    cs.recolorClass(i, color);
    const mine = old ? counts.get(old.name.toLowerCase()) || [] : [];
    for (const r of mine) await api.patchRoi(r.id!, { color });
    if (mine.length) await useStore.getState().refresh();
  };
  const remove = async (i: number) => {
    const c = cs.classes[i];
    if (!c || cs.classes.length <= 2) return;
    const mine = counts.get(c.name.toLowerCase()) || [];
    if (mine.length && !window.confirm(`Delete class '${c.name}' and its ${mine.length} trace${mine.length === 1 ? "" : "s"}?`)) return;
    if (mine.length) {
      await api.deleteRois(mine.map((r) => r.id!));
      await useStore.getState().refresh();
    }
    cs.removeClass(i);
  };
  const adoptModelClasses = () => {
    if (!classesTable) return;
    cs.setClasses(classesTable.rows.map((r, i) => ({ name: String(r[1]), color: /^#[0-9a-f]{6}$/i.test(String(r[2])) ? String(r[2]).toLowerCase() : cs.classes[i]?.color || "#ffffff" })));
  };
  const modelClassesDiffer = trained && !!classesTable && (classesTable.rows.length !== cs.classes.length || classesTable.rows.some((r, i) => String(r[1]).toLowerCase() !== cs.classes[i]?.name.toLowerCase()));

  // ---------------------------------------------------------------- traces
  const addSelection = async () => {
    if (!selection) return toast("Draw a selection first", "error");
    if (!active) return;
    if (await addTrace(img, selection, active, cs.traceStyle, cs.traceWidth)) useStore.getState().setSelection(img.id, null);
  };
  const toggleTraceMode = () => {
    const on = !cs.traceMode;
    cs.setTraceMode(on);
    const st = useStore.getState();
    if (on && (st.tool === "pointer" || st.tool === "hand")) st.setTool("freehand");
    if (on && !st.showManagerRois) st.setShowManagerRois(true);
  };
  const togglePen = () => {
    const st = useStore.getState();
    if (!ops[TRACE_TOOL_OP]) return toast("The Classifier Trace Tool is not registered", "error");
    st.setTool(penActive ? "pointer" : TRACE_TOOL);
    if (!penActive && !st.showManagerRois) st.setShowManagerRois(true);
  };
  const selectTraces = (name: string) => {
    const mine = counts.get(name.toLowerCase()) || [];
    const st = useStore.getState();
    st.setSelectedRoiIds(mine.map((r) => r.id!));
    if (!st.showManagerRois) st.setShowManagerRois(true);
  };
  const clearTraces = async (name: string) => {
    const mine = counts.get(name.toLowerCase()) || [];
    if (!mine.length) return;
    if (!window.confirm(`Delete the ${mine.length} '${name}' trace${mine.length === 1 ? "" : "s"}?`)) return;
    await api.deleteRois(mine.map((r) => r.id!));
    useStore.getState().setSelectedRoiIds([]);
    await useStore.getState().refresh();
  };

  // ---------------------------------------------------------------- classifier actions
  const train = () => run("train", async () => {
    if (!traces.length) return toast("Draw training traces first (trace mode, the pen tool or the brush)", "error");
    const t0 = performance.now();
    const res = await runQuiet(ids.train, trainParams(mode, cs.settings, cs.classes, "Overlay"), img.id);
    if (res) {
      cs.setLastTrainMs(performance.now() - t0);
      toast(res.message || "Trained", "success");
    }
  });
  const all = scope === "stack";
  const applyOverlay = (whole: boolean) => run("apply", async () => {
    const res = await runQuiet(ids.apply, applyParams(mode, cs.settings, "Overlay", whole), img.id);
    if (res?.message) toast(res.message, "info");
  });
  const applyOther = () => run("apply", async () => {
    if (!otherId) return toast("Choose the image to classify", "error");
    await useStore.getState().runOp(ids.apply, applyParams(mode, cs.settings, "Labels", true, otherId), { imageId: img.id, roi: null });
  });
  const toggleOverlay = () => run("overlay", async () => {
    await runQuiet("segment.classifier_overlay", { action: "Toggle", opacity: cs.settings.overlayOpacity }, img.id);
  });
  const getProbability = () => run("apply", async () => {
    await useStore.getState().runOp(ids.apply, applyParams(mode, cs.settings, "Probabilities", all), { imageId: img.id, roi: null });
  });
  const createResult = () => run("apply", async () => {
    await useStore.getState().runOp(ids.apply, applyParams(mode, cs.settings, "Labels", all), { imageId: img.id, roi: null });
  });
  const modelPath = (name: string) => (meta?.home ? `${meta.home.replace(/[\\/]+$/, "")}/${name}` : name);
  const save = () => ops[ids.save] && useStore.getState().openSheet(ops[ids.save], { path: String(summary.File || modelPath(mode === "3d" ? "classifier3d.joblib" : "classifier.joblib")) });
  const load = () => ops[ids.load] && useStore.getState().openSheet(ops[ids.load], { path: String(summary.File || modelPath(mode === "3d" ? "classifier3d.joblib" : "classifier.joblib")) });
  const info = () => run("info", async () => {
    await useStore.getState().runOp("segment.classifier_info", { kind: mode === "3d" ? "3-D" : "2-D" }, { roi: null });
  });
  const folder = () => {
    const spec = ops["segment.apply_classifier_folder"];
    if (spec) useStore.getState().openSheet(spec, { use_3d: mode === "3d", smooth: cs.settings.smooth });
  };

  // ---------------------------------------------------------------- status line
  const sampleText = classesTable ? classesTable.rows.map((r) => `${r[1]} ${r[3]}`).join(" · ") : "";
  const trainedOn = summary["Trained on"] ? String(summary["Trained on"]) : "";
  const status = trained
    ? `${summary.Classifier || "Random forest"} · ${summary.Features ?? "?"} features · ${summary["Training samples"] ?? "?"} samples (${sampleText}) · ${summary["Training time (s)"] ?? "?"} s` +
      (summary.Accuracy && summary.Accuracy !== "n/a" ? ` · accuracy ${summary.Accuracy}` : "") +
      (trainedOn && trainedOn !== img.name ? ` · trained on ${trainedOn}` : "")
    : `No ${mode === "3d" ? "3-D " : ""}classifier yet: draw traces of every class and press Train.`;

  if (!haveOps) return <div className="hint-text">The pixel classifier commands are not registered.</div>;

  return (
    <div className="clf-panel">
      {isStack && (
        <div className="clf-modes seg" title="2-D classifies plane by plane; 3-D learns volumetric features of the z stack">
          <button className={mode === "2d" ? "on" : ""} onClick={() => cs.setMode("2d")}>2-D</button>
          <button className={mode === "3d" ? "on" : ""} onClick={() => cs.setMode("3d")}>3-D</button>
        </div>
      )}
      <div className="clf-classes">
        {cs.classes.map((c, i) => {
          const n = (counts.get(c.name.toLowerCase()) || []).length;
          return (
            <div key={i} className={"clf-class" + (i === cs.active ? " sel" : "")} onClick={() => cs.setActive(i)} onDoubleClick={() => setEditing(i)} title={`${c.name}: ${n} trace${n === 1 ? "" : "s"} – click to make it the active class, double-click to rename`}>
              <input type="color" className="clf-swatch" value={c.color} onClick={(e) => e.stopPropagation()} onChange={(e) => void recolor(i, e.target.value)} title="Class colour" />
              {editing === i ? (
                <input autoFocus className="clf-name-edit" defaultValue={c.name} onClick={(e) => e.stopPropagation()} onBlur={(e) => void rename(i, e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") void rename(i, (e.target as HTMLInputElement).value); if (e.key === "Escape") setEditing(null); }} />
              ) : (
                <span className="clf-name">{i === 0 ? `${c.name} (background)` : c.name}</span>
              )}
              <button className="clf-count" onClick={(e) => { e.stopPropagation(); selectTraces(c.name); }} title="Select the traces of this class in the ROI manager">{n}</button>
              <button className="icon-btn clf-x" onClick={(e) => { e.stopPropagation(); void clearTraces(c.name); }} disabled={!n} title="Delete the traces of this class"><Layers size={12} /></button>
              <button className="icon-btn clf-x" onClick={(e) => { e.stopPropagation(); void remove(i); }} disabled={cs.classes.length <= 2} title="Delete the class"><Trash2 size={12} /></button>
            </div>
          );
        })}
      </div>
      <div className="btn-row" style={{ marginTop: 4 }}>
        <button className="btn sm" onClick={() => cs.addClass()} title="Add a class (traces of the active class are drawn in its colour)"><Plus size={12} /> Add class</button>
        {modelClassesDiffer && <button className="btn sm" onClick={adoptModelClasses} title="The trained / loaded classifier has other classes: use its class list and colours here">Use model's classes</button>}
      </div>

      <div className="clf-trace-row">
        <button className={"btn sm" + (cs.traceMode ? " primary" : "")} onClick={toggleTraceMode} title="Trace mode: every selection drawn with the freehand, rectangle, oval or line tool (and every brush stroke) becomes a trace of the active class"><Pencil size={12} /> Trace mode</button>
        <button className={"btn sm" + (penActive ? " primary" : "")} onClick={togglePen} title="Pen tool (tool rail): drag = stroke, click = dot, ⌥-click removes the nearest trace of the class"><Wand2 size={12} /> Pen</button>
        <span className="seg" title="Freehand traces: a pen line of the given width, or the filled outline">
          <button className={cs.traceStyle === "line" ? "on" : ""} onClick={() => cs.setTraceStyle("line")}>Line</button>
          <button className={cs.traceStyle === "area" ? "on" : ""} onClick={() => cs.setTraceStyle("area")}>Area</button>
        </span>
        <label className="clf-width" title="Width of pen traces (px)">
          <NumberInput min={1} max={200} step={1} integer value={cs.traceWidth} aria-label="Trace width (px)" onChange={(v) => cs.setTraceWidth(v)} /> px
        </label>
      </div>
      <div className="btn-row" style={{ marginTop: 0 }}>
        <button className="btn sm" onClick={() => void addSelection()} disabled={!selection || !active} title="Add the current selection as a trace of the active class (polygons and polylines too)">Add selection</button>
        <button className={"btn sm" + (showManagerRois ? " primary" : "")} onClick={() => useStore.getState().setShowManagerRois(!showManagerRois)} title="Show the traces (all ROI-manager entries) on the image"><Eye size={12} /> Traces</button>
      </div>

      <div className="clf-actions">
        <button className="btn sm primary" onClick={train} disabled={busy || !traces.length} title="Learn the classifier from the traces and show the class overlay of the current plane"><Play size={12} /> {cs.running === "train" ? "Training…" : "Train"}</button>
        <span className="seg" title="Scope of Apply / Get probability / Create result">
          <button className={scope === "plane" ? "on" : ""} onClick={() => setScope("plane")}>{mode === "3d" ? "Frame" : "Plane"}</button>
          <button className={scope === "stack" ? "on" : ""} onClick={() => setScope("stack")} disabled={mode === "3d" ? img.shape.t <= 1 : !isStack && img.shape.t <= 1}>{mode === "3d" ? "All frames" : "Stack"}</button>
        </span>
        <button className="btn sm" onClick={() => applyOverlay(all)} disabled={busy || !trained} title="Classify and show the result as a transparent class overlay">Apply</button>
        <button className={"btn sm" + (hasOverlay ? " primary" : "")} onClick={toggleOverlay} disabled={busy} title="Show / hide the class overlay (Weka's Toggle overlay)">Overlay</button>
        <button className="btn sm" onClick={getProbability} disabled={busy || !trained} title="Probability maps, one channel per class">Get probability</button>
        <button className="btn sm" onClick={createResult} disabled={busy || !trained} title="The classification as a label image">Create result</button>
      </div>
      <div className="clf-actions">
        <select value={otherId} onChange={(e) => setOtherId(e.target.value)} title="Apply the classifier to another open image (label image of its whole stack)">
          <option value="">Other image…</option>
          {others.map((im) => <option key={im.id} value={im.id}>{im.name}</option>)}
        </select>
        <button className="btn sm" onClick={applyOther} disabled={busy || !trained || !otherId}>Apply to image</button>
        <button className="btn sm" onClick={folder} disabled={!trained || !ops["segment.apply_classifier_folder"]} title="Classify every image file of a folder (batch, with progress)"><FolderOpen size={12} /> Folder…</button>
      </div>
      <div className="clf-actions">
        <button className="btn sm" onClick={save} disabled={!trained || !ops[ids.save]} title="Save the classifier (estimator, features, classes, colours) as a .joblib file"><Save size={12} /> Save</button>
        <button className="btn sm" onClick={load} disabled={!ops[ids.load]} title="Load a saved classifier"><FolderOpen size={12} /> Load</button>
        <button className={"btn sm" + (cs.settingsOpen ? " primary" : "")} onClick={() => cs.setSettingsOpen(!cs.settingsOpen)} title="Training features and classifier settings"><Settings2 size={12} /> Settings</button>
        <button className="btn sm" onClick={info} disabled={busy || !trained} title="Feature importances, class list and training summary tables"><Info size={12} /> Info</button>
      </div>
      <div className="clf-status" title={status}>{status}</div>
      {cs.settingsOpen && <SettingsForm mode={mode} img={img} params={trainSpec.params} settings={cs.settings} features={features} onPatch={cs.patchSettings} />}
      <div className="hint-text">
        Traces are ROI-manager entries named <span className="mono">class:&lt;name&gt;</span> on the plane they were drawn on. Weka loop: trace → Train → look at the overlay → add traces where it is wrong → Train again.
      </div>
    </div>
  );
}

function SettingsForm({ mode, img, params, settings: s, features, onPatch }: { mode: ClassifierMode; img: ImageSummary; params: ParamSpec[]; settings: ClassifierSettings; features: string[]; onPatch: (p: Partial<ClassifierSettings>) => void }) {
  const featureChoices = choicesOf(params, "features");
  const classifierChoices = choicesOf(params, "classifier");
  const featureKey = mode === "3d" ? "features3d" : "features";
  const toggleFeature = (f: string) => {
    const next = features.includes(f) ? features.filter((x) => x !== f) : featureChoices.filter((x) => x === f || features.includes(x));
    onPatch({ [featureKey]: next.length ? next : [featureChoices[0]] } as Partial<ClassifierSettings>);
  };
  const c = s.classifier;
  const forest = c === "Random forest" || c === "Extra trees";
  const field = (label: string, key: keyof ClassifierSettings, min: number, max: number, step = 1, title?: string) => (
    <label className="clf-field" title={title}>
      <span>{label}</span>
      <NumberInput min={min} max={max} step={step} integer={Number.isInteger(step)} value={Number(s[key])} onChange={(v) => onPatch({ [key]: v } as Partial<ClassifierSettings>)} />
    </label>
  );
  return (
    <div className="clf-settings">
      <div className="clf-grid">
        <label className="clf-field"><span>Classifier</span>
          <select value={c} onChange={(e) => onPatch({ classifier: e.target.value })}>{classifierChoices.map((x) => <option key={x} value={x}>{x}</option>)}</select>
        </label>
        {(forest || c === "Gradient boosting") && field(forest ? "Trees" : "Iterations", "nTrees", 1, 1000)}
        {(forest || c === "Gradient boosting") && field("Max depth (0 = ∞)", "maxDepth", 0, 100)}
        {forest && field("Features / node (0 = √F)", "featuresPerNode", 0, 10000, 1, "Random features tried at every split, like Weka's 'features per node'")}
        {c === "SVM" && field("SVM max samples", "svmMaxSamples", 100, 200000, 100)}
        {c === "k-NN" && field("Neighbours", "kNeighbors", 1, 100)}
        {c === "MLP" && field("Hidden units", "hiddenUnits", 1, 1024)}
        {field("Max samples / class", "maxSamplesPerClass", 100, 5000000, 1000)}
        <label className="clf-field clf-check"><input type="checkbox" checked={s.balanceClasses} onChange={(e) => onPatch({ balanceClasses: e.target.checked })} /><span>Balance classes</span></label>
        {img.shape.c > 1 && !img.rgb && (
          <label className="clf-field"><span>Channels</span>
            <select value={s.channels} onChange={(e) => onPatch({ channels: e.target.value as ClassifierSettings["channels"] })}>
              <option value="Current channel">Current channel</option><option value="All channels">All channels</option>
            </select>
          </label>
        )}
        {field("Min sigma", "minSigma", 0.5, 32, 0.5)}
        {field("Max sigma", "maxSigma", 0.5, mode === "3d" ? 16 : 32, 1)}
        {mode === "2d" && field("Membrane thickness", "membraneThickness", 1, 15)}
        {mode === "2d" && field("Membrane patch size", "membranePatchSize", 3, 99, 2)}
        {field("Smooth probabilities", "smooth", 0, 10, 0.5, "Gaussian sigma applied to the probability maps before the result")}
        <label className="clf-field" title="Opacity of the class overlay"><span>Overlay opacity</span>
          <input type="range" min={0.05} max={1} step={0.05} value={s.overlayOpacity} onChange={(e) => onPatch({ overlayOpacity: num(e.target.value, s.overlayOpacity) })} />
        </label>
      </div>
      <div className="clf-features">
        {featureChoices.map((f) => (
          <label key={f} className={"chip" + (features.includes(f) ? " on" : " off")}><input type="checkbox" checked={features.includes(f)} onChange={() => toggleFeature(f)} />{f}</label>
        ))}
      </div>
      <div className="btn-row" style={{ marginTop: 4 }}>
        <button className="btn sm" onClick={() => onPatch({ ...DEFAULT_SETTINGS })} title="Weka's defaults (random forest on Gaussian, DoG, Sobel, Laplacian, Hessian and Variance features, sigma 1–8)">Reset defaults</button>
      </div>
    </div>
  );
}
