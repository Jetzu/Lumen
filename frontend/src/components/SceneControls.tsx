import React, { useState } from "react";
import { Camera, ChevronDown, ChevronRight, Crosshair, Eye, EyeOff, FolderOpen, Film, Plus, Save, Square, Trash2, Video } from "lucide-react";
import { animationGifUrl, defaultClip, defaultLights, defaultTransform, useScene, type SceneObject, type Transform, type Vec3, type ViewPreset } from "../render/scene";
import type { ImageSummary } from "../types";

export interface LandmarkRow {
  id: string;
  name: string;
  /** calibrated coordinates in the primary image's unit (null while the object is not loaded) */
  cal: Vec3 | null;
  unit: string;
  color: string;
}

export interface RecordState {
  done: number;
  total: number;
  label: string;
}

interface Props {
  img: ImageSummary;
  images: ImageSummary[];
  landmarkRows: LandmarkRow[];
  recording: RecordState | null;
  freehand: boolean;
  lastAnimation: { id: string; fps: number } | null;
  onPreset: (p: ViewPreset) => void;
  onRecord: (axis: "x" | "y" | "z", frames: number, degrees: number, fps: number) => void;
  onStop: () => void;
  onFreehand: (on: boolean, fps: number) => void;
  onSnapshot: (width: number, height: number, supersample: number) => void;
  onSaveScene: () => void;
  onLoadScene: () => void;
  onLandmarksToResults: (addRois: boolean) => void;
}

/** Open / closed state of the sub-sections survives remounts (inspector ↔ large view). */
const openState: Record<string, boolean> = { view: false, slices: false, objects: false, landmarks: true, animation: false };

function Sub({ id, title, badge, children }: { id: string; title: string; badge?: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(!!openState[id]);
  const toggle = () => {
    openState[id] = !open;
    setOpen(!open);
  };
  return (
    <div className={"scn-sub" + (open ? " open" : "")}>
      <button className="scn-sub-head" onClick={toggle} aria-expanded={open}>
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        <span className="scn-sub-title">{title}</span>
        {badge && <span className="badge">{badge}</span>}
      </button>
      {open && <div className="scn-sub-body">{children}</div>}
    </div>
  );
}

function Num({ value, onChange, step = 1, min, max, title, width = 52 }: { value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; title?: string; width?: number }) {
  return <input type="number" className="scn-num" style={{ width }} value={Number.isFinite(value) ? +value.toFixed(4) : 0} step={step} min={min} max={max} title={title} onChange={(e) => onChange(Number(e.target.value))} />;
}

function Slider({ label, value, onChange, min = 0, max = 1, step = 0.01, title, fmt }: { label: string; value: number; onChange: (v: number) => void; min?: number; max?: number; step?: number; title?: string; fmt?: (v: number) => string }) {
  return (
    <label className="scn-slider" title={title}>
      <span className="scn-slider-label">{label}</span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <span className="val">{fmt ? fmt(value) : value.toFixed(2)}</span>
    </label>
  );
}

function TransformEditor({ t, unit, onChange }: { t: Transform; unit: string; onChange: (t: Transform) => void }) {
  const tr = (i: number, v: number) => onChange({ ...t, translate: t.translate.map((x, k) => (k === i ? v : x)) as Vec3 });
  const ro = (i: number, v: number) => onChange({ ...t, rotate: t.rotate.map((x, k) => (k === i ? v : x)) as Vec3 });
  return (
    <div className="scn-transform">
      <div className="scn-row">
        <span className="scn-key" title={`Translation in ${unit}`}>Move</span>
        {[0, 1, 2].map((i) => <Num key={i} value={t.translate[i]} step={1} onChange={(v) => tr(i, v)} title={["x", "y", "z"][i] + ` (${unit})`} />)}
      </div>
      <div className="scn-row">
        <span className="scn-key" title="Rotation in degrees about x, y, z (about the object's centre)">Rotate</span>
        {[0, 1, 2].map((i) => <Num key={i} value={t.rotate[i]} step={5} min={-360} max={360} onChange={(v) => ro(i, v)} title={["x", "y", "z"][i] + " °"} />)}
      </div>
      <div className="scn-row">
        <span className="scn-key">Scale</span>
        <Num value={t.scale} step={0.1} min={0.01} max={100} onChange={(v) => onChange({ ...t, scale: v > 0 ? v : 1 })} />
        <span className="grow" />
        <button className="btn sm" onClick={() => onChange(defaultTransform())} title="Identity transform">Reset</button>
      </div>
    </div>
  );
}

function SlicesEditor({ img, value, onChange }: { img: ImageSummary; value: { pos: Vec3; show: [boolean, boolean, boolean] }; onChange: (v: { pos: Vec3; show: [boolean, boolean, boolean] }) => void }) {
  const n = [img.shape.x, img.shape.y, img.shape.z];
  const names = ["YZ (x)", "XZ (y)", "XY (z)"];
  return (
    <div className="scn-slices">
      {[0, 1, 2].map((i) => (
        <label className="scn-slice" key={i} title={`Show the ${names[i]} plane and move it along ${["x", "y", "z"][i]}`}>
          <input type="checkbox" checked={value.show[i]} onChange={(e) => onChange({ ...value, show: value.show.map((s, k) => (k === i ? e.target.checked : s)) as [boolean, boolean, boolean] })} />
          <span className="scn-slice-name">{names[i]}</span>
          <input type="range" min={0} max={1} step={1 / Math.max(1, n[i] - 1)} value={value.pos[i]} disabled={!value.show[i]} onChange={(e) => onChange({ ...value, pos: value.pos.map((p, k) => (k === i ? Number(e.target.value) : p)) as Vec3 })} />
          <span className="val">{Math.round(value.pos[i] * Math.max(0, n[i] - 1))}</span>
        </label>
      ))}
    </div>
  );
}

const PRESETS: ViewPreset[] = ["front", "back", "left", "right", "top", "bottom"];

export default function SceneControls({ img, images, landmarkRows, recording, freehand, lastAnimation, onPreset, onRecord, onStop, onFreehand, onSnapshot, onSaveScene, onLoadScene, onLandmarksToResults }: Props) {
  const sc = useScene();
  const unit = img.calibration.unit && !/^pixels?$/i.test(img.calibration.unit) ? img.calibration.unit : "px";
  const others = images.filter((im) => im.id !== img.id);
  const [axis, setAxis] = useState<"x" | "y" | "z">("y");
  const [frames, setFrames] = useState(36);
  const [degrees, setDegrees] = useState(360);
  const [fps, setFps] = useState(15);
  const [snapW, setSnapW] = useState(1920);
  const [snapH, setSnapH] = useState(1080);
  const [snapSS, setSnapSS] = useState(2);
  const [addRois, setAddRois] = useState(true);
  const [addId, setAddId] = useState("");
  const busy = !!recording;

  const setLights = (patch: Partial<typeof sc.lights>) => sc.set({ lights: { ...sc.lights, ...patch } });
  const setClip = (which: "min" | "max", i: number, v: number) => {
    const c = { min: [...sc.clip.min] as Vec3, max: [...sc.clip.max] as Vec3 };
    c[which][i] = v;
    if (c.min[i] > c.max[i]) (which === "min" ? c.max : c.min)[i] = v;
    sc.set({ clip: c });
  };
  const lightDir = (i: number, v: number) => setLights({ direction: sc.lights.direction.map((d, k) => (k === i ? v : d)) as Vec3 });

  const addObject = (id: string) => {
    const im = images.find((x) => x.id === id);
    if (!im) return;
    sc.addObject(im, im.shape.z > 1 ? "volume" : "orthoslice");
    setAddId("");
    openState.objects = true;
  };

  const objectEditor = (o: SceneObject) => {
    const oi = images.find((x) => x.id === o.image);
    const up = (patch: Partial<SceneObject>) => sc.updateObject(o.id, patch);
    return (
      <div className="scn-obj-body">
        <div className="scn-row">
          <span className="scn-key">Display</span>
          <select value={o.display} onChange={(e) => up({ display: e.target.value as SceneObject["display"] })} title="Display as volume, orthoslices or iso-surface">
            <option value="volume">Volume</option>
            <option value="orthoslice">Orthoslice</option>
            <option value="surface">Surface</option>
          </select>
          {o.display === "volume" && (
            <select value={o.mode} onChange={(e) => up({ mode: e.target.value as SceneObject["mode"] })} title="Volume rendering">
              <option value="mip">MIP</option>
              <option value="alpha">Alpha</option>
              <option value="iso">Iso</option>
            </select>
          )}
          {oi && oi.shape.c > 1 && (
            <select value={o.channel === null ? "all" : String(o.channel)} onChange={(e) => up({ channel: e.target.value === "all" ? null : Number(e.target.value) })} title="Channel">
              <option value="all">All</option>
              {oi.display.channels.map((c, k) => <option key={k} value={k}>{c.name || `C${k + 1}`}</option>)}
            </select>
          )}
        </div>
        <div className="scn-row">
          <span className="scn-key">Colour</span>
          <input type="color" value={o.color || "#ffffff"} onChange={(e) => up({ color: e.target.value })} title="Flat colour (3D Viewer ▸ Edit ▸ Change color)" />
          <button className={"btn sm" + (o.color ? "" : " primary")} onClick={() => up({ color: null })} title="Use the channel LUTs">LUT</button>
        </div>
        <Slider label="Transp." value={o.transparency} onChange={(v) => up({ transparency: v })} max={0.95} title="Transparency (Edit ▸ Change transparency)" />
        {(o.display === "surface" || (o.display === "volume" && o.mode === "iso")) ? (
          <Slider label="Level" value={o.level} onChange={(v) => up({ level: v })} min={0.02} max={0.98} title="Iso-surface level (0..1 of the display range; Edit ▸ Adjust threshold)" />
        ) : (
          <Slider label="Cut" value={o.threshold} onChange={(v) => up({ threshold: v })} max={0.98} title="Suppress intensities below this level (Edit ▸ Adjust threshold)" />
        )}
        <Slider label="Gain" value={o.brightness} onChange={(v) => up({ brightness: v })} min={0.1} max={3} step={0.05} title="Brightness" />
        {oi && <SlicesEditor img={oi} value={o.slices} onChange={(slices) => up({ slices })} />}
        <TransformEditor t={o.transform} unit={unit} onChange={(transform) => up({ transform })} />
      </div>
    );
  };

  const n = landmarkRows.length;
  const fmtCal = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(1) : v.toFixed(2));

  return (
    <div className="scn">
      <Sub id="view" title="View" badge={sc.projection === "orthographic" ? "ortho" : undefined}>
        <div className="scn-row scn-presets">
          {PRESETS.map((p) => <button key={p} className="btn sm" onClick={() => onPreset(p)} title={`${p[0].toUpperCase()}${p.slice(1)} view`}>{p[0].toUpperCase() + p.slice(1)}</button>)}
        </div>
        <div className="scn-row">
          <span className="seg" title="Projection">
            <button className={sc.projection === "perspective" ? "on" : ""} onClick={() => sc.set({ projection: "perspective" })}>Persp.</button>
            <button className={sc.projection === "orthographic" ? "on" : ""} onClick={() => sc.set({ projection: "orthographic" })}>Ortho</button>
          </span>
          <span className="grow" />
          <span className="scn-key">Background</span>
          <input type="color" value={sc.background || "#000000"} onChange={(e) => sc.set({ background: e.target.value })} title="Background colour (View ▸ Change background color)" />
          <button className={"btn sm" + (sc.background ? "" : " primary")} onClick={() => sc.set({ background: null })} title="Follow the theme">Theme</button>
        </div>
        <div className="scn-group">
          <div className="scn-group-title">Light <label className="scn-check"><input type="checkbox" checked={sc.lights.headlight} onChange={(e) => setLights({ headlight: e.target.checked })} /> head light</label>
            <span className="grow" /><button className="btn sm" onClick={() => sc.set({ lights: defaultLights() })}>Reset</button></div>
          <Slider label="Ambient" value={sc.lights.ambient} onChange={(v) => setLights({ ambient: v })} />
          <Slider label="Diffuse" value={sc.lights.diffuse} onChange={(v) => setLights({ diffuse: v })} max={1.5} />
          <Slider label="Specular" value={sc.lights.specular} onChange={(v) => setLights({ specular: v })} max={1} />
          <div className="scn-row">
            <span className="scn-key" title={sc.lights.headlight ? "Direction relative to the camera (x right, y up, z towards you)" : "Direction in world axes"}>Direction</span>
            {[0, 1, 2].map((i) => <Num key={i} value={sc.lights.direction[i]} step={0.1} min={-1} max={1} onChange={(v) => lightDir(i, v)} title={["x", "y", "z"][i]} width={48} />)}
          </div>
        </div>
        <div className="scn-group">
          <div className="scn-group-title">Clipping <span className="grow" /><button className="btn sm" onClick={() => sc.set({ clip: defaultClip() })}>Reset</button></div>
          {[0, 1, 2].map((i) => (
            <div className="scn-row scn-clip" key={i}>
              <span className="scn-key">{["X", "Y", "Z"][i]}</span>
              <input type="range" min={0} max={1} step={0.01} value={sc.clip.min[i]} onChange={(e) => setClip("min", i, Number(e.target.value))} title="Lower bound" />
              <input type="range" min={0} max={1} step={0.01} value={sc.clip.max[i]} onChange={(e) => setClip("max", i, Number(e.target.value))} title="Upper bound" />
            </div>
          ))}
        </div>
      </Sub>

      <Sub id="slices" title="Orthoslices & transform" badge={sc.primary.slices.show.some(Boolean) ? sc.primary.slices.show.map((s, i) => (s ? "XYZ"[i] : "")).join("") : undefined}>
        <SlicesEditor img={img} value={sc.primary.slices} onChange={(slices) => sc.set({ primary: { ...sc.primary, slices } })} />
        <div className="scn-row">
          <span className="scn-key">Colour</span>
          <input type="color" value={sc.primary.color || "#ffffff"} onChange={(e) => sc.set({ primary: { ...sc.primary, color: e.target.value } })} title="Flat colour for this image's volume / surfaces / slices" />
          <button className={"btn sm" + (sc.primary.color ? "" : " primary")} onClick={() => sc.set({ primary: { ...sc.primary, color: null } })} title="Use the channel LUTs">LUT</button>
        </div>
        <TransformEditor t={sc.primary.transform} unit={unit} onChange={(transform) => sc.set({ primary: { ...sc.primary, transform } })} />
      </Sub>

      <Sub id="objects" title="Objects" badge={sc.objects.length ? String(sc.objects.length + 1) : undefined}>
        <div className="scn-row">
          <select value={addId} onChange={(e) => setAddId(e.target.value)} title="Add another open image to the scene (same calibrated space)" className="grow">
            <option value="">Add image…</option>
            {others.map((im) => <option key={im.id} value={im.id}>{im.name} ({im.shape.x}×{im.shape.y}×{im.shape.z})</option>)}
          </select>
          <button className="btn sm" disabled={!addId} onClick={() => addObject(addId)} title="Add to the scene"><Plus size={12} /></button>
        </div>
        {!others.length && <div className="hint-text">Open another image to add it to the scene.</div>}
        <div className="scn-objs">
          <div className="scn-obj primary"><span className="scn-obj-name" title="The current image">{img.name}</span><span className="scn-obj-tag">current</span></div>
          {sc.objects.map((o) => (
            <div className={"scn-obj" + (sc.selectedObject === o.id ? " sel" : "") + (o.missing ? " missing" : "")} key={o.id}>
              <div className="scn-obj-head" onClick={() => sc.set({ selectedObject: sc.selectedObject === o.id ? null : o.id })}>
                <button className="icon-btn" style={{ width: 20, height: 20 }} onClick={(e) => { e.stopPropagation(); sc.updateObject(o.id, { visible: !o.visible }); }} title={o.visible ? "Hide" : "Show"}>{o.visible ? <Eye size={13} /> : <EyeOff size={13} />}</button>
                <span className="sw" style={{ background: o.color || "transparent", borderStyle: o.color ? "solid" : "dashed" }} />
                <span className="scn-obj-name" title={o.missing ? "This image is not open" : o.name}>{o.name}{o.missing ? " (not open)" : ""}</span>
                <span className="scn-obj-tag">{o.display}</span>
                <button className="icon-btn" style={{ width: 20, height: 20 }} onClick={(e) => { e.stopPropagation(); sc.removeObject(o.id); }} title="Remove from the scene"><Trash2 size={13} /></button>
              </div>
              {sc.selectedObject === o.id && !o.missing && objectEditor(o)}
            </div>
          ))}
        </div>
        <div className="btn-row" style={{ marginTop: 6 }}>
          <button className="btn sm" onClick={onSaveScene} title="Save objects, camera, lights, background, clipping and landmarks as JSON (scene.save)"><Save size={12} /> Save scene…</button>
          <button className="btn sm" onClick={onLoadScene} title="Load a scene JSON (scene.load)"><FolderOpen size={12} /> Load…</button>
          <button className="btn sm" onClick={() => { sc.reset(); }} title="Remove every object and landmark, reset view settings">Clear</button>
        </div>
      </Sub>

      <Sub id="landmarks" title="Landmarks" badge={n ? String(n) : undefined}>
        <div className="btn-row" style={{ marginTop: 0 }}>
          <button className={"btn sm" + (sc.placing ? " primary" : "")} onClick={() => sc.set({ placing: !sc.placing })} title="Click on a surface in the 3-D view to place a landmark (volume modes: the first voxel above the level along the ray)"><Crosshair size={12} /> {sc.placing ? "Placing… (click the view)" : "Place"}</button>
          <button className="btn sm" disabled={!n} onClick={() => sc.clearLandmarks()} title="Remove all landmarks">Clear</button>
        </div>
        {n > 0 && (
          <div className="scn-lms">
            {landmarkRows.map((r) => (
              <div className={"scn-lm" + (sc.selectedLandmark === r.id ? " sel" : "")} key={r.id} onClick={() => sc.set({ selectedLandmark: r.id })}>
                <span className="sw" style={{ background: r.color }} />
                <input className="scn-lm-name" value={r.name} onChange={(e) => sc.updateLandmark(r.id, { name: e.target.value })} onClick={(e) => e.stopPropagation()} title="Landmark name" />
                <span className="scn-lm-xyz" title={`x, y, z in ${r.unit}`}>{r.cal ? `${fmtCal(r.cal[0])}, ${fmtCal(r.cal[1])}, ${fmtCal(r.cal[2])} ${r.unit}` : "…"}</span>
                <button className="icon-btn" style={{ width: 20, height: 20 }} onClick={(e) => { e.stopPropagation(); sc.removeLandmark(r.id); }} title="Remove"><Trash2 size={12} /></button>
              </div>
            ))}
          </div>
        )}
        {n > 1 && n <= 8 && (
          <table className="scn-dist" title={`Pairwise distances (${landmarkRows[0].unit})`}>
            <thead><tr><th></th>{landmarkRows.slice(1).map((r) => <th key={r.id}>{r.name}</th>)}</tr></thead>
            <tbody>
              {landmarkRows.slice(0, -1).map((a, i) => (
                <tr key={a.id}><th>{a.name}</th>{landmarkRows.slice(1).map((b, j) => <td key={b.id}>{j >= i && a.cal && b.cal ? fmtCal(Math.hypot(a.cal[0] - b.cal[0], a.cal[1] - b.cal[1], a.cal[2] - b.cal[2])) : ""}</td>)}</tr>
              ))}
            </tbody>
          </table>
        )}
        <div className="btn-row" style={{ marginTop: 4 }}>
          <button className="btn sm" disabled={!n} onClick={() => onLandmarksToResults(addRois)} title="Calibrated coordinates → Results, pairwise distances → table (scene.landmarks_to_results)">→ Results</button>
          <label className="scn-check" title="Also add one point ROI per landmark (pinned to its slice) to the ROI Manager"><input type="checkbox" checked={addRois} onChange={(e) => setAddRois(e.target.checked)} /> point ROIs</label>
        </div>
      </Sub>

      <Sub id="animation" title="Animation & snapshot" badge={recording ? `${recording.done}/${recording.total}` : freehand ? "REC" : undefined}>
        <div className="scn-row">
          <span className="scn-key">Rotate</span>
          <span className="seg" title="Rotation axis">
            {(["x", "y", "z"] as const).map((a) => <button key={a} className={axis === a ? "on" : ""} onClick={() => setAxis(a)}>{a.toUpperCase()}</button>)}
          </span>
          <Num value={frames} step={1} min={2} max={720} onChange={(v) => setFrames(Math.max(2, Math.round(v)))} title="Frames" width={48} />
          <span className="scn-key">frames</span>
          <Num value={degrees} step={15} min={1} max={720} onChange={(v) => setDegrees(v)} title="Total rotation (degrees)" width={52} />°
        </div>
        <div className="scn-row">
          <span className="scn-key">fps</span>
          <Num value={fps} step={1} min={1} max={60} onChange={(v) => setFps(Math.max(1, v))} title="Frames per second of the recorded stack / GIF" width={44} />
          <span className="grow" />
          {recording ? (
            <button className="btn sm primary" onClick={onStop} title="Stop"><Square size={12} /> Stop {recording.done}/{recording.total}</button>
          ) : (
            <>
              <button className="btn sm" disabled={busy || freehand} onClick={() => onRecord(axis, frames, degrees, fps)} title="Rotate the scene step by step and record every frame into a new RGB stack (3D Viewer ▸ Record 360° rotation)"><Film size={12} /> Record</button>
              <button className={"btn sm" + (freehand ? " primary" : "")} disabled={busy} onClick={() => onFreehand(!freehand, fps)} title="Record frames while you orbit / zoom (Freehand recording); stop to open them as a stack"><Video size={12} /> {freehand ? "Stop" : "Freehand"}</button>
            </>
          )}
        </div>
        {lastAnimation && (
          <div className="hint-text" style={{ margin: "2px 0" }}>
            Last recording opened as a stack · <a href={animationGifUrl(lastAnimation.id, lastAnimation.fps)} download>download as GIF</a> · File ▸ Export ▸ Video for MP4.
          </div>
        )}
        <div className="scn-row">
          <span className="scn-key">Snapshot</span>
          <Num value={snapW} step={1} min={16} max={8192} onChange={(v) => setSnapW(Math.round(v))} title="Width (px)" width={56} />×
          <Num value={snapH} step={1} min={16} max={8192} onChange={(v) => setSnapH(Math.round(v))} title="Height (px)" width={56} />
          <select value={snapSS} onChange={(e) => setSnapSS(Number(e.target.value))} title="Supersampling (rendered larger, then downscaled)">
            <option value={1}>1×</option>
            <option value={2}>2×</option>
            <option value={3}>3×</option>
            <option value={4}>4×</option>
          </select>
          <button className="btn sm" disabled={busy} onClick={() => onSnapshot(snapW, snapH, snapSS)} title="Render at this resolution and open it as a new image"><Camera size={12} /></button>
        </div>
      </Sub>
    </div>
  );
}
