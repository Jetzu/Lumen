import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { create } from "zustand";
import { Eye, EyeOff, Maximize2, X, RotateCcw, Camera, Play, Pause, Box, Download, Grid3x3, Ruler } from "lucide-react";
import { useStore } from "../store";
import { api } from "../api";
import { lutColor } from "../render/compositor";
import { VolumeRenderer, PRIMARY, fetchVolume, zRatio, cssBackground, defaultView, viewToCamera, cameraBasis, FOV, type VolumeData, type VolumeMode, type ChannelSettings, type OrbitView, type ObjectRender, type RenderOptions, type LandmarkRender } from "../render/volume";
import { fetchMesh, meshKey, meshFileUrl, isoLevelToRaw, formatCount, type MeshData, type MeshRequest, type MeshFormat } from "../render/mesh";
import { useScene, sceneDoc, pushScene, fetchScene, postAnimation, canvasBlob, presetView, hexToRgb, imageExtent, imageZRatio, pixelToLocal, localToPixel, pixelMapping, landmarkToPixel, objectModel, primaryModel, worldToCalibrated, rayFromNdc, intersectRayMesh, firstVoxelAbove, mat4Invert, mat4Mul, mat4RotateEuler, transformPoint, transformDir, type Mat4, type Vec3, type SceneDoc, type Landmark, type ViewPreset, type DisplayMode, type ObjectVolumeMode } from "../render/scene";
import SceneControls, { type LandmarkRow, type RecordState } from "./SceneControls";
import type { ImageSummary } from "../types";
import { useFocusTrap } from "../focusTrap";

type Resolution = 128 | 256 | 384;
type MeshStep = 1 | 2 | 3;

interface VolSettings {
  mode: VolumeMode;
  resolution: Resolution;
  autoRotate: boolean;
  showBox: boolean;
  expanded: boolean;
  view: OrbitView;
  /** mesh mode: draw triangle edges */
  wireframe: boolean;
  /** mesh mode: Taubin smoothing iterations */
  meshSmooth: number;
  /** mesh mode: marching-cubes step (voxels) */
  meshStep: MeshStep;
  meshFormat: MeshFormat;
  /** Per image, only the keys the user changed in the 3-D panel; everything else is derived from the Display panel. */
  channels: Record<string, Partial<ChannelSettings>[]>;
  /** Per source image: the stack the last recording was opened as (for the 'download as GIF' hint); survives a remount. */
  lastAnimation: Record<string, { id: string; fps: number }>;
  set: (p: Partial<Omit<VolSettings, "set" | "setChannel">>) => void;
  setChannel: (imageId: string, i: number, patch: Partial<ChannelSettings>) => void;
}

const DEFAULT_CHANNEL: ChannelSettings = { visible: true, threshold: 0.12, isoLevel: 0.4, opacity: 0.4, brightness: 1, meshOpacity: 1 };

/** Viewer settings live outside the component so the inspector and the large view share them. */
export const useVolSettings = create<VolSettings>((set, get) => ({
  mode: "mip",
  resolution: 256,
  autoRotate: false,
  showBox: true,
  expanded: false,
  view: defaultView(),
  wireframe: false,
  meshSmooth: 0,
  meshStep: 1,
  meshFormat: "obj",
  channels: {},
  lastAnimation: {},
  set: (p) => set(p),
  setChannel: (imageId, i, patch) => {
    const cur = get().channels[imageId] || [];
    const arr = cur.slice(); // sparse on purpose: untouched channels keep following the Display panel
    arr[i] = { ...(cur[i] || {}), ...patch };
    set({ channels: { ...get().channels, [imageId]: arr } });
  },
}));

/** Defaults (visibility from the Display panel) overridden only by the keys the user touched in the 3-D panel. */
function channelSettings(stored: Partial<ChannelSettings>[] | undefined, img: ImageSummary): ChannelSettings[] {
  const n = img.rgb ? 3 : img.shape.c;
  const cur = stored || [];
  return Array.from({ length: n }, (_, k) => ({ ...DEFAULT_CHANNEL, visible: img.display.channels[k]?.visible !== false, ...(cur[k] || {}) }));
}

function lutNamesFor(img: ImageSummary): string[] {
  return img.rgb ? ["Red", "Green", "Blue"] : img.display.channels.map((c) => c.lut);
}

/** One mesh request per visible channel (null for hidden ones); the iso level is the Iso-mode slider in raw units. */
function meshRequestsFor(img: ImageSummary, geomSig: string): (MeshRequest | null)[] {
  const st = useVolSettings.getState();
  const cs = channelSettings(st.channels[img.id], img);
  const chans = img.display.channels;
  return cs.map((c, i) => {
    if (!c.visible) return null;
    const ch = chans[i] || { min: 0, max: 255, gamma: 1 };
    return { id: img.id, rev: img.rev, channel: i, t: img.display.t, level: isoLevelToRaw(c.isoLevel, ch), step: st.meshStep, maxDim: st.resolution, smooth: st.meshSmooth, geomSig };
  });
}

interface MeshState {
  id: string;
  list: (MeshData | null)[];
}

// ------------------------------------------------------------------ inspector section content
export default function VolumeSection({ img }: { img: ImageSummary | undefined }) {
  const expanded = useVolSettings((s) => s.expanded);
  if (!img) return <div className="hint-text">No image selected.</div>;
  if (img.shape.z < 2) return <div className="hint-text">Needs a Z stack. Open a 3-D image (e.g. File ▸ Open Sample ▸ “Cells (3-D, 2 channels)”) to render it as a volume.</div>;
  if (expanded) {
    return (
      <div className="hint-text">
        Shown in the large view. <button className="btn sm" onClick={() => useVolSettings.getState().set({ expanded: false })}>Bring back</button>
      </div>
    );
  }
  return <Viewer img={img} large={false} />;
}

/** Mounted once by the inspector: hosts the large modal when expanded. */
export function VolumeModalHost({ img }: { img: ImageSummary | undefined }) {
  const expanded = useVolSettings((s) => s.expanded);
  const close = useCallback(() => useVolSettings.getState().set({ expanded: false }), []);
  useEffect(() => () => useVolSettings.getState().set({ expanded: false }), []);
  // a real dialog: takes focus when it opens, keeps Tab inside, Esc closes, focus returns to the opener (E7-19)
  const modalRef = useRef<HTMLDivElement>(null);
  const trap = useFocusTrap(modalRef, { onEscape: close, initialFocus: "container", resetKey: expanded ? 1 : 0 });
  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopImmediatePropagation();
        e.preventDefault();
        close();
      }
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [expanded, close]);
  if (!expanded || !img || img.shape.z < 2) return null;
  return createPortal(
    <div className="modal-backdrop vol-backdrop" onMouseDown={close}>
      <div className="vol-modal" ref={modalRef} role="dialog" aria-modal="true" aria-label={`3-D view of ${img.name}`} tabIndex={-1} onMouseDown={(e) => e.stopPropagation()} onKeyDown={trap.onKeyDown}>
        <Viewer img={img} large />
      </div>
    </div>,
    document.body,
  );
}

// ------------------------------------------------------------------ the viewer
function Viewer({ img, large }: { img: ImageSummary; large: boolean }) {
  const luts = useStore((s) => s.luts);
  const theme = useStore((s) => s.theme);
  const toast = useStore((s) => s.toast);
  const mode = useVolSettings((s) => s.mode);
  const resolution = useVolSettings((s) => s.resolution);
  const autoRotate = useVolSettings((s) => s.autoRotate);
  const showBox = useVolSettings((s) => s.showBox);
  const wireframe = useVolSettings((s) => s.wireframe);
  const meshSmooth = useVolSettings((s) => s.meshSmooth);
  const meshStep = useVolSettings((s) => s.meshStep);
  const meshFormat = useVolSettings((s) => s.meshFormat);
  const chStored = useVolSettings((s) => s.channels[img.id]);
  const visSig = img.display.channels.map((c) => (c.visible ? 1 : 0)).join("");
  const chSettings = useMemo(() => channelSettings(chStored, img), [chStored, img.id, img.rgb, img.shape.c, visSig]); // eslint-disable-line react-hooks/exhaustive-deps
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const rendererRef = useRef<VolumeRenderer | null>(null);
  const viewRef = useRef<OrbitView>(useVolSettings.getState().view);
  const imgRef = useRef(img);
  imgRef.current = img;
  const rafRef = useRef<number | null>(null);
  const idleTimer = useRef<number | null>(null);
  const [volume, setVolume] = useState<VolumeData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [snapBusy, setSnapBusy] = useState(false);
  const [meshState, setMeshState] = useState<MeshState | null>(null);
  const [meshLoading, setMeshLoading] = useState(false);
  const [meshError, setMeshError] = useState<string | null>(null);
  // ---- scene (Plugins ▸ 3D Viewer parity): objects, landmarks, lights, clipping, animation recording
  const scene = useScene();
  const images = useStore((s) => s.state?.images) || [];
  const lastRun = useStore((s) => s.lastRun);
  const view = useVolSettings((s) => s.view);
  const imagesRef = useRef(images);
  imagesRef.current = images;
  /** meshes of the current image (mesh mode) for picking */
  const primaryMeshRef = useRef<(MeshData | null)[]>([]);
  /** meshes of the scene objects by object id */
  const objMeshRef = useRef<Map<string, (MeshData | null)[]>>(new Map());
  const uploadedRef = useRef<Set<string>>(new Set());
  /** extra rotation of the whole scene (degrees about world x, y, z) while a turntable is recorded */
  const sceneRotRef = useRef<Vec3>([0, 0, 0]);
  const cancelRef = useRef(false);
  const freehandRef = useRef<{ fps: number; last: number; n: number; frames: { i: number; b: Blob }[] } | null>(null);
  const labelsRef = useRef<HTMLDivElement>(null);
  const [recording, setRecording] = useState<RecordState | null>(null);
  const [freehand, setFreehand] = useState(false);
  // kept in the settings store (not component state): the hint must outlive a remount of the viewer
  const lastAnimationStored = useVolSettings((s) => s.lastAnimation[img.id] ?? null);
  const lastAnimation = lastAnimationStored && images.some((im) => im.id === lastAnimationStored.id) ? lastAnimationStored : null;
  // Only what changes the voxel intensities the server returns; 'inverted' is a colour-only LUT reversal done here.
  const displaySig = JSON.stringify((img.rgb ? img.display.channels.slice(0, 3) : img.display.channels).map((c) => [c.min, c.max, c.gamma]));
  // Calibration (z ratio) changes the resampled geometry without bumping rev.
  const geomSig = String(zRatio(img.calibration));
  const lutNames = lutNamesFor(img);
  const lutInv = img.rgb ? [false, false, false] : img.display.channels.map((c) => !!c.inverted);
  const lutKey = lutNames.map((n, i) => `${n}|${lutInv[i] ? 1 : 0}`).join(",");

  // ---- scene geometry helpers (refs only: safe inside the render callbacks)
  /** Model matrix, extent and half extents of the object (primary or scene object) that owns image `imageId`. */
  const objectFrame = useCallback((imageId: string, withRotation: boolean): { key: string; model: Mat4; ext: Vec3; half: Vec3; img: ImageSummary } | null => {
    const r = rendererRef.current;
    if (!r) return null;
    const im = imgRef.current;
    const half = r.halfExtents;
    const sc = useScene.getState();
    const rot = sceneRotRef.current;
    const R = withRotation && (rot[0] || rot[1] || rot[2]) ? mat4RotateEuler(rot[0], rot[1], rot[2]) : null;
    const spin = (m: Mat4) => (R ? mat4Mul(R, m) : m);
    if (imageId === im.id) return { key: PRIMARY, model: spin(primaryModel(im, half, sc.primary.transform)), ext: imageExtent(im), half, img: im };
    const o = sc.objects.find((x) => x.image === imageId && !x.missing);
    const oi = imagesRef.current.find((x) => x.id === imageId);
    const oh = o ? r.objectHalf(o.id) : null;
    if (!o || !oi || !oh) return null;
    return { key: o.id, model: spin(objectModel(im, half, oi, oh, o.transform)), ext: imageExtent(oi), half: oh, img: oi };
  }, []);

  const landmarkWorld = useCallback((lm: Landmark, withRotation = true): Vec3 | null => {
    const f = objectFrame(lm.image, withRotation);
    if (!f) return null;
    return transformPoint(f.model, pixelToLocal(landmarkToPixel(lm, imageZRatio(f.img)), f.ext, f.half));
  }, [objectFrame]);

  /** Everything the renderer needs for one frame, from the stores and refs. */
  const buildOptions = useCallback((quality: number): RenderOptions | null => {
    const r = rendererRef.current;
    if (!r) return null;
    const st = useVolSettings.getState();
    const sc = useScene.getState();
    const im = imgRef.current;
    const half = r.halfExtents;
    const rot = sceneRotRef.current;
    const R = rot[0] || rot[1] || rot[2] ? mat4RotateEuler(rot[0], rot[1], rot[2]) : null;
    const spin = (m: Mat4) => (R ? mat4Mul(R, m) : m);
    const objects: ObjectRender[] = [];
    for (const o of sc.objects) {
      // an object of the image this viewer shows is dormant: the image is drawn once, as the primary
      if (o.missing || !o.visible || o.image === im.id) continue;
      const oi = imagesRef.current.find((x) => x.id === o.image);
      const oh = r.objectHalf(o.id);
      if (!oi || !oh) continue;
      const nCh = r.objectVolume(o.id)?.channels.length ?? objMeshRef.current.get(o.id)?.length ?? 1;
      const channels: ChannelSettings[] = Array.from({ length: nCh }, () => ({ visible: true, threshold: o.threshold, isoLevel: o.level, opacity: 0.4, brightness: o.brightness, meshOpacity: 1 }));
      objects.push({ key: o.id, display: o.display, mode: o.mode, channels, model: spin(objectModel(im, half, oi, oh, o.transform)), color: hexToRgb(o.color), opacity: 1 - o.transparency, slices: o.slices, visible: true });
    }
    const landmarks: LandmarkRender[] = [];
    for (const lm of sc.landmarks) {
      const w = landmarkWorld(lm);
      if (w) landmarks.push({ world: w, color: hexToRgb(lm.color) || [1, 0.84, 0.29], selected: lm.id === sc.selectedLandmark });
    }
    return {
      mode: st.mode,
      channels: channelSettings(st.channels[im.id], im),
      camera: viewToCamera(viewRef.current, half),
      quality,
      background: hexToRgb(sc.background) || cssBackground(),
      showBox: st.showBox,
      wireframe: st.wireframe,
      projection: sc.projection,
      lights: sc.lights,
      clip: sc.clip,
      primary: { model: spin(primaryModel(im, half, sc.primary.transform)), color: hexToRgb(sc.primary.color), slices: sc.primary.slices },
      objects,
      landmarks,
      landmarkSize: sc.landmarkSize,
    };
  }, [landmarkWorld]);

  /** Place the HTML landmark labels next to their projected points (no React re-render). */
  const updateLabels = useCallback(() => {
    const r = rendererRef.current;
    const box = labelsRef.current;
    if (!r || !box) return;
    const sc = useScene.getState();
    for (const el of Array.from(box.children) as HTMLElement[]) {
      const lm = sc.landmarks.find((l) => l.id === el.dataset.id);
      const w = lm ? landmarkWorld(lm) : null;
      const p = w ? r.project(w) : null;
      if (!p || p[2] > 1 || p[2] < -1) {
        el.style.display = "none";
        continue;
      }
      el.style.display = "";
      el.style.transform = `translate(${Math.round(p[0] + 7)}px, ${Math.round(p[1] - 9)}px)`;
    }
  }, [landmarkWorld]);

  // ---- rendering
  const renderNow = useCallback((quality: number) => {
    const r = rendererRef.current;
    if (!r) return;
    const dpr = Math.min(window.devicePixelRatio || 1, large ? 1.5 : 2);
    r.resize(dpr);
    const opts = buildOptions(quality);
    if (!opts) return;
    r.render(opts);
    updateLabels();
    // freehand recording: grab a frame at most every 1/fps s while the user interacts
    const fh = freehandRef.current;
    const canvas = canvasRef.current;
    if (fh && canvas) {
      const now = performance.now();
      if (now - fh.last >= 1000 / fh.fps && fh.n < 900) {
        fh.last = now;
        const i = fh.n++;
        canvasBlob(canvas).then((b) => fh.frames.push({ i, b })).catch(() => undefined);
      }
    }
  }, [large, buildOptions, updateLabels]);

  const requestRender = useCallback((quality = 1) => {
    if (rafRef.current !== null) return;
    rafRef.current = requestAnimationFrame(() => {
      rafRef.current = null;
      renderNow(quality);
      if (quality < 1) {
        if (idleTimer.current) window.clearTimeout(idleTimer.current);
        idleTimer.current = window.setTimeout(() => {
          idleTimer.current = null;
          if (!useVolSettings.getState().autoRotate) requestRender(1);
        }, 220);
      }
    });
  }, [renderNow]);

  // create / dispose the WebGL renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      rendererRef.current = new VolumeRenderer(canvas);
      setError(null);
    } catch (e: any) {
      setError(e.message || String(e));
      return;
    }
    const ro = new ResizeObserver(() => requestRender(1));
    ro.observe(canvas);
    return () => {
      ro.disconnect();
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      if (idleTimer.current) window.clearTimeout(idleTimer.current);
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, [requestRender]);

  // fetch the volume (debounced; immediate for the first load)
  useEffect(() => {
    let alive = true;
    const delay = volume ? 300 : 0;
    const timer = window.setTimeout(() => {
      setLoading(true);
      fetchVolume(img.id, img.rev, img.display.t, resolution, "all", displaySig, geomSig)
        .then((v) => {
          if (!alive) return;
          setVolume(v);
          if (rendererRef.current) setError(null); // keep a renderer-creation error visible
        })
        .catch((e) => alive && setError(e.message || String(e)))
        .finally(() => alive && setLoading(false));
    }, delay);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [img.id, img.rev, img.display.t, resolution, displaySig, geomSig]); // eslint-disable-line react-hooks/exhaustive-deps

  // upload volume + LUTs
  useEffect(() => {
    const r = rendererRef.current;
    if (!r || !volume) return;
    try {
      r.setVolume(volume);
      r.setLuts(luts, lutNames, lutInv);
      requestRender(1);
    } catch (e: any) {
      setError(e.message || String(e));
    }
  }, [volume]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    const r = rendererRef.current;
    if (!r || !volume) return;
    r.setLuts(luts, lutNames, lutInv);
    requestRender(1);
  }, [luts, lutKey]); // eslint-disable-line react-hooks/exhaustive-deps

  // mesh mode: fetch one iso-surface per visible channel (debounced while a slider is dragged)
  const meshSig = mode === "mesh" ? meshRequestsFor(img, geomSig).map((q) => (q ? meshKey(q) : "-")).join("|") : "";
  useEffect(() => {
    if (mode !== "mesh") return;
    const reqs = meshRequestsFor(img, geomSig);
    let alive = true;
    const delay = meshState && meshState.id === img.id ? 250 : 0;
    const timer = window.setTimeout(() => {
      setMeshLoading(true);
      Promise.all(reqs.map((q) => (q ? fetchMesh(q) : Promise.resolve(null))))
        .then((list) => {
          if (!alive) return;
          setMeshState({ id: img.id, list });
          setMeshError(null);
        })
        .catch((e) => alive && setMeshError(e.message || String(e)))
        .finally(() => alive && setMeshLoading(false));
    }, delay);
    return () => {
      alive = false;
      window.clearTimeout(timer);
    };
  }, [mode, meshSig]); // eslint-disable-line react-hooks/exhaustive-deps

  // upload the meshes (a surface of another image is never drawn in this image's box)
  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;
    try {
      r.setMeshes(meshState && meshState.id === img.id ? meshState.list : []);
      requestRender(1);
    } catch (e: any) {
      setMeshError(e.message || String(e));
    }
  }, [meshState, img.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // settings / theme changes → re-render
  useEffect(() => {
    requestRender(1);
  }, [mode, showBox, wireframe, chSettings, theme, requestRender]);

  // the primary meshes for picking
  useEffect(() => {
    primaryMeshRef.current = meshState && meshState.id === img.id ? meshState.list : [];
  }, [meshState, img.id]);

  // scene changes (objects, landmarks, lights, clipping, background, projection, primary transform) → re-render
  useEffect(() => {
    requestRender(1);
  }, [scene.objects, scene.landmarks, scene.selectedLandmark, scene.lights, scene.clip, scene.background, scene.projection, scene.primary, scene.landmarkSize, requestRender]);

  // resources of the scene objects: a volume (volume / orthoslice display, or when slices are shown) and / or surfaces
  const objSig = scene.objects
    .map((o) => {
      const oi = images.find((x) => x.id === o.image);
      if (!oi) return `${o.id}:missing`;
      const disp = JSON.stringify(oi.display.channels.map((c) => [c.min, c.max, c.gamma, c.lut, !!c.inverted]));
      return [o.id, o.image, oi.rev, oi.display.t, o.display, o.mode, o.channel ?? "all", o.level, o.slices.show.join(""), disp, zRatio(oi.calibration)].join(":");
    })
    .join("|");
  useEffect(() => {
    const r = rendererRef.current;
    if (!r) return;
    let alive = true;
    const sc = useScene.getState();
    // objects of the image this viewer shows are dormant (drawn once, as the primary): neither fetched nor uploaded,
    // and their GPU resources are freed until another image is current again
    const active = sc.objects.filter((o) => o.image !== imgRef.current.id);
    const present = new Set(active.map((o) => o.id));
    for (const key of Array.from(uploadedRef.current)) {
      if (!present.has(key)) {
        r.removeObject(key);
        objMeshRef.current.delete(key);
        uploadedRef.current.delete(key);
      }
    }
    (async () => {
      for (const o of active) {
        const oi = imagesRef.current.find((x) => x.id === o.image);
        if (!oi) {
          if (!o.missing) useScene.getState().updateObject(o.id, { missing: true });
          continue;
        }
        if (o.missing) useScene.getState().updateObject(o.id, { missing: false, name: oi.name });
        const names = lutNamesFor(oi);
        const inv = oi.rgb ? [false, false, false] : oi.display.channels.map((c) => !!c.inverted);
        const idx = o.channel === null || o.channel >= names.length ? names.map((_, i) => i) : [o.channel];
        const displaySig2 = JSON.stringify((oi.rgb ? oi.display.channels.slice(0, 3) : oi.display.channels).map((c) => [c.min, c.max, c.gamma]));
        const geomSig2 = String(zRatio(oi.calibration));
        const rr = rendererRef.current;
        if (!alive || !rr) return;
        try {
          rr.setObjectLuts(o.id, luts, idx.map((i) => names[i]), idx.map((i) => inv[i]));
          if (o.display !== "surface" || o.slices.show.some(Boolean)) {
            const v = await fetchVolume(oi.id, oi.rev, oi.display.t, resolution, idx.length === names.length ? "all" : String(idx[0]), displaySig2, geomSig2);
            if (!alive) return;
            rr.setObjectVolume(o.id, v);
          } else {
            rr.setObjectVolume(o.id, null);
          }
          if (o.display === "surface") {
            if (oi.shape.z < 2) throw new Error("a surface needs a Z stack");
            const list = await Promise.all(idx.map((i) => fetchMesh({ id: oi.id, rev: oi.rev, channel: i, t: oi.display.t, level: isoLevelToRaw(o.level, oi.display.channels[i] || { min: 0, max: 255, gamma: 1 }), step: meshStep, maxDim: resolution, smooth: meshSmooth, geomSig: geomSig2 })));
            if (!alive) return;
            objMeshRef.current.set(o.id, list);
            rr.setObjectMeshes(o.id, list);
          } else {
            objMeshRef.current.delete(o.id);
            rr.setObjectMeshes(o.id, []);
          }
          uploadedRef.current.add(o.id);
        } catch (e: any) {
          if (alive) toast(`${oi.name}: ${e.message || e}`, "error");
        }
        if (alive) requestRender(1);
      }
    })();
    return () => {
      alive = false;
    };
  }, [objSig, luts, resolution, meshStep, meshSmooth, img.id]); // eslint-disable-line react-hooks/exhaustive-deps

  // publish the scene to the session (debounced) so scene.save / Landmarks to Results work from the palette too
  useEffect(() => {
    const t = window.setTimeout(() => {
      const st = useVolSettings.getState();
      pushScene(sceneDoc(useScene.getState(), imgRef.current, st.view, st.mode, st.showBox)).catch(() => undefined);
    }, 400);
    return () => window.clearTimeout(t);
  }, [scene.objects, scene.landmarks, scene.lights, scene.clip, scene.background, scene.projection, scene.primary, view, mode, showBox, img.id]);

  /** Apply a loaded scene document: objects / landmarks / lights … into the store, the camera into the view. */
  const applyLoaded = useCallback((doc: SceneDoc) => {
    // the entry of the image this viewer shows (not necessarily the saved primary) feeds the primary slot
    useScene.getState().applyDoc(doc, useStore.getState().state?.images || [], imgRef.current.id);
    const c = doc.camera;
    viewRef.current = { theta: c.theta, phi: c.phi, zoom: c.zoom, target: c.target, roll: c.roll };
    const patch: Partial<Omit<VolSettings, "set" | "setChannel">> = { view: viewRef.current, showBox: doc.show_box };
    if (doc.mode === "mip" || doc.mode === "alpha" || doc.mode === "iso" || doc.mode === "mesh") patch.mode = doc.mode;
    useVolSettings.getState().set(patch);
    requestRender(1);
  }, [requestRender]);

  // Load 3D Scene (from the button or the command palette) → pick up the loaded document
  useEffect(() => {
    if (lastRun?.opId !== "scene.load") return;
    fetchScene()
      .then(({ scene: doc }) => doc && applyLoaded(doc))
      .catch(() => undefined);
  }, [lastRun, applyLoaded]);

  // auto-rotate loop
  useEffect(() => {
    if (!autoRotate) return;
    let id = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      viewRef.current = { ...viewRef.current, theta: viewRef.current.theta + dt * 0.6 };
      renderNow(0.7);
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => {
      cancelAnimationFrame(id);
      useVolSettings.getState().set({ view: viewRef.current });
      requestRender(1);
    };
  }, [autoRotate, renderNow, requestRender]);

  // ---- interaction
  const pickRef = useRef<(clientX: number, clientY: number) => void>(() => undefined);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let drag: { x: number; y: number; pan: boolean; sx: number; sy: number; moved: boolean } | null = null;
    const commit = () => useVolSettings.getState().set({ view: viewRef.current });
    const down = (e: PointerEvent) => {
      if (e.button > 2) return;
      drag = { x: e.clientX, y: e.clientY, pan: e.button !== 0 || e.shiftKey, sx: e.clientX, sy: e.clientY, moved: false };
      canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
    };
    const move = (e: PointerEvent) => {
      if (!drag) return;
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      const moved = drag.moved || Math.hypot(e.clientX - drag.sx, e.clientY - drag.sy) > 3;
      drag = { ...drag, x: e.clientX, y: e.clientY, moved };
      if (!moved) return;
      const v = viewRef.current;
      if (drag.pan) {
        const r = rendererRef.current;
        const half = r ? r.halfExtents : ([1, 1, 1] as [number, number, number]);
        const cam = viewToCamera(v, half);
        const k = (2 * cam.dist * Math.tan(FOV / 2)) / Math.max(1, canvas.clientHeight);
        const { right, up } = cameraBasis(cam);
        viewRef.current = { ...v, target: [v.target[0] - dx * k * right[0] + dy * k * up[0], v.target[1] - dx * k * right[1] + dy * k * up[1], v.target[2] - dx * k * right[2] + dy * k * up[2]] };
      } else {
        viewRef.current = { ...v, theta: v.theta - dx * 0.0085, phi: Math.max(-1.52, Math.min(1.52, v.phi + dy * 0.0085)) };
      }
      requestRender(0.5);
    };
    const up = (e: PointerEvent) => {
      if (!drag) return;
      const click = !drag.moved && !drag.pan;
      drag = null;
      try {
        canvas.releasePointerCapture(e.pointerId);
      } catch {
        /* ignore */
      }
      commit();
      requestRender(1);
      // a plain click (no drag) places a landmark while "Place" is active
      if (click && useScene.getState().placing) pickRef.current(e.clientX, e.clientY);
    };
    const wheel = (e: WheelEvent) => {
      e.preventDefault();
      const v = viewRef.current;
      const f = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0018));
      viewRef.current = { ...v, zoom: Math.max(0.25, Math.min(12, v.zoom * f)) };
      commit();
      requestRender(0.6);
    };
    const dbl = () => {
      viewRef.current = defaultView();
      commit();
      requestRender(1);
    };
    const ctx = (e: Event) => e.preventDefault();
    canvas.addEventListener("pointerdown", down);
    canvas.addEventListener("pointermove", move);
    canvas.addEventListener("pointerup", up);
    canvas.addEventListener("pointercancel", up);
    canvas.addEventListener("wheel", wheel, { passive: false });
    canvas.addEventListener("dblclick", dbl);
    canvas.addEventListener("contextmenu", ctx);
    return () => {
      canvas.removeEventListener("pointerdown", down);
      canvas.removeEventListener("pointermove", move);
      canvas.removeEventListener("pointerup", up);
      canvas.removeEventListener("pointercancel", up);
      canvas.removeEventListener("wheel", wheel);
      canvas.removeEventListener("dblclick", dbl);
      canvas.removeEventListener("contextmenu", ctx);
    };
  }, [requestRender]);

  const reset = () => {
    viewRef.current = defaultView();
    useVolSettings.getState().set({ view: viewRef.current });
    requestRender(1);
  };

  /**
   * After an upload that activated a new image (snapshots: POST /api/images/upload always activates), make the stack
   * being viewed active again BEFORE the state is refreshed: the 3-D View section is bound to the active Z stack and
   * would unmount this viewer otherwise.  api.select, not store.selectImage (a no-op while the local state still
   * shows this image as active).
   */
  const keepViewing = async () => {
    await api.select(img.id).catch(() => undefined);
    await useStore.getState().refresh();
  };

  const snapshot = () => {
    const canvas = canvasRef.current;
    if (!canvas || !rendererRef.current || !volume) return;
    setSnapBusy(true);
    renderNow(1);
    canvas.toBlob(async (blob) => {
      try {
        if (!blob) throw new Error("Could not read the canvas");
        const base = img.name.replace(/\.[^.]+$/, "") || "volume";
        const file = new File([blob], `${base}_3D.png`, { type: "image/png" });
        await api.upload(file);
        await keepViewing();
        toast("Snapshot opened as a new image (see the image list)", "success");
      } catch (e: any) {
        toast(`Snapshot failed: ${e.message || e}`, "error");
      } finally {
        setSnapBusy(false);
      }
    }, "image/png");
  };

  // ---- landmarks: pick on the surface under the cursor (ray ↔ mesh, or the first voxel above the level in volume modes)
  pickRef.current = (clientX: number, clientY: number) => {
    const r = rendererRef.current;
    const canvas = canvasRef.current;
    if (!r || !canvas) return;
    const rect = canvas.getBoundingClientRect();
    if (rect.width < 1 || rect.height < 1) return;
    const ndcX = ((clientX - rect.left) / rect.width) * 2 - 1;
    const ndcY = 1 - ((clientY - rect.top) / rect.height) * 2;
    const sc = useScene.getState();
    const st = useVolSettings.getState();
    const im = imgRef.current;
    const ray = rayFromNdc(viewToCamera(viewRef.current, r.halfExtents), ndcX, ndcY, rect.width / rect.height, sc.projection);
    type Cand = { image: string; key: string; display: DisplayMode; mode: ObjectVolumeMode; channels: ChannelSettings[]; meshes: (MeshData | null)[] };
    const cands: Cand[] = [{ image: im.id, key: PRIMARY, display: st.mode === "mesh" ? "surface" : "volume", mode: st.mode === "mesh" ? "iso" : st.mode, channels: channelSettings(st.channels[im.id], im), meshes: primaryMeshRef.current }];
    for (const o of sc.objects) {
      if (o.missing || !o.visible || o.display === "orthoslice" || o.image === im.id) continue; // dormant: the primary is the pick target
      const n = r.objectVolume(o.id)?.channels.length ?? objMeshRef.current.get(o.id)?.length ?? 0;
      cands.push({ image: o.image, key: o.id, display: o.display, mode: o.mode, channels: Array.from({ length: n }, () => ({ visible: true, threshold: o.threshold, isoLevel: o.level, opacity: 0.4, brightness: o.brightness, meshOpacity: 1 })), meshes: objMeshRef.current.get(o.id) || [] });
    }
    let best: { t: number; image: string; px: Vec3 } | null = null;
    for (const c of cands) {
      const f = objectFrame(c.image, false);
      if (!f) continue;
      const inv = mat4Invert(f.model);
      const oL = transformPoint(inv, ray.origin);
      const dL = transformDir(inv, ray.dir);
      let hit: { t: number; point: Vec3 } | null = null;
      if (c.display === "surface") {
        const { scale, offset } = pixelMapping(f.ext, f.half);
        const oP: Vec3 = [(oL[0] - offset[0]) / scale[0], (oL[1] - offset[1]) / scale[1], (oL[2] - offset[2]) / scale[2]];
        const dP: Vec3 = [dL[0] / scale[0], dL[1] / scale[1], dL[2] / scale[2]];
        for (let i = 0; i < c.meshes.length; i++) {
          const m = c.meshes[i];
          if (!m || !(c.channels[i]?.visible ?? true)) continue;
          const h = intersectRayMesh(oP, dP, m.vertices, m.faces);
          if (h && (!hit || h.t < hit.t)) hit = { t: h.t, point: pixelToLocal(h.point, f.ext, f.half) };
        }
      } else {
        const vol = r.objectVolume(c.key);
        if (!vol) continue;
        const levels = vol.channels.map((_, i) => {
          const cs = c.channels[i];
          if (!cs || !cs.visible) return null;
          return c.mode === "iso" ? cs.isoLevel : Math.max(0.02, cs.threshold);
        });
        const h = firstVoxelAbove(vol, f.half, oL, dL, levels);
        if (h) hit = { t: h.t, point: h.point };
      }
      if (hit && (!best || hit.t < best.t)) {
        const p = localToPixel(hit.point, f.ext, f.half);
        best = { t: hit.t, image: c.image, px: [p[0], p[1], p[2] / imageZRatio(f.img)] };
      }
    }
    if (!best) {
      toast("Nothing under the cursor: click on the surface", "info");
      return;
    }
    useScene.getState().addLandmark({ image: best.image, px: best.px });
  };

  const set = useVolSettings.getState().set;
  const setChannel = (i: number, patch: Partial<ChannelSettings>) => useVolSettings.getState().setChannel(img.id, i, patch);
  const names = img.rgb ? ["Red", "Green", "Blue"] : img.display.channels.map((c, i) => c.name || `C${i + 1}`);
  const meshes: (MeshData | null)[] = meshState && meshState.id === img.id ? meshState.list : [];
  const meshTotals = meshes.reduce((acc, m) => (m ? { v: acc.v + m.nVertices, f: acc.f + m.nFaces } : acc), { v: 0, f: 0 });
  const hasMesh = meshTotals.f > 0;

  /** Download the visible surfaces as files (the server sends Content-Disposition: attachment). */
  const exportMesh = () => {
    const reqs = meshRequestsFor(img, geomSig).filter((q): q is MeshRequest => !!q);
    if (!reqs.length) {
      toast("No visible channel to export", "error");
      return;
    }
    reqs.forEach((q, k) => {
      window.setTimeout(() => {
        const a = document.createElement("a");
        a.href = meshFileUrl(q, meshFormat);
        a.download = "";
        a.rel = "noopener";
        document.body.appendChild(a);
        a.click();
        a.remove();
      }, k * 500);
    });
    toast(`Exporting ${reqs.length} surface${reqs.length > 1 ? "s" : ""} as ${meshFormat.toUpperCase()}`, "info");
  };

  /** Surface area / enclosed volume per component → Results (Plugins ▸ 3D ▸ Measure Surface). */
  const measureMesh = async () => {
    const reqs = meshRequestsFor(img, geomSig).filter((q): q is MeshRequest => !!q);
    if (!reqs.length) {
      toast("No visible channel to measure", "error");
      return;
    }
    for (const q of reqs) {
      await useStore.getState().runOp("mesh.measure", { channel: q.channel, level: q.level, step: q.step, smooth: q.smooth, max_dim: q.maxDim }, { imageId: img.id, roi: null });
    }
  };

  // ---- scene actions: recording, snapshots at a chosen resolution, view presets, scene files, landmarks → Results
  const base = img.name.replace(/\.[^.]+$/, "") || "volume";
  const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));

  /** Upload the recorded frames: they become an RGB time stack (a new image; GIF / video from there). */
  const finishRecording = async (blobs: Blob[], fps: number, name: string) => {
    if (!blobs.length) {
      toast("No frames were recorded", "info");
      return;
    }
    try {
      const res = await postAnimation(blobs, name, fps); // the stack opens beside this one (not activated): the viewer stays
      await useStore.getState().refresh();
      const vs = useVolSettings.getState();
      vs.set({ lastAnimation: { ...vs.lastAnimation, [img.id]: { id: res.id, fps } } });
      toast(res.message, "success");
    } catch (e: any) {
      toast(`Recording failed: ${e.message || e}`, "error");
    }
  };

  /** 3D Viewer ▸ Record 360° rotation: rotate the whole scene about a world axis, one frame per step. */
  const recordTurntable = async (axis: "x" | "y" | "z", frames: number, degrees: number, fps: number) => {
    const canvas = canvasRef.current;
    if (!canvas || !rendererRef.current || recording) return;
    if (autoRotate) set({ autoRotate: false });
    cancelRef.current = false;
    const total = Math.max(2, Math.round(frames));
    const label = `Rotating about ${axis.toUpperCase()}`;
    setRecording({ done: 0, total, label });
    const blobs: Blob[] = [];
    try {
      for (let i = 0; i < total; i++) {
        if (cancelRef.current) break;
        const ang = (degrees * i) / total;
        sceneRotRef.current = axis === "x" ? [ang, 0, 0] : axis === "y" ? [0, ang, 0] : [0, 0, ang];
        renderNow(1);
        blobs.push(await canvasBlob(canvas));
        setRecording({ done: i + 1, total, label });
        await nextFrame();
      }
    } catch (e: any) {
      toast(`Recording failed: ${e.message || e}`, "error");
    }
    sceneRotRef.current = [0, 0, 0];
    requestRender(1);
    setRecording(null);
    await finishRecording(blobs, fps, `${base}_rotate${axis.toUpperCase()}`);
  };

  const stopRecording = () => {
    cancelRef.current = true;
  };

  /** Freehand recording: frames are captured while the user orbits / zooms / pans (see renderNow). */
  const toggleFreehand = async (on: boolean, fps: number) => {
    if (on) {
      freehandRef.current = { fps: Math.max(1, fps), last: 0, n: 0, frames: [] };
      setFreehand(true);
      toast("Recording: orbit, zoom or pan the view, then press Stop", "info");
      renderNow(1);
      return;
    }
    const fh = freehandRef.current;
    freehandRef.current = null;
    setFreehand(false);
    if (!fh) return;
    await new Promise((resolve) => window.setTimeout(resolve, 150)); // let the last toBlob callbacks land
    await finishRecording(fh.frames.sort((a, b) => a.i - b.i).map((f) => f.b), fh.fps, `${base}_freehand`);
  };

  /** Off-screen snapshot at a chosen resolution (supersampled). */
  const snapshotAt = async (w: number, h: number, ss: number) => {
    const r = rendererRef.current;
    if (!r || recording) return;
    const opts = buildOptions(1);
    if (!opts) return;
    setSnapBusy(true);
    try {
      const blob = await r.renderToBlob(opts, w, h, ss);
      await api.upload(new File([blob], `${base}_3D_${Math.round(w)}x${Math.round(h)}.png`, { type: "image/png" }));
      await keepViewing();
      toast(`Snapshot ${Math.round(w)}×${Math.round(h)} opened as a new image (see the image list)`, "success");
    } catch (e: any) {
      toast(`Snapshot failed: ${e.message || e}`, "error");
    } finally {
      setSnapBusy(false);
    }
  };

  const onPreset = (p: ViewPreset) => {
    viewRef.current = presetView(p, viewRef.current);
    set({ view: viewRef.current });
    requestRender(1);
  };

  const currentDoc = () => {
    const st = useVolSettings.getState();
    return sceneDoc(useScene.getState(), img, st.view, st.mode, st.showBox);
  };
  const saveScene = () =>
    useStore.getState().setFileDialog({
      mode: "pick-save", title: "Save 3-D scene", filename: `${base}_scene`, extensions: [".json"],
      onPick: (path) => void useStore.getState().runOp("scene.save", { path, scene: JSON.stringify(currentDoc()) }, { imageId: img.id, roi: null }),
    });
  const loadScene = () =>
    useStore.getState().setFileDialog({
      mode: "pick-open", title: "Load 3-D scene", extensions: [".json"],
      onPick: (path) => void useStore.getState().runOp("scene.load", { path }, { imageId: img.id, roi: null }), // the lastRun effect applies it
    });

  const halfNow = rendererRef.current?.halfExtents || ([1, 1, 1] as Vec3);
  const unitName = img.calibration.unit && !/^pixels?$/i.test(img.calibration.unit) ? img.calibration.unit : "px";
  const landmarkRows: LandmarkRow[] = scene.landmarks.map((lm) => {
    const w = landmarkWorld(lm, false);
    return { id: lm.id, name: lm.name, cal: w ? worldToCalibrated(img, halfNow, w) : null, unit: unitName, color: lm.color || "#ffd54a" };
  });
  const landmarksToResults = (addRois: boolean) => {
    const lms = scene.landmarks.map((lm, i) => ({ id: lm.id, name: lm.name, image: lm.image, px: lm.px, ...(landmarkRows[i].cal ? { cal: landmarkRows[i].cal } : {}) }));
    if (!lms.length) {
      toast("Place a landmark first", "info");
      return;
    }
    void useStore.getState().runOp("scene.landmarks_to_results", { landmarks: JSON.stringify(lms), add_rois: addRois, distances: true }, { imageId: img.id, roi: null });
  };

  const controls = (
    <div className="vol-controls">
      <div className="row" style={{ marginTop: 2 }}>
        <span className="seg" title="Rendering mode">
          {(["mip", "alpha", "iso", "mesh"] as const).map((m) => (
            <button key={m} className={mode === m ? "on" : ""} onClick={() => set({ mode: m })}>{m === "mip" ? "MIP" : m === "alpha" ? "Alpha" : m === "iso" ? "Iso" : "Mesh"}</button>
          ))}
        </span>
        <span className="grow" />
        <span className="seg" title={mode === "mesh" ? "Mesh resolution (longest axis, voxels)" : "Volume resolution (longest axis, voxels)"}>
          {([128, 256, 384] as const).map((r) => (
            <button key={r} className={resolution === r ? "on" : ""} onClick={() => set({ resolution: r })}>{r}</button>
          ))}
        </span>
      </div>
      <div className="btn-row" style={{ marginTop: 4 }}>
        <button className={"btn sm" + (autoRotate ? " primary" : "")} onClick={() => set({ autoRotate: !autoRotate })} title="Auto-rotate">{autoRotate ? <Pause size={12} /> : <Play size={12} />} Spin</button>
        <button className="btn sm" onClick={reset} title="Reset the camera"><RotateCcw size={12} /> Reset</button>
        <button className={"btn sm" + (showBox ? " primary" : "")} onClick={() => set({ showBox: !showBox })} title="Show the bounding box"><Box size={12} /></button>
        <button className="btn sm" onClick={snapshot} disabled={!volume || snapBusy} title="Open a snapshot of this view as a new image"><Camera size={12} /> Snapshot</button>
        <span className="grow" />
        {large ? (
          <button className="btn sm" onClick={() => set({ expanded: false })} title="Close the large view (Esc)"><X size={12} /> Close</button>
        ) : (
          <button className="btn sm" onClick={() => set({ expanded: true })} title="Open in a large view"><Maximize2 size={12} /></button>
        )}
      </div>
      {mode === "mesh" && (
        <div className="vol-mesh-bar">
          <div className="row">
            <button className={"btn sm" + (wireframe ? " primary" : "")} onClick={() => set({ wireframe: !wireframe })} title="Draw the triangle edges on the surface"><Grid3x3 size={12} /> Wire</button>
            <span className="seg" title="Marching-cubes step (voxels): larger = coarser, faster surface">
              {([1, 2, 3] as const).map((st) => (
                <button key={st} className={meshStep === st ? "on" : ""} onClick={() => set({ meshStep: st })}>{st}</button>
              ))}
            </span>
            <label className="vol-mesh-smooth" title="Smoothing iterations (Taubin: no shrinkage)">
              Smooth
              <input type="range" min={0} max={30} step={1} value={meshSmooth} onChange={(e) => set({ meshSmooth: Number(e.target.value) })} />
              <span className="val">{meshSmooth}</span>
            </label>
          </div>
          <div className="row">
            <select value={meshFormat} onChange={(e) => set({ meshFormat: e.target.value as MeshFormat })} title="Mesh file format" className="vol-mesh-format">
              <option value="obj">OBJ</option>
              <option value="stl">STL</option>
              <option value="ply">PLY</option>
            </select>
            <button className="btn sm" onClick={exportMesh} disabled={!hasMesh || meshLoading} title="Download the visible surfaces (calibrated units when the image is scaled)"><Download size={12} /> Export {meshFormat.toUpperCase()}…</button>
            <button className="btn sm" onClick={measureMesh} disabled={!hasMesh || meshLoading} title="Surface area and enclosed volume per connected component → Results"><Ruler size={12} /> Measure</button>
          </div>
        </div>
      )}
      <div className="vol-channels">
        {chSettings.map((cs, i) => (
          <div className="vol-ch" key={i}>
            <div className="vol-ch-head">
              <button className="icon-btn" style={{ width: 22, height: 22 }} title={cs.visible ? "Hide channel" : "Show channel"} onClick={() => setChannel(i, { visible: !cs.visible })}>{cs.visible ? <Eye size={14} /> : <EyeOff size={14} />}</button>
              <span className="sw" style={{ background: lutColor(luts, lutNames[i]) }} />
              <span className="nm">{names[i]}</span>
            </div>
            <div className={"vol-sliders" + (mode === "alpha" || mode === "mesh" ? " three" : "")}>
              <label title={mode === "iso" || mode === "mesh" ? "Iso-surface level" : "Suppress intensities below this level"}>{mode === "iso" || mode === "mesh" ? "Level" : "Cut"}</label>
              {mode === "iso" || mode === "mesh" ? (
                <input type="range" min={0.02} max={0.98} step={0.01} value={cs.isoLevel} onChange={(e) => setChannel(i, { isoLevel: Number(e.target.value) })} />
              ) : (
                <input type="range" min={0} max={0.98} step={0.01} value={cs.threshold} onChange={(e) => setChannel(i, { threshold: Number(e.target.value) })} />
              )}
              {mode === "alpha" && (
                <>
                  <label title="Opacity (density) in translucent mode">Opacity</label>
                  <input type="range" min={0.02} max={1} step={0.01} value={cs.opacity} onChange={(e) => setChannel(i, { opacity: Number(e.target.value) })} />
                </>
              )}
              {mode === "mesh" && (
                <>
                  <label title="Surface opacity">Opacity</label>
                  <input type="range" min={0.05} max={1} step={0.01} value={cs.meshOpacity} onChange={(e) => setChannel(i, { meshOpacity: Number(e.target.value) })} />
                </>
              )}
              <label title="Brightness">Gain</label>
              <input type="range" min={0.1} max={3} step={0.05} value={cs.brightness} onChange={(e) => setChannel(i, { brightness: Number(e.target.value) })} />
            </div>
            {mode === "mesh" && cs.visible && meshes[i] && (
              <div className="vol-mesh-info" title={meshes[i]!.notes || undefined}>
                {meshes[i]!.nFaces > 0
                  ? `${formatCount(meshes[i]!.nVertices)} vertices · ${formatCount(meshes[i]!.nFaces)} faces · ${meshes[i]!.components} part${meshes[i]!.components === 1 ? "" : "s"} · level ${meshes[i]!.level.toPrecision(5)}`
                  : `no surface at level ${meshes[i]!.level.toPrecision(5)}`}
              </div>
            )}
          </div>
        ))}
      </div>
      <SceneControls img={img} images={images} landmarkRows={landmarkRows} recording={recording} freehand={freehand} lastAnimation={lastAnimation} onPreset={onPreset} onRecord={(a, f, d, fps) => void recordTurntable(a, f, d, fps)} onStop={stopRecording} onFreehand={(on, fps) => void toggleFreehand(on, fps)} onSnapshot={(w, h, ss) => void snapshotAt(w, h, ss)} onSaveScene={saveScene} onLoadScene={loadScene} onLandmarksToResults={landmarksToResults} />
      {volume && mode !== "mesh" && (
        <div className="hint-text" style={{ marginBottom: 0 }}>
          {volume.shape[2]} × {volume.shape[1]} × {volume.shape[0]} voxels · z/xy {volume.spacing.toFixed(2)} · Drag to orbit, wheel to zoom, ⇧-drag / right-drag to pan, double-click to reset.
        </div>
      )}
      {mode === "mesh" && (
        <div className="hint-text" style={{ marginBottom: 0 }}>
          {hasMesh ? `${formatCount(meshTotals.v)} vertices · ${formatCount(meshTotals.f)} faces · ` : ""}Marching cubes at the Level of each channel (same scale as Iso). Drag to orbit, wheel to zoom, ⇧-drag / right-drag to pan, double-click to reset.
        </div>
      )}
    </div>
  );

  return (
    <div className={"vol-viewer" + (large ? " large" : "")}>
      <div className="vol-canvas-wrap" ref={wrapRef}>
        <canvas ref={canvasRef} className={"vol-canvas" + (scene.placing ? " placing" : "")} />
        <div className="vol-landmark-labels" ref={labelsRef} aria-hidden>
          {scene.landmarks.map((lm) => (
            <div key={lm.id} className={"vol-landmark-label" + (lm.id === scene.selectedLandmark ? " sel" : "")} data-id={lm.id} style={{ color: lm.color || "#ffd54a" }}>{lm.name}</div>
          ))}
        </div>
        {!loading && !recording && !freehand && scene.placing && <div className="vol-overlay">Click a surface to place a landmark</div>}
        {recording && <div className="vol-overlay rec">Recording {recording.done}/{recording.total}…</div>}
        {freehand && !recording && <div className="vol-overlay rec">● Recording — orbit the view, then Stop</div>}
        {loading && <div className="vol-overlay">Loading volume…</div>}
        {!loading && mode === "mesh" && meshLoading && <div className="vol-overlay">Computing surface…</div>}
        {error && <div className="vol-overlay err">{error}</div>}
        {!error && mode === "mesh" && meshError && <div className="vol-overlay err">{meshError}</div>}
      </div>
      {controls}
    </div>
  );
}
