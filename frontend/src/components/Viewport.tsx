import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FolderOpen, Image as ImageIcon, Sparkles, ZoomIn, ZoomOut, Maximize } from "lucide-react";
import { useStore, getPlanes } from "../store";
import { composite, lutColor } from "../render/compositor";
import { viewportRef } from "../actions";
import { doAction } from "../actions";
import type { ImageSummary, OpSpec, ParamSpec, PlaneData, Roi, Tool, ViewTransform } from "../types";
import { pluginToolId } from "../types";
import { api } from "../api";
import { rgbToHex, valueToHex } from "./ColorPicker";
import DimensionBar from "./DimensionBar";
import { chooseStride } from "../render/stride";
import { kbd } from "../platform";
import OperationSheet from "./OperationSheet";
import { ArrowShape, CalibrationBar, ImageOverlay, TextShape, TimeStamp, TracksLayer, anchor, measureText, hitTestTracks, onPlane, trackColoring, trackStatRows, trackUnits, type ArrowItem, type CalibrationBarItem, type ImageOverlayItem, type TextItem, type TimeStampItem, type TrackItem, type TrackLegend } from "./OverlayItems";
import { toHex6 } from "../prefs";
import { captureBrushStroke } from "../classifierStore";
import { trackSignature, useTrackStore, type ModelSpot } from "../trackStore";
import { commandCount } from "../menus";


type Drag =
  | { kind: "pan"; sx: number; sy: number; view: ViewTransform }
  | { kind: "draw"; ix: number; iy: number; tool: string; points?: number[][] }
  | { kind: "move"; ix: number; iy: number; roi: Roi }
  | { kind: "resize"; handle: string; roi: Roi; ix: number; iy: number }
  | { kind: "vertex"; index: number; roi: Roi }
  | { kind: "stroke"; points: number[][]; erase: boolean }
  | { kind: "arrow"; ix: number; iy: number }
  /** A plugin tool (`op:<id>`): press position, modifiers; `started` once the pointer moved far enough for a drag. */
  | { kind: "plugin"; opId: string; ix: number; iy: number; button: number; shift: boolean; alt: boolean; ctrl: boolean; started: boolean };

const HANDLE_PX = 7;

/** Tools whose options are shown in the strip above the canvas. */
const TOOLS_WITH_OPTIONS = new Set<Tool>(["brush", "bucket", "wand", "text", "arrow", "line", "polyline", "angle", "rect", "point", "track"]);

/** Polygon approximating a rectangle with rounded corners (clockwise, `segs` segments per corner). */
export function roundedRectPolygon(x: number, y: number, w: number, h: number, radius: number, segs = 6): number[][] {
  const r = Math.max(0, Math.min(radius, w / 2, h / 2));
  const r2 = (v: number) => Math.round(v * 100) / 100;
  if (r <= 0) return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
  const pts: number[][] = [];
  const corner = (cx: number, cy: number, a0: number) => {
    for (let i = 0; i <= segs; i++) {
      const a = a0 + (Math.PI / 2) * (i / segs);
      pts.push([r2(cx + r * Math.cos(a)), r2(cy + r * Math.sin(a))]);
    }
  };
  corner(x + w - r, y + r, -Math.PI / 2); // top-right
  corner(x + w - r, y + h - r, 0); // bottom-right
  corner(x + r, y + h - r, Math.PI / 2); // bottom-left
  corner(x + r, y + r, Math.PI); // top-left
  return pts;
}

/**
 * The id of a freshly created manual track: a number in the op message that names an existing
 * "manual-<n>" overlay item, else the newest manual-<n> item, else the first number in the message.
 */
function parseTrackId(message: string | undefined, overlay: unknown[] | undefined): number | null {
  const ids = new Set<number>();
  let newest: number | null = null;
  for (const o of (overlay || []) as Record<string, unknown>[]) {
    const mm = /^manual-(\d+)$/.exec(String(o?.id ?? ""));
    if (!mm) continue;
    const n = Number(mm[1]);
    ids.add(n);
    if (newest === null || n > newest) newest = n;
  }
  const nums = Array.from((message || "").matchAll(/\d+/g), (m) => Number(m[0]));
  const named = nums.find((n) => ids.has(n));
  if (named !== undefined) return named;
  if (newest !== null) return newest;
  return nums.length ? nums[0] : null;
}

/** Start a new manual track on the active image (tracking.manual_new_track); returns its id. */
async function startManualTrack(): Promise<number | null> {
  const st = useStore.getState();
  const img = st.activeImage();
  if (!img) {
    st.toast("Open an image first", "error");
    return null;
  }
  const spec = st.ops["tracking.manual_new_track"];
  if (!spec) {
    st.toast("Manual tracking commands not available", "error");
    return null;
  }
  const res = await st.runOp("tracking.manual_new_track", pickParams(spec, {}), { silent: true, roi: null, imageId: img.id });
  if (!res) return null;
  const id = parseTrackId(res.message, useStore.getState().image(img.id)?.overlay);
  if (id === null) {
    useStore.getState().toast("Manual track created, but its id could not be read from the reply", "error");
    return null;
  }
  useStore.getState().setManualTrack({ id });
  useStore.getState().toast(`Manual track ${id} started – click to add points`, "info");
  return id;
}

interface PendingStroke { points: number[][]; width: number; erase: boolean; rev: number; color: string }

export default function Viewport() {
  const img = useStore((s) => s.activeImage());
  const state = useStore((s) => s.state);
  const tool = useStore((s) => s.tool);
  const spaceDown = useStore((s) => s.spaceDown);
  const luts = useStore((s) => s.luts);
  const views = useStore((s) => s.views);
  const selection = useStore((s) => (img ? s.selections[img.id] ?? null : null));
  const setSelection = useStore((s) => s.setSelection);
  const previewPlanes = useStore((s) => s.previewPlanes);
  const previewFor = useStore((s) => s.previewFor);
  const showManagerRois = useStore((s) => s.showManagerRois);
  const selectedRoiIds = useStore((s) => s.selectedRoiIds);
  const wandTolerance = useStore((s) => s.wandTolerance);
  const sheet = useStore((s) => s.sheet);
  const brushSize = useStore((s) => s.brushSize);
  const toolOptions = useStore((s) => s.toolOptions);
  const showOverlay = useStore((s) => s.showOverlay);
  const showTrackLabels = useStore((s) => s.showTrackLabels);
  const showTrackFuture = useStore((s) => s.showTrackFuture);
  const prefs = useStore((s) => s.prefs);
  const manualTrack = useStore((s) => s.manualTrack);
  const selectedTrackIds = useStore((s) => s.selectedTrackIds);
  const trackColorBy = useStore((s) => s.trackColorBy);
  const trackLabelMode = useStore((s) => s.trackLabelMode);
  const trackShowOnlySelected = useStore((s) => s.trackShowOnlySelected);
  // display mode / tail length set by tracking.set_display (served with the track model of this image)
  const trackDisplay = useTrackStore((s) => (img && s.model?.image === img.id ? s.model.display : null));
  // load the track model (display settings, lineage) for an image that already carries track overlays, e.g. after a
  // page reload with the Tracks section collapsed
  const trackSig = img ? trackSignature(img.overlay as unknown[]) : "";
  useEffect(() => {
    if (img && trackSig) void useTrackStore.getState().fetchModel(img.id, trackSig);
  }, [img?.id, trackSig]);
  const tracksTable = useStore((s) => s.state?.tables?.Tracks);
  // Tracks wizard detection preview (E8-4): the unlinked spots of the current frame, drawn on their own frame only
  const spotPreview = useSpotPreview(img);
  // track under the pointer (pointer tool); its name floats next to the cursor via tipRef
  const [hoverTrack, setHoverTrack] = useState<{ id: string; name: string } | null>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const tipPosRef = useRef({ sx: 0, sy: 0 }); // last pointer position (screen px): where the tip mounts

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const planesRef = useRef<PlaneData | null>(null);
  const rgbaRef = useRef<Uint8ClampedArray<ArrayBuffer> | null>(null);
  const dragRef = useRef<Drag | null>(null);
  const lastDimsRef = useRef<Record<string, string>>({});
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [planesVersion, setPlanesVersion] = useState(0);
  const [inProgress, setInProgress] = useState<{ type: "polygon" | "polyline"; points: number[][] } | null>(null);
  const [hoverPt, setHoverPt] = useState<number[] | null>(null);
  const [dropping, setDropping] = useState(false);
  const [draftRoi, setDraftRoi] = useState<Roi | null>(null);
  // drawing tools
  const [stroke, setStroke] = useState<{ points: number[][]; erase: boolean } | null>(null);
  const [pendingStrokes, setPendingStrokes] = useState<PendingStroke[]>([]);
  const [brushPt, setBrushPt] = useState<number[] | null>(null);
  const [draftArrow, setDraftArrow] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const [anglePts, setAnglePts] = useState<number[][]>([]);
  const [textEdit, setTextEdit] = useState<{ ix: number; iy: number; value: string } | null>(null);
  const textEditRef = useRef(textEdit); // guards against a double commit (Enter followed by blur)
  textEditRef.current = textEdit;
  // plugin tools (tool rail): the drag in progress, drawn as a dashed line
  const [toolDrag, setToolDrag] = useState<{ x1: number; y1: number; x2: number; y2: number } | null>(null);
  const pid = pluginToolId(tool);
  const pluginSpec = useStore((s) => (pid ? s.ops[pid] : undefined));
  // Image ▸ Overlay ▸ Overlay Options… (overlay_* settings): default fill / stroke width and how labels are drawn
  const settings = useStore((s) => s.settings);
  const ovOpts = useMemo(() => overlayOptions(settings), [settings]);

  const view: ViewTransform = (img && views[img.id]) || { scale: 1, tx: 0, ty: 0 };
  const threshold = img ? state?.thresholds[img.id] : undefined;
  const stride = img ? chooseStride(img, view.scale) : 1;

  // ---------------------------------------------------------- view (with optional sync)
  /** Set the view of an image; when "Synchronize Views" is on, every open image of the same size follows. */
  const setView = useCallback((imageId: string, v: ViewTransform) => {
    const st = useStore.getState();
    if (st.syncViews && st.state) {
      const me = st.image(imageId);
      const patch: Record<string, ViewTransform> = { [imageId]: v };
      if (me) for (const im of st.state.images) if (im.id !== imageId && im.shape.x === me.shape.x && im.shape.y === me.shape.y) patch[im.id] = v;
      st.setViews(patch);
    } else st.setView(imageId, v);
  }, []);

  // ---------------------------------------------------------- sizing / fit
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  // the view each image was last auto-fitted to: while the user has not zoomed or panned away from it, a change of
  // the container's size (the channel chips / tool-options bar taking their height after the first image opened,
  // the inspector resized, the split view) re-fits instead of leaving the bottom of the image behind the strip (E8-6)
  const autoFitRef = useRef<Record<string, ViewTransform>>({});
  const fit = useCallback(() => {
    if (!img || !size.w) return;
    const s = Math.min(size.w / img.shape.x, size.h / img.shape.y) * 0.96;
    const v = { scale: s, tx: (size.w - img.shape.x * s) / 2, ty: (size.h - img.shape.y * s) / 2 };
    autoFitRef.current[img.id] = v;
    setView(img.id, v);
  }, [img, size, setView]);

  const fitRect = useCallback((x: number, y: number, w: number, h: number) => {
    if (!img || !size.w) return;
    const ww = Math.max(w, 1);
    const hh = Math.max(h, 1);
    const s = Math.min(64, Math.min(size.w / ww, size.h / hh) * 0.85);
    setView(img.id, { scale: s, tx: size.w / 2 - (x + ww / 2) * s, ty: size.h / 2 - (y + hh / 2) * s });
  }, [img, size, setView]);

  useEffect(() => {
    if (!img || !size.w) return;
    const dims = `${img.shape.x}x${img.shape.y}`;
    const cur = views[img.id];
    const auto = autoFitRef.current[img.id];
    // still at the auto-fit view (no manual zoom / pan since): follow the container's size (E8-6)
    const untouched = !!cur && !!auto && cur.scale === auto.scale && cur.tx === auto.tx && cur.ty === auto.ty;
    if (!cur || lastDimsRef.current[img.id] !== dims || untouched) {
      lastDimsRef.current[img.id] = dims;
      fit();
    }
  }, [img?.id, img?.shape.x, img?.shape.y, size.w, size.h]); // eslint-disable-line react-hooks/exhaustive-deps

  const zoomAt = useCallback((factor: number, sx?: number, sy?: number) => {
    if (!img) return;
    const v = useStore.getState().views[img.id] || view;
    const cx = sx ?? size.w / 2;
    const cy = sy ?? size.h / 2;
    const ns = Math.min(64, Math.max(0.02, v.scale * factor));
    const k = ns / v.scale;
    setView(img.id, { scale: ns, tx: cx - (cx - v.tx) * k, ty: cy - (cy - v.ty) * k });
  }, [img, view, size, setView]);

  useEffect(() => {
    viewportRef.current = {
      zoomBy: (f) => zoomAt(f),
      fit,
      fitRect,
      setScale: (s) => {
        if (!img) return;
        const v = useStore.getState().views[img.id] || view;
        zoomAt(s / v.scale);
      },
    };
  }, [zoomAt, fit, fitRect, img, view]);

  // ---------------------------------------------------------- planes
  useEffect(() => {
    if (!img) {
      planesRef.current = null;
      return;
    }
    let alive = true;
    getPlanes(img, img.display.t, img.display.z, stride).then((pd) => {
      if (!alive) return;
      planesRef.current = pd;
      setPlanesVersion((v) => v + 1);
      // strokes submitted before this revision are now burnt into the pixels
      setPendingStrokes((ps) => (ps.length ? ps.filter((s) => s.rev >= img.rev) : ps));
    }).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [img?.id, img?.rev, img?.display.t, img?.display.z, stride]); // eslint-disable-line react-hooks/exhaustive-deps

  // reset tool-local state when the image or the tool changes
  useEffect(() => {
    setAnglePts([]);
    setTextEdit(null);
    setDraftArrow(null);
    setStroke(null);
    setBrushPt(null);
    setHoverTrack(null);
    // an unfinished polygon/polyline survives switching between those two tools only
    if (tool !== "polygon" && tool !== "polyline") {
      setInProgress(null);
      setHoverPt(null);
    }
  }, [tool, img?.id]);

  // an unfinished polygon/polyline is in the coordinates of the image it was started on:
  // drop it whenever the active image changes, whatever the tool
  useEffect(() => {
    setInProgress(null);
    setHoverPt(null);
    if (useStore.getState().manualTrack.id !== null) useStore.getState().setManualTrack({ id: null });
    // track ids are per image ("track-0" exists on every tracked image): the selection does not carry over
    setHoverTrack(null);
    if (useStore.getState().selectedTrackIds.length) useStore.getState().setSelectedTrackIds([]);
  }, [img?.id]);

  // ---------------------------------------------------------- tracks (selection, hover, colouring)
  const allTracks = useMemo(() => ((img?.overlay || NO_OVERLAY) as unknown as Record<string, any>[]).filter((o) => o.type === "track") as unknown as TrackItem[], [img?.overlay]);
  const tracks = showOverlay ? allTracks : NO_TRACKS;
  // tracks removed by Filter Tracks / Delete / Clear Tracks leave the selection
  useEffect(() => {
    const sel = useStore.getState().selectedTrackIds;
    if (!sel.length) return;
    const ids = new Set(allTracks.map((tr) => tr.id));
    const kept = sel.filter((id) => ids.has(id));
    if (kept.length !== sel.length) useStore.getState().setSelectedTrackIds(kept);
    setHoverTrack((h) => (h && !ids.has(h.id) ? null : h));
  }, [allTracks]);
  const trackColors = useMemo(() => {
    if (!tracks.length || trackColorBy === "track") return { colors: null, legend: null as TrackLegend | null };
    const rows = trackStatRows(tracks, tracksTable);
    return trackColoring(rows, trackColorBy, luts, trackUnits(img?.calibration, tracksTable));
  }, [tracks, trackColorBy, tracksTable, luts, img?.calibration]);
  const selectedTrackSet = useMemo(() => new Set(selectedTrackIds), [selectedTrackIds]);
  /** Track drawn under image point (ix, iy) within ~6 screen px, honouring "show only selected". */
  const trackAt = (ix: number, iy: number): TrackItem | null => {
    if (!img || !tracks.length) return null;
    const only = trackShowOnlySelected && selectedTrackSet.size > 0;
    const id = hitTestTracks(tracks, ix, iy, 6 / view.scale, img.display.t, showTrackFuture, only ? (tid) => selectedTrackSet.has(tid) : undefined);
    return id === null ? null : tracks.find((tr) => tr.id === id) || null;
  };
  const trackHoverEnabled = tool === "pointer" && tracks.length > 0;
  const updateHover = (ix: number, iy: number, sx: number, sy: number) => {
    const tr = trackHoverEnabled && !dragRef.current ? trackAt(ix, iy) : null;
    tipPosRef.current = { sx, sy };
    const tip = tipRef.current;
    if (tip) {
      // moved directly: a pointer move over a track must not re-render the viewport
      tip.style.left = `${sx}px`;
      tip.style.top = `${sy}px`;
    }
    setHoverTrack((h) => (tr ? (h && h.id === tr.id ? h : { id: tr.id, name: tr.name || tr.id }) : h ? null : h));
  };

  // ---------------------------------------------------------- composite
  const usePreview = !!(previewPlanes && img && previewFor === img.id);
  useEffect(() => {
    const pd = usePreview ? previewPlanes : planesRef.current;
    if (!img || !pd) return;
    const [, h, w] = pd.shape;
    let off = offRef.current;
    if (!off) off = offRef.current = document.createElement("canvas");
    if (off.width !== w || off.height !== h) {
      off.width = w;
      off.height = h;
      rgbaRef.current = null;
    }
    // the threshold overlay follows the thresholded channel's visibility: a hidden channel shows no red (E5-5)
    const thrCh = img.display.channels[img.display.c];
    const thrVisible = img.rgb || img.display.mode !== "composite" || !thrCh || thrCh.visible;
    const rgba = composite(pd.channels, w, h, img.display, img.rgb, luts, { threshold: thrVisible ? threshold ?? null : null, thresholdChannel: img.display.c }, rgbaRef.current ?? undefined);
    rgbaRef.current = rgba;
    const ctx = off.getContext("2d")!;
    ctx.putImageData(new ImageData(rgba, w, h), 0, 0);
    draw();
  }, [planesVersion, img?.display, img?.rgb, luts, threshold, usePreview, previewPlanes]); // eslint-disable-line react-hooks/exhaustive-deps

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    const off = offRef.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== size.w * dpr || cv.height !== size.h * dpr) {
      cv.width = size.w * dpr;
      cv.height = size.h * dpr;
      cv.style.width = size.w + "px";
      cv.style.height = size.h + "px";
    }
    const ctx = cv.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size.w, size.h);
    if (!img || !off) return;
    const v = useStore.getState().views[img.id] || view;
    const effScale = v.scale * (off.width ? img.shape.x / off.width : 1);
    // zoomed out: always filter; zoomed in: crisp pixels unless Preferences ▸ Interpolate zoomed images
    ctx.imageSmoothingEnabled = effScale < 1 || prefs.interpolate_zoom;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, v.tx, v.ty, img.shape.x * v.scale, img.shape.y * v.scale);
  }, [img, size, view, prefs.interpolate_zoom]);

  useEffect(() => {
    const id = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(id);
  }, [draw, view.scale, view.tx, view.ty, size.w, size.h]);

  // ---------------------------------------------------------- coordinates
  const toImage = useCallback((e: { clientX: number; clientY: number }) => {
    const rect = wrapRef.current!.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const v = img ? useStore.getState().views[img.id] || view : view;
    return { sx, sy, ix: (sx - v.tx) / v.scale, iy: (sy - v.ty) / v.scale };
  }, [img, view]);

  const clampPt = (x: number, y: number): [number, number] => (img ? [Math.min(Math.max(x, 0), img.shape.x), Math.min(Math.max(y, 0), img.shape.y)] : [x, y]);

  // ---------------------------------------------------------- wheel
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (!img) return;
      const st = useStore.getState();
      if (e.shiftKey && img.shape.z > 1) {
        const dz = Math.sign(e.deltaY || e.deltaX);
        st.setPosition(img.id, { z: Math.min(img.shape.z - 1, Math.max(0, img.display.z + dz)) });
        return;
      }
      if (e.altKey && img.shape.t > 1) {
        const dt = Math.sign(e.deltaY || e.deltaX);
        st.setPosition(img.id, { t: Math.min(img.shape.t - 1, Math.max(0, img.display.t + dt)) });
        return;
      }
      const { sx, sy } = toImage(e);
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0022));
      zoomAt(factor, sx, sy);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [img, toImage, zoomAt]);

  // ---------------------------------------------------------- roi helpers
  const handlesFor = (roi: Roi | null): { id: string; x: number; y: number }[] => {
    if (!roi) return [];
    if (roi.type === "rect" || roi.type === "oval") {
      const { x = 0, y = 0, w = 0, h = 0 } = roi;
      return [
        { id: "nw", x, y }, { id: "n", x: x + w / 2, y }, { id: "ne", x: x + w, y }, { id: "e", x: x + w, y: y + h / 2 },
        { id: "se", x: x + w, y: y + h }, { id: "s", x: x + w / 2, y: y + h }, { id: "sw", x, y: y + h }, { id: "w", x, y: y + h / 2 },
      ];
    }
    if (roi.type === "line") return [{ id: "p0", x: roi.x1!, y: roi.y1! }, { id: "p1", x: roi.x2!, y: roi.y2! }];
    if (roi.points && roi.points.length <= 120 && roi.type !== "freehand") return roi.points.map((p, i) => ({ id: `v${i}`, x: p[0], y: p[1] }));
    return [];
  };

  const hitHandle = (roi: Roi | null, sx: number, sy: number): string | null => {
    const v = view;
    for (const h of handlesFor(roi)) {
      const hx = h.x * v.scale + v.tx;
      const hy = h.y * v.scale + v.ty;
      if (Math.abs(hx - sx) <= HANDLE_PX && Math.abs(hy - sy) <= HANDLE_PX) return h.id;
    }
    return null;
  };

  const inside = (roi: Roi | null, ix: number, iy: number): boolean => {
    if (!roi) return false;
    const tol = 6 / view.scale;
    if (roi.type === "rect" || roi.type === "mask") return ix >= roi.x! && ix <= roi.x! + roi.w! && iy >= roi.y! && iy <= roi.y! + roi.h!;
    if (roi.type === "oval") {
      const cx = roi.x! + roi.w! / 2;
      const cy = roi.y! + roi.h! / 2;
      const a = roi.w! / 2 || 1e-6;
      const b = roi.h! / 2 || 1e-6;
      return ((ix - cx) / a) ** 2 + ((iy - cy) / b) ** 2 <= 1;
    }
    if (roi.type === "line") return distToSeg(ix, iy, roi.x1!, roi.y1!, roi.x2!, roi.y2!) <= tol;
    if (roi.type === "polyline") {
      const p = roi.points!;
      for (let i = 0; i < p.length - 1; i++) if (distToSeg(ix, iy, p[i][0], p[i][1], p[i + 1][0], p[i + 1][1]) <= tol) return true;
      return false;
    }
    if (roi.type === "point") return roi.points!.some((p) => Math.hypot(p[0] - ix, p[1] - iy) <= tol);
    return pointInPolygon(ix, iy, roi.points || []);
  };

  const moveRoi = (roi: Roi, dx: number, dy: number): Roi => {
    if (roi.type === "rect" || roi.type === "oval" || roi.type === "mask") return { ...roi, x: roi.x! + dx, y: roi.y! + dy };
    if (roi.type === "line") return { ...roi, x1: roi.x1! + dx, y1: roi.y1! + dy, x2: roi.x2! + dx, y2: roi.y2! + dy };
    return { ...roi, points: roi.points!.map((p) => [p[0] + dx, p[1] + dy]) };
  };

  const resizeRoi = (roi: Roi, handle: string, ix: number, iy: number, shift: boolean): Roi => {
    if (roi.type === "line") return handle === "p0" ? { ...roi, x1: ix, y1: iy } : { ...roi, x2: ix, y2: iy };
    let { x = 0, y = 0, w = 0, h = 0 } = roi;
    const x2 = x + w;
    const y2 = y + h;
    let nx = x, ny = y, nx2 = x2, ny2 = y2;
    if (handle.includes("w")) nx = ix;
    if (handle.includes("e")) nx2 = ix;
    if (handle.includes("n")) ny = iy;
    if (handle.includes("s")) ny2 = iy;
    if (shift) {
      const s = Math.max(Math.abs(nx2 - nx), Math.abs(ny2 - ny));
      if (handle.includes("w")) nx = nx2 - s; else nx2 = nx + s;
      if (handle.includes("n")) ny = ny2 - s; else ny2 = ny + s;
    }
    return { ...roi, x: Math.min(nx, nx2), y: Math.min(ny, ny2), w: Math.abs(nx2 - nx), h: Math.abs(ny2 - ny) };
  };

  const commitSelection = (roi: Roi | null) => {
    if (!img) return;
    setSelection(img.id, roi);
  };

  // ---------------------------------------------------------- drawing-tool commits
  /** Colour the live brush stroke the way the painted value will look (top of the current LUT). */
  const strokeColor = (erase: boolean): string => {
    if (erase) return "#000000";
    if (!img || img.rgb) return "#ffffff";
    const ch = img.display.channels[Math.min(img.display.c, img.display.channels.length - 1)];
    return ch && img.display.mode !== "grayscale" ? lutColor(luts, ch.lut) : "#ffffff";
  };

  const commitStroke = (points: number[][], erase: boolean) => {
    if (!img || !points.length) return;
    if (captureBrushStroke(points, useStore.getState().brushSize)) return; // Pixel Classifier trace mode: the stroke becomes a class trace
    const st = useStore.getState();
    const width = st.brushSize;
    const color = strokeColor(erase);
    const pending: PendingStroke = { points, width, erase, rev: img.rev, color };
    setPendingStrokes((ps) => [...ps, pending]);
    const pts = points.map((p) => [Math.round(p[0] * 100) / 100, Math.round(p[1] * 100) / 100]);
    void st.runOp("edit.paint_stroke", { points: JSON.stringify(pts), width, erase, all_channels: st.toolOptions.allChannels }, { silent: true, roi: null }).then((res) => {
      if (!res) setPendingStrokes((ps) => ps.filter((s) => s !== pending));
    });
  };

  /**
   * Add an overlay item: through the dedicated registry op when another module provides it
   * (`extra` holds op-only params, `roi` the selection it wants), else via overlay.add_item.
   */
  const addOverlayItem = (preferredOp: string, item: Record<string, unknown>, extra: Record<string, unknown> = {}, roi: Roi | null = null) => {
    const st = useStore.getState();
    const spec = st.ops[preferredOp];
    if (spec) void st.runOp(preferredOp, pickParams(spec, { ...item, ...extra }), { silent: true, roi });
    else void st.runOp("overlay.add_item", { item: JSON.stringify(item) }, { silent: true, roi: null });
  };

  const commitText = (te: { ix: number; iy: number; value: string } | null) => {
    if (!te || te !== textEditRef.current) return; // already committed or cancelled
    textEditRef.current = null;
    setTextEdit(null);
    if (!img) return;
    const text = te.value.replace(/\s+$/, "");
    if (!text.trim()) return;
    const o = useStore.getState().toolOptions;
    addOverlayItem("overlay.text", { type: "text", x: Math.round(te.ix * 10) / 10, y: Math.round(te.iy * 10) / 10, text, font_size: o.fontSize, color: o.textColor, background: o.textBackground }, { at_selection: false });
  };

  const commitArrow = (a: { x1: number; y1: number; x2: number; y2: number }) => {
    if (!img) return;
    const o = useStore.getState().toolOptions;
    const r = (v: number) => Math.round(v * 10) / 10;
    const line: Roi = { type: "line", x1: r(a.x1), y1: r(a.y1), x2: r(a.x2), y2: r(a.y2), width: 1 };
    addOverlayItem("overlay.arrow", { type: "arrow", x1: line.x1, y1: line.y1, x2: line.x2, y2: line.y2, color: o.arrowColor, stroke_width: o.arrowWidth, head_size: o.arrowHead }, { double_headed: false }, line);
  };

  const finishAngle = (pts: number[][]) => {
    setAnglePts([]);
    setHoverPt(null);
    if (!img) return;
    const roi: Roi = { type: "polyline", points: pts, width: 1 };
    commitSelection(roi);
    const st = useStore.getState();
    const spec = st.ops["analyze.angle"];
    if (spec) {
      const defaults: Record<string, unknown> = {};
      for (const p of spec.params) defaults[p.name] = p.type === "image" ? null : p.default;
      void st.runOp("analyze.angle", defaults, { roi });
    } else st.toast(`Angle: ${angleAt(pts[0], pts[1], pts[2]).toFixed(2)}°`, "info");
  };

  /** Manual tracking tool: add this frame's point to the active track (creating one first), ⌥ deletes it. */
  const trackClick = async (ix: number, iy: number, del: boolean) => {
    if (!img) return;
    const st = useStore.getState();
    const addSpec = st.ops["tracking.manual_add_point"];
    const delSpec = st.ops["tracking.manual_delete_point"];
    if (!addSpec || !st.ops["tracking.manual_new_track"]) return st.toast("Manual tracking commands not available", "error");
    const t = img.display.t;
    let id = st.manualTrack.id;
    if (del) {
      if (id === null) return st.toast("No active manual track – click to start one", "info");
      if (!delSpec) return st.toast("Manual tracking commands not available", "error");
      await st.runOp("tracking.manual_delete_point", pickParams(delSpec, { track_id: id, t, x: ix, y: iy }), { silent: true, roi: null, imageId: img.id });
      return;
    }
    if (id === null) {
      id = await startManualTrack();
      if (id === null) return;
    }
    const [cx, cy] = clampPt(ix, iy);
    const x = Math.round(cx * 100) / 100;
    const y = Math.round(cy * 100) / 100;
    const res = await useStore.getState().runOp("tracking.manual_add_point", pickParams(addSpec, { track_id: id, x, y, t }), { silent: true, roi: null, imageId: img.id });
    if (!res) return;
    const after = useStore.getState();
    const cur = after.image(img.id);
    if (after.manualTrack.autoAdvance && cur && t < cur.shape.t - 1) after.setPosition(img.id, { t: t + 1 });
  };

  /**
   * Run a plugin tool (`@op(..., tool=True)`) for an interaction: its own params come from the tool-options
   * bar (store.toolParams), the interaction is added as x, y, x2, y2, button, shift, alt, ctrl, event.
   */
  const runPluginTool = (opId: string, event: "click" | "drag-start" | "drag-end", x: number, y: number, mods: { button: number; shift: boolean; alt: boolean; ctrl: boolean }, x2?: number, y2?: number) => {
    const st = useStore.getState();
    const spec = st.ops[opId];
    if (!spec) return st.toast("This plugin tool is no longer registered – choose another tool", "error");
    if (spec.requires_image && !img) return st.toast("Open an image first", "error");
    const own = st.toolParams[opId] || {};
    const params: Record<string, unknown> = {};
    for (const p of spec.params) params[p.name] = p.type === "image" ? null : own[p.name] !== undefined ? own[p.name] : p.default;
    const r = (v: number) => Math.round(v * 100) / 100;
    Object.assign(params, { x: r(x), y: r(y), x2: x2 === undefined ? null : r(x2), y2: y2 === undefined ? null : r(y2), button: mods.button, shift: mods.shift, alt: mods.alt, ctrl: mods.ctrl, event });
    void st.runOp(opId, params, { scope: spec.kind === "plane" ? "plane" : "stack", silent: event === "drag-start" });
  };

  // ---------------------------------------------------------- pointer events
  const onPointerDown = (e: React.PointerEvent) => {
    if (!img) return;
    wrapRef.current?.focus();
    if (e.button === 1 || spaceDown || tool === "hand") {
      dragRef.current = { kind: "pan", sx: e.clientX, sy: e.clientY, view };
      (e.target as Element).setPointerCapture?.(e.pointerId);
      return;
    }
    if (pid) {
      // plugin tools get every button (right-click included; the context menu is suppressed)
      const { ix, iy } = toImage(e);
      const [cx, cy] = clampPt(ix, iy);
      dragRef.current = { kind: "plugin", opId: pid, ix: cx, iy: cy, button: e.button, shift: e.shiftKey, alt: e.altKey, ctrl: e.ctrlKey || e.metaKey, started: false };
      (e.target as Element).setPointerCapture?.(e.pointerId);
      return;
    }
    if (e.button !== 0) return;
    const { sx, sy, ix, iy } = toImage(e);
    const st = useStore.getState();

    if (tool === "zoom") {
      zoomAt(e.altKey ? 0.5 : 2, sx, sy);
      return;
    }
    if (tool === "picker") {
      const pd = planesRef.current;
      if (pd) {
        const [, h, w] = pd.shape;
        const px = Math.floor(ix / pd.stride);
        const py = Math.floor(iy / pd.stride);
        if (px >= 0 && py >= 0 && px < w && py < h) {
          // single-value ops: the other colour setting is left untouched (ImageJ: click = foreground, Alt-click = background);
          // the colour picker's hex companion (foreground_color / background_color) follows: the pixel's colour on RGB
          // images, the value's grey otherwise
          const which = e.altKey ? "background" : "foreground";
          let v = pd.channels[Math.min(img.display.c, pd.channels.length - 1)][py * w + px];
          let hex: string;
          if (img.rgb) {
            const [r, g, b] = [0, 1, 2].map((c) => pd.channels[Math.min(c, pd.channels.length - 1)][py * w + px]);
            v = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
            hex = rgbToHex(r, g, b);
          } else hex = valueToHex(v, img);
          void st.runOp(`edit.set_${which}`, { value: v }, { silent: true });
          void api.putSettings({ [`${which}_color`]: hex }).then((raw) => useStore.setState({ settings: raw || {} })).catch(() => undefined);
          st.toast(`${which === "background" ? "Background" : "Foreground"} ${img.rgb ? `colour set to ${hex}` : `value set to ${v}`}`, "success");
        }
      }
      return;
    }
    if (tool === "wand") {
      void st.runOp("selection.wand", { x: Math.floor(ix), y: Math.floor(iy), tolerance: wandTolerance, add: e.shiftKey }, { silent: true });
      return;
    }
    if (tool === "brush") {
      const [cx, cy] = clampPt(ix, iy);
      const erase = e.altKey || st.toolOptions.erase;
      dragRef.current = { kind: "stroke", points: [[cx, cy]], erase };
      setStroke({ points: [[cx, cy]], erase });
      (e.target as Element).setPointerCapture?.(e.pointerId);
      return;
    }
    if (tool === "bucket") {
      const px = Math.floor(ix);
      const py = Math.floor(iy);
      if (px < 0 || py < 0 || px >= img.shape.x || py >= img.shape.y) return;
      void st.runOp("edit.flood_fill", { x: px, y: py, tolerance: st.toolOptions.bucketTolerance, connectivity: st.toolOptions.bucketConnectivity }, { silent: true, roi: null });
      return;
    }
    if (tool === "text") {
      if (textEdit) commitText(textEdit);
      if (ix < 0 || iy < 0 || ix > img.shape.x || iy > img.shape.y) return;
      // keep the browser from moving focus to the canvas on the compat mousedown, which would blur the new input
      e.preventDefault();
      setTextEdit({ ix, iy, value: "" });
      return;
    }
    if (tool === "arrow") {
      const [cx, cy] = clampPt(ix, iy);
      dragRef.current = { kind: "arrow", ix: cx, iy: cy };
      (e.target as Element).setPointerCapture?.(e.pointerId);
      return;
    }
    if (tool === "track") {
      void trackClick(ix, iy, e.altKey);
      return;
    }
    if (tool === "angle") {
      const [cx, cy] = clampPt(ix, iy);
      const pts = [...anglePts, [cx, cy]];
      if (pts.length >= 3) finishAngle(pts);
      else {
        setAnglePts(pts);
        setHoverPt([cx, cy]);
      }
      return;
    }
    if (tool === "polygon" || tool === "polyline") {
      const [cx, cy] = clampPt(ix, iy);
      if (inProgress) {
        const first = inProgress.points[0];
        const closeHit = tool === "polygon" && Math.hypot((first[0] - ix) * view.scale, (first[1] - iy) * view.scale) < 8 && inProgress.points.length >= 3;
        if (closeHit) finishPolygon();
        else setInProgress({ ...inProgress, points: [...inProgress.points, [cx, cy]] });
      } else {
        // editing an existing polygon? check handles first
        const hid = hitHandle(selection, sx, sy);
        if (hid && hid.startsWith("v") && selection) {
          dragRef.current = { kind: "vertex", index: Number(hid.slice(1)), roi: selection };
          (e.target as Element).setPointerCapture?.(e.pointerId);
          return;
        }
        if (selection && inside(selection, ix, iy)) {
          dragRef.current = { kind: "move", ix, iy, roi: selection };
          (e.target as Element).setPointerCapture?.(e.pointerId);
          return;
        }
        setInProgress({ type: tool, points: [[cx, cy]] });
      }
      return;
    }
    if (tool === "point") {
      const [cx, cy] = clampPt(ix, iy);
      const hid = hitHandle(selection, sx, sy);
      if (hid && selection?.type === "point") {
        dragRef.current = { kind: "vertex", index: Number(hid.slice(1)), roi: selection };
        (e.target as Element).setPointerCapture?.(e.pointerId);
        return;
      }
      const pt = [Math.floor(cx) + 0.5, Math.floor(cy) + 0.5];
      if (e.shiftKey && selection?.type === "point") commitSelection({ ...selection, points: [...selection.points!, pt] });
      else commitSelection({ type: "point", points: [pt] });
      return;
    }
    // pointer / rect / oval / line / freehand
    const hid = hitHandle(selection, sx, sy);
    if (hid && selection) {
      if (hid.startsWith("v")) dragRef.current = { kind: "vertex", index: Number(hid.slice(1)), roi: selection };
      else dragRef.current = { kind: "resize", handle: hid, roi: selection, ix, iy };
      (e.target as Element).setPointerCapture?.(e.pointerId);
      return;
    }
    if (selection && inside(selection, ix, iy) && (tool === "pointer" || selection.type === tool || (tool === "freehand" && selection.type === "freehand"))) {
      dragRef.current = { kind: "move", ix, iy, roi: selection };
      (e.target as Element).setPointerCapture?.(e.pointerId);
      return;
    }
    if (tool === "pointer" && tracks.length) {
      // click on a track overlay selects it (⇧ toggles it in a multi-selection); a click elsewhere deselects
      const hit = trackAt(ix, iy);
      const sel = st.selectedTrackIds;
      if (hit) {
        if (e.shiftKey) st.setSelectedTrackIds(sel.includes(hit.id) ? sel.filter((id) => id !== hit.id) : [...sel, hit.id]);
        else st.setSelectedTrackIds([hit.id]);
        return;
      }
      if (sel.length && !e.shiftKey) st.setSelectedTrackIds([]);
    }
    const drawTool = tool === "pointer" ? "rect" : tool;
    const [cx, cy] = clampPt(ix, iy);
    dragRef.current = { kind: "draw", ix: cx, iy: cy, tool: drawTool, points: drawTool === "freehand" ? [[cx, cy]] : undefined };
    commitSelection(null);
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!img) return;
    const { sx, sy, ix, iy } = toImage(e);
    if (trackHoverEnabled || hoverTrack) updateHover(ix, iy, sx, sy);
    // cursor readout
    const pd = planesRef.current;
    const px = Math.floor(ix);
    const py = Math.floor(iy);
    if (px >= 0 && py >= 0 && px < img.shape.x && py < img.shape.y) {
      let values: (number | null)[] | null = null;
      if (pd) {
        const [, h, w] = pd.shape;
        const qx = Math.min(Math.floor(ix / pd.stride), w - 1);
        const qy = Math.min(Math.floor(iy / pd.stride), h - 1);
        values = pd.channels.map((c) => {
          const v = c[qy * w + qx];
          return Number.isFinite(v) ? v : null;
        });
      }
      useStore.getState().setCursor({ x: px, y: py, values });
    } else useStore.getState().setCursor(null);

    if (inProgress || anglePts.length) setHoverPt(clampPt(ix, iy));
    if (tool === "brush") setBrushPt([ix, iy]);
    const d = dragRef.current;
    if (!d) return;
    if (d.kind === "pan") {
      setView(img.id, { scale: d.view.scale, tx: d.view.tx + (e.clientX - d.sx), ty: d.view.ty + (e.clientY - d.sy) });
      return;
    }
    const [cx, cy] = clampPt(ix, iy);
    if (d.kind === "plugin") {
      if (!d.started) {
        if (Math.hypot(cx - d.ix, cy - d.iy) * view.scale < 4) return; // a jitter-free press stays a click
        d.started = true;
        runPluginTool(d.opId, "drag-start", d.ix, d.iy, d);
      }
      setToolDrag({ x1: d.ix, y1: d.iy, x2: cx, y2: cy });
      return;
    }
    if (d.kind === "stroke") {
      const last = d.points[d.points.length - 1];
      if (Math.hypot(cx - last[0], cy - last[1]) >= 0.5 / view.scale) {
        d.points.push([cx, cy]);
        setStroke({ points: d.points.slice(), erase: d.erase });
      }
      return;
    }
    if (d.kind === "arrow") {
      let x2 = cx, y2 = cy;
      if (e.shiftKey) {
        const ang = Math.round(Math.atan2(cy - d.iy, cx - d.ix) / (Math.PI / 4)) * (Math.PI / 4);
        const L = Math.hypot(cx - d.ix, cy - d.iy);
        x2 = d.ix + L * Math.cos(ang);
        y2 = d.iy + L * Math.sin(ang);
      }
      setDraftArrow({ x1: d.ix, y1: d.iy, x2, y2 });
      return;
    }
    if (d.kind === "draw") {
      let roi: Roi;
      if (d.tool === "freehand") {
        d.points!.push([cx, cy]);
        roi = { type: "freehand", points: d.points!.slice() };
      } else if (d.tool === "line") {
        let x2 = cx, y2 = cy;
        if (e.shiftKey) {
          const ang = Math.round(Math.atan2(cy - d.iy, cx - d.ix) / (Math.PI / 4)) * (Math.PI / 4);
          const L = Math.hypot(cx - d.ix, cy - d.iy);
          x2 = d.ix + L * Math.cos(ang);
          y2 = d.iy + L * Math.sin(ang);
        }
        roi = { type: "line", x1: d.ix, y1: d.iy, x2, y2, width: toolOptions.lineWidth || 1 };
      } else {
        let w = cx - d.ix;
        let h = cy - d.iy;
        if (e.shiftKey) {
          const s = Math.max(Math.abs(w), Math.abs(h));
          w = Math.sign(w || 1) * s;
          h = Math.sign(h || 1) * s;
        }
        let x = d.ix, y = d.iy;
        if (e.altKey) {
          x = d.ix - w;
          y = d.iy - h;
          w *= 2;
          h *= 2;
        }
        roi = { type: d.tool as "rect" | "oval", x: Math.min(x, x + w), y: Math.min(y, y + h), w: Math.abs(w), h: Math.abs(h) };
        if (d.tool === "rect" || d.tool === "oval") {
          roi.x = Math.round(roi.x!); roi.y = Math.round(roi.y!); roi.w = Math.round(roi.w!); roi.h = Math.round(roi.h!);
        }
      }
      setDraftRoi(roi);
    } else if (d.kind === "move") {
      let dx = ix - d.ix;
      let dy = iy - d.iy;
      if (d.roi.type === "rect" || d.roi.type === "oval" || d.roi.type === "mask") {
        dx = Math.round(dx);
        dy = Math.round(dy);
      }
      setDraftRoi(moveRoi(d.roi, dx, dy));
    } else if (d.kind === "resize") {
      const r = resizeRoi(d.roi, d.handle, cx, cy, e.shiftKey);
      if (r.type === "rect" || r.type === "oval") {
        r.x = Math.round(r.x!); r.y = Math.round(r.y!); r.w = Math.round(r.w!); r.h = Math.round(r.h!);
      }
      setDraftRoi(r);
    } else if (d.kind === "vertex") {
      const pts = d.roi.points!.map((p, i) => (i === d.index ? [cx, cy] : p));
      setDraftRoi({ ...d.roi, points: pts });
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (!d || !img) return;
    if (d.kind === "pan") return;
    if (d.kind === "plugin") {
      setToolDrag(null);
      if (d.started) {
        const { ix, iy } = toImage(e);
        const [cx, cy] = clampPt(ix, iy);
        runPluginTool(d.opId, "drag-end", d.ix, d.iy, d, cx, cy);
      } else runPluginTool(d.opId, "click", d.ix, d.iy, d);
      return;
    }
    if (d.kind === "stroke") {
      setStroke(null);
      commitStroke(d.points, d.erase);
      return;
    }
    if (d.kind === "arrow") {
      const a = draftArrow;
      setDraftArrow(null);
      if (a && Math.hypot(a.x2 - a.x1, a.y2 - a.y1) >= 2 / view.scale) commitArrow(a);
      return;
    }
    const roi = draftRoi;
    setDraftRoi(null);
    if (d.kind === "draw") {
      if (!roi) return commitSelection(null);
      if ((roi.type === "rect" || roi.type === "oval") && (roi.w! < 1 || roi.h! < 1)) return commitSelection(null);
      if (roi.type === "line" && Math.hypot(roi.x2! - roi.x1!, roi.y2! - roi.y1!) < 1) return commitSelection(null);
      if (roi.type === "freehand") {
        if (roi.points!.length < 3) return commitSelection(null);
        const pts = roi.points!.filter((_, i) => i % 2 === 0 || i === roi.points!.length - 1);
        roi.points = pts;
      }
      // rectangle tool with a corner radius: the committed selection is a polygon approximating the rounded rect
      const radius = useStore.getState().toolOptions.rectRadius;
      if (roi.type === "rect" && tool === "rect" && radius > 0) return commitSelection({ type: "polygon", points: roundedRectPolygon(roi.x!, roi.y!, roi.w!, roi.h!, radius) });
      commitSelection(roi);
    } else if (roi) commitSelection(roi);
  };

  /** A cancelled pointer (touch gesture, capture lost) never gets a pointerup: discard the drag, do not commit it. */
  const onPointerCancel = () => {
    if (!dragRef.current) return;
    dragRef.current = null;
    setDraftRoi(null);
    setDraftArrow(null);
    setStroke(null);
    setToolDrag(null);
  };

  const onDoubleClick = () => {
    if (inProgress) finishPolygon();
  };

  const finishPolygon = () => {
    if (!inProgress) return;
    // a double-click delivers two clicks first: drop trailing vertices that repeat the previous one
    const pts = inProgress.points.filter((p, i, arr) => i === 0 || Math.hypot(p[0] - arr[i - 1][0], p[1] - arr[i - 1][1]) > 1e-6);
    setInProgress(null);
    setHoverPt(null);
    if (inProgress.type === "polygon" && pts.length >= 3) commitSelection({ type: "polygon", points: pts });
    else if (inProgress.type === "polyline" && pts.length >= 2) commitSelection({ type: "polyline", points: pts, width: toolOptions.lineWidth || 1 });
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!img) return;
    if (e.key === "Escape") {
      if (inProgress) {
        setInProgress(null);
        setHoverPt(null);
      } else if (anglePts.length) {
        setAnglePts([]);
        setHoverPt(null);
      } else if (useStore.getState().sheet) useStore.getState().closeSheet();
      else if (useStore.getState().selectedTrackIds.length) useStore.getState().setSelectedTrackIds([]);
      else commitSelection(null);
      e.stopPropagation();
      return;
    }
    if (e.key === "Enter" && inProgress) {
      finishPolygon();
      e.stopPropagation();
      return;
    }
    if (e.key === "Backspace" && inProgress) {
      setInProgress(inProgress.points.length > 1 ? { ...inProgress, points: inProgress.points.slice(0, -1) } : null);
      e.stopPropagation();
      return;
    }
    if (e.key === "Backspace" && anglePts.length) {
      setAnglePts(anglePts.slice(0, -1));
      e.stopPropagation();
      return;
    }
    if (["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(e.key) && selection && !e.metaKey && !e.ctrlKey) {
      const step = e.shiftKey ? 10 : 1;
      const dx = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0;
      const dy = e.key === "ArrowUp" ? -step : e.key === "ArrowDown" ? step : 0;
      commitSelection(moveRoi(selection, dx, dy));
      e.preventDefault();
      e.stopPropagation();
    }
  };

  // ---------------------------------------------------------- drag & drop
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDropping(false);
    if (e.dataTransfer.files.length) void useStore.getState().openFiles(e.dataTransfer.files);
  };

  // ---------------------------------------------------------- render overlay
  const shown = draftRoi || selection;
  const managerRois = useMemo(() => (showManagerRois && img && state ? state.rois.filter((r) => !r.image_id || r.image_id === img.id) : []), [showManagerRois, img, state]);
  // per-plane items (position: {t, z, c}) are drawn only on their own plane, as burn_overlay does
  const overlayItems = useMemo(() => {
    if (!showOverlay || !img) return [] as Record<string, any>[];
    const { t, z } = img.display;
    return (img.overlay as unknown as Record<string, any>[]).filter((o) => onPlane(o, t, z));
  }, [showOverlay, img?.overlay, img?.display.t, img?.display.z]);
  const panning = spaceDown || dragRef.current?.kind === "pan";
  const cursorClass = (panning ? " panning" : pid ? " tool-plugin" : ` tool-${tool}`) + (hoverTrack && trackHoverEnabled && !dragRef.current ? " over-track" : "");
  const pluginCursor = pid && !panning ? pluginSpec?.tool_cursor || "crosshair" : undefined;
  const anglePreview = tool === "angle" ? (anglePts.length === 2 && hoverPt ? [anglePts[0], anglePts[1], hoverPt] : selection?.type === "polyline" && selection.points?.length === 3 && !anglePts.length ? selection.points : null) : null;
  const liveStrokes: { points: number[][]; width: number; erase: boolean; color: string; key: string }[] = [
    ...pendingStrokes.map((s, i) => ({ ...s, key: "p" + i })),
    ...(stroke ? [{ points: stroke.points, width: brushSize, erase: stroke.erase, color: strokeColor(stroke.erase), key: "live" }] : []),
  ];

  return (
    <div className="viewport" onDragOver={(e) => { e.preventDefault(); setDropping(true); }} onDragLeave={() => setDropping(false)} onDrop={onDrop}>
      {/* always present while an image is open: the bar's height is reserved, so picking a tool never shifts the canvas (E7-27) */}
      {img && <ToolOptionsBar tool={tool} img={img} selection={selection} />}
      <div ref={wrapRef} className={"canvas-wrap" + cursorClass} style={pluginCursor ? { cursor: pluginCursor } : undefined} tabIndex={0} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerCancel} onPointerLeave={() => { useStore.getState().setCursor(null); setBrushPt(null); setHoverTrack(null); }} onDoubleClick={onDoubleClick} onKeyDown={onKeyDown} onContextMenu={(e) => e.preventDefault()}>
        <canvas ref={canvasRef} />
        {img && (
          <svg className="overlay" textRendering={prefs.antialiased_text ? "geometricPrecision" : "optimizeSpeed"} style={{ fontFamily: "var(--overlay-font)" }}>
            <g transform={`translate(${view.tx} ${view.ty}) scale(${view.scale})`}>
              {overlayItems.map((o, i) => {
                const t = o.type as string;
                if (t === "track") return null;
                let node: React.ReactNode;
                if (t === "scalebar") node = <ScaleBar bar={o as Partial<ScaleBarItem>} img={img} scale={view.scale} />;
                else if (t === "image") node = <ImageOverlay item={o as ImageOverlayItem} images={state?.images || []} />;
                else if (t === "text") node = <TextShape item={o as TextItem} scale={view.scale} />;
                else if (t === "arrow") node = <ArrowShape item={o as ArrowItem} scale={view.scale} />;
                else if (t === "calibration_bar") node = <CalibrationBar item={o as CalibrationBarItem} img={img} luts={luts} index={i} scale={view.scale} />;
                else if (t === "time_stamp") node = <TimeStamp item={o as TimeStampItem} img={img} scale={view.scale} />;
                else {
                  // Overlay Options: an item's own label always shows; "show labels" adds one (name or index) to the others
                  const roi = o as Roi;
                  const explicit = o.label === undefined || o.label === null || o.label === "" ? undefined : String(o.label);
                  const auto = ovOpts.showLabels ? (ovOpts.useNames && o.name ? String(o.name) : String(img.overlay.indexOf(o as unknown as Roi) + 1)) : undefined;
                  const fill = roi.fill || roi.fill_color || ovOpts.fill;
                  node = <RoiShape roi={fill && fill !== roi.fill ? { ...roi, fill } : roi} color={o.color || "#ffd54a"} width={o.stroke_width || ovOpts.strokeWidth || 1.5} scale={view.scale} label={explicit ?? auto}
                    labelFontSize={ovOpts.fontSize} labelColor={ovOpts.labelColor} labelBackground={ovOpts.labelBackground} pointSize={typeof o.point_size === "number" ? o.point_size : undefined} />;
                }
                // one malformed item (e.g. from overlay.add_item) must not unmount the whole app
                return <OverlayItemBoundary key={o.id || t + i}>{node}</OverlayItemBoundary>;
              })}
              {tracks.length > 0 && (
                <TracksLayer tracks={tracks} t={img.display.t} scale={view.scale} showFuture={showTrackFuture} labelMode={showTrackLabels ? trackLabelMode : "none"} selected={selectedTrackIds} hovered={hoverTrack?.id ?? null} colors={trackColors.colors} onlySelected={trackShowOnlySelected} displayMode={trackDisplay?.display_mode} tailLength={trackDisplay?.tail_length} />
              )}
              {spotPreview.spots.length > 0 && <SpotPreviewLayer spots={spotPreview.spots} img={img} scale={view.scale} />}
              {managerRois.map((r) => <RoiShape key={r.id} roi={r} color={selectedRoiIds.includes(r.id!) ? "#34d399" : (r.color || "rgba(80,220,255,0.85)")} width={selectedRoiIds.includes(r.id!) ? 2 : 1} scale={view.scale} label={r.name} />)}
              {liveStrokes.map((s) => (
                <polyline key={s.key} points={s.points.map((p) => p.join(",")).join(" ")} fill="none" stroke={s.color} strokeWidth={s.width} strokeLinecap="round" strokeLinejoin="round" opacity={0.9} />
              ))}
              {tool === "brush" && brushPt && !stroke && (
                <g>
                  <circle cx={brushPt[0]} cy={brushPt[1]} r={brushSize / 2} fill="none" stroke="rgba(0,0,0,0.7)" strokeWidth={2.5 / view.scale} />
                  <circle cx={brushPt[0]} cy={brushPt[1]} r={brushSize / 2} fill="none" stroke="#fff" strokeWidth={1 / view.scale} />
                </g>
              )}
              {draftArrow && <ArrowShape item={{ type: "arrow", ...draftArrow, color: toolOptions.arrowColor, stroke_width: toolOptions.arrowWidth, head_size: toolOptions.arrowHead }} scale={view.scale} dashed />}
              {toolDrag && (
                <g>
                  <line x1={toolDrag.x1} y1={toolDrag.y1} x2={toolDrag.x2} y2={toolDrag.y2} stroke="var(--roi)" strokeWidth={1.5 / view.scale} strokeDasharray={`${4 / view.scale} ${3 / view.scale}`} />
                  <circle cx={toolDrag.x1} cy={toolDrag.y1} r={3 / view.scale} fill="var(--roi)" />
                  <circle cx={toolDrag.x2} cy={toolDrag.y2} r={3 / view.scale} fill="none" stroke="var(--roi)" strokeWidth={1 / view.scale} />
                </g>
              )}
              {shown && <RoiShape roi={shown} color="var(--roi)" width={prefs.roi_stroke_width || 1.5} scale={view.scale} dashed={!!draftRoi} pointSize={prefs.point_size} pointColor={toHex6(prefs.point_color, "#ffd54a")} showCounter={toolOptions.pointCounter} rx={draftRoi?.type === "rect" && tool === "rect" && toolOptions.rectRadius > 0 ? toolOptions.rectRadius : undefined} />}
              {inProgress && (
                <g>
                  <polyline fill="none" stroke="var(--roi)" strokeWidth={1.5 / view.scale} points={[...inProgress.points, ...(hoverPt ? [hoverPt] : [])].map((p) => p.join(",")).join(" ")} />
                  {inProgress.type === "polygon" && inProgress.points.length >= 2 && hoverPt && <line x1={hoverPt[0]} y1={hoverPt[1]} x2={inProgress.points[0][0]} y2={inProgress.points[0][1]} stroke="var(--roi)" strokeWidth={1 / view.scale} strokeDasharray={`${4 / view.scale} ${4 / view.scale}`} />}
                  {inProgress.points.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={(i === 0 ? 5 : 3) / view.scale} fill={i === 0 ? "var(--roi)" : "none"} stroke="var(--roi)" strokeWidth={1 / view.scale} />)}
                </g>
              )}
              {tool === "angle" && anglePts.length > 0 && (
                <g>
                  <polyline fill="none" stroke="var(--roi)" strokeWidth={1.5 / view.scale} points={[...anglePts, ...(hoverPt ? [hoverPt] : [])].map((p) => p.join(",")).join(" ")} />
                  {anglePts.map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r={3.5 / view.scale} fill={i === 1 ? "var(--roi)" : "none"} stroke="var(--roi)" strokeWidth={1 / view.scale} />)}
                </g>
              )}
              {anglePreview && <AngleArc a={anglePreview[0]} v={anglePreview[1]} b={anglePreview[2]} scale={view.scale} />}
            </g>
            {shown && !draftRoi && handlesFor(shown).map((h) => (
              <rect key={h.id} x={h.x * view.scale + view.tx - 4} y={h.y * view.scale + view.ty - 4} width={8} height={8} fill="var(--bg1)" stroke="var(--roi)" strokeWidth={1.2} rx={1.5} />
            ))}
          </svg>
        )}
        {img && textEdit && (
          <input
            className="text-float"
            autoFocus
            value={textEdit.value}
            placeholder="Type, ↵ to add"
            style={{ left: textEdit.ix * view.scale + view.tx, top: textEdit.iy * view.scale + view.ty, fontSize: toolOptions.fontSize * view.scale, color: toolOptions.textColor }}
            onPointerDown={(e) => e.stopPropagation()}
            onChange={(e) => setTextEdit({ ...textEdit, value: e.target.value })}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === "Enter") commitText(textEdit);
              else if (e.key === "Escape") { textEditRef.current = null; setTextEdit(null); }
            }}
            onBlur={() => commitText(textEdit)}
          />
        )}
        {img && hoverTrack && trackHoverEnabled && <div ref={tipRef} className="track-tip" style={{ left: tipPosRef.current.sx, top: tipPosRef.current.sy }}>{hoverTrack.name}</div>}
        {img && tracks.length > 0 && trackColors.legend && <TrackLegendChip legend={trackColors.legend} />}
        {!img && <EmptyState />}
        {dropping && <div className="drop-overlay">Drop images to open</div>}
        {img && (
          <div className="viewport-hud">
            {usePreview && <span className="hud-chip preview"><Sparkles size={12} /> preview</span>}
            {threshold && <span className="hud-chip" style={{ color: "#f87171" }}>threshold {fmtShort(threshold.lower)}–{fmtShort(threshold.upper)}</span>}
            {stride > 1 && <span className="hud-chip" title="Downsampled for display; zoom in for full resolution">1/{stride}</span>}
            {tool === "brush" && <span className="hud-chip" title="[ and ] change the brush size">brush {brushSize} px{toolOptions.erase ? " · erase" : ""}</span>}
            {tool === "track" && <span className="hud-chip" title="Manual tracking: click adds this frame's point, ⌥-click deletes it">{manualTrack.id === null ? "track: click to start" : `track #${manualTrack.id}`} · frame {img.display.t + 1}/{img.shape.t}</span>}
            {pluginSpec && <span className="hud-chip" title={pluginSpec.description}>{pluginSpec.tool_icon || "⚙"} {pluginSpec.name} · click or drag</span>}
            {spotPreview.active && (
              <span className="hud-chip spot-preview-chip" title={`Tracks wizard: the detected spots of this frame (${spotPreview.total} in all frames${spotPreview.truncated ? ", only the first ones served are drawn" : ""}); nothing is added to the ROI Manager`}>
                {spotPreview.spots.length} spot{spotPreview.spots.length === 1 ? "" : "s"} on frame {img.display.t + 1}
              </span>
            )}
            {!showOverlay && img.overlay.length > 0 && <span className="hud-chip" title="View › Show Overlay (⇧O)">overlay hidden</span>}
          </div>
        )}
        {img && (
          <div className="zoom-controls">
            <button className="icon-btn" title="Zoom out (−)" onClick={() => zoomAt(1 / 1.5)}><ZoomOut size={14} /></button>
            <span className="zoom-val" onClick={() => zoomAt(1 / view.scale)} title="Reset to 100%">{Math.round(view.scale * 100)}%</span>
            <button className="icon-btn" title="Zoom in (+)" onClick={() => zoomAt(1.5)}><ZoomIn size={14} /></button>
            <button className="icon-btn" title="Fit (0)" onClick={fit}><Maximize size={14} /></button>
          </div>
        )}
        {sheet && <OperationSheet />}
      </div>
      {img && <DimensionBar img={img} />}
    </div>
  );
}

function fmtShort(v: number): string {
  return Number.isInteger(v) ? String(v) : v.toPrecision(4);
}

const NO_OVERLAY: unknown[] = [];
const NO_TRACKS: TrackItem[] = [];
const NO_SPOTS: ModelSpot[] = [];

interface SpotPreview {
  /** The wizard is on a detection step with a detection run: the preview chip is shown (even with 0 spots on this frame). */
  active: boolean;
  /** The unlinked spots of the current frame. */
  spots: ModelSpot[];
  /** Unlinked spots in all frames, and whether the server served only a part of them. */
  total: number;
  truncated: boolean;
}

const NO_PREVIEW: SpotPreview = { active: false, spots: NO_SPOTS, total: 0, truncated: false };

/**
 * Tracks wizard detection preview (E8-4): while the wizard is open on its detection steps (Detector, Initial
 * threshold, Spot filters) and a detection has run, the track model's unlinked spots are drawn on the frame each spot
 * belongs to – a purely client-side layer, so the shared ROI Manager and the recorder stay untouched.  Gone as soon
 * as the wizard moves on to the linker, is reset, or the panel switches back to the track list.
 */
function useSpotPreview(img: ImageSummary | undefined): SpotPreview {
  const model = useTrackStore((s) => s.model);
  const mode = useTrackStore((s) => s.panelMode);
  const step = useTrackStore((s) => s.wizard.step);
  const detected = useTrackStore((s) => !!s.wizard.results[0]);
  const t = img?.display.t ?? 0;
  return useMemo(() => {
    if (!img || !model || model.image !== img.id || mode !== "wizard" || step > 2 || !detected) return NO_PREVIEW;
    const all = model.free_spots || [];
    const spots = all.filter((s) => Math.round(s.t) === t);
    return { active: true, spots: spots.length ? spots : NO_SPOTS, total: model.n_free_spots ?? all.length, truncated: (model.n_free_spots ?? all.length) > all.length };
  }, [img?.id, t, model, mode, step, detected]); // eslint-disable-line react-hooks/exhaustive-deps
}

/** The detection preview's circles (image px); spots on another z slice of a stack are drawn faint. */
function SpotPreviewLayer({ spots, img, scale }: { spots: ModelSpot[]; img: ImageSummary; scale: number }) {
  const maxR = Math.max(img.shape.x, img.shape.y) / 2;
  const z = img.display.z;
  return (
    <g className="spot-preview" aria-hidden="true">
      {spots.map((s, i) => {
        const r = Math.min(maxR, Math.max(1.5, Number.isFinite(s.r) && s.r > 0 ? s.r : 3));
        const faint = img.shape.z > 1 && Number.isFinite(s.z) && Math.abs(s.z - z) > 0.5;
        return (
          <g key={s.id ?? i} opacity={faint ? 0.35 : 1}>
            <circle cx={s.x} cy={s.y} r={r} fill="none" stroke="rgba(0,0,0,0.7)" strokeWidth={3 / scale} />
            <circle cx={s.x} cy={s.y} r={r} fill="none" stroke="#e879f9" strokeWidth={1.5 / scale} />
          </g>
        );
      })}
    </g>
  );
}

/** Legend of the track colour-by mode (bottom-left of the canvas): title, colormap bar, range. */
function TrackLegendChip({ legend }: { legend: TrackLegend }) {
  return (
    <div className="track-legend" title="Track colour-by (Inspector › Tracks)">
      <span>{legend.title}</span>
      <div className="bar" style={{ background: legend.gradient }} />
      {legend.categorical ? <span className="ends"><span>one hue per track</span></span> : <span className="ends"><span>{fmtShort(legend.lo)}</span><span>{fmtShort(legend.hi)}</span></span>}
    </div>
  );
}

/** Map a generic item onto the parameter names an op declares (tolerates a few aliases). */
function pickParams(spec: OpSpec, item: Record<string, unknown>): Record<string, unknown> {
  const alias: Record<string, string> = { size: "font_size", fontsize: "font_size", font: "font_size", colour: "color", width: "stroke_width", line_width: "stroke_width", head: "head_size", arrow_head: "head_size", label: "text", string: "text" };
  const out: Record<string, unknown> = {};
  for (const p of spec.params) {
    const key = p.name in item ? p.name : alias[p.name] && alias[p.name] in item ? alias[p.name] : null;
    out[p.name] = key ? item[key] : p.type === "image" ? null : p.default;
  }
  return out;
}

/** Angle (degrees, 0–180) at vertex v between arms v→a and v→b. */
export function angleAt(a: number[], v: number[], b: number[]): number {
  const a1 = Math.atan2(a[1] - v[1], a[0] - v[0]);
  const a2 = Math.atan2(b[1] - v[1], b[0] - v[0]);
  let d = Math.abs(a1 - a2) * (180 / Math.PI);
  if (d > 180) d = 360 - d;
  return d;
}

function AngleArc({ a, v, b, scale }: { a: number[]; v: number[]; b: number[]; scale: number }) {
  const la = Math.hypot(a[0] - v[0], a[1] - v[1]);
  const lb = Math.hypot(b[0] - v[0], b[1] - v[1]);
  if (la < 1e-6 || lb < 1e-6) return null;
  const r = Math.min(22 / scale, la * 0.45, lb * 0.45);
  const a1 = Math.atan2(a[1] - v[1], a[0] - v[0]);
  const a2 = Math.atan2(b[1] - v[1], b[0] - v[0]);
  let delta = a2 - a1;
  while (delta > Math.PI) delta -= 2 * Math.PI;
  while (delta < -Math.PI) delta += 2 * Math.PI;
  const sweep = delta > 0 ? 1 : 0;
  const p1 = [v[0] + r * Math.cos(a1), v[1] + r * Math.sin(a1)];
  const p2 = [v[0] + r * Math.cos(a2), v[1] + r * Math.sin(a2)];
  const mid = a1 + delta / 2;
  const lx = v[0] + r * 1.9 * Math.cos(mid);
  const ly = v[1] + r * 1.9 * Math.sin(mid);
  const deg = angleAt(a, v, b);
  return (
    <g>
      <path d={`M ${p1[0]} ${p1[1]} A ${r} ${r} 0 0 ${sweep} ${p2[0]} ${p2[1]}`} fill="none" stroke="var(--roi)" strokeWidth={1.2 / scale} />
      <text x={lx} y={ly} fontSize={11 / scale} fill="var(--roi)" textAnchor="middle" dominantBaseline="middle" style={{ paintOrder: "stroke", stroke: "rgba(0,0,0,0.7)", strokeWidth: 2.5 / scale }}>{deg.toFixed(1)}°</text>
    </g>
  );
}

/** Slim strip above the canvas with the options of the active tool. */
function ToolOptionsBar({ tool, img, selection }: { tool: Tool; img: ImageSummary; selection: Roi | null }) {
  const brushSize = useStore((s) => s.brushSize);
  const setBrushSize = useStore((s) => s.setBrushSize);
  const o = useStore((s) => s.toolOptions);
  const setOpts = useStore((s) => s.setToolOptions);
  const wandTolerance = useStore((s) => s.wandTolerance);
  const setWandTolerance = useStore((s) => s.setWandTolerance);
  const setSelection = useStore((s) => s.setSelection);
  const prefs = useStore((s) => s.prefs);
  const setPrefs = useStore((s) => s.setPrefs);
  const manualTrack = useStore((s) => s.manualTrack);
  const setManualTrack = useStore((s) => s.setManualTrack);
  // plugin tools: the op's own Params are the tool options (store.toolParams)
  const pid = pluginToolId(tool);
  const pluginSpec = useStore((s) => (pid ? s.ops[pid] : undefined));
  const pluginValues = useStore((s) => (pid ? s.toolParams[pid] : undefined));
  const setToolParams = useStore((s) => s.setToolParams);
  const num = (v: string, fallback: number) => (Number.isFinite(parseFloat(v)) ? parseFloat(v) : fallback);
  const names: Partial<Record<Tool, string>> = { brush: "Brush", bucket: "Bucket", wand: "Wand", text: "Text", arrow: "Arrow", line: "Line", polyline: "Polyline", angle: "Angle", rect: "Rectangle", point: "Points", track: "Track", pointer: "Select", hand: "Pan", oval: "Oval", polygon: "Polygon", freehand: "Freehand", zoom: "Zoom", picker: "Picker" };
  const plainHints: Partial<Record<Tool, string>> = { pointer: "click a selection to move it · drag its handles · arrows nudge", hand: "drag to pan · wheel zooms", oval: "drag to draw · ⇧ circle · ⌥ from the centre", polygon: "click the vertices · double-click or ↵ to close · ⌫ removes the last", freehand: "drag to trace", zoom: "click to zoom in · ⌥-click out · 0 fits", picker: "click to read the value into the foreground colour" };
  const multiChannel = img.shape.c > 1 && !img.rgb;
  return (
    <div className="tool-options">
      <span className="tool-name">{pluginSpec ? pluginSpec.name : names[tool]}</span>
      {!pluginSpec && plainHints[tool] && <span className="hint">{plainHints[tool]}</span>}
      {pluginSpec && pid && (
        <>
          {pluginSpec.params.filter((p) => p.type !== "image").map((p) => (
            <PluginToolParam key={p.name} spec={p} value={pluginValues?.[p.name] !== undefined ? pluginValues[p.name] : p.default} onChange={(v) => setToolParams(pid, { [p.name]: v })} />
          ))}
          <span className="hint" title={pluginSpec.description}>{pluginSpec.description.length > 110 ? pluginSpec.description.slice(0, 107) + "…" : pluginSpec.description}</span>
        </>
      )}
      {tool === "brush" && (
        <>
          <label>Size <input type="number" min={1} max={500} step={1} value={brushSize} onChange={(e) => setBrushSize(num(e.target.value, brushSize))} /> px</label>
          <label><input type="checkbox" checked={o.erase} onChange={(e) => setOpts({ erase: e.target.checked })} /> Erase (background value)</label>
          {multiChannel && <label><input type="checkbox" checked={o.allChannels} onChange={(e) => setOpts({ allChannels: e.target.checked })} /> All channels</label>}
          <span className="hint">[ / ] resize · ⌥-drag erases · paints the foreground value (Edit › Colour Settings)</span>
        </>
      )}
      {tool === "bucket" && (
        <>
          <label>Tolerance <input type="number" min={0} step={1} value={o.bucketTolerance} onChange={(e) => setOpts({ bucketTolerance: Math.max(0, num(e.target.value, o.bucketTolerance)) })} /></label>
          <label>Connectivity <select value={o.bucketConnectivity} onChange={(e) => setOpts({ bucketConnectivity: e.target.value as "4" | "8" })}><option value="4">4-connected</option><option value="8">8-connected</option></select></label>
          <span className="hint">click to fill the connected region with the foreground value</span>
        </>
      )}
      {tool === "wand" && (
        <>
          <label>Tolerance <input type="number" min={0} step={1} value={wandTolerance} onChange={(e) => setWandTolerance(Math.max(0, num(e.target.value, wandTolerance)))} /></label>
          <span className="hint">click to trace a region · ⇧-click adds to the selection</span>
        </>
      )}
      {tool === "text" && (
        <>
          <label>Font <input type="number" min={6} max={200} step={1} value={o.fontSize} onChange={(e) => setOpts({ fontSize: Math.max(6, num(e.target.value, o.fontSize)) })} /> px (image)</label>
          <label>Colour <input type="color" value={o.textColor} onChange={(e) => setOpts({ textColor: e.target.value })} /></label>
          <label><input type="checkbox" checked={o.textBackground} onChange={(e) => setOpts({ textBackground: e.target.checked })} /> Background box</label>
          <span className="hint">click on the image, type, ↵ to add to the overlay</span>
        </>
      )}
      {tool === "arrow" && (
        <>
          <label>Width <input type="number" min={0.5} max={50} step={0.5} value={o.arrowWidth} onChange={(e) => setOpts({ arrowWidth: Math.max(0.5, num(e.target.value, o.arrowWidth)) })} /> px</label>
          <label>Head <input type="number" min={2} max={200} step={1} value={o.arrowHead} onChange={(e) => setOpts({ arrowHead: Math.max(2, num(e.target.value, o.arrowHead)) })} /> px (image)</label>
          <label>Colour <input type="color" value={o.arrowColor} onChange={(e) => setOpts({ arrowColor: e.target.value })} /></label>
          <span className="hint">drag from tail to tip · ⇧ constrains to 45°</span>
        </>
      )}
      {(tool === "line" || tool === "polyline") && (
        <>
          <label>Line width <input type="number" min={1} max={200} step={1} value={o.lineWidth} onChange={(e) => {
            const w = Math.max(1, Math.round(num(e.target.value, o.lineWidth)));
            setOpts({ lineWidth: w });
            if (selection && (selection.type === "line" || selection.type === "polyline")) setSelection(img.id, { ...selection, width: w });
          }} /> px</label>
          <span className="hint">{tool === "line" ? "drag to draw · ⇧ constrains to 45°" : "click points · double-click or ↵ to finish"}</span>
        </>
      )}
      {tool === "angle" && <span className="hint">click the end of the first arm, then the vertex, then the end of the second arm</span>}
      {tool === "rect" && (
        <>
          <label>Corner radius <input type="number" min={0} max={5000} step={1} value={o.rectRadius} onChange={(e) => setOpts({ rectRadius: Math.max(0, Math.round(num(e.target.value, o.rectRadius))) })} /> px</label>
          <span className="hint">{o.rectRadius > 0 ? "rounded rectangles are committed as polygon selections" : "drag to draw · ⇧ square · ⌥ from the centre"}</span>
        </>
      )}
      {tool === "point" && (
        <>
          <label>Size <input type="number" min={1} max={100} step={1} value={prefs.point_size} onChange={(e) => void setPrefs({ point_size: Math.max(1, Math.min(100, Math.round(num(e.target.value, prefs.point_size)))) })} /> px</label>
          <label>Colour <input type="color" value={toHex6(prefs.point_color)} onChange={(e) => void setPrefs({ point_color: e.target.value })} /></label>
          <label><input type="checkbox" checked={o.pointCounter} onChange={(e) => setOpts({ pointCounter: e.target.checked })} /> Counter</label>
          <span className="hint">click to place · ⇧-click adds · size and colour are saved in Preferences</span>
        </>
      )}
      {tool === "track" && (
        <>
          <span className="mono">{manualTrack.id === null ? "no active track" : `track #${manualTrack.id}`}</span>
          <label><input type="checkbox" checked={manualTrack.autoAdvance} onChange={(e) => setManualTrack({ autoAdvance: e.target.checked })} /> Auto-advance</label>
          <button className="btn sm" onClick={() => void startManualTrack()}>New track</button>
          <button className="btn sm" disabled={manualTrack.id === null} onClick={() => setManualTrack({ id: null })}>Finish</button>
          <span className="hint">click adds this frame's point{manualTrack.autoAdvance ? " and moves to the next frame" : ""} · ⌥-click deletes it</span>
        </>
      )}
    </div>
  );
}

function EmptyState() {
  const runOp = useStore((s) => s.runOp);
  return (
    <div className="empty-viewport">
      <ImageIcon size={44} strokeWidth={1.2} />
      <h2>Drop an image here</h2>
      <div className="muted">TIFF, OME-TIFF, PNG, JPEG, ND2, CZI, LIF · hyperstacks with channels, slices and frames</div>
      <div className="hint" style={{ marginTop: 8 }}>
        <button className="btn primary" onClick={() => doAction("open")}><FolderOpen size={14} /> Open file…</button>
        <button className="btn" onClick={() => void runOp("file.sample", { name: "Synthetic nuclei (3 channels)" })}>Sample: 3-channel cells</button>
        <button className="btn" onClick={() => void runOp("file.sample", { name: "Cells (3-D, 2 channels)" })}>Sample: 3-D stack</button>
        <button className="btn" onClick={() => void runOp("file.sample", { name: "Blobs" })}>Sample: blobs</button>
      </div>
      <div className="dim" style={{ marginTop: 10 }}>Press <span className="kbd">{kbd("⌘K")}</span> to search all {commandCount(useStore.getState().meta?.ops) || ""} commands</div>
    </div>
  );
}

/** One option of a plugin tool in the tool-options bar (a Param of the op rendered as a small control). */
function PluginToolParam({ spec, value, onChange }: { spec: ParamSpec; value: any; onChange: (v: any) => void }) {
  const t = spec.type;
  if (t === "bool") return <label title={spec.help}><input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} /> {spec.label}</label>;
  if (t === "choice" || t === "multichoice") {
    return (
      <label title={spec.help}>{spec.label} <select value={String(value ?? "")} onChange={(e) => onChange(e.target.value)}>{(spec.choices || []).map((c) => <option key={c} value={c}>{c}</option>)}</select></label>
    );
  }
  if (t === "int" || t === "float") {
    return (
      <label title={spec.help}>{spec.label} <input type="number" min={spec.min} max={spec.max} step={spec.step ?? (t === "int" ? 1 : "any")} value={value ?? ""} onChange={(e) => { const n = t === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value); if (Number.isFinite(n)) onChange(n); }} /> {spec.unit || ""}</label>
    );
  }
  if (t === "color") return <label title={spec.help}>{spec.label} <input type="color" value={toHex6(String(value ?? "#ffd54a"))} onChange={(e) => onChange(e.target.value)} /></label>;
  return <label title={spec.help}>{spec.label} <input type="text" value={String(value ?? "")} onChange={(e) => onChange(e.target.value)} spellCheck={false} style={{ width: 120 }} /></label>;
}

/** The overlay_* settings (Image ▸ Overlay ▸ Overlay Options…) as drawing options; a missing key keeps the previous behaviour. */
export interface OverlayDrawOptions {
  /** Default fill (#rrggbb[aa]) of closed items without their own; undefined / "none" = outline only. */
  fill?: string;
  /** true: items without a label get one (name or index); false / undefined: only explicit labels. */
  showLabels?: boolean;
  useNames: boolean;
  /** Label font size in screen px (undefined = 10). */
  fontSize?: number;
  /** Label colour (undefined = the item's colour). */
  labelColor?: string;
  labelBackground: boolean;
  /** Stroke width of items without their own (undefined = 1.5). */
  strokeWidth?: number;
}

export function overlayOptions(s: Record<string, unknown>): OverlayDrawOptions {
  const num = (k: string) => {
    const v = s[k];
    const n = typeof v === "number" ? v : typeof v === "string" ? parseFloat(v) : NaN;
    return Number.isFinite(n) && n > 0 ? n : undefined;
  };
  const bool = (k: string) => (typeof s[k] === "boolean" ? (s[k] as boolean) : undefined);
  const color = (k: string) => {
    const v = typeof s[k] === "string" ? (s[k] as string).trim() : "";
    return v && v.toLowerCase() !== "none" && /^#[0-9a-fA-F]{3,8}$/.test(v) ? v : undefined;
  };
  return { fill: color("overlay_default_fill"), showLabels: bool("overlay_show_labels"), useNames: bool("overlay_use_names") ?? false, fontSize: num("overlay_label_font_size"), labelColor: color("overlay_label_color"), labelBackground: bool("overlay_label_background") ?? false, strokeWidth: num("overlay_stroke_width") };
}

export function RoiShape({ roi, color, width, scale, dashed, label, pointSize, pointColor, showCounter, rx, labelFontSize, labelColor, labelBackground }: { roi: Roi; color: string; width: number; scale: number; dashed?: boolean; label?: string; pointSize?: number; pointColor?: string; showCounter?: boolean; rx?: number; labelFontSize?: number; labelColor?: string; labelBackground?: boolean }) {
  const sw = width / scale;
  const common = { fill: "none", stroke: color, strokeWidth: sw, strokeDasharray: dashed ? `${4 / scale} ${3 / scale}` : undefined } as const;
  // closed shapes (rect / rounded rect / oval / polygon / freehand) and mask runs take the item's fill
  // colour (#rrggbb or #rrggbbaa; `fill_color` is the legacy key); line / polyline / point stay unfilled
  const fillColor = roi.fill || roi.fill_color || undefined;
  const closed = { ...common, fill: fillColor || "none" } as const;
  let shape: React.ReactNode = null;
  let lx = 0, ly = 0;
  if (roi.type === "rect" || roi.type === "mask") {
    shape = <rect x={roi.x} y={roi.y} width={roi.w} height={roi.h} rx={rx && rx > 0 ? Math.min(rx, (roi.w || 0) / 2, (roi.h || 0) / 2) : undefined} {...closed} />;
    lx = roi.x!; ly = roi.y!;
    if (roi.type === "mask") shape = <MaskShape roi={roi} color={color} sw={sw} fill={fillColor} />;
  } else if (roi.type === "oval") {
    shape = <ellipse cx={roi.x! + roi.w! / 2} cy={roi.y! + roi.h! / 2} rx={roi.w! / 2} ry={roi.h! / 2} {...closed} />;
    lx = roi.x!; ly = roi.y!;
  } else if (roi.type === "line") {
    shape = <line x1={roi.x1} y1={roi.y1} x2={roi.x2} y2={roi.y2} {...common} strokeWidth={Math.max(sw, (roi.width || 1) > 1 ? roi.width! : sw)} opacity={(roi.width || 1) > 1 ? 0.6 : 1} />;
    lx = roi.x1!; ly = roi.y1!;
  } else if (roi.type === "polygon" || roi.type === "freehand") {
    shape = <polygon points={(roi.points || []).map((p) => p.join(",")).join(" ")} {...closed} />;
    lx = roi.points?.[0]?.[0] ?? 0; ly = roi.points?.[0]?.[1] ?? 0;
  } else if (roi.type === "polyline") {
    shape = <polyline points={(roi.points || []).map((p) => p.join(",")).join(" ")} {...common} strokeWidth={Math.max(sw, (roi.width || 1) > 1 ? roi.width! : sw)} />;
    lx = roi.points?.[0]?.[0] ?? 0; ly = roi.points?.[0]?.[1] ?? 0;
  } else if (roi.type === "point") {
    // point size is a screen size (Preferences ▸ Points), like ImageJ's point tool
    const r = (pointSize && pointSize > 0 ? pointSize : 4) / scale;
    const pc = pointColor || color;
    const counter = showCounter === undefined ? (roi.points || []).length > 1 : showCounter;
    shape = (
      <g>
        {(roi.points || []).map((p, i) => (
          <g key={i}>
            <circle cx={p[0]} cy={p[1]} r={r} fill="none" stroke={pc} strokeWidth={sw} />
            <line x1={p[0] - r * 1.6} x2={p[0] + r * 1.6} y1={p[1]} y2={p[1]} stroke={pc} strokeWidth={sw * 0.7} />
            <line y1={p[1] - r * 1.6} y2={p[1] + r * 1.6} x1={p[0]} x2={p[0]} stroke={pc} strokeWidth={sw * 0.7} />
            {counter && <text x={p[0] + r * 1.5} y={p[1] - r} fontSize={Math.max(10, (pointSize && pointSize > 0 ? pointSize : 4) * 1.8) / scale} fill={pc} style={{ paintOrder: "stroke", stroke: "rgba(0,0,0,0.6)", strokeWidth: 2 / scale }}>{i + 1}</text>}
          </g>
        ))}
      </g>
    );
  }
  // labels: font size in screen px (Overlay Options), the item's colour unless a label colour is set, optional box
  const fs = (labelFontSize && labelFontSize > 0 ? labelFontSize : 10) / scale;
  const lcol = labelColor || color;
  const ty = ly - 3 / scale;
  const tw = label ? measureText(label, fs) : 0;
  return (
    <g>
      {shape}
      {label && labelBackground && <rect x={lx - 2 / scale} y={ty - fs} width={tw + 4 / scale} height={fs + 3 / scale} fill="rgba(0,0,0,0.65)" rx={2 / scale} />}
      {label && <text x={lx} y={ty} fontSize={fs} fill={lcol} style={labelBackground ? undefined : { paintOrder: "stroke", stroke: "rgba(0,0,0,0.6)", strokeWidth: 2 / scale }}>{label}</text>}
    </g>
  );
}

function MaskShape({ roi, color, sw, fill }: { roi: Roi; color: string; sw: number; fill?: string }) {
  // draw RLE mask as rects (one per run) – fine for moderately sized masks
  const rects = useMemo(() => {
    const out: { x: number; y: number; w: number }[] = [];
    const rle = roi.rle || [];
    const W = roi.w || 1;
    let pos = 0;
    let val = false;
    for (const run of rle) {
      if (val) {
        let p = pos;
        let rem = run;
        while (rem > 0) {
          const y = Math.floor(p / W);
          const x = p % W;
          const len = Math.min(rem, W - x);
          out.push({ x: x + (roi.x || 0), y: y + (roi.y || 0), w: len });
          p += len;
          rem -= len;
        }
      }
      pos += run;
      val = !val;
      if (out.length > 20000) break;
    }
    return out;
  }, [roi]);
  return (
    <g>
      <rect x={roi.x} y={roi.y} width={roi.w} height={roi.h} fill="none" stroke={color} strokeWidth={sw} strokeDasharray={`${sw * 3} ${sw * 3}`} opacity={0.5} />
      {rects.map((r, i) => <rect key={i} x={r.x} y={r.y} width={r.w} height={1} fill={fill || color} opacity={fill ? undefined : 0.28} />)}
    </g>
  );
}

function distToSeg(px: number, py: number, x1: number, y1: number, x2: number, y2: number): number {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const L2 = dx * dx + dy * dy || 1e-9;
  let t = ((px - x1) * dx + (py - y1) * dy) / L2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

function pointInPolygon(x: number, y: number, pts: number[][]): boolean {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const xi = pts[i][0], yi = pts[i][1], xj = pts[j][0], yj = pts[j][1];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}


interface ScaleBarItem { type: "scalebar"; id?: string; px: number; length: number; unit: string; thickness: number; position: string; color: string; show_label: boolean; font_size: number }

/** Error boundary for a single overlay item: a malformed item renders nothing instead of unmounting the root. */
class OverlayItemBoundary extends React.Component<{ children: React.ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  componentDidCatch(err: unknown) {
    console.warn("overlay item failed to render", err);
  }
  render() {
    return this.state.failed ? null : this.props.children;
  }
}

/**
 * Scale bar in image pixels with the layout of overlay_render._draw_scalebar (label centred under
 * the bar).  Items added through overlay.add_item may miss fields, so every one is defaulted and
 * coerced like the backend does.
 */
function ScaleBar({ bar, img, scale }: { bar: Partial<ScaleBarItem>; img: ImageSummary; scale: number }) {
  const W = img.shape.x;
  const H = img.shape.y;
  const pw = img.calibration.pixel_width || 1;
  const px = Number(bar.px) || (Number(bar.length) > 0 ? Number(bar.length) / pw : 50);
  const length = Number(bar.length) || px * pw;
  const unit = bar.unit ?? img.calibration.unit ?? "";
  const th = Number(bar.thickness) || 6;
  const fs = Number(bar.font_size) || 14; // image px
  const color = bar.color || "#ffffff";
  const showLabel = bar.show_label ?? true;
  const label = showLabel ? `${Number.isInteger(length) ? length : +length.toPrecision(6)} ${unit}`.trim() : ""; // backend: f"{length:g} {unit}"
  const tw = label ? measureText(label, fs) : 0;
  const w = Math.max(px, tw);
  const h = th + (label ? fs + 2 : 0);
  const { x: x0, y: y0 } = anchor(String(bar.position ?? "Lower Right"), W, H, w, h);
  const bx = x0 + (w - px) / 2;
  return (
    <g>
      <rect x={bx} y={y0} width={px} height={th} fill={color} />
      {label && (
        <text x={x0 + w / 2} y={y0 + th + 2 + fs * 0.8} fontSize={fs} fill={color} textAnchor="middle" className="scalebar-label" style={{ strokeWidth: 2 / scale }}>
          {label}
        </text>
      )}
    </g>
  );
}
