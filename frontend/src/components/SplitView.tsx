import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { X, ArrowLeftRight } from "lucide-react";
import { useStore, getPlanes, useOrderedImages } from "../store";
import { composite } from "../render/compositor";
import { chooseStride } from "../render/stride";
import type { ImageSummary, PlaneData, Roi, ViewTransform } from "../types";
import Viewport, { RoiShape } from "./Viewport";
import { onPlane } from "./OverlayItems";
import DimensionBar from "./DimensionBar";

/**
 * Side-by-side view: the active image in the full-featured <Viewport /> (tools, selections,
 * operation sheet) and a second image in a lightweight compare pane.
 *
 * Limitation: <Viewport /> always renders the store's active image and has no imageId prop or
 * context, so it cannot be mounted twice for different images without editing it.  The second
 * pane is therefore a read-only compositing pane (LUTs, display ranges, threshold, selection
 * and ROI overlays; zoom/pan, slice/frame wheel and dimension sliders) – click it to make that
 * image active, which moves the full viewport into that pane.
 */
const DEFAULT_VIEW: ViewTransform = { scale: 1, tx: 0, ty: 0 };
const ROI_TYPES = new Set(["rect", "oval", "polygon", "freehand", "line", "polyline", "point", "mask"]);

export default function SplitView() {
  const images = useOrderedImages();
  const active = useStore((s) => s.state?.active ?? null);
  const split = useStore((s) => s.split);
  const setSplit = useStore((s) => s.setSplit);
  const selectImage = useStore((s) => s.selectImage);
  const prevActive = useRef<string | null>(active);
  const [ratio, setRatio] = useState(() => {
    const v = Number(localStorage.getItem("lumen.splitRatio"));
    return v > 0.15 && v < 0.85 ? v : 0.5;
  });
  const [dragging, setDragging] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  // keep `other` a second open image that differs from the active one; when the user activates
  // the compare image through a tab, the images swap panes (the previous active image becomes `other`)
  useEffect(() => {
    const ids = images.map((im) => im.id);
    let other = split.other;
    if (other === active || (other && !ids.includes(other))) {
      const prev = prevActive.current;
      other = other === active && prev && prev !== active && ids.includes(prev) ? prev : ids.find((id) => id !== active) ?? null;
    } else if (!other) other = ids.find((id) => id !== active) ?? null;
    if (other !== split.other) setSplit({ other });
  }, [images, active, split.other, setSplit]);
  useEffect(() => {
    prevActive.current = active;
  }, [active]);
  // nothing left to compare: leave the split view
  useEffect(() => {
    if (images.length === 0) setSplit({ on: false });
  }, [images.length, setSplit]);

  const otherImg = images.find((im) => im.id === split.other);
  const activeImg = images.find((im) => im.id === active);

  /** Make the compare image active while keeping each image in its pane. */
  const activateOther = useCallback(() => {
    const st = useStore.getState();
    const o = st.split.other;
    const act = st.state?.active ?? null;
    if (!o || o === act) return;
    st.setSplit({ other: act, activeSide: st.split.activeSide === 0 ? 1 : 0 });
    st.selectImage(o);
  }, []);

  // divider drag
  const onDividerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  };
  const onDividerMove = (e: React.PointerEvent) => {
    if (!dragging || !rootRef.current) return;
    const r = rootRef.current.getBoundingClientRect();
    setRatio(Math.min(0.8, Math.max(0.2, (e.clientX - r.left) / Math.max(1, r.width))));
  };
  const onDividerUp = () => {
    if (!dragging) return;
    setDragging(false);
    localStorage.setItem("lumen.splitRatio", String(ratio));
  };

  const activePane = (
    <div className="split-pane active-pane" style={{ flexGrow: split.activeSide === 0 ? ratio : 1 - ratio }}>
      <div className="split-head">
        <span className="tag">active</span>
        <select value={active ?? ""} onChange={(e) => selectImage(e.target.value)} title="Image shown in this pane" aria-label="Active image (this pane)">
          {images.map((im) => <option key={im.id} value={im.id}>{im.name}</option>)}
        </select>
        <span className="grow" />
        <span className="dim small">{activeImg ? `${activeImg.shape.x}×${activeImg.shape.y} · ${activeImg.bit_depth}` : ""}</span>
      </div>
      <Viewport />
    </div>
  );
  const comparePane = (
    <div className="split-pane" style={{ flexGrow: split.activeSide === 0 ? 1 - ratio : ratio }}>
      <div className="split-head">
        <span className="tag">compare</span>
        <select value={split.other ?? ""} onChange={(e) => setSplit({ other: e.target.value || null })} title="Image to compare with" aria-label="Compare image (this pane)">
          {images.filter((im) => im.id !== active).map((im) => <option key={im.id} value={im.id}>{im.name}</option>)}
          {images.filter((im) => im.id !== active).length === 0 && <option value="">(open a second image)</option>}
        </select>
        <span className="grow" />
        <button className="icon-btn" style={{ width: 22, height: 22 }} title="Swap panes" aria-label="Swap panes" onClick={() => setSplit({ activeSide: split.activeSide === 0 ? 1 : 0 })} disabled={!otherImg}><ArrowLeftRight size={13} /></button>
        <button className="icon-btn" style={{ width: 22, height: 22 }} title="Close split view" aria-label="Close split view" onClick={() => setSplit({ on: false })}><X size={13} /></button>
      </div>
      {otherImg ? (
        <>
          <LitePane key={otherImg.id} img={otherImg} onActivate={activateOther} />
          <DimensionBar img={otherImg} />
        </>
      ) : (
        <div className="split-empty">
          <div>Open a second image to compare.</div>
          <button className="btn sm" onClick={() => setSplit({ on: false })}>Leave split view</button>
        </div>
      )}
    </div>
  );
  const divider = <div className={"split-divider" + (dragging ? " dragging" : "")} onPointerDown={onDividerDown} onPointerMove={onDividerMove} onPointerUp={onDividerUp} onPointerCancel={onDividerUp} onDoubleClick={() => { setRatio(0.5); localStorage.removeItem("lumen.splitRatio"); }} title="Drag to resize · double-click to reset" />;

  return (
    <div className="split-view" ref={rootRef}>
      {split.activeSide === 0 ? activePane : comparePane}
      {divider}
      {split.activeSide === 0 ? comparePane : activePane}
    </div>
  );
}

/** Read-only compositing pane for the compare image (zoom / pan / slice wheel; click to activate). */
function LitePane({ img, onActivate }: { img: ImageSummary; onActivate: () => void }) {
  const luts = useStore((s) => s.luts);
  const view = useStore((s) => s.views[img.id]) || DEFAULT_VIEW;
  const threshold = useStore((s) => s.state?.thresholds[img.id]);
  const selection = useStore((s) => s.selections[img.id] ?? null);
  const showOverlay = useStore((s) => s.showOverlay);
  const showManagerRois = useStore((s) => s.showManagerRois);
  const selectedRoiIds = useStore((s) => s.selectedRoiIds);
  const rois = useStore((s) => s.state?.rois);
  const prefs = useStore((s) => s.prefs);
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const rgbaRef = useRef<Uint8ClampedArray<ArrayBuffer> | null>(null);
  const planesRef = useRef<PlaneData | null>(null);
  const dragRef = useRef<{ sx: number; sy: number; view: ViewTransform; moved: boolean; button: number } | null>(null);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const [planesVersion, setPlanesVersion] = useState(0);
  const [panning, setPanning] = useState(false);
  const stride = chooseStride(img, view.scale);

  /** Set this image's view; with "Synchronize Views" every open image of the same size follows. */
  const setView = useCallback((v: ViewTransform) => {
    const st = useStore.getState();
    if (st.syncViews && st.state) {
      const patch: Record<string, ViewTransform> = { [img.id]: v };
      for (const im of st.state.images) if (im.id !== img.id && im.shape.x === img.shape.x && im.shape.y === img.shape.y) patch[im.id] = v;
      st.setViews(patch);
    } else st.setView(img.id, v);
  }, [img.id, img.shape.x, img.shape.y]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: el.clientWidth, h: el.clientHeight }));
    ro.observe(el);
    setSize({ w: el.clientWidth, h: el.clientHeight });
    return () => ro.disconnect();
  }, []);

  // the compare pane fits its image once it knows its size (E7-22): the stored view belongs to the full-width
  // viewport, so a 149 % zoom would leave the image offset and cropped in a half-width pane
  const fitted = useRef(false);
  useEffect(() => {
    if (!size.w || fitted.current) return;
    fitted.current = true;
    const s = Math.min(size.w / img.shape.x, size.h / img.shape.y) * 0.96;
    setView({ scale: s, tx: (size.w - img.shape.x * s) / 2, ty: (size.h - img.shape.y * s) / 2 });
  }, [img.id, img.shape.x, img.shape.y, size.w, size.h, setView]);

  useEffect(() => {
    let alive = true;
    getPlanes(img, img.display.t, img.display.z, stride).then((pd) => {
      if (!alive) return;
      planesRef.current = pd;
      setPlanesVersion((v) => v + 1);
    }).catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [img.id, img.rev, img.display.t, img.display.z, stride]); // eslint-disable-line react-hooks/exhaustive-deps

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
    if (!off) return;
    const v = useStore.getState().views[img.id] || view;
    const effScale = v.scale * (off.width ? img.shape.x / off.width : 1);
    ctx.imageSmoothingEnabled = effScale < 1 || prefs.interpolate_zoom;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(off, v.tx, v.ty, img.shape.x * v.scale, img.shape.y * v.scale);
  }, [img, size, view, prefs.interpolate_zoom]);

  useEffect(() => {
    const pd = planesRef.current;
    if (!pd) return;
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
    off.getContext("2d")!.putImageData(new ImageData(rgba, w, h), 0, 0);
    draw();
  }, [planesVersion, img.display, img.rgb, luts, threshold]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const id = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(id);
  }, [draw]);

  const zoomAt = useCallback((factor: number, sx: number, sy: number) => {
    const v = useStore.getState().views[img.id] || view;
    const ns = Math.min(64, Math.max(0.02, v.scale * factor));
    const k = ns / v.scale;
    setView({ scale: ns, tx: sx - (sx - v.tx) * k, ty: sy - (sy - v.ty) * k });
  }, [img.id, view, setView]);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const st = useStore.getState();
      const cur = st.image(img.id);
      if (!cur) return;
      if (e.shiftKey && cur.shape.z > 1) {
        st.setPosition(cur.id, { z: Math.min(cur.shape.z - 1, Math.max(0, cur.display.z + Math.sign(e.deltaY || e.deltaX))) });
        return;
      }
      if (e.altKey && cur.shape.t > 1) {
        st.setPosition(cur.id, { t: Math.min(cur.shape.t - 1, Math.max(0, cur.display.t + Math.sign(e.deltaY || e.deltaX))) });
        return;
      }
      const r = el.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0022)), e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [img.id, zoomAt]);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0 && e.button !== 1) return;
    dragRef.current = { sx: e.clientX, sy: e.clientY, view: useStore.getState().views[img.id] || view, moved: false, button: e.button };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = dragRef.current;
    if (!d) return;
    if (!d.moved && Math.hypot(e.clientX - d.sx, e.clientY - d.sy) < 3) return;
    if (!d.moved) {
      d.moved = true;
      setPanning(true);
    }
    setView({ scale: d.view.scale, tx: d.view.tx + (e.clientX - d.sx), ty: d.view.ty + (e.clientY - d.sy) });
  };
  const onPointerUp = () => {
    const d = dragRef.current;
    dragRef.current = null;
    setPanning(false);
    if (d && !d.moved && d.button === 0) onActivate();
  };

  // per-plane items (position: {t, z, c}) only on their own plane, like Viewport / burn_overlay
  const overlayRois = useMemo(
    () => (showOverlay ? (img.overlay as unknown as Record<string, any>[]).filter((o) => ROI_TYPES.has(String(o.type)) && onPlane(o, img.display.t, img.display.z)) as unknown as Roi[] : []),
    [showOverlay, img.overlay, img.display.t, img.display.z],
  );
  const managerRois = useMemo(() => (showManagerRois && rois ? rois.filter((r) => !r.image_id || r.image_id === img.id) : []), [showManagerRois, rois, img.id]);

  return (
    <div ref={wrapRef} className={"lite-pane" + (panning ? " panning" : "")} tabIndex={-1} onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={onPointerUp} onPointerCancel={onPointerUp} onContextMenu={(e) => e.preventDefault()}>
      <canvas ref={canvasRef} />
      <svg className="overlay" style={{ fontFamily: "var(--overlay-font)" }}>
        <g transform={`translate(${view.tx} ${view.ty}) scale(${view.scale})`}>
          {overlayRois.map((r, i) => <RoiShape key={(r as any).id || i} roi={r} color={r.color || "#ffd54a"} width={r.stroke_width || 1.5} scale={view.scale} label={(r as any).label} />)}
          {managerRois.map((r) => <RoiShape key={r.id} roi={r} color={selectedRoiIds.includes(r.id!) ? "#34d399" : "rgba(80,220,255,0.85)"} width={selectedRoiIds.includes(r.id!) ? 2 : 1} scale={view.scale} label={r.name} />)}
          {selection && <RoiShape roi={selection} color="var(--roi)" width={prefs.roi_stroke_width || 1.5} scale={view.scale} pointSize={prefs.point_size} />}
        </g>
      </svg>
      <div className="viewport-hud">
        <span className="hud-chip" title={img.name}>{img.name.length > 40 ? img.name.slice(0, 38) + "…" : img.name}</span>
        {threshold && <span className="hud-chip" style={{ color: "#f87171" }}>threshold</span>}
        {stride > 1 && <span className="hud-chip" title="Downsampled for display; zoom in for full resolution">1/{stride}</span>}
      </div>
      <div className="zoom-controls"><span className="zoom-val" title="Zoom">{Math.round(view.scale * 100)}%</span></div>
      <div className="activate-hint">Click to make this image active · drag to pan · wheel to zoom</div>
    </div>
  );
}
