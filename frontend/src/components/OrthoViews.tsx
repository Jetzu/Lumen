import React, { useEffect, useMemo, useRef, useState } from "react";
import { useStore } from "../store";
import { composite } from "../render/compositor";
import type { Calibration, ImageSummary, Roi, Threshold, TypedArray } from "../types";

interface OrthoData {
  dtype: string;
  /** XZ plane: rows = z, cols = x */
  xz: { channels: TypedArray[]; w: number; h: number };
  /** YZ plane: rows = z, cols = y */
  yz: { channels: TypedArray[]; w: number; h: number };
  point: [number, number];
}

/**
 * Z stretch that makes voxels isotropic, from the live calibration (Properties / Set Scale change it
 * without bumping `rev`, so it must not come from the fetched X-Spacing header).  Mirrors the server guard.
 */
function zStretch(cal: Calibration): number {
  const r = cal.pixel_width && cal.pixel_depth ? cal.pixel_depth / cal.pixel_width : 1;
  return Number.isFinite(r) && r > 0 ? r : 1;
}

function makeArray(dtype: string, buf: ArrayBuffer, offsetElems: number, n: number): TypedArray {
  if (dtype === "uint8") return new Uint8Array(buf, offsetElems, n);
  if (dtype === "uint16") return new Uint16Array(buf, offsetElems * 2, n);
  return new Float32Array(buf, offsetElems * 4, n);
}

async function fetchOrtho(id: string, x: number, y: number, t: number): Promise<OrthoData> {
  const res = await fetch(`/api/images/${id}/ortho?x=${x}&y=${y}&t=${t}`);
  if (!res.ok) {
    const j = await res.json().catch(() => ({}));
    throw new Error(j.error || res.statusText || "ortho fetch failed");
  }
  const dtype = res.headers.get("x-dtype") || "uint8";
  const [zx, X] = (res.headers.get("x-shape-xz") || "1,1").split(",").map(Number);
  const [zy, Y] = (res.headers.get("x-shape-yz") || "1,1").split(",").map(Number);
  const C = Number(res.headers.get("x-channels") || 1);
  const pt = (res.headers.get("x-point") || `${x},${y}`).split(",").map(Number) as [number, number];
  const buf = await res.arrayBuffer();
  const nxz = zx * X;
  const nyz = zy * Y;
  const xz: TypedArray[] = [];
  const yz: TypedArray[] = [];
  for (let c = 0; c < C; c++) xz.push(makeArray(dtype, buf, c * nxz, nxz));
  for (let c = 0; c < C; c++) yz.push(makeArray(dtype, buf, C * nxz + c * nyz, nyz));
  return { dtype, xz: { channels: xz, w: X, h: zx }, yz: { channels: yz, w: Y, h: zy }, point: pt };
}

/** The reference point for the cross-sections: point ROI, centre of an area selection, else the image centre. */
function pointFromSelection(img: ImageSummary, sel: Roi | null | undefined): [number, number] | null {
  if (!sel) return null;
  const clamp = (x: number, y: number): [number, number] => [Math.max(0, Math.min(img.shape.x - 1, Math.round(x))), Math.max(0, Math.min(img.shape.y - 1, Math.round(y)))];
  if (sel.type === "point" && sel.points?.length) return clamp(sel.points[0][0], sel.points[0][1]);
  if ((sel.type === "rect" || sel.type === "oval") && sel.w !== undefined && sel.h !== undefined) return clamp((sel.x || 0) + sel.w / 2, (sel.y || 0) + sel.h / 2);
  if (sel.type === "line" && sel.x1 !== undefined && sel.x2 !== undefined) return clamp(((sel.x1 || 0) + (sel.x2 || 0)) / 2, ((sel.y1 || 0) + (sel.y2 || 0)) / 2);
  if (sel.points?.length) {
    let sx = 0;
    let sy = 0;
    for (const p of sel.points) {
      sx += p[0];
      sy += p[1];
    }
    return clamp(sx / sel.points.length, sy / sel.points.length);
  }
  if (sel.x !== undefined && sel.y !== undefined && sel.w !== undefined && sel.h !== undefined) return clamp(sel.x + sel.w / 2, sel.y + sel.h / 2);
  return null;
}

export default function OrthoViews({ img }: { img: ImageSummary | undefined }) {
  if (!img) return <div className="hint-text">No image selected.</div>;
  if (img.shape.z < 2) return <div className="hint-text">Needs a Z stack. Orthogonal views show the XZ and YZ cross-sections of a 3-D image through a point.</div>;
  return <Ortho img={img} />;
}

function Ortho({ img }: { img: ImageSummary }) {
  const luts = useStore((s) => s.luts);
  const selection = useStore((s) => s.selections[img.id]);
  const thr = useStore((s) => s.state?.thresholds[img.id]);
  const [follow, setFollow] = useState(false);
  const cursor = useStore((s) => (follow ? s.cursor : null));
  const [data, setData] = useState<OrthoData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [width, setWidth] = useState(300);
  const wrapRef = useRef<HTMLDivElement>(null);
  const xzRef = useRef<HTMLCanvasElement>(null);
  const yzRef = useRef<HTMLCanvasElement>(null);
  const lastFetch = useRef(0);
  const seq = useRef(0);

  const point = useMemo<[number, number]>(() => {
    if (follow && cursor && cursor.x >= 0 && cursor.y >= 0 && cursor.x < img.shape.x && cursor.y < img.shape.y) return [Math.floor(cursor.x), Math.floor(cursor.y)];
    return pointFromSelection(img, selection) || [Math.floor(img.shape.x / 2), Math.floor(img.shape.y / 2)];
  }, [follow, cursor?.x, cursor?.y, selection, img.shape.x, img.shape.y]); // eslint-disable-line react-hooks/exhaustive-deps

  // panel width
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setWidth(Math.max(60, Math.floor(el.clientWidth))));
    ro.observe(el);
    setWidth(Math.max(60, Math.floor(el.clientWidth)));
    return () => ro.disconnect();
  }, []);

  // fetch (throttled to ~10/s while the point moves)
  useEffect(() => {
    const id = ++seq.current;
    const run = () => {
      lastFetch.current = performance.now();
      fetchOrtho(img.id, point[0], point[1], img.display.t)
        .then((d) => {
          if (seq.current !== id) return;
          setData(d);
          setError(null);
        })
        .catch((e) => seq.current === id && setError(e.message || String(e)));
    };
    const wait = Math.max(0, 100 - (performance.now() - lastFetch.current));
    const timer = window.setTimeout(run, wait);
    return () => window.clearTimeout(timer);
  }, [img.id, img.rev, img.display.t, point[0], point[1]]); // eslint-disable-line react-hooks/exhaustive-deps

  // draw (the stretch comes from the live calibration, so a Properties change redraws without a refetch)
  const stretch = zStretch(img.calibration);
  useEffect(() => {
    if (!data) return;
    drawPlane(xzRef.current, data.xz, img, luts, stretch, width, img.display.z, data.point[0], thr);
    drawPlane(yzRef.current, data.yz, img, luts, stretch, width, img.display.z, data.point[1], thr);
  }, [data, img.display, img.rgb, luts, width, thr, img, stretch]);

  const pickZ = (canvas: HTMLCanvasElement | null, e: React.MouseEvent, h: number) => {
    if (!canvas || e.buttons === 0 && e.type !== "click") return;
    const r = canvas.getBoundingClientRect();
    const fy = (e.clientY - r.top) / Math.max(1, r.height);
    const z = Math.max(0, Math.min(img.shape.z - 1, Math.floor(fy * h)));
    if (z !== img.display.z) useStore.getState().setPosition(img.id, { z });
  };

  return (
    <div ref={wrapRef} className="ortho">
      <div className="row" style={{ marginTop: 0 }}>
        <label className="switch" title="Use the mouse position in the main view as the reference point"><input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> Follow cursor</label>
        <span className="grow" />
        <span className="dim small mono">x {point[0]} · y {point[1]} · z {img.display.z + 1}/{img.shape.z}</span>
      </div>
      {error && <div className="hint-text" style={{ color: "var(--danger)" }}>{error}</div>}
      <div className="ortho-label">XZ <span className="dim">at y = {point[1]}</span></div>
      <canvas ref={xzRef} className="ortho-canvas" onClick={(e) => pickZ(xzRef.current, e, data?.xz.h || img.shape.z)} onMouseMove={(e) => pickZ(xzRef.current, e, data?.xz.h || img.shape.z)} onMouseDown={(e) => pickZ(xzRef.current, e, data?.xz.h || img.shape.z)} />
      <div className="ortho-label">YZ <span className="dim">at x = {point[0]}</span></div>
      <canvas ref={yzRef} className="ortho-canvas" onClick={(e) => pickZ(yzRef.current, e, data?.yz.h || img.shape.z)} onMouseMove={(e) => pickZ(yzRef.current, e, data?.yz.h || img.shape.z)} onMouseDown={(e) => pickZ(yzRef.current, e, data?.yz.h || img.shape.z)} />
      <div className="hint-text">
        Sections through the {selection ? "selection" : follow ? "cursor" : "image centre"}; draw a point or area selection to move them. Click or drag in a section to change the current slice.
        {stretch !== 1 && <> Z is stretched ×{stretch.toFixed(2)} to make voxels isotropic.</>}
      </div>
    </div>
  );
}

function drawPlane(canvas: HTMLCanvasElement | null, plane: { channels: TypedArray[]; w: number; h: number }, img: ImageSummary, luts: Record<string, number[][]>, spacing: number, cssW: number, zCur: number, col: number, thr: Threshold | undefined) {
  if (!canvas) return;
  const { w, h } = plane;
  const scale = cssW / w;
  const cssH = Math.max(6, Math.round(h * spacing * scale));
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const rgba = composite(plane.channels, w, h, img.display, img.rgb, luts, { threshold: thr, thresholdChannel: img.display.c });
  const off = document.createElement("canvas");
  off.width = w;
  off.height = h;
  const octx = off.getContext("2d");
  if (!octx) return;
  octx.putImageData(new ImageData(rgba, w, h), 0, 0);
  ctx.imageSmoothingEnabled = scale < 1;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  // crosshair: current z (horizontal) and the reference column (vertical)
  const zy = ((zCur + 0.5) / h) * canvas.height;
  const cx = ((col + 0.5) / w) * canvas.width;
  ctx.lineWidth = Math.max(1, dpr);
  ctx.strokeStyle = "rgba(52, 211, 153, 0.9)";
  ctx.beginPath();
  ctx.moveTo(0, zy);
  ctx.lineTo(canvas.width, zy);
  ctx.stroke();
  ctx.strokeStyle = "rgba(255, 213, 74, 0.75)";
  ctx.beginPath();
  ctx.moveTo(cx, 0);
  ctx.lineTo(cx, canvas.height);
  ctx.stroke();
}
