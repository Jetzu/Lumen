/**
 * Plugins ▸ Utilities ▸ Capture Screen / Capture Delayed… / Capture Image.
 *
 * Capture Screen grabs ONE video frame of the screen (or window / tab) the user picks in the browser's
 * getDisplayMedia dialog, draws it on a canvas and uploads it as a PNG so that it opens as a new RGB image;
 * Capture Delayed does the same after a countdown.  Capture Image flattens the current view client-side:
 * the composited plane (render/snapshot.ts: LUTs, display ranges, composite mode, threshold colouring) plus
 * the overlay as the viewport draws it, at full image resolution.
 */
import { flushSync } from "react-dom";
import { api } from "./api";
import { useStore } from "./store";
import { renderImagePng } from "./render/snapshot";
import type { ImageSummary, ViewTransform } from "./types";

function stamp(): string {
  return new Date().toISOString().slice(0, 19).replace("T", " ").replace(/:/g, "-");
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => window.setTimeout(r, ms));
}

function toBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encoding failed"))), "image/png"));
}

/** Upload a PNG through the normal upload path so that it opens as a new (RGB) image. */
async function openPng(blob: Blob, name: string): Promise<void> {
  await api.upload(new File([blob], name, { type: "image/png" }));
  await useStore.getState().refresh();
}

/** Wait until the video element has a decoded frame to draw. */
async function firstFrame(video: HTMLVideoElement): Promise<void> {
  if (video.readyState < 2) {
    await new Promise<void>((resolve, reject) => {
      const t = window.setTimeout(() => reject(new Error("no video frame arrived")), 8000);
      video.addEventListener("loadeddata", () => { window.clearTimeout(t); resolve(); }, { once: true });
      video.addEventListener("error", () => { window.clearTimeout(t); reject(new Error("the capture stream failed")); }, { once: true });
    });
  }
  const rvfc = (video as HTMLVideoElement & { requestVideoFrameCallback?: (cb: () => void) => number }).requestVideoFrameCallback;
  if (typeof rvfc === "function") await new Promise<void>((r) => rvfc.call(video, () => r()));
  else await sleep(120);
}

/** Capture Screen (delay 0) / Capture Delayed… (countdown in seconds, after the browser's screen picker). */
export async function captureScreen(delaySeconds = 0): Promise<void> {
  const st = useStore.getState();
  st.setMenuOpen(null);
  const md = navigator.mediaDevices as MediaDevices | undefined;
  if (!window.isSecureContext || !md || typeof md.getDisplayMedia !== "function") {
    return st.toast("This browser cannot capture the screen here (getDisplayMedia needs localhost or https) – use Capture Image or File ▸ Export ▸ PNG instead", "error");
  }
  let stream: MediaStream;
  try {
    stream = await md.getDisplayMedia({ video: true, audio: false });
  } catch (e: any) {
    const name = e?.name || "";
    if (name === "NotAllowedError" || name === "AbortError" || name === "SecurityError" || name === "NotFoundError") return st.toast("Screen capture was cancelled or denied by the browser", "info");
    return st.toast(`Screen capture failed: ${e?.message || e}`, "error");
  }
  try {
    for (let s = Math.round(delaySeconds); s > 0; s--) {
      useStore.getState().toast(`Capturing the screen in ${s} s…`, "info");
      await sleep(1000);
      if (!stream.active) throw new Error("screen sharing was stopped");
    }
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.srcObject = stream;
    await video.play();
    await firstFrame(video);
    const w = video.videoWidth;
    const h = video.videoHeight;
    if (!w || !h) throw new Error("no video frame");
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    canvas.getContext("2d")!.drawImage(video, 0, 0, w, h);
    video.pause();
    video.srcObject = null;
    const blob = await toBlob(canvas);
    await openPng(blob, `Screen ${stamp()}.png`);
    useStore.getState().toast(`Captured the screen (${w}×${h} px) as a new image`, "success");
  } catch (e: any) {
    useStore.getState().toast(`Screen capture failed: ${e?.message || e}`, "error");
  } finally {
    for (const t of stream.getTracks()) t.stop();
  }
}

/** Capture Delayed…: ask for the delay (seconds), then capture. */
export function captureDelayed(arg?: unknown): void {
  let secs = typeof arg === "number" ? arg : NaN;
  if (!Number.isFinite(secs)) {
    const answer = window.prompt("Capture the screen after how many seconds? (the browser asks which screen first)", "5");
    if (answer === null) return;
    secs = parseFloat(answer);
  }
  if (!Number.isFinite(secs) || secs < 0) return useStore.getState().toast("Enter a delay in seconds", "error");
  void captureScreen(Math.min(600, secs));
}

/** Resolve CSS custom properties (var(--roi), var(--overlay-font) …) so that the SVG renders outside the document. */
function inlineCssVars(xml: string): string {
  const cs = getComputedStyle(document.documentElement);
  let out = xml;
  for (let pass = 0; pass < 4 && out.includes("var("); pass++) {
    out = out.replace(/var\((--[A-Za-z0-9_-]+)\)/g, (_m, name: string) => cs.getPropertyValue(name).trim() || "#ffd54a");
  }
  return out;
}

/** The live viewport overlay SVG and its zoomed group (`translate(tx ty) scale(s)`). */
function liveOverlay(): { svg: SVGSVGElement; g: Element } | null {
  const svg = document.querySelector(".canvas-wrap svg.overlay") as SVGSVGElement | null;
  const g = svg ? Array.from(svg.children).find((el) => el.tagName === "g" && el.hasAttribute("transform")) : undefined;
  return svg && g ? { svg, g } : null;
}

function parseViewTransform(g: Element): ViewTransform | null {
  const m = /translate\(\s*([-+\d.eE]+)[\s,]+([-+\d.eE]+)\s*\)\s*scale\(\s*([-+\d.eE]+)\s*\)/.exec(g.getAttribute("transform") || "");
  if (!m) return null;
  const v = { tx: Number(m[1]), ty: Number(m[2]), scale: Number(m[3]) };
  return Number.isFinite(v.tx) && Number.isFinite(v.ty) && Number.isFinite(v.scale) && v.scale > 0 ? v : null;
}

/**
 * A deep copy of the overlay SVG as the viewport draws it at 100 %.  The viewport sizes ROI outlines, labels, point
 * markers, dash patterns and handles in SCREEN pixels (every `/scale` in RoiShape & co.), so a clone of the zoomed
 * DOM with its transform stripped would burn `width / zoom` image pixels: 6 px outlines at 25 %, invisible ones at
 * 800 %.  Instead the viewport is re-rendered synchronously at scale 1 (flushSync commits before anything is painted,
 * so the user never sees it), cloned, and put back to the view it had.
 */
function overlayCloneAtScale1(img: ImageSummary): SVGSVGElement | null {
  const live = liveOverlay();
  if (!live) return null;
  const current = parseViewTransform(live.g);
  if (!current || current.scale === 1) return live.svg.cloneNode(true) as SVGSVGElement;
  const store = useStore.getState();
  const saved: ViewTransform | undefined = store.views[img.id];
  flushSync(() => store.setView(img.id, { scale: 1, tx: 0, ty: 0 }));
  try {
    return (liveOverlay()?.svg ?? live.svg).cloneNode(true) as SVGSVGElement;
  } finally {
    flushSync(() => useStore.getState().setView(img.id, saved ?? current));
  }
}

/** Fetch a same-origin resource as a data: URL (image overlay items inside an SVG-as-image cannot load URLs). */
async function dataUrl(href: string): Promise<string | null> {
  try {
    const res = await fetch(href, { credentials: "same-origin" });
    if (!res.ok) return null;
    const blob = await res.blob();
    return await new Promise<string | null>((resolve) => {
      const fr = new FileReader();
      fr.onload = () => resolve(typeof fr.result === "string" ? fr.result : null);
      fr.onerror = () => resolve(null);
      fr.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

/** The stylesheet rules the overlay relies on (class-driven, so lost in a serialised SVG) with the CSS variables resolved. */
function overlayStyle(): string {
  const font = getComputedStyle(document.documentElement).getPropertyValue("--font").trim() || "system-ui, sans-serif";
  return `.scalebar-label{font-family:${font};font-weight:600;paint-order:stroke;stroke:rgba(0,0,0,0.6)}`;
}

/**
 * The viewport's SVG overlay (overlay items, ROI-manager ROIs, the selection) rasterised in IMAGE coordinates at
 * 100 % zoom, the way Burn In sizes strokes and labels; null when there is nothing to draw or the SVG cannot be rendered.
 */
async function overlayImage(src: ImageSummary, w: number, h: number): Promise<HTMLImageElement | null> {
  const clone = overlayCloneAtScale1(src);
  if (!clone) return null;
  const g = Array.from(clone.children).find((el) => el.tagName === "g" && el.hasAttribute("transform"));
  if (!g) return null;
  if (!g.children.length) return null;
  g.removeAttribute("transform"); // identity at scale 1; 1 user unit = 1 image pixel
  for (const el of Array.from(clone.children)) if (el !== g) el.remove(); // selection handles are in screen px
  // image overlay items reference /api/images/…/rgb.png: embed them, an SVG loaded as <img> fetches nothing
  for (const im of Array.from(clone.querySelectorAll("image"))) {
    const href = im.getAttribute("href") || im.getAttribute("xlink:href") || "";
    const data = href && !href.startsWith("data:") ? await dataUrl(href) : href || null;
    if (data) {
      im.setAttribute("href", data);
      im.removeAttribute("xlink:href");
    } else im.remove();
  }
  const style = document.createElementNS("http://www.w3.org/2000/svg", "style");
  style.textContent = overlayStyle();
  clone.insertBefore(style, clone.firstChild);
  clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
  clone.setAttribute("xmlns:xlink", "http://www.w3.org/1999/xlink");
  clone.setAttribute("width", String(w));
  clone.setAttribute("height", String(h));
  clone.setAttribute("viewBox", `0 0 ${w} ${h}`);
  clone.removeAttribute("class");
  const xml = inlineCssVars(new XMLSerializer().serializeToString(clone));
  const img = new Image();
  img.decoding = "async";
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error("overlay render failed"));
    img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(xml);
  });
  return img;
}

/** Capture Image: the current view flattened client-side (pixels as displayed + overlay) → a new RGB image. */
export async function captureImage(): Promise<void> {
  const st = useStore.getState();
  st.setMenuOpen(null);
  const img = st.activeImage();
  if (!img) return st.toast("Open an image first", "error");
  try {
    const w = img.shape.x;
    const h = img.shape.y;
    const pixels = await renderImagePng(img, st.luts, st.state?.thresholds[img.id] ?? null);
    let blob = pixels;
    let withOverlay = false;
    if (st.showOverlay) {
      try {
        const ov = await overlayImage(img, w, h);
        if (ov) {
          const base = await createImageBitmap(pixels);
          const canvas = document.createElement("canvas");
          canvas.width = w;
          canvas.height = h;
          const ctx = canvas.getContext("2d")!;
          ctx.drawImage(base, 0, 0);
          ctx.drawImage(ov, 0, 0, w, h);
          blob = await toBlob(canvas);
          withOverlay = true;
        }
      } catch (e) {
        console.warn("capture: overlay skipped", e);
      }
    }
    const base = img.name.replace(/\.[^.]+$/, "") || "image";
    await openPng(blob, `${base} (captured).png`);
    useStore.getState().toast(`Captured ${w}×${h} px as a new RGB image${withOverlay ? " (with overlay)" : ""}`, "success");
  } catch (e: any) {
    useStore.getState().toast(`Capture Image failed: ${e?.message || e}`, "error");
  }
}
