/**
 * Full-resolution PNG of the composited current plane (what the viewport shows: LUTs, display
 * ranges, composite/single mode, threshold colouring), optionally cropped.  Used by
 * Edit ▸ Copy to System.
 */
import { getPlanes } from "../store";
import { composite, type LutTables } from "./compositor";
import type { ImageSummary, Threshold } from "../types";

export async function renderImagePng(img: ImageSummary, luts: LutTables, threshold: Threshold | null, crop?: { x: number; y: number; w: number; h: number }): Promise<Blob> {
  const pd = await getPlanes(img, img.display.t, img.display.z, 1);
  const [, h, w] = pd.shape;
  const rgba = composite(pd.channels, w, h, img.display, img.rgb, luts, { threshold, thresholdChannel: img.display.c });
  const full = document.createElement("canvas");
  full.width = w;
  full.height = h;
  full.getContext("2d")!.putImageData(new ImageData(rgba, w, h), 0, 0);
  let out = full;
  if (crop) {
    const x0 = Math.max(0, Math.floor(crop.x));
    const y0 = Math.max(0, Math.floor(crop.y));
    const cw = Math.min(w - x0, Math.max(1, Math.round(crop.w)));
    const ch = Math.min(h - y0, Math.max(1, Math.round(crop.h)));
    if (cw > 0 && ch > 0 && (cw < w || ch < h)) {
      out = document.createElement("canvas");
      out.width = cw;
      out.height = ch;
      out.getContext("2d")!.drawImage(full, x0, y0, cw, ch, 0, 0, cw, ch);
    }
  }
  return new Promise<Blob>((resolve, reject) => out.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encoding failed"))), "image/png"));
}
