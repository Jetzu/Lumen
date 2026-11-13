import type { ImageSummary } from "../types";

export function chooseStride(img: ImageSummary, scale: number): number {
  const px = img.shape.x * img.shape.y;
  if (px < 1_500_000) return 1;
  let s = 1;
  while (s < 8 && scale * s * 2 <= 1 && Math.min(img.shape.x, img.shape.y) / (s * 2) >= 256) s *= 2;
  return s;
}

export function histStride(img: ImageSummary): number {
  const px = img.shape.x * img.shape.y;
  return px > 1_500_000 ? 4 : 1;
}
