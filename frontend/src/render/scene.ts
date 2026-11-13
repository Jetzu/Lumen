/**
 * 3-D scene state and geometry for the 3-D View (Plugins ▸ 3D Viewer parity).
 *
 *  - the scene document (mirrors lumen/core/ops/scene_ops.py): extra objects (other open images shown as
 *    volume / orthoslices / surface with their own colour, transparency, threshold, visibility and rigid
 *    transform), camera projection, lights, background, clipping box and landmarks;
 *  - column-major mat4 helpers shared with the renderer;
 *  - placement of objects in the shared calibrated space of the primary image;
 *  - ray picking: ray ↔ triangle mesh (Möller–Trumbore) and the first voxel above a level along a ray;
 *  - view presets, the `useScene` store and the fetch helpers for /api/scene and /api/scene/animation.
 *
 * Frames: *pixel* coordinates of an image are (x px, y px, z in xy-pixel units = slice · z-ratio), the same
 * box the mesh endpoint uses; an object's *local* frame is its centred box (longest axis = 1, y up, slice 0
 * at +z); *world* is the local frame of the primary image.  Landmarks are stored as `px` = (x, y, slice).
 */
import { create } from "zustand";
import type { ImageSummary } from "../types";
import type { Camera, OrbitView, VolumeData } from "./volume";

export type Vec3 = [number, number, number];
/** column-major 4×4 */
export type Mat4 = Float32Array;

export type DisplayMode = "volume" | "orthoslice" | "surface";
export type ObjectVolumeMode = "mip" | "alpha" | "iso";
export type Projection = "perspective" | "orthographic";

export interface Transform {
  /** calibrated units of the primary image (pixels when uncalibrated) */
  translate: Vec3;
  /** degrees about x, y, z (applied x, then y, then z) about the object's centre */
  rotate: Vec3;
  /** uniform */
  scale: number;
}

export interface SliceSettings {
  /** 0..1 along pixel x, y, z */
  pos: Vec3;
  /** YZ (x), XZ (y), XY (z) planes */
  show: [boolean, boolean, boolean];
}

export interface SceneObject {
  id: string;
  /** image id */
  image: string;
  name: string;
  /** one channel, or null = all channels of the image */
  channel: number | null;
  display: DisplayMode;
  mode: ObjectVolumeMode;
  /** '#rrggbb' override, or null = the channel LUTs */
  color: string | null;
  transparency: number;
  /** 0..1 of the display range: MIP / alpha cut-off and orthoslice cut-off */
  threshold: number;
  /** 0..1 of the display range: iso level / surface level */
  level: number;
  brightness: number;
  visible: boolean;
  transform: Transform;
  slices: SliceSettings;
  /** the image is not open (a loaded scene) */
  missing?: boolean;
}

export interface Landmark {
  id: string;
  name: string;
  /** image id the point belongs to (the primary or a scene object) */
  image: string;
  /** x px, y px, slice (fractional) */
  px: Vec3;
  color?: string | null;
}

export interface Lights {
  ambient: number;
  diffuse: number;
  specular: number;
  /** camera-relative (x right, y up, z towards the viewer) when `headlight`, else world axes */
  direction: Vec3;
  headlight: boolean;
}

export interface ClipBox {
  /** 0..1 fractions along pixel x, y, z of the primary box */
  min: Vec3;
  max: Vec3;
}

export interface SceneCamera {
  theta: number;
  phi: number;
  roll: number;
  zoom: number;
  target: Vec3;
  projection: Projection;
}

/** The JSON document scene.save writes (lumen/core/ops/scene_ops.py normalises it). */
/** A colour as a scene document carries it: '#rrggbb' (the viewer) or [r, g, b] 0..1 (scene_ops.normalize_scene emits this). */
export type DocColor = string | Vec3 | null | undefined;

export interface SceneDoc {
  version: number;
  primary: string;
  primary_name: string;
  objects: Array<Omit<SceneObject, "color"> & { color: DocColor }>;
  camera: SceneCamera;
  lights: Lights;
  background: Vec3 | null;
  clipping: ClipBox;
  landmarks: Array<Omit<Landmark, "color"> & { color?: DocColor }>;
  mode: string;
  show_box: boolean;
}

export const defaultTransform = (): Transform => ({ translate: [0, 0, 0], rotate: [0, 0, 0], scale: 1 });
export const defaultSlices = (): SliceSettings => ({ pos: [0.5, 0.5, 0.5], show: [false, false, false] });
export const defaultLights = (): Lights => ({ ambient: 0.22, diffuse: 0.78, specular: 0.3, direction: [0.35, 0.5, 0.8], headlight: true });
export const defaultClip = (): ClipBox => ({ min: [0, 0, 0], max: [1, 1, 1] });
export const isIdentityTransform = (t: Transform) => t.scale === 1 && t.translate.every((v) => v === 0) && t.rotate.every((v) => v === 0);
export const isClipActive = (c: ClipBox) => c.min.some((v) => v > 0) || c.max.some((v) => v < 1);

// ------------------------------------------------------------------ mat4 (column-major, like WebGL)
export function mat4Identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = m[5] = m[10] = m[15] = 1;
  return m;
}

export function mat4Mul(a: Mat4, b: Mat4): Mat4 {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) o[c * 4 + r] = a[r] * b[c * 4] + a[4 + r] * b[c * 4 + 1] + a[8 + r] * b[c * 4 + 2] + a[12 + r] * b[c * 4 + 3];
  return o;
}

export function mat4Translate(t: Vec3): Mat4 {
  const m = mat4Identity();
  m[12] = t[0];
  m[13] = t[1];
  m[14] = t[2];
  return m;
}

export function mat4Scale(s: Vec3): Mat4 {
  const m = new Float32Array(16);
  m[0] = s[0];
  m[5] = s[1];
  m[10] = s[2];
  m[15] = 1;
  return m;
}

/** R = Rz · Ry · Rx (degrees). */
export function mat4RotateEuler(rx: number, ry: number, rz: number): Mat4 {
  const d = Math.PI / 180;
  const cx = Math.cos(rx * d), sx = Math.sin(rx * d);
  const cy = Math.cos(ry * d), sy = Math.sin(ry * d);
  const cz = Math.cos(rz * d), sz = Math.sin(rz * d);
  const m = mat4Identity();
  // column 0
  m[0] = cz * cy;
  m[1] = sz * cy;
  m[2] = -sy;
  // column 1
  m[4] = cz * sy * sx - sz * cx;
  m[5] = sz * sy * sx + cz * cx;
  m[6] = cy * sx;
  // column 2
  m[8] = cz * sy * cx + sz * sx;
  m[9] = sz * sy * cx - cz * sx;
  m[10] = cy * cx;
  return m;
}

/** General 4×4 inverse (returns identity for a singular matrix). */
export function mat4Invert(m: Mat4): Mat4 {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];
  const b00 = a00 * a11 - a01 * a10, b01 = a00 * a12 - a02 * a10, b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11, b04 = a01 * a13 - a03 * a11, b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30, b07 = a20 * a32 - a22 * a30, b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31, b10 = a21 * a33 - a23 * a31, b11 = a22 * a33 - a23 * a32;
  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det || !Number.isFinite(det)) return mat4Identity();
  det = 1 / det;
  const o = new Float32Array(16);
  o[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  o[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  o[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  o[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  o[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  o[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  o[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  o[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  o[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  o[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  o[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  o[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  o[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  o[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  o[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  o[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return o;
}

export function transformPoint(m: Mat4, p: Vec3): Vec3 {
  const w = m[3] * p[0] + m[7] * p[1] + m[11] * p[2] + m[15] || 1;
  return [(m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12]) / w, (m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13]) / w, (m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]) / w];
}

export function transformDir(m: Mat4, d: Vec3): Vec3 {
  return [m[0] * d[0] + m[4] * d[1] + m[8] * d[2], m[1] * d[0] + m[5] * d[1] + m[9] * d[2], m[2] * d[0] + m[6] * d[1] + m[10] * d[2]];
}

/** Uniform scale factor of a rigid+scale model matrix (length of its first column). */
export function mat4ScaleFactor(m: Mat4): number {
  return Math.hypot(m[0], m[1], m[2]) || 1;
}

export const vlen = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
export const vnorm = (a: Vec3): Vec3 => {
  const l = vlen(a) || 1;
  return [a[0] / l, a[1] / l, a[2] / l];
};
export const vsub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
export const vadd = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export const vscale = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
export const vdot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const vcross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

// ------------------------------------------------------------------ frames
/** Half extents (longest axis = 1) of a (W, H, D) box. */
export function boxHalf(ext: Vec3): Vec3 {
  const m = Math.max(ext[0], ext[1], ext[2]) || 1;
  return [ext[0] / m, ext[1] / m, ext[2] / m];
}

/** (W, H, D) of an image in pixel coordinates (D = slices · z-ratio), from the live calibration. */
export function imageExtent(img: ImageSummary): Vec3 {
  const r = img.calibration.pixel_width && img.calibration.pixel_depth ? img.calibration.pixel_depth / img.calibration.pixel_width : 1;
  const zr = Number.isFinite(r) && r > 0 ? r : 1;
  return [img.shape.x, img.shape.y, img.shape.z * zr];
}

export function imageZRatio(img: ImageSummary): number {
  const r = img.calibration.pixel_width && img.calibration.pixel_depth ? img.calibration.pixel_depth / img.calibration.pixel_width : 1;
  return Number.isFinite(r) && r > 0 ? r : 1;
}

/** Pixel → local box mapping (the mesh shader's uScale / uOffset): local = pixel · scale + offset. */
export function pixelMapping(ext: Vec3, half: Vec3): { scale: Vec3; offset: Vec3 } {
  return { scale: [(2 * half[0]) / (ext[0] || 1), (-2 * half[1]) / (ext[1] || 1), (-2 * half[2]) / (ext[2] || 1)], offset: [-half[0], half[1], half[2]] };
}

export function pixelToLocal(p: Vec3, ext: Vec3, half: Vec3): Vec3 {
  const { scale, offset } = pixelMapping(ext, half);
  return [p[0] * scale[0] + offset[0], p[1] * scale[1] + offset[1], p[2] * scale[2] + offset[2]];
}

export function localToPixel(l: Vec3, ext: Vec3, half: Vec3): Vec3 {
  const { scale, offset } = pixelMapping(ext, half);
  return [(l[0] - offset[0]) / scale[0], (l[1] - offset[1]) / scale[1], (l[2] - offset[2]) / scale[2]];
}

/** Landmark px (x, y, slice) → pixel coordinates (z in xy-pixel units). */
export const landmarkToPixel = (lm: Landmark, zr: number): Vec3 => [lm.px[0], lm.px[1], lm.px[2] * zr];

/** World units per calibrated unit of the primary image (pixels when uncalibrated). */
export function worldPerUnit(primary: ImageSummary, primaryHalf: Vec3): number {
  const pw = primary.calibration.unit && !/^pixels?$/i.test(primary.calibration.unit) && primary.calibration.pixel_width > 0 ? primary.calibration.pixel_width : 1;
  return (2 * primaryHalf[0]) / (primary.shape.x || 1) / pw;
}

function pixelWidthOf(img: ImageSummary): number {
  const cal = img.calibration;
  return cal.unit && !/^pixels?$/i.test(cal.unit) && cal.pixel_width > 0 && Number.isFinite(cal.pixel_width) ? cal.pixel_width : 1;
}

/** The rigid user transform about the object's local origin (its centre): T(t) · R · S. */
export function transformMatrix(t: Transform, worldPerUnitK: number): Mat4 {
  const tr = mat4Translate([t.translate[0] * worldPerUnitK, -t.translate[1] * worldPerUnitK, -t.translate[2] * worldPerUnitK]);
  // pixel axes (y down, z into the stack) → world (y up, z out): a reflection of y and z flips the sense of the y and z rotations
  const r = mat4RotateEuler(t.rotate[0], -t.rotate[1], -t.rotate[2]);
  return mat4Mul(tr, mat4Mul(r, mat4Scale([t.scale, t.scale, t.scale])));
}

/**
 * Model matrix (local box of `obj` → world) that places `obj` in the calibrated space of `primary` (its
 * pixel origin on the primary's pixel origin, sizes in calibrated units), followed by the user transform.
 * With uncalibrated images pixels coincide.
 */
export function objectModel(primary: ImageSummary, primaryHalf: Vec3, obj: ImageSummary, objHalf: Vec3, transform: Transform): Mat4 {
  const k1 = (2 * primaryHalf[0]) / (primary.shape.x || 1); // world per primary xy pixel
  const k2 = (2 * objHalf[0]) / (obj.shape.x || 1); // local per object xy pixel
  const s = ((pixelWidthOf(obj) / pixelWidthOf(primary)) * k1) / k2;
  const base = mat4Mul(mat4Translate([-primaryHalf[0] + s * objHalf[0], primaryHalf[1] - s * objHalf[1], primaryHalf[2] - s * objHalf[2]]), mat4Scale([s, s, s]));
  // the user transform rotates / scales about the object's centre (placed first), then translates
  const user = transformMatrix(transform, worldPerUnit(primary, primaryHalf));
  const centre = transformPoint(base, [0, 0, 0]);
  return mat4Mul(mat4Translate(centre), mat4Mul(user, mat4Mul(mat4Translate([-centre[0], -centre[1], -centre[2]]), base)));
}

export function primaryModel(primary: ImageSummary, primaryHalf: Vec3, transform: Transform): Mat4 {
  return isIdentityTransform(transform) ? mat4Identity() : transformMatrix(transform, worldPerUnit(primary, primaryHalf));
}

/** World → calibrated coordinates of the primary image (x right, y down, z into the stack; unit of the primary). */
export function worldToCalibrated(primary: ImageSummary, primaryHalf: Vec3, w: Vec3): Vec3 {
  const ext = imageExtent(primary);
  const p = localToPixel(w, ext, primaryHalf);
  const pw = pixelWidthOf(primary);
  const ph = primary.calibration.pixel_height > 0 && pw !== 1 ? primary.calibration.pixel_height : pw;
  return [p[0] * pw, p[1] * ph, p[2] * pw];
}

// ------------------------------------------------------------------ rays
export const FOV_RAD = (35 * Math.PI) / 180;

/** Camera basis with roll (shared with the renderer). */
export function basisOf(cam: Camera): { eye: Vec3; right: Vec3; up: Vec3; fwd: Vec3 } {
  const cp = Math.cos(cam.phi);
  const eye: Vec3 = [cam.target[0] + cam.dist * cp * Math.sin(cam.theta), cam.target[1] + cam.dist * Math.sin(cam.phi), cam.target[2] + cam.dist * cp * Math.cos(cam.theta)];
  const fwd = vnorm(vsub(cam.target, eye));
  let right = vcross(fwd, [0, 1, 0]);
  if (vlen(right) < 1e-6) right = [Math.cos(cam.theta), 0, -Math.sin(cam.theta)]; // looking straight down / up
  right = vnorm(right);
  let up = vcross(right, fwd);
  const roll = cam.roll || 0;
  if (roll) {
    const c = Math.cos(roll), s = Math.sin(roll);
    const r2: Vec3 = [right[0] * c + up[0] * s, right[1] * c + up[1] * s, right[2] * c + up[2] * s];
    up = [-right[0] * s + up[0] * c, -right[1] * s + up[1] * c, -right[2] * s + up[2] * c];
    right = r2;
  }
  return { eye, right, up, fwd };
}

/** Depth range of a render (the orthographic projection spans −far..far around the eye plane, the perspective one 0.05..far). */
export function orthoFar(cam: Camera): number {
  return Math.max(50, cam.dist * 4);
}

/** The world-space ray through normalised device coordinates (−1..1, y up). */
export function rayFromNdc(cam: Camera, ndcX: number, ndcY: number, aspect: number, projection: Projection): { origin: Vec3; dir: Vec3 } {
  const { eye, right, up, fwd } = basisOf(cam);
  const tanHalf = Math.tan(FOV_RAD / 2);
  if (projection === "orthographic") {
    const hh = cam.dist * tanHalf;
    // parallel rays start on the orthographic near plane (−far behind the eye), not on the eye plane: zoomed in, the
    // eye plane cuts through the volume while the projection still shows what lies behind it, so a pick must see the
    // same geometry the frame draws (rayBox / intersectRayMesh only accept t ≥ 0)
    const origin = vadd(vsub(eye, vscale(fwd, orthoFar(cam))), vadd(vscale(right, ndcX * hh * aspect), vscale(up, ndcY * hh)));
    return { origin, dir: fwd };
  }
  const dir = vnorm(vadd(fwd, vadd(vscale(right, ndcX * tanHalf * aspect), vscale(up, ndcY * tanHalf))));
  return { origin: eye, dir };
}

/** Nearest ray / triangle-mesh hit (Möller–Trumbore, back faces included). `t` is in units of |dir|. */
export function intersectRayMesh(origin: Vec3, dir: Vec3, vertices: Float32Array, faces: Uint32Array): { t: number; point: Vec3; face: number } | null {
  const ox = origin[0], oy = origin[1], oz = origin[2];
  const dx = dir[0], dy = dir[1], dz = dir[2];
  let best = Infinity;
  let bestFace = -1;
  const n = faces.length / 3;
  for (let f = 0; f < n; f++) {
    const i0 = faces[f * 3] * 3, i1 = faces[f * 3 + 1] * 3, i2 = faces[f * 3 + 2] * 3;
    const v0x = vertices[i0], v0y = vertices[i0 + 1], v0z = vertices[i0 + 2];
    const e1x = vertices[i1] - v0x, e1y = vertices[i1 + 1] - v0y, e1z = vertices[i1 + 2] - v0z;
    const e2x = vertices[i2] - v0x, e2y = vertices[i2 + 1] - v0y, e2z = vertices[i2 + 2] - v0z;
    const px = dy * e2z - dz * e2y, py = dz * e2x - dx * e2z, pz = dx * e2y - dy * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (det > -1e-12 && det < 1e-12) continue;
    const inv = 1 / det;
    const tx = ox - v0x, ty = oy - v0y, tz = oz - v0z;
    const u = (tx * px + ty * py + tz * pz) * inv;
    if (u < -1e-6 || u > 1 + 1e-6) continue;
    const qx = ty * e1z - tz * e1y, qy = tz * e1x - tx * e1z, qz = tx * e1y - ty * e1x;
    const v = (dx * qx + dy * qy + dz * qz) * inv;
    if (v < -1e-6 || u + v > 1 + 1e-6) continue;
    const t = (e2x * qx + e2y * qy + e2z * qz) * inv;
    if (t > 1e-9 && t < best) {
      best = t;
      bestFace = f;
    }
  }
  if (bestFace < 0) return null;
  return { t: best, point: [ox + dx * best, oy + dy * best, oz + dz * best], face: bestFace };
}

/** (t near, t far) of a ray against an axis-aligned box; null when it misses. Works with an unnormalised direction. */
export function rayBox(origin: Vec3, dir: Vec3, lo: Vec3, hi: Vec3): [number, number] | null {
  let tn = -Infinity;
  let tf = Infinity;
  for (let i = 0; i < 3; i++) {
    if (Math.abs(dir[i]) < 1e-12) {
      if (origin[i] < lo[i] || origin[i] > hi[i]) return null;
      continue;
    }
    const a = (lo[i] - origin[i]) / dir[i];
    const b = (hi[i] - origin[i]) / dir[i];
    tn = Math.max(tn, Math.min(a, b));
    tf = Math.min(tf, Math.max(a, b));
  }
  tn = Math.max(tn, 0);
  return tf > tn ? [tn, tf] : null;
}

/**
 * First sample along a ray (in the object's local box frame) whose voxel value is >= `level` (0..1 of the uint8
 * volume); the position is refined by bisection.  `channels` lists the channel indices to test (the brightest
 * above its level wins at each sample, like the iso shader).  Returns the local hit point or null.
 */
export function firstVoxelAbove(vol: VolumeData, half: Vec3, origin: Vec3, dir: Vec3, levels: (number | null)[]): { t: number; point: Vec3; channel: number } | null {
  const [Z, Y, X] = vol.shape;
  const texScale: Vec3 = [0.5 / half[0], -0.5 / half[1], -0.5 / half[2]];
  const span = rayBox(origin, dir, [-half[0], -half[1], -half[2]], half);
  if (!span) return null;
  const dl = vlen(dir) || 1;
  const voxel = (2 * Math.max(half[0], half[1], half[2])) / Math.max(X, Y, Z * vol.spacing);
  const step = (voxel * 0.5) / dl;
  const active: number[] = [];
  for (let i = 0; i < vol.channels.length; i++) if (levels[i] !== null && levels[i] !== undefined) active.push(i);
  if (!active.length) return null;
  const sample = (t: number): number => {
    const px = origin[0] + dir[0] * t, py = origin[1] + dir[1] * t, pz = origin[2] + dir[2] * t;
    const tx = px * texScale[0] + 0.5, ty = py * texScale[1] + 0.5, tz = pz * texScale[2] + 0.5;
    const ix = Math.min(X - 1, Math.max(0, Math.floor(tx * X)));
    const iy = Math.min(Y - 1, Math.max(0, Math.floor(ty * Y)));
    const iz = Math.min(Z - 1, Math.max(0, Math.floor(tz * Z)));
    const k = (iz * Y + iy) * X + ix;
    let bestCh = -1;
    let bestExcess = -1;
    for (const c of active) {
      const v = vol.channels[c][k] / 255;
      const ex = v - (levels[c] as number);
      if (ex >= 0 && ex > bestExcess) {
        bestExcess = ex;
        bestCh = c;
      }
    }
    return bestCh;
  };
  const maxSteps = 8192;
  let prev = span[0];
  for (let i = 0, t = span[0]; i < maxSteps && t <= span[1]; i++, t += step) {
    const ch = sample(t);
    if (ch >= 0) {
      let a = prev, b = t;
      for (let k = 0; k < 6; k++) {
        const m = 0.5 * (a + b);
        if (sample(m) >= 0) b = m;
        else a = m;
      }
      return { t: b, point: [origin[0] + dir[0] * b, origin[1] + dir[1] * b, origin[2] + dir[2] * b], channel: ch };
    }
    prev = t;
  }
  return null;
}

// ------------------------------------------------------------------ view presets
export const VIEW_PRESETS = {
  front: { theta: 0, phi: 0 },
  back: { theta: Math.PI, phi: 0 },
  left: { theta: -Math.PI / 2, phi: 0 },
  right: { theta: Math.PI / 2, phi: 0 },
  top: { theta: 0, phi: 1.5698 },
  bottom: { theta: 0, phi: -1.5698 },
} as const;
export type ViewPreset = keyof typeof VIEW_PRESETS;

export function presetView(name: ViewPreset, current: OrbitView): OrbitView {
  const p = VIEW_PRESETS[name];
  return { ...current, theta: p.theta, phi: p.phi, roll: 0, target: [0, 0, 0] };
}

/** '#rrggbb' → [r, g, b] 0..1; a [r, g, b] list (the backend's form) passes through, anything else is null. */
export function hexToRgb(hex: DocColor): Vec3 | null {
  if (!hex) return null;
  if (Array.isArray(hex)) {
    if (hex.length !== 3) return null;
    const k = (v: unknown) => Math.min(1, Math.max(0, Number(v) || 0));
    return [k(hex[0]), k(hex[1]), k(hex[2])];
  }
  const m = String(hex).trim().match(/^#?([0-9a-f]{6})$/i);
  if (!m) return null;
  const s = m[1];
  return [parseInt(s.slice(0, 2), 16) / 255, parseInt(s.slice(2, 4), 16) / 255, parseInt(s.slice(4, 6), 16) / 255];
}

export function rgbToHex(c: Vec3 | null | undefined): string | null {
  if (!c) return null;
  const h = (v: number) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, "0");
  return `#${h(c[0])}${h(c[1])}${h(c[2])}`;
}

/** A document colour as the store keeps it ('#rrggbb' or null): the backend serialises colours as [r, g, b] lists. */
export function docColorToHex(c: DocColor): string | null {
  if (!c) return null;
  if (Array.isArray(c)) return c.length === 3 ? rgbToHex(hexToRgb(c)) : null;
  return hexToRgb(c) ? `#${String(c).trim().replace(/^#/, "").toLowerCase()}` : null;
}

export const LANDMARK_COLORS = ["#ffd54a", "#4fc3f7", "#f06292", "#aed581", "#ffb74d", "#ba68c8", "#4dd0e1", "#e57373"];

// ------------------------------------------------------------------ store
let nextId = 1;
const newId = (p: string) => `${p}${Date.now().toString(36)}${(nextId++).toString(36)}`;

export interface SceneState {
  projection: Projection;
  lights: Lights;
  /** '#rrggbb' or null = the theme's viewport background */
  background: string | null;
  clip: ClipBox;
  /** the current image's own transform / orthoslices / colour override */
  primary: { transform: Transform; slices: SliceSettings; color: string | null };
  objects: SceneObject[];
  selectedObject: string | null;
  landmarks: Landmark[];
  selectedLandmark: string | null;
  /** clicks on the canvas place landmarks */
  placing: boolean;
  /** landmark sphere size (px) */
  landmarkSize: number;
  set: (p: Partial<Omit<SceneState, "set" | "addObject" | "updateObject" | "removeObject" | "addLandmark" | "updateLandmark" | "removeLandmark" | "clearLandmarks" | "reset" | "applyDoc">>) => void;
  addObject: (img: ImageSummary, display?: DisplayMode) => SceneObject;
  updateObject: (id: string, patch: Partial<SceneObject>) => void;
  removeObject: (id: string) => void;
  addLandmark: (lm: { image: string; px: Vec3; name?: string }) => Landmark;
  updateLandmark: (id: string, patch: Partial<Landmark>) => void;
  removeLandmark: (id: string) => void;
  clearLandmarks: () => void;
  reset: () => void;
  /**
   * Apply a scene document (scene.load); objects whose image is not open are kept but flagged.  `currentId` is the
   * image the viewer shows: its entry (not necessarily the saved primary) feeds the `primary` slot, every other entry
   * - the saved primary included, when the scene was saved from another image - becomes a regular object.
   */
  applyDoc: (doc: SceneDoc, images: ImageSummary[], currentId?: string) => void;
}

const baseState = () => ({
  projection: "perspective" as Projection,
  lights: defaultLights(),
  background: null as string | null,
  clip: defaultClip(),
  primary: { transform: defaultTransform(), slices: defaultSlices(), color: null as string | null },
  objects: [] as SceneObject[],
  selectedObject: null as string | null,
  landmarks: [] as Landmark[],
  selectedLandmark: null as string | null,
  placing: false,
  landmarkSize: 9,
});

export const useScene = create<SceneState>((set, get) => ({
  ...baseState(),
  set: (p) => set(p),
  addObject: (img, display = "volume") => {
    const obj: SceneObject = {
      id: newId("o"), image: img.id, name: img.name, channel: null, display, mode: "mip", color: null, transparency: 0, threshold: 0.12, level: 0.4,
      brightness: 1, visible: true, transform: defaultTransform(), slices: defaultSlices(),
    };
    set({ objects: [...get().objects, obj], selectedObject: obj.id });
    return obj;
  },
  updateObject: (id, patch) => set({ objects: get().objects.map((o) => (o.id === id ? { ...o, ...patch } : o)) }),
  removeObject: (id) => set({ objects: get().objects.filter((o) => o.id !== id), selectedObject: get().selectedObject === id ? null : get().selectedObject }),
  addLandmark: (lm) => {
    const n = get().landmarks.length;
    const out: Landmark = { id: newId("l"), name: lm.name || `L${n + 1}`, image: lm.image, px: lm.px, color: LANDMARK_COLORS[n % LANDMARK_COLORS.length] };
    set({ landmarks: [...get().landmarks, out], selectedLandmark: out.id });
    return out;
  },
  updateLandmark: (id, patch) => set({ landmarks: get().landmarks.map((l) => (l.id === id ? { ...l, ...patch } : l)) }),
  removeLandmark: (id) => set({ landmarks: get().landmarks.filter((l) => l.id !== id), selectedLandmark: get().selectedLandmark === id ? null : get().selectedLandmark }),
  clearLandmarks: () => set({ landmarks: [], selectedLandmark: null }),
  reset: () => set(baseState()),
  applyDoc: (doc, images, currentId) => {
    const open = new Map(images.map((im) => [im.id, im]));
    // colours arrive as [r, g, b] lists from the backend (normalize_scene): the store holds '#rrggbb' because the colour
    // inputs, swatches and label styles expect it, and hexToRgb() runs on it inside the render callback
    const all: SceneObject[] = doc.objects.map((o) => ({ ...o, color: docColorToHex(o.color), missing: !open.has(o.image), name: open.get(o.image)?.name || o.name }));
    const mine = currentId ? all.find((o) => o.image === currentId) : doc.primary ? all.find((o) => o.image === doc.primary) : undefined;
    const seen = new Set<string>();
    const objects = all
      .filter((o) => o !== mine && o.image !== currentId)
      .map((o) => {
        // "primary" is the renderer's own key for the current image (volume.ts PRIMARY) and ids must be unique
        const id = !o.id || o.id === "primary" || seen.has(o.id) ? newId("o") : o.id;
        seen.add(id);
        return id === o.id ? o : { ...o, id };
      });
    set({
      projection: doc.camera.projection,
      lights: { ...defaultLights(), ...doc.lights },
      background: rgbToHex(doc.background),
      clip: doc.clipping,
      primary: mine ? { transform: mine.transform, slices: mine.slices, color: mine.color } : { transform: defaultTransform(), slices: defaultSlices(), color: null },
      objects,
      landmarks: doc.landmarks.map((l, i) => ({ ...l, color: docColorToHex(l.color) || LANDMARK_COLORS[i % LANDMARK_COLORS.length] })),
      selectedLandmark: null,
      selectedObject: null,
    });
  },
}));

/** The scene document for scene.save / PUT /api/scene (the primary image is object 0 so its settings round-trip). */
export function sceneDoc(st: SceneState, primary: ImageSummary, view: OrbitView, mode: string, showBox: boolean): SceneDoc {
  const primaryObj: SceneObject = {
    id: "primary", image: primary.id, name: primary.name, channel: null, display: mode === "mesh" ? "surface" : "volume",
    mode: mode === "mesh" ? "iso" : (mode as ObjectVolumeMode), color: st.primary.color, transparency: 0, threshold: 0.12, level: 0.4, brightness: 1,
    visible: true, transform: st.primary.transform, slices: st.primary.slices,
  };
  return {
    version: 1,
    primary: primary.id,
    primary_name: primary.name,
    // an object of the current image is dormant while that image is the primary (drawn once, as the primary): never list it twice
    objects: [primaryObj, ...st.objects.filter((o) => o.image !== primary.id).map((o) => ({ ...o }))],
    camera: { theta: view.theta, phi: view.phi, roll: view.roll || 0, zoom: view.zoom, target: view.target, projection: st.projection },
    lights: st.lights,
    background: hexToRgb(st.background),
    clipping: st.clip,
    landmarks: st.landmarks,
    mode,
    show_box: showBox,
  };
}

// ------------------------------------------------------------------ server
async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const j = await res.json().catch(() => ({}));
    throw new Error((j as { error?: string }).error || res.statusText || "request failed");
  }
  return (await res.json()) as T;
}

export const pushScene = (doc: SceneDoc) => fetch("/api/scene", { method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ scene: doc }) }).then((r) => handle<{ rev: number }>(r));

export const fetchScene = () => fetch("/api/scene").then((r) => handle<{ scene: SceneDoc | null; rev: number }>(r));

export interface AnimationResult {
  id: string;
  frames: number;
  width: number;
  height: number;
  fps: number;
  path?: string;
  message: string;
}

/** POST the recorded canvas frames: they become an RGB time stack (and optionally a GIF / video file). */
export function postAnimation(frames: Blob[], name: string, fps: number, format = "", path = ""): Promise<AnimationResult> {
  const fd = new FormData();
  frames.forEach((b, i) => fd.append("frames", b, `frame${String(i).padStart(4, "0")}.png`));
  fd.append("name", name);
  fd.append("fps", String(fps));
  fd.append("format", format);
  fd.append("path", path);
  return fetch("/api/scene/animation", { method: "POST", body: fd }).then((r) => handle<AnimationResult>(r));
}

export const animationGifUrl = (imageId: string, fps: number) => `/api/images/${imageId}/animation.gif?fps=${encodeURIComponent(String(fps))}`;

export function canvasBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not read the canvas"))), "image/png"));
}
