/**
 * WebGL2 ray-casting volume renderer.
 *
 * Each channel is uploaded as an R8 3-D texture; a full-screen fragment shader marches
 * rays front-to-back through the (anisotropy-corrected) bounding box.  Modes:
 *   mip   – maximum intensity projection
 *   alpha – translucent accumulation with per-channel opacity and a soft threshold
 *   iso   – first sample above the threshold, shaded from the local gradient (writes depth)
 *   mesh  – server-side marching-cubes iso-surfaces (see ./mesh.ts) drawn as lit triangle meshes
 *
 * Scene features (Plugins ▸ 3D Viewer parity, see ./scene.ts): several *objects* (the current image plus other
 * open images, each with its own textures / meshes, rigid model matrix, colour override, opacity and display
 * mode), orthoslices (three lit textured planes per object), landmarks (depth-tested shaded points), a world
 * clipping box, perspective / orthographic projection, adjustable lights and background, and off-screen
 * snapshots at any resolution.  Draw order per frame: iso volumes (depth) → opaque surfaces / slices /
 * landmarks (depth-tested) → translucent surfaces → MIP / alpha volumes blended on top → box.
 *
 * No float textures, no extensions required: every modern WebGL2 implementation runs it.
 */
import type { LutTables } from "./compositor";
import type { MeshData } from "./mesh";
import { basisOf, boxHalf, hexToRgb, isClipActive, mat4Identity, mat4Invert, mat4Mul, mat4ScaleFactor, orthoFar, transformDir, vnorm, type ClipBox, type DisplayMode, type Lights, type Mat4, type Projection, type Vec3 } from "./scene";

export type VolumeMode = "mip" | "alpha" | "iso" | "mesh";

export interface VolumeData {
  /** (Z, Y, X) voxels per channel */
  shape: [number, number, number];
  /** z voxel size / xy voxel size of this volume */
  spacing: number;
  /** source pixels per voxel in xy */
  downsample: number;
  channels: Uint8Array[];
  /** source channel indices */
  index: number[];
  luts: string[];
}

export interface ChannelSettings {
  visible: boolean;
  /** 0..1 – values below are suppressed (cut-off for mip, soft threshold for alpha) */
  threshold: number;
  /** 0..1 – iso-surface level */
  isoLevel: number;
  /** 0..1 – alpha mode density */
  opacity: number;
  /** 0..3 */
  brightness: number;
  /** 0..1 – surface opacity in mesh mode */
  meshOpacity: number;
}

/** Camera distance at which the whole box (bounding sphere) fits the view. */
export function fitDistance(half: [number, number, number]): number {
  return (Math.hypot(half[0], half[1], half[2]) / Math.sin(FOV / 2)) * 1.02;
}

export interface Camera {
  theta: number; // azimuth (rad)
  phi: number; // elevation (rad)
  dist: number;
  target: [number, number, number];
  /** rotation about the viewing axis (rad) */
  roll?: number;
}

export const FOV = (35 * Math.PI) / 180;

export function defaultCamera(): Camera {
  return { theta: 0.55, phi: 0.38, dist: 3.4, target: [0, 0, 0], roll: 0 };
}

/** Orbit view kept by the UI: distance is expressed as a zoom factor relative to the fitted distance. */
export interface OrbitView {
  theta: number;
  phi: number;
  zoom: number;
  target: [number, number, number];
  roll?: number;
}

export function defaultView(): OrbitView {
  return { theta: 0.55, phi: 0.38, zoom: 1, target: [0, 0, 0], roll: 0 };
}

export function viewToCamera(v: OrbitView, half: [number, number, number]): Camera {
  return { theta: v.theta, phi: v.phi, dist: fitDistance(half) / v.zoom, target: v.target, roll: v.roll || 0 };
}

// ------------------------------------------------------------------ fetch + cache
const cache = new Map<string, Promise<VolumeData>>();
const CACHE_ENTRIES = 4;

/** z voxel / xy voxel from a calibration, with the same guard the server uses (1 when uncalibrated or degenerate). */
export function zRatio(cal: { pixel_width: number; pixel_depth: number }): number {
  const r = cal.pixel_width && cal.pixel_depth ? cal.pixel_depth / cal.pixel_width : 1;
  return Number.isFinite(r) && r > 0 ? r : 1;
}

/**
 * `rev` only tracks pixel data; the resampled geometry also depends on the calibration
 * (z ratio), which Properties / Set Scale change without bumping `rev`, so it is part of the key.
 */
export function volumeKey(id: string, rev: number, t: number, maxDim: number, channels: string, displaySig: string, geomSig: string): string {
  return `${id}:${rev}:${t}:${maxDim}:${channels}:${displaySig}:${geomSig}`;
}

export function fetchVolume(id: string, rev: number, t: number, maxDim: number, channels: string, displaySig: string, geomSig: string): Promise<VolumeData> {
  const key = volumeKey(id, rev, t, maxDim, channels, displaySig, geomSig);
  let p = cache.get(key);
  if (!p) {
    p = (async () => {
      const res = await fetch(`/api/images/${id}/volume?t=${t}&max_dim=${maxDim}&channels=${encodeURIComponent(channels)}`);
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || res.statusText || "volume fetch failed");
      }
      const shape = (res.headers.get("x-shape") || "1,1,1").split(",").map(Number) as [number, number, number];
      const n = Number(res.headers.get("x-channels") || 1);
      const spacing = Number(res.headers.get("x-spacing") || 1) || 1;
      const downsample = Number(res.headers.get("x-downsample") || 1) || 1;
      const index = (res.headers.get("x-channel-index") || "0").split(",").map(Number);
      const luts = (res.headers.get("x-channel-luts") || "Grays").split(",");
      const buf = await res.arrayBuffer();
      const per = shape[0] * shape[1] * shape[2];
      const chans: Uint8Array[] = [];
      for (let i = 0; i < n; i++) chans.push(new Uint8Array(buf, i * per, per));
      return { shape, spacing, downsample, channels: chans, index, luts };
    })();
    cache.set(key, p);
    p.catch(() => cache.delete(key));
    while (cache.size > CACHE_ENTRIES) {
      const first = cache.keys().next().value;
      if (first === undefined) break;
      cache.delete(first);
    }
  }
  return p;
}

// ------------------------------------------------------------------ projection / view matrices (column-major)
function perspective(fov: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fov / 2);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

/** Orthographic projection with the same framing as `perspective` at the camera distance (half height `hh`). */
function orthographic(hh: number, aspect: number, near: number, far: number): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1 / (hh * aspect);
  m[5] = 1 / hh;
  m[10] = -2 / (far - near);
  m[14] = -(far + near) / (far - near);
  m[15] = 1;
  return m;
}

/** View matrix from an orthonormal basis (right, up, forward) at `eye`. */
function viewMatrix(eye: Vec3, s: Vec3, u: Vec3, f: Vec3): Mat4 {
  const m = new Float32Array(16);
  m[0] = s[0]; m[4] = s[1]; m[8] = s[2];
  m[1] = u[0]; m[5] = u[1]; m[9] = u[2];
  m[2] = -f[0]; m[6] = -f[1]; m[10] = -f[2];
  m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
  m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
  m[14] = f[0] * eye[0] + f[1] * eye[1] + f[2] * eye[2];
  m[15] = 1;
  return m;
}

/** Camera basis (eye, right, up, forward) for a turntable camera (with roll). */
export function cameraBasis(cam: Camera): { eye: Vec3; right: Vec3; up: Vec3; fwd: Vec3 } {
  return basisOf(cam);
}

/** View-projection matrix and camera basis of a render (also used for picking and label placement). */
export function cameraMatrices(cam: Camera, aspect: number, projection: Projection): { vp: Mat4; basis: { eye: Vec3; right: Vec3; up: Vec3; fwd: Vec3 } } {
  const basis = cameraBasis(cam);
  const near = 0.05;
  const far = orthoFar(cam); // shared with the pick rays (scene.ts rayFromNdc): the orthographic slab is −far..far
  const proj = projection === "orthographic" ? orthographic(cam.dist * Math.tan(FOV / 2), aspect, -far, far) : perspective(FOV, aspect, near, far);
  return { vp: mat4Mul(proj, viewMatrix(basis.eye, basis.right, basis.up, basis.fwd)), basis };
}

// ------------------------------------------------------------------ shaders
const VERT = `#version 300 es
in vec2 aPos;
out vec2 vNdc;
void main() { vNdc = aPos; gl_Position = vec4(aPos, 0.0, 1.0); }`;

// Shared GLSL: the world-space ray of a fragment (perspective or orthographic), the clip box and lighting.
const RAY_COMMON = `
uniform vec3 uEye, uRight, uUp, uFwd;
uniform float uTanHalf, uAspect;
uniform int uOrtho;          // 1 = orthographic: parallel rays through the image plane
uniform float uOrthoHalf;    // half height of the orthographic view at the target (world)
uniform int uClipOn;
uniform vec3 uClipMin, uClipMax;   // world-space clipping box
uniform float uAmbient, uDiffuse, uSpecular;
uniform vec3 uLightDir;      // world, unit, towards the light
uniform int uUseFlat;
uniform vec3 uFlat;          // colour override (instead of the LUTs)
uniform float uOpacity;      // object opacity

void worldRay(out vec3 ro, out vec3 rd) {
  if (uOrtho == 1) {
    ro = uEye + uRight * (vNdc.x * uOrthoHalf * uAspect) + uUp * (vNdc.y * uOrthoHalf);
    rd = uFwd;
  } else {
    ro = uEye;
    rd = normalize(uFwd + uRight * (vNdc.x * uTanHalf * uAspect) + uUp * (vNdc.y * uTanHalf));
  }
}

// slab test; works with an unnormalised direction (t stays in the caller's units)
vec2 slab(vec3 ro, vec3 rd, vec3 lo, vec3 hi) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (lo - ro) * inv;
  vec3 t1 = (hi - ro) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}
`;

function buildFrag(n: number): string {
  const ch = (i: number, s: string) => s.replace(/\$/g, String(i));
  const decl = Array.from({ length: n }, (_, i) => ch(i, "uniform sampler3D uTex$; uniform sampler2D uLut$; uniform vec4 uCh$;")).join("\n");
  const sampleFn = `float sampleCh(int i, vec3 tc) {\n${Array.from({ length: n }, (_, i) => ch(i, "  if (i == $) return texture(uTex$, tc).r;")).join("\n")}\n  return 0.0;\n}`;
  const lutFn = `vec3 lutCh(int i, float v) {\n  if (uUseFlat == 1) return uFlat;\n${Array.from({ length: n }, (_, i) => ch(i, "  if (i == $) return texture(uLut$, vec2(v, 0.5)).rgb;")).join("\n")}\n  return vec3(v);\n}`;
  const chFn = `vec4 chParams(int i) {\n${Array.from({ length: n }, (_, i) => ch(i, "  if (i == $) return uCh$;")).join("\n")}\n  return vec4(0.0);\n}`;
  const mipBody = Array.from({ length: n }, (_, i) => ch(i, `    if (uCh$.x > 0.5) { float v = texture(uTex$, tc).r; v = max(0.0, (v - uCh$.y) / max(1.0 - uCh$.y, 1e-3)); m$ = max(m$, v); }`)).join("\n");
  const mipDecl = Array.from({ length: n }, (_, i) => ch(i, "float m$ = 0.0;")).join(" ");
  const mipOut = Array.from({ length: n }, (_, i) => ch(i, `  if (uCh$.x > 0.5 && m$ > 0.0) { col += (uUseFlat == 1 ? uFlat * m$ : texture(uLut$, vec2(m$, 0.5)).rgb) * uCh$.w; cover = max(cover, min(1.0, m$ * uCh$.w)); }`)).join("\n");
  const alphaBody = Array.from({ length: n }, (_, i) => ch(i, `    if (uCh$.x > 0.5) {
      float v = texture(uTex$, tc).r;
      float w = smoothstep(uCh$.y, uCh$.y + 0.12, v);
      float a = clamp(w * (0.25 + 0.75 * v) * uCh$.z * uAlphaScale, 0.0, 1.0);
      if (a > 0.0) { vec3 c = (uUseFlat == 1 ? uFlat * min(1.0, 0.15 + v) : texture(uLut$, vec2(min(1.0, 0.15 + v), 0.5)).rgb) * uCh$.w; acc += (1.0 - accA) * a * c; accA += (1.0 - accA) * a; }
    }`)).join("\n");
  const isoBody = Array.from({ length: n }, (_, i) => ch(i, `    if (uCh$.x > 0.5) { float v = texture(uTex$, tc).r; if (v >= uCh$.y && v - uCh$.y > best) { best = v - uCh$.y; hit = $; } }`)).join("\n");
  return `#version 300 es
precision highp float;
precision highp sampler3D;
in vec2 vNdc;
out vec4 outColor;
uniform vec3 uHalf;       // half extents of the object's box (local)
uniform vec3 uTexScale;   // local -> texture coordinate scale (with axis flips)
uniform vec3 uTexel;      // 1 / texture size
uniform float uStep;      // march step (world units)
uniform float uAlphaScale;
uniform int uMode;
uniform mat4 uModel, uInvModel, uViewProj;
${RAY_COMMON}
${decl}
${sampleFn}
${lutFn}
${chFn}
const int MAX_STEPS = 2048;

vec3 gradient(int i, vec3 tc) {
  vec3 e = uTexel * 1.5;
  return vec3(
    sampleCh(i, tc + vec3(e.x, 0.0, 0.0)) - sampleCh(i, tc - vec3(e.x, 0.0, 0.0)),
    sampleCh(i, tc + vec3(0.0, e.y, 0.0)) - sampleCh(i, tc - vec3(0.0, e.y, 0.0)),
    sampleCh(i, tc + vec3(0.0, 0.0, e.z)) - sampleCh(i, tc - vec3(0.0, 0.0, e.z)));
}

void main() {
  vec3 roW, rdW;
  worldRay(roW, rdW);
  // the ray in the object's local box frame; uInvModel is rigid + uniform scale, so t keeps its world meaning
  vec3 ro = (uInvModel * vec4(roW, 1.0)).xyz;
  vec3 rd = (uInvModel * vec4(rdW, 0.0)).xyz;
  vec2 hitT = slab(ro, rd, -uHalf, uHalf);
  // perspective: never march behind the eye; orthographic: the projection spans -far..far around the eye plane
  // (meshes, slices and landmarks behind it are drawn), so parallel rays may enter the box at a negative t -
  // otherwise a zoomed-in ortho view cuts the front of the volume at the eye plane
  float tn = (uOrtho == 1) ? hitT.x : max(hitT.x, 0.0);
  float tf = hitT.y;
  if (uClipOn == 1) { vec2 c = slab(roW, rdW, uClipMin, uClipMax); tn = max(tn, c.x); tf = min(tf, c.y); }
  gl_FragDepth = 1.0;
  if (tf <= tn) discard;
  // jitter the start to hide banding
  float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);
  float t = tn + uStep * jitter;
  vec3 col = vec3(0.0);
  float cover = 0.0;
  if (uMode == 0) {
    ${mipDecl}
    for (int i = 0; i < MAX_STEPS; i++) {
      if (t > tf) break;
      vec3 tc = (ro + rd * t) * uTexScale + 0.5;
${mipBody}
      t += uStep;
    }
${mipOut}
    outColor = vec4(min(col, vec3(1.0)) * uOpacity, cover * uOpacity); // premultiplied
    return;
  }
  if (uMode == 1) {
    vec3 acc = vec3(0.0);
    float accA = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
      if (t > tf || accA > 0.985) break;
      vec3 tc = (ro + rd * t) * uTexScale + 0.5;
${alphaBody}
      t += uStep;
    }
    outColor = vec4(min(acc, vec3(1.0)) * uOpacity, accA * uOpacity);
    return;
  }
  // iso surface
  int hit = -1;
  float best = -1.0;
  vec3 p = ro;
  for (int i = 0; i < MAX_STEPS; i++) {
    if (t > tf) break;
    p = ro + rd * t;
    vec3 tc = p * uTexScale + 0.5;
${isoBody}
    if (hit >= 0) break;
    t += uStep;
  }
  if (hit < 0) discard;
  vec4 cp = chParams(hit);
  // bisection refinement between the previous and the hit sample
  vec3 a = p - rd * uStep;
  vec3 b = p;
  for (int k = 0; k < 5; k++) {
    vec3 m = 0.5 * (a + b);
    if (sampleCh(hit, m * uTexScale + 0.5) >= cp.y) b = m; else a = m;
  }
  vec3 tc = b * uTexScale + 0.5;
  vec3 g = gradient(hit, tc) * uTexScale / uTexel; // texture-space differences -> local orientation & anisotropy
  vec3 nrmL = length(g) > 1e-5 ? normalize(-g) : -normalize(rd);
  vec3 nrm = normalize(mat3(uModel) * nrmL);
  vec3 v = -rdW;
  if (dot(nrm, v) < 0.0) nrm = -nrm;
  float diff = max(dot(nrm, uLightDir), 0.0);
  vec3 h = normalize(uLightDir + v);
  float spec = pow(max(dot(nrm, h), 0.0), 40.0) * uSpecular;
  vec3 base = lutCh(hit, 0.85) * cp.w;
  col = base * (uAmbient + uDiffuse * diff) + vec3(spec);
  // depth of the hit so that surfaces, slices and landmarks interleave correctly
  float tHit = dot(b - ro, rd) / max(dot(rd, rd), 1e-12);
  vec4 clipPos = uViewProj * vec4(roW + rdW * tHit, 1.0);
  gl_FragDepth = clamp(clipPos.z / clipPos.w * 0.5 + 0.5, 0.0, 1.0);
  outColor = vec4(min(col, vec3(1.0)) * uOpacity, uOpacity);
}`;
}

const BOX_VERT = `#version 300 es
in vec3 aPos;
uniform mat4 uViewProj, uModel;
uniform vec3 uHalf;
void main() { gl_Position = uViewProj * uModel * vec4(aPos * uHalf, 1.0); }`;
const BOX_FRAG = `#version 300 es
precision mediump float;
uniform vec4 uColor;
out vec4 outColor;
void main() { outColor = uColor; }`;

// Mesh mode: vertices arrive in image pixel coordinates (x right, y down, z into the stack) and are
// mapped onto the object's local box (y up, slice 0 at +z) with uScale / uOffset, then by the object's
// model matrix into the world; the normals need the axis flips and the model rotation (uniform scale).
const MESH_VERT = `#version 300 es
in vec3 aPos;
in vec3 aNrm;
uniform mat4 uViewProj, uModel;
uniform vec3 uScale, uOffset;
out vec3 vNrm;
out vec3 vWorld;
void main() {
  vec4 w = uModel * vec4(aPos * uScale + uOffset, 1.0);
  vWorld = w.xyz;
  vNrm = mat3(uModel) * (aNrm * sign(uScale));
  gl_Position = uViewProj * w;
}`;
const MESH_FRAG = `#version 300 es
precision highp float;
in vec3 vNrm;
in vec3 vWorld;
uniform vec3 uColor;
uniform float uOpacity;
uniform vec3 uEye, uFwd;
uniform int uOrtho;
uniform vec3 uLightDir;
uniform float uAmbient, uDiffuse, uSpecular;
uniform int uWire;
uniform int uClipOn;
uniform vec3 uClipMin, uClipMax;
out vec4 outColor;
void main() {
  if (uClipOn == 1 && (any(lessThan(vWorld, uClipMin)) || any(greaterThan(vWorld, uClipMax)))) discard;
  if (uWire == 1) { outColor = vec4(uColor, uOpacity); return; }
  vec3 v = uOrtho == 1 ? -uFwd : normalize(uEye - vWorld);
  vec3 n = normalize(vNrm);
  if (dot(n, v) < 0.0) n = -n; // two-sided: culling is off and the inside of open surfaces is visible
  float diff = max(dot(n, uLightDir), 0.0);
  vec3 h = normalize(uLightDir + v);
  float spec = pow(max(dot(n, h), 0.0), 48.0) * uSpecular;
  vec3 col = uColor * (uAmbient + uDiffuse * diff) + vec3(spec);
  outColor = vec4(min(col, vec3(1.0)), uOpacity);
}`;

// Orthoslices: a unit quad placed on one of the three texture-space planes of an object's box and lit like
// the surfaces (its normal is the plane axis in world space).
const SLICE_VERT = `#version 300 es
in vec2 aUv;
uniform int uAxis;
uniform float uPos;
uniform vec3 uTexScale;
uniform mat4 uViewProj, uModel;
out vec3 vTc;
out vec3 vWorld;
void main() {
  vec3 tc = uAxis == 0 ? vec3(uPos, aUv.x, aUv.y) : (uAxis == 1 ? vec3(aUv.x, uPos, aUv.y) : vec3(aUv.x, aUv.y, uPos));
  vec4 w = uModel * vec4((tc - 0.5) / uTexScale, 1.0);
  vTc = tc;
  vWorld = w.xyz;
  gl_Position = uViewProj * w;
}`;

function buildSliceFrag(n: number): string {
  const ch = (i: number, s: string) => s.replace(/\$/g, String(i));
  const decl = Array.from({ length: n }, (_, i) => ch(i, "uniform sampler3D uTex$; uniform sampler2D uLut$; uniform vec4 uCh$;")).join("\n");
  const body = Array.from({ length: n }, (_, i) => ch(i, `  if (uCh$.x > 0.5) { float v = texture(uTex$, vTc).r; v = max(0.0, (v - uCh$.y) / max(1.0 - uCh$.y, 1e-3)); col += (uUseFlat == 1 ? uFlat * v : texture(uLut$, vec2(v, 0.5)).rgb) * uCh$.w; }`)).join("\n");
  return `#version 300 es
precision highp float;
precision highp sampler3D;
in vec3 vTc;
in vec3 vWorld;
out vec4 outColor;
uniform vec3 uNormal;
uniform vec3 uLightDir;
uniform float uAmbient, uDiffuse;
uniform float uOpacity;
uniform int uClipOn;
uniform vec3 uClipMin, uClipMax;
uniform int uUseFlat;
uniform vec3 uFlat;
${decl}
void main() {
  if (uClipOn == 1 && (any(lessThan(vWorld, uClipMin)) || any(greaterThan(vWorld, uClipMax)))) discard;
  vec3 col = vec3(0.0);
${body}
  float shade = min(1.0, uAmbient + uDiffuse * abs(dot(normalize(uNormal), uLightDir)));
  outColor = vec4(min(col, vec3(1.0)) * shade, uOpacity);
}`;
}

// Landmarks: depth-tested point sprites shaded as small spheres (a white ring marks the selected one).
const POINT_VERT = `#version 300 es
in vec3 aPos;
uniform mat4 uViewProj;
uniform float uSize;
void main() { gl_Position = uViewProj * vec4(aPos, 1.0); gl_PointSize = uSize; }`;
const POINT_FRAG = `#version 300 es
precision mediump float;
uniform vec3 uColor;
uniform int uRing;
out vec4 outColor;
void main() {
  vec2 c = gl_PointCoord * 2.0 - 1.0;
  c.y = -c.y;
  float r2 = dot(c, c);
  if (r2 > 1.0) discard;
  vec3 n = vec3(c, sqrt(1.0 - r2));
  vec3 l = normalize(vec3(0.35, 0.5, 0.8));
  float diff = max(dot(n, l), 0.0);
  float spec = pow(max(dot(n, normalize(l + vec3(0.0, 0.0, 1.0))), 0.0), 32.0) * 0.4;
  vec3 col = uColor * (0.3 + 0.7 * diff) + vec3(spec);
  if (uRing == 1 && r2 > 0.55) col = vec3(1.0);
  outColor = vec4(min(col, vec3(1.0)), 1.0);
}`;

interface GlMesh {
  vao: WebGLVertexArrayObject;
  vbo: WebGLBuffer;
  nbo: WebGLBuffer;
  ibo: WebGLBuffer;
  /** line-list index buffer, built on first wireframe draw */
  edges: WebGLBuffer | null;
  nFaces: number;
  faces: Uint32Array;
  extent: [number, number, number];
}

interface GpuVolume {
  data: VolumeData;
  textures: WebGLTexture[];
  /** half extents of the local box (longest axis = 1) */
  half: Vec3;
}

/** GPU resources of one scene object (the current image is uploaded under the key "primary"). */
interface GpuObject {
  vol: GpuVolume | null;
  meshes: (GlMesh | null)[];
  luts: WebGLTexture[];
  lutKeys: string[];
  lutColors: Vec3[];
  /** The tables object the LUT textures were built from; a refetched `luts` (new object) re-uploads them. */
  lutSource: LutTables | null;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type);
  if (!sh) throw new Error("createShader failed");
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh) || "";
    gl.deleteShader(sh);
    throw new Error("Shader compile error: " + log);
  }
  return sh;
}

function link(gl: WebGL2RenderingContext, vs: string, fs: string, attribs?: Record<string, number>): WebGLProgram {
  const p = gl.createProgram();
  if (!p) throw new Error("createProgram failed");
  const a = compile(gl, gl.VERTEX_SHADER, vs);
  const b = compile(gl, gl.FRAGMENT_SHADER, fs);
  gl.attachShader(p, a);
  gl.attachShader(p, b);
  if (attribs) for (const [name, loc] of Object.entries(attribs)) gl.bindAttribLocation(p, loc, name);
  gl.linkProgram(p);
  gl.deleteShader(a);
  gl.deleteShader(b);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(p) || "";
    gl.deleteProgram(p);
    throw new Error("Program link error: " + log);
  }
  return p;
}

// ------------------------------------------------------------------ renderer
const DEFAULT_CS: ChannelSettings = { visible: true, threshold: 0.1, isoLevel: 0.4, opacity: 0.3, brightness: 1, meshOpacity: 1 };
const DEFAULT_LIGHTS: Lights = { ambient: 0.22, diffuse: 0.78, specular: 0.3, direction: [0.35, 0.5, 0.8], headlight: true };
/** Resource key of the current image. */
export const PRIMARY = "primary";

/** One object of a frame: which uploaded resources (`key`) to draw and how. */
export interface ObjectRender {
  key: string;
  display: DisplayMode;
  /** volume display: mip | alpha | iso ("mesh" counts as iso; surfaces are the `surface` display) */
  mode: VolumeMode;
  channels: ChannelSettings[];
  /** local box → world; null = identity (the primary image without a transform) */
  model?: Mat4 | null;
  /** flat colour instead of the LUTs */
  color?: Vec3 | null;
  /** 1 − transparency */
  opacity?: number;
  /** orthoslices (texture-space positions 0..1 along pixel x, y, z; YZ / XZ / XY planes) */
  slices?: { pos: Vec3; show: [boolean, boolean, boolean] } | null;
  wireframe?: boolean;
  visible?: boolean;
}

export interface LandmarkRender {
  world: Vec3;
  color: Vec3;
  selected?: boolean;
}

export interface RenderOptions {
  mode: VolumeMode;
  channels: ChannelSettings[];
  camera: Camera;
  /** 0..1 quality (step density); use ~0.5 while interacting */
  quality: number;
  background: [number, number, number];
  showBox: boolean;
  /** mesh mode: draw the triangle edges on top of the shaded surface */
  wireframe?: boolean;
  projection?: Projection;
  lights?: Lights;
  /** clipping box as fractions of the primary box along the pixel axes */
  clip?: ClipBox;
  /** model / colour / opacity / orthoslices / display of the primary image (defaults follow `mode`) */
  primary?: Partial<Omit<ObjectRender, "key" | "channels">>;
  /** further objects (resources uploaded under their keys with setObject*) */
  objects?: ObjectRender[];
  landmarks?: LandmarkRender[];
  /** landmark diameter in CSS pixels */
  landmarkSize?: number;
}

type Basis = { eye: Vec3; right: Vec3; up: Vec3; fwd: Vec3 };

interface ClipWorld {
  on: boolean;
  min: Vec3;
  max: Vec3;
}

/** Per-frame state shared by the draw calls. */
interface Frame {
  vp: Mat4;
  basis: Basis;
  cam: Camera;
  aspect: number;
  ortho: boolean;
  lights: Lights;
  /** unit light direction in world space */
  light: Vec3;
  clip: ClipWorld;
  quality: number;
}

/** World light direction: camera-relative for a head light (x right, y up, z towards the viewer), else world axes. */
function lightDirection(l: Lights, b: Basis): Vec3 {
  const d = l.direction;
  if (!l.headlight) return vnorm(d);
  return vnorm([b.right[0] * d[0] + b.up[0] * d[1] - b.fwd[0] * d[2], b.right[1] * d[0] + b.up[1] * d[1] - b.fwd[1] * d[2], b.right[2] * d[0] + b.up[2] * d[1] - b.fwd[2] * d[2]]);
}

export class VolumeRenderer {
  readonly gl: WebGL2RenderingContext;
  private canvas: HTMLCanvasElement;
  /** ray-cast programs by channel count */
  private rayPrograms = new Map<number, WebGLProgram>();
  private slicePrograms = new Map<number, WebGLProgram>();
  private boxProgram: WebGLProgram;
  private meshProgram: WebGLProgram;
  private pointProgram: WebGLProgram;
  private objects = new Map<string, GpuObject>();
  private quadVao: WebGLVertexArrayObject;
  private quadBuf: WebGLBuffer;
  private boxVao: WebGLVertexArrayObject;
  private boxBuf: WebGLBuffer;
  private boxIdx: WebGLBuffer;
  private sliceVao: WebGLVertexArrayObject;
  private sliceBuf: WebGLBuffer;
  private pointVao: WebGLVertexArrayObject;
  private pointBuf: WebGLBuffer;
  /** view-projection of the last frame (label placement) */
  private lastVp: Mat4 | null = null;
  private disposed = false;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true, powerPreference: "high-performance" });
    if (!gl) throw new Error("WebGL2 is not available in this browser");
    this.gl = gl;
    this.boxProgram = link(gl, BOX_VERT, BOX_FRAG);
    try {
      this.meshProgram = link(gl, MESH_VERT, MESH_FRAG);
      this.pointProgram = link(gl, POINT_VERT, POINT_FRAG);
    } catch (e) {
      gl.deleteProgram(this.boxProgram);
      throw e;
    }
    // full-screen triangle
    const quadVao = gl.createVertexArray();
    const quadBuf = gl.createBuffer();
    if (!quadVao || !quadBuf) throw new Error("WebGL allocation failed");
    this.quadVao = quadVao;
    this.quadBuf = quadBuf;
    gl.bindVertexArray(quadVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    // box edges
    const boxVao = gl.createVertexArray();
    const boxBuf = gl.createBuffer();
    const boxIdx = gl.createBuffer();
    if (!boxVao || !boxBuf || !boxIdx) throw new Error("WebGL allocation failed");
    this.boxVao = boxVao;
    this.boxBuf = boxBuf;
    this.boxIdx = boxIdx;
    gl.bindVertexArray(boxVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, boxBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(this.boxProgram, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, boxIdx);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array([0, 1, 1, 2, 2, 3, 3, 0, 4, 5, 5, 6, 6, 7, 7, 4, 0, 4, 1, 5, 2, 6, 3, 7]), gl.STATIC_DRAW);
    // orthoslice quad (unit square; aUv is bound to location 0 in every slice program)
    const sliceVao = gl.createVertexArray();
    const sliceBuf = gl.createBuffer();
    if (!sliceVao || !sliceBuf) throw new Error("WebGL allocation failed");
    this.sliceVao = sliceVao;
    this.sliceBuf = sliceBuf;
    gl.bindVertexArray(sliceVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, sliceBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    // landmark point (one position, rewritten per landmark)
    const pointVao = gl.createVertexArray();
    const pointBuf = gl.createBuffer();
    if (!pointVao || !pointBuf) throw new Error("WebGL allocation failed");
    this.pointVao = pointVao;
    this.pointBuf = pointBuf;
    gl.bindVertexArray(pointVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, pointBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(3), gl.DYNAMIC_DRAW);
    const ploc = gl.getAttribLocation(this.pointProgram, "aPos");
    gl.enableVertexAttribArray(ploc);
    gl.vertexAttribPointer(ploc, 3, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  }

  private obj(key: string): GpuObject {
    let o = this.objects.get(key);
    if (!o) {
      o = { vol: null, meshes: [], luts: [], lutKeys: [], lutColors: [], lutSource: null };
      this.objects.set(key, o);
    }
    return o;
  }

  get hasVolume(): boolean {
    return !!this.objects.get(PRIMARY)?.vol;
  }

  /** Half extents of the box in world units (longest axis = 1). */
  get halfExtents(): Vec3 {
    return this.objectHalf(PRIMARY) || [1, 1, 1];
  }

  /** Half extents of an object's local box (from its volume, else its first mesh); null when nothing is uploaded. */
  objectHalf(key: string): Vec3 | null {
    const o = this.objects.get(key);
    if (!o) return null;
    if (o.vol) return o.vol.half;
    const m = o.meshes.find((x) => !!x);
    return m ? boxHalf(m.extent) : null;
  }

  objectVolume(key: string): VolumeData | null {
    return this.objects.get(key)?.vol?.data || null;
  }

  hasObject(key: string): boolean {
    return this.objects.has(key);
  }

  setVolume(vol: VolumeData): void {
    this.setObjectVolume(PRIMARY, vol);
  }

  /** Upload the volume of an object (null drops it). */
  setObjectVolume(key: string, vol: VolumeData | null): void {
    const gl = this.gl;
    const o = this.obj(key);
    // Drop the previous volume first so that a failure below leaves nothing stale on screen.
    if (o.vol) {
      for (const t of o.vol.textures) gl.deleteTexture(t);
      o.vol = null;
    }
    if (!vol || this.disposed) return;
    const n = vol.channels.length;
    // One sampler3D + one LUT sampler2D per channel; WebGL2 only guarantees 16 fragment samplers
    // (ANGLE Metal / D3D11 expose exactly 16), so refuse before touching any GL state.
    const maxCh = Math.floor(((gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) as number) || 16) / 2);
    if (n > maxCh) throw new Error(`The 3-D view can show at most ${maxCh} channels on this GPU (this image has ${n}). Hide some channels or split the image.`);
    const [Z, Y, X] = vol.shape;
    const textures: WebGLTexture[] = [];
    try {
      for (const data of vol.channels) {
        const tex = gl.createTexture();
        if (!tex) throw new Error("createTexture failed");
        textures.push(tex);
        gl.bindTexture(gl.TEXTURE_3D, tex);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
        gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, X, Y, Z, 0, gl.RED, gl.UNSIGNED_BYTE, data);
      }
      gl.bindTexture(gl.TEXTURE_3D, null);
      // link now so that a shader problem surfaces here (reported by the caller) rather than mid-frame
      this.rayProgram(n);
      this.sliceProgram(n);
    } catch (e) {
      for (const t of textures) gl.deleteTexture(t);
      throw e;
    }
    o.vol = { data: vol, textures, half: boxHalf([X, Y, Z * vol.spacing]) };
  }

  /**
   * Upload (or refresh) the 256×1 LUT textures for each channel.  `inverted[i]` reverses the
   * table (colour only), matching the 2-D "Invert LUT" semantics; the voxel densities are untouched.
   */
  setLuts(luts: LutTables, names: string[], inverted: boolean[] = []): void {
    this.setObjectLuts(PRIMARY, luts, names, inverted);
  }

  setObjectLuts(key: string, luts: LutTables, names: string[], inverted: boolean[] = []): void {
    const gl = this.gl;
    const o = this.obj(key);
    // a new tables object means a table may have changed under an unchanged name (Edit LUT, 8-bit Color)
    const fresh = o.lutSource !== luts;
    o.lutSource = luts;
    for (let i = 0; i < names.length; i++) {
      const inv = !!inverted[i];
      const lkey = `${names[i]}|${inv ? 1 : 0}`;
      if (!fresh && o.luts[i] && o.lutKeys[i] === lkey) continue;
      const src = luts[names[i]] || luts["Grays"];
      const px = new Uint8Array(256 * 3);
      for (let k = 0; k < 256; k++) {
        const j = inv ? 255 - k : k;
        const row = src ? src[j] : [j, j, j];
        px[k * 3] = row[0];
        px[k * 3 + 1] = row[1];
        px[k * 3 + 2] = row[2];
      }
      let tex = o.luts[i];
      if (!tex) {
        const t = gl.createTexture();
        if (!t) throw new Error("createTexture failed");
        tex = t;
        o.luts[i] = tex;
      }
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, 256, 1, 0, gl.RGB, gl.UNSIGNED_BYTE, px);
      o.lutKeys[i] = lkey;
      // surface colour for mesh mode: the same 85 % table entry the iso shader uses
      const k = 217 * 3;
      o.lutColors[i] = [px[k] / 255, px[k + 1] / 255, px[k + 2] / 255];
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  /** Number of meshes currently uploaded for the current image (channels with a surface). */
  get meshCount(): number {
    return (this.objects.get(PRIMARY)?.meshes || []).filter((m) => !!m).length;
  }

  private deleteMesh(m: GlMesh | null): void {
    if (!m) return;
    const gl = this.gl;
    gl.deleteBuffer(m.vbo);
    gl.deleteBuffer(m.nbo);
    gl.deleteBuffer(m.ibo);
    if (m.edges) gl.deleteBuffer(m.edges);
    gl.deleteVertexArray(m.vao);
  }

  /**
   * Upload the iso-surfaces for mesh mode; `list[i]` belongs to channel `i` of the volume (null = none).
   * Previous buffers are released first, so calling this with `[]` frees everything.
   */
  setMeshes(list: (MeshData | null)[]): void {
    this.setObjectMeshes(PRIMARY, list);
  }

  setObjectMeshes(key: string, list: (MeshData | null)[]): void {
    const gl = this.gl;
    const o = this.obj(key);
    for (const m of o.meshes) this.deleteMesh(m);
    o.meshes = [];
    if (this.disposed) return;
    const posLoc = gl.getAttribLocation(this.meshProgram, "aPos");
    const nrmLoc = gl.getAttribLocation(this.meshProgram, "aNrm");
    for (const md of list) {
      if (!md || md.nFaces === 0 || md.nVertices === 0) {
        o.meshes.push(null);
        continue;
      }
      const vao = gl.createVertexArray();
      const vbo = gl.createBuffer();
      const nbo = gl.createBuffer();
      const ibo = gl.createBuffer();
      if (!vao || !vbo || !nbo || !ibo) {
        if (vao) gl.deleteVertexArray(vao);
        if (vbo) gl.deleteBuffer(vbo);
        if (nbo) gl.deleteBuffer(nbo);
        if (ibo) gl.deleteBuffer(ibo);
        throw new Error("WebGL allocation failed");
      }
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, md.vertices, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ARRAY_BUFFER, nbo);
      gl.bufferData(gl.ARRAY_BUFFER, md.normals, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(nrmLoc);
      gl.vertexAttribPointer(nrmLoc, 3, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, md.faces, gl.STATIC_DRAW);
      gl.bindVertexArray(null);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      o.meshes.push({ vao, vbo, nbo, ibo, edges: null, nFaces: md.nFaces, faces: md.faces, extent: md.extent });
    }
  }

  /** Release every resource of an object (the primary included). */
  removeObject(key: string): void {
    const o = this.objects.get(key);
    if (!o) return;
    const gl = this.gl;
    if (o.vol) for (const t of o.vol.textures) gl.deleteTexture(t);
    for (const t of o.luts) if (t) gl.deleteTexture(t);
    for (const m of o.meshes) this.deleteMesh(m);
    this.objects.delete(key);
  }

  private rayProgram(n: number): WebGLProgram {
    let p = this.rayPrograms.get(n);
    if (!p) {
      p = link(this.gl, VERT, buildFrag(n), { aPos: 0 });
      this.rayPrograms.set(n, p);
    }
    return p;
  }

  private sliceProgram(n: number): WebGLProgram {
    let p = this.slicePrograms.get(n);
    if (!p) {
      p = link(this.gl, SLICE_VERT, buildSliceFrag(n), { aUv: 0 });
      this.slicePrograms.set(n, p);
    }
    return p;
  }

  private ensureEdges(m: GlMesh): WebGLBuffer {
    if (m.edges) return m.edges;
    const gl = this.gl;
    const f = m.faces;
    const e = new Uint32Array(m.nFaces * 6);
    for (let i = 0, j = 0; i < m.nFaces; i++, j += 6) {
      const a = f[i * 3], b = f[i * 3 + 1], c = f[i * 3 + 2];
      e[j] = a; e[j + 1] = b; e[j + 2] = b; e[j + 3] = c; e[j + 4] = c; e[j + 5] = a;
    }
    const buf = gl.createBuffer();
    if (!buf) throw new Error("WebGL allocation failed");
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buf);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, e, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    m.edges = buf;
    return buf;
  }

  /** Uniforms every object program shares: camera ray, projection, clip box, lights, colour override, opacity, matrices. */
  private setCommon(prog: WebGLProgram, f: Frame, model: Mat4, color: Vec3 | null, opacity: number): void {
    const gl = this.gl;
    const u = (name: string) => gl.getUniformLocation(prog, name);
    gl.uniform3fv(u("uEye"), f.basis.eye);
    gl.uniform3fv(u("uRight"), f.basis.right);
    gl.uniform3fv(u("uUp"), f.basis.up);
    gl.uniform3fv(u("uFwd"), f.basis.fwd);
    gl.uniform1f(u("uTanHalf"), Math.tan(FOV / 2));
    gl.uniform1f(u("uAspect"), f.aspect);
    gl.uniform1i(u("uOrtho"), f.ortho ? 1 : 0);
    gl.uniform1f(u("uOrthoHalf"), f.cam.dist * Math.tan(FOV / 2));
    gl.uniform1i(u("uClipOn"), f.clip.on ? 1 : 0);
    gl.uniform3fv(u("uClipMin"), f.clip.min);
    gl.uniform3fv(u("uClipMax"), f.clip.max);
    gl.uniform1f(u("uAmbient"), f.lights.ambient);
    gl.uniform1f(u("uDiffuse"), f.lights.diffuse);
    gl.uniform1f(u("uSpecular"), f.lights.specular);
    gl.uniform3fv(u("uLightDir"), f.light);
    gl.uniform1i(u("uUseFlat"), color ? 1 : 0);
    gl.uniform3fv(u("uFlat"), color || [1, 1, 1]);
    gl.uniform1f(u("uOpacity"), Math.min(1, Math.max(0, opacity)));
    gl.uniformMatrix4fv(u("uModel"), false, model);
    gl.uniformMatrix4fv(u("uViewProj"), false, f.vp);
  }

  /** One full-screen ray-cast pass for a volume object (blend / depth state is set by the caller). */
  private drawVolume(o: ObjectRender, g: GpuObject, f: Frame): void {
    const gl = this.gl;
    const vol = g.vol;
    if (!vol) return;
    const n = vol.data.channels.length;
    const prog = this.rayProgram(n);
    gl.useProgram(prog);
    const u = (name: string) => gl.getUniformLocation(prog, name);
    const model = o.model || mat4Identity();
    this.setCommon(prog, f, model, o.color ?? null, o.opacity ?? 1);
    gl.uniformMatrix4fv(u("uInvModel"), false, mat4Invert(model));
    gl.uniform3fv(u("uHalf"), vol.half);
    gl.uniform3f(u("uTexScale"), 0.5 / vol.half[0], -0.5 / vol.half[1], -0.5 / vol.half[2]);
    const [Z, Y, X] = vol.data.shape;
    gl.uniform3f(u("uTexel"), 1 / X, 1 / Y, 1 / Z);
    const maxDim = Math.max(X, Y, Z * vol.data.spacing);
    const voxel = 2 / maxDim; // local size of one voxel along the longest axis (half extent 1)
    const q = Math.min(1, Math.max(0.2, f.quality));
    const stepLocal = voxel / (1.6 * q);
    gl.uniform1f(u("uStep"), stepLocal * mat4ScaleFactor(model)); // the shader marches in world units
    gl.uniform1f(u("uAlphaScale"), stepLocal * 128);
    gl.uniform1i(u("uMode"), o.mode === "mip" ? 0 : o.mode === "alpha" ? 1 : 2);
    for (let i = 0; i < n; i++) {
      const cs = o.channels[i] || DEFAULT_CS;
      gl.activeTexture(gl.TEXTURE0 + i * 2);
      gl.bindTexture(gl.TEXTURE_3D, vol.textures[i]);
      gl.uniform1i(u(`uTex${i}`), i * 2);
      gl.activeTexture(gl.TEXTURE0 + i * 2 + 1);
      gl.bindTexture(gl.TEXTURE_2D, g.luts[i] || null);
      gl.uniform1i(u(`uLut${i}`), i * 2 + 1);
      gl.uniform4f(u(`uCh${i}`), cs.visible ? 1 : 0, o.mode === "iso" ? cs.isoLevel : cs.threshold, cs.opacity, cs.brightness);
    }
    gl.bindVertexArray(this.quadVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
  }

  /** The surfaces of an object; `translucentPass` selects the opaque (depth-written) or blended ones. */
  private drawMeshes(o: ObjectRender, g: GpuObject, f: Frame, translucentPass: boolean): void {
    const gl = this.gl;
    if (!g.meshes.some((m) => !!m)) return;
    const prog = this.meshProgram;
    gl.useProgram(prog);
    const u = (name: string) => gl.getUniformLocation(prog, name);
    const model = o.model || mat4Identity();
    this.setCommon(prog, f, model, null, 1);
    gl.disable(gl.CULL_FACE);
    const objOpacity = o.opacity ?? 1;
    for (let i = 0; i < g.meshes.length; i++) {
      const m = g.meshes[i];
      const cs = o.channels[i] || DEFAULT_CS;
      if (!m || !cs.visible) continue;
      const alpha = Math.min(1, Math.max(0.02, cs.meshOpacity * objOpacity));
      if ((alpha < 0.995) !== translucentPass) continue;
      const [W, H, D] = m.extent;
      // the volume's box when one is loaded (the surface then fits the drawn box exactly, like the picking frame)
      const half = g.vol ? g.vol.half : boxHalf(m.extent);
      gl.uniform3f(u("uScale"), (2 * half[0]) / (W || 1), (-2 * half[1]) / (H || 1), (-2 * half[2]) / (D || 1));
      gl.uniform3f(u("uOffset"), -half[0], half[1], half[2]);
      const c = o.color || g.lutColors[i] || [0.8, 0.8, 0.8];
      const gain = cs.brightness;
      gl.uniform3f(u("uColor"), Math.min(1, c[0] * gain), Math.min(1, c[1] * gain), Math.min(1, c[2] * gain));
      gl.uniform1f(u("uOpacity"), alpha);
      gl.uniform1i(u("uWire"), 0);
      gl.bindVertexArray(m.vao);
      if (o.wireframe) {
        gl.enable(gl.POLYGON_OFFSET_FILL);
        gl.polygonOffset(1, 1);
      }
      gl.drawElements(gl.TRIANGLES, m.nFaces * 3, gl.UNSIGNED_INT, 0);
      if (o.wireframe) {
        gl.disable(gl.POLYGON_OFFSET_FILL);
        const edges = this.ensureEdges(m);
        gl.uniform3f(u("uColor"), Math.min(1, 0.35 + c[0] * gain * 0.8), Math.min(1, 0.35 + c[1] * gain * 0.8), Math.min(1, 0.35 + c[2] * gain * 0.8));
        gl.uniform1i(u("uWire"), 1);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, edges);
        gl.drawElements(gl.LINES, m.nFaces * 6, gl.UNSIGNED_INT, 0);
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.ibo); // restore the VAO's element buffer binding
      }
      gl.bindVertexArray(null);
    }
  }

  /** The orthoslices of an object (needs its volume textures). */
  private drawSlices(o: ObjectRender, g: GpuObject, f: Frame, translucentPass: boolean): void {
    const gl = this.gl;
    const vol = g.vol;
    const sl = o.slices;
    if (!vol || !sl || !sl.show.some(Boolean)) return;
    const alpha = o.opacity ?? 1;
    if ((alpha < 0.995) !== translucentPass) return;
    const n = vol.data.channels.length;
    const prog = this.sliceProgram(n);
    gl.useProgram(prog);
    const u = (name: string) => gl.getUniformLocation(prog, name);
    const model = o.model || mat4Identity();
    this.setCommon(prog, f, model, o.color ?? null, alpha);
    gl.uniform3f(u("uTexScale"), 0.5 / vol.half[0], -0.5 / vol.half[1], -0.5 / vol.half[2]);
    for (let i = 0; i < n; i++) {
      const cs = o.channels[i] || DEFAULT_CS;
      gl.activeTexture(gl.TEXTURE0 + i * 2);
      gl.bindTexture(gl.TEXTURE_3D, vol.textures[i]);
      gl.uniform1i(u(`uTex${i}`), i * 2);
      gl.activeTexture(gl.TEXTURE0 + i * 2 + 1);
      gl.bindTexture(gl.TEXTURE_2D, g.luts[i] || null);
      gl.uniform1i(u(`uLut${i}`), i * 2 + 1);
      gl.uniform4f(u(`uCh${i}`), cs.visible ? 1 : 0, cs.threshold, cs.opacity, cs.brightness);
    }
    gl.disable(gl.CULL_FACE);
    gl.bindVertexArray(this.sliceVao);
    for (let axis = 0; axis < 3; axis++) {
      if (!sl.show[axis]) continue;
      gl.uniform1i(u("uAxis"), axis);
      gl.uniform1f(u("uPos"), Math.min(0.999, Math.max(0.001, sl.pos[axis])));
      const e: Vec3 = axis === 0 ? [1, 0, 0] : axis === 1 ? [0, 1, 0] : [0, 0, 1];
      gl.uniform3fv(u("uNormal"), vnorm(transformDir(model, e)));
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.bindVertexArray(null);
  }

  /** Landmarks as depth-tested shaded points (`size` in CSS pixels). */
  private drawLandmarks(list: LandmarkRender[], size: number, f: Frame): void {
    const gl = this.gl;
    const prog = this.pointProgram;
    gl.useProgram(prog);
    const u = (name: string) => gl.getUniformLocation(prog, name);
    gl.uniformMatrix4fv(u("uViewProj"), false, f.vp);
    const dpr = this.canvas.width / Math.max(1, this.canvas.clientWidth || this.canvas.width);
    gl.bindVertexArray(this.pointVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.pointBuf);
    for (const lm of list) {
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(lm.world), gl.DYNAMIC_DRAW);
      gl.uniform1f(u("uSize"), size * dpr * (lm.selected ? 1.35 : 1));
      gl.uniform3fv(u("uColor"), lm.color);
      gl.uniform1i(u("uRing"), lm.selected ? 1 : 0);
      gl.drawArrays(gl.POINTS, 0, 1);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindVertexArray(null);
  }

  /** The clipping box (fractions of the primary box along the pixel axes) as a world-space AABB. */
  private clipWorld(clip: ClipBox | undefined): ClipWorld {
    if (!clip || !isClipActive(clip)) return { on: false, min: [-1e9, -1e9, -1e9], max: [1e9, 1e9, 1e9] };
    const h = this.halfExtents;
    const xs = [-h[0] + clip.min[0] * 2 * h[0], -h[0] + clip.max[0] * 2 * h[0]];
    const ys = [h[1] - clip.max[1] * 2 * h[1], h[1] - clip.min[1] * 2 * h[1]]; // pixel y runs down, world y up
    const zs = [h[2] - clip.max[2] * 2 * h[2], h[2] - clip.min[2] * 2 * h[2]]; // slice 0 sits at +z
    return { on: true, min: [Math.min(...xs), Math.min(...ys), Math.min(...zs)], max: [Math.max(...xs), Math.max(...ys), Math.max(...zs)] };
  }

  /** Resize the drawing buffer to the canvas' CSS size × dpr. Returns true if it changed. */
  resize(dpr: number): boolean {
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
      return true;
    }
    return false;
  }

  render(opts: RenderOptions): void {
    const gl = this.gl;
    if (this.disposed) return;
    const w = this.canvas.width;
    const h = this.canvas.height;
    gl.viewport(0, 0, w, h);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.BLEND);
    const [br, bgc, bb] = opts.background;
    gl.depthMask(true);
    gl.clearColor(br, bgc, bb, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const aspect = w / h;
    const projection: Projection = opts.projection || "perspective";
    const { vp, basis } = cameraMatrices(opts.camera, aspect, projection);
    this.lastVp = vp;
    const lights = opts.lights || DEFAULT_LIGHTS;
    const f: Frame = { vp, basis, cam: opts.camera, aspect, ortho: projection === "orthographic", lights, light: lightDirection(lights, basis), clip: this.clipWorld(opts.clip), quality: opts.quality };
    // the primary image is object 0; its display follows the mode unless overridden
    const p = opts.primary || {};
    const list: ObjectRender[] = [
      { key: PRIMARY, display: p.display ?? (opts.mode === "mesh" ? "surface" : "volume"), mode: opts.mode === "mesh" ? "iso" : opts.mode, channels: opts.channels, model: p.model ?? null, color: p.color ?? null, opacity: p.opacity ?? 1, slices: p.slices ?? null, wireframe: p.wireframe ?? opts.wireframe, visible: p.visible ?? true },
      ...(opts.objects || []),
    ];
    const items: { o: ObjectRender; g: GpuObject }[] = [];
    for (const o of list) {
      const g = this.objects.get(o.key);
      if (g && o.visible !== false) items.push({ o, g });
    }
    const premultiplied = () => gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    // 1. iso volumes: opaque hits that write depth
    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(true);
    gl.enable(gl.BLEND);
    premultiplied();
    for (const { o, g } of items) if (o.display === "volume" && o.mode === "iso" && g.vol) this.drawVolume(o, g, f);
    gl.disable(gl.BLEND);
    // 2. opaque surfaces and orthoslices (depth-tested against the iso hits and each other)
    for (const { o, g } of items) {
      if (o.display === "surface") this.drawMeshes(o, g, f, false);
      if (o.slices) this.drawSlices(o, g, f, false);
    }
    // 3. landmarks
    if (opts.landmarks && opts.landmarks.length) this.drawLandmarks(opts.landmarks, opts.landmarkSize ?? 9, f);
    // 4. translucent surfaces and slices
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.depthMask(false);
    for (const { o, g } of items) {
      if (o.display === "surface") this.drawMeshes(o, g, f, true);
      if (o.slices) this.drawSlices(o, g, f, true);
    }
    // 5. MIP / alpha volumes blended on top of everything (they carry no depth)
    gl.disable(gl.DEPTH_TEST);
    premultiplied();
    for (const { o, g } of items) if (o.display === "volume" && o.mode !== "iso" && g.vol) this.drawVolume(o, g, f);
    gl.depthMask(true);
    gl.disable(gl.BLEND);
    // 6. the box of the primary image
    if (opts.showBox) {
      const bp = this.boxProgram;
      gl.useProgram(bp);
      gl.uniformMatrix4fv(gl.getUniformLocation(bp, "uViewProj"), false, vp);
      gl.uniformMatrix4fv(gl.getUniformLocation(bp, "uModel"), false, p.model || mat4Identity());
      gl.uniform3fv(gl.getUniformLocation(bp, "uHalf"), this.halfExtents);
      const light = br + bgc + bb > 1.5;
      gl.uniform4f(gl.getUniformLocation(bp, "uColor"), light ? 0.25 : 0.75, light ? 0.28 : 0.78, light ? 0.35 : 0.85, 0.45);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.bindVertexArray(this.boxVao);
      gl.drawElements(gl.LINES, 24, gl.UNSIGNED_SHORT, 0);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    }
    gl.flush();
  }

  /** CSS-pixel position (and NDC depth) of a world point in the last rendered frame; null when behind the camera. */
  project(world: Vec3): [number, number, number] | null {
    const m = this.lastVp;
    if (!m) return null;
    const x = m[0] * world[0] + m[4] * world[1] + m[8] * world[2] + m[12];
    const y = m[1] * world[0] + m[5] * world[1] + m[9] * world[2] + m[13];
    const z = m[2] * world[0] + m[6] * world[1] + m[10] * world[2] + m[14];
    const ww = m[3] * world[0] + m[7] * world[1] + m[11] * world[2] + m[15];
    if (ww <= 1e-9) return null;
    return [(x / ww * 0.5 + 0.5) * this.canvas.clientWidth, (1 - (y / ww * 0.5 + 0.5)) * this.canvas.clientHeight, z / ww];
  }

  /**
   * Render off-screen at `width` × `height` CSS-independent pixels (supersampled `supersample`× and
   * downscaled with high-quality filtering) and return a PNG blob; the on-screen size is restored afterwards.
   */
  async renderToBlob(opts: RenderOptions, width: number, height: number, supersample = 1): Promise<Blob> {
    const gl = this.gl;
    const canvas = this.canvas;
    const maxSize = Math.min(8192, (gl.getParameter(gl.MAX_RENDERBUFFER_SIZE) as number) || 4096);
    const ss = Math.max(1, Math.min(4, supersample));
    width = Math.max(1, Math.round(width));
    height = Math.max(1, Math.round(height));
    let rw = Math.round(width * ss);
    let rh = Math.round(height * ss);
    if (rw > maxSize || rh > maxSize) {
      const k = maxSize / Math.max(rw, rh);
      rw = Math.max(1, Math.floor(rw * k));
      rh = Math.max(1, Math.floor(rh * k));
    }
    const ow = canvas.width;
    const oh = canvas.height;
    canvas.width = rw;
    canvas.height = rh;
    try {
      this.render(opts);
      let src: HTMLCanvasElement = canvas;
      if (rw !== width || rh !== height) {
        const out = document.createElement("canvas");
        out.width = width;
        out.height = height;
        const ctx = out.getContext("2d");
        if (!ctx) throw new Error("2-D canvas unavailable");
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "high";
        ctx.drawImage(canvas, 0, 0, width, height);
        src = out;
      }
      return await new Promise<Blob>((resolve, reject) => src.toBlob((b) => (b ? resolve(b) : reject(new Error("PNG encoding failed"))), "image/png"));
    } finally {
      canvas.width = ow;
      canvas.height = oh;
      this.render(opts);
    }
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    const gl = this.gl;
    for (const key of Array.from(this.objects.keys())) this.removeObject(key);
    for (const p of this.rayPrograms.values()) gl.deleteProgram(p);
    for (const p of this.slicePrograms.values()) gl.deleteProgram(p);
    this.rayPrograms.clear();
    this.slicePrograms.clear();
    gl.deleteProgram(this.boxProgram);
    gl.deleteProgram(this.meshProgram);
    gl.deleteProgram(this.pointProgram);
    gl.deleteBuffer(this.quadBuf);
    gl.deleteBuffer(this.boxBuf);
    gl.deleteBuffer(this.boxIdx);
    gl.deleteBuffer(this.sliceBuf);
    gl.deleteBuffer(this.pointBuf);
    gl.deleteVertexArray(this.quadVao);
    gl.deleteVertexArray(this.boxVao);
    gl.deleteVertexArray(this.sliceVao);
    gl.deleteVertexArray(this.pointVao);
    // A canvas owns exactly one WebGL context object.  React StrictMode (dev) and effect re-runs
    // dispose and immediately re-create a renderer on the SAME <canvas>; losing the context here
    // would hand the next renderer a permanently lost context (restoreContext() is asynchronous and
    // in Chromium only works after a webglcontextlost listener called preventDefault()).  Release it
    // eagerly only once the canvas has left the document, i.e. on a real unmount (React removes the
    // host node before running the passive-effect cleanup).
    if (!this.canvas.isConnected) {
      const lose = gl.getExtension("WEBGL_lose_context");
      if (lose) lose.loseContext();
    }
  }
}

/** Parse the CSS custom property `--viewport-bg` (hex or rgb) into 0..1 components. */
export function cssBackground(): [number, number, number] {
  try {
    const v = getComputedStyle(document.documentElement).getPropertyValue("--viewport-bg").trim();
    const hex = v.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
    if (hex) {
      let s = hex[1];
      if (s.length === 3) s = s.split("").map((c) => c + c).join("");
      return [parseInt(s.slice(0, 2), 16) / 255, parseInt(s.slice(2, 4), 16) / 255, parseInt(s.slice(4, 6), 16) / 255];
    }
    const rgb = v.match(/rgba?\(([^)]+)\)/);
    if (rgb) {
      const p = rgb[1].split(",").map((x) => parseFloat(x));
      return [p[0] / 255, p[1] / 255, p[2] / 255];
    }
  } catch {
    /* ignore */
  }
  return [0.04, 0.047, 0.063];
}
