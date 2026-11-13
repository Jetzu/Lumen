/**
 * Iso-surface meshes for the 3-D viewer: fetch + parse of `GET /api/images/{id}/mesh`.
 *
 * Vertices come in image pixel coordinates (x in [0, W], y in [0, H], z in [0, nz · z/xy ratio]);
 * `extent` is that box, which the renderer maps onto its world-space bounding box.
 */

export interface MeshData {
  /** source channel index */
  channel: number;
  /** xyz triplets, pixel coordinates */
  vertices: Float32Array;
  /** unit normals, xyz triplets */
  normals: Float32Array;
  /** uint32 vertex indices, 3 per triangle (counter-clockwise) */
  faces: Uint32Array;
  nVertices: number;
  nFaces: number;
  /** iso level in raw intensity units (resolved by the server) */
  level: number;
  /** (W, H, D) of the whole volume box in the vertex coordinate system */
  extent: [number, number, number];
  /** x0, y0, z0, x1, y1, z1 of the vertices */
  bounds: [number, number, number, number, number, number];
  components: number;
  lut: string;
  notes: string;
}

export interface MeshRequest {
  id: string;
  rev: number;
  channel: number;
  t: number;
  /** raw intensity units */
  level: number;
  step: number;
  maxDim: number;
  smooth: number;
  /** calibration signature (the z ratio changes the geometry without bumping rev) */
  geomSig: string;
}

const cache = new Map<string, Promise<MeshData>>();
const CACHE_ENTRIES = 12;

export function meshKey(q: MeshRequest): string {
  return `${q.id}:${q.rev}:${q.channel}:${q.t}:${q.level.toPrecision(7)}:${q.step}:${q.maxDim}:${q.smooth}:${q.geomSig}`;
}

export function meshQuery(q: MeshRequest): string {
  return `c=${q.channel}&t=${q.t}&level=${encodeURIComponent(q.level.toPrecision(8))}&step=${q.step}&max_dim=${q.maxDim}&smooth=${q.smooth}`;
}

export type MeshFormat = "obj" | "stl" | "ply";

/**
 * URL of the downloadable mesh file (OBJ / STL / PLY; calibrated units when the image is scaled).
 * Same parameters as the on-screen surface, so the download is served from the server cache and
 * matches what is shown; the `mesh.export` command exports at native resolution (max_dim 0).
 */
export function meshFileUrl(q: MeshRequest, format: MeshFormat): string {
  return `/api/images/${q.id}/mesh.${format}?${meshQuery(q)}`;
}

export function fetchMesh(q: MeshRequest): Promise<MeshData> {
  const key = meshKey(q);
  let p = cache.get(key);
  if (!p) {
    p = (async () => {
      const res = await fetch(`/api/images/${q.id}/mesh?${meshQuery(q)}`);
      if (!res.ok) {
        const j = await res.json().catch(() => ({}));
        throw new Error(j.error || res.statusText || "mesh fetch failed");
      }
      const nVertices = Number(res.headers.get("x-vertices") || 0);
      const nFaces = Number(res.headers.get("x-faces") || 0);
      const level = Number(res.headers.get("x-level") || 0);
      const extent = (res.headers.get("x-extent") || "1,1,1").split(",").map(Number) as [number, number, number];
      const bounds = (res.headers.get("x-bounds") || "0,0,0,0,0,0").split(",").map(Number) as MeshData["bounds"];
      const components = Number(res.headers.get("x-components") || 0);
      const lut = res.headers.get("x-lut") || "Grays";
      const notes = res.headers.get("x-notes") || "";
      const buf = await res.arrayBuffer();
      const need = nVertices * 24 + nFaces * 12;
      if (buf.byteLength < need) throw new Error(`Truncated mesh body (${buf.byteLength} of ${need} bytes)`);
      const vertices = new Float32Array(buf, 0, nVertices * 3);
      const normals = new Float32Array(buf, nVertices * 12, nVertices * 3);
      const faces = new Uint32Array(buf, nVertices * 24, nFaces * 3);
      return { channel: q.channel, vertices, normals, faces, nVertices, nFaces, level, extent, bounds, components, lut, notes };
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

/** Raw-unit iso level matching the Iso mode slider: the uint8 volume is the display range with gamma applied. */
export function isoLevelToRaw(isoLevel: number, ch: { min: number; max: number; gamma?: number }): number {
  const g = ch.gamma && ch.gamma > 0 ? ch.gamma : 1;
  const v = Math.min(1, Math.max(0, isoLevel));
  return ch.min + Math.pow(v, g) * (ch.max - ch.min);
}

export function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}
