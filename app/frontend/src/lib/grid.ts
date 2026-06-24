/**
 * Grid helpers for the basemap-backed plan editor.
 *
 * `polygonToCells` rasterises a closed lon/lat polygon to the set of
 * uv `(i, j)` cell indices whose centres lie inside, mirroring voxcity's
 * `matplotlib.path.Path.contains_points` approach in
 * `voxcity.geoprocessor.draw.edit_landcover` / `edit_tree`.
 *
 * `pointInRing` is a standard ray-casting test on a single closed ring.
 */

/**
 * Geometry of the voxel grid produced by compute_grid_geometry().
 *
 * Invariants:
 *   - u_vec and v_vec are unit vectors (1 lon/lat degree per metre scale).
 *   - adj_mesh = [dx_m, dy_m]: actual cell size in metres along each axis;
 *     dx_m = dist(side_1) / nx, dy_m = dist(side_2) / ny.
 *   - Cell (i, j) centre = origin + (i+0.5)*dx_m*u_vec + (j+0.5)*dy_m*v_vec.
 */
export interface GridGeom {
  origin: [number, number];      // [lon, lat] of reference corner (rectangle_vertices[0])
  side_1: [number, number];      // vector v0→v1 in lon/lat
  side_2: [number, number];      // vector v0→v3 in lon/lat
  u_vec:  [number, number];      // lon/lat degrees per metre along side_1
  v_vec:  [number, number];      // lon/lat degrees per metre along side_2
  adj_mesh: [number, number];    // [dx_m, dy_m] adjusted cell size in metres
  grid_size: [number, number];   // [nx, ny] cells along u and v axes
  meshsize_m?: number;           // nominal cell size in metres
}

/** Standard ray-casting point-in-polygon (ring is `[lon, lat]` pairs). */
export function pointInRing(lon: number, lat: number, ring: [number, number][]): boolean {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const intersect =
      (yi > lat) !== (yj > lat) &&
      lon < ((xj - xi) * (lat - yi)) / (yj - yi + 1e-30) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/** Cell centre in lon/lat coordinates. */
export function cellCentre(i: number, j: number, g: GridGeom): [number, number] {
  const [dx, dy] = g.adj_mesh;
  const o = g.origin;
  const u = g.u_vec;
  const v = g.v_vec;
  return [
    o[0] + (i + 0.5) * dx * u[0] + (j + 0.5) * dy * v[0],
    o[1] + (i + 0.5) * dx * u[1] + (j + 0.5) * dy * v[1],
  ];
}

/**
 * Rasterise a closed polygon (ring of `[lon, lat]` pairs, no need to repeat
 * the first vertex at the end) into the set of uv `(i, j)` cell indices whose
 * centres lie inside.
 *
 * For typical voxcity grids (≤200 × 200) this is a few-tens-of-thousands
 * point-in-polygon tests — fast enough on the main thread.
 */
export function polygonToCells(
  ring: [number, number][],
  grid: GridGeom,
): [number, number][] {
  if (ring.length < 3) return [];
  const [nx, ny] = grid.grid_size;

  // Bounding box in lon/lat to skip cells trivially outside.
  let minLon = Infinity, maxLon = -Infinity, minLat = Infinity, maxLat = -Infinity;
  for (const [lon, lat] of ring) {
    if (lon < minLon) minLon = lon;
    if (lon > maxLon) maxLon = lon;
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
  }

  const out: [number, number][] = [];
  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const [lon, lat] = cellCentre(i, j, grid);
      if (lon < minLon || lon > maxLon || lat < minLat || lat > maxLat) continue;
      if (pointInRing(lon, lat, ring)) out.push([i, j]);
    }
  }
  return out;
}

/**
 * Voxcity-style 3-corner rectangle: given three lon/lat clicks p1, p2, p3,
 * build a true orthogonal rectangle in Web-Mercator metres.
 *
 *   - p1 → p2 defines one side (the "width" vector w).
 *   - The fourth corner is p2 + h_len · perp(w),
 *     where h_len = (p3 − p1) · perp_unit(w)  (signed scalar projection).
 *   - The third stored corner is p2 + h_len · perp(w),
 *     fourth is p1 + h_len · perp(w).
 *
 * Mirrors `voxcity.geoprocessor.draw.edit_building.build_rect`. Returns the
 * four lon/lat corners of the rectangle, or `null` when the side or the
 * extrusion is shorter than 0.5 m (matches voxcity's guard).
 *
 * Implementation note: we use Leaflet's spherical-mercator projection in pure
 * JS so this helper is independent of the live `Map` instance.
 */
/** Spherical-Mercator earth radius (metres) — EPSG:3857 / Leaflet convention. */
const SPHERICAL_MERCATOR_R = 6378137;

export function extrudeRectFromSide(
  p1: [number, number],   // [lon, lat]
  p2: [number, number],
  p3: [number, number],
): [number, number][] | null {
  const R = SPHERICAL_MERCATOR_R;
  const D2R = Math.PI / 180;

  const project = ([lon, lat]: [number, number]): [number, number] => {
    const x = R * lon * D2R;
    const y = R * Math.log(Math.tan(Math.PI / 4 + (lat * D2R) / 2));
    return [x, y];
  };
  const unproject = ([x, y]: [number, number]): [number, number] => {
    const lon = (x / R) / D2R;
    const lat = (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) / D2R;
    return [lon, lat];
  };

  const [x1, y1] = project(p1);
  const [x2, y2] = project(p2);
  const [x3, y3] = project(p3);

  const wx = x2 - x1;
  const wy = y2 - y1;
  const wLen = Math.hypot(wx, wy);
  if (wLen < 0.5) return null;

  // Perpendicular unit vector (rotate 90° CCW).
  const ux = wx / wLen;
  const uy = wy / wLen;
  const px = -uy;
  const py = ux;

  // Signed scalar projection of (p3 − p1) onto perp(w).
  const vx = x3 - x1;
  const vy = y3 - y1;
  const hLen = vx * px + vy * py;
  if (Math.abs(hLen) < 0.5) return null;

  const hx = px * hLen;
  const hy = py * hLen;
  const corners: [number, number][] = [
    [x1, y1],
    [x2, y2],
    [x2 + hx, y2 + hy],
    [x1 + hx, y1 + hy],
  ];
  return corners.map(unproject);
}

/**
 * IDs of building features whose footprint intersects the polygon. Uses a
 * simple "vertex-of-the-other inside" + "edge-edge intersection" test against
 * the polygon's bounding box — fast enough for hundreds of features.
 */
export function buildingsInPolygon(
  buildingFc: any,
  ring: [number, number][],
): number[] {
  if (!buildingFc?.features?.length || ring.length < 3) return [];

  // Polygon bounding box.
  let minLon = Infinity, maxLon = -Infinity, minLat = Infinity, maxLat = -Infinity;
  for (const [lon, lat] of ring) {
    if (lon < minLon) minLon = lon;
    if (lon > maxLon) maxLon = lon;
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
  }

  const ids: number[] = [];
  for (const feat of buildingFc.features) {
    const id = feat?.properties?.idx;
    if (id == null) continue;
    const geom = feat?.geometry;
    if (!geom) continue;
    const polys: [number, number][][] =
      geom.type === 'Polygon'
        ? [geom.coordinates[0]]
        : geom.type === 'MultiPolygon'
        ? geom.coordinates.map((p: any) => p[0])
        : [];

    let hit = false;
    for (const outer of polys) {
      // Quick bbox prune.
      let bMinLon = Infinity, bMaxLon = -Infinity, bMinLat = Infinity, bMaxLat = -Infinity;
      for (const [lon, lat] of outer) {
        if (lon < bMinLon) bMinLon = lon;
        if (lon > bMaxLon) bMaxLon = lon;
        if (lat < bMinLat) bMinLat = lat;
        if (lat > bMaxLat) bMaxLat = lat;
      }
      if (bMaxLon < minLon || bMinLon > maxLon || bMaxLat < minLat || bMinLat > maxLat) continue;

      // Test: any building vertex inside polygon?
      for (const [lon, lat] of outer) {
        if (pointInRing(lon, lat, ring)) { hit = true; break; }
      }
      if (hit) break;
      // Test: any polygon vertex inside building?
      for (const [lon, lat] of ring) {
        if (pointInRing(lon, lat, outer)) { hit = true; break; }
      }
      if (hit) break;
    }
    if (hit) ids.push(Number(id));
  }
  return ids;
}

const GEOM_EPS = 1e-10;
const ORIENTATION_EPS_FACTOR = 128 * Number.EPSILON;

function segmentEps(a: [number, number], b: [number, number], c?: [number, number], d?: [number, number]): number {
  const coords = c && d ? [a, b, c, d] : [a, b];
  let scale = 0;
  for (let i = 0; i < coords.length; i++) {
    for (let j = i + 1; j < coords.length; j++) {
      scale = Math.max(
        scale,
        Math.abs(coords[i][0] - coords[j][0]),
        Math.abs(coords[i][1] - coords[j][1]),
      );
    }
  }
  scale = Math.max(scale, 1e-12);
  return ORIENTATION_EPS_FACTOR * scale * scale;
}

function normalisedRing(ring: [number, number][]): [number, number][] {
  if (ring.length < 2) return ring.slice();
  const first = ring[0];
  const last = ring[ring.length - 1];
  if (Math.abs(first[0] - last[0]) <= GEOM_EPS && Math.abs(first[1] - last[1]) <= GEOM_EPS) {
    return ring.slice(0, -1);
  }
  return ring.slice();
}

function pointOnSegment(
  p: [number, number],
  a: [number, number],
  b: [number, number],
): boolean {
  const cross = (p[1] - a[1]) * (b[0] - a[0]) - (p[0] - a[0]) * (b[1] - a[1]);
  const eps = segmentEps(a, b);
  if (Math.abs(cross) > eps) return false;
  const dot = (p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1]);
  if (dot < -eps) return false;
  const lenSq = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2;
  return dot <= lenSq + eps;
}

function pointInRingInclusive(
  lon: number,
  lat: number,
  ring: [number, number][],
): boolean {
  const vertices = normalisedRing(ring);
  if (vertices.length < 3) return false;
  const p: [number, number] = [lon, lat];
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i];
    const b = vertices[(i + 1) % vertices.length];
    if (pointOnSegment(p, a, b)) return true;
  }
  return pointInRing(lon, lat, vertices);
}

function orientation(a: [number, number], b: [number, number], c: [number, number]): number {
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}

function segmentsProperlyIntersect(
  a1: [number, number],
  a2: [number, number],
  b1: [number, number],
  b2: [number, number],
): boolean {
  const eps = segmentEps(a1, a2, b1, b2);
  const sign = (value: number): number => {
    if (Math.abs(value) <= eps) return 0;
    return value > 0 ? 1 : -1;
  };
  const o1 = orientation(a1, a2, b1);
  const o2 = orientation(a1, a2, b2);
  const o3 = orientation(b1, b2, a1);
  const o4 = orientation(b1, b2, a2);
  return sign(o1) * sign(o2) < 0 && sign(o3) * sign(o4) < 0;
}

function ringFullyContainedInRing(
  subject: [number, number][],
  container: [number, number][],
): boolean {
  const subjectRing = normalisedRing(subject);
  const containerRing = normalisedRing(container);
  if (subjectRing.length < 3 || containerRing.length < 3) return false;

  for (const [lon, lat] of subjectRing) {
    if (!pointInRingInclusive(lon, lat, containerRing)) return false;
  }

  for (let i = 0; i < subjectRing.length; i++) {
    const a = subjectRing[i];
    const b = subjectRing[(i + 1) % subjectRing.length];
    const mid: [number, number] = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    if (!pointInRingInclusive(mid[0], mid[1], containerRing)) return false;
    for (let j = 0; j < containerRing.length; j++) {
      const c = containerRing[j];
      const d = containerRing[(j + 1) % containerRing.length];
      if (segmentsProperlyIntersect(a, b, c, d)) return false;
    }
  }
  return true;
}

export function buildingsFullyContainedInPolygon(
  buildingFc: any,
  ring: [number, number][],
): number[] {
  if (!buildingFc?.features?.length || ring.length < 3) return [];
  const ids: number[] = [];
  for (const feat of buildingFc.features) {
    const id = feat?.properties?.idx;
    if (id == null) continue;
    const geom = feat?.geometry;
    if (!geom) continue;
    const polys: [number, number][][] =
      geom.type === 'Polygon'
        ? [geom.coordinates[0]]
        : geom.type === 'MultiPolygon'
        ? geom.coordinates.map((p: any) => p[0])
        : [];
    if (polys.length === 0) continue;
    if (polys.every((outer) => ringFullyContainedInRing(outer, ring))) {
      ids.push(Number(id));
    }
  }
  return ids;
}

/**
 * Convert a list of (i, j) cells into per-cell lon/lat quadrilateral rings,
 * suitable for rendering pending edits as a Leaflet overlay.
 *
 * Each cell becomes one ring `[bl, br, tr, tl]` using the same math voxcity
 * itself uses (`build_canopy_geojson`):
 *
 *     bl = origin + (i  * dx) * u + (j  * dy) * v
 *     br = origin + ((i+1)*dx) * u + (j  * dy) * v
 *     tr = origin + ((i+1)*dx) * u + ((j+1)*dy) * v
 *     tl = origin + (i  * dx) * u + ((j+1)*dy) * v
 *
 * No merging is performed — N is small in practice (tree brushes, single
 * polygon stamps) and Leaflet handles a few hundred polygons fine.
 */
export function cellsToQuads(
  cells: [number, number][],
  g: GridGeom,
): [number, number][][] {
  const out: [number, number][][] = [];
  const ox = g.origin[0], oy = g.origin[1];
  const ux = g.u_vec[0],  uy = g.u_vec[1];
  const vx = g.v_vec[0],  vy = g.v_vec[1];
  const dx = g.adj_mesh[0];
  const dy = g.adj_mesh[1];
  for (const [i, j] of cells) {
    const a = i * dx, b = (i + 1) * dx;
    const c = j * dy, d = (j + 1) * dy;
    const blx = ox + a * ux + c * vx, bly = oy + a * uy + c * vy;
    const brx = ox + b * ux + c * vx, bry = oy + b * uy + c * vy;
    const trx = ox + b * ux + d * vx, try_ = oy + b * uy + d * vy;
    const tlx = ox + a * ux + d * vx, tly = oy + a * uy + d * vy;
    out.push([
      [blx, bly], [brx, bry], [trx, try_], [tlx, tly], [blx, bly],
    ]);
  }
  return out;
}

/** String key for a cell `(i, j)` — used for `Set`-based mask membership.
 *  Format: `"${i},${j}"`. Inverse: `keyToCell`. */
export const cellKey = (i: number, j: number): string => `${i},${j}`;

/** Parse a key produced by `cellKey` back to `[i, j]`. */
export function keyToCell(s: string): [number, number] {
  const k = s.indexOf(',');
  return [Number(s.slice(0, k)), Number(s.slice(k + 1))];
}

/**
 * Derive the set of cells that the canopy backdrop covers, by testing each
 * cell centre against the merged canopy GeoJSON returned from
 * `build_canopy_geojson`. Polygon holes (interior rings) are honoured.
 *
 * Returned set holds `cellKey(i, j)` strings — cheap to subtract pending
 * `delete_trees` cells from before re-rendering the canopy backdrop.
 */
export function canopyMaskFromGeoJson(
  canopyFc: any,
  g: GridGeom,
): Set<string> {
  const out = new Set<string>();
  const features: any[] = canopyFc?.features ?? [];
  if (features.length === 0) return out;
  const [nx, ny] = g.grid_size;

  // Pre-extract per-feature outer rings + holes + bbox.
  type Poly = { outer: [number, number][]; holes: [number, number][][];
                minLon: number; maxLon: number; minLat: number; maxLat: number };
  const polys: Poly[] = [];
  const pushPoly = (rings: any[]) => {
    if (!rings || rings.length === 0) return;
    const outer = rings[0] as [number, number][];
    const holes = rings.slice(1) as [number, number][][];
    let minLon = Infinity, maxLon = -Infinity, minLat = Infinity, maxLat = -Infinity;
    for (const [lon, lat] of outer) {
      if (lon < minLon) minLon = lon;
      if (lon > maxLon) maxLon = lon;
      if (lat < minLat) minLat = lat;
      if (lat > maxLat) maxLat = lat;
    }
    polys.push({ outer, holes, minLon, maxLon, minLat, maxLat });
  };
  for (const f of features) {
    const geom = f?.geometry;
    if (!geom) continue;
    if (geom.type === 'Polygon') pushPoly(geom.coordinates);
    else if (geom.type === 'MultiPolygon') {
      for (const p of geom.coordinates) pushPoly(p);
    }
  }
  if (polys.length === 0) return out;

  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      const [lon, lat] = cellCentre(i, j, g);
      for (const p of polys) {
        if (lon < p.minLon || lon > p.maxLon || lat < p.minLat || lat > p.maxLat) continue;
        if (!pointInRing(lon, lat, p.outer)) continue;
        let inHole = false;
        for (const h of p.holes) {
          if (pointInRing(lon, lat, h)) { inHole = true; break; }
        }
        if (!inHole) { out.add(cellKey(i, j)); break; }
      }
    }
  }
  return out;
}

/**
 * Build the lon/lat → uv_m projector for `<SceneViewer>` zone outlines.
 *
 * uv_m: u_m = metres along u_vec from grid origin,
 *        v_m = metres along v_vec from grid origin.
 *
 * After Phase 3, build_voxel_buffers places cell (i, j) at scene (i*du, j*dv),
 * so lonLatToUvM() gives the direct scene position — no flip needed.
 *
 * Returns `undefined` when `geo` is missing or the grid is degenerate.
 */
export function lonLatToUvM(
  geo: { grid_geom: GridGeom } | null | undefined,
): ((lon: number, lat: number) => [number, number]) | undefined {
  if (!geo) return undefined;
  const [du, dv] = geo.grid_geom.adj_mesh;
  const [ox, oy] = geo.grid_geom.origin;
  const [ux, uy] = geo.grid_geom.u_vec;
  const [vx, vy] = geo.grid_geom.v_vec;
  const a = ux * du, b = vx * dv;
  const c = uy * du, d = vy * dv;
  const det = a * d - b * c;
  if (Math.abs(det) < 1e-30) return undefined;
  const inv00 = d / det, inv01 = -b / det;
  const inv10 = -c / det, inv11 = a / det;
  return (lon: number, lat: number): [number, number] => {
    const dl = lon - ox;
    const dla = lat - oy;
    const u_cell = inv00 * dl + inv01 * dla;
    const v_cell = inv10 * dl + inv11 * dla;
    // Scene convention: X = east = v_metres, Y = north = u_metres.
    return [v_cell * dv, u_cell * du];
  };
}

/**
 * Inverse of `lonLatToUvM`: scene metres `[east, north]` -> `[lon, lat]`.
 *
 * `lonLatToUvM` solves `[lon, lat] = origin + M * [u_cell, v_cell]` for
 * `[u_cell, v_cell]` by inverting the 2x2 matrix `M = [[a, b], [c, d]]`
 * (built from `u_vec`/`v_vec` scaled by the cell sizes `adj_mesh`), then
 * scales back to metres. This function runs the *same* matrix `M` in its
 * original, non-inverted direction: given scene metres, recover the cell
 * fractions (`u_cell`, `v_cell`) by dividing out the cell sizes, then apply
 * `M` directly to land back on `[lon, lat]` relative to `origin`. Because it
 * reuses `M` as-is (no re-inversion), it is exactly the forward half of the
 * same affine map `lonLatToUvM` inverts — so composing the two is a no-op
 * round trip by construction, not by coincidence.
 */
export function sceneXYToLonLat(geo: GridGeom, eastM: number, northM: number): [number, number] {
  const [du, dv] = geo.adj_mesh;
  const [ox, oy] = geo.origin;
  const [ux, uy] = geo.u_vec;
  const [vx, vy] = geo.v_vec;
  const a = ux * du, b = vx * dv;
  const c = uy * du, d = vy * dv;
  const u_cell = northM / du;
  const v_cell = eastM / dv;
  const dlon = a * u_cell + b * v_cell;
  const dlat = c * u_cell + d * v_cell;
  return [ox + dlon, oy + dlat];
}
