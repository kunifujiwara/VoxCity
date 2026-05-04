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
    // Direct scene position: no (nx - u) flip needed after Phase 3.
    return [u_cell * du, v_cell * dv];
  };
}
