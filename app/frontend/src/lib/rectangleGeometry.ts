/**
 * Pure-JS rectangle geometry helpers that mirror the logic in
 * src/voxcity/geoprocessor/draw/rectangle.py.
 *
 * All functions work in [lon, lat] pairs (WGS-84) using a Web Mercator
 * (EPSG:3857) intermediate projection for accurate distance and rotation.
 *
 * Conventions
 * -----------
 * - Input/output coordinates: [lon, lat] number pairs (voxcity convention).
 * - SW → NW → NE → SE vertex order (indices 0-1-2-3).
 * - Rotation angle: degrees, positive = counter-clockwise.
 */

/** Earth radius used by Web Mercator (EPSG:3857 / Leaflet). */
const R = 6378137;
const D2R = Math.PI / 180;

/** Project [lon, lat] → [x, y] in Web Mercator metres. */
function project([lon, lat]: [number, number]): [number, number] {
  return [
    R * lon * D2R,
    R * Math.log(Math.tan(Math.PI / 4 + (lat * D2R) / 2)),
  ];
}

/** Unproject [x, y] Web Mercator metres → [lon, lat]. */
function unproject([x, y]: [number, number]): [number, number] {
  return [
    x / R / D2R,
    (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2) / D2R,
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Aligned (two-click) rectangle
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build an axis-aligned rectangle from two opposite corners in lon/lat.
 * Returns vertices in SW → NW → NE → SE order.
 */
export function buildAlignedRectangle(
  a: [number, number],
  b: [number, number],
): [number, number][] {
  const west  = Math.min(a[0], b[0]);
  const east  = Math.max(a[0], b[0]);
  const south = Math.min(a[1], b[1]);
  const north = Math.max(a[1], b[1]);
  return [
    [west, south], // SW
    [west, north], // NW
    [east, north], // NE
    [east, south], // SE
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotated (three-click) rectangle
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a rotated rectangle from 3 clicked lon/lat points.
 *
 * - p1 → p2 defines one side (the "width" edge).
 * - p3 sets the depth: the signed scalar projection of (p3 − p1) onto the
 *   perpendicular of the width vector gives the extrusion height.
 *
 * Returns four lon/lat vertices in SW → NW → NE → SE order, or null when
 * either the width or the extrusion is shorter than 0.5 m.
 *
 * Mirrors voxcity._build_rect() in rectangle.py.
 */
export function buildRotatedRectangleFromClicks(
  p1: [number, number],
  p2: [number, number],
  p3: [number, number],
): [number, number][] | null {
  const [x1, y1] = project(p1);
  const [x2, y2] = project(p2);
  const [x3, y3] = project(p3);

  const wx = x2 - x1;
  const wy = y2 - y1;
  const wLen = Math.hypot(wx, wy);
  if (wLen < 0.5) return null;

  // Perpendicular unit vector (rotate 90° CCW)
  const ux = wx / wLen;
  const uy = wy / wLen;
  const px = -uy;
  const py =  ux;

  // Signed scalar projection of (p3 − p1) onto perp(w)
  const vx = x3 - x1;
  const vy = y3 - y1;
  const hLen = vx * px + vy * py;
  if (Math.abs(hLen) < 0.5) return null;

  const hx = px * hLen;
  const hy = py * hLen;

  // Raw four corners: p1 → p2 → p2+h → p1+h
  const corners: [number, number][] = [
    unproject([x1,      y1     ]),
    unproject([x2,      y2     ]),
    unproject([x2 + hx, y2 + hy]),
    unproject([x1 + hx, y1 + hy]),
  ];

  return normalizeRectangleVertices(corners);
}

// ─────────────────────────────────────────────────────────────────────────────
// Vertex normalization
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Reorder four lon/lat rectangle vertices to the canonical SW → NW → NE → SE
 * order expected by VoxCity (rectangle_vertices[0] = SW corner, v0→v1 edge
 * points roughly northward).
 *
 * Mirrors voxcity._normalize_rect_vertices() in rectangle.py.
 */
export function normalizeRectangleVertices(
  verts: [number, number][],
): [number, number][] {
  if (verts.length !== 4) return verts;

  const proj = verts.map(project);

  // Find the edge (among 4 consecutive edges) whose bearing from north is in
  // [-90, 90), i.e. the one pointing generally northward.
  let bestI = 0;
  let bestAngle = Infinity;
  for (let i = 0; i < 4; i++) {
    const j = (i + 1) % 4;
    const dx = proj[j][0] - proj[i][0];
    const dy = proj[j][1] - proj[i][1];
    const angle = (Math.atan2(dx, dy) / D2R); // degrees from north
    if (angle >= -90 && angle < 90 && Math.abs(angle) < Math.abs(bestAngle)) {
      bestI = i;
      bestAngle = angle;
    }
  }

  // Rotate the vertex list so bestI is index 0
  const ordered     = [0, 1, 2, 3].map((k) => verts[(bestI + k) % 4]);
  const projOrdered = [0, 1, 2, 3].map((k) => proj [(bestI + k) % 4]);

  // Ensure v3 is to the RIGHT of v0→v1 (east/SE side).
  // cross(v0→v1, v0→v3) < 0 means v3 is to the right → correct.
  const dx01 = projOrdered[1][0] - projOrdered[0][0];
  const dy01 = projOrdered[1][1] - projOrdered[0][1];
  const dx03 = projOrdered[3][0] - projOrdered[0][0];
  const dy03 = projOrdered[3][1] - projOrdered[0][1];
  const cross = dx01 * dy03 - dy01 * dx03;

  if (cross > 0) {
    // v3 is on the wrong (west) side — reverse the list
    return [ordered[3], ordered[2], ordered[1], ordered[0]];
  }
  return ordered as [number, number][];
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotation in-place
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Rotate a set of lon/lat vertices by `angleDeg` degrees (positive = CCW)
 * around their centroid in Web Mercator space.
 *
 * Mirrors voxcity._rotate_vertices() in rectangle.py.
 */
export function rotateVertices(
  vertices: [number, number][],
  angleDeg: number,
): [number, number][] {
  if (angleDeg === 0) return vertices.map(([lon, lat]) => [lon, lat]);

  const projected = vertices.map(project);
  const cx = projected.reduce((s, [x]) => s + x, 0) / projected.length;
  const cy = projected.reduce((s, [, y]) => s + y, 0) / projected.length;

  // VoxCity uses negative angle_rad (positive = CCW in lon/lat east-up frame)
  const rad = -angleDeg * D2R;
  const cosA = Math.cos(rad);
  const sinA = Math.sin(rad);

  return projected.map(([x, y]) => {
    const dx = x - cx;
    const dy = y - cy;
    return unproject([
      dx * cosA - dy * sinA + cx,
      dx * sinA + dy * cosA + cy,
    ]);
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Fixed-dimension helpers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Compute the axis-aligned base vertices for a rectangle of `widthM` × `heightM`
 * metres centred on [lon, lat].  Returns SW → NW → NE → SE in lon/lat.
 *
 * Uses the small-angle flat-Earth approximation (same method as the existing
 * dimensions endpoint on the backend, which is accurate enough for city-scale
 * areas).
 *
 * A more accurate version hits the `/api/rectangle-from-dimensions` endpoint;
 * this helper is used for the live preview before the backend responds.
 */
export function buildDimensionRectangleApprox(
  centerLon: number,
  centerLat: number,
  widthM: number,
  heightM: number,
  rotationDeg: number = 0,
): [number, number][] {
  const R_merc = 6378137;
  const latRad = centerLat * D2R;
  // 1 metre in degrees at this latitude
  const mPerDegLat = (Math.PI * R_merc) / 180;
  const mPerDegLon = mPerDegLat * Math.cos(latRad);

  const dLon = (widthM  / 2) / mPerDegLon;
  const dLat = (heightM / 2) / mPerDegLat;

  const base: [number, number][] = [
    [centerLon - dLon, centerLat - dLat], // SW
    [centerLon - dLon, centerLat + dLat], // NW
    [centerLon + dLon, centerLat + dLat], // NE
    [centerLon + dLon, centerLat - dLat], // SE
  ];

  return rotationDeg === 0 ? base : rotateVertices(base, rotationDeg);
}
