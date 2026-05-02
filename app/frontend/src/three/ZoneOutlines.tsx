/**
 * Renders zone polygon outlines that drape over the simulation mesh.
 *
 * Two modes, picked per zone:
 *   - Flat outline: when the underlying mesh inside the zone has uniform
 *     height (Δz < ½·meshsize). A single horizontal `<Line/>` is drawn at
 *     the mesh top + ε (so it doesn't z-fight the mesh).
 *   - Curtain: when there is height variation inside the zone. A simple
 *     vertical extrusion of the 2-D outline is drawn from the zone's
 *     min height to its max height. The full closed outline is rendered
 *     at both top and bottom edges as bright lines so the boundary stays
 *     legible through the scene.
 *
 * Mesh heights are sampled by hashing the overlay quad centroids that
 * fall inside the zone's polygon (point-in-polygon ray test).
 */
import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import * as THREE from 'three';

import type { Zone } from '../types/zones';
import type { OverlayGeometry } from './types';

export interface ZoneOutlinesProps {
  zones: Zone[];
  /** Optional projection from lon/lat -> world-XY metres. */
  lonLatToXY?: (lon: number, lat: number) => [number, number];
  /** Z height (metres) used as a fallback when no overlay is available. */
  zHeight?: number;
  /** Optional explicit colour per zone id (defaults to `zone.color`). */
  colorOverride?: Record<string, string>;
  /** Line width in pixels (drei `<Line>` honours this). */
  lineWidth?: number;
  /** Line width used for curtain top/bottom edges (defaults to ~half of `lineWidth`). */
  curtainEdgeLineWidth?: number;
  /** Side-fill opacity for curtain extrusions. */
  curtainOpacity?: number;
  /** Sim overlay; when provided the outline conforms to its top surface. */
  overlay?: OverlayGeometry | null;
  /** Mesh size in metres. */
  meshsize?: number;
}

const DEFAULT_COLOR = '#ff3366';
/** Vertical lift to avoid z-fighting with the sim mesh. */
const Z_EPS = 0.05;

/** Pre-compute centroids of each overlay quad. */
function buildQuadCentroids(
  overlay: OverlayGeometry | null | undefined,
): { cx: Float32Array; cy: Float32Array; cz: Float32Array } | null {
  if (!overlay) return null;
  const positions = overlay.chunk.positions;
  if (!positions || positions.length < 12) return null;
  const stride = 12; // 4 verts * 3 floats per quad
  const n = Math.floor(positions.length / stride);
  if (n === 0) return null;
  const cx = new Float32Array(n);
  const cy = new Float32Array(n);
  const cz = new Float32Array(n);
  for (let k = 0; k < n; k++) {
    const o = k * stride;
    cx[k] = (positions[o] + positions[o + 3] + positions[o + 6] + positions[o + 9]) * 0.25;
    cy[k] = (positions[o + 1] + positions[o + 4] + positions[o + 7] + positions[o + 10]) * 0.25;
    cz[k] = (positions[o + 2] + positions[o + 5] + positions[o + 8] + positions[o + 11]) * 0.25;
  }
  return { cx, cy, cz };
}

/**
 * Ray-casting point-in-polygon for world XY metres (closed ring).
 * Distinct from grid.ts `pointInRing`, which operates on lon/lat degrees.
 */
function pointInRing(x: number, y: number, ring: [number, number][]): boolean {
  let inside = false;
  const n = ring.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const [xi, yi] = ring[i];
    const [xj, yj] = ring[j];
    const intersect =
      yi > y !== yj > y &&
      x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-30) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

/** Project a ring of lon/lat to world XY metres (closed).
 *  `lonLatToXY` must be provided — the fallback to raw lon/lat is only for
 *  graceful no-op rendering before the scene geometry has loaded. */
function projectRing(
  ring: [number, number][],
  lonLatToXY?: (lon: number, lat: number) => [number, number],
): [number, number][] {
  if (ring.length < 3) return [];
  const xy = ring.map(([lon, lat]) =>
    // Without lonLatToXY the coordinates are degrees, not metres — zone
    // outlines will be invisible or in the wrong position until it loads.
    lonLatToXY ? lonLatToXY(lon, lat) : ([lon, lat] as [number, number]),
  );
  // Close the ring.
  const a = xy[0];
  const b = xy[xy.length - 1];
  if (a[0] !== b[0] || a[1] !== b[1]) xy.push([a[0], a[1]]);
  return xy;
}

/** Axis-aligned bounding box of a 2-D ring. */
function ringBBox(ring: [number, number][]): [number, number, number, number] {
  let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
  for (const [x, y] of ring) {
    if (x < xmin) xmin = x;
    if (y < ymin) ymin = y;
    if (x > xmax) xmax = x;
    if (y > ymax) ymax = y;
  }
  return [xmin, ymin, xmax, ymax];
}

/** Find min/max overlay quad height within the polygon. */
function zoneHeightRange(
  ring: [number, number][],
  centroids: { cx: Float32Array; cy: Float32Array; cz: Float32Array },
): { min: number; max: number; n: number } {
  const [xmin, ymin, xmax, ymax] = ringBBox(ring);
  const { cx, cy, cz } = centroids;
  let min = Infinity;
  let max = -Infinity;
  let n = 0;
  for (let i = 0; i < cx.length; i++) {
    const x = cx[i];
    const y = cy[i];
    if (x < xmin || x > xmax || y < ymin || y > ymax) continue;
    if (!pointInRing(x, y, ring)) continue;
    const z = cz[i];
    if (z < min) min = z;
    if (z > max) max = z;
    n++;
  }
  return { min, max, n };
}

/** Build a vertical ribbon BufferGeometry for one ring at constant min/max z. */
function buildExtrusionGeometry(
  xy: [number, number][],
  zBottom: number,
  zTop: number,
): THREE.BufferGeometry {
  const n = xy.length;
  // 2 verts (top, bottom) per ring vertex.
  const positions = new Float32Array(n * 2 * 3);
  for (let i = 0; i < n; i++) {
    const [x, y] = xy[i];
    positions[i * 6 + 0] = x;
    positions[i * 6 + 1] = y;
    positions[i * 6 + 2] = zTop;
    positions[i * 6 + 3] = x;
    positions[i * 6 + 4] = y;
    positions[i * 6 + 5] = zBottom;
  }
  // 2 triangles per segment.
  const nSeg = n - 1;
  const indices = new Uint32Array(nSeg * 6);
  for (let i = 0; i < nSeg; i++) {
    const a = i * 2;        // top  i
    const b = i * 2 + 1;    // base i
    const c = (i + 1) * 2;  // top  i+1
    const d = (i + 1) * 2 + 1; // base i+1
    indices[i * 6 + 0] = a;
    indices[i * 6 + 1] = b;
    indices[i * 6 + 2] = d;
    indices[i * 6 + 3] = a;
    indices[i * 6 + 4] = d;
    indices[i * 6 + 5] = c;
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geom.setIndex(new THREE.BufferAttribute(indices, 1));
  geom.computeVertexNormals();
  return geom;
}

interface FlatLine {
  kind: 'flat';
  id: string;
  color: string;
  points: [number, number, number][];
}

interface CurtainEntry {
  kind: 'curtain';
  id: string;
  color: string;
  geometry: THREE.BufferGeometry;
  topPoints: [number, number, number][];
  bottomPoints: [number, number, number][];
}

export function ZoneOutlines({
  zones,
  lonLatToXY,
  zHeight = 0.5,
  colorOverride,
  lineWidth = 2.5,
  curtainEdgeLineWidth = 1.2,
  curtainOpacity = 0.45,
  overlay,
  meshsize = 5,
}: ZoneOutlinesProps) {
  const entries = useMemo<(FlatLine | CurtainEntry)[]>(() => {
    const centroids = buildQuadCentroids(overlay);
    const flatEps = meshsize * 0.5;

    const out: (FlatLine | CurtainEntry)[] = [];
    for (const z of zones) {
      const ring = z.ring_lonlat ?? [];
      if (ring.length < 3) continue;
      const color = colorOverride?.[z.id] ?? z.color ?? DEFAULT_COLOR;

      const xy = projectRing(ring, lonLatToXY);
      if (xy.length < 2) continue;

      // Without overlay, fall back to the flat zHeight outline.
      if (!centroids) {
        const pts: [number, number, number][] = xy.map(([x, y]) => [x, y, zHeight]);
        out.push({ kind: 'flat', id: z.id, color, points: pts });
        continue;
      }

      const { min, max, n } = zoneHeightRange(xy, centroids);
      if (n === 0) {
        const pts: [number, number, number][] = xy.map(([x, y]) => [x, y, zHeight]);
        out.push({ kind: 'flat', id: z.id, color, points: pts });
        continue;
      }

      if (max - min < flatEps) {
        // Flat outline at the (uniform) mesh top + ε.
        const top = max + Z_EPS;
        const pts: [number, number, number][] = xy.map(([x, y]) => [x, y, top]);
        out.push({ kind: 'flat', id: z.id, color, points: pts });
        continue;
      }

      // Curtain: simple uniform extrusion of the 2-D outline from min to max.
      const zBottom = min - Z_EPS;
      const zTop = max + Z_EPS;
      const geometry = buildExtrusionGeometry(xy, zBottom, zTop);
      const topPoints: [number, number, number][] = xy.map(([x, y]) => [x, y, zTop]);
      const bottomPoints: [number, number, number][] = xy.map(([x, y]) => [x, y, zBottom]);
      out.push({
        kind: 'curtain',
        id: z.id,
        color,
        geometry,
        topPoints,
        bottomPoints,
      });
    }
    return out;
  }, [zones, lonLatToXY, zHeight, colorOverride, overlay, meshsize]);

  return (
    <group renderOrder={999}>
      {entries.map((e) => {
        if (e.kind === 'flat') {
          return (
            <Line
              key={`${e.id}-flat`}
              points={e.points}
              color={new THREE.Color(e.color)}
              lineWidth={lineWidth}
              dashed
              dashSize={1.0}
              gapSize={0.6}
              depthTest={false}
              depthWrite={false}
              renderOrder={999}
              transparent
            />
          );
        }
        const baseColor = new THREE.Color(e.color);
        return (
          <group key={`${e.id}-curtain`}>
            <mesh geometry={e.geometry} renderOrder={998}>
              <meshBasicMaterial
                color={baseColor}
                transparent
                opacity={curtainOpacity}
                side={THREE.DoubleSide}
                depthTest={false}
                depthWrite={false}
              />
            </mesh>
            <Line
              points={e.topPoints}
              color={baseColor}
              lineWidth={curtainEdgeLineWidth}
              dashed
              dashSize={1.0}
              gapSize={0.6}
              depthTest={false}
              depthWrite={false}
              renderOrder={999}
              transparent
            />
            <Line
              points={e.bottomPoints}
              color={baseColor}
              lineWidth={curtainEdgeLineWidth}
              dashed
              dashSize={1.0}
              gapSize={0.6}
              depthTest={false}
              depthWrite={false}
              renderOrder={999}
              transparent
            />
          </group>
        );
      })}
    </group>
  );
}
