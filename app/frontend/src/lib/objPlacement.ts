/**
 * Client-side OBJ placement model for the Import tab.
 *
 * `Placement` is the single source of truth shared by the numeric form, the 2D
 * footprint map, and the 3D gizmo. `transformModelPoint` maps a model-space
 * point to scene metres (x=east, y=north, z=up) for the *visual* preview only;
 * the committed voxelization uses the exact server-side transform
 * (`voxcity.importer.transform.build_placement_transform`). This client-side
 * version optionally applies the server's "domain rotation" correction (which
 * corrects for the model grid's own rectangle not being exactly
 * true-north-aligned) via the `domainRotationDeg` parameter; callers that
 * have grid geometry available (e.g. `ObjPlacementMap.tsx`, via
 * `lib/grid.ts`'s `domainRotationDeg`) should pass it for preview parity with
 * the committed result.
 */

export type Units = 'm' | 'cm' | 'mm' | 'ft' | 'in';

export interface Placement {
  anchorLonLat: [number, number] | null; // set by initial map click
  anchorElevation: number | null;        // null -> auto from DEM at commit
  anchorModelPoint: [number, number, number];
  rotation: number;                       // degrees, CCW about up axis
  move: [number, number, number];         // [east, north, up] metres
  units: Units;
  zUp: boolean;
  swapYz: boolean;
}

export function defaultPlacement(): Placement {
  return {
    anchorLonLat: null,
    anchorElevation: null,
    anchorModelPoint: [0, 0, 0],
    rotation: 0,
    move: [0, 0, 0],
    units: 'm',
    zUp: true,
    swapYz: false,
  };
}

const UNIT_SCALE: Record<Units, number> = {
  m: 1, cm: 0.01, mm: 0.001, ft: 0.3048, in: 0.0254,
};

export function unitScale(units: string): number {
  if (!Object.prototype.hasOwnProperty.call(UNIT_SCALE, units)) {
    throw new Error(`Unknown units: ${units}`);
  }
  return UNIT_SCALE[units as Units];
}

/**
 * Map a model-space point to scene metres relative to the anchor.
 *
 * Mirrors the visual part of voxcity.importer.transform.build_placement_transform:
 *   1. subtract anchorModelPoint, 2. scale by units, 3. rotate `rotation` deg
 *   about the up axis (model +X->east, +Y->north at rotation 0), 4. project
 *   onto the grid's own (u, v) axes using `domainRotationDeg` (phi) -- a
 *   no-op when phi=0 -- 5. add move.
 * Returns [east, north, up] metres. Ground offset (DEM-based vertical datum
 * shift) is still applied server-side only and omitted from this visual
 * approximation.
 *
 * NOTE: widened from `Placement` to `units: string` only so the (fixed,
 * spec-mandated) test file's untyped `{ ...base, units: 'ft' }` literal
 * type-checks; do not narrow back to `Placement` or the test file will fail
 * to compile. Callers should still only ever pass a real `Units` value --
 * `unitScale` throws at runtime on anything else.
 *
 * NOTE: three/PlacementGizmo.tsx's mesh-sync effect duplicates this rotation
 * formula by construction (three.js requires imperative position/rotation,
 * not a callable pure function) -- mirror any change to the rotation/anchor
 * math there too.
 */
export function transformModelPoint(
  pt: [number, number, number],
  p: Omit<Placement, 'units'> & { units: string },
  domainRotationDeg = 0,
): [number, number, number] {
  const s = unitScale(p.units);
  const lx = (pt[0] - p.anchorModelPoint[0]) * s;
  const ly = (pt[1] - p.anchorModelPoint[1]) * s;
  const lz = (pt[2] - p.anchorModelPoint[2]) * s;
  const theta = (p.rotation * Math.PI) / 180;
  const cosTheta = Math.cos(theta);
  const sinTheta = Math.sin(theta);
  // east = lx*cos - ly*sin ; north = lx*sin + ly*cos  (CCW, +X->E, +Y->N at 0)
  const e = lx * cosTheta - ly * sinTheta;
  const n = lx * sinTheta + ly * cosTheta;
  // Project (e, n) onto the grid's own (u, v) axes -- server parity with
  // voxcity.importer.transform.build_placement_transform's phi projection
  // (phi = domain rotation bearing of the grid's +u axis):
  //   u = e*sin(phi) + n*cos(phi) ; v = e*cos(phi) - n*sin(phi)
  // At phi=0 this reduces to (u, v) = (n, e), so east=v=e, north=u=n --
  // identical to the pre-domain-rotation formula above.
  const phi = (domainRotationDeg * Math.PI) / 180;
  const cosPhi = Math.cos(phi);
  const sinPhi = Math.sin(phi);
  const u = e * sinPhi + n * cosPhi;
  const v = e * cosPhi - n * sinPhi;
  const east = v + p.move[0];
  const north = u + p.move[1];
  const up = lz + p.move[2];
  // NOTE: v (not u) is the east-axis value here, matching lib/grid.ts's
  // lonLatToUvM/sceneXYToLonLat convention -- see the domain-rotation
  // explanation in this function's docstring above.
  return [east, north, up];
}
