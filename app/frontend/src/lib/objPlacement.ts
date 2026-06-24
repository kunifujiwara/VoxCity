/**
 * Client-side OBJ placement model for the Import tab.
 *
 * `Placement` is the single source of truth shared by the numeric form, the 2D
 * footprint map, and the 3D gizmo. `transformModelPoint` maps a model-space
 * point to scene metres (x=east, y=north, z=up) for the *visual* preview only;
 * the committed voxelization uses the exact server-side transform
 * (`voxcity.importer.transform.build_placement_transform`). This client-side
 * version intentionally omits the server-side "domain rotation" correction
 * (which corrects for the model grid's own rectangle not being exactly
 * true-north-aligned), since that correction is not knowable purely
 * client-side without fetching grid geometry from the backend.
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
  const s = UNIT_SCALE[(units as Units)];
  if (s === undefined) throw new Error(`Unknown units: ${units}`);
  return s;
}

/**
 * Map a model-space point to scene metres relative to the anchor.
 *
 * Mirrors the visual part of voxcity.importer.transform.build_placement_transform:
 *   1. subtract anchorModelPoint, 2. scale by units, 3. rotate `rotation` deg
 *   about the up axis (model +X->east, +Y->north at rotation 0), 4. add move.
 * Returns [east, north, up] metres. Domain rotation + ground offset are applied
 * server-side and intentionally omitted from this visual approximation.
 */
export function transformModelPoint(
  pt: [number, number, number],
  p: Omit<Placement, 'units'> & { units: string },
): [number, number, number] {
  const s = unitScale(p.units);
  const lx = (pt[0] - p.anchorModelPoint[0]) * s;
  const ly = (pt[1] - p.anchorModelPoint[1]) * s;
  const lz = (pt[2] - p.anchorModelPoint[2]) * s;
  const theta = (p.rotation * Math.PI) / 180;
  const cos = Math.cos(theta);
  const sin = Math.sin(theta);
  // east = lx*cos - ly*sin ; north = lx*sin + ly*cos  (CCW, +X->E, +Y->N at 0)
  const east = lx * cos - ly * sin + p.move[0];
  const north = lx * sin + ly * cos + p.move[1];
  const up = lz + p.move[2];
  return [east, north, up];
}
