import { describe, it, expect } from 'vitest';
import { lonLatToUvM, type GridGeom } from '../lib/grid';

/**
 * Regression guard for the lon/lat <-> scene-metres round trip used by
 * ObjPlacementMap to place OBJ footprint vertices on the basemap.
 *
 * This duplicates ObjPlacementMap.tsx's local `sceneXYToLonLat` helper rather
 * than importing it, because that component module imports `leaflet`, which
 * requires a `window` global unavailable in this project's default
 * (non-DOM) vitest environment. Keep this copy in sync with the real
 * implementation in ObjPlacementMap.tsx if that function ever changes.
 */
function sceneXYToLonLat(geo: GridGeom, eastM: number, northM: number): [number, number] {
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

describe('sceneXYToLonLat round-trips with lonLatToUvM', () => {
  it('recovers the original lon/lat for an axis-aligned grid', () => {
    const grid_geom: GridGeom = {
      origin: [139.0, 35.0],
      side_1: [0, 0],
      side_2: [0, 0],
      u_vec: [1 / 85000, 0],
      v_vec: [0, 1 / 111000],
      adj_mesh: [5, 5],
      grid_size: [100, 100],
    };
    const fwd = lonLatToUvM({ grid_geom })!;
    const points: [number, number][] = [
      [139.01, 35.01],
      [139.001, 35.002],
      [138.995, 34.998],
    ];
    for (const [lon, lat] of points) {
      const [eastM, northM] = fwd(lon, lat);
      const [lon2, lat2] = sceneXYToLonLat(grid_geom, eastM, northM);
      expect(lon2).toBeCloseTo(lon, 9);
      expect(lat2).toBeCloseTo(lat, 9);
    }
  });

  it('recovers the original lon/lat for a rotated/skewed grid', () => {
    const grid_geom: GridGeom = {
      origin: [139.0, 35.0],
      side_1: [0, 0],
      side_2: [0, 0],
      u_vec: [0.7 / 85000, 0.7 / 111000],
      v_vec: [-0.7 / 85000, 0.7 / 111000],
      adj_mesh: [4.3, 6.1],
      grid_size: [80, 60],
    };
    const fwd = lonLatToUvM({ grid_geom })!;
    const points: [number, number][] = [
      [139.02, 35.015],
      [138.98, 34.99],
      [139.0001, 35.0002],
    ];
    for (const [lon, lat] of points) {
      const [eastM, northM] = fwd(lon, lat);
      const [lon2, lat2] = sceneXYToLonLat(grid_geom, eastM, northM);
      expect(lon2).toBeCloseTo(lon, 9);
      expect(lat2).toBeCloseTo(lat, 9);
    }
  });
});
