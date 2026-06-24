import { describe, it, expect } from 'vitest';
import { lonLatToUvM, sceneXYToLonLat, type GridGeom } from '../lib/grid';
import { transformModelPoint, type Placement } from '../lib/objPlacement';

/**
 * Regression guard for the lon/lat <-> scene-metres round trip used by
 * ObjPlacementMap to place OBJ footprint vertices on the basemap.
 *
 * `sceneXYToLonLat` lives in lib/grid.ts (leaflet-free) so it can be imported
 * directly here without pulling in `leaflet`, which requires a `window`
 * global unavailable in this project's default (non-DOM) vitest environment.
 */
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

  it('composes lonLatToUvM + transformModelPoint + sceneXYToLonLat exactly as ObjPlacementMap\'s redraw effect does, recovering the anchor lon/lat when move/rotation are zero', () => {
    const grid_geom: GridGeom = {
      origin: [139.0, 35.0],
      side_1: [0, 0],
      side_2: [0, 0],
      u_vec: [1 / 85000, 0],
      v_vec: [0, 1 / 111000],
      adj_mesh: [5, 5],
      grid_size: [100, 100],
    };
    const anchorLonLat: [number, number] = [139.01, 35.01];
    const placement: Placement = {
      anchorLonLat,
      anchorElevation: null,
      anchorModelPoint: [0, 0, 0],
      rotation: 0,
      move: [0, 0, 0],
      units: 'm',
      zUp: true,
      swapYz: false,
    };

    // Step 1: anchor's own scene position (same as ObjPlacementMap's `fwd`).
    const fwd = lonLatToUvM({ grid_geom })!;
    const [anchorEastM, anchorNorthM] = fwd(anchorLonLat[0], anchorLonLat[1]);

    // Step 2: placing the model's own anchor point should yield exactly
    // `placement.move` (per Task 4's "places the anchor_model_point at move
    // offset (rotation 0)" test) -- i.e. [0, 0, 0] here.
    const [eastOffset, northOffset] = transformModelPoint(placement.anchorModelPoint, placement);
    expect(eastOffset).toBeCloseTo(0, 9);
    expect(northOffset).toBeCloseTo(0, 9);

    // Step 3: sum (unchanged, since the offset is zero) and convert back to
    // lon/lat exactly as the component's redraw effect does.
    const [lon, lat] = sceneXYToLonLat(grid_geom, anchorEastM + eastOffset, anchorNorthM + northOffset);

    expect(lon).toBeCloseTo(anchorLonLat[0], 6);
    expect(lat).toBeCloseTo(anchorLonLat[1], 6);
  });
});
