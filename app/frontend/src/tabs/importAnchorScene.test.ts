import { describe, it, expect } from 'vitest';
import { anchorSceneUp } from './importAnchorScene';

describe('anchorSceneUp', () => {
  it('returns 0 when no ground datum is available', () => {
    expect(anchorSceneUp(null, null)).toBe(0);
  });
  it('seats move_up=0 at (effElev - dem_min) + meshsize, auto elevation', () => {
    // ground.dem_elevation=12, dem_min=4, meshsize=2 -> (12-4)+2 = 10
    expect(anchorSceneUp(null, { dem_elevation: 12, dem_min: 4, meshsize_m: 2 })).toBeCloseTo(10, 9);
  });
  it('uses the manual elevation override when set', () => {
    // override=20, dem_min=4, meshsize=2 -> (20-4)+2 = 18
    expect(anchorSceneUp(20, { dem_elevation: 12, dem_min: 4, meshsize_m: 2 })).toBeCloseTo(18, 9);
  });
});
