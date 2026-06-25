import type { AnchorGroundResult } from '../api';

/**
 * Scene-Z (up, metres) at which the imported building's model z=0 sits so that
 * move_up = 0 seats it on the ground — mirroring the commit transform's
 * `(anchor_elevation - dem_min) + meshsize`. `anchorElevation` is the user's
 * manual override (or null to auto-use the DEM sample at the anchor). Returns 0
 * until the ground datum is available.
 */
export function anchorSceneUp(
  anchorElevation: number | null,
  ground: AnchorGroundResult | null,
): number {
  if (!ground) return 0;
  const effElev = anchorElevation ?? ground.dem_elevation;
  return effElev - ground.dem_min + ground.meshsize_m;
}
