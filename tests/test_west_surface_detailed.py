"""
Detailed debug test for west surface shadowing.

This test analyzes individual surfaces to understand the shadowing pattern.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from voxcity.generator.io import load_voxcity
from voxcity.simulator_gpu.solar import get_building_global_solar_irradiance_using_epw
from voxcity.simulator_gpu.solar import reset_solar_taichi_cache


def main():
    print("=" * 70)
    print("Detailed West Surface Analysis at 3 PM")
    print("=" * 70)
    print()
    
    # Reset cache
    reset_solar_taichi_cache()
    
    # Load voxcity
    input_path = "demo/output/voxcity_mini2.pkl"
    print(f"Loading voxcity from {input_path}...")
    voxcity = load_voxcity(input_path)
    
    # Run for afternoon (sun in West)
    result = get_building_global_solar_irradiance_using_epw(
        voxcity,
        calc_type="instantaneous",
        epw_file_path="demo/output/phoenix-sky.harbor.intl.ap_az_usa.epw",
        calc_time="06-01 15:00:00",
        with_reflections=False,
        progress_report=True,
    )
    
    # Get face data from the Trimesh result
    normals = result.face_normals
    centers = result.triangles_center
    direct = result.metadata.get("direct", np.zeros(len(normals)))
    global_irr = result.metadata.get("global", np.zeros(len(normals)))
    
    print()
    print("Sun position info from metadata:")
    for key in ['sun_azimuth', 'sun_elevation', 'sun_direction']:
        if key in result.metadata:
            print(f"  {key}: {result.metadata[key]}")
    print()
    
    # Filter to west-facing surfaces (-Y normal)
    west_mask = (np.abs(normals[:, 0] - 0) < 0.1) & \
                (np.abs(normals[:, 1] - (-1)) < 0.1) & \
                (np.abs(normals[:, 2] - 0) < 0.1)
    
    n_west = np.sum(west_mask)
    print(f"Found {n_west} west-facing surfaces")
    
    west_centers = centers[west_mask]
    west_direct = direct[west_mask]
    west_normals = normals[west_mask]
    
    # Separate lit and shadowed surfaces
    lit_mask = west_direct > 0
    shadowed_mask = ~lit_mask & ~np.isnan(west_direct)
    nan_mask = np.isnan(west_direct)
    
    print(f"  Lit (direct > 0): {np.sum(lit_mask)}")
    print(f"  Shadowed (direct = 0): {np.sum(shadowed_mask)}")
    print(f"  NaN (boundary): {np.sum(nan_mask)}")
    print()
    
    # Analyze position of lit vs shadowed surfaces
    if np.sum(lit_mask) > 0:
        lit_centers = west_centers[lit_mask]
        print("LIT west-facing surfaces position:")
        print(f"  X (N-S): min={np.min(lit_centers[:, 0]):.1f}, max={np.max(lit_centers[:, 0]):.1f}")
        print(f"  Y (W-E): min={np.min(lit_centers[:, 1]):.1f}, max={np.max(lit_centers[:, 1]):.1f}")
        print(f"  Z (Up):  min={np.min(lit_centers[:, 2]):.1f}, max={np.max(lit_centers[:, 2]):.1f}")
        print(f"  Direct irradiance: {np.mean(west_direct[lit_mask]):.1f} W/m2")
    
    if np.sum(shadowed_mask) > 0:
        shaded_centers = west_centers[shadowed_mask]
        print()
        print("SHADOWED west-facing surfaces position:")
        print(f"  X (N-S): min={np.min(shaded_centers[:, 0]):.1f}, max={np.max(shaded_centers[:, 0]):.1f}")
        print(f"  Y (W-E): min={np.min(shaded_centers[:, 1]):.1f}, max={np.max(shaded_centers[:, 1]):.1f}")
        print(f"  Z (Up):  min={np.min(shaded_centers[:, 2]):.1f}, max={np.max(shaded_centers[:, 2]):.1f}")
    
    print()
    print("=" * 70)
    print("Interpretation:")
    print("=" * 70)
    print()
    
    # The key insight: if shadowed surfaces are at larger Y values (more East),
    # they are correctly being shadowed by buildings to their west
    
    if np.sum(lit_mask) > 0 and np.sum(shadowed_mask) > 0:
        lit_y_mean = np.mean(lit_centers[:, 1])
        shaded_y_mean = np.mean(shaded_centers[:, 1])
        
        print(f"Mean Y position of LIT surfaces: {lit_y_mean:.1f}")
        print(f"Mean Y position of SHADOWED surfaces: {shaded_y_mean:.1f}")
        print()
        
        if shaded_y_mean > lit_y_mean:
            print("CORRECT BEHAVIOR:")
            print("  Shadowed surfaces are more to the EAST (larger Y)")
            print("  They are being blocked by buildings to their west (smaller Y)")
            print("  This is expected when sun is coming from the west")
        else:
            print("POTENTIAL ISSUE:")
            print("  Shadowed surfaces are more to the WEST (smaller Y)")
            print("  This might indicate incorrect shadow calculation")
    
    print()
    print("=" * 70)
    print("Now checking EAST-facing surfaces for comparison")
    print("=" * 70)
    print()
    
    # Filter to east-facing surfaces (+Y normal)
    east_mask = (np.abs(normals[:, 0] - 0) < 0.1) & \
                (np.abs(normals[:, 1] - 1) < 0.1) & \
                (np.abs(normals[:, 2] - 0) < 0.1)
    
    n_east = np.sum(east_mask)
    print(f"Found {n_east} east-facing surfaces")
    
    east_direct = direct[east_mask]
    
    lit_east = np.sum(east_direct > 0)
    shaded_east = np.sum((east_direct == 0) & ~np.isnan(east_direct))
    nan_east = np.sum(np.isnan(east_direct))
    
    print(f"  Lit (direct > 0): {lit_east}")
    print(f"  Shadowed (direct = 0): {shaded_east}")
    print(f"  NaN (boundary): {nan_east}")
    
    if lit_east > 0:
        print(f"  Mean direct irradiance of lit surfaces: {np.nanmean(east_direct[east_direct > 0]):.1f} W/m2")
    
    print()
    print("=" * 70)
    print("CONCLUSION:")
    print("=" * 70)
    print()
    
    west_lit_pct = 100 * np.sum(lit_mask) / max(1, n_west - np.sum(nan_mask))
    east_lit_pct = 100 * lit_east / max(1, n_east - nan_east)
    
    print(f"At 3 PM (sun in west):")
    print(f"  West-facing surfaces lit: {west_lit_pct:.1f}%")
    print(f"  East-facing surfaces lit: {east_lit_pct:.1f}%")
    print()
    
    if west_lit_pct > east_lit_pct:
        print("CORRECT: More west-facing surfaces are lit than east-facing")
    else:
        print("ISSUE: More east-facing surfaces are lit than west-facing (unexpected)")


if __name__ == "__main__":
    main()
