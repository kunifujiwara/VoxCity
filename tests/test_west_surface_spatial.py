"""
Debug script to analyze spatial distribution of irradiance on west-facing surfaces.

Issue reported:
- West vertical surfaces show shadows even when direct radiation comes from west
- Some west vertical surfaces near west edge show higher irradiance
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from voxcity.generator.io import load_voxcity
from voxcity.simulator_gpu.solar import get_building_global_solar_irradiance_using_epw
from voxcity.simulator_gpu.solar import reset_solar_taichi_cache


def analyze_west_surface_spatial():
    """Analyze spatial distribution of irradiance on west-facing surfaces."""
    
    print("=" * 70)
    print("Spatial Analysis of West-Facing Surfaces at 3 PM")
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
        progress_report=False,
    )
    
    # Get face data
    normals = result.face_normals
    positions = result.triangles_center  # 3D positions (triangle centers)
    direct = result.metadata.get("direct", np.zeros(len(normals)))
    global_irr = result.metadata.get("global", np.zeros(len(normals)))
    
    # Filter to west-facing surfaces (-y normal)
    west_mask = (np.abs(normals[:, 0] - 0) < 0.1) & \
                (np.abs(normals[:, 1] - (-1)) < 0.1) & \
                (np.abs(normals[:, 2] - 0) < 0.1)
    
    west_positions = positions[west_mask]
    west_direct = direct[west_mask]
    
    print(f"\nFound {np.sum(west_mask)} west-facing surfaces")
    print()
    
    # Get sun direction from metadata
    sun_dir = result.metadata.get("sun_direction", None)
    if sun_dir is not None:
        print(f"Sun direction: ({sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f})")
    print()
    
    # Analyze by Y position (West-East)
    y_coords = west_positions[:, 1]
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    
    print(f"Y coordinate range: {y_min:.1f} to {y_max:.1f} meters")
    print("(Smaller Y = more West, Larger Y = more East)")
    print()
    
    # Divide into bins
    n_bins = 5
    y_bins = np.linspace(y_min, y_max, n_bins + 1)
    
    print("Direct irradiance by Y position (West to East):")
    for i in range(n_bins):
        bin_mask = (y_coords >= y_bins[i]) & (y_coords < y_bins[i + 1])
        if i == n_bins - 1:  # Include right edge
            bin_mask = (y_coords >= y_bins[i]) & (y_coords <= y_bins[i + 1])
        
        count = np.sum(bin_mask)
        if count > 0:
            mean_direct = np.mean(west_direct[bin_mask])
            max_direct = np.max(west_direct[bin_mask])
            min_direct = np.min(west_direct[bin_mask])
            nonzero = np.sum(west_direct[bin_mask] > 0)
            pos_label = "West edge" if i == 0 else ("East edge" if i == n_bins - 1 else "Middle")
            print(f"  Y={y_bins[i]:.0f}-{y_bins[i+1]:.0f} ({pos_label}): {count} faces, "
                  f"mean={mean_direct:.1f}, nonzero={nonzero}")
    
    print()
    
    # Also analyze by X position (North-South)
    x_coords = west_positions[:, 0]
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    
    print(f"X coordinate range: {x_min:.1f} to {x_max:.1f} meters")
    print("(Smaller X = more North, Larger X = more South)")
    print()
    
    x_bins = np.linspace(x_min, x_max, n_bins + 1)
    
    print("Direct irradiance by X position (North to South):")
    for i in range(n_bins):
        bin_mask = (x_coords >= x_bins[i]) & (x_coords < x_bins[i + 1])
        if i == n_bins - 1:
            bin_mask = (x_coords >= x_bins[i]) & (x_coords <= x_bins[i + 1])
        
        count = np.sum(bin_mask)
        if count > 0:
            mean_direct = np.mean(west_direct[bin_mask])
            nonzero = np.sum(west_direct[bin_mask] > 0)
            pos_label = "North edge" if i == 0 else ("South edge" if i == n_bins - 1 else "Middle")
            print(f"  X={x_bins[i]:.0f}-{x_bins[i+1]:.0f} ({pos_label}): {count} faces, "
                  f"mean={mean_direct:.1f}, nonzero={nonzero}")
    
    print()
    
    # Check surfaces that ARE receiving direct sun
    lit_mask = west_direct > 0
    shaded_mask = west_direct == 0
    
    print(f"Lit west-facing surfaces: {np.sum(lit_mask)}")
    print(f"Shaded west-facing surfaces: {np.sum(shaded_mask)}")
    print()
    
    if np.sum(lit_mask) > 0:
        lit_positions = west_positions[lit_mask]
        print("Position range of LIT west surfaces:")
        print(f"  X (N-S): {np.min(lit_positions[:, 0]):.1f} to {np.max(lit_positions[:, 0]):.1f}")
        print(f"  Y (W-E): {np.min(lit_positions[:, 1]):.1f} to {np.max(lit_positions[:, 1]):.1f}")
        print(f"  Z (Up):  {np.min(lit_positions[:, 2]):.1f} to {np.max(lit_positions[:, 2]):.1f}")
    
    if np.sum(shaded_mask) > 0:
        shaded_positions = west_positions[shaded_mask]
        print("Position range of SHADED west surfaces:")
        print(f"  X (N-S): {np.min(shaded_positions[:, 0]):.1f} to {np.max(shaded_positions[:, 0]):.1f}")
        print(f"  Y (W-E): {np.min(shaded_positions[:, 1]):.1f} to {np.max(shaded_positions[:, 1]):.1f}")
        print(f"  Z (Up):  {np.min(shaded_positions[:, 2]):.1f} to {np.max(shaded_positions[:, 2]):.1f}")
    
    print()
    print("=" * 70)
    print("Checking ray tracing direction")
    print("=" * 70)
    
    # The issue might be in the ray direction - we trace FROM the surface TO the sun
    # Sun direction vector points FROM the surface TOWARD the sun
    # So if sun is in the West (negative Y), sun_dir_y should be NEGATIVE
    # 
    # For a west-facing surface (-y normal) to receive sun:
    # - The sun must be on the west side (sun_y < 0)
    # - The ray should NOT be blocked
    
    print()
    print("Checking ray offset direction issue...")
    print()
    
    # Let me check if perhaps the ray is being cast in the wrong direction
    # or if there's an offset issue
    
    if sun_dir is not None:
        print(f"Sun direction vector: ({sun_dir[0]:.4f}, {sun_dir[1]:.4f}, {sun_dir[2]:.4f})")
        print()
        
        # For west surfaces (-y normal), direct sun should come when sun_y < 0
        if sun_dir[1] < 0:
            print("Correct: sun_y < 0 means sun is in the west, should illuminate west surfaces")
        else:
            print("ERROR: sun_y >= 0 means sun is NOT in the west at 3 PM!")


def check_shadow_ray_direction():
    """Check the shadow ray tracing direction in detail."""
    print()
    print("=" * 70)
    print("Shadow Ray Direction Analysis")
    print("=" * 70)
    print()
    
    # The key question: when we shoot a ray from a surface toward the sun,
    # are we using the correct direction?
    
    # The code in compute_direct_shadows:
    # 1. Gets surface position
    # 2. Checks if surface normal faces toward sun
    # 3. If yes, traces ray in sun_dir direction
    
    # For west-facing surface (-y normal):
    # - Normal = (0, -1, 0)
    # - For sun in west, sun_dir = (0, -0.7, 0.7) approximately
    # - dot(normal, sun_dir) = 0*0 + (-1)*(-0.7) + 0*0.7 = 0.7 > 0
    # - So the surface DOES face the sun (correct)
    
    # But the raytracing code uses:
    # face_sun = 1 if sun_dir[1] < 0 else 0  (for direction 3, ISOUTH)
    # 
    # This is correct! sun_dir[1] < 0 means sun is in the west
    
    print("The face_sun check logic in raytracing.py is:")
    print("  direction 3 (ISOUTH, -y normal, West-facing):")
    print("    face_sun = 1 if sun_dir[1] < 0 else 0")
    print()
    print("This is CORRECT because:")
    print("  - sun_dir points FROM surface TOWARD sun")
    print("  - If sun is in the west, sun_dir has negative Y component")
    print("  - West-facing surfaces (-y normal) face the west")
    print("  - So they should receive sun when sun_dir[1] < 0")
    print()
    
    # The issue might be elsewhere. Let me check the surface extraction...


if __name__ == "__main__":
    analyze_west_surface_spatial()
    check_shadow_ray_direction()
