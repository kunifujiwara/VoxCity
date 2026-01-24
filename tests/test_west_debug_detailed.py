"""
Debug test to identify why some west-facing surfaces show zero irradiance
when there appear to be no blocking objects.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from voxcity.generator.io import load_voxcity
from voxcity.simulator_gpu.solar import get_building_global_solar_irradiance_using_epw
from voxcity.simulator_gpu.solar import reset_solar_taichi_cache


def main():
    print("=" * 70)
    print("Detailed Debug: West-facing surfaces with zero direct irradiance")
    print("=" * 70)
    print()
    
    # Reset cache
    reset_solar_taichi_cache()
    
    # Load voxcity
    input_path = "demo/output/voxcity_mini2.pkl"
    print(f"Loading voxcity from {input_path}...")
    voxcity = load_voxcity(input_path)
    
    # Get domain size
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    print(f"Domain shape: {voxel_data.shape}")
    print(f"Meshsize: {meshsize}m")
    print(f"Domain size: {voxel_data.shape[0]*meshsize}m x {voxel_data.shape[1]*meshsize}m x {voxel_data.shape[2]*meshsize}m")
    print()
    
    # Run for afternoon (sun in West at 3PM)
    print("Running simulation for 3 PM (sun in west)...")
    result = get_building_global_solar_irradiance_using_epw(
        voxcity,
        calc_type="instantaneous",
        epw_file_path="demo/output/phoenix-sky.harbor.intl.ap_az_usa.epw",
        calc_time="06-01 15:00:00",
        with_reflections=False,
        progress_report=False,
    )
    
    # Get data
    normals = result.face_normals
    positions = result.triangles_center
    direct = result.metadata.get("direct", np.zeros(len(normals)))
    
    # Get sun direction
    sun_dir = result.metadata.get("sun_direction", None)
    sun_azimuth = result.metadata.get("sun_azimuth", None)
    
    print(f"\nSun azimuth: {sun_azimuth:.1f}°" if sun_azimuth else "Sun azimuth: not available")
    if sun_dir is not None:
        print(f"Sun direction: ({sun_dir[0]:.4f}, {sun_dir[1]:.4f}, {sun_dir[2]:.4f})")
    print()
    
    # Filter to west-facing surfaces (-y normal)
    west_mask = (np.abs(normals[:, 0] - 0) < 0.1) & \
                (np.abs(normals[:, 1] - (-1)) < 0.1) & \
                (np.abs(normals[:, 2] - 0) < 0.1)
    
    west_positions = positions[west_mask]
    west_direct = direct[west_mask]
    west_indices = np.where(west_mask)[0]
    
    print(f"Total west-facing surfaces: {np.sum(west_mask)}")
    print(f"  With direct > 0: {np.sum(west_direct > 0)}")
    print(f"  With direct = 0: {np.sum(west_direct == 0)}")
    print()
    
    # Analyze zero-irradiance surfaces
    zero_mask = west_direct == 0
    zero_positions = west_positions[zero_mask]
    
    if len(zero_positions) > 0:
        print("Position range of ZERO-irradiance west surfaces:")
        print(f"  X (N-S): {np.min(zero_positions[:, 0]):.1f} to {np.max(zero_positions[:, 0]):.1f}")
        print(f"  Y (W-E): {np.min(zero_positions[:, 1]):.1f} to {np.max(zero_positions[:, 1]):.1f}")
        print(f"  Z (Up):  {np.min(zero_positions[:, 2]):.1f} to {np.max(zero_positions[:, 2]):.1f}")
        print()
        
        # Check: are these surfaces on the EAST side of the domain?
        domain_y_max = voxel_data.shape[1] * meshsize
        print(f"Domain Y range: 0 to {domain_y_max:.1f}m")
        print()
        
        # Group by Y position
        print("Zero-irradiance surfaces by Y position:")
        y_bins = [0, 50, 100, 150, 200, 250]
        for i in range(len(y_bins)-1):
            bin_mask = (zero_positions[:, 1] >= y_bins[i]) & (zero_positions[:, 1] < y_bins[i+1])
            count = np.sum(bin_mask)
            if count > 0:
                z_vals = zero_positions[bin_mask, 2]
                print(f"  Y={y_bins[i]}-{y_bins[i+1]}m: {count} faces, Z range: {np.min(z_vals):.1f}-{np.max(z_vals):.1f}m")
        print()
        
        # Check if there are buildings blocking these surfaces
        print("Checking for potential blocking objects...")
        print()
        
        # For each zero-irradiance surface, check what's to its west (lower Y)
        # The sun is coming from the west (negative Y direction in sun_dir)
        
        # Sample a few surfaces to debug
        sample_indices = np.random.choice(len(zero_positions), min(5, len(zero_positions)), replace=False)
        
        print("Sample zero-irradiance west-facing surfaces:")
        for idx in sample_indices:
            pos = zero_positions[idx]
            print(f"  Position: X={pos[0]:.1f}, Y={pos[1]:.1f}, Z={pos[2]:.1f}")
            
            # Convert to voxel indices
            vox_i = int(pos[0] / meshsize)
            vox_j = int(pos[1] / meshsize)
            vox_k = int(pos[2] / meshsize)
            
            # Check for solids in the west direction (decreasing j)
            blocking = []
            for dj in range(1, vox_j + 1):
                check_j = vox_j - dj
                if 0 <= check_j < voxel_data.shape[1]:
                    # Check at same height and one above/below
                    for dk in [-1, 0, 1]:
                        check_k = vox_k + dk
                        if 0 <= check_k < voxel_data.shape[2]:
                            val = voxel_data[vox_i, check_j, check_k]
                            if val != 0 and val != -2:  # Not air, not tree
                                blocking.append((check_j, check_k, val))
                                break
                    if blocking:
                        break
            
            if blocking:
                print(f"    -> Blocked by voxel at j={blocking[0][0]}, k={blocking[0][1]} (val={blocking[0][2]})")
            else:
                print(f"    -> NO blocking object found to the west! (BUG?)")
    
    print()
    
    # Now check lit surfaces
    lit_mask = west_direct > 0
    lit_positions = west_positions[lit_mask]
    
    if len(lit_positions) > 0:
        print("Position range of LIT west surfaces:")
        print(f"  X (N-S): {np.min(lit_positions[:, 0]):.1f} to {np.max(lit_positions[:, 0]):.1f}")
        print(f"  Y (W-E): {np.min(lit_positions[:, 1]):.1f} to {np.max(lit_positions[:, 1]):.1f}")
        print(f"  Z (Up):  {np.min(lit_positions[:, 2]):.1f} to {np.max(lit_positions[:, 2]):.1f}")
        print()
        
        print("Lit surfaces by Y position:")
        for i in range(len(y_bins)-1):
            bin_mask = (lit_positions[:, 1] >= y_bins[i]) & (lit_positions[:, 1] < y_bins[i+1])
            count = np.sum(bin_mask)
            if count > 0:
                mean_direct = np.mean(west_direct[lit_mask][bin_mask])
                print(f"  Y={y_bins[i]}-{y_bins[i+1]}m: {count} faces, mean direct={mean_direct:.1f} W/m²")
    
    print()
    print("=" * 70)
    print("Ray Direction Check")
    print("=" * 70)
    print()
    
    if sun_dir is not None:
        print(f"Sun direction vector: ({sun_dir[0]:.4f}, {sun_dir[1]:.4f}, {sun_dir[2]:.4f})")
        print()
        print("For a west-facing surface (normal = [0, -1, 0]):")
        print("  The sun should illuminate it when sun_dir[1] < 0 (sun in west)")
        print(f"  sun_dir[1] = {sun_dir[1]:.4f} {'< 0 ✓' if sun_dir[1] < 0 else '>= 0 ✗'}")
        print()
        
        # Check: when we trace a ray from surface toward sun, what direction does it go?
        print("Ray tracing check:")
        print("  Ray origin: surface position")
        print(f"  Ray direction: sun_dir = ({sun_dir[0]:.3f}, {sun_dir[1]:.3f}, {sun_dir[2]:.3f})")
        print()
        print("  For a west-facing surface at Y=150m:")
        print(f"    Ray goes in direction sun_dir[1]={sun_dir[1]:.3f}")
        if sun_dir[1] < 0:
            print("    This means ray goes WEST (decreasing Y) - CORRECT for sun in west")
        else:
            print("    This means ray goes EAST (increasing Y) - WRONG!")


if __name__ == "__main__":
    main()
