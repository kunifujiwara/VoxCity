"""
Detailed analysis of the 72 shadowed west-facing surfaces.

This script checks each shadowed west face to determine if there's
actually an obstacle to its west that would block sunlight.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def analyze_west_shadows():
    """Detailed analysis of shadowed west-facing surfaces."""
    print("\n" + "=" * 70)
    print("DETAILED SHADOW ANALYSIS: West-facing surfaces")
    print("=" * 70)
    
    from voxcity.generator.io import load_voxcity
    from voxcity.simulator_gpu.solar import get_building_global_solar_irradiance_using_epw
    
    # Load VoxCity data
    voxcity = load_voxcity("demo/output/voxcity.pkl")
    print(f"Loaded VoxCity data")
    
    # Get voxel grid info
    voxel_grid = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    bounds = voxcity.voxels.meta.bounds
    
    print(f"\nVoxel grid shape: {voxel_grid.shape}")
    print(f"Meshsize: {meshsize} m")
    print(f"Bounds: {bounds}")
    
    ny_vc, nx_vc, nz = voxel_grid.shape
    print(f"\nDomain dimensions in meters:")
    print(f"  X (East-West): 0 to {nx_vc * meshsize} m")
    print(f"  Y (North-South): 0 to {ny_vc * meshsize} m")
    print(f"  Z (Up): 0 to {nz * meshsize} m")
    
    # Find EPW file
    epw_path = "demo/output/phoenix-sky.harbor.intl.ap_az_usa.epw"
    
    # Run simulation
    print("\nRunning GPU solar simulation at 3 PM June 1 (sun in west)...")
    
    result_mesh = get_building_global_solar_irradiance_using_epw(
        voxcity,
        calc_type="instantaneous",
        epw_file_path=epw_path,
        calc_time="06-01 15:00:00",
        with_reflections=False,
        progress_report=False,
    )
    
    # Get face data
    face_normals = result_mesh.face_normals
    face_centers = result_mesh.triangles_center
    direct_irradiance = result_mesh.metadata.get('direct', None)
    global_irradiance = result_mesh.metadata.get('global', None)
    
    # Find west-facing faces
    tolerance = 0.5
    west_mask = face_normals[:, 0] < -tolerance
    valid_mask = ~np.isnan(global_irradiance)
    west_valid = west_mask & valid_mask
    
    west_indices = np.where(west_valid)[0]
    west_centers = face_centers[west_valid]
    west_direct = direct_irradiance[west_valid]
    west_global = global_irradiance[west_valid]
    
    # Identify shadowed faces (direct irradiance near 0)
    shadowed_mask = west_direct < 1.0
    shadowed_indices = np.where(shadowed_mask)[0]
    
    print(f"\nTotal west-facing faces: {len(west_indices)}")
    print(f"Shadowed west-facing faces: {len(shadowed_indices)}")
    print(f"Lit west-facing faces: {len(west_indices) - len(shadowed_indices)}")
    
    # Get building mask in voxel grid
    building_class = -3
    
    # Show the building layout at each z level
    print("\n" + "=" * 70)
    print("BUILDING LAYOUT (class=-3) by Z level:")
    print("=" * 70)
    for k in range(nz):
        layer = (voxel_grid[:, :, k] == building_class)
        n_building = np.sum(layer)
        if n_building > 0:
            print(f"\nZ level {k} (z={k*meshsize:.1f}-{(k+1)*meshsize:.1f}m): {n_building} building voxels")
            # Show sparse representation
            rows, cols = np.where(layer)
            if len(rows) <= 30:
                for r, c in zip(rows, cols):
                    x_min_cell = c * meshsize
                    y_min_cell = r * meshsize
                    print(f"  Voxel at row={r}, col={c} -> x=[{x_min_cell:.0f},{x_min_cell+meshsize:.0f}], y=[{y_min_cell:.0f},{y_min_cell+meshsize:.0f}]")
    
    print("\n" + "=" * 70)
    print("ANALYSIS OF SHADOWED WEST FACES")
    print("=" * 70)
    
    # Sun position for June 1, 3 PM Phoenix
    sun_az_rad = np.radians(262)
    sun_elev_rad = np.radians(54)
    
    sun_dir_x = np.cos(sun_elev_rad) * np.sin(sun_az_rad)
    sun_dir_y = np.cos(sun_elev_rad) * np.cos(sun_az_rad)
    sun_dir_z = np.sin(sun_elev_rad)
    
    print(f"\nSun direction (toward surface): ({sun_dir_x:.3f}, {sun_dir_y:.3f}, {sun_dir_z:.3f})")
    print(f"  sun_dir_x = {sun_dir_x:.3f} ({'negative=westward' if sun_dir_x < 0 else 'ERROR: should be negative'})")
    
    # Check a few specific shadowed faces in detail
    suspicious_count = 0
    for idx in shadowed_indices[:15]:  # First 15 shadowed faces
        center = west_centers[idx]
        x, y, z = center
        direct = west_direct[idx]
        
        print(f"\n--- Shadowed west face ---")
        print(f"  Center: ({x:.1f}, {y:.1f}, {z:.1f}) m")
        print(f"  Direct: {direct:.1f} W/m²")
        
        # Convert mesh coordinates to voxel indices
        # Mesh uses: x = col * meshsize, y = row * meshsize
        # West face at mesh x corresponds to voxel column col = x / meshsize
        # Note: west face of voxel at column 'col' is at x = col * meshsize
        col = int(x / meshsize)
        row = int(y / meshsize)
        k = int(z / meshsize)
        
        # Clamp to valid range
        col = max(0, min(nx_vc - 1, col))
        row = max(0, min(ny_vc - 1, row))
        k = max(0, min(nz - 1, k))
        
        print(f"  Voxel indices: row={row}, col={col}, k={k}")
        
        # Check if this voxel is actually a building
        if 0 <= row < ny_vc and 0 <= col < nx_vc and 0 <= k < nz:
            voxel_class = voxel_grid[row, col, k]
            print(f"  Voxel class: {voxel_class} ({'building' if voxel_class == -3 else 'not building'})")
        
        # Look for obstacles to the west (lower col values)
        print(f"  Checking for buildings to the west...")
        found_obstacle = False
        
        for check_col in range(col - 1, -1, -1):
            # Check at same row and z, and nearby
            for dr in [-1, 0, 1]:
                check_row = row + dr
                if 0 <= check_row < ny_vc:
                    for check_k in range(max(0, k-1), min(nz, k+2)):
                        if voxel_grid[check_row, check_col, check_k] == building_class:
                            dist = (col - check_col) * meshsize
                            print(f"    Found building at row={check_row}, col={check_col}, k={check_k} ({dist:.0f}m to west)")
                            found_obstacle = True
                            break
                    if found_obstacle:
                        break
            if found_obstacle:
                break
        
        if not found_obstacle:
            print(f"  -> NO BUILDING FOUND TO WEST - THIS FACE SHOULD BE LIT!")
            suspicious_count += 1
    
    # Compare lit vs shadowed face positions
    print("\n" + "=" * 70)
    print("SPATIAL COMPARISON: Lit vs Shadowed west faces")
    print("=" * 70)
    
    lit_mask = west_direct >= 1.0
    lit_centers = west_centers[lit_mask]
    shadowed_centers = west_centers[shadowed_mask]
    
    print(f"\nLit west faces ({np.sum(lit_mask)}):")
    print(f"  X range: {lit_centers[:,0].min():.1f} to {lit_centers[:,0].max():.1f} m")
    print(f"  Y range: {lit_centers[:,1].min():.1f} to {lit_centers[:,1].max():.1f} m")
    print(f"  Z range: {lit_centers[:,2].min():.1f} to {lit_centers[:,2].max():.1f} m")
    
    print(f"\nShadowed west faces ({np.sum(shadowed_mask)}):")
    print(f"  X range: {shadowed_centers[:,0].min():.1f} to {shadowed_centers[:,0].max():.1f} m")
    print(f"  Y range: {shadowed_centers[:,1].min():.1f} to {shadowed_centers[:,1].max():.1f} m")
    print(f"  Z range: {shadowed_centers[:,2].min():.1f} to {shadowed_centers[:,2].max():.1f} m")
    
    domain_max_x = nx_vc * meshsize
    print(f"\nDomain X extent: 0 to {domain_max_x:.0f} m")
    
    mid_x = domain_max_x / 2
    shadowed_east = np.sum(shadowed_centers[:, 0] > mid_x)
    shadowed_west = np.sum(shadowed_centers[:, 0] <= mid_x)
    print(f"Shadowed faces in eastern half (x > {mid_x:.0f}m): {shadowed_east}")
    print(f"Shadowed faces in western half (x <= {mid_x:.0f}m): {shadowed_west}")
    
    if suspicious_count > 0:
        print(f"\n⚠ WARNING: Found {suspicious_count} suspicious faces (no building to west)")
    else:
        print(f"\n✓ All checked shadowed faces have buildings to their west")


if __name__ == "__main__":
    analyze_west_shadows()
