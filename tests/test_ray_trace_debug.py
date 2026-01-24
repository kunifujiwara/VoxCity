"""
Direct test of palm_solar ray tracing for suspicious west-facing surfaces.

This script traces rays for specific suspicious west faces to understand
why they're being marked as shadowed.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def trace_ray_manually():
    """Manually trace ray from suspicious west face."""
    print("\n" + "=" * 70)
    print("MANUAL RAY TRACING TEST")
    print("=" * 70)
    
    from voxcity.generator.io import load_voxcity
    from voxcity.simulator_gpu.solar import Domain
    from voxcity.simulator_gpu.solar.integration import _convert_voxel_data_to_arrays
    
    # Load VoxCity data
    voxcity = load_voxcity("demo/output/voxcity.pkl")
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    
    ny_vc, nx_vc, nz = voxel_data.shape
    print(f"VoxCity grid: {ny_vc} x {nx_vc} x {nz}")
    print(f"Meshsize: {meshsize}m")
    
    # Domain uses nx=ny_vc, ny=nx_vc (same for 40x40)
    ni, nj, nk = ny_vc, nx_vc, nz
    print(f"Palm_solar domain: nx={ni}, ny={nj}, nz={nk}")
    
    # Set solid cells
    is_solid_np, _ = _convert_voxel_data_to_arrays(voxel_data, 2.0)
    
    # The suspicious face is at mesh coordinates (x=135, y=161.7, z=8.3)
    # These correspond to domain indices:
    #   domain uses: x = i * dx, y = j * dy, z = k * dz
    #   So i = x/dx = 135/5 = 27, j = y/dy = 161.7/5 = 32.34, k = z/dz = 8.3/5 = 1.66
    
    # The west face at x=135 is the left face of cell i=27
    # West face center is at x = 27*5 = 135 (yes, left edge)
    # But surface center is placed at cell center - dx/2 = (27+0.5)*5 - 2.5 = 135
    
    face_x = 135.0
    face_y = 162.5  # cell center y (32+0.5)*5 = 162.5
    face_z = 7.5    # cell center z (1+0.5)*5 = 7.5
    
    face_i = 27
    face_j = 32
    face_k = 1
    
    print(f"\nWest face at cell (i={face_i}, j={face_j}, k={face_k})")
    print(f"Face center position: ({face_x}, {face_y}, {face_z})")
    
    if 0 <= face_i < ni and 0 <= face_j < nj and 0 <= face_k < nk:
        cell_solid = is_solid_np[face_i, face_j, face_k]
        voxel_class = voxel_data[face_i, face_j, face_k]
        print(f"Cell at (i,j,k): is_solid={cell_solid}, voxel_class={voxel_class}")
    
    # Sun direction
    import math
    azimuth = 262.0
    elevation = 54.0
    
    az_rad = math.radians(azimuth)
    el_rad = math.radians(elevation)
    
    sun_dir_x = math.cos(el_rad) * math.sin(az_rad)
    sun_dir_y = math.cos(el_rad) * math.cos(az_rad)
    sun_dir_z = math.sin(el_rad)
    
    print(f"\nSun direction: ({sun_dir_x:.4f}, {sun_dir_y:.4f}, {sun_dir_z:.4f})")
    
    # Now trace ray from face center in sun_dir direction
    print("\n" + "-" * 50)
    print("Manual ray trace from face center:")
    print("-" * 50)
    
    # Domain bounds
    domain_max_x = ni * meshsize  # 200
    domain_max_y = nj * meshsize  # 200
    domain_max_z = nk * meshsize  # 15
    
    print(f"Domain bounds: x=[0, {domain_max_x}], y=[0, {domain_max_y}], z=[0, {domain_max_z}]")
    
    # Step along ray
    ray_x = face_x
    ray_y = face_y
    ray_z = face_z
    
    step = 0.5  # 0.5m steps
    max_steps = 500
    
    hit_solid = False
    for s in range(max_steps):
        t = s * step
        x = ray_x + sun_dir_x * t
        y = ray_y + sun_dir_y * t
        z = ray_z + sun_dir_z * t
        
        # Check if outside domain
        if x < 0 or x >= domain_max_x:
            print(f"  Step {s}: ({x:.1f}, {y:.1f}, {z:.1f}) - LEFT DOMAIN (x)")
            break
        if y < 0 or y >= domain_max_y:
            print(f"  Step {s}: ({x:.1f}, {y:.1f}, {z:.1f}) - LEFT DOMAIN (y)")
            break
        if z < 0 or z >= domain_max_z:
            print(f"  Step {s}: ({x:.1f}, {y:.1f}, {z:.1f}) - LEFT DOMAIN (z)")
            break
        
        # Get voxel at this position
        i = int(x / meshsize)
        j = int(y / meshsize)
        k = int(z / meshsize)
        
        i = max(0, min(ni - 1, i))
        j = max(0, min(nj - 1, j))
        k = max(0, min(nk - 1, k))
        
        if is_solid_np[i, j, k] == 1:
            vox_class = voxel_data[i, j, k]
            print(f"  Step {s} (t={t:.1f}m): ({x:.1f}, {y:.1f}, {z:.1f}) -> cell (i={i}, j={j}, k={k})")
            print(f"    HIT SOLID! is_solid=1, voxel_class={vox_class}")
            hit_solid = True
            break
    
    if not hit_solid:
        print(f"  No solid hit after {max_steps} steps - ray reaches sky!")
    
    # Now let's check why we might hit something at k=0
    print("\n" + "-" * 50)
    print("Checking cells in ray path:")
    print("-" * 50)
    
    # At t=0, we're at (135, 162.5, 7.5)
    # At small t, we move slightly west, slightly south, and UP
    # We should NOT hit ground at k=0
    
    # Let's check the starting cell
    start_i = int(face_x / meshsize)
    start_j = int(face_y / meshsize)
    start_k = int(face_z / meshsize)
    print(f"Starting cell: (i={start_i}, j={start_j}, k={start_k})")
    print(f"  is_solid: {is_solid_np[start_i, start_j, start_k]}")
    print(f"  voxel_class: {voxel_data[start_i, start_j, start_k]}")
    
    # Check cells at k=1 near our starting point
    print(f"\nCells at k=1 near starting point:")
    for di in range(-2, 3):
        for dj in range(-2, 3):
            ii = start_i + di
            jj = start_j + dj
            if 0 <= ii < ni and 0 <= jj < nj:
                if is_solid_np[ii, jj, 1] == 1:
                    print(f"  (i={ii}, j={jj}, k=1): is_solid=1, class={voxel_data[ii, jj, 1]}")


if __name__ == "__main__":
    trace_ray_manually()
