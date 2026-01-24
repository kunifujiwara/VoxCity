"""
Debug test to check coordinate mapping between mesh and domain surfaces.

This test verifies that:
1. Mesh face centers match domain surface centers
2. Mesh face normals match domain surface normals
3. The irradiance values are being mapped correctly
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from voxcity.generator.io import load_voxcity
from voxcity.simulator_gpu.solar import reset_solar_taichi_cache
from voxcity.simulator_gpu.solar.integration import _compute_sun_direction


def main():
    print("=" * 70)
    print("Coordinate Mapping Debug Test")
    print("=" * 70)
    print()
    
    reset_solar_taichi_cache()
    
    # Load voxcity
    input_path = "demo/output/voxcity_mini2.pkl"
    print(f"Loading voxcity from {input_path}...")
    voxcity = load_voxcity(input_path)
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    
    print(f"Voxel data shape: {voxel_data.shape}")
    print(f"Meshsize: {meshsize}m")
    print()
    
    # Create domain
    from voxcity.simulator_gpu.solar.domain import Domain, extract_surfaces_from_domain
    from voxcity.simulator_gpu.solar.integration import _convert_voxel_data_to_arrays, _set_solid_array, _update_topo_from_solid
    
    ni, nj, nk = voxel_data.shape
    domain = Domain(nx=ni, ny=nj, nz=nk, dx=meshsize, dy=meshsize, dz=meshsize)
    
    is_solid_np, lad_np = _convert_voxel_data_to_arrays(voxel_data, 2.0)
    _set_solid_array(domain, is_solid_np)
    _update_topo_from_solid(domain)
    
    surfaces = extract_surfaces_from_domain(domain, 0.2)
    n_surfaces = surfaces.count
    
    print(f"Extracted {n_surfaces} surfaces from domain")
    print()
    
    # Get domain surface positions and directions
    surf_centers = surfaces.center.to_numpy()[:n_surfaces]
    surf_dirs = surfaces.direction.to_numpy()[:n_surfaces]
    
    # Create building mesh
    from voxcity.geoprocessor.mesh import create_voxel_mesh
    
    building_class_id = -3
    building_mesh = create_voxel_mesh(voxel_data, building_class_id, meshsize, mesh_type='open_air')
    
    if building_mesh is None:
        print("No building mesh created!")
        return
    
    n_mesh_faces = len(building_mesh.faces)
    print(f"Created mesh with {n_mesh_faces} faces")
    print()
    
    # Get mesh face centers and normals
    mesh_centers = building_mesh.triangles_center
    mesh_normals = building_mesh.face_normals
    
    print("Coordinate ranges:")
    print(f"  Domain surfaces: x=[{surf_centers[:,0].min():.1f}, {surf_centers[:,0].max():.1f}], "
          f"y=[{surf_centers[:,1].min():.1f}, {surf_centers[:,1].max():.1f}], "
          f"z=[{surf_centers[:,2].min():.1f}, {surf_centers[:,2].max():.1f}]")
    print(f"  Mesh faces:      x=[{mesh_centers[:,0].min():.1f}, {mesh_centers[:,0].max():.1f}], "
          f"y=[{mesh_centers[:,1].min():.1f}, {mesh_centers[:,1].max():.1f}], "
          f"z=[{mesh_centers[:,2].min():.1f}, {mesh_centers[:,2].max():.1f}]")
    print()
    
    # Check if coordinates are swapped
    print("Checking for coordinate swaps...")
    
    # Get building surfaces only from domain
    bldg_mask = np.zeros(n_surfaces, dtype=bool)
    for s_idx in range(n_surfaces):
        i_idx, j_idx, z_idx = surfaces.position.to_numpy()[s_idx]
        i, j, z = int(i_idx), int(j_idx), int(z_idx)
        if 0 <= i < ni and 0 <= j < nj and 0 <= z < nk:
            if voxel_data[i, j, z] == building_class_id:
                bldg_mask[s_idx] = True
    
    bldg_centers = surf_centers[bldg_mask]
    bldg_dirs = surf_dirs[bldg_mask]
    
    print(f"Domain building surfaces: {np.sum(bldg_mask)}")
    print(f"  x range: [{bldg_centers[:,0].min():.1f}, {bldg_centers[:,0].max():.1f}]")
    print(f"  y range: [{bldg_centers[:,1].min():.1f}, {bldg_centers[:,1].max():.1f}]")
    print()
    
    # Check if mesh X range matches domain X range
    # If they're swapped, mesh X would match domain Y and vice versa
    domain_x_range = (bldg_centers[:,0].min(), bldg_centers[:,0].max())
    domain_y_range = (bldg_centers[:,1].min(), bldg_centers[:,1].max())
    mesh_x_range = (mesh_centers[:,0].min(), mesh_centers[:,0].max())
    mesh_y_range = (mesh_centers[:,1].min(), mesh_centers[:,1].max())
    
    print("Comparing X/Y ranges:")
    print(f"  Domain X: [{domain_x_range[0]:.1f}, {domain_x_range[1]:.1f}]")
    print(f"  Mesh X:   [{mesh_x_range[0]:.1f}, {mesh_x_range[1]:.1f}]")
    print(f"  Domain Y: [{domain_y_range[0]:.1f}, {domain_y_range[1]:.1f}]")
    print(f"  Mesh Y:   [{mesh_y_range[0]:.1f}, {mesh_y_range[1]:.1f}]")
    print()
    
    # Check if there's a swap
    domain_x_span = domain_x_range[1] - domain_x_range[0]
    domain_y_span = domain_y_range[1] - domain_y_range[0]
    mesh_x_span = mesh_x_range[1] - mesh_x_range[0]
    mesh_y_span = mesh_y_range[1] - mesh_y_range[0]
    
    if abs(domain_x_span - mesh_y_span) < abs(domain_x_span - mesh_x_span):
        print("WARNING: X/Y coordinates appear to be SWAPPED between mesh and domain!")
        print(f"  Domain X span ({domain_x_span:.1f}) matches Mesh Y span ({mesh_y_span:.1f})")
    else:
        print("X/Y coordinates appear to be CONSISTENT between mesh and domain")
    print()
    
    # Check normal directions for west-facing surfaces
    print("=" * 70)
    print("Checking west-facing surfaces specifically")
    print("=" * 70)
    print()
    
    # West-facing in VoxCity = -y normal = ISOUTH (direction 3)
    ISOUTH = 3
    domain_west_mask = (bldg_dirs == ISOUTH)
    domain_west_centers = bldg_centers[domain_west_mask]
    
    mesh_west_mask = (np.abs(mesh_normals[:, 0] - 0) < 0.1) & \
                     (np.abs(mesh_normals[:, 1] - (-1)) < 0.1) & \
                     (np.abs(mesh_normals[:, 2] - 0) < 0.1)
    mesh_west_centers = mesh_centers[mesh_west_mask]
    
    print(f"Domain west-facing (dir=3, -y normal): {np.sum(domain_west_mask)} surfaces")
    if np.sum(domain_west_mask) > 0:
        print(f"  Y range: [{domain_west_centers[:,1].min():.1f}, {domain_west_centers[:,1].max():.1f}]")
    
    print(f"Mesh west-facing (-y normal): {np.sum(mesh_west_mask)} faces")
    if np.sum(mesh_west_mask) > 0:
        print(f"  Y range: [{mesh_west_centers[:,1].min():.1f}, {mesh_west_centers[:,1].max():.1f}]")
    print()
    
    # Test sun direction for 3 PM
    print("=" * 70)
    print("Sun direction test for 3 PM")
    print("=" * 70)
    print()
    
    # Approximate sun azimuth at 3 PM in Phoenix in June
    # Around 260-270 degrees (SW/W)
    test_azimuth = 262.0
    test_elevation = 58.0
    
    sun_x, sun_y, sun_z, cos_zen = _compute_sun_direction(test_azimuth, test_elevation)
    print(f"Sun azimuth: {test_azimuth}° (SW/West)")
    print(f"Sun elevation: {test_elevation}°")
    print(f"Sun direction: ({sun_x:.4f}, {sun_y:.4f}, {sun_z:.4f})")
    print()
    
    # For west-facing surfaces (-y normal):
    # They should receive sun when sun_y < 0
    print(f"sun_y = {sun_y:.4f}")
    if sun_y < 0:
        print("  => sun_y < 0, so west-facing surfaces SHOULD receive direct sun")
    else:
        print("  => sun_y >= 0, so west-facing surfaces should NOT receive direct sun")
    print()
    
    # Check the actual raytracing condition
    print("Raytracing face_sun condition for direction 3 (ISOUTH, -y normal, West-facing):")
    print("  face_sun = 1 if sun_dir[1] < 0 else 0")
    print(f"  sun_dir[1] = {sun_y:.4f}")
    face_sun = 1 if sun_y < 0 else 0
    print(f"  face_sun = {face_sun}")


if __name__ == "__main__":
    main()
