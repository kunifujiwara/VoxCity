"""
VoxCity Integration Module for simulator_gpu.visibility

This module provides utilities for loading VoxCity models and using them
with the GPU-accelerated visibility analysis tools.

This module emulates the voxcity.simulator.visibility API using Taichi GPU acceleration.

VoxCity models contain:
- 3D voxel grids with building, tree, and ground information
- Land cover classification codes
- Building heights and IDs
- Tree canopy data

API Compatibility:
    The functions in this module match the voxcity.simulator.visibility API:
    - get_view_index() - same signature as voxcity.simulator.visibility.get_view_index
    - get_sky_view_factor_map() - same signature as voxcity.simulator.visibility.get_sky_view_factor_map
    - get_surface_view_factor() - same signature as voxcity.simulator.visibility.get_surface_view_factor
    - get_landmark_visibility_map() - same signature as voxcity.simulator.visibility.get_landmark_visibility_map
    - get_surface_landmark_visibility() - same signature as voxcity.simulator.visibility.get_surface_landmark_visibility
    - mark_building_by_id() - same as voxcity.simulator.visibility.mark_building_by_id
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from dataclasses import dataclass
import math

from ..domain import Domain
from .view import ViewCalculator, SurfaceViewFactorCalculator
from .landmark import LandmarkVisibilityCalculator, SurfaceLandmarkVisibilityCalculator
from .landmark import mark_building_by_id


# VoxCity voxel class codes
VOXCITY_GROUND_CODE = -1
VOXCITY_TREE_CODE = -2
VOXCITY_BUILDING_CODE = -3

# Green view target codes
GREEN_VIEW_CODES = (-2, 2, 5, 6, 7, 8)  # Trees and vegetation classes


def create_domain_from_voxcity(voxcity) -> Domain:
    """
    Create a Domain object from a VoxCity model.
    
    Args:
        voxcity: VoxCity object with voxels attribute
    
    Returns:
        Domain object configured for view analysis
    """
    voxel_data = voxcity.voxels.classes
    nx, ny, nz = voxel_data.shape
    meshsize = voxcity.voxels.meta.meshsize
    
    domain = Domain(
        nx=nx, ny=ny, nz=nz,
        dx=meshsize, dy=meshsize, dz=meshsize
    )
    
    # Set domain from voxel data
    domain.set_from_voxel_data(voxel_data, tree_code=VOXCITY_TREE_CODE)
    
    return domain


def get_view_index_gpu(
    voxcity,
    mode: str = None,
    hit_values: Tuple[int, ...] = None,
    inclusion_mode: bool = True,
    view_point_height: float = 1.5,
    n_azimuth: int = 120,
    n_elevation: int = 20,
    elevation_min_degrees: float = -30.0,
    elevation_max_degrees: float = 30.0,
    ray_sampling: str = "grid",
    n_rays: int = None,
    tree_k: float = 0.5,
    tree_lad: float = 1.0,
    show_plot: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated View Index calculation for VoxCity.
    
    This function emulates voxcity.simulator.visibility.view.get_view_index
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        mode: Predefined mode ('green', 'sky', or None for custom)
        hit_values: Target voxel values to count as visible
        inclusion_mode: If True, count hits on targets; if False, count non-blocked rays
        view_point_height: Observer height above ground (meters)
        n_azimuth: Number of azimuthal divisions
        n_elevation: Number of elevation divisions
        elevation_min_degrees: Minimum viewing angle
        elevation_max_degrees: Maximum viewing angle
        ray_sampling: 'grid' or 'fibonacci'
        n_rays: Total rays for fibonacci sampling
        tree_k: Tree extinction coefficient
        tree_lad: Leaf area density
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters
    
    Returns:
        2D array of view index values
    """
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    nx, ny, nz = voxel_data.shape
    
    # Create domain
    domain = Domain(
        nx=nx, ny=ny, nz=nz,
        dx=meshsize, dy=meshsize, dz=meshsize
    )
    
    # Create calculator
    calc = ViewCalculator(
        domain,
        n_azimuth=n_azimuth,
        n_elevation=n_elevation,
        ray_sampling=ray_sampling,
        n_rays=n_rays
    )
    
    # Compute view index
    vi_map = calc.compute_view_index(
        voxel_data=voxel_data,
        mode=mode,
        hit_values=hit_values,
        inclusion_mode=inclusion_mode,
        view_point_height=view_point_height,
        elevation_min_degrees=elevation_min_degrees,
        elevation_max_degrees=elevation_max_degrees,
        tree_k=tree_k,
        tree_lad=tree_lad
    )
    
    # Note: ViewCalculator.compute_view_index already flips to match VoxCity coordinate system
    
    # Plot if requested
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            colormap = kwargs.get('colormap', 'viridis')
            vmin = kwargs.get('vmin', 0.0)
            vmax = kwargs.get('vmax', 1.0)
            
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(vi_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='View Index')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    return vi_map


# VoxCity API-compatible function names (recommended interface)
def get_view_index(voxcity, mode=None, hit_values=None, inclusion_mode=True, fast_path=True, **kwargs):
    """
    GPU-accelerated View Index calculation for VoxCity.
    
    This function matches the signature of voxcity.simulator.visibility.get_view_index
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        mode: Predefined mode ('green', 'sky', or None for custom)
        hit_values: Target voxel values to count as visible
        inclusion_mode: If True, count hits on targets; if False, count non-blocked rays
        **kwargs: Additional parameters including:
            - view_point_height (float): Observer height above ground (default: 1.5)
            - N_azimuth (int): Number of azimuthal divisions (default: 120)
            - N_elevation (int): Number of elevation divisions (default: 20)
            - elevation_min_degrees (float): Minimum viewing angle (default: -30)
            - elevation_max_degrees (float): Maximum viewing angle (default: 30)
            - ray_sampling (str): 'grid' or 'fibonacci' (default: 'grid')
            - N_rays (int): Total rays for fibonacci sampling
            - tree_k (float): Tree extinction coefficient (default: 0.5)
            - tree_lad (float): Leaf area density (default: 1.0)
            - colormap (str): Matplotlib colormap name (default: 'viridis')
            - vmin, vmax (float): Colormap limits (default: 0.0, 1.0)
            - obj_export (bool): Whether to export OBJ file (default: False)
    
    Returns:
        2D array of view index values
    """
    # Map VoxCity-style kwargs to our internal parameter names
    n_azimuth = kwargs.pop('N_azimuth', kwargs.pop('n_azimuth', 120))
    n_elevation = kwargs.pop('N_elevation', kwargs.pop('n_elevation', 20))
    n_rays = kwargs.pop('N_rays', kwargs.pop('n_rays', None))
    view_point_height = kwargs.pop('view_point_height', 1.5)
    elevation_min_degrees = kwargs.pop('elevation_min_degrees', -30.0)
    elevation_max_degrees = kwargs.pop('elevation_max_degrees', 30.0)
    ray_sampling = kwargs.pop('ray_sampling', 'grid')
    tree_k = kwargs.pop('tree_k', 0.5)
    tree_lad = kwargs.pop('tree_lad', 1.0)
    
    return get_view_index_gpu(
        voxcity,
        mode=mode,
        hit_values=hit_values,
        inclusion_mode=inclusion_mode,
        view_point_height=view_point_height,
        n_azimuth=n_azimuth,
        n_elevation=n_elevation,
        elevation_min_degrees=elevation_min_degrees,
        elevation_max_degrees=elevation_max_degrees,
        ray_sampling=ray_sampling,
        n_rays=n_rays,
        tree_k=tree_k,
        tree_lad=tree_lad,
        show_plot=True,  # VoxCity default shows plot
        **kwargs
    )


def get_sky_view_factor_map(voxcity, show_plot=False, **kwargs):
    """
    GPU-accelerated Sky View Factor calculation for VoxCity.
    
    This function matches the signature of voxcity.simulator.visibility.get_sky_view_factor_map
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters including:
            - view_point_height (float): Observer height above ground (default: 1.5)
            - N_azimuth (int): Number of azimuthal divisions (default: 120)
            - N_elevation (int): Number of elevation divisions (default: 20)
            - tree_k (float): Tree extinction coefficient (default: 0.6)
            - tree_lad (float): Leaf area density (default: 1.0)
            - colormap (str): Matplotlib colormap name (default: 'BuPu_r')
    
    Returns:
        2D array of SVF values
    """
    n_azimuth = kwargs.pop('N_azimuth', kwargs.pop('n_azimuth', 120))
    n_elevation = kwargs.pop('N_elevation', kwargs.pop('n_elevation', 20))
    view_point_height = kwargs.pop('view_point_height', 1.5)
    tree_k = kwargs.pop('tree_k', 0.6)
    tree_lad = kwargs.pop('tree_lad', 1.0)
    colormap = kwargs.pop('colormap', 'BuPu_r')
    
    return get_view_index_gpu(
        voxcity,
        mode='sky',
        view_point_height=view_point_height,
        n_azimuth=n_azimuth,
        n_elevation=n_elevation,
        elevation_min_degrees=0.0,
        elevation_max_degrees=90.0,
        tree_k=tree_k,
        tree_lad=tree_lad,
        show_plot=show_plot,
        colormap=colormap,
        **kwargs
    )


def get_surface_view_factor(voxcity, mode=None, **kwargs):
    """
    GPU-accelerated Surface View Factor calculation for VoxCity.
    
    This function matches the signature of voxcity.simulator.visibility.get_surface_view_factor
    using Taichi GPU acceleration.
    
    Computes view factors for building surface faces by tracing rays from
    face centers through the voxel domain.
    
    Args:
        voxcity: VoxCity object
        **kwargs: Additional parameters including:
            - value_name (str): Name for the metadata field (default: 'view_factor_values')
            - colormap (str): Matplotlib colormap name (default: 'BuPu_r')
            - vmin, vmax (float): Colormap limits (default: 0.0, 1.0)
            - N_azimuth (int): Number of azimuthal divisions (default: 120)
            - N_elevation (int): Number of elevation divisions (default: 20)
            - ray_sampling (str): 'grid' or 'fibonacci' (default: 'grid')
            - tree_k (float): Tree extinction coefficient (default: 0.6)
            - tree_lad (float): Leaf area density (default: 1.0)
            - target_values (tuple): Target voxel values (default: (0,))
            - inclusion_mode (bool): Inclusion vs exclusion mode (default: False)
            - building_class_id (int): Building class ID for mesh extraction (default: -3)
            - progress_report (bool): Show progress (default: False)
            - obj_export (bool): Export mesh to OBJ (default: False)
    
    Returns:
        Trimesh object with view factor values in metadata
    """
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    building_id_grid = voxcity.buildings.ids
    nx, ny, nz = voxel_data.shape
    
    value_name = kwargs.get('value_name', 'view_factor_values')
    n_azimuth = kwargs.get('N_azimuth', kwargs.get('n_azimuth', 120))
    n_elevation = kwargs.get('N_elevation', kwargs.get('n_elevation', 20))
    ray_sampling = kwargs.get('ray_sampling', 'grid')
    n_rays = kwargs.get('N_rays', kwargs.get('n_rays', None))
    tree_k = kwargs.get('tree_k', 0.6)
    tree_lad = kwargs.get('tree_lad', 1.0)
    building_class_id = kwargs.get('building_class_id', -3)
    progress_report = kwargs.get('progress_report', False)
    
    # Handle mode parameter
    if mode == 'sky':
        target_values = (0,)
        inclusion_mode = False
    elif mode == 'green':
        target_values = (-2, 2, 5, 6, 7, 8)
        inclusion_mode = True
    else:
        target_values = kwargs.get('target_values', (0,))
        inclusion_mode = kwargs.get('inclusion_mode', False)
    
    # Try to import mesh creation utility
    try:
        from voxcity.geoprocessor.mesh import create_voxel_mesh
    except ImportError:
        raise ImportError("VoxCity geoprocessor.mesh module required for surface view factor calculation")
    
    # Create mesh from building voxels
    try:
        building_mesh = create_voxel_mesh(
            voxel_data,
            building_class_id,
            meshsize,
            building_id_grid=building_id_grid,
            mesh_type='open_air'
        )
        if building_mesh is None or len(building_mesh.faces) == 0:
            print("No surfaces found in voxel data for the specified class.")
            return None
    except Exception as e:
        print(f"Error during mesh extraction: {e}")
        return None
    
    if progress_report:
        print(f"Processing view factor for {len(building_mesh.faces)} faces...")
    
    face_centers = building_mesh.triangles_center.astype(np.float32)
    face_normals = building_mesh.face_normals.astype(np.float32)
    
    # Create domain and calculator
    domain = Domain(
        nx=nx, ny=ny, nz=nz,
        dx=meshsize, dy=meshsize, dz=meshsize
    )
    
    calc = SurfaceViewFactorCalculator(
        domain,
        n_azimuth=n_azimuth,
        n_elevation=n_elevation,
        ray_sampling=ray_sampling,
        n_rays=n_rays
    )
    
    # Compute surface view factors
    face_vf_values = calc.compute_surface_view_factor(
        face_centers=face_centers,
        face_normals=face_normals,
        voxel_data=voxel_data,
        target_values=target_values,
        inclusion_mode=inclusion_mode,
        tree_k=tree_k,
        tree_lad=tree_lad
    )
    
    # Add values to mesh metadata
    if not hasattr(building_mesh, 'metadata'):
        building_mesh.metadata = {}
    building_mesh.metadata[value_name] = face_vf_values
    
    # Export if requested
    obj_export = kwargs.get('obj_export', False)
    if obj_export:
        import os
        output_dir = kwargs.get('output_directory', 'output')
        output_file_name = kwargs.get('output_file_name', 'surface_view_factor')
        os.makedirs(output_dir, exist_ok=True)
        try:
            building_mesh.export(f"{output_dir}/{output_file_name}.obj")
            if progress_report:
                print(f"Exported surface mesh to {output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    
    return building_mesh


def get_landmark_visibility_map(voxcity, building_gdf=None, **kwargs):
    """
    GPU-accelerated Landmark Visibility Map calculation for VoxCity.
    
    This function matches the signature of voxcity.simulator.visibility.get_landmark_visibility_map
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        building_gdf: GeoDataFrame of buildings (optional, will use voxcity.extras['building_gdf'])
        **kwargs: Additional parameters including:
            - view_point_height (float): Observer height above ground (default: 1.5)
            - colormap (str): Matplotlib colormap name (default: 'viridis')
            - landmark_building_ids (list): List of building IDs to mark as landmarks
            - landmark_polygon: Polygon to select landmark buildings
            - tree_k (float): Tree extinction coefficient (default: 0.6)
            - tree_lad (float): Leaf area density (default: 1.0)
            - obj_export (bool): Export results to OBJ (default: False)
    
    Returns:
        Tuple of (visibility_map, modified_voxel_data)
    """
    landmark_building_ids = kwargs.pop('landmark_building_ids', None)
    view_point_height = kwargs.pop('view_point_height', 1.5)
    tree_k = kwargs.pop('tree_k', 0.6)
    tree_lad = kwargs.pop('tree_lad', 1.0)
    colormap = kwargs.pop('colormap', 'viridis')
    
    # Get landmark IDs from various sources if not provided
    if landmark_building_ids is None:
        landmark_polygon = kwargs.get('landmark_polygon', None)
        if landmark_polygon is not None and building_gdf is not None:
            try:
                from voxcity.geoprocessor.selection import get_buildings_in_drawn_polygon
                # Convert landmark_polygon to VoxCity expected format
                # VoxCity expects: [{'vertices': [(x1,y1), (x2,y2), ...]}]
                if hasattr(landmark_polygon, 'exterior'):
                    # Single shapely Polygon - convert to VoxCity format
                    polygons = [{'vertices': list(landmark_polygon.exterior.coords)}]
                elif isinstance(landmark_polygon, list) and len(landmark_polygon) > 0:
                    if isinstance(landmark_polygon[0], dict) and 'vertices' in landmark_polygon[0]:
                        # Already in VoxCity format
                        polygons = landmark_polygon
                    elif hasattr(landmark_polygon[0], 'exterior'):
                        # List of shapely Polygons
                        polygons = [{'vertices': list(p.exterior.coords)} for p in landmark_polygon]
                    else:
                        # Assume list of coordinate tuples - wrap as single polygon
                        polygons = [{'vertices': landmark_polygon}]
                else:
                    polygons = landmark_polygon
                # Use 'intersect' to find buildings that touch/overlap the polygon
                landmark_building_ids = get_buildings_in_drawn_polygon(building_gdf, polygons, operation='intersect')
            except ImportError:
                pass
        
        if landmark_building_ids is None:
            rectangle_vertices = kwargs.get('rectangle_vertices', None)
            if rectangle_vertices is None:
                rectangle_vertices = voxcity.extras.get('rectangle_vertices', None)
            if rectangle_vertices is not None and building_gdf is not None:
                try:
                    from voxcity.geoprocessor.selection import find_building_containing_point
                    lons = [coord[0] for coord in rectangle_vertices]
                    lats = [coord[1] for coord in rectangle_vertices]
                    center_lon = (min(lons) + max(lons)) / 2
                    center_lat = (min(lats) + max(lats)) / 2
                    target_point = (center_lon, center_lat)
                    landmark_building_ids = find_building_containing_point(building_gdf, target_point)
                except ImportError:
                    pass
    
    if landmark_building_ids is None:
        print("Cannot set landmark buildings. You need to input either of rectangle_vertices or landmark_building_ids.")
        return None
    
    return get_landmark_visibility_map_gpu(
        voxcity,
        building_gdf=building_gdf,
        landmark_building_ids=landmark_building_ids,
        view_point_height=view_point_height,
        tree_k=tree_k,
        tree_lad=tree_lad,
        show_plot=True,  # VoxCity default shows plot
        colormap=colormap,
        **kwargs
    )


def get_surface_landmark_visibility(voxcity, building_gdf=None, **kwargs):
    """
    GPU-accelerated Surface Landmark Visibility calculation for VoxCity.
    
    This function matches the signature of voxcity.simulator.visibility.get_surface_landmark_visibility
    using Taichi GPU acceleration.
    
    Computes landmark visibility for building surface faces.
    
    Args:
        voxcity: VoxCity object
        building_gdf: GeoDataFrame of buildings
        **kwargs: Additional parameters including:
            - landmark_building_ids (list): List of building IDs to mark as landmarks
            - landmark_polygon: Polygon to select landmark buildings
            - tree_k (float): Tree extinction coefficient (default: 0.6)
            - tree_lad (float): Leaf area density (default: 1.0)
            - building_class_id (int): Building class ID (default: -3)
            - progress_report (bool): Show progress (default: False)
            - colormap (str): Matplotlib colormap name (default: 'RdYlGn')
            - obj_export (bool): Export mesh to OBJ (default: False)
    
    Returns:
        Tuple of (building_mesh with visibility, modified_voxel_data)
    """
    if building_gdf is None:
        building_gdf = voxcity.extras.get('building_gdf', None)
        if building_gdf is None:
            raise ValueError("building_gdf not provided and not found in voxcity.extras['building_gdf']")
    
    voxel_data = voxcity.voxels.classes
    building_id_grid = voxcity.buildings.ids
    meshsize = voxcity.voxels.meta.meshsize
    nx, ny, nz = voxel_data.shape
    
    progress_report = kwargs.get('progress_report', False)
    landmark_building_ids = kwargs.get('landmark_building_ids', None)
    landmark_polygon = kwargs.get('landmark_polygon', None)
    tree_k = kwargs.get('tree_k', 0.6)
    tree_lad = kwargs.get('tree_lad', 1.0)
    building_class_id = kwargs.get('building_class_id', -3)
    colormap = kwargs.get('colormap', 'RdYlGn')
    landmark_value = -30
    
    # Get landmark IDs
    if landmark_building_ids is None:
        if landmark_polygon is not None:
            try:
                from voxcity.geoprocessor.selection import get_buildings_in_drawn_polygon
                # Convert landmark_polygon to VoxCity expected format
                # VoxCity expects: [{'vertices': [(x1,y1), (x2,y2), ...]}]
                if hasattr(landmark_polygon, 'exterior'):
                    # Single shapely Polygon - convert to VoxCity format
                    polygons = [{'vertices': list(landmark_polygon.exterior.coords)}]
                elif isinstance(landmark_polygon, list) and len(landmark_polygon) > 0:
                    if isinstance(landmark_polygon[0], dict) and 'vertices' in landmark_polygon[0]:
                        # Already in VoxCity format
                        polygons = landmark_polygon
                    elif hasattr(landmark_polygon[0], 'exterior'):
                        # List of shapely Polygons
                        polygons = [{'vertices': list(p.exterior.coords)} for p in landmark_polygon]
                    else:
                        # Assume list of coordinate tuples - wrap as single polygon
                        polygons = [{'vertices': landmark_polygon}]
                else:
                    polygons = landmark_polygon
                # Use 'intersect' to find buildings that touch/overlap the polygon
                landmark_building_ids = get_buildings_in_drawn_polygon(building_gdf, polygons, operation='intersect')
            except ImportError:
                pass
        else:
            rectangle_vertices = kwargs.get('rectangle_vertices', None)
            if rectangle_vertices is None:
                rectangle_vertices = voxcity.extras.get('rectangle_vertices', None)
            if rectangle_vertices is None:
                print("Cannot set landmark buildings. You need to input either rectangle_vertices or landmark_building_ids.")
                return None, None
            try:
                from voxcity.geoprocessor.selection import find_building_containing_point
                lons = [coord[0] for coord in rectangle_vertices]
                lats = [coord[1] for coord in rectangle_vertices]
                center_lon = (min(lons) + max(lons)) / 2
                center_lat = (min(lats) + max(lats)) / 2
                target_point = (center_lon, center_lat)
                landmark_building_ids = find_building_containing_point(building_gdf, target_point)
            except ImportError:
                pass
    
    if landmark_building_ids is None:
        print("Cannot set landmark buildings. No landmark_building_ids found.")
        return None, None
    
    # Prepare voxel data
    voxel_data_for_mesh = voxel_data.copy()
    voxel_data_modified = voxel_data.copy()
    voxel_data_modified = mark_building_by_id(voxel_data_modified, building_id_grid, landmark_building_ids, landmark_value)
    voxel_data_for_mesh = mark_building_by_id(voxel_data_for_mesh, building_id_grid, landmark_building_ids, 0)
    
    landmark_positions = np.argwhere(voxel_data_modified == landmark_value).astype(np.float32)
    if landmark_positions.shape[0] == 0:
        print(f"No landmarks found after marking buildings with IDs: {landmark_building_ids}")
        return None, None
    
    if progress_report:
        print(f"Found {landmark_positions.shape[0]} landmark voxels")
        print(f"Landmark building IDs: {landmark_building_ids}")
    
    # Create mesh
    try:
        from voxcity.geoprocessor.mesh import create_voxel_mesh
        building_mesh = create_voxel_mesh(
            voxel_data_for_mesh,
            building_class_id,
            meshsize,
            building_id_grid=building_id_grid,
            mesh_type='open_air'
        )
        if building_mesh is None or len(building_mesh.faces) == 0:
            print("No non-landmark building surfaces found in voxel data.")
            return None, None
    except ImportError:
        raise ImportError("VoxCity geoprocessor.mesh module required for surface landmark visibility")
    except Exception as e:
        print(f"Error during mesh extraction: {e}")
        return None, None
    
    if progress_report:
        print(f"Processing landmark visibility for {len(building_mesh.faces)} faces...")
    
    face_centers = building_mesh.triangles_center.astype(np.float32)
    face_normals = building_mesh.face_normals.astype(np.float32)
    
    # Create domain and calculator
    domain = Domain(
        nx=nx, ny=ny, nz=nz,
        dx=meshsize, dy=meshsize, dz=meshsize
    )
    
    calc = SurfaceLandmarkVisibilityCalculator(domain)
    calc.set_landmarks_from_positions(landmark_positions)
    
    # Compute surface landmark visibility
    visibility_values = calc.compute_surface_landmark_visibility(
        face_centers=face_centers,
        face_normals=face_normals,
        voxel_data=voxel_data_modified,
        landmark_value=landmark_value,
        tree_k=tree_k,
        tree_lad=tree_lad
    )
    
    # Add to mesh metadata
    building_mesh.metadata = getattr(building_mesh, 'metadata', {})
    building_mesh.metadata['landmark_visibility'] = visibility_values
    
    if progress_report:
        valid_mask = ~np.isnan(visibility_values)
        n_valid = np.sum(valid_mask)
        n_visible = np.sum(visibility_values[valid_mask] > 0.5)
        print(f"Landmark visibility statistics:")
        print(f"  Total faces: {len(visibility_values)}")
        print(f"  Valid faces: {n_valid}")
        print(f"  Faces with landmark visibility: {n_visible} ({n_visible/n_valid*100:.1f}%)")
    
    # Export if requested
    obj_export = kwargs.get('obj_export', False)
    if obj_export:
        import os
        try:
            import matplotlib.pyplot as plt
            output_dir = kwargs.get('output_directory', 'output')
            output_file_name = kwargs.get('output_file_name', 'surface_landmark_visibility')
            os.makedirs(output_dir, exist_ok=True)
            
            cmap = plt.cm.get_cmap(colormap)
            face_colors = np.zeros((len(visibility_values), 4))
            for i, val in enumerate(visibility_values):
                if np.isnan(val):
                    face_colors[i] = [0.7, 0.7, 0.7, 1.0]
                else:
                    face_colors[i] = cmap(val)
            building_mesh.visual.face_colors = face_colors
            building_mesh.export(f"{output_dir}/{output_file_name}.obj")
            if progress_report:
                print(f"Exported surface mesh to {output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    
    return building_mesh, voxel_data_modified


def get_sky_view_factor_map_gpu(
    voxcity,
    view_point_height: float = 1.5,
    n_azimuth: int = 120,
    n_elevation: int = 20,
    tree_k: float = 0.6,
    tree_lad: float = 1.0,
    show_plot: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated Sky View Factor calculation for VoxCity.
    
    Legacy function - use get_sky_view_factor_map() for VoxCity API compatibility.
    
    Args:
        voxcity: VoxCity object
        view_point_height: Observer height above ground (meters)
        n_azimuth: Number of azimuthal divisions
        n_elevation: Number of elevation divisions
        tree_k: Tree extinction coefficient
        tree_lad: Leaf area density
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters
    
    Returns:
        2D array of SVF values
    """
    return get_view_index_gpu(
        voxcity,
        mode='sky',
        view_point_height=view_point_height,
        n_azimuth=n_azimuth,
        n_elevation=n_elevation,
        elevation_min_degrees=0.0,
        elevation_max_degrees=90.0,
        tree_k=tree_k,
        tree_lad=tree_lad,
        show_plot=show_plot,
        **kwargs
    )


def get_landmark_visibility_map_gpu(
    voxcity,
    building_gdf=None,
    landmark_building_ids: List[int] = None,
    view_point_height: float = 1.5,
    tree_k: float = 0.6,
    tree_lad: float = 1.0,
    show_plot: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated Landmark Visibility calculation for VoxCity.
    
    Legacy function - use get_landmark_visibility_map() for VoxCity API compatibility.
    
    Args:
        voxcity: VoxCity object
        building_gdf: GeoDataFrame of buildings (optional)
        landmark_building_ids: List of building IDs to mark as landmarks
        view_point_height: Observer height above ground (meters)
        tree_k: Tree extinction coefficient
        tree_lad: Leaf area density
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters
    
    Returns:
        Tuple of (visibility_map, modified_voxel_data)
    """
    if landmark_building_ids is None:
        raise ValueError("landmark_building_ids must be provided")
    
    voxel_data = voxcity.voxels.classes
    building_id_grid = voxcity.buildings.ids
    meshsize = voxcity.voxels.meta.meshsize
    nx, ny, nz = voxel_data.shape
    
    view_height_voxel = int(view_point_height / meshsize)
    
    # Mark landmark buildings
    target_value = -30
    voxel_data_modified = mark_building_by_id(
        voxel_data, building_id_grid, landmark_building_ids, target_value
    )
    
    # Create domain
    domain = Domain(
        nx=nx, ny=ny, nz=nz,
        dx=meshsize, dy=meshsize, dz=meshsize
    )
    
    # Create calculator
    calc = LandmarkVisibilityCalculator(domain)
    calc.set_landmarks_from_voxel_value(voxel_data_modified, target_value)
    
    # Compute visibility
    visibility_map = calc.compute_visibility_map(
        voxel_data=voxel_data_modified,
        view_height_voxel=view_height_voxel,
        tree_k=tree_k,
        tree_lad=tree_lad
    )
    
    # Plot if requested
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            colormap = kwargs.get('colormap', 'viridis')
            
            cmap = plt.cm.get_cmap(colormap, 2).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(visibility_map, origin='lower', cmap=cmap, vmin=0, vmax=1)
            visible_patch = mpatches.Patch(color=cmap(1.0), label='Visible (1)')
            not_visible_patch = mpatches.Patch(color=cmap(0.0), label='Not Visible (0)')
            plt.legend(handles=[visible_patch, not_visible_patch], 
                      loc='center left', bbox_to_anchor=(1.0, 0.5))
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    return visibility_map, voxel_data_modified
