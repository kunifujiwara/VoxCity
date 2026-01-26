"""
CityGML Voxelizer
=================

Voxelizes CityGML LOD2 building geometry to 3D voxel grids compatible with VoxCity.

Key features:
- Surface voxelization using triangle-AABB intersection
- Solid voxelization using ray casting
- Variable terrain from DTM GeoTIFFs or embedded DEM
- Support for buildings, vegetation, bridges, city furniture
- Numba acceleration when available

The voxelizers produce grids using VoxCity semantic codes for seamless integration
with the rest of the VoxCity pipeline.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union

try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

try:
    import rasterio
    from rasterio.windows import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from shapely.geometry import Polygon, Point
    from shapely.prepared import prep
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False

from .models import Building, PLATEAUBuilding
from .constants import (
    GROUND_CODE, TREE_CODE, BUILDING_CODE, BRIDGE_CODE, CITY_FURNITURE_CODE,
    DEFAULT_VOXEL_SIZE, DEFAULT_GROUND_THICKNESS,
)
from .utils import parse_pos_list, triangulate_polygon
from .parsers import PLATEAUParser


# =============================================================================
# Numba-Accelerated Kernels
# =============================================================================

@jit(nopython=True)
def _axis_test_x(v0_y, v0_z, v1_y, v1_z, boxhalfsize_y, boxhalfsize_z, a, b, fa, fb):
    """SAT test on X axis."""
    p0 = a * v0_y - b * v0_z
    p1 = a * v1_y - b * v1_z
    min_p = min(p0, p1)
    max_p = max(p0, p1)
    rad = fa * boxhalfsize_y + fb * boxhalfsize_z
    return min_p > rad or max_p < -rad

@jit(nopython=True)
def _axis_test_y(v0_x, v0_z, v1_x, v1_z, boxhalfsize_x, boxhalfsize_z, a, b, fa, fb):
    """SAT test on Y axis."""
    p0 = -a * v0_x + b * v0_z
    p1 = -a * v1_x + b * v1_z
    min_p = min(p0, p1)
    max_p = max(p0, p1)
    rad = fa * boxhalfsize_x + fb * boxhalfsize_z
    return min_p > rad or max_p < -rad

@jit(nopython=True)
def _axis_test_z(v0_x, v0_y, v1_x, v1_y, boxhalfsize_x, boxhalfsize_y, a, b, fa, fb):
    """SAT test on Z axis."""
    p0 = a * v0_x - b * v0_y
    p1 = a * v1_x - b * v1_y
    min_p = min(p0, p1)
    max_p = max(p0, p1)
    rad = fa * boxhalfsize_x + fb * boxhalfsize_y
    return min_p > rad or max_p < -rad

@jit(nopython=True)
def _plane_box_overlap(normal, d, boxhalfsize):
    """Check if plane (defined by normal and d) overlaps with box."""
    vmin = np.zeros(3)
    vmax = np.zeros(3)
    for q in range(3):
        if normal[q] > 0.0:
            vmin[q] = -boxhalfsize[q]
            vmax[q] = boxhalfsize[q]
        else:
            vmin[q] = boxhalfsize[q]
            vmax[q] = -boxhalfsize[q]
    if normal[0] * vmin[0] + normal[1] * vmin[1] + normal[2] * vmin[2] + d > 0.0:
        return False
    if normal[0] * vmax[0] + normal[1] * vmax[1] + normal[2] * vmax[2] + d >= 0.0:
        return True
    return False

@jit(nopython=True)
def triangle_aabb_intersection(v0, v1, v2, box_center, box_halfsize):
    """
    Check if triangle intersects axis-aligned bounding box (AABB).
    
    Based on Tomas Akenine-Möller's algorithm using the full Separating Axis
    Theorem (SAT) with all 9 edge cross-product tests.
    
    Returns True if triangle and AABB intersect.
    """
    # Translate triangle to box center
    tv0 = v0 - box_center
    tv1 = v1 - box_center
    tv2 = v2 - box_center
    
    # Compute edge vectors
    e0 = tv1 - tv0
    e1 = tv2 - tv1
    e2 = tv0 - tv2
    
    # Bullet 3: Test 9 edge cross-product axes
    fe0 = np.abs(e0)
    fe1 = np.abs(e1)
    fe2 = np.abs(e2)
    
    # Edge 0
    if _axis_test_x(tv0[1], tv0[2], tv2[1], tv2[2], box_halfsize[1], box_halfsize[2], e0[2], e0[1], fe0[2], fe0[1]):
        return False
    if _axis_test_y(tv0[0], tv0[2], tv2[0], tv2[2], box_halfsize[0], box_halfsize[2], e0[2], e0[0], fe0[2], fe0[0]):
        return False
    if _axis_test_z(tv1[0], tv1[1], tv2[0], tv2[1], box_halfsize[0], box_halfsize[1], e0[1], e0[0], fe0[1], fe0[0]):
        return False
    
    # Edge 1
    if _axis_test_x(tv0[1], tv0[2], tv2[1], tv2[2], box_halfsize[1], box_halfsize[2], e1[2], e1[1], fe1[2], fe1[1]):
        return False
    if _axis_test_y(tv0[0], tv0[2], tv2[0], tv2[2], box_halfsize[0], box_halfsize[2], e1[2], e1[0], fe1[2], fe1[0]):
        return False
    if _axis_test_z(tv0[0], tv0[1], tv1[0], tv1[1], box_halfsize[0], box_halfsize[1], e1[1], e1[0], fe1[1], fe1[0]):
        return False
    
    # Edge 2
    if _axis_test_x(tv0[1], tv0[2], tv1[1], tv1[2], box_halfsize[1], box_halfsize[2], e2[2], e2[1], fe2[2], fe2[1]):
        return False
    if _axis_test_y(tv0[0], tv0[2], tv1[0], tv1[2], box_halfsize[0], box_halfsize[2], e2[2], e2[0], fe2[2], fe2[0]):
        return False
    if _axis_test_z(tv1[0], tv1[1], tv2[0], tv2[1], box_halfsize[0], box_halfsize[1], e2[1], e2[0], fe2[1], fe2[0]):
        return False
    
    # Bullet 1: AABB of triangle
    min_x = min(tv0[0], tv1[0], tv2[0])
    max_x = max(tv0[0], tv1[0], tv2[0])
    if min_x > box_halfsize[0] or max_x < -box_halfsize[0]:
        return False
    
    min_y = min(tv0[1], tv1[1], tv2[1])
    max_y = max(tv0[1], tv1[1], tv2[1])
    if min_y > box_halfsize[1] or max_y < -box_halfsize[1]:
        return False
    
    min_z = min(tv0[2], tv1[2], tv2[2])
    max_z = max(tv0[2], tv1[2], tv2[2])
    if min_z > box_halfsize[2] or max_z < -box_halfsize[2]:
        return False
    
    # Bullet 2: Plane-box intersection
    normal = np.cross(e0, e1)
    d = -np.dot(normal, tv0)
    if not _plane_box_overlap(normal, d, box_halfsize):
        return False
    
    return True


@jit(nopython=True, parallel=True)
def _voxelize_building_kernel(
    voxel_grid,
    triangles_v0,
    triangles_v1,
    triangles_v2,
    grid_origin,
    voxel_size,
    ground_level
):
    """
    Numba-accelerated voxelization using ray casting along Z-axis.
    """
    nx, ny, nz = voxel_grid.shape
    num_triangles = triangles_v0.shape[0]
    
    for i in prange(nx):
        for j in range(ny):
            px = grid_origin[0] + (i + 0.5) * voxel_size
            py = grid_origin[1] + (j + 0.5) * voxel_size
            
            origin_z = grid_origin[2] - 10.0
            
            # Collect intersection Z values
            intersections = np.zeros(100)
            num_intersections = 0
            
            for t in range(num_triangles):
                v0 = triangles_v0[t]
                v1 = triangles_v1[t]
                v2 = triangles_v2[t]
                
                # Bounding box check
                if (px < min(v0[0], v1[0], v2[0]) or px > max(v0[0], v1[0], v2[0]) or
                    py < min(v0[1], v1[1], v2[1]) or py > max(v0[1], v1[1], v2[1])):
                    continue
                
                # Möller-Trumbore ray-triangle intersection
                edge1_x = v1[0] - v0[0]
                edge1_y = v1[1] - v0[1]
                edge1_z = v1[2] - v0[2]
                
                edge2_x = v2[0] - v0[0]
                edge2_y = v2[1] - v0[1]
                edge2_z = v2[2] - v0[2]
                
                # h = cross(direction, edge2), direction = (0, 0, 1)
                h_x = -edge2_y
                h_y = edge2_x
                h_z = 0.0
                
                a = edge1_x * h_x + edge1_y * h_y
                
                if abs(a) < 1e-8:
                    continue
                
                f = 1.0 / a
                
                s_x = px - v0[0]
                s_y = py - v0[1]
                s_z = origin_z - v0[2]
                
                u = f * (s_x * h_x + s_y * h_y)
                
                if u < 0.0 or u > 1.0:
                    continue
                
                q_x = s_y * edge1_z - s_z * edge1_y
                q_y = s_z * edge1_x - s_x * edge1_z
                q_z = s_x * edge1_y - s_y * edge1_x
                
                v = f * q_z
                
                if v < 0.0 or u + v > 1.0:
                    continue
                
                t_val = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)
                
                if t_val > 0 and num_intersections < 100:
                    intersections[num_intersections] = origin_z + t_val
                    num_intersections += 1
            
            if num_intersections == 0:
                continue
            
            # Sort intersections
            for ii in range(num_intersections):
                for jj in range(ii + 1, num_intersections):
                    if intersections[jj] < intersections[ii]:
                        tmp = intersections[ii]
                        intersections[ii] = intersections[jj]
                        intersections[jj] = tmp
            
            # Fill between pairs
            for ii in range(0, num_intersections - 1, 2):
                z_min = intersections[ii]
                z_max = intersections[ii + 1]
                
                k_min = int((z_min - grid_origin[2]) / voxel_size) + ground_level
                k_max = int((z_max - grid_origin[2]) / voxel_size) + 1 + ground_level
                
                k_min = max(ground_level, k_min)
                k_max = min(nz, k_max)
                
                for k in range(k_min, k_max):
                    voxel_grid[i, j, k] = -3  # BUILDING_CODE


@jit(nopython=True, parallel=True)
def _voxelize_with_terrain_kernel(
    voxel_grid,
    triangles_v0,
    triangles_v1,
    triangles_v2,
    grid_origin,
    voxel_size,
    dem_grid,
    dem_absolute,
    base_ground_thickness
):
    """
    Voxelization with variable terrain heights following VoxCity approach.
    """
    nx, ny, nz = voxel_grid.shape
    num_triangles = triangles_v0.shape[0]
    
    for i in prange(nx):
        for j in range(ny):
            local_ground_level = int(dem_grid[i, j] / voxel_size + 0.5) + base_ground_thickness
            terrain_z = dem_absolute[i, j]
            
            # Fill ground voxels
            for k in range(min(local_ground_level, nz)):
                if voxel_grid[i, j, k] == 0:
                    voxel_grid[i, j, k] = -1  # GROUND_CODE
            
            px = grid_origin[0] + (i + 0.5) * voxel_size
            py = grid_origin[1] + (j + 0.5) * voxel_size
            origin_z = grid_origin[2] - 10.0
            
            intersections = np.zeros(100)
            num_intersections = 0
            
            for t in range(num_triangles):
                v0 = triangles_v0[t]
                v1 = triangles_v1[t]
                v2 = triangles_v2[t]
                
                if (px < min(v0[0], v1[0], v2[0]) or px > max(v0[0], v1[0], v2[0]) or
                    py < min(v0[1], v1[1], v2[1]) or py > max(v0[1], v1[1], v2[1])):
                    continue
                
                edge1_x = v1[0] - v0[0]
                edge1_y = v1[1] - v0[1]
                edge1_z = v1[2] - v0[2]
                
                edge2_x = v2[0] - v0[0]
                edge2_y = v2[1] - v0[1]
                edge2_z = v2[2] - v0[2]
                
                h_x = -edge2_y
                h_y = edge2_x
                
                a = edge1_x * h_x + edge1_y * h_y
                
                if abs(a) < 1e-8:
                    continue
                
                f = 1.0 / a
                
                s_x = px - v0[0]
                s_y = py - v0[1]
                s_z = origin_z - v0[2]
                
                u = f * (s_x * h_x + s_y * h_y)
                
                if u < 0.0 or u > 1.0:
                    continue
                
                q_x = s_y * edge1_z - s_z * edge1_y
                q_y = s_z * edge1_x - s_x * edge1_z
                q_z = s_x * edge1_y - s_y * edge1_x
                
                v = f * q_z
                
                if v < 0.0 or u + v > 1.0:
                    continue
                
                t_val = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z)
                
                if t_val > 0 and num_intersections < 100:
                    intersections[num_intersections] = origin_z + t_val
                    num_intersections += 1
            
            if num_intersections == 0:
                continue
            
            for ii in range(num_intersections):
                for jj in range(ii + 1, num_intersections):
                    if intersections[jj] < intersections[ii]:
                        tmp = intersections[ii]
                        intersections[ii] = intersections[jj]
                        intersections[jj] = tmp
            
            for ii in range(0, num_intersections - 1, 2):
                z_min_abs = intersections[ii]
                z_max_abs = intersections[ii + 1]
                
                building_min_height = z_min_abs - terrain_z
                building_max_height = z_max_abs - terrain_z
                
                k_min = local_ground_level + int(building_min_height / voxel_size)
                k_max = local_ground_level + int(building_max_height / voxel_size) + 1
                
                k_min = max(local_ground_level, k_min)
                k_max = min(nz, k_max)
                
                for k in range(k_min, k_max):
                    voxel_grid[i, j, k] = -3  # BUILDING_CODE


@jit(nopython=True, parallel=True)
def _voxelize_surface_kernel(
    voxel_grid,
    triangles_v0,
    triangles_v1,
    triangles_v2,
    grid_origin,
    voxel_size,
    dem_grid_relative,
    dem_grid_absolute,
    ground_thickness,
    building_id_grid
):
    """
    Surface voxelization kernel using triangle-AABB intersection.
    
    This approach voxelizes the actual surfaces (walls, roofs, etc.) rather than
    trying to fill solid volumes. Works well for LOD2 building representations
    where the mesh is not necessarily watertight.
    
    Following the reference implementation, this uses per-cell terrain adjustment:
    for each (i, j) cell, the triangle Z is adjusted relative to THAT cell's terrain.
    
    IMPORTANT: Only voxelizes cells that are within a building footprint to prevent
    artifacts from terrain discontinuities at building edges.
    """
    nx, ny, nz = voxel_grid.shape
    num_triangles = triangles_v0.shape[0]
    half_size = voxel_size / 2.0
    box_halfsize = np.array([half_size, half_size, half_size])
    
    # First pass: add terrain-following ground layer
    for i in prange(nx):
        for j in range(ny):
            terrain_rel = dem_grid_relative[i, j]
            
            # Ground level in voxel indices
            local_ground_level = ground_thickness + int(terrain_rel / voxel_size)
            
            # Fill ground from 0 to local_ground_level
            for k in range(local_ground_level + 1):
                if voxel_grid[i, j, k] == 0:
                    voxel_grid[i, j, k] = -1  # GROUND_CODE
    
    # Second pass: voxelize each triangle
    for t in prange(num_triangles):
        v0 = triangles_v0[t]
        v1 = triangles_v1[t]
        v2 = triangles_v2[t]
        
        # Get triangle bounding box in world coordinates
        min_x = min(v0[0], v1[0], v2[0])
        max_x = max(v0[0], v1[0], v2[0])
        min_y = min(v0[1], v1[1], v2[1])
        max_y = max(v0[1], v1[1], v2[1])
        
        # Convert XY to grid indices
        i_min = max(0, int((min_x - grid_origin[0]) / voxel_size) - 1)
        i_max = min(nx - 1, int((max_x - grid_origin[0]) / voxel_size) + 1)
        j_min = max(0, int((min_y - grid_origin[1]) / voxel_size) - 1)
        j_max = min(ny - 1, int((max_y - grid_origin[1]) / voxel_size) + 1)
        
        # Check each XY cell in the bounding box
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                # CRITICAL: Only process cells within building footprints
                # This prevents artifacts from terrain discontinuities at edges
                if building_id_grid[i, j] == 0:
                    continue
                
                # Get terrain at this cell
                terrain_abs = dem_grid_absolute[i, j]
                terrain_rel = dem_grid_relative[i, j]
                local_ground_level = ground_thickness + int(terrain_rel / voxel_size)
                
                # Adjust triangle Z relative to this cell's terrain
                # Building Z is absolute, we convert to height above terrain, then to voxel space
                v0_height = v0[2] - terrain_abs  # height above terrain
                v1_height = v1[2] - terrain_abs
                v2_height = v2[2] - terrain_abs
                
                # Convert to voxel Z coordinates (above local ground level)
                v0_voxel_z = grid_origin[2] + (local_ground_level + v0_height / voxel_size) * voxel_size
                v1_voxel_z = grid_origin[2] + (local_ground_level + v1_height / voxel_size) * voxel_size
                v2_voxel_z = grid_origin[2] + (local_ground_level + v2_height / voxel_size) * voxel_size
                
                # Create adjusted vertices for intersection tests
                v0_adj = np.array([v0[0], v0[1], v0_voxel_z])
                v1_adj = np.array([v1[0], v1[1], v1_voxel_z])
                v2_adj = np.array([v2[0], v2[1], v2_voxel_z])
                
                # Z range for this adjusted triangle
                min_z_adj = min(v0_voxel_z, v1_voxel_z, v2_voxel_z)
                max_z_adj = max(v0_voxel_z, v1_voxel_z, v2_voxel_z)
                
                k_min = max(local_ground_level + 1, int((min_z_adj - grid_origin[2]) / voxel_size) - 1)
                k_max = min(nz - 1, int((max_z_adj - grid_origin[2]) / voxel_size) + 2)
                
                for k in range(k_min, k_max + 1):
                    if k <= local_ground_level:
                        continue
                    
                    # Voxel center in world coordinates
                    voxel_center = np.array([
                        grid_origin[0] + (i + 0.5) * voxel_size,
                        grid_origin[1] + (j + 0.5) * voxel_size,
                        grid_origin[2] + (k + 0.5) * voxel_size
                    ])
                    
                    if triangle_aabb_intersection(v0_adj, v1_adj, v2_adj, voxel_center, box_halfsize):
                        voxel_grid[i, j, k] = -3  # BUILDING_CODE


@jit(nopython=True, parallel=True)
def _fill_building_columns_guided(voxel_grid, ground_thickness, dem_grid_relative, building_id_grid, voxel_size):
    """
    Fill interior of buildings using footprint guidance (LOD2-Solid).
    
    This ensures:
    - Solid buildings (no hollow interiors)
    - Anchored to ground (no floating voxels)
    - Preserves roof shapes from surface voxelization
    
    For each cell (i,j) that is within a building footprint:
    1. Find the highest voxelized building point (roof)
    2. Fill from roof down to ground level
    """
    nx, ny, nz = voxel_grid.shape
    
    for i in prange(nx):
        for j in range(ny):
            # Check if this cell is within a building footprint
            if building_id_grid[i, j] == 0:
                continue
            
            # Get local ground level
            terrain_rel = dem_grid_relative[i, j]
            local_ground_level = ground_thickness + int(terrain_rel / voxel_size)
            
            # Find max building voxel (roof) - search from top down for performance
            k_max_bldg = -1
            for k in range(nz - 1, local_ground_level, -1):
                if voxel_grid[i, j, k] == -3:  # BUILDING_CODE
                    k_max_bldg = k
                    break
            
            # If we found a roof, fill everything below it down to ground
            if k_max_bldg > local_ground_level:
                for k in range(local_ground_level + 1, k_max_bldg):
                    if voxel_grid[i, j, k] != -1:  # Don't overwrite ground
                        voxel_grid[i, j, k] = -3  # BUILDING_CODE


@jit(nopython=True, parallel=True)
def _voxelize_objects_kernel(
    voxel_grid,
    triangles_v0,
    triangles_v1,
    triangles_v2,
    grid_origin,
    voxel_size,
    voxel_code
):
    """
    Generic surface voxelization for additional objects (trees, bridges, etc.).
    """
    nx, ny, nz = voxel_grid.shape
    num_triangles = triangles_v0.shape[0]
    half_size = voxel_size / 2.0
    box_halfsize = np.array([half_size, half_size, half_size])
    
    for t in prange(num_triangles):
        v0 = triangles_v0[t]
        v1 = triangles_v1[t]
        v2 = triangles_v2[t]
        
        min_x = min(v0[0], v1[0], v2[0])
        max_x = max(v0[0], v1[0], v2[0])
        min_y = min(v0[1], v1[1], v2[1])
        max_y = max(v0[1], v1[1], v2[1])
        min_z = min(v0[2], v1[2], v2[2])
        max_z = max(v0[2], v1[2], v2[2])
        
        i_min = max(0, int((min_x - grid_origin[0]) / voxel_size) - 1)
        i_max = min(nx - 1, int((max_x - grid_origin[0]) / voxel_size) + 1)
        j_min = max(0, int((min_y - grid_origin[1]) / voxel_size) - 1)
        j_max = min(ny - 1, int((max_y - grid_origin[1]) / voxel_size) + 1)
        k_min = max(0, int((min_z - grid_origin[2]) / voxel_size) - 1)
        k_max = min(nz - 1, int((max_z - grid_origin[2]) / voxel_size) + 1)
        
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                for k in range(k_min, k_max + 1):
                    if voxel_grid[i, j, k] != 0:
                        continue
                    
                    voxel_center = np.array([
                        grid_origin[0] + (i + 0.5) * voxel_size,
                        grid_origin[1] + (j + 0.5) * voxel_size,
                        grid_origin[2] + (k + 0.5) * voxel_size
                    ])
                    
                    if triangle_aabb_intersection(v0, v1, v2, voxel_center, box_halfsize):
                        voxel_grid[i, j, k] = voxel_code


# =============================================================================
# CityGML Voxelizer (German/European Format)
# =============================================================================

class CityGMLVoxelizer:
    """
    Voxelizer for standard CityGML LOD2 buildings with DTM terrain support.
    
    Creates VoxCity-compatible 3D voxel grids from CityGML geometry.
    
    Example::
    
        voxelizer = CityGMLVoxelizer(voxel_size=1.0)
        voxelizer.load_dtm("path/to/dtm.tif")  # Optional
        voxelizer.parse_citygml("path/to/building.gml")
        voxel_grid = voxelizer.voxelize(use_terrain=True)
    """
    
    def __init__(self, voxel_size: float = DEFAULT_VOXEL_SIZE, voxel_dtype=np.int8):
        """
        Initialize the voxelizer.
        
        Args:
            voxel_size: Size of each voxel cube in meters.
            voxel_dtype: NumPy dtype for voxel grid.
        """
        self.voxel_size = float(voxel_size)
        self.voxel_dtype = voxel_dtype
        self.buildings: List[Building] = []
        
        self.bounds_min: Optional[np.ndarray] = None
        self.bounds_max: Optional[np.ndarray] = None
        
        # DTM data
        self.dtm_data: Optional[np.ndarray] = None
        self.dtm_transform = None
        self.dtm_crs = None
        self.terrain_min_elevation = 0.0
    
    def load_dtm(self, dtm_path: str) -> np.ndarray:
        """
        Load a DTM GeoTIFF file for terrain representation.
        
        Args:
            dtm_path: Path to the GeoTIFF file.
            
        Returns:
            2D array of elevation values.
        """
        if not HAS_RASTERIO:
            raise ImportError("rasterio is required for DTM loading")
        
        print(f"Loading DTM: {dtm_path}")
        
        with rasterio.open(dtm_path) as src:
            self.dtm_data = src.read(1)
            self.dtm_transform = src.transform
            self.dtm_crs = src.crs
        
        print(f"  DTM shape: {self.dtm_data.shape}")
        print(f"  DTM range: {np.nanmin(self.dtm_data):.2f} - {np.nanmax(self.dtm_data):.2f} m")
        
        return self.dtm_data
    
    def get_terrain_elevation(self, x: float, y: float) -> float:
        """Get terrain elevation at world coordinate (x, y)."""
        if self.dtm_data is None or self.dtm_transform is None:
            return 0.0
        
        col, row = ~self.dtm_transform * (x, y)
        row, col = int(row), int(col)
        
        if row < 0 or row >= self.dtm_data.shape[0] or col < 0 or col >= self.dtm_data.shape[1]:
            return 0.0
        
        elevation = self.dtm_data[row, col]
        return 0.0 if np.isnan(elevation) else float(elevation)
    
    def create_dem_grid(self, grid_origin: np.ndarray, nx: int, ny: int
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create DEM grid aligned with voxel grid.
        
        Returns:
            (dem_relative, dem_absolute) - relative starts at 0, absolute is actual elevations.
        """
        dem_absolute = np.zeros((nx, ny), dtype=np.float32)
        
        if self.dtm_data is None:
            return dem_absolute, dem_absolute.copy()
        
        for i in range(nx):
            for j in range(ny):
                x = grid_origin[0] + (i + 0.5) * self.voxel_size
                y = grid_origin[1] + (j + 0.5) * self.voxel_size
                dem_absolute[i, j] = self.get_terrain_elevation(x, y)
        
        min_elev = np.min(dem_absolute[dem_absolute > 0]) if np.any(dem_absolute > 0) else 0
        dem_relative = dem_absolute - min_elev
        dem_relative[dem_relative < 0] = 0
        
        return dem_relative, dem_absolute
    
    def create_building_id_grid(self, grid_origin: np.ndarray, nx: int, ny: int) -> np.ndarray:
        """Create 2D grid with building IDs for each cell."""
        building_id_grid = np.zeros((nx, ny), dtype=np.int32)
        
        for bid, building in enumerate(self.buildings, start=1):
            if not building.footprint_bounds:
                continue
            
            xmin, ymin, xmax, ymax = building.footprint_bounds
            
            i_min = max(0, int((xmin - grid_origin[0]) / self.voxel_size))
            i_max = min(nx, int((xmax - grid_origin[0]) / self.voxel_size) + 1)
            j_min = max(0, int((ymin - grid_origin[1]) / self.voxel_size))
            j_max = min(ny, int((ymax - grid_origin[1]) / self.voxel_size) + 1)
            
            footprint = None
            if HAS_SHAPELY and building.footprint_polygon is None:
                building.compute_footprint_polygon()
            footprint = building.footprint_polygon
            
            if footprint is not None and not footprint.is_empty:
                prepared = prep(footprint)
                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        px = grid_origin[0] + (i + 0.5) * self.voxel_size
                        py = grid_origin[1] + (j + 0.5) * self.voxel_size
                        if prepared.contains(Point(px, py)):
                            building_id_grid[i, j] = bid
            else:
                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        building_id_grid[i, j] = bid
        
        return building_id_grid
    
    def flatten_dem_under_buildings(self, dem_grid: np.ndarray,
                                    building_id_grid: np.ndarray) -> np.ndarray:
        """
        Flatten DEM under each building by averaging elevation within footprint.
        """
        result = dem_grid.copy()
        
        unique_ids = np.unique(building_id_grid[building_id_grid != 0])
        for bid in unique_ids:
            mask = (building_id_grid == bid)
            if np.any(mask):
                result[mask] = np.mean(dem_grid[mask])
        
        return result
    
    def parse_citygml(self, filepath: str) -> List[Building]:
        """Parse CityGML file and extract building geometry."""
        print(f"Parsing CityGML: {filepath}")
        
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        building_elements = root.findall('.//{http://www.opengis.net/citygml/building/2.0}Building')
        if not building_elements:
            building_elements = root.findall('.//{http://www.opengis.net/citygml/building/1.0}Building')
        
        print(f"  Found {len(building_elements)} buildings")
        
        for building_elem in building_elements:
            building_id = building_elem.get('{http://www.opengis.net/gml}id', 'unknown')
            building = Building(id=building_id)
            
            for pos_list in building_elem.iter('{http://www.opengis.net/gml}posList'):
                if pos_list.text:
                    try:
                        vertices = parse_pos_list(pos_list.text)
                        if len(vertices) >= 3:
                            triangles = triangulate_polygon(vertices)
                            building.triangles.extend(triangles)
                    except Exception:
                        continue
            
            if building.triangles:
                building.compute_bounds()
                self.buildings.append(building)
        
        self._compute_scene_bounds()
        
        total_triangles = sum(len(b.triangles) for b in self.buildings)
        print(f"  Extracted {total_triangles} triangles from {len(self.buildings)} buildings")
        
        return self.buildings
    
    def _compute_scene_bounds(self) -> None:
        """Compute overall scene bounding box."""
        if not self.buildings:
            return
        
        all_coords = []
        for building in self.buildings:
            for v0, v1, v2 in building.triangles:
                all_coords.extend([v0, v1, v2])
        
        all_coords = np.array(all_coords)
        self.bounds_min = np.min(all_coords, axis=0)
        self.bounds_max = np.max(all_coords, axis=0)
    
    def voxelize(self,
                 add_ground: bool = True,
                 ground_thickness: int = DEFAULT_GROUND_THICKNESS,
                 use_terrain: bool = True) -> np.ndarray:
        """
        Voxelize all parsed buildings.
        
        Args:
            add_ground: Whether to add ground layer.
            ground_thickness: Number of base ground voxel layers.
            use_terrain: Whether to use loaded DTM for variable terrain.
            
        Returns:
            3D numpy array with VoxCity semantic codes.
        """
        if not self.buildings:
            raise ValueError("No buildings parsed. Call parse_citygml first.")
        
        has_terrain = use_terrain and self.dtm_data is not None
        print(f"Voxelizing with voxel_size={self.voxel_size}m, terrain={'enabled' if has_terrain else 'disabled'}...")
        
        extent = self.bounds_max - self.bounds_min
        margin = self.voxel_size * 2
        
        grid_origin = np.array([
            self.bounds_min[0] - margin,
            self.bounds_min[1] - margin,
            self.bounds_min[2]
        ])
        
        nx = int(np.ceil((extent[0] + 2 * margin) / self.voxel_size))
        ny = int(np.ceil((extent[1] + 2 * margin) / self.voxel_size))
        
        # Handle terrain
        terrain_z_extra = 0
        dem_grid_relative = None
        dem_grid_absolute = None
        
        if has_terrain:
            dem_grid_relative, dem_grid_absolute = self.create_dem_grid(grid_origin, nx, ny)
            building_id_grid = self.create_building_id_grid(grid_origin, nx, ny)
            dem_grid_absolute = self.flatten_dem_under_buildings(dem_grid_absolute, building_id_grid)
            
            min_elev = np.min(dem_grid_absolute[dem_grid_absolute > 0]) if np.any(dem_grid_absolute > 0) else 0
            self.terrain_min_elevation = min_elev
            dem_grid_relative = dem_grid_absolute - min_elev
            dem_grid_relative[dem_grid_relative < 0] = 0
            
            terrain_z_extra = int(np.ceil(np.max(dem_grid_relative) / self.voxel_size)) + 5
        
        nz = int(np.ceil((extent[2] + ground_thickness * self.voxel_size + terrain_z_extra * self.voxel_size + 5) / self.voxel_size))
        
        print(f"  Grid size: {nx} x {ny} x {nz} voxels")
        
        voxel_grid = np.zeros((nx, ny, nz), dtype=self.voxel_dtype)
        
        if add_ground and not has_terrain:
            voxel_grid[:, :, :ground_thickness] = GROUND_CODE
        
        # Collect all triangles
        all_v0 = []
        all_v1 = []
        all_v2 = []
        
        for building in self.buildings:
            for v0, v1, v2 in building.triangles:
                all_v0.append(v0)
                all_v1.append(v1)
                all_v2.append(v2)
        
        if not all_v0:
            return voxel_grid
        
        triangles_v0 = np.array(all_v0, dtype=np.float64)
        triangles_v1 = np.array(all_v1, dtype=np.float64)
        triangles_v2 = np.array(all_v2, dtype=np.float64)
        
        if NUMBA_AVAILABLE:
            voxel_grid_i32 = voxel_grid.astype(np.int32)
            
            if has_terrain:
                print("  Using Numba-accelerated voxelization with terrain...")
                _voxelize_with_terrain_kernel(
                    voxel_grid_i32,
                    triangles_v0, triangles_v1, triangles_v2,
                    grid_origin, self.voxel_size,
                    dem_grid_relative.astype(np.float32),
                    dem_grid_absolute.astype(np.float32),
                    ground_thickness
                )
            else:
                print("  Using Numba-accelerated voxelization...")
                _voxelize_building_kernel(
                    voxel_grid_i32,
                    triangles_v0, triangles_v1, triangles_v2,
                    grid_origin, self.voxel_size,
                    ground_thickness
                )
            
            voxel_grid[:] = voxel_grid_i32.astype(self.voxel_dtype)
        else:
            print("  Using Python voxelization (Numba not available)...")
            self._voxelize_python(voxel_grid, triangles_v0, triangles_v1, triangles_v2,
                                  grid_origin, ground_thickness)
        
        print(f"  Building voxels: {np.sum(voxel_grid == BUILDING_CODE)}")
        print(f"  Ground voxels: {np.sum(voxel_grid == GROUND_CODE)}")
        
        return voxel_grid
    
    def _voxelize_python(self, voxel_grid, triangles_v0, triangles_v1, triangles_v2,
                         grid_origin, ground_level):
        """Pure Python fallback voxelization."""
        nx, ny, nz = voxel_grid.shape
        num_triangles = triangles_v0.shape[0]
        
        for i in range(nx):
            if i % 50 == 0:
                print(f"    Progress: {i}/{nx}")
            
            for j in range(ny):
                px = grid_origin[0] + (i + 0.5) * self.voxel_size
                py = grid_origin[1] + (j + 0.5) * self.voxel_size
                
                z_hits = []
                
                for t in range(num_triangles):
                    v0, v1, v2 = triangles_v0[t], triangles_v1[t], triangles_v2[t]
                    
                    if (px < min(v0[0], v1[0], v2[0]) or px > max(v0[0], v1[0], v2[0]) or
                        py < min(v0[1], v1[1], v2[1]) or py > max(v0[1], v1[1], v2[1])):
                        continue
                    
                    # Simple Z-ray intersection
                    hit, t_val = self._ray_triangle_z(px, py, v0, v1, v2)
                    if hit:
                        z_hits.append(t_val)
                
                if z_hits:
                    z_hits.sort()
                    for idx in range(0, len(z_hits) - 1, 2):
                        z_min, z_max = z_hits[idx], z_hits[idx + 1]
                        k_min = max(ground_level, int((z_min - grid_origin[2]) / self.voxel_size))
                        k_max = min(nz, int((z_max - grid_origin[2]) / self.voxel_size) + 1)
                        voxel_grid[i, j, k_min:k_max] = BUILDING_CODE
    
    def _ray_triangle_z(self, px, py, v0, v1, v2):
        """Cast ray along Z and find intersection."""
        origin_z = v0[2] - 100.0
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # h = cross((0,0,1), edge2)
        h = np.array([-edge2[1], edge2[0], 0.0])
        a = np.dot(edge1, h)
        
        if abs(a) < 1e-8:
            return False, 0.0
        
        f = 1.0 / a
        s = np.array([px - v0[0], py - v0[1], origin_z - v0[2]])
        u = f * np.dot(s, h)
        
        if u < 0.0 or u > 1.0:
            return False, 0.0
        
        q = np.cross(s, edge1)
        v = f * q[2]  # dot((0,0,1), q)
        
        if v < 0.0 or u + v > 1.0:
            return False, 0.0
        
        t = f * np.dot(edge2, q)
        
        if t > 0:
            return True, origin_z + t
        
        return False, 0.0
    
    def voxelize_from_file(self, filepath: str, **kwargs) -> np.ndarray:
        """Parse and voxelize in one call."""
        self.parse_citygml(filepath)
        return self.voxelize(**kwargs)


# =============================================================================
# PLATEAU Voxelizer
# =============================================================================

class PLATEAUVoxelizer:
    """
    Voxelizer for Japanese PLATEAU CityGML data.
    
    Handles full PLATEAU data including buildings, DEM, vegetation, bridges,
    city furniture, and land use.
    
    Example::
    
        voxelizer = PLATEAUVoxelizer(voxel_size=1.0)
        voxelizer.parse_plateau_directory("path/to/plateau_data")
        voxel_grid = voxelizer.voxelize(
            include_vegetation=True,
            include_bridges=True
        )
    """
    
    def __init__(self, voxel_size: float = DEFAULT_VOXEL_SIZE, voxel_dtype=np.int8):
        """Initialize the voxelizer."""
        self.voxel_size = float(voxel_size)
        self.voxel_dtype = voxel_dtype
        self.parser: Optional[PLATEAUParser] = None
        
        self.bounds_min: Optional[np.ndarray] = None
        self.bounds_max: Optional[np.ndarray] = None
    
    def parse_plateau_directory(self, base_path: str, **kwargs) -> None:
        """
        Parse PLATEAU directory for buildings, DEM, and additional objects.
        
        Args:
            base_path: Path to PLATEAU data folder.
            **kwargs: Passed to PLATEAUParser.parse_plateau_directory().
        """
        self.parser = PLATEAUParser()
        self.parser.parse_plateau_directory(base_path, **kwargs)
        
        self.bounds_min = self.parser.bounds_min
        self.bounds_max = self.parser.bounds_max
    
    def voxelize(self,
                 add_ground: bool = True,
                 ground_thickness: int = DEFAULT_GROUND_THICKNESS,
                 include_vegetation: bool = False,
                 include_bridges: bool = False,
                 include_city_furniture: bool = False,
                 rectangle_vertices: Optional[List[Tuple[float, float]]] = None,
                 target_grid_size: Optional[Tuple[int, int]] = None,
                 external_dem_grid: Optional[np.ndarray] = None,
                 external_building_id_grid: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Voxelize all parsed PLATEAU data.
        
        Args:
            add_ground: Whether to add ground layer.
            ground_thickness: Number of base ground voxel layers.
            include_vegetation: Whether to include vegetation objects.
            include_bridges: Whether to include bridges.
            include_city_furniture: Whether to include city furniture.
            rectangle_vertices: Optional bounding rectangle [(lon, lat), ...] to 
                constrain voxelization area. If provided, grid size is based on
                this rectangle rather than all parsed data.
            target_grid_size: Optional (nx, ny) to force output grid to match
                VoxCity standard mode grid dimensions. If None, grid size is
                calculated using VoxCity's standard formula.
            external_dem_grid: Optional pre-computed DEM grid (already flattened
                under buildings). If provided, this is used instead of creating
                DEM from parsed terrain triangles. Should be in SOUTH_UP orientation
                matching VoxCity standard mode.
            external_building_id_grid: Optional pre-computed building ID grid.
                If provided along with external_dem_grid, used for terrain-relative
                building placement. Should be in SOUTH_UP orientation.
            
        Returns:
            3D numpy array with VoxCity semantic codes.
        """
        if self.parser is None or not self.parser.buildings:
            raise ValueError("No buildings parsed. Call parse_plateau_directory first.")
        
        print(f"Voxelizing PLATEAU data with voxel_size={self.voxel_size}m...")
        
        # Determine bounds and grid size
        # Use VoxCity's standard grid size calculation when rectangle_vertices is provided
        if rectangle_vertices is not None:
            # Use VoxCity's standard grid size calculation (geodetic distance)
            from pyproj import Geod
            geod = Geod(ellps="WGS84")
            
            vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]
            
            # Calculate geodetic distances (same as geoprocessor/raster/core.py)
            _, _, dist_side_1 = geod.inv(vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
            _, _, dist_side_2 = geod.inv(vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])
            
            # VoxCity standard formula: int(dist / meshsize + 0.5)
            # target_grid_size is (rows, cols) = (Y, X), so swap to (nx, ny) = (X, Y)
            if target_grid_size is not None:
                # target_grid_size is in VoxCity convention (rows, cols) = (Y, X)
                # We need (nx, ny) = (X, Y) for internal grid creation
                ny, nx = target_grid_size  # Swap: rows -> ny (Y), cols -> nx (X)
                print(f"  Using target grid size: {target_grid_size[0]} x {target_grid_size[1]} (rows x cols)")
            else:
                nx = max(1, int(dist_side_1 / self.voxel_size + 0.5))
                ny = max(1, int(dist_side_2 / self.voxel_size + 0.5))
            
            print(f"  Rectangle distances: {dist_side_1:.1f}m x {dist_side_2:.1f}m -> internal grid {nx} x {ny}")
            
            # Transform rectangle vertices to local coordinates for voxelization
            transformer = self.parser.transformer
            if transformer is None:
                raise ValueError("Coordinate transformer not initialized")
            
            rect_local = []
            for lon, lat in rectangle_vertices:
                x, y = transformer.transform(lon, lat)
                rect_local.append((x, y))
            
            rect_x = [p[0] for p in rect_local]
            rect_y = [p[1] for p in rect_local]
            
            bounds_min = np.array([min(rect_x), min(rect_y), self.bounds_min[2]])
            bounds_max = np.array([max(rect_x), max(rect_y), self.bounds_max[2]])
        else:
            bounds_min = self.bounds_min
            bounds_max = self.bounds_max
            extent = bounds_max - bounds_min
            
            if target_grid_size is not None:
                nx, ny = target_grid_size
                print(f"  Using target grid size: {nx} x {ny}")
            else:
                nx = max(1, int(extent[0] / self.voxel_size + 0.5))
                ny = max(1, int(extent[1] / self.voxel_size + 0.5))
        
        # Calculate Z extent from all parsed geometry bounds
        extent_z = self.bounds_max[2] - self.bounds_min[2]
        
        # Grid origin is at bounds_min (no margin needed since we match standard VoxCity)
        grid_origin = np.array([
            bounds_min[0],
            bounds_min[1],
            bounds_min[2]
        ])
        
        # Create DEM grid - use external DEM if provided, otherwise from terrain triangles
        dem_grid_relative = np.zeros((nx, ny), dtype=np.float32)
        dem_grid_absolute = np.zeros((nx, ny), dtype=np.float32)
        
        # CityGML reference terrain for height calculations (in CityGML's coordinate system)
        # When using external DEM, we need to estimate terrain from CityGML geometry
        # since external DEM uses different vertical datum (orthometric vs ellipsoidal)
        citygml_terrain_grid = np.zeros((nx, ny), dtype=np.float32)
        use_citygml_terrain_for_heights = False
        
        if external_dem_grid is not None:
            # Use external DEM (already flattened under buildings)
            # External DEM is in SOUTH_UP orientation, need to flip for internal processing
            print("  Using external DEM grid (pre-flattened)...")
            dem_grid_input = np.flipud(external_dem_grid).astype(np.float32)
            # Handle size mismatch by resizing
            if dem_grid_input.shape != (nx, ny):
                from scipy.ndimage import zoom
                zoom_factors = (nx / dem_grid_input.shape[0], ny / dem_grid_input.shape[1])
                dem_grid_input = zoom(dem_grid_input, zoom_factors, order=1).astype(np.float32)
                dem_grid_input = dem_grid_input[:nx, :ny]
            dem_grid_absolute = dem_grid_input.copy()
            min_elev = np.min(dem_grid_absolute[dem_grid_absolute > 0]) if np.any(dem_grid_absolute > 0) else np.min(dem_grid_absolute)
            dem_grid_relative = dem_grid_absolute - min_elev
            dem_grid_relative[dem_grid_relative < 0] = 0
            
            # When using external DEM, we need CityGML terrain estimate for height calculations
            # External DEM uses orthometric heights, CityGML uses ellipsoidal heights
            # The difference (geoid undulation) is ~30-40m in Japan
            use_citygml_terrain_for_heights = True
            
        elif self.parser.terrain_triangles:
            print("  Building DEM grid from terrain triangles...")
            dem_grid_absolute = self._create_dem_from_triangles(grid_origin, nx, ny)
            min_elev = np.min(dem_grid_absolute[dem_grid_absolute > 0]) if np.any(dem_grid_absolute > 0) else 0
            dem_grid_relative = dem_grid_absolute - min_elev
            dem_grid_relative[dem_grid_relative < 0] = 0
        
        # Create or use building footprint ID grid
        if external_building_id_grid is not None:
            print("  Using external building ID grid...")
            building_id_grid_input = np.flipud(external_building_id_grid).astype(np.int32)
            if building_id_grid_input.shape != (nx, ny):
                from scipy.ndimage import zoom
                zoom_factors = (nx / building_id_grid_input.shape[0], ny / building_id_grid_input.shape[1])
                building_id_grid_input = zoom(building_id_grid_input.astype(float), zoom_factors, order=0).astype(np.int32)
                building_id_grid_input = building_id_grid_input[:nx, :ny]
            building_id_grid = building_id_grid_input
        else:
            print("  Creating building footprint ID grid from LOD2 geometry...")
            building_id_grid = self._create_building_id_grid(grid_origin, nx, ny)
        
        # Flatten DEM under buildings if:
        # 1. Using external DEM but no external building IDs (need to flatten under LOD2 footprints)
        # 2. Using internal terrain triangles
        if external_dem_grid is not None and external_building_id_grid is None and np.any(building_id_grid > 0):
            print("  Flattening external DEM under LOD2 building footprints...")
            dem_grid_absolute = self._flatten_dem_under_buildings(dem_grid_absolute, building_id_grid)
            min_elev = np.min(dem_grid_absolute[dem_grid_absolute > 0]) if np.any(dem_grid_absolute > 0) else np.min(dem_grid_absolute)
            dem_grid_relative = dem_grid_absolute - min_elev
            dem_grid_relative[dem_grid_relative < 0] = 0
        elif external_dem_grid is None and self.parser.terrain_triangles and np.any(building_id_grid > 0):
            print("  Flattening terrain under building footprints...")
            dem_grid_absolute = self._flatten_dem_under_buildings(dem_grid_absolute, building_id_grid)
            # Recalculate relative grid after flattening
            min_elev = np.min(dem_grid_absolute[dem_grid_absolute > 0]) if np.any(dem_grid_absolute > 0) else 0
            dem_grid_relative = dem_grid_absolute - min_elev
            dem_grid_relative[dem_grid_relative < 0] = 0
        
        # Create CityGML terrain grid for height calculations when using external DEM
        # This estimates terrain elevation from CityGML building footprint minimum Z values
        if use_citygml_terrain_for_heights:
            print("  Creating CityGML terrain estimate from building footprints...")
            citygml_terrain_grid = self._create_citygml_terrain_grid(grid_origin, nx, ny)
            # Check if we have valid terrain data (non-zero count indicates data was found)
            has_terrain_data = np.any(citygml_terrain_grid != 0) or (self.parser.buildings and len(self.parser.buildings) > 0)
            if has_terrain_data:
                valid_mask = citygml_terrain_grid != 0
                if np.any(valid_mask):
                    print(f"    CityGML terrain range: {np.min(citygml_terrain_grid[valid_mask]):.1f}m - {np.max(citygml_terrain_grid):.1f}m")
                else:
                    print(f"    CityGML terrain reference: {self.bounds_min[2]:.1f}m (from geometry bounds)")
            else:
                print("    Warning: No CityGML terrain data available, using bounds_min[2] as reference")
                citygml_terrain_grid[:] = self.bounds_min[2]
        else:
            # When using CityGML terrain, dem_grid_absolute is already in CityGML coordinates
            citygml_terrain_grid = dem_grid_absolute.copy()
        
        terrain_z_extra = int(np.ceil(np.max(dem_grid_relative) / self.voxel_size)) + 5
        nz = int(np.ceil((extent_z + ground_thickness * self.voxel_size + terrain_z_extra * self.voxel_size + 10) / self.voxel_size))
        
        print(f"  Grid size: {nx} x {ny} x {nz} voxels")
        
        voxel_grid = np.zeros((nx, ny, nz), dtype=self.voxel_dtype)
        
        # Add ground layer with terrain variation
        if add_ground:
            for i in range(nx):
                for j in range(ny):
                    local_ground = ground_thickness + int(dem_grid_relative[i, j] / self.voxel_size)
                    voxel_grid[i, j, :local_ground] = GROUND_CODE
        
        # Voxelize buildings (surface + interior fill)
        print("  Voxelizing buildings...")
        self._voxelize_buildings(voxel_grid, grid_origin, dem_grid_relative, citygml_terrain_grid, ground_thickness, building_id_grid)
        
        # Voxelize additional objects (terrain-relative using CityGML terrain estimate)
        if include_vegetation and self.parser.vegetation:
            print(f"  Voxelizing {len(self.parser.vegetation)} vegetation objects...")
            self._voxelize_objects_terrain_relative(
                voxel_grid, grid_origin, self.parser.vegetation, TREE_CODE,
                dem_grid_relative, citygml_terrain_grid, ground_thickness
            )
        
        if include_bridges and self.parser.bridges:
            print(f"  Voxelizing {len(self.parser.bridges)} bridges...")
            self._voxelize_objects_terrain_relative(
                voxel_grid, grid_origin, self.parser.bridges, BRIDGE_CODE,
                dem_grid_relative, citygml_terrain_grid, ground_thickness
            )
        
        if include_city_furniture and self.parser.city_furniture:
            print(f"  Voxelizing {len(self.parser.city_furniture)} city furniture...")
            self._voxelize_objects_terrain_relative(
                voxel_grid, grid_origin, self.parser.city_furniture, CITY_FURNITURE_CODE,
                dem_grid_relative, citygml_terrain_grid, ground_thickness
            )
        
        print(f"  Building voxels: {np.sum(voxel_grid == BUILDING_CODE)}")
        print(f"  Ground voxels: {np.sum(voxel_grid == GROUND_CODE)}")
        if include_vegetation:
            print(f"  Vegetation voxels: {np.sum(voxel_grid == TREE_CODE)}")
        
        # Apply -90 degree clockwise rotation around Z axis to align with VoxCity standard mode
        # This corrects the coordinate system mismatch between LOD2 CityGML and standard mode
        voxel_grid = np.rot90(voxel_grid, k=1, axes=(0, 1))
        print(f"  Applied coordinate rotation (-90° clockwise around Z)")
        
        return voxel_grid
    
    def _create_dem_from_triangles(self, grid_origin: np.ndarray, nx: int, ny: int) -> np.ndarray:
        """
        Create DEM grid from parsed terrain triangles using KDTree for fast lookup.
        
        Following the reference implementation, uses triangle centroids with their
        average elevation for nearest-neighbor interpolation.
        
        Args:
            grid_origin: (x, y, z) origin of the voxel grid.
            nx: Number of cells in X direction.
            ny: Number of cells in Y direction.
            
        Returns:
            2D array of absolute terrain elevations.
        """
        dem_grid = np.zeros((nx, ny), dtype=np.float32)
        
        if not self.parser.terrain_triangles:
            return dem_grid
        
        # Try to use KDTree for fast lookup (like reference implementation)
        try:
            from scipy.spatial import cKDTree
            
            # Extract centroids and elevations from terrain triangles
            centroids = np.array([[tri.centroid[0], tri.centroid[1]] 
                                   for tri in self.parser.terrain_triangles])
            elevations = np.array([tri.elevation for tri in self.parser.terrain_triangles])
            
            # Build KDTree
            kdtree = cKDTree(centroids)
            
            # Create grid of query points (cell centers)
            x_coords = grid_origin[0] + (np.arange(nx) + 0.5) * self.voxel_size
            y_coords = grid_origin[1] + (np.arange(ny) + 0.5) * self.voxel_size
            
            # Create meshgrid of all query points
            xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
            query_points = np.column_stack([xx.ravel(), yy.ravel()])
            
            # Query all points at once (very fast)
            _, indices = kdtree.query(query_points, k=1)
            
            # Reshape results back to grid
            dem_grid = elevations[indices].reshape(nx, ny).astype(np.float32)
            
            print(f"    DEM created using KDTree ({len(self.parser.terrain_triangles)} triangles)")
            
        except ImportError:
            # Fallback to slow method using centroids
            print("    Warning: scipy not available, using slow terrain lookup")
            for i in range(nx):
                for j in range(ny):
                    x = grid_origin[0] + (i + 0.5) * self.voxel_size
                    y = grid_origin[1] + (j + 0.5) * self.voxel_size
                    
                    # Find nearest terrain triangle centroid
                    min_dist = float('inf')
                    nearest_elev = 0.0
                    
                    for tri in self.parser.terrain_triangles:
                        dx = tri.centroid[0] - x
                        dy = tri.centroid[1] - y
                        dist = dx * dx + dy * dy
                        if dist < min_dist:
                            min_dist = dist
                            nearest_elev = tri.elevation
                    
                    dem_grid[i, j] = nearest_elev
        
        return dem_grid
    
    def _flatten_dem_under_buildings(self, dem_grid: np.ndarray, building_id_grid: np.ndarray) -> np.ndarray:
        """
        Flatten DEM under each building by averaging elevation within footprint.
        
        Following VoxCity's approach: each building gets a uniform terrain elevation
        equal to the average of all cells within its footprint. This prevents
        building shape distortion due to terrain variation under a single building.
        
        Args:
            dem_grid: 2D array of terrain elevations.
            building_id_grid: 2D array of building IDs (0 = no building).
            
        Returns:
            Modified DEM grid with flattened terrain under buildings.
        """
        result = dem_grid.copy()
        
        if not np.any(building_id_grid != 0):
            return result
        
        # Get unique building IDs
        unique_ids = np.unique(building_id_grid[building_id_grid != 0])
        
        for bid in unique_ids:
            mask = (building_id_grid == bid)
            if np.any(mask):
                avg_elevation = np.mean(dem_grid[mask])
                result[mask] = avg_elevation
        
        print(f"    Flattened terrain under {len(unique_ids)} buildings")
        
        return result

    def _create_building_id_grid(self, grid_origin: np.ndarray, nx: int, ny: int) -> np.ndarray:
        """
        Create 2D grid marking which cells are within building footprints.
        
        Uses building bounding boxes or Shapely polygon footprints when available
        for more accurate footprint detection.
        
        Args:
            grid_origin: Origin of the voxel grid in local coordinates.
            nx: Number of cells in X direction.
            ny: Number of cells in Y direction.
            
        Returns:
            2D numpy array with building IDs (0 = no building, 1+ = building ID).
        """
        building_id_grid = np.zeros((nx, ny), dtype=np.int32)
        
        if not self.parser.buildings:
            return building_id_grid
        
        use_shapely = HAS_SHAPELY
        buildings_with_polygons = 0
        buildings_with_bbox_only = 0
        
        for bid, building in enumerate(self.parser.buildings, start=1):
            # Get building bounds from triangles
            if not building.triangles:
                continue
            
            # Collect all vertices
            all_verts = []
            for v0, v1, v2 in building.triangles:
                all_verts.extend([v0, v1, v2])
            all_verts = np.array(all_verts)
            
            xmin, ymin = np.min(all_verts[:, :2], axis=0)
            xmax, ymax = np.max(all_verts[:, :2], axis=0)
            
            # Convert to grid indices
            i_min = max(0, int((xmin - grid_origin[0]) / self.voxel_size))
            i_max = min(nx, int((xmax - grid_origin[0]) / self.voxel_size) + 1)
            j_min = max(0, int((ymin - grid_origin[1]) / self.voxel_size))
            j_max = min(ny, int((ymax - grid_origin[1]) / self.voxel_size) + 1)
            
            # Try to create polygon footprint for more accurate detection
            footprint = None
            if use_shapely:
                try:
                    from shapely.geometry import Polygon as ShapelyPolygon
                    from shapely.ops import unary_union
                    from shapely.validation import make_valid
                    
                    # Project triangles to XY and union them
                    tri_polys = []
                    for v0, v1, v2 in building.triangles:
                        coords = [(v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1])]
                        if len(set(coords)) >= 3:
                            try:
                                p = ShapelyPolygon(coords)
                                if p.is_valid and p.area > 1e-10:
                                    tri_polys.append(p)
                                elif not p.is_valid:
                                    p = make_valid(p)
                                    if p.area > 1e-10:
                                        tri_polys.append(p)
                            except Exception:
                                pass
                    
                    if tri_polys:
                        footprint = unary_union(tri_polys)
                        if footprint.is_empty:
                            footprint = None
                except Exception:
                    footprint = None
            
            if footprint is not None:
                buildings_with_polygons += 1
                try:
                    from shapely.prepared import prep
                    prepared = prep(footprint)
                    
                    for i in range(i_min, i_max):
                        for j in range(j_min, j_max):
                            px = grid_origin[0] + (i + 0.5) * self.voxel_size
                            py = grid_origin[1] + (j + 0.5) * self.voxel_size
                            
                            point = Point(px, py)
                            if prepared.contains(point):
                                building_id_grid[i, j] = bid
                except Exception:
                    # Fall back to bounding box
                    for i in range(i_min, i_max):
                        for j in range(j_min, j_max):
                            building_id_grid[i, j] = bid
                    buildings_with_bbox_only += 1
            else:
                buildings_with_bbox_only += 1
                # Fall back to bounding box
                for i in range(i_min, i_max):
                    for j in range(j_min, j_max):
                        building_id_grid[i, j] = bid
        
        print(f"    Building footprints: {buildings_with_polygons} polygon, {buildings_with_bbox_only} bbox")
        print(f"    Cells with buildings: {np.sum(building_id_grid > 0)}")
        
        return building_id_grid
    
    def _create_citygml_terrain_grid(self, grid_origin: np.ndarray, nx: int, ny: int) -> np.ndarray:
        """
        Estimate terrain elevation grid from CityGML building footprint minimum Z values.
        
        When using external DEM (orthometric heights) with CityGML geometry (ellipsoidal
        heights), we need terrain elevations in the CityGML coordinate system for correct
        height calculations. This method estimates terrain from building footprint 
        minimum Z values.
        
        Args:
            grid_origin: Origin of the voxel grid in local coordinates.
            nx: Number of cells in X direction.
            ny: Number of cells in Y direction.
            
        Returns:
            2D numpy array of estimated terrain elevations in CityGML coordinates.
        """
        terrain_grid = np.zeros((nx, ny), dtype=np.float32)
        count_grid = np.zeros((nx, ny), dtype=np.int32)
        
        if not self.parser.buildings:
            # Use bounds_min[2] as fallback
            terrain_grid[:] = self.bounds_min[2] if self.bounds_min is not None else 0
            return terrain_grid
        
        # For each building, compute minimum Z (ground level) at each cell
        for building in self.parser.buildings:
            if not building.triangles:
                continue
            
            # Collect all vertices
            all_verts = []
            for v0, v1, v2 in building.triangles:
                all_verts.extend([v0, v1, v2])
            all_verts = np.array(all_verts)
            
            xmin, ymin = np.min(all_verts[:, :2], axis=0)
            xmax, ymax = np.max(all_verts[:, :2], axis=0)
            min_z = np.min(all_verts[:, 2])  # Building footprint ground level
            
            # Convert to grid indices
            i_min = max(0, int((xmin - grid_origin[0]) / self.voxel_size))
            i_max = min(nx, int((xmax - grid_origin[0]) / self.voxel_size) + 1)
            j_min = max(0, int((ymin - grid_origin[1]) / self.voxel_size))
            j_max = min(ny, int((ymax - grid_origin[1]) / self.voxel_size) + 1)
            
            # Assign minimum Z to cells (accumulate for averaging)
            for i in range(i_min, i_max):
                for j in range(j_min, j_max):
                    terrain_grid[i, j] += min_z
                    count_grid[i, j] += 1
        
        # Average where we have data
        mask = count_grid > 0
        terrain_grid[mask] /= count_grid[mask]
        
        # Fill cells without building data using global minimum
        if np.any(~mask):
            global_min_z = self.bounds_min[2] if self.bounds_min is not None else np.min(terrain_grid[mask])
            terrain_grid[~mask] = global_min_z
        
        return terrain_grid

    def _voxelize_buildings(self, voxel_grid, grid_origin, dem_relative, citygml_terrain, ground_thickness, building_id_grid):
        """
        Voxelize buildings with terrain-aware placement.
        
        Uses a two-step approach following the reference implementation:
        1. Surface voxelization using triangle-AABB intersection
        2. Footprint-guided fill to make buildings solid from roof to ground
        
        Args:
            citygml_terrain: Terrain elevations in CityGML coordinate system (ellipsoidal heights).
                            Used to compute building heights above terrain consistently.
        """
        all_v0, all_v1, all_v2 = [], [], []
        
        for building in self.parser.buildings:
            for v0, v1, v2 in building.triangles:
                all_v0.append(v0)
                all_v1.append(v1)
                all_v2.append(v2)
        
        if not all_v0:
            return
        
        triangles_v0 = np.array(all_v0, dtype=np.float64)
        triangles_v1 = np.array(all_v1, dtype=np.float64)
        triangles_v2 = np.array(all_v2, dtype=np.float64)
        
        if NUMBA_AVAILABLE:
            voxel_grid_i32 = voxel_grid.astype(np.int32)
            
            # Step 1: Surface voxelization using triangle-AABB intersection
            print("    Step 1: Surface voxelization (triangle-AABB)...")
            _voxelize_surface_kernel(
                voxel_grid_i32,
                triangles_v0, triangles_v1, triangles_v2,
                grid_origin, self.voxel_size,
                dem_relative.astype(np.float32),
                citygml_terrain.astype(np.float32),
                ground_thickness,
                building_id_grid.astype(np.int32)
            )
            
            # Step 2: Fill building interiors from roof to ground
            print("    Step 2: Filling building interiors (roof-to-ground)...")
            _fill_building_columns_guided(
                voxel_grid_i32,
                ground_thickness,
                dem_relative.astype(np.float32),
                building_id_grid.astype(np.int32),
                self.voxel_size
            )
            
            voxel_grid[:] = voxel_grid_i32.astype(self.voxel_dtype)
    
    def _voxelize_objects(self, voxel_grid, grid_origin, objects, voxel_code):
        """Voxelize a list of objects (vegetation, bridges, etc.)."""
        all_v0, all_v1, all_v2 = [], [], []
        
        for obj in objects:
            for v0, v1, v2 in obj.triangles:
                all_v0.append(v0)
                all_v1.append(v1)
                all_v2.append(v2)
        
        if not all_v0:
            return
        
        triangles_v0 = np.array(all_v0, dtype=np.float64)
        triangles_v1 = np.array(all_v1, dtype=np.float64)
        triangles_v2 = np.array(all_v2, dtype=np.float64)
        
        if NUMBA_AVAILABLE:
            voxel_grid_i32 = voxel_grid.astype(np.int32)
            _voxelize_objects_kernel(
                voxel_grid_i32,
                triangles_v0, triangles_v1, triangles_v2,
                grid_origin, self.voxel_size,
                voxel_code
            )
            voxel_grid[:] = voxel_grid_i32.astype(self.voxel_dtype)
    
    def _voxelize_objects_terrain_relative(self, voxel_grid, grid_origin, objects, voxel_code,
                                           dem_relative, citygml_terrain, ground_thickness):
        """
        Voxelize objects (bridges, vegetation, etc.) with terrain-relative placement.
        
        Unlike buildings that are processed per-cell, objects like bridges span multiple
        cells. We compute the average terrain height across the object's footprint and
        use that as the reference for height calculation.
        
        This ensures objects are placed at correct heights relative to the normalized
        DEM grid, even when CityGML uses different vertical datum than external DEM.
        
        Args:
            voxel_grid: 3D voxel grid to fill.
            grid_origin: Grid origin in world coordinates.
            objects: List of object data to voxelize.
            voxel_code: Voxel code to assign.
            dem_relative: Relative DEM grid (normalized to start from 0).
            citygml_terrain: Terrain elevations in CityGML coordinate system.
            ground_thickness: Base ground layer thickness.
        """
        if not objects:
            return
        
        nx, ny, nz = voxel_grid.shape
        
        for obj in objects:
            if not obj.triangles:
                continue
            
            # Collect all vertices to find object footprint and Z range
            all_verts = []
            for v0, v1, v2 in obj.triangles:
                all_verts.extend([v0, v1, v2])
            all_verts = np.array(all_verts)
            
            # Object bounding box
            xmin, ymin = np.min(all_verts[:, :2], axis=0)
            xmax, ymax = np.max(all_verts[:, :2], axis=0)
            zmin_citygml = np.min(all_verts[:, 2])  # Minimum Z in CityGML coords
            
            # Convert to grid indices
            i_min = max(0, int((xmin - grid_origin[0]) / self.voxel_size))
            i_max = min(nx - 1, int((xmax - grid_origin[0]) / self.voxel_size))
            j_min = max(0, int((ymin - grid_origin[1]) / self.voxel_size))
            j_max = min(ny - 1, int((ymax - grid_origin[1]) / self.voxel_size))
            
            # Compute average terrain height under object footprint
            # Use CityGML terrain for height reference
            citygml_terrain_under_obj = []
            dem_relative_under_obj = []
            for i in range(i_min, i_max + 1):
                for j in range(j_min, j_max + 1):
                    if citygml_terrain[i, j] > 0:
                        citygml_terrain_under_obj.append(citygml_terrain[i, j])
                    dem_relative_under_obj.append(dem_relative[i, j])
            
            # Reference terrain elevation from CityGML
            if citygml_terrain_under_obj:
                ref_terrain_citygml = np.mean(citygml_terrain_under_obj)
            else:
                # Fallback: use object minimum Z as ground reference
                ref_terrain_citygml = zmin_citygml
            
            # Average normalized DEM height under object
            if dem_relative_under_obj:
                avg_dem_relative = np.mean(dem_relative_under_obj)
            else:
                avg_dem_relative = 0
            
            # Object's minimum height above CityGML terrain
            height_above_citygml_terrain = zmin_citygml - ref_terrain_citygml
            
            # Target ground level in voxel grid
            avg_ground_level = ground_thickness + int(avg_dem_relative / self.voxel_size)
            
            # Voxelize each triangle with terrain-relative offset
            for v0, v1, v2 in obj.triangles:
                # Compute height above CityGML terrain for each vertex
                v0_height = v0[2] - ref_terrain_citygml
                v1_height = v1[2] - ref_terrain_citygml
                v2_height = v2[2] - ref_terrain_citygml
                
                # Target K indices (above ground level)
                k0 = avg_ground_level + int(v0_height / self.voxel_size)
                k1 = avg_ground_level + int(v1_height / self.voxel_size)
                k2 = avg_ground_level + int(v2_height / self.voxel_size)
                
                k_min_tri = max(0, min(k0, k1, k2) - 1)
                k_max_tri = min(nz - 1, max(k0, k1, k2) + 1)
                
                if k_max_tri < 0 or k_min_tri >= nz:
                    continue
                
                # Triangle XY bounds
                tri_i_min = max(0, int((min(v0[0], v1[0], v2[0]) - grid_origin[0]) / self.voxel_size) - 1)
                tri_i_max = min(nx - 1, int((max(v0[0], v1[0], v2[0]) - grid_origin[0]) / self.voxel_size) + 1)
                tri_j_min = max(0, int((min(v0[1], v1[1], v2[1]) - grid_origin[1]) / self.voxel_size) - 1)
                tri_j_max = min(ny - 1, int((max(v0[1], v1[1], v2[1]) - grid_origin[1]) / self.voxel_size) + 1)
                
                # Create adjusted vertices for intersection test
                v0_adj = np.array([v0[0], v0[1], grid_origin[2] + k0 * self.voxel_size])
                v1_adj = np.array([v1[0], v1[1], grid_origin[2] + k1 * self.voxel_size])
                v2_adj = np.array([v2[0], v2[1], grid_origin[2] + k2 * self.voxel_size])
                
                half_size = self.voxel_size / 2.0
                box_halfsize = np.array([half_size, half_size, half_size])
                
                for i in range(tri_i_min, tri_i_max + 1):
                    for j in range(tri_j_min, tri_j_max + 1):
                        for k in range(k_min_tri, k_max_tri + 1):
                            if voxel_grid[i, j, k] != 0:
                                continue
                            
                            voxel_center = np.array([
                                grid_origin[0] + (i + 0.5) * self.voxel_size,
                                grid_origin[1] + (j + 0.5) * self.voxel_size,
                                grid_origin[2] + (k + 0.5) * self.voxel_size
                            ])
                            
                            if triangle_aabb_intersection(v0_adj, v1_adj, v2_adj, voxel_center, box_halfsize):
                                voxel_grid[i, j, k] = voxel_code


# =============================================================================
# Convenience Functions
# =============================================================================

def parse_citygml_subset(filepath: str,
                         rectangle_vertices: Optional[List[Tuple[float, float]]] = None,
                         voxel_size: float = 1.0) -> np.ndarray:
    """
    Parse and voxelize a CityGML file with optional spatial filtering.
    
    Args:
        filepath: Path to GML file.
        rectangle_vertices: Optional bounding rectangle for filtering.
        voxel_size: Voxel size in meters.
        
    Returns:
        3D voxel grid.
    """
    voxelizer = CityGMLVoxelizer(voxel_size=voxel_size)
    voxelizer.parse_citygml(filepath)
    return voxelizer.voxelize()


# =============================================================================
# CityGML Pipeline Helper Functions
# =============================================================================

def resolve_citygml_path(url_citygml, citygml_path, output_dir, ssl_verify=True, ca_bundle=None, timeout=60):
    """
    Resolve CityGML path from URL or local path.
    
    Args:
        url_citygml: URL to download CityGML data from.
        citygml_path: Local path to CityGML directory.
        output_dir: Directory for extracted files.
        ssl_verify: Whether to verify SSL certificates.
        ca_bundle: Custom CA bundle path.
        timeout: Download timeout in seconds.
        
    Returns:
        Resolved path to CityGML directory.
        
    Raises:
        ValueError: If neither url_citygml nor citygml_path is specified.
    """
    from ...downloader.citygml import download_and_extract_zip
    
    if url_citygml:
        citygml_path_resolved, foldername = download_and_extract_zip(
            url_citygml, extract_to=output_dir,
            ssl_verify=ssl_verify, ca_bundle=ca_bundle, timeout=timeout
        )
        # Check for nested folder structure
        udx_path = os.path.join(citygml_path_resolved, 'udx')
        if not os.path.exists(udx_path):
            udx_path_2 = os.path.join(citygml_path_resolved, foldername, 'udx')
            if os.path.exists(udx_path_2):
                citygml_path_resolved = os.path.join(citygml_path_resolved, foldername)
        return citygml_path_resolved
    elif citygml_path:
        # Check for nested folder structure  
        udx_path = os.path.join(citygml_path, 'udx')
        if not os.path.exists(udx_path):
            foldername = os.path.basename(citygml_path)
            udx_path_2 = os.path.join(citygml_path, foldername, 'udx')
            if os.path.exists(udx_path_2):
                return os.path.join(citygml_path, foldername)
        return citygml_path
    else:
        raise ValueError("Either url_citygml or citygml_path must be specified")


def voxelize_buildings_citygml(citygml_path_resolved, rectangle_vertices, meshsize, 
                                use_lod2, include_bridges, include_city_furniture,
                                grid_vis, **kwargs):
    """
    Step 1: Building, bridge, and furniture voxelization.
    
    Options:
    - LOD1: Footprint-based from GeoDataFrames
    - LOD2: Triangulated geometry from CityGML
    
    Args:
        citygml_path_resolved: Path to CityGML directory.
        rectangle_vertices: Bounding rectangle vertices.
        meshsize: Voxel size in meters.
        use_lod2: Whether to use LOD2 triangulated geometry.
        include_bridges: Whether to include bridges (LOD2 mode).
        include_city_furniture: Whether to include city furniture (LOD2 mode).
        grid_vis: Whether to visualize grids.
        **kwargs: Additional options.
        
    Returns:
        Dictionary containing:
        - use_lod2: Whether LOD2 mode is actually used (may fall back to LOD1)
        - building_gdf: Building GeoDataFrame
        - building_height_grid: Building height grid
        - building_min_height_grid: Building minimum height grid
        - building_id_grid: Building ID grid
        - lod2_voxelizer: PLATEAUVoxelizer instance (if LOD2 mode)
    """
    import geopandas as gpd
    from .parsers import load_lod1_citygml
    from ...geoprocessor.raster import create_building_height_grid_from_gdf_polygon
    from ...visualizer.grids import visualize_numerical_grid
    
    lod2_voxelizer = None
    
    if use_lod2:
        print(f"  Mode: LOD2 (triangulated geometry)")
        print(f"  Include bridges: {include_bridges}")
        print(f"  Include city furniture: {include_city_furniture}")
        
        lod2_voxelizer = PLATEAUVoxelizer(voxel_size=meshsize)
        lod2_voxelizer.parse_plateau_directory(
            base_path=citygml_path_resolved,
            rectangle_vertices=rectangle_vertices,
            parse_buildings=True,
            parse_dem=False,
            parse_vegetation=False,
            parse_bridges=include_bridges,
            parse_city_furniture=include_city_furniture,
            parse_land_use=False,
        )
        
        if lod2_voxelizer.parser is None or len(lod2_voxelizer.parser.buildings) == 0:
            print("  Warning: No LOD2 buildings found. Falling back to LOD1 mode.")
            use_lod2 = False
            lod2_voxelizer = None
        else:
            print(f"  Found {len(lod2_voxelizer.parser.buildings)} LOD2 buildings")
            if include_bridges and lod2_voxelizer.parser.bridges:
                print(f"  Found {len(lod2_voxelizer.parser.bridges)} bridges")
            if include_city_furniture and lod2_voxelizer.parser.city_furniture:
                print(f"  Found {len(lod2_voxelizer.parser.city_furniture)} city furniture items")
    
    # Parse LOD1 buildings for height grids (needed even in LOD2 mode for compatibility)
    print("  Parsing LOD1 building footprints for height grids...")
    building_gdf, _, _ = load_lod1_citygml(
        citygml_path=citygml_path_resolved,
        rectangle_vertices=rectangle_vertices,
        parse_buildings=not use_lod2,
        parse_terrain=False,
        parse_vegetation=False,
    )
    
    # Ensure CRS is EPSG:4326
    if building_gdf is None and use_lod2:
        building_gdf = gpd.GeoDataFrame(
            columns=['building_id', 'height', 'storeys', 'ground_elevation', 'geometry', 'source_file', 'id'],
            geometry='geometry', crs='EPSG:4326'
        )
    elif building_gdf is not None:
        if building_gdf.crs is None:
            building_gdf = building_gdf.set_crs(epsg=4326)
        elif getattr(building_gdf.crs, 'to_epsg', lambda: None)() != 4326:
            building_gdf = building_gdf.to_crs(epsg=4326)
    
    # Create building height grids
    print("  Creating building height grid")
    building_height_grid, building_min_height_grid, building_id_grid, _ = create_building_height_grid_from_gdf_polygon(
        building_gdf, meshsize, rectangle_vertices
    )
    
    if grid_vis and not use_lod2:
        building_height_grid_nan = building_height_grid.copy()
        building_height_grid_nan[building_height_grid_nan == 0] = np.nan
        visualize_numerical_grid(np.flipud(building_height_grid_nan), meshsize, "building height (m)", cmap='viridis', label='Value')
    
    return {
        'use_lod2': use_lod2,
        'building_gdf': building_gdf,
        'building_height_grid': building_height_grid,
        'building_min_height_grid': building_min_height_grid,
        'building_id_grid': building_id_grid,
        'lod2_voxelizer': lod2_voxelizer,
    }


def voxelize_trees_citygml(citygml_path_resolved, rectangle_vertices, meshsize,
                           land_cover_source, canopy_height_source, use_lod2,
                           include_lod2_vegetation, trunk_height_ratio, output_dir, **kwargs):
    """
    Step 2: Tree voxelization.
    
    Options:
    - LOD1: From CityGML vegetation GeoDataFrames
    - LOD2: From CityGML triangulated vegetation (if include_lod2_vegetation=True)
    - External: From canopy_height_source (GEE, static, etc.)
    
    Args:
        citygml_path_resolved: Path to CityGML directory.
        rectangle_vertices: Bounding rectangle vertices.
        meshsize: Voxel size in meters.
        land_cover_source: Land cover data source.
        canopy_height_source: Canopy height data source.
        use_lod2: Whether LOD2 mode is active.
        include_lod2_vegetation: Whether to use LOD2 vegetation geometry.
        trunk_height_ratio: Ratio of trunk height to total tree height.
        output_dir: Output directory.
        **kwargs: Additional options.
        
    Returns:
        Dictionary containing:
        - canopy_height_grid: Canopy height grid
        - canopy_bottom_height_grid: Canopy bottom height grid
    """
    from .parsers import load_lod1_citygml
    from ...generator.grids import get_land_cover_grid, get_canopy_height_grid
    from ...geoprocessor.raster import create_vegetation_height_grid_from_gdf_polygon
    from ...utils.lc import get_land_cover_classes
    
    # Get CityGML vegetation (LOD1)
    print("  Parsing CityGML vegetation data...")
    _, _, vegetation_gdf = load_lod1_citygml(
        citygml_path=citygml_path_resolved,
        rectangle_vertices=rectangle_vertices,
        parse_buildings=False,
        parse_terrain=False,
        parse_vegetation=True,
    )
    
    if vegetation_gdf is not None:
        if vegetation_gdf.crs is None:
            vegetation_gdf = vegetation_gdf.set_crs(epsg=4326)
        elif getattr(vegetation_gdf.crs, 'to_epsg', lambda: None)() != 4326:
            vegetation_gdf = vegetation_gdf.to_crs(epsg=4326)
    
    # Get external canopy height source (complementary)
    if canopy_height_source == "Static":
        print(f"  Using static tree height")
        static_tree_height = kwargs.get("static_tree_height", 10.0)
        land_cover_grid_temp = get_land_cover_grid(rectangle_vertices, meshsize, land_cover_source, output_dir, **kwargs)
        canopy_height_grid_comp = np.zeros_like(land_cover_grid_temp, dtype=float)
        
        _classes = get_land_cover_classes(land_cover_source)
        _class_to_int = {name: i for i, name in enumerate(_classes.values())}
        _tree_labels = ["Tree", "Trees", "Tree Canopy"]
        _tree_indices = [_class_to_int[label] for label in _tree_labels if label in _class_to_int]
        tree_mask = np.isin(land_cover_grid_temp, _tree_indices) if _tree_indices else np.zeros_like(land_cover_grid_temp, dtype=bool)
        canopy_height_grid_comp[tree_mask] = static_tree_height
        canopy_bottom_height_grid_comp = canopy_height_grid_comp * float(trunk_height_ratio)
    else:
        print(f"  Using external canopy height source: {canopy_height_source}")
        canopy_height_grid_comp, canopy_bottom_height_grid_comp = get_canopy_height_grid(
            rectangle_vertices, meshsize, canopy_height_source, output_dir, **kwargs
        )
    
    # Create canopy grids from CityGML vegetation
    if vegetation_gdf is not None and len(vegetation_gdf) > 0:
        print(f"  Found {len(vegetation_gdf)} CityGML vegetation objects")
        canopy_height_grid = create_vegetation_height_grid_from_gdf_polygon(vegetation_gdf, meshsize, rectangle_vertices)
        canopy_bottom_height_grid = canopy_height_grid * float(trunk_height_ratio)
    else:
        print("  No CityGML vegetation found")
        canopy_height_grid = np.zeros_like(canopy_height_grid_comp)
        canopy_bottom_height_grid = np.zeros_like(canopy_height_grid_comp)
    
    # Merge: use external source where CityGML has no data
    mask = (canopy_height_grid == 0) & (canopy_height_grid_comp != 0)
    canopy_height_grid[mask] = canopy_height_grid_comp[mask]
    mask_b = (canopy_bottom_height_grid == 0) & (canopy_bottom_height_grid_comp != 0)
    canopy_bottom_height_grid[mask_b] = canopy_bottom_height_grid_comp[mask_b]
    canopy_bottom_height_grid = np.minimum(canopy_bottom_height_grid, canopy_height_grid)
    
    print(f"  Canopy coverage: {np.sum(canopy_height_grid > 0)} cells")
    
    return {
        'canopy_height_grid': canopy_height_grid,
        'canopy_bottom_height_grid': canopy_bottom_height_grid,
    }


def voxelize_terrain_citygml(citygml_path_resolved, rectangle_vertices, meshsize,
                              land_cover_source, dem_source, building_id_grid,
                              grid_vis, output_dir, **kwargs):
    """
    Step 3: Terrain and land cover voxelization.
    
    Options:
    - CityGML terrain: From DEM triangulated geometry
    - External DEM: From GEE-based sources or local files
    
    Terrain is flattened under building footprints.
    
    Args:
        citygml_path_resolved: Path to CityGML directory.
        rectangle_vertices: Bounding rectangle vertices.
        meshsize: Voxel size in meters.
        land_cover_source: Land cover data source.
        dem_source: DEM data source (None for CityGML terrain).
        building_id_grid: Building ID grid for terrain flattening.
        grid_vis: Whether to visualize grids.
        output_dir: Output directory.
        **kwargs: Additional options.
        
    Returns:
        Dictionary containing:
        - dem_grid: DEM grid
        - land_cover_grid: Land cover grid
    """
    from .parsers import load_lod1_citygml
    from ...generator.grids import get_land_cover_grid, get_dem_grid
    from ...geoprocessor.raster import process_grid, create_dem_grid_from_gdf_polygon
    from ...utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP
    from ...visualizer.grids import visualize_numerical_grid
    
    # Get land cover grid
    print("  Creating land cover grid...")
    land_cover_grid = get_land_cover_grid(rectangle_vertices, meshsize, land_cover_source, output_dir, **kwargs)
    
    # Get DEM grid
    if kwargs.get('flat_dem', False):
        print("  Using flat DEM")
        dem_grid = np.zeros_like(land_cover_grid, dtype=float)
    elif dem_source is not None:
        print(f"  Using external DEM source: {dem_source}")
        dem_grid = get_dem_grid(rectangle_vertices, meshsize, dem_source, output_dir, **kwargs)
    else:
        print("  Using CityGML terrain data")
        _, terrain_gdf, _ = load_lod1_citygml(
            citygml_path=citygml_path_resolved,
            rectangle_vertices=rectangle_vertices,
            parse_buildings=False,
            parse_terrain=True,
            parse_vegetation=False,
        )
        
        if terrain_gdf is not None:
            if terrain_gdf.crs is None:
                terrain_gdf = terrain_gdf.set_crs(epsg=4326)
            elif getattr(terrain_gdf.crs, 'to_epsg', lambda: None)() != 4326:
                terrain_gdf = terrain_gdf.to_crs(epsg=4326)
        
        dem_grid = create_dem_grid_from_gdf_polygon(terrain_gdf, meshsize, rectangle_vertices)
        
        if dem_grid is None or (np.all(dem_grid == 0) and terrain_gdf is None):
            print("  Warning: No terrain data found in CityGML. Using flat DEM.")
            dem_grid = np.zeros_like(land_cover_grid, dtype=float)
    
    # Flatten terrain under buildings (standard VoxCity approach)
    # This is done by process_grid which averages DEM under each building footprint
    print("  Flattening terrain under building footprints...")
    dem_grid_oriented = ensure_orientation(dem_grid.copy(), ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)
    building_id_grid_oriented = ensure_orientation(building_id_grid.copy(), ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)
    dem_grid_flattened = process_grid(building_id_grid_oriented, dem_grid_oriented - np.min(dem_grid_oriented))
    # Restore original orientation and offset
    dem_grid = ensure_orientation(dem_grid_flattened + np.min(dem_grid), ORIENTATION_SOUTH_UP, ORIENTATION_NORTH_UP)
    
    if grid_vis and dem_grid is not None:
        visualize_numerical_grid(np.flipud(dem_grid), meshsize, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')
    
    print(f"  DEM range: {np.min(dem_grid):.2f} - {np.max(dem_grid):.2f} m")
    
    return {
        'dem_grid': dem_grid,
        'land_cover_grid': land_cover_grid,
    }


def apply_citygml_post_processing(building_height_grid, building_min_height_grid, building_id_grid,
                                   canopy_height_grid, canopy_bottom_height_grid, meshsize, grid_vis, **kwargs):
    """
    Apply post-processing: perimeter removal, min canopy height filtering.
    
    Args:
        building_height_grid: Building height grid.
        building_min_height_grid: Building minimum height grid.
        building_id_grid: Building ID grid.
        canopy_height_grid: Canopy height grid.
        canopy_bottom_height_grid: Canopy bottom height grid.
        meshsize: Voxel size in meters.
        grid_vis: Whether to visualize grids.
        **kwargs: Additional options including:
            - min_canopy_height: Minimum canopy height filter
            - remove_perimeter_object: Perimeter removal ratio
    """
    # Min canopy height filter
    min_canopy_height = kwargs.get("min_canopy_height")
    if min_canopy_height is not None:
        canopy_height_grid[canopy_height_grid < min_canopy_height] = 0
        canopy_bottom_height_grid[canopy_height_grid == 0] = 0
    
    # Perimeter object removal
    remove_perimeter_object = kwargs.get("remove_perimeter_object")
    if remove_perimeter_object is not None and remove_perimeter_object > 0:
        print("  Applying perimeter removal...")
        w_peri = int(remove_perimeter_object * building_height_grid.shape[0] + 0.5)
        h_peri = int(remove_perimeter_object * building_height_grid.shape[1] + 0.5)
        
        # Remove canopy at perimeter
        canopy_height_grid[:w_peri, :] = canopy_height_grid[-w_peri:, :] = 0
        canopy_height_grid[:, :h_peri] = canopy_height_grid[:, -h_peri:] = 0
        canopy_bottom_height_grid[:w_peri, :] = canopy_bottom_height_grid[-w_peri:, :] = 0
        canopy_bottom_height_grid[:, :h_peri] = canopy_bottom_height_grid[:, -h_peri:] = 0
        
        # Remove buildings at perimeter
        ids1 = np.unique(building_id_grid[:w_peri, :][building_id_grid[:w_peri, :] > 0])
        ids2 = np.unique(building_id_grid[-w_peri:, :][building_id_grid[-w_peri:, :] > 0])
        ids3 = np.unique(building_id_grid[:, :h_peri][building_id_grid[:, :h_peri] > 0])
        ids4 = np.unique(building_id_grid[:, -h_peri:][building_id_grid[:, -h_peri:] > 0])
        remove_ids = np.concatenate((ids1, ids2, ids3, ids4))
        
        for remove_id in remove_ids:
            positions = np.where(building_id_grid == remove_id)
            building_height_grid[positions] = 0
            building_min_height_grid[positions] = [[] for _ in range(len(building_min_height_grid[positions]))]


# =============================================================================
# OPTIMIZED VERSIONS (accept pre-cached data to avoid redundant parsing)
# =============================================================================

def voxelize_buildings_citygml_optimized(citygml_path_resolved, rectangle_vertices, meshsize, 
                                          lod, include_bridges, include_city_furniture,
                                          grid_vis, building_gdf_cached=None, **kwargs):
    """
    Optimized version of building voxelization that accepts pre-cached building GDF.
    
    This avoids redundant LOD1 parsing when data has already been parsed.
    
    Args:
        citygml_path_resolved: Path to CityGML directory.
        rectangle_vertices: Bounding rectangle vertices.
        meshsize: Voxel size in meters.
        lod: LOD mode - 'lod1', 'lod2', or None for auto-detection.
        include_bridges: Whether to include bridges (LOD2 mode).
        include_city_furniture: Whether to include city furniture (LOD2 mode).
        grid_vis: Whether to visualize grids.
        building_gdf_cached: Pre-parsed building GeoDataFrame (optimization).
        **kwargs: Additional options.
        
    Returns:
        Dictionary containing building voxelization results.
    """
    import geopandas as gpd
    from ...geoprocessor.raster import create_building_height_grid_from_gdf_polygon
    from ...visualizer.grids import visualize_numerical_grid
    
    lod2_voxelizer = None
    
    # Determine LOD mode: 'lod1', 'lod2', or None (auto-detect)
    if lod is None:
        # Auto-detect: try LOD2 first, fall back to LOD1 if not available
        print(f"  Mode: Auto-detecting LOD...")
        lod2_voxelizer = PLATEAUVoxelizer(voxel_size=meshsize)
        lod2_voxelizer.parse_plateau_directory(
            base_path=citygml_path_resolved,
            rectangle_vertices=rectangle_vertices,
            parse_buildings=True,
            parse_dem=False,
            parse_vegetation=False,
            parse_bridges=include_bridges,
            parse_city_furniture=include_city_furniture,
            parse_land_use=False,
        )
        
        if lod2_voxelizer.parser is not None and len(lod2_voxelizer.parser.buildings) > 0:
            lod = 'lod2'
            print(f"  Auto-detected: LOD2 (found {len(lod2_voxelizer.parser.buildings)} buildings)")
        else:
            lod = 'lod1'
            print(f"  Auto-detected: LOD1 (no LOD2 buildings found)")
            lod2_voxelizer = None
    elif lod == 'lod2' or lod == 'LOD2':
        lod = 'lod2'
        print(f"  Mode: LOD2 (triangulated geometry)")
        print(f"  Include bridges: {include_bridges}")
        print(f"  Include city furniture: {include_city_furniture}")
        
        lod2_voxelizer = PLATEAUVoxelizer(voxel_size=meshsize)
        lod2_voxelizer.parse_plateau_directory(
            base_path=citygml_path_resolved,
            rectangle_vertices=rectangle_vertices,
            parse_buildings=True,
            parse_dem=False,
            parse_vegetation=False,
            parse_bridges=include_bridges,
            parse_city_furniture=include_city_furniture,
            parse_land_use=False,
        )
        
        if lod2_voxelizer.parser is None or len(lod2_voxelizer.parser.buildings) == 0:
            print("  Warning: No LOD2 buildings found. Falling back to LOD1 mode.")
            lod = 'lod1'
            lod2_voxelizer = None
        else:
            print(f"  Found {len(lod2_voxelizer.parser.buildings)} LOD2 buildings")
            if include_bridges and lod2_voxelizer.parser.bridges:
                print(f"  Found {len(lod2_voxelizer.parser.bridges)} bridges")
            if include_city_furniture and lod2_voxelizer.parser.city_furniture:
                print(f"  Found {len(lod2_voxelizer.parser.city_furniture)} city furniture items")
    else:
        # LOD1 mode (explicit or fallback)
        lod = 'lod1'
        print(f"  Mode: LOD1 (footprint-based)")
    
    use_lod2 = (lod == 'lod2')
    
    # Use cached building GDF if available, otherwise parse
    if building_gdf_cached is not None:
        print("  Using pre-cached building footprints")
        building_gdf = building_gdf_cached
    else:
        print("  Parsing LOD1 building footprints for height grids...")
        from .parsers import load_lod1_citygml
        building_gdf, _, _ = load_lod1_citygml(
            citygml_path=citygml_path_resolved,
            rectangle_vertices=rectangle_vertices,
            parse_buildings=not use_lod2,
            parse_terrain=False,
            parse_vegetation=False,
        )
    
    # Ensure CRS is EPSG:4326
    if building_gdf is None and use_lod2:
        building_gdf = gpd.GeoDataFrame(
            columns=['building_id', 'height', 'storeys', 'ground_elevation', 'geometry', 'source_file', 'id'],
            geometry='geometry', crs='EPSG:4326'
        )
    elif building_gdf is not None:
        if building_gdf.crs is None:
            building_gdf = building_gdf.set_crs(epsg=4326)
        elif getattr(building_gdf.crs, 'to_epsg', lambda: None)() != 4326:
            building_gdf = building_gdf.to_crs(epsg=4326)
    
    # Create building height grids
    print("  Creating building height grid")
    building_height_grid, building_min_height_grid, building_id_grid, _ = create_building_height_grid_from_gdf_polygon(
        building_gdf, meshsize, rectangle_vertices
    )
    
    if grid_vis and not use_lod2:
        building_height_grid_nan = building_height_grid.copy()
        building_height_grid_nan[building_height_grid_nan == 0] = np.nan
        visualize_numerical_grid(np.flipud(building_height_grid_nan), meshsize, "building height (m)", cmap='viridis', label='Value')
    
    return {
        'lod': lod,
        'use_lod2': use_lod2,  # For backward compatibility
        'building_gdf': building_gdf,
        'building_height_grid': building_height_grid,
        'building_min_height_grid': building_min_height_grid,
        'building_id_grid': building_id_grid,
        'lod2_voxelizer': lod2_voxelizer,
    }


def voxelize_trees_citygml_optimized(citygml_path_resolved, rectangle_vertices, meshsize,
                                      land_cover_source, canopy_height_source, use_lod2,
                                      include_lod2_vegetation, trunk_height_ratio, output_dir,
                                      vegetation_gdf_cached=None, land_cover_grid_cached=None, **kwargs):
    """
    Optimized version of tree voxelization that accepts pre-cached data.
    
    Args:
        citygml_path_resolved: Path to CityGML directory.
        rectangle_vertices: Bounding rectangle vertices.
        meshsize: Voxel size in meters.
        land_cover_source: Land cover data source.
        canopy_height_source: Canopy height data source.
        use_lod2: Whether LOD2 mode is active.
        include_lod2_vegetation: Whether to use LOD2 vegetation geometry.
        trunk_height_ratio: Ratio of trunk height to total tree height.
        output_dir: Output directory.
        vegetation_gdf_cached: Pre-parsed vegetation GeoDataFrame (optimization).
        land_cover_grid_cached: Pre-downloaded land cover grid (optimization).
        **kwargs: Additional options.
        
    Returns:
        Dictionary containing tree voxelization results.
    """
    from ...generator.grids import get_land_cover_grid, get_canopy_height_grid
    from ...geoprocessor.raster import create_vegetation_height_grid_from_gdf_polygon
    from ...utils.lc import get_land_cover_classes
    
    # Use cached vegetation GDF if available
    if vegetation_gdf_cached is not None:
        print("  Using pre-cached CityGML vegetation data")
        vegetation_gdf = vegetation_gdf_cached
    else:
        from .parsers import load_lod1_citygml
        print("  Parsing CityGML vegetation data...")
        _, _, vegetation_gdf = load_lod1_citygml(
            citygml_path=citygml_path_resolved,
            rectangle_vertices=rectangle_vertices,
            parse_buildings=False,
            parse_terrain=False,
            parse_vegetation=True,
        )
    
    if vegetation_gdf is not None:
        if vegetation_gdf.crs is None:
            vegetation_gdf = vegetation_gdf.set_crs(epsg=4326)
        elif getattr(vegetation_gdf.crs, 'to_epsg', lambda: None)() != 4326:
            vegetation_gdf = vegetation_gdf.to_crs(epsg=4326)
    
    # Get external canopy height source (complementary)
    if canopy_height_source == "Static":
        print(f"  Using static tree height")
        static_tree_height = kwargs.get("static_tree_height", 10.0)
        
        # Use cached land cover grid if available
        if land_cover_grid_cached is not None:
            print("  Using pre-cached land cover grid")
            land_cover_grid_temp = land_cover_grid_cached
        else:
            land_cover_grid_temp = get_land_cover_grid(rectangle_vertices, meshsize, land_cover_source, output_dir, **kwargs)
        
        canopy_height_grid_comp = np.zeros_like(land_cover_grid_temp, dtype=float)
        
        _classes = get_land_cover_classes(land_cover_source)
        _class_to_int = {name: i for i, name in enumerate(_classes.values())}
        _tree_labels = ["Tree", "Trees", "Tree Canopy"]
        _tree_indices = [_class_to_int[label] for label in _tree_labels if label in _class_to_int]
        tree_mask = np.isin(land_cover_grid_temp, _tree_indices) if _tree_indices else np.zeros_like(land_cover_grid_temp, dtype=bool)
        canopy_height_grid_comp[tree_mask] = static_tree_height
        canopy_bottom_height_grid_comp = canopy_height_grid_comp * float(trunk_height_ratio)
    else:
        print(f"  Using external canopy height source: {canopy_height_source}")
        canopy_height_grid_comp, canopy_bottom_height_grid_comp = get_canopy_height_grid(
            rectangle_vertices, meshsize, canopy_height_source, output_dir, **kwargs
        )
    
    # Create canopy grids from CityGML vegetation
    if vegetation_gdf is not None and len(vegetation_gdf) > 0:
        print(f"  Found {len(vegetation_gdf)} CityGML vegetation objects")
        canopy_height_grid = create_vegetation_height_grid_from_gdf_polygon(vegetation_gdf, meshsize, rectangle_vertices)
        canopy_bottom_height_grid = canopy_height_grid * float(trunk_height_ratio)
    else:
        print("  No CityGML vegetation found")
        canopy_height_grid = np.zeros_like(canopy_height_grid_comp)
        canopy_bottom_height_grid = np.zeros_like(canopy_height_grid_comp)
    
    # Merge: use external source where CityGML has no data
    mask = (canopy_height_grid == 0) & (canopy_height_grid_comp != 0)
    canopy_height_grid[mask] = canopy_height_grid_comp[mask]
    mask_b = (canopy_bottom_height_grid == 0) & (canopy_bottom_height_grid_comp != 0)
    canopy_bottom_height_grid[mask_b] = canopy_bottom_height_grid_comp[mask_b]
    canopy_bottom_height_grid = np.minimum(canopy_bottom_height_grid, canopy_height_grid)
    
    print(f"  Canopy coverage: {np.sum(canopy_height_grid > 0)} cells")
    
    return {
        'canopy_height_grid': canopy_height_grid,
        'canopy_bottom_height_grid': canopy_bottom_height_grid,
    }


def voxelize_terrain_citygml_optimized(citygml_path_resolved, rectangle_vertices, meshsize,
                                        land_cover_source, dem_source, building_id_grid,
                                        grid_vis, output_dir, terrain_gdf_cached=None,
                                        land_cover_grid_cached=None, **kwargs):
    """
    Optimized version of terrain voxelization that accepts pre-cached data.
    
    Uses KDTree-based DEM interpolation for significantly faster terrain processing
    compared to the scipy.interpolate.griddata approach.
    
    Args:
        citygml_path_resolved: Path to CityGML directory.
        rectangle_vertices: Bounding rectangle vertices.
        meshsize: Voxel size in meters.
        land_cover_source: Land cover data source.
        dem_source: DEM data source (None for CityGML terrain).
        building_id_grid: Building ID grid for terrain flattening.
        grid_vis: Whether to visualize grids.
        output_dir: Output directory.
        terrain_gdf_cached: Pre-parsed terrain GeoDataFrame (optimization).
        land_cover_grid_cached: Pre-downloaded land cover grid (optimization).
        **kwargs: Additional options.
        
    Returns:
        Dictionary containing terrain voxelization results.
    """
    from ...generator.grids import get_land_cover_grid, get_dem_grid
    from ...geoprocessor.raster import process_grid, create_dem_grid_from_gdf_polygon, create_dem_grid_from_gdf_kdtree
    from ...utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP
    from ...visualizer.grids import visualize_numerical_grid
    
    # Use cached land cover grid if available
    if land_cover_grid_cached is not None:
        print("  Using pre-cached land cover grid")
        land_cover_grid = land_cover_grid_cached
    else:
        print("  Creating land cover grid...")
        land_cover_grid = get_land_cover_grid(rectangle_vertices, meshsize, land_cover_source, output_dir, **kwargs)
    
    # Get DEM grid
    if kwargs.get('flat_dem', False):
        print("  Using flat DEM")
        dem_grid = np.zeros_like(land_cover_grid, dtype=float)
    elif dem_source is not None:
        print(f"  Using external DEM source: {dem_source}")
        dem_grid = get_dem_grid(rectangle_vertices, meshsize, dem_source, output_dir, **kwargs)
    else:
        print("  Using CityGML terrain data")
        
        # Use cached terrain GDF if available
        if terrain_gdf_cached is not None:
            print("  Using pre-cached terrain data")
            terrain_gdf = terrain_gdf_cached
        else:
            from .parsers import load_lod1_citygml
            _, terrain_gdf, _ = load_lod1_citygml(
                citygml_path=citygml_path_resolved,
                rectangle_vertices=rectangle_vertices,
                parse_buildings=False,
                parse_terrain=True,
                parse_vegetation=False,
            )
        
        if terrain_gdf is not None:
            if terrain_gdf.crs is None:
                terrain_gdf = terrain_gdf.set_crs(epsg=4326)
            elif getattr(terrain_gdf.crs, 'to_epsg', lambda: None)() != 4326:
                terrain_gdf = terrain_gdf.to_crs(epsg=4326)
        
        # Check if interpolation is requested
        dem_interpolation = kwargs.get('dem_interpolation', False)
        
        if dem_interpolation:
            # Use scipy.interpolate.griddata for smooth interpolation
            print("  Using interpolated DEM (scipy.interpolate.griddata)")
            dem_grid = create_dem_grid_from_gdf_polygon(
                terrain_gdf, meshsize, rectangle_vertices, 
                interpolation=True, method='linear'
            )
        else:
            # Use KDTree-based nearest-neighbor for faster processing
            print("  Using KDTree-based DEM (nearest-neighbor, optimized)")
            dem_grid = create_dem_grid_from_gdf_kdtree(terrain_gdf, meshsize, rectangle_vertices)
        
        if dem_grid is None or (np.all(dem_grid == 0) and terrain_gdf is None):
            print("  Warning: No terrain data found in CityGML. Using flat DEM.")
            dem_grid = np.zeros_like(land_cover_grid, dtype=float)
    
    # Flatten terrain under buildings (standard VoxCity approach)
    print("  Flattening terrain under building footprints...")
    dem_grid_oriented = ensure_orientation(dem_grid.copy(), ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)
    building_id_grid_oriented = ensure_orientation(building_id_grid.copy(), ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)
    dem_grid_flattened = process_grid(building_id_grid_oriented, dem_grid_oriented - np.min(dem_grid_oriented))
    # Restore original orientation and offset
    dem_grid = ensure_orientation(dem_grid_flattened + np.min(dem_grid), ORIENTATION_SOUTH_UP, ORIENTATION_NORTH_UP)
    
    if grid_vis and dem_grid is not None:
        visualize_numerical_grid(np.flipud(dem_grid), meshsize, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')
    
    print(f"  DEM range: {np.min(dem_grid):.2f} - {np.max(dem_grid):.2f} m")
    
    return {
        'dem_grid': dem_grid,
        'land_cover_grid': land_cover_grid,
    }


def merge_lod2_voxels(voxelizer, lod2_voxelizer, building_height_grid, building_min_height_grid,
                      building_id_grid, land_cover_grid, dem_grid, canopy_height_grid,
                      canopy_bottom_height_grid, rectangle_vertices, meshsize,
                      include_bridges, include_city_furniture):
    """
    Merge LOD2 building/bridge/furniture voxels with base terrain/vegetation grid.
    
    Process:
    1. Generate base grid with terrain, land cover, and vegetation (no buildings)
    2. Voxelize LOD2 buildings/bridges/furniture
    3. Align LOD2 voxels with terrain elevation
    4. Merge into final grid
    
    Args:
        voxelizer: Standard Voxelizer instance for base grid generation.
        lod2_voxelizer: PLATEAUVoxelizer instance with parsed LOD2 data.
        building_height_grid: Building height grid.
        building_min_height_grid: Building minimum height grid.
        building_id_grid: Building ID grid.
        land_cover_grid: Land cover grid.
        dem_grid: DEM grid.
        canopy_height_grid: Canopy height grid.
        canopy_bottom_height_grid: Canopy bottom height grid.
        rectangle_vertices: Bounding rectangle vertices.
        meshsize: Voxel size in meters.
        include_bridges: Whether to include bridges.
        include_city_furniture: Whether to include city furniture.
        
    Returns:
        Merged voxel grid.
    """
    from .constants import BUILDING_CODE, BRIDGE_CODE, CITY_FURNITURE_CODE
    from ...utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP
    
    print("  Using LOD2 triangulated building voxelization")
    
    # Create empty building grids for base voxelization
    empty_building_min_height_grid = np.empty_like(building_min_height_grid, dtype=object)
    for i in range(empty_building_min_height_grid.shape[0]):
        for j in range(empty_building_min_height_grid.shape[1]):
            empty_building_min_height_grid[i, j] = []
    
    # Generate base grid WITHOUT buildings
    print("  Generating base grid (terrain + land cover + vegetation)...")
    voxcity_grid = voxelizer.generate_combined(
        building_height_grid_ori=np.zeros_like(building_height_grid),
        building_min_height_grid_ori=empty_building_min_height_grid,
        building_id_grid_ori=np.zeros_like(building_id_grid),
        land_cover_grid_ori=land_cover_grid,
        dem_grid_ori=dem_grid,
        tree_grid_ori=canopy_height_grid,
        canopy_bottom_height_grid_ori=canopy_bottom_height_grid,
    )
    
    # Voxelize LOD2 buildings/bridges/furniture
    print("  Voxelizing LOD2 geometry...")
    nx, ny, nz = voxcity_grid.shape
    
    # Pass flattened DEM to LOD2 voxelizer for terrain-relative building placement
    # Only pass building_id_grid if it has actual buildings (not all zeros)
    # If all zeros, let LOD2 voxelizer create its own from LOD2 geometry
    dem_grid_for_lod2 = ensure_orientation(dem_grid.copy(), ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)
    
    # Check if we have LOD1 building footprints to pass
    has_lod1_buildings = np.any(building_id_grid > 0)
    if has_lod1_buildings:
        building_id_grid_for_lod2 = ensure_orientation(building_id_grid.copy(), ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)
        print(f"    Using LOD1 building footprints for terrain alignment")
    else:
        building_id_grid_for_lod2 = None
        print(f"    No LOD1 footprints, using LOD2 geometry for building footprints")
    
    lod2_voxels = lod2_voxelizer.voxelize(
        add_ground=False,
        ground_thickness=0,
        include_vegetation=False,
        include_bridges=include_bridges,
        include_city_furniture=include_city_furniture,
        rectangle_vertices=rectangle_vertices,
        target_grid_size=(nx, ny),
        external_dem_grid=dem_grid_for_lod2,
        external_building_id_grid=building_id_grid_for_lod2,
    )
    
    # Resize if needed
    lod2_nx, lod2_ny, lod2_nz = lod2_voxels.shape
    if (lod2_nx, lod2_ny) != (nx, ny):
        print(f"  Resizing LOD2 voxels from ({lod2_nx}, {lod2_ny}) to ({nx}, {ny})...")
        from scipy.ndimage import zoom
        zoom_factors = (nx / lod2_nx, ny / lod2_ny, 1.0)
        lod2_voxels = zoom(lod2_voxels.astype(float), zoom_factors, order=0).astype(lod2_voxels.dtype)
        lod2_voxels = lod2_voxels[:nx, :ny, :]
    
    # Since LOD2 voxelizer now uses the same flattened DEM as the base grid,
    # the Z levels should align. However, there's a +1 offset difference between
    # the standard voxelizer (ground_level = dem/voxel + 0.5 + 1) and the LOD2 
    # voxelizer (local_ground = dem/voxel). We apply a uniform +1 Z offset.
    print("  Merging LOD2 voxels into base grid...")
    
    # Expand base grid if needed
    total_nz = max(nz, lod2_voxels.shape[2] + 1)
    if total_nz > nz:
        voxcity_grid = np.pad(voxcity_grid, ((0, 0), (0, 0), (0, total_nz - nz)), mode='constant')
    
    # Merge LOD2 voxels with +1 Z offset to match standard voxelizer ground level formula
    lod2_codes = [BUILDING_CODE, BRIDGE_CODE, CITY_FURNITURE_CODE]
    merged_count = 0
    z_offset = 1  # Fixed offset: standard uses +1 in ground_level formula
    
    for i in range(nx):
        for j in range(ny):
            for k in range(lod2_voxels.shape[2]):
                if lod2_voxels[i, j, k] in lod2_codes:
                    new_k = k + z_offset
                    if new_k < voxcity_grid.shape[2]:
                        voxcity_grid[i, j, new_k] = lod2_voxels[i, j, k]
                        merged_count += 1
    
    print(f"  Merged {merged_count:,} LOD2 voxels")
    print(f"    Buildings: {np.sum(voxcity_grid == BUILDING_CODE):,}")
    if include_bridges:
        print(f"    Bridges: {np.sum(voxcity_grid == BRIDGE_CODE):,}")
    if include_city_furniture:
        print(f"    City Furniture: {np.sum(voxcity_grid == CITY_FURNITURE_CODE):,}")
    
    return voxcity_grid
