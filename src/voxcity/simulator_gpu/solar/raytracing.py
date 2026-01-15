"""
Ray tracing module for solar simulation.

This module provides the RayTracer class for GPU-accelerated radiation calculations.
Shared ray tracing functions are imported from simulator_gpu.raytracing.

Usage:
    from .raytracing import RayTracer, ray_voxel_first_hit, ray_canopy_absorption
"""

import taichi as ti
import math
from typing import Tuple, Optional

from .core import Vector3, Point3, EXT_COEF

# Import shared ray tracing functions from parent module
from ..raytracing import (
    ray_aabb_intersect,
    ray_voxel_first_hit,
    ray_canopy_absorption,
    ray_voxel_transmissivity,
    ray_trace_to_target,
    ray_point_to_point_transmissivity,
    sample_hemisphere_direction,
    hemisphere_solid_angle,
)


@ti.data_oriented
class RayTracer:
    """
    GPU-accelerated ray tracer for radiation calculations.
    
    Traces rays through the voxel domain to compute:
    - Shadow factors (direct sunlight blocking)
    - Sky view factors (visible sky fraction)
    - Canopy sink factors (absorption by vegetation)
    """
    
    def __init__(self, domain):
        """
        Initialize ray tracer with domain.
        
        Args:
            domain: Domain object with grid geometry
        """
        self.domain = domain
        self.nx = domain.nx
        self.ny = domain.ny
        self.nz = domain.nz
        self.dx = domain.dx
        self.dy = domain.dy
        self.dz = domain.dz
        
        # Maximum ray distance (diagonal of domain)
        self.max_dist = math.sqrt(
            (self.nx * self.dx)**2 + 
            (self.ny * self.dy)**2 + 
            (self.nz * self.dz)**2
        )
        
        self.ext_coef = EXT_COEF
    
    @ti.kernel
    def compute_direct_shadows(
        self,
        surf_pos: ti.template(),
        surf_dir: ti.template(),
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template(),
        n_surf: ti.i32,
        shadow_factor: ti.template()
    ):
        """
        Compute shadow factors for all surfaces.
        
        shadow_factor = 0 means fully sunlit
        shadow_factor = 1 means fully shaded
        """
        for i in range(n_surf):
            # Get surface position
            pos = surf_pos[i]
            direction = surf_dir[i]
            
            # Check if surface faces sun
            # For upward (0), downward (1), north (2), south (3), east (4), west (5)
            face_sun = 1
            if direction == 0:  # Up
                face_sun = 1 if sun_dir[2] > 0 else 0
            elif direction == 1:  # Down
                face_sun = 1 if sun_dir[2] < 0 else 0
            elif direction == 2:  # North
                face_sun = 1 if sun_dir[1] > 0 else 0
            elif direction == 3:  # South
                face_sun = 1 if sun_dir[1] < 0 else 0
            elif direction == 4:  # East
                face_sun = 1 if sun_dir[0] > 0 else 0
            elif direction == 5:  # West
                face_sun = 1 if sun_dir[0] < 0 else 0
            
            if face_sun == 0:
                shadow_factor[i] = 1.0
            else:
                # Trace ray toward sun
                ray_origin = Vector3(pos[0], pos[1], pos[2])
                
                hit, _, _, _, _ = ray_voxel_first_hit(
                    ray_origin, sun_dir,
                    is_solid,
                    self.nx, self.ny, self.nz,
                    self.dx, self.dy, self.dz,
                    self.max_dist
                )
                
                shadow_factor[i] = ti.cast(hit, ti.f32)
    
    @ti.kernel
    def compute_direct_with_canopy(
        self,
        surf_pos: ti.template(),
        surf_dir: ti.template(),
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template(),
        lad: ti.template(),
        n_surf: ti.i32,
        shadow_factor: ti.template(),
        canopy_transmissivity: ti.template()
    ):
        """
        Compute shadow factors including canopy absorption.
        """
        for i in range(n_surf):
            pos = surf_pos[i]
            direction = surf_dir[i]
            
            # Check if surface faces sun
            face_sun = 1
            if direction == 0:
                face_sun = 1 if sun_dir[2] > 0 else 0
            elif direction == 1:
                face_sun = 1 if sun_dir[2] < 0 else 0
            elif direction == 2:
                face_sun = 1 if sun_dir[1] > 0 else 0
            elif direction == 3:
                face_sun = 1 if sun_dir[1] < 0 else 0
            elif direction == 4:
                face_sun = 1 if sun_dir[0] > 0 else 0
            elif direction == 5:
                face_sun = 1 if sun_dir[0] < 0 else 0
            
            if face_sun == 0:
                shadow_factor[i] = 1.0
                canopy_transmissivity[i] = 0.0
            else:
                ray_origin = Vector3(pos[0], pos[1], pos[2])
                
                trans, _ = ray_canopy_absorption(
                    ray_origin, sun_dir,
                    lad, is_solid,
                    self.nx, self.ny, self.nz,
                    self.dx, self.dy, self.dz,
                    self.max_dist,
                    self.ext_coef
                )
                
                canopy_transmissivity[i] = trans
                shadow_factor[i] = 1.0 - trans


# Re-export all symbols for backward compatibility
__all__ = [
    'RayTracer',
    'ray_aabb_intersect',
    'ray_voxel_first_hit',
    'ray_canopy_absorption',
    'ray_voxel_transmissivity',
    'ray_trace_to_target',
    'ray_point_to_point_transmissivity',
    'sample_hemisphere_direction',
    'hemisphere_solid_angle',
]
