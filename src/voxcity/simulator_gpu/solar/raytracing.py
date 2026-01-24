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
        # Small offset to ensure ray origin is outside the solid voxel
        eps = 0.01
        
        for i in range(n_surf):
            # Get surface position
            pos = surf_pos[i]
            direction = surf_dir[i]
            
            # Check if surface normal faces toward sun (dot product > 0)
            # Direction indices: 0=Up, 1=Down, 2=INORTH(+y), 3=ISOUTH(-y), 4=IEAST(+x), 5=IWEST(-x)
            # In VoxCity grid: +x = South, +y = East, +z = Up
            face_sun = 1
            normal = Vector3(0.0, 0.0, 0.0)
            if direction == 0:  # Up (+z normal)
                face_sun = 1 if sun_dir[2] > 0 else 0
                normal = Vector3(0.0, 0.0, 1.0)
            elif direction == 1:  # Down (-z normal)
                face_sun = 1 if sun_dir[2] < 0 else 0
                normal = Vector3(0.0, 0.0, -1.0)
            elif direction == 2:  # INORTH (+y normal, East-facing in VoxCity)
                face_sun = 1 if sun_dir[1] > 0 else 0
                normal = Vector3(0.0, 1.0, 0.0)
            elif direction == 3:  # ISOUTH (-y normal, West-facing in VoxCity)
                face_sun = 1 if sun_dir[1] < 0 else 0
                normal = Vector3(0.0, -1.0, 0.0)
            elif direction == 4:  # IEAST (+x normal, South-facing in VoxCity)
                face_sun = 1 if sun_dir[0] > 0 else 0
                normal = Vector3(1.0, 0.0, 0.0)
            elif direction == 5:  # IWEST (-x normal, North-facing in VoxCity)
                face_sun = 1 if sun_dir[0] < 0 else 0
                normal = Vector3(-1.0, 0.0, 0.0)
            
            if face_sun == 0:
                shadow_factor[i] = 1.0
            else:
                # Offset ray origin slightly along surface normal to avoid self-intersection
                ray_origin = Vector3(pos[0] + normal[0] * eps,
                                     pos[1] + normal[1] * eps,
                                     pos[2] + normal[2] * eps)
                
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
        # Small offset to ensure ray origin is outside the solid voxel
        eps = 0.01
        
        for i in range(n_surf):
            pos = surf_pos[i]
            direction = surf_dir[i]
            
            # Check if surface normal faces toward sun (dot product > 0)
            # In VoxCity grid: +x = South, +y = East, +z = Up
            face_sun = 1
            normal = Vector3(0.0, 0.0, 0.0)
            if direction == 0:  # Up (+z)
                face_sun = 1 if sun_dir[2] > 0 else 0
                normal = Vector3(0.0, 0.0, 1.0)
            elif direction == 1:  # Down (-z)
                face_sun = 1 if sun_dir[2] < 0 else 0
                normal = Vector3(0.0, 0.0, -1.0)
            elif direction == 2:  # INORTH (+y, East-facing)
                face_sun = 1 if sun_dir[1] > 0 else 0
                normal = Vector3(0.0, 1.0, 0.0)
            elif direction == 3:  # ISOUTH (-y, West-facing)
                face_sun = 1 if sun_dir[1] < 0 else 0
                normal = Vector3(0.0, -1.0, 0.0)
            elif direction == 4:  # IEAST (+x, South-facing)
                face_sun = 1 if sun_dir[0] > 0 else 0
                normal = Vector3(1.0, 0.0, 0.0)
            elif direction == 5:  # IWEST (-x, North-facing)
                face_sun = 1 if sun_dir[0] < 0 else 0
                normal = Vector3(-1.0, 0.0, 0.0)
            
            if face_sun == 0:
                shadow_factor[i] = 1.0
                canopy_transmissivity[i] = 0.0
            else:
                # Offset ray origin slightly along surface normal to avoid self-intersection
                ray_origin = Vector3(pos[0] + normal[0] * eps,
                                     pos[1] + normal[1] * eps,
                                     pos[2] + normal[2] * eps)
                
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
