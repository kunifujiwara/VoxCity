"""
Ray tracing module for palm-solar.

Implements GPU-accelerated ray tracing through a 3D voxel grid
to compute shadows, sky view factors, and canopy absorption.
Uses 3D-DDA (Digital Differential Analyzer) for voxel traversal.

PALM Alignment:
- Beer-Lambert law for canopy: trans = exp(-ext_coef * LAD * path_length)
- Solid obstacles block rays completely (trans = 0)
- Ray tracing replaces PALM's raytrace_2d subroutine with GPU-parallel version

Difference from PALM:
- PALM pre-computes dsitrans for discrete solar directions
- palm_solar traces rays dynamically for exact sun position
- This gives identical physics with slightly different numerical approach

Note: This module contains solar-specific ray tracing implementations that use
LAD (Leaf Area Density) fields. For simpler view-based ray tracing using 
is_tree masks, see simulator_gpu.raytracing.
"""

import taichi as ti
import math
from typing import Tuple, Optional

from .core import Vector3, Point3, EXT_COEF


@ti.func
def ray_aabb_intersect(
    ray_origin: Vector3,
    ray_dir: Vector3,
    box_min: Vector3,
    box_max: Vector3,
    t_min: ti.f32,
    t_max: ti.f32
):
    """
    Ray-AABB intersection using slab method.
    
    Args:
        ray_origin: Ray origin point
        ray_dir: Ray direction (normalized)
        box_min: AABB minimum corner
        box_max: AABB maximum corner
        t_min: Minimum t value
        t_max: Maximum t value
    
    Returns:
        Tuple of (hit, t_enter, t_exit)
    """
    t_enter = t_min
    t_exit = t_max
    hit = 1
    
    for i in ti.static(range(3)):
        if ti.abs(ray_dir[i]) < 1e-10:
            # Ray parallel to slab
            if ray_origin[i] < box_min[i] or ray_origin[i] > box_max[i]:
                hit = 0
        else:
            inv_d = 1.0 / ray_dir[i]
            t1 = (box_min[i] - ray_origin[i]) * inv_d
            t2 = (box_max[i] - ray_origin[i]) * inv_d
            
            if t1 > t2:
                t1, t2 = t2, t1
            
            t_enter = ti.max(t_enter, t1)
            t_exit = ti.min(t_exit, t2)
    
    if t_enter > t_exit:
        hit = 0
    
    return hit, t_enter, t_exit


@ti.func
def ray_voxel_first_hit(
    ray_origin: Vector3,
    ray_dir: Vector3,
    is_solid: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    dx: ti.f32,
    dy: ti.f32,
    dz: ti.f32,
    max_dist: ti.f32
):
    """
    3D-DDA ray marching to find first solid voxel hit.
    
    Args:
        ray_origin: Ray origin
        ray_dir: Ray direction (normalized)
        is_solid: 3D field of solid cells
        nx, ny, nz: Grid dimensions
        dx, dy, dz: Cell sizes
        max_dist: Maximum ray distance
    
    Returns:
        Tuple of (hit, t_hit, ix, iy, iz)
    """
    hit = 0
    t_hit = max_dist
    hit_ix, hit_iy, hit_iz = 0, 0, 0
    
    # Find entry into domain
    domain_min = Vector3(0.0, 0.0, 0.0)
    domain_max = Vector3(nx * dx, ny * dy, nz * dz)
    
    in_domain, t_enter, t_exit = ray_aabb_intersect(
        ray_origin, ray_dir, domain_min, domain_max, 0.0, max_dist
    )
    
    if in_domain == 1:
        # Start position (slightly inside domain)
        t = t_enter + 1e-5
        pos = ray_origin + ray_dir * t
        
        # Current voxel indices
        ix = ti.cast(ti.floor(pos[0] / dx), ti.i32)
        iy = ti.cast(ti.floor(pos[1] / dy), ti.i32)
        iz = ti.cast(ti.floor(pos[2] / dz), ti.i32)
        
        # Clamp to valid range
        ix = ti.max(0, ti.min(nx - 1, ix))
        iy = ti.max(0, ti.min(ny - 1, iy))
        iz = ti.max(0, ti.min(nz - 1, iz))
        
        # Step directions
        step_x = 1 if ray_dir[0] >= 0 else -1
        step_y = 1 if ray_dir[1] >= 0 else -1
        step_z = 1 if ray_dir[2] >= 0 else -1
        
        # Initialize DDA variables
        t_max_x = 1e30
        t_max_y = 1e30
        t_max_z = 1e30
        t_delta_x = 1e30
        t_delta_y = 1e30
        t_delta_z = 1e30
        
        # t values for next boundary crossing
        if ti.abs(ray_dir[0]) > 1e-10:
            if step_x > 0:
                t_max_x = ((ix + 1) * dx - pos[0]) / ray_dir[0] + t
            else:
                t_max_x = (ix * dx - pos[0]) / ray_dir[0] + t
            t_delta_x = ti.abs(dx / ray_dir[0])
        
        if ti.abs(ray_dir[1]) > 1e-10:
            if step_y > 0:
                t_max_y = ((iy + 1) * dy - pos[1]) / ray_dir[1] + t
            else:
                t_max_y = (iy * dy - pos[1]) / ray_dir[1] + t
            t_delta_y = ti.abs(dy / ray_dir[1])
        
        if ti.abs(ray_dir[2]) > 1e-10:
            if step_z > 0:
                t_max_z = ((iz + 1) * dz - pos[2]) / ray_dir[2] + t
            else:
                t_max_z = (iz * dz - pos[2]) / ray_dir[2] + t
            t_delta_z = ti.abs(dz / ray_dir[2])
        
        # 3D-DDA traversal - optimized with done flag to reduce branch divergence
        # Using done flag pattern is more GPU-friendly than break statements
        max_steps = nx + ny + nz
        done = 0
        
        for _ in range(max_steps):
            # Use done flag pattern for GPU-friendly early termination
            # This reduces warp divergence compared to break statements
            if done == 0:
                # Bounds check - exit if outside domain
                if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                    done = 1
                elif t > t_exit:
                    done = 1
                # Check current voxel for solid hit
                elif is_solid[ix, iy, iz] == 1:
                    hit = 1
                    t_hit = t
                    hit_ix = ix
                    hit_iy = iy
                    hit_iz = iz
                    done = 1
                else:
                    # Step to next voxel using branchless min selection
                    # This is more GPU-efficient than nested if-else
                    if t_max_x < t_max_y and t_max_x < t_max_z:
                        t = t_max_x
                        ix += step_x
                        t_max_x += t_delta_x
                    elif t_max_y < t_max_z:
                        t = t_max_y
                        iy += step_y
                        t_max_y += t_delta_y
                    else:
                        t = t_max_z
                        iz += step_z
                        t_max_z += t_delta_z
    
    return hit, t_hit, hit_ix, hit_iy, hit_iz


@ti.func
def ray_canopy_absorption(
    ray_origin: Vector3,
    ray_dir: Vector3,
    lad: ti.template(),
    is_solid: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    dx: ti.f32,
    dy: ti.f32,
    dz: ti.f32,
    max_dist: ti.f32,
    ext_coef: ti.f32
):
    """
    Trace ray through canopy computing Beer-Lambert absorption.
    
    Args:
        ray_origin: Ray origin
        ray_dir: Ray direction (normalized)
        lad: 3D field of Leaf Area Density
        is_solid: 3D field of solid cells (buildings/terrain)
        nx, ny, nz: Grid dimensions
        dx, dy, dz: Cell sizes
        max_dist: Maximum ray distance
        ext_coef: Extinction coefficient
    
    Returns:
        Tuple of (transmissivity, path_length_through_canopy)
    """
    transmissivity = 1.0
    total_lad_path = 0.0
    
    # Find entry into domain
    domain_min = Vector3(0.0, 0.0, 0.0)
    domain_max = Vector3(nx * dx, ny * dy, nz * dz)
    
    in_domain, t_enter, t_exit = ray_aabb_intersect(
        ray_origin, ray_dir, domain_min, domain_max, 0.0, max_dist
    )
    
    if in_domain == 1:
        t = t_enter + 1e-5
        pos = ray_origin + ray_dir * t
        
        ix = ti.cast(ti.floor(pos[0] / dx), ti.i32)
        iy = ti.cast(ti.floor(pos[1] / dy), ti.i32)
        iz = ti.cast(ti.floor(pos[2] / dz), ti.i32)
        
        ix = ti.max(0, ti.min(nx - 1, ix))
        iy = ti.max(0, ti.min(ny - 1, iy))
        iz = ti.max(0, ti.min(nz - 1, iz))
        
        step_x = 1 if ray_dir[0] >= 0 else -1
        step_y = 1 if ray_dir[1] >= 0 else -1
        step_z = 1 if ray_dir[2] >= 0 else -1
        
        # Initialize all DDA variables
        t_max_x = 1e30
        t_max_y = 1e30
        t_max_z = 1e30
        t_delta_x = 1e30
        t_delta_y = 1e30
        t_delta_z = 1e30
        
        if ti.abs(ray_dir[0]) > 1e-10:
            if step_x > 0:
                t_max_x = ((ix + 1) * dx - pos[0]) / ray_dir[0] + t
            else:
                t_max_x = (ix * dx - pos[0]) / ray_dir[0] + t
            t_delta_x = ti.abs(dx / ray_dir[0])
        
        if ti.abs(ray_dir[1]) > 1e-10:
            if step_y > 0:
                t_max_y = ((iy + 1) * dy - pos[1]) / ray_dir[1] + t
            else:
                t_max_y = (iy * dy - pos[1]) / ray_dir[1] + t
            t_delta_y = ti.abs(dy / ray_dir[1])
        
        if ti.abs(ray_dir[2]) > 1e-10:
            if step_z > 0:
                t_max_z = ((iz + 1) * dz - pos[2]) / ray_dir[2] + t
            else:
                t_max_z = (iz * dz - pos[2]) / ray_dir[2] + t
            t_delta_z = ti.abs(dz / ray_dir[2])
        
        t_prev = t
        max_steps = nx + ny + nz
        done = 0
        
        for _ in range(max_steps):
            # GPU-friendly done flag pattern reduces warp divergence
            if done == 0:
                # Bounds and exit checks
                if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                    done = 1
                elif t > t_exit:
                    done = 1
                # Hit solid -> ray blocked
                elif is_solid[ix, iy, iz] == 1:
                    transmissivity = 0.0
                    done = 1
                else:
                    # Get step distance using branchless min
                    t_next = ti.min(t_max_x, ti.min(t_max_y, t_max_z))
                    
                    # Path length through this cell
                    path_len = t_next - t_prev
                    
                    # Accumulate absorption from LAD
                    # Using fused multiply-add for efficiency
                    cell_lad = lad[ix, iy, iz]
                    if cell_lad > 0.0:
                        lad_path = cell_lad * path_len
                        total_lad_path += lad_path
                        # Beer-Lambert: T = exp(-ext_coef * LAD * path)
                        transmissivity *= ti.exp(-ext_coef * lad_path)
                    
                    t_prev = t_next
                    
                    # Step to next voxel
                    if t_max_x < t_max_y and t_max_x < t_max_z:
                        t = t_max_x
                        ix += step_x
                        t_max_x += t_delta_x
                    elif t_max_y < t_max_z:
                        t = t_max_y
                        iy += step_y
                        t_max_y += t_delta_y
                    else:
                        t = t_max_z
                        iz += step_z
                        t_max_z += t_delta_z
    
    return transmissivity, total_lad_path


@ti.func
def ray_point_to_point_transmissivity(
    pos_from: Vector3,
    pos_to: Vector3,
    lad: ti.template(),
    is_solid: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    dx: ti.f32,
    dy: ti.f32,
    dz: ti.f32,
    ext_coef: ti.f32
):
    """
    Compute transmissivity of radiation between two points through canopy.
    
    This is used for surface-to-surface reflections where reflected radiation
    must pass through any intervening vegetation.
    
    Args:
        pos_from: Start position (emitting surface center)
        pos_to: End position (receiving surface center)
        lad: 3D field of Leaf Area Density
        is_solid: 3D field of solid cells (buildings/terrain)
        nx, ny, nz: Grid dimensions
        dx, dy, dz: Cell sizes
        ext_coef: Extinction coefficient
    
    Returns:
        Tuple of (transmissivity, blocked_by_solid)
        - transmissivity: 0-1 fraction of radiation that gets through
        - blocked_by_solid: 1 if ray hits a solid cell, 0 otherwise
    """
    # Compute ray direction and distance
    diff = pos_to - pos_from
    dist = diff.norm()
    
    transmissivity = 1.0
    blocked = 0
    
    # Only trace if distance is significant
    if dist >= 0.01:
        ray_dir = diff / dist
        
        # Starting voxel
        pos = pos_from + ray_dir * 0.01  # Slight offset to avoid self-intersection
        
        ix = ti.cast(ti.floor(pos[0] / dx), ti.i32)
        iy = ti.cast(ti.floor(pos[1] / dy), ti.i32)
        iz = ti.cast(ti.floor(pos[2] / dz), ti.i32)
        
        # Clamp to valid range
        ix = ti.max(0, ti.min(nx - 1, ix))
        iy = ti.max(0, ti.min(ny - 1, iy))
        iz = ti.max(0, ti.min(nz - 1, iz))
        
        # Step directions
        step_x = 1 if ray_dir[0] >= 0 else -1
        step_y = 1 if ray_dir[1] >= 0 else -1
        step_z = 1 if ray_dir[2] >= 0 else -1
        
        # Initialize DDA variables
        t_max_x = 1e30
        t_max_y = 1e30
        t_max_z = 1e30
        t_delta_x = 1e30
        t_delta_y = 1e30
        t_delta_z = 1e30
        
        t = 0.01  # Start offset
        
        if ti.abs(ray_dir[0]) > 1e-10:
            if step_x > 0:
                t_max_x = ((ix + 1) * dx - pos_from[0]) / ray_dir[0]
            else:
                t_max_x = (ix * dx - pos_from[0]) / ray_dir[0]
            t_delta_x = ti.abs(dx / ray_dir[0])
        
        if ti.abs(ray_dir[1]) > 1e-10:
            if step_y > 0:
                t_max_y = ((iy + 1) * dy - pos_from[1]) / ray_dir[1]
            else:
                t_max_y = (iy * dy - pos_from[1]) / ray_dir[1]
            t_delta_y = ti.abs(dy / ray_dir[1])
        
        if ti.abs(ray_dir[2]) > 1e-10:
            if step_z > 0:
                t_max_z = ((iz + 1) * dz - pos_from[2]) / ray_dir[2]
            else:
                t_max_z = (iz * dz - pos_from[2]) / ray_dir[2]
            t_delta_z = ti.abs(dz / ray_dir[2])
        
        t_prev = t
        max_steps = nx + ny + nz
        done = 0
        
        for _ in range(max_steps):
            if done == 1:
                continue  # Skip remaining iterations
            
            if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                done = 1
                continue
            if t > dist:  # Reached target
                done = 1
                continue
            
            # Check for solid obstruction (but skip first and last cell as they're the surfaces)
            if is_solid[ix, iy, iz] == 1 and t > 0.1 and t < dist - 0.1:
                blocked = 1
                transmissivity = 0.0
                done = 1
                continue
            
            # Get step distance
            t_next = t_max_x
            if t_max_y < t_next:
                t_next = t_max_y
            if t_max_z < t_next:
                t_next = t_max_z
            
            # Limit to target distance
            t_next = ti.min(t_next, dist)
            
            # Path length through this cell
            path_len = t_next - t_prev
            
            # Accumulate absorption from LAD
            cell_lad = lad[ix, iy, iz]
            if cell_lad > 0.0:
                # Beer-Lambert: T = exp(-ext_coef * LAD * path)
                transmissivity *= ti.exp(-ext_coef * cell_lad * path_len)
            
            t_prev = t_next
            
            # Step to next voxel
            if t_max_x < t_max_y and t_max_x < t_max_z:
                t = t_max_x
                ix += step_x
                t_max_x += t_delta_x
            elif t_max_y < t_max_z:
                t = t_max_y
                iy += step_y
                t_max_y += t_delta_y
            else:
                t = t_max_z
                iz += step_z
                t_max_z += t_delta_z
    
    return transmissivity, blocked


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


@ti.func
def sample_hemisphere_direction(i_azim: ti.i32, i_elev: ti.i32, n_azim: ti.i32, n_elev: ti.i32) -> Vector3:
    """
    Generate a direction on the upper hemisphere.
    
    Args:
        i_azim: Azimuthal index (0 to n_azim-1)
        i_elev: Elevation index (0 to n_elev-1)
        n_azim: Number of azimuthal divisions
        n_elev: Number of elevation divisions
    
    Returns:
        Unit direction vector
    """
    PI = 3.14159265359
    
    # Elevation angle (from zenith)
    elev = (i_elev + 0.5) * (PI / 2.0) / n_elev
    
    # Azimuth angle
    azim = (i_azim + 0.5) * (2.0 * PI) / n_azim
    
    # Convert to Cartesian (z up)
    sin_elev = ti.sin(elev)
    cos_elev = ti.cos(elev)
    
    x = sin_elev * ti.sin(azim)
    y = sin_elev * ti.cos(azim)
    z = cos_elev
    
    return Vector3(x, y, z)


@ti.func
def hemisphere_solid_angle(i_elev: ti.i32, n_azim: ti.i32, n_elev: ti.i32) -> ti.f32:
    """
    Calculate solid angle for a hemisphere segment.
    """
    PI = 3.14159265359
    
    elev_low = i_elev * (PI / 2.0) / n_elev
    elev_high = (i_elev + 1) * (PI / 2.0) / n_elev
    
    d_omega = (2.0 * PI / n_azim) * (ti.cos(elev_low) - ti.cos(elev_high))
    
    return d_omega
