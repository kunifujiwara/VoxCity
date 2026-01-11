"""
Shared ray tracing module for simulator_gpu.

Implements GPU-accelerated ray tracing through a 3D voxel grid
for both solar radiation and view analysis.
Uses 3D-DDA (Digital Differential Analyzer) for voxel traversal.

Key features:
- Beer-Lambert law for canopy: trans = exp(-ext_coef * LAD * path_length)
- Solid obstacles block rays completely (trans = 0)
- Tree canopy attenuates rays based on LAD and path length
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
        max_steps = nx + ny + nz
        done = 0
        
        for _ in range(max_steps):
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
def ray_voxel_transmissivity(
    ray_origin: Vector3,
    ray_dir: Vector3,
    is_solid: ti.template(),
    is_tree: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    dx: ti.f32,
    dy: ti.f32,
    dz: ti.f32,
    max_dist: ti.f32,
    tree_k: ti.f32,
    tree_lad: ti.f32
):
    """
    3D-DDA ray marching with tree canopy transmissivity calculation.
    
    Args:
        ray_origin: Ray origin
        ray_dir: Ray direction (normalized)
        is_solid: 3D field of solid cells (buildings, ground)
        is_tree: 3D field of tree cells
        nx, ny, nz: Grid dimensions
        dx, dy, dz: Cell sizes
        max_dist: Maximum ray distance
        tree_k: Tree extinction coefficient
        tree_lad: Leaf area density
    
    Returns:
        Tuple of (blocked_by_solid, transmissivity)
        - blocked_by_solid: 1 if ray hit solid, 0 otherwise
        - transmissivity: 0-1 fraction of light that gets through trees
    """
    blocked = 0
    transmissivity = 1.0
    
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
            if done == 0:
                if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                    done = 1
                elif t > t_exit:
                    done = 1
                elif is_solid[ix, iy, iz] == 1:
                    blocked = 1
                    transmissivity = 0.0
                    done = 1
                else:
                    # Get step distance
                    t_next = ti.min(t_max_x, ti.min(t_max_y, t_max_z))
                    
                    # Path length through this cell
                    path_len = t_next - t_prev
                    
                    # Accumulate absorption from tree canopy
                    if is_tree[ix, iy, iz] == 1:
                        # Beer-Lambert: T = exp(-k * LAD * path)
                        segment_trans = ti.exp(-tree_k * tree_lad * path_len)
                        transmissivity *= segment_trans
                        
                        # Early termination if transmissivity is negligible
                        if transmissivity < 0.01:
                            done = 1
                    
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
    
    return blocked, transmissivity


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
            if done == 0:
                if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
                    done = 1
                elif t > t_exit:
                    done = 1
                elif is_solid[ix, iy, iz] == 1:
                    transmissivity = 0.0
                    done = 1
                else:
                    # Get step distance
                    t_next = ti.min(t_max_x, ti.min(t_max_y, t_max_z))
                    
                    # Path length through this cell
                    path_len = t_next - t_prev
                    
                    # Accumulate absorption from LAD
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
def ray_trace_to_target(
    origin: Vector3,
    target: Vector3,
    is_solid: ti.template(),
    is_tree: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    dx: ti.f32,
    dy: ti.f32,
    dz: ti.f32,
    tree_att: ti.f32,
    att_cutoff: ti.f32
):
    """
    Trace ray from origin to target, checking for visibility.
    
    Args:
        origin: Start position (in voxel coordinates)
        target: End position (in voxel coordinates)
        is_solid: 3D field of solid cells
        is_tree: 3D field of tree cells
        nx, ny, nz: Grid dimensions
        dx, dy, dz: Cell sizes (typically all 1.0 for voxel coords)
        tree_att: Attenuation factor per voxel for trees
        att_cutoff: Minimum transmissivity before considering blocked
    
    Returns:
        1 if target is visible, 0 otherwise
    """
    diff = target - origin
    dist = diff.norm()
    
    if dist < 0.01:
        return 1
    
    ray_dir = diff / dist
    
    x, y, z = origin[0] + 0.5, origin[1] + 0.5, origin[2] + 0.5
    i = ti.cast(ti.floor(origin[0]), ti.i32)
    j = ti.cast(ti.floor(origin[1]), ti.i32)
    k = ti.cast(ti.floor(origin[2]), ti.i32)
    
    ti_x = ti.cast(ti.floor(target[0]), ti.i32)
    tj_y = ti.cast(ti.floor(target[1]), ti.i32)
    tk_z = ti.cast(ti.floor(target[2]), ti.i32)
    
    step_x = 1 if ray_dir[0] >= 0 else -1
    step_y = 1 if ray_dir[1] >= 0 else -1
    step_z = 1 if ray_dir[2] >= 0 else -1
    
    BIG = 1e30
    t_max_x, t_max_y, t_max_z = BIG, BIG, BIG
    t_delta_x, t_delta_y, t_delta_z = BIG, BIG, BIG
    
    if ray_dir[0] != 0.0:
        t_max_x = (((i + (1 if step_x > 0 else 0)) - x) / ray_dir[0])
        t_delta_x = ti.abs(1.0 / ray_dir[0])
    if ray_dir[1] != 0.0:
        t_max_y = (((j + (1 if step_y > 0 else 0)) - y) / ray_dir[1])
        t_delta_y = ti.abs(1.0 / ray_dir[1])
    if ray_dir[2] != 0.0:
        t_max_z = (((k + (1 if step_z > 0 else 0)) - z) / ray_dir[2])
        t_delta_z = ti.abs(1.0 / ray_dir[2])
    
    T = 1.0
    visible = 1
    max_steps = nx + ny + nz
    done = 0
    
    for _ in range(max_steps):
        if done == 0:
            if i < 0 or i >= nx or j < 0 or j >= ny or k < 0 or k >= nz:
                visible = 0
                done = 1
            elif is_solid[i, j, k] == 1:
                visible = 0
                done = 1
            elif is_tree[i, j, k] == 1:
                T *= tree_att
                if T < att_cutoff:
                    visible = 0
                    done = 1
            
            if done == 0:
                # Check if we've reached the target
                if i == ti_x and j == tj_y and k == tk_z:
                    done = 1
                else:
                    # Step to next voxel
                    if t_max_x < t_max_y:
                        if t_max_x < t_max_z:
                            t_max_x += t_delta_x
                            i += step_x
                        else:
                            t_max_z += t_delta_z
                            k += step_z
                    else:
                        if t_max_y < t_max_z:
                            t_max_y += t_delta_y
                            j += step_y
                        else:
                            t_max_z += t_delta_z
                            k += step_z
    
    return visible


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
