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
from .surface_override import NO_PATCH

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

# The `my_patch >= 0` guards in ray_voxel_first_hit_skip_patch and
# ray_canopy_absorption_skip_patch below are deliberately NOT `my_patch !=
# NO_PATCH`. build_cell_patch_grid (surface_override.py) treats *any*
# negative id as unpatched (`has_patch = (patch >= 0) & ...`), so a stray
# id like -5 must disable skipping too, not just exactly NO_PATCH. This
# assertion documents -- and would catch -- NO_PATCH ever becoming
# non-negative, which would silently break that equivalence.
assert NO_PATCH < 0, "my_patch >= 0 guards below assume NO_PATCH is negative"


@ti.func
def ray_voxel_first_hit_skip_patch(
    ray_origin: Vector3,
    ray_dir: Vector3,
    is_solid: ti.template(),
    cell_patch: ti.template(),
    my_patch: ti.i32,
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    dx: ti.f32,
    dy: ti.f32,
    dz: ti.f32,
    max_dist: ti.f32
):
    """
    3D-DDA ray marching to find first solid voxel hit, ignoring voxels that
    belong to the ray's own source patch.

    Why this exists: a voxelised building facade is a staircase of
    axis-aligned faces. A ray leaving a planar polygon patch at a shallow
    angle can immediately re-enter a concave notch of that same patch's own
    voxelisation and be stopped by it -- a self-occlusion artifact, not a
    real shadow. A planar patch cannot legitimately occlude a ray leaving
    its own front side, so any voxel tagged with the ray's own patch id is
    treated as transparent for this ray. Voxels belonging to a different
    patch (a neighbouring building, etc.) still block normally.

    This is a verbatim copy of ``ray_voxel_first_hit`` from
    ``simulator_gpu.raytracing`` with one line changed (the solid test
    inside the DDA loop). It is a copy rather than a parameterisation of
    the shared function because that function has callers outside the
    solar package that have no notion of patches. When ``my_patch < 0``
    the extra condition can never trigger, so this function is bit-identical
    to the original.

    Args:
        ray_origin: Ray origin
        ray_dir: Ray direction (normalized)
        is_solid: 3D field of solid cells
        cell_patch: 3D field mapping each cell to the polygon patch id that
            "owns" it (-1 if the cell isn't owned by any patch)
        my_patch: The patch id the ray originates from (-1 disables skipping)
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
                # the only line that differs from ray_voxel_first_hit
                # my_patch < 0 disables the skip. Any negative id counts as unpatched, matching
                # build_cell_patch_grid in surface_override.py -- deliberately NOT `!= NO_PATCH`,
                # which would let a stray -5 enable skipping against a grid that never seeded it.
                elif (is_solid[ix, iy, iz] == 1
                      and not (my_patch >= 0 and cell_patch[ix, iy, iz] == my_patch)):
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
def ray_canopy_absorption_skip_patch(
    ray_origin: Vector3,
    ray_dir: Vector3,
    lad: ti.template(),
    is_solid: ti.template(),
    cell_patch: ti.template(),
    my_patch: ti.i32,
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
    Trace ray through canopy computing Beer-Lambert absorption, ignoring
    solid voxels that belong to the ray's own source patch.

    Why this exists: the canopy shadow path folds the solid check into the
    same DDA march as the Beer-Lambert absorption accumulation, so simply
    running the plain skip-DDA (``ray_voxel_first_hit_skip_patch``) first
    and absorption second would still zero the transmissivity on the
    surface's own voxelisation staircase -- the solid test has to be
    patch-aware inside this single march too. As with the first-hit
    variant, a planar patch cannot legitimately occlude a ray leaving its
    own front side, so voxels tagged with the ray's own patch id are
    treated as transparent (canopy absorption in those cells, if any, is
    still accumulated as normal -- only the *solid* short-circuit is
    skipped).

    This is a verbatim copy of ``ray_canopy_absorption`` from
    ``simulator_gpu.raytracing`` with one line changed (the solid test
    inside the DDA loop). It is a copy rather than a parameterisation of
    the shared function because that function has callers outside the
    solar package that have no notion of patches. When ``my_patch < 0``
    the extra condition can never trigger, so this function is bit-identical
    to the original.

    Args:
        ray_origin: Ray origin
        ray_dir: Ray direction (normalized)
        lad: 3D field of Leaf Area Density
        is_solid: 3D field of solid cells (buildings/terrain)
        cell_patch: 3D field mapping each cell to the polygon patch id that
            "owns" it (-1 if the cell isn't owned by any patch)
        my_patch: The patch id the ray originates from (-1 disables skipping)
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
                # the only line that differs from ray_canopy_absorption
                # my_patch < 0 disables the skip. Any negative id counts as unpatched, matching
                # build_cell_patch_grid in surface_override.py -- deliberately NOT `!= NO_PATCH`,
                # which would let a stray -5 enable skipping against a grid that never seeded it.
                elif (is_solid[ix, iy, iz] == 1
                      and not (my_patch >= 0 and cell_patch[ix, iy, iz] == my_patch)):
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
        surf_normal: ti.template(),
        surf_patch: ti.template(),
        cell_patch: ti.template(),
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template(),
        n_surf: ti.i32,
        shadow_factor: ti.template()
    ):
        """
        Compute shadow factors for all surfaces.

        shadow_factor = 0 means fully sunlit
        shadow_factor = 1 means fully shaded

        The facing test and the shadow ray both use the surface's stored
        normal (``surf_normal``), not a per-direction sign switch. For
        surfaces built from occupancy (``extract_surfaces_from_domain``)
        that normal is exactly one of the six axis vectors, so
        ``dot(normal, sun_dir) > 0`` reduces algebraically to the old
        per-direction sign checks (e.g. direction 4 / IEAST's old test
        ``sun_dir[0] > 0`` *is* ``dot((1,0,0), sun_dir) > 0``) -- this
        kernel is behaviourally identical for those surfaces. For a true,
        non-axis-aligned normal (e.g. a 45-degree wall from
        ``surfaces_from_override``) the dot product test is also the
        *correct* one: the old switch would keep declaring such a face
        "away from sun" the moment the sun left its dominant axis, even
        while the true surface was still lit.

        The shadow ray is cast with ``ray_voxel_first_hit_skip_patch``,
        which treats any voxel tagged with the ray's own patch id
        (``surf_patch[i]``) as transparent. This is what removes the
        staircase self-occlusion: a voxelised slanted facade is a stair-step
        of axis-aligned faces, and a ray leaving one step at a shallow angle
        would otherwise immediately re-enter a concave notch of its own
        voxelisation and be reported as shadowed by itself. Surfaces with
        ``surf_patch[i] < 0`` (every occupancy-built surface) disable the
        skip entirely, so the DDA march is bit-identical to the original
        ``ray_voxel_first_hit`` in that case.
        """
        # Small offset to ensure ray origin is outside the solid voxel
        eps = 0.01

        for i in range(n_surf):
            pos = surf_pos[i]
            normal = surf_normal[i]

            cos_inc = (sun_dir[0] * normal[0] + sun_dir[1] * normal[1]
                       + sun_dir[2] * normal[2])

            if cos_inc <= 0.0:
                shadow_factor[i] = 1.0
            else:
                ray_origin = Vector3(pos[0] + normal[0] * eps,
                                     pos[1] + normal[1] * eps,
                                     pos[2] + normal[2] * eps)

                hit, _, _, _, _ = ray_voxel_first_hit_skip_patch(
                    ray_origin, sun_dir,
                    is_solid, cell_patch, surf_patch[i],
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
        surf_normal: ti.template(),
        surf_patch: ti.template(),
        cell_patch: ti.template(),
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template(),
        lad: ti.template(),
        n_surf: ti.i32,
        shadow_factor: ti.template(),
        canopy_transmissivity: ti.template()
    ):
        """
        Compute shadow factors including canopy absorption.

        Same restructuring as ``compute_direct_shadows``: the facing test
        and the shadow/absorption ray both use the surface's stored normal
        (``surf_normal``) rather than a per-direction sign switch, which is
        behaviourally identical for the axis normals produced by
        ``extract_surfaces_from_domain`` and correct for the true,
        non-axis-aligned normals produced by ``surfaces_from_override``.

        This is the live path -- ``Domain.lad`` is always an allocated
        Taichi field, so ``compute_shortwave_radiation`` always calls this
        kernel rather than ``compute_direct_shadows``. The shadow/absorption
        march itself uses ``ray_canopy_absorption_skip_patch``, which folds
        the same own-patch skip into the Beer-Lambert accumulation: a voxel
        tagged with the ray's own patch id (``surf_patch[i]``) never
        short-circuits the march as "solid" (though canopy absorption in
        that cell, if any, still accumulates normally). That skip is what
        removes the staircase self-occlusion from a voxelised slanted
        facade. Surfaces with ``surf_patch[i] < 0`` disable the skip, so the
        march is bit-identical to the original ``ray_canopy_absorption``.
        """
        # Small offset to ensure ray origin is outside the solid voxel
        eps = 0.01

        for i in range(n_surf):
            pos = surf_pos[i]
            normal = surf_normal[i]

            cos_inc = (sun_dir[0] * normal[0] + sun_dir[1] * normal[1]
                       + sun_dir[2] * normal[2])

            if cos_inc <= 0.0:
                shadow_factor[i] = 1.0
                canopy_transmissivity[i] = 0.0
            else:
                ray_origin = Vector3(pos[0] + normal[0] * eps,
                                     pos[1] + normal[1] * eps,
                                     pos[2] + normal[2] * eps)

                trans, _ = ray_canopy_absorption_skip_patch(
                    ray_origin, sun_dir,
                    lad, is_solid, cell_patch, surf_patch[i],
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
    'ray_voxel_first_hit_skip_patch',
    'ray_canopy_absorption',
    'ray_canopy_absorption_skip_patch',
    'ray_voxel_transmissivity',
    'ray_trace_to_target',
    'ray_point_to_point_transmissivity',
    'sample_hemisphere_direction',
    'hemisphere_solid_angle',
]
