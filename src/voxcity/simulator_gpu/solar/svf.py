"""
Sky View Factor (SVF) calculation for palm-solar.

Computes the fraction of sky hemisphere visible from each surface element.
Uses GPU-accelerated ray tracing to sample the hemisphere.

Coordinate System:
    SVF calculations use a local ENU-style coordinate system for hemisphere
    sampling (x=East-like, y=North-like, z=Up). This is self-consistent
    within SVF and does not affect sun direction calculations.
    
    Direction indices follow PALM naming convention:
    - 0 (IUP): +z, upward-facing
    - 1 (IDOWN): -z, downward-facing
    - 2 (INORTH): +y normal (East-facing in geographic terms)
    - 3 (ISOUTH): -y normal (West-facing in geographic terms)
    - 4 (IEAST): +x normal (South-facing in geographic terms)
    - 5 (IWEST): -x normal (North-facing in geographic terms)

PALM Alignment:
- Uses PALM's vffrac_up formula: (cos(2*elev_low) - cos(2*elev_high)) / (2*n_azim)
- Default discretization: n_azimuth=80, n_elevation=40 (PALM defaults)
- svf output is equivalent to PALM's skyvft (transmissivity-weighted sky view factor)
- svf_urban output is equivalent to PALM's skyvf (urban-only, no canopy)
- Ray accumulation: SUM(vffrac * trans) matching PALM's methodology
  PALM: skyvft = SUM(ztransp * vffrac, MASK=(itarget < 0))
"""

import taichi as ti
import math
from typing import Tuple, Optional

from .core import Vector3, Point3, PI, TWO_PI
from .raytracing import (
    ray_voxel_first_hit,
    ray_voxel_first_hit_skip_patch,
    ray_canopy_absorption,
    ray_canopy_absorption_skip_patch,
    sample_hemisphere_direction,
    hemisphere_solid_angle,
)


def unobstructed_svf_for_normal(normal, n_azimuth: int = 72,
                                 n_elevation: int = 36) -> float:
    """Cosine-weighted sky fraction for an unobstructed surface, ground below.

    Full-sphere reference for the kernel's override branch. The closed form is
    (1 + n_z)/2; this integrates it numerically so the kernel and the maths can
    be compared through a similar discretisation rather than against an ideal.

    Note this reference uses elevation measured **from the horizon** (its own
    convention, hence ``cos(E)`` for the solid angle) -- deliberately
    different from the kernel's zenith convention (``compute_svf`` measures
    elevation from the zenith, weighting bands by ``sin``), so that agreement
    between the two is meaningful rather than a shared assumption.
    """
    import numpy as np

    n = np.asarray(normal, dtype=np.float64)
    n = n / max(np.linalg.norm(n), 1e-12)

    az = (np.arange(n_azimuth) + 0.5) * (2.0 * np.pi / n_azimuth)
    el = (np.arange(2 * n_elevation) + 0.5) * (np.pi / (2 * n_elevation)) - np.pi / 2
    A, E = np.meshgrid(az, el, indexing="ij")
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], axis=-1)
    solid = np.cos(E)                                  # dOmega weight

    w = np.clip(d @ n, 0.0, None) * solid
    total = w.sum()
    if total <= 0:
        return 1.0
    return float(w[d[..., 2] > 0].sum() / total)


@ti.data_oriented
class SVFCalculator:
    """
    GPU-accelerated Sky View Factor calculator.
    
    Computes SVF by tracing rays from each surface to the hemisphere.
    SVF = fraction of hemisphere visible from surface.
    """
    
    def __init__(self, domain, n_azimuth: int = 80, n_elevation: int = 40):
        """
        Initialize SVF calculator.
        
        Args:
            domain: Domain object with grid geometry
            n_azimuth: Number of azimuthal divisions (default 80)
            n_elevation: Number of elevation divisions (default 40)
        """
        self.domain = domain
        self.nx = domain.nx
        self.ny = domain.ny
        self.nz = domain.nz
        self.dx = domain.dx
        self.dy = domain.dy
        self.dz = domain.dz
        
        self.n_azimuth = n_azimuth
        self.n_elevation = n_elevation
        self.n_directions = n_azimuth * n_elevation
        
        # Maximum ray distance
        self.max_dist = math.sqrt(
            (self.nx * self.dx)**2 + 
            (self.ny * self.dy)**2 + 
            (self.nz * self.dz)**2
        )
        
        # Pre-compute directions and view factor fractions
        # PALM uses separate vffrac_up (for upward surfaces) and vffrac_vert (for vertical surfaces)
        # See radiation_model_mod.f90 lines 12093-12105
        # vffrac_up is pre-computed; vffrac_vert is computed on-the-fly since it depends on surface orientation
        self.directions = ti.Vector.field(3, dtype=ti.f32, shape=(n_azimuth, n_elevation))
        self.solid_angles = ti.field(dtype=ti.f32, shape=(n_azimuth, n_elevation))  # vffrac_up for upward surfaces
        self.total_solid_angle = ti.field(dtype=ti.f32, shape=())
        
        self._init_directions()
    
    @ti.kernel
    def _init_directions(self):
        """
        Pre-compute hemisphere directions and view factor fractions for upward surfaces.
        
        Uses PALM's formulas for view factor fractions (radiation_model_mod.f90 lines 12093-12105):
        
        1. For upward surfaces (vffrac_up, pre-computed here):
           vffrac_up = (cos(2*elev_low) - cos(2*elev_high)) / (2*n_azim)
           This accounts for cosine-weighted projected area on horizontal surface.
        
        2. For vertical surfaces (vffrac_vert, computed on-the-fly in compute_svf):
           vffrac_vert = (sin(az2) - sin(az1)) * elev_terms / (2*pi)
           where elev_terms = elev_high - elev_low + sin(elev_low)*cos(elev_low) - sin(elev_high)*cos(elev_high)
           Azimuth is measured relative to surface normal: az = azim_angle_relative - pi/2
           
        Elevation angle is measured from zenith (0 = up, π/2 = horizon).
        The sum of vffrac_up over all rays in the upper hemisphere equals 1.0.
        """
        total_omega = 0.0
        n_azim_f = ti.cast(self.n_azimuth, ti.f32)
        n_elev_f = ti.cast(self.n_elevation, ti.f32)
        
        d_azim = TWO_PI / n_azim_f  # Azimuth step size
        
        for i_azim, i_elev in ti.ndrange(self.n_azimuth, self.n_elevation):
            # Elevation boundaries (from zenith)
            elev_low = ti.cast(i_elev, ti.f32) * (PI / 2.0) / n_elev_f
            elev_high = ti.cast(i_elev + 1, ti.f32) * (PI / 2.0) / n_elev_f
            elev_center = (elev_low + elev_high) / 2.0
            
            # Azimuth center
            azim_center = (ti.cast(i_azim, ti.f32) + 0.5) * d_azim
            
            # Direction vector (x=East, y=North, z=Up)
            sin_elev = ti.sin(elev_center)
            cos_elev = ti.cos(elev_center)
            
            x = sin_elev * ti.sin(azim_center)
            y = sin_elev * ti.cos(azim_center)
            z = cos_elev
            
            self.directions[i_azim, i_elev] = Vector3(x, y, z)
            
            # View factor fraction for upward-facing surfaces (PALM formula)
            # vffrac_up = (cos(2*elev_low) - cos(2*elev_high)) / (2*n_azim)
            vf_up = (ti.cos(2.0 * elev_low) - ti.cos(2.0 * elev_high)) / (2.0 * n_azim_f)
            self.solid_angles[i_azim, i_elev] = vf_up
            total_omega += vf_up
        
        self.total_solid_angle[None] = total_omega
    
    @ti.kernel
    def compute_svf(
        self,
        surf_pos: ti.template(),
        surf_dir: ti.template(),
        surf_normal: ti.template(),
        surf_patch: ti.template(),
        cell_patch: ti.template(),
        is_solid: ti.template(),
        n_surf: ti.i32,
        svf: ti.template()
    ):
        """
        Compute Sky View Factor for all surfaces.

        Uses PALM's methodology (radiation_model_mod.f90):
        - For upward surfaces: vffrac_up = (cos(2*elev_low) - cos(2*elev_high)) / (2*n_azim)
        - For vertical surfaces: vffrac_vert = (sin(az2) - sin(az1)) * elev_terms / (2*pi)
          where az is measured relative to surface normal and elev_terms accounts for
          the elevation integration.

        Surfaces with surf_patch[i] >= 0 (an externally supplied true normal,
        see surface_override.py) take a second branch instead: cosine-weight
        against the true normal, rescale by the analytic (1 + n_z) / 2 sky
        fraction (the direction grid only covers the upper hemisphere), and
        skip the ray's own patch during occlusion. See the in-loop comments
        for the physics. Surfaces with surf_patch[i] < 0 take the untouched
        six-way analytic branch, bit-identical to before this branch existed.

        Args:
            surf_pos: Surface positions (n_surf, 3)
            surf_dir: Surface directions (n_surf,)
            surf_normal: True surface normals (n_surf,) -- only read when
                surf_patch[i] >= 0
            surf_patch: Coplanar patch id per surface (n_surf,); < 0 means
                "no override, use the six-way analytic branch"
            cell_patch: Per-cell patch id grid for the own-patch ray skip
            is_solid: 3D field of solid cells
            n_surf: Number of surfaces
            svf: Output SVF values (n_surf,)
        """
        n_azim_f = ti.cast(self.n_azimuth, ti.f32)
        n_elev_f = ti.cast(self.n_elevation, ti.f32)
        d_azim = TWO_PI / n_azim_f  # Azimuth step size
        
        for i in range(n_surf):
            pos = Vector3(surf_pos[i][0], surf_pos[i][1], surf_pos[i][2])
            direction = surf_dir[i]
            
            use_override = surf_patch[i] >= 0

            # Get surface normal and azimuth of normal (for vertical surfaces)
            normal = Vector3(0.0, 0.0, 0.0)
            normal_azim = 0.0  # Azimuth angle of surface normal (for vertical surfaces)

            if use_override:
                # A caller-supplied true polygon normal replaces the axis
                # normal entirely; normal_azim stays 0 and is never read
                # below because the override branch does not use it.
                normal = surf_normal[i]
            elif direction == 0:  # Up
                normal = Vector3(0.0, 0.0, 1.0)
            elif direction == 1:  # Down
                normal = Vector3(0.0, 0.0, -1.0)
            elif direction == 2:  # INORTH: +y normal (East-facing in geographic terms)
                normal = Vector3(0.0, 1.0, 0.0)
                normal_azim = 0.0  # +y is azimuth 0 in this local coordinate system
            elif direction == 3:  # ISOUTH: -y normal (West-facing in geographic terms)
                normal = Vector3(0.0, -1.0, 0.0)
                normal_azim = PI  # -y is azimuth π
            elif direction == 4:  # IEAST: +x normal (South-facing in geographic terms)
                normal = Vector3(1.0, 0.0, 0.0)
                normal_azim = PI / 2.0  # +x is azimuth π/2
            elif direction == 5:  # IWEST: -x normal (North-facing in geographic terms)
                normal = Vector3(-1.0, 0.0, 0.0)
                normal_azim = 3.0 * PI / 2.0  # -x is azimuth 3π/2

            eps = ti.min(self.dx, ti.min(self.dy, self.dz)) * 1e-4
            ray_origin = Vector3(
                pos[0] + normal[0] * eps,
                pos[1] + normal[1] * eps,
                pos[2] + normal[2] * eps,
            )

            visible_vf = 0.0
            total_vf = 0.0

            # Trace rays to all hemisphere directions
            for i_azim, i_elev in ti.ndrange(self.n_azimuth, self.n_elevation):
                ray_dir = self.directions[i_azim, i_elev]

                # Check if direction is above surface (dot product with normal > 0)
                cos_angle = ray_dir[0] * normal[0] + ray_dir[1] * normal[1] + ray_dir[2] * normal[2]

                if cos_angle > 0.001:  # Small threshold to avoid numerical issues
                    # Compute view factor fraction based on surface orientation
                    elev_low = ti.cast(i_elev, ti.f32) * (PI / 2.0) / n_elev_f
                    elev_high = ti.cast(i_elev + 1, ti.f32) * (PI / 2.0) / n_elev_f

                    vf_frac = 0.0
                    if use_override:
                        # Elevation is measured FROM THE ZENITH in this grid
                        # (z = cos(elev)), so a band's solid angle goes as
                        # sin(elev_mid). Cosine-weight against the true normal.
                        vf_frac = cos_angle * ti.sin(0.5 * (elev_low + elev_high))
                    elif direction == 0:  # Upward surface
                        # PALM: vffrac_up = (cos(2*elev_low) - cos(2*elev_high)) / (2*n_azim)
                        vf_frac = (ti.cos(2.0 * elev_low) - ti.cos(2.0 * elev_high)) / (2.0 * n_azim_f)
                    elif direction == 1:  # Downward surface
                        # Use same formula as upward (symmetric)
                        vf_frac = (ti.cos(2.0 * elev_low) - ti.cos(2.0 * elev_high)) / (2.0 * n_azim_f)
                    else:
                        # Vertical surfaces: use PALM's vffrac_vert formula
                        # Compute azimuth relative to surface normal
                        azim_low = ti.cast(i_azim, ti.f32) * d_azim
                        azim_high = ti.cast(i_azim + 1, ti.f32) * d_azim

                        # Relative azimuth measured from the surface normal.
                        az1_rel = azim_low - normal_azim
                        az2_rel = azim_high - normal_azim

                        # Elevation terms for vertical surface
                        # elev_terms = elev_high - elev_low + sin(elev_low)*cos(elev_low) - sin(elev_high)*cos(elev_high)
                        elev_terms = (elev_high - elev_low
                                     + ti.sin(elev_low) * ti.cos(elev_low)
                                     - ti.sin(elev_high) * ti.cos(elev_high))

                        # vffrac_vert = (sin(az2) - sin(az1)) * elev_terms / (2*π)
                        vf_frac = (ti.sin(az2_rel) - ti.sin(az1_rel)) * elev_terms / TWO_PI

                        # Only positive contributions (ray in front of surface)
                        if vf_frac < 0.0:
                            vf_frac = 0.0

                    total_vf += vf_frac

                    # Trace ray, skipping voxels tagged with this surface's own
                    # patch (a bit-identical no-op when surf_patch[i] < 0).
                    hit, _, _, _, _ = ray_voxel_first_hit_skip_patch(
                        ray_origin, ray_dir,
                        is_solid, cell_patch, surf_patch[i],
                        self.nx, self.ny, self.nz,
                        self.dx, self.dy, self.dz,
                        self.max_dist
                    )

                    if hit == 0:
                        visible_vf += vf_frac

            if use_override:
                # The direction grid covers the upper hemisphere only, but a
                # tilted surface's front hemisphere dips below the horizontal,
                # where every ray ends in ground. That analytic fraction is
                # (1 + n_z)/2 -- for a vertical wall it reproduces the old
                # doubling exactly, for a flat roof it is 1.
                if total_vf > 0.001:
                    svf[i] = visible_vf / total_vf * (1.0 + normal[2]) * 0.5
                else:
                    # No grid direction is in front of this surface at all
                    # (it faces downward). The old fallback of 1.0 would be
                    # badly wrong; the analytic value is right.
                    svf[i] = ti.max(0.0, (1.0 + normal[2]) * 0.5)
            else:
                # For vertical surfaces (direction 2-5), account for ground blocking
                # Directions with z < 0 see ground, not sky
                # We sample hemisphere with z > 0, but for vertical walls the mirrored rays (z < 0)
                # would have same view factor contribution and are ALL blocked by ground
                # This effectively doubles the total and leaves visible unchanged
                if direction >= 2:  # Vertical surfaces
                    # The lower hemisphere (z < 0) contribution equals upper hemisphere (z > 0) by symmetry
                    # All lower hemisphere rays are blocked by ground
                    total_vf = total_vf * 2.0

                # Normalize SVF so that unobstructed surface = 1.0
                if total_vf > 0.001:
                    svf[i] = visible_vf / total_vf
                else:
                    svf[i] = 1.0
    
    @ti.kernel
    def compute_svf_with_canopy(
        self,
        surf_pos: ti.template(),
        surf_dir: ti.template(),
        surf_normal: ti.template(),
        surf_patch: ti.template(),
        cell_patch: ti.template(),
        is_solid: ti.template(),
        lad: ti.template(),
        n_surf: ti.i32,
        ext_coef: ti.f32,
        svf: ti.template(),
        svf_urban: ti.template()
    ):
        """
        Compute SVF including canopy absorption.

        Uses PALM's methodology (radiation_model_mod.f90):
        - For upward surfaces: vffrac_up = (cos(2*elev_low) - cos(2*elev_high)) / (2*n_azim)
        - For vertical surfaces: vffrac_vert = (sin(az2) - sin(az1)) * elev_terms / (2*pi)

        This is the live path: Domain.lad is always an allocated Taichi
        field, so RadiationModel.compute_svf always calls this kernel rather
        than compute_svf (above). Surfaces with surf_patch[i] >= 0 take the
        same override branch as compute_svf -- see its docstring and in-loop
        comments for the physics; surf_patch[i] < 0 takes the untouched
        six-way analytic branch.

        Args:
            surf_pos: Surface positions
            surf_dir: Surface directions
            surf_normal: True surface normals (n_surf,) -- only read when
                surf_patch[i] >= 0
            surf_patch: Coplanar patch id per surface (n_surf,); < 0 means
                "no override, use the six-way analytic branch"
            cell_patch: Per-cell patch id grid for the own-patch ray skip
            is_solid: 3D solid field
            lad: 3D Leaf Area Density field
            n_surf: Number of surfaces
            ext_coef: Extinction coefficient
            svf: Output SVF with canopy (n_surf,)
            svf_urban: Output SVF without canopy (n_surf,)
        """
        n_azim_f = ti.cast(self.n_azimuth, ti.f32)
        n_elev_f = ti.cast(self.n_elevation, ti.f32)
        d_azim = TWO_PI / n_azim_f

        for i in range(n_surf):
            pos = Vector3(surf_pos[i][0], surf_pos[i][1], surf_pos[i][2])
            direction = surf_dir[i]

            use_override = surf_patch[i] >= 0

            # Get surface normal and azimuth of normal (for vertical surfaces)
            normal = Vector3(0.0, 0.0, 0.0)
            normal_azim = 0.0

            if use_override:
                normal = surf_normal[i]
            elif direction == 0:  # Up
                normal = Vector3(0.0, 0.0, 1.0)
            elif direction == 1:  # Down
                normal = Vector3(0.0, 0.0, -1.0)
            elif direction == 2:  # North
                normal = Vector3(0.0, 1.0, 0.0)
                normal_azim = 0.0
            elif direction == 3:  # South
                normal = Vector3(0.0, -1.0, 0.0)
                normal_azim = PI
            elif direction == 4:  # East
                normal = Vector3(1.0, 0.0, 0.0)
                normal_azim = PI / 2.0
            elif direction == 5:  # West
                normal = Vector3(-1.0, 0.0, 0.0)
                normal_azim = 3.0 * PI / 2.0

            eps = ti.min(self.dx, ti.min(self.dy, self.dz)) * 1e-4
            ray_origin = Vector3(
                pos[0] + normal[0] * eps,
                pos[1] + normal[1] * eps,
                pos[2] + normal[2] * eps,
            )

            visible_omega = 0.0
            visible_omega_urban = 0.0
            total_omega = 0.0

            for i_azim, i_elev in ti.ndrange(self.n_azimuth, self.n_elevation):
                ray_dir = self.directions[i_azim, i_elev]

                cos_angle = ray_dir[0] * normal[0] + ray_dir[1] * normal[1] + ray_dir[2] * normal[2]

                if cos_angle > 0.001:
                    # Compute view factor fraction based on surface orientation
                    elev_low = ti.cast(i_elev, ti.f32) * (PI / 2.0) / n_elev_f
                    elev_high = ti.cast(i_elev + 1, ti.f32) * (PI / 2.0) / n_elev_f

                    vf_frac = 0.0
                    if use_override:
                        # Elevation is measured FROM THE ZENITH in this grid
                        # (z = cos(elev)), so a band's solid angle goes as
                        # sin(elev_mid). Cosine-weight against the true normal.
                        vf_frac = cos_angle * ti.sin(0.5 * (elev_low + elev_high))
                    elif direction == 0 or direction == 1:  # Upward or Downward surface
                        # PALM: vffrac_up
                        vf_frac = (ti.cos(2.0 * elev_low) - ti.cos(2.0 * elev_high)) / (2.0 * n_azim_f)
                    else:
                        # Vertical surfaces: use PALM's vffrac_vert formula
                        azim_low = ti.cast(i_azim, ti.f32) * d_azim
                        azim_high = ti.cast(i_azim + 1, ti.f32) * d_azim

                        # Relative azimuth measured from the surface normal.
                        az1_rel = azim_low - normal_azim
                        az2_rel = azim_high - normal_azim

                        # Elevation terms
                        elev_terms = (elev_high - elev_low
                                     + ti.sin(elev_low) * ti.cos(elev_low)
                                     - ti.sin(elev_high) * ti.cos(elev_high))

                        # vffrac_vert = (sin(az2) - sin(az1)) * elev_terms / (2*π)
                        vf_frac = (ti.sin(az2_rel) - ti.sin(az1_rel)) * elev_terms / TWO_PI

                        if vf_frac < 0.0:
                            vf_frac = 0.0

                    total_omega += vf_frac

                    # Trace with canopy absorption, skipping voxels tagged
                    # with this surface's own patch (a bit-identical no-op
                    # when surf_patch[i] < 0).
                    trans, _ = ray_canopy_absorption_skip_patch(
                        ray_origin, ray_dir,
                        lad, is_solid, cell_patch, surf_patch[i],
                        self.nx, self.ny, self.nz,
                        self.dx, self.dy, self.dz,
                        self.max_dist,
                        ext_coef
                    )

                    # SVF with canopy considers transparency (PALM's skyvft)
                    visible_omega += vf_frac * trans

                    # SVF urban (only solid obstacles, PALM's skyvf)
                    hit, _, _, _, _ = ray_voxel_first_hit_skip_patch(
                        ray_origin, ray_dir,
                        is_solid, cell_patch, surf_patch[i],
                        self.nx, self.ny, self.nz,
                        self.dx, self.dy, self.dz,
                        self.max_dist
                    )
                    if hit == 0:
                        visible_omega_urban += vf_frac

            if use_override:
                # Same (1 + n_z)/2 rescale as compute_svf -- see its
                # in-loop comment for the physics.
                sky_frac = ti.max(0.0, (1.0 + normal[2]) * 0.5)
                if total_omega > 0.001:
                    svf[i] = visible_omega / total_omega * sky_frac
                    svf_urban[i] = visible_omega_urban / total_omega * sky_frac
                else:
                    svf[i] = sky_frac
                    svf_urban[i] = sky_frac
            else:
                # For vertical surfaces (direction 2-5), account for ground blocking
                # Directions with z < 0 see ground, not sky
                # The lower hemisphere has equal total_omega contribution, all blocked by ground
                if direction >= 2:  # Vertical surfaces
                    total_omega = total_omega * 2.0

                if total_omega > 1e-10:
                    svf[i] = visible_omega / total_omega
                    svf_urban[i] = visible_omega_urban / total_omega
                else:
                    svf[i] = 0.0
                    svf_urban[i] = 0.0


@ti.kernel
def compute_svf_grid_kernel(
    topo_top: ti.template(),
    is_solid: ti.template(),
    directions: ti.template(),
    solid_angles: ti.template(),
    nx: ti.i32,
    ny: ti.i32,
    nz: ti.i32,
    dx: ti.f32,
    dy: ti.f32,
    dz: ti.f32,
    n_azim: ti.i32,
    n_elev: ti.i32,
    max_dist: ti.f32,
    svf_grid: ti.template()
):
    """
    Compute SVF for a 2D grid at terrain level.
    
    svf_grid[i, j] = SVF at terrain surface (i, j)
    """
    for i, j in ti.ndrange(nx, ny):
        k = topo_top[i, j]
        
        if k < nz:
            pos = Vector3((i + 0.5) * dx, (j + 0.5) * dy, (k + 0.5) * dz)
            normal = Vector3(0.0, 0.0, 1.0)
            
            visible_omega = 0.0
            total_omega = 0.0
            
            for i_azim, i_elev in ti.ndrange(n_azim, n_elev):
                ray_dir = directions[i_azim, i_elev]
                d_omega = solid_angles[i_azim, i_elev]
                
                cos_angle = ray_dir[2]  # Normal is (0, 0, 1)
                
                if cos_angle > 0:
                    total_omega += d_omega * cos_angle
                    
                    hit, _, _, _, _ = ray_voxel_first_hit(
                        pos, ray_dir,
                        is_solid,
                        nx, ny, nz,
                        dx, dy, dz,
                        max_dist
                    )
                    
                    if hit == 0:
                        visible_omega += d_omega * cos_angle
            
            if total_omega > 1e-10:
                svf_grid[i, j] = visible_omega / total_omega
            else:
                svf_grid[i, j] = 1.0
        else:
            svf_grid[i, j] = 0.0  # Inside terrain
