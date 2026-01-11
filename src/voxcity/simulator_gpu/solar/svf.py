"""
Sky View Factor (SVF) calculation for palm-solar.

Computes the fraction of sky hemisphere visible from each surface element.
Uses GPU-accelerated ray tracing to sample the hemisphere.

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
from .raytracing import ray_voxel_first_hit, ray_canopy_absorption, sample_hemisphere_direction, hemisphere_solid_angle


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
        
        Args:
            surf_pos: Surface positions (n_surf, 3)
            surf_dir: Surface directions (n_surf,)
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
            
            # Get surface normal and azimuth of normal (for vertical surfaces)
            normal = Vector3(0.0, 0.0, 0.0)
            normal_azim = 0.0  # Azimuth angle of surface normal (for vertical surfaces)
            
            if direction == 0:  # Up
                normal = Vector3(0.0, 0.0, 1.0)
            elif direction == 1:  # Down
                normal = Vector3(0.0, 0.0, -1.0)
            elif direction == 2:  # North (normal points +y)
                normal = Vector3(0.0, 1.0, 0.0)
                normal_azim = 0.0  # North is azimuth 0
            elif direction == 3:  # South (normal points -y)
                normal = Vector3(0.0, -1.0, 0.0)
                normal_azim = PI  # South is azimuth π
            elif direction == 4:  # East (normal points +x)
                normal = Vector3(1.0, 0.0, 0.0)
                normal_azim = PI / 2.0  # East is azimuth π/2
            elif direction == 5:  # West (normal points -x)
                normal = Vector3(-1.0, 0.0, 0.0)
                normal_azim = 3.0 * PI / 2.0  # West is azimuth 3π/2
            
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
                    if direction == 0:  # Upward surface
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
                        
                        # Relative azimuth (measured from surface normal)
                        # PALM shifts by -π/2 so that az=0 is at surface normal
                        az1_rel = azim_low - normal_azim - PI / 2.0
                        az2_rel = azim_high - normal_azim - PI / 2.0
                        
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
                    
                    # Trace ray
                    hit, _, _, _, _ = ray_voxel_first_hit(
                        pos, ray_dir,
                        is_solid,
                        self.nx, self.ny, self.nz,
                        self.dx, self.dy, self.dz,
                        self.max_dist
                    )
                    
                    if hit == 0:
                        visible_vf += vf_frac
            
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
        
        Args:
            surf_pos: Surface positions
            surf_dir: Surface directions
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
            
            # Get surface normal and azimuth of normal (for vertical surfaces)
            normal = Vector3(0.0, 0.0, 0.0)
            normal_azim = 0.0
            
            if direction == 0:  # Up
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
                    if direction == 0 or direction == 1:  # Upward or Downward surface
                        # PALM: vffrac_up
                        vf_frac = (ti.cos(2.0 * elev_low) - ti.cos(2.0 * elev_high)) / (2.0 * n_azim_f)
                    else:
                        # Vertical surfaces: use PALM's vffrac_vert formula
                        azim_low = ti.cast(i_azim, ti.f32) * d_azim
                        azim_high = ti.cast(i_azim + 1, ti.f32) * d_azim
                        
                        # Relative azimuth (measured from surface normal)
                        az1_rel = azim_low - normal_azim - PI / 2.0
                        az2_rel = azim_high - normal_azim - PI / 2.0
                        
                        # Elevation terms
                        elev_terms = (elev_high - elev_low 
                                     + ti.sin(elev_low) * ti.cos(elev_low) 
                                     - ti.sin(elev_high) * ti.cos(elev_high))
                        
                        # vffrac_vert = (sin(az2) - sin(az1)) * elev_terms / (2*π)
                        vf_frac = (ti.sin(az2_rel) - ti.sin(az1_rel)) * elev_terms / TWO_PI
                        
                        if vf_frac < 0.0:
                            vf_frac = 0.0
                    
                    total_omega += vf_frac
                    
                    # Trace with canopy absorption
                    trans, _ = ray_canopy_absorption(
                        pos, ray_dir,
                        lad, is_solid,
                        self.nx, self.ny, self.nz,
                        self.dx, self.dy, self.dz,
                        self.max_dist,
                        ext_coef
                    )
                    
                    # SVF with canopy considers transparency (PALM's skyvft)
                    visible_omega += vf_frac * trans
                    
                    # SVF urban (only solid obstacles, PALM's skyvf)
                    hit, _, _, _, _ = ray_voxel_first_hit(
                        pos, ray_dir,
                        is_solid,
                        self.nx, self.ny, self.nz,
                        self.dx, self.dy, self.dz,
                        self.max_dist
                    )
                    if hit == 0:
                        visible_omega_urban += vf_frac
            
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
