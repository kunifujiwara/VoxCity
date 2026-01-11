"""
Canopy Sink Factor (CSF) calculation for palm-solar.

Computes how much radiation is absorbed by plant canopy (LAD) before
reaching surfaces. Based on PALM's RTM methodology using Beer-Lambert law.

PALM CSF Structure (from radiation_model_mod.f90 lines ~920-930):
- TYPE t_csf contains:
  - isurfs: Index of source face (-1 for sky, >= 0 for surface sources)
  - rcvf: Canopy view factor for faces / canopy sink factor for sky
  
PALM Canopy Absorption (radiation_model_mod.f90 lines ~9200-9250):
- Diffuse from sky: pcbinswdif = csf * rad_sw_in_diff
- Direct from sun: pcbinswdir = rad_sw_in_dir * pc_box_area * pc_abs_frac * dsitransc
- From reflections: pcbinsw += csf * surfoutsl(isurfsrc) * asrc * grid_volume_inverse

palm_solar implements equivalent physics with GPU-parallel raytracing.
"""

import taichi as ti
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass

from .core import Vector3, Point3, EXT_COEF, PI, TWO_PI
from .raytracing import ray_aabb_intersect


# Prototype LAD for computing effective absorption coefficient (PALM default)
PROTOTYPE_LAD = 1.0

# Source type constants (matching PALM's isurfs convention)
CSF_SOURCE_SKY = -1  # Sky source (diffuse sky radiation)


@ti.func
def box_absorb_mc(
    boxsize_z: ti.f32, boxsize_y: ti.f32, boxsize_x: ti.f32,
    uvec_z: ti.f32, uvec_y: ti.f32, uvec_x: ti.f32,
    dens: ti.f32, ext_coef: ti.f32, resol: ti.i32
) -> ti.types.vector(2, ti.f32):
    """
    PALM's box_absorb: Monte Carlo integration for box absorption.
    
    Computes effective cross-sectional area and transmissivity by
    tracing multiple rays through a box at the given angle.
    
    Args:
        boxsize_z, boxsize_y, boxsize_x: Box dimensions
        uvec_z, uvec_y, uvec_x: Unit vector of incoming flux (must have uvec_z > 0)
        dens: Box density (LAD)
        ext_coef: Extinction coefficient
        resol: Number of rays in x and y directions
        
    Returns:
        Vector of (area, transmissivity)
    """
    # Compute shift of ray footprint due to angle
    xshift = uvec_x / uvec_z * boxsize_z
    yshift = uvec_y / uvec_z * boxsize_z
    
    xmin = ti.min(0.0, -xshift)
    xmax = boxsize_x + ti.max(0.0, -xshift)
    ymin = ti.min(0.0, -yshift)
    ymax = boxsize_y + ti.max(0.0, -yshift)
    
    transp = 0.0
    
    # Monte Carlo integration over ray entry points
    for i in range(resol):
        xorig = xmin + (xmax - xmin) * (i + 0.5) / resol
        for j in range(resol):
            yorig = ymin + (ymax - ymin) * (j + 0.5) / resol
            
            # Find ray path through box (entry and exit t values)
            dz1 = 0.0
            dz2 = boxsize_z / uvec_z
            
            # Y boundaries
            dy1 = -1e30
            dy2 = 1e30
            if uvec_y > 1e-10:
                dy1 = -yorig / uvec_y
                dy2 = (boxsize_y - yorig) / uvec_y
            elif uvec_y < -1e-10:
                dy1 = (boxsize_y - yorig) / uvec_y
                dy2 = -yorig / uvec_y
            
            # X boundaries
            dx1 = -1e30
            dx2 = 1e30
            if uvec_x > 1e-10:
                dx1 = -xorig / uvec_x
                dx2 = (boxsize_x - xorig) / uvec_x
            elif uvec_x < -1e-10:
                dx1 = (boxsize_x - xorig) / uvec_x
                dx2 = -xorig / uvec_x
            
            # Path length through box
            t_enter = ti.max(dz1, ti.max(dy1, dx1))
            t_exit = ti.min(dz2, ti.min(dy2, dx2))
            crdist = ti.max(0.0, t_exit - t_enter)
            
            # Transmissivity for this ray
            transp += ti.exp(-ext_coef * dens * crdist)
    
    # Average transmissivity
    transp = transp / (resol * resol)
    
    # Effective area (footprint including slant)
    area = (boxsize_x + ti.abs(xshift)) * (boxsize_y + ti.abs(yshift))
    
    return ti.Vector([area, transp])


@ti.data_oriented
class CSFCalculator:
    """
    GPU-accelerated Canopy Sink Factor calculator.
    
    CSF represents the fraction of radiation absorbed by each vegetation
    cell along ray paths from surfaces to sky/sun.
    
    Following PALM's methodology:
    - CSF entries have a source index (isurfs): -1 for sky, >= 0 for surfaces
    - During reflection iterations, canopy absorption is accumulated using CSF
    - pcbinsw += csf * surfoutsl(isurfsrc) * asrc * grid_volume_inverse
    """
    
    def __init__(self, domain, n_azimuth: int = 80, n_elevation: int = 40,
                 max_surfaces: int = 10000):
        """
        Initialize CSF calculator.
        
        Args:
            domain: Domain object with grid geometry and LAD
            n_azimuth: Number of azimuthal divisions
            n_elevation: Number of elevation divisions
            max_surfaces: Maximum number of surfaces (for CSF from reflections)
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
        self.ext_coef = EXT_COEF
        self.max_surfaces = max_surfaces
        
        # Maximum ray distance
        self.max_dist = math.sqrt(
            (self.nx * self.dx)**2 + 
            (self.ny * self.dy)**2 + 
            (self.nz * self.dz)**2
        )
        
        # CSF storage: fraction of radiation absorbed per canopy cell
        # Indexed by (canopy_i, canopy_j, canopy_k)
        # This stores total absorbed power (W) - divide by grid volume for W/m³
        self.csf = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # CSF from sky only (isurfs = -1 in PALM terminology)
        # Stores view factor × absorption fraction from sky to each canopy cell
        self.csf_sky = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # CSF from surfaces (indexed by surface and canopy cell)
        # For memory efficiency, we use a dense 4D array for moderate domain sizes
        # csf_surf[surf_idx, i, j, k] = view factor × absorption from surface surf_idx
        # Only allocated if needed (for reflection-step canopy absorption)
        self._csf_surf_allocated = False
        self._max_csf_surfaces = min(max_surfaces, 5000)  # Limit memory usage
        
        # Accumulated LAD path for diagnostics
        self.lad_path = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # PALM-style: dsitransc - direct solar transmissivity for each canopy box
        # Transmissivity from sky to each canopy cell along sun direction
        self.dsitransc = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # Pre-computed box absorption parameters (updated per sun position)
        self.pc_box_area = ti.field(dtype=ti.f32, shape=())  # Effective cross-sectional area
        self.pc_abs_eff = ti.field(dtype=ti.f32, shape=())   # Effective absorption coefficient
        
        # Grid volume (m³) for normalization
        self.grid_volume = self.dx * self.dy * self.dz
        self.grid_volume_inverse = 1.0 / self.grid_volume
    
    def allocate_surface_csf(self, n_surfaces: int):
        """
        Allocate surface-to-canopy CSF storage for reflection calculations.
        
        This is called when canopy absorption during reflections is needed.
        
        Args:
            n_surfaces: Number of surfaces in the domain
        """
        if self._csf_surf_allocated:
            return
        
        n_to_alloc = min(n_surfaces, self._max_csf_surfaces)
        # csf_surf[surf_idx, i, j, k] stores CSF from surface surf_idx to canopy (i,j,k)
        self.csf_surf = ti.field(dtype=ti.f32, 
                                 shape=(n_to_alloc, self.nx, self.ny, self.nz))
        self._n_csf_surfaces = n_to_alloc
        self._csf_surf_allocated = True
        print(f"Allocated surface-indexed CSF storage for {n_to_alloc} surfaces")
    
    @ti.kernel
    def reset_csf(self):
        """Reset CSF fields to zero."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.csf[i, j, k] = 0.0
            self.csf_sky[i, j, k] = 0.0
            self.lad_path[i, j, k] = 0.0
            self.dsitransc[i, j, k] = 0.0
    
    @ti.kernel
    def _reset_csf_surf(self, n_surfaces: ti.i32):
        """Reset surface-indexed CSF fields to zero."""
        for s, i, j, k in ti.ndrange(n_surfaces, self.nx, self.ny, self.nz):
            self.csf_surf[s, i, j, k] = 0.0
    
    def reset_surface_csf(self, n_surfaces: int):
        """Reset surface-indexed CSF storage."""
        if self._csf_surf_allocated:
            self._reset_csf_surf(min(n_surfaces, self._n_csf_surfaces))
    
    @ti.kernel
    def _compute_box_params(
        self,
        sun_dir: ti.types.vector(3, ti.f32),
        prototype_lad: ti.f32,
        mc_resolution: ti.i32
    ):
        """
        Compute effective box absorption parameters (PALM's box_absorb).
        
        This precomputes pc_box_area and pc_abs_eff for the current sun position.
        These are used for all canopy boxes.
        
        PALM uses CSHIFT to reorder dimensions so the largest sun direction
        component is first. Then adjusts area by the ratio of the shifted
        first component to the original first component (z).
        
        Args:
            sun_dir: Sun direction unit vector (pointing toward sun)
            prototype_lad: Reference LAD for computing effective coefficient
            mc_resolution: Monte Carlo resolution (rays per dimension)
        """
        # PALM's sunorig is (z, y, x) - we need to handle dimension reordering
        # sunorig(1) = z component, sunorig(2) = y component, sunorig(3) = x component
        abs_z = ti.abs(sun_dir[2])  # sunorig(1)
        abs_y = ti.abs(sun_dir[1])  # sunorig(2)
        abs_x = ti.abs(sun_dir[0])  # sunorig(3)
        
        # Find dominant direction (PALM: MAXLOC(ABS(sunorig), 1) - 1)
        # dimshift = 0: z dominant, dimshift = 1: y dominant, dimshift = 2: x dominant
        dimshift = 0
        max_component = abs_z
        if abs_y > max_component:
            dimshift = 1
            max_component = abs_y
        if abs_x > max_component:
            dimshift = 2
            max_component = abs_x
        
        # Reorder box dimensions and direction vector (CSHIFT)
        # Original order: (dz, dy, dx), (abs_z, abs_y, abs_x)
        boxsize_0 = 0.0
        boxsize_1 = 0.0
        boxsize_2 = 0.0
        uvec_0 = 0.0
        uvec_1 = 0.0
        uvec_2 = 0.0
        
        if dimshift == 0:
            # z dominant: no shift
            boxsize_0, boxsize_1, boxsize_2 = self.dz, self.dy, self.dx
            uvec_0, uvec_1, uvec_2 = abs_z, abs_y, abs_x
        elif dimshift == 1:
            # y dominant: shift by 1 -> (dy, dx, dz), (abs_y, abs_x, abs_z)
            boxsize_0, boxsize_1, boxsize_2 = self.dy, self.dx, self.dz
            uvec_0, uvec_1, uvec_2 = abs_y, abs_x, abs_z
        else:
            # x dominant: shift by 2 -> (dx, dz, dy), (abs_x, abs_z, abs_y)
            boxsize_0, boxsize_1, boxsize_2 = self.dx, self.dz, self.dy
            uvec_0, uvec_1, uvec_2 = abs_x, abs_z, abs_y
        
        if uvec_0 > 1e-6:
            result = box_absorb_mc(
                boxsize_0, boxsize_1, boxsize_2,
                uvec_0, uvec_1, uvec_2,
                prototype_lad, self.ext_coef, mc_resolution
            )
            
            area = result[0]
            transp = result[1]
            
            # Adjust area for dimension shift (PALM: pc_box_area * sunorig(dimshift+1) / sunorig(1))
            # dimshift+1 index in shifted array is uvec_0, original sunorig(1) is abs_z
            if abs_z > 1e-10:
                area = area * uvec_0 / abs_z
            
            self.pc_box_area[None] = area
            
            # Compute effective absorption coefficient
            # pc_abs_eff = LOG(1 - pc_abs_frac) / prototype_lad = LOG(transp) / prototype_lad
            abs_frac = 1.0 - transp
            if abs_frac > 1e-10 and abs_frac < 1.0 - 1e-10:
                self.pc_abs_eff[None] = ti.log(1.0 - abs_frac) / prototype_lad
            else:
                # Fallback for edge cases (very transparent or very opaque)
                self.pc_abs_eff[None] = -self.ext_coef * boxsize_0 / uvec_0
    
    @ti.kernel
    def _compute_dsitransc(
        self,
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template(),
        lad: ti.template()
    ):
        """
        Compute direct solar transmissivity to each canopy box (PALM's dsitransc).
        
        Traces rays from each canopy box toward the sun to compute how much
        direct radiation reaches that box.
        
        Args:
            sun_dir: Sun direction unit vector (pointing toward sun)
            is_solid: 3D solid field
            lad: 3D Leaf Area Density field
        """
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            # Only compute for canopy cells
            if lad[i, j, k] > 0.0:
                # Start from center of this canopy box
                pos = Vector3(
                    (i + 0.5) * self.dx,
                    (j + 0.5) * self.dy,
                    (k + 0.5) * self.dz
                )
                
                # Trace toward sun, accumulating opacity
                domain_min = Vector3(0.0, 0.0, 0.0)
                domain_max = Vector3(self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)
                
                cumulative_opacity = 0.0
                
                # Start slightly offset in sun direction
                t = 0.01
                current_pos = pos + sun_dir * t
                
                ci = ti.cast(ti.floor(current_pos[0] / self.dx), ti.i32)
                cj = ti.cast(ti.floor(current_pos[1] / self.dy), ti.i32)
                ck = ti.cast(ti.floor(current_pos[2] / self.dz), ti.i32)
                
                ci = ti.max(0, ti.min(self.nx - 1, ci))
                cj = ti.max(0, ti.min(self.ny - 1, cj))
                ck = ti.max(0, ti.min(self.nz - 1, ck))
                
                step_x = 1 if sun_dir[0] >= 0 else -1
                step_y = 1 if sun_dir[1] >= 0 else -1
                step_z = 1 if sun_dir[2] >= 0 else -1
                
                t_max_x = 1e30
                t_max_y = 1e30
                t_max_z = 1e30
                t_delta_x = 1e30
                t_delta_y = 1e30
                t_delta_z = 1e30
                
                if ti.abs(sun_dir[0]) > 1e-10:
                    if step_x > 0:
                        t_max_x = ((ci + 1) * self.dx - current_pos[0]) / sun_dir[0] + t
                    else:
                        t_max_x = (ci * self.dx - current_pos[0]) / sun_dir[0] + t
                    t_delta_x = ti.abs(self.dx / sun_dir[0])
                
                if ti.abs(sun_dir[1]) > 1e-10:
                    if step_y > 0:
                        t_max_y = ((cj + 1) * self.dy - current_pos[1]) / sun_dir[1] + t
                    else:
                        t_max_y = (cj * self.dy - current_pos[1]) / sun_dir[1] + t
                    t_delta_y = ti.abs(self.dy / sun_dir[1])
                
                if ti.abs(sun_dir[2]) > 1e-10:
                    if step_z > 0:
                        t_max_z = ((ck + 1) * self.dz - current_pos[2]) / sun_dir[2] + t
                    else:
                        t_max_z = (ck * self.dz - current_pos[2]) / sun_dir[2] + t
                    t_delta_z = ti.abs(self.dz / sun_dir[2])
                
                t_prev = t
                max_steps = self.nx + self.ny + self.nz
                hit_solid = 0
                
                # GPU-optimized loop with done flag pattern
                done = 0
                for _ in range(max_steps):
                    if done == 0:
                        # Check bounds
                        if ci < 0 or ci >= self.nx or cj < 0 or cj >= self.ny or ck < 0 or ck >= self.nz:
                            done = 1
                        # Stop if hit solid
                        elif is_solid[ci, cj, ck] == 1:
                            hit_solid = 1
                            done = 1
                        else:
                            # Get path length through this cell using branchless min
                            t_next = ti.min(t_max_x, ti.min(t_max_y, t_max_z))
                            path_len = t_next - t_prev
                            
                            # Accumulate opacity from LAD (skip the starting cell)
                            if not (ci == i and cj == j and ck == k):
                                cell_lad = lad[ci, cj, ck]
                                if cell_lad > 0.0:
                                    cumulative_opacity += self.ext_coef * cell_lad * path_len
                            
                            t_prev = t_next
                            
                            # Step to next cell
                            if t_max_x < t_max_y and t_max_x < t_max_z:
                                ci += step_x
                                t_max_x += t_delta_x
                            elif t_max_y < t_max_z:
                                cj += step_y
                                t_max_y += t_delta_y
                            else:
                                ck += step_z
                                t_max_z += t_delta_z
                
                # Store transmissivity (0 if hit solid)
                if hit_solid == 1:
                    self.dsitransc[i, j, k] = 0.0
                else:
                    self.dsitransc[i, j, k] = ti.exp(-cumulative_opacity)
    
    @ti.kernel
    def _compute_pcbinswdir_palm(
        self,
        lad: ti.template(),
        incoming_flux: ti.f32,
        grid_volume: ti.f32
    ):
        """
        Compute canopy absorption using PALM's exact formula.
        
        pcbinswdir = rad_sw_in_dir * pc_box_area * pc_abs_frac * dsitransc / grid_volume
        
        Args:
            lad: 3D Leaf Area Density field
            incoming_flux: Incoming direct solar flux (W/m²)
            grid_volume: Volume of one grid cell (m³)
        """
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            cell_lad = lad[i, j, k]
            if cell_lad > 0.0:
                # PALM's formula: pc_abs_frac = 1 - exp(pc_abs_eff * lad)
                pc_abs_frac = 1.0 - ti.exp(self.pc_abs_eff[None] * cell_lad)
                
                # Absorbed power = flux * area * absorption_fraction * transmissivity_to_box
                absorbed_power = (incoming_flux * self.pc_box_area[None] * 
                                 pc_abs_frac * self.dsitransc[i, j, k])
                
                # Convert to W/m³
                self.csf[i, j, k] = absorbed_power / grid_volume
            else:
                self.csf[i, j, k] = 0.0
    
    def compute_canopy_absorption_direct_palm(
        self,
        sun_dir,
        is_solid,
        lad,
        incoming_flux: float,
        prototype_lad: float = PROTOTYPE_LAD,
        mc_resolution: int = 60
    ):
        """
        Compute direct solar canopy absorption using PALM's method.
        
        This is the main entry point that follows PALM's approach:
        1. Compute box absorption parameters (pc_box_area, pc_abs_eff) 
        2. Compute dsitransc (transmissivity to each canopy box)
        3. Compute absorption per box using PALM's formula
        
        Args:
            sun_dir: Sun direction vector (ti.Vector or list)
            is_solid: 3D solid field
            lad: 3D LAD field
            incoming_flux: Direct solar flux (W/m²)
            prototype_lad: Reference LAD for effective coefficient
            mc_resolution: Monte Carlo resolution for box_absorb
        """
        # Convert sun_dir to ti.Vector if needed
        if hasattr(sun_dir, '__len__') and not isinstance(sun_dir, ti.lang.matrix.VectorType):
            sun_dir_vec = ti.Vector([float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2])])
        else:
            sun_dir_vec = sun_dir
        
        grid_volume = self.dx * self.dy * self.dz
        
        # Step 1: Compute box parameters
        self._compute_box_params(sun_dir_vec, prototype_lad, mc_resolution)
        
        # Step 2: Compute transmissivity to each canopy box
        self._compute_dsitransc(sun_dir_vec, is_solid, lad)
        
        # Step 3: Compute absorption using PALM's formula
        self._compute_pcbinswdir_palm(lad, incoming_flux, grid_volume)
    
    @ti.kernel
    def _compute_pcbinswdif_palm(
        self,
        lad: ti.template(),
        diffuse_flux: ti.f32,
        pcbinswdif: ti.template()
    ):
        """
        Compute diffuse canopy absorption using PALM's formula.
        
        The csf_sky field contains the hemisphere-integrated absorption factor:
            csf_sky = integral(trans_above * abs_in_cell * cos_elev * d_omega / pi)
        
        This is dimensionless and represents the fraction of isotropic sky
        radiance that gets absorbed by this cell.
        
        To convert to absorbed power per unit volume:
            absorbed_power = diffuse_flux * horizontal_area * csf_sky
            pcbinswdif = absorbed_power / grid_volume
                       = diffuse_flux * dx * dy * csf_sky / (dx * dy * dz)
                       = diffuse_flux * csf_sky / dz
        
        Args:
            lad: 3D Leaf Area Density field
            diffuse_flux: Diffuse sky flux (W/m²)
            pcbinswdif: Output array for diffuse absorbed (W/m³)
        """
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            if lad[i, j, k] > 0.0:
                # Power absorbed = diffuse_flux * horizontal_area * csf_sky
                # Rate (W/m³) = Power / grid_volume = diffuse_flux * csf_sky / dz
                pcbinswdif[i, j, k] = self.csf_sky[i, j, k] * diffuse_flux / self.dz
            else:
                pcbinswdif[i, j, k] = 0.0
    
    def compute_canopy_absorption_diffuse_palm(
        self,
        is_solid,
        lad,
        diffuse_flux: float,
        pcbinswdif,
        n_azimuth: int = None,
        n_elevation: int = None
    ):
        """
        Compute diffuse sky canopy absorption using PALM's method.
        
        This computes:
        1. CSF from sky (if not already computed)
        2. Diffuse absorption using pcbinswdif = csf_sky * diffuse_flux * grid_volume_inverse
        
        Args:
            is_solid: 3D solid field
            lad: 3D LAD field
            diffuse_flux: Diffuse sky flux (W/m²)
            pcbinswdif: Output array for diffuse absorbed (W/m³)
            n_azimuth: Number of azimuthal divisions for sky CSF
            n_elevation: Number of elevation divisions for sky CSF
        """
        n_azim = n_azimuth if n_azimuth is not None else self.n_azimuth
        n_elev = n_elevation if n_elevation is not None else self.n_elevation
        
        # Compute CSF from sky (this is isurfs = -1 in PALM)
        self.compute_csf_sky(is_solid, lad, n_azim, n_elev)
        
        # Compute diffuse absorption
        self._compute_pcbinswdif_palm(lad, diffuse_flux, pcbinswdif)

    @ti.kernel
    def compute_csf_sky(
        self,
        is_solid: ti.template(),
        lad: ti.template(),
        n_azim: ti.i32,
        n_elev: ti.i32
    ):
        """
        Compute canopy sink factors from sky (isurfs = -1 in PALM terminology).
        
        For each canopy cell, traces rays toward the sky hemisphere and
        computes the fraction of diffuse sky radiation that would be
        absorbed by this cell.
        
        The result is stored in csf_sky as a view factor × absorption fraction.
        To get absorbed power: csf_sky[i,j,k] * diffuse_flux * horizontal_area / grid_volume
        
        Args:
            is_solid: 3D solid field
            lad: 3D LAD field
            n_azim: Number of azimuthal divisions
            n_elev: Number of elevation divisions
        """
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            cell_lad = lad[i, j, k]
            if cell_lad > 0.0:
                # Cell center position
                pos = Vector3(
                    (i + 0.5) * self.dx,
                    (j + 0.5) * self.dy,
                    (k + 0.5) * self.dz
                )
                
                domain_min = Vector3(0.0, 0.0, 0.0)
                domain_max = Vector3(self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)
                
                total_sky_factor = 0.0
                
                # Trace rays to sky hemisphere
                for i_azim, i_elev in ti.ndrange(n_azim, n_elev):
                    # Compute direction toward sky
                    elev_angle = (i_elev + 0.5) * (PI / 2.0) / n_elev
                    azim_angle = (i_azim + 0.5) * TWO_PI / n_azim
                    
                    sin_elev = ti.sin(elev_angle)
                    cos_elev = ti.cos(elev_angle)
                    
                    ray_dir = Vector3(
                        sin_elev * ti.sin(azim_angle),
                        sin_elev * ti.cos(azim_angle),
                        cos_elev  # Upward
                    )
                    
                    # Solid angle weight
                    elev_low = i_elev * (PI / 2.0) / n_elev
                    elev_high = (i_elev + 1) * (PI / 2.0) / n_elev
                    d_omega = (TWO_PI / n_azim) * (ti.cos(elev_low) - ti.cos(elev_high))
                    
                    # Trace ray from cell toward sky (opposite direction for finding opacity)
                    cumulative_opacity_above = 0.0
                    blocked = 0
                    
                    # Start from cell center and trace upward
                    t = 0.01
                    current_pos = pos + ray_dir * t
                    
                    ci = ti.cast(ti.floor(current_pos[0] / self.dx), ti.i32)
                    cj = ti.cast(ti.floor(current_pos[1] / self.dy), ti.i32)
                    ck = ti.cast(ti.floor(current_pos[2] / self.dz), ti.i32)
                    
                    step_x = 1 if ray_dir[0] >= 0 else -1
                    step_y = 1 if ray_dir[1] >= 0 else -1
                    step_z = 1 if ray_dir[2] >= 0 else -1
                    
                    t_max_x, t_max_y, t_max_z = 1e30, 1e30, 1e30
                    t_delta_x, t_delta_y, t_delta_z = 1e30, 1e30, 1e30
                    
                    if ti.abs(ray_dir[0]) > 1e-10:
                        if step_x > 0:
                            t_max_x = ((ci + 1) * self.dx - current_pos[0]) / ray_dir[0] + t
                        else:
                            t_max_x = (ci * self.dx - current_pos[0]) / ray_dir[0] + t
                        t_delta_x = ti.abs(self.dx / ray_dir[0])
                    
                    if ti.abs(ray_dir[1]) > 1e-10:
                        if step_y > 0:
                            t_max_y = ((cj + 1) * self.dy - current_pos[1]) / ray_dir[1] + t
                        else:
                            t_max_y = (cj * self.dy - current_pos[1]) / ray_dir[1] + t
                        t_delta_y = ti.abs(self.dy / ray_dir[1])
                    
                    if ti.abs(ray_dir[2]) > 1e-10:
                        if step_z > 0:
                            t_max_z = ((ck + 1) * self.dz - current_pos[2]) / ray_dir[2] + t
                        else:
                            t_max_z = (ck * self.dz - current_pos[2]) / ray_dir[2] + t
                        t_delta_z = ti.abs(self.dz / ray_dir[2])
                    
                    t_prev = t
                    max_steps = self.nx + self.ny + self.nz
                    
                    for _ in range(max_steps):
                        if ci < 0 or ci >= self.nx or cj < 0 or cj >= self.ny:
                            break
                        if ck >= self.nz:  # Reached top of domain (sky)
                            break
                        if ck < 0:  # Went below domain
                            blocked = 1
                            break
                        
                        if is_solid[ci, cj, ck] == 1:
                            blocked = 1
                            break
                        
                        t_next = ti.min(t_max_x, ti.min(t_max_y, t_max_z))
                        path_len = t_next - t_prev
                        
                        # Accumulate opacity from LAD above this cell
                        if not (ci == i and cj == j and ck == k):
                            above_lad = lad[ci, cj, ck]
                            if above_lad > 0.0:
                                cumulative_opacity_above += self.ext_coef * above_lad * path_len
                        
                        t_prev = t_next
                        
                        if t_max_x < t_max_y and t_max_x < t_max_z:
                            ci += step_x
                            t_max_x += t_delta_x
                        elif t_max_y < t_max_z:
                            cj += step_y
                            t_max_y += t_delta_y
                        else:
                            ck += step_z
                            t_max_z += t_delta_z
                    
                    if blocked == 0:
                        # Transmissivity from sky to this cell
                        trans_to_cell = ti.exp(-cumulative_opacity_above)
                        
                        # Path length through this cell (approximate)
                        path_in_cell = self.dz / cos_elev if cos_elev > 0.1 else self.dz * 10.0
                        
                        # Absorption in this cell
                        abs_in_cell = 1.0 - ti.exp(-self.ext_coef * cell_lad * path_in_cell)
                        
                        # Contribution from this sky direction
                        # Weight by solid angle and cosine (Lambertian sky)
                        total_sky_factor += trans_to_cell * abs_in_cell * d_omega * cos_elev / PI
                
                self.csf_sky[i, j, k] = total_sky_factor
            else:
                self.csf_sky[i, j, k] = 0.0

    @ti.kernel
    def compute_csf_direct(
        self,
        surf_pos: ti.template(),
        surf_area: ti.template(),
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template(),
        lad: ti.template(),
        n_surf: ti.i32,
        incoming_flux: ti.f32
    ):
        """
        Compute CSF for direct solar radiation.
        
        Traces rays from surfaces toward sun and accumulates absorption
        in each canopy cell along the path.
        
        Args:
            surf_pos: Surface positions
            surf_area: Surface areas
            sun_dir: Sun direction unit vector
            is_solid: 3D solid field
            lad: 3D Leaf Area Density field
            n_surf: Number of surfaces
            incoming_flux: Incoming direct solar flux (W/m²)
        """
        for surf_i in range(n_surf):
            pos = Vector3(surf_pos[surf_i][0], surf_pos[surf_i][1], surf_pos[surf_i][2])
            area = surf_area[surf_i]
            
            # Total power from this surface toward sun
            power = incoming_flux * area
            
            # Find entry into domain
            domain_min = Vector3(0.0, 0.0, 0.0)
            domain_max = Vector3(self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)
            
            in_domain, t_enter, t_exit = ray_aabb_intersect(
                pos, sun_dir, domain_min, domain_max, 0.0, self.max_dist
            )
            
            if in_domain == 1:
                t = 1e-5  # Start slightly above surface
                cumulative_opacity = 0.0
                
                # 3D-DDA traversal
                step_x = 1 if sun_dir[0] >= 0 else -1
                step_y = 1 if sun_dir[1] >= 0 else -1
                step_z = 1 if sun_dir[2] >= 0 else -1
                
                current_pos = pos + sun_dir * t
                
                ix = ti.cast(ti.floor(current_pos[0] / self.dx), ti.i32)
                iy = ti.cast(ti.floor(current_pos[1] / self.dy), ti.i32)
                iz = ti.cast(ti.floor(current_pos[2] / self.dz), ti.i32)
                
                ix = ti.max(0, ti.min(self.nx - 1, ix))
                iy = ti.max(0, ti.min(self.ny - 1, iy))
                iz = ti.max(0, ti.min(self.nz - 1, iz))
                
                # Initialize t_max values (must be before branching for Taichi)
                t_max_x = 1e30
                t_max_y = 1e30
                t_max_z = 1e30
                t_delta_x = 1e30
                t_delta_y = 1e30
                t_delta_z = 1e30
                
                if ti.abs(sun_dir[0]) > 1e-10:
                    if step_x > 0:
                        t_max_x = ((ix + 1) * self.dx - current_pos[0]) / sun_dir[0] + t
                    else:
                        t_max_x = (ix * self.dx - current_pos[0]) / sun_dir[0] + t
                    t_delta_x = ti.abs(self.dx / sun_dir[0])
                
                if ti.abs(sun_dir[1]) > 1e-10:
                    if step_y > 0:
                        t_max_y = ((iy + 1) * self.dy - current_pos[1]) / sun_dir[1] + t
                    else:
                        t_max_y = (iy * self.dy - current_pos[1]) / sun_dir[1] + t
                    t_delta_y = ti.abs(self.dy / sun_dir[1])
                
                if ti.abs(sun_dir[2]) > 1e-10:
                    if step_z > 0:
                        t_max_z = ((iz + 1) * self.dz - current_pos[2]) / sun_dir[2] + t
                    else:
                        t_max_z = (iz * self.dz - current_pos[2]) / sun_dir[2] + t
                    t_delta_z = ti.abs(self.dz / sun_dir[2])
                
                t_prev = t
                max_steps = self.nx + self.ny + self.nz
                
                for _ in range(max_steps):
                    if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny or iz < 0 or iz >= self.nz:
                        break
                    if t > t_exit:
                        break
                    
                    # Stop at solid obstacle
                    if is_solid[ix, iy, iz] == 1:
                        break
                    
                    # Find next t
                    t_next = ti.min(t_max_x, ti.min(t_max_y, t_max_z))
                    path_len = t_next - t_prev
                    
                    # Process canopy cell
                    cell_lad = lad[ix, iy, iz]
                    if cell_lad > 0.0:
                        # Transmissivity to this point
                        trans_before = ti.exp(-cumulative_opacity)
                        
                        # Opacity through this cell
                        cell_opacity = self.ext_coef * cell_lad * path_len
                        
                        # Transmissivity after this cell
                        trans_after = ti.exp(-(cumulative_opacity + cell_opacity))
                        
                        # Absorbed fraction in this cell
                        absorbed_frac = trans_before - trans_after
                        
                        # Add to CSF (atomic add for thread safety)
                        ti.atomic_add(self.csf[ix, iy, iz], absorbed_frac * power)
                        ti.atomic_add(self.lad_path[ix, iy, iz], cell_lad * path_len)
                        
                        # Update cumulative opacity
                        cumulative_opacity += cell_opacity
                    
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
    
    @ti.kernel
    def compute_csf_diffuse_hemisphere(
        self,
        surf_pos: ti.template(),
        surf_dir: ti.template(),
        surf_area: ti.template(),
        is_solid: ti.template(),
        lad: ti.template(),
        n_surf: ti.i32,
        diffuse_flux: ti.f32,
        n_azim: ti.i32,
        n_elev: ti.i32
    ):
        """
        Compute CSF for diffuse sky radiation.
        
        Traces rays from surfaces to multiple sky directions.
        """
        for surf_i in range(n_surf):
            pos = Vector3(surf_pos[surf_i][0], surf_pos[surf_i][1], surf_pos[surf_i][2])
            direction = surf_dir[surf_i]
            area = surf_area[surf_i]
            
            # Get surface normal
            normal = Vector3(0.0, 0.0, 0.0)
            if direction == 0:
                normal = Vector3(0.0, 0.0, 1.0)
            elif direction == 1:
                normal = Vector3(0.0, 0.0, -1.0)
            elif direction == 2:
                normal = Vector3(0.0, 1.0, 0.0)
            elif direction == 3:
                normal = Vector3(0.0, -1.0, 0.0)
            elif direction == 4:
                normal = Vector3(1.0, 0.0, 0.0)
            elif direction == 5:
                normal = Vector3(-1.0, 0.0, 0.0)
            
            domain_min = Vector3(0.0, 0.0, 0.0)
            domain_max = Vector3(self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)
            
            # Loop over hemisphere directions
            for i_azim, i_elev in ti.ndrange(n_azim, n_elev):
                # Compute direction
                elev_angle = (i_elev + 0.5) * (PI / 2.0) / n_elev
                azim_angle = (i_azim + 0.5) * TWO_PI / n_azim
                
                sin_elev = ti.sin(elev_angle)
                cos_elev = ti.cos(elev_angle)
                
                ray_dir = Vector3(
                    sin_elev * ti.sin(azim_angle),
                    sin_elev * ti.cos(azim_angle),
                    cos_elev
                )
                
                # Solid angle
                elev_low = i_elev * (PI / 2.0) / n_elev
                elev_high = (i_elev + 1) * (PI / 2.0) / n_elev
                d_omega = (TWO_PI / n_azim) * (ti.cos(elev_low) - ti.cos(elev_high))
                
                # Check if direction is valid for this surface
                cos_angle = (ray_dir[0] * normal[0] + ray_dir[1] * normal[1] + 
                            ray_dir[2] * normal[2])
                
                if cos_angle > 0:
                    # Fraction of diffuse flux from this direction
                    dir_flux = diffuse_flux * cos_angle * d_omega / PI * area
                    
                    # Ray trace with CSF accumulation
                    in_domain, t_enter, t_exit = ray_aabb_intersect(
                        pos, ray_dir, domain_min, domain_max, 0.0, self.max_dist
                    )
                    
                    if in_domain == 1:
                        t = 1e-5
                        cumulative_opacity = 0.0
                        
                        current_pos = pos + ray_dir * t
                        
                        ix = ti.cast(ti.floor(current_pos[0] / self.dx), ti.i32)
                        iy = ti.cast(ti.floor(current_pos[1] / self.dy), ti.i32)
                        iz = ti.cast(ti.floor(current_pos[2] / self.dz), ti.i32)
                        
                        ix = ti.max(0, ti.min(self.nx - 1, ix))
                        iy = ti.max(0, ti.min(self.ny - 1, iy))
                        iz = ti.max(0, ti.min(self.nz - 1, iz))
                        
                        step_x = 1 if ray_dir[0] >= 0 else -1
                        step_y = 1 if ray_dir[1] >= 0 else -1
                        step_z = 1 if ray_dir[2] >= 0 else -1
                        
                        # Initialize t_max values before branching (for Taichi)
                        t_max_x = 1e30
                        t_max_y = 1e30
                        t_max_z = 1e30
                        t_delta_x = 1e30
                        t_delta_y = 1e30
                        t_delta_z = 1e30
                        
                        if ti.abs(ray_dir[0]) > 1e-10:
                            if step_x > 0:
                                t_max_x = ((ix + 1) * self.dx - current_pos[0]) / ray_dir[0] + t
                            else:
                                t_max_x = (ix * self.dx - current_pos[0]) / ray_dir[0] + t
                            t_delta_x = ti.abs(self.dx / ray_dir[0])
                        
                        if ti.abs(ray_dir[1]) > 1e-10:
                            if step_y > 0:
                                t_max_y = ((iy + 1) * self.dy - current_pos[1]) / ray_dir[1] + t
                            else:
                                t_max_y = (iy * self.dy - current_pos[1]) / ray_dir[1] + t
                            t_delta_y = ti.abs(self.dy / ray_dir[1])
                        
                        if ti.abs(ray_dir[2]) > 1e-10:
                            if step_z > 0:
                                t_max_z = ((iz + 1) * self.dz - current_pos[2]) / ray_dir[2] + t
                            else:
                                t_max_z = (iz * self.dz - current_pos[2]) / ray_dir[2] + t
                            t_delta_z = ti.abs(self.dz / ray_dir[2])
                        
                        t_prev = t
                        max_steps = self.nx + self.ny + self.nz
                        
                        for _ in range(max_steps):
                            if ix < 0 or ix >= self.nx or iy < 0 or iy >= self.ny or iz < 0 or iz >= self.nz:
                                break
                            if t > t_exit:
                                break
                            if is_solid[ix, iy, iz] == 1:
                                break
                            
                            t_next = ti.min(t_max_x, ti.min(t_max_y, t_max_z))
                            path_len = t_next - t_prev
                            
                            cell_lad = lad[ix, iy, iz]
                            if cell_lad > 0.0:
                                trans_before = ti.exp(-cumulative_opacity)
                                cell_opacity = self.ext_coef * cell_lad * path_len
                                trans_after = ti.exp(-(cumulative_opacity + cell_opacity))
                                absorbed_frac = trans_before - trans_after
                                
                                ti.atomic_add(self.csf[ix, iy, iz], absorbed_frac * dir_flux)
                                cumulative_opacity += cell_opacity
                            
                            t_prev = t_next
                            
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
    
    @ti.kernel
    def compute_canopy_absorption_direct(
        self,
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template(),
        lad: ti.template(),
        incoming_flux: ti.f32
    ):
        """
        Compute direct solar absorption in canopy by tracing rays from sky.
        
        This traces ONE ray per column from the top of the domain downward,
        following the sun direction. This correctly computes absorption without
        overcounting from multiple surfaces.
        
        The result is stored in self.csf as total absorbed power (W) per cell.
        To convert to PALM-compatible W/m³, divide by grid_volume:
            pcbinswdir = csf[i,j,k] * grid_volume_inverse
        
        Note: This is an alternative method to compute_canopy_absorption_direct_palm
        which directly follows PALM's box_absorb methodology.
        
        Args:
            sun_dir: Sun direction unit vector (pointing toward sun)
            is_solid: 3D solid field
            lad: 3D Leaf Area Density field
            incoming_flux: Incoming direct solar flux (W/m²)
        """
        # Trace from each (i,j) column on the top of the domain
        for ix, iy in ti.ndrange(self.nx, self.ny):
            # Start position at top of domain
            start_x = (ix + 0.5) * self.dx
            start_y = (iy + 0.5) * self.dy
            start_z = self.nz * self.dz - 0.01  # Just below top
            
            pos = Vector3(start_x, start_y, start_z)
            
            # Ray direction: opposite of sun direction (tracing FROM sun)
            ray_dir = Vector3(-sun_dir[0], -sun_dir[1], -sun_dir[2])
            
            # Only trace if sun is above horizon (ray goes down)
            if ray_dir[2] < 0:
                domain_min = Vector3(0.0, 0.0, 0.0)
                domain_max = Vector3(self.nx * self.dx, self.ny * self.dy, self.nz * self.dz)
                
                in_domain, t_enter, t_exit = ray_aabb_intersect(
                    pos, ray_dir, domain_min, domain_max, 0.0, self.max_dist
                )
                
                if in_domain == 1:
                    t = 0.01  # Start tracing
                    cumulative_opacity = 0.0
                    
                    # Initialize position
                    ci = ix
                    cj = iy
                    ck = ti.cast(ti.floor((pos[2] + ray_dir[2] * t) / self.dz), ti.i32)
                    ck = ti.max(0, ti.min(self.nz - 1, ck))
                    
                    step_x = 1 if ray_dir[0] >= 0 else -1
                    step_y = 1 if ray_dir[1] >= 0 else -1
                    step_z = -1  # Always going down
                    
                    # Initialize t_max values
                    t_max_x = 1e30
                    t_max_y = 1e30
                    t_max_z = 1e30
                    t_delta_x = 1e30
                    t_delta_y = 1e30
                    t_delta_z = 1e30
                    
                    current_pos = pos + ray_dir * t
                    
                    if ti.abs(ray_dir[0]) > 1e-10:
                        if step_x > 0:
                            t_max_x = ((ci + 1) * self.dx - current_pos[0]) / ray_dir[0] + t
                        else:
                            t_max_x = (ci * self.dx - current_pos[0]) / ray_dir[0] + t
                        t_delta_x = ti.abs(self.dx / ray_dir[0])
                    
                    if ti.abs(ray_dir[1]) > 1e-10:
                        if step_y > 0:
                            t_max_y = ((cj + 1) * self.dy - current_pos[1]) / ray_dir[1] + t
                        else:
                            t_max_y = (cj * self.dy - current_pos[1]) / ray_dir[1] + t
                        t_delta_y = ti.abs(self.dy / ray_dir[1])
                    
                    if ti.abs(ray_dir[2]) > 1e-10:
                        # Always step_z = -1 (going down)
                        t_max_z = (ck * self.dz - current_pos[2]) / ray_dir[2] + t
                        t_delta_z = ti.abs(self.dz / ray_dir[2])
                    
                    t_prev = t
                    max_steps = self.nx + self.ny + self.nz
                    
                    # Cross-sectional area of the cell perpendicular to sun
                    # For a cell of size dx*dy, when sun is at zenith angle θ:
                    # The horizontal area is dx*dy, flux is per horizontal m²
                    # So power through cell = flux * dx * dy
                    cell_area = self.dx * self.dy
                    
                    for _ in range(max_steps):
                        if ci < 0 or ci >= self.nx or cj < 0 or cj >= self.ny or ck < 0 or ck >= self.nz:
                            break
                        
                        # Stop at solid obstacle
                        if is_solid[ci, cj, ck] == 1:
                            break
                        
                        # Find next t
                        t_next = ti.min(t_max_x, ti.min(t_max_y, t_max_z))
                        path_len = t_next - t_prev
                        
                        # Process canopy cell
                        cell_lad = lad[ci, cj, ck]
                        if cell_lad > 0.0:
                            trans_before = ti.exp(-cumulative_opacity)
                            cell_opacity = self.ext_coef * cell_lad * path_len
                            trans_after = ti.exp(-(cumulative_opacity + cell_opacity))
                            absorbed_frac = trans_before - trans_after
                            
                            # Power absorbed = flux * area * absorbed_frac
                            # Store as power (W) - will be divided by volume later
                            ti.atomic_add(self.csf[ci, cj, ck], absorbed_frac * incoming_flux * cell_area)
                            
                            cumulative_opacity += cell_opacity
                        
                        t_prev = t_next
                        
                        # Step to next voxel
                        if t_max_x < t_max_y and t_max_x < t_max_z:
                            t = t_max_x
                            ci += step_x
                            t_max_x += t_delta_x
                        elif t_max_y < t_max_z:
                            t = t_max_y
                            cj += step_y
                            t_max_y += t_delta_y
                        else:
                            t = t_max_z
                            ck += step_z
                            t_max_z += t_delta_z

    def get_csf_numpy(self):
        """Get CSF field as numpy array (units depend on computation method)."""
        return self.csf.to_numpy()
    
    def get_csf_wm3(self):
        """
        Get CSF field as numpy array in W/m³ (PALM-compatible units).
        
        This divides the stored values by grid_volume to ensure consistent units.
        """
        return self.csf.to_numpy() * self.grid_volume_inverse
    
    def get_total_canopy_absorption(self) -> float:
        """
        Get total radiation absorbed by canopy (W).
        
        For PALM-compatible pcbinsw in W/m³, use get_csf_wm3().
        """
        return float(self.csf.to_numpy().sum())
