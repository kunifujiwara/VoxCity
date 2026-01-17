"""Volumetric radiative flux calculation for palm-solar.

Computes 3D radiation fields at each grid cell, not just at surfaces.
Based on PALM's radiation_volumetric_flux feature.

Key outputs:
- skyvf_vol: Volumetric sky view factor at each (i, j, k)
- swflux_vol: Omnidirectional volumetric SW flux at each (i, j, k)
- swflux_reflected_vol: Reflected radiation from surfaces at each (i, j, k)
- shadow_top: Shadow height for each solar direction

Reflected radiation mode:
- When include_reflections=True, the volumetric flux includes radiation
  reflected from buildings, ground, and tree surfaces
- Reflections are traced from each surface element to volumetric grid cells
- Uses Beer-Lambert attenuation through vegetation
"""

import taichi as ti
import numpy as np
import math
from typing import Optional, Tuple, List, Union
from enum import Enum

from .core import Vector3, Point3, PI, TWO_PI, EXT_COEF


class VolumetricFluxMode(Enum):
    """Mode for volumetric flux computation."""
    DIRECT_DIFFUSE = "direct_diffuse"  # Only direct + diffuse sky radiation
    WITH_REFLECTIONS = "with_reflections"  # Include reflected radiation from surfaces


@ti.data_oriented
class VolumetricFluxCalculator:
    """
    GPU-accelerated volumetric radiative flux calculator.
    
    Computes 3D radiation fields throughout the domain volume,
    not just at surface elements. This is useful for:
    - Mean Radiant Temperature (MRT) calculations
    - Photolysis rate estimation
    - Plant canopy light availability
    - Pedestrian thermal comfort
    
    Modes:
    - DIRECT_DIFFUSE: Only direct solar + diffuse sky radiation (faster)
    - WITH_REFLECTIONS: Includes reflected radiation from buildings/ground/trees
    """
    
    def __init__(
        self,
        domain,
        n_azimuth: int = 36,
        min_opaque_lad: float = 0.5
    ):
        """
        Initialize volumetric flux calculator.
        
        Args:
            domain: Domain object with grid geometry
            n_azimuth: Number of azimuthal directions for horizon tracing
            min_opaque_lad: Minimum LAD value considered opaque for shadow purposes
        """
        self.domain = domain
        self.nx = domain.nx
        self.ny = domain.ny
        self.nz = domain.nz
        self.dx = domain.dx
        self.dy = domain.dy
        self.dz = domain.dz
        
        self.n_azimuth = n_azimuth
        self.min_opaque_lad = min_opaque_lad
        
        # Default mode
        self.mode = VolumetricFluxMode.DIRECT_DIFFUSE
        
        # Maximum trace distance
        self.max_dist = math.sqrt(
            (self.nx * self.dx)**2 + 
            (self.ny * self.dy)**2 + 
            (self.nz * self.dz)**2
        )
        
        # Volumetric sky view factor: fraction of sky visible from each grid cell
        self.skyvf_vol = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # Volumetric SW flux: omnidirectional flux at each grid cell (W/m²)
        # Represents average irradiance onto an imaginary sphere
        self.swflux_vol = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # Volumetric reflected SW flux: radiation reflected from surfaces (W/m²)
        self.swflux_reflected_vol = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # Separate components for analysis
        self.swflux_direct_vol = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        self.swflux_diffuse_vol = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # Opaque top: highest level that blocks direct radiation
        # Considers both buildings and dense vegetation
        self.opaque_top = ti.field(dtype=ti.i32, shape=(self.nx, self.ny))
        
        # Shadow top per solar direction: highest level in shadow
        # For single solar direction (current sun position)
        self.shadow_top = ti.field(dtype=ti.i32, shape=(self.nx, self.ny))
        
        # Horizon angle for each (i, j, k, azimuth) - temporary storage
        # Stored as tangent of elevation angle
        self._horizon_tan = ti.field(dtype=ti.f32, shape=(self.nx, self.ny, self.nz))
        
        # Pre-computed azimuth directions
        self.azim_dir_x = ti.field(dtype=ti.f32, shape=(n_azimuth,))
        self.azim_dir_y = ti.field(dtype=ti.f32, shape=(n_azimuth,))
        
        self._init_azimuth_directions()
        
        # Flag for computed state
        self._skyvf_computed = False
        
        # ========== Cell-to-Surface View Factor (C2S-VF) Matrix Caching ==========
        # Pre-compute which surfaces each voxel cell can see and their view factors.
        # This makes reflected flux computation O(nnz) instead of O(N_cells * N_surfaces).
        # 
        # Stored in sparse COO format:
        # - c2s_cell_idx[i]: Linear cell index (i * ny * nz + j * nz + k)
        # - c2s_surf_idx[i]: Surface index
        # - c2s_vf[i]: View factor (includes geometry + transmissivity)
        #
        # For multi-timestep simulations with reflections, call compute_c2s_matrix()
        # once to pre-compute, then use fast compute_reflected_flux_vol_cached().
        
        self._c2s_matrix_cached = False
        self._c2s_nnz = 0
        
        # Estimate max non-zeros: assume each cell sees ~50 surfaces on average
        # For a 200x200x50 domain = 2M cells * 50 = 100M entries max
        # Cap at reasonable memory limit (~1.6GB for the 4 arrays)
        n_cells = self.nx * self.ny * self.nz
        estimated_entries = min(n_cells * 50, 100_000_000)
        self._max_c2s_entries = estimated_entries
        
        # Sparse COO arrays for C2S-VF matrix (allocated on demand)
        self._c2s_cell_idx = None
        self._c2s_surf_idx = None
        self._c2s_vf = None
        self._c2s_count = None
        
        # Pre-allocated surface outgoing field for efficient repeated calls
        self._surf_out_field = None
        self._surf_out_max_size = 0
        
        # ========== Cumulative Terrain-Following Accumulation (Optimization) ==========
        # For cumulative simulations, accumulate terrain-following slices directly on GPU
        # instead of transferring full 3D arrays each timestep/patch.
        # This provides 10-15x speedup for cumulative volumetric calculations.
        
        # 2D cumulative map for terrain-following irradiance accumulation
        self._cumulative_map = ti.field(dtype=ti.f64, shape=(self.nx, self.ny))
        
        # Ground k-levels for terrain-following extraction (set by init_cumulative_accumulation)
        self._ground_k = ti.field(dtype=ti.i32, shape=(self.nx, self.ny))
        
        # Height offset for extraction (cells above ground)
        self._height_offset_k = 0
        
        # Whether cumulative accumulation is initialized
        self._cumulative_initialized = False
    
    @ti.kernel
    def _init_azimuth_directions(self):
        """Pre-compute azimuth direction vectors."""
        for iaz in range(self.n_azimuth):
            azimuth = (ti.cast(iaz, ti.f32) + 0.5) * TWO_PI / ti.cast(self.n_azimuth, ti.f32)
            # x = east (sin), y = north (cos)
            self.azim_dir_x[iaz] = ti.sin(azimuth)
            self.azim_dir_y[iaz] = ti.cos(azimuth)
    
    @ti.kernel
    def _compute_opaque_top(
        self,
        is_solid: ti.template(),
        lad: ti.template(),
        has_lad: ti.i32
    ):
        """
        Compute the opaque top level for each column.
        
        Considers both solid obstacles (buildings) and dense vegetation.
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            # Start with terrain/building top
            top_k = 0
            for k in range(self.nz):
                if is_solid[i, j, k] == 1:
                    top_k = k
            
            # Check vegetation above solid top (iterate downward from top)
            if has_lad == 1:
                # Taichi doesn't support 3-arg range with step, so we iterate forward
                # and compute the reversed index
                num_levels = self.nz - 1 - top_k
                for k_rev in range(num_levels):
                    k = self.nz - 1 - k_rev
                    if lad[i, j, k] >= self.min_opaque_lad:
                        if k > top_k:
                            top_k = k
                        break
            
            self.opaque_top[i, j] = top_k
    
    @ti.func
    def _trace_horizon_single_azimuth(
        self,
        i_start: ti.i32,
        j_start: ti.i32,
        k_level: ti.i32,
        dir_x: ti.f32,
        dir_y: ti.f32,
        is_solid: ti.template()
    ) -> ti.f32:
        """
        Trace horizon in a single azimuth direction from a point.
        
        Returns the tangent of the horizon elevation angle.
        A higher value means more sky is blocked.
        Only considers solid obstacles (buildings), not vegetation.
        """
        # Starting position (center of grid cell)
        x0 = (ti.cast(i_start, ti.f32) + 0.5) * self.dx
        y0 = (ti.cast(j_start, ti.f32) + 0.5) * self.dy
        z0 = (ti.cast(k_level, ti.f32) + 0.5) * self.dz
        
        max_horizon_tan = -1e10  # Start below horizon
        
        # Step along the direction
        step_dist = ti.min(self.dx, self.dy)
        n_steps = ti.cast(self.max_dist / step_dist, ti.i32) + 1
        
        for step in range(1, n_steps):
            dist = ti.cast(step, ti.f32) * step_dist
            
            x = x0 + dir_x * dist
            y = y0 + dir_y * dist
            
            # Check if out of domain
            if x < 0.0 or x >= self.nx * self.dx:
                break
            if y < 0.0 or y >= self.ny * self.dy:
                break
            
            # Grid indices
            ix = ti.cast(ti.floor(x / self.dx), ti.i32)
            iy = ti.cast(ti.floor(y / self.dy), ti.i32)
            
            ix = ti.max(0, ti.min(self.nx - 1, ix))
            iy = ti.max(0, ti.min(self.ny - 1, iy))
            
            # Find solid top at this location (not opaque_top which includes vegetation)
            solid_top_k = 0
            for kk in range(self.nz):
                if is_solid[ix, iy, kk] == 1:
                    solid_top_k = kk
            
            obstacle_z = (ti.cast(solid_top_k, ti.f32) + 1.0) * self.dz  # Top of obstacle
            
            # Compute elevation angle tangent to obstacle top
            dz = obstacle_z - z0
            horizon_tan = dz / dist
            
            if horizon_tan > max_horizon_tan:
                max_horizon_tan = horizon_tan
        
        return max_horizon_tan
    
    @ti.func
    def _trace_transmissivity_zenith(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        zenith_angle: ti.f32,
        azimuth: ti.f32,
        is_solid: ti.template(),
        lad: ti.template(),
        has_lad: ti.i32
    ) -> ti.f32:
        """
        Trace transmissivity from a point toward sky at given zenith/azimuth.
        
        Returns transmissivity [0, 1] accounting for:
        - Solid obstacles (transmissivity = 0)
        - Vegetation (Beer-Lambert attenuation)
        
        Args:
            i, j, k: Starting grid cell
            zenith_angle: Angle from vertical (0 = straight up)
            azimuth: Horizontal angle (0 = north, π/2 = east)
            is_solid: Solid obstacle field
            lad: Leaf Area Density field
            has_lad: Whether LAD field exists
        """
        # Direction vector (pointing toward sky)
        sin_zen = ti.sin(zenith_angle)
        cos_zen = ti.cos(zenith_angle)
        dir_x = sin_zen * ti.sin(azimuth)  # East component
        dir_y = sin_zen * ti.cos(azimuth)  # North component
        dir_z = cos_zen                     # Up component
        
        # Starting position
        x = (ti.cast(i, ti.f32) + 0.5) * self.dx
        y = (ti.cast(j, ti.f32) + 0.5) * self.dy
        z = (ti.cast(k, ti.f32) + 0.5) * self.dz
        
        # Accumulated LAD path length
        cumulative_lad_path = 0.0
        transmissivity = 1.0
        
        # Step size based on grid resolution
        step_dist = ti.min(self.dx, ti.min(self.dy, self.dz)) * 0.5
        max_steps = ti.cast(self.max_dist / step_dist, ti.i32) + 1
        
        for step in range(1, max_steps):
            dist = ti.cast(step, ti.f32) * step_dist
            
            # Current position
            cx = x + dir_x * dist
            cy = y + dir_y * dist
            cz = z + dir_z * dist
            
            # Check bounds
            if cx < 0.0 or cx >= self.nx * self.dx:
                break
            if cy < 0.0 or cy >= self.ny * self.dy:
                break
            if cz < 0.0 or cz >= self.nz * self.dz:
                break  # Exited domain through top - reached sky
            
            # Grid indices
            ix = ti.cast(ti.floor(cx / self.dx), ti.i32)
            iy = ti.cast(ti.floor(cy / self.dy), ti.i32)
            iz = ti.cast(ti.floor(cz / self.dz), ti.i32)
            
            ix = ti.max(0, ti.min(self.nx - 1, ix))
            iy = ti.max(0, ti.min(self.ny - 1, iy))
            iz = ti.max(0, ti.min(self.nz - 1, iz))
            
            # Check for solid obstacle - completely blocks
            if is_solid[ix, iy, iz] == 1:
                transmissivity = 0.0
                break
            
            # Accumulate LAD for Beer-Lambert
            if has_lad == 1:
                cell_lad = lad[ix, iy, iz]
                if cell_lad > 0.0:
                    cumulative_lad_path += cell_lad * step_dist
        
        # Apply Beer-Lambert if passed through vegetation
        if transmissivity > 0.0 and cumulative_lad_path > 0.0:
            transmissivity = ti.exp(-EXT_COEF * cumulative_lad_path)
        
        return transmissivity
    
    @ti.kernel
    def _compute_skyvf_vol_kernel(
        self,
        is_solid: ti.template()
    ):
        """
        Compute volumetric sky view factor for all grid cells.
        
        For each cell, traces horizons in all azimuth directions
        and integrates the visible sky fraction.
        This version only considers solid obstacles (no vegetation).
        """
        n_az_f = ti.cast(self.n_azimuth, ti.f32)
        
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            # Skip cells inside solid obstacles
            if is_solid[i, j, k] == 1:
                self.skyvf_vol[i, j, k] = 0.0
                continue
            
            # Integrate sky view over all azimuths
            total_svf = 0.0
            
            for iaz in range(self.n_azimuth):
                dir_x = self.azim_dir_x[iaz]
                dir_y = self.azim_dir_y[iaz]
                
                # Get horizon tangent in this direction (solid obstacles only)
                horizon_tan = self._trace_horizon_single_azimuth(
                    i, j, k, dir_x, dir_y, is_solid
                )
                
                # Convert tangent to elevation angle, then to cos(zenith)
                cos_zen = 0.0
                if horizon_tan >= 0.0:
                    cos_zen = horizon_tan / ti.sqrt(1.0 + horizon_tan * horizon_tan)
                
                # Sky view contribution: (1 - cos_zenith_of_horizon)
                svf_contrib = (1.0 - cos_zen)
                total_svf += svf_contrib
            
            # Normalize: divide by number of azimuths and factor of 2 for hemisphere
            self.skyvf_vol[i, j, k] = total_svf / (2.0 * n_az_f)
    
    @ti.kernel
    def _compute_skyvf_vol_with_lad_kernel(
        self,
        is_solid: ti.template(),
        lad: ti.template(),
        n_zenith: ti.i32
    ):
        """
        Compute volumetric sky view factor with vegetation transmissivity.
        
        Integrates over hemisphere using discrete zenith and azimuth angles,
        applying Beer-Lambert attenuation through vegetation.
        
        SVF = (1/2π) ∫∫ τ(θ,φ) cos(θ) sin(θ) dθ dφ
        
        where τ is transmissivity through vegetation/obstacles.
        """
        n_az_f = ti.cast(self.n_azimuth, ti.f32)
        n_zen_f = ti.cast(n_zenith, ti.f32)
        
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            # Skip cells inside solid obstacles
            if is_solid[i, j, k] == 1:
                self.skyvf_vol[i, j, k] = 0.0
                continue
            
            # Integrate over hemisphere
            # SVF = (1/2π) ∫₀^(π/2) ∫₀^(2π) τ(θ,φ) cos(θ) sin(θ) dθ dφ
            # Discretized with uniform spacing
            total_weighted_trans = 0.0
            total_weight = 0.0
            
            for izen in range(n_zenith):
                # Zenith angle from 0 (up) to π/2 (horizontal)
                # Use midpoint of each bin
                zenith = (ti.cast(izen, ti.f32) + 0.5) * (PI / 2.0) / n_zen_f
                
                # Weight: cos(θ) * sin(θ) * dθ
                # This accounts for solid angle and projection
                cos_zen = ti.cos(zenith)
                sin_zen = ti.sin(zenith)
                weight = cos_zen * sin_zen
                
                for iaz in range(self.n_azimuth):
                    azimuth = (ti.cast(iaz, ti.f32) + 0.5) * TWO_PI / n_az_f
                    
                    # Trace transmissivity toward sky
                    trans = self._trace_transmissivity_zenith(
                        i, j, k, zenith, azimuth, is_solid, lad, 1
                    )
                    
                    total_weighted_trans += trans * weight
                    total_weight += weight
            
            # Normalize by total weight (integral of cos*sin over hemisphere = 0.5)
            if total_weight > 0.0:
                self.skyvf_vol[i, j, k] = total_weighted_trans / total_weight
            else:
                self.skyvf_vol[i, j, k] = 0.0
    
    @ti.kernel
    def _compute_shadow_top_kernel(
        self,
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template()
    ):
        """
        Compute shadow top for current solar direction.
        
        Shadow top is the highest grid level that is in shadow
        (direct solar radiation blocked).
        """
        # Horizontal direction magnitude
        horiz_mag = ti.sqrt(sun_dir[0]**2 + sun_dir[1]**2)
        
        # Tangent of solar elevation
        solar_tan = 1e10  # Default: sun near zenith
        if horiz_mag > 1e-6:
            solar_tan = sun_dir[2] / horiz_mag
        
        # Horizontal direction components (normalized)
        dir_x = 0.0
        dir_y = 1.0
        if horiz_mag > 1e-6:
            dir_x = sun_dir[0] / horiz_mag
            dir_y = sun_dir[1] / horiz_mag
        
        for i, j in ti.ndrange(self.nx, self.ny):
            # Start from opaque top
            shadow_k = self.opaque_top[i, j]
            
            # Trace upward to find where horizon drops below solar elevation
            for k in range(self.opaque_top[i, j] + 1, self.nz):
                # Get horizon in sun direction
                horizon_tan = self._trace_horizon_single_azimuth(
                    i, j, k, dir_x, dir_y, is_solid
                )
                
                # If horizon is below sun, this level is sunlit
                if horizon_tan < solar_tan:
                    break
                
                shadow_k = k
            
            self.shadow_top[i, j] = shadow_k
    
    @ti.kernel
    def _compute_swflux_vol_kernel(
        self,
        sw_direct: ti.f32,
        sw_diffuse: ti.f32,
        cos_zenith: ti.f32,
        is_solid: ti.template()
    ):
        """
        Compute volumetric shortwave flux at each grid cell.
        
        The flux represents average irradiance onto an imaginary sphere,
        combining direct and diffuse components.
        Also stores separate direct and diffuse components.
        """
        # Sun direct factor (convert horizontal to normal)
        sun_factor = 1.0
        if cos_zenith > 0.0262:  # min_stable_coszen
            sun_factor = 1.0 / cos_zenith
        
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            # Skip solid cells
            if is_solid[i, j, k] == 1:
                self.swflux_vol[i, j, k] = 0.0
                self.swflux_direct_vol[i, j, k] = 0.0
                self.swflux_diffuse_vol[i, j, k] = 0.0
                continue
            
            direct_flux = 0.0
            diffuse_flux = 0.0
            
            # Direct component: only above shadow level
            if k > self.shadow_top[i, j] and cos_zenith > 0.0:
                # For a sphere, the ratio of projected area to surface area is 1/4
                # Direct flux onto sphere = sw_direct * sun_factor * 0.25
                direct_flux = sw_direct * sun_factor * 0.25
            
            # Diffuse component: weighted by volumetric sky view factor
            # For a sphere receiving isotropic diffuse radiation:
            # diffuse_flux = sw_diffuse * skyvf_vol
            diffuse_flux = sw_diffuse * self.skyvf_vol[i, j, k]
            
            self.swflux_direct_vol[i, j, k] = direct_flux
            self.swflux_diffuse_vol[i, j, k] = diffuse_flux
            self.swflux_vol[i, j, k] = direct_flux + diffuse_flux
    
    @ti.kernel
    def _compute_swflux_vol_with_lad_kernel(
        self,
        sw_direct: ti.f32,
        sw_diffuse: ti.f32,
        cos_zenith: ti.f32,
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template(),
        lad: ti.template()
    ):
        """
        Compute volumetric shortwave flux with LAD attenuation.
        
        The flux is attenuated through vegetation using Beer-Lambert law.
        Direct radiation is traced toward the sun with proper attenuation.
        """
        # Sun direct factor (convert horizontal irradiance to normal)
        sun_factor = 1.0
        if cos_zenith > 0.0262:
            sun_factor = 1.0 / cos_zenith
        
        # Compute solar zenith angle for transmissivity tracing
        solar_zenith = ti.acos(ti.max(-1.0, ti.min(1.0, cos_zenith)))
        
        # Compute solar azimuth from sun direction
        solar_azimuth = ti.atan2(sun_dir[0], sun_dir[1])  # atan2(east, north)
        
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            # Skip solid cells
            if is_solid[i, j, k] == 1:
                self.swflux_vol[i, j, k] = 0.0
                self.swflux_direct_vol[i, j, k] = 0.0
                self.swflux_diffuse_vol[i, j, k] = 0.0
                continue
            
            direct_flux = 0.0
            diffuse_flux = 0.0
            
            # Direct component with full 3D transmissivity tracing
            if cos_zenith > 0.0262:  # Sun is up
                # Trace transmissivity toward sun through vegetation and obstacles
                trans = self._trace_transmissivity_zenith(
                    i, j, k, solar_zenith, solar_azimuth, is_solid, lad, 1
                )
                
                # Direct flux onto sphere = sw_direct * sun_factor * 0.25 * transmissivity
                direct_flux = sw_direct * sun_factor * 0.25 * trans
            
            # Diffuse component: SVF already accounts for vegetation attenuation
            # (computed by _compute_skyvf_vol_with_lad_kernel)
            diffuse_flux = sw_diffuse * self.skyvf_vol[i, j, k]
            
            self.swflux_direct_vol[i, j, k] = direct_flux
            self.swflux_diffuse_vol[i, j, k] = diffuse_flux
            self.swflux_vol[i, j, k] = direct_flux + diffuse_flux
    
    @ti.func
    def _compute_canopy_transmissivity(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        sun_dir: ti.types.vector(3, ti.f32),
        is_solid: ti.template(),
        lad: ti.template()
    ) -> ti.f32:
        """
        Compute transmissivity through canopy from point (i,j,k) toward sun.
        
        Uses simplified vertical integration (for efficiency).
        Full 3D ray tracing is done in CSF calculator.
        """
        cumulative_lad_path = 0.0
        blocked = 0
        
        # Integrate upward through canopy
        # Simplified: just sum LAD in vertical column above
        # More accurate would trace along sun direction
        for kk in range(k + 1, self.nz):
            if blocked == 0:
                if is_solid[i, j, kk] == 1:
                    # Hit solid - mark as fully blocked
                    blocked = 1
                    cumulative_lad_path = 1e10  # Large value for zero transmissivity
                else:
                    cell_lad = lad[i, j, kk]
                    if cell_lad > 0.0:
                        # Path length through cell (vertical)
                        # For non-vertical sun, would need angle correction
                        path_len = self.dz / ti.max(0.1, sun_dir[2])  # Avoid division by zero
                        cumulative_lad_path += cell_lad * path_len
        
        # Beer-Lambert transmissivity
        return ti.exp(-EXT_COEF * cumulative_lad_path)
    
    def compute_opaque_top(self):
        """
        Compute opaque top levels considering buildings and vegetation.
        """
        has_lad = 1 if self.domain.lad is not None else 0
        
        if has_lad:
            self._compute_opaque_top(
                self.domain.is_solid,
                self.domain.lad,
                has_lad
            )
        else:
            self._compute_opaque_top_no_lad(self.domain.is_solid)
    
    @ti.kernel
    def _compute_opaque_top_no_lad(self, is_solid: ti.template()):
        """Compute opaque top without vegetation."""
        for i, j in ti.ndrange(self.nx, self.ny):
            top_k = 0
            for k in range(self.nz):
                if is_solid[i, j, k] == 1:
                    top_k = k
            self.opaque_top[i, j] = top_k
    
    def compute_skyvf_vol(self, n_zenith: int = 9):
        """
        Compute volumetric sky view factors for all grid cells.
        
        This is computationally expensive - call once per domain setup
        or when geometry changes.
        
        Args:
            n_zenith: Number of zenith angle divisions for hemisphere integration.
                      Higher values give more accurate results but slower computation.
                      Default 9 gives ~10° resolution.
        """
        print("Computing opaque top levels...")
        self.compute_opaque_top()
        
        has_lad = self.domain.lad is not None
        
        if has_lad:
            print(f"Computing volumetric sky view factors with vegetation...")
            print(f"  ({self.n_azimuth} azimuths × {n_zenith} zenith angles)")
            self._compute_skyvf_vol_with_lad_kernel(
                self.domain.is_solid,
                self.domain.lad,
                n_zenith
            )
        else:
            print(f"Computing volumetric sky view factors ({self.n_azimuth} azimuths)...")
            self._compute_skyvf_vol_kernel(self.domain.is_solid)
        
        self._skyvf_computed = True
        print("Volumetric SVF computation complete.")
    
    def compute_shadow_top(self, sun_direction: Tuple[float, float, float]):
        """
        Compute shadow top for a given solar direction.
        
        Args:
            sun_direction: Unit vector pointing toward sun (x, y, z)
        """
        if not self._skyvf_computed:
            self.compute_opaque_top()
        
        sun_dir = ti.Vector([sun_direction[0], sun_direction[1], sun_direction[2]])
        self._compute_shadow_top_kernel(sun_dir, self.domain.is_solid)
    
    def compute_swflux_vol(
        self,
        sw_direct: float,
        sw_diffuse: float,
        cos_zenith: float,
        sun_direction: Tuple[float, float, float],
        lad: Optional[ti.template] = None
    ):
        """
        Compute volumetric shortwave flux for all grid cells.
        
        Args:
            sw_direct: Direct normal irradiance (W/m²)
            sw_diffuse: Diffuse horizontal irradiance (W/m²)
            cos_zenith: Cosine of solar zenith angle
            sun_direction: Unit vector toward sun (x, y, z)
            lad: Optional LAD field for canopy attenuation
        """
        if not self._skyvf_computed:
            print("Warning: Volumetric SVF not computed, computing now...")
            self.compute_skyvf_vol()
        
        # Compute shadow heights for current sun position
        self.compute_shadow_top(sun_direction)
        
        # Compute flux (with or without LAD attenuation)
        if lad is not None:
            sun_dir = ti.Vector([sun_direction[0], sun_direction[1], sun_direction[2]])
            self._compute_swflux_vol_with_lad_kernel(
                sw_direct,
                sw_diffuse,
                cos_zenith,
                sun_dir,
                self.domain.is_solid,
                lad
            )
        else:
            self._compute_swflux_vol_kernel(
                sw_direct,
                sw_diffuse,
                cos_zenith,
                self.domain.is_solid
            )
    
    def get_skyvf_vol(self) -> np.ndarray:
        """Get volumetric sky view factor as numpy array."""
        return self.skyvf_vol.to_numpy()
    
    def get_swflux_vol(self) -> np.ndarray:
        """Get volumetric SW flux as numpy array (W/m²)."""
        return self.swflux_vol.to_numpy()
    
    def get_shadow_top(self) -> np.ndarray:
        """Get shadow top indices as numpy array."""
        return self.shadow_top.to_numpy()
    
    def get_opaque_top(self) -> np.ndarray:
        """Get opaque top indices as numpy array."""
        return self.opaque_top.to_numpy()
    
    def get_shadow_mask_3d(self) -> np.ndarray:
        """
        Get 3D shadow mask (1=shadowed, 0=sunlit).
        
        Returns:
            3D boolean array where True indicates shadowed cells
        """
        shadow_top = self.shadow_top.to_numpy()
        is_solid = self.domain.is_solid.to_numpy()
        
        mask = np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        
        for i in range(self.nx):
            for j in range(self.ny):
                k_shadow = shadow_top[i, j]
                mask[i, j, :k_shadow+1] = True
        
        # Also mark solid cells
        mask[is_solid == 1] = True
        
        return mask
    
    def get_horizontal_slice(self, k: int, field: str = 'swflux') -> np.ndarray:
        """
        Get horizontal slice of a volumetric field.
        
        Args:
            k: Vertical level index
            field: 'swflux' or 'skyvf'
        
        Returns:
            2D array at level k
        """
        if field == 'swflux':
            return self.swflux_vol.to_numpy()[:, :, k]
        elif field == 'skyvf':
            return self.skyvf_vol.to_numpy()[:, :, k]
        else:
            raise ValueError(f"Unknown field: {field}")
    
    def get_vertical_slice(
        self, 
        axis: str, 
        index: int, 
        field: str = 'swflux'
    ) -> np.ndarray:
        """
        Get vertical slice of a volumetric field.
        
        Args:
            axis: 'x' or 'y'
            index: Index along the axis
            field: 'swflux' or 'skyvf'
        
        Returns:
            2D array (horizontal_coord, z)
        """
        if field == 'swflux':
            data = self.swflux_vol.to_numpy()
        elif field == 'skyvf':
            data = self.skyvf_vol.to_numpy()
        else:
            raise ValueError(f"Unknown field: {field}")
        
        if axis == 'x':
            return data[index, :, :]
        elif axis == 'y':
            return data[:, index, :]
        else:
            raise ValueError(f"Unknown axis: {axis}")
    
    def set_mode(self, mode: Union[VolumetricFluxMode, str]):
        """
        Set the volumetric flux computation mode.
        
        Args:
            mode: Either a VolumetricFluxMode enum or string:
                  'direct_diffuse' - Only direct + diffuse sky radiation
                  'with_reflections' - Include reflected radiation from surfaces
        """
        if isinstance(mode, str):
            mode = VolumetricFluxMode(mode)
        self.mode = mode
    
    @ti.func
    def _trace_transmissivity_to_surface(
        self,
        i: ti.i32,
        j: ti.i32,
        k: ti.i32,
        surf_x: ti.f32,
        surf_y: ti.f32,
        surf_z: ti.f32,
        surf_nx: ti.f32,
        surf_ny: ti.f32,
        surf_nz: ti.f32,
        is_solid: ti.template(),
        lad: ti.template(),
        has_lad: ti.i32
    ) -> ti.f32:
        """
        Trace transmissivity from grid cell (i,j,k) to a surface element.
        
        Returns transmissivity [0, 1] accounting for:
        - Solid obstacles (transmissivity = 0)
        - Vegetation (Beer-Lambert attenuation)
        - Visibility check (normal pointing toward cell)
        
        Args:
            i, j, k: Grid cell indices
            surf_x, surf_y, surf_z: Surface center position
            surf_nx, surf_ny, surf_nz: Surface normal vector
            is_solid: Solid obstacle field
            lad: Leaf Area Density field
            has_lad: Whether LAD field exists
        """
        # Cell center position
        cell_x = (ti.cast(i, ti.f32) + 0.5) * self.dx
        cell_y = (ti.cast(j, ti.f32) + 0.5) * self.dy
        cell_z = (ti.cast(k, ti.f32) + 0.5) * self.dz
        
        # Direction from surface to cell
        dx = cell_x - surf_x
        dy = cell_y - surf_y
        dz = cell_z - surf_z
        dist = ti.sqrt(dx*dx + dy*dy + dz*dz)
        
        transmissivity = 0.0
        
        if dist > 0.01:  # Avoid self-intersection
            # Normalize direction
            dir_x = dx / dist
            dir_y = dy / dist
            dir_z = dz / dist
            
            # Check if surface faces the cell (dot product with normal > 0)
            cos_angle = dir_x * surf_nx + dir_y * surf_ny + dir_z * surf_nz
            
            if cos_angle > 0.0:
                transmissivity = 1.0
                cumulative_lad_path = 0.0
                
                # Step along the ray from surface to cell
                step_dist = ti.min(self.dx, ti.min(self.dy, self.dz)) * 0.5
                n_steps = ti.cast(dist / step_dist, ti.i32) + 1
                
                for step in range(1, n_steps):
                    t = ti.cast(step, ti.f32) * step_dist
                    if t >= dist:
                        break
                    
                    # Current position along ray
                    cx = surf_x + dir_x * t
                    cy = surf_y + dir_y * t
                    cz = surf_z + dir_z * t
                    
                    # Check bounds
                    if cx < 0.0 or cx >= self.nx * self.dx:
                        break
                    if cy < 0.0 or cy >= self.ny * self.dy:
                        break
                    if cz < 0.0 or cz >= self.nz * self.dz:
                        break
                    
                    # Grid indices
                    ix = ti.cast(ti.floor(cx / self.dx), ti.i32)
                    iy = ti.cast(ti.floor(cy / self.dy), ti.i32)
                    iz = ti.cast(ti.floor(cz / self.dz), ti.i32)
                    
                    ix = ti.max(0, ti.min(self.nx - 1, ix))
                    iy = ti.max(0, ti.min(self.ny - 1, iy))
                    iz = ti.max(0, ti.min(self.nz - 1, iz))
                    
                    # Check for solid obstacle - blocks completely
                    if is_solid[ix, iy, iz] == 1:
                        transmissivity = 0.0
                        break
                    
                    # Accumulate LAD for Beer-Lambert
                    if has_lad == 1:
                        cell_lad = lad[ix, iy, iz]
                        if cell_lad > 0.0:
                            cumulative_lad_path += cell_lad * step_dist
                
                # Apply Beer-Lambert attenuation
                if transmissivity > 0.0 and cumulative_lad_path > 0.0:
                    transmissivity = ti.exp(-EXT_COEF * cumulative_lad_path)
                
                # Apply geometric factor: cos(angle) / distance^2
                # Normalized to produce flux in W/m²
                transmissivity *= cos_angle
        
        return transmissivity
    
    @ti.kernel
    def _compute_reflected_flux_kernel(
        self,
        n_surfaces: ti.i32,
        surf_center: ti.template(),
        surf_normal: ti.template(),
        surf_area: ti.template(),
        surf_outgoing: ti.template(),
        is_solid: ti.template(),
        lad: ti.template(),
        has_lad: ti.i32
    ):
        """
        Compute volumetric reflected flux from surface outgoing radiation.
        
        For each grid cell, integrates reflected radiation from all visible
        surfaces weighted by view factor and transmissivity.
        
        Uses a max distance cutoff for performance - view factor drops as 1/d²,
        so distant surfaces contribute negligibly.
        
        Args:
            n_surfaces: Number of surface elements
            surf_center: Surface center positions (n_surfaces, 3)
            surf_normal: Surface normal vectors (n_surfaces, 3)
            surf_area: Surface areas (n_surfaces,)
            surf_outgoing: Surface outgoing radiation in W/m² (n_surfaces,)
            is_solid: Solid obstacle field
            lad: Leaf Area Density field
            has_lad: Whether LAD field exists
        """
        # For a sphere at each grid cell, reflected flux is:
        # flux = Σ (surfout * area * transmissivity * cos_angle) / (4 * π * dist²)
        # The factor 0.25 accounts for sphere geometry (projected area / surface area)
        
        # Maximum distance for reflections (meters). Beyond this, VF is negligible.
        # At 30m with 1m² area, VF contribution is ~1/(π*900) ≈ 0.035%
        max_dist_sq = 900.0  # 30m max distance
        
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            # Skip solid cells
            if is_solid[i, j, k] == 1:
                self.swflux_reflected_vol[i, j, k] = 0.0
                continue
            
            cell_x = (ti.cast(i, ti.f32) + 0.5) * self.dx
            cell_y = (ti.cast(j, ti.f32) + 0.5) * self.dy
            cell_z = (ti.cast(k, ti.f32) + 0.5) * self.dz
            
            total_reflected = 0.0
            
            for surf_idx in range(n_surfaces):
                outgoing = surf_outgoing[surf_idx]
                
                # Skip surfaces with negligible outgoing radiation
                if outgoing > 0.1:  # W/m² threshold
                    surf_x = surf_center[surf_idx][0]
                    surf_y = surf_center[surf_idx][1]
                    surf_z = surf_center[surf_idx][2]
                    
                    # Distance to surface - early exit for distant surfaces
                    dx = cell_x - surf_x
                    dy = cell_y - surf_y
                    dz = cell_z - surf_z
                    dist_sq = dx*dx + dy*dy + dz*dz
                    
                    # Skip if beyond max distance or too close
                    if dist_sq > 0.01 and dist_sq < max_dist_sq:
                        surf_nx = surf_normal[surf_idx][0]
                        surf_ny = surf_normal[surf_idx][1]
                        surf_nz = surf_normal[surf_idx][2]
                        area = surf_area[surf_idx]
                        
                        dist = ti.sqrt(dist_sq)
                        
                        # Direction from surface to cell (normalized)
                        dir_x = dx / dist
                        dir_y = dy / dist
                        dir_z = dz / dist
                        
                        # Cosine of angle between normal and direction
                        cos_angle = dir_x * surf_nx + dir_y * surf_ny + dir_z * surf_nz
                        
                        if cos_angle > 0.0:  # Surface faces the cell
                            # Get transmissivity through vegetation/obstacles
                            trans = self._trace_transmissivity_to_surface(
                                i, j, k, surf_x, surf_y, surf_z,
                                surf_nx, surf_ny, surf_nz,
                                is_solid, lad, has_lad
                            )
                            
                            if trans > 0.0:
                                # View factor contribution: (A * cos_θ) / (π * d²)
                                # For omnidirectional sphere: multiply by 0.25
                                vf = area * cos_angle / (PI * dist_sq)
                                contribution = outgoing * vf * trans * 0.25
                                total_reflected += contribution
            
            self.swflux_reflected_vol[i, j, k] = total_reflected
    
    def compute_reflected_flux_vol(
        self,
        surfaces,
        surf_outgoing: np.ndarray
    ):
        """
        Compute volumetric reflected flux from surface outgoing radiation.
        
        This propagates reflected radiation from surfaces into the 3D volume.
        Should be called after surface reflection calculations are complete.
        
        NOTE: This is O(N_cells * N_surfaces) and can be slow for large domains.
        For repeated calls, use compute_c2s_matrix() once followed by
        compute_reflected_flux_vol_cached() for O(nnz) computation.
        
        Args:
            surfaces: Surfaces object with geometry (center, normal, area)
            surf_outgoing: Array of surface outgoing radiation (W/m²)
                           Shape: (n_surfaces,)
        """
        n_surfaces = surfaces.n_surfaces[None]
        
        if n_surfaces == 0:
            return
        
        # Re-use pre-allocated field if available, otherwise create temporary
        if self._surf_out_field is not None and self._surf_out_max_size >= n_surfaces:
            self._surf_out_field.from_numpy(surf_outgoing[:n_surfaces].astype(np.float32))
            surf_out_field = self._surf_out_field
        else:
            # Create temporary taichi field for outgoing radiation
            surf_out_field = ti.field(dtype=ti.f32, shape=(n_surfaces,))
            surf_out_field.from_numpy(surf_outgoing[:n_surfaces].astype(np.float32))
        
        has_lad = 1 if self.domain.lad is not None else 0
        
        if has_lad:
            self._compute_reflected_flux_kernel(
                n_surfaces,
                surfaces.center,
                surfaces.normal,
                surfaces.area,
                surf_out_field,
                self.domain.is_solid,
                self.domain.lad,
                has_lad
            )
        else:
            self._compute_reflected_flux_kernel(
                n_surfaces,
                surfaces.center,
                surfaces.normal,
                surfaces.area,
                surf_out_field,
                self.domain.is_solid,
                self.domain.lad,
                0
            )
    
    @ti.kernel
    def _add_reflected_to_total(self):
        """Add reflected flux to total volumetric flux."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.swflux_vol[i, j, k] += self.swflux_reflected_vol[i, j, k]
    
    @ti.kernel
    def _clear_reflected_flux(self):
        """Clear reflected flux field."""
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            self.swflux_reflected_vol[i, j, k] = 0.0
    
    @ti.kernel
    def _compute_reflected_flux_terrain_kernel(
        self,
        n_surfaces: ti.i32,
        surf_center: ti.template(),
        surf_normal: ti.template(),
        surf_area: ti.template(),
        surf_outgoing: ti.template(),
        is_solid: ti.template(),
        lad: ti.template(),
        has_lad: ti.i32,
        height_offset: ti.i32
    ):
        """
        Compute reflected flux ONLY at terrain-following extraction level.
        
        This is an optimized version that only computes for the cells at
        (i, j, ground_k[i,j] + height_offset_k) instead of all 3D cells.
        
        ~61x faster than full volumetric reflection computation.
        
        Requires init_cumulative_accumulation() to be called first.
        """
        max_dist_sq = 900.0  # 30m max distance
        
        for i, j in ti.ndrange(self.nx, self.ny):
            k = self._ground_k[i, j] + height_offset
            
            # Skip out-of-bounds or solid cells
            if k < 0 or k >= self.nz:
                continue
            if is_solid[i, j, k] == 1:
                continue
            
            cell_x = (ti.cast(i, ti.f32) + 0.5) * self.dx
            cell_y = (ti.cast(j, ti.f32) + 0.5) * self.dy
            cell_z = (ti.cast(k, ti.f32) + 0.5) * self.dz
            
            total_reflected = 0.0
            
            for surf_idx in range(n_surfaces):
                outgoing = surf_outgoing[surf_idx]
                
                if outgoing > 0.1:
                    surf_x = surf_center[surf_idx][0]
                    surf_y = surf_center[surf_idx][1]
                    surf_z = surf_center[surf_idx][2]
                    
                    dx = cell_x - surf_x
                    dy = cell_y - surf_y
                    dz = cell_z - surf_z
                    dist_sq = dx*dx + dy*dy + dz*dz
                    
                    if dist_sq > 0.01 and dist_sq < max_dist_sq:
                        surf_nx = surf_normal[surf_idx][0]
                        surf_ny = surf_normal[surf_idx][1]
                        surf_nz = surf_normal[surf_idx][2]
                        area = surf_area[surf_idx]
                        
                        dist = ti.sqrt(dist_sq)
                        
                        dir_x = dx / dist
                        dir_y = dy / dist
                        dir_z = dz / dist
                        
                        cos_angle = dir_x * surf_nx + dir_y * surf_ny + dir_z * surf_nz
                        
                        if cos_angle > 0.0:
                            trans = self._trace_transmissivity_to_surface(
                                i, j, k, surf_x, surf_y, surf_z,
                                surf_nx, surf_ny, surf_nz,
                                is_solid, lad, has_lad
                            )
                            
                            if trans > 0.0:
                                vf = area * cos_angle / (PI * dist_sq)
                                contribution = outgoing * vf * trans * 0.25
                                total_reflected += contribution
            
            self.swflux_reflected_vol[i, j, k] = total_reflected
    
    def compute_reflected_flux_terrain_following(
        self,
        surfaces,
        surf_outgoing: np.ndarray
    ):
        """
        Compute reflected flux only at terrain-following extraction level.
        
        This is ~61x faster than compute_reflected_flux_vol() because it only
        computes for O(nx*ny) cells instead of O(nx*ny*nz) cells.
        
        Requires init_cumulative_accumulation() to be called first.
        
        Args:
            surfaces: Surfaces object with geometry (center, normal, area)
            surf_outgoing: Array of surface outgoing radiation (W/m²)
        """
        if not self._cumulative_initialized:
            raise RuntimeError("Must call init_cumulative_accumulation() first")
        
        n_surfaces = surfaces.n_surfaces[None]
        if n_surfaces == 0:
            return
        
        # Re-use pre-allocated field if available
        if self._surf_out_field is not None and self._surf_out_max_size >= n_surfaces:
            self._surf_out_field.from_numpy(surf_outgoing[:n_surfaces].astype(np.float32))
            surf_out_field = self._surf_out_field
        else:
            surf_out_field = ti.field(dtype=ti.f32, shape=(n_surfaces,))
            surf_out_field.from_numpy(surf_outgoing[:n_surfaces].astype(np.float32))
        
        has_lad = 1 if self.domain.lad is not None else 0
        
        # Clear reflected flux field before computing
        self._clear_reflected_flux()
        
        self._compute_reflected_flux_terrain_kernel(
            n_surfaces,
            surfaces.center,
            surfaces.normal,
            surfaces.area,
            surf_out_field,
            self.domain.is_solid,
            self.domain.lad,
            has_lad,
            self._height_offset_k
        )
    
    def compute_swflux_vol_with_reflections(
        self,
        sw_direct: float,
        sw_diffuse: float,
        cos_zenith: float,
        sun_direction: Tuple[float, float, float],
        surfaces,
        surf_outgoing: np.ndarray,
        lad: Optional[ti.template] = None
    ):
        """
        Compute volumetric shortwave flux including reflected radiation.
        
        This is a convenience method that combines direct/diffuse computation
        with reflected radiation from surfaces.
        
        Args:
            sw_direct: Direct normal irradiance (W/m²)
            sw_diffuse: Diffuse horizontal irradiance (W/m²)
            cos_zenith: Cosine of solar zenith angle
            sun_direction: Unit vector toward sun (x, y, z)
            surfaces: Surfaces object with geometry
            surf_outgoing: Surface outgoing radiation array (W/m²)
            lad: Optional LAD field for canopy attenuation
        """
        # Compute direct + diffuse
        self.compute_swflux_vol(sw_direct, sw_diffuse, cos_zenith, sun_direction, lad)
        
        # Compute and add reflected
        self.compute_reflected_flux_vol(surfaces, surf_outgoing)
        self._add_reflected_to_total()
    
    def get_swflux_reflected_vol(self) -> np.ndarray:
        """Get volumetric reflected SW flux as numpy array (W/m²)."""
        return self.swflux_reflected_vol.to_numpy()
    
    def get_swflux_direct_vol(self) -> np.ndarray:
        """Get volumetric direct SW flux as numpy array (W/m²)."""
        return self.swflux_direct_vol.to_numpy()
    
    def get_swflux_diffuse_vol(self) -> np.ndarray:
        """Get volumetric diffuse SW flux as numpy array (W/m²)."""
        return self.swflux_diffuse_vol.to_numpy()
    
    def get_flux_components(self) -> dict:
        """
        Get all volumetric flux components as a dictionary.
        
        Returns:
            Dictionary with keys:
            - 'total': Total SW flux (direct + diffuse + reflected if enabled)
            - 'direct': Direct solar component
            - 'diffuse': Diffuse sky component
            - 'reflected': Reflected from surfaces (if computed)
            - 'skyvf': Sky view factor
        """
        return {
            'total': self.swflux_vol.to_numpy(),
            'direct': self.swflux_direct_vol.to_numpy(),
            'diffuse': self.swflux_diffuse_vol.to_numpy(),
            'reflected': self.swflux_reflected_vol.to_numpy(),
            'skyvf': self.skyvf_vol.to_numpy()
        }

    # =========================================================================
    # Cell-to-Surface View Factor (C2S-VF) Matrix Caching
    # =========================================================================
    # These methods pre-compute which surfaces each voxel cell can see,
    # making repeated reflected flux calculations O(nnz) instead of O(N*M).
    
    def compute_c2s_matrix(
        self,
        surfaces,
        is_solid,
        lad=None,
        min_vf_threshold: float = 1e-6,
        progress_report: bool = False
    ):
        """
        Pre-compute Cell-to-Surface View Factor matrix for fast reflections.
        
        This is O(N_cells * N_surfaces) but only needs to be done once for
        fixed geometry. Subsequent calls to compute_reflected_flux_vol_cached()
        become O(nnz) instead of O(N*M).
        
        Call this before running multi-timestep simulations with reflections.
        
        Args:
            surfaces: Surfaces object with center, normal, area fields
            is_solid: 3D solid obstacle field
            lad: Optional LAD field for vegetation attenuation
            min_vf_threshold: Minimum view factor to store (sparsity threshold)
            progress_report: Print progress messages
        """
        if self._c2s_matrix_cached:
            if progress_report:
                print("C2S-VF matrix already cached, skipping recomputation.")
            return
        
        n_surfaces = surfaces.n_surfaces[None]
        n_cells = self.nx * self.ny * self.nz
        
        if progress_report:
            print(f"Pre-computing C2S-VF matrix: {n_cells:,} cells × {n_surfaces:,} surfaces")
            print("  This is O(N*M) but only runs once for fixed geometry.")
        
        # Allocate sparse COO arrays if not already done
        if self._c2s_cell_idx is None:
            self._c2s_cell_idx = ti.field(dtype=ti.i32, shape=(self._max_c2s_entries,))
            self._c2s_surf_idx = ti.field(dtype=ti.i32, shape=(self._max_c2s_entries,))
            self._c2s_vf = ti.field(dtype=ti.f32, shape=(self._max_c2s_entries,))
            self._c2s_count = ti.field(dtype=ti.i32, shape=())
        
        has_lad = 1 if lad is not None else 0
        
        # Compute the matrix
        self._c2s_count[None] = 0
        self._compute_c2s_matrix_kernel(
            n_surfaces,
            surfaces.center,
            surfaces.normal,
            surfaces.area,
            is_solid,
            lad if lad is not None else self.domain.lad,  # Fallback to domain LAD
            has_lad,
            min_vf_threshold
        )
        
        computed_nnz = int(self._c2s_count[None])
        if computed_nnz > self._max_c2s_entries:
            print(f"Warning: C2S-VF matrix truncated! {computed_nnz:,} > {self._max_c2s_entries:,}")
            print("  Consider increasing _max_c2s_entries.")
            self._c2s_nnz = self._max_c2s_entries
        else:
            self._c2s_nnz = computed_nnz
        
        self._c2s_matrix_cached = True
        
        sparsity_pct = self._c2s_nnz / (n_cells * n_surfaces) * 100 if n_surfaces > 0 else 0
        if progress_report:
            print(f"  C2S-VF matrix computed: {self._c2s_nnz:,} non-zero entries")
            print(f"  Sparsity: {sparsity_pct:.4f}% of full matrix")
            speedup = (n_cells * n_surfaces) / max(1, self._c2s_nnz)
            print(f"  Speedup factor: ~{speedup:.0f}x per timestep")
    
    @ti.kernel
    def _compute_c2s_matrix_kernel(
        self,
        n_surfaces: ti.i32,
        surf_center: ti.template(),
        surf_normal: ti.template(),
        surf_area: ti.template(),
        is_solid: ti.template(),
        lad: ti.template(),
        has_lad: ti.i32,
        min_threshold: ti.f32
    ):
        """
        Compute C2S-VF matrix entries.
        
        For each (cell, surface) pair, compute the view factor including
        geometry and transmissivity. Store if above threshold.
        """
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            # Skip solid cells
            if is_solid[i, j, k] == 1:
                continue
            
            cell_idx = i * (self.ny * self.nz) + j * self.nz + k
            cell_x = (ti.cast(i, ti.f32) + 0.5) * self.dx
            cell_y = (ti.cast(j, ti.f32) + 0.5) * self.dy
            cell_z = (ti.cast(k, ti.f32) + 0.5) * self.dz
            
            for surf_idx in range(n_surfaces):
                surf_x = surf_center[surf_idx][0]
                surf_y = surf_center[surf_idx][1]
                surf_z = surf_center[surf_idx][2]
                surf_nx = surf_normal[surf_idx][0]
                surf_ny = surf_normal[surf_idx][1]
                surf_nz = surf_normal[surf_idx][2]
                area = surf_area[surf_idx]
                
                # Distance to surface
                dx = cell_x - surf_x
                dy = cell_y - surf_y
                dz = cell_z - surf_z
                dist_sq = dx*dx + dy*dy + dz*dz
                
                if dist_sq > 0.01:  # Avoid numerical issues
                    dist = ti.sqrt(dist_sq)
                    
                    # Direction from surface to cell (normalized)
                    dir_x = dx / dist
                    dir_y = dy / dist
                    dir_z = dz / dist
                    
                    # Cosine of angle between normal and direction
                    cos_angle = dir_x * surf_nx + dir_y * surf_ny + dir_z * surf_nz
                    
                    if cos_angle > 0.0:  # Surface faces the cell
                        # Get transmissivity
                        trans = self._trace_transmissivity_to_surface(
                            i, j, k, surf_x, surf_y, surf_z,
                            surf_nx, surf_ny, surf_nz,
                            is_solid, lad, has_lad
                        )
                        
                        if trans > 0.0:
                            # View factor: (A * cos_θ) / (π * d²) * 0.25 for sphere
                            vf = area * cos_angle / (PI * dist_sq) * trans * 0.25
                            
                            if vf > min_threshold:
                                idx = ti.atomic_add(self._c2s_count[None], 1)
                                if idx < self._max_c2s_entries:
                                    self._c2s_cell_idx[idx] = cell_idx
                                    self._c2s_surf_idx[idx] = surf_idx
                                    self._c2s_vf[idx] = vf
    
    def compute_reflected_flux_vol_cached(
        self,
        surf_outgoing: np.ndarray,
        progress_report: bool = False
    ):
        """
        Compute volumetric reflected flux using cached C2S-VF matrix.
        
        This is O(nnz) instead of O(N_cells * N_surfaces), providing
        massive speedup for repeated calls with different surface radiation.
        
        Must call compute_c2s_matrix() first.
        
        Args:
            surf_outgoing: Array of surface outgoing radiation (W/m²)
            progress_report: Print progress messages
        """
        if not self._c2s_matrix_cached:
            raise RuntimeError("C2S-VF matrix not computed. Call compute_c2s_matrix() first.")
        
        n_surfaces = len(surf_outgoing)
        
        # Allocate or resize surface outgoing field if needed
        if self._surf_out_field is None or self._surf_out_max_size < n_surfaces:
            self._surf_out_field = ti.field(dtype=ti.f32, shape=(n_surfaces,))
            self._surf_out_max_size = n_surfaces
        
        # Copy outgoing radiation to Taichi field
        self._surf_out_field.from_numpy(surf_outgoing.astype(np.float32))
        
        # Clear reflected flux field
        self._clear_reflected_flux()
        
        # Use sparse matrix-vector multiply
        self._apply_c2s_matrix_kernel(self._c2s_nnz)
        
        if progress_report:
            print(f"Computed volumetric reflected flux using cached C2S-VF ({self._c2s_nnz:,} entries)")
    
    @ti.kernel
    def _apply_c2s_matrix_kernel(self, c2s_nnz: ti.i32):
        """
        Apply C2S-VF matrix to compute reflected flux.
        
        flux[cell] = Σ (vf[cell, surf] * outgoing[surf])
        
        Uses atomic operations for parallel accumulation.
        """
        for idx in range(c2s_nnz):
            cell_idx = self._c2s_cell_idx[idx]
            surf_idx = self._c2s_surf_idx[idx]
            vf = self._c2s_vf[idx]
            
            outgoing = self._surf_out_field[surf_idx]
            
            if outgoing > 0.1:  # Threshold for negligible contributions
                # Reconstruct 3D indices from linear index
                # cell_idx = i * (ny * nz) + j * nz + k
                tmp = cell_idx
                k = tmp % self.nz
                tmp //= self.nz
                j = tmp % self.ny
                i = tmp // self.ny
                
                ti.atomic_add(self.swflux_reflected_vol[i, j, k], outgoing * vf)
    
    def invalidate_c2s_cache(self):
        """
        Invalidate the cached C2S-VF matrix.
        
        Call this if geometry (buildings, terrain, vegetation) changes.
        """
        self._c2s_matrix_cached = False
        self._c2s_nnz = 0
    
    @property
    def c2s_matrix_cached(self) -> bool:
        """Check if C2S-VF matrix is currently cached."""
        return self._c2s_matrix_cached
    
    @property
    def c2s_matrix_entries(self) -> int:
        """Get number of non-zero entries in cached C2S-VF matrix."""
        return self._c2s_nnz
    
    def compute_swflux_vol_with_reflections_cached(
        self,
        sw_direct: float,
        sw_diffuse: float,
        cos_zenith: float,
        sun_direction: Tuple[float, float, float],
        surf_outgoing: np.ndarray,
        lad=None
    ):
        """
        Compute volumetric shortwave flux with reflections using cached matrix.
        
        This is the fast path for multi-timestep simulations. Must call
        compute_c2s_matrix() once before using this method.
        
        Args:
            sw_direct: Direct normal irradiance (W/m²)
            sw_diffuse: Diffuse horizontal irradiance (W/m²)
            cos_zenith: Cosine of solar zenith angle
            sun_direction: Unit vector toward sun (x, y, z)
            surf_outgoing: Surface outgoing radiation array (W/m²)
            lad: Optional LAD field for canopy attenuation
        """
        # Compute direct + diffuse
        self.compute_swflux_vol(sw_direct, sw_diffuse, cos_zenith, sun_direction, lad)
        
        # Compute reflected using cached matrix
        self.compute_reflected_flux_vol_cached(surf_outgoing)
        self._add_reflected_to_total()

    # =========================================================================
    # Cumulative Terrain-Following Accumulation (GPU-Optimized)
    # =========================================================================
    # These methods enable efficient cumulative volumetric simulation by
    # accumulating terrain-following slices directly on GPU, avoiding the
    # expensive GPU-to-CPU transfer of full 3D arrays for each timestep/patch.
    
    def init_cumulative_accumulation(
        self,
        ground_k: np.ndarray,
        height_offset_k: int,
        is_solid: np.ndarray
    ):
        """
        Initialize GPU-side cumulative terrain-following accumulation.
        
        Must be called before using accumulate_terrain_following_slice_gpu().
        
        Args:
            ground_k: 2D array (nx, ny) of ground k-levels. -1 means no valid ground.
            height_offset_k: Number of cells above ground for extraction.
            is_solid: 3D array (nx, ny, nz) of solid flags.
        """
        # Copy ground_k to GPU
        self._ground_k.from_numpy(ground_k.astype(np.int32))
        self._height_offset_k = height_offset_k
        
        # Clear cumulative map
        self._clear_cumulative_map()
        
        self._cumulative_initialized = True
    
    @ti.kernel
    def _clear_cumulative_map(self):
        """Clear the cumulative terrain-following map."""
        for i, j in ti.ndrange(self.nx, self.ny):
            self._cumulative_map[i, j] = 0.0
    
    @ti.kernel
    def _accumulate_terrain_slice_kernel(
        self,
        height_offset_k: ti.i32,
        weight: ti.f64,
        is_solid: ti.template()
    ):
        """
        Accumulate terrain-following slice from swflux_vol directly on GPU.
        
        For each (i,j), extracts swflux_vol[i,j,k_extract] * weight
        where k_extract = ground_k[i,j] + height_offset_k.
        
        Args:
            height_offset_k: Number of cells above ground for extraction.
            weight: Multiplier for values before accumulating (e.g., time_step_hours).
            is_solid: 3D solid field for masking.
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            gk = self._ground_k[i, j]
            if gk < 0:
                continue  # No valid ground
            
            k_extract = gk + height_offset_k
            if k_extract >= self.nz:
                continue  # Out of bounds
            
            # Skip if extraction point is inside solid
            if is_solid[i, j, k_extract] == 1:
                continue
            
            # Accumulate the flux value (atomic add for thread safety)
            flux_val = ti.cast(self.swflux_vol[i, j, k_extract], ti.f64)
            ti.atomic_add(self._cumulative_map[i, j], flux_val * weight)
    
    @ti.kernel
    def _accumulate_terrain_slice_from_svf_kernel(
        self,
        height_offset_k: ti.i32,
        weight: ti.f64,
        is_solid: ti.template()
    ):
        """
        Accumulate terrain-following slice from skyvf_vol directly on GPU.
        
        For each (i,j), extracts skyvf_vol[i,j,k_extract] * weight
        where k_extract = ground_k[i,j] + height_offset_k.
        
        Args:
            height_offset_k: Number of cells above ground for extraction.
            weight: Multiplier (e.g., total_dhi for diffuse contribution).
            is_solid: 3D solid field for masking.
        """
        for i, j in ti.ndrange(self.nx, self.ny):
            gk = self._ground_k[i, j]
            if gk < 0:
                continue
            
            k_extract = gk + height_offset_k
            if k_extract >= self.nz:
                continue
            
            if is_solid[i, j, k_extract] == 1:
                continue
            
            svf_val = ti.cast(self.skyvf_vol[i, j, k_extract], ti.f64)
            ti.atomic_add(self._cumulative_map[i, j], svf_val * weight)
    
    def accumulate_terrain_following_slice_gpu(
        self,
        weight: float = 1.0
    ):
        """
        Accumulate current swflux_vol terrain-following slice to cumulative map on GPU.
        
        This is the fast path for cumulative simulations. Must call 
        init_cumulative_accumulation() first.
        
        Args:
            weight: Multiplier for values (e.g., time_step_hours for Wh conversion).
        """
        if not self._cumulative_initialized:
            raise RuntimeError("Cumulative accumulation not initialized. "
                             "Call init_cumulative_accumulation() first.")
        
        self._accumulate_terrain_slice_kernel(
            self._height_offset_k,
            float(weight),
            self.domain.is_solid
        )
    
    def accumulate_svf_diffuse_gpu(
        self,
        total_dhi: float
    ):
        """
        Accumulate diffuse contribution using SVF field directly on GPU.
        
        Args:
            total_dhi: Total cumulative diffuse horizontal irradiance (Wh/m²).
        """
        if not self._cumulative_initialized:
            raise RuntimeError("Cumulative accumulation not initialized. "
                             "Call init_cumulative_accumulation() first.")
        
        self._accumulate_terrain_slice_from_svf_kernel(
            self._height_offset_k,
            float(total_dhi),
            self.domain.is_solid
        )
    
    def get_cumulative_map(self) -> np.ndarray:
        """
        Get the accumulated terrain-following cumulative map.
        
        Returns:
            2D numpy array (nx, ny) of cumulative irradiance values.
        """
        if not self._cumulative_initialized:
            raise RuntimeError("Cumulative accumulation not initialized.")
        
        return self._cumulative_map.to_numpy()
    
    def finalize_cumulative_map(self, apply_nan_mask: bool = True) -> np.ndarray:
        """
        Get final cumulative map with optional NaN masking for invalid cells.
        
        Args:
            apply_nan_mask: If True, set cells with no valid ground or inside
                           solid to NaN.
        
        Returns:
            2D numpy array (nx, ny) of cumulative irradiance values.
        """
        if not self._cumulative_initialized:
            raise RuntimeError("Cumulative accumulation not initialized.")
        
        result = self._cumulative_map.to_numpy()
        
        if apply_nan_mask:
            ground_k_np = self._ground_k.to_numpy()
            is_solid_np = self.domain.is_solid.to_numpy()
            
            for i in range(self.nx):
                for j in range(self.ny):
                    gk = ground_k_np[i, j]
                    if gk < 0:
                        result[i, j] = np.nan
                        continue
                    k_extract = gk + self._height_offset_k
                    if k_extract >= self.nz:
                        result[i, j] = np.nan
                        continue
                    if is_solid_np[i, j, k_extract] == 1:
                        result[i, j] = np.nan
        
        return result
    
    def reset_cumulative_accumulation(self):
        """Reset the cumulative map to zero without reinitializing ground_k."""
        if self._cumulative_initialized:
            self._clear_cumulative_map()
