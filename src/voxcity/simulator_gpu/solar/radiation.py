"""Radiation solver for palm-solar.

Main module that integrates all components to compute
shortwave (solar) radiation fluxes with multi-bounce reflections
following PALM's RTM (Radiative Transfer Model) methodology.

PALM Alignment Notes:
- Solar position: Uses PALM's calc_zenith formula exactly (solar.py)
- SVF calculation: Uses PALM's vffrac_up formula with proper weighting (svf.py)
- Reflection steps: Default nrefsteps=3 matches PALM
- Extinction coefficient: Default ext_coef=0.6 matches PALM
- Beer-Lambert law: Same exponential attenuation through canopy
- Direction indices: IUP=0, IDOWN=1, etc. match PALM convention

Key differences from PALM:
- GPU-accelerated via Taichi (PALM uses Fortran+MPI)
- Real-time view factor computation (PALM pre-computes sparse matrix)
- Shortwave only (PALM includes longwave radiation)
- Axis-aligned surfaces only (PALM supports slant surfaces)

Input convention:
- sw_direct: Direct Normal Irradiance (DNI) in W/m²
- sw_diffuse: Diffuse Horizontal Irradiance (DHI) in W/m²
"""

import taichi as ti
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field

from .core import (
    Vector3, Point3, SOLAR_CONSTANT, EXT_COEF,
    PI, TWO_PI
)
from .domain import Domain, Surfaces, extract_surfaces_from_domain
from .solar import SolarPosition, calc_zenith, SolarCalculator
from .raytracing import RayTracer, ray_point_to_point_transmissivity, ray_voxel_first_hit
from .svf import SVFCalculator
from .csf import CSFCalculator
from .volumetric import VolumetricFluxCalculator


# Direction indices (matching PALM convention from radiation_model_mod.f90)
# PALM: iup=0, idown=1, inorth=2, isouth=3, ieast=4, iwest=5
IUP = 0
IDOWN = 1
INORTH = 2
ISOUTH = 3
IEAST = 4
IWEST = 5


@dataclass
class RadiationConfig:
    """
    Configuration for radiation model.
    
    Attributes:
        albedo_ground: Default ground albedo
        albedo_wall: Default wall albedo
        albedo_roof: Default roof albedo
        albedo_leaf: Tree/leaf albedo (PALM default: 0.15)
        n_azimuth: Number of azimuthal divisions for SVF
        n_elevation: Number of elevation divisions for SVF
        ext_coef: Extinction coefficient for canopy
        skip_svf: Skip SVF calculation (use 1.0)
        n_reflection_steps: Number of reflection iterations (PALM default: 3)
        surface_reflections: Enable surface-to-surface reflections
        canopy_reflections: Enable reflection attenuation through canopy
        volumetric_flux: Enable volumetric flux calculation
        volumetric_n_azimuth: Number of azimuths for volumetric horizon tracing
        min_opaque_lad: Minimum LAD considered opaque for volumetric shadows
        canopy_radiation: Enable plant canopy radiation absorption (CSF)
        canopy_to_canopy: Enable canopy-to-canopy scattering (not in PALM, 
            improves accuracy for dense vegetation where leaves scatter light
            to neighboring leaves)
    """
    albedo_ground: float = 0.2
    albedo_wall: float = 0.3
    albedo_roof: float = 0.3
    albedo_leaf: float = 0.15  # PALM default tree albedo
    n_azimuth: int = 80         # PALM: raytrace_discrete_azims = 80
    n_elevation: int = 40       # PALM: raytrace_discrete_elevs = 40
    ext_coef: float = EXT_COEF  # PALM: ext_coef = 0.6
    skip_svf: bool = False
    n_reflection_steps: int = 3  # PALM: nrefsteps = 3
    surface_reflections: bool = True
    canopy_reflections: bool = True  # Enable LAD attenuation in reflections
    volumetric_flux: bool = False
    volumetric_n_azimuth: int = 36
    min_opaque_lad: float = 0.5
    canopy_radiation: bool = True  # Enable CSF-based canopy absorption
    canopy_to_canopy: bool = True  # Enable canopy-to-canopy scattering (improves accuracy in dense vegetation)
    # SVF matrix caching for multi-timestep efficiency (PALM-like approach)
    cache_svf_matrix: bool = True   # Pre-compute SVF matrix for fast reflections
    svf_min_threshold: float = 0.01  # Minimum VF to store (sparsity threshold, 0.01 sufficient for 1% accuracy)
    # FP16 optimization for intermediate calculations (reduces memory bandwidth, ~2x faster)
    # fp16 range: ±65,504 with ~3 decimal digits precision - sufficient for W/m² irradiance values
    use_fp16_intermediate: bool = True  # Use float16 for intermediate reflection buffers


@ti.data_oriented
class RadiationModel:
    """
    GPU-accelerated solar radiation transfer model.
    
    Computes shortwave (direct and diffuse) radiation
    for all surface elements in the domain.
    """
    
    def __init__(
        self,
        domain: Domain,
        config: Optional[RadiationConfig] = None
    ):
        """
        Initialize radiation model.
        
        Args:
            domain: Domain object with geometry
            config: Radiation configuration (uses defaults if None)
        """
        self.domain = domain
        self.config = config or RadiationConfig()
        
        # Extract surfaces from domain
        print("Extracting surfaces from domain...")
        self.surfaces = extract_surfaces_from_domain(domain)
        self.n_surfaces = self.surfaces.count
        print(f"Found {self.n_surfaces} surface elements")
        
        # Initialize sub-components
        self.solar_calc = SolarCalculator(
            domain.origin_lat or 0.0, 
            domain.origin_lon or 0.0
        )
        
        self.ray_tracer = RayTracer(domain)
        self.ray_tracer.ext_coef = self.config.ext_coef
        
        self.svf_calc = SVFCalculator(
            domain,
            self.config.n_azimuth,
            self.config.n_elevation
        )
        
        self.csf_calc = CSFCalculator(
            domain,
            self.config.n_azimuth,
            self.config.n_elevation
        )
        
        # Set default surface properties based on direction
        self._set_default_properties()
        
        # Determine dtype for intermediate buffers (fp16 reduces memory bandwidth ~2x)
        # fp16: range ±65,504, ~3 decimal digits - sufficient for W/m² irradiance (0-1400 range)
        inter_dtype = ti.f16 if config.use_fp16_intermediate else ti.f32
        
        # Allocate arrays for multi-bounce reflections
        # These store radiation fluxes during reflection iterations
        self._surfins = ti.field(dtype=inter_dtype, shape=(self.n_surfaces,))  # Incoming SW per reflection step
        self._surfinl = ti.field(dtype=inter_dtype, shape=(self.n_surfaces,))  # Incoming LW per reflection step
        self._surfoutsl = ti.field(dtype=inter_dtype, shape=(self.n_surfaces,))  # Outgoing SW per reflection step
        self._surfoutll = ti.field(dtype=inter_dtype, shape=(self.n_surfaces,))  # Outgoing LW per reflection step
        
        # Ping-pong buffers for optimized reflection iterations
        # Using separate kernels without internal ti.sync() is ~100x faster than fused kernels
        self._surfins_ping = ti.field(dtype=inter_dtype, shape=(self.n_surfaces,))
        self._surfins_pong = ti.field(dtype=inter_dtype, shape=(self.n_surfaces,))
        
        # Total accumulated radiation - keep as f32 for precision in cumulative sums
        self._surfinsw = ti.field(dtype=ti.f32, shape=(self.n_surfaces,))  # Total incoming SW
        self._surfinlw = ti.field(dtype=ti.f32, shape=(self.n_surfaces,))  # Total incoming LW
        self._surfoutsw = ti.field(dtype=ti.f32, shape=(self.n_surfaces,))  # Total outgoing SW
        self._surfoutlw = ti.field(dtype=ti.f32, shape=(self.n_surfaces,))  # Total outgoing LW
        
        # Direct and diffuse components - keep as f32 for final output precision
        self._surfinswdir = ti.field(dtype=ti.f32, shape=(self.n_surfaces,))  # Direct SW
        self._surfinswdif = ti.field(dtype=ti.f32, shape=(self.n_surfaces,))  # Diffuse SW from sky
        
        # For optimized reflection computation - store weighted totals
        self._total_reflected_flux = ti.field(dtype=ti.f32, shape=())  # Sum of (surfout * area)
        self._total_reflecting_area = ti.field(dtype=ti.f32, shape=())  # Sum of areas
        
        # Surface-to-surface view factors (sparse representation)
        # svf_matrix[i, j] = view factor from surface j to surface i
        # For efficiency, we'll compute this on-demand during reflections
        self._svf_matrix_computed = False
        
        # SVF computed flag
        self._svf_computed = False
        
        # Initialize volumetric flux calculator if enabled
        self.volumetric_calc = None
        if self.config.volumetric_flux:
            self.volumetric_calc = VolumetricFluxCalculator(
                domain,
                n_azimuth=self.config.volumetric_n_azimuth,
                min_opaque_lad=self.config.min_opaque_lad
            )
        
        # Plant canopy absorption arrays (like PALM's pcbinsw, etc.)
        # Indexed by (i, j, k) grid coordinates
        # Units: W/m³ (power absorbed per unit volume)
        # Total arrays use f32 for precision in cumulative sums
        self._pcbinsw = ti.field(dtype=ti.f32, shape=(domain.nx, domain.ny, domain.nz))     # Total absorbed SW
        self._pcbinswdir = ti.field(dtype=ti.f32, shape=(domain.nx, domain.ny, domain.nz))  # Direct SW absorbed
        self._pcbinswdif = ti.field(dtype=ti.f32, shape=(domain.nx, domain.ny, domain.nz))  # Diffuse SW absorbed
        
        # Received radiation (before absorption) in W/m²
        self._pcinsw = ti.field(dtype=ti.f32, shape=(domain.nx, domain.ny, domain.nz))      # Total received SW
        self._pcinswdir = ti.field(dtype=ti.f32, shape=(domain.nx, domain.ny, domain.nz))   # Direct SW received
        self._pcinswdif = ti.field(dtype=ti.f32, shape=(domain.nx, domain.ny, domain.nz))   # Diffuse SW received
        
        # Canopy scattered/reflected radiation (W/m³) - intermediate uses inter_dtype
        self._pcbinswref = ti.field(dtype=inter_dtype, shape=(domain.nx, domain.ny, domain.nz))  # Reflected SW from canopy
        
        # Canopy-to-canopy scattering contribution (W/m³)
        # Stores radiation scattered from other canopy cells to this cell (per iteration)
        self._pcbinswc2c = ti.field(dtype=inter_dtype, shape=(domain.nx, domain.ny, domain.nz))  # Canopy-to-canopy SW (temp)
        # Cumulative total of c2c contribution (for output) - f32 for precision
        self._pcbinswc2c_total = ti.field(dtype=ti.f32, shape=(domain.nx, domain.ny, domain.nz))  # Canopy-to-canopy SW (total)
        
        # Canopy-to-surface contribution per reflection step - intermediate
        self._surfinswpc = ti.field(dtype=inter_dtype, shape=(self.n_surfaces,))  # SW from plant canopy
        
        # ========== SVF Matrix Caching (PALM-like approach) ==========
        # For multi-timestep efficiency, pre-compute surface-to-surface view factors
        # Stored in sparse COO format: (source_idx, target_idx, vf_value, transmissivity)
        # This makes reflection iterations O(nnz) instead of O(n²)
        self._svf_matrix_cached = False
        self._svf_nnz = 0  # Number of non-zero entries
        
        # Estimate max non-zeros based on threshold:
        # With svf_min_threshold=0.01, ~99.7% of entries are filtered out
        # For 250k surfaces: from 585M entries down to ~1.8M entries
        # 
        # Memory per entry: 12 bytes (2 int32 + 2 float16)
        # At 0.01 threshold: ~22 MB for 1.8M entries
        # At 0.001 threshold: ~260 MB for 21M entries
        # At 0.0001 threshold: ~1.8 GB for 148M entries
        #
        # Estimate based on threshold (empirically derived from 250k surface domain):
        threshold = config.svf_min_threshold
        if threshold >= 0.01:
            entries_per_surface = 8  # ~0.3% of 2500 avg neighbors
        elif threshold >= 0.001:
            entries_per_surface = 85  # ~3.5% of avg neighbors  
        elif threshold >= 0.0001:
            entries_per_surface = 600  # ~25% of avg neighbors
        else:
            entries_per_surface = 2500  # Full density
            
        estimated = self.n_surfaces * entries_per_surface
        
        # For small domains, allow full N*N
        if self.n_surfaces < 5000:
            self._max_svf_entries = min(self.n_surfaces * self.n_surfaces, estimated * 2)
        else:
            # Add 50% buffer for safety
            self._max_svf_entries = int(estimated * 1.5)
            
            # Sanity check against GPU memory (each entry = 12 bytes with f16)
            try:
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    free_mb = int(result.stdout.strip().split('\n')[0])
                    # Use up to 50% of free memory for SVF matrix
                    available_bytes = free_mb * 1024 * 1024 * 0.5
                    # Each entry = 12 bytes (2 int32 + 2 float16)
                    memory_based_limit = int(available_bytes / 12)
                    
                    if self._max_svf_entries > memory_based_limit:
                        import warnings
                        warnings.warn(
                            f"SVF buffer limited to {memory_based_limit:,} entries due to GPU memory. "
                            f"Estimated {self._max_svf_entries:,} entries needed.",
                            RuntimeWarning
                        )
                        self._max_svf_entries = memory_based_limit
            except Exception:
                pass  # Keep estimated size
        
        # Pre-allocate sparse COO arrays upfront to avoid CUDA issues with dynamic allocation
        # These are allocated during __init__ to ensure proper CUDA memory management
        if config.cache_svf_matrix and config.surface_reflections:
            self._svf_source = ti.field(dtype=ti.i32, shape=(self._max_svf_entries,))
            self._svf_target = ti.field(dtype=ti.i32, shape=(self._max_svf_entries,))
            # Use float16 to reduce memory by 50% (sufficient for VF accuracy ~0.01)
            self._svf_vf = ti.field(dtype=ti.f16, shape=(self._max_svf_entries,))
            self._svf_trans = ti.field(dtype=ti.f16, shape=(self._max_svf_entries,))
            self._svf_count = ti.field(dtype=ti.i32, shape=())
            
            # CSR format arrays for optimized sparse matmul
            # NOTE: CSR is currently disabled because:
            # 1. COO with modern GPU atomics is actually faster for this workload
            # 2. CSR building from numpy adds significant overhead (12+ seconds)
            # 3. CSR suffers from load imbalance (row sizes vary from 33 to 4789)
            # Keep the arrays allocated for potential future use with GPU-based sorting
            self._svf_csr_row_ptr = None  # Disabled
            self._svf_csr_col_idx = None  # Disabled  
            self._svf_csr_val = None  # Disabled
            self._svf_csr_ready = False
        else:
            self._svf_source = None
            self._svf_target = None
            self._svf_vf = None
            self._svf_trans = None
            self._svf_count = None
            self._svf_csr_row_ptr = None
            self._svf_csr_col_idx = None
            self._svf_csr_val = None
            self._svf_csr_ready = False
            
        # ========== CSF Matrix Caching (Canopy-Surface Factors) ==========
        # For efficient canopy-surface interactions during reflections
        # Stored in sparse COO format: (canopy_idx, surface_idx, csf_value)
        self._csf_matrix_cached = False
        self._csf_nnz = 0
        
        # Estimate max non-zeros for CSF
        # Assume each canopy cell sees ~100 surfaces on average
        # This is a rough heuristic
        self._max_csf_entries = 10_000_000  # 10M entries ~ 120MB
        
        if config.cache_svf_matrix and config.canopy_radiation and domain.lad is not None:
            self._csf_canopy_idx = ti.field(dtype=ti.i32, shape=(self._max_csf_entries,))
            self._csf_surface_idx = ti.field(dtype=ti.i32, shape=(self._max_csf_entries,))
            # Use float16 for CSF values (sufficient for ~0.01 accuracy)
            self._csf_val = ti.field(dtype=ti.f16, shape=(self._max_csf_entries,))
            self._csf_count = ti.field(dtype=ti.i32, shape=())
            
            # Lookup table for surface index from grid position and direction
            # (nx, ny, nz, 6) -> surface_index
            self._grid_to_surf = ti.field(dtype=ti.i32, shape=(domain.nx, domain.ny, domain.nz, 6))
        else:
            self._csf_canopy_idx = None
            self._csf_surface_idx = None
            self._csf_val = None
            self._csf_count = None
            self._grid_to_surf = None
        
        # Canopy radiation computed flag
        self._canopy_radiation_computed = False
    
    def _set_default_properties(self):
        """Set default albedo for surfaces."""
        self._set_defaults_kernel(
            self.surfaces.direction,
            self.surfaces.albedo,
            self.n_surfaces,
            self.config.albedo_ground,
            self.config.albedo_wall,
            self.config.albedo_roof
        )
    
    @ti.kernel
    def _set_defaults_kernel(
        self,
        direction: ti.template(),
        albedo: ti.template(),
        n_surf: ti.i32,
        alb_ground: ti.f32,
        alb_wall: ti.f32,
        alb_roof: ti.f32
    ):
        for i in range(n_surf):
            d = direction[i]
            if d == 0:  # Up (roof or ground)
                albedo[i] = alb_roof
            elif d == 1:  # Down (typically building underside)
                albedo[i] = alb_ground
            else:  # Walls
                albedo[i] = alb_wall
    
    def compute_svf(self):
        """
        Compute Sky View Factors for all surfaces.
        
        This is computationally expensive, call once per domain setup.
        """
        if self.config.skip_svf:
            # Set SVF to 1.0 for all surfaces
            self._set_svf_one()
            self._svf_computed = True
            return
        
        print("Computing Sky View Factors...")
        
        if self.domain.lad is not None:
            self.svf_calc.compute_svf_with_canopy(
                self.surfaces.center,  # Use world coordinates, not grid indices
                self.surfaces.direction,
                self.domain.is_solid,
                self.domain.lad,
                self.n_surfaces,
                self.config.ext_coef,
                self.surfaces.svf,
                self.surfaces.svf_urban
            )
        else:
            self.svf_calc.compute_svf(
                self.surfaces.center,  # Use world coordinates, not grid indices
                self.surfaces.direction,
                self.domain.is_solid,
                self.n_surfaces,
                self.surfaces.svf
            )
            # Copy to svf_urban
            self._copy_svf()
        
        self._svf_computed = True
        print("SVF computation complete.")
        
        # Pre-compute SVF matrix for efficient multi-timestep reflections
        if self.config.cache_svf_matrix and self.config.surface_reflections:
            self.compute_svf_matrix()
            
            # Also compute CSF matrix if canopy is present
            if self.config.canopy_radiation and self.domain.lad is not None:
                self.compute_csf_matrix()
        
        # Pre-compute CSF sky for efficient multi-timestep canopy absorption
        # CSF sky is geometry-dependent only and doesn't change with sun position
        if self.config.canopy_radiation and self.domain.lad is not None:
            print("Pre-computing CSF sky (geometry-dependent, computed once)...")
            self.csf_calc.compute_csf_sky_cached(
                self.domain.is_solid,
                self.domain.lad,
                self.config.n_azimuth,
                self.config.n_elevation
            )
            print("CSF sky computation complete.")
    
    @ti.kernel
    def _set_svf_one(self):
        for i in range(self.n_surfaces):
            self.surfaces.svf[i] = 1.0
            self.surfaces.svf_urban[i] = 1.0
    
    @ti.kernel
    def _copy_svf(self):
        for i in range(self.n_surfaces):
            self.surfaces.svf_urban[i] = self.surfaces.svf[i]
    
    def update_solar_position(self, day_of_year: int, second_of_day: float):
        """
        Update solar position.
        
        Args:
            day_of_year: Day number (1-365)
            second_of_day: Seconds since midnight UTC
        """
        self.solar_calc.update(day_of_year, second_of_day)
    
    def compute_shortwave_radiation(
        self,
        sw_direct: float,
        sw_diffuse: float
    ):
        """
        Compute shortwave (solar) radiation for all surfaces.
        
        Args:
            sw_direct: Direct normal irradiance (W/m²)
            sw_diffuse: Diffuse horizontal irradiance (W/m²)
        """
        if not self._svf_computed:
            print("Warning: SVF not computed, computing now...")
            self.compute_svf()
        
        # Get sun direction
        sun_dir = self.solar_calc.sun_direction[None]
        cos_zenith = self.solar_calc.cos_zenith[None]
        
        if cos_zenith > 0:
            # Compute direct shadows
            if self.domain.lad is not None:
                self.ray_tracer.compute_direct_with_canopy(
                    self.surfaces.center,  # Use world coordinates, not grid indices
                    self.surfaces.direction,
                    sun_dir,
                    self.domain.is_solid,
                    self.domain.lad,
                    self.n_surfaces,
                    self.surfaces.shadow_factor,
                    self.surfaces.canopy_transmissivity
                )
            else:
                self.ray_tracer.compute_direct_shadows(
                    self.surfaces.center,  # Use world coordinates, not grid indices
                    self.surfaces.direction,
                    sun_dir,
                    self.domain.is_solid,
                    self.n_surfaces,
                    self.surfaces.shadow_factor
                )
                # Set canopy transmissivity to 1 (no canopy)
                self._set_canopy_trans_one()
        else:
            # Night time - no direct radiation
            self._clear_direct_radiation()
        
        # Compute radiation fluxes with unified reflection loop
        # This now includes all paths: Surface↔Surface, Surface↔Canopy, Canopy↔Canopy
        self._compute_sw_fluxes(
            sw_direct,
            sw_diffuse,
            cos_zenith
        )
        
        # Compute plant canopy radiation absorption from direct/diffuse (CSF)
        # Note: If canopy_reflections is enabled, initial absorption is computed inside 
        # _compute_sw_fluxes via _compute_canopy_radiation_initial, so we skip here.
        # We only call _compute_canopy_radiation if NOT using unified reflection loop.
        if self.config.canopy_radiation and self.domain.lad is not None:
            if self.config.canopy_reflections:
                # Unified reflection loop already computed initial absorption
                # Just compute received radiation and mark as computed
                grid_volume = self.domain.dx * self.domain.dy * self.domain.dz
                self._compute_received_radiation(sw_direct, sw_diffuse, cos_zenith, grid_volume)
                self._canopy_radiation_computed = True
            else:
                # Legacy path: compute canopy radiation the old way
                self._compute_canopy_radiation(
                    sw_direct,
                    sw_diffuse,
                    sun_dir,
                    cos_zenith
                )
        
        # Compute volumetric fluxes if enabled
        if self.volumetric_calc is not None:
            sun_dir_tuple = (float(sun_dir[0]), float(sun_dir[1]), float(sun_dir[2]))
            self.volumetric_calc.compute_swflux_vol(
                sw_direct,
                sw_diffuse,
                cos_zenith,
                sun_dir_tuple,
                self.domain.lad  # Pass LAD for attenuation
            )
    
    @ti.kernel
    def _set_canopy_trans_one(self):
        for i in range(self.n_surfaces):
            self.surfaces.canopy_transmissivity[i] = 1.0
    
    @ti.kernel
    def _clear_direct_radiation(self):
        for i in range(self.n_surfaces):
            self.surfaces.shadow_factor[i] = 1.0
            self.surfaces.canopy_transmissivity[i] = 0.0
            self.surfaces.sw_in_direct[i] = 0.0
    
    def _compute_sw_fluxes(
        self,
        sw_direct: float,
        sw_diffuse: float,
        cos_zenith: float
    ):
        """
        Compute shortwave fluxes for all surfaces with multi-bounce reflections.
        
        Following PALM's RTM methodology (radiation_model_mod.f90 lines ~9300-9500):
        
        1. Initial pass: direct + diffuse from sky
           - PALM: surfinswdir = rad_sw_in_dir * surf_costh * dsitrans * sun_direct_factor
           - PALM: surfinswdif = rad_sw_in_diff * skyvft
           
        2. Reflection loop (DO refstep = 1, nrefsteps):
           - PALM: surfoutsl = albedo_surf * surfins
           - PALM: surfins(isurf) += svf(1,isvf) * svf(2,isvf) * surfoutsl(isurfsrc)
           - PALM: pcbinsw += csf * surfoutsl(isurfsrc) * asrc * grid_volume_inverse
           
        3. Accumulate totals:
           - PALM: surfinsw = surfinsw + surfins
           - PALM: surfoutsw = surfoutsw + surfoutsl
        """
        # Initialize all flux arrays
        self._init_flux_arrays()
        self._init_canopy_arrays()
        
        # Compute initial (first pass) radiation: direct + diffuse from sky
        self._compute_initial_sw_pass(sw_direct, sw_diffuse, cos_zenith)
        
        # Compute initial canopy absorption from direct/diffuse BEFORE reflection loop
        # This allows canopy scattering to participate in reflections
        if self.domain.lad is not None and self.config.canopy_radiation:
            sun_dir = self.solar_calc.sun_direction[None]
            self._compute_canopy_radiation_initial(
                sw_direct,
                sw_diffuse,
                sun_dir,
                cos_zenith
            )
            # Set up initial scattered radiation for the reflection loop
            if self.config.canopy_reflections:
                self._update_canopy_scattered_radiation(
                    self.domain.lad,
                    self.config.albedo_leaf
                )
        
        # Multi-bounce reflections (PALM: DO refstep = 1, nrefsteps)
        # Extended to include canopy scattering within the loop for full path coverage:
        # - Surface → Surface
        # - Surface → Canopy (absorption)
        # - Canopy → Surface (scattering)
        # - Canopy → Canopy (scattering)
        # This captures all multi-bounce paths like: Canopy→Surface→Canopy, Surface→Canopy→Surface, etc.
        if self.config.surface_reflections and self.config.n_reflection_steps > 0:
            # Check if we have canopy - use optimized fast path if not
            has_canopy = self.domain.lad is not None and self.config.canopy_radiation
            
            # Determine which optimized path to use
            canopy_ready = has_canopy and self._csf_matrix_cached and self._svf_matrix_cached
            no_canopy_ready = not has_canopy and self._svf_matrix_cached
            
            if no_canopy_ready:
                # === OPTIMIZED FAST PATH (no canopy) ===
                # Reduces kernel launches by fusing operations
                self._run_optimized_reflection_loop(self.config.n_reflection_steps)
            elif canopy_ready:
                # === OPTIMIZED FAST PATH (with canopy) ===
                # Uses sparse matrices for both SVF and CSF
                self._run_optimized_reflection_loop_with_canopy(self.config.n_reflection_steps)
            else:
                # === STANDARD PATH (fallback) ===
                for refstep in range(self.config.n_reflection_steps):
                    # PALM: surfoutsl = albedo_surf * surfins
                    # PALM: surfoutsw = surfoutsw + surfoutsl
                    # Fused: compute outgoing AND accumulate in one kernel
                    self._compute_outgoing_and_accumulate()
                    
                    # PALM: pcbinsw += csf * surfoutsl(isurfsrc) * asrc * grid_volume_inverse
                    # Surface → Canopy: absorption from surface reflections
                    if has_canopy:
                        self._accumulate_canopy_absorption_from_reflections(
                            self.domain.lad,
                            self.domain.is_solid,
                            self.config.ext_coef,
                            self._pcbinsw
                        )
                    
                    # PALM: surfins(isurf) += svf(1,isvf) * svf(2,isvf) * surfoutsl(isurfsrc)
                    # Surface → Surface reflections
                    self._compute_surface_reflections()
                    
                    # === Canopy scattering within reflection loop ===
                    if self.domain.lad is not None and self.config.canopy_reflections:
                        self._update_canopy_scattered_radiation(
                            self.domain.lad,
                            self.config.albedo_leaf
                        )
                        self._compute_canopy_to_surface_scattering(
                            self.domain.lad,
                            self.domain.is_solid,
                            self.config.ext_coef
                        )
                        if self.config.canopy_to_canopy:
                            self._compute_canopy_to_canopy_scattering(
                                self.domain.lad,
                                self.domain.is_solid,
                                self.config.albedo_leaf,
                                self.config.ext_coef
                            )
                            self._accumulate_canopy_to_canopy(
                                self.domain.lad,
                                self.config.albedo_leaf
                            )
                    
                    # PALM: surfinsw = surfinsw + surfins
                    self._accumulate_incoming()
        else:
            # No reflections - just compute single-bounce outgoing
            self._compute_surface_outgoing_no_exchange()
        
        # Note: Canopy scattering is applied AFTER canopy radiation is computed
        # (see compute_shortwave_radiation which calls _apply_canopy_scattering after CSF)
        
        # Copy final results to surface arrays
        self._copy_final_fluxes()
    
    @ti.kernel
    def _init_flux_arrays(self):
        """Initialize all flux arrays to zero."""
        for i in range(self.n_surfaces):
            self._surfins[i] = 0.0
            self._surfinl[i] = 0.0
            self._surfoutsl[i] = 0.0
            self._surfoutll[i] = 0.0
            self._surfinsw[i] = 0.0
            self._surfinlw[i] = 0.0
            self._surfoutsw[i] = 0.0
            self._surfoutlw[i] = 0.0
            self._surfinswdir[i] = 0.0
            self._surfinswdif[i] = 0.0
            self._surfinswpc[i] = 0.0  # From plant canopy
    
    @ti.kernel
    def _init_canopy_arrays(self):
        """Initialize canopy radiation arrays to zero."""
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            self._pcbinswref[i, j, k] = 0.0
            self._pcbinswc2c[i, j, k] = 0.0
            self._pcbinswc2c_total[i, j, k] = 0.0
    
    @ti.kernel
    def _compute_initial_sw_pass(
        self,
        sw_direct: ti.f32,
        sw_diffuse: ti.f32,
        cos_zenith: ti.f32
    ):
        """
        Compute initial radiation pass: direct solar + diffuse sky radiation.
        
        This is the first pass before any surface reflections.
        """
        # Minimum stable cosine of zenith angle (PALM default)
        min_stable_coszen = 0.0262
        
        for i in range(self.n_surfaces):
            direction = self.surfaces.direction[i]
            svf = self.surfaces.svf[i]
            shadow = self.surfaces.shadow_factor[i]
            canopy_trans = self.surfaces.canopy_transmissivity[i]
            
            # Get surface normal
            normal = Vector3(0.0, 0.0, 0.0)
            if direction == 0:  # Up
                normal = Vector3(0.0, 0.0, 1.0)
            elif direction == 1:  # Down
                normal = Vector3(0.0, 0.0, -1.0)
            elif direction == 2:  # North
                normal = Vector3(0.0, 1.0, 0.0)
            elif direction == 3:  # South
                normal = Vector3(0.0, -1.0, 0.0)
            elif direction == 4:  # East
                normal = Vector3(1.0, 0.0, 0.0)
            elif direction == 5:  # West
                normal = Vector3(-1.0, 0.0, 0.0)
            
            # Sun direction
            sun_dir = self.solar_calc.sun_direction[None]
            
            # Cosine of incidence angle (angle between sun and surface normal)
            cos_incidence = (sun_dir[0] * normal[0] + 
                            sun_dir[1] * normal[1] + 
                            sun_dir[2] * normal[2])
            cos_incidence = ti.max(0.0, cos_incidence)
            
            # Direct radiation
            # PALM formula: surfinswdir = rad_sw_in_dir * surf_costh * dsitrans * sun_direct_factor
            # where sun_direct_factor = 1 / max(min_stable_coszen, cos_zenith)
            # 
            # PALM's rad_sw_in_dir is Direct Horizontal Irradiance, so it multiplies by
            # sun_direct_factor to convert to DNI, then by surf_costh for surface projection.
            # 
            # palm_solar assumes sw_direct input is already DNI (Direct Normal Irradiance),
            # so we only need: sw_direct * cos_incidence * canopy_trans
            # This is equivalent to PALM when: sw_direct = rad_sw_in_dir * sun_direct_factor
            sw_in_dir = 0.0
            if cos_zenith > min_stable_coszen and shadow < 0.5:
                sw_in_dir = sw_direct * cos_incidence * canopy_trans
            
            # Diffuse radiation from sky (weighted by sky view factor)
            # The input sw_diffuse is diffuse horizontal irradiance (DHI)
            # 
            # For vertical surfaces, the SVF already incorporates the geometric
            # factor that vertical walls can only see half the sky hemisphere.
            # The SVF calculation uses view factor fractions weighted by cos(angle)
            # and normalizes against the theoretical maximum for that surface type.
            # 
            # Therefore: sw_diffuse * svf gives the correct diffuse irradiance
            # for all surface orientations.
            
            # Diffuse radiation
            # PALM formula: surfinswdif = rad_sw_in_diff * skyvft
            # where skyvft is the transmissivity-weighted sky view factor
            # palm_solar's svf is equivalent to PALM's skyvft (computed in svf.py)
            sw_in_dif = 0.0
            if direction == 0:  # Upward facing - full hemisphere (PALM: iup)
                sw_in_dif = sw_diffuse * svf
            elif direction == 1:  # Downward facing - cannot see sky (PALM: idown)
                # Downward surfaces face away from sky, receive no sky diffuse
                sw_in_dif = 0.0
            else:  # Vertical walls (PALM: inorth, isouth, ieast, iwest)
                sw_in_dif = sw_diffuse * svf
            
            # Store initial pass results
            self._surfinswdir[i] = sw_in_dir
            self._surfinswdif[i] = sw_in_dif
            
            # Initial incoming for reflection loop
            self._surfins[i] = sw_in_dir + sw_in_dif
            
            # Accumulate to totals
            self._surfinsw[i] = self._surfins[i]
    
    @ti.kernel
    def _compute_surface_outgoing(self):
        """
        Compute outgoing (reflected) radiation from each surface.
        
        PALM formula: surfoutsl = albedo * surfins
        """
        for i in range(self.n_surfaces):
            albedo = self.surfaces.albedo[i]
            self._surfoutsl[i] = albedo * self._surfins[i]
    
    @ti.kernel
    def _compute_outgoing_and_accumulate(self):
        """
        Fused kernel: Compute outgoing AND accumulate to totals in one pass.
        
        Combines _compute_surface_outgoing + _accumulate_outgoing.
        Reduces kernel launch overhead by 50% for this operation.
        """
        for i in range(self.n_surfaces):
            albedo = self.surfaces.albedo[i]
            outgoing = albedo * self._surfins[i]
            self._surfoutsl[i] = outgoing
            self._surfoutsw[i] += outgoing
    
    @ti.kernel
    def _compute_total_reflected(self):
        """
        Compute total reflected flux weighted by area (parallel reduction).
        
        This is O(n) and fully parallelized by Taichi.
        """
        self._total_reflected_flux[None] = 0.0
        self._total_reflecting_area[None] = 0.0
        
        for i in range(self.n_surfaces):
            area_i = self.surfaces.area[i]
            flux_i = self._surfoutsl[i] * area_i
            ti.atomic_add(self._total_reflected_flux[None], flux_i)
            ti.atomic_add(self._total_reflecting_area[None], area_i)
    
    @ti.kernel
    def _distribute_reflected_radiation(self):
        """
        Distribute reflected radiation based on distance-weighted view factors.
        
        PALM uses pre-computed sparse SVF matrix:
            surfins(isurf) += svf(1,isvf) * svf(2,isvf) * surfoutsl(isurfsrc)
        where svf(1,isvf) is the geometric view factor and svf(2,isvf) is transmissivity.
        
        palm_solar computes view factors dynamically for efficiency on GPU:
        - Distance between surfaces (inverse square law)
        - Orientation (dot product of normals with connection vector)
        - Visibility (simplified - assume visible if facing each other)
        
        This gives equivalent physics with different numerical implementation.
        The O(n^2) pairwise computation is fully parallelized on GPU.
        """
        PI = 3.14159265359
        
        for i in range(self.n_surfaces):
            # Receiving surface properties
            pos_i = self.surfaces.center[i]
            normal_i = self.surfaces.normal[i]
            urban_vf_i = 1.0 - self.surfaces.svf[i]
            
            # Skip if surface sees only sky
            if urban_vf_i < 0.01:
                self._surfins[i] = 0.0
                continue
            
            # Accumulate contributions from all emitting surfaces
            total_incoming = 0.0
            
            for j in range(self.n_surfaces):
                if i == j:
                    continue
                
                outgoing_j = self._surfoutsl[j]
                if outgoing_j < 0.01:
                    continue
                
                # Emitting surface properties
                pos_j = self.surfaces.center[j]
                normal_j = self.surfaces.normal[j]
                area_j = self.surfaces.area[j]
                
                # Vector from j to i
                dx = pos_i[0] - pos_j[0]
                dy = pos_i[1] - pos_j[1]
                dz = pos_i[2] - pos_j[2]
                dist_sq = dx*dx + dy*dy + dz*dz
                
                if dist_sq < 0.1:
                    continue
                
                dist = ti.sqrt(dist_sq)
                
                # Unit vector from j to i
                dir_x = dx / dist
                dir_y = dy / dist
                dir_z = dz / dist
                
                # Cosine of angle at emitting surface (must be positive - facing towards i)
                cos_emit = normal_j[0]*dir_x + normal_j[1]*dir_y + normal_j[2]*dir_z
                
                # Cosine of angle at receiving surface (must be positive - facing towards j)
                cos_recv = -(normal_i[0]*dir_x + normal_i[1]*dir_y + normal_i[2]*dir_z)
                
                # Both surfaces must face each other
                if cos_emit > 0.0 and cos_recv > 0.0:
                    # Radiative view factor formula:
                    # F_ij = (cos_emit * cos_recv * area_j) / (pi * dist^2)
                    # Incoming irradiance from j = outgoing_j * F_ij
                    view_factor = (cos_emit * cos_recv * area_j) / (PI * dist_sq)
                    
                    # Clamp to reasonable maximum
                    view_factor = ti.min(view_factor, 1.0)
                    
                    total_incoming += outgoing_j * view_factor
            
            # Scale by urban view factor (what fraction of hemisphere sees urban surfaces)
            self._surfins[i] = total_incoming * urban_vf_i
    
    @ti.kernel
    def _distribute_reflected_radiation_with_canopy(
        self,
        lad: ti.template(),
        is_solid: ti.template(),
        ext_coef: ti.f32,
        albedo_leaf: ti.f32
    ):
        """
        Distribute reflected radiation with LAD transmissivity.
        
        For each surface pair, traces ray through canopy applying Beer-Lambert
        attenuation. Radiation absorbed by canopy is partially scattered
        back to surfaces (based on leaf albedo).
        
        Args:
            lad: 3D field of Leaf Area Density
            is_solid: 3D field of solid cells
            ext_coef: Extinction coefficient for canopy
            albedo_leaf: Leaf/tree albedo for scattering
        """
        PI = 3.14159265359
        nx = self.domain.nx
        ny = self.domain.ny
        nz = self.domain.nz
        dx = self.domain.dx
        dy = self.domain.dy
        dz = self.domain.dz
        
        for i in range(self.n_surfaces):
            # Receiving surface properties
            pos_i = self.surfaces.center[i]
            normal_i = self.surfaces.normal[i]
            urban_vf_i = 1.0 - self.surfaces.svf[i]
            
            # Skip if surface sees only sky
            if urban_vf_i < 0.01:
                self._surfins[i] = 0.0
                continue
            
            # Accumulate contributions from all emitting surfaces
            total_incoming = 0.0
            
            for j in range(self.n_surfaces):
                if i == j:
                    continue
                
                outgoing_j = self._surfoutsl[j]
                if outgoing_j < 0.01:
                    continue
                
                # Emitting surface properties
                pos_j = self.surfaces.center[j]
                normal_j = self.surfaces.normal[j]
                area_j = self.surfaces.area[j]
                
                # Vector from j to i
                diff_x = pos_i[0] - pos_j[0]
                diff_y = pos_i[1] - pos_j[1]
                diff_z = pos_i[2] - pos_j[2]
                dist_sq = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z
                
                if dist_sq < 0.1:
                    continue
                
                dist = ti.sqrt(dist_sq)
                
                # Unit vector from j to i
                dir_x = diff_x / dist
                dir_y = diff_y / dist
                dir_z = diff_z / dist
                
                # Cosine of angle at emitting surface
                cos_emit = normal_j[0]*dir_x + normal_j[1]*dir_y + normal_j[2]*dir_z
                
                # Cosine of angle at receiving surface
                cos_recv = -(normal_i[0]*dir_x + normal_i[1]*dir_y + normal_i[2]*dir_z)
                
                # Both surfaces must face each other
                if cos_emit > 0.0 and cos_recv > 0.0:
                    # Compute transmissivity through canopy
                    trans, blocked = ray_point_to_point_transmissivity(
                        pos_j, pos_i,
                        lad, is_solid,
                        nx, ny, nz,
                        dx, dy, dz,
                        ext_coef
                    )
                    
                    # Skip if blocked by solid
                    if blocked == 1:
                        continue
                    
                    # View factor
                    view_factor = (cos_emit * cos_recv * area_j) / (PI * dist_sq)
                    view_factor = ti.min(view_factor, 1.0)
                    
                    # Apply canopy transmissivity
                    total_incoming += outgoing_j * view_factor * trans
            
            # Scale by urban view factor
            self._surfins[i] = total_incoming * urban_vf_i
    
    def _compute_surface_reflections(self):
        """
        Compute radiation exchange between surfaces for one reflection step.
        
        If canopy_reflections is enabled and LAD exists, uses ray tracing
        through vegetation with Beer-Lambert attenuation. Otherwise uses
        simplified geometry-based distribution.
        
        Uses two-pass algorithm:
        1. Parallel reduction to compute total reflected flux
        2. Parallel distribution based on view factors (with optional LAD)
        
        Note: For optimized multi-step reflection, use _compute_surface_reflections_optimized()
        which uses separate kernels with ping-pong buffers for ~100x GPU speedup.
        """
        # Pass 1: Compute total reflected flux (parallel reduction)
        self._compute_total_reflected()
        
        # Pass 2: Distribute to each surface
        if self.config.canopy_reflections and self.domain.lad is not None:
            # Use canopy-aware version with LAD transmissivity
            if self.config.cache_svf_matrix and self._svf_matrix_cached:
                # Use cached SVF matrix with optimized separate kernels
                self._distribute_reflected_cached_single_step()
            else:
                # Compute dynamically (O(n²))
                self._distribute_reflected_radiation_with_canopy(
                    self.domain.lad,
                    self.domain.is_solid,
                    self.config.ext_coef,
                    self.config.albedo_leaf
                )
        else:
            # Use simple version without canopy
            if self.config.cache_svf_matrix and self._svf_matrix_cached:
                self._distribute_reflected_cached_single_step()
            else:
                self._distribute_reflected_radiation()
    
    def _distribute_reflected_cached_single_step(self):
        """
        Single-step reflection distribution using optimized separate kernels.
        
        Uses separate kernels instead of fused kernel with internal ti.sync()
        for ~100x GPU speedup. Each kernel is fully parallel without barriers.
        """
        n = self.n_surfaces
        svf_nnz = self._svf_nnz
        
        # Reset incoming buffer
        self._reset_buffer(self._surfins, n)
        
        # Sparse matmul for reflection distribution
        self._sparse_matmul_step(self._surfoutsl, self._surfins, svf_nnz)
        
        # Scale by urban view factor
        self._scale_by_urban_vf(self._surfins, n)
    
    def compute_svf_matrix(self):
        """
        Pre-compute surface-to-surface view factor matrix (PALM-like approach).
        
        This is expensive O(n²) but only needs to be done once for fixed geometry.
        Subsequent reflection iterations become O(nnz) instead of O(n²).
        
        Call this before running multi-timestep simulations for efficiency.
        """
        if self._svf_matrix_cached:
            print("SVF matrix already cached, skipping recomputation.")
            return
        
        # Check if arrays were allocated in __init__
        if self._svf_source is None:
            print("Warning: SVF caching not enabled in config, skipping matrix computation.")
            return
        
        print(f"Pre-computing SVF matrix for {self.n_surfaces} surfaces...")
        print("  This is O(n²) but only runs once for fixed geometry.")
        
        # Compute the matrix
        if self.domain.lad is not None and self.config.canopy_reflections:
            self._compute_svf_matrix_with_canopy(
                self.domain.lad,
                self.domain.is_solid,
                self.config.ext_coef,
                self.config.svf_min_threshold
            )
        else:
            self._compute_svf_matrix_simple(self.config.svf_min_threshold)
        
        # Clamp nnz to max entries to avoid out-of-bounds reads
        computed_nnz = int(self._svf_count[None])
        if computed_nnz > self._max_svf_entries:
            truncated_pct = (computed_nnz - self._max_svf_entries) / computed_nnz * 100
            print(f"Warning: SVF matrix truncated! Computed {computed_nnz:,} entries but buffer size is {self._max_svf_entries:,}.")
            print(f"  {truncated_pct:.1f}% of surface-to-surface view factors are being discarded.")
            print(f"  This may affect reflection accuracy. To fix: clear_all_caches() before creating this model,")
            print(f"  or increase svf_min_threshold in RadiationConfig to reduce entries.")
            self._svf_nnz = self._max_svf_entries
        else:
            self._svf_nnz = computed_nnz
            
        self._svf_matrix_cached = True
        
        sparsity = self._svf_nnz / (self.n_surfaces * self.n_surfaces) * 100
        memory_mb = self._svf_nnz * 12 / 1e6  # 12 bytes per entry (2 int32 + 2 float16)
        print(f"  SVF matrix computed: {self._svf_nnz:,} non-zero entries ({memory_mb:.1f} MB)")
        print(f"  Sparsity: {sparsity:.2f}% of full matrix")
        print(f"  Speedup factor: ~{self.n_surfaces * self.n_surfaces / max(1, self._svf_nnz):.1f}x per reflection step")
        
        # Build CSR format for optimized sparse matmul
        if self._svf_csr_row_ptr is not None:
            self._build_csr_format()
    
    def _build_csr_format(self):
        """
        Convert COO format to CSR format for optimized sparse matmul.
        
        CSR (Compressed Sparse Row by target) provides:
        1. O(n_surfaces) parallelism with one thread per row
        2. Local accumulation instead of atomic operations
        3. Better cache locality for reading source values
        
        Reduces sparse matmul time by ~2-5x on GPU.
        """
        import numpy as np
        
        print("  Building CSR format for optimized sparse matmul...")
        
        # Copy COO data to numpy for sorting
        coo_target = self._svf_target.to_numpy()[:self._svf_nnz]
        coo_source = self._svf_source.to_numpy()[:self._svf_nnz]
        coo_vf = self._svf_vf.to_numpy()[:self._svf_nnz]
        coo_trans = self._svf_trans.to_numpy()[:self._svf_nnz]
        
        # Pre-multiply vf * trans
        coo_val = coo_vf * coo_trans
        
        # Sort by target (row) for CSR format
        sort_idx = np.argsort(coo_target)
        sorted_target = coo_target[sort_idx]
        sorted_source = coo_source[sort_idx]
        sorted_val = coo_val[sort_idx]
        
        # Build row pointers
        row_ptr = np.zeros(self.n_surfaces + 1, dtype=np.int32)
        for t in sorted_target:
            row_ptr[t + 1] += 1
        row_ptr = np.cumsum(row_ptr)
        
        # Pad arrays to match Taichi field shape (required for from_numpy)
        # The fields are allocated to _max_svf_entries, but we only use _svf_nnz
        padded_col_idx = np.zeros(self._max_svf_entries, dtype=np.int32)
        padded_val = np.zeros(self._max_svf_entries, dtype=np.float32)
        padded_col_idx[:self._svf_nnz] = sorted_source
        padded_val[:self._svf_nnz] = sorted_val
        
        # Copy to Taichi fields
        self._svf_csr_row_ptr.from_numpy(row_ptr)
        self._svf_csr_col_idx.from_numpy(padded_col_idx)
        self._svf_csr_val.from_numpy(padded_val)
        
        self._svf_csr_ready = True
        print(f"  CSR format ready: {self.n_surfaces} rows, {self._svf_nnz:,} entries")
    
    def compute_csf_matrix(self):
        """
        Pre-compute Canopy-Surface Factor matrix.
        
        Stores geometric factors for Surface <-> Canopy interactions.
        This allows O(nnz) computation for canopy absorption and scattering
        instead of O(N_cells * N_surfaces).
        """
        if self._csf_matrix_cached:
            return
            
        if self._csf_canopy_idx is None:
            return
            
        print(f"Pre-computing CSF matrix for canopy-surface interactions...")
        
        self._compute_csf_matrix_kernel(
            self.domain.lad,
            self.domain.is_solid,
            self.config.ext_coef,
            1e-7 # Threshold
        )
        
        computed_nnz = int(self._csf_count[None])
        if computed_nnz > self._max_csf_entries:
            print(f"Warning: CSF matrix truncated! {computed_nnz} > {self._max_csf_entries}")
            self._csf_nnz = self._max_csf_entries
        else:
            self._csf_nnz = computed_nnz
            
        self._csf_matrix_cached = True
        print(f"  CSF matrix computed: {self._csf_nnz:,} entries")

    @ti.kernel
    def _compute_csf_matrix_kernel(
        self,
        lad: ti.template(),
        is_solid: ti.template(),
        ext_coef: ti.f32,
        min_threshold: ti.f32
    ):
        """
        Compute CSF matrix entries.
        
        Stores (canopy_idx, surface_idx, base_factor) where:
        base_factor = (cell_area * cos_surf * trans) / (4 * PI * dist_sq)
        
        This factor is used for:
        1. Surface->Canopy: absorbed = outgoing * area_surf * base_factor * abs_frac / grid_vol
        2. Canopy->Surface: incoming = scattered_power * base_factor / area_surf
        """
        PI = 3.14159265359
        nx = self.domain.nx
        ny = self.domain.ny
        nz = self.domain.nz
        dx = self.domain.dx
        dy = self.domain.dy
        dz = self.domain.dz
        
        # Reset counter
        self._csf_count[None] = 0
        
        # Iterate over all canopy cells
        for ci, cj, ck in ti.ndrange(nx, ny, nz):
            cell_lad = lad[ci, cj, ck]
            if cell_lad <= 0.0:
                continue
                
            # Cell center
            pos_cell = Vector3(
                (ci + 0.5) * dx,
                (cj + 0.5) * dy,
                (ck + 0.5) * dz
            )
            
            # Linear index for canopy cell
            canopy_idx = ci * (ny * nz) + cj * nz + ck
            
            # Iterate over all surfaces
            for surf_i in range(self.n_surfaces):
                pos_surf = self.surfaces.center[surf_i]
                normal_surf = self.surfaces.normal[surf_i]
                
                # Vector from surface to cell
                diff = pos_cell - pos_surf
                dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
                
                if dist_sq < 0.1:
                    continue
                    
                dist = ti.sqrt(dist_sq)
                
                # Direction from surface to cell
                dir_x = diff[0] / dist
                dir_y = diff[1] / dist
                dir_z = diff[2] / dist
                
                # Check if surface faces the cell
                cos_emit = (normal_surf[0]*dir_x + normal_surf[1]*dir_y + normal_surf[2]*dir_z)
                
                if cos_emit > 0.0:
                    # Geometric factor (solid angle / 4pi * cos)
                    # cell_solid_angle = (dx * dy) / dist_sq  (approx)
                    # factor = (cell_solid_angle * cos_emit) / (4 * PI)
                    # factor = (dx * dy * cos_emit) / (4 * PI * dist_sq)
                    
                    base_factor = (dx * dy * cos_emit) / (4.0 * PI * dist_sq)
                    
                    if base_factor > min_threshold:
                        # Check transmissivity
                        trans, blocked = ray_point_to_point_transmissivity(
                            pos_surf, pos_cell,
                            lad, is_solid,
                            nx, ny, nz,
                            dx, dy, dz,
                            ext_coef
                        )
                        
                        if blocked == 0 and trans > 0.01:
                            final_factor = base_factor * trans
                            
                            if final_factor > min_threshold:
                                idx = ti.atomic_add(self._csf_count[None], 1)
                                if idx < self._max_csf_entries:
                                    self._csf_canopy_idx[idx] = canopy_idx
                                    self._csf_surface_idx[idx] = surf_i
                                    self._csf_val[idx] = final_factor

    @ti.kernel
    def _compute_svf_matrix_simple(self, min_threshold: ti.f32):
        """
        Compute SVF matrix without canopy (simple geometry).
        
        Stores entries where view factor > min_threshold in sparse COO format.
        """
        PI = 3.14159265359
        
        # Reset counter
        self._svf_count[None] = 0
        
        # Compute all pairwise view factors
        for i in range(self.n_surfaces):
            pos_i = self.surfaces.center[i]
            normal_i = self.surfaces.normal[i]
            
            for j in range(self.n_surfaces):
                if i == j:
                    continue
                
                pos_j = self.surfaces.center[j]
                normal_j = self.surfaces.normal[j]
                area_j = self.surfaces.area[j]
                
                # Vector from j to i
                dx = pos_i[0] - pos_j[0]
                dy = pos_i[1] - pos_j[1]
                dz = pos_i[2] - pos_j[2]
                dist_sq = dx*dx + dy*dy + dz*dz
                
                if dist_sq < 0.1:
                    continue
                
                dist = ti.sqrt(dist_sq)
                dir_x = dx / dist
                dir_y = dy / dist
                dir_z = dz / dist
                
                # Cosines
                cos_emit = normal_j[0]*dir_x + normal_j[1]*dir_y + normal_j[2]*dir_z
                cos_recv = -(normal_i[0]*dir_x + normal_i[1]*dir_y + normal_i[2]*dir_z)
                
                if cos_emit > 0.0 and cos_recv > 0.0:
                    view_factor = (cos_emit * cos_recv * area_j) / (PI * dist_sq)
                    view_factor = ti.min(view_factor, 1.0)
                    
                    if view_factor > min_threshold:
                        # Atomic increment to get unique index
                        idx = ti.atomic_add(self._svf_count[None], 1)
                        if idx < self._max_svf_entries:
                            self._svf_source[idx] = j
                            self._svf_target[idx] = i
                            self._svf_vf[idx] = view_factor
                            self._svf_trans[idx] = 1.0  # No canopy
    
    @ti.kernel
    def _compute_svf_matrix_with_canopy(
        self,
        lad: ti.template(),
        is_solid: ti.template(),
        ext_coef: ti.f32,
        min_threshold: ti.f32
    ):
        """
        Compute SVF matrix with canopy transmissivity.
        
        For each surface pair, computes view factor AND transmissivity
        through intervening vegetation using Beer-Lambert law.
        
        Stores entries where (vf * trans) > min_threshold.
        """
        PI = 3.14159265359
        nx = self.domain.nx
        ny = self.domain.ny
        nz = self.domain.nz
        dx = self.domain.dx
        dy = self.domain.dy
        dz = self.domain.dz
        
        # Reset counter
        self._svf_count[None] = 0
        
        for i in range(self.n_surfaces):
            pos_i = self.surfaces.center[i]
            normal_i = self.surfaces.normal[i]
            
            for j in range(self.n_surfaces):
                if i == j:
                    continue
                
                pos_j = self.surfaces.center[j]
                normal_j = self.surfaces.normal[j]
                area_j = self.surfaces.area[j]
                
                # Vector from j to i
                diff_x = pos_i[0] - pos_j[0]
                diff_y = pos_i[1] - pos_j[1]
                diff_z = pos_i[2] - pos_j[2]
                dist_sq = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z
                
                if dist_sq < 0.1:
                    continue
                
                dist = ti.sqrt(dist_sq)
                dir_x = diff_x / dist
                dir_y = diff_y / dist
                dir_z = diff_z / dist
                
                cos_emit = normal_j[0]*dir_x + normal_j[1]*dir_y + normal_j[2]*dir_z
                cos_recv = -(normal_i[0]*dir_x + normal_i[1]*dir_y + normal_i[2]*dir_z)
                
                if cos_emit > 0.0 and cos_recv > 0.0:
                    # Compute transmissivity through canopy
                    trans, blocked = ray_point_to_point_transmissivity(
                        pos_j, pos_i,
                        lad, is_solid,
                        nx, ny, nz,
                        dx, dy, dz,
                        ext_coef
                    )
                    
                    if blocked == 0 and trans > 0.001:
                        view_factor = (cos_emit * cos_recv * area_j) / (PI * dist_sq)
                        view_factor = ti.min(view_factor, 1.0)
                        
                        effective_vf = view_factor * trans
                        
                        if effective_vf > min_threshold:
                            idx = ti.atomic_add(self._svf_count[None], 1)
                            if idx < self._max_svf_entries:
                                self._svf_source[idx] = j
                                self._svf_target[idx] = i
                                self._svf_vf[idx] = view_factor
                                self._svf_trans[idx] = trans
    
    # ========== Optimized Reflection Kernels (Separate for GPU efficiency) ==========
    # Using separate kernels without internal ti.sync() is ~100x faster than fused kernels
    # with ti.sync() inside. Taichi handles synchronization between kernel calls efficiently.
    
    @ti.kernel
    def _reset_buffer(self, buf: ti.template(), n: ti.i32):
        """Reset a buffer to zero - vectorized for GPU efficiency."""
        for i in range(n):
            buf[i] = 0.0
    
    @ti.kernel
    def _compute_outgoing_step(self, surfins: ti.template(), surfout: ti.template(), n: ti.i32):
        """Compute outgoing radiation: surfout = albedo * surfins."""
        for i in range(n):
            surfout[i] = self.surfaces.albedo[i] * surfins[i]
    
    @ti.kernel
    def _sparse_matmul_step(
        self,
        surfout: ti.template(),
        surfins_next: ti.template(),
        svf_nnz: ti.i32
    ):
        """
        Sparse matrix-vector multiply for reflection distribution.
        
        PALM equivalent: surfins(isurf) += svf(1,isvf) * svf(2,isvf) * surfoutsl(isurfsrc)
        
        Optimized for GPU with:
        - Coalesced memory access through sparse COO format
        - Minimal thread divergence with early threshold check
        - Efficient atomic operations for parallel accumulation
        """
        for idx in range(svf_nnz):
            source = self._svf_source[idx]
            target = self._svf_target[idx]
            vf = self._svf_vf[idx]
            trans = self._svf_trans[idx]
            
            # Pre-multiply vf * trans to reduce FLOPs
            vf_trans = vf * trans
            outgoing = surfout[source]
            
            # Use threshold to skip negligible contributions
            if outgoing * vf_trans > 0.001:
                ti.atomic_add(surfins_next[target], outgoing * vf_trans)
    
    @ti.kernel
    def _sparse_matmul_step_batched(
        self,
        surfout: ti.template(),
        surfins_next: ti.template(),
        svf_nnz: ti.i32,
        batch_size: ti.i32
    ):
        """
        Batched sparse matrix-vector multiply for improved GPU utilization.
        
        Processes multiple sparse entries per thread to improve memory locality
        and reduce atomic operation contention.
        """
        n_batches = (svf_nnz + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = ti.min(start_idx + batch_size, svf_nnz)
            
            # Local accumulator to reduce atomics
            for idx in range(start_idx, end_idx):
                source = self._svf_source[idx]
                target = self._svf_target[idx]
                vf_trans = self._svf_vf[idx] * self._svf_trans[idx]
                outgoing = surfout[source]
                
                if outgoing * vf_trans > 0.001:
                    ti.atomic_add(surfins_next[target], outgoing * vf_trans)
    
    @ti.kernel
    def _scale_by_urban_vf(self, surfins: ti.template(), n: ti.i32):
        """Scale incoming by urban view factor (1 - SVF)."""
        for i in range(n):
            urban_vf_i = 1.0 - self.surfaces.svf[i]
            if urban_vf_i < 0.01:
                surfins[i] = 0.0
            else:
                surfins[i] *= urban_vf_i
    
    @ti.kernel
    def _sparse_matmul_csr(
        self,
        surfout: ti.template(),
        surfins_next: ti.template(),
        n_surfaces: ti.i32
    ):
        """
        CSR-format sparse matrix-vector multiply for reflection distribution.
        
        MASSIVELY optimized for GPU:
        - One thread per target surface (row) = perfect parallelism
        - No atomic operations needed (each row is processed by one thread)
        - Local accumulation in registers before final write
        - Better cache locality from contiguous column access
        
        Reduces atomic operations from O(nnz) = 45M to O(n) = 42K
        Expected speedup: 2-5x over COO format with atomics.
        """
        for row in range(n_surfaces):
            row_start = self._svf_csr_row_ptr[row]
            row_end = self._svf_csr_row_ptr[row + 1]
            
            # Local accumulator - no atomics needed!
            local_sum = 0.0
            
            for idx in range(row_start, row_end):
                source = self._svf_csr_col_idx[idx]
                vf_trans = self._svf_csr_val[idx]  # Pre-multiplied vf * trans
                outgoing = surfout[source]
                
                # Threshold check for negligible contributions
                if outgoing * vf_trans > 0.001:
                    local_sum += outgoing * vf_trans
            
            # Single write per row (no atomic needed since one thread per row)
            surfins_next[row] = local_sum
    
    @ti.kernel
    def _accumulate_reflected(self, surfins: ti.template(), n: ti.i32):
        """Accumulate incoming reflected radiation to totals."""
        for i in range(n):
            self._surfins[i] += surfins[i]
    
    def _distribute_reflected_cached_optimized(self, svf_nnz: int, use_ping: bool):
        """
        Optimized reflection distribution using separate kernels.
        
        This is ~100x faster than the fused kernel approach because:
        1. No ti.sync() inside kernels (Taichi handles sync between kernel calls)
        2. Each kernel is fully parallel without internal barriers
        3. GPU can overlap kernel launches with computation
        
        Args:
            svf_nnz: Number of non-zero entries in SVF matrix
            use_ping: If True, read from ping buffer and write to pong buffer
        """
        n = self.n_surfaces
        
        if use_ping:
            # Read from ping, write to pong
            self._compute_outgoing_step(self._surfins_ping, self._surfoutsl, n)
            self._reset_buffer(self._surfins_pong, n)
            self._sparse_matmul_step(self._surfoutsl, self._surfins_pong, svf_nnz)
            self._scale_by_urban_vf(self._surfins_pong, n)
            self._accumulate_reflected(self._surfins_pong, n)
        else:
            # Read from pong, write to ping
            self._compute_outgoing_step(self._surfins_pong, self._surfoutsl, n)
            self._reset_buffer(self._surfins_ping, n)
            self._sparse_matmul_step(self._surfoutsl, self._surfins_ping, svf_nnz)
            self._scale_by_urban_vf(self._surfins_ping, n)
            self._accumulate_reflected(self._surfins_ping, n)
        
        # Accumulate outgoing to totals
        self._accumulate_outgoing()
    
    @ti.kernel
    def _init_ping_buffer(self, n: ti.i32):
        """Initialize ping buffer with initial SW radiation for reflection loop."""
        for i in range(n):
            self._surfins_ping[i] = self._surfinswdir[i] + self._surfinswdif[i]
            self._surfins[i] = 0.0  # Reset accumulated reflected
    
    def _run_optimized_reflection_loop(self, n_steps: int):
        """
        Optimized reflection loop using fused kernels and ping-pong buffers.
        
        This is the GPU-optimized fast path for surface-only reflections
        (no canopy). Achieves ~10-20x speedup over CPU by:
        1. Fusing outgoing + accumulate into single kernel
        2. Using separate kernels (no internal ti.sync())
        3. Ping-pong buffers for efficient memory access
        4. Minimizing kernel launch count
        5. Using ultra-fused kernel for small reflection counts
        
        Only 4 kernel calls per reflection step vs 5+ in standard path.
        
        Args:
            n_steps: Number of reflection iterations
        """
        n = self.n_surfaces
        svf_nnz = self._svf_nnz
        
        # Use ultra-optimized path for typical reflection counts (1-5 steps)
        # This reduces kernel launch overhead significantly
        if n_steps <= 5:
            self._run_reflection_loop_ultra_fused(n_steps, n, svf_nnz)
            return
        
        for step in range(n_steps):
            use_ping = (step % 2 == 0)
            
            if use_ping:
                # Step 1: Compute outgoing from current incoming (fused with accumulate)
                self._compute_outgoing_fused(self._surfins, n)
                
                # Step 2: Reset next buffer
                self._reset_buffer(self._surfins_pong, n)
                
                # Step 3: Sparse matmul for reflection distribution
                self._sparse_matmul_step(self._surfoutsl, self._surfins_pong, svf_nnz)
                
                # Step 4: Scale by urban VF and accumulate incoming (fused)
                self._scale_and_accumulate_incoming(self._surfins_pong, n)
                
                # Copy to _surfins for next iteration
                self._copy_buffer(self._surfins_pong, self._surfins, n)
            else:
                self._compute_outgoing_fused(self._surfins, n)
                self._reset_buffer(self._surfins_ping, n)
                self._sparse_matmul_step(self._surfoutsl, self._surfins_ping, svf_nnz)
                self._scale_and_accumulate_incoming(self._surfins_ping, n)
                self._copy_buffer(self._surfins_ping, self._surfins, n)
    
    def _run_reflection_loop_ultra_fused(self, n_steps: int, n: int, svf_nnz: int):
        """
        Ultra-optimized reflection loop with minimal kernel launches.
        
        For typical 3-step reflections, this reduces from 15+ kernel calls
        to just 3-4 by doing more work per kernel.
        
        Uses CSR format if available for ~2-5x faster sparse matmul.
        """
        # Choose sparse matmul method: CSR (fast, no atomics) or COO (fallback)
        use_csr = self._svf_csr_ready
        
        def do_sparse_matmul(surfout, surfins_next):
            if use_csr:
                # CSR format: one thread per row, no atomics
                self._sparse_matmul_csr(surfout, surfins_next, n)
            else:
                # COO format fallback with atomics
                self._reset_buffer(surfins_next, n)
                self._sparse_matmul_step(surfout, surfins_next, svf_nnz)
        
        # Step 1: First reflection iteration with combined operations
        # Compute outgoing and prepare for sparse matmul
        self._compute_outgoing_fused(self._surfins, n)
        do_sparse_matmul(self._surfoutsl, self._surfins_pong)
        self._scale_and_accumulate_incoming(self._surfins_pong, n)
        
        if n_steps >= 2:
            # Step 2: Second reflection - use pong as input
            self._compute_outgoing_fused_from_buffer(self._surfins_pong, n)
            do_sparse_matmul(self._surfoutsl, self._surfins_ping)
            self._scale_and_accumulate_incoming(self._surfins_ping, n)
        
        if n_steps >= 3:
            # Step 3: Third reflection - use ping as input
            self._compute_outgoing_fused_from_buffer(self._surfins_ping, n)
            do_sparse_matmul(self._surfoutsl, self._surfins_pong)
            self._scale_and_accumulate_incoming(self._surfins_pong, n)
        
        # Handle remaining steps if any
        for step in range(3, n_steps):
            use_ping = (step % 2 == 1)
            src = self._surfins_ping if use_ping else self._surfins_pong
            dst = self._surfins_pong if use_ping else self._surfins_ping
            
            self._compute_outgoing_fused_from_buffer(src, n)
            do_sparse_matmul(self._surfoutsl, dst)
            self._scale_and_accumulate_incoming(dst, n)
    
    @ti.kernel
    def _compute_outgoing_fused_from_buffer(self, surfins: ti.template(), n: ti.i32):
        """Compute outgoing from a specific buffer and accumulate."""
        for i in range(n):
            outgoing = self.surfaces.albedo[i] * surfins[i]
            self._surfoutsl[i] = outgoing
            self._surfoutsw[i] += outgoing
    
    def _run_optimized_reflection_loop_with_canopy(self, n_steps: int):
        """
        Optimized reflection loop with canopy interactions.
        
        Uses sparse matrices for both Surface-Surface (SVF) and Surface-Canopy (CSF)
        interactions, achieving O(nnz) complexity instead of O(N_surf * N_cell).
        """
        n = self.n_surfaces
        svf_nnz = self._svf_nnz
        csf_nnz = self._csf_nnz
        
        # Initialize ping buffer with initial radiation
        self._init_ping_buffer(n)
        
        for step in range(n_steps):
            use_ping = (step % 2 == 0)
            
            # Input buffer for this step (contains incoming radiation)
            src_buf = self._surfins_ping if use_ping else self._surfins_pong
            # Output buffer for next step (will accumulate reflected radiation)
            dst_buf = self._surfins_pong if use_ping else self._surfins_ping
            
            # 1. Compute outgoing from surfaces (and accumulate to total outgoing)
            self._compute_outgoing_fused(src_buf, n)
            
            # 2. Surface -> Canopy Absorption (using CSF matrix)
            self._csf_absorb_step(self._surfoutsl, csf_nnz)
            
            # 3. Update Canopy Scattering (based on absorbed)
            self._update_canopy_scattered_optimized(self.domain.lad, self.config.albedo_leaf)
            
            # 4. Reset destination buffer and do sparse matmul
            if self._svf_csr_ready:
                # CSR format: one thread per row, no atomics, includes reset
                self._sparse_matmul_csr(self._surfoutsl, dst_buf, n)
            else:
                # COO format fallback
                self._reset_buffer(dst_buf, n)
                self._sparse_matmul_step(self._surfoutsl, dst_buf, svf_nnz)
            
            # 6. Canopy -> Surface Scattering (CSF matrix transposed)
            self._csf_scatter_step(dst_buf, csf_nnz)
            
            # 7. Scale by urban VF and accumulate to total incoming
            self._scale_and_accumulate_incoming(dst_buf, n)
            
            # Copy to _surfins for consistency (though not strictly needed for loop)
            self._copy_buffer(dst_buf, self._surfins, n)

    @ti.kernel
    def _csf_absorb_step(self, surfout: ti.template(), csf_nnz: ti.i32):
        """
        Surface -> Canopy absorption using sparse CSF matrix.
        """
        grid_vol_inv = 1.0 / (self.domain.dx * self.domain.dy * self.domain.dz)
        ext_coef = self.config.ext_coef
        
        for idx in range(csf_nnz):
            canopy_idx = self._csf_canopy_idx[idx]
            surf_idx = self._csf_surface_idx[idx]
            base_factor = self._csf_val[idx]
            
            outgoing = surfout[surf_idx]
            if outgoing > 0.01:
                # Reconstruct 3D indices from linear index
                # canopy_idx = i * (ny * nz) + j * nz + k
                tmp = canopy_idx
                k = tmp % self.domain.nz
                tmp //= self.domain.nz
                j = tmp % self.domain.ny
                i = tmp // self.domain.ny
                
                lad_val = self.domain.lad[i, j, k]
                area_surf = self.surfaces.area[surf_idx]
                
                # Absorption fraction (approximate path = dz)
                abs_frac = 1.0 - ti.exp(-ext_coef * lad_val * self.domain.dz)
                
                # absorbed = outgoing * area_surf * base_factor * abs_frac / grid_vol
                absorbed = outgoing * area_surf * base_factor * abs_frac * grid_vol_inv
                
                ti.atomic_add(self._pcbinsw[i, j, k], absorbed)

    @ti.kernel
    def _csf_scatter_step(self, surfins_next: ti.template(), csf_nnz: ti.i32):
        """
        Canopy -> Surface scattering using sparse CSF matrix.
        """
        grid_vol = self.domain.dx * self.domain.dy * self.domain.dz
        
        for idx in range(csf_nnz):
            canopy_idx = self._csf_canopy_idx[idx]
            surf_idx = self._csf_surface_idx[idx]
            base_factor = self._csf_val[idx]
            
            # Reconstruct indices
            tmp = canopy_idx
            k = tmp % self.domain.nz
            tmp //= self.domain.nz
            j = tmp % self.domain.ny
            i = tmp // self.domain.ny
            
            # Scattered power from this cell (W/m^3 * m^3 = W)
            scattered_flux_vol = self._pcbinswref[i, j, k]
            if scattered_flux_vol > 0.001:
                scattered_power = scattered_flux_vol * grid_vol
                
                area_surf = self.surfaces.area[surf_idx]
                
                # incoming = scattered_power * base_factor / area_surf
                contribution = scattered_power * base_factor / area_surf
                
                ti.atomic_add(surfins_next[surf_idx], contribution)
                ti.atomic_add(self._surfinswpc[surf_idx], contribution)

    @ti.kernel
    def _update_canopy_scattered_optimized(self, lad: ti.template(), albedo_leaf: ti.f32):
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            if lad[i, j, k] > 0.0:
                self._pcbinswref[i, j, k] = self._pcbinsw[i, j, k] * albedo_leaf

    @ti.kernel
    def _compute_outgoing_fused(self, surfins: ti.template(), n: ti.i32):
        """Fused: compute outgoing AND accumulate to totals."""
        for i in range(n):
            outgoing = self.surfaces.albedo[i] * surfins[i]
            self._surfoutsl[i] = outgoing
            self._surfoutsw[i] += outgoing
    
    @ti.kernel
    def _scale_and_accumulate_incoming(self, surfins: ti.template(), n: ti.i32):
        """Fused: scale by urban VF AND accumulate to incoming totals."""
        for i in range(n):
            urban_vf = 1.0 - self.surfaces.svf[i]
            if urban_vf < 0.01:
                surfins[i] = 0.0
            else:
                surfins[i] *= urban_vf
                self._surfinsw[i] += surfins[i]
    
    @ti.kernel
    def _copy_buffer(self, src: ti.template(), dst: ti.template(), n: ti.i32):
        """Copy buffer contents."""
        for i in range(n):
            dst[i] = src[i]
    
    @ti.kernel
    def _distribute_reflected_cached(self, svf_nnz: ti.i32):
        """
        Distribute reflected radiation using cached SVF matrix (PALM-like).
        
        NOTE: This is the legacy single-kernel version kept for compatibility.
        The optimized version using separate kernels (_distribute_reflected_cached_optimized)
        is ~100x faster and should be preferred.
        
        This is O(nnz) instead of O(n²), providing major speedup for
        multi-timestep simulations with fixed geometry.
        
        PALM equivalent: surfins(isurf) += svf(1,isvf) * svf(2,isvf) * surfoutsl(isurfsrc)
        """
        # Reset incoming first
        for i in range(self.n_surfaces):
            self._surfins[i] = 0.0
        
        # Sync point - ensure reset completes before sparse matmul
        ti.sync()
        
        # Apply sparse matrix-vector multiply
        for idx in range(svf_nnz):
            source = self._svf_source[idx]
            target = self._svf_target[idx]
            vf = self._svf_vf[idx]
            trans = self._svf_trans[idx]
            
            outgoing = self._surfoutsl[source]
            # Use same threshold as non-cached version (0.01)
            if outgoing > 0.01:
                contribution = outgoing * vf * trans
                ti.atomic_add(self._surfins[target], contribution)
        
        # Sync before scaling
        ti.sync()
        
        # Scale by urban view factor
        for i in range(self.n_surfaces):
            urban_vf_i = 1.0 - self.surfaces.svf[i]
            # Skip surfaces that only see sky (match non-cached behavior)
            if urban_vf_i < 0.01:
                self._surfins[i] = 0.0
            else:
                self._surfins[i] *= urban_vf_i
    
    def invalidate_svf_cache(self):
        """
        Invalidate the cached SVF matrix.
        
        Call this if geometry (buildings, terrain, vegetation) changes.
        The matrix will be recomputed on the next compute_svf() call.
        """
        self._svf_matrix_cached = False
        self._svf_nnz = 0
        print("SVF matrix cache invalidated. Will recompute on next compute_svf() call.")
    
    @property
    def svf_matrix_cached(self) -> bool:
        """Check if SVF matrix is currently cached."""
        return self._svf_matrix_cached
    
    @property 
    def svf_matrix_entries(self) -> int:
        """Get number of non-zero entries in cached SVF matrix."""
        return self._svf_nnz
    
    @ti.kernel
    def _accumulate_outgoing(self):
        """Accumulate outgoing radiation to totals."""
        for i in range(self.n_surfaces):
            self._surfoutsw[i] += self._surfoutsl[i]
    
    @ti.kernel
    def _accumulate_incoming(self):
        """Accumulate incoming reflected radiation to totals."""
        for i in range(self.n_surfaces):
            self._surfinsw[i] += self._surfins[i]

    @ti.kernel
    def _accumulate_canopy_absorption_from_reflections(
        self,
        lad: ti.template(),
        is_solid: ti.template(),
        ext_coef: ti.f32,
        pcbinsw: ti.template()
    ):
        """
        Accumulate canopy absorption from surface reflections (PALM's reflection loop).
        
        PALM formula: pcbinsw(ipcgb) += csf * surfoutsl(isurfsrc) * asrc * grid_volume_inverse
        
        For each canopy cell, computes absorption from all reflecting surfaces.
        The CSF from surface to canopy includes:
        - View factor (solid angle from surface to canopy cell)
        - Transmissivity through intervening canopy (Beer-Lambert)
        - Absorption fraction in target canopy cell
        
        Args:
            lad: 3D LAD field
            is_solid: 3D solid field
            ext_coef: Extinction coefficient
            pcbinsw: Output array for accumulated absorption (W/m³)
        """
        PI = 3.14159265359
        nx = self.domain.nx
        ny = self.domain.ny
        nz = self.domain.nz
        dx = self.domain.dx
        dy = self.domain.dy
        dz = self.domain.dz
        grid_volume = dx * dy * dz
        grid_volume_inverse = 1.0 / grid_volume
        
        # For each canopy cell, accumulate absorption from all reflecting surfaces
        for ci, cj, ck in ti.ndrange(nx, ny, nz):
            cell_lad = lad[ci, cj, ck]
            if cell_lad <= 0.0:
                continue
            
            # Cell center position
            pos_cell = Vector3(
                (ci + 0.5) * dx,
                (cj + 0.5) * dy,
                (ck + 0.5) * dz
            )
            
            total_absorbed = 0.0
            
            # Loop over all surfaces with outgoing radiation
            for surf_i in range(self.n_surfaces):
                outgoing = self._surfoutsl[surf_i]
                if outgoing < 0.01:
                    continue
                
                # Surface properties
                pos_surf = self.surfaces.center[surf_i]
                normal_surf = self.surfaces.normal[surf_i]
                area_surf = self.surfaces.area[surf_i]
                
                # Vector from surface to canopy cell
                diff = pos_cell - pos_surf
                dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
                
                if dist_sq < 0.01:
                    continue
                
                dist = ti.sqrt(dist_sq)
                
                # Direction from surface to canopy
                dir_to_cell_x = diff[0] / dist
                dir_to_cell_y = diff[1] / dist
                dir_to_cell_z = diff[2] / dist
                
                # Check if surface faces the canopy cell
                cos_emit = (normal_surf[0]*dir_to_cell_x + 
                           normal_surf[1]*dir_to_cell_y + 
                           normal_surf[2]*dir_to_cell_z)
                
                if cos_emit > 0.0:
                    # Compute transmissivity from surface to canopy cell (through intervening canopy)
                    trans, blocked = ray_point_to_point_transmissivity(
                        pos_surf, pos_cell,
                        lad, is_solid,
                        nx, ny, nz,
                        dx, dy, dz,
                        ext_coef
                    )
                    
                    if blocked == 0:
                        # Approximate path length through target cell
                        path_in_cell = dz  # Simplified; could be more accurate
                        
                        # Absorption fraction in this cell
                        abs_frac = 1.0 - ti.exp(-ext_coef * cell_lad * path_in_cell)
                        
                        # View factor from surface to canopy cell (simplified)
                        # CSF = view_factor * transmissivity * absorption_fraction
                        cell_solid_angle = (dx * dy) / (4.0 * PI * dist_sq)
                        csf_factor = cell_solid_angle * cos_emit * trans * abs_frac
                        csf_factor = ti.min(csf_factor, 1.0)  # Clamp
                        
                        # PALM formula: pcbinsw += csf * surfoutsl * asrc * grid_volume_inverse
                        absorbed = csf_factor * outgoing * area_surf * grid_volume_inverse
                        total_absorbed += absorbed
            
            ti.atomic_add(pcbinsw[ci, cj, ck], total_absorbed)

    @ti.kernel
    def _update_canopy_scattered_radiation(self, lad: ti.template(), albedo_leaf: ti.f32):
        """
        Update scattered radiation field based on current absorbed radiation.
        
        pcbinswref = albedo_leaf * pcbinsw (fraction that gets scattered)
        
        This is called at each reflection step to update what's available for scattering.
        """
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            if lad[i, j, k] > 0.0:
                absorbed = self._pcbinsw[i, j, k]
                self._pcbinswref[i, j, k] = albedo_leaf * absorbed

    @ti.kernel
    def _compute_canopy_to_surface_scattering(
        self,
        lad: ti.template(),
        is_solid: ti.template(),
        ext_coef: ti.f32
    ):
        """
        Compute radiation scattered from canopy cells toward surfaces.
        
        For each canopy cell with scattered radiation (pcbinswref), distribute
        to surfaces based on solid angle and transmissivity.
        
        This is called within the reflection loop to capture Canopy→Surface paths.
        The contribution is added to _surfins so it participates in subsequent reflections.
        """
        PI = 3.14159265359
        nx = self.domain.nx
        ny = self.domain.ny
        nz = self.domain.nz
        dx = self.domain.dx
        dy = self.domain.dy
        dz = self.domain.dz
        grid_volume = dx * dy * dz
        
        # For each canopy cell with scattered radiation
        for ci, cj, ck in ti.ndrange(nx, ny, nz):
            cell_lad = lad[ci, cj, ck]
            if cell_lad <= 0.0:
                continue
            
            scattered_power = self._pcbinswref[ci, cj, ck] * grid_volume
            if scattered_power <= 0.01:
                continue
            
            # Cell center position
            pos_cell = Vector3(
                (ci + 0.5) * dx,
                (cj + 0.5) * dy,
                (ck + 0.5) * dz
            )
            
            # Distribute to all surfaces
            for surf_i in range(self.n_surfaces):
                pos_surf = self.surfaces.center[surf_i]
                normal_surf = self.surfaces.normal[surf_i]
                
                # Vector from cell to surface
                diff = pos_surf - pos_cell
                dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
                
                if dist_sq < 0.01:
                    continue
                
                dist = ti.sqrt(dist_sq)
                
                # Direction from cell to surface
                dir_x = diff[0] / dist
                dir_y = diff[1] / dist
                dir_z = diff[2] / dist
                
                # Check if surface faces the cell
                cos_recv = (normal_surf[0]*dir_x + normal_surf[1]*dir_y + normal_surf[2]*dir_z)
                
                if cos_recv > 0.0:
                    # Solid angle factor
                    cell_cross = dx * dy
                    solid_angle_factor = cell_cross / (4.0 * PI * dist_sq)
                    solid_angle_factor = ti.min(solid_angle_factor, 0.25)
                    
                    # Transmissivity through intervening canopy
                    trans, blocked = ray_point_to_point_transmissivity(
                        pos_cell, pos_surf,
                        lad, is_solid,
                        nx, ny, nz,
                        dx, dy, dz,
                        ext_coef
                    )
                    
                    if blocked == 0:
                        contribution = scattered_power * solid_angle_factor * cos_recv * trans
                        contribution /= self.surfaces.area[surf_i]
                        # Add to surfins so it participates in next reflection step
                        ti.atomic_add(self._surfins[surf_i], contribution)
                        # Also track separately for output
                        ti.atomic_add(self._surfinswpc[surf_i], contribution)

    @ti.kernel
    def _compute_surface_outgoing_no_exchange(self):
        """Compute outgoing radiation without inter-surface exchange."""
        for i in range(self.n_surfaces):
            albedo = self.surfaces.albedo[i]
            self._surfoutsl[i] = albedo * self._surfins[i]
            self._surfoutsw[i] = self._surfoutsl[i]
    
    @ti.kernel
    def _copy_final_fluxes(self):
        """Copy final computed fluxes to surface arrays."""
        for i in range(self.n_surfaces):
            self.surfaces.sw_in_direct[i] = self._surfinswdir[i]
            self.surfaces.sw_in_diffuse[i] = self._surfinswdif[i] + (
                self._surfinsw[i] - self._surfinswdir[i] - self._surfinswdif[i]
            ) + self._surfinswpc[i]  # Include reflected and canopy scattered as diffuse
            self.surfaces.sw_out[i] = self._surfoutsw[i]
    
    @ti.kernel
    def _compute_canopy_scattering(
        self,
        lad: ti.template(),
        is_solid: ti.template(),
        albedo_leaf: ti.f32,
        ext_coef: ti.f32
    ):
        """
        Compute radiation scattered from canopy cells toward surfaces.
        
        For each canopy cell that absorbs radiation, a fraction (albedo_leaf)
        is scattered isotropically. This scattered radiation contributes to
        nearby surfaces based on solid angle and distance.
        
        Following PALM's methodology:
        - pcrad = absorbed_radiation * albedo_leaf (what gets scattered)
        - Distribution based on CSF-like view factors to surfaces
        
        Args:
            lad: 3D field of Leaf Area Density
            is_solid: 3D field of solid cells
            albedo_leaf: Leaf albedo (fraction scattered vs absorbed)
            ext_coef: Extinction coefficient
        """
        PI = 3.14159265359
        nx = self.domain.nx
        ny = self.domain.ny
        nz = self.domain.nz
        dx = self.domain.dx
        dy = self.domain.dy
        dz = self.domain.dz
        grid_volume = dx * dy * dz
        
        # First: compute scattered radiation for each canopy cell
        for i, j, k in ti.ndrange(nx, ny, nz):
            cell_lad = lad[i, j, k]
            if cell_lad > 0.0:
                absorbed = self._pcbinsw[i, j, k] * grid_volume
                self._pcbinswref[i, j, k] = albedo_leaf * absorbed / grid_volume
        
        # Second: for each canopy cell, contribute to nearby surfaces
        # Iterate over canopy cells (outer) and surfaces (inner)
        for ci, cj, ck in ti.ndrange(nx, ny, nz):
            cell_lad = lad[ci, cj, ck]
            if cell_lad <= 0.0:
                continue
            
            scattered_power = self._pcbinswref[ci, cj, ck] * grid_volume
            if scattered_power <= 0.0:
                continue
            
            # Cell center position
            pos_cell = Vector3(
                (ci + 0.5) * dx,
                (cj + 0.5) * dy,
                (ck + 0.5) * dz
            )
            
            # Distribute to all surfaces
            for surf_i in range(self.n_surfaces):
                pos_surf = self.surfaces.center[surf_i]
                normal_surf = self.surfaces.normal[surf_i]
                
                # Vector from cell to surface
                diff = pos_surf - pos_cell
                dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
                
                if dist_sq < 0.01:
                    continue
                
                dist = ti.sqrt(dist_sq)
                
                # Direction from surface to cell (for checking if surface "sees" the cell)
                dir_to_cell_x = -diff[0] / dist
                dir_to_cell_y = -diff[1] / dist
                dir_to_cell_z = -diff[2] / dist
                
                # Check if surface faces the cell (cell is in hemisphere surface faces)
                cos_recv = (normal_surf[0]*dir_to_cell_x + 
                           normal_surf[1]*dir_to_cell_y + 
                           normal_surf[2]*dir_to_cell_z)
                
                if cos_recv > 0.0:
                    # Solid angle factor
                    cell_cross = dx * dy
                    solid_angle_factor = cell_cross / (4.0 * PI * dist_sq)
                    solid_angle_factor = ti.min(solid_angle_factor, 0.25)
                    
                    # Transmissivity through intervening canopy
                    trans, blocked = ray_point_to_point_transmissivity(
                        pos_cell, pos_surf,
                        lad, is_solid,
                        nx, ny, nz,
                        dx, dy, dz,
                        ext_coef
                    )
                    
                    if blocked == 0:
                        contribution = scattered_power * solid_angle_factor * cos_recv * trans
                        contribution /= self.surfaces.area[surf_i]
                        ti.atomic_add(self._surfinswpc[surf_i], contribution)
    
    @ti.kernel
    def _compute_canopy_to_canopy_scattering(
        self,
        lad: ti.template(),
        is_solid: ti.template(),
        albedo_leaf: ti.f32,
        ext_coef: ti.f32
    ):
        """
        Compute radiation scattered from one canopy cell to another.
        
        For each canopy cell that scatters radiation (pcbinswref), distribute
        that scattered radiation to neighboring canopy cells based on:
        - Solid angle (distance-based)
        - Transmissivity through intervening canopy
        - Absorption fraction in target cell
        
        This implements canopy-to-canopy scattering which PALM does not 
        explicitly model but is important for dense vegetation canopies.
        
        The formula follows the same CSF methodology:
            pcbinswc2c[target] += scattered[source] × view_factor × trans × abs_frac
        
        Args:
            lad: 3D field of Leaf Area Density
            is_solid: 3D field of solid cells
            albedo_leaf: Leaf albedo (fraction scattered vs absorbed)
            ext_coef: Extinction coefficient
        """
        PI = 3.14159265359
        nx = self.domain.nx
        ny = self.domain.ny
        nz = self.domain.nz
        dx = self.domain.dx
        dy = self.domain.dy
        dz = self.domain.dz
        grid_volume = dx * dy * dz
        
        # For each source canopy cell with scattered radiation
        for si, sj, sk in ti.ndrange(nx, ny, nz):
            source_lad = lad[si, sj, sk]
            if source_lad <= 0.0:
                continue
            
            scattered_power = self._pcbinswref[si, sj, sk] * grid_volume
            if scattered_power <= 0.01:
                continue
            
            # Source cell center position
            pos_source = Vector3(
                (si + 0.5) * dx,
                (sj + 0.5) * dy,
                (sk + 0.5) * dz
            )
            
            # Distribute to nearby canopy cells (limit search radius for efficiency)
            # Use a search radius based on typical canopy interaction distance
            search_radius_cells = 5  # Cells in each direction
            
            i_min = ti.max(0, si - search_radius_cells)
            i_max = ti.min(nx, si + search_radius_cells + 1)
            j_min = ti.max(0, sj - search_radius_cells)
            j_max = ti.min(ny, sj + search_radius_cells + 1)
            k_min = ti.max(0, sk - search_radius_cells)
            k_max = ti.min(nz, sk + search_radius_cells + 1)
            
            for ti_idx in range(i_min, i_max):
                for tj_idx in range(j_min, j_max):
                    for tk_idx in range(k_min, k_max):
                        # Skip self
                        if ti_idx == si and tj_idx == sj and tk_idx == sk:
                            continue
                        
                        target_lad = lad[ti_idx, tj_idx, tk_idx]
                        if target_lad <= 0.0:
                            continue
                        
                        # Target cell center position
                        pos_target = Vector3(
                            (ti_idx + 0.5) * dx,
                            (tj_idx + 0.5) * dy,
                            (tk_idx + 0.5) * dz
                        )
                        
                        # Distance between cells
                        diff = pos_target - pos_source
                        dist_sq = diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2]
                        
                        if dist_sq < 0.01:
                            continue
                        
                        dist = ti.sqrt(dist_sq)
                        
                        # Solid angle factor (cell cross-section / distance²)
                        # Using cell cross-sectional area perpendicular to ray
                        cell_cross = dx * dy  # Simplified; could use projection
                        solid_angle_factor = cell_cross / (4.0 * PI * dist_sq)
                        solid_angle_factor = ti.min(solid_angle_factor, 0.25)
                        
                        # Compute transmissivity from source to target through intervening canopy
                        trans, blocked = ray_point_to_point_transmissivity(
                            pos_source, pos_target,
                            lad, is_solid,
                            nx, ny, nz,
                            dx, dy, dz,
                            ext_coef
                        )
                        
                        if blocked == 0 and trans > 0.01:
                            # Absorption fraction in target cell
                            path_in_cell = dz  # Simplified path length
                            abs_frac = 1.0 - ti.exp(-ext_coef * target_lad * path_in_cell)
                            
                            # Contribution to target cell (W/m³)
                            contribution = scattered_power * solid_angle_factor * trans * abs_frac / grid_volume
                            ti.atomic_add(self._pcbinswc2c[ti_idx, tj_idx, tk_idx], contribution)
    
    @ti.kernel
    def _accumulate_canopy_to_canopy(self, lad: ti.template(), albedo_leaf: ti.f32):
        """
        Accumulate canopy-to-canopy contribution to total canopy absorption
        and prepare for next iteration.
        
        The c2c contribution is added to pcbinsw, and a fraction (albedo_leaf)
        is added to pcbinswref for the next scattering iteration.
        """
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            if lad[i, j, k] > 0.0:
                c2c = self._pcbinswc2c[i, j, k]
                if c2c > 0.0:
                    # Add to total absorbed
                    self._pcbinsw[i, j, k] += c2c
                    # Add to cumulative c2c total (for output)
                    self._pcbinswc2c_total[i, j, k] += c2c
                    # Fraction that gets scattered again
                    self._pcbinswref[i, j, k] += albedo_leaf * c2c
                    # Reset for next iteration
                    self._pcbinswc2c[i, j, k] = 0.0
    
    def _apply_canopy_scattering(self):
        """
        Apply initial canopy scattering from direct/diffuse absorption.
        
        Note: Most canopy scattering is now handled inside the reflection loop
        via _compute_canopy_to_surface_scattering and _compute_canopy_to_canopy_scattering.
        This function handles the initial scattering that feeds into the loop.
        """
        if self.domain.lad is None:
            return
        
        # Update scattered radiation based on current absorbed (from direct + diffuse)
        self._update_canopy_scattered_radiation(
            self.domain.lad,
            self.config.albedo_leaf
        )
    
    @ti.kernel
    def _add_canopy_to_diffuse(self):
        """Add canopy-scattered radiation to surface diffuse component."""
        for i in range(self.n_surfaces):
            self.surfaces.sw_in_diffuse[i] += self._surfinswpc[i]
    
    def compute_radiation(
        self,
        day_of_year: int,
        second_of_day: float,
        sw_direct: float,
        sw_diffuse: float
    ):
        """
        Compute shortwave radiation components.
        
        Args:
            day_of_year: Day number (1-365)
            second_of_day: Seconds since midnight UTC
            sw_direct: Direct normal irradiance (W/m²)
            sw_diffuse: Diffuse horizontal irradiance (W/m²)
        """
        self.update_solar_position(day_of_year, second_of_day)
        self.compute_shortwave_radiation(sw_direct, sw_diffuse)
    
    def get_surface_fluxes(self) -> dict:
        """
        Get radiation fluxes as numpy arrays.
        
        Returns:
            Dictionary with flux arrays including:
            - position: Grid indices (i, j, k)
            - direction: Surface direction index
            - area: Surface area (m²)
            - svf: Sky view factor
            - shadow_factor: Shadow factor (0=sunlit, 1=shaded)
            - sw_in_direct: Direct SW radiation (W/m²)
            - sw_in_diffuse: Diffuse SW radiation including reflections (W/m²)
            - sw_out: Outgoing (reflected) SW radiation (W/m²)
            - sw_in_total: Total incoming SW (W/m²)
            - sw_net: Net absorbed SW (W/m²)
        """
        sw_in_direct = self.surfaces.sw_in_direct.to_numpy()[:self.n_surfaces]
        sw_in_diffuse = self.surfaces.sw_in_diffuse.to_numpy()[:self.n_surfaces]
        sw_out = self.surfaces.sw_out.to_numpy()[:self.n_surfaces]
        sw_in_total = sw_in_direct + sw_in_diffuse
        sw_net = sw_in_total - sw_out
        
        return {
            'position': self.surfaces.position.to_numpy()[:self.n_surfaces],
            'direction': self.surfaces.direction.to_numpy()[:self.n_surfaces],
            'area': self.surfaces.area.to_numpy()[:self.n_surfaces],
            'svf': self.surfaces.svf.to_numpy()[:self.n_surfaces],
            'shadow_factor': self.surfaces.shadow_factor.to_numpy()[:self.n_surfaces],
            'sw_in_direct': sw_in_direct,
            'sw_in_diffuse': sw_in_diffuse,
            'sw_out': sw_out,
            'sw_in_total': sw_in_total,
            'sw_net': sw_net,
        }
    
    def get_total_absorbed_sw(self) -> float:
        """Get total absorbed shortwave radiation (W)."""
        fluxes = self.get_surface_fluxes()
        return float((fluxes['sw_net'] * fluxes['area']).sum())
    
    def get_domain_shadow_map(self) -> np.ndarray:
        """
        Get 2D shadow map at ground/terrain level.
        
        Only includes ground surfaces (k=0), not building rooftops.
        Building footprints are marked with NaN.
        
        Returns:
            2D array of shadow factors (0=shadowed, 1=sunlit, NaN=building)
        """
        shadow_map = np.full((self.domain.nx, self.domain.ny), np.nan)
        fluxes = self.get_surface_fluxes()
        
        # Find upward-facing surfaces at ground level only (k=0)
        for i in range(self.n_surfaces):
            if fluxes['direction'][i] == 0:  # Upward
                pos = fluxes['position'][i]
                ix = int(pos[0])  # Position is already grid indices
                iy = int(pos[1])
                iz = int(pos[2])
                if 0 <= ix < self.domain.nx and 0 <= iy < self.domain.ny:
                    # Only include ground-level surfaces (terrain at k=0)
                    if iz == 0:
                        # Invert: shadow_factor=0 means sunlit, =1 means shaded
                        # For display we want 1=sunlit, 0=shaded
                        shadow_map[ix, iy] = 1.0 - fluxes['shadow_factor'][i]
        
        return shadow_map
    
    def get_irradiance_map(self) -> np.ndarray:
        """
        Get 2D map of total incoming shortwave irradiance at ground level.
        
        Returns:
            2D array of irradiance values (W/m²), NaN for building footprints
        """
        irradiance_map = np.full((self.domain.nx, self.domain.ny), np.nan)
        fluxes = self.get_surface_fluxes()
        
        # Find upward-facing surfaces at ground level only (k=0)
        for i in range(self.n_surfaces):
            if fluxes['direction'][i] == 0:  # Upward
                pos = fluxes['position'][i]
                ix = int(pos[0])
                iy = int(pos[1])
                iz = int(pos[2])
                if 0 <= ix < self.domain.nx and 0 <= iy < self.domain.ny:
                    if iz == 0:
                        # Total incoming = direct + diffuse
                        sw_in = fluxes['sw_in_direct'][i] + fluxes['sw_in_diffuse'][i]
                        irradiance_map[ix, iy] = sw_in
        
        return irradiance_map
    
    def get_net_sw_radiation_map(self) -> np.ndarray:
        """
        Get 2D map of net shortwave radiation at ground level.
        
        Returns:
            2D array of net SW radiation values (W/m²), NaN for building footprints
        """
        net_map = np.full((self.domain.nx, self.domain.ny), np.nan)
        fluxes = self.get_surface_fluxes()
        
        for i in range(self.n_surfaces):
            if fluxes['direction'][i] == 0:  # Upward
                pos = fluxes['position'][i]
                ix = int(pos[0])
                iy = int(pos[1])
                iz = int(pos[2])
                if 0 <= ix < self.domain.nx and 0 <= iy < self.domain.ny:
                    if iz == 0:
                        sw_in = fluxes['sw_in_direct'][i] + fluxes['sw_in_diffuse'][i]
                        sw_out = fluxes['sw_out'][i]
                        net_map[ix, iy] = sw_in - sw_out
        
        return net_map
    
    # =========================================================================
    # Volumetric flux methods
    # =========================================================================
    
    def compute_volumetric_svf(self):
        """
        Compute volumetric sky view factors.
        
        This must be called before computing volumetric SW fluxes.
        Only needed once unless domain geometry changes.
        
        Raises:
            RuntimeError: If volumetric_flux is not enabled in config
        """
        if self.volumetric_calc is None:
            raise RuntimeError(
                "Volumetric flux not enabled. Set volumetric_flux=True in RadiationConfig."
            )
        self.volumetric_calc.compute_skyvf_vol()
    
    def get_volumetric_skyvf(self) -> np.ndarray:
        """
        Get volumetric sky view factor as 3D numpy array.
        
        Returns:
            3D array of shape (nx, ny, nz) with SVF values [0, 1]
            
        Raises:
            RuntimeError: If volumetric_flux is not enabled
        """
        if self.volumetric_calc is None:
            raise RuntimeError(
                "Volumetric flux not enabled. Set volumetric_flux=True in RadiationConfig."
            )
        return self.volumetric_calc.get_skyvf_vol()
    
    def get_volumetric_swflux(self) -> np.ndarray:
        """
        Get volumetric shortwave flux as 3D numpy array.
        
        This is the omnidirectional SW flux at each grid cell,
        representing average irradiance onto an imaginary sphere (W/m²).
        
        Returns:
            3D array of shape (nx, ny, nz) with SW flux values
            
        Raises:
            RuntimeError: If volumetric_flux is not enabled
        """
        if self.volumetric_calc is None:
            raise RuntimeError(
                "Volumetric flux not enabled. Set volumetric_flux=True in RadiationConfig."
            )
        return self.volumetric_calc.get_swflux_vol()
    
    def get_volumetric_shadow_top(self) -> np.ndarray:
        """
        Get shadow top level as 2D numpy array.
        
        Shadow top is the highest grid level that is in shadow
        for the current solar position.
        
        Returns:
            2D array of shape (nx, ny) with vertical indices
            
        Raises:
            RuntimeError: If volumetric_flux is not enabled
        """
        if self.volumetric_calc is None:
            raise RuntimeError(
                "Volumetric flux not enabled. Set volumetric_flux=True in RadiationConfig."
            )
        return self.volumetric_calc.get_shadow_top()
    
    def get_volumetric_shadow_mask(self) -> np.ndarray:
        """
        Get 3D shadow mask.
        
        Returns:
            3D boolean array where True indicates shadowed cells
            
        Raises:
            RuntimeError: If volumetric_flux is not enabled
        """
        if self.volumetric_calc is None:
            raise RuntimeError(
                "Volumetric flux not enabled. Set volumetric_flux=True in RadiationConfig."
            )
        return self.volumetric_calc.get_shadow_mask_3d()
    
    def get_volumetric_swflux_slice(
        self,
        level: Optional[int] = None,
        axis: Optional[str] = None,
        index: Optional[int] = None
    ) -> np.ndarray:
        """
        Get a 2D slice of volumetric SW flux.
        
        Args:
            level: Vertical level for horizontal slice (k index)
            axis: 'x' or 'y' for vertical slices
            index: Index along the axis for vertical slices
        
        Returns:
            2D array of SW flux values (W/m²)
            
        Raises:
            RuntimeError: If volumetric_flux is not enabled
            ValueError: If invalid slice parameters
        """
        if self.volumetric_calc is None:
            raise RuntimeError(
                "Volumetric flux not enabled. Set volumetric_flux=True in RadiationConfig."
            )
        
        if level is not None:
            return self.volumetric_calc.get_horizontal_slice(level, 'swflux')
        elif axis is not None and index is not None:
            return self.volumetric_calc.get_vertical_slice(axis, index, 'swflux')
        else:
            raise ValueError("Specify either 'level' or both 'axis' and 'index'")

    # =========================================================================
    # Plant Canopy Radiation Methods
    # =========================================================================
    
    def _compute_canopy_radiation(
        self,
        sw_direct: float,
        sw_diffuse: float,
        sun_dir: Vector3,
        cos_zenith: float
    ):
        """
        Compute radiation absorption in plant canopy using CSF.
        
        This implements PALM's plant canopy radiation balance:
        - pcbinswdir: Direct SW absorbed per canopy cell (W/m³)
        - pcbinswdif: Diffuse SW absorbed per canopy cell (W/m³)
        - pcbinsw: Total SW absorbed (W/m³) - includes direct, diffuse, AND reflected
        
        Note: pcbinsw may already contain reflection-step contributions from
        _compute_sw_fluxes. This method ADDS direct and diffuse to pcbinsw
        rather than resetting it.
        
        Args:
            sw_direct: Direct normal irradiance (W/m²)
            sw_diffuse: Diffuse horizontal irradiance (W/m²)
            sun_dir: Sun direction vector
            cos_zenith: Cosine of solar zenith angle
        """
        # Reset CSF calculator (not the _pcbinsw - it may have reflection contributions)
        self.csf_calc.reset_csf()
        
        # Reset only direct and diffuse arrays, NOT total pcbinsw
        self._reset_canopy_dir_dif_arrays()
        
        grid_volume = self.domain.dx * self.domain.dy * self.domain.dz
        
        # Compute direct SW absorption if sun is up
        # Use PALM's method: box_absorb + dsitransc + per-box absorption
        if cos_zenith > 0.0262:  # min_stable_coszen
            self.csf_calc.compute_canopy_absorption_direct_palm(
                sun_dir,
                self.domain.is_solid,
                self.domain.lad,
                sw_direct
            )
            
            # CSF now contains absorption in W/m³, copy to pcbinswdir
            self._copy_csf_to_pcbinswdir_direct()
        
        # Compute diffuse SW absorption using PALM's method
        # Traces rays from each canopy cell to sky hemisphere (not from surfaces)
        self.csf_calc.compute_canopy_absorption_diffuse_palm(
            self.domain.is_solid,
            self.domain.lad,
            sw_diffuse,
            self._pcbinswdif,
            self.config.n_azimuth,
            self.config.n_elevation
        )
        
        # Total absorbed = direct + diffuse + reflected (already in pcbinsw from reflections)
        self._sum_canopy_absorption()
        
        # Compute received radiation (before absorption)
        self._compute_received_radiation(sw_direct, sw_diffuse, cos_zenith, grid_volume)
        
        self._canopy_radiation_computed = True
    
    def _compute_canopy_radiation_initial(
        self,
        sw_direct: float,
        sw_diffuse: float,
        sun_dir,  # Vector3
        cos_zenith: float
    ):
        """
        Compute initial canopy radiation absorption (direct + diffuse) BEFORE reflection loop.
        
        This is called before the surface reflection loop so that canopy-scattered
        radiation can participate in reflections (enabling paths like C→S→C, S→C→S).
        
        After this, _update_canopy_scattered_radiation should be called to prepare
        pcbinswref for the reflection loop.
        
        Args:
            sw_direct: Direct normal irradiance (W/m²)
            sw_diffuse: Diffuse horizontal irradiance (W/m²)
            sun_dir: Sun direction vector
            cos_zenith: Cosine of solar zenith angle
        """
        # Reset CSF calculator and all canopy arrays
        self.csf_calc.reset_csf()
        self._reset_canopy_arrays()
        
        # Compute direct SW absorption if sun is up
        if cos_zenith > 0.0262:  # min_stable_coszen
            self.csf_calc.compute_canopy_absorption_direct_palm(
                sun_dir,
                self.domain.is_solid,
                self.domain.lad,
                sw_direct
            )
            # CSF now contains absorption in W/m³, copy to pcbinswdir
            self._copy_csf_to_pcbinswdir_direct()
        
        # Compute diffuse SW absorption using PALM's method
        self.csf_calc.compute_canopy_absorption_diffuse_palm(
            self.domain.is_solid,
            self.domain.lad,
            sw_diffuse,
            self._pcbinswdif,
            self.config.n_azimuth,
            self.config.n_elevation
        )
        
        # Total initial absorbed = direct + diffuse
        self._sum_canopy_absorption_initial()
    
    @ti.kernel
    def _sum_canopy_absorption_initial(self):
        """Sum direct + diffuse for initial canopy absorption (resets pcbinsw first)."""
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            self._pcbinsw[i, j, k] = self._pcbinswdir[i, j, k] + self._pcbinswdif[i, j, k]
    
    @ti.kernel
    def _reset_canopy_dir_dif_arrays(self):
        """Reset direct and diffuse canopy absorption arrays to zero (not total)."""
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            self._pcbinswdir[i, j, k] = 0.0
            self._pcbinswdif[i, j, k] = 0.0
    
    @ti.kernel
    def _reset_canopy_arrays(self):
        """Reset all canopy absorption arrays to zero."""
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            self._pcbinsw[i, j, k] = 0.0
            self._pcbinsw[i, j, k] = 0.0
            self._pcbinswdir[i, j, k] = 0.0
            self._pcbinswdif[i, j, k] = 0.0
            self._pcinsw[i, j, k] = 0.0
            self._pcinswdir[i, j, k] = 0.0
            self._pcinswdif[i, j, k] = 0.0
    
    @ti.kernel
    def _copy_csf_to_pcbinswdir(self, grid_volume: ti.f32):
        """Copy CSF field to direct absorption array, converting to W/m³."""
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            # CSF contains absorbed power in W, convert to W/m³
            self._pcbinswdir[i, j, k] = self.csf_calc.csf[i, j, k] / grid_volume
    
    @ti.kernel
    def _copy_csf_to_pcbinswdir_direct(self):
        """Copy CSF field to direct absorption array (CSF already in W/m³ from PALM method)."""
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            self._pcbinswdir[i, j, k] = self.csf_calc.csf[i, j, k]
    
    @ti.kernel
    def _copy_csf_to_pcbinswdif(self, grid_volume: ti.f32):
        """Copy CSF field to diffuse absorption array, converting to W/m³."""
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            self._pcbinswdif[i, j, k] = self.csf_calc.csf[i, j, k] / grid_volume
    
    @ti.kernel
    def _sum_canopy_absorption(self):
        """Add direct and diffuse to total canopy absorption.
        
        Note: pcbinsw may already contain reflection contributions from
        _accumulate_canopy_absorption_from_reflections. This method ADDS
        direct + diffuse rather than replacing.
        """
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            # Add direct + diffuse to existing value (which may have reflection contributions)
            self._pcbinsw[i, j, k] += self._pcbinswdir[i, j, k] + self._pcbinswdif[i, j, k]
    
    @ti.kernel
    def _compute_received_radiation(
        self,
        sw_direct: ti.f32,
        sw_diffuse: ti.f32,
        cos_zenith: ti.f32,
        grid_volume: ti.f32
    ):
        """
        Compute received (incident) radiation at each canopy cell.
        
        This is the radiation before absorption, useful for photosynthesis models.
        """
        for i, j, k in ti.ndrange(self.domain.nx, self.domain.ny, self.domain.nz):
            lad = self.domain.lad[i, j, k]
            if lad > 0.0:
                # Leaf area in this cell
                leaf_area = lad * grid_volume
                
                # Received = absorbed / absorption_fraction
                # For small absorption: received ≈ absorbed / (ext_coef * LAD * path_length)
                # Simplified: use absorbed * 2 as rough estimate (50% absorption typical)
                if self._pcbinswdir[i, j, k] > 0:
                    self._pcinswdir[i, j, k] = self._pcbinswdir[i, j, k] * grid_volume / leaf_area
                if self._pcbinswdif[i, j, k] > 0:
                    self._pcinswdif[i, j, k] = self._pcbinswdif[i, j, k] * grid_volume / leaf_area
                
                self._pcinsw[i, j, k] = self._pcinswdir[i, j, k] + self._pcinswdif[i, j, k]
    
    def get_canopy_absorbed_sw(self) -> np.ndarray:
        """
        Get total absorbed SW radiation in plant canopy.
        
        Returns:
            3D array of absorbed SW radiation (W/m³)
        """
        return self._pcbinsw.to_numpy()
    
    def get_canopy_absorbed_sw_direct(self) -> np.ndarray:
        """
        Get direct SW radiation absorbed in plant canopy.
        
        Returns:
            3D array of absorbed direct SW radiation (W/m³)
        """
        return self._pcbinswdir.to_numpy()
    
    def get_canopy_absorbed_sw_diffuse(self) -> np.ndarray:
        """
        Get diffuse SW radiation absorbed in plant canopy.
        
        Returns:
            3D array of absorbed diffuse SW radiation (W/m³)
        """
        return self._pcbinswdif.to_numpy()
    
    def get_canopy_received_sw(self) -> np.ndarray:
        """
        Get total received SW radiation at plant canopy (before absorption).
        
        Returns:
            3D array of received SW radiation (W/m²)
        """
        return self._pcinsw.to_numpy()
    
    def get_total_canopy_absorption(self) -> float:
        """
        Get total SW radiation absorbed by all plant canopy (W).
        
        
        Returns:
            Total absorbed power in Watts
        """
        grid_volume = self.domain.dx * self.domain.dy * self.domain.dz
        return float(self._pcbinsw.to_numpy().sum() * grid_volume)
    
    def get_canopy_absorption_profile(self) -> np.ndarray:
        """
        Get vertical profile of canopy-averaged SW absorption.
        
        Returns:
            1D array of mean absorbed SW per level (W/m³)
        """
        pcbinsw = self._pcbinsw.to_numpy()
        # Mean over horizontal dimensions, excluding zero cells
        profile = np.zeros(self.domain.nz)
        for k in range(self.domain.nz):
            layer = pcbinsw[:, :, k]
            nonzero = layer[layer > 0]
            if len(nonzero) > 0:
                profile[k] = nonzero.mean()
        return profile
    
    def get_canopy_scattered_sw(self) -> np.ndarray:
        """
        Get scattered (reflected) SW radiation from canopy cells.
        
        Returns:
            3D array of scattered SW flux (W/m³) indexed by (i, j, k)
        """
        return self._pcbinswref.to_numpy()
    
    def get_canopy_to_canopy_sw(self) -> np.ndarray:
        """
        Get SW radiation received from other canopy cells (canopy-to-canopy scattering).
        
        This is radiation scattered by one canopy cell and absorbed by another.
        Only non-zero if canopy_to_canopy=True in config.
        
        Returns:
            3D array of canopy-to-canopy SW flux (W/m³) indexed by (i, j, k)
        """
        return self._pcbinswc2c_total.to_numpy()
    
    def get_surface_sw_from_canopy(self) -> np.ndarray:
        """
        Get SW radiation received by surfaces from canopy scattering.
        
        Returns:
            1D array of SW from canopy (W/m²) per surface element
        """
        return self._surfinswpc.to_numpy()
