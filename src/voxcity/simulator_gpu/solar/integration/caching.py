"""
Caching infrastructure for VoxCity solar integration module.

This module provides cache dataclasses and management functions for:
- RadiationModel (ground-level calculations)
- Building RadiationModel (building surface calculations)
- GPU Ray Tracer (simple ray-tracing)
- Volumetric Flux Calculator
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

from .utils import (
    get_location_from_voxcity,
    convert_voxel_data_to_arrays,
    compute_valid_ground_vectorized,
    VOXCITY_BUILDING_CODE,
)


# =============================================================================
# Data Classes for Caching
# =============================================================================

@dataclass
class LandCoverAlbedo:
    """
    Mapping of land cover classes to albedo values.
    
    Default values are based on literature values for typical urban materials.
    References:
    - Oke, T.R. (1987) Boundary Layer Climates
    - Sailor, D.J. (1995) Simulated urban climate response to modifications
    """
    # OpenStreetMap / Standard land cover classes
    bareland: float = 0.20
    rangeland: float = 0.25
    shrub: float = 0.20
    agriculture: float = 0.20
    tree: float = 0.15
    wetland: float = 0.12
    mangrove: float = 0.12
    water: float = 0.06
    snow_ice: float = 0.80
    developed: float = 0.20
    road: float = 0.12
    building_ground: float = 0.20
    
    # Building surfaces
    building_wall: float = 0.30
    building_roof: float = 0.25
    
    # Vegetation
    leaf: float = 0.15
    
    def get_land_cover_albedo(self, class_code: int) -> float:
        """Get albedo value for a land cover class code."""
        albedo_map = {
            0: self.bareland,
            1: self.rangeland,
            2: self.shrub,
            3: self.agriculture,
            4: self.tree,
            5: self.wetland,
            6: self.mangrove,
            7: self.water,
            8: self.snow_ice,
            9: self.developed,
            10: self.road,
            11: self.building_ground,
        }
        return albedo_map.get(class_code, self.developed)


@dataclass
class VoxCityDomainResult:
    """Result of VoxCity to palm_solar conversion."""
    domain: object
    surface_land_cover: Optional[np.ndarray] = None
    surface_material_type: Optional[np.ndarray] = None
    land_cover_albedo: Optional[LandCoverAlbedo] = None


@dataclass
class CachedRadiationModel:
    """Cached RadiationModel with associated metadata."""
    model: object
    valid_ground: np.ndarray
    ground_k: np.ndarray
    voxcity_shape: Tuple[int, int, int]
    meshsize: float
    n_reflection_steps: int
    grid_indices: Optional[np.ndarray] = None
    surface_indices: Optional[np.ndarray] = None
    positions_np: Optional[np.ndarray] = None
    directions_np: Optional[np.ndarray] = None


@dataclass
class CachedBuildingRadiationModel:
    """Cached RadiationModel for building surface calculations."""
    model: object
    voxcity_shape: Tuple[int, int, int]
    meshsize: float
    n_reflection_steps: int
    is_building_surf: np.ndarray
    building_svf_mesh: object
    bldg_indices: Optional[np.ndarray] = None
    mesh_to_surface_idx: Optional[np.ndarray] = None
    mesh_face_centers: Optional[np.ndarray] = None
    mesh_face_normals: Optional[np.ndarray] = None
    boundary_mask: Optional[np.ndarray] = None
    cached_building_mesh: object = None


@dataclass
class CachedGPURayTracer:
    """Cached Taichi fields for GPU ray tracing."""
    is_solid_field: object
    lad_field: object
    transmittance_field: object
    topo_top_field: object
    trace_rays_kernel: object
    voxel_shape: Tuple[int, int, int]
    meshsize: float
    voxel_data_id: int = 0


# =============================================================================
# Module-Level Caches
# =============================================================================

_radiation_model_cache: Optional[CachedRadiationModel] = None
_building_radiation_model_cache: Optional[CachedBuildingRadiationModel] = None
_gpu_ray_tracer_cache: Optional[CachedGPURayTracer] = None
_volumetric_flux_cache: Optional[Dict] = None
_cached_topo_kernel = None
_cached_trace_rays_kernel = None


# =============================================================================
# Cache Management Functions
# =============================================================================

def clear_radiation_model_cache():
    """Clear the cached RadiationModel to free memory or force recomputation."""
    global _radiation_model_cache
    _radiation_model_cache = None


def clear_building_radiation_model_cache():
    """Clear the cached Building RadiationModel to free memory."""
    global _building_radiation_model_cache
    _building_radiation_model_cache = None


def clear_gpu_ray_tracer_cache():
    """Clear the cached GPU ray tracer fields to free memory or force recomputation."""
    global _gpu_ray_tracer_cache
    _gpu_ray_tracer_cache = None


def clear_volumetric_flux_cache():
    """Clear the cached VolumetricFluxCalculator to free memory."""
    global _volumetric_flux_cache
    _volumetric_flux_cache = None


def clear_all_caches():
    """Clear all GPU caches (RadiationModel, Building RadiationModel, GPU ray tracer, Volumetric)."""
    global _radiation_model_cache, _building_radiation_model_cache, _gpu_ray_tracer_cache, _volumetric_flux_cache
    _radiation_model_cache = None
    _building_radiation_model_cache = None
    _gpu_ray_tracer_cache = None
    _volumetric_flux_cache = None


def clear_all_radiation_caches():
    """Clear all cached RadiationModels to free GPU memory."""
    clear_radiation_model_cache()
    clear_building_radiation_model_cache()


def reset_solar_taichi_cache():
    """
    Reset Taichi runtime and clear all solar caches.
    
    Call this function when you encounter:
    - CUDA_ERROR_OUT_OF_MEMORY errors
    - TaichiRuntimeError: FieldsBuilder finalized
    
    After calling this, the next solar calculation will create fresh
    Taichi fields.
    """
    global _radiation_model_cache, _building_radiation_model_cache, _gpu_ray_tracer_cache, _volumetric_flux_cache
    _radiation_model_cache = None
    _building_radiation_model_cache = None
    _gpu_ray_tracer_cache = None
    _volumetric_flux_cache = None
    
    import taichi as ti
    try:
        ti.reset()
        ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)
    except Exception:
        try:
            ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32)
        except Exception:
            pass


# =============================================================================
# Radiation Model Cache Access
# =============================================================================

def get_radiation_model_cache() -> Optional[CachedRadiationModel]:
    """Get the current radiation model cache."""
    return _radiation_model_cache


def set_radiation_model_cache(cache: CachedRadiationModel):
    """Set the radiation model cache."""
    global _radiation_model_cache
    _radiation_model_cache = cache


def get_building_radiation_model_cache() -> Optional[CachedBuildingRadiationModel]:
    """Get the current building radiation model cache."""
    return _building_radiation_model_cache


def set_building_radiation_model_cache(cache: CachedBuildingRadiationModel):
    """Set the building radiation model cache."""
    global _building_radiation_model_cache
    _building_radiation_model_cache = cache


def get_gpu_ray_tracer_cache() -> Optional[CachedGPURayTracer]:
    """Get the current GPU ray tracer cache."""
    return _gpu_ray_tracer_cache


def set_gpu_ray_tracer_cache(cache: CachedGPURayTracer):
    """Set the GPU ray tracer cache."""
    global _gpu_ray_tracer_cache
    _gpu_ray_tracer_cache = cache


def get_volumetric_flux_cache() -> Optional[Dict]:
    """Get the current volumetric flux cache."""
    return _volumetric_flux_cache


def set_volumetric_flux_cache(cache: Dict):
    """Set the volumetric flux cache."""
    global _volumetric_flux_cache
    _volumetric_flux_cache = cache


# =============================================================================
# Radiation Model Creation
# =============================================================================

def get_or_create_radiation_model(
    voxcity,
    n_reflection_steps: int = 2,
    progress_report: bool = False,
    **kwargs
) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Get cached RadiationModel or create a new one if cache is invalid.
    
    The SVF and CSF matrices are O(n²) to compute and only depend on geometry,
    not solar position. This function caches the model for reuse.
    
    Args:
        voxcity: VoxCity object
        n_reflection_steps: Number of reflection bounces
        progress_report: Print progress messages
        **kwargs: Additional RadiationConfig parameters
        
    Returns:
        Tuple of (RadiationModel, valid_ground array, ground_k array)
    """
    global _radiation_model_cache
    
    from ..radiation import RadiationModel, RadiationConfig
    from ..domain import Domain, IUP
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ni, nj, nk = voxel_data.shape
    
    # Check if cache is valid
    cache_valid = False
    if _radiation_model_cache is not None:
        cache = _radiation_model_cache
        if (cache.voxcity_shape == voxel_data.shape and
            cache.meshsize == meshsize and
            cache.n_reflection_steps == n_reflection_steps):
            cache_valid = True
            if progress_report:
                print("Using cached RadiationModel (SVF/CSF already computed)")
    
    if cache_valid:
        return (_radiation_model_cache.model, 
                _radiation_model_cache.valid_ground, 
                _radiation_model_cache.ground_k)
    
    # Need to create new model
    if progress_report:
        print("Creating new RadiationModel (computing SVF/CSF matrices)...")
    
    # Get location
    origin_lat, origin_lon = get_location_from_voxcity(voxcity)
    
    # Create domain
    domain = Domain(
        nx=ni, ny=nj, nz=nk,
        dx=meshsize, dy=meshsize, dz=meshsize,
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    # Convert VoxCity voxel data to domain arrays
    default_lad = kwargs.get('default_lad', 1.0)
    is_solid_np, lad_np = convert_voxel_data_to_arrays(voxel_data, default_lad)
    
    # Compute valid ground cells
    valid_ground, _ = compute_valid_ground_vectorized(voxel_data)
    
    # Set domain arrays
    _set_solid_array(domain, is_solid_np)
    domain.set_lad_from_array(lad_np)
    _update_topo_from_solid(domain)
    
    # Create RadiationModel
    config = RadiationConfig(
        n_reflection_steps=n_reflection_steps,
        n_azimuth=kwargs.get('n_azimuth', 40),
        n_elevation=kwargs.get('n_elevation', 10)
    )
    
    model = RadiationModel(domain, config)
    
    # Compute SVF
    if progress_report:
        print("Computing Sky View Factors...")
    model.compute_svf()
    
    # Pre-compute ground_k and surface mappings
    n_surfaces = model.surfaces.count
    positions = model.surfaces.position.to_numpy()[:n_surfaces]
    directions = model.surfaces.direction.to_numpy()[:n_surfaces]
    
    ground_k = np.full((ni, nj), -1, dtype=np.int32)
    for idx in range(n_surfaces):
        pos_i, pos_j, k = positions[idx]
        direction = directions[idx]
        if direction == IUP:
            ii, jj = int(pos_i), int(pos_j)
            if 0 <= ii < ni and 0 <= jj < nj:
                if not valid_ground[ii, jj]:
                    continue
                if ground_k[ii, jj] < 0 or k < ground_k[ii, jj]:
                    ground_k[ii, jj] = int(k)
    
    # Pre-compute surface-to-grid mapping
    if progress_report:
        print("Pre-computing surface-to-grid mapping...")
    surface_to_grid_map = {}
    for idx in range(n_surfaces):
        direction = directions[idx]
        if direction == IUP:
            pi = int(positions[idx, 0])
            pj = int(positions[idx, 1])
            pk = int(positions[idx, 2])
            if 0 <= pi < ni and 0 <= pj < nj:
                if valid_ground[pi, pj] and pk == ground_k[pi, pj]:
                    surface_to_grid_map[(pi, pj)] = idx
    
    # Convert to arrays
    if surface_to_grid_map:
        grid_indices = np.array(list(surface_to_grid_map.keys()), dtype=np.int32)
        surface_indices = np.array(list(surface_to_grid_map.values()), dtype=np.int32)
    else:
        grid_indices = np.empty((0, 2), dtype=np.int32)
        surface_indices = np.empty((0,), dtype=np.int32)
    
    # Cache the model
    _radiation_model_cache = CachedRadiationModel(
        model=model,
        valid_ground=valid_ground,
        ground_k=ground_k,
        voxcity_shape=voxel_data.shape,
        meshsize=meshsize,
        n_reflection_steps=n_reflection_steps,
        grid_indices=grid_indices,
        surface_indices=surface_indices,
        positions_np=positions,
        directions_np=directions
    )
    
    if progress_report:
        print(f"RadiationModel cached. Valid ground cells: {np.sum(valid_ground)}, mapped surfaces: {len(surface_indices)}")
    
    return model, valid_ground, ground_k


def get_or_create_building_radiation_model(
    voxcity,
    n_reflection_steps: int = 2,
    progress_report: bool = False,
    building_class_id: int = -3,
    **kwargs
) -> Tuple[object, np.ndarray]:
    """
    Get cached RadiationModel for building surfaces or create a new one.
    
    Args:
        voxcity: VoxCity object
        n_reflection_steps: Number of reflection bounces
        progress_report: Print progress messages
        building_class_id: Building voxel class code
        **kwargs: Additional RadiationConfig parameters
        
    Returns:
        Tuple of (RadiationModel, is_building_surf boolean array)
    """
    global _building_radiation_model_cache
    
    from ..radiation import RadiationModel, RadiationConfig
    from ..domain import Domain
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ny_vc, nx_vc, nz = voxel_data.shape
    
    # Check if cache is valid
    cache_valid = False
    if _building_radiation_model_cache is not None:
        cache = _building_radiation_model_cache
        if (cache.voxcity_shape == voxel_data.shape and
            cache.meshsize == meshsize):
            if n_reflection_steps == 0 or cache.n_reflection_steps > 0:
                cache_valid = True
                if progress_report:
                    print("Using cached Building RadiationModel (SVF/CSF already computed)")
    
    if cache_valid:
        return (_building_radiation_model_cache.model,
                _building_radiation_model_cache.is_building_surf)
    
    # Need to create new model
    if progress_report:
        print("Creating new Building RadiationModel (computing SVF/CSF matrices)...")
    
    # Get location
    origin_lat, origin_lon = get_location_from_voxcity(voxcity)
    
    ni, nj, nk = ny_vc, nx_vc, nz
    
    domain = Domain(
        nx=ni, ny=nj, nz=nk,
        dx=meshsize, dy=meshsize, dz=meshsize,
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    # Convert VoxCity voxel data
    default_lad = kwargs.get('default_lad', 2.0)
    is_solid_np, lad_np = convert_voxel_data_to_arrays(voxel_data, default_lad)
    
    # Set domain arrays
    _set_solid_array(domain, is_solid_np)
    domain.set_lad_from_array(lad_np)
    _update_topo_from_solid(domain)
    
    # When n_reflection_steps=0, disable surface reflections
    surface_reflections = n_reflection_steps > 0
    
    config = RadiationConfig(
        n_reflection_steps=n_reflection_steps,
        n_azimuth=40,
        n_elevation=10,
        surface_reflections=surface_reflections,
        cache_svf_matrix=surface_reflections,
    )
    
    model = RadiationModel(domain, config)
    
    if progress_report:
        print("Computing Sky View Factors...")
    model.compute_svf()
    
    # Pre-compute building surface mask
    n_surfaces = model.surfaces.count
    surf_positions_all = model.surfaces.position.to_numpy()[:n_surfaces]
    
    is_building_surf = np.zeros(n_surfaces, dtype=bool)
    for s_idx in range(n_surfaces):
        i_idx, j_idx, z_idx = surf_positions_all[s_idx]
        i, j, z = int(i_idx), int(j_idx), int(z_idx)
        if 0 <= i < ni and 0 <= j < nj and 0 <= z < nk:
            if voxel_data[i, j, z] == building_class_id:
                is_building_surf[s_idx] = True
    
    if progress_report:
        print(f"Building RadiationModel cached. Building surfaces: {np.sum(is_building_surf)}/{n_surfaces}")
    
    bldg_indices = np.where(is_building_surf)[0]
    
    _building_radiation_model_cache = CachedBuildingRadiationModel(
        model=model,
        voxcity_shape=voxel_data.shape,
        meshsize=meshsize,
        n_reflection_steps=n_reflection_steps,
        is_building_surf=is_building_surf,
        building_svf_mesh=None,
        bldg_indices=bldg_indices,
        mesh_to_surface_idx=None
    )
    
    return model, is_building_surf


# =============================================================================
# GPU Ray Tracer Creation
# =============================================================================

def get_or_create_gpu_ray_tracer(
    voxel_data: np.ndarray,
    meshsize: float,
    tree_lad: float = 1.0
) -> CachedGPURayTracer:
    """
    Get cached GPU ray tracer or create new one if cache is invalid.
    """
    global _gpu_ray_tracer_cache
    
    import taichi as ti
    from ...init_taichi import ensure_initialized
    ensure_initialized()
    
    nx, ny, nz = voxel_data.shape
    
    # Check if cache is valid
    if _gpu_ray_tracer_cache is not None:
        cache = _gpu_ray_tracer_cache
        if cache.voxel_shape == (nx, ny, nz) and cache.meshsize == meshsize:
            if cache.voxel_data_id == id(voxel_data):
                return cache
            
            # Data changed, re-upload
            is_solid, lad_array = convert_voxel_data_to_arrays(voxel_data, tree_lad)
            cache.is_solid_field.from_numpy(is_solid)
            cache.lad_field.from_numpy(lad_array)
            cache.voxel_data_id = id(voxel_data)
            
            _compute_topo_gpu(cache.is_solid_field, cache.topo_top_field, nz)
            return cache
    
    # Create new cache
    is_solid, lad_array = convert_voxel_data_to_arrays(voxel_data, tree_lad)
    
    is_solid_field = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    lad_field = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
    transmittance_field = ti.field(dtype=ti.f32, shape=(nx, ny))
    topo_top_field = ti.field(dtype=ti.i32, shape=(nx, ny))
    
    is_solid_field.from_numpy(is_solid)
    lad_field.from_numpy(lad_array)
    
    _compute_topo_gpu(is_solid_field, topo_top_field, nz)
    
    trace_rays_kernel = _get_cached_trace_rays_kernel()
    
    _gpu_ray_tracer_cache = CachedGPURayTracer(
        is_solid_field=is_solid_field,
        lad_field=lad_field,
        transmittance_field=transmittance_field,
        topo_top_field=topo_top_field,
        trace_rays_kernel=trace_rays_kernel,
        voxel_shape=(nx, ny, nz),
        meshsize=meshsize,
        voxel_data_id=id(voxel_data)
    )
    
    return _gpu_ray_tracer_cache


# =============================================================================
# Volumetric Calculator Creation
# =============================================================================

def get_or_create_volumetric_calculator(
    voxcity,
    n_azimuth: int = 36,
    n_zenith: int = 9,
    progress_report: bool = False,
    **kwargs
):
    """
    Get cached VolumetricFluxCalculator or create a new one if cache is invalid.
    """
    global _volumetric_flux_cache
    
    from ..volumetric import VolumetricFluxCalculator
    from ..domain import Domain
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ni, nj, nk = voxel_data.shape
    
    # Check if cache is valid
    cache_valid = False
    if _volumetric_flux_cache is not None:
        cache = _volumetric_flux_cache
        if (cache.get('voxcity_shape') == voxel_data.shape and
            cache.get('meshsize') == meshsize and
            cache.get('n_azimuth') == n_azimuth):
            cache_valid = True
            if progress_report:
                print("Using cached VolumetricFluxCalculator (SVF already computed)")
    
    if cache_valid:
        return _volumetric_flux_cache['calculator'], _volumetric_flux_cache['domain']
    
    if progress_report:
        print("Creating new VolumetricFluxCalculator...")
    
    origin_lat, origin_lon = get_location_from_voxcity(voxcity)
    
    domain = Domain(
        nx=ni, ny=nj, nz=nk,
        dx=meshsize, dy=meshsize, dz=meshsize,
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    default_lad = kwargs.get('default_lad', 1.0)
    is_solid_np, lad_np = convert_voxel_data_to_arrays(voxel_data, default_lad)
    
    _set_solid_array(domain, is_solid_np)
    domain.set_lad_from_array(lad_np)
    _update_topo_from_solid(domain)
    
    calculator = VolumetricFluxCalculator(
        domain,
        n_azimuth=n_azimuth,
        min_opaque_lad=kwargs.get('min_opaque_lad', 0.5)
    )
    
    if progress_report:
        print("Computing volumetric sky view factors...")
    calculator.compute_skyvf_vol(n_zenith=n_zenith)
    
    _volumetric_flux_cache = {
        'calculator': calculator,
        'domain': domain,
        'voxcity_shape': voxel_data.shape,
        'meshsize': meshsize,
        'n_azimuth': n_azimuth
    }
    
    if progress_report:
        print(f"VolumetricFluxCalculator cached.")
    
    return calculator, domain


# =============================================================================
# Internal Helper Functions
# =============================================================================

def _set_solid_array(domain, solid_array: np.ndarray) -> None:
    """Set domain solid cells from numpy array."""
    import taichi as ti
    from ...init_taichi import ensure_initialized
    ensure_initialized()
    
    @ti.kernel
    def _set_solid_kernel(domain: ti.template(), solid: ti.types.ndarray()):
        for i, j, k in domain.is_solid:
            domain.is_solid[i, j, k] = solid[i, j, k]
    
    _set_solid_kernel(domain, solid_array)


def _update_topo_from_solid(domain) -> None:
    """Update topography field from solid array."""
    import taichi as ti
    from ...init_taichi import ensure_initialized
    ensure_initialized()
    
    @ti.kernel
    def _update_topo_kernel(domain: ti.template()):
        for i, j in domain.topo_top:
            max_k = 0
            for k in range(domain.nz):
                if domain.is_solid[i, j, k] == 1:
                    max_k = k
            domain.topo_top[i, j] = max_k
    
    _update_topo_kernel(domain)


def _get_cached_topo_kernel():
    """Get or create cached topography kernel."""
    global _cached_topo_kernel
    if _cached_topo_kernel is not None:
        return _cached_topo_kernel
    
    import taichi as ti
    from ...init_taichi import ensure_initialized
    ensure_initialized()
    
    @ti.kernel
    def _topo_kernel(
        is_solid_f: ti.template(),
        topo_f: ti.template(),
        grid_nz: ti.i32
    ):
        for i, j in topo_f:
            # Find terrain surface: top of the first contiguous solid from below.
            # This gives the actual ground level, NOT the top of elevated
            # structures (bridges, overpasses) that may exist higher up.
            # Without this, columns under bridges would start ray tracing
            # above the bridge, missing the occlusion entirely.
            ground_k = -1
            for k in range(grid_nz):
                if is_solid_f[i, j, k] == 1:
                    ground_k = k
                else:
                    if ground_k >= 0:
                        break  # First air gap above contiguous ground
            topo_f[i, j] = ground_k
    
    _cached_topo_kernel = _topo_kernel
    return _cached_topo_kernel


def _compute_topo_gpu(is_solid_field, topo_top_field, nz: int):
    """Compute topography using GPU."""
    kernel = _get_cached_topo_kernel()
    kernel(is_solid_field, topo_top_field, nz)


def _get_cached_trace_rays_kernel():
    """Get or create cached ray tracing kernel."""
    global _cached_trace_rays_kernel
    if _cached_trace_rays_kernel is not None:
        return _cached_trace_rays_kernel
    
    import taichi as ti
    from ...init_taichi import ensure_initialized
    ensure_initialized()
    
    @ti.kernel
    def trace_rays_kernel(
        is_solid_f: ti.template(),
        lad_f: ti.template(),
        topo_f: ti.template(),
        trans_f: ti.template(),
        sun_x: ti.f32, sun_y: ti.f32, sun_z: ti.f32,
        vhk: ti.i32, ext: ti.f32,
        dx: ti.f32, step: ti.f32, max_dist: ti.f32,
        grid_nx: ti.i32, grid_ny: ti.i32, grid_nz: ti.i32
    ):
        for i, j in trans_f:
            ground_k = topo_f[i, j]
            start_k = ground_k + vhk
            if start_k < 0:
                start_k = 0
            if start_k >= grid_nz:
                start_k = grid_nz - 1
            
            while start_k < grid_nz - 1 and is_solid_f[i, j, start_k] == 1:
                start_k += 1
            
            if is_solid_f[i, j, start_k] == 1:
                trans_f[i, j] = 0.0
            else:
                ox = (float(i) + 0.5) * dx
                oy = (float(j) + 0.5) * dx
                oz = (float(start_k) + 0.5) * dx
                
                trans = 1.0
                t = step
                
                while t < max_dist and trans > 0.001:
                    px = ox + sun_x * t
                    py = oy + sun_y * t
                    pz = oz + sun_z * t
                    
                    gi = int(px / dx)
                    gj = int(py / dx)
                    gk = int(pz / dx)
                    
                    if gi < 0 or gi >= grid_nx or gj < 0 or gj >= grid_ny:
                        break
                    if gk < 0 or gk >= grid_nz:
                        break
                    
                    if is_solid_f[gi, gj, gk] == 1:
                        trans = 0.0
                        break
                    
                    lad_val = lad_f[gi, gj, gk]
                    if lad_val > 0.0:
                        trans *= ti.exp(-ext * lad_val * step)
                    
                    t += step
                
                trans_f[i, j] = trans
    
    _cached_trace_rays_kernel = trace_rays_kernel
    return _cached_trace_rays_kernel


def compute_direct_transmittance_map_gpu(
    voxel_data: np.ndarray,
    sun_direction: Tuple[float, float, float],
    view_point_height: float,
    meshsize: float,
    tree_k: float = 0.6,
    tree_lad: float = 1.0
) -> np.ndarray:
    """
    Compute direct solar transmittance map using GPU ray tracing.
    """
    nx, ny, nz = voxel_data.shape
    
    cache = get_or_create_gpu_ray_tracer(voxel_data, meshsize, tree_lad)
    
    sun_dir_x = float(sun_direction[0])
    sun_dir_y = float(sun_direction[1])
    sun_dir_z = float(sun_direction[2])
    view_height_k = max(1, int(view_point_height / meshsize))
    step_size = meshsize * 0.5
    max_trace_dist = float(max(nx, ny, nz) * meshsize * 2)
    
    cache.trace_rays_kernel(
        cache.is_solid_field,
        cache.lad_field,
        cache.topo_top_field,
        cache.transmittance_field,
        sun_dir_x, sun_dir_y, sun_dir_z,
        view_height_k, tree_k,
        meshsize, step_size, max_trace_dist,
        nx, ny, nz
    )
    
    return cache.transmittance_field.to_numpy()


# =============================================================================
# Reset and Cleanup Functions
# =============================================================================

def reset_solar_taichi_cache():
    """
    Reset Taichi runtime and clear all solar caches.
    
    Call this function when you encounter:
    - CUDA_ERROR_OUT_OF_MEMORY errors
    - TaichiRuntimeError: FieldsBuilder finalized
    
    After calling this, the next solar calculation will create fresh
    Taichi fields.
    """
    global _radiation_model_cache, _building_radiation_model_cache
    global _gpu_ray_tracer_cache, _volumetric_flux_cache
    _radiation_model_cache = None
    _building_radiation_model_cache = None
    _gpu_ray_tracer_cache = None
    _volumetric_flux_cache = None
    
    import taichi as ti
    try:
        ti.reset()
        # Reinitialize Taichi after reset
        ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)
    except Exception:
        try:
            # Fallback to CPU if CUDA fails
            ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32)
        except Exception:
            pass  # Ignore if already initialized


def clear_all_radiation_caches():
    """Clear all cached RadiationModels to free GPU memory."""
    clear_radiation_model_cache()
    clear_building_radiation_model_cache()


# =============================================================================
# VoxCity Load/Convert Functions
# =============================================================================

def load_voxcity(filepath):
    """
    Load VoxCity data from pickle file.
    
    Attempts to use the voxcity package if available, otherwise
    loads as raw pickle with fallback handling.
    
    Args:
        filepath: Path to the VoxCity pickle file
        
    Returns:
        VoxCity object or dict containing the model data
    """
    import pickle
    from pathlib import Path
    
    filepath = Path(filepath)
    
    try:
        # Try using voxcity package loader
        from voxcity.generator.io import load_voxcity as voxcity_load
        return voxcity_load(str(filepath))
    except ImportError:
        # Fallback: load as raw pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Handle wrapper dict format (has 'voxcity' key)
        if isinstance(data, dict) and 'voxcity' in data:
            return data['voxcity']
        
        return data


def convert_voxcity_to_domain(
    voxcity_data,
    default_lad: float = 2.0,
    land_cover_albedo: Optional[LandCoverAlbedo] = None,
    origin_lat: Optional[float] = None,
    origin_lon: Optional[float] = None
) -> 'VoxCityDomainResult':
    """
    Convert VoxCity voxel grid to palm_solar Domain with material properties.
    
    Args:
        voxcity_data: VoxCity object or dict from load_voxcity()
        default_lad: Default Leaf Area Density for tree voxels (m²/m³)
        land_cover_albedo: Custom land cover to albedo mapping
        origin_lat: Override latitude (degrees)
        origin_lon: Override longitude (degrees)
        
    Returns:
        VoxCityDomainResult with Domain and material information
    """
    from ..domain import Domain
    
    if land_cover_albedo is None:
        land_cover_albedo = LandCoverAlbedo()
    
    # Extract data from VoxCity object or dict
    if hasattr(voxcity_data, 'voxels'):
        # New VoxCity dataclass format
        voxel_grid = voxcity_data.voxels.classes
        meshsize = voxcity_data.voxels.meta.meshsize
        land_cover_grid = voxcity_data.land_cover.classes
        extras = getattr(voxcity_data, 'extras', {})
        rectangle_vertices = extras.get('rectangle_vertices', None)
    else:
        # Legacy dict format
        voxel_grid = voxcity_data['voxcity_grid']
        meshsize = voxcity_data['meshsize']
        land_cover_grid = voxcity_data.get('land_cover_grid', None)
        rectangle_vertices = voxcity_data.get('rectangle_vertices', None)
    
    ny, nx, nz = voxel_grid.shape
    dx = dy = dz = float(meshsize)
    
    # Determine location
    if origin_lat is None or origin_lon is None:
        if rectangle_vertices is not None and len(rectangle_vertices) > 0:
            lons = [v[0] for v in rectangle_vertices]
            lats = [v[1] for v in rectangle_vertices]
            if origin_lon is None:
                origin_lon = np.mean(lons)
            if origin_lat is None:
                origin_lat = np.mean(lats)
        else:
            if origin_lat is None:
                origin_lat = 1.35
            if origin_lon is None:
                origin_lon = 103.82
    
    # Create palm_solar Domain
    domain = Domain(
        nx=nx, ny=ny, nz=nz,
        dx=dx, dy=dy, dz=dz,
        origin=(0.0, 0.0, 0.0),
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    # Create arrays for conversion
    is_solid_np = np.zeros((nx, ny, nz), dtype=np.int32)
    lad_np = np.zeros((nx, ny, nz), dtype=np.float32)
    surface_land_cover_grid = np.full((nx, ny), -1, dtype=np.int32)
    
    VOXCITY_GROUND_CODE = -1
    VOXCITY_TREE_CODE = -2
    
    # Convert from VoxCity [row, col, z] to palm_solar [x, y, z]
    for row in range(ny):
        for col in range(nx):
            x_idx = col
            y_idx = row
            
            if land_cover_grid is not None:
                lc_val = land_cover_grid[row, col]
                if lc_val > 0:
                    surface_land_cover_grid[x_idx, y_idx] = int(lc_val) - 1
                else:
                    surface_land_cover_grid[x_idx, y_idx] = 9
            
            for z in range(nz):
                voxel_val = voxel_grid[row, col, z]
                
                if voxel_val == VOXCITY_BUILDING_CODE:
                    is_solid_np[x_idx, y_idx, z] = 1
                elif voxel_val == VOXCITY_GROUND_CODE:
                    is_solid_np[x_idx, y_idx, z] = 1
                elif voxel_val == VOXCITY_TREE_CODE:
                    lad_np[x_idx, y_idx, z] = default_lad
                elif voxel_val > 0:
                    is_solid_np[x_idx, y_idx, z] = 1
    
    # Set domain arrays
    _set_solid_array(domain, is_solid_np)
    domain.set_lad_from_array(lad_np)
    _update_topo_from_solid(domain)
    
    return VoxCityDomainResult(
        domain=domain,
        surface_land_cover=surface_land_cover_grid,
        land_cover_albedo=land_cover_albedo
    )


def apply_voxcity_albedo(model, voxcity_result: VoxCityDomainResult) -> None:
    """
    Apply VoxCity land cover-based albedo values to radiation model surfaces.
    
    Args:
        model: RadiationModel instance (after surface extraction)
        voxcity_result: Result from convert_voxcity_to_domain()
    """
    import taichi as ti
    from ...init_taichi import ensure_initialized
    ensure_initialized()
    
    if voxcity_result.surface_land_cover is None:
        print("Warning: No land cover data available, using default albedos")
        return
    
    domain = voxcity_result.domain
    lc_grid = voxcity_result.surface_land_cover
    lc_albedo = voxcity_result.land_cover_albedo
    
    n_surfaces = model.surfaces.n_surfaces[None]
    max_surfaces = model.surfaces.max_surfaces
    positions = model.surfaces.position.to_numpy()[:n_surfaces]
    directions = model.surfaces.direction.to_numpy()[:n_surfaces]
    
    albedo_values = np.zeros(max_surfaces, dtype=np.float32)
    
    IUP = 0
    IDOWN = 1
    
    for idx in range(n_surfaces):
        i, j, k = positions[idx]
        direction = directions[idx]
        
        if direction == IUP:
            if k == 0 or k == 1:
                lc_code = lc_grid[i, j]
                if lc_code >= 0:
                    albedo_values[idx] = lc_albedo.get_land_cover_albedo(lc_code)
                else:
                    albedo_values[idx] = lc_albedo.developed
            else:
                albedo_values[idx] = lc_albedo.building_roof
        elif direction == IDOWN:
            albedo_values[idx] = lc_albedo.building_wall
        else:
            albedo_values[idx] = lc_albedo.building_wall
    
    model.surfaces.albedo.from_numpy(albedo_values)


def create_radiation_config_for_voxcity(
    land_cover_albedo: Optional[LandCoverAlbedo] = None,
    **kwargs
):
    """
    Create a RadiationConfig suitable for VoxCity simulations.
    
    Args:
        land_cover_albedo: Land cover albedo mapping (for reference)
        **kwargs: Additional RadiationConfig parameters
        
    Returns:
        RadiationConfig instance
    """
    from ..radiation import RadiationConfig
    
    if land_cover_albedo is None:
        land_cover_albedo = LandCoverAlbedo()
    
    defaults = {
        'albedo_ground': land_cover_albedo.developed,
        'albedo_wall': land_cover_albedo.building_wall,
        'albedo_roof': land_cover_albedo.building_roof,
        'albedo_leaf': land_cover_albedo.leaf,
        'n_azimuth': 40,
        'n_elevation': 10,
        'n_reflection_steps': 2,
    }
    
    defaults.update(kwargs)
    
    return RadiationConfig(**defaults)

