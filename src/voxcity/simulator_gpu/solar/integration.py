"""
VoxCity Integration Module for palm_solar

This module provides utilities for loading VoxCity models and converting them
to palm_solar Domain objects with proper material-specific albedo values.

VoxCity models contain:
- 3D voxel grids with building, tree, and ground information
- Land cover classification codes
- DEM (Digital Elevation Model) for terrain
- Building heights and IDs
- Tree canopy data

This module handles:
- Loading VoxCity pickle files
- Converting voxel grids to palm_solar Domain
- Mapping land cover classes to surface albedo values
- Creating surface material types for accurate radiation simulation
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

from .domain import Domain
from .radiation import RadiationConfig


# VoxCity voxel class codes (from voxcity/generator/voxelizer.py)
VOXCITY_GROUND_CODE = -1
VOXCITY_TREE_CODE = -2
VOXCITY_BUILDING_CODE = -3


# =============================================================================
# Common Helper Functions (reduces code duplication)
# =============================================================================

def _get_location_from_voxcity(voxcity, default_lat: float = 1.35, default_lon: float = 103.82) -> Tuple[float, float]:
    """
    Extract latitude/longitude from VoxCity object or return defaults.
    
    Args:
        voxcity: VoxCity object with extras containing rectangle_vertices
        default_lat: Default latitude if not found (Singapore)
        default_lon: Default longitude if not found (Singapore)
        
    Returns:
        Tuple of (origin_lat, origin_lon)
    """
    extras = getattr(voxcity, 'extras', None)
    if isinstance(extras, dict):
        rectangle_vertices = extras.get('rectangle_vertices', None)
    else:
        rectangle_vertices = None
    
    if rectangle_vertices is not None and len(rectangle_vertices) > 0:
        lons = [v[0] for v in rectangle_vertices]
        lats = [v[1] for v in rectangle_vertices]
        return np.mean(lats), np.mean(lons)
    
    return default_lat, default_lon


def _convert_voxel_data_to_arrays(
    voxel_data: np.ndarray,
    default_lad: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert VoxCity voxel codes to is_solid and LAD arrays using vectorized operations.
    
    This is 10-100x faster than triple-nested Python loops for large grids.
    
    Args:
        voxel_data: 3D array of VoxCity voxel class codes
        default_lad: Default Leaf Area Density for tree voxels (m²/m³)
        
    Returns:
        Tuple of (is_solid, lad) numpy arrays with same shape as voxel_data
    """
    # Vectorized solid detection: buildings (-3), ground (-1), or positive land cover codes
    is_solid = (
        (voxel_data == VOXCITY_BUILDING_CODE) |
        (voxel_data == VOXCITY_GROUND_CODE) |
        (voxel_data > 0)
    ).astype(np.int32)
    
    # Vectorized LAD assignment: only tree voxels (-2) have LAD
    lad = np.where(voxel_data == VOXCITY_TREE_CODE, default_lad, 0.0).astype(np.float32)
    
    return is_solid, lad


def _compute_valid_ground_vectorized(voxel_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute valid ground mask and ground k-levels using vectorized operations.
    
    Valid ground cells are those where:
    - The transition from solid to air/tree occurs
    - The solid below is not water (7,8,9) or building/underground (negative codes)
    
    Args:
        voxel_data: 3D array of VoxCity voxel class codes (ni, nj, nk)
        
    Returns:
        Tuple of (valid_ground 2D bool array, ground_k 2D int array)
        ground_k[i,j] = -1 means no valid ground found
    """
    ni, nj, nk = voxel_data.shape
    
    # Water/special class codes to exclude
    WATER_CLASSES = {7, 8, 9}
    AIR_OR_TREE = {0, VOXCITY_TREE_CODE}
    
    valid_ground = np.zeros((ni, nj), dtype=bool)
    ground_k = np.full((ni, nj), -1, dtype=np.int32)
    
    # Vectorize over k: find first transition from solid to air/tree
    # For each (i,j), scan upward to find the first air/tree cell above a solid cell
    for k in range(1, nk):
        # Current cell is air (0) or tree (-2)
        curr_is_air_or_tree = (voxel_data[:, :, k] == 0) | (voxel_data[:, :, k] == VOXCITY_TREE_CODE)
        
        # Cell below is NOT air or tree (i.e., it's solid)
        below_val = voxel_data[:, :, k - 1]
        below_is_solid = (below_val != 0) & (below_val != VOXCITY_TREE_CODE)
        
        # This is a transition point
        is_transition = curr_is_air_or_tree & below_is_solid
        
        # Only process cells that haven't been assigned yet
        unassigned = (ground_k == -1)
        new_transitions = is_transition & unassigned
        
        if not np.any(new_transitions):
            continue
        
        # Check validity: below is not water (7,8,9) and not negative (building/underground)
        below_is_water = (below_val == 7) | (below_val == 8) | (below_val == 9)
        below_is_negative = (below_val < 0)
        below_is_invalid = below_is_water | below_is_negative
        
        # Valid ground: transition point where below is valid
        valid_new = new_transitions & ~below_is_invalid
        invalid_new = new_transitions & below_is_invalid
        
        # Assign ground_k for valid transitions
        ground_k[valid_new] = k
        valid_ground[valid_new] = True
        
        # Mark invalid transitions so we don't process them again
        # (set ground_k to -2 temporarily to distinguish from unassigned)
        ground_k[invalid_new] = -2
    
    # Reset -2 markers back to -1 (no valid ground)
    ground_k[ground_k == -2] = -1
    
    return valid_ground, ground_k


def _filter_df_to_period(df, start_time: str, end_time: str, tz: float):
    """
    Filter weather DataFrame to specified time period and convert to UTC.
    
    Args:
        df: pandas DataFrame with datetime index
        start_time: Start time in format 'MM-DD HH:MM:SS'
        end_time: End time in format 'MM-DD HH:MM:SS'
        tz: Timezone offset in hours
        
    Returns:
        Tuple of (df_period_utc, df with hour_of_year column)
        
    Raises:
        ValueError: If time format is invalid or no data in period
    """
    from datetime import datetime
    import pytz
    
    try:
        start_dt = datetime.strptime(start_time, '%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time, '%m-%d %H:%M:%S')
    except ValueError as ve:
        raise ValueError("start_time and end_time must be in format 'MM-DD HH:MM:SS'") from ve
    
    # Add hour_of_year column
    df = df.copy()
    df['hour_of_year'] = (df.index.dayofyear - 1) * 24 + df.index.hour + 1
    
    # Calculate start/end hours
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1
    
    # Filter to period
    if start_hour <= end_hour:
        df_period = df[(df['hour_of_year'] >= start_hour) & (df['hour_of_year'] <= end_hour)]
    else:
        df_period = df[(df['hour_of_year'] >= start_hour) | (df['hour_of_year'] <= end_hour)]
    
    if df_period.empty:
        raise ValueError("No weather data in the specified period.")
    
    # Localize and convert to UTC
    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)
    
    return df_period_utc


def _load_epw_data(
    epw_file_path: Optional[str] = None,
    download_nearest_epw: bool = False,
    voxcity = None,
    **kwargs
) -> Tuple:
    """
    Load EPW weather data, optionally downloading the nearest file.
    
    Args:
        epw_file_path: Path to EPW file (required if download_nearest_epw=False)
        download_nearest_epw: If True, download nearest EPW based on location
        voxcity: VoxCity object (needed for location when downloading)
        **kwargs: Additional parameters (output_dir, max_distance, rectangle_vertices)
        
    Returns:
        Tuple of (df, lon, lat, tz) where df is the weather DataFrame
        
    Raises:
        ValueError: If EPW file not provided and download_nearest_epw=False
        ImportError: If required modules not available
    """
    rectangle_vertices = kwargs.get('rectangle_vertices', None)
    if rectangle_vertices is None and voxcity is not None:
        extras = getattr(voxcity, 'extras', None)
        if isinstance(extras, dict):
            rectangle_vertices = extras.get('rectangle_vertices', None)
    
    if download_nearest_epw:
        if rectangle_vertices is None:
            raise ValueError("rectangle_vertices required to download nearest EPW file")
        
        try:
            from voxcity.utils.weather import get_nearest_epw_from_climate_onebuilding
            lons = [coord[0] for coord in rectangle_vertices]
            lats = [coord[1] for coord in rectangle_vertices]
            center_lon = (min(lons) + max(lons)) / 2
            center_lat = (min(lats) + max(lats)) / 2
            output_dir = kwargs.get('output_dir', 'output')
            max_distance = kwargs.get('max_distance', 100)
            
            epw_file_path, weather_data, metadata = get_nearest_epw_from_climate_onebuilding(
                longitude=center_lon,
                latitude=center_lat,
                output_dir=output_dir,
                max_distance=max_distance,
                extract_zip=True,
                load_data=True
            )
        except ImportError:
            raise ImportError("VoxCity weather utilities required for EPW download")
    
    if not epw_file_path:
        raise ValueError("epw_file_path must be provided when download_nearest_epw is False")
    
    # Read EPW file
    try:
        from voxcity.utils.weather import read_epw_for_solar_simulation
        df, lon, lat, tz, elevation_m = read_epw_for_solar_simulation(epw_file_path)
    except ImportError:
        # Fallback to our EPW reader
        from .epw import read_epw_header, read_epw_solar_data
        location = read_epw_header(epw_file_path)
        df = read_epw_solar_data(epw_file_path)
        lon, lat, tz = location.longitude, location.latitude, location.timezone
    
    if df.empty:
        raise ValueError("No data in EPW file.")
    
    return df, lon, lat, tz


def _compute_sun_direction(azimuth_degrees_ori: float, elevation_degrees: float) -> Tuple[float, float, float, float]:
    """
    Compute sun direction vector from azimuth and elevation angles.
    
    Args:
        azimuth_degrees_ori: Solar azimuth in VoxCity convention (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        
    Returns:
        Tuple of (sun_dir_x, sun_dir_y, sun_dir_z, cos_zenith)
    """
    # Convert from VoxCity convention to model coordinates
    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_radians = np.deg2rad(azimuth_degrees)
    elevation_radians = np.deg2rad(elevation_degrees)
    
    cos_elev = np.cos(elevation_radians)
    sin_elev = np.sin(elevation_radians)
    
    sun_dir_x = cos_elev * np.cos(azimuth_radians)
    sun_dir_y = cos_elev * np.sin(azimuth_radians)
    sun_dir_z = sin_elev
    cos_zenith = sin_elev  # cos(zenith) = sin(elevation)
    
    return sun_dir_x, sun_dir_y, sun_dir_z, cos_zenith


# =============================================================================
# Data Classes
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
    # OpenStreetMap / Standard land cover classes (0-indexed after +1 in voxelizer)
    # These map to land_cover_grid values in VoxCity
    bareland: float = 0.20          # Class 0: Bare soil/dirt
    rangeland: float = 0.25         # Class 1: Grassland/rangeland
    shrub: float = 0.20             # Class 2: Shrubland
    agriculture: float = 0.20       # Class 3: Agricultural land
    tree: float = 0.15              # Class 4: Tree cover (ground under canopy)
    wetland: float = 0.12           # Class 5: Wetland
    mangrove: float = 0.12          # Class 6: Mangrove
    water: float = 0.06             # Class 7: Water bodies
    snow_ice: float = 0.80          # Class 8: Snow and ice
    developed: float = 0.20         # Class 9: Developed/paved areas
    road: float = 0.12              # Class 10: Roads (asphalt)
    building_ground: float = 0.20   # Class 11: Building footprint area
    
    # Building surfaces (walls and roofs)
    building_wall: float = 0.30     # Vertical building surfaces
    building_roof: float = 0.25     # Building rooftops
    
    # Vegetation
    leaf: float = 0.15              # Plant canopy (PALM default)
    
    def get_land_cover_albedo(self, class_code: int) -> float:
        """
        Get albedo value for a land cover class code.
        
        Args:
            class_code: Land cover class code (0-11 for standard classes)
            
        Returns:
            Albedo value for the class
        """
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
        return albedo_map.get(class_code, self.developed)  # Default to developed


@dataclass
class VoxCityDomainResult:
    """Result of VoxCity to palm_solar conversion."""
    domain: Domain
    surface_land_cover: Optional[np.ndarray] = None  # Land cover code per surface
    surface_material_type: Optional[np.ndarray] = None  # 0=ground, 1=wall, 2=roof


# =============================================================================
# RadiationModel Caching for Cumulative Calculations
# =============================================================================
# The SVF and CSF matrices are geometry-dependent and expensive to compute.
# We cache the RadiationModel so it can be reused across multiple solar positions.

@dataclass
class _CachedRadiationModel:
    """Cached RadiationModel with associated metadata."""
    model: object  # RadiationModel instance
    valid_ground: np.ndarray  # Valid ground mask
    ground_k: np.ndarray  # Ground level k indices
    voxcity_shape: Tuple[int, int, int]  # Shape of voxel data for cache validation
    meshsize: float  # Meshsize for cache validation
    n_reflection_steps: int  # Number of reflection steps used
    # Performance optimization: pre-computed surface-to-grid mapping
    grid_indices: Optional[np.ndarray] = None  # (N, 2) array of (i, j) grid coords for valid ground surfaces
    surface_indices: Optional[np.ndarray] = None  # (N,) array of surface indices matching grid_indices
    # Cached numpy arrays (positions/directions don't change)
    positions_np: Optional[np.ndarray] = None  # Cached positions array
    directions_np: Optional[np.ndarray] = None  # Cached directions array


# Module-level cache for RadiationModel
_radiation_model_cache: Optional[_CachedRadiationModel] = None

# Module-level cache for GPU ray tracer (forward declaration, actual class defined later)
_gpu_ray_tracer_cache = None


def _get_or_create_radiation_model(
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
    
    from .radiation import RadiationModel, RadiationConfig
    from .domain import IUP
    
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
    
    # Get location using helper function
    origin_lat, origin_lon = _get_location_from_voxcity(voxcity)
    
    # Create domain
    domain = Domain(
        nx=ni, ny=nj, nz=nk,
        dx=meshsize, dy=meshsize, dz=meshsize,
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    # Convert VoxCity voxel data to domain arrays using vectorized helper
    default_lad = kwargs.get('default_lad', 1.0)
    is_solid_np, lad_np = _convert_voxel_data_to_arrays(voxel_data, default_lad)
    
    # Compute valid ground cells using vectorized helper
    valid_ground, _ = _compute_valid_ground_vectorized(voxel_data)
    
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
    
    # Compute SVF (this is the expensive part)
    if progress_report:
        print("Computing Sky View Factors...")
    model.compute_svf()
    
    # Pre-compute ground_k for surface mapping
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
    
    # Pre-compute surface-to-grid mapping for fast vectorized extraction
    # This maps which surface indices correspond to which grid cells
    if progress_report:
        print("Pre-computing surface-to-grid mapping...")
    surface_to_grid_map = {}  # (i, j) -> surface_idx
    for idx in range(n_surfaces):
        direction = directions[idx]
        if direction == IUP:
            pi = int(positions[idx, 0])
            pj = int(positions[idx, 1])
            pk = int(positions[idx, 2])
            if 0 <= pi < ni and 0 <= pj < nj:
                if valid_ground[pi, pj] and pk == ground_k[pi, pj]:
                    surface_to_grid_map[(pi, pj)] = idx
    
    # Convert to arrays for vectorized access
    if surface_to_grid_map:
        grid_indices = np.array(list(surface_to_grid_map.keys()), dtype=np.int32)
        surface_indices = np.array(list(surface_to_grid_map.values()), dtype=np.int32)
    else:
        grid_indices = np.empty((0, 2), dtype=np.int32)
        surface_indices = np.empty((0,), dtype=np.int32)
    
    # Cache the model with pre-computed mappings
    _radiation_model_cache = _CachedRadiationModel(
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


def clear_radiation_model_cache():
    """Clear the cached RadiationModel to free memory or force recomputation."""
    global _radiation_model_cache
    _radiation_model_cache = None


def _compute_ground_k_from_voxels(voxel_data: np.ndarray) -> np.ndarray:
    """
    Compute ground surface k-level for each (i,j) cell from voxel data.
    
    This finds the terrain top - the highest k where the cell below the first air
    cell is solid ground (not building). This is used for terrain-following
    height extraction in volumetric calculations.
    
    Water areas (voxel classes 7, 8, 9) and building/underground cells (negative codes)
    are excluded and marked as -1. This uses the same logic as the with_reflections=True
    path in _get_or_create_radiation_model for consistency.
    
    Args:
        voxel_data: 3D array of voxel class codes
        
    Returns:
        2D array of ground k-levels (ni, nj). -1 means no valid ground found.
    """
    # Use the vectorized helper for consistency
    _, ground_k = _compute_valid_ground_vectorized(voxel_data)
    return ground_k


def _extract_terrain_following_slice(
    flux_3d: np.ndarray,
    ground_k: np.ndarray,
    height_offset_k: int,
    is_solid: np.ndarray
) -> np.ndarray:
    """
    Extract a terrain-following 2D slice from a 3D flux field (vectorized).
    
    For each (i,j), extracts the value at ground_k[i,j] + height_offset_k.
    Cells that are solid at the extraction point, have no valid ground,
    or are above the domain are marked as NaN.
    
    Args:
        flux_3d: 3D array of flux values (ni, nj, nk)
        ground_k: 2D array of ground k-levels (ni, nj), -1 means no valid ground
        height_offset_k: Number of cells above ground to extract
        is_solid: 3D array marking solid cells (ni, nj, nk)
        
    Returns:
        2D array of extracted values (ni, nj) with NaN for invalid cells
    """
    ni, nj, nk = flux_3d.shape
    result = np.full((ni, nj), np.nan, dtype=np.float64)
    
    # Calculate extraction k-levels
    k_extract = ground_k + height_offset_k
    
    # Create valid mask: ground exists, within bounds, not solid
    valid_ground = ground_k >= 0
    within_bounds = k_extract < nk
    valid_mask = valid_ground & within_bounds
    
    # Get indices for valid cells
    ii, jj = np.where(valid_mask)
    kk = k_extract[valid_mask]
    
    # Check solid cells at extraction points
    not_solid = is_solid[ii, jj, kk] != 1
    
    # Extract values for non-solid cells
    ii_valid = ii[not_solid]
    jj_valid = jj[not_solid]
    kk_valid = kk[not_solid]
    
    result[ii_valid, jj_valid] = flux_3d[ii_valid, jj_valid, kk_valid]
    
    return result


def _accumulate_terrain_following_slice(
    cumulative_map: np.ndarray,
    flux_3d: np.ndarray,
    ground_k: np.ndarray,
    height_offset_k: int,
    is_solid: np.ndarray,
    weight: float = 1.0
) -> None:
    """
    Accumulate terrain-following values from a 3D flux field into a 2D map (vectorized, in-place).
    
    For each (i,j), adds flux_3d[i,j,k_extract] * weight to cumulative_map[i,j]
    where k_extract = ground_k[i,j] + height_offset_k.
    
    Args:
        cumulative_map: 2D array to accumulate into (ni, nj), modified in-place
        flux_3d: 3D array of flux values (ni, nj, nk)
        ground_k: 2D array of ground k-levels (ni, nj), -1 means no valid ground
        height_offset_k: Number of cells above ground to extract
        is_solid: 3D array marking solid cells (ni, nj, nk)
        weight: Multiplier for values before accumulating (e.g., time_step_hours)
    """
    ni, nj, nk = flux_3d.shape
    
    # Calculate extraction k-levels
    k_extract = ground_k + height_offset_k
    
    # Create valid mask: ground exists, within bounds
    valid_ground = ground_k >= 0
    within_bounds = k_extract < nk
    valid_mask = valid_ground & within_bounds
    
    # Get indices for valid cells
    ii, jj = np.where(valid_mask)
    kk = k_extract[valid_mask]
    
    # Check solid cells at extraction points
    not_solid = is_solid[ii, jj, kk] != 1
    
    # Accumulate for non-solid cells
    ii_valid = ii[not_solid]
    jj_valid = jj[not_solid]
    kk_valid = kk[not_solid]
    
    # Use np.add.at for proper in-place accumulation (handles duplicate indices)
    np.add.at(cumulative_map, (ii_valid, jj_valid), flux_3d[ii_valid, jj_valid, kk_valid] * weight)


def clear_gpu_ray_tracer_cache():
    """Clear the cached GPU ray tracer fields to free memory or force recomputation."""
    global _gpu_ray_tracer_cache
    _gpu_ray_tracer_cache = None


def clear_all_caches():
    """Clear all GPU caches (RadiationModel, Building RadiationModel, GPU ray tracer)."""
    global _radiation_model_cache, _building_radiation_model_cache, _gpu_ray_tracer_cache
    _radiation_model_cache = None
    _building_radiation_model_cache = None
    _gpu_ray_tracer_cache = None


# =============================================================================
# Building RadiationModel Caching
# =============================================================================
# Separate cache for building solar irradiance calculations

@dataclass
class _CachedBuildingRadiationModel:
    """Cached RadiationModel for building surface calculations."""
    model: object  # RadiationModel instance
    voxcity_shape: Tuple[int, int, int]  # Shape of voxel data for cache validation
    meshsize: float  # Meshsize for cache validation
    n_reflection_steps: int  # Number of reflection steps used
    is_building_surf: np.ndarray  # Boolean mask for building surfaces
    building_svf_mesh: object  # Building mesh (can be None)
    # Performance optimization: pre-computed mesh face to surface mapping
    bldg_indices: Optional[np.ndarray] = None  # Indices of building surfaces
    mesh_to_surface_idx: Optional[np.ndarray] = None  # Direct mapping: mesh face -> surface index
    # Cached mesh geometry to avoid recomputing each call
    mesh_face_centers: Optional[np.ndarray] = None  # Pre-computed triangles_center
    mesh_face_normals: Optional[np.ndarray] = None  # Pre-computed face_normals
    boundary_mask: Optional[np.ndarray] = None  # Pre-computed boundary vertical face mask
    # Cached building mesh (expensive to create, ~2.4s)
    cached_building_mesh: object = None  # Pre-computed building mesh from create_voxel_mesh


# Module-level cache for Building RadiationModel  
_building_radiation_model_cache: Optional[_CachedBuildingRadiationModel] = None


def _get_or_create_building_radiation_model(
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
    
    from .radiation import RadiationModel, RadiationConfig
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ny_vc, nx_vc, nz = voxel_data.shape
    
    # Check if cache is valid
    # A cached model with reflections (n_reflection_steps > 0) can be reused for non-reflection calls
    # But a cached model without reflections cannot be used for reflection calls
    cache_valid = False
    if _building_radiation_model_cache is not None:
        cache = _building_radiation_model_cache
        if (cache.voxcity_shape == voxel_data.shape and
            cache.meshsize == meshsize):
            # Cache is valid if:
            # 1. We don't need reflections (n_reflection_steps=0), OR
            # 2. Cached model has reflections enabled (can handle any n_reflection_steps)
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
    
    # Get location using helper function
    origin_lat, origin_lon = _get_location_from_voxcity(voxcity)
    
    # Create domain - consistent with ground-level model
    # VoxCity uses [row, col, z] = [i, j, k] convention
    # We create domain with nx=ny_vc, ny=nx_vc to match the palm_solar convention
    # but keep the same indexing as the ground model for consistency
    ni, nj, nk = ny_vc, nx_vc, nz  # Rename for clarity (matches ground model naming)
    
    domain = Domain(
        nx=ni, ny=nj, nz=nk,
        dx=meshsize, dy=meshsize, dz=meshsize,
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    # Convert VoxCity voxel data to domain arrays using vectorized helper
    default_lad = kwargs.get('default_lad', 2.0)
    is_solid_np, lad_np = _convert_voxel_data_to_arrays(voxel_data, default_lad)
    
    # Set domain arrays
    _set_solid_array(domain, is_solid_np)
    domain.set_lad_from_array(lad_np)
    _update_topo_from_solid(domain)
    
    # When n_reflection_steps=0, disable surface reflections to skip expensive SVF matrix computation
    surface_reflections = n_reflection_steps > 0
    
    config = RadiationConfig(
        n_reflection_steps=n_reflection_steps,
        n_azimuth=40,
        n_elevation=10,
        surface_reflections=surface_reflections,  # Disable when no reflections needed
        cache_svf_matrix=surface_reflections,     # Skip SVF matrix when reflections disabled
    )
    
    model = RadiationModel(domain, config)
    
    # Compute SVF (expensive! but only for sky view, not surface-to-surface when disabled)
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
    
    # Pre-compute bldg_indices for caching
    bldg_indices = np.where(is_building_surf)[0]
    
    # Cache the model
    _building_radiation_model_cache = _CachedBuildingRadiationModel(
        model=model,
        voxcity_shape=voxel_data.shape,
        meshsize=meshsize,
        n_reflection_steps=n_reflection_steps,
        is_building_surf=is_building_surf,
        building_svf_mesh=None,
        bldg_indices=bldg_indices,
        mesh_to_surface_idx=None  # Will be computed on first use with a specific mesh
    )
    
    return model, is_building_surf


def clear_building_radiation_model_cache():
    """Clear the cached Building RadiationModel to free memory."""
    global _building_radiation_model_cache
    _building_radiation_model_cache = None


def clear_all_radiation_caches():
    """Clear all cached RadiationModels to free GPU memory."""
    clear_radiation_model_cache()
    clear_building_radiation_model_cache()
    land_cover_albedo: Optional[LandCoverAlbedo] = None


def load_voxcity(filepath: Union[str, Path]):
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
) -> VoxCityDomainResult:
    """
    Convert VoxCity voxel grid to palm_solar Domain with material properties.
    
    This function:
    1. Extracts voxel grid, dimensions, and location from VoxCity data
    2. Creates a palm_solar Domain with solid cells and LAD
    3. Tracks land cover information for surface albedo assignment
    
    Args:
        voxcity_data: VoxCity object or dict from load_voxcity()
        default_lad: Default Leaf Area Density for tree voxels (m²/m³)
        land_cover_albedo: Custom land cover to albedo mapping
        origin_lat: Override latitude (degrees)
        origin_lon: Override longitude (degrees)
        
    Returns:
        VoxCityDomainResult with Domain and material information
    """
    if land_cover_albedo is None:
        land_cover_albedo = LandCoverAlbedo()
    
    # Extract data from VoxCity object or dict
    if hasattr(voxcity_data, 'voxels'):
        # New VoxCity dataclass format
        voxel_grid = voxcity_data.voxels.classes
        meshsize = voxcity_data.voxels.meta.meshsize
        land_cover_grid = voxcity_data.land_cover.classes
        dem_grid = voxcity_data.dem.elevation
        extras = getattr(voxcity_data, 'extras', {})
        rectangle_vertices = extras.get('rectangle_vertices', None)
    else:
        # Legacy dict format
        voxel_grid = voxcity_data['voxcity_grid']
        meshsize = voxcity_data['meshsize']
        land_cover_grid = voxcity_data.get('land_cover_grid', None)
        dem_grid = voxcity_data.get('dem_grid', None)
        rectangle_vertices = voxcity_data.get('rectangle_vertices', None)
    
    # Get grid dimensions (VoxCity is [row, col, z] = [y, x, z])
    ny, nx, nz = voxel_grid.shape
    
    # Use meshsize as voxel size
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
            # Default to Singapore
            if origin_lat is None:
                origin_lat = 1.35
            if origin_lon is None:
                origin_lon = 103.82
    
    print(f"VoxCity grid shape: ({ny}, {nx}, {nz})")
    print(f"Voxel size: {dx} m")
    print(f"Domain size: {nx*dx:.1f} x {ny*dy:.1f} x {nz*dz:.1f} m")
    print(f"Location: lat={origin_lat:.4f}, lon={origin_lon:.4f}")
    
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
    
    # Surface land cover tracking (indexed by grid position)
    # This will store the land cover code for ground-level surfaces
    surface_land_cover_grid = np.full((nx, ny), -1, dtype=np.int32)
    
    # Convert from VoxCity [row, col, z] to palm_solar [x, y, z]
    for row in range(ny):
        for col in range(nx):
            x_idx = col
            y_idx = row
            
            # Get land cover for this column (from ground surface)
            if land_cover_grid is not None:
                # Land cover grid is [row, col], values are class codes
                lc_val = land_cover_grid[row, col]
                if lc_val > 0:
                    # VoxCity adds +1 to land cover codes, so subtract 1
                    surface_land_cover_grid[x_idx, y_idx] = int(lc_val) - 1
                else:
                    surface_land_cover_grid[x_idx, y_idx] = 9  # Default: developed
            
            for z in range(nz):
                voxel_val = voxel_grid[row, col, z]
                
                if voxel_val == VOXCITY_BUILDING_CODE:
                    is_solid_np[x_idx, y_idx, z] = 1
                elif voxel_val == VOXCITY_GROUND_CODE:
                    is_solid_np[x_idx, y_idx, z] = 1
                elif voxel_val == VOXCITY_TREE_CODE:
                    lad_np[x_idx, y_idx, z] = default_lad
                elif voxel_val > 0:
                    # Positive values are land cover codes on ground
                    is_solid_np[x_idx, y_idx, z] = 1
    
    # Set domain arrays
    _set_solid_array(domain, is_solid_np)
    domain.set_lad_from_array(lad_np)
    _update_topo_from_solid(domain)
    
    # Count statistics
    solid_count = is_solid_np.sum()
    lad_count = (lad_np > 0).sum()
    print(f"Solid voxels: {solid_count:,}")
    print(f"Vegetation voxels (LAD > 0): {lad_count:,}")
    
    return VoxCityDomainResult(
        domain=domain,
        surface_land_cover=surface_land_cover_grid,
        land_cover_albedo=land_cover_albedo
    )


def apply_voxcity_albedo(
    model,
    voxcity_result: VoxCityDomainResult
) -> None:
    """
    Apply VoxCity land cover-based albedo values to radiation model surfaces.
    
    This function sets surface albedo values based on:
    - Land cover class for ground surfaces
    - Building wall/roof albedo for building surfaces
    
    Args:
        model: RadiationModel instance (after surface extraction)
        voxcity_result: Result from convert_voxcity_to_domain()
    """
    import taichi as ti
    from ..init_taichi import ensure_initialized
    ensure_initialized()
    
    if voxcity_result.surface_land_cover is None:
        print("Warning: No land cover data available, using default albedos")
        return
    
    domain = voxcity_result.domain
    lc_grid = voxcity_result.surface_land_cover
    lc_albedo = voxcity_result.land_cover_albedo
    
    # Get surface data
    n_surfaces = model.surfaces.n_surfaces[None]
    max_surfaces = model.surfaces.max_surfaces
    positions = model.surfaces.position.to_numpy()[:n_surfaces]
    directions = model.surfaces.direction.to_numpy()[:n_surfaces]
    
    # Create albedo array with full size (must match Taichi field shape)
    albedo_values = np.zeros(max_surfaces, dtype=np.float32)
    
    # Direction codes
    IUP = 0
    IDOWN = 1
    
    for idx in range(n_surfaces):
        i, j, k = positions[idx]
        direction = directions[idx]
        
        if direction == IUP:  # Upward facing
            if k == 0 or k == 1:
                # Ground level - use land cover albedo
                lc_code = lc_grid[i, j]
                if lc_code >= 0:
                    albedo_values[idx] = lc_albedo.get_land_cover_albedo(lc_code)
                else:
                    albedo_values[idx] = lc_albedo.developed
            else:
                # Roof
                albedo_values[idx] = lc_albedo.building_roof
        elif direction == IDOWN:  # Downward facing
            albedo_values[idx] = lc_albedo.building_wall
        else:  # Walls (N, S, E, W)
            albedo_values[idx] = lc_albedo.building_wall
    
    # Apply albedo values to surfaces
    model.surfaces.albedo.from_numpy(albedo_values)
    
    # Print summary
    unique_albedos = np.unique(albedo_values[:n_surfaces])
    print(f"Applied {len(unique_albedos)} unique albedo values to {n_surfaces} surfaces")


def _set_solid_array(domain: Domain, solid_array: np.ndarray) -> None:
    """Set domain solid cells from numpy array."""
    import taichi as ti
    from ..init_taichi import ensure_initialized
    ensure_initialized()
    
    @ti.kernel
    def _set_solid_kernel(domain: ti.template(), solid: ti.types.ndarray()):
        for i, j, k in domain.is_solid:
            domain.is_solid[i, j, k] = solid[i, j, k]
    
    _set_solid_kernel(domain, solid_array)


def _update_topo_from_solid(domain: Domain) -> None:
    """Update topography field from solid array."""
    import taichi as ti
    from ..init_taichi import ensure_initialized
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


def create_radiation_config_for_voxcity(
    land_cover_albedo: Optional[LandCoverAlbedo] = None,
    **kwargs
) -> RadiationConfig:
    """
    Create a RadiationConfig suitable for VoxCity simulations.
    
    This sets appropriate default values for urban environments.
    
    Args:
        land_cover_albedo: Land cover albedo mapping (for reference)
        **kwargs: Additional RadiationConfig parameters
        
    Returns:
        RadiationConfig instance
    """
    if land_cover_albedo is None:
        land_cover_albedo = LandCoverAlbedo()
    
    # Set defaults suitable for urban environments
    defaults = {
        'albedo_ground': land_cover_albedo.developed,
        'albedo_wall': land_cover_albedo.building_wall,
        'albedo_roof': land_cover_albedo.building_roof,
        'albedo_leaf': land_cover_albedo.leaf,
        'n_azimuth': 40,  # Reduced for faster computation
        'n_elevation': 10,
        'n_reflection_steps': 2,
    }
    
    # Override with user-provided values
    defaults.update(kwargs)
    
    return RadiationConfig(**defaults)


def _compute_ground_irradiance_with_reflections(
    voxcity,
    azimuth_degrees_ori: float,
    elevation_degrees: float,
    direct_normal_irradiance: float,
    diffuse_irradiance: float,
    view_point_height: float = 1.5,
    n_reflection_steps: int = 2,
    progress_report: bool = False,
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ground-level irradiance using full RadiationModel with reflections.
    
    Uses a cached RadiationModel to avoid recomputing SVF/CSF matrices for each
    solar position. The geometry-dependent matrices are computed once and reused.
    
    Note: The diffuse component includes sky diffuse + multi-bounce surface reflections + 
    canopy scattering, as computed by the RadiationModel.
    
    Args:
        voxcity: VoxCity object
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        diffuse_irradiance: DHI in W/m²
        view_point_height: Observer height above ground (default: 1.5)
        n_reflection_steps: Number of reflection bounces (default: 2)
        progress_report: Print progress (default: False)
        **kwargs: Additional parameters
    
    Returns:
        Tuple of (direct_map, diffuse_map, reflected_map) as 2D numpy arrays
    """
    from .domain import IUP
    
    voxel_data = voxcity.voxels.classes
    ni, nj, nk = voxel_data.shape
    
    # Remove parameters that we pass explicitly to avoid duplicates
    filtered_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ('n_reflection_steps', 'progress_report', 'view_point_height')}
    
    # Get or create cached RadiationModel (SVF/CSF only computed once)
    model, valid_ground, ground_k = _get_or_create_radiation_model(
        voxcity,
        n_reflection_steps=n_reflection_steps,
        progress_report=progress_report,
        **filtered_kwargs
    )
    
    # Set solar position for this timestep
    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_radians = np.deg2rad(azimuth_degrees)
    elevation_radians = np.deg2rad(elevation_degrees)
    
    sun_dir_x = np.cos(elevation_radians) * np.cos(azimuth_radians)
    sun_dir_y = np.cos(elevation_radians) * np.sin(azimuth_radians)
    sun_dir_z = np.sin(elevation_radians)
    
    # Set sun direction and cos_zenith directly on the SolarCalculator fields
    model.solar_calc.sun_direction[None] = (sun_dir_x, sun_dir_y, sun_dir_z)
    model.solar_calc.cos_zenith[None] = np.sin(elevation_radians)  # cos(zenith) = sin(elevation)
    model.solar_calc.sun_up[None] = 1 if elevation_degrees > 0 else 0
    
    # Compute shortwave radiation (uses cached SVF/CSF matrices)
    model.compute_shortwave_radiation(
        sw_direct=direct_normal_irradiance,
        sw_diffuse=diffuse_irradiance
    )
    
    # Extract surface irradiance using cached mapping for vectorized extraction
    # This is much faster than iterating through all surfaces
    n_surfaces = model.surfaces.count
    
    # Initialize output arrays
    direct_map = np.full((ni, nj), np.nan, dtype=np.float32)
    diffuse_map = np.full((ni, nj), np.nan, dtype=np.float32)
    reflected_map = np.zeros((ni, nj), dtype=np.float32)
    
    # Use pre-computed surface-to-grid mapping if available (from cache)
    if (_radiation_model_cache is not None and 
        _radiation_model_cache.grid_indices is not None and 
        len(_radiation_model_cache.grid_indices) > 0):
        
        grid_indices = _radiation_model_cache.grid_indices
        surface_indices = _radiation_model_cache.surface_indices
        
        # Extract only the irradiance values we need (vectorized)
        sw_in_direct = model.surfaces.sw_in_direct.to_numpy()
        sw_in_diffuse = model.surfaces.sw_in_diffuse.to_numpy()
        
        # Vectorized assignment using pre-computed indices
        direct_map[grid_indices[:, 0], grid_indices[:, 1]] = sw_in_direct[surface_indices]
        diffuse_map[grid_indices[:, 0], grid_indices[:, 1]] = sw_in_diffuse[surface_indices]
    else:
        # Fallback to original loop if no cached mapping
        from .domain import IUP
        positions = model.surfaces.position.to_numpy()[:n_surfaces]
        directions = model.surfaces.direction.to_numpy()[:n_surfaces]
        sw_in_direct = model.surfaces.sw_in_direct.to_numpy()[:n_surfaces]
        sw_in_diffuse = model.surfaces.sw_in_diffuse.to_numpy()[:n_surfaces]
        
        for idx in range(n_surfaces):
            pos_i, pos_j, k = positions[idx]
            direction = directions[idx]
            
            if direction == IUP:
                ii, jj = int(pos_i), int(pos_j)
                if 0 <= ii < ni and 0 <= jj < nj:
                    if not valid_ground[ii, jj]:
                        continue
                    if int(k) == ground_k[ii, jj]:
                        if np.isnan(direct_map[ii, jj]):
                            direct_map[ii, jj] = sw_in_direct[idx]
                            diffuse_map[ii, jj] = sw_in_diffuse[idx]
    
    # Flip to match VoxCity coordinate system
    direct_map = np.flipud(direct_map)
    diffuse_map = np.flipud(diffuse_map)
    reflected_map = np.flipud(reflected_map)
    
    return direct_map, diffuse_map, reflected_map


# =============================================================================
# VoxCity API-Compatible Solar Irradiance Functions
# =============================================================================
# These functions match the voxcity.simulator.solar API signatures for 
# drop-in replacement with GPU acceleration.

def get_direct_solar_irradiance_map(
    voxcity,
    azimuth_degrees_ori: float,
    elevation_degrees: float,
    direct_normal_irradiance: float,
    show_plot: bool = False,
    with_reflections: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated direct horizontal irradiance map computation.
    
    This function matches the signature of voxcity.simulator.solar.get_direct_solar_irradiance_map
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        show_plot: Whether to display a matplotlib plot
        with_reflections: If True, use full RadiationModel with multi-bounce 
            reflections. If False (default), use simple ray-tracing for 
            faster but less accurate results.
        **kwargs: Additional parameters including:
            - view_point_height (float): Observer height above ground (default: 1.5)
            - tree_k (float): Tree extinction coefficient (default: 0.6)
            - tree_lad (float): Leaf area density (default: 1.0)
            - colormap (str): Matplotlib colormap name (default: 'magma')
            - vmin, vmax (float): Colormap limits
            - obj_export (bool): Export to OBJ file (default: False)
            - n_reflection_steps (int): Number of reflection bounces when 
                with_reflections=True (default: 2)
            - progress_report (bool): Print progress (default: False)
    
    Returns:
        2D numpy array of direct horizontal irradiance (W/m²)
    """
    import taichi as ti
    from ..init_taichi import ensure_initialized
    ensure_initialized()
    
    colormap = kwargs.get('colormap', 'magma')
    vmin = kwargs.get('vmin', 0.0)
    vmax = kwargs.get('vmax', direct_normal_irradiance)
    
    if with_reflections:
        # Use full RadiationModel with reflections
        direct_map, _, _ = _compute_ground_irradiance_with_reflections(
            voxcity=voxcity,
            azimuth_degrees_ori=azimuth_degrees_ori,
            elevation_degrees=elevation_degrees,
            direct_normal_irradiance=direct_normal_irradiance,
            diffuse_irradiance=0.0,  # Only compute direct component
            **kwargs
        )
    else:
        # Use simple ray-tracing (faster but no reflections)
        voxel_data = voxcity.voxels.classes
        meshsize = voxcity.voxels.meta.meshsize
        
        view_point_height = kwargs.get('view_point_height', 1.5)
        tree_k = kwargs.get('tree_k', 0.6)
        tree_lad = kwargs.get('tree_lad', 1.0)
        
        # Convert to sun direction vector
        # VoxCity convention: azimuth 0=North, clockwise
        # Convert to standard: 180 - azimuth
        azimuth_degrees = 180 - azimuth_degrees_ori
        azimuth_radians = np.deg2rad(azimuth_degrees)
        elevation_radians = np.deg2rad(elevation_degrees)
        
        dx_dir = np.cos(elevation_radians) * np.cos(azimuth_radians)
        dy_dir = np.cos(elevation_radians) * np.sin(azimuth_radians)
        dz_dir = np.sin(elevation_radians)
        
        # Compute transmittance map using ray tracing
        transmittance_map = _compute_direct_transmittance_map_gpu(
            voxel_data=voxel_data,
            sun_direction=(dx_dir, dy_dir, dz_dir),
            view_point_height=view_point_height,
            meshsize=meshsize,
            tree_k=tree_k,
            tree_lad=tree_lad
        )
        
        # Convert to horizontal irradiance
        sin_elev = np.sin(elevation_radians)
        direct_map = transmittance_map * direct_normal_irradiance * sin_elev
        
        # Flip to match VoxCity coordinate system
        direct_map = np.flipud(direct_map)
    
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(direct_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Direct Solar Irradiance (W/m²)')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    if kwargs.get('obj_export', False):
        _export_irradiance_to_obj(
            voxcity, direct_map, 
            output_name=kwargs.get('output_file_name', 'direct_solar_irradiance'),
            **kwargs
        )
    
    return direct_map


def get_diffuse_solar_irradiance_map(
    voxcity,
    diffuse_irradiance: float = 1.0,
    show_plot: bool = False,
    with_reflections: bool = False,
    azimuth_degrees_ori: float = 180.0,
    elevation_degrees: float = 45.0,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated diffuse horizontal irradiance map computation using SVF.
    
    This function matches the signature of voxcity.simulator.solar.get_diffuse_solar_irradiance_map
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        diffuse_irradiance: Diffuse horizontal irradiance in W/m²
        show_plot: Whether to display a matplotlib plot
        with_reflections: If True, use full RadiationModel with multi-bounce 
            reflections (requires azimuth_degrees_ori and elevation_degrees).
            If False (default), use simple SVF-based computation.
        azimuth_degrees_ori: Solar azimuth in degrees (only used when with_reflections=True)
        elevation_degrees: Solar elevation in degrees (only used when with_reflections=True)
        **kwargs: Additional parameters including:
            - view_point_height (float): Observer height above ground (default: 1.5)
            - N_azimuth (int): Number of azimuthal divisions (default: 120)
            - N_elevation (int): Number of elevation divisions (default: 20)
            - tree_k (float): Tree extinction coefficient (default: 0.6)
            - tree_lad (float): Leaf area density (default: 1.0)
            - colormap (str): Matplotlib colormap name (default: 'magma')
            - vmin, vmax (float): Colormap limits
            - obj_export (bool): Export to OBJ file (default: False)
            - n_reflection_steps (int): Number of reflection bounces when 
                with_reflections=True (default: 2)
            - progress_report (bool): Print progress (default: False)
    
    Returns:
        2D numpy array of diffuse horizontal irradiance (W/m²)
    """
    colormap = kwargs.get('colormap', 'magma')
    vmin = kwargs.get('vmin', 0.0)
    vmax = kwargs.get('vmax', diffuse_irradiance)
    
    if with_reflections:
        # Use full RadiationModel with reflections
        # Remove parameters we explicitly set to avoid conflicts
        refl_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ('direct_normal_irradiance', 'diffuse_irradiance')}
        _, diffuse_map, reflected_map = _compute_ground_irradiance_with_reflections(
            voxcity=voxcity,
            azimuth_degrees_ori=azimuth_degrees_ori,
            elevation_degrees=elevation_degrees,
            direct_normal_irradiance=kwargs.get('direct_normal_irradiance', 0.0),
            diffuse_irradiance=diffuse_irradiance,
            **refl_kwargs
        )
        # Include reflected component in diffuse when using reflection model
        diffuse_map = np.where(np.isnan(diffuse_map), np.nan, diffuse_map + reflected_map)
    else:
        # Use simple SVF-based computation (faster but no reflections)
        # Import the visibility SVF function
        from ..visibility.integration import get_sky_view_factor_map as get_svf_map
        
        # Get SVF map using GPU-accelerated visibility module
        svf_kwargs = kwargs.copy()
        svf_kwargs['colormap'] = 'BuPu_r'
        svf_kwargs['vmin'] = 0
        svf_kwargs['vmax'] = 1
        
        SVF_map = get_svf_map(voxcity, show_plot=False, **svf_kwargs)
        diffuse_map = SVF_map * diffuse_irradiance
    
    if show_plot:
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(diffuse_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Diffuse Solar Irradiance (W/m²)')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    if kwargs.get('obj_export', False):
        _export_irradiance_to_obj(
            voxcity, diffuse_map,
            output_name=kwargs.get('output_file_name', 'diffuse_solar_irradiance'),
            **kwargs
        )
    
    return diffuse_map


def get_global_solar_irradiance_map(
    voxcity,
    azimuth_degrees_ori: float,
    elevation_degrees: float,
    direct_normal_irradiance: float,
    diffuse_irradiance: float,
    show_plot: bool = False,
    with_reflections: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated global (direct + diffuse) horizontal irradiance map.
    
    This function matches the signature of voxcity.simulator.solar.get_global_solar_irradiance_map
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        diffuse_irradiance: DHI in W/m²
        show_plot: Whether to display a matplotlib plot
        with_reflections: If True, use full RadiationModel with multi-bounce 
            reflections. If False (default), use simple ray-tracing/SVF for 
            faster but less accurate results.
        **kwargs: Additional parameters (see get_direct_solar_irradiance_map)
            - computation_mask (np.ndarray): Optional 2D boolean mask for sub-area computation
            - n_reflection_steps (int): Number of reflection bounces when 
                with_reflections=True (default: 2)
            - progress_report (bool): Print progress (default: False)
    
    Returns:
        2D numpy array of global horizontal irradiance (W/m²)
    """
    # Extract computation_mask from kwargs
    computation_mask = kwargs.pop('computation_mask', None)
    
    if with_reflections:
        # Use full RadiationModel with reflections (single call for all components)
        direct_map, diffuse_map, reflected_map = _compute_ground_irradiance_with_reflections(
            voxcity=voxcity,
            azimuth_degrees_ori=azimuth_degrees_ori,
            elevation_degrees=elevation_degrees,
            direct_normal_irradiance=direct_normal_irradiance,
            diffuse_irradiance=diffuse_irradiance,
            **kwargs
        )
        # Combine all components: direct + diffuse + reflected
        global_map = np.where(
            np.isnan(direct_map), 
            np.nan, 
            direct_map + diffuse_map + reflected_map
        )
    else:
        # Compute direct and diffuse components separately (no reflections)
        direct_map = get_direct_solar_irradiance_map(
            voxcity,
            azimuth_degrees_ori,
            elevation_degrees,
            direct_normal_irradiance,
            show_plot=False,
            with_reflections=False,
            **kwargs
        )
        
        diffuse_map = get_diffuse_solar_irradiance_map(
            voxcity,
            diffuse_irradiance=diffuse_irradiance,
            show_plot=False,
            with_reflections=False,
            **kwargs
        )
        
        # Combine: where direct is NaN, use only diffuse
        global_map = np.where(np.isnan(direct_map), diffuse_map, direct_map + diffuse_map)
    
    if show_plot:
        colormap = kwargs.get('colormap', 'magma')
        vmin = kwargs.get('vmin', 0.0)
        vmax = kwargs.get('vmax', max(float(np.nanmax(global_map)), 1.0))
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(global_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Global Solar Irradiance (W/m²)')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    if kwargs.get('obj_export', False):
        _export_irradiance_to_obj(
            voxcity, global_map,
            output_name=kwargs.get('output_file_name', 'global_solar_irradiance'),
            **kwargs
        )
    
    # Apply computation mask if provided
    if computation_mask is not None:
        # Ensure mask shape matches output shape (note: output is flipped)
        if computation_mask.shape == global_map.shape:
            global_map = np.where(np.flipud(computation_mask), global_map, np.nan)
        elif computation_mask.T.shape == global_map.shape:
            global_map = np.where(np.flipud(computation_mask.T), global_map, np.nan)
        else:
            # Try to match without flip
            if computation_mask.shape == global_map.shape:
                global_map = np.where(computation_mask, global_map, np.nan)
    
    return global_map


def get_cumulative_global_solar_irradiance(
    voxcity,
    df,
    lon: float,
    lat: float,
    tz: float,
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    show_plot: bool = False,
    with_reflections: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated cumulative global solar irradiance over a period.
    
    This function matches the signature of voxcity.simulator.solar.get_cumulative_global_solar_irradiance
    using Taichi GPU acceleration with sky patch optimization.
    
    OPTIMIZATIONS IMPLEMENTED:
    1. Vectorized sun position binning using bin_sun_positions_to_tregenza_fast
    2. Pre-allocated output arrays for patch loop
    3. Cached model reuse across patches (SVF/CSF computed only once)
    4. Efficient array extraction with pre-computed surface-to-grid mapping
    
    Args:
        voxcity: VoxCity object
        df: pandas DataFrame with 'DNI' and 'DHI' columns, datetime-indexed
        lon: Longitude in degrees
        lat: Latitude in degrees
        tz: Timezone offset in hours
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        show_plot: Whether to display a matplotlib plot
        with_reflections: If True, use full RadiationModel with multi-bounce 
            reflections for each timestep/patch. If False (default), use simple 
            ray-tracing/SVF for faster computation.
        **kwargs: Additional parameters including:
            - computation_mask (np.ndarray): Optional 2D boolean mask for sub-area computation
            - start_time (str): Start time 'MM-DD HH:MM:SS' (default: '01-01 05:00:00')
            - end_time (str): End time 'MM-DD HH:MM:SS' (default: '01-01 20:00:00')
            - view_point_height (float): Observer height (default: 1.5)
            - use_sky_patches (bool): Use sky patch optimization (default: True)
            - sky_discretization (str): 'tregenza', 'reinhart', 'uniform', 'fibonacci'
            - progress_report (bool): Print progress (default: False)
            - colormap (str): Colormap name (default: 'magma')
            - n_reflection_steps (int): Number of reflection bounces when 
                with_reflections=True (default: 2)
    
    Returns:
        2D numpy array of cumulative irradiance (Wh/m²)
    """
    import time
    from datetime import datetime
    import pytz
    
    # Extract parameters that we pass explicitly (use pop to avoid duplicate kwargs)
    kwargs = kwargs.copy()  # Don't modify the original
    computation_mask = kwargs.pop('computation_mask', None)
    view_point_height = kwargs.pop('view_point_height', 1.5)
    colormap = kwargs.pop('colormap', 'magma')
    start_time = kwargs.pop('start_time', '01-01 05:00:00')
    end_time = kwargs.pop('end_time', '01-01 20:00:00')
    progress_report = kwargs.pop('progress_report', False)
    use_sky_patches = kwargs.pop('use_sky_patches', True)
    sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
    
    if df.empty:
        raise ValueError("No data in EPW dataframe.")
    
    # Parse time range
    try:
        start_dt = datetime.strptime(start_time, '%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time, '%m-%d %H:%M:%S')
    except ValueError as ve:
        raise ValueError("start_time and end_time must be in format 'MM-DD HH:MM:SS'") from ve
    
    # Filter dataframe to period
    df = df.copy()
    df['hour_of_year'] = (df.index.dayofyear - 1) * 24 + df.index.hour + 1
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1
    
    if start_hour <= end_hour:
        df_period = df[(df['hour_of_year'] >= start_hour) & (df['hour_of_year'] <= end_hour)]
    else:
        df_period = df[(df['hour_of_year'] >= start_hour) | (df['hour_of_year'] <= end_hour)]
    
    if df_period.empty:
        raise ValueError("No EPW data in the specified period.")
    
    # Localize and convert to UTC
    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)
    
    # Get solar positions
    solar_positions = _get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Compute base diffuse map (SVF-based for efficiency, or with reflections if requested)
    # Note: For cumulative with_reflections, we still use SVF-based base for diffuse sky contribution
    # The reflection component is computed per timestep when with_reflections=True
    diffuse_kwargs = kwargs.copy()
    diffuse_kwargs.update({'show_plot': False, 'obj_export': False})
    base_diffuse_map = get_diffuse_solar_irradiance_map(
        voxcity,
        diffuse_irradiance=1.0,
        with_reflections=False,  # Always use SVF for base diffuse in cumulative mode
        **diffuse_kwargs
    )
    
    voxel_data = voxcity.voxels.classes
    nx, ny, _ = voxel_data.shape
    cumulative_map = np.zeros((nx, ny))
    mask_map = np.ones((nx, ny), dtype=bool)
    
    direct_kwargs = kwargs.copy()
    direct_kwargs.update({
        'show_plot': False, 
        'view_point_height': view_point_height, 
        'obj_export': False,
        'with_reflections': with_reflections  # Pass through to direct/global map calls
    })
    
    if use_sky_patches:
        # Use sky patch aggregation for efficiency
        from .sky import (
            generate_tregenza_patches,
            generate_reinhart_patches,
            generate_uniform_grid_patches,
            generate_fibonacci_patches,
            get_tregenza_patch_index
        )
        
        t0 = time.perf_counter() if progress_report else 0
        
        # Extract arrays
        azimuth_arr = solar_positions['azimuth'].to_numpy()
        elevation_arr = solar_positions['elevation'].to_numpy()
        dni_arr = df_period_utc['DNI'].to_numpy() * direct_normal_irradiance_scaling
        dhi_arr = df_period_utc['DHI'].to_numpy() * diffuse_irradiance_scaling
        time_step_hours = kwargs.get('time_step_hours', 1.0)
        
        # Generate sky patches
        if sky_discretization.lower() == 'tregenza':
            patches, directions, solid_angles = generate_tregenza_patches()
        elif sky_discretization.lower() == 'reinhart':
            mf = kwargs.get('reinhart_mf', kwargs.get('mf', 4))
            patches, directions, solid_angles = generate_reinhart_patches(mf=mf)
        elif sky_discretization.lower() == 'uniform':
            n_az = kwargs.get('sky_n_azimuth', kwargs.get('n_azimuth', 36))
            n_el = kwargs.get('sky_n_elevation', kwargs.get('n_elevation', 9))
            patches, directions, solid_angles = generate_uniform_grid_patches(n_az, n_el)
        elif sky_discretization.lower() == 'fibonacci':
            n_patches = kwargs.get('sky_n_patches', kwargs.get('n_patches', 145))
            patches, directions, solid_angles = generate_fibonacci_patches(n_patches=n_patches)
        else:
            raise ValueError(f"Unknown sky discretization method: {sky_discretization}")
        
        n_patches = len(patches)
        n_timesteps = len(azimuth_arr)
        cumulative_dni = np.zeros(n_patches, dtype=np.float64)
        
        # OPTIMIZATION: Vectorized DHI accumulation (only for positive values)
        # This replaces the loop-based accumulation
        valid_dhi_mask = dhi_arr > 0
        total_cumulative_dhi = np.sum(dhi_arr[valid_dhi_mask]) * time_step_hours
        
        # DNI binning - loop is already fast (~7ms for 731 timesteps)
        # The loop is necessary because patch assignment depends on sun position
        for i in range(n_timesteps):
            elev = elevation_arr[i]
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            
            if dni <= 0:
                continue
            
            patch_idx = int(get_tregenza_patch_index(float(az), float(elev)))
            if patch_idx >= 0 and patch_idx < n_patches:
                cumulative_dni[patch_idx] += dni * time_step_hours
        
        active_mask = cumulative_dni > 0
        n_active = int(np.sum(active_mask))
        
        if progress_report:
            bin_time = time.perf_counter() - t0
            print(f"Sky patch optimization: {n_timesteps} timesteps -> {n_active} active patches ({sky_discretization})")
            print(f"  Sun position binning: {bin_time:.3f}s")
            print(f"  Total cumulative DHI: {total_cumulative_dhi:.1f} Wh/m²")
            if with_reflections:
                print("  Using RadiationModel with multi-bounce reflections")
        
        # Diffuse component
        cumulative_diffuse = base_diffuse_map * total_cumulative_dhi
        cumulative_map += np.nan_to_num(cumulative_diffuse, nan=0.0)
        mask_map &= ~np.isnan(cumulative_diffuse)
        
        # Direct component - loop over active patches
        # When with_reflections=True, use get_global_solar_irradiance_map to include 
        # reflections for each patch direction
        active_indices = np.where(active_mask)[0]
        
        # OPTIMIZATION: Pre-warm the model (ensures JIT compilation is done)
        if with_reflections and len(active_indices) > 0:
            # Ensure model is created and cached before timing
            n_reflection_steps = kwargs.get('n_reflection_steps', 2)
            _ = _get_or_create_radiation_model(
                voxcity,
                n_reflection_steps=n_reflection_steps,
                progress_report=progress_report
            )
        
        if progress_report:
            t_patch_start = time.perf_counter()
        
        for i, patch_idx in enumerate(active_indices):
            az_deg = patches[patch_idx, 0]
            el_deg = patches[patch_idx, 1]
            cumulative_dni_patch = cumulative_dni[patch_idx]
            
            if with_reflections:
                # Use full RadiationModel: compute direct + reflected for this direction
                # We set diffuse_irradiance=0 since we handle diffuse separately
                direct_map, _, reflected_map = _compute_ground_irradiance_with_reflections(
                    voxcity=voxcity,
                    azimuth_degrees_ori=az_deg,
                    elevation_degrees=el_deg,
                    direct_normal_irradiance=1.0,
                    diffuse_irradiance=0.0,
                    view_point_height=view_point_height,
                    **kwargs
                )
                # Include reflections in patch contribution
                patch_contribution = (direct_map + np.nan_to_num(reflected_map, nan=0.0)) * cumulative_dni_patch
            else:
                # Simple ray tracing (no reflections)
                direct_map = get_direct_solar_irradiance_map(
                    voxcity,
                    az_deg,
                    el_deg,
                    direct_normal_irradiance=1.0,
                    **direct_kwargs
                )
                patch_contribution = direct_map * cumulative_dni_patch
            
            mask_map &= ~np.isnan(patch_contribution)
            cumulative_map += np.nan_to_num(patch_contribution, nan=0.0)
            
            if progress_report and ((i + 1) % max(1, len(active_indices) // 10) == 0 or i == len(active_indices) - 1):
                elapsed = time.perf_counter() - t_patch_start
                pct = (i + 1) * 100.0 / len(active_indices)
                avg_per_patch = elapsed / (i + 1)
                eta = avg_per_patch * (len(active_indices) - i - 1)
                print(f"  Patch {i+1}/{len(active_indices)} ({pct:.1f}%) - elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s, avg: {avg_per_patch*1000:.1f}ms/patch")
        
        if progress_report:
            total_patch_time = time.perf_counter() - t_patch_start
            print(f"  Total patch processing: {total_patch_time:.2f}s ({n_active} patches)")
    
    else:
        # Per-timestep path
        if progress_report and with_reflections:
            print("  Using RadiationModel with multi-bounce reflections (per-timestep)")
        
        for idx, (time_utc, row) in enumerate(df_period_utc.iterrows()):
            DNI = float(row['DNI']) * direct_normal_irradiance_scaling
            DHI = float(row['DHI']) * diffuse_irradiance_scaling
            
            solpos = solar_positions.loc[time_utc]
            azimuth_degrees = float(solpos['azimuth'])
            elevation_degrees_val = float(solpos['elevation'])
            
            if with_reflections:
                # Use full RadiationModel for this timestep
                direct_map, diffuse_map_ts, reflected_map = _compute_ground_irradiance_with_reflections(
                    voxcity=voxcity,
                    azimuth_degrees_ori=azimuth_degrees,
                    elevation_degrees=elevation_degrees_val,
                    direct_normal_irradiance=DNI,
                    diffuse_irradiance=DHI,
                    view_point_height=view_point_height,
                    **kwargs
                )
                # Combine all components
                combined = (np.nan_to_num(direct_map, nan=0.0) + 
                           np.nan_to_num(diffuse_map_ts, nan=0.0) + 
                           np.nan_to_num(reflected_map, nan=0.0))
                mask_map &= ~np.isnan(direct_map)
            else:
                # Simple ray tracing (no reflections)
                direct_map = get_direct_solar_irradiance_map(
                    voxcity,
                    azimuth_degrees,
                    elevation_degrees_val,
                    direct_normal_irradiance=DNI,
                    **direct_kwargs  # with_reflections already in direct_kwargs
                )
                
                diffuse_contrib = base_diffuse_map * DHI
                combined = np.nan_to_num(direct_map, nan=0.0) + np.nan_to_num(diffuse_contrib, nan=0.0)
                mask_map &= ~np.isnan(direct_map) & ~np.isnan(diffuse_contrib)
            
            cumulative_map += combined
            
            if progress_report and (idx + 1) % max(1, len(df_period_utc) // 10) == 0:
                pct = (idx + 1) * 100.0 / len(df_period_utc)
                print(f"  Timestep {idx+1}/{len(df_period_utc)} ({pct:.1f}%)")
    
    # Apply mask for plotting
    cumulative_map = np.where(mask_map, cumulative_map, np.nan)
    
    # Apply computation mask if provided
    if computation_mask is not None:
        # Handle different shape orientations
        if computation_mask.shape == cumulative_map.shape:
            cumulative_map = np.where(np.flipud(computation_mask), cumulative_map, np.nan)
        elif computation_mask.T.shape == cumulative_map.shape:
            cumulative_map = np.where(np.flipud(computation_mask.T), cumulative_map, np.nan)
    
    if show_plot:
        vmax = kwargs.get('vmax', float(np.nanmax(cumulative_map)) if not np.all(np.isnan(cumulative_map)) else 1.0)
        vmin = kwargs.get('vmin', 0.0)
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(cumulative_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label='Cumulative Global Solar Irradiance (Wh/m²)')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    return cumulative_map


def get_building_solar_irradiance(
    voxcity,
    building_svf_mesh=None,
    azimuth_degrees_ori: float = None,
    elevation_degrees: float = None,
    direct_normal_irradiance: float = None,
    diffuse_irradiance: float = None,
    **kwargs
):
    """
    GPU-accelerated building surface solar irradiance computation.
    
    This function matches the signature of voxcity.simulator.solar.get_building_solar_irradiance
    using Taichi GPU acceleration with multi-bounce reflections.
    
    Uses cached RadiationModel to avoid recomputing SVF/CSF matrices for each timestep.
    
    Args:
        voxcity: VoxCity object
        building_svf_mesh: Pre-computed mesh with SVF values (optional, for VoxCity API compatibility)
            If provided, SVF values from mesh metadata will be used.
            If None, SVF will be computed internally.
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        diffuse_irradiance: DHI in W/m²
        **kwargs: Additional parameters including:
            - with_reflections (bool): Enable multi-bounce surface reflections (default: False).
                Set to True for more accurate results but slower computation.
            - n_reflection_steps (int): Number of reflection bounces when with_reflections=True (default: 2)
            - tree_k (float): Tree extinction coefficient (default: 0.6)
            - building_class_id (int): Building voxel class code (default: -3)
            - progress_report (bool): Print progress (default: False)
            - colormap (str): Colormap name (default: 'magma')
            - obj_export (bool): Export mesh to OBJ (default: False)
    
    Returns:
        Trimesh object with irradiance values in metadata
    """
    # Handle positional argument order from VoxCity API:
    # VoxCity: get_building_solar_irradiance(voxcity, building_svf_mesh, azimuth, elevation, dni, dhi, **kwargs)
    # If building_svf_mesh is a number, assume old GPU-only API call where second arg is azimuth
    if isinstance(building_svf_mesh, (int, float)):
        # Old API: get_building_solar_irradiance(voxcity, azimuth, elevation, dni, dhi, ...)
        diffuse_irradiance = direct_normal_irradiance
        direct_normal_irradiance = elevation_degrees
        elevation_degrees = azimuth_degrees_ori
        azimuth_degrees_ori = building_svf_mesh
        building_svf_mesh = None
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    building_id_grid = voxcity.buildings.ids
    ny_vc, nx_vc, nz = voxel_data.shape
    
    # Extract parameters that we pass explicitly (to avoid duplicate kwargs error)
    progress_report = kwargs.pop('progress_report', False)
    building_class_id = kwargs.pop('building_class_id', -3)
    n_reflection_steps = kwargs.pop('n_reflection_steps', 2)
    colormap = kwargs.pop('colormap', 'magma')
    with_reflections = kwargs.pop('with_reflections', False)  # Default False for speed; set True for multi-bounce reflections
    
    # If with_reflections=False, set n_reflection_steps=0 to skip expensive SVF matrix computation
    if not with_reflections:
        n_reflection_steps = 0
    
    # Get cached or create new RadiationModel (SVF/CSF computed only once)
    model, is_building_surf = _get_or_create_building_radiation_model(
        voxcity,
        n_reflection_steps=n_reflection_steps,
        progress_report=progress_report,
        building_class_id=building_class_id,
        **kwargs
    )
    
    # Set solar position for this timestep
    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_radians = np.deg2rad(azimuth_degrees)
    elevation_radians = np.deg2rad(elevation_degrees)
    
    sun_dir_x = np.cos(elevation_radians) * np.cos(azimuth_radians)
    sun_dir_y = np.cos(elevation_radians) * np.sin(azimuth_radians)
    sun_dir_z = np.sin(elevation_radians)
    
    # Set sun direction and cos_zenith directly on the SolarCalculator fields
    model.solar_calc.sun_direction[None] = (sun_dir_x, sun_dir_y, sun_dir_z)
    model.solar_calc.cos_zenith[None] = np.sin(elevation_radians)
    model.solar_calc.sun_up[None] = 1 if elevation_degrees > 0 else 0
    
    # Compute shortwave radiation (uses cached SVF/CSF matrices)
    model.compute_shortwave_radiation(
        sw_direct=direct_normal_irradiance,
        sw_diffuse=diffuse_irradiance
    )
    
    # Extract surface irradiance from palm_solar model
    # Note: Use to_numpy() without slicing - slicing is VERY slow on Taichi arrays
    # The mesh_to_surface_idx will handle extracting only the values we need
    n_surfaces = model.surfaces.count
    sw_in_direct_all = model.surfaces.sw_in_direct.to_numpy()
    sw_in_diffuse_all = model.surfaces.sw_in_diffuse.to_numpy()

    if hasattr(model.surfaces, 'sw_in_reflected'):
        sw_in_reflected_all = model.surfaces.sw_in_reflected.to_numpy()
    else:
        sw_in_reflected_all = np.zeros_like(sw_in_direct_all)

    total_sw_all = sw_in_direct_all + sw_in_diffuse_all + sw_in_reflected_all

    # Get building indices from cache (avoids np.where every time)
    bldg_indices = _building_radiation_model_cache.bldg_indices if _building_radiation_model_cache else np.where(is_building_surf)[0]
    if progress_report:
        print(f"  palm_solar surfaces: {n_surfaces}, building surfaces: {len(bldg_indices)}")

    # Get or create building mesh - use cached mesh if available (expensive: ~2.4s)
    cache = _building_radiation_model_cache
    if building_svf_mesh is not None:
        # Use provided mesh directly (no copy needed - we only update metadata)
        building_mesh = building_svf_mesh
        # Extract SVF from mesh metadata if available
        if hasattr(building_mesh, 'metadata') and 'svf' in building_mesh.metadata:
            face_svf = building_mesh.metadata['svf']
        else:
            face_svf = None
        # Cache mesh geometry on first use (avoids recomputing triangles_center/face_normals)
        if cache is not None and cache.mesh_face_centers is None:
            cache.mesh_face_centers = building_mesh.triangles_center.copy()
            cache.mesh_face_normals = building_mesh.face_normals.copy()
    elif cache is not None and cache.cached_building_mesh is not None:
        # Use cached mesh (avoids expensive ~2.4s mesh creation each call)
        building_mesh = cache.cached_building_mesh
        face_svf = None
    else:
        # Create mesh for building surfaces (expensive, ~2.4s)
        try:
            from voxcity.geoprocessor.mesh import create_voxel_mesh
            if progress_report:
                print("  Creating building mesh (first call, will be cached)...")
            building_mesh = create_voxel_mesh(
                voxel_data,
                building_class_id,
                meshsize,
                building_id_grid=building_id_grid,
                mesh_type='open_air'
            )
            if building_mesh is None or len(building_mesh.faces) == 0:
                print("No building surfaces found.")
                return None
            # Cache the mesh for future calls
            if cache is not None:
                cache.cached_building_mesh = building_mesh
                if progress_report:
                    print(f"  Cached building mesh with {len(building_mesh.faces)} faces")
        except ImportError:
            print("VoxCity geoprocessor.mesh required for mesh creation")
            return None
        face_svf = None

    n_mesh_faces = len(building_mesh.faces)

    # Map palm_solar building surface values to building mesh faces.
    # Use cached mapping if available (avoids expensive KDTree query every call)
    if len(bldg_indices) > 0:
        # Check if we have cached mesh_to_surface_idx mapping
        if (cache is not None and 
            cache.mesh_to_surface_idx is not None and 
            len(cache.mesh_to_surface_idx) == n_mesh_faces):
            # Use cached direct mapping: mesh face -> surface index
            mesh_to_surface_idx = cache.mesh_to_surface_idx
            
            # Fast vectorized indexing using pre-computed mapping
            sw_in_direct = sw_in_direct_all[mesh_to_surface_idx]
            sw_in_diffuse = sw_in_diffuse_all[mesh_to_surface_idx]
            sw_in_reflected = sw_in_reflected_all[mesh_to_surface_idx]
            total_sw = total_sw_all[mesh_to_surface_idx]
        else:
            # Need to compute mapping (first call with this mesh)
            from scipy.spatial import cKDTree
            
            # Get surface centers (only needed for KDTree building)
            surf_centers_all = model.surfaces.center.to_numpy()[:n_surfaces]
            bldg_centers = surf_centers_all[bldg_indices]
            
            # Use cached geometry if available, otherwise compute from mesh
            if cache is not None and cache.mesh_face_centers is not None:
                mesh_face_centers = cache.mesh_face_centers
            else:
                mesh_face_centers = building_mesh.triangles_center
                if cache is not None:
                    cache.mesh_face_centers = mesh_face_centers.copy()
                    cache.mesh_face_normals = building_mesh.face_normals.copy()
            
            if progress_report:
                print(f"  Computing mesh-to-surface mapping (first call)...")
                print(f"  palm_solar bldg centers: x=[{bldg_centers[:,0].min():.1f}, {bldg_centers[:,0].max():.1f}], "
                      f"y=[{bldg_centers[:,1].min():.1f}, {bldg_centers[:,1].max():.1f}], "
                      f"z=[{bldg_centers[:,2].min():.1f}, {bldg_centers[:,2].max():.1f}]")
                print(f"  mesh face centers: x=[{mesh_face_centers[:,0].min():.1f}, {mesh_face_centers[:,0].max():.1f}], "
                      f"y=[{mesh_face_centers[:,1].min():.1f}, {mesh_face_centers[:,1].max():.1f}], "
                      f"z=[{mesh_face_centers[:,2].min():.1f}, {mesh_face_centers[:,2].max():.1f}]")
            
            tree = cKDTree(bldg_centers)
            distances, nearest_idx = tree.query(mesh_face_centers, k=1)
            
            if progress_report:
                print(f"  KDTree match distances: min={distances.min():.2f}, mean={distances.mean():.2f}, max={distances.max():.2f}")
            
            # Create direct mapping: mesh face -> surface index
            # This combines bldg_indices[nearest_idx] into a single array
            mesh_to_surface_idx = bldg_indices[nearest_idx]
            
            # Cache the mapping for subsequent calls
            if cache is not None:
                cache.mesh_to_surface_idx = mesh_to_surface_idx
                cache.bldg_indices = bldg_indices
                if progress_report:
                    print(f"  Cached mesh-to-surface mapping for {n_mesh_faces} faces")
            
            # Map irradiance arrays
            sw_in_direct = sw_in_direct_all[mesh_to_surface_idx]
            sw_in_diffuse = sw_in_diffuse_all[mesh_to_surface_idx]
            sw_in_reflected = sw_in_reflected_all[mesh_to_surface_idx]
            total_sw = total_sw_all[mesh_to_surface_idx]
    else:
        # Fallback: no building surfaces in palm_solar model (edge case)
        sw_in_direct = np.zeros(n_mesh_faces, dtype=np.float32)
        sw_in_diffuse = np.zeros(n_mesh_faces, dtype=np.float32)
        sw_in_reflected = np.zeros(n_mesh_faces, dtype=np.float32)
        total_sw = np.zeros(n_mesh_faces, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Set vertical faces on domain perimeter to NaN (matching VoxCity behavior)
    # Use cached boundary mask if available to avoid expensive mesh ops
    # -------------------------------------------------------------------------
    cache = _building_radiation_model_cache
    if cache is not None and cache.boundary_mask is not None and len(cache.boundary_mask) == n_mesh_faces:
        # Use cached boundary mask
        is_boundary_vertical = cache.boundary_mask
    else:
        # Compute and cache boundary mask (first call)
        ny_vc, nx_vc, nz = voxel_data.shape
        grid_bounds_real = np.array([
            [0.0, 0.0, 0.0],
            [nx_vc * meshsize, ny_vc * meshsize, nz * meshsize]
        ], dtype=np.float64)
        boundary_epsilon = meshsize * 0.05

        # Use cached geometry if available, otherwise compute and cache
        if cache is not None and cache.mesh_face_centers is not None:
            mesh_face_centers = cache.mesh_face_centers
            mesh_face_normals = cache.mesh_face_normals
        else:
            mesh_face_centers = building_mesh.triangles_center
            mesh_face_normals = building_mesh.face_normals
            # Cache geometry for future calls
            if cache is not None:
                cache.mesh_face_centers = mesh_face_centers
                cache.mesh_face_normals = mesh_face_normals

        # Detect vertical faces (normal z-component near zero)
        is_vertical = np.abs(mesh_face_normals[:, 2]) < 0.01

        # Detect faces on domain boundary
        on_x_min = np.abs(mesh_face_centers[:, 0] - grid_bounds_real[0, 0]) < boundary_epsilon
        on_y_min = np.abs(mesh_face_centers[:, 1] - grid_bounds_real[0, 1]) < boundary_epsilon
        on_x_max = np.abs(mesh_face_centers[:, 0] - grid_bounds_real[1, 0]) < boundary_epsilon
        on_y_max = np.abs(mesh_face_centers[:, 1] - grid_bounds_real[1, 1]) < boundary_epsilon

        is_boundary_vertical = is_vertical & (on_x_min | on_y_min | on_x_max | on_y_max)
        
        # Cache the boundary mask
        if cache is not None:
            cache.boundary_mask = is_boundary_vertical

    # Set boundary vertical faces to NaN using np.where (avoids expensive astype conversion)
    sw_in_direct = np.where(is_boundary_vertical, np.nan, sw_in_direct)
    sw_in_diffuse = np.where(is_boundary_vertical, np.nan, sw_in_diffuse)
    sw_in_reflected = np.where(is_boundary_vertical, np.nan, sw_in_reflected)
    total_sw = np.where(is_boundary_vertical, np.nan, total_sw)

    if progress_report:
        n_boundary = np.sum(is_boundary_vertical)
        print(f"  Boundary vertical faces set to NaN: {n_boundary}/{n_mesh_faces} ({100*n_boundary/n_mesh_faces:.1f}%)")

    building_mesh.metadata = {
        'irradiance_direct': sw_in_direct,
        'irradiance_diffuse': sw_in_diffuse,
        'irradiance_reflected': sw_in_reflected,
        'irradiance_total': total_sw,
        'direct': sw_in_direct,  # VoxCity API compatibility alias
        'diffuse': sw_in_diffuse,  # VoxCity API compatibility alias
        'global': total_sw,  # VoxCity API compatibility alias
    }
    if face_svf is not None:
        building_mesh.metadata['svf'] = face_svf
    
    if kwargs.get('obj_export', False):
        import os
        output_dir = kwargs.get('output_directory', 'output')
        output_file_name = kwargs.get('output_file_name', 'building_solar_irradiance')
        os.makedirs(output_dir, exist_ok=True)
        try:
            building_mesh.export(f"{output_dir}/{output_file_name}.obj")
            if progress_report:
                print(f"Exported to {output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    
    return building_mesh


def get_cumulative_building_solar_irradiance(
    voxcity,
    building_svf_mesh,
    weather_df,
    lon: float,
    lat: float,
    tz: float,
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    **kwargs
):
    """
    GPU-accelerated cumulative solar irradiance on building surfaces.
    
    This function matches the signature of voxcity.simulator.solar.get_cumulative_building_solar_irradiance
    using Taichi GPU acceleration.
    
    Integrates solar irradiance over a time period from weather data,
    returning cumulative Wh/m² on building faces.
    
    Args:
        voxcity: VoxCity object
        building_svf_mesh: Trimesh object with SVF in metadata
        weather_df: pandas DataFrame with 'DNI' and 'DHI' columns
        lon: Longitude in degrees
        lat: Latitude in degrees
        tz: Timezone offset in hours
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        **kwargs: Additional parameters including:
            - period_start (str): Start time 'MM-DD HH:MM:SS' (default: '01-01 00:00:00')
            - period_end (str): End time 'MM-DD HH:MM:SS' (default: '12-31 23:59:59')
            - time_step_hours (float): Time step in hours (default: 1.0)
            - use_sky_patches (bool): Use sky patch optimization (default: True)
            - sky_discretization (str): 'tregenza', 'reinhart', etc.
            - progress_report (bool): Print progress (default: False)
            - with_reflections (bool): Enable multi-bounce surface reflections (default: False).
                Set to True for more accurate results but slower computation.
            - fast_path (bool): Use optimized paths (default: True)
    
    Returns:
        Trimesh object with cumulative irradiance (Wh/m²) in metadata
    """
    from datetime import datetime
    import pytz
    
    # Extract parameters that we pass explicitly (use pop to avoid duplicate kwargs)
    kwargs = dict(kwargs)  # Copy to avoid modifying original
    period_start = kwargs.pop('period_start', '01-01 00:00:00')
    period_end = kwargs.pop('period_end', '12-31 23:59:59')
    time_step_hours = float(kwargs.pop('time_step_hours', 1.0))
    progress_report = kwargs.pop('progress_report', False)
    use_sky_patches = kwargs.pop('use_sky_patches', False)  # Default False for accuracy, True for speed
    
    if weather_df.empty:
        raise ValueError("No data in weather dataframe.")
    
    # Parse period
    try:
        start_dt = datetime.strptime(period_start, '%m-%d %H:%M:%S')
        end_dt = datetime.strptime(period_end, '%m-%d %H:%M:%S')
    except ValueError:
        raise ValueError("period_start and period_end must be in format 'MM-DD HH:MM:SS'")
    
    # Filter dataframe to period
    df = weather_df.copy()
    df['hour_of_year'] = (df.index.dayofyear - 1) * 24 + df.index.hour + 1
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1
    
    if start_hour <= end_hour:
        df_period = df[(df['hour_of_year'] >= start_hour) & (df['hour_of_year'] <= end_hour)]
    else:
        df_period = df[(df['hour_of_year'] >= start_hour) | (df['hour_of_year'] <= end_hour)]
    
    if df_period.empty:
        raise ValueError("No weather data in the specified period.")
    
    # Localize and convert to UTC
    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)
    
    # Get solar positions
    solar_positions = _get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Initialize cumulative arrays
    result_mesh = building_svf_mesh.copy() if hasattr(building_svf_mesh, 'copy') else building_svf_mesh
    n_faces = len(result_mesh.faces) if hasattr(result_mesh, 'faces') else 0
    
    if n_faces == 0:
        raise ValueError("Building mesh has no faces")
    
    cumulative_direct = np.zeros(n_faces, dtype=np.float64)
    cumulative_diffuse = np.zeros(n_faces, dtype=np.float64)
    cumulative_global = np.zeros(n_faces, dtype=np.float64)
    
    # Get SVF from mesh if available
    face_svf = None
    if hasattr(result_mesh, 'metadata') and 'svf' in result_mesh.metadata:
        face_svf = result_mesh.metadata['svf']
    
    if progress_report:
        print(f"Computing cumulative irradiance for {n_faces} faces...")
    
    # Extract arrays for processing
    azimuth_arr = solar_positions['azimuth'].to_numpy()
    elevation_arr = solar_positions['elevation'].to_numpy()
    dni_arr = df_period_utc['DNI'].to_numpy() * direct_normal_irradiance_scaling
    dhi_arr = df_period_utc['DHI'].to_numpy() * diffuse_irradiance_scaling
    n_timesteps = len(azimuth_arr)
    
    if use_sky_patches:
        # Use sky patch aggregation for efficiency (same as ground-level)
        from .sky import generate_sky_patches, get_tregenza_patch_index
        
        sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
        
        # Get method-specific parameters
        sky_kwargs = {}
        if sky_discretization.lower() == 'reinhart':
            sky_kwargs['mf'] = kwargs.pop('reinhart_mf', kwargs.pop('mf', 4))
        elif sky_discretization.lower() == 'uniform':
            sky_kwargs['n_azimuth'] = kwargs.pop('sky_n_azimuth', kwargs.pop('n_azimuth', 36))
            sky_kwargs['n_elevation'] = kwargs.pop('sky_n_elevation', kwargs.pop('n_elevation', 9))
        elif sky_discretization.lower() == 'fibonacci':
            sky_kwargs['n_patches'] = kwargs.pop('sky_n_patches', kwargs.pop('n_patches', 145))
        
        # Generate sky patches using unified interface
        sky_patches = generate_sky_patches(sky_discretization, **sky_kwargs)
        patches = sky_patches.patches  # (N, 2) azimuth, elevation
        directions = sky_patches.directions  # (N, 3) unit vectors
        
        n_patches = sky_patches.n_patches
        cumulative_dni_per_patch = np.zeros(n_patches, dtype=np.float64)
        total_cumulative_dhi = 0.0
        
        # Bin sun positions to patches
        for i in range(n_timesteps):
            elev = elevation_arr[i]
            dhi = dhi_arr[i]
            
            if dhi > 0:
                total_cumulative_dhi += dhi * time_step_hours
            
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            
            if dni <= 0:
                continue
            
            # Find nearest patch based on method
            if sky_discretization.lower() == 'tregenza':
                patch_idx = int(get_tregenza_patch_index(float(az), float(elev)))
            else:
                # For other methods, find nearest patch by direction vector
                elev_rad = np.deg2rad(elev)
                az_rad = np.deg2rad(az)
                sun_dir = np.array([
                    np.cos(elev_rad) * np.sin(az_rad),  # East
                    np.cos(elev_rad) * np.cos(az_rad),  # North  
                    np.sin(elev_rad)                     # Up
                ])
                dots = np.sum(directions * sun_dir, axis=1)
                patch_idx = int(np.argmax(dots))
            
            if 0 <= patch_idx < n_patches:
                cumulative_dni_per_patch[patch_idx] += dni * time_step_hours
        
        active_mask = cumulative_dni_per_patch > 0
        n_active = int(np.sum(active_mask))
        
        if progress_report:
            print(f"  Sky patch optimization: {n_timesteps} timesteps -> {n_active} active patches ({sky_discretization})")
            print(f"  Total cumulative DHI: {total_cumulative_dhi:.1f} Wh/m²")
        
        # First pass: compute diffuse component using SVF (if available) or a single call
        if face_svf is not None and len(face_svf) == n_faces:
            cumulative_diffuse = face_svf * total_cumulative_dhi
        else:
            # Compute diffuse using a single call with sun at zenith
            diffuse_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=180.0,
                elevation_degrees=45.0,
                direct_normal_irradiance=0.0,
                diffuse_irradiance=1.0,
                progress_report=False,
                **kwargs
            )
            if diffuse_mesh is not None and 'diffuse' in diffuse_mesh.metadata:
                base_diffuse = diffuse_mesh.metadata['diffuse']
                cumulative_diffuse = np.nan_to_num(base_diffuse, nan=0.0) * total_cumulative_dhi
        
        # Second pass: loop over active patches for direct component
        active_indices = np.where(active_mask)[0]
        for i, patch_idx in enumerate(active_indices):
            az_deg = patches[patch_idx, 0]
            el_deg = patches[patch_idx, 1]
            cumulative_dni_patch = cumulative_dni_per_patch[patch_idx]
            
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=az_deg,
                elevation_degrees=el_deg,
                direct_normal_irradiance=1.0,  # Unit irradiance, scale by cumulative
                diffuse_irradiance=0.0,  # Diffuse handled separately
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and hasattr(irradiance_mesh, 'metadata'):
                if 'direct' in irradiance_mesh.metadata:
                    direct_vals = irradiance_mesh.metadata['direct']
                    if len(direct_vals) == n_faces:
                        cumulative_direct += np.nan_to_num(direct_vals, nan=0.0) * cumulative_dni_patch
            
            if progress_report and ((i + 1) % max(1, len(active_indices) // 10) == 0 or i == len(active_indices) - 1):
                pct = (i + 1) * 100.0 / len(active_indices)
                print(f"  Patch {i+1}/{len(active_indices)} ({pct:.1f}%)")
        
        # Combine direct and diffuse
        cumulative_global = cumulative_direct + cumulative_diffuse
    
    else:
        # Per-timestep path (no optimization)
        if progress_report:
            print(f"  Processing {n_timesteps} timesteps (no sky patch optimization)...")
        
        for t_idx, (timestamp, row) in enumerate(df_period_utc.iterrows()):
            dni = float(row['DNI']) * direct_normal_irradiance_scaling
            dhi = float(row['DHI']) * diffuse_irradiance_scaling
            
            elevation = float(solar_positions.loc[timestamp, 'elevation'])
            azimuth = float(solar_positions.loc[timestamp, 'azimuth'])
            
            # Skip nighttime
            if elevation <= 0 or (dni <= 0 and dhi <= 0):
                continue
            
            # Compute instantaneous irradiance for this timestep
            irradiance_mesh = get_building_solar_irradiance(
                voxcity,
                building_svf_mesh=building_svf_mesh,
                azimuth_degrees_ori=azimuth,
                elevation_degrees=elevation,
                direct_normal_irradiance=dni,
                diffuse_irradiance=dhi,
                progress_report=False,
                **kwargs
            )
            
            if irradiance_mesh is not None and hasattr(irradiance_mesh, 'metadata'):
                # Accumulate (convert W/m² to Wh/m² by multiplying by time_step_hours)
                if 'direct' in irradiance_mesh.metadata:
                    direct_vals = irradiance_mesh.metadata['direct']
                    if len(direct_vals) == n_faces:
                        cumulative_direct += np.nan_to_num(direct_vals, nan=0.0) * time_step_hours
                if 'diffuse' in irradiance_mesh.metadata:
                    diffuse_vals = irradiance_mesh.metadata['diffuse']
                    if len(diffuse_vals) == n_faces:
                        cumulative_diffuse += np.nan_to_num(diffuse_vals, nan=0.0) * time_step_hours
                if 'global' in irradiance_mesh.metadata:
                    global_vals = irradiance_mesh.metadata['global']
                    if len(global_vals) == n_faces:
                        cumulative_global += np.nan_to_num(global_vals, nan=0.0) * time_step_hours
            
            if progress_report and (t_idx + 1) % max(1, n_timesteps // 10) == 0:
                print(f"  Processed {t_idx + 1}/{n_timesteps} timesteps ({100*(t_idx+1)/n_timesteps:.1f}%)")
    
    # -------------------------------------------------------------------------
    # Set vertical faces on domain perimeter to NaN (matching VoxCity behavior)
    # -------------------------------------------------------------------------
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ny_vc, nx_vc, nz = voxel_data.shape
    grid_bounds_real = np.array([
        [0.0, 0.0, 0.0],
        [ny_vc * meshsize, nx_vc * meshsize, nz * meshsize]
    ], dtype=np.float64)
    boundary_epsilon = meshsize * 0.05

    mesh_face_centers = result_mesh.triangles_center
    mesh_face_normals = result_mesh.face_normals

    # Detect vertical faces (normal z-component near zero)
    is_vertical = np.abs(mesh_face_normals[:, 2]) < 0.01

    # Detect faces on domain boundary
    on_x_min = np.abs(mesh_face_centers[:, 0] - grid_bounds_real[0, 0]) < boundary_epsilon
    on_y_min = np.abs(mesh_face_centers[:, 1] - grid_bounds_real[0, 1]) < boundary_epsilon
    on_x_max = np.abs(mesh_face_centers[:, 0] - grid_bounds_real[1, 0]) < boundary_epsilon
    on_y_max = np.abs(mesh_face_centers[:, 1] - grid_bounds_real[1, 1]) < boundary_epsilon

    is_boundary_vertical = is_vertical & (on_x_min | on_y_min | on_x_max | on_y_max)

    # Set boundary vertical faces to NaN
    cumulative_direct[is_boundary_vertical] = np.nan
    cumulative_diffuse[is_boundary_vertical] = np.nan
    cumulative_global[is_boundary_vertical] = np.nan

    if progress_report:
        n_boundary = np.sum(is_boundary_vertical)
        print(f"  Boundary vertical faces set to NaN: {n_boundary}/{n_faces} ({100*n_boundary/n_faces:.1f}%)")

    # Store results in mesh metadata
    result_mesh.metadata = getattr(result_mesh, 'metadata', {})
    result_mesh.metadata['cumulative_direct'] = cumulative_direct
    result_mesh.metadata['cumulative_diffuse'] = cumulative_diffuse
    result_mesh.metadata['cumulative_global'] = cumulative_global
    result_mesh.metadata['direct'] = cumulative_direct  # VoxCity API alias
    result_mesh.metadata['diffuse'] = cumulative_diffuse  # VoxCity API alias
    result_mesh.metadata['global'] = cumulative_global  # VoxCity API alias
    if face_svf is not None:
        result_mesh.metadata['svf'] = face_svf
    
    if progress_report:
        valid_mask = ~np.isnan(cumulative_global)
        total_irradiance = np.nansum(cumulative_global)
        print(f"Cumulative irradiance computation complete:")
        print(f"  Total faces: {n_faces}, Valid: {np.sum(valid_mask)}")
        print(f"  Mean cumulative: {np.nanmean(cumulative_global):.1f} Wh/m²")
        print(f"  Max cumulative: {np.nanmax(cumulative_global):.1f} Wh/m²")
    
    # Export if requested
    if kwargs.get('obj_export', False):
        import os
        output_dir = kwargs.get('output_directory', 'output')
        output_file_name = kwargs.get('output_file_name', 'cumulative_building_irradiance')
        os.makedirs(output_dir, exist_ok=True)
        try:
            result_mesh.export(f"{output_dir}/{output_file_name}.obj")
            if progress_report:
                print(f"Exported to {output_dir}/{output_file_name}.obj")
        except Exception as e:
            print(f"Error exporting mesh: {e}")
    
    return result_mesh


def get_building_global_solar_irradiance_using_epw(
    voxcity,
    calc_type: str = 'instantaneous',
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    building_svf_mesh=None,
    **kwargs
):
    """
    GPU-accelerated building surface irradiance using EPW weather data.
    
    This function matches the signature of voxcity.simulator.solar.get_building_global_solar_irradiance_using_epw
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        calc_type: 'instantaneous' or 'cumulative'
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        building_svf_mesh: Pre-computed building mesh with SVF (optional)
        **kwargs: Additional parameters including:
            - epw_file_path (str): Path to EPW file
            - download_nearest_epw (bool): Download nearest EPW (default: False)
            - calc_time (str): For instantaneous: 'MM-DD HH:MM:SS'
            - period_start, period_end (str): For cumulative: 'MM-DD HH:MM:SS'
            - rectangle_vertices: Location vertices
            - progress_report (bool): Print progress
            - with_reflections (bool): Enable multi-bounce surface reflections (default: False).
                Set to True for more accurate results but slower computation.
    
    Returns:
        Trimesh object with irradiance values (W/m² or Wh/m²) in metadata
    """
    from datetime import datetime
    import pytz
    
    # NOTE: We frequently forward **kwargs to lower-level functions; ensure
    # we don't pass duplicate keyword args (e.g., progress_report).
    progress_report = kwargs.get('progress_report', False)
    kwargs = dict(kwargs)
    kwargs.pop('progress_report', None)
    
    # Load EPW data using helper function
    df, lon, lat, tz = _load_epw_data(
        epw_file_path=kwargs.pop('epw_file_path', None),
        download_nearest_epw=kwargs.pop('download_nearest_epw', False),
        voxcity=voxcity,
        **kwargs
    )
    
    # Create building mesh for output (just geometry, no SVF computation)
    # The RadiationModel computes SVF internally for voxel surfaces, so we don't need
    # the expensive get_surface_view_factor() call. We just need the mesh geometry.
    if building_svf_mesh is None:
        try:
            from voxcity.geoprocessor.mesh import create_voxel_mesh
            building_class_id = kwargs.get('building_class_id', -3)
            voxel_data = voxcity.voxels.classes
            meshsize = voxcity.voxels.meta.meshsize
            building_id_grid = voxcity.buildings.ids
            
            building_svf_mesh = create_voxel_mesh(
                voxel_data,
                building_class_id,
                meshsize,
                building_id_grid=building_id_grid,
                mesh_type='open_air'
            )
            if progress_report:
                n_faces = len(building_svf_mesh.faces) if building_svf_mesh is not None else 0
                print(f"Created building mesh with {n_faces} faces")
        except ImportError:
            pass  # Will fail later with "Building mesh has no faces" error
    
    if calc_type == 'instantaneous':
        calc_time = kwargs.get('calc_time', '01-01 12:00:00')
        try:
            calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%S')
        except ValueError:
            raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'")
        
        df_period = df[
            (df.index.month == calc_dt.month) &
            (df.index.day == calc_dt.day) &
            (df.index.hour == calc_dt.hour)
        ]
        if df_period.empty:
            raise ValueError("No EPW data at the specified time.")
        
        # Get solar position
        offset_minutes = int(tz * 60)
        local_tz = pytz.FixedOffset(offset_minutes)
        df_local = df_period.copy()
        df_local.index = df_local.index.tz_localize(local_tz)
        df_utc = df_local.tz_convert(pytz.UTC)
        
        solar_positions = _get_solar_positions_astral(df_utc.index, lon, lat)
        DNI = float(df_utc.iloc[0]['DNI']) * direct_normal_irradiance_scaling
        DHI = float(df_utc.iloc[0]['DHI']) * diffuse_irradiance_scaling
        azimuth_degrees = float(solar_positions.iloc[0]['azimuth'])
        elevation_degrees = float(solar_positions.iloc[0]['elevation'])
        
        return get_building_solar_irradiance(
            voxcity,
            building_svf_mesh=building_svf_mesh,
            azimuth_degrees_ori=azimuth_degrees,
            elevation_degrees=elevation_degrees,
            direct_normal_irradiance=DNI,
            diffuse_irradiance=DHI,
            **kwargs
        )
    
    elif calc_type == 'cumulative':
        period_start = kwargs.get('period_start', '01-01 00:00:00')
        period_end = kwargs.get('period_end', '12-31 23:59:59')
        time_step_hours = float(kwargs.get('time_step_hours', 1.0))

        # Avoid passing duplicates: we pass these explicitly below.
        kwargs.pop('period_start', None)
        kwargs.pop('period_end', None)
        kwargs.pop('time_step_hours', None)
        
        return get_cumulative_building_solar_irradiance(
            voxcity,
            building_svf_mesh=building_svf_mesh,
            weather_df=df,
            lon=lon,
            lat=lat,
            tz=tz,
            direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
            diffuse_irradiance_scaling=diffuse_irradiance_scaling,
            period_start=period_start,
            period_end=period_end,
            time_step_hours=time_step_hours,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown calc_type: {calc_type}. Use 'instantaneous' or 'cumulative'.")


# =============================================================================
# Volumetric Solar Irradiance Functions
# =============================================================================
# Module-level cache for VolumetricFluxCalculator
_volumetric_flux_cache: Optional[Dict] = None


def clear_volumetric_flux_cache():
    """Clear the cached VolumetricFluxCalculator to free memory or force recomputation."""
    global _volumetric_flux_cache
    _volumetric_flux_cache = None


def _get_or_create_volumetric_calculator(
    voxcity,
    n_azimuth: int = 36,
    n_zenith: int = 9,
    progress_report: bool = False,
    **kwargs
):
    """
    Get cached VolumetricFluxCalculator or create a new one if cache is invalid.
    
    Args:
        voxcity: VoxCity object
        n_azimuth: Number of azimuthal directions
        n_zenith: Number of zenith angle divisions
        progress_report: Print progress messages
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (VolumetricFluxCalculator, Domain)
    """
    global _volumetric_flux_cache
    
    from .volumetric import VolumetricFluxCalculator
    from .domain import Domain
    
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
    
    # Need to create new calculator
    if progress_report:
        print("Creating new VolumetricFluxCalculator...")
    
    # Get location using helper function
    origin_lat, origin_lon = _get_location_from_voxcity(voxcity)
    
    # Create domain
    domain = Domain(
        nx=ni, ny=nj, nz=nk,
        dx=meshsize, dy=meshsize, dz=meshsize,
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    # Convert VoxCity voxel data to domain arrays using vectorized helper
    default_lad = kwargs.get('default_lad', 1.0)
    is_solid_np, lad_np = _convert_voxel_data_to_arrays(voxel_data, default_lad)
    
    # Set domain arrays
    _set_solid_array(domain, is_solid_np)
    domain.set_lad_from_array(lad_np)
    _update_topo_from_solid(domain)
    
    # Create VolumetricFluxCalculator
    calculator = VolumetricFluxCalculator(
        domain,
        n_azimuth=n_azimuth,
        min_opaque_lad=kwargs.get('min_opaque_lad', 0.5)
    )
    
    # Compute volumetric sky view factors (expensive, do once)
    if progress_report:
        print("Computing volumetric sky view factors...")
    calculator.compute_skyvf_vol(n_zenith=n_zenith)
    
    # Cache the calculator
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


def get_volumetric_solar_irradiance_map(
    voxcity,
    azimuth_degrees_ori: float,
    elevation_degrees: float,
    direct_normal_irradiance: float,
    diffuse_irradiance: float,
    volumetric_height: float = 1.5,
    with_reflections: bool = False,
    show_plot: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated volumetric solar irradiance map at a specified height.
    
    Computes the 3D radiation field at each grid cell and extracts a 2D horizontal
    slice at the specified height. This is useful for:
    - Mean Radiant Temperature (MRT) calculations
    - Pedestrian thermal comfort analysis
    - Light availability assessment
    
    Args:
        voxcity: VoxCity object
        azimuth_degrees_ori: Solar azimuth in degrees (0=North, clockwise)
        elevation_degrees: Solar elevation in degrees above horizon
        direct_normal_irradiance: DNI in W/m²
        diffuse_irradiance: DHI in W/m²
        volumetric_height: Height above ground for irradiance extraction (meters)
        with_reflections: If True, include reflected radiation from surfaces.
            If False (default), only direct + diffuse sky radiation.
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters:
            - n_azimuth (int): Number of azimuthal directions for SVF (default: 36)
            - n_zenith (int): Number of zenith angles for SVF (default: 9)
            - progress_report (bool): Print progress (default: False)
            - colormap (str): Colormap for plot (default: 'magma')
            - vmin, vmax (float): Colormap bounds
            - n_reflection_steps (int): Reflection bounces when with_reflections=True (default: 2)
    
    Returns:
        2D numpy array of volumetric irradiance at the specified height (W/m²)
    """
    import math
    
    kwargs = kwargs.copy()  # Don't modify caller's kwargs
    progress_report = kwargs.pop('progress_report', False)
    n_azimuth = kwargs.pop('n_azimuth', 36)
    n_zenith = kwargs.pop('n_zenith', 9)
    
    # Get or create cached calculator
    calculator, domain = _get_or_create_volumetric_calculator(
        voxcity,
        n_azimuth=n_azimuth,
        n_zenith=n_zenith,
        progress_report=progress_report,
        **kwargs
    )
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ni, nj, nk = voxel_data.shape
    
    # Convert solar angles to direction vector
    # Match the coordinate system used in ground-level functions:
    # azimuth_degrees_ori: 0=North, 90=East (clockwise from North)
    # Transform to model coordinates: 180 - azimuth_degrees_ori
    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_rad = math.radians(azimuth_degrees)
    elevation_rad = math.radians(elevation_degrees)
    
    cos_elev = math.cos(elevation_rad)
    sin_elev = math.sin(elevation_rad)
    
    # Direction toward sun (matching ground-level function convention)
    sun_dir_x = cos_elev * math.cos(azimuth_rad)
    sun_dir_y = cos_elev * math.sin(azimuth_rad)
    sun_dir_z = sin_elev
    sun_direction = (sun_dir_x, sun_dir_y, sun_dir_z)
    
    cos_zenith = sin_elev  # cos(zenith) = sin(elevation)
    
    # Compute volumetric flux
    if with_reflections:
        # Use full reflection model
        if progress_report:
            print("Computing volumetric flux with reflections...")
        
        # Get radiation model for surface reflections
        n_reflection_steps = kwargs.pop('n_reflection_steps', 2)
        model, valid_ground, ground_k = _get_or_create_radiation_model(
            voxcity,
            n_reflection_steps=n_reflection_steps,
            progress_report=progress_report,
            **kwargs
        )
        
        # Manually set solar position on the model's solar_calc
        model.solar_calc.cos_zenith[None] = cos_zenith
        model.solar_calc.sun_direction[None] = [sun_direction[0], sun_direction[1], sun_direction[2]]
        model.solar_calc.sun_up[None] = 1 if cos_zenith > 0 else 0
        
        # Compute shortwave radiation with the set solar position
        model.compute_shortwave_radiation(
            sw_direct=direct_normal_irradiance,
            sw_diffuse=diffuse_irradiance
        )
        
        # Get surface outgoing radiation
        n_surfaces = model.surfaces.count
        surf_outgoing = model.surfaces.sw_out.to_numpy()[:n_surfaces]
        
        # Compute volumetric flux including reflections
        # Use cached C2S-VF matrix if available, otherwise compute dynamically
        if calculator.c2s_matrix_cached:
            calculator.compute_swflux_vol_with_reflections_cached(
                sw_direct=direct_normal_irradiance,
                sw_diffuse=diffuse_irradiance,
                cos_zenith=cos_zenith,
                sun_direction=sun_direction,
                surf_outgoing=surf_outgoing,
                lad=domain.lad
            )
        else:
            calculator.compute_swflux_vol_with_reflections(
                sw_direct=direct_normal_irradiance,
                sw_diffuse=diffuse_irradiance,
                cos_zenith=cos_zenith,
                sun_direction=sun_direction,
                surfaces=model.surfaces,
                surf_outgoing=surf_outgoing,
                lad=domain.lad
            )
    else:
        # Simple direct + diffuse only
        if progress_report:
            print("Computing volumetric flux (direct + diffuse)...")
        
        calculator.compute_swflux_vol(
            sw_direct=direct_normal_irradiance,
            sw_diffuse=diffuse_irradiance,
            cos_zenith=cos_zenith,
            sun_direction=sun_direction,
            lad=domain.lad
        )
        
        # Compute ground_k for terrain-following extraction
        ground_k = _compute_ground_k_from_voxels(voxel_data)
    
    # Extract terrain-following horizontal slice at specified height above ground
    # For each (i,j), extract at ground_k[i,j] + height_offset_k
    height_offset_k = max(1, int(round(volumetric_height / meshsize)))
    if progress_report:
        print(f"Extracting volumetric irradiance at {volumetric_height}m above terrain (offset={height_offset_k} cells)")
    
    # Get full 3D volumetric flux and is_solid arrays
    swflux_3d = calculator.get_swflux_vol()
    is_solid = domain.is_solid.to_numpy()
    
    # Create output array
    volumetric_map = np.full((ni, nj), np.nan, dtype=np.float64)
    
    # Extract terrain-following values
    for i in range(ni):
        for j in range(nj):
            gk = ground_k[i, j]
            if gk < 0:
                # No valid ground - keep NaN
                continue
            k_extract = gk + height_offset_k
            if k_extract >= nk:
                # Above domain - keep NaN
                continue
            # Check if extraction point is in a solid cell
            if is_solid[i, j, k_extract] == 1:
                # Inside solid (building) - keep NaN
                continue
            volumetric_map[i, j] = swflux_3d[i, j, k_extract]
    
    # Flip to match VoxCity coordinate system
    volumetric_map = np.flipud(volumetric_map)
    
    if show_plot:
        colormap = kwargs.get('colormap', 'magma')
        vmin = kwargs.get('vmin', 0.0)
        vmax = kwargs.get('vmax', max(float(np.nanmax(volumetric_map)), 1.0))
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(volumetric_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label=f'Volumetric Solar Irradiance at {volumetric_height}m (W/m²)')
            plt.title(f'Volumetric Irradiance (reflections={"on" if with_reflections else "off"})')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    return volumetric_map


def get_cumulative_volumetric_solar_irradiance(
    voxcity,
    df,
    lon: float,
    lat: float,
    tz: float,
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    volumetric_height: float = 1.5,
    with_reflections: bool = False,
    show_plot: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated cumulative volumetric solar irradiance over a period.
    
    Integrates the 3D radiation field over time and extracts a 2D horizontal
    slice at the specified height.
    
    Args:
        voxcity: VoxCity object
        df: pandas DataFrame with 'DNI' and 'DHI' columns, datetime-indexed
        lon: Longitude in degrees
        lat: Latitude in degrees
        tz: Timezone offset in hours
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        volumetric_height: Height above ground for irradiance extraction (meters)
        with_reflections: If True, include reflected radiation from buildings,
            ground, and tree canopy surfaces. If False (default), only direct
            + diffuse sky radiation.
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters:
            - start_time (str): Start time 'MM-DD HH:MM:SS' (default: '01-01 05:00:00')
            - end_time (str): End time 'MM-DD HH:MM:SS' (default: '01-01 20:00:00')
            - use_sky_patches (bool): Use sky patch optimization (default: True)
            - sky_discretization (str): 'tregenza', 'reinhart', 'uniform', 'fibonacci'
            - progress_report (bool): Print progress (default: False)
            - n_reflection_steps (int): Reflection bounces when with_reflections=True (default: 2)
    
    Returns:
        2D numpy array of cumulative volumetric irradiance at the specified height (Wh/m²)
    """
    import time
    from datetime import datetime
    import pytz
    import math
    
    # Extract parameters
    kwargs = kwargs.copy()
    progress_report = kwargs.pop('progress_report', False)
    start_time = kwargs.pop('start_time', '01-01 05:00:00')
    end_time = kwargs.pop('end_time', '01-01 20:00:00')
    use_sky_patches = kwargs.pop('use_sky_patches', True)
    sky_discretization = kwargs.pop('sky_discretization', 'tregenza')
    n_azimuth = kwargs.pop('n_azimuth', 36)
    n_zenith = kwargs.pop('n_zenith', 9)
    
    if df.empty:
        raise ValueError("No data in EPW dataframe.")
    
    # Parse time range
    try:
        start_dt = datetime.strptime(start_time, '%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time, '%m-%d %H:%M:%S')
    except ValueError as ve:
        raise ValueError("start_time and end_time must be in format 'MM-DD HH:MM:SS'") from ve
    
    # Filter dataframe to period
    df = df.copy()
    df['hour_of_year'] = (df.index.dayofyear - 1) * 24 + df.index.hour + 1
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1
    
    if start_hour <= end_hour:
        df_period = df[(df['hour_of_year'] >= start_hour) & (df['hour_of_year'] <= end_hour)]
    else:
        df_period = df[(df['hour_of_year'] >= start_hour) | (df['hour_of_year'] <= end_hour)]
    
    if df_period.empty:
        raise ValueError("No EPW data in the specified period.")
    
    # Localize and convert to UTC
    offset_minutes = int(tz * 60)
    local_tz = pytz.FixedOffset(offset_minutes)
    df_period_local = df_period.copy()
    df_period_local.index = df_period_local.index.tz_localize(local_tz)
    df_period_utc = df_period_local.tz_convert(pytz.UTC)
    
    # Get solar positions
    solar_positions = _get_solar_positions_astral(df_period_utc.index, lon, lat)
    
    # Get or create cached calculator
    calculator, domain = _get_or_create_volumetric_calculator(
        voxcity,
        n_azimuth=n_azimuth,
        n_zenith=n_zenith,
        progress_report=progress_report,
        **kwargs
    )
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    ni, nj, nk = voxel_data.shape
    
    # Compute terrain-following extraction parameters
    height_offset_k = max(1, int(round(volumetric_height / meshsize)))
    if progress_report:
        print(f"Extracting volumetric irradiance at {volumetric_height}m above terrain (offset={height_offset_k} cells)")
    
    # Get is_solid array for masking
    is_solid = domain.is_solid.to_numpy()
    
    # Initialize cumulative map (will be NaN-masked at the end)
    cumulative_map = np.zeros((ni, nj), dtype=np.float64)
    time_step_hours = kwargs.get('time_step_hours', 1.0)
    
    # Get radiation model for reflections if needed
    model = None
    ground_k = None
    if with_reflections:
        n_reflection_steps = kwargs.pop('n_reflection_steps', 2)
        model, valid_ground, ground_k = _get_or_create_radiation_model(
            voxcity,
            n_reflection_steps=n_reflection_steps,
            progress_report=progress_report,
            **kwargs
        )
        
        # Note: C2S-VF matrix caching is disabled for large domains because
        # the matrix itself is too large (N_cells * N_surfaces can be > 1 trillion).
        # Instead, we use per-timestep computation with a distance cutoff which
        # is much more memory efficient.
    else:
        # Compute ground_k for terrain-following extraction
        ground_k = _compute_ground_k_from_voxels(voxel_data)
    
    # Extract arrays
    azimuth_arr = solar_positions['azimuth'].to_numpy()
    elevation_arr = solar_positions['elevation'].to_numpy()
    dni_arr = df_period_utc['DNI'].to_numpy() * direct_normal_irradiance_scaling
    dhi_arr = df_period_utc['DHI'].to_numpy() * diffuse_irradiance_scaling
    
    n_timesteps = len(azimuth_arr)
    
    if progress_report:
        print(f"Computing cumulative volumetric irradiance for {n_timesteps} timesteps...")
        print(f"  Height: {volumetric_height}m, Reflections: {'on' if with_reflections else 'off'}")
    
    t0 = time.perf_counter() if progress_report else 0
    
    if use_sky_patches:
        # Use sky patch aggregation for efficiency
        from .sky import (
            generate_tregenza_patches,
            generate_reinhart_patches,
            generate_uniform_grid_patches,
            generate_fibonacci_patches,
            get_tregenza_patch_index
        )
        
        # Generate sky patches
        if sky_discretization.lower() == 'tregenza':
            patches, directions, solid_angles = generate_tregenza_patches()
        elif sky_discretization.lower() == 'reinhart':
            mf = kwargs.get('reinhart_mf', kwargs.get('mf', 4))
            patches, directions, solid_angles = generate_reinhart_patches(mf=mf)
        elif sky_discretization.lower() == 'uniform':
            n_az = kwargs.get('sky_n_azimuth', 36)
            n_el = kwargs.get('sky_n_elevation', 9)
            patches, directions, solid_angles = generate_uniform_grid_patches(n_az, n_el)
        elif sky_discretization.lower() == 'fibonacci':
            n_patches = kwargs.get('sky_n_patches', 145)
            patches, directions, solid_angles = generate_fibonacci_patches(n_patches=n_patches)
        else:
            raise ValueError(f"Unknown sky discretization method: {sky_discretization}")
        
        n_patches = len(patches)
        cumulative_dni = np.zeros(n_patches, dtype=np.float64)
        total_dhi = 0.0
        
        # Bin sun positions to patches
        for i in range(n_timesteps):
            elev = elevation_arr[i]
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            dhi = dhi_arr[i]
            
            if dni > 0:
                patch_idx = get_tregenza_patch_index(az, elev)
                if 0 <= patch_idx < n_patches:
                    cumulative_dni[patch_idx] += dni * time_step_hours
            
            if dhi > 0:
                total_dhi += dhi * time_step_hours
        
        # Process each patch with accumulated DNI
        patches_with_dni = np.where(cumulative_dni > 0)[0]
        
        if progress_report:
            print(f"  Processing {len(patches_with_dni)} sky patches with accumulated DNI...")
        
        for idx, patch_idx in enumerate(patches_with_dni):
            patch_dni = cumulative_dni[patch_idx]
            patch_dir = directions[patch_idx]
            
            # Convert patch direction to azimuth/elevation
            patch_azimuth_ori = math.degrees(math.atan2(patch_dir[0], patch_dir[1]))
            if patch_azimuth_ori < 0:
                patch_azimuth_ori += 360
            patch_elevation = math.degrees(math.asin(patch_dir[2]))
            
            # Apply same coordinate transform as ground-level functions
            patch_azimuth = 180 - patch_azimuth_ori
            azimuth_rad = math.radians(patch_azimuth)
            elevation_rad = math.radians(patch_elevation)
            cos_elev = math.cos(elevation_rad)
            sin_elev = math.sin(elevation_rad)
            
            cos_zenith = sin_elev
            sun_dir_x = cos_elev * math.cos(azimuth_rad)
            sun_dir_y = cos_elev * math.sin(azimuth_rad)
            sun_dir_z = sin_elev
            sun_direction = (sun_dir_x, sun_dir_y, sun_dir_z)
            
            if with_reflections and model is not None:
                # Set solar position on the model
                model.solar_calc.cos_zenith[None] = cos_zenith
                model.solar_calc.sun_direction[None] = [sun_direction[0], sun_direction[1], sun_direction[2]]
                model.solar_calc.sun_up[None] = 1 if cos_zenith > 0 else 0
                
                # Compute surface irradiance with reflections (uses cached SVF matrix)
                model.compute_shortwave_radiation(
                    sw_direct=patch_dni / time_step_hours,  # Instantaneous for reflection calc
                    sw_diffuse=0.0  # DHI handled separately
                )
                
                n_surfaces = model.surfaces.count
                surf_outgoing = model.surfaces.sw_out.to_numpy()[:n_surfaces]
                
                # Use non-cached volumetric reflection computation with distance cutoff
                calculator.compute_swflux_vol_with_reflections(
                    sw_direct=patch_dni / time_step_hours,
                    sw_diffuse=0.0,
                    cos_zenith=cos_zenith,
                    sun_direction=sun_direction,
                    surfaces=model.surfaces,
                    surf_outgoing=surf_outgoing,
                    lad=domain.lad
                )
            else:
                calculator.compute_swflux_vol(
                    sw_direct=patch_dni / time_step_hours,
                    sw_diffuse=0.0,
                    cos_zenith=cos_zenith,
                    sun_direction=sun_direction,
                    lad=domain.lad
                )
            
            # Accumulate using terrain-following extraction
            swflux_3d = calculator.get_swflux_vol()
            _accumulate_terrain_following_slice(
                cumulative_map, swflux_3d, ground_k, height_offset_k, is_solid,
                weight=time_step_hours
            )
            
            if progress_report and (idx + 1) % 10 == 0:
                elapsed = time.perf_counter() - t0
                print(f"    Processed {idx + 1}/{len(patches_with_dni)} patches ({elapsed:.1f}s)")
        
        # Add diffuse contribution (use SVF-based diffuse with terrain-following extraction)
        if total_dhi > 0:
            svf_vol = calculator.get_skyvf_vol()
            _accumulate_terrain_following_slice(
                cumulative_map, svf_vol, ground_k, height_offset_k, is_solid,
                weight=total_dhi
            )
    
    else:
        # Process each timestep individually
        for i in range(n_timesteps):
            elev = elevation_arr[i]
            if elev <= 0:
                continue
            
            az = azimuth_arr[i]
            dni = dni_arr[i]
            dhi = dhi_arr[i]
            
            if dni <= 0 and dhi <= 0:
                continue
            
            # Convert to direction vector
            # Match the coordinate system used in ground-level functions
            azimuth_degrees = 180 - az
            azimuth_rad = math.radians(azimuth_degrees)
            elevation_rad = math.radians(elev)
            cos_elev = math.cos(elevation_rad)
            sin_elev = math.sin(elevation_rad)
            
            sun_dir_x = cos_elev * math.cos(azimuth_rad)
            sun_dir_y = cos_elev * math.sin(azimuth_rad)
            sun_dir_z = sin_elev
            sun_direction = (sun_dir_x, sun_dir_y, sun_dir_z)
            cos_zenith = sin_elev
            
            if with_reflections and model is not None:
                # Set solar position on the model
                model.solar_calc.cos_zenith[None] = cos_zenith
                model.solar_calc.sun_direction[None] = [sun_direction[0], sun_direction[1], sun_direction[2]]
                model.solar_calc.sun_up[None] = 1 if cos_zenith > 0 else 0
                
                model.compute_shortwave_radiation(
                    sw_direct=dni,
                    sw_diffuse=dhi
                )
                
                n_surfaces = model.surfaces.count
                surf_outgoing = model.surfaces.sw_out.to_numpy()[:n_surfaces]
                
                # Use non-cached volumetric reflection computation with distance cutoff
                calculator.compute_swflux_vol_with_reflections(
                    sw_direct=dni,
                    sw_diffuse=dhi,
                    cos_zenith=cos_zenith,
                    sun_direction=sun_direction,
                    surfaces=model.surfaces,
                    surf_outgoing=surf_outgoing,
                    lad=domain.lad
                )
            else:
                calculator.compute_swflux_vol(
                    sw_direct=dni,
                    sw_diffuse=dhi,
                    cos_zenith=cos_zenith,
                    sun_direction=sun_direction,
                    lad=domain.lad
                )
            
            # Accumulate using terrain-following extraction
            swflux_3d = calculator.get_swflux_vol()
            _accumulate_terrain_following_slice(
                cumulative_map, swflux_3d, ground_k, height_offset_k, is_solid,
                weight=time_step_hours
            )
            
            if progress_report and (i + 1) % 100 == 0:
                elapsed = time.perf_counter() - t0
                print(f"  Processed {i + 1}/{n_timesteps} timesteps ({elapsed:.1f}s)")
    
    if progress_report:
        elapsed = time.perf_counter() - t0
        print(f"Cumulative volumetric irradiance complete in {elapsed:.2f}s")
    
    # Apply terrain-following NaN mask for cells without valid ground
    for i in range(ni):
        for j in range(nj):
            gk = ground_k[i, j]
            if gk < 0:
                cumulative_map[i, j] = np.nan
                continue
            k_extract = gk + height_offset_k
            if k_extract >= nk:
                cumulative_map[i, j] = np.nan
                continue
            if is_solid[i, j, k_extract] == 1:
                cumulative_map[i, j] = np.nan
    
    # Flip to match VoxCity coordinate system
    cumulative_map = np.flipud(cumulative_map)
    
    if show_plot:
        colormap = kwargs.get('colormap', 'magma')
        vmin = kwargs.get('vmin', 0.0)
        vmax = kwargs.get('vmax', max(float(np.nanmax(cumulative_map)), 1.0))
        try:
            import matplotlib.pyplot as plt
            cmap = plt.cm.get_cmap(colormap).copy()
            cmap.set_bad(color='lightgray')
            plt.figure(figsize=(10, 8))
            plt.imshow(cumulative_map, origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)
            plt.colorbar(label=f'Cumulative Volumetric Irradiance at {volumetric_height}m (Wh/m²)')
            plt.title(f'Cumulative Volumetric Irradiance (reflections={"on" if with_reflections else "off"})')
            plt.axis('off')
            plt.show()
        except ImportError:
            pass
    
    return cumulative_map


def get_volumetric_solar_irradiance_using_epw(
    voxcity,
    calc_type: str = 'instantaneous',
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    volumetric_height: float = 1.5,
    with_reflections: bool = False,
    show_plot: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated volumetric solar irradiance from EPW file.
    
    Computes 3D radiation fields and extracts a 2D horizontal slice at the 
    specified height above ground. This is useful for:
    - Mean Radiant Temperature (MRT) calculations
    - Pedestrian thermal comfort analysis
    - Light availability assessment
    
    Args:
        voxcity: VoxCity object
        calc_type: 'instantaneous' or 'cumulative'
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        volumetric_height: Height above ground for irradiance extraction (meters)
        with_reflections: If True, include reflected radiation from buildings,
            ground, and tree canopy surfaces. If False (default), only direct
            + diffuse sky radiation.
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters including:
            - epw_file_path (str): Path to EPW file
            - download_nearest_epw (bool): Download nearest EPW (default: False)
            - calc_time (str): For instantaneous: 'MM-DD HH:MM:SS'
            - start_time, end_time (str): For cumulative: 'MM-DD HH:MM:SS'
            - rectangle_vertices: Location vertices (for EPW download)
            - n_reflection_steps (int): Reflection bounces when with_reflections=True (default: 2)
    
    Returns:
        2D numpy array of volumetric irradiance at the specified height (W/m² or Wh/m²)
    """
    from datetime import datetime
    import pytz
    
    # Load EPW data using helper function
    kwargs_copy = dict(kwargs)
    df, lon, lat, tz = _load_epw_data(
        epw_file_path=kwargs_copy.pop('epw_file_path', None),
        download_nearest_epw=kwargs_copy.pop('download_nearest_epw', False),
        voxcity=voxcity,
        **kwargs_copy
    )
    
    if calc_type == 'instantaneous':
        calc_time = kwargs.get('calc_time', '01-01 12:00:00')
        try:
            calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%SS')
        except ValueError:
            try:
                calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'")
        
        df_period = df[
            (df.index.month == calc_dt.month) &
            (df.index.day == calc_dt.day) &
            (df.index.hour == calc_dt.hour)
        ]
        if df_period.empty:
            raise ValueError("No EPW data at the specified time.")
        
        # Get solar position
        offset_minutes = int(tz * 60)
        local_tz = pytz.FixedOffset(offset_minutes)
        df_local = df_period.copy()
        df_local.index = df_local.index.tz_localize(local_tz)
        df_utc = df_local.tz_convert(pytz.UTC)
        
        solar_positions = _get_solar_positions_astral(df_utc.index, lon, lat)
        DNI = float(df_utc.iloc[0]['DNI']) * direct_normal_irradiance_scaling
        DHI = float(df_utc.iloc[0]['DHI']) * diffuse_irradiance_scaling
        azimuth_degrees = float(solar_positions.iloc[0]['azimuth'])
        elevation_degrees = float(solar_positions.iloc[0]['elevation'])
        
        return get_volumetric_solar_irradiance_map(
            voxcity,
            azimuth_degrees,
            elevation_degrees,
            DNI,
            DHI,
            volumetric_height=volumetric_height,
            with_reflections=with_reflections,
            show_plot=show_plot,
            **kwargs
        )
    
    elif calc_type == 'cumulative':
        return get_cumulative_volumetric_solar_irradiance(
            voxcity,
            df,
            lon,
            lat,
            tz,
            direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
            diffuse_irradiance_scaling=diffuse_irradiance_scaling,
            volumetric_height=volumetric_height,
            with_reflections=with_reflections,
            show_plot=show_plot,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown calc_type: {calc_type}. Use 'instantaneous' or 'cumulative'.")


def get_global_solar_irradiance_using_epw(
    voxcity,
    temporal_mode: str = 'instantaneous',
    spatial_mode: str = 'horizontal',
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    show_plot: bool = False,
    calc_type: str = None,  # Deprecated, for backward compatibility
    computation_mask: np.ndarray = None,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated global irradiance from EPW file.
    
    This function matches the signature of voxcity.simulator.solar.get_global_solar_irradiance_using_epw
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        temporal_mode: Time integration mode:
            - 'instantaneous': Single time point (requires calc_time)
            - 'cumulative': Integrate over time range (requires start_time, end_time)
        spatial_mode: Spatial computation mode:
            - 'horizontal': 2D ground-level irradiance at view_point_height
            - 'volumetric': 3D radiation field extracted at volumetric_height above terrain
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        show_plot: Whether to display a matplotlib plot
        calc_type: DEPRECATED. Use temporal_mode and spatial_mode instead.
            Legacy values 'instantaneous', 'cumulative', 'volumetric' are still supported.
        computation_mask: Optional 2D boolean numpy array of shape (nx, ny).
            If provided, only cells where mask is True will be computed.
            Cells where mask is False will be set to NaN in the output.
            Use create_computation_mask() to create masks easily.
        **kwargs: Additional parameters including:
            - epw_file_path (str): Path to EPW file
            - download_nearest_epw (bool): Download nearest EPW (default: False)
            - calc_time (str): For instantaneous: 'MM-DD HH:MM:SS'
            - start_time, end_time (str): For cumulative: 'MM-DD HH:MM:SS'
            - rectangle_vertices: Location vertices (for EPW download)
            - view_point_height (float): Height for horizontal mode (default: 1.5)
            - volumetric_height (float): Height for volumetric mode (default: 1.5)
            - with_reflections (bool): Include reflections (default: False)
            - n_reflection_steps (int): Reflection bounces (default: 2)
    
    Returns:
        2D numpy array of irradiance (W/m² for instantaneous, Wh/m² for cumulative)
    
    Examples:
        # Instantaneous ground-level irradiance
        grid = get_global_solar_irradiance_using_epw(
            voxcity,
            temporal_mode='instantaneous',
            spatial_mode='horizontal',
            calc_time='08-03 10:00:00',
            epw_file_path='weather.epw'
        )
        
        # Cumulative volumetric irradiance with reflections
        grid = get_global_solar_irradiance_using_epw(
            voxcity,
            temporal_mode='cumulative',
            spatial_mode='volumetric',
            start_time='01-01 09:00:00',
            end_time='01-31 19:00:00',
            volumetric_height=1.5,
            with_reflections=True,
            epw_file_path='weather.epw'
        )
    """
    from datetime import datetime
    import pytz
    import warnings
    
    # Handle backward compatibility with calc_type parameter
    if calc_type is not None:
        warnings.warn(
            "calc_type parameter is deprecated. Use temporal_mode and spatial_mode instead. "
            "Example: temporal_mode='cumulative', spatial_mode='volumetric'",
            DeprecationWarning,
            stacklevel=2
        )
        if calc_type == 'instantaneous':
            temporal_mode = 'instantaneous'
            spatial_mode = 'horizontal'
        elif calc_type == 'cumulative':
            temporal_mode = 'cumulative'
            spatial_mode = 'horizontal'
        elif calc_type == 'volumetric':
            # Legacy volumetric: determine temporal mode from time parameters
            spatial_mode = 'volumetric'
            calc_time = kwargs.get('calc_time', None)
            start_time = kwargs.get('start_time', None)
            if calc_time is not None and start_time is None:
                temporal_mode = 'instantaneous'
            else:
                temporal_mode = 'cumulative'
        else:
            raise ValueError(f"Unknown calc_type: {calc_type}. Use temporal_mode/spatial_mode instead.")
    
    # Validate parameters
    if temporal_mode not in ('instantaneous', 'cumulative'):
        raise ValueError(f"temporal_mode must be 'instantaneous' or 'cumulative', got '{temporal_mode}'")
    if spatial_mode not in ('horizontal', 'volumetric'):
        raise ValueError(f"spatial_mode must be 'horizontal' or 'volumetric', got '{spatial_mode}'")
    
    # Load EPW data using helper function
    kwargs_copy = dict(kwargs)
    df, lon, lat, tz = _load_epw_data(
        epw_file_path=kwargs_copy.pop('epw_file_path', None),
        download_nearest_epw=kwargs_copy.pop('download_nearest_epw', False),
        voxcity=voxcity,
        **kwargs_copy
    )
    
    # Add computation_mask to kwargs for passing to underlying functions
    if computation_mask is not None:
        kwargs['computation_mask'] = computation_mask
    
    # Route to appropriate function based on temporal_mode × spatial_mode
    if spatial_mode == 'horizontal':
        # Ground-level horizontal irradiance
        if temporal_mode == 'instantaneous':
            calc_time = kwargs.get('calc_time', '01-01 12:00:00')
            try:
                calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'")
            
            df_period = df[
                (df.index.month == calc_dt.month) &
                (df.index.day == calc_dt.day) &
                (df.index.hour == calc_dt.hour)
            ]
            if df_period.empty:
                raise ValueError("No EPW data at the specified time.")
            
            offset_minutes = int(tz * 60)
            local_tz = pytz.FixedOffset(offset_minutes)
            df_local = df_period.copy()
            df_local.index = df_local.index.tz_localize(local_tz)
            df_utc = df_local.tz_convert(pytz.UTC)
            
            solar_positions = _get_solar_positions_astral(df_utc.index, lon, lat)
            DNI = float(df_utc.iloc[0]['DNI']) * direct_normal_irradiance_scaling
            DHI = float(df_utc.iloc[0]['DHI']) * diffuse_irradiance_scaling
            azimuth_degrees = float(solar_positions.iloc[0]['azimuth'])
            elevation_degrees = float(solar_positions.iloc[0]['elevation'])
            
            return get_global_solar_irradiance_map(
                voxcity,
                azimuth_degrees,
                elevation_degrees,
                DNI,
                DHI,
                show_plot=show_plot,
                **kwargs
            )
        
        else:  # cumulative
            return get_cumulative_global_solar_irradiance(
                voxcity,
                df,
                lon,
                lat,
                tz,
                direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
                diffuse_irradiance_scaling=diffuse_irradiance_scaling,
                show_plot=show_plot,
                **kwargs
            )
    
    else:  # volumetric
        # 3D volumetric radiation field
        volumetric_height = kwargs.pop('volumetric_height', kwargs.pop('view_point_height', 1.5))
        with_reflections = kwargs.pop('with_reflections', False)
        
        if temporal_mode == 'instantaneous':
            calc_time = kwargs.get('calc_time', '01-01 12:00:00')
            try:
                calc_dt = datetime.strptime(calc_time, '%m-%d %H:%M:%S')
            except ValueError:
                raise ValueError("calc_time must be in format 'MM-DD HH:MM:SS'")
            
            df_period = df[
                (df.index.month == calc_dt.month) &
                (df.index.day == calc_dt.day) &
                (df.index.hour == calc_dt.hour)
            ]
            if df_period.empty:
                raise ValueError("No EPW data at the specified time.")
            
            offset_minutes = int(tz * 60)
            local_tz = pytz.FixedOffset(offset_minutes)
            df_local = df_period.copy()
            df_local.index = df_local.index.tz_localize(local_tz)
            df_utc = df_local.tz_convert(pytz.UTC)
            
            solar_positions = _get_solar_positions_astral(df_utc.index, lon, lat)
            DNI = float(df_utc.iloc[0]['DNI']) * direct_normal_irradiance_scaling
            DHI = float(df_utc.iloc[0]['DHI']) * diffuse_irradiance_scaling
            azimuth_degrees = float(solar_positions.iloc[0]['azimuth'])
            elevation_degrees = float(solar_positions.iloc[0]['elevation'])
            
            return get_volumetric_solar_irradiance_map(
                voxcity,
                azimuth_degrees,
                elevation_degrees,
                DNI,
                DHI,
                volumetric_height=volumetric_height,
                with_reflections=with_reflections,
                show_plot=show_plot,
                **kwargs
            )
        
        else:  # cumulative
            return get_cumulative_volumetric_solar_irradiance(
                voxcity,
                df,
                lon,
                lat,
                tz,
                direct_normal_irradiance_scaling=direct_normal_irradiance_scaling,
                diffuse_irradiance_scaling=diffuse_irradiance_scaling,
                volumetric_height=volumetric_height,
                with_reflections=with_reflections,
                show_plot=show_plot,
                **kwargs
            )


def save_irradiance_mesh(mesh, filepath: str) -> None:
    """
    Save irradiance mesh to pickle file.
    
    Args:
        mesh: Trimesh object with irradiance metadata
        filepath: Output file path
    """
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(mesh, f)


def load_irradiance_mesh(filepath: str):
    """
    Load irradiance mesh from pickle file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Trimesh object with irradiance metadata
    """
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Internal Helper Functions
# =============================================================================

# Module-level cache for GPU ray tracer fields
@dataclass
class _CachedGPURayTracer:
    """Cached Taichi fields for GPU ray tracing."""
    is_solid_field: object  # ti.field
    lad_field: object  # ti.field
    transmittance_field: object  # ti.field
    topo_top_field: object  # ti.field
    trace_rays_kernel: object  # compiled kernel
    voxel_shape: Tuple[int, int, int]
    meshsize: float
    voxel_data_id: int = 0  # id() of last voxel_data array to detect changes


_gpu_ray_tracer_cache: Optional[_CachedGPURayTracer] = None

# Module-level cached kernel for topo computation
_cached_topo_kernel = None


def _get_cached_topo_kernel():
    """Get or create cached topography kernel."""
    global _cached_topo_kernel
    if _cached_topo_kernel is not None:
        return _cached_topo_kernel
    
    import taichi as ti
    from ..init_taichi import ensure_initialized
    ensure_initialized()
    
    @ti.kernel
    def _topo_kernel(
        is_solid_f: ti.template(),
        topo_f: ti.template(),
        grid_nz: ti.i32
    ):
        for i, j in topo_f:
            max_k = -1
            for k in range(grid_nz):
                if is_solid_f[i, j, k] == 1:
                    max_k = k
            topo_f[i, j] = max_k
    
    _cached_topo_kernel = _topo_kernel
    return _cached_topo_kernel


def _compute_topo_gpu(is_solid_field, topo_top_field, nz: int):
    """Compute topography (highest solid voxel) using GPU."""
    kernel = _get_cached_topo_kernel()
    kernel(is_solid_field, topo_top_field, nz)


# Module-level cached kernel for ray tracing
_cached_trace_rays_kernel = None


def _get_cached_trace_rays_kernel():
    """Get or create cached ray tracing kernel."""
    global _cached_trace_rays_kernel
    if _cached_trace_rays_kernel is not None:
        return _cached_trace_rays_kernel
    
    import taichi as ti
    from ..init_taichi import ensure_initialized
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


def _get_or_create_gpu_ray_tracer(
    voxel_data: np.ndarray,
    meshsize: float,
    tree_lad: float = 1.0
) -> _CachedGPURayTracer:
    """
    Get cached GPU ray tracer or create new one if cache is invalid.
    
    The Taichi fields and kernels are expensive to create, so we cache them.
    """
    global _gpu_ray_tracer_cache
    
    import taichi as ti
    from ..init_taichi import ensure_initialized
    ensure_initialized()
    
    nx, ny, nz = voxel_data.shape
    
    # Check if cache is valid
    if _gpu_ray_tracer_cache is not None:
        cache = _gpu_ray_tracer_cache
        if cache.voxel_shape == (nx, ny, nz) and cache.meshsize == meshsize:
            # Check if voxel data has changed (same array object = same data)
            if cache.voxel_data_id == id(voxel_data):
                # Data hasn't changed, reuse cached fields directly
                return cache
            
            # Data changed, need to re-upload (but keep fields)
            # Use vectorized helper
            is_solid, lad_array = _convert_voxel_data_to_arrays(voxel_data, tree_lad)
            
            cache.is_solid_field.from_numpy(is_solid)
            cache.lad_field.from_numpy(lad_array)
            cache.voxel_data_id = id(voxel_data)
            
            # Recompute topo
            _compute_topo_gpu(cache.is_solid_field, cache.topo_top_field, nz)
            return cache
    
    # Need to create new cache - use vectorized helper
    is_solid, lad_array = _convert_voxel_data_to_arrays(voxel_data, tree_lad)
    
    # Create Taichi fields
    is_solid_field = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    lad_field = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
    transmittance_field = ti.field(dtype=ti.f32, shape=(nx, ny))
    topo_top_field = ti.field(dtype=ti.i32, shape=(nx, ny))
    
    is_solid_field.from_numpy(is_solid)
    lad_field.from_numpy(lad_array)
    
    # Compute topography using cached kernel
    _compute_topo_gpu(is_solid_field, topo_top_field, nz)
    
    # Get cached ray tracing kernel
    trace_rays_kernel = _get_cached_trace_rays_kernel()
    
    # Cache it
    _gpu_ray_tracer_cache = _CachedGPURayTracer(
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


def _compute_direct_transmittance_map_gpu(
    voxel_data: np.ndarray,
    sun_direction: Tuple[float, float, float],
    view_point_height: float,
    meshsize: float,
    tree_k: float = 0.6,
    tree_lad: float = 1.0
) -> np.ndarray:
    """
    Compute direct solar transmittance map using GPU ray tracing.
    
    Returns a 2D array where each cell contains the transmittance (0-1)
    for direct sunlight from the given direction.
    
    Uses cached Taichi fields to avoid expensive re-creation.
    """
    nx, ny, nz = voxel_data.shape
    
    # Get or create cached ray tracer
    cache = _get_or_create_gpu_ray_tracer(voxel_data, meshsize, tree_lad)
    
    # Run ray tracing with current sun direction
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
        nx, ny, nz  # Grid dimensions as parameters
    )
    
    return cache.transmittance_field.to_numpy()


def _get_solar_positions_astral(times, lon: float, lat: float):
    """
    Compute solar azimuth and elevation using Astral library.
    """
    import pandas as pd
    try:
        from astral import Observer
        from astral.sun import elevation, azimuth
        
        observer = Observer(latitude=lat, longitude=lon)
        df_pos = pd.DataFrame(index=times, columns=['azimuth', 'elevation'], dtype=float)
        for t in times:
            el = elevation(observer=observer, dateandtime=t)
            az = azimuth(observer=observer, dateandtime=t)
            df_pos.at[t, 'elevation'] = el
            df_pos.at[t, 'azimuth'] = az
        return df_pos
    except ImportError:
        raise ImportError("Astral library required for solar position calculation. Install with: pip install astral")


# Public alias for VoxCity API compatibility
def get_solar_positions_astral(times, lon: float, lat: float):
    """
    Compute solar azimuth and elevation for given times and location using Astral.
    
    This function matches the signature of voxcity.simulator.solar.get_solar_positions_astral.
    
    Args:
        times: Pandas DatetimeIndex of times (should be timezone-aware, preferably UTC)
        lon: Longitude in degrees
        lat: Latitude in degrees
    
    Returns:
        DataFrame indexed by times with columns ['azimuth', 'elevation'] in degrees
    """
    return _get_solar_positions_astral(times, lon, lat)


def _export_irradiance_to_obj(voxcity, irradiance_map: np.ndarray, output_name: str = 'irradiance', **kwargs):
    """Export irradiance map to OBJ file using VoxCity utilities."""
    try:
        from voxcity.exporter.obj import grid_to_obj
        meshsize = voxcity.voxels.meta.meshsize
        dem_grid = voxcity.dem.elevation if hasattr(voxcity, 'dem') and voxcity.dem else np.zeros_like(irradiance_map)
        output_dir = kwargs.get('output_directory', 'output')
        view_point_height = kwargs.get('view_point_height', 1.5)
        colormap = kwargs.get('colormap', 'magma')
        vmin = kwargs.get('vmin', 0.0)
        vmax = kwargs.get('vmax', float(np.nanmax(irradiance_map)) if not np.all(np.isnan(irradiance_map)) else 1.0)
        num_colors = kwargs.get('num_colors', 10)
        alpha = kwargs.get('alpha', 1.0)
        
        grid_to_obj(
            irradiance_map,
            dem_grid,
            output_dir,
            output_name,
            meshsize,
            view_point_height,
            colormap_name=colormap,
            num_colors=num_colors,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax
        )
    except ImportError:
        print("VoxCity exporter.obj required for OBJ export")
