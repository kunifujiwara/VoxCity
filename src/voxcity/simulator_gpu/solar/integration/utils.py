"""
Common utility functions for VoxCity solar integration module.

This module contains shared helper functions used across ground, building,
and volumetric solar irradiance calculations to reduce code duplication.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


# VoxCity voxel class codes (from voxcity/generator/voxelizer.py)
VOXCITY_GROUND_CODE = -1
VOXCITY_TREE_CODE = -2
VOXCITY_BUILDING_CODE = -3


# =============================================================================
# Location Helpers
# =============================================================================

def get_location_from_voxcity(
    voxcity,
    default_lat: float = 1.35,
    default_lon: float = 103.82
) -> Tuple[float, float]:
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


# =============================================================================
# Voxel Data Conversion
# =============================================================================

def convert_voxel_data_to_arrays(
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


def compute_valid_ground_vectorized(voxel_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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
    
    valid_ground = np.zeros((ni, nj), dtype=bool)
    ground_k = np.full((ni, nj), -1, dtype=np.int32)
    
    # Vectorize over k: find first transition from solid to air/tree
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
        ground_k[invalid_new] = -2
    
    # Reset -2 markers back to -1 (no valid ground)
    ground_k[ground_k == -2] = -1
    
    return valid_ground, ground_k


def compute_ground_k_from_voxels(voxel_data: np.ndarray) -> np.ndarray:
    """
    Compute ground surface k-level for each (i,j) cell from voxel data.
    
    This finds the terrain top - the highest k where the cell below the first air
    cell is solid ground (not building). This is used for terrain-following
    height extraction in volumetric calculations.
    
    Args:
        voxel_data: 3D array of voxel class codes
        
    Returns:
        2D array of ground k-levels (ni, nj). -1 means no valid ground found.
    """
    _, ground_k = compute_valid_ground_vectorized(voxel_data)
    return ground_k


# =============================================================================
# Sun Direction Computation
# =============================================================================

def compute_sun_direction(
    azimuth_degrees_ori: float,
    elevation_degrees: float
) -> Tuple[float, float, float, float]:
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
# Time Period Filtering
# =============================================================================

def parse_time_period(
    start_time: str,
    end_time: str
) -> Tuple[datetime, datetime]:
    """
    Parse start and end time strings.
    
    Args:
        start_time: Start time in format 'MM-DD HH:MM:SS'
        end_time: End time in format 'MM-DD HH:MM:SS'
        
    Returns:
        Tuple of (start_dt, end_dt) datetime objects
        
    Raises:
        ValueError: If time format is invalid
    """
    try:
        start_dt = datetime.strptime(start_time, '%m-%d %H:%M:%S')
        end_dt = datetime.strptime(end_time, '%m-%d %H:%M:%S')
    except ValueError as ve:
        raise ValueError("start_time and end_time must be in format 'MM-DD HH:MM:SS'") from ve
    
    return start_dt, end_dt


def filter_df_to_period(df, start_time: str, end_time: str, tz: float):
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
    import pytz
    
    start_dt, end_dt = parse_time_period(start_time, end_time)
    
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


def get_hour_range_from_period(start_time: str, end_time: str) -> Tuple[int, int]:
    """
    Get hour-of-year range from time period strings.
    
    Args:
        start_time: Start time in format 'MM-DD HH:MM:SS'
        end_time: End time in format 'MM-DD HH:MM:SS'
        
    Returns:
        Tuple of (start_hour, end_hour) as hour-of-year values
    """
    start_dt, end_dt = parse_time_period(start_time, end_time)
    
    start_doy = datetime(2000, start_dt.month, start_dt.day).timetuple().tm_yday
    end_doy = datetime(2000, end_dt.month, end_dt.day).timetuple().tm_yday
    start_hour = (start_doy - 1) * 24 + start_dt.hour + 1
    end_hour = (end_doy - 1) * 24 + end_dt.hour + 1
    
    return start_hour, end_hour


# =============================================================================
# EPW Data Loading
# =============================================================================

def load_epw_data(
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
        from ..epw import read_epw_header, read_epw_solar_data
        location = read_epw_header(epw_file_path)
        df = read_epw_solar_data(epw_file_path)
        lon, lat, tz = location.longitude, location.latitude, location.timezone
    
    if df.empty:
        raise ValueError("No data in EPW file.")
    
    return df, lon, lat, tz


# =============================================================================
# Solar Position Calculation
# =============================================================================

def get_solar_positions_astral(times, lon: float, lat: float):
    """
    Compute solar azimuth and elevation for given times and location using Astral.
    
    Args:
        times: Pandas DatetimeIndex of times (should be timezone-aware, preferably UTC)
        lon: Longitude in degrees
        lat: Latitude in degrees
    
    Returns:
        DataFrame indexed by times with columns ['azimuth', 'elevation'] in degrees
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


# =============================================================================
# Terrain-Following Extraction
# =============================================================================

def extract_terrain_following_slice(
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


def accumulate_terrain_following_slice(
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


# =============================================================================
# Metadata Array Helper
# =============================================================================

class ArrayWithMetadata(np.ndarray):
    """NumPy array subclass that can hold metadata."""
    
    def __new__(cls, input_array, metadata=None):
        obj = np.asarray(input_array).view(cls)
        obj.metadata = metadata if metadata is not None else {}
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.metadata = getattr(obj, 'metadata', {})


def add_metadata_to_array(arr: np.ndarray, metadata: dict) -> np.ndarray:
    """
    Add metadata dict to a numpy array as an attribute.
    
    Args:
        arr: Input numpy array
        metadata: Dictionary of metadata to attach
        
    Returns:
        Array with metadata attribute
    """
    return ArrayWithMetadata(arr, metadata)


# =============================================================================
# Boundary/Mask Utilities
# =============================================================================

def compute_boundary_vertical_mask(
    mesh_face_centers: np.ndarray,
    mesh_face_normals: np.ndarray,
    grid_bounds: np.ndarray,
    boundary_epsilon: float
) -> np.ndarray:
    """
    Compute mask for vertical faces on domain boundary.
    
    Args:
        mesh_face_centers: (N, 3) array of face center coordinates
        mesh_face_normals: (N, 3) array of face normal vectors
        grid_bounds: (2, 3) array of [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        boundary_epsilon: Tolerance for boundary detection
        
    Returns:
        Boolean mask (N,) - True for vertical boundary faces
    """
    # Detect vertical faces (normal z-component near zero)
    is_vertical = np.abs(mesh_face_normals[:, 2]) < 0.01
    
    # Detect faces on domain boundary
    on_x_min = np.abs(mesh_face_centers[:, 0] - grid_bounds[0, 0]) < boundary_epsilon
    on_y_min = np.abs(mesh_face_centers[:, 1] - grid_bounds[0, 1]) < boundary_epsilon
    on_x_max = np.abs(mesh_face_centers[:, 0] - grid_bounds[1, 0]) < boundary_epsilon
    on_y_max = np.abs(mesh_face_centers[:, 1] - grid_bounds[1, 1]) < boundary_epsilon
    
    return is_vertical & (on_x_min | on_y_min | on_x_max | on_y_max)


def apply_computation_mask_to_faces(
    values: np.ndarray,
    mesh_face_centers: np.ndarray,
    computation_mask: np.ndarray,
    meshsize: float,
    grid_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Apply 2D computation mask to mesh face values.
    
    Args:
        values: (N,) array of face values
        mesh_face_centers: (N, 3) array of face center coordinates
        computation_mask: 2D boolean mask (ny, nx)
        meshsize: Grid cell size
        grid_shape: (ny_vc, nx_vc) grid dimensions
        
    Returns:
        Modified values array with NaN for masked-out faces
    """
    ny_vc, nx_vc = grid_shape
    
    # Convert face XY positions to grid indices
    face_x = mesh_face_centers[:, 0]
    face_y = mesh_face_centers[:, 1]
    
    grid_i = (face_y / meshsize).astype(int)
    grid_j = (face_x / meshsize).astype(int)
    
    # Handle mask shape orientation
    if computation_mask.shape == (ny_vc, nx_vc):
        mask_shape = computation_mask.shape
    elif computation_mask.T.shape == (ny_vc, nx_vc):
        computation_mask = computation_mask.T
        mask_shape = computation_mask.shape
    else:
        mask_shape = computation_mask.shape
    
    # Clamp indices to valid range
    grid_i = np.clip(grid_i, 0, mask_shape[0] - 1)
    grid_j = np.clip(grid_j, 0, mask_shape[1] - 1)
    
    # Flip mask to match coordinate system
    flipped_mask = np.flipud(computation_mask)
    outside_mask = ~flipped_mask[grid_i, grid_j]
    
    # Set values outside mask to NaN
    result = values.copy()
    result[outside_mask] = np.nan
    
    return result
