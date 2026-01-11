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
    
    Uses the RadiationModel to compute direct and diffuse (with reflections) components
    at ground level (upward-facing horizontal surfaces).
    
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
    from .radiation import RadiationModel, RadiationConfig
    from .domain import IUP
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    # VoxCity uses [i, j, k] indexing - keep this convention for consistency
    # with simple raytracing (no coordinate swap)
    ni, nj, nk = voxel_data.shape
    
    # Get location
    rectangle_vertices = getattr(voxcity, 'extras', {}).get('rectangle_vertices', None)
    if rectangle_vertices is not None:
        lons = [v[0] for v in rectangle_vertices]
        lats = [v[1] for v in rectangle_vertices]
        origin_lat = np.mean(lats)
        origin_lon = np.mean(lons)
    else:
        origin_lat = 1.35
        origin_lon = 103.82
    
    # Create domain with same shape as VoxCity data (no coordinate swap)
    # This matches the simple raytracing approach
    domain = Domain(
        nx=ni, ny=nj, nz=nk,
        dx=meshsize, dy=meshsize, dz=meshsize,
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    # Convert VoxCity voxel data to domain arrays
    # Keep the same indexing as VoxCity (no coordinate swap)
    is_solid_np = np.zeros((ni, nj, nk), dtype=np.int32)
    lad_np = np.zeros((ni, nj, nk), dtype=np.float32)
    # Use same default LAD as simple raytracing for consistency
    default_lad = kwargs.get('default_lad', 1.0)  # Match simple raytracing default
    tree_k = kwargs.get('tree_k', 0.6)  # Extinction coefficient
    
    # Track valid ground cells following VoxCity logic:
    # Valid = observer in air (0 or -2), and voxel below is positive land cover code
    #         (not -1, not water codes 7/8/9, not negative)
    # This matches the logic in VoxCity's compute_direct_solar_irradiance_map_binary
    valid_ground = np.zeros((ni, nj), dtype=bool)
    
    for i in range(ni):
        for j in range(nj):
            for k in range(nk):
                voxel_val = voxel_data[i, j, k]
                
                if voxel_val == VOXCITY_BUILDING_CODE:  # -3
                    is_solid_np[i, j, k] = 1
                elif voxel_val == VOXCITY_GROUND_CODE:  # -1
                    is_solid_np[i, j, k] = 1
                elif voxel_val == VOXCITY_TREE_CODE:  # -2
                    lad_np[i, j, k] = default_lad
                elif voxel_val > 0:
                    # Positive values are land cover codes on ground
                    is_solid_np[i, j, k] = 1
            
            # Determine if this column has a valid ground cell
            # Following VoxCity logic: find first air (0 or -2) above non-air
            for k in range(1, nk):
                curr_val = voxel_data[i, j, k]
                below_val = voxel_data[i, j, k - 1]
                # Observer in air (0) or tree (-2), below is not air/tree
                if curr_val in (0, VOXCITY_TREE_CODE) and below_val not in (0, VOXCITY_TREE_CODE):
                    # Check if below is valid: NOT water (7,8,9) and NOT negative
                    if below_val in (7, 8, 9) or below_val < 0:
                        valid_ground[i, j] = False
                    else:
                        valid_ground[i, j] = True
                    break
    
    # Set domain arrays using helper functions
    _set_solid_array(domain, is_solid_np)
    domain.set_lad_from_array(lad_np)
    _update_topo_from_solid(domain)
    
    config = RadiationConfig(
        n_reflection_steps=n_reflection_steps,
        n_azimuth=kwargs.get('n_azimuth', 40),
        n_elevation=kwargs.get('n_elevation', 10)
    )
    
    model = RadiationModel(domain, config)
    
    # Set solar position
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
    
    # Compute SVF and radiation
    if progress_report:
        print("Computing Sky View Factors...")
    model.compute_svf()
    
    if progress_report:
        print("Computing shortwave radiation with reflections...")
    model.compute_shortwave_radiation(
        sw_direct=direct_normal_irradiance,
        sw_diffuse=diffuse_irradiance
    )
    
    # Extract surface irradiance
    # Note: sw_in_diffuse includes sky diffuse + surface reflections + canopy scattering
    n_surfaces = model.surfaces.count
    positions = model.surfaces.position.to_numpy()[:n_surfaces]
    directions = model.surfaces.direction.to_numpy()[:n_surfaces]
    sw_in_direct = model.surfaces.sw_in_direct.to_numpy()[:n_surfaces]
    sw_in_diffuse = model.surfaces.sw_in_diffuse.to_numpy()[:n_surfaces]  # Includes reflections
    
    # Map to ground-level 2D arrays
    # Output shape matches VoxCity input shape: (ni, nj)
    direct_map = np.full((ni, nj), np.nan, dtype=np.float32)
    diffuse_map = np.full((ni, nj), np.nan, dtype=np.float32)
    # Reflected is already included in diffuse, but we return zeros for API compatibility
    reflected_map = np.zeros((ni, nj), dtype=np.float32)
    
    # Find ground-level upward-facing surfaces
    # positions are [x, y, z] which maps to [i, j, k] since we didn't swap coordinates
    # Only consider valid ground cells (matching VoxCity logic)
    ground_k = np.full((ni, nj), -1, dtype=np.int32)
    for idx in range(n_surfaces):
        pos_i, pos_j, k = positions[idx]
        direction = directions[idx]
        if direction == IUP:
            ii, jj = int(pos_i), int(pos_j)
            if 0 <= ii < ni and 0 <= jj < nj:
                # Only consider cells that are valid ground according to VoxCity logic
                if not valid_ground[ii, jj]:
                    continue
                if ground_k[ii, jj] < 0 or k < ground_k[ii, jj]:
                    ground_k[ii, jj] = int(k)
    
    # Now map surfaces at ground level
    for idx in range(n_surfaces):
        pos_i, pos_j, k = positions[idx]
        direction = directions[idx]
        
        if direction == IUP:
            ii, jj = int(pos_i), int(pos_j)
            if 0 <= ii < ni and 0 <= jj < nj:
                # Only use valid ground cells
                if not valid_ground[ii, jj]:
                    continue
                # Only use the lowest up-facing surface at each location
                if int(k) == ground_k[ii, jj]:
                    if np.isnan(direct_map[ii, jj]):
                        direct_map[ii, jj] = sw_in_direct[idx]
                        diffuse_map[ii, jj] = sw_in_diffuse[idx]
    
    if progress_report:
        print(f"  Valid ground cells (VoxCity logic): {np.sum(valid_ground)}")
        print(f"  Ground cells found in RadiationModel: {np.sum(~np.isnan(direct_map))}")
    
    # Flip to match VoxCity coordinate system (same as simple raytracing)
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
        from ..visibility.voxcity import get_sky_view_factor_map as get_svf_map
        
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
            - n_reflection_steps (int): Number of reflection bounces when 
                with_reflections=True (default: 2)
            - progress_report (bool): Print progress (default: False)
    
    Returns:
        2D numpy array of global horizontal irradiance (W/m²)
    """
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
    from datetime import datetime
    import pytz
    
    view_point_height = kwargs.get('view_point_height', 1.5)
    colormap = kwargs.get('colormap', 'magma')
    start_time = kwargs.get('start_time', '01-01 05:00:00')
    end_time = kwargs.get('end_time', '01-01 20:00:00')
    progress_report = kwargs.get('progress_report', False)
    use_sky_patches = kwargs.get('use_sky_patches', True)
    sky_discretization = kwargs.get('sky_discretization', 'tregenza')
    
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
        cumulative_dni = np.zeros(n_patches, dtype=np.float64)
        total_cumulative_dhi = 0.0
        n_timesteps = len(azimuth_arr)
        
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
            
            # Find nearest patch
            patch_idx = int(get_tregenza_patch_index(float(az), float(elev)))
            if patch_idx >= 0 and patch_idx < n_patches:
                cumulative_dni[patch_idx] += dni * time_step_hours
        
        active_mask = cumulative_dni > 0
        n_active = int(np.sum(active_mask))
        
        if progress_report:
            print(f"Sky patch optimization: {n_timesteps} timesteps → {n_active} active patches ({sky_discretization})")
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
                pct = (i + 1) * 100.0 / len(active_indices)
                print(f"  Patch {i+1}/{len(active_indices)} ({pct:.1f}%)")
    
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
                    with_reflections=False,
                    **direct_kwargs
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
            - n_reflection_steps (int): Number of reflection bounces (default: 2)
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
    from .radiation import RadiationModel, RadiationConfig
    from .solar import SolarCalculator
    
    voxel_data = voxcity.voxels.classes
    meshsize = voxcity.voxels.meta.meshsize
    building_id_grid = voxcity.buildings.ids
    # VoxCity uses [row, col, z] = [i, j, k] where shape is (ny, nx, nz)
    ny_vc, nx_vc, nz = voxel_data.shape
    
    progress_report = kwargs.get('progress_report', False)
    building_class_id = kwargs.get('building_class_id', -3)
    n_reflection_steps = kwargs.get('n_reflection_steps', 2)
    colormap = kwargs.get('colormap', 'magma')
    
    # Get location
    rectangle_vertices = getattr(voxcity, 'extras', {}).get('rectangle_vertices', None)
    if rectangle_vertices is not None:
        lons = [v[0] for v in rectangle_vertices]
        lats = [v[1] for v in rectangle_vertices]
        origin_lat = np.mean(lats)
        origin_lon = np.mean(lons)
    else:
        origin_lat = 1.35
        origin_lon = 103.82
    
    # Create domain - palm_solar uses [x, y, z] convention
    domain = Domain(
        nx=nx_vc, ny=ny_vc, nz=nz,
        dx=meshsize, dy=meshsize, dz=meshsize,
        origin_lat=origin_lat,
        origin_lon=origin_lon
    )
    
    # Convert VoxCity voxel data to domain arrays
    # VoxCity [row, col, z] -> palm_solar [x, y, z] = [col, row, z]
    is_solid_np = np.zeros((nx_vc, ny_vc, nz), dtype=np.int32)
    lad_np = np.zeros((nx_vc, ny_vc, nz), dtype=np.float32)
    default_lad = kwargs.get('default_lad', 2.0)
    
    for row in range(ny_vc):
        for col in range(nx_vc):
            x_idx = col
            y_idx = row
            for z in range(nz):
                voxel_val = voxel_data[row, col, z]
                
                if voxel_val == VOXCITY_BUILDING_CODE:
                    is_solid_np[x_idx, y_idx, z] = 1
                elif voxel_val == VOXCITY_GROUND_CODE:
                    is_solid_np[x_idx, y_idx, z] = 1
                elif voxel_val == VOXCITY_TREE_CODE:
                    lad_np[x_idx, y_idx, z] = default_lad
                elif voxel_val > 0:
                    # Positive values are land cover codes on ground
                    is_solid_np[x_idx, y_idx, z] = 1
    
    # Set domain arrays using helper functions
    _set_solid_array(domain, is_solid_np)
    domain.set_lad_from_array(lad_np)
    _update_topo_from_solid(domain)
    
    config = RadiationConfig(
        n_reflection_steps=n_reflection_steps,
        n_azimuth=40,
        n_elevation=10
    )
    
    model = RadiationModel(domain, config)
    
    # Set solar position
    # Convert VoxCity azimuth convention
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
    
    # Compute SVF and radiation
    if progress_report:
        print("Computing Sky View Factors...")
    model.compute_svf()
    
    if progress_report:
        print("Computing shortwave radiation...")
    model.compute_shortwave_radiation(
        sw_direct=direct_normal_irradiance,
        sw_diffuse=diffuse_irradiance
    )
    
    # Extract surface irradiance from palm_solar model (all surfaces: buildings + ground + etc.)
    n_surfaces = model.surfaces.count
    sw_in_direct_all = model.surfaces.sw_in_direct.to_numpy()[:n_surfaces]
    sw_in_diffuse_all = model.surfaces.sw_in_diffuse.to_numpy()[:n_surfaces]

    # Some implementations store reflections inside sw_in_diffuse (and do not
    # expose a separate sw_in_reflected field).
    if hasattr(model.surfaces, 'sw_in_reflected'):
        sw_in_reflected_all = model.surfaces.sw_in_reflected.to_numpy()[:n_surfaces]
    else:
        sw_in_reflected_all = np.zeros_like(sw_in_direct_all)

    total_sw_all = sw_in_direct_all + sw_in_diffuse_all + sw_in_reflected_all

    # Get surface centers (world coords) and positions (grid indices)
    surf_centers_all = model.surfaces.center.to_numpy()[:n_surfaces]  # shape (n_surfaces, 3)
    surf_positions_all = model.surfaces.position.to_numpy()[:n_surfaces]  # shape (n_surfaces, 3) grid (i,j,k)

    # Filter to only building surfaces by checking voxel class at each surface position.
    # Remember: palm_solar uses [x, y, z] = [col, row, z]; VoxCity uses [row, col, z].
    is_building_surf = np.zeros(n_surfaces, dtype=bool)
    for s_idx in range(n_surfaces):
        x_idx, y_idx, z_idx = surf_positions_all[s_idx]  # palm_solar grid
        row, col = int(y_idx), int(x_idx)  # back to VoxCity indexing
        z = int(z_idx)
        if 0 <= row < ny_vc and 0 <= col < nx_vc and 0 <= z < nz:
            if voxel_data[row, col, z] == building_class_id:
                is_building_surf[s_idx] = True

    bldg_indices = np.where(is_building_surf)[0]
    if progress_report:
        print(f"  palm_solar surfaces: {n_surfaces}, building surfaces: {len(bldg_indices)}")

    # Use provided building_svf_mesh or create new mesh
    if building_svf_mesh is not None:
        building_mesh = building_svf_mesh.copy() if hasattr(building_svf_mesh, 'copy') else building_svf_mesh
        # Extract SVF from mesh metadata if available
        if hasattr(building_mesh, 'metadata') and 'svf' in building_mesh.metadata:
            face_svf = building_mesh.metadata['svf']
        else:
            face_svf = None
    else:
        # Create mesh for building surfaces
        try:
            from voxcity.geoprocessor.mesh import create_voxel_mesh
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
        except ImportError:
            print("VoxCity geoprocessor.mesh required for mesh creation")
            return None
        face_svf = None

    n_mesh_faces = len(building_mesh.faces)

    # Map palm_solar building surface values to building mesh faces via spatial nearest-neighbor.
    # Build KDTree from building surface centers.
    if len(bldg_indices) > 0:
        from scipy.spatial import cKDTree
        bldg_centers = surf_centers_all[bldg_indices]  # (n_bldg, 3) in world coords

        # Building mesh face centers (trimesh uses vertex positions directly)
        mesh_face_centers = building_mesh.triangles_center  # (n_mesh_faces, 3)

        # Debug: print coordinate ranges to detect mismatch
        if progress_report:
            print(f"  palm_solar bldg centers: x=[{bldg_centers[:,0].min():.1f}, {bldg_centers[:,0].max():.1f}], "
                  f"y=[{bldg_centers[:,1].min():.1f}, {bldg_centers[:,1].max():.1f}], "
                  f"z=[{bldg_centers[:,2].min():.1f}, {bldg_centers[:,2].max():.1f}]")
            print(f"  mesh face centers: x=[{mesh_face_centers[:,0].min():.1f}, {mesh_face_centers[:,0].max():.1f}], "
                  f"y=[{mesh_face_centers[:,1].min():.1f}, {mesh_face_centers[:,1].max():.1f}], "
                  f"z=[{mesh_face_centers[:,2].min():.1f}, {mesh_face_centers[:,2].max():.1f}]")

        # VoxCity mesh uses [row, col, z] = [i, j, k] in its coordinate system where:
        #   x_mesh = col * meshsize, y_mesh = row * meshsize, z_mesh = k * meshsize
        # palm_solar uses [x, y, z] = [col, row, z] (we transposed when building domain):
        #   x_palm = (col + 0.5) * dx, y_palm = (row + 0.5) * dy, z_palm = (k + 0.5) * dz
        # 
        # The mesh coordinates should align if both use meshsize and same origin.
        # However, VoxCity mesh may have different axis ordering. Let's check and swap if needed.
        #
        # From the debug output we can see if axes are swapped. For now, try swapping x<->y
        # in palm_solar centers to match VoxCity mesh convention.
        
        # Transform palm_solar centers: palm_solar (x=col, y=row, z) -> mesh (x=row, y=col, z)
        # i.e., swap x and y
        bldg_centers_transformed = bldg_centers.copy()
        bldg_centers_transformed[:, 0] = bldg_centers[:, 1]  # mesh x = palm_solar y
        bldg_centers_transformed[:, 1] = bldg_centers[:, 0]  # mesh y = palm_solar x

        if progress_report:
            print(f"  transformed bldg centers: x=[{bldg_centers_transformed[:,0].min():.1f}, {bldg_centers_transformed[:,0].max():.1f}], "
                  f"y=[{bldg_centers_transformed[:,1].min():.1f}, {bldg_centers_transformed[:,1].max():.1f}]")

        tree = cKDTree(bldg_centers_transformed)
        distances, nearest_idx = tree.query(mesh_face_centers, k=1)

        if progress_report:
            print(f"  KDTree match distances: min={distances.min():.2f}, mean={distances.mean():.2f}, max={distances.max():.2f}")

        # Map irradiance arrays
        sw_in_direct = sw_in_direct_all[bldg_indices][nearest_idx]
        sw_in_diffuse = sw_in_diffuse_all[bldg_indices][nearest_idx]
        sw_in_reflected = sw_in_reflected_all[bldg_indices][nearest_idx]
        total_sw = total_sw_all[bldg_indices][nearest_idx]
    else:
        # Fallback: no building surfaces in palm_solar model (edge case)
        sw_in_direct = np.zeros(n_mesh_faces, dtype=np.float32)
        sw_in_diffuse = np.zeros(n_mesh_faces, dtype=np.float32)
        sw_in_reflected = np.zeros(n_mesh_faces, dtype=np.float32)
        total_sw = np.zeros(n_mesh_faces, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Set vertical faces on domain perimeter to NaN (matching VoxCity behavior)
    # -------------------------------------------------------------------------
    # Compute domain bounds in world coordinates
    ny_vc, nx_vc, nz = voxel_data.shape
    grid_bounds_real = np.array([
        [0.0, 0.0, 0.0],
        [ny_vc * meshsize, nx_vc * meshsize, nz * meshsize]
    ], dtype=np.float64)
    boundary_epsilon = meshsize * 0.05

    mesh_face_centers = building_mesh.triangles_center
    mesh_face_normals = building_mesh.face_normals

    # Detect vertical faces (normal z-component near zero)
    is_vertical = np.abs(mesh_face_normals[:, 2]) < 0.01

    # Detect faces on domain boundary
    on_x_min = np.abs(mesh_face_centers[:, 0] - grid_bounds_real[0, 0]) < boundary_epsilon
    on_y_min = np.abs(mesh_face_centers[:, 1] - grid_bounds_real[0, 1]) < boundary_epsilon
    on_x_max = np.abs(mesh_face_centers[:, 0] - grid_bounds_real[1, 0]) < boundary_epsilon
    on_y_max = np.abs(mesh_face_centers[:, 1] - grid_bounds_real[1, 1]) < boundary_epsilon

    is_boundary_vertical = is_vertical & (on_x_min | on_y_min | on_x_max | on_y_max)

    # Set boundary vertical faces to NaN
    sw_in_direct = sw_in_direct.astype(np.float64)
    sw_in_diffuse = sw_in_diffuse.astype(np.float64)
    sw_in_reflected = sw_in_reflected.astype(np.float64)
    total_sw = total_sw.astype(np.float64)

    sw_in_direct[is_boundary_vertical] = np.nan
    sw_in_diffuse[is_boundary_vertical] = np.nan
    sw_in_reflected[is_boundary_vertical] = np.nan
    total_sw[is_boundary_vertical] = np.nan

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
            - fast_path (bool): Use optimized paths (default: True)
    
    Returns:
        Trimesh object with cumulative irradiance (Wh/m²) in metadata
    """
    from datetime import datetime
    import pytz
    
    period_start = kwargs.get('period_start', '01-01 00:00:00')
    period_end = kwargs.get('period_end', '12-31 23:59:59')
    time_step_hours = float(kwargs.get('time_step_hours', 1.0))
    progress_report = kwargs.get('progress_report', False)
    use_sky_patches = kwargs.get('use_sky_patches', True)
    
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
        print(f"Computing cumulative irradiance for {n_faces} faces over {len(df_period_utc)} timesteps...")
    
    # Process timesteps
    n_timesteps = len(df_period_utc)
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
    
    # Get EPW file
    epw_file_path = kwargs.get('epw_file_path', None)
    download_nearest_epw = kwargs.get('download_nearest_epw', False)
    
    rectangle_vertices = kwargs.get('rectangle_vertices', None)
    if rectangle_vertices is None:
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
    
    # Read EPW
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
    
    # Compute or get building SVF mesh
    if building_svf_mesh is None:
        if progress_report:
            print("Computing Sky View Factor for building surfaces...")
        # Import the visibility module to compute SVF
        try:
            from ..visibility import get_surface_view_factor
            building_svf_mesh = get_surface_view_factor(
                voxcity,
                mode='sky',
                progress_report=progress_report,
                **kwargs
            )
        except ImportError:
            # Fallback: compute without pre-computed SVF
            pass
    
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


def get_global_solar_irradiance_using_epw(
    voxcity,
    calc_type: str = 'instantaneous',
    direct_normal_irradiance_scaling: float = 1.0,
    diffuse_irradiance_scaling: float = 1.0,
    show_plot: bool = False,
    **kwargs
) -> np.ndarray:
    """
    GPU-accelerated global irradiance from EPW file.
    
    This function matches the signature of voxcity.simulator.solar.get_global_solar_irradiance_using_epw
    using Taichi GPU acceleration.
    
    Args:
        voxcity: VoxCity object
        calc_type: 'instantaneous' or 'cumulative'
        direct_normal_irradiance_scaling: Scaling factor for DNI
        diffuse_irradiance_scaling: Scaling factor for DHI
        show_plot: Whether to display a matplotlib plot
        **kwargs: Additional parameters including:
            - epw_file_path (str): Path to EPW file
            - download_nearest_epw (bool): Download nearest EPW (default: False)
            - calc_time (str): For instantaneous: 'MM-DD HH:MM:SS'
            - start_time, end_time (str): For cumulative: 'MM-DD HH:MM:SS'
            - rectangle_vertices: Location vertices (for EPW download)
    
    Returns:
        2D numpy array of irradiance (W/m² or Wh/m²)
    """
    from datetime import datetime
    import pytz
    
    # Get EPW file
    epw_file_path = kwargs.get('epw_file_path', None)
    download_nearest_epw = kwargs.get('download_nearest_epw', False)
    
    rectangle_vertices = kwargs.get('rectangle_vertices', None)
    if rectangle_vertices is None:
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
    
    # Read EPW
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
        
        return get_global_solar_irradiance_map(
            voxcity,
            azimuth_degrees,
            elevation_degrees,
            DNI,
            DHI,
            show_plot=show_plot,
            **kwargs
        )
    
    elif calc_type == 'cumulative':
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
    
    else:
        raise ValueError(f"Unknown calc_type: {calc_type}. Use 'instantaneous' or 'cumulative'.")


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
    """
    import taichi as ti
    from ..init_taichi import ensure_initialized
    
    # Ensure Taichi is initialized before creating fields
    ensure_initialized()
    
    nx, ny, nz = voxel_data.shape
    
    # Prepare voxel data for GPU
    is_solid = np.zeros((nx, ny, nz), dtype=np.int32)
    lad_array = np.zeros((nx, ny, nz), dtype=np.float32)
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                val = voxel_data[i, j, k]
                if val == VOXCITY_BUILDING_CODE or val == VOXCITY_GROUND_CODE or val > 0:
                    is_solid[i, j, k] = 1
                elif val == VOXCITY_TREE_CODE:
                    lad_array[i, j, k] = tree_lad
    
    # Create Taichi fields
    is_solid_field = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
    lad_field = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
    transmittance_field = ti.field(dtype=ti.f32, shape=(nx, ny))
    topo_top_field = ti.field(dtype=ti.i32, shape=(nx, ny))
    
    is_solid_field.from_numpy(is_solid)
    lad_field.from_numpy(lad_array)
    
    # Compute topography (highest solid voxel in each column)
    @ti.kernel
    def compute_topo():
        for i, j in topo_top_field:
            max_k = -1  # Start at -1 to handle columns with no solid
            for k in range(nz):
                if is_solid_field[i, j, k] == 1:
                    max_k = k
            topo_top_field[i, j] = max_k
    
    compute_topo()
    
    # Ray trace for each grid cell - trace towards sun to check for occlusions
    sun_dir_x = float(sun_direction[0])
    sun_dir_y = float(sun_direction[1])
    sun_dir_z = float(sun_direction[2])
    dx = dy = dz = meshsize
    view_height_k = max(1, int(view_point_height / meshsize))
    ext_coef = tree_k
    
    @ti.kernel
    def trace_rays(
        sun_x: ti.f32, sun_y: ti.f32, sun_z: ti.f32,
        vhk: ti.i32, ext: ti.f32, 
        step: ti.f32, max_dist: ti.f32
    ):
        for i, j in transmittance_field:
            # Start position: just ABOVE the ground (in the air above solid)
            ground_k = topo_top_field[i, j]
            
            # Start above the solid ground
            start_k = ground_k + vhk
            if start_k < 0:
                start_k = 0
            if start_k >= nz:
                start_k = nz - 1
            
            # Make sure we're starting in air (not solid)
            while start_k < nz - 1 and is_solid_field[i, j, start_k] == 1:
                start_k += 1
            
            # If still in solid, no sunlight reaches here
            if is_solid_field[i, j, start_k] == 1:
                transmittance_field[i, j] = 0.0
            else:
                # Convert to world coordinates (center of voxel)
                ox = (float(i) + 0.5) * dx
                oy = (float(j) + 0.5) * dy
                oz = (float(start_k) + 0.5) * dz
                
                # Ray direction (towards sun) - this is the direction we trace
                rx = sun_x
                ry = sun_y
                rz = sun_z
                
                # DDA ray marching - trace towards sun
                trans = 1.0
                t = step  # Start at one step to avoid self-hit
                
                while t < max_dist and trans > 0.001:
                    px = ox + rx * t
                    py = oy + ry * t
                    pz = oz + rz * t
                    
                    # Convert to grid indices
                    gi = int(px / dx)
                    gj = int(py / dy)
                    gk = int(pz / dz)
                    
                    # Check bounds - if we exit domain, ray is clear
                    if gi < 0 or gi >= nx or gj < 0 or gj >= ny:
                        break
                    if gk < 0 or gk >= nz:
                        break
                    
                    # Check solid hit - complete occlusion
                    if is_solid_field[gi, gj, gk] == 1:
                        trans = 0.0
                        break
                    
                    # Check vegetation attenuation (Beer-Lambert)
                    lad_val = lad_field[gi, gj, gk]
                    if lad_val > 0.0:
                        trans *= ti.exp(-ext * lad_val * step)
                    
                    t += step
                
                transmittance_field[i, j] = trans
    
    step_size = meshsize * 0.5
    max_trace_dist = float(max(nx, ny, nz) * meshsize * 2)
    
    trace_rays(
        sun_dir_x, sun_dir_y, sun_dir_z,
        view_height_k, ext_coef,
        step_size, max_trace_dist
    )
    
    return transmittance_field.to_numpy()


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
