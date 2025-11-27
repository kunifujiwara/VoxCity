"""Functions for updating VoxCity objects with new grid data."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np

from ..models import (
    GridMetadata,
    BuildingGrid,
    LandCoverGrid,
    DemGrid,
    VoxelGrid,
    CanopyGrid,
    VoxCity,
)
from .voxelizer import Voxelizer


def update_voxcity(
    city: VoxCity,
    *,
    buildings: Optional[BuildingGrid] = None,
    building_heights: Optional[np.ndarray] = None,
    building_min_heights: Optional[np.ndarray] = None,
    building_ids: Optional[np.ndarray] = None,
    land_cover: Optional[Union[LandCoverGrid, np.ndarray]] = None,
    dem: Optional[Union[DemGrid, np.ndarray]] = None,
    tree_canopy: Optional[Union[CanopyGrid, np.ndarray]] = None,
    canopy_top: Optional[np.ndarray] = None,
    canopy_bottom: Optional[np.ndarray] = None,
    building_gdf=None,
    tree_gdf=None,
    tree_gdf_mode: str = "replace",
    land_cover_source: Optional[str] = None,
    trunk_height_ratio: Optional[float] = None,
    voxel_dtype=None,
    max_voxel_ram_mb: Optional[float] = None,
    inplace: bool = False,
) -> VoxCity:
    """
    Update a VoxCity object with new grid data and regenerate the VoxelGrid.

    This function allows partial updates - only the grids you provide will be
    updated, while the rest will be taken from the existing VoxCity object.
    The VoxelGrid is always regenerated from the (updated) component grids.

    Parameters
    ----------
    city : VoxCity
        The existing VoxCity object to update.

    buildings : BuildingGrid, optional
        Complete replacement for the building grid. If provided, takes precedence
        over individual building_heights/building_min_heights/building_ids.

    building_heights : np.ndarray, optional
        2D array of building heights. If buildings is not provided, this updates
        only the heights while keeping existing min_heights and ids.

    building_min_heights : np.ndarray, optional
        2D object-dtype array of lists containing [min_height, max_height] pairs
        for each building segment per cell.

    building_ids : np.ndarray, optional
        2D array of building IDs per cell.

    land_cover : LandCoverGrid or np.ndarray, optional
        New land cover data. Can be a LandCoverGrid or a raw numpy array.

    dem : DemGrid or np.ndarray, optional
        New DEM/elevation data. Can be a DemGrid or a raw numpy array.

    tree_canopy : CanopyGrid or np.ndarray, optional
        New tree canopy data. Can be a CanopyGrid (with top/bottom) or a raw
        numpy array (interpreted as canopy top heights).

    canopy_top : np.ndarray, optional
        2D array of tree canopy top heights. Takes precedence over tree_canopy
        for top heights if both are provided.

    canopy_bottom : np.ndarray, optional
        2D array of tree canopy bottom heights (crown base).

    building_gdf : GeoDataFrame, optional
        Updated building GeoDataFrame. If provided without building grids
        (building_heights, building_min_heights, building_ids), the function
        will automatically generate the building grids from the GeoDataFrame
        using create_building_height_grid_from_gdf_polygon. The GeoDataFrame
        is also stored in city.extras['building_gdf'].

    tree_gdf : GeoDataFrame, optional
        Updated tree GeoDataFrame. If provided without tree canopy data
        (tree_canopy, canopy_top, canopy_bottom), the function will
        automatically generate the canopy grids from the GeoDataFrame
        using create_canopy_grids_from_tree_gdf. The GeoDataFrame must
        contain 'top_height', 'bottom_height', 'crown_diameter', and
        'geometry' columns. The GeoDataFrame is stored in city.extras['tree_gdf'].

    tree_gdf_mode : str, default "replace"
        How to combine tree_gdf with existing canopy data. Options:
        - "replace": Replace the existing canopy grids with new ones from tree_gdf.
        - "add": Merge the tree_gdf grids with existing canopy grids by taking
          the maximum height at each cell (preserves existing trees).

    land_cover_source : str, optional
        The land cover source name for proper voxelization. If not provided,
        attempts to use the source from city.extras or defaults to 'OpenStreetMap'.

    trunk_height_ratio : float, optional
        Ratio of trunk height to total tree height for canopy bottom calculation.
        Default is approximately 0.588 (11.76/19.98).

    voxel_dtype : dtype, optional
        NumPy dtype for the voxel grid. Defaults to np.int8.

    max_voxel_ram_mb : float, optional
        Maximum RAM in MB for voxel grid allocation. Raises MemoryError if exceeded.

    inplace : bool, default False
        If True, modifies the input city object directly and returns it.
        If False, creates and returns a new VoxCity object.

    Returns
    -------
    VoxCity
        The updated VoxCity object with regenerated VoxelGrid.

    Examples
    --------
    Update building heights and regenerate voxels:

    >>> import numpy as np
    >>> new_heights = city.buildings.heights.copy()
    >>> new_heights[10:20, 10:20] = 50.0  # Increase height in a region
    >>> updated = update_voxcity(city, building_heights=new_heights)

    Update with a complete new BuildingGrid:

    >>> from voxcity.models import BuildingGrid
    >>> new_buildings = BuildingGrid(heights=..., min_heights=..., ids=..., meta=city.buildings.meta)
    >>> updated = update_voxcity(city, buildings=new_buildings)

    Update land cover and DEM together:

    >>> updated = update_voxcity(city, land_cover=new_lc_array, dem=new_dem_array)

    Update buildings from GeoDataFrame (automatic grid generation):

    >>> updated = update_voxcity(city, building_gdf=updated_building_gdf)

    Update trees from GeoDataFrame (replace existing canopy):

    >>> updated = update_voxcity(city, tree_gdf=updated_tree_gdf)

    Add trees from GeoDataFrame to existing canopy:

    >>> updated = update_voxcity(city, tree_gdf=new_tree_gdf, tree_gdf_mode="add")
    """
    # Resolve metadata from existing city
    meta = city.buildings.meta
    meshsize = meta.meshsize

    # --- Auto-generate building grids from GeoDataFrame if provided ---
    if building_gdf is not None and buildings is None and building_heights is None:
        # Auto-generate building grids from the GeoDataFrame
        from ..geoprocessor.raster import create_building_height_grid_from_gdf_polygon
        
        rectangle_vertices = city.extras.get("rectangle_vertices")
        if rectangle_vertices is None:
            raise ValueError(
                "Cannot auto-generate building grids: 'rectangle_vertices' not found in city.extras. "
                "Provide building_heights, building_min_heights, and building_ids explicitly."
            )
        
        building_heights, building_min_heights, building_ids, _ = (
            create_building_height_grid_from_gdf_polygon(
                building_gdf,
                meshsize,
                rectangle_vertices,
            )
        )

    # --- Auto-generate canopy grids from tree GeoDataFrame if provided ---
    if tree_gdf is not None and tree_canopy is None and canopy_top is None:
        # Validate tree_gdf_mode
        if tree_gdf_mode not in ("replace", "add"):
            raise ValueError(
                f"Invalid tree_gdf_mode '{tree_gdf_mode}'. Must be 'replace' or 'add'."
            )
        
        # Auto-generate canopy grids from the tree GeoDataFrame
        from ..geoprocessor.raster import create_canopy_grids_from_tree_gdf
        
        rectangle_vertices = city.extras.get("rectangle_vertices")
        if rectangle_vertices is None:
            raise ValueError(
                "Cannot auto-generate canopy grids: 'rectangle_vertices' not found in city.extras. "
                "Provide canopy_top and canopy_bottom explicitly."
            )
        
        new_canopy_top, new_canopy_bottom = create_canopy_grids_from_tree_gdf(
            tree_gdf,
            meshsize,
            rectangle_vertices,
        )
        
        if tree_gdf_mode == "add":
            # Merge with existing canopy by taking maximum values
            existing_top = city.tree_canopy.top
            existing_bottom = city.tree_canopy.bottom
            if existing_top is not None:
                canopy_top = np.maximum(existing_top, new_canopy_top)
            else:
                canopy_top = new_canopy_top
            if existing_bottom is not None:
                canopy_bottom = np.maximum(existing_bottom, new_canopy_bottom)
            else:
                canopy_bottom = new_canopy_bottom
        else:
            # Replace mode: use new canopy grids directly
            canopy_top = new_canopy_top
            canopy_bottom = new_canopy_bottom

    # --- Resolve building data ---
    if buildings is not None:
        # Use provided BuildingGrid directly
        final_building_heights = buildings.heights
        final_building_min_heights = buildings.min_heights
        final_building_ids = buildings.ids
        final_building_meta = buildings.meta
    else:
        # Use individual arrays or fall back to existing
        final_building_heights = (
            building_heights if building_heights is not None else city.buildings.heights
        )
        final_building_min_heights = (
            building_min_heights
            if building_min_heights is not None
            else city.buildings.min_heights
        )
        final_building_ids = (
            building_ids if building_ids is not None else city.buildings.ids
        )
        final_building_meta = meta

    # --- Resolve land cover data ---
    if land_cover is not None:
        if isinstance(land_cover, LandCoverGrid):
            final_land_cover = land_cover.classes
        else:
            final_land_cover = land_cover
    else:
        final_land_cover = city.land_cover.classes

    # --- Resolve DEM data ---
    if dem is not None:
        if isinstance(dem, DemGrid):
            final_dem = dem.elevation
        else:
            final_dem = dem
    else:
        final_dem = city.dem.elevation

    # --- Resolve canopy data ---
    # Priority: canopy_top/canopy_bottom > tree_canopy > existing
    if canopy_top is not None:
        final_canopy_top = canopy_top
    elif tree_canopy is not None:
        if isinstance(tree_canopy, CanopyGrid):
            final_canopy_top = tree_canopy.top
        else:
            final_canopy_top = tree_canopy
    else:
        final_canopy_top = city.tree_canopy.top

    if canopy_bottom is not None:
        final_canopy_bottom = canopy_bottom
    elif tree_canopy is not None and isinstance(tree_canopy, CanopyGrid):
        final_canopy_bottom = tree_canopy.bottom
    else:
        final_canopy_bottom = city.tree_canopy.bottom

    # --- Determine land cover source ---
    if land_cover_source is None:
        # Try to get from extras
        land_cover_source = city.extras.get("land_cover_source")
        if land_cover_source is None:
            selected = city.extras.get("selected_sources", {})
            land_cover_source = selected.get("land_cover_source", "OpenStreetMap")

    # --- Build updated extras ---
    new_extras = dict(city.extras)
    if building_gdf is not None:
        new_extras["building_gdf"] = building_gdf
    if tree_gdf is not None:
        new_extras["tree_gdf"] = tree_gdf
    new_extras["canopy_top"] = final_canopy_top
    new_extras["canopy_bottom"] = final_canopy_bottom

    # --- Shape validation ---
    expected_shape = final_land_cover.shape
    shapes = {
        "building_heights": final_building_heights.shape,
        "building_min_heights": final_building_min_heights.shape,
        "building_ids": final_building_ids.shape,
        "land_cover": final_land_cover.shape,
        "dem": final_dem.shape,
        "canopy_top": final_canopy_top.shape if final_canopy_top is not None else None,
    }
    
    mismatched = {k: v for k, v in shapes.items() if v is not None and v != expected_shape}
    if mismatched:
        raise ValueError(
            f"Grid shape mismatch! Expected {expected_shape}, but got: {mismatched}. "
            f"All grids must have the same shape."
        )

    # --- Create Voxelizer and regenerate voxel grid ---
    _voxel_dtype = voxel_dtype if voxel_dtype is not None else np.int8
    voxelizer = Voxelizer(
        voxel_size=meshsize,
        land_cover_source=land_cover_source,
        trunk_height_ratio=trunk_height_ratio,
        voxel_dtype=_voxel_dtype,
        max_voxel_ram_mb=max_voxel_ram_mb,
    )

    new_voxel_classes = voxelizer.generate_combined(
        building_height_grid_ori=final_building_heights,
        building_min_height_grid_ori=final_building_min_heights,
        building_id_grid_ori=final_building_ids,
        land_cover_grid_ori=final_land_cover,
        dem_grid_ori=final_dem,
        tree_grid_ori=final_canopy_top,
        canopy_bottom_height_grid_ori=final_canopy_bottom,
    )

    # --- Assemble result ---
    new_voxels = VoxelGrid(classes=new_voxel_classes, meta=meta)
    new_buildings = BuildingGrid(
        heights=final_building_heights,
        min_heights=final_building_min_heights,
        ids=final_building_ids,
        meta=final_building_meta,
    )
    new_land_cover = LandCoverGrid(classes=final_land_cover, meta=meta)
    new_dem = DemGrid(elevation=final_dem, meta=meta)
    new_canopy = CanopyGrid(
        top=final_canopy_top,
        bottom=final_canopy_bottom,
        meta=meta,
    )

    if inplace:
        city.voxels = new_voxels
        city.buildings = new_buildings
        city.land_cover = new_land_cover
        city.dem = new_dem
        city.tree_canopy = new_canopy
        city.extras = new_extras
        return city
    else:
        return VoxCity(
            voxels=new_voxels,
            buildings=new_buildings,
            land_cover=new_land_cover,
            dem=new_dem,
            tree_canopy=new_canopy,
            extras=new_extras,
        )


def regenerate_voxels(
    city: VoxCity,
    *,
    land_cover_source: Optional[str] = None,
    trunk_height_ratio: Optional[float] = None,
    voxel_dtype=None,
    max_voxel_ram_mb: Optional[float] = None,
    inplace: bool = False,
) -> VoxCity:
    """
    Regenerate only the VoxelGrid from existing component grids.

    This is a convenience function for when you've modified the grids in-place
    and need to regenerate the voxels without passing all parameters.

    Parameters
    ----------
    city : VoxCity
        The VoxCity object whose voxels should be regenerated.

    land_cover_source : str, optional
        Land cover source for voxelization. Defaults to source from extras.

    trunk_height_ratio : float, optional
        Trunk height ratio for tree canopy calculation.

    voxel_dtype : dtype, optional
        NumPy dtype for voxel grid.

    max_voxel_ram_mb : float, optional
        Maximum RAM in MB for voxel allocation.

    inplace : bool, default False
        If True, modifies city directly; otherwise returns a new object.

    Returns
    -------
    VoxCity
        The VoxCity object with regenerated VoxelGrid.

    Examples
    --------
    >>> # Modify building heights in place
    >>> city.buildings.heights[50:60, 50:60] = 100.0
    >>> # Regenerate voxels to reflect the change
    >>> city = regenerate_voxels(city, inplace=True)
    """
    return update_voxcity(
        city,
        land_cover_source=land_cover_source,
        trunk_height_ratio=trunk_height_ratio,
        voxel_dtype=voxel_dtype,
        max_voxel_ram_mb=max_voxel_ram_mb,
        inplace=inplace,
    )

