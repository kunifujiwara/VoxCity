import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, mapping
from shapely.errors import GEOSException
from affine import Affine
from rtree import index
from rasterio import features

from ..utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    convert_format_lat_lon,
)
from ..heights import (
    extract_building_heights_from_geotiff,
    extract_building_heights_from_gdf,
    complement_building_heights_from_gdf,
)
from ..overlap import (
    process_building_footprints_by_overlap,
)
from ...downloader.gee import (
    get_roi,
    save_geotiff_open_buildings_temporal,
)
from .core import calculate_grid_size, create_cell_polygon


def create_building_height_grid_from_gdf_polygon(
    gdf: gpd.GeoDataFrame,
    meshsize: float,
    rectangle_vertices: List[Tuple[float, float]],
    overlapping_footprint: any = "auto",
    gdf_comp: Optional[gpd.GeoDataFrame] = None,
    geotiff_path_comp: Optional[str] = None,
    complement_building_footprints: Optional[bool] = None,
    complement_height: Optional[float] = None
):
    """
    Create a building height grid from GeoDataFrame data within a polygon boundary.
    Returns: (building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings)
    """
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]

    dist_side_1 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_1[0], vertex_1[1])
    dist_side_2 = calculate_distance(geod, vertex_0[0], vertex_0[1], vertex_3[0], vertex_3[1])

    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)

    extent = [
        min(coord[1] for coord in rectangle_vertices),
        max(coord[1] for coord in rectangle_vertices),
        min(coord[0] for coord in rectangle_vertices),
        max(coord[0] for coord in rectangle_vertices)
    ]
    plotting_box = box(extent[2], extent[0], extent[3], extent[1])
    
    # Handle None or empty GeoDataFrame
    if gdf is None or len(gdf) == 0:
        filtered_gdf = gpd.GeoDataFrame(
            columns=['building_id', 'height', 'storeys', 'ground_elevation', 'geometry', 'source_file', 'id'],
            geometry='geometry'
        )
        if gdf is not None and gdf.crs is not None:
            filtered_gdf = filtered_gdf.set_crs(gdf.crs)
        print("0 of the total 0 building footprint from the base data source did not have height data.")
    else:
        filtered_gdf = gdf[gdf.geometry.intersects(plotting_box)].copy()
        zero_height_count = len(filtered_gdf[filtered_gdf['height'] == 0]) if 'height' in filtered_gdf.columns else 0
        nan_height_count = len(filtered_gdf[filtered_gdf['height'].isna()]) if 'height' in filtered_gdf.columns else 0
        print(f"{zero_height_count+nan_height_count} of the total {len(filtered_gdf)} building footprint from the base data source did not have height data.")

    if gdf_comp is not None:
        filtered_gdf_comp = gdf_comp[gdf_comp.geometry.intersects(plotting_box)].copy()
        if complement_building_footprints:
            filtered_gdf = complement_building_heights_from_gdf(filtered_gdf, filtered_gdf_comp)
        else:
            filtered_gdf = extract_building_heights_from_gdf(filtered_gdf, filtered_gdf_comp)
    elif geotiff_path_comp:
        filtered_gdf = extract_building_heights_from_geotiff(geotiff_path_comp, filtered_gdf)

    filtered_gdf = process_building_footprints_by_overlap(filtered_gdf, overlap_threshold=0.5)

    mode = overlapping_footprint
    if mode is None:
        mode = "auto"
    mode_norm = mode.strip().lower() if isinstance(mode, str) else mode

    def _decide_auto_mode(gdf_in) -> bool:
        try:
            n_buildings = len(gdf_in)
            if n_buildings == 0:
                return False
            num_cells = max(1, int(grid_size[0]) * int(grid_size[1]))
            density = float(n_buildings) / float(num_cells)

            sample_n = min(800, n_buildings)
            idx_rt = index.Index()
            geoms = []
            areas = []
            for i, geom in enumerate(gdf_in.geometry):
                g = geom
                if not getattr(g, "is_valid", True):
                    try:
                        g = g.buffer(0)
                    except Exception:
                        pass
                geoms.append(g)
                try:
                    areas.append(g.area)
                except Exception:
                    areas.append(0.0)
                try:
                    idx_rt.insert(i, g.bounds)
                except Exception:
                    pass
            with_overlap = 0
            step = max(1, n_buildings // sample_n)
            checked = 0
            for i in range(0, n_buildings, step):
                if checked >= sample_n:
                    break
                gi = geoms[i]
                ai = areas[i] if i < len(areas) else 0.0
                if gi is None:
                    continue
                try:
                    potentials = list(idx_rt.intersection(gi.bounds))
                except Exception:
                    potentials = []
                overlapped = False
                for j in potentials:
                    if j == i or j >= len(geoms):
                        continue
                    gj = geoms[j]
                    if gj is None:
                        continue
                    try:
                        if gi.intersects(gj):
                            inter = gi.intersection(gj)
                            inter_area = getattr(inter, "area", 0.0)
                            if inter_area > 0.0:
                                aj = areas[j] if j < len(areas) else 0.0
                                ref_area = max(1e-9, min(ai, aj) if ai > 0 and aj > 0 else (ai if ai > 0 else aj))
                                if (inter_area / ref_area) >= 0.2:
                                    overlapped = True
                                    break
                    except Exception:
                        continue
                if overlapped:
                    with_overlap += 1
                checked += 1
            overlap_ratio = (with_overlap / checked) if checked > 0 else 0.0
            if overlap_ratio >= 0.15:
                return True
            if overlap_ratio >= 0.08 and density > 0.15:
                return True
            if n_buildings <= 200 and overlap_ratio >= 0.05:
                return True
            return False
        except Exception:
            return False

    if mode_norm == "auto":
        use_precise = _decide_auto_mode(filtered_gdf)
    elif mode_norm is True:
        use_precise = True
    else:
        use_precise = False

    if use_precise:
        return _process_with_geometry_intersection(
            filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec, complement_height
        )
    return _process_with_rasterio(
        filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec,
        rectangle_vertices, complement_height
    )


def _process_with_geometry_intersection(filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec, complement_height):
    # Use (rows, cols) = (grid_size[1], grid_size[0]) = (Y, X) convention to match other VoxCity grids
    output_shape = (grid_size[1], grid_size[0])
    building_height_grid = np.zeros(output_shape)
    building_id_grid = np.zeros(output_shape)
    building_min_height_grid = np.empty(output_shape, dtype=object)
    for i in range(output_shape[0]):
        for j in range(output_shape[1]):
            building_min_height_grid[i, j] = []

    building_polygons = []
    for idx_b, row in filtered_gdf.iterrows():
        polygon = row.geometry
        height = row.get('height', None)
        if complement_height is not None and (height == 0 or height is None):
            height = complement_height
        min_height = row.get('min_height', 0)
        if pd.isna(min_height):
            min_height = 0
        is_inner = row.get('is_inner', False)
        # Fix: Handle NaN values for is_inner (NaN is truthy, causing buildings to be skipped)
        if pd.isna(is_inner):
            is_inner = False
        feature_id = row.get('id', idx_b)
        if not polygon.is_valid:
            try:
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    polygon = polygon.simplify(1e-8)
            except Exception:
                pass
        bounding_box = polygon.bounds
        building_polygons.append((
            polygon, bounding_box, height, min_height, is_inner, feature_id
        ))

    idx = index.Index()
    for i_b, (poly, bbox, _, _, _, _) in enumerate(building_polygons):
        idx.insert(i_b, bbox)

    INTERSECTION_THRESHOLD = 0.3
    # Loop over grid in (X, Y) order for cell creation, but store in (Y, X) order
    for ix in range(grid_size[0]):  # X direction (along u_vec)
        for iy in range(grid_size[1]):  # Y direction (along v_vec)
            cell = create_cell_polygon(origin, ix, iy, adjusted_meshsize, u_vec, v_vec)
            if not cell.is_valid:
                cell = cell.buffer(0)
            cell_area = cell.area
            potential = list(idx.intersection(cell.bounds))
            if not potential:
                continue
            cell_buildings = []
            for k in potential:
                bpoly, bbox, height, minh, inr, fid = building_polygons[k]
                sort_val = height if (height is not None) else -float('inf')
                cell_buildings.append((k, bpoly, bbox, height, minh, inr, fid, sort_val))
            cell_buildings.sort(key=lambda x: x[-1], reverse=True)

            found_intersection = False
            all_zero_or_nan = True
            # Store in (Y, X) = (iy, ix) order to match (rows, cols) convention
            row_idx, col_idx = iy, ix
            for (k, polygon, bbox, height, min_height, is_inner, feature_id, _) in cell_buildings:
                try:
                    minx_p, miny_p, maxx_p, maxy_p = bbox
                    minx_c, miny_c, maxx_c, maxy_c = cell.bounds
                    overlap_minx = max(minx_p, minx_c)
                    overlap_miny = max(miny_p, miny_c)
                    overlap_maxx = min(maxx_p, maxx_c)
                    overlap_maxy = min(maxy_p, maxy_c)
                    if (overlap_maxx <= overlap_minx) or (overlap_maxy <= overlap_miny):
                        continue
                    bbox_intersect_area = (overlap_maxx - overlap_minx) * (overlap_maxy - overlap_miny)
                    if bbox_intersect_area < INTERSECTION_THRESHOLD * cell_area:
                        continue
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    if cell.intersects(polygon):
                        intersection = cell.intersection(polygon)
                        inter_area = intersection.area
                        if (inter_area / cell_area) > INTERSECTION_THRESHOLD:
                            found_intersection = True
                            if not is_inner:
                                building_min_height_grid[row_idx, col_idx].append([min_height, height])
                                building_id_grid[row_idx, col_idx] = feature_id
                                if (height is not None and not np.isnan(height) and height > 0):
                                    all_zero_or_nan = False
                                    current_height = building_height_grid[row_idx, col_idx]
                                    if (current_height == 0 or np.isnan(current_height) or current_height < height):
                                        building_height_grid[row_idx, col_idx] = height
                            else:
                                building_min_height_grid[row_idx, col_idx] = [[0, 0]]
                                building_height_grid[row_idx, col_idx] = 0
                                found_intersection = True
                                all_zero_or_nan = False
                                break
                except (GEOSException, ValueError):
                    try:
                        simplified_polygon = polygon.simplify(1e-8)
                        if simplified_polygon.is_valid:
                            intersection = cell.intersection(simplified_polygon)
                            inter_area = intersection.area
                            if (inter_area / cell_area) > INTERSECTION_THRESHOLD:
                                found_intersection = True
                                if not is_inner:
                                    building_min_height_grid[row_idx, col_idx].append([min_height, height])
                                    building_id_grid[row_idx, col_idx] = feature_id
                                    if (height is not None and not np.isnan(height) and height > 0):
                                        all_zero_or_nan = False
                                        if (building_height_grid[row_idx, col_idx] == 0 or 
                                            np.isnan(building_height_grid[row_idx, col_idx]) or 
                                            building_height_grid[row_idx, col_idx] < height):
                                            building_height_grid[row_idx, col_idx] = height
                                else:
                                    building_min_height_grid[row_idx, col_idx] = [[0, 0]]
                                    building_height_grid[row_idx, col_idx] = 0
                                    found_intersection = True
                                    all_zero_or_nan = False
                                    break
                    except Exception:
                        continue
            if found_intersection and all_zero_or_nan:
                building_height_grid[row_idx, col_idx] = np.nan

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_gdf


def _process_with_rasterio(filtered_gdf, grid_size, adjusted_meshsize, origin, u_vec, v_vec, rectangle_vertices, complement_height):
    u_step = adjusted_meshsize[0] * u_vec
    v_step = adjusted_meshsize[1] * v_vec
    top_left = origin + grid_size[1] * v_step
    transform = Affine(u_step[0], -v_step[0], top_left[0],
                       u_step[1], -v_step[1], top_left[1])

    filtered_gdf = filtered_gdf.copy()
    if complement_height is not None:
        mask = (filtered_gdf['height'] == 0) | (filtered_gdf['height'].isna())
        filtered_gdf.loc[mask, 'height'] = complement_height

    # Preserve existing min_height values; only set default for missing/NaN
    if 'min_height' not in filtered_gdf.columns:
        filtered_gdf['min_height'] = 0
    else:
        filtered_gdf['min_height'] = filtered_gdf['min_height'].fillna(0)
    if 'is_inner' not in filtered_gdf.columns:
        filtered_gdf['is_inner'] = False
    else:
        try:
            filtered_gdf['is_inner'] = filtered_gdf['is_inner'].fillna(False).astype(bool)
        except Exception:
            filtered_gdf['is_inner'] = False
    if 'id' not in filtered_gdf.columns:
        filtered_gdf['id'] = range(len(filtered_gdf))

    regular_buildings = filtered_gdf[~filtered_gdf['is_inner']].copy()
    regular_buildings = regular_buildings.sort_values('height', ascending=True, na_position='first')

    height_raster = np.zeros((grid_size[1], grid_size[0]), dtype=np.float64)
    id_raster = np.zeros((grid_size[1], grid_size[0]), dtype=np.float64)

    if len(regular_buildings) > 0:
        valid_buildings = regular_buildings[regular_buildings.geometry.is_valid].copy()
        if len(valid_buildings) > 0:
            height_shapes = [(mapping(geom), height) for geom, height in 
                           zip(valid_buildings.geometry, valid_buildings['height']) 
                           if pd.notna(height) and height > 0]
            if height_shapes:
                height_raster = features.rasterize(
                    height_shapes,
                    out_shape=(grid_size[1], grid_size[0]),
                    transform=transform,
                    fill=0,
                    dtype=np.float64
                )
            id_shapes = [(mapping(geom), id_val) for geom, id_val in 
                        zip(valid_buildings.geometry, valid_buildings['id'])]
            if id_shapes:
                id_raster = features.rasterize(
                    id_shapes,
                    out_shape=(grid_size[1], grid_size[0]),
                    transform=transform,
                    fill=0,
                    dtype=np.float64
                )

    inner_buildings = filtered_gdf[filtered_gdf['is_inner']].copy()
    if len(inner_buildings) > 0:
        inner_shapes = [(mapping(geom), 1) for geom in inner_buildings.geometry if geom.is_valid]
        if inner_shapes:
            inner_mask = features.rasterize(
                inner_shapes,
                out_shape=(grid_size[1], grid_size[0]),
                transform=transform,
                fill=0,
                dtype=np.uint8
            )
            height_raster[inner_mask > 0] = 0
            id_raster[inner_mask > 0] = 0

    building_min_height_grid = np.empty(grid_size, dtype=object)
    min_heights_raster = np.zeros((grid_size[1], grid_size[0]), dtype=np.float64)
    if len(regular_buildings) > 0:
        valid_buildings = regular_buildings[regular_buildings.geometry.is_valid].copy()
        if len(valid_buildings) > 0:
            min_height_shapes = [(mapping(geom), min_h) for geom, min_h in 
                               zip(valid_buildings.geometry, valid_buildings['min_height']) 
                               if pd.notna(min_h)]
            if min_height_shapes:
                min_heights_raster = features.rasterize(
                    min_height_shapes,
                    out_shape=(grid_size[1], grid_size[0]),
                    transform=transform,
                    fill=0,
                    dtype=np.float64
                )

    # Apply flipud for north-up orientation (matching other VoxCity grids like dem_grid)
    # Do NOT transpose - keep (rows, cols) = (Y, X) convention
    building_height_grid = np.flipud(height_raster)
    building_id_grid = np.flipud(id_raster)
    min_heights = np.flipud(min_heights_raster)

    # Initialize min_height_grid with correct shape (grid_size[1], grid_size[0]) = (rows, cols)
    building_min_height_grid = np.empty((grid_size[1], grid_size[0]), dtype=object)
    for i in range(grid_size[1]):
        for j in range(grid_size[0]):
            if building_height_grid[i, j] > 0:
                building_min_height_grid[i, j] = [[min_heights[i, j], building_height_grid[i, j]]]
            else:
                building_min_height_grid[i, j] = []

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_gdf


def create_building_height_grid_from_open_building_temporal_polygon(meshsize, rectangle_vertices, output_dir):
    """
    Create a building height grid from OpenBuildings temporal data within a polygon.
    """
    roi = get_roi(rectangle_vertices)
    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "building_height.tif")
    save_geotiff_open_buildings_temporal(roi, geotiff_path)
    from .raster import create_height_grid_from_geotiff_polygon
    building_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)

    building_min_height_grid = np.empty(building_height_grid.shape, dtype=object)
    for i in range(building_height_grid.shape[0]):
        for j in range(building_height_grid.shape[1]):
            if building_height_grid[i, j] <= 0:
                building_min_height_grid[i, j] = []
            else:
                building_min_height_grid[i, j] = [[0, building_height_grid[i, j]]]

    filtered_buildings = gpd.GeoDataFrame()
    building_id_grid = np.zeros_like(building_height_grid, dtype=int)
    non_zero_positions = np.nonzero(building_height_grid)
    sequence = np.arange(1, len(non_zero_positions[0]) + 1)
    building_id_grid[non_zero_positions] = sequence

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings


