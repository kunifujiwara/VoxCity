"""
Height extraction and complement utilities for building footprints.
"""

from typing import List, Dict

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.errors import GEOSException
from shapely.geometry import shape
from rtree import index
import rasterio
from pyproj import Transformer, CRS


def extract_building_heights_from_gdf(gdf_0: gpd.GeoDataFrame, gdf_1: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Extract building heights from one GeoDataFrame and apply them to another based on spatial overlap.
    """
    gdf_primary = gdf_0.copy()
    gdf_ref = gdf_1.copy()

    if 'height' not in gdf_primary.columns:
        gdf_primary['height'] = 0.0
    if 'height' not in gdf_ref.columns:
        gdf_ref['height'] = 0.0

    count_0 = 0
    count_1 = 0
    count_2 = 0

    spatial_index = index.Index()
    for i, geom in enumerate(gdf_ref.geometry):
        if geom.is_valid:
            spatial_index.insert(i, geom.bounds)

    for idx_primary, row in gdf_primary.iterrows():
        if row['height'] <= 0 or pd.isna(row['height']):
            count_0 += 1
            geom = row.geometry

            overlapping_height_area = 0
            overlapping_area = 0

            potential_matches = list(spatial_index.intersection(geom.bounds))

            for ref_idx in potential_matches:
                if ref_idx >= len(gdf_ref):
                    continue

                ref_row = gdf_ref.iloc[ref_idx]
                try:
                    if geom.intersects(ref_row.geometry):
                        overlap_area = geom.intersection(ref_row.geometry).area
                        overlapping_height_area += ref_row['height'] * overlap_area
                        overlapping_area += overlap_area
                except GEOSException:
                    try:
                        fixed_ref_geom = ref_row.geometry.buffer(0)
                        if geom.intersects(fixed_ref_geom):
                            overlap_area = geom.intersection(fixed_ref_geom).area
                            overlapping_height_area += ref_row['height'] * overlap_area
                            overlapping_area += overlap_area
                    except Exception:
                        print(f"Failed to fix polygon")
                    continue

            if overlapping_height_area > 0:
                count_1 += 1
                new_height = overlapping_height_area / overlapping_area
                gdf_primary.at[idx_primary, 'height'] = new_height
            else:
                count_2 += 1
                gdf_primary.at[idx_primary, 'height'] = np.nan

    if count_0 > 0:
        print(f"For {count_1} of these building footprints without height, values from the complementary source were assigned.")
        print(f"For {count_2} of these building footprints without height, no data exist in complementary data.")

    return gdf_primary


def complement_building_heights_from_gdf(gdf_0, gdf_1, primary_id='id', ref_id='id'):
    """
    Vectorized approach with GeoPandas to compute weighted heights and add non-intersecting buildings.
    Returns a single combined GeoDataFrame.
    """
    gdf_primary = gdf_0.copy()
    gdf_ref = gdf_1.copy()

    if 'height' not in gdf_primary.columns:
        gdf_primary['height'] = 0.0
    if 'height' not in gdf_ref.columns:
        gdf_ref['height'] = 0.0

    gdf_primary = gdf_primary.rename(columns={'height': 'height_primary'})
    gdf_ref = gdf_ref.rename(columns={'height': 'height_ref'})

    intersect_gdf = gpd.overlay(gdf_primary, gdf_ref, how='intersection')
    intersect_gdf['intersect_area'] = intersect_gdf.area
    intersect_gdf['height_area'] = intersect_gdf['height_ref'] * intersect_gdf['intersect_area']

    group_cols = {
        'height_area': 'sum',
        'intersect_area': 'sum'
    }
    grouped = intersect_gdf.groupby(f'{primary_id}_1').agg(group_cols)
    grouped['weighted_height'] = grouped['height_area'] / grouped['intersect_area']

    gdf_primary = gdf_primary.merge(grouped['weighted_height'],
                                    left_on=primary_id,
                                    right_index=True,
                                    how='left')

    zero_or_nan_mask = (gdf_primary['height_primary'] == 0) | (gdf_primary['height_primary'].isna())
    valid_weighted_height_mask = zero_or_nan_mask & gdf_primary['weighted_height'].notna()
    gdf_primary.loc[valid_weighted_height_mask, 'height_primary'] = gdf_primary.loc[valid_weighted_height_mask, 'weighted_height']
    gdf_primary['height_primary'] = gdf_primary['height_primary'].fillna(np.nan)

    sjoin_gdf = gpd.sjoin(gdf_ref, gdf_primary, how='left', predicate='intersects')
    non_intersect_mask = sjoin_gdf[f'{primary_id}_right'].isna()
    non_intersect_ids = sjoin_gdf[non_intersect_mask][f'{ref_id}_left'].unique()
    gdf_ref_non_intersect = gdf_ref[gdf_ref[ref_id].isin(non_intersect_ids)]
    gdf_ref_non_intersect = gdf_ref_non_intersect.rename(columns={'height_ref': 'height'})

    gdf_primary = gdf_primary.rename(columns={'height_primary': 'height'})
    if 'weighted_height' in gdf_primary.columns:
        gdf_primary.drop(columns='weighted_height', inplace=True)

    final_gdf = pd.concat([gdf_primary, gdf_ref_non_intersect], ignore_index=True)

    count_total = len(gdf_primary)
    count_0 = len(gdf_primary[zero_or_nan_mask])
    count_1 = len(gdf_primary[valid_weighted_height_mask])
    count_2 = count_0 - count_1
    count_3 = len(gdf_ref_non_intersect)
    count_4 = count_3
    height_mask = gdf_ref_non_intersect['height'].notna() & (gdf_ref_non_intersect['height'] > 0)
    count_5 = len(gdf_ref_non_intersect[height_mask])
    count_6 = count_4 - count_5
    final_height_mask = final_gdf['height'].notna() & (final_gdf['height'] > 0)
    count_7 = len(final_gdf[final_height_mask])
    count_8 = len(final_gdf)

    if count_0 > 0:
        print(f"{count_0} of the total {count_total} building footprints from base data source did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from complementary data were assigned.")
        print(f"For the rest {count_2}, no data exists in complementary data.")
        print(f"Footprints of {count_3} buildings were added from the complementary source.")
        print(f"Of these {count_4} additional building footprints, {count_5} had height data while {count_6} had no height data.")
        print(f"In total, {count_7} buildings had height data out of {count_8} total building footprints.")

    return final_gdf


def extract_building_heights_from_geotiff(geotiff_path, gdf):
    """
    Extract building heights from a GeoTIFF raster for building footprints in a GeoDataFrame.
    """
    gdf = gdf.copy()

    count_0 = 0
    count_1 = 0
    count_2 = 0

    with rasterio.open(geotiff_path) as src:
        transformer = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)

        mask_condition = (gdf.geometry.geom_type == 'Polygon') & ((gdf.get('height', 0) <= 0) | gdf.get('height').isna())
        buildings_to_process = gdf[mask_condition]
        count_0 = len(buildings_to_process)

        for idx, row in buildings_to_process.iterrows():
            coords = list(row.geometry.exterior.coords)
            transformed_coords = [transformer.transform(lon, lat) for lon, lat in coords]
            polygon = shape({"type": "Polygon", "coordinates": [transformed_coords]})

            try:
                masked_data, _ = rasterio.mask.mask(src, [polygon], crop=True, all_touched=True)
                heights = masked_data[0][masked_data[0] != src.nodata]
                if len(heights) > 0:
                    count_1 += 1
                    gdf.at[idx, 'height'] = float(np.mean(heights))
                else:
                    count_2 += 1
                    gdf.at[idx, 'height'] = np.nan
            except ValueError as e:
                print(f"Error processing building at index {idx}. Error: {str(e)}")
                gdf.at[idx, 'height'] = None

    if count_0 > 0:
        print(f"{count_0} of the total {len(gdf)} building footprint from OSM did not have height data.")
        print(f"For {count_1} of these building footprints without height, values from complementary data were assigned.")
        print(f"For {count_2} of these building footprints without height, no data exist in complementary data.")

    return gdf


