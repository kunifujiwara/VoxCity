"""
Utilities for processing overlaps between building footprints.
"""

from rtree import index
from shapely.errors import GEOSException


def process_building_footprints_by_overlap(filtered_gdf, overlap_threshold=0.5):
    """
    Merge overlapping buildings based on area overlap ratio, assigning the ID of the larger building
    to smaller overlapping ones.
    """
    gdf = filtered_gdf.copy()

    if 'id' not in gdf.columns:
        gdf['id'] = gdf.index

    if gdf.crs is None:
        gdf_projected = gdf.copy()
    else:
        gdf_projected = gdf.to_crs("EPSG:3857")

    gdf_projected['area'] = gdf_projected.geometry.area
    gdf_projected = gdf_projected.sort_values(by='area', ascending=False)
    gdf_projected = gdf_projected.reset_index(drop=True)

    spatial_idx = index.Index()
    for i, geom in enumerate(gdf_projected.geometry):
        if geom.is_valid:
            spatial_idx.insert(i, geom.bounds)
        else:
            fixed_geom = geom.buffer(0)
            if fixed_geom.is_valid:
                spatial_idx.insert(i, fixed_geom.bounds)

    id_mapping = {}

    for i in range(1, len(gdf_projected)):
        current_poly = gdf_projected.iloc[i].geometry
        current_area = gdf_projected.iloc[i].area
        current_id = gdf_projected.iloc[i]['id']

        if current_id in id_mapping:
            continue

        if not current_poly.is_valid:
            current_poly = current_poly.buffer(0)
            if not current_poly.is_valid:
                continue

        potential_overlaps = [j for j in spatial_idx.intersection(current_poly.bounds) if j < i]

        for j in potential_overlaps:
            larger_poly = gdf_projected.iloc[j].geometry
            larger_id = gdf_projected.iloc[j]['id']

            if larger_id in id_mapping:
                larger_id = id_mapping[larger_id]

            if not larger_poly.is_valid:
                larger_poly = larger_poly.buffer(0)
                if not larger_poly.is_valid:
                    continue

            try:
                if current_poly.intersects(larger_poly):
                    overlap = current_poly.intersection(larger_poly)
                    overlap_ratio = overlap.area / current_area
                    if overlap_ratio > overlap_threshold:
                        id_mapping[current_id] = larger_id
                        gdf_projected.at[i, 'id'] = larger_id
                        break
            except (GEOSException, ValueError):
                continue

    for i, row in filtered_gdf.iterrows():
        orig_id = row.get('id')
        if orig_id in id_mapping:
            filtered_gdf.at[i, 'id'] = id_mapping[orig_id]

    return filtered_gdf


