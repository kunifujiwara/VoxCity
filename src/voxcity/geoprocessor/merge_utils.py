"""
Utilities to merge GeoDataFrames while resolving ID conflicts.
"""

import pandas as pd


def _merge_gdfs_with_missing_columns(gdf_1, gdf_2):
    """
    Helper to merge two GeoDataFrames while handling missing columns by filling with None.
    """
    columns_1 = set(gdf_1.columns)
    columns_2 = set(gdf_2.columns)

    only_in_1 = columns_1 - columns_2
    only_in_2 = columns_2 - columns_1

    for col in only_in_2:
        gdf_1[col] = None
    for col in only_in_1:
        gdf_2[col] = None

    all_columns = sorted(list(columns_1.union(columns_2)))
    gdf_1 = gdf_1[all_columns]
    gdf_2 = gdf_2[all_columns]

    merged_gdf = pd.concat([gdf_1, gdf_2], ignore_index=True)
    return merged_gdf


def merge_gdfs_with_id_conflict_resolution(gdf_1, gdf_2, id_columns=['id', 'building_id']):
    """
    Merge two GeoDataFrames while resolving ID conflicts by modifying IDs in the second GeoDataFrame.
    """
    gdf_primary = gdf_1.copy()
    gdf_secondary = gdf_2.copy()

    missing_columns = []
    for col in id_columns:
        if col not in gdf_primary.columns:
            missing_columns.append(f"'{col}' missing from gdf_1")
        if col not in gdf_secondary.columns:
            missing_columns.append(f"'{col}' missing from gdf_2")

    if missing_columns:
        print(f"Warning: Missing ID columns: {', '.join(missing_columns)}")
        id_columns = [col for col in id_columns if col in gdf_primary.columns and col in gdf_secondary.columns]

    if not id_columns:
        print("Warning: No valid ID columns found. Merging without ID conflict resolution.")
        merged_gdf = _merge_gdfs_with_missing_columns(gdf_primary, gdf_secondary)
        return merged_gdf

    max_ids = {}
    for col in id_columns:
        if gdf_primary[col].dtype in ['int64', 'int32', 'float64', 'float32']:
            max_ids[col] = gdf_primary[col].max()
        else:
            max_ids[col] = len(gdf_primary)

    next_ids = {col: max_ids[col] + 1 for col in id_columns}
    modified_buildings = 0

    for idx, row in gdf_secondary.iterrows():
        needs_new_ids = False
        for col in id_columns:
            current_id = row[col]
            if current_id in gdf_primary[col].values:
                needs_new_ids = True
                break
        if needs_new_ids:
            modified_buildings += 1
            for col in id_columns:
                new_id = next_ids[col]
                gdf_secondary.at[idx, col] = new_id
                next_ids[col] += 1

    merged_gdf = _merge_gdfs_with_missing_columns(gdf_primary, gdf_secondary)

    total_buildings = len(merged_gdf)
    primary_buildings = len(gdf_primary)
    secondary_buildings = len(gdf_secondary)

    print(f"Merged {primary_buildings} buildings from primary dataset with {secondary_buildings} buildings from secondary dataset.")
    print(f"Total buildings in merged dataset: {total_buildings}")
    if modified_buildings > 0:
        print(f"Modified IDs for {modified_buildings} buildings in secondary dataset to resolve conflicts.")

    return merged_gdf


