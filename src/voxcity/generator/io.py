def save_voxcity_data(output_path, voxcity_grid, building_height_grid, building_min_height_grid, 
                     building_id_grid, canopy_height_grid, land_cover_grid, dem_grid, 
                     building_gdf, meshsize, rectangle_vertices):
    import pickle
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data_dict = {
        'voxcity_grid': voxcity_grid,
        'building_height_grid': building_height_grid,
        'building_min_height_grid': building_min_height_grid,
        'building_id_grid': building_id_grid,
        'canopy_height_grid': canopy_height_grid,
        'land_cover_grid': land_cover_grid,
        'dem_grid': dem_grid,
        'building_gdf': building_gdf,
        'meshsize': meshsize,
        'rectangle_vertices': rectangle_vertices
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"Voxcity data saved to {output_path}")


def load_voxcity(input_path):
    import pickle
    import numpy as np
    from ..models import GridMetadata, VoxelGrid, BuildingGrid, LandCoverGrid, DemGrid, CanopyGrid, VoxCity

    with open(input_path, 'rb') as f:
        d = pickle.load(f)

    rv = d.get('rectangle_vertices') or []
    if rv:
        xs = [p[0] for p in rv]
        ys = [p[1] for p in rv]
        bounds = (min(xs), min(ys), max(xs), max(ys))
    else:
        ny, nx = d['land_cover_grid'].shape
        ms = float(d['meshsize'])
        bounds = (0.0, 0.0, nx * ms, ny * ms)

    meta = GridMetadata(crs='EPSG:4326', bounds=bounds, meshsize=float(d['meshsize']))

    voxels = VoxelGrid(classes=d['voxcity_grid'], meta=meta)
    buildings = BuildingGrid(
        heights=d['building_height_grid'],
        min_heights=d['building_min_height_grid'],
        ids=d['building_id_grid'],
        meta=meta,
    )
    land = LandCoverGrid(classes=d['land_cover_grid'], meta=meta)
    dem = DemGrid(elevation=d['dem_grid'], meta=meta)
    canopy = CanopyGrid(top=d.get('canopy_height_grid'), bottom=None, meta=meta)

    extras = {
        'rectangle_vertices': d.get('rectangle_vertices'),
        'building_gdf': d.get('building_gdf'),
    }

    return VoxCity(voxels=voxels, buildings=buildings, land_cover=land, dem=dem, tree_canopy=canopy, extras=extras)



def save_voxcity(output_path, city):
    """Save a VoxCity instance to disk.

    This is a simplified, object-centric alternative to save_voxcity_data.
    """
    import pickle
    import os
    from ..models import VoxCity as _VoxCity

    if not isinstance(city, _VoxCity):
        raise TypeError("save_voxcity expects a VoxCity instance")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data_dict = {
        'voxcity_grid': city.voxels.classes,
        'building_height_grid': city.buildings.heights,
        'building_min_height_grid': city.buildings.min_heights,
        'building_id_grid': city.buildings.ids,
        'canopy_height_grid': getattr(city.tree_canopy, 'top', None) if city.tree_canopy is not None else None,
        'land_cover_grid': city.land_cover.classes,
        'dem_grid': city.dem.elevation,
        'building_gdf': city.extras.get('building_gdf'),
        'meshsize': city.voxels.meta.meshsize,
        'rectangle_vertices': city.extras.get('rectangle_vertices')
    }

    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f)

    print(f"Voxcity data saved to {output_path}")
