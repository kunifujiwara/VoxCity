import numpy as np
import geopandas as gpd
from shapely.geometry import box
from pyproj import CRS, Transformer


def grid_to_geodataframe(grid_ori, rectangle_vertices, meshsize):
    """
    Converts a 2D grid to a GeoDataFrame with cell polygons and values.
    Output CRS: EPSG:4326
    """
    grid = np.flipud(grid_ori.copy())

    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)

    rows, cols = grid.shape

    wgs84 = CRS.from_epsg(4326)
    web_mercator = CRS.from_epsg(3857)
    transformer_to_mercator = Transformer.from_crs(wgs84, web_mercator, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(web_mercator, wgs84, always_xy=True)

    min_x, min_y = transformer_to_mercator.transform(min_lon, min_lat)
    max_x, max_y = transformer_to_mercator.transform(max_lon, max_lat)

    cell_size_x = (max_x - min_x) / cols
    cell_size_y = (max_y - min_y) / rows

    polygons = []
    values = []

    for i in range(rows):
        for j in range(cols):
            cell_min_x = min_x + j * cell_size_x
            cell_max_x = min_x + (j + 1) * cell_size_x
            cell_min_y = max_y - (i + 1) * cell_size_y
            cell_max_y = max_y - i * cell_size_y
            cell_min_lon, cell_min_lat = transformer_to_wgs84.transform(cell_min_x, cell_min_y)
            cell_max_lon, cell_max_lat = transformer_to_wgs84.transform(cell_max_x, cell_max_y)
            cell_poly = box(cell_min_lon, cell_min_lat, cell_max_lon, cell_max_lat)
            polygons.append(cell_poly)
            values.append(grid[i, j])

    gdf = gpd.GeoDataFrame({'geometry': polygons, 'value': values}, crs=CRS.from_epsg(4326))
    return gdf


def grid_to_point_geodataframe(grid_ori, rectangle_vertices, meshsize):
    """
    Converts a 2D grid to a GeoDataFrame with point geometries at cell centers and values.
    Output CRS: EPSG:4326
    """
    import geopandas as gpd
    from shapely.geometry import Point

    grid = np.flipud(grid_ori.copy())

    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)

    rows, cols = grid.shape

    wgs84 = CRS.from_epsg(4326)
    web_mercator = CRS.from_epsg(3857)
    transformer_to_mercator = Transformer.from_crs(wgs84, web_mercator, always_xy=True)
    transformer_to_wgs84 = Transformer.from_crs(web_mercator, wgs84, always_xy=True)

    min_x, min_y = transformer_to_mercator.transform(min_lon, min_lat)
    max_x, max_y = transformer_to_mercator.transform(max_lon, max_lat)

    cell_size_x = (max_x - min_x) / cols
    cell_size_y = (max_y - min_y) / rows

    points = []
    values = []
    for i in range(rows):
        for j in range(cols):
            cell_center_x = min_x + (j + 0.5) * cell_size_x
            cell_center_y = max_y - (i + 0.5) * cell_size_y
            center_lon, center_lat = transformer_to_wgs84.transform(cell_center_x, cell_center_y)
            points.append(Point(center_lon, center_lat))
            values.append(grid[i, j])

    gdf = gpd.GeoDataFrame({'geometry': points, 'value': values}, crs=CRS.from_epsg(4326))
    return gdf


