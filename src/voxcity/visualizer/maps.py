from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import contextily as ctx
from shapely.geometry import Polygon
from pyproj import CRS

from ..geoprocessor.grid import (
    calculate_grid_size,
    create_coordinate_mesh,
)
from ..geoprocessor.utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    setup_transformer,
    transform_coords,
)
from ..utils.lc import get_land_cover_classes


def plot_grid(grid, origin, adjusted_meshsize, u_vec, v_vec, transformer, vertices, data_type, vmin=None, vmax=None, color_map=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light', **kwargs):
    fig, ax = plt.subplots(figsize=(12, 12))

    if data_type == 'land_cover':
        land_cover_classes = kwargs.get('land_cover_classes')
        colors = [mcolors.to_rgb(f'#{r:02x}{g:02x}{b:02x}') for r, g, b in land_cover_classes.keys()]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(range(len(land_cover_classes)+1), cmap.N)
    elif data_type == 'building_height':
        masked_grid = np.ma.masked_array(grid, mask=(np.isnan(grid) | (grid == 0)))
        cmap = plt.cm.viridis
        if vmin is None:
            vmin = np.nanmin(masked_grid[masked_grid > 0])
        if vmax is None:
            vmax = np.nanmax(masked_grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    elif data_type == 'dem':
        cmap = plt.cm.terrain
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    elif data_type == 'canopy_height':
        cmap = plt.cm.Greens
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    elif data_type in ('green_view_index', 'sky_view_index'):
        cmap = plt.cm.get_cmap('BuPu_r').copy() if data_type == 'sky_view_index' else plt.cm.Greens
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        cmap = plt.cm.viridis
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    if color_map:
        cmap = sns.color_palette(color_map, as_cmap=True).copy()

    grid = grid.T

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cell = create_cell_polygon(origin, j, i, adjusted_meshsize, u_vec, v_vec)  # type: ignore[name-defined]
            x, y = cell.exterior.xy
            x, y = zip(*[transformer.transform(lon, lat) for lat, lon in zip(x, y)])
            value = grid[i, j]
            if data_type == 'building_height':
                if np.isnan(value):
                    ax.fill(x, y, alpha=alpha, fc='gray', ec='black' if edge else None, linewidth=0.1)
                elif value == 0:
                    if edge:
                        ax.plot(x, y, color='black', linewidth=0.1)
                elif value > 0:
                    ax.fill(x, y, alpha=alpha, fc=cmap(norm(value)), ec='black' if edge else None, linewidth=0.1)
            elif data_type == 'canopy_height':
                color = cmap(norm(value))
                if value == 0:
                    if edge:
                        ax.plot(x, y, color='black', linewidth=0.1)
                else:
                    ax.fill(x, y, alpha=alpha, fc=color, ec='black' if edge else None, linewidth=0.1)
            elif 'view' in data_type:
                if np.isnan(value):
                    if edge:
                        ax.plot(x, y, color='black', linewidth=0.1)
                elif value >= 0:
                    ax.fill(x, y, alpha=alpha, fc=cmap(norm(value)), ec='black' if edge else None, linewidth=0.1)
            else:
                color = cmap(norm(value))
                ax.fill(x, y, alpha=alpha, fc=color, ec='black' if edge else None, linewidth=0.1)

    crs_epsg_3857 = CRS.from_epsg(3857)
    basemaps = {
      'CartoDB dark': ctx.providers.CartoDB.DarkMatter,
      'CartoDB light': ctx.providers.CartoDB.Positron,
      'CartoDB voyager': ctx.providers.CartoDB.Voyager,
      'CartoDB light no labels': ctx.providers.CartoDB.PositronNoLabels,
      'CartoDB dark no labels': ctx.providers.CartoDB.DarkMatterNoLabels,
    }
    ctx.add_basemap(ax, crs=crs_epsg_3857, source=basemaps[basemap])

    all_coords = np.array(vertices)
    x, y = zip(*[transformer.transform(lon, lat) for lat, lon in all_coords])
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    if x_min != x_max and y_min != y_max and buf != 0:
        dist_x = x_max - x_min
        dist_y = y_max - y_min
        ax.set_xlim(x_min - buf * dist_x, x_max + buf * dist_x)
        ax.set_ylim(y_min - buf * dist_y, y_max + buf * dist_y)
    else:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.axis('off')
    plt.tight_layout()
    plt.show()


def visualize_land_cover_grid_on_map(grid, rectangle_vertices, meshsize, source='Urbanwatch', vmin=None, vmax=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light'):
    geod = initialize_geod()
    land_cover_classes = get_land_cover_classes(source)
    vertex_0 = rectangle_vertices[0]; vertex_1 = rectangle_vertices[1]; vertex_3 = rectangle_vertices[3]
    dist_side_1 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_1[1], vertex_1[0])
    dist_side_2 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_3[1], vertex_3[0])
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)
    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)
    transformer = setup_transformer(CRS.from_epsg(4326), CRS.from_epsg(3857))
    plot_grid(grid, origin, adjusted_meshsize, u_vec, v_vec, transformer, rectangle_vertices, 'land_cover', alpha=alpha, buf=buf, edge=edge, basemap=basemap, land_cover_classes=land_cover_classes)


def visualize_building_height_grid_on_map(building_height_grid, filtered_buildings, rectangle_vertices, meshsize, vmin=None, vmax=None, color_map=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light'):
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]
    dist_side_1 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_1[1], vertex_1[0])
    dist_side_2 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_3[1], vertex_3[0])
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)
    origin = np.array(rectangle_vertices[0])
    _, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)
    transformer = setup_transformer(CRS.from_epsg(4326), CRS.from_epsg(3857))
    plot_grid(building_height_grid, origin, adjusted_meshsize, u_vec, v_vec, transformer,
              rectangle_vertices, 'building_height', vmin=vmin, vmax=vmax, color_map=color_map, alpha=alpha, buf=buf, edge=edge, basemap=basemap, buildings=filtered_buildings)


def visualize_numerical_grid_on_map(canopy_height_grid, rectangle_vertices, meshsize, type, vmin=None, vmax=None, color_map=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light'):
    geod = initialize_geod()
    vertex_0, vertex_1, vertex_3 = rectangle_vertices[0], rectangle_vertices[1], rectangle_vertices[3]
    dist_side_1 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_1[1], vertex_1[0])
    dist_side_2 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_3[1], vertex_3[0])
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)
    origin = np.array(rectangle_vertices[0])
    _, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)
    transformer = setup_transformer(CRS.from_epsg(4326), CRS.from_epsg(3857))
    plot_grid(canopy_height_grid, origin, adjusted_meshsize, u_vec, v_vec, transformer,
              rectangle_vertices, type, vmin=vmin, vmax=vmax, color_map=color_map, alpha=alpha, buf=buf, edge=edge, basemap=basemap)





