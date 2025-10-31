from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import contextily as ctx

from ..geoprocessor.grid import grid_to_geodataframe
from ..utils.lc import get_land_cover_classes


def visualize_landcover_grid_on_basemap(landcover_grid, rectangle_vertices, meshsize, source='Standard', alpha=0.6, figsize=(12, 8), basemap='CartoDB light', show_edge=False, edge_color='black', edge_width=0.5):
    land_cover_classes = get_land_cover_classes(source)
    gdf = grid_to_geodataframe(landcover_grid, rectangle_vertices, meshsize)
    colors = [(r/255, g/255, b/255) for (r,g,b) in land_cover_classes.keys()]
    cmap = ListedColormap(colors)
    bounds = np.arange(len(colors) + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    gdf_web = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=figsize)
    gdf_web.plot(column='value', ax=ax, alpha=alpha, cmap=cmap, norm=norm, legend=True,
                 legend_kwds={'label': 'Land Cover Class', 'ticks': bounds[:-1] + 0.5, 'boundaries': bounds,
                              'format': lambda x, p: list(land_cover_classes.values())[int(x)]},
                 edgecolor=edge_color if show_edge else 'none', linewidth=edge_width if show_edge else 0)
    basemaps = {
        'CartoDB dark': ctx.providers.CartoDB.DarkMatter,
        'CartoDB light': ctx.providers.CartoDB.Positron,
        'CartoDB voyager': ctx.providers.CartoDB.Voyager,
        'CartoDB light no labels': ctx.providers.CartoDB.PositronNoLabels,
        'CartoDB dark no labels': ctx.providers.CartoDB.DarkMatterNoLabels,
    }
    ctx.add_basemap(ax, source=basemaps[basemap])
    ax.set_axis_off()
    plt.tight_layout(); plt.show()


def visualize_numerical_grid_on_basemap(grid, rectangle_vertices, meshsize, value_name="value", cmap='viridis', vmin=None, vmax=None,
                                        alpha=0.6, figsize=(12, 8), basemap='CartoDB light', show_edge=False, edge_color='black', edge_width=0.5):
    gdf = grid_to_geodataframe(grid, rectangle_vertices, meshsize)
    gdf_web = gdf.to_crs(epsg=3857)
    fig, ax = plt.subplots(figsize=figsize)
    gdf_web.plot(column='value', ax=ax, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax, legend=True,
                 legend_kwds={'label': value_name}, edgecolor=edge_color if show_edge else 'none', linewidth=edge_width if show_edge else 0)
    basemaps = {
        'CartoDB dark': ctx.providers.CartoDB.DarkMatter,
        'CartoDB light': ctx.providers.CartoDB.Positron,
        'CartoDB voyager': ctx.providers.CartoDB.Voyager,
        'CartoDB light no labels': ctx.providers.CartoDB.PositronNoLabels,
        'CartoDB dark no labels': ctx.providers.CartoDB.DarkMatterNoLabels,
    }
    ctx.add_basemap(ax, source=basemaps[basemap])
    ax.set_axis_off()
    plt.tight_layout(); plt.show()


def visualize_numerical_gdf_on_basemap(gdf, value_name="value", cmap='viridis', vmin=None, vmax=None,
                                       alpha=0.6, figsize=(12, 8), basemap='CartoDB light',
                                       show_edge=False, edge_color='black', edge_width=0.5, input_crs=None):
    if gdf.crs is None:
        if input_crs is not None:
            gdf = gdf.set_crs(input_crs, allow_override=True)
        else:
            try:
                minx, miny, maxx, maxy = gdf.total_bounds
                looks_like_lonlat = (-180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0)
            except Exception:
                looks_like_lonlat = False
            if looks_like_lonlat:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            else:
                raise ValueError("Input GeoDataFrame has no CRS. Provide 'input_crs' or set gdf.crs.")

    gdf_web = gdf.to_crs(epsg=3857) if str(gdf.crs) != 'EPSG:3857' else gdf
    fig, ax = plt.subplots(figsize=figsize)
    gdf_web.plot(column=value_name, ax=ax, alpha=alpha, cmap=cmap, vmin=vmin, vmax=vmax, legend=True,
                 legend_kwds={'label': value_name}, edgecolor=edge_color if show_edge else 'none', linewidth=edge_width if show_edge else 0)
    basemaps = {
        'CartoDB dark': ctx.providers.CartoDB.DarkMatter,
        'CartoDB light': ctx.providers.CartoDB.Positron,
        'CartoDB voyager': ctx.providers.CartoDB.Voyager,
        'CartoDB light no labels': ctx.providers.CartoDB.PositronNoLabels,
        'CartoDB dark no labels': ctx.providers.CartoDB.DarkMatterNoLabels,
    }
    ctx.add_basemap(ax, source=basemaps[basemap])
    ax.set_axis_off()
    plt.tight_layout(); plt.show()


def visualize_point_gdf_on_basemap(point_gdf, value_name='value', **kwargs):
    import contextily as ctx
    defaults = {
        'figsize': (12, 8),
        'colormap': 'viridis',
        'markersize': 20,
        'alpha': 0.7,
        'vmin': None,
        'vmax': None,
        'title': None,
        'basemap_style': ctx.providers.CartoDB.Positron,
        'zoom': 15
    }
    settings = {**defaults, **kwargs}
    fig, ax = plt.subplots(figsize=settings['figsize'])
    point_gdf_web = point_gdf.to_crs(epsg=3857)
    point_gdf_web.plot(column=value_name, ax=ax, cmap=settings['colormap'], markersize=settings['markersize'], alpha=settings['alpha'], vmin=settings['vmin'], vmax=settings['vmax'], legend=True, legend_kwds={'label': value_name})
    ctx.add_basemap(ax, source=settings['basemap_style'], zoom=settings['zoom'])
    if settings['title']:
        plt.title(settings['title'])
    ax.set_axis_off(); plt.tight_layout(); plt.show()


def visualize_land_cover_grid(grid, mesh_size, color_map, land_cover_classes):
    all_classes = list(land_cover_classes.values())
    unique_classes = list(dict.fromkeys(all_classes))
    colors = [color_map[cls] for cls in unique_classes]
    cmap = mcolors.ListedColormap(colors)
    bounds = np.arange(len(unique_classes) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    class_to_num = {cls: i for i, cls in enumerate(unique_classes)}
    numeric_grid = np.vectorize(class_to_num.get)(grid)
    plt.figure(figsize=(10, 10))
    im = plt.imshow(numeric_grid, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = plt.colorbar(im, ticks=bounds[:-1] + 0.5)
    cbar.set_ticklabels(unique_classes)
    plt.title(f'Land Use/Land Cover Grid (Mesh Size: {mesh_size}m)')
    plt.xlabel('Grid Cells (X)')
    plt.ylabel('Grid Cells (Y)')
    plt.show()


def visualize_numerical_grid(grid, mesh_size, title, cmap='viridis', label='Value', vmin=None, vmax=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label=label)
    plt.title(f'{title} (Mesh Size: {mesh_size}m)')
    plt.xlabel('Grid Cells (X)')
    plt.ylabel('Grid Cells (Y)')
    plt.show()


