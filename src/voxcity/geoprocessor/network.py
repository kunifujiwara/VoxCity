import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, Polygon
import shapely.ops as ops
import networkx as nx
import osmnx as ox
import os
import shapely
from shapely.geometry import Point
from shapely.ops import transform
import pyproj
from pyproj import Transformer
from joblib import Parallel, delayed

from .grid import grid_to_geodataframe

def vectorized_edge_values(G, polygons_gdf, value_col='value'):
    """
    Compute average polygon values along each edge by:
    1) Building an Edge GeoDataFrame in linestring form
    2) Using gpd.overlay or sjoin to get their intersection with polygons
    3) Computing length-weighted average
    """
    # 1) Build edge GeoDataFrame (EPSG:4326)
    records = []
    for i, (u, v, k, data) in enumerate(G.edges(keys=True, data=True)):
        if 'geometry' in data:
            edge_geom = data['geometry']
        else:
            start_node = G.nodes[u]
            end_node = G.nodes[v]
            edge_geom = LineString([(start_node['x'], start_node['y']),
                                    (end_node['x'], end_node['y'])])
        records.append({
            'edge_id': i,  # unique ID for grouping
            'u': u,
            'v': v,
            'k': k,
            'geometry': edge_geom
        })

    edges_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    if polygons_gdf.crs != edges_gdf.crs:
        polygons_gdf = polygons_gdf.to_crs(edges_gdf.crs)

    # 2) Use a projected CRS for length calculations
    edges_3857 = edges_gdf.to_crs(epsg=3857)
    polys_3857 = polygons_gdf.to_crs(epsg=3857)

    # 3) Intersection: lines vs polygons -> lines clipped to polygons
    #    gpd.overlay with how='intersection' can yield partial lines
    intersected = gpd.overlay(edges_3857, polys_3857, how='intersection')

    # Now each row is a geometry representing the intersection segment,
    # with columns from edges + polygons.
    # For lines, 'intersection' yields the line portion inside each polygon.
    # We'll compute the length, then do a length-weighted average of value_col.

    intersected['seg_length'] = intersected.geometry.length
    # Weighted contribution = seg_length * polygon_value
    intersected['weighted_val'] = intersected['seg_length'] * intersected[value_col]

    # 4) Group by edge_id
    grouped = intersected.groupby('edge_id')
    results = grouped.apply(
        lambda df: df['weighted_val'].sum() / df['seg_length'].sum()
        if df['seg_length'].sum() > 0 else np.nan
    )
    # results is a Series with index=edge_id

    # 5) Map results back to edges
    edge_values = {}
    for edge_id, val in results.items():
        rec = edges_gdf.iloc[edge_id]
        edge_values[(rec['u'], rec['v'], rec['k'])] = val

    return edge_values

def get_network_values(
    grid,
    rectangle_vertices,
    meshsize,
    value_name='value',
    **kwargs
):
    defaults = {
        'network_type': 'walk',
        'vis_graph': True,
        'colormap': 'viridis',
        'vmin': None,
        'vmax': None,
        'edge_width': 1,
        'fig_size': (15,15),
        'zoom': 16,
        'basemap_style': ctx.providers.CartoDB.Positron,
        'save_path': None
    }
    settings = {**defaults, **kwargs}

    # Build polygons GDF if needed
    polygons_gdf = (grid if isinstance(grid, gpd.GeoDataFrame) 
                    else grid_to_geodataframe(grid, rectangle_vertices, meshsize))
    if polygons_gdf.crs is None:
        polygons_gdf.set_crs(epsg=4326, inplace=True)

    # BBox
    north, south = rectangle_vertices[1][1], rectangle_vertices[0][1]
    east,  west  = rectangle_vertices[2][0], rectangle_vertices[0][0]
    bbox = (west, south, east, north)

    # Download OSMnx network
    G = ox.graph.graph_from_bbox(
        bbox=bbox,
        network_type=settings['network_type'],
        simplify=True
    )

    # Compute edge values with the vectorized function
    edge_values = vectorized_edge_values(G, polygons_gdf, value_col="value")
    nx.set_edge_attributes(G, edge_values, name=value_name)

    # Build edge GDF
    edges_with_values = []
    for u, v, k, data in G.edges(data=True, keys=True):
        if 'geometry' in data:
            geom = data['geometry']
        else:
            start_node = G.nodes[u]
            end_node = G.nodes[v]
            geom = LineString([(start_node['x'], start_node['y']),
                               (end_node['x'], end_node['y'])])

        val = data.get(value_name, np.nan)
        edges_with_values.append({
            'u': u, 'v': v, 'key': k,
            'geometry': geom,
            value_name: val
        })

    edge_gdf = gpd.GeoDataFrame(edges_with_values, crs="EPSG:4326")

    # Save
    if settings['save_path']:
        edge_gdf.to_file(settings['save_path'], driver="GPKG")

    if settings['vis_graph']:
        edge_gdf_web = edge_gdf.to_crs(epsg=3857)
        fig, ax = plt.subplots(figsize=settings['fig_size'])
        edge_gdf_web.plot(
            column=value_name,
            ax=ax,
            cmap=settings['colormap'],
            legend=True,
            vmin=settings['vmin'],
            vmax=settings['vmax'],
            linewidth=settings['edge_width'],
            legend_kwds={'label': value_name, 'shrink': 0.5}
        )
        ctx.add_basemap(ax, source=settings['basemap_style'], zoom=settings['zoom'])
        ax.set_axis_off()
        plt.show()

    return G, edge_gdf

# -------------------------------------------------------------------
# 1) Functions for interpolation, parallelization, and slope
# -------------------------------------------------------------------

def interpolate_points_along_line(line, interval):
    """
    Interpolate points along a single LineString at a given interval (in meters).
    If the line is shorter than `interval`, only start/end points are returned.

    Parameters
    ----------
    line : shapely.geometry.LineString
        Edge geometry in EPSG:4326 (lon/lat).
    interval : float
        Distance in meters between interpolated points.

    Returns
    -------
    list of shapely.geometry.Point
        Points in EPSG:4326 along the line.
    """
    if line.is_empty:
        return []

    # Transformers for metric distance calculations
    project = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    project_rev = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    # Project line to Web Mercator
    line_merc = shapely.ops.transform(project, line)
    length_m = line_merc.length
    if length_m == 0:
        return [Point(line.coords[0])]

    # If line is shorter than interval, just start & end
    if length_m < interval:
        return [Point(line.coords[0]), Point(line.coords[-1])]

    # Otherwise, create distances
    num_points = int(length_m // interval)
    dists = [i * interval for i in range(num_points + 1)]
    # Ensure end
    if dists[-1] < length_m:
        dists.append(length_m)

    # Interpolate
    points_merc = [line_merc.interpolate(d) for d in dists]
    # Reproject back
    return [shapely.ops.transform(project_rev, pt) for pt in points_merc]


def gather_interpolation_points(G, interval=10.0, n_jobs=1):
    """
    Gather all interpolation points for each edge in the graph into a single GeoDataFrame.
    Can be parallelized with `n_jobs`.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        OSMnx graph with 'geometry' attributes or x,y coordinates in the nodes.
    interval : float, default=10.0
        Interpolation distance interval in meters.
    n_jobs : int, default=1
        Number of parallel jobs (1 => no parallelization).

    Returns
    -------
    gpd.GeoDataFrame
        Columns: edge_id, index_in_edge, geometry (EPSG:4326).
    """
    edges = list(G.edges(keys=True, data=True))

    def process_edge(u, v, k, data, idx):
        if 'geometry' in data:
            line = data['geometry']
        else:
            # If no geometry, build from node coords
            start_node = G.nodes[u]
            end_node = G.nodes[v]
            line = LineString([(start_node['x'], start_node['y']),
                               (end_node['x'], end_node['y'])])

        pts = interpolate_points_along_line(line, interval)
        df = pd.DataFrame({
            'edge_id': [idx]*len(pts),
            'index_in_edge': np.arange(len(pts)),
            'geometry': pts
        })
        return df

    # Parallel interpolation
    results = Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(process_edge)(u, v, k, data, i)
        for i, (u, v, k, data) in enumerate(edges)
    )

    all_points_df = pd.concat(results, ignore_index=True)
    points_gdf = gpd.GeoDataFrame(all_points_df, geometry='geometry', crs="EPSG:4326")
    return points_gdf


def fetch_elevations_for_points(points_gdf_3857, dem_gdf_3857, elevation_col='value'):
    """
    Do a spatial join (nearest) in a projected CRS (EPSG:3857) to fetch DEM elevations.

    Parameters
    ----------
    points_gdf_3857 : gpd.GeoDataFrame
        Interpolation points in EPSG:3857.
    dem_gdf_3857 : gpd.GeoDataFrame
        DEM polygons in EPSG:3857, must have `elevation_col`.
    elevation_col : str, default='value'
        Column with elevation values in dem_gdf_3857.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of points_gdf_3857 with new column 'elevation'.
    """
    joined = gpd.sjoin_nearest(
        points_gdf_3857, 
        dem_gdf_3857[[elevation_col, 'geometry']].copy(),
        how='left',
        distance_col='dist_to_poly'
    )
    joined.rename(columns={elevation_col: 'elevation'}, inplace=True)
    return joined


def compute_slope_for_group(df):
    """
    Given a subset of points for a single edge, compute average slope between
    consecutive points, using columns: geometry, elevation, index_in_edge.

    Note: We assume df is already in EPSG:3857 for direct distance calculations.
    """
    # Sort by position along the edge
    df = df.sort_values("index_in_edge")

    # Coordinates
    xs = df.geometry.x.to_numpy()
    ys = df.geometry.y.to_numpy()
    elevs = df["elevation"].to_numpy()

    # Differences
    dx = np.diff(xs)
    dy = np.diff(ys)
    horizontal_dist = np.sqrt(dx**2 + dy**2)
    elev_diff = np.diff(elevs)

    # Slope in %
    valid_mask = horizontal_dist > 0
    slopes = (np.abs(elev_diff[valid_mask]) / horizontal_dist[valid_mask]) * 100

    if len(slopes) == 0:
        return np.nan
    return slopes.mean()


def calculate_edge_slopes_from_join(joined_points_gdf, n_edges):
    """
    Calculate average slopes for each edge by grouping joined points.

    Parameters
    ----------
    joined_points_gdf : gpd.GeoDataFrame
        Must have columns: edge_id, index_in_edge, elevation, geometry (EPSG:3857).
    n_edges : int
        Number of edges from the graph.

    Returns
    -------
    dict
        edge_id -> average slope (in %).
    """
    # We'll group by edge_id, ignoring the group columns in apply (pandas >= 2.1).
    # If your pandas version < 2.1, just do a column subset after groupby.
    # E.g. .groupby("edge_id", group_keys=False)[["geometry","elevation","index_in_edge"]]...
    grouped = joined_points_gdf.groupby("edge_id", group_keys=False)
    results = grouped[["geometry", "elevation", "index_in_edge"]].apply(compute_slope_for_group)

    # Convert series -> dict
    slope_dict = results.to_dict()

    # Fill any missing edge IDs with NaN
    for i in range(n_edges):
        if i not in slope_dict:
            slope_dict[i] = np.nan

    return slope_dict

# -------------------------------------------------------------------
# 2) Main function to analyze network slopes
# -------------------------------------------------------------------

def analyze_network_slopes(
    dem_grid,
    meshsize,
    value_name='slope',
    interval=10.0,
    n_jobs=1,
    **kwargs
):
    """
    Analyze and visualize network slopes based on DEM data, using vectorized + parallel methods.

    Parameters
    ----------
    dem_grid : array-like
        DEM grid data.
    meshsize : float
        Mesh grid size.
    value_name : str, default='slope'
        Column name for slopes assigned to each edge.
    interval : float, default=10.0
        Interpolation distance in meters.
    n_jobs : int, default=1
        Parallelization for edge interpolation (1 => sequential).
    **kwargs : dict
        Additional parameters:
          - rectangle_vertices : list of (x, y) in EPSG:4326
          - network_type : str, default='walk'
          - vis_graph : bool, default=True
          - colormap, vmin, vmax, edge_width, fig_size, zoom, basemap_style, alpha
          - output_directory, output_file_name
    """
    defaults = {
        'rectangle_vertices': None,
        'network_type': 'walk',
        'vis_graph': True,
        'colormap': 'viridis',
        'vmin': None,
        'vmax': None,
        'edge_width': 1,
        'fig_size': (15, 15),
        'zoom': 16,
        'basemap_style': ctx.providers.CartoDB.Positron,
        'output_directory': None,
        'output_file_name': 'network_slopes',
        'alpha': 1.0
    }
    settings = {**defaults, **kwargs}

    # Validate bounding box
    if settings['rectangle_vertices'] is None:
        raise ValueError("Must supply 'rectangle_vertices' in kwargs.")

    # 1) Build DEM GeoDataFrame in EPSG:4326
    dem_gdf = grid_to_geodataframe(dem_grid, settings['rectangle_vertices'], meshsize)
    if dem_gdf.crs is None:
        dem_gdf.set_crs(epsg=4326, inplace=True)

    # 2) Download bounding box from rectangle_vertices
    north, south = settings['rectangle_vertices'][1][1], settings['rectangle_vertices'][0][1]
    east, west = settings['rectangle_vertices'][2][0], settings['rectangle_vertices'][0][0]
    bbox = (west, south, east, north)

    G = ox.graph.graph_from_bbox(
        bbox=bbox,
        network_type=settings['network_type'],
        simplify=True
    )

    # 3) Interpolate points along edges (EPSG:4326)
    points_gdf_4326 = gather_interpolation_points(G, interval=interval, n_jobs=n_jobs)
    
    # 4) Reproject DEM + Points to EPSG:3857 for correct distance operations
    dem_gdf_3857 = dem_gdf.to_crs(epsg=3857)
    points_gdf_3857 = points_gdf_4326.to_crs(epsg=3857)

    # 5) Perform spatial join to get elevations
    joined_points_3857 = fetch_elevations_for_points(points_gdf_3857, dem_gdf_3857, elevation_col='value')

    # 6) Compute slopes for each edge
    n_edges = len(list(G.edges(keys=True)))
    slope_dict = calculate_edge_slopes_from_join(joined_points_3857, n_edges)

    # 7) Assign slopes back to G
    edges = list(G.edges(keys=True, data=True))
    edge_slopes = {}
    for i, (u, v, k, data) in enumerate(edges):
        edge_slopes[(u, v, k)] = slope_dict.get(i, np.nan)
    nx.set_edge_attributes(G, edge_slopes, name=value_name)

    # 8) Build an edge GeoDataFrame in EPSG:4326
    edges_with_values = []
    for (u, v, k, data), edge_id in zip(edges, range(len(edges))):
        if 'geometry' in data:
            geom = data['geometry']
        else:
            start_node = G.nodes[u]
            end_node = G.nodes[v]
            geom = LineString([(start_node['x'], start_node['y']),
                               (end_node['x'], end_node['y'])])

        edges_with_values.append({
            'u': u,
            'v': v,
            'key': k,
            'geometry': geom,
            value_name: slope_dict.get(edge_id, np.nan)
        })

    edge_gdf = gpd.GeoDataFrame(edges_with_values, crs="EPSG:4326")

    # 9) Save output if requested
    if settings['output_directory']:
        os.makedirs(settings['output_directory'], exist_ok=True)
        out_path = os.path.join(
            settings['output_directory'],
            f"{settings['output_file_name']}.gpkg"
        )
        edge_gdf.to_file(out_path, driver="GPKG")

    # 10) Visualization
    if settings['vis_graph']:
        # Create a Polygon from the rectangle vertices
        rectangle_polygon = Polygon(rectangle_vertices)

        # Convert the rectangle polygon to the same CRS as edge_gdf_web
        rectangle_gdf = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[rectangle_polygon])
        rectangle_gdf_web = rectangle_gdf.to_crs(epsg=3857)

        # Get the bounding box of the rectangle
        minx, miny, maxx, maxy = rectangle_gdf_web.total_bounds

        # Plot the edges
        edge_gdf_web = edge_gdf.to_crs(epsg=3857)
        fig, ax = plt.subplots(figsize=settings['fig_size'])
        edge_gdf_web.plot(
            column=value_name, 
            ax=ax, 
            cmap=settings['colormap'], 
            legend=True, 
            vmin=settings['vmin'], 
            vmax=settings['vmax'],
            linewidth=settings['edge_width'],
            alpha=settings['alpha'],
            legend_kwds={'label': f"{value_name} (%)"}
        )

        # Add basemap with the same extent as the rectangle
        ctx.add_basemap(
            ax,
            source=settings['basemap_style'],
            zoom=settings['zoom'],
            bounds=(minx, miny, maxx, maxy)  # Explicitly set the bounds of the basemap
        )

        # Set the plot limits to the bounding box of the rectangle
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

        # Turn off the axis
        ax.set_axis_off()

        # Add title
        plt.title(f'Network {value_name} Analysis', pad=20)

        # Show the plot
        plt.show()
        # edge_gdf_web = edge_gdf.to_crs(epsg=3857)
        # fig, ax = plt.subplots(figsize=settings['fig_size'])
        # edge_gdf_web.plot(
        #     column=value_name, 
        #     ax=ax, 
        #     cmap=settings['colormap'], 
        #     legend=True, 
        #     vmin=settings['vmin'], 
        #     vmax=settings['vmax'],
        #     linewidth=settings['edge_width'],
        #     alpha=settings['alpha'],
        #     legend_kwds={'label': f"{value_name} (%)"}
        # )
        # ctx.add_basemap(ax, source=settings['basemap_style'], zoom=settings['zoom'])
        # ax.set_axis_off()
        # plt.title(f'Network {value_name} Analysis', pad=20)
        # plt.show()

    return G, edge_gdf