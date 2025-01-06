import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
import networkx as nx
import osmnx as ox

from .grid import grid_to_geodataframe

def calculate_edge_values(G, gdf, value_col='value'):
    """
    Calculate average values for graph edges based on intersection with polygons.
    
    Parameters:
    -----------
    G : NetworkX Graph
        Input graph with edges to analyze
    gdf : GeoDataFrame
        Grid containing polygons with values
    value_col : str, default 'value'
        Name of the column containing values in the grid
        
    Returns:
    --------
    dict
        Dictionary with edge identifiers (u,v,k) as keys and average values as values
    """
    edge_values = {}
    for u, v, k, data in G.edges(data=True, keys=True):
        if 'geometry' in data:
            edge_line = data['geometry']
        else:
            start_node = G.nodes[u]
            end_node = G.nodes[v]
            edge_line = LineString([(start_node['x'], start_node['y']),
                                  (end_node['x'], end_node['y'])])
        
        intersecting_polys = gdf[gdf.geometry.intersects(edge_line)]
        
        if len(intersecting_polys) > 0:
            total_length = 0
            weighted_sum = 0
            
            for idx, poly in intersecting_polys.iterrows():
                if pd.isna(poly[value_col]):
                    continue
                
                intersection = edge_line.intersection(poly.geometry)
                if not intersection.is_empty:
                    length = intersection.length
                    total_length += length
                    weighted_sum += length * poly[value_col]
            
            if total_length > 0:
                avg_value = weighted_sum / total_length
                edge_values[(u, v, k)] = avg_value
            else:
                edge_values[(u, v, k)] = np.nan
        else:
            edge_values[(u, v, k)] = np.nan
    
    return edge_values

def get_network_values(grid, rectangle_vertices, meshsize, value_name='value', **kwargs):
    """
    Analyze and visualize network values based on grid intersections.
    
    Parameters:
    -----------
    grid : GeoDataFrame
        Input grid with geometries and values
    rectangle_vertices : list
        List of coordinates defining the bounding box vertices
    meshsize : float
        Size of the mesh grid
    value_name : str, default 'value'
        Name of the column containing values in the grid
    **kwargs : dict
        Optional arguments including:
        - network_type : str, default 'walk'
            Type of network to download ('walk', 'drive', 'all', etc.)
        - vis_graph : bool, default True
            Whether to visualize the graph
        - colormap : str, default 'viridis'
            Matplotlib colormap name for visualization
        - vmin : float, optional
            Minimum value for color scaling
        - vmax : float, optional
            Maximum value for color scaling
        - edge_width : float, default 1
            Width of the edges in visualization
        - fig_size : tuple, default (15,15)
            Figure size for visualization
        - zoom : int, default 16
            Zoom level for the basemap
        - basemap_style : ctx.providers, default CartoDB.Positron
            Contextily basemap provider
        - save_path : str, optional
            Path to save the output GeoPackage
        
    Returns:
    --------
    tuple : (NetworkX Graph, GeoDataFrame)
        Returns the processed graph and edge GeoDataFrame
    """
    # Set default values for optional arguments
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
    
    # Update defaults with provided kwargs
    settings = defaults.copy()
    settings.update(kwargs)

    grid_gdf = grid_to_geodataframe(grid, rectangle_vertices, meshsize)
    
    # Extract bounding box coordinates
    north, south = rectangle_vertices[1][1], rectangle_vertices[0][1]
    east, west   = rectangle_vertices[2][0], rectangle_vertices[0][0]
    bbox = (west, south, east, north)
    
    # Download the road network
    G = ox.graph.graph_from_bbox(bbox=bbox, network_type=settings['network_type'], simplify=True)
    
    # Calculate edge values using the separate function
    edge_values = calculate_edge_values(G, grid_gdf, "value")
    
    # Add values to the graph
    nx.set_edge_attributes(G, edge_values, value_name)
    
    # Create GeoDataFrame from edges
    edges_with_values = []
    for u, v, k, data in G.edges(data=True, keys=True):
        if 'geometry' in data:
            edge_line = data['geometry']
        else:
            start_node = G.nodes[u]
            end_node = G.nodes[v]
            edge_line = LineString([(start_node['x'], start_node['y']),
                                  (end_node['x'], end_node['y'])])
        
        edges_with_values.append({
            'geometry': edge_line,
            value_name: data.get(value_name, np.nan),
            'u': u,
            'v': v,
            'key': k
        })
    
    edge_gdf = gpd.GeoDataFrame(edges_with_values)
    
    # Set CRS and save if requested
    if edge_gdf.crs is None:
        edge_gdf.set_crs(epsg=4326, inplace=True)
    
    if settings['save_path']:
        edge_gdf.to_file(settings['save_path'], driver="GPKG")
    
    # Visualize if requested
    if settings['vis_graph']:
        edge_gdf_web = edge_gdf.to_crs(epsg=3857)
        
        fig, ax = plt.subplots(figsize=settings['fig_size'])
        
        plot = edge_gdf_web.plot(column=value_name,
                                ax=ax,
                                cmap=settings['colormap'],
                                legend=True,
                                vmin=settings['vmin'],
                                vmax=settings['vmax'],
                                linewidth=settings['edge_width'],
                                legend_kwds={'label': value_name})
        
        ctx.add_basemap(ax,
                       source=settings['basemap_style'],
                       zoom=settings['zoom'])
        
        ax.set_axis_off()
        # plt.title(f'Network {value_name} Analysis', pad=20)
        plt.show()
    
    return G, edge_gdf