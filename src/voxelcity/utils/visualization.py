import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib.colors as mcolors
import contextily as ctx
from shapely.geometry import Polygon
import plotly.graph_objects as go
from tqdm import tqdm
import pyproj
# import rasterio
from pyproj import CRS
# from shapely.geometry import box
import seaborn as sns

from .lc import get_land_cover_classes
# from ..geo.geojson import filter_buildings
from ..geo.grid import (
    calculate_grid_size,
    create_coordinate_mesh,
    create_cell_polygon
)

from ..geo.utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    setup_transformer,
    transform_coords,
)

# default_voxel_color_map = {
#     -3: [180, 187, 216],  #(lightgray) 'Building',
#     -2: [48, 176, 158],   #(forestgreen) 'Tree',
#     -1: [188, 143, 143],  #(saddle brown) 'Underground',
#     #0: 'Air (Void)',
#     1: [239, 228, 176],   #'Bareland (ground surface)',
#     2: [183, 226, 150],   #(greenyellow) 'Rangeland (ground surface)',
#     3: [108, 119, 129],   #(darkgray) 'Developed space (ground surface)',
#     4: [59, 62, 87],      #(dimgray) 'Road (ground surface)',
#     5: [183, 226, 150],   #(greenyellow) 'Tree (ground surface)',
#     6: [80, 142, 204],    #(blue) 'Water (ground surface)',
#     7: [150, 226, 180],   #(lightgreen) 'Agriculture land (ground surface)',
#     8: [150, 166, 190]    #(lightgray) 'Building (ground surface)'
# }

# Define the color map (using the provided color map)
default_voxel_color_map = {
    -99: [0, 0, 0],  # void,
    -3: [180, 187, 216],  # (lightgray) 'Building',
    -2: [78, 99, 63],   # (forestgreen) 'Tree',
    -1: [188, 143, 143],  # (saddle brown) 'Underground',
    #0: 'Air (Void)',
    1: [239, 228, 176],   # 'Bareland (ground surface)',
    2: [123, 130, 59],   # (greenyellow) 'Rangeland (ground surface)',
    3: [108, 119, 129],   # (darkgray) 'Developed space (ground surface)',
    4: [59, 62, 87],      # (dimgray) 'Road (ground surface)',
    5: [116, 150, 66],   # (greenyellow) 'Tree (ground surface)',
    6: [44, 66, 133],    # (blue) 'Water (ground surface)',
    7: [112, 120, 56],   # (lightgreen) 'Agriculture land (ground surface)',
    8: [150, 166, 190],    # (lightgray) 'Building (ground surface)'
}

def visualize_3d_voxel(voxel_grid, color_map = default_voxel_color_map, voxel_size=2.0, save_path=None):
    print("\tVisualizing 3D voxel data")
    # Create a figure and a 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    print("\tProcessing voxels...")
    filled_voxels = voxel_grid != 0
    colors = np.zeros(voxel_grid.shape + (4,))  # RGBA

    for val in range(-3, 13):  # Updated range to include -3 and -2
        mask = voxel_grid == val
        if val in color_map:
            rgb = [x/255 for x in color_map[val]]  # Normalize RGB values to [0, 1]
            # alpha = 0.7 if ((val == -1) or (val == -2)) else 0.9  # More transparent for underground and below
            alpha = 0.0 if (val == -99) else 1
            # alpha = 1
            colors[mask] = rgb + [alpha]
        else:
            colors[mask] = [0, 0, 0, 0.9]  # Default color if not in color_map

    with tqdm(total=np.prod(voxel_grid.shape)) as pbar:
        ax.voxels(filled_voxels, facecolors=colors, edgecolors=None)
        pbar.update(np.prod(voxel_grid.shape))

    # print("Finalizing plot...")
    # Set labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z (meters)')
    # ax.set_title('3D Voxel Visualization')

    # Adjust z-axis ticks to show every 10 cells or less
    z_max = voxel_grid.shape[2]
    if z_max <= 10:
        z_ticks = range(0, z_max + 1)
    else:
        z_ticks = range(0, z_max + 1, 10)
    # Remove axes
    ax.axis('off')
    # ax.set_zticks(z_ticks)
    # ax.set_zticklabels([f"{z * voxel_size:.1f}" for z in z_ticks])

    # Set aspect ratio to be equal
    max_range = np.array([voxel_grid.shape[0], voxel_grid.shape[1], voxel_grid.shape[2]]).max()
    ax.set_box_aspect((voxel_grid.shape[0]/max_range, voxel_grid.shape[1]/max_range, voxel_grid.shape[2]/max_range))

    ax.set_zlim(bottom=0)
    ax.set_zlim(top=150)

    # print("Visualization complete. Displaying plot...")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visualize_3d_voxel_plotly(voxel_grid, color_map = default_voxel_color_map, voxel_size=2.0):
    print("Preparing visualization...")

    print("Processing voxels...")
    x, y, z = [], [], []
    i, j, k = [], [], []
    colors = []
    edge_x, edge_y, edge_z = [], [], []
    vertex_index = 0

    # Define cube faces
    cube_i = [7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2]
    cube_j = [3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3]
    cube_k = [0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6]

    with tqdm(total=np.prod(voxel_grid.shape)) as pbar:
        for xi in range(voxel_grid.shape[0]):
            for yi in range(voxel_grid.shape[1]):
                for zi in range(voxel_grid.shape[2]):
                    if voxel_grid[xi, yi, zi] != 0:
                        # Add cube vertices
                        cube_vertices = [
                            [xi, yi, zi], [xi+1, yi, zi], [xi+1, yi+1, zi], [xi, yi+1, zi],
                            [xi, yi, zi+1], [xi+1, yi, zi+1], [xi+1, yi+1, zi+1], [xi, yi+1, zi+1]
                        ]
                        x.extend([v[0] for v in cube_vertices])
                        y.extend([v[1] for v in cube_vertices])
                        z.extend([v[2] for v in cube_vertices])

                        # Add cube faces
                        i.extend([x + vertex_index for x in cube_i])
                        j.extend([x + vertex_index for x in cube_j])
                        k.extend([x + vertex_index for x in cube_k])

                        # Add color
                        color = color_map.get(voxel_grid[xi, yi, zi], [0, 0, 0])
                        colors.extend([color] * 8)

                        # Add edges
                        edges = [
                            (0,1), (1,2), (2,3), (3,0),  # Bottom face
                            (4,5), (5,6), (6,7), (7,4),  # Top face
                            (0,4), (1,5), (2,6), (3,7)   # Vertical edges
                        ]
                        for start, end in edges:
                            edge_x.extend([cube_vertices[start][0], cube_vertices[end][0], None])
                            edge_y.extend([cube_vertices[start][1], cube_vertices[end][1], None])
                            edge_z.extend([cube_vertices[start][2], cube_vertices[end][2], None])

                        vertex_index += 8
                    pbar.update(1)

    print("Creating Plotly figure...")
    mesh = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        vertexcolor=colors,
        opacity=1,
        flatshading=True,
        name='Voxel Grid'
    )

    # Add lighting to the mesh
    mesh.update(
        lighting=dict(ambient=0.7,
                      diffuse=1,
                      fresnel=0.1,
                      specular=1,
                      roughness=0.05,
                      facenormalsepsilon=1e-15,
                      vertexnormalsepsilon=1e-15),
        lightposition=dict(x=100,
                           y=200,
                           z=0)
    )

    # Create edge lines
    edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='lightgrey', width=1),
        name='Edges'
    )

    fig = go.Figure(data=[mesh, edges])

    # Set labels, title, and use orthographic projection
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z (meters)',
            aspectmode='data',
            camera=dict(
                projection=dict(type="orthographic")
            )
        ),
        title='3D Voxel Visualization'
    )

    # Adjust z-axis ticks to show every 10 cells or less
    z_max = voxel_grid.shape[2]
    if z_max <= 10:
        z_ticks = list(range(0, z_max + 1))
    else:
        z_ticks = list(range(0, z_max + 1, 10))

    fig.update_layout(
        scene=dict(
            zaxis=dict(
                tickvals=z_ticks,
                ticktext=[f"{z * voxel_size:.1f}" for z in z_ticks]
            )
        )
    )

    print("Visualization complete. Displaying plot...")
    fig.show()

# def plot_grid(grid, origin, adjusted_meshsize, u_vec, v_vec, transformer, vertices, data_type, vmin=None, vmax=None, alpha=0.5, buf=0.2, edge=True, **kwargs):
#     fig, ax = plt.subplots(figsize=(12, 12))

#     if data_type == 'land_cover':
#         land_cover_classes = kwargs.get('land_cover_classes')
#         colors = [mcolors.to_rgb(f'#{r:02x}{g:02x}{b:02x}') for r, g, b in land_cover_classes.keys()]
#         cmap = mcolors.ListedColormap(colors)
#         norm = mcolors.BoundaryNorm(range(len(land_cover_classes)+1), cmap.N)
#         title = 'Grid Cells with Dominant Land Cover Classes'
#         label = 'Land Cover Class'
#         tick_labels = list(land_cover_classes.values())
#     elif data_type == 'building_height':
#         # Create a masked array to handle special values
#         masked_grid = np.ma.masked_array(grid, mask=(np.isnan(grid) | (grid == 0)))
        
#         # Set up colormap and normalization for positive values
#         cmap = plt.cm.viridis
#         if vmin is None:
#             vmin = np.nanmin(masked_grid[masked_grid > 0])
#         if vmax is None:
#             vmax = np.nanmax(masked_grid)
#         norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        
#         title = 'Grid Cells with Building Heights'
#         label = 'Building Height (m)'
#         tick_labels = None
#     elif data_type == 'dem':
#         cmap = plt.cm.terrain
#         if vmin is None:
#             vmin = np.nanmin(grid)
#         if vmax is None:
#             vmax = np.nanmax(grid)
#         norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#         title = 'DEM Grid Overlaid on Map'
#         label = 'Elevation (m)'
#         tick_labels = None
#     elif data_type == 'canopy_height':
#         cmap = plt.cm.Greens
#         if vmin is None:
#             vmin = np.nanmin(grid)
#         if vmax is None:
#             vmax = np.nanmax(grid)
#         norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
#         title = 'Canopy Height Grid Overlaid on Map'
#         label = 'Canopy Height (m)'
#         tick_labels = None
#     else:
#         raise ValueError("Invalid data_type. Choose 'land_cover', 'building_height', 'canopy_height', or 'dem'.")

#     # Ensure grid is in the correct orientation
#     grid = grid.T

#     for i in range(grid.shape[0]):
#         for j in range(grid.shape[1]):
#             cell = create_cell_polygon(origin, j, i, adjusted_meshsize, u_vec, v_vec)  # Note the swap of i and j
#             x, y = cell.exterior.xy
#             x, y = zip(*[transformer.transform(lon, lat) for lat, lon in zip(x, y)])

#             value = grid[i, j]
            
#             if data_type == 'building_height':
#                 if np.isnan(value):
#                     # White fill for NaN values
#                     ax.fill(x, y, alpha=alpha, fc='white', ec='black' if edge else None, linewidth=0.1)
#                 elif value == 0:
#                     # No fill for zero values, only edges if enabled
#                     if edge:
#                         ax.plot(x, y, color='black', linewidth=0.1)
#                 elif value > 0:
#                     # Viridis colormap for positive values
#                     color = cmap(norm(value))
#                     ax.fill(x, y, alpha=alpha, fc=color, ec='black' if edge else None, linewidth=0.1)
#             else:
#                 color = cmap(norm(value))
#                 if edge:
#                     ax.fill(x, y, alpha=alpha, fc=color, ec='black', linewidth=0.1)
#                 else:
#                     ax.fill(x, y, alpha=alpha, fc=color, ec=None)

#     crs_epsg_3857 = CRS.from_epsg(3857)
#     ctx.add_basemap(ax, crs=crs_epsg_3857, source=ctx.providers.CartoDB.DarkMatter)

#     if data_type == 'building_height':
#         buildings = kwargs.get('buildings', [])
#         for building in buildings:
#             polygon = Polygon(building['geometry']['coordinates'][0])
#             x, y = polygon.exterior.xy
#             x, y = zip(*[transformer.transform(lon, lat) for lat, lon in zip(x, y)])
#             ax.plot(x, y, color='red', linewidth=1)

#     # Safe calculation of plot limits
#     all_coords = np.array(vertices)
#     x, y = zip(*[transformer.transform(lon, lat) for lat, lon in all_coords])
    
#     # Calculate limits safely
#     x_min, x_max = min(x), max(x)
#     y_min, y_max = min(y), max(y)
    
#     if x_min != x_max and y_min != y_max and buf != 0:
#         dist_x = x_max - x_min
#         dist_y = y_max - y_min
#         # Set limits with buffer
#         ax.set_xlim(x_min - buf * dist_x, x_max + buf * dist_x)
#         ax.set_ylim(y_min - buf * dist_y, y_max + buf * dist_y)
#     else:
#         # If coordinates are the same or buffer is 0, set limits without buffer
#         ax.set_xlim(x_min, x_max)
#         ax.set_ylim(y_min, y_max)

#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

def plot_grid(grid, origin, adjusted_meshsize, u_vec, v_vec, transformer, vertices, data_type, vmin=None, vmax=None, color_map=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light', **kwargs):
    fig, ax = plt.subplots(figsize=(12, 12))

    if data_type == 'land_cover':
        land_cover_classes = kwargs.get('land_cover_classes')
        colors = [mcolors.to_rgb(f'#{r:02x}{g:02x}{b:02x}') for r, g, b in land_cover_classes.keys()]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(range(len(land_cover_classes)+1), cmap.N)
        title = 'Grid Cells with Dominant Land Cover Classes'
        label = 'Land Cover Class'
        tick_labels = list(land_cover_classes.values())
    elif data_type == 'building_height':
        # Create a masked array to handle special values
        masked_grid = np.ma.masked_array(grid, mask=(np.isnan(grid) | (grid == 0)))

        # Set up colormap and normalization for positive values
        cmap = plt.cm.viridis
        if vmin is None:
            vmin = np.nanmin(masked_grid[masked_grid > 0])
        if vmax is None:
            vmax = np.nanmax(masked_grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        title = 'Grid Cells with Building Heights'
        label = 'Building Height (m)'
        tick_labels = None
    elif data_type == 'dem':
        cmap = plt.cm.terrain
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        title = 'DEM Grid Overlaid on Map'
        label = 'Elevation (m)'
        tick_labels = None
    elif data_type == 'canopy_height':
        cmap = plt.cm.Greens
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        title = 'Canopy Height Grid Overlaid on Map'
        label = 'Canopy Height (m)'
        tick_labels = None
    elif data_type == 'green_view_index':
        cmap = plt.cm.Greens
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        title = 'Green View Index Grid Overlaid on Map'
        label = 'Green View Index'
        tick_labels = None
    elif data_type == 'sky_view_index':
        cmap = plt.cm.get_cmap('BuPu_r').copy()
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        title = 'Sky View Index Grid Overlaid on Map'
        label = 'Sky View Index'
        tick_labels = None
    else:
        cmap = plt.cm.viridis
        if vmin is None:
            vmin = np.nanmin(grid)
        if vmax is None:
            vmax = np.nanmax(grid)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        tick_labels = None
        
    if color_map:
        # cmap = plt.cm.get_cmap(color_map).copy()
        cmap = sns.color_palette(color_map, as_cmap=True).copy()

    # Ensure grid is in the correct orientation
    grid = grid.T

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            cell = create_cell_polygon(origin, j, i, adjusted_meshsize, u_vec, v_vec)  # Note the swap of i and j
            x, y = cell.exterior.xy
            x, y = zip(*[transformer.transform(lon, lat) for lat, lon in zip(x, y)])

            value = grid[i, j]

            if data_type == 'building_height':
                if np.isnan(value):
                    # White fill for NaN values
                    ax.fill(x, y, alpha=alpha, fc='gray', ec='black' if edge else None, linewidth=0.1)
                elif value == 0:
                    # No fill for zero values, only edges if enabled
                    if edge:
                        ax.plot(x, y, color='black', linewidth=0.1)
                elif value > 0:
                    # Viridis colormap for positive values
                    color = cmap(norm(value))
                    ax.fill(x, y, alpha=alpha, fc=color, ec='black' if edge else None, linewidth=0.1)
            elif data_type == 'canopy_height':
                color = cmap(norm(value))
                if value == 0:
                    # No fill for zero values, only edges if enabled
                    if edge:
                        ax.plot(x, y, color='black', linewidth=0.1)
                else:
                    if edge:
                        ax.fill(x, y, alpha=alpha, fc=color, ec='black', linewidth=0.1)
                    else:
                        ax.fill(x, y, alpha=alpha, fc=color, ec=None)
            elif 'view' in data_type:
                if np.isnan(value):
                    # No fill for zero values, only edges if enabled
                    if edge:
                        ax.plot(x, y, color='black', linewidth=0.1)
                elif value >= 0:
                    # Viridis colormap for positive values
                    color = cmap(norm(value))
                    ax.fill(x, y, alpha=alpha, fc=color, ec='black' if edge else None, linewidth=0.1)
            else:
                color = cmap(norm(value))
                if edge:
                    ax.fill(x, y, alpha=alpha, fc=color, ec='black', linewidth=0.1)
                else:
                    ax.fill(x, y, alpha=alpha, fc=color, ec=None)

    crs_epsg_3857 = CRS.from_epsg(3857)

    basemaps = {
      'CartoDB dark': ctx.providers.CartoDB.DarkMatter,  # Popular dark option
      'CartoDB light': ctx.providers.CartoDB.Positron,  # Popular dark option
      'CartoDB voyager': ctx.providers.CartoDB.Voyager,  # Popular dark option
      'CartoDB dark no labels': ctx.providers.CartoDB.DarkMatterNoLabels,
    }
    ctx.add_basemap(ax, crs=crs_epsg_3857, source=basemaps[basemap])
    # if basemap == "dark":
    #     ctx.add_basemap(ax, crs=crs_epsg_3857, source=ctx.providers.CartoDB.DarkMatter)
    # elif basemap == 'light':
    #     ctx.add_basemap(ax, crs=crs_epsg_3857, source=ctx.providers.CartoDB.Positron)
    # elif basemap == 'voyager':
    #     ctx.add_basemap(ax, crs=crs_epsg_3857, source=ctx.providers.CartoDB.Voyager)

    if data_type == 'building_height':
        buildings = kwargs.get('buildings', [])
        for building in buildings:
            polygon = Polygon(building['geometry']['coordinates'][0])
            x, y = polygon.exterior.xy
            x, y = zip(*[transformer.transform(lon, lat) for lat, lon in zip(x, y)])
            ax.plot(x, y, color='red', linewidth=1.5)
            # print(polygon)

    # Safe calculation of plot limits
    all_coords = np.array(vertices)
    x, y = zip(*[transformer.transform(lon, lat) for lat, lon in all_coords])

    # Calculate limits safely
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    if x_min != x_max and y_min != y_max and buf != 0:
        dist_x = x_max - x_min
        dist_y = y_max - y_min
        # Set limits with buffer
        ax.set_xlim(x_min - buf * dist_x, x_max + buf * dist_x)
        ax.set_ylim(y_min - buf * dist_y, y_max + buf * dist_y)
    else:
        # If coordinates are the same or buffer is 0, set limits without buffer
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_land_cover_grid_on_map(grid, rectangle_vertices, meshsize, source = 'Urbanwatch', vmin=None, vmax=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light'):

    geod = initialize_geod()

    land_cover_classes = get_land_cover_classes(source)

    vertex_0 = rectangle_vertices[0]
    vertex_1 = rectangle_vertices[1]
    vertex_3 = rectangle_vertices[3]

    dist_side_1 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_1[1], vertex_1[0])
    dist_side_2 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_3[1], vertex_3[0])

    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)

    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)

    print(f"Calculated grid size: {grid_size}")
    # print(f"Adjusted mesh size: {adjusted_meshsize}")

    geotiff_crs = CRS.from_epsg(3857)
    transformer = setup_transformer(CRS.from_epsg(4326), geotiff_crs)

    cell_coords = create_coordinate_mesh(origin, grid_size, adjusted_meshsize, u_vec, v_vec)
    cell_coords_flat = cell_coords.reshape(2, -1).T
    transformed_coords = np.array([transform_coords(transformer, lon, lat) for lat, lon in cell_coords_flat])
    transformed_coords = transformed_coords.reshape(grid_size[::-1] + (2,))

    # print(f"Grid shape: {grid.shape}")

    plot_grid(grid, origin, adjusted_meshsize, u_vec, v_vec, transformer,
              rectangle_vertices, 'land_cover', alpha=alpha, buf=buf, edge=edge, basemap=basemap, land_cover_classes=land_cover_classes)

    unique_indices = np.unique(grid)
    unique_classes = [list(land_cover_classes.values())[i] for i in unique_indices]
    # print(f"Unique classes in the grid: {unique_classes}")

def visualize_building_height_grid_on_map(building_height_grid, filtered_buildings, rectangle_vertices, meshsize, vmin=None, vmax=None, color_map=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light'):
    # Calculate grid and normalize vectors
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

    # Setup transformer and plotting extent
    transformer = setup_transformer(CRS.from_epsg(4326), CRS.from_epsg(3857))

    # Plot the results
    plot_grid(building_height_grid, origin, adjusted_meshsize, u_vec, v_vec, transformer,
              rectangle_vertices, 'building_height', vmin=vmin, vmax=vmax, color_map=color_map, alpha=alpha, buf=buf, edge=edge, basemap=basemap, buildings=filtered_buildings)
    
def visualize_numerical_grid_on_map(canopy_height_grid, rectangle_vertices, meshsize, type, vmin=None, vmax=None, color_map=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light'):
    # Calculate grid and normalize vectors
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

    # Setup transformer and plotting extent
    transformer = setup_transformer(CRS.from_epsg(4326), CRS.from_epsg(3857))

    # Plot the results
    plot_grid(canopy_height_grid, origin, adjusted_meshsize, u_vec, v_vec, transformer,
              rectangle_vertices, type, vmin=vmin, vmax=vmax, color_map=color_map, alpha=alpha, buf=buf, edge=edge, basemap=basemap)
    
def visualize_land_cover_grid(grid, mesh_size, color_map, land_cover_classes):
    all_classes = list(land_cover_classes.values())# + ['No Data']
    # for cls in all_classes:
    #     if cls not in color_map:
    #         color_map[cls] = [0.5, 0.5, 0.5]

    sorted_classes = sorted(all_classes)
    colors = [color_map[cls] for cls in sorted_classes]
    cmap = mcolors.ListedColormap(colors)

    bounds = np.arange(len(sorted_classes) + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    class_to_num = {cls: i for i, cls in enumerate(sorted_classes)}
    numeric_grid = np.vectorize(class_to_num.get)(grid)

    plt.figure(figsize=(10, 10))
    im = plt.imshow(numeric_grid, cmap=cmap, norm=norm, interpolation='nearest')
    cbar = plt.colorbar(im, ticks=bounds[:-1] + 0.5)
    cbar.set_ticklabels(sorted_classes)
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