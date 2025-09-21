"""
VoxelCity Visualization Utilities

This module provides comprehensive visualization tools for 3D voxel city data,
including support for multiple color schemes, 3D plotting with matplotlib and plotly,
grid visualization on basemaps, and mesh-based rendering with PyVista.

The module handles various data types including:
- Land cover classifications
- Building heights and footprints
- Digital elevation models (DEM)
- Canopy heights
- View indices (sky view factor, green view index)
- Simulation results on building surfaces

Key Features:
- Multiple predefined color schemes for voxel visualization
- 2D and 3D plotting capabilities
- Interactive web maps with folium
- Mesh export functionality (OBJ format)
- Multi-view scene generation
- Custom simulation result overlays
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.cm as cm
import contextily as ctx
from shapely.geometry import Polygon
import plotly.graph_objects as go
from tqdm import tqdm
import pyproj
# import rasterio
from pyproj import CRS
# from shapely.geometry import box
import seaborn as sns
import random
import folium
import math
import trimesh
import pyvista as pv
from IPython.display import display
import os
import sys

# Import utility functions for land cover classification
from .lc import get_land_cover_classes
# from ..geo.geojson import filter_buildings

# Import grid processing functions
from ..geoprocessor.grid import (
    calculate_grid_size,
    create_coordinate_mesh,
    create_cell_polygon,
    grid_to_geodataframe
)

# Import geospatial utility functions
from ..geoprocessor.utils import (
    initialize_geod,
    calculate_distance,
    normalize_to_one_meter,
    setup_transformer,
    transform_coords,
)

# Import mesh generation and export functions
from ..geoprocessor.mesh import (
    create_voxel_mesh,
    create_sim_surface_mesh,
    create_city_meshes,
    export_meshes,
    save_obj_from_colored_mesh
)
# from ..exporter.obj import save_obj_from_colored_mesh

# Import material property functions
from .material import get_material_dict

def get_voxel_color_map(color_scheme='default'):
    """
    Returns a color map for voxel visualization based on the specified color scheme.
    
    This function provides multiple predefined color schemes for visualizing voxel data.
    Each scheme maps voxel class IDs to RGB color values [0-255]. The class IDs follow
    a specific convention where negative values represent built environment elements
    and positive values represent natural/ground surface elements.
    
    Voxel Class ID Convention:
        -99: Void/empty space (black)
        -30: Landmark buildings (special highlighting)
        -17 to -11: Building materials (plaster, glass, stone, metal, concrete, wood, brick)
        -3: Generic building structures
        -2: Trees/vegetation (above ground)
        -1: Underground/subsurface
        1-14: Ground surface land cover types (bareland, vegetation, water, etc.)
    
    Parameters:
    -----------
    color_scheme : str, optional
        The name of the color scheme to use. Available options:
        
        Basic Schemes:
        - 'default': Original balanced color scheme for general use
        - 'high_contrast': High contrast colors for better visibility
        - 'monochrome': Shades of blue for academic presentations
        - 'pastel': Softer, muted colors for aesthetic appeal
        - 'dark_mode': Darker colors for dark backgrounds
        - 'grayscale': Black and white gradient with color accents
        - 'white_mode': Light minimal palette for white backgrounds
        
        Thematic Schemes:
        - 'autumn': Warm reds, oranges, and browns
        - 'cool': Cool blues, purples, and cyans
        - 'earth_tones': Natural earth colors
        - 'vibrant': Very bright, saturated colors
        
        Stylistic Schemes:
        - 'cyberpunk': Neon-like purples, pinks, and blues
        - 'tropical': Vibrant greens, oranges, pinks (island vibes)
        - 'vintage': Muted, sepia-like tones
        - 'neon_dreams': Super-bright, nightclub neon palette

    Returns:
    --------
    dict
        A dictionary mapping voxel class IDs (int) to RGB color values (list of 3 ints [0-255])
        
    Examples:
    ---------
    >>> colors = get_voxel_color_map('default')
    >>> print(colors[-3])  # Building color
    [180, 187, 216]
    
    >>> colors = get_voxel_color_map('cyberpunk')
    >>> print(colors[9])   # Water color in cyberpunk scheme
    [51, 0, 102]
    
    Notes:
    ------
    - All color values are in RGB format with range [0, 255]
    - The 'default' scheme should not be modified to maintain consistency
    - Unknown color schemes will fall back to 'default' with a warning
    - Color schemes can be extended by adding new elif blocks
    """
    # ----------------------
    # DO NOT MODIFY DEFAULT
    # ----------------------
    if color_scheme == 'default':
        return {
            -99: [0, 0, 0],  # void,
            -30: [255, 0, 102],  # (Pink) 'Landmark',
            -17: [238, 242, 234],  # (light gray) 'plaster',
            -16: [56, 78, 84],  # (Dark blue) 'glass',
            -15: [147, 140, 114],  # (Light brown) 'stone',
            -14: [139, 149, 159],  # (Gray) 'metal',
            -13: [186, 187, 181],  # (Gray) 'concrete',
            -12: [248, 166, 2],  # (Orange) 'wood',
            -11: [81, 59, 56],  # (Dark red) 'brick',
            -3: [180, 187, 216],  # Building
            -2: [78, 99, 63],     # Tree
            -1: [188, 143, 143],  # Underground
            1: [239, 228, 176],   # 'Bareland (ground surface)',
            2: [123, 130, 59],   # 'Rangeland (ground surface)',
            3: [97, 140, 86],   # 'Shrub (ground surface)',
            4: [112, 120, 56],   #  'Agriculture land (ground surface)',
            5: [116, 150, 66],   #  'Tree (ground surface)',
            6: [187, 204, 40],   #  'Moss and lichen (ground surface)',
            7: [77, 118, 99],    #  'Wet land (ground surface)',
            8: [22, 61, 51],    #  'Mangrove (ground surface)',
            9: [44, 66, 133],    #  'Water (ground surface)',
            10: [205, 215, 224],    #  'Snow and ice (ground surface)',
            11: [108, 119, 129],   #  'Developed space (ground surface)',
            12: [59, 62, 87],      # 'Road (ground surface)',
            13: [150, 166, 190],    #  'Building (ground surface)'
            14: [239, 228, 176],    #  'No Data (ground surface)'
        }

    elif color_scheme == 'high_contrast':
        return {
            -99: [0, 0, 0],  # void
            -30: [255, 0, 255],  # (Bright Magenta) 'Landmark'
            -17: [255, 255, 255],  # (Pure White) 'plaster'
            -16: [0, 0, 255],  # (Bright Blue) 'glass'
            -15: [153, 76, 0],  # (Dark Brown) 'stone'
            -14: [192, 192, 192],  # (Silver) 'metal'
            -13: [128, 128, 128],  # (Gray) 'concrete'
            -12: [255, 128, 0],  # (Bright Orange) 'wood'
            -11: [153, 0, 0],  # (Dark Red) 'brick'
            -3: [0, 255, 255],  # (Cyan) Building
            -2: [0, 153, 0],  # (Green) Tree
            -1: [204, 0, 102],  # (Dark Pink) Underground
            1: [255, 255, 153],  # (Light Yellow) 'Bareland'
            2: [102, 153, 0],  # (Olive Green) 'Rangeland'
            3: [0, 204, 0],  # (Bright Green) 'Shrub'
            4: [153, 204, 0],  # (Yellowish Green) 'Agriculture land'
            5: [0, 102, 0],  # (Dark Green) 'Tree'
            6: [204, 255, 51],  # (Lime Green) 'Moss and lichen'
            7: [0, 153, 153],  # (Teal) 'Wet land'
            8: [0, 51, 0],  # (Very Dark Green) 'Mangrove'
            9: [0, 102, 204],  # (Bright Blue) 'Water'
            10: [255, 255, 255],  # (White) 'Snow and ice'
            11: [76, 76, 76],  # (Dark Gray) 'Developed space'
            12: [0, 0, 0],  # (Black) 'Road'
            13: [102, 102, 255],  # (Light Purple) 'Building'
            14: [255, 204, 153],  # (Light Orange) 'No Data'
        }

    elif color_scheme == 'monochrome':
        return {
            -99: [0, 0, 0],  # void
            -30: [28, 28, 99],  # 'Landmark'
            -17: [242, 242, 242],  # 'plaster'
            -16: [51, 51, 153],  # 'glass'
            -15: [102, 102, 204],  # 'stone'
            -14: [153, 153, 204],  # 'metal'
            -13: [204, 204, 230],  # 'concrete'
            -12: [76, 76, 178],  # 'wood'
            -11: [25, 25, 127],  # 'brick'
            -3: [179, 179, 230],  # Building
            -2: [51, 51, 153],  # Tree
            -1: [102, 102, 178],  # Underground
            1: [230, 230, 255],  # 'Bareland'
            2: [128, 128, 204],  # 'Rangeland'
            3: [102, 102, 204],  # 'Shrub'
            4: [153, 153, 230],  # 'Agriculture land'
            5: [76, 76, 178],  # 'Tree'
            6: [204, 204, 255],  # 'Moss and lichen'
            7: [76, 76, 178],  # 'Wet land'
            8: [25, 25, 127],  # 'Mangrove'
            9: [51, 51, 204],  # 'Water'
            10: [242, 242, 255],  # 'Snow and ice'
            11: [128, 128, 178],  # 'Developed space'
            12: [51, 51, 127],  # 'Road'
            13: [153, 153, 204],  # 'Building'
            14: [230, 230, 255],  # 'No Data'
        }

    elif color_scheme == 'pastel':
        return {
            -99: [0, 0, 0],  # void
            -30: [255, 179, 217],  # (Pastel Pink) 'Landmark'
            -17: [245, 245, 245],  # (Off White) 'plaster'
            -16: [173, 196, 230],  # (Pastel Blue) 'glass'
            -15: [222, 213, 196],  # (Pastel Brown) 'stone'
            -14: [211, 219, 226],  # (Pastel Gray) 'metal'
            -13: [226, 226, 226],  # (Light Gray) 'concrete'
            -12: [255, 223, 179],  # (Pastel Orange) 'wood'
            -11: [204, 168, 166],  # (Pastel Red) 'brick'
            -3: [214, 217, 235],   # (Pastel Purple) Building
            -2: [190, 207, 180],   # (Pastel Green) Tree
            -1: [235, 204, 204],   # (Pastel Pink) Underground
            1: [250, 244, 227],    # (Cream) 'Bareland'
            2: [213, 217, 182],    # (Pastel Olive) 'Rangeland'
            3: [200, 226, 195],    # (Pastel Green) 'Shrub'
            4: [209, 214, 188],    # (Pastel Yellow-Green) 'Agriculture land'
            5: [195, 220, 168],    # (Light Pastel Green) 'Tree'
            6: [237, 241, 196],    # (Pastel Yellow) 'Moss and lichen'
            7: [180, 210, 205],    # (Pastel Teal) 'Wet land'
            8: [176, 196, 190],    # (Darker Pastel Teal) 'Mangrove'
            9: [188, 206, 235],    # (Pastel Blue) 'Water'
            10: [242, 245, 250],   # (Light Blue-White) 'Snow and ice'
            11: [209, 213, 219],   # (Pastel Gray) 'Developed space'
            12: [189, 190, 204],   # (Pastel Blue-Gray) 'Road'
            13: [215, 221, 232],   # (Very Light Pastel Blue) 'Building'
            14: [250, 244, 227],   # (Cream) 'No Data'
        }

    elif color_scheme == 'dark_mode':
        return {
            -99: [0, 0, 0],  # void
            -30: [153, 51, 102],   # (Dark Pink) 'Landmark'
            -17: [76, 76, 76],     # (Dark Gray) 'plaster'
            -16: [33, 46, 51],     # (Very Dark Blue) 'glass'
            -15: [89, 84, 66],     # (Very Dark Brown) 'stone'
            -14: [83, 89, 94],     # (Dark Gray) 'metal'
            -13: [61, 61, 61],     # (Dark Gray) 'concrete'
            -12: [153, 102, 0],    # (Dark Orange) 'wood'
            -11: [51, 35, 33],     # (Very Dark Red) 'brick'
            -3: [78, 82, 99],      # (Dark Purple) Building
            -2: [46, 58, 37],      # (Dark Green) Tree
            -1: [99, 68, 68],      # (Dark Pink) Underground
            1: [102, 97, 75],      # (Dark Yellow) 'Bareland'
            2: [61, 66, 31],       # (Dark Olive) 'Rangeland'
            3: [46, 77, 46],       # (Dark Green) 'Shrub'
            4: [56, 61, 28],       # (Dark Yellow-Green) 'Agriculture land'
            5: [54, 77, 31],       # (Dark Green) 'Tree'
            6: [89, 97, 20],       # (Dark Yellow) 'Moss and lichen'
            7: [38, 59, 49],       # (Dark Teal) 'Wet land'
            8: [16, 31, 26],       # (Very Dark Green) 'Mangrove'
            9: [22, 33, 66],       # (Dark Blue) 'Water'
            10: [82, 87, 92],      # (Dark Blue-Gray) 'Snow and ice'
            11: [46, 51, 56],      # (Dark Gray) 'Developed space'
            12: [25, 31, 43],      # (Very Dark Blue) 'Road'
            13: [56, 64, 82],      # (Dark Blue-Gray) 'Building'
            14: [102, 97, 75],     # (Dark Yellow) 'No Data'
        }

    elif color_scheme == 'grayscale':
        return {
            -99: [0, 0, 0],      # void (black)
            -30: [253, 231, 37],  # (Pink) 'Landmark',
            -17: [240, 240, 240], # 'plaster'
            -16: [60, 60, 60],    # 'glass'
            -15: [130, 130, 130], # 'stone'
            -14: [150, 150, 150], # 'metal'
            -13: [180, 180, 180], # 'concrete'
            -12: [170, 170, 170], # 'wood'
            -11: [70, 70, 70],    # 'brick'
            -3: [190, 190, 190],  # Building
            -2: [90, 90, 90],     # Tree
            -1: [160, 160, 160],  # Underground
            1: [230, 230, 230],   # 'Bareland'
            2: [120, 120, 120],   # 'Rangeland'
            3: [110, 110, 110],   # 'Shrub'
            4: [115, 115, 115],   # 'Agriculture land'
            5: [100, 100, 100],   # 'Tree'
            6: [210, 210, 210],   # 'Moss and lichen'
            7: [95, 95, 95],      # 'Wet land'
            8: [40, 40, 40],      # 'Mangrove'
            9: [50, 50, 50],      # 'Water'
            10: [220, 220, 220],  # 'Snow and ice'
            11: [140, 140, 140],  # 'Developed space'
            12: [30, 30, 30],     # 'Road'
            13: [170, 170, 170],  # 'Building'
            14: [230, 230, 230],  # 'No Data'
        }

    elif color_scheme == 'white_mode':
        return {
            -99: [0, 0, 0],          # void (transparent in rendering)
            -30: [220, 80, 80],      # subtle highlight for landmarks
            -17: [250, 250, 250],    # plaster (near white)
            -16: [210, 225, 235],    # glass (light blue-gray)
            -15: [230, 225, 215],    # stone (warm light gray)
            -14: [225, 230, 235],    # metal (cool light gray)
            -13: [236, 236, 236],    # concrete (very light gray)
            -12: [245, 232, 210],    # wood (light beige)
            -11: [235, 210, 205],    # brick (light rose)
            -3: [225, 230, 240],     # Building (soft blue-gray)
            -2: [190, 210, 190],     # Tree (soft green)
            -1: [230, 215, 215],     # Underground (soft pinkish)
            1: [248, 245, 235],      # Bareland
            2: [225, 235, 215],      # Rangeland
            3: [220, 235, 220],      # Shrub
            4: [240, 235, 215],      # Agriculture land
            5: [210, 230, 210],      # Tree (ground)
            6: [245, 250, 235],      # Moss and lichen
            7: [220, 235, 230],      # Wet land
            8: [205, 215, 210],      # Mangrove
            9: [200, 220, 245],      # Water (pale blue)
            10: [252, 252, 252],     # Snow and ice (almost white)
            11: [230, 230, 230],     # Developed space
            12: [210, 210, 215],     # Road (light neutral)
            13: [230, 235, 240],     # Building (ground surface)
            14: [248, 245, 235],     # No Data
        }

    elif color_scheme == 'autumn':
        return {
            -99: [0, 0, 0],          # void
            -30: [227, 66, 52],      # (Red) 'Landmark'
            -17: [250, 240, 230],    # (Antique White) 'plaster'
            -16: [94, 33, 41],       # (Dark Red) 'glass'
            -15: [160, 120, 90],     # (Medium Brown) 'stone'
            -14: [176, 141, 87],     # (Bronze) 'metal'
            -13: [205, 186, 150],    # (Tan) 'concrete'
            -12: [204, 85, 0],       # (Dark Orange) 'wood'
            -11: [128, 55, 36],      # (Rust) 'brick'
            -3: [222, 184, 135],     # (Tan) Building
            -2: [107, 68, 35],       # (Brown) Tree
            -1: [165, 105, 79],      # (Copper) Underground
            1: [255, 235, 205],      # (Blanched Almond) 'Bareland'
            2: [133, 99, 99],        # (Brown) 'Rangeland'
            3: [139, 69, 19],        # (Saddle Brown) 'Shrub'
            4: [160, 82, 45],        # (Sienna) 'Agriculture land'
            5: [101, 67, 33],        # (Dark Brown) 'Tree'
            6: [255, 228, 196],      # (Bisque) 'Moss and lichen'
            7: [138, 51, 36],        # (Rust) 'Wet land'
            8: [85, 45, 23],         # (Deep Brown) 'Mangrove'
            9: [175, 118, 70],       # (Light Brown) 'Water'
            10: [255, 250, 240],     # (Floral White) 'Snow and ice'
            11: [188, 143, 143],     # (Rosy Brown) 'Developed space'
            12: [69, 41, 33],        # (Very Dark Brown) 'Road'
            13: [210, 180, 140],     # (Tan) 'Building'
            14: [255, 235, 205],     # (Blanched Almond) 'No Data'
        }

    elif color_scheme == 'cool':
        return {
            -99: [0, 0, 0],          # void
            -30: [180, 82, 205],     # (Purple) 'Landmark'
            -17: [240, 248, 255],    # (Alice Blue) 'plaster'
            -16: [70, 130, 180],     # (Steel Blue) 'glass'
            -15: [100, 149, 237],    # (Cornflower Blue) 'stone'
            -14: [176, 196, 222],    # (Light Steel Blue) 'metal'
            -13: [240, 255, 255],    # (Azure) 'concrete'
            -12: [65, 105, 225],     # (Royal Blue) 'wood'
            -11: [95, 158, 160],     # (Cadet Blue) 'brick'
            -3: [135, 206, 235],     # (Sky Blue) Building
            -2: [0, 128, 128],       # (Teal) Tree
            -1: [127, 255, 212],     # (Aquamarine) Underground
            1: [220, 240, 250],      # (Light Blue) 'Bareland'
            2: [72, 209, 204],       # (Medium Turquoise) 'Rangeland'
            3: [0, 191, 255],        # (Deep Sky Blue) 'Shrub'
            4: [100, 149, 237],      # (Cornflower Blue) 'Agriculture land'
            5: [0, 128, 128],        # (Teal) 'Tree'
            6: [175, 238, 238],      # (Pale Turquoise) 'Moss and lichen'
            7: [32, 178, 170],       # (Light Sea Green) 'Wet land'
            8: [25, 25, 112],        # (Midnight Blue) 'Mangrove'
            9: [30, 144, 255],       # (Dodger Blue) 'Water'
            10: [240, 255, 255],     # (Azure) 'Snow and ice'
            11: [119, 136, 153],     # (Light Slate Gray) 'Developed space'
            12: [25, 25, 112],       # (Midnight Blue) 'Road'
            13: [173, 216, 230],     # (Light Blue) 'Building'
            14: [220, 240, 250],     # (Light Blue) 'No Data'
        }

    elif color_scheme == 'earth_tones':
        return {
            -99: [0, 0, 0],          # void
            -30: [210, 105, 30],     # (Chocolate) 'Landmark'
            -17: [245, 245, 220],    # (Beige) 'plaster'
            -16: [139, 137, 137],    # (Gray) 'glass'
            -15: [160, 120, 90],     # (Medium Brown) 'stone'
            -14: [169, 169, 169],    # (Dark Gray) 'metal'
            -13: [190, 190, 180],    # (Light Gray-Tan) 'concrete'
            -12: [160, 82, 45],      # (Sienna) 'wood'
            -11: [139, 69, 19],      # (Saddle Brown) 'brick'
            -3: [210, 180, 140],     # (Tan) Building
            -2: [85, 107, 47],       # (Dark Olive Green) Tree
            -1: [133, 94, 66],       # (Beaver) Underground
            1: [222, 184, 135],      # (Burlywood) 'Bareland'
            2: [107, 142, 35],       # (Olive Drab) 'Rangeland'
            3: [85, 107, 47],        # (Dark Olive Green) 'Shrub'
            4: [128, 128, 0],        # (Olive) 'Agriculture land'
            5: [34, 139, 34],        # (Forest Green) 'Tree'
            6: [189, 183, 107],      # (Dark Khaki) 'Moss and lichen'
            7: [143, 188, 143],      # (Dark Sea Green) 'Wet land'
            8: [46, 139, 87],        # (Sea Green) 'Mangrove'
            9: [95, 158, 160],       # (Cadet Blue) 'Water'
            10: [238, 232, 205],     # (Light Tan) 'Snow and ice'
            11: [169, 169, 169],     # (Dark Gray) 'Developed space'
            12: [90, 90, 90],        # (Dark Gray) 'Road'
            13: [188, 170, 152],     # (Tan) 'Building'
            14: [222, 184, 135],     # (Burlywood) 'No Data'
        }

    elif color_scheme == 'vibrant':
        return {
            -99: [0, 0, 0],          # void
            -30: [255, 0, 255],      # (Magenta) 'Landmark'
            -17: [255, 255, 255],    # (White) 'plaster'
            -16: [0, 191, 255],      # (Deep Sky Blue) 'glass'
            -15: [255, 215, 0],      # (Gold) 'stone'
            -14: [0, 250, 154],      # (Medium Spring Green) 'metal'
            -13: [211, 211, 211],    # (Light Gray) 'concrete'
            -12: [255, 69, 0],       # (Orange Red) 'wood'
            -11: [178, 34, 34],      # (Firebrick) 'brick'
            -3: [123, 104, 238],     # (Medium Slate Blue) Building
            -2: [50, 205, 50],       # (Lime Green) Tree
            -1: [255, 20, 147],      # (Deep Pink) Underground
            1: [255, 255, 0],        # (Yellow) 'Bareland'
            2: [0, 255, 0],          # (Lime) 'Rangeland'
            3: [0, 128, 0],          # (Green) 'Shrub'
            4: [154, 205, 50],       # (Yellow Green) 'Agriculture land'
            5: [34, 139, 34],        # (Forest Green) 'Tree'
            6: [127, 255, 0],        # (Chartreuse) 'Moss and lichen'
            7: [64, 224, 208],       # (Turquoise) 'Wet land'
            8: [0, 100, 0],          # (Dark Green) 'Mangrove'
            9: [0, 0, 255],          # (Blue) 'Water'
            10: [240, 248, 255],     # (Alice Blue) 'Snow and ice'
            11: [128, 128, 128],     # (Gray) 'Developed space'
            12: [47, 79, 79],        # (Dark Slate Gray) 'Road'
            13: [135, 206, 250],     # (Light Sky Blue) 'Building'
            14: [255, 255, 224],     # (Light Yellow) 'No Data'
        }

    # ------------------------------------------------
    # NEWLY ADDED STYLISH COLOR SCHEMES BELOW:
    # ------------------------------------------------
    elif color_scheme == 'cyberpunk':
        """
        Vibrant neon purples, pinks, and blues with deep blacks.
        Think futuristic city vibes and bright neon signs.
        """
        return {
            -99: [0, 0, 0],           # void (keep it pitch black)
            -30: [255, 0, 255],       # (Neon Magenta) 'Landmark'
            -17: [255, 255, 255],     # (Bright White) 'plaster'
            -16: [0, 255, 255],       # (Neon Cyan) 'glass'
            -15: [128, 0, 128],       # (Purple) 'stone'
            -14: [50, 50, 50],        # (Dark Gray) 'metal'
            -13: [102, 0, 102],       # (Dark Magenta) 'concrete'
            -12: [255, 20, 147],      # (Deep Pink) 'wood'
            -11: [153, 0, 76],        # (Deep Purple-Red) 'brick'
            -3: [124, 0, 255],        # (Strong Neon Purple) Building
            -2: [0, 255, 153],        # (Neon Greenish Cyan) Tree
            -1: [255, 0, 102],        # (Hot Pink) Underground
            1: [255, 255, 153],       # (Pale Yellow) 'Bareland'
            2: [0, 204, 204],         # (Teal) 'Rangeland'
            3: [153, 51, 255],        # (Light Purple) 'Shrub'
            4: [0, 153, 255],         # (Bright Neon Blue) 'Agriculture land'
            5: [0, 255, 153],         # (Neon Greenish Cyan) 'Tree'
            6: [204, 0, 255],         # (Vivid Violet) 'Moss and lichen'
            7: [0, 255, 255],         # (Neon Cyan) 'Wet land'
            8: [0, 102, 102],         # (Dark Teal) 'Mangrove'
            9: [51, 0, 102],          # (Deep Indigo) 'Water'
            10: [255, 255, 255],      # (White) 'Snow and ice'
            11: [102, 102, 102],      # (Gray) 'Developed space'
            12: [0, 0, 0],            # (Black) 'Road'
            13: [204, 51, 255],       # (Bright Magenta) 'Building'
            14: [255, 255, 153],      # (Pale Yellow) 'No Data'
        }

    elif color_scheme == 'tropical':
        """
        Bold, bright 'tropical vacation' color palette.
        Lots of greens, oranges, pinks, reminiscent of island florals.
        """
        return {
            -99: [0, 0, 0],            # void
            -30: [255, 99, 164],       # (Bright Tropical Pink) 'Landmark'
            -17: [255, 248, 220],      # (Cornsilk) 'plaster'
            -16: [0, 150, 136],        # (Teal) 'glass'
            -15: [255, 140, 0],        # (Dark Orange) 'stone'
            -14: [255, 215, 180],      # (Light Peach) 'metal'
            -13: [210, 210, 210],      # (Light Gray) 'concrete'
            -12: [255, 165, 0],        # (Orange) 'wood'
            -11: [205, 92, 92],        # (Indian Red) 'brick'
            -3: [255, 193, 37],        # (Tropical Yellow) Building
            -2: [34, 139, 34],         # (Forest Green) Tree
            -1: [255, 160, 122],       # (Light Salmon) Underground
            1: [240, 230, 140],        # (Khaki) 'Bareland'
            2: [60, 179, 113],         # (Medium Sea Green) 'Rangeland'
            3: [46, 139, 87],          # (Sea Green) 'Shrub'
            4: [255, 127, 80],         # (Coral) 'Agriculture land'
            5: [50, 205, 50],          # (Lime Green) 'Tree'
            6: [255, 239, 213],        # (Papaya Whip) 'Moss and lichen'
            7: [255, 99, 71],          # (Tomato) 'Wet land'
            8: [47, 79, 79],           # (Dark Slate Gray) 'Mangrove'
            9: [0, 128, 128],          # (Teal) 'Water'
            10: [224, 255, 255],       # (Light Cyan) 'Snow and ice'
            11: [218, 112, 214],       # (Orchid) 'Developed space'
            12: [85, 107, 47],         # (Dark Olive Green) 'Road'
            13: [253, 245, 230],       # (Old Lace) 'Building'
            14: [240, 230, 140],       # (Khaki) 'No Data'
        }

    elif color_scheme == 'vintage':
        """
        A muted, old-photo or sepia-inspired palette 
        for a nostalgic or antique look.
        """
        return {
            -99: [0, 0, 0],            # void
            -30: [133, 94, 66],        # (Beaver/Brownish) 'Landmark'
            -17: [250, 240, 230],      # (Antique White) 'plaster'
            -16: [169, 157, 143],      # (Muted Brown-Gray) 'glass'
            -15: [181, 166, 127],      # (Khaki Tan) 'stone'
            -14: [120, 106, 93],       # (Faded Gray-Brown) 'metal'
            -13: [190, 172, 145],      # (Light Brown) 'concrete'
            -12: [146, 109, 83],       # (Leather Brown) 'wood'
            -11: [125, 80, 70],        # (Dusty Brick) 'brick'
            -3: [201, 174, 146],       # (Tanned Beige) Building
            -2: [112, 98, 76],         # (Faded Olive-Brown) Tree
            -1: [172, 140, 114],       # (Light Saddle Brown) Underground
            1: [222, 202, 166],        # (Light Tan) 'Bareland'
            2: [131, 114, 83],         # (Brownish) 'Rangeland'
            3: [105, 96, 74],          # (Dark Olive Brown) 'Shrub'
            4: [162, 141, 118],        # (Beige Brown) 'Agriculture land'
            5: [95, 85, 65],           # (Muted Dark Brown) 'Tree'
            6: [212, 200, 180],        # (Off-White Tan) 'Moss and lichen'
            7: [140, 108, 94],         # (Dusky Mauve-Brown) 'Wet land'
            8: [85, 73, 60],           # (Dark Taupe) 'Mangrove'
            9: [166, 152, 121],        # (Pale Brown) 'Water'
            10: [250, 245, 235],       # (Light Antique White) 'Snow and ice'
            11: [120, 106, 93],        # (Faded Gray-Brown) 'Developed space'
            12: [77, 66, 55],          # (Dark Taupe) 'Road'
            13: [203, 188, 162],       # (Light Warm Gray) 'Building'
            14: [222, 202, 166],       # (Light Tan) 'No Data'
        }

    elif color_scheme == 'neon_dreams':
        """
        A super-bright, high-energy neon palette.
        Perfect if you want a 'nightclub in 2080' vibe.
        """
        return {
            -99: [0, 0, 0],           # void
            -30: [255, 0, 255],       # (Magenta) 'Landmark'
            -17: [255, 255, 255],     # (White) 'plaster'
            -16: [0, 255, 255],       # (Cyan) 'glass'
            -15: [255, 255, 0],       # (Yellow) 'stone'
            -14: [0, 255, 0],         # (Lime) 'metal'
            -13: [128, 128, 128],     # (Gray) 'concrete'
            -12: [255, 165, 0],       # (Neon Orange) 'wood'
            -11: [255, 20, 147],      # (Deep Pink) 'brick'
            -3: [75, 0, 130],         # (Indigo) Building
            -2: [102, 255, 0],        # (Bright Lime Green) Tree
            -1: [255, 51, 153],       # (Neon Pink) Underground
            1: [255, 153, 0],         # (Bright Orange) 'Bareland'
            2: [153, 204, 0],         # (Vivid Yellow-Green) 'Rangeland'
            3: [102, 205, 170],       # (Aquamarine-ish) 'Shrub'
            4: [0, 250, 154],         # (Medium Spring Green) 'Agriculture land'
            5: [173, 255, 47],        # (Green-Yellow) 'Tree'
            6: [127, 255, 0],         # (Chartreuse) 'Moss and lichen'
            7: [64, 224, 208],        # (Turquoise) 'Wet land'
            8: [0, 128, 128],         # (Teal) 'Mangrove'
            9: [0, 0, 255],           # (Blue) 'Water'
            10: [224, 255, 255],      # (Light Cyan) 'Snow and ice'
            11: [192, 192, 192],      # (Silver) 'Developed space'
            12: [25, 25, 25],         # (Near Black) 'Road'
            13: [75, 0, 130],         # (Indigo) 'Building'
            14: [255, 153, 0],        # (Bright Orange) 'No Data'
        }

    else:
        # If an unknown color scheme is specified, return the default
        print(f"Unknown color scheme '{color_scheme}'. Using default instead.")
        return get_voxel_color_map('default')

def plot_grid(grid, origin, adjusted_meshsize, u_vec, v_vec, transformer, vertices, data_type, vmin=None, vmax=None, color_map=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light', **kwargs):
    """
    Core function for plotting 2D grid data overlaid on basemaps.
    
    This function handles the visualization of various types of grid data by creating
    colored polygons for each grid cell and overlaying them on a web basemap. It supports
    different data types with appropriate color schemes and handles special values like
    NaN and zero appropriately.
    
    Parameters:
    -----------
    grid : numpy.ndarray
        2D array containing the grid data values to visualize.
        
    origin : numpy.ndarray
        Geographic coordinates [lon, lat] of the grid's origin point.
        
    adjusted_meshsize : float
        Size of each grid cell in meters after grid size adjustments.
        
    u_vec, v_vec : numpy.ndarray
        Unit vectors defining the grid orientation in geographic space.
        
    transformer : pyproj.Transformer
        Coordinate transformer for converting between geographic and projected coordinates.
        
    vertices : list
        List of [lon, lat] coordinates defining the grid boundary.
        
    data_type : str
        Type of data being visualized. Supported types:
        - 'land_cover': Land use/land cover classifications
        - 'building_height': Building height values
        - 'dem': Digital elevation model
        - 'canopy_height': Vegetation height
        - 'green_view_index': Green visibility index
        - 'sky_view_index': Sky visibility factor
        
    vmin, vmax : float, optional
        Min/max values for color scaling. Auto-calculated if not provided.
        
    color_map : str, optional
        Matplotlib colormap name to override default schemes.
        
    alpha : float, optional
        Transparency of grid overlay (0-1). Default is 0.5.
        
    buf : float, optional
        Buffer around grid for plot extent as fraction of grid size. Default is 0.2.
        
    edge : bool, optional
        Whether to draw cell edges. Default is True.
        
    basemap : str, optional
        Basemap style name. Default is 'CartoDB light'.
        
    **kwargs : dict
        Additional parameters specific to data types:
        - land_cover_classes: Dictionary for land cover data
        - buildings: List of building polygons for building_height data
        
    Returns:
    --------
    None
        Displays the plot with matplotlib.
        
    Notes:
    ------
    - Grid is transposed to match geographic orientation
    - Special handling for NaN, zero, and negative values depending on data type
    - Basemap is added using contextily for geographic context
    - Plot extent is automatically calculated from grid vertices
    """
    # Create matplotlib figure and axis
    fig, ax = plt.subplots(figsize=(12, 12))

    # Configure visualization parameters based on data type
    if data_type == 'land_cover':
        # Land cover uses discrete color categories
        land_cover_classes = kwargs.get('land_cover_classes')
        colors = [mcolors.to_rgb(f'#{r:02x}{g:02x}{b:02x}') for r, g, b in land_cover_classes.keys()]
        cmap = mcolors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(range(len(land_cover_classes)+1), cmap.N)
        title = 'Grid Cells with Dominant Land Cover Classes'
        label = 'Land Cover Class'
        tick_labels = list(land_cover_classes.values())
        
    elif data_type == 'building_height':
        # Building height uses continuous colormap with special handling for zero/NaN
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
        # Digital elevation model uses terrain colormap
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
      'CartoDB light no labels': ctx.providers.CartoDB.PositronNoLabels,  # Popular dark option
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

def display_builing_ids_on_map(building_geojson, rectangle_vertices):
    """
    Creates an interactive folium map displaying building footprints with selectable IDs.
    
    This function generates a web map showing building polygons within a circular area
    around the center of the specified rectangle. Each building is labeled with its
    ID and additional information, making it easy to identify specific buildings
    for analysis or selection.
    
    Parameters:
    -----------
    building_geojson : list
        List of GeoJSON feature dictionaries representing building polygons.
        Each feature should have:
        - 'geometry': GeoJSON polygon geometry
        - 'properties': Dictionary with 'id' and optional 'name' fields
        
    rectangle_vertices : list
        List of [lat, lon] coordinate pairs defining the area of interest.
        Used to calculate the map center and intersection area.
        
    Returns:
    --------
    folium.Map
        Interactive folium map object with building polygons and labels.
        
    Notes:
    ------
    - Only buildings intersecting with a 200m radius circle are displayed
    - Building IDs are displayed as selectable text labels
    - Map is automatically centered on the rectangle vertices
    - Popup information includes building ID and name (if available)
    - Building polygons are styled with blue color and semi-transparent fill
    
    Examples:
    ---------
    >>> vertices = [[40.7580, -73.9855], [40.7590, -73.9855], 
    ...             [40.7590, -73.9845], [40.7580, -73.9845]]
    >>> buildings = [{'geometry': {...}, 'properties': {'id': '123', 'name': 'Building A'}}]
    >>> map_obj = display_builing_ids_on_map(buildings, vertices)
    >>> map_obj.save('buildings_map.html')
    """
    # Parse the GeoJSON data
    geojson_data = building_geojson

    # Calculate the center point of the rectangle for map centering
    # Extract all latitudes and longitudes
    lats = [coord[0] for coord in rectangle_vertices]
    lons = [coord[1] for coord in rectangle_vertices]
    
    # Calculate center by averaging min and max values
    center_lat = (min(lats) + max(lats)) / 2
    center_lon = (min(lons) + max(lons)) / 2

    # Create circle polygon for intersection testing (200m radius)
    circle = create_circle_polygon(center_lat, center_lon, 200)

    # Create a map centered on the data
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17)

    # Process each building feature
    # Add building footprints to the map
    for feature in geojson_data:
        # Convert coordinates if needed
        coords = convert_coordinates(feature['geometry']['coordinates'][0])
        building_polygon = Polygon(coords)
        
        # Only process buildings that intersect with the circular area
        # Check if building intersects with circle
        if building_polygon.intersects(circle):
            # Extract building information from properties
            # Get and format building properties
            # building_id = format_building_id(feature['properties'].get('id', 0))
            building_id = str(feature['properties'].get('id', 0))
            building_name = feature['properties'].get('name:en', 
                                                    feature['properties'].get('name', f'Building {building_id}'))
            
            # Create popup content with selectable ID
            popup_content = f"""
            <div>
                Building ID: <span style="user-select: all">{building_id}</span><br>
                Name: {building_name}
            </div>
            """
            
            # Add building polygon to map with popup information
            # Add polygon to map
            folium.Polygon(
                locations=coords,
                popup=folium.Popup(popup_content),
                color='blue',
                weight=2,
                fill=True,
                fill_color='blue',
                fill_opacity=0.2
            ).add_to(m)
            
            # Add building ID label at the polygon centroid
            # Calculate centroid for label placement
            centroid = calculate_centroid(coords)
            
            # Add building ID as a selectable label
            folium.Marker(
                centroid,
                icon=folium.DivIcon(
                    html=f'''
                    <div style="
                        position: relative;
                        font-family: monospace;
                        font-size: 12px;
                        color: black;
                        background-color: rgba(255, 255, 255, 0.9);
                        padding: 5px 8px;
                        margin: -10px -15px;
                        border: 1px solid black;
                        border-radius: 4px;
                        user-select: all;
                        cursor: text;
                        white-space: nowrap;
                        display: inline-block;
                        box-shadow: 0 0 3px rgba(0,0,0,0.2);
                    ">{building_id}</div>
                    ''',
                    class_name="building-label"
                )
            ).add_to(m)

    # Save the map
    return m

def visualize_land_cover_grid_on_map(grid, rectangle_vertices, meshsize, source = 'Urbanwatch', vmin=None, vmax=None, alpha=0.5, buf=0.2, edge=True, basemap='CartoDB light'):
    """
    Visualizes land cover classification grid overlaid on a basemap.
    
    This function creates a map visualization of land cover data using predefined
    color schemes for different land cover classes. Each grid cell is colored
    according to its dominant land cover type and overlaid on a web basemap
    for geographic context.
    
    Parameters:
    -----------
    grid : numpy.ndarray
        2D array containing land cover class indices. Values should correspond
        to indices in the land cover classification system.
        
    rectangle_vertices : list
        List of [lon, lat] coordinate pairs defining the grid boundary corners.
        Should contain exactly 4 vertices in geographic coordinates.
        
    meshsize : float
        Target size of each grid cell in meters. May be adjusted during processing.
        
    source : str, optional
        Source of land cover classification system. Default is 'Urbanwatch'.
        See get_land_cover_classes() for available options.
        
    vmin, vmax : float, optional
        Not used for land cover (discrete categories). Included for API consistency.
        
    alpha : float, optional
        Transparency of grid overlay (0-1). Default is 0.5.
        
    buf : float, optional
        Buffer around grid for plot extent as fraction of grid size. Default is 0.2.
        
    edge : bool, optional
        Whether to draw cell edges. Default is True.
        
    basemap : str, optional
        Basemap style name. Options include:
        - 'CartoDB light' (default)
        - 'CartoDB dark'
        - 'CartoDB voyager'
        Default is 'CartoDB light'.
        
    Returns:
    --------
    None
        Displays the plot and prints information about unique land cover classes.
        
    Notes:
    ------
    - Grid coordinates are calculated using geodetic calculations
    - Land cover classes are mapped to predefined colors
    - Unique classes present in the grid are printed for reference
    - Uses Web Mercator projection (EPSG:3857) for basemap compatibility
    
    Examples:
    ---------
    >>> # Basic land cover visualization
    >>> vertices = [[lon1, lat1], [lon2, lat2], [lon3, lat3], [lon4, lat4]]
    >>> visualize_land_cover_grid_on_map(lc_grid, vertices, 10.0)
    
    >>> # With custom styling
    >>> visualize_land_cover_grid_on_map(lc_grid, vertices, 10.0, 
    ...                                  alpha=0.7, edge=False, 
    ...                                  basemap='CartoDB dark')
    """
    # Initialize geodetic calculator for distance measurements
    geod = initialize_geod()

    # Get land cover class definitions and colors
    land_cover_classes = get_land_cover_classes(source)

    # Extract key vertices for grid calculations
    vertex_0 = rectangle_vertices[0]
    vertex_1 = rectangle_vertices[1]
    vertex_3 = rectangle_vertices[3]

    # Calculate distances between vertices using geodetic calculations
    dist_side_1 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_1[1], vertex_1[0])
    dist_side_2 = calculate_distance(geod, vertex_0[1], vertex_0[0], vertex_3[1], vertex_3[0])

    # Calculate side vectors in geographic coordinates
    side_1 = np.array(vertex_1) - np.array(vertex_0)
    side_2 = np.array(vertex_3) - np.array(vertex_0)

    # Create normalized unit vectors for grid orientation
    u_vec = normalize_to_one_meter(side_1, dist_side_1)
    v_vec = normalize_to_one_meter(side_2, dist_side_2)

    # Set grid origin and calculate optimal grid size
    origin = np.array(rectangle_vertices[0])
    grid_size, adjusted_meshsize = calculate_grid_size(side_1, side_2, u_vec, v_vec, meshsize)

    print(f"Calculated grid size: {grid_size}")
    # print(f"Adjusted mesh size: {adjusted_meshsize}")

    # Set up coordinate transformation for basemap compatibility
    geotiff_crs = CRS.from_epsg(3857)
    transformer = setup_transformer(CRS.from_epsg(4326), geotiff_crs)

    # Generate grid cell coordinates (not currently used but available for advanced processing)
    cell_coords = create_coordinate_mesh(origin, grid_size, adjusted_meshsize, u_vec, v_vec)
    cell_coords_flat = cell_coords.reshape(2, -1).T
    transformed_coords = np.array([transform_coords(transformer, lon, lat) for lat, lon in cell_coords_flat])
    transformed_coords = transformed_coords.reshape(grid_size[::-1] + (2,))

    # print(f"Grid shape: {grid.shape}")

    # Create the visualization using the general plot_grid function
    plot_grid(grid, origin, adjusted_meshsize, u_vec, v_vec, transformer,
              rectangle_vertices, 'land_cover', alpha=alpha, buf=buf, edge=edge, basemap=basemap, land_cover_classes=land_cover_classes)

    # Display information about the land cover classes present in the grid
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
    all_classes = list(land_cover_classes.values())
    unique_classes = list(dict.fromkeys(all_classes))  # Preserve order and remove duplicates

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

def convert_coordinates(coords):
    return coords

def calculate_centroid(coords):
    lat_sum = sum(coord[0] for coord in coords)
    lon_sum = sum(coord[1] for coord in coords)
    return [lat_sum / len(coords), lon_sum / len(coords)]

def calculate_center(features):
    lats = []
    lons = []
    for feature in features:
        coords = feature['geometry']['coordinates'][0]
        for lat, lon in coords:
            lats.append(lat)
            lons.append(lon)
    return sum(lats) / len(lats), sum(lons) / len(lons)

def create_circle_polygon(center_lat, center_lon, radius_meters):
    """Create a circular polygon with given center and radius"""
    # Convert radius from meters to degrees (approximate)
    radius_deg = radius_meters / 111000  # 1 degree ≈ 111km at equator
    
    # Create circle points
    points = []
    for angle in range(361):  # 0 to 360 degrees
        rad = math.radians(angle)
        lat = center_lat + (radius_deg * math.cos(rad))
        lon = center_lon + (radius_deg * math.sin(rad) / math.cos(math.radians(center_lat)))
        points.append((lat, lon))
    return Polygon(points)

def visualize_landcover_grid_on_basemap(landcover_grid, rectangle_vertices, meshsize, source='Standard', alpha=0.6, figsize=(12, 8), 
                                     basemap='CartoDB light', show_edge=False, edge_color='black', edge_width=0.5):
    """Visualizes a land cover grid GeoDataFrame using predefined color schemes.
    
    Args:
        gdf: GeoDataFrame containing grid cells with 'geometry' and 'value' columns
        source: Source of land cover classification (e.g., 'Standard', 'Urbanwatch', etc.)
        title: Title for the plot (default: None)
        alpha: Transparency of the grid overlay (default: 0.6)
        figsize: Figure size in inches (default: (12, 8))
        basemap: Basemap style (default: 'CartoDB light')
        show_edge: Whether to show cell edges (default: True)
        edge_color: Color of cell edges (default: 'black')
        edge_width: Width of cell edges (default: 0.5)
    """
    # Get land cover classes and colors
    land_cover_classes = get_land_cover_classes(source)

    gdf = grid_to_geodataframe(landcover_grid, rectangle_vertices, meshsize)
    
    # Convert RGB tuples to normalized RGB values
    colors = [(r/255, g/255, b/255) for (r,g,b) in land_cover_classes.keys()]
    
    # Create custom colormap
    cmap = ListedColormap(colors)
    
    # Create bounds for discrete colorbar
    bounds = np.arange(len(colors) + 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Convert to Web Mercator
    gdf_web = gdf.to_crs(epsg=3857)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the GeoDataFrame
    gdf_web.plot(column='value',
                 ax=ax,
                 alpha=alpha,
                 cmap=cmap,
                 norm=norm,
                 legend=True,
                 legend_kwds={
                     'label': 'Land Cover Class',
                     'ticks': bounds[:-1] + 0.5,
                     'boundaries': bounds,
                     'format': lambda x, p: list(land_cover_classes.values())[int(x)]
                 },
                 edgecolor=edge_color if show_edge else 'none',
                 linewidth=edge_width if show_edge else 0)
    
    # Add basemap
    basemaps = {
        'CartoDB dark': ctx.providers.CartoDB.DarkMatter,
        'CartoDB light': ctx.providers.CartoDB.Positron,
        'CartoDB voyager': ctx.providers.CartoDB.Voyager,
        'CartoDB light no labels': ctx.providers.CartoDB.PositronNoLabels,
        'CartoDB dark no labels': ctx.providers.CartoDB.DarkMatterNoLabels,
    }
    ctx.add_basemap(ax, source=basemaps[basemap])
    
    # Set title and remove axes
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

def visualize_numerical_grid_on_basemap(grid, rectangle_vertices, meshsize, value_name="value", cmap='viridis', vmin=None, vmax=None, 
                                          alpha=0.6, figsize=(12, 8), basemap='CartoDB light',
                                          show_edge=False, edge_color='black', edge_width=0.5):
    """Visualizes a numerical grid GeoDataFrame (e.g., heights) on a basemap.
    
    Args:
        gdf: GeoDataFrame containing grid cells with 'geometry' and 'value' columns
        title: Title for the plot (default: None)
        cmap: Colormap to use (default: 'viridis')
        vmin: Minimum value for colormap scaling (default: None)
        vmax: Maximum value for colormap scaling (default: None)
        alpha: Transparency of the grid overlay (default: 0.6)
        figsize: Figure size in inches (default: (12, 8))
        basemap: Basemap style (default: 'CartoDB light')
        show_edge: Whether to show cell edges (default: True)
        edge_color: Color of cell edges (default: 'black')
        edge_width: Width of cell edges (default: 0.5)
    """

    gdf = grid_to_geodataframe(grid, rectangle_vertices, meshsize)

    # Convert to Web Mercator
    gdf_web = gdf.to_crs(epsg=3857)
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the GeoDataFrame
    gdf_web.plot(column='value',
                 ax=ax,
                 alpha=alpha,
                 cmap=cmap,
                 vmin=vmin,
                 vmax=vmax,
                 legend=True,
                 legend_kwds={'label': value_name},
                 edgecolor=edge_color if show_edge else 'none',
                 linewidth=edge_width if show_edge else 0)
    
    # Add basemap
    basemaps = {
        'CartoDB dark': ctx.providers.CartoDB.DarkMatter,
        'CartoDB light': ctx.providers.CartoDB.Positron,
        'CartoDB voyager': ctx.providers.CartoDB.Voyager,
        'CartoDB light no labels': ctx.providers.CartoDB.PositronNoLabels,
        'CartoDB dark no labels': ctx.providers.CartoDB.DarkMatterNoLabels,
    }
    ctx.add_basemap(ax, source=basemaps[basemap])
    
    # Set title and remove axes
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

def visualize_numerical_gdf_on_basemap(gdf, value_name="value", cmap='viridis', vmin=None, vmax=None,
                            alpha=0.6, figsize=(12, 8), basemap='CartoDB light',
                            show_edge=False, edge_color='black', edge_width=0.5,
                            input_crs=None):
    """Visualizes a GeoDataFrame with numerical values on a basemap.
    
    Args:
        gdf: GeoDataFrame containing grid cells with 'geometry' and 'value' columns
        value_name: Name of the value column and legend label (default: "value")
        cmap: Colormap to use (default: 'viridis')
        vmin: Minimum value for colormap scaling (default: None)
        vmax: Maximum value for colormap scaling (default: None)
        alpha: Transparency of the grid overlay (default: 0.6)
        figsize: Figure size in inches (default: (12, 8))
        basemap: Basemap style (default: 'CartoDB light')
        show_edge: Whether to show cell edges (default: False)
        edge_color: Color of cell edges (default: 'black')
        edge_width: Width of cell edges (default: 0.5)
        input_crs: Optional CRS to assign if the GeoDataFrame has no CRS. If not provided
            and CRS is missing, the function will attempt to infer WGS84 (EPSG:4326)
            when coordinates look like lon/lat; otherwise it will raise a clear error.
    """
    # Ensure CRS is defined; if missing, assign or infer
    if gdf.crs is None:
        if input_crs is not None:
            gdf = gdf.set_crs(input_crs, allow_override=True)
        else:
            # Try to infer WGS84 if bounds look like lon/lat
            try:
                minx, miny, maxx, maxy = gdf.total_bounds
                looks_like_lonlat = (
                    -180.0 <= minx <= 180.0 and -180.0 <= maxx <= 180.0 and
                    -90.0 <= miny <= 90.0 and -90.0 <= maxy <= 90.0
                )
            except Exception:
                looks_like_lonlat = False
            if looks_like_lonlat:
                gdf = gdf.set_crs("EPSG:4326", allow_override=True)
            else:
                raise ValueError(
                    "Input GeoDataFrame has no CRS. Provide 'input_crs' (e.g., 'EPSG:4326' or 'EPSG:XXXX') "
                    "or set gdf.crs before calling visualize_numerical_gdf_on_basemap."
                )

    # Convert to Web Mercator if not already in that CRS
    if str(gdf.crs) != 'EPSG:3857':
        gdf_web = gdf.to_crs(epsg=3857)
    else:
        gdf_web = gdf
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the GeoDataFrame
    gdf_web.plot(column=value_name,
                 ax=ax,
                 alpha=alpha,
                 cmap=cmap,
                 vmin=vmin,
                 vmax=vmax,
                 legend=True,
                 legend_kwds={'label': value_name},
                 edgecolor=edge_color if show_edge else 'none',
                 linewidth=edge_width if show_edge else 0)
    
    # Add basemap
    basemaps = {
        'CartoDB dark': ctx.providers.CartoDB.DarkMatter,
        'CartoDB light': ctx.providers.CartoDB.Positron,
        'CartoDB voyager': ctx.providers.CartoDB.Voyager,
        'CartoDB light no labels': ctx.providers.CartoDB.PositronNoLabels,
        'CartoDB dark no labels': ctx.providers.CartoDB.DarkMatterNoLabels,
    }
    ctx.add_basemap(ax, source=basemaps[basemap])
    
    # Set title and remove axes
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()

def visualize_point_gdf_on_basemap(point_gdf, value_name='value', **kwargs):
    """Visualizes a point GeoDataFrame on a basemap with colors based on values.
    
    Args:
        point_gdf: GeoDataFrame with point geometries and values
        value_name: Name of the column containing values to visualize (default: 'value')
        **kwargs: Optional visualization parameters including:
            - figsize: Tuple for figure size (default: (12, 8))
            - colormap: Matplotlib colormap name (default: 'viridis')
            - markersize: Size of points (default: 20)
            - alpha: Transparency of points (default: 0.7)
            - vmin: Minimum value for colormap scaling (default: None)
            - vmax: Maximum value for colormap scaling (default: None)
            - title: Plot title (default: None)
            - basemap_style: Contextily basemap style (default: CartoDB.Positron)
            - zoom: Basemap zoom level (default: 15)
            
    Returns:
        matplotlib figure and axis objects
    """
    import matplotlib.pyplot as plt
    import contextily as ctx
    
    # Set default parameters
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
    
    # Update defaults with provided kwargs
    settings = {**defaults, **kwargs}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=settings['figsize'])
    
    # Convert to Web Mercator for basemap compatibility
    point_gdf_web = point_gdf.to_crs(epsg=3857)
    
    # Plot points
    scatter = point_gdf_web.plot(
        column=value_name,
        ax=ax,
        cmap=settings['colormap'],
        markersize=settings['markersize'],
        alpha=settings['alpha'],
        vmin=settings['vmin'],
        vmax=settings['vmax'],
        legend=True,
        legend_kwds={
            'label': value_name,
            'orientation': 'vertical',
            'shrink': 0.8
        }
    )
    
    # Add basemap
    ctx.add_basemap(
        ax,
        source=settings['basemap_style'],
        zoom=settings['zoom']
    )
    
    # Set title if provided
    if settings['title']:
        plt.title(settings['title'])
    
    # Remove axes
    ax.set_axis_off()
    
    # Adjust layout to prevent colorbar cutoff
    plt.tight_layout()
    plt.show()

def create_multi_view_scene(meshes, output_directory="output", projection_type="perspective", distance_factor=1.0):
    """
    Creates multiple rendered views of 3D city meshes from different camera angles.
    
    This function generates a comprehensive set of views including isometric and
    orthographic projections of the 3D city model. Each view is rendered as a
    high-quality image and saved to the specified directory.
    
    Parameters:
    -----------
    meshes : dict
        Dictionary mapping mesh names/IDs to trimesh.Trimesh objects.
        Each mesh represents a different component of the city model.
        
    output_directory : str, optional
        Directory path where rendered images will be saved. Default is "output".
        
    projection_type : str, optional
        Camera projection type. Options:
        - "perspective": Natural perspective projection (default)
        - "orthographic": Technical orthographic projection
        
    distance_factor : float, optional
        Multiplier for camera distance from the scene. Default is 1.0.
        Higher values move camera further away, lower values bring it closer.
        
    Returns:
    --------
    list of tuple
        List of (view_name, filename) pairs for each generated image.
        
    Notes:
    ------
    - Generates 9 different views: 4 isometric + 5 orthographic
    - Isometric views: front-right, front-left, back-right, back-left
    - Orthographic views: top, front, back, left, right
    - Uses PyVista for high-quality rendering with proper lighting
    - Camera positions are automatically calculated based on scene bounds
    - Images are saved as PNG files with high DPI
    
    Technical Details:
    ------------------
    - Scene bounding box is computed from all mesh vertices
    - Camera distance is scaled based on scene diagonal
    - Orthographic projection uses parallel scaling for technical drawings
    - Each view uses optimized lighting for clarity
    
    Examples:
    ---------
    >>> meshes = {'buildings': building_mesh, 'ground': ground_mesh}
    >>> views = create_multi_view_scene(meshes, "renders/", "orthographic", 1.5)
    >>> print(f"Generated {len(views)} views")
    """
    # Precompute PyVista meshes once to avoid repeated conversion per view
    pv_meshes = {}
    for class_id, mesh in meshes.items():
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue
        # PyVista expects a faces array where each face is prefixed by its vertex count (3 for triangles)
        faces = np.hstack([[3, *face] for face in mesh.faces])
        pv_mesh = pv.PolyData(mesh.vertices, faces)
        # Attach per-cell colors if provided
        colors = getattr(mesh.visual, 'face_colors', None)
        if colors is not None:
            colors = np.asarray(colors)
            if colors.size and colors.max() > 1:
                colors = colors / 255.0
            pv_mesh.cell_data['colors'] = colors
        pv_meshes[class_id] = pv_mesh

    # Compute overall bounding box across all meshes without stacking
    min_xyz = np.array([np.inf, np.inf, np.inf], dtype=float)
    max_xyz = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for mesh in meshes.values():
        if mesh is None or len(mesh.vertices) == 0:
            continue
        v = mesh.vertices
        # update mins and maxs
        min_xyz = np.minimum(min_xyz, v.min(axis=0))
        max_xyz = np.maximum(max_xyz, v.max(axis=0))
    bbox = np.vstack([min_xyz, max_xyz])

    # Compute the center and diagonal of the bounding box
    center = (bbox[1] + bbox[0]) / 2
    diagonal = np.linalg.norm(bbox[1] - bbox[0])

    # Adjust distance based on projection type
    if projection_type.lower() == "orthographic":
        distance = diagonal * 5  # Increase distance for orthographic to capture full scene
    else:
        distance = diagonal * 1.8 * distance_factor  # Original distance for perspective

    # Define the isometric viewing angles
    iso_angles = {
        'iso_front_right': (1, 1, 0.7),
        'iso_front_left': (-1, 1, 0.7),
        'iso_back_right': (1, -1, 0.7),
        'iso_back_left': (-1, -1, 0.7)
    }

    # Compute camera positions for isometric views
    camera_positions = {}
    for name, direction in iso_angles.items():
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)
        camera_pos = center + direction * distance
        camera_positions[name] = [camera_pos, center, (0, 0, 1)]

    # Add orthographic views
    ortho_views = {
        'xy_top': [center + np.array([0, 0, distance]), center, (-1, 0, 0)],
        'yz_right': [center + np.array([distance, 0, 0]), center, (0, 0, 1)],
        'xz_front': [center + np.array([0, distance, 0]), center, (0, 0, 1)],
        'yz_left': [center + np.array([-distance, 0, 0]), center, (0, 0, 1)],
        'xz_back': [center + np.array([0, -distance, 0]), center, (0, 0, 1)]
    }
    camera_positions.update(ortho_views)

    images = []
    for view_name, camera_pos in camera_positions.items():
        # Create new plotter for each view (avoid notebook backend in headless/CI)
        plotter = pv.Plotter(off_screen=True)
        
        # Set the projection type
        if projection_type.lower() == "orthographic":
            plotter.enable_parallel_projection()
            # Set parallel scale to ensure the whole scene is visible
            plotter.camera.parallel_scale = diagonal * 0.4 * distance_factor  # Adjust this factor as needed

        elif projection_type.lower() != "perspective":
            print(f"Warning: Unknown projection_type '{projection_type}'. Using perspective projection.")

        # Add each precomputed mesh to the scene
        for class_id, pv_mesh in pv_meshes.items():
            has_colors = 'colors' in pv_mesh.cell_data
            plotter.add_mesh(
                pv_mesh,
                rgb=True,
                scalars='colors' if has_colors else None
            )

        # Set camera position for this view
        plotter.camera_position = camera_pos

        # Save screenshot
        filename = f'{output_directory}/city_view_{view_name}.png'
        plotter.screenshot(filename)
        images.append((view_name, filename))
        plotter.close()

    return images

def visualize_voxcity_multi_view(voxel_array, meshsize, **kwargs):
    """
    Creates comprehensive 3D visualizations of voxel city data with multiple viewing angles.
    
    This is the primary function for generating publication-quality renderings of voxel
    city models. It converts voxel data to 3D meshes, optionally overlays simulation
    results, and produces multiple rendered views from different camera positions.
    
    Parameters:
    -----------
    voxel_array : numpy.ndarray
        3D array containing voxel class IDs. Shape should be (x, y, z).
        
    meshsize : float
        Physical size of each voxel in meters.
        
    **kwargs : dict
        Optional visualization parameters:
        
        Color and Style:
        - voxel_color_map (str): Color scheme name, default 'default'
        - output_directory (str): Directory for output files, default 'output'
        - output_file_name (str): Base name for exported files
        
        Simulation Overlay:
        - sim_grid (numpy.ndarray): 2D simulation results to overlay
        - dem_grid (numpy.ndarray): Digital elevation model for height reference
        - view_point_height (float): Height offset for simulation surface, default 1.5m
        - colormap (str): Matplotlib colormap for simulation data, default 'viridis'
        - vmin, vmax (float): Color scale limits for simulation data
        
        Camera and Rendering:
        - projection_type (str): 'perspective' or 'orthographic', default 'perspective'
        - distance_factor (float): Camera distance multiplier, default 1.0
        - window_width, window_height (int): Render resolution, default 1024x768
        
        Output Control:
        - show_views (bool): Whether to display rendered views, default True
        - save_obj (bool): Whether to export OBJ mesh files, default False
        
    Returns:
    --------
    None
        Displays rendered views and optionally saves files to disk.
        
    Notes:
    ------
    - Automatically configures PyVista for headless rendering
    - Generates meshes for each voxel class with appropriate colors
    - Creates colorbar for simulation data if provided
    - Produces 9 different camera views (4 isometric + 5 orthographic)
    - Exports mesh files in OBJ format if requested
    
    Technical Requirements:
    -----------------------
    - Requires Xvfb for headless rendering on Linux systems
    - Uses PyVista for high-quality 3D rendering
    - Simulation data is interpolated onto elevated surface mesh
    
    Examples:
    ---------
    >>> # Basic visualization
    >>> visualize_voxcity_multi_view(voxel_array, meshsize=2.0)
    
    >>> # With simulation results overlay
    >>> visualize_voxcity_multi_view(
    ...     voxel_array, 2.0,
    ...     sim_grid=temperature_data,
    ...     dem_grid=elevation_data,
    ...     colormap='plasma',
    ...     output_file_name='temperature_analysis'
    ... )
    
    >>> # High-resolution orthographic technical drawings
    >>> visualize_voxcity_multi_view(
    ...     voxel_array, 2.0,
    ...     projection_type='orthographic',
    ...     window_width=2048,
    ...     window_height=1536,
    ...     save_obj=True
    ... )
    """
    # Configure PyVista settings for high-quality rendering
    pv.set_plot_theme('document')
    pv.global_theme.background = 'white'
    pv.global_theme.window_size = [1024, 768]
    pv.global_theme.jupyter_backend = 'static'
    
    # Parse visualization parameters from kwargs
    voxel_color_map = kwargs.get("voxel_color_map", 'default')
    vox_dict = get_voxel_color_map(voxel_color_map)
    output_directory = kwargs.get("output_directory", 'output')
    base_filename = kwargs.get("output_file_name", None)
    sim_grid = kwargs.get("sim_grid", None)
    dem_grid_ori = kwargs.get("dem_grid", None)
    
    # Normalize DEM grid to start from zero elevation
    if dem_grid_ori is not None:
        dem_grid = dem_grid_ori - np.min(dem_grid_ori)
        
    # Simulation overlay parameters
    z_offset = kwargs.get("view_point_height", 1.5)
    cmap_name = kwargs.get("colormap", "viridis")
    vmin = kwargs.get("vmin", np.nanmin(sim_grid) if sim_grid is not None else None)
    vmax = kwargs.get("vmax", np.nanmax(sim_grid) if sim_grid is not None else None)
    
    # Camera and rendering parameters
    projection_type = kwargs.get("projection_type", "perspective")
    distance_factor = kwargs.get("distance_factor", 1.0)
    
    # Output control parameters
    save_obj = kwargs.get("save_obj", False)
    show_views = kwargs.get("show_views", True)
    # Disable rendering in CI/headless to avoid VTK segfaults
    if os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true' or os.getenv('CI', 'false').lower() == 'true':
        show_views = False

    # Set up headless rendering environment for PyVista only if we will render
    if show_views and sys.platform.startswith('linux'):
        try:
            import pyvista as _pv
            _pv.start_xvfb()
        except Exception:
            os.system('Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &')
            os.environ['DISPLAY'] = ':99'

    # Create 3D meshes from voxel data
    print("Creating voxel meshes...")
    meshes = create_city_meshes(voxel_array, vox_dict, meshsize=meshsize)

    # Add simulation results as elevated surface mesh if provided
    if sim_grid is not None and dem_grid is not None:
        print("Creating sim_grid surface mesh...")
        sim_mesh = create_sim_surface_mesh(
            sim_grid, dem_grid,
            meshsize=meshsize,
            z_offset=z_offset,
            cmap_name=cmap_name,
            vmin=vmin,
            vmax=vmax
        )
        if sim_mesh is not None:
            meshes["sim_surface"] = sim_mesh
            
        # Create and display colorbar for simulation data
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap_name)
        
        fig, ax = plt.subplots(figsize=(6, 1))
        plt.colorbar(scalar_map, cax=ax, orientation='horizontal')
        plt.tight_layout()
        plt.show()

    # # Export mesh files if requested
    # if base_filename is not None:
    #     print(f"Exporting files to '{base_filename}.*' ...")
    #     os.makedirs(output_directory, exist_ok=True)
    #     export_meshes(meshes, output_directory, base_filename)

    # Export OBJ mesh files if requested
    if save_obj:
        output_directory = kwargs.get('output_directory', 'output')
        output_file_name = kwargs.get('output_file_name', 'voxcity_mesh')
        obj_path, mtl_path = save_obj_from_colored_mesh(meshes, output_directory, output_file_name)
        print(f"Saved mesh files to:\n  {obj_path}\n  {mtl_path}")

    # Generate and display multiple camera views
    if show_views:  
        print("Creating multiple views...")        
        os.makedirs(output_directory, exist_ok=True)
        image_files = create_multi_view_scene(meshes, output_directory=output_directory, projection_type=projection_type, distance_factor=distance_factor)

        # Display each rendered view
        for view_name, img_file in image_files:
            plt.figure(figsize=(24, 16))
            img = plt.imread(img_file)
            plt.imshow(img)
            plt.title(view_name.replace('_', ' ').title(), pad=20)
            plt.axis('off')
            plt.show()
            plt.close()
    
def visualize_voxcity_multi_view_with_multiple_sim_grids(voxel_array, meshsize, sim_configs, **kwargs):
    """
    Create multiple views of the voxel city data with multiple simulation grids.
    
    Args:
        voxel_array: 3D numpy array containing voxel data
        meshsize: Size of each voxel/cell
        sim_configs: List of dictionaries, each containing configuration for a simulation grid:
            {
                'sim_grid': 2D numpy array of simulation values,
                'z_offset': height offset in meters (default: 1.5),
                'cmap_name': colormap name (default: 'viridis'),
                'vmin': minimum value for colormap (optional),
                'vmax': maximum value for colormap (optional),
                'label': label for the colorbar (optional)
            }
        **kwargs: Additional arguments including:
            - vox_dict: Dictionary mapping voxel values to colors
            - output_directory: Directory to save output images
            - output_file_name: Base filename for exports
            - dem_grid: DEM grid for height information
            - projection_type: 'perspective' or 'orthographic'
            - distance_factor: Factor to adjust camera distance
    """
    os.system('Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &')
    os.environ['DISPLAY'] = ':99'

    # Configure PyVista settings
    pv.set_plot_theme('document')
    pv.global_theme.background = 'white'
    window_width = kwargs.get("window_width", 1024)
    window_height = kwargs.get("window_height", 768)
    pv.global_theme.window_size = [window_width, window_height]
    pv.global_theme.jupyter_backend = 'static'

    # Parse general kwargs
    voxel_color_map = kwargs.get("voxel_color_map", 'default')
    vox_dict = get_voxel_color_map(voxel_color_map)
    output_directory = kwargs.get("output_directory", 'output')
    base_filename = kwargs.get("output_file_name", None)
    dem_grid_ori = kwargs.get("dem_grid", None)
    projection_type = kwargs.get("projection_type", "perspective")
    distance_factor = kwargs.get("distance_factor", 1.0)
    show_views = kwargs.get("show_views", True)
    save_obj = kwargs.get("save_obj", False)

    if dem_grid_ori is not None:
        dem_grid = dem_grid_ori - np.min(dem_grid_ori)
    
    # Create meshes
    print("Creating voxel meshes...")
    meshes = create_city_meshes(voxel_array, vox_dict, meshsize=meshsize)

    # Process each simulation grid
    for i, config in enumerate(sim_configs):
        sim_grid = config['sim_grid']
        if sim_grid is None or dem_grid is None:
            continue

        z_offset = config.get('z_offset', 1.5)
        cmap_name = config.get('cmap_name', 'viridis')
        vmin = config.get('vmin', np.nanmin(sim_grid))
        vmax = config.get('vmax', np.nanmax(sim_grid))
        label = config.get('label', f'Simulation {i+1}')

        print(f"Creating sim_grid surface mesh for {label}...")
        sim_mesh = create_sim_surface_mesh(
            sim_grid, dem_grid,
            meshsize=meshsize,
            z_offset=z_offset,
            cmap_name=cmap_name,
            vmin=vmin,
            vmax=vmax
        )
        
        if sim_mesh is not None:
            meshes[f"sim_surface_{i}"] = sim_mesh
            
            # Create colorbar for this simulation
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap_name)
            
            fig, ax = plt.subplots(figsize=(6, 1))
            plt.colorbar(scalar_map, cax=ax, orientation='horizontal', label=label)
            plt.tight_layout()
            plt.show()

    # Export if filename provided
    if base_filename is not None:
        print(f"Exporting files to '{base_filename}.*' ...")
        os.makedirs(output_directory, exist_ok=True)
        export_meshes(meshes, output_directory, base_filename)

    if show_views:
        # Create and save multiple views
        print("Creating multiple views...")        
        os.makedirs(output_directory, exist_ok=True)
        image_files = create_multi_view_scene(
            meshes, 
            output_directory=output_directory,
            projection_type=projection_type,
            distance_factor=distance_factor
        )

        # Display each view separately
        for view_name, img_file in image_files:
            plt.figure(figsize=(24, 16))
            img = plt.imread(img_file)
            plt.imshow(img)
            plt.title(view_name.replace('_', ' ').title(), pad=20)
            plt.axis('off')
            plt.show()
            plt.close()

    # After creating the meshes and before visualization
    if save_obj:
        output_directory = kwargs.get('output_directory', 'output')
        output_file_name = kwargs.get('output_file_name', 'voxcity_mesh')
        obj_path, mtl_path = save_obj_from_colored_mesh(meshes, output_directory, output_file_name)
        print(f"Saved mesh files to:\n  {obj_path}\n  {mtl_path}")

def visualize_voxcity_with_sim_meshes(voxel_array, meshsize, custom_meshes=None, **kwargs):
    """
    Creates 3D visualizations of voxel city data with custom simulation mesh overlays.
    
    This advanced visualization function allows replacement of specific voxel classes
    with custom simulation result meshes. It's particularly useful for overlaying
    detailed simulation results (like computational fluid dynamics, thermal analysis,
    or environmental factors) onto specific building or infrastructure components.
    
    The function supports simulation meshes with metadata containing numerical values
    that can be visualized using color mapping, making it ideal for displaying
    spatially-varying simulation results on building surfaces or other urban elements.
    
    Parameters:
    -----------
    voxel_array : np.ndarray
        3D array of voxel values representing the base city model.
        Shape should be (x, y, z) where each element is a voxel class ID.
        
    meshsize : float
        Size of each voxel in meters, used for coordinate scaling.
        
    custom_meshes : dict, optional
        Dictionary mapping voxel class IDs to custom trimesh.Trimesh objects.
        Example: {-3: building_simulation_mesh, -2: vegetation_mesh}
        These meshes will replace the standard voxel representation for visualization.
        Default is None.
        
    **kwargs:
        Extensive configuration options organized by category:
        
        Base Visualization:
        - vox_dict (dict): Dictionary mapping voxel class IDs to colors
        - output_directory (str): Directory for saving output files
        - output_file_name (str): Base filename for exported meshes
        
        Simulation Result Display:
        - value_name (str): Name of metadata field containing simulation values
        - colormap (str): Matplotlib colormap name for simulation results
        - vmin, vmax (float): Color scale limits for simulation data
        - colorbar_title (str): Title for the simulation result colorbar
        - nan_color (str/tuple): Color for NaN/invalid simulation values, default 'gray'
        
        Ground Surface Overlay:
        - sim_grid (np.ndarray): 2D array with ground-level simulation values
        - dem_grid (np.ndarray): Digital elevation model for terrain height
        - view_point_height (float): Height offset for ground simulation surface
        
        Camera and Rendering:
        - projection_type (str): 'perspective' or 'orthographic', default 'perspective'
        - distance_factor (float): Camera distance multiplier, default 1.0
        - window_width, window_height (int): Render resolution
        
        Output Control:
        - show_views (bool): Whether to display rendered views, default True
        - save_obj (bool): Whether to export OBJ mesh files, default False
        
    Returns:
    --------
    list
        List of (view_name, image_file_path) tuples for generated views.
        Only returned if show_views=True.
        
    Notes:
    ------
    Simulation Mesh Requirements:
    - Custom meshes should have simulation values stored in mesh.metadata[value_name]
    - Values can include NaN for areas without valid simulation data
    - Mesh geometry should align with the voxel grid coordinate system
    
    Color Mapping:
    - Simulation values are mapped to colors using the specified colormap
    - NaN values are rendered in the specified nan_color
    - A colorbar is automatically generated and displayed
    
    Technical Implementation:
    - Uses PyVista for high-quality 3D rendering
    - Supports both individual mesh coloring and ground surface overlays
    - Automatically handles coordinate system transformations
    - Generates multiple camera views for comprehensive visualization
    
    Examples:
    ---------
    >>> # Basic usage with building simulation results
    >>> building_mesh = trimesh.load('building_with_cfd_results.ply')
    >>> building_mesh.metadata = {'temperature': temperature_values}
    >>> custom_meshes = {-3: building_mesh}  # -3 is building class ID
    >>> 
    >>> visualize_voxcity_with_sim_meshes(
    ...     voxel_array, meshsize=2.0,
    ...     custom_meshes=custom_meshes,
    ...     value_name='temperature',
    ...     colormap='plasma',
    ...     colorbar_title='Temperature (°C)',
    ...     vmin=15, vmax=35
    ... )
    
    >>> # With ground-level wind simulation overlay
    >>> wind_mesh = create_wind_simulation_mesh(wind_data)
    >>> visualize_voxcity_with_sim_meshes(
    ...     voxel_array, 2.0,
    ...     custom_meshes={-3: wind_mesh},
    ...     value_name='wind_speed',
    ...     sim_grid=ground_wind_grid,
    ...     dem_grid=elevation_grid,
    ...     colormap='viridis',
    ...     projection_type='orthographic',
    ...     save_obj=True
    ... )
    
    >>> # Multiple simulation types with custom styling
    >>> meshes = {
    ...     -3: building_thermal_mesh,  # Buildings with thermal data
    ...     -2: vegetation_co2_mesh     # Vegetation with CO2 absorption
    ... }
    >>> visualize_voxcity_with_sim_meshes(
    ...     voxel_array, 2.0,
    ...     custom_meshes=meshes,
    ...     value_name='co2_flux',
    ...     colormap='RdYlBu_r',
    ...     nan_color='lightgray',
    ...     distance_factor=1.5,
    ...     output_file_name='co2_analysis'
    ... )
    
    See Also:
    ---------
    - visualize_building_sim_results(): Specialized function for building simulations
    - visualize_voxcity_multi_view(): Basic voxel visualization without custom meshes
    - create_multi_view_scene(): Lower-level rendering function
    """
    # Setup offscreen rendering only when needed and supported
    if sys.platform.startswith('linux'):
        if 'DISPLAY' not in os.environ:
            os.system('Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &')
            os.environ['DISPLAY'] = ':99'

    # Configure PyVista settings
    pv.set_plot_theme('document')
    pv.global_theme.background = 'white'
    window_width = kwargs.get("window_width", 1024)
    window_height = kwargs.get("window_height", 768)
    pv.global_theme.window_size = [window_width, window_height]
    pv.global_theme.jupyter_backend = 'static'
    
    # Parse kwargs
    voxel_color_map = kwargs.get("voxel_color_map", 'default')
    vox_dict = get_voxel_color_map(voxel_color_map)
    output_directory = kwargs.get("output_directory", 'output')
    base_filename = kwargs.get("output_file_name", None)
    sim_grid = kwargs.get("sim_grid", None)
    dem_grid_ori = kwargs.get("dem_grid", None)
    if dem_grid_ori is not None:
        dem_grid = dem_grid_ori - np.min(dem_grid_ori)
    z_offset = kwargs.get("view_point_height", 1.5)
    cmap_name = kwargs.get("colormap", "viridis")
    vmin = kwargs.get("vmin", None)
    vmax = kwargs.get("vmax", None)
    projection_type = kwargs.get("projection_type", "perspective")
    distance_factor = kwargs.get("distance_factor", 1.0)
    colorbar_title = kwargs.get("colorbar_title", "")
    value_name = kwargs.get("value_name", None)
    nan_color = kwargs.get("nan_color", "gray")
    show_views = kwargs.get("show_views", True)
    save_obj = kwargs.get("save_obj", False)
    include_classes = kwargs.get("include_classes", None)
    exclude_classes = kwargs.get("exclude_classes", None)
    copy_custom_mesh = kwargs.get("copy_custom_mesh", False)
    
    if value_name is None:
        print("Set value_name")

    # Create meshes from voxel data
    print("Creating voxel meshes...")
    # Skip generating voxel meshes for classes that will be replaced by custom meshes
    if exclude_classes is None and custom_meshes is not None:
        exclude_classes = list(custom_meshes.keys())
    meshes = create_city_meshes(
        voxel_array,
        vox_dict,
        meshsize=meshsize,
        include_classes=include_classes,
        exclude_classes=exclude_classes,
    )
    
    # Replace specific voxel class meshes with custom simulation meshes
    if custom_meshes is not None:
        for class_id, custom_mesh in custom_meshes.items():
            # Apply coloring to custom meshes if they have metadata values
            if hasattr(custom_mesh, 'metadata') and value_name in custom_mesh.metadata:
                # Create a colored copy of the mesh for visualization
                import matplotlib.cm as cm
                import matplotlib.colors as mcolors
                
                # Get values from metadata
                values = np.asarray(custom_mesh.metadata[value_name])
                
                # Set vmin/vmax if not provided
                finite_mask = np.isfinite(values)
                if not np.any(finite_mask):
                    local_vmin = 0.0 if vmin is None else vmin
                    local_vmax = 1.0 if vmax is None else vmax
                else:
                    local_vmin = vmin if vmin is not None else float(np.nanmin(values[finite_mask]))
                    local_vmax = vmax if vmax is not None else float(np.nanmax(values[finite_mask]))

                # Create colors
                cmap = cm.get_cmap(cmap_name)
                norm = mcolors.Normalize(vmin=local_vmin, vmax=local_vmax)
                
                # Handle NaN values with custom color
                face_colors = np.zeros((len(values), 4), dtype=float)
                
                # Convert string color to RGBA if needed
                if isinstance(nan_color, str):
                    import matplotlib.colors as mcolors
                    nan_rgba = np.array(mcolors.to_rgba(nan_color))
                else:
                    # Assume it's already a tuple/list of RGBA values
                    nan_rgba = np.array(nan_color)
                
                # Apply colors: NaN values get nan_color, others get colormap colors
                nan_mask = ~finite_mask
                face_colors[~nan_mask] = cmap(norm(values[~nan_mask]))
                face_colors[nan_mask] = nan_rgba
                
                # Create a copy with colors
                vis_mesh = custom_mesh.copy() if copy_custom_mesh else custom_mesh
                vis_mesh.visual.face_colors = face_colors

                # Prepare the colormap and create colorbar if views will be shown
                if show_views:
                    norm = mcolors.Normalize(vmin=local_vmin, vmax=local_vmax)
                    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap_name)
                    fig, ax = plt.subplots(figsize=(6, 1))
                    cbar = plt.colorbar(scalar_map, cax=ax, orientation='horizontal')
                    if colorbar_title:
                        cbar.set_label(colorbar_title)
                    plt.tight_layout()
                    plt.show()
                    plt.close(fig)
                
                if class_id in meshes:
                    print(f"Replacing voxel class {class_id} with colored custom simulation mesh")
                    meshes[class_id] = vis_mesh
                else:
                    print(f"Adding colored custom simulation mesh for class {class_id}")
                    meshes[class_id] = vis_mesh
            else:
                # No metadata values, use the mesh as is
                if class_id in meshes:
                    print(f"Replacing voxel class {class_id} with custom simulation mesh")
                    meshes[class_id] = custom_mesh
                else:
                    print(f"Adding custom simulation mesh for class {class_id}")
                    meshes[class_id] = custom_mesh

    # Create sim_grid surface mesh if provided
    if sim_grid is not None and dem_grid is not None:
        print("Creating sim_grid surface mesh...")
        
        # If vmin/vmax not provided, use actual min/max of the valid sim data
        if vmin is None:
            vmin = np.nanmin(sim_grid)
        if vmax is None:
            vmax = np.nanmax(sim_grid)
            
        sim_mesh = create_sim_surface_mesh(
            sim_grid, dem_grid,
            meshsize=meshsize,
            z_offset=z_offset,
            cmap_name=cmap_name,
            vmin=vmin,
            vmax=vmax,
            nan_color=nan_color  # Pass nan_color to the mesh creation
        )
        if sim_mesh is not None:
            meshes["sim_surface"] = sim_mesh
            
        # # Prepare the colormap and create colorbar
        # norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        # scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap_name)
        
        # # Create a figure and axis for the colorbar but don't display
        # fig, ax = plt.subplots(figsize=(6, 1))
        # cbar = plt.colorbar(scalar_map, cax=ax, orientation='horizontal')
        # if colorbar_title:
        #     cbar.set_label(colorbar_title)
        # plt.tight_layout()
        # plt.show()

    # # Export if filename provided
    # if base_filename is not None:
    #     print(f"Exporting files to '{base_filename}.*' ...")
    #     # Create output directory if it doesn't exist
    #     os.makedirs(output_directory, exist_ok=True)
    #     export_meshes(meshes, output_directory, base_filename) 
 
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # After creating the meshes and before visualization
    if save_obj:
        output_directory = kwargs.get('output_directory', 'output')
        output_file_name = kwargs.get('output_file_name', 'voxcity_mesh')
        # Default: do NOT quantize to preserve exact face colors like in images
        max_materials = kwargs.get('max_materials', None)
        if max_materials is None:
            obj_path, mtl_path = save_obj_from_colored_mesh(meshes, output_directory, output_file_name)
        else:
            obj_path, mtl_path = save_obj_from_colored_mesh(meshes, output_directory, output_file_name, max_materials=max_materials)
        print(f"Saved mesh files to:\n  {obj_path}\n  {mtl_path}")    

    if show_views:
        # Create and save multiple views
        print("Creating multiple views...")       
        image_files = create_multi_view_scene(meshes, output_directory=output_directory, 
                                         projection_type=projection_type, 
                                         distance_factor=distance_factor)

        # Display each view separately
        for view_name, img_file in image_files:
            plt.figure(figsize=(24, 16))
            img = plt.imread(img_file)
            plt.imshow(img)
            plt.title(view_name.replace('_', ' ').title(), pad=20)
            plt.axis('off')
            plt.show()
            plt.close()       


def visualize_building_sim_results(voxel_array, meshsize, building_sim_mesh, **kwargs):
    """
    Visualize building simulation results by replacing building meshes in the original model.
    
    This is a specialized wrapper around visualize_voxcity_with_sim_meshes that specifically
    targets building simulation meshes (assuming building class ID is -3).
    
    Parameters
    ----------
    voxel_array : np.ndarray
        3D array of voxel values.
    meshsize : float
        Size of each voxel in meters.
    building_sim_mesh : trimesh.Trimesh
        Simulation result mesh for buildings with values stored in metadata.
    **kwargs:
        Same parameters as visualize_voxcity_with_sim_meshes.
        Additional parameters:
        value_name : str
            Name of the field in metadata containing values to visualize (default: 'svf_values')
        nan_color : str or tuple
            Color for NaN values (default: 'gray')
        
    Returns
    -------
    list
        List of (view_name, image_file_path) tuples for the generated views.
    """
    # Building class ID is typically -3 in voxcity
    building_class_id = kwargs.get("building_class_id", -3)
    
    # Create custom meshes dictionary with the building simulation mesh
    custom_meshes = {building_class_id: building_sim_mesh}

    # Performance: skip voxel mesh generation for building class by default
    if "include_classes" not in kwargs and "exclude_classes" not in kwargs:
        kwargs["exclude_classes"] = [building_class_id]

    # Memory-safety: do not duplicate the provided mesh unless requested
    kwargs.setdefault("copy_custom_mesh", False)
    
    # Add colorbar title if not provided
    if "colorbar_title" not in kwargs:
        # Try to guess a title based on the mesh name/type
        if hasattr(building_sim_mesh, 'name') and building_sim_mesh.name:
            kwargs["colorbar_title"] = building_sim_mesh.name
        else:
            # Use value_field name as fallback
            value_name = kwargs.get("value_name", "svf_values")
            pretty_name = value_name.replace('_', ' ').title()
            kwargs["colorbar_title"] = pretty_name
    
    # Call the more general visualization function
    visualize_voxcity_with_sim_meshes(
        voxel_array,
        meshsize,
        custom_meshes=custom_meshes,
        **kwargs
    )


def _rgb_tuple_to_plotly_color(rgb_tuple):
    """
    Convert [R, G, B] or (R, G, B) with 0-255 range to plotly 'rgb(r,g,b)' string.
    """
    try:
        r, g, b = rgb_tuple
        r = int(max(0, min(255, r)))
        g = int(max(0, min(255, g)))
        b = int(max(0, min(255, b)))
        return f"rgb({r},{g},{b})"
    except Exception:
        return "rgb(128,128,128)"


def visualize_building_sim_results_plotly(
    voxel_array,
    meshsize,
    building_sim_mesh,
    **kwargs
):
    """
    Interactive Plotly visualization of voxels with an overlaid building simulation mesh.

    This function reuses visualize_voxcity_plotly to render voxel cubes and adds a
    Plotly Mesh3d trace for a provided building simulation mesh (e.g., SVF, temperature).

    Parameters (kwargs)
    -------------------
    classes : list[int] or None
        Classes to render for voxel cubes. Default: all non-zero classes present.
    voxel_color_map : str or dict
        Scheme name understood by get_voxel_color_map or explicit mapping {class_id: [R,G,B]}.
    downsample : int or None
        Stride for voxel cubes. 1 means no downsampling.
    cubes_opacity : float
        Opacity for voxel cubes (default 0.95 for buildings, 0.6 for others via function's per-class logic; here default 0.9).
    title : str
        Figure title.
    width, height : int
        Figure size.
    value_name : str
        Metadata field name on building_sim_mesh storing per-face values (default 'svf_values').
    colormap : str
        Matplotlib colormap name for the simulation values (default 'viridis').
    vmin, vmax : float or None
        Value range for color mapping. If None, computed from finite values.
    nan_color : str
        Color name for NaN values (default 'gray').
    building_opacity : float
        Opacity for the simulation mesh (default 1.0).
    shaded : bool
        If True, apply lighting-based shading to the simulation mesh. Default False (unlit colors).
    render_voxel_buildings : bool
        If True, also render voxel buildings (-3) from voxcity_grid. Default False (hide voxel buildings
        so only simulation mesh buildings are visible).
    show : bool, return_fig : bool
        Standard display controls.
    """
    classes = kwargs.get('classes')
    voxel_color_map = kwargs.get('voxel_color_map', 'default')
    downsample = kwargs.get('downsample')
    title = kwargs.get('title', None)
    width = kwargs.get('width', 1000)
    height = kwargs.get('height', 800)
    cubes_opacity = kwargs.get('cubes_opacity', 0.9)
    show = kwargs.get('show', True)
    return_fig = kwargs.get('return_fig', False)
    render_voxel_buildings = kwargs.get('render_voxel_buildings', False)

    # Determine classes for voxel cubes and exclude buildings (-3) by default
    if classes is None:
        classes_all = np.unique(voxel_array[voxel_array != 0]).tolist()
    else:
        classes_all = list(classes)
    classes_cubes = classes_all if render_voxel_buildings else [c for c in classes_all if int(c) != -3]

    # Render voxel cubes background (or blank scene if nothing to render)
    if len(classes_cubes) > 0:
        fig = visualize_voxcity_plotly(
            voxel_array,
            meshsize,
            classes=classes_cubes,
            voxel_color_map=voxel_color_map,
            opacity=cubes_opacity,
            downsample=downsample,
            title=title or "Building Simulation (Plotly)",
            width=width,
            height=height,
            show=False,
            return_fig=True,
        )
    else:
        fig = go.Figure()
        fig.update_layout(
            title=title or "Building Simulation (Plotly)",
            width=width,
            height=height,
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
            )
        )

    # Nothing to overlay
    if building_sim_mesh is None or getattr(building_sim_mesh, 'vertices', None) is None:
        if show:
            fig.show()
        if return_fig:
            return fig
        return None

    # Extract geometry
    V = np.asarray(building_sim_mesh.vertices)
    F = np.asarray(building_sim_mesh.faces)
    values = None
    value_name = kwargs.get('value_name', 'svf_values')
    if hasattr(building_sim_mesh, 'metadata') and isinstance(building_sim_mesh.metadata, dict):
        values = building_sim_mesh.metadata.get(value_name)
    if values is not None:
        values = np.asarray(values)

    # Compute per-face scalar to avoid color interpolation across edges
    face_vals = None
    if values is not None and values.size == len(F):
        # Already per-face
        face_vals = values.astype(float)
    elif values is not None and values.size == len(V):
        # Average the three vertex values per face
        vals_v = values.astype(float)
        face_vals = np.nanmean(vals_v[F], axis=1)

    # Map to colors
    cmap_name = kwargs.get('colormap', 'viridis')
    vmin = kwargs.get('vmin')
    vmax = kwargs.get('vmax')
    nan_color = kwargs.get('nan_color', 'gray')
    building_opacity = kwargs.get('building_opacity', 1.0)

    facecolor = None
    if face_vals is not None:
        # Compute range
        finite = np.isfinite(face_vals)
        if vmin is None:
            vmin = float(np.nanmin(face_vals[finite])) if np.any(finite) else 0.0
        if vmax is None:
            vmax = float(np.nanmax(face_vals[finite])) if np.any(finite) else 1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(cmap_name)
        # Colors per face (constant on each triangle)
        colors_rgba = np.zeros((len(F), 4), dtype=float)
        colors_rgba[finite] = cmap(norm(face_vals[finite]))
        nan_rgba = np.array(mcolors.to_rgba(nan_color))
        colors_rgba[~finite] = nan_rgba
        facecolor = [
            f"rgb({int(255*c[0])},{int(255*c[1])},{int(255*c[2])})" for c in colors_rgba
        ]

    # Lighting (disable shading by default for true color rendering)
    shaded = kwargs.get('shaded', False)
    if shaded:
        lighting = dict(ambient=0.35, diffuse=1.0, specular=0.4, roughness=0.5, fresnel=0.1)
        flat = False
    else:
        # Unlit: make colors independent of lighting
        lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0, fresnel=0.0)
        flat = False
    cx = float((V[:,0].min() + V[:,0].max()) * 0.5)
    cy = float((V[:,1].min() + V[:,1].max()) * 0.5)
    cz = float((V[:,2].min() + V[:,2].max()) * 0.5)
    lx = cx + (V[:,0].max() - V[:,0].min() + meshsize) * 0.9
    ly = cy + (V[:,1].max() - V[:,1].min() + meshsize) * 0.6
    lz = cz + (V[:,2].max() - V[:,2].min() + meshsize) * 1.4

    fig.add_trace(
        go.Mesh3d(
            x=V[:,0], y=V[:,1], z=V[:,2],
            i=F[:,0], j=F[:,1], k=F[:,2],
            facecolor=facecolor if facecolor is not None else None,
            color=None if facecolor is not None else 'rgb(200,200,200)',
            opacity=float(building_opacity),
            flatshading=flat,
            lighting=lighting,
            lightposition=dict(x=lx, y=ly, z=lz),
            name=value_name if facecolor is not None else 'building_mesh'
        )
    )

    if show:
        fig.show()
    if return_fig:
        return fig
    return None


def visualize_ground_sim_results_plotly(
    voxel_array,
    meshsize,
    sim_grid,
    dem_grid,
    **kwargs
):
    """
    Interactive Plotly visualization of voxels with an elevated ground-level simulation surface.

    This renders voxel cubes for the city model and overlays a triangulated surface mesh
    created from a 2D ground simulation grid (e.g., temperature, wind) elevated by the DEM
    plus a configurable offset. The surface is colored using a matplotlib colormap.

    Parameters (kwargs)
    -------------------
    classes : list[int] or None
        Classes to render for voxel cubes. Default: all non-zero classes present.
    voxel_color_map : str or dict
        Scheme name understood by get_voxel_color_map or explicit mapping {class_id: [R,G,B]}.
    downsample : int or None
        Stride for voxel cubes. 1 means no downsampling.
    cubes_opacity : float
        Opacity for voxel cubes (default 0.9).
    title : str
        Figure title.
    width, height : int
        Figure size.
    view_point_height : float
        Height offset above DEM for the simulation surface (default 1.5m).
    colormap : str
        Matplotlib colormap name for the simulation values (default 'viridis').
    vmin, vmax : float or None
        Value range for color mapping. If None, computed from finite values.
    sim_surface_opacity : float
        Opacity for the simulation surface (default 0.95).
    shaded : bool
        If True, apply lighting-based shading to the simulation mesh. Default False (unlit colors).
    show, return_fig : bool
        Standard display controls.
    """
    # Background voxel cubes
    classes = kwargs.get('classes')
    voxel_color_map = kwargs.get('voxel_color_map', 'default')
    downsample = kwargs.get('downsample')
    title = kwargs.get('title', None)
    width = kwargs.get('width', 1000)
    height = kwargs.get('height', 800)
    cubes_opacity = kwargs.get('cubes_opacity', 0.9)
    show = kwargs.get('show', True)
    return_fig = kwargs.get('return_fig', False)

    if classes is None and voxel_array is not None:
        try:
            classes = np.unique(voxel_array[voxel_array != 0]).tolist()
        except Exception:
            classes = []

    if voxel_array is not None and len(classes) > 0:
        fig = visualize_voxcity_plotly(
            voxel_array,
            meshsize,
            classes=classes,
            voxel_color_map=voxel_color_map,
            opacity=cubes_opacity,
            downsample=downsample,
            title=title or "Ground Simulation (Plotly)",
            width=width,
            height=height,
            show=False,
            return_fig=True,
        )
    else:
        fig = go.Figure()
        fig.update_layout(
            title=title or "Ground Simulation (Plotly)",
            width=width,
            height=height,
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
                camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
            )
        )

    # Nothing to overlay
    if sim_grid is None or dem_grid is None:
        if show:
            fig.show()
        if return_fig:
            return fig
        return None

    # Determine color mapping range
    sim_vals = np.asarray(sim_grid, dtype=float)
    finite = np.isfinite(sim_vals)
    vmin = kwargs.get('vmin')
    vmax = kwargs.get('vmax')
    if vmin is None:
        vmin = float(np.nanmin(sim_vals[finite])) if np.any(finite) else 0.0
    if vmax is None:
        vmax = float(np.nanmax(sim_vals[finite])) if np.any(finite) else 1.0

    # Build elevated simulation surface (triangulated) using existing mesh utility
    # Elevation offset above ground; support both names for consistency
    z_offset = kwargs.get('z_offset', kwargs.get('view_point_height', 1.5))
    try:
        z_offset = float(z_offset)
    except Exception:
        z_offset = 1.5
    # Ensure offset is at least one meshsize and round up to a multiple of meshsize
    if meshsize is not None:
        try:
            ms = float(meshsize)
            if z_offset < ms:
                z_offset = ms
            # Round up to next multiple of meshsize
            z_offset = ms * math.ceil(z_offset / ms)
        except Exception:
            pass
    cmap_name = kwargs.get('colormap', 'viridis')
    sim_surface_opacity = kwargs.get('sim_surface_opacity', 0.95)
    shaded = kwargs.get('shaded', False)

    # Normalize DEM so its minimum becomes 0, matching voxel Z coordinates
    try:
        dem_norm = np.asarray(dem_grid, dtype=float)
        dem_norm = dem_norm - np.nanmin(dem_norm)
    except Exception:
        dem_norm = dem_grid

    sim_mesh = create_sim_surface_mesh(
        sim_grid, dem_norm,
        meshsize=meshsize,
        z_offset=z_offset,
        cmap_name=cmap_name,
        vmin=vmin,
        vmax=vmax,
    )

    # If mesh successfully created, overlay it as a Mesh3d trace with per-face colors
    if sim_mesh is not None and getattr(sim_mesh, 'vertices', None) is not None:
        V = np.asarray(sim_mesh.vertices)
        F = np.asarray(sim_mesh.faces)

        # Prefer face colors from the mesh if available
        facecolor = None
        try:
            colors_rgba = np.asarray(sim_mesh.visual.face_colors)
            if colors_rgba.ndim == 2 and colors_rgba.shape[0] == len(F):
                facecolor = [
                    f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors_rgba
                ]
        except Exception:
            facecolor = None

        # Lighting setup
        if shaded:
            lighting = dict(ambient=0.35, diffuse=1.0, specular=0.4, roughness=0.5, fresnel=0.1)
            flat = False
        else:
            lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0, fresnel=0.0)
            flat = False

        # Place a directional light near scene center
        cx = float((V[:,0].min() + V[:,0].max()) * 0.5)
        cy = float((V[:,1].min() + V[:,1].max()) * 0.5)
        cz = float((V[:,2].min() + V[:,2].max()) * 0.5)
        lx = cx + (V[:,0].max() - V[:,0].min() + meshsize) * 0.9
        ly = cy + (V[:,1].max() - V[:,1].min() + meshsize) * 0.6
        lz = cz + (V[:,2].max() - V[:,2].min() + meshsize) * 1.4

        fig.add_trace(
            go.Mesh3d(
                x=V[:,0], y=V[:,1], z=V[:,2],
                i=F[:,0], j=F[:,1], k=F[:,2],
                facecolor=facecolor,
                color=None if facecolor is not None else 'rgb(200,200,200)',
                opacity=float(sim_surface_opacity),
                flatshading=flat,
                lighting=lighting,
                lightposition=dict(x=lx, y=ly, z=lz),
                name='sim_surface'
            )
        )

    if show:
        fig.show()
    if return_fig:
        return fig
    return None

def visualize_voxcity_plotly(
    voxel_array,
    meshsize,
    classes=None,
    voxel_color_map='default',
    opacity=1.0,
    max_dimension=160,
    downsample=None,
    title=None,
    width=1000,
    height=800,
    show=True,
    return_fig=False,
):
    """
    Interactive 3D visualization where each occupied cell is rendered as a cube (six faces)
    using Plotly Mesh3d. One Mesh3d trace per class.

    Parameters are similar to visualize_voxcity_plotly, but rendering is via exact cubes.
    voxel_color_map may be either a scheme name (str) understood by get_voxel_color_map,
    or a dict mapping class_id -> [R, G, B] (0-255).
    """
    if voxel_array is None or getattr(voxel_array, 'ndim', 0) != 3:
        raise ValueError("voxel_array must be a 3D numpy array (nx, ny, nz)")

    vox = voxel_array

    # Downsample for performance if requested or auto-needed
    # Respect explicit downsample even when it is 1 (no auto-downsample)
    if downsample is not None:
        stride = max(1, int(downsample))
    else:
        stride = 1
        nx, ny, nz = vox.shape
        max_dim = max(nx, ny, nz)
        if max_dim > max_dimension:
            stride = int(np.ceil(max_dim / max_dimension))

    if stride > 1:
        vox = vox[::stride, ::stride, ::stride]

    nx, ny, nz = vox.shape

    # Coordinate of voxel centers in meters
    dx = meshsize * stride
    dy = meshsize * stride
    dz = meshsize * stride
    x = np.arange(nx, dtype=float) * dx
    y = np.arange(ny, dtype=float) * dy
    z = np.arange(nz, dtype=float) * dz

    # Choose classes
    if classes is None:
        classes = np.unique(vox[vox != 0]).tolist()
    if not classes:
        raise ValueError("No classes to visualize (voxel grid may be empty)")

    # Resolve color map: accept scheme name or explicit dict
    if isinstance(voxel_color_map, dict):
        vox_dict = voxel_color_map
    else:
        vox_dict = get_voxel_color_map(voxel_color_map)

    def exposed_face_masks(occ):
        # occ shape (nx, ny, nz)
        p = np.pad(occ, ((0,1),(0,0),(0,0)), constant_values=False)
        posx = occ & (~p[1:,:,:])
        p = np.pad(occ, ((1,0),(0,0),(0,0)), constant_values=False)
        negx = occ & (~p[:-1,:,:])
        p = np.pad(occ, ((0,0),(0,1),(0,0)), constant_values=False)
        posy = occ & (~p[:,1:,:])
        p = np.pad(occ, ((0,0),(1,0),(0,0)), constant_values=False)
        negy = occ & (~p[:,:-1,:])
        p = np.pad(occ, ((0,0),(0,0),(0,1)), constant_values=False)
        posz = occ & (~p[:,:,1:])
        p = np.pad(occ, ((0,0),(0,0),(1,0)), constant_values=False)
        negz = occ & (~p[:,:,:-1])
        return posx, negx, posy, negy, posz, negz

    def add_faces(fig, mask, plane, color_rgb):
        idx = np.argwhere(mask)
        if idx.size == 0:
            return
        xi, yi, zi = idx[:,0], idx[:,1], idx[:,2]
        xc = x[xi]; yc = y[yi]; zc = z[zi]
        x0, x1 = xc - dx/2.0, xc + dx/2.0
        y0, y1 = yc - dy/2.0, yc + dy/2.0
        z0, z1 = zc - dz/2.0, zc + dz/2.0

        if plane == '+x':
            verts = np.column_stack([
                np.repeat(x1, 4),
                np.stack([y0, y1, y1, y0], axis=1).reshape(-1)[0::4].repeat(4)  # placeholder, will rebuild below
            ])
        # Build vertices for all faces vectorized per plane
        if plane == '+x':
            vx = np.stack([x1, x1, x1, x1], axis=1)
            vy = np.stack([y0, y1, y1, y0], axis=1)
            vz = np.stack([z0, z0, z1, z1], axis=1)
        elif plane == '-x':
            vx = np.stack([x0, x0, x0, x0], axis=1)
            vy = np.stack([y0, y1, y1, y0], axis=1)
            vz = np.stack([z1, z1, z0, z0], axis=1)
        elif plane == '+y':
            vx = np.stack([x0, x1, x1, x0], axis=1)
            vy = np.stack([y1, y1, y1, y1], axis=1)
            vz = np.stack([z0, z0, z1, z1], axis=1)
        elif plane == '-y':
            vx = np.stack([x0, x1, x1, x0], axis=1)
            vy = np.stack([y0, y0, y0, y0], axis=1)
            vz = np.stack([z1, z1, z0, z0], axis=1)
        elif plane == '+z':
            vx = np.stack([x0, x1, x1, x0], axis=1)
            vy = np.stack([y0, y0, y1, y1], axis=1)
            vz = np.stack([z1, z1, z1, z1], axis=1)
        elif plane == '-z':
            vx = np.stack([x0, x1, x1, x0], axis=1)
            vy = np.stack([y1, y1, y0, y0], axis=1)
            vz = np.stack([z0, z0, z0, z0], axis=1)
        else:
            return

        V = np.column_stack([vx.reshape(-1), vy.reshape(-1), vz.reshape(-1)])
        # Two triangles per quad
        n = idx.shape[0]
        starts = np.arange(0, 4*n, 4, dtype=np.int32)
        tris = np.vstack([
            np.stack([starts, starts+1, starts+2], axis=1),
            np.stack([starts, starts+2, starts+3], axis=1)
        ])

        # Lighting for shading contrast (higher ambient + diffuse + specular)
        lighting = dict(
            ambient=0.35,
            diffuse=1.0,
            specular=0.4,
            roughness=0.5,
            fresnel=0.1,
        )
        # Directional light position near the scene center
        cx = (x.min() + x.max()) * 0.5 if len(x) > 0 else 0.0
        cy = (y.min() + y.max()) * 0.5 if len(y) > 0 else 0.0
        cz = (z.min() + z.max()) * 0.5 if len(z) > 0 else 0.0
        lx = cx + (x.max() - x.min() + dx) * 0.9
        ly = cy + (y.max() - y.min() + dy) * 0.6
        lz = cz + (z.max() - z.min() + dz) * 1.4

        fig.add_trace(
            go.Mesh3d(
                x=V[:,0], y=V[:,1], z=V[:,2],
                i=tris[:,0], j=tris[:,1], k=tris[:,2],
                color=_rgb_tuple_to_plotly_color(color_rgb),
                opacity=float(opacity),
                flatshading=False,
                lighting=lighting,
                lightposition=dict(x=lx, y=ly, z=lz),
                name=f"{plane}"
            )
        )

    fig = go.Figure()

    for cls in classes:
        if not np.any(vox == cls):
            continue
        occ = (vox == cls)
        posx, negx, posy, negy, posz, negz = exposed_face_masks(occ)
        color_rgb = vox_dict.get(int(cls), [128,128,128])
        add_faces(fig, posx, '+x', color_rgb)
        add_faces(fig, negx, '-x', color_rgb)
        add_faces(fig, posy, '+y', color_rgb)
        add_faces(fig, negy, '-y', color_rgb)
        add_faces(fig, posz, '+z', color_rgb)
        add_faces(fig, negz, '-z', color_rgb)

    fig.update_layout(
        title=title or "VoxCity 3D (Voxel Cubes)",
        width=width,
        height=height,
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
        )
    )

    if show:
        fig.show()
    if return_fig:
        return fig
    return None