"""
Land Cover Classification Utilities for VoxelCity

This module provides utilities for handling land cover data from various sources,
including color-based classification, data conversion between different land cover
classification systems, and spatial analysis of land cover polygons.

Supported land cover data sources:
- Urbanwatch
- OpenEarthMapJapan
- ESRI 10m Annual Land Cover
- ESA WorldCover
- Dynamic World V1
- OpenStreetMap
- Standard classification
"""

import numpy as np
from shapely.geometry import Polygon
from rtree import index
from collections import Counter

def rgb_distance(color1, color2):
    """
    Calculate the Euclidean distance between two RGB colors.
    
    Args:
        color1 (tuple): RGB values as (R, G, B) tuple
        color2 (tuple): RGB values as (R, G, B) tuple
        
    Returns:
        float: Euclidean distance between the two colors
    """
    return np.sqrt(np.sum((np.array(color1) - np.array(color2))**2))  


# Standard land cover classes mapping (1-based indices for voxel representation)
# land_cover_classes = {
#     (128, 0, 0): 'Bareland',              1         
#     (0, 255, 36): 'Rangeland',            2
#     (97, 140, 86): 'Shrub',               3
#     (75, 181, 73): 'Agriculture land',    4
#     (34, 97, 38): 'Tree',                 5
#     (255, 255, 0): 'Moss and lichen',     6
#     (77, 118, 99): 'Wet land',            7
#     (22, 61, 51): 'Mangrove',             8
#     (0, 69, 255): 'Water',                9
#     (205, 215, 224): 'Snow and ice',      10
#     (148, 148, 148): 'Developed space',   11
#     (255, 255, 255): 'Road',              12
#     (222, 31, 7): 'Building',             13
#     (128, 0, 0): 'No Data',               14
# }

def get_land_cover_classes(source):
    """
    Get land cover classification mapping for a specific data source.
    
    Each data source has its own color-to-class mapping system. This function
    returns the appropriate RGB color to land cover class dictionary based on
    the specified source.
    
    Args:
        source (str): Name of the land cover data source. Supported sources:
                     "Urbanwatch", "OpenEarthMapJapan", "ESRI 10m Annual Land Cover",
                     "ESA WorldCover", "Dynamic World V1", "Standard", "OpenStreetMap"
                     
    Returns:
        dict: Dictionary mapping RGB tuples to land cover class names
        
    Example:
        >>> classes = get_land_cover_classes("Urbanwatch")
        >>> print(classes[(255, 0, 0)])  # Returns 'Building'
    """
    if source == "Urbanwatch":
        # Urbanwatch color scheme - focused on urban features
        land_cover_classes = {
            (255, 0, 0): 'Building',
            (133, 133, 133): 'Road',
            (255, 0, 192): 'Parking Lot',
            (34, 139, 34): 'Tree Canopy',
            (128, 236, 104): 'Grass/Shrub',
            (255, 193, 37): 'Agriculture',
            (0, 0, 255): 'Water',
            (234, 234, 234): 'Barren',
            (255, 255, 255): 'Unknown',
            (0, 0, 0): 'Sea'
        }    
    elif (source == "OpenEarthMapJapan"):
        # OpenEarthMap Japan specific classification
        land_cover_classes = {
            (128, 0, 0): 'Bareland',
            (0, 255, 36): 'Rangeland',
            (148, 148, 148): 'Developed space',
            (255, 255, 255): 'Road',
            (34, 97, 38): 'Tree',
            (0, 69, 255): 'Water',
            (75, 181, 73): 'Agriculture land',
            (222, 31, 7): 'Building'
        }
    elif source == "ESRI 10m Annual Land Cover":
        # ESRI's global 10-meter resolution land cover classification
        land_cover_classes = {
            (255, 255, 255): 'No Data',
            (26, 91, 171): 'Water',
            (53, 130, 33): 'Trees',
            (167, 210, 130): 'Grass',
            (135, 209, 158): 'Flooded Vegetation',
            (255, 219, 92): 'Crops',
            (238, 207, 168): 'Scrub/Shrub',
            (237, 2, 42): 'Built Area',
            (237, 233, 228): 'Bare Ground',
            (242, 250, 255): 'Snow/Ice',
            (200, 200, 200): 'Clouds'
        }
    elif source == "ESA WorldCover":
        # European Space Agency WorldCover 10m classification
        land_cover_classes = {
            (0, 112, 0): 'Trees',
            (255, 224, 80): 'Shrubland',
            (255, 255, 170): 'Grassland',
            (255, 176, 176): 'Cropland',
            (230, 0, 0): 'Built-up',
            (191, 191, 191): 'Barren / sparse vegetation',
            (192, 192, 255): 'Snow and ice',
            (0, 60, 255): 'Open water',
            (0, 236, 230): 'Herbaceous wetland',
            (0, 255, 0): 'Mangroves',
            (255, 255, 0): 'Moss and lichen'
        }
    elif source == "Dynamic World V1":
        # Google's Dynamic World near real-time land cover
        # Convert hex colors to RGB tuples
        land_cover_classes = {
            (65, 155, 223): 'Water',            # #419bdf
            (57, 125, 73): 'Trees',             # #397d49
            (136, 176, 83): 'Grass',            # #88b053
            (122, 135, 198): 'Flooded Vegetation', # #7a87c6
            (228, 150, 53): 'Crops',            # #e49635
            (223, 195, 90): 'Shrub and Scrub',  # #dfc35a
            (196, 40, 27): 'Built',             # #c4281b
            (165, 155, 143): 'Bare',            # #a59b8f
            (179, 159, 225): 'Snow and Ice'     # #b39fe1
        }
    elif (source == 'Standard') or (source == "OpenStreetMap"):
        # Standard/OpenStreetMap classification - comprehensive land cover types
        land_cover_classes = {
            (128, 0, 0): 'Bareland',
            (0, 255, 36): 'Rangeland',
            (255, 224, 80): 'Shrub',
            (255, 255, 0): 'Moss and lichen',
            (75, 181, 73): 'Agriculture land',
            (34, 97, 38): 'Tree',
            (0, 236, 230): 'Wet land',
            (22, 61, 51): 'Mangroves',
            (0, 69, 255): 'Water',
            (192, 192, 255): 'Snow and ice',
            (148, 148, 148): 'Developed space',
            (255, 255, 255): 'Road',
            (222, 31, 7): 'Building',
            (0, 0, 0): 'No Data'
        }
    return land_cover_classes


def get_source_class_descriptions(source):
    """
    Get a formatted string describing land cover classes for a specific source.
    
    Args:
        source (str): Name of the land cover data source.
        
    Returns:
        str: Formatted string describing the source's land cover classes.
    """
    land_cover_classes = get_land_cover_classes(source)
    # Get unique class names (values from the dict)
    class_names = list(dict.fromkeys(land_cover_classes.values()))
    
    lines = [f"\n{source} Land Cover Classes (source-specific, 0-based indices):"]
    lines.append("-" * 55)
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx:2d}: {name}")
    lines.append("-" * 55)
    
    # Special note for OpenStreetMap/Standard which uses same class names
    if source in ("OpenStreetMap", "Standard"):
        lines.append("Note: OpenStreetMap uses VoxCity Standard class names.")
        lines.append("      Indices shift from 0-based to 1-based during voxelization.")
    else:
        lines.append("Note: These source-specific classes will be converted to")
        lines.append("      VoxCity Standard Classes (1-14) during voxelization.")
    
    lines.append("")
    lines.append("Access 2D land cover grid from VoxCity object:")
    lines.append("  land_cover_grid = voxcity.land_cover.classes")
    lines.append("")
    return "\n".join(lines)


# Standard land cover classes with numeric indices (1-based for voxel representation)
# land_cover_classes = {
#     (128, 0, 0): 'Bareland',              1         
#     (0, 255, 36): 'Rangeland',            2
#     (97, 140, 86): 'Shrub',               3
#     (75, 181, 73): 'Agriculture land',    4
#     (34, 97, 38): 'Tree',                 5
#     (255, 255, 0): 'Moss and lichen',     6
#     (77, 118, 99): 'Wet land',            7
#     (22, 61, 51): 'Mangrove',             8
#     (0, 69, 255): 'Water',                9
#     (205, 215, 224): 'Snow and ice',      10
#     (148, 148, 148): 'Developed space',   11
#     (255, 255, 255): 'Road',              12
#     (222, 31, 7): 'Building',             13
#     (128, 0, 0): 'No Data',               14
# }



def convert_land_cover(input_array, land_cover_source='Urbanwatch'):   
    """
    Optimized version using direct numpy array indexing instead of np.vectorize.
    This is 10-100x faster than the original.
    
    Returns 1-based class indices (1-14) for consistency with voxel representations.
    """
    # Define mappings (1-based indices: 1=Bareland, 2=Rangeland, ..., 14=No Data)
    if land_cover_source == 'Urbanwatch':
        mapping = {0: 13, 1: 12, 2: 11, 3: 5, 4: 2, 5: 4, 6: 9, 7: 1, 8: 14, 9: 9}
    elif land_cover_source == 'ESA WorldCover':
        mapping = {0: 5, 1: 3, 2: 2, 3: 4, 4: 11, 5: 1, 6: 10, 7: 9, 8: 7, 9: 8, 10: 6}
    elif land_cover_source == "ESRI 10m Annual Land Cover":
        mapping = {0: 14, 1: 9, 2: 5, 3: 2, 4: 7, 5: 4, 6: 3, 7: 11, 8: 1, 9: 10, 10: 14}
    elif land_cover_source == "Dynamic World V1":
        mapping = {0: 9, 1: 5, 2: 2, 3: 7, 4: 4, 5: 3, 6: 11, 7: 1, 8: 10}    
    elif land_cover_source == "OpenEarthMapJapan":
        mapping = {0: 1, 1: 2, 2: 11, 3: 12, 4: 5, 5: 9, 6: 4, 7: 13}
    else:
        # If unknown source, return as-is with +1 offset for consistency
        return input_array.copy() + 1
    
    # Create a full mapping array for all possible values (0-255 for uint8)
    max_val = max(max(mapping.keys()), input_array.max()) + 1
    lookup = np.arange(max_val, dtype=input_array.dtype)
    
    # Apply the mapping
    for old_val, new_val in mapping.items():
        if old_val < max_val:
            lookup[old_val] = new_val
    
    # Use fancy indexing for fast conversion
    return lookup[input_array]

def get_class_priority(source):
    """
    Get priority rankings for land cover classes to resolve conflicts during classification.
    
    When multiple land cover classes are present in the same area, this priority system
    determines which class should take precedence. Higher priority values indicate
    classes that should override lower priority classes.
    
    Args:
        source (str): Name of the land cover data source
        
    Returns:
        dict: Dictionary mapping class names to priority values (higher = more priority)
        
    Priority Logic for OpenStreetMap:
        - Built Environment: Highest priority (most definitive structures)
        - Water Bodies: High priority (clearly defined features)  
        - Vegetation: Medium priority (managed vs natural)
        - Natural Non-Vegetation: Lower priority (often default classifications)
        - Uncertain/No Data: Lowest priority
    """
    if source == "OpenStreetMap":
        return {
            # Built Environment (highest priority as they're most definitively mapped)
            'Building': 2,          # Most definitive built structure
            'Road': 1,             # Critical infrastructure
            'Developed space': 13,   # Other developed areas
            
            # Water Bodies (next priority as they're clearly defined)
            'Water': 3,            # Open water
            'Wet land': 4,          # Semi-aquatic areas
            'Moss and lichen': 5,          # Semi-aquatic areas
            'Mangrove': 6,          # Special water-associated vegetation
            
            # Vegetation (medium priority)
            'Tree': 12,              # Distinct tree cover
            'Agriculture land': 11,   # Managed vegetation
            'Shrub': 10,             # Medium height vegetation
            'Rangeland': 9,         # Low vegetation
            
            # Natural Non-Vegetation (lower priority as they're often default classifications)
            'Snow and ice': 8,      # Distinct natural cover
            'Bareland': 7,          # Exposed ground
            
            # Uncertain
            'No Data': 14            # Lowest priority as it represents uncertainty
        }
        # Legacy priority system - kept for reference
        # return { 
        #     'Bareland': 4, 
        #     'Rangeland': 6, 
        #     'Developed space': 8, 
        #     'Road': 1, 
        #     'Tree': 7, 
        #     'Water': 3, 
        #     'Agriculture land': 5, 
        #     'Building': 2 
        # }

def create_land_cover_polygons(land_cover_geojson):
    """
    Create polygon geometries and spatial index from land cover GeoJSON data.
    
    This function processes GeoJSON land cover data to create Shapely polygon
    geometries and builds an R-tree spatial index for efficient spatial queries.
    
    Args:
        land_cover_geojson (list): List of GeoJSON feature dictionaries containing
                                  land cover polygons with geometry and properties
                                  
    Returns:
        tuple: A tuple containing:
            - land_cover_polygons (list): List of tuples (polygon, class_name)
            - idx (rtree.index.Index): Spatial index for efficient polygon lookup
            
    Note:
        Each GeoJSON feature should have:
        - geometry.coordinates[0]: List of coordinate pairs defining the polygon
        - properties.class: String indicating the land cover class
    """
    land_cover_polygons = []
    idx = index.Index()
    count = 0
    for i, land_cover in enumerate(land_cover_geojson):
        # print(land_cover['geometry']['coordinates'][0])
        polygon = Polygon(land_cover['geometry']['coordinates'][0])
        # land_cover_index = class_mapping[land_cover['properties']['class']]
        land_cover_class = land_cover['properties']['class']
        # if (height <= 0) or (height == None):
        #     # print("A building with a height of 0 meters was found. A height of 10 meters was set instead.")
        #     count += 1
        #     height = 10
        # land_cover_polygons.append((polygon, land_cover_index))
        land_cover_polygons.append((polygon, land_cover_class))
        idx.insert(i, polygon.bounds)
    
    # print(f"{count} of the total {len(filtered_buildings)} buildings did not have height data. A height of 10 meters was set instead.")
    return land_cover_polygons, idx

def get_nearest_class(pixel, land_cover_classes):
    """
    Find the nearest land cover class for a given pixel color using RGB distance.
    
    This function determines the most appropriate land cover class for a pixel
    by finding the class with the minimum RGB color distance to the pixel's color.
    
    Args:
        pixel (tuple): RGB color values as (R, G, B) tuple
        land_cover_classes (dict): Dictionary mapping RGB tuples to class names
        
    Returns:
        str: Name of the nearest land cover class
        
    Example:
        >>> classes = {(255, 0, 0): 'Building', (0, 255, 0): 'Tree'}
        >>> nearest = get_nearest_class((250, 5, 5), classes)
        >>> print(nearest)  # Returns 'Building'
    """
    distances = {class_name: rgb_distance(pixel, color) 
                 for color, class_name in land_cover_classes.items()}
    return min(distances, key=distances.get)

def get_dominant_class(cell_data, land_cover_classes):
    """
    Determine the dominant land cover class in a cell based on pixel majority.
    
    This function analyzes all pixels within a cell, classifies each pixel to its
    nearest land cover class, and returns the most frequently occurring class.
    
    Args:
        cell_data (numpy.ndarray): 3D array of RGB pixel data for the cell
        land_cover_classes (dict): Dictionary mapping RGB tuples to class names
        
    Returns:
        str: Name of the dominant land cover class in the cell
        
    Note:
        If the cell contains no data, returns 'No Data'
    """
    if cell_data.size == 0:
        return 'No Data'
    # Classify each pixel in the cell to its nearest land cover class
    pixel_classes = [get_nearest_class(tuple(pixel), land_cover_classes) 
                     for pixel in cell_data.reshape(-1, 3)]
    # Count occurrences of each class
    class_counts = Counter(pixel_classes)
    # Return the most common class
    return class_counts.most_common(1)[0][0]

def convert_land_cover_array(input_array, land_cover_classes):
    """
    Convert an array of land cover class names to integer indices.
    
    This function maps string-based land cover class names to integer indices
    for numerical processing and storage efficiency.
    
    Args:
        input_array (numpy.ndarray): Array containing land cover class names as strings
        land_cover_classes (dict): Dictionary mapping RGB tuples to class names
        
    Returns:
        numpy.ndarray: Array with 0-based integer indices corresponding to land cover classes
        
    Note:
        Classes not found in the mapping are assigned index -1
        Indices are 0-based as source-specific indices. Use convert_land_cover() to 
        remap to standard 1-based indices for voxel representation.
    """
    # Create a mapping of class names to integers (0-based, source-specific order)
    class_to_int = {name: i for i, name in enumerate(land_cover_classes.values())}

    # Create a vectorized function to map string values to integers
    vectorized_map = np.vectorize(lambda x: class_to_int.get(x, -1))

    # Apply the mapping to the input array
    output_array = vectorized_map(input_array)

    return output_array