import numpy as np
from shapely.geometry import Polygon
from rtree import index
from collections import Counter

def rgb_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2))**2))  

def get_land_cover_classes(source):
    if source == "Urbanwatch":
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
    elif (source == "OpenEarthMapJapan") or (source == "OpenStreetMap"):
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
    return land_cover_classes

def convert_land_cover(input_array, land_cover_source='Urbanwatch'):  

    if land_cover_source == 'Urbanwatch':
        # Define the mapping from #urbanwatch to #general(integration)
        convert_dict = {
            0: 7,  # Building
            1: 3,  # Road
            2: 2,  # Parking Lot
            3: 4,  # Tree Canopy
            4: 1,  # Grass/Shrub
            5: 6,  # Agriculture
            6: 5,  # Water
            7: 0,  # Barren
            8: 0,  # Unknown
            9: 5   # Sea
        }
    elif land_cover_source == 'ESA WorldCover':
        convert_dict = {
            0: 4,  # Trees
            1: 1,  # Shrubland
            2: 1,  # Grassland
            3: 6,  # Cropland
            4: 2,  # Built-up
            5: 0,  # Barren / sparse vegetation
            6: 0,  # Snow and ice
            7: 5,  # Open water
            8: 5,  # Herbaceous wetland
            9: 5,  # Mangroves
            10: 1  # Moss and lichen
        }
    elif land_cover_source == "ESRI 10m Annual Land Cover":
        convert_dict = {
            0: 0,  # (255, 255, 255): 'No Data',
            1: 5,  # (26, 91, 171): 'Water',
            2: 4,  # (53, 130, 33): 'Trees',
            3: 1,  # (167, 210, 130): 'Grass',
            4: 5,  # (135, 209, 158): 'Flooded Vegetation',
            5: 6,  # (255, 219, 92): 'Crops',
            6: 1,  # (238, 207, 168): 'Scrub/Shrub',
            7: 2,  # (237, 2, 42): 'Built Area',
            8: 0,  # (237, 233, 228): 'Bare Ground',
            9: 0,  # (242, 250, 255): 'Snow/Ice',
            10: 0  # (200, 200, 200): 'Clouds'
        }
    elif land_cover_source == "Dynamic World V1":
        # Convert hex colors to RGB tuples
        convert_dict = {
            0: 5,# 'Water',            
            1: 4,# 'Trees',             
            2: 1,# 'Grass',            
            3: 5,# 'Flooded Vegetation', 
            4: 6,# 'Crops',            
            5: 1,# 'Shrub and Scrub',  
            6: 2,# 'Built',             
            7: 0,# 'Bare',            
            8: 0,# 'Snow and Ice'     
        }
        
    # Create a vectorized function for the conversion
    vectorized_convert = np.vectorize(lambda x: convert_dict.get(x, x))
    
    # Apply the conversion to the input array
    converted_array = vectorized_convert(input_array)
    
    return converted_array

def get_class_priority(source):
    if source == "OpenStreetMap":
        return { 
            'Bareland': 4, 
            'Rangeland': 6, 
            'Developed space': 8, 
            'Road': 1, 
            'Tree': 7, 
            'Water': 3, 
            'Agriculture land': 5, 
            'Building': 2 
        }

def create_land_cover_polygons(land_cover_geojson):
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
    distances = {class_name: rgb_distance(pixel, color) 
                 for color, class_name in land_cover_classes.items()}
    return min(distances, key=distances.get)

def get_dominant_class(cell_data, land_cover_classes):
    if cell_data.size == 0:
        return 'No Data'
    pixel_classes = [get_nearest_class(tuple(pixel), land_cover_classes) 
                     for pixel in cell_data.reshape(-1, 3)]
    class_counts = Counter(pixel_classes)
    return class_counts.most_common(1)[0][0]

def convert_land_cover_array(input_array, land_cover_classes):
    # Create a mapping of class names to integers
    class_to_int = {name: i for i, name in enumerate(land_cover_classes.values())}

    # Create a vectorized function to map string values to integers
    vectorized_map = np.vectorize(lambda x: class_to_int.get(x, -1))

    # Apply the mapping to the input array
    output_array = vectorized_map(input_array)

    return output_array