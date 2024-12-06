import numpy as np
from shapely.geometry import Polygon
from rtree import index
from collections import Counter

def rgb_distance(color1, color2):
    return np.sqrt(np.sum((np.array(color1) - np.array(color2))**2))  


# land_cover_classes = {
#     (128, 0, 0): 'Bareland',              0         
#     (0, 255, 36): 'Rangeland',            1
#     (97, 140, 86): 'Shrub',               2
#     (75, 181, 73): 'Agriculture land',    3
#     (34, 97, 38): 'Tree',                 4
#     (77, 118, 99): 'Wet land',            5
#     (22, 61, 51): 'Mangrove',             6
#     (0, 69, 255): 'Water',                7
#     (205, 215, 224): 'Snow and ice',      8
#     (148, 148, 148): 'Developed space',   9
#     (255, 255, 255): 'Road',              10
#     (222, 31, 7): 'Building',             11
#     (128, 0, 0): 'No Data',               12
# }

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
    elif (source == "OpenEarthMapJapan"):
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
    elif (source == 'Standard') or (source == "OpenStreetMap"):
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

# land_cover_classes = {
#     (128, 0, 0): 'Bareland',              0         
#     (0, 255, 36): 'Rangeland',            1
#     (97, 140, 86): 'Shrub',               2
#     (75, 181, 73): 'Agriculture land',    3
#     (34, 97, 38): 'Tree',                 4
#     (34, 97, 38): 'Moss and lichen',      5
#     (77, 118, 99): 'Wet land',            6
#     (22, 61, 51): 'Mangrove',             7
#     (0, 69, 255): 'Water',                8
#     (205, 215, 224): 'Snow and ice',      9
#     (148, 148, 148): 'Developed space',   10
#     (255, 255, 255): 'Road',              11
#     (222, 31, 7): 'Building',             12
#     (128, 0, 0): 'No Data',               13
# }



def convert_land_cover(input_array, land_cover_source='Urbanwatch'):   

    if land_cover_source == 'Urbanwatch':
        # Define the mapping from Urbanwatch to new standardized classes
        convert_dict = {
            0: 12,  # Building
            1: 11,  # Road
            2: 10,  # Parking Lot -> Developed space
            3: 4,   # Tree Canopy -> Tree
            4: 1,   # Grass/Shrub -> Rangeland
            5: 3,   # Agriculture -> Agriculture land
            6: 8,   # Water
            7: 0,   # Barren -> Bareland
            8: 13,  # Unknown -> No Data
            9: 8    # Sea -> Water
        }
    elif land_cover_source == 'ESA WorldCover':
        convert_dict = {
            0: 4,   # Trees -> Tree
            1: 2,   # Shrubland -> Shrub
            2: 1,   # Grassland -> Rangeland
            3: 3,   # Cropland -> Agriculture land
            4: 10,  # Built-up -> Developed space
            5: 0,   # Barren / sparse vegetation -> Bareland
            6: 9,   # Snow and ice
            7: 8,   # Open water -> Water
            8: 6,   # Herbaceous wetland -> Wet land
            9: 7,   # Mangroves
            10: 5   # Moss and lichen
        }
    elif land_cover_source == "ESRI 10m Annual Land Cover":
        convert_dict = {
            0: 13,  # No Data
            1: 8,   # Water
            2: 4,   # Trees -> Tree
            3: 1,   # Grass -> Rangeland
            4: 6,   # Flooded Vegetation -> Wet land
            5: 3,   # Crops -> Agriculture land
            6: 2,   # Scrub/Shrub -> Shrub
            7: 10,  # Built Area -> Developed space
            8: 0,   # Bare Ground -> Bareland
            9: 9,   # Snow/Ice
            10: 13  # Clouds -> No Data
        }
    elif land_cover_source == "Dynamic World V1":
        convert_dict = {
            0: 8,   # Water
            1: 4,   # Trees -> Tree
            2: 1,   # Grass -> Rangeland
            3: 6,   # Flooded Vegetation -> Wet land
            4: 3,   # Crops -> Agriculture land
            5: 2,   # Shrub and Scrub -> Shrub
            6: 10,  # Built -> Developed space
            7: 0,   # Bare -> Bareland
            8: 9    # Snow and Ice
        }    
    elif land_cover_source == "OpenEarthMapJapan":
        convert_dict = {
            0: 0,   # Bareland
            1: 1,   # Rangeland
            2: 10,  # Developed space
            3: 11,  # Road
            4: 4,   # Tree
            5: 8,   # Water
            6: 3,   # Agriculture land
            7: 12,  # Building
        }

    # Create a vectorized function for the conversion
    vectorized_convert = np.vectorize(lambda x: convert_dict.get(x, x))
    
    # Apply the conversion to the input array
    converted_array = vectorized_convert(input_array)
    
    return converted_array

def get_class_priority(source):
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