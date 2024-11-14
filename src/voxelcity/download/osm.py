import requests
from shapely.geometry import Polygon
# Import libraries
import requests
from osm2geojson import json2geojson
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import transform
import pyproj

def load_geojsons_from_openstreetmap(rectangle_vertices):
    # Create a bounding box from the rectangle vertices
    min_lat = min(v[0] for v in rectangle_vertices)
    max_lat = max(v[0] for v in rectangle_vertices)
    min_lon = min(v[1] for v in rectangle_vertices)
    max_lon = max(v[1] for v in rectangle_vertices)
    
    # Enhanced Overpass API query with recursive member extraction
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["building:part"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
      way["tourism"="artwork"]["area"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["tourism"="artwork"]["area"="yes"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    (._; >;);  // Recursively get all nodes, ways, and relations within relations
    out geom;
    """
    
    # Send the request to the Overpass API
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    # Process the response and create GeoJSON features
    features = []
    
    def process_coordinates(geometry):
        """Helper function to process and reverse coordinate pairs"""
        return [coord[::-1] for coord in geometry]
    
    def get_height_from_properties(properties):
        """Helper function to extract height from properties, using levels if height is not available"""
        height = properties.get('height', properties.get('building:height', None))
        if height is not None:
            try:
                return float(height)
            except ValueError:
                pass
        
        levels = properties.get('building:levels', properties.get('levels', None))
        if levels is not None:
            try:
                return float(levels) * 5.0  # Assume 5 meters per level
            except ValueError:
                pass
        
        return 0  # Default height if no valid height or levels found
    
    def extract_properties(element):
        """Helper function to extract and process properties"""
        properties = element.get('tags', {})
        
        # Get height (now using the helper function)
        height = get_height_from_properties(properties)
            
        # Get min_height and min_level
        min_height = properties.get('min_height', '0')
        min_level = properties.get('building:min_level', properties.get('min_level', '0'))
        try:
            min_height = float(min_height)
        except ValueError:
            try:
                min_height = float(min_level) * 5.0
            except ValueError:
                min_height = 0
        
        levels = properties.get('building:levels', properties.get('levels', None))
        try:
            levels = float(levels) if levels is not None else None
        except ValueError:
            levels = None
            
        # Extract additional properties, including those relevant to artworks
        extracted_props = {
            "height": height,
            "min_height": min_height,
            "confidence": -1.0,
            "is_inner": False,
            "levels": levels,
            "height_source": "explicit" if properties.get('height') or properties.get('building:height') 
                               else "levels" if levels is not None 
                               else "default",
            "min_level": min_level if min_level != '0' else None,
            "building": properties.get('building', 'no'),
            "building_part": properties.get('building:part', 'no'),
            "building_material": properties.get('building:material'),
            "building_colour": properties.get('building:colour'),
            "roof_shape": properties.get('roof:shape'),
            "roof_material": properties.get('roof:material'),
            "roof_angle": properties.get('roof:angle'),
            "roof_colour": properties.get('roof:colour'),
            "roof_direction": properties.get('roof:direction'),
            "architect": properties.get('architect'),
            "start_date": properties.get('start_date'),
            "name": properties.get('name'),
            "name:en": properties.get('name:en'),
            "name:es": properties.get('name:es'),
            "email": properties.get('email'),
            "phone": properties.get('phone'),
            "wheelchair": properties.get('wheelchair'),
            "tourism": properties.get('tourism'),
            "artwork_type": properties.get('artwork_type'),
            "area": properties.get('area'),
            "layer": properties.get('layer')
        }
        
        # Remove None values to keep the properties clean
        return {k: v for k, v in extracted_props.items() if v is not None}
    
    def create_polygon_feature(coords, properties, is_inner=False):
        """Helper function to create a polygon feature"""
        if len(coords) >= 4:
            properties = properties.copy()
            properties["is_inner"] = is_inner
            return {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [process_coordinates(coords)]
                }
            }
        return None
    
    # Process each element, handling relations and their way members
    for element in data['elements']:
        if element['type'] == 'way':
            coords = [(node['lon'], node['lat']) for node in element['geometry']]
            feature = create_polygon_feature(coords, extract_properties(element))
            if feature:
                features.append(feature)
                
        elif element['type'] == 'relation':
            properties = extract_properties(element)
            
            # Process each member of the relation
            for member in element['members']:
                if member['type'] == 'way' and 'geometry' in member:
                    coords = [(node['lon'], node['lat']) for node in member['geometry']]
                    is_inner = member['role'] == 'inner'
                    
                    feature = create_polygon_feature(coords, properties, is_inner)
                    if feature:
                        feature['properties']['role'] = member['role']
                        features.append(feature)
    
    return features














# def load_geojsons_from_openstreetmap(rectangle_vertices):
#     # Create a bounding box from the rectangle vertices
#     min_lat = min(v[0] for v in rectangle_vertices)
#     max_lat = max(v[0] for v in rectangle_vertices)
#     min_lon = min(v[1] for v in rectangle_vertices)
#     max_lon = max(v[1] for v in rectangle_vertices)
    
#     # Construct the Overpass API query to include building parts
#     overpass_url = "http://overpass-api.de/api/interpreter"
#     overpass_query = f"""
#     [out:json];
#     (
#       way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
#       way["building:part"]({min_lat},{min_lon},{max_lat},{max_lon});
#       relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
#     );
#     out geom;
#     """
    
#     # Send the request to the Overpass API
#     response = requests.get(overpass_url, params={'data': overpass_query})
#     data = response.json()
    
#     # Process the response and create GeoJSON features
#     features = []
    
#     def process_coordinates(geometry):
#         """Helper function to process and reverse coordinate pairs"""
#         return [coord[::-1] for coord in geometry]
    
#     def get_height_from_properties(properties):
#         """Helper function to extract height from properties, using levels if height is not available"""
#         # Try to get explicit height first
#         height = properties.get('height', properties.get('building:height', None))
#         if height is not None:
#             try:
#                 return float(height)
#             except ValueError:
#                 pass
        
#         # If no height, try to get levels
#         levels = properties.get('building:levels', properties.get('levels', None))
#         if levels is not None:
#             try:
#                 return float(levels) * 5.0  # Assume 5 meters per level
#             except ValueError:
#                 pass
        
#         return 0  # Default height if no valid height or levels found
    
#     def extract_properties(element):
#         """Helper function to extract and process properties"""
#         properties = element.get('tags', {})
        
#         # Get height (now using the helper function)
#         height = get_height_from_properties(properties)
            
#         # Get min_height and min_level
#         min_height = properties.get('min_height', '0')
#         min_level = properties.get('building:min_level', properties.get('min_level', '0'))
#         try:
#             min_height = float(min_height)
#         except ValueError:
#             try:
#                 # Try to calculate min_height from min_level if available
#                 min_height = float(min_level) * 5.0
#             except ValueError:
#                 min_height = 0
        
#         # Get levels information
#         levels = properties.get('building:levels', properties.get('levels', None))
#         try:
#             levels = float(levels) if levels is not None else None
#         except ValueError:
#             levels = None
            
#         # Extract additional building part properties
#         extracted_props = {
#             "height": height,
#             "min_height": min_height,
#             "confidence": -1.0,
#             "is_inner": False,  # Changed from 'inner' to 'is_inner'
#             "levels": levels,
#             "height_source": "explicit" if properties.get('height') or properties.get('building:height') 
#                            else "levels" if levels is not None 
#                            else "default",
#             "min_level": min_level if min_level != '0' else None,
#             "building_part": properties.get('building:part', 'no'),
#             "building_material": properties.get('building:material'),
#             "building_colour": properties.get('building:colour'),
#             "roof_shape": properties.get('roof:shape'),
#             "roof_material": properties.get('roof:material'),
#             "roof_angle": properties.get('roof:angle'),
#             "roof_colour": properties.get('roof:colour'),
#             "roof_direction": properties.get('roof:direction'),
#             "architect": properties.get('architect'),
#             "start_date": properties.get('start_date'),
#             "name": properties.get('name'),
#             "email": properties.get('email'),
#             "phone": properties.get('phone'),
#             "wheelchair": properties.get('wheelchair')
#         }
        
#         # Remove None values to keep the properties clean
#         return {k: v for k, v in extracted_props.items() if v is not None}
    
#     def create_polygon_feature(coords, properties, is_inner=False):
#         """Helper function to create a polygon feature"""
#         if len(coords) >= 4:
#             properties = properties.copy()  # Create a copy to avoid modifying the original
#             properties["is_inner"] = is_inner  # Changed from 'inner' to 'is_inner'
#             return {
#                 "type": "Feature",
#                 "properties": properties,
#                 "geometry": {
#                     "type": "Polygon",
#                     "coordinates": [process_coordinates(coords)]
#                 }
#             }
#         return None
    
#     for element in data['elements']:
#         if element['type'] == 'way':
#             # Process simple polygons from ways
#             coords = [(node['lon'], node['lat']) for node in element['geometry']]
#             feature = create_polygon_feature(coords, extract_properties(element))
#             if feature:
#                 features.append(feature)
                
#         elif element['type'] == 'relation':
#             properties = extract_properties(element)
            
#             # Process each member separately
#             for member in element['members']:
#                 if 'geometry' in member:
#                     coords = [(node['lon'], node['lat']) for node in member['geometry']]
#                     is_inner = member['role'] == 'inner'
                    
#                     # Create a separate feature for each ring
#                     feature = create_polygon_feature(coords, properties, is_inner)
#                     if feature:
#                         feature['properties']['role'] = member['role']
#                         features.append(feature)
    
#     return features

























# def load_geojsons_from_openstreetmap(rectangle_vertices):
#     # Create a bounding box from the rectangle vertices
#     min_lat = min(v[0] for v in rectangle_vertices)
#     max_lat = max(v[0] for v in rectangle_vertices)
#     min_lon = min(v[1] for v in rectangle_vertices)
#     max_lon = max(v[1] for v in rectangle_vertices)
    
#     # Construct the Overpass API query
#     overpass_url = "http://overpass-api.de/api/interpreter"
#     overpass_query = f"""
#     [out:json];
#     (
#       way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
#       way["building:part"]({min_lat},{min_lon},{max_lat},{max_lon});
#       relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
#     );
#     out geom;
#     """
    
#     # Send the request to the Overpass API
#     response = requests.get(overpass_url, params={'data': overpass_query})
#     data = response.json()
    
#     # Process the response and create GeoJSON features
#     features = []
    
#     def process_coordinates(geometry):
#         """Helper function to process and reverse coordinate pairs"""
#         return [coord[::-1] for coord in geometry]
    
#     def get_height_from_properties(properties):
#         """Helper function to extract height from properties, using levels if height is not available"""
#         # Try to get explicit height first
#         height = properties.get('height', properties.get('building:height', None))
#         if height is not None:
#             try:
#                 return float(height)
#             except ValueError:
#                 pass
        
#         # If no height, try to get levels
#         levels = properties.get('building:levels', properties.get('levels', None))
#         if levels is not None:
#             try:
#                 return float(levels) * 5.0  # Assume 5 meters per level
#             except ValueError:
#                 pass
        
#         return 0  # Default height if no valid height or levels found
    
#     def extract_properties(element):
#         """Helper function to extract and process properties"""
#         properties = element.get('tags', {})
        
#         # Get height (now using the helper function)
#         height = get_height_from_properties(properties)
            
#         # Get min_height value
#         min_height = properties.get('min_height', '0')
#         try:
#             min_height = float(min_height)
#         except ValueError:
#             min_height = 0
        
#         # Get levels information (store it in properties even if we use it for height)
#         levels = properties.get('building:levels', properties.get('levels', None))
#         try:
#             levels = float(levels) if levels is not None else None
#         except ValueError:
#             levels = None
            
#         return {
#             "height": height,
#             "min_height": min_height,
#             "confidence": -1.0,
#             "is_inner": False,  # Default value, will be overridden for inner rings
#             "levels": levels,  # Store the levels information in properties
#             "height_source": "explicit" if properties.get('height') or properties.get('building:height') 
#                            else "levels" if levels is not None 
#                            else "default"  # Add information about height source
#         }
    
#     def create_polygon_feature(coords, properties, is_inner=False):
#         """Helper function to create a polygon feature"""
#         if len(coords) >= 4:
#             properties = properties.copy()  # Create a copy to avoid modifying the original
#             properties["is_inner"] = is_inner
#             return {
#                 "type": "Feature",
#                 "properties": properties,
#                 "geometry": {
#                     "type": "Polygon",
#                     "coordinates": [process_coordinates(coords)]
#                 }
#             }
#         return None
    
#     for element in data['elements']:
#         if element['type'] == 'way':
#             # Process simple polygons from ways
#             coords = [(node['lon'], node['lat']) for node in element['geometry']]
#             feature = create_polygon_feature(coords, extract_properties(element))
#             if feature:
#                 features.append(feature)
                
#         elif element['type'] == 'relation':
#             properties = extract_properties(element)
            
#             # Process each member separately
#             for member in element['members']:
#                 if 'geometry' in member:
#                     coords = [(node['lon'], node['lat']) for node in member['geometry']]
#                     is_inner = member['role'] == 'inner'
                    
#                     # Create a separate feature for each ring
#                     feature = create_polygon_feature(coords, properties, is_inner)
#                     if feature:
#                         # For inner rings, we might want to add additional properties
#                         if is_inner:
#                             feature['properties']['role'] = 'inner'
#                         else:
#                             feature['properties']['role'] = 'outer'
                        
#                         features.append(feature)
    
#     return features

# Convert Overpass JSON to GeoJSON
def overpass_to_geojson(data):
    nodes = {}
    for element in data['elements']:
        if element['type'] == 'node':
            nodes[element['id']] = (element['lat'], element['lon'])

    features = []
    for element in data['elements']:
        if element['type'] == 'way':
            coords = [nodes[node_id] for node_id in element['nodes']]
            properties = element.get('tags', {})
            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coords],
                },
                'properties': properties,
            }
            features.append(feature)

    geojson = {
        'type': 'FeatureCollection',
        'features': features,
    }
    return geojson

def load_geojsons_from_osmbuildings(rectangle_vertices): 

    # Extract latitudes and longitudes
    lats = [coord[0] for coord in rectangle_vertices]
    lons = [coord[1] for coord in rectangle_vertices]

    # Find minimum and maximum values
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    # Overpass API query to get buildings with 3D attributes
    overpass_url = "https://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:60];
    (
      way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out body;
    >;
    out skel qt;
    """

    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()

    geojson_data = overpass_to_geojson(data)

    # Load your current GeoJSON data
    # Replace 'your_current_geojson_string' with your actual data or file path
    current_geojson = geojson_data

    desirable_features = []

    for feature in current_geojson['features']:
        converted_feature = convert_feature(feature)
        if converted_feature:
            desirable_features.append(converted_feature)
    
    return desirable_features

def convert_feature(feature):
    new_feature = {}
    new_feature['type'] = 'Feature'
    new_feature['properties'] = {}
    new_feature['geometry'] = {}

    # Convert geometry
    geometry = feature['geometry']
    geom_type = geometry['type']

    # Convert MultiPolygon to Polygon if necessary
    if geom_type == 'MultiPolygon':
        # Flatten MultiPolygon to Polygon by taking the first polygon
        # Alternatively, you can merge all polygons into one if needed
        coordinates = geometry['coordinates'][0]  # Take the first polygon
        if len(coordinates[0]) < 3:
            return None
    elif geom_type == 'Polygon':
        coordinates = geometry['coordinates']
        if len(coordinates[0]) < 3:
            return None
    else:
        # Skip features that are not polygons
        return None

    # Reformat coordinates: convert lists to tuples
    new_coordinates = []
    for ring in coordinates:
        new_ring = []
        for coord in ring:
            # Swap the order if needed (assuming original is [lat, lon])
            lat, lon = coord
            new_ring.append((lat, lon))
        new_coordinates.append(new_ring)

    new_feature['geometry']['type'] = 'Polygon'
    new_feature['geometry']['coordinates'] = new_coordinates

    # Process properties
    properties = feature.get('properties', {})
    height = properties.get('height')

    # If height is not available, estimate it (optional)
    if not height:
        levels = properties.get('building:levels')
        if levels:
            if type(levels)==str:
                # Default height if not specified
                height = 10.0  # You can adjust this default value as needed
            else:
                # Assuming average height per level is 3 meters
                height = float(levels) * 3.0
        else:
            # Default height if not specified
            height = 10.0  # You can adjust this default value as needed

    new_feature['properties']['height'] = float(height)
    new_feature['properties']['confidence'] = -1.0  # As per your desirable format

    return new_feature

# Land cover classifications mapping with class names
classification_mapping = {
    1: {'name': 'Bareland', 'tags': ['quarry', 'brownfield', 'bare_rock', 'scree', 'shingle', 'rock', 'sand', 'desert', 'landfill', 'beach']},
    2: {'name': 'Rangeland', 'tags': ['grass', 'meadow', 'grassland', 'heath', 'scrub', 'garden', 'park']},
    3: {'name': 'Developed space', 'tags': ['industrial', 'retail', 'commercial', 'residential', 'construction', 'railway', 'parking', 'islet', 'island']},
    4: {'name': 'Road', 'tags': ['highway']},
    5: {'name': 'Tree', 'tags': ['wood', 'forest', 'tree', 'tree_row']},
    6: {'name': 'Water', 'tags': ['water', 'waterway', 'reservoir', 'basin', 'bay', 'ocean']},
    7: {'name': 'Agriculture land', 'tags': ['farmland', 'orchard', 'vineyard', 'plant_nursery', 'greenhouse_horticulture', 'flowerbed', 'allotments']},
    8: {'name': 'Building', 'tags': ['building']}
}
# Function to assign classification code and name based on tags
def get_classification(tags):
    for code, info in classification_mapping.items():
        tag_list = info['tags']
        for tag in tag_list:
            if tag in tags.values():
                return code, info['name']
            # Special handling for keys with any value
            if tag == 'highway' and 'highway' in tags:
                return code, info['name']
            if tag == 'building' and 'building' in tags:
                return code, info['name']
            if tag == 'waterway' and 'waterway' in tags:
                return code, info['name']
            if tag in ['islet', 'island'] and 'place' in tags and tags['place'] == tag:
                return code, info['name']
        # Additional check for 'area:highway' (roads mapped as areas)
        if 'area:highway' in tags:
            return 4, 'Road'
    return None, None

# Function to swap coordinates from (lon, lat) to (lat, lon)
def swap_coordinates(geom_mapping):
    geom_type = geom_mapping['type']
    coords = geom_mapping['coordinates']

    def swap_coords(coord_list):
        if isinstance(coord_list[0], (list, tuple)):
            return [swap_coords(c) for c in coord_list]
        else:
            lon, lat = coord_list
            return [lat, lon]

    new_coords = swap_coords(coords)
    geom_mapping['coordinates'] = new_coords
    return geom_mapping

def load_land_cover_geojson_from_osm(rectangle_vertices_ori):
    # Close the rectangle polygon if needed
    rectangle_vertices = rectangle_vertices_ori.copy()
    rectangle_vertices.append(rectangle_vertices_ori[0])

    # Convert vertices to a string for the Overpass query (lat lon)
    polygon_coords = ' '.join(f"{lat} {lon}" for lat, lon in rectangle_vertices)

    # Extract OSM keys and values from the classification mapping
    osm_keys_values = {
        'landuse': [],
        'natural': [],
        'leisure': [],
        'amenity': [],
        'highway': [],
        'building': [],
        'place': []
    }

    for info in classification_mapping.values():
        tags = info['tags']
        for tag in tags:
            if tag in ['islet', 'island']:
                osm_keys_values['place'].append(tag)  # Adding 'place' key
            if tag in ['industrial', 'retail', 'commercial', 'construction', 'railway', 'farmland', 'orchard', 'vineyard', 'residential', 
                      'plant_nursery', 'greenhouse_horticulture', 'flowerbed', 'allotments', 'quarry', 'brownfield', 'landfill',
                      'grass', 'meadow', 'forest', 'reservoir', 'basin']:
                osm_keys_values['landuse'].append(tag)
            elif tag in ['wood', 'tree', 'tree_row', 'bare_rock', 'scree', 'shingle', 'rock', 'sand', 'desert',
                        'grassland', 'heath', 'scrub', 'water', 'beach']:
                osm_keys_values['natural'].append(tag)
            elif tag in ['garden', 'park']:
                osm_keys_values['leisure'].append(tag)
            elif tag == 'parking':
                osm_keys_values['amenity'].append(tag)
            elif tag == 'highway':
                osm_keys_values['highway'].append('*')  # Fetch all highways
            elif tag == 'building':
                osm_keys_values['building'].append('*')  # Fetch all buildings
            elif tag in ['bay', 'ocean']:
                osm_keys_values['natural'].append(tag)
            elif tag in ['water', 'reservoir', 'basin']:
                osm_keys_values['natural'].append(tag)
            elif tag == 'waterway':
                osm_keys_values['waterway'] = ['*']
            elif tag in ['reservoir', 'basin']:
                osm_keys_values['landuse'].append(tag)

    # Build the Overpass API query
    query_parts = []

    # Add queries for each key
    for key, values in osm_keys_values.items():
        if values:
            if key == 'place':
                # Handle 'place' separately
                for place_value in values:
                    query_parts.append(f'way["place"="{place_value}"](poly:"{polygon_coords}");')
                    query_parts.append(f'relation["place"="{place_value}"](poly:"{polygon_coords}");')
            if values == ['*']:
                # Fetch all features with this key
                query_parts.append(f'way["{key}"](poly:"{polygon_coords}");')
                query_parts.append(f'relation["{key}"](poly:"{polygon_coords}");')
            else:
                # Remove duplicates
                values = list(set(values))
                # Fetch features with specific values
                values_regex = '|'.join(values)
                query_parts.append(f'way["{key}"~"^{values_regex}$"](poly:"{polygon_coords}");')
                query_parts.append(f'relation["{key}"~"^{values_regex}$"](poly:"{polygon_coords}");')

    # Add waterway separately if present
    if 'waterway' in osm_keys_values:
        query_parts.append(f'way["waterway"](poly:"{polygon_coords}");')
        query_parts.append(f'relation["waterway"](poly:"{polygon_coords}");')

    # Combine all query parts
    query_body = "\n  ".join(query_parts)
    query = (
        "[out:json];\n"
        "(\n"
        f"  {query_body}\n"
        ");\n"
        "out body;\n"
        ">;\n"
        "out skel qt;"
    )

    overpass_url = "http://overpass-api.de/api/interpreter"

    # Fetch data from Overpass API
    print("Fetching data from Overpass API...")
    response = requests.get(overpass_url, params={'data': query})
    response.raise_for_status()  # Check for request errors
    data = response.json()

    # Convert OSM data to GeoJSON
    print("Converting data to GeoJSON format...")
    geojson_data = json2geojson(data)

    # Create a shapely polygon from your rectangle (using (lon, lat) order)
    rectangle_polygon = Polygon([(lon, lat) for lat, lon in rectangle_vertices])

    # Center of the rectangle (for projection parameters)
    center_lat = sum(lat for lat, lon in rectangle_vertices) / len(rectangle_vertices)
    center_lon = sum(lon for lat, lon in rectangle_vertices) / len(rectangle_vertices)

    # Define the coordinate reference systems
    wgs84 = pyproj.CRS('EPSG:4326')
    aea = pyproj.CRS(proj='aea', lat_1=rectangle_polygon.bounds[1], lat_2=rectangle_polygon.bounds[3], lat_0=center_lat, lon_0=center_lon)

    # Create transformer objects
    project = pyproj.Transformer.from_crs(wgs84, aea, always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(aea, wgs84, always_xy=True).transform

    # Filter features that intersect with the rectangle and assign classification codes
    filtered_features = []

    for feature in geojson_data['features']:
        feature_id = feature['properties'].get('id', 'unknown')
        geom = shape(feature['geometry'])
        if not (geom.is_valid and geom.intersects(rectangle_polygon)):
            continue  # Skip invalid or non-intersecting geometries

        # Assign classification code and name
        tags = feature['properties'].get('tags', {})
        classification_code, classification_name = get_classification(tags)
        if classification_code is None:
            feature_id = feature.get('id', 'unknown id')
            print(f"Skipping feature id: {feature_id} due to missing classification.")
            continue  # Skip if no classification

        # print(f"Processing feature id: {feature_id}, type: {classification_name}")

        # Exclude footpaths for roads
        if classification_code == 4:
            highway_value = feature['properties']['tags'].get('highway', '')
            # Exclude footpaths, paths, pedestrian, steps, cycleway, bridleway
            if highway_value in ['footway', 'path', 'pedestrian', 'steps', 'cycleway', 'bridleway']:
                continue  # Skip this feature

            # Get width or lanes
            width_value = feature['properties']['tags'].get('width')
            lanes_value = feature['properties']['tags'].get('lanes')

            # Initialize buffer_distance
            buffer_distance = None

            if width_value is not None:
                try:
                    width_meters = float(width_value)
                    buffer_distance = width_meters / 2  # Half width for buffering
                except ValueError:
                    pass  # Invalid width value
            elif lanes_value is not None:
                try:
                    num_lanes = float(lanes_value)
                    width_meters = num_lanes * 3.0  # Assuming 3 meters per lane
                    buffer_distance = width_meters / 2
                except ValueError:
                    pass  # Invalid lanes value
            else:
                # Set a default width for roads without width or lanes information
                default_width_meters = 5.0  # Adjust as needed
                buffer_distance = default_width_meters / 2

            if buffer_distance is None:
                continue  # Skip if buffer_distance is None

            if geom.geom_type == 'LineString' or geom.geom_type == 'MultiLineString':
                # Project to a planar coordinate system for buffering
                geom_proj = transform(project, geom)
                # Buffer the line
                buffered_geom_proj = geom_proj.buffer(buffer_distance)
                # Project back to WGS84
                buffered_geom = transform(project_back, buffered_geom_proj)
                # Clip to rectangle polygon
                buffered_geom_clipped = buffered_geom.intersection(rectangle_polygon)
                # Use the buffered geometry
                geom = buffered_geom_clipped
            else:
                continue  # Skip if not LineString or MultiLineString
        else:
            # For non-road features, use the geometry as is
            pass  # 'geom' is already defined

        # Now, handle Polygon and MultiPolygon
        if geom.is_empty:
            continue  # Skip empty geometries

        # Now, handle Polygon and MultiPolygon
        if geom.geom_type == 'Polygon':
            # Swap coordinates to (lat, lon) order
            geom_mapping = mapping(geom)
            geom_mapping = swap_coordinates(geom_mapping)
            new_feature = {
                'type': 'Feature',
                'properties': {
                    'class': classification_name
                },
                'geometry': geom_mapping
            }
            filtered_features.append(new_feature)
        elif geom.geom_type == 'MultiPolygon':
            # Split into multiple Polygon features
            for poly in geom.geoms:
                # Swap coordinates to (lat, lon) order
                geom_mapping = mapping(poly)
                geom_mapping = swap_coordinates(geom_mapping)
                new_feature = {
                    'type': 'Feature',
                    'properties': {
                        'class': classification_name
                    },
                    'geometry': geom_mapping
                }
                filtered_features.append(new_feature)
        else:
            # Skip other geometry types
            pass

    # Create the final GeoJSON object
    return filtered_features
