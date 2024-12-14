"""
Module for downloading and processing OpenStreetMap data.

This module provides functionality to download and process building footprints, land cover,
and other geographic features from OpenStreetMap. It handles downloading data via the Overpass API,
processing the responses, and converting them to standardized GeoJSON format with proper properties.
"""

import requests
from shapely.geometry import Polygon
# Import libraries
import requests
from osm2geojson import json2geojson
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import transform
import pyproj
from collections import defaultdict
import requests
import json
from shapely.geometry import shape, mapping, Polygon
from shapely.ops import transform
import pyproj
from osm2geojson import json2geojson

def load_geojsons_from_openstreetmap(rectangle_vertices):
    """Download and process building footprint data from OpenStreetMap.
    
    Args:
        rectangle_vertices: List of (lat, lon) coordinates defining the bounding box
        
    Returns:
        list: List of GeoJSON features containing building footprints with standardized properties
    """
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
    
    # Build a mapping from (type, id) to element
    id_map = {}
    for element in data['elements']:
        id_map[(element['type'], element['id'])] = element
    
    # Process the response and create GeoJSON features
    features = []
    
    def process_coordinates(geometry):
        """Helper function to process and reverse coordinate pairs.
        
        Args:
            geometry: List of coordinate pairs to process
            
        Returns:
            list: Processed coordinate pairs with reversed order
        """
        return [coord[::-1] for coord in geometry]
    
    def get_height_from_properties(properties):
        """Helper function to extract height from properties.
        
        Args:
            properties: Dictionary of feature properties
            
        Returns:
            float: Extracted or calculated height value
        """
        height = properties.get('height', properties.get('building:height', None))
        if height is not None:
            try:
                return float(height)
            except ValueError:
                pass
        
        return 0  # Default height if no valid height found
    
    def extract_properties(element):
        """Helper function to extract and process properties from an element.
        
        Args:
            element: OSM element containing tags and properties
            
        Returns:
            dict: Processed properties dictionary
        """
        properties = element.get('tags', {})
        
        # Get height (now using the helper function)
        height = get_height_from_properties(properties)
            
        # Get min_height and min_level
        min_height = properties.get('min_height', '0')
        min_level = properties.get('building:min_level', properties.get('min_level', '0'))
        try:
            min_height = float(min_height)
        except ValueError:
            min_height = 0
        
        levels = properties.get('building:levels', properties.get('levels', None))
        try:
            levels = float(levels) if levels is not None else None
        except ValueError:
            levels = None
                
        # Extract additional properties, including those relevant to artworks
        extracted_props = {
            "id": element['id'],
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
        """Helper function to create a polygon feature.
        
        Args:
            coords: List of coordinate pairs defining the polygon
            properties: Dictionary of feature properties
            is_inner: Boolean indicating if this is an inner ring
            
        Returns:
            dict: GeoJSON Feature object or None if invalid
        """
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
            if 'geometry' in element:
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                properties = extract_properties(element)
                feature = create_polygon_feature(coords, properties)
                if feature:
                    features.append(feature)
                    
        elif element['type'] == 'relation':
            properties = extract_properties(element)
            
            # Process each member of the relation
            for member in element['members']:
                if member['type'] == 'way':
                    # Look up the way in id_map
                    way = id_map.get(('way', member['ref']))
                    if way and 'geometry' in way:
                        coords = [(node['lon'], node['lat']) for node in way['geometry']]
                        is_inner = member['role'] == 'inner'
                        member_properties = properties.copy()
                        member_properties['member_id'] = way['id']  # Include id of the way
                        feature = create_polygon_feature(coords, member_properties, is_inner)
                        if feature:
                            feature['properties']['role'] = member['role']
                            features.append(feature)
        
    return features

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


# New classification_mapping with reordered classes
classification_mapping = {
    11: {'name': 'Road', 'tags': ['highway', 'road', 'path', 'track', 'street']},
    12: {'name': 'Building', 'tags': ['building', 'house', 'apartment', 'commercial_building', 'industrial_building']},
    10: {'name': 'Developed space', 'tags': ['industrial', 'retail', 'commercial', 'residential', 'construction', 'railway', 'parking', 'islet', 'island']},
    0: {'name': 'Bareland', 'tags': ['quarry', 'brownfield', 'bare_rock', 'scree', 'shingle', 'rock', 'sand', 'desert', 'landfill', 'beach']},
    1: {'name': 'Rangeland', 'tags': ['grass', 'meadow', 'grassland', 'heath', 'garden', 'park']},
    2: {'name': 'Shrub', 'tags': ['scrub', 'shrubland', 'bush', 'thicket']},
    3: {'name': 'Agriculture land', 'tags': ['farmland', 'orchard', 'vineyard', 'plant_nursery', 'greenhouse_horticulture', 'flowerbed', 'allotments', 'cropland']},
    4: {'name': 'Tree', 'tags': ['wood', 'forest', 'tree', 'tree_row', 'tree_canopy']},
    5: {'name': 'Moss and lichen', 'tags': ['moss', 'lichen', 'tundra_vegetation']},
    6: {'name': 'Wet land', 'tags': ['wetland', 'marsh', 'swamp', 'bog', 'fen', 'flooded_vegetation']},
    7: {'name': 'Mangrove', 'tags': ['mangrove', 'mangrove_forest', 'mangrove_swamp']},
    8: {'name': 'Water', 'tags': ['water', 'waterway', 'reservoir', 'basin', 'bay', 'ocean', 'sea', 'river', 'lake']},
    9: {'name': 'Snow and ice', 'tags': ['glacier', 'snow', 'ice', 'snowfield', 'ice_shelf']},
    13: {'name': 'No Data', 'tags': ['unknown', 'no_data', 'clouds', 'undefined']}
}

# Mapping of classification tags to OSM key-value pairs
tag_osm_key_value_mapping = {
    # Road
    'highway': {'highway': '*'},
    'road': {'highway': '*'},
    'path': {'highway': 'path'},
    'track': {'highway': 'track'},
    'street': {'highway': '*'},
    
    # Building
    'building': {'building': '*'},
    'house': {'building': 'house'},
    'apartment': {'building': 'apartments'},
    'commercial_building': {'building': 'commercial'},
    'industrial_building': {'building': 'industrial'},
    
    # Developed space
    'industrial': {'landuse': 'industrial'},
    'retail': {'landuse': 'retail'},
    'commercial': {'landuse': 'commercial'},
    'residential': {'landuse': 'residential'},
    'construction': {'landuse': 'construction'},
    'railway': {'landuse': 'railway'},
    'parking': {'amenity': 'parking'},
    'islet': {'place': 'islet'},
    'island': {'place': 'island'},
    
    # Bareland
    'quarry': {'landuse': 'quarry'},
    'brownfield': {'landuse': 'brownfield'},
    'bare_rock': {'natural': 'bare_rock'},
    'scree': {'natural': 'scree'},
    'shingle': {'natural': 'shingle'},
    'rock': {'natural': 'rock'},
    'sand': {'natural': 'sand'},
    'desert': {'natural': 'desert'},
    'landfill': {'landuse': 'landfill'},
    'beach': {'natural': 'beach'},
    
    # Rangeland
    'grass': {'landuse': 'grass'},
    'meadow': {'landuse': 'meadow'},
    'grassland': {'natural': 'grassland'},
    'heath': {'natural': 'heath'},
    'garden': {'leisure': 'garden'},
    'park': {'leisure': 'park'},
    
    # Shrub
    'scrub': {'natural': 'scrub'},
    'shrubland': {'natural': 'scrub'},
    'bush': {'natural': 'scrub'},
    'thicket': {'natural': 'scrub'},
    
    # Agriculture land
    'farmland': {'landuse': 'farmland'},
    'orchard': {'landuse': 'orchard'},
    'vineyard': {'landuse': 'vineyard'},
    'plant_nursery': {'landuse': 'plant_nursery'},
    'greenhouse_horticulture': {'landuse': 'greenhouse_horticulture'},
    'flowerbed': {'landuse': 'flowerbed'},
    'allotments': {'landuse': 'allotments'},
    'cropland': {'landuse': 'farmland'},
    
    # Tree
    'wood': {'natural': 'wood'},
    'forest': {'landuse': 'forest'},
    'tree': {'natural': 'tree'},
    'tree_row': {'natural': 'tree_row'},
    'tree_canopy': {'natural': 'tree_canopy'},
    
    # Moss and lichen
    'moss': {'natural': 'fell'},
    'lichen': {'natural': 'fell'},
    'tundra_vegetation': {'natural': 'fell'},
    
    # Wet land
    'wetland': {'natural': 'wetland'},
    'marsh': {'wetland': 'marsh'},
    'swamp': {'wetland': 'swamp'},
    'bog': {'wetland': 'bog'},
    'fen': {'wetland': 'fen'},
    'flooded_vegetation': {'natural': 'wetland'},
    
    # Mangrove
    'mangrove': {'natural': 'wetland', 'wetland': 'mangrove'},
    'mangrove_forest': {'natural': 'wetland', 'wetland': 'mangrove'},
    'mangrove_swamp': {'natural': 'wetland', 'wetland': 'mangrove'},
    
    # Water
    'water': {'natural': 'water'},
    'waterway': {'waterway': '*'},
    'reservoir': {'landuse': 'reservoir'},
    'basin': {'landuse': 'basin'},
    'bay': {'natural': 'bay'},
    'ocean': {'natural': 'water', 'water': 'ocean'},
    'sea': {'natural': 'water', 'water': 'sea'},
    'river': {'waterway': 'river'},
    'lake': {'natural': 'water', 'water': 'lake'},
    
    # Snow and ice
    'glacier': {'natural': 'glacier'},
    'snow': {'natural': 'glacier'},
    'ice': {'natural': 'glacier'},
    'snowfield': {'natural': 'glacier'},
    'ice_shelf': {'natural': 'glacier'},
    
    # No Data
    'unknown': {'FIXME': '*'},
    'no_data': {'FIXME': '*'},
    'clouds': {'natural': 'cloud'},
    'undefined': {'FIXME': '*'}
}

# Function to assign classification code and name based on tags
def get_classification(tags):
    for code, info in classification_mapping.items():
        for tag in info['tags']:
            osm_mappings = tag_osm_key_value_mapping.get(tag)
            if osm_mappings:
                for key, value in osm_mappings.items():
                    if key in tags:
                        if value == '*' or tags[key] == value:
                            # Debug statement to trace matching
                            # print(f"Matched tag '{tag}' with OSM mapping '{key}: {value}' for class '{info['name']}'")
                            return code, info['name']
            # Special handling for 'place' with 'islet' and 'island'
            if tag in ['islet', 'island'] and tags.get('place') == tag:
                return code, info['name']
    # Additional check for 'area:highway' (roads mapped as areas)
    if 'area:highway' in tags:
        return 11, 'Road'
    return None, None

# Function to swap coordinates from (lon, lat) to (lat, lon)
def swap_coordinates(geom_mapping):
    coords = geom_mapping['coordinates']

    def swap_coords(coord_list):
        if isinstance(coord_list[0], (list, tuple)):
            return [swap_coords(c) for c in coord_list]
        else:
            lon, lat = coord_list
            return [lat, lon]

    geom_mapping['coordinates'] = swap_coords(coords)
    return geom_mapping

def load_land_cover_geojson_from_osm(rectangle_vertices_ori):
    # Close the rectangle polygon if needed
    rectangle_vertices = rectangle_vertices_ori.copy()
    rectangle_vertices.append(rectangle_vertices_ori[0])

    # Convert vertices to a string for the Overpass query (lat lon)
    polygon_coords = ' '.join(f"{lat} {lon}" for lat, lon in rectangle_vertices)

    # Initialize osm_keys_values
    osm_keys_values = defaultdict(list)

    # Map tags to osm_keys_values
    for info in classification_mapping.values():
        tags = info['tags']
        for tag in tags:
            osm_mappings = tag_osm_key_value_mapping.get(tag)
            if osm_mappings:
                for key, value in osm_mappings.items():
                    if value == '*':
                        osm_keys_values[key] = ['*']  # Fetch all values for this key
                    else:
                        if osm_keys_values[key] != ['*'] and value not in osm_keys_values[key]:
                            osm_keys_values[key].append(value)

    # Build the Overpass API query
    query_parts = []

    # Add queries for each key
    for key, values in osm_keys_values.items():
        if values:
            if values == ['*']:
                # Fetch all features with this key
                query_parts.append(f'way["{key}"](poly:"{polygon_coords}");')
                query_parts.append(f'relation["{key}"](poly:"{polygon_coords}");')
            else:
                # Remove duplicates
                values = list(set(values))
                # Build a regex pattern for the values
                values_regex = '|'.join(values)
                query_parts.append(f'way["{key}"~"^{values_regex}$"](poly:"{polygon_coords}");')
                query_parts.append(f'relation["{key}"~"^{values_regex}$"](poly:"{polygon_coords}");')

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
        geom = shape(feature['geometry'])
        if not (geom.is_valid and geom.intersects(rectangle_polygon)):
            continue  # Skip invalid or non-intersecting geometries

        # Assign classification code and name
        tags = feature['properties'].get('tags', {})
        classification_code, classification_name = get_classification(tags)
        if classification_code is None:
            continue  # Skip if no classification

        # Exclude footpaths for roads
        if classification_code == 11:
            highway_value = tags.get('highway', '')
            # Exclude footpaths, paths, pedestrian, steps, cycleway, bridleway
            if highway_value in ['footway', 'path', 'pedestrian', 'steps', 'cycleway', 'bridleway']:
                continue  # Skip this feature

            # Get width or lanes
            width_value = tags.get('width')
            lanes_value = tags.get('lanes')

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

            if geom.geom_type in ['LineString', 'MultiLineString']:
                # Project to a planar coordinate system for buffering
                geom_proj = transform(project, geom)
                # Buffer the line
                buffered_geom_proj = geom_proj.buffer(buffer_distance)
                # Project back to WGS84
                buffered_geom = transform(project_back, buffered_geom_proj)
                # Clip to rectangle polygon
                geom = buffered_geom.intersection(rectangle_polygon)
            else:
                continue  # Skip if not LineString or MultiLineString

        # Now, handle Polygon and MultiPolygon
        if geom.is_empty:
            continue  # Skip empty geometries

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

    return filtered_features