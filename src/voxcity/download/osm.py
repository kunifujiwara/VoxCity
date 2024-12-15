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

def convert_feature(feature):
    """Convert a GeoJSON feature to the desired format with height information.
    
    Args:
        feature (dict): Input GeoJSON feature
        
    Returns:
        dict: Converted feature with height and confidence values, or None if invalid
    """
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

    # If height is not available, estimate it based on building levels
    if not height:
        levels = properties.get('building:levels')
        if levels:
            if type(levels)==str:
                # If levels is a string (invalid format), use default height
                height = 10.0  # Default height in meters
            else:
                # Calculate height based on number of levels
                height = float(levels) * 3.0  # Assume 3m per level
        else:
            # No level information available, use default height
            height = 10.0  # Default height in meters

    new_feature['properties']['height'] = float(height)
    new_feature['properties']['confidence'] = -1.0  # Confidence score for height estimate

    return new_feature


# Classification mapping defines the land cover/use classes and their associated tags
# The numbers (0-13) represent class codes used in the system
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

# Maps classification tags to specific OSM key-value pairs
# '*' means match any value for that key
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

def get_classification(tags):
    """Determine the classification code and name for a feature based on its OSM tags.
    
    Args:
        tags (dict): Dictionary of OSM tags
        
    Returns:
        tuple: (classification_code, classification_name) or (None, None) if no match
    """
    # Iterate through each classification code and its associated info
    for code, info in classification_mapping.items():
        # Check each tag associated with this classification
        for tag in info['tags']:
            osm_mappings = tag_osm_key_value_mapping.get(tag)
            if osm_mappings:
                # Check if the feature's tags match any of the OSM key-value pairs
                for key, value in osm_mappings.items():
                    if key in tags:
                        if value == '*' or tags[key] == value:
                            return code, info['name']
            # Special case for islets and islands
            if tag in ['islet', 'island'] and tags.get('place') == tag:
                return code, info['name']
    # Special case for roads mapped as areas
    if 'area:highway' in tags:
        return 11, 'Road'
    return None, None

def swap_coordinates(geom_mapping):
    """Swap coordinates from (lon, lat) to (lat, lon) order.
    
    Args:
        geom_mapping (dict): GeoJSON geometry object
        
    Returns:
        dict: Geometry with swapped coordinates
    """
    coords = geom_mapping['coordinates']

    def swap_coords(coord_list):
        # Recursively swap coordinates for nested lists
        if isinstance(coord_list[0], (list, tuple)):
            return [swap_coords(c) for c in coord_list]
        else:
            # Swap lon/lat to lat/lon
            lon, lat = coord_list
            return [lat, lon]

    geom_mapping['coordinates'] = swap_coords(coords)
    return geom_mapping

def load_land_cover_geojson_from_osm(rectangle_vertices_ori):
    """Load land cover data from OpenStreetMap within a given rectangular area.
    
    Args:
        rectangle_vertices_ori (list): List of (lat, lon) coordinates defining the rectangle
        
    Returns:
        list: List of GeoJSON features with land cover classifications
    """
    # Close the rectangle polygon by adding first vertex at the end
    rectangle_vertices = rectangle_vertices_ori.copy()
    rectangle_vertices.append(rectangle_vertices_ori[0])

    # Convert vertices to space-separated string for Overpass query
    polygon_coords = ' '.join(f"{lat} {lon}" for lat, lon in rectangle_vertices)

    # Initialize dictionary to store OSM keys and their allowed values
    osm_keys_values = defaultdict(list)

    # Build mapping of OSM keys to their possible values from classification mapping
    for info in classification_mapping.values():
        tags = info['tags']
        for tag in tags:
            osm_mappings = tag_osm_key_value_mapping.get(tag)
            if osm_mappings:
                for key, value in osm_mappings.items():
                    if value == '*':
                        osm_keys_values[key] = ['*']  # Match all values
                    else:
                        if osm_keys_values[key] != ['*'] and value not in osm_keys_values[key]:
                            osm_keys_values[key].append(value)

    # Build Overpass API query parts for each key-value pair
    query_parts = []
    for key, values in osm_keys_values.items():
        if values:
            if values == ['*']:
                # Query for any value of this key
                query_parts.append(f'way["{key}"](poly:"{polygon_coords}");')
                query_parts.append(f'relation["{key}"](poly:"{polygon_coords}");')
            else:
                # Remove duplicate values
                values = list(set(values))
                # Build regex pattern for specific values
                values_regex = '|'.join(values)
                query_parts.append(f'way["{key}"~"^{values_regex}$"](poly:"{polygon_coords}");')
                query_parts.append(f'relation["{key}"~"^{values_regex}$"](poly:"{polygon_coords}");')

    # Combine query parts into complete Overpass query
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

    # Overpass API endpoint
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Fetch data from Overpass API
    print("Fetching data from Overpass API...")
    response = requests.get(overpass_url, params={'data': query})
    response.raise_for_status()
    data = response.json()

    # Convert OSM data to GeoJSON format
    print("Converting data to GeoJSON format...")
    geojson_data = json2geojson(data)

    # Create shapely polygon from rectangle vertices (in lon,lat order)
    rectangle_polygon = Polygon([(lon, lat) for lat, lon in rectangle_vertices])

    # Calculate center point for projection
    center_lat = sum(lat for lat, lon in rectangle_vertices) / len(rectangle_vertices)
    center_lon = sum(lon for lat, lon in rectangle_vertices) / len(rectangle_vertices)

    # Set up coordinate reference systems for projection
    wgs84 = pyproj.CRS('EPSG:4326')  # Standard lat/lon
    # Albers Equal Area projection centered on area of interest
    aea = pyproj.CRS(proj='aea', lat_1=rectangle_polygon.bounds[1], lat_2=rectangle_polygon.bounds[3], lat_0=center_lat, lon_0=center_lon)

    # Create transformers for projecting coordinates
    project = pyproj.Transformer.from_crs(wgs84, aea, always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(aea, wgs84, always_xy=True).transform

    # Process and filter features
    filtered_features = []

    for feature in geojson_data['features']:
        # Convert feature geometry to shapely object
        geom = shape(feature['geometry'])
        if not (geom.is_valid and geom.intersects(rectangle_polygon)):
            continue

        # Get classification for feature
        tags = feature['properties'].get('tags', {})
        classification_code, classification_name = get_classification(tags)
        if classification_code is None:
            continue

        # Special handling for roads
        if classification_code == 11:
            highway_value = tags.get('highway', '')
            # Skip minor paths and walkways
            if highway_value in ['footway', 'path', 'pedestrian', 'steps', 'cycleway', 'bridleway']:
                continue

            # Determine road width for buffering
            width_value = tags.get('width')
            lanes_value = tags.get('lanes')
            buffer_distance = None

            # Calculate buffer distance based on width or number of lanes
            if width_value is not None:
                try:
                    width_meters = float(width_value)
                    buffer_distance = width_meters / 2
                except ValueError:
                    pass
            elif lanes_value is not None:
                try:
                    num_lanes = float(lanes_value)
                    width_meters = num_lanes * 3.0  # 3m per lane
                    buffer_distance = width_meters / 2
                except ValueError:
                    pass
            else:
                # Default road width
                buffer_distance = 2.5  # 5m total width

            if buffer_distance is None:
                continue

            # Buffer line features to create polygons
            if geom.geom_type in ['LineString', 'MultiLineString']:
                # Project to planar CRS, buffer, and project back
                geom_proj = transform(project, geom)
                buffered_geom_proj = geom_proj.buffer(buffer_distance)
                buffered_geom = transform(project_back, buffered_geom_proj)
                # Clip to rectangle
                geom = buffered_geom.intersection(rectangle_polygon)
            else:
                continue

        # Skip empty geometries
        if geom.is_empty:
            continue

        # Convert geometry to GeoJSON feature
        if geom.geom_type == 'Polygon':
            # Create single polygon feature
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
            # Split into separate polygon features
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

    return filtered_features