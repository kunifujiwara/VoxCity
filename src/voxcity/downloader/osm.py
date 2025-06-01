"""
Module for downloading and processing OpenStreetMap data.

This module provides functionality to download and process building footprints, land cover,
and other geographic features from OpenStreetMap. It handles downloading data via the Overpass API,
processing the responses, and converting them to standardized GeoJSON format with proper properties.

The module includes functions for:
- Converting OSM JSON to GeoJSON format
- Processing building footprints with height information
- Handling land cover classifications
- Managing coordinate systems and projections
- Processing roads and other geographic features
"""

import requests
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import transform
import pyproj
from collections import defaultdict
import requests
import json
from shapely.geometry import shape, mapping, Polygon, LineString, Point, MultiPolygon
from shapely.ops import transform
import pyproj
import pandas as pd
import geopandas as gpd

def osm_json_to_geojson(osm_data):
    """
    Convert OSM JSON data to GeoJSON format with proper handling of complex relations.
    
    Args:
        osm_data (dict): OSM JSON data from Overpass API
        
    Returns:
        dict: GeoJSON FeatureCollection
    """
    features = []
    
    # Create a mapping of node IDs to their coordinates
    nodes = {}
    ways = {}
    
    # First pass: index all nodes and ways
    for element in osm_data['elements']:
        if element['type'] == 'node':
            nodes[element['id']] = (element['lon'], element['lat'])
        elif element['type'] == 'way':
            ways[element['id']] = element
    
    # Second pass: generate features
    for element in osm_data['elements']:
        if element['type'] == 'node' and 'tags' in element and element['tags']:
            # Convert POI nodes to Point features
            feature = {
                'type': 'Feature',
                'properties': {
                    'id': element['id'],
                    'type': 'node',
                    'tags': element.get('tags', {})
                },
                'geometry': {
                    'type': 'Point',
                    'coordinates': [element['lon'], element['lat']]
                }
            }
            features.append(feature)
            
        elif element['type'] == 'way' and 'nodes' in element:
            # Skip ways that are part of relations - we'll handle those in relation processing
            if is_part_of_relation(element['id'], osm_data):
                continue
                
            # Process standalone way
            coords = get_way_coords(element, nodes)
            if not coords or len(coords) < 2:
                continue
                
            # Determine if it's a polygon or a line
            is_polygon = is_way_polygon(element)
            
            # Make sure polygons have valid geometry (closed loop with at least 4 points)
            if is_polygon:
                # For closed ways, make sure first and last coordinates are the same
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                
                # Check if we have enough coordinates for a valid polygon (at least 4)
                if len(coords) < 4:
                    # Not enough coordinates for a polygon, convert to LineString
                    is_polygon = False
            
            feature = {
                'type': 'Feature',
                'properties': {
                    'id': element['id'],
                    'type': 'way',
                    'tags': element.get('tags', {})
                },
                'geometry': {
                    'type': 'Polygon' if is_polygon else 'LineString',
                    'coordinates': [coords] if is_polygon else coords
                }
            }
            features.append(feature)
            
        elif element['type'] == 'relation' and 'members' in element and 'tags' in element:
            tags = element.get('tags', {})
            
            # Process multipolygon relations
            if tags.get('type') == 'multipolygon' or any(key in tags for key in ['natural', 'water', 'waterway']):
                # Group member ways by role
                members_by_role = {'outer': [], 'inner': []}
                
                for member in element['members']:
                    if member['type'] == 'way' and member['ref'] in ways:
                        role = member['role']
                        if role not in ['outer', 'inner']:
                            role = 'outer'  # Default to outer if role not specified
                        members_by_role[role].append(member['ref'])
                
                # Skip if no outer members
                if not members_by_role['outer']:
                    continue
                
                # Create rings from member ways
                outer_rings = create_rings_from_ways(members_by_role['outer'], ways, nodes)
                inner_rings = create_rings_from_ways(members_by_role['inner'], ways, nodes)
                
                # Skip if no valid outer rings
                if not outer_rings:
                    continue
                
                # Create feature based on number of outer rings
                if len(outer_rings) == 1:
                    # Single polygon with possible inner rings
                    feature = {
                        'type': 'Feature',
                        'properties': {
                            'id': element['id'],
                            'type': 'relation',
                            'tags': tags
                        },
                        'geometry': {
                            'type': 'Polygon',
                            'coordinates': [outer_rings[0]] + inner_rings
                        }
                    }
                else:
                    # MultiPolygon
                    # Each outer ring forms a polygon, and we assign inner rings to each polygon
                    # This is a simplification - proper assignment would check for containment
                    multipolygon_coords = []
                    for outer_ring in outer_rings:
                        polygon_coords = [outer_ring]
                        # For simplicity, assign all inner rings to the first polygon
                        # A more accurate implementation would check which outer ring contains each inner ring
                        if len(multipolygon_coords) == 0:
                            polygon_coords.extend(inner_rings)
                        multipolygon_coords.append(polygon_coords)
                    
                    feature = {
                        'type': 'Feature',
                        'properties': {
                            'id': element['id'],
                            'type': 'relation',
                            'tags': tags
                        },
                        'geometry': {
                            'type': 'MultiPolygon',
                            'coordinates': multipolygon_coords
                        }
                    }
                
                features.append(feature)
    
    return {
        'type': 'FeatureCollection',
        'features': features
    }

def is_part_of_relation(way_id, osm_data):
    """Check if a way is part of any relation in the OSM data.
    
    Args:
        way_id (int): The ID of the way to check
        osm_data (dict): OSM JSON data containing elements
        
    Returns:
        bool: True if the way is part of a relation, False otherwise
    """
    for element in osm_data['elements']:
        if element['type'] == 'relation' and 'members' in element:
            for member in element['members']:
                if member['type'] == 'way' and member['ref'] == way_id:
                    return True
    return False

def is_way_polygon(way):
    """Determine if a way should be treated as a polygon based on OSM tags and geometry.
    
    A way is considered a polygon if:
    1. It forms a closed loop (first and last nodes are the same)
    2. It has tags indicating it represents an area (building, landuse, etc.)
    
    Args:
        way (dict): OSM way element with nodes and tags
        
    Returns:
        bool: True if the way should be treated as a polygon, False otherwise
    """
    # Check if the way is closed (first and last nodes are the same)
    if 'nodes' in way and way['nodes'][0] == way['nodes'][-1]:
        # Check for tags that indicate this is an area
        if 'tags' in way:
            tags = way['tags']
            if 'building' in tags or ('area' in tags and tags['area'] == 'yes'):
                return True
            if any(k in tags for k in ['landuse', 'natural', 'water', 'leisure', 'amenity']):
                return True
    return False

def get_way_coords(way, nodes):
    """Extract coordinates for a way from its node references.
    
    Args:
        way (dict): OSM way element containing node references
        nodes (dict): Dictionary mapping node IDs to their coordinates
        
    Returns:
        list: List of coordinate pairs [(lon, lat), ...] for the way,
             or empty list if any nodes are missing
    """
    coords = []
    if 'nodes' not in way:
        return coords
        
    for node_id in way['nodes']:
        if node_id in nodes:
            coords.append(nodes[node_id])
        else:
            # Missing node - skip this way
            return []
    
    return coords

def create_rings_from_ways(way_ids, ways, nodes):
    """Create continuous rings by connecting ways that share nodes.
    
    This function handles complex relations by:
    1. Connecting ways that share end nodes
    2. Handling reversed way directions
    3. Closing rings when possible
    4. Converting node references to coordinates
    
    Args:
        way_ids (list): List of way IDs that make up the ring(s)
        ways (dict): Dictionary mapping way IDs to way elements
        nodes (dict): Dictionary mapping node IDs to coordinates
        
    Returns:
        list: List of rings, where each ring is a list of coordinate pairs [(lon, lat), ...]
              forming a closed polygon with at least 4 points
    """
    if not way_ids:
        return []
    
    # Extract node IDs for each way
    way_nodes = {}
    for way_id in way_ids:
        if way_id in ways and 'nodes' in ways[way_id]:
            way_nodes[way_id] = ways[way_id]['nodes']
    
    # If we have no valid ways, return empty list
    if not way_nodes:
        return []
    
    # Connect the ways to form rings
    rings = []
    unused_ways = set(way_nodes.keys())
    
    while unused_ways:
        # Start a new ring with the first unused way
        current_way_id = next(iter(unused_ways))
        unused_ways.remove(current_way_id)
        
        # Get the first and last node IDs of the current way
        current_nodes = way_nodes[current_way_id]
        if not current_nodes:
            continue
            
        # Start building a ring with the nodes of the first way
        ring_nodes = list(current_nodes)
        
        # Try to connect more ways to complete the ring
        connected = True
        while connected and unused_ways:
            connected = False
            
            # Get the first and last nodes of the current ring
            first_node = ring_nodes[0]
            last_node = ring_nodes[-1]
            
            # Try to find a way that connects to either end of our ring
            for way_id in list(unused_ways):
                nodes_in_way = way_nodes[way_id]
                if not nodes_in_way:
                    unused_ways.remove(way_id)
                    continue
                
                # Check if this way connects at the start of our ring
                if nodes_in_way[-1] == first_node:
                    # This way connects to the start of our ring (reversed)
                    ring_nodes = nodes_in_way[:-1] + ring_nodes
                    unused_ways.remove(way_id)
                    connected = True
                    break
                elif nodes_in_way[0] == first_node:
                    # This way connects to the start of our ring
                    ring_nodes = list(reversed(nodes_in_way))[:-1] + ring_nodes
                    unused_ways.remove(way_id)
                    connected = True
                    break
                # Check if this way connects at the end of our ring
                elif nodes_in_way[0] == last_node:
                    # This way connects to the end of our ring
                    ring_nodes.extend(nodes_in_way[1:])
                    unused_ways.remove(way_id)
                    connected = True
                    break
                elif nodes_in_way[-1] == last_node:
                    # This way connects to the end of our ring (reversed)
                    ring_nodes.extend(list(reversed(nodes_in_way))[1:])
                    unused_ways.remove(way_id)
                    connected = True
                    break
        
        # Check if the ring is closed (first node equals last node)
        if ring_nodes and ring_nodes[0] == ring_nodes[-1] and len(ring_nodes) >= 4:
            # Convert node IDs to coordinates
            ring_coords = []
            for node_id in ring_nodes:
                if node_id in nodes:
                    ring_coords.append(nodes[node_id])
                else:
                    # Missing node - skip this ring
                    ring_coords = []
                    break
            
            if ring_coords and len(ring_coords) >= 4:
                rings.append(ring_coords)
        else:
            # Try to close the ring if it's almost complete
            if ring_nodes and len(ring_nodes) >= 3 and ring_nodes[0] != ring_nodes[-1]:
                ring_nodes.append(ring_nodes[0])
                
                # Convert node IDs to coordinates
                ring_coords = []
                for node_id in ring_nodes:
                    if node_id in nodes:
                        ring_coords.append(nodes[node_id])
                    else:
                        # Missing node - skip this ring
                        ring_coords = []
                        break
                
                if ring_coords and len(ring_coords) >= 4:
                    rings.append(ring_coords)
    
    return rings

def load_gdf_from_openstreetmap(rectangle_vertices):
    """Download and process building footprint data from OpenStreetMap.
    
    This function:
    1. Downloads building data using the Overpass API
    2. Processes complex relations and their members
    3. Extracts height information and other properties
    4. Converts features to a GeoDataFrame with standardized properties
    
    Args:
        rectangle_vertices (list): List of (lon, lat) coordinates defining the bounding box
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing building footprints with properties:
            - geometry: Polygon or MultiPolygon
            - height: Building height in meters
            - levels: Number of building levels
            - min_height: Minimum height (for elevated structures)
            - building_type: Type of building
            - And other OSM tags as properties
    """
    # Create a bounding box from the rectangle vertices
    min_lon = min(v[0] for v in rectangle_vertices)
    max_lon = max(v[0] for v in rectangle_vertices)
    min_lat = min(v[1] for v in rectangle_vertices)
    max_lat = max(v[1] for v in rectangle_vertices)
    
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
    
    # Process the response and create features list
    features = []
    
    def process_coordinates(geometry):
        """Helper function to process and reverse coordinate pairs.
        
        Args:
            geometry: List of coordinate pairs to process
            
        Returns:
            list: Processed coordinate pairs with reversed order
        """
        return [coord for coord in geometry]  # Keep original order since already (lon, lat)
    
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
    
    # Convert features list to GeoDataFrame
    if not features:
        return gpd.GeoDataFrame()
        
    geometries = []
    properties_list = []
    
    for feature in features:
        geometries.append(shape(feature['geometry']))
        properties_list.append(feature['properties'])
        
    gdf = gpd.GeoDataFrame(properties_list, geometry=geometries, crs="EPSG:4326")
    return gdf

def convert_feature(feature):
    """Convert a GeoJSON feature to a standardized format with height information.
    
    This function:
    1. Handles both Polygon and MultiPolygon geometries
    2. Extracts and validates height information
    3. Ensures coordinate order consistency (lon, lat)
    4. Adds confidence scores for height estimates
    
    Args:
        feature (dict): Input GeoJSON feature with geometry and properties
        
    Returns:
        dict: Converted feature with:
            - Standardized geometry (always Polygon)
            - Height information in properties
            - Confidence score for height values
            Or None if the feature is invalid or not a polygon
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
            new_ring.append((lon, lat))  # Changed to (lon, lat)
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
    """Determine the land cover/use classification based on OSM tags.
    
    This function maps OSM tags to standardized land cover classes using:
    1. A hierarchical classification system (codes 0-13)
    2. Tag matching patterns for different feature types
    3. Special cases for roads, water bodies, etc.
    
    Args:
        tags (dict): Dictionary of OSM tags (key-value pairs)
        
    Returns:
        tuple: (classification_code, classification_name) where:
            - classification_code (int): Numeric code (0-13) for the land cover class
            - classification_name (str): Human-readable name of the class
            Or (None, None) if no matching classification is found
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
    """Swap coordinate order in a GeoJSON geometry object.
    
    This function:
    1. Handles nested coordinate structures (Polygons, MultiPolygons)
    2. Preserves the original coordinate order if already correct
    3. Works recursively for complex geometries
    
    Args:
        geom_mapping (dict): GeoJSON geometry object with coordinates
        
    Returns:
        dict: Geometry with coordinates in the correct order (lon, lat)
    """
    coords = geom_mapping['coordinates']

    def swap_coords(coord_list):
        # Recursively swap coordinates for nested lists
        if isinstance(coord_list[0], (list, tuple)):
            return [swap_coords(c) for c in coord_list]
        else:
            # Keep original order since already (lon, lat)
            return coord_list

    geom_mapping['coordinates'] = swap_coords(coords)
    return geom_mapping

def load_land_cover_gdf_from_osm(rectangle_vertices_ori):
    """Load and classify land cover data from OpenStreetMap.
    
    This function:
    1. Downloads land cover features using the Overpass API
    2. Classifies features based on OSM tags
    3. Handles special cases like roads with width information
    4. Projects geometries for accurate buffering
    5. Creates a standardized GeoDataFrame with classifications
    
    Args:
        rectangle_vertices_ori (list): List of (lon, lat) coordinates defining the area
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame with:
            - geometry: Polygon or MultiPolygon features
            - class: Land cover classification name
            - Additional properties from OSM tags
    """
    # Close the rectangle polygon by adding first vertex at the end
    rectangle_vertices = rectangle_vertices_ori.copy()
    rectangle_vertices.append(rectangle_vertices_ori[0])

    # Instead of using poly:"lat lon lat lon...", use area coordinates
    min_lat = min(lat for lon, lat in rectangle_vertices)
    max_lat = max(lat for lon, lat in rectangle_vertices)
    min_lon = min(lon for lon, lat in rectangle_vertices)
    max_lon = max(lon for lon, lat in rectangle_vertices)

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
                # Query for any value of this key using bounding box
                query_parts.append(f'way["{key}"]({min_lat},{min_lon},{max_lat},{max_lon});')
                query_parts.append(f'relation["{key}"]({min_lat},{min_lon},{max_lat},{max_lon});')
            else:
                # Remove duplicate values
                values = list(set(values))
                # Build regex pattern for specific values
                values_regex = '|'.join(values)
                query_parts.append(f'way["{key}"~"^{values_regex}$"]({min_lat},{min_lon},{max_lat},{max_lon});')
                query_parts.append(f'relation["{key}"~"^{values_regex}$"]({min_lat},{min_lon},{max_lat},{max_lon});')

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

    # Convert OSM data to GeoJSON format using our custom converter instead of json2geojson
    print("Converting data to GeoJSON format...")
    geojson_data = osm_json_to_geojson(data)

    # Create shapely polygon from rectangle vertices (in lon,lat order)
    rectangle_polygon = Polygon(rectangle_vertices)

    # Calculate center point for projection
    center_lat = sum(lat for lon, lat in rectangle_vertices) / len(rectangle_vertices)
    center_lon = sum(lon for lon, lat in rectangle_vertices) / len(rectangle_vertices)

    # Set up coordinate reference systems for projection
    wgs84 = pyproj.CRS('EPSG:4326')  # Standard lat/lon
    # Albers Equal Area projection centered on area of interest
    aea = pyproj.CRS(proj='aea', lat_1=rectangle_polygon.bounds[1], lat_2=rectangle_polygon.bounds[3], lat_0=center_lat, lon_0=center_lon)

    # Create transformers for projecting coordinates
    project = pyproj.Transformer.from_crs(wgs84, aea, always_xy=True).transform
    project_back = pyproj.Transformer.from_crs(aea, wgs84, always_xy=True).transform

    # Lists to store geometries and properties for GeoDataFrame
    geometries = []
    properties = []

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

        # Add geometries and properties
        if geom.geom_type == 'Polygon':
            geometries.append(geom)
            properties.append({'class': classification_name})
        elif geom.geom_type == 'MultiPolygon':
            for poly in geom.geoms:
                geometries.append(poly)
                properties.append({'class': classification_name})

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
    return gdf