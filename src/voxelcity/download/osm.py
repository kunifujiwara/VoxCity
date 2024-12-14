"""
Module for downloading and processing OpenStreetMap data.

This module provides functionality to download and process building footprints, land cover,
and other geographic features from OpenStreetMap. It handles downloading data via the Overpass API,
processing the responses, and converting them to standardized GeoJSON format with proper properties.
"""

import requests
from osm2geojson import json2geojson
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import transform
import pyproj

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
    # Query gets buildings, building parts, building relations, and artwork areas
    # The (._; >;) syntax recursively gets all referenced nodes/ways/relations
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
    
    # Build a mapping from (type, id) to element for quick lookups when processing relations
    id_map = {}
    for element in data['elements']:
        id_map[(element['type'], element['id'])] = element
    
    # Process the response and create GeoJSON features
    features = []
    
    def process_coordinates(geometry):
        """Helper function to process and reverse coordinate pairs.
        
        OSM returns coordinates as [lon,lat] but GeoJSON expects [lat,lon],
        so we need to reverse each coordinate pair.
        
        Args:
            geometry: List of coordinate pairs to process
            
        Returns:
            list: Processed coordinate pairs with reversed order
        """
        return [coord[::-1] for coord in geometry]
    
    def get_height_from_properties(properties):
        """Helper function to extract height from properties.
        
        Attempts to get height from either 'height' or 'building:height' tags.
        Returns 0 if no valid height found.
        
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
        
        Extracts and standardizes properties including:
        - Building heights and levels
        - Building materials and colors
        - Roof characteristics
        - Architectural details
        - Contact information
        - Accessibility info
        - Tourism/artwork properties
        
        Args:
            element: OSM element containing tags and properties
            
        Returns:
            dict: Processed properties dictionary with None values removed
        """
        properties = element.get('tags', {})
        
        # Get height (now using the helper function)
        height = get_height_from_properties(properties)
            
        # Get min_height and min_level for multi-level structures
        min_height = properties.get('min_height', '0')
        min_level = properties.get('building:min_level', properties.get('min_level', '0'))
        try:
            min_height = float(min_height)
        except ValueError:
            min_height = 0
        
        # Extract number of levels, used as fallback for height calculation
        levels = properties.get('building:levels', properties.get('levels', None))
        try:
            levels = float(levels) if levels is not None else None
        except ValueError:
            levels = None
                
        # Extract comprehensive set of properties for detailed building information
        extracted_props = {
            "id": element['id'],
            "height": height,
            "min_height": min_height,
            "confidence": -1.0,  # Default confidence for OSM data
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
        
        Creates a GeoJSON Feature object if the polygon has at least 4 points
        (minimum required for a valid polygon including closure).
        
        Args:
            coords: List of coordinate pairs defining the polygon
            properties: Dictionary of feature properties
            is_inner: Boolean indicating if this is an inner ring (hole in building)
            
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
    
    # Process each element, handling both simple ways and complex relations
    for element in data['elements']:
        if element['type'] == 'way':
            # Process simple polygons (ways)
            if 'geometry' in element:
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
                properties = extract_properties(element)
                feature = create_polygon_feature(coords, properties)
                if feature:
                    features.append(feature)
                    
        elif element['type'] == 'relation':
            # Process relations (complex polygons with possible holes)
            properties = extract_properties(element)
            
            # Process each member of the relation
            for member in element['members']:
                if member['type'] == 'way':
                    # Look up the way in id_map
                    way = id_map.get(('way', member['ref']))
                    if way and 'geometry' in way:
                        coords = [(node['lon'], node['lat']) for node in way['geometry']]
                        is_inner = member['role'] == 'inner'  # Check if this is a hole in the building
                        member_properties = properties.copy()
                        member_properties['member_id'] = way['id']  # Include id of the way
                        feature = create_polygon_feature(coords, member_properties, is_inner)
                        if feature:
                            feature['properties']['role'] = member['role']
                            features.append(feature)
        
    return features
