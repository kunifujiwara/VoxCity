import requests
from shapely.geometry import Polygon

def load_geojsons_from_openstreetmap(rectangle_vertices):
    # Create a bounding box from the rectangle vertices
    min_lat = min(v[0] for v in rectangle_vertices)
    max_lat = max(v[0] for v in rectangle_vertices)
    min_lon = min(v[1] for v in rectangle_vertices)
    max_lon = max(v[1] for v in rectangle_vertices)
    
    # Construct the Overpass API query
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    (
      way["building"]({min_lat},{min_lon},{max_lat},{max_lon});
      relation["building"]({min_lat},{min_lon},{max_lat},{max_lon});
    );
    out geom;
    """
    
    # Send the request to the Overpass API
    response = requests.get(overpass_url, params={'data': overpass_query})
    data = response.json()
    
    # Process the response and create GeoJSON features
    features = []
    for element in data['elements']:
        if element['type'] in ['way', 'relation']:
            coords = []
            if element['type'] == 'way':
                coords = [(node['lon'], node['lat']) for node in element['geometry']]
            elif element['type'] == 'relation':
                # For simplicity, we'll just use the outer way of the relation
                outer = next((member for member in element['members'] if member['role'] == 'outer'), None)
                if outer:
                    coords = [(node['lon'], node['lat']) for node in outer['geometry']]
            
            # Check if we have at least 4 coordinates
            if len(coords) >= 4:
                properties = element.get('tags', {})
                height = properties.get('height', properties.get('building:height', '0'))  # Default to 3 meters if no height is specified
                try:
                    height = float(height)
                except ValueError:
                    # print("No building height data was found. A height of 10 meters was set instead.")
                    height = 0  # Default height if conversion fails
                
                feature = {
                    "type": "Feature",
                    "properties": {
                        "height": height,
                        "confidence": -1.0  # Set confidence to -1.0 as we don't have this information from OSM
                    },
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[coord[::-1] for coord in coords]]  # Reverse lat and lon
                    }
                }
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
            # Assuming average height per level is 3 meters
            height = float(levels) * 3.0
        else:
            # Default height if not specified
            height = 10.0  # You can adjust this default value as needed

    new_feature['properties']['height'] = float(height)
    new_feature['properties']['confidence'] = -1.0  # As per your desirable format

    return new_feature