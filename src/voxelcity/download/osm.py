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
                height = properties.get('height', properties.get('building:height', '3'))  # Default to 3 meters if no height is specified
                try:
                    height = float(height)
                except ValueError:
                    height = 3  # Default height if conversion fails
                
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