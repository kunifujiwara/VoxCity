import mercantile
import requests
import mapbox_vector_tile
from shapely.geometry import shape, mapping
from shapely.affinity import affine_transform
import shapely.ops
import json
from pyproj import Transformer
import json

def load_geojsons_from_openmaptiles(rectangle_vertices, API_KEY):

    # Extract latitudes and longitudes
    lats = [coord[0] for coord in rectangle_vertices]
    lons = [coord[1] for coord in rectangle_vertices]

    # Find minimum and maximum values
    min_lat = min(lats)
    max_lat = max(lats)
    min_lon = min(lons)
    max_lon = max(lons)

    # Define the zoom level
    zoom = 15  # Adjust as needed

    # Generate a list of tiles covering the bounding box
    tiles = list(mercantile.tiles(min_lon, min_lat, max_lon, max_lat, zoom))

    # Initialize a list to store building features
    building_features = []

    # Set up the transformer from Web Mercator to WGS84
    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    for tile in tiles:
        x, y, z = tile.x, tile.y, tile.z

        # Construct the tile URL
        tile_url = f'https://api.maptiler.com/tiles/v3/{z}/{x}/{y}.pbf?key={API_KEY}'

        print(f'Downloading tile {z}/{x}/{y}')
        response = requests.get(tile_url)

        if response.status_code != 200:
            print(f'Failed to download tile {z}/{x}/{y}')
            continue

        # Decode the vector tile
        tile_data = mapbox_vector_tile.decode(response.content)

        # Check if the 'building' layer exists
        if 'building' in tile_data:
            building_layer = tile_data['building']
            for feature in building_layer['features']:
                geometry = shape(feature['geometry'])

                # Tile coordinate extent
                x_min, y_min = 0, 0
                x_max, y_max = 4096, 4096

                # Get the tile's bounding box in Web Mercator
                tile_bbox_mercator = mercantile.xy_bounds(x, y, z)

                # Calculate scale factors
                scale_x = (tile_bbox_mercator.right - tile_bbox_mercator.left) / (x_max - x_min)
                scale_y = (tile_bbox_mercator.bottom - tile_bbox_mercator.top) / (y_max - y_min)

                # Affine transformation parameters
                a = scale_x
                b = 0
                d = 0
                e = -scale_y
                xoff = tile_bbox_mercator.left
                yoff = tile_bbox_mercator.bottom

                transform_matrix = [a, b, d, e, xoff, yoff]

                # Apply affine transformation to get Web Mercator coordinates
                transformed_geom = affine_transform(geometry, transform_matrix)

                # Transform from Web Mercator to geographic coordinates
                transformed_geometry = shapely.ops.transform(transformer.transform, transformed_geom)

                # Create GeoJSON feature
                geojson_feature = {
                    'type': 'Feature',
                    'geometry': mapping(transformed_geometry),
                    'properties': feature['properties']
                }

                building_features.append(geojson_feature)

    # Create a FeatureCollection and save to GeoJSON file
    geojson_data = {
        'type': 'FeatureCollection',
        'features': building_features
    }

    # with open('buildings.geojson', 'w') as f:
    #     json.dump(geojson_data, f)

    # print('Saved building data to buildings.geojson')

    converted_geojson_data = convert_geojson_format(geojson_data)

    return converted_geojson_data

def convert_geojson_format(geojson_data):
    # Assuming 'current_data' is your current GeoJSON data loaded as a dictionary.
    current_data = geojson_data

    # Initialize a list to store the new features in the desirable format
    new_features = []

    for feature in current_data['features']:
        geometry = feature['geometry']
        properties = feature['properties']

        if geometry['type'] == 'MultiPolygon':
            # Iterate over each Polygon in the MultiPolygon
            for polygon_coords in geometry['coordinates']:
                # Create a new feature with geometry type 'Polygon'
                new_geometry = {
                    'type': 'Polygon',
                    'coordinates': polygon_coords
                }
                # Copy and adjust properties if needed
                new_properties = properties.copy()
                # Optionally, map 'render_height' to 'height' and add 'confidence'
                new_properties['height'] = new_properties.pop('render_height', None)
                new_properties['confidence'] = -1.0  # Or any other default value

                # Adjust coordinate order from (lon, lat) to (lat, lon) if necessary
                new_coords = []
                for ring in polygon_coords:
                    new_ring = [(coord[1], coord[0]) for coord in ring]
                    new_coords.append(new_ring)
                new_geometry['coordinates'] = new_coords

                # Create the new feature
                new_feature = {
                    'type': 'Feature',
                    'properties': new_properties,
                    'geometry': new_geometry
                }
                new_features.append(new_feature)
        elif geometry['type'] == 'Polygon':
            # Adjust coordinate order from (lon, lat) to (lat, lon) if necessary
            new_coords = []
            for ring in geometry['coordinates']:
                new_ring = [(coord[1], coord[0]) for coord in ring]
                new_coords.append(new_ring)
            geometry['coordinates'] = new_coords

            # Copy and adjust properties if needed
            new_properties = properties.copy()
            # Optionally, map 'render_height' to 'height' and add 'confidence'
            new_properties['height'] = new_properties.pop('render_height', None)
            new_properties['confidence'] = -1.0  # Or any other default value

            # Create the new feature
            new_feature = {
                'type': 'Feature',
                'properties': new_properties,
                'geometry': geometry
            }
            new_features.append(new_feature)
        else:
            # Handle other geometry types if necessary
            pass

    # Now 'new_features' contains the data in the desirable format
    # Let's output it to a JSON file or print it

    # # Optionally, write the output to a file
    # with open('desired_format.json', 'w') as f:
    #     json.dump(new_features, f, indent=2)
    
    return new_features

