"""
This module provides functions for drawing and manipulating rectangles on maps.
"""

import math
from pyproj import Proj, transform
from ipyleaflet import Map, DrawControl, Rectangle, Polygon as LeafletPolygon
import ipyleaflet
from geopy import distance
from .utils import get_coordinates_from_cityname
import shapely.geometry as geom

def rotate_rectangle(m, rectangle_vertices, angle):
    """
    Project rectangle to Mercator, rotate, and re-project to lat-lon.

    Args:
        m (ipyleaflet.Map): Map object to draw the rotated rectangle on
        rectangle_vertices (list): List of (lat, lon) tuples defining the rectangle vertices
        angle (float): Rotation angle in degrees

    Returns:
        list: List of rotated (lat, lon) tuples defining the new rectangle vertices
    """
    if not rectangle_vertices:
        print("Draw a rectangle first!")
        return

    # Define projections - need to convert between coordinate systems for accurate rotation
    wgs84 = Proj(init='epsg:4326')  # WGS84 lat-lon (standard GPS coordinates)
    mercator = Proj(init='epsg:3857')  # Web Mercator (projection used by most web maps)

    # Project vertices from WGS84 to Web Mercator for proper distance calculations
    projected_vertices = [transform(wgs84, mercator, lon, lat) for lat, lon in rectangle_vertices]

    # Calculate the centroid to use as rotation center
    centroid_x = sum(x for x, y in projected_vertices) / len(projected_vertices)
    centroid_y = sum(y for x, y in projected_vertices) / len(projected_vertices)

    # Convert angle to radians (negative for clockwise rotation)
    angle_rad = -math.radians(angle)

    # Rotate each vertex around the centroid using standard 2D rotation matrix
    rotated_vertices = []
    for x, y in projected_vertices:
        # Translate point to origin for rotation
        temp_x = x - centroid_x
        temp_y = y - centroid_y

        # Apply rotation matrix
        rotated_x = temp_x * math.cos(angle_rad) - temp_y * math.sin(angle_rad)
        rotated_y = temp_x * math.sin(angle_rad) + temp_y * math.cos(angle_rad)

        # Translate point back to original position
        new_x = rotated_x + centroid_x
        new_y = rotated_y + centroid_y

        rotated_vertices.append((new_x, new_y))

    # Convert coordinates back to WGS84 (lat/lon)
    new_vertices = [transform(mercator, wgs84, x, y) for x, y in rotated_vertices]

    # Reorder coordinates from (lon,lat) to (lat,lon) format
    new_vertices = [(lat, lon) for lon, lat in new_vertices]

    # Create and add new polygon layer to map
    polygon = ipyleaflet.Polygon(
        locations=new_vertices,
        color="red",
        fill_color="red"
    )
    m.add_layer(polygon)

    return new_vertices

def draw_rectangle_map(center=(40, -100), zoom=4):
    """
    Create an interactive map for drawing rectangles.

    Args:
        center (tuple): Center coordinates (lat, lon) for the map view. Defaults to (40, -100).
        zoom (int): Initial zoom level for the map. Defaults to 4.

    Returns:
        tuple: (Map object, list of rectangle vertices)
            - Map object for displaying and interacting with the map
            - Empty list that will be populated with rectangle vertices when drawn
    """
    # Initialize the map centered at specified coordinates
    m = Map(center=center, zoom=zoom)

    # List to store the vertices of drawn rectangle
    rectangle_vertices = []

    def handle_draw(target, action, geo_json):
        """Handle draw events on the map."""
        # Clear any previously stored vertices
        rectangle_vertices.clear()

        # Process only if a rectangle polygon was drawn
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            # Extract coordinates from GeoJSON format
            coordinates = geo_json['geometry']['coordinates'][0]
            print("Vertices of the drawn rectangle:")
            # Store all vertices except last (GeoJSON repeats first vertex at end)
            for coord in coordinates[:-1]:
                # Convert from GeoJSON (lon,lat) to standard (lat,lon) format
                rectangle_vertices.append((coord[1], coord[0]))
                print(f"Longitude: {coord[0]}, Latitude: {coord[1]}")

    # Configure drawing controls - only enable rectangle drawing
    draw_control = DrawControl()
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circle = {}
    draw_control.rectangle = {
        "shapeOptions": {
            "color": "#6bc2e5",
            "weight": 4,
        }
    }
    m.add_control(draw_control)

    # Register event handler for drawing actions
    draw_control.on_draw(handle_draw)

    return m, rectangle_vertices

def draw_rectangle_map_cityname(cityname, zoom=15):
    """
    Create an interactive map centered on a specified city for drawing rectangles.

    Args:
        cityname (str): Name of the city to center the map on
        zoom (int): Initial zoom level for the map. Defaults to 15.

    Returns:
        tuple: (Map object, list of rectangle vertices)
            - Map object centered on the specified city
            - Empty list that will be populated with rectangle vertices when drawn
    """
    # Get coordinates for the specified city
    center = get_coordinates_from_cityname(cityname)
    m, rectangle_vertices = draw_rectangle_map(center=center, zoom=zoom)
    return m, rectangle_vertices

def center_location_map_cityname(cityname, east_west_length, north_south_length, zoom=15):
    """
    Create an interactive map centered on a city where clicking creates a rectangle of specified dimensions.

    Args:
        cityname (str): Name of the city to center the map on
        east_west_length (float): Width of the rectangle in meters
        north_south_length (float): Height of the rectangle in meters
        zoom (int): Initial zoom level for the map. Defaults to 15.

    Returns:
        tuple: (Map object, list of rectangle vertices)
            - Map object centered on the specified city
            - Empty list that will be populated with rectangle vertices when a point is clicked
    """
    
    # Get coordinates for the specified city
    center = get_coordinates_from_cityname(cityname)
    
    # Initialize map centered on the city
    m = Map(center=center, zoom=zoom)

    # List to store rectangle vertices
    rectangle_vertices = []

    def handle_draw(target, action, geo_json):
        """Handle draw events on the map."""
        # Clear previous vertices and remove any existing rectangles
        rectangle_vertices.clear()
        for layer in m.layers:
            if isinstance(layer, Rectangle):
                m.remove_layer(layer)

        # Process only if a point was drawn on the map
        if action == 'created' and geo_json['geometry']['type'] == 'Point':
            # Extract point coordinates (converting from GeoJSON lon,lat to lat,lon)
            lat, lon = geo_json['geometry']['coordinates'][1], geo_json['geometry']['coordinates'][0]
            print(f"Point drawn at Latitude: {lat}, Longitude: {lon}")
            
            # Calculate corner points using geopy's distance calculator
            # Each point is calculated as a destination from center point using bearing
            north = distance.distance(meters=north_south_length/2).destination((lat, lon), bearing=0)
            south = distance.distance(meters=north_south_length/2).destination((lat, lon), bearing=180)
            east = distance.distance(meters=east_west_length/2).destination((lat, lon), bearing=90)
            west = distance.distance(meters=east_west_length/2).destination((lat, lon), bearing=270)

            # Create rectangle vertices in counter-clockwise order
            rectangle_vertices.extend([
                (south.latitude, west.longitude),
                (north.latitude, west.longitude),
                (north.latitude, east.longitude),
                (south.latitude, east.longitude)                
            ])

            # Create and add new rectangle to map
            rectangle = Rectangle(
                bounds=[(north.latitude, west.longitude), (south.latitude, east.longitude)],
                color="red",
                fill_color="red",
                fill_opacity=0.2
            )
            m.add_layer(rectangle)

            print("Rectangle vertices:")
            for vertex in rectangle_vertices:
                print(f"Latitude: {vertex[0]}, Longitude: {vertex[1]}")

    # Configure drawing controls - only enable point drawing
    draw_control = DrawControl()
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circle = {}
    draw_control.rectangle = {}
    draw_control.marker = {}
    m.add_control(draw_control)

    # Register event handler for drawing actions
    draw_control.on_draw(handle_draw)

    return m, rectangle_vertices

def display_buildings_and_draw_polygon(building_geojson, zoom=17):
    """
    Displays building footprints (in Lat-Lon order) on an ipyleaflet map,
    and allows the user to draw a polygon whose vertices are returned
    in a Python list (also in Lat-Lon).

    Args:
        building_geojson (list): A list of GeoJSON features (Polygons),
                                 BUT with coordinates in [lat, lon] order.
        zoom (int): Initial zoom level for the map. Default=17.

    Returns:
        (map_object, drawn_polygon_vertices)
          - map_object: ipyleaflet Map instance
          - drawn_polygon_vertices: a Python list that gets updated whenever
            a new polygon is created. The list is in (lat, lon) order.
    """
    # ---------------------------------------------------------
    # 1. Determine a suitable map center via bounding box logic
    # ---------------------------------------------------------
    all_lats = []
    all_lons = []
    for feature in building_geojson:
        # Handle only Polygons here; skip MultiPolygon if present
        if feature['geometry']['type'] == 'Polygon':
            # Coordinates in this data are [ [lat, lon], [lat, lon], ... ]
            coords = feature['geometry']['coordinates'][0]  # outer ring
            all_lats.extend(pt[0] for pt in coords)
            all_lons.extend(pt[1] for pt in coords)

    if not all_lats or not all_lons:
        # Fallback: If no footprints or invalid data, pick a default
        center_lat, center_lon = 40.0, -100.0
    else:
        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lon, max_lon = min(all_lons), max(all_lons)
        center_lat = (min_lat + max_lat) / 2
        center_lon = (min_lon + max_lon) / 2

    # Create the ipyleaflet map
    m = Map(center=(center_lat, center_lon), zoom=zoom, scroll_wheel_zoom=True)

    # -----------------------------------------
    # 2. Add each building footprint to the map
    # -----------------------------------------
    for feature in building_geojson:
        # Only handle simple Polygons
        if feature['geometry']['type'] == 'Polygon':
            coords = feature['geometry']['coordinates'][0]
            # Because your data is already lat-lon, we do NOT swap:
            lat_lon_coords = [(c[0], c[1]) for c in coords]

            # Create the polygon layer
            bldg_layer = LeafletPolygon(
                locations=lat_lon_coords,
                color="blue",
                fill_color="blue",
                fill_opacity=0.2,
                weight=2
            )
            m.add_layer(bldg_layer)

    # -----------------------------------------------------------------
    # 3. Enable drawing of polygons, capturing the vertices in Lat-Lon
    # -----------------------------------------------------------------
    drawn_polygon_vertices = []  # We'll store the newly drawn polygon's vertices here (lat, lon).

    draw_control = DrawControl(
        polygon={
            "shapeOptions": {
                "color": "#6bc2e5",
                "fillColor": "#6bc2e5",
                "fillOpacity": 0.2
            }
        },
        rectangle={},     # Disable rectangles (or enable if needed)
        circle={},        # Disable circles
        circlemarker={},  # Disable circlemarkers
        polyline={},      # Disable polylines
        marker={}         # Disable markers
    )

    def handle_draw(self, action, geo_json):
        """
        Callback for whenever a shape is created or edited.
        ipyleaflet's DrawControl returns standard GeoJSON (lon, lat).
        We'll convert them to (lat, lon).
        """
        # Clear any previously stored vertices
        drawn_polygon_vertices.clear()

        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            # The polygon’s first ring
            coordinates = geo_json['geometry']['coordinates'][0]
            print("Vertices of the drawn polygon (Lat-Lon):")

            # By default, drawn polygon coords are [ [lon, lat], [lon, lat], ... ]
            # The last coordinate repeats the first -> skip it with [:-1]
            for coord in coordinates[:-1]:
                lon = coord[0]
                lat = coord[1]
                drawn_polygon_vertices.append((lat, lon))
                print(f" - (lat, lon) = ({lat}, {lon})")

    draw_control.on_draw(handle_draw)
    m.add_control(draw_control)

    return m, drawn_polygon_vertices