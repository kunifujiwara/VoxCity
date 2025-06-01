"""
This module provides functions for drawing and manipulating rectangles and polygons on interactive maps.
It serves as a core component for defining geographical regions of interest in the VoxCity library.

Key Features:
    - Interactive rectangle drawing on maps using ipyleaflet
    - Rectangle rotation with coordinate system transformations
    - City-centered map initialization
    - Fixed-dimension rectangle creation from center points
    - Building footprint visualization and polygon drawing
    - Support for both WGS84 and Web Mercator projections
    - Coordinate format handling between (lon,lat) and (lat,lon)

The module maintains consistent coordinate order conventions:
    - Internal storage: (lon,lat) format to match GeoJSON standard
    - ipyleaflet interface: (lat,lon) format as required by the library
    - All return values: (lon,lat) format for consistency

Dependencies:
    - ipyleaflet: For interactive map display and drawing controls
    - pyproj: For coordinate system transformations
    - geopy: For distance calculations
    - shapely: For geometric operations
"""

import math
from pyproj import Proj, transform
from ipyleaflet import Map, DrawControl, Rectangle, Polygon as LeafletPolygon
import ipyleaflet
from geopy import distance
import shapely.geometry as geom

from .utils import get_coordinates_from_cityname

def rotate_rectangle(m, rectangle_vertices, angle):
    """
    Project rectangle to Mercator, rotate, and re-project to lat-lon coordinates.
    
    This function performs a rotation of a rectangle in geographic space by:
    1. Converting coordinates from WGS84 (lat/lon) to Web Mercator projection
    2. Performing the rotation in the projected space for accurate distance preservation
    3. Converting back to WGS84 coordinates
    4. Visualizing the result on the provided map
    
    The rotation is performed around the rectangle's centroid using a standard 2D rotation matrix.
    The function handles coordinate system transformations to ensure geometrically accurate rotations
    despite the distortions inherent in geographic projections.

    Args:
        m (ipyleaflet.Map): Map object to draw the rotated rectangle on.
            The map must be initialized and have a valid center and zoom level.
        rectangle_vertices (list): List of (lon, lat) tuples defining the rectangle vertices.
            The vertices should be ordered in a counter-clockwise direction.
            Example: [(lon1,lat1), (lon2,lat2), (lon3,lat3), (lon4,lat4)]
        angle (float): Rotation angle in degrees.
            Positive angles rotate counter-clockwise.
            Negative angles rotate clockwise.

    Returns:
        list: List of rotated (lon, lat) tuples defining the new rectangle vertices.
            The vertices maintain their original ordering.
            Returns None if no rectangle vertices are provided.

    Note:
        The function uses EPSG:4326 (WGS84) for geographic coordinates and
        EPSG:3857 (Web Mercator) for the rotation calculations.
    """
    if not rectangle_vertices:
        print("Draw a rectangle first!")
        return

    # Define projections - need to convert between coordinate systems for accurate rotation
    wgs84 = Proj(init='epsg:4326')  # WGS84 lat-lon (standard GPS coordinates)
    mercator = Proj(init='epsg:3857')  # Web Mercator (projection used by most web maps)

    # Project vertices from WGS84 to Web Mercator for proper distance calculations
    projected_vertices = [transform(wgs84, mercator, lon, lat) for lon, lat in rectangle_vertices]

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

    # Convert coordinates back to WGS84 (lon/lat)
    new_vertices = [transform(mercator, wgs84, x, y) for x, y in rotated_vertices]

    # Create and add new polygon layer to map
    polygon = ipyleaflet.Polygon(
        locations=[(lat, lon) for lon, lat in new_vertices],  # Convert to (lat,lon) for ipyleaflet
        color="red",
        fill_color="red"
    )
    m.add_layer(polygon)

    return new_vertices

def draw_rectangle_map(center=(40, -100), zoom=4):
    """
    Create an interactive map for drawing rectangles with ipyleaflet.
    
    This function initializes an interactive map that allows users to draw rectangles
    by clicking and dragging on the map surface. The drawn rectangles are captured
    and their vertices are stored in geographic coordinates.

    The map interface provides:
    - A rectangle drawing tool activated by default
    - Real-time coordinate capture of drawn shapes
    - Automatic vertex ordering in counter-clockwise direction
    - Console output of vertex coordinates for verification
    
    Drawing Controls:
    - Click and drag to draw a rectangle
    - Release to complete the rectangle
    - Only one rectangle can be active at a time
    - Drawing a new rectangle clears the previous one

    Args:
        center (tuple): Center coordinates (lat, lon) for the map view.
            Defaults to (40, -100) which centers on the continental United States.
            Format: (latitude, longitude) in decimal degrees.
        zoom (int): Initial zoom level for the map. Defaults to 4.
            Range: 0 (most zoomed out) to 18 (most zoomed in).
            Recommended: 3-6 for countries, 10-15 for cities.

    Returns:
        tuple: (Map object, list of rectangle vertices)
            - Map object: ipyleaflet.Map instance for displaying and interacting with the map
            - rectangle_vertices: Empty list that will be populated with (lon,lat) tuples
              when a rectangle is drawn. Coordinates are stored in GeoJSON order (lon,lat).

    Note:
        The function disables all drawing tools except rectangles to ensure
        consistent shape creation. The rectangle vertices are automatically
        converted to (lon,lat) format when stored, regardless of the input
        center coordinate order.
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
                # Keep GeoJSON (lon,lat) format
                rectangle_vertices.append((coord[0], coord[1]))
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
    
    This function extends draw_rectangle_map() by automatically centering the map
    on a specified city using geocoding. It provides a convenient way to focus
    the drawing interface on a particular urban area without needing to know
    its exact coordinates.

    The function uses the utils.get_coordinates_from_cityname() function to
    geocode the city name and obtain its coordinates. The resulting map is
    zoomed to an appropriate level for urban-scale analysis.

    Args:
        cityname (str): Name of the city to center the map on.
            Can include country or state for better accuracy.
            Examples: "Tokyo, Japan", "New York, NY", "Paris, France"
        zoom (int): Initial zoom level for the map. Defaults to 15.
            Range: 0 (most zoomed out) to 18 (most zoomed in).
            Default of 15 is optimized for city-level visualization.

    Returns:
        tuple: (Map object, list of rectangle vertices)
            - Map object: ipyleaflet.Map instance centered on the specified city
            - rectangle_vertices: Empty list that will be populated with (lon,lat)
              tuples when a rectangle is drawn

    Note:
        If the city name cannot be geocoded, the function will raise an error.
        For better results, provide specific city names with country/state context.
        The function inherits all drawing controls and behavior from draw_rectangle_map().
    """
    # Get coordinates for the specified city
    center = get_coordinates_from_cityname(cityname)
    m, rectangle_vertices = draw_rectangle_map(center=center, zoom=zoom)
    return m, rectangle_vertices

def center_location_map_cityname(cityname, east_west_length, north_south_length, zoom=15):
    """
    Create an interactive map centered on a city where clicking creates a rectangle of specified dimensions.
    
    This function provides a specialized interface for creating fixed-size rectangles
    centered on user-selected points. Instead of drawing rectangles by dragging,
    users click a point on the map and a rectangle of the specified dimensions
    is automatically created centered on that point.

    The function handles:
    - Automatic city geocoding and map centering
    - Distance calculations in meters using geopy
    - Conversion between geographic and metric distances
    - Rectangle creation with specified dimensions
    - Visualization of created rectangles

    Workflow:
    1. Map is centered on the specified city
    2. User clicks a point on the map
    3. A rectangle is created centered on that point
    4. Rectangle dimensions are maintained in meters regardless of latitude
    5. Previous rectangles are automatically cleared

    Args:
        cityname (str): Name of the city to center the map on.
            Can include country or state for better accuracy.
            Examples: "Tokyo, Japan", "New York, NY"
        east_west_length (float): Width of the rectangle in meters.
            This is the dimension along the east-west direction.
            The actual ground distance is maintained regardless of projection distortion.
        north_south_length (float): Height of the rectangle in meters.
            This is the dimension along the north-south direction.
            The actual ground distance is maintained regardless of projection distortion.
        zoom (int): Initial zoom level for the map. Defaults to 15.
            Range: 0 (most zoomed out) to 18 (most zoomed in).
            Default of 15 is optimized for city-level visualization.

    Returns:
        tuple: (Map object, list of rectangle vertices)
            - Map object: ipyleaflet.Map instance centered on the specified city
            - rectangle_vertices: Empty list that will be populated with (lon,lat)
              tuples when a point is clicked and the rectangle is created

    Note:
        - Rectangle dimensions are specified in meters but stored as geographic coordinates
        - The function uses geopy's distance calculations for accurate metric distances
        - Only one rectangle can exist at a time; clicking a new point removes the previous rectangle
        - Rectangle vertices are returned in GeoJSON (lon,lat) order
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
            # Extract point coordinates from GeoJSON (lon,lat)
            lon, lat = geo_json['geometry']['coordinates'][0], geo_json['geometry']['coordinates'][1]
            print(f"Point drawn at Longitude: {lon}, Latitude: {lat}")
            
            # Calculate corner points using geopy's distance calculator
            # Each point is calculated as a destination from center point using bearing
            north = distance.distance(meters=north_south_length/2).destination((lat, lon), bearing=0)
            south = distance.distance(meters=north_south_length/2).destination((lat, lon), bearing=180)
            east = distance.distance(meters=east_west_length/2).destination((lat, lon), bearing=90)
            west = distance.distance(meters=east_west_length/2).destination((lat, lon), bearing=270)

            # Create rectangle vertices in counter-clockwise order (lon,lat)
            rectangle_vertices.extend([
                (west.longitude, south.latitude),
                (west.longitude, north.latitude),
                (east.longitude, north.latitude),
                (east.longitude, south.latitude)                
            ])

            # Create and add new rectangle to map (ipyleaflet expects lat,lon)
            rectangle = Rectangle(
                bounds=[(north.latitude, west.longitude), (south.latitude, east.longitude)],
                color="red",
                fill_color="red",
                fill_opacity=0.2
            )
            m.add_layer(rectangle)

            print("Rectangle vertices:")
            for vertex in rectangle_vertices:
                print(f"Longitude: {vertex[0]}, Latitude: {vertex[1]}")

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

def display_buildings_and_draw_polygon(building_gdf=None, rectangle_vertices=None, zoom=17):
    """
    Displays building footprints and enables polygon drawing on an interactive map.
    
    This function creates an interactive map that visualizes building footprints and
    allows users to draw arbitrary polygons. It's particularly useful for selecting
    specific buildings or areas within an urban context.

    The function provides three key features:
    1. Building Footprint Visualization:
       - Displays building polygons from a GeoDataFrame
       - Uses consistent styling for all buildings
       - Handles simple polygon geometries only
    
    2. Interactive Polygon Drawing:
       - Enables free-form polygon drawing
       - Captures vertices in consistent (lon,lat) format
       - Maintains GeoJSON compatibility
    
    3. Map Initialization:
       - Automatic centering based on input data
       - Fallback to default location if no data provided
       - Support for both building data and rectangle bounds

    Args:
        building_gdf (GeoDataFrame, optional): A GeoDataFrame containing building footprints.
            Must have geometry column with Polygon type features.
            Geometries should be in [lon, lat] coordinate order.
            If None, only the base map is displayed.
        rectangle_vertices (list, optional): List of [lon, lat] coordinates defining rectangle corners.
            Used to set the initial map view extent.
            Takes precedence over building_gdf for determining map center.
        zoom (int): Initial zoom level for the map. Default=17.
            Range: 0 (most zoomed out) to 18 (most zoomed in).
            Default of 17 is optimized for building-level detail.

    Returns:
        tuple: (map_object, drawn_polygon_vertices)
            - map_object: ipyleaflet Map instance with building footprints and drawing controls
            - drawn_polygon_vertices: List that gets updated with (lon,lat) coordinates
              whenever a new polygon is drawn. Coordinates are in GeoJSON order.

    Note:
        - Building footprints are displayed in blue with 20% opacity
        - Only simple Polygon geometries are supported (no MultiPolygons)
        - Drawing tools are restricted to polygon creation only
        - All coordinates are handled in (lon,lat) order internally
        - The function automatically determines appropriate map bounds
    """
    # ---------------------------------------------------------
    # 1. Determine a suitable map center via bounding box logic
    # ---------------------------------------------------------
    if rectangle_vertices is not None:
        # Get bounds from rectangle vertices
        lons = [v[0] for v in rectangle_vertices]
        lats = [v[1] for v in rectangle_vertices]
        min_lon, max_lon = min(lons), max(lons)
        min_lat, max_lat = min(lats), max(lats)
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
    elif building_gdf is not None and len(building_gdf) > 0:
        # Get bounds from GeoDataFrame
        bounds = building_gdf.total_bounds  # Returns [minx, miny, maxx, maxy]
        min_lon, min_lat, max_lon, max_lat = bounds
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
    else:
        # Fallback: If no inputs or invalid data, pick a default
        center_lon, center_lat = -100.0, 40.0

    # Create the ipyleaflet map (needs lat,lon)
    m = Map(center=(center_lat, center_lon), zoom=zoom, scroll_wheel_zoom=True)

    # -----------------------------------------
    # 2. Add building footprints to the map if provided
    # -----------------------------------------
    if building_gdf is not None:
        for idx, row in building_gdf.iterrows():
            # Only handle simple Polygons
            if isinstance(row.geometry, geom.Polygon):
                # Get coordinates from geometry
                coords = list(row.geometry.exterior.coords)
                # Convert to (lat,lon) for ipyleaflet, skip last repeated coordinate
                lat_lon_coords = [(c[1], c[0]) for c in coords[:-1]]

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
    # 3. Enable drawing of polygons, capturing the vertices in Lon-Lat
    # -----------------------------------------------------------------
    drawn_polygon_vertices = []  # We'll store the newly drawn polygon's vertices here (lon, lat).

    draw_control = DrawControl(
        polygon={
            "shapeOptions": {
                "color": "red",
                "fillColor": "red",
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
        We'll keep them as (lon, lat).
        """
        # Clear any previously stored vertices
        drawn_polygon_vertices.clear()

        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            # The polygon's first ring
            coordinates = geo_json['geometry']['coordinates'][0]
            print("Vertices of the drawn polygon (Lon-Lat):")

            # Keep GeoJSON (lon,lat) format, skip last repeated coordinate
            for coord in coordinates[:-1]:
                lon = coord[0]
                lat = coord[1]
                drawn_polygon_vertices.append((lon, lat))
                print(f" - (lon, lat) = ({lon}, {lat})")

    draw_control.on_draw(handle_draw)
    m.add_control(draw_control)

    return m, drawn_polygon_vertices