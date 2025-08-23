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
from ipyleaflet import (
    Map, 
    DrawControl, 
    Rectangle, 
    Polygon as LeafletPolygon,
    WidgetControl,
    Circle,
    basemaps,
    basemap_to_tiles,
    TileLayer
)
from geopy import distance
import shapely.geometry as geom
import geopandas as gpd
from ipywidgets import VBox, HBox, Button, FloatText, Label, Output, HTML, Checkbox
import pandas as pd
from IPython.display import display, clear_output

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
    polygon = LeafletPolygon(
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
       - Supports multiple polygons with unique IDs and colors
    
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
        tuple: (map_object, drawn_polygons)
            - map_object: ipyleaflet Map instance with building footprints and drawing controls
            - drawn_polygons: List of dictionaries with 'id', 'vertices', and 'color' keys for all drawn polygons.
              Each polygon has a unique ID and color for easy identification.

    Note:
        - Building footprints are displayed in blue with 20% opacity
        - Only simple Polygon geometries are supported (no MultiPolygons)
        - Drawing tools are restricted to polygon creation only
        - All coordinates are handled in (lon,lat) order internally
        - The function automatically determines appropriate map bounds
        - Each polygon gets a unique ID and different colors for easy identification
        - Use get_polygon_vertices() helper function to extract specific polygon data
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
    # Store multiple polygons with IDs and colors
    drawn_polygons = []  # List of dicts with 'id', 'vertices', 'color' keys
    polygon_counter = 0
    polygon_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

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
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            nonlocal polygon_counter
            polygon_counter += 1
            
            # The polygon's first ring
            coordinates = geo_json['geometry']['coordinates'][0]
            vertices = [(coord[0], coord[1]) for coord in coordinates[:-1]]
            
            # Assign color (cycle through colors)
            color = polygon_colors[polygon_counter % len(polygon_colors)]
            
            # Store polygon data
            polygon_data = {
                'id': polygon_counter,
                'vertices': vertices,
                'color': color
            }
            drawn_polygons.append(polygon_data)
            
            print(f"Polygon {polygon_counter} drawn with {len(vertices)} vertices (color: {color}):")
            for i, (lon, lat) in enumerate(vertices):
                print(f"  Vertex {i+1}: (lon, lat) = ({lon}, {lat})")
            print(f"Total polygons: {len(drawn_polygons)}")

    draw_control.on_draw(handle_draw)
    m.add_control(draw_control)

    return m, drawn_polygons

def draw_additional_buildings(building_gdf=None, initial_center=None, zoom=17, rectangle_vertices=None):
    """
    Creates an interactive map for drawing building footprints with height input.
    
    This function provides an interface for users to:
    1. Draw building footprints on an interactive map
    2. Set building height values through a UI widget
    3. Add new buildings to the existing building_gdf
    
    The workflow is:
    - User draws a polygon on the map
    - Height input widget appears
    - User enters height and clicks "Add Building"
    - Building is added to GeoDataFrame and displayed on map
    
    Args:
        building_gdf (GeoDataFrame, optional): Existing building footprints to display.
            If None, creates a new empty GeoDataFrame.
            Expected columns: ['id', 'height', 'min_height', 'geometry', 'building_id']
            - 'id': Integer ID from data sources (e.g., OSM building id)
            - 'height': Building height in meters (set by user input)
            - 'min_height': Minimum height in meters (defaults to 0.0)
            - 'geometry': Building footprint polygon
            - 'building_id': Unique building identifier
        initial_center (tuple, optional): Initial map center as (lon, lat).
            If None, centers on existing buildings or defaults to (-100, 40).
        zoom (int): Initial zoom level (default=17).
    
    Returns:
        tuple: (map_object, updated_building_gdf)
            - map_object: ipyleaflet Map instance with drawing controls
            - updated_building_gdf: GeoDataFrame that automatically updates when buildings are added
    
    Example:
        >>> # Start with empty buildings
        >>> m, buildings = draw_additional_buildings()
        >>> # Draw buildings on the map...
        >>> print(buildings)  # Will contain all drawn buildings
    """
    
    # Initialize or copy the building GeoDataFrame
    if building_gdf is None:
        # Create empty GeoDataFrame with required columns
        updated_gdf = gpd.GeoDataFrame(
            columns=['id', 'height', 'min_height', 'geometry', 'building_id'],
            crs='EPSG:4326'
        )
    else:
        # Make a copy to avoid modifying the original
        updated_gdf = building_gdf.copy()
        # Ensure all required columns exist
        if 'height' not in updated_gdf.columns:
            updated_gdf['height'] = 10.0  # Default height
        if 'min_height' not in updated_gdf.columns:
            updated_gdf['min_height'] = 0.0  # Default min_height
        if 'building_id' not in updated_gdf.columns:
            updated_gdf['building_id'] = range(len(updated_gdf))
        if 'id' not in updated_gdf.columns:
            updated_gdf['id'] = range(len(updated_gdf))
    
    # Determine map center
    if initial_center is not None:
        center_lon, center_lat = initial_center
    elif updated_gdf is not None and len(updated_gdf) > 0:
        bounds = updated_gdf.total_bounds
        min_lon, min_lat, max_lon, max_lat = bounds
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
    elif rectangle_vertices is not None:
        center_lon, center_lat = (rectangle_vertices[0][0] + rectangle_vertices[2][0]) / 2, (rectangle_vertices[0][1] + rectangle_vertices[2][1]) / 2
    else:
        center_lon, center_lat = -100.0, 40.0
    
    # Create the map
    m = Map(center=(center_lat, center_lon), zoom=zoom, scroll_wheel_zoom=True)
    
    # Display existing buildings
    building_layers = {}
    for idx, row in updated_gdf.iterrows():
        if isinstance(row.geometry, geom.Polygon):
            coords = list(row.geometry.exterior.coords)
            lat_lon_coords = [(c[1], c[0]) for c in coords[:-1]]
            
            height = row.get('height', 10.0)
            min_height = row.get('min_height', 0.0)
            building_id = row.get('building_id', idx)
            bldg_id = row.get('id', idx)
            bldg_layer = LeafletPolygon(
                locations=lat_lon_coords,
                color="blue",
                fill_color="blue",
                fill_opacity=0.3,
                weight=2,
                popup=HTML(f"<b>Building ID:</b> {building_id}<br>"
                          f"<b>ID:</b> {bldg_id}<br>"
                          f"<b>Height:</b> {height}m<br>"
                          f"<b>Min Height:</b> {min_height}m")
            )
            m.add_layer(bldg_layer)
            building_layers[idx] = bldg_layer
    
    # Create UI widgets
    height_input = FloatText(
        value=10.0,
        description='Height (m):',
        disabled=False,
        style={'description_width': 'initial'}
    )
    
    add_button = Button(
        description='Add Building',
        button_style='success',
        disabled=True
    )
    
    clear_button = Button(
        description='Clear Drawing',
        button_style='warning',
        disabled=True
    )
    
    status_output = Output()
    hover_info = HTML("")
    
    # Create control panel
    control_panel = VBox([
        HTML("<h3>Draw Building Tool</h3>"),
        HTML("<p>1. Draw a polygon on the map<br>2. Set height<br>3. Click 'Add Building'</p>"),
        height_input,
        HBox([add_button, clear_button]),
        status_output
    ])
    
    # Add control panel to map
    widget_control = WidgetControl(widget=control_panel, position='topright')
    m.add_control(widget_control)
    
    # Store the current drawn polygon
    current_polygon = {'vertices': [], 'layer': None}
    
    # Drawing control
    draw_control = DrawControl(
        polygon={
            "shapeOptions": {
                "color": "red",
                "fillColor": "red",
                "fillOpacity": 0.3,
                "weight": 3
            }
        },
        rectangle={},
        circle={},
        circlemarker={},
        polyline={},
        marker={}
    )
    
    def handle_draw(self, action, geo_json):
        """Handle polygon drawing events"""
        with status_output:
            clear_output()
            
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            # Store vertices
            coordinates = geo_json['geometry']['coordinates'][0]
            current_polygon['vertices'] = [(coord[0], coord[1]) for coord in coordinates[:-1]]
            
            # Enable buttons
            add_button.disabled = False
            clear_button.disabled = False
            
            with status_output:
                print(f"Polygon drawn with {len(current_polygon['vertices'])} vertices")
                print("Set height and click 'Add Building'")
    
    def add_building_click(b):
        """Handle add building button click"""
        # Use nonlocal to modify the outer scope variable
        nonlocal updated_gdf
        
        with status_output:
            clear_output()
            
            if current_polygon['vertices']:
                # Create polygon geometry
                polygon = geom.Polygon(current_polygon['vertices'])
                
                # Get next building ID and ID values (ensure uniqueness)
                if len(updated_gdf) > 0:
                    next_building_id = int(updated_gdf['building_id'].max() + 1)
                    next_id = int(updated_gdf['id'].max() + 1)
                else:
                    next_building_id = 1
                    next_id = 1
                
                # Create new row data
                new_row_data = {
                    'geometry': polygon,
                    'height': float(height_input.value),
                    'min_height': 0.0,  # Default value as requested
                    'building_id': next_building_id,
                    'id': next_id
                }
                
                # Add any additional columns
                for col in updated_gdf.columns:
                    if col not in new_row_data:
                        new_row_data[col] = None
                
                # Append the new building in-place
                new_index = len(updated_gdf)
                updated_gdf.loc[new_index] = new_row_data
                
                # Add to map
                coords = list(polygon.exterior.coords)
                lat_lon_coords = [(c[1], c[0]) for c in coords[:-1]]
                
                new_layer = LeafletPolygon(
                    locations=lat_lon_coords,
                    color="blue",
                    fill_color="blue",
                    fill_opacity=0.3,
                    weight=2,
                    popup=HTML(f"<b>Building ID:</b> {next_building_id}<br>"
                              f"<b>ID:</b> {next_id}<br>"
                              f"<b>Height:</b> {height_input.value}m<br>"
                              f"<b>Min Height:</b> 0.0m")
                )
                m.add_layer(new_layer)
                
                # Clear drawing
                draw_control.clear()
                current_polygon['vertices'] = []
                add_button.disabled = True
                clear_button.disabled = True
                
                print(f"Building {next_building_id} added successfully!")
                print(f"ID: {next_id}, Height: {height_input.value}m, Min Height: 0.0m")
                print(f"Total buildings: {len(updated_gdf)}")
    
    def clear_drawing_click(b):
        """Handle clear drawing button click"""
        with status_output:
            clear_output()
            draw_control.clear()
            current_polygon['vertices'] = []
            add_button.disabled = True
            clear_button.disabled = True
            print("Drawing cleared")
    
    # Connect event handlers
    draw_control.on_draw(handle_draw)
    add_button.on_click(add_building_click)
    clear_button.on_click(clear_drawing_click)
    
    # Add draw control to map
    m.add_control(draw_control)
    
    # Display initial status
    with status_output:
        print(f"Total buildings loaded: {len(updated_gdf)}")
        print("Draw a polygon to add a new building")
    
    return m, updated_gdf


def get_polygon_vertices(drawn_polygons, polygon_id=None):
    """
    Extract vertices from drawn polygons data structure.
    
    This helper function provides a convenient way to extract polygon vertices
    from the drawn_polygons list returned by display_buildings_and_draw_polygon().
    
    Args:
        drawn_polygons: The drawn_polygons list returned from display_buildings_and_draw_polygon()
        polygon_id (int, optional): Specific polygon ID to extract. If None, returns all polygons.
    
    Returns:
        If polygon_id is specified: List of (lon, lat) tuples for that polygon
        If polygon_id is None: List of lists, where each inner list contains (lon, lat) tuples
    
    Example:
        >>> m, polygons = display_buildings_and_draw_polygon()
        >>> # Draw some polygons...
        >>> vertices = get_polygon_vertices(polygons, polygon_id=1)  # Get polygon 1
        >>> all_vertices = get_polygon_vertices(polygons)  # Get all polygons
    """
    if not drawn_polygons:
        return []
    
    if polygon_id is not None:
        # Return specific polygon
        for polygon in drawn_polygons:
            if polygon['id'] == polygon_id:
                return polygon['vertices']
        return []  # Polygon not found
    else:
        # Return all polygons
        return [polygon['vertices'] for polygon in drawn_polygons]


# Simple convenience function
def create_building_editor(building_gdf=None, initial_center=None, zoom=17, rectangle_vertices=None):
    """
    Creates and displays an interactive building editor.
    
    Args:
        building_gdf: Existing buildings GeoDataFrame (optional)
        initial_center: Map center as (lon, lat) tuple (optional)
        zoom: Initial zoom level (default=17)
    
    Returns:
        GeoDataFrame: The building GeoDataFrame that automatically updates
    
    Example:
        >>> buildings = create_building_editor()
        >>> # Draw buildings on the displayed map
        >>> print(buildings)  # Automatically contains all drawn buildings
    """
    m, gdf = draw_additional_buildings(building_gdf, initial_center, zoom, rectangle_vertices)
    display(m)
    return gdf


def draw_additional_trees(tree_gdf=None, initial_center=None, zoom=17, rectangle_vertices=None):
    """
    Creates an interactive map to add trees by clicking and setting parameters.
    
    Users can:
    - Set tree parameters: top height, bottom height, crown diameter
    - Click multiple times to add multiple trees with the same parameters
    - Update parameters at any time to change subsequent trees
    
    Args:
        tree_gdf (GeoDataFrame, optional): Existing trees to display.
            Expected columns: ['tree_id', 'top_height', 'bottom_height', 'crown_diameter', 'geometry']
        initial_center (tuple, optional): (lon, lat) for initial map center.
        zoom (int): Initial zoom level. Default=17.
        rectangle_vertices (list, optional): If provided, used to set center like buildings.
    
    Returns:
        tuple: (map_object, updated_tree_gdf)
    """
    # Initialize or copy the tree GeoDataFrame
    if tree_gdf is None:
        updated_trees = gpd.GeoDataFrame(
            columns=['tree_id', 'top_height', 'bottom_height', 'crown_diameter', 'geometry'],
            crs='EPSG:4326'
        )
    else:
        updated_trees = tree_gdf.copy()
        # Ensure required columns exist
        if 'tree_id' not in updated_trees.columns:
            updated_trees['tree_id'] = range(1, len(updated_trees) + 1)
        for col, default in [('top_height', 10.0), ('bottom_height', 4.0), ('crown_diameter', 6.0)]:
            if col not in updated_trees.columns:
                updated_trees[col] = default

    # Determine map center
    if initial_center is not None:
        center_lon, center_lat = initial_center
    elif updated_trees is not None and len(updated_trees) > 0:
        min_lon, min_lat, max_lon, max_lat = updated_trees.total_bounds
        center_lon = (min_lon + max_lon) / 2
        center_lat = (min_lat + max_lat) / 2
    elif rectangle_vertices is not None:
        center_lon, center_lat = (rectangle_vertices[0][0] + rectangle_vertices[2][0]) / 2, (rectangle_vertices[0][1] + rectangle_vertices[2][1]) / 2
    else:
        center_lon, center_lat = -100.0, 40.0

    # Create map
    m = Map(center=(center_lat, center_lon), zoom=zoom, scroll_wheel_zoom=True)
    # Add Google Satellite basemap with Esri fallback
    try:
        google_sat = TileLayer(
            url='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            name='Google Satellite',
            attribution='Google Satellite'
        )
        # Replace default base layer with Google Satellite
        m.layers = tuple([google_sat])
    except Exception:
        try:
            m.layers = tuple([basemap_to_tiles(basemaps.Esri.WorldImagery)])
        except Exception:
            # Fallback silently if basemap cannot be added
            pass

    # If rectangle_vertices provided, draw its edges on the map
    if rectangle_vertices is not None and len(rectangle_vertices) >= 4:
        try:
            lat_lon_coords = [(lat, lon) for lon, lat in rectangle_vertices]
            rect_outline = LeafletPolygon(
                locations=lat_lon_coords,
                color="#fed766",
                weight=2,
                fill_color="#fed766",
                fill_opacity=0.0
            )
            m.add_layer(rect_outline)
        except Exception:
            pass

    # Display existing trees as circles
    tree_layers = {}
    for idx, row in updated_trees.iterrows():
        if row.geometry is not None and hasattr(row.geometry, 'x'):
            lat = row.geometry.y
            lon = row.geometry.x
            # Ensure integer radius in meters as required by ipyleaflet Circle
            radius_m = max(int(round(float(row.get('crown_diameter', 6.0)) / 2.0)), 1)
            tree_id_val = int(row.get('tree_id', idx+1))
            circle = Circle(location=(lat, lon), radius=radius_m, color='#2ab7ca', weight=1, opacity=1.0, fill_color='#2ab7ca', fill_opacity=0.3)
            m.add_layer(circle)
            tree_layers[tree_id_val] = circle

    # UI widgets for parameters
    top_height_input = FloatText(value=10.0, description='Top height (m):', disabled=False, style={'description_width': 'initial'})
    bottom_height_input = FloatText(value=4.0, description='Bottom height (m):', disabled=False, style={'description_width': 'initial'})
    crown_diameter_input = FloatText(value=6.0, description='Crown diameter (m):', disabled=False, style={'description_width': 'initial'})
    fixed_prop_checkbox = Checkbox(value=True, description='Fixed proportion', indent=False)

    add_mode_button = Button(description='Add', button_style='success')
    remove_mode_button = Button(description='Remove', button_style='')
    status_output = Output()
    hover_info = HTML("")

    control_panel = VBox([
        HTML("<h3 style=\"margin:0 0 4px 0;\">Tree Placement Tool</h3>"),
        HTML("<div style=\"margin:0 0 6px 0;\">1. Choose Add/Remove mode<br>2. Set tree parameters (top, bottom, crown)<br>3. Click on the map to add/remove consecutively<br>4. Hover over a tree to view parameters</div>"),
        HBox([add_mode_button, remove_mode_button]),
        top_height_input,
        bottom_height_input,
        crown_diameter_input,
        fixed_prop_checkbox,
        hover_info,
        status_output
    ])

    widget_control = WidgetControl(widget=control_panel, position='topright')
    m.add_control(widget_control)

    # State for mode
    mode = 'add'
    # Fixed proportion state
    base_bottom_ratio = bottom_height_input.value / top_height_input.value if top_height_input.value else 0.4
    base_crown_ratio = crown_diameter_input.value / top_height_input.value if top_height_input.value else 0.6
    updating_params = False

    def recompute_from_top(new_top: float):
        nonlocal updating_params
        if new_top <= 0:
            return
        new_bottom = max(0.0, base_bottom_ratio * new_top)
        new_crown = max(0.0, base_crown_ratio * new_top)
        updating_params = True
        bottom_height_input.value = new_bottom
        crown_diameter_input.value = new_crown
        updating_params = False

    def recompute_from_bottom(new_bottom: float):
        nonlocal updating_params
        if base_bottom_ratio <= 0:
            return
        new_top = max(0.0, new_bottom / base_bottom_ratio)
        new_crown = max(0.0, base_crown_ratio * new_top)
        updating_params = True
        top_height_input.value = new_top
        crown_diameter_input.value = new_crown
        updating_params = False

    def recompute_from_crown(new_crown: float):
        nonlocal updating_params
        if base_crown_ratio <= 0:
            return
        new_top = max(0.0, new_crown / base_crown_ratio)
        new_bottom = max(0.0, base_bottom_ratio * new_top)
        updating_params = True
        top_height_input.value = new_top
        bottom_height_input.value = new_bottom
        updating_params = False

    def on_toggle_fixed(change):
        nonlocal base_bottom_ratio, base_crown_ratio
        if change['name'] == 'value':
            if change['new']:
                # Capture current ratios as baseline
                top = float(top_height_input.value) or 1.0
                bot = float(bottom_height_input.value)
                crn = float(crown_diameter_input.value)
                base_bottom_ratio = max(0.0, bot / top)
                base_crown_ratio = max(0.0, crn / top)
            else:
                # Keep last ratios but do not auto-update
                pass

    def on_top_change(change):
        if change['name'] == 'value' and fixed_prop_checkbox.value and not updating_params:
            try:
                recompute_from_top(float(change['new']))
            except Exception:
                pass

    def on_bottom_change(change):
        if change['name'] == 'value' and fixed_prop_checkbox.value and not updating_params:
            try:
                recompute_from_bottom(float(change['new']))
            except Exception:
                pass

    def on_crown_change(change):
        if change['name'] == 'value' and fixed_prop_checkbox.value and not updating_params:
            try:
                recompute_from_crown(float(change['new']))
            except Exception:
                pass

    fixed_prop_checkbox.observe(on_toggle_fixed, names='value')
    top_height_input.observe(on_top_change, names='value')
    bottom_height_input.observe(on_bottom_change, names='value')
    crown_diameter_input.observe(on_crown_change, names='value')

    def set_mode(new_mode):
        nonlocal mode
        mode = new_mode
        # Visual feedback
        add_mode_button.button_style = 'success' if mode == 'add' else ''
        remove_mode_button.button_style = 'danger' if mode == 'remove' else ''
        # No on-screen mode label

    def on_click_add(b):
        set_mode('add')

    def on_click_remove(b):
        set_mode('remove')

    add_mode_button.on_click(on_click_add)
    remove_mode_button.on_click(on_click_remove)

    # Consecutive placements by map click
    def handle_map_click(**kwargs):
        nonlocal updated_trees
        with status_output:
            clear_output()

        if kwargs.get('type') == 'click':
            lat, lon = kwargs.get('coordinates', (None, None))
            if lat is None or lon is None:
                return
            if mode == 'add':
                # Determine next tree_id
                next_tree_id = int(updated_trees['tree_id'].max() + 1) if len(updated_trees) > 0 else 1

                # Clamp/validate parameters
                th = float(top_height_input.value)
                bh = float(bottom_height_input.value)
                cd = float(crown_diameter_input.value)
                if bh > th:
                    bh, th = th, bh
                if cd < 0:
                    cd = 0.0

                # Create new tree row
                new_row = {
                    'tree_id': next_tree_id,
                    'top_height': th,
                    'bottom_height': bh,
                    'crown_diameter': cd,
                    'geometry': geom.Point(lon, lat)
                }

                # Append
                new_index = len(updated_trees)
                updated_trees.loc[new_index] = new_row

                # Add circle layer representing crown diameter (radius in meters)
                radius_m = max(int(round(new_row['crown_diameter'] / 2.0)), 1)
                circle = Circle(location=(lat, lon), radius=radius_m, color='#2ab7ca', weight=1, opacity=1.0, fill_color='#2ab7ca', fill_opacity=0.3)
                m.add_layer(circle)

                tree_layers[next_tree_id] = circle

                with status_output:
                    print(f"Tree {next_tree_id} added at (lon, lat)=({lon:.6f}, {lat:.6f})")
                    print(f"Top: {new_row['top_height']} m, Bottom: {new_row['bottom_height']} m, Crown: {new_row['crown_diameter']} m")
                    print(f"Total trees: {len(updated_trees)}")
            else:
                # Remove mode: find the nearest tree within its crown radius + 5m
                candidate_id = None
                candidate_idx = None
                candidate_dist = None
                for idx2, row2 in updated_trees.iterrows():
                    if row2.geometry is None or not hasattr(row2.geometry, 'x'):
                        continue
                    lat2 = row2.geometry.y
                    lon2 = row2.geometry.x
                    dist_m = distance.distance((lat, lon), (lat2, lon2)).meters
                    rad_m = max(int(round(float(row2.get('crown_diameter', 6.0)) / 2.0)), 1)
                    thr_m = rad_m + 5
                    if (candidate_dist is None and dist_m <= thr_m) or (candidate_dist is not None and dist_m < candidate_dist and dist_m <= thr_m):
                        candidate_dist = dist_m
                        candidate_id = int(row2.get('tree_id', idx2+1))
                        candidate_idx = idx2

                if candidate_id is not None:
                    # Remove layer
                    layer = tree_layers.get(candidate_id)
                    if layer is not None:
                        m.remove_layer(layer)
                        del tree_layers[candidate_id]
                    # Remove from gdf
                    updated_trees.drop(index=candidate_idx, inplace=True)
                    updated_trees.reset_index(drop=True, inplace=True)
                    with status_output:
                        print(f"Removed tree {candidate_id} (distance {candidate_dist:.2f} m)")
                        print(f"Total trees: {len(updated_trees)}")
                else:
                    with status_output:
                        print("No tree near the clicked location to remove")
        elif kwargs.get('type') == 'mousemove':
            lat, lon = kwargs.get('coordinates', (None, None))
            if lat is None or lon is None:
                return
            # Find a tree the cursor is over (within crown radius)
            shown = False
            for _, row2 in updated_trees.iterrows():
                if row2.geometry is None or not hasattr(row2.geometry, 'x'):
                    continue
                lat2 = row2.geometry.y
                lon2 = row2.geometry.x
                dist_m = distance.distance((lat, lon), (lat2, lon2)).meters
                rad_m = max(int(round(float(row2.get('crown_diameter', 6.0)) / 2.0)), 1)
                if dist_m <= rad_m:
                    hover_info.value = (
                        f"<div style=\"color:#d61f1f; font-weight:600; margin:2px 0;\">"
                        f"Tree {int(row2.get('tree_id', 0))} | Top {float(row2.get('top_height', 10.0))} m | "
                        f"Bottom {float(row2.get('bottom_height', 0.0))} m | Crown {float(row2.get('crown_diameter', 6.0))} m"
                        f"</div>"
                    )
                    shown = True
                    break
            if not shown:
                hover_info.value = ""
    m.on_interaction(handle_map_click)

    with status_output:
        print(f"Total trees loaded: {len(updated_trees)}")
        print("Set parameters, then click on the map to add trees")

    return m, updated_trees


def create_tree_editor(tree_gdf=None, initial_center=None, zoom=17, rectangle_vertices=None):
    """
    Convenience wrapper to display the tree editor map and return the GeoDataFrame.
    """
    m, gdf = draw_additional_trees(tree_gdf, initial_center, zoom, rectangle_vertices)
    display(m)
    return gdf