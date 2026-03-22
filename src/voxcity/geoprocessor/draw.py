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
import time
import numpy as np
from pyproj import Transformer
from ipyleaflet import (
    Map,
    DrawControl,
    Rectangle,
    Polygon as LeafletPolygon,
    Polyline,
    WidgetControl,
    Circle,
    basemaps,
    basemap_to_tiles,
    TileLayer,
    GeoJSON,
)
from geopy import distance
import shapely.geometry as geom
import geopandas as gpd
from ipywidgets import (
    VBox,
    HBox,
    Button,
    FloatText,
    Label,
    Output,
    HTML,
    Checkbox,
    ToggleButton,
    Layout,
)
import pandas as pd
from IPython.display import display, clear_output

from .utils import get_coordinates_from_cityname

# Import VoxCity for type checking (avoid circular import with TYPE_CHECKING)
try:
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from ..models import VoxCity
except ImportError:
    pass

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

    # Define transformers (modern pyproj API)
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    # Project vertices from WGS84 to Web Mercator for proper distance calculations
    projected_vertices = [to_merc.transform(lon, lat) for lon, lat in rectangle_vertices]

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
    new_vertices = [to_wgs84.transform(x, y) for x, y in rotated_vertices]

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
            # First calculate north/south latitudes from center
            north = distance.distance(meters=north_south_length/2).destination((lat, lon), bearing=0)
            south = distance.distance(meters=north_south_length/2).destination((lat, lon), bearing=180)
            
            # Calculate east/west at the SOUTH latitude to ensure correct grid dimensions
            # The grid size calculation uses the SW corner (vertex_0) as reference,
            # measuring E-W distance along the south edge. By calculating east/west
            # at the south latitude, we ensure the E-W distance matches the requested length.
            east_at_south = distance.distance(meters=east_west_length/2).destination((south.latitude, lon), bearing=90)
            west_at_south = distance.distance(meters=east_west_length/2).destination((south.latitude, lon), bearing=270)

            # Create rectangle vertices in counter-clockwise order (lon,lat)
            # Using the east/west longitudes calculated at south latitude for all corners
            rectangle_vertices.extend([
                (west_at_south.longitude, south.latitude),
                (west_at_south.longitude, north.latitude),
                (east_at_south.longitude, north.latitude),
                (east_at_south.longitude, south.latitude)                
            ])

            # Create and add new rectangle to map (ipyleaflet expects lat,lon)
            rectangle = Rectangle(
                bounds=[(north.latitude, west_at_south.longitude), (south.latitude, east_at_south.longitude)],
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

def display_buildings_and_draw_polygon(voxcity=None, building_gdf=None, rectangle_vertices=None, zoom=17):
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
        voxcity (VoxCity, optional): A VoxCity object from which to extract building_gdf 
            and rectangle_vertices. If provided, these values will be used unless 
            explicitly overridden by the building_gdf or rectangle_vertices parameters.
        building_gdf (GeoDataFrame, optional): A GeoDataFrame containing building footprints.
            Must have geometry column with Polygon type features.
            Geometries should be in [lon, lat] coordinate order.
            If None and voxcity is provided, uses voxcity.extras['building_gdf'].
            If None and no voxcity provided, only the base map is displayed.
        rectangle_vertices (list, optional): List of [lon, lat] coordinates defining rectangle corners.
            Used to set the initial map view extent.
            Takes precedence over building_gdf for determining map center.
            If None and voxcity is provided, uses voxcity.extras['rectangle_vertices'].
        zoom (int): Initial zoom level for the map. Default=17.
            Range: 0 (most zoomed out) to 18 (most zoomed in).
            Default of 17 is optimized for building-level detail.

    Returns:
        tuple: (map_object, drawn_polygons)
            - map_object: ipyleaflet Map instance with building footprints and drawing controls
            - drawn_polygons: List of dictionaries with 'id', 'vertices', and 'color' keys for all drawn polygons.
              Each polygon has a unique ID and color for easy identification.

    Examples:
        Using a VoxCity object:
        >>> m, polygons = display_buildings_and_draw_polygon(voxcity=my_voxcity)
        
        Using explicit parameters:
        >>> m, polygons = display_buildings_and_draw_polygon(building_gdf=buildings, rectangle_vertices=rect)
        
        Override specific parameters from VoxCity:
        >>> m, polygons = display_buildings_and_draw_polygon(voxcity=my_voxcity, zoom=15)

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
    # 0. Extract data from VoxCity object if provided
    # ---------------------------------------------------------
    if voxcity is not None:
        # Extract building_gdf if not explicitly provided
        if building_gdf is None:
            building_gdf = voxcity.extras.get('building_gdf', None)
        
        # Extract rectangle_vertices if not explicitly provided
        if rectangle_vertices is None:
            rectangle_vertices = voxcity.extras.get('rectangle_vertices', None)
    
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



def draw_additional_buildings(
    voxcity=None,
    building_gdf=None,
    initial_center=None,
    zoom=17,
    rectangle_vertices=None,
):
    """
    Interactive map editor: Draw rectangles, freehand polygons, and DELETE existing buildings.
    
    Args:
        initial_center (tuple): (Longitude, Latitude) - Standard GeoJSON order.
    """
    # --- Data Initialization ---
    if voxcity is not None:
        if building_gdf is None:
            building_gdf = voxcity.extras.get("building_gdf", None)
        if rectangle_vertices is None:
            rectangle_vertices = voxcity.extras.get("rectangle_vertices", None)
    
    if building_gdf is None:
        updated_gdf = gpd.GeoDataFrame(
            columns=["id", "height", "min_height", "geometry", "building_id"],
            crs="EPSG:4326",
        )
    else:
        updated_gdf = building_gdf.copy()
        updated_gdf = updated_gdf.reset_index(drop=True)
        defaults = {"height": 10.0, "min_height": 0.0, "building_id": 0, "id": 0}
        for col, val in defaults.items():
            if col not in updated_gdf.columns:
                updated_gdf[col] = (
                    val if col not in ["building_id", "id"] else range(len(updated_gdf))
                )
    
    # --- Map Setup (Corrected for Lon, Lat input) ---
    if initial_center is not None:
        # User provides (Lon, Lat) per GeoJSON standard
        center_lon, center_lat = initial_center
    elif not updated_gdf.empty:
        # GDF bounds are (minx, miny, maxx, maxy) -> (Lon, Lat, Lon, Lat)
        b = updated_gdf.total_bounds
        center_lon, center_lat = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    else:
        center_lon, center_lat = -100.0, 40.0
    
    # ipyleaflet expects (Lat, Lon), so we flip it here
    m = Map(center=(center_lat, center_lon), zoom=zoom, scroll_wheel_zoom=True)

    # Use CartoDB Positron as basemap (clean, light style)
    carto_light = TileLayer(
        url="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}@2x.png",
        attribution='&copy; <a href="https://carto.com/">CARTO</a>',
        name="CartoDB Positron",
        max_zoom=20,
    )
    m.layers = (carto_light,)

    # Remove any default DrawControl that may have been added
    for ctrl in list(m.controls):
        if isinstance(ctrl, DrawControl):
            m.remove_control(ctrl)

    # --- UI Setup (Gemini-style minimal design) ---
    style_html = HTML(
        """
    <style>
        /* Hide Leaflet.draw toolbar if present */
        .leaflet-draw { display: none !important; }

        /* Force crosshair cursor during drawing modes */
        .drawing-mode,
        .drawing-mode .leaflet-container,
        .drawing-mode .leaflet-interactive,
        .drawing-mode .leaflet-grab,
        .drawing-mode .leaflet-overlay-pane,
        .drawing-mode .leaflet-overlay-pane * {
            cursor: crosshair !important;
        }
        .delete-mode,
        .delete-mode .leaflet-container,
        .delete-mode .leaflet-interactive,
        .delete-mode .leaflet-grab {
            cursor: pointer !important;
        }

        /* ── Gemini-style panel ── */
        .gm-root {
            font-family: 'Google Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
            color: #1f1f1f;
            line-height: 1.5;
        }
        .gm-root * { box-sizing: border-box; }

        /* title */
        .gm-title {
            font-size: 14px; font-weight: 500; color: #1f1f1f;
            padding-bottom: 8px;
            border-bottom: 1px solid #e8eaed;
            margin-bottom: 12px;
        }
        /* section labels */
        .gm-label {
            font-size: 11px; font-weight: 500; color: #5f6368;
            letter-spacing: 0.3px;
            margin: 0 0 6px 0;
        }
        /* separator */
        .gm-sep { height: 1px; background: #e8eaed; margin: 12px 0; }

        /* status chip */
        .gm-status {
            padding: 6px 12px; border-radius: 16px;
            font-size: 11px; font-weight: 400; line-height: 1.3;
            margin-top: 10px; text-align: center;
        }
        .gm-status-info    { background: #f0f4ff; color: #1a73e8; }
        .gm-status-success { background: #e6f4ea; color: #137333; }
        .gm-status-warn    { background: #fef7e0; color: #b06000; }
        .gm-status-danger  { background: #fce8e6; color: #c5221f; }

        /* ── Override ipywidgets button styles ── */
        .gm-root .jupyter-button {
            border-radius: 18px !important;
            font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            border: 1px solid #dadce0 !important;
            box-shadow: none !important;
            transition: background 0.15s, border-color 0.15s, box-shadow 0.15s !important;
        }
        .gm-root .jupyter-button:hover {
            box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        }

        /* default / neutral buttons */
        .gm-root .jupyter-button:not(.mod-primary):not(.mod-danger):not(.mod-warning):not(.mod-success):not(.mod-info) {
            background: #f8f9fa !important;
            color: #3c4043 !important;
            border-color: #dadce0 !important;
        }
        .gm-root .jupyter-button:not(.mod-primary):not(.mod-danger):not(.mod-warning):not(.mod-success):not(.mod-info):hover {
            background: #f1f3f4 !important;
        }

        /* primary (Add) */
        .gm-root .mod-primary {
            background: #1a73e8 !important;
            color: #fff !important;
            border-color: #1a73e8 !important;
        }
        .gm-root .mod-primary:hover {
            background: #1765cc !important;
            border-color: #1765cc !important;
        }
        .gm-root .mod-primary:disabled {
            background: #e8eaed !important;
            color: #9aa0a6 !important;
            border-color: #e8eaed !important;
        }

        /* danger (Remove toggles) */
        .gm-root .mod-danger {
            background: #fff !important;
            color: #c5221f !important;
            border-color: #f1c8c6 !important;
        }
        .gm-root .mod-danger:hover {
            background: #fce8e6 !important;
        }
        .gm-root .mod-danger.mod-active {
            background: #fce8e6 !important;
            border-color: #c5221f !important;
        }

        /* toggle active state */
        .gm-root .jupyter-button.mod-active:not(.mod-danger) {
            background: #e8f0fe !important;
            color: #1a73e8 !important;
            border-color: #1a73e8 !important;
        }

        /* input fields */
        .gm-root input[type="number"] {
            border-radius: 8px !important;
            border: 1px solid #dadce0 !important;
            font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif !important;
            font-size: 12px !important;
            padding: 2px 6px !important;
        }
        .gm-root input[type="number"]:focus {
            border-color: #1a73e8 !important;
            outline: none !important;
        }
        .gm-root .widget-label {
            font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif !important;
            font-size: 11px !important;
            color: #5f6368 !important;
            font-weight: 500 !important;
        }
    </style>
    """
    )

    # --- ADD Section ---
    add_label = HTML("<div class='gm-label'>Add</div>")

    rect_btn = ToggleButton(
        value=False,
        description="Rectangle",
        icon="",
        layout=Layout(width="92px", height="30px"),
        tooltip="Click 3 corners on map to draw rectangle",
    )
    poly_btn = ToggleButton(
        value=False,
        description="Polygon",
        icon="",
        layout=Layout(width="82px", height="30px"),
        tooltip="Click to draw polygon, double-click to finish",
    )

    h_in = FloatText(
        value=10.0,
        description="Height",
        layout=Layout(width="115px", height="28px"),
        style={"description_width": "42px"},
    )
    mh_in = FloatText(
        value=0.0,
        description="Base",
        layout=Layout(width="100px", height="28px"),
        style={"description_width": "34px"},
    )
    add_btn = Button(
        description="Add",
        button_style="primary",
        icon="plus",
        disabled=True,
        layout=Layout(flex="1", height="32px"),
    )
    clr_btn = Button(
        description="Clear",
        button_style="",
        icon="",
        disabled=True,
        layout=Layout(width="64px", height="32px"),
        tooltip="Clear drawing",
    )

    # --- REMOVE Section ---
    sep = HTML("<div class='gm-sep'></div>")
    remove_label = HTML("<div class='gm-label'>Remove</div>")

    del_btn = ToggleButton(
        value=False,
        description="Click",
        icon="",
        button_style="danger",
        layout=Layout(width="72px", height="30px"),
        tooltip="Click on buildings to remove",
    )
    poly_del_btn = ToggleButton(
        value=False,
        description="Area",
        icon="",
        button_style="danger",
        layout=Layout(width="68px", height="30px"),
        tooltip="Draw polygon to remove buildings inside",
    )

    # --- Status ---
    status_bar = HTML(
        value="<div class='gm-status gm-status-info'>Ready</div>"
    )

    # Layout
    add_tools_row = HBox([rect_btn, poly_btn], layout=Layout(margin="0 0 6px 0", align_items="center", gap="6px"))
    input_row = HBox([h_in, mh_in], layout=Layout(margin="0 0 6px 0"))
    action_row = HBox([add_btn, clr_btn], layout=Layout(margin="0", gap="6px"))
    remove_tools_row = HBox([del_btn, poly_del_btn], layout=Layout(margin="0", gap="6px"))

    panel = VBox(
        [
            style_html,
            HTML("<div class='gm-title'>Building Editor</div>"),
            add_label,
            add_tools_row,
            input_row,
            action_row,
            sep,
            remove_label,
            remove_tools_row,
            status_bar,
        ],
        layout=Layout(
            width="260px",
            padding="14px 16px",
        ),
    )
    panel.add_class("gm-root")

    # Wrap in an outer container for card styling (border-radius, shadow)
    card = VBox(
        [panel],
        layout=Layout(
            background_color="white",
            border_radius="16px",
            box_shadow="0 1px 3px rgba(0,0,0,0.1), 0 4px 16px rgba(0,0,0,0.06)",
            overflow="hidden",
        ),
    )

    m.add_control(WidgetControl(widget=card, position="topright"))

    # --- Global State & Transformers ---
    state = {"poly": [], "clicks": [], "temp_layers": [], "preview": None, "removal_poly": None, "removal_preview": None, "removal_clicks": [], "removal_layers": [], "poly_clicks": [], "poly_layers": [], "poly_preview": None}
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    _style_attr = getattr(m, "default_style", {})
    original_style = _style_attr.copy() if isinstance(_style_attr, dict) else {"cursor": "grab"}
    
    # --- Helper Functions ---
    def set_status(msg, type="info"):
        status_bar.value = (
            f"<div class='gm-panel gm-status gm-status-{type}'>{msg}</div>"
        )

    # --- Batch render buildings via GeoJSON layer (fast) ---
    def _build_geojson_data():
        """Build a GeoJSON FeatureCollection from the current GeoDataFrame."""
        features = []
        for idx, row in updated_gdf.iterrows():
            if isinstance(row.geometry, geom.Polygon):
                coords = [list(row.geometry.exterior.coords)]
                # Sanitize: skip if coordinates contain NaN
                if any(math.isnan(c) for ring in coords for pt in ring for c in pt):
                    continue
                h = row.get("height", 0)
                h = 0.0 if (h is None or (isinstance(h, float) and math.isnan(h))) else float(h)
                features.append({
                    "type": "Feature",
                    "id": str(idx),
                    "properties": {"idx": int(idx), "height": h},
                    "geometry": {"type": "Polygon", "coordinates": coords},
                })
        return {"type": "FeatureCollection", "features": features}

    _geojson_style = {
        "color": "#2196F3",
        "fillColor": "#2196F3",
        "fillOpacity": 0.4,
        "weight": 1,
    }

    buildings_geojson = GeoJSON(
        data=_build_geojson_data(),
        style=_geojson_style,
        hover_style={"fillOpacity": 0.6},
    )

    def _on_geojson_click(event=None, feature=None, id=None, properties=None, **kwargs):
        if del_btn.value and properties:
            gdf_index = properties.get("idx")
            if gdf_index is not None:
                try:
                    updated_gdf.drop(index=gdf_index, inplace=True)
                    buildings_geojson.data = _build_geojson_data()
                    set_status(f"Removed #{gdf_index}", "danger")
                except KeyError:
                    pass

    buildings_geojson.on_click(_on_geojson_click)
    m.add_layer(buildings_geojson)

    def clear_removal_preview():
        if state["removal_preview"]:
            try:
                m.remove_layer(state["removal_preview"])
            except Exception:
                pass
            state["removal_preview"] = None
        state["removal_poly"] = None
        # Clear custom polygon drawing layers
        while state["removal_layers"]:
            try:
                m.remove_layer(state["removal_layers"].pop())
            except Exception:
                pass
        state["removal_clicks"] = []

    # --- Cursor helpers ---
    def _set_drawing_cursor():
        m.default_style = {"cursor": "crosshair"}
        m.remove_class("delete-mode")
        m.add_class("drawing-mode")

    def _set_delete_cursor():
        m.default_style = {"cursor": "pointer"}
        m.remove_class("drawing-mode")
        m.add_class("delete-mode")

    def _reset_cursor():
        m.default_style = original_style
        m.remove_class("drawing-mode")
        m.remove_class("delete-mode")

    # --- Logic ---
    def on_mode_change(change):
        if change["owner"] is rect_btn and change["new"]:
            poly_btn.value = False
            del_btn.value = False
            poly_del_btn.value = False
            clear_removal_preview()
            clear_poly_draw()
            _set_drawing_cursor()
            set_status("Step 1/3 \u2014 Click first corner", "info")
        elif change["owner"] is poly_btn and change["new"]:
            rect_btn.value = False
            del_btn.value = False
            poly_del_btn.value = False
            clear_all(None)
            clear_removal_preview()
            _set_drawing_cursor()
            m.double_click_zoom = False
            set_status("Click to draw polygon, click first point to close", "info")
        elif change["owner"] is del_btn and change["new"]:
            rect_btn.value = False
            poly_btn.value = False
            poly_del_btn.value = False
            clear_all(None)
            clear_removal_preview()
            clear_poly_draw()
            _set_delete_cursor()
            set_status("Click buildings to delete", "danger")
        elif change["owner"] is poly_del_btn and change["new"]:
            rect_btn.value = False
            poly_btn.value = False
            del_btn.value = False
            clear_all(None)
            clear_removal_preview()
            clear_poly_draw()
            _set_drawing_cursor()
            m.double_click_zoom = False
            set_status("Click to draw removal area, click first point to close", "danger")
        elif not rect_btn.value and not del_btn.value and not poly_del_btn.value and not poly_btn.value:
            _reset_cursor()
            m.double_click_zoom = True
            clear_removal_preview()
            clear_poly_draw()
            set_status("Ready", "info")

    rect_btn.observe(on_mode_change, names="value")
    poly_btn.observe(on_mode_change, names="value")
    del_btn.observe(on_mode_change, names="value")
    poly_del_btn.observe(on_mode_change, names="value")

    def clear_preview():
        if state["preview"]:
            try:
                m.remove_layer(state["preview"])
            except Exception:
                pass
            state["preview"] = None
    
    def clear_poly_draw():
        """Clear polygon drawing layers."""
        while state["poly_layers"]:
            try:
                m.remove_layer(state["poly_layers"].pop())
            except Exception:
                pass
        if state["poly_preview"]:
            try:
                m.remove_layer(state["poly_preview"])
            except Exception:
                pass
            state["poly_preview"] = None
        state["poly_clicks"] = []

    def clear_temps():
        while state["temp_layers"]:
            try:
                m.remove_layer(state["temp_layers"].pop())
            except Exception:
                pass

    def refresh_markers():
        clear_temps()
        for lon, lat in state["clicks"]:
            pt = Circle(
                location=(lat, lon),
                radius=2,
                color="red",
                fill_color="red",
                fill_opacity=1.0,
            )
            m.add_layer(pt)
            state["temp_layers"].append(pt)
        if len(state["clicks"]) >= 2:
            (l1, la1), (l2, la2) = state["clicks"][0], state["clicks"][1]
            line = Polyline(
                locations=[(la1, l1), (la2, l2)],
                color="red",
                weight=3,
            )
            m.add_layer(line)
            state["temp_layers"].append(line)

    def build_rect(points):
        (lon1, lat1), (lon2, lat2), (lon3, lat3) = points[:3]
        x1, y1 = to_merc.transform(lon1, lat1)
        x2, y2 = to_merc.transform(lon2, lat2)
        x3, y3 = to_merc.transform(lon3, lat3)
        wx, wy = x2 - x1, y2 - y1
        if math.hypot(wx, wy) < 0.5:
            return None, "Width too small"
        ux, uy = wx / math.hypot(wx, wy), wy / math.hypot(wx, wy)
        px, py = -uy, ux
        vx, vy = x3 - x1, y3 - y1
        h_len = vx * px + vy * py
        if abs(h_len) < 0.5:
            return None, "Height too small"
        hx, hy = px * h_len, py * h_len
        corners_merc = [
            (x1, y1),
            (x2, y2),
            (x2 + hx, y2 + hy),
            (x1 + hx, y1 + hy),
        ]
        return [to_geo.transform(*p) for p in corners_merc], None

    # Mousemove throttle state
    _last_mousemove = [0.0]
    _last_removal_click = [0.0]  # Guard against click events from dblclick
    _last_poly_click = [0.0]    # Guard for polygon draw mode

    def _refresh_poly_markers():
        """Redraw markers and edges for the polygon being drawn."""
        while state["poly_layers"]:
            try:
                m.remove_layer(state["poly_layers"].pop())
            except Exception:
                pass
        pts = state["poly_clicks"]
        for lon, lat in pts:
            pt = Circle(location=(lat, lon), radius=2, color="#4CAF50", fill_color="#4CAF50", fill_opacity=1.0)
            m.add_layer(pt)
            state["poly_layers"].append(pt)
        for i in range(len(pts) - 1):
            line = Polyline(locations=[(pts[i][1], pts[i][0]), (pts[i+1][1], pts[i+1][0])], color="#4CAF50", weight=2)
            m.add_layer(line)
            state["poly_layers"].append(line)

    # Proximity threshold in degrees (~10 pixels at typical zoom)
    _CLOSE_THRESHOLD = 0.0001

    def _is_near_first(pts, lon, lat):
        """Check if (lon, lat) is close to the first point in pts."""
        if len(pts) < 3:
            return False
        dx = lon - pts[0][0]
        dy = lat - pts[0][1]
        return (dx * dx + dy * dy) < (_CLOSE_THRESHOLD * _CLOSE_THRESHOLD)

    def _finish_poly_footprint(pts):
        """Finalize a polygon footprint from clicked points."""
        clear_poly_draw()
        state["poly"] = list(pts)
        poly_locs = [(lat, lon) for lon, lat in pts]
        preview = LeafletPolygon(
            locations=poly_locs,
            color="#4CAF50",
            fill_color="#4CAF50",
            fill_opacity=0.3,
        )
        state["preview"] = preview
        m.add_layer(preview)
        add_btn.disabled = False
        clr_btn.disabled = False
        set_status("Shape ready \u2014 set height and +Add", "success")

    def _execute_polygon_removal(polygon_coords):
        """Remove buildings within the drawn polygon."""
        removal_polygon = geom.Polygon(polygon_coords)
        buildings_to_remove = []
        for idx, row in updated_gdf.iterrows():
            if isinstance(row.geometry, geom.Polygon):
                if removal_polygon.contains(row.geometry) or removal_polygon.intersects(row.geometry):
                    buildings_to_remove.append(idx)
        if buildings_to_remove:
            removed_count = 0
            for idx in buildings_to_remove:
                try:
                    updated_gdf.drop(index=idx, inplace=True)
                    removed_count += 1
                except KeyError:
                    pass
            buildings_geojson.data = _build_geojson_data()
            clear_removal_preview()
            # Stay in removal mode for next area
            _set_drawing_cursor()
            m.double_click_zoom = False
            set_status(f"Removed {removed_count} \u2014 draw next area or deselect", "success")
        else:
            clear_removal_preview()
            set_status("No buildings in selected area", "warn")

    def _refresh_removal_markers():
        """Redraw markers and edges for the removal polygon being drawn."""
        # Clear old layers
        while state["removal_layers"]:
            try:
                m.remove_layer(state["removal_layers"].pop())
            except Exception:
                pass
        pts = state["removal_clicks"]
        # Draw vertices
        for lon, lat in pts:
            pt = Circle(location=(lat, lon), radius=2, color="#FF0000", fill_color="#FF0000", fill_opacity=1.0)
            m.add_layer(pt)
            state["removal_layers"].append(pt)
        # Draw edges
        for i in range(len(pts) - 1):
            line = Polyline(locations=[(pts[i][1], pts[i][0]), (pts[i+1][1], pts[i+1][0])], color="#FF0000", weight=2)
            m.add_layer(line)
            state["removal_layers"].append(line)

    def handle_map_interaction(**kwargs):
        # --- Custom polygon drawing for adding a building footprint ---
        if poly_btn.value:
            if kwargs.get("type") == "dblclick":
                pts = state["poly_clicks"]
                if len(pts) >= 3:
                    _finish_poly_footprint(pts)
                else:
                    set_status("Need at least 3 points", "warn")
                return
            elif kwargs.get("type") == "click":
                now = time.time()
                if now - _last_poly_click[0] < 0.3:
                    return
                _last_poly_click[0] = now
                coords = kwargs.get("coordinates")
                if not coords:
                    return
                lat, lon = coords
                # Close polygon if clicking near the first vertex
                if _is_near_first(state["poly_clicks"], lon, lat):
                    _finish_poly_footprint(state["poly_clicks"])
                    return
                state["poly_clicks"].append((lon, lat))
                _refresh_poly_markers()
                n = len(state["poly_clicks"])
                set_status(f"{n} point(s) \u2014 click first point to close", "info")
            elif kwargs.get("type") == "mousemove":
                now = time.time()
                if now - _last_mousemove[0] < 0.05:
                    return
                _last_mousemove[0] = now
                pts = state["poly_clicks"]
                if pts:
                    coords = kwargs.get("coordinates")
                    if not coords:
                        return
                    lat_c, lon_c = coords
                    if state["poly_preview"] and isinstance(state["poly_preview"], Polyline):
                        state["poly_preview"].locations = [(pts[-1][1], pts[-1][0]), (lat_c, lon_c)]
                    else:
                        if state["poly_preview"]:
                            try:
                                m.remove_layer(state["poly_preview"])
                            except Exception:
                                pass
                        line = Polyline(
                            locations=[(pts[-1][1], pts[-1][0]), (lat_c, lon_c)],
                            color="#4CAF50", weight=2, dash_array="5, 5",
                        )
                        state["poly_preview"] = line
                        m.add_layer(line)
            return

        # --- Custom polygon drawing for area removal ---
        if poly_del_btn.value:
            if kwargs.get("type") == "dblclick":
                pts = state["removal_clicks"]
                if len(pts) >= 3:
                    _execute_polygon_removal(pts)
                else:
                    set_status("Need at least 3 points", "warn")
                return
            elif kwargs.get("type") == "click":
                # Skip click events that are part of a double-click
                now = time.time()
                if now - _last_removal_click[0] < 0.3:
                    return
                _last_removal_click[0] = now
                coords = kwargs.get("coordinates")
                if not coords:
                    return
                lat, lon = coords
                # Close polygon if clicking near the first vertex
                if _is_near_first(state["removal_clicks"], lon, lat):
                    _execute_polygon_removal(state["removal_clicks"])
                    return
                state["removal_clicks"].append((lon, lat))
                _refresh_removal_markers()
                n = len(state["removal_clicks"])
                set_status(f"{n} point(s) \u2014 click first point to close", "danger")
            elif kwargs.get("type") == "mousemove":
                now = time.time()
                if now - _last_mousemove[0] < 0.05:
                    return
                _last_mousemove[0] = now
                pts = state["removal_clicks"]
                if pts:
                    coords = kwargs.get("coordinates")
                    if not coords:
                        return
                    lat_c, lon_c = coords
                    # Show a dashed preview line from last point to cursor
                    if state["removal_preview"] and isinstance(state["removal_preview"], Polyline):
                        state["removal_preview"].locations = [(pts[-1][1], pts[-1][0]), (lat_c, lon_c)]
                    else:
                        if state["removal_preview"]:
                            try:
                                m.remove_layer(state["removal_preview"])
                            except Exception:
                                pass
                        line = Polyline(
                            locations=[(pts[-1][1], pts[-1][0]), (lat_c, lon_c)],
                            color="#FF0000", weight=2, dash_array="5, 5",
                        )
                        state["removal_preview"] = line
                        m.add_layer(line)
            return

        if not rect_btn.value:
            return

        if kwargs.get("type") == "click":
            coords = kwargs.get("coordinates")
            if not coords:
                return
            lat, lon = coords
            state["clicks"].append((lon, lat))
            refresh_markers()
            
            count = len(state["clicks"])
            if count == 1:
                # Entering drawing mode: clear any reused preview so it starts fresh
                clear_preview()
                set_status("Step 2/3 — Click second corner", "info")
            elif count == 2:
                (l1, la1), (l2, la2) = state["clicks"]
                x1, y1 = to_merc.transform(l1, la1)
                x2, y2 = to_merc.transform(l2, la2)
                if math.hypot(x2 - x1, y2 - y1) < 0.5:
                    state["clicks"].pop()
                    refresh_markers()
                    set_status("Too close — click further away", "warn")
                else:
                    # Switching from line to polygon preview: clear the Polyline
                    clear_preview()
                    set_status("Step 3/3 — Click opposite side", "info")
            elif count == 3:
                verts, err = build_rect(state["clicks"])
                if err:
                    state["clicks"].pop()
                    set_status(f"{err} — try again", "warn")
                else:
                    clear_preview()
                    clear_temps()
                    state["poly"] = verts
                    poly_locs = [(lat, lon) for lon, lat in verts]
                    preview = LeafletPolygon(
                        locations=poly_locs,
                        color="#4CAF50",
                        fill_color="#4CAF50",
                        fill_opacity=0.3,
                    )
                    state["preview"] = preview
                    m.add_layer(preview)
                    add_btn.disabled = False
                    clr_btn.disabled = False
                    state["clicks"] = []
                    set_status("Shape ready — set height and +Add", "success")

        elif kwargs.get("type") == "mousemove":
            # Throttle: skip if less than 50ms since last update
            now = time.time()
            if now - _last_mousemove[0] < 0.05:
                return
            _last_mousemove[0] = now

            coords = kwargs.get("coordinates")
            if not coords:
                return
            lat_c, lon_c = coords
            if len(state["clicks"]) == 1:
                (lon1, lat1) = state["clicks"][0]
                new_locs = [(lat1, lon1), (lat_c, lon_c)]
                # Reuse existing Polyline if possible
                if state["preview"] and isinstance(state["preview"], Polyline):
                    state["preview"].locations = new_locs
                else:
                    clear_preview()
                    line = Polyline(
                        locations=new_locs,
                        color="#FF5722",
                        weight=2,
                        dash_array="5, 5",
                    )
                    state["preview"] = line
                    m.add_layer(line)
            elif len(state["clicks"]) == 2:
                tentative_clicks = state["clicks"] + [(lon_c, lat_c)]
                verts, err = build_rect(tentative_clicks)
                if not err:
                    poly_locs = [(lat, lon) for lon, lat in verts]
                    # Reuse existing LeafletPolygon if possible
                    if state["preview"] and isinstance(state["preview"], LeafletPolygon):
                        state["preview"].locations = poly_locs
                    else:
                        clear_preview()
                        poly = LeafletPolygon(
                            locations=poly_locs,
                            color="#FF5722",
                            weight=2,
                            fill_color="#FF5722",
                            fill_opacity=0.1,
                            dash_array="5, 5",
                        )
                        state["preview"] = poly
                        m.add_layer(poly)

    m.on_interaction(handle_map_interaction)

    def handle_freehand(self, action, geo_json):
        if action == "created" and geo_json["geometry"]["type"] == "Polygon":
            coords = geo_json["geometry"]["coordinates"][0]
            polygon_coords = [(c[0], c[1]) for c in coords[:-1]]
            
            # Normal mode - adding a freehand polygon as building
            rect_btn.value = False
            del_btn.value = False
            poly_del_btn.value = False
            state["clicks"] = []
            clear_preview()
            clear_temps()
            state["poly"] = polygon_coords
            add_btn.disabled = False
            clr_btn.disabled = False
            set_status("Shape ready — set height and add", "success")

    draw_control = DrawControl(
        polygon={"shapeOptions": {"color": "#FF5722", "fillColor": "#FF5722", "fillOpacity": 0.2}},
        rectangle={},
        circle={},
        polyline={},
        marker={},
        circlemarker={},
    )
    draw_control.on_draw(handle_freehand)
    m.add_control(draw_control)

    def add_geom(b):
        if not state["poly"]:
            return
        try:
            poly = geom.Polygon(state["poly"])
            new_idx = (updated_gdf.index.max() + 1) if not updated_gdf.empty else 1
            updated_gdf.loc[new_idx] = {
                "geometry": poly,
                "height": h_in.value,
                "min_height": mh_in.value,
                "building_id": new_idx,
                "id": new_idx,
            }
            buildings_geojson.data = _build_geojson_data()
            # Clear drawn shape but stay in current mode
            clear_preview()
            clear_temps()
            clear_poly_draw()
            state["clicks"] = []
            state["poly"] = []
            add_btn.disabled = True
            clr_btn.disabled = True
            if rect_btn.value:
                set_status(f"Added #{new_idx} — draw next rectangle", "success")
            elif poly_btn.value:
                set_status(f"Added #{new_idx} — draw next polygon", "success")
            else:
                set_status(f"Added — {h_in.value}m (#{new_idx})", "success")
        except Exception as e:
            set_status(f"Error: {str(e)[:30]}", "danger")

    def clear_all(b):
        clear_preview()
        clear_temps()
        state["clicks"] = []
        state["poly"] = []
        add_btn.disabled = True
        clr_btn.disabled = True
        if b:
            set_status("Cleared", "info")

    add_btn.on_click(add_geom)
    clr_btn.on_click(clear_all)

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
    m, gdf = draw_additional_buildings(
        building_gdf=building_gdf,
        initial_center=initial_center,
        zoom=zoom,
        rectangle_vertices=rectangle_vertices,
    )
    display(m)
    return gdf


def draw_additional_trees(voxcity=None, initial_center=None, zoom=17):
    """
    Interactive map editor for trees: add tree points, remove tree points,
    visualise the existing canopy grid (uniform colour), and remove canopy
    cells by clicking or drawing an area polygon.

    Users can:
    - Set tree parameters: top height, bottom height, crown diameter
    - Click multiple times to add multiple trees with the same parameters
    - Update parameters at any time to change subsequent trees
    - View existing canopy grid cells as a uniform-colour overlay
    - Remove individual canopy cells by clicking (Remove / Click mode)
    - Remove trees and canopy cells in bulk by drawing an area polygon

    Args:
        voxcity (VoxCity, optional): A VoxCity object from which to extract
            tree_gdf, rectangle_vertices, building_gdf, and tree_canopy grids.
        initial_center (tuple, optional): (lon, lat) for initial map center.
        zoom (int): Initial zoom level. Default=17.

    Returns:
        tuple: (map_object, updated_tree_gdf, canopy_top, canopy_bottom)
            - map_object: ipyleaflet Map
            - updated_tree_gdf: GeoDataFrame of tree points (mutated in-place)
            - canopy_top: np.ndarray or None — modified canopy top heights
            - canopy_bottom: np.ndarray or None — modified canopy bottom heights

    Examples:
        >>> m, tree_gdf, ct, cb = draw_additional_trees(voxcity=vc)
    """
    # ---------------------------------------------------------
    # Extract data from VoxCity object if provided
    # ---------------------------------------------------------
    tree_gdf = None
    rectangle_vertices = None
    building_gdf = None
    canopy_top = None
    canopy_bottom = None
    if voxcity is not None:
        tree_gdf = voxcity.extras.get('tree_gdf', None)
        rectangle_vertices = voxcity.extras.get('rectangle_vertices', None)
        building_gdf = voxcity.extras.get('building_gdf', None)
        # Extract canopy data for visualization and removal
        if voxcity.tree_canopy is not None and voxcity.tree_canopy.top is not None:
            canopy_top = voxcity.tree_canopy.top.copy()
            if voxcity.tree_canopy.bottom is not None:
                canopy_bottom = voxcity.tree_canopy.bottom.copy()
    
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
    m.layout.height = '600px'
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

    # ── Building footprints overlay ──────────────────────────
    def _build_building_geojson(bld_gdf):
        """Build a GeoJSON FeatureCollection from a building GeoDataFrame."""
        features = []
        if bld_gdf is None or len(bld_gdf) == 0:
            return {"type": "FeatureCollection", "features": features}
        for idx, row in bld_gdf.iterrows():
            if isinstance(row.geometry, geom.Polygon):
                coords = [list(row.geometry.exterior.coords)]
                if any(math.isnan(c) for ring in coords for pt in ring for c in pt):
                    continue
                features.append({
                    "type": "Feature",
                    "id": str(idx),
                    "properties": {},
                    "geometry": {"type": "Polygon", "coordinates": coords},
                })
        return {"type": "FeatureCollection", "features": features}

    _bld_style = {
        "color": "#1565c0",
        "fillColor": "#42a5f5",
        "fillOpacity": 0.35,
        "weight": 1,
    }

    buildings_overlay = GeoJSON(
        data=_build_building_geojson(building_gdf),
        style=_bld_style,
    )
    # Add by default (will be toggled via checkbox)
    _bld_layer_on_map = True
    if building_gdf is not None and len(building_gdf) > 0:
        m.add_layer(buildings_overlay)
    else:
        _bld_layer_on_map = False

    # ── Canopy grid overlay ──────────────────────────────
    _canopy_grid_geom = None
    if canopy_top is not None and rectangle_vertices is not None:
        from .utils import initialize_geod, calculate_distance, normalize_to_one_meter
        from .raster.core import calculate_grid_size

        _v0 = np.array(rectangle_vertices[0])
        _v1 = np.array(rectangle_vertices[1])
        _v3 = np.array(rectangle_vertices[3])
        _side_1 = _v1 - _v0
        _side_2 = _v3 - _v0
        _meshsize = voxcity.tree_canopy.meta.meshsize

        _geod = initialize_geod()
        _dist_1 = calculate_distance(_geod, _v0[0], _v0[1], _v1[0], _v1[1])
        _dist_2 = calculate_distance(_geod, _v0[0], _v0[1], _v3[0], _v3[1])
        _u_vec = normalize_to_one_meter(_side_1, _dist_1)
        _v_vec = normalize_to_one_meter(_side_2, _dist_2)

        _grid_size, _adj_mesh = calculate_grid_size(_side_1, _side_2, _u_vec, _v_vec, _meshsize)

        _canopy_grid_geom = {
            'origin': _v0,
            'side_1': _side_1,
            'side_2': _side_2,
            'u_vec': _u_vec,
            'v_vec': _v_vec,
            'grid_size': _grid_size,
            'adj_mesh': _adj_mesh,
        }

    def _geo_to_cell(lon, lat):
        """Convert geographic (lon, lat) to canopy grid indices (i, j).

        The canopy array has shape (nx, ny) where:
          - first axis  (i) runs along side_1 (v0→v1), scaled by adj_mesh[0]
          - second axis (j) runs along side_2 (v0→v3), scaled by adj_mesh[1]
        """
        if _canopy_grid_geom is None or canopy_top is None:
            return None, None
        origin = _canopy_grid_geom['origin']
        s1 = _canopy_grid_geom['side_1']
        s2 = _canopy_grid_geom['side_2']
        nx, ny = _canopy_grid_geom['grid_size']
        delta = np.array([lon, lat]) - origin
        det = s1[0] * s2[1] - s1[1] * s2[0]
        if abs(det) < 1e-15:
            return None, None
        # alpha = fractional position along side_1 → maps to first axis (i)
        alpha = (delta[0] * s2[1] - delta[1] * s2[0]) / det
        # beta  = fractional position along side_2 → maps to second axis (j)
        beta = (s1[0] * delta[1] - s1[1] * delta[0]) / det
        i = int(math.floor(alpha * nx))
        j = int(math.floor(beta * ny))
        if 0 <= i < canopy_top.shape[0] and 0 <= j < canopy_top.shape[1]:
            return i, j
        return None, None

    def _build_canopy_geojson():
        """Build GeoJSON from non-zero canopy cells (uniform style).

        Uses vectorised numpy to compute all cell corners at once, then
        merges adjacent cells via shapely unary_union to minimise the
        number of GeoJSON features sent to the browser.
        """
        empty = {"type": "FeatureCollection", "features": []}
        if canopy_top is None or _canopy_grid_geom is None:
            return empty
        origin = _canopy_grid_geom['origin']
        u = _canopy_grid_geom['u_vec']
        v = _canopy_grid_geom['v_vec']
        dx = _canopy_grid_geom['adj_mesh'][0]
        dy = _canopy_grid_geom['adj_mesh'][1]

        mask = canopy_top > 0
        if not np.any(mask):
            return empty

        ii, jj = np.nonzero(mask)
        n = len(ii)

        # Vectorised corner computation (all cells at once)
        # bl = origin + i*dx*u + j*dy*v, etc.
        i_f = ii.astype(float)
        j_f = jj.astype(float)
        # shape (n, 2) for each corner
        bl = origin + np.outer(i_f * dx, u) + np.outer(j_f * dy, v)
        br = origin + np.outer((i_f + 1) * dx, u) + np.outer(j_f * dy, v)
        tr = origin + np.outer((i_f + 1) * dx, u) + np.outer((j_f + 1) * dy, v)
        tl = origin + np.outer(i_f * dx, u) + np.outer((j_f + 1) * dy, v)

        # Build shapely boxes and merge adjacent ones
        from shapely.geometry import box as _sbox
        from shapely.ops import unary_union as _uunion
        polys = [
            geom.Polygon([bl[k], br[k], tr[k], tl[k]])
            for k in range(n)
        ]
        merged = _uunion(polys)

        # Convert merged geometry into GeoJSON features
        features = []
        if merged.is_empty:
            return empty

        def _poly_to_feature(poly, fid):
            # Include exterior ring AND interior rings (holes)
            coords = [list(poly.exterior.coords)]
            for interior in poly.interiors:
                coords.append(list(interior.coords))
            return {
                "type": "Feature", "id": str(fid),
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": coords},
            }

        if merged.geom_type == 'Polygon':
            features.append(_poly_to_feature(merged, 0))
        elif merged.geom_type in ('MultiPolygon', 'GeometryCollection'):
            fid = 0
            for part in merged.geoms:
                if part.geom_type == 'Polygon' and not part.is_empty:
                    features.append(_poly_to_feature(part, fid))
                    fid += 1
        return {"type": "FeatureCollection", "features": features}

    _canopy_style = {
        "color": "#00ff7f",
        "fillColor": "#00ff7f",
        "fillOpacity": 0.35,
        "weight": 0.5,
    }

    canopy_overlay = GeoJSON(
        data=_build_canopy_geojson(),
        style=_canopy_style,
    )
    _canopy_on_map = False
    if canopy_top is not None and np.any(canopy_top > 0):
        m.add_layer(canopy_overlay)
        _canopy_on_map = True

    # Display existing trees as circles
    tree_layers = {}
    for idx, row in updated_trees.iterrows():
        if row.geometry is not None and hasattr(row.geometry, 'x'):
            lat = row.geometry.y
            lon = row.geometry.x
            # Ensure integer radius in meters as required by ipyleaflet Circle
            radius_m = max(int(round(float(row.get('crown_diameter', 6.0)) / 2.0)), 1)
            tree_id_val = int(row.get('tree_id', idx+1))
            circle = Circle(location=(lat, lon), radius=radius_m, color='#00ff7f', weight=1, opacity=1.0, fill_color='#00ff7f', fill_opacity=0.3)
            m.add_layer(circle)
            tree_layers[tree_id_val] = circle

    # UI widgets for parameters (Gemini-style)
    style_html = HTML(
        """
    <style>
        /* ── Gemini-style panel (trees) ── */
        .gm-tree-root {
            font-family: 'Google Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
            color: #1f1f1f;
            line-height: 1.5;
        }
        .gm-tree-root * { box-sizing: border-box; }

        .gm-tree-root .gm-title {
            font-size: 13px; font-weight: 500; color: #1f1f1f;
            padding-bottom: 6px;
            border-bottom: 1px solid #e8eaed;
            margin-bottom: 8px;
        }
        .gm-tree-root .gm-label {
            font-size: 11px; font-weight: 500; color: #5f6368;
            letter-spacing: 0.3px;
            margin: 0 0 4px 0;
        }
        .gm-tree-root .gm-sep { height: 1px; background: #e8eaed; margin: 8px 0; }
        .gm-tree-root .gm-hint {
            font-size: 10px; color: #80868b; line-height: 1.4; margin: 0 0 8px 0;
        }
        .gm-tree-root .gm-hover {
            font-size: 11px; color: #1a73e8; font-weight: 500;
            padding: 4px 10px; border-radius: 12px;
            background: #f0f4ff; margin-top: 6px;
            min-height: 0; line-height: 1.3;
        }
        .gm-tree-root .gm-hover:empty { display: none; }

        /* Override ipywidgets buttons */
        .gm-tree-root .jupyter-button,
        .gm-tree-root .jupyter-widgets.widget-toggle-button button {
            border-radius: 18px !important;
            font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            border: 1px solid #dadce0 !important;
            box-shadow: none !important;
            transition: background 0.15s, border-color 0.15s !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            padding: 0 !important;
            line-height: 1 !important;
        }
        .gm-tree-root .jupyter-button:hover,
        .gm-tree-root .jupyter-widgets.widget-toggle-button button:hover {
            box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
        }
        /* ToggleButton container */
        .gm-tree-root .widget-toggle-button {
            height: 26px !important;
        }
        /* neutral */
        .gm-tree-root .jupyter-button:not(.mod-success):not(.mod-danger) {
            background: #f8f9fa !important;
            color: #3c4043 !important;
            border-color: #dadce0 !important;
        }
        .gm-tree-root .jupyter-button:not(.mod-success):not(.mod-danger):hover {
            background: #f1f3f4 !important;
        }
        /* success (Add active) */
        .gm-tree-root .mod-success {
            background: #1a73e8 !important;
            color: #fff !important;
            border-color: #1a73e8 !important;
        }
        .gm-tree-root .mod-success:hover {
            background: #1765cc !important;
            border-color: #1765cc !important;
        }
        /* danger (Remove active) */
        .gm-tree-root .mod-danger {
            background: #c5221f !important;
            color: #fff !important;
            border-color: #c5221f !important;
        }
        .gm-tree-root .mod-danger:hover {
            background: #a8201e !important;
            border-color: #a8201e !important;
        }

        /* inputs */
        .gm-tree-root input[type="number"] {
            border-radius: 8px !important;
            border: 1px solid #dadce0 !important;
            font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif !important;
            font-size: 12px !important;
            padding: 2px 6px !important;
        }
        .gm-tree-root input[type="number"]:focus {
            border-color: #1a73e8 !important;
            outline: none !important;
        }
        .gm-tree-root .widget-label {
            font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif !important;
            font-size: 11px !important;
            color: #5f6368 !important;
            font-weight: 500 !important;
        }
        /* checkbox */
        .gm-tree-root .widget-checkbox { margin-top: 2px; }
        .gm-tree-root .widget-checkbox .widget-label {
            font-size: 10px !important;
            color: #80868b !important;
        }
    </style>
    """
    )

    top_height_input = FloatText(
        value=10.0, description='Top (m):',
        layout=Layout(width='160px', height='24px'),
        style={'description_width': '62px'},
    )
    bottom_height_input = FloatText(
        value=4.0, description='Trunk (m):',
        layout=Layout(width='160px', height='24px'),
        style={'description_width': '62px'},
    )
    crown_diameter_input = FloatText(
        value=6.0, description='Dia. (m):',
        layout=Layout(width='160px', height='24px'),
        style={'description_width': '62px'},
    )
    fixed_prop_checkbox = Checkbox(
        value=True, description='Fixed proportion', indent=False,
        layout=Layout(width='auto', height='20px'),
    )

    add_mode_button = Button(
        description='Add', button_style='success',
        layout=Layout(flex='1', height='26px'),
    )
    remove_mode_button = Button(
        description='Click', button_style='',
        layout=Layout(flex='1', height='26px'),
    )
    area_remove_button = ToggleButton(
        value=False,
        description='Area',
        button_style='danger',
        layout=Layout(flex='1', height='26px'),
        tooltip='Draw polygon to remove trees & canopy inside',
    )

    status_bar = HTML(
        value="<div style='padding:3px 8px;border-radius:10px;font-size:10px;text-align:center;background:#f0f4ff;color:#1a73e8;'>Ready</div>"
    )
    hover_info = HTML("")

    # Layout
    mode_row = HBox(
        [add_mode_button],
        layout=Layout(margin='0', gap='4px'),
    )
    remove_row = HBox(
        [remove_mode_button, area_remove_button],
        layout=Layout(margin='0', gap='4px'),
    )
    param_col = VBox(
        [top_height_input, bottom_height_input, crown_diameter_input, fixed_prop_checkbox],
        layout=Layout(margin='0', gap='0px'),
    )

    panel = VBox(
        [
            style_html,
            HTML("<div class='gm-title' style='margin-bottom:4px;padding-bottom:4px;'>Tree Editor</div>"),
            HTML("<div class='gm-label' style='margin:0 0 2px 0;'>Add</div>"),
            mode_row,
            HTML("<div class='gm-sep' style='margin:4px 0;'></div>"),
            param_col,
            HTML("<div class='gm-sep' style='margin:4px 0;'></div>"),
            HTML("<div class='gm-label' style='margin:0 0 2px 0;'>Remove</div>"),
            remove_row,
            status_bar,
            hover_info,
        ],
        layout=Layout(
            width='200px',
            padding='8px 10px',
            overflow_y='auto',
        ),
    )
    panel.add_class("gm-tree-root")

    card = VBox(
        [panel],
        layout=Layout(
            background_color='white',
            border_radius='16px',
            box_shadow='0 1px 3px rgba(0,0,0,0.1), 0 4px 16px rgba(0,0,0,0.06)',
            overflow='hidden',
        ),
    )

    widget_control = WidgetControl(widget=card, position='topright')
    m.add_control(widget_control)

    # State for mode
    mode = 'add'
    # Area-removal polygon drawing state
    _area_state = {
        'clicks': [],
        'layers': [],
        'preview': None,
    }
    _last_area_click = [0.0]
    _last_mousemove = [0.0]
    _CLOSE_THRESHOLD = 0.0001
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

    def _set_status(msg, stype='info'):
        colors = {
            'info': ('background:#f0f4ff;color:#1a73e8;', ''),
            'success': ('background:#e6f4ea;color:#137333;', ''),
            'danger': ('background:#fce8e6;color:#c5221f;', ''),
            'warn': ('background:#fef7e0;color:#b06000;', ''),
        }
        style_str = colors.get(stype, colors['info'])[0]
        status_bar.value = (
            f"<div style='padding:3px 8px;border-radius:10px;font-size:10px;"
            f"text-align:center;{style_str}'>{msg}</div>"
        )

    def _clear_area_draw():
        """Clear area-removal polygon drawing state."""
        while _area_state['layers']:
            try:
                m.remove_layer(_area_state['layers'].pop())
            except Exception:
                pass
        if _area_state['preview']:
            try:
                m.remove_layer(_area_state['preview'])
            except Exception:
                pass
            _area_state['preview'] = None
        _area_state['clicks'] = []

    def _is_near_first(pts, lon, lat):
        if len(pts) < 3:
            return False
        dx = lon - pts[0][0]
        dy = lat - pts[0][1]
        return (dx * dx + dy * dy) < (_CLOSE_THRESHOLD * _CLOSE_THRESHOLD)

    def _refresh_area_markers():
        while _area_state['layers']:
            try:
                m.remove_layer(_area_state['layers'].pop())
            except Exception:
                pass
        pts = _area_state['clicks']
        for lon_p, lat_p in pts:
            pt = Circle(location=(lat_p, lon_p), radius=2, color='#FF0000',
                        fill_color='#FF0000', fill_opacity=1.0)
            m.add_layer(pt)
            _area_state['layers'].append(pt)
        for i in range(len(pts) - 1):
            line = Polyline(
                locations=[(pts[i][1], pts[i][0]), (pts[i+1][1], pts[i+1][0])],
                color='#FF0000', weight=2,
            )
            m.add_layer(line)
            _area_state['layers'].append(line)

    def _execute_area_removal(polygon_coords):
        """Remove all trees (points + canopy cells) inside the drawn polygon."""
        nonlocal updated_trees
        removal_polygon = geom.Polygon(polygon_coords)
        removed_trees = 0
        removed_cells = 0

        # Remove tree points inside the polygon
        indices_to_drop = []
        for idx2, row2 in updated_trees.iterrows():
            if row2.geometry is not None and hasattr(row2.geometry, 'x'):
                if removal_polygon.contains(row2.geometry):
                    tid = int(row2.get('tree_id', idx2 + 1))
                    layer = tree_layers.get(tid)
                    if layer is not None:
                        try:
                            m.remove_layer(layer)
                        except Exception:
                            pass
                        del tree_layers[tid]
                    indices_to_drop.append(idx2)
                    removed_trees += 1
        if indices_to_drop:
            updated_trees.drop(index=indices_to_drop, inplace=True)
            updated_trees.reset_index(drop=True, inplace=True)

        # Remove canopy cells inside the polygon (vectorised)
        if canopy_top is not None and _canopy_grid_geom is not None:
            origin = _canopy_grid_geom['origin']
            u = _canopy_grid_geom['u_vec']
            v = _canopy_grid_geom['v_vec']
            dx = _canopy_grid_geom['adj_mesh'][0]
            dy = _canopy_grid_geom['adj_mesh'][1]

            nz_mask = canopy_top > 0
            ii_nz, jj_nz = np.nonzero(nz_mask)
            if len(ii_nz) > 0:
                # Vectorised center computation
                centers = (origin
                           + np.outer((ii_nz + 0.5) * dx, u)
                           + np.outer((jj_nz + 0.5) * dy, v))
                # Use prepared geometry for fast containment check
                from shapely.prepared import prep as _prep
                prep_poly = _prep(removal_polygon)
                pts = [geom.Point(c[0], c[1]) for c in centers]
                inside = np.array([prep_poly.contains(p) for p in pts])
                if np.any(inside):
                    removed_cells = int(inside.sum())
                    canopy_top[ii_nz[inside], jj_nz[inside]] = 0
                    if canopy_bottom is not None:
                        canopy_bottom[ii_nz[inside], jj_nz[inside]] = 0
                    canopy_overlay.data = _build_canopy_geojson()

        _clear_area_draw()
        m.double_click_zoom = True
        if removed_trees > 0 or removed_cells > 0:
            parts = []
            if removed_trees:
                parts.append(f"{removed_trees} tree(s)")
            if removed_cells:
                parts.append(f"{removed_cells} cell(s)")
            _set_status(f"Removed {', '.join(parts)}", 'success')
        else:
            _set_status('No trees in selected area', 'warn')

    def set_mode(new_mode):
        nonlocal mode
        mode = new_mode
        _clear_area_draw()
        if new_mode != 'area':
            area_remove_button.value = False
            m.double_click_zoom = True
        # Visual feedback
        add_mode_button.button_style = 'success' if mode == 'add' else ''
        remove_mode_button.button_style = 'danger' if mode == 'remove' else ''
        if mode == 'add':
            _set_status('Click map to add tree', 'info')
        elif mode == 'remove':
            _set_status('Click trees/canopy to remove', 'danger')

    def on_click_add(b):
        set_mode('add')

    def on_click_remove(b):
        set_mode('remove')

    def on_area_toggle(change):
        if change['name'] == 'value':
            if change['new']:
                set_mode('area')
                area_remove_button.value = True
                area_remove_button.button_style = 'danger'
                add_mode_button.button_style = ''
                remove_mode_button.button_style = ''
                m.double_click_zoom = False
                _set_status('Click to draw removal area, click first point to close', 'danger')
            else:
                _clear_area_draw()
                m.double_click_zoom = True
                set_mode('add')

    add_mode_button.on_click(on_click_add)
    remove_mode_button.on_click(on_click_remove)
    area_remove_button.observe(on_area_toggle, names='value')

    # Consecutive placements by map click
    def handle_map_click(**kwargs):
        nonlocal updated_trees

        # ── Area removal polygon drawing ──
        if area_remove_button.value:
            if kwargs.get('type') == 'dblclick':
                pts = _area_state['clicks']
                if len(pts) >= 3:
                    _execute_area_removal(pts)
                else:
                    _set_status('Need at least 3 points', 'warn')
                return
            elif kwargs.get('type') == 'click':
                now = time.time()
                if now - _last_area_click[0] < 0.3:
                    return
                _last_area_click[0] = now
                coords = kwargs.get('coordinates')
                if not coords:
                    return
                lat, lon = coords
                if _is_near_first(_area_state['clicks'], lon, lat):
                    _execute_area_removal(_area_state['clicks'])
                    return
                _area_state['clicks'].append((lon, lat))
                _refresh_area_markers()
                n = len(_area_state['clicks'])
                _set_status(f'{n} point(s) \u2014 click first point to close', 'danger')
            elif kwargs.get('type') == 'mousemove':
                now = time.time()
                if now - _last_mousemove[0] < 0.05:
                    return
                _last_mousemove[0] = now
                pts = _area_state['clicks']
                if pts:
                    coords = kwargs.get('coordinates')
                    if not coords:
                        return
                    lat_c, lon_c = coords
                    if _area_state['preview'] and isinstance(_area_state['preview'], Polyline):
                        _area_state['preview'].locations = [(pts[-1][1], pts[-1][0]), (lat_c, lon_c)]
                    else:
                        if _area_state['preview']:
                            try:
                                m.remove_layer(_area_state['preview'])
                            except Exception:
                                pass
                        line = Polyline(
                            locations=[(pts[-1][1], pts[-1][0]), (lat_c, lon_c)],
                            color='#FF0000', weight=2, dash_array='5, 5',
                        )
                        _area_state['preview'] = line
                        m.add_layer(line)
            return

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
                circle = Circle(location=(lat, lon), radius=radius_m, color='#00ff7f', weight=1, opacity=1.0, fill_color='#00ff7f', fill_opacity=0.3)
                m.add_layer(circle)

                tree_layers[next_tree_id] = circle
                _set_status(f'Added tree #{next_tree_id}', 'success')
            else:
                # Remove mode: find the nearest tree within its crown radius + 5m
                removed_something = False
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
                    removed_something = True
                    _set_status(f'Removed tree #{candidate_id}', 'danger')

                # Also remove canopy cell at click location
                ci, cj = _geo_to_cell(lon, lat)
                if ci is not None and canopy_top is not None and canopy_top[ci, cj] > 0:
                    canopy_top[ci, cj] = 0
                    if canopy_bottom is not None:
                        canopy_bottom[ci, cj] = 0
                    canopy_overlay.data = _build_canopy_geojson()
                    removed_something = True
                    if candidate_id is None:
                        _set_status('Removed canopy cell', 'danger')

                if not removed_something:
                    _set_status('Nothing to remove here', 'warn')
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
                        f"<div class='gm-hover'>"
                        f"#{int(row2.get('tree_id', 0))} &nbsp; Top {float(row2.get('top_height', 10.0))}m &middot; "
                        f"Base {float(row2.get('bottom_height', 0.0))}m &middot; Crown {float(row2.get('crown_diameter', 6.0))}m"
                        f"</div>"
                    )
                    shown = True
                    break
            if not shown:
                hover_info.value = ""
    m.on_interaction(handle_map_click)

    return m, updated_trees, canopy_top, canopy_bottom


def create_tree_editor(tree_gdf=None, initial_center=None, zoom=17, rectangle_vertices=None):
    """
    Convenience wrapper to display the tree editor map and return the GeoDataFrame.
    """
    result = draw_additional_trees(tree_gdf, initial_center, zoom, rectangle_vertices)
    display(result[0])
    return result[1]