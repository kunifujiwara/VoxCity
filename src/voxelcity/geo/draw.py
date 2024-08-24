import math
from pyproj import Proj, transform
from ipyleaflet import Map, DrawControl
import ipyleaflet

def rotate_rectangle(m, rectangle_vertices, angle):
    """
    Project rectangle to Mercator, rotate, and re-project to lat-lon.

    Args:
    rectangle_vertices: List of (lat, lon) tuples
    angle: Rotation angle in degrees

    Returns:
    List of rotated (lat, lon) tuples
    """
    if not rectangle_vertices:
        print("Draw a rectangle first!")
        return

    # Define projections
    wgs84 = Proj(init='epsg:4326')  # WGS84 lat-lon
    mercator = Proj(init='epsg:3857')  # Web Mercator

    # Project to Mercator
    projected_vertices = [transform(wgs84, mercator, lon, lat) for lat, lon in rectangle_vertices]

    # Calculate the centroid
    centroid_x = sum(x for x, y in projected_vertices) / len(projected_vertices)
    centroid_y = sum(y for x, y in projected_vertices) / len(projected_vertices)

    # Convert angle to radians
    angle_rad = -math.radians(angle)

    # Rotate in Mercator projection
    rotated_vertices = []
    for x, y in projected_vertices:
        # Translate point to origin
        temp_x = x - centroid_x
        temp_y = y - centroid_y

        # Rotate point
        rotated_x = temp_x * math.cos(angle_rad) - temp_y * math.sin(angle_rad)
        rotated_y = temp_x * math.sin(angle_rad) + temp_y * math.cos(angle_rad)

        # Translate point back
        new_x = rotated_x + centroid_x
        new_y = rotated_y + centroid_y

        rotated_vertices.append((new_x, new_y))

    # Project back to WGS84
    new_vertices = [transform(mercator, wgs84, x, y) for x, y in rotated_vertices]

    # Convert to (lat, lon) format
    new_vertices = [(lat, lon) for lon, lat in new_vertices]

    # Draw the new rotated rectangle
    polygon = ipyleaflet.Polygon(
        locations=new_vertices,
        color="red",
        fill_color="red"
    )
    m.add_layer(polygon)

    return new_vertices

def draw_rectangle_map(center=(40, -100), zoom=4):
    # Initialize the map
    m = Map(center=center, zoom=zoom)

    # List to hold coordinates of rectangle
    rectangle_vertices = []

    def handle_draw(target, action, geo_json):
        """Handle draw events on the map."""
        # Clear previous vertices
        rectangle_vertices.clear()

        # Check if a rectangle was drawn
        if action == 'created' and geo_json['geometry']['type'] == 'Polygon':
            # Extracting coordinates
            coordinates = geo_json['geometry']['coordinates'][0]
            print("Vertices of the drawn rectangle:")
            for coord in coordinates[:-1]:
                # Append each vertex to the list
                rectangle_vertices.append((coord[1], coord[0]))  # Appending as (latitude, longitude)
                print(f"Longitude: {coord[0]}, Latitude: {coord[1]}")

    # Add drawing controls
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

    # Set the event handler for drawing on the map
    draw_control.on_draw(handle_draw)

    return m, rectangle_vertices