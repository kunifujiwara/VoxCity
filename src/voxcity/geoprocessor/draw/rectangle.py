"""
Interactive rectangle drawing and rotation utilities for geographic maps.

Provides functions for:
- Drawing rectangles on interactive ipyleaflet maps
- Rotating rectangles with proper coordinate transformations
- City-centered map initialization with rectangle drawing
- Fixed-dimension rectangle creation from center points
"""

from __future__ import annotations

import math

from pyproj import Transformer
from ipyleaflet import (
    Map,
    DrawControl,
    Rectangle,
    Polygon as LeafletPolygon,
    Polyline,
)
from geopy import distance

from ..utils import get_coordinates_from_cityname


def rotate_rectangle(m, rectangle_vertices, angle):
    """
    Project rectangle to Mercator, rotate, and re-project to lat-lon coordinates.

    The rotation is performed around the rectangle's centroid using a standard
    2D rotation matrix in Web Mercator space for accurate distance preservation.

    Args:
        m (ipyleaflet.Map): Map object to draw the rotated rectangle on.
        rectangle_vertices (list): List of (lon, lat) tuples defining the rectangle.
        angle (float): Rotation angle in degrees (positive = counter-clockwise).

    Returns:
        list: Rotated (lon, lat) tuples, or None if no vertices provided.
    """
    if not rectangle_vertices:
        print("Draw a rectangle first!")
        return

    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    projected_vertices = [to_merc.transform(lon, lat) for lon, lat in rectangle_vertices]

    centroid_x = sum(x for x, y in projected_vertices) / len(projected_vertices)
    centroid_y = sum(y for x, y in projected_vertices) / len(projected_vertices)

    angle_rad = -math.radians(angle)

    rotated_vertices = []
    for x, y in projected_vertices:
        temp_x = x - centroid_x
        temp_y = y - centroid_y
        rotated_x = temp_x * math.cos(angle_rad) - temp_y * math.sin(angle_rad)
        rotated_y = temp_x * math.sin(angle_rad) + temp_y * math.cos(angle_rad)
        new_x = rotated_x + centroid_x
        new_y = rotated_y + centroid_y
        rotated_vertices.append((new_x, new_y))

    new_vertices = [to_wgs84.transform(x, y) for x, y in rotated_vertices]

    polygon = LeafletPolygon(
        locations=[(lat, lon) for lon, lat in new_vertices],
        color="red",
        fill_color="red",
    )
    m.add_layer(polygon)

    return new_vertices


def draw_rectangle_map(center=(40, -100), zoom=4):
    """
    Create an interactive map for drawing rectangles with ipyleaflet.

    Args:
        center (tuple): Center coordinates (lat, lon). Defaults to (40, -100).
        zoom (int): Initial zoom level. Defaults to 4.

    Returns:
        tuple: (Map object, list that will be populated with (lon, lat) vertices)
    """
    m = Map(center=center, zoom=zoom)
    rectangle_vertices = []

    def handle_draw(target, action, geo_json):
        rectangle_vertices.clear()
        if action == "created" and geo_json["geometry"]["type"] == "Polygon":
            coordinates = geo_json["geometry"]["coordinates"][0]
            print("Vertices of the drawn rectangle:")
            for coord in coordinates[:-1]:
                rectangle_vertices.append((coord[0], coord[1]))
                print(f"Longitude: {coord[0]}, Latitude: {coord[1]}")

    draw_control = DrawControl()
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circle = {}
    draw_control.rectangle = {
        "shapeOptions": {"color": "#6bc2e5", "weight": 4}
    }
    m.add_control(draw_control)
    draw_control.on_draw(handle_draw)

    return m, rectangle_vertices


def draw_rectangle_map_cityname(cityname, zoom=15):
    """
    Create an interactive map centered on a specified city for drawing rectangles.

    Args:
        cityname (str): Name of the city (e.g. "Tokyo, Japan").
        zoom (int): Initial zoom level. Defaults to 15.

    Returns:
        tuple: (Map object, list that will be populated with (lon, lat) vertices)
    """
    center = get_coordinates_from_cityname(cityname)
    m, rectangle_vertices = draw_rectangle_map(center=center, zoom=zoom)
    return m, rectangle_vertices


def _rotate_vertices(base_vertices, angle_deg):
    """Rotate axis-aligned vertices by *angle_deg* around their centroid in Web Mercator.

    Returns a new list of (lon, lat) tuples.
    """
    if angle_deg == 0:
        return list(base_vertices)

    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    projected = [to_merc.transform(lon, lat) for lon, lat in base_vertices]
    cx = sum(x for x, y in projected) / len(projected)
    cy = sum(y for x, y in projected) / len(projected)

    angle_rad = -math.radians(angle_deg)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    rotated_proj = []
    for x, y in projected:
        dx, dy = x - cx, y - cy
        rotated_proj.append((dx * cos_a - dy * sin_a + cx,
                             dx * sin_a + dy * cos_a + cy))

    return [to_wgs84.transform(x, y) for x, y in rotated_proj]


def center_location_map_cityname(cityname, east_west_length, north_south_length, zoom=15, rotation_angle=0):
    """
    Create a map centered on a city where clicking creates a rectangle of specified dimensions.

    After placing a center point, an interactive rotation slider appears so the
    rectangle can be rotated on the map in real time.

    Args:
        cityname (str): Name of the city.
        east_west_length (float): Width of the rectangle in meters.
        north_south_length (float): Height of the rectangle in meters.
        zoom (int): Initial zoom level. Defaults to 15.
        rotation_angle (float): Initial rotation angle in degrees
            (positive = counter-clockwise). Defaults to 0.

    Returns:
        tuple: (widget, list that will be populated with (lon, lat) vertices)
            *widget* is an ipywidgets VBox containing the map and a rotation
            slider.  Display it with ``display(widget)`` or as the last
            expression in a notebook cell.
    """
    import ipywidgets as widgets

    center = get_coordinates_from_cityname(cityname)
    m = Map(center=center, zoom=zoom)
    rectangle_vertices = []
    # Internal state: axis-aligned base vertices (before rotation)
    _state = {"base_vertices": None, "polygon_layer": None, "aux_lines": []}

    rotation_slider = widgets.FloatSlider(
        value=rotation_angle,
        min=-90.0,
        max=90.0,
        step=1.0,
        description="Rotation (°):",
        continuous_update=True,
        style={"description_width": "initial"},
        layout=widgets.Layout(width="100%"),
    )

    def _update_polygon(angle_deg):
        """Recompute rotated vertices, update the map polygon and the output list."""
        base = _state["base_vertices"]
        if base is None:
            return

        # Remove previous polygon and auxiliary lines
        if _state["polygon_layer"] is not None:
            try:
                m.remove_layer(_state["polygon_layer"])
            except Exception:
                pass
        for line in _state["aux_lines"]:
            try:
                m.remove_layer(line)
            except Exception:
                pass
        _state["aux_lines"] = []

        vertices = _rotate_vertices(base, angle_deg)

        rectangle_vertices.clear()
        rectangle_vertices.extend(vertices)

        poly = LeafletPolygon(
            locations=[(lat, lon) for lon, lat in vertices],
            color="red",
            fill_color="red",
            fill_opacity=0.2,
        )
        _state["polygon_layer"] = poly
        m.add_layer(poly)

        # Draw auxiliary crosshair lines through midpoints of opposite sides
        # vertices order: [v0, v1, v2, v3]
        v0, v1, v2, v3 = vertices
        mid_01 = ((v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2)  # midpoint left side
        mid_32 = ((v3[0] + v2[0]) / 2, (v3[1] + v2[1]) / 2)  # midpoint right side
        mid_03 = ((v0[0] + v3[0]) / 2, (v0[1] + v3[1]) / 2)  # midpoint bottom side
        mid_12 = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)  # midpoint top side

        for start, end in [(mid_01, mid_32), (mid_03, mid_12)]:
            line = Polyline(
                locations=[(start[1], start[0]), (end[1], end[0])],
                color="red",
                weight=1,
                opacity=0.6,
                dash_array="6 4",
            )
            _state["aux_lines"].append(line)
            m.add_layer(line)

    def handle_draw(target, action, geo_json):
        if action == "created" and geo_json["geometry"]["type"] == "Point":
            lon, lat = geo_json["geometry"]["coordinates"][0], geo_json["geometry"]["coordinates"][1]
            print(f"Point drawn at Longitude: {lon}, Latitude: {lat}")

            north = distance.distance(meters=north_south_length / 2).destination((lat, lon), bearing=0)
            south = distance.distance(meters=north_south_length / 2).destination((lat, lon), bearing=180)
            east_at_south = distance.distance(meters=east_west_length / 2).destination(
                (south.latitude, lon), bearing=90,
            )
            west_at_south = distance.distance(meters=east_west_length / 2).destination(
                (south.latitude, lon), bearing=270,
            )

            _state["base_vertices"] = [
                (west_at_south.longitude, south.latitude),
                (west_at_south.longitude, north.latitude),
                (east_at_south.longitude, north.latitude),
                (east_at_south.longitude, south.latitude),
            ]

            _update_polygon(rotation_slider.value)

            print(f"Rectangle vertices (rotation={rotation_slider.value}°):")
            for vertex in rectangle_vertices:
                print(f"Longitude: {vertex[0]}, Latitude: {vertex[1]}")

    draw_control = DrawControl()
    draw_control.polyline = {}
    draw_control.polygon = {}
    draw_control.circle = {}
    draw_control.rectangle = {}
    draw_control.marker = {}
    m.add_control(draw_control)
    draw_control.on_draw(handle_draw)

    def _on_slider_change(change):
        _update_polygon(change["new"])

    rotation_slider.observe(_on_slider_change, names="value")

    ui = widgets.VBox([m, rotation_slider])
    return ui, rectangle_vertices
