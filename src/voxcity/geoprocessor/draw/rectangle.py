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
    Circle,
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


def _build_rect(clicks, to_merc, to_geo):
    """Build a rectangle from 3 clicked points in Mercator space.

    Args:
        clicks: List of 3 (lon, lat) tuples.
        to_merc: Transformer from WGS84 to Web Mercator.
        to_geo: Transformer from Web Mercator to WGS84.

    Returns:
        tuple: (list of 4 (lon, lat) corners, error_string or None)
    """
    (lon1, lat1), (lon2, lat2), (lon3, lat3) = clicks[:3]
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
        (x1, y1), (x2, y2), (x2 + hx, y2 + hy), (x1 + hx, y1 + hy),
    ]
    verts = [to_geo.transform(*p) for p in corners_merc]
    return _normalize_rect_vertices(verts, to_merc), None


def _normalize_rect_vertices(verts, to_merc):
    """Reorder rectangle vertices to match the SW→NW→NE→SE convention.

    The pipeline expects:
      [0]=SW, [1]=NW, [2]=NE, [3]=SE
    so that the v0→v1 edge points roughly northward (rotation_angle ∈ [-90, 90]).

    The algorithm:
    1. Pick the edge (among the 4 consecutive edges) whose bearing from north
       is in [-90, 90).  This becomes the v0→v1 edge.
    2. Ensure v3 is to the RIGHT of v0→v1 (i.e. the eastward/SE side).
       If not, use the opposite parallel northward edge instead.
    """
    proj = [to_merc.transform(lon, lat) for lon, lat in verts]

    # Find the edge whose bearing (atan2(dx, dy)) is in [-90, 90)
    best_i = 0
    best_angle = 999
    for i in range(4):
        j = (i + 1) % 4
        dx = proj[j][0] - proj[i][0]
        dy = proj[j][1] - proj[i][1]
        angle = math.degrees(math.atan2(dx, dy))
        if -90 <= angle < 90 and abs(angle) < abs(best_angle):
            best_i = i
            best_angle = angle

    # Rotate list so that best_i becomes index 0
    ordered = [verts[(best_i + k) % 4] for k in range(4)]
    proj_ordered = [proj[(best_i + k) % 4] for k in range(4)]

    # Check winding: v3 must be to the RIGHT of v0→v1 (i.e. the east/SE side).
    # In standard 2D (x=east, y=north), "right of" = negative cross product.
    # cross(v0→v1, v0→v3) < 0  ⟹  v3 is to the right  ⟹  correct.
    # cross(v0→v1, v0→v3) > 0  ⟹  v3 is to the left   ⟹  need to fix.
    dx01 = proj_ordered[1][0] - proj_ordered[0][0]
    dy01 = proj_ordered[1][1] - proj_ordered[0][1]
    dx03 = proj_ordered[3][0] - proj_ordered[0][0]
    dy03 = proj_ordered[3][1] - proj_ordered[0][1]
    cross = dx01 * dy03 - dy01 * dx03

    if cross > 0:
        # v3 is on the wrong (left/west) side.  Use the opposite parallel
        # northward edge as v0→v1 by fully reversing the vertex list.
        ordered = [ordered[3], ordered[2], ordered[1], ordered[0]]

    return ordered


def draw_rectangle_map(center=(40, -100), zoom=4):
    """
    Create an interactive map for drawing rectangles with two modes.

    - **Aligned** mode: click two opposite corners to draw an axis-aligned rectangle.
    - **Rotated** mode: click 3 points to draw a rotated rectangle.

    Drawing starts immediately when a mode button is selected.

    Args:
        center (tuple): Center coordinates (lat, lon). Defaults to (40, -100).
        zoom (int): Initial zoom level. Defaults to 4.

    Returns:
        tuple: (ipywidgets.VBox, list that will be populated with (lon, lat) vertices)
    """
    import time as _time
    import ipywidgets as widgets

    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True)
    rectangle_vertices = []

    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    state = {
        "clicks": [], "temp_layers": [], "preview": None,
        "polygon_layer": None, "aux_lines": [],
        "mode": "aligned",  # "aligned" or "rotated"
    }
    _last_mousemove = [0.0]

    # --- Mode buttons ---
    aligned_btn = widgets.ToggleButton(
        value=True, description="Aligned", icon="",
        layout=widgets.Layout(width="100px", height="30px"),
        tooltip="Draw an axis-aligned rectangle (2 clicks: opposite corners)",
    )
    rotated_btn = widgets.ToggleButton(
        value=False, description="Rotated", icon="",
        layout=widgets.Layout(width="100px", height="30px"),
        tooltip="Draw a rotated rectangle (3 clicks)",
    )
    status_label = widgets.HTML(
        value="<i>Click two opposite corners to draw a rectangle</i>",
    )

    # --- Helpers ---
    def _clear_temps():
        while state["temp_layers"]:
            try:
                m.remove_layer(state["temp_layers"].pop())
            except Exception:
                pass

    def _clear_preview():
        if state["preview"]:
            try:
                m.remove_layer(state["preview"])
            except Exception:
                pass
            state["preview"] = None

    def _clear_polygon():
        if state["polygon_layer"]:
            try:
                m.remove_layer(state["polygon_layer"])
            except Exception:
                pass
            state["polygon_layer"] = None
        for line in state["aux_lines"]:
            try:
                m.remove_layer(line)
            except Exception:
                pass
        state["aux_lines"] = []

    def _clear_all_drawing():
        _clear_temps()
        _clear_preview()
        _clear_polygon()
        state["clicks"] = []

    def _refresh_markers():
        _clear_temps()
        for lon, lat in state["clicks"]:
            pt = Circle(location=(lat, lon), radius=2, color="red", fill_color="red", fill_opacity=1.0)
            m.add_layer(pt)
            state["temp_layers"].append(pt)
        if len(state["clicks"]) >= 2 and state["mode"] == "rotated":
            (l1, la1), (l2, la2) = state["clicks"][0], state["clicks"][1]
            line = Polyline(locations=[(la1, l1), (la2, l2)], color="red", weight=3)
            m.add_layer(line)
            state["temp_layers"].append(line)

    def _draw_polygon_with_crosshair(verts, color="red"):
        _clear_polygon()
        poly = LeafletPolygon(
            locations=[(lat, lon) for lon, lat in verts],
            color=color,
            fill_color=color,
            fill_opacity=0.2,
        )
        state["polygon_layer"] = poly
        m.add_layer(poly)

        v0, v1, v2, v3 = verts
        mid_01 = ((v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2)
        mid_32 = ((v3[0] + v2[0]) / 2, (v3[1] + v2[1]) / 2)
        mid_03 = ((v0[0] + v3[0]) / 2, (v0[1] + v3[1]) / 2)
        mid_12 = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)

        for start, end in [(mid_01, mid_32), (mid_03, mid_12)]:
            line = Polyline(
                locations=[(start[1], start[0]), (end[1], end[0])],
                color=color,
                weight=1,
                opacity=0.6,
                dash_array="6 4",
            )
            state["aux_lines"].append(line)
            m.add_layer(line)

    # --- Interaction handler ---
    def _handle_interaction(**kwargs):
        if kwargs.get("type") == "click":
            coords = kwargs.get("coordinates")
            if not coords:
                return
            lat, lon = coords
            state["clicks"].append((lon, lat))
            _refresh_markers()

            if state["mode"] == "aligned":
                count = len(state["clicks"])
                if count == 1:
                    _clear_preview()
                    _clear_polygon()
                    rectangle_vertices.clear()
                    status_label.value = "<i>Click 1/2 — click the opposite corner</i>"
                elif count >= 2:
                    (lon1, lat1), (lon2, lat2) = state["clicks"][0], state["clicks"][1]
                    # Always produce SW→NW→NE→SE order
                    west = min(lon1, lon2)
                    east = max(lon1, lon2)
                    south = min(lat1, lat2)
                    north = max(lat1, lat2)
                    verts = [
                        (west, south),   # SW
                        (west, north),   # NW
                        (east, north),   # NE
                        (east, south),   # SE
                    ]
                    _clear_preview()
                    _clear_temps()
                    _draw_polygon_with_crosshair(verts)
                    rectangle_vertices.clear()
                    rectangle_vertices.extend(verts)
                    state["clicks"] = []
                    status_label.value = "<i>Rectangle drawn! Click again to redraw.</i>"
                    print("Rectangle drawn! Vertices:")
                    for v in verts:
                        print(f"  Longitude: {v[0]}, Latitude: {v[1]}")

            elif state["mode"] == "rotated":
                count = len(state["clicks"])
                if count == 1:
                    _clear_preview()
                    _clear_polygon()
                    rectangle_vertices.clear()
                    status_label.value = "<i>Click 1/3 — click the second corner</i>"
                elif count == 2:
                    (l1, la1), (l2, la2) = state["clicks"]
                    x1, y1 = to_merc.transform(l1, la1)
                    x2, y2 = to_merc.transform(l2, la2)
                    if math.hypot(x2 - x1, y2 - y1) < 0.5:
                        state["clicks"].pop()
                        _refresh_markers()
                        status_label.value = "<i>Too close — click further away</i>"
                    else:
                        _clear_preview()
                        status_label.value = "<i>Click 2/3 — click the opposite side to set depth</i>"
                elif count == 3:
                    verts, err = _build_rect(state["clicks"], to_merc, to_geo)
                    if err:
                        state["clicks"].pop()
                        status_label.value = f"<i>{err} — try again</i>"
                    else:
                        _clear_preview()
                        _clear_temps()
                        _draw_polygon_with_crosshair(verts)
                        rectangle_vertices.clear()
                        rectangle_vertices.extend(verts)
                        state["clicks"] = []
                        status_label.value = "<i>Rotated rectangle drawn! Click again to redraw.</i>"
                        print("Rotated rectangle drawn! Vertices:")
                        for v in verts:
                            print(f"  Longitude: {v[0]}, Latitude: {v[1]}")

        elif kwargs.get("type") == "mousemove":
            now = _time.time()
            if now - _last_mousemove[0] < 0.05:
                return
            _last_mousemove[0] = now
            coords = kwargs.get("coordinates")
            if not coords:
                return
            lat_c, lon_c = coords

            if state["mode"] == "aligned" and len(state["clicks"]) == 1:
                (lon1, lat1) = state["clicks"][0]
                preview_verts = [
                    (lon1, lat1), (lon1, lat_c), (lon_c, lat_c), (lon_c, lat1),
                ]
                poly_locs = [(lat, lon) for lon, lat in preview_verts]
                if state["preview"] and isinstance(state["preview"], LeafletPolygon):
                    state["preview"].locations = poly_locs
                else:
                    _clear_preview()
                    poly = LeafletPolygon(
                        locations=poly_locs, color="#6bc2e5", weight=2,
                        fill_color="#6bc2e5", fill_opacity=0.1, dash_array="5, 5",
                    )
                    state["preview"] = poly
                    m.add_layer(poly)

            elif state["mode"] == "rotated":
                if len(state["clicks"]) == 1:
                    (lon1, lat1) = state["clicks"][0]
                    new_locs = [(lat1, lon1), (lat_c, lon_c)]
                    if state["preview"] and isinstance(state["preview"], Polyline):
                        state["preview"].locations = new_locs
                    else:
                        _clear_preview()
                        line = Polyline(
                            locations=new_locs, color="#6bc2e5",
                            weight=2, dash_array="5, 5",
                        )
                        state["preview"] = line
                        m.add_layer(line)
                elif len(state["clicks"]) == 2:
                    tentative = state["clicks"] + [(lon_c, lat_c)]
                    verts, err = _build_rect(tentative, to_merc, to_geo)
                    if not err:
                        poly_locs = [(lat, lon) for lon, lat in verts]
                        if state["preview"] and isinstance(state["preview"], LeafletPolygon):
                            state["preview"].locations = poly_locs
                        else:
                            _clear_preview()
                            poly = LeafletPolygon(
                                locations=poly_locs, color="#6bc2e5", weight=2,
                                fill_color="#6bc2e5", fill_opacity=0.1, dash_array="5, 5",
                            )
                            state["preview"] = poly
                            m.add_layer(poly)

    m.on_interaction(_handle_interaction)

    # --- Mode switching ---
    def _activate_aligned_mode():
        state["mode"] = "aligned"
        _clear_all_drawing()
        m.default_style = {"cursor": "crosshair"}
        status_label.value = "<i>Click two opposite corners to draw a rectangle</i>"

    def _activate_rotated_mode():
        state["mode"] = "rotated"
        _clear_all_drawing()
        m.default_style = {"cursor": "crosshair"}
        status_label.value = "<i>Click 3 points to draw a rotated rectangle</i>"

    def _on_aligned_btn(change):
        if change["new"]:
            rotated_btn.value = False
            _activate_aligned_mode()
        elif not rotated_btn.value:
            aligned_btn.value = True

    def _on_rotated_btn(change):
        if change["new"]:
            aligned_btn.value = False
            _activate_rotated_mode()
        elif not aligned_btn.value:
            rotated_btn.value = True

    aligned_btn.observe(_on_aligned_btn, names="value")
    rotated_btn.observe(_on_rotated_btn, names="value")

    # Set initial cursor
    m.default_style = {"cursor": "crosshair"}

    toolbar = widgets.HBox(
        [aligned_btn, rotated_btn, status_label],
        layout=widgets.Layout(align_items="center", gap="8px"),
    )
    ui = widgets.VBox([toolbar, m])
    return ui, rectangle_vertices


def draw_rectangle_map_cityname(cityname, zoom=15):
    """
    Create an interactive map centered on a specified city for drawing rectangles.

    Two modes are available via toggle buttons:

    - **Aligned**: click two opposite corners (axis-aligned rectangle).
    - **Rotated**: click 3 points to draw a rotated rectangle.

    Args:
        cityname (str): Name of the city (e.g. "Tokyo, Japan").
        zoom (int): Initial zoom level. Defaults to 15.

    Returns:
        tuple: (ipywidgets.VBox, list that will be populated with (lon, lat) vertices)
    """
    center = get_coordinates_from_cityname(cityname)
    ui, rectangle_vertices = draw_rectangle_map(center=center, zoom=zoom)
    return ui, rectangle_vertices


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


def rectangle_map(center=(40, -100), zoom=15, width=500, height=500, cityname=""):
    """Create an interactive map with a side panel for drawing rectangles.

    Three drawing modes are provided via the panel:

    - **Aligned**: click two opposite corners for an axis-aligned rectangle.
    - **Rotated**: click 3 points to draw a rotated rectangle.
    - **Fixed**: click a center point to place a rectangle with the specified
      *width* (east-west) and *height* (north-south), then adjust the rotation
      angle with a slider.

    The panel includes a city-name field that re-centres the map on submission.

    Args:
        center (tuple): Map center as (lat, lon). Defaults to (40, -100).
        zoom (int): Initial zoom level. Defaults to 15.
        width (float): Initial east-west length in metres (for Fixed mode).
        height (float): Initial north-south length in metres (for Fixed mode).
        cityname (str): Pre-filled city name for the location field.

    Returns:
        tuple: (ipywidgets.VBox, list that will be populated with (lon, lat) vertices)
    """
    import time as _time
    from ipywidgets import (
        VBox, HBox, Layout, HTML, ToggleButton, FloatText, FloatSlider,
        Button, Text,
    )
    from ipyleaflet import WidgetControl, TileLayer
    from ._common import generate_editor_css, make_status_setter

    _carto = TileLayer(
        url="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}.png",
        attribution='&copy; <a href="https://carto.com/">CARTO</a>',
        name="CartoDB Positron",
    )
    m = Map(center=center, zoom=zoom, scroll_wheel_zoom=True, basemap=_carto)
    m.layout.height = "600px"
    rectangle_vertices = []

    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    state = {
        "clicks": [], "temp_layers": [], "preview": None,
        "polygon_layer": None, "aux_lines": [],
        "mode": "aligned",
        "base_vertices": None,  # for fixed mode
    }
    _last_mousemove = [0.0]

    # ── Panel widgets ──────────────────────────────────────────
    style_html = HTML(generate_editor_css("gm-root"))

    city_in = Text(
        value=cityname, placeholder="e.g. Tokyo",
        layout=Layout(width="150px"),
    )
    city_go_btn = Button(
        description="Go", button_style="primary",
        layout=Layout(width="46px"),
    )

    aligned_btn = ToggleButton(
        value=True, description="Aligned", icon="",
        layout=Layout(width="78px", height="30px"),
        tooltip="2 clicks: opposite corners",
    )
    rotated_btn = ToggleButton(
        value=False, description="Rotated", icon="",
        layout=Layout(width="78px", height="30px"),
        tooltip="3 clicks: edge + depth",
    )
    fixed_btn = ToggleButton(
        value=False, description="Fixed", icon="",
        layout=Layout(width="62px", height="30px"),
        tooltip="Click center, set width/height",
    )

    width_in = FloatText(
        value=width,
        layout=Layout(width="100px"),
    )
    height_in = FloatText(
        value=height,
        layout=Layout(width="100px"),
    )
    rotation_slider = FloatSlider(
        value=0, min=-90.0, max=90.0, step=0.1,
        readout_format=".1f",
        continuous_update=True,
        style={"description_width": "0px"},
        layout=Layout(width="100%"),
    )

    reset_btn = Button(
        description="Reset", button_style="",
        layout=Layout(width="70px", height="30px"),
        tooltip="Clear current drawing",
    )

    status_bar = HTML(
        value="<div class='gm-status gm-status-info'>Click two opposite corners</div>"
    )

    set_status = make_status_setter(status_bar, use_gm_class=True)

    # ── Helpers ────────────────────────────────────────────────
    def _clear_temps():
        while state["temp_layers"]:
            try:
                m.remove_layer(state["temp_layers"].pop())
            except Exception:
                pass

    def _clear_preview():
        if state["preview"]:
            try:
                m.remove_layer(state["preview"])
            except Exception:
                pass
            state["preview"] = None

    def _clear_polygon():
        if state["polygon_layer"]:
            try:
                m.remove_layer(state["polygon_layer"])
            except Exception:
                pass
            state["polygon_layer"] = None
        for line in state["aux_lines"]:
            try:
                m.remove_layer(line)
            except Exception:
                pass
        state["aux_lines"] = []

    def _clear_all():
        _clear_temps()
        _clear_preview()
        _clear_polygon()
        state["clicks"] = []
        state["base_vertices"] = None

    def _refresh_markers():
        _clear_temps()
        for lon, lat in state["clicks"]:
            pt = Circle(location=(lat, lon), radius=2, color="red",
                        fill_color="red", fill_opacity=1.0)
            m.add_layer(pt)
            state["temp_layers"].append(pt)
        if len(state["clicks"]) >= 2 and state["mode"] == "rotated":
            (l1, la1), (l2, la2) = state["clicks"][0], state["clicks"][1]
            line = Polyline(locations=[(la1, l1), (la2, l2)], color="red", weight=3)
            m.add_layer(line)
            state["temp_layers"].append(line)

    def _draw_polygon_with_crosshair(verts, color="red"):
        _clear_polygon()
        poly = LeafletPolygon(
            locations=[(lat, lon) for lon, lat in verts],
            color=color, fill_color=color, fill_opacity=0.2,
        )
        state["polygon_layer"] = poly
        m.add_layer(poly)

        v0, v1, v2, v3 = verts
        mid_01 = ((v0[0] + v1[0]) / 2, (v0[1] + v1[1]) / 2)
        mid_32 = ((v3[0] + v2[0]) / 2, (v3[1] + v2[1]) / 2)
        mid_03 = ((v0[0] + v3[0]) / 2, (v0[1] + v3[1]) / 2)
        mid_12 = ((v1[0] + v2[0]) / 2, (v1[1] + v2[1]) / 2)
        for start, end in [(mid_01, mid_32), (mid_03, mid_12)]:
            line = Polyline(
                locations=[(start[1], start[0]), (end[1], end[0])],
                color=color, weight=1, opacity=0.6, dash_array="6 4",
            )
            state["aux_lines"].append(line)
            m.add_layer(line)

    def _finish_rect(verts, label="Rectangle"):
        _clear_preview()
        _clear_temps()
        _draw_polygon_with_crosshair(verts)
        rectangle_vertices.clear()
        rectangle_vertices.extend(verts)
        state["clicks"] = []
        set_status(f"{label} drawn!", "success")

    # ── Fixed mode helpers ─────────────────────────────────────
    def _build_fixed(lon, lat, ew, ns):
        """Build axis-aligned base vertices from a center point and dimensions."""
        north = distance.distance(meters=ns / 2).destination((lat, lon), bearing=0)
        south = distance.distance(meters=ns / 2).destination((lat, lon), bearing=180)
        east_at_south = distance.distance(meters=ew / 2).destination(
            (south.latitude, lon), bearing=90)
        west_at_south = distance.distance(meters=ew / 2).destination(
            (south.latitude, lon), bearing=270)
        return [
            (west_at_south.longitude, south.latitude),
            (west_at_south.longitude, north.latitude),
            (east_at_south.longitude, north.latitude),
            (east_at_south.longitude, south.latitude),
        ]

    def _update_fixed_polygon():
        base = state["base_vertices"]
        if base is None:
            return
        angle = rotation_slider.value
        verts = _rotate_vertices(base, angle)
        _clear_polygon()
        _draw_polygon_with_crosshair(verts)
        rectangle_vertices.clear()
        rectangle_vertices.extend(verts)

    # ── Interaction handler ────────────────────────────────────
    def _handle_interaction(**kwargs):
        if kwargs.get("type") == "click":
            coords = kwargs.get("coordinates")
            if not coords:
                return
            lat, lon = coords
            mode = state["mode"]

            if mode == "aligned":
                state["clicks"].append((lon, lat))
                _refresh_markers()
                count = len(state["clicks"])
                if count == 1:
                    _clear_preview()
                    _clear_polygon()
                    rectangle_vertices.clear()
                    set_status("Click 1/2 — click the opposite corner", "info")
                elif count >= 2:
                    (lon1, lat1), (lon2, lat2) = state["clicks"][0], state["clicks"][1]
                    west, east = min(lon1, lon2), max(lon1, lon2)
                    south, north = min(lat1, lat2), max(lat1, lat2)
                    verts = [(west, south), (west, north), (east, north), (east, south)]
                    _finish_rect(verts, "Aligned rectangle")

            elif mode == "rotated":
                state["clicks"].append((lon, lat))
                _refresh_markers()
                count = len(state["clicks"])
                if count == 1:
                    _clear_preview()
                    _clear_polygon()
                    rectangle_vertices.clear()
                    set_status("Click 1/3 — click the second corner", "info")
                elif count == 2:
                    (l1, la1), (l2, la2) = state["clicks"]
                    x1, y1 = to_merc.transform(l1, la1)
                    x2, y2 = to_merc.transform(l2, la2)
                    if math.hypot(x2 - x1, y2 - y1) < 0.5:
                        state["clicks"].pop()
                        _refresh_markers()
                        set_status("Too close — click further away", "warn")
                    else:
                        _clear_preview()
                        set_status("Click 2/3 — click opposite side for depth", "info")
                elif count == 3:
                    verts, err = _build_rect(state["clicks"], to_merc, to_geo)
                    if err:
                        state["clicks"].pop()
                        set_status(f"{err} — try again", "warn")
                    else:
                        _finish_rect(verts, "Rotated rectangle")

            elif mode == "fixed":
                _clear_all()
                ew = width_in.value
                ns = height_in.value
                if ew <= 0 or ns <= 0:
                    set_status("Width and height must be > 0", "warn")
                    return
                state["base_vertices"] = _build_fixed(lon, lat, ew, ns)
                rotation_slider.value = 0
                _update_fixed_polygon()
                set_status("Fixed rectangle placed! Adjust angle.", "success")

        elif kwargs.get("type") == "mousemove":
            now = _time.time()
            if now - _last_mousemove[0] < 0.05:
                return
            _last_mousemove[0] = now
            coords = kwargs.get("coordinates")
            if not coords:
                return
            lat_c, lon_c = coords
            mode = state["mode"]

            if mode == "aligned" and len(state["clicks"]) == 1:
                (lon1, lat1) = state["clicks"][0]
                preview_verts = [(lon1, lat1), (lon1, lat_c), (lon_c, lat_c), (lon_c, lat1)]
                poly_locs = [(la, lo) for lo, la in preview_verts]
                if state["preview"] and isinstance(state["preview"], LeafletPolygon):
                    state["preview"].locations = poly_locs
                else:
                    _clear_preview()
                    poly = LeafletPolygon(
                        locations=poly_locs, color="#6bc2e5", weight=2,
                        fill_color="#6bc2e5", fill_opacity=0.1, dash_array="5, 5",
                    )
                    state["preview"] = poly
                    m.add_layer(poly)

            elif mode == "rotated":
                if len(state["clicks"]) == 1:
                    (lon1, lat1) = state["clicks"][0]
                    new_locs = [(lat1, lon1), (lat_c, lon_c)]
                    if state["preview"] and isinstance(state["preview"], Polyline):
                        state["preview"].locations = new_locs
                    else:
                        _clear_preview()
                        line = Polyline(
                            locations=new_locs, color="#6bc2e5",
                            weight=2, dash_array="5, 5",
                        )
                        state["preview"] = line
                        m.add_layer(line)
                elif len(state["clicks"]) == 2:
                    tentative = state["clicks"] + [(lon_c, lat_c)]
                    verts, err = _build_rect(tentative, to_merc, to_geo)
                    if not err:
                        poly_locs = [(la, lo) for lo, la in verts]
                        if state["preview"] and isinstance(state["preview"], LeafletPolygon):
                            state["preview"].locations = poly_locs
                        else:
                            _clear_preview()
                            poly = LeafletPolygon(
                                locations=poly_locs, color="#6bc2e5", weight=2,
                                fill_color="#6bc2e5", fill_opacity=0.1, dash_array="5, 5",
                            )
                            state["preview"] = poly
                            m.add_layer(poly)

    m.on_interaction(_handle_interaction)

    # ── Mode switching ─────────────────────────────────────────
    _mode_buttons = [aligned_btn, rotated_btn, fixed_btn]
    _mode_names = ["aligned", "rotated", "fixed"]
    _mode_hints = {
        "aligned": "Click two opposite corners",
        "rotated": "Click 3 points: edge + depth",
        "fixed": "Click to place center point",
    }

    def _activate_mode(name):
        state["mode"] = name
        _clear_all()
        m.default_style = {"cursor": "crosshair"}
        m.add_class("drawing-mode")
        set_status(_mode_hints[name], "info")
        # Show/hide fixed-mode widgets
        _fixed_visible = "block" if name == "fixed" else "none"
        dim_section.layout.display = _fixed_visible
        angle_section.layout.display = _fixed_visible

    def _make_observer(idx):
        def obs(change):
            if change["new"]:
                for j, btn in enumerate(_mode_buttons):
                    if j != idx:
                        btn.value = False
                _activate_mode(_mode_names[idx])
            elif not any(b.value for b in _mode_buttons):
                _mode_buttons[idx].value = True
        return obs

    for i, btn in enumerate(_mode_buttons):
        btn.observe(_make_observer(i), names="value")

    def _on_reset(_):
        _clear_all()
        rectangle_vertices.clear()
        set_status(_mode_hints[state["mode"]], "info")
    reset_btn.on_click(_on_reset)

    def _on_slider_change(change):
        if state["mode"] == "fixed":
            _update_fixed_polygon()
    rotation_slider.observe(_on_slider_change, names="value")

    def _on_dim_change(_):
        if state["mode"] == "fixed" and state["base_vertices"] is not None:
            # Recompute base vertices from the centroid of the current base
            base = state["base_vertices"]
            cx = sum(v[0] for v in base) / 4
            cy = sum(v[1] for v in base) / 4
            state["base_vertices"] = _build_fixed(cx, cy, width_in.value, height_in.value)
            _update_fixed_polygon()
    width_in.observe(_on_dim_change, names="value")
    height_in.observe(_on_dim_change, names="value")

    def _on_city_go(_):
        name = city_in.value.strip()
        if not name:
            return
        try:
            new_center = get_coordinates_from_cityname(name)
            m.center = new_center
            m.zoom = zoom
            set_status(f"Moved to {name}", "success")
        except Exception:
            set_status("City not found", "warn")
    city_go_btn.on_click(_on_city_go)
    city_in.on_submit(lambda _: _on_city_go(None))

    # ── Grouped sections (show/hide as units) ─────────────────
    city_row = HBox(
        [city_in, city_go_btn],
        layout=Layout(align_items="center", gap="4px"),
    )
    mode_row = HBox(
        [aligned_btn, rotated_btn, fixed_btn],
        layout=Layout(align_items="center", gap="4px"),
    )
    dim_row = HBox(
        [width_in, height_in],
        layout=Layout(align_items="center", gap="6px"),
    )
    dim_section = VBox(
        [
            HTML("<div class='gm-sep'></div>"),
            HTML("<div class='gm-label'>Dimensions</div>"),
            dim_row,
        ],
        layout=Layout(overflow="hidden"),
    )
    angle_section = VBox(
        [
            HTML("<div class='gm-label' style='margin-top:4px;'>Angle</div>"),
            rotation_slider,
        ],
        layout=Layout(overflow="hidden"),
    )

    # ── Initial state: hide fixed widgets ──────────────────────
    dim_section.layout.display = "none"
    angle_section.layout.display = "none"
    m.default_style = {"cursor": "crosshair"}
    m.add_class("drawing-mode")

    # ── Panel layout ──────────────────────────────────────────
    panel = VBox(
        [
            style_html,
            HTML("<div class='gm-title' style='margin-bottom:8px;padding-bottom:6px;'>"
                 "Rectangle Tool</div>"),
            HTML("<div class='gm-label'>Location</div>"),
            city_row,
            HTML("<div class='gm-sep'></div>"),
            HTML("<div class='gm-label'>Mode</div>"),
            mode_row,
            dim_section,
            angle_section,
            HTML("<div class='gm-sep'></div>"),
            HBox([reset_btn], layout=Layout(margin="0", gap="6px")),
            status_bar,
        ],
        layout=Layout(width="250px", padding="10px 12px", overflow="hidden"),
    )
    panel.add_class("gm-root")

    card = VBox(
        [panel],
        layout=Layout(
            background_color="white", border_radius="16px",
            box_shadow="0 1px 3px rgba(0,0,0,0.1), 0 4px 16px rgba(0,0,0,0.06)",
            overflow="hidden",
        ),
    )
    m.add_control(WidgetControl(widget=card, position="topright"))

    ui = VBox([m])
    return ui, rectangle_vertices


def create_rectangle_map(cityname=None, zoom=15, width=500, height=500):
    """Create an interactive rectangle-drawing map with a side panel.

    Combines aligned / rotated drawing and fixed-dimension placement into a
    single map.  If *cityname* is provided the map is centred on that city
    and the location field is pre-filled; otherwise a world-level view is
    shown and the user can type a city name into the panel.

    Args:
        cityname (str, optional): Name of the city (e.g. ``"Tokyo"``).
        zoom (int): Initial zoom level. Defaults to 15.
        width (float): Initial east-west length in metres (for Fixed mode).
        height (float): Initial north-south length in metres (for Fixed mode).

    Returns:
        tuple: (ipywidgets.VBox, list that will be populated with (lon, lat)
            vertices in SW → NW → NE → SE order)
    """
    if cityname is not None:
        center = get_coordinates_from_cityname(cityname)
    else:
        center = (40, -100)
        cityname = ""
    return rectangle_map(center=center, zoom=zoom, width=width, height=height,
                         cityname=cityname)
