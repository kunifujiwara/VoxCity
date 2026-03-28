"""
Interactive building editor for ipyleaflet maps.

Provides:
- edit_building: Full interactive editor (add rectangles/polygons, delete buildings)
- create_building_editor: Convenience wrapper that displays the map
"""

from __future__ import annotations

import math
import time

import numpy as np
import geopandas as gpd
import shapely.geometry as geom
from pyproj import Transformer
from ipyleaflet import (
    Map,
    DrawControl,
    Polygon as LeafletPolygon,
    Polyline,
    WidgetControl,
    Circle,
    GeoJSON,
)
from ipywidgets import (
    VBox,
    HBox,
    Button,
    FloatText,
    HTML,
    Checkbox,
    ToggleButton,
    Layout,
    Dropdown,
)
from IPython.display import display

from ._common import (
    create_basemap_tiles,
    compute_grid_geometry,
    build_building_geojson,
    build_canopy_geojson,
    build_lc_geojson,
    lc_style_callback,
    generate_editor_css,
    is_near_first,
    make_layer_toggle,
    make_status_setter,
)

# Import VoxCity for type checking
try:
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from ...models import VoxCity
except ImportError:
    pass


def edit_building(
    voxcity=None,
    building_gdf=None,
    initial_center=None,
    zoom=17,
    rectangle_vertices=None,
):
    """
    Interactive map editor: Draw rectangles, freehand polygons, and DELETE existing buildings.

    Args:
        voxcity (VoxCity, optional): VoxCity object to extract data from.
        building_gdf (GeoDataFrame, optional): Existing buildings.
        initial_center (tuple, optional): (lon, lat) map center.
        zoom (int): Initial zoom level. Default=17.
        rectangle_vertices (list, optional): Rectangle corner coordinates.

    Returns:
        tuple: (Map, updated_gdf)
    """
    # --- Data Initialization ---
    tree_gdf = None
    land_cover = None
    land_cover_source = None
    canopy_top = None
    if voxcity is not None:
        if building_gdf is None:
            building_gdf = voxcity.extras.get("building_gdf", None)
        if rectangle_vertices is None:
            rectangle_vertices = voxcity.extras.get("rectangle_vertices", None)
        tree_gdf = voxcity.extras.get("tree_gdf", None)
        if voxcity.tree_canopy is not None and voxcity.tree_canopy.top is not None:
            canopy_top = voxcity.tree_canopy.top.copy()
        if voxcity.land_cover is not None and voxcity.land_cover.classes is not None:
            land_cover = voxcity.land_cover.classes.copy()
        land_cover_source = voxcity.extras.get("land_cover_source", None)
        if land_cover_source is None:
            selected = voxcity.extras.get("selected_sources", {})
            land_cover_source = selected.get("land_cover_source", "OpenStreetMap")

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

    # --- Map Setup ---
    if initial_center is not None:
        center_lon, center_lat = initial_center
    elif not updated_gdf.empty:
        b = updated_gdf.total_bounds
        center_lon, center_lat = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    else:
        center_lon, center_lat = -100.0, 40.0

    m = Map(center=(center_lat, center_lon), zoom=zoom, scroll_wheel_zoom=True)
    m.layout.height = "600px"

    # Basemap
    _basemap_tiles = create_basemap_tiles()
    _current_basemap = [_basemap_tiles["Google Satellite"]]
    m.layers = (_current_basemap[0],)

    for ctrl in list(m.controls):
        if isinstance(ctrl, DrawControl):
            m.remove_control(ctrl)

    # --- Cross-layer overlays ---
    _overlay_layers = {}

    # Tree canopy overlay
    _canopy_grid_geom = None
    if canopy_top is not None and rectangle_vertices is not None and voxcity is not None:
        _canopy_grid_geom = compute_grid_geometry(
            rectangle_vertices, voxcity.tree_canopy.meta.meshsize,
        )

    if canopy_top is not None and _canopy_grid_geom is not None:
        _tree_overlay = GeoJSON(
            data=build_canopy_geojson(canopy_top, _canopy_grid_geom),
            style={"color": "#00ff7f", "fillColor": "#00ff7f", "fillOpacity": 0.35, "weight": 0.5},
        )
        _overlay_layers["Trees"] = [_tree_overlay, False]

    # Land cover overlay
    _lc_grid_geom = None
    if land_cover is not None and rectangle_vertices is not None and voxcity is not None:
        _lc_grid_geom = compute_grid_geometry(
            rectangle_vertices, voxcity.land_cover.meta.meshsize,
        )

    if land_cover is not None and _lc_grid_geom is not None:
        _lc_overlay = GeoJSON(
            data=build_lc_geojson(land_cover, _lc_grid_geom, land_cover_source),
            style_callback=lc_style_callback,
        )
        _overlay_layers["Land Cover"] = [_lc_overlay, False]

    # --- UI Setup ---
    style_html = HTML(generate_editor_css("gm-root"))

    add_label = HTML("<div class='gm-label'>Add</div>")

    rect_btn = ToggleButton(
        value=False, description="Rectangle", icon="",
        layout=Layout(width="92px", height="30px"),
        tooltip="Click 3 corners on map to draw rectangle",
    )
    poly_btn = ToggleButton(
        value=False, description="Polygon", icon="",
        layout=Layout(width="82px", height="30px"),
        tooltip="Click to draw polygon, double-click to finish",
    )

    h_in = FloatText(
        value=10.0, description="Height",
        layout=Layout(width="115px", height="28px"),
        style={"description_width": "42px"},
    )
    mh_in = FloatText(
        value=0.0, description="Base",
        layout=Layout(width="100px", height="28px"),
        style={"description_width": "34px"},
    )
    add_btn = Button(
        description="Add", button_style="primary", icon="plus",
        disabled=True, layout=Layout(flex="1", height="32px"),
    )
    clr_btn = Button(
        description="Clear", button_style="", icon="",
        disabled=True, layout=Layout(width="64px", height="32px"),
        tooltip="Clear drawing",
    )

    sep = HTML("<div class='gm-sep'></div>")
    remove_label = HTML("<div class='gm-label'>Remove</div>")

    del_btn = ToggleButton(
        value=False, description="Click", icon="", button_style="danger",
        layout=Layout(width="72px", height="30px"),
        tooltip="Click on buildings to remove",
    )
    poly_del_btn = ToggleButton(
        value=False, description="Area", icon="", button_style="danger",
        layout=Layout(width="68px", height="30px"),
        tooltip="Draw polygon to remove buildings inside",
    )

    status_bar = HTML(value="<div class='gm-status gm-status-info'>Ready</div>")

    basemap_dropdown = Dropdown(
        options=list(_basemap_tiles.keys()),
        value="Google Satellite",
        layout=Layout(width="140px", height="26px"),
    )

    def _on_basemap_change(change):
        name = change["new"]
        new_tile = _basemap_tiles[name]
        layers = list(m.layers)
        layers[0] = new_tile
        m.layers = tuple(layers)
        _current_basemap[0] = new_tile

    basemap_dropdown.observe(_on_basemap_change, names="value")

    # Layer toggle checkboxes
    _layer_checkboxes = []
    for _lyr_name, (_lyr_obj, _lyr_on) in _overlay_layers.items():
        cb = Checkbox(
            value=_lyr_on, description=_lyr_name, indent=False,
            layout=Layout(width="auto", height="20px"),
        )
        cb.observe(make_layer_toggle(m, _overlay_layers, _lyr_name, _lyr_obj), names="value")
        _layer_checkboxes.append(cb)

    _layer_widgets = []
    if _layer_checkboxes or True:
        _layer_widgets.append(HTML("<div class='gm-sep'></div>"))
        _layer_widgets.append(HTML("<div class='gm-label'>Basemap</div>"))
        _layer_widgets.append(basemap_dropdown)
    if _layer_checkboxes:
        _layer_widgets.append(HTML("<div class='gm-label' style='margin-top:6px;'>Layers</div>"))
        _layer_widgets.extend(_layer_checkboxes)

    # Layout
    add_tools_row = HBox([rect_btn, poly_btn], layout=Layout(margin="0 0 6px 0", align_items="center", gap="6px"))
    input_row = HBox([h_in, mh_in], layout=Layout(margin="0 0 6px 0"))
    action_row = HBox([add_btn, clr_btn], layout=Layout(margin="0", gap="6px"))
    remove_tools_row = HBox([del_btn, poly_del_btn], layout=Layout(margin="0", gap="6px"))

    panel = VBox(
        [
            style_html,
            HTML("<div class='gm-title'>Building Editor</div>"),
            add_label, add_tools_row, input_row, action_row,
            sep, remove_label, remove_tools_row, status_bar,
        ] + _layer_widgets,
        layout=Layout(width="260px", padding="14px 16px"),
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

    # --- Global State & Transformers ---
    state = {
        "poly": [], "clicks": [], "temp_layers": [], "preview": None,
        "removal_poly": None, "removal_preview": None, "removal_clicks": [],
        "removal_layers": [], "poly_clicks": [], "poly_layers": [], "poly_preview": None,
    }
    to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    to_geo = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    _style_attr = getattr(m, "default_style", {})
    original_style = _style_attr.copy() if isinstance(_style_attr, dict) else {"cursor": "grab"}

    set_status = make_status_setter(status_bar, use_gm_class=True)

    # --- Building GeoJSON layer ---
    def _build_geojson_data():
        return build_building_geojson(updated_gdf, include_height=True)

    _geojson_style = {
        "color": "#2196F3", "fillColor": "#2196F3", "fillOpacity": 0.4, "weight": 1,
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

    # --- Helpers ---
    def clear_removal_preview():
        if state["removal_preview"]:
            try:
                m.remove_layer(state["removal_preview"])
            except Exception:
                pass
            state["removal_preview"] = None
        state["removal_poly"] = None
        while state["removal_layers"]:
            try:
                m.remove_layer(state["removal_layers"].pop())
            except Exception:
                pass
        state["removal_clicks"] = []

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

    def clear_preview():
        if state["preview"]:
            try:
                m.remove_layer(state["preview"])
            except Exception:
                pass
            state["preview"] = None

    def clear_poly_draw():
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
            pt = Circle(location=(lat, lon), radius=2, color="red", fill_color="red", fill_opacity=1.0)
            m.add_layer(pt)
            state["temp_layers"].append(pt)
        if len(state["clicks"]) >= 2:
            (l1, la1), (l2, la2) = state["clicks"][0], state["clicks"][1]
            line = Polyline(locations=[(la1, l1), (la2, l2)], color="red", weight=3)
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
            (x1, y1), (x2, y2), (x2 + hx, y2 + hy), (x1 + hx, y1 + hy),
        ]
        return [to_geo.transform(*p) for p in corners_merc], None

    _last_mousemove = [0.0]
    _last_removal_click = [0.0]
    _last_poly_click = [0.0]

    _CLOSE_THRESHOLD = 0.0001

    def _refresh_poly_markers():
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
            line = Polyline(locations=[(pts[i][1], pts[i][0]), (pts[i + 1][1], pts[i + 1][0])], color="#4CAF50", weight=2)
            m.add_layer(line)
            state["poly_layers"].append(line)

    def _finish_poly_footprint(pts):
        clear_poly_draw()
        state["poly"] = list(pts)
        poly_locs = [(lat, lon) for lon, lat in pts]
        preview = LeafletPolygon(
            locations=poly_locs, color="#4CAF50",
            fill_color="#4CAF50", fill_opacity=0.3,
        )
        state["preview"] = preview
        m.add_layer(preview)
        add_btn.disabled = False
        clr_btn.disabled = False
        set_status("Shape ready \u2014 set height and +Add", "success")

    def _execute_polygon_removal(polygon_coords):
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
            _set_drawing_cursor()
            m.double_click_zoom = False
            set_status(f"Removed {removed_count} \u2014 draw next area or deselect", "success")
        else:
            clear_removal_preview()
            set_status("No buildings in selected area", "warn")

    def _refresh_removal_markers():
        while state["removal_layers"]:
            try:
                m.remove_layer(state["removal_layers"].pop())
            except Exception:
                pass
        pts = state["removal_clicks"]
        for lon, lat in pts:
            pt = Circle(location=(lat, lon), radius=2, color="#FF0000", fill_color="#FF0000", fill_opacity=1.0)
            m.add_layer(pt)
            state["removal_layers"].append(pt)
        for i in range(len(pts) - 1):
            line = Polyline(locations=[(pts[i][1], pts[i][0]), (pts[i + 1][1], pts[i + 1][0])], color="#FF0000", weight=2)
            m.add_layer(line)
            state["removal_layers"].append(line)

    # --- Mode change logic ---
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

    # --- Map interaction ---
    def handle_map_interaction(**kwargs):
        # Polygon drawing for adding
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
                if is_near_first(state["poly_clicks"], lon, lat):
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

        # Polygon drawing for area removal
        if poly_del_btn.value:
            if kwargs.get("type") == "dblclick":
                pts = state["removal_clicks"]
                if len(pts) >= 3:
                    _execute_polygon_removal(pts)
                else:
                    set_status("Need at least 3 points", "warn")
                return
            elif kwargs.get("type") == "click":
                now = time.time()
                if now - _last_removal_click[0] < 0.3:
                    return
                _last_removal_click[0] = now
                coords = kwargs.get("coordinates")
                if not coords:
                    return
                lat, lon = coords
                if is_near_first(state["removal_clicks"], lon, lat):
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
                clear_preview()
                set_status("Step 2/3 \u2014 Click second corner", "info")
            elif count == 2:
                (l1, la1), (l2, la2) = state["clicks"]
                x1, y1 = to_merc.transform(l1, la1)
                x2, y2 = to_merc.transform(l2, la2)
                if math.hypot(x2 - x1, y2 - y1) < 0.5:
                    state["clicks"].pop()
                    refresh_markers()
                    set_status("Too close \u2014 click further away", "warn")
                else:
                    clear_preview()
                    set_status("Step 3/3 \u2014 Click opposite side", "info")
            elif count == 3:
                verts, err = build_rect(state["clicks"])
                if err:
                    state["clicks"].pop()
                    set_status(f"{err} \u2014 try again", "warn")
                else:
                    clear_preview()
                    clear_temps()
                    state["poly"] = verts
                    poly_locs = [(lat, lon) for lon, lat in verts]
                    preview = LeafletPolygon(
                        locations=poly_locs, color="#4CAF50",
                        fill_color="#4CAF50", fill_opacity=0.3,
                    )
                    state["preview"] = preview
                    m.add_layer(preview)
                    add_btn.disabled = False
                    clr_btn.disabled = False
                    state["clicks"] = []
                    set_status("Shape ready \u2014 set height and +Add", "success")

        elif kwargs.get("type") == "mousemove":
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
                if state["preview"] and isinstance(state["preview"], Polyline):
                    state["preview"].locations = new_locs
                else:
                    clear_preview()
                    line = Polyline(
                        locations=new_locs, color="#FF5722",
                        weight=2, dash_array="5, 5",
                    )
                    state["preview"] = line
                    m.add_layer(line)
            elif len(state["clicks"]) == 2:
                tentative_clicks = state["clicks"] + [(lon_c, lat_c)]
                verts, err = build_rect(tentative_clicks)
                if not err:
                    poly_locs = [(lat, lon) for lon, lat in verts]
                    if state["preview"] and isinstance(state["preview"], LeafletPolygon):
                        state["preview"].locations = poly_locs
                    else:
                        clear_preview()
                        poly = LeafletPolygon(
                            locations=poly_locs, color="#FF5722", weight=2,
                            fill_color="#FF5722", fill_opacity=0.1, dash_array="5, 5",
                        )
                        state["preview"] = poly
                        m.add_layer(poly)

    m.on_interaction(handle_map_interaction)

    def handle_freehand(self, action, geo_json):
        if action == "created" and geo_json["geometry"]["type"] == "Polygon":
            coords = geo_json["geometry"]["coordinates"][0]
            polygon_coords = [(c[0], c[1]) for c in coords[:-1]]
            rect_btn.value = False
            del_btn.value = False
            poly_del_btn.value = False
            state["clicks"] = []
            clear_preview()
            clear_temps()
            state["poly"] = polygon_coords
            add_btn.disabled = False
            clr_btn.disabled = False
            set_status("Shape ready \u2014 set height and add", "success")

    draw_control = DrawControl(
        polygon={"shapeOptions": {"color": "#FF5722", "fillColor": "#FF5722", "fillOpacity": 0.2}},
        rectangle={}, circle={}, polyline={}, marker={}, circlemarker={},
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
            clear_preview()
            clear_temps()
            clear_poly_draw()
            state["clicks"] = []
            state["poly"] = []
            add_btn.disabled = True
            clr_btn.disabled = True
            if rect_btn.value:
                set_status(f"Added #{new_idx} \u2014 draw next rectangle", "success")
            elif poly_btn.value:
                set_status(f"Added #{new_idx} \u2014 draw next polygon", "success")
            else:
                set_status(f"Added \u2014 {h_in.value}m (#{new_idx})", "success")
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


def create_building_editor(building_gdf=None, initial_center=None, zoom=17, rectangle_vertices=None):
    """
    Creates and displays an interactive building editor.

    Returns:
        GeoDataFrame: Automatically-updating building GeoDataFrame.
    """
    m, gdf = edit_building(
        building_gdf=building_gdf,
        initial_center=initial_center,
        zoom=zoom,
        rectangle_vertices=rectangle_vertices,
    )
    display(m)
    return gdf
