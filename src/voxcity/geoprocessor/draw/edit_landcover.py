"""
Interactive land-cover editor for ipyleaflet maps.

Provides:
- edit_landcover: Full interactive editor (paint cells by click or area polygon)
"""

from __future__ import annotations

import math
import time

import numpy as np
import shapely.geometry as geom
from ipyleaflet import (
    Map,
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
    HTML,
    Checkbox,
    ToggleButton,
    Layout,
    Dropdown,
)

from ._common import (
    LC_COLORS_BY_NAME,
    create_basemap_tiles,
    compute_grid_geometry,
    build_building_geojson,
    build_canopy_geojson,
    build_lc_geojson,
    build_lc_legend_html,
    get_lc_source_colors,
    lc_style_callback,
    generate_editor_css,
    geo_to_cell,
    is_near_first,
    clear_area_draw,
    refresh_area_markers,
    handle_area_mousemove,
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


def edit_landcover(voxcity=None, initial_center=None, zoom=17):
    """
    Interactive map editor for land-cover classes.

    Users can select a land-cover class from the palette and paint individual
    cells by clicking or in bulk by drawing an area polygon.

    Args:
        voxcity (VoxCity, optional): VoxCity object for data extraction.
        initial_center (tuple, optional): (lon, lat) for initial map centre.
        zoom (int): Initial zoom level. Default=17.

    Returns:
        tuple: (map_object, land_cover_classes np.ndarray)
    """
    # --- Data extraction ---
    land_cover = None
    rectangle_vertices = None
    building_gdf = None
    land_cover_source = None
    canopy_top_lc = None
    if voxcity is not None:
        if voxcity.land_cover is not None and voxcity.land_cover.classes is not None:
            land_cover = voxcity.land_cover.classes.copy()
        rectangle_vertices = voxcity.extras.get("rectangle_vertices", None)
        building_gdf = voxcity.extras.get("building_gdf", None)
        land_cover_source = voxcity.extras.get("land_cover_source", None)
        if land_cover_source is None:
            selected = voxcity.extras.get("selected_sources", {})
            land_cover_source = selected.get("land_cover_source", "OpenStreetMap")
        if voxcity.tree_canopy is not None and voxcity.tree_canopy.top is not None:
            canopy_top_lc = voxcity.tree_canopy.top.copy()

    if land_cover is None:
        raise ValueError("VoxCity object must contain a land_cover grid.")

    # --- Class lookup ---
    from ...utils.lc import get_land_cover_classes

    _src_classes = get_land_cover_classes(land_cover_source)
    _class_names = list(dict.fromkeys(_src_classes.values()))
    _LC_NAMES = {i: name for i, name in enumerate(_class_names)}
    _name_to_hex = get_lc_source_colors(land_cover_source)
    _LC_COLORS = {i: _name_to_hex.get(name, "#808080") for i, name in enumerate(_class_names)}
    _num_classes = len(_class_names)
    _NON_EDITABLE_NAMES = {"Tree", "Tree Canopy", "Trees", "Building", "Built Area", "Built-up", "Built"}
    _EDITABLE_CLASSES = [i for i, name in enumerate(_class_names) if name not in _NON_EDITABLE_NAMES]

    # --- Grid geometry ---
    _lc_grid_geom = None
    if rectangle_vertices is not None and voxcity is not None:
        _lc_grid_geom = compute_grid_geometry(rectangle_vertices, voxcity.land_cover.meta.meshsize)

    # --- GeoJSON builders ---
    def _build_lc_geojson_local():
        return build_lc_geojson(land_cover, _lc_grid_geom, land_cover_source)

    # --- Map centre ---
    if initial_center is not None:
        center_lon, center_lat = initial_center
    elif rectangle_vertices is not None:
        center_lon = (rectangle_vertices[0][0] + rectangle_vertices[2][0]) / 2
        center_lat = (rectangle_vertices[0][1] + rectangle_vertices[2][1]) / 2
    else:
        center_lon, center_lat = -100.0, 40.0

    # --- Create map ---
    m = Map(center=(center_lat, center_lon), zoom=zoom, scroll_wheel_zoom=True)
    m.layout.height = "600px"

    _basemap_tiles = create_basemap_tiles()
    _current_basemap = [_basemap_tiles["Google Satellite"]]
    m.layers = tuple([_current_basemap[0]])

    # Rectangle outline
    if rectangle_vertices is not None and len(rectangle_vertices) >= 4:
        try:
            lat_lon_coords = [(lat, lon) for lon, lat in rectangle_vertices]
            rect_outline = LeafletPolygon(
                locations=lat_lon_coords,
                color="#fed766", weight=2,
                fill_color="#fed766", fill_opacity=0.0,
            )
            m.add_layer(rect_outline)
        except Exception:
            pass

    # --- Overlays ---
    _overlay_layers = {}

    buildings_overlay = GeoJSON(
        data=build_building_geojson(building_gdf),
        style={"color": "#1565c0", "fillColor": "#42a5f5", "fillOpacity": 0.35, "weight": 1},
    )
    _overlay_layers["Buildings"] = [buildings_overlay, False]

    # Tree / canopy overlay
    _canopy_grid_geom_lc = None
    if canopy_top_lc is not None and rectangle_vertices is not None and voxcity is not None:
        _canopy_grid_geom_lc = compute_grid_geometry(rectangle_vertices, voxcity.tree_canopy.meta.meshsize)

    if canopy_top_lc is not None and _canopy_grid_geom_lc is not None:
        _tree_overlay = GeoJSON(
            data=build_canopy_geojson(canopy_top_lc, _canopy_grid_geom_lc),
            style={"color": "#00ff7f", "fillColor": "#00ff7f", "fillOpacity": 0.35, "weight": 0.5},
        )
        _overlay_layers["Trees"] = [_tree_overlay, False]

    # Land-cover overlay (always visible)
    lc_overlay = GeoJSON(data=_build_lc_geojson_local(), style_callback=lc_style_callback)
    m.add_layer(lc_overlay)

    # --- UI ---
    # Generate per-class CSS for colored palette buttons
    # add_class() puts the class on the <button> element itself, so use
    # .jupyter-button.gm-lc-cN (same element) not > (child combinator).
    _palette_css_parts = []
    for _cls in range(len(_class_names)):
        _c = _LC_COLORS.get(_cls, "#808080")
        _r, _g, _b = int(_c[1:3], 16), int(_c[3:5], 16), int(_c[5:7], 16)
        _lum = 0.299 * _r + 0.587 * _g + 0.114 * _b
        _tc = "#fff" if _lum < 140 else "#1f1f1f"
        _palette_css_parts.append(
            f"    .gm-lc-root .jupyter-button.gm-lc-c{_cls}:not(#_) {{"
            f" background:{_c} !important;color:{_tc} !important;"
            f" border-color:rgba(0,0,0,0.18) !important;}}"
        )
    # Selection ring for colored buttons
    _palette_css_parts.append(
        "    .gm-lc-root .jupyter-button[class*='gm-lc-c'].mod-success:not(#_)"
        " { outline:2px solid #1a73e8;outline-offset:-2px; }"
    )
    _palette_css_parts.append(
        "    .gm-lc-root .jupyter-button[class*='gm-lc-c']:hover:not(#_)"
        " { filter:brightness(0.92) !important; }"
    )
    _palette_extra_css = "\n".join(_palette_css_parts)

    style_html = HTML(generate_editor_css(
        "gm-lc-root",
        drawing_mode_class="lc-drawing-mode",
        delete_mode_class="lc-delete-mode",
        extra_css=_palette_extra_css,
    ))

    _selected_class = [_EDITABLE_CLASSES[0]]

    class_buttons = []
    for cls_code in _EDITABLE_CLASSES:
        name = _LC_NAMES.get(cls_code, "?")
        btn = Button(
            description=name,
            layout=Layout(width="auto", height="20px", padding="0 4px"),
            button_style="success" if cls_code == _selected_class[0] else "",
        )
        btn._lc_code = cls_code
        btn.add_class(f"gm-lc-c{cls_code}")
        class_buttons.append(btn)

    def _on_class_button(b):
        _selected_class[0] = b._lc_code
        for cb in class_buttons:
            cb.button_style = "success" if cb._lc_code == _selected_class[0] else ""
        _set_status(f"Selected: {_LC_NAMES.get(b._lc_code, '?')}", "info")

    for btn in class_buttons:
        btn.on_click(_on_class_button)

    _class_rows = []
    for i in range(0, len(class_buttons), 3):
        row = class_buttons[i : i + 3]
        _class_rows.append(HBox(row, layout=Layout(gap="3px", margin="1px 0")))
    class_palette = VBox(_class_rows, layout=Layout(margin="0", gap="0"))

    paint_click_button = Button(description="Click", button_style="success", layout=Layout(flex="1", height="26px"))
    paint_area_button = ToggleButton(
        value=False, description="Area", button_style="",
        layout=Layout(flex="1", height="26px"), tooltip="Draw polygon to paint area",
    )

    status_bar = HTML(
        value="<div style='padding:3px 8px;border-radius:10px;font-size:10px;"
              "text-align:center;background:#f0f4ff;color:#1a73e8;'>Ready</div>"
    )
    hover_info = HTML("")
    _set_status = make_status_setter(status_bar, use_gm_class=False)

    paint_row = HBox([paint_click_button, paint_area_button], layout=Layout(margin="0", gap="4px"))

    # Basemap switcher
    basemap_dropdown = Dropdown(
        options=list(_basemap_tiles.keys()), value="Google Satellite",
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

    # Layer toggles
    _layer_checkboxes = []
    for _lyr_name, (_lyr_obj, _lyr_on) in _overlay_layers.items():
        cb = Checkbox(
            value=_lyr_on, description=_lyr_name, indent=False,
            layout=Layout(width="auto", height="20px"),
        )
        cb.observe(make_layer_toggle(m, _overlay_layers, _lyr_name, _lyr_obj,
                                       front_layers=[lc_overlay]), names="value")
        _layer_checkboxes.append(cb)

    _layer_widgets = [
        HTML("<div class='gm-sep' style='margin:4px 0;'></div>"),
        HTML("<div class='gm-label' style='margin:0 0 2px 0;'>Basemap</div>"),
        basemap_dropdown,
    ]
    if _layer_checkboxes:
        _layer_widgets.append(HTML("<div class='gm-label' style='margin:4px 0 2px 0;'>Layers</div>"))
        _layer_widgets.extend(_layer_checkboxes)

    panel = VBox(
        [
            style_html,
            HTML("<div class='gm-title' style='margin-bottom:4px;padding-bottom:4px;'>Land Cover Editor</div>"),
            HTML("<div class='gm-label' style='margin:0 0 2px 0;'>Class</div>"),
            class_palette,
            HTML("<div class='gm-sep' style='margin:4px 0;'></div>"),
            HTML("<div class='gm-label' style='margin:0 0 2px 0;'>Paint</div>"),
            paint_row,
            status_bar,
            hover_info,
        ] + _layer_widgets,
        layout=Layout(width="220px", padding="8px 10px"),
    )
    panel.add_class("gm-lc-root")

    card = VBox(
        [panel],
        layout=Layout(
            background_color="white", border_radius="16px",
            box_shadow="0 1px 3px rgba(0,0,0,0.1), 0 4px 16px rgba(0,0,0,0.06)",
            overflow="hidden",
        ),
    )
    m.add_control(WidgetControl(widget=card, position="topright"))

    # --- State ---
    mode = "paint_click"
    _area_state = {"clicks": [], "layers": [], "preview": None}
    _last_area_click = [0.0]
    _last_mousemove = [0.0]
    _toggling = [False]

    def _refresh_lc_overlay():
        lc_overlay.data = _build_lc_geojson_local()

    def _paint_cell(lon, lat):
        ci, cj = geo_to_cell(lon, lat, _lc_grid_geom, land_cover.shape)
        if ci is None:
            _set_status("Outside grid", "warn")
            return
        cls_val = _selected_class[0]
        land_cover[ci, cj] = cls_val
        _refresh_lc_overlay()
        name = _LC_NAMES.get(cls_val, "?")
        _set_status(f"Painted ({ci},{cj}) \u2192 {name}", "success")

    def _execute_area_paint(polygon_coords):
        from matplotlib.path import Path as _MplPath

        cls_val = _selected_class[0]
        painted = 0
        if _lc_grid_geom is not None:
            origin = _lc_grid_geom["origin"]
            u = _lc_grid_geom["u_vec"]
            v = _lc_grid_geom["v_vec"]
            dx = _lc_grid_geom["adj_mesh"][0]
            dy = _lc_grid_geom["adj_mesh"][1]
            nx, ny = land_cover.shape

            ii_all = np.arange(nx)
            jj_all = np.arange(ny)
            ii_grid, jj_grid = np.meshgrid(ii_all, jj_all, indexing="ij")
            ii_flat = ii_grid.ravel()
            jj_flat = jj_grid.ravel()

            centers = origin + np.outer((ii_flat + 0.5) * dx, u) + np.outer((jj_flat + 0.5) * dy, v)
            mpl_path = _MplPath(polygon_coords)
            inside = mpl_path.contains_points(centers)
            if np.any(inside):
                painted = int(inside.sum())
                land_cover[ii_flat[inside], jj_flat[inside]] = cls_val
                _refresh_lc_overlay()

        clear_area_draw(m, _area_state)
        m.double_click_zoom = True
        name = _LC_NAMES.get(cls_val, "?")
        if painted > 0:
            _set_status(f"Painted {painted} cell(s) \u2192 {name}", "success")
        else:
            _set_status("No cells in drawn area", "warn")

    # --- Mode management ---
    def _update_button_styles():
        paint_click_button.button_style = "success" if mode == "paint_click" else ""
        paint_area_button.button_style = "success" if mode == "paint_area" else ""
        paint_area_button.value = (mode == "paint_area")

    def set_mode(new_mode):
        nonlocal mode
        mode = new_mode
        clear_area_draw(m, _area_state)
        if new_mode == "paint_area":
            m.double_click_zoom = False
            m.default_style = {"cursor": "crosshair"}
            m.add_class("lc-drawing-mode")
        else:
            m.double_click_zoom = True
            m.default_style = {"cursor": "grab"}
            m.remove_class("lc-drawing-mode")
        _toggling[0] = True
        _update_button_styles()
        _toggling[0] = False
        if mode == "paint_click":
            _set_status("Click map to paint cell", "info")
        elif mode == "paint_area":
            _set_status("Draw polygon to paint area", "info")

    def on_click_paint_click(b):
        set_mode("paint_click")

    def on_paint_area_toggle(change):
        if _toggling[0] or change["name"] != "value":
            return
        _toggling[0] = True
        set_mode("paint_area" if change["new"] else "paint_click")
        _toggling[0] = False

    paint_click_button.on_click(on_click_paint_click)
    paint_area_button.observe(on_paint_area_toggle, names="value")

    # --- Map interaction ---
    def handle_map_click(**kwargs):
        # Area polygon drawing
        if mode == "paint_area":
            _area_color = "#1a73e8"
            if kwargs.get("type") == "dblclick":
                pts = _area_state["clicks"]
                if len(pts) >= 3:
                    _execute_area_paint(pts)
                else:
                    _set_status("Need at least 3 points", "warn")
                return
            elif kwargs.get("type") == "click":
                now = time.time()
                if now - _last_area_click[0] < 0.3:
                    return
                _last_area_click[0] = now
                coords = kwargs.get("coordinates")
                if not coords:
                    return
                lat, lon = coords
                if is_near_first(_area_state["clicks"], lon, lat):
                    _execute_area_paint(_area_state["clicks"])
                    return
                _area_state["clicks"].append((lon, lat))
                refresh_area_markers(m, _area_state, _area_color)
                n = len(_area_state["clicks"])
                _set_status(f"{n} point(s) \u2014 click first point to close", "info")
            elif kwargs.get("type") == "mousemove":
                now = time.time()
                if now - _last_mousemove[0] < 0.05:
                    return
                _last_mousemove[0] = now
                coords = kwargs.get("coordinates")
                if coords:
                    handle_area_mousemove(m, _area_state, coords, _area_color)
            return

        # Click painting
        if kwargs.get("type") == "click":
            lat, lon = kwargs.get("coordinates", (None, None))
            if lat is None or lon is None:
                return
            if mode == "paint_click":
                _paint_cell(lon, lat)
        elif kwargs.get("type") == "mousemove":
            lat, lon = kwargs.get("coordinates", (None, None))
            if lat is None or lon is None:
                return
            ci, cj = geo_to_cell(lon, lat, _lc_grid_geom, land_cover.shape)
            if ci is not None:
                cls_val = int(land_cover[ci, cj])
                name = _LC_NAMES.get(cls_val, f"Code {cls_val}")
                color = _LC_COLORS.get(cls_val, "#808080")
                hover_info.value = (
                    f"<div class='gm-hover'>"
                    f"<span style='display:inline-block;width:10px;height:10px;"
                    f"border-radius:2px;background:{color};margin-right:4px;'></span>"
                    f"({ci},{cj}) {name}"
                    f"</div>"
                )
            else:
                hover_info.value = ""

    m.on_interaction(handle_map_click)

    return m, land_cover
