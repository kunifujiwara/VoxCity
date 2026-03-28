"""
Interactive tree editor for ipyleaflet maps.

Provides:
- edit_tree: Full interactive editor (add/remove tree points, canopy grid editing)
- create_tree_editor: Convenience wrapper that displays the map
"""

from __future__ import annotations

import math
import time

import numpy as np
import geopandas as gpd
import shapely.geometry as geom
from geopy import distance
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
    build_lc_legend_html,
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


def edit_tree(voxcity=None, initial_center=None, zoom=17):
    """
    Interactive map editor for trees.

    Users can add tree points, remove tree points, visualise the existing
    canopy grid, and remove canopy cells by clicking or drawing an area polygon.

    Args:
        voxcity (VoxCity, optional): VoxCity object for data extraction.
        initial_center (tuple, optional): (lon, lat) for initial map center.
        zoom (int): Initial zoom level. Default=17.

    Returns:
        tuple: (map_object, updated_tree_gdf, canopy_top, canopy_bottom)
    """
    # --- Data extraction ---
    tree_gdf = None
    rectangle_vertices = None
    building_gdf = None
    canopy_top = None
    canopy_bottom = None
    land_cover_t = None
    land_cover_source_t = None
    if voxcity is not None:
        tree_gdf = voxcity.extras.get("tree_gdf", None)
        rectangle_vertices = voxcity.extras.get("rectangle_vertices", None)
        building_gdf = voxcity.extras.get("building_gdf", None)
        if voxcity.tree_canopy is not None and voxcity.tree_canopy.top is not None:
            canopy_top = voxcity.tree_canopy.top.copy()
            if voxcity.tree_canopy.bottom is not None:
                canopy_bottom = voxcity.tree_canopy.bottom.copy()
        if voxcity.land_cover is not None and voxcity.land_cover.classes is not None:
            land_cover_t = voxcity.land_cover.classes.copy()
        land_cover_source_t = voxcity.extras.get("land_cover_source", None)
        if land_cover_source_t is None:
            selected = voxcity.extras.get("selected_sources", {})
            land_cover_source_t = selected.get("land_cover_source", "OpenStreetMap")

    # Initialize or copy tree GeoDataFrame
    if tree_gdf is None:
        updated_trees = gpd.GeoDataFrame(
            columns=["tree_id", "top_height", "bottom_height", "crown_diameter", "geometry"],
            crs="EPSG:4326",
        )
    else:
        updated_trees = tree_gdf.copy()
        if "tree_id" not in updated_trees.columns:
            updated_trees["tree_id"] = range(1, len(updated_trees) + 1)
        for col, default in [("top_height", 10.0), ("bottom_height", 4.0), ("crown_diameter", 6.0)]:
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
        center_lon = (rectangle_vertices[0][0] + rectangle_vertices[2][0]) / 2
        center_lat = (rectangle_vertices[0][1] + rectangle_vertices[2][1]) / 2
    else:
        center_lon, center_lat = -100.0, 40.0

    # Create map
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

    # Building overlay
    buildings_overlay = GeoJSON(
        data=build_building_geojson(building_gdf),
        style={"color": "#1565c0", "fillColor": "#42a5f5", "fillOpacity": 0.35, "weight": 1},
    )
    _overlay_layers["Buildings"] = [buildings_overlay, False]

    # Land cover overlay
    _lc_grid_geom = None
    if land_cover_t is not None and rectangle_vertices is not None and voxcity is not None:
        _lc_grid_geom = compute_grid_geometry(rectangle_vertices, voxcity.land_cover.meta.meshsize)

    if land_cover_t is not None and _lc_grid_geom is not None:
        _lc_overlay = GeoJSON(
            data=build_lc_geojson(land_cover_t, _lc_grid_geom, land_cover_source_t),
            style_callback=lc_style_callback,
        )
        _overlay_layers["Land Cover"] = [_lc_overlay, False]

    # Canopy grid overlay
    _canopy_grid_geom = None
    if canopy_top is not None and rectangle_vertices is not None and voxcity is not None:
        _canopy_grid_geom = compute_grid_geometry(rectangle_vertices, voxcity.tree_canopy.meta.meshsize)

    _canopy_style = {
        "color": "#00ff7f", "fillColor": "#00ff7f", "fillOpacity": 0.35, "weight": 0.5,
    }

    canopy_overlay = GeoJSON(data=build_canopy_geojson(canopy_top, _canopy_grid_geom), style=_canopy_style)
    _canopy_on_map = False
    if canopy_top is not None and np.any(canopy_top > 0):
        m.add_layer(canopy_overlay)
        _canopy_on_map = True

    # Draw existing trees as circles
    tree_layers = {}
    for idx, row in updated_trees.iterrows():
        if row.geometry is not None and hasattr(row.geometry, "x"):
            lat = row.geometry.y
            lon = row.geometry.x
            radius_m = max(int(round(float(row.get("crown_diameter", 6.0)) / 2.0)), 1)
            tree_id_val = int(row.get("tree_id", idx + 1))
            circle = Circle(
                location=(lat, lon), radius=radius_m,
                color="#00ff7f", weight=1, opacity=1.0,
                fill_color="#00ff7f", fill_opacity=0.3,
            )
            m.add_layer(circle)
            tree_layers[tree_id_val] = circle

    # --- UI ---
    style_html = HTML(generate_editor_css(
        "gm-tree-root",
        drawing_mode_class="tree-drawing-mode",
        delete_mode_class="tree-delete-mode",
    ))

    top_height_input = FloatText(
        value=10.0, description="Top (m):",
        layout=Layout(width="160px", height="24px"),
        style={"description_width": "62px"},
    )
    bottom_height_input = FloatText(
        value=4.0, description="Trunk (m):",
        layout=Layout(width="160px", height="24px"),
        style={"description_width": "62px"},
    )
    crown_diameter_input = FloatText(
        value=6.0, description="Dia. (m):",
        layout=Layout(width="160px", height="24px"),
        style={"description_width": "62px"},
    )
    fixed_prop_checkbox = Checkbox(
        value=True, description="Fixed proportion", indent=False,
        layout=Layout(width="auto", height="20px"),
    )

    add_click_button = Button(description="Click", button_style="success", layout=Layout(flex="1", height="26px"))
    add_area_button = ToggleButton(
        value=False, description="Area", button_style="",
        layout=Layout(flex="1", height="26px"), tooltip="Draw polygon to fill with trees",
    )
    remove_click_button = Button(description="Click", button_style="", layout=Layout(flex="1", height="26px"))
    remove_area_button = ToggleButton(
        value=False, description="Area", button_style="",
        layout=Layout(flex="1", height="26px"), tooltip="Draw polygon to remove trees & canopy inside",
    )

    status_bar = HTML(
        value="<div style='padding:3px 8px;border-radius:10px;font-size:10px;"
              "text-align:center;background:#f0f4ff;color:#1a73e8;'>Ready</div>"
    )
    hover_info = HTML("")
    _set_status = make_status_setter(status_bar, use_gm_class=False)

    add_row = HBox([add_click_button, add_area_button], layout=Layout(margin="0", gap="4px"))
    remove_row = HBox([remove_click_button, remove_area_button], layout=Layout(margin="0", gap="4px"))
    param_col = VBox(
        [top_height_input, bottom_height_input, crown_diameter_input, fixed_prop_checkbox],
        layout=Layout(margin="0", gap="0px"),
    )

    crown_diameter_input.layout.display = ""

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
    _lc_legend_widget = HTML(layout=Layout(display="none"))
    _legend_raw = build_lc_legend_html(land_cover_source_t)
    _layer_checkboxes = []
    for _lyr_name, (_lyr_obj, _lyr_on) in _overlay_layers.items():
        cb = Checkbox(
            value=_lyr_on, description=_lyr_name, indent=False,
            layout=Layout(width="auto", height="20px"),
        )
        _toggle_cb = make_layer_toggle(
            m, _overlay_layers, _lyr_name, _lyr_obj,
            front_layers=lambda: [canopy_overlay] + list(tree_layers.values()),
        )
        if _lyr_name == "Land Cover" and _legend_raw:
            def _lc_toggle(change, _orig=_toggle_cb):
                _orig(change)
                if change["new"]:
                    _lc_legend_widget.value = _legend_raw
                    _lc_legend_widget.layout.display = None
                else:
                    _lc_legend_widget.layout.display = "none"
            cb.observe(_lc_toggle, names="value")
            if _lyr_on:
                _lc_legend_widget.value = _legend_raw
                _lc_legend_widget.layout.display = None
        else:
            cb.observe(_toggle_cb, names="value")
        _layer_checkboxes.append(cb)

    _layer_widgets = [
        HTML("<div class='gm-sep' style='margin:4px 0;'></div>"),
        HTML("<div class='gm-label' style='margin:0 0 2px 0;'>Basemap</div>"),
        basemap_dropdown,
    ]
    if _layer_checkboxes:
        _layer_widgets.append(HTML("<div class='gm-label' style='margin:4px 0 2px 0;'>Layers</div>"))
        _layer_widgets.extend(_layer_checkboxes)
    if _legend_raw:
        _layer_widgets.append(_lc_legend_widget)

    panel = VBox(
        [
            style_html,
            HTML("<div class='gm-title' style='margin-bottom:4px;padding-bottom:4px;'>Tree Editor</div>"),
            HTML("<div class='gm-label' style='margin:0 0 2px 0;'>Add</div>"),
            add_row,
            HTML("<div class='gm-sep' style='margin:4px 0;'></div>"),
            param_col,
            HTML("<div class='gm-sep' style='margin:4px 0;'></div>"),
            HTML("<div class='gm-label' style='margin:0 0 2px 0;'>Remove</div>"),
            remove_row,
            status_bar,
            hover_info,
        ] + _layer_widgets,
        layout=Layout(width="200px", padding="8px 10px"),
    )
    panel.add_class("gm-tree-root")

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
    mode = "add_click"
    _area_state = {"clicks": [], "layers": [], "preview": None}
    _last_area_click = [0.0]
    _last_mousemove = [0.0]

    # Fixed proportion state
    base_bottom_ratio = bottom_height_input.value / top_height_input.value if top_height_input.value else 0.4
    base_crown_ratio = crown_diameter_input.value / top_height_input.value if top_height_input.value else 0.6
    updating_params = False

    def recompute_from_top(new_top: float):
        nonlocal updating_params
        if new_top <= 0:
            return
        updating_params = True
        bottom_height_input.value = max(0.0, base_bottom_ratio * new_top)
        crown_diameter_input.value = max(0.0, base_crown_ratio * new_top)
        updating_params = False

    def recompute_from_bottom(new_bottom: float):
        nonlocal updating_params
        if base_bottom_ratio <= 0:
            return
        new_top = max(0.0, new_bottom / base_bottom_ratio)
        updating_params = True
        top_height_input.value = new_top
        crown_diameter_input.value = max(0.0, base_crown_ratio * new_top)
        updating_params = False

    def recompute_from_crown(new_crown: float):
        nonlocal updating_params
        if base_crown_ratio <= 0:
            return
        new_top = max(0.0, new_crown / base_crown_ratio)
        updating_params = True
        top_height_input.value = new_top
        bottom_height_input.value = max(0.0, base_bottom_ratio * new_top)
        updating_params = False

    def on_toggle_fixed(change):
        nonlocal base_bottom_ratio, base_crown_ratio
        if change["name"] == "value":
            if change["new"]:
                top = float(top_height_input.value) or 1.0
                base_bottom_ratio = max(0.0, float(bottom_height_input.value) / top)
                base_crown_ratio = max(0.0, float(crown_diameter_input.value) / top)

    def on_top_change(change):
        if change["name"] == "value" and fixed_prop_checkbox.value and not updating_params:
            try:
                recompute_from_top(float(change["new"]))
            except Exception:
                pass

    def on_bottom_change(change):
        if change["name"] == "value" and fixed_prop_checkbox.value and not updating_params:
            try:
                recompute_from_bottom(float(change["new"]))
            except Exception:
                pass

    def on_crown_change(change):
        if change["name"] == "value" and fixed_prop_checkbox.value and not updating_params:
            try:
                recompute_from_crown(float(change["new"]))
            except Exception:
                pass

    fixed_prop_checkbox.observe(on_toggle_fixed, names="value")
    top_height_input.observe(on_top_change, names="value")
    bottom_height_input.observe(on_bottom_change, names="value")
    crown_diameter_input.observe(on_crown_change, names="value")

    def _update_param_visibility():
        crown_diameter_input.layout.display = "" if mode == "add_click" else "none"

    # --- Area operations ---
    def _execute_area_addition(polygon_coords):
        from matplotlib.path import Path as _MplPath

        th = float(top_height_input.value)
        bh = float(bottom_height_input.value)
        if bh > th:
            bh, th = th, bh

        added_cells = 0
        if canopy_top is not None and _canopy_grid_geom is not None:
            origin = _canopy_grid_geom["origin"]
            u = _canopy_grid_geom["u_vec"]
            v = _canopy_grid_geom["v_vec"]
            dx = _canopy_grid_geom["adj_mesh"][0]
            dy = _canopy_grid_geom["adj_mesh"][1]
            nx, ny = canopy_top.shape

            ii_all = np.arange(nx)
            jj_all = np.arange(ny)
            ii_grid, jj_grid = np.meshgrid(ii_all, jj_all, indexing="ij")
            ii_flat = ii_grid.ravel()
            jj_flat = jj_grid.ravel()

            centers = origin + np.outer((ii_flat + 0.5) * dx, u) + np.outer((jj_flat + 0.5) * dy, v)
            mpl_path = _MplPath(polygon_coords)
            inside = mpl_path.contains_points(centers)

            if np.any(inside):
                added_cells = int(inside.sum())
                canopy_top[ii_flat[inside], jj_flat[inside]] = th
                if canopy_bottom is not None:
                    canopy_bottom[ii_flat[inside], jj_flat[inside]] = bh
                canopy_overlay.data = build_canopy_geojson(canopy_top, _canopy_grid_geom)

        clear_area_draw(m, _area_state)
        m.double_click_zoom = True
        if added_cells > 0:
            _set_status(f"Added {added_cells} canopy cell(s)", "success")
        else:
            _set_status("No cells in drawn area", "warn")

    def _execute_area_removal(polygon_coords):
        nonlocal updated_trees
        removal_polygon = geom.Polygon(polygon_coords)
        removed_trees = 0
        removed_cells = 0

        # Remove tree points
        indices_to_drop = []
        for idx2, row2 in updated_trees.iterrows():
            if row2.geometry is not None and hasattr(row2.geometry, "x"):
                if removal_polygon.contains(row2.geometry):
                    tid = int(row2.get("tree_id", idx2 + 1))
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

        # Remove canopy cells
        if canopy_top is not None and _canopy_grid_geom is not None:
            from matplotlib.path import Path as _MplPath

            origin = _canopy_grid_geom["origin"]
            u = _canopy_grid_geom["u_vec"]
            v = _canopy_grid_geom["v_vec"]
            dx = _canopy_grid_geom["adj_mesh"][0]
            dy = _canopy_grid_geom["adj_mesh"][1]

            nz_mask = canopy_top > 0
            ii_nz, jj_nz = np.nonzero(nz_mask)
            if len(ii_nz) > 0:
                centers = origin + np.outer((ii_nz + 0.5) * dx, u) + np.outer((jj_nz + 0.5) * dy, v)
                mpl_path = _MplPath(polygon_coords)
                inside = mpl_path.contains_points(centers)
                if np.any(inside):
                    removed_cells = int(inside.sum())
                    canopy_top[ii_nz[inside], jj_nz[inside]] = 0
                    if canopy_bottom is not None:
                        canopy_bottom[ii_nz[inside], jj_nz[inside]] = 0
                    canopy_overlay.data = build_canopy_geojson(canopy_top, _canopy_grid_geom)

        clear_area_draw(m, _area_state)
        m.double_click_zoom = True
        if removed_trees > 0 or removed_cells > 0:
            parts = []
            if removed_trees:
                parts.append(f"{removed_trees} tree(s)")
            if removed_cells:
                parts.append(f"{removed_cells} cell(s)")
            _set_status(f"Removed {', '.join(parts)}", "success")
        else:
            _set_status("No trees in selected area", "warn")

    # --- Mode management ---
    _toggling = [False]

    def _update_button_styles():
        add_click_button.button_style = "success" if mode == "add_click" else ""
        add_area_button.button_style = "success" if mode == "add_area" else ""
        add_area_button.value = (mode == "add_area")
        remove_click_button.button_style = "danger" if mode == "remove_click" else ""
        remove_area_button.button_style = "danger" if mode == "remove_area" else ""
        remove_area_button.value = (mode == "remove_area")

    def set_mode(new_mode):
        nonlocal mode
        mode = new_mode
        clear_area_draw(m, _area_state)
        if new_mode in ("add_area", "remove_area"):
            m.double_click_zoom = False
            m.default_style = {"cursor": "crosshair"}
            m.add_class("tree-drawing-mode")
        else:
            m.double_click_zoom = True
            m.default_style = {"cursor": "grab"}
            m.remove_class("tree-drawing-mode")
        _toggling[0] = True
        _update_button_styles()
        _toggling[0] = False
        _update_param_visibility()
        status_map = {
            "add_click": ("Click map to add tree", "info"),
            "add_area": ("Draw polygon to fill with trees", "info"),
            "remove_click": ("Click trees/canopy to remove", "danger"),
            "remove_area": ("Draw polygon to remove trees", "danger"),
        }
        msg, stype = status_map.get(new_mode, ("Ready", "info"))
        _set_status(msg, stype)

    def on_click_add_click(b):
        set_mode("add_click")

    def on_click_remove_click(b):
        set_mode("remove_click")

    def on_add_area_toggle(change):
        if _toggling[0] or change["name"] != "value":
            return
        _toggling[0] = True
        set_mode("add_area" if change["new"] else "add_click")
        _toggling[0] = False

    def on_remove_area_toggle(change):
        if _toggling[0] or change["name"] != "value":
            return
        _toggling[0] = True
        set_mode("remove_area" if change["new"] else "add_click")
        _toggling[0] = False

    add_click_button.on_click(on_click_add_click)
    add_area_button.observe(on_add_area_toggle, names="value")
    remove_click_button.on_click(on_click_remove_click)
    remove_area_button.observe(on_remove_area_toggle, names="value")

    # --- Map interaction ---
    def handle_map_click(**kwargs):
        nonlocal updated_trees

        # Area polygon drawing (add_area or remove_area)
        if mode in ("add_area", "remove_area"):
            _area_color = "#1a73e8" if mode == "add_area" else "#FF0000"
            _area_stype = "info" if mode == "add_area" else "danger"
            _area_execute = _execute_area_addition if mode == "add_area" else _execute_area_removal

            if kwargs.get("type") == "dblclick":
                pts = _area_state["clicks"]
                if len(pts) >= 3:
                    _area_execute(pts)
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
                    _area_execute(_area_state["clicks"])
                    return
                _area_state["clicks"].append((lon, lat))
                refresh_area_markers(m, _area_state, _area_color)
                n = len(_area_state["clicks"])
                _set_status(f"{n} point(s) \u2014 click first point to close", _area_stype)
            elif kwargs.get("type") == "mousemove":
                now = time.time()
                if now - _last_mousemove[0] < 0.05:
                    return
                _last_mousemove[0] = now
                coords = kwargs.get("coordinates")
                if coords:
                    handle_area_mousemove(m, _area_state, coords, _area_color)
            return

        if kwargs.get("type") == "click":
            lat, lon = kwargs.get("coordinates", (None, None))
            if lat is None or lon is None:
                return
            if mode == "add_click":
                next_tree_id = int(updated_trees["tree_id"].max() + 1) if len(updated_trees) > 0 else 1
                th = float(top_height_input.value)
                bh = float(bottom_height_input.value)
                cd = float(crown_diameter_input.value)
                if bh > th:
                    bh, th = th, bh
                if cd < 0:
                    cd = 0.0

                new_row = {
                    "tree_id": next_tree_id,
                    "top_height": th,
                    "bottom_height": bh,
                    "crown_diameter": cd,
                    "geometry": geom.Point(lon, lat),
                }
                new_index = len(updated_trees)
                updated_trees.loc[new_index] = new_row

                radius_m = max(int(round(cd / 2.0)), 1)
                circle = Circle(
                    location=(lat, lon), radius=radius_m,
                    color="#00ff7f", weight=1, opacity=1.0,
                    fill_color="#00ff7f", fill_opacity=0.3,
                )
                m.add_layer(circle)
                tree_layers[next_tree_id] = circle
                _set_status(f"Added tree #{next_tree_id}", "success")

            elif mode == "remove_click":
                removed_something = False
                candidate_id = None
                candidate_idx = None
                candidate_dist = None
                for idx2, row2 in updated_trees.iterrows():
                    if row2.geometry is None or not hasattr(row2.geometry, "x"):
                        continue
                    lat2 = row2.geometry.y
                    lon2 = row2.geometry.x
                    dist_m = distance.distance((lat, lon), (lat2, lon2)).meters
                    rad_m = max(int(round(float(row2.get("crown_diameter", 6.0)) / 2.0)), 1)
                    thr_m = rad_m + 5
                    if (candidate_dist is None and dist_m <= thr_m) or (
                        candidate_dist is not None and dist_m < candidate_dist and dist_m <= thr_m
                    ):
                        candidate_dist = dist_m
                        candidate_id = int(row2.get("tree_id", idx2 + 1))
                        candidate_idx = idx2

                if candidate_id is not None:
                    layer = tree_layers.get(candidate_id)
                    if layer is not None:
                        m.remove_layer(layer)
                        del tree_layers[candidate_id]
                    updated_trees.drop(index=candidate_idx, inplace=True)
                    updated_trees.reset_index(drop=True, inplace=True)
                    removed_something = True
                    _set_status(f"Removed tree #{candidate_id}", "danger")

                ci, cj = geo_to_cell(lon, lat, _canopy_grid_geom, canopy_top.shape if canopy_top is not None else None)
                if ci is not None and canopy_top is not None and canopy_top[ci, cj] > 0:
                    canopy_top[ci, cj] = 0
                    if canopy_bottom is not None:
                        canopy_bottom[ci, cj] = 0
                    canopy_overlay.data = build_canopy_geojson(canopy_top, _canopy_grid_geom)
                    removed_something = True
                    if candidate_id is None:
                        _set_status("Removed canopy cell", "danger")

                if not removed_something:
                    _set_status("Nothing to remove here", "warn")

        elif kwargs.get("type") == "mousemove":
            lat, lon = kwargs.get("coordinates", (None, None))
            if lat is None or lon is None:
                return
            shown = False
            for _, row2 in updated_trees.iterrows():
                if row2.geometry is None or not hasattr(row2.geometry, "x"):
                    continue
                lat2 = row2.geometry.y
                lon2 = row2.geometry.x
                dist_m = distance.distance((lat, lon), (lat2, lon2)).meters
                rad_m = max(int(round(float(row2.get("crown_diameter", 6.0)) / 2.0)), 1)
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
    """Convenience wrapper to display the tree editor map."""
    result = edit_tree(tree_gdf, initial_center, zoom, rectangle_vertices)
    display(result[0])
    return result[1]
