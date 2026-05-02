"""
Shared constants, helpers, CSS generators, and GeoJSON builders used across draw editors.

This module centralises code that was duplicated in edit_building, edit_tree,
and edit_landcover, including:
- Land-cover colour palette
- Basemap tile-layer factory
- Grid-geometry computation from rectangle vertices
- GeoJSON builders for buildings, canopy strips, and land-cover grids
- Reusable area-polygon drawing helpers for interactive maps
- Parameterised CSS generator for editor panels
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import shapely.geometry as geom
from ipyleaflet import (
    Circle,
    GeoJSON,
    Polyline,
    TileLayer,
)

# ─────────────────────────────────────────────────────────────
# Land-cover class colour palette (by class name)
# ─────────────────────────────────────────────────────────────
LC_COLORS_BY_NAME: dict[str, str] = {
    "Bareland":          "#c4a882",
    "Rangeland":         "#b8d68e",
    "Shrub":             "#6b8e23",
    "Agriculture land":  "#f5deb3",
    "Tree":              "#228b22",
    "Moss and lichen":   "#8fbc8f",
    "Wet land":          "#4682b4",
    "Mangrove":          "#2e8b57",
    "Mangroves":         "#2e8b57",
    "Water":             "#1e90ff",
    "Snow and ice":      "#f0f8ff",
    "Developed space":   "#a9a9a9",
    "Road":              "#696969",
    "Building":          "#cd853f",
    "No Data":           "#808080",
    # Source-specific aliases
    "Parking Lot":       "#a9a9a9",
    "Tree Canopy":       "#228b22",
    "Grass/Shrub":       "#b8d68e",
    "Agriculture":       "#f5deb3",
    "Barren":            "#c4a882",
    "Unknown":           "#808080",
    "Sea":               "#1e90ff",
    "Trees":             "#228b22",
    "Grass":             "#b8d68e",
    "Grassland":         "#b8d68e",
    "Flooded Vegetation": "#4682b4",
    "Crops":             "#f5deb3",
    "Cropland":          "#f5deb3",
    "Scrub/Shrub":       "#6b8e23",
    "Shrubland":         "#6b8e23",
    "Built Area":        "#cd853f",
    "Built-up":          "#cd853f",
    "Built":             "#cd853f",
    "Bare Ground":       "#c4a882",
    "Bare":              "#c4a882",
    "Barren / sparse vegetation": "#c4a882",
    "Snow/Ice":          "#f0f8ff",
    "Snow and Ice":      "#f0f8ff",
    "Clouds":            "#808080",
    "Open water":        "#1e90ff",
    "Herbaceous wetland": "#4682b4",
    "Shrub and Scrub":   "#6b8e23",
}

# Keep a backward-compatible alias
_LC_COLORS_BY_NAME = LC_COLORS_BY_NAME


def get_lc_source_colors(land_cover_source: str | None) -> dict[str, str]:
    """Return ``{class_name: hex_color}`` derived from source RGB keys.

    Uses the RGB tuples defined in ``get_land_cover_classes()`` so that the
    editor UI matches the colours used in visualiser and exporter modules.
    Falls back to :data:`LC_COLORS_BY_NAME` when *land_cover_source* is None.
    """
    if land_cover_source is None:
        return dict(LC_COLORS_BY_NAME)

    from ...utils.lc import get_land_cover_classes

    src_classes = get_land_cover_classes(land_cover_source)
    colors: dict[str, str] = {}
    for rgb, name in src_classes.items():
        if name not in colors:
            r, g, b = rgb
            colors[name] = f"#{r:02x}{g:02x}{b:02x}"
    return colors


# ─────────────────────────────────────────────────────────────
# Basemap tile layers
# ─────────────────────────────────────────────────────────────
def create_basemap_tiles() -> dict[str, TileLayer]:
    """Return a fresh dict of basemap TileLayers (Google Sat, CartoDB, OSM)."""
    return {
        "Google Satellite": TileLayer(
            url="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            name="Google Satellite",
            attribution="Google Satellite",
        ),
        "CartoDB Positron": TileLayer(
            url="https://cartodb-basemaps-a.global.ssl.fastly.net/light_all/{z}/{x}/{y}@2x.png",
            attribution='&copy; <a href="https://carto.com/">CARTO</a>',
            name="CartoDB Positron",
            max_zoom=20,
        ),
        "OpenStreetMap": TileLayer(
            url="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
            name="OpenStreetMap",
            max_zoom=19,
        ),
    }


# ─────────────────────────────────────────────────────────────
# Grid-geometry computation
# ─────────────────────────────────────────────────────────────
def compute_grid_geometry(rectangle_vertices: list, meshsize: float) -> dict | None:
    """Compute grid geometry dict from rectangle vertices and mesh size.

    Returns a :class:`~voxcity.utils.projector.GridGeom` dict with keys:
    origin, side_1, side_2, u_vec, v_vec, grid_size, adj_mesh, meshsize_m —
    or *None* if inputs are insufficient.

    Delegates to :func:`geoprocessor.raster.core.compute_grid_geometry`.
    """
    from ..raster.core import compute_grid_geometry as _compute_grid_geometry
    return _compute_grid_geometry(rectangle_vertices, meshsize)


# ─────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────
def geo_to_cell(
    lon: float,
    lat: float,
    grid_geom: dict | None,
    array_shape: tuple[int, int] | None,
) -> tuple[int | None, int | None]:
    """lon_lat → ij_north integer cell index, clamped to array_shape.

    Returns (None, None) when the point is outside the grid or inputs are
    missing. Uses :class:`~voxcity.utils.projector.GridProjector` internally.
    """
    if grid_geom is None or array_shape is None:
        return None, None
    from voxcity.utils.projector import GridProjector
    try:
        proj = GridProjector(grid_geom)
    except (KeyError, ValueError):
        return None, None
    u, v = proj.lon_lat_to_ij_north(lon, lat)
    i = int(math.floor(u))
    j = int(math.floor(v))
    if 0 <= i < array_shape[0] and 0 <= j < array_shape[1]:
        return i, j
    return None, None


# ─────────────────────────────────────────────────────────────
# GeoJSON builders
# ─────────────────────────────────────────────────────────────
_EMPTY_FC: dict = {"type": "FeatureCollection", "features": []}


def build_building_geojson(
    building_gdf: Any,
    *,
    include_height: bool = False,
) -> dict:
    """Build a GeoJSON FeatureCollection from a building GeoDataFrame.

    Parameters
    ----------
    building_gdf : GeoDataFrame or None
    include_height : bool
        If *True*, each feature's properties will contain ``idx`` and ``height``.
    """
    features: list[dict] = []
    if building_gdf is None or len(building_gdf) == 0:
        return dict(_EMPTY_FC)
    for idx, row in building_gdf.iterrows():
        if isinstance(row.geometry, geom.Polygon):
            coords = [list(row.geometry.exterior.coords)]
            if any(math.isnan(c) for ring in coords for pt in ring for c in pt):
                continue
            props: dict[str, Any] = {}
            if include_height:
                h = row.get("height", 0)
                h = 0.0 if (h is None or (isinstance(h, float) and math.isnan(h))) else float(h)
                estimated = bool(row.get("height_estimated", False))
                props = {"idx": int(idx), "height": h, "height_estimated": estimated}
            features.append({
                "type": "Feature",
                "id": str(idx),
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": coords},
            })
    return {"type": "FeatureCollection", "features": features}


def build_canopy_geojson(canopy_top: np.ndarray | None, grid_geom: dict | None) -> dict:
    """Build a merged GeoJSON overlay for non-zero canopy cells.

    Uses row-strip merging and unary_union for efficient polygon reduction.
    """
    if canopy_top is None or grid_geom is None:
        return dict(_EMPTY_FC)

    origin = grid_geom["origin"]
    u = grid_geom["u_vec"]
    v = grid_geom["v_vec"]
    dx = grid_geom["adj_mesh"][0]
    dy = grid_geom["adj_mesh"][1]

    mask = canopy_top > 0
    if not np.any(mask):
        return dict(_EMPTY_FC)

    nx, ny = mask.shape
    strips: list = []
    for i in range(nx):
        row = mask[i]
        if not np.any(row):
            continue
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for s, e in zip(starts, ends):
            bl = origin + (i * dx) * u + (s * dy) * v
            br = origin + ((i + 1) * dx) * u + (s * dy) * v
            tr = origin + ((i + 1) * dx) * u + (e * dy) * v
            tl = origin + (i * dx) * u + (e * dy) * v
            strips.append(geom.Polygon([bl, br, tr, tl]))

    if not strips:
        return dict(_EMPTY_FC)

    from shapely.ops import unary_union

    merged = unary_union(strips)
    if merged.is_empty:
        return dict(_EMPTY_FC)

    def _poly_feature(poly, fid):
        coords = [list(poly.exterior.coords)]
        for interior in poly.interiors:
            coords.append(list(interior.coords))
        return {
            "type": "Feature",
            "id": str(fid),
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": coords},
        }

    features: list[dict] = []
    if merged.geom_type == "Polygon":
        features.append(_poly_feature(merged, 0))
    elif merged.geom_type in ("MultiPolygon", "GeometryCollection"):
        fid = 0
        for part in merged.geoms:
            if part.geom_type == "Polygon" and not part.is_empty:
                features.append(_poly_feature(part, fid))
                fid += 1
    return {"type": "FeatureCollection", "features": features}


def build_lc_geojson(
    land_cover: np.ndarray | None,
    grid_geom: dict | None,
    land_cover_source: str | None,
) -> dict:
    """Build colour-coded GeoJSON from a land-cover grid.

    Merges contiguous cells of the same class per row into strip polygons.
    """
    if land_cover is None or grid_geom is None:
        return dict(_EMPTY_FC)

    from ...utils.lc import get_land_cover_classes

    src_classes = get_land_cover_classes(land_cover_source)
    class_names = list(dict.fromkeys(src_classes.values()))
    name_to_hex = get_lc_source_colors(land_cover_source)
    lc_colors = {i: name_to_hex.get(name, "#808080") for i, name in enumerate(class_names)}
    num = len(class_names)

    origin = grid_geom["origin"]
    u = grid_geom["u_vec"]
    v = grid_geom["v_vec"]
    # du/dv = cell size in metres along u_vec/v_vec (not Cartesian x/y).
    # For a rotated rectangle these axes may not be axis-aligned.
    du = grid_geom["adj_mesh"][0]
    dv = grid_geom["adj_mesh"][1]
    dx, dy = du, dv  # local aliases used in corner-point formulas below

    nx, ny = land_cover.shape
    features: list[dict] = []
    fid = 0
    for i in range(nx):
        row = land_cover[i]
        j = 0
        while j < ny:
            cls_val = int(row[j])
            if cls_val < 0 or cls_val >= num:
                j += 1
                continue
            j_end = j + 1
            while j_end < ny and int(row[j_end]) == cls_val:
                j_end += 1
            bl = origin + (i * dx) * u + (j * dy) * v
            br = origin + ((i + 1) * dx) * u + (j * dy) * v
            tr = origin + ((i + 1) * dx) * u + (j_end * dy) * v
            tl = origin + (i * dx) * u + (j_end * dy) * v
            coords = [bl.tolist(), br.tolist(), tr.tolist(), tl.tolist(), bl.tolist()]
            color = lc_colors.get(cls_val, "#808080")
            features.append({
                "type": "Feature",
                "id": str(fid),
                "properties": {
                    "cls": cls_val,
                    "style": {
                        "color": color,
                        "fillColor": color,
                        "fillOpacity": 0.55,
                        "weight": 0.3,
                    },
                },
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            })
            fid += 1
            j = j_end
    return {"type": "FeatureCollection", "features": features}


def lc_style_callback(feature: dict) -> dict:
    """Default style callback for land-cover GeoJSON layers."""
    return feature["properties"].get("style", {
        "color": "#808080",
        "fillColor": "#808080",
        "fillOpacity": 0.55,
        "weight": 0.3,
    })


# ─────────────────────────────────────────────────────────────
# Land-cover legend builder
# ─────────────────────────────────────────────────────────────
def build_lc_legend_html(land_cover_source: str | None) -> str:
    """Return an HTML string showing a compact land-cover class legend.

    Only classes that belong to the given *land_cover_source* are shown.
    Returns an empty string when the source is ``None``.
    """
    if land_cover_source is None:
        return ""

    from ...utils.lc import get_land_cover_classes

    src_classes = get_land_cover_classes(land_cover_source)
    class_names = list(dict.fromkeys(src_classes.values()))
    name_to_hex = get_lc_source_colors(land_cover_source)

    items: list[str] = []
    for name in class_names:
        color = name_to_hex.get(name, "#808080")
        items.append(
            f"<span style='display:inline-flex;align-items:center;margin:0 4px 1px 0;'>"
            f"<span style='width:8px;height:8px;border-radius:1px;"
            f"background:{color};display:inline-block;margin-right:2px;"
            f"border:1px solid rgba(0,0,0,0.15);flex-shrink:0;'></span>"
            f"<span style='font-size:9px;color:#3c4043;white-space:nowrap;line-height:1;'>{name}</span>"
            f"</span>"
        )
    return (
        "<div style='display:flex;flex-wrap:wrap;gap:0;margin-top:2px;'>"
        + "".join(items)
        + "</div>"
    )


# ─────────────────────────────────────────────────────────────
# Interactive area-polygon drawing helpers
# ─────────────────────────────────────────────────────────────
CLOSE_THRESHOLD = 0.0001


def is_near_first(pts: list[tuple[float, float]], lon: float, lat: float) -> bool:
    """Return True if (lon, lat) is within CLOSE_THRESHOLD of pts[0]."""
    if len(pts) < 3:
        return False
    dx = lon - pts[0][0]
    dy = lat - pts[0][1]
    return (dx * dx + dy * dy) < (CLOSE_THRESHOLD * CLOSE_THRESHOLD)


def clear_area_draw(m, area_state: dict) -> None:
    """Remove all layers and reset state for an area-polygon drawing session."""
    while area_state["layers"]:
        try:
            m.remove_layer(area_state["layers"].pop())
        except Exception:
            pass
    if area_state["preview"]:
        try:
            m.remove_layer(area_state["preview"])
        except Exception:
            pass
        area_state["preview"] = None
    area_state["clicks"] = []


def refresh_area_markers(m, area_state: dict, color: str = "#FF0000") -> None:
    """Redraw vertex markers and edges for an area-polygon being drawn."""
    while area_state["layers"]:
        try:
            m.remove_layer(area_state["layers"].pop())
        except Exception:
            pass
    pts = area_state["clicks"]
    for lon_p, lat_p in pts:
        pt = Circle(
            location=(lat_p, lon_p),
            radius=2,
            color=color,
            fill_color=color,
            fill_opacity=1.0,
        )
        m.add_layer(pt)
        area_state["layers"].append(pt)
    for i in range(len(pts) - 1):
        line = Polyline(
            locations=[(pts[i][1], pts[i][0]), (pts[i + 1][1], pts[i + 1][0])],
            color=color,
            weight=2,
        )
        m.add_layer(line)
        area_state["layers"].append(line)


def handle_area_mousemove(
    m,
    area_state: dict,
    coords: tuple[float, float],
    color: str = "#FF0000",
) -> None:
    """Update the dashed preview line from the last click to the cursor."""
    pts = area_state["clicks"]
    if not pts:
        return
    lat_c, lon_c = coords
    if area_state["preview"] and isinstance(area_state["preview"], Polyline):
        area_state["preview"].locations = [(pts[-1][1], pts[-1][0]), (lat_c, lon_c)]
    else:
        if area_state["preview"]:
            try:
                m.remove_layer(area_state["preview"])
            except Exception:
                pass
        line = Polyline(
            locations=[(pts[-1][1], pts[-1][0]), (lat_c, lon_c)],
            color=color,
            weight=2,
            dash_array="5, 5",
        )
        area_state["preview"] = line
        m.add_layer(line)


# ─────────────────────────────────────────────────────────────
# Layer-toggle checkbox factory
# ─────────────────────────────────────────────────────────────
def make_layer_toggle(m, overlay_layers: dict, layer_name: str, layer_obj,
                      front_layers=None):
    """Return an observe callback that toggles *layer_obj* on/off the map.

    Parameters
    ----------
    front_layers : list[Layer] | callable, optional
        Layers that must always remain in front (on top) of overlay layers.
        When an overlay is toggled on the entire ``m.layers`` tuple is rebuilt
        so that these layers are guaranteed to be last (= rendered on top).
        Can also be a callable returning such a list.
    """
    def _toggle(change):
        if change["new"]:
            overlay_layers[layer_name][1] = True
            # Rebuild m.layers so front layers stay on top.
            _fl = front_layers() if callable(front_layers) else (front_layers or [])
            _fl_ids = {id(x) for x in _fl}
            on_map_ids = {id(l) for l in m.layers}
            # Base layers = everything currently on map except front layers
            base = [l for l in m.layers if id(l) not in _fl_ids]
            # Insert the new overlay
            if id(layer_obj) not in on_map_ids:
                base.append(layer_obj)
            # Re-append front layers that were already on the map
            front_on_map = [f for f in _fl if id(f) in on_map_ids]
            m.layers = tuple(base + front_on_map)
        else:
            overlay_layers[layer_name][1] = False
            layers = [l for l in m.layers if l is not layer_obj]
            m.layers = tuple(layers)
    return _toggle


# ─────────────────────────────────────────────────────────────
# CSS generator
# ─────────────────────────────────────────────────────────────
def generate_editor_css(
    root_class: str = "gm-root",
    *,
    extra_css: str = "",
    hide_leaflet_draw: bool = True,
    drawing_mode_class: str = "drawing-mode",
    delete_mode_class: str = "delete-mode",
) -> str:
    """Return the CSS string for an editor panel.

    Parameters
    ----------
    root_class : str
        The CSS class applied to the panel root widget.
    extra_css : str
        Additional CSS rules appended after the standard block.
    hide_leaflet_draw : bool
        Whether to include the rule hiding the Leaflet.draw toolbar.
    drawing_mode_class, delete_mode_class : str
        CSS classes toggled on the map widget for cursor overrides.
    """
    hide_draw = ""
    if hide_leaflet_draw:
        hide_draw = "    .leaflet-draw { display: none !important; }\n"

    return f"""<style>
{hide_draw}
    .{drawing_mode_class},
    .{drawing_mode_class} .leaflet-container,
    .{drawing_mode_class} .leaflet-interactive,
    .{drawing_mode_class} .leaflet-grab,
    .{drawing_mode_class} .leaflet-overlay-pane,
    .{drawing_mode_class} .leaflet-overlay-pane * {{
        cursor: crosshair !important;
    }}
    .{delete_mode_class},
    .{delete_mode_class} .leaflet-container,
    .{delete_mode_class} .leaflet-interactive,
    .{delete_mode_class} .leaflet-grab {{
        cursor: pointer !important;
    }}

    .{root_class} {{
        font-family: 'Google Sans', 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: #1f1f1f;
        line-height: 1.5;
    }}
    .{root_class} * {{ box-sizing: border-box; }}

    .{root_class} .gm-title {{
        font-size: 14px; font-weight: 500; color: #1f1f1f;
        padding-bottom: 6px;
        border-bottom: 1px solid #e8eaed;
        margin-bottom: 8px;
    }}
    .{root_class} .gm-label {{
        font-size: 11px; font-weight: 500; color: #5f6368;
        letter-spacing: 0.3px;
        margin: 0 0 4px 0;
    }}
    .{root_class} .gm-sep {{ height: 1px; background: #e8eaed; margin: 8px 0; }}

    .{root_class} .gm-status {{
        padding: 4px 10px; border-radius: 16px;
        font-size: 11px; font-weight: 400; line-height: 1.3;
        margin-top: 6px; text-align: center;
    }}
    .{root_class} .gm-status-info    {{ background: #f0f4ff; color: #1a73e8; }}
    .{root_class} .gm-status-success {{ background: #e6f4ea; color: #137333; }}
    .{root_class} .gm-status-warn    {{ background: #fef7e0; color: #b06000; }}
    .{root_class} .gm-status-danger  {{ background: #fce8e6; color: #c5221f; }}

    .{root_class} .gm-hint {{
        font-size: 10px; color: #80868b; line-height: 1.4; margin: 0 0 8px 0;
    }}
    .{root_class} .gm-hover {{
        font-size: 11px; color: #1a73e8; font-weight: 500;
        padding: 4px 10px; border-radius: 12px;
        background: #f0f4ff; margin-top: 6px;
        min-height: 0; line-height: 1.3;
    }}
    .{root_class} .gm-hover:empty {{ display: none; }}

    .{root_class} .jupyter-button,
    .{root_class} .jupyter-widgets.widget-toggle-button button {{
        border-radius: 18px !important;
        font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        border: 1px solid #dadce0 !important;
        box-shadow: none !important;
        transition: background 0.15s, border-color 0.15s, box-shadow 0.15s !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        padding: 0 !important;
        line-height: 1 !important;
    }}
    .{root_class} .jupyter-button:hover,
    .{root_class} .jupyter-widgets.widget-toggle-button button:hover {{
        box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
    }}
    .{root_class} .widget-toggle-button {{ height: 26px !important; }}

    /* neutral */
    .{root_class} .jupyter-button:not(.mod-primary):not(.mod-danger):not(.mod-warning):not(.mod-success):not(.mod-info):not(.mod-active) {{
        background: #f8f9fa !important;
        color: #3c4043 !important;
        border-color: #dadce0 !important;
    }}
    .{root_class} .jupyter-button:not(.mod-primary):not(.mod-danger):not(.mod-warning):not(.mod-success):not(.mod-info):not(.mod-active):hover {{
        background: #f1f3f4 !important;
    }}

    /* primary (Add) */
    .{root_class} .mod-primary {{
        background: #1a73e8 !important;
        color: #fff !important;
        border-color: #1a73e8 !important;
    }}
    .{root_class} .mod-primary:hover {{
        background: #1765cc !important;
        border-color: #1765cc !important;
    }}
    .{root_class} .mod-primary:disabled {{
        background: #e8eaed !important;
        color: #9aa0a6 !important;
        border-color: #e8eaed !important;
    }}

    /* danger (Remove) */
    .{root_class} .mod-danger {{
        background: #fff !important;
        color: #c5221f !important;
        border-color: #f1c8c6 !important;
    }}
    .{root_class} .mod-danger:hover {{
        background: #fce8e6 !important;
    }}
    .{root_class} .mod-danger.mod-active {{
        background: #fce8e6 !important;
        border-color: #c5221f !important;
    }}

    /* success (toggle active) */
    .{root_class} .mod-success {{
        background: #1a73e8 !important;
        color: #fff !important;
        border-color: #1a73e8 !important;
    }}
    .{root_class} .mod-success:hover {{
        background: #1765cc !important;
        border-color: #1765cc !important;
    }}

    /* toggle active state (non-danger) */
    .{root_class} .jupyter-button.mod-active:not(.mod-danger) {{
        background: #e8f0fe !important;
        color: #1a73e8 !important;
        border-color: #1a73e8 !important;
    }}

    /* inputs */
    .{root_class} input[type="number"] {{
        border-radius: 8px !important;
        border: 1px solid #dadce0 !important;
        font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif !important;
        font-size: 12px !important;
        padding: 2px 6px !important;
    }}
    .{root_class} input[type="number"]:focus {{
        border-color: #1a73e8 !important;
        outline: none !important;
    }}
    .{root_class} .widget-label {{
        font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif !important;
        font-size: 11px !important;
        color: #5f6368 !important;
        font-weight: 500 !important;
    }}
    .{root_class} .widget-checkbox {{ margin-top: 2px; }}
    .{root_class} .widget-checkbox .widget-label {{
        font-size: 10px !important;
        color: #80868b !important;
    }}

{extra_css}
</style>
"""


def make_status_setter(status_bar, *, use_gm_class: bool = True):
    """Return a ``set_status(msg, type)`` closure for the given HTML widget.

    Parameters
    ----------
    use_gm_class : bool
        If True, uses the ``gm-status gm-status-{type}`` CSS classes.
        If False, uses inline styles (compact variant used in tree / lc editors).
    """
    if use_gm_class:
        def set_status(msg: str, stype: str = "info"):
            status_bar.value = (
                f"<div class='gm-panel gm-status gm-status-{stype}'>{msg}</div>"
            )
    else:
        _colors = {
            "info":    "background:#f0f4ff;color:#1a73e8;",
            "success": "background:#e6f4ea;color:#137333;",
            "danger":  "background:#fce8e6;color:#c5221f;",
            "warn":    "background:#fef7e0;color:#b06000;",
        }

        def set_status(msg: str, stype: str = "info"):
            style_str = _colors.get(stype, _colors["info"])
            status_bar.value = (
                f"<div style='padding:3px 8px;border-radius:10px;font-size:10px;"
                f"text-align:center;{style_str}'>{msg}</div>"
            )
    return set_status
