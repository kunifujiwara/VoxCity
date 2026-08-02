"""GeoJSON builders — the stable public surface for grid/GeoDataFrame overlays.

This module is the supported home of the GeoJSON builders that were
historically private helpers in ``voxcity.geoprocessor.draw._common``.
External consumers (e.g. the VoxCity web app backend) should import them
from here; the ``draw`` editors continue to use them via ``_common``.

Why a separate module instead of re-exporting from
``voxcity.geoprocessor.draw``: measured cold-import of
``voxcity.geoprocessor.draw`` (voxcity conda env, 2026-08) takes ~1.5 s and
drags the interactive-map stack (``ipyleaflet`` + ``ipywidgets``) because the
draw editors need it at module import time. The builders here are pure
numpy/shapely code and must stay importable in headless server contexts
without that cost (measured cold-import of this module: ~0.14 s beyond
numpy/shapely, no ipyleaflet/ipywidgets).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import shapely.geometry as geom

__all__ = [
    "LC_COLORS_BY_NAME",
    "get_lc_source_colors",
    "build_building_geojson",
    "build_canopy_geojson",
    "build_lc_geojson",
]

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


def get_lc_source_colors(land_cover_source: str | None) -> dict[str, str]:
    """Return ``{class_name: hex_color}`` derived from source RGB keys.

    Uses the RGB tuples defined in ``get_land_cover_classes()`` so that the
    editor UI matches the colours used in visualiser and exporter modules.
    Falls back to :data:`LC_COLORS_BY_NAME` when *land_cover_source* is None.
    """
    if land_cover_source is None:
        return dict(LC_COLORS_BY_NAME)

    from ..utils.lc import get_land_cover_classes

    src_classes = get_land_cover_classes(land_cover_source)
    colors: dict[str, str] = {}
    for rgb, name in src_classes.items():
        if name not in colors:
            r, g, b = rgb
            colors[name] = f"#{r:02x}{g:02x}{b:02x}"
    return colors


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
        If *True*, each feature's properties will contain ``idx``, ``height``,
        ``min_height``, and ``height_estimated``.
    """
    features: list[dict] = []
    if building_gdf is None or len(building_gdf) == 0:
        return dict(_EMPTY_FC)

    def _props(idx: Any, row: Any) -> dict:
        if not include_height:
            return {}
        h = row.get("height", 0)
        h = 0.0 if (h is None or (isinstance(h, float) and math.isnan(h))) else float(h)
        try:
            min_height = float(row.get("min_height", 0.0))
        except (TypeError, ValueError):
            min_height = 0.0
        if not math.isfinite(min_height):
            min_height = 0.0
        return {
            "idx": int(idx),
            "height": h,
            "min_height": min_height,
            "height_estimated": bool(row.get("height_estimated", False)),
        }

    def _ring_has_nan(ring: Any) -> bool:
        return any(math.isnan(c) for pt in ring for c in pt)

    for idx, row in building_gdf.iterrows():
        g = row.geometry
        # A building footprint may be a single Polygon or, when its parts are
        # disjoint (e.g. an imported OBJ with separate wings or a block of
        # buildings unioned together), a MultiPolygon. Emit both so the 2D plan
        # map and any GeoJSON consumer see every footprint. Exterior rings only,
        # mirroring the single-Polygon behaviour (holes are not emitted).
        if isinstance(g, geom.Polygon):
            ring = list(g.exterior.coords)
            if _ring_has_nan(ring):
                continue
            geometry = {"type": "Polygon", "coordinates": [ring]}
        elif isinstance(g, geom.MultiPolygon):
            polys = []
            for sub in g.geoms:
                ring = list(sub.exterior.coords)
                if _ring_has_nan(ring):
                    continue
                polys.append([ring])
            if not polys:
                continue
            geometry = {"type": "MultiPolygon", "coordinates": polys}
        else:
            continue
        features.append({
            "type": "Feature",
            "id": str(idx),
            "properties": _props(idx, row),
            "geometry": geometry,
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

    from ..utils.lc import get_land_cover_classes

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
