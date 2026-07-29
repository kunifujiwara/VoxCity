"""Pure DXF parsing for auxiliary reference lines (no FastAPI/state deps).

Extracts LINE / LWPOLYLINE / POLYLINE geometry, flattened to 2D and grouped by
DXF layer, for use as non-voxelized overlay polylines.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import List, Optional

import ezdxf
import numpy as np
from ezdxf import colors as ezcolors
from ezdxf.recover import read as recover_read


class DxfParseError(ValueError):
    """Raised when the input cannot be parsed as a DXF document."""


# $INSUNITS code -> our units enum (only the ones the placement form supports).
_INSUNITS_TO_ENUM = {6: "m", 5: "cm", 4: "mm", 2: "ft", 1: "in"}
_DEFAULT_COLOR = "#888888"


@dataclass
class ParsedDxfLayer:
    name: str
    color: str                              # "#rrggbb"
    polylines: List[List[List[float]]]      # [ [ [x, y], ... ], ... ]


@dataclass
class ParsedDxf:
    layers: List[ParsedDxfLayer]
    bounds: List[List[float]]               # [[xmin, ymin], [xmax, ymax]]
    center: List[float]                     # [cx, cy]
    detected_units: Optional[str]
    insert_count: int = 0
    file_name: Optional[str] = None         # set by the upload endpoint


def _rgb_to_hex(rgb) -> Optional[str]:
    if not rgb:
        return None
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"


def _resolve_color(entity, doc) -> str:
    # Entity true-color / ACI first, then fall back to the layer's color.
    try:
        if entity.dxf.hasattr("true_color"):
            hexc = _rgb_to_hex(entity.rgb)
            if hexc:
                return hexc
        aci = entity.dxf.color
        if aci and 0 < aci < 256:
            return _rgb_to_hex(ezcolors.aci2rgb(aci)) or _DEFAULT_COLOR
    except Exception:
        pass
    try:
        layer = doc.layers.get(entity.dxf.layer)
        if layer.dxf.hasattr("true_color"):
            hexc = _rgb_to_hex(layer.rgb)
            if hexc:
                return hexc
        aci = layer.dxf.color
        if aci and 0 < aci < 256:
            return _rgb_to_hex(ezcolors.aci2rgb(aci)) or _DEFAULT_COLOR
    except Exception:
        pass
    return _DEFAULT_COLOR


def _entity_polyline(entity) -> Optional[List[List[float]]]:
    t = entity.dxftype()
    if t == "LINE":
        s, e = entity.dxf.start, entity.dxf.end
        return [[float(s.x), float(s.y)], [float(e.x), float(e.y)]]
    if t == "LWPOLYLINE":
        pts = [[float(p[0]), float(p[1])] for p in entity.get_points()]
        if entity.closed and pts:
            pts.append([pts[0][0], pts[0][1]])
        return pts or None
    if t == "POLYLINE":
        # Old-style 2D polyline backed by VERTEX entities.
        pts = [[float(v.dxf.location.x), float(v.dxf.location.y)] for v in entity.vertices]
        if entity.is_closed and pts:
            pts.append([pts[0][0], pts[0][1]])
        return pts or None
    return None


def parse_dxf(data: bytes) -> ParsedDxf:
    try:
        doc, _auditor = recover_read(io.BytesIO(data))
        msp = doc.modelspace()  # touch modelspace here so a broken doc fails now
    except Exception as exc:  # ezdxf raises several types on bad input
        raise DxfParseError(f"Could not parse DXF: {exc}") from exc

    units_code = 0
    try:
        units_code = int(doc.header.get("$INSUNITS", 0))
    except Exception:
        units_code = 0
    detected_units = _INSUNITS_TO_ENUM.get(units_code)

    insert_count = 0
    order: List[str] = []
    by_layer: dict[str, ParsedDxfLayer] = {}
    xs: List[float] = []
    ys: List[float] = []

    for entity in msp:
        t = entity.dxftype()
        if t == "INSERT":
            insert_count += 1
            continue
        pts = _entity_polyline(entity)
        if not pts:
            continue
        name = entity.dxf.layer
        if name not in by_layer:
            by_layer[name] = ParsedDxfLayer(name=name, color=_resolve_color(entity, doc), polylines=[])
            order.append(name)
        by_layer[name].polylines.append(pts)
        for x, y in pts:
            xs.append(x)
            ys.append(y)

    layers = [by_layer[n] for n in order]
    if xs and ys:
        bounds = [[min(xs), min(ys)], [max(xs), max(ys)]]
        center = [(bounds[0][0] + bounds[1][0]) / 2.0, (bounds[0][1] + bounds[1][1]) / 2.0]
    else:
        bounds = [[0.0, 0.0], [0.0, 0.0]]
        center = [0.0, 0.0]

    return ParsedDxf(
        layers=layers,
        bounds=bounds,
        center=center,
        detected_units=detected_units,
        insert_count=insert_count,
    )


def bake_polylines_to_lonlat(polylines, mat, grid_geom) -> List[List[List[float]]]:
    """Map model-XY polylines to lon/lat via a model->voxel-index matrix + grid.

    `mat` is the 4x4 from voxcity.importer.transform.build_placement_transform;
    `grid_geom` is the dict from compute_grid_geometry (origin/u_vec/v_vec/adj_mesh).
    """
    origin = np.asarray(grid_geom["origin"], dtype=float)
    u_vec = np.asarray(grid_geom["u_vec"], dtype=float)
    v_vec = np.asarray(grid_geom["v_vec"], dtype=float)
    dx = float(grid_geom["adj_mesh"][0])
    dy = float(grid_geom["adj_mesh"][1])
    out: List[List[List[float]]] = []
    for ring in polylines:
        baked: List[List[float]] = []
        for x, y in ring:
            ijk = mat @ np.array([float(x), float(y), 0.0, 1.0])
            i_f, j_f = float(ijk[0]), float(ijk[1])
            lon = origin[0] + (i_f * dx) * u_vec[0] + (j_f * dy) * v_vec[0]
            lat = origin[1] + (i_f * dx) * u_vec[1] + (j_f * dy) * v_vec[1]
            baked.append([float(lon), float(lat)])
        out.append(baked)
    return out
