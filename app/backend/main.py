"""
FastAPI application – VoxCity Web API.

Provides REST endpoints for the React frontend to:
  1. Geocode cities
  2. Generate VoxCity 3D models
  3. Run solar / view / landmark simulations
  4. Export CityLES / OBJ / NetCDF
  5. Return Plotly figures as JSON

All heavy lifting delegates to the *current* voxcity package API.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import geopandas as gpd
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .models import (
    AutoDetectSourcesRequest,
    ExportCitylesRequest,
    ExportObjRequest,
    GenerateRequest,
    GeocodeRequest,
    GeocodeResponse,
    LandmarkRequest,
    OverlayGeometryResponse,
    RectangleFromDimensions,
    RerenderRequest,
    SceneGeometryResponse,
    SimGeometryRequest,
    SolarRequest,
    StatusResponse,
    ViewRequest,
    ZoneSpec,
    ZoneStat,
    ZoneStatsRequest,
    ZoneStatsResponse,
)
from .scene_geometry import (
    build_building_highlight_buffers,
    build_building_overlay_buffers,
    build_ground_overlay_buffers,
    build_voxel_buffers,
)
from .state import app_state
from .zoning import (
    grid_xy_to_lonlat,
    mesh_face_data,
    points_in_polygon_lonlat,
    polygon_lonlat_to_cells,
    stats_from_values,
)

# ---------------------------------------------------------------------------
# VoxCity imports (validated against current package)
# ---------------------------------------------------------------------------
from voxcity.generator import get_voxcity, get_voxcity_CityGML, Voxelizer, auto_select_data_sources
from voxcity.generator.update import regenerate_voxels
from voxcity.models import (
    BuildingGrid,
    CanopyGrid,
    DemGrid,
    GridMetadata,
    LandCoverGrid,
    VoxCity as VoxCityModel,
    VoxelGrid,
)
from voxcity.simulator_gpu.solar import (
    get_building_global_solar_irradiance_using_epw,
    get_global_solar_irradiance_using_epw,
)
from voxcity.simulator_gpu.visibility import (
    get_landmark_visibility_map,
    get_surface_view_factor,
    get_view_index,
    mark_building_by_id,
)
from voxcity.exporter.cityles import export_cityles
from voxcity.exporter.obj import export_obj
from voxcity.visualizer import visualize_voxcity_plotly
from voxcity.utils.lc import get_land_cover_classes

# Ensure Taichi is initialized early (before any simulation calls).
# This must happen at module-import time so that reloaded worker
# processes always have a valid Taichi runtime.
from voxcity.simulator_gpu.init_taichi import ensure_initialized, reset as reset_taichi_flag
ensure_initialized()

# Attempt to initialize Google Earth Engine for background users.
try:
    from voxcity.downloader.gee import initialize_earth_engine

    _gee_project = os.environ.get("GEE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if _gee_project:
        initialize_earth_engine(project=_gee_project)
    else:
        initialize_earth_engine()
except Exception:
    # Keep startup resilient; EE init will be attempted again on demand.
    pass

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_OUTPUT_DIR = os.environ.get("VOXCITY_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "voxcity_output"))
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

APP_DIR = os.path.dirname(os.path.dirname(__file__))  # app/
DEFAULT_TOKYO_EPW = os.path.join(
    APP_DIR, "data", "temp", "epw_tokyo", "JPN_TK_Tokyo-Chiyoda.476620_TMYx.epw"
)
CITYGML_PATH = os.environ.get(
    "CITYGML_PATH",
    r"C:\Users\kunih\OneDrive\00_Codes\python\VoxelCity\app\data\plateau\13100_tokyo23-ku_2020_citygml_3_2_op",
)

# nDSM cache path (Cloud-Optimized GeoTIFF for canopy refinement in plateau/Japan mode)
NDSM_COG_PATH = os.path.join(APP_DIR, "data", "temp", "ndsm_cog.tif")

app = FastAPI(title="VoxCity Web API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _vertices_to_tuples(verts: List[List[float]]) -> list:
    """Convert [[lon,lat], ...] to [(lon,lat), ...]."""
    return [(v[0], v[1]) for v in verts]


def _is_japan(rectangle_vertices: List[List[float]]) -> bool:
    center_lon = (rectangle_vertices[0][0] + rectangle_vertices[2][0]) / 2.0
    center_lat = (rectangle_vertices[0][1] + rectangle_vertices[2][1]) / 2.0
    return 122.0 <= center_lon <= 154.0 and 24.0 <= center_lat <= 46.5


def _load_citygml_cache(rectangle_vertices):
    """Load cached CityGML buildings from GeoParquet/FlatGeobuf."""
    try:
        cache_dir = os.path.join(APP_DIR, "data", "temp", "citygml_cache")
        b_parquet = os.path.join(cache_dir, "buildings.parquet")
        b_fgb = os.path.join(cache_dir, "buildings.fgb")
        t_parquet = os.path.join(cache_dir, "terrain.parquet")
        t_fgb = os.path.join(cache_dir, "terrain.fgb")

        gdf = None
        t_gdf = None

        if os.path.exists(b_parquet):
            gdf = gpd.read_parquet(b_parquet)
        elif os.path.exists(b_fgb):
            gdf = gpd.read_file(b_fgb)

        if gdf is None or gdf.empty:
            return None

        # Normalize CRS
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif getattr(gdf.crs, "to_epsg", lambda: None)() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        # Ensure height column
        if "height" not in gdf.columns:
            if "height_m" in gdf.columns:
                gdf["height"] = gdf["height_m"].astype(float).fillna(0.0)
            else:
                gdf["height"] = 0.0
        else:
            gdf["height"] = gdf["height"].astype(float).fillna(0.0)

        if "id" not in gdf.columns:
            gdf["id"] = range(len(gdf))

        # Terrain
        try:
            if os.path.exists(t_parquet):
                t_gdf = gpd.read_parquet(t_parquet)
            elif os.path.exists(t_fgb):
                t_gdf = gpd.read_file(t_fgb)
        except Exception:
            t_gdf = None

        # Clip to rectangle
        from shapely.geometry import Polygon as ShapelyPolygon

        if rectangle_vertices and len(rectangle_vertices) >= 4:
            try:
                rect_poly = ShapelyPolygon([(v[0], v[1]) for v in rectangle_vertices])
                gdf = gdf[gdf.intersects(rect_poly)].copy()
                if t_gdf is not None and not t_gdf.empty:
                    t_gdf = t_gdf[t_gdf.intersects(rect_poly)].copy()
            except Exception:
                pass

        if gdf is None or gdf.empty:
            return None
        return (gdf, t_gdf)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# nDSM canopy refinement helpers (pure numpy, no external dependencies)
# ---------------------------------------------------------------------------

def _load_ndsm_grid(rectangle_vertices, meshsize: float):
    """Load nDSM grid from the cached COG GeoTIFF.

    Returns the nDSM 2-D numpy array, or None if the file is unavailable.
    """
    if not os.path.exists(NDSM_COG_PATH):
        return None
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    from pyproj import Transformer

    with rasterio.open(NDSM_COG_PATH) as src:
        if src.crs is None:
            raise ValueError("nDSM COG has no CRS")
        to_src = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        minx, miny = to_src.transform(
            float(min(v[0] for v in rectangle_vertices)),
            float(min(v[1] for v in rectangle_vertices)),
        )
        maxx, maxy = to_src.transform(
            float(max(v[0] for v in rectangle_vertices)),
            float(max(v[1] for v in rectangle_vertices)),
        )
        minx, maxx = min(minx, maxx), max(minx, maxx)
        miny, maxy = min(miny, maxy), max(miny, maxy)

        win = from_bounds(minx, miny, maxx, maxy, src.transform)
        win = win.round_offsets().round_lengths()

        width_m = maxx - minx
        height_m = maxy - miny
        out_w = max(1, int(round(width_m / float(meshsize))))
        out_h = max(1, int(round(height_m / float(meshsize))))

        arr = src.read(1, window=win, out_shape=(out_h, out_w), resampling=Resampling.average)
        nodata = src.nodata
        arr = arr.astype(float, copy=False)
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        # Normalize orientation to VoxCity convention (south-up, row 0 = south)
        if float(src.transform.e) < 0:
            arr = np.flipud(arr)
        return arr


def _align_ndsm_to_grid(ndsm_grid: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Nearest-neighbor resize nDSM to match target grid shape."""
    if ndsm_grid.shape == target_shape:
        return ndsm_grid
    H, W = ndsm_grid.shape
    Hn, Wn = target_shape
    r_idx = np.clip(np.round(np.linspace(0, H - 1, Hn)).astype(int), 0, H - 1)
    c_idx = np.clip(np.round(np.linspace(0, W - 1, Wn)).astype(int), 0, W - 1)
    return ndsm_grid[np.ix_(r_idx, c_idx)]


def _build_canopy_from_ndsm(
    ndsm_grid: np.ndarray,
    land_cover_grid: np.ndarray,
    tree_id: int,
) -> np.ndarray:
    """Extract canopy heights at tree cells; NaN elsewhere. Clamp negatives to 0."""
    canopy = np.full(ndsm_grid.shape, np.nan, dtype=float)
    tree_mask = land_cover_grid == tree_id
    ndsm = np.where(np.isnan(ndsm_grid), np.nan, np.maximum(ndsm_grid.astype(float), 0.0))
    valid = tree_mask & ~np.isnan(ndsm)
    canopy[valid] = ndsm[valid]
    return canopy


def _fix_building_leakage(
    canopy: np.ndarray,
    building_heights: np.ndarray,
    tree_mask: np.ndarray,
    tolerance_m: float = 5.0,
    replacement_m: float = 10.0,
) -> np.ndarray:
    """Fix nDSM leakage where tree cells pick up neighboring building roof heights.

    For each tree cell adjacent to a building, compare its canopy value against
    the maximum building height in a 3×3 neighborhood.  If they match within
    *tolerance_m*, the nDSM value is assumed to be building-roof leakage and is
    replaced with *replacement_m* (static tree height).
    """
    can = canopy.astype(float, copy=True)
    bh = np.asarray(building_heights, dtype=float)

    # 3×3 max-filter of building heights (vectorized, no Python loops)
    H, W = bh.shape
    padded = np.pad(bh, 1, mode="constant", constant_values=0.0)
    nearby_bld_h = np.zeros((H, W), dtype=float)
    for dr in range(3):
        for dc in range(3):
            np.maximum(nearby_bld_h, padded[dr:dr + H, dc:dc + W], out=nearby_bld_h)

    suspect = (
        tree_mask
        & np.isfinite(can)
        & (nearby_bld_h > 0)                            # adjacent to a building
        & (np.abs(can - nearby_bld_h) < tolerance_m)    # height ≈ building roof
    )
    n_fixed = int(np.count_nonzero(suspect))
    if n_fixed:
        can[suspect] = replacement_m
        print(f"[nDSM] Fixed {n_fixed} tree cells with building-height leakage")
    return can


def _refine_canopy_with_ndsm(
    voxcity_obj,
    rectangle_vertices,
    meshsize: float,
    land_cover_source: str,
    static_tree_height: float = 10.0,
) -> bool:
    """Refine canopy heights using cached nDSM COG, then regenerate voxels in-place."""
    land_cover_grid = voxcity_obj.land_cover.classes
    lc_classes = get_land_cover_classes(land_cover_source)
    name_to_id = {name: i for i, name in enumerate(lc_classes.values())}
    tree_id = name_to_id.get("Tree") or name_to_id.get("Trees") or name_to_id.get("Tree Canopy") or 4

    ndsm_grid = _load_ndsm_grid(rectangle_vertices, meshsize)
    if ndsm_grid is None:
        print("[nDSM] No nDSM COG cache found — skipping canopy refinement")
        return False

    # Align to land cover grid shape
    ndsm_aligned = _align_ndsm_to_grid(ndsm_grid, land_cover_grid.shape)
    print(f"[nDSM] Grid loaded: nDSM {ndsm_grid.shape} → aligned {ndsm_aligned.shape}")

    # Build canopy from nDSM (tree cells only)
    canopy = _build_canopy_from_ndsm(ndsm_aligned, land_cover_grid, tree_id)

    # Fill missing tree cells with static height
    tree_mask = land_cover_grid == tree_id
    missing = tree_mask & np.isnan(canopy)
    if np.any(missing):
        canopy[missing] = float(static_tree_height)

    # Fix nDSM leakage: tree cells that picked up building roof heights
    canopy = _fix_building_leakage(
        canopy,
        building_heights=voxcity_obj.buildings.heights,
        tree_mask=tree_mask,
        tolerance_m=5.0,
        replacement_m=float(static_tree_height),
    )

    # NaN → 0
    canopy = np.nan_to_num(canopy, nan=0.0)

    # Compute canopy bottom from trunk height ratio
    trunk_ratio = 11.76 / 19.98
    canopy_bottom = np.minimum(canopy * trunk_ratio, canopy)

    # Update in-place
    voxcity_obj.tree_canopy.top[:] = canopy
    if voxcity_obj.tree_canopy.bottom is not None:
        voxcity_obj.tree_canopy.bottom[:] = canopy_bottom
    else:
        voxcity_obj.tree_canopy.bottom = canopy_bottom

    # Regenerate voxels with the refined canopy
    regenerate_voxels(voxcity_obj, land_cover_source=land_cover_source, inplace=True)

    print(f"[nDSM] Canopy refinement applied — tree cells: {int(np.count_nonzero(canopy > 0))}")
    return True


def _clear_sim_caches() -> None:
    """Free old simulation results and cached figure JSON before a new run."""
    app_state.last_sim_grid = None
    app_state.last_sim_mesh = None
    app_state.last_base_fig_json = None
    app_state.last_sim_voxcity_grid = None
    gc.collect()


def _round_values(obj, ndigits: int = 2):
    """Recursively round floats in a nested structure for compact JSON."""
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, np.floating):
        return round(float(obj), ndigits)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        if np.issubdtype(obj.dtype, np.floating):
            return np.round(obj, ndigits).tolist()
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _round_values(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_round_values(x, ndigits) for x in obj]
    return obj


def _compact_fig_json(fig) -> str:
    """Serialize a Plotly figure with rounded floats and no whitespace."""
    fig_dict = fig.to_dict()
    return json.dumps(_round_values(fig_dict), separators=(',', ':'))


def _make_plotly_json(
    voxcity_grid: np.ndarray,
    meshsize: float,
    plot_kwargs: dict,
) -> str:
    """Render a voxcity plotly figure and return the JSON string."""
    fig = visualize_voxcity_plotly(
        voxcity_grid,
        meshsize,
        downsample=1,
        show=False,
        return_fig=True,
        **plot_kwargs,
    )
    if fig is None:
        raise HTTPException(
            status_code=500,
            detail="Visualization engine returned no figure.",
        )
    fig_json = _compact_fig_json(fig)
    approx_mb = len(fig_json.encode("utf-8")) / (1024.0 * 1024.0)
    print(f"[_make_plotly_json] {approx_mb:.1f} MB")
    if approx_mb > 500:
        raise HTTPException(
            status_code=413,
            detail=(
                "The 3D visualization is too large to display "
                f"({approx_mb:.0f} MB). "
                "Please decrease the target area or increase the mesh size."
            ),
        )
    return fig_json


# ---------------------------------------------------------------------------
# Helpers – fast rerender (sim overlay only, skip voxel face extraction)
# ---------------------------------------------------------------------------

_VOXEL_TRACE_NAMES = frozenset({'+x', '-x', '+y', '-y', '+z', '-z'})


def _mpl_cmap_to_plotly_colorscale(cmap_name: str, n: int = 256) -> list:
    """Convert a matplotlib colormap name to a Plotly colorscale list."""
    import matplotlib.cm as mcm
    try:
        cmap = mcm.get_cmap(cmap_name)
    except Exception:
        cmap = mcm.get_cmap('viridis')
    scale = []
    for i in range(n):
        x = i / (n - 1)
        r, g, b, _ = cmap(x)
        scale.append([x, f"rgb({int(255*r)},{int(255*g)},{int(255*b)})"])
    return scale


def _build_sim_overlay_traces(
    sim_target: str,
    sim_type: str,
    sim_grid,
    sim_mesh,
    voxcity_grid: np.ndarray,
    meshsize: float,
    colormap: str,
    vmin: Optional[float],
    vmax: Optional[float],
    view_point_height: float,
    colorbar_title: str = "",
) -> list:
    """Build only the simulation overlay Plotly traces (no voxels).

    Returns a list of trace dicts ready to splice into a cached figure.
    """
    import plotly.graph_objects as go
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    from voxcity.geoprocessor.mesh import create_sim_surface_mesh

    fig = go.Figure()

    if sim_target == "ground" and sim_grid is not None:
        # -- Derive DEM from voxel grid (same logic as renderer.py) --
        if voxcity_grid is not None and voxcity_grid.ndim == 3:
            lc_mask = (voxcity_grid >= 1)
            k_indices = np.arange(voxcity_grid.shape[2])
            masked_k = np.where(lc_mask, k_indices[None, None, :], -1)
            k_top_grid = np.max(masked_k, axis=2)
            k_top_grid = np.maximum(k_top_grid, 0)
            dem_norm = np.flipud(k_top_grid.astype(float) * float(meshsize))
        else:
            dem_norm = np.zeros_like(sim_grid, dtype=float)

        # -- z-offset (same as renderer.py) --
        z_off = float(meshsize) + max(float(view_point_height), float(meshsize))

        sim_vals = np.asarray(sim_grid, dtype=float)
        finite = np.isfinite(sim_vals)
        vmin_g = vmin if vmin is not None else (float(np.nanmin(sim_vals[finite])) if np.any(finite) else 0.0)
        vmax_g = vmax if vmax is not None else (float(np.nanmax(sim_vals[finite])) if np.any(finite) else 1.0)

        sim_mesh_obj = create_sim_surface_mesh(
            sim_grid, dem_norm,
            meshsize=meshsize, z_offset=z_off,
            cmap_name=colormap, vmin=vmin_g, vmax=vmax_g,
        )

        if sim_mesh_obj is not None and getattr(sim_mesh_obj, 'vertices', None) is not None:
            V = np.asarray(sim_mesh_obj.vertices)
            F = np.asarray(sim_mesh_obj.faces)
            facecolor = None
            try:
                colors_rgba = np.asarray(sim_mesh_obj.visual.face_colors)
                if colors_rgba.ndim == 2 and colors_rgba.shape[0] == len(F):
                    facecolor = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors_rgba]
            except Exception:
                pass

            lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0, fresnel=0.0)
            cx = float((V[:, 0].min() + V[:, 0].max()) * 0.5)
            cy = float((V[:, 1].min() + V[:, 1].max()) * 0.5)
            lx = cx + (V[:, 0].max() - V[:, 0].min() + meshsize) * 0.9
            ly = cy + (V[:, 1].max() - V[:, 1].min() + meshsize) * 0.6
            lz = float((V[:, 2].min() + V[:, 2].max()) * 0.5) + (V[:, 2].max() - V[:, 2].min() + meshsize) * 1.4

            fig.add_trace(go.Mesh3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2],
                i=F[:, 0], j=F[:, 1], k=F[:, 2],
                facecolor=facecolor,
                color=None if facecolor is not None else 'rgb(200,200,200)',
                opacity=0.95, flatshading=False,
                lighting=lighting,
                lightposition=dict(x=lx, y=ly, z=lz),
                name='sim_surface',
                meta=dict(sim_overlay=True),
            ))

            colorscale_g = _mpl_cmap_to_plotly_colorscale(colormap)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode='markers',
                marker=dict(
                    size=0.1, color=[vmin_g, vmax_g],
                    colorscale=colorscale_g, cmin=vmin_g, cmax=vmax_g,
                    colorbar=dict(title=colorbar_title or 'Ground', len=0.5, y=0.2), showscale=True,
                ),
                showlegend=False, hoverinfo='skip',
            ))

    elif sim_target == "building" and sim_mesh is not None:
        if getattr(sim_mesh, 'vertices', None) is not None:
            Vb = np.asarray(sim_mesh.vertices)
            Fb = np.asarray(sim_mesh.faces)
            value_name = "global" if sim_type == "solar" else "view_factor_values"

            values = None
            if hasattr(sim_mesh, 'metadata') and isinstance(sim_mesh.metadata, dict):
                values = sim_mesh.metadata.get(value_name)
            if values is not None:
                values = np.asarray(values)

            face_vals = None
            if values is not None and len(values) == len(Fb):
                face_vals = values.astype(float)
            elif values is not None and len(values) == len(Vb):
                vals_v = values.astype(float)
                face_vals = np.nanmean(vals_v[Fb], axis=1)

            facecolor = None
            vmin_b, vmax_b = 0.0, 1.0
            if face_vals is not None:
                finite = np.isfinite(face_vals)
                vmin_b = vmin if vmin is not None else (float(np.nanmin(face_vals[finite])) if np.any(finite) else 0.0)
                vmax_b = vmax if vmax is not None else (float(np.nanmax(face_vals[finite])) if np.any(finite) else 1.0)
                norm_b = mcolors.Normalize(vmin=vmin_b, vmax=vmax_b)
                cmap_b = mcm.get_cmap(colormap)
                colors_rgba = np.zeros((len(Fb), 4), dtype=float)
                colors_rgba[finite] = cmap_b(norm_b(face_vals[finite]))
                nan_rgba = np.array(mcolors.to_rgba('gray'))
                colors_rgba[~finite] = nan_rgba
                facecolor = [f"rgb({int(255*c[0])},{int(255*c[1])},{int(255*c[2])})" for c in colors_rgba]

            lighting_b = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=0.0, fresnel=0.0)
            cx = float((Vb[:, 0].min() + Vb[:, 0].max()) * 0.5)
            cy = float((Vb[:, 1].min() + Vb[:, 1].max()) * 0.5)
            lx = cx + (Vb[:, 0].max() - Vb[:, 0].min() + meshsize) * 0.9
            ly = cy + (Vb[:, 1].max() - Vb[:, 1].min() + meshsize) * 0.6
            lz = float((Vb[:, 2].min() + Vb[:, 2].max()) * 0.5) + (Vb[:, 2].max() - Vb[:, 2].min() + meshsize) * 1.4

            fig.add_trace(go.Mesh3d(
                x=Vb[:, 0], y=Vb[:, 1], z=Vb[:, 2],
                i=Fb[:, 0], j=Fb[:, 1], k=Fb[:, 2],
                facecolor=facecolor if facecolor is not None else None,
                color=None if facecolor is not None else 'rgb(200,200,200)',
                opacity=1.0, flatshading=False,
                lighting=lighting_b,
                lightposition=dict(x=lx, y=ly, z=lz),
                name=value_name if facecolor is not None else 'building_mesh',
                meta=dict(sim_overlay=True),
            ))

            colorscale_b = _mpl_cmap_to_plotly_colorscale(colormap)
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None], mode='markers',
                marker=dict(
                    size=0.1, color=[vmin_b, vmax_b],
                    colorscale=colorscale_b, cmin=vmin_b, cmax=vmax_b,
                    colorbar=dict(title=colorbar_title or value_name, len=0.5, y=0.8), showscale=True,
                ),
                showlegend=False, hoverinfo='skip',
            ))

    # Extract trace dicts from the temporary figure
    return _round_values(fig.to_dict().get('data', []))


# ---------------------------------------------------------------------------
# Helpers – Taichi / GPU cache management
# ---------------------------------------------------------------------------

def _reset_taichi_and_caches():
    """Reset all Taichi fields and GPU caches so a new model can be simulated."""
    import taichi as ti
    from voxcity.simulator_gpu.visibility.integration import clear_visibility_cache
    from voxcity.simulator_gpu.solar.integration.caching import clear_all_caches as clear_all_solar_caches

    # Clear domain / radiation model caches (does not touch ti runtime)
    clear_visibility_cache()
    try:
        clear_all_solar_caches()
    except Exception:
        pass

    # Reset Taichi runtime and re-initialise
    try:
        ti.reset()
    except Exception:
        pass
    reset_taichi_flag()
    try:
        ti.init(arch=ti.cuda, default_fp=ti.f32, default_ip=ti.i32)
    except Exception:
        try:
            ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32)
        except Exception:
            pass

    # Clear last-sim state
    app_state.last_sim_type = None
    app_state.last_sim_target = None
    app_state.last_sim_grid = None
    app_state.last_sim_mesh = None
    app_state.last_sim_voxcity_grid = None
    app_state.last_sim_view_point_height = 1.5
    app_state.last_base_fig_json = None
    app_state.last_hidden_classes = None
    app_state.last_colorbar_title = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "has_model": app_state.has_model}


@app.post("/api/reset")
async def reset_session():
    """Reset backend state so the user can start fresh (e.g. after page reload)."""
    try:
        _reset_taichi_and_caches()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/geocode", response_model=GeocodeResponse)
async def geocode_city(req: GeocodeRequest):
    """Geocode a city name using Nominatim."""
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": req.city_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "VoxCityApp/1.0"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            raise HTTPException(status_code=404, detail="City not found")
        item = data[0]
        lat, lon = float(item["lat"]), float(item["lon"])
        bbox = None
        if item.get("boundingbox") and len(item["boundingbox"]) == 4:
            s, n, w, e = map(float, item["boundingbox"])
            bbox = (w, s, e, n)
        return GeocodeResponse(lat=lat, lon=lon, bbox=bbox)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rectangle-from-dimensions")
async def rectangle_from_dimensions(req: RectangleFromDimensions):
    """Compute rectangle vertices from a center + width/height in meters."""
    from geopy import distance

    lat_c, lon_c = req.center_lat, req.center_lon
    north = distance.distance(meters=req.height_m / 2.0).destination((lat_c, lon_c), bearing=0)
    south = distance.distance(meters=req.height_m / 2.0).destination((lat_c, lon_c), bearing=180)
    east = distance.distance(meters=req.width_m / 2.0).destination((lat_c, lon_c), bearing=90)
    west = distance.distance(meters=req.width_m / 2.0).destination((lat_c, lon_c), bearing=270)

    vertices = [
        [west.longitude, south.latitude],
        [west.longitude, north.latitude],
        [east.longitude, north.latitude],
        [east.longitude, south.latitude],
    ]
    return {"vertices": vertices}


@app.post("/api/auto-detect-sources")
async def detect_sources(req: AutoDetectSourcesRequest):
    """Auto-detect the best data sources for a given area."""
    try:
        verts = _vertices_to_tuples(req.rectangle_vertices)
        sources = auto_select_data_sources(verts)
        return sources
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate")
async def generate_model(req: GenerateRequest):
    """Generate a VoxCity model.

    The function calls the *current* voxcity API which returns a VoxCity
    dataclass directly (no longer a tuple of arrays).
    """
    try:
        # Reset Taichi & GPU caches so fields from the previous model don't
        # conflict with the new grid dimensions.
        _reset_taichi_and_caches()

        rectangle_vertices = _vertices_to_tuples(req.rectangle_vertices)
        output_dir = os.path.join(BASE_OUTPUT_DIR, "test")
        os.makedirs(output_dir, exist_ok=True)

        kwargs: Dict[str, Any] = {
            "building_complement_height": req.building_complement_height,
            "overlapping_footprint": req.overlapping_footprint,
            "output_dir": output_dir,
            "dem_interpolation": req.dem_interpolation,
            "static_tree_height": req.static_tree_height,
            "gridvis": False,
            "mapvis": False,
        }

        if req.mode == "plateau":
            # ── PLATEAU mode ──────────────────────────────────────
            is_jp = _is_japan(req.rectangle_vertices)
            land_cover_source = "OpenEarthMapJapan" if is_jp else (req.land_cover_source or "OpenStreetMap")

            kwargs["building_complementary_source"] = "None"
            kwargs["complement_building_footprints"] = True

            # Attempt cached CityGML first
            cached = None
            if req.use_citygml_cache:
                cached = _load_citygml_cache(req.rectangle_vertices)

            if cached is not None:
                cached_buildings, cached_terrain = (
                    cached if isinstance(cached, tuple) else (cached, None)
                )
                # Use terrain GeoDataFrame for DEM when available;
                # fall back to flat DEM when no terrain cache exists.
                _dem_src = "GeoDataFrame" if cached_terrain is not None else "Flat"
                voxcity_result = get_voxcity(
                    rectangle_vertices,
                    meshsize=req.meshsize,
                    building_source="GeoDataFrame",
                    land_cover_source=land_cover_source,
                    canopy_height_source=req.canopy_height_source or "Static",
                    dem_source=_dem_src,
                    building_gdf=cached_buildings,
                    terrain_gdf=cached_terrain,
                    **kwargs,
                )
            else:
                voxcity_result = get_voxcity_CityGML(
                    rectangle_vertices,
                    land_cover_source,
                    req.canopy_height_source or "Static",
                    req.meshsize,
                    citygml_path=CITYGML_PATH,
                    **kwargs,
                )
        else:
            # ── Normal mode ───────────────────────────────────────
            # Pass None for any source not specified → get_voxcity auto-selects
            voxcity_result = get_voxcity(
                rectangle_vertices,
                meshsize=req.meshsize,
                building_source=req.building_source,
                land_cover_source=req.land_cover_source,
                canopy_height_source=req.canopy_height_source,
                dem_source=req.dem_source,
                building_complementary_source=req.building_complementary_source,
                **kwargs,
            )

        # Resolve effective land_cover_source for state
        effective_lc = (
            voxcity_result.extras.get("land_cover_source")
            or voxcity_result.extras.get("selected_sources", {}).get("land_cover_source")
            or req.land_cover_source
            or "OpenStreetMap"
        )

        # Optional nDSM canopy refinement (plateau mode, Japan, OpenEarthMapJapan)
        _ndsm_conditions = {
            "mode==plateau": req.mode == "plateau",
            "use_ndsm_canopy": req.use_ndsm_canopy,
            "is_japan": _is_japan(req.rectangle_vertices),
            "OEMJ_lc": "OpenEarthMapJapan" in str(effective_lc),
        }
        if all(_ndsm_conditions.values()):
            try:
                applied = _refine_canopy_with_ndsm(
                    voxcity_result,
                    rectangle_vertices,
                    meshsize=req.meshsize,
                    land_cover_source=effective_lc,
                    static_tree_height=req.static_tree_height,
                )
                if not applied:
                    print("[nDSM] _refine_canopy_with_ndsm returned False (see above for reason)")
            except Exception as ndsm_err:
                print(f"[nDSM] Canopy refinement failed (non-fatal): {ndsm_err}")
        else:
            _failed = [k for k, v in _ndsm_conditions.items() if not v]
            print(f"[nDSM] Skipping canopy refinement — conditions not met: {_failed}")

        # Store in app state
        app_state.store_generation_result(
            voxcity_obj=voxcity_result,
            meshsize=req.meshsize,
            rectangle_vertices=req.rectangle_vertices,
            land_cover_source=effective_lc,
        )

        # Build 3D preview
        fig_json = _make_plotly_json(
            app_state.voxcity.voxels.classes,
            req.meshsize,
            {"title": "VoxCity 3D"},
        )

        return {
            "status": "ok",
            "grid_shape": list(app_state.voxcity.voxels.classes.shape),
            "meshsize": req.meshsize,
            "figure_json": fig_json,
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/solar")
async def run_solar(req: SolarRequest):
    """Run solar radiation analysis."""
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")

    _clear_sim_caches()

    try:
        voxcity = app_state.voxcity

        # Resolve EPW
        epw_path = None
        if req.epw_source == "default" and os.path.exists(DEFAULT_TOKYO_EPW):
            epw_path = DEFAULT_TOKYO_EPW
        if not epw_path:
            raise HTTPException(status_code=400, detail="No EPW file available")

        output_dir = os.path.join(BASE_OUTPUT_DIR, "solar")
        os.makedirs(output_dir, exist_ok=True)

        if req.analysis_target == "ground":
            solar_kwargs: Dict[str, Any] = {
                "download_nearest_epw": False,
                "view_point_height": req.view_point_height,
                "tree_k": 0.6,
                "tree_lad": 0.5,
                "dem_grid": voxcity.dem.elevation,
                "colormap": req.colormap,
                "obj_export": False,
                "output_directory": output_dir,
                "alpha": 1.0,
                "vmin": req.vmin,
                "vmax": req.vmax,
                "epw_file_path": epw_path,
            }

            temporal_mode = req.calc_type  # "instantaneous" or "cumulative"
            if temporal_mode == "instantaneous":
                solar_kwargs["calc_time"] = req.calc_time or "01-01 12:00:00"
            else:
                solar_kwargs["start_time"] = req.start_time
                solar_kwargs["end_time"] = req.end_time

            solar_grid = get_global_solar_irradiance_using_epw(
                voxcity,
                temporal_mode=temporal_mode,
                direct_normal_irradiance_scaling=1.0,
                diffuse_irradiance_scaling=1.0,
                **solar_kwargs,
            )

            # Store for re-rendering
            app_state.last_sim_type = "solar"
            app_state.last_sim_target = "ground"
            app_state.last_sim_grid = solar_grid
            app_state.last_sim_mesh = None
            app_state.last_sim_voxcity_grid = voxcity.voxels.classes
            app_state.last_sim_view_point_height = req.view_point_height
            app_state.last_colorbar_title = (
                "Inst. Solar Irradiance (W/m²)" if req.calc_type == "instantaneous"
                else "Cum. Solar Irradiance (Wh/m²)"
            )

            # Build overlay figure
            voxcity_grid = voxcity.voxels.classes
            present_classes = np.unique(voxcity_grid[voxcity_grid != 0]).tolist()
            classes_include = [int(c) for c in present_classes if int(c) not in req.hidden_classes]

            fig_json = _make_plotly_json(
                voxcity_grid,
                app_state.meshsize,
                {
                    "classes": classes_include,
                    "voxel_color_map": "grayscale",
                    "ground_sim_grid": solar_grid,
                    "ground_dem_grid": voxcity.dem.elevation,
                    "ground_view_point_height": req.view_point_height,
                    "ground_z_offset": max(req.view_point_height, app_state.meshsize),
                    "ground_colormap": req.colormap,
                    "ground_vmin": req.vmin,
                    "ground_vmax": req.vmax,
                    "sim_surface_opacity": 0.95,
                    "title": "Solar overlay",
                    "ground_colorbar_title": app_state.last_colorbar_title,
                },
            )
            app_state.last_base_fig_json = fig_json
            app_state.last_hidden_classes = list(req.hidden_classes)

            return {"status": "ok", "figure_json": fig_json}

        else:  # building surfaces
            irradiance_kwargs: Dict[str, Any] = {
                "download_nearest_epw": False,
                "epw_file_path": epw_path,
            }

            if req.calc_type == "instantaneous":
                irradiance_kwargs["calc_type"] = "instantaneous"
                irradiance_kwargs["calc_time"] = req.calc_time or "01-01 12:00:00"
            else:
                irradiance_kwargs["calc_type"] = "cumulative"
                irradiance_kwargs["period_start"] = req.start_time
                irradiance_kwargs["period_end"] = req.end_time
                irradiance_kwargs["use_sky_patches"] = True

            irradiance = get_building_global_solar_irradiance_using_epw(
                voxcity, **irradiance_kwargs
            )

            if irradiance is None or getattr(irradiance, 'vertices', None) is None:
                raise HTTPException(
                    status_code=500,
                    detail="Building solar simulation returned no mesh – the model may lack building geometry.",
                )

            # Store for re-rendering
            app_state.last_sim_type = "solar"
            app_state.last_sim_target = "building"
            app_state.last_sim_grid = None
            app_state.last_sim_mesh = irradiance
            app_state.last_sim_voxcity_grid = voxcity.voxels.classes
            app_state.last_colorbar_title = (
                "Inst. Global Irradiance (W/m²)" if req.calc_type == "instantaneous"
                else "Cum. Global Irradiance (Wh/m²)"
            )

            voxcity_grid = voxcity.voxels.classes
            present_classes = np.unique(voxcity_grid[voxcity_grid != 0]).tolist()
            classes_include = [int(c) for c in present_classes if int(c) not in req.hidden_classes]

            fig_json = _make_plotly_json(
                voxcity_grid,
                app_state.meshsize,
                {
                    "classes": classes_include,
                    "voxel_color_map": "grayscale",
                    "building_sim_mesh": irradiance,
                    "building_value_name": "global",
                    "building_colorbar_title": app_state.last_colorbar_title,
                    "building_colormap": req.colormap,
                    "building_vmin": req.vmin,
                    "building_vmax": req.vmax,
                    "building_opacity": 1.0,
                    "building_shaded": False,
                    "render_voxel_buildings": False,
                    "title": "Building Surface Solar (Global)",
                },
            )
            app_state.last_base_fig_json = fig_json
            app_state.last_hidden_classes = list(req.hidden_classes)

            return {"status": "ok", "figure_json": fig_json}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/view")
async def run_view(req: ViewRequest):
    """Run view index analysis."""
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")

    _clear_sim_caches()

    try:
        voxcity = app_state.voxcity
        output_dir = os.path.join(BASE_OUTPUT_DIR, "view")
        os.makedirs(output_dir, exist_ok=True)

        if req.analysis_target == "ground":
            view_kwargs: Dict[str, Any] = {
                "view_point_height": req.view_point_height,
                "dem_grid": voxcity.dem.elevation,
                "obj_export": req.export_obj,
                "output_directory": output_dir,
                "colormap": req.colormap,
                "vmin": req.vmin,
                "vmax": req.vmax,
                "N_azimuth": req.n_azimuth,
                "N_elevation": req.n_elevation,
                "elevation_min_degrees": req.elevation_min_degrees,
                "elevation_max_degrees": req.elevation_max_degrees,
            }

            if req.view_type == "custom":
                if not req.custom_classes:
                    raise HTTPException(status_code=400, detail="Select at least one class")
                view_grid = get_view_index(
                    voxcity,
                    mode=None,
                    hit_values=tuple(req.custom_classes),
                    inclusion_mode=req.inclusion_mode,
                    **view_kwargs,
                )
            else:
                mode = "green" if req.view_type == "green" else "sky"
                view_grid = get_view_index(voxcity, mode=mode, **view_kwargs)

            # Store for re-rendering
            app_state.last_sim_type = "view"
            app_state.last_sim_target = "ground"
            app_state.last_sim_grid = view_grid
            app_state.last_sim_mesh = None
            app_state.last_sim_voxcity_grid = voxcity.voxels.classes
            app_state.last_sim_view_point_height = req.view_point_height
            app_state.last_colorbar_title = "View Index"

            voxcity_grid = voxcity.voxels.classes
            present_classes = np.unique(voxcity_grid[voxcity_grid != 0]).tolist()
            classes_include = [int(c) for c in present_classes if int(c) not in req.hidden_classes]

            fig_json = _make_plotly_json(
                voxcity_grid,
                app_state.meshsize,
                {
                    "classes": classes_include,
                    "voxel_color_map": "grayscale",
                    "ground_sim_grid": view_grid,
                    "ground_dem_grid": voxcity.dem.elevation,
                    "ground_view_point_height": req.view_point_height,
                    "ground_z_offset": max(req.view_point_height, app_state.meshsize),
                    "ground_colormap": req.colormap,
                    "ground_vmin": req.vmin,
                    "ground_vmax": req.vmax,
                    "sim_surface_opacity": 0.95,
                    "title": req.view_type.replace("_", " ").title() + " View Index",
                    "ground_colorbar_title": "View Index",
                },
            )
            app_state.last_base_fig_json = fig_json
            app_state.last_hidden_classes = list(req.hidden_classes)

            return {"status": "ok", "figure_json": fig_json}

        else:  # building surfaces
            target_values = tuple(req.custom_classes) if req.view_type == "custom" else (
                (-2,) if req.view_type == "green" else (0,)
            )
            inclusion_mode = req.inclusion_mode if req.view_type == "custom" else (
                True if req.view_type == "green" else False
            )

            mesh = get_surface_view_factor(
                voxcity,
                target_values=target_values,
                inclusion_mode=inclusion_mode,
                colormap=req.colormap,
                vmin=req.vmin,
                vmax=req.vmax,
                obj_export=req.export_obj,
                output_directory=output_dir,
                N_azimuth=req.n_azimuth,
                N_elevation=req.n_elevation,
                ray_sampling="fibonacci",
                N_rays=req.n_azimuth * req.n_elevation,
            )

            if mesh is None:
                raise HTTPException(status_code=500, detail="No building surfaces found")

            # Store for re-rendering
            app_state.last_sim_type = "view"
            app_state.last_sim_target = "building"
            app_state.last_sim_grid = None
            app_state.last_sim_mesh = mesh
            app_state.last_sim_voxcity_grid = voxcity.voxels.classes
            app_state.last_colorbar_title = "View Factor"

            voxcity_grid = voxcity.voxels.classes
            present_classes = np.unique(voxcity_grid[voxcity_grid != 0]).tolist()
            classes_include = [int(c) for c in present_classes if int(c) not in req.hidden_classes]

            fig_json = _make_plotly_json(
                voxcity_grid,
                app_state.meshsize,
                {
                    "classes": classes_include,
                    "voxel_color_map": "grayscale",
                    "building_sim_mesh": mesh,
                    "building_value_name": "view_factor_values",
                    "building_colorbar_title": "View Factor",
                    "building_colormap": req.colormap,
                    "building_vmin": req.vmin,
                    "building_vmax": req.vmax,
                    "render_voxel_buildings": False,
                    "title": "Surface View Factor",
                },
            )
            app_state.last_base_fig_json = fig_json
            app_state.last_hidden_classes = list(req.hidden_classes)

            return {"status": "ok", "figure_json": fig_json}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/landmark")
async def run_landmark(req: LandmarkRequest):
    """Run landmark visibility analysis."""
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")

    _clear_sim_caches()

    try:
        voxcity = app_state.voxcity
        output_dir = os.path.join(BASE_OUTPUT_DIR, "landmark")
        os.makedirs(output_dir, exist_ok=True)

        voxcity_grid = voxcity.voxels.classes.copy()

        if len(req.landmark_ids) > 0:
            # Mark specified buildings as landmarks
            voxcity_grid = mark_building_by_id(
                voxcity_grid,
                voxcity.buildings.ids,
                req.landmark_ids,
                -30,
            )
        else:
            # Use center-building fallback via get_landmark_visibility_map
            building_gdf = voxcity.extras.get("building_gdf")
            vis_map, voxcity_grid = get_landmark_visibility_map(
                voxcity,
                building_gdf=building_gdf,
                output_directory=output_dir,
            )

        if req.analysis_target == "ground":
            # Build a new temporary VoxCity with marked voxels for view_index
            marked_voxcity = VoxCityModel(
                voxels=VoxelGrid(classes=voxcity_grid, meta=voxcity.voxels.meta),
                buildings=voxcity.buildings,
                land_cover=voxcity.land_cover,
                dem=voxcity.dem,
                tree_canopy=voxcity.tree_canopy,
                extras=voxcity.extras,
            )

            landmark_grid = get_view_index(
                marked_voxcity,
                mode=None,
                hit_values=(-30,),
                inclusion_mode=True,
                view_point_height=req.view_point_height,
                dem_grid=voxcity.dem.elevation,
                N_azimuth=req.n_azimuth,
                N_elevation=req.n_elevation,
                elevation_min_degrees=req.elevation_min_degrees,
                elevation_max_degrees=req.elevation_max_degrees,
            )

            # Zeros → NaN for visualization
            lg = np.asarray(landmark_grid, dtype=float)
            lg[lg == 0.0] = np.nan

            # Store for re-rendering
            app_state.last_sim_type = "landmark"
            app_state.last_sim_target = "ground"
            app_state.last_sim_grid = lg
            app_state.last_sim_mesh = None
            app_state.last_sim_voxcity_grid = voxcity_grid
            app_state.last_sim_view_point_height = req.view_point_height
            app_state.last_colorbar_title = "Visibility"

            present_classes = np.unique(voxcity_grid[voxcity_grid != 0]).tolist()
            classes_include = [int(c) for c in present_classes if int(c) not in req.hidden_classes]

            fig_json = _make_plotly_json(
                voxcity_grid,
                app_state.meshsize,
                {
                    "classes": classes_include,
                    "voxel_color_map": "grayscale",
                    "ground_sim_grid": lg,
                    "ground_dem_grid": voxcity.dem.elevation,
                    "ground_view_point_height": req.view_point_height,
                    "ground_z_offset": max(req.view_point_height, app_state.meshsize),
                    "ground_colormap": req.colormap,
                    "ground_vmin": req.vmin,
                    "ground_vmax": req.vmax,
                    "sim_surface_opacity": 0.95,
                    "title": "Landmark Visibility (Ground)",
                    "ground_colorbar_title": "Visibility",
                },
            )
            app_state.last_base_fig_json = fig_json
            app_state.last_hidden_classes = list(req.hidden_classes)

            return {"status": "ok", "figure_json": fig_json}

        else:  # building surfaces
            marked_voxcity = VoxCityModel(
                voxels=VoxelGrid(classes=voxcity_grid, meta=voxcity.voxels.meta),
                buildings=voxcity.buildings,
                land_cover=voxcity.land_cover,
                dem=voxcity.dem,
                tree_canopy=voxcity.tree_canopy,
                extras=voxcity.extras,
            )

            landmark_mesh = get_surface_view_factor(
                marked_voxcity,
                target_values=(-30,),
                inclusion_mode=True,
                progress_report=True,
                colormap=req.colormap,
                vmin=req.vmin,
                vmax=req.vmax,
                obj_export=False,
                output_directory=output_dir,
                N_azimuth=req.n_azimuth,
                N_elevation=req.n_elevation,
                ray_sampling="fibonacci",
                N_rays=req.n_azimuth * req.n_elevation,
            )

            if landmark_mesh is None:
                raise HTTPException(status_code=500, detail="No surfaces generated")

            # Store for re-rendering
            app_state.last_sim_type = "landmark"
            app_state.last_sim_target = "building"
            app_state.last_sim_grid = None
            app_state.last_sim_mesh = landmark_mesh
            app_state.last_sim_voxcity_grid = voxcity_grid
            app_state.last_colorbar_title = "Visibility"

            present_classes = np.unique(voxcity_grid[voxcity_grid != 0]).tolist()
            classes_include = [int(c) for c in present_classes if int(c) not in req.hidden_classes]

            fig_json = _make_plotly_json(
                voxcity_grid,
                app_state.meshsize,
                {
                    "classes": classes_include,
                    "voxel_color_map": "grayscale",
                    "building_sim_mesh": landmark_mesh,
                    "building_value_name": "view_factor_values",
                    "building_colorbar_title": "Visibility",
                    "building_colormap": req.colormap,
                    "building_vmin": req.vmin,
                    "building_vmax": req.vmax,
                    "render_voxel_buildings": False,
                    "title": "Landmark Visibility (Surface)",
                },
            )
            app_state.last_base_fig_json = fig_json
            app_state.last_hidden_classes = list(req.hidden_classes)

            return {"status": "ok", "figure_json": fig_json}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/rerender")
async def rerender(req: RerenderRequest):
    """Re-render the last simulation result with new visualization settings.

    This avoids re-running the expensive simulation when only the colormap,
    value range, or hidden classes change.
    """
    if app_state.last_sim_type is None:
        raise HTTPException(status_code=400, detail="No simulation result to re-render")

    try:
        voxcity_grid = app_state.last_sim_voxcity_grid
        meshsize = app_state.meshsize

        hidden_set = set(req.hidden_classes)
        cached_hidden = set(app_state.last_hidden_classes or [])

        # --- Fast path: hidden classes unchanged → rebuild only sim overlay ---
        if hidden_set == cached_hidden and app_state.last_base_fig_json and app_state.last_base_fig_json != "{}":
            fig_dict = json.loads(app_state.last_base_fig_json)
            voxel_traces = [t for t in fig_dict.get('data', [])
                           if t.get('name') in _VOXEL_TRACE_NAMES]

            sim_traces = _build_sim_overlay_traces(
                sim_target=app_state.last_sim_target,
                sim_type=app_state.last_sim_type,
                sim_grid=app_state.last_sim_grid,
                sim_mesh=app_state.last_sim_mesh,
                voxcity_grid=voxcity_grid,
                meshsize=meshsize,
                colormap=req.colormap,
                vmin=req.vmin,
                vmax=req.vmax,
                view_point_height=app_state.last_sim_view_point_height,
                colorbar_title=app_state.last_colorbar_title or "",
            )

            fig_dict['data'] = voxel_traces + sim_traces
            fig_json = json.dumps(fig_dict, separators=(',', ':'))
            return {"status": "ok", "figure_json": fig_json}

        # --- Slow path: hidden classes changed → re-render voxels at cached stride ---
        present_classes = np.unique(voxcity_grid[voxcity_grid != 0]).tolist()
        classes_include = [int(c) for c in present_classes if int(c) not in req.hidden_classes]

        if app_state.last_sim_target == "ground":
            sim_grid = app_state.last_sim_grid
            title_map = {
                "solar": "Solar overlay",
                "view": "View Index",
                "landmark": "Landmark Visibility (Ground)",
            }
            fig_json = _make_plotly_json(
                voxcity_grid,
                meshsize,
                {
                    "classes": classes_include,
                    "voxel_color_map": "grayscale",
                    "ground_sim_grid": sim_grid,
                    "ground_dem_grid": app_state.voxcity.dem.elevation,
                    "ground_view_point_height": app_state.last_sim_view_point_height,
                    "ground_z_offset": max(app_state.last_sim_view_point_height, meshsize),
                    "ground_colormap": req.colormap,
                    "ground_vmin": req.vmin,
                    "ground_vmax": req.vmax,
                    "sim_surface_opacity": 0.95,
                    "title": title_map.get(app_state.last_sim_type, "Simulation"),
                    "ground_colorbar_title": app_state.last_colorbar_title,
                },
            )
        else:
            sim_mesh = app_state.last_sim_mesh
            value_name = (
                "global" if app_state.last_sim_type == "solar" else "view_factor_values"
            )
            title_map = {
                "solar": "Building Surface Solar (Global)",
                "view": "Surface View Factor",
                "landmark": "Landmark Visibility (Surface)",
            }
            plot_kwargs: Dict[str, Any] = {
                "classes": classes_include,
                "voxel_color_map": "grayscale",
                "building_sim_mesh": sim_mesh,
                "building_value_name": value_name,
                "building_colormap": req.colormap,
                "building_vmin": req.vmin,
                "building_vmax": req.vmax,
                "render_voxel_buildings": False,
                "title": title_map.get(app_state.last_sim_type, "Simulation"),
                "building_colorbar_title": app_state.last_colorbar_title,
            }
            if app_state.last_sim_type == "solar":
                plot_kwargs["building_opacity"] = 1.0
                plot_kwargs["building_shaded"] = False
            fig_json = _make_plotly_json(
                voxcity_grid, meshsize, plot_kwargs,
            )

        # Update cache for next rerender
        app_state.last_base_fig_json = fig_json
        app_state.last_hidden_classes = list(req.hidden_classes)

        return {"status": "ok", "figure_json": fig_json}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/cityles")
async def export_cityles_endpoint(req: ExportCitylesRequest):
    """Export to CityLES format and return as a ZIP download."""
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")

    try:
        cityles_dir = os.path.join(BASE_OUTPUT_DIR, "cityles")
        os.makedirs(cityles_dir, exist_ok=True)

        voxcity = app_state.voxcity
        # Pass land_cover_source through extras for CityLES
        if "land_cover_source" not in voxcity.extras:
            voxcity.extras["land_cover_source"] = app_state.land_cover_source

        export_cityles(
            voxcity,
            output_directory=cityles_dir,
            building_material=req.building_material,
            tree_type=req.tree_type,
            trunk_height_ratio=req.trunk_height_ratio,
            canopy_bottom_height_grid=(
                voxcity.tree_canopy.bottom if voxcity.tree_canopy.bottom is not None else None
            ),
            land_cover_source=app_state.land_cover_source,
        )

        zip_buf = app_state.zip_directory_to_bytes(cityles_dir)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=cityles_outputs.zip"},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export/obj")
async def export_obj_endpoint(req: ExportObjRequest):
    """Export to OBJ format and return as a ZIP download."""
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")

    try:
        output_dir = os.path.join(BASE_OUTPUT_DIR, "obj")
        os.makedirs(output_dir, exist_ok=True)

        voxcity = app_state.voxcity

        # export_obj accepts either VoxCity or raw array
        export_obj(
            voxcity.voxels.classes,
            output_dir,
            req.filename,
            voxcity.voxels.meta.meshsize,
        )

        files = [
            os.path.join(output_dir, f"{req.filename}.obj"),
            os.path.join(output_dir, f"{req.filename}.mtl"),
        ]

        if req.export_netcdf:
            try:
                from voxcity.exporter.netcdf import save_voxel_netcdf

                nc_dir = os.path.join(BASE_OUTPUT_DIR, "netcdf")
                os.makedirs(nc_dir, exist_ok=True)
                nc_path = os.path.join(nc_dir, f"{req.filename}.nc")
                save_voxel_netcdf(
                    voxcity.voxels.classes,
                    nc_path,
                    voxcity.voxels.meta.meshsize,
                    rectangle_vertices=app_state.rectangle_vertices,
                )
                files.append(nc_path)
            except Exception:
                pass

        zip_buf = app_state.zip_files_to_bytes(files)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={req.filename}.zip"},
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/epw/upload")
async def upload_epw(file: UploadFile = File(...)):
    """Upload an EPW file for solar simulations."""
    try:
        epw_dir = os.path.join(BASE_OUTPUT_DIR, "epw")
        os.makedirs(epw_dir, exist_ok=True)
        path = os.path.join(epw_dir, file.filename)
        with open(path, "wb") as f:
            content = await file.read()
            f.write(content)
        return {"status": "ok", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model/preview")
async def model_preview():
    """Return the 3D preview figure for the current model."""
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")

    fig_json = _make_plotly_json(
        app_state.voxcity.voxels.classes,
        app_state.meshsize,
        {"title": "VoxCity 3D"},
    )
    return {"figure_json": fig_json}


@app.get("/api/model/info")
async def model_info():
    """Return metadata about the current model."""
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")

    vc = app_state.voxcity
    building_gdf = vc.extras.get("building_gdf")
    n_buildings = len(building_gdf) if building_gdf is not None else 0

    return {
        "grid_shape": list(vc.voxels.classes.shape),
        "meshsize": app_state.meshsize,
        "n_buildings": n_buildings,
        "rectangle_vertices": app_state.rectangle_vertices,
        "land_cover_source": app_state.land_cover_source,
    }


@app.get("/api/buildings/list")
async def buildings_list():
    """Return a list of building IDs with centroid positions (in local voxel coords)."""
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")

    vc = app_state.voxcity
    bid_grid = vc.buildings.ids  # 2D (nx, ny)
    meshsize = app_state.meshsize
    voxcity_grid = vc.voxels.classes  # 3D (nx, ny, nz)

    if bid_grid is None:
        return {"buildings": []}

    # Flip building ID grid to match voxcity_grid orientation (same as mark_building_by_id)
    from voxcity.utils.orientation import ensure_orientation, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP
    bid_aligned = ensure_orientation(bid_grid, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)

    unique_ids = np.unique(bid_aligned)
    unique_ids = unique_ids[unique_ids != 0]  # exclude 0 (no building)

    buildings = []
    for bid in unique_ids:
        mask_2d = (bid_aligned == bid)
        xs, ys = np.where(mask_2d)
        if len(xs) == 0:
            continue
        # Centroid in voxel index space → local coords
        cx = (float(xs.mean()) + 0.5) * meshsize
        cy = (float(ys.mean()) + 0.5) * meshsize
        # Find building top z from voxel grid
        max_z = 0.0
        for xi, yi in zip(xs, ys):
            col = voxcity_grid[xi, yi, :]
            bz = np.where(col == -3)[0]
            if len(bz) > 0:
                max_z = max(max_z, float(bz.max() + 1) * meshsize)
        cz = max_z / 2.0
        buildings.append({
            "id": int(bid),
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "cz": round(cz, 2),
            "top_z": round(max_z, 2),
        })

    return {"buildings": buildings}


@app.get("/api/buildings/at")
async def building_at(x: float, y: float):
    """Return the building id at world XY (in metres), or ``null`` if none.

    Uses the orientation-aligned building id grid (the same one used by the
    highlight endpoint), so the result matches what the user sees in 3D.
    """
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")
    vc = app_state.voxcity
    bid_grid = getattr(vc.buildings, "ids", None)
    if bid_grid is None:
        return {"building_id": None}

    from voxcity.utils.orientation import (
        ORIENTATION_NORTH_UP,
        ORIENTATION_SOUTH_UP,
        ensure_orientation,
    )
    bid_aligned = ensure_orientation(bid_grid, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)

    meshsize = app_state.meshsize
    nx, ny = bid_aligned.shape
    # The click XY can land exactly on a voxel boundary (wall hit), so the
    # naive floor() can fall into an empty neighbour cell. Probe the centre
    # cell plus its 8 immediate neighbours and return the closest building.
    fx = float(x) / meshsize
    fy = float(y) / meshsize
    ci = int(fx)
    cj = int(fy)
    best_bid = 0
    best_d2 = float("inf")
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i = ci + di
            j = cj + dj
            if i < 0 or j < 0 or i >= nx or j >= ny:
                continue
            bid = int(bid_aligned[i, j])
            if bid == 0:
                continue
            # Distance from click XY to cell centre, in voxel units.
            cx = i + 0.5
            cy = j + 0.5
            d2 = (cx - fx) ** 2 + (cy - fy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_bid = bid
    return {"building_id": best_bid if best_bid != 0 else None}


@app.get("/api/landmark/preview")
async def landmark_preview():
    """Return a 3D preview figure with per-face building ID metadata for interactive selection."""
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")

    try:
        vc = app_state.voxcity
        voxcity_grid = vc.voxels.classes
        meshsize = app_state.meshsize
        bid_grid = vc.buildings.ids

        fig = visualize_voxcity_plotly(
            voxcity_grid,
            meshsize,
            downsample=1,
            show=False,
            return_fig=True,
            building_id_grid=bid_grid,
            title="Landmark Selection",
        )
        if fig is None:
            raise HTTPException(status_code=500, detail="Visualization engine returned no figure.")

        fig_json = _compact_fig_json(fig)

        return {"figure_json": fig_json}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Edit-tab helpers (ported from reference/optree_app/backend/main.py)
# ---------------------------------------------------------------------------

def _require_model() -> None:
    if not app_state.has_model:
        raise HTTPException(status_code=400, detail="No model generated yet")


def _parse_cells(payload: dict) -> List[List[int]]:
    """Pull a list of [i, j] integer cells from request JSON. NORTH_UP."""
    raw = payload.get("cells")
    if not isinstance(raw, list) or len(raw) == 0:
        raise HTTPException(status_code=400, detail="cells must be a non-empty list of [i,j]")
    cells: List[List[int]] = []
    for entry in raw:
        try:
            i = int(entry[0])
            j = int(entry[1])
        except (TypeError, ValueError, IndexError):
            raise HTTPException(status_code=400, detail="each cell must be [i, j] integers")
        cells.append([i, j])
    return cells


def _cells_to_native_mask(cells: List[List[int]], shape_native: tuple) -> np.ndarray:
    nx, ny = shape_native
    mask = np.zeros(shape_native, dtype=bool)
    for i, j in cells:
        if 0 <= i < nx and 0 <= j < ny:
            mask[i, j] = True
    return mask


def _cells_to_polygon(cells: List[List[int]], grid_geom: dict):
    """Build a shapely (Multi)Polygon covering the given (i, j) cells."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    if not cells:
        return None
    origin = np.asarray(grid_geom["origin"], dtype=float)
    u = np.asarray(grid_geom["u_vec"], dtype=float)
    v = np.asarray(grid_geom["v_vec"], dtype=float)
    dx = float(grid_geom["adj_mesh"][0])
    dy = float(grid_geom["adj_mesh"][1])
    polys = []
    for i, j in cells:
        bl = origin + (i       * dx) * u + (j       * dy) * v
        br = origin + ((i + 1) * dx) * u + (j       * dy) * v
        tr = origin + ((i + 1) * dx) * u + ((j + 1) * dy) * v
        tl = origin + (i       * dx) * u + ((j + 1) * dy) * v
        polys.append(ShapelyPolygon([bl, br, tr, tl]))
    if not polys:
        return None
    merged = unary_union(polys)
    return merged if not merged.is_empty else None


def _append_building_to_gdf(vc, polygon, building_id: int, height_m: float, min_height_m: float) -> None:
    """Append a footprint to ``vc.extras['building_gdf']`` (creating it if needed)."""
    if not isinstance(vc.extras, dict):
        vc.extras = {}
    gdf = vc.extras.get("building_gdf")
    new_row = {
        "geometry":         polygon,
        "height":           float(height_m),
        "min_height":       float(min_height_m),
        "building_id":      int(building_id),
        "id":               int(building_id),
        "height_estimated": False,
    }
    if gdf is None or len(gdf) == 0:
        vc.extras["building_gdf"] = gpd.GeoDataFrame([new_row], geometry="geometry", crs="EPSG:4326")
        return
    crs = getattr(gdf, "crs", None) or "EPSG:4326"
    appended = gpd.GeoDataFrame(
        [*gdf.to_dict("records"), new_row],
        geometry="geometry",
        crs=crs,
    )
    vc.extras["building_gdf"] = appended


def _apply_add_building(
    vc,
    cells: List[List[int]],
    height_m: float,
    min_height_m: float,
    ring: Optional[List[List[float]]] = None,
) -> dict:
    """Stamp a new building footprint into the grids + ``building_gdf``."""
    bld = vc.buildings
    mask_n = _cells_to_native_mask(cells, bld.heights.shape)
    if not mask_n.any():
        return {"n_changed": 0, "building_id": None}

    new_id = int(bld.ids.max()) + 1 if bld.ids.size else 1
    if new_id <= 0:
        new_id = 1

    bld.heights[mask_n] = float(height_m)
    bld.ids[mask_n] = new_id
    if bld.min_heights is not None:
        mh_pair = [[float(min_height_m), float(height_m)]]
        for r, c in np.argwhere(mask_n):
            bld.min_heights[int(r), int(c)] = list(mh_pair)

    try:
        poly = None
        if ring is not None and len(ring) >= 3:
            from shapely.geometry import Polygon as ShapelyPolygon
            try:
                poly = ShapelyPolygon([(float(p[0]), float(p[1])) for p in ring])
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    poly = None
            except Exception:
                poly = None

        if poly is None:
            from voxcity.geoprocessor.draw._common import compute_grid_geometry
            rect = app_state.rectangle_vertices
            if rect is None and isinstance(vc.extras, dict):
                rect = vc.extras.get("rectangle_vertices")
            if rect is not None:
                gg = compute_grid_geometry(rect, float(app_state.meshsize))
                if gg is not None:
                    cells_native = [[int(r), int(c)] for r, c in np.argwhere(mask_n)]
                    poly = _cells_to_polygon(cells_native, gg)

        if poly is not None:
            if poly.geom_type == "MultiPolygon":
                poly = max(poly.geoms, key=lambda p: p.area)
            _append_building_to_gdf(vc, poly, new_id, height_m, min_height_m)
    except Exception:
        traceback.print_exc()

    return {"n_changed": int(mask_n.sum()), "building_id": new_id}


def _apply_delete_buildings(vc, building_ids: List[int]) -> dict:
    """Remove the listed buildings from grids + GeoDataFrame.

    ``building_ids`` are the **row positions** the frontend sees, exactly as
    emitted by ``build_building_geojson`` (``properties.idx = int(idx)`` from
    ``building_gdf.iterrows()``). Those positions are *not* guaranteed to
    equal the source feature ids stored in ``buildings.ids`` and
    ``building_gdf['id']``: in Plateau (CityGML) mode they happen to align,
    but in Normal mode (e.g. OSM) the grid stores OSM way ids while the
    frontend sends 0-based row indices, so a naive ``np.isin`` matched
    nothing and silently returned ``n_deleted=0``.

    We therefore translate row positions → source ids via the current
    ``building_gdf`` before touching the grid.
    """
    from voxcity.utils.orientation import (
        ensure_orientation,
        ORIENTATION_NORTH_UP,
        ORIENTATION_SOUTH_UP,
    )

    bld = vc.buildings
    gdf = vc.extras.get("building_gdf") if isinstance(vc.extras, dict) else None

    # Row positions (0..N-1) coming from the frontend.
    row_positions = [int(i) for i in building_ids]
    if not row_positions:
        n_buildings = int(len(gdf)) if gdf is not None else 0
        return {"n_deleted": 0, "n_buildings": n_buildings}

    # Translate row position -> source id stored in the grid.
    delete_set: set[int] = set()
    if gdf is not None and "id" in getattr(gdf, "columns", []) and len(gdf) > 0:
        id_values = gdf["id"].tolist()
        n_rows = len(id_values)
        for pos in row_positions:
            if 0 <= pos < n_rows:
                try:
                    delete_set.add(int(id_values[pos]))
                except (TypeError, ValueError):
                    continue
    else:
        # No id column to translate through – fall back to identity, which
        # matches the historical Plateau-mode behaviour.
        delete_set = {pos for pos in row_positions if pos > 0}

    if not delete_set:
        n_buildings = int(len(gdf)) if gdf is not None else 0
        return {"n_deleted": 0, "n_buildings": n_buildings}

    ids_grid = ensure_orientation(bld.ids, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)
    mask_south = np.isin(ids_grid, list(delete_set))
    if not mask_south.any():
        print(
            f"[delete_buildings] no grid cells matched ids={sorted(delete_set)} "
            f"(row positions {row_positions}); model unchanged."
        )
        n_buildings = int(len(gdf)) if gdf is not None else 0
        return {"n_deleted": 0, "n_buildings": n_buildings}

    mask_north = ensure_orientation(mask_south, ORIENTATION_SOUTH_UP, ORIENTATION_NORTH_UP)
    bld.heights[mask_north] = 0.0
    bld.ids[mask_north] = 0
    if bld.min_heights is not None:
        it = np.nditer(mask_north, flags=["multi_index"])
        for hit in it:
            if bool(hit):
                r, c = it.multi_index
                bld.min_heights[r, c] = []

    if gdf is not None and "id" in getattr(gdf, "columns", []):
        try:
            vc.extras["building_gdf"] = gdf[~gdf["id"].isin(delete_set)].copy()
        except Exception:
            pass

    final_gdf = vc.extras.get("building_gdf") if isinstance(vc.extras, dict) else None
    n_buildings = int(len(final_gdf)) if final_gdf is not None else 0
    return {"n_deleted": int(len(delete_set)), "n_buildings": n_buildings}


def _apply_add_trees(
    vc,
    cells: List[List[int]],
    height_m: float,
    bottom_m: float,
    tops: Optional[List[float]] = None,
    bottoms: Optional[List[float]] = None,
) -> dict:
    tc = vc.tree_canopy
    if tc is None or tc.top is None:
        return {"n_changed": 0}
    mask_n = _cells_to_native_mask(cells, tc.top.shape)
    if not mask_n.any():
        return {"n_changed": 0}
    if tc.bottom is None:
        tc.bottom = np.zeros_like(tc.top)

    if tops is not None and bottoms is not None:
        if len(tops) != len(cells) or len(bottoms) != len(cells):
            raise ValueError("tops/bottoms must have the same length as cells")
        nx, ny = tc.top.shape
        rows: list[int] = []
        cols: list[int] = []
        ts: list[float] = []
        bs: list[float] = []
        for (i, j), t, b in zip(cells, tops, bottoms):
            ii = int(i); jj = int(j)
            if ii < 0 or ii >= nx or jj < 0 or jj >= ny:
                continue
            rows.append(ii)
            cols.append(jj)
            ts.append(float(t))
            bs.append(float(b))
        if rows:
            r_arr = np.asarray(rows, dtype=np.intp)
            c_arr = np.asarray(cols, dtype=np.intp)
            tc.top[r_arr, c_arr]    = np.asarray(ts, dtype=tc.top.dtype)
            tc.bottom[r_arr, c_arr] = np.asarray(bs, dtype=tc.bottom.dtype)
        return {"n_changed": int(len(rows))}

    tc.top[mask_n] = float(height_m)
    tc.bottom[mask_n] = float(bottom_m)
    return {"n_changed": int(mask_n.sum())}


def _apply_delete_trees(vc, cells: List[List[int]]) -> dict:
    tc = vc.tree_canopy
    if tc is None or tc.top is None:
        return {"n_changed": 0}
    mask_n = _cells_to_native_mask(cells, tc.top.shape)
    if not mask_n.any():
        return {"n_changed": 0}
    tc.top[mask_n] = 0.0
    if tc.bottom is not None:
        tc.bottom[mask_n] = 0.0
    return {"n_changed": int(mask_n.sum())}


def _source_lc_class_names(source: str | None) -> list[str]:
    if source is None:
        return []
    src_classes = get_land_cover_classes(source)
    return list(dict.fromkeys(src_classes.values()))


def _standard_to_source_lc_index(class_index: int, source: str | None) -> int | None:
    """Translate the standard 1-based ``LAND_COVER_CLASSES`` index → per-source 0-based index."""
    from voxcity.utils.classes import LAND_COVER_CLASSES

    name = LAND_COVER_CLASSES.get(int(class_index))
    if name is None:
        return None
    names = _source_lc_class_names(source)
    try:
        return names.index(name)
    except ValueError:
        return None


def _apply_paint_lc(vc, cells: List[List[int]], class_index: int) -> dict:
    lc = vc.land_cover
    if lc is None or lc.classes is None:
        return {"n_changed": 0}
    mask_n = _cells_to_native_mask(cells, lc.classes.shape)
    if not mask_n.any():
        return {"n_changed": 0}
    src_idx = _standard_to_source_lc_index(class_index, app_state.land_cover_source)
    if src_idx is None:
        src_idx = int(class_index)
    lc.classes[mask_n] = int(src_idx)
    return {"n_changed": int(mask_n.sum())}


def _render_edit_preview(vc, title: str = "Edit Model") -> str:
    """Render the standard edit-tab Plotly figure (with building IDs).

    Reuses ``_make_plotly_json`` so the existing 500 MB safeguard applies.
    """
    return _make_plotly_json(
        vc.voxels.classes,
        app_state.meshsize,
        {"building_id_grid": vc.buildings.ids, "title": title},
    )


# ---------------------------------------------------------------------------
# Edit-tab endpoints
# ---------------------------------------------------------------------------

@app.get("/api/land-cover/classes")
async def land_cover_classes():
    """Return the editable land cover class palette for the editor."""
    from voxcity.utils.classes import LAND_COVER_CLASSES
    from voxcity.geoprocessor.draw._common import get_lc_source_colors

    palette = {
        1:  "#c2b280",  # Bareland
        2:  "#9acd32",  # Rangeland
        3:  "#6b8e23",  # Shrub
        4:  "#daa520",  # Agriculture land
        5:  "#228b22",  # Tree
        6:  "#8fbc8f",  # Moss and lichen
        7:  "#5f9ea0",  # Wet land
        8:  "#2e8b57",  # Mangrove
        9:  "#1e90ff",  # Water
        10: "#f0ffff",  # Snow and ice
        11: "#a9a9a9",  # Developed space
        12: "#696969",  # Road
        13: "#8b0000",  # Building
        14: "#000000",  # No Data
    }
    non_editable = {"Tree", "Tree Canopy", "Trees", "Building", "Built Area", "Built-up", "Built"}

    src = app_state.land_cover_source
    src_names: set[str] | None = None
    src_colors: dict[str, str] = {}
    if src:
        try:
            src_names = set(_source_lc_class_names(src))
            src_colors = get_lc_source_colors(src)
        except Exception:
            src_names = None
            src_colors = {}

    classes_out = []
    for k in sorted(LAND_COVER_CLASSES.keys()):
        name = LAND_COVER_CLASSES[k]
        if src_names is not None and name not in src_names:
            continue
        color = src_colors.get(name) or palette.get(k, "#808080")
        classes_out.append({
            "index": k,
            "name": name,
            "color": color,
            "editable": name not in non_editable,
        })
    return {"classes": classes_out}


@app.get("/api/model/geo")
async def model_geo():
    """Return geo-anchored overlays + grid geometry for the basemap editor."""
    _require_model()
    try:
        from voxcity.geoprocessor.draw._common import (
            build_building_geojson,
            build_canopy_geojson,
            build_lc_geojson,
            compute_grid_geometry,
        )

        vc = app_state.voxcity
        rect = app_state.rectangle_vertices
        if rect is None:
            extras = vc.extras if isinstance(vc.extras, dict) else {}
            rect = extras.get("rectangle_vertices")
        if rect is None:
            raise HTTPException(status_code=400, detail="Model has no rectangle_vertices")

        meshsize = float(app_state.meshsize)
        grid_geom = compute_grid_geometry(rect, meshsize)
        if grid_geom is None:
            raise HTTPException(status_code=500, detail="compute_grid_geometry returned None")

        building_gdf = vc.extras.get("building_gdf") if isinstance(vc.extras, dict) else None
        lc_source = app_state.land_cover_source
        if lc_source is None and isinstance(vc.extras, dict):
            lc_source = vc.extras.get("land_cover_source")

        canopy_top = vc.tree_canopy.top if vc.tree_canopy is not None else None
        land_cover = vc.land_cover.classes if vc.land_cover is not None else None

        building_fc   = build_building_geojson(building_gdf, include_height=True)
        canopy_fc     = build_canopy_geojson(canopy_top, grid_geom)
        land_cover_fc = build_lc_geojson(land_cover, grid_geom, lc_source)

        nx, ny = vc.buildings.heights.shape
        lons = [v[0] for v in rect]
        lats = [v[1] for v in rect]
        center = [float(sum(lats) / len(lats)), float(sum(lons) / len(lons))]

        return {
            "rectangle_vertices": [[float(v[0]), float(v[1])] for v in rect],
            "meshsize_m": meshsize,
            "grid_shape": [int(nx), int(ny)],
            "center": center,
            "grid_geom": {
                "origin":    [float(grid_geom["origin"][0]),  float(grid_geom["origin"][1])],
                "side_1":    [float(grid_geom["side_1"][0]),  float(grid_geom["side_1"][1])],
                "side_2":    [float(grid_geom["side_2"][0]),  float(grid_geom["side_2"][1])],
                "u_vec":     [float(grid_geom["u_vec"][0]),   float(grid_geom["u_vec"][1])],
                "v_vec":     [float(grid_geom["v_vec"][0]),   float(grid_geom["v_vec"][1])],
                "adj_mesh":  [float(grid_geom["adj_mesh"][0]),float(grid_geom["adj_mesh"][1])],
                "grid_size": [int(grid_geom["grid_size"][0]), int(grid_geom["grid_size"][1])],
            },
            "land_cover_source": lc_source,
            "building_geojson": building_fc,
            "canopy_geojson": canopy_fc,
            "land_cover_geojson": land_cover_fc,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Zoning
# ---------------------------------------------------------------------------

def _grid_geom_for_zoning() -> dict:
    """Build the same grid_geom dict that /api/model/geo returns."""
    from voxcity.geoprocessor.draw._common import compute_grid_geometry
    rect = app_state.rectangle_vertices
    if rect is None:
        vc = app_state.voxcity
        if vc is not None and isinstance(vc.extras, dict):
            rect = vc.extras.get("rectangle_vertices")
    if rect is None:
        raise HTTPException(status_code=400, detail="Model has no rectangle_vertices")
    gg = compute_grid_geometry(rect, float(app_state.meshsize))
    if gg is None:
        raise HTTPException(status_code=500, detail="compute_grid_geometry returned None")
    return {
        "origin":    [float(gg["origin"][0]),  float(gg["origin"][1])],
        "u_vec":     [float(gg["u_vec"][0]),   float(gg["u_vec"][1])],
        "v_vec":     [float(gg["v_vec"][0]),   float(gg["v_vec"][1])],
        "adj_mesh":  [float(gg["adj_mesh"][0]),float(gg["adj_mesh"][1])],
        "grid_size": [int(gg["grid_size"][0]), int(gg["grid_size"][1])],
    }


def _zone_stats_ground(zones: List[ZoneSpec]) -> List[ZoneStat]:
    sim = app_state.last_sim_grid
    if sim is None:
        raise HTTPException(status_code=400, detail="No cached ground simulation result")
    grid_geom = _grid_geom_for_zoning()
    sim_arr = np.asarray(sim)
    nx, ny = sim_arr.shape
    out: List[ZoneStat] = []
    for z in zones:
        cells = polygon_lonlat_to_cells(z.ring_lonlat, grid_geom)
        if not cells:
            out.append(stats_from_values(z.id, 0, np.array([], dtype=float)))
            continue
        ii = np.fromiter((c[0] for c in cells), dtype=int, count=len(cells))
        jj = np.fromiter((c[1] for c in cells), dtype=int, count=len(cells))
        valid_idx = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
        ii = ii[valid_idx]; jj = jj[valid_idx]
        vals = sim_arr[ii, jj].astype(float, copy=False)
        out.append(stats_from_values(z.id, len(cells), vals))
    return out


def _zone_stats_building(zones: List[ZoneSpec]) -> List[ZoneStat]:
    mesh = app_state.last_sim_mesh
    if mesh is None:
        raise HTTPException(status_code=400, detail="No cached building-surface simulation result")
    centroids_xy, values, areas = mesh_face_data(mesh, app_state.last_sim_type or "")
    grid_geom = _grid_geom_for_zoning()
    centroids_lonlat = grid_xy_to_lonlat(centroids_xy, grid_geom)
    out: List[ZoneStat] = []
    for z in zones:
        mask = points_in_polygon_lonlat(centroids_lonlat, z.ring_lonlat)
        v = values[mask]
        a = areas[mask]
        out.append(stats_from_values(z.id, int(mask.sum()), v, weights=a))
    return out


@app.post("/api/zones/stats", response_model=ZoneStatsResponse)
def zone_stats(req: ZoneStatsRequest) -> ZoneStatsResponse:
    if app_state.voxcity is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    if app_state.last_sim_type is None:
        raise HTTPException(status_code=400, detail="Run a simulation first")

    target = app_state.last_sim_target
    if target == "ground":
        stats = _zone_stats_ground(req.zones)
    elif target == "building":
        stats = _zone_stats_building(req.zones)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported target: {target}")

    return ZoneStatsResponse(
        target=target,
        sim_type=app_state.last_sim_type,
        unit_label=app_state.last_colorbar_title,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# Three.js raw geometry endpoints (R3F migration)
# ---------------------------------------------------------------------------

@app.get("/api/scene/geometry", response_model=SceneGeometryResponse)
def scene_geometry(
    downsample: int = 1,
    color_scheme: str = "default",
) -> SceneGeometryResponse:
    """Return the static city voxel geometry as raw BufferGeometry chunks.

    ``color_scheme`` is forwarded to
    :func:`voxcity.visualizer.palette.get_voxel_color_map`. The simulation
    tabs request ``"grayscale"`` so the per-class voxel colours don't
    compete with the coloured sim overlay.
    """
    if app_state.voxcity is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    grid = app_state.voxcity.voxels.classes
    meshsize = app_state.meshsize
    return build_voxel_buffers(
        grid,
        meshsize,
        downsample=max(1, int(downsample)),
        color_scheme=color_scheme,
    )


@app.get("/api/buildings/highlight")
def buildings_highlight(
    ids: str = "",
    colormap: Optional[str] = None,
    emissive: bool = False,
) -> dict:
    """Return raw mesh chunks highlighting the given building IDs.

    ``ids`` is a comma-separated list of integer building IDs (the same IDs
    returned by ``/api/buildings/list``). The response is a list of
    ``MeshChunk`` records (one per face plane) tagged with a bright highlight
    colour, ready to be rendered as a translucent overlay on top of the
    grayscale city scene.

    Optional ``colormap`` makes the highlight colour match the maximum value
    of that matplotlib colormap; ``emissive=true`` tags the chunks so the
    frontend renders them with self-illumination.
    """
    if app_state.voxcity is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    parsed: List[int] = []
    for tok in (ids or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            parsed.append(int(tok))
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid building id: {tok!r}")
    if not parsed:
        return {"chunks": []}

    vc = app_state.voxcity
    bid_grid = getattr(vc.buildings, "ids", None)
    if bid_grid is None:
        return {"chunks": []}

    from voxcity.utils.orientation import (
        ORIENTATION_NORTH_UP,
        ORIENTATION_SOUTH_UP,
        ensure_orientation,
    )
    bid_aligned = ensure_orientation(bid_grid, ORIENTATION_NORTH_UP, ORIENTATION_SOUTH_UP)

    chunks = build_building_highlight_buffers(
        vc.voxels.classes,
        bid_aligned,
        parsed,
        app_state.meshsize,
        colormap=colormap,
        emissive=bool(emissive),
    )
    return {"chunks": [c.model_dump() for c in chunks]}


@app.post("/api/sim/{kind}/geometry", response_model=OverlayGeometryResponse)
def sim_geometry(kind: str, req: SimGeometryRequest) -> OverlayGeometryResponse:
    """Return the most recent sim result as a coloured Three.js overlay.

    ``kind`` is one of ``solar``, ``view``, ``landmark`` and must match the
    last simulation that was run; otherwise a 400 is returned.
    """
    if kind not in {"solar", "view", "landmark"}:
        raise HTTPException(status_code=404, detail=f"Unknown sim kind: {kind}")
    if app_state.last_sim_type is None:
        raise HTTPException(status_code=400, detail="Run a simulation first")
    if app_state.last_sim_type != kind:
        raise HTTPException(
            status_code=400,
            detail=f"Last sim was '{app_state.last_sim_type}', not '{kind}'",
        )

    target = app_state.last_sim_target
    unit = app_state.last_colorbar_title or ""

    if target == "ground":
        if app_state.last_sim_grid is None:
            raise HTTPException(status_code=400, detail="No ground sim grid cached")
        return build_ground_overlay_buffers(
            np.asarray(app_state.last_sim_grid),
            np.asarray(app_state.last_sim_voxcity_grid)
            if app_state.last_sim_voxcity_grid is not None
            else None,
            app_state.meshsize,
            app_state.last_sim_view_point_height,
            sim_type=kind,
            colormap=req.colormap,
            vmin=req.vmin,
            vmax=req.vmax,
            unit_label=unit,
        )
    elif target == "building":
        if app_state.last_sim_mesh is None:
            raise HTTPException(status_code=400, detail="No building sim mesh cached")
        return build_building_overlay_buffers(
            app_state.last_sim_mesh,
            sim_type=kind,
            colormap=req.colormap,
            vmin=req.vmin,
            vmax=req.vmax,
            unit_label=unit,
            zero_as_nan=(kind == "landmark"),
        )
    raise HTTPException(status_code=400, detail=f"Unsupported target: {target}")


@app.post("/api/model/apply_edits")
async def apply_edits(payload: dict):
    """Apply a batch of vector edits, then voxelize once and render the figure.

    Body: ``{"edits": [...]}``. Each edit has a ``kind`` and operation-specific
    fields. After all edits are applied, ``regenerate_voxels`` runs once, the
    raw cache is refreshed, and the standard edit-tab Plotly figure is rendered.
    """
    _require_model()
    edits = payload.get("edits")
    if not isinstance(edits, list):
        raise HTTPException(status_code=400, detail="edits must be a list")

    try:
        vc = app_state.voxcity
        n_changed_total = 0
        building_ids: List[int] = []

        for idx, edit in enumerate(edits):
            if not isinstance(edit, dict):
                raise HTTPException(status_code=400, detail=f"edit #{idx} must be an object")
            kind = edit.get("kind")

            if kind == "add_building":
                cells = _parse_cells(edit)
                try:
                    height_m = float(edit.get("height_m"))
                except (TypeError, ValueError):
                    raise HTTPException(status_code=400, detail=f"edit #{idx}: height_m required")
                if height_m <= 0:
                    raise HTTPException(status_code=400, detail=f"edit #{idx}: height_m must be > 0")
                min_height_m = float(edit.get("min_height_m") or 0.0)
                if min_height_m < 0 or min_height_m >= height_m:
                    raise HTTPException(status_code=400, detail=f"edit #{idx}: min_height_m must be in [0, height_m)")
                ring_raw = edit.get("ring")
                ring_arg: Optional[List[List[float]]] = None
                if ring_raw is not None:
                    if not isinstance(ring_raw, list) or len(ring_raw) < 3:
                        raise HTTPException(status_code=400, detail=f"edit #{idx}: ring must be a list of >=3 [lon, lat] pairs")
                    try:
                        ring_arg = [[float(p[0]), float(p[1])] for p in ring_raw]
                    except (TypeError, ValueError, IndexError):
                        raise HTTPException(status_code=400, detail=f"edit #{idx}: ring entries must be [lon, lat] numbers")
                r = _apply_add_building(vc, cells, height_m, min_height_m, ring_arg)
                n_changed_total += int(r["n_changed"])
                if r["building_id"] is not None:
                    building_ids.append(int(r["building_id"]))

            elif kind == "delete_building":
                ids_raw = edit.get("building_ids") or []
                try:
                    ids_to_delete = [int(v) for v in ids_raw]
                except (TypeError, ValueError):
                    raise HTTPException(status_code=400, detail=f"edit #{idx}: building_ids must be integers")
                r = _apply_delete_buildings(vc, ids_to_delete)
                n_changed_total += int(r.get("n_deleted", 0))

            elif kind == "add_trees":
                cells = _parse_cells(edit)
                try:
                    height_m = float(edit.get("height_m"))
                except (TypeError, ValueError):
                    raise HTTPException(status_code=400, detail=f"edit #{idx}: height_m required")
                if height_m <= 0:
                    raise HTTPException(status_code=400, detail=f"edit #{idx}: height_m must be > 0")
                bottom_m = float(edit.get("bottom_m") or 0.0)
                if bottom_m < 0 or bottom_m >= height_m:
                    raise HTTPException(status_code=400, detail=f"edit #{idx}: bottom_m must be in [0, height_m)")
                tops_raw = edit.get("tops")
                bottoms_raw = edit.get("bottoms")
                tops_arg: Optional[List[float]] = None
                bottoms_arg: Optional[List[float]] = None
                if tops_raw is not None or bottoms_raw is not None:
                    if not isinstance(tops_raw, list) or not isinstance(bottoms_raw, list):
                        raise HTTPException(status_code=400, detail=f"edit #{idx}: tops and bottoms must both be lists")
                    if len(tops_raw) != len(cells) or len(bottoms_raw) != len(cells):
                        raise HTTPException(status_code=400, detail=f"edit #{idx}: tops/bottoms length must match cells")
                    try:
                        tops_arg = [float(v) for v in tops_raw]
                        bottoms_arg = [float(v) for v in bottoms_raw]
                    except (TypeError, ValueError):
                        raise HTTPException(status_code=400, detail=f"edit #{idx}: tops/bottoms must be numbers")
                r = _apply_add_trees(vc, cells, height_m, bottom_m, tops_arg, bottoms_arg)
                n_changed_total += int(r["n_changed"])

            elif kind == "delete_trees":
                cells = _parse_cells(edit)
                r = _apply_delete_trees(vc, cells)
                n_changed_total += int(r["n_changed"])

            elif kind == "paint_lc":
                cells = _parse_cells(edit)
                try:
                    class_index = int(edit.get("class_index"))
                except (TypeError, ValueError):
                    raise HTTPException(status_code=400, detail=f"edit #{idx}: class_index must be an integer")
                if class_index < 1 or class_index > 14:
                    raise HTTPException(status_code=400, detail=f"edit #{idx}: class_index must be 1..14")
                r = _apply_paint_lc(vc, cells, class_index)
                n_changed_total += int(r["n_changed"])

            else:
                raise HTTPException(status_code=400, detail=f"edit #{idx}: unknown kind {kind!r}")

        # Single voxelization + render after the whole batch.
        regenerate_voxels(vc, inplace=True)
        app_state.refresh_raw_cache()
        return {
            "figure_json": _render_edit_preview(vc),
            "n_edits": len(edits),
            "n_changed_total": n_changed_total,
            "building_ids": building_ids,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
