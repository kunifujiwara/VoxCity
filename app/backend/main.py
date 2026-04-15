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
    RectangleFromDimensions,
    RerenderRequest,
    SolarRequest,
    StatusResponse,
    ViewRequest,
)
from .state import app_state

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
