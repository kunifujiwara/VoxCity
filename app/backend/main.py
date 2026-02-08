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

import json
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import geopandas as gpd
import numpy as np
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from .models import (
    ExportCitylesRequest,
    ExportObjRequest,
    GenerateRequest,
    GeocodeRequest,
    GeocodeResponse,
    LandmarkRequest,
    RectangleFromDimensions,
    SolarRequest,
    StatusResponse,
    ViewRequest,
)
from .state import app_state

# ---------------------------------------------------------------------------
# VoxCity imports (validated against current package)
# ---------------------------------------------------------------------------
from voxcity.generator import get_voxcity, get_voxcity_CityGML, Voxelizer
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
from voxcity.simulator_gpu.init_taichi import ensure_initialized
ensure_initialized()

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


def _make_plotly_json(
    voxcity_grid: np.ndarray,
    meshsize: float,
    plot_kwargs: dict,
) -> str:
    """Render a voxcity plotly figure and return its JSON string."""
    candidates = (1, 2, 3, 4, 5, 8, 10)
    for ds in candidates:
        try:
            fig = visualize_voxcity_plotly(
                voxcity_grid,
                meshsize,
                downsample=ds,
                show=False,
                return_fig=True,
                **plot_kwargs,
            )
            if fig is None:
                continue
            fig_json = fig.to_json()
            approx_mb = len(fig_json.encode("utf-8")) / (1024.0 * 1024.0)
            if approx_mb > 50:
                continue
            return fig_json
        except Exception:
            continue
    return "{}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "has_model": app_state.has_model}


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


@app.post("/api/generate")
async def generate_model(req: GenerateRequest):
    """Generate a VoxCity model.

    The function calls the *current* voxcity API which returns a VoxCity
    dataclass directly (no longer a tuple of arrays).
    """
    try:
        rectangle_vertices = _vertices_to_tuples(req.rectangle_vertices)
        output_dir = os.path.join(BASE_OUTPUT_DIR, "test")
        os.makedirs(output_dir, exist_ok=True)

        is_jp = _is_japan(req.rectangle_vertices)
        land_cover_source = "OpenEarthMapJapan" if is_jp else req.land_cover_source

        kwargs: Dict[str, Any] = {
            "building_complementary_source": "None",
            "complement_building_footprints": True,
            "building_complement_height": req.building_complement_height,
            "overlapping_footprint": req.overlapping_footprint,
            "output_dir": output_dir,
            "dem_interpolation": req.dem_interpolation,
            "static_tree_height": req.static_tree_height,
            "gridvis": False,
            "mapvis": False,
        }

        # Attempt cached CityGML first
        cached = None
        if req.use_citygml_cache:
            cached = _load_citygml_cache(req.rectangle_vertices)

        if cached is not None:
            cached_buildings, cached_terrain = (
                cached if isinstance(cached, tuple) else (cached, None)
            )
            voxcity_result = get_voxcity(
                rectangle_vertices,
                meshsize=req.meshsize,
                building_source="GeoDataFrame",
                land_cover_source=land_cover_source,
                canopy_height_source=req.canopy_height_source,
                dem_source="Flat",
                building_gdf=cached_buildings,
                terrain_gdf=cached_terrain,
                **kwargs,
            )
        else:
            voxcity_result = get_voxcity_CityGML(
                rectangle_vertices,
                land_cover_source,
                req.canopy_height_source,
                req.meshsize,
                citygml_path=CITYGML_PATH,
                **kwargs,
            )

        # Store in app state
        app_state.store_generation_result(
            voxcity_obj=voxcity_result,
            meshsize=req.meshsize,
            rectangle_vertices=req.rectangle_vertices,
            land_cover_source=land_cover_source,
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
                    "ground_z_offset": app_state.meshsize,
                    "ground_colormap": req.colormap,
                    "ground_vmin": req.vmin,
                    "ground_vmax": req.vmax,
                    "sim_surface_opacity": 0.95,
                    "title": "Solar overlay",
                },
            )

            return {"status": "ok", "figure_json": fig_json}

        else:  # building surfaces
            irradiance_kwargs: Dict[str, Any] = {
                "download_nearest_epw": False,
                "epw_file_path": epw_path,
            }

            if req.calc_type == "instantaneous":
                irradiance_kwargs["temporal_mode"] = "instantaneous"
                irradiance_kwargs["calc_time"] = req.calc_time or "01-01 12:00:00"
            else:
                irradiance_kwargs["temporal_mode"] = "cumulative"
                irradiance_kwargs["period_start"] = req.start_time
                irradiance_kwargs["period_end"] = req.end_time

            irradiance = get_building_global_solar_irradiance_using_epw(
                voxcity, **irradiance_kwargs
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
                    "building_colormap": req.colormap,
                    "building_vmin": req.vmin,
                    "building_vmax": req.vmax,
                    "building_opacity": 1.0,
                    "building_shaded": False,
                    "render_voxel_buildings": False,
                    "title": "Building Surface Solar (Global)",
                },
            )

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
                    "ground_z_offset": app_state.meshsize,
                    "ground_colormap": req.colormap,
                    "ground_vmin": req.vmin,
                    "ground_vmax": req.vmax,
                    "sim_surface_opacity": 0.95,
                    "title": req.view_type.replace("_", " ").title() + " View Index",
                },
            )

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
                    "building_colormap": req.colormap,
                    "building_vmin": req.vmin,
                    "building_vmax": req.vmax,
                    "render_voxel_buildings": False,
                    "title": "Surface View Factor",
                },
            )

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
                    "ground_z_offset": app_state.meshsize,
                    "ground_colormap": req.colormap,
                    "ground_vmin": req.vmin,
                    "ground_vmax": req.vmax,
                    "sim_surface_opacity": 0.95,
                    "title": "Landmark Visibility (Ground)",
                },
            )

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
                    "building_colormap": req.colormap,
                    "building_vmin": req.vmin,
                    "building_vmax": req.vmax,
                    "render_voxel_buildings": False,
                    "title": "Landmark Visibility (Surface)",
                },
            )

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
