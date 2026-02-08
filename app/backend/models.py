"""Pydantic models for the VoxCity web API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class Vertex(BaseModel):
    lon: float
    lat: float


class RectangleVertices(BaseModel):
    vertices: List[Vertex] = Field(..., min_length=4, max_length=4)


class GeocodeRequest(BaseModel):
    city_name: str


class GeocodeResponse(BaseModel):
    lat: float
    lon: float
    bbox: Optional[Tuple[float, float, float, float]] = None


class RectangleFromDimensions(BaseModel):
    center_lon: float
    center_lat: float
    width_m: float
    height_m: float


class GenerateRequest(BaseModel):
    rectangle_vertices: List[List[float]]  # [[lon, lat], ...]
    meshsize: float = 5.0
    building_source: str = "OpenStreetMap"
    land_cover_source: str = "OpenStreetMap"
    canopy_height_source: str = "Static"
    dem_source: str = "Flat"
    building_complement_height: float = 10.0
    static_tree_height: float = 10.0
    overlapping_footprint: Any = "auto"
    dem_interpolation: bool = True
    use_citygml_cache: bool = True
    use_ndsm_canopy: bool = True


class SolarRequest(BaseModel):
    calc_type: str = "instantaneous"  # "instantaneous" or "cumulative"
    analysis_target: str = "ground"  # "ground" or "building"
    calc_time: Optional[str] = None  # "MM-DD HH:MM:SS"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    epw_source: str = "default"  # "default" or "upload"
    view_point_height: float = 1.5
    colormap: str = "magma"
    vmin: Optional[float] = 0.0
    vmax: Optional[float] = None
    hidden_classes: List[int] = Field(default_factory=list)


class ViewRequest(BaseModel):
    view_type: str = "green"  # "green", "sky", "custom"
    analysis_target: str = "ground"  # "ground" or "building"
    view_point_height: float = 1.5
    custom_classes: List[int] = Field(default_factory=list)
    inclusion_mode: bool = True
    n_azimuth: int = 60
    n_elevation: int = 10
    elevation_min_degrees: float = -30.0
    elevation_max_degrees: float = 30.0
    colormap: str = "viridis"
    vmin: Optional[float] = 0.0
    vmax: Optional[float] = 1.0
    hidden_classes: List[int] = Field(default_factory=list)
    export_obj: bool = False


class LandmarkRequest(BaseModel):
    analysis_target: str = "ground"  # "ground" or "building"
    landmark_ids: List[int] = Field(default_factory=list)
    view_point_height: float = 1.5
    n_azimuth: int = 60
    n_elevation: int = 10
    elevation_min_degrees: float = -30.0
    elevation_max_degrees: float = 30.0
    colormap: str = "viridis"
    vmin: Optional[float] = 0.0
    vmax: Optional[float] = 1.0
    hidden_classes: List[int] = Field(default_factory=list)


class ExportCitylesRequest(BaseModel):
    building_material: str = "default"
    tree_type: str = "default"
    trunk_height_ratio: float = 0.3


class ExportObjRequest(BaseModel):
    filename: str = "voxcity"
    export_netcdf: bool = False


class PlotlyFigureResponse(BaseModel):
    """JSON-serialized Plotly figure for rendering in the React frontend."""
    figure_json: str
    info: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
