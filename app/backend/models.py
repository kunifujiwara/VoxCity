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


class AutoDetectSourcesRequest(BaseModel):
    rectangle_vertices: List[List[float]]  # [[lon, lat], ...]


class GenerateRequest(BaseModel):
    rectangle_vertices: List[List[float]]  # [[lon, lat], ...]
    meshsize: float = 5.0
    mode: str = "plateau"  # "plateau" or "normal"
    # Normal-mode data sources (ignored when mode="plateau")
    building_source: Optional[str] = None  # None = auto-select
    land_cover_source: Optional[str] = None  # None = auto-select
    canopy_height_source: Optional[str] = None  # None = auto-select
    dem_source: Optional[str] = None  # None = auto-select
    building_complementary_source: Optional[str] = None  # None = auto-select
    # Shared parameters
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


class RerenderRequest(BaseModel):
    """Re-render the last simulation with new visualization parameters."""
    colormap: str = "viridis"
    vmin: Optional[float] = 0.0
    vmax: Optional[float] = None
    hidden_classes: List[int] = Field(default_factory=list)


class PlotlyFigureResponse(BaseModel):
    """JSON-serialized Plotly figure for rendering in the React frontend."""
    figure_json: str
    info: Optional[str] = None


class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Zoning
# ---------------------------------------------------------------------------

class ZoneSpec(BaseModel):
    """A 2D zone footprint as a lon/lat ring (does not need to be closed)."""
    id: str
    name: str
    ring_lonlat: List[List[float]] = Field(..., min_length=3)


class ZoneStatsRequest(BaseModel):
    zones: List[ZoneSpec]


class ZoneStat(BaseModel):
    zone_id: str
    cell_count: int            # cells/faces inside the zone
    valid_count: int           # of those, with finite values
    mean: Optional[float] = None
    min:  Optional[float] = None
    max:  Optional[float] = None
    std:  Optional[float] = None


class ZoneStatsResponse(BaseModel):
    target:     str            # "ground" | "building"
    sim_type:   Optional[str] = None  # "solar" | "view" | "landmark"
    unit_label: Optional[str] = None
    stats:      List[ZoneStat]


# ---------------------------------------------------------------------------
# Three.js raw geometry (R3F migration)
# ---------------------------------------------------------------------------

class MeshChunk(BaseModel):
    """One BufferGeometry-friendly chunk.

    Attributes
    ----------
    name : Logical name (e.g. ``"buildings+x"``, ``"ground_overlay"``).
    positions : Flat XYZ vertex array, length ``= 3 * vertex_count``.
    indices : Flat triangle indices, length ``= 3 * triangle_count``.
    color : Optional uniform RGB ``[r, g, b]`` in ``[0, 1]``.
    colors : Optional per-vertex RGB array, length ``= 3 * vertex_count``.
        When set, overrides ``color``.
    opacity : Material opacity in ``[0, 1]``.
    flat_shading : Whether to use ``flatShading`` on the material.
    metadata : Free-form per-chunk metadata (e.g. building IDs per face).
    """
    name: str
    positions: List[float]
    indices: List[int]
    color: Optional[List[float]] = None
    colors: Optional[List[float]] = None
    opacity: float = 1.0
    flat_shading: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SceneGeometryResponse(BaseModel):
    """Static city geometry payload for the R3F viewer."""
    chunks: List[MeshChunk]
    bbox_min: List[float]   # [x, y, z] world metres — always [0, 0, 0]
    bbox_max: List[float]   # [nx*ms, ny*ms, nz*ms] world metres (SOUTH_UP cell layout)
    meshsize_m: float
    # Topmost ground (land-cover) elevation in metres across the whole scene.
    # Used by the frontend to render zone outlines slightly above ground level
    # so they don't get hidden inside the terrain.
    ground_top_m: float = 0.0


class SimGeometryRequest(BaseModel):
    """Body for ``POST /api/sim/{kind}/geometry``."""
    colormap: str = "viridis"
    vmin: Optional[float] = None
    vmax: Optional[float] = None


class OverlayGeometryResponse(BaseModel):
    """Per-tab simulation overlay payload for the R3F viewer.

    The single ``chunk`` carries per-vertex colors that match the requested
    colormap. ``face_to_cell`` (ground sims) or ``face_to_building``
    (building sims) provide click-pick metadata for the frontend ``Picker``.
    """
    target: str                                  # "ground" | "building"
    sim_type: str                                # "solar" | "view" | "landmark"
    chunk: MeshChunk
    face_to_cell: Optional[List[List[int]]] = None  # [[i, j], ...] ij_north (NORTH_UP) per triangle
    face_to_building: Optional[List[int]] = None
    value_min: float
    value_max: float
    colormap: str
    unit_label: str = ""
