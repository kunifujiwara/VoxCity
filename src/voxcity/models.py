from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any

import numpy as np


@dataclass
class GridMetadata:
    crs: str
    bounds: Tuple[float, float, float, float]
    meshsize: float


@dataclass
class BuildingGrid:
    heights: np.ndarray
    min_heights: np.ndarray  # object-dtype array of lists per cell
    ids: np.ndarray
    meta: GridMetadata


@dataclass
class LandCoverGrid:
    classes: np.ndarray
    meta: GridMetadata


@dataclass
class DemGrid:
    elevation: np.ndarray
    meta: GridMetadata


@dataclass
class VoxelGrid:
    classes: np.ndarray
    meta: GridMetadata


@dataclass
class CanopyGrid:
    top: np.ndarray
    meta: GridMetadata
    bottom: Optional[np.ndarray] = None


@dataclass
class VoxCity:
    voxels: VoxelGrid
    buildings: BuildingGrid
    land_cover: LandCoverGrid
    dem: DemGrid
    tree_canopy: CanopyGrid
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_xarray(self):
        """Named-dimension view of the model's grids as an ``xarray.Dataset``.

        Dims are the axis-contract tokens (``voxel``: ``("north", "east",
        "up")``; 2-D grids: ``("north", "east")``), so every access carries
        its axis names — ``ds.dem.isel(north=0)`` is unambiguously the south
        edge. Coordinates are cell-centre metres in the (possibly rotated)
        grid frame: ``(index + 0.5) * meshsize``. DataArrays wrap the
        existing ndarrays without copying. ``min_heights`` (object dtype)
        and ``extras`` are not representable and stay on the dataclass.
        lon/lat is derivable via ``voxcity.utils.projector.GridProjector``.
        ``rotation_angle`` is derived from ``extras['rectangle_vertices']``
        when present (the same source of truth ``save_results_h5`` uses),
        falling back to ``extras['rotation_angle']`` then ``0.0``.
        """
        import xarray as xr

        from .utils.orientation import AXES, AXES_ATTR
        from .geoprocessor.utils import compute_rotation_angle

        ms = float(self.voxels.meta.meshsize)
        ni, nj, nk = self.voxels.classes.shape
        coords = {
            AXES[0]: (np.arange(ni) + 0.5) * ms,
            AXES[1]: (np.arange(nj) + 0.5) * ms,
            AXES[2]: (np.arange(nk) + 0.5) * ms,
        }
        dims2 = AXES[:2]
        data_vars = {
            "voxel": (AXES, self.voxels.classes),
            "building_height": (dims2, self.buildings.heights),
            "building_id": (dims2, self.buildings.ids),
            "land_cover": (dims2, self.land_cover.classes),
            "dem": (dims2, self.dem.elevation),
        }
        if self.tree_canopy is not None and self.tree_canopy.top is not None:
            data_vars["canopy_top"] = (dims2, self.tree_canopy.top)
        if self.tree_canopy is not None and self.tree_canopy.bottom is not None:
            data_vars["canopy_bottom"] = (dims2, self.tree_canopy.bottom)

        extras = self.extras or {}
        rect = extras.get("rectangle_vertices")
        if rect is not None:
            # rectangle_vertices is the source of truth (as at save time), so
            # derive the angle rather than trust a possibly-stale
            # extras['rotation_angle']; this keeps the view consistent with the
            # file the same city would save to.
            rot = compute_rotation_angle(rect)
        else:
            rot = extras.get("rotation_angle", 0.0)
        attrs = {
            "axes": AXES_ATTR,
            "rotation_angle": float(rot),
            "meshsize": ms,
            "crs": self.voxels.meta.crs,
        }
        if rect is not None:
            attrs["rectangle_vertices"] = np.asarray(rect, dtype=np.float64)
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)


@dataclass
class PipelineConfig:
    rectangle_vertices: Any
    meshsize: float
    building_source: Optional[str] = None
    land_cover_source: Optional[str] = None
    canopy_height_source: Optional[str] = None
    dem_source: Optional[str] = None
    output_dir: str = "output"
    trunk_height_ratio: Optional[float] = None
    static_tree_height: Optional[float] = None
    remove_perimeter_object: Optional[float] = None
    mapvis: bool = False
    gridvis: bool = True
    # Parallel download mode: if True, downloads run concurrently using ThreadPoolExecutor
    parallel_download: bool = True
    # Structured options for strategies and I/O/visualization
    land_cover_options: Dict[str, Any] = field(default_factory=dict)
    building_options: Dict[str, Any] = field(default_factory=dict)
    canopy_options: Dict[str, Any] = field(default_factory=dict)
    dem_options: Dict[str, Any] = field(default_factory=dict)
    io_options: Dict[str, Any] = field(default_factory=dict)
    visualize_options: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Mesh data structures
# -----------------------------

@dataclass
class MeshModel:
    vertices: np.ndarray  # (N, 3) float
    faces: np.ndarray     # (M, 3|4) int
    colors: Optional[np.ndarray] = None  # (M, 4) uint8 or None
    name: Optional[str] = None


@dataclass
class MeshCollection:
    """Container for named meshes with simple add/access helpers."""
    meshes: Dict[str, MeshModel] = field(default_factory=dict)

    def add(self, name: str, mesh: MeshModel) -> None:
        self.meshes[name] = mesh

    def get(self, name: str) -> Optional[MeshModel]:
        return self.meshes.get(name)

    def __iter__(self):
        return iter(self.meshes.items())

    # Compatibility: some renderers expect `collection.items.items()`
    @property
    def items(self) -> Dict[str, MeshModel]:
        return self.meshes


