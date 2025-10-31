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


