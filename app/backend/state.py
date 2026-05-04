"""
VoxCity state manager – holds the generated VoxCity object in memory
and provides helper methods for the FastAPI endpoints.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from voxcity.models import (
    BuildingGrid,
    CanopyGrid,
    DemGrid,
    GridMetadata,
    LandCoverGrid,
    VoxCity,
    VoxelGrid,
)


@dataclass
class AppState:
    """In-memory singleton holding the current session data."""

    voxcity: Optional[VoxCity] = None
    raw_data: Dict[str, Any] = field(default_factory=dict)
    rectangle_vertices: Optional[List[List[float]]] = None  # [[lon, lat], ...] WGS84
    land_cover_source: str = "OpenStreetMap"

    # Last simulation results (kept for re-rendering without re-running)
    last_sim_type: Optional[str] = None          # "solar" | "view" | "landmark"
    last_sim_target: Optional[str] = None        # "ground" | "building"
    last_sim_grid: Optional[Any] = None          # 2D ndarray (uv_m/SOUTH_UP, axis 0 = north/u) for ground-level
    last_sim_mesh: Optional[Any] = None          # mesh object for building surfaces
    last_sim_voxcity_grid: Optional[Any] = None  # voxcity_grid at sim time; uv_m/SOUTH_UP (may have marked buildings)
    last_sim_view_point_height: float = 1.5      # view_point_height used in last sim
    last_colorbar_title: Optional[str] = None     # Colorbar title (with unit) for last sim

    # Render cache for fast rerender (skip voxel face extraction)
    last_base_fig_json: Optional[str] = None     # Full Plotly figure JSON from last sim render
    last_hidden_classes: Optional[List[int]] = None  # Hidden classes used in last render

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def store_generation_result(
        self,
        voxcity_obj: VoxCity,
        meshsize: float,
        rectangle_vertices: List[List[float]],
        land_cover_source: str,
    ) -> None:
        """Persist the VoxCity result from get_voxcity / get_voxcity_CityGML."""
        self.voxcity = voxcity_obj
        self.rectangle_vertices = rectangle_vertices
        self.land_cover_source = land_cover_source
        # Keep handy raw data dict for backward compat
        self.raw_data = {
            "voxcity_grid": voxcity_obj.voxels.classes,
            "building_height_grid": voxcity_obj.buildings.heights,
            "building_min_height_grid": voxcity_obj.buildings.min_heights,
            "building_id_grid": voxcity_obj.buildings.ids,
            "canopy_height_grid": voxcity_obj.tree_canopy.top,
            "canopy_bottom_height_grid": (
                voxcity_obj.tree_canopy.bottom
                if voxcity_obj.tree_canopy.bottom is not None
                else None
            ),
            "land_cover_grid": voxcity_obj.land_cover.classes,
            "dem_grid": voxcity_obj.dem.elevation,
            "building_gdf": voxcity_obj.extras.get("building_gdf"),
            "meshsize": meshsize,
            "rectangle_vertices": rectangle_vertices,
        }

    @property
    def meshsize(self) -> float:
        if self.voxcity is not None:
            return self.voxcity.voxels.meta.meshsize
        return float(self.raw_data.get("meshsize", 5.0))

    @property
    def has_model(self) -> bool:
        return self.voxcity is not None

    def refresh_raw_cache(self) -> None:
        """Re-publish freshly-edited grids into ``raw_data`` so downstream
        tabs (Solar / View / Landmark / Export) see the latest state."""
        vc = self.voxcity
        if vc is None:
            return
        if self.raw_data is None:
            self.raw_data = {}
        self.raw_data["voxcity_grid"] = vc.voxels.classes
        self.raw_data["building_height_grid"] = vc.buildings.heights
        self.raw_data["building_min_height_grid"] = vc.buildings.min_heights
        self.raw_data["building_id_grid"] = vc.buildings.ids
        if vc.tree_canopy is not None:
            self.raw_data["canopy_height_grid"] = vc.tree_canopy.top
            self.raw_data["canopy_bottom_height_grid"] = vc.tree_canopy.bottom
        if vc.land_cover is not None:
            self.raw_data["land_cover_grid"] = vc.land_cover.classes
        if isinstance(vc.extras, dict):
            self.raw_data["building_gdf"] = vc.extras.get("building_gdf")

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------
    @staticmethod
    def zip_directory_to_bytes(dir_path: str) -> BytesIO:
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    full_path = os.path.join(root, fname)
                    arcname = os.path.relpath(full_path, start=dir_path)
                    try:
                        zf.write(full_path, arcname)
                    except Exception:
                        pass
        memory_file.seek(0)
        return memory_file

    @staticmethod
    def zip_files_to_bytes(file_paths: list) -> BytesIO:
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in file_paths:
                if os.path.isfile(p):
                    try:
                        zf.write(p, arcname=os.path.basename(p))
                    except Exception:
                        pass
        memory_file.seek(0)
        return memory_file


# Global singleton
app_state = AppState()
