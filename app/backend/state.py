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
class SimulationResultCache:
    """Cached result for a single simulation type (solar / view / landmark)."""

    sim_type: str                               # "solar" | "view" | "landmark"
    target: str                                 # "ground" | "building"
    grid: Optional[Any] = None                  # 2D ndarray for ground-level sims
    mesh: Optional[Any] = None                  # mesh object for building-surface sims
    voxcity_grid: Optional[Any] = None          # voxcity_grid snapshot at sim time
    view_point_height: float = 1.5              # relevant for view sims
    colorbar_title: Optional[str] = None        # unit label for the colorbar
    building_id_grid: Optional[Any] = None      # building_id_grid snapshot at sim time


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

    # Per-type simulation cache (one entry per sim kind, keyed by sim_type string)
    sim_results_by_type: Dict[str, SimulationResultCache] = field(default_factory=dict)

    # Render cache for fast rerender (skip voxel face extraction)
    last_base_fig_json: Optional[str] = None     # Full Plotly figure JSON from last sim render
    last_hidden_classes: Optional[List[int]] = None  # Hidden classes used in last render

    # ------------------------------------------------------------------
    # Per-type simulation cache helpers
    # ------------------------------------------------------------------
    def store_sim_result(
        self,
        sim_type: str,
        target: str,
        grid: Optional[Any] = None,
        mesh: Optional[Any] = None,
        voxcity_grid: Optional[Any] = None,
        view_point_height: float = 1.5,
        colorbar_title: Optional[str] = None,
        building_id_grid: Optional[Any] = None,
    ) -> None:
        """Persist a simulation result both in the per-type dict and the legacy
        last_sim_* fields so existing render/export paths keep working."""
        bid_snapshot = np.asarray(building_id_grid).copy() if building_id_grid is not None else None
        entry = SimulationResultCache(
            sim_type=sim_type,
            target=target,
            grid=grid,
            mesh=mesh,
            voxcity_grid=voxcity_grid,
            view_point_height=view_point_height,
            colorbar_title=colorbar_title,
            building_id_grid=bid_snapshot,
        )
        self.sim_results_by_type[sim_type] = entry
        # Update legacy fields
        self.last_sim_type = sim_type
        self.last_sim_target = target
        self.last_sim_grid = grid
        self.last_sim_mesh = mesh
        self.last_sim_voxcity_grid = voxcity_grid
        self.last_sim_view_point_height = view_point_height
        self.last_colorbar_title = colorbar_title

    def get_sim_result(self, sim_type: Optional[str]) -> Optional[SimulationResultCache]:
        """Return the cached result for the requested *sim_type*, or the most
        recent result when *sim_type* is None/absent (backward-compat)."""
        if sim_type is not None:
            return self.sim_results_by_type.get(sim_type)
        # Fall back to "latest overall" via legacy last_sim_type
        if self.last_sim_type is not None:
            return self.sim_results_by_type.get(self.last_sim_type)
        return None

    def clear_sim_results(self) -> None:
        """Clear all per-type caches and legacy last_sim_* fields (full reset)."""
        self.sim_results_by_type.clear()
        self.last_sim_type = None
        self.last_sim_target = None
        self.last_sim_grid = None
        self.last_sim_mesh = None
        self.last_sim_voxcity_grid = None
        self.last_sim_view_point_height = 1.5
        self.last_colorbar_title = None

    def reset_for_session_load(self) -> None:
        """Clear caches that are tied to the previous voxcity grid."""
        self.raw_data = {}
        self.clear_sim_results()
        self.last_base_fig_json = None
        self.last_hidden_classes = None

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
