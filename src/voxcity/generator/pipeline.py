import os
from ..utils.logging import get_logger
from typing import Optional
import numpy as np

from ..models import (
    GridMetadata,
    BuildingGrid,
    LandCoverGrid,
    DemGrid,
    VoxelGrid,
    CanopyGrid,
    VoxCity,
    PipelineConfig,
)

from .grids import (
    get_land_cover_grid,
    get_building_height_grid,
    get_canopy_height_grid,
    get_dem_grid,
)
from .voxelizer import Voxelizer


class VoxCityPipeline:
    def __init__(self, meshsize: float, rectangle_vertices, crs: str = "EPSG:4326") -> None:
        self.meshsize = float(meshsize)
        self.rectangle_vertices = rectangle_vertices
        self.crs = crs

    def _bounds(self):
        xs = [p[0] for p in self.rectangle_vertices]
        ys = [p[1] for p in self.rectangle_vertices]
        return (min(xs), min(ys), max(xs), max(ys))

    def _meta(self) -> GridMetadata:
        return GridMetadata(crs=self.crs, bounds=self._bounds(), meshsize=self.meshsize)

    def assemble_voxcity(
        self,
        voxcity_grid: np.ndarray,
        building_height_grid: np.ndarray,
        building_min_height_grid: np.ndarray,
        building_id_grid: np.ndarray,
        land_cover_grid: np.ndarray,
        dem_grid: np.ndarray,
        canopy_height_top: Optional[np.ndarray] = None,
        canopy_height_bottom: Optional[np.ndarray] = None,
        extras: Optional[dict] = None,
    ) -> VoxCity:
        meta = self._meta()
        buildings = BuildingGrid(
            heights=building_height_grid,
            min_heights=building_min_height_grid,
            ids=building_id_grid,
            meta=meta,
        )
        land = LandCoverGrid(classes=land_cover_grid, meta=meta)
        dem = DemGrid(elevation=dem_grid, meta=meta)
        voxels = VoxelGrid(classes=voxcity_grid, meta=meta)
        canopy = CanopyGrid(top=canopy_height_top if canopy_height_top is not None else np.zeros_like(land_cover_grid, dtype=float),
                            bottom=canopy_height_bottom,
                            meta=meta)
        _extras = {
            "rectangle_vertices": self.rectangle_vertices,
            "canopy_top": canopy.top,
            "canopy_bottom": canopy.bottom,
        }
        if extras:
            _extras.update(extras)
        return VoxCity(voxels=voxels, buildings=buildings, land_cover=land, dem=dem, tree_canopy=canopy, extras=_extras)

    def run(self, cfg: PipelineConfig, building_gdf=None, terrain_gdf=None, **kwargs) -> VoxCity:
        os.makedirs(cfg.output_dir, exist_ok=True)
        land_strategy = LandCoverSourceFactory.create(cfg.land_cover_source)
        build_strategy = BuildingSourceFactory.create(cfg.building_source)
        canopy_strategy = CanopySourceFactory.create(cfg.canopy_height_source, cfg)
        dem_strategy = DemSourceFactory.create(cfg.dem_source)

        # Prefer structured options from cfg; allow legacy kwargs for back-compat
        land_cover_grid = land_strategy.build_grid(
            cfg.rectangle_vertices, cfg.meshsize, cfg.output_dir,
            **{**cfg.land_cover_options, **kwargs}
        )
        # Detect effective land cover source (e.g., Urbanwatch -> OpenStreetMap fallback)
        try:
            from .grids import get_last_effective_land_cover_source
            lc_src_effective = get_last_effective_land_cover_source() or cfg.land_cover_source
        except Exception:
            lc_src_effective = cfg.land_cover_source
        bh, bmin, bid, building_gdf_out = build_strategy.build_grids(
            cfg.rectangle_vertices, cfg.meshsize, cfg.output_dir,
            building_gdf=building_gdf,
            **{**cfg.building_options, **kwargs}
        )
        canopy_top, canopy_bottom = canopy_strategy.build_grids(
            cfg.rectangle_vertices, cfg.meshsize, land_cover_grid, cfg.output_dir,
            land_cover_source=lc_src_effective,
            **{**cfg.canopy_options, **kwargs}
        )
        dem = dem_strategy.build_grid(
            cfg.rectangle_vertices, cfg.meshsize, land_cover_grid, cfg.output_dir,
            terrain_gdf=terrain_gdf,
            land_cover_like=land_cover_grid,
            **{**cfg.dem_options, **kwargs}
        )

        ro = cfg.remove_perimeter_object
        if (ro is not None) and (ro > 0):
            w_peri = int(ro * bh.shape[0] + 0.5)
            h_peri = int(ro * bh.shape[1] + 0.5)
            canopy_top[:w_peri, :] = canopy_top[-w_peri:, :] = canopy_top[:, :h_peri] = canopy_top[:, -h_peri:] = 0
            canopy_bottom[:w_peri, :] = canopy_bottom[-w_peri:, :] = canopy_bottom[:, :h_peri] = canopy_bottom[:, -h_peri:] = 0
            ids1 = np.unique(bid[:w_peri, :][bid[:w_peri, :] > 0]); ids2 = np.unique(bid[-w_peri:, :][bid[-w_peri:, :] > 0])
            ids3 = np.unique(bid[:, :h_peri][bid[:, :h_peri] > 0]); ids4 = np.unique(bid[:, -h_peri:][bid[:, -h_peri:] > 0])
            for rid in np.concatenate((ids1, ids2, ids3, ids4)):
                pos = np.where(bid == rid)
                bh[pos] = 0
                bmin[pos] = [[] for _ in range(len(bmin[pos]))]

        voxelizer = Voxelizer(
            voxel_size=cfg.meshsize,
            land_cover_source=lc_src_effective,
            trunk_height_ratio=cfg.trunk_height_ratio,
            voxel_dtype=kwargs.get("voxel_dtype", np.int8),
            max_voxel_ram_mb=kwargs.get("max_voxel_ram_mb"),
        )
        vox = voxelizer.generate_combined(
            building_height_grid_ori=bh,
            building_min_height_grid_ori=bmin,
            building_id_grid_ori=bid,
            land_cover_grid_ori=land_cover_grid,
            dem_grid_ori=dem,
            tree_grid_ori=canopy_top,
            canopy_bottom_height_grid_ori=canopy_bottom,
        )
        return self.assemble_voxcity(
            voxcity_grid=vox,
            building_height_grid=bh,
            building_min_height_grid=bmin,
            building_id_grid=bid,
            land_cover_grid=land_cover_grid,
            dem_grid=dem,
            canopy_height_top=canopy_top,
            canopy_height_bottom=canopy_bottom,
            extras={
                "building_gdf": building_gdf_out,
                "land_cover_source": lc_src_effective,
                "building_source": cfg.building_source,
                "dem_source": cfg.dem_source,
            },
        )


class LandCoverSourceStrategy:  # ABC simplified to avoid dependency in split
    def build_grid(self, rectangle_vertices, meshsize: float, output_dir: str, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError


class DefaultLandCoverStrategy(LandCoverSourceStrategy):
    def __init__(self, source: str) -> None:
        self.source = source

    def build_grid(self, rectangle_vertices, meshsize: float, output_dir: str, **kwargs):
        return get_land_cover_grid(rectangle_vertices, meshsize, self.source, output_dir, **kwargs)


class LandCoverSourceFactory:
    @staticmethod
    def create(source: str) -> LandCoverSourceStrategy:
        return DefaultLandCoverStrategy(source)


class BuildingSourceStrategy:  # ABC simplified
    def build_grids(self, rectangle_vertices, meshsize: float, output_dir: str, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError


class DefaultBuildingSourceStrategy(BuildingSourceStrategy):
    def __init__(self, source: str) -> None:
        self.source = source

    def build_grids(self, rectangle_vertices, meshsize: float, output_dir: str, **kwargs):
        return get_building_height_grid(rectangle_vertices, meshsize, self.source, output_dir, **kwargs)


class BuildingSourceFactory:
    @staticmethod
    def create(source: str) -> BuildingSourceStrategy:
        return DefaultBuildingSourceStrategy(source)


class CanopySourceStrategy:  # ABC simplified
    def build_grids(self, rectangle_vertices, meshsize: float, land_cover_grid: np.ndarray, output_dir: str, **kwargs):  # pragma: no cover
        raise NotImplementedError


class StaticCanopyStrategy(CanopySourceStrategy):
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg

    def build_grids(self, rectangle_vertices, meshsize: float, land_cover_grid: np.ndarray, output_dir: str, **kwargs):
        canopy_top = np.zeros_like(land_cover_grid, dtype=float)
        static_h = self.cfg.static_tree_height if self.cfg.static_tree_height is not None else kwargs.get("static_tree_height", 10.0)
        from ..utils.lc import get_land_cover_classes
        _classes = get_land_cover_classes(self.cfg.land_cover_source)
        _class_to_int = {name: i for i, name in enumerate(_classes.values())}
        _tree_labels = ["Tree", "Trees", "Tree Canopy"]
        _tree_idx = [_class_to_int[label] for label in _tree_labels if label in _class_to_int]
        tree_mask = np.isin(land_cover_grid, _tree_idx) if _tree_idx else np.zeros_like(land_cover_grid, dtype=bool)
        canopy_top[tree_mask] = static_h
        tr = self.cfg.trunk_height_ratio if self.cfg.trunk_height_ratio is not None else (11.76 / 19.98)
        canopy_bottom = canopy_top * float(tr)
        return canopy_top, canopy_bottom


class SourceCanopyStrategy(CanopySourceStrategy):
    def __init__(self, source: str) -> None:
        self.source = source

    def build_grids(self, rectangle_vertices, meshsize: float, land_cover_grid: np.ndarray, output_dir: str, **kwargs):
        # Provide land_cover_like for graceful fallback sizing without EE
        return get_canopy_height_grid(
            rectangle_vertices,
            meshsize,
            self.source,
            output_dir,
            land_cover_like=land_cover_grid,
            **kwargs,
        )


class CanopySourceFactory:
    @staticmethod
    def create(source: str, cfg: PipelineConfig) -> CanopySourceStrategy:
        if source == "Static":
            return StaticCanopyStrategy(cfg)
        return SourceCanopyStrategy(source)


class DemSourceStrategy:  # ABC simplified
    def build_grid(self, rectangle_vertices, meshsize: float, land_cover_grid: np.ndarray, output_dir: str, **kwargs):  # pragma: no cover
        raise NotImplementedError


class FlatDemStrategy(DemSourceStrategy):
    def build_grid(self, rectangle_vertices, meshsize: float, land_cover_grid: np.ndarray, output_dir: str, **kwargs):
        return np.zeros_like(land_cover_grid)


class SourceDemStrategy(DemSourceStrategy):
    def __init__(self, source: str) -> None:
        self.source = source

    def build_grid(self, rectangle_vertices, meshsize: float, land_cover_grid: np.ndarray, output_dir: str, **kwargs):
        terrain_gdf = kwargs.get("terrain_gdf")
        if terrain_gdf is not None:
            from ..geoprocessor.raster import create_dem_grid_from_gdf_polygon
            return create_dem_grid_from_gdf_polygon(terrain_gdf, meshsize, rectangle_vertices)
        try:
            return get_dem_grid(rectangle_vertices, meshsize, self.source, output_dir, **kwargs)
        except Exception as e:
            # Fallback to flat DEM if source fails or unsupported
            logger = get_logger(__name__)
            logger.warning("DEM source '%s' failed (%s). Falling back to flat DEM.", self.source, e)
            return np.zeros_like(land_cover_grid)


class DemSourceFactory:
    @staticmethod
    def create(source: str) -> DemSourceStrategy:
        # Normalize and auto-fallback: None/"none" -> Flat
        try:
            src_norm = (source or "").strip().lower()
        except Exception:
            src_norm = ""
        if (not source) or (src_norm in {"flat", "none", "null"}):
            return FlatDemStrategy()
        return SourceDemStrategy(source)


