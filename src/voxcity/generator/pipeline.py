import os
from ..utils.logging import get_logger
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
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

        # Check if parallel download is enabled
        parallel_download = getattr(cfg, 'parallel_download', False)

        if parallel_download and cfg.canopy_height_source != "Static":
            # All 4 downloads are independent - run in parallel
            land_cover_grid, bh, bmin, bid, building_gdf_out, canopy_top, canopy_bottom, dem, lc_src_effective = \
                self._run_parallel_downloads(
                    cfg, land_strategy, build_strategy, canopy_strategy, dem_strategy,
                    building_gdf, terrain_gdf, kwargs
                )
            # Run visualizations after parallel downloads complete (if gridvis enabled)
            if kwargs.get('gridvis', cfg.gridvis):
                self._visualize_grids_after_parallel(
                    land_cover_grid, bh, canopy_top, dem, 
                    lc_src_effective, cfg.meshsize
                )
        elif parallel_download and cfg.canopy_height_source == "Static":
            # Static canopy needs land_cover_grid for tree mask
            # Run land_cover + building + dem in parallel, then canopy sequentially
            land_cover_grid, bh, bmin, bid, building_gdf_out, dem, lc_src_effective = \
                self._run_parallel_downloads_static_canopy(
                    cfg, land_strategy, build_strategy, dem_strategy,
                    building_gdf, terrain_gdf, kwargs
                )
            # Visualize land_cover, building, dem after parallel (if gridvis enabled)
            if kwargs.get('gridvis', cfg.gridvis):
                self._visualize_grids_after_parallel(
                    land_cover_grid, bh, None, dem,
                    lc_src_effective, cfg.meshsize
                )
            # Now run canopy with land_cover_grid available (this will visualize itself)
            canopy_top, canopy_bottom = canopy_strategy.build_grids(
                cfg.rectangle_vertices, cfg.meshsize, land_cover_grid, cfg.output_dir,
                land_cover_source=lc_src_effective,
                **{**cfg.canopy_options, **kwargs}
            )
        else:
            # Sequential mode (original behavior)
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
        
        # Build extras dict
        extras_dict = {
            "building_gdf": building_gdf_out,
            "land_cover_source": lc_src_effective,
            "building_source": cfg.building_source,
            "dem_source": cfg.dem_source,
            "canopy_height_source": cfg.canopy_height_source,
        }
        
        # Include tree_gdf if the canopy strategy created one (e.g., OSM source)
        if hasattr(canopy_strategy, 'tree_gdf') and canopy_strategy.tree_gdf is not None:
            extras_dict["tree_gdf"] = canopy_strategy.tree_gdf
        
        return self.assemble_voxcity(
            voxcity_grid=vox,
            building_height_grid=bh,
            building_min_height_grid=bmin,
            building_id_grid=bid,
            land_cover_grid=land_cover_grid,
            dem_grid=dem,
            canopy_height_top=canopy_top,
            canopy_height_bottom=canopy_bottom,
            extras=extras_dict,
        )

    def _visualize_grids_after_parallel(
        self, land_cover_grid, building_height_grid, canopy_top, dem_grid,
        land_cover_source, meshsize
    ):
        """
        Run grid visualizations after parallel downloads complete.
        This ensures matplotlib calls happen sequentially on the main thread.
        """
        from ..visualizer.grids import visualize_land_cover_grid, visualize_numerical_grid
        from ..utils.lc import get_land_cover_classes
        
        # Visualize land cover (convert int grid back to string for visualization)
        try:
            land_cover_classes = get_land_cover_classes(land_cover_source)
            # Create reverse mapping: int -> string class name
            int_to_class = {i: name for i, name in enumerate(land_cover_classes.values())}
            # Convert integer grid to string grid for visualization
            land_cover_grid_str = np.empty(land_cover_grid.shape, dtype=object)
            for i, name in int_to_class.items():
                land_cover_grid_str[land_cover_grid == i] = name
            color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}
            visualize_land_cover_grid(np.flipud(land_cover_grid_str), meshsize, color_map, land_cover_classes)
        except Exception as e:
            get_logger(__name__).warning("Land cover visualization failed: %s", e)
        
        # Visualize building height
        try:
            building_height_grid_nan = building_height_grid.copy().astype(float)
            building_height_grid_nan[building_height_grid_nan == 0] = np.nan
            visualize_numerical_grid(np.flipud(building_height_grid_nan), meshsize, "building height (m)", cmap='viridis', label='Value')
        except Exception as e:
            get_logger(__name__).warning("Building height visualization failed: %s", e)
        
        # Visualize canopy height (if provided)
        if canopy_top is not None:
            try:
                canopy_height_grid_nan = canopy_top.copy()
                canopy_height_grid_nan[canopy_height_grid_nan == 0] = np.nan
                visualize_numerical_grid(np.flipud(canopy_height_grid_nan), meshsize, "Tree canopy height", cmap='Greens', label='Tree canopy height (m)')
            except Exception as e:
                get_logger(__name__).warning("Canopy height visualization failed: %s", e)
        
        # Visualize DEM
        try:
            visualize_numerical_grid(np.flipud(dem_grid), meshsize, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')
        except Exception as e:
            get_logger(__name__).warning("DEM visualization failed: %s", e)

    def _run_parallel_downloads(
        self, cfg, land_strategy, build_strategy, canopy_strategy, dem_strategy,
        building_gdf, terrain_gdf, kwargs
    ):
        """
        Run all 4 downloads (land_cover, building, canopy, dem) in parallel.
        Used when canopy source is NOT 'Static' (no land_cover dependency).
        """
        import logging
        logger = get_logger(__name__)
        
        # Print clean header for parallel mode
        print("\n" + "="*60)
        print("Downloading data in parallel mode (4 concurrent downloads)")
        print("="*60)
        print(f"  • Land Cover:  {cfg.land_cover_source}")
        print(f"  • Building:    {cfg.building_source}")
        print(f"  • Canopy:      {cfg.canopy_height_source}")
        print(f"  • DEM:         {cfg.dem_source}")
        print("-"*60)
        print("Downloading... (this may take a moment)")
        
        results = {}
        
        # Disable gridvis and verbose prints in parallel mode
        # Also suppress httpx INFO logs during parallel downloads
        parallel_kwargs = {**kwargs, 'gridvis': False, 'print_class_info': False, 'quiet': True}
        lc_opts = {**cfg.land_cover_options, 'gridvis': False, 'print_class_info': False, 'quiet': True}
        bld_opts = {**cfg.building_options, 'gridvis': False, 'quiet': True}
        canopy_opts = {**cfg.canopy_options, 'gridvis': False, 'quiet': True}
        dem_opts = {**cfg.dem_options, 'gridvis': False, 'quiet': True}
        
        # Suppress httpx verbose logging during parallel downloads
        httpx_logger = logging.getLogger("httpx")
        original_httpx_level = httpx_logger.level
        httpx_logger.setLevel(logging.WARNING)
        
        def download_land_cover():
            grid = land_strategy.build_grid(
                cfg.rectangle_vertices, cfg.meshsize, cfg.output_dir,
                **{**lc_opts, **parallel_kwargs}
            )
            # Get effective source after download
            try:
                from .grids import get_last_effective_land_cover_source
                effective = get_last_effective_land_cover_source() or cfg.land_cover_source
            except Exception:
                effective = cfg.land_cover_source
            return ('land_cover', (grid, effective))
        
        def download_building():
            bh, bmin, bid, gdf_out = build_strategy.build_grids(
                cfg.rectangle_vertices, cfg.meshsize, cfg.output_dir,
                building_gdf=building_gdf,
                **{**bld_opts, **parallel_kwargs}
            )
            return ('building', (bh, bmin, bid, gdf_out))
        
        def download_canopy():
            # For non-static canopy, we don't need land_cover_grid
            # Pass None or empty array as placeholder - the strategy will download from GEE
            placeholder_grid = np.zeros((1, 1), dtype=float)
            top, bottom = canopy_strategy.build_grids(
                cfg.rectangle_vertices, cfg.meshsize, placeholder_grid, cfg.output_dir,
                land_cover_source=cfg.land_cover_source,
                **{**canopy_opts, **parallel_kwargs}
            )
            return ('canopy', (top, bottom))
        
        def download_dem():
            # DEM no longer depends on land_cover_like for shape
            dem = dem_strategy.build_grid(
                cfg.rectangle_vertices, cfg.meshsize, None, cfg.output_dir,
                terrain_gdf=terrain_gdf,
                **{**dem_opts, **parallel_kwargs}
            )
            return ('dem', dem)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(download_land_cover),
                executor.submit(download_building),
                executor.submit(download_canopy),
                executor.submit(download_dem),
            ]
            completed_count = 0
            for future in as_completed(futures):
                try:
                    key, value = future.result()
                    results[key] = value
                    completed_count += 1
                    print(f"  ✓ {key.replace('_', ' ').title()} complete ({completed_count}/4)")
                except Exception as e:
                    logger.error("Parallel download failed: %s", e)
                    httpx_logger.setLevel(original_httpx_level)  # Restore before raising
                    raise
        
        # Restore httpx logging level
        httpx_logger.setLevel(original_httpx_level)
        
        print("-"*60)
        print("All downloads complete!")
        print("="*60 + "\n")
        
        land_cover_grid, lc_src_effective = results['land_cover']
        bh, bmin, bid, building_gdf_out = results['building']
        canopy_top, canopy_bottom = results['canopy']
        dem = results['dem']
        
        return land_cover_grid, bh, bmin, bid, building_gdf_out, canopy_top, canopy_bottom, dem, lc_src_effective

    def _run_parallel_downloads_static_canopy(
        self, cfg, land_strategy, build_strategy, dem_strategy,
        building_gdf, terrain_gdf, kwargs
    ):
        """
        Run land_cover, building, and dem downloads in parallel.
        Canopy (Static mode) will be run sequentially after, as it needs land_cover_grid.
        """
        import logging
        logger = get_logger(__name__)
        
        # Print clean header for parallel mode
        print("\n" + "="*60)
        print("Downloading data in parallel mode (3 concurrent + 1 deferred)")
        print("="*60)
        print(f"  • Land Cover:  {cfg.land_cover_source}")
        print(f"  • Building:    {cfg.building_source}")
        print(f"  • DEM:         {cfg.dem_source}")
        print(f"  • Canopy:      {cfg.canopy_height_source} (deferred)")
        print("-"*60)
        print("Downloading... (this may take a moment)")
        
        results = {}
        
        # Disable gridvis and verbose prints in parallel mode
        parallel_kwargs = {**kwargs, 'gridvis': False, 'print_class_info': False, 'quiet': True}
        lc_opts = {**cfg.land_cover_options, 'gridvis': False, 'print_class_info': False, 'quiet': True}
        bld_opts = {**cfg.building_options, 'gridvis': False, 'quiet': True}
        dem_opts = {**cfg.dem_options, 'gridvis': False, 'quiet': True}
        
        # Suppress httpx verbose logging during parallel downloads
        httpx_logger = logging.getLogger("httpx")
        original_httpx_level = httpx_logger.level
        httpx_logger.setLevel(logging.WARNING)
        
        def download_land_cover():
            grid = land_strategy.build_grid(
                cfg.rectangle_vertices, cfg.meshsize, cfg.output_dir,
                **{**lc_opts, **parallel_kwargs}
            )
            try:
                from .grids import get_last_effective_land_cover_source
                effective = get_last_effective_land_cover_source() or cfg.land_cover_source
            except Exception:
                effective = cfg.land_cover_source
            return ('land_cover', (grid, effective))
        
        def download_building():
            bh, bmin, bid, gdf_out = build_strategy.build_grids(
                cfg.rectangle_vertices, cfg.meshsize, cfg.output_dir,
                building_gdf=building_gdf,
                **{**bld_opts, **parallel_kwargs}
            )
            return ('building', (bh, bmin, bid, gdf_out))
        
        def download_dem():
            dem = dem_strategy.build_grid(
                cfg.rectangle_vertices, cfg.meshsize, None, cfg.output_dir,
                terrain_gdf=terrain_gdf,
                **{**dem_opts, **parallel_kwargs}
            )
            return ('dem', dem)
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(download_land_cover),
                executor.submit(download_building),
                executor.submit(download_dem),
            ]
            completed_count = 0
            for future in as_completed(futures):
                try:
                    key, value = future.result()
                    results[key] = value
                    completed_count += 1
                    print(f"  ✓ {key.replace('_', ' ').title()} complete ({completed_count}/3)")
                except Exception as e:
                    logger.error("Parallel download failed: %s", e)
                    httpx_logger.setLevel(original_httpx_level)
                    raise
        
        # Restore httpx logging level
        httpx_logger.setLevel(original_httpx_level)
        
        print("-"*60)
        print("Parallel downloads complete! Processing canopy...")
        
        land_cover_grid, lc_src_effective = results['land_cover']
        bh, bmin, bid, building_gdf_out = results['building']
        dem = results['dem']
        
        return land_cover_grid, bh, bmin, bid, building_gdf_out, dem, lc_src_effective


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
        self.tree_gdf = None  # Store tree_gdf if created

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


class OSMCanopyStrategy(CanopySourceStrategy):
    """Canopy strategy that downloads individual tree data from OpenStreetMap."""
    
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self.tree_gdf = None  # Will store the downloaded tree GeoDataFrame
    
    def build_grids(self, rectangle_vertices, meshsize: float, land_cover_grid: np.ndarray, output_dir: str, **kwargs):
        from ..downloader.osm import load_tree_gdf_from_osm
        from ..geoprocessor.raster import create_canopy_grids_from_tree_gdf
        
        # Get OSM tree parameters from canopy_options or use defaults
        default_top_height = kwargs.get('default_top_height', 10.0)
        default_trunk_height = kwargs.get('default_trunk_height', 4.0)
        default_crown_diameter = kwargs.get('default_crown_diameter', None)
        default_crown_ratio = kwargs.get('default_crown_ratio', 0.6)
        
        # Download tree data from OpenStreetMap
        print("Downloading tree data from OpenStreetMap...")
        self.tree_gdf = load_tree_gdf_from_osm(
            rectangle_vertices,
            default_top_height=default_top_height,
            default_trunk_height=default_trunk_height,
            default_crown_diameter=default_crown_diameter,
            default_crown_ratio=default_crown_ratio,
        )
        print(f"  Downloaded {len(self.tree_gdf)} trees from OSM")
        
        # Create canopy height grids from tree_gdf
        if len(self.tree_gdf) == 0:
            # Return empty grids if no trees found
            if land_cover_grid is not None:
                return np.zeros_like(land_cover_grid, dtype=float), np.zeros_like(land_cover_grid, dtype=float)
            else:
                from ..geoprocessor.raster.core import compute_grid_shape
                grid_shape = compute_grid_shape(rectangle_vertices, meshsize)
                return np.zeros(grid_shape, dtype=float), np.zeros(grid_shape, dtype=float)
        
        canopy_top, canopy_bottom = create_canopy_grids_from_tree_gdf(
            self.tree_gdf, meshsize, rectangle_vertices
        )
        print(f"  Created canopy grids: {canopy_top.shape}")
        
        # Visualize canopy height grid (consistent with other sources)
        grid_vis = kwargs.get("gridvis", True)
        if grid_vis:
            from ..visualizer.grids import visualize_numerical_grid
            canopy_vis = canopy_top.copy()
            canopy_vis[canopy_vis == 0] = np.nan
            visualize_numerical_grid(np.flipud(canopy_vis), meshsize, "Tree canopy height (top)", cmap='Greens', label='Tree canopy height (m)')
        
        return canopy_top, canopy_bottom


class CanopySourceFactory:
    @staticmethod
    def create(source: str, cfg: PipelineConfig) -> CanopySourceStrategy:
        if source == "Static":
            return StaticCanopyStrategy(cfg)
        if source == "OpenStreetMap":
            return OSMCanopyStrategy(cfg)
        return SourceCanopyStrategy(source)


class DemSourceStrategy:  # ABC simplified
    def build_grid(self, rectangle_vertices, meshsize: float, land_cover_grid: np.ndarray, output_dir: str, **kwargs):  # pragma: no cover
        raise NotImplementedError


class FlatDemStrategy(DemSourceStrategy):
    def build_grid(self, rectangle_vertices, meshsize: float, land_cover_grid: np.ndarray, output_dir: str, **kwargs):
        # Compute shape from rectangle_vertices if land_cover_grid is None
        if land_cover_grid is None:
            from ..geoprocessor.raster.core import compute_grid_shape
            grid_shape = compute_grid_shape(rectangle_vertices, meshsize)
            return np.zeros(grid_shape, dtype=float)
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
            # Compute shape from rectangle_vertices if land_cover_grid is None
            if land_cover_grid is None:
                from ..geoprocessor.raster.core import compute_grid_shape
                grid_shape = compute_grid_shape(rectangle_vertices, meshsize)
                return np.zeros(grid_shape, dtype=float)
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


