"""
Timing analysis for get_voxcity_CityGML to identify bottlenecks.

This script measures the execution time of each step in the CityGML processing pipeline.
"""
import numpy as np
import os
import time
from functools import wraps
from contextlib import contextmanager
from collections import defaultdict

import ee
ee.Authenticate()
ee.Initialize(project='ee-project-250322')

# =============================================================================
# Timing Infrastructure
# =============================================================================
class TimingContext:
    """Global timing context for collecting measurements."""
    def __init__(self):
        self.timings = defaultdict(list)
        self.current_path = []
        
    def reset(self):
        self.timings = defaultdict(list)
        self.current_path = []
    
    @contextmanager
    def measure(self, name):
        """Context manager for timing a code block."""
        path = ".".join(self.current_path + [name])
        self.current_path.append(name)
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.timings[path].append(elapsed)
            self.current_path.pop()
    
    def report(self):
        """Generate a timing report."""
        print("\n" + "=" * 80)
        print("TIMING REPORT")
        print("=" * 80)
        
        # Calculate totals
        total_items = []
        for path, times in sorted(self.timings.items()):
            total = sum(times)
            count = len(times)
            avg = total / count if count > 0 else 0
            depth = path.count(".")
            total_items.append((path, total, count, avg, depth))
        
        # Sort by total time (descending)
        total_items.sort(key=lambda x: -x[1])
        
        # Print report
        max_path_len = max(len(item[0]) for item in total_items) if total_items else 40
        print(f"{'Step':<{max_path_len}}  {'Total (s)':>10}  {'Count':>6}  {'Avg (s)':>10}  {'%':>6}")
        print("-" * (max_path_len + 40))
        
        # Get top-level total for percentage calculation
        top_total = sum(t for p, t, _, _, d in total_items if d == 0)
        
        for path, total, count, avg, depth in total_items:
            indent = "  " * depth
            short_name = path.split(".")[-1] if "." in path else path
            display = f"{indent}{short_name}"
            pct = (total / top_total * 100) if top_total > 0 else 0
            print(f"{display:<{max_path_len}}  {total:>10.3f}  {count:>6}  {avg:>10.3f}  {pct:>5.1f}%")
        
        print("-" * (max_path_len + 40))
        print(f"{'TOTAL':<{max_path_len}}  {top_total:>10.3f}")
        print("=" * 80)
        
        # Print bottleneck summary
        print("\n🔍 TOP 5 BOTTLENECKS (by total time):")
        for i, (path, total, count, avg, depth) in enumerate(total_items[:5], 1):
            pct = (total / top_total * 100) if top_total > 0 else 0
            print(f"  {i}. {path}: {total:.3f}s ({pct:.1f}%)")


TIMING = TimingContext()

# =============================================================================
# Patched Functions with Timing
# =============================================================================

def patch_voxelize_functions():
    """Patch the main voxelization functions to add timing."""
    from voxcity.geoprocessor.citygml import voxelizer as vox_module
    
    # Store originals
    orig_voxelize_buildings = vox_module.voxelize_buildings_citygml
    orig_voxelize_trees = vox_module.voxelize_trees_citygml
    orig_voxelize_terrain = vox_module.voxelize_terrain_citygml
    orig_apply_post = vox_module.apply_citygml_post_processing
    orig_merge_lod2 = vox_module.merge_lod2_voxels
    
    def timed_voxelize_buildings(*args, **kwargs):
        with TIMING.measure("voxelize_buildings"):
            # Sub-timings inside the function
            result = orig_voxelize_buildings(*args, **kwargs)
        return result
    
    def timed_voxelize_trees(*args, **kwargs):
        with TIMING.measure("voxelize_trees"):
            result = orig_voxelize_trees(*args, **kwargs)
        return result
    
    def timed_voxelize_terrain(*args, **kwargs):
        with TIMING.measure("voxelize_terrain"):
            result = orig_voxelize_terrain(*args, **kwargs)
        return result
    
    def timed_apply_post(*args, **kwargs):
        with TIMING.measure("apply_post_processing"):
            result = orig_apply_post(*args, **kwargs)
        return result
    
    def timed_merge_lod2(*args, **kwargs):
        with TIMING.measure("merge_lod2_voxels"):
            result = orig_merge_lod2(*args, **kwargs)
        return result
    
    # Apply patches
    vox_module.voxelize_buildings_citygml = timed_voxelize_buildings
    vox_module.voxelize_trees_citygml = timed_voxelize_trees
    vox_module.voxelize_terrain_citygml = timed_voxelize_terrain
    vox_module.apply_citygml_post_processing = timed_apply_post
    vox_module.merge_lod2_voxels = timed_merge_lod2


def patch_parser_functions():
    """Patch parser functions for detailed timing."""
    from voxcity.geoprocessor.citygml import parsers as parser_module
    
    orig_load_lod1 = parser_module.load_lod1_citygml
    
    def timed_load_lod1(*args, **kwargs):
        with TIMING.measure("load_lod1_citygml"):
            result = orig_load_lod1(*args, **kwargs)
        return result
    
    parser_module.load_lod1_citygml = timed_load_lod1


def patch_plateau_voxelizer():
    """Patch PLATEAUVoxelizer methods for detailed timing."""
    from voxcity.geoprocessor.citygml.voxelizer import PLATEAUVoxelizer
    
    orig_parse = PLATEAUVoxelizer.parse_plateau_directory
    orig_voxelize = PLATEAUVoxelizer.voxelize
    
    def timed_parse(self, *args, **kwargs):
        with TIMING.measure("PLATEAUVoxelizer.parse_plateau_directory"):
            result = orig_parse(self, *args, **kwargs)
        return result
    
    def timed_voxelize(self, *args, **kwargs):
        with TIMING.measure("PLATEAUVoxelizer.voxelize"):
            result = orig_voxelize(self, *args, **kwargs)
        return result
    
    PLATEAUVoxelizer.parse_plateau_directory = timed_parse
    PLATEAUVoxelizer.voxelize = timed_voxelize


def patch_grid_functions():
    """Patch grid generation functions for timing."""
    from voxcity.generator import grids as grids_module
    
    if hasattr(grids_module, 'get_land_cover_grid'):
        orig_lc = grids_module.get_land_cover_grid
        def timed_lc(*args, **kwargs):
            with TIMING.measure("get_land_cover_grid"):
                return orig_lc(*args, **kwargs)
        grids_module.get_land_cover_grid = timed_lc
    
    if hasattr(grids_module, 'get_canopy_height_grid'):
        orig_ch = grids_module.get_canopy_height_grid
        def timed_ch(*args, **kwargs):
            with TIMING.measure("get_canopy_height_grid"):
                return orig_ch(*args, **kwargs)
        grids_module.get_canopy_height_grid = timed_ch
    
    if hasattr(grids_module, 'get_dem_grid'):
        orig_dem = grids_module.get_dem_grid
        def timed_dem(*args, **kwargs):
            with TIMING.measure("get_dem_grid"):
                return orig_dem(*args, **kwargs)
        grids_module.get_dem_grid = timed_dem


def patch_raster_functions():
    """Patch raster processing functions for timing."""
    from voxcity.geoprocessor import raster as raster_module
    
    if hasattr(raster_module, 'create_building_height_grid_from_gdf_polygon'):
        orig_bh = raster_module.create_building_height_grid_from_gdf_polygon
        def timed_bh(*args, **kwargs):
            with TIMING.measure("create_building_height_grid"):
                return orig_bh(*args, **kwargs)
        raster_module.create_building_height_grid_from_gdf_polygon = timed_bh
    
    if hasattr(raster_module, 'create_vegetation_height_grid_from_gdf_polygon'):
        orig_vh = raster_module.create_vegetation_height_grid_from_gdf_polygon
        def timed_vh(*args, **kwargs):
            with TIMING.measure("create_vegetation_height_grid"):
                return orig_vh(*args, **kwargs)
        raster_module.create_vegetation_height_grid_from_gdf_polygon = timed_vh
    
    if hasattr(raster_module, 'create_dem_grid_from_gdf_polygon'):
        orig_dg = raster_module.create_dem_grid_from_gdf_polygon
        def timed_dg(*args, **kwargs):
            with TIMING.measure("create_dem_grid_from_gdf"):
                return orig_dg(*args, **kwargs)
        raster_module.create_dem_grid_from_gdf_polygon = timed_dg
    
    if hasattr(raster_module, 'process_grid'):
        orig_pg = raster_module.process_grid
        def timed_pg(*args, **kwargs):
            with TIMING.measure("process_grid_flatten"):
                return orig_pg(*args, **kwargs)
        raster_module.process_grid = timed_pg


def patch_voxelizer_generate():
    """Patch Voxelizer.generate_combined for timing."""
    from voxcity.generator.voxelizer import Voxelizer
    
    orig_generate = Voxelizer.generate_combined
    
    def timed_generate(self, *args, **kwargs):
        with TIMING.measure("Voxelizer.generate_combined"):
            return orig_generate(self, *args, **kwargs)
    
    Voxelizer.generate_combined = timed_generate


# =============================================================================
# Test Configuration
# =============================================================================
TEST_AREAS = {
    "tokyo_station": {
        "name": "Tokyo Station",
        "center": (139.7660, 35.6825),
        "size": 200,
    },
    "marunouchi": {
        "name": "Marunouchi District",
        "center": (139.7645, 35.6810),
        "size": 250,
    },
    "small": {
        "name": "Small Test",
        "center": (139.7660, 35.6825),
        "size": 100,
    },
}


def create_rectangle(center_lon, center_lat, size_meters):
    """Create rectangle vertices from center point and size."""
    lat_offset = (size_meters / 2) / 111000
    lon_offset = (size_meters / 2) / (111000 * np.cos(np.radians(center_lat)))
    
    return [
        (center_lon - lon_offset, center_lat - lat_offset),
        (center_lon + lon_offset, center_lat - lat_offset),
        (center_lon + lon_offset, center_lat + lat_offset),
        (center_lon - lon_offset, center_lat + lat_offset),
    ]


def run_timing_test(area_name="tokyo_station", size=None, meshsize=1.0, use_lod2=True):
    """Run timing test for a given area."""
    
    # Reset timing
    TIMING.reset()
    
    # Apply patches (only for non-optimized functions to track internal timings)
    patch_parser_functions()
    patch_plateau_voxelizer()
    patch_grid_functions()
    patch_raster_functions()
    patch_voxelizer_generate()
    
    # Get area config
    area_info = TEST_AREAS.get(area_name, TEST_AREAS["tokyo_station"])
    center = area_info["center"]
    test_size = size if size else area_info["size"]
    
    rectangle_vertices = create_rectangle(center[0], center[1], test_size)
    
    # Paths
    citygml_path = r"data\13101_chiyoda-ku_pref_2023_citygml_2_op"
    output_dir = "output/timing_test"
    os.makedirs(output_dir, exist_ok=True)
    
    lod_mode = "LOD2" if use_lod2 else "LOD1"
    
    print("=" * 80)
    print(f"TIMING TEST: {area_info['name']} ({lod_mode} mode)")
    print("=" * 80)
    print(f"Area: {test_size}m x {test_size}m")
    print(f"Meshsize: {meshsize}m")
    print(f"Expected grid size: ~{int(test_size/meshsize)} x {int(test_size/meshsize)}")
    print()
    
    # Import and run with timing enabled
    from voxcity.generator.api import get_voxcity_CityGML
    
    # Time the entire process - use built-in timing option
    with TIMING.measure("TOTAL"):
        city = get_voxcity_CityGML(
            rectangle_vertices=rectangle_vertices,
            land_cover_source="OpenEarthMapJapan",
            canopy_height_source="Static",
            meshsize=meshsize,
            citygml_path=citygml_path,
            output_dir=output_dir,
            use_lod2=use_lod2,
            include_bridges=use_lod2,
            include_city_furniture=False,
            include_lod2_vegetation=use_lod2,
            gridvis=False,
            save_voxcity_data=False,  # Skip saving for timing
            timing=True,  # Enable built-in timing
        )
    
    # Print results summary
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"VoxCity grid shape: {city.voxels.classes.shape}")
    
    unique, counts = np.unique(city.voxels.classes, return_counts=True)
    code_names = {0: "Empty", -1: "Ground", -2: "Tree", -3: "Building", -4: "Bridge", -5: "City Furniture"}
    for code, count in zip(unique, counts):
        name = code_names.get(code, f"Unknown ({code})")
        print(f"  {name}: {count:,}")
    
    # Print timing report
    TIMING.report()
    
    return city


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Timing analysis for CityGML processing")
    parser.add_argument("--area", type=str, default="tokyo_station",
                        choices=list(TEST_AREAS.keys()),
                        help="Test area")
    parser.add_argument("--size", type=int, help="Override area size (meters)")
    parser.add_argument("--meshsize", type=float, default=1.0, help="Voxel size (meters)")
    parser.add_argument("--lod1", action="store_true", help="Use LOD1 mode instead of LOD2")
    
    args = parser.parse_args()
    
    run_timing_test(
        area_name=args.area,
        size=args.size,
        meshsize=args.meshsize,
        use_lod2=not args.lod1,
    )
