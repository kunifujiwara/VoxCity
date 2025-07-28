#!/usr/bin/env python3
"""
Example script demonstrating how to use get_building_height_grid with a GeoDataFrame.

This example shows how to:
1. Create a GeoDataFrame with building footprints and heights
2. Use the GeoDataFrame directly with get_building_height_grid
3. Generate building height grids from custom data
"""

import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import os

# Import voxcity functions
from src.voxcity.generator import get_building_height_grid, get_voxcity

def create_sample_building_gdf():
    """
    Create a sample GeoDataFrame with building footprints and heights.
    
    Returns:
        gpd.GeoDataFrame: Sample building data with geometry and height columns
    """
    # Define sample building footprints as polygons
    buildings = [
        # Building 1: Simple rectangle
        Polygon([(0, 0), (10, 0), (10, 15), (0, 15)]),
        # Building 2: Another rectangle
        Polygon([(15, 5), (25, 5), (25, 20), (15, 20)]),
        # Building 3: Complex shape
        Polygon([(30, 0), (40, 0), (40, 10), (35, 10), (35, 15), (30, 15)]),
    ]
    
    # Define corresponding heights
    heights = [20.0, 15.0, 25.0]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame({
        'geometry': buildings,
        'height': heights,
        'id': [1, 2, 3],
        'min_height': [0, 0, 0],  # Ground level
        'is_inner': [False, False, False]  # Not inner courtyards
    }, crs='EPSG:4326')
    
    return gdf

def example_direct_gdf_usage():
    """
    Example of using get_building_height_grid with a GeoDataFrame directly.
    """
    print("=== Example: Using get_building_height_grid with GeoDataFrame ===")
    
    # Create sample building data
    building_gdf = create_sample_building_gdf()
    print(f"Created sample GeoDataFrame with {len(building_gdf)} buildings")
    print(building_gdf)
    
    # Define area of interest (rectangle vertices)
    rectangle_vertices = [
        (0, 0),    # Bottom-left
        (50, 0),   # Bottom-right
        (50, 25),  # Top-right
        (0, 25)    # Top-left
    ]
    
    # Parameters
    meshsize = 2.0  # 2-meter grid cells
    source = "GeoDataFrame"  # Indicate we're using a GeoDataFrame
    output_dir = "output_example"
    
    # Generate building height grid using the GeoDataFrame
    building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = get_building_height_grid(
        rectangle_vertices=rectangle_vertices,
        meshsize=meshsize,
        source=source,
        output_dir=output_dir,
        building_gdf=building_gdf,
        gridvis=True  # Show visualization
    )
    
    print(f"\nGenerated building height grid with shape: {building_height_grid.shape}")
    print(f"Maximum building height: {np.max(building_height_grid)} meters")
    print(f"Number of buildings detected: {len(np.unique(building_id_grid[building_id_grid > 0]))}")
    
    return building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings

def example_voxcity_with_gdf():
    """
    Example of using get_voxcity with a GeoDataFrame for building data.
    """
    print("\n=== Example: Using get_voxcity with GeoDataFrame ===")
    
    # Create sample building data
    building_gdf = create_sample_building_gdf()
    
    # Define area of interest
    rectangle_vertices = [
        (0, 0),    # Bottom-left
        (50, 0),   # Bottom-right
        (50, 25),  # Top-right
        (0, 25)    # Top-left
    ]
    
    # Parameters
    meshsize = 2.0
    building_source = "GeoDataFrame"
    land_cover_source = "Static"  # Use static land cover for simplicity
    canopy_height_source = "Static"
    dem_source = "Flat"
    
    # Generate complete voxel city model
    voxcity_grid, building_height_grid, building_min_height_grid, building_id_grid, \
    canopy_height_grid, land_cover_grid, dem_grid, building_gdf_result = get_voxcity(
        rectangle_vertices=rectangle_vertices,
        building_source=building_source,
        land_cover_source=land_cover_source,
        canopy_height_source=canopy_height_source,
        dem_source=dem_source,
        meshsize=meshsize,
        building_gdf=building_gdf,
        output_dir="output_voxcity_example",
        gridvis=True,
        voxelvis=True,
        voxelvis_img_save_path="output_voxcity_example/3d_visualization.png"
    )
    
    print(f"Generated voxel city model with shape: {voxcity_grid.shape}")
    print(f"Voxel grid contains {np.sum(voxcity_grid != 0)} non-empty voxels")
    
    return voxcity_grid, building_height_grid, land_cover_grid, dem_grid

if __name__ == "__main__":
    # Run examples
    try:
        # Example 1: Direct usage with GeoDataFrame
        building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = example_direct_gdf_usage()
        
        # Example 2: Complete voxel city generation
        voxcity_grid, building_height_grid, land_cover_grid, dem_grid = example_voxcity_with_gdf()
        
        print("\n=== Examples completed successfully! ===")
        print("Check the output directories for generated files and visualizations.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc() 