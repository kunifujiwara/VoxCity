"""
Test LOD2 CityGML processing with Chiyoda-ku PLATEAU data.

Usage:
    python test_lod2.py                    # Default area (Tokyo Station)
    python test_lod2.py --area imperial    # Imperial Palace area
    python test_lod2.py --area marunouchi  # Marunouchi district
    python test_lod2.py --area ginza       # Ginza area
    python test_lod2.py --lon 139.76 --lat 35.68 --size 200  # Custom area
"""
import numpy as np
import os
import argparse

import ee
ee.Authenticate()
ee.Initialize(project='ee-project-250322')

# =============================================================================
# Predefined Test Areas in Chiyoda-ku
# =============================================================================
TEST_AREAS = {
    "tokyo_station": {
        "name": "Tokyo Station",
        "center": (139.7660, 35.6825),
        "size": 200,  # meters
    },
    "imperial": {
        "name": "Imperial Palace East Garden",
        "center": (139.7565, 35.6852),
        "size": 300,
    },
    "marunouchi": {
        "name": "Marunouchi District",
        "center": (139.7645, 35.6810),
        "size": 250,
    },
    "otemachi": {
        "name": "Otemachi",
        "center": (139.7660, 35.6870),
        "size": 200,
    },
    "hibiya": {
        "name": "Hibiya Park",
        "center": (139.7545, 35.6735),
        "size": 300,
    },
}

def create_rectangle(center_lon, center_lat, size_meters):
    """Create rectangle vertices from center point and size."""
    # Approximate conversion: 1 degree latitude ≈ 111km, longitude varies with latitude
    lat_offset = (size_meters / 2) / 111000
    lon_offset = (size_meters / 2) / (111000 * np.cos(np.radians(center_lat)))
    
    return [
        (center_lon - lon_offset, center_lat - lat_offset),  # SW
        (center_lon + lon_offset, center_lat - lat_offset),  # SE
        (center_lon + lon_offset, center_lat + lat_offset),  # NE
        (center_lon - lon_offset, center_lat + lat_offset),  # NW
    ]

# Parse command line arguments
parser = argparse.ArgumentParser(description="Test LOD2 CityGML processing")
parser.add_argument("--area", type=str, default="tokyo_station",
                    choices=list(TEST_AREAS.keys()),
                    help="Predefined test area")
parser.add_argument("--lon", type=float, help="Center longitude (custom area)")
parser.add_argument("--lat", type=float, help="Center latitude (custom area)")
parser.add_argument("--size", type=int, default=200, help="Area size in meters")
parser.add_argument("--meshsize", type=float, default=1.0, help="Voxel size in meters")
parser.add_argument("--no-vis", action="store_true", help="Skip visualization")
parser.add_argument("--lod", type=str, default=None, choices=["lod1", "lod2", "LOD1", "LOD2"],
                    help="LOD mode: 'lod1' or 'lod2'. If not specified, auto-detects.")
args = parser.parse_args()

# Determine the test area
if args.lon and args.lat:
    # Custom area
    area_name = f"Custom ({args.lon:.4f}, {args.lat:.4f})"
    center = (args.lon, args.lat)
    size = args.size
else:
    # Predefined area
    area_info = TEST_AREAS[args.area]
    area_name = area_info["name"]
    center = area_info["center"]
    size = args.size if args.size != 200 else area_info["size"]

rectangle_vertices = create_rectangle(center[0], center[1], size)

# Path to the PLATEAU data
citygml_path = r"data\13101_chiyoda-ku_pref_2023_citygml_2_op"

# Output directory
output_dir = "output/lod2_test"
os.makedirs(output_dir, exist_ok=True)

# Determine LOD mode
lod_mode = args.lod.lower() if args.lod else None
lod_display = args.lod.upper() if args.lod else "Auto"

print("=" * 60)
print(f"Testing CityGML Processing")
print("=" * 60)
print(f"Area: {area_name}")
print(f"Center: ({center[0]:.4f}, {center[1]:.4f})")
print(f"Size: {size}m x {size}m")
print(f"Voxel size: {args.meshsize}m")
print(f"LOD mode: {lod_display}")
print(f"Data path: {citygml_path}")
print(f"Rectangle: {rectangle_vertices}")
print()

# Test with the new LOD mode
from voxcity.generator.api import get_voxcity_CityGML

city = get_voxcity_CityGML(
    rectangle_vertices=rectangle_vertices,
    land_cover_source="OpenEarthMapJapan",
    canopy_height_source="Static",
    meshsize=args.meshsize,
    citygml_path=citygml_path,
    output_dir=output_dir,
    lod=lod_mode,
    include_bridges=True,
    include_city_furniture=False,
    include_lod2_vegetation=True,
    gridvis=False,
    save_voxcity_data=True,
    timing=True,  # Enable built-in timing report
    # dem_source="COPERNICUS",  # Use external DEM instead of CityGML terrain
)

print("\n" + "=" * 60)
print("Results")
print("=" * 60)

# VoxCity object structure
voxcity_grid = city.voxels.classes
print(f"VoxCity grid shape: {voxcity_grid.shape}")
print(f"Meshsize: {city.voxels.meta.meshsize}")

# Count voxel types
unique, counts = np.unique(voxcity_grid, return_counts=True)
print("\nVoxel counts by type:")
code_names = {
    0: "Empty",
    -1: "Ground",
    -2: "Tree",
    -3: "Building",
    -4: "Bridge",
    -5: "City Furniture",
}
for code, count in zip(unique, counts):
    name = code_names.get(code, f"Unknown ({code})")
    print(f"  {name}: {count:,}")

# Check extras  
extras = city.extras if hasattr(city, 'extras') else city.get('extras', {})
if extras:
    print("\nExtras:")
    for key, value in extras.items():
        if not isinstance(value, np.ndarray):
            print(f"  {key}: {value}")

print("\nTest completed successfully!")

# Visualize the result interactively
if not args.no_vis:
    print("\n" + "=" * 60)
    print("Launching interactive visualization...")
    print("=" * 60)

    from voxcity.visualizer import visualize_voxcity

    visualize_voxcity(
        city,
        mode="interactive",
        title=f"LOD2 {area_name}",
        opacity=1.0,
        max_dimension=300,
        show=True,
        downsample=1
    )
else:
    print("\nVisualization skipped (--no-vis flag).")
