"""
Test LOD2 CityGML processing with PLATEAU (Japanese) and Generic (German) data.

Usage:
    python test_lod2.py                    # Default area (Tokyo Station, PLATEAU)
    python test_lod2.py --area imperial    # Imperial Palace area (PLATEAU)
    python test_lod2.py --data german      # German CityGML data
    python test_lod2.py --data german --area nuremberg_center  # Nuremberg center
    python test_lod2.py --lon 139.76 --lat 35.68 --size 200  # Custom area
"""
import numpy as np
import os
import argparse

# =============================================================================
# Predefined Test Areas
# =============================================================================

# Japanese PLATEAU data (Chiyoda-ku, Tokyo)
PLATEAU_AREAS = {
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

# German CityGML data (Bavaria/Nuremberg area)
GERMAN_AREAS = {
    "nuremberg_center": {
        "name": "Nuremberg Center",
        "center": (11.0767, 35.4489),  # Approximate center of the data
        "size": 500,
    },
    "german_default": {
        "name": "German CityGML Sample",
        "center": (11.08, 49.45),  # Center based on data bounds
        "size": 300,
    },
    "german_small": {
        "name": "German Small Test",
        "center": (11.08, 49.45),
        "size": 100,
    },
}

# Data source configurations
DATA_SOURCES = {
    "plateau": {
        "path": r"data\13101_chiyoda-ku_pref_2023_citygml_2_op",
        "areas": PLATEAU_AREAS,
        "default_area": "tokyo_station",
        "land_cover": "OpenEarthMapJapan",
        "canopy": "Static",
        "requires_ee": True,
    },
    "german": {
        "path": r"data\download_20251223_063810",
        "areas": GERMAN_AREAS,
        "default_area": "german_default",
        "land_cover": "OpenStreetMap",
        "canopy": "Static",
        "requires_ee": False,
        "dem_source": r"data\download_20251223_063810\citygml_area_dgm1.tif",
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
parser.add_argument("--data", type=str, default="plateau",
                    choices=list(DATA_SOURCES.keys()),
                    help="Data source: 'plateau' (Japanese) or 'german' (European)")
parser.add_argument("--area", type=str, default=None,
                    help="Predefined test area (depends on data source)")
parser.add_argument("--lon", type=float, help="Center longitude (custom area)")
parser.add_argument("--lat", type=float, help="Center latitude (custom area)")
parser.add_argument("--size", type=int, default=None, help="Area size in meters")
parser.add_argument("--meshsize", type=float, default=1.0, help="Voxel size in meters")
parser.add_argument("--no-vis", action="store_true", help="Skip visualization")
parser.add_argument("--no-gpu", action="store_true", help="Skip GPU rendering")
parser.add_argument("--lod", type=str, default=None, choices=["lod1", "lod2", "LOD1", "LOD2"],
                    help="LOD mode: 'lod1' or 'lod2'. If not specified, auto-detects.")
parser.add_argument("--list-areas", action="store_true", help="List available areas and exit")
args = parser.parse_args()

# Get data source configuration
data_config = DATA_SOURCES[args.data]

# List areas if requested
if args.list_areas:
    print(f"\nAvailable areas for '{args.data}' data source:")
    print("=" * 50)
    for area_key, area_info in data_config["areas"].items():
        print(f"  {area_key:20s} - {area_info['name']} ({area_info['size']}m)")
    print()
    exit(0)

# Initialize Earth Engine only if needed
if data_config["requires_ee"]:
    import ee
    ee.Authenticate()
    ee.Initialize(project='ee-project-250322')

# Determine the test area
if args.lon and args.lat:
    # Custom area
    area_name = f"Custom ({args.lon:.4f}, {args.lat:.4f})"
    center = (args.lon, args.lat)
    size = args.size if args.size else 200
else:
    # Predefined area
    area_key = args.area if args.area else data_config["default_area"]
    if area_key not in data_config["areas"]:
        print(f"Error: Area '{area_key}' not found for data source '{args.data}'")
        print(f"Available areas: {list(data_config['areas'].keys())}")
        exit(1)
    area_info = data_config["areas"][area_key]
    area_name = area_info["name"]
    center = area_info["center"]
    size = args.size if args.size else area_info["size"]

rectangle_vertices = create_rectangle(center[0], center[1], size)

# Path to the CityGML data
citygml_path = data_config["path"]

# Land cover and canopy sources
land_cover_source = data_config["land_cover"]
canopy_source = data_config["canopy"]
dem_source = data_config.get("dem_source", None)  # Local DEM file if available

# Output directory
output_dir = f"output/lod2_test_{args.data}"
os.makedirs(output_dir, exist_ok=True)

# Determine LOD mode
lod_mode = args.lod.lower() if args.lod else None
lod_display = args.lod.upper() if args.lod else "Auto"

print("=" * 60)
print(f"Testing CityGML Processing ({args.data.upper()} format)")
print("=" * 60)
print(f"Data source: {args.data}")
print(f"Area: {area_name}")
print(f"Center: ({center[0]:.4f}, {center[1]:.4f})")
print(f"Size: {size}m x {size}m")
print(f"Voxel size: {args.meshsize}m")
print(f"LOD mode: {lod_display}")
print(f"Land cover: {land_cover_source}")
print(f"Canopy: {canopy_source}")
print(f"DEM source: {dem_source if dem_source else 'CityGML terrain'}")
print(f"Data path: {citygml_path}")
print(f"Rectangle: {rectangle_vertices}")
print()

# Test with the new LOD mode
from voxcity.generator.api import get_voxcity_CityGML

city = get_voxcity_CityGML(
    rectangle_vertices=rectangle_vertices,
    land_cover_source=land_cover_source,
    canopy_height_source=canopy_source,
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
    dem_source=dem_source,  # Use local DEM file for German data
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
        title=f"{args.data.upper()} {area_name}",
        opacity=1.0,
        max_dimension=300,
        show=True,
        downsample=1
    )
else:
    print("\nVisualization skipped (--no-vis flag).")

# GPU Rendering
if not args.no_gpu:
    print("\n" + "=" * 60)
    print("GPU Rendering...")
    print("=" * 60)

    from voxcity.visualizer.renderer_gpu import visualize_voxcity_gpu

    # Single image rendering
    gpu_output_path = os.path.join(output_dir, f"gpu_render_{args.data}.png")

    img = visualize_voxcity_gpu(
        city,
        voxel_color_map="default",
        width=1920,
        height=1080,
        samples_per_pixel=64,
        output_path=gpu_output_path,
        show_progress=True,
    )

    print(f"\nGPU rendered image saved to: {gpu_output_path}")
else:
    print("\nGPU rendering skipped (--no-gpu flag).")
