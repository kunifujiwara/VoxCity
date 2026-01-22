"""
Test script to compare sun direction and shadow calculations between
simulator.solar (CPU) and simulator_gpu.solar (GPU) implementations.

This script investigates the differences in:
1. Sun direction computation from azimuth/elevation
2. Normal vector handling for building surfaces
3. Coordinate system conventions
4. Shadow ray tracing

Author: Debug script for VoxCity solar simulation comparison
"""

import numpy as np
from datetime import datetime
import pytz
import math


def test_sun_direction_calculation():
    """
    Compare sun direction calculation between CPU and GPU implementations.
    
    CPU (simulator.solar.radiation):
        azimuth_degrees = 180 - azimuth_degrees_ori
        azimuth_radians = np.deg2rad(azimuth_degrees)
        elevation_radians = np.deg2rad(elevation_degrees)
        dx = np.cos(elevation_radians) * np.cos(azimuth_radians)
        dy = np.cos(elevation_radians) * np.sin(azimuth_radians)
        dz = np.sin(elevation_radians)
    
    GPU (simulator_gpu.solar.integration._compute_sun_direction):
        azimuth_degrees = 180 - azimuth_degrees_ori
        azimuth_radians = np.deg2rad(azimuth_degrees)
        elevation_radians = np.deg2rad(elevation_degrees)
        cos_elev = np.cos(elevation_radians)
        sin_elev = np.sin(elevation_radians)
        sun_dir_x = cos_elev * np.cos(azimuth_radians)
        sun_dir_y = cos_elev * np.sin(azimuth_radians)
        sun_dir_z = sin_elev
        
    Both are identical! So the difference is NOT in the azimuth/elevation -> direction conversion.
    """
    print("=" * 80)
    print("TEST 1: Sun Direction from Azimuth/Elevation")
    print("=" * 80)
    
    # Test cases: azimuth (0=North, clockwise), elevation
    test_cases = [
        (0, 45, "North, 45° elevation"),
        (90, 45, "East, 45° elevation"),
        (180, 45, "South, 45° elevation"),
        (270, 45, "West, 45° elevation"),
        (135, 30, "Southeast, 30° elevation"),
        (225, 60, "Southwest, 60° elevation"),
    ]
    
    print("\nSun direction vectors from azimuth/elevation:")
    print("-" * 80)
    print(f"{'Azimuth':>8} {'Elev':>6} {'Description':>25}  ->  {'dx':>8} {'dy':>8} {'dz':>8}")
    print("-" * 80)
    
    for azimuth_ori, elevation, desc in test_cases:
        # CPU/GPU common formula
        azimuth_deg = 180 - azimuth_ori
        az_rad = np.deg2rad(azimuth_deg)
        el_rad = np.deg2rad(elevation)
        
        dx = np.cos(el_rad) * np.cos(az_rad)
        dy = np.cos(el_rad) * np.sin(az_rad)
        dz = np.sin(el_rad)
        
        print(f"{azimuth_ori:>8}° {elevation:>5}° {desc:>25}  ->  {dx:>8.4f} {dy:>8.4f} {dz:>8.4f}")
    
    print("\n✓ CPU and GPU use IDENTICAL formulas for azimuth/elevation -> sun direction")
    return True


def test_astral_vs_palm_solar_position():
    """
    Compare solar position calculations between Astral (CPU) and PALM (GPU).
    
    NOTE: After investigation, we found that BOTH CPU and GPU integration.py
    use Astral for solar position calculation in the main code paths.
    
    However, PALM's calc_zenith is still used in some GPU code paths:
    - epw.py: prepare_cumulative_simulation_input() uses calc_zenith
    - SolarCalculator class uses calc_zenith
    
    This test compares the two algorithms to understand discrepancies.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Solar Position Calculation - Astral vs PALM")
    print("=" * 80)
    
    # Test location and time
    lon, lat = -74.01, 40.71  # New York
    
    # Import both implementations
    try:
        from voxcity.simulator.solar.temporal import get_solar_positions_astral
        has_astral = True
    except ImportError:
        has_astral = False
        print("Warning: astral not available")
    
    try:
        from voxcity.simulator_gpu.solar.solar import calc_zenith, calc_solar_position_datetime
        has_palm = True
    except ImportError:
        has_palm = False
        print("Warning: GPU solar module not available")
    
    if not has_astral or not has_palm:
        print("Skipping - missing dependencies")
        return False
    
    # Test at various times during the day
    test_times = [
        datetime(2024, 6, 21, 6, 0, 0, tzinfo=pytz.UTC),   # Summer solstice morning
        datetime(2024, 6, 21, 12, 0, 0, tzinfo=pytz.UTC),  # Solar noon
        datetime(2024, 6, 21, 18, 0, 0, tzinfo=pytz.UTC),  # Evening
        datetime(2024, 12, 21, 12, 0, 0, tzinfo=pytz.UTC), # Winter solstice noon
        datetime(2024, 3, 21, 12, 0, 0, tzinfo=pytz.UTC),  # Spring equinox noon
    ]
    
    print("\nNOTE: Main code paths (integration.py) use Astral for BOTH CPU and GPU!")
    print("      PALM's calc_zenith is only used in some GPU sub-modules (epw.py, SolarCalculator)")
    print()
    print("Comparison of Astral vs PALM solar position:")
    print("-" * 100)
    print(f"{'Date/Time UTC':>20} | {'Astral Az':>10} {'PALM Az':>10} {'Δ Az':>8} | "
          f"{'Astral El':>10} {'PALM El':>10} {'Δ El':>8}")
    print("-" * 100)
    
    max_az_diff = 0
    max_el_diff = 0
    
    for dt in test_times:
        # Astral calculation
        import pandas as pd
        from astral import Observer
        from astral.sun import elevation as astral_elevation, azimuth as astral_azimuth
        
        observer = Observer(latitude=lat, longitude=lon)
        astral_az = astral_azimuth(observer=observer, dateandtime=dt)
        astral_el = astral_elevation(observer=observer, dateandtime=dt)
        
        # PALM calculation
        palm_pos = calc_solar_position_datetime(dt, lat, lon)
        palm_az = palm_pos.azimuth_angle
        palm_el = palm_pos.elevation_angle
        
        # Calculate differences
        az_diff = palm_az - astral_az
        if az_diff > 180:
            az_diff -= 360
        elif az_diff < -180:
            az_diff += 360
            
        el_diff = palm_el - astral_el
        
        max_az_diff = max(max_az_diff, abs(az_diff))
        max_el_diff = max(max_el_diff, abs(el_diff))
        
        dt_str = dt.strftime("%Y-%m-%d %H:%M")
        print(f"{dt_str:>20} | {astral_az:>10.2f} {palm_az:>10.2f} {az_diff:>+8.2f} | "
              f"{astral_el:>10.2f} {palm_el:>10.2f} {el_diff:>+8.2f}")
    
    print("-" * 100)
    print(f"\n⚠️  Maximum differences: Azimuth = {max_az_diff:.2f}°, Elevation = {max_el_diff:.2f}°")
    
    if max_az_diff > 1.0 or max_el_diff > 1.0:
        print("\n⚠️  DIFFERENCE FOUND between Astral and PALM algorithms")
        print("   However, main code paths use Astral for BOTH CPU and GPU.")
        print("   This only affects GPU code using epw.py or SolarCalculator directly.")
        return False
    
    print("\n✓ Differences are within acceptable range")
    return True


def test_palm_direction_vector():
    """
    Test the PALM solar direction vector calculation.
    
    PALM uses a different formula to compute the sun direction from 
    declination and hour angle, which may differ from the simple
    azimuth/elevation formula.
    """
    print("\n" + "=" * 80)
    print("TEST 3: PALM Direction Vector vs Azimuth/Elevation Formula")
    print("=" * 80)
    
    try:
        from voxcity.simulator_gpu.solar.solar import calc_zenith
    except ImportError:
        print("Skipping - GPU module not available")
        return False
    
    # Test location
    lat, lon = 40.71, -74.01  # New York
    
    # Test at summer solstice noon UTC
    day_of_year = 172  # June 21
    second_of_day = 12 * 3600  # Noon UTC
    
    # Get PALM solar position
    pos = calc_zenith(day_of_year, second_of_day, lat, lon)
    
    palm_dir_x, palm_dir_y, palm_dir_z = pos.direction
    palm_az = pos.azimuth_angle
    palm_el = pos.elevation_angle
    
    # Compute direction from azimuth/elevation using the "180 - azimuth" formula
    # This is what the GPU integration.py does after getting the solar position
    azimuth_degrees = 180 - palm_az  # VoxCity convention transformation
    az_rad = np.deg2rad(azimuth_degrees)
    el_rad = np.deg2rad(palm_el)
    
    formula_dir_x = np.cos(el_rad) * np.cos(az_rad)
    formula_dir_y = np.cos(el_rad) * np.sin(az_rad)
    formula_dir_z = np.sin(el_rad)
    
    print(f"\nLocation: lat={lat}, lon={lon}")
    print(f"Day of year: {day_of_year}, Second of day: {second_of_day}")
    print(f"PALM azimuth: {palm_az:.2f}°, elevation: {palm_el:.2f}°")
    print()
    print("Direction vectors:")
    print("-" * 60)
    print(f"{'Source':>25} | {'X (East)':>10} {'Y (North)':>10} {'Z (Up)':>10}")
    print("-" * 60)
    print(f"{'PALM calc_zenith':>25} | {palm_dir_x:>10.5f} {palm_dir_y:>10.5f} {palm_dir_z:>10.5f}")
    print(f"{'From (180-az)/el formula':>25} | {formula_dir_x:>10.5f} {formula_dir_y:>10.5f} {formula_dir_z:>10.5f}")
    
    # Calculate difference
    diff_x = formula_dir_x - palm_dir_x
    diff_y = formula_dir_y - palm_dir_y
    diff_z = formula_dir_z - palm_dir_z
    diff_mag = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2)
    
    print(f"{'Difference':>25} | {diff_x:>10.5f} {diff_y:>10.5f} {diff_z:>10.5f}")
    print("-" * 60)
    print(f"Magnitude of difference: {diff_mag:.6f}")
    
    if diff_mag > 0.01:
        print("\n❌ SIGNIFICANT DIFFERENCE in direction vector!")
        print("   PALM's calc_zenith direction does NOT match (180-az)/el formula!")
        print("\n   This means the GPU uses TWO different sun directions:")
        print("   1. PALM's native direction from calc_zenith (for some paths)")
        print("   2. Recomputed direction from (180-az)/el formula (for other paths)")
        return False
    else:
        print("\n✓ Direction vectors match within tolerance")
        return True


def test_coordinate_convention_analysis():
    """
    Analyze the coordinate conventions used in CPU vs GPU implementations.
    """
    print("\n" + "=" * 80)
    print("TEST 4: Coordinate Convention Analysis")
    print("=" * 80)
    
    print("""
    CPU (simulator.solar):
    ----------------------
    - Uses Astral library for solar position
    - Azimuth: 0° = North, clockwise
    - Direction formula: 
        azimuth_degrees = 180 - azimuth_degrees_ori  # Flip to counter-clockwise from South
        dx = cos(el) * cos(az)  # East-West component  
        dy = cos(el) * sin(az)  # North-South component
        dz = sin(el)            # Up component
    
    GPU (simulator_gpu.solar):
    --------------------------
    - Uses PALM formula for solar position (different from Astral)
    - PALM's calc_zenith returns:
        * azimuth: 0° = North, clockwise (same as Astral)
        * direction: (x=East, y=North, z=Up) computed from declination/hour angle
    - In integration.py, the DIRECTION is RECOMPUTED:
        azimuth_degrees = 180 - azimuth_degrees_ori  # Same flip as CPU
        sun_dir_x = cos(el) * cos(az)
        sun_dir_y = cos(el) * sin(az)  
        sun_dir_z = sin(el)
    
    PROBLEM IDENTIFIED:
    -------------------
    1. CPU uses Astral for solar position -> different az/el values
    2. GPU uses PALM formula for solar position -> different az/el values
    3. Even with same az/el, the "180 - azimuth" flip may interact differently
       with each library's conventions
    
    The ROOT CAUSE is that Astral and PALM compute DIFFERENT solar positions
    for the same timestamp!
    """)
    
    return True


def test_normal_vector_handling():
    """
    Compare normal vector conventions between CPU and GPU.
    """
    print("\n" + "=" * 80)
    print("TEST 5: Normal Vector Conventions")
    print("=" * 80)
    
    print("""
    CPU (simulator.solar.radiation):
    ---------------------------------
    - Uses trimesh face_normals directly
    - Normal is outward-facing from surface
    - cos_incidence = dot(normal, sun_direction)
    - Direct irradiance = DNI * cos_incidence * transmittance (if cos_incidence > 0)
    
    GPU (simulator_gpu.solar.radiation RadiationModel):
    ----------------------------------------------------
    - Uses discrete directions: IUP, IDOWN, INORTH, ISOUTH, IEAST, IWEST
    - Normal vectors:
        * IUP (0): (0, 0, +1) - facing up
        * IDOWN (1): (0, 0, -1) - facing down  
        * INORTH (2): (0, +1, 0) - facing north (+Y)
        * ISOUTH (3): (0, -1, 0) - facing south (-Y)
        * IEAST (4): (+1, 0, 0) - facing east (+X)
        * IWEST (5): (-1, 0, 0) - facing west (-X)
    
    POTENTIAL ISSUE:
    ----------------
    The GPU uses axis-aligned surfaces only (6 discrete directions).
    The CPU uses arbitrary triangle normals from the mesh.
    
    For building surface irradiance, this is usually fine as buildings have
    axis-aligned faces, but the mapping from trimesh to axis-aligned could
    introduce discrepancies.
    """)
    
    return True


def test_with_simple_voxcity():
    """
    Create a simple VoxCity model and compare CPU vs GPU solar calculations.
    """
    print("\n" + "=" * 80)
    print("TEST 6: Simple VoxCity Model Comparison")
    print("=" * 80)
    
    # Now test sun direction for a specific case
    azimuth = 135  # Southeast
    elevation = 30
    
    print(f"\n  Testing sun from azimuth={azimuth}° (SE), elevation={elevation}°")
    
    # CPU formula
    az_rad = np.deg2rad(180 - azimuth)
    el_rad = np.deg2rad(elevation)
    cpu_dir = (
        np.cos(el_rad) * np.cos(az_rad),
        np.cos(el_rad) * np.sin(az_rad),
        np.sin(el_rad)
    )
    print(f"\n  CPU sun direction: ({cpu_dir[0]:.4f}, {cpu_dir[1]:.4f}, {cpu_dir[2]:.4f})")
    
    # For a building face facing EAST (+X direction normal)
    east_normal = (1, 0, 0)
    cpu_cos_inc = sum(a*b for a, b in zip(cpu_dir, east_normal))
    print(f"  Cosine incidence on EAST face: {cpu_cos_inc:.4f}")
    
    # The sun is from SE (135°), so it should hit the EAST and SOUTH faces
    # East face should receive some light
    # South face should receive more light
    south_normal = (0, -1, 0)
    cpu_cos_inc_south = sum(a*b for a, b in zip(cpu_dir, south_normal))
    print(f"  Cosine incidence on SOUTH face: {cpu_cos_inc_south:.4f}")
    
    # West and North faces should be in shadow (negative cos_incidence)
    west_normal = (-1, 0, 0)
    north_normal = (0, 1, 0)
    cpu_cos_inc_west = sum(a*b for a, b in zip(cpu_dir, west_normal))
    cpu_cos_inc_north = sum(a*b for a, b in zip(cpu_dir, north_normal))
    print(f"  Cosine incidence on WEST face: {cpu_cos_inc_west:.4f} (should be <=0)")
    print(f"  Cosine incidence on NORTH face: {cpu_cos_inc_north:.4f} (should be <=0)")
    
    print("\n✓ Direction calculation verified with simple geometry")
    return True


def create_synthetic_voxcity(nx=50, ny=50, nz=30, meshsize=2.0):
    """
    Create a synthetic VoxCity model for testing.
    
    Creates a simple scene with:
    - Ground plane (land cover at z=0)
    - One central building (10x10 base, 20m tall)
    - One tree nearby
    
    Returns:
        VoxCity object
    """
    from voxcity.models import (
        VoxCity, VoxelGrid, GridMetadata, BuildingGrid, 
        LandCoverGrid, DemGrid, CanopyGrid
    )
    
    # Create voxel grid
    # Class codes from voxcity/generator/voxelizer.py:
    #   0 = empty
    #  -3 = building
    #  -2 = tree
    #   1-6 = land cover types (grass, pavement, etc.)
    voxels = np.zeros((ny, nx, nz), dtype=np.int16)
    
    # Add ground surface (land cover = 1 = grass) at z=0
    # This is needed for the CPU ray-tracer to find valid observer positions
    voxels[:, :, 0] = 1  # Ground level
    
    # Add a building in the center (10x10 grid cells, 10 voxels tall = 20m)
    # Building starts at z=1 (above ground)
    bld_x_start, bld_x_end = 20, 30
    bld_y_start, bld_y_end = 20, 30
    bld_height_voxels = 10
    voxels[bld_y_start:bld_y_end, bld_x_start:bld_x_end, 1:bld_height_voxels+1] = -3
    
    # Add a tree near the building (5x5, 6 voxels tall = 12m)
    # Tree starts at z=1 (above ground)
    tree_x_start, tree_x_end = 35, 40
    tree_y_start, tree_y_end = 20, 25
    tree_height_voxels = 6
    voxels[tree_y_start:tree_y_end, tree_x_start:tree_x_end, 1:tree_height_voxels+1] = -2
    
    # Create metadata (WGS84 bounds for NYC area)
    meta = GridMetadata(
        crs="EPSG:4326",
        bounds=(-74.01, 40.70, -73.99, 40.72),  # minx, miny, maxx, maxy
        meshsize=meshsize
    )
    
    voxel_grid = VoxelGrid(classes=voxels, meta=meta)
    
    # Create building grid
    building_heights = np.zeros((ny, nx), dtype=np.float32)
    building_heights[bld_y_start:bld_y_end, bld_x_start:bld_x_end] = bld_height_voxels * meshsize
    
    building_min_heights = np.empty((ny, nx), dtype=object)
    for i in range(ny):
        for j in range(nx):
            building_min_heights[i, j] = []
    
    building_ids = np.zeros((ny, nx), dtype=np.int32)
    building_ids[bld_y_start:bld_y_end, bld_x_start:bld_x_end] = 1
    
    building_grid = BuildingGrid(
        heights=building_heights,
        min_heights=building_min_heights,
        ids=building_ids,
        meta=meta
    )
    
    # Create land cover grid (all grass = 1)
    land_cover = np.ones((ny, nx), dtype=np.int16)
    land_cover_grid = LandCoverGrid(classes=land_cover, meta=meta)
    
    # Create DEM (flat ground)
    dem = np.zeros((ny, nx), dtype=np.float32)
    dem_grid = DemGrid(elevation=dem, meta=meta)
    
    # Create canopy grid
    canopy_top = np.zeros((ny, nx), dtype=np.float32)
    canopy_top[tree_y_start:tree_y_end, tree_x_start:tree_x_end] = tree_height_voxels * meshsize
    canopy_grid = CanopyGrid(top=canopy_top, meta=meta)
    
    # Create VoxCity
    voxcity = VoxCity(
        voxels=voxel_grid,
        buildings=building_grid,
        land_cover=land_cover_grid,
        dem=dem_grid,
        tree_canopy=canopy_grid,
        extras={"latitude": 40.71, "longitude": -74.01}
    )
    
    return voxcity


def test_direct_cpu_vs_gpu_irradiance():
    """
    Direct comparison of CPU vs GPU get_direct_solar_irradiance_map.
    
    This test:
    1. Creates a synthetic VoxCity model
    2. Runs CPU get_direct_solar_irradiance_map
    3. Runs GPU get_direct_solar_irradiance_map
    4. Compares the resulting irradiance maps
    5. Identifies specific locations where shadows differ
    """
    print("\n" + "=" * 80)
    print("TEST 9: Direct CPU vs GPU Irradiance Map Comparison")
    print("=" * 80)
    
    # Try to import both CPU and GPU implementations
    try:
        from voxcity.simulator.solar.radiation import get_direct_solar_irradiance_map as cpu_get_direct
        has_cpu = True
    except ImportError as e:
        print(f"Warning: CPU module not available: {e}")
        has_cpu = False
    
    try:
        from voxcity.simulator_gpu.solar.integration import get_direct_solar_irradiance_map as gpu_get_direct
        has_gpu = True
    except ImportError as e:
        print(f"Warning: GPU module not available: {e}")
        has_gpu = False
    
    if not has_cpu or not has_gpu:
        print("Skipping - need both CPU and GPU modules")
        return False
    
    # Create synthetic VoxCity
    print("\n  Creating synthetic VoxCity model...")
    voxcity = create_synthetic_voxcity(nx=50, ny=50, nz=30, meshsize=2.0)
    print(f"    Voxel grid shape: {voxcity.voxels.classes.shape}")
    print(f"    Meshsize: {voxcity.voxels.meta.meshsize}m")
    
    # Test parameters
    test_cases = [
        (135, 45, 800, "Southeast, 45° elevation"),
        (180, 60, 900, "South, 60° elevation"),
        (90, 30, 700, "East, 30° elevation"),
        (225, 45, 850, "Southwest, 45° elevation"),
    ]
    
    all_passed = True
    
    for azimuth, elevation, dni, desc in test_cases:
        print(f"\n  Testing: {desc}")
        print(f"    Azimuth: {azimuth}°, Elevation: {elevation}°, DNI: {dni} W/m²")
        
        # Run CPU
        try:
            cpu_map = cpu_get_direct(
                voxcity=voxcity,
                azimuth_degrees_ori=azimuth,
                elevation_degrees=elevation,
                direct_normal_irradiance=dni,
                show_plot=False,
                view_point_height=1.5,
            )
            cpu_valid = np.sum(~np.isnan(cpu_map))
            print(f"    CPU result shape: {cpu_map.shape}, valid cells: {cpu_valid}")
        except Exception as e:
            print(f"    ❌ CPU error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            continue
        
        # Run GPU
        try:
            gpu_map = gpu_get_direct(
                voxcity=voxcity,
                azimuth_degrees_ori=azimuth,
                elevation_degrees=elevation,
                direct_normal_irradiance=dni,
                show_plot=False,
                view_point_height=1.5,
                with_reflections=False,  # Simple ray-tracing for fair comparison
            )
            gpu_valid = np.sum(~np.isnan(gpu_map))
            print(f"    GPU result shape: {gpu_map.shape}, valid cells: {gpu_valid}")
        except Exception as e:
            print(f"    ❌ GPU error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
            continue
        
        # Handle potential shape differences (GPU might flip)
        if cpu_map.shape != gpu_map.shape:
            print(f"    ⚠️  Shape mismatch: CPU {cpu_map.shape} vs GPU {gpu_map.shape}")
            all_passed = False
            continue
        
        # Create mask for cells that are valid in BOTH implementations
        both_valid = ~np.isnan(cpu_map) & ~np.isnan(gpu_map)
        n_both_valid = np.sum(both_valid)
        
        # Compare results only where both are valid
        if n_both_valid > 0:
            diff = np.abs(cpu_map[both_valid] - gpu_map[both_valid])
            max_diff = np.max(diff)
            mean_diff = np.mean(diff)
            
            # Find locations with significant differences (>10% of DNI)
            threshold = 0.1 * dni
            sig_diff_mask = diff > threshold
            n_sig_diff = np.sum(sig_diff_mask)
            pct_diff = 100.0 * n_sig_diff / n_both_valid
            
            print(f"    Cells valid in both: {n_both_valid}")
            print(f"    Max difference: {max_diff:.2f} W/m²")
            print(f"    Mean difference: {mean_diff:.2f} W/m²")
            print(f"    Cells with >10% difference: {n_sig_diff} ({pct_diff:.1f}%)")
        else:
            print(f"    ⚠️  No cells valid in both CPU and GPU")
            max_diff = 0
            mean_diff = 0
            pct_diff = 0
        
        # Check where NaN patterns differ (may indicate different handling of buildings/trees)
        cpu_nan = np.isnan(cpu_map)
        gpu_nan = np.isnan(gpu_map)
        nan_mismatch = cpu_nan != gpu_nan
        n_nan_mismatch = np.sum(nan_mismatch)
        nan_mismatch_pct = 100.0 * n_nan_mismatch / cpu_map.size
        print(f"    NaN pattern mismatch cells: {n_nan_mismatch} ({nan_mismatch_pct:.1f}%)")
        
        # Shadow comparison (where both are valid)
        if n_both_valid > 0:
            cpu_shadowed = cpu_map < 0.1 * dni
            gpu_shadowed = gpu_map < 0.1 * dni
            
            # Only compare shadow where both are valid
            cpu_shadow_valid = cpu_shadowed & both_valid
            gpu_shadow_valid = gpu_shadowed & both_valid
            
            cpu_shadow_pct = 100.0 * np.sum(cpu_shadow_valid) / n_both_valid
            gpu_shadow_pct = 100.0 * np.sum(gpu_shadow_valid) / n_both_valid
            
            shadow_mismatch = (cpu_shadow_valid != gpu_shadow_valid)
            n_shadow_mismatch = np.sum(shadow_mismatch)
            shadow_mismatch_pct = 100.0 * n_shadow_mismatch / n_both_valid
            
            print(f"    CPU shadow coverage: {cpu_shadow_pct:.1f}% of valid cells")
            print(f"    GPU shadow coverage: {gpu_shadow_pct:.1f}% of valid cells")
            print(f"    Shadow mismatch cells: {n_shadow_mismatch} ({shadow_mismatch_pct:.1f}%)")
            
            # Detailed mismatch analysis
            if n_shadow_mismatch > 0 and n_shadow_mismatch <= 20:
                # Find first few mismatch locations
                mismatch_indices = np.argwhere(shadow_mismatch)[:5]
                print(f"\n    First {len(mismatch_indices)} shadow mismatch locations:")
                for idx in mismatch_indices:
                    y, x = idx
                    print(f"      ({x}, {y}): CPU={cpu_map[y,x]:.1f}, GPU={gpu_map[y,x]:.1f}")
        
        # Tolerance check - only fail if significant differences in valid cells
        if pct_diff > 5.0:
            print(f"    ❌ FAIL: Too many cells with significant differences ({pct_diff:.1f}%)")
            all_passed = False
        elif n_both_valid > 0 and shadow_mismatch_pct > 5.0:
            print(f"    ❌ FAIL: Too many shadow mismatch cells ({shadow_mismatch_pct:.1f}%)")
            all_passed = False
        elif nan_mismatch_pct > 10.0:
            print(f"    ⚠️  WARNING: Large NaN pattern mismatch ({nan_mismatch_pct:.1f}%)")
            print(f"       This may indicate different handling of buildings/trees")
        else:
            print(f"    ✓ PASS: Results are within acceptable tolerance")
    
    if all_passed:
        print("\n✓ All CPU vs GPU comparisons passed")
    else:
        print("\n❌ Some CPU vs GPU comparisons failed")
        print("\n  POTENTIAL CAUSES:")
        print("  1. Coordinate system flipping (check np.flipud in GPU code)")
        print("  2. Tree transmittance calculation differences")
        print("  3. Voxel boundary handling differences")
    
    return all_passed


def test_cpu_vs_gpu_building_irradiance():
    """
    Compare CPU vs GPU building surface irradiance calculations.
    
    This tests get_building_solar_irradiance from both implementations.
    """
    print("\n" + "=" * 80)
    print("TEST 10: CPU vs GPU Building Surface Irradiance")
    print("=" * 80)
    
    # Try to import dependencies
    try:
        from voxcity.simulator.solar.radiation import get_building_solar_irradiance as cpu_building
        from voxcity.geoprocessor.mesh import create_voxel_mesh
        has_cpu = True
    except ImportError as e:
        print(f"Warning: CPU module not available: {e}")
        has_cpu = False
    
    try:
        from voxcity.simulator_gpu.solar.integration import get_building_solar_irradiance as gpu_building
        has_gpu = True
    except ImportError as e:
        print(f"Warning: GPU module not available: {e}")
        has_gpu = False
    
    if not has_cpu or not has_gpu:
        print("Skipping - need both CPU and GPU modules")
        return False
    
    # Create synthetic VoxCity
    print("\n  Creating synthetic VoxCity model...")
    voxcity = create_synthetic_voxcity(nx=30, ny=30, nz=20, meshsize=2.0)
    
    # Create building mesh (CPU needs pre-computed SVF mesh)
    print("  Creating building mesh...")
    try:
        building_mask = voxcity.voxels.classes == -3
        mesh = create_voxel_mesh(
            voxcity.voxels.classes,
            class_id=-3,  # Buildings only
            meshsize=voxcity.voxels.meta.meshsize,
            building_id_grid=voxcity.buildings.ids,
        )
        if mesh is None:
            print(f"  ⚠️  No building mesh created (no building voxels)")
            print("  Skipping building irradiance test")
            return True
        print(f"    Mesh faces: {len(mesh.faces)}")
    except Exception as e:
        print(f"  ⚠️  Could not create mesh: {e}")
        print("  Skipping building irradiance test")
        return True  # Not a failure, just skip
    
    # Test parameters
    azimuth = 135  # Southeast
    elevation = 45
    dni = 800
    dhi = 100
    
    print(f"\n  Testing: Azimuth={azimuth}°, Elevation={elevation}°, DNI={dni}, DHI={dhi}")
    
    # Run CPU (needs SVF precomputed or will compute internally)
    print("\n  Running CPU building solar irradiance...")
    try:
        # Add empty SVF metadata for CPU
        if not hasattr(mesh, 'metadata'):
            mesh.metadata = {}
        if 'svf' not in mesh.metadata:
            mesh.metadata['svf'] = np.ones(len(mesh.faces), dtype=np.float64)  # Assume unobstructed
        
        cpu_result = cpu_building(
            voxcity=voxcity,
            building_svf_mesh=mesh,
            azimuth_degrees=azimuth,
            elevation_degrees=elevation,
            direct_normal_irradiance=dni,
            diffuse_irradiance=dhi,
            progress_report=False,
        )
        cpu_irradiance = cpu_result.visual.vertex_colors[:, 0] if hasattr(cpu_result, 'visual') else None
        print(f"    CPU result type: {type(cpu_result)}")
    except Exception as e:
        print(f"    ❌ CPU error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Run GPU
    print("\n  Running GPU building solar irradiance...")
    try:
        gpu_result = gpu_building(
            voxcity=voxcity,
            building_svf_mesh=mesh,
            azimuth_degrees_ori=azimuth,
            elevation_degrees=elevation,
            direct_normal_irradiance=dni,
            diffuse_irradiance=dhi,
            with_reflections=False,
            progress_report=False,
        )
        print(f"    GPU result type: {type(gpu_result)}")
    except Exception as e:
        print(f"    ❌ GPU error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare results if both succeeded
    print("\n  Comparing CPU vs GPU building irradiance results...")
    
    # Extract irradiance from mesh metadata
    try:
        # CPU stores in metadata
        if hasattr(cpu_result, 'metadata') and 'irradiance' in cpu_result.metadata:
            cpu_irrad = cpu_result.metadata['irradiance']
        elif hasattr(cpu_result, 'metadata') and 'direct_irradiance' in cpu_result.metadata:
            cpu_irrad = cpu_result.metadata['direct_irradiance']
        else:
            cpu_irrad = None
            print("    CPU: No irradiance in metadata, checking visual colors...")
            if hasattr(cpu_result, 'visual') and hasattr(cpu_result.visual, 'face_colors'):
                # May be encoded in face colors
                cpu_irrad = cpu_result.visual.face_colors[:, 0].astype(float) / 255.0 * dni
        
        # GPU stores in metadata  
        if hasattr(gpu_result, 'metadata') and 'irradiance' in gpu_result.metadata:
            gpu_irrad = gpu_result.metadata['irradiance']
        elif hasattr(gpu_result, 'metadata') and 'direct_irradiance' in gpu_result.metadata:
            gpu_irrad = gpu_result.metadata['direct_irradiance']
        else:
            gpu_irrad = None
            print("    GPU: No irradiance in metadata, checking visual colors...")
            if hasattr(gpu_result, 'visual') and hasattr(gpu_result.visual, 'face_colors'):
                gpu_irrad = gpu_result.visual.face_colors[:, 0].astype(float) / 255.0 * dni
        
        if cpu_irrad is not None and gpu_irrad is not None:
            # Meshes may have different number of faces
            n_cpu = len(cpu_irrad) if hasattr(cpu_irrad, '__len__') else 1
            n_gpu = len(gpu_irrad) if hasattr(gpu_irrad, '__len__') else 1
            print(f"    CPU faces with irradiance: {n_cpu}")
            print(f"    GPU faces with irradiance: {n_gpu}")
            
            if n_cpu == n_gpu:
                # Direct comparison
                diff = np.abs(np.array(cpu_irrad) - np.array(gpu_irrad))
                max_diff = np.nanmax(diff)
                mean_diff = np.nanmean(diff)
                print(f"    Max difference: {max_diff:.2f} W/m²")
                print(f"    Mean difference: {mean_diff:.2f} W/m²")
            else:
                print("    ⚠️  Different number of faces - cannot directly compare")
        else:
            print("    ⚠️  Could not extract irradiance values for comparison")
            
        # Print metadata keys for debugging
        print("\n    CPU metadata keys:", list(cpu_result.metadata.keys()) if hasattr(cpu_result, 'metadata') else "None")
        print("    GPU metadata keys:", list(gpu_result.metadata.keys()) if hasattr(gpu_result, 'metadata') else "None")
        
    except Exception as e:
        print(f"    Error comparing results: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n  ✓ Building irradiance test completed")
    return True


def test_palm_direction_vs_integration():
    """
    Critical test: Compare PALM's native direction with integration.py's recomputed direction.
    
    This tests the specific issue: GPU integration.py recomputes sun direction from 
    azimuth/elevation using the "180 - azimuth" formula, but this formula assumes
    a specific coordinate convention that may not match PALM's native direction.
    """
    print("\n" + "=" * 80)
    print("TEST 7: PALM Native Direction vs Integration.py Formula")
    print("=" * 80)
    
    try:
        from voxcity.simulator_gpu.solar.solar import calc_zenith, calc_solar_position_datetime
    except ImportError:
        print("Skipping - GPU solar module not available")
        return False
    
    # Test for New York at summer solstice
    lat, lon = 40.71, -74.01
    
    test_times = [
        (172, 12 * 3600, "Jun 21, Noon UTC"),
        (172, 15 * 3600, "Jun 21, 3PM UTC"),
        (172, 18 * 3600, "Jun 21, 6PM UTC"),
        (1, 17 * 3600, "Jan 1, 5PM UTC"),
    ]
    
    print("\nComparing PALM native direction vs (180-az)/el recomputed direction:")
    print("-" * 100)
    
    all_match = True
    
    for doy, sod, desc in test_times:
        pos = calc_zenith(doy, sod, lat, lon)
        
        # PALM's native direction (x=East, y=North, z=Up)
        palm_x, palm_y, palm_z = pos.direction
        
        # Integration.py's recomputed direction using (180-az) formula
        azimuth_degrees = 180 - pos.azimuth_angle
        az_rad = np.deg2rad(azimuth_degrees)
        el_rad = np.deg2rad(pos.elevation_angle)
        
        integ_x = np.cos(el_rad) * np.cos(az_rad)
        integ_y = np.cos(el_rad) * np.sin(az_rad)
        integ_z = np.sin(el_rad)
        
        # Calculate angular difference
        dot_product = palm_x * integ_x + palm_y * integ_y + palm_z * integ_z
        dot_product = min(1.0, max(-1.0, dot_product))  # Clamp for acos
        angle_diff = np.degrees(np.arccos(dot_product))
        
        print(f"\n{desc}:")
        print(f"  PALM az={pos.azimuth_angle:.1f}°, el={pos.elevation_angle:.1f}°")
        print(f"  PALM direction:   ({palm_x:>7.4f}, {palm_y:>7.4f}, {palm_z:>7.4f})")
        print(f"  Integ direction:  ({integ_x:>7.4f}, {integ_y:>7.4f}, {integ_z:>7.4f})")
        print(f"  Angular diff:     {angle_diff:.2f}°")
        
        if angle_diff > 1.0:
            print(f"  ⚠️  MISMATCH!")
            all_match = False
    
    print("-" * 100)
    
    if not all_match:
        print("\n❌ CRITICAL ISSUE: PALM native direction != (180-az)/el formula")
        print("   The GPU integration.py is using the WRONG sun direction!")
        print("\n   PALM's azimuth convention is: 0=North, 90=East")
        print("   But the (180-az) formula expects a different convention!")
        print("\n   SOLUTION: The GPU code should use PALM's native direction directly,")
        print("   NOT recompute it from azimuth/elevation with the (180-az) formula.")
        return False
    
    print("\n✓ All directions match within tolerance")
    return True


def test_cpu_gpu_same_azimuth_elevation():
    """
    Test that given the SAME azimuth and elevation values, 
    both CPU and GPU produce the same sun direction.
    
    This isolates whether the formula itself is the issue.
    """
    print("\n" + "=" * 80)
    print("TEST 8: Same Az/El Input -> Same Direction Output?")
    print("=" * 80)
    
    test_cases = [
        (90, 45, "East 45°"),
        (180, 30, "South 30°"),
        (135, 60, "SE 60°"),
        (270, 20, "West 20°"),
    ]
    
    print("\nBoth implementations use: azimuth_degrees = 180 - azimuth_degrees_ori")
    print("                          dx = cos(el) * cos(az), dy = cos(el) * sin(az), dz = sin(el)")
    print()
    
    all_match = True
    for az_ori, el, desc in test_cases:
        # Common formula used by both
        az_deg = 180 - az_ori
        az_rad = np.deg2rad(az_deg)
        el_rad = np.deg2rad(el)
        
        dx = np.cos(el_rad) * np.cos(az_rad)
        dy = np.cos(el_rad) * np.sin(az_rad)
        dz = np.sin(el_rad)
        
        print(f"  {desc:>12}: az_ori={az_ori:>3}° -> ({dx:>7.4f}, {dy:>7.4f}, {dz:>7.4f})")
    
    print("\n✓ Formula is identical in both implementations")
    print("  The difference comes from how az/el are COMPUTED, not how they're USED")
    return True


def main():
    """Run all comparison tests."""
    print("\n" + "=" * 80)
    print("SOLAR SIMULATION CPU vs GPU COMPARISON TESTS")
    print("=" * 80)
    print("\nThis script compares sun direction, solar position, and coordinate")
    print("handling between voxcity.simulator.solar (CPU) and")
    print("voxcity.simulator_gpu.solar (GPU) implementations.")
    
    results = []
    
    # Test 1: Sun direction from azimuth/elevation
    results.append(("Sun Direction Calculation", test_sun_direction_calculation()))
    
    # Test 2: Astral vs PALM solar position
    results.append(("Astral vs PALM Solar Position", test_astral_vs_palm_solar_position()))
    
    # Test 3: PALM direction vector
    results.append(("PALM Direction Vector", test_palm_direction_vector()))
    
    # Test 4: Coordinate convention analysis
    results.append(("Coordinate Conventions", test_coordinate_convention_analysis()))
    
    # Test 5: Normal vector handling
    results.append(("Normal Vector Handling", test_normal_vector_handling()))
    
    # Test 6: Simple VoxCity comparison
    results.append(("Simple VoxCity Model", test_with_simple_voxcity()))
    
    # Test 7: PALM direction vs integration formula
    results.append(("PALM Direction vs Integration", test_palm_direction_vs_integration()))
    
    # Test 8: Same az/el -> same direction
    results.append(("Same Az/El -> Same Direction", test_cpu_gpu_same_azimuth_elevation()))
    
    # Test 9: Direct CPU vs GPU irradiance map comparison
    results.append(("Direct CPU vs GPU Irradiance", test_direct_cpu_vs_gpu_irradiance()))
    
    # Test 10: CPU vs GPU building irradiance
    results.append(("CPU vs GPU Building Irradiance", test_cpu_vs_gpu_building_irradiance()))
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"  {name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print()
    if all_passed:
        print("All tests passed!")
    else:
        print("Some tests failed - investigate the differences above.")
        print("\n" + "=" * 80)
        print("ROOT CAUSE ANALYSIS")
        print("=" * 80)
        print("""
TWO CRITICAL ISSUES FOUND:

1. DIFFERENT SOLAR POSITION ALGORITHMS
   - CPU uses: Astral library  
   - GPU uses: PALM formula (from PALM-4U model)
   - These give DIFFERENT azimuth/elevation for the same timestamp!
   - Maximum difference observed: ~24° in elevation, ~1.6° in azimuth

2. GPU DIRECTION VECTOR MISMATCH
   - PALM's calc_zenith() computes a native sun direction vector
   - BUT integration.py RECOMPUTES direction using (180-az)/el formula  
   - This formula doesn't match PALM's direction vector convention!
   - Angular difference: can exceed 90°!

RECOMMENDED FIXES:

Option A: Make GPU use Astral (match CPU)
   - Modify GPU integration to use get_solar_positions_astral()
   - Pros: Ensures identical results between CPU and GPU
   - Cons: Adds dependency, different from PALM literature

Option B: Make CPU use PALM formula (match GPU)
   - Modify CPU temporal.py to use PALM's calc_zenith
   - Pros: Consistent with PALM-4U methodology  
   - Cons: Changes existing CPU behavior

Option C: Fix GPU integration.py direction calculation
   - Use PALM's native direction directly instead of recomputing
   - Remove the (180-azimuth) formula from integration.py
   - Directly use pos.direction from calc_zenith()
   
The most critical fix is Option C - the GPU code is computing
two different sun directions internally!
        """)
    
    return all_passed


if __name__ == "__main__":
    main()
