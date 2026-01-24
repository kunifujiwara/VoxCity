"""
Debug test script for west vertical surface shadowing issue.

Issue:
- West vertical surfaces show shadows even when direct radiation comes from west
- Some west vertical surfaces near the west edge show higher irradiance
- East-facing surfaces show reasonable results
- Horizontal surfaces are fine

VoxCity coordinate system:
- x (index i): increases from North to South
- y (index j): increases from West to East
- z (index k): increases upward

Surface direction indices (from raytracing.py):
- 0=IUP (+z normal): upward-facing
- 1=IDOWN (-z normal): downward-facing
- 2=INORTH (+y normal): East-facing (receives sun when sun_y > 0)
- 3=ISOUTH (-y normal): West-facing (receives sun when sun_y < 0)
- 4=IEAST (+x normal): South-facing (receives sun when sun_x > 0)
- 5=IWEST (-x normal): North-facing (receives sun when sun_x < 0)
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from voxcity.simulator_gpu.solar import reset_solar_taichi_cache


def test_sun_direction_computation():
    """Test the sun direction computation logic."""
    from voxcity.simulator_gpu.solar.integration import _compute_sun_direction
    
    print("=" * 70)
    print("TEST 1: Sun Direction Computation")
    print("=" * 70)
    print()
    print("VoxCity grid: x=South(+), y=East(+), z=Up(+)")
    print()
    
    # Test each cardinal direction at 45 degree elevation
    test_cases = [
        (0, "North", "sun_x < 0, sun_y = 0"),
        (90, "East", "sun_x = 0, sun_y > 0"),
        (180, "South", "sun_x > 0, sun_y = 0"),
        (270, "West", "sun_x = 0, sun_y < 0"),
        (225, "Southwest", "sun_x > 0, sun_y < 0"),
        (315, "Northwest", "sun_x < 0, sun_y < 0"),
    ]
    
    print(f"{'Azimuth':>10} | {'sun_x':>8} {'sun_y':>8} {'sun_z':>8} | Expected")
    print("-" * 65)
    
    all_passed = True
    for az, name, expected in test_cases:
        sx, sy, sz, _ = _compute_sun_direction(az, 45)
        status = ""
        
        # Check if result matches expectation
        if name == "North":
            if sx >= 0 or abs(sy) > 0.1:
                status = "FAIL"
                all_passed = False
        elif name == "East":
            if sy <= 0 or abs(sx) > 0.1:
                status = "FAIL"
                all_passed = False
        elif name == "South":
            if sx <= 0 or abs(sy) > 0.1:
                status = "FAIL"
                all_passed = False
        elif name == "West":
            if sy >= 0 or abs(sx) > 0.1:
                status = "FAIL"
                all_passed = False
        elif name == "Southwest":
            if sx <= 0 or sy >= 0:
                status = "FAIL"
                all_passed = False
        elif name == "Northwest":
            if sx >= 0 or sy >= 0:
                status = "FAIL"
                all_passed = False
        
        print(f"{name:>10} | {sx:>8.3f} {sy:>8.3f} {sz:>8.3f} | {expected} {status}")
    
    print()
    if all_passed:
        print("All sun direction tests PASSED")
    else:
        print("Some sun direction tests FAILED - check coordinate transformation!")
    
    return all_passed


def test_surface_normal_check():
    """Test the surface normal vs sun direction check in raytracing."""
    from voxcity.simulator_gpu.solar.integration import _compute_sun_direction
    
    print()
    print("=" * 70)
    print("TEST 2: Surface Normal vs Sun Direction Check")
    print("=" * 70)
    print()
    
    # The check in raytracing.py:
    # direction 2 (INORTH, +y normal): face_sun = 1 if sun_y > 0 else 0
    # direction 3 (ISOUTH, -y normal): face_sun = 1 if sun_y < 0 else 0
    # direction 4 (IEAST, +x normal): face_sun = 1 if sun_x > 0 else 0
    # direction 5 (IWEST, -x normal): face_sun = 1 if sun_x < 0 else 0
    
    surface_info = [
        (2, "+y", "East-facing", "sun_y > 0", lambda sx, sy, sz: sy > 0),
        (3, "-y", "West-facing", "sun_y < 0", lambda sx, sy, sz: sy < 0),
        (4, "+x", "South-facing", "sun_x > 0", lambda sx, sy, sz: sx > 0),
        (5, "-x", "North-facing", "sun_x < 0", lambda sx, sy, sz: sx < 0),
    ]
    
    azimuths = [
        (90, "East"),
        (270, "West"),
        (180, "South"),
        (0, "North"),
    ]
    
    print("Testing which surfaces receive direct sun for each sun azimuth:")
    print()
    
    all_passed = True
    for az, sun_pos in azimuths:
        sx, sy, sz, _ = _compute_sun_direction(az, 45)
        print(f"Sun from {sun_pos} (azimuth={az}): sun_dir = ({sx:.3f}, {sy:.3f}, {sz:.3f})")
        
        for dir_idx, normal, geo_name, condition, check_func in surface_info:
            receives_sun = check_func(sx, sy, sz)
            status = "receives sun" if receives_sun else "in shadow"
            
            # Verify correctness:
            # - Sun from East (az=90) should illuminate East-facing (+y) surfaces
            # - Sun from West (az=270) should illuminate West-facing (-y) surfaces
            expected_lit = False
            if sun_pos == "East" and geo_name == "East-facing":
                expected_lit = True
            elif sun_pos == "West" and geo_name == "West-facing":
                expected_lit = True
            elif sun_pos == "South" and geo_name == "South-facing":
                expected_lit = True
            elif sun_pos == "North" and geo_name == "North-facing":
                expected_lit = True
            
            check = ""
            if receives_sun != expected_lit:
                check = " <-- ERROR!"
                all_passed = False
            
            print(f"  Dir {dir_idx} ({geo_name}, {normal} normal): {status}{check}")
        print()
    
    if all_passed:
        print("All surface normal checks PASSED")
    else:
        print("Some surface normal checks FAILED!")
    
    return all_passed


def test_with_voxcity_instantaneous():
    """Test with actual VoxCity data at specific times."""
    from voxcity.generator.io import load_voxcity
    from voxcity.simulator_gpu.solar import get_building_global_solar_irradiance_using_epw
    
    print()
    print("=" * 70)
    print("TEST 3: VoxCity Instantaneous Test - West vs East facing surfaces")
    print("=" * 70)
    print()
    
    # Reset cache to ensure clean state
    reset_solar_taichi_cache()
    
    # Load voxcity
    input_path = "demo/output/voxcity_mini2.pkl"
    print(f"Loading voxcity from {input_path}...")
    try:
        voxcity = load_voxcity(input_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {input_path}")
        print("Please ensure the file exists.")
        return False
    
    print(f"Voxcity loaded successfully")
    print()
    
    # Test afternoon (sun in West - should illuminate West-facing surfaces)
    # June 1st at 3:00 PM - sun should be in the West
    print("Test at 15:00 (3 PM) - Sun should be in the West")
    print("-" * 50)
    
    result_pm = get_building_global_solar_irradiance_using_epw(
        voxcity,
        calc_type="instantaneous",
        epw_file_path="demo/output/phoenix-sky.harbor.intl.ap_az_usa.epw",
        calc_time="06-01 15:00:00",
        with_reflections=False,
        progress_report=False,
    )
    
    # Get face normals and irradiance values
    normals = result_pm.face_normals
    direct = result_pm.metadata.get("direct", np.zeros(len(normals)))
    diffuse = result_pm.metadata.get("diffuse", np.zeros(len(normals)))
    global_irr = result_pm.metadata.get("global", np.zeros(len(normals)))
    
    # Also get the sun position from metadata if available
    sun_azimuth = result_pm.metadata.get("sun_azimuth", None)
    sun_elevation = result_pm.metadata.get("sun_elevation", None)
    sun_direction = result_pm.metadata.get("sun_direction", None)
    
    if sun_azimuth is not None:
        print(f"Sun azimuth: {sun_azimuth:.1f} degrees")
    if sun_elevation is not None:
        print(f"Sun elevation: {sun_elevation:.1f} degrees")
    if sun_direction is not None:
        print(f"Sun direction vector: ({sun_direction[0]:.3f}, {sun_direction[1]:.3f}, {sun_direction[2]:.3f})")
    print()
    
    # Analyze by surface direction
    # Find surfaces by normal direction
    def analyze_direction(normals, direct, name, nx, ny, nz):
        """Analyze surfaces facing a particular direction."""
        mask = (np.abs(normals[:, 0] - nx) < 0.1) & \
               (np.abs(normals[:, 1] - ny) < 0.1) & \
               (np.abs(normals[:, 2] - nz) < 0.1)
        count = np.sum(mask)
        if count > 0:
            mean_direct = np.nanmean(direct[mask])
            max_direct = np.nanmax(direct[mask])
            min_direct = np.nanmin(direct[mask])
            nonzero = np.sum(direct[mask] > 0)
            print(f"  {name}: {count:3d} faces, direct mean={mean_direct:.1f}, "
                  f"min={min_direct:.1f}, max={max_direct:.1f} W/m2, {nonzero} with direct>0")
            return mean_direct
        else:
            print(f"  {name}: 0 faces")
            return 0
    
    print("Direct irradiance by surface orientation:")
    
    # +y normal = East-facing surfaces (INORTH direction=2)
    east_facing = analyze_direction(normals, direct, "East-facing (+y)", 0, 1, 0)
    
    # -y normal = West-facing surfaces (ISOUTH direction=3)
    west_facing = analyze_direction(normals, direct, "West-facing (-y)", 0, -1, 0)
    
    # +x normal = South-facing surfaces (IEAST direction=4)
    south_facing = analyze_direction(normals, direct, "South-facing (+x)", 1, 0, 0)
    
    # -x normal = North-facing surfaces (IWEST direction=5)
    north_facing = analyze_direction(normals, direct, "North-facing (-x)", -1, 0, 0)
    
    # +z normal = Upward-facing surfaces (IUP direction=0)
    up_facing = analyze_direction(normals, direct, "Upward (+z)", 0, 0, 1)
    
    print()
    
    # At 3 PM, sun is in the west, so:
    # - West-facing surfaces should receive significant direct radiation
    # - East-facing surfaces should receive little/no direct radiation
    if west_facing < east_facing:
        print("WARNING: West-facing surfaces have LESS direct radiation than East-facing!")
        print("         This is incorrect for afternoon sun (sun in West).")
        print()
    
    # Now test morning (sun in East)
    print()
    print("Test at 09:00 (9 AM) - Sun should be in the East")
    print("-" * 50)
    
    reset_solar_taichi_cache()
    
    result_am = get_building_global_solar_irradiance_using_epw(
        voxcity,
        calc_type="instantaneous",
        epw_file_path="demo/output/phoenix-sky.harbor.intl.ap_az_usa.epw",
        calc_time="06-01 09:00:00",
        with_reflections=False,
        progress_report=False,
    )
    
    normals_am = result_am.face_normals
    direct_am = result_am.metadata.get("direct", np.zeros(len(normals_am)))
    
    sun_azimuth_am = result_am.metadata.get("sun_azimuth", None)
    sun_direction_am = result_am.metadata.get("sun_direction", None)
    
    if sun_azimuth_am is not None:
        print(f"Sun azimuth: {sun_azimuth_am:.1f} degrees")
    if sun_direction_am is not None:
        print(f"Sun direction vector: ({sun_direction_am[0]:.3f}, {sun_direction_am[1]:.3f}, {sun_direction_am[2]:.3f})")
    print()
    
    print("Direct irradiance by surface orientation:")
    east_facing_am = analyze_direction(normals_am, direct_am, "East-facing (+y)", 0, 1, 0)
    west_facing_am = analyze_direction(normals_am, direct_am, "West-facing (-y)", 0, -1, 0)
    south_facing_am = analyze_direction(normals_am, direct_am, "South-facing (+x)", 1, 0, 0)
    north_facing_am = analyze_direction(normals_am, direct_am, "North-facing (-x)", -1, 0, 0)
    up_facing_am = analyze_direction(normals_am, direct_am, "Upward (+z)", 0, 0, 1)
    
    print()
    
    # At 9 AM, sun is in the east, so:
    # - East-facing surfaces should receive significant direct radiation
    # - West-facing surfaces should receive little/no direct radiation
    if east_facing_am < west_facing_am:
        print("WARNING: East-facing surfaces have LESS direct radiation than West-facing!")
        print("         This is incorrect for morning sun (sun in East).")
    
    print()
    print("SUMMARY:")
    print("-" * 50)
    print(f"At 9 AM (sun in East): East-facing={east_facing_am:.1f}, West-facing={west_facing_am:.1f}")
    print(f"At 3 PM (sun in West): East-facing={east_facing:.1f}, West-facing={west_facing:.1f}")
    print()
    
    # Final assessment
    passed = True
    if west_facing < east_facing:
        print("ISSUE DETECTED: At 3 PM, West-facing surfaces should have MORE direct irradiance")
        passed = False
    if east_facing_am < west_facing_am:
        print("ISSUE DETECTED: At 9 AM, East-facing surfaces should have MORE direct irradiance")
        passed = False
    
    if passed:
        print("Both morning and afternoon tests show expected behavior.")
    
    return passed


if __name__ == "__main__":
    print("West Surface Shadowing Debug Test")
    print("=" * 70)
    print()
    
    # Test 1: Sun direction computation
    test1_passed = test_sun_direction_computation()
    
    # Test 2: Surface normal check logic
    test2_passed = test_surface_normal_check()
    
    # Test 3: Actual VoxCity test
    test3_passed = test_with_voxcity_instantaneous()
    
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Test 1 (Sun direction computation): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Surface normal check): {'PASSED' if test2_passed else 'FAILED'}")
    print(f"Test 3 (VoxCity instantaneous): {'PASSED' if test3_passed else 'FAILED'}")
