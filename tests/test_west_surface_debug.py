"""
Debug test for west-facing vertical surface solar irradiance issue.

Issue: West-facing surfaces show shadows even when direct radiation comes from west.
However, some west vertical surfaces near the west edge have higher irradiance.

Coordinate conventions:
- Domain: x = West to East (so +x = East), y = South to North (so +y = North)
- Plotly visualization: -y = W, +y = E, -x = N, +x = S (flipped!)
- Solar azimuth (VoxCity): 0 = North, clockwise (so 270 = West)
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_sun_direction_conversion():
    """Test that sun direction is computed correctly for different azimuths."""
    print("\n" + "=" * 70)
    print("TEST: Sun direction conversion from azimuth/elevation")
    print("=" * 70)
    
    # Test various azimuths
    test_cases = [
        # (azimuth_ori, elevation, expected_description)
        (0, 45, "North (sun in north)"),
        (90, 45, "East (sun in east, morning)"),
        (180, 45, "South (sun in south, noon)"),
        (270, 45, "West (sun in west, afternoon)"),
    ]
    
    print("\nDomain coordinate convention:")
    print("  +x = East, -x = West")
    print("  +y = North, -y = South")
    print("  +z = Up")
    print()
    
    for azimuth_ori, elevation, desc in test_cases:
        # FIXED conversion (from integration.py after fix)
        azimuth_radians = np.deg2rad(azimuth_ori)  # No "180 - " transformation
        elevation_radians = np.deg2rad(elevation)
        
        cos_elev = np.cos(elevation_radians)
        
        # Standard spherical-to-Cartesian: x=sin(az), y=cos(az) for azimuth from North
        sun_dir_x = cos_elev * np.sin(azimuth_radians)  # East component
        sun_dir_y = cos_elev * np.cos(azimuth_radians)  # North component
        sun_dir_z = np.sin(elevation_radians)
        
        print(f"Azimuth={azimuth_ori:3d}° ({desc})")
        print(f"  Using FIXED conversion: x=sin(az), y=cos(az)")
        print(f"  sun_dir = ({sun_dir_x:.4f}, {sun_dir_y:.4f}, {sun_dir_z:.4f})")
        
        # Analyze: which surfaces should receive direct radiation?
        # A surface receives direct radiation when dot(sun_dir, normal) > 0
        # (sun_dir points toward sun, normal points outward from surface)
        normals = {
            "Up (IUP=0)": (0, 0, 1),
            "Down (IDOWN=1)": (0, 0, -1),
            "North (INORTH=2)": (0, 1, 0),
            "South (ISOUTH=3)": (0, -1, 0),
            "East (IEAST=4)": (1, 0, 0),
            "West (IWEST=5)": (-1, 0, 0),
        }
        
        print("  Expected illumination (dot > 0):")
        for face_name, normal in normals.items():
            dot = sun_dir_x * normal[0] + sun_dir_y * normal[1] + sun_dir_z * normal[2]
            illuminated = "YES" if dot > 0 else "no"
            print(f"    {face_name}: dot={dot:.4f} -> {illuminated}")
        print()


def test_face_sun_check_in_raytracing():
    """Test the face_sun check logic from raytracing.py."""
    print("\n" + "=" * 70)
    print("TEST: face_sun check logic from raytracing.py (with FIXED sun_dir)")
    print("=" * 70)
    
    # This is the current logic in raytracing.py:
    # direction == 4 (East):  face_sun = 1 if sun_dir[0] > 0 else 0
    # direction == 5 (West):  face_sun = 1 if sun_dir[0] < 0 else 0
    
    print("\nCurrent logic in raytracing.py:")
    print("  East (dir=4):  face_sun = 1 if sun_dir[0] > 0")
    print("  West (dir=5):  face_sun = 1 if sun_dir[0] < 0")
    print()
    
    # Test with sun coming from west (azimuth=270)
    azimuth_ori = 270  # West
    elevation = 45
    
    # FIXED conversion
    azimuth_radians = np.deg2rad(azimuth_ori)
    elevation_radians = np.deg2rad(elevation)
    
    cos_elev = np.cos(elevation_radians)
    sun_dir_x = cos_elev * np.sin(azimuth_radians)  # East component
    sun_dir_y = cos_elev * np.cos(azimuth_radians)  # North component
    sun_dir_z = np.sin(elevation_radians)
    
    print(f"Sun azimuth = {azimuth_ori}° (West)")
    print(f"  sun_dir = ({sun_dir_x:.4f}, {sun_dir_y:.4f}, {sun_dir_z:.4f})")
    print(f"  sun_dir[0] = {sun_dir_x:.4f}")
    print()
    
    # Apply raytracing logic
    east_face_sun = 1 if sun_dir_x > 0 else 0
    west_face_sun = 1 if sun_dir_x < 0 else 0
    
    print("Raytracing face_sun results:")
    print(f"  East-facing (dir=4): face_sun = {east_face_sun}")
    print(f"  West-facing (dir=5): face_sun = {west_face_sun}")
    print()
    
    # What should happen:
    # Sun in west -> sun is at negative x -> sun_dir points toward negative x
    # So sun_dir[0] should be NEGATIVE when sun is in west
    # West-facing surface (normal = -x) should be illuminated when sun is in west
    # The check "sun_dir[0] < 0" for west is correct!
    # But wait, let's check the actual sun_dir value...
    
    print("Analysis:")
    print(f"  Sun in west means sun is at -x side of domain")
    print(f"  Vector pointing TOWARD sun should have NEGATIVE x component")
    print(f"  Actual sun_dir[0] = {sun_dir_x:.4f}")
    if sun_dir_x < 0:
        print("  CORRECT: sun_dir[0] IS negative, as expected for sun in west")
        print("  West-facing surfaces will be correctly illuminated!")
    else:
        print("  BUG: sun_dir[0] is POSITIVE but sun is in west!")
        print("  This means the azimuth conversion is incorrect!")


def test_correct_azimuth_conversion():
    """Derive the correct azimuth conversion formula."""
    print("\n" + "=" * 70)
    print("TEST: Deriving correct azimuth conversion")
    print("=" * 70)
    
    # VoxCity/Astral azimuth convention:
    # - 0° = North
    # - 90° = East
    # - 180° = South
    # - 270° = West
    
    # Domain coordinate system:
    # - +x = East
    # - +y = North
    
    # For sun position, we need:
    # - sun in North (azimuth=0): sun_dir should point toward +y → (0, +, +)
    # - sun in East (azimuth=90): sun_dir should point toward +x → (+, 0, +)
    # - sun in South (azimuth=180): sun_dir should point toward -y → (0, -, +)
    # - sun in West (azimuth=270): sun_dir should point toward -x → (-, 0, +)
    
    print("\nExpected sun_dir for each azimuth:")
    print("  Azimuth=0° (North):   sun_dir = (0, +, z)")
    print("  Azimuth=90° (East):   sun_dir = (+, 0, z)")
    print("  Azimuth=180° (South): sun_dir = (0, -, z)")
    print("  Azimuth=270° (West):  sun_dir = (-, 0, z)")
    print()
    
    # Standard spherical to Cartesian conversion:
    # x = r * cos(elevation) * sin(azimuth)
    # y = r * cos(elevation) * cos(azimuth)
    # z = r * sin(elevation)
    # This assumes azimuth is measured from +y (North) clockwise
    
    print("Using standard conversion (azimuth from North, clockwise):")
    print("  x = cos(elev) * sin(azimuth)")
    print("  y = cos(elev) * cos(azimuth)")
    print()
    
    for azimuth_ori in [0, 90, 180, 270]:
        elevation = 45
        az_rad = np.deg2rad(azimuth_ori)
        el_rad = np.deg2rad(elevation)
        
        x = np.cos(el_rad) * np.sin(az_rad)
        y = np.cos(el_rad) * np.cos(az_rad)
        z = np.sin(el_rad)
        
        print(f"  Azimuth={azimuth_ori:3d}°: sun_dir = ({x:.4f}, {y:.4f}, {z:.4f})")
    
    print()
    print("This is the CORRECT conversion!")
    print()
    print("Current buggy code does: azimuth_degrees = 180 - azimuth_ori")
    print("This flips the direction incorrectly!")


def test_with_voxcity_data():
    """Test with actual VoxCity data."""
    print("\n" + "=" * 70)
    print("TEST: Using actual VoxCity data")
    print("=" * 70)
    
    try:
        from voxcity.generator.io import load_voxcity
        
        # Try multiple possible paths
        possible_paths = [
            "demo/output/voxcity.pkl",
            "output/voxcity.pkl",
            "../demo/output/voxcity.pkl",
        ]
        
        voxcity = None
        for path in possible_paths:
            try:
                voxcity = load_voxcity(path)
                print(f"Loaded VoxCity data from: {path}")
                break
            except FileNotFoundError:
                continue
        
        if voxcity is None:
            print("Could not find voxcity.pkl in expected locations")
            return
        
        # Get location
        extras = getattr(voxcity, 'extras', None)
        if isinstance(extras, dict):
            rect_vertices = extras.get('rectangle_vertices', None)
            if rect_vertices:
                lats = [v[1] for v in rect_vertices]
                lons = [v[0] for v in rect_vertices]
                lat = np.mean(lats)
                lon = np.mean(lons)
                print(f"Location: lat={lat:.4f}, lon={lon:.4f}")
        
        # Test instantaneous calculation with sun from west
        print("\nTesting instantaneous calculation with sun from west (3 PM on June 1)...")
        
        # First, let's just check what the solar position calculation gives us
        from voxcity.simulator_gpu.solar.integration import _get_solar_positions_astral
        import pandas as pd
        from datetime import datetime
        import pytz
        
        # Phoenix location (from the EPW file path)
        lat = 33.45  # Phoenix approximate latitude
        lon = -112.07  # Phoenix approximate longitude
        tz_offset = -7  # MST
        
        # Create a datetime for June 1, 3 PM local time
        local_tz = pytz.FixedOffset(int(tz_offset * 60))
        dt_local = datetime(2024, 6, 1, 15, 0, 0, tzinfo=local_tz)
        dt_utc = dt_local.astimezone(pytz.UTC)
        
        dates = pd.DatetimeIndex([dt_utc])
        solar_pos = _get_solar_positions_astral(dates, lon, lat)
        
        azimuth = float(solar_pos.iloc[0]['azimuth'])
        elevation = float(solar_pos.iloc[0]['elevation'])
        
        print(f"Solar position at 3 PM June 1 (Phoenix):")
        print(f"  Azimuth: {azimuth:.2f}°")
        print(f"  Elevation: {elevation:.2f}°")
        
        # Calculate sun_dir using current (buggy) method
        azimuth_degrees = 180 - azimuth
        az_rad = np.deg2rad(azimuth_degrees)
        el_rad = np.deg2rad(elevation)
        
        sun_dir_x_buggy = np.cos(el_rad) * np.cos(az_rad)
        sun_dir_y_buggy = np.cos(el_rad) * np.sin(az_rad)
        sun_dir_z = np.sin(el_rad)
        
        print(f"\nCurrent (buggy) sun_dir calculation:")
        print(f"  azimuth_degrees = 180 - {azimuth:.2f} = {azimuth_degrees:.2f}")
        print(f"  sun_dir = ({sun_dir_x_buggy:.4f}, {sun_dir_y_buggy:.4f}, {sun_dir_z:.4f})")
        
        # Calculate sun_dir using correct method
        az_rad_correct = np.deg2rad(azimuth)
        sun_dir_x_correct = np.cos(el_rad) * np.sin(az_rad_correct)
        sun_dir_y_correct = np.cos(el_rad) * np.cos(az_rad_correct)
        
        print(f"\nCorrect sun_dir calculation:")
        print(f"  sun_dir = ({sun_dir_x_correct:.4f}, {sun_dir_y_correct:.4f}, {sun_dir_z:.4f})")
        
        # Check which surfaces should be illuminated
        print(f"\nFace illumination analysis (sun coming from ~{azimuth:.0f}° azimuth):")
        if 225 < azimuth < 315:
            print(f"  Sun is in the WEST, so WEST-facing surfaces should be illuminated")
            print(f"  West-facing surface has normal = (-1, 0, 0)")
            print(f"  Correct sun_dir has x = {sun_dir_x_correct:.4f}")
            if sun_dir_x_correct < 0:
                print(f"  CORRECT: sun_dir[0] < 0, west surfaces would be illuminated")
            else:
                print(f"  BUG: sun_dir[0] > 0, west surfaces would be shadowed!")
        
    except Exception as e:
        print(f"Could not load VoxCity data: {e}")
        print("Skipping VoxCity test")


def print_fix_recommendation():
    """Print the recommended fix."""
    print("\n" + "=" * 70)
    print("RECOMMENDED FIX")
    print("=" * 70)
    
    print("""
The bug is in the azimuth-to-sun_dir conversion in integration.py.

CURRENT (BUGGY) CODE:
    azimuth_degrees = 180 - azimuth_degrees_ori
    azimuth_radians = np.deg2rad(azimuth_degrees)
    sun_dir_x = cos_elev * np.cos(azimuth_radians)
    sun_dir_y = cos_elev * np.sin(azimuth_radians)

CORRECT CODE:
    azimuth_radians = np.deg2rad(azimuth_degrees_ori)
    sun_dir_x = cos_elev * np.sin(azimuth_radians)
    sun_dir_y = cos_elev * np.cos(azimuth_radians)

The issue is that the standard spherical-to-Cartesian conversion for 
azimuth measured from North (clockwise) is:
    x = r * cos(elevation) * sin(azimuth)  # East component
    y = r * cos(elevation) * cos(azimuth)  # North component

The current code incorrectly applies "180 - azimuth" and uses cos/sin
in the wrong order (cos for x, sin for y).

Files to fix:
1. src/voxcity/simulator_gpu/solar/integration.py
   - Function: _compute_sun_direction (line ~288)
   - Function: get_building_solar_irradiance (line ~2420)
""")


if __name__ == "__main__":
    test_sun_direction_conversion()
    test_face_sun_check_in_raytracing()
    test_correct_azimuth_conversion()
    test_with_voxcity_data()
    print_fix_recommendation()
