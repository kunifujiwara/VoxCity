"""
Test sun position calculation for Phoenix on June 1st.

This verifies that the sun position calculation is correct.
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from voxcity.simulator_gpu.solar.integration import _compute_sun_direction


def get_sun_position_astral(lat, lon, datetime_str, tz_offset=-7):
    """Get sun position using astral for comparison."""
    from datetime import datetime, timezone, timedelta
    from astral import Observer
    from astral.sun import elevation, azimuth
    
    # Parse datetime (assume local time)
    dt_local = datetime.fromisoformat(datetime_str)
    
    # Create timezone
    tz = timezone(timedelta(hours=tz_offset))
    dt_aware = dt_local.replace(tzinfo=tz)
    
    observer = Observer(latitude=lat, longitude=lon)
    
    el = elevation(observer=observer, dateandtime=dt_aware)
    az = azimuth(observer=observer, dateandtime=dt_aware)
    
    return az, el


def main():
    print("=" * 70)
    print("Sun Position Verification for Phoenix, AZ on June 1st")
    print("=" * 70)
    print()
    
    # Phoenix coordinates (from EPW)
    lat = 33.45  # Phoenix latitude
    lon = -111.98  # Phoenix longitude
    
    print(f"Location: Phoenix, AZ ({lat:.2f}°N, {lon:.2f}°W)")
    print()
    
    # Test times
    test_times = [
        ("2024-06-01 09:00:00", "Morning"),
        ("2024-06-01 12:00:00", "Noon"),
        ("2024-06-01 15:00:00", "Afternoon"),
        ("2024-06-01 18:00:00", "Evening"),
    ]
    
    print("Comparison of sun positions:")
    print(f"{'Time':<12} | {'Azimuth':>8} {'Elev':>6} | {'sun_x':>7} {'sun_y':>7} {'sun_z':>6} | Direction")
    print("-" * 80)
    
    for dt_str, label in test_times:
        try:
            azimuth, elevation = get_sun_position_astral(lat, lon, dt_str)
            
            # Compute sun direction
            sx, sy, sz, _ = _compute_sun_direction(azimuth, elevation)
            
            # Determine direction
            if azimuth < 90:
                direction = "NE"
            elif azimuth < 180:
                direction = "SE"
            elif azimuth < 270:
                direction = "SW"
            else:
                direction = "NW"
            
            print(f"{label:<12} | {azimuth:>8.1f} {elevation:>6.1f} | {sx:>7.3f} {sy:>7.3f} {sz:>6.3f} | {direction}")
            
            # Verify direction
            print(f"             | Sun should be in {direction}:")
            
            # Check x (North-South)
            if "N" in direction:
                if sx >= 0:
                    print(f"             |   ERROR: sun_x={sx:.3f} should be < 0 for North")
                else:
                    print(f"             |   OK: sun_x={sx:.3f} < 0 (sun in North)")
            else:  # S
                if sx <= 0:
                    print(f"             |   ERROR: sun_x={sx:.3f} should be > 0 for South")
                else:
                    print(f"             |   OK: sun_x={sx:.3f} > 0 (sun in South)")
            
            # Check y (West-East)
            if "E" in direction:
                if sy <= 0:
                    print(f"             |   ERROR: sun_y={sy:.3f} should be > 0 for East")
                else:
                    print(f"             |   OK: sun_y={sy:.3f} > 0 (sun in East)")
            else:  # W
                if sy >= 0:
                    print(f"             |   ERROR: sun_y={sy:.3f} should be < 0 for West")
                else:
                    print(f"             |   OK: sun_y={sy:.3f} < 0 (sun in West)")
            
            print()
            
        except Exception as e:
            print(f"{label:<12} | Error: {e}")
    
    print()
    print("VoxCity coordinate system reminder:")
    print("  x = South direction (+x = South)")
    print("  y = East direction (+y = East)")
    print("  z = Up direction (+z = Up)")
    print()
    print("Surface face_sun conditions:")
    print("  INORTH (+y, East-facing):  face_sun = 1 if sun_y > 0 (sun in East)")
    print("  ISOUTH (-y, West-facing):  face_sun = 1 if sun_y < 0 (sun in West)")
    print("  IEAST  (+x, South-facing): face_sun = 1 if sun_x > 0 (sun in South)")
    print("  IWEST  (-x, North-facing): face_sun = 1 if sun_x < 0 (sun in North)")


if __name__ == "__main__":
    main()
