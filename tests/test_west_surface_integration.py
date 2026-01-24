"""
Integration test for west-facing vertical surface solar irradiance.

This test verifies that the fix for the azimuth-to-sun_dir conversion
correctly illuminates west-facing surfaces when the sun is in the west.

Run this test after the fix has been applied to integration.py.
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_west_facing_irradiance():
    """
    Test that west-facing surfaces receive irradiance when sun is in the west.
    
    This is an integration test that:
    1. Loads actual VoxCity data
    2. Runs the GPU solar simulation at 3 PM (sun in west)
    3. Verifies that west-facing surfaces receive direct irradiance
    """
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: West-facing surface irradiance")
    print("=" * 70)
    
    try:
        from voxcity.generator.io import load_voxcity
        from voxcity.simulator_gpu.solar import get_building_global_solar_irradiance_using_epw
        
        # Load VoxCity data
        possible_paths = [
            "demo/output/voxcity.pkl",
            "output/voxcity.pkl",
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
            print("Could not find voxcity.pkl - skipping integration test")
            return True
        
        # Find EPW file
        epw_paths = [
            "demo/output/phoenix-sky.harbor.intl.ap_az_usa.epw",
            "output/phoenix-sky.harbor.intl.ap_az_usa.epw",
        ]
        
        epw_path = None
        for path in epw_paths:
            if Path(path).exists():
                epw_path = path
                print(f"Using EPW file: {path}")
                break
        
        if epw_path is None:
            print("Could not find EPW file - skipping integration test")
            return True
        
        # Run simulation at 3 PM June 1 (sun in west, azimuth ~262°)
        print("\nRunning GPU solar simulation at 3 PM June 1 (sun in west)...")
        
        cumulative_kwargs = {
            "calc_type": "instantaneous",
            "epw_file_path": epw_path,
            "calc_time": "06-01 15:00:00",  # June 1st at 3 PM
            "with_reflections": False,
            "progress_report": True,
        }
        
        result_mesh = get_building_global_solar_irradiance_using_epw(
            voxcity,
            **cumulative_kwargs
        )
        
        if result_mesh is None:
            print("ERROR: No result mesh returned")
            return False
        
        # Analyze results by face direction
        print("\nAnalyzing irradiance by face direction...")
        
        face_normals = result_mesh.face_normals
        global_irradiance = result_mesh.metadata.get('global', None)
        
        if global_irradiance is None:
            print("ERROR: No global irradiance in metadata")
            return False
        
        # Classify faces by direction
        # West-facing: normal has significant negative x component
        # East-facing: normal has significant positive x component
        
        tolerance = 0.5  # cos(60°) - faces within 60° of direction
        
        west_mask = face_normals[:, 0] < -tolerance
        east_mask = face_normals[:, 0] > tolerance
        up_mask = face_normals[:, 2] > tolerance
        
        # Filter out NaN values
        valid_mask = ~np.isnan(global_irradiance)
        
        west_irradiance = global_irradiance[west_mask & valid_mask]
        east_irradiance = global_irradiance[east_mask & valid_mask]
        up_irradiance = global_irradiance[up_mask & valid_mask]
        
        print(f"\nFace counts:")
        print(f"  West-facing: {len(west_irradiance)} faces")
        print(f"  East-facing: {len(east_irradiance)} faces")
        print(f"  Up-facing:   {len(up_irradiance)} faces")
        
        print(f"\nMean irradiance (W/m²):")
        if len(west_irradiance) > 0:
            print(f"  West-facing: {np.mean(west_irradiance):.1f} (should be HIGH - sun in west)")
        if len(east_irradiance) > 0:
            print(f"  East-facing: {np.mean(east_irradiance):.1f} (should be LOW - sun NOT in east)")
        if len(up_irradiance) > 0:
            print(f"  Up-facing:   {np.mean(up_irradiance):.1f}")
        
        # Verification
        success = True
        
        if len(west_irradiance) > 0 and len(east_irradiance) > 0:
            west_mean = np.mean(west_irradiance)
            east_mean = np.mean(east_irradiance)
            
            if west_mean > east_mean:
                print(f"\n✓ PASS: West-facing surfaces ({west_mean:.1f} W/m²) receive more "
                      f"irradiance than east-facing ({east_mean:.1f} W/m²)")
            else:
                print(f"\n✗ FAIL: West-facing surfaces ({west_mean:.1f} W/m²) should receive more "
                      f"irradiance than east-facing ({east_mean:.1f} W/m²) when sun is in west!")
                success = False
        
        # Check that west surfaces have non-trivial direct irradiance
        direct_irradiance = result_mesh.metadata.get('direct', None)
        if direct_irradiance is not None:
            west_direct = direct_irradiance[west_mask & valid_mask]
            if len(west_direct) > 0:
                west_direct_mean = np.mean(west_direct)
                if west_direct_mean > 50:  # Should have significant direct radiation
                    print(f"✓ PASS: West-facing surfaces have direct irradiance: {west_direct_mean:.1f} W/m²")
                else:
                    print(f"✗ FAIL: West-facing surfaces have low direct irradiance: {west_direct_mean:.1f} W/m²")
                    print("        (Expected >50 W/m² when sun is in west)")
                    success = False
        
        # Detailed analysis of west-facing surfaces
        print("\n" + "-" * 50)
        print("Detailed analysis of west-facing surfaces:")
        print("-" * 50)
        
        if len(west_irradiance) > 0:
            # Statistics
            print(f"\nWest-facing global irradiance statistics:")
            print(f"  Count: {len(west_irradiance)}")
            print(f"  Min:   {np.min(west_irradiance):.1f} W/m²")
            print(f"  Max:   {np.max(west_irradiance):.1f} W/m²")
            print(f"  Mean:  {np.mean(west_irradiance):.1f} W/m²")
            print(f"  Std:   {np.std(west_irradiance):.1f} W/m²")
            
            # Check for unreasonably low values
            # With sun in west at ~54° elevation, direct irradiance on west faces should be significant
            # DNI * cos(incidence_angle) where incidence is angle from surface normal
            # For west face with sun at azimuth 262° (almost due west), incidence ~ 0-30°
            # So expect at least 50% of DNI on unobstructed faces
            low_threshold = 100  # Faces below this might be shadowed or have issues
            very_low_threshold = 20  # Faces below this are definitely problematic if unobstructed
            
            low_count = np.sum(west_irradiance < low_threshold)
            very_low_count = np.sum(west_irradiance < very_low_threshold)
            
            print(f"\nWest faces with irradiance < {low_threshold} W/m²: {low_count} ({100*low_count/len(west_irradiance):.1f}%)")
            print(f"West faces with irradiance < {very_low_threshold} W/m²: {very_low_count} ({100*very_low_count/len(west_irradiance):.1f}%)")
            
            # These low values could be due to:
            # 1. Legitimate shadowing from other buildings
            # 2. Being on domain boundary (set to NaN, already filtered)
            # 3. Bug in the code (what we're checking for)
            
            if very_low_count > 0:
                print(f"\n  Note: {very_low_count} west faces have very low irradiance.")
                print("  These could be legitimately shadowed by other buildings.")
                
                # Show distribution
                percentiles = [0, 10, 25, 50, 75, 90, 100]
                print(f"\n  Percentile distribution of west-facing irradiance:")
                for p in percentiles:
                    val = np.percentile(west_irradiance, p)
                    print(f"    {p:3d}th percentile: {val:.1f} W/m²")
            
            # Compare direct component for west faces
            if direct_irradiance is not None:
                west_direct = direct_irradiance[west_mask & valid_mask]
                print(f"\nWest-facing DIRECT irradiance statistics:")
                print(f"  Min:   {np.min(west_direct):.1f} W/m²")
                print(f"  Max:   {np.max(west_direct):.1f} W/m²")
                print(f"  Mean:  {np.mean(west_direct):.1f} W/m²")
                
                # Check faces with 0 direct (fully shadowed)
                zero_direct = np.sum(west_direct < 1)
                print(f"  Faces with ~0 direct (shadowed): {zero_direct} ({100*zero_direct/len(west_direct):.1f}%)")
                
                # Analyze spatial distribution of shadowed west faces
                print("\n" + "-" * 50)
                print("Spatial analysis of shadowed west faces:")
                print("-" * 50)
                
                west_face_indices = np.where(west_mask & valid_mask)[0]
                shadowed_west = west_direct < 1
                lit_west = west_direct >= 1
                
                # Get face centers
                face_centers = result_mesh.triangles_center
                west_centers = face_centers[west_mask & valid_mask]
                
                if np.sum(shadowed_west) > 0 and np.sum(lit_west) > 0:
                    shadowed_centers = west_centers[shadowed_west]
                    lit_centers = west_centers[lit_west]
                    
                    print(f"\nShadowed west faces (x range): {shadowed_centers[:,0].min():.1f} to {shadowed_centers[:,0].max():.1f}")
                    print(f"Lit west faces (x range):      {lit_centers[:,0].min():.1f} to {lit_centers[:,0].max():.1f}")
                    
                    # Check if shadowed faces are on eastern part of domain
                    # (blocked by buildings to their west)
                    domain_x_mid = (face_centers[:,0].min() + face_centers[:,0].max()) / 2
                    shadowed_east_of_mid = np.sum(shadowed_centers[:,0] > domain_x_mid)
                    shadowed_west_of_mid = np.sum(shadowed_centers[:,0] <= domain_x_mid)
                    
                    print(f"\nShadowed west faces location:")
                    print(f"  East of domain center: {shadowed_east_of_mid}")
                    print(f"  West of domain center: {shadowed_west_of_mid}")
                    
                    if shadowed_east_of_mid > shadowed_west_of_mid:
                        print("  -> Most shadowed faces are in eastern part of domain")
                        print("     This is expected: they're blocked by buildings to their west")
                    else:
                        print("  -> Shadowed faces are distributed across domain")
                        print("     This suggests inter-building shadowing")
        
        return success
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_west_facing_irradiance()
    print("\n" + "=" * 70)
    if success:
        print("TEST PASSED: West-facing surfaces correctly illuminated")
    else:
        print("TEST FAILED: Issue with west-facing surface illumination")
    print("=" * 70)
    sys.exit(0 if success else 1)
