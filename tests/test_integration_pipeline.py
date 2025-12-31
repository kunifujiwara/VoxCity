import pytest
import os
import tempfile
import shutil
from pathlib import Path

# Apply Earth Engine patch for CI service account authentication
def _patch_voxcity_gee_for_service_account():
    """Patch VoxelCity's Earth Engine initialization for service account support."""
    try:
        import ee
        import json
        from voxcity.downloader import gee
        
        original_init = gee.initialize_earth_engine
        
        def patched_init(**kwargs):
            credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
            if credentials_path and os.path.exists(credentials_path):
                try:
                    with open(credentials_path, 'r') as f:
                        key_data = json.load(f)
                    credentials = ee.ServiceAccountCredentials(
                        email=key_data['client_email'], 
                        key_file=credentials_path
                    )
                    ee.Initialize(credentials)
                    return
                except Exception:
                    pass
            return original_init(**kwargs)
        
        gee.initialize_earth_engine = patched_init
    except ImportError:
        pass

_patch_voxcity_gee_for_service_account()


def run_voxelcity_pipeline(
    rectangle_vertices,
    building_source,
    building_complementary_source,
    land_cover_source,
    canopy_height_source,
    dem_source,
    meshsize,
    kwargs
):
    """VoxelCity pipeline integration test function."""
    import os
    from time import perf_counter
    import numpy as np

    run_sim = bool(kwargs.get("run_sim", False))

    step_times = {}
    def t_start():
        return perf_counter()
    def t_end(label, t0):
        dt = perf_counter() - t0
        step_times[label] = dt
        print(f"{label}: {dt:.2f}s")

    # 4.1 Get voxel city data
    from voxcity.generator import get_voxcity
    t0 = t_start()
    city = get_voxcity(
        rectangle_vertices,
        meshsize,
        building_source=building_source,
        land_cover_source=land_cover_source,
        canopy_height_source=canopy_height_source,
        dem_source=dem_source,
        **kwargs
    )
    t_end("4.1 get_voxcity", t0)

    # 4.2 Visualize voxel city
    from voxcity.utils.visualization import visualize_voxcity_multi_view
    t0 = t_start()
    # Avoid rendering heavy 3D views in CI to prevent VTK segfaults
    visualize_voxcity_multi_view(city.voxels.classes, city.voxels.meta.meshsize, show_views=False)
    t_end("4.2 visualize_voxcity_multi_view", t0)

    # 5.1 ENVI-MET INX and EDB
    from voxcity.exporter.envimet import export_inx, generate_edb_file
    envimet_kwargs = {
        "output_directory": os.path.join(kwargs.get("output_dir", "output"), "envimet"),
        "file_basename": 'voxcity',
        "author_name": "VoxCity Test",
        "model_description": "generated and exported using VoxCity",
        "domain_building_max_height_ratio": 2,
        "useTelescoping_grid": True,
        "verticalStretch": 20,
        "min_grids_Z": 20,
        "lad": 1.0
    }
    os.makedirs(envimet_kwargs["output_directory"], exist_ok=True)
    t0 = t_start()
    export_inx(
        city,
        output_directory=envimet_kwargs["output_directory"],
        file_basename=envimet_kwargs["file_basename"],
        author_name=envimet_kwargs["author_name"],
        model_description=envimet_kwargs["model_description"],
        domain_building_max_height_ratio=envimet_kwargs["domain_building_max_height_ratio"],
        useTelescoping_grid=envimet_kwargs["useTelescoping_grid"],
        verticalStretch=envimet_kwargs["verticalStretch"],
        min_grids_Z=envimet_kwargs["min_grids_Z"],
    )
    t_end("5.1 export_inx", t0)

    t0 = t_start()
    generate_edb_file(**envimet_kwargs)
    t_end("5.1 generate_edb_file", t0)

    # 5.2 MagicaVoxel VOX
    from voxcity.exporter.magicavoxel import export_magicavoxel_vox
    mv_outdir = os.path.join(kwargs.get("output_dir", "output"), "magicavoxel")
    os.makedirs(mv_outdir, exist_ok=True)
    t0 = t_start()
    export_magicavoxel_vox(city.voxels.classes, mv_outdir)
    t_end("5.2 export_magicavoxel_vox", t0)

    # 5.3 OBJ export
    from voxcity.exporter.obj import export_obj
    obj_outdir = os.path.join(kwargs.get("output_dir", "output"), "obj")
    os.makedirs(obj_outdir, exist_ok=True)
    t0 = t_start()
    export_obj(city.voxels.classes, obj_outdir, "voxcity", city.voxels.meta.meshsize)
    t_end("5.3 export_obj", t0)

    # 5.4 CityLES export
    from voxcity.exporter.cityles import export_cityles
    cityles_kwargs = {
        "output_directory": os.path.join(kwargs.get("output_dir", "output"), "cityles"),
        "building_material": "concrete",
        "tree_type": "deciduous",
        "tree_base_ratio": 0.3
    }
    os.makedirs(cityles_kwargs["output_directory"], exist_ok=True)
    t0 = t_start()
    export_cityles(
        city,
        output_directory=cityles_kwargs["output_directory"],
        building_material=cityles_kwargs["building_material"],
        tree_type=cityles_kwargs["tree_type"],
        tree_base_ratio=cityles_kwargs["tree_base_ratio"],
    )
    t_end("5.4 export_cityles", t0)

    # Initialize results; simulations filled only if run_sim=True
    results = {
        "voxcity_grid": city.voxels.classes,
        "building_height_grid": city.buildings.heights,
        "building_min_height_grid": city.buildings.min_heights,
        "building_id_grid": city.buildings.ids,
        "canopy_height_grid": (city.tree_canopy.top if city.tree_canopy is not None else None),
        "land_cover_grid": city.land_cover.classes,
        "dem_grid": city.dem.elevation,
        "building_gdf": city.extras.get("building_gdf"),
        "step_times": step_times
    }

    if not run_sim:
        print("\nTiming summary (descending):")
        for label, sec in sorted(step_times.items(), key=lambda x: x[1], reverse=True):
            print(f"- {label}: {sec:.2f}s")
        return results

    # -------- Simulations (run only when run_sim=True) --------
    # Simplified simulation tests to avoid long execution times
    try:
        # 6.1.1 Solar on building surfaces (instantaneous) - simplified
        from voxcity.simulator.solar import get_building_global_solar_irradiance_using_epw
        irradiance_kwargs = {
            "calc_type": "instantaneous",
            "download_nearest_epw": True,
            "rectangle_vertices": rectangle_vertices,
            "calc_time": "01-01 12:00:00",
            "building_id_grid": city.buildings.ids
        }
        t0 = t_start()
        instantaneous_irradiance = get_building_global_solar_irradiance_using_epw(
            city.voxels.classes, city.voxels.meta.meshsize, **irradiance_kwargs
        )
        t_end("6.1.1 building_irradiance_instantaneous", t0)
        results["instantaneous_irradiance"] = instantaneous_irradiance
    except Exception as e:
        print(f"Skipping solar simulation due to: {e}")

    print("\nTiming summary (descending):")
    for label, sec in sorted(step_times.items(), key=lambda x: x[1], reverse=True):
        print(f"- {label}: {sec:.2f}s")

    return results


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp(prefix="voxcity_test_")
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.slow
@pytest.mark.integration
def test_voxelcity_pipeline_tokyo_small(temp_output_dir):
    """Test VoxelCity pipeline with small Tokyo area (fast version)."""
    test_case = {
        "cityname": "tokyo_test",
        "rectangle_vertices": [
            (139.75664352213442, 35.67358479332452),
            (139.75664352213442, 35.67509479332452),  # Smaller area
            (139.75816647786555, 35.67509479332452),  # Smaller area
            (139.75816647786555, 35.67358479332452)
        ],
        "building_source": "OpenStreetMap",
        "building_complementary_source": "None",
        "land_cover_source": "OpenEarthMapJapan",
        "canopy_height_source": "High Resolution 1m Global Canopy Height Maps",
        "dem_source": "DeltaDTM",
        "meshsize": 10,  # Larger meshsize for faster processing
        "kwargs": {
            "building_complementary_source": "None",
            "building_complement_height": 10,
            "output_dir": temp_output_dir,
            "dem_interpolation": True,
            "debug_voxel": True,
            "run_sim": False  # Skip heavy simulations for CI
        }
    }

    # Run the pipeline
    results = run_voxelcity_pipeline(
        rectangle_vertices=test_case["rectangle_vertices"],
        building_source=test_case["building_source"],
        building_complementary_source=test_case["building_complementary_source"],
        land_cover_source=test_case["land_cover_source"],
        canopy_height_source=test_case["canopy_height_source"],
        dem_source=test_case["dem_source"],
        meshsize=test_case["meshsize"],
        kwargs=test_case["kwargs"]
    )

    # Verify results
    assert results is not None
    assert "voxcity_grid" in results
    assert "building_height_grid" in results
    assert "step_times" in results
    
    # Check that grids have reasonable shapes
    assert results["voxcity_grid"].shape[0] > 0
    assert results["voxcity_grid"].shape[1] > 0
    assert results["voxcity_grid"].shape[2] > 0
    
    # Check that some outputs were created
    output_dirs = ["envimet", "magicavoxel", "obj", "cityles"]
    for dir_name in output_dirs:
        output_path = Path(temp_output_dir) / dir_name
        assert output_path.exists(), f"Output directory {dir_name} was not created"
    
    # Check timing data
    assert len(results["step_times"]) > 0
    assert "4.1 get_voxcity" in results["step_times"]


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("city_config", [
    {
        "name": "amsterdam_small",
        "rectangle_vertices": [
            (4.892469800736853, 52.36396229930818),
            (4.892469800736853, 52.36496229930818),  # Very small area
            (4.893469800736853, 52.36496229930818),
            (4.893469800736853, 52.36396229930818)
        ],
        "building_source": "OpenStreetMap",
        "land_cover_source": "OpenStreetMap",
        "canopy_height_source": "High Resolution 1m Global Canopy Height Maps",
        "dem_source": "Netherlands 0.5m DTM",
        "meshsize": 15
    },
    {
        "name": "london_small",
        "rectangle_vertices": [
            (-0.1510, 51.4792109596354),  # Adjusted coordinates
            (-0.1510, 51.4822109596354),  # Slightly larger area
            (-0.1480, 51.4822109596354),
            (-0.1480, 51.4792109596354)
        ],
        "building_source": "OpenStreetMap",
        "land_cover_source": "ESA WorldCover", 
        "canopy_height_source": "High Resolution 1m Global Canopy Height Maps",
        "dem_source": "England 1m DTM",
        "meshsize": 15
    }
])
def test_voxelcity_pipeline_multiple_cities(city_config, temp_output_dir):
    """Test VoxelCity pipeline with multiple small city configurations."""
    city_output_dir = os.path.join(temp_output_dir, city_config["name"])
    
    kwargs = {
        "building_complementary_source": "None",
        "building_complement_height": 10,
        "output_dir": city_output_dir,
        "dem_interpolation": True,
        "debug_voxel": True,
        "run_sim": False
    }

    # Run the pipeline with error handling
    try:
        results = run_voxelcity_pipeline(
            rectangle_vertices=city_config["rectangle_vertices"],
            building_source=city_config["building_source"],
            building_complementary_source="None",
            land_cover_source=city_config["land_cover_source"],
            canopy_height_source=city_config["canopy_height_source"],
            dem_source=city_config["dem_source"],
            meshsize=city_config["meshsize"],
            kwargs=kwargs
        )

        # Basic validation
        assert results is not None
        assert "voxcity_grid" in results
        assert results["voxcity_grid"] is not None
        assert len(results["step_times"]) > 0
        
    except (AttributeError, ValueError) as e:
        if "geometry column" in str(e) or "geometry data type" in str(e):
            pytest.skip(f"Skipping {city_config['name']} due to empty building data: {e}")
        else:
            raise


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv('GEE_AUTHENTICATED', 'false').lower() != 'true',
    reason="Google Earth Engine authentication not available"
)
def test_voxelcity_pipeline_with_gee_source(temp_output_dir):
    """Test VoxelCity pipeline with Google Earth Engine data sources."""
    test_case = {
        "rectangle_vertices": [
            (139.75664352213442, 35.67358479332452),
            (139.75664352213442, 35.67408479332452),  # Tiny area
            (139.75714352213442, 35.67408479332452),
            (139.75714352213442, 35.67358479332452)
        ],
        "building_source": "OpenStreetMap", 
        "land_cover_source": "Dynamic World V1",  # GEE source
        "canopy_height_source": "ETH Global Sentinel-2 10m Canopy Height (2020)",  # GEE source
        "dem_source": "DeltaDTM",
        "meshsize": 20,
        "kwargs": {
            "building_complementary_source": "None",
            "output_dir": temp_output_dir,
            "dem_interpolation": True,
            "dynamic_world_date": '2021-04-02',
            "run_sim": False
        }
    }

    # This test will only run if GEE is authenticated
    results = run_voxelcity_pipeline(
        rectangle_vertices=test_case["rectangle_vertices"],
        building_source=test_case["building_source"],
        building_complementary_source="None",
        land_cover_source=test_case["land_cover_source"],
        canopy_height_source=test_case["canopy_height_source"],
        dem_source=test_case["dem_source"],
        meshsize=test_case["meshsize"],
        kwargs=test_case["kwargs"]
    )

    # Verify GEE data was successfully integrated
    assert results is not None
    assert "land_cover_grid" in results
    assert results["land_cover_grid"] is not None


def test_pipeline_function_imports():
    """Test that all required modules can be imported."""
    # Test critical imports
    from voxcity.generator import get_voxcity
    from voxcity.utils.visualization import visualize_voxcity_multi_view
    from voxcity.exporter.envimet import export_inx, generate_edb_file
    from voxcity.exporter.magicavoxel import export_magicavoxel_vox
    from voxcity.exporter.obj import export_obj
    from voxcity.exporter.cityles import export_cityles
    
    # Optional imports that may fail gracefully
    try:
        from voxcity.simulator.solar import get_building_global_solar_irradiance_using_epw
        from voxcity.simulator.view import get_view_index
    except ImportError as e:
        pytest.skip(f"Optional simulator modules not available: {e}")


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__])
