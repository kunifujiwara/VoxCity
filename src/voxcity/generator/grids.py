import os
import numpy as np

from ..downloader.mbfp import get_mbfp_gdf
from ..downloader.osm import load_gdf_from_openstreetmap, load_land_cover_gdf_from_osm
from ..downloader.oemj import save_oemj_as_geotiff
from ..downloader.eubucco import load_gdf_from_eubucco
from ..downloader.overture import load_gdf_from_overture
from ..downloader.gba import load_gdf_from_gba

from ..downloader.gee import (
    initialize_earth_engine,
    get_roi,
    get_ee_image_collection,
    get_ee_image,
    save_geotiff,
    get_dem_image,
    save_geotiff_esa_land_cover,
    save_geotiff_esri_landcover,
    save_geotiff_dynamic_world_v1,
    save_geotiff_open_buildings_temporal,
    save_geotiff_dsm_minus_dtm,
)

from ..geoprocessor.raster import (
    process_grid,
    create_land_cover_grid_from_geotiff_polygon,
    create_height_grid_from_geotiff_polygon,
    create_building_height_grid_from_gdf_polygon,
    create_dem_grid_from_geotiff_polygon,
    create_land_cover_grid_from_gdf_polygon,
    create_building_height_grid_from_open_building_temporal_polygon,
    create_canopy_grids_from_tree_gdf,
)

from ..utils.lc import convert_land_cover_array, get_land_cover_classes
from ..geoprocessor.io import get_gdf_from_gpkg
from ..visualizer.grids import visualize_land_cover_grid, visualize_numerical_grid


def get_land_cover_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):
    print("Creating Land Use Land Cover grid\n ")
    print(f"Data source: {source}")

    if source not in ["OpenStreetMap", "OpenEarthMapJapan"]:
        initialize_earth_engine()

    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "land_cover.tif")

    if source == 'Urbanwatch':
        roi = get_roi(rectangle_vertices)
        collection_name = "projects/sat-io/open-datasets/HRLC/urban-watch-cities"
        image = get_ee_image_collection(collection_name, roi)
        save_geotiff(image, geotiff_path)
    elif source == 'ESA WorldCover':
        roi = get_roi(rectangle_vertices)
        save_geotiff_esa_land_cover(roi, geotiff_path)
    elif source == 'ESRI 10m Annual Land Cover':
        esri_landcover_year = kwargs.get("esri_landcover_year")
        roi = get_roi(rectangle_vertices)
        save_geotiff_esri_landcover(roi, geotiff_path, year=esri_landcover_year)
    elif source == 'Dynamic World V1':
        dynamic_world_date = kwargs.get("dynamic_world_date")
        roi = get_roi(rectangle_vertices)
        save_geotiff_dynamic_world_v1(roi, geotiff_path, dynamic_world_date)
    elif source == 'OpenEarthMapJapan':
        ssl_verify = kwargs.get('ssl_verify', kwargs.get('verify', True))
        allow_insecure_ssl = kwargs.get('allow_insecure_ssl', False)
        allow_http_fallback = kwargs.get('allow_http_fallback', False)
        timeout_s = kwargs.get('timeout', 30)

        save_oemj_as_geotiff(
            rectangle_vertices,
            geotiff_path,
            ssl_verify=ssl_verify,
            allow_insecure_ssl=allow_insecure_ssl,
            allow_http_fallback=allow_http_fallback,
            timeout_s=timeout_s,
        )
        if not os.path.exists(geotiff_path):
            raise FileNotFoundError(
                f"OEMJ download failed; expected GeoTIFF not found: {geotiff_path}. "
                "You can try setting ssl_verify=False or allow_http_fallback=True in kwargs."
            )
    elif source == 'OpenStreetMap':
        land_cover_gdf = load_land_cover_gdf_from_osm(rectangle_vertices)

    land_cover_classes = get_land_cover_classes(source)

    if source == 'OpenStreetMap':
        default_class = kwargs.get('default_land_cover_class', 'Developed space')
        land_cover_grid_str = create_land_cover_grid_from_gdf_polygon(land_cover_gdf, meshsize, source, rectangle_vertices, default_class=default_class)
    else:
        land_cover_grid_str = create_land_cover_grid_from_geotiff_polygon(geotiff_path, meshsize, land_cover_classes, rectangle_vertices)

    color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}

    grid_vis = kwargs.get("gridvis", True)
    if grid_vis:
        visualize_land_cover_grid(np.flipud(land_cover_grid_str), meshsize, color_map, land_cover_classes)

    land_cover_grid_int = convert_land_cover_array(land_cover_grid_str, land_cover_classes)
    return land_cover_grid_int


def get_building_height_grid(rectangle_vertices, meshsize, source, output_dir, building_gdf=None, **kwargs):
    ee_required_sources = {"Open Building 2.5D Temporal"}
    if source in ee_required_sources:
        initialize_earth_engine()

    print("Creating Building Height grid\n ")
    print(f"Data source: {source}")

    os.makedirs(output_dir, exist_ok=True)

    if building_gdf is not None:
        gdf = building_gdf
        print("Using provided GeoDataFrame for building data")
    else:
        floor_height = kwargs.get("floor_height", 3.0)
        if source == 'Microsoft Building Footprints':
            gdf = get_mbfp_gdf(output_dir, rectangle_vertices)
        elif source == 'OpenStreetMap':
            gdf = load_gdf_from_openstreetmap(rectangle_vertices, floor_height=floor_height)
        elif source == "Open Building 2.5D Temporal":
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_open_building_temporal_polygon(meshsize, rectangle_vertices, output_dir)
        elif source == 'EUBUCCO v0.1':
            gdf = load_gdf_from_eubucco(rectangle_vertices, output_dir)
        elif source == "Overture":
            gdf = load_gdf_from_overture(rectangle_vertices, floor_height=floor_height)
        elif source in ("GBA", "Global Building Atlas"):
            clip_gba = kwargs.get("gba_clip", False)
            gba_download_dir = kwargs.get("gba_download_dir")
            gdf = load_gdf_from_gba(rectangle_vertices, download_dir=gba_download_dir, clip_to_rectangle=clip_gba)
        elif source == "Local file":
            _, extension = os.path.splitext(kwargs["building_path"])
            if extension == ".gpkg":
                gdf = get_gdf_from_gpkg(kwargs["building_path"], rectangle_vertices)
        elif source == "GeoDataFrame":
            raise ValueError("When source is 'GeoDataFrame', building_gdf parameter must be provided")

    building_complementary_source = kwargs.get("building_complementary_source")
    building_complement_height = kwargs.get("building_complement_height")
    overlapping_footprint = kwargs.get("overlapping_footprint", "auto")

    if (building_complementary_source is None) or (building_complementary_source=='None'):
        if source != "Open Building 2.5D Temporal":
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
    else:
        if building_complementary_source == "Open Building 2.5D Temporal":
            roi = get_roi(rectangle_vertices)
            os.makedirs(output_dir, exist_ok=True)
            geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
            save_geotiff_open_buildings_temporal(roi, geotiff_path_comp)
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
        elif building_complementary_source in ["England 1m DSM - DTM", "Netherlands 0.5m DSM - DTM"]:
            roi = get_roi(rectangle_vertices)
            os.makedirs(output_dir, exist_ok=True)
            geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
            save_geotiff_dsm_minus_dtm(roi, geotiff_path_comp, meshsize, building_complementary_source)
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
        else:
            if building_complementary_source == 'Microsoft Building Footprints':
                gdf_comp = get_mbfp_gdf(output_dir, rectangle_vertices)
            elif building_complementary_source == 'OpenStreetMap':
                gdf_comp = load_gdf_from_openstreetmap(rectangle_vertices, floor_height=floor_height)
            elif building_complementary_source == 'EUBUCCO v0.1':
                gdf_comp = load_gdf_from_eubucco(rectangle_vertices, output_dir)
            elif building_complementary_source == "Overture":
                gdf_comp = load_gdf_from_overture(rectangle_vertices, floor_height=floor_height)
            elif building_complementary_source in ("GBA", "Global Building Atlas"):
                clip_gba = kwargs.get("gba_clip", False)
                gba_download_dir = kwargs.get("gba_download_dir")
                gdf_comp = load_gdf_from_gba(rectangle_vertices, download_dir=gba_download_dir, clip_to_rectangle=clip_gba)
            elif building_complementary_source == "Local file":
                _, extension = os.path.splitext(kwargs["building_complementary_path"])
                if extension == ".gpkg":
                    gdf_comp = get_gdf_from_gpkg(kwargs["building_complementary_path"], rectangle_vertices)

            complement_building_footprints = kwargs.get("complement_building_footprints")
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, gdf_comp=gdf_comp, complement_building_footprints=complement_building_footprints, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)

    grid_vis = kwargs.get("gridvis", True)
    if grid_vis:
        building_height_grid_nan = building_height_grid.copy()
        building_height_grid_nan[building_height_grid_nan == 0] = np.nan
        visualize_numerical_grid(np.flipud(building_height_grid_nan), meshsize, "building height (m)", cmap='viridis', label='Value')

    return building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings


def get_canopy_height_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):
    print("Creating Canopy Height grid\n ")
    print(f"Data source: {source}")

    os.makedirs(output_dir, exist_ok=True)

    if source in ('GeoDataFrame', 'tree_gdf', 'Tree_GeoDataFrame', 'GDF'):
        tree_gdf = kwargs.get('tree_gdf')
        tree_gdf_path = kwargs.get('tree_gdf_path')
        if tree_gdf is None and tree_gdf_path is not None:
            _, ext = os.path.splitext(tree_gdf_path)
            if ext.lower() == '.gpkg':
                tree_gdf = get_gdf_from_gpkg(tree_gdf_path, rectangle_vertices)
            else:
                raise ValueError("Unsupported tree file format. Use .gpkg or pass a GeoDataFrame.")
        if tree_gdf is None:
            raise ValueError("When source='GeoDataFrame', provide 'tree_gdf' or 'tree_gdf_path'.")

        canopy_top, canopy_bottom = create_canopy_grids_from_tree_gdf(tree_gdf, meshsize, rectangle_vertices)

        grid_vis = kwargs.get("gridvis", True)
        if grid_vis:
            vis = canopy_top.copy()
            vis[vis == 0] = np.nan
            visualize_numerical_grid(np.flipud(vis), meshsize, "Tree canopy height (top)", cmap='Greens', label='Tree canopy height (m)')

        return canopy_top, canopy_bottom

    print("Data source: High Resolution Canopy Height Maps by WRI and Meta")
    initialize_earth_engine()

    geotiff_path = os.path.join(output_dir, "canopy_height.tif")

    roi = get_roi(rectangle_vertices)
    if source == 'High Resolution 1m Global Canopy Height Maps':
        collection_name = "projects/meta-forest-monitoring-okw37/assets/CanopyHeight"
        image = get_ee_image_collection(collection_name, roi)
    elif source == 'ETH Global Sentinel-2 10m Canopy Height (2020)':
        collection_name = "users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1"
        image = get_ee_image(collection_name, roi)
    else:
        raise ValueError(f"Unsupported canopy source: {source}")

    save_geotiff(image, geotiff_path, resolution=meshsize)
    canopy_height_grid = create_height_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices)

    trunk_height_ratio = kwargs.get("trunk_height_ratio")
    if trunk_height_ratio is None:
        trunk_height_ratio = 11.76 / 19.98
    canopy_bottom_grid = canopy_height_grid * float(trunk_height_ratio)

    grid_vis = kwargs.get("gridvis", True)
    if grid_vis:
        canopy_height_grid_nan = canopy_height_grid.copy()
        canopy_height_grid_nan[canopy_height_grid_nan == 0] = np.nan
        visualize_numerical_grid(np.flipud(canopy_height_grid_nan), meshsize, "Tree canopy height", cmap='Greens', label='Tree canopy height (m)')
    return canopy_height_grid, canopy_bottom_grid


def get_dem_grid(rectangle_vertices, meshsize, source, output_dir, **kwargs):
    print("Creating Digital Elevation Model (DEM) grid\n ")
    print(f"Data source: {source}")

    if source == "Local file":
        geotiff_path = kwargs["dem_path"]
    else:
        initialize_earth_engine()

        geotiff_path = os.path.join(output_dir, "dem.tif")

        buffer_distance = 100
        roi = get_roi(rectangle_vertices)
        roi_buffered = roi.buffer(buffer_distance)

        image = get_dem_image(roi_buffered, source)

        if source in ["England 1m DTM", 'DEM France 1m', 'DEM France 5m', 'AUSTRALIA 5M DEM', 'Netherlands 0.5m DTM']:
            save_geotiff(image, geotiff_path, scale=meshsize, region=roi_buffered, crs='EPSG:4326')
        elif source == 'USGS 3DEP 1m':
            scale = max(meshsize, 1.25)
            save_geotiff(image, geotiff_path, scale=scale, region=roi_buffered, crs='EPSG:4326')
        else:
            save_geotiff(image, geotiff_path, scale=30, region=roi_buffered)

    dem_interpolation = kwargs.get("dem_interpolation")
    dem_grid = create_dem_grid_from_geotiff_polygon(geotiff_path, meshsize, rectangle_vertices, dem_interpolation=dem_interpolation)

    grid_vis = kwargs.get("gridvis", True)
    if grid_vis:
        visualize_numerical_grid(np.flipud(dem_grid), meshsize, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')

    return dem_grid


