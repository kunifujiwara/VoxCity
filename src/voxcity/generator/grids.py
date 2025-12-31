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

from ..utils.lc import convert_land_cover_array, get_land_cover_classes, get_source_class_descriptions
from ..geoprocessor.io import get_gdf_from_gpkg
from ..visualizer.grids import visualize_land_cover_grid, visualize_numerical_grid


# Track last effective land cover source to help downstream components (e.g., voxelizer)
_LAST_EFFECTIVE_LC_SOURCE = None

def get_last_effective_land_cover_source():
    return _LAST_EFFECTIVE_LC_SOURCE


def get_land_cover_grid(rectangle_vertices, meshsize, source, output_dir, print_class_info=True, **kwargs):
    quiet = kwargs.get('quiet', False)
    if not quiet:
        print("Creating Land Use Land Cover grid\n ")
        print(f"Data source: {source}")
        if print_class_info:
            print(get_source_class_descriptions(source))

    if source not in ["OpenStreetMap", "OpenEarthMapJapan"]:
        try:
            initialize_earth_engine()
        except Exception as e:
            if not quiet:
                print("Earth Engine unavailable (", str(e), ") — falling back to OpenStreetMap for land cover.")
            source = 'OpenStreetMap'

    os.makedirs(output_dir, exist_ok=True)
    geotiff_path = os.path.join(output_dir, "land_cover.tif")

    # Track effective source to allow fallback behavior
    effective_source = source

    if source == 'Urbanwatch':
        roi = get_roi(rectangle_vertices)
        collection_name = "projects/sat-io/open-datasets/HRLC/urban-watch-cities"
        try:
            image = get_ee_image_collection(collection_name, roi)
            # If collection is empty, image operations may fail; guard with try/except
            save_geotiff(image, geotiff_path)
            if (not os.path.exists(geotiff_path)) or (os.path.getsize(geotiff_path) == 0):
                raise RuntimeError("Urbanwatch export produced no file")
        except Exception as e:
            if not quiet:
                print("Urbanwatch coverage not found for AOI; falling back to OpenStreetMap (reason:", str(e), ")")
            effective_source = 'OpenStreetMap'
            land_cover_gdf = load_land_cover_gdf_from_osm(rectangle_vertices)
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

    land_cover_classes = get_land_cover_classes(effective_source)

    if effective_source == 'OpenStreetMap':
        default_class = kwargs.get('default_land_cover_class', 'Developed space')
        land_cover_grid_str = create_land_cover_grid_from_gdf_polygon(land_cover_gdf, meshsize, effective_source, rectangle_vertices, default_class=default_class)
    else:
        land_cover_grid_str = create_land_cover_grid_from_geotiff_polygon(geotiff_path, meshsize, land_cover_classes, rectangle_vertices)

    color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}

    grid_vis = kwargs.get("gridvis", True)
    if grid_vis:
        visualize_land_cover_grid(np.flipud(land_cover_grid_str), meshsize, color_map, land_cover_classes)

    # Record effective source for downstream consumers
    global _LAST_EFFECTIVE_LC_SOURCE
    _LAST_EFFECTIVE_LC_SOURCE = effective_source

    land_cover_grid_int = convert_land_cover_array(land_cover_grid_str, land_cover_classes)
    return land_cover_grid_int


def get_building_height_grid(rectangle_vertices, meshsize, source, output_dir, building_gdf=None, **kwargs):
    ee_required_sources = {"Open Building 2.5D Temporal"}
    if source in ee_required_sources:
        initialize_earth_engine()

    quiet = kwargs.get('quiet', False)
    if not quiet:
        print("Creating Building Height grid\n ")
        print(f"Base data source: {source}")

    os.makedirs(output_dir, exist_ok=True)

    if building_gdf is not None:
        gdf = building_gdf
        if not quiet:
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
    try:
        comp_label = building_complementary_source if building_complementary_source not in (None, "") else "None"
        if not quiet:
            print(f"Complementary data source: {comp_label}")
    except Exception:
        pass
    building_complement_height = kwargs.get("building_complement_height")
    overlapping_footprint = kwargs.get("overlapping_footprint", "auto")

    if (building_complementary_source is None) or (building_complementary_source=='None'):
        if source != "Open Building 2.5D Temporal":
            building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
    else:
        if building_complementary_source == "Open Building 2.5D Temporal":
            try:
                roi = get_roi(rectangle_vertices)
                os.makedirs(output_dir, exist_ok=True)
                geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
                save_geotiff_open_buildings_temporal(roi, geotiff_path_comp)
                building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
            except Exception as e:
                if not quiet:
                    print("Open Building 2.5D Temporal requires Earth Engine (", str(e), ") — proceeding without complementary raster.")
                building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
        elif building_complementary_source in ["England 1m DSM - DTM", "Netherlands 0.5m DSM - DTM"]:
            try:
                roi = get_roi(rectangle_vertices)
                os.makedirs(output_dir, exist_ok=True)
                geotiff_path_comp = os.path.join(output_dir, "building_height.tif")
                save_geotiff_dsm_minus_dtm(roi, geotiff_path_comp, meshsize, building_complementary_source)
                building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, geotiff_path_comp=geotiff_path_comp, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
            except Exception as e:
                if not quiet:
                    print("DSM-DTM complementary raster requires Earth Engine (", str(e), ") — proceeding without complementary raster.")
                building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(gdf, meshsize, rectangle_vertices, complement_height=building_complement_height, overlapping_footprint=overlapping_footprint)
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
    quiet = kwargs.get('quiet', False)
    if not quiet:
        print("Creating Canopy Height grid\n ")
        print(f"Data source: {source}")

    os.makedirs(output_dir, exist_ok=True)

    # Explicit static path (no EE): use land cover mask with static height
    if source == 'Static':
        land_cover_grid = kwargs.get('land_cover_like')
        if land_cover_grid is None:
            # Minimal fallback if caller didn't provide land_cover_like
            canopy_top = np.zeros((1, 1), dtype=float)
            trunk_height_ratio = kwargs.get('trunk_height_ratio')
            if trunk_height_ratio is None:
                trunk_height_ratio = 11.76 / 19.98
            canopy_bottom = canopy_top * float(trunk_height_ratio)
            return canopy_top, canopy_bottom

        from ..utils.lc import get_land_cover_classes
        land_cover_source = kwargs.get('land_cover_source', 'OpenStreetMap')
        classes_map = get_land_cover_classes(land_cover_source)
        class_to_int = {name: i for i, name in enumerate(classes_map.values())}
        tree_labels = ["Tree", "Trees", "Tree Canopy"]
        tree_indices = [class_to_int[label] for label in tree_labels if label in class_to_int]

        canopy_top = np.zeros_like(land_cover_grid, dtype=float)
        static_tree_height = kwargs.get('static_tree_height', 10.0)
        tree_mask = np.isin(land_cover_grid, tree_indices) if tree_indices else np.zeros_like(land_cover_grid, dtype=bool)
        canopy_top[tree_mask] = static_tree_height

        trunk_height_ratio = kwargs.get('trunk_height_ratio')
        if trunk_height_ratio is None:
            trunk_height_ratio = 11.76 / 19.98
        canopy_bottom = canopy_top * float(trunk_height_ratio)

        grid_vis = kwargs.get("gridvis", True)
        if grid_vis:
            vis = canopy_top.copy(); vis[vis == 0] = np.nan
            visualize_numerical_grid(np.flipud(vis), meshsize, "Tree canopy height (top)", cmap='Greens', label='Tree canopy height (m)')
        return canopy_top, canopy_bottom

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

    try:
        initialize_earth_engine()
    except Exception as e:
        if not quiet:
            print("Earth Engine unavailable (", str(e), ") — falling back to Static canopy heights.")
        # Re-enter with explicit Static logic using land cover mask
        return get_canopy_height_grid(rectangle_vertices, meshsize, 'Static', output_dir, **kwargs)

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
    quiet = kwargs.get('quiet', False)
    if not quiet:
        print("Creating Digital Elevation Model (DEM) grid\n ")
        print(f"Data source: {source}")

    if source == "Local file":
        geotiff_path = kwargs["dem_path"]
    else:
        try:
            initialize_earth_engine()
        except Exception as e:
            if not quiet:
                print("Earth Engine unavailable (", str(e), ") — falling back to flat DEM.")
            # Compute grid shape directly from rectangle_vertices and meshsize
            from ..geoprocessor.raster.core import compute_grid_shape
            grid_shape = compute_grid_shape(rectangle_vertices, meshsize)
            dem_grid = np.zeros(grid_shape, dtype=float)
            return dem_grid

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


