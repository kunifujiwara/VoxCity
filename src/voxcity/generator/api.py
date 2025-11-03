import os
import numpy as np

from ..models import PipelineConfig
from .pipeline import VoxCityPipeline
from .grids import get_land_cover_grid
from .io import save_voxcity_data

from ..downloader.citygml import load_buid_dem_veg_from_citygml
from ..downloader.mbfp import get_mbfp_gdf
from ..downloader.osm import load_gdf_from_openstreetmap
from ..downloader.eubucco import load_gdf_from_eubucco
from ..downloader.overture import load_gdf_from_overture
from ..downloader.gba import load_gdf_from_gba
from ..downloader.gee import (
    get_roi,
    save_geotiff_open_buildings_temporal,
    save_geotiff_dsm_minus_dtm,
)

from ..geoprocessor.raster import (
    create_building_height_grid_from_gdf_polygon,
    create_vegetation_height_grid_from_gdf_polygon,
    create_dem_grid_from_gdf_polygon,
)
from ..utils.lc import get_land_cover_classes
from ..geoprocessor.io import get_gdf_from_gpkg
from ..visualizer.grids import visualize_numerical_grid
from ..utils.logging import get_logger


_logger = get_logger(__name__)

def _center_of_rectangle(rectangle_vertices):
    """
    Compute center (lon, lat) of a rectangle defined by vertices [(lon, lat), ...].
    Accepts open or closed rings; uses simple average of vertices.
    """
    lons = [p[0] for p in rectangle_vertices]
    lats = [p[1] for p in rectangle_vertices]
    return (sum(lons) / len(lons), sum(lats) / len(lats))


def auto_select_data_sources(rectangle_vertices):
    """
    Automatically choose data sources for buildings, land cover, canopy height, and DEM
    based on the target area's location.

    Rules (heuristic, partially inferred from latest availability):
    - Buildings (base): 'OpenStreetMap'.
    - Buildings (complementary):
        * USA, Europe, Australia -> 'Microsoft Building Footprints'
        * England -> 'England 1m DSM - DTM' (height from DSM-DTM)
        * Netherlands -> 'Netherlands 0.5m DSM - DTM' (height from DSM-DTM)
        * Africa, South Asia, SE Asia, Latin America & Caribbean -> 'Open Building 2.5D Temporal'
        * Otherwise -> 'None'
    - Land cover: USA -> 'Urbanwatch'; Japan -> 'OpenEarthMapJapan'; otherwise 'OpenStreetMap'.
      (If OSM is insufficient, consider 'ESA WorldCover' manually.)
    - Canopy height: 'High Resolution 1m Global Canopy Height Maps'.
    - DEM: High-resolution where available (USA, England, Australia, France, Netherlands), else 'FABDEM'.

    Returns a dict with keys: building_source, building_complementary_source,
    land_cover_source, canopy_height_source, dem_source.
    """
    try:
        from ..geoprocessor.utils import get_country_name
    except Exception:
        get_country_name = None

    center_lon, center_lat = _center_of_rectangle(rectangle_vertices)

    # Country detection (best-effort)
    country = None
    if get_country_name is not None:
        try:
            country = get_country_name(center_lon, center_lat)
        except Exception:
            country = None

    # Region helpers
    eu_countries = {
        'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Czech Republic',
        'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland',
        'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland',
        'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'
    }
    is_usa = (country == 'United States' or country == 'United States of America') or (-170 <= center_lon <= -65 and 20 <= center_lat <= 72)
    is_canada = (country == 'Canada')
    is_australia = (country == 'Australia')
    is_france = (country == 'France')
    is_england = (country == 'United Kingdom')  # Approximation: dataset covers England specifically
    is_netherlands = (country == 'Netherlands')
    is_japan = (country == 'Japan') or (127 <= center_lon <= 146 and 24 <= center_lat <= 46)
    is_europe = (country in eu_countries) or (-75 <= center_lon <= 60 and 25 <= center_lat <= 85)

    # Broad regions for OB 2.5D Temporal
    in_africa = (-25 <= center_lon <= 80 and -55 <= center_lat <= 45)
    in_south_asia = (50 <= center_lon <= 100 and 0 <= center_lat <= 35)
    in_se_asia = (90 <= center_lon <= 150 and -10 <= center_lat <= 25)
    in_latam_carib = (-110 <= center_lon <= -30 and -60 <= center_lat <= 30)

    # Building base source
    building_source = 'OpenStreetMap'

    # Building complementary source
    building_complementary_source = 'None'
    if is_england:
        building_complementary_source = 'England 1m DSM - DTM'
    elif is_netherlands:
        building_complementary_source = 'Netherlands 0.5m DSM - DTM'
    elif is_usa or is_australia or is_europe:
        building_complementary_source = 'Microsoft Building Footprints'
    elif in_africa or in_south_asia or in_se_asia or in_latam_carib:
        building_complementary_source = 'Open Building 2.5D Temporal'

    # Land cover source
    if is_usa:
        land_cover_source = 'Urbanwatch'
    elif is_japan:
        land_cover_source = 'OpenEarthMapJapan'
    else:
        land_cover_source = 'OpenStreetMap'

    # Canopy height source
    canopy_height_source = 'High Resolution 1m Global Canopy Height Maps'

    # DEM source
    if is_usa:
        dem_source = 'USGS 3DEP 1m'
    elif is_england:
        dem_source = 'England 1m DTM'
    elif is_australia:
        dem_source = 'AUSTRALIA 5M DEM'
    elif is_france:
        dem_source = 'DEM France 1m'
    elif is_netherlands:
        dem_source = 'Netherlands 0.5m DTM'
    else:
        dem_source = 'FABDEM'

    return {
        'building_source': building_source,
        'building_complementary_source': building_complementary_source,
        'land_cover_source': land_cover_source,
        'canopy_height_source': canopy_height_source,
        'dem_source': dem_source,
    }


def get_voxcity_auto(rectangle_vertices, meshsize, building_gdf=None, terrain_gdf=None, **kwargs):
    """
    Convenience wrapper that auto-selects sources from rectangle_vertices and
    calls get_voxcity. Additional kwargs are forwarded as-is.
    """
    selection = auto_select_data_sources(rectangle_vertices)
    building_source = selection['building_source']
    land_cover_source = selection['land_cover_source']
    canopy_height_source = selection['canopy_height_source']
    dem_source = selection['dem_source']
    # Ensure complementary source is respected
    kwargs = dict(kwargs)
    if 'building_complementary_source' not in kwargs:
        kwargs['building_complementary_source'] = selection['building_complementary_source']
    # Ensure missing building heights are complemented to 10 m
    if 'building_complement_height' not in kwargs:
        kwargs['building_complement_height'] = 10
    # Log explicit selection for user visibility
    try:
        _logger.info(
            "Selected data sources | Buildings(base)=%s, Buildings(comp)=%s, LandCover=%s, Canopy=%s, DEM=%s, ComplementHeight=%s",
            building_source,
            kwargs.get('building_complementary_source'),
            land_cover_source,
            canopy_height_source,
            dem_source,
            kwargs.get('building_complement_height'),
        )
    except Exception:
        pass

    city = get_voxcity(
        rectangle_vertices,
        building_source,
        land_cover_source,
        canopy_height_source,
        dem_source,
        meshsize,
        building_gdf=building_gdf,
        terrain_gdf=terrain_gdf,
        **kwargs,
    )

    # Attach selected sources to extras for downstream consumers
    try:
        selection_payload = dict(selection)
        selection_payload['building_complement_height'] = kwargs.get('building_complement_height')
        city.extras['selected_sources'] = selection_payload
    except Exception:
        pass

    return city

def get_voxcity(rectangle_vertices, building_source, land_cover_source, canopy_height_source, dem_source, meshsize, building_gdf=None, terrain_gdf=None, **kwargs):
    output_dir = kwargs.get("output_dir", "output")
    # Group incoming kwargs into structured options for consistency
    land_cover_keys = {
        # examples: source-specific options (placeholders kept broad for back-compat)
        "land_cover_path", "land_cover_resample", "land_cover_classes",
    }
    building_keys = {
        "overlapping_footprint", "gdf_comp", "geotiff_path_comp",
        "complement_building_footprints", "complement_height", "floor_height",
        "building_complementary_source", "building_complement_height",
        "building_complementary_path", "gba_clip", "gba_download_dir",
    }
    canopy_keys = {
        "min_canopy_height", "trunk_height_ratio", "static_tree_height",
    }
    dem_keys = {
        "flat_dem",
    }
    visualize_keys = {"gridvis", "mapvis"}
    io_keys = {"save_voxcity_data", "save_voxctiy_data", "save_data_path"}

    land_cover_options = {k: v for k, v in kwargs.items() if k in land_cover_keys}
    building_options = {k: v for k, v in kwargs.items() if k in building_keys}
    canopy_options = {k: v for k, v in kwargs.items() if k in canopy_keys}
    dem_options = {k: v for k, v in kwargs.items() if k in dem_keys}
    # Auto-set flat DEM when dem_source is None/empty and user didn't specify
    if (dem_source in (None, "", "None")) and ("flat_dem" not in dem_options):
        dem_options["flat_dem"] = True
    visualize_options = {k: v for k, v in kwargs.items() if k in visualize_keys}
    io_options = {k: v for k, v in kwargs.items() if k in io_keys}

    cfg = PipelineConfig(
        rectangle_vertices=rectangle_vertices,
        meshsize=float(meshsize),
        building_source=building_source,
        land_cover_source=land_cover_source,
        canopy_height_source=canopy_height_source,
        dem_source=dem_source,
        output_dir=output_dir,
        trunk_height_ratio=kwargs.get("trunk_height_ratio"),
        static_tree_height=kwargs.get("static_tree_height"),
        remove_perimeter_object=kwargs.get("remove_perimeter_object"),
        mapvis=bool(kwargs.get("mapvis", False)),
        gridvis=bool(kwargs.get("gridvis", True)),
        land_cover_options=land_cover_options,
        building_options=building_options,
        canopy_options=canopy_options,
        dem_options=dem_options,
        io_options=io_options,
        visualize_options=visualize_options,
    )
    city = VoxCityPipeline(meshsize=cfg.meshsize, rectangle_vertices=cfg.rectangle_vertices).run(cfg, building_gdf=building_gdf, terrain_gdf=terrain_gdf, **{k: v for k, v in kwargs.items() if k != 'output_dir'})

    # Backwards compatible save flag: prefer correct key, fallback to legacy misspelling
    _save_flag = io_options.get("save_voxcity_data", kwargs.get("save_voxcity_data", kwargs.get("save_voxctiy_data", True)))
    if _save_flag:
        save_path = io_options.get("save_data_path", kwargs.get("save_data_path", f"{output_dir}/voxcity_data.pkl"))
        save_voxcity_data(
            save_path,
            city.voxels.classes,
            city.buildings.heights,
            city.buildings.min_heights,
            city.buildings.ids,
            city.tree_canopy.top,
            city.land_cover.classes,
            city.dem.elevation,
            city.extras.get("building_gdf"),
            meshsize,
            rectangle_vertices,
        )

    return city


def get_voxcity_CityGML(rectangle_vertices, land_cover_source, canopy_height_source, meshsize, url_citygml=None, citygml_path=None, **kwargs):
    output_dir = kwargs.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    kwargs.pop('output_dir', None)

    ssl_verify = kwargs.pop('ssl_verify', kwargs.pop('verify', True))
    ca_bundle = kwargs.pop('ca_bundle', None)
    timeout = kwargs.pop('timeout', 60)

    building_gdf, terrain_gdf, vegetation_gdf = load_buid_dem_veg_from_citygml(
        url=url_citygml,
        citygml_path=citygml_path,
        base_dir=output_dir,
        rectangle_vertices=rectangle_vertices,
        ssl_verify=ssl_verify,
        ca_bundle=ca_bundle,
        timeout=timeout
    )

    try:
        import geopandas as gpd  # noqa: F401
        if building_gdf is not None:
            if building_gdf.crs is None:
                building_gdf = building_gdf.set_crs(epsg=4326)
            elif getattr(building_gdf.crs, 'to_epsg', lambda: None)() != 4326 and building_gdf.crs != "EPSG:4326":
                building_gdf = building_gdf.to_crs(epsg=4326)
        if terrain_gdf is not None:
            if terrain_gdf.crs is None:
                terrain_gdf = terrain_gdf.set_crs(epsg=4326)
            elif getattr(terrain_gdf.crs, 'to_epsg', lambda: None)() != 4326 and terrain_gdf.crs != "EPSG:4326":
                terrain_gdf = terrain_gdf.to_crs(epsg=4326)
        if vegetation_gdf is not None:
            if vegetation_gdf.crs is None:
                vegetation_gdf = vegetation_gdf.set_crs(epsg=4326)
            elif getattr(vegetation_gdf.crs, 'to_epsg', lambda: None)() != 4326 and vegetation_gdf.crs != "EPSG:4326":
                vegetation_gdf = vegetation_gdf.to_crs(epsg=4326)
    except Exception:
        pass

    land_cover_grid = get_land_cover_grid(rectangle_vertices, meshsize, land_cover_source, output_dir, **kwargs)

    print("Creating building height grid")
    building_complementary_source = kwargs.get("building_complementary_source")
    gdf_comp = None
    geotiff_path_comp = None
    complement_building_footprints = kwargs.get("complement_building_footprints")
    if complement_building_footprints is None and (building_complementary_source not in (None, "None")):
        complement_building_footprints = True

    if (building_complementary_source is not None) and (building_complementary_source != "None"):
        floor_height = kwargs.get("floor_height", 3.0)
        if building_complementary_source == 'Microsoft Building Footprints':
            gdf_comp = get_mbfp_gdf(kwargs.get("output_dir", "output"), rectangle_vertices)
        elif building_complementary_source == 'OpenStreetMap':
            gdf_comp = load_gdf_from_openstreetmap(rectangle_vertices, floor_height=floor_height)
        elif building_complementary_source == 'EUBUCCO v0.1':
            gdf_comp = load_gdf_from_eubucco(rectangle_vertices, kwargs.get("output_dir", "output"))
        elif building_complementary_source == 'Overture':
            gdf_comp = load_gdf_from_overture(rectangle_vertices, floor_height=floor_height)
        elif building_complementary_source in ("GBA", "Global Building Atlas"):
            clip_gba = kwargs.get("gba_clip", False)
            gba_download_dir = kwargs.get("gba_download_dir")
            gdf_comp = load_gdf_from_gba(rectangle_vertices, download_dir=gba_download_dir, clip_to_rectangle=clip_gba)
        elif building_complementary_source == 'Local file':
            comp_path = kwargs.get("building_complementary_path")
            if comp_path is not None:
                _, extension = os.path.splitext(comp_path)
                if extension == ".gpkg":
                    gdf_comp = get_gdf_from_gpkg(comp_path, rectangle_vertices)
        if gdf_comp is not None:
            try:
                if gdf_comp.crs is None:
                    gdf_comp = gdf_comp.set_crs(epsg=4326)
                elif getattr(gdf_comp.crs, 'to_epsg', lambda: None)() != 4326 and gdf_comp.crs != "EPSG:4326":
                    gdf_comp = gdf_comp.to_crs(epsg=4326)
            except Exception:
                pass
        elif building_complementary_source == "Open Building 2.5D Temporal":
            roi = get_roi(rectangle_vertices)
            os.makedirs(kwargs.get("output_dir", "output"), exist_ok=True)
            geotiff_path_comp = os.path.join(kwargs.get("output_dir", "output"), "building_height.tif")
            save_geotiff_open_buildings_temporal(roi, geotiff_path_comp)
        elif building_complementary_source in ["England 1m DSM - DTM", "Netherlands 0.5m DSM - DTM"]:
            roi = get_roi(rectangle_vertices)
            os.makedirs(kwargs.get("output_dir", "output"), exist_ok=True)
            geotiff_path_comp = os.path.join(kwargs.get("output_dir", "output"), "building_height.tif")
            save_geotiff_dsm_minus_dtm(roi, geotiff_path_comp, meshsize, building_complementary_source)

    _allowed_building_kwargs = {
        "overlapping_footprint",
        "gdf_comp",
        "geotiff_path_comp",
        "complement_building_footprints",
        "complement_height",
    }
    _building_kwargs = {k: v for k, v in kwargs.items() if k in _allowed_building_kwargs}
    if gdf_comp is not None:
        _building_kwargs["gdf_comp"] = gdf_comp
    if geotiff_path_comp is not None:
        _building_kwargs["geotiff_path_comp"] = geotiff_path_comp
    if complement_building_footprints is not None:
        _building_kwargs["complement_building_footprints"] = complement_building_footprints

    comp_height_user = kwargs.get("building_complement_height")
    if comp_height_user is not None:
        _building_kwargs["complement_height"] = comp_height_user
    if _building_kwargs.get("complement_building_footprints") and ("complement_height" not in _building_kwargs):
        _building_kwargs["complement_height"] = 10.0

    building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = create_building_height_grid_from_gdf_polygon(
        building_gdf, meshsize, rectangle_vertices, **_building_kwargs
    )

    grid_vis = kwargs.get("gridvis", True)
    if grid_vis:
        building_height_grid_nan = building_height_grid.copy()
        building_height_grid_nan[building_height_grid_nan == 0] = np.nan
        visualize_numerical_grid(np.flipud(building_height_grid_nan), meshsize, "building height (m)", cmap='viridis', label='Value')

    if canopy_height_source == "Static":
        canopy_height_grid_comp = np.zeros_like(land_cover_grid, dtype=float)
        static_tree_height = kwargs.get("static_tree_height", 10.0)
        _classes = get_land_cover_classes(land_cover_source)
        _class_to_int = {name: i for i, name in enumerate(_classes.values())}
        _tree_labels = ["Tree", "Trees", "Tree Canopy"]
        _tree_indices = [_class_to_int[label] for label in _tree_labels if label in _class_to_int]
        tree_mask = np.isin(land_cover_grid, _tree_indices) if _tree_indices else np.zeros_like(land_cover_grid, dtype=bool)
        canopy_height_grid_comp[tree_mask] = static_tree_height
        trunk_height_ratio = kwargs.get("trunk_height_ratio")
        if trunk_height_ratio is None:
            trunk_height_ratio = 11.76 / 19.98
        canopy_bottom_height_grid_comp = canopy_height_grid_comp * float(trunk_height_ratio)
    else:
        from .grids import get_canopy_height_grid
        canopy_height_grid_comp, canopy_bottom_height_grid_comp = get_canopy_height_grid(rectangle_vertices, meshsize, canopy_height_source, output_dir, **kwargs)

    if vegetation_gdf is not None:
        canopy_height_grid = create_vegetation_height_grid_from_gdf_polygon(vegetation_gdf, meshsize, rectangle_vertices)
        trunk_height_ratio = kwargs.get("trunk_height_ratio")
        if trunk_height_ratio is None:
            trunk_height_ratio = 11.76 / 19.98
        canopy_bottom_height_grid = canopy_height_grid * float(trunk_height_ratio)
    else:
        canopy_height_grid = np.zeros_like(building_height_grid)
        canopy_bottom_height_grid = np.zeros_like(building_height_grid)

    mask = (canopy_height_grid == 0) & (canopy_height_grid_comp != 0)
    canopy_height_grid[mask] = canopy_height_grid_comp[mask]
    mask_b = (canopy_bottom_height_grid == 0) & (canopy_bottom_height_grid_comp != 0)
    canopy_bottom_height_grid[mask_b] = canopy_bottom_height_grid_comp[mask_b]
    canopy_bottom_height_grid = np.minimum(canopy_bottom_height_grid, canopy_height_grid)

    if kwargs.pop('flat_dem', None):
        dem_grid = np.zeros_like(land_cover_grid)
    else:
        print("Creating Digital Elevation Model (DEM) grid")
        dem_grid = create_dem_grid_from_gdf_polygon(terrain_gdf, meshsize, rectangle_vertices)
        grid_vis = kwargs.get("gridvis", True)
        if grid_vis:
            visualize_numerical_grid(np.flipud(dem_grid), meshsize, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')

    min_canopy_height = kwargs.get("min_canopy_height")
    if min_canopy_height is not None:
        canopy_height_grid[canopy_height_grid < kwargs["min_canopy_height"]] = 0
        canopy_bottom_height_grid[canopy_height_grid == 0] = 0

    remove_perimeter_object = kwargs.get("remove_perimeter_object")
    if (remove_perimeter_object is not None) and (remove_perimeter_object > 0):
        print("apply perimeter removal")
        w_peri = int(remove_perimeter_object * building_height_grid.shape[0] + 0.5)
        h_peri = int(remove_perimeter_object * building_height_grid.shape[1] + 0.5)

        canopy_height_grid[:w_peri, :] = canopy_height_grid[-w_peri:, :] = canopy_height_grid[:, :h_peri] = canopy_height_grid[:, -h_peri:] = 0
        canopy_bottom_height_grid[:w_peri, :] = canopy_bottom_height_grid[-w_peri:, :] = canopy_bottom_height_grid[:, :h_peri] = canopy_bottom_height_grid[:, -h_peri:] = 0

        ids1 = np.unique(building_id_grid[:w_peri, :][building_id_grid[:w_peri, :] > 0])
        ids2 = np.unique(building_id_grid[-w_peri:, :][building_id_grid[-w_peri:, :] > 0])
        ids3 = np.unique(building_id_grid[:, :h_peri][building_id_grid[:, :h_peri] > 0])
        ids4 = np.unique(building_id_grid[:, -h_peri:][building_id_grid[:, -h_peri:] > 0])
        remove_ids = np.concatenate((ids1, ids2, ids3, ids4))

        for remove_id in remove_ids:
            positions = np.where(building_id_grid == remove_id)
            building_height_grid[positions] = 0
            building_min_height_grid[positions] = [[] for _ in range(len(building_min_height_grid[positions]))]

        grid_vis = kwargs.get("gridvis", True)
        if grid_vis:
            building_height_grid_nan = building_height_grid.copy()
            building_height_grid_nan[building_height_grid_nan == 0] = np.nan
            visualize_numerical_grid(
                np.flipud(building_height_grid_nan),
                meshsize,
                "building height (m)",
                cmap='viridis',
                label='Value'
            )
            canopy_height_grid_nan = canopy_height_grid.copy()
            canopy_height_grid_nan[canopy_height_grid_nan == 0] = np.nan
            visualize_numerical_grid(
                np.flipud(canopy_height_grid_nan),
                meshsize,
                "Tree canopy height (m)",
                cmap='Greens',
                label='Tree canopy height (m)'
            )

    from .voxelizer import Voxelizer
    voxelizer = Voxelizer(
        voxel_size=meshsize,
        land_cover_source=land_cover_source,
        trunk_height_ratio=kwargs.get("trunk_height_ratio"),
    )
    voxcity_grid = voxelizer.generate_combined(
        building_height_grid_ori=building_height_grid,
        building_min_height_grid_ori=building_min_height_grid,
        building_id_grid_ori=building_id_grid,
        land_cover_grid_ori=land_cover_grid,
        dem_grid_ori=dem_grid,
        tree_grid_ori=canopy_height_grid,
        canopy_bottom_height_grid_ori=locals().get("canopy_bottom_height_grid"),
    )

    from .pipeline import VoxCityPipeline as _Pipeline
    pipeline = _Pipeline(meshsize=meshsize, rectangle_vertices=rectangle_vertices)
    city = pipeline.assemble_voxcity(
        voxcity_grid=voxcity_grid,
        building_height_grid=building_height_grid,
        building_min_height_grid=building_min_height_grid,
        building_id_grid=building_id_grid,
        land_cover_grid=land_cover_grid,
        dem_grid=dem_grid,
        canopy_height_top=canopy_height_grid,
        canopy_height_bottom=locals().get("canopy_bottom_height_grid"),
        extras={"building_gdf": building_gdf},
    )

    # Backwards compatible save flag: prefer correct key, fallback to legacy misspelling
    _save_flag = kwargs.get("save_voxcity_data", kwargs.get("save_voxctiy_data", True))
    if _save_flag:
        save_path = kwargs.get("save_data_path", f"{output_dir}/voxcity_data.pkl")
        save_voxcity_data(
            save_path,
            city.voxels.classes,
            city.buildings.heights,
            city.buildings.min_heights,
            city.buildings.ids,
            city.extras.get("canopy_top"),
            city.land_cover.classes,
            city.dem.elevation,
            building_gdf,
            meshsize,
            rectangle_vertices,
        )

    return city


