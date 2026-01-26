import os
import numpy as np

from ..models import PipelineConfig
from .pipeline import VoxCityPipeline
from .grids import get_land_cover_grid
from .io import save_voxcity

from ..downloader.citygml import download_and_extract_zip
from ..geoprocessor.citygml import load_lod1_citygml, LOD1CityGMLParser
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

_SOURCE_URLS = {
    # General
    'OpenStreetMap': 'https://www.openstreetmap.org',
    'Local file': None,
    'None': None,
    'Flat': None,
    # Buildings
    'Microsoft Building Footprints': 'https://github.com/microsoft/GlobalMLBuildingFootprints',
    'Open Building 2.5D Temporal': 'https://sites.research.google/gr/open-buildings/temporal/',
    'EUBUCCO v0.1': 'https://eubucco.com/',
    'Overture': 'https://overturemaps.org/',
    'GBA': 'https://gee-community-catalog.org/projects/gba/',
    'Global Building Atlas': 'https://gee-community-catalog.org/projects/gba/',
    'England 1m DSM - DTM': 'https://developers.google.com/earth-engine/datasets/catalog/UK_EA_ENGLAND_1M_TERRAIN_2022',
    'Netherlands 0.5m DSM - DTM': 'https://developers.google.com/earth-engine/datasets/catalog/AHN_AHN4',
    # Land cover
    'OpenEarthMapJapan': 'https://www.open-earth-map.org/demo/Japan/leaflet.html',
    'Urbanwatch': 'https://gee-community-catalog.org/projects/urban-watch/',
    'ESA WorldCover': 'https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v200',
    'ESRI 10m Annual Land Cover': 'https://gee-community-catalog.org/projects/S2TSLULC/',
    'Dynamic World V1': 'https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_DYNAMICWORLD_V1',
    # Canopy height
    'High Resolution 1m Global Canopy Height Maps': 'https://gee-community-catalog.org/projects/meta_trees/',
    'ETH Global Sentinel-2 10m Canopy Height (2020)': 'https://gee-community-catalog.org/projects/canopy/',
    'Static': None,
    # Note: 'OpenStreetMap' for canopy uses the same URL as above (already defined)
    # DEM
    'USGS 3DEP 1m': 'https://developers.google.com/earth-engine/datasets/catalog/USGS_3DEP_1m',
    'England 1m DTM': 'https://developers.google.com/earth-engine/datasets/catalog/UK_EA_ENGLAND_1M_TERRAIN_2022',
    'DEM France 1m': 'https://developers.google.com/earth-engine/datasets/catalog/IGN_RGE_ALTI_1M_2_0',
    'DEM France 5m': 'https://gee-community-catalog.org/projects/france5m/',
    'AUSTRALIA 5M DEM': 'https://developers.google.com/earth-engine/datasets/catalog/AU_GA_AUSTRALIA_5M_DEM',
    'Netherlands 0.5m DTM': 'https://developers.google.com/earth-engine/datasets/catalog/AHN_AHN4',
    'FABDEM': 'https://gee-community-catalog.org/projects/fabdem/',
    'DeltaDTM': 'https://gee-community-catalog.org/projects/delta_dtm/',
}

def _url_for_source(name):
    try:
        return _SOURCE_URLS.get(name)
    except Exception:
        return None

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

    # Report detected country (best-effort)
    try:
        _logger.info(
            "Detected country for ROI center (%.4f, %.4f): %s",
            center_lon,
            center_lat,
            country or "Unknown",
        )
    except Exception:
        pass

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

    # Broad regions for OB 2.5D Temporal (prefer country membership; fallback to bbox if unknown)
    africa_countries = {
        'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde',
        'Cameroon', 'Central African Republic', 'Chad', 'Comoros', 'Congo',
        'Republic of the Congo', 'Democratic Republic of the Congo', 'Congo (DRC)',
        'DR Congo', 'Cote dIvoire', "Côte d’Ivoire", 'Ivory Coast', 'Djibouti', 'Egypt',
        'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia', 'Ghana',
        'Guinea', 'Guinea-Bissau', 'Kenya', 'Lesotho', 'Liberia', 'Libya', 'Madagascar',
        'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia',
        'Niger', 'Nigeria', 'Rwanda', 'Sao Tome and Principe', 'Senegal', 'Seychelles',
        'Sierra Leone', 'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo',
        'Tunisia', 'Uganda', 'Zambia', 'Zimbabwe', 'Western Sahara'
    }
    south_asia_countries = {
        'Afghanistan', 'Bangladesh', 'Bhutan', 'India', 'Maldives', 'Nepal', 'Pakistan', 'Sri Lanka'
    }
    se_asia_countries = {
        'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Lao PDR', 'Malaysia', 'Myanmar',
        'Philippines', 'Singapore', 'Thailand', 'Timor-Leste', 'Vietnam', 'Viet Nam'
    }
    latam_carib_countries = {
        # Latin America (Mexico, Central, South America) + Caribbean
        'Mexico',
        'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Nicaragua', 'Panama',
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana',
        'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela',
        'Antigua and Barbuda', 'Bahamas', 'Barbados', 'Cuba', 'Dominica', 'Dominican Republic',
        'Grenada', 'Haiti', 'Jamaica', 'Saint Kitts and Nevis', 'Saint Lucia',
        'Saint Vincent and the Grenadines', 'Trinidad and Tobago',
    }

    # Normalize some common aliases for matching
    _alias = {
        'United States of America': 'United States',
        'Czech Republic': 'Czechia',
        'Viet Nam': 'Vietnam',
        'Lao PDR': 'Laos',
        'Ivory Coast': "Côte d’Ivoire",
        'Congo, Democratic Republic of the': 'Democratic Republic of the Congo',
        'Congo, Republic of the': 'Republic of the Congo',
    }
    country_norm = _alias.get(country, country) if country else None

    in_africa = (country_norm in africa_countries) if country_norm else (-25 <= center_lon <= 80 and -55 <= center_lat <= 45)
    in_south_asia = (country_norm in south_asia_countries) if country_norm else (50 <= center_lon <= 100 and 0 <= center_lat <= 35)
    in_se_asia = (country_norm in se_asia_countries) if country_norm else (90 <= center_lon <= 150 and -10 <= center_lat <= 25)
    in_latam_carib = (country_norm in latam_carib_countries) if country_norm else (-110 <= center_lon <= -30 and -60 <= center_lat <= 30)

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


def get_voxcity(rectangle_vertices, meshsize, building_source=None, land_cover_source=None, canopy_height_source=None, dem_source=None, building_complementary_source=None, building_gdf=None, terrain_gdf=None, **kwargs):
    """
    Generate a VoxCity model with automatic or custom data source selection.
    
    This function supports both auto mode and custom mode:
    - Auto mode: When sources are not specified (None), they are automatically selected based on location
    - Custom mode: When sources are explicitly specified, they are used as-is
    - Hybrid mode: Specify some sources and auto-select others
    
    Args:
        rectangle_vertices: List of (lon, lat) tuples defining the area of interest
        meshsize: Grid resolution in meters (required)
        building_source: Building base source (default: auto-selected based on location)
        land_cover_source: Land cover source (default: auto-selected based on location)
        canopy_height_source: Canopy height source (default: auto-selected based on location)
        dem_source: Digital elevation model source (default: auto-selected based on location)
        building_complementary_source: Building complementary source (default: auto-selected based on location)
        building_gdf: Optional pre-loaded building GeoDataFrame
        terrain_gdf: Optional pre-loaded terrain GeoDataFrame
        **kwargs: Additional options for building, land cover, canopy, DEM, visualization, and I/O.
                  Performance options include:
                  - parallel_download: bool, if True downloads run concurrently (default: False)
                  I/O options include:
                  - output_dir: Directory for intermediate/downloaded data (default: "output")
                  - save_path: Full file path to save the VoxCity object (overrides output_dir default)
                  - save_voxcity_data / save_voxctiy_data: bool flag to enable saving (default: True)
    
    Returns:
        VoxCity object containing the generated 3D city model
    """
    
    # Check if building_complementary_source was provided via kwargs (for backward compatibility)
    if building_complementary_source is None and 'building_complementary_source' in kwargs:
        building_complementary_source = kwargs.pop('building_complementary_source')
    
    # Determine if we need to auto-select any sources
    sources_to_select = []
    if building_source is None:
        sources_to_select.append('building_source')
    if land_cover_source is None:
        sources_to_select.append('land_cover_source')
    if canopy_height_source is None:
        sources_to_select.append('canopy_height_source')
    if dem_source is None:
        sources_to_select.append('dem_source')
    if building_complementary_source is None:
        sources_to_select.append('building_complementary_source')
    
    # Auto-select missing sources if needed
    if sources_to_select:
        _logger.info("Auto-selecting data sources for: %s", ", ".join(sources_to_select))
        auto_sources = auto_select_data_sources(rectangle_vertices)
        
        # Check Earth Engine availability for auto-selected sources
        ee_available = True
        try:
            from ..downloader.gee import initialize_earth_engine
            initialize_earth_engine()
        except Exception:
            ee_available = False
        
        if not ee_available:
            # Downgrade EE-dependent sources to non-GEE alternatives
            # Land cover: fallback to OpenStreetMap (unless already non-GEE)
            if auto_sources['land_cover_source'] not in ('OpenStreetMap', 'OpenEarthMapJapan'):
                auto_sources['land_cover_source'] = 'OpenStreetMap'
            
            # Canopy height: region-dependent fallback
            # - Japan: 'Static' (OpenEarthMapJapan land cover provides good tree coverage)
            # - Other regions: 'OpenStreetMap' (provides actual tree locations and forest polygons)
            is_japan_area = (auto_sources['land_cover_source'] == 'OpenEarthMapJapan')
            if is_japan_area:
                auto_sources['canopy_height_source'] = 'Static'
            else:
                auto_sources['canopy_height_source'] = 'OpenStreetMap'
            
            # DEM: fallback to Flat (no elevation data without GEE)
            auto_sources['dem_source'] = 'Flat'
            
            # Building complementary sources that require GEE
            ee_dependent_comp = {
                'Open Building 2.5D Temporal',
                'England 1m DSM - DTM',
                'Netherlands 0.5m DSM - DTM',
            }
            if auto_sources.get('building_complementary_source') in ee_dependent_comp:
                auto_sources['building_complementary_source'] = 'Microsoft Building Footprints'
        
        # Apply auto-selected sources only where not specified
        if building_source is None:
            building_source = auto_sources['building_source']
        if land_cover_source is None:
            land_cover_source = auto_sources['land_cover_source']
        if canopy_height_source is None:
            canopy_height_source = auto_sources['canopy_height_source']
        if dem_source is None:
            dem_source = auto_sources['dem_source']
        if building_complementary_source is None:
            building_complementary_source = auto_sources.get('building_complementary_source', 'None')
        
        # Auto-set complement height if not provided
        if 'building_complement_height' not in kwargs:
            kwargs['building_complement_height'] = 10
    
    # Ensure building_complementary_source is passed through kwargs
    if building_complementary_source is not None:
        kwargs['building_complementary_source'] = building_complementary_source
    
    # Default DEM interpolation to True unless explicitly provided
    if 'dem_interpolation' not in kwargs:
        kwargs['dem_interpolation'] = True
    
    # Ensure default complement height even if all sources are user-specified
    if 'building_complement_height' not in kwargs:
        kwargs['building_complement_height'] = 10
    
    # Log selected data sources (always)
    try:
        _logger.info("Selected data sources:")
        b_base_url = _url_for_source(building_source)
        _logger.info("- Buildings(base)=%s%s", building_source, f" | {b_base_url}" if b_base_url else "")
        b_comp_url = _url_for_source(building_complementary_source)
        _logger.info("- Buildings(comp)=%s%s", building_complementary_source, f" | {b_comp_url}" if b_comp_url else "")
        lc_url = _url_for_source(land_cover_source)
        _logger.info("- LandCover=%s%s", land_cover_source, f" | {lc_url}" if lc_url else "")
        canopy_url = _url_for_source(canopy_height_source)
        _logger.info("- Canopy=%s%s", canopy_height_source, f" | {canopy_url}" if canopy_url else "")
        dem_url = _url_for_source(dem_source)
        _logger.info("- DEM=%s%s", dem_source, f" | {dem_url}" if dem_url else "")
        _logger.info("- ComplementHeight=%s", kwargs.get('building_complement_height'))
    except Exception:
        pass
    
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
        "dem_path",
    }
    visualize_keys = {"gridvis", "mapvis"}
    io_keys = {"save_voxcity_data", "save_voxctiy_data", "save_data_path", "save_path"}

    land_cover_options = {k: v for k, v in kwargs.items() if k in land_cover_keys}
    building_options = {k: v for k, v in kwargs.items() if k in building_keys}
    canopy_options = {k: v for k, v in kwargs.items() if k in canopy_keys}
    dem_options = {k: v for k, v in kwargs.items() if k in dem_keys}
    # Auto-set flat DEM when dem_source is None/empty and user didn't specify
    if (dem_source in (None, "", "None")) and ("flat_dem" not in dem_options):
        dem_options["flat_dem"] = True
    visualize_options = {k: v for k, v in kwargs.items() if k in visualize_keys}
    io_options = {k: v for k, v in kwargs.items() if k in io_keys}

    # Parallel download mode
    parallel_download = kwargs.get("parallel_download", False)

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
        parallel_download=parallel_download,
        land_cover_options=land_cover_options,
        building_options=building_options,
        canopy_options=canopy_options,
        dem_options=dem_options,
        io_options=io_options,
        visualize_options=visualize_options,
    )
    city = VoxCityPipeline(meshsize=cfg.meshsize, rectangle_vertices=cfg.rectangle_vertices).run(cfg, building_gdf=building_gdf, terrain_gdf=terrain_gdf, **{k: v for k, v in kwargs.items() if k != 'output_dir'})

    # Optional shape normalization (pad/crop) to a target (x, y, z)
    target_voxel_shape = kwargs.get("target_voxel_shape", None)
    if target_voxel_shape is not None:
        try:
            from ..utils.shape import normalize_voxcity_shape  # late import to avoid cycles
            align_xy = kwargs.get("pad_align_xy", "center")
            allow_crop_xy = bool(kwargs.get("allow_crop_xy", True))
            allow_crop_z = bool(kwargs.get("allow_crop_z", False))
            pad_values = kwargs.get("pad_values", None)
            city = normalize_voxcity_shape(
                city,
                tuple(target_voxel_shape),
                align_xy=align_xy,
                pad_values=pad_values,
                allow_crop_xy=allow_crop_xy,
                allow_crop_z=allow_crop_z,
            )
            try:
                _logger.info("Applied target voxel shape %s -> final voxel shape %s", tuple(target_voxel_shape), tuple(city.voxels.classes.shape))
            except Exception:
                pass
        except Exception as e:
            try:
                _logger.warning("Shape normalization skipped due to error: %s", str(e))
            except Exception:
                pass

    # Backwards compatible save flag: prefer correct key, fallback to legacy misspelling
    _save_flag = io_options.get("save_voxcity_data", kwargs.get("save_voxcity_data", kwargs.get("save_voxctiy_data", True)))
    if _save_flag:
        # Prefer explicit save_path if provided; fall back to legacy save_data_path; else default
        save_path = (
            io_options.get("save_path")
            or kwargs.get("save_path")
            or io_options.get("save_data_path")
            or kwargs.get("save_data_path")
            or f"{output_dir}/voxcity.pkl"
        )
        save_voxcity(save_path, city)

    # Attach selected sources (final resolved) to extras for downstream consumers
    try:
        city.extras['selected_sources'] = {
            'building_source': building_source,
            'building_complementary_source': building_complementary_source or 'None',
            'land_cover_source': land_cover_source,
            'canopy_height_source': canopy_height_source,
            'dem_source': dem_source,
            'building_complement_height': kwargs.get('building_complement_height'),
        }
    except Exception:
        pass

    return city


def get_voxcity_CityGML(rectangle_vertices, land_cover_source, canopy_height_source, meshsize, url_citygml=None, citygml_path=None, **kwargs):
    """
    Generate a VoxCity model from CityGML data.
    
    This function supports both:
    - **PLATEAU format**: Japanese CityGML with udx/ folder structure (lat/lon coordinates)
    - **Generic format**: European/German CityGML with GML files (UTM coordinates)
    
    The format is auto-detected based on directory structure. You can also specify
    it explicitly using the `citygml_format` parameter.
    
    Modes for building voxelization:
    
    1. **Building/Bridge/Furniture voxelization**:
       - LOD1 mode: Footprint-based building heights from GeoDataFrames
       - LOD2 mode: Direct 3D voxelization from LOD2 triangulated geometry
    
    2. **Tree voxelization**:
       - LOD1 mode: From CityGML vegetation GeoDataFrames
       - LOD2 mode: From CityGML LOD2 vegetation geometry (when `include_lod2_vegetation=True`)
       - External sources: Canopy height from GEE or other sources
    
    3. **Terrain and land cover voxelization**:
       - CityGML terrain: From DEM triangulated geometry
       - External DEM: From GEE-based sources or local files
       - Local DTM: From GeoTIFF file (for generic CityGML)
       - Land cover is placed at terrain surface
       - Terrain is flattened under building footprints
    
    Args:
        rectangle_vertices: List of (lon, lat) tuples defining the area of interest
        land_cover_source: Land cover data source
        canopy_height_source: Canopy height data source
        meshsize: Grid resolution in meters
        url_citygml: URL to download CityGML data from
        citygml_path: Path to local CityGML directory
        **kwargs: Additional options including:
            - citygml_format: str, force format - 'plateau', 'generic', or None for auto
            - lod: str, LOD mode - 'lod1', 'lod2', or None for auto-detection
              (default: None, auto-detects based on available data)
            - include_bridges: bool, include bridge geometry in LOD2 mode (default: True)
            - include_city_furniture: bool, include city furniture in LOD2 mode (default: False)
            - include_lod2_vegetation: bool, use LOD2 vegetation geometry (default: True)
            - dem_source: str, external DEM source (default: None, uses CityGML terrain)
            - dem_path: str, path to local DEM GeoTIFF (required when dem_source='Local file')
            - dtm_path: str, path to local DTM GeoTIFF (for generic CityGML format)
            - output_dir: Directory for intermediate data (default: "output")
            - ssl_verify, ca_bundle, timeout: Network options for URL downloads
            - timing: bool, if True prints timing information for each step (default: False)
    
    Returns:
        VoxCity object containing the generated 3D city model
        
    Example for PLATEAU data::
    
        city = get_voxcity_CityGML(
            rectangle_vertices=[(139.75, 35.68), (139.76, 35.68), (139.76, 35.69), (139.75, 35.69)],
            land_cover_source='OpenEarthMapJapan',
            canopy_height_source='Static',
            meshsize=1.0,
            citygml_path='path/to/plateau_data'
        )
    
    Example for generic (European) CityGML data::
    
        city = get_voxcity_CityGML(
            rectangle_vertices=[(11.5, 48.0), (11.6, 48.0), (11.6, 48.1), (11.5, 48.1)],
            land_cover_source='OpenStreetMap',
            canopy_height_source='Static',
            meshsize=1.0,
            citygml_path='path/to/gml_folder',
            dtm_path='path/to/dtm.tif'  # Optional, auto-detected if named *dgm*.tif/*dtm*.tif
        )
    """
    import time as _time
    from .pipeline import VoxCityPipeline as _Pipeline
    from .voxelizer import Voxelizer
    from ..geoprocessor.citygml import (
        detect_citygml_format,
        resolve_citygml_path,
        voxelize_buildings_citygml_optimized,
        voxelize_trees_citygml_optimized,
        voxelize_terrain_citygml_optimized,
        apply_citygml_post_processing,
        merge_lod2_voxels,
    )
    
    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    output_dir = kwargs.pop("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    ssl_verify = kwargs.pop('ssl_verify', kwargs.pop('verify', True))
    ca_bundle = kwargs.pop('ca_bundle', None)
    timeout = kwargs.pop('timeout', 60)
    
    # CityGML format: 'plateau', 'generic', or None (auto-detect)
    citygml_format = kwargs.get('citygml_format', None)
    
    # Handle LOD mode: 'lod1', 'lod2', or None (auto-detect)
    lod_mode = kwargs.pop('lod', None)
    # Support legacy use_lod2 parameter for backward compatibility
    if lod_mode is None and 'use_lod2' in kwargs:
        use_lod2_legacy = kwargs.pop('use_lod2', False)
        lod_mode = 'lod2' if use_lod2_legacy else 'lod1'
    
    include_bridges = kwargs.pop('include_bridges', True)
    include_city_furniture = kwargs.pop('include_city_furniture', False)
    include_lod2_vegetation = kwargs.pop('include_lod2_vegetation', True)
    
    dem_source = kwargs.get("dem_source", None)
    grid_vis = kwargs.get("gridvis", True)
    trunk_height_ratio = kwargs.get("trunk_height_ratio", 11.76 / 19.98)
    show_timing = kwargs.pop("timing", False)
    
    # Timing helper
    _timings = {}
    def _record_time(name, start):
        elapsed = _time.perf_counter() - start
        _timings[name] = elapsed
        if show_timing:
            print(f"  [TIMING] {name}: {elapsed:.2f}s")
    
    total_start = _time.perf_counter()
    
    # =========================================================================
    # RESOLVE CITYGML PATH
    # =========================================================================
    t0 = _time.perf_counter()
    citygml_path_resolved = resolve_citygml_path(
        url_citygml, citygml_path, output_dir, ssl_verify, ca_bundle, timeout
    )
    _record_time("resolve_citygml_path", t0)
    
    # =========================================================================
    # STEP 1: BUILDING, BRIDGE, AND FURNITURE VOXELIZATION
    # Options: LOD1 (footprint-based) or LOD2 (triangulated geometry)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Building/Bridge/Furniture Voxelization")
    print("=" * 60)
    
    t0 = _time.perf_counter()
    building_result = voxelize_buildings_citygml_optimized(
        citygml_path_resolved=citygml_path_resolved,
        rectangle_vertices=rectangle_vertices,
        meshsize=meshsize,
        lod=lod_mode,
        include_bridges=include_bridges,
        include_city_furniture=include_city_furniture,
        grid_vis=grid_vis,
        citygml_format=citygml_format,
        **kwargs
    )
    _record_time("voxelize_buildings", t0)
    
    lod_mode = building_result['lod']  # May be updated based on available data
    use_lod2 = (lod_mode == 'lod2')
    building_gdf = building_result['building_gdf']
    building_height_grid = building_result['building_height_grid']
    building_min_height_grid = building_result['building_min_height_grid']
    building_id_grid = building_result['building_id_grid']
    lod2_voxelizer = building_result.get('lod2_voxelizer')
    
    # =========================================================================
    # STEP 2: TREE VOXELIZATION
    # Options: LOD1 (CityGML vegetation GDF), LOD2 (triangulated), or external sources
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Tree Voxelization")
    print("=" * 60)
    
    t0 = _time.perf_counter()
    tree_result = voxelize_trees_citygml_optimized(
        citygml_path_resolved=citygml_path_resolved,
        rectangle_vertices=rectangle_vertices,
        meshsize=meshsize,
        land_cover_source=land_cover_source,
        canopy_height_source=canopy_height_source,
        use_lod2=use_lod2,
        include_lod2_vegetation=include_lod2_vegetation,
        trunk_height_ratio=trunk_height_ratio,
        output_dir=output_dir,
        **kwargs
    )
    _record_time("voxelize_trees", t0)
    
    canopy_height_grid = tree_result['canopy_height_grid']
    canopy_bottom_height_grid = tree_result['canopy_bottom_height_grid']
    
    # =========================================================================
    # STEP 3: TERRAIN AND LAND COVER VOXELIZATION
    # Options: CityGML terrain or external DEM sources
    # Land cover is placed at terrain surface, terrain flattened under buildings
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Terrain and Land Cover Voxelization")
    print("=" * 60)
    
    t0 = _time.perf_counter()
    terrain_result = voxelize_terrain_citygml_optimized(
        citygml_path_resolved=citygml_path_resolved,
        rectangle_vertices=rectangle_vertices,
        meshsize=meshsize,
        land_cover_source=land_cover_source,
        dem_source=dem_source,
        building_id_grid=building_id_grid,
        grid_vis=grid_vis,
        output_dir=output_dir,
        citygml_format=citygml_format,
        dtm_path=kwargs.get('dtm_path'),
        dem_interpolation=kwargs.get('dem_interpolation', True),
        dem_path=kwargs.get('dem_path'),
    )
    _record_time("voxelize_terrain", t0)
    
    dem_grid = terrain_result['dem_grid']
    land_cover_grid = terrain_result['land_cover_grid']
    
    # =========================================================================
    # STEP 4: APPLY POST-PROCESSING (perimeter removal, min canopy height)
    # =========================================================================
    t0 = _time.perf_counter()
    apply_citygml_post_processing(
        building_height_grid=building_height_grid,
        building_min_height_grid=building_min_height_grid,
        building_id_grid=building_id_grid,
        canopy_height_grid=canopy_height_grid,
        canopy_bottom_height_grid=canopy_bottom_height_grid,
        meshsize=meshsize,
        grid_vis=grid_vis,
        **kwargs
    )
    _record_time("apply_post_processing", t0)
    
    # =========================================================================
    # STEP 5: MERGE ALL VOXELIZATIONS
    # Combine terrain/land cover + trees + buildings/bridges/furniture
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 4: Merging All Voxel Layers")
    print("=" * 60)
    
    t0 = _time.perf_counter()
    voxelizer = Voxelizer(
        voxel_size=meshsize,
        land_cover_source=land_cover_source,
        trunk_height_ratio=trunk_height_ratio,
    )
    
    if use_lod2 and lod2_voxelizer is not None:
        voxcity_grid = merge_lod2_voxels(
            voxelizer=voxelizer,
            lod2_voxelizer=lod2_voxelizer,
            building_height_grid=building_height_grid,
            building_min_height_grid=building_min_height_grid,
            building_id_grid=building_id_grid,
            land_cover_grid=land_cover_grid,
            dem_grid=dem_grid,
            canopy_height_grid=canopy_height_grid,
            canopy_bottom_height_grid=canopy_bottom_height_grid,
            rectangle_vertices=rectangle_vertices,
            meshsize=meshsize,
            include_bridges=include_bridges,
            include_city_furniture=include_city_furniture,
        )
    else:
        # Standard mode: use footprint-based building voxelization
        print("  Using standard footprint-based voxelization")
        voxcity_grid = voxelizer.generate_combined(
            building_height_grid_ori=building_height_grid,
            building_min_height_grid_ori=building_min_height_grid,
            building_id_grid_ori=building_id_grid,
            land_cover_grid_ori=land_cover_grid,
            dem_grid_ori=dem_grid,
            tree_grid_ori=canopy_height_grid,
            canopy_bottom_height_grid_ori=canopy_bottom_height_grid,
        )
    _record_time("merge_voxels", t0)
    
    # =========================================================================
    # ASSEMBLE AND SAVE
    # =========================================================================
    t0 = _time.perf_counter()
    pipeline = _Pipeline(meshsize=meshsize, rectangle_vertices=rectangle_vertices)
    city = pipeline.assemble_voxcity(
        voxcity_grid=voxcity_grid,
        building_height_grid=building_height_grid,
        building_min_height_grid=building_min_height_grid,
        building_id_grid=building_id_grid,
        land_cover_grid=land_cover_grid,
        dem_grid=dem_grid,
        canopy_height_top=canopy_height_grid,
        canopy_height_bottom=canopy_bottom_height_grid,
        extras={
            "building_gdf": building_gdf,
            "lod2_mode": use_lod2 and lod2_voxelizer is not None,
            "include_bridges": include_bridges if use_lod2 else False,
            "include_city_furniture": include_city_furniture if use_lod2 else False,
            "num_lod2_buildings": len(lod2_voxelizer.parser.buildings) if (lod2_voxelizer and lod2_voxelizer.parser) else 0,
        },
    )

    _save_flag = kwargs.get("save_voxcity_data", kwargs.get("save_voxctiy_data", True))
    if _save_flag:
        save_path = kwargs.get("save_path") or kwargs.get("save_data_path") or f"{output_dir}/voxcity.pkl"
        save_voxcity(save_path, city)
    _record_time("assemble_and_save", t0)

    total_elapsed = _time.perf_counter() - total_start
    _timings['TOTAL'] = total_elapsed

    if use_lod2 and lod2_voxelizer is not None:
        print("\n" + "=" * 60)
        print("LOD2 VoxCity generation complete!")
        print("=" * 60)

    # Print timing summary if requested
    if show_timing:
        print("\n" + "=" * 60)
        print("TIMING SUMMARY")
        print("=" * 60)
        for name, elapsed in sorted(_timings.items(), key=lambda x: -x[1]):
            pct = (elapsed / total_elapsed * 100) if total_elapsed > 0 else 0
            print(f"  {name}: {elapsed:.2f}s ({pct:.1f}%)")

    return city