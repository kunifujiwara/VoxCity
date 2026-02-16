"""
Module for interacting with Google Earth Engine API and downloading geospatial data.

This module provides functionality to initialize Earth Engine, create regions of interest,
download various types of satellite imagery and terrain data, and save them as GeoTIFF files.
It supports multiple data sources including DEMs, land cover maps, and building footprints.

The module offers the following main functionalities:
1. Earth Engine initialization and region of interest (ROI) management
2. Digital Elevation Model (DEM) data access from multiple sources
3. Land cover data retrieval from ESA WorldCover, Dynamic World, and ESRI
4. Building footprint and height data extraction
5. GeoTIFF export with customizable parameters

Dependencies:
    - ee: Google Earth Engine Python API
    - geemap: Python package for interactive mapping with Earth Engine

Note: Most functions require Earth Engine authentication to be set up beforehand.
"""

# Earth Engine and geospatial imports
import ee
import geemap
import json
import os

# Local imports
# from ..geo.utils import convert_format_lat_lon

def _load_service_account_credentials():
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path or not os.path.exists(credentials_path):
        return None
    try:
        with open(credentials_path, "r", encoding="utf-8") as f:
            key_data = json.load(f)
        client_email = key_data.get("client_email")
        if not client_email:
            return None
        return ee.ServiceAccountCredentials(email=client_email, key_file=credentials_path)
    except Exception:
        return None


def initialize_earth_engine(**initialize_kwargs):
    """Initialize the Earth Engine API if not already initialized.

    Uses a public-behavior check to determine whether Earth Engine is already
    initialized by attempting to access asset roots. If that call fails, it will
    initialize Earth Engine using the provided keyword arguments.

    Arguments are passed through to ``ee.Initialize`` to support contexts such as
    specifying a ``project`` or service account credentials.
    """
    try:
        # If this succeeds, EE is already initialized
        ee.data.getAssetRoots()
        return
    except Exception:
        pass

    # Prefer service account credentials if provided via env
    credentials = _load_service_account_credentials()
    if credentials is not None:
        project = (
            initialize_kwargs.pop("project", None)
            or os.environ.get("GEE_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        try:
            if project:
                ee.Initialize(credentials, project=project)
            else:
                ee.Initialize(credentials)
            return
        except Exception:
            pass

    ee.Initialize(**initialize_kwargs)

def get_roi(input_coords):
    """Create an Earth Engine region of interest polygon from coordinates.
    
    Args:
        input_coords: List of coordinate pairs defining the polygon vertices in (lon, lat) format.
                     The coordinates should form a valid polygon (non-self-intersecting).
        
    Returns:
        ee.Geometry.Polygon: Earth Engine polygon geometry representing the ROI
        
    Note:
        The function automatically closes the polygon by connecting the last vertex
        to the first vertex if they are not the same.
    """
    coords = input_coords.copy()
    coords.append(input_coords[0])
    return ee.Geometry.Polygon(coords)

def get_center_point(roi):
    """Get the centroid coordinates of a region of interest.
    
    Args:
        roi: Earth Engine geometry object representing the region of interest
        
    Returns:
        tuple: (longitude, latitude) coordinates of the centroid
        
    Note:
        The centroid is calculated using Earth Engine's geometric centroid algorithm,
        which may not always fall within the geometry for complex shapes.
    """
    center_point = roi.centroid()
    center_coords = center_point.coordinates().getInfo()
    return center_coords[0], center_coords[1]

def get_ee_image_collection(collection_name, roi):
    """Get the first image from an Earth Engine ImageCollection filtered by region.
    
    Args:
        collection_name: Name of the Earth Engine ImageCollection (e.g., 'LANDSAT/LC08/C02/T1_TOA')
        roi: Earth Engine geometry to filter by
        
    Returns:
        ee.Image: First image from collection clipped to ROI, with any masked pixels unmasked
        
    Note:
        The function sorts images by time (earliest first) and unmasks any masked pixels
        in the final image. This is useful for ensuring complete coverage of the ROI.
    """
    # Filter collection by bounds and get first image
    collection = ee.ImageCollection(collection_name).filterBounds(roi)
    return collection.sort('system:time_start').first().clip(roi).unmask()

def get_ee_image(collection_name, roi):
    """Get an Earth Engine Image clipped to a region.
    
    Args:
        collection_name: Name of the Earth Engine Image asset
        roi: Earth Engine geometry to clip to
        
    Returns:
        ee.Image: Image clipped to ROI with masked pixels unmasked to 0
        
    Note:
        Unlike get_ee_image_collection(), this function works with single image assets
        rather than image collections. It's useful for static datasets like DEMs.
        Masked pixels are replaced with 0 via unmask() so that downstream consumers
        receive numeric values instead of dataset-specific nodata sentinels (e.g. 255).
    """
    collection = ee.Image(collection_name)
    return collection.clip(roi).unmask()

def save_geotiff(image, filename, resolution=1, scale=None, region=None, crs=None):
    """Save an Earth Engine image as a GeoTIFF file.
    
    This function provides flexible options for exporting Earth Engine images to GeoTIFF format.
    It handles different export scenarios based on the provided parameters.
    
    Args:
        image: Earth Engine image to save
        filename: Output filename for the GeoTIFF
        resolution: Output resolution in degrees (default: 1), used when scale is not provided
        scale: Output scale in meters (overrides resolution if provided)
        region: Region to export (required if scale is provided)
        crs: Coordinate reference system (e.g., 'EPSG:4326')
    
    Note:
        - If scale and region are provided, uses ee_export_image()
        - Otherwise, uses ee_to_geotiff() with resolution parameter
        - The function automatically converts output to Cloud Optimized GeoTIFF (COG)
          format when using ee_to_geotiff()
    """
    # Handle different export scenarios based on provided parameters
    if scale and region:
        if crs:
            geemap.ee_export_image(image, filename=filename, scale=scale, region=region, file_per_band=False, crs=crs)
        else:
            geemap.ee_export_image(image, filename=filename, scale=scale, region=region, file_per_band=False)
    else:
        if crs:
            geemap.ee_to_geotiff(image, filename, resolution=resolution, to_cog=True, crs=crs)
        else:
            geemap.ee_to_geotiff(image, filename, resolution=resolution, to_cog=True)

def get_dem_image(roi_buffered, source):
    """Get a digital elevation model (DEM) image for a region.
    
    This function provides access to various global and regional Digital Elevation Model (DEM)
    datasets through Earth Engine. Each source has different coverage areas and resolutions.
    
    Args:
        roi_buffered: Earth Engine geometry with buffer - should be larger than the actual
                     area of interest to ensure smooth interpolation at edges
        source: DEM source, one of:
               - 'NASA': SRTM 30m global DEM
               - 'COPERNICUS': Copernicus 30m global DEM
               - 'DeltaDTM': Deltares global DTM
               - 'FABDEM': Forest And Buildings removed MERIT DEM
               - 'England 1m DTM': UK Environment Agency 1m terrain model
               - 'DEM France 5m': IGN RGE ALTI 5m France coverage
               - 'DEM France 1m': IGN RGE ALTI 1m France coverage
               - 'AUSTRALIA 5M DEM': Geoscience Australia 5m DEM
               - 'USGS 3DEP 1m': USGS 3D Elevation Program 1m DEM
               
    Returns:
        ee.Image: DEM image clipped to region
        
    Note:
        Some sources may have limited coverage or require special access permissions.
        The function will raise an error if the selected source is not available for
        the specified region.
    """
    # Handle different DEM sources
    if source == 'NASA':
        collection_name = 'USGS/SRTMGL1_003'
        dem = ee.Image(collection_name)
    elif source == 'COPERNICUS':
        collection_name = 'COPERNICUS/DEM/GLO30'
        collection = ee.ImageCollection(collection_name)
        # Get the most recent image and select the DEM band
        dem = collection.select('DEM').mosaic()
    elif source == 'DeltaDTM':
        collection_name = 'projects/sat-io/open-datasets/DELTARES/deltadtm_v1'
        elevation = ee.Image(collection_name).select('b1')
        dem = elevation.updateMask(elevation.neq(10))
    elif source == 'FABDEM':
        collection_name = "projects/sat-io/open-datasets/FABDEM"
        collection = ee.ImageCollection(collection_name)
        # Get the most recent image and select the DEM band
        dem = collection.select('b1').mosaic()
    elif source == 'England 1m DTM':
        collection_name = 'UK/EA/ENGLAND_1M_TERRAIN/2022'
        dem = ee.Image(collection_name).select('dtm')
    elif source == 'DEM France 5m':
        collection_name = "projects/sat-io/open-datasets/IGN_RGE_Alti_5m"
        dem = ee.Image(collection_name)
    elif source == 'DEM France 1m':
        collection_name = 'IGN/RGE_ALTI/1M/2_0/FXX'
        dem = ee.Image(collection_name).select('MNT')
    elif source == 'AUSTRALIA 5M DEM':
        collection_name = 'AU/GA/AUSTRALIA_5M_DEM'
        collection = ee.ImageCollection(collection_name)
        dem = collection.select('elevation').mosaic()
    elif source == 'Netherlands 0.5m DTM':
        collection_name = 'AHN/AHN4'
        collection = ee.ImageCollection(collection_name)
        dem = collection.select('dtm').mosaic()
    elif source == 'USGS 3DEP 1m':
        collection_name = 'USGS/3DEP/1m'
        dem = ee.ImageCollection(collection_name).mosaic()
    # Commented out sources that are not yet implemented
    # elif source == 'Canada High Resolution DTM':
    #     collection_name = "projects/sat-io/open-datasets/OPEN-CANADA/CAN_ELV/HRDEM_1M_DTM"
    #     collection = ee.ImageCollection(collection_name)
    #     dem = collection.mosaic() 

    # elif source == 'FABDEM':
    # If we reach here without assigning `dem`, the source is unsupported
    try:
        return dem.clip(roi_buffered)
    except UnboundLocalError:
        raise ValueError(f"Unsupported or unimplemented DEM source: {source}")

def save_geotiff_esa_land_cover(roi, geotiff_path):
    """Save ESA WorldCover land cover data as a colored GeoTIFF.
    
    Downloads and exports the ESA WorldCover 10m resolution global land cover map.
    The output is a colored GeoTIFF where each land cover class is represented by
    a unique color as defined in the color_map.
    
    Args:
        roi: Earth Engine geometry defining region of interest
        geotiff_path: Output path for GeoTIFF file
        
    Land cover classes and their corresponding colors:
        - Trees (10): Dark green
        - Shrubland (20): Orange
        - Grassland (30): Yellow
        - Cropland (40): Purple
        - Built-up (50): Red
        - Barren/sparse vegetation (60): Gray
        - Snow and ice (70): White
        - Open water (80): Blue
        - Herbaceous wetland (90): Teal
        - Mangroves (95): Light green
        - Moss and lichen (100): Beige
        
    Note:
        The output GeoTIFF is exported at 10m resolution, which is the native
        resolution of the ESA WorldCover dataset.
    """
    # Initialize Earth Engine
    initialize_earth_engine()

    # Load and clip the ESA WorldCover dataset
    esa = ee.ImageCollection("ESA/WorldCover/v200").first()
    esa_clipped = esa.clip(roi)

    # Define color mapping for different land cover classes
    color_map = {
        10: '006400',  # Trees
        20: 'ffbb22',  # Shrubland
        30: 'ffff4c',  # Grassland
        40: 'f096ff',  # Cropland
        50: 'fa0000',  # Built-up
        60: 'b4b4b4',  # Barren / sparse vegetation
        70: 'f0f0f0',  # Snow and ice
        80: '0064c8',  # Open water
        90: '0096a0',  # Herbaceous wetland
        95: '00cf75',  # Mangroves
        100: 'fae6a0'  # Moss and lichen
    }

    # Create ordered color palette
    colors = [color_map[i] for i in sorted(color_map.keys())]

    # Remap classes and apply color visualization
    esa_colored = esa_clipped.remap(
        list(color_map.keys()),
        list(range(len(color_map)))
    ).visualize(palette=colors, min=0, max=len(color_map)-1)

    # Export colored image
    geemap.ee_export_image(esa_colored, geotiff_path, scale=10, region=roi)

    print(f"Colored GeoTIFF saved to: {geotiff_path}")

def save_geotiff_dynamic_world_v1(roi, geotiff_path, date=None):
    """Save Dynamic World land cover data as a colored GeoTIFF.
    
    Downloads and exports Google's Dynamic World near real-time land cover classification.
    The data is available globally at 10m resolution from 2015 onwards, updated every 2-5 days.
    
    Args:
        roi: Earth Engine geometry defining region of interest
        geotiff_path: Output path for GeoTIFF file
        date: Optional date string (YYYY-MM-DD) to get data for specific time.
              If None, uses the most recent available image.
    
    Land cover classes and their colors:
        - water: Blue (#419bdf)
        - trees: Dark green (#397d49)
        - grass: Light green (#88b053)
        - flooded_vegetation: Purple (#7a87c6)
        - crops: Orange (#e49635)
        - shrub_and_scrub: Tan (#dfc35a)
        - built: Red (#c4281b)
        - bare: Gray (#a59b8f)
        - snow_and_ice: Light purple (#b39fe1)
        
    Note:
        If a specific date is provided but no image is available, the function
        will use the closest available date and print a message indicating the
        actual date used.
    """
    # Initialize Earth Engine
    initialize_earth_engine()

    # Load and filter Dynamic World dataset
    # Load the Dynamic World dataset and filter by ROI
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(roi)

    # Check if there are any images in the filtered collection
    count = dw.size().getInfo()
    if count == 0:
        print("No Dynamic World images found for the specified ROI.")
        return

    if date is None:
        # Get the latest available image
        dw_image = dw.sort('system:time_start', False).first()
        image_date = ee.Date(dw_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        print(f"No date specified. Using the latest available image from {image_date}.")
    else:
        # Convert the date string to an ee.Date object
        target_date = ee.Date(date)
        target_date_millis = target_date.millis()

        # Function to compute date difference and set as property
        def add_date_difference(image):
            image_date_millis = image.date().millis()
            diff = image_date_millis.subtract(target_date_millis).abs()
            return image.set('date_difference', diff)

        # Map over the collection to compute date differences
        dw_with_diff = dw.map(add_date_difference)

        # Sort the collection by date difference
        dw_sorted = dw_with_diff.sort('date_difference')

        # Get the first image (closest in time)
        dw_image = ee.Image(dw_sorted.first())
        image_date = ee.Date(dw_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        print(f"Using image closest to the specified date. Image date: {image_date}")

    # Clip the image to the ROI
    dw_clipped = dw_image.clip(roi)

    # Define class names and palette
    class_names = [
        'water',
        'trees',
        'grass',
        'flooded_vegetation',
        'crops',
        'shrub_and_scrub',
        'built',
        'bare',
        'snow_and_ice',
    ]

    color_palette = [
        '419bdf',  # water
        '397d49',  # trees
        '88b053',  # grass
        '7a87c6',  # flooded_vegetation
        'e49635',  # crops
        'dfc35a',  # shrub_and_scrub
        'c4281b',  # built
        'a59b8f',  # bare
        'b39fe1',  # snow_and_ice
    ]

    # Get the 'label' band
    label = dw_clipped.select('label')

    # Visualize the label band using the palette
    label_visualized = label.visualize(min=0, max=8, palette=color_palette)

    # Export the image
    geemap.ee_export_image(
        label_visualized, geotiff_path, scale=10, region=roi, file_per_band=False, crs='EPSG:4326'
    )

    print(f"Colored GeoTIFF saved to: {geotiff_path}")
    print(f"Image date: {image_date}")

def save_geotiff_esri_landcover(roi, geotiff_path, year=None):
    """Save ESRI Land Cover data as a colored GeoTIFF.
    
    Downloads and exports ESRI's 10m resolution global land cover classification.
    This dataset is updated annually and provides consistent global coverage.
    
    Args:
        roi: Earth Engine geometry defining region of interest
        geotiff_path: Output path for GeoTIFF file
        year: Optional year (YYYY) to get data for specific time.
              If None, uses the most recent available year.
    
    Land cover classes and colors:
        - Water (#1A5BAB): Water bodies
        - Trees (#358221): Tree cover
        - Flooded Vegetation (#87D19E): Vegetation in water-logged areas
        - Crops (#FFDB5C): Agricultural areas
        - Built Area (#ED022A): Urban and built-up areas
        - Bare Ground (#EDE9E4): Exposed soil and rock
        - Snow/Ice (#F2FAFF): Permanent snow and ice
        - Clouds (#C8C8C8): Cloud cover
        - Rangeland (#C6AD8D): Natural vegetation
        
    Note:
        The function will print the year of the data actually used, which may
        differ from the requested year if data is not available for that time.
    """
    # Initialize Earth Engine
    initialize_earth_engine()

    # Load the ESRI Land Cover dataset and filter by ROI
    esri_lulc = ee.ImageCollection("projects/sat-io/open-datasets/landcover/ESRI_Global-LULC_10m_TS").filterBounds(roi)

    # Check if there are any images in the filtered collection
    count = esri_lulc.size().getInfo()
    if count == 0:
        print("No ESRI Land Cover images found for the specified ROI.")
        return

    if year is None:
        # Get the latest available image
        esri_image = esri_lulc.sort('system:time_start', False).first()
        year = ee.Date(esri_image.get('system:time_start')).get('year').getInfo()
        print(f"No date specified. Using the latest available image from {year}.")
    else:
        # Extract the year from the date string
        # target_date = ee.Date(date)
        # target_year = target_date.get('year').getInfo()
        # Create date range for that year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        # Filter the collection to that year
        images_for_year = esri_lulc.filterDate(start_date, end_date)
        count = images_for_year.size().getInfo()
        if count == 0:
            print(f"No ESRI Land Cover images found for the year {year}.")
            return
        else:
            esri_image = images_for_year.mosaic()
            print(f"Using image for the specified year: {year}")

    # Clip the image to the ROI
    esri_clipped = esri_image.clip(roi)

    # Remap the image
    label = esri_clipped.select('b1').remap([1,2,4,5,7,8,9,10,11], [1,2,3,4,5,6,7,8,9])

    # Define class names and palette
    class_names = [
        "Water",
        "Trees",
        "Flooded Vegetation",
        "Crops",
        "Built Area",
        "Bare Ground",
        "Snow/Ice",
        "Clouds",
        "Rangeland"
    ]

    color_palette = [
        "#1A5BAB",  # Water
        "#358221",  # Trees
        "#87D19E",  # Flooded Vegetation
        "#FFDB5C",  # Crops
        "#ED022A",  # Built Area
        "#EDE9E4",  # Bare Ground
        "#F2FAFF",  # Snow/Ice
        "#C8C8C8",  # Clouds
        "#C6AD8D",  # Rangeland
    ]

    # Visualize the label band using the palette
    label_visualized = label.visualize(min=1, max=9, palette=color_palette)

    # Export the image
    geemap.ee_export_image(
        label_visualized, geotiff_path, scale=10, region=roi, file_per_band=False, crs='EPSG:4326'
    )

    print(f"Colored GeoTIFF saved to: {geotiff_path}")
    print(f"Image date: {year}")

def save_geotiff_open_buildings_temporal(aoi, geotiff_path):
    """Save Open Buildings temporal data as a GeoTIFF.
    
    Downloads and exports building height data from Google's Open Buildings dataset.
    This dataset provides building footprints and heights derived from satellite imagery.
    
    Args:
        aoi: Earth Engine geometry defining area of interest
        geotiff_path: Output path for GeoTIFF file
        
    Note:
        - The output GeoTIFF contains building heights in meters
        - The dataset is updated periodically and may not cover all regions
        - Resolution is fixed at 4 meters per pixel
        - Areas without buildings will have no-data values
    """
    # Initialize Earth Engine
    initialize_earth_engine()

    # Load the dataset
    collection = ee.ImageCollection('GOOGLE/Research/open-buildings-temporal/v1')

    # Get the latest image in the collection for the AOI
    latest_image = collection.filterBounds(aoi).sort('system:time_start', False).first()

    # Select the building height band
    building_height = latest_image.select('building_height')

    # Clip the image to the AOI
    clipped_image = building_height.clip(aoi)

    # Export the GeoTIFF
    geemap.ee_export_image(
        clipped_image,
        filename=geotiff_path,
        scale=4,
        region=aoi,
        file_per_band=False
    )

def save_geotiff_dsm_minus_dtm(roi, geotiff_path, meshsize, source):
    """Get the height difference between DSM and DTM from terrain data.
    
    Calculates the difference between Digital Surface Model (DSM) and Digital Terrain
    Model (DTM) to estimate heights of buildings, vegetation, and other above-ground
    features.
    
    Args:
        roi: Earth Engine geometry defining area of interest
        geotiff_path: Output path for GeoTIFF file
        meshsize: Size of each grid cell in meters - determines output resolution
        source: Source of terrain data, one of:
               - 'England 1m DSM - DTM': UK Environment Agency 1m resolution
               - 'Netherlands 0.5m DSM - DTM': AHN4 0.5m resolution
        
    Note:
        - A 100m buffer is automatically added around the ROI to ensure smooth
          interpolation at edges
        - The output represents height above ground level in meters
        - Negative values may indicate data artifacts or actual below-ground features
        - The function requires both DSM and DTM data to be available for the region
    """
    # Initialize Earth Engine
    initialize_earth_engine()

    # Add buffer around ROI to ensure smooth interpolation at edges
    buffer_distance = 100
    roi_buffered = roi.buffer(buffer_distance)

    if source == 'England 1m DSM - DTM':
        collection_name = 'UK/EA/ENGLAND_1M_TERRAIN/2022'
        dtm = ee.Image(collection_name).select('dtm')
        dsm = ee.Image(collection_name).select('dsm_first')
    elif source == 'Netherlands 0.5m DSM - DTM':
        collection = ee.ImageCollection('AHN/AHN4').filterBounds(roi_buffered)
        dtm = collection.select('dtm').mosaic()
        dsm = collection.select('dsm').mosaic()
    else:
        raise ValueError("Source must be either 'England' or 'Netherlands'")
    
    # Subtract DTM from DSM to get height difference
    height_diff = dsm.subtract(dtm)

    # Clip to buffered ROI
    image = height_diff.clip(roi_buffered)

    # Export as GeoTIFF using meshsize as scale
    save_geotiff(image, geotiff_path, scale=meshsize, region=roi_buffered, crs='EPSG:4326')