"""
Module for interacting with Google Earth Engine API and downloading geospatial data.

This module provides functionality to initialize Earth Engine, create regions of interest,
download various types of satellite imagery and terrain data, and save them as GeoTIFF files.
It supports multiple data sources including DEMs, land cover maps, and building footprints.
"""

# Earth Engine and geospatial imports
import ee
import geemap

# Local imports
from ..geo.utils import convert_format_lat_lon

def initialize_earth_engine():
    """Initialize the Earth Engine API."""
    ee.Initialize()

def get_roi(input_coords):
    """Create an Earth Engine region of interest polygon from coordinates.
    
    Args:
        input_coords: List of coordinate pairs defining the polygon vertices
        
    Returns:
        ee.Geometry.Polygon: Earth Engine polygon geometry
    """
    coords = convert_format_lat_lon(input_coords)
    return ee.Geometry.Polygon(coords)

def get_center_point(roi):
    """Get the centroid coordinates of a region of interest.
    
    Args:
        roi: Earth Engine geometry
        
    Returns:
        tuple: (longitude, latitude) of centroid
    """
    center_point = roi.centroid()
    center_coords = center_point.coordinates().getInfo()
    return center_coords[0], center_coords[1]

def get_ee_image_collection(collection_name, roi):
    """Get the first image from an Earth Engine ImageCollection filtered by region.
    
    Args:
        collection_name: Name of the Earth Engine ImageCollection
        roi: Earth Engine geometry to filter by
        
    Returns:
        ee.Image: First image from collection clipped to ROI
    """
    # Filter collection by bounds and get first image
    collection = ee.ImageCollection(collection_name).filterBounds(roi)
    return collection.sort('system:time_start').first().clip(roi).unmask()

def get_ee_image(collection_name, roi):
    """Get an Earth Engine Image clipped to a region.
    
    Args:
        collection_name: Name of the Earth Engine Image
        roi: Earth Engine geometry to clip to
        
    Returns:
        ee.Image: Image clipped to ROI
    """
    collection = ee.Image(collection_name)
    return collection.clip(roi)

def save_geotiff(image, filename, resolution=1, scale=None, region=None, crs=None):
    """Save an Earth Engine image as a GeoTIFF file.
    
    Args:
        image: Earth Engine image to save
        filename: Output filename
        resolution: Output resolution in degrees (default: 1)
        scale: Output scale in meters
        region: Region to export
        crs: Coordinate reference system
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
    
    Args:
        roi_buffered: Earth Engine geometry with buffer
        source: DEM source ('NASA', 'COPERNICUS', 'DeltaDTM', 'FABDEM', 'England 1m DTM',
               'DEM France 5m', 'DEM France 1m', 'AUSTRALIA 5M DEM', 'USGS 3DEP 1m')
               
    Returns:
        ee.Image: DEM image clipped to region
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
    elif source == 'USGS 3DEP 1m':
        collection_name = 'USGS/3DEP/1m'
        dem = ee.ImageCollection(collection_name).mosaic()
    # Commented out sources that are not yet implemented
    # elif source == 'Canada High Resolution DTM':
    #     collection_name = "projects/sat-io/open-datasets/OPEN-CANADA/CAN_ELV/HRDEM_1M_DTM"
    #     collection = ee.ImageCollection(collection_name)
    #     dem = collection.mosaic() 

    # elif source == 'FABDEM':
    return dem.clip(roi_buffered)

def save_geotiff_esa_land_cover(roi, geotiff_path):
    """Save ESA WorldCover land cover data as a colored GeoTIFF.
    
    Args:
        roi: Earth Engine geometry defining region of interest
        geotiff_path: Output path for GeoTIFF file
    """
    # Initialize Earth Engine
    ee.Initialize()

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
    
    Args:
        roi: Earth Engine geometry defining region of interest
        geotiff_path: Output path for GeoTIFF file
        date: Optional date string to get data for specific time
    """
    # Initialize Earth Engine
    ee.Initialize()

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
    
    Args:
        roi: Earth Engine geometry defining region of interest
        geotiff_path: Output path for GeoTIFF file
        year: Optional year to get data for specific time
    """
    # Initialize Earth Engine
    ee.Initialize()

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
    
    Args:
        aoi: Earth Engine geometry defining area of interest
        geotiff_path: Output path for GeoTIFF file
    """
    # Initialize Earth Engine
    ee.Initialize()

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