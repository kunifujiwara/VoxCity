import numpy as np
import rasterio
from affine import Affine
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ee
import geemap
from pyproj import CRS, Transformer
import rasterio
from pyproj.geod import Geod

from ..geo.utils import convert_format_lat_lon

def initialize_earth_engine():
    ee.Initialize()

def get_roi(input_coords):
    coords = convert_format_lat_lon(input_coords)
    return ee.Geometry.Polygon(coords)

def get_center_point(roi):
    center_point = roi.centroid()
    center_coords = center_point.coordinates().getInfo()
    return center_coords[0], center_coords[1]

def get_ee_image_collection(collection_name, roi):
    collection = ee.ImageCollection(collection_name).filterBounds(roi)
    return collection.sort('system:time_start').first().clip(roi).unmask()

def get_ee_image(collection_name, roi):
    collection = ee.Image(collection_name)
    return collection.clip(roi)

def save_geotiff(image, filename, resolution=1, scale=None, region=None, crs=None):
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
    # elif source == 'Canada High Resolution DTM':
    #     collection_name = "projects/sat-io/open-datasets/OPEN-CANADA/CAN_ELV/HRDEM_1M_DTM"
    #     collection = ee.ImageCollection(collection_name)
    #     dem = collection.mosaic() 

    # elif source == 'FABDEM':
    return dem.clip(roi_buffered)

def save_geotiff_esa_land_cover(roi, geotiff_path):
    # Initialize Earth Engine
    ee.Initialize()

    # Load the ESA WorldCover dataset
    esa = ee.ImageCollection("ESA/WorldCover/v200").first()

    # Clip the image to the AOI
    esa_clipped = esa.clip(roi)

    # Define the color palette based on the provided image
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

    # Create a list of colors in the order of class values
    colors = [color_map[i] for i in sorted(color_map.keys())]

    # Apply the color palette to the image
    esa_colored = esa_clipped.remap(
        list(color_map.keys()),
        list(range(len(color_map)))
    ).visualize(palette=colors, min=0, max=len(color_map)-1)

    geemap.ee_export_image(esa_colored, geotiff_path, scale=10, region=roi)

    print(f"Colored GeoTIFF saved to: {geotiff_path}")

def save_geotiff_dynamic_world_v1(roi, geotiff_path, date=None):

    # Initialize Earth Engine
    ee.Initialize()

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

# def get_grid_gee(tag, collection_name, coords, mesh_size, land_cover_classes=None, buffer_distance=None):
#     initialize_earth_engine()

#     roi = get_roi(coords)
#     center_lon, center_lat = get_center_point(roi)

#     if buffer_distance:
#         roi_buffered = roi.buffer(buffer_distance)
#         image = get_dem_image(roi_buffered)
#         save_geotiff(image, f"{tag}.tif", scale=30, region=roi_buffered)
#     else:
#         image = get_image_collection(collection_name, roi)
#         save_geotiff(image, f"{tag}.tif")

#     if tag == 'canopy_height':
#         grid = create_canopy_height_grid(f"{tag}.tif", mesh_size)
#         visualize_grid(grid, mesh_size, title=f'{tag.replace("_", " ").title()} Grid')
#     elif tag == 'land_cover':
#         grid = create_land_cover_grid(f"{tag}.tif", mesh_size, land_cover_classes)
#         color_map = {cls: [r/255, g/255, b/255] for (r,g,b), cls in land_cover_classes.items()}
#         # color_map['No Data'] = [0.5, 0.5, 0.5]
#         visualize_land_cover_grid(grid, mesh_size, color_map, land_cover_classes)
#         grid = convert_land_cover_array(grid, land_cover_classes)
#     elif tag == 'nasa_dem':
#         converted_coords = convert_format(coords)
#         roi_shapely = Polygon(converted_coords)
#         grid = create_dem_grid(f"{tag}.tif", mesh_size, roi_shapely)
#         visualize_grid(grid, mesh_size, title='Digital Elevation Model', cmap='terrain', label='Elevation (m)')

#     print(f"Resulting grid shape: {grid.shape}")

#     return grid