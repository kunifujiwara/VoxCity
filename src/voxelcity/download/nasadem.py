import numpy as np
from pyproj import CRS
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.interpolate import griddata, NearestNDInterpolator
import requests
import io

def get_utm_zone(lon):
    return int((lon + 180) / 6) + 1

def get_utm_crs(lat, lon):
    utm_zone = get_utm_zone(lon)
    hemisphere = 'north' if lat >= 0 else 'south'
    return CRS(f'+proj=utm +zone={utm_zone} +{hemisphere} +ellps=WGS84 +datum=WGS84 +units=m +no_defs')

def download_nasa_dem(bbox, api_key):
    nasa_url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": "NASADEM",
        "south": bbox[1],
        "north": bbox[3],
        "west": bbox[0],
        "east": bbox[2],
        "outputFormat": "GTiff",
        "API_Key": api_key
    }
    response = requests.get(nasa_url, params=params)
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        raise Exception(f"Failed to download DEM data: {response.text}")

def interpolate_dem(dem_data, grid_coords, dst_crs):
    with rasterio.open(dem_data) as src:
        src_crs = src.crs
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs, dst_crs, src.width, src.height, *src.bounds)

        dst_shape = (src.count, dst_height, dst_width)
        dst_array = np.zeros(dst_shape, dtype=src.dtypes[0])

        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array[0],
            src_transform=src.transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear
        )

        rows, cols = np.mgrid[0:dst_height, 0:dst_width]
        raster_coords = np.column_stack((cols.flatten(), rows.flatten()))
        raster_coords = np.column_stack((dst_transform * (raster_coords[:, 0], raster_coords[:, 1])))

        interpolated = griddata(raster_coords, dst_array[0].flatten(), grid_coords, method='cubic')

        mask = np.isnan(interpolated)
        if np.any(mask):
            nn_interpolator = NearestNDInterpolator(raster_coords, dst_array[0].flatten())
            interpolated[mask] = nn_interpolator(grid_coords[mask])

        return interpolated