#To download microsoft building footprints (Beta)

import pandas as pd
import os
from .utils import download_file
from ..geo.utils import tile_from_lat_lon, quadkey_to_tile
from ..file.geojson import load_geojsons_from_multiple_gz, swap_coordinates

def get_geojson_links(output_dir):

    # print("Downloading dataset-links.csv")
    
    # URL of the file you want to download
    url = "https://minedbuildings.blob.core.windows.net/global-buildings/dataset-links.csv"

    # Local filename to save the downloaded file
    filepath = os.path.join(output_dir, "dataset-links.csv")

    # Call the function to download the file
    download_file(url, filepath)

    data_types = {
        'Location': 'str',
        'QuadKey': 'str',
        'Url': 'str',
        'Size': 'str'
    }

    df_links = pd.read_csv(filepath, dtype=data_types)
    
    return df_links

def find_row_for_location(df, lat, lon):
    for index, row in df.iterrows():
        quadkey = str(row['QuadKey'])
        if not isinstance(quadkey, str) or len(quadkey) == 0:
            continue
        try:
            loc_tile_x, loc_tile_y = tile_from_lat_lon(lat, lon, len(quadkey))
            qk_tile_x, qk_tile_y, _ = quadkey_to_tile(quadkey)
            if loc_tile_x == qk_tile_x and loc_tile_y == qk_tile_y:
                return row
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    return None

def get_mbfp_geojson(output_dir, rectangle_vertices):
    print("Downloading geojson files")
    # print_flush(f"Testing get_geojson_links with output_dir: {output_dir}")
    df_links = get_geojson_links(output_dir)

    # Find and download files
    filenames = []
    for vertex in rectangle_vertices:
        lat, lon = vertex
        row = find_row_for_location(df_links, lat, lon)
        if row is not None:
            location = row["Location"]
            quadkey = row["QuadKey"]
            filename = os.path.join(output_dir, f"{location}_{quadkey}.gz")
            if filename not in filenames:
                filenames.append(filename)
                download_file(row["Url"], filename)
        else:
            print("No matching row found.")

    # Load and process GeoJSON data
    geojson_data = load_geojsons_from_multiple_gz(filenames)
    swap_coordinates(geojson_data)

    return geojson_data