#To download microsoft building footprints (Beta)

import pandas as pd
import os
from .utils import download_file
from ..geo.utils import tile_from_lat_lon, quadkey_to_tile

def get_geojson_links(output_dir):

    os.makedirs(output_dir, exist_ok=True) 
    
    print("Ensuring output directory exists: ", output_dir)

    print("Downloading dataset-links.csv...")
    
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