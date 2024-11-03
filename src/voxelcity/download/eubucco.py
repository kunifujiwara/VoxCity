import json
from shapely.geometry import Polygon, mapping, shape, MultiPolygon
import requests
import zipfile
import os
from io import BytesIO
import fiona
from shapely.ops import transform
from fiona.transform import transform_geom
import logging
import shapely

from ..geo.utils import get_country_name

country_links = {
        "Austria": "https://api.eubucco.com/v0.1/files/5e63fc15-b602-474b-a618-43080fa49db4/download",
        "Belgium": "https://api.eubucco.com/v0.1/files/08b81844-f46d-403b-bbd0-871fc5017b62/download",
        "Bulgaria": "https://api.eubucco.com/v0.1/files/4d0c2559-a3ec-4aea-bd56-e87d0261f5e5/download",
        "Croatia": "https://api.eubucco.com/v0.1/files/3638edd8-85af-4244-8208-128eedc85dcf/download",
        "Cyprus": "https://api.eubucco.com/v0.1/files/bee810d8-d4eb-4ddc-abf9-86d5247e05cf/download",
        "Czechia": "https://api.eubucco.com/v0.1/files/a130fd04-5d8c-4ea5-8383-62019cbec87f/download",
        "Czechia Other-license": "https://api.eubucco.com/v0.1/files/d044e347-7106-4516-811e-b340deac2041/download",
        "Denmark": "https://api.eubucco.com/v0.1/files/dc11a549-15db-4855-876d-ff3dfc401f76/download",
        "Estonia": "https://api.eubucco.com/v0.1/files/021d17ba-49a1-42e6-ae0b-fe73e8f02efb/download",
        "Finland": "https://api.eubucco.com/v0.1/files/c3131458-d835-4a22-8203-7d6367ae6f8f/download",
        "France": "https://api.eubucco.com/v0.1/files/0602abfe-d522-4683-a792-4dc4143a23fa/download",
        "Germany": "https://api.eubucco.com/v0.1/files/90148cbc-5bb1-4d1c-9935-8572a2a8c609/download",
        "Greece": "https://api.eubucco.com/v0.1/files/8d43e4c4-9e03-4ef1-b8c3-7bbb2ac23f4a/download",
        "Hungary": "https://api.eubucco.com/v0.1/files/bc0f8941-1b68-4c1c-9219-424c0d56d55a/download",
        "Ireland": "https://api.eubucco.com/v0.1/files/f580d806-9b32-4d1c-93ef-a4ff49889d56/download",
        "Italy": "https://api.eubucco.com/v0.1/files/e987077e-5c72-4903-a0d8-20ef8b9016de/download",
        "Italy Other-license": "https://api.eubucco.com/v0.1/files/d5a8e03f-0397-4f89-bcdd-a4bc61352ef6/download",
        "Latvia": "https://api.eubucco.com/v0.1/files/b7b3efdc-c9e1-4b70-bc81-2c4291cbdf8e/download",
        "Lithuania": "https://api.eubucco.com/v0.1/files/28862c49-25fe-4019-8c6b-cbad18d1c090/download",
        "Luxembourg": "https://api.eubucco.com/v0.1/files/0045dd6d-a2e0-4439-91c4-722856682cd6/download",
        "Malta": "https://api.eubucco.com/v0.1/files/2b4ecf81-365e-4a9b-91f6-70838c52487d/download",
        "Netherlands": "https://api.eubucco.com/v0.1/files/9f95ccbc-a095-4495-916c-6ea932f3ae10/download",
        "Poland": "https://api.eubucco.com/v0.1/files/6e8ea7fc-afcb-42b9-adbe-e12fa24048a7/download",
        "Portugal": "https://api.eubucco.com/v0.1/files/5d079772-5dd5-4dfc-95d2-393ca8edaa68/download",
        "Romania": "https://api.eubucco.com/v0.1/files/41cb29ed-e778-4b5b-807e-917ac93d48ca/download",
        "Slovakia": "https://api.eubucco.com/v0.1/files/17f71454-e4c4-41e8-b3ee-20b957abf546/download",
        "Slovenia": "https://api.eubucco.com/v0.1/files/e120065e-d6c2-42c6-b136-588072954e51/download",
        "Spain": "https://api.eubucco.com/v0.1/files/34dd019b-871f-443e-9b5d-61c29f8cb92c/download",
        "Sweden": "https://api.eubucco.com/v0.1/files/46fb09cc-38a9-46c0-bce2-d159e4b62963/download",
        "Switzerland": "https://api.eubucco.com/v0.1/files/1f4b7797-6e1e-44ab-a281-95d575360c9a/download",
}

# ================================
# Configuration
# ================================

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_and_convert_gdf_to_geojson_eubucco(gpkg_file, layer_name, rectangle_vertices, output_geojson):
    """
    Filters features in a GeoPackage that intersect with a given rectangle and writes them to a GeoJSON file.

    Parameters:
    - gpkg_file (str): Path to the GeoPackage file.
    - layer_name (str): Name of the layer within the GeoPackage to process.
    - rectangle_vertices (list of tuples): List of (latitude, longitude) tuples defining the rectangle.
    - output_geojson (str): Path to the output GeoJSON file.
    """
    # Create rectangle polygon in (longitude, latitude) order
    rectangle_vertices_lonlat = [(lon, lat) for lat, lon in rectangle_vertices]
    rectangle_polygon = Polygon(rectangle_vertices_lonlat)

    # Define a function to swap coordinates from (lon, lat) to (lat, lon)
    def swap_coordinates_gdf(x, y, z=None):
        return y, x

    # Determine Shapely version
    shapely_version = shapely.__version__
    major_version = int(shapely_version.split('.')[0])

    logging.info(f"Using Shapely version: {shapely_version}")

    # Open the GeoPackage layer with Fiona
    with fiona.Env():
        with fiona.open(gpkg_file, layer=layer_name, mode='r') as src:
            src_crs = src.crs
            if not src_crs:
                logging.error("Input GeoPackage layer has no CRS defined.")
                raise ValueError("Input GeoPackage layer has no CRS defined.")

            # Assume the rectangle is in EPSG:4326 (WGS84)
            target_crs = 'EPSG:4326'
            if src_crs.get('init') != 'epsg:4326' and 'epsg:4326' not in src_crs.values():
                transform_required = True
                logging.info("Transforming rectangle polygon to match source CRS.")
                rectangle_polygon_transformed = transform_geom(target_crs, src_crs, mapping(rectangle_polygon))
                rectangle_polygon = shape(rectangle_polygon_transformed)
            else:
                transform_required = False

            # Define the output schema
            output_schema = {
                'geometry': 'Polygon',
                'properties': {
                    'height': 'float',
                    'confidence': 'float'
                },
            }

            # Open the output GeoJSON file
            with fiona.open(
                output_geojson,
                'w',
                driver='GeoJSON',
                crs=target_crs,
                schema=output_schema
            ) as dst:
                feature_count = 0
                missing_height_count = 0
                for idx, feature in enumerate(src):
                    if idx % 10000 == 0 and idx > 0:
                        logging.info(f"Processed {idx} features...")

                    geom = feature['geometry']
                    if not geom:
                        continue  # Skip features without geometry

                    shapely_geom = shape(geom)

                    # Check if the geometry intersects with the rectangle
                    if not shapely_geom.intersects(rectangle_polygon):
                        continue  # Skip non-overlapping features

                    # Get the intersection geometry
                    intersection = shapely_geom.intersection(rectangle_polygon)

                    if intersection.is_empty:
                        continue  # No intersection

                    # Split MultiPolygons into individual Polygons
                    if isinstance(intersection, Polygon):
                        polygons = [intersection]
                    elif isinstance(intersection, MultiPolygon):
                        if major_version < 2:
                            polygons = list(intersection)
                        else:
                            polygons = list(intersection.geoms)
                    else:
                        continue  # Skip other geometry types

                    # Process each Polygon
                    for poly in polygons:
                        # Transform the polygon to EPSG:4326 if required
                        if transform_required:
                            transformed_geom = transform_geom(src_crs, target_crs, mapping(poly))
                            shapely_transformed_poly = shape(transformed_geom)
                        else:
                            shapely_transformed_poly = poly

                        # Swap coordinates from (lon, lat) to (lat, lon)
                        swapped_poly = transform(swap_coordinates_gdf, shapely_transformed_poly)

                        # Extract coordinates
                        coords = []
                        # Exterior ring
                        exterior = list(swapped_poly.exterior.coords)
                        coords.append(exterior)
                        # Interior rings (holes), if any
                        for interior in swapped_poly.interiors:
                            coords.append(list(interior.coords))

                        # Build the GeoJSON geometry
                        out_geom = {
                            'type': 'Polygon',
                            'coordinates': coords
                        }

                        # Safely extract and convert 'height'
                        height = feature['properties'].get('height', -1.0)
                        if height is None:
                            height = -1.0
                            missing_height_count += 1
                            logging.debug(f"Feature ID {feature.get('id', 'N/A')} has missing 'height'. Assigned default value.")

                        try:
                            height = float(height)
                        except (TypeError, ValueError):
                            height = -1.0
                            missing_height_count += 1
                            logging.debug(f"Feature ID {feature.get('id', 'N/A')} has invalid 'height'. Assigned default value.")

                        # Build the properties
                        properties = {
                            'height': height,
                            'confidence': -1.0  # As per your initial requirement
                        }

                        # Create the output feature
                        out_feature = {
                            'geometry': out_geom,
                            'properties': properties,
                            'type': 'Feature'
                        }

                        # Write the feature to the output GeoJSON
                        dst.write(out_feature)
                        feature_count += 1

                logging.info(f"Total features written to GeoJSON: {feature_count}")
                logging.info(f"Total features with missing or invalid 'height': {missing_height_count}")


def download_extract_open_gpkg_from_eubucco(url, output_dir):
    """
    Downloads a ZIP file from a URL, extracts the GeoPackage (.gpkg) file, and returns its path.

    Parameters:
    - url (str): URL to download the ZIP file containing the GeoPackage.

    Returns:
    - str: Path to the extracted GeoPackage file.
    """

    # Download the file
    logging.info("Downloading file...")
    response = requests.get(url)
    if response.status_code != 200:
        logging.error(f"Failed to download file. Status code: {response.status_code}")
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    # Extract the ZIP file
    logging.info("Extracting ZIP file...")
    with zipfile.ZipFile(BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(f"{output_dir}/EUBUCCO_raw")

    # Find the GPKG file
    gpkg_file = None
    for root, dirs, files in os.walk(f"{output_dir}/EUBUCCO_raw"):
        for file in files:
            if file.endswith(".gpkg"):
                gpkg_file = os.path.join(root, file)
                break
        if gpkg_file:
            break

    if not gpkg_file:
        logging.error("No GPKG file found in the extracted files.")
        raise Exception("No GPKG file found in the extracted files.")

    logging.info(f"GeoPackage file found: {gpkg_file}")
    return gpkg_file

def save_geojson_from_eubucco(rectangle_vertices, country_links, output_dir, file_name):
    """
    Downloads, extracts, filters, and converts GeoPackage data to GeoJSON based on the rectangle vertices.

    Parameters:
    - rectangle_vertices (list of tuples): List of (latitude, longitude) tuples defining the rectangle.
    - country_links (dict): Dictionary mapping country names to their respective GeoPackage URLs.

    Returns:
    - None: Writes the output to a GeoJSON file.
    """
    country_name = get_country_name(rectangle_vertices[0][0], rectangle_vertices[0][1])
    if country_name in country_links:
        url = country_links[country_name]
    else:
        logging.error("Your target area does not have data in EUBUCCO.")
        return

    gpkg_file = download_extract_open_gpkg_from_eubucco(url, output_dir)

    # Determine the layer name (assuming the first layer)
    with fiona.Env():
        layers = fiona.listlayers(gpkg_file)
        if not layers:
            logging.error("No layers found in the GeoPackage.")
            return
        layer_name = layers[0]  # Modify if you need a specific layer
        logging.info(f"Using layer: {layer_name}")

    file_path = f"{output_dir}/{file_name}"

    # Process and convert to GeoJSON
    filter_and_convert_gdf_to_geojson_eubucco(
        gpkg_file=gpkg_file,
        layer_name=layer_name,
        rectangle_vertices=rectangle_vertices,
        output_geojson=file_path
    )

    logging.info(f"GeoJSON file has been created at: {file_path}")

def load_geojson_from_eubucco(rectangle_vertices, output_dir):

    # Path to save and load the GeoJSON file
    file_name = 'building.geojson' 
    file_path = f"{output_dir}/{file_name}"

    # Get GeoJSON from EUBUCCO
    save_geojson_from_eubucco(rectangle_vertices, country_links, output_dir, file_name)   

    # Load raw JSON
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    geojson_data = raw_data['features']

    return geojson_data