"""
CityGML Parser Module for PLATEAU Data

This module provides functionality to parse CityGML files from Japan's PLATEAU dataset,
extracting building footprints, terrain information, and vegetation data.
The module handles various LOD (Level of Detail) representations and coordinate systems.

Main features:
- Download and extract PLATEAU data from URLs
- Parse CityGML files for buildings, terrain, and vegetation
- Handle coordinate transformations and validations
- Support for mesh code decoding
"""

import requests
import zipfile
import io
import os
import numpy as np
from urllib.parse import urlparse
from pathlib import Path
import lxml.etree as ET
import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------------------
# Script to get tile boundary from file name
# --------------------------------------------------------------------
import re
from shapely.geometry import Polygon

def decode_2nd_level_mesh(mesh6):
    """
    Decode a standard (2nd-level) mesh code to geographic coordinates.

    Args:
        mesh6 (str): A 6-digit mesh code string.

    Returns:
        tuple: (lat_sw, lon_sw, lat_ne, lon_ne) coordinates in degrees representing
               the southwest and northeast corners of the mesh.

    Notes:
        - The mesh system divides Japan into a grid of cells
        - Each 2nd-level mesh is 1/12° latitude × 0.125° longitude
    """
    code = int(mesh6)
    # Extract each piece
    N1 = code // 10000              # first 2 digits
    M1 = (code // 100) % 100        # next 2 digits
    row_2nd = (code // 10) % 10     # 5th digit
    col_2nd = code % 10             # 6th digit
    
    # 1st-level mesh "southwest" corner
    lat_sw_1 = (N1 * 40.0) / 60.0    # each N1 => 40' => 2/3 degrees
    lon_sw_1 = 100.0 + M1           # each M1 => offset from 100°E
    
    # 2nd-level mesh subdivides that 8×8 => each cell = 1/12° lat x 0.125° lon
    dlat_2nd = (40.0 / 60.0) / 8.0   # 1/12°
    dlon_2nd = 1.0 / 8.0            # 0.125°
    
    lat_sw = lat_sw_1 + row_2nd * dlat_2nd
    lon_sw = lon_sw_1 + col_2nd * dlon_2nd
    lat_ne = lat_sw + dlat_2nd
    lon_ne = lon_sw + dlon_2nd
    
    return (lat_sw, lon_sw, lat_ne, lon_ne)

def decode_mesh_code(mesh_str):
    """
    Decode mesh codes into geographic boundary coordinates.

    Args:
        mesh_str (str): A mesh code string (6 or 8 digits).

    Returns:
        list: List of (lon, lat) tuples forming a closed polygon in WGS84.

    Raises:
        ValueError: If mesh code length is invalid or unsupported.

    Notes:
        - 6-digit codes represent standard 2nd-level mesh
        - 8-digit codes represent 2nd-level mesh subdivided 10×10
    """
    if len(mesh_str) < 6:
        raise ValueError(f"Mesh code '{mesh_str}' is too short.")
    
    # Decode the first 6 digits as a 2nd-level mesh
    mesh6 = mesh_str[:6]
    lat_sw_2, lon_sw_2, lat_ne_2, lon_ne_2 = decode_2nd_level_mesh(mesh6)
    
    # If exactly 6 digits => full 2nd-level tile
    if len(mesh_str) == 6:
        return [
            (lon_sw_2, lat_sw_2),
            (lon_ne_2, lat_sw_2),
            (lon_ne_2, lat_ne_2),
            (lon_sw_2, lat_ne_2),
            (lon_sw_2, lat_sw_2)
        ]
    
    # If 8 digits => last 2 subdivide the tile 10×10
    elif len(mesh_str) == 8:
        row_10 = int(mesh_str[6])  # 7th digit
        col_10 = int(mesh_str[7])  # 8th digit
        
        # Sub-tile size in lat/lon
        dlat_10 = (lat_ne_2 - lat_sw_2) / 10.0
        dlon_10 = (lon_ne_2 - lon_sw_2) / 10.0
        
        lat_sw = lat_sw_2 + row_10 * dlat_10
        lon_sw = lon_sw_2 + col_10 * dlon_10
        lat_ne = lat_sw + dlat_10
        lon_ne = lon_sw + dlon_10
        
        return [
            (lon_sw, lat_sw),
            (lon_ne, lat_sw),
            (lon_ne, lat_ne),
            (lon_sw, lat_ne),
            (lon_sw, lat_sw)
        ]
    
    else:
        raise ValueError(
            f"Unsupported mesh code length '{mesh_str}'. "
            "This script only handles 6-digit or 8-digit codes."
        )

def get_tile_polygon_from_filename(filename):
    """
    Extract and decode mesh code from PLATEAU filename into boundary polygon.

    Args:
        filename (str): PLATEAU format filename (e.g. '51357348_bldg_6697_op.gml')

    Returns:
        list: List of (lon, lat) tuples forming the tile boundary polygon in WGS84.

    Raises:
        ValueError: If no mesh code found in filename.
    """
    # Look for leading digits until the first underscore
    m = re.match(r'^(\d+)_', filename)
    if not m:
        # If no match, you can either raise an error or return None
        raise ValueError(f"No leading digit code found in filename: {filename}")
    
    mesh_code = m.group(1)
    return decode_mesh_code(mesh_code)

# --------------------------------------------------------------------
# Original script logic
# --------------------------------------------------------------------

def download_and_extract_zip(url, extract_to='.', ssl_verify=True, ca_bundle=None, timeout=60):
    """
    Download and extract a zip file from a URL to specified directory.

    Args:
        url (str): URL of the zip file to download.
        extract_to (str): Directory to extract files to (default: current directory).
        ssl_verify (bool): Whether to verify SSL certificates (default: True).
        ca_bundle (str|None): Path to a CA bundle file. Overrides verify when provided.
        timeout (int|float): Request timeout in seconds (default: 60).

    Returns:
        tuple: (extraction_path, folder_name) where files were extracted.

    Notes:
        - Creates a subdirectory named after the zip file (without .zip)
        - Prints status messages for success/failure
    """
    verify_arg = ca_bundle if ca_bundle else ssl_verify
    try:
        response = requests.get(url, verify=verify_arg, timeout=timeout)
        if response.status_code == 200:
            parsed_url = urlparse(url)
            zip_filename = os.path.basename(parsed_url.path)
            folder_name = os.path.splitext(zip_filename)[0]  # Remove the .zip extension

            extraction_path = os.path.join(extract_to, folder_name)
            os.makedirs(extraction_path, exist_ok=True)

            zip_file = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_file) as z:
                z.extractall(extraction_path)
                print(f"Extracted to {extraction_path}")
        else:
            print(f"Failed to download the file. Status code: {response.status_code}")
    except requests.exceptions.SSLError as e:
        print("SSL error when downloading CityGML zip. You can pass 'ssl_verify=False' to skip verification, "
              "or provide a CA bundle path via 'ca_bundle'. Error:", e)
        raise

    return extraction_path, folder_name


def validate_coords(coords):
    """
    Validate that coordinates are finite numbers.

    Args:
        coords (list): List of coordinate tuples.

    Returns:
        bool: True if all coordinates are valid (not inf/NaN), False otherwise.
    """
    return all(not np.isinf(x) and not np.isnan(x) for coord in coords for x in coord)


def swap_coordinates(polygon):
    """
    Swap coordinate order in a polygon (lat/lon to lon/lat or vice versa).

    Args:
        polygon (Polygon/MultiPolygon): Input polygon with coordinates to swap.

    Returns:
        Polygon/MultiPolygon: New polygon with swapped coordinates.

    Notes:
        - Handles both single Polygon and MultiPolygon geometries
        - Creates new geometry objects rather than modifying in place
    """
    if isinstance(polygon, MultiPolygon):
        new_polygons = []
        for geom in polygon.geoms:
            coords = list(geom.exterior.coords)
            swapped_coords = [(y, x) for x, y in coords]
            new_polygons.append(Polygon(swapped_coords))
        return MultiPolygon(new_polygons)
    else:
        coords = list(polygon.exterior.coords)
        swapped_coords = [(y, x) for x, y in coords]
        return Polygon(swapped_coords)


def extract_terrain_info(file_path, namespaces):
    """
    Extract terrain elevation data from CityGML file.

    Args:
        file_path (str): Path to CityGML file.
        namespaces (dict): XML namespace mappings.

    Returns:
        list: List of dictionaries containing terrain features:
            - relief_id: Feature identifier
            - tin_id: TIN surface identifier
            - triangle_id/breakline_id/mass_point_id: Specific element ID
            - elevation: Height value
            - geometry: Shapely geometry object
            - source_file: Original file name

    Notes:
        - Processes TIN Relief, breaklines, and mass points
        - Validates all geometries before inclusion
        - Handles coordinate conversion and validation
    """
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        terrain_elements = []

        # Look for Relief features in the CityGML file
        for relief in root.findall('.//dem:ReliefFeature', namespaces):
            relief_id = relief.get('{http://www.opengis.net/gml}id')

            # Extract TIN Relief components
            for tin in relief.findall('.//dem:TINRelief', namespaces):
                tin_id = tin.get('{http://www.opengis.net/gml}id')

                triangles = tin.findall('.//gml:Triangle', namespaces)
                for i, triangle in enumerate(triangles):
                    pos_lists = triangle.findall('.//gml:posList', namespaces)
                    for pos_list in pos_lists:
                        try:
                            coords_text = pos_list.text.strip().split()
                            coords = []
                            elevations = []

                            for j in range(0, len(coords_text), 3):
                                if j + 2 < len(coords_text):
                                    x = float(coords_text[j])
                                    y = float(coords_text[j+1])
                                    z = float(coords_text[j+2])

                                    if not np.isinf(x) and not np.isinf(y) and not np.isinf(z):
                                        coords.append((x, y))
                                        elevations.append(z)

                            if len(coords) >= 3 and validate_coords(coords):
                                polygon = Polygon(coords)
                                if polygon.is_valid:
                                    centroid = polygon.centroid
                                    avg_elevation = np.mean(elevations)
                                    terrain_elements.append({
                                        'relief_id': relief_id,
                                        'tin_id': tin_id,
                                        'triangle_id': f"{tin_id}_tri_{i}",
                                        'elevation': avg_elevation,
                                        'geometry': centroid,
                                        'polygon': polygon,
                                        'source_file': Path(file_path).name
                                    })
                        except (ValueError, IndexError) as e:
                            print(f"Error processing triangle in relief {relief_id}: {e}")
                            continue

            # Extract breaklines
            for breakline in relief.findall('.//dem:breaklines', namespaces):
                for line in breakline.findall('.//gml:LineString', namespaces):
                    line_id = line.get('{http://www.opengis.net/gml}id')
                    pos_list = line.find('.//gml:posList', namespaces)
                    if pos_list is not None:
                        try:
                            coords_text = pos_list.text.strip().split()
                            points = []
                            elevations = []

                            for j in range(0, len(coords_text), 3):
                                if j + 2 < len(coords_text):
                                    x = float(coords_text[j])
                                    y = float(coords_text[j+1])
                                    z = float(coords_text[j+2])
                                    if not np.isinf(x) and not np.isinf(y) and not np.isinf(z):
                                        points.append(Point(x, y))
                                        elevations.append(z)

                            for k, point in enumerate(points):
                                if point.is_valid:
                                    terrain_elements.append({
                                        'relief_id': relief_id,
                                        'breakline_id': line_id,
                                        'point_id': f"{line_id}_pt_{k}",
                                        'elevation': elevations[k],
                                        'geometry': point,
                                        'polygon': None,
                                        'source_file': Path(file_path).name
                                    })
                        except (ValueError, IndexError) as e:
                            print(f"Error processing breakline {line_id}: {e}")
                            continue

            # Extract mass points
            for mass_point in relief.findall('.//dem:massPoint', namespaces):
                for point in mass_point.findall('.//gml:Point', namespaces):
                    point_id = point.get('{http://www.opengis.net/gml}id')
                    pos = point.find('.//gml:pos', namespaces)
                    if pos is not None:
                        try:
                            coords = pos.text.strip().split()
                            if len(coords) >= 3:
                                x = float(coords[0])
                                y = float(coords[1])
                                z = float(coords[2])
                                if not np.isinf(x) and not np.isinf(y) and not np.isinf(z):
                                    point_geom = Point(x, y)
                                    if point_geom.is_valid:
                                        terrain_elements.append({
                                            'relief_id': relief_id,
                                            'mass_point_id': point_id,
                                            'elevation': z,
                                            'geometry': point_geom,
                                            'polygon': None,
                                            'source_file': Path(file_path).name
                                        })
                        except (ValueError, IndexError) as e:
                            print(f"Error processing mass point {point_id}: {e}")
                            continue

        print(f"Extracted {len(terrain_elements)} terrain elements from {Path(file_path).name}")
        return terrain_elements

    except Exception as e:
        print(f"Error processing terrain in file {Path(file_path).name}: {e}")
        return []


def extract_vegetation_info(file_path, namespaces):
    """
    Extract vegetation features from CityGML file.

    Args:
        file_path (str): Path to CityGML file.
        namespaces (dict): XML namespace mappings.

    Returns:
        list: List of dictionaries containing vegetation features:
            - object_type: 'PlantCover' or 'SolitaryVegetationObject'
            - vegetation_id: Feature identifier
            - height: Vegetation height (if available)
            - geometry: Shapely geometry object
            - source_file: Original file name

    Notes:
        - Handles both PlantCover and SolitaryVegetationObject features
        - Processes multiple LOD representations
        - Validates geometries before inclusion
    """
    vegetation_elements = []
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Build namespaces dynamically from the file and merge with provided ones
        nsmap = root.nsmap or {}
        # Fallbacks in case discovery fails
        fallback_ns = {
            'core': 'http://www.opengis.net/citygml/2.0',
            'bldg': 'http://www.opengis.net/citygml/building/2.0',
            'gml': 'http://www.opengis.net/gml',
            'uro': 'https://www.geospatial.jp/iur/uro/3.0',
            'dem': 'http://www.opengis.net/citygml/relief/2.0',
            'veg': 'http://www.opengis.net/citygml/vegetation/2.0'
        }

        def pick_ns(prefix, keyword=None, fallback_key=None):
            # Prefer provided namespaces if valid
            if isinstance(namespaces, dict) and prefix in namespaces and namespaces[prefix]:
                return namespaces[prefix]
            # Then try from document nsmap
            uri = nsmap.get(prefix)
            if uri:
                return uri
            # Then try keyword search
            if keyword:
                for v in nsmap.values():
                    if isinstance(v, str) and keyword in v:
                        return v
            # Finally fallback
            return fallback_ns[fallback_key or prefix]

        ns = {
            'core': pick_ns('core', keyword='citygml', fallback_key='core'),
            'bldg': pick_ns('bldg', keyword='building', fallback_key='bldg'),
            'gml': pick_ns('gml', keyword='gml', fallback_key='gml'),
            'uro': pick_ns('uro', keyword='iur/uro', fallback_key='uro'),
            'dem': pick_ns('dem', keyword='relief', fallback_key='dem'),
            'veg': pick_ns('veg', keyword='vegetation', fallback_key='veg')
        }
    except Exception as e:
        print(f"Error parsing CityGML file {Path(file_path).name}: {e}")
        return vegetation_elements

    # Helper: parse polygons in <gml:MultiSurface> or <veg:lodXMultiSurface>
    def parse_lod_multisurface(lod_elem):
        polygons = []
        for poly_node in lod_elem.findall('.//gml:Polygon', ns):
            ring_node = poly_node.find('.//gml:exterior//gml:LinearRing//gml:posList', ns)
            if ring_node is None or ring_node.text is None:
                continue
            coords_text = ring_node.text.strip().split()
            coords = []
            for i in range(0, len(coords_text), 3):
                try:
                    x = float(coords_text[i])
                    y = float(coords_text[i+1])
                    # z = float(coords_text[i+2])  # If you need Z
                    coords.append((x, y))
                except:
                    pass
            if len(coords) >= 3:
                polygon = Polygon(coords)
                if polygon.is_valid:
                    polygons.append(polygon)

        if not polygons:
            return None
        elif len(polygons) == 1:
            return polygons[0]
        else:
            return MultiPolygon(polygons)

    def get_veg_geometry(veg_elem):
        """
        Search for geometry under lod0Geometry, lod1Geometry, lod2Geometry,
        lod3Geometry, lod4Geometry, as well as lod0MultiSurface ... lod4MultiSurface.
        Return a Shapely geometry (Polygon or MultiPolygon) if found.
        """
        geometry_lods = [
            "lod0Geometry", "lod1Geometry", "lod2Geometry", "lod3Geometry", "lod4Geometry",
            "lod0MultiSurface", "lod1MultiSurface", "lod2MultiSurface", "lod3MultiSurface", "lod4MultiSurface"
        ]
        for lod_tag in geometry_lods:
            lod_elem = veg_elem.find(f'.//veg:{lod_tag}', ns)
            if lod_elem is not None:
                geom = parse_lod_multisurface(lod_elem)
                if geom is not None:
                    return geom
        return None

    def compute_lod_height(veg_elem):
        """
        Fallback: compute vegetation height from Z values in any gml:posList
        under the available LOD geometry elements. Returns (max_z - min_z)
        if any Z values are found; otherwise None.
        """
        z_values = []
        geometry_lods = [
            "lod0Geometry", "lod1Geometry", "lod2Geometry", "lod3Geometry", "lod4Geometry",
            "lod0MultiSurface", "lod1MultiSurface", "lod2MultiSurface", "lod3MultiSurface", "lod4MultiSurface"
        ]
        try:
            for lod_tag in geometry_lods:
                lod_elem = veg_elem.find(f'.//veg:{lod_tag}', ns)
                if lod_elem is None:
                    continue
                for pos_list in lod_elem.findall('.//gml:posList', ns):
                    if pos_list.text is None:
                        continue
                    coords_text = pos_list.text.strip().split()
                    # Expect triplets (x,y,z) or (lat,lon,z). Z should be each 3rd value
                    for i in range(2, len(coords_text), 3):
                        try:
                            z = float(coords_text[i])
                            if not np.isinf(z) and not np.isnan(z):
                                z_values.append(z)
                        except Exception:
                            continue
            if z_values:
                return float(max(z_values) - min(z_values))
        except Exception:
            pass
        return None

    # 1) PlantCover
    for plant_cover in root.findall('.//veg:PlantCover', ns):
        cover_id = plant_cover.get('{http://www.opengis.net/gml}id')
        avg_height_elem = plant_cover.find('.//veg:averageHeight', ns)
        if avg_height_elem is not None and avg_height_elem.text:
            try:
                vegetation_height = float(avg_height_elem.text)
                # Treat sentinel values like -9999 as missing
                if vegetation_height <= -9998:
                    vegetation_height = None
            except:
                vegetation_height = None
        else:
            vegetation_height = None

        # Fallback to geometry-derived height if needed
        if vegetation_height is None:
            derived_h = compute_lod_height(plant_cover)
            if derived_h is not None:
                vegetation_height = derived_h

        geometry = get_veg_geometry(plant_cover)
        if geometry is not None and not geometry.is_empty:
            vegetation_elements.append({
                'object_type': 'PlantCover',
                'vegetation_id': cover_id,
                'height': vegetation_height,
                'geometry': geometry,
                'source_file': Path(file_path).name
            })

    # 2) SolitaryVegetationObject
    for solitary in root.findall('.//veg:SolitaryVegetationObject', ns):
        veg_id = solitary.get('{http://www.opengis.net/gml}id')
        height_elem = solitary.find('.//veg:height', ns)
        if height_elem is not None and height_elem.text:
            try:
                veg_height = float(height_elem.text)
                # Treat sentinel values like -9999 as missing
                if veg_height <= -9998:
                    veg_height = None
            except:
                veg_height = None
        else:
            veg_height = None

        # Fallback to geometry-derived height if attribute is missing/unparseable
        if veg_height is None:
            derived_h = compute_lod_height(solitary)
            if derived_h is not None:
                veg_height = derived_h

        geometry = get_veg_geometry(solitary)
        if geometry is not None and not geometry.is_empty:
            vegetation_elements.append({
                'object_type': 'SolitaryVegetationObject',
                'vegetation_id': veg_id,
                'height': veg_height,
                'geometry': geometry,
                'source_file': Path(file_path).name
            })

    if vegetation_elements:
        print(f"Extracted {len(vegetation_elements)} vegetation objects from {Path(file_path).name}")
    return vegetation_elements


def extract_building_footprint(building, namespaces):
    """
    Extract building footprint from CityGML building element.

    Args:
        building (Element): XML element representing a building.
        namespaces (dict): XML namespace mappings.

    Returns:
        tuple: (pos_list, ground_elevation) where:
            - pos_list: XML element containing footprint coordinates
            - ground_elevation: Ground level elevation if available

    Notes:
        - Tries multiple LOD representations (LOD0-LOD2)
        - For LOD1/LOD2 solids, finds the bottom face
        - Returns None if no valid footprint found
    """
    lod_tags = [
        # LOD0
        './/bldg:lod0FootPrint//gml:MultiSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
        './/bldg:lod0RoofEdge//gml:MultiSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
        './/bldg:lod0Solid//gml:Solid//gml:exterior//gml:CompositeSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
        
        # LOD1
        './/bldg:lod1Solid//gml:Solid//gml:exterior//gml:CompositeSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
        
        # LOD2
        './/bldg:lod2Solid//gml:Solid//gml:exterior//gml:CompositeSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
        
        # fallback
        './/gml:MultiSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
        './/gml:Polygon//gml:exterior//gml:LinearRing//gml:posList'
    ]
    
    for tag in lod_tags:
        pos_list_elements = building.findall(tag, namespaces)
        if pos_list_elements:
            # If in LOD1/LOD2 solid, we look for the bottom face
            if 'lod1Solid' in tag or 'lod2Solid' in tag or 'lod0Solid' in tag:
                lowest_z = float('inf')
                footprint_pos_list = None
                for pos_list_elem in pos_list_elements:
                    coords_text = pos_list_elem.text.strip().split()
                    z_values = [float(coords_text[i+2]) 
                                for i in range(0, len(coords_text), 3) 
                                if i+2 < len(coords_text)]
                    if z_values and all(z == z_values[0] for z in z_values) and z_values[0] < lowest_z:
                        lowest_z = z_values[0]
                        footprint_pos_list = pos_list_elem
                if footprint_pos_list:
                    return footprint_pos_list, lowest_z
            else:
                # For simpler LOD0 footprints, just return the first
                return pos_list_elements[0], None
    return None, None


def process_citygml_file(file_path):
    """
    Process a CityGML file to extract all relevant features.

    Args:
        file_path (str): Path to CityGML file.

    Returns:
        tuple: (buildings, terrain_elements, vegetation_elements) where each is a list
               of dictionaries containing feature information.

    Notes:
        - Processes buildings, terrain, and vegetation features
        - Validates all geometries
        - Handles coordinate transformations
        - Includes error handling and reporting
    """
    buildings = []
    terrain_elements = []
    vegetation_elements = []

    # Default/fallback namespaces (used if not present in the file)
    fallback_namespaces = {
        'core': 'http://www.opengis.net/citygml/2.0',
        'bldg': 'http://www.opengis.net/citygml/building/2.0',
        'gml': 'http://www.opengis.net/gml',
        'uro': 'https://www.geospatial.jp/iur/uro/3.0',
        'dem': 'http://www.opengis.net/citygml/relief/2.0',
        'veg': 'http://www.opengis.net/citygml/vegetation/2.0'
    }

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        # Build namespaces dynamically from the file, falling back to defaults.
        nsmap = root.nsmap or {}

        def pick_ns(prefix, keyword=None, fallback_key=None):
            # Try explicit prefix first
            uri = nsmap.get(prefix)
            if uri:
                return uri
            # Try to discover by keyword in URI (e.g., 'vegetation', 'building', 'relief')
            if keyword:
                for v in nsmap.values():
                    if isinstance(v, str) and keyword in v:
                        return v
            # Fallback to defaults
            return fallback_namespaces[fallback_key or prefix]

        namespaces = {
            'core': pick_ns('core', keyword='citygml', fallback_key='core'),
            'bldg': pick_ns('bldg', keyword='building', fallback_key='bldg'),
            'gml': pick_ns('gml', keyword='gml', fallback_key='gml'),
            'uro': pick_ns('uro', keyword='iur/uro', fallback_key='uro'),
            'dem': pick_ns('dem', keyword='relief', fallback_key='dem'),
            # Accept CityGML 2.0 or 3.0 vegetation namespaces
            'veg': pick_ns('veg', keyword='vegetation', fallback_key='veg')
        }

        # Extract Buildings
        for building in root.findall('.//bldg:Building', namespaces):
            building_id = building.get('{http://www.opengis.net/gml}id')
            
            measured_height = building.find('.//bldg:measuredHeight', namespaces)
            height = float(measured_height.text) if measured_height is not None and measured_height.text else None
            
            storeys = building.find('.//bldg:storeysAboveGround', namespaces)
            num_storeys = int(storeys.text) if storeys is not None and storeys.text else None
            
            pos_list, ground_elevation = extract_building_footprint(building, namespaces)
            if pos_list is not None:
                try:
                    coords_text = pos_list.text.strip().split()
                    coords = []
                    
                    # Decide if we have (x,y) pairs or (x,y,z) triplets
                    coord_step = 3 if (len(coords_text) % 3) == 0 else 2

                    for i in range(0, len(coords_text), coord_step):
                        if i + coord_step - 1 < len(coords_text):
                            lon = float(coords_text[i])
                            lat = float(coords_text[i+1])
                            if coord_step == 3 and i+2 < len(coords_text):
                                z = float(coords_text[i+2])
                                if ground_elevation is None:
                                    ground_elevation = z
                            if not np.isinf(lon) and not np.isinf(lat):
                                coords.append((lon, lat))

                    if len(coords) >= 3 and validate_coords(coords):
                        polygon = Polygon(coords)
                        if polygon.is_valid:
                            buildings.append({
                                'building_id': building_id,
                                'height': height,
                                'storeys': num_storeys,
                                'ground_elevation': ground_elevation,
                                'geometry': polygon,
                                'source_file': Path(file_path).name
                            })
                except (ValueError, IndexError) as e:
                    print(f"Error processing building {building_id} footprint in {Path(file_path).name}: {e}")

        # Extract Terrain
        terrain_elements = extract_terrain_info(file_path, namespaces)

        # Extract Vegetation
        vegetation_elements = extract_vegetation_info(file_path, namespaces)

        print(f"Processed {Path(file_path).name}: "
              f"{len(buildings)} buildings, {len(terrain_elements)} terrain elements, "
              f"{len(vegetation_elements)} vegetation objects")

    except Exception as e:
        print(f"Error processing file {Path(file_path).name}: {e}")

    return buildings, terrain_elements, vegetation_elements


def parse_file(file_path, file_type=None):
    """
    Parse a file based on its type (auto-detected or specified).

    Args:
        file_path (str): Path to file to parse.
        file_type (str, optional): Force specific file type parsing.
            Valid values: 'citygml', 'geojson', 'xml'

    Returns:
        tuple: (buildings, terrain_elements, vegetation_elements) lists.

    Notes:
        - Auto-detects file type from extension if not specified
        - Currently fully implements CityGML parsing only
        - Returns empty lists for unsupported types
    """
    if file_type is None:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.gml':
            file_type = 'citygml'
        elif file_ext == '.xml':
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                ns = root.nsmap
                if any('citygml' in ns_uri.lower() for ns_uri in ns.values()):
                    file_type = 'citygml'
                else:
                    file_type = 'xml'
            except:
                file_type = 'xml'
        elif file_ext in ['.json', '.geojson']:
            file_type = 'geojson'
        else:
            print(f"Unsupported file type: {file_ext}")
            return None, None, None
    
    if file_type == 'citygml':
        return process_citygml_file(file_path)
    elif file_type == 'geojson':
        print(f"GeoJSON processing not implemented for {file_path}")
        return [], [], []
    elif file_type == 'xml':
        print(f"Generic XML processing not implemented for {file_path}")
        return [], [], []
    else:
        print(f"Unsupported file type: {file_type}")
        return [], [], []


def swap_coordinates_if_needed(gdf, geometry_col='geometry'):
    """
    Ensure correct coordinate order in GeoDataFrame geometries.

    Args:
        gdf (GeoDataFrame): Input GeoDataFrame.
        geometry_col (str): Name of geometry column.

    Returns:
        list: List of geometries with corrected coordinate order.

    Notes:
        - Assumes input is EPSG:6697 but may be in lat-lon order
        - Handles Polygon, MultiPolygon, and Point geometries
        - Returns geometries in lon-lat order
    """
    swapped_geometries = []
    for geom in gdf[geometry_col]:
        # If it's a Polygon or MultiPolygon, use swap_coordinates()
        if isinstance(geom, (Polygon, MultiPolygon)):
            swapped_geometries.append(swap_coordinates(geom))
        elif isinstance(geom, Point):
            swapped_geometries.append(Point(geom.y, geom.x))
        else:
            swapped_geometries.append(geom)
    return swapped_geometries


def load_buid_dem_veg_from_citygml(url=None, 
                              base_dir='.', 
                              citygml_path=None,
                              rectangle_vertices=None,
                              ssl_verify=True,
                              ca_bundle=None,
                              timeout=60):
    """
    Load and process PLATEAU data from URL or local files.

    Args:
        url (str, optional): URL to download PLATEAU data from.
        base_dir (str): Base directory for file operations.
        citygml_path (str, optional): Path to local CityGML files.
        rectangle_vertices (list, optional): List of (lon, lat) tuples defining
            a bounding rectangle for filtering tiles.

    Returns:
        tuple: (gdf_buildings, gdf_terrain, gdf_vegetation) GeoDataFrames
            containing processed features.

    Notes:
        - Can process from URL (download & extract) or local files
        - Optionally filters tiles by geographic extent
        - Handles coordinate transformations
        - Creates GeoDataFrames with proper CRS
    """
    all_buildings = []
    all_terrain = []
    all_vegetation = []
    
    # Build the rectangle polygon if given
    rectangle_polygon = None
    if rectangle_vertices and len(rectangle_vertices) >= 3:
        rectangle_polygon = Polygon(rectangle_vertices)
    
    if url:
        citygml_path, foldername = download_and_extract_zip(
            url, extract_to=base_dir, ssl_verify=ssl_verify, ca_bundle=ca_bundle, timeout=timeout
        )
    elif citygml_path:
        foldername = os.path.basename(citygml_path)
    else:
        print("Either url or citygml_path must be specified")
        return None, None, None

    # Identify CityGML files in typical folder structure
    try:
        citygml_dir = os.path.join(citygml_path, 'udx')
        if not os.path.exists(citygml_dir):
            citygml_dir_2 = os.path.join(citygml_path, foldername, 'udx')
            if os.path.exists(citygml_dir_2):
                citygml_dir = citygml_dir_2
        
        # Potential sub-folders
        bldg_dir = os.path.join(citygml_dir, 'bldg')
        dem_dir = os.path.join(citygml_dir, 'dem')
        veg_dir = os.path.join(citygml_dir, 'veg')
        
        citygml_files = []
        for folder in [bldg_dir, dem_dir, veg_dir, citygml_dir]:
            if os.path.exists(folder):
                citygml_files += [
                    os.path.join(folder, f) for f in os.listdir(folder) 
                    if f.endswith(('.gml', '.xml'))
                ]
        
        print(f"Found {len(citygml_files)} CityGML files to process")

        for file_path in tqdm(citygml_files, desc="Processing files"):
            filename = os.path.basename(file_path)

            # If a rectangle is given, check tile intersection
            if rectangle_polygon is not None:
                try:
                    tile_polygon_lonlat = get_tile_polygon_from_filename(filename)  # returns [(lon, lat), ...]
                    tile_polygon = Polygon(tile_polygon_lonlat)
                    
                    # If no overlap, skip processing
                    if not tile_polygon.intersects(rectangle_polygon):
                        continue
                except Exception as e:
                    # If we cannot parse a tile boundary, skip or handle as you wish
                    print(f"Warning: could not get tile boundary from {filename}: {e}, extracting the tile whether it is in the rectangle or not.")
                    # continue

            # Parse the file
            buildings, terrain_elements, vegetation_elements = parse_file(file_path)
            all_buildings.extend(buildings)
            all_terrain.extend(terrain_elements)
            all_vegetation.extend(vegetation_elements)
    
    except Exception as e:
        print(f"Error finding CityGML files: {e}")
        return None, None, None

    # Convert to GeoDataFrames
    gdf_buildings = None
    gdf_terrain = None
    gdf_vegetation = None

    if all_buildings:
        gdf_buildings = gpd.GeoDataFrame(all_buildings, geometry='geometry')
        gdf_buildings.set_crs(epsg=6697, inplace=True)  # or "EPSG:4326", depending on your data
        # Swap if needed
        gdf_buildings['geometry'] = swap_coordinates_if_needed(gdf_buildings, geometry_col='geometry')
        # Add an ID
        gdf_buildings['id'] = range(len(gdf_buildings))

    if all_terrain:
        gdf_terrain = gpd.GeoDataFrame(all_terrain, geometry='geometry')
        gdf_terrain.set_crs(epsg=6697, inplace=True)
        gdf_terrain['geometry'] = swap_coordinates_if_needed(gdf_terrain, geometry_col='geometry')

    if all_vegetation:
        gdf_vegetation = gpd.GeoDataFrame(all_vegetation, geometry='geometry')
        gdf_vegetation.set_crs(epsg=6697, inplace=True)
        gdf_vegetation['geometry'] = swap_coordinates_if_needed(gdf_vegetation, geometry_col='geometry')

    return gdf_buildings, gdf_terrain, gdf_vegetation


def process_single_file(file_path):
    """
    Process a single CityGML file for testing purposes.

    Args:
        file_path (str): Path to CityGML file.

    Returns:
        tuple: (buildings, terrain, vegetation) lists of extracted features.

    Notes:
        - Useful for testing and debugging
        - Saves building data to GeoJSON if successful
        - Prints processing statistics
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in ['.gml', '.xml']:
        buildings, terrain, vegetation = parse_file(file_path)
        print(f"\nProcessed {file_path}:")
        print(f"  - {len(buildings)} buildings extracted")
        print(f"  - {len(terrain)} terrain elements extracted")
        print(f"  - {len(vegetation)} vegetation objects extracted")
        
        # Example: create building GeoDataFrame and save to GeoJSON
        if buildings:
            gdf_buildings = gpd.GeoDataFrame(buildings, geometry='geometry')
            gdf_buildings.set_crs(epsg=6697, inplace=True)
            output_file = os.path.splitext(file_path)[0] + "_buildings.geojson"
            gdf_buildings.to_file(output_file, driver='GeoJSON')
            print(f"Buildings saved to {output_file}")
        
        return buildings, terrain, vegetation
    else:
        print(f"Unsupported file type: {file_ext}")
        return None, None, None