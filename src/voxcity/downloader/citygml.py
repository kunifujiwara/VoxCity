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

def download_and_extract_zip(url, extract_to='.'):
    """
    Download and extract a zip file from a URL
    """
    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Extract the base name of the zip file from the URL
        parsed_url = urlparse(url)
        zip_filename = os.path.basename(parsed_url.path)
        folder_name = os.path.splitext(zip_filename)[0]  # Remove the .zip extension

        # Create the extraction directory
        extraction_path = os.path.join(extract_to, folder_name)
        os.makedirs(extraction_path, exist_ok=True)

        # Create a BytesIO object from the response content
        zip_file = io.BytesIO(response.content)

        # Open the zip file
        with zipfile.ZipFile(zip_file) as z:
            # Extract all the contents of the zip file to the specified directory
            z.extractall(extraction_path)
            print(f"Extracted to {extraction_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")

    return extraction_path, folder_name


def validate_coords(coords):
    """
    Validate that coordinates are not infinite or NaN
    """
    return all(not np.isinf(x) and not np.isnan(x) for coord in coords for x in coord)


def swap_coordinates(polygon):
    """
    Swap coordinates in a polygon (lat/lon to lon/lat or vice versa)
    """
    if isinstance(polygon, MultiPolygon):
        # Handle MultiPolygon objects
        new_polygons = []
        for geom in polygon.geoms:
            coords = list(geom.exterior.coords)
            swapped_coords = [(y, x) for x, y in coords]
            new_polygons.append(Polygon(swapped_coords))
        return MultiPolygon(new_polygons)
    else:
        # Handle regular Polygon objects
        coords = list(polygon.exterior.coords)
        swapped_coords = [(y, x) for x, y in coords]
        return Polygon(swapped_coords)


def extract_terrain_info(file_path, namespaces):
    """
    Extract terrain elevation information from a CityGML file
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

                # Extract triangulated surface
                triangles = tin.findall('.//gml:Triangle', namespaces)
                for i, triangle in enumerate(triangles):
                    # Extract the coordinates of each triangle
                    pos_lists = triangle.findall('.//gml:posList', namespaces)

                    for pos_list in pos_lists:
                        try:
                            # Process the coordinates
                            coords_text = pos_list.text.strip().split()
                            coords = []
                            elevations = []

                            # Process coordinates in triplets (x, y, z)
                            for j in range(0, len(coords_text), 3):
                                if j + 2 < len(coords_text):
                                    x = float(coords_text[j])
                                    y = float(coords_text[j + 1])
                                    z = float(coords_text[j + 2])  # Elevation

                                    if not np.isinf(x) and not np.isinf(y) and not np.isinf(z):
                                        coords.append((x, y))
                                        elevations.append(z)

                            if len(coords) >= 3 and validate_coords(coords):
                                polygon = Polygon(coords)
                                if polygon.is_valid:
                                    # Calculate centroid for point representation
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
                                    y = float(coords_text[j + 1])
                                    z = float(coords_text[j + 2])

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
    Extract vegetation features (PlantCover, SolitaryVegetationObject)
    from a CityGML file, handling LOD0..LOD3 geometry and MultiSurface/CompositeSurface.
    """
    vegetation_elements = []

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing CityGML file {Path(file_path).name}: {e}")
        return vegetation_elements

    # ----------------------------------------------------------------------------
    # Helper: parse all polygons from a <gml:MultiSurface> or <veg:lodXMultiSurface>
    # ----------------------------------------------------------------------------
    def parse_lod_multisurface(lod_elem):
        """Return a Shapely (Multi)Polygon from gml:Polygon elements under lod_elem."""
        polygons = []
        # Find all Polygons (including nested in CompositeSurface)
        for poly_node in lod_elem.findall('.//gml:Polygon', namespaces):
            ring_node = poly_node.find('.//gml:exterior//gml:LinearRing//gml:posList', namespaces)
            if ring_node is None or ring_node.text is None:
                continue

            # Parse coordinate text
            coords_text = ring_node.text.strip().split()
            coords = []
            # Typically posList is in triplets: (x, y, z)
            for i in range(0, len(coords_text), 3):
                try:
                    x = float(coords_text[i])
                    y = float(coords_text[i+1])
                    # z = float(coords_text[i+2])  # if you want z
                    coords.append((x, y))
                except:
                    # Skip any parse error or incomplete coordinate
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

    # ----------------------------------------------------------------------------
    # Helper: retrieve geometry from all LOD tags
    # ----------------------------------------------------------------------------
    def get_veg_geometry(veg_elem):
        """
        Search for geometry under lod0Geometry, lod1Geometry, lod2Geometry,
        lod3Geometry, lod4Geometry, as well as lod0MultiSurface ... lod3MultiSurface, etc.
        Return a Shapely geometry (Polygon or MultiPolygon) if found.
        """
        geometry_lods = [
            "lod0Geometry", "lod1Geometry", "lod2Geometry", "lod3Geometry", "lod4Geometry",
            "lod0MultiSurface", "lod1MultiSurface", "lod2MultiSurface", "lod3MultiSurface", "lod4MultiSurface"
        ]

        for lod_tag in geometry_lods:
            # e.g. .//veg:lod3Geometry
            lod_elem = veg_elem.find(f'.//veg:{lod_tag}', namespaces)
            if lod_elem is not None:
                geom = parse_lod_multisurface(lod_elem)
                if geom is not None:
                    return geom

        return None

    # ----------------------------------------------------------------------------
    # 1) PlantCover
    # ----------------------------------------------------------------------------
    for plant_cover in root.findall('.//veg:PlantCover', namespaces):
        cover_id = plant_cover.get('{http://www.opengis.net/gml}id')
        # averageHeight (if present)
        avg_height_elem = plant_cover.find('.//veg:averageHeight', namespaces)
        if avg_height_elem is not None and avg_height_elem.text:
            try:
                vegetation_height = float(avg_height_elem.text)
            except:
                vegetation_height = None
        else:
            vegetation_height = None

        # parse geometry from LOD0..LOD3
        geometry = get_veg_geometry(plant_cover)

        if geometry is not None and not geometry.is_empty:
            vegetation_elements.append({
                'object_type': 'PlantCover',
                'vegetation_id': cover_id,
                'height': vegetation_height,
                'geometry': geometry,
                'source_file': Path(file_path).name
            })

    # ----------------------------------------------------------------------------
    # 2) SolitaryVegetationObject
    # ----------------------------------------------------------------------------
    for solitary in root.findall('.//veg:SolitaryVegetationObject', namespaces):
        veg_id = solitary.get('{http://www.opengis.net/gml}id')
        height_elem = solitary.find('.//veg:height', namespaces)
        if height_elem is not None and height_elem.text:
            try:
                veg_height = float(height_elem.text)
            except:
                veg_height = None
        else:
            veg_height = None

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

def process_citygml_file(file_path):
    """
    Process a CityGML file to extract building, terrain, and vegetation information
    """
    buildings = []
    terrain_elements = []
    vegetation_elements = []

    # Namespaces (now includes 'veg')
    namespaces = {
        'core': 'http://www.opengis.net/citygml/2.0',
        'bldg': 'http://www.opengis.net/citygml/building/2.0',
        'gml': 'http://www.opengis.net/gml',
        'uro': 'https://www.geospatial.jp/iur/uro/3.0',
        'dem': 'http://www.opengis.net/citygml/relief/2.0',
        'veg': 'http://www.opengis.net/citygml/vegetation/2.0'
    }

    try:
        # Parse the file once at the start (optional; if you want to share 'root' among sub-extractors)
        tree = ET.parse(file_path)
        root = tree.getroot()

        # --- Extract Building Info (existing approach) ---
        for building in root.findall('.//bldg:Building', namespaces):
            building_id = building.get('{http://www.opengis.net/gml}id')
            measured_height = building.find('.//bldg:measuredHeight', namespaces)
            height = float(measured_height.text) if measured_height is not None else None

            # Extract the footprint (LOD0)
            lod0_roof_edge = building.find('.//bldg:lod0RoofEdge//gml:posList', namespaces)
            if lod0_roof_edge is not None:
                try:
                    pos_list = lod0_roof_edge.text.strip().split()
                    coords = []
                    for i in range(0, len(pos_list), 3):
                        if i + 2 < len(pos_list):
                            lon = float(pos_list[i])
                            lat = float(pos_list[i + 1])
                            elevation = float(pos_list[i + 2])  # z value
                            if not np.isinf(lon) and not np.isinf(lat):
                                coords.append((lon, lat))

                    if len(coords) >= 3 and validate_coords(coords):
                        polygon = Polygon(coords)
                        if polygon.is_valid:
                            buildings.append({
                                'building_id': building_id,
                                'height': height,
                                'ground_elevation': elevation,  # Add ground elevation if relevant
                                'geometry': polygon,
                                'source_file': Path(file_path).name
                            })
                except (ValueError, IndexError) as e:
                    print(f"Error processing building {building_id} in file {Path(file_path).name}: {e}")

        # --- Extract Terrain Info (existing function) ---
        terrain_elements = extract_terrain_info(file_path, namespaces)

        # --- Extract Vegetation Info (new function) ---
        vegetation_elements = extract_vegetation_info(file_path, namespaces)

        print(f"Processed {Path(file_path).name}: "
              f"{len(buildings)} buildings, {len(terrain_elements)} terrain elements, "
              f"{len(vegetation_elements)} vegetation objects")

    except Exception as e:
        print(f"Error processing file {Path(file_path).name}: {e}")

    return buildings, terrain_elements, vegetation_elements

def load_plateau_with_terrain(url, base_dir):
    """
    Load PLATEAU data, extracting Buildings, Terrain, and Vegetation data from CityGML.
    """
    # 1) Download & unzip
    citygml_path, foldername = download_and_extract_zip(url, extract_to=base_dir)

    # 2) Identify CityGML files in typical folder structure
    try:
        citygml_dir = os.path.join(citygml_path, 'udx')
        if not os.path.exists(citygml_dir):
            citygml_dir = os.path.join(citygml_path, foldername, 'udx')

        bldg_dir = os.path.join(citygml_dir, 'bldg')
        dem_dir = os.path.join(citygml_dir, 'dem')

        # NEW: check for vegetation folder
        veg_dir = os.path.join(citygml_dir, 'veg')

        citygml_files = []

        # If there's a building folder, gather .gml from there
        if os.path.exists(bldg_dir):
            citygml_files += [
                os.path.join(bldg_dir, f) for f in os.listdir(bldg_dir) if f.endswith('.gml')
            ]
        else:
            # If no 'bldg' folder, look directly in 'udx'
            citygml_files += [
                os.path.join(citygml_dir, f) for f in os.listdir(citygml_dir) if f.endswith('.gml')
            ]

        # Also gather DEM .gml (terrain)
        if os.path.exists(dem_dir):
            citygml_files += [
                os.path.join(dem_dir, f) for f in os.listdir(dem_dir) if f.endswith('.gml')
            ]

        # ADD THIS: gather VEG .gml (vegetation)
        if os.path.exists(veg_dir):
            citygml_files += [
                os.path.join(veg_dir, f) for f in os.listdir(veg_dir) if f.endswith('.gml')
            ]

        total_files = len(citygml_files)
        print(f"Found {total_files} CityGML files to process")

    except Exception as e:
        print(f"Error finding CityGML files: {e}")
        return None, None, None

    all_buildings = []
    all_terrain = []
    all_vegetation = []

    # 3) Process each CityGML
    for file_path in tqdm(citygml_files, desc="Processing CityGML files"):
        buildings, terrain_elements, vegetation_elements = process_citygml_file(file_path)
        all_buildings.extend(buildings)
        all_terrain.extend(terrain_elements)
        all_vegetation.extend(vegetation_elements)

    # 4) Create GeoDataFrame for Buildings
    if all_buildings:
        gdf_buildings = gpd.GeoDataFrame(all_buildings, geometry='geometry')
        gdf_buildings.set_crs(epsg=6697, inplace=True)

        # Swap coords from (lon, lat) to (lat, lon) if needed
        swapped_geometries = [swap_coordinates(geom) for geom in gdf_buildings.geometry]
        gdf_buildings_swapped = gpd.GeoDataFrame(
            {
                'building_id': gdf_buildings['building_id'],
                'height': gdf_buildings['height'],
                'ground_elevation': gdf_buildings['ground_elevation'],
                'source_file': gdf_buildings['source_file'],
                'geometry': swapped_geometries
            },
            crs='EPSG:6697'
        )

        # Save
        gdf_buildings_swapped['id'] = gdf_buildings_swapped.index
        gdf_buildings_swapped.to_file('all_buildings_with_elevation.geojson', driver='GeoJSON')
        print(f"\nBuildings saved to all_buildings_with_elevation.geojson")
    else:
        gdf_buildings_swapped = None

    # 5) Create GeoDataFrame for Terrain
    if all_terrain:
        gdf_terrain = gpd.GeoDataFrame(all_terrain, geometry='geometry')
        gdf_terrain.set_crs(epsg=6697, inplace=True)

        swapped_geometries = []
        for geom in gdf_terrain.geometry:
            if isinstance(geom, (Polygon, MultiPolygon)):
                swapped_geometries.append(swap_coordinates(geom))
            elif isinstance(geom, Point):
                swapped_geometries.append(Point(geom.y, geom.x))
            else:
                swapped_geometries.append(geom)

        terrain_data = {
            'relief_id': gdf_terrain.get('relief_id', ''),
            'tin_id': gdf_terrain.get('tin_id', ''),
            'triangle_id': gdf_terrain.get('triangle_id', ''),
            'breakline_id': gdf_terrain.get('breakline_id', ''),
            'mass_point_id': gdf_terrain.get('mass_point_id', ''),
            'point_id': gdf_terrain.get('point_id', ''),
            'elevation': gdf_terrain['elevation'],
            'source_file': gdf_terrain['source_file'],
            'geometry': swapped_geometries
        }

        gdf_terrain_swapped = gpd.GeoDataFrame(terrain_data, geometry='geometry', crs='EPSG:6697')
        gdf_terrain_swapped.to_file('terrain_elevation.geojson', driver='GeoJSON')
        print(f"Terrain saved to terrain_elevation.geojson")
    else:
        gdf_terrain_swapped = None

    # 6) Create GeoDataFrame for Vegetation
    if all_vegetation:
        gdf_veg = gpd.GeoDataFrame(all_vegetation, geometry='geometry')
        gdf_veg.set_crs(epsg=6697, inplace=True)

        swapped_geometries = []
        for geom in gdf_veg.geometry:
            if isinstance(geom, (Polygon, MultiPolygon)):
                swapped_geometries.append(swap_coordinates(geom))
            elif isinstance(geom, Point):
                swapped_geometries.append(Point(geom.y, geom.x))
            else:
                swapped_geometries.append(geom)

        vegetation_data = {
            'object_type':    gdf_veg.get('object_type', ''),
            'vegetation_id':  gdf_veg.get('vegetation_id', ''),
            'height':         gdf_veg.get('height', None),
            'avg_elevation':  gdf_veg.get('avg_elevation', None),  # Use .get() with a default
            'source_file':    gdf_veg.get('source_file', ''),
            'geometry':       swapped_geometries
        }
        gdf_vegetation_swapped = gpd.GeoDataFrame(vegetation_data, geometry='geometry', crs='EPSG:6697')
        gdf_vegetation_swapped.to_file('vegetation_elevation.geojson', driver='GeoJSON')
        print(f"Vegetation saved to vegetation_elevation.geojson")
    else:
        gdf_vegetation_swapped = None

    return gdf_buildings_swapped, gdf_terrain_swapped, gdf_vegetation_swapped