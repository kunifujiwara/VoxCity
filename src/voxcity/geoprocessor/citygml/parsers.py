"""
CityGML Parsers
===============

Parsers for CityGML LOD2 building data from various sources:

- **CityGMLParser**: Standard European/German CityGML format (ETRS89_UTM)
- **PLATEAUParser**: Japanese PLATEAU CityGML format (JGD2011/EPSG:6697)

Both parsers extract triangulated 3D geometry that can be voxelized using
the companion voxelizer module.
"""

import os
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

try:
    from shapely.geometry import Polygon, box
    from shapely.validation import make_valid
    from shapely.prepared import prep
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    Polygon = None
    prep = None

# Try to use lxml for faster XML parsing
try:
    import lxml.etree as lxml_ET
    HAS_LXML = True
except ImportError:
    HAS_LXML = False
    lxml_ET = None

from .models import (
    Building,
    PLATEAUBuilding,
    PLATEAUVegetation,
    PLATEAUBridge,
    PLATEAUCityFurniture,
    PLATEAULandUse,
    TerrainTriangle,
)
from .constants import NAMESPACES, PLATEAU_NAMESPACES
from .utils import (
    CoordinateTransformer,
    parse_pos_list,
    triangulate_polygon,
    compute_triangle_normal,
    decode_mesh_code,
    get_mesh_code_from_filename,
)


# =============================================================================
# German/European CityGML Parser
# =============================================================================

class CityGMLParser:
    """
    Parser for standard CityGML files (German/European format).
    
    Extracts LOD2 building geometry from CityGML files. Supports both
    CityGML 1.0 and 2.0 namespace versions.
    
    Example::
    
        parser = CityGMLParser()
        parser.parse_file("path/to/building.gml")
        # or parse entire directory
        parser.parse_directory("path/to/citygml_folder")
        
        # Get triangle data for rendering or voxelization
        vertices, normals, material_ids = parser.get_triangle_data()
    """
    
    def __init__(self, coordinate_transform: bool = True, target_crs: str = 'EPSG:4326'):
        """
        Initialize the parser.
        
        Args:
            coordinate_transform: Whether to transform coordinates.
            target_crs: Target coordinate reference system.
        """
        self.coordinate_transform = coordinate_transform
        self.target_crs = target_crs
        self.triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.normals: List[np.ndarray] = []
        self.building_ids: List[str] = []
        self.bounds_min: Optional[np.ndarray] = None
        self.bounds_max: Optional[np.ndarray] = None
    
    def parse_file(self, filepath: str) -> None:
        """
        Parse a single CityGML file.
        
        Args:
            filepath: Path to the GML file.
        """
        print(f"Parsing: {filepath}")
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find all building elements - try multiple namespace versions
            buildings = root.findall('.//bldg:Building', NAMESPACES)
            if not buildings:
                buildings = root.findall('.//{http://www.opengis.net/citygml/building/1.0}Building')
            if not buildings:
                buildings = root.findall('.//{http://www.opengis.net/citygml/building/2.0}Building')
            
            print(f"  Found {len(buildings)} buildings")
            
            for building in buildings:
                self._parse_building(building)
                
        except Exception as e:
            print(f"  Error parsing file: {e}")
            import traceback
            traceback.print_exc()
    
    def parse_directory(self, directory: str, pattern: str = "*.gml") -> None:
        """
        Parse all CityGML files in a directory.
        
        Args:
            directory: Path to the directory.
            pattern: Glob pattern for GML files.
        """
        dir_path = Path(directory)
        gml_files = list(dir_path.glob(pattern))
        print(f"Found {len(gml_files)} GML files")
        
        for gml_file in gml_files:
            self.parse_file(str(gml_file))
        
        print(f"Total triangles parsed: {len(self.triangles)}")
    
    def _parse_building(self, building: ET.Element) -> None:
        """Parse a single building element."""
        building_id = building.get('{http://www.opengis.net/gml}id', 'unknown')
        
        # Find all posLists from the building
        pos_lists = set()
        for pos_list in building.iter('{http://www.opengis.net/gml}posList'):
            if pos_list.text:
                pos_lists.add(pos_list.text)
        
        for pos_list_text in pos_lists:
            try:
                vertices = parse_pos_list(pos_list_text)
                if len(vertices) >= 3:
                    triangles = triangulate_polygon(vertices)
                    for v0, v1, v2 in triangles:
                        normal = compute_triangle_normal(v0, v1, v2)
                        self.triangles.append((v0, v1, v2))
                        self.normals.append(normal)
                        self.building_ids.append(building_id)
            except Exception:
                continue
    
    def normalize_coordinates(self, center_at_origin: bool = True, scale: float = 1.0,
                              swap_yz: bool = True) -> None:
        """
        Normalize coordinates to be centered at origin and scaled.
        
        Args:
            center_at_origin: Whether to center at origin.
            scale: Scale factor.
            swap_yz: Whether to swap Y and Z (convert from UTM to Y-up rendering coord).
        """
        if len(self.triangles) == 0:
            return
        
        # Collect all vertices
        all_vertices = []
        for v0, v1, v2 in self.triangles:
            all_vertices.extend([v0, v1, v2])
        all_vertices = np.array(all_vertices)
        
        # Calculate bounds
        self.bounds_min = np.min(all_vertices, axis=0)
        self.bounds_max = np.max(all_vertices, axis=0)
        
        # Center on XY only, use minimum Z as ground reference
        center_xy = (self.bounds_min[:2] + self.bounds_max[:2]) / 2
        z_min = self.bounds_min[2]
        center = np.array([center_xy[0], center_xy[1], z_min])
        
        extent = self.bounds_max - self.bounds_min
        max_extent = np.max(extent[:2])
        
        print(f"Original bounds: min={self.bounds_min}, max={self.bounds_max}")
        print(f"Center XY: {center_xy}, Z ground: {z_min}, Max extent: {max_extent}")
        
        if max_extent > 0:
            normalized_triangles = []
            normalized_normals = []
            
            for (v0, v1, v2), normal in zip(self.triangles, self.normals):
                # Center and shift Z
                v0_norm = v0 - center
                v1_norm = v1 - center
                v2_norm = v2 - center
                
                # Scale
                v0_norm = v0_norm / max_extent * scale
                v1_norm = v1_norm / max_extent * scale
                v2_norm = v2_norm / max_extent * scale
                
                if swap_yz:
                    # Convert from UTM (X=East, Y=North, Z=Up) to rendering (X=East, Y=Up, Z=-North)
                    v0_norm = np.array([v0_norm[0], v0_norm[2], -v0_norm[1]])
                    v1_norm = np.array([v1_norm[0], v1_norm[2], -v1_norm[1]])
                    v2_norm = np.array([v2_norm[0], v2_norm[2], -v2_norm[1]])
                    normal = np.array([normal[0], normal[2], -normal[1]])
                
                normalized_triangles.append((v0_norm, v1_norm, v2_norm))
                normalized_normals.append(normal)
            
            self.triangles = normalized_triangles
            self.normals = normalized_normals
            
            # Recalculate bounds
            all_transformed = []
            for v0, v1, v2 in self.triangles:
                all_transformed.extend([v0, v1, v2])
            all_transformed = np.array(all_transformed)
            self.bounds_min = np.min(all_transformed, axis=0)
            self.bounds_max = np.max(all_transformed, axis=0)
            
            print(f"Normalized bounds: min={self.bounds_min}, max={self.bounds_max}")
    
    def add_ground_plane(self, grid_size: int = 50, ground_offset: float = 0.01) -> None:
        """
        Add a terrain mesh based on building base heights.
        
        Creates a height map by sampling the minimum Y at each grid cell.
        Should be called after normalize_coordinates.
        
        Args:
            grid_size: Number of grid cells in each direction.
            ground_offset: Offset below building bases (positive = lower).
        """
        if self.bounds_min is None or self.bounds_max is None or len(self.triangles) == 0:
            return
        
        margin = 1.0
        x_min = self.bounds_min[0] - margin
        x_max = self.bounds_max[0] + margin
        z_min = self.bounds_min[2] - margin
        z_max = self.bounds_max[2] + margin
        
        dx = (x_max - x_min) / grid_size
        dz = (z_max - z_min) / grid_size
        
        # Build height map from building base vertices
        height_map = np.full((grid_size + 1, grid_size + 1), np.inf, dtype=np.float32)
        
        for v0, v1, v2 in self.triangles:
            for v in [v0, v1, v2]:
                gi = int((v[0] - x_min) / dx)
                gj = int((v[2] - z_min) / dz)
                gi = max(0, min(grid_size, gi))
                gj = max(0, min(grid_size, gj))
                if v[1] < height_map[gi, gj]:
                    height_map[gi, gj] = v[1]
        
        # Fill holes using neighbor interpolation
        valid_mask = height_map < np.inf
        if np.any(valid_mask):
            global_min = np.min(height_map[valid_mask])
        else:
            global_min = 0.0
        
        for _ in range(10):
            filled = height_map.copy()
            for i in range(grid_size + 1):
                for j in range(grid_size + 1):
                    if height_map[i, j] == np.inf:
                        neighbors = []
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                ni, nj = i + di, j + dj
                                if 0 <= ni <= grid_size and 0 <= nj <= grid_size:
                                    if height_map[ni, nj] < np.inf:
                                        neighbors.append(height_map[ni, nj])
                        if neighbors:
                            filled[i, j] = np.mean(neighbors)
            height_map = filled
        
        height_map[height_map == np.inf] = global_min
        height_map -= ground_offset
        
        print(f"Terrain mesh: grid={grid_size}x{grid_size}, "
              f"height range=[{np.min(height_map):.4f}, {np.max(height_map):.4f}]")
        
        # Create terrain triangles
        ground_triangles = []
        ground_normals = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                x0 = x_min + i * dx
                x1 = x_min + (i + 1) * dx
                z0 = z_min + j * dz
                z1 = z_min + (j + 1) * dz
                
                h00 = height_map[i, j]
                h10 = height_map[i + 1, j]
                h01 = height_map[i, j + 1]
                h11 = height_map[i + 1, j + 1]
                
                v0 = np.array([x0, h00, z0], dtype=np.float32)
                v1 = np.array([x1, h10, z0], dtype=np.float32)
                v2 = np.array([x1, h11, z1], dtype=np.float32)
                v3 = np.array([x0, h01, z1], dtype=np.float32)
                
                normal1 = compute_triangle_normal(v0, v1, v2)
                normal2 = compute_triangle_normal(v0, v2, v3)
                
                if normal1[1] < 0:
                    normal1 = -normal1
                if normal2[1] < 0:
                    normal2 = -normal2
                
                ground_triangles.append((v0, v1, v2))
                ground_normals.append(normal1)
                ground_triangles.append((v0, v2, v3))
                ground_normals.append(normal2)
        
        print(f"Added {len(ground_triangles)} terrain triangles")
        
        self.triangles.extend(ground_triangles)
        self.normals.extend(ground_normals)
        self.building_ids.extend(['ground'] * len(ground_triangles))
    
    def get_triangle_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get triangle data as numpy arrays.
        
        Returns:
            vertices: (N, 3, 3) array of triangle vertices.
            normals: (N, 3) array of triangle normals.
            material_ids: (N,) array of material IDs (0=building, 1=ground).
        """
        if len(self.triangles) == 0:
            return (np.zeros((0, 3, 3), dtype=np.float32),
                    np.zeros((0, 3), dtype=np.float32),
                    np.zeros((0,), dtype=np.int32))
        
        vertices = np.array([[v0, v1, v2] for v0, v1, v2 in self.triangles], dtype=np.float32)
        normals = np.array(self.normals, dtype=np.float32)
        material_ids = np.array([1 if bid == 'ground' else 0 for bid in self.building_ids], dtype=np.int32)
        
        return vertices, normals, material_ids
    
    def get_stats(self) -> Dict:
        """Get parsing statistics."""
        return {
            'num_triangles': len(self.triangles),
            'num_buildings': len(set(self.building_ids)),
            'bounds_min': self.bounds_min,
            'bounds_max': self.bounds_max,
        }


# =============================================================================
# Japanese PLATEAU Parser
# =============================================================================

class PLATEAUParser:
    """
    Parser for Japanese PLATEAU CityGML data.
    
    Supports parsing:
    - Buildings from udx/bldg/*.gml
    - DEM terrain from udx/dem/*.gml
    - Vegetation from udx/veg/*.gml
    - Bridges from udx/brid/*.gml
    - City furniture from udx/frn/*.gml
    - Land use from udx/luse/*.gml
    
    Coordinates are automatically transformed from lat/lon to local meters.
    
    Example::
    
        parser = PLATEAUParser()
        parser.parse_plateau_directory(
            "path/to/plateau_data",
            parse_buildings=True,
            parse_dem=True,
            parse_vegetation=True
        )
        
        # Access parsed data
        buildings = parser.buildings
        terrain = parser.terrain_triangles
    """
    
    def __init__(self):
        """Initialize the parser."""
        self.buildings: List[PLATEAUBuilding] = []
        self.terrain_triangles: List[TerrainTriangle] = []
        self.vegetation: List[PLATEAUVegetation] = []
        self.bridges: List[PLATEAUBridge] = []
        self.city_furniture: List[PLATEAUCityFurniture] = []
        self.land_use: List[PLATEAULandUse] = []
        
        self.bounds_min: Optional[np.ndarray] = None
        self.bounds_max: Optional[np.ndarray] = None
        self.transformer: Optional[CoordinateTransformer] = None
        
        self.latlon_bounds_min: Optional[np.ndarray] = None
        self.latlon_bounds_max: Optional[np.ndarray] = None
        
        self._filter_polygon = None  # Shapely polygon for spatial filtering
        self._prepared_filter = None  # Prepared geometry for faster contains
        self._filter_bounds = None  # Bounding box (minx, miny, maxx, maxy)
    
    def _point_in_filter(self, lon: float, lat: float) -> bool:
        """
        Fast check if a point is within the filter polygon.
        Uses bounding box pre-check + prepared geometry for speed.
        """
        if self._filter_bounds is None:
            return True  # No filter
        
        # Fast bounding box pre-check
        minx, miny, maxx, maxy = self._filter_bounds
        if lon < minx or lon > maxx or lat < miny or lat > maxy:
            return False
        
        # Use prepared geometry if available (10-100x faster)
        if self._prepared_filter is not None:
            from shapely.geometry import Point
            return self._prepared_filter.contains(Point(lon, lat))
        
        # Fallback to regular contains check
        if self._filter_polygon is not None:
            from shapely.geometry import Point
            return self._filter_polygon.contains(Point(lon, lat))
        
        return True
    
    def _get_namespaces(self, root) -> Dict[str, str]:
        """Get namespaces from XML root, with fallback to defaults."""
        nsmap = root.nsmap if hasattr(root, 'nsmap') else {}
        ns = PLATEAU_NAMESPACES.copy()
        
        for prefix, uri in nsmap.items():
            if prefix and uri:
                if 'building' in uri:
                    ns['bldg'] = uri
                elif 'relief' in uri:
                    ns['dem'] = uri
                elif 'vegetation' in uri:
                    ns['veg'] = uri
                elif 'citygml' in uri and 'building' not in uri and 'relief' not in uri:
                    ns['core'] = uri
                elif prefix == 'gml' or 'opengis.net/gml' in uri:
                    ns['gml'] = uri
        
        return ns
    
    def parse_plateau_directory(self,
                                base_path: str,
                                rectangle_vertices: Optional[List[Tuple[float, float]]] = None,
                                parse_buildings: bool = True,
                                parse_dem: bool = True,
                                parse_vegetation: bool = False,
                                parse_bridges: bool = False,
                                parse_city_furniture: bool = False,
                                parse_land_use: bool = False) -> None:
        """
        Parse all PLATEAU CityGML files in a directory.
        
        Args:
            base_path: Path to PLATEAU data folder (containing udx/ subfolder).
            rectangle_vertices: Optional bounding rectangle [(lon, lat), ...] to filter tiles.
            parse_buildings: Whether to parse building files.
            parse_dem: Whether to parse DEM files.
            parse_vegetation: Whether to parse vegetation files.
            parse_bridges: Whether to parse bridge files.
            parse_city_furniture: Whether to parse city furniture files.
            parse_land_use: Whether to parse land use files.
        """
        print(f"Parsing PLATEAU directory: {base_path}")
        
        udx_path = os.path.join(base_path, 'udx')
        if not os.path.exists(udx_path):
            if os.path.basename(base_path) == 'udx':
                udx_path = base_path
            else:
                raise ValueError(f"Could not find udx folder in {base_path}")
        
        # Build rectangle polygon for filtering with prepared geometry for faster queries
        rectangle_polygon = None
        self._filter_polygon = None  # Store for per-element filtering
        self._prepared_filter = None  # Prepared geometry for faster contains checks
        self._filter_bounds = None  # Bounding box for fast pre-filtering
        if rectangle_vertices and HAS_SHAPELY:
            rectangle_polygon = Polygon(rectangle_vertices)
            # Buffer slightly to include edge triangles
            self._filter_polygon = rectangle_polygon.buffer(0.001)  # ~100m buffer in degrees
            # Prepare geometry for 10-100x faster contains checks
            if prep is not None:
                self._prepared_filter = prep(self._filter_polygon)
            # Get bounding box for fast pre-filtering
            self._filter_bounds = self._filter_polygon.bounds  # (minx, miny, maxx, maxy)
        
        # Define paths
        paths = {
            'bldg': os.path.join(udx_path, 'bldg'),
            'dem': os.path.join(udx_path, 'dem'),
            'veg': os.path.join(udx_path, 'veg'),
            'brid': os.path.join(udx_path, 'brid'),
            'frn': os.path.join(udx_path, 'frn'),
            'luse': os.path.join(udx_path, 'luse'),
        }
        
        def should_parse_tile(filename: str) -> bool:
            if not rectangle_polygon:
                return True
            mesh_code = get_mesh_code_from_filename(filename)
            if mesh_code:
                try:
                    tile_coords = decode_mesh_code(mesh_code)
                    tile_polygon = Polygon(tile_coords)
                    return tile_polygon.intersects(rectangle_polygon)
                except Exception:
                    pass
            return True
        
        # Parse each type
        if parse_buildings and os.path.exists(paths['bldg']):
            files = sorted([f for f in os.listdir(paths['bldg']) if f.endswith('.gml')])
            print(f"  Found {len(files)} building files")
            for filename in files:
                if should_parse_tile(filename):
                    self._parse_building_file(os.path.join(paths['bldg'], filename))
        
        if parse_dem and os.path.exists(paths['dem']):
            files = sorted([f for f in os.listdir(paths['dem']) if f.endswith('.gml')])
            print(f"  Found {len(files)} DEM files")
            for filename in files:
                if should_parse_tile(filename):
                    self._parse_dem_file(os.path.join(paths['dem'], filename))
        
        if parse_vegetation and os.path.exists(paths['veg']):
            files = sorted([f for f in os.listdir(paths['veg']) if f.endswith('.gml')])
            print(f"  Found {len(files)} vegetation files")
            for filename in files:
                if should_parse_tile(filename):
                    self._parse_vegetation_file(os.path.join(paths['veg'], filename))
        
        if parse_bridges and os.path.exists(paths['brid']):
            files = sorted([f for f in os.listdir(paths['brid']) if f.endswith('.gml')])
            print(f"  Found {len(files)} bridge files")
            for filename in files:
                if should_parse_tile(filename):
                    self._parse_bridge_file(os.path.join(paths['brid'], filename))
        
        if parse_city_furniture and os.path.exists(paths['frn']):
            files = sorted([f for f in os.listdir(paths['frn']) if f.endswith('.gml')])
            print(f"  Found {len(files)} city furniture files")
            for filename in files:
                if should_parse_tile(filename):
                    self._parse_city_furniture_file(os.path.join(paths['frn'], filename))
        
        if parse_land_use and os.path.exists(paths['luse']):
            files = sorted([f for f in os.listdir(paths['luse']) if f.endswith('.gml')])
            print(f"  Found {len(files)} land use files")
            for filename in files:
                if should_parse_tile(filename):
                    self._parse_land_use_file(os.path.join(paths['luse'], filename))
        
        # Transform coordinates
        self._setup_transformer()
        self._transform_all_coordinates()
        self._compute_scene_bounds()
        
        print(f"\nParsing complete:")
        print(f"  Buildings: {len(self.buildings)}")
        print(f"  Terrain triangles: {len(self.terrain_triangles)}")
        print(f"  Vegetation objects: {len(self.vegetation)}")
        print(f"  Bridges: {len(self.bridges)}")
        print(f"  City furniture: {len(self.city_furniture)}")
        print(f"  Land use polygons: {len(self.land_use)}")
        if self.bounds_min is not None:
            print(f"  Bounds (local m): {self.bounds_min} to {self.bounds_max}")
    
    def _parse_building_file(self, filepath: str) -> None:
        """Parse a single building GML file with optimized filtering."""
        print(f"    Parsing buildings: {os.path.basename(filepath)}")
        
        try:
            # Use lxml if available for faster parsing
            if HAS_LXML:
                tree = lxml_ET.parse(filepath)
            else:
                tree = ET.parse(filepath)
            root = tree.getroot()
            ns = self._get_namespaces(root)
            
            buildings = root.findall('.//{http://www.opengis.net/citygml/building/2.0}Building')
            if not buildings:
                buildings = root.findall('.//bldg:Building', ns)
            
            extracted = 0
            filtered = 0
            for building_elem in buildings:
                building = self._extract_building(building_elem, ns)
                if building and building.triangles:
                    # Filter by rectangle using fast method
                    if self._filter_bounds is not None and HAS_SHAPELY:
                        # Compute building centroid from triangles
                        all_verts = []
                        for tri in building.triangles:
                            all_verts.extend(tri)
                        if all_verts:
                            centroid = np.mean(all_verts, axis=0)
                            # Use fast filter method (lon, lat)
                            if not self._point_in_filter(centroid[1], centroid[0]):
                                filtered += 1
                                continue
                    self.buildings.append(building)
                    extracted += 1
            
            if filtered > 0:
                print(f"      Extracted {extracted} buildings (filtered {filtered})")
            else:
                print(f"      Extracted {extracted} buildings")
            
        except Exception as e:
            print(f"      Error: {e}")
    
    def _extract_building(self, building_elem: ET.Element, ns: Dict[str, str]) -> Optional[PLATEAUBuilding]:
        """Extract building geometry from XML element."""
        building_id = building_elem.get('{http://www.opengis.net/gml}id', 'unknown')
        building = PLATEAUBuilding(id=building_id)
        
        # Get measured height
        height_elem = building_elem.find('.//bldg:measuredHeight', ns)
        if height_elem is not None and height_elem.text:
            try:
                building.measured_height = float(height_elem.text)
            except ValueError:
                pass
        
        # Extract all posLists
        for pos_list in building_elem.iter('{http://www.opengis.net/gml}posList'):
            if pos_list.text:
                try:
                    vertices = self._parse_pos_list(pos_list.text)
                    if len(vertices) >= 3:
                        triangles = self._triangulate_polygon(vertices)
                        building.triangles.extend(triangles)
                except Exception:
                    continue
        
        return building if building.triangles else None
    
    def _parse_dem_file(self, filepath: str) -> None:
        """Parse a DEM GML file with optimized filtering."""
        print(f"    Parsing DEM: {os.path.basename(filepath)}")
        
        try:
            # Use lxml if available for faster parsing
            if HAS_LXML:
                tree = lxml_ET.parse(filepath)
            else:
                tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Collect all triangles first, then batch filter
            all_verts = []
            all_centroids = []
            
            for tin in root.iter('{http://www.opengis.net/citygml/relief/2.0}TINRelief'):
                for triangle in tin.iter('{http://www.opengis.net/gml}Triangle'):
                    pos_lists = triangle.findall('.//{http://www.opengis.net/gml}posList')
                    for pos_list in pos_lists:
                        if pos_list.text:
                            try:
                                vertices = self._parse_pos_list(pos_list.text)
                                if len(vertices) >= 3:
                                    verts = vertices[:3]
                                    centroid = np.mean(verts, axis=0)
                                    all_verts.append(verts)
                                    all_centroids.append(centroid)
                            except Exception:
                                continue
            
            if not all_centroids:
                print(f"      No terrain triangles found")
                return
            
            # Vectorized filtering using numpy
            centroids = np.array(all_centroids)  # Shape: (N, 3) - lat, lon, elev
            
            if self._filter_bounds is not None:
                minx, miny, maxx, maxy = self._filter_bounds
                # Fast vectorized bounding box filter (lon=centroids[:,1], lat=centroids[:,0])
                mask = (
                    (centroids[:, 1] >= minx) & (centroids[:, 1] <= maxx) &
                    (centroids[:, 0] >= miny) & (centroids[:, 0] <= maxy)
                )
                # Further filter with prepared geometry for points inside bbox
                if self._prepared_filter is not None and np.sum(mask) > 0:
                    from shapely.geometry import Point
                    bbox_indices = np.where(mask)[0]
                    for idx in bbox_indices:
                        lon, lat = centroids[idx, 1], centroids[idx, 0]
                        if not self._prepared_filter.contains(Point(lon, lat)):
                            mask[idx] = False
            else:
                mask = np.ones(len(centroids), dtype=bool)
            
            # Add filtered triangles
            triangles_count = 0
            filtered_count = len(centroids) - np.sum(mask)
            
            for i, keep in enumerate(mask):
                if keep:
                    verts = all_verts[i]
                    centroid = all_centroids[i]
                    elevation = np.mean(verts[:, 2])
                    self.terrain_triangles.append(TerrainTriangle(
                        vertices=verts,
                        centroid=centroid,
                        elevation=elevation,
                    ))
                    triangles_count += 1
            
            if filtered_count > 0:
                print(f"      Extracted {triangles_count} terrain triangles (filtered {filtered_count})")
            else:
                print(f"      Extracted {triangles_count} terrain triangles")
            
        except Exception as e:
            print(f"      Error: {e}")
    
    def _parse_vegetation_file(self, filepath: str) -> None:
        """Parse a vegetation GML file."""
        print(f"    Parsing vegetation: {os.path.basename(filepath)}")
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            ns = self._get_namespaces(root)
            
            veg_count = 0
            
            for plant_cover in root.iter('{http://www.opengis.net/citygml/vegetation/2.0}PlantCover'):
                veg = self._extract_vegetation(plant_cover, ns, 'PlantCover')
                if veg and veg.triangles:
                    self.vegetation.append(veg)
                    veg_count += 1
            
            for solitary in root.iter('{http://www.opengis.net/citygml/vegetation/2.0}SolitaryVegetationObject'):
                veg = self._extract_vegetation(solitary, ns, 'SolitaryVegetationObject')
                if veg and veg.triangles:
                    self.vegetation.append(veg)
                    veg_count += 1
            
            print(f"      Extracted {veg_count} vegetation objects")
            
        except Exception as e:
            print(f"      Error: {e}")
    
    def _extract_vegetation(self, veg_elem: ET.Element, ns: Dict[str, str],
                            obj_type: str) -> Optional[PLATEAUVegetation]:
        """Extract vegetation geometry."""
        veg_id = veg_elem.get('{http://www.opengis.net/gml}id', 'unknown')
        veg = PLATEAUVegetation(id=veg_id, object_type=obj_type)
        
        if obj_type == 'PlantCover':
            height_elem = veg_elem.find('.//{http://www.opengis.net/citygml/vegetation/2.0}averageHeight')
        else:
            height_elem = veg_elem.find('.//{http://www.opengis.net/citygml/vegetation/2.0}height')
        
        if height_elem is not None and height_elem.text:
            try:
                h = float(height_elem.text)
                if h > -9998:
                    veg.height = h
                    veg.average_height = h
            except ValueError:
                pass
        
        for pos_list in veg_elem.iter('{http://www.opengis.net/gml}posList'):
            if pos_list.text:
                try:
                    vertices = self._parse_pos_list(pos_list.text)
                    if len(vertices) >= 3:
                        triangles = self._triangulate_polygon(vertices)
                        veg.triangles.extend(triangles)
                except Exception:
                    continue
        
        return veg if veg.triangles else None
    
    def _parse_bridge_file(self, filepath: str) -> None:
        """Parse a bridge GML file."""
        print(f"    Parsing bridges: {os.path.basename(filepath)}")
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            ns = self._get_namespaces(root)
            
            bridge_count = 0
            
            for bridge_elem in root.iter('{http://www.opengis.net/citygml/bridge/2.0}Bridge'):
                bridge = self._extract_bridge(bridge_elem, ns)
                if bridge and bridge.triangles:
                    self.bridges.append(bridge)
                    bridge_count += 1
            
            print(f"      Extracted {bridge_count} bridges")
            
        except Exception as e:
            print(f"      Error: {e}")
    
    def _extract_bridge(self, bridge_elem: ET.Element, ns: Dict[str, str]) -> Optional[PLATEAUBridge]:
        """Extract bridge geometry."""
        bridge_id = bridge_elem.get('{http://www.opengis.net/gml}id', 'unknown')
        bridge = PLATEAUBridge(id=bridge_id)
        
        class_elem = bridge_elem.find('.//{http://www.opengis.net/citygml/bridge/2.0}class')
        if class_elem is not None and class_elem.text:
            bridge.bridge_class = class_elem.text
        
        func_elem = bridge_elem.find('.//{http://www.opengis.net/citygml/bridge/2.0}function')
        if func_elem is not None and func_elem.text:
            bridge.function = func_elem.text
        
        for pos_list in bridge_elem.iter('{http://www.opengis.net/gml}posList'):
            if pos_list.text:
                try:
                    vertices = self._parse_pos_list(pos_list.text)
                    if len(vertices) >= 3:
                        triangles = self._triangulate_polygon(vertices)
                        bridge.triangles.extend(triangles)
                except Exception:
                    continue
        
        return bridge if bridge.triangles else None
    
    def _parse_city_furniture_file(self, filepath: str) -> None:
        """Parse a city furniture GML file."""
        print(f"    Parsing city furniture: {os.path.basename(filepath)}")
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            ns = self._get_namespaces(root)
            
            furniture_count = 0
            
            for frn_elem in root.iter('{http://www.opengis.net/citygml/cityfurniture/2.0}CityFurniture'):
                furniture = self._extract_city_furniture(frn_elem, ns)
                if furniture and furniture.triangles:
                    self.city_furniture.append(furniture)
                    furniture_count += 1
            
            print(f"      Extracted {furniture_count} city furniture objects")
            
        except Exception as e:
            print(f"      Error: {e}")
    
    def _extract_city_furniture(self, frn_elem: ET.Element, ns: Dict[str, str]) -> Optional[PLATEAUCityFurniture]:
        """Extract city furniture geometry."""
        frn_id = frn_elem.get('{http://www.opengis.net/gml}id', 'unknown')
        furniture = PLATEAUCityFurniture(id=frn_id)
        
        class_elem = frn_elem.find('.//{http://www.opengis.net/citygml/cityfurniture/2.0}class')
        if class_elem is not None and class_elem.text:
            furniture.furniture_class = class_elem.text
        
        func_elem = frn_elem.find('.//{http://www.opengis.net/citygml/cityfurniture/2.0}function')
        if func_elem is not None and func_elem.text:
            furniture.function = func_elem.text
        
        for pos_list in frn_elem.iter('{http://www.opengis.net/gml}posList'):
            if pos_list.text:
                try:
                    vertices = self._parse_pos_list(pos_list.text)
                    if len(vertices) >= 3:
                        triangles = self._triangulate_polygon(vertices)
                        furniture.triangles.extend(triangles)
                except Exception:
                    continue
        
        return furniture if furniture.triangles else None
    
    def _parse_land_use_file(self, filepath: str) -> None:
        """Parse a land use GML file."""
        print(f"    Parsing land use: {os.path.basename(filepath)}")
        
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
            ns = self._get_namespaces(root)
            ns['luse'] = 'http://www.opengis.net/citygml/landuse/2.0'
            
            landuse_count = 0
            
            for luse_elem in root.iter('{http://www.opengis.net/citygml/landuse/2.0}LandUse'):
                landuse = self._extract_land_use(luse_elem, ns)
                if landuse and landuse.polygon_coords:
                    self.land_use.append(landuse)
                    landuse_count += 1
            
            print(f"      Extracted {landuse_count} land use polygons")
            
        except Exception as e:
            print(f"      Error: {e}")
    
    def _extract_land_use(self, luse_elem: ET.Element, ns: Dict[str, str]) -> Optional[PLATEAULandUse]:
        """Extract land use polygon."""
        luse_id = luse_elem.get('{http://www.opengis.net/gml}id', 'unknown')
        landuse = PLATEAULandUse(id=luse_id)
        
        class_elem = luse_elem.find('.//{http://www.opengis.net/citygml/landuse/2.0}class')
        if class_elem is not None and class_elem.text:
            landuse.land_use_class = class_elem.text.strip()
        
        org_elem = luse_elem.find('.//{https://www.geospatial.jp/iur/uro/3.1}orgLandUse')
        if org_elem is not None and org_elem.text:
            landuse.org_land_use = org_elem.text.strip()
        
        area_elem = luse_elem.find('.//{https://www.geospatial.jp/iur/uro/3.1}areaInSquareMeter')
        if area_elem is not None and area_elem.text:
            try:
                landuse.area_sqm = float(area_elem.text)
            except ValueError:
                pass
        
        for pos_list in luse_elem.iter('{http://www.opengis.net/gml}posList'):
            if pos_list.text:
                try:
                    coords = [float(x) for x in pos_list.text.strip().split()]
                    vertices = np.array(coords).reshape(-1, 3)
                    landuse.polygon_coords.append(vertices)
                except Exception:
                    continue
        
        return landuse if landuse.polygon_coords else None
    
    def _parse_pos_list(self, pos_list_text: str) -> np.ndarray:
        """Parse GML posList (PLATEAU uses lat, lon, z format)."""
        coords = [float(x) for x in pos_list_text.strip().split()]
        return np.array(coords).reshape(-1, 3)
    
    def _triangulate_polygon(self, vertices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Fan triangulation for polygon."""
        if len(vertices) < 3:
            return []
        
        if np.allclose(vertices[0], vertices[-1]):
            vertices = vertices[:-1]
        
        if len(vertices) < 3:
            return []
        
        triangles = []
        v0 = vertices[0]
        for i in range(1, len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]
            triangles.append((v0.copy(), v1.copy(), v2.copy()))
        
        return triangles
    
    def _setup_transformer(self) -> None:
        """Set up coordinate transformer based on data extent."""
        all_lats = []
        all_lons = []
        
        # Collect from all parsed data
        for building in self.buildings:
            for v0, v1, v2 in building.triangles:
                all_lats.extend([v0[0], v1[0], v2[0]])
                all_lons.extend([v0[1], v1[1], v2[1]])
        
        for tri in self.terrain_triangles:
            for v in tri.vertices:
                all_lats.append(v[0])
                all_lons.append(v[1])
        
        for veg in self.vegetation:
            for v0, v1, v2 in veg.triangles:
                all_lats.extend([v0[0], v1[0], v2[0]])
                all_lons.extend([v0[1], v1[1], v2[1]])
        
        for bridge in self.bridges:
            for v0, v1, v2 in bridge.triangles:
                all_lats.extend([v0[0], v1[0], v2[0]])
                all_lons.extend([v0[1], v1[1], v2[1]])
        
        for frn in self.city_furniture:
            for v0, v1, v2 in frn.triangles:
                all_lats.extend([v0[0], v1[0], v2[0]])
                all_lons.extend([v0[1], v1[1], v2[1]])
        
        if all_lats and all_lons:
            center_lat = np.mean(all_lats)
            center_lon = np.mean(all_lons)
            
            self.latlon_bounds_min = np.array([min(all_lats), min(all_lons), 0])
            self.latlon_bounds_max = np.array([max(all_lats), max(all_lons), 0])
            
            print(f"  Data center: ({center_lat:.6f}, {center_lon:.6f})")
            print(f"  Setting up coordinate transformer...")
            
            self.transformer = CoordinateTransformer(center_lat, center_lon)
    
    def _transform_all_coordinates(self) -> None:
        """Transform all coordinates from lat/lon to local meters."""
        if self.transformer is None:
            print("  Warning: No transformer available, coordinates remain in lat/lon")
            return
        
        print("  Transforming coordinates to local meters...")
        
        # Transform buildings
        for building in self.buildings:
            new_triangles = []
            for v0, v1, v2 in building.triangles:
                new_v0 = self.transformer.transform_coords(v0.reshape(1, 3))[0]
                new_v1 = self.transformer.transform_coords(v1.reshape(1, 3))[0]
                new_v2 = self.transformer.transform_coords(v2.reshape(1, 3))[0]
                new_triangles.append((new_v0, new_v1, new_v2))
            building.triangles = new_triangles
            building.compute_bounds()
        
        # Transform terrain
        for tri in self.terrain_triangles:
            tri.vertices = self.transformer.transform_coords(tri.vertices)
            tri.centroid = np.mean(tri.vertices, axis=0)
            tri.elevation = np.mean(tri.vertices[:, 2])
        
        # Transform vegetation
        for veg in self.vegetation:
            new_triangles = []
            for v0, v1, v2 in veg.triangles:
                new_v0 = self.transformer.transform_coords(v0.reshape(1, 3))[0]
                new_v1 = self.transformer.transform_coords(v1.reshape(1, 3))[0]
                new_v2 = self.transformer.transform_coords(v2.reshape(1, 3))[0]
                new_triangles.append((new_v0, new_v1, new_v2))
            veg.triangles = new_triangles
            veg.compute_bounds()
        
        # Transform bridges
        for bridge in self.bridges:
            new_triangles = []
            for v0, v1, v2 in bridge.triangles:
                new_v0 = self.transformer.transform_coords(v0.reshape(1, 3))[0]
                new_v1 = self.transformer.transform_coords(v1.reshape(1, 3))[0]
                new_v2 = self.transformer.transform_coords(v2.reshape(1, 3))[0]
                new_triangles.append((new_v0, new_v1, new_v2))
            bridge.triangles = new_triangles
            bridge.compute_bounds()
        
        # Transform city furniture
        for frn in self.city_furniture:
            new_triangles = []
            for v0, v1, v2 in frn.triangles:
                new_v0 = self.transformer.transform_coords(v0.reshape(1, 3))[0]
                new_v1 = self.transformer.transform_coords(v1.reshape(1, 3))[0]
                new_v2 = self.transformer.transform_coords(v2.reshape(1, 3))[0]
                new_triangles.append((new_v0, new_v1, new_v2))
            frn.triangles = new_triangles
            frn.compute_bounds()
        
        # Transform land use and create Shapely polygons
        for luse in self.land_use:
            new_coords = []
            for coords in luse.polygon_coords:
                transformed = self.transformer.transform_coords(coords)
                new_coords.append(transformed)
            luse.polygon_coords = new_coords
            luse.compute_bounds()
            
            if HAS_SHAPELY and luse.polygon_coords:
                try:
                    outer_ring = luse.polygon_coords[0][:, :2]
                    luse.shapely_polygon = Polygon(outer_ring)
                    if not luse.shapely_polygon.is_valid:
                        luse.shapely_polygon = make_valid(luse.shapely_polygon)
                except Exception:
                    luse.shapely_polygon = None
    
    def _compute_scene_bounds(self) -> None:
        """Compute overall scene bounding box."""
        all_coords = []
        
        for building in self.buildings:
            for v0, v1, v2 in building.triangles:
                all_coords.extend([v0, v1, v2])
        
        for tri in self.terrain_triangles:
            all_coords.extend(tri.vertices.tolist())
        
        for veg in self.vegetation:
            for v0, v1, v2 in veg.triangles:
                all_coords.extend([v0, v1, v2])
        
        for bridge in self.bridges:
            for v0, v1, v2 in bridge.triangles:
                all_coords.extend([v0, v1, v2])
        
        for frn in self.city_furniture:
            for v0, v1, v2 in frn.triangles:
                all_coords.extend([v0, v1, v2])
        
        if all_coords:
            all_coords = np.array(all_coords)
            self.bounds_min = np.min(all_coords, axis=0)
            self.bounds_max = np.max(all_coords, axis=0)


# =============================================================================
# Convenience Functions
# =============================================================================

def load_citygml(path: str, scale: float = 10.0, add_ground: bool = True
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load CityGML and return normalized triangle data.
    
    Args:
        path: Path to GML file or directory containing GML files.
        scale: Scale factor for normalization.
        add_ground: Whether to add a ground plane.
        
    Returns:
        vertices: (N, 3, 3) array of triangle vertices.
        normals: (N, 3) array of triangle normals.
        material_ids: (N,) array of material IDs (0=building, 1=ground).
    """
    parser = CityGMLParser()
    
    path_obj = Path(path)
    if path_obj.is_dir():
        parser.parse_directory(path)
    else:
        parser.parse_file(path)
    
    parser.normalize_coordinates(center_at_origin=True, scale=scale, swap_yz=True)
    
    if add_ground:
        parser.add_ground_plane(grid_size=100, ground_offset=0.005)
    
    stats = parser.get_stats()
    print(f"Loaded: {stats['num_triangles']} triangles from {stats['num_buildings']} buildings")
    
    return parser.get_triangle_data()


def load_plateau_citygml(base_path: str,
                         rectangle_vertices: Optional[List[Tuple[float, float]]] = None,
                         **kwargs) -> PLATEAUParser:
    """
    Convenience function to load PLATEAU CityGML data.
    
    Args:
        base_path: Path to PLATEAU data folder.
        rectangle_vertices: Optional bounding rectangle [(lon, lat), ...].
        **kwargs: Additional arguments passed to parse_plateau_directory.
        
    Returns:
        PLATEAUParser instance with parsed data.
    """
    parser = PLATEAUParser()
    parser.parse_plateau_directory(base_path, rectangle_vertices=rectangle_vertices, **kwargs)
    return parser


# =============================================================================
# LOD1 CityGML Parser (for GeoDataFrame-based processing)
# =============================================================================

class LOD1CityGMLParser:
    """
    Parser for CityGML LOD1 data that outputs GeoDataFrames.
    
    This parser extracts building footprints, terrain elevation points, and
    vegetation polygons from PLATEAU CityGML files and returns them as
    GeoDataFrames suitable for 2D grid-based processing (as opposed to
    PLATEAUParser which focuses on 3D triangulated geometry).
    
    This is the standard processing mode for VoxCity when LOD2 mode is not enabled.
    
    Example::
    
        parser = LOD1CityGMLParser()
        parser.parse_directory("path/to/udx", rectangle_vertices=[(lon1, lat1), ...])
        
        building_gdf = parser.get_building_gdf()
        terrain_gdf = parser.get_terrain_gdf()
        vegetation_gdf = parser.get_vegetation_gdf()
    
    Note:
        Building footprints are extracted from LOD0/LOD1 geometry with heights
        from bldg:measuredHeight. Terrain uses TIN triangle centroids with
        averaged elevation values. Vegetation uses polygon boundaries with
        height from veg:height or veg:averageHeight attributes.
    """
    
    def __init__(self):
        """Initialize the LOD1 parser."""
        self.buildings: List[Dict] = []
        self.terrain_elements: List[Dict] = []
        self.vegetation_elements: List[Dict] = []
        
        # Default namespaces
        self._default_namespaces = {
            'core': 'http://www.opengis.net/citygml/2.0',
            'bldg': 'http://www.opengis.net/citygml/building/2.0',
            'gml': 'http://www.opengis.net/gml',
            'uro': 'https://www.geospatial.jp/iur/uro/3.0',
            'dem': 'http://www.opengis.net/citygml/relief/2.0',
            'veg': 'http://www.opengis.net/citygml/vegetation/2.0'
        }
    
    def _get_namespaces(self, root) -> Dict[str, str]:
        """Build namespaces from document with fallback to defaults."""
        nsmap = root.nsmap if hasattr(root, 'nsmap') else {}
        
        def pick_ns(prefix: str, keyword: Optional[str] = None, fallback_key: Optional[str] = None) -> str:
            # Try explicit prefix first
            uri = nsmap.get(prefix)
            if uri:
                return uri
            # Try keyword search
            if keyword:
                for v in nsmap.values():
                    if isinstance(v, str) and keyword in v:
                        return v
            # Fallback to defaults
            return self._default_namespaces[fallback_key or prefix]
        
        return {
            'core': pick_ns('core', keyword='citygml', fallback_key='core'),
            'bldg': pick_ns('bldg', keyword='building', fallback_key='bldg'),
            'gml': pick_ns('gml', keyword='gml', fallback_key='gml'),
            'uro': pick_ns('uro', keyword='iur/uro', fallback_key='uro'),
            'dem': pick_ns('dem', keyword='relief', fallback_key='dem'),
            'veg': pick_ns('veg', keyword='vegetation', fallback_key='veg')
        }
    
    @staticmethod
    def _validate_coords(coords: List[Tuple[float, float]]) -> bool:
        """Check that all coordinates are finite."""
        return all(not np.isinf(x) and not np.isnan(x) for coord in coords for x in coord)
    
    @staticmethod
    def _swap_coordinates(polygon):
        """Swap lat/lon to lon/lat in a polygon."""
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
        except ImportError:
            return polygon
        
        if isinstance(polygon, MultiPolygon):
            new_polygons = []
            for geom in polygon.geoms:
                coords = list(geom.exterior.coords)
                swapped_coords = [(y, x) for x, y in coords]
                new_polygons.append(ShapelyPolygon(swapped_coords))
            return MultiPolygon(new_polygons)
        else:
            coords = list(polygon.exterior.coords)
            swapped_coords = [(y, x) for x, y in coords]
            return ShapelyPolygon(swapped_coords)
    
    def _extract_building_footprint(self, building, namespaces) -> Tuple[Optional[any], Optional[float]]:
        """Extract footprint posList and ground elevation from building element."""
        lod_tags = [
            # LOD0
            './/bldg:lod0FootPrint//gml:MultiSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
            './/bldg:lod0RoofEdge//gml:MultiSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
            './/bldg:lod0Solid//gml:Solid//gml:exterior//gml:CompositeSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
            # LOD1
            './/bldg:lod1Solid//gml:Solid//gml:exterior//gml:CompositeSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
            # LOD2
            './/bldg:lod2Solid//gml:Solid//gml:exterior//gml:CompositeSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
            # Fallback
            './/gml:MultiSurface//gml:surfaceMember//gml:Polygon//gml:exterior//gml:LinearRing//gml:posList',
            './/gml:Polygon//gml:exterior//gml:LinearRing//gml:posList'
        ]
        
        for tag in lod_tags:
            pos_list_elements = building.findall(tag, namespaces)
            if pos_list_elements:
                if 'lod1Solid' in tag or 'lod2Solid' in tag or 'lod0Solid' in tag:
                    # Find bottom face
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
                    if footprint_pos_list is not None:
                        return footprint_pos_list, lowest_z
                else:
                    return pos_list_elements[0], None
        return None, None
    
    def _parse_building_file(self, file_path: str) -> List[Dict]:
        """Parse a single CityGML file for building footprints."""
        buildings = []
        
        try:
            if HAS_LXML:
                tree = lxml_ET.parse(file_path)
            else:
                tree = ET.parse(file_path)
            root = tree.getroot()
            namespaces = self._get_namespaces(root)
            source_file_name = Path(file_path).name
            
            for building in root.findall('.//bldg:Building', namespaces):
                building_id = building.get('{http://www.opengis.net/gml}id')
                
                measured_height = building.find('.//bldg:measuredHeight', namespaces)
                height = float(measured_height.text) if measured_height is not None and measured_height.text else None
                
                storeys = building.find('.//bldg:storeysAboveGround', namespaces)
                num_storeys = int(storeys.text) if storeys is not None and storeys.text else None
                
                pos_list, ground_elevation = self._extract_building_footprint(building, namespaces)
                if pos_list is not None:
                    try:
                        from shapely.geometry import Polygon as ShapelyPolygon
                        
                        coords_text = pos_list.text.strip().split()
                        coords = []
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
                        
                        if len(coords) >= 3 and self._validate_coords(coords):
                            polygon = ShapelyPolygon(coords)
                            if polygon.is_valid:
                                buildings.append({
                                    'building_id': building_id,
                                    'height': height,
                                    'storeys': num_storeys,
                                    'ground_elevation': ground_elevation,
                                    'geometry': polygon,
                                    'source_file': source_file_name
                                })
                    except (ValueError, IndexError) as e:
                        continue
        except Exception as e:
            print(f"Error parsing building file {Path(file_path).name}: {e}")
        
        return buildings
    
    def _parse_terrain_file(self, file_path: str) -> List[Dict]:
        """Parse a single CityGML file for terrain data."""
        terrain_elements = []
        
        try:
            from shapely.geometry import Point
            
            if HAS_LXML:
                tree = lxml_ET.parse(file_path)
            else:
                tree = ET.parse(file_path)
            root = tree.getroot()
            namespaces = self._get_namespaces(root)
            source_file_name = Path(file_path).name
            
            for relief in root.findall('.//dem:ReliefFeature', namespaces):
                relief_id = relief.get('{http://www.opengis.net/gml}id')
                
                for tin in relief.findall('.//dem:TINRelief', namespaces):
                    tin_id = tin.get('{http://www.opengis.net/gml}id')
                    triangles = tin.findall('.//gml:Triangle', namespaces)
                    num_triangles = len(triangles)
                    
                    if num_triangles > 10000:
                        # Optimized batch processing
                        centroids_x, centroids_y, elevations = [], [], []
                        
                        for triangle in triangles:
                            pos_lists = triangle.findall('.//gml:posList', namespaces)
                            for pos_list in pos_lists:
                                try:
                                    coords_text = pos_list.text.strip().split()
                                    if len(coords_text) >= 9:
                                        x0, y0, z0 = float(coords_text[0]), float(coords_text[1]), float(coords_text[2])
                                        x1, y1, z1 = float(coords_text[3]), float(coords_text[4]), float(coords_text[5])
                                        x2, y2, z2 = float(coords_text[6]), float(coords_text[7]), float(coords_text[8])
                                        
                                        cx = (x0 + x1 + x2) / 3.0
                                        cy = (y0 + y1 + y2) / 3.0
                                        avg_elev = (z0 + z1 + z2) / 3.0
                                        
                                        if not (np.isinf(cx) or np.isinf(cy) or np.isinf(avg_elev)):
                                            centroids_x.append(cx)
                                            centroids_y.append(cy)
                                            elevations.append(avg_elev)
                                except (ValueError, IndexError):
                                    continue
                        
                        for i, (cx, cy, elev) in enumerate(zip(centroids_x, centroids_y, elevations)):
                            terrain_elements.append({
                                'relief_id': relief_id,
                                'tin_id': tin_id,
                                'triangle_id': f"{tin_id}_tri_{i}",
                                'elevation': elev,
                                'geometry': Point(cx, cy),
                                'polygon': None,
                                'source_file': source_file_name
                            })
                    else:
                        # Standard processing for small datasets
                        from shapely.geometry import Polygon as ShapelyPolygon
                        
                        for i, triangle in enumerate(triangles):
                            pos_lists = triangle.findall('.//gml:posList', namespaces)
                            for pos_list in pos_lists:
                                try:
                                    coords_text = pos_list.text.strip().split()
                                    coords = []
                                    tri_elevations = []
                                    
                                    for j in range(0, len(coords_text), 3):
                                        if j + 2 < len(coords_text):
                                            x = float(coords_text[j])
                                            y = float(coords_text[j+1])
                                            z = float(coords_text[j+2])
                                            if not np.isinf(x) and not np.isinf(y) and not np.isinf(z):
                                                coords.append((x, y))
                                                tri_elevations.append(z)
                                    
                                    if len(coords) >= 3 and self._validate_coords(coords):
                                        polygon = ShapelyPolygon(coords)
                                        if polygon.is_valid:
                                            centroid = polygon.centroid
                                            avg_elevation = np.mean(tri_elevations)
                                            terrain_elements.append({
                                                'relief_id': relief_id,
                                                'tin_id': tin_id,
                                                'triangle_id': f"{tin_id}_tri_{i}",
                                                'elevation': avg_elevation,
                                                'geometry': centroid,
                                                'polygon': polygon,
                                                'source_file': source_file_name
                                            })
                                except (ValueError, IndexError):
                                    continue
                
                # Breaklines
                for breakline in relief.findall('.//dem:breaklines', namespaces):
                    for line in breakline.findall('.//gml:LineString', namespaces):
                        line_id = line.get('{http://www.opengis.net/gml}id')
                        pos_list = line.find('.//gml:posList', namespaces)
                        if pos_list is not None:
                            try:
                                coords_text = pos_list.text.strip().split()
                                points = []
                                line_elevations = []
                                
                                for j in range(0, len(coords_text), 3):
                                    if j + 2 < len(coords_text):
                                        x = float(coords_text[j])
                                        y = float(coords_text[j+1])
                                        z = float(coords_text[j+2])
                                        if not np.isinf(x) and not np.isinf(y) and not np.isinf(z):
                                            points.append(Point(x, y))
                                            line_elevations.append(z)
                                
                                for k, point in enumerate(points):
                                    if point.is_valid:
                                        terrain_elements.append({
                                            'relief_id': relief_id,
                                            'breakline_id': line_id,
                                            'point_id': f"{line_id}_pt_{k}",
                                            'elevation': line_elevations[k],
                                            'geometry': point,
                                            'polygon': None,
                                            'source_file': source_file_name
                                        })
                            except (ValueError, IndexError):
                                continue
                
                # Mass points
                for mass_point in relief.findall('.//dem:massPoint', namespaces):
                    for point_elem in mass_point.findall('.//gml:Point', namespaces):
                        point_id = point_elem.get('{http://www.opengis.net/gml}id')
                        pos = point_elem.find('.//gml:pos', namespaces)
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
                                                'source_file': source_file_name
                                            })
                            except (ValueError, IndexError):
                                continue
            
            if terrain_elements:
                print(f"Extracted {len(terrain_elements)} terrain elements from {source_file_name}")
        
        except Exception as e:
            print(f"Error parsing terrain file {Path(file_path).name}: {e}")
        
        return terrain_elements
    
    def _parse_vegetation_file(self, file_path: str) -> List[Dict]:
        """Parse a single CityGML file for vegetation data."""
        vegetation_elements = []
        
        try:
            from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon
            
            if HAS_LXML:
                tree = lxml_ET.parse(file_path)
            else:
                tree = ET.parse(file_path)
            root = tree.getroot()
            ns = self._get_namespaces(root)
            source_file_name = Path(file_path).name
            
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
                            coords.append((x, y))
                        except:
                            pass
                    if len(coords) >= 3:
                        polygon = ShapelyPolygon(coords)
                        if polygon.is_valid:
                            polygons.append(polygon)
                
                if not polygons:
                    return None
                elif len(polygons) == 1:
                    return polygons[0]
                else:
                    return MultiPolygon(polygons)
            
            def get_veg_geometry(veg_elem):
                geometry_lods = [
                    "lod0Geometry", "lod1Geometry", "lod2Geometry", "lod3Geometry", "lod4Geometry",
                    "lod0MultiSurface", "lod1MultiSurface", "lod2MultiSurface", "lod3MultiSurface", "lod4MultiSurface"
                ]
                for lod_tag in geometry_lods:
                    lod_elem_inner = veg_elem.find(f'.//veg:{lod_tag}', ns)
                    if lod_elem_inner is not None:
                        geom = parse_lod_multisurface(lod_elem_inner)
                        if geom is not None:
                            return geom
                return None
            
            def compute_lod_height(veg_elem):
                z_values = []
                geometry_lods = [
                    "lod0Geometry", "lod1Geometry", "lod2Geometry", "lod3Geometry", "lod4Geometry",
                    "lod0MultiSurface", "lod1MultiSurface", "lod2MultiSurface", "lod3MultiSurface", "lod4MultiSurface"
                ]
                try:
                    for lod_tag in geometry_lods:
                        lod_elem_inner = veg_elem.find(f'.//veg:{lod_tag}', ns)
                        if lod_elem_inner is None:
                            continue
                        for pos_list in lod_elem_inner.findall('.//gml:posList', ns):
                            if pos_list.text is None:
                                continue
                            coords_text = pos_list.text.strip().split()
                            for i in range(2, len(coords_text), 3):
                                try:
                                    z = float(coords_text[i])
                                    if not np.isinf(z) and not np.isnan(z):
                                        z_values.append(z)
                                except:
                                    continue
                    if z_values:
                        return float(max(z_values) - min(z_values))
                except:
                    pass
                return None
            
            # PlantCover
            for plant_cover in root.findall('.//veg:PlantCover', ns):
                cover_id = plant_cover.get('{http://www.opengis.net/gml}id')
                avg_height_elem = plant_cover.find('.//veg:averageHeight', ns)
                if avg_height_elem is not None and avg_height_elem.text:
                    try:
                        vegetation_height = float(avg_height_elem.text)
                        if vegetation_height <= -9998:
                            vegetation_height = None
                    except:
                        vegetation_height = None
                else:
                    vegetation_height = None
                
                if vegetation_height is None:
                    vegetation_height = compute_lod_height(plant_cover)
                
                geometry = get_veg_geometry(plant_cover)
                if geometry is not None and not geometry.is_empty:
                    vegetation_elements.append({
                        'object_type': 'PlantCover',
                        'vegetation_id': cover_id,
                        'height': vegetation_height,
                        'geometry': geometry,
                        'source_file': source_file_name
                    })
            
            # SolitaryVegetationObject
            for solitary in root.findall('.//veg:SolitaryVegetationObject', ns):
                veg_id = solitary.get('{http://www.opengis.net/gml}id')
                height_elem = solitary.find('.//veg:height', ns)
                if height_elem is not None and height_elem.text:
                    try:
                        veg_height = float(height_elem.text)
                        if veg_height <= -9998:
                            veg_height = None
                    except:
                        veg_height = None
                else:
                    veg_height = None
                
                if veg_height is None:
                    veg_height = compute_lod_height(solitary)
                
                geometry = get_veg_geometry(solitary)
                if geometry is not None and not geometry.is_empty:
                    vegetation_elements.append({
                        'object_type': 'SolitaryVegetationObject',
                        'vegetation_id': veg_id,
                        'height': veg_height,
                        'geometry': geometry,
                        'source_file': source_file_name
                    })
            
            if vegetation_elements:
                print(f"Extracted {len(vegetation_elements)} vegetation objects from {source_file_name}")
        
        except Exception as e:
            print(f"Error parsing vegetation file {Path(file_path).name}: {e}")
        
        return vegetation_elements
    
    def parse_directory(self,
                        citygml_path: str,
                        rectangle_vertices: Optional[List[Tuple[float, float]]] = None,
                        parse_buildings: bool = True,
                        parse_terrain: bool = True,
                        parse_vegetation: bool = True) -> None:
        """
        Parse CityGML files from a PLATEAU directory structure.
        
        Args:
            citygml_path: Path to CityGML directory (should contain udx/ folder
                         or be the udx folder itself).
            rectangle_vertices: Optional list of (lon, lat) tuples defining
                              a bounding rectangle to filter tiles.
            parse_buildings: Whether to parse building files.
            parse_terrain: Whether to parse DEM/terrain files.
            parse_vegetation: Whether to parse vegetation files.
        """
        from tqdm import tqdm
        
        # Build rectangle polygon for filtering
        rectangle_polygon = None
        if rectangle_vertices and len(rectangle_vertices) >= 3:
            try:
                from shapely.geometry import Polygon as ShapelyPolygon
                rectangle_polygon = ShapelyPolygon(rectangle_vertices)
            except ImportError:
                rectangle_polygon = None
        
        # Find udx directory
        udx_path = os.path.join(citygml_path, 'udx')
        if not os.path.exists(udx_path):
            # Maybe citygml_path is already udx or contains nested folder
            if os.path.basename(citygml_path) == 'udx':
                udx_path = citygml_path
            else:
                # Try to find udx in subfolders
                for item in os.listdir(citygml_path):
                    potential_udx = os.path.join(citygml_path, item, 'udx')
                    if os.path.exists(potential_udx):
                        udx_path = potential_udx
                        break
        
        if not os.path.exists(udx_path):
            print(f"Warning: Could not find udx directory in {citygml_path}")
            return
        
        # Collect files to process
        files_to_process = []
        
        bldg_dir = os.path.join(udx_path, 'bldg')
        dem_dir = os.path.join(udx_path, 'dem')
        veg_dir = os.path.join(udx_path, 'veg')
        
        if parse_buildings and os.path.exists(bldg_dir):
            for f in os.listdir(bldg_dir):
                if f.endswith(('.gml', '.xml')):
                    files_to_process.append((os.path.join(bldg_dir, f), 'building'))
        
        if parse_terrain and os.path.exists(dem_dir):
            for f in os.listdir(dem_dir):
                if f.endswith(('.gml', '.xml')):
                    files_to_process.append((os.path.join(dem_dir, f), 'terrain'))
        
        if parse_vegetation and os.path.exists(veg_dir):
            for f in os.listdir(veg_dir):
                if f.endswith(('.gml', '.xml')):
                    files_to_process.append((os.path.join(veg_dir, f), 'vegetation'))
        
        if not files_to_process:
            print("No CityGML files to process")
            return
        
        print(f"Found {len(files_to_process)} CityGML files to process")
        
        for file_path, file_type in tqdm(files_to_process, desc="Processing files"):
            filename = os.path.basename(file_path)
            
            # Check tile intersection with rectangle
            if rectangle_polygon is not None:
                try:
                    from shapely.geometry import Polygon as ShapelyPolygon
                    tile_boundary = get_mesh_code_from_filename(filename)
                    if tile_boundary:
                        tile_polygon = ShapelyPolygon(decode_mesh_code(tile_boundary))
                        if not tile_polygon.intersects(rectangle_polygon):
                            continue
                except Exception:
                    # If we can't determine tile boundary, process anyway
                    pass
            
            if file_type == 'building':
                self.buildings.extend(self._parse_building_file(file_path))
            elif file_type == 'terrain':
                self.terrain_elements.extend(self._parse_terrain_file(file_path))
            elif file_type == 'vegetation':
                self.vegetation_elements.extend(self._parse_vegetation_file(file_path))
        
        print(f"Parsed: {len(self.buildings)} buildings, "
              f"{len(self.terrain_elements)} terrain elements, "
              f"{len(self.vegetation_elements)} vegetation objects")
    
    def _swap_gdf_coordinates(self, gdf, geometry_col: str = 'geometry'):
        """Swap lat/lon to lon/lat for all geometries in a GeoDataFrame."""
        try:
            from shapely.geometry import Point
        except ImportError:
            return gdf[geometry_col].tolist()
        
        swapped = []
        for geom in gdf[geometry_col]:
            if isinstance(geom, Point):
                swapped.append(Point(geom.y, geom.x))
            else:
                swapped.append(self._swap_coordinates(geom))
        return swapped
    
    def get_building_gdf(self):
        """
        Get buildings as a GeoDataFrame.
        
        Returns:
            GeoDataFrame with building footprints, heights, and metadata.
            Returns None if no buildings were parsed.
        """
        if not self.buildings:
            return None
        
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas is required for get_building_gdf()")
        
        gdf = gpd.GeoDataFrame(self.buildings, geometry='geometry')
        gdf.set_crs(epsg=6697, inplace=True)
        gdf['geometry'] = self._swap_gdf_coordinates(gdf)
        gdf['id'] = range(len(gdf))
        return gdf
    
    def get_terrain_gdf(self):
        """
        Get terrain elements as a GeoDataFrame.
        
        Returns:
            GeoDataFrame with terrain elevation points/polygons.
            Returns None if no terrain was parsed.
        """
        if not self.terrain_elements:
            return None
        
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas is required for get_terrain_gdf()")
        
        gdf = gpd.GeoDataFrame(self.terrain_elements, geometry='geometry')
        gdf.set_crs(epsg=6697, inplace=True)
        gdf['geometry'] = self._swap_gdf_coordinates(gdf)
        return gdf
    
    def get_vegetation_gdf(self):
        """
        Get vegetation elements as a GeoDataFrame.
        
        Returns:
            GeoDataFrame with vegetation polygons and heights.
            Returns None if no vegetation was parsed.
        """
        if not self.vegetation_elements:
            return None
        
        try:
            import geopandas as gpd
        except ImportError:
            raise ImportError("geopandas is required for get_vegetation_gdf()")
        
        gdf = gpd.GeoDataFrame(self.vegetation_elements, geometry='geometry')
        gdf.set_crs(epsg=6697, inplace=True)
        gdf['geometry'] = self._swap_gdf_coordinates(gdf)
        return gdf


def load_lod1_citygml(citygml_path: str,
                      rectangle_vertices: Optional[List[Tuple[float, float]]] = None,
                      parse_buildings: bool = True,
                      parse_terrain: bool = True,
                      parse_vegetation: bool = True) -> Tuple:
    """
    Load PLATEAU CityGML data and return GeoDataFrames for buildings, terrain, vegetation.
    
    This is the main convenience function for LOD1-based CityGML processing,
    returning GeoDataFrames suitable for grid-based voxelization.
    
    Args:
        citygml_path: Path to PLATEAU CityGML directory (containing udx/ folder).
        rectangle_vertices: Optional list of (lon, lat) tuples defining a bounding
                          rectangle to filter which tiles are processed.
        parse_buildings: Whether to parse building files.
        parse_terrain: Whether to parse DEM/terrain files.
        parse_vegetation: Whether to parse vegetation files.
    
    Returns:
        Tuple of (building_gdf, terrain_gdf, vegetation_gdf), where each is either
        a GeoDataFrame or None if no data was found/parsing was skipped.
    
    Example::
    
        from voxcity.geoprocessor.citygml import load_lod1_citygml
        
        building_gdf, terrain_gdf, vegetation_gdf = load_lod1_citygml(
            "path/to/plateau_data",
            rectangle_vertices=[(139.7, 35.6), (139.8, 35.6), (139.8, 35.7), (139.7, 35.7)]
        )
    """
    parser = LOD1CityGMLParser()
    parser.parse_directory(
        citygml_path,
        rectangle_vertices=rectangle_vertices,
        parse_buildings=parse_buildings,
        parse_terrain=parse_terrain,
        parse_vegetation=parse_vegetation
    )
    
    return (
        parser.get_building_gdf(),
        parser.get_terrain_gdf(),
        parser.get_vegetation_gdf()
    )