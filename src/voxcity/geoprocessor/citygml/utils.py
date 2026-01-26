"""
CityGML Utilities
=================

Common utilities for CityGML processing including:
- Coordinate transformations
- Mesh code encoding/decoding
- Geometry helpers
- Land use mapping functions

These utilities are shared across parsers, voxelizers, and other modules.
"""

import numpy as np
import re
from typing import List, Tuple, Optional

try:
    import pyproj
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

from .constants import PLATEAU_LANDUSE_TO_VOXCITY


# =============================================================================
# Coordinate Transformation
# =============================================================================

class CoordinateTransformer:
    """
    Transforms coordinates from WGS84/JGD2011 lat/lon to local meters.
    
    Uses a Transverse Mercator projection centered on the data extent
    for accurate local distance measurements. Falls back to approximate
    conversion if pyproj is not available.
    
    Example::
    
        transformer = CoordinateTransformer(center_lat=35.6, center_lon=139.7)
        x, y = transformer.transform(lon=139.75, lat=35.65)
    """
    
    def __init__(self, center_lat: float, center_lon: float):
        """
        Initialize transformer with center point for projection.
        
        Args:
            center_lat: Center latitude in degrees.
            center_lon: Center longitude in degrees.
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        
        if HAS_PYPROJ:
            # Use Transverse Mercator centered on data
            self.proj_wgs84 = pyproj.CRS.from_epsg(4326)
            self.proj_local = pyproj.CRS.from_proj4(
                f"+proj=tmerc +lat_0={center_lat} +lon_0={center_lon} "
                f"+k=1 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
            )
            self.transformer = pyproj.Transformer.from_crs(
                self.proj_wgs84, self.proj_local, always_xy=True
            )
            self._inverse_transformer = pyproj.Transformer.from_crs(
                self.proj_local, self.proj_wgs84, always_xy=True
            )
        else:
            # Approximate conversion using simple formula
            self.meters_per_degree_lat = 111320.0
            self.meters_per_degree_lon = 111320.0 * np.cos(np.radians(center_lat))
    
    def transform(self, lon: float, lat: float) -> Tuple[float, float]:
        """
        Transform single coordinate from lon/lat to local meters.
        
        Args:
            lon: Longitude in degrees.
            lat: Latitude in degrees.
            
        Returns:
            (x, y) in local meters.
        """
        if HAS_PYPROJ:
            return self.transformer.transform(lon, lat)
        else:
            x = (lon - self.center_lon) * self.meters_per_degree_lon
            y = (lat - self.center_lat) * self.meters_per_degree_lat
            return x, y
    
    def inverse_transform(self, x: float, y: float) -> Tuple[float, float]:
        """
        Transform from local meters back to lon/lat.
        
        Args:
            x: X coordinate in local meters.
            y: Y coordinate in local meters.
            
        Returns:
            (lon, lat) in degrees.
        """
        if HAS_PYPROJ:
            return self._inverse_transformer.transform(x, y)
        else:
            lon = x / self.meters_per_degree_lon + self.center_lon
            lat = y / self.meters_per_degree_lat + self.center_lat
            return lon, lat
    
    def transform_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Transform array of coordinates (N x 3) from lat/lon/z to x/y/z meters.
        
        Note: PLATEAU data has coordinates as (lat, lon, z).
        
        Args:
            coords: Nx3 array with columns [lat, lon, z].
            
        Returns:
            Nx3 array with columns [x, y, z] in local meters.
        """
        result = np.zeros_like(coords)
        
        for i in range(len(coords)):
            lat, lon, z = coords[i]
            x, y = self.transform(lon, lat)
            result[i] = [x, y, z]
        
        return result
    
    def transform_coords_array(self, lons: np.ndarray, lats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform arrays of lon/lat to x/y.
        
        Args:
            lons: Array of longitudes.
            lats: Array of latitudes.
            
        Returns:
            (xs, ys) arrays in local meters.
        """
        if HAS_PYPROJ:
            return self.transformer.transform(lons, lats)
        else:
            xs = (lons - self.center_lon) * self.meters_per_degree_lon
            ys = (lats - self.center_lat) * self.meters_per_degree_lat
            return xs, ys


# =============================================================================
# Japanese Mesh Code Utilities
# =============================================================================

def decode_2nd_level_mesh(mesh6: str) -> Tuple[float, float, float, float]:
    """
    Decode a standard (2nd-level) mesh code to geographic coordinates.
    
    The Japanese standard mesh system divides Japan into a grid of cells.
    Each 2nd-level mesh is approximately 10km x 10km.
    
    Args:
        mesh6: A 6-digit mesh code string.
        
    Returns:
        (lat_sw, lon_sw, lat_ne, lon_ne) coordinates in degrees representing
        the southwest and northeast corners of the mesh.
    """
    code = int(mesh6)
    N1 = code // 10000              # First 2 digits
    M1 = (code // 100) % 100        # Next 2 digits
    row_2nd = (code // 10) % 10     # 5th digit
    col_2nd = code % 10             # 6th digit
    
    # 1st-level mesh southwest corner
    lat_sw_1 = (N1 * 40.0) / 60.0    # Each N1 => 40' => 2/3 degrees
    lon_sw_1 = 100.0 + M1            # Each M1 => offset from 100°E
    
    # 2nd-level mesh subdivides 8x8 => each cell = 1/12° lat x 0.125° lon
    dlat_2nd = (40.0 / 60.0) / 8.0   # 1/12°
    dlon_2nd = 1.0 / 8.0             # 0.125°
    
    lat_sw = lat_sw_1 + row_2nd * dlat_2nd
    lon_sw = lon_sw_1 + col_2nd * dlon_2nd
    lat_ne = lat_sw + dlat_2nd
    lon_ne = lon_sw + dlon_2nd
    
    return (lat_sw, lon_sw, lat_ne, lon_ne)


def decode_mesh_code(mesh_str: str) -> List[Tuple[float, float]]:
    """
    Decode mesh codes into geographic boundary polygon.
    
    Supports 6-digit (2nd-level) and 8-digit (3rd-level) mesh codes.
    
    Args:
        mesh_str: A mesh code string (6 or 8 digits).
        
    Returns:
        List of (lon, lat) tuples forming a closed polygon in WGS84.
        
    Raises:
        ValueError: If mesh code length is invalid.
    """
    if len(mesh_str) < 6:
        raise ValueError(f"Mesh code '{mesh_str}' is too short.")
    
    mesh6 = mesh_str[:6]
    lat_sw_2, lon_sw_2, lat_ne_2, lon_ne_2 = decode_2nd_level_mesh(mesh6)
    
    if len(mesh_str) == 6:
        return [
            (lon_sw_2, lat_sw_2),
            (lon_ne_2, lat_sw_2),
            (lon_ne_2, lat_ne_2),
            (lon_sw_2, lat_ne_2),
            (lon_sw_2, lat_sw_2),
        ]
    elif len(mesh_str) == 8:
        row_10 = int(mesh_str[6])
        col_10 = int(mesh_str[7])
        
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
            (lon_sw, lat_sw),
        ]
    else:
        raise ValueError(
            f"Unsupported mesh code length '{mesh_str}'. "
            "Only 6-digit or 8-digit codes are supported."
        )


def get_mesh_code_from_filename(filename: str) -> Optional[str]:
    """
    Extract mesh code from PLATEAU filename.
    
    PLATEAU files typically start with mesh code digits followed by underscore.
    Example: '51357348_bldg_6697_op.gml' -> '51357348'
    
    Args:
        filename: PLATEAU format filename.
        
    Returns:
        Mesh code string, or None if not found.
    """
    m = re.match(r'^(\d+)_', filename)
    if m:
        return m.group(1)
    return None


def get_tile_polygon_from_filename(filename: str) -> List[Tuple[float, float]]:
    """
    Extract and decode mesh code from PLATEAU filename into boundary polygon.
    
    Args:
        filename: PLATEAU format filename (e.g., '51357348_bldg_6697_op.gml').
        
    Returns:
        List of (lon, lat) tuples forming the tile boundary polygon.
        
    Raises:
        ValueError: If no mesh code found in filename.
    """
    mesh_code = get_mesh_code_from_filename(filename)
    if not mesh_code:
        raise ValueError(f"No mesh code found in filename: {filename}")
    return decode_mesh_code(mesh_code)


# =============================================================================
# Land Use Mapping
# =============================================================================

def get_voxcity_landcover_code(plateau_code: str) -> int:
    """
    Convert PLATEAU land use code to VoxCity standard land cover class.
    
    VoxCity Standard Classes (1-based indices):
        1: Bareland, 2: Rangeland, 3: Shrub, 4: Agriculture, 5: Tree,
        6: Moss/lichen, 7: Wetland, 8: Mangrove, 9: Water, 10: Snow/ice,
        11: Developed space, 12: Road, 13: Building, 14: No Data
    
    Args:
        plateau_code: PLATEAU land use code (e.g., "211", "214").
        
    Returns:
        VoxCity land cover class (1-14). Returns 14 (No Data) for unknown codes.
    """
    return PLATEAU_LANDUSE_TO_VOXCITY.get(str(plateau_code), 14)


# =============================================================================
# Geometry Utilities
# =============================================================================

def parse_pos_list(pos_list_text: str) -> np.ndarray:
    """
    Parse a GML posList string into a numpy array of 3D coordinates.
    
    Args:
        pos_list_text: Space-separated coordinate string from GML posList element.
        
    Returns:
        Nx3 numpy array of coordinates.
    """
    coords = [float(x) for x in pos_list_text.strip().split()]
    return np.array(coords).reshape(-1, 3)


def triangulate_polygon(vertices: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Triangulate a polygon using fan triangulation.
    
    Works well for convex and mostly convex polygons. For complex concave
    polygons, consider using more sophisticated ear clipping algorithms.
    
    Args:
        vertices: Nx3 array of polygon vertices.
        
    Returns:
        List of triangles as (v0, v1, v2) tuples, each vertex is ndarray[3].
    """
    if len(vertices) < 3:
        return []
    
    # Remove duplicate last vertex if present (closed polygon)
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


def compute_triangle_normal(v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """
    Compute the unit normal of a triangle.
    
    Args:
        v0, v1, v2: Triangle vertices as 3D arrays.
        
    Returns:
        Unit normal vector (or zero vector if degenerate).
    """
    edge1 = v1 - v0
    edge2 = v2 - v0
    normal = np.cross(edge1, edge2)
    length = np.linalg.norm(normal)
    if length > 1e-10:
        normal = normal / length
    return normal


def point_in_triangle_2d(px: float, py: float,
                         v0: np.ndarray, v1: np.ndarray, v2: np.ndarray) -> bool:
    """
    Check if point (px, py) is inside triangle projected to XY plane.
    
    Args:
        px, py: Point coordinates to test.
        v0, v1, v2: Triangle vertices (uses only x, y components).
        
    Returns:
        True if point is inside the triangle.
    """
    def sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    
    p = np.array([px, py])
    d1 = sign(p, v0[:2], v1[:2])
    d2 = sign(p, v1[:2], v2[:2])
    d3 = sign(p, v2[:2], v0[:2])
    
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    
    return not (has_neg and has_pos)


def swap_coordinates(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Swap lat/lon to lon/lat in coordinate list.
    
    PLATEAU data often uses (lat, lon) order, while GIS tools expect (lon, lat).
    
    Args:
        coords: List of (lat, lon) tuples.
        
    Returns:
        List of (lon, lat) tuples.
    """
    return [(lon, lat) for lat, lon in coords]


def compute_bounds_from_triangles(
    triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute axis-aligned bounding box from triangle list.
    
    Args:
        triangles: List of (v0, v1, v2) triangles.
        
    Returns:
        (bounds_min, bounds_max) as 3D arrays.
    """
    if not triangles:
        return np.zeros(3), np.zeros(3)
    
    all_verts = []
    for v0, v1, v2 in triangles:
        all_verts.extend([v0, v1, v2])
    all_verts = np.array(all_verts)
    
    return np.min(all_verts, axis=0), np.max(all_verts, axis=0)
