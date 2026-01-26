"""
CityGML Data Models
===================

Dataclasses representing various CityGML objects including buildings, terrain,
vegetation, bridges, city furniture, and land use. These models are used by
both parsers and voxelizers.

The models are designed to be interoperable with VoxCity's existing data structures
while supporting the full richness of CityGML LOD2 data.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

try:
    from shapely.geometry import Polygon, MultiPolygon, Point
    from shapely.ops import unary_union
    from shapely.validation import make_valid
    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False
    Polygon = None
    MultiPolygon = None


@dataclass
class Building:
    """
    Represents a building with triangulated 3D geometry for voxelization.
    
    Compatible with both German CityGML and generic building representations.
    
    Attributes:
        id: Unique building identifier (from GML id).
        triangles: List of triangles as (v0, v1, v2) tuples, each vertex is np.ndarray[3].
        min_height: Minimum Z coordinate (building base elevation).
        max_height: Maximum Z coordinate (building top elevation).
        footprint_bounds: 2D bounding box (x_min, y_min, x_max, y_max).
        footprint_polygon: Shapely polygon of actual 2D footprint (if computed).
    """
    id: str
    triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    min_height: float = 0.0
    max_height: float = 0.0
    footprint_bounds: Optional[Tuple[float, float, float, float]] = None
    footprint_polygon: Optional[object] = None
    
    def compute_bounds(self) -> None:
        """Compute min/max heights and XY bounding box from triangles."""
        if not self.triangles:
            return
        
        all_verts = []
        for v0, v1, v2 in self.triangles:
            all_verts.extend([v0, v1, v2])
        all_verts = np.array(all_verts)
        
        self.min_height = float(np.min(all_verts[:, 2]))
        self.max_height = float(np.max(all_verts[:, 2]))
        self.footprint_bounds = (
            float(np.min(all_verts[:, 0])),
            float(np.min(all_verts[:, 1])),
            float(np.max(all_verts[:, 0])),
            float(np.max(all_verts[:, 1])),
        )
    
    def compute_footprint_polygon(self) -> Optional[object]:
        """
        Compute 2D footprint polygon by projecting 3D triangles onto XY plane.
        
        Creates the actual building footprint shape by unioning all projected
        triangle polygons. This gives more accurate results than just using
        the bounding box.
        
        Returns:
            Shapely polygon of the footprint, or None if computation fails.
        """
        if not HAS_SHAPELY or not self.triangles:
            return None
        
        try:
            triangle_polygons = []
            for v0, v1, v2 in self.triangles:
                # Project to XY plane
                coords = [(v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1])]
                
                # Skip degenerate triangles
                if len(set(coords)) < 3:
                    continue
                
                try:
                    tri_poly = Polygon(coords)
                    if tri_poly.is_valid and tri_poly.area > 1e-10:
                        triangle_polygons.append(tri_poly)
                    elif not tri_poly.is_valid:
                        fixed = make_valid(tri_poly)
                        if fixed.area > 1e-10:
                            triangle_polygons.append(fixed)
                except Exception:
                    continue
            
            if not triangle_polygons:
                return None
            
            footprint = unary_union(triangle_polygons)
            if footprint.is_empty:
                return None
            
            self.footprint_polygon = footprint
            return footprint
            
        except Exception:
            return None
    
    @property
    def height(self) -> float:
        """Building height (max - min)."""
        return self.max_height - self.min_height
    
    @property
    def triangle_count(self) -> int:
        """Number of triangles in the building mesh."""
        return len(self.triangles)


@dataclass
class PLATEAUBuilding:
    """
    Represents a building from Japanese PLATEAU CityGML data.
    
    Extends basic Building with PLATEAU-specific attributes like measured_height
    and ground_elevation from the dataset metadata.
    
    Attributes:
        id: Unique building identifier (gml:id).
        triangles: List of triangles in local meter coordinates.
        measured_height: Height from PLATEAU metadata (if available).
        ground_elevation: Ground elevation from PLATEAU metadata.
        min_z: Minimum Z coordinate in local meters.
        max_z: Maximum Z coordinate in local meters.
        footprint_bounds: 2D bounding box in local meters.
        footprint_polygon: Shapely polygon of footprint.
    """
    id: str
    triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    measured_height: Optional[float] = None
    ground_elevation: Optional[float] = None
    min_z: float = 0.0
    max_z: float = 0.0
    footprint_bounds: Optional[Tuple[float, float, float, float]] = None
    footprint_polygon: Optional[object] = None
    
    def compute_bounds(self) -> None:
        """Compute bounding box from triangles."""
        if not self.triangles:
            return
        
        all_verts = []
        for v0, v1, v2 in self.triangles:
            all_verts.extend([v0, v1, v2])
        all_verts = np.array(all_verts)
        
        self.min_z = float(np.min(all_verts[:, 2]))
        self.max_z = float(np.max(all_verts[:, 2]))
        self.footprint_bounds = (
            float(np.min(all_verts[:, 0])),
            float(np.min(all_verts[:, 1])),
            float(np.max(all_verts[:, 0])),
            float(np.max(all_verts[:, 1])),
        )
    
    def compute_footprint_polygon(self) -> Optional[object]:
        """Compute 2D footprint polygon from 3D triangles."""
        if not HAS_SHAPELY or not self.triangles:
            return None
        
        try:
            triangle_polygons = []
            for v0, v1, v2 in self.triangles:
                coords = [(v0[0], v0[1]), (v1[0], v1[1]), (v2[0], v2[1])]
                if len(set(coords)) < 3:
                    continue
                try:
                    tri_poly = Polygon(coords)
                    if tri_poly.is_valid and tri_poly.area > 1e-10:
                        triangle_polygons.append(tri_poly)
                    elif not tri_poly.is_valid:
                        fixed = make_valid(tri_poly)
                        if fixed.area > 1e-10:
                            triangle_polygons.append(fixed)
                except Exception:
                    continue
            
            if not triangle_polygons:
                return None
            
            footprint = unary_union(triangle_polygons)
            if footprint.is_empty:
                return None
            
            self.footprint_polygon = footprint
            return footprint
        except Exception:
            return None
    
    @property
    def height(self) -> float:
        """Building height from geometry."""
        return self.max_z - self.min_z


@dataclass
class TerrainTriangle:
    """
    Represents a terrain triangle from DEM CityGML (TIN relief).
    
    Attributes:
        vertices: 3x3 array of triangle vertices [[x,y,z], [x,y,z], [x,y,z]].
        centroid: Triangle centroid [x, y, z].
        elevation: Average elevation of the triangle.
    """
    vertices: np.ndarray  # Shape: (3, 3)
    centroid: np.ndarray  # Shape: (3,)
    elevation: float


@dataclass
class PLATEAUVegetation:
    """
    Represents vegetation from PLATEAU CityGML (PlantCover or SolitaryVegetationObject).
    
    Attributes:
        id: Unique vegetation identifier.
        object_type: 'PlantCover' or 'SolitaryVegetationObject'.
        triangles: Triangulated 3D geometry.
        height: Specified height from metadata.
        average_height: Computed average height from geometry.
        min_z, max_z: Z coordinate range.
        footprint_bounds: 2D bounding box.
    """
    id: str
    object_type: str  # 'PlantCover' or 'SolitaryVegetationObject'
    triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    height: Optional[float] = None
    average_height: Optional[float] = None
    min_z: float = 0.0
    max_z: float = 0.0
    footprint_bounds: Optional[Tuple[float, float, float, float]] = None
    
    def compute_bounds(self) -> None:
        """Compute bounding box from triangles."""
        if not self.triangles:
            return
        all_verts = []
        for v0, v1, v2 in self.triangles:
            all_verts.extend([v0, v1, v2])
        all_verts = np.array(all_verts)
        self.min_z = float(np.min(all_verts[:, 2]))
        self.max_z = float(np.max(all_verts[:, 2]))
        self.footprint_bounds = (
            float(np.min(all_verts[:, 0])),
            float(np.min(all_verts[:, 1])),
            float(np.max(all_verts[:, 0])),
            float(np.max(all_verts[:, 1])),
        )


@dataclass
class PLATEAUBridge:
    """
    Represents a bridge from PLATEAU CityGML.
    
    Attributes:
        id: Unique bridge identifier.
        triangles: Triangulated 3D geometry.
        bridge_class: Bridge classification code.
        function: Bridge function code.
        min_z, max_z: Z coordinate range.
        footprint_bounds: 2D bounding box.
    """
    id: str
    triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    bridge_class: Optional[str] = None
    function: Optional[str] = None
    min_z: float = 0.0
    max_z: float = 0.0
    footprint_bounds: Optional[Tuple[float, float, float, float]] = None
    
    def compute_bounds(self) -> None:
        """Compute bounding box from triangles."""
        if not self.triangles:
            return
        all_verts = []
        for v0, v1, v2 in self.triangles:
            all_verts.extend([v0, v1, v2])
        all_verts = np.array(all_verts)
        self.min_z = float(np.min(all_verts[:, 2]))
        self.max_z = float(np.max(all_verts[:, 2]))
        self.footprint_bounds = (
            float(np.min(all_verts[:, 0])),
            float(np.min(all_verts[:, 1])),
            float(np.max(all_verts[:, 0])),
            float(np.max(all_verts[:, 1])),
        )


@dataclass
class PLATEAUCityFurniture:
    """
    Represents city furniture from PLATEAU CityGML (benches, lamps, etc.).
    
    Attributes:
        id: Unique furniture identifier.
        triangles: Triangulated 3D geometry.
        furniture_class: Furniture classification code.
        function: Furniture function code.
        min_z, max_z: Z coordinate range.
        footprint_bounds: 2D bounding box.
    """
    id: str
    triangles: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=list)
    furniture_class: Optional[str] = None
    function: Optional[str] = None
    min_z: float = 0.0
    max_z: float = 0.0
    footprint_bounds: Optional[Tuple[float, float, float, float]] = None
    
    def compute_bounds(self) -> None:
        """Compute bounding box from triangles."""
        if not self.triangles:
            return
        all_verts = []
        for v0, v1, v2 in self.triangles:
            all_verts.extend([v0, v1, v2])
        all_verts = np.array(all_verts)
        self.min_z = float(np.min(all_verts[:, 2]))
        self.max_z = float(np.max(all_verts[:, 2]))
        self.footprint_bounds = (
            float(np.min(all_verts[:, 0])),
            float(np.min(all_verts[:, 1])),
            float(np.max(all_verts[:, 0])),
            float(np.max(all_verts[:, 1])),
        )


@dataclass
class PLATEAULandUse:
    """
    Represents a land use polygon from PLATEAU CityGML.
    
    PLATEAU land use types are mapped to VoxCity Standard Land Cover Classes.
    
    Attributes:
        id: Unique land use polygon identifier.
        polygon_coords: List of coordinate arrays defining the polygon.
        land_use_class: PLATEAU land use code (e.g., "211", "214").
        org_land_use: Detailed organization land use code.
        area_sqm: Area in square meters.
        footprint_bounds: 2D bounding box.
        shapely_polygon: Shapely polygon after coordinate transformation.
    """
    id: str
    polygon_coords: List[np.ndarray] = field(default_factory=list)
    land_use_class: Optional[str] = None
    org_land_use: Optional[str] = None
    area_sqm: Optional[float] = None
    footprint_bounds: Optional[Tuple[float, float, float, float]] = None
    shapely_polygon: Optional[object] = None
    
    def compute_bounds(self) -> None:
        """Compute bounding box from polygon coordinates."""
        if not self.polygon_coords:
            return
        all_coords = np.vstack(self.polygon_coords)
        self.footprint_bounds = (
            float(np.min(all_coords[:, 0])),
            float(np.min(all_coords[:, 1])),
            float(np.max(all_coords[:, 0])),
            float(np.max(all_coords[:, 1])),
        )
