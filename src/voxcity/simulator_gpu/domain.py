"""
Shared domain definition for simulator_gpu.

Represents the 3D computational domain with:
- Grid cells (dx, dy, dz spacing)
- Topography (terrain height)
- Building geometry (3D obstacles)
- Plant canopy (Leaf Area Density - LAD)
- Tree mask for view analysis
"""

import taichi as ti
import numpy as np
from typing import Tuple, Optional, Union
from .core import Vector3, Point3, EXT_COEF
from .init_taichi import ensure_initialized


# Surface direction indices (matching PALM convention)
IUP = 0      # Upward facing (horizontal roof/ground)
IDOWN = 1    # Downward facing
INORTH = 2   # North facing (positive y)
ISOUTH = 3   # South facing (negative y)
IEAST = 4    # East facing (positive x)
IWEST = 5    # West facing (negative x)

# Direction normal vectors (x, y, z)
DIR_NORMALS = {
    IUP: (0.0, 0.0, 1.0),
    IDOWN: (0.0, 0.0, -1.0),
    INORTH: (0.0, 1.0, 0.0),
    ISOUTH: (0.0, -1.0, 0.0),
    IEAST: (1.0, 0.0, 0.0),
    IWEST: (-1.0, 0.0, 0.0),
}


@ti.data_oriented
class Domain:
    """
    3D computational domain for simulation.
    
    The domain uses a regular grid with:
    - x: West to East
    - y: South to North  
    - z: Ground to Sky
    
    Attributes:
        nx, ny, nz: Number of grid cells in each direction
        dx, dy, dz: Grid spacing in meters
        origin: (x, y, z) coordinates of domain origin
    """
    
    def __init__(
        self,
        nx: int,
        ny: int, 
        nz: int,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        origin_lat: Optional[float] = None,
        origin_lon: Optional[float] = None
    ):
        """
        Initialize the domain.
        
        Args:
            nx, ny, nz: Grid dimensions
            dx, dy, dz: Grid spacing (m)
            origin: Domain origin coordinates
            origin_lat: Latitude for solar calculations (degrees)
            origin_lon: Longitude for solar calculations (degrees)
        """
        # Ensure Taichi is initialized before creating any fields
        ensure_initialized()
        
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.origin = origin
        self.origin_lat = origin_lat if origin_lat is not None else 0.0
        self.origin_lon = origin_lon if origin_lon is not None else 0.0
        
        # Domain bounds
        self.x_min = origin[0]
        self.x_max = origin[0] + nx * dx
        self.y_min = origin[1]
        self.y_max = origin[1] + ny * dy
        self.z_min = origin[2]
        self.z_max = origin[2] + nz * dz
        
        # Grid cell volume
        self.cell_volume = dx * dy * dz
        
        # Topography: terrain height at each (i, j) column
        self.topo_top = ti.field(dtype=ti.i32, shape=(nx, ny))
        
        # Building mask: 1 if cell is solid (building), 0 if air
        self.is_solid = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        
        # Tree mask: 1 if cell is tree canopy, 0 otherwise
        self.is_tree = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        
        # Leaf Area Density (m^2/m^3) for plant canopy
        self.lad = ti.field(dtype=ti.f32, shape=(nx, ny, nz))
        
        # Plant canopy top index for each column
        self.plant_top = ti.field(dtype=ti.i32, shape=(nx, ny))
        
        # Surface count
        self.n_surfaces = ti.field(dtype=ti.i32, shape=())
        
        # Initialize arrays
        self._init_arrays()
        
    @ti.kernel
    def _init_arrays(self):
        """Initialize all arrays to default values."""
        for i, j in self.topo_top:
            self.topo_top[i, j] = 0
            self.plant_top[i, j] = 0
        
        for i, j, k in self.is_solid:
            self.is_solid[i, j, k] = 0
            self.is_tree[i, j, k] = 0
            self.lad[i, j, k] = 0.0
    
    def set_flat_terrain(self, height: float = 0.0):
        """Set flat terrain at given height."""
        k_top = int(height / self.dz)
        self._set_flat_terrain_kernel(k_top)
    
    def initialize_terrain(self, height: float = 0.0):
        """Alias for set_flat_terrain."""
        self.set_flat_terrain(height)
    
    @ti.kernel
    def _set_flat_terrain_kernel(self, k_top: ti.i32):
        for i, j in self.topo_top:
            self.topo_top[i, j] = k_top
            for k in range(k_top + 1):
                self.is_solid[i, j, k] = 1
    
    def set_terrain_from_array(self, terrain_height: np.ndarray):
        """
        Set terrain from 2D numpy array of heights.
        
        Args:
            terrain_height: 2D array (nx, ny) of terrain heights in meters
        """
        terrain_k = (terrain_height / self.dz).astype(np.int32)
        self._set_terrain_kernel(terrain_k)
    
    @ti.kernel
    def _set_terrain_kernel(self, terrain_k: ti.types.ndarray()):
        for i, j in self.topo_top:
            k_top = terrain_k[i, j]
            self.topo_top[i, j] = k_top
            for k in range(self.nz):
                if k <= k_top:
                    self.is_solid[i, j, k] = 1
                else:
                    self.is_solid[i, j, k] = 0
    
    def add_building(
        self,
        x_range: Optional[Tuple[int, int]] = None,
        y_range: Optional[Tuple[int, int]] = None,
        z_range: Optional[Tuple[int, int]] = None,
        *,
        x_start: Optional[int] = None,
        x_end: Optional[int] = None,
        y_start: Optional[int] = None,
        y_end: Optional[int] = None,
        height: Optional[float] = None
    ):
        """
        Add a rectangular building to the domain.
        """
        # Handle convenience parameters
        if x_start is not None and x_end is not None:
            x_range = (x_start, x_end)
        if y_start is not None and y_end is not None:
            y_range = (y_start, y_end)
        if height is not None and z_range is None:
            k_top = int(height / self.dz) + 1
            z_range = (0, k_top)
        
        if x_range is None or y_range is None or z_range is None:
            raise ValueError("Must provide either range tuples or individual parameters")
        
        self._add_building_kernel(x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1])
    
    @ti.kernel
    def _add_building_kernel(self, i_min: ti.i32, i_max: ti.i32, j_min: ti.i32, j_max: ti.i32, k_min: ti.i32, k_max: ti.i32):
        for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_min, k_max)):
            self.is_solid[i, j, k] = 1
    
    def add_tree(
        self,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        z_range: Tuple[int, int],
        lad_value: float = 1.0
    ):
        """
        Add a tree canopy region to the domain.
        
        Args:
            x_range, y_range, z_range: Grid index ranges
            lad_value: Leaf Area Density value
        """
        self._add_tree_kernel(x_range[0], x_range[1], y_range[0], y_range[1], z_range[0], z_range[1], lad_value)
    
    @ti.kernel
    def _add_tree_kernel(self, i_min: ti.i32, i_max: ti.i32, j_min: ti.i32, j_max: ti.i32, k_min: ti.i32, k_max: ti.i32, lad: ti.f32):
        for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_min, k_max)):
            self.is_tree[i, j, k] = 1
            self.lad[i, j, k] = lad
    
    def set_from_voxel_data(self, voxel_data: np.ndarray, tree_code: int = -2, solid_codes: Optional[list] = None):
        """
        Set domain from a 3D voxel data array.
        
        Args:
            voxel_data: 3D numpy array with voxel class codes
            tree_code: Class code for trees (default -2)
            solid_codes: List of codes that are solid (default: all non-zero except tree_code)
        """
        if solid_codes is None:
            # All non-zero codes except tree are solid
            solid_codes = []
        
        self._set_from_voxel_data_kernel(voxel_data, tree_code)
    
    @ti.kernel
    def _set_from_voxel_data_kernel(self, voxel_data: ti.types.ndarray(), tree_code: ti.i32):
        for i, j, k in ti.ndrange(self.nx, self.ny, self.nz):
            val = voxel_data[i, j, k]
            if val == tree_code:
                self.is_tree[i, j, k] = 1
                self.is_solid[i, j, k] = 0
            elif val != 0:
                self.is_solid[i, j, k] = 1
                self.is_tree[i, j, k] = 0
            else:
                self.is_solid[i, j, k] = 0
                self.is_tree[i, j, k] = 0
    
    def get_max_dist(self) -> float:
        """Get maximum ray distance (domain diagonal)."""
        import math
        return math.sqrt(
            (self.nx * self.dx)**2 + 
            (self.ny * self.dy)**2 + 
            (self.nz * self.dz)**2
        )
