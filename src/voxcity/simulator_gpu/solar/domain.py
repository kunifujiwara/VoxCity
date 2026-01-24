"""Domain definition for palm-solar.

Represents the 3D computational domain with:
- Grid cells (dx, dy, dz spacing)
- Topography (terrain height)
- Building geometry (3D obstacles)
- Plant canopy (Leaf Area Density - LAD)
- Surface properties (albedo)

Coordinate System Notes:
    VoxCity uses a grid where:
    - i (row index) increases from North to South
    - j (column index) increases from West to East
    - k (layer index) increases upward
    
    The Domain class maps grid indices to coordinates as:
    - x = i * dx (grid row direction, increases toward South)
    - y = j * dy (grid column direction, increases toward East)
    - z = k * dz (vertical, increases upward)
    
    Surface direction labels (INORTH, ISOUTH, etc.) follow PALM conventions
    but in VoxCity's grid:
    - IEAST (direction 4): surface at i+ boundary (South-facing in geographic terms)
    - IWEST (direction 5): surface at i- boundary (North-facing in geographic terms)
    - INORTH (direction 2): surface at j+ boundary (East-facing in geographic terms)
    - ISOUTH (direction 3): surface at j- boundary (West-facing in geographic terms)
"""

import taichi as ti
import numpy as np
from typing import Tuple, Optional, Union
from .core import Vector3, Point3, EXT_COEF
from ..init_taichi import ensure_initialized


# Surface direction indices (matching PALM convention)
# Note: In VoxCity's grid, +x = South, +y = East (not the PALM labels)
IUP = 0      # Upward facing (horizontal roof/ground)
IDOWN = 1    # Downward facing
INORTH = 2   # +y normal (East-facing in VoxCity geographic terms)
ISOUTH = 3   # -y normal (West-facing in VoxCity geographic terms)
IEAST = 4    # +x normal (South-facing in VoxCity geographic terms)
IWEST = 5    # -x normal (North-facing in VoxCity geographic terms)

# Direction normal vectors (x, y, z) in grid-index coordinates
DIR_NORMALS = {
    IUP: (0.0, 0.0, 1.0),
    IDOWN: (0.0, 0.0, -1.0),
    INORTH: (0.0, 1.0, 0.0),   # +y = +j = East
    ISOUTH: (0.0, -1.0, 0.0),  # -y = -j = West
    IEAST: (1.0, 0.0, 0.0),    # +x = +i = South
    IWEST: (-1.0, 0.0, 0.0),   # -x = -i = North
}


@ti.data_oriented
class Domain:
    """
    3D computational domain for solar radiation simulation.
    
    The domain uses a regular grid aligned with VoxCity indices:
    - x (first index i): Row direction, increases North to South
    - y (second index j): Column direction, increases West to East
    - z (third index k): Vertical, increases Ground to Sky
    
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
        # Value is the k-index of the topmost solid cell
        self.topo_top = ti.field(dtype=ti.i32, shape=(nx, ny))
        
        # Building mask: 1 if cell is solid (building), 0 if air
        self.is_solid = ti.field(dtype=ti.i32, shape=(nx, ny, nz))
        
        # Tree mask: 1 if cell is tree canopy, 0 otherwise (for view calculations)
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
    
    # Alias for backwards compatibility
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
        
        Can be called with either range tuples or individual parameters:
        
        Args:
            x_range: (i_start, i_end) grid indices
            y_range: (j_start, j_end) grid indices
            z_range: (k_start, k_end) grid indices
            
        Or with keyword arguments:
            x_start, x_end: X grid indices
            y_start, y_end: Y grid indices
            height: Building height in meters (z_range computed from this)
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
        
        self._add_building_kernel(
            x_range[0], x_range[1],
            y_range[0], y_range[1],
            z_range[0], z_range[1]
        )
    
    @ti.kernel
    def _add_building_kernel(
        self,
        i0: ti.i32, i1: ti.i32,
        j0: ti.i32, j1: ti.i32,
        k0: ti.i32, k1: ti.i32
    ):
        for i in range(i0, i1):
            for j in range(j0, j1):
                for k in range(k0, k1):
                    if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                        self.is_solid[i, j, k] = 1
                        if k > self.topo_top[i, j]:
                            self.topo_top[i, j] = k
    
    def set_lad_from_array(self, lad_array: np.ndarray):
        """
        Set Leaf Area Density from 3D numpy array.
        
        Args:
            lad_array: 3D array (nx, ny, nz) of LAD values (m^2/m^3)
        """
        self._set_lad_kernel(lad_array)
        self._update_plant_top()
    
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
    
    def add_tree_box(
        self,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        z_range: Tuple[int, int],
        lad_value: float = 1.0
    ):
        """
        Add a box-shaped tree canopy region to the domain.
        
        This is a simpler alternative to add_tree() for rectangular tree regions.
        
        Args:
            x_range, y_range, z_range: Grid index ranges (start, end)
            lad_value: Leaf Area Density value (m^2/m^3)
        """
        self._add_tree_box_kernel(x_range[0], x_range[1], y_range[0], y_range[1], 
                                   z_range[0], z_range[1], lad_value)
        self._update_plant_top()
    
    @ti.kernel
    def _add_tree_box_kernel(self, i_min: ti.i32, i_max: ti.i32, j_min: ti.i32, 
                              j_max: ti.i32, k_min: ti.i32, k_max: ti.i32, lad: ti.f32):
        for i, j, k in ti.ndrange((i_min, i_max), (j_min, j_max), (k_min, k_max)):
            if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                self.is_tree[i, j, k] = 1
                self.lad[i, j, k] = lad

    @ti.kernel
    def _set_lad_kernel(self, lad_array: ti.types.ndarray()):
        for i, j, k in self.lad:
            self.lad[i, j, k] = lad_array[i, j, k]
    
    @ti.kernel
    def _update_plant_top(self):
        """Update plant canopy top index for each column."""
        for i, j in self.plant_top:
            max_k = 0
            # Taichi doesn't support 3-arg range, so iterate forward and track highest
            for k in range(self.nz):
                if self.lad[i, j, k] > 0.0:
                    max_k = k
            self.plant_top[i, j] = max_k
    
    def add_tree(
        self,
        center: Optional[Tuple[float, float]] = None,
        height: Optional[float] = None,
        crown_radius: Optional[float] = None,
        crown_height: Optional[float] = None,
        trunk_height: Optional[float] = None,
        max_lad: float = 1.0,
        *,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
        lad: Optional[float] = None
    ):
        """
        Add a simple tree with cylindrical trunk and spherical crown.
        
        Args:
            center: (x, y) position in meters
            height: Total tree height in meters (optional, computed from crown+trunk)
            crown_radius: Radius of crown in meters
            crown_height: Height of crown sphere in meters
            trunk_height: Height of trunk (no leaves) in meters
            max_lad: Maximum LAD at crown center
            
        Or with keyword arguments:
            center_x, center_y: Position in meters
            lad: Alias for max_lad
        """
        # Handle convenience parameters
        if center_x is not None and center_y is not None:
            center = (center_x, center_y)
        if lad is not None:
            max_lad = lad
        
        if center is None or crown_radius is None or crown_height is None or trunk_height is None:
            raise ValueError("Must provide center, crown_radius, crown_height, and trunk_height")
        # Convert to grid indices
        ci = int((center[0] - self.origin[0]) / self.dx)
        cj = int((center[1] - self.origin[1]) / self.dy)
        crown_center_k = int((trunk_height + crown_height / 2) / self.dz)
        
        ri = int(crown_radius / self.dx) + 1
        rj = int(crown_radius / self.dy) + 1
        rk = int(crown_height / 2 / self.dz) + 1
        
        self._add_tree_kernel(ci, cj, crown_center_k, ri, rj, rk, 
                              crown_radius, crown_height / 2, max_lad)
        self._update_plant_top()
    
    @ti.kernel
    def _add_tree_kernel(
        self,
        ci: ti.i32, cj: ti.i32, ck: ti.i32,
        ri: ti.i32, rj: ti.i32, rk: ti.i32,
        rx: ti.f32, rz: ti.f32, max_lad: ti.f32
    ):
        for di in range(-ri, ri + 1):
            for dj in range(-rj, rj + 1):
                for dk in range(-rk, rk + 1):
                    i = ci + di
                    j = cj + dj
                    k = ck + dk
                    
                    if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
                        # Ellipsoid distance
                        dx_norm = (di * self.dx) / rx
                        dy_norm = (dj * self.dy) / rx
                        dz_norm = (dk * self.dz) / rz
                        dist = ti.sqrt(dx_norm**2 + dy_norm**2 + dz_norm**2)
                        
                        if dist <= 1.0:
                            # LAD decreases from center
                            lad_val = max_lad * (1.0 - dist**2)
                            if lad_val > self.lad[i, j, k]:
                                self.lad[i, j, k] = lad_val
                            # Mark as tree voxel
                            self.is_tree[i, j, k] = 1
    
    @ti.func
    def get_cell_indices(self, point: Point3) -> ti.math.ivec3:
        """Get grid cell indices for a point."""
        i = ti.cast((point.x - self.origin[0]) / self.dx, ti.i32)
        j = ti.cast((point.y - self.origin[1]) / self.dy, ti.i32)
        k = ti.cast((point.z - self.origin[2]) / self.dz, ti.i32)
        return ti.math.ivec3(i, j, k)
    
    @ti.func
    def get_cell_center(self, i: ti.i32, j: ti.i32, k: ti.i32) -> Point3:
        """Get center coordinates of grid cell."""
        x = self.origin[0] + (i + 0.5) * self.dx
        y = self.origin[1] + (j + 0.5) * self.dy
        z = self.origin[2] + (k + 0.5) * self.dz
        return Point3(x, y, z)
    
    @ti.func
    def is_inside(self, point: Point3) -> ti.i32:
        """Check if point is inside domain."""
        inside = 1
        if point.x < self.x_min or point.x > self.x_max:
            inside = 0
        if point.y < self.y_min or point.y > self.y_max:
            inside = 0
        if point.z < self.z_min or point.z > self.z_max:
            inside = 0
        return inside
    
    @ti.func
    def is_cell_solid(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.i32:
        """Check if cell is solid (building or terrain)."""
        solid = 0
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            solid = self.is_solid[i, j, k]
        return solid
    
    def get_max_dist(self) -> float:
        """Get maximum ray distance (domain diagonal)."""
        import math
        return math.sqrt(
            (self.nx * self.dx)**2 + 
            (self.ny * self.dy)**2 + 
            (self.nz * self.dz)**2
        )
    
    @ti.func
    def get_lad(self, i: ti.i32, j: ti.i32, k: ti.i32) -> ti.f32:
        """Get LAD value at cell."""
        lad_val = 0.0
        if 0 <= i < self.nx and 0 <= j < self.ny and 0 <= k < self.nz:
            lad_val = self.lad[i, j, k]
        return lad_val


@ti.data_oriented
class Surfaces:
    """
    Collection of surface elements for radiation calculations.
    
    Each surface has:
    - Position (grid indices i, j, k)
    - Direction (normal direction index)
    - Area
    - Albedo (reflectivity)
    """
    
    def __init__(self, max_surfaces: int):
        """
        Initialize surface storage.
        
        Args:
            max_surfaces: Maximum number of surfaces to allocate
        """
        self.max_surfaces = max_surfaces
        self.n_surfaces = ti.field(dtype=ti.i32, shape=())
        
        # Surface geometry
        self.position = ti.Vector.field(3, dtype=ti.i32, shape=(max_surfaces,))  # i, j, k
        self.direction = ti.field(dtype=ti.i32, shape=(max_surfaces,))  # direction index
        self.center = ti.Vector.field(3, dtype=ti.f32, shape=(max_surfaces,))  # world coords
        self.normal = ti.Vector.field(3, dtype=ti.f32, shape=(max_surfaces,))
        self.area = ti.field(dtype=ti.f32, shape=(max_surfaces,))
        
        # Surface properties
        self.albedo = ti.field(dtype=ti.f32, shape=(max_surfaces,))
        
        # Radiation fluxes (shortwave only)
        self.sw_in_direct = ti.field(dtype=ti.f32, shape=(max_surfaces,))
        self.sw_in_diffuse = ti.field(dtype=ti.f32, shape=(max_surfaces,))
        self.sw_out = ti.field(dtype=ti.f32, shape=(max_surfaces,))
        
        # Sky view factor (total SVF and SVF from urban surfaces only)
        self.svf = ti.field(dtype=ti.f32, shape=(max_surfaces,))
        self.svf_urban = ti.field(dtype=ti.f32, shape=(max_surfaces,))
        
        # Shadow factor (0 = fully shadowed, 1 = fully lit)
        self.shadow = ti.field(dtype=ti.f32, shape=(max_surfaces,))
        self.shadow_factor = ti.field(dtype=ti.f32, shape=(max_surfaces,))  # same as shadow, for compatibility
        
        # Canopy transmissivity (for direct solar through vegetation)
        self.canopy_transmissivity = ti.field(dtype=ti.f32, shape=(max_surfaces,))
        
        self.n_surfaces[None] = 0
    
    @ti.func
    def add_surface(
        self,
        i: ti.i32, j: ti.i32, k: ti.i32,
        direction: ti.i32,
        center: Point3,
        normal: Vector3,
        area: ti.f32,
        albedo: ti.f32 = 0.2
    ) -> ti.i32:
        """Add a surface and return its index."""
        idx = ti.atomic_add(self.n_surfaces[None], 1)
        if idx < self.max_surfaces:
            self.position[idx] = ti.math.ivec3(i, j, k)
            self.direction[idx] = direction
            self.center[idx] = center
            self.normal[idx] = normal
            self.area[idx] = area
            self.albedo[idx] = albedo
            self.svf[idx] = 1.0
            self.shadow[idx] = 1.0
        return idx
    
    @property
    def count(self) -> int:
        """Get current number of surfaces."""
        return self.n_surfaces[None]
    
    def get_count(self) -> int:
        """Get current number of surfaces (alias for count property)."""
        return self.n_surfaces[None]
    
    @ti.kernel
    def reset_fluxes(self):
        """Reset all radiation fluxes to zero."""
        for idx in range(self.n_surfaces[None]):
            self.sw_in_direct[idx] = 0.0
            self.sw_in_diffuse[idx] = 0.0
            self.sw_out[idx] = 0.0


def extract_surfaces_from_domain(domain: Domain, 
                                  default_albedo: float = 0.2) -> Surfaces:
    """
    Extract all surface elements from domain geometry.
    
    Creates surface elements at all interfaces between solid and air cells.
    
    Args:
        domain: The computational domain
        default_albedo: Default surface albedo
    
    Returns:
        Surfaces object containing all extracted surfaces
    """
    # Estimate max surfaces (6 faces per building cell, 1 top face per ground cell)
    max_surfaces = domain.nx * domain.ny * 6  # Conservative estimate
    surfaces = Surfaces(max_surfaces)
    
    _extract_surfaces_kernel(domain, surfaces, default_albedo)
    
    return surfaces


@ti.kernel
def _extract_surfaces_kernel(
    domain: ti.template(),
    surfaces: ti.template(),
    default_albedo: ti.f32
):
    """Kernel to extract surfaces from domain."""
    dx = domain.dx
    dy = domain.dy
    dz = domain.dz
    
    for i, j, k in domain.is_solid:
        if domain.is_solid[i, j, k] == 1:
            # Check each neighbor direction for air interface
            
            # Up (z+)
            if k + 1 >= domain.nz or domain.is_solid[i, j, k + 1] == 0:
                center = domain.get_cell_center(i, j, k)
                center.z += dz / 2
                normal = Vector3(0.0, 0.0, 1.0)
                area = dx * dy
                surfaces.add_surface(i, j, k, IUP, center, normal, area,
                                     default_albedo)
            
            # North (y+)
            if j + 1 >= domain.ny or domain.is_solid[i, j + 1, k] == 0:
                center = domain.get_cell_center(i, j, k)
                center.y += dy / 2
                normal = Vector3(0.0, 1.0, 0.0)
                area = dx * dz
                surfaces.add_surface(i, j, k, INORTH, center, normal, area,
                                     default_albedo)
            
            # South (y-)
            if j - 1 < 0 or domain.is_solid[i, j - 1, k] == 0:
                center = domain.get_cell_center(i, j, k)
                center.y -= dy / 2
                normal = Vector3(0.0, -1.0, 0.0)
                area = dx * dz
                surfaces.add_surface(i, j, k, ISOUTH, center, normal, area,
                                     default_albedo)
            
            # East (x+)
            if i + 1 >= domain.nx or domain.is_solid[i + 1, j, k] == 0:
                center = domain.get_cell_center(i, j, k)
                center.x += dx / 2
                normal = Vector3(1.0, 0.0, 0.0)
                area = dy * dz
                surfaces.add_surface(i, j, k, IEAST, center, normal, area,
                                     default_albedo)
            
            # West (x-)
            if i - 1 < 0 or domain.is_solid[i - 1, j, k] == 0:
                center = domain.get_cell_center(i, j, k)
                center.x -= dx / 2
                normal = Vector3(-1.0, 0.0, 0.0)
                area = dy * dz
                surfaces.add_surface(i, j, k, IWEST, center, normal, area,
                                     default_albedo)
