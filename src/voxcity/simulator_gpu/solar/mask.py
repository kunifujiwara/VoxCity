"""
Computation Mask Utilities for Solar Irradiance Simulation

This module provides utilities for creating computation masks to limit
solar irradiance calculations to specific sub-areas within the domain.

Using a computation mask can significantly speed up calculations when
you only need results for a portion of the area.
"""

import numpy as np
from typing import List, Tuple, Optional, Union


def create_computation_mask(
    voxcity,
    method: str = 'center',
    fraction: float = 0.5,
    i_range: Optional[Tuple[int, int]] = None,
    j_range: Optional[Tuple[int, int]] = None,
    polygon_vertices: Optional[List[Tuple[float, float]]] = None,
    center: Optional[Tuple[float, float]] = None,
    radius_m: float = 500.0,
) -> np.ndarray:
    """
    Create a 2D boolean computation mask for sub-area solar calculations.
    
    This function creates a mask that specifies which grid cells should be
    computed. Cells where mask is True will be calculated; cells where mask
    is False will be set to NaN in the output.
    
    Args:
        voxcity: VoxCity object containing voxel data and metadata
        method: Method for creating the mask. Options:
            - 'center': Fraction-based center crop (uses `fraction` parameter)
            - 'indices': Direct grid index specification (uses `i_range`, `j_range`)
            - 'polygon': From geographic polygon (uses `polygon_vertices`)
            - 'buffer': Circular buffer around a point (uses `center`, `radius_m`)
            - 'full': No masking, compute entire area
        fraction: For 'center' method, the fraction of the area to include.
            0.5 means center 50% of the area. Default: 0.5
        i_range: For 'indices' method, tuple of (start_i, end_i) grid indices.
            Indices are inclusive. If None, uses full i range.
        j_range: For 'indices' method, tuple of (start_j, end_j) grid indices.
            Indices are inclusive. If None, uses full j range.
        polygon_vertices: For 'polygon' method, list of (lon, lat) coordinates
            defining the polygon boundary.
        center: For 'buffer' method, tuple of (lon, lat) for the center point.
        radius_m: For 'buffer' method, radius in meters. Default: 500.0
    
    Returns:
        2D numpy boolean array of shape (nx, ny) where True indicates cells
        to compute and False indicates cells to skip.
    
    Examples:
        # Center 50% of the area
        >>> mask = create_computation_mask(voxcity, method='center', fraction=0.5)
        
        # Specific grid region
        >>> mask = create_computation_mask(voxcity, method='indices',
        ...                                 i_range=(100, 200), j_range=(100, 200))
        
        # From polygon coordinates
        >>> mask = create_computation_mask(voxcity, method='polygon',
        ...                                 polygon_vertices=[(lon1, lat1), ...])
        
        # 500m buffer around a point
        >>> mask = create_computation_mask(voxcity, method='buffer',
        ...                                 center=(lon, lat), radius_m=500)
    """
    # Get grid dimensions from voxcity
    voxel_data = voxcity.voxels.classes
    nx, ny = voxel_data.shape[0], voxel_data.shape[1]
    
    if method == 'full':
        return np.ones((nx, ny), dtype=bool)
    
    elif method == 'center':
        return _create_center_mask(nx, ny, fraction)
    
    elif method == 'indices':
        return _create_indices_mask(nx, ny, i_range, j_range)
    
    elif method == 'polygon':
        if polygon_vertices is None:
            raise ValueError("polygon_vertices required for method='polygon'")
        return _create_polygon_mask(voxcity, polygon_vertices)
    
    elif method == 'buffer':
        if center is None:
            raise ValueError("center (lon, lat) required for method='buffer'")
        return _create_buffer_mask(voxcity, center, radius_m)
    
    else:
        raise ValueError(f"Unknown method: {method}. "
                        f"Choose from 'center', 'indices', 'polygon', 'buffer', 'full'")


def _create_center_mask(nx: int, ny: int, fraction: float) -> np.ndarray:
    """Create a mask for the center fraction of the grid."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}")
    
    mask = np.zeros((nx, ny), dtype=bool)
    
    # Calculate center region bounds
    margin_i = int(nx * (1 - fraction) / 2)
    margin_j = int(ny * (1 - fraction) / 2)
    
    # Ensure at least 1 cell is included
    start_i = max(0, margin_i)
    end_i = min(nx, nx - margin_i)
    start_j = max(0, margin_j)
    end_j = min(ny, ny - margin_j)
    
    # Ensure we have at least 1 cell even for very small fractions
    if end_i <= start_i:
        start_i = nx // 2
        end_i = start_i + 1
    if end_j <= start_j:
        start_j = ny // 2
        end_j = start_j + 1
    
    mask[start_i:end_i, start_j:end_j] = True
    return mask


def _create_indices_mask(
    nx: int, 
    ny: int, 
    i_range: Optional[Tuple[int, int]], 
    j_range: Optional[Tuple[int, int]]
) -> np.ndarray:
    """Create a mask from grid index ranges."""
    mask = np.zeros((nx, ny), dtype=bool)
    
    # Default to full range if not specified
    start_i = 0 if i_range is None else max(0, i_range[0])
    end_i = nx if i_range is None else min(nx, i_range[1] + 1)  # +1 for inclusive
    start_j = 0 if j_range is None else max(0, j_range[0])
    end_j = ny if j_range is None else min(ny, j_range[1] + 1)  # +1 for inclusive
    
    mask[start_i:end_i, start_j:end_j] = True
    return mask


def _create_polygon_mask(
    voxcity,
    polygon_vertices: List[Tuple[float, float]]
) -> np.ndarray:
    """Create a mask from a geographic polygon."""
    from shapely.geometry import Polygon, Point
    
    voxel_data = voxcity.voxels.classes
    nx, ny = voxel_data.shape[0], voxel_data.shape[1]
    meshsize = voxcity.voxels.meta.meshsize
    
    # Get the rectangle vertices for coordinate transformation
    rectangle_vertices = None
    if hasattr(voxcity, 'extras') and isinstance(voxcity.extras, dict):
        rectangle_vertices = voxcity.extras.get('rectangle_vertices', None)
    
    if rectangle_vertices is None:
        raise ValueError("voxcity must have rectangle_vertices in extras for polygon masking")
    
    # Calculate origin and transformation
    lons = [v[0] for v in rectangle_vertices]
    lats = [v[1] for v in rectangle_vertices]
    origin_lon = min(lons)
    origin_lat = min(lats)
    
    # Convert polygon to grid coordinates
    # Use approximate meters per degree at this latitude
    lat_center = (min(lats) + max(lats)) / 2
    meters_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
    meters_per_deg_lat = 110540
    
    # Create shapely polygon in grid coordinates
    polygon_grid_coords = []
    for lon, lat in polygon_vertices:
        grid_x = (lon - origin_lon) * meters_per_deg_lon / meshsize
        grid_y = (lat - origin_lat) * meters_per_deg_lat / meshsize
        polygon_grid_coords.append((grid_x, grid_y))
    
    polygon = Polygon(polygon_grid_coords)
    
    # Create mask by checking which grid cells are inside the polygon
    mask = np.zeros((nx, ny), dtype=bool)
    
    # For efficiency, first check bounding box
    minx, miny, maxx, maxy = polygon.bounds
    i_min = max(0, int(minx))
    i_max = min(nx, int(maxx) + 1)
    j_min = max(0, int(miny))
    j_max = min(ny, int(maxy) + 1)
    
    # Check each cell center within bounding box
    for i in range(i_min, i_max):
        for j in range(j_min, j_max):
            # Cell center
            cell_center = Point(i + 0.5, j + 0.5)
            if polygon.contains(cell_center):
                mask[i, j] = True
    
    return mask


def _create_buffer_mask(
    voxcity,
    center: Tuple[float, float],
    radius_m: float
) -> np.ndarray:
    """Create a circular buffer mask around a geographic point."""
    voxel_data = voxcity.voxels.classes
    nx, ny = voxel_data.shape[0], voxel_data.shape[1]
    meshsize = voxcity.voxels.meta.meshsize
    
    # Get the rectangle vertices for coordinate transformation
    rectangle_vertices = None
    if hasattr(voxcity, 'extras') and isinstance(voxcity.extras, dict):
        rectangle_vertices = voxcity.extras.get('rectangle_vertices', None)
    
    if rectangle_vertices is None:
        raise ValueError("voxcity must have rectangle_vertices in extras for buffer masking")
    
    # Calculate origin and transformation
    lons = [v[0] for v in rectangle_vertices]
    lats = [v[1] for v in rectangle_vertices]
    origin_lon = min(lons)
    origin_lat = min(lats)
    
    # Convert center to grid coordinates
    lat_center = (min(lats) + max(lats)) / 2
    meters_per_deg_lon = 111320 * np.cos(np.radians(lat_center))
    meters_per_deg_lat = 110540
    
    center_lon, center_lat = center
    center_i = (center_lon - origin_lon) * meters_per_deg_lon / meshsize
    center_j = (center_lat - origin_lat) * meters_per_deg_lat / meshsize
    
    # Convert radius to grid cells
    radius_cells = radius_m / meshsize
    
    # Create mask using distance from center
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing='ij')
    distances = np.sqrt((ii - center_i)**2 + (jj - center_j)**2)
    mask = distances <= radius_cells
    
    return mask


def draw_computation_mask(
    voxcity,
    zoom: int = 17,
):
    """
    Interactive map for drawing a computation mask polygon.
    
    This function displays an interactive map where users can draw a polygon
    to define the computation area. After drawing, call `get_mask_from_drawing()`
    with the returned polygon to create the mask.
    
    Args:
        voxcity: VoxCity object containing voxel data and metadata
        zoom: Initial zoom level for the map. Default: 17
    
    Returns:
        tuple: (map_object, drawn_polygons)
            - map_object: ipyleaflet Map instance with drawing controls
            - drawn_polygons: List that will contain drawn polygon vertices
              after the user draws on the map
    
    Example:
        # Step 1: Display map and draw polygon
        >>> m, polygons = draw_computation_mask(voxcity)
        >>> display(m)  # In Jupyter, draw a polygon on the map
        
        # Step 2: After drawing, get the mask
        >>> mask = get_mask_from_drawing(voxcity, polygons)
        
        # Step 3: Use the mask in solar calculation
        >>> solar_grid = get_global_solar_irradiance_using_epw(
        ...     voxcity, computation_mask=mask, ...
        ... )
    """
    try:
        from ipyleaflet import Map, DrawControl, TileLayer
    except ImportError:
        raise ImportError("ipyleaflet is required for interactive mask drawing. "
                         "Install with: pip install ipyleaflet")
    
    # Get rectangle vertices for map center
    rectangle_vertices = None
    if hasattr(voxcity, 'extras') and isinstance(voxcity.extras, dict):
        rectangle_vertices = voxcity.extras.get('rectangle_vertices', None)
    
    if rectangle_vertices is not None:
        lons = [v[0] for v in rectangle_vertices]
        lats = [v[1] for v in rectangle_vertices]
        center_lon = (min(lons) + max(lons)) / 2
        center_lat = (min(lats) + max(lats)) / 2
    else:
        center_lon, center_lat = -100.0, 40.0
    
    # Create map
    m = Map(center=(center_lat, center_lon), zoom=zoom, scroll_wheel_zoom=True)
    
    # Store drawn polygons
    drawn_polygons = []
    
    # Add draw control for polygons only
    draw_control = DrawControl(
        polygon={
            "shapeOptions": {
                "color": "red",
                "fillColor": "red",
                "fillOpacity": 0.3
            }
        },
        rectangle={
            "shapeOptions": {
                "color": "blue",
                "fillColor": "blue", 
                "fillOpacity": 0.3
            }
        },
        circle={},
        circlemarker={},
        polyline={},
        marker={}
    )
    
    def handle_draw(self, action, geo_json):
        if action == 'created':
            geom_type = geo_json['geometry']['type']
            if geom_type == 'Polygon':
                coordinates = geo_json['geometry']['coordinates'][0]
                vertices = [(coord[0], coord[1]) for coord in coordinates[:-1]]
                drawn_polygons.clear()  # Only keep the last polygon
                drawn_polygons.append(vertices)
                print(f"Computation area polygon drawn with {len(vertices)} vertices")
            elif geom_type == 'Rectangle':
                # Handle rectangle (treated as polygon)
                coordinates = geo_json['geometry']['coordinates'][0]
                vertices = [(coord[0], coord[1]) for coord in coordinates[:-1]]
                drawn_polygons.clear()
                drawn_polygons.append(vertices)
                print(f"Computation area rectangle drawn with {len(vertices)} vertices")
    
    draw_control.on_draw(handle_draw)
    m.add_control(draw_control)
    
    return m, drawn_polygons


def get_mask_from_drawing(
    voxcity,
    drawn_polygons: List,
) -> np.ndarray:
    """
    Create a computation mask from interactively drawn polygon(s).
    
    Args:
        voxcity: VoxCity object containing voxel data and metadata
        drawn_polygons: List of polygon vertices from draw_computation_mask()
    
    Returns:
        2D numpy boolean array mask
    
    Example:
        >>> m, polygons = draw_computation_mask(voxcity)
        >>> display(m)  # Draw polygon
        >>> mask = get_mask_from_drawing(voxcity, polygons)
    """
    if not drawn_polygons:
        raise ValueError("No polygons drawn. Draw a polygon on the map first.")
    
    # Use the first (or only) polygon
    polygon_vertices = drawn_polygons[0] if isinstance(drawn_polygons[0], list) else drawn_polygons
    
    return create_computation_mask(voxcity, method='polygon', polygon_vertices=polygon_vertices)


def visualize_computation_mask(
    voxcity,
    mask: np.ndarray,
    show_plot: bool = True,
) -> Optional[np.ndarray]:
    """
    Visualize a computation mask overlaid on the voxcity grid.
    
    Args:
        voxcity: VoxCity object
        mask: 2D boolean mask array
        show_plot: Whether to display the plot. Default: True
    
    Returns:
        The mask array (for chaining)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib required for visualization")
        return mask
    
    if show_plot:
        voxel_data = voxcity.voxels.classes
        
        # Create a simple ground-level view
        ground_level = np.zeros((voxel_data.shape[0], voxel_data.shape[1]))
        for i in range(voxel_data.shape[0]):
            for j in range(voxel_data.shape[1]):
                # Find highest non-zero value
                col = voxel_data[i, j, :]
                non_zero = np.where(col != 0)[0]
                if len(non_zero) > 0:
                    ground_level[i, j] = col[non_zero[-1]]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Ground level view
        axes[0].imshow(np.flipud(ground_level.T), cmap='gray', alpha=0.5)
        axes[0].set_title('Voxcity Grid')
        axes[0].axis('off')
        
        # Right: Mask overlay
        axes[1].imshow(np.flipud(ground_level.T), cmap='gray', alpha=0.3)
        mask_display = np.ma.masked_where(~mask.T, np.ones_like(mask.T))
        axes[1].imshow(np.flipud(mask_display), cmap='Reds', alpha=0.5)
        axes[1].set_title(f'Computation Mask ({np.sum(mask)} of {mask.size} cells)')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return mask


def get_mask_info(mask: np.ndarray) -> dict:
    """
    Get information about a computation mask.
    
    Args:
        mask: 2D boolean mask array
    
    Returns:
        Dictionary with mask statistics
    """
    total_cells = mask.size
    active_cells = int(np.sum(mask))
    
    return {
        'shape': mask.shape,
        'total_cells': total_cells,
        'active_cells': active_cells,
        'inactive_cells': total_cells - active_cells,
        'coverage_fraction': active_cells / total_cells if total_cells > 0 else 0,
        'coverage_percent': 100 * active_cells / total_cells if total_cells > 0 else 0,
    }
