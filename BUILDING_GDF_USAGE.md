# Using GeoDataFrame Input for Building Data

This document explains how to use the new GeoDataFrame input functionality in VoxCity for building height data.

## Overview

The `get_building_height_grid` and `get_voxcity` functions now support direct input of GeoDataFrames containing building footprints and heights. This allows users to:

- Use custom building datasets
- Process pre-existing GeoDataFrames without downloading from external sources
- Integrate with other geospatial workflows
- Test with sample data

## GeoDataFrame Requirements

Your GeoDataFrame must contain the following columns:

### Required Columns:
- `geometry`: Shapely Polygon geometries representing building footprints
- `height`: Numeric values representing building heights in meters

### Optional Columns:
- `id`: Unique identifier for each building
- `min_height`: Minimum height of building (defaults to 0)
- `is_inner`: Boolean indicating if the building is an inner courtyard (defaults to False)

### Example GeoDataFrame Structure:
```python
import geopandas as gpd
from shapely.geometry import Polygon

# Sample building data
buildings = [
    Polygon([(0, 0), (10, 0), (10, 15), (0, 15)]),
    Polygon([(15, 5), (25, 5), (25, 20), (15, 20)]),
]

heights = [20.0, 15.0]

gdf = gpd.GeoDataFrame({
    'geometry': buildings,
    'height': heights,
    'id': [1, 2],
    'min_height': [0, 0],
    'is_inner': [False, False]
}, crs='EPSG:4326')
```

## Usage Examples

### 1. Direct Usage with get_building_height_grid

```python
from src.voxcity.generator import get_building_height_grid

# Define area of interest
rectangle_vertices = [
    (0, 0),    # Bottom-left
    (50, 0),   # Bottom-right
    (50, 25),  # Top-right
    (0, 25)    # Top-left
]

# Generate building height grid
building_height_grid, building_min_height_grid, building_id_grid, filtered_buildings = get_building_height_grid(
    rectangle_vertices=rectangle_vertices,
    meshsize=2.0,
    source="GeoDataFrame",
    output_dir="output",
    building_gdf=your_building_gdf,
    gridvis=True
)
```

### 2. Complete Voxel City Generation

```python
from src.voxcity.generator import get_voxcity

# Generate complete voxel city model
voxcity_grid, building_height_grid, building_min_height_grid, building_id_grid, \
canopy_height_grid, land_cover_grid, dem_grid, building_gdf_result = get_voxcity(
    rectangle_vertices=rectangle_vertices,
    building_source="GeoDataFrame",
    land_cover_source="Static",
    canopy_height_source="Static",
    dem_source="Flat",
    meshsize=2.0,
    building_gdf=your_building_gdf,
    output_dir="output",
    gridvis=True,
    voxelvis=True
)
```

### 3. Loading from File

```python
import geopandas as gpd

# Load building data from file
building_gdf = gpd.read_file("path/to/your/buildings.gpkg")

# Use with voxcity
voxcity_grid, ... = get_voxcity(
    rectangle_vertices=rectangle_vertices,
    building_source="GeoDataFrame",
    building_gdf=building_gdf,
    # ... other parameters
)
```

## Coordinate System

- Your GeoDataFrame should be in WGS84 (EPSG:4326) coordinate system
- If your data is in a different CRS, reproject it before use:
  ```python
  gdf = gdf.to_crs('EPSG:4326')
  ```

## Data Validation

The function will automatically:
- Filter buildings that intersect with the specified area
- Handle invalid geometries by attempting to fix them
- Process overlapping buildings using configurable thresholds
- Apply height complementation if specified

## Error Handling

Common issues and solutions:

1. **Missing height column**: Ensure your GeoDataFrame has a 'height' column
2. **Invalid geometries**: The function will attempt to fix invalid geometries automatically
3. **Wrong CRS**: Convert your data to EPSG:4326 before use
4. **No buildings in area**: Ensure your building footprints intersect with the rectangle_vertices

## Complete Example

See `example_building_gdf.py` for a complete working example that demonstrates:
- Creating sample building data
- Using the GeoDataFrame with get_building_height_grid
- Generating a complete voxel city model

## Integration with Other Data Sources

You can combine GeoDataFrame input with complementary data sources:

```python
# Use GeoDataFrame as primary source with complementary height data
building_height_grid, ... = get_building_height_grid(
    rectangle_vertices=rectangle_vertices,
    meshsize=2.0,
    source="GeoDataFrame",
    building_gdf=your_building_gdf,
    building_complementary_source="Open Building 2.5D Temporal",
    complement_height=10.0  # Default height for buildings without height data
)
```

This allows you to use your custom building footprints while supplementing missing height information from external sources. 