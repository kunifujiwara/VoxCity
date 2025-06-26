```{raw} html
<style>
@import url("_static/custom.css");
</style>
```

# VoxCity

A seamless framework for open geospatial data integration, grid-based semantic 3D city model generation, and urban environment simulation.

## Overview

VoxCity is a comprehensive Python package that enables the generation of 3D voxel-based city models from open geospatial data sources. It provides tools for urban environment simulation, including solar analysis, view index calculations, and landmark visibility mapping.

## Key Features

- **Multi-Source Data Integration**: Supports various open geospatial data sources
- **3D Voxel Generation**: Creates semantic 3D city models with customizable resolution
- **Export Formats**: ENVI-MET, MagicaVoxel VOX, and OBJ formats
- **Urban Simulations**: Solar analysis, view indices, and network analysis
- **Earth Engine Integration**: Leverages Google Earth Engine for large-scale processing

## Quick Start

```python
import voxcity
from voxcity.generator import get_voxcity

# Define your target area
rectangle_vertices = [
    (-122.33587348582083, 47.59830044521263),
    (-122.32922451417917, 47.60279755390168)
]

# Generate voxel city model
voxcity_grid, building_height_grid, building_id_grid, \
canopy_height_grid, land_cover_grid, dem_grid, building_gdf = get_voxcity(
    rectangle_vertices,
    building_source='OpenStreetMap',
    land_cover_source='OpenStreetMap',
    canopy_height_source='High Resolution 1m Global Canopy Height Maps',
    dem_source='DeltaDTM',
    meshsize=5
)
```

For detailed installation and usage instructions, see the [examples](examples/index) section.

```{toctree}
:maxdepth: 1
:hidden:
:caption: Getting Started:
example
autoapi/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Tutorial:
examples/index
```

```{toctree}
:maxdepth: 1
:hidden:
:caption: Project Information:
changelog
contributing
conduct
references
```