# Choose and configure data sources

VoxCity builds each city model from four layers — building footprints/heights,
land cover, tree canopy height, and terrain elevation. This guide shows how to
select the source for each layer. For the full catalog of datasets and their
coverage, see the {doc}`Data sources reference <../reference/data_sources>`.

## Let VoxCity choose (auto mode)

If you omit the source arguments, VoxCity selects appropriate datasets based on
your location:

```python
from voxcity.generator import get_voxcity

voxcity = get_voxcity(rectangle_vertices, meshsize=5)
```

## Specify every source (custom mode)

Pass each source explicitly to control exactly which datasets are used:

```python
from voxcity.generator import get_voxcity

voxcity = get_voxcity(
    rectangle_vertices,
    meshsize=5,
    building_source='OpenStreetMap',
    land_cover_source='OpenStreetMap',
    canopy_height_source='High Resolution 1m Global Canopy Height Maps',
    dem_source='DeltaDTM',
)
```

## Mix specified and auto sources (hybrid mode)

Specify only the layers you care about and let VoxCity auto-select the rest:

```python
from voxcity.generator import get_voxcity

voxcity = get_voxcity(
    rectangle_vertices,
    meshsize=5,
    building_source='Overture',  # custom
    # land_cover_source, canopy_height_source, dem_source auto-selected
)
```

## Choosing a source

Pick sources based on your area's coverage and the resolution you need:

- **Buildings** — `OpenStreetMap` has the broadest reach; regional datasets such
  as `EUBUCCO` (Europe), Microsoft Building Footprints, or `Overture` can give
  better coverage or heights in specific regions.
- **Land cover** — global options (`ESA WorldCover`, `ESRI`, `Dynamic World`)
  work anywhere at 10 m; higher-resolution regional sources exist for the US and
  Japan.
- **Canopy height** — the 1 m Global Canopy Height maps give the finest detail;
  the ETH 10 m product is a global fallback.
- **Terrain** — `DeltaDTM` and `FABDEM` are global; 1 m national LiDAR DEMs are
  available for the US, England, France, and Australia.

Consult the {doc}`Data sources reference <../reference/data_sources>` for exact
coverage, resolution, and provenance before committing to a source.
