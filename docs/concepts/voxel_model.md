# The voxel city model

VoxCity represents a city as a **grid-based 3D voxel model**: a regular 3D array
where every cell (voxel) carries a semantic class such as building, tree, road,
or water. This page explains how that representation is built and why it is
useful.

## From geospatial layers to voxels

A model is assembled from four input layers, each fetched from open data
sources (see the {doc}`data sources reference <../reference/data_sources>`):

- **Building footprints and heights** — extruded into solid building voxels.
- **Tree canopy height** — turned into vegetation voxels above the ground.
- **Land cover** — classifies the ground surface (road, water, developed space,
  and so on; see {doc}`land cover classes <../reference/land_cover>`).
- **Terrain elevation (DEM)** — sets the ground height beneath everything.

These layers are rasterized onto a common horizontal grid and then stacked
vertically to fill the voxel array.

## The grid

The horizontal grid is defined by the target area (`rectangle_vertices`) and a
**mesh size** in meters. Every voxel is a cube of `meshsize` on each side, so a
smaller mesh size yields finer detail at the cost of more voxels.

The grid uses a simple, consistent indexing invariant:

> `voxcity_grid[i, j, k]` corresponds to scene coordinates
> `(i · meshsize, j · meshsize, k · meshsize)`.

This makes the model easy to reason about and to export to downstream formats
(OBJ, MagicaVoxel, ENVI-met) and simulators (solar, view index, network).

## Why voxels

A uniform voxel grid gives every analysis the same discrete, addressable
structure:

- Simulations (solar irradiance, visibility, wind/microclimate) operate directly
  on the grid.
- Exports to voxel and mesh formats are straightforward.
- Semantic classes travel with geometry, so results stay interpretable.

## Related

- {doc}`coordinate_systems` — how grid coordinates relate to real-world
  longitude/latitude.
- {doc}`../reference/data_sources` — the datasets behind each layer.
