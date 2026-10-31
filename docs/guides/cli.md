# Use the command line

`pip install voxcity` also installs a `voxcity` command. It covers the path
that needs no Python: build a model for an area, save it, and export it to the
format your simulation tool reads. Simulations, custom data and plots stay in
Python.

## Build a model

```bash
# lon/lat bounds: WEST SOUTH EAST NORTH
voxcity generate --bbox -122.3359 47.5983 -122.3292 47.6028 --mesh 5 --out seattle.h5

# or a square of a given size around a point
voxcity generate --center 139.765 35.681 --size 500 --out tokyo.h5
```

Data sources are chosen for the location, as in `get_voxcity`, and no account
is needed. Override any of them with `--building-source`, `--building-complement`,
`--land-cover-source`, `--canopy-source` and `--dem-source`, using the names from
{doc}`data_sources`:

```bash
voxcity generate --bbox 2.33 48.86 2.35 48.87 --dem-source FABDEM --land-cover-source OpenStreetMap
```

Downloaded rasters go to a temporary folder that is removed afterwards; pass
`--work-dir DIR` to keep them. Add `-q` before the command to print only
warnings.

## Export

Add `--export` to `generate`, or export a saved model later:

```bash
voxcity generate --bbox ... --export obj,envimet
voxcity export seattle.h5 --to palm,geotiff --export-dir exports --name seattle
```

| Format | Files written | Reads it |
|---|---|---|
| `envimet` | `NAME.INX`, `projectdatabase.edb` | ENVI-met |
| `palm` | `NAME_static` | PALM |
| `obj` | `NAME.obj`, `NAME.mtl` | Rhino, Blender, most 3D tools |
| `vox` | `NAME_*.vox` | MagicaVoxel |
| `netcdf` | `NAME.nc` | xarray, Panoply |
| `geotiff` | `NAME_*.tif` | QGIS, ArcGIS |

Files land in `export/` next to the model unless `--export-dir` says otherwise.
Exporter options such as ENVI-met's telescoping grid or PALM's leaf area
density keep their defaults; use the Python functions in
{doc}`envimet_export` for those.

## Ready-made cities

```bash
voxcity examples list
voxcity examples fetch tokyo --export obj
```

`fetch` downloads the model once, caches it, and copies it to `tokyo.h5` (or
`--out FILE`). See {doc}`example_cities` for what the models contain.

## Inspect a model

```bash
voxcity info seattle.h5
```

prints the grid size, voxel size, building count, terrain range, extent and
the data source of each layer.

## Exit codes

`0` on success, `1` when a download or export fails (the message says why),
`2` for a usage error.
