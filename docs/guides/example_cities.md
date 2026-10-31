# Start from a ready-made city

VoxCity ships a catalog of prebuilt city models, hosted on
[Hugging Face](https://huggingface.co/datasets/kunifujiwara/voxcity-example-cities).
Use one to try the simulators and exporters without waiting for data downloads,
for teaching, or as a fixed benchmark everyone can reproduce.

## List and load

```python
from voxcity.examples import list_example_cities, load_example_city

for c in list_example_cities():
    print(c["name"], "-", c["title"])

city = load_example_city("tokyo")
```

The first call downloads the model (about 1 MB) and checks its SHA-256 checksum.
Later calls read the cached copy. Names ignore case, and spaces or hyphens work
too, so `"New York"`, `"new-york"` and `"new_york"` are the same city.

The result is an ordinary `VoxCity` model. Everything else in the docs works on it:

```python
from voxcity.visualizer import visualize_voxcity
from voxcity.simulator.view import get_view_index

visualize_voxcity(city, mode="interactive")
gvi = get_view_index(city, mode="green")
```

## What is in each model

Every model covers 1 km × 1 km of a city centre in 5 m cubes. It holds the
voxel classes, building heights, IDs and footprints, land cover, terrain and
tree canopy. Each catalog entry also records the centre, grid shape, building
count and the data source used for each layer.

All sources need no account:

| Layer | Source |
|---|---|
| Buildings | OpenStreetMap, plus Microsoft Building Footprints in the USA, Europe and Australia |
| Land cover | OpenStreetMap |
| Tree canopy | Meta/WRI High Resolution Canopy Height Maps v2 |
| Terrain | USGS 3DEP 10 m (USA), GSI DEM (Japan), FABDEM elsewhere |

The dataset is released under the Open Database License (ODbL) 1.0, as a
derived database of OpenStreetMap. Models outside Japan and the US also use
FABDEM terrain, licensed CC BY-NC-SA 4.0, so use them for non-commercial work
only. Credit the sources when you publish results.

## Cache and mirrors

| Setting | Effect |
|---|---|
| `cache_dir=` | Folder for downloads. Default: `~/.cache/voxcity/examples` |
| `VOXCITY_CACHE_DIR` | Environment variable for the same folder, used when `cache_dir` is not given |
| `force_download=True` | Download again even when a verified copy is cached |
| `revision=` | Branch, tag or commit of the dataset repository, to pin a version |
| `VOXCITY_EXAMPLES_REPO` | Use a fork or mirror of the dataset |

`example_city_path(name)` downloads a model and returns the local HDF5 path
without loading it.

## Rebuild or extend the catalog

The models are built by `scripts/build_example_cities.py` in the repository.
It pins every source, writes the HDF5 files, previews, `cities.csv` and the
dataset card, and refreshes the catalog bundled with the package.
