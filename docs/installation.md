# Installation

VoxCity runs on **Python 3.10-3.13**. GDAL is the one dependency that is easiest to
install through conda; everything else is available from PyPI.

## Requirements

- Python 3.10-3.13
- GDAL (install via conda-forge, see below)

No account is required. The default data sources (OpenStreetMap, Overture,
Global Building Atlas, ESA WorldCover, CHMv2 canopy, Copernicus DEM, FABDEM and
others) download freely. Google Earth Engine is optional and adds more sources.

## Install with conda + pip (recommended)

```bash
conda create --name voxcity python=3.12
conda activate voxcity
conda install -c conda-forge gdal timezonefinder
pip install voxcity
```

## Install on Google Colab

VoxCity installs directly with pip on Colab:

```text
!pip install voxcity
```

Or skip installing and open the
[quickstart notebook in Colab](https://colab.research.google.com/github/kunifujiwara/VoxCity/blob/main/docs/examples/demo_quickstart.ipynb).

## Optional: GPU acceleration

```bash
pip install "voxcity[gpu]"   # adds Taichi; enables voxcity.simulator_gpu
```

## Verify the installation

```python
import voxcity
print(voxcity.__version__)
```

## Optional: set up Google Earth Engine

Skip this unless you want the Earth Engine sources (Open Buildings 2.5D,
Dynamic World, ESRI land cover, and several high-resolution national DEMs).
Create an
Earth Engine–enabled Cloud Project by following the
[official setup guide](https://developers.google.com/earth-engine/cloud/earthengine_cloud_project_setup),
then authenticate:

```bash
# Local environment
earthengine authenticate
```

```text
# Google Colab: click the displayed link, generate a token, then paste it
!earthengine authenticate --auth_mode=notebook
```

For a full walkthrough of authentication and project configuration, see the
{doc}`Earth Engine setup guide <guides/earth_engine>`.

## Next steps

- Run the {doc}`quickstart tutorial <examples/demo_quickstart>`, then the other {doc}`tutorial notebooks <examples/index>`.
- Browse the {doc}`how-to guides <guides/index>` for task-focused recipes.
