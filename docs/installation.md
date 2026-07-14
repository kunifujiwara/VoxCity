# Installation

VoxCity runs on **Python 3.12**. GDAL is the one dependency that is easiest to
install through conda; everything else is available from PyPI.

## Requirements

- Python 3.12
- GDAL (install via conda-forge, see below)
- A Google Earth Engine account for the cloud-served data sources
  (see {doc}`guides/earth_engine`)

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

## Verify the installation

```python
import voxcity
print(voxcity.__version__)
```

## Set up Google Earth Engine

Many VoxCity data sources are served through Google Earth Engine. Create an
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

- Follow the {doc}`tutorial notebooks <examples/index>` to generate your first model.
- Browse the {doc}`how-to guides <guides/index>` for task-focused recipes.
