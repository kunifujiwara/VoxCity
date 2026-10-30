# VoxCity

**Generate grid-based 3D city models anywhere on Earth from open geospatial
data — then simulate solar, view, and microclimate.**

VoxCity turns open building, land cover, canopy, and terrain data into a
semantic 3D voxel model you can visualize, export, and run urban simulations on.
No account is needed: the default sources are open and free to download.

```{button-link} https://colab.research.google.com/github/kunifujiwara/VoxCity/blob/main/docs/examples/demo_quickstart.ipynb
:color: primary

Try it in Google Colab, nothing to install
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 Install
:link: installation
:link-type: doc

Install with pip or conda. Google Earth Engine is optional.
:::

:::{grid-item-card} 📚 Tutorials
:link: examples/index
:link-type: doc

Hands-on notebooks: generate your first model, then visualize and export it.
:::

:::{grid-item-card} 🛠️ How-to guides
:link: guides/index
:link-type: doc

Task-focused recipes for data sources, ENVI-met, Rhino and the optional Earth Engine.
:::

:::{grid-item-card} 💡 Concepts
:link: concepts/index
:link-type: doc

Understand the voxel model and the coordinate systems VoxCity uses.
:::

:::{grid-item-card} 📖 Reference
:link: reference/index
:link-type: doc

Data sources, land cover classes, and the full `voxcity` API.
:::

:::{grid-item-card} 🔖 Bibliography
:link: bibliography
:link-type: doc

How to cite VoxCity and credit the datasets you use.
:::

::::

## Quick start

```bash
pip install voxcity
```

```python
from voxcity.generator import get_voxcity

# A ~450 m box in Seattle (lon, lat corners)
rectangle_vertices = [
    (-122.3359, 47.5983), (-122.3359, 47.6028),
    (-122.3292, 47.6028), (-122.3292, 47.5983),
]

# Sources are auto-selected by location; no account needed
voxcity = get_voxcity(rectangle_vertices, meshsize=5)
```

New to VoxCity? Start with the {doc}`installation guide <installation>`, then
the {doc}`quickstart tutorial <examples/demo_quickstart>`.

```{toctree}
:hidden:
:caption: Get Started

installation
```

```{toctree}
:hidden:
:caption: Tutorials

examples/index
```

```{toctree}
:hidden:
:caption: How-to Guides

guides/index
```

```{toctree}
:hidden:
:caption: Concepts

concepts/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:caption: Project Information

changelog
contributing
conduct
bibliography
```