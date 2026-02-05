# VoxCity Tutorial Notebooks

Welcome to the VoxCity tutorials! These notebooks will guide you through generating 3D voxel-based urban models and running various urban simulations.

## 🚀 Getting Started

| Notebook | Description | Difficulty |
|----------|-------------|------------|
| [demo_quickstart](demo_quickstart.ipynb) | **Start here!** Quick 5-minute introduction with automatic data source selection | ⭐ Beginner |
| [demo_basic](demo_basic.ipynb) | Complete guide to all features, data sources, and options | ⭐⭐ Intermediate |
| [demo_data_sources](demo_data_sources.ipynb) | Comprehensive overview of all available data sources | ⭐⭐ Intermediate |

## 🔬 Urban Simulations

| Notebook | Description | Applications |
|----------|-------------|--------------|
| [demo_solar](demo_solar.ipynb) | Solar irradiance analysis (ground & building surfaces) | Energy, comfort, shading |
| [demo_view](demo_view.ipynb) | Green View Index & Sky View Index | Walkability, urban greening |
| [demo_landmark](demo_landmark.ipynb) | Landmark visibility analysis | Planning, tourism, heritage |

## 🎨 Visualization & Export

| Notebook | Description | Output Formats |
|----------|-------------|----------------|
| [demo_3d_visualization](demo_3d_visualization.ipynb) | 3D rendering options and simulation overlays | Images, interactive views |
| [demo_obj](demo_obj.ipynb) | Export to OBJ format for 3D software | Blender, Rhino, SketchUp |
| [demo_envi-met](demo_envi-met.ipynb) | Export to ENVI-met INX format | CFD simulations |

## 📚 Learning Path

### For Beginners
1. Start with **demo_quickstart** to generate your first model
2. Explore **demo_3d_visualization** to see your model in 3D
3. Try **demo_view** for a simple simulation

### For Advanced Users
1. Read **demo_data_sources** to understand all options
2. Follow **demo_basic** for full control
3. Use **demo_solar** for detailed irradiance analysis
4. Export with **demo_envi-met** for CFD workflows

## 🌐 Data Requirements

Most tutorials require **Google Earth Engine** access for downloading geospatial data. See the [data sources notebook](demo_data_sources.ipynb) for:
- Authentication instructions
- Available data sources by region
- Offline alternatives

## 💡 Tips

- **Start small**: Use areas under 500m × 500m while learning
- **Mesh size**: 5m is a good default; use 2-3m for detail, 10m for large areas
- **Save your work**: Use `save_voxcity()` to avoid re-downloading data

```{toctree}
:maxdepth: 1
:caption: Tutorials:

demo_quickstart
demo_basic
demo_data_sources
demo_view
demo_solar
demo_landmark
demo_3d_visualization
demo_obj
demo_envi-met
```