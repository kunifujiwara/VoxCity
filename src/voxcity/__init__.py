__author__ = """Kunihiko Fujiwara"""
__email__ = 'kunihiko@nus.edu.sg'

# Dynamically resolve version from package metadata so it stays in sync
# with pyproject.toml without manual updates.
try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version("voxcity")
except Exception:
    __version__ = "0.0.0"  # fallback for editable installs without metadata

# Keep package __init__ lightweight to avoid import-time failures.
# Heavy modules are not imported eagerly; the coordinate-contract symbols
# below are re-exported lazily via module __getattr__ so a bare
# ``import voxcity`` stays lightweight.
# Downstream modules should import directly from their subpackages, e.g.:
#   from voxcity.geoprocessor.draw import draw_rectangle_map_cityname

# Coordinate-contract symbols are re-exported LAZILY: accessing e.g.
# ``voxcity.AXES`` imports voxcity.utils.orientation on first use, but a bare
# ``import voxcity`` stays lightweight (importing voxcity.utils eagerly would
# pull in shapely/rtree/pandas via utils/__init__'s star-imports).
_LAZY_ATTRS = {
    "AXES": ("voxcity.utils.orientation", "AXES"),
    "direction_to_axis_vector": ("voxcity.utils.orientation", "direction_to_axis_vector"),
    "check_axes": ("voxcity.utils.orientation", "check_axes"),
    "GridProjector": ("voxcity.utils.projector", "GridProjector"),
}


def __getattr__(name):
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(target[0]), target[1])


def __dir__():
    return sorted(list(globals()) + list(_LAZY_ATTRS))


__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "AXES",
    "direction_to_axis_vector",
    "check_axes",
    "GridProjector",
]
