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
# Re-exports of heavy modules/classes are intentionally omitted here.
# Downstream modules should import directly from their subpackages, e.g.:
#   from voxcity.geoprocessor.draw import draw_rectangle_map_cityname

__all__ = [
    "__author__",
    "__email__",
    "__version__",
]
