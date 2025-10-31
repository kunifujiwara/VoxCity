__author__ = """Kunihiko Fujiwara"""
__email__ = 'kunihiko@nus.edu.sg'
__version__ = '0.1.0'

# Keep package __init__ lightweight to avoid import-time failures.
# Re-exports of heavy modules/classes are intentionally omitted here.
# Downstream modules should import directly from their subpackages, e.g.:
#   from voxcity.geoprocessor.draw import draw_rectangle_map_cityname

__all__ = [
    "__author__",
    "__email__",
    "__version__",
]
