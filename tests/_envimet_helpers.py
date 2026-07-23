"""Shared helper(s) for ENVI-met exporter tests.

Reuses the minimal ``VoxCity`` construction pattern from ``tests/conftest.py``
(``make_city``) and customizes it to include a 3D-plant-eligible cell (a
tree_canopy.top > 0 cell sitting on a building_height == 0 cell), so
``create_xml_content`` emits at least one ``<3Dplants>`` block referencing a
``HxxW01`` plant ID.
"""

from tests.conftest import make_city


def make_minimal_city_with_trees():
    """Return a small VoxCity with one tree canopy cell (no building beneath it)."""
    city = make_city(shape=(4, 4, 3), meshsize=2.0)
    # Put a tree in a corner cell that has no building underneath.
    city.tree_canopy.top[1, 1] = 5.0
    return city
