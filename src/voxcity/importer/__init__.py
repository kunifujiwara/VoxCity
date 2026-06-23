"""VoxCity importer subpackage: import external 3D geometry into VoxCity models."""

from .rhino_obj import add_buildings_from_obj

__all__ = ["add_buildings_from_obj"]
