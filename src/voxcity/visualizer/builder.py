from __future__ import annotations

import numpy as np
import trimesh

from ..models import MeshModel, MeshCollection, VoxelGrid
from ..geoprocessor.mesh import create_city_meshes
from .palette import get_voxel_color_map


class MeshBuilder:
    """Build mesh collections from voxel grids for rendering/export."""

    @staticmethod
    def from_voxel_grid(voxel_grid: VoxelGrid, meshsize: float, voxel_color_map: "str|dict" = "default",
                        include_classes=None, exclude_classes=None) -> MeshCollection:
        if isinstance(voxel_color_map, dict):
            vox_dict = voxel_color_map
        else:
            vox_dict = get_voxel_color_map(voxel_color_map)

        meshes = create_city_meshes(
            voxel_grid.classes,
            vox_dict,
            meshsize=meshsize,
            include_classes=include_classes,
            exclude_classes=exclude_classes,
        )

        collection = MeshCollection()
        for key, m in meshes.items():
            if m is None:
                continue
            colors = getattr(m.visual, 'face_colors', None)
            collection.add(str(key), MeshModel(
                vertices=m.vertices.copy(),
                faces=m.faces.copy(),
                colors=colors.copy() if colors is not None else None,
                name=str(key)
            ))
        return collection


