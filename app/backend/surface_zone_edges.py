from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ZoneSpec
from .surface_zones import make_surface_face_key

AXIS_EPSILON_M = 1e-6
BUILDING_CLASS = -3
AIR_CLASS = 0
TREE_CLASS = -2

PLANE_ORDER = ("+x", "-x", "+y", "-y", "+z", "-z")
PLANE_NORMALS = {
    "+x": (1.0, 0.0, 0.0),
    "-x": (-1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-z": (0.0, 0.0, -1.0),
}
# Adjacency offsets in (u, v, k) array space.
# x = east = v-axis, y = north = u-axis, z = up = k-axis.
PLANE_OFFSETS = {
    "+x": (0, 1, 0),
    "-x": (0, -1, 0),
    "+y": (1, 0, 0),
    "-y": (-1, 0, 0),
    "+z": (0, 0, 1),
    "-z": (0, 0, -1),
}


@dataclass(frozen=True)
class VoxelSurfaceRecord:
    building_id: int
    plane: str
    voxel_uvw: tuple[int, int, int]
    surface_kind: str
    orientation: str | None
    corners: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    face_keys: tuple[str, str]


# ---------------------------------------------------------------------------
# Plane classification helpers
# ---------------------------------------------------------------------------

def _surface_kind(plane: str) -> str:
    if plane == "+z":
        return "roof"
    if plane == "-z":
        return "bottom"
    return "wall"


def _orientation(plane: str) -> str | None:
    return {
        "+x": "E",
        "-x": "W",
        "+y": "N",
        "-y": "S",
    }.get(plane)


# ---------------------------------------------------------------------------
# Scene-space corner generation
# ---------------------------------------------------------------------------
# Coordinate convention: array axes (u, v, k) map to scene (x=v*ms, y=u*ms, z=k*ms).
# This matches create_voxel_mesh which does:
#   scene_coords = [face_coords[:, 1], face_coords[:, 0], face_coords[:, 2]]
# i.e. x = east = v-axis, y = north = u-axis.

def _corners_for_face(u: int, v: int, k: int, plane: str, meshsize: float):
    x0, x1 = v * meshsize, (v + 1) * meshsize
    y0, y1 = u * meshsize, (u + 1) * meshsize
    z0, z1 = k * meshsize, (k + 1) * meshsize
    if plane == "+x":
        return ((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1))
    if plane == "-x":
        return ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))
    if plane == "+y":
        return ((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0))
    if plane == "-y":
        return ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))
    if plane == "+z":
        return ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))
    if plane == "-z":
        return ((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0))
    raise ValueError(f"Unsupported plane: {plane}")


# ---------------------------------------------------------------------------
# Record extraction
# ---------------------------------------------------------------------------

def _is_exposed(voxels: np.ndarray, u: int, v: int, k: int, plane: str) -> bool:
    du, dv, dk = PLANE_OFFSETS[plane]
    nu, nv, nk = u + du, v + dv, k + dk
    if nu < 0 or nv < 0 or nk < 0 or nu >= voxels.shape[0] or nv >= voxels.shape[1] or nk >= voxels.shape[2]:
        return True
    return int(voxels[nu, nv, nk]) in {AIR_CLASS, TREE_CLASS}


def _triangle_keys(building_id: int, corners, normal, first_face_index: int) -> tuple[str, str]:
    tri1 = np.asarray([corners[0], corners[1], corners[2]], dtype=float)
    tri2 = np.asarray([corners[0], corners[2], corners[3]], dtype=float)
    return (
        make_surface_face_key(building_id, tri1.mean(axis=0), normal, first_face_index),
        make_surface_face_key(building_id, tri2.mean(axis=0), normal, first_face_index + 1),
    )


def extract_voxel_surface_records(voxels: np.ndarray, building_id_grid: np.ndarray, meshsize: float) -> list[VoxelSurfaceRecord]:
    if getattr(voxels, "ndim", 0) != 3:
        return []
    if getattr(building_id_grid, "ndim", 0) != 2:
        return []
    if building_id_grid.shape != voxels.shape[:2]:
        return []

    records: list[VoxelSurfaceRecord] = []
    face_index = 0
    for u, v, k in np.argwhere(voxels == BUILDING_CLASS):
        building_id = int(building_id_grid[int(u), int(v)])
        for plane in PLANE_ORDER:
            if not _is_exposed(voxels, int(u), int(v), int(k), plane):
                continue
            corners = _corners_for_face(int(u), int(v), int(k), plane, float(meshsize))
            normal = PLANE_NORMALS[plane]
            keys = _triangle_keys(building_id, corners, normal, face_index)
            records.append(VoxelSurfaceRecord(
                building_id=building_id,
                plane=plane,
                voxel_uvw=(int(u), int(v), int(k)),
                surface_kind=_surface_kind(plane),
                orientation=_orientation(plane),
                corners=corners,
                face_keys=keys,
            ))
            face_index += 2
    return records


# ---------------------------------------------------------------------------
# Stubs for Task 4 functions (imported by test file)
# ---------------------------------------------------------------------------

def select_records_for_zone(records, zone):
    raise NotImplementedError("Implemented in Task 4")


def build_surface_zone_edge_payloads(voxels, building_id_grid, meshsize, zones):
    raise NotImplementedError("Implemented in Task 4")


def is_axis_aligned_segment(segment):
    raise NotImplementedError("Implemented in Task 4")


def normalize_segment_key(segment):
    raise NotImplementedError("Implemented in Task 4")
