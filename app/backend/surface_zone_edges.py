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
# Task 4: response dataclass, selector helpers, segment builders
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SurfaceZoneEdgePayload:
    id: str
    segments: list[tuple[float, float, float, float, float, float]]


def _record_matches_positive(record: VoxelSurfaceRecord, selector) -> bool:
    if record.building_id != selector.building_id:
        return False
    if record.surface_kind not in {"roof", "wall"}:
        return False
    if selector.mode == "whole":
        return True
    if selector.mode == "roof":
        return record.surface_kind == "roof"
    if selector.mode == "all_walls":
        return record.surface_kind == "wall"
    if selector.mode == "wall_orientation":
        return record.surface_kind == "wall" and record.orientation == selector.orientation
    if selector.mode == "faces":
        return any(key in set(selector.face_keys or []) for key in record.face_keys)
    return False


def _record_is_excluded(record: VoxelSurfaceRecord, selector) -> bool:
    return (
        selector.mode == "exclude_faces"
        and record.building_id == selector.building_id
        and any(key in set(selector.face_keys or []) for key in record.face_keys)
    )


def select_records_for_zone(records: list[VoxelSurfaceRecord], zone) -> list[VoxelSurfaceRecord]:
    selectors = zone.selectors or []
    selected: list[VoxelSurfaceRecord] = []
    for record in records:
        if any(_record_matches_positive(record, selector) for selector in selectors):
            selected.append(record)
    return [record for record in selected if not any(_record_is_excluded(record, selector) for selector in selectors)]


def is_axis_aligned_segment(segment) -> bool:
    x1, y1, z1, x2, y2, z2 = [float(value) for value in segment]
    deltas = [abs(x2 - x1), abs(y2 - y1), abs(z2 - z1)]
    changed = sum(delta > AXIS_EPSILON_M for delta in deltas)
    return changed == 1


def normalize_segment_key(segment):
    point_a = tuple(round(float(value), 6) for value in segment[:3])
    point_b = tuple(round(float(value), 6) for value in segment[3:])
    return (point_a, point_b) if point_a <= point_b else (point_b, point_a)


def _dedupe_axis_segments(segments):
    """Keep only boundary (outer envelope) segments.

    Each segment is counted across all input rectangles.  Segments shared by
    two adjacent rectangles appear twice and cancel (even count → interior
    edge, dropped).  Unshared segments appear once (odd count → boundary
    edge, kept).  This is the standard parity / XOR boundary algorithm.
    """
    counts: dict = {}
    coords: dict = {}
    for segment in segments:
        if not is_axis_aligned_segment(segment):
            continue
        key = normalize_segment_key(segment)
        counts[key] = counts.get(key, 0) + 1
        if key not in coords:
            x1, y1, z1, x2, y2, z2 = (float(v) for v in segment)
            coords[key] = (x1, y1, z1, x2, y2, z2)
    return [coords[key] for key, count in counts.items() if count % 2 == 1]


def _group_key(record: VoxelSurfaceRecord):
    if record.plane in {"+x", "-x"}:
        return (record.building_id, record.plane, record.corners[0][0])
    if record.plane in {"+y", "-y"}:
        return (record.building_id, record.plane, record.corners[0][1])
    return (record.building_id, record.plane, record.corners[0][2])


def _mask_coords(record: VoxelSurfaceRecord):
    u, v, k = record.voxel_uvw
    if record.plane in {"+x", "-x"}:
        return (u, k)
    if record.plane in {"+y", "-y"}:
        return (v, k)
    return (u, v)


def _records_by_cell(records):
    by_cell: dict[tuple[int, int], VoxelSurfaceRecord] = {}
    for record in records:
        by_cell[_mask_coords(record)] = record
    return by_cell


def _rectangle_bounds(rect_records):
    points = [point for record in rect_records for point in record.corners]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    zs = [point[2] for point in points]
    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def _rectangle_segments(plane: str, rect_records):
    min_x, max_x, min_y, max_y, min_z, max_z = _rectangle_bounds(rect_records)
    if plane in {"+x", "-x"}:
        x = rect_records[0].corners[0][0]
        return [
            (x, min_y, min_z, x, max_y, min_z),
            (x, max_y, min_z, x, max_y, max_z),
            (x, max_y, max_z, x, min_y, max_z),
            (x, min_y, max_z, x, min_y, min_z),
        ]
    if plane in {"+y", "-y"}:
        y = rect_records[0].corners[0][1]
        return [
            (min_x, y, min_z, max_x, y, min_z),
            (max_x, y, min_z, max_x, y, max_z),
            (max_x, y, max_z, min_x, y, max_z),
            (min_x, y, max_z, min_x, y, min_z),
        ]
    z = rect_records[0].corners[0][2]
    return [
        (min_x, min_y, z, max_x, min_y, z),
        (max_x, min_y, z, max_x, max_y, z),
        (max_x, max_y, z, min_x, max_y, z),
        (min_x, max_y, z, min_x, min_y, z),
    ]


def _greedy_rectangles_for_group(records) -> list[list[VoxelSurfaceRecord]]:
    by_cell = _records_by_cell(records)
    selected = set(by_cell.keys())
    visited: set[tuple[int, int]] = set()
    rectangles: list[list[VoxelSurfaceRecord]] = []

    for start in sorted(selected):
        if start in visited:
            continue
        row, col = start
        width = 1
        while (row, col + width) in selected and (row, col + width) not in visited:
            width += 1

        height = 1
        while True:
            next_row = row + height
            next_cells = [(next_row, col + offset) for offset in range(width)]
            if all(cell in selected and cell not in visited for cell in next_cells):
                height += 1
                continue
            break

        cells = [(row + r, col + c) for r in range(height) for c in range(width)]
        visited.update(cells)
        rectangles.append([by_cell[cell] for cell in cells])

    return rectangles


def _segments_from_selected_records(records: list[VoxelSurfaceRecord]):
    groups: dict[tuple, list[VoxelSurfaceRecord]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)

    segments = []
    for _key, group_records in sorted(groups.items(), key=lambda item: item[0]):
        plane = group_records[0].plane
        for rectangle in _greedy_rectangles_for_group(group_records):
            segments.extend(_rectangle_segments(plane, rectangle))
    return segments


def build_surface_zone_edge_payloads(
    voxels: np.ndarray,
    building_id_grid: np.ndarray,
    meshsize: float,
    zones,
) -> list[SurfaceZoneEdgePayload]:
    records = extract_voxel_surface_records(voxels, building_id_grid, meshsize)
    payloads: list[SurfaceZoneEdgePayload] = []
    for zone in zones:
        if zone.type != "building_surface":
            continue
        selected = select_records_for_zone(records, zone)
        if not selected:
            continue
        segments = _segments_from_selected_records(selected)
        payloads.append(SurfaceZoneEdgePayload(id=zone.id, segments=_dedupe_axis_segments(segments)))
    return payloads
