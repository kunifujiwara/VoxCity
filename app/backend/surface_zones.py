"""Pure helpers for building surface zone classification and statistics.

No FastAPI or global app_state imports — keeps this module testable in isolation.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .models import SurfaceFaceMeta, SurfaceSelector, ZoneStat
from .zoning import stats_from_values

SELECTABLE_KINDS = {"roof", "wall"}


# ---------------------------------------------------------------------------
# Face-key generation
# ---------------------------------------------------------------------------

def make_surface_face_key(
    building_id: int,
    centroid: Sequence[float],
    normal: Sequence[float],
    face_index: int,
) -> str:
    c = [int(round(float(v) * 1000)) for v in centroid]
    n = [int(round(float(v) * 1000)) for v in normal]
    return f"b{int(building_id)}:c{c[0]}_{c[1]}_{c[2]}:n{n[0]}_{n[1]}_{n[2]}:i{int(face_index)}"


# ---------------------------------------------------------------------------
# Face classification
# ---------------------------------------------------------------------------

def classify_surface_kind(normal: Sequence[float]) -> str:
    nx, ny, nz = [float(v) for v in normal]
    if nz > 0.5:
        return "roof"
    if nz < -0.5:
        return "bottom"
    if abs(nx) > 0.2 or abs(ny) > 0.2:
        return "wall"
    return "other"


def wall_orientation(normal: Sequence[float]) -> Optional[str]:
    nx, ny, nz = [float(v) for v in normal]
    if abs(nz) > 0.5:
        return None
    if abs(nx) >= abs(ny):
        return "E" if nx > 0 else "W"
    return "N" if ny > 0 else "S"


def _get(meta: Any, key: str, default: Any = None) -> Any:
    """Get a value from either a dict or an object with attributes."""
    if isinstance(meta, dict):
        return meta.get(key, default)
    return getattr(meta, key, default)


def classify_surface_faces(mesh: Any) -> list[SurfaceFaceMeta]:
    """Classify each face of a mesh and return SurfaceFaceMeta for each face."""
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    # Get building IDs per face
    meta = getattr(mesh, "metadata", {}) or {}
    if isinstance(meta, dict):
        bid_per_face = meta.get("building_id", None)
        if bid_per_face is None:
            bid_per_face = meta.get("building_face_ids", None)
        provided_normals = meta.get("provided_face_normals", None)
    else:
        bid_per_face = getattr(meta, "building_id", None)
        if bid_per_face is None:
            bid_per_face = getattr(meta, "building_face_ids", None)
        provided_normals = getattr(meta, "provided_face_normals", None)

    if bid_per_face is None:
        bid_per_face = np.zeros(len(faces), dtype=int)
    bid_per_face = np.asarray(bid_per_face, dtype=int)

    if isinstance(meta, dict):
        cls_per_face = meta.get("face_voxel_class", None)
    else:
        cls_per_face = getattr(meta, "face_voxel_class", None)
    if cls_per_face is not None:
        cls_per_face = np.asarray(cls_per_face, dtype=int)

    result = []
    for i, face in enumerate(faces):
        verts = vertices[face]
        centroid = verts.mean(axis=0)

        if provided_normals is not None:
            normal = np.asarray(provided_normals[i], dtype=float)
        else:
            # Compute face normal from vertices
            v0, v1, v2 = verts
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal = normal / norm_len
            else:
                normal = np.array([0.0, 0.0, 1.0])

        bid = int(bid_per_face[i])
        kind = classify_surface_kind(normal)
        orient = wall_orientation(normal) if kind == "wall" else None
        face_key = make_surface_face_key(bid, centroid, normal, i)
        is_window = bool(cls_per_face is not None and int(cls_per_face[i]) == -16)

        result.append(SurfaceFaceMeta(
            face_key=face_key,
            building_id=bid,
            surface_kind=kind,
            orientation=orient,
            is_window=is_window,
        ))
    return result


def attach_surface_face_meta(mesh: Any, reference_mesh: Any = None) -> Any:
    """Attach surface face metadata to a mesh in-place.

    Idempotent — safe to call multiple times.
    If reference_mesh is provided and has matching face topology, copy its keys directly.
    """
    if not hasattr(mesh, "metadata") or mesh.metadata is None:
        mesh.metadata = {}

    if _can_copy_surface_meta(mesh, reference_mesh):
        ref_meta = reference_mesh.metadata
        if isinstance(ref_meta, dict):
            meta_list = ref_meta.get("surface_face_meta", [])
        else:
            meta_list = getattr(ref_meta, "surface_face_meta", [])
        mesh.metadata["surface_face_meta"] = list(meta_list)
        mesh.metadata["surface_face_meta_version"] = 1
        return mesh

    meta = classify_surface_faces(mesh)
    mesh.metadata["surface_face_meta"] = [m.model_dump() for m in meta]
    mesh.metadata["surface_face_meta_version"] = 1
    return mesh


def _can_copy_surface_meta(mesh: Any, reference_mesh: Any) -> bool:
    """Check if reference_mesh has compatible topology and valid surface meta."""
    if reference_mesh is None:
        return False
    ref_meta = getattr(reference_mesh, "metadata", None)
    if ref_meta is None:
        return False
    if isinstance(ref_meta, dict):
        version = ref_meta.get("surface_face_meta_version")
        ref_face_meta = ref_meta.get("surface_face_meta")
    else:
        version = getattr(ref_meta, "surface_face_meta_version", None)
        ref_face_meta = getattr(ref_meta, "surface_face_meta", None)

    if version != 1 or not ref_face_meta:
        return False

    mesh_faces = np.asarray(mesh.faces, dtype=int)
    ref_faces = np.asarray(reference_mesh.faces, dtype=int)
    if len(mesh_faces) != len(ref_faces):
        return False

    return True


def _surface_meta_from_cached_mesh(mesh: Any) -> list:
    """Extract surface_face_meta from a cached mesh if version matches face count."""
    if mesh is None:
        return []
    meta = getattr(mesh, "metadata", None)
    if meta is None:
        return []
    if isinstance(meta, dict):
        version = meta.get("surface_face_meta_version")
        face_meta = meta.get("surface_face_meta")
    else:
        version = getattr(meta, "surface_face_meta_version", None)
        face_meta = getattr(meta, "surface_face_meta", None)

    if version != 1 or not face_meta:
        return []

    faces = getattr(mesh, "faces", [])
    if len(face_meta) != len(faces):
        return []

    return face_meta


# ---------------------------------------------------------------------------
# Selector mask
# ---------------------------------------------------------------------------

def surface_zone_mask(
    face_meta: Sequence[Any],
    selectors: Sequence[SurfaceSelector],
) -> np.ndarray:
    """Return a boolean mask of faces selected by the given selectors."""
    n = len(face_meta)
    positive = np.zeros(n, dtype=bool)
    excluded = np.zeros(n, dtype=bool)

    if n == 0:
        return positive

    building_ids = np.array([int(_get(m, "building_id")) for m in face_meta])
    kinds = np.array([str(_get(m, "surface_kind")) for m in face_meta])
    orientations = np.array([_get(m, "orientation") for m in face_meta], dtype=object)
    face_keys = np.array([str(_get(m, "face_key")) for m in face_meta], dtype=object)
    windows = np.array([bool(_get(m, "is_window", False)) for m in face_meta])
    selectable = np.isin(kinds, list(SELECTABLE_KINDS))

    for selector in selectors:
        base = (building_ids == selector.building_id) & selectable
        if selector.mode == "whole":
            positive |= base
        elif selector.mode == "roof":
            positive |= base & (kinds == "roof")
        elif selector.mode == "all_walls":
            positive |= base & (kinds == "wall")
        elif selector.mode == "window":
            positive |= base & windows
        elif selector.mode == "wall_orientation":
            positive |= base & (kinds == "wall") & (orientations == selector.orientation)
        elif selector.mode == "faces":
            positive |= base & np.isin(face_keys, selector.face_keys or [])
        elif selector.mode == "exclude_faces":
            excluded |= base & np.isin(face_keys, selector.face_keys or [])

    return positive & ~excluded


# ---------------------------------------------------------------------------
# Surface zone stats
# ---------------------------------------------------------------------------

def stats_for_surface_zone(
    zone_id: str,
    face_meta: Sequence[Any],
    selectors: Sequence[SurfaceSelector],
    values: np.ndarray,
    areas: np.ndarray,
) -> ZoneStat:
    """Compute area-weighted statistics for a building surface zone."""
    mask = surface_zone_mask(face_meta, selectors)
    selected_count = int(mask.sum())

    if selected_count == 0:
        return stats_from_values(zone_id, 0, np.array([], dtype=float))

    selected_values = values[mask]
    selected_areas = areas[mask]

    # Filter to finite values
    finite_mask = np.isfinite(selected_values)
    finite_values = selected_values[finite_mask]
    finite_areas = selected_areas[finite_mask]

    if len(finite_values) == 0:
        return ZoneStat(
            zone_id=zone_id,
            cell_count=selected_count,
            valid_count=0,
            mean=None,
            min=None,
            max=None,
            std=None,
        )

    total_area = float(finite_areas.sum())
    if total_area > 0:
        weighted_mean = float((finite_values * finite_areas).sum() / total_area)
    else:
        weighted_mean = float(finite_values.mean())

    return ZoneStat(
        zone_id=zone_id,
        cell_count=selected_count,
        valid_count=int(finite_mask.sum()),
        mean=weighted_mean,
        min=float(finite_values.min()),
        max=float(finite_values.max()),
        std=None,  # std not defined for area-weighted; set to None per spec
    )
