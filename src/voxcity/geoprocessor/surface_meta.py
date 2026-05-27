"""Pure helpers for building surface face classification and metadata.

No Pydantic or app imports — keeps this module portable for subprocess use.
"""

from __future__ import annotations
from typing import Any, Optional, Sequence

import numpy as np

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
    """Create a unique identifier for a surface face."""
    c = [int(round(float(v) * 1000)) for v in centroid]
    n = [int(round(float(v) * 1000)) for v in normal]
    return f"b{int(building_id)}:c{c[0]}_{c[1]}_{c[2]}:n{n[0]}_{n[1]}_{n[2]}:i{int(face_index)}"


# ---------------------------------------------------------------------------
# Face classification
# ---------------------------------------------------------------------------

def classify_surface_kind(normal: Sequence[float]) -> str:
    """Classify surface kind based on normal vector."""
    nx, ny, nz = [float(v) for v in normal]
    if nz > 0.5:
        return "roof"
    if nz < -0.5:
        return "bottom"
    if abs(nx) > 0.2 or abs(ny) > 0.2:
        return "wall"
    return "other"


def wall_orientation(normal: Sequence[float]) -> Optional[str]:
    """Determine wall orientation (N/E/S/W) from normal vector."""
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


def classify_surface_faces(mesh: Any) -> list[dict]:
    """Classify each face of a mesh and return metadata dicts for each face.
    
    Returns a list of dicts with keys: face_key, building_id, surface_kind, orientation.
    """
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

        result.append({
            "face_key": face_key,
            "building_id": bid,
            "surface_kind": kind,
            "orientation": orient,
        })
    return result


def attach_surface_face_meta(mesh: Any, reference_mesh: Any = None) -> Any:
    """Attach surface face metadata to a mesh in-place.

    Idempotent — safe to call multiple times.
    If reference_mesh is provided and has matching face topology, copy its keys directly.
    
    Stores mesh.metadata["surface_face_meta"] as list of dicts and version 1.
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
    mesh.metadata["surface_face_meta"] = meta
    mesh.metadata["surface_face_meta_version"] = 1
    return mesh


def _can_copy_surface_meta(mesh: Any, reference_mesh: Any) -> bool:
    """Check if reference_mesh has compatible topology and valid surface meta.
    
    Robust against missing .faces attribute - returns False if either mesh lacks .faces
    rather than raising, allowing fallback classification to produce clearer errors.
    """
    if reference_mesh is None:
        return False
    
    # Check if both meshes have .faces attribute
    if not hasattr(mesh, "faces") or not hasattr(reference_mesh, "faces"):
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

    try:
        mesh_faces = np.asarray(mesh.faces, dtype=int)
        ref_faces = np.asarray(reference_mesh.faces, dtype=int)
    except (ValueError, TypeError):
        # If .faces can't be converted to array, can't copy
        return False
    
    if len(mesh_faces) != len(ref_faces):
        return False

    return True


# ---------------------------------------------------------------------------
# Selector mask
# ---------------------------------------------------------------------------

def surface_zone_mask(
    face_meta: Sequence[Any],
    selectors: Sequence[Any],
) -> np.ndarray:
    """Return a boolean mask of faces selected by the given selectors.
    
    Accepts both dict and object selectors, handles both snake_case and camelCase keys.
    Skips selectors with unknown/missing building_id gracefully.
    """
    n = len(face_meta)
    positive = np.zeros(n, dtype=bool)
    excluded = np.zeros(n, dtype=bool)

    if n == 0:
        return positive

    building_ids = np.array([int(_get(m, "building_id")) for m in face_meta])
    kinds = np.array([str(_get(m, "surface_kind")) for m in face_meta])
    orientations = np.array([_get(m, "orientation") for m in face_meta], dtype=object)
    face_keys = np.array([str(_get(m, "face_key")) for m in face_meta], dtype=object)
    selectable = np.isin(kinds, list(SELECTABLE_KINDS))

    for selector in selectors:
        # Support both building_id and buildingId
        bid = _get(selector, "building_id")
        if bid is None:
            bid = _get(selector, "buildingId")
        
        # Skip selectors with missing/unknown building_id
        if bid is None:
            continue
        
        bid = int(bid)
        mode = _get(selector, "mode")
        if mode is None:
            continue
        
        base = (building_ids == bid) & selectable
        
        if mode == "whole":
            positive |= base
        elif mode == "roof":
            positive |= base & (kinds == "roof")
        elif mode == "all_walls":
            positive |= base & (kinds == "wall")
        elif mode == "wall_orientation":
            orientation = _get(selector, "orientation")
            if orientation is not None:
                positive |= base & (kinds == "wall") & (orientations == orientation)
        elif mode == "faces":
            # Support both face_keys and faceKeys
            fkeys = _get(selector, "face_keys")
            if fkeys is None:
                fkeys = _get(selector, "faceKeys")
            if fkeys:
                positive |= base & np.isin(face_keys, fkeys)
        elif mode == "exclude_faces":
            # Support both face_keys and faceKeys
            fkeys = _get(selector, "face_keys")
            if fkeys is None:
                fkeys = _get(selector, "faceKeys")
            if fkeys:
                excluded |= base & np.isin(face_keys, fkeys)

    return positive & ~excluded


def resolve_target_face_mask(mesh: Any, target_selectors: Sequence) -> np.ndarray:
    """Resolve target_selectors to a boolean face mask over mesh.faces.

    Ensures surface_face_meta is attached first (classifying if needed), then
    delegates to surface_zone_mask. Returns an (n_faces,) bool array.
    """
    meta = getattr(mesh, "metadata", None)
    have_meta = isinstance(meta, dict) and meta.get("surface_face_meta")
    if not have_meta:
        attach_surface_face_meta(mesh)
        meta = mesh.metadata
    face_meta = meta["surface_face_meta"]
    return surface_zone_mask(face_meta, target_selectors).astype(bool)


# ---------------------------------------------------------------------------
# Face area computation
# ---------------------------------------------------------------------------

def compute_face_areas(mesh: Any) -> np.ndarray:
    """Return the area of each triangular face of mesh as a float32 array.

    Assumes triangular faces (mesh.faces shape (n_faces, 3)). Voxcity's
    create_voxel_mesh produces triangulated output, so this fits the
    existing pipeline. If a future mesh source produces quads, triangulate
    first.
    """
    faces = np.asarray(mesh.faces)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise NotImplementedError("Only triangulated meshes are supported")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    tris = vertices[faces]
    e1 = tris[:, 1, :] - tris[:, 0, :]
    e2 = tris[:, 2, :] - tris[:, 0, :]
    return (0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)).astype(np.float32)
