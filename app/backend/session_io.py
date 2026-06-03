"""Per-session save / load of editing state.

Pure-Python helpers (no FastAPI imports) so the module is easy to unit-test.
"""

from __future__ import annotations

import datetime
import io
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


SESSION_FORMAT_VERSION = 1
DEFAULT_MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB

_VOXCITY_NAME = "voxcity.h5"
_META_NAME = "meta.json"
_FRONTEND_STATE_NAME = "frontend_state.json"


class SessionLoadError(Exception):
    """Raised by parse_session_zip / apply_session_to_state when input is invalid."""


@dataclass
class ParsedSession:
    """Output of parse_session_zip — fully validated, in-memory representation."""

    meta: Dict[str, Any]
    voxcity_h5_path: str
    frontend_state: Optional[str] = None
    sim_results: Optional[Dict[str, Any]] = None  # filled in by a later task


def _utc_iso_now() -> str:
    return datetime.datetime.now(tz=datetime.timezone.utc).isoformat(timespec="seconds")


def _app_version() -> str:
    try:
        import importlib.metadata as metadata
        return metadata.version("voxcity")
    except Exception:
        return "unknown"


def _build_meta(state, include_sim_results: bool) -> Dict[str, Any]:
    voxcity = state.voxcity
    meshsize = float(voxcity.voxels.meta.meshsize) if voxcity is not None else 0.0
    return {
        "format_version": SESSION_FORMAT_VERSION,
        "app_version": _app_version(),
        "saved_at_utc": _utc_iso_now(),
        "rectangle_vertices": state.rectangle_vertices,
        "land_cover_source": state.land_cover_source,
        "meshsize": meshsize,
        "has_sim_results": bool(
            include_sim_results and getattr(state, "sim_results_by_type", None)
        ),
    }


def save_session_to_zip(
    state,
    frontend_state: Optional[str] = None,
    include_sim_results: bool = False,
) -> io.BytesIO:
    """Serialize *state* to a session zip and return a BytesIO ready to stream."""
    if state.voxcity is None:
        raise ValueError("Cannot save session: no scene has been generated.")

    from voxcity.io import save_voxcity

    tmp = Path(tempfile.mkdtemp(prefix="voxcity-session-save-"))
    try:
        save_voxcity(str(tmp / _VOXCITY_NAME), state.voxcity)
        (tmp / _META_NAME).write_text(
            json.dumps(_build_meta(state, include_sim_results)),
            encoding="utf-8",
        )
        if frontend_state is not None:
            try:
                json.loads(frontend_state)
            except json.JSONDecodeError as exc:
                raise ValueError(f"frontend_state must be valid JSON: {exc}") from exc
            (tmp / _FRONTEND_STATE_NAME).write_text(frontend_state, encoding="utf-8")

        if include_sim_results:
            _serialize_sim_results(state, tmp)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in tmp.rglob("*"):
                if path.is_file():
                    arcname = str(path.relative_to(tmp)).replace("\\", "/")
                    zf.write(path, arcname=arcname)
        buf.seek(0)
        return buf
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def parse_session_zip(stream, max_bytes: int = DEFAULT_MAX_UPLOAD_BYTES) -> ParsedSession:
    """Read and validate a session zip without mutating application state."""
    data = _drain_with_limit(stream, max_bytes)

    tmp = Path(tempfile.mkdtemp(prefix="voxcity-session-load-"))
    try:
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                _safe_extract_all(zf, tmp)
        except zipfile.BadZipFile as exc:
            raise SessionLoadError(f"File is not a valid zip archive: {exc}") from exc

        meta_path = tmp / _META_NAME
        voxcity_path = tmp / _VOXCITY_NAME

        if not voxcity_path.is_file():
            raise SessionLoadError("Saved session is missing required file: voxcity.h5")
        if not meta_path.is_file():
            raise SessionLoadError("Saved session is missing required file: meta.json")

        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SessionLoadError(f"meta.json is not valid JSON: {exc}") from exc

        format_version = meta.get("format_version")
        if not isinstance(format_version, int):
            raise SessionLoadError("meta.json is missing an integer 'format_version'.")
        if format_version > SESSION_FORMAT_VERSION:
            raise SessionLoadError(
                "This file was saved by a newer version of the app "
                f"(format_version={format_version}, this version supports "
                f"{SESSION_FORMAT_VERSION})."
            )

        frontend_state = None
        fs_path = tmp / _FRONTEND_STATE_NAME
        if fs_path.is_file():
            raw = fs_path.read_text(encoding="utf-8")
            try:
                json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SessionLoadError(
                    f"frontend_state.json is not valid JSON: {exc}"
                ) from exc
            frontend_state = raw

        sim_results = _parse_sim_results(tmp / _SIM_RESULTS_DIR)

        return ParsedSession(
            meta=meta,
            voxcity_h5_path=str(voxcity_path),
            frontend_state=frontend_state,
            sim_results=sim_results,
        )
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def parsed_session_temp_root(parsed: ParsedSession) -> str:
    """Return the temp dir that owns *parsed*'s extracted files."""
    return os.path.dirname(parsed.voxcity_h5_path)


def _drain_with_limit(stream, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    read = getattr(stream, "read", None)
    if read is None:
        raise SessionLoadError("Upload stream does not support .read()")
    while True:
        chunk = read(64 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise SessionLoadError(
                f"Upload exceeds maximum size ({max_bytes // (1024 * 1024)} MB)."
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_extract_all(zf: zipfile.ZipFile, dst: Path) -> None:
    """Like ZipFile.extractall, but refuses path-traversal and absolute paths."""
    dst_resolved = dst.resolve()
    for info in zf.infolist():
        name = info.filename
        if name.startswith("/") or name.startswith("\\") or ":" in name.split("/")[0]:
            raise SessionLoadError(f"Refusing unsafe absolute path in zip: {name!r}")
        target = (dst / name).resolve()
        try:
            target.relative_to(dst_resolved)
        except ValueError as exc:
            raise SessionLoadError(
                f"Refusing unsafe path traversal in zip: {name!r}"
            ) from exc
    zf.extractall(dst)


_SIM_RESULTS_DIR = "sim_results"
_SIM_INDEX_NAME = "sim_results/index.json"


@dataclass
class _SerializedMesh:
    vertices: Any
    faces: Any
    metadata: Dict[str, Any]


def _validate_sim_type_path_component(sim_type: Any) -> str:
    if not isinstance(sim_type, str) or not sim_type:
        raise ValueError("sim_results entry missing sim_type")
    if "/" in sim_type or "\\" in sim_type or sim_type in {".", ".."}:
        raise ValueError(f"sim_results entry has unsafe sim_type: {sim_type!r}")
    return sim_type


def _serialize_sim_results(state, dst_root: Path) -> None:
    import numpy as np

    def save_array(path: Path, value: Any, label: str) -> None:
        try:
            np.save(path, np.asarray(value))
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Failed to serialize {label}: {exc}") from exc

    def save_metadata(path: Path, metadata: Dict[str, Any], label: str) -> None:
        arrays = {}
        for key, value in metadata.items():
            try:
                arrays[str(key)] = np.asarray(value)
            except (TypeError, ValueError):
                continue
        try:
            np.savez(path, **arrays)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(f"Failed to serialize {label}: {exc}") from exc

    cache = getattr(state, "sim_results_by_type", None) or {}
    if not cache:
        return

    sim_dir = dst_root / _SIM_RESULTS_DIR
    sim_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for raw_sim_type, entry in cache.items():
        sim_type = _validate_sim_type_path_component(raw_sim_type)
        entry_dir = sim_dir / sim_type
        entry_dir.mkdir(parents=True, exist_ok=True)

        if entry.grid is not None:
            save_array(entry_dir / "grid.npy", entry.grid, f"{sim_type} grid")
        if entry.voxcity_grid is not None:
            save_array(entry_dir / "voxcity_grid.npy", entry.voxcity_grid, f"{sim_type} voxcity_grid")
        if entry.building_id_grid is not None:
            save_array(entry_dir / "building_id_grid.npy", entry.building_id_grid, f"{sim_type} building_id_grid")
        if entry.mesh is not None:
            save_array(entry_dir / "mesh_vertices.npy", np.asarray(entry.mesh.vertices, dtype=np.float32), f"{sim_type} mesh vertices")
            save_array(entry_dir / "mesh_faces.npy", np.asarray(entry.mesh.faces, dtype=np.int32), f"{sim_type} mesh faces")
            metadata = getattr(entry.mesh, "metadata", None) or {}
            save_metadata(entry_dir / "mesh_metadata.npz", metadata, f"{sim_type} mesh metadata")

        entries.append({
            "sim_type": sim_type,
            "target": entry.target,
            "view_point_height": float(entry.view_point_height),
            "colorbar_title": entry.colorbar_title,
            "has_grid": entry.grid is not None,
            "has_mesh": entry.mesh is not None,
            "has_voxcity_grid": entry.voxcity_grid is not None,
            "has_building_id_grid": entry.building_id_grid is not None,
        })

    index = {
        "schema_version": 1,
        "last_sim_type": getattr(state, "last_sim_type", None),
        "entries": entries,
    }
    (sim_dir / "index.json").write_text(json.dumps(index), encoding="utf-8")


def _parse_sim_results(sim_dir: Path) -> Optional[Dict[str, Any]]:
    import numpy as np

    index_path = sim_dir / "index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SessionLoadError(f"sim_results/index.json is not valid JSON: {exc}") from exc
    if not isinstance(index, dict):
        raise SessionLoadError("sim_results/index.json must contain a JSON object.")
    raw_entries = index.get("entries", [])
    if not isinstance(raw_entries, list):
        raise SessionLoadError("sim_results/index.json 'entries' must be a list.")

    def load_array(path: Path, label: str):
        try:
            return np.load(path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise SessionLoadError(f"Failed to load {label}: {exc}") from exc

    entries_out = []
    for raw in raw_entries:
        if not isinstance(raw, dict):
            raise SessionLoadError("sim_results entry must be a JSON object.")
        try:
            sim_type = _validate_sim_type_path_component(raw.get("sim_type"))
        except ValueError as exc:
            raise SessionLoadError(str(exc)) from exc
        entry_dir = sim_dir / sim_type

        grid = None
        if raw.get("has_grid") and (entry_dir / "grid.npy").is_file():
            grid = load_array(entry_dir / "grid.npy", f"sim_results/{sim_type}/grid.npy")

        voxcity_grid = None
        if raw.get("has_voxcity_grid") and (entry_dir / "voxcity_grid.npy").is_file():
            voxcity_grid = load_array(entry_dir / "voxcity_grid.npy", f"sim_results/{sim_type}/voxcity_grid.npy")

        building_id_grid = None
        if raw.get("has_building_id_grid") and (entry_dir / "building_id_grid.npy").is_file():
            building_id_grid = load_array(entry_dir / "building_id_grid.npy", f"sim_results/{sim_type}/building_id_grid.npy")

        mesh = None
        if raw.get("has_mesh"):
            vertices_path = entry_dir / "mesh_vertices.npy"
            faces_path = entry_dir / "mesh_faces.npy"
            metadata_path = entry_dir / "mesh_metadata.npz"
            if not vertices_path.is_file() or not faces_path.is_file():
                raise SessionLoadError(f"sim_results/{sim_type}: has_mesh=true but vertices/faces are missing")
            vertices = load_array(vertices_path, f"sim_results/{sim_type}/mesh_vertices.npy")
            faces = load_array(faces_path, f"sim_results/{sim_type}/mesh_faces.npy")
            metadata: Dict[str, Any] = {}
            if metadata_path.is_file():
                try:
                    npz = np.load(metadata_path, allow_pickle=False)
                except (OSError, ValueError) as exc:
                    raise SessionLoadError(f"Failed to load sim_results/{sim_type}/mesh_metadata.npz: {exc}") from exc
                try:
                    metadata = {key: npz[key] for key in npz.files}
                finally:
                    npz.close()
            mesh = _SerializedMesh(vertices=vertices, faces=faces, metadata=metadata)

        entries_out.append({
            "sim_type": sim_type,
            "target": raw.get("target", "ground"),
            "view_point_height": float(raw.get("view_point_height", 1.5)),
            "colorbar_title": raw.get("colorbar_title"),
            "grid": grid,
            "mesh": mesh,
            "voxcity_grid": voxcity_grid,
            "building_id_grid": building_id_grid,
        })

    return {
        "schema_version": index.get("schema_version", 1),
        "last_sim_type": index.get("last_sim_type"),
        "entries": entries_out,
    }


def apply_session_to_state(parsed: ParsedSession, state) -> Dict[str, Any]:
    """Replace *state* atomically and return a summary for the frontend."""
    try:
        from voxcity.simulator_gpu.visibility.integration import clear_visibility_cache
        clear_visibility_cache()
    except Exception:
        pass
    try:
        from voxcity.simulator_gpu.solar.integration.caching import (
            clear_all_caches as clear_all_solar_caches,
        )
        clear_all_solar_caches()
    except Exception:
        pass

    from voxcity.io import load_voxcity

    new_voxcity = load_voxcity(parsed.voxcity_h5_path)
    state.voxcity = new_voxcity
    state.rectangle_vertices = parsed.meta.get("rectangle_vertices")
    state.land_cover_source = parsed.meta.get("land_cover_source", "OpenStreetMap")
    state.reset_for_session_load()
    try:
        state.refresh_raw_cache()
    except Exception:
        pass
    if parsed.sim_results is not None:
        _restore_sim_results(parsed.sim_results, state)

    return {
        "has_voxcity": state.voxcity is not None,
        "rectangle_vertices": state.rectangle_vertices,
        "land_cover_source": state.land_cover_source,
        "frontend_state": parsed.frontend_state,
    }


def _restore_sim_results(blob: Dict[str, Any], state) -> None:
    """Replay saved sim entries through AppState.store_sim_result."""
    last_sim_type = blob.get("last_sim_type")
    entries = blob.get("entries") or []
    ordered = sorted(
        entries,
        key=lambda entry: 1 if entry.get("sim_type") == last_sim_type else 0,
    )
    for entry in ordered:
        state.store_sim_result(
            sim_type=entry["sim_type"],
            target=entry.get("target", "ground"),
            grid=entry.get("grid"),
            mesh=entry.get("mesh"),
            voxcity_grid=entry.get("voxcity_grid"),
            view_point_height=float(entry.get("view_point_height", 1.5)),
            colorbar_title=entry.get("colorbar_title"),
            building_id_grid=entry.get("building_id_grid"),
        )
