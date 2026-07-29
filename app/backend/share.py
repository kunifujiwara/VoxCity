"""Server-side persisted share snapshots (share-by-URL).

A share is a session zip written under SHARE_DIR/<token>/session.zip with a
meta.json sidecar. Tokens are unguessable URL-safe strings; possession of the
URL is the access model (no auth, no enumeration endpoint).

Adapted from optree_voxcity's share.py for VoxCity's single global app_state:
the optimization-job branch is dropped (VoxCity has no optimizer).
"""

from __future__ import annotations

import datetime
import json
import re
import secrets
import shutil
from pathlib import Path
from typing import Optional

from . import config
from .session_io import _app_version, save_session_to_zip

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")

_share_base_dir = Path(config.SHARE_DIR)


def set_share_base_dir(path) -> None:
    """Test hook to relocate the share directory."""
    global _share_base_dir
    _share_base_dir = Path(path)


def share_base_dir() -> Path:
    return _share_base_dir


def is_valid_token(token: str) -> bool:
    return bool(_TOKEN_RE.fullmatch(token))


def share_zip_path(token: str) -> Optional[Path]:
    """Return the stored session.zip for *token*, or None if invalid/unknown.

    Token validation happens before any filesystem access, so traversal-shaped
    input never touches the disk.
    """
    if not is_valid_token(token):
        return None
    path = _share_base_dir / token / "session.zip"
    return path if path.is_file() else None


def create_share(state, frontend_state: Optional[str] = None) -> str:
    """Persist a share snapshot of *state* and return its token.

    Sim results are embedded iff cached. Creation is atomic: files are staged in
    a sibling ".tmp-<token>" dir and renamed into place, so a concurrent load
    can never observe a half-written share.

    Raises ValueError (from save_session_to_zip) when the state has no scene.
    """
    include_sim_results = bool(getattr(state, "sim_results_by_type", None))

    buf = save_session_to_zip(
        state,
        include_sim_results=include_sim_results,
        frontend_state=frontend_state,
    )
    data = buf.getvalue()

    token = secrets.token_urlsafe(16)
    base = _share_base_dir
    base.mkdir(parents=True, exist_ok=True)
    staging = base / f".tmp-{token}"
    staging.mkdir()
    try:
        (staging / "session.zip").write_bytes(data)
        meta = {
            "created_at_utc": datetime.datetime.now(
                tz=datetime.timezone.utc
            ).isoformat(timespec="seconds"),
            "app_version": _app_version(),
            "has_sim_results": include_sim_results,
            "size_bytes": len(data),
        }
        (staging / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
        staging.rename(base / token)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return token
