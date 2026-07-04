"""Centralized, dependency-free runtime configuration for the VoxCity backend.

Importing this module stays cheap: it uses only the standard library (plus
python-dotenv when installed). Every per-environment difference is supplied via
environment variables (or a local untracked .env for native dev); real
environment variables always win.
"""

from __future__ import annotations

import os
import tempfile
from typing import Optional

_APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # app/
_REPO_ROOT = os.path.dirname(_APP_DIR)

# Optionally load a local .env for native dev. override=False (the default) so
# real environment variables — e.g. those baked into the Docker image — win.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
except ImportError:
    pass


def _env_or(var: str, default: str) -> str:
    value = os.environ.get(var)
    return value if value else default


# --- Base directories ------------------------------------------------------
DATA_DIR = _env_or("VOXCITY_DATA_DIR", os.path.join(_APP_DIR, "data"))
OUTPUT_DIR = _env_or(
    "VOXCITY_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "voxcity_output")
)

# --- Derived data paths ----------------------------------------------------
DEFAULT_TOKYO_EPW = _env_or(
    "DEFAULT_TOKYO_EPW",
    os.path.join(
        DATA_DIR, "temp", "epw_tokyo", "JPN_TK_Tokyo-Chiyoda.476620_TMYx.epw"
    ),
)
NDSM_COG_PATH = _env_or("NDSM_COG_PATH", os.path.join(DATA_DIR, "temp", "ndsm_cog.tif"))
CITYGML_CACHE_DIR = os.path.join(DATA_DIR, "temp", "citygml_cache")


def _resolve_citygml_path() -> Optional[str]:
    explicit = os.environ.get("CITYGML_PATH")
    if explicit:
        return explicit
    candidate = os.path.join(DATA_DIR, "plateau")
    return candidate if os.path.isdir(candidate) else None


CITYGML_PATH = _resolve_citygml_path()


def _resolve_frontend_dist() -> Optional[str]:
    explicit = os.environ.get("VOXCITY_FRONTEND_DIST")
    if explicit:
        return explicit
    candidate = os.path.join(_APP_DIR, "frontend", "dist")
    return candidate if os.path.isdir(candidate) else None


FRONTEND_DIST = _resolve_frontend_dist()
