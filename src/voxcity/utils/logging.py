"""
Lightweight, centralized logging utilities for the voxcity package.

Usage:
    from voxcity.utils.logging import get_logger
    logger = get_logger(__name__)

Environment variables:
    VOXCITY_LOG_LEVEL: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
"""

from __future__ import annotations

import logging
import os
from typing import Optional


_LEVEL_NAMES = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}


def _resolve_level(env_value: Optional[str]) -> int:
    if not env_value:
        return logging.INFO
    return _LEVEL_NAMES.get(env_value.strip().upper(), logging.INFO)


def _configure_root_once() -> None:
    root = logging.getLogger("voxcity")
    if root.handlers:
        return
    level = _resolve_level(os.getenv("VOXCITY_LOG_LEVEL"))
    root.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a child logger under the package root logger.

    - Ensures a single configuration for the package
    - Respects VOXCITY_LOG_LEVEL if set
    """
    _configure_root_once()
    pkg_logger = logging.getLogger("voxcity")
    return pkg_logger.getChild(name) if name else pkg_logger


