"""Tests for voxcity.utils.logging module."""
import pytest
import logging
import os
from unittest.mock import patch

from voxcity.utils.logging import (
    get_logger,
    _resolve_level,
    _LEVEL_NAMES,
)


class TestResolveLevelNames:
    def test_level_names_dict(self):
        assert _LEVEL_NAMES["DEBUG"] == logging.DEBUG
        assert _LEVEL_NAMES["INFO"] == logging.INFO
        assert _LEVEL_NAMES["WARNING"] == logging.WARNING
        assert _LEVEL_NAMES["ERROR"] == logging.ERROR
        assert _LEVEL_NAMES["CRITICAL"] == logging.CRITICAL


class TestResolveLevel:
    def test_none_returns_info(self):
        assert _resolve_level(None) == logging.INFO

    def test_empty_string_returns_info(self):
        assert _resolve_level("") == logging.INFO

    def test_valid_level_debug(self):
        assert _resolve_level("DEBUG") == logging.DEBUG

    def test_valid_level_warning(self):
        assert _resolve_level("WARNING") == logging.WARNING

    def test_case_insensitive(self):
        assert _resolve_level("debug") == logging.DEBUG
        assert _resolve_level("Debug") == logging.DEBUG
        assert _resolve_level("DEBUG") == logging.DEBUG

    def test_with_whitespace(self):
        assert _resolve_level("  DEBUG  ") == logging.DEBUG

    def test_invalid_level_returns_info(self):
        assert _resolve_level("INVALID") == logging.INFO
        assert _resolve_level("NOTAREAL") == logging.INFO


class TestGetLogger:
    def test_returns_logger(self):
        logger = get_logger("test_module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self):
        logger = get_logger("my_module")
        assert "voxcity" in logger.name
        assert "my_module" in logger.name

    def test_without_name_returns_package_logger(self):
        logger = get_logger()
        assert "voxcity" in logger.name

    def test_logger_can_log(self):
        logger = get_logger("test")
        # Should not raise
        logger.info("Test message")
        logger.debug("Debug message")
        logger.warning("Warning message")

    def test_multiple_calls_same_logger(self):
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        # Should be the same logger instance
        assert logger1 is logger2

    def test_different_names_different_loggers(self):
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")
        assert logger1 is not logger2


class TestLoggerConfiguration:
    def test_root_logger_configured(self):
        get_logger("config_test")
        root = logging.getLogger("voxcity")
        # Should have at least one handler after get_logger is called
        assert len(root.handlers) >= 1

    def test_propagate_disabled(self):
        get_logger("propagate_test")
        root = logging.getLogger("voxcity")
        assert root.propagate is False
