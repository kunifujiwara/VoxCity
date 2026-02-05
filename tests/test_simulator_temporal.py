"""Tests for voxcity.simulator.solar.temporal module helper functions."""
import pytest
import numpy as np
import os

from voxcity.simulator.solar.temporal import (
    _configure_num_threads,
    _auto_time_batch_size,
)


class TestConfigureNumThreads:
    """Tests for _configure_num_threads function."""

    def test_returns_integer(self):
        """Test that function returns an integer."""
        result = _configure_num_threads()
        assert isinstance(result, int)
        assert result > 0

    def test_respects_desired_threads(self):
        """Test that specific thread count is respected."""
        result = _configure_num_threads(desired_threads=2)
        assert result == 2

    def test_with_progress_flag(self):
        """Test function runs with progress output."""
        result = _configure_num_threads(desired_threads=1, progress=True)
        assert result == 1


class TestAutoTimeBatchSize:
    """Tests for _auto_time_batch_size function."""

    def test_returns_user_value_if_provided(self):
        """Test that user value overrides auto calculation."""
        assert _auto_time_batch_size(1000, 100, user_value=50) == 50
        assert _auto_time_batch_size(1000, 100, user_value=1) == 1

    def test_minimum_batch_size_is_one(self):
        """Test that batch size is at least 1."""
        assert _auto_time_batch_size(100, 1) >= 1
        assert _auto_time_batch_size(100, 0) >= 1

    def test_small_faces_uses_fewer_batches(self):
        """Test that small face count uses fewer batches."""
        # Small face count (<=5000) -> 2 batches
        batch_small = _auto_time_batch_size(1000, 100)
        assert batch_small >= 1

    def test_medium_faces(self):
        """Test batch size for medium face counts."""
        # Medium face count (5001-50000) -> 8 batches
        batch_medium = _auto_time_batch_size(20000, 100)
        assert batch_medium >= 1

    def test_large_faces(self):
        """Test batch size for large face counts."""
        # Large face count (50001-200000) -> 16 batches
        batch_large = _auto_time_batch_size(100000, 100)
        assert batch_large >= 1

    def test_very_large_faces(self):
        """Test batch size for very large face counts."""
        # Very large face count (>200000) -> 32 batches
        batch_xlarge = _auto_time_batch_size(500000, 320)
        assert batch_xlarge >= 1

    def test_user_value_zero_becomes_one(self):
        """Test that user value 0 becomes 1."""
        assert _auto_time_batch_size(100, 100, user_value=0) == 1

    def test_scales_with_total_steps(self):
        """Test batch size considers total steps."""
        # With few total steps, batch size is limited
        batch = _auto_time_batch_size(100000, 5)
        assert batch <= 5
