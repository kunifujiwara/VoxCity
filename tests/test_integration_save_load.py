"""
Tests for voxcity.simulator.solar.integration save/load functions.
"""

import os
import pickle
import numpy as np
import pytest

from voxcity.simulator.solar.integration import (
    save_irradiance_mesh,
    load_irradiance_mesh,
)


class TestSaveLoadIrradianceMesh:
    def test_round_trip(self, tmp_path):
        data = {"vertices": np.array([[0, 0, 0]]), "irradiance": [1.0, 2.0]}
        path = str(tmp_path / "subdir" / "mesh.pkl")
        save_irradiance_mesh(data, path)
        assert os.path.exists(path)
        loaded = load_irradiance_mesh(path)
        assert loaded["irradiance"] == [1.0, 2.0]
        np.testing.assert_array_equal(loaded["vertices"], data["vertices"])

    def test_save_creates_directory(self, tmp_path):
        path = str(tmp_path / "deep" / "nested" / "dir" / "mesh.pkl")
        save_irradiance_mesh({"key": "value"}, path)
        assert os.path.exists(path)

    def test_complex_data(self, tmp_path):
        data = {
            "mesh": np.random.rand(100, 3),
            "faces": np.random.randint(0, 100, (50, 3)),
            "direct": np.random.rand(50),
            "diffuse": np.random.rand(50),
            "meta": {"source": "test", "steps": 24},
        }
        path = str(tmp_path / "complex.pkl")
        save_irradiance_mesh(data, path)
        loaded = load_irradiance_mesh(path)
        np.testing.assert_array_almost_equal(loaded["mesh"], data["mesh"])
        assert loaded["meta"]["steps"] == 24
