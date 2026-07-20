"""Tests for the strict v3 HDF5 format: save stamps, strict load, migrate_h5, CLI."""

import json
import math

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from voxcity.io import save_results_h5, load_results_h5, FORMAT_V3
from voxcity.utils.orientation import AXES_ATTR, check_axes
from voxcity.models import (
    GridMetadata,
    VoxelGrid,
    BuildingGrid,
    LandCoverGrid,
    DemGrid,
    CanopyGrid,
    VoxCity,
)

RECT = [(0.0, 0.0), (0.0, 0.01), (0.01, 0.01), (0.01, 0.0)]  # axis-aligned, ~1.1 km


def make_city(shape=(4, 5, 6), meshsize=2.0, extras=None):
    ny, nx, nz = shape
    meta = GridMetadata(crs="EPSG:4326", bounds=(0.0, 0.0, 0.01, 0.01), meshsize=meshsize)
    min_heights = np.empty((ny, nx), dtype=object)
    for idx in np.ndindex((ny, nx)):
        min_heights[idx] = []
    return VoxCity(
        voxels=VoxelGrid(classes=np.zeros(shape, dtype=np.int8), meta=meta),
        buildings=BuildingGrid(
            heights=np.zeros((ny, nx)),
            min_heights=min_heights,
            ids=np.zeros((ny, nx)),
            meta=meta,
        ),
        land_cover=LandCoverGrid(classes=np.ones((ny, nx), dtype=int), meta=meta),
        dem=DemGrid(elevation=np.zeros((ny, nx)), meta=meta),
        tree_canopy=CanopyGrid(top=np.zeros((ny, nx)), meta=meta),
        extras=dict(extras) if extras is not None else {"rectangle_vertices": RECT},
    )


def rotated_rect(angle_deg, size_deg=0.01):
    a = math.radians(angle_deg)
    d1 = (size_deg * math.sin(a), size_deg * math.cos(a))
    d2 = (size_deg * math.cos(a), -size_deg * math.sin(a))
    return [
        (0.0, 0.0),
        (d1[0], d1[1]),
        (d1[0] + d2[0], d1[1] + d2[1]),
        (d2[0], d2[1]),
    ]


class TestV3Save:
    def test_root_attrs_and_geometry(self, tmp_path):
        p = str(tmp_path / "city.h5")
        save_results_h5(p, make_city())
        with h5py.File(p, "r") as f:
            assert f.attrs["__format__"] == FORMAT_V3
            assert f.attrs["axes"] == AXES_ATTR
            assert float(f.attrs["rotation_angle"]) == 0.0
            np.testing.assert_allclose(
                f["rectangle_vertices"][:], np.asarray(RECT, dtype=np.float64)
            )

    def test_axes_stamped_on_group_and_dataset(self, tmp_path):
        p = str(tmp_path / "city.h5")
        save_results_h5(p, make_city())
        with h5py.File(p, "r") as f:
            assert f["voxcity"].attrs["axes"] == AXES_ATTR
            assert f["voxcity"]["voxel_grid"].attrs["axes"] == AXES_ATTR
            check_axes(f)  # root passes the contract check

    def test_rotated_vertices_yield_rotation_angle(self, tmp_path):
        p = str(tmp_path / "rot.h5")
        rect = rotated_rect(25.0)
        save_results_h5(p, make_city(extras={"rectangle_vertices": rect}))
        with h5py.File(p, "r") as f:
            assert float(f.attrs["rotation_angle"]) == pytest.approx(25.0, abs=1e-3)

    def test_extras_mismatch_errors_at_save(self, tmp_path):
        p = str(tmp_path / "bad.h5")
        city = make_city(
            extras={"rectangle_vertices": RECT, "rotation_angle": 45.0}
        )
        with pytest.raises(ValueError, match="rotation_angle"):
            save_results_h5(p, city)

    def test_extras_consistent_rotation_passes(self, tmp_path):
        p = str(tmp_path / "ok.h5")
        rect = rotated_rect(25.0)
        city = make_city(
            extras={"rectangle_vertices": rect, "rotation_angle": 25.0}
        )
        save_results_h5(p, city)  # no raise

    def test_no_vertices_falls_back_to_bounds(self, tmp_path):
        p = str(tmp_path / "nb.h5")
        save_results_h5(p, make_city(extras={}))
        with h5py.File(p, "r") as f:
            rv = f["rectangle_vertices"][:]
            # bounds (0, 0, 0.01, 0.01) -> SW, NW, NE, SE
            np.testing.assert_allclose(
                rv,
                [[0.0, 0.0], [0.0, 0.01], [0.01, 0.01], [0.01, 0.0]],
            )
            assert float(f.attrs["rotation_angle"]) == 0.0

    def test_five_point_ring_accepted(self, tmp_path):
        p = str(tmp_path / "ring.h5")
        ring = RECT + [RECT[0]]  # closed 5-point ring
        save_results_h5(p, make_city(extras={"rectangle_vertices": ring}))
        with h5py.File(p, "r") as f:
            np.testing.assert_allclose(
                f["rectangle_vertices"][:], np.asarray(RECT, dtype=np.float64)
            )

    def test_malformed_vertices_raise(self, tmp_path):
        p = str(tmp_path / "bad_shape.h5")
        city = make_city(
            extras={"rectangle_vertices": [(0.0, 0.0), (0.0, 0.01), (0.01, 0.01)]}
        )
        with pytest.raises(ValueError):
            save_results_h5(p, city)
