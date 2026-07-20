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


def write_v2_file(path, with_vertices=True):
    """Hand-write a minimal pre-v3 (v2) file, as 1.x versions produced."""
    ny, nx, nz = 4, 5, 6
    extras = {"source": "test"}
    if with_vertices:
        extras["rectangle_vertices"] = RECT
    with h5py.File(path, "w") as f:
        f.attrs["__format__"] = "voxcity_results.v2"
        f.attrs["crs"] = "EPSG:4326"
        f.attrs["meshsize"] = 2.0
        f.attrs["bounds"] = [0.0, 0.0, 0.01, 0.01]
        vc = f.create_group("voxcity")
        vc.create_dataset("voxel_grid", data=np.zeros((ny, nx, nz), dtype=np.int8))
        vc.create_dataset("building_height", data=np.zeros((ny, nx)))
        vc.create_dataset("building_id", data=np.zeros((ny, nx)))
        vc.create_dataset("dem", data=np.zeros((ny, nx)))
        vc.create_dataset("land_cover", data=np.ones((ny, nx), dtype=int))
        vc.attrs["extras_json"] = json.dumps(extras)
    return str(path)


class TestStrictLoad:
    def test_v3_round_trip(self, tmp_path):
        p = str(tmp_path / "rt.h5")
        save_results_h5(p, make_city())
        out = load_results_h5(p)
        assert out["meta"]["rotation_angle"] == 0.0
        assert [tuple(v) for v in out["meta"]["rectangle_vertices"]] == [
            (0.0, 0.0), (0.0, 0.01), (0.01, 0.01), (0.01, 0.0)
        ]
        ex = out["voxcity"].extras
        assert ex["rotation_angle"] == 0.0
        assert [tuple(v) for v in ex["rectangle_vertices"]] == [
            (0.0, 0.0), (0.0, 0.01), (0.01, 0.01), (0.01, 0.0)
        ]

    def test_v2_file_refused_with_migrate_pointer(self, tmp_path):
        p = write_v2_file(tmp_path / "old.h5")
        with pytest.raises(ValueError, match="migrate_h5"):
            load_results_h5(p)

    def test_foreign_file_refused(self, tmp_path):
        p = str(tmp_path / "foreign.h5")
        with h5py.File(p, "w") as f:
            f.create_dataset("data", data=np.zeros(3))
        with pytest.raises(ValueError, match="migrate_h5"):
            load_results_h5(p)

    def test_v3_tag_but_missing_geometry_raises_clear_error(self, tmp_path):
        # A file that declares v3 but was truncated before the geometry was
        # written must give a clear error, not a raw h5py KeyError.
        p = str(tmp_path / "truncated.h5")
        with h5py.File(p, "w") as f:
            f.attrs["__format__"] = FORMAT_V3
            f.attrs["axes"] = AXES_ATTR
        with pytest.raises(ValueError, match="corrupted"):
            load_results_h5(p)


class TestMigrateH5:
    def test_migrated_file_loads_and_passes_check_axes(self, tmp_path):
        from voxcity.io import migrate_h5

        src = write_v2_file(tmp_path / "old.h5")
        dst = str(tmp_path / "new.h5")
        migrate_h5(src, dst)
        out = load_results_h5(dst)
        assert out["meta"]["rotation_angle"] == 0.0
        check_axes(dst)

    def test_provenance_extras_geometry(self, tmp_path):
        from voxcity.io import migrate_h5

        src = write_v2_file(tmp_path / "old.h5", with_vertices=True)
        dst = str(tmp_path / "new.h5")
        migrate_h5(src, dst)
        with h5py.File(dst, "r") as f:
            assert f.attrs["migrated_from"] == "voxcity_results.v2"
            assert f.attrs["geometry_source"] == "extras"
            np.testing.assert_allclose(
                f["rectangle_vertices"][:], np.asarray(RECT, dtype=np.float64)
            )

    def test_provenance_bounds_fallback(self, tmp_path):
        from voxcity.io import migrate_h5

        src = write_v2_file(tmp_path / "old.h5", with_vertices=False)
        dst = str(tmp_path / "new.h5")
        migrate_h5(src, dst)
        with h5py.File(dst, "r") as f:
            assert f.attrs["geometry_source"] == "bounds"
            np.testing.assert_allclose(
                f["rectangle_vertices"][:],
                [[0.0, 0.0], [0.0, 0.01], [0.01, 0.01], [0.01, 0.0]],
            )

    def test_refuses_in_place(self, tmp_path):
        from voxcity.io import migrate_h5

        src = write_v2_file(tmp_path / "old.h5")
        with pytest.raises(ValueError, match="overwrite"):
            migrate_h5(src, src)

    def test_refuses_already_v3(self, tmp_path):
        from voxcity.io import migrate_h5

        p = str(tmp_path / "v3.h5")
        save_results_h5(p, make_city())
        with pytest.raises(ValueError, match="already"):
            migrate_h5(p, str(tmp_path / "v3b.h5"))

    def test_source_untouched(self, tmp_path):
        from voxcity.io import migrate_h5

        src = write_v2_file(tmp_path / "old.h5")
        migrate_h5(src, str(tmp_path / "new.h5"))
        with h5py.File(src, "r") as f:
            assert f.attrs["__format__"] == "voxcity_results.v2"

    def test_existing_dst_preserved_when_src_already_v3(self, tmp_path):
        from voxcity.io import migrate_h5

        v3 = str(tmp_path / "already.h5")
        save_results_h5(v3, make_city())
        dst = tmp_path / "dst.h5"
        dst.write_bytes(b"precious original contents")
        with pytest.raises(ValueError, match="already"):
            migrate_h5(v3, str(dst))
        # dst must be untouched — validation happens before any copy.
        assert dst.read_bytes() == b"precious original contents"

    def test_five_point_ring_via_extras_migrates_to_four(self, tmp_path):
        from voxcity.io import migrate_h5

        src = str(tmp_path / "ring.h5")
        ring = RECT + [RECT[0]]  # closed 5-point ring in extras_json
        with h5py.File(src, "w") as f:
            f.attrs["__format__"] = "voxcity_results.v2"
            f.attrs["crs"] = "EPSG:4326"
            f.attrs["meshsize"] = 2.0
            f.attrs["bounds"] = [0.0, 0.0, 0.01, 0.01]
            vc = f.create_group("voxcity")
            vc.create_dataset("voxel_grid", data=np.zeros((4, 5, 6), dtype=np.int8))
            vc.attrs["extras_json"] = json.dumps({"rectangle_vertices": ring})
        dst = str(tmp_path / "ring_v3.h5")
        migrate_h5(src, dst)
        with h5py.File(dst, "r") as f:
            assert f["rectangle_vertices"].shape == (4, 2)
            assert f.attrs["geometry_source"] == "extras"
            np.testing.assert_allclose(
                f["rectangle_vertices"][:], np.asarray(RECT, dtype=np.float64)
            )
