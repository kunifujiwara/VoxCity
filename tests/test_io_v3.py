"""Tests for the strict v3 HDF5 format: save stamps, strict load, migrate_h5, CLI."""

import json

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from voxcity.io import save_results_h5, load_results_h5, FORMAT_V3
from voxcity.utils.orientation import AXES_ATTR, check_axes

from tests.conftest import make_city, rotated_rect, write_v2_file, RECT  # shared fixtures


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


class TestMigrateCli:
    def test_default_destination(self, tmp_path):
        from voxcity.migrate import main

        src = write_v2_file(tmp_path / "old.h5")
        assert main([src]) == 0
        dst = str(tmp_path / "old_v3.h5")
        check_axes(dst)

    def test_out_flag(self, tmp_path):
        from voxcity.migrate import main

        src = write_v2_file(tmp_path / "old.h5")
        dst = str(tmp_path / "explicit.h5")
        assert main([src, "--out", dst]) == 0
        check_axes(dst)

    def test_out_with_multiple_inputs_rejected(self, tmp_path):
        from voxcity.migrate import main

        a = write_v2_file(tmp_path / "a.h5")
        b = write_v2_file(tmp_path / "b.h5")
        with pytest.raises(SystemExit):
            main([a, b, "--out", str(tmp_path / "x.h5")])

    def test_batch(self, tmp_path):
        from voxcity.migrate import main

        a = write_v2_file(tmp_path / "a.h5")
        b = write_v2_file(tmp_path / "b.h5")
        assert main([a, b]) == 0
        check_axes(str(tmp_path / "a_v3.h5"))
        check_axes(str(tmp_path / "b_v3.h5"))

    def test_batch_continues_past_failure_and_reports_nonzero(self, tmp_path):
        from voxcity.migrate import main

        missing = str(tmp_path / "does_not_exist.h5")
        good = write_v2_file(tmp_path / "good.h5")
        # The missing file errors but the good one still migrates; overall nonzero.
        rc = main([missing, good])
        assert rc == 1
        check_axes(str(tmp_path / "good_v3.h5"))

    def test_nonexistent_input_is_clean_error(self, tmp_path, capsys):
        from voxcity.migrate import main

        rc = main([str(tmp_path / "nope.h5")])
        assert rc == 1
        err = capsys.readouterr().err
        assert "nope.h5" in err
