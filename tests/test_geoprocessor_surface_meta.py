"""Tests for voxcity.geoprocessor.surface_meta face-key comparison.

Face keys used to end in ``:i<face_index>`` -- the face's position in
whatever enumeration minted it. That made a key depend on enumeration order,
so two producers walking the same faces in different orders minted different
keys for the same face. The suffix is gone.

Keys minted before that change are still in circulation: downstream apps
persist zone selectors verbatim in saved session archives and share
snapshots, and a session-restored cached mesh can carry old-format metadata
(``surface_face_meta_version`` is deliberately still 1 -- normalization, not
a version bump, is what reconciles the two formats). So both sides of every
face-key comparison are normalized, and a stored old-format key must keep
selecting the face it was picked on -- statistics included, not just
outlines.
"""
import numpy as np

from voxcity.geoprocessor.surface_meta import (
    make_surface_face_key,
    normalize_surface_face_key,
    surface_zone_mask,
)

BUILDING = 7
SOUTH = (0.0, -1.0, 0.0)

# Two distinct faces of one building, keyed the way the current code mints them.
KEY_NEAR = make_surface_face_key(BUILDING, (1.0, 2.0, 3.0), SOUTH)
KEY_FAR = make_surface_face_key(BUILDING, (1.0, 8.0, 3.0), SOUTH)
# A face of the same building that no selector below ever names.
KEY_ABSENT = make_surface_face_key(BUILDING, (5.0, 5.0, 5.0), SOUTH)


def _meta(*face_keys):
    return [
        {"face_key": key, "building_id": BUILDING, "surface_kind": "wall",
         "orientation": "S", "is_window": False}
        for key in face_keys
    ]


def _legacy(face_key, index):
    """The same face key as it was minted before the index was dropped."""
    return f"{face_key}:i{index}"


def test_normalize_strips_only_the_index_suffix():
    assert normalize_surface_face_key(_legacy(KEY_NEAR, 41)) == KEY_NEAR
    # Already-normalized keys, and keys of any other shape, pass through.
    assert normalize_surface_face_key(KEY_NEAR) == KEY_NEAR
    assert normalize_surface_face_key("f0") == "f0"
    # Only a *trailing* index, and only digits: nothing else may be eaten.
    assert normalize_surface_face_key("b1:c0_0_0:i3:n0_0_0") == "b1:c0_0_0:i3:n0_0_0"
    assert normalize_surface_face_key("b1:c0_0_0:iX") == "b1:c0_0_0:iX"


def test_a_selector_key_stored_in_the_old_format_selects_its_face():
    """The guarantee the whole normalization exists for, on the statistics side."""
    mask = surface_zone_mask(
        _meta(KEY_NEAR, KEY_FAR),
        [{"building_id": BUILDING, "mode": "faces",
          "face_keys": [_legacy(KEY_NEAR, 41)]}],
    )
    np.testing.assert_array_equal(mask, [True, False])


def test_an_exclusion_key_stored_in_the_old_format_excludes_its_face():
    mask = surface_zone_mask(
        _meta(KEY_NEAR, KEY_FAR),
        [
            {"building_id": BUILDING, "mode": "whole"},
            {"building_id": BUILDING, "mode": "exclude_faces",
             "face_keys": [_legacy(KEY_NEAR, 41)]},
        ],
    )
    np.testing.assert_array_equal(mask, [False, True])


def test_an_old_format_key_for_another_face_selects_nothing():
    """Normalization must not blur distinct faces together.

    Stripping the suffix has to leave the part that identifies the face
    intact, or every legacy key would match every face and the two tests
    above would pass for the wrong reason.
    """
    mask = surface_zone_mask(
        _meta(KEY_NEAR, KEY_FAR),
        [{"building_id": BUILDING, "mode": "faces",
          "face_keys": [_legacy(KEY_ABSENT, 41)]}],
    )
    np.testing.assert_array_equal(mask, [False, False])


def test_old_format_metadata_matches_a_new_format_selector_key():
    """The reverse direction: stale keys can arrive on the metadata side too.

    A mesh restored from a saved session can carry surface_face_meta minted
    before the format change, while the selector holds a freshly picked
    new-format key. Normalizing only the selector side would miss this.
    """
    mask = surface_zone_mask(
        _meta(_legacy(KEY_NEAR, 41), _legacy(KEY_FAR, 42)),
        [{"building_id": BUILDING, "mode": "faces", "face_keys": [KEY_NEAR]}],
    )
    np.testing.assert_array_equal(mask, [True, False])


def test_normalizer_matches_its_javascript_twin_on_the_edge_cases():
    """Two divergences that used to exist between the Python and JS regexes.

    Neither is reachable from a minted key, but the two are meant to be the
    same function on every input, not just on the ones we produce.
    """
    # `$` would strip before a trailing newline; `\Z` does not.
    assert normalize_surface_face_key(KEY_NEAR + ":i7\n") == KEY_NEAR + ":i7\n"
    # `\d` is Unicode-wide in Python; `[0-9]` is not, matching JavaScript.
    assert normalize_surface_face_key(KEY_NEAR + ":i\u0967") == KEY_NEAR + ":i\u0967"
