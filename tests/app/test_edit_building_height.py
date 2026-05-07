import types

import geopandas as gpd
import numpy as np
import pytest
from fastapi.testclient import TestClient
from shapely.geometry import Polygon

from app.backend.main import app
from app.backend.state import app_state


@pytest.fixture
def client():
    return TestClient(app)


def _object_grid(rows, cols):
    grid = np.empty((rows, cols), dtype=object)
    for row in range(rows):
        for col in range(cols):
            grid[row, col] = []
    return grid


def _make_building_gdf():
    gdf = gpd.GeoDataFrame(
        [
            {
                "geometry": Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                "id": 101,
                "building_id": 101,
                "height": 12.0,
                "min_height": 2.0,
                "height_estimated": False,
            },
            {
                "geometry": Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                "id": 202,
                "building_id": 202,
                "height": 8.0,
                "min_height": 0.0,
                "height_estimated": False,
            },
            {
                "geometry": Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
                "id": 303,
                "building_id": 303,
                "height": 9.0,
                "min_height": 1.0,
                "height_estimated": False,
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    gdf.index = [10, 20, 30]
    return gdf


def _make_voxcity():
    min_heights = _object_grid(2, 2)
    min_heights[0, 0] = [[2.0, 12.0]]
    min_heights[0, 1] = [[2.0, 12.0]]
    min_heights[1, 0] = [[0.0, 8.0]]
    min_heights[1, 1] = [[1.0, 9.0]]

    return types.SimpleNamespace(
        buildings=types.SimpleNamespace(
            heights=np.array([[12.0, 12.0], [8.0, 9.0]], dtype=float),
            ids=np.array([[101, 101], [202, 303]], dtype=np.int32),
            min_heights=min_heights,
        ),
        voxels=types.SimpleNamespace(
            classes=np.zeros((2, 2, 3), dtype=np.uint8),
            meta=types.SimpleNamespace(meshsize=1.0),
        ),
        tree_canopy=types.SimpleNamespace(top=None, bottom=None),
        land_cover=types.SimpleNamespace(classes=np.zeros((2, 2), dtype=np.int32)),
        extras={"building_gdf": _make_building_gdf()},
    )


def _install_edit_model(monkeypatch, voxcity):
    monkeypatch.setattr(app_state, "voxcity", voxcity)
    monkeypatch.setattr(app_state, "raw_data", {})
    monkeypatch.setattr(app_state, "rectangle_vertices", [[0, 0], [0, 2], [2, 2], [2, 0]])
    monkeypatch.setattr("app.backend.main.regenerate_voxels", lambda vc, inplace=True: vc)
    monkeypatch.setattr("app.backend.main._render_edit_preview", lambda vc: "{}")


def _post_height_edit(client, edit):
    return client.post("/api/model/apply_edits", json={"edits": [edit]})


def test_apply_edits_sets_height_for_sparse_dataframe_index(client, monkeypatch):
    voxcity = _make_voxcity()
    _install_edit_model(monkeypatch, voxcity)

    response = _post_height_edit(
        client,
        {"kind": "set_building_height", "building_ids": [10], "height_m": 20.0},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["n_edits"] == 1
    assert body["n_changed_total"] == 2
    assert voxcity.buildings.heights.tolist() == [[20.0, 20.0], [8.0, 9.0]]
    assert voxcity.buildings.min_heights[0, 0] == [[2.0, 20.0]]
    assert voxcity.buildings.min_heights[0, 1] == [[2.0, 20.0]]
    gdf = voxcity.extras["building_gdf"]
    row = gdf[gdf["id"] == 101].iloc[0]
    assert row["height"] == 20.0
    assert row["min_height"] == 2.0


def test_apply_edits_sets_group_height_and_min_height(client, monkeypatch):
    voxcity = _make_voxcity()
    _install_edit_model(monkeypatch, voxcity)

    response = _post_height_edit(
        client,
        {
            "kind": "set_building_height",
            "building_ids": [10, 20],
            "height_m": 24.0,
            "min_height_m": 3.0,
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["n_changed_total"] == 3
    assert voxcity.buildings.heights.tolist() == [[24.0, 24.0], [24.0, 9.0]]
    assert voxcity.buildings.min_heights[0, 0] == [[3.0, 24.0]]
    assert voxcity.buildings.min_heights[0, 1] == [[3.0, 24.0]]
    assert voxcity.buildings.min_heights[1, 0] == [[3.0, 24.0]]
    assert voxcity.buildings.min_heights[1, 1] == [[1.0, 9.0]]
    gdf = voxcity.extras["building_gdf"]
    rows = gdf[gdf["id"].isin([101, 202])]
    assert rows["height"].tolist() == [24.0, 24.0]
    assert rows["min_height"].tolist() == [3.0, 3.0]


def test_apply_edits_replays_height_then_delete_in_order(client, monkeypatch):
    voxcity = _make_voxcity()
    original_101_mask = voxcity.buildings.ids == 101
    _install_edit_model(monkeypatch, voxcity)

    response = client.post(
        "/api/model/apply_edits",
        json={
            "edits": [
                {"kind": "set_building_height", "building_ids": [10], "height_m": 20.0},
                {"kind": "delete_building", "building_ids": [10]},
            ],
        },
    )

    assert response.status_code == 200, response.text
    assert np.all(voxcity.buildings.heights[original_101_mask] == 0.0)
    assert np.all(voxcity.buildings.ids[original_101_mask] == 0)
    for row, col in np.argwhere(original_101_mask):
        assert voxcity.buildings.min_heights[row, col] == []


def test_apply_edits_preserves_unrelated_min_height_segments(client, monkeypatch):
    voxcity = _make_voxcity()
    voxcity.buildings.min_heights[0, 0] = [[0.0, 5.0], [2.0, 12.0]]
    _install_edit_model(monkeypatch, voxcity)

    response = _post_height_edit(
        client,
        {"kind": "set_building_height", "building_ids": [10], "height_m": 20.0},
    )

    assert response.status_code == 200, response.text
    assert voxcity.buildings.min_heights[0, 0] == [[0.0, 5.0], [2.0, 20.0]]


def test_apply_edits_uses_existing_min_fallback_when_no_segment_matches_previous_height(client, monkeypatch):
    voxcity = _make_voxcity()
    voxcity.buildings.min_heights[1, 0] = [[1.5, 6.0]]
    _install_edit_model(monkeypatch, voxcity)

    response = _post_height_edit(
        client,
        {"kind": "set_building_height", "building_ids": [20], "height_m": 16.0},
    )

    assert response.status_code == 200, response.text
    assert voxcity.buildings.min_heights[1, 0] == [[1.5, 16.0]]


@pytest.mark.parametrize(
    ("edit", "detail"),
    [
        ({"kind": "set_building_height", "building_ids": [10], "height_m": 0}, "height_m must be > 0"),
        ({"kind": "set_building_height", "building_ids": [10], "height_m": -1}, "height_m must be > 0"),
        ({"kind": "set_building_height", "building_ids": [10]}, "height_m required"),
        ({"kind": "set_building_height", "building_ids": [10], "height_m": "tall"}, "height_m required"),
        ({"kind": "set_building_height", "building_ids": [10], "height_m": True}, "height_m required"),
        ({"kind": "set_building_height", "building_ids": [10], "height_m": 10, "min_height_m": -1}, "min_height_m must be in [0, height_m)"),
        ({"kind": "set_building_height", "building_ids": [10], "height_m": 10, "min_height_m": 10}, "min_height_m must be in [0, height_m)"),
        ({"kind": "set_building_height", "building_ids": [10], "height_m": 10, "min_height_m": False}, "min_height_m must be a number"),
        ({"kind": "set_building_height", "building_ids": "10", "height_m": 10}, "building_ids must be a non-empty list"),
        ({"kind": "set_building_height", "building_ids": [], "height_m": 10}, "building_ids must be a non-empty list"),
        ({"kind": "set_building_height", "building_ids": ["abc"], "height_m": 10}, "building_ids must be integers"),
        ({"kind": "set_building_height", "building_ids": [10.5], "height_m": 10}, "building_ids must be integers"),
        ({"kind": "set_building_height", "building_ids": [True], "height_m": 10}, "building_ids must be integers"),
    ],
)
def test_apply_edits_rejects_invalid_height_payloads(client, monkeypatch, edit, detail):
    _install_edit_model(monkeypatch, _make_voxcity())

    response = _post_height_edit(client, edit)

    assert response.status_code == 400
    assert detail in response.json()["detail"]


@pytest.mark.parametrize(
    ("raw_value", "field", "detail"),
    [
        ("NaN", "height_m", "height_m must be finite"),
        ("Infinity", "height_m", "height_m must be finite"),
        ("NaN", "min_height_m", "min_height_m must be finite"),
    ],
)
def test_apply_edits_rejects_non_finite_height_payloads(client, monkeypatch, raw_value, field, detail):
    _install_edit_model(monkeypatch, _make_voxcity())
    if field == "height_m":
        body = f'{{"edits":[{{"kind":"set_building_height","building_ids":[10],"height_m":{raw_value}}}]}}'
    else:
        body = f'{{"edits":[{{"kind":"set_building_height","building_ids":[10],"height_m":10,"min_height_m":{raw_value}}}]}}'

    response = client.post(
        "/api/model/apply_edits",
        content=body,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert detail in response.json()["detail"]


def test_apply_edits_allows_stale_non_matching_building_id(client, monkeypatch):
    _install_edit_model(monkeypatch, _make_voxcity())

    response = _post_height_edit(
        client,
        {"kind": "set_building_height", "building_ids": [999], "height_m": 20.0},
    )

    assert response.status_code == 200, response.text
    assert response.json()["n_changed_total"] == 0


def test_apply_edits_does_not_update_empty_cells_when_gdf_id_is_zero(client, monkeypatch):
    voxcity = _make_voxcity()
    voxcity.buildings.heights[1, 1] = 0.0
    voxcity.buildings.ids[1, 1] = 0
    voxcity.buildings.min_heights[1, 1] = []
    voxcity.extras["building_gdf"].loc[30, "id"] = 0
    voxcity.extras["building_gdf"].loc[30, "building_id"] = 0
    _install_edit_model(monkeypatch, voxcity)

    response = _post_height_edit(
        client,
        {"kind": "set_building_height", "building_ids": [30], "height_m": 20.0},
    )

    assert response.status_code == 200, response.text
    assert response.json()["n_changed_total"] == 0
    assert voxcity.buildings.heights[1, 1] == 0.0
    assert voxcity.buildings.ids[1, 1] == 0
    assert voxcity.buildings.min_heights[1, 1] == []