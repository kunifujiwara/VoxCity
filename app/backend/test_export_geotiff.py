"""HTTP-level test for /api/export/geotiff."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    # Fake export_geotiffs: write dummy .tif files (+ a README.md, mirroring the
    # real exporter's write_readme default), return {layer/"readme": path}.
    def fake_export_geotiffs(city, output_directory, base_filename="voxcity", *,
                             write_readme=True, **kwargs):
        out = {}
        for layer in ("land_cover", "building_height", "dem", "canopy_height"):
            p = Path(output_directory) / f"{base_filename}_{layer}.tif"
            p.write_bytes(b"II*\x00")  # minimal TIFF magic
            out[layer] = str(p)
        if write_readme:
            r = Path(output_directory) / "README.md"
            r.write_text("# VoxCity GeoTIFF Export\n", encoding="utf-8")
            out["readme"] = str(r)
        return out

    import voxcity.exporter.geotiff as geo
    monkeypatch.setattr(geo, "export_geotiffs", fake_export_geotiffs, raising=False)

    from backend.main import app, app_state
    app_state.voxcity = SimpleNamespace(extras={})
    app_state.rectangle_vertices = [[139.0, 35.0], [139.1, 35.0], [139.1, 35.1], [139.0, 35.1]]
    app_state.land_cover_source = "OpenStreetMap"
    yield TestClient(app)
    app_state.voxcity = None


def test_export_geotiff_returns_zip_of_tifs_and_readme(client: TestClient):
    r = client.post("/api/export/geotiff", json={"filename": "mycity"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = {Path(n).name for n in zf.namelist()}
    assert names == {
        "mycity_land_cover.tif", "mycity_building_height.tif",
        "mycity_dem.tif", "mycity_canopy_height.tif", "README.md",
    }


def test_export_geotiff_requires_model(monkeypatch):
    from backend.main import app, app_state
    app_state.voxcity = None
    r = TestClient(app).post("/api/export/geotiff", json={"filename": "x"})
    assert r.status_code == 400
