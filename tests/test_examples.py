"""Tests for voxcity.examples: ready-made city models downloaded from Hugging Face."""

import hashlib
import io

import pytest

from tests.conftest import make_city
from voxcity import examples
from voxcity.errors import DownloaderError
from voxcity.io import save_voxcity


def _h5_bytes(tmp_path):
    path = tmp_path / "src.h5"
    save_voxcity(str(path), make_city())
    return path.read_bytes()


class _Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size=1):
        buf = io.BytesIO(self.payload)
        while True:
            chunk = buf.read(chunk_size)
            if not chunk:
                return
            yield chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def fake_hub(tmp_path, monkeypatch):
    """A catalog with one city and a requests.get that serves its HDF5 file."""
    payload = _h5_bytes(tmp_path)
    catalog = [{
        "name": "testville",
        "title": "Testville, Nowhere",
        "file": "cities/testville.h5",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }]
    calls = []

    def fake_get(url, stream=False, timeout=None, headers=None):
        calls.append(url)
        return _Response(payload)

    monkeypatch.setattr(examples, "_load_catalog", lambda: catalog)
    monkeypatch.setattr(examples.requests, "get", fake_get)
    monkeypatch.delenv("VOXCITY_EXAMPLES_REPO", raising=False)
    return {"catalog": catalog, "calls": calls, "payload": payload, "cache": tmp_path / "cache"}


def test_bundled_catalog_lists_cities_with_required_fields():
    cities = examples.list_example_cities()
    assert len(cities) >= 10
    names = [c["name"] for c in cities]
    assert len(names) == len(set(names))
    for c in cities:
        for key in ("name", "title", "country", "center", "size_m", "meshsize", "sources", "file", "sha256", "bytes"):
            assert key in c, (c.get("name"), key)
        assert c["file"].startswith("cities/") and c["file"].endswith(".h5")
        assert c["sources"]["dem_source"] in ("FABDEM", "USGS 3DEP 10m", "GSI DEM Japan")


def test_list_returns_copies(fake_hub):
    cities = examples.list_example_cities()
    cities[0]["name"] = "changed"
    assert examples.list_example_cities()[0]["name"] == "testville"


def test_load_downloads_once_then_uses_cache(fake_hub):
    city = examples.load_example_city("testville", cache_dir=fake_hub["cache"])
    assert city.voxels.classes.shape == (4, 5, 6)
    assert len(fake_hub["calls"]) == 1
    url = fake_hub["calls"][0]
    assert url == f"https://huggingface.co/datasets/{examples.EXAMPLES_REPO}/resolve/main/cities/testville.h5"

    examples.load_example_city("testville", cache_dir=fake_hub["cache"])
    assert len(fake_hub["calls"]) == 1          # served from the cache

    examples.load_example_city("testville", cache_dir=fake_hub["cache"], force_download=True)
    assert len(fake_hub["calls"]) == 2


@pytest.mark.parametrize("name", ["Testville", " testville ", "TESTVILLE"])
def test_name_is_case_and_space_insensitive(fake_hub, name):
    examples.load_example_city(name, cache_dir=fake_hub["cache"])


def test_spaces_and_hyphens_map_to_underscores(fake_hub):
    fake_hub["catalog"][0]["name"] = "new_york"
    examples.load_example_city("New York", cache_dir=fake_hub["cache"])
    examples.load_example_city("new-york", cache_dir=fake_hub["cache"])


def test_unknown_name_lists_available(fake_hub):
    with pytest.raises(ValueError, match="testville"):
        examples.load_example_city("atlantis", cache_dir=fake_hub["cache"])


def test_checksum_mismatch_raises_and_keeps_no_file(fake_hub):
    fake_hub["catalog"][0]["sha256"] = "0" * 64
    with pytest.raises(DownloaderError, match="checksum"):
        examples.load_example_city("testville", cache_dir=fake_hub["cache"])
    assert not any(p.is_file() for p in fake_hub["cache"].rglob("*"))


def test_corrupt_cached_file_is_downloaded_again(fake_hub):
    examples.load_example_city("testville", cache_dir=fake_hub["cache"])
    cached = next(p for p in fake_hub["cache"].rglob("*.h5"))
    cached.write_bytes(b"broken")
    examples.load_example_city("testville", cache_dir=fake_hub["cache"])
    assert len(fake_hub["calls"]) == 2


def test_http_error_becomes_downloader_error(fake_hub, monkeypatch):
    monkeypatch.setattr(examples.requests, "get", lambda *a, **k: _Response(b"", status=404))
    with pytest.raises(DownloaderError, match="testville"):
        examples.load_example_city("testville", cache_dir=fake_hub["cache"])


def test_repo_and_revision_can_be_overridden(fake_hub, monkeypatch):
    monkeypatch.setenv("VOXCITY_EXAMPLES_REPO", "someone/other-cities")
    examples.load_example_city("testville", cache_dir=fake_hub["cache"], revision="v1")
    assert fake_hub["calls"][-1] == "https://huggingface.co/datasets/someone/other-cities/resolve/v1/cities/testville.h5"


def test_default_cache_dir_honours_env(fake_hub, monkeypatch, tmp_path):
    monkeypatch.setenv("VOXCITY_CACHE_DIR", str(tmp_path / "envcache"))
    path = examples.example_city_path("testville")
    assert str(path).startswith(str(tmp_path / "envcache"))
    assert path.is_file()
