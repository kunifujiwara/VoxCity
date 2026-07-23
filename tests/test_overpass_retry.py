"""Tests for Overpass retry policy: short connect timeout, dead-endpoint
skipping, and endpoint rotation so one dead mirror can't stall generation."""

import requests
import pytest

from voxcity.downloader import osm as osm_mod


class _Resp:
    status_code = 200
    headers = {"Content-Type": "application/json"}
    def json(self):
        return {"elements": []}


def test_connect_timeout_is_short(monkeypatch):
    """requests.get must be called with a (connect, read) timeout tuple whose
    connect part is small, so a dead endpoint costs seconds, not 60 s."""
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen["timeout"] = timeout
        return _Resp()

    monkeypatch.setattr(osm_mod.requests, "get", fake_get)
    osm_mod._fetch_overpass_with_retry("[out:json];", timeout=60)
    assert isinstance(seen["timeout"], tuple), "timeout must be (connect, read)"
    connect, read = seen["timeout"]
    assert connect <= 10
    assert read == 60


def test_dead_endpoint_skipped_after_connect_failure(monkeypatch):
    """An endpoint that raises ConnectionError/ConnectTimeout must not be
    retried on later attempts within the same call."""
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        if "dead" in url:
            raise requests.exceptions.ConnectTimeout("no route")
        raise requests.exceptions.Timeout("read timed out")  # transient

    monkeypatch.setattr(osm_mod.requests, "get", fake_get)
    with pytest.raises(Exception):
        osm_mod._fetch_overpass_with_retry(
            "[out:json];",
            timeout=1,
            max_retries=3,
            initial_delay=0.0,
            endpoints=["https://dead.example/api", "https://slow.example/api"],
        )
    dead_calls = [c for c in calls if "dead" in c]
    assert len(dead_calls) == 1, f"dead endpoint retried {len(dead_calls)} times"


def test_success_on_second_endpoint(monkeypatch):
    def fake_get(url, params=None, headers=None, timeout=None):
        if "dead" in url:
            raise requests.exceptions.ConnectTimeout("no route")
        return _Resp()

    monkeypatch.setattr(osm_mod.requests, "get", fake_get)
    data = osm_mod._fetch_overpass_with_retry(
        "[out:json];",
        timeout=1,
        initial_delay=0.0,
        endpoints=["https://dead.example/api", "https://ok.example/api"],
    )
    assert data == {"elements": []}
