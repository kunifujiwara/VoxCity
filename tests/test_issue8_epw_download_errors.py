"""Regression test for issue #8: EPW download errors must not be silently
swallowed into a None return. A network/SSL failure should surface as a clear
exception whose message names the real cause, so callers don't emit misleading
downstream errors (e.g. Path(None) TypeError)."""

import pytest
import requests

from voxcity.utils.weather import onebuilding


def test_ssl_error_is_surfaced_not_swallowed(monkeypatch, tmp_path):
    def raise_ssl(*a, **k):
        raise requests.exceptions.SSLError("certificate verify failed")

    monkeypatch.setattr(onebuilding.requests, "get", raise_ssl)

    with pytest.raises(Exception) as excinfo:
        onebuilding.get_nearest_epw_from_climate_onebuilding(
            longitude=4.9, latitude=52.4, output_dir=str(tmp_path)
        )
    msg = str(excinfo.value).lower()
    assert "epw" in msg or "ssl" in msg or "station" in msg or "download" in msg
