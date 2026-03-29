from unittest.mock import patch, MagicMock

import requests

from voxcity.downloader import utils


def test_download_file(tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.iter_content = MagicMock(return_value=[b'test content'])
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch('requests.get', return_value=mock_response):
        filepath = tmp_path / "test.txt"
        utils.download_file("http://test.com/file", str(filepath))
        assert filepath.exists()
        assert filepath.read_bytes() == b'test content'


def test_download_file_failure(tmp_path):
    """Test download_file with failed HTTP status raises after retries."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)

    with patch('requests.get', return_value=mock_response):
        filepath = tmp_path / "test.txt"
        import pytest
        with pytest.raises(requests.HTTPError, match="Failed to download"):
            utils.download_file(
                "http://test.com/notfound", str(filepath),
                max_retries=1, initial_delay=0.0,
            )
        # File should not be created on failure
        assert not filepath.exists()

