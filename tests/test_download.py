import pytest
from unittest.mock import patch, MagicMock
import requests
from pathlib import Path

from voxcity.download import utils

def test_download_file(tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'test content'
    
    with patch('requests.get', return_value=mock_response):
        filepath = tmp_path / "test.txt"
        utils.download_file("http://test.com/file", str(filepath))
        assert filepath.exists()
        assert filepath.read_bytes() == b'test content'

@patch('gdown.download')
def test_download_file_google_drive(mock_gdown, tmp_path):
    mock_gdown.return_value = True
    result = utils.download_file_google_drive("test_id", str(tmp_path / "test.txt"))
    assert result is True
    mock_gdown.assert_called_once() 