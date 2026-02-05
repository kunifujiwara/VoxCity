from unittest.mock import patch, MagicMock

from voxcity.downloader import utils


def test_download_file(tmp_path):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b'test content'

    with patch('requests.get', return_value=mock_response):
        filepath = tmp_path / "test.txt"
        utils.download_file("http://test.com/file", str(filepath))
        assert filepath.exists()
        assert filepath.read_bytes() == b'test content'


def test_download_file_failure(tmp_path, capsys):
    """Test download_file with failed HTTP status."""
    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch('requests.get', return_value=mock_response):
        filepath = tmp_path / "test.txt"
        utils.download_file("http://test.com/notfound", str(filepath))
        # File should not be created on failure
        assert not filepath.exists()
        # Should print error message
        captured = capsys.readouterr()
        assert "Failed to download" in captured.out
        assert "404" in captured.out


@patch('gdown.download')
def test_download_file_google_drive(mock_gdown, tmp_path):
    mock_gdown.return_value = True
    result = utils.download_file_google_drive("test_id", str(tmp_path / "test.txt"))
    assert result is True
    mock_gdown.assert_called_once()


@patch('gdown.download')
def test_download_file_google_drive_failure(mock_gdown, tmp_path, capsys):
    """Test download_file_google_drive with exception."""
    mock_gdown.side_effect = Exception("Network error")
    result = utils.download_file_google_drive("test_id", str(tmp_path / "test.txt"))
    assert result is False
    # Should print error message
    captured = capsys.readouterr()
    assert "Error downloading" in captured.out
    assert "Network error" in captured.out

