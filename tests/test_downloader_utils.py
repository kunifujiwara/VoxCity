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

