"""Tests for voxcity.utils.weather.files module."""
import pytest
import zipfile
from pathlib import Path

from voxcity.utils.weather.files import safe_rename, safe_extract


class TestSafeRename:
    """Tests for safe_rename function."""

    def test_rename_new_file(self, tmp_path):
        """Test renaming to a non-existent destination."""
        src = tmp_path / "source.txt"
        src.write_text("content")
        dst = tmp_path / "dest.txt"
        
        result = safe_rename(src, dst)
        
        assert result == dst
        assert dst.exists()
        assert not src.exists()
        assert dst.read_text() == "content"

    def test_rename_with_existing_destination(self, tmp_path):
        """Test renaming when destination already exists."""
        src = tmp_path / "source.txt"
        src.write_text("new content")
        dst = tmp_path / "dest.txt"
        dst.write_text("existing content")
        
        result = safe_rename(src, dst)
        
        # Should create numbered version
        assert result == tmp_path / "dest_1.txt"
        assert result.exists()
        assert not src.exists()
        assert result.read_text() == "new content"
        assert dst.read_text() == "existing content"

    def test_rename_with_multiple_existing(self, tmp_path):
        """Test renaming when multiple numbered versions exist."""
        src = tmp_path / "source.txt"
        src.write_text("content")
        dst = tmp_path / "dest.txt"
        dst.write_text("original")
        (tmp_path / "dest_1.txt").write_text("first")
        (tmp_path / "dest_2.txt").write_text("second")
        
        result = safe_rename(src, dst)
        
        assert result == tmp_path / "dest_3.txt"
        assert result.exists()

    def test_preserves_extension(self, tmp_path):
        """Test that extension is preserved in numbered names."""
        src = tmp_path / "file.epw"
        src.write_text("weather data")
        dst = tmp_path / "output.epw"
        dst.write_text("existing")
        
        result = safe_rename(src, dst)
        
        assert result.suffix == ".epw"
        assert result.name == "output_1.epw"


class TestSafeExtract:
    """Tests for safe_extract function."""

    def test_extract_new_file(self, tmp_path):
        """Test extracting to a non-existent destination."""
        # Create a zip file with a test file
        zip_path = tmp_path / "test.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.writestr("inner.txt", "zip content")
        
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        
        with zipfile.ZipFile(zip_path, 'r') as zf:
            result = safe_extract(zf, "inner.txt", extract_dir)
        
        assert result == extract_dir / "inner.txt"
        assert result.exists()
        assert result.read_text() == "zip content"
