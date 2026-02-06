"""
Tests for voxcity.utils.weather.files module.
"""

import os
import tempfile
import zipfile
from pathlib import Path

import pytest


class TestSafeRename:
    """Tests for safe_rename function."""

    def test_rename_no_conflict(self):
        """Test renaming when destination doesn't exist."""
        from voxcity.utils.weather.files import safe_rename
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.txt"
            dst = Path(tmpdir) / "dest.txt"
            
            src.write_text("content")
            
            result = safe_rename(src, dst)
            
            assert result == dst
            assert dst.exists()
            assert not src.exists()
            assert dst.read_text() == "content"

    def test_rename_with_conflict(self):
        """Test renaming when destination already exists."""
        from voxcity.utils.weather.files import safe_rename
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.txt"
            dst = Path(tmpdir) / "dest.txt"
            
            src.write_text("new content")
            dst.write_text("existing content")
            
            result = safe_rename(src, dst)
            
            # Should create dest_1.txt
            assert result == Path(tmpdir) / "dest_1.txt"
            assert result.exists()
            assert result.read_text() == "new content"
            assert dst.read_text() == "existing content"  # Original unchanged

    def test_rename_with_multiple_conflicts(self):
        """Test renaming when multiple conflicts exist."""
        from voxcity.utils.weather.files import safe_rename
        
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.txt"
            dst = Path(tmpdir) / "dest.txt"
            
            src.write_text("new content")
            dst.write_text("existing content")
            (Path(tmpdir) / "dest_1.txt").write_text("conflict 1")
            (Path(tmpdir) / "dest_2.txt").write_text("conflict 2")
            
            result = safe_rename(src, dst)
            
            # Should create dest_3.txt
            assert result == Path(tmpdir) / "dest_3.txt"
            assert result.exists()
            assert result.read_text() == "new content"


class TestSafeExtract:
    """Tests for safe_extract function."""

    def test_extract_basic(self):
        """Test basic file extraction."""
        from voxcity.utils.weather.files import safe_extract
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            zip_path = tmpdir / "test.zip"
            extract_dir = tmpdir / "extracted"
            extract_dir.mkdir()
            
            # Create test zip
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("test_file.txt", "hello world")
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zf:
                result = safe_extract(zf, "test_file.txt", extract_dir)
            
            assert result == extract_dir / "test_file.txt"
            assert result.exists()
            assert result.read_text() == "hello world"

    def test_extract_nested_file(self):
        """Test extracting file from nested directory in zip."""
        from voxcity.utils.weather.files import safe_extract
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            zip_path = tmpdir / "test.zip"
            extract_dir = tmpdir / "extracted"
            extract_dir.mkdir()
            
            # Create test zip with nested structure
            with zipfile.ZipFile(zip_path, 'w') as zf:
                zf.writestr("dir/subdir/file.txt", "nested content")
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zf:
                result = safe_extract(zf, "dir/subdir/file.txt", extract_dir)
            
            assert result == extract_dir / "dir/subdir/file.txt"
            assert result.exists()
            assert result.read_text() == "nested content"
