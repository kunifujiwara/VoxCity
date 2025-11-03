from pathlib import Path
import os
import zipfile


def safe_rename(src: Path, dst: Path) -> Path:
    """
    Safely rename a file, handling existing files by adding a number suffix.
    """
    if not dst.exists():
        src.rename(dst)
        return dst
    base = dst.stem
    ext = dst.suffix
    counter = 1
    while True:
        new_dst = dst.with_name(f"{base}_{counter}{ext}")
        if not new_dst.exists():
            src.rename(new_dst)
            return new_dst
        counter += 1


def safe_extract(zip_ref: zipfile.ZipFile, filename: str, extract_dir: Path) -> Path:
    """
    Safely extract a file from zip, handling existing files.
    """
    try:
        zip_ref.extract(filename, extract_dir)
        return extract_dir / filename
    except FileExistsError:
        temp_name = f"temp_{os.urandom(4).hex()}_{filename}"
        zip_ref.extract(filename, extract_dir, temp_name)
        return extract_dir / temp_name


