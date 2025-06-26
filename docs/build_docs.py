#!/usr/bin/env python3
"""
Build script for VoxCity documentation.
This script can be used both locally and in CI/CD environments.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(cmd, cwd=None):
    """Run a command and return the result."""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result.stdout

def build_documentation():
    """Build the documentation."""
    docs_dir = Path(__file__).parent
    
    # Clean previous builds
    build_dir = docs_dir / "_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    
    # Build HTML documentation
    print("Building HTML documentation...")
    run_command("make html", cwd=docs_dir)
    
    # Copy .nojekyll file to build directory for GitHub Pages
    nojekyll_file = build_dir / "html" / ".nojekyll"
    nojekyll_file.touch()
    
    print("Documentation built successfully!")
    print(f"HTML files are in: {build_dir / 'html'}")
    
    return build_dir / "html"

def serve_documentation(build_dir):
    """Serve the documentation locally."""
    print(f"Serving documentation from: {build_dir}")
    print("Open http://localhost:8000 in your browser")
    print("Press Ctrl+C to stop the server")
    
    try:
        run_command(f"python -m http.server 8000", cwd=build_dir)
    except KeyboardInterrupt:
        print("\nServer stopped.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build VoxCity documentation")
    parser.add_argument("--serve", action="store_true", help="Serve documentation locally after building")
    parser.add_argument("--clean", action="store_true", help="Clean build directory before building")
    
    args = parser.parse_args()
    
    if args.clean:
        docs_dir = Path(__file__).parent
        build_dir = docs_dir / "_build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            print("Cleaned build directory")
    
    build_dir = build_documentation()
    
    if args.serve:
        serve_documentation(build_dir) 