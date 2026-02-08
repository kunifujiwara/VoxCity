"""
Run script for the VoxCity web application.

Usage:
    python run.py              # Start both backend and frontend
    python run.py --backend    # Start backend only
    python run.py --frontend   # Start frontend only
"""

import argparse
import socket
import subprocess
import sys
import os
import shutil
import signal
from pathlib import Path

APP_DIR = Path(__file__).parent
BACKEND_DIR = APP_DIR / "backend"
FRONTEND_DIR = APP_DIR / "frontend"

BACKEND_PORT = 8000
FRONTEND_PORT = 3000


def _port_in_use(port: int) -> bool:
    """Check whether a TCP port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def _get_python() -> str:
    """Return the Python executable path, preferring the active conda env."""
    # If running inside an activated conda env, sys.executable is correct.
    # Also check CONDA_PREFIX to build the path explicitly.
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidate = os.path.join(conda_prefix, "python.exe" if sys.platform == "win32" else "bin/python")
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def run_backend():
    """Start the FastAPI backend with uvicorn."""
    python = _get_python()
    print(f"[backend] Starting FastAPI server on http://localhost:8000  (python: {python})")
    return subprocess.Popen(
        [
            python,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
            "--reload-dir",
            "backend",
        ],
        cwd=str(APP_DIR),
    )


def run_frontend():
    """Start the Vite dev server."""
    print("[frontend] Starting Vite dev server on http://localhost:3000 ...")
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    return subprocess.Popen(
        [npm_cmd, "run", "dev"],
        cwd=str(FRONTEND_DIR),
    )


def install_frontend_deps():
    """Install frontend npm dependencies if node_modules does not exist."""
    if not (FRONTEND_DIR / "node_modules").exists():
        print("[frontend] Installing npm dependencies ...")
        npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
        subprocess.check_call([npm_cmd, "install"], cwd=str(FRONTEND_DIR))


def main():
    parser = argparse.ArgumentParser(description="Run VoxCity Web App")
    parser.add_argument("--backend", action="store_true", help="Start backend only")
    parser.add_argument("--frontend", action="store_true", help="Start frontend only")
    args = parser.parse_args()

    # If neither flag is set, run both
    run_be = not args.frontend or args.backend
    run_fe = not args.backend or args.frontend
    if not args.backend and not args.frontend:
        run_be = run_fe = True

    procs = []

    try:
        # Pre-flight: check ports
        if run_be and _port_in_use(BACKEND_PORT):
            print(f"[error] Port {BACKEND_PORT} is already in use. Stop the other process first.")
            sys.exit(1)
        if run_fe and _port_in_use(FRONTEND_PORT):
            print(f"[error] Port {FRONTEND_PORT} is already in use. Stop the other process first.")
            sys.exit(1)

        # Install frontend deps first (before backend starts watching files)
        if run_fe:
            install_frontend_deps()

        if run_be:
            procs.append(run_backend())

        # If running both, wait for the backend to be reachable before
        # starting the frontend so the Vite proxy doesn't hit ECONNREFUSED.
        if run_be and run_fe:
            import time
            print("[startup] Waiting for backend to be ready ...", end="", flush=True)
            for _ in range(60):  # up to 30 seconds
                if _port_in_use(BACKEND_PORT):
                    print(" OK")
                    break
                print(".", end="", flush=True)
                time.sleep(0.5)
            else:
                print("\n[warning] Backend not reachable yet – starting frontend anyway")

        if run_fe:
            procs.append(run_frontend())

        # Wait for any process to exit
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\nShutting down ...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()


if __name__ == "__main__":
    main()
