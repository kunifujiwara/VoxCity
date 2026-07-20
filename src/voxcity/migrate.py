"""Batch converter for pre-v3 VoxCity HDF5 files.

Usage::

    python -m voxcity.migrate FILE [FILE ...]        # writes <stem>_v3.h5 next to each
    python -m voxcity.migrate FILE --out PATH        # explicit destination (single input)

Thin argparse wrapper over :func:`voxcity.io.migrate_h5`.
"""

import argparse
import os
import sys

from .io import migrate_h5


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m voxcity.migrate",
        description="Convert pre-v3 VoxCity HDF5 files to the v3 format.",
    )
    parser.add_argument("files", nargs="+", help="pre-v3 .h5 files to convert")
    parser.add_argument(
        "--out",
        default=None,
        help="destination path (only valid with a single input file); "
        "default: <stem>_v3.h5 next to each input",
    )
    args = parser.parse_args(argv)

    if args.out and len(args.files) > 1:
        parser.error("--out is only valid with a single input file")

    failures = 0
    for src in args.files:
        stem, ext = os.path.splitext(src)
        dst = args.out or f"{stem}_v3{ext or '.h5'}"
        try:
            migrate_h5(src, dst)
        except (OSError, ValueError) as e:
            print(f"error: {src}: {e}", file=sys.stderr)
            failures += 1
            continue
        print(f"{src} -> {dst}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
