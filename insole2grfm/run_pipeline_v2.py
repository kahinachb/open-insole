#!/usr/bin/env python3
"""Run stages 1 and 2 of the pipeline: synchronisation, then the dataset (see docs/pipeline.md)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
DEFAULT_RAW = REPO.parent / "manip"
SYNC_BATCH = REPO.parent / "acquisition" / "process_manip_v2.py"


def run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--prepared-dir", type=Path, default=REPO.parent / "dataset_v3.7.1")
    parser.add_argument("--dataset-version", default="3.7.1")
    parser.add_argument("--static-plate-fz", type=Path, default=REPO / "static_plate_fz.csv",
                        help="static plate Fz of extract_static_plate_fz.py; without it, body weights fall back "
                        "to the balance trials")
    parser.add_argument("--force-sync", action="store_true")
    parser.add_argument("--skip-sync", action="store_true")
    args = parser.parse_args()

    python = sys.executable
    if not args.skip_sync:
        run([python, str(SYNC_BATCH), str(args.raw_root), *(["--force"] if args.force_sync else [])])
    run([
        python, str(REPO / "prepare_npy_joint_masked_v3.py"), str(args.raw_root),
        "--dataset-version", args.dataset_version,
        *(["--static-plate-fz", str(args.static_plate_fz)] if args.static_plate_fz.exists() else []),
        "--output-dir", str(args.prepared_dir),
    ])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
