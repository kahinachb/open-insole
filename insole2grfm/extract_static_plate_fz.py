"""Extract the force-plate Fz of every static trial's C3D to one CSV (ticket #26).

Runs in the separate ``c3d-io`` conda env, which has ezc3d (the main env has not)::

    conda create -n c3d-io -c conda-forge python=3.11 ezc3d numpy
    envs/c3d-io/bin/python extract_static_plate_fz.py \
        .../manip --output .../static_plate_fz.csv

For each ``<manip>/<Subject>/static*/`` directory holding a ``.c3d`` (``*_gapf.c3d``
preferred, like the sync stage), it writes one row per analog sample with the
global vertical force of both plates, as computed by ezc3d's force-platform filter
(the same ``force_z_p<n>_global_N`` as the synchronised trials, positive upwards
under load, no low-pass). Columns: ``subject, trial, c3d_file, time_s,
force_z_p1_global_N, force_z_p2_global_N``. The builder
(``prepare_npy_joint_masked_v3.py --static-plate-fz``) reads that file in the main env
and computes the Body weight from it. Read-only on the raw data. Numpy only, no pandas.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

COLUMNS = ("subject", "trial", "c3d_file", "time_s", "force_z_p1_global_N", "force_z_p2_global_N")


def read_plate_fz(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    """Analog rate (Hz) and global Fz (N) of plates 1 and 2 of a C3D, via ezc3d."""
    import ezc3d  # only in the c3d-io env

    c3d = ezc3d.c3d(str(path), extract_forceplat_data=True)
    rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])
    platforms = c3d["data"].get("platform", [])
    if len(platforms) < 2:
        raise ValueError(f"{path}: {len(platforms)} plateforme(s), 2 attendues")
    fz = [np.asarray(platform["force"], dtype=float)[2] for platform in platforms[:2]]
    return rate, fz[0], fz[1]


def choose_c3d(trial_dir: Path) -> Path | None:
    files = sorted(trial_dir.glob("*.c3d"))
    gap_filled = [path for path in files if path.stem.endswith("_gapf")]
    return (gap_filled or files or [None])[0]


def static_trial_dirs(manip_root: Path) -> list[Path]:
    return sorted(
        path for path in manip_root.glob("*/*")
        if path.is_dir() and path.name.casefold().startswith("static")
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manip_root", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="CSV ecrit")
    parser.add_argument("--subjects", nargs="+", help="limite a ces Subjects")
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for trial_dir in static_trial_dirs(args.manip_root):
            subject = trial_dir.parent.name
            if args.subjects and subject not in args.subjects:
                continue
            c3d = choose_c3d(trial_dir)
            if c3d is None:
                print(f"SANS C3D {subject}/{trial_dir.name}")
                continue
            rate, fz1, fz2 = read_plate_fz(c3d)
            relative = c3d.relative_to(args.manip_root).as_posix()
            for index, (a, b) in enumerate(zip(fz1, fz2)):
                writer.writerow((subject, trial_dir.name, relative,
                                 repr(round(index / rate, 6)), f"{a:.3f}", f"{b:.3f}"))
            print(f"OK {subject}/{trial_dir.name}: {len(fz1)} echantillons a {rate:g} Hz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
