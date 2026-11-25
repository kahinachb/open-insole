#!/usr/bin/env python3
"""Merge engine runs trained on disjoint targets into one 12-target run (per-target models).

Each input run is an engine run dir trained with ``--targets`` (one model per target, or the
Fz_L + Fz_R pair with ``--vertical-share``), with the same folds. For every fold present in
all of them, the predictions are joined on the sample (trial, time and states). A run only
writes the samples where one of its targets is supervised, so a sample missing from a run
gets that run's targets masked. The merged run has the engine's fold layout, so
``evaluate_grfm.py`` scores it as one model.

Usage::

    python merge_target_runs.py RUN_DIR [RUN_DIR ...] --output-dir OUT
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from evaluate_grfm import TARGETS

KEYS = ["test_batch", "test_group", "task_family", "time_s", "state_L", "state_R",
        "airborne_label_L", "airborne_label_R"]


def run_folds(run_dir: Path) -> dict[str, Path]:
    return {path.parent.name: path.parent for path in sorted(run_dir.rglob("predictions.csv"))}


def merge_fold(fold_dirs: list[Path]) -> pd.DataFrame:
    merged, owner = None, {}
    for fold_dir in fold_dirs:
        frame = pd.read_csv(fold_dir / "predictions.csv")
        targets = [t for t in TARGETS if f"pred_{t}" in frame]
        clash = sorted(set(targets) & set(owner))
        if clash:
            raise ValueError(f"{fold_dir}: targets {', '.join(clash)} already come from {owner[clash[0]]}")
        owner.update({target: fold_dir for target in targets})
        columns = KEYS + [c for t in targets for c in (t, f"pred_{t}", f"mask_{t}")]
        frame = frame[columns]
        merged = frame if merged is None else merged.merge(frame, on=KEYS, how="outer")
    missing = sorted(set(TARGETS) - set(owner))
    if missing:
        raise ValueError(f"no run predicts {', '.join(missing)}")
    for target in TARGETS:
        merged[f"mask_{target}"] = merged[f"mask_{target}"].fillna(0).astype(int)
    merged = merged.sort_values(["test_batch", "time_s"], kind="stable")
    return merged[KEYS + [c for t in TARGETS for c in (t, f"pred_{t}", f"mask_{t}")]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    folds = [run_folds(run_dir) for run_dir in args.run_dirs]
    common = sorted(set.intersection(*(set(f) for f in folds)))
    if not common:
        parser.error("no fold common to every run")
    for name in common:
        fold_dirs = [f[name] for f in folds]
        try:
            merged = merge_fold(fold_dirs)
        except ValueError as exc:
            parser.error(str(exc))
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out / "predictions.csv", index=False)
        shutil.copy(fold_dirs[0] / "split_manifest.json", out / "split_manifest.json")
        metadata = json.loads((fold_dirs[0] / "metadata.json").read_text(encoding="utf-8"))
        metadata["merged_from"] = [str(path.resolve()) for path in fold_dirs]
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{name}: {len(merged)} samples from {len(fold_dirs)} runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
