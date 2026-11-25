#!/usr/bin/env python3
"""Train one Full CV fold of a queue plan, as one task of a Slurm job array (#45).

``overnight_queue.py`` runs every fold of a plan in turn under one lock; a job
array trains its folds in parallel instead, one engine run per task. This
script builds the same engine command as the queue for one fold
(``common_args``, then the model's ``engine_args``, then ``--only-test-group P``)
and runs it in the queue's results tree, ``<results-root>/<model>/<task_set>/fold_<P>/``,
so ``evaluate_grfm.py``, ``score_selection.py`` and
``evaluate_sensor_configurations.py`` read it as any queue run. The entry gate
is not repeated: the plan's model has passed it on the Fixed split.

The plan is a queue plan with ``"split": "full_cv"``, one model, one job, and
``"folds"``: the test Participants in array order. A fold already trained
(``metadata.json`` + checkpoint) is skipped, so a resubmitted array only runs
the missing folds; any other fold directory is an interrupted run and is wiped.

Usage::

    python full_cv_fold.py PLAN DATA_DIR --results-root R --fold-index $SLURM_ARRAY_TASK_ID [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from overnight_queue import ENGINE, fold_is_done, load_plan


def fold_command(plan: dict, data_dir: Path, results_root: Path, index: int, device: str) -> tuple[list[str], Path, dict]:
    """Engine command of fold ``index``, its output dir and the plan's model spec."""
    if plan.get("split") != "full_cv":
        raise ValueError(f'plan "split" must be "full_cv", not {plan.get("split")!r}')
    (job,) = plan["jobs"]
    spec = plan["models"][job["model"]]
    folds = plan["folds"]
    if not 0 <= index < len(folds):
        raise ValueError(f"fold index {index} outside 0..{len(folds) - 1}")
    participant = folds[index]
    fold_dir = results_root / job["model"] / job["task_set"] / f"fold_{participant}"
    command = [
        sys.executable, "-u", str(ENGINE), str(data_dir),
        "--output-dir", str(fold_dir),
        "--task-set", job["task_set"],
        "--model-type", spec["model_type"],
        *plan["common_args"],
        *spec.get("engine_args", []),
        "--device", device,
        "--only-test-group", participant,
    ]
    return command, fold_dir, spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("plan", type=Path)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--fold-index", type=int, required=True, help="index in the plan's folds ($SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        command, fold_dir, spec = fold_command(plan, args.data_dir, args.results_root, args.fold_index, args.device)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    participant = fold_dir.name.removeprefix("fold_")
    if fold_is_done(fold_dir, spec["model_type"], participant):
        print(f"{fold_dir}: already trained, skipped")
        return 0
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True)
    (fold_dir / "fold_command.json").write_text(json.dumps(command, indent=1), encoding="utf-8")
    print(" ".join(command), flush=True)
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
