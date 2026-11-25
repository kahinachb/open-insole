#!/usr/bin/env python3
"""Test-set scores of a Sensor-dropout run under fixed Sensor configurations (#45).

For every fold checkpoint under ``RUN_DIR`` (``metadata.json`` + ``models/*.pt``,
e.g. the 15 folds of a Full CV) the test trials are predicted again, by
inference only, with each Sensor configuration: the groups left out are zeroed
and their presence columns set to 0, as in ``rank_sensor_configurations.py``.
Body weight is known. The ``full`` configuration is always evaluated and
reproduces the engine's own test predictions.

Configurations: ``--pareto PARETO_JSON`` (a ``rank_sensor_configurations.py``
summary) adds ``minimal`` and ``patches_<N>``, its best configuration for each
number of Patches; ``--configuration NAME="GROUP GROUP ..."`` adds any other.

``--without-sensor-dropout`` also accepts a run trained without Sensor dropout: its
removed groups read their training mean, the values a Sensor-dropout model is fed,
without the presence inputs it never had. It measures what Sensor dropout buys.

Outputs in ``OUT``: ``<configuration>/<fold>/predictions.csv`` and
``metadata.json``; ``evaluation/`` (``evaluate_grfm.py``, one run per
configuration); ``configurations.json``: per configuration its groups, each
test Participant's mean NRMSE of the 12 targets (Measured, all task families
pooled), and their mean and SD.

Usage::

    python evaluate_sensor_configurations.py RUN_DIR --output-dir OUT \\
        [--pareto PARETO_JSON] [--configuration NAME="GROUP ..."] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import evaluate_grfm
from score_selection import fold_dirs, fold_predictions, write_predictions


def pareto_configurations(path: Path) -> dict[str, list[str]]:
    """``minimal``, and ``patches_<N>`` when the summary has a best configuration per Patch count (ADR 0001)."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    named = {"minimal": summary["minimal"]["configuration"].split()}
    for count, entry in sorted(summary.get("best_per_patch_count", {}).items(), key=lambda item: int(item[0])):
        named[f"patches_{count}"] = entry["configuration"].split()
    return named


def parse_configuration(text: str) -> tuple[str, list[str]]:
    """``NAME=GROUP GROUP ...``."""
    name, _, groups = text.partition("=")
    if not name or not groups.split():
        raise argparse.ArgumentTypeError(f'attendu NOM="GROUPE GROUPE ...", recu {text!r}')
    return name, groups.split()


def evaluate(run_dir: Path, output_dir: Path, named: dict[str, list[str] | None], data_dir: Path | None,
             device: torch.device, batch_size: int, without_sensor_dropout: bool = False) -> dict:
    folds = fold_dirs(run_dir)
    if not folds:
        raise ValueError(f"{run_dir}: no fold dir (metadata.json + models/*.pt)")
    for name, groups in named.items():
        for fold_dir in folds:
            rows, fields, metadata = fold_predictions(fold_dir, data_dir, device, batch_size, "test", groups,
                                                      without_sensor_dropout)
            write_predictions(rows, fields, metadata, output_dir / name / fold_dir.name)
    evaluation = output_dir / "evaluation"
    runs = [argument for name in named for argument in ("--run", f"{name}={output_dir / name}")]
    if evaluate_grfm.main([*runs, "--output-dir", str(evaluation)]) != 0:
        raise ValueError("evaluate_grfm failed")
    per_participant = pd.read_csv(evaluation / "per_participant.csv")
    rows = per_participant[(per_participant["subset"] == "measured") & (per_participant["task_family"] == "ALL")]
    summary = {}
    for name, groups in named.items():
        scores = rows[rows["run"] == name].groupby("participant")["nrmse_range"].mean()
        summary[name] = {
            "groups": groups,
            "participants": {str(participant): float(value) for participant, value in scores.items()},
            "mean_nrmse": float(scores.mean()),
            "sd_nrmse": float(scores.std()) if len(scores) > 1 else float("nan"),
        }
    return {
        "run_dir": str(run_dir.resolve()),
        "metric": "per test Participant: mean NRMSE of the targets (Measured, all task families)",
        "configurations": summary,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="engine run dir (one or more fold dirs below it)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pareto", type=Path, help="rank_sensor_configurations.py pareto.json")
    parser.add_argument("--configuration", type=parse_configuration, action="append", default=[],
                        help='NAME="GROUP GROUP ...", repeatable')
    parser.add_argument("--data-dir", type=Path, help="prepared dataset (default: the run's data_dir)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--without-sensor-dropout", action="store_true",
                        help="accept a run trained without Sensor dropout (removed groups read their training mean)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    named: dict[str, list[str] | None] = {"full": None}
    try:
        if args.pareto:
            named.update(pareto_configurations(args.pareto))
        named.update(dict(args.configuration))
        result = evaluate(args.run_dir, args.output_dir, named, args.data_dir, torch.device(args.device),
                          args.batch_size, args.without_sensor_dropout)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    (args.output_dir / "configurations.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, entry in result["configurations"].items():
        print(f"{name}: mean NRMSE {entry['mean_nrmse']:.4f} (SD {entry['sd_nrmse']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
