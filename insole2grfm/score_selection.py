#!/usr/bin/env python3
"""Selection score of a Fixed-split run over its validation and test Participants.

Architecture and recipe choices compare runs on this score (owner's decision,
2026-10-26): the mean, over the validation Participant (P07) and the test
Participants (P04, P13), of each Participant's mean NRMSE of the 12 targets
(Measured samples, all task families pooled; NRMSE = RMSE / (p99.5 - p0.5) of
the reference, as in ``evaluate_grfm.py``). Published numbers still come from
the Full CV.

For every fold checkpoint under ``RUN_DIR`` (``metadata.json`` + ``models/*.pt``)
the validation trials are predicted as the engine predicts test trials: body
weight known, all sensors present. Their ``predictions.csv`` go to
``OUT/validation/<fold>/``; ``evaluate_grfm.py`` then scores them together with
the run's own test predictions in ``OUT/evaluation/``.

``OUT/selection.json``: per Participant its role and mean NRMSE, and the
selection score.

Usage::

    python score_selection.py RUN_DIR --output-dir OUT [--device cuda]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import evaluate_grfm
from evaluate_body_weight_modes import fold_inputs, load_model, run_arguments
from train_lstm_two_feet_masked import (
    MaskedWindowDataset,
    SensorDropout,
    predict,
    prediction_csv_fields,
    prediction_rows,
    split_group,
)


def fold_dirs(run_dir: Path) -> list[Path]:
    """Engine output dirs of one fold each: a ``metadata.json`` next to ``models/*.pt``."""
    return sorted(
        path.parent for path in run_dir.rglob("metadata.json") if any((path.parent / "models").glob("*.pt"))
    )


def fold_predictions(
    fold_dir: Path, data_dir: Path | None, device: torch.device, batch_size: int,
    split: str = "validation", configuration: list[str] | None = None, without_sensor_dropout: bool = False,
):
    """``predictions.csv`` rows and columns of the fold's ``split`` trials (body weight known).

    A Sensor-dropout checkpoint is fed ``configuration`` (Sensor group names; by
    default the full configuration). A checkpoint without it only takes the full one,
    unless ``without_sensor_dropout``: the removed groups then read their training mean
    (0 once standardised, as a Sensor-dropout model is fed), with no presence input.
    """
    metadata = json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))
    (checkpoint_path,) = sorted((fold_dir / "models").glob("*.pt"))
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    run = run_arguments(metadata)
    trials, targets, target_indexes, _ = fold_inputs(
        run, saved, data_dir or Path(metadata["data_dir"]), saved[f"{split}_batches"]
    )
    if saved.get("sensor_dropout"):
        sensors = SensorDropout.from_feature_names(saved["feature_names"], configuration)
    elif configuration is not None and without_sensor_dropout:
        sensors = SensorDropout.from_feature_names(saved["feature_names"], configuration, presence=False)
    elif configuration is not None:
        raise ValueError(f"{checkpoint_path}: trained without Sensor dropout, only the full configuration")
    else:
        sensors = None
    model = load_model(run, saved, device)
    rows = []
    for trial in trials:
        dataset = MaskedWindowDataset(
            [trial], saved["x_mean"], saved["x_std"], saved["y_mean"], saved["y_std"], target_indexes,
            saved["window_samples"], not saved["causal"], saved["zero_target_weight"],
            target_scaling=saved["target_scaling"],
            foot_height_mean=saved.get("foot_height_mean_mm"),
            foot_height_std=saved.get("foot_height_std_mm"),
            sensor_dropout=sensors,
        )
        if not len(dataset):
            continue
        outputs = predict(
            model, dataset, DataLoader(dataset, batch_size, shuffle=False), saved["y_mean"], saved["y_std"],
            device, target_indexes, saved["hard_airborne_zero"],
        )
        rows.extend(prediction_rows(trial, split_group(trial, saved["split_level"]), targets, outputs))
    fields = prediction_csv_fields(targets, saved.get("foot_height_output", False), saved.get("vertical_share", False))
    return rows, fields, metadata


def participant_scores(per_participant: pd.DataFrame, roles: dict[str, str]) -> dict[str, dict]:
    """Mean NRMSE of the targets per Participant (Measured, all task families pooled)."""
    rows = per_participant[(per_participant["subset"] == "measured") & (per_participant["task_family"] == "ALL")]
    scores = {}
    for name, role in sorted(roles.items()):
        values = rows[rows["participant"] == name]["nrmse_range"]
        scores[name] = {"role": role, "mean_nrmse": float(values.mean()), "targets": int(values.count())}
    return scores


def write_predictions(rows: list[dict], fields: list[str], metadata: dict, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    with (target / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (target / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def score(run_dir: Path, output_dir: Path, data_dir: Path | None, device: torch.device, batch_size: int) -> dict:
    folds = fold_dirs(run_dir)
    if not folds:
        raise ValueError(f"{run_dir}: no fold dir (metadata.json + models/*.pt)")
    roles = {}
    for fold_dir in folds:
        rows, fields, metadata = fold_predictions(fold_dir, data_dir, device, batch_size)
        write_predictions(rows, fields, metadata, output_dir / "validation" / fold_dir.name)
        roles.update({row["test_group"]: "validation" for row in rows})
        tested = pd.read_csv(fold_dir / "predictions.csv", usecols=["test_group"])["test_group"]
        roles.update({str(name): "test" for name in tested.unique()})
    evaluation = output_dir / "evaluation"
    if evaluate_grfm.main([
        "--run", f"test={run_dir}", "--run", f"validation={output_dir / 'validation'}", "--output-dir", str(evaluation),
    ]) != 0:
        raise ValueError("evaluate_grfm failed")
    participants = participant_scores(pd.read_csv(evaluation / "per_participant.csv"), roles)
    return {
        "run_dir": str(run_dir.resolve()),
        "metric": "mean over Participants of the mean NRMSE of the targets (Measured, all task families)",
        "participants": participants,
        "selection_score": sum(entry["mean_nrmse"] for entry in participants.values()) / len(participants),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="engine run dir (one or more fold dirs below it)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="prepared dataset (default: the run's data_dir)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        selection = score(args.run_dir, args.output_dir, args.data_dir, torch.device(args.device), args.batch_size)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    (args.output_dir / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    for name, entry in selection["participants"].items():
        print(f"{name} ({entry['role']}): mean NRMSE {entry['mean_nrmse']:.4f}")
    print(f"selection score: {selection['selection_score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
