#!/usr/bin/env python3
"""Evaluate a trained body-weight run with the body weight provided, missing and shifted (#28).

Reloads every fold checkpoint of a run directory written by
``train_lstm_two_feet_masked.py --body-weight-input`` (``metadata.json`` +
``models/*.pt``), rebuilds the inputs of its test trials from the prepared
dataset exactly as the engine does, and predicts them once per mode, on the
same split:

- ``provided``: the Subject's body weight as the run used it (checkpoint
  ``body_weights_N``), "body weight missing" flag 0. It reproduces the run's
  own ``predictions.csv``;
- ``missing``: the body weight replaced by the default value (the fold's
  training mean, standardised like the input) and the flag set to 1. A model
  trained without ``--body-weight-masking`` has no flag column: it only sees
  the default value, which measures how much it leans on the declared weight;
- ``provided+<k>kg`` / ``provided-<k>kg`` for each ``--shift-kg k``: the
  declared body weight shifted by ``±k * 9.81`` N, flag 0. Nothing but the
  body-weight column changes.

Outputs in ``--output-dir``:

- ``<mode>/predictions.csv``: the engine's columns plus ``body_weight_mode``,
  ``body_weight_shift_kg``, ``body_weight_input_N`` (the body weight fed to the
  model, N) and ``body_weight_missing`` (0/1); ``<mode>/metadata.json``: the
  run's metadata plus the mode, so ``evaluate_grfm.py --run <mode>=<dir>``
  finds the dataset (and its body weights for %BW);
- ``evaluation/``: ``evaluate_grfm.py`` on all modes side by side;
- ``modes_summary.csv``: per mode, the RMSE and mean error of the vertical
  forces (Measured ``Fz_L``, ``Fz_R``; both feet Measured ``Fz_sum``; all task
  families pooled, mean +/- SD over test Participants) and the change from
  ``provided``; ``body_weight_modes.md``: the same as a table.

Usage::

    python evaluate_body_weight_modes.py RUN_DIR --output-dir OUT [--shift-kg 1 2 5]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import evaluate_grfm
from grfm_schema import BODY_WEIGHT_FEATURE, BODY_WEIGHT_MISSING_FEATURE
from sensor_token_model import prepare_sensor_token_inputs
from train_lstm_two_feet_masked import (
    BodyWeightMode,
    MaskedWindowDataset,
    append_body_weight_input,
    append_body_weight_missing_flag,
    append_sensor_presence,
    load_acquisitions,
    model_from_args,
    predict,
    prediction_csv_fields,
    prediction_rows,
    prepare_input_features,
    restrict_to_sensor_configuration,
    split_group,
    subject_session_from_batch,
)

GRAVITY_M_S2 = 9.81
PROVIDED, MISSING = "provided", "missing"
MODE_FIELDS = ["body_weight_mode", "body_weight_shift_kg", "body_weight_input_N", "body_weight_missing"]
SUMMARY_TARGETS = (("measured", "Fz_L"), ("measured", "Fz_R"), (evaluate_grfm.SUM_SUBSET, "Fz_sum"))
SUMMARY_METRICS = ("rmse", "mean_error")


def shift_name(shift_kg: float) -> str:
    return f"{PROVIDED}{shift_kg:+g}kg"


def modes(shifts_kg: list[float]) -> list[tuple[str, bool, float]]:
    """(name, missing, shift in kg): provided, missing, then provided -k / +k for each k."""
    result = [(PROVIDED, False, 0.0), (MISSING, True, 0.0)]
    for shift in sorted({abs(value) for value in shifts_kg if value}):
        result += [(shift_name(-shift), False, -shift), (shift_name(shift), False, shift)]
    return result


def run_arguments(metadata: dict) -> argparse.Namespace:
    """The engine options the run was trained with (its ``metadata.json`` is ``vars(args)``)."""
    return argparse.Namespace(**metadata)


def test_inputs(run: argparse.Namespace, saved: dict, data_dir: Path) -> tuple[list, list[str], list[int], dict]:
    """The fold's test trials with the inputs the engine built for them (see ``fold_inputs``)."""
    return fold_inputs(run, saved, data_dir, saved["test_batches"])


def fold_inputs(
    run: argparse.Namespace, saved: dict, data_dir: Path, trial_names: list[str]
) -> tuple[list, list[str], list[int], dict]:
    """Trials ``trial_names`` of a fold with the inputs the engine built for them.

    Only the files of those trials' sessions are read (ADR 0001: ranking on the
    validation Participant never opens the test ones). The body weight, when
    the model has it, is the one the checkpoint records per Subject; a Sensor
    configuration keeps its channels; ``--sensor-dropout`` models get their
    presence columns. Raises ``ValueError`` if the rebuilt feature names differ
    from the checkpoint's.
    """
    sessions = sorted({subject_session_from_batch(name) for name in trial_names})
    loaded, source_metadata, targets, target_indexes = load_acquisitions(
        data_dir, tuple(saved["targets"]), sessions, saved["task_set"]
    )
    trials = [trial for trial in loaded if trial.name in set(trial_names)]
    source_names = list(source_metadata["feature_names"])
    if saved.get("sensor_configuration"):
        source_names = restrict_to_sensor_configuration(trials, source_names, saved["sensor_configuration"])
    names = prepare_input_features(
        trials,
        source_names,
        saved["input_features"],
        saved["gauge_proportion_epsilon_n"],
        saved["load_calibration_seconds"],
        estimate_load=saved["target_scaling"] == "subject_load",
        loadsol_heel_imu=saved.get("loadsol_heel_imu", False),
    )
    weights = {}
    if saved.get("body_weight_input"):
        declared = {
            subject: {"body_weight_N": weight} for subject, weight in (saved.get("body_weights_N") or {}).items()
        }
        names, weights = append_body_weight_input(trials, names, {"body_weights": {"subjects": declared}})
        if BODY_WEIGHT_MISSING_FEATURE in saved["feature_names"]:
            names = append_body_weight_missing_flag(trials, names)
    if saved.get("sensor_dropout"):
        names = append_sensor_presence(trials, names)
    if saved["model_type"] == "sensor_token":
        names = prepare_sensor_token_inputs(trials, names, targets)
    if names != list(saved["feature_names"]):
        raise ValueError("rebuilt input features differ from the checkpoint's feature_names")
    return trials, list(targets), target_indexes, weights


def load_model(run: argparse.Namespace, saved: dict, device: torch.device) -> torch.nn.Module:
    model = model_from_args(
        run, saved["input_size"], len(saved["targets"]), feature_names=list(saved["feature_names"]),
        targets=list(saved["targets"]),
    )
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval()


def evaluate_checkpoint(path: Path, run: argparse.Namespace, data_dir: Path, mode_list, device, batch_size: int):
    """Rows per mode name of one fold checkpoint, their columns, the fold, its default body weight (N)
    and whether the model has the "body weight missing" flag input."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if not saved.get("body_weight_input"):
        raise ValueError(f"{path.name}: trained without --body-weight-input")
    test, targets, target_indexes, weights = test_inputs(run, saved, data_dir)
    model = load_model(run, saved, device)
    x_mean, x_std = saved["x_mean"], saved["x_std"]
    column = list(saved["feature_names"]).index(BODY_WEIGHT_FEATURE)
    default_n = float(x_mean[column])
    rows: dict[str, list[dict]] = {name: [] for name, _, _ in mode_list}
    for name, missing, shift_kg in mode_list:
        body_weight = BodyWeightMode.from_statistics(
            saved["feature_names"], x_mean, x_std, missing=missing, shift_n=shift_kg * GRAVITY_M_S2
        )
        for item in test:
            dataset = MaskedWindowDataset(
                [item], x_mean, x_std, saved["y_mean"], saved["y_std"], target_indexes,
                saved["window_samples"], not saved["causal"], saved["zero_target_weight"],
                target_scaling=saved["target_scaling"],
                foot_height_mean=saved.get("foot_height_mean_mm"),
                foot_height_std=saved.get("foot_height_std_mm"),
                body_weight=body_weight,
            )
            if not len(dataset):
                continue
            outputs = predict(
                model, dataset, DataLoader(dataset, batch_size, shuffle=False), saved["y_mean"], saved["y_std"],
                device, target_indexes, saved["hard_airborne_zero"],
            )
            extra = {
                "body_weight_mode": name,
                "body_weight_shift_kg": shift_kg,
                "body_weight_input_N": default_n if missing else weights[item.subject_session] + shift_kg * GRAVITY_M_S2,
                "body_weight_missing": int(missing),
            }
            rows[name].extend(prediction_rows(item, split_group(item, saved["split_level"]), targets, outputs, extra))
    fields = prediction_csv_fields(
        targets, saved.get("foot_height_output", False), saved.get("vertical_share", False), MODE_FIELDS
    )
    flagged = BODY_WEIGHT_MISSING_FEATURE in saved["feature_names"]
    return rows, fields, saved["test_group"], default_n, flagged


def summary_table(summary: pd.DataFrame, mode_list) -> pd.DataFrame:
    """Vertical-force RMSE / mean error per mode, all task families pooled, and the change from provided."""
    pooled = summary[summary.task_family == evaluate_grfm.ALL_FAMILIES]
    records = []
    for subset, target in SUMMARY_TARGETS:
        for metric in SUMMARY_METRICS:
            selected = pooled[(pooled.subset == subset) & (pooled.target == target) & (pooled.metric == metric)]
            values = selected.set_index("run")
            if PROVIDED not in values.index:
                continue
            reference = float(values.loc[PROVIDED, "mean"])
            for name, missing, shift_kg in mode_list:
                if name not in values.index:
                    continue
                mean = float(values.loc[name, "mean"])
                records.append({
                    "mode": name, "body_weight_missing": int(missing), "shift_kg": shift_kg,
                    "subset": subset, "target": target, "metric": metric, "mean": mean,
                    "sd": float(values.loc[name, "sd"]), "n_participants": int(values.loc[name, "n_participants"]),
                    "delta_vs_provided": mean - reference,
                })
    return pd.DataFrame(records)


def table_lines(table: pd.DataFrame) -> list[str]:
    """Markdown table of a ``modes_summary.csv``: one row per mode (file order), Δ = mode − provided."""
    header = ["mode"] + [f"{target} {metric} (N)" for _, target in SUMMARY_TARGETS for metric in SUMMARY_METRICS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in table["mode"].drop_duplicates():
        cells = [name]
        for subset, target in SUMMARY_TARGETS:
            for metric in SUMMARY_METRICS:
                row = table[(table["mode"] == name) & (table.subset == subset) & (table.target == target)
                            & (table.metric == metric)]
                if row.empty:
                    cells.append("n/a")
                    continue
                row = row.iloc[0]
                cell = f"{row['mean']:.1f}" + ("" if math.isnan(row["sd"]) else f" ± {row['sd']:.1f}")
                cells.append(cell if name == PROVIDED else f"{cell} (Δ {row['delta_vs_provided']:+.1f})")
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def markdown(table: pd.DataFrame, run_dir: Path, flagged: bool) -> str:
    lines = [f"# Body weight provided / missing / shifted: `{run_dir}`", "",
             "Mean ± SD over test Participants, all task families pooled. Δ = mode − provided. "
             + ("The model has the \"body weight missing\" flag." if flagged else
                "The model was trained without masking (no flag): `missing` only feeds the default value."), ""]
    return "\n".join(lines + table_lines(table)) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="engine output dir (metadata.json, models/*.pt)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shift-kg", type=float, nargs="*", default=[1.0, 2.0, 5.0],
                        help="robustness sweep: provided body weight shifted by -k and +k kg for each k")
    parser.add_argument("--data-dir", type=Path, help="prepared dataset (default: the run's data_dir)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    metadata_path = args.run_dir / "metadata.json"
    checkpoints = sorted((args.run_dir / "models").glob("*.pt"))
    if not metadata_path.is_file() or not checkpoints:
        parser.error(f"{args.run_dir}: no metadata.json or models/*.pt")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("body_weight_input"):
        parser.error(f"{args.run_dir}: trained without --body-weight-input")
    run = run_arguments(metadata)
    data_dir = args.data_dir or Path(metadata["data_dir"])
    device = torch.device(args.device)
    mode_list = modes(args.shift_kg)

    rows: dict[str, list[dict]] = {name: [] for name, _, _ in mode_list}
    defaults, fields, flagged = {}, None, True
    try:
        for path in checkpoints:
            fold_rows, fields, fold, default_n, fold_flagged = evaluate_checkpoint(
                path, run, data_dir, mode_list, device, args.batch_size
            )
            defaults[fold] = default_n
            flagged = flagged and fold_flagged
            for name, values in fold_rows.items():
                rows[name].extend(values)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    for name, missing, shift_kg in mode_list:
        mode_dir = args.output_dir / name
        mode_dir.mkdir(parents=True, exist_ok=True)
        with (mode_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows[name])
        mode_metadata = {
            **metadata,
            "data_dir": str(data_dir.resolve()),
            "body_weight_evaluation": {
                "mode": name, "missing": missing, "shift_kg": shift_kg, "gravity_m_s2": GRAVITY_M_S2,
                "source_run": str(args.run_dir.resolve()), "default_body_weight_N": defaults,
                "missing_flag_input": flagged,
            },
        }
        (mode_dir / "metadata.json").write_text(json.dumps(mode_metadata, indent=2, ensure_ascii=False),
                                                encoding="utf-8")

    evaluation_dir = args.output_dir / "evaluation"
    evaluate_grfm.main([
        *(token for name, _, _ in mode_list for token in ("--run", f"{name}={args.output_dir / name}")),
        "--output-dir", str(evaluation_dir),
    ])
    table = summary_table(pd.read_csv(evaluation_dir / "summary.csv"), mode_list)
    table.to_csv(args.output_dir / "modes_summary.csv", index=False)
    (args.output_dir / "body_weight_modes.md").write_text(
        markdown(table, args.run_dir, flagged), encoding="utf-8"
    )
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
