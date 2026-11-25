"""Measured-only model selection (#18): early stopping and the LR scheduler on the Measured subset."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from synthetic_prepared import PLAIN_MLP_ONE_FOLD, run_training_cli

# One fold, with a learning rate high enough for the Valid-subset and Measured-subset
# validation losses to reach their minimum at different epochs (the synthetic Airborne
# zeros cannot be predicted from the inputs), and a patience above the epoch count so
# early stopping never cuts the comparison short.
SELECTION_FOLD = (
    *PLAIN_MLP_ONE_FOLD,
    "--epochs", "6", "--patience", "20", "--learning-rate", "1e-1", "--zero-target-weight", "1.0",
)


def history(out):
    return json.loads((out / "training_history.json").read_text(encoding="utf-8"))["T01"]


def selected_epoch(rows):
    # The engine marks the epoch whose weights it kept with the comparable validation score.
    (row,) = [row for row in rows if "validation_physical_nmse" in row]
    return row["epoch"]


def metadata(out):
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def test_measured_selection_keeps_the_epoch_of_the_measured_validation_loss(synthetic_prepared, trained_once):
    valid = trained_once(synthetic_prepared.data_dir, *SELECTION_FOLD)
    measured = trained_once(synthetic_prepared.data_dir, *SELECTION_FOLD, "--selection-subset", "measured")
    valid_rows, measured_rows = history(valid), history(measured)

    measured_losses = [row["validation_measured_loss"] for row in measured_rows]
    mixed_losses = [row["validation_loss"] for row in measured_rows]
    best_measured = int(np.argmin(measured_losses)) + 1
    best_mixed = int(np.argmin(mixed_losses)) + 1
    assert best_measured != best_mixed  # the two selections disagree on this run
    assert selected_epoch(measured_rows) == best_measured
    assert selected_epoch(valid_rows) == best_mixed
    # The kept weights are those of the selected epoch, so the test predictions differ.
    predictions = [pd.read_csv(out / "predictions.csv")["pred_Fz_L"] for out in (valid, measured)]
    assert not np.allclose(predictions[0], predictions[1])


def test_measured_selection_leaves_training_unchanged(synthetic_prepared, trained_once):
    # Airborne samples stay in the training loss: the same seed trains exactly the same
    # epochs (same training and Valid-subset validation losses); only the selection differs.
    valid_rows = history(trained_once(synthetic_prepared.data_dir, *SELECTION_FOLD))
    measured_rows = history(trained_once(synthetic_prepared.data_dir, *SELECTION_FOLD, "--selection-subset", "measured"))
    assert [row["train_loss"] for row in measured_rows] == [row["train_loss"] for row in valid_rows]
    assert [row["validation_loss"] for row in measured_rows] == [row["validation_loss"] for row in valid_rows]
    # The Measured-subset loss leaves the Airborne samples of the validation jump trials out.
    assert all(row["validation_measured_loss"] != row["validation_loss"] for row in measured_rows)


def test_selection_subset_is_off_by_default_and_recorded(synthetic_prepared, trained_once):
    valid = trained_once(synthetic_prepared.data_dir, *SELECTION_FOLD)
    measured = trained_once(synthetic_prepared.data_dir, *SELECTION_FOLD, "--selection-subset", "measured")
    assert metadata(valid)["selection_subset"] == "valid"
    assert metadata(measured)["selection_subset"] == "measured"
    # Off by default, the history keeps its spec #1 fields only.
    assert all("validation_measured_loss" not in row for row in history(valid))


def test_measured_selection_drives_early_stopping_and_the_scheduler(synthetic_prepared, tmp_path):
    args = (*SELECTION_FOLD, "--selection-subset", "measured", "--lr-scheduler", "plateau",
            "--lr-patience", "1", "--patience", "3")
    out = tmp_path / "run"
    run_training_cli(synthetic_prepared.data_dir, out, *args)
    rows = history(out)
    losses = [row["validation_measured_loss"] for row in rows]
    rates = [row["learning_rate"] for row in rows]
    # Replays ReduceLROnPlateau (patience 1, factor 0.5) and early stopping (patience 3)
    # on the Measured-subset losses: the recorded learning rates and the last epoch follow.
    best, stale, plateau_best, plateau_stale, rate = np.inf, 0, np.inf, 0, 1e-1
    expected_rates, last_epoch = [], None
    for epoch, loss in enumerate(losses, start=1):
        expected_rates.append(rate)
        if loss < best - 1e-6:
            best, stale = loss, 0
        else:
            stale += 1
            if stale >= 3:
                last_epoch = epoch
                break
        if loss < plateau_best * (1 - 1e-4):
            plateau_best, plateau_stale = loss, 0
        else:
            plateau_stale += 1
            if plateau_stale > 1:
                rate, plateau_stale = max(rate * 0.5, 1e-6), 0
    assert rates == expected_rates
    assert len(rows) == (last_epoch or 6)


def test_selection_subset_changes_the_queue_config_signature(synthetic_prepared, tmp_path):
    from test_overnight_queue import run_queue, write_plan

    plan = write_plan(
        tmp_path / "plan.json",
        models={
            "mlp": {"model_type": "mlp", "engine_args": ["--layers", "1"], "baseline": True},
            "mlp_measured": {
                "model_type": "mlp", "engine_args": ["--layers", "1", "--selection-subset", "measured"],
                "baseline": True,
            },
        },
        jobs=[
            {"name": "mlp_all", "model": "mlp", "task_set": "all"},
            {"name": "mlp_measured_all", "model": "mlp_measured", "task_set": "all"},
        ],
    )
    root = tmp_path / "results"
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--folds", "T01")

    configs = {
        model: json.loads((root / model / "all" / "queue_config.json").read_text(encoding="utf-8"))
        for model in ("mlp", "mlp_measured")
    }
    assert configs["mlp"]["config_signature"] != configs["mlp_measured"]["config_signature"]
    folds = {model: metadata(root / model / "all" / "fold_T01") for model in configs}
    assert folds["mlp"]["selection_subset"] == "valid"
    assert folds["mlp_measured"]["selection_subset"] == "measured"
