"""Foot height auxiliary output (#20): training CLI on the synthetic mini-dataset."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from grfm_schema import STATE_MASKED
from synthetic_prepared import ONE_FOLD, TARGETS, checkpoint, run_training_cli, write_synthetic_prepared

MODELS = {
    "mlp": ("--model-type", "mlp", "--layers", "2"),
    "transformer": ("--model-type", "transformer", "--layers", "1", "--transformer-heads", "2"),
    "sensor_token": (
        "--model-type", "sensor_token", "--layers", "1", "--transformer-heads", "2",
        "--transformer-feedforward-size", "16",
    ),
}
HEAD = ("--foot-height-output",)
TRAIN_SUBJECTS = {"T04", "T04b", "T05", "T05b"}
VALIDATION_SUBJECTS = {"T03", "T03b"}
FOOT_HEIGHT_COLUMNS = [
    f"{prefix}{name}_{side}"
    for side in ("L", "R")
    for prefix, name in (("", "foot_height_mm"), ("pred_", "foot_height_mm"), ("", "foot_height_valid"))
]


@pytest.fixture(scope="module")
def with_heights(tmp_path_factory):
    return write_synthetic_prepared(
        tmp_path_factory.mktemp("foot_height"), foot_height=True, walk_both_feet_masked=True
    )


def predictions(out):
    return pd.read_csv(out / "predictions.csv")


def history(out):
    return json.loads((out / "training_history.json").read_text(encoding="utf-8"))["T01"]


def metadata(out):
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def rewrite(source, destination, subjects, change):
    """Copy a prepared dataset, applying ``change(arrays)`` to the NPZ arrays of ``subjects``' trials."""
    shutil.copytree(source, destination)
    for path in destination.glob("*.npz"):
        if path.stem.split("_")[0] not in subjects:
            continue
        with np.load(path) as values:
            arrays = {key: values[key].copy() for key in values}
        change(arrays)
        np.savez(path, **arrays)
    return destination


def both_feet_masked(arrays):
    return (arrays["state_L"] == STATE_MASKED) & (arrays["state_R"] == STATE_MASKED)


@pytest.mark.parametrize("model_type", MODELS)
def test_each_model_type_trains_the_head_and_writes_foot_height_predictions(with_heights, trained_once, model_type):
    out = trained_once(with_heights.data_dir, *MODELS[model_type], *ONE_FOLD, *HEAD)
    table = predictions(out)
    assert set(FOOT_HEIGHT_COLUMNS) <= set(table.columns)
    for side in ("L", "R"):
        assert table[f"pred_foot_height_mm_{side}"].notna().all()
        valid = table[f"foot_height_valid_{side}"] == 1
        # The reference is in mm where valid and empty where not (marker dropout).
        assert table.loc[valid, f"foot_height_mm_{side}"].notna().all()
        assert table.loc[~valid, f"foot_height_mm_{side}"].isna().all()
        assert (~valid).any()
    rows = history(out)
    assert all(np.isfinite(row["train_foot_height_loss"]) for row in rows)
    assert all(np.isfinite(row["validation_foot_height_loss"]) for row in rows)


def test_masked_samples_get_foot_height_predictions(with_heights, trained_once):
    table = predictions(trained_once(with_heights.data_dir, *MODELS["mlp"], *ONE_FOLD, *HEAD))
    no_grfm = (table["state_L"] == STATE_MASKED) & (table["state_R"] == STATE_MASKED)
    assert no_grfm.any()
    rows = table[no_grfm]
    assert (rows[[f"mask_{target}" for target in TARGETS]] == 0).all().all()
    for side in ("L", "R"):
        assert rows[f"foot_height_valid_{side}"].eq(1).all()
        assert rows[f"pred_foot_height_mm_{side}"].notna().all()


def test_foot_height_loss_applies_on_masked_samples(with_heights, trained_once, tmp_path):
    # Foot height valid only where both feet are Masked, i.e. where no GRFM target exists.
    def masked_only(arrays):
        masked = both_feet_masked(arrays)
        for side in ("L", "R"):
            arrays[f"foot_height_valid_{side}"] = masked
            arrays[f"foot_height_mm_{side}"][~masked] = np.nan

    data = rewrite(with_heights.data_dir, tmp_path / "masked_only", set(with_heights.subjects), masked_only)
    out = trained_once(data, *MODELS["mlp"], *ONE_FOLD, *HEAD, "--epochs", "3")
    losses = [row["train_foot_height_loss"] for row in history(out)]
    assert all(loss > 0.0 for loss in losses)
    assert losses[-1] < losses[0]
    # The test Masked samples are predicted and scored too.
    table = predictions(out)
    no_grfm = (table["state_L"] == STATE_MASKED) & (table["state_R"] == STATE_MASKED)
    assert no_grfm.any()
    assert (table["foot_height_valid_L"] == 1).equals(no_grfm)


def test_grfm_loss_still_ignores_masked_samples_with_the_head_on(with_heights, trained_once, tmp_path):
    def other_grfm(arrays):
        arrays["y"][both_feet_masked(arrays)] = 1e4

    args = (*MODELS["mlp"], *ONE_FOLD, *HEAD)
    grfm = rewrite(with_heights.data_dir, tmp_path / "grfm", TRAIN_SUBJECTS, other_grfm)
    # The (meaningless) GRFM values of Masked samples never reach the loss.
    assert history(trained_once(grfm, *args)) == history(trained_once(with_heights.data_dir, *args))


def test_gradnorm_still_balances_exactly_the_twelve_grfm_targets(with_heights, trained_once):
    out = trained_once(with_heights.data_dir, *MODELS["mlp"], *ONE_FOLD, *HEAD, "--gradnorm", "--epochs", "3")
    table = pd.read_csv(out / "gradnorm_weights.csv")
    assert list(table.columns) == ["test_group", "epoch", *TARGETS]
    np.testing.assert_allclose(table[TARGETS].to_numpy().sum(axis=1), 12.0, rtol=1e-5)
    assert np.abs(table[TARGETS].to_numpy() - 1.0).max() > 1e-3
    rows = history(out)
    assert all(list(row["gradnorm_weights"]) == TARGETS for row in rows)
    assert all(np.isfinite(row["train_foot_height_loss"]) for row in rows)


def test_foot_height_stays_out_of_the_early_stopping_loss(with_heights, trained_once, tmp_path):
    # Only the validation Participant's Foot height differs: training and selection must not move.
    args = (*MODELS["mlp"], *ONE_FOLD, *HEAD, "--epochs", "5", "--patience", "1", "--learning-rate", "3e-2")

    def shifted(arrays):
        for side in ("L", "R"):
            arrays[f"foot_height_mm_{side}"] = arrays[f"foot_height_mm_{side}"] * 3.0 + 100.0

    validation = rewrite(with_heights.data_dir, tmp_path / "validation", VALIDATION_SUBJECTS, shifted)
    base_out, other_out = trained_once(with_heights.data_dir, *args), trained_once(validation, *args)
    base, other = history(base_out), history(other_out)
    assert [row["validation_foot_height_loss"] for row in base] != [
        row["validation_foot_height_loss"] for row in other
    ]
    for key in ("epoch", "train_loss", "validation_loss"):
        assert [row[key] for row in base] == [row[key] for row in other]
    assert predictions(base_out).equals(predictions(other_out))


def test_loss_weight_is_recorded_and_changes_training(with_heights, trained_once):
    base = trained_once(with_heights.data_dir, *MODELS["mlp"], *ONE_FOLD, *HEAD)
    heavy = trained_once(with_heights.data_dir, *MODELS["mlp"], *ONE_FOLD, *HEAD, "--foot-height-loss-weight", "5")
    assert metadata(base)["foot_height_output"] is True
    assert metadata(base)["foot_height_loss_weight"] == 1.0
    assert metadata(heavy)["foot_height_loss_weight"] == 5.0
    saved = checkpoint(heavy)
    assert saved["foot_height_output"] is True
    assert saved["foot_height_loss_weight"] == 5.0
    assert [row["train_loss"] for row in history(base)] != [row["train_loss"] for row in history(heavy)]


def test_foot_height_is_standardised_on_the_training_participants(with_heights, trained_once):
    saved = checkpoint(trained_once(with_heights.data_dir, *MODELS["mlp"], *ONE_FOLD, *HEAD))
    for column, side in enumerate(("L", "R")):
        values = np.concatenate([
            np.load(with_heights.data_dir / f"{batch}.npz")[f"foot_height_mm_{side}"]
            for batch in with_heights.batches
            if batch.split("_")[0] in TRAIN_SUBJECTS
        ])
        values = values[np.isfinite(values)]
        assert saved["foot_height_mean_mm"][column] == pytest.approx(values.mean(), rel=1e-5)
        assert saved["foot_height_std_mm"][column] == pytest.approx(values.std(), rel=1e-5)


@pytest.mark.parametrize("model_type", MODELS)
def test_with_the_head_off_behaviour_is_unchanged(with_heights, trained_once, tmp_path_factory, model_type):
    # The same dataset without the Foot height keys: head off trains exactly the same.
    plain = write_synthetic_prepared(tmp_path_factory.mktemp("plain"), walk_both_feet_masked=True)
    args = (*MODELS[model_type], *ONE_FOLD)
    off, reference = trained_once(with_heights.data_dir, *args), trained_once(plain.data_dir, *args)
    assert (off / "training_history.json").read_text() == (reference / "training_history.json").read_text()
    assert predictions(off).equals(predictions(reference))
    assert not any("foot_height" in column for column in predictions(off).columns)
    saved = checkpoint(off, model_type)
    assert saved["foot_height_output"] is False
    assert saved["foot_height_mean_mm"] is None and saved["foot_height_std_mm"] is None
    assert metadata(off)["foot_height_output"] is False


def test_the_head_needs_foot_height_in_the_dataset(synthetic_prepared, tmp_path):
    result = run_training_cli(
        synthetic_prepared.data_dir, tmp_path / "run", *MODELS["mlp"], *ONE_FOLD, *HEAD, check=False
    )
    assert result.returncode != 0
    assert "foot_height_mm_L" in result.stderr
    assert not (tmp_path / "run" / "metadata.json").exists()
