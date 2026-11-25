"""Overfit-test run mode of the training CLI on the synthetic dataset."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from synthetic_prepared import TARGETS, run_training_cli, write_synthetic_prepared

OVERFIT = (
    "--overfit-test", "--causal", "--split-level", "subject_identity",
    "--model-type", "mlp", "--layers", "2",
    "--overfit-windows-per-stratum", "16", "--overfit-epochs", "5",
)


def verdict(out):
    return json.loads((out / "overfit_verdict.json").read_text(encoding="utf-8"))


def test_overfit_mode_writes_curves_weights_and_verdict(synthetic_prepared, tmp_path):
    out = tmp_path / "overfit"
    run_training_cli(synthetic_prepared.data_dir, out, *OVERFIT, check=False)

    curves = pd.read_csv(out / "overfit_curves.csv")
    assert list(curves["epoch"]) == [0, 1, 2, 3, 4, 5]
    assert [f"nmse_{target}" for target in TARGETS] == [
        column for column in curves.columns if column.startswith("nmse_")
    ]
    weights = pd.read_csv(out / "gradnorm_weights.csv")
    assert list(weights.columns) == ["test_group", "epoch", *TARGETS]
    assert list(weights["epoch"]) == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(weights[TARGETS].sum(axis=1), 12.0, rtol=1e-5)

    result = verdict(out)
    assert result["passed"] in (True, False)
    assert set(result["final_nmse"]) == set(TARGETS)
    # 5 epochs with hidden 8 cannot fit the targets: a failing verdict that names why.
    assert result["passed"] is False
    assert result["cause"] == "nmse_above_threshold"
    # The verdict reads the median of the last (here all 5) trained epochs; final_nmse stays the last epoch.
    trained = curves[curves["epoch"] >= 1]
    np.testing.assert_allclose(
        [result["median_final_nmse"][target] for target in TARGETS],
        [trained[f"nmse_{target}"].median() for target in TARGETS],
    )
    np.testing.assert_allclose(
        [result["final_nmse"][target] for target in TARGETS],
        [curves[f"nmse_{target}"].iloc[-1] for target in TARGETS],
    )
    assert result["median_window_epochs"] == 5
    assert set(result["failing_targets"]) <= set(TARGETS) and result["failing_targets"]
    # Not trained as a regular run: no predictions or checkpoints.
    assert not (out / "predictions.csv").exists()


def test_sampled_set_is_stratified_and_identical_across_epochs(tmp_path):
    # Loaded subjects get longer trials: a proportional draw would favour them.
    data = write_synthetic_prepared(tmp_path / "data", samples=120, loaded_samples=300)
    out = tmp_path / "overfit"
    run_training_cli(
        data.data_dir, out, *OVERFIT, "--validation-group-count", "1", check=False
    )

    windows = pd.read_csv(out / "overfit_windows.csv")
    strata = windows.groupby(["task_family", "load_condition"]).size()
    families = ["balance", "hulahoop", "lateral_2supports", "squat"]
    conditions = ["unloaded", "loaded"]
    assert sorted(strata.index) == sorted((f, c) for f in families for c in conditions)
    assert set(strata) == {16}
    # First fold: test T01, validation T03 -> training participants T04 and T05.
    assert set(windows["subject"]) <= {"T04", "T04b", "T05", "T05b"}
    loaded_subjects = set(windows.loc[windows["load_condition"] == "loaded", "subject"])
    assert loaded_subjects <= {"T04b", "T05b"}
    assert not windows.duplicated(["batch", "target_index"]).any()

    curves = pd.read_csv(out / "overfit_curves.csv")
    assert set(curves["windows_seen"]) == {128}
    assert curves["windows_digest"].nunique() == 1
    result = verdict(out)
    assert result["windows"] == 128
    assert result["strata"] == {f"{f}/{c}": 16 for f in families for c in conditions}
    assert curves["windows_digest"].iloc[0] == result["windows_digest"]

    # The draw only depends on the data and the seed, not on the run length.
    longer = tmp_path / "longer"
    run_training_cli(
        data.data_dir, longer, *OVERFIT, "--validation-group-count", "1",
        "--overfit-epochs", "7", check=False,
    )
    pd.testing.assert_frame_equal(windows, pd.read_csv(longer / "overfit_windows.csv"))


# One entry per model type of the engine factory (add new types here):
# (model flags, windows per stratum, epochs, learning rate), the cheapest settings
# that memorise the fixed set with a margin (max NMSE about half the 0.05 threshold or less).
MODEL_TYPES = {
    "mlp": (("--model-type", "mlp", "--layers", "2", "--hidden-size", "64", "--dropout", "0.3"), 8, 60, 5e-3),
    "transformer": ((
        "--model-type", "transformer", "--layers", "1", "--hidden-size", "32",
        "--transformer-heads", "2", "--transformer-feedforward-size", "64", "--dropout", "0.3",
    ), 8, 80, 5e-3),
    "lstm": (("--model-type", "lstm", "--hidden-size", "32"), 16, 150, 3e-3),
    "bilstm": (("--model-type", "bilstm", "--hidden-size", "32"), 8, 120, 5e-3),
    "tcn": (("--model-type", "tcn", "--layers", "2", "--hidden-size", "32", "--dropout", "0.3"), 8, 100, 5e-3),
}


@pytest.mark.parametrize("settings", MODEL_TYPES.values(), ids=MODEL_TYPES.keys())
def test_every_model_type_passes_on_the_mini_dataset(synthetic_prepared, tmp_path, settings):
    model_args, windows, epochs, learning_rate = settings
    out = tmp_path / "overfit"
    result = run_training_cli(
        synthetic_prepared.data_dir, out, *OVERFIT, *model_args, "--overfit-windows-per-stratum", str(windows),
        "--overfit-epochs", str(epochs), "--learning-rate", str(learning_rate),
    )

    assert result.returncode == 0
    outcome = verdict(out)
    assert outcome["schema_version"] == 1
    assert outcome["passed"] is True
    assert outcome["cause"] is None
    assert outcome["failing_targets"] == []
    assert outcome["epochs_completed"] == epochs
    assert all(value < 0.05 for value in outcome["median_final_nmse"].values())
    assert sum(outcome["final_gradnorm_weights"].values()) == pytest.approx(12.0, rel=1e-5)


def test_sensor_token_model_passes_on_the_mirror_symmetric_mini_dataset(tmp_path):
    # The sensor-token model is mirror-consistent by construction and masks
    # Inactive gauges: it needs active gauges and mirror-symmetric targets.
    data = write_synthetic_prepared(tmp_path / "data", gauge_offset_n=40.0, mirror_symmetric=True)
    out = tmp_path / "overfit"
    result = run_training_cli(
        data.data_dir, out, *OVERFIT, "--model-type", "sensor_token", "--layers", "2",
        "--hidden-size", "16", "--transformer-heads", "4", "--transformer-feedforward-size", "32",
        "--overfit-windows-per-stratum", "8", "--overfit-epochs", "200", "--learning-rate", "1e-2", check=False,
    )

    outcome = verdict(out)
    assert result.returncode == 0, outcome["detail"]
    assert outcome["passed"] is True
    assert outcome["model_type"] == "sensor_token"
    assert all(value < 0.05 for value in outcome["median_final_nmse"].values())


def test_optimiser_overrides_apply_and_are_recorded(synthetic_prepared, tmp_path):
    out = tmp_path / "overfit"
    # 128 windows / batch 64 = 2 optimiser steps per epoch; 6 warmup steps.
    run_training_cli(
        synthetic_prepared.data_dir, out, *OVERFIT, "--learning-rate", "0.006",
        "--warmup-steps", "6", "--gradient-clip", "0.5", "--gradnorm-alpha", "0.5",
        check=False,
    )

    curves = pd.read_csv(out / "overfit_curves.csv").set_index("epoch")
    np.testing.assert_allclose(
        curves.loc[1:5, "learning_rate"], [0.001, 0.002, 0.004, 0.006, 0.006]
    )
    outcome = verdict(out)
    assert outcome["optimizer"]["learning_rate"] == 0.006
    assert outcome["optimizer"]["warmup_steps"] == 6
    assert outcome["optimizer"]["gradient_clip"] == 0.5
    assert outcome["optimizer"]["weight_decay"] == 0.0
    assert outcome["gradnorm"]["alpha"] == 0.5
    assert outcome["dropout"] == 0.0


def test_warmup_also_ramps_a_regular_training_run(synthetic_prepared, tmp_path):
    out = tmp_path / "run"
    # Fold T01: train = T04 + T05 = 24 trials x 100 windows -> 38 steps of 64 per epoch.
    run_training_cli(
        synthetic_prepared.data_dir, out, "--causal", "--split-level", "subject_identity",
        "--only-test-group", "T01", "--model-type", "mlp", "--epochs", "3",
        "--patience", "10", "--learning-rate", "0.0076", "--warmup-steps", "76",
    )

    history = json.loads((out / "training_history.json").read_text(encoding="utf-8"))
    np.testing.assert_allclose(
        [row["learning_rate"] for row in history["T01"]], [0.0001, 0.0038, 0.0076]
    )


def test_missing_stratum_still_writes_a_failing_verdict(synthetic_prepared, tmp_path):
    out = tmp_path / "overfit"
    # Unloaded Subjects only: the four loaded strata are empty.
    result = run_training_cli(
        synthetic_prepared.data_dir, out, *OVERFIT, "--subjects", "T01", "T03", "T04", "T05",
        check=False,
    )

    assert result.returncode == 3
    outcome = verdict(out)
    assert outcome["cause"] == "missing_strata"
    assert outcome["strata"]["squat/loaded"] == 0
    assert outcome["strata"]["squat/unloaded"] == 16


@pytest.mark.parametrize(
    ("learning_rate", "cause"),
    [("inf", "nan"), ("100", "divergence")],
    ids=["nan", "divergence"],
)
def test_nan_or_divergence_fails_and_names_the_cause(synthetic_prepared, tmp_path, learning_rate, cause):
    out = tmp_path / "overfit"
    result = run_training_cli(
        synthetic_prepared.data_dir, out, *OVERFIT, "--overfit-epochs", "20",
        "--learning-rate", learning_rate, "--gradient-clip", "0", check=False,
    )

    assert result.returncode == 3
    outcome = verdict(out)
    assert outcome["passed"] is False
    assert outcome["cause"] == cause
    assert outcome["detail"]
    # The run stops at the faulty epoch instead of training on.
    assert outcome["stopped_epoch"] == outcome["epochs_completed"] < 20
    curves = pd.read_csv(out / "overfit_curves.csv")
    assert curves["epoch"].iloc[-1] == outcome["stopped_epoch"]


def test_a_single_diverging_target_fails_the_test_and_is_named(synthetic_prepared, tmp_path):
    out = tmp_path / "overfit"
    # lr 3 without clipping: after epoch 1 some targets exceed 6 x their untrained NMSE
    # while the mean over the 12 targets stays below 6 x the untrained mean.
    result = run_training_cli(
        synthetic_prepared.data_dir, out, *OVERFIT, "--overfit-epochs", "20", "--learning-rate", "3",
        "--gradient-clip", "0", "--overfit-divergence-factor", "6", check=False,
    )

    assert result.returncode == 3
    outcome = verdict(out)
    assert outcome["cause"] == "divergence"
    curves = pd.read_csv(out / "overfit_curves.csv").set_index("epoch")
    stopped = outcome["stopped_epoch"]
    assert curves.loc[stopped, "mean_nmse"] < 6 * curves.loc[0, "mean_nmse"]  # the mean alone would not stop
    ratio = {t: curves.loc[stopped, f"nmse_{t}"] / curves.loc[0, f"nmse_{t}"] for t in TARGETS}
    diverging = [t for t in TARGETS if ratio[t] > 6]
    assert diverging and outcome["failing_targets"] == diverging
    assert all(target in outcome["detail"] for target in diverging)


def test_the_verdict_reads_the_median_of_the_last_epochs_so_one_spike_does_not_fail():
    import train_lstm_two_feet_masked as engine

    history = [np.array([0.01, 0.002])] * 9 + [np.array([0.08, 0.002])]  # spike on the last epoch only
    assert engine.final_overfit_nmse(history, window=10).tolist() == pytest.approx([0.01, 0.002])
    persistent = [np.array([0.08, 0.002])] * 6 + [np.array([0.01, 0.002])] * 4
    assert engine.final_overfit_nmse(persistent, window=10)[0] == pytest.approx(0.08)
    assert engine.final_overfit_nmse(history[:3], window=10).tolist() == pytest.approx([0.01, 0.002])
