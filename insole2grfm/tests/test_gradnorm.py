"""GradNorm loss balancing through the training CLI on the synthetic dataset."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from synthetic_prepared import ONE_FOLD, PLAIN_MLP_ONE_FOLD, TARGETS, run_training_cli



def weights_table(out):
    return pd.read_csv(out / "gradnorm_weights.csv")


GRADNORM_3_EPOCHS = ("--gradnorm", "--epochs", "3", "--patience", "10")
MLP = ("--model-type", "mlp", "--layers", "2")


@pytest.mark.parametrize(
    "model_args",
    [MLP, ("--model-type", "transformer", "--layers", "1", "--transformer-heads", "2")],
    ids=["mlp", "transformer"],
)
def test_gradnorm_logs_per_epoch_weights_summing_to_twelve(synthetic_prepared, trained_once, model_args):
    out = trained_once(synthetic_prepared.data_dir, *model_args, *ONE_FOLD, *GRADNORM_3_EPOCHS)

    table = weights_table(out)
    assert list(table.columns) == ["test_group", "epoch", *TARGETS]
    assert set(table["test_group"]) == {"T01"}
    assert list(table["epoch"]) == [1, 2, 3]
    weights = table[TARGETS].to_numpy()
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.sum(axis=1), 12.0, rtol=1e-5)
    # The balancer actually moves the weights away from uniform.
    assert np.abs(weights - 1.0).max() > 1e-3
    assert (out / "predictions.csv").exists()


def predictions(out):
    return pd.read_csv(out / "predictions.csv")


def test_without_gradnorm_no_weights_are_logged(synthetic_prepared, trained_once):
    out = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD)

    assert not (out / "gradnorm_weights.csv").exists()
    history = json.loads((out / "training_history.json").read_text(encoding="utf-8"))
    assert all("gradnorm_weights" not in row for row in history["T01"])


def test_frozen_unit_weights_reproduce_the_plain_masked_loss(synthetic_prepared, trained_once, tmp_path):
    plain, frozen = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD), tmp_path / "frozen"
    run_training_cli(
        synthetic_prepared.data_dir, frozen, *PLAIN_MLP_ONE_FOLD, "--gradnorm", "--gradnorm-learning-rate", "0",
    )

    np.testing.assert_allclose(weights_table(frozen)[TARGETS].to_numpy(), 1.0)
    pred_columns = [f"pred_{target}" for target in TARGETS]
    np.testing.assert_allclose(
        predictions(frozen)[pred_columns].to_numpy(),
        predictions(plain)[pred_columns].to_numpy(),
        rtol=1e-4, atol=1e-4,
    )
    history = [
        json.loads((out / "training_history.json").read_text(encoding="utf-8"))["T01"]
        for out in (plain, frozen)
    ]
    np.testing.assert_allclose(
        [row["validation_loss"] for row in history[1]],
        [row["validation_loss"] for row in history[0]],
        rtol=1e-4,
    )


@pytest.mark.parametrize("model_type", ["lstm", "bilstm", "tcn"])
def test_every_architecture_trains_with_gradnorm(synthetic_prepared, tmp_path, model_type):
    out = tmp_path / "run"
    run_training_cli(
        synthetic_prepared.data_dir, out, "--model-type", model_type, "--layers", "1",
        *ONE_FOLD, "--gradnorm", "--epochs", "1",
    )
    weights = weights_table(out)[TARGETS].to_numpy()
    np.testing.assert_allclose(weights.sum(axis=1), 12.0, rtol=1e-5)
    assert np.abs(weights - 1.0).max() > 1e-3


def test_gradnorm_alpha_is_configurable(synthetic_prepared, trained_once):
    runs = {}
    # Default alpha 1.5 (the run of the weights test) against 0.5.
    for alpha, extra in (("1.5", ()), ("0.5", ("--gradnorm-alpha", "0.5"))):
        out = trained_once(synthetic_prepared.data_dir, *MLP, *ONE_FOLD, *GRADNORM_3_EPOCHS, *extra)
        metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["gradnorm"] is True
        assert metadata["gradnorm_alpha"] == float(alpha)
        runs[alpha] = weights_table(out)[TARGETS].to_numpy()
    assert np.abs(runs["1.5"] - runs["0.5"]).max() > 1e-4
