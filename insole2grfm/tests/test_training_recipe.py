"""Training recipe (#30): cosine schedule with warmup, weight EMA, anomalous-batch guard."""

from __future__ import annotations

import json
import math

import pytest
import torch
from torch import nn

import train_lstm_two_feet_masked as engine
from synthetic_prepared import PLAIN_MLP_ONE_FOLD, checkpoint, run_training_cli

RECIPE = (
    "--lr-scheduler", "cosine", "--warmup-fraction", "0.1", "--min-learning-rate", "1e-5",
    "--ema-decay", "0.99", "--anomalous-loss-threshold", "100",
)
RECIPE_FOLD = (*PLAIN_MLP_ONE_FOLD, "--epochs", "4", "--patience", "20", *RECIPE)


def history(out):
    return json.loads((out / "training_history.json").read_text(encoding="utf-8"))["T01"]


def rates_over(steps: int, warmup: int, total: int, base: float = 1e-2, minimum: float = 1e-4) -> list[float]:
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=base)
    engine.add_cosine_schedule(optimizer, warmup, total, minimum)
    rates = []
    for _ in range(steps):
        optimizer.step()
        rates.append(optimizer.param_groups[0]["lr"])
    return rates


def test_cosine_schedule_warms_up_then_decays_to_the_minimum_and_stays_there():
    rates = rates_over(steps=30, warmup=5, total=20)
    assert rates[:5] == pytest.approx([1e-2 * (k + 1) / 5 for k in range(5)])
    assert rates[5] == pytest.approx(1e-2)
    decay = rates[5:20]
    assert all(later < earlier for earlier, later in zip(decay, decay[1:]))
    assert rates[12] == pytest.approx(1e-4 + (1e-2 - 1e-4) * 0.5 * (1 + math.cos(math.pi * 7 / 15)))
    assert rates[20:] == pytest.approx([1e-4] * 10)


def test_weight_average_warms_its_decay_up_and_applied_restores_the_raw_weights():
    model = nn.Linear(1, 1, bias=False)
    nn.init.zeros_(model.weight)
    average = engine.WeightAverage(model, max_decay=0.99)
    with torch.no_grad():
        model.weight.fill_(1.0)
    average.update()  # decay = min(0.99, 1/10) = 0.1: the average moves 90 % of the way
    assert float(average.shadow[0]) == pytest.approx(0.9)
    with average.applied():
        assert float(model.weight) == pytest.approx(0.9)
    assert float(model.weight) == pytest.approx(1.0)


def test_guard_rejects_non_finite_and_too_large_losses_only():
    guard = engine.AnomalousBatchGuard(10.0)
    assert not guard.rejects(torch.tensor(3.0))
    assert not guard.rejects(None)
    assert guard.rejects(torch.tensor(float("nan")))
    assert guard.rejects(torch.tensor(float("inf")))
    assert guard.rejects(torch.tensor(11.0))
    assert guard.skipped == 3


def test_recipe_run_records_its_schedule_selection_and_skips(synthetic_prepared, trained_once):
    out = trained_once(synthetic_prepared.data_dir, *RECIPE_FOLD)
    rows = history(out)
    rates = [row["learning_rate"] for row in rows]
    assert rates[-1] < max(rates)
    for row in rows:
        assert row["selected_weights"] in {"raw", "ema"}
        assert row["selection_loss"] == min(row["validation_loss"], row["validation_ema_loss"])
        assert row["skipped_anomalous_batches"] == 0
    saved = checkpoint(out)
    assert saved["ema_decay"] == 0.99
    (best,) = [row for row in rows if "validation_physical_nmse" in row]
    assert saved["selected_weights"] == best["selected_weights"]


def test_recipe_works_with_gradnorm(synthetic_prepared, trained_once):
    out = trained_once(synthetic_prepared.data_dir, *RECIPE_FOLD, "--gradnorm")
    assert all(row["selected_weights"] in {"raw", "ema"} for row in history(out))


def test_a_tiny_threshold_skips_every_training_batch(synthetic_prepared, trained_once):
    for extra in ((), ("--gradnorm",)):
        out = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, "--anomalous-loss-threshold", "1e-12", *extra)
        assert all(row["skipped_anomalous_batches"] > 0 for row in history(out))


@pytest.mark.parametrize("arguments", [
    ("--warmup-fraction", "0.1"),
    ("--lr-scheduler", "cosine", "--warmup-fraction", "0.1", "--warmup-steps", "5"),
    ("--lr-scheduler", "cosine", "--min-learning-rate", "1", "--learning-rate", "1e-3"),
    ("--ema-decay", "1.0"),
    ("--anomalous-loss-threshold", "-1"),
])
def test_invalid_recipe_arguments_are_rejected(synthetic_prepared, tmp_path, arguments):
    result = run_training_cli(synthetic_prepared.data_dir, tmp_path / "run", *PLAIN_MLP_ONE_FOLD, *arguments, check=False)
    assert result.returncode == 2


def test_weight_decay_is_configurable_and_recorded(synthetic_prepared, trained_once):
    default = checkpoint(trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD))
    assert default["weight_decay"] == 1e-4
    stronger = checkpoint(trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, "--weight-decay", "0.05"))
    assert stronger["weight_decay"] == 0.05
    assert not torch.equal(default["state_dict"]["network.1.weight"], stronger["state_dict"]["network.1.weight"])


def test_negative_weight_decay_is_rejected(synthetic_prepared, tmp_path):
    result = run_training_cli(synthetic_prepared.data_dir, tmp_path / "run", *PLAIN_MLP_ONE_FOLD, "--weight-decay", "-1", check=False)
    assert result.returncode == 2
