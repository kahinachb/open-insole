"""Test-set scores of a Sensor-dropout run under fixed Sensor configurations (Full CV, #45)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import evaluate_sensor_configurations as configurations
from score_selection import fold_dirs
from synthetic_prepared import PLAIN_MLP_ONE_FOLD

DROPOUT_FOLD = (*PLAIN_MLP_ONE_FOLD, "--sensor-dropout")
MINIMAL = "heel_gauges forefoot_gauges forefoot_acc"


@pytest.fixture(scope="module")
def evaluated(synthetic_prepared, trained_once, tmp_path_factory):
    run = trained_once(synthetic_prepared.data_dir, *DROPOUT_FOLD)
    out = tmp_path_factory.mktemp("configurations")
    pareto = out / "pareto.json"
    pareto.write_text(json.dumps({
        "minimal": {"configuration": MINIMAL},
        "best_per_patch_count": {"1": {"configuration": "forefoot_gauges forefoot_acc forefoot_gyro"}},
    }), encoding="utf-8")
    assert configurations.main([str(run), "--output-dir", str(out), "--pareto", str(pareto),
                                "--configuration", "heel=heel_gauges heel_acc heel_gyro"]) == 0
    return run, out


def test_the_full_configuration_reproduces_the_engine_test_predictions(evaluated):
    run, out = evaluated
    (fold,) = fold_dirs(run)
    engine = pd.read_csv(fold / "predictions.csv")
    rebuilt = pd.read_csv(out / "full" / fold.name / "predictions.csv")
    assert set(rebuilt["test_group"]) == {"T01"}
    predictions = [column for column in engine.columns if column.startswith("pred_")]
    assert np.allclose(rebuilt[predictions].to_numpy(), engine[predictions].to_numpy(), atol=1e-4, equal_nan=True)


def test_every_configuration_is_scored_per_participant(evaluated):
    _, out = evaluated
    summary = json.loads((out / "configurations.json").read_text(encoding="utf-8"))["configurations"]
    assert list(summary) == ["full", "minimal", "patches_1", "heel"]
    assert summary["minimal"]["groups"] == MINIMAL.split()
    per_participant = pd.read_csv(out / "evaluation" / "per_participant.csv")
    rows = per_participant[(per_participant["subset"] == "measured") & (per_participant["task_family"] == "ALL")]
    for name, entry in summary.items():
        expected = rows[(rows["run"] == name) & (rows["participant"] == "T01")]["nrmse_range"].mean()
        assert entry["participants"] == {"T01": pytest.approx(expected)}
        assert entry["mean_nrmse"] == pytest.approx(expected)
    assert summary["minimal"]["mean_nrmse"] != summary["full"]["mean_nrmse"]


def test_a_run_without_sensor_dropout_only_takes_the_full_configuration(synthetic_prepared, trained_once, tmp_path):
    run = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD)
    with pytest.raises(SystemExit):
        configurations.main([str(run), "--output-dir", str(tmp_path), "--configuration", f"minimal={MINIMAL}"])


def test_a_run_without_sensor_dropout_is_scored_with_mean_filled_groups_on_request(synthetic_prepared, trained_once, tmp_path):
    run = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD)
    assert configurations.main([str(run), "--output-dir", str(tmp_path), "--configuration", f"minimal={MINIMAL}",
                                "--without-sensor-dropout"]) == 0
    summary = json.loads((tmp_path / "configurations.json").read_text(encoding="utf-8"))["configurations"]
    assert summary["minimal"]["mean_nrmse"] != summary["full"]["mean_nrmse"]
    # The full configuration reproduces the engine's own test predictions.
    full = pd.read_csv(tmp_path / "full" / next((tmp_path / "full").iterdir()).name / "predictions.csv")
    engine = pd.read_csv(next(run.rglob("predictions.csv")))
    assert np.allclose(full["pred_Fz_L"], engine["pred_Fz_L"], atol=1e-4)


def test_an_unknown_group_is_rejected(synthetic_prepared, trained_once, tmp_path):
    run = trained_once(synthetic_prepared.data_dir, *DROPOUT_FOLD)
    with pytest.raises(SystemExit):
        configurations.main([str(run), "--output-dir", str(tmp_path), "--configuration", "bad=heel_gauges ankle_acc"])
