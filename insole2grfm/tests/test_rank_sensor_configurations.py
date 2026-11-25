"""Sensor-configuration ranking of a Sensor-dropout checkpoint, with its Pareto summary (#42)."""

from __future__ import annotations

import json
import shutil

import pandas as pd
import pytest
import torch

import rank_sensor_configurations as ranking
from synthetic_prepared import PLAIN_MLP_ONE_FOLD, checkpoint

DROPOUT_FOLD = (*PLAIN_MLP_ONE_FOLD, "--sensor-dropout")


@pytest.fixture(scope="module")
def ranked(synthetic_prepared, tmp_path_factory, trained_once):
    run = trained_once(synthetic_prepared.data_dir, *DROPOUT_FOLD)
    out = tmp_path_factory.mktemp("ranking")
    assert ranking.main([str(run), "--output-dir", str(out), "--window-stride", "1",
                         "--reference", "loadsol-like=8:0.5"]) == 0
    return run, out


def test_every_non_empty_configuration_is_ranked_with_its_cost(ranked):
    _, out = ranked
    table = pd.read_csv(out / "configurations.csv")
    assert len(table) == 4095
    full = table[table["group_count"] == 12].iloc[0]
    assert full["patches"] == 4 and full["channels"] == 78
    heel_gauges = table[table["configuration"] == "heel_gauges"].iloc[0]
    assert heel_gauges["patches"] == 1 and heel_gauges["channels"] == 8
    assert {"mean_nrmse", "force_rmse_pct_bw", "physical_nmse"} <= set(table.columns)
    # The synthetic dataset declares no body weight: %BW is undefined, never a silent 0.
    assert table["force_rmse_pct_bw"].isna().all()


def test_the_full_configuration_reproduces_the_engine_validation_score(ranked):
    run, out = ranked
    table = pd.read_csv(out / "configurations.csv")
    full = table[table["group_count"] == 12].iloc[0]
    assert full["physical_nmse"] == pytest.approx(checkpoint(run)["validation_physical_nmse"], rel=1e-5)


def test_the_summary_and_figure_are_written(ranked):
    _, out = ranked
    summary = json.loads((out / "pareto.json").read_text(encoding="utf-8"))
    assert summary["validation_participants"] == ["T03"]
    assert "best_per_patch_count" not in summary
    assert (out / "pareto.png").exists()
    assert summary["reproduces_checkpoint_score"] is True
    assert summary["references"] == {"loadsol-like": {"channels": 8, "mean_nrmse": 0.5}}


def test_the_test_participant_is_never_read(synthetic_prepared, trained_once, tmp_path):
    run = trained_once(synthetic_prepared.data_dir, *DROPOUT_FOLD)
    data = tmp_path / "data"
    shutil.copytree(synthetic_prepared.data_dir, data, ignore=shutil.ignore_patterns("T01_*.npz"))
    assert ranking.main([str(run), "--output-dir", str(tmp_path / "out"), "--data-dir", str(data)]) == 0


def test_the_pareto_front_runs_on_channels_and_lower_nrmse_breaks_ties():
    table = pd.DataFrame([
        {"configuration": "eight", "patches": 1, "channels": 8, "mean_nrmse": 0.30},
        {"configuration": "fourteen_worse", "patches": 2, "channels": 14, "mean_nrmse": 0.21},
        {"configuration": "fourteen", "patches": 2, "channels": 14, "mean_nrmse": 0.20},
        {"configuration": "twenty_worse", "patches": 1, "channels": 20, "mean_nrmse": 0.25},
        {"configuration": "full", "patches": 4, "channels": 78, "mean_nrmse": 0.10},
    ])
    front = ranking.pareto_front(table)
    assert list(front["configuration"]) == ["eight", "fourteen", "full"]


def test_each_participant_is_scored_on_its_own_then_averaged():
    # T01: 2 samples, error 1, reference range 10. T03: 6 samples, error 4, range 10.
    truth = torch.tensor([[0.0], [10.0], [0.0], [10.0], [0.0], [10.0], [0.0], [10.0]])
    error = torch.tensor([[1.0], [-1.0], [4.0], [-4.0], [4.0], [-4.0], [4.0], [-4.0]])
    mask = torch.ones_like(truth)
    participant = torch.tensor([0, 0, 1, 1, 1, 1, 1, 1])
    nrmse = ranking.participant_nrmse(error, mask, truth, participant)
    assert nrmse.shape == (2, 1)
    assert nrmse[:, 0].tolist() == pytest.approx([0.1, 0.4], rel=0.02)  # per-Participant ranges 9.9


@pytest.mark.parametrize("text", ["loadsol=0.5", "=8:0.5", "loadsol=eight:0.5"])
def test_a_reference_needs_its_channel_count(text, tmp_path):
    with pytest.raises(SystemExit):
        ranking.main(["run", "--output-dir", str(tmp_path), "--reference", text])


def test_the_minimal_configuration_has_the_fewest_channels_within_ten_percent():
    table = pd.DataFrame([
        {"configuration": "full", "group_count": 12, "patches": 4, "channels": 78, "mean_nrmse": 0.100},
        {"configuration": "few_patches", "group_count": 3, "patches": 1, "channels": 20, "mean_nrmse": 0.105},
        {"configuration": "tie_worse", "group_count": 3, "patches": 3, "channels": 18, "mean_nrmse": 0.109},
        {"configuration": "tie_better", "group_count": 3, "patches": 2, "channels": 18, "mean_nrmse": 0.108},
        {"configuration": "outside", "group_count": 2, "patches": 2, "channels": 16, "mean_nrmse": 0.111},
        {"configuration": "tiny", "group_count": 1, "patches": 1, "channels": 6, "mean_nrmse": 0.250},
    ])
    summary = ranking.pareto_summary(table, tolerance=0.10)
    # Patches play no part: 18 channels beat 20, and at 18 the lower NRMSE wins; 16 is outside 0.110.
    assert summary["minimal"]["configuration"] == "tie_better"
    assert summary["full"]["configuration"] == "full"
    assert "best_per_patch_count" not in summary


def test_a_checkpoint_without_sensor_dropout_is_rejected(synthetic_prepared, trained_once, tmp_path):
    run = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD)
    with pytest.raises(SystemExit):
        ranking.main([str(run), "--output-dir", str(tmp_path / "out")])
