"""Architecture and recipe selection score over the validation and test Participants."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import score_selection
from synthetic_prepared import PLAIN_MLP_ONE_FOLD


@pytest.fixture(scope="module")
def scored(synthetic_prepared, trained_once, tmp_path_factory):
    run = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD)
    out = tmp_path_factory.mktemp("selection")
    assert score_selection.main([str(run), "--output-dir", str(out)]) == 0
    return run, out


def test_validation_predictions_cover_only_the_validation_participant(scored):
    _, out = scored
    (path,) = (out / "validation").rglob("predictions.csv")
    predictions = pd.read_csv(path)
    assert set(predictions["test_group"]) == {"T03"}
    assert {"pred_Fz_L", "mask_Fz_L"} <= set(predictions.columns)


def test_the_score_is_the_mean_over_participants_of_their_mean_nrmse(scored):
    _, out = scored
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    participants = selection["participants"]
    assert {name: entry["role"] for name, entry in participants.items()} == {"T01": "test", "T03": "validation"}
    per_participant = pd.read_csv(out / "evaluation" / "per_participant.csv")
    rows = per_participant[(per_participant["subset"] == "measured") & (per_participant["task_family"] == "ALL")]
    for name, entry in participants.items():
        expected = rows[rows["participant"] == name]["nrmse_range"].mean()
        assert entry["mean_nrmse"] == pytest.approx(expected)
        assert entry["targets"] == 12
    assert selection["selection_score"] == pytest.approx(
        sum(entry["mean_nrmse"] for entry in participants.values()) / 2
    )


def test_a_run_without_checkpoint_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        score_selection.main([str(tmp_path), "--output-dir", str(tmp_path / "out")])
