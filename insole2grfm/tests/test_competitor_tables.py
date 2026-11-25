"""Competitor comparison and protocol tables of evaluate_grfm.py (#39)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from test_evaluate_grfm import evaluate, trial, wave, write_run


def comparison(out):
    return pd.read_csv(out / "competitor_comparison.csv")


def test_walk_only_metrics_and_their_mean_over_the_12_targets(tmp_path):
    run = write_run(tmp_path / "run", [
        trial("T02_balance", "T02", "balance", wave, offset=6.0),
        trial("T02_walk", "T02", "walk", wave, offset=3.0),
        trial("T03_walk", "T03", "walk", wave, offset=3.0),
    ])
    rows = comparison(evaluate(tmp_path, {"mlp": run}))
    walk = rows[rows.comparison == "walk"]
    assert set(walk.task_family) == {"walk"}
    assert set(walk.metric) == {"nrmse_range", "pcc_trial_mean", "rmse"}
    fz = walk[(walk.target == "Fz_L") & (walk.metric == "rmse")].iloc[0]
    assert fz["mean"] == pytest.approx(3.0) and fz["n_participants"] == 2
    mean_row = walk[(walk.target == "mean_12") & (walk.metric == "pcc_trial_mean")].iloc[0]
    assert mean_row["mean"] == pytest.approx(1.0)


def test_sum_of_both_feet_per_continuous_task_in_percent_body_weight(tmp_path):
    run = write_run(tmp_path / "run", [
        trial("T02_balance", "T02", "balance", wave, offset=6.0),
        trial("T02_squat", "T02", "squat", wave, offset=3.0),
        trial("T02_walk", "T02", "walk", wave, offset=3.0),
    ])
    rows = comparison(evaluate(tmp_path, {"mlp": run}))
    total = rows[(rows.comparison == "both_feet_sum") & (rows.metric == "rmse_pct_bw")]
    assert set(total.task_family) == {"balance", "squat"}  # continuous tasks only
    balance = total[total.task_family == "balance"].set_index("target")
    # Each foot is 6 N off, so the sum is 12 N off: 2 % of the 600 N body weight.
    assert balance.loc["Fz_sum", "mean"] == pytest.approx(2.0)
    assert balance.loc["forces_mean", "mean"] == pytest.approx(2.0)


def test_protocol_table_lists_the_competitors_and_each_run(tmp_path):
    run = write_run(tmp_path / "run", [trial("T02_balance", "T02", "balance", wave, offset=1.0)])
    (run / "fold_T02" / "metadata.json").write_text(json.dumps({
        "causal": True, "test_groups": ["T02"], "validation_groups": ["T03"], "task_set": "all",
    }), encoding="utf-8")
    table = pd.read_csv(evaluate(tmp_path, {"mlp": run}) / "protocol_table.csv").set_index("study")
    assert {"Sabbah 2024", "Li 2025", "Mahmud 2026"} <= set(table.index)
    ours = table.loc["mlp"]
    assert ours["split"] == "Fixed split (test: T02; validation: T03)"
    assert ours["causal"] == "yes"
    assert ours["tasks"] == "balance"


def test_full_cv_is_named_only_when_every_participant_of_the_split_is_tested(tmp_path):
    frames = [trial(f"{name}_balance", name, "balance", wave, offset=1.0) for name in ("T02", "T03")]
    run = write_run(tmp_path / "run", frames)
    for name in ("T02", "T03"):
        (run / f"fold_{name}" / "metadata.json").write_text(json.dumps({"causal": True}), encoding="utf-8")
        (run / f"fold_{name}" / "split_manifest.json").write_text(
            json.dumps({"split": "cv", "groups": ["T02", "T03", "T08"]}), encoding="utf-8"
        )
    table = pd.read_csv(evaluate(tmp_path, {"mlp": run}) / "protocol_table.csv").set_index("study")
    assert table.loc["mlp", "split"] == "partial CV (2 of 3 Participants tested)"
    assert table.loc["mlp", "participants"] == "2 evaluated"
    for name in ("T02", "T03"):
        (run / f"fold_{name}" / "split_manifest.json").write_text(
            json.dumps({"split": "cv", "groups": ["T02", "T03"]}), encoding="utf-8"
        )
    table = pd.read_csv(evaluate(tmp_path / "again", {"mlp": run}) / "protocol_table.csv").set_index("study")
    assert table.loc["mlp", "split"] == "Full CV (2 folds, one per Participant)"
