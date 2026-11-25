"""Foot height metrics of the evaluation command (#20), kept apart from the GRFM tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from grfm_schema import STATE_AIRBORNE, STATE_MASKED, STATE_MEASURED
from test_evaluate_grfm import SAMPLES, evaluate, per_participant, trial, wave, write_run


def with_heights(frame, truth_mm, error_mm, valid=True):
    """Add the engine's Foot height columns: prediction = truth + error (mm), per foot."""
    frame = frame.copy()
    for side in ("L", "R"):
        error = error_mm[side] if isinstance(error_mm, dict) else error_mm
        frame[f"foot_height_mm_{side}"] = np.where(valid, truth_mm, np.nan)
        frame[f"pred_foot_height_mm_{side}"] = truth_mm + error
        frame[f"foot_height_valid_{side}"] = np.asarray(valid, dtype=int) * np.ones(SAMPLES, dtype=int)
    return frame


def heights(out):
    return pd.read_csv(out / "foot_height_per_participant.csv")


def row(table, **keys):
    selected = table
    for key, value in keys.items():
        selected = selected[selected[key] == value]
    assert len(selected) == 1, selected
    return selected.iloc[0]


@pytest.fixture
def run_dir(tmp_path):
    t = np.arange(SAMPLES)
    flat = np.zeros(SAMPLES)
    lifted = 50.0 + 10.0 * np.sin(t / 5.0)
    # T01: walk with the left foot Masked (error 4 mm) and the right foot Measured (error 1 mm),
    # jump Airborne (error 3 mm); T03: walk with a constant 2 mm error and invalid Foot height on one half.
    valid_half = t < SAMPLES // 2
    frames = [
        with_heights(trial("T01_walk", "T01", "walk", wave, 1.0, state_l=STATE_MASKED), lifted,
                     {"L": 4.0, "R": 1.0}),
        with_heights(trial("T01_jump1", "T01", "jump", wave, 1.0, STATE_AIRBORNE, STATE_AIRBORNE), lifted, 3.0),
        with_heights(trial("T03_walk", "T03", "walk", wave, 1.0), flat, np.where(valid_half, 2.0, 99.0), valid_half),
    ]
    return write_run(tmp_path / "run", frames)


def test_foot_height_rmse_is_reported_in_mm_per_foot_state_and_task_family(tmp_path, run_dir):
    table = heights(evaluate(tmp_path, {"m": run_dir}))
    assert row(table, participant="T01", foot="L", foot_state="masked", task_family="walk")["rmse_mm"] == pytest.approx(4.0)
    assert row(table, participant="T01", foot="R", foot_state="measured", task_family="walk")["rmse_mm"] == pytest.approx(1.0)
    assert row(table, participant="T01", foot="L", foot_state="airborne", task_family="jump")["rmse_mm"] == pytest.approx(3.0)
    # All states and families pooled: left foot 4 mm on walk, 3 mm on jump, equal sample counts.
    pooled = row(table, participant="T01", foot="L", foot_state="all", task_family="ALL")
    assert pooled["rmse_mm"] == pytest.approx(np.sqrt((16.0 + 9.0) / 2.0))
    assert pooled["n"] == 2 * SAMPLES
    # Invalid Foot height is never scored (its prediction error is 99 mm).
    bob = row(table, participant="T03", foot="L", foot_state="measured", task_family="walk")
    assert bob["rmse_mm"] == pytest.approx(2.0)
    assert bob["n"] == SAMPLES // 2


def test_foot_height_summary_is_mean_and_sd_over_participants(tmp_path, run_dir):
    summary = pd.read_csv(evaluate(tmp_path, {"m": run_dir}) / "foot_height_summary.csv")
    walk = summary[(summary["foot"] == "R") & (summary["foot_state"] == "measured")
                   & (summary["task_family"] == "walk") & (summary["metric"] == "rmse_mm")]
    (mean,) = walk["mean"]
    (sd,) = walk["sd"]
    assert mean == pytest.approx(1.5)
    assert sd == pytest.approx(np.std([1.0, 2.0], ddof=1))


def test_foot_height_stays_out_of_the_grfm_tables(tmp_path, run_dir):
    out = evaluate(tmp_path, {"m": run_dir})
    grfm = per_participant(out)
    assert not grfm["target"].str.contains("foot_height").any()
    assert set(grfm["subset"]) <= {"measured", "airborne", "both_feet_sum"}


def test_runs_without_foot_height_write_no_foot_height_table(tmp_path):
    run = write_run(tmp_path / "plain", [trial("T01_walk", "T01", "walk", wave, 1.0)])
    out = evaluate(tmp_path, {"plain": run})
    assert not (out / "foot_height_per_participant.csv").exists()
    assert (out / "per_participant.csv").exists()
