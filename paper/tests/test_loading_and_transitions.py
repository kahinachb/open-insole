"""Tests of paper/scripts/loading_and_transitions.py: the backpack split and the contact/lift-off windows."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import loading_and_transitions as lt  # noqa: E402


def trial(batch, states, true, pred, family="walk"):
    """One foot (L) of one trial at 200 Hz; the right foot copies it."""
    n = len(states)
    frame = pd.DataFrame({"test_batch": batch, "test_group": batch.split("_")[0].rstrip("b"), "task_family": family,
                          "time_s": np.arange(n) * 0.005})
    for side in "LR":
        frame[f"state_{side}"] = states
        frame[f"Fz_{side}"] = true
        frame[f"pred_Fz_{side}"] = pred
        frame[f"mask_Fz_{side}"] = [1 if s in (1, 2) else 0 for s in states]
    return frame


def test_a_subject_with_a_trailing_b_carries_the_backpack():
    assert lt.condition_of("P03b_walk2") == "backpack"
    assert lt.condition_of("P03_walk2") == "unloaded"


def test_samples_within_the_window_of_a_state_change_are_labelled_by_the_event():
    # stance 30 samples, air 30, stance 30; window 2 samples
    states = np.array([1] * 30 + [2] * 30 + [1] * 30)
    labels = lt.transition_labels(states, np.arange(90) * 0.005, window=2)

    assert list(labels[27:33]) == ["stance", "lift_off", "lift_off", "lift_off", "lift_off", "air"]
    assert list(labels[58:62]) == ["contact"] * 4
    assert labels[0] == "stance" and labels[45] == "air"


def test_a_gap_in_time_is_not_a_transition():
    states = np.array([1] * 10 + [2] * 10)
    time = np.concatenate([np.arange(10) * 0.005, 10 + np.arange(10) * 0.005])  # a new segment starts in the air
    labels = lt.transition_labels(states, time, window=2)
    assert set(labels) == {"stance", "air"}


def test_unmeasured_samples_are_never_scored():
    labels = lt.transition_labels(np.array([1, 1, 0, 0, 2, 2]), np.arange(6) * 0.005, window=1)
    assert labels[2] == labels[3] == ""


def test_transition_errors_give_the_rmse_and_mean_error_of_each_event_per_participant():
    states = [1] * 30 + [2] * 30 + [1] * 30
    true = np.array([500.0] * 30 + [0.0] * 30 + [500.0] * 30)
    pred = true + 10.0
    frame = trial("P01_walk", states, true, pred)

    table = lt.transition_errors(frame, window=2).set_index(["participant", "foot", "event"])

    row = table.loc[("P01", "L", "contact")]
    assert row.rmse_N == pytest.approx(10.0)
    assert row.mean_error_N == pytest.approx(10.0)
    assert row.n_events == 1


def test_loading_errors_split_each_participant_by_backpack():
    frame = pd.concat([
        trial("P01_walk", [1] * 20, np.full(20, 500.0), np.full(20, 510.0)),
        trial("P01b_walk", [1] * 20, np.full(20, 500.0), np.full(20, 530.0)),
    ])

    table = lt.loading_errors(frame, targets=["Fz_L"]).set_index(["participant", "condition", "target"])

    assert table.loc[("P01", "unloaded", "Fz_L")].rmse == pytest.approx(10.0)
    assert table.loc[("P01", "backpack", "Fz_L")].rmse == pytest.approx(30.0)


def test_a_change_of_state_across_a_short_excluded_gap_is_a_transition():
    # air 20, excluded 10 (50 ms, foot near the ground before the plate loads), stance 20; window 2
    states = np.array([2] * 20 + [0] * 10 + [1] * 20)
    labels = lt.transition_labels(states, np.arange(50) * 0.005, window=2)

    assert list(labels[18:20]) == ["contact"] * 2  # last airborne samples
    assert list(labels[30:32]) == ["contact"] * 2  # first measured samples
    assert labels[17] == "air" and labels[32] == "stance"
    assert lt._events(states, np.arange(50) * 0.005) == {"lift_off": 0, "contact": 1}


def test_a_change_of_state_across_missing_rows_is_a_transition_when_short():
    states = np.array([1] * 10 + [2] * 10)
    short = np.concatenate([np.arange(10) * 0.005, 0.15 + np.arange(10) * 0.005])  # 100 ms of rows dropped
    assert lt._events(states, short) == {"lift_off": 1, "contact": 0}
