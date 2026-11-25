"""Sabbah 2024's Random Forest retrained under our Full CV, offline as published (#46)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter

import competitor_folds as folds
import competitor_sabbah as sabbah
import evaluate_grfm
from synthetic_prepared import FAMILIES, PARTICIPANTS, write_synthetic_prepared
from train_lstm_two_feet_masked import MaskedWindowDataset

BODY_WEIGHTS_N = {
    subject: 600.0 + 50.0 * index + (150.0 if loaded else 0.0)
    for index, participant in enumerate(PARTICIPANTS)
    for subject, loaded in ((participant, False), (f"{participant}b", True))
}


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    return write_synthetic_prepared(tmp_path_factory.mktemp("weighted"), body_weights_n=BODY_WEIGHTS_N)


@pytest.fixture(scope="module")
def run(weighted, tmp_path_factory):
    out = tmp_path_factory.mktemp("sabbah") / "sabbah_rf_loadsol" / "all"
    assert sabbah.main([str(weighted.data_dir), "--inputs", "loadsol", "--output-dir", str(out),
                        "--participant", "T01", "--train-stride", "1", "--jobs", "1"]) == 0
    return out / "fold_T01"


def test_the_filter_is_zero_phase_and_low_pass():
    fs = 200.0
    t = np.arange(0, 4, 1 / fs)
    slow, fast = np.sin(2 * np.pi * 2 * t), np.sin(2 * np.pi * 40 * t)
    sos = butter(sabbah.FILTER_ORDER, sabbah.FILTER_CUTOFF_HZ, fs=fs, output="sos")
    filtered = sabbah.zero_phase(np.stack([slow, fast], axis=1), sos, [slice(0, len(t))])
    middle = slice(100, -100)
    # 2 Hz passes with no lag: a causal 5th-order filter would delay it by several samples.
    assert np.abs(filtered[middle, 0] - slow[middle]).max() < 0.01
    assert np.abs(filtered[middle, 1]).max() < 0.01


def test_the_filter_never_crosses_a_segment():
    sos = butter(sabbah.FILTER_ORDER, sabbah.FILTER_CUTOFF_HZ, fs=200.0, output="sos")
    values = np.r_[np.zeros(50), np.full(50, 100.0)][:, None]
    filtered = sabbah.zero_phase(values, sos, folds.segment_runs(np.r_[np.zeros(50), np.ones(50)]))
    assert np.allclose(filtered[:50], 0.0) and np.allclose(filtered[50:], 100.0)


def test_the_scored_samples_are_those_of_the_causal_mlp(weighted):
    acquisitions, *_ = folds.load_folds(weighted.data_dir)
    zeros = np.zeros(acquisitions[0].x.shape[1], np.float32)
    ones = np.ones(acquisitions[0].y.shape[1], np.float32)
    dataset = MaskedWindowDataset(acquisitions, zeros, zeros + 1, ones * 0, ones, list(range(12)),
                                  folds.REFERENCE_WINDOW, centered=False, zero_weight=0.25)
    for index, item in enumerate(acquisitions):
        engine = [row for trial, row in dataset.index if trial == index]
        assert folds.scored_indexes(item).tolist() == engine


def test_the_fold_is_the_engine_fold_and_never_trains_on_its_test_participant(run):
    manifest = json.loads((run / "split_manifest.json").read_text(encoding="utf-8"))
    (fold,) = manifest["folds"]
    assert fold["test_groups"] == ["T01"] and fold["validation_groups"] == ["T03", "T04"]
    assert fold["train_groups"] == ["T05"]
    assert not any(name.startswith("T01") for name in fold["train_batches"])
    assert json.loads((run / "metadata.json").read_text(encoding="utf-8"))["causal"] is False


def test_every_scored_sample_of_every_test_trial_is_predicted(weighted, run):
    acquisitions, *_ = folds.load_folds(weighted.data_dir)
    predictions = pd.read_csv(run / "predictions.csv")
    for item in acquisitions:
        if item.subject_identity == "T01":
            assert (predictions["test_batch"] == item.name).sum() == len(folds.scored_indexes(item))
    assert predictions.filter(like="pred_").notna().all().all()


def test_one_forest_is_fitted_per_task_family_and_foot(weighted, tmp_path, monkeypatch):
    fitted = []
    original = sabbah.RandomForestRegressor.fit
    monkeypatch.setattr(sabbah.RandomForestRegressor, "fit", lambda self, x, y: fitted.append(x.shape) or original(self, x, y))
    assert sabbah.main([str(weighted.data_dir), "--inputs", "raw", "--output-dir", str(tmp_path),
                        "--participant", "T01", "--jobs", "1"]) == 0
    assert len(fitted) == 2 * len(FAMILIES)
    # 78 raw channels and the subject weight.
    assert {shape[1] for shape in fitted} == {79}


def test_evaluate_grfm_scores_the_run_unchanged(run, tmp_path):
    assert evaluate_grfm.main(["--run", f"sabbah={run.parent}", "--output-dir", str(tmp_path)]) == 0
    summary = pd.read_csv(tmp_path / "summary.csv")
    assert not summary.empty


def test_a_dataset_without_body_weights_is_rejected(synthetic_prepared, tmp_path):
    with pytest.raises(SystemExit):
        sabbah.main([str(synthetic_prepared.data_dir), "--inputs", "raw", "--output-dir", str(tmp_path)])


def test_the_total_target_is_one_forest_per_family_on_the_summed_forces(weighted, tmp_path, monkeypatch):
    targets = []
    original = sabbah.RandomForestRegressor.fit
    monkeypatch.setattr(sabbah.RandomForestRegressor, "fit", lambda self, x, y: targets.append(y) or original(self, x, y))
    out = tmp_path / "all"
    assert sabbah.main([str(weighted.data_dir), "--inputs", "loadsol", "--target", "total", "--output-dir", str(out),
                        "--participant", "T01", "--train-stride", "1", "--jobs", "1"]) == 0
    assert len(targets) == len(FAMILIES) and {y.shape[1] for y in targets} == {3}
    # Each foot gets half the total, so the evaluator's both-feet sum is the forest's prediction.
    predictions = pd.read_csv(out / "fold_T01" / "predictions.csv")
    for component in ("Fx", "Fy", "Fz"):
        assert np.allclose(predictions[f"pred_{component}_L"], predictions[f"pred_{component}_R"])
    assert evaluate_grfm.main(["--run", f"total={out}", "--output-dir", str(tmp_path / "eval")]) == 0
    per_participant = pd.read_csv(tmp_path / "eval" / "per_participant.csv")
    assert (per_participant["subset"] == evaluate_grfm.SUM_SUBSET).any()
