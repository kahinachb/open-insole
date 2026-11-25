"""Sensor-token Transformer model type of the training engine (ticket #8)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sensor_tokens import SensorTokenizer
from synthetic_prepared import TARGETS, checkpoint, run_training_cli, write_synthetic_prepared

SENSOR_TOKEN = (
    "--model-type", "sensor_token", "--causal",
    "--layers", "1", "--transformer-heads", "2", "--transformer-feedforward-size", "16",
)
# Gauges around 40 N so that they are active Sensor tokens (p99 >= 5 N).
GAUGE_OFFSET_N = 40.0


@pytest.fixture(scope="module")
def active_gauges(tmp_path_factory):
    return write_synthetic_prepared(
        tmp_path_factory.mktemp("active_gauges"), gauge_offset_n=GAUGE_OFFSET_N, mirror_symmetric=True
    )


FULL_CV = ("--gradnorm", "--split-level", "subject_identity", "--task-set", "continuous")


@pytest.fixture(scope="module")
def full_cv(active_gauges, tmp_path_factory):
    out = tmp_path_factory.mktemp("full_cv") / "run"
    run_training_cli(active_gauges.data_dir, out, *SENSOR_TOKEN, *FULL_CV)
    return out


def test_full_cv_with_gradnorm_writes_the_usual_outputs(active_gauges, full_cv):
    out = full_cv

    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert sorted(f["test_group"] for f in manifest["folds"]) == sorted(active_gauges.participants)
    predictions = pd.read_csv(out / "predictions.csv")
    assert set(predictions["test_group"]) == set(active_gauges.participants)
    assert np.isfinite(predictions[[f"pred_{t}" for t in TARGETS]].to_numpy()).all()
    metrics = pd.read_csv(out / "fold_metrics.csv")
    assert set(metrics["model"]) == {"sensor_token_two_feet_masked"}
    for participant in active_gauges.participants:
        assert (out / "models" / f"sensor_token_test_{participant}.pt").exists()
    weights = pd.read_csv(out / "gradnorm_weights.csv")
    assert np.allclose(weights[TARGETS].sum(axis=1), 12.0, atol=1e-3)
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["model_type"] == "sensor_token"
    assert metadata["token_dropout"] == pytest.approx(0.1)


FOOT_SIGN = np.array([1, -1, 1, -1, 1, -1], dtype=np.float32)  # Fy, Mx, Mz flip under the mirror
CHANGE_AFTER_S = 0.3


def add_trial(data_dir, source_batch, new_batch, transform):
    """Copy one trial of a prepared dataset under a new batch name after ``transform``."""
    arrays = dict(np.load(data_dir / f"{source_batch}.npz"))
    np.savez(data_dir / f"{new_batch}.npz", **transform(arrays))
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    report = next(r for r in metadata["reports"] if r["batch"] == source_batch)
    metadata["batches"].append(new_batch)
    metadata["reports"].append({**report, "batch": new_batch, "evaluation_group": new_batch})
    (data_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def swapped_feet(arrays):
    """Left and right feet exchanged, inputs and targets mirrored into each other."""
    out = dict(arrays)
    out["X"] = SensorTokenizer.from_layout().swap_feet(arrays["X"]).astype(np.float32)
    y = arrays["y"]
    out["y"] = np.concatenate([y[:, 6:] * FOOT_SIGN, y[:, :6] * FOOT_SIGN], axis=1)
    mask = arrays["target_mask"]
    out["target_mask"] = np.concatenate([mask[:, 6:], mask[:, :6]], axis=1)
    out["state_L"], out["state_R"] = arrays["state_R"], arrays["state_L"]
    out["airborne_label_L"], out["airborne_label_R"] = arrays["airborne_label_R"], arrays["airborne_label_L"]
    return out


def changed_future(arrays):
    """Every input channel changed after CHANGE_AFTER_S (gauges stay active)."""
    out = dict(arrays)
    x = arrays["X"].copy()
    later = arrays["target_time_s"] > CHANGE_AFTER_S
    x[later] = x[later] * 1.5 + 2.0
    out["X"] = x
    return out


@pytest.fixture(scope="module")
def probe_run(tmp_path_factory):
    """One trained fold whose test participant also holds a feet-swapped and a future-changed copy."""
    root = tmp_path_factory.mktemp("probe")
    data = write_synthetic_prepared(root / "data", gauge_offset_n=GAUGE_OFFSET_N, mirror_symmetric=True)
    add_trial(data.data_dir, "T01_squat", "T01_squatswapped", swapped_feet)
    add_trial(data.data_dir, "T01_squat", "T01_squatfuture", changed_future)
    out = root / "run"
    run_training_cli(
        data.data_dir, out, *SENSOR_TOKEN, "--dropout", "0.1",
        "--split-level", "subject_identity", "--only-test-group", "T01",
    )
    predictions = pd.read_csv(out / "predictions.csv")
    return {batch: rows.reset_index(drop=True) for batch, rows in predictions.groupby("test_batch")}


def predicted(rows):
    return rows[[f"pred_{t}" for t in TARGETS]].to_numpy()


def test_changing_samples_after_the_prediction_time_does_not_change_the_prediction(probe_run):
    original, changed = probe_run["T01_squat"], probe_run["T01_squatfuture"]
    assert np.array_equal(original["time_s"], changed["time_s"])
    before = original["time_s"].to_numpy() <= CHANGE_AFTER_S + 1e-9
    assert before.sum() > 10 and (~before).sum() > 10
    np.testing.assert_allclose(predicted(changed)[before], predicted(original)[before], rtol=1e-5, atol=1e-4)
    assert not np.allclose(predicted(changed)[~before], predicted(original)[~before], atol=1e-2)


def test_swapping_the_feet_with_mirrored_inputs_yields_mirrored_outputs(probe_run):
    original, swapped = predicted(probe_run["T01_squat"]), predicted(probe_run["T01_squatswapped"])
    expected = np.concatenate([original[:, 6:] * FOOT_SIGN, original[:, :6] * FOOT_SIGN], axis=1)
    np.testing.assert_allclose(swapped, expected, rtol=1e-4, atol=1e-2)
    assert not np.allclose(original[:, :6], original[:, 6:], atol=1.0)



def test_input_layer_norm_option_adds_a_token_embedding_norm(active_gauges, full_cv, tmp_path):
    out = tmp_path / "run"
    # Same configuration as the full-CV run's first fold (T01), plus the option.
    run_training_cli(
        active_gauges.data_dir, out, *SENSOR_TOKEN, *FULL_CV, "--only-test-group", "T01", "--input-layer-norm",
    )
    assert checkpoint(out, "sensor_token")["input_layer_norm"] is True
    assert json.loads((out / "metadata.json").read_text(encoding="utf-8"))["input_layer_norm"] is True
    assert json.loads((full_cv / "metadata.json").read_text(encoding="utf-8"))["input_layer_norm"] is False
    normed = pd.read_csv(out / "predictions.csv")
    plain = pd.read_csv(full_cv / "predictions.csv")
    plain = plain[plain["test_group"] == "T01"]
    assert len(normed) == len(plain)
    assert not np.allclose(predicted(normed), predicted(plain), atol=1e-3)
