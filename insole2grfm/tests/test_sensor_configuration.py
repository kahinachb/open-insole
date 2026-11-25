"""Inputs restricted to a Sensor configuration, for dedicated retraining (#33)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import infer_lstm_two_feet_npz as infer
from synthetic_prepared import ONE_FOLD, PLAIN_MLP_ONE_FOLD, checkpoint, run_training_cli

ALL_GROUPS = tuple(
    f"{patch}_{modality}" for patch in ("heel", "midfoot", "forefoot", "toes") for modality in ("gauges", "acc", "gyro")
)


def test_only_the_named_groups_reach_the_model(synthetic_prepared, trained_once):
    out = trained_once(
        synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, "--sensor-configuration", "heel_gauges", "toes_gyro",
    )
    saved = checkpoint(out)
    assert saved["sensor_configuration"] == ["heel_gauges", "toes_gyro"]
    names = saved["feature_names"]
    assert len(names) == saved["input_size"] == 8 + 6  # 4 heel Gauges and 3 toes gyro axes, per foot
    assert all("_p1_force_" in name or "_p9_force_" in name or "_gyro_" in name for name in names)
    assert saved["source_feature_names"] == names


def test_inference_rebuilds_the_restricted_inputs(synthetic_prepared, trained_once):
    out = trained_once(
        synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, "--sensor-configuration", "heel_gauges", "toes_gyro",
    )
    saved = checkpoint(out)
    trial = next(synthetic_prepared.data_dir.glob("T01_*.npz"))
    metadata = json.loads((synthetic_prepared.data_dir / "metadata.json").read_text(encoding="utf-8"))
    acquisition, _ = infer.load_acquisition(trial, saved, metadata)
    assert acquisition.x.shape[1] == 14


def test_the_full_configuration_reproduces_a_run_without_the_option(synthetic_prepared, trained_once):
    plain = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD)
    full = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, "--sensor-configuration", *ALL_GROUPS)
    pd.testing.assert_frame_equal(pd.read_csv(plain / "predictions.csv"), pd.read_csv(full / "predictions.csv"))


def test_the_patch_transformer_trains_on_a_configuration(synthetic_prepared, trained_once):
    out = trained_once(
        synthetic_prepared.data_dir, "--model-type", "patch_transformer", "--layers", "1",
        "--transformer-heads", "2", *ONE_FOLD, "--sensor-configuration", "forefoot_acc",
    )
    assert checkpoint(out, "patch_transformer")["input_size"] == 6


@pytest.mark.parametrize("arguments", [
    ("--sensor-configuration", "heel_bogus"),
    ("--sensor-configuration", "heel_gauges", "--input-features", "gauge_augmented"),
    ("--sensor-configuration", "heel_gauges", "--sensor-dropout"),
    ("--sensor-configuration", "heel_gauges", "--target-scaling", "subject_load"),
])
def test_invalid_configurations_are_rejected(synthetic_prepared, tmp_path, arguments):
    result = run_training_cli(synthetic_prepared.data_dir, tmp_path / "run", *PLAIN_MLP_ONE_FOLD, *arguments, check=False)
    assert result.returncode == 2
