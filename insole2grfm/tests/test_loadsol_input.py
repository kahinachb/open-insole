"""Loadsol-like input representation: 4 zonal gauge sums per foot, optional heel IMU (#34)."""

from __future__ import annotations

import json

import numpy as np
import pytest

import infer_lstm_two_feet_npz as infer
import train_lstm_two_feet_masked as engine
from synthetic_prepared import FEATURE_NAMES, PLAIN_MLP_ONE_FOLD, checkpoint, run_training_cli

LOADSOL = ("--input-features", "loadsol")
ZONES = ("heel", "medial", "lateral", "total")


def column(name: str) -> int:
    return FEATURE_NAMES.index(name)


def test_zonal_sums_follow_the_gauge_positions():
    x = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    loads = {
        "L_p1_force_1_N": 10.0, "L_p1_force_4_N": 5.0,      # heel
        "L_p3_force_2_N": 7.0,                               # midfoot: total only
        "L_p5_force_2_N": 3.0, "L_p7_force_4_N": 2.0,        # medial forefoot / toes
        "L_p5_force_1_N": 4.0, "L_p7_force_3_N": 1.0,        # lateral forefoot / toes
        "R_p13_force_4_N": 6.0, "R_p15_force_3_N": 8.0,      # right: medial, lateral
    }
    for name, load in loads.items():
        x[:, column(name)] = load
    zones, names = engine.loadsol_inputs(x, list(FEATURE_NAMES))
    assert names == [f"{side}_zone_{zone}_N" for side in "LR" for zone in ZONES]
    assert zones[0].tolist() == [15.0, 5.0, 5.0, 32.0, 0.0, 6.0, 8.0, 14.0]


def test_the_heel_imu_variant_appends_the_heel_patch_imu_channels():
    x = np.arange(3 * len(FEATURE_NAMES), dtype=np.float32).reshape(3, -1)
    zones, names = engine.loadsol_inputs(x, list(FEATURE_NAMES), heel_imu=True)
    imu = [name for name in names if "_zone_" not in name]
    assert imu == [name for name in FEATURE_NAMES if name[:5] in ("L_p1_", "R_p9_") and "force" not in name]
    assert np.array_equal(zones[:, names.index("L_p1_acc_x_cal_g")], x[:, column("L_p1_acc_x_cal_g")])


@pytest.mark.parametrize("extra, size", [((), 8), (("--loadsol-heel-imu",), 8 + 12)])
def test_a_model_trains_on_the_loadsol_inputs_and_inference_rebuilds_them(synthetic_prepared, trained_once, extra, size):
    out = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, *LOADSOL, *extra)
    saved = checkpoint(out)
    assert saved["input_features"] == "loadsol" and saved["input_size"] == size
    assert saved["loadsol_heel_imu"] == bool(extra)
    trial = next(synthetic_prepared.data_dir.glob("T01_*.npz"))
    metadata = json.loads((synthetic_prepared.data_dir / "metadata.json").read_text(encoding="utf-8"))
    acquisition, _ = infer.load_acquisition(trial, saved, metadata)
    assert acquisition.x.shape[1] == size


@pytest.mark.parametrize("arguments", [
    ("--loadsol-heel-imu",),
    (*LOADSOL, "--sensor-dropout"),
    (*LOADSOL, "--target-scaling", "subject_load"),
])
def test_invalid_loadsol_arguments_are_rejected(synthetic_prepared, tmp_path, arguments):
    result = run_training_cli(synthetic_prepared.data_dir, tmp_path / "run", *PLAIN_MLP_ONE_FOLD, *arguments, check=False)
    assert result.returncode == 2
