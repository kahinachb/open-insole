"""Sensor groups, presence indicators and Sensor dropout (#32)."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest
import torch

import train_lstm_two_feet_masked as engine
from synthetic_prepared import FEATURE_NAMES, PLAIN_MLP_ONE_FOLD, checkpoint

DROPOUT = ("--sensor-dropout",)


def dropout(**options) -> engine.SensorDropout:
    names = engine.append_sensor_presence_names(list(FEATURE_NAMES))
    return engine.SensorDropout.from_feature_names(names, **options)


def window(value: float = 1.0) -> np.ndarray:
    names = engine.append_sensor_presence_names(list(FEATURE_NAMES))
    return np.full((5, len(names)), value, dtype=np.float32)


def kept_groups(sensor_dropout, x) -> tuple[int, ...]:
    presence = x[0, sensor_dropout.presence_columns]
    return tuple(int(index) for index in np.flatnonzero(presence > 0.0))


def test_the_78_channels_form_12_sensor_groups_shared_by_both_feet():
    groups = engine.sensor_groups(list(FEATURE_NAMES))
    assert list(groups) == [
        f"{patch}_{modality}"
        for patch in ("heel", "midfoot", "forefoot", "toes")
        for modality in ("gauges", "acc", "gyro")
    ]
    assert sorted(column for columns in groups.values() for column in columns) == list(range(78))
    assert len(groups["heel_gauges"]) == 8  # 4 Gauges per foot
    assert len(groups["toes_gauges"]) == 6  # 3 Gauges per foot
    assert {FEATURE_NAMES[column][0] for column in groups["toes_gyro"]} == {"L", "R"}


def test_configuration_sizes_are_uniform_apart_from_the_full_configuration_share():
    torch.manual_seed(0)
    sensor_dropout = dropout(random=True, full_probability=0.3)
    sizes = Counter(len(kept_groups(sensor_dropout, sensor_dropout.apply(window()))) for _ in range(6000))
    assert set(sizes) <= set(range(1, 13))
    expected_small = 6000 * 0.7 / 12
    assert all(abs(sizes[k] - expected_small) < 0.25 * expected_small for k in range(1, 12))
    assert abs(sizes[12] - 6000 * (0.3 + 0.7 / 12)) < 0.1 * 6000 * (0.3 + 0.7 / 12)


def test_dropped_groups_are_zero_on_both_feet_and_their_indicator_is_zero():
    torch.manual_seed(1)
    sensor_dropout = dropout(random=True, full_probability=0.0)
    groups = list(engine.sensor_groups(list(FEATURE_NAMES)).values())
    for _ in range(50):
        x = sensor_dropout.apply(window())
        kept = kept_groups(sensor_dropout, x)
        for index, columns in enumerate(groups):
            expected = 1.0 if index in kept else 0.0
            assert np.all(x[:, columns] == expected)  # both feet together
            assert np.all(x[:, sensor_dropout.presence_columns[index]] == expected)


def test_explicit_configuration_is_applied_without_randomness():
    sensor_dropout = dropout(configuration=("heel_gauges", "forefoot_gauges"))
    x = sensor_dropout.apply(window())
    assert kept_groups(sensor_dropout, x) == (0, 6)
    full = dropout()  # evaluation default: the full configuration
    assert kept_groups(full, full.apply(window())) == tuple(range(12))


def test_channel_drops_and_inactive_sensors_lower_the_group_indicator():
    sensor_dropout = dropout(configuration=tuple(engine.sensor_groups(list(FEATURE_NAMES))))
    heel_gauge = FEATURE_NAMES.index("L_p1_force_1_N")
    x = sensor_dropout.apply(window(), inactive_columns=(heel_gauge,))
    assert np.all(x[:, heel_gauge] == 0.0)
    assert x[0, sensor_dropout.presence_columns[0]] == pytest.approx(7 / 8)
    torch.manual_seed(2)
    noisy = dropout(random=True, full_probability=1.0, channel_probability=0.5)
    x = noisy.apply(window())
    fractions = x[0, noisy.presence_columns]
    assert np.all((fractions > 0.0) & (fractions < 1.0))


def test_inactive_columns_are_gauges_whose_99th_percentile_stays_below_5_n():
    names = list(FEATURE_NAMES)
    x = np.full((100, len(names)), 50.0, dtype=np.float32)
    dead = names.index("R_p13_force_2_N")
    quiet_imu = names.index("R_p13_acc_x_cal_g")
    x[:, dead] = 1.0
    x[:, quiet_imu] = 0.0  # IMUs are never inactive
    assert engine.inactive_gauge_columns(x, names) == (dead,)


def history(out):
    return json.loads((out / "training_history.json").read_text(encoding="utf-8"))["T01"]


def test_dropout_is_off_by_default(synthetic_prepared, trained_once):
    saved = checkpoint(trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD))
    assert not any(name.startswith("presence_") for name in saved["feature_names"])
    assert saved["sensor_dropout"] is None


def test_a_model_trained_with_sensor_dropout_reacts_to_a_dropped_group(synthetic_prepared, trained_once):
    out = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, *DROPOUT)
    saved = checkpoint(out)
    names = saved["feature_names"]
    assert names[-12:] == [f"presence_{group}" for group in engine.sensor_groups(list(FEATURE_NAMES))]
    assert saved["sensor_dropout"]["full_probability"] == 0.3
    model = engine.build_model(
        "mlp", len(names), saved["window_samples"], saved["hidden_size"], saved["layers"], 0.0, 12,
        feature_names=names,
    )
    model.load_state_dict(saved["state_dict"])
    model.eval()
    full = engine.SensorDropout.from_feature_names(names)
    reduced = engine.SensorDropout.from_feature_names(names, configuration=("heel_gauges",))
    base = np.random.default_rng(0).normal(size=(saved["window_samples"], len(names))).astype(np.float32)
    with torch.no_grad():
        outputs = [model(torch.from_numpy(rule.apply(base.copy()))[None]) for rule in (full, reduced)]
    assert not torch.allclose(*outputs)


def test_patch_transformer_is_conditioned_on_the_presence_indicators():
    names = engine.append_sensor_presence_names(list(FEATURE_NAMES))
    built = engine.build_model("patch_transformer", len(names), 10, 16, 1, 0.0, 12, transformer_heads=2,
                               feature_names=names, patch_samples=5)
    assert built.condition_columns == list(range(78, 90))


def test_overfit_test_runs_with_sensor_dropout(synthetic_prepared, tmp_path):
    from synthetic_prepared import run_training_cli

    out = tmp_path / "overfit"
    result = run_training_cli(
        synthetic_prepared.data_dir, out, *PLAIN_MLP_ONE_FOLD, *DROPOUT, "--overfit-test", "--overfit-epochs", "2",
        check=False,
    )
    assert result.returncode in (0, engine.OVERFIT_EXIT_FAIL)
    assert (out / "overfit_verdict.json").exists()


@pytest.mark.parametrize("arguments", [
    ("--sensor-dropout", "--sensor-full-probability", "1.5"),
    ("--sensor-dropout", "--sensor-channel-dropout", "-0.1"),
    ("--sensor-full-probability", "0.5"),
])
def test_invalid_sensor_dropout_arguments_are_rejected(synthetic_prepared, tmp_path, arguments):
    from synthetic_prepared import run_training_cli

    result = run_training_cli(synthetic_prepared.data_dir, tmp_path / "run", *PLAIN_MLP_ONE_FOLD, *arguments, check=False)
    assert result.returncode == 2


def test_inference_rebuilds_the_presence_columns_of_a_dropout_checkpoint(synthetic_prepared, trained_once):
    import infer_lstm_two_feet_npz as infer

    saved = checkpoint(trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, *DROPOUT))
    trial = next(synthetic_prepared.data_dir.glob("T01_*.npz"))
    metadata = json.loads((synthetic_prepared.data_dir / "metadata.json").read_text(encoding="utf-8"))
    acquisition, _ = infer.load_acquisition(trial, saved, metadata)
    dataset = infer.InferenceWindowDataset(
        acquisition, np.asarray(saved["x_mean"]), np.asarray(saved["x_std"]), saved["window_samples"], True
    )
    x, _ = dataset[0]
    names = saved["feature_names"]
    presence = {name: float(x[0, names.index(name)]) for name in names if name.startswith("presence_")}
    assert presence["presence_heel_acc"] == 1.0  # IMUs are never Inactive
    # The synthetic Gauges read N(0, 1): all Inactive (p99 < 5 N), so every gauge group reads 0 present.
    assert presence["presence_heel_gauges"] == 0.0


def test_without_presence_the_removed_groups_are_zeroed_and_nothing_else_changes():
    names = list(FEATURE_NAMES)
    sensor_dropout = engine.SensorDropout.from_feature_names(names, ["heel_gauges", "toes_acc"], presence=False)
    x = sensor_dropout.apply(np.ones((5, len(names)), dtype=np.float32))
    kept = engine.sensor_groups(names)["heel_gauges"] + engine.sensor_groups(names)["toes_acc"]
    assert sensor_dropout.presence_columns == ()
    assert (x[:, kept] == 1).all()
    assert (np.delete(x, kept, axis=1) == 0).all()
