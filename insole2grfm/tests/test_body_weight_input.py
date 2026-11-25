"""Subject body weight as an optional model input (#19)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from synthetic_prepared import ONE_FOLD, PARTICIPANTS, checkpoint, run_training_cli, write_synthetic_prepared

# Body weight per Subject (#16 metadata), as the v3.7 builder writes it.
BODY_WEIGHTS_N = {
    subject: 600.0 + 50.0 * index + (150.0 if loaded else 0.0)
    for index, participant in enumerate(PARTICIPANTS)
    for subject, loaded in ((participant, False), (f"{participant}b", True))
}
# Same dataset except the test Participant's body weights.
HEAVIER_TEST_N = {**BODY_WEIGHTS_N, "T01": 900.0, "T01b": 1100.0}

MODELS = {
    "mlp": ("--model-type", "mlp", "--layers", "2"),
    "transformer": ("--model-type", "transformer", "--layers", "1", "--transformer-heads", "2"),
    "sensor_token": (
        "--model-type", "sensor_token", "--layers", "1", "--transformer-heads", "2",
        "--transformer-feedforward-size", "16",
    ),
}


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    return write_synthetic_prepared(tmp_path_factory.mktemp("weighted"), body_weights_n=BODY_WEIGHTS_N)


@pytest.fixture(scope="module")
def heavier_test(tmp_path_factory):
    return write_synthetic_prepared(tmp_path_factory.mktemp("heavier"), body_weights_n=HEAVIER_TEST_N)


def metadata(out):
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def predictions(out):
    return pd.read_csv(out / "predictions.csv")


@pytest.mark.parametrize("model_type", MODELS)
def test_body_weight_input_trains_every_model_type_and_is_recorded(weighted, trained_once, model_type):
    out = trained_once(weighted.data_dir, *MODELS[model_type], *ONE_FOLD, "--body-weight-input")
    assert metadata(out)["body_weight_input"] is True
    # The Subjects' body weights the run used, in N.
    assert metadata(out)["body_weights_N"] == BODY_WEIGHTS_N
    saved = checkpoint(out, model_type)
    assert saved["body_weight_input"] is True
    assert "body_weight_N" in saved["feature_names"]
    assert len(predictions(out))


@pytest.mark.parametrize("model_type", MODELS)
def test_predictions_change_with_the_test_subjects_body_weight(weighted, heavier_test, trained_once, model_type):
    # Only the test Participant's body weight differs: training is identical, the test inputs are not.
    args = (*MODELS[model_type], *ONE_FOLD, "--body-weight-input")
    base, heavier = trained_once(weighted.data_dir, *args), trained_once(heavier_test.data_dir, *args)
    history = [json.loads((out / "training_history.json").read_text())["T01"] for out in (base, heavier)]
    assert history[0] == history[1]
    first, second = predictions(base), predictions(heavier)
    assert first["time_s"].equals(second["time_s"])
    assert not np.allclose(first["pred_Fz_L"], second["pred_Fz_L"])


def test_body_weight_is_standardised_on_the_training_participants_only(weighted, heavier_test, trained_once):
    args = (*MODELS["mlp"], *ONE_FOLD, "--body-weight-input")
    saved = [checkpoint(trained_once(data.data_dir, *args), "mlp") for data in (weighted, heavier_test)]
    column = saved[0]["feature_names"].index("body_weight_N")
    # Train = T04, T04b, T05, T05b with equal sample counts (fixture: 120 samples per trial).
    train_weights = [BODY_WEIGHTS_N[s] for s in ("T04", "T04b", "T05", "T05b")]
    assert saved[0]["x_mean"][column] == pytest.approx(np.mean(train_weights), rel=1e-6)
    assert saved[0]["x_std"][column] == pytest.approx(np.std(train_weights), rel=1e-5)
    # The test Participant's weights never reach the statistics.
    assert np.array_equal(saved[0]["x_mean"], saved[1]["x_mean"])
    assert np.array_equal(saved[0]["x_std"], saved[1]["x_std"])


def test_without_the_option_the_body_weights_are_ignored(weighted, synthetic_prepared, trained_once):
    # Off by default: a dataset with body weights trains exactly like one without.
    args = (*MODELS["mlp"], *ONE_FOLD)
    off, plain = trained_once(weighted.data_dir, *args), trained_once(synthetic_prepared.data_dir, *args)
    assert metadata(off)["body_weight_input"] is False
    assert metadata(off)["body_weights_N"] is None
    assert "body_weight_N" not in checkpoint(off, "mlp")["feature_names"]
    assert (off / "training_history.json").read_text() == (plain / "training_history.json").read_text()
    assert predictions(off).equals(predictions(plain))


def test_body_weight_input_needs_every_subjects_body_weight(tmp_path):
    missing = {subject: value for subject, value in BODY_WEIGHTS_N.items() if subject != "T05b"}
    data = write_synthetic_prepared(tmp_path / "data", body_weights_n=missing)
    result = run_training_cli(
        data.data_dir, tmp_path / "run", *MODELS["mlp"], *ONE_FOLD, "--body-weight-input", check=False
    )
    assert result.returncode != 0
    assert "T05b" in result.stderr
    assert not (tmp_path / "run" / "metadata.json").exists()


def test_body_weight_input_changes_the_queue_config_signature(weighted, tmp_path):
    from test_overnight_queue import run_queue, write_plan

    plan = write_plan(
        tmp_path / "plan.json",
        models={
            "mlp": {"model_type": "mlp", "engine_args": ["--layers", "1"], "baseline": True},
            "mlp_bw": {
                "model_type": "mlp", "engine_args": ["--layers", "1", "--body-weight-input"], "baseline": True,
            },
        },
        jobs=[
            {"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"},
            {"name": "mlp_bw_continuous", "model": "mlp_bw", "task_set": "continuous"},
        ],
    )
    root = tmp_path / "results"
    run_queue(weighted.data_dir, root, "--plan", str(plan), "--folds", "T01")
    configs = {
        model: json.loads((root / model / "continuous" / "queue_config.json").read_text(encoding="utf-8"))
        for model in ("mlp", "mlp_bw")
    }
    assert configs["mlp"]["config_signature"] != configs["mlp_bw"]["config_signature"]
    assert metadata(root / "mlp" / "continuous" / "fold_T01")["body_weight_input"] is False
    assert metadata(root / "mlp_bw" / "continuous" / "fold_T01")["body_weight_input"] is True
