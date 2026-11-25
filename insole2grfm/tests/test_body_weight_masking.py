"""Optional body weight: random masking with a missing flag, evaluation by mode (#28)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
import pytest
import torch

from synthetic_prepared import ONE_FOLD, PARTICIPANTS, checkpoint, write_synthetic_prepared

BODY_WEIGHTS_N = {
    subject: 600.0 + 50.0 * index + (150.0 if loaded else 0.0)
    for index, participant in enumerate(PARTICIPANTS)
    for subject, loaded in ((participant, False), (f"{participant}b", True))
}
MLP = ("--model-type", "mlp", "--layers", "2")
BW = ("--body-weight-input",)
MASKED = (*BW, "--body-weight-masking", "0.3")


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    return write_synthetic_prepared(tmp_path_factory.mktemp("weighted"), body_weights_n=BODY_WEIGHTS_N)


def metadata(out):
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def history(out):
    return json.loads((out / "training_history.json").read_text(encoding="utf-8"))["T01"]


def test_about_p_of_the_training_windows_see_the_default_with_the_flag(weighted, trained_once):
    out = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *MASKED)
    rows = history(out)
    for row in rows:
        assert row["train_body_weight_missing_fraction"] == pytest.approx(0.3, abs=0.05)
        # Validation uses the requested mode (provided by default): never masked.
        assert row["validation_body_weight_missing_fraction"] == 0.0
    saved = checkpoint(out)
    assert saved["feature_names"][-2:] == ["body_weight_N", "body_weight_missing"]
    # Default value = the training-set mean body weight (T04, T04b, T05, T05b; equal sample counts).
    train_mean = sum(BODY_WEIGHTS_N[s] for s in ("T04", "T04b", "T05", "T05b")) / 4
    for record in (saved, metadata(out)):
        assert record["body_weight_masking"] == 0.3
    assert saved["body_weight_default_N"] == pytest.approx(train_mean, rel=1e-6)
    assert metadata(out)["body_weight_default_N"] == {"T01": pytest.approx(train_mean, rel=1e-6)}


def test_validation_and_test_use_the_requested_mode(weighted, trained_once):
    out = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *MASKED, "--body-weight-mode", "missing")
    assert all(row["validation_body_weight_missing_fraction"] == 1.0 for row in history(out))
    frame = pd.read_csv(out / "predictions.csv")
    assert len(frame) and (frame["body_weight_mode"] == "missing").all()
    assert metadata(out)["body_weight_mode"] == "missing"


def test_masking_off_is_the_current_behaviour(weighted, trained_once):
    plain = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *BW)
    off = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *BW, "--body-weight-masking", "0")
    assert (off / "training_history.json").read_text() == (plain / "training_history.json").read_text()
    assert (off / "predictions.csv").read_text() == (plain / "predictions.csv").read_text()
    assert checkpoint(off)["feature_names"] == checkpoint(plain)["feature_names"]
    assert metadata(off)["body_weight_default_N"] is None


MODELS = {
    "mlp": MLP,
    "transformer": ("--model-type", "transformer", "--layers", "1", "--transformer-heads", "2"),
    "sensor_token": (
        "--model-type", "sensor_token", "--layers", "1", "--transformer-heads", "2",
        "--transformer-feedforward-size", "16",
    ),
}


@pytest.mark.parametrize("model_type", MODELS)
def test_every_model_type_reads_the_missing_flag(weighted, trained_once, model_type):
    out = trained_once(weighted.data_dir, *MODELS[model_type], *ONE_FOLD, *MASKED)
    saved = checkpoint(out, model_type)
    assert "body_weight_missing" in saved["feature_names"]
    assert np.isfinite(pd.read_csv(out / "predictions.csv").filter(like="pred_").to_numpy()).all()
    if model_type == "sensor_token":
        # The Readout term reads the body weight and the flag.
        assert saved["state_dict"]["body_weight_projection.weight"].shape[1] == 2


def run_modes_cli(run_dir, output_dir, *args):
    """``evaluate_body_weight_modes.py`` in-process (its ``main(argv)``)."""
    import evaluate_body_weight_modes

    assert evaluate_body_weight_modes.main([str(run_dir), "--output-dir", str(output_dir), "--device", "cpu", *args]) == 0
    return output_dir


def mode_predictions(out, mode):
    return pd.read_csv(out / mode / "predictions.csv")


def test_provided_mode_reproduces_the_training_runs_predictions(weighted, trained_once, tmp_path):
    run = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *MASKED)
    out = run_modes_cli(run, tmp_path / "modes")
    provided, engine = mode_predictions(out, "provided"), pd.read_csv(run / "predictions.csv")
    assert (provided["body_weight_mode"] == "provided").all()
    assert provided["test_batch"].equals(engine["test_batch"])
    assert np.allclose(provided["pred_Fz_L"], engine["pred_Fz_L"], atol=1e-4)
    # The declared body weight is the Subject's, as the run used it.
    subjects = provided["test_batch"].str.split("_").str[0]
    assert np.allclose(provided["body_weight_input_N"], subjects.map(BODY_WEIGHTS_N))
    assert (provided["body_weight_missing"] == 0).all()


def edited_run(run, tmp_path, edit):
    """Copy of run dir ``run`` whose checkpoint's state_dict went through ``edit(state, saved)``."""
    copy = tmp_path / "edited_run"
    shutil.copytree(run, copy)
    path = copy / "models" / "mlp_test_T01.pt"
    saved = torch.load(path, weights_only=False)
    edit(saved["state_dict"], saved)
    torch.save(saved, path)
    return copy


def test_a_masked_model_predicts_in_both_modes_and_ignores_the_declared_weight_when_missing(
    weighted, tmp_path_factory, trained_once, tmp_path
):
    heavier = write_synthetic_prepared(
        tmp_path_factory.mktemp("heavier"), body_weights_n={**BODY_WEIGHTS_N, "T01": 900.0, "T01b": 1100.0}
    )
    args = (*MLP, *ONE_FOLD, *MASKED, "--body-weight-mode", "missing")
    base, other = trained_once(weighted.data_dir, *args), trained_once(heavier.data_dir, *args)
    # Same training (the test weights never enter it); flag 1 at test: identical predictions.
    assert history(base) == history(other)
    assert pd.read_csv(base / "predictions.csv").equals(pd.read_csv(other / "predictions.csv"))
    out = run_modes_cli(trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *MASKED), tmp_path / "modes")
    provided, missing = mode_predictions(out, "provided"), mode_predictions(out, "missing")
    for frame in (provided, missing):
        assert len(frame) and np.isfinite(frame.filter(like="pred_").to_numpy()).all()
    assert (missing["body_weight_missing"] == 1).all()
    assert not np.allclose(provided["pred_Fz_L"], missing["pred_Fz_L"])


def test_the_modes_are_evaluated_side_by_side_even_for_a_model_trained_without_masking(
    weighted, trained_once, tmp_path
):
    out = run_modes_cli(trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *BW), tmp_path / "modes")
    modes = ["provided", "missing", "provided-1kg", "provided+1kg", "provided-2kg", "provided+2kg",
             "provided-5kg", "provided+5kg"]
    # No flag column: "missing" feeds the default value only.
    missing = mode_predictions(out, "missing")
    assert len(missing) and np.isfinite(missing.filter(like="pred_").to_numpy()).all()
    summary = pd.read_csv(out / "evaluation" / "summary.csv")
    assert set(summary["run"]) == set(modes)
    table = pd.read_csv(out / "modes_summary.csv")
    rmse = table[(table.target == "Fz_sum") & (table.metric == "rmse")].set_index("mode")
    assert list(rmse.index) == modes
    assert rmse.loc["provided", "delta_vs_provided"] == 0.0
    assert (rmse["mean"] > 0).all()
    report = (out / "body_weight_modes.md").read_text(encoding="utf-8")
    assert "without masking" in report and "provided+5kg" in report


def test_the_v37_named_plan_has_the_masked_body_weight_jobs():
    from test_overnight_queue import print_plan

    plan = json.loads(print_plan("--named-plan", "v3.7"))
    assert plan["split"] == {"test": ["P04", "P13"], "validation": ["P07"]}  # Fixed split v2
    masking = ["--body-weight-masking", "0.3"]
    assert plan["models"]["mlp_bw_mask"]["engine_args"] == [*plan["models"]["mlp_bw"]["engine_args"], *masking]
    assert plan["models"]["mlp_share_mask"]["engine_args"] == [
        *plan["models"]["mlp_share"]["engine_args"], *masking
    ]
    for name, model in (("mlp_bw_mask_continuous", "mlp_bw_mask"), ("mlp_share_mask_continuous", "mlp_share_mask")):
        assert {"name": name, "model": model, "task_set": "continuous"} in plan["jobs"]


def test_the_masking_probability_and_default_rule_are_in_the_queue_signature(weighted, tmp_path):
    from test_overnight_queue import run_queue, write_plan

    models = {
        name: {"model_type": "mlp", "engine_args": ["--layers", "1", "--body-weight-input", *extra], "baseline": True}
        for name, extra in (("bw", []), ("mask3", ["--body-weight-masking", "0.3"]),
                            ("mask5", ["--body-weight-masking", "0.5"]))
    }
    plan = write_plan(
        tmp_path / "plan.json", models=models,
        jobs=[{"name": f"{name}_continuous", "model": name, "task_set": "continuous"} for name in models],
    )
    root = tmp_path / "results"
    run_queue(weighted.data_dir, root, "--plan", str(plan), "--folds", "T01")
    configs = {
        name: json.loads((root / name / "continuous" / "queue_config.json").read_text(encoding="utf-8"))
        for name in models
    }
    assert len({config["config_signature"] for config in configs.values()}) == 3
    assert "body_weight_default=train_mean" not in configs["bw"]["config_args"]
    assert "body_weight_default=train_mean" in configs["mask3"]["config_args"]
    assert metadata(root / "mask5" / "continuous" / "fold_T01")["body_weight_masking"] == 0.5


def test_the_shift_changes_only_the_body_weight_input(weighted, trained_once, tmp_path):
    run = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *MASKED)
    out = run_modes_cli(run, tmp_path / "modes", "--shift-kg", "2")
    provided, heavier = mode_predictions(out, "provided"), mode_predictions(out, "provided+2kg")
    assert np.allclose(heavier["body_weight_input_N"] - provided["body_weight_input_N"], 2 * 9.81)
    assert not np.allclose(heavier["pred_Fz_L"], provided["pred_Fz_L"])

    def blind_to_body_weight(state, saved):
        # Zero the first layer's weights on the body-weight column of every window sample.
        names = saved["feature_names"]
        first = state["network.1.weight"].view(state["network.1.weight"].shape[0], -1, len(names))
        first[:, :, names.index("body_weight_N")] = 0.0

    blind = run_modes_cli(edited_run(run, tmp_path, blind_to_body_weight), tmp_path / "blind", "--shift-kg", "2")
    for mode in ("provided-2kg", "provided+2kg"):
        shifted = mode_predictions(blind, mode)
        assert shifted.filter(like="pred_").equals(mode_predictions(blind, "provided").filter(like="pred_"))
    assert sorted(path.name for path in blind.iterdir() if (path / "predictions.csv").exists()) == [
        "missing", "provided", "provided+2kg", "provided-2kg",
    ]


def test_total_share_with_the_body_weight_masked_is_anchored_on_the_default(weighted, trained_once, tmp_path):
    run = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *MASKED, "--vertical-share")
    saved = checkpoint(run)
    vertical = [saved["targets"].index(target) for target in ("Fz_L", "Fz_R")]

    def zero_vertical_outputs(state, _):
        # Raw a_L = a_R = 0: total = the body weight the model is given, shared equally.
        state["base.network.9.weight"][vertical] = 0.0
        state["base.network.9.bias"][vertical] = 0.0

    out = run_modes_cli(edited_run(run, tmp_path, zero_vertical_outputs), tmp_path / "modes", "--shift-kg")
    missing, provided = mode_predictions(out, "missing"), mode_predictions(out, "provided")
    assert np.allclose(missing["pred_vertical_total_N"], saved["body_weight_default_N"], rtol=1e-5)
    assert np.allclose(missing["body_weight_input_N"], saved["body_weight_default_N"])
    subjects = provided["test_batch"].str.split("_").str[0]
    assert np.allclose(provided["pred_vertical_total_N"], subjects.map(BODY_WEIGHTS_N), rtol=1e-5)
