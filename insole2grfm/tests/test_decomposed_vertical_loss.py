"""Decomposed total/share vertical loss with dedicated heads, and the sum-loss-only diagnostic (#29)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from grfm_schema import BODY_WEIGHT_FEATURE
from synthetic_prepared import ONE_FOLD, TARGETS, checkpoint, run_training_cli, write_synthetic_prepared
from test_vertical_share import BODY_WEIGHTS_N, MLP, SHARE, history, metadata, predictions

DECOMPOSED = (*SHARE, "--vertical-loss", "decomposed")
SUM_ONLY = ("--body-weight-input", "--vertical-loss", "sum")
TASKS = [
    "Fx_L", "Fy_L", "vertical_total", "Mx_L", "My_L", "Mz_L",
    "Fx_R", "Fy_R", "vertical_share", "Mx_R", "My_R", "Mz_R",
]


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    return write_synthetic_prepared(tmp_path_factory.mktemp("weighted"), body_weights_n=BODY_WEIGHTS_N)


def test_decomposed_heads_give_feet_that_sum_to_the_total_with_twelve_gradnorm_tasks(weighted, trained_once):
    out = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *DECOMPOSED, "--gradnorm")
    frame = predictions(out)
    total, share = frame["pred_vertical_total_N"], frame["pred_vertical_share_L"]
    assert len(frame)
    assert np.allclose(frame["pred_Fz_L"] + frame["pred_Fz_R"], total, rtol=1e-5, atol=1e-2)
    assert share.between(0.0, 1.0).all()
    # GradNorm balances the 10 shear/moment terms + total + share, and nothing else.
    weights = pd.read_csv(out / "gradnorm_weights.csv")
    assert list(weights.columns) == ["test_group", "epoch", *TASKS]
    rows = history(out)
    assert list(rows[-1]["gradnorm_weights"]) == TASKS
    assert sum(rows[-1]["gradnorm_weights"].values()) == pytest.approx(12.0)
    # No separate sum term in this mode; the two decomposed terms are logged instead.
    assert "train_vertical_sum_loss" not in rows[0]
    for split in ("train", "validation"):
        assert all(np.isfinite(row[f"{split}_vertical_total_loss"]) for row in rows)
        assert all(np.isfinite(row[f"{split}_vertical_share_loss"]) for row in rows)
    # Dedicated, named output units: two more than the 12 targets.
    saved = checkpoint(out)
    assert saved["vertical_loss"] == "decomposed"
    assert saved["vertical_heads"] == ["vertical_total_delta", "vertical_share_offset"]
    assert saved["state_dict"]["base.network.9.weight"].shape[0] == 14
    scales = saved["vertical_loss_scales_N"]
    assert scales["sigma_sum"] > 0.0 and scales["sigma_share"] > 0.0
    run = metadata(out)
    assert run["vertical_loss"] == "decomposed"
    assert run["vertical_loss_scales_N"]["T01"] == scales


def test_an_airborne_foot_gets_share_zero_with_the_decomposed_loss(tmp_path_factory, trained_once):
    # Jump holds single support (left Airborne, right Measured): s* = 0, total* = the right foot's Fz.
    data = write_synthetic_prepared(
        tmp_path_factory.mktemp("single_support"), body_weights_n=BODY_WEIGHTS_N, one_foot_airborne=True
    )
    out = trained_once(
        data.data_dir, *MLP, *DECOMPOSED, "--causal", "--split-level", "subject_identity",
        "--only-test-group", "T01", "--epochs", "20", "--patience", "20",
        "--learning-rate", "1e-2", "--zero-target-weight", "1.0",
    )
    frame = predictions(out)
    single = frame[(frame.state_L == 2) & (frame.state_R == 1)]
    zero = single[single["pred_vertical_share_L"] == 0.0]
    assert len(single) and len(zero) >= 5
    assert np.allclose(zero["pred_Fz_L"], 0.0, atol=1e-2)
    assert np.allclose(zero["pred_Fz_R"], zero["pred_vertical_total_N"], rtol=1e-5)
    assert history(out)[-1]["train_vertical_share_saturated_fraction"] > 0.0


class PerSampleRaw(torch.nn.Module):
    """A 'model' whose raw outputs (12 targets + the 2 dedicated heads) are one trainable row per sample."""

    def __init__(self, rows):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.zeros(rows, len(TARGETS) + 2))

    def forward(self, x):
        return self.raw


LEFT, RIGHT = TARGETS.index("Fz_L"), TARGETS.index("Fz_R")
DELTA, OFFSET = len(TARGETS), len(TARGETS) + 1
# (Fz_L*, Fz_R*, left weight, right weight): weight 0 = Masked; an Airborne foot has Fz* = 0, weight 1.
CASES = {
    "both_measured": (400.0, 200.0, 1.0, 1.0),
    "both_measured_light": (20.0, 20.0, 1.0, 1.0),
    "left_airborne": (0.0, 500.0, 1.0, 1.0),
    "left_masked": (0.0, 300.0, 0.0, 1.0),
    "both_airborne": (0.0, 0.0, 1.0, 1.0),
}


def decomposed_terms(cases):
    """Model (body weight 600 N, delta = 0, share offset = 0, unit statistics), loss terms of ``cases``."""
    from train_lstm_two_feet_masked import DecomposedVerticalLoss, VerticalShareModel

    rows = [CASES[name] for name in cases]
    model = VerticalShareModel(PerSampleRaw(len(rows)), ["gauge", BODY_WEIGHT_FEATURE], TARGETS, decomposed=True)
    model.set_vertical_share_statistics([0.0, 0.0], [1.0, 1.0], [0.0] * len(TARGETS), [1.0] * len(TARGETS))
    x = torch.tensor([[[0.0, 600.0]]] * len(rows))
    y = torch.zeros(len(rows), len(TARGETS))
    weight = torch.ones(len(rows), len(TARGETS))
    for row, (left, right, left_weight, right_weight) in enumerate(rows):
        y[row, LEFT], y[row, RIGHT] = left, right
        weight[row, LEFT], weight[row, RIGHT] = left_weight, right_weight
    loss = DecomposedVerticalLoss([LEFT, RIGHT], [0.0, 0.0], [1.0, 1.0], sigma_sum=1.0, sigma_share=1.0)
    prediction = model(x)
    numerators, denominators = loss.terms(model, prediction, y, weight)
    return model, numerators, denominators


def test_the_share_term_reads_only_feet_with_a_valid_target_and_a_total_above_50_n():
    model, numerators, denominators = decomposed_terms(list(CASES))
    assert numerators.shape == denominators.shape == (12,)
    # Predicted: total = 600 N, share = 0.5 everywhere. Share term on both_measured
    # ((0.5 - 2/3)^2 * 600^2 = 10000) and left_airborne (s* = 0: 0.5^2 * 500^2 = 62500) only:
    # not below 50 N, not with a Masked foot, not with both feet Airborne.
    assert float(denominators[RIGHT]) == 2.0
    assert float(numerators[RIGHT]) == pytest.approx(72500.0)
    # Total term on the 4 samples with both feet valid ((600 - total*)^2: 0, 560^2, 100^2, 600^2),
    # plus the left_masked right foot's own term ((1 - 0.5) * 600 - 300)^2 = 0.
    assert float(denominators[LEFT]) == 5.0
    assert float(numerators[LEFT]) == pytest.approx(313600.0 + 10000.0 + 360000.0)
    # The 10 shear/moment terms are the ordinary per-target terms (zero error, weight 1 per sample).
    others = [index for index in range(12) if index not in (LEFT, RIGHT)]
    assert torch.equal(denominators[others], torch.full((10,), 5.0))
    assert torch.equal(numerators[others], torch.zeros(10))


def gradient(term, cases):
    model, numerators, _ = decomposed_terms(cases)
    numerators[term].backward()
    return model.base.raw.grad


def test_total_and_share_heads_are_trained_by_separate_gradient_paths():
    valid = ["both_measured", "both_measured_light", "left_airborne", "both_airborne"]
    share_gradient = gradient(RIGHT, valid)
    assert torch.count_nonzero(share_gradient[:, OFFSET]) == 2  # the two share-term samples
    assert torch.equal(share_gradient[:, DELTA], torch.zeros(len(valid)))  # total* is a constant weight
    total_gradient = gradient(LEFT, valid)
    assert torch.count_nonzero(total_gradient[:, DELTA]) == 3  # both_measured has zero total error
    assert torch.equal(total_gradient[:, OFFSET], torch.zeros(len(valid)))
    # The base model's own Fz_L, Fz_R units are unused: no gradient from either term.
    for gradient_ in (share_gradient, total_gradient):
        assert torch.equal(gradient_[:, [LEFT, RIGHT]], torch.zeros(len(valid), 2))


def test_the_overfit_gate_trains_the_decomposed_objective(weighted, tmp_path):
    out = tmp_path / "overfit"
    result = run_training_cli(
        weighted.data_dir, out, *MLP, *DECOMPOSED, "--overfit-test", "--causal", "--split-level",
        "subject_identity", "--overfit-windows-per-stratum", "16", "--overfit-epochs", "2", check=False,
    )
    assert result.returncode in (0, 3), result.stdout + result.stderr  # pass or fail verdict, no crash
    assert list(pd.read_csv(out / "gradnorm_weights.csv").columns) == ["test_group", "epoch", *TASKS]
    verdict = json.loads((out / "overfit_verdict.json").read_text(encoding="utf-8"))
    assert list(verdict["final_gradnorm_weights"]) == TASKS
    assert list(verdict["final_nmse"]) == TARGETS  # the gate still scores the 12 reported outputs


@pytest.mark.parametrize("gradnorm", [(), ("--gradnorm",)], ids=["plain", "gradnorm"])
def test_the_sum_diagnostic_adds_the_sum_term_to_a_plain_model(weighted, trained_once, gradnorm):
    plain = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, "--body-weight-input", *gradnorm)
    summed = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *SUM_ONLY, *gradnorm)
    rows = history(summed)
    assert all(np.isfinite(row["train_vertical_sum_loss"]) for row in rows)
    assert all(np.isfinite(row["validation_vertical_sum_loss"]) for row in rows)
    # Same seed and data: training differs only through the sum term's gradient.
    assert rows[-1]["train_loss"] != history(plain)[-1]["train_loss"]
    # No reparametrisation: no total / share outputs, no saturation counts, a plain state dict.
    assert not {"pred_vertical_total_N", "pred_vertical_share_L"} & set(predictions(summed))
    assert "train_vertical_share_saturated_fraction" not in rows[0]
    saved = checkpoint(summed)
    assert saved["vertical_share"] is False and saved["vertical_loss"] == "sum"
    assert not any(key.startswith("base.") for key in saved["state_dict"])
    assert metadata(summed)["vertical_loss"] == "sum"
    if gradnorm:
        assert list(history(summed)[-1]["gradnorm_weights"]) == TARGETS  # still the 12 targets


def test_the_vertical_loss_is_off_by_default_and_unchanged(weighted, trained_once):
    for args in ((*MLP, *ONE_FOLD, *SHARE, "--gradnorm"), (*MLP, *ONE_FOLD, "--body-weight-input")):
        out = trained_once(weighted.data_dir, *args)
        same = trained_once(weighted.data_dir, *args, "--vertical-loss", "default")
        assert metadata(out)["vertical_loss"] == "default"
        assert metadata(out)["vertical_loss_scales_N"] is None
        assert checkpoint(out)["vertical_loss"] == "default"
        assert checkpoint(out)["vertical_heads"] is None
        assert history(out) == history(same)
        assert predictions(out).equals(predictions(same))
        assert "train_vertical_total_loss" not in history(out)[0]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (("--body-weight-input", "--vertical-loss", "decomposed"), "--vertical-share"),
        (("--body-weight-input", "--vertical-share", "--vertical-loss", "sum"), "--vertical-share"),
        (("--model-type", "sensor_token", *DECOMPOSED), "sensor_token"),
    ],
    ids=["decomposed_without_share", "sum_with_share", "decomposed_sensor_token"],
)
def test_invalid_vertical_loss_combinations_are_refused(weighted, tmp_path, args, message):
    result = run_training_cli(weighted.data_dir, tmp_path / "run", *MLP, *ONE_FOLD, *args, check=False)
    assert result.returncode != 0
    assert message in result.stderr
    assert not (tmp_path / "run" / "metadata.json").exists()


def test_v37_named_plan_has_the_decomposed_and_sum_loss_jobs():
    from test_overnight_queue import print_plan

    plan = json.loads(print_plan("--named-plan", "v3.7"))
    models = plan["models"]
    assert models["mlp_share_decomposed"]["engine_args"] == [
        *models["mlp_share"]["engine_args"], "--vertical-loss", "decomposed"
    ]
    assert models["mlp_bw_sumloss"]["engine_args"] == [*models["mlp_bw"]["engine_args"], "--vertical-loss", "sum"]
    for model in ("mlp_share_decomposed", "mlp_bw_sumloss"):
        assert models[model]["model_type"] == "mlp"
        for task_set in ("continuous", "all"):
            assert {"name": f"{model}_{task_set}", "model": model, "task_set": task_set} in plan["jobs"]
    assert plan["split"] == {"test": ["P04", "P13"], "validation": ["P07"]}
