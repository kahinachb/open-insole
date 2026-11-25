"""Total + left/right share parametrisation of the vertical force (#27)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from synthetic_prepared import ONE_FOLD, PARTICIPANTS, checkpoint, run_training_cli, write_synthetic_prepared

BODY_WEIGHTS_N = {
    subject: 600.0 + 50.0 * index + (150.0 if loaded else 0.0)
    for index, participant in enumerate(PARTICIPANTS)
    for subject, loaded in ((participant, False), (f"{participant}b", True))
}
MLP = ("--model-type", "mlp", "--layers", "2")
SHARE = ("--body-weight-input", "--vertical-share")


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    return write_synthetic_prepared(tmp_path_factory.mktemp("weighted"), body_weights_n=BODY_WEIGHTS_N)


def metadata(out):
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


def history(out):
    return json.loads((out / "training_history.json").read_text(encoding="utf-8"))["T01"]


def predictions(out):
    return pd.read_csv(out / "predictions.csv")


@pytest.mark.parametrize("gradnorm", [(), ("--gradnorm",)], ids=["plain", "gradnorm"])
def test_vertical_forces_sum_to_the_predicted_total_with_a_share_in_unit_interval(
    weighted, trained_once, gradnorm
):
    out = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, *SHARE, *gradnorm)
    frame = predictions(out)
    total, share = frame["pred_vertical_total_N"], frame["pred_vertical_share_L"]
    assert len(frame)
    assert np.allclose(frame["pred_Fz_L"] + frame["pred_Fz_R"], total, rtol=1e-5, atol=1e-2)
    assert np.allclose(frame["pred_Fz_L"], share * total, rtol=1e-5, atol=1e-2)
    assert share.between(0.0, 1.0).all()
    assert (total >= 0.0).all()
    # Anchored on body weight: the untrained total starts near the test Subject's weight.
    subjects = frame["test_batch"].str.split("_").str[0]
    ratio = total / subjects.map(BODY_WEIGHTS_N)
    assert ratio.median() == pytest.approx(1.0, abs=0.5)
    assert metadata(out)["vertical_share"] is True
    assert checkpoint(out)["vertical_share"] is True


def test_an_airborne_foot_can_get_zero_share(tmp_path_factory, trained_once):
    # Jump holds single support (left Airborne, right Measured): all the load must go to the right foot.
    data = write_synthetic_prepared(
        tmp_path_factory.mktemp("single_support"), body_weights_n=BODY_WEIGHTS_N, one_foot_airborne=True
    )
    out = trained_once(
        data.data_dir, *MLP, *SHARE, "--causal", "--split-level", "subject_identity",
        "--only-test-group", "T01", "--epochs", "20", "--patience", "20",
        "--learning-rate", "1e-2", "--zero-target-weight", "1.0",
    )
    frame = predictions(out)
    single = frame[(frame.state_L == 2) & (frame.state_R == 1)]
    zero = single[single["pred_vertical_share_L"] == 0.0]
    assert len(single) and len(zero) >= 5
    assert np.allclose(zero["pred_Fz_L"], 0.0, atol=1e-2)
    # Every epoch logs the share of samples on the clamps; the single-support samples saturate the share.
    rows = history(out)
    for split in ("train", "validation"):
        for part in ("total", "share"):
            assert all(0.0 <= row[f"{split}_vertical_{part}_saturated_fraction"] <= 1.0 for row in rows)
    assert rows[-1]["train_vertical_share_saturated_fraction"] > 0.0
    assert np.allclose(zero["pred_Fz_R"], zero["pred_vertical_total_N"], rtol=1e-5)


def with_vertical_truth_off_measured_feet(data_dir, value):
    """Rewrite Fz_L / Fz_R wherever that foot is not Measured (Airborne or Masked)."""
    metadata_ = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    columns = {side: metadata_["targets"].index(f"Fz_{side}") for side in ("L", "R")}
    for batch in metadata_["batches"]:
        path = data_dir / f"{batch}.npz"
        arrays = dict(np.load(path))
        for side, column in columns.items():
            arrays["y"][arrays[f"state_{side}"] != 1, column] = value
        np.savez(path, **arrays)


@pytest.mark.parametrize("gradnorm", [(), ("--gradnorm",)], ids=["plain", "gradnorm"])
def test_the_sum_loss_reads_only_samples_where_both_feet_are_measured(tmp_path_factory, trained_once, gradnorm):
    # With a zero Airborne weight, the per-target terms never read a non-Measured foot's Fz;
    # a sum term over single support or a Masked foot would.
    plain = write_synthetic_prepared(tmp_path_factory.mktemp("sum_plain"), body_weights_n=BODY_WEIGHTS_N)
    changed = write_synthetic_prepared(tmp_path_factory.mktemp("sum_changed"), body_weights_n=BODY_WEIGHTS_N)
    with_vertical_truth_off_measured_feet(changed.data_dir, 777.0)
    args = (*MLP, *ONE_FOLD, *SHARE, *gradnorm, "--zero-target-weight", "0.0", "--task-set", "all")
    first, second = trained_once(plain.data_dir, *args), trained_once(changed.data_dir, *args)
    assert history(first) == history(second)
    predicted = [column for column in predictions(first) if column.startswith("pred_")]
    assert predictions(first)[predicted].equals(predictions(second)[predicted])
    rows = history(first)
    assert all(np.isfinite(row["train_vertical_sum_loss"]) for row in rows)
    assert all(np.isfinite(row["validation_vertical_sum_loss"]) for row in rows)


def test_the_sum_loss_weight_changes_training(weighted, trained_once):
    args = (*MLP, *ONE_FOLD, *SHARE)
    on = history(trained_once(weighted.data_dir, *args))
    off = history(trained_once(weighted.data_dir, *args, "--vertical-sum-loss-weight", "0"))
    # Same seed and data: the GRFM losses differ only through the sum term's gradient.
    assert on[-1]["train_loss"] != off[-1]["train_loss"]
    assert on[-1]["validation_loss"] != off[-1]["validation_loss"]


def test_off_by_default_and_unchanged(weighted, trained_once):
    out = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, "--body-weight-input")
    assert metadata(out)["vertical_share"] is False
    assert checkpoint(out)["vertical_share"] is False
    assert not {"pred_vertical_total_N", "pred_vertical_share_L"} & set(predictions(out))
    assert "train_vertical_sum_loss" not in history(out)[0]
    # The sum loss weight alone changes nothing without the option.
    same = trained_once(weighted.data_dir, *MLP, *ONE_FOLD, "--body-weight-input", "--vertical-sum-loss-weight", "5")
    assert history(out) == history(same)
    assert predictions(out).equals(predictions(same))


def test_vertical_share_needs_the_body_weight_input(weighted, tmp_path):
    result = run_training_cli(weighted.data_dir, tmp_path / "run", *MLP, *ONE_FOLD, "--vertical-share", check=False)
    assert result.returncode != 0
    assert "--body-weight-input" in result.stderr
    assert not (tmp_path / "run" / "metadata.json").exists()


def test_v37_named_plan_has_the_mlp_share_job():
    from test_overnight_queue import print_plan

    plan = json.loads(print_plan("--named-plan", "v3.7"))
    share = plan["models"]["mlp_share"]
    assert share["model_type"] == "mlp"
    # Same MLP as the body-weight job, plus the parametrisation only.
    assert share["engine_args"] == [*plan["models"]["mlp_bw"]["engine_args"], "--vertical-share"]
    assert {"name": "mlp_share_continuous", "model": "mlp_share", "task_set": "continuous"} in plan["jobs"]


class FixedRaw(torch.nn.Module):
    """A 'model' whose raw outputs are one trainable vector, whatever the input."""

    def __init__(self, raw):
        super().__init__()
        self.raw = torch.nn.Parameter(torch.as_tensor([raw], dtype=torch.float32))

    def forward(self, x):
        return self.raw.expand(len(x), -1)


def share_gradient(raw_left, raw_right, true_left_n, true_right_n):
    """Gradient of the Fz squared error on the raw Fz_L, Fz_R outputs (body weight 600 N, unit stats)."""
    from grfm_schema import BODY_WEIGHT_FEATURE
    from train_lstm_two_feet_masked import VerticalShareModel
    from synthetic_prepared import TARGETS

    raw = [0.0] * len(TARGETS)
    left, right = TARGETS.index("Fz_L"), TARGETS.index("Fz_R")
    raw[left], raw[right] = raw_left, raw_right
    model = VerticalShareModel(FixedRaw(raw), ["gauge", BODY_WEIGHT_FEATURE], TARGETS)
    model.set_vertical_share_statistics([0.0, 0.0], [1.0, 1.0], [0.0] * len(TARGETS), [1.0] * len(TARGETS))
    x = torch.tensor([[[0.0, 600.0]]])
    output = model(x)
    loss = (output[0, left] - true_left_n) ** 2 + (output[0, right] - true_right_n) ** 2
    loss.backward()
    gradient = model.base.raw.grad[0]
    return output[0, [left, right]].detach().numpy(), gradient[[left, right]].numpy()


def test_a_saturated_wrong_prediction_still_gets_a_gradient():
    # Both feet predicted Airborne (total clamped to exactly 0) while both carry 300 N.
    forces, gradient = share_gradient(-3.0, -3.0, 300.0, 300.0)
    assert np.array_equal(forces, [0.0, 0.0])  # the forward pass keeps the exact zero
    assert (gradient < 0).all()  # descent raises a_L and a_R: the total grows
    # All the load predicted on the left foot (share clamped to exactly 1) while the right carries it all.
    forces, gradient = share_gradient(1.5, -1.5, 0.0, 600.0)
    assert forces[1] == 0.0
    assert gradient[0] > gradient[1]  # descent moves the share towards the right foot
