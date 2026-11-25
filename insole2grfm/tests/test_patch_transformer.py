"""Patch Transformer (#31): temporal patch tokens, one Readout token per foot, conditioned queries."""

from __future__ import annotations

import json

import pytest
import torch

import train_lstm_two_feet_masked as engine
from synthetic_prepared import ONE_FOLD, PARTICIPANTS, checkpoint, run_training_cli, write_synthetic_prepared

PATCH = (
    "--model-type", "patch_transformer", "--layers", "1", "--transformer-heads", "2",
    "--transformer-feedforward-size", "32", "--patch-samples", "4",
)

BODY_WEIGHTS_N = {
    subject: 600.0 + 50.0 * index + (150.0 if loaded else 0.0)
    for index, participant in enumerate(PARTICIPANTS)
    for subject, loaded in ((participant, False), (f"{participant}b", True))
}


@pytest.fixture(scope="module")
def weighted(tmp_path_factory):
    return write_synthetic_prepared(tmp_path_factory.mktemp("weighted"), body_weights_n=BODY_WEIGHTS_N)


def model(window: int = 21, patch: int = 5, condition=(), output_size: int = 12) -> engine.GRFMPatchTransformer:
    torch.manual_seed(0)
    return engine.GRFMPatchTransformer(
        input_size=6, window_samples=window, hidden_size=16, layers=1, dropout=0.0,
        output_size=output_size, heads=2, feedforward_size=32, patch_samples=patch,
        condition_columns=condition,
    ).eval()


def test_window_not_a_multiple_of_the_patch_is_edge_padded_on_the_left():
    network = model(window=21, patch=5)
    assert network.patch_count == 5
    x = torch.randn(3, 21, 6)
    padded = torch.cat((x[:, :1].expand(-1, 4, -1), x), dim=1)
    exact = model(window=25, patch=5)
    exact.load_state_dict(network.state_dict())
    assert torch.allclose(network(x), exact(padded), atol=1e-6)


def test_untrained_head_predicts_about_the_mean_standardised_target():
    output = model()(torch.randn(8, 21, 6))
    assert output.shape == (8, 12)
    assert output.abs().max() < 1e-2


def test_only_present_and_past_samples_matter_and_conditioning_reaches_the_output():
    network = model(condition=(4, 5))
    with torch.no_grad():
        network.head[-1].weight.normal_()  # make the head sensitive
    x = torch.randn(2, 21, 6)
    shorter = network(x[:, 1:])  # one sample less in the past: still a valid (shorter) window
    assert shorter.shape == (2, 12)
    conditioned = x.clone()
    conditioned[:, -1, 4] += 3.0
    assert not torch.allclose(network(x), network(conditioned))


def test_condition_columns_come_from_the_body_weight_inputs():
    names = ["g1", "g2", engine.BODY_WEIGHT_FEATURE, engine.BODY_WEIGHT_MISSING_FEATURE]
    built = engine.build_model("patch_transformer", 4, 10, 16, 1, 0.0, 12, transformer_heads=2,
                               feature_names=names, patch_samples=5)
    assert built.condition_columns == [2, 3]
    plain = engine.build_model("patch_transformer", 4, 10, 16, 1, 0.0, 12, transformer_heads=2,
                               feature_names=names[:2], patch_samples=5)
    assert plain.condition is None


def test_cli_trains_with_gradnorm_and_vertical_share_and_records_the_patch(weighted, trained_once):
    out = trained_once(
        weighted.data_dir, *PATCH, *ONE_FOLD, "--gradnorm",
        "--body-weight-input", "--vertical-share",
    )
    saved = checkpoint(out, "patch_transformer")
    assert saved["patch_samples"] == 4 and saved["decoder_layers"] == 1
    history = json.loads((out / "training_history.json").read_text(encoding="utf-8"))["T01"]
    assert len(history) == 2


def test_overfit_test_runs_on_the_patch_transformer(synthetic_prepared, tmp_path):
    out = tmp_path / "overfit"
    result = run_training_cli(
        synthetic_prepared.data_dir, out, *PATCH, *ONE_FOLD, "--overfit-test", "--overfit-epochs", "2",
        check=False,
    )
    assert result.returncode in (0, engine.OVERFIT_EXIT_FAIL)
    assert (out / "overfit_verdict.json").exists()


@pytest.mark.parametrize("arguments", [
    ("--patch-samples", "0"),
    ("--decoder-layers", "0"),
    ("--hidden-size", "9"),
])
def test_invalid_patch_transformer_arguments_are_rejected(synthetic_prepared, tmp_path, arguments):
    result = run_training_cli(synthetic_prepared.data_dir, tmp_path / "run", *PATCH, *ONE_FOLD, *arguments, check=False)
    assert result.returncode == 2
