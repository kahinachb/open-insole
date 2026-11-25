"""End-to-end behaviour of the training CLI on the synthetic prepared dataset."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from synthetic_prepared import ONE_FOLD, PLAIN_MLP_ONE_FOLD, checkpoint, run_training_cli

MLP = ("--model-type", "mlp", "--causal", "--layers", "2")


def participant(subject_or_batch: str) -> str:
    return subject_or_batch.split("_", 1)[0].removesuffix("b")


def test_mlp_full_cv_writes_outputs_with_disjoint_participants(synthetic_prepared, tmp_path):
    out = tmp_path / "run"
    run_training_cli(
        synthetic_prepared.data_dir, out, *MLP,
        "--split-level", "subject_identity", "--validation-group-count", "1",
    )

    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    folds = manifest["folds"]
    assert sorted(fold["test_group"] for fold in folds) == sorted(synthetic_prepared.participants)
    for fold in folds:
        subsets = [
            {participant(b) for b in fold[f"{subset}_batches"]}
            for subset in ("train", "validation", "test")
        ]
        assert all(subsets)
        assert not subsets[0] & subsets[1]
        assert not subsets[0] & subsets[2]
        assert not subsets[1] & subsets[2]

    predictions = pd.read_csv(out / "predictions.csv")
    assert set(predictions["test_group"]) == set(synthetic_prepared.participants)
    assert set(predictions["test_batch"]) == set(synthetic_prepared.batches)
    assert {"pred_Fz_L", "mask_Fz_R"} <= set(predictions.columns)
    assert set(predictions["state_L"]) == {0, 1, 2}

    metrics = pd.read_csv(out / "fold_metrics.csv")
    assert set(metrics["test_batch"]) == set(synthetic_prepared.batches)
    assert {"all_valid", "platform", "airborne_zero"} <= set(metrics["subset"])


def test_no_plots_flag_skips_figures(synthetic_prepared, trained_once):
    out = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD)  # --no-plots: fast test default
    assert (out / "predictions.csv").exists()
    assert not list(out.glob("plots/*.png"))


INPUT_NORM_MODELS = {
    "mlp": ("--model-type", "mlp", "--layers", "2"),
    "transformer": ("--model-type", "transformer", "--layers", "1", "--transformer-heads", "2"),
    "tcn": ("--model-type", "tcn", "--layers", "1"),
    "lstm": ("--model-type", "lstm"),
}


@pytest.mark.parametrize("model_args", INPUT_NORM_MODELS.values(), ids=INPUT_NORM_MODELS.keys())
def test_input_layer_norm_option_changes_the_model_and_is_recorded(synthetic_prepared, trained_once, model_args):
    # Auto-correction (#9) step "pre-norm / input LayerNorm" = --input-layer-norm.
    one_fold = (*model_args, *ONE_FOLD)  # for the MLP, the shared PLAIN_MLP_ONE_FOLD run
    runs = {name: trained_once(synthetic_prepared.data_dir, *one_fold, *extra)
            for name, extra in (("plain", ()), ("normed", ("--input-layer-norm",)))}
    model_type = model_args[1]
    metadata = json.loads((runs["normed"] / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["input_layer_norm"] is True
    normed, plain = checkpoint(runs["normed"], model_type), checkpoint(runs["plain"], model_type)
    assert normed["input_layer_norm"] is True and plain["input_layer_norm"] is False
    # Off by default (the default architecture is unchanged); on, it changes what the model predicts.
    assert json.loads((runs["plain"] / "metadata.json").read_text(encoding="utf-8"))["input_layer_norm"] is False
    predictions = [pd.read_csv(runs[name] / "predictions.csv")["pred_Fz_L"] for name in runs]
    assert not np.allclose(predictions[0], predictions[1])


FIXED_SPLIT = (
    "--split-level", "subject_identity",
    "--test-groups", "T01", "T03", "--validation-groups", "T04",
)


def test_fixed_split_trains_one_fold_with_the_named_participants(synthetic_prepared, trained_once):
    out = trained_once(synthetic_prepared.data_dir, *MLP, *FIXED_SPLIT)

    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "fixed"
    [fold] = manifest["folds"]
    assert fold["test_groups"] == ["T01", "T03"]
    assert fold["validation_groups"] == ["T04"]
    assert fold["train_groups"] == ["T05"]
    subsets = {
        subset: {participant(b) for b in fold[f"{subset}_batches"]}
        for subset in ("train", "validation", "test")
    }
    assert subsets == {"train": {"T05"}, "validation": {"T04"}, "test": {"T01", "T03"}}
    # Both Subjects (unloaded Name, loaded Name2) of every Participant stay together.
    assert fold["test_sessions"] == ["T01", "T01b", "T03", "T03b"]
    assert fold["validation_sessions"] == ["T04", "T04b"]
    assert fold["train_sessions"] == ["T05", "T05b"]

    # One model; predictions and metrics per test Participant.
    assert len(list((out / "models").glob("*.pt"))) == 1
    history = json.loads((out / "training_history.json").read_text(encoding="utf-8"))
    assert list(history) == [fold["test_group"]]
    predictions = pd.read_csv(out / "predictions.csv")
    assert set(predictions["test_group"]) == {"T01", "T03"}
    for row in predictions[["test_batch", "test_group"]].drop_duplicates().itertuples():
        assert participant(row.test_batch) == row.test_group


def test_fixed_split_is_selectable_by_its_fold_name(synthetic_prepared, tmp_path):
    out = tmp_path / "run"
    run_training_cli(synthetic_prepared.data_dir, out, *MLP, *FIXED_SPLIT, "--dry-run-split")
    [fold] = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))["folds"]

    run_training_cli(synthetic_prepared.data_dir, out, *MLP, *FIXED_SPLIT, "--only-test-group", fold["test_group"])
    assert (out / "models" / f"mlp_test_{fold['test_group']}.pt").is_file()


@pytest.mark.parametrize("split_args", [
    ("--test-groups", "T01", "--validation-groups", "T01"),  # overlap
    ("--test-groups", "T01", "--validation-groups", "T07"),  # unknown Participant
    ("--test-groups", "T01", "T03", "--validation-groups", "T04", "T05"),  # no training Participant
    ("--test-groups", "T01"),  # validation list missing
], ids=["overlap", "unknown", "no_training", "no_validation"])
def test_fixed_split_rejects_invalid_lists(synthetic_prepared, tmp_path, split_args):
    result = run_training_cli(
        synthetic_prepared.data_dir, tmp_path / "run", *MLP, "--split-level", "subject_identity",
        *split_args, "--dry-run-split", check=False,
    )
    assert result.returncode == 2


def test_fixed_split_needs_participant_level_groups(synthetic_prepared, tmp_path):
    result = run_training_cli(
        synthetic_prepared.data_dir, tmp_path / "run", *MLP,
        "--test-groups", "T01", "--validation-groups", "T04", "--dry-run-split", check=False,
    )
    assert result.returncode == 2


def test_without_fixed_split_the_manifest_is_full_cv(synthetic_prepared, tmp_path):
    out = tmp_path / "run"
    run_training_cli(synthetic_prepared.data_dir, out, *MLP, "--split-level", "subject_identity", "--dry-run-split")
    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "full_cv"
    assert len(manifest["folds"]) == len(synthetic_prepared.participants)
    assert all(fold["test_groups"] == [fold["test_group"]] for fold in manifest["folds"])
