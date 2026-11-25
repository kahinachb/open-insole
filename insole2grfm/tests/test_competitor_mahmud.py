"""Mahmud 2026's Ke2KeNet, ported to PyTorch and retrained under our Full CV, offline as published (#47)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

import competitor_folds as folds
import competitor_mahmud as mahmud
import evaluate_grfm
import ke2kenet
from synthetic_prepared import PARTICIPANTS, write_synthetic_prepared

# Small network and segments: the synthetic trials hold 120 samples.
SMALL = ("--window", "32", "--train-stride", "16", "--depth", "2", "--width", "2", "--epochs", "2", "--device", "cpu")


@pytest.fixture(scope="module")
def run(synthetic_prepared, tmp_path_factory):
    out = tmp_path_factory.mktemp("mahmud") / "all"
    assert mahmud.main([str(synthetic_prepared.data_dir), "--inputs", "gauges", "--output-dir", str(out),
                        "--participant", "T01", *SMALL]) == 0
    return out / "fold_T01"


def test_the_port_has_the_published_architecture_and_resolutions():
    model = ke2kenet.build(16)
    heads = model(torch.zeros(1, 16, 1024))
    # Deep supervision: the bottleneck and 4 decoder levels, then the full-resolution output.
    assert [head.shape[-1] for head in heads] == [32, 64, 128, 256, 512, 1024]
    # The authors' code at d = 5, n = 32 has about 57 M parameters; the paper's 5.22 M is not reproducible.
    assert ke2kenet.parameter_count(model) == pytest.approx(57.1e6, rel=0.01)


@pytest.mark.parametrize("stride", [1, 2, 4, 8, 16])
def test_keras_same_padding_lengths(stride):
    x = torch.randn(2, 3, 64)
    assert ke2kenet.same_pool(x, stride, "max").shape[-1] == -(-64 // stride)
    trans = ke2kenet.TransConv(5, stride)
    assert trans(torch.randn(2, 3, 8)).shape[-1] == 8 * stride


def test_average_pooling_excludes_the_padding_like_keras():
    x = torch.ones(1, 1, 4)
    # Keras 'same' average pooling divides by the real samples only: a constant stays constant.
    assert torch.allclose(ke2kenet.same_pool(x, 1, "avg"), torch.ones(1, 1, 4))


def test_segments_cover_every_sample_and_never_cross_a_segment():
    assert mahmud.segment_starts(100, 32, 32, cover=True) == [0, 32, 64, 68]
    assert mahmud.segment_starts(100, 32, 32, cover=False) == [0, 32, 64]
    assert mahmud.segment_starts(20, 32, 32, cover=True) == [0]
    item = folds.Acquisition(
        name="t", evaluation_group="t", subject_session="A", subject_identity="A", task_family="balance",
        x=np.arange(80, dtype=np.float32)[:, None], y=np.zeros((80, 1), np.float32), mask=np.ones((80, 1), bool),
        time_s=np.arange(80) / 200, segment_id=np.r_[np.zeros(50), np.ones(30)].astype(int),
        state_l=np.ones(80, np.int8), state_r=np.ones(80, np.int8),
        airborne_l=np.zeros(80, bool), airborne_r=np.zeros(80, bool),
    )
    x, _, m, origin = mahmud.windows([item], [0], 0, (np.zeros(1), np.ones(1)), (0.0, 1.0), 32, 32, True)
    for window, mask in zip(x, m):
        values = window[0][mask[0] > 0]
        # Every unmasked sample of a window belongs to one segment (0-49 or 50-79).
        assert (values < 50).all() or (values >= 50).all()


def test_the_fold_is_complete_offline_and_never_reads_its_test_participant(synthetic_prepared, run):
    acquisitions, *_ = folds.load_folds(synthetic_prepared.data_dir)
    predictions = pd.read_csv(run / "predictions.csv")
    for item in acquisitions:
        if item.subject_identity == "T01":
            assert (predictions["test_batch"] == item.name).sum() == len(folds.scored_indexes(item))
    assert predictions.filter(like="pred_").notna().all().all()
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["causal"] is False
    (fold,) = json.loads((run / "split_manifest.json").read_text(encoding="utf-8"))["folds"]
    assert fold["validation_groups"] == ["T03", "T04"] and fold["train_groups"] == ["T05"]


def test_normalisation_is_fitted_on_the_training_participants_only(synthetic_prepared, tmp_path, monkeypatch):
    acquisitions, *_, fold_list = folds.load_folds(synthetic_prepared.data_dir)
    train = folds.subset(acquisitions, folds.fold_of(fold_list, "T01"), "train")
    seen = []
    original = mahmud.minmax
    monkeypatch.setattr(mahmud, "minmax", lambda values: seen.append(len(values)) or original(values))
    assert mahmud.main([str(synthetic_prepared.data_dir), "--inputs", "raw", "--output-dir", str(tmp_path),
                        "--participant", "T01", "--targets", "Fz_L", *SMALL]) == 0
    index = folds.load_folds(synthetic_prepared.data_dir)[2].index("Fz_L")
    assert seen == [sum(len(item.x) for item in train), sum(int(item.mask[:, index].sum()) for item in train)]


def test_a_stopped_fold_resumes_without_retraining_its_finished_targets(synthetic_prepared, run, tmp_path, monkeypatch):
    fold = tmp_path / "all" / "fold_T01"
    fold.mkdir(parents=True)
    for path in (run / "targets").iterdir():
        (fold / "targets").mkdir(exist_ok=True)
        (fold / "targets" / path.name).write_bytes(path.read_bytes())
    monkeypatch.setattr(mahmud, "train_target", lambda *a, **k: pytest.fail("retrained a finished target"))
    assert mahmud.main([str(synthetic_prepared.data_dir), "--inputs", "gauges", "--output-dir", str(tmp_path / "all"),
                        "--participant", "T01", *SMALL]) == 0
    assert (fold / "predictions.csv").exists()


def test_the_gauges_variant_reads_the_predicted_foot_only(synthetic_prepared):
    _, metadata, *_ = folds.load_folds(synthetic_prepared.data_dir)
    names = metadata["feature_names"]
    assert all(names[c].startswith("L_") for c in mahmud.input_columns(names, "gauges", "Fz_L"))
    assert all(names[c].startswith("R_") for c in mahmud.input_columns(names, "gauges", "Mx_R"))
    assert len(mahmud.input_columns(names, "raw", "Fz_L")) == len(names)


def test_the_test_participant_is_read_only_to_be_predicted(synthetic_prepared, tmp_path, monkeypatch):
    calls = []
    original = mahmud.windows
    monkeypatch.setattr(mahmud, "windows", lambda items, *a: calls.append({i.subject_identity for i in items}) or original(items, *a))
    assert mahmud.main([str(synthetic_prepared.data_dir), "--inputs", "raw", "--output-dir", str(tmp_path),
                        "--participant", "T01", "--targets", "Fz_L", *SMALL]) == 0
    # Training, validation, then test windows: T01 only in the last, built after training.
    assert calls == [{"T05"}, {"T03", "T04"}, {"T01"}]


def test_evaluate_grfm_scores_the_run_unchanged(run, tmp_path):
    assert evaluate_grfm.main(["--run", f"mahmud={run.parent}", "--output-dir", str(tmp_path)]) == 0
    assert not pd.read_csv(tmp_path / "summary.csv").empty


def test_an_unknown_target_is_rejected(synthetic_prepared, tmp_path):
    with pytest.raises(SystemExit):
        mahmud.main([str(synthetic_prepared.data_dir), "--inputs", "raw", "--output-dir", str(tmp_path),
                     "--targets", "Fz_Left", *SMALL])
