"""One Full CV fold of a queue plan per Slurm array task (#45)."""

from __future__ import annotations

import json

import pytest

import full_cv_fold
from overnight_queue import fold_is_done


def write_plan(path, **overrides):
    plan = {
        "split": "full_cv",
        "folds": ["T01", "T03"],
        "common_args": ["--causal", "--split-level", "subject_identity", "--validation-group-count", "1",
                        "--epochs", "3", "--batch-size", "64", "--no-plots"],
        "models": {"mlp_small": {"model_type": "mlp", "engine_args": ["--layers", "1", "--epochs", "1"]}},
        "jobs": [{"name": "mlp_small_all", "model": "mlp_small", "task_set": "all"}],
        **overrides,
    }
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_the_fold_is_trained_in_the_queue_tree_with_model_args_last(synthetic_prepared, tmp_path, capsys):
    plan = write_plan(tmp_path / "plan.json")
    root = tmp_path / "results"
    args = [str(plan), str(synthetic_prepared.data_dir), "--results-root", str(root), "--fold-index", "1",
            "--device", "cpu"]
    assert full_cv_fold.main(args) == 0
    fold_dir = root / "mlp_small" / "all" / "fold_T03"
    assert fold_is_done(fold_dir, "mlp", "T03")
    command = json.loads((fold_dir / "fold_command.json").read_text(encoding="utf-8"))
    assert command[-2:] == ["--only-test-group", "T03"]
    assert command.index("--epochs", command.index("--epochs") + 1) > command.index("--no-plots")
    assert full_cv_fold.main(args) == 0
    assert "already trained, skipped" in capsys.readouterr().out


def test_fold_index_0_is_the_first_participant_of_the_plan(tmp_path):
    plan = write_plan(tmp_path / "plan.json")
    command, fold_dir, _ = full_cv_fold.fold_command(
        full_cv_fold.load_plan(plan), tmp_path, tmp_path / "results", 0, "cpu"
    )
    assert fold_dir == tmp_path / "results" / "mlp_small" / "all" / "fold_T01"
    assert "--device" in command


@pytest.mark.parametrize("overrides, index", [({"split": {"test": ["T01"], "validation": ["T03"]}}, 0), ({}, 2)])
def test_a_fixed_split_plan_or_an_index_outside_the_folds_is_rejected(tmp_path, overrides, index):
    plan = write_plan(tmp_path / "plan.json", **overrides)
    with pytest.raises(SystemExit):
        full_cv_fold.main([str(plan), str(tmp_path), "--results-root", str(tmp_path / "r"),
                           "--fold-index", str(index), "--device", "cpu"])
