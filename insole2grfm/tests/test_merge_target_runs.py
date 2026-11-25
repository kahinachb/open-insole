"""Per-target models: runs trained on disjoint targets merged into one 12-target run."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evaluate_grfm
import merge_target_runs as merge
from synthetic_prepared import PLAIN_MLP_ONE_FOLD

LEFT = ("Fx_L", "Fy_L", "Fz_L", "Mx_L", "My_L", "Mz_L")
RIGHT = ("Fx_R", "Fy_R", "Fz_R", "Mx_R", "My_R", "Mz_R")


def test_two_runs_on_disjoint_targets_merge_into_one_scorable_run(synthetic_prepared, trained_once, tmp_path):
    left = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, "--targets", *LEFT)
    right = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, "--targets", *RIGHT)
    out = tmp_path / "merged"
    assert merge.main([str(left), str(right), "--output-dir", str(out)]) == 0
    (fold,) = [path for path in out.iterdir()]
    merged = pd.read_csv(fold / "predictions.csv")
    for run, targets in ((left, LEFT), (right, RIGHT)):
        source = pd.read_csv(next(run.rglob("predictions.csv")))
        joined = source[merge.KEYS + [f"pred_{t}" for t in targets]].merge(merged, on=merge.KEYS, suffixes=("", "_m"))
        assert len(joined) == len(source)
        for target in targets:
            assert np.allclose(joined[f"pred_{target}"], joined[f"pred_{target}_m"], equal_nan=True)
    assert evaluate_grfm.main(["--run", f"per_target={out}", "--output-dir", str(tmp_path / "eval")]) == 0


def test_a_target_predicted_twice_or_never_is_rejected(synthetic_prepared, trained_once, tmp_path):
    left = trained_once(synthetic_prepared.data_dir, *PLAIN_MLP_ONE_FOLD, "--targets", *LEFT)
    with pytest.raises(SystemExit):
        merge.main([str(left), str(left), "--output-dir", str(tmp_path / "twice")])
    with pytest.raises(SystemExit):
        merge.main([str(left), "--output-dir", str(tmp_path / "never")])
