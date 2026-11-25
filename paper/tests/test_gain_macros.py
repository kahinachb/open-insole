"""Static load ratio macros (make_artefacts.characterisation_macros)."""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_artefacts as ma  # noqa: E402


def test_the_static_load_ratio_gives_its_quartiles_and_its_median_per_foot(tmp_path):
    gain = tmp_path / "gain"
    gain.mkdir()
    (gain / "summary.json").write_text(json.dumps({
        "trials": 4, "gain_total": {"median": 0.97}, "fit_total": {"residual_median_abs_N": 25.0},
    }))
    pd.DataFrame({"gain_total": [0.24, 0.94, 1.00, 1.04]}).to_csv(gain / "static_gain.csv", index=False)
    pd.DataFrame({"foot": ["L", "L", "R", "R"], "gain": [0.98, 1.00, 0.94, 0.96]}).to_csv(
        gain / "static_gain_per_foot.csv", index=False)

    macros = ma.characterisation_macros(tmp_path)

    assert r"\newcommand{\gainLowerQuartile}{0.76}" in macros
    assert r"\newcommand{\gainUpperQuartile}{1.01}" in macros
    assert r"\newcommand{\gainMedianLeft}{0.99}" in macros
    assert r"\newcommand{\gainMedianRight}{0.95}" in macros


def test_the_clock_drift_gives_its_median_and_maximum_over_the_trials(tmp_path):
    drift = tmp_path / "drift"
    drift.mkdir()
    (drift / "summary.json").write_text(json.dumps({
        "trials": 3, "accumulated_drift_ms": {"median": 0.29, "max": 4.51, "p95": 1.47},
    }))

    macros = ma.characterisation_macros(tmp_path)

    assert r"\newcommand{\clockDriftMedianMs}{0.3}" in macros
    assert r"\newcommand{\clockDriftMaxMs}{4.5}" in macros


def test_the_reference_sensitivity_is_computed_from_its_assumptions():
    macros = ma.reference_sensitivity_macros()

    assert r"\newcommand{\sensitivityFxN}{12.2}" in macros  # 700 N x sin 1 degree
    assert r"\newcommand{\sensitivityMxNm}{1.4}" in macros  # 700 N x 2 mm


def test_the_rmse_of_each_component_and_foot_is_the_mean_over_participants(tmp_path):
    pd.DataFrame({
        "run": ["mlp"] * 4, "subset": ["measured"] * 4, "task_family": ["ALL"] * 4,
        "component": ["Fx", "Fx", "Mx", "Mx"], "foot": ["L", "L", "R", "R"], "participant": ["P01", "P02"] * 2,
        "rmse": [11.0, 12.0, 1.30, 1.32],
    }).to_csv(tmp_path / "per_participant.csv", index=False)

    macros = ma.per_foot_rmse_macros(tmp_path, "mlp")

    assert r"\newcommand{\rmseCompFxLeft}{11.5}" in macros
    assert r"\newcommand{\rmseCompMxRight}{1.31}" in macros


def test_the_spread_of_the_minimal_configuration_counts_the_participants_above_the_target(monkeypatch):
    import types

    import sensor_study as ss

    scores = {
        "full": pd.Series({"P01": 7.0, "P02": 7.0, "P03": 7.0}),
        "minimal_28ch": pd.Series({"P01": 7.5, "P02": 7.8, "P03": 7.9}),  # +7.1, +11.4, +12.9 %
    }
    monkeypatch.setattr(ss, "method_score", lambda directory, run: types.SimpleNamespace(per_participant_pct=scores[run]))

    macros = ss.minimal_spread_macros(Path("unused"))

    assert r"\newcommand{\minimalOverTargetCount}{two}" in macros
    assert r"\newcommand{\minimalMaxChangePct}{12.9}" in macros
    assert r"\newcommand{\minimalMaxChangeCode}{P03}" in macros
