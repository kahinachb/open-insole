"""Behaviour of the literature-comparable evaluation command (evaluate_grfm.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluate_grfm import main
from grfm_schema import FORCES as SCHEMA_FORCES
from grfm_schema import STATE_AIRBORNE, STATE_MEASURED
from synthetic_prepared import TARGETS

FORCES = list(SCHEMA_FORCES)

SAMPLES = 200


def trial(name, participant, family, truth, offset, state_l=STATE_MEASURED, state_r=STATE_MEASURED):
    """Rows of one Trial: `truth(target, t)` gives the reference, prediction = truth + offset."""
    t = np.arange(SAMPLES)
    data = {
        "test_batch": name,
        "test_group": participant,
        "task_family": family,
        "time_s": t * 0.005,
        "state_L": state_l,
        "state_R": state_r,
        "airborne_label_L": int(state_l == STATE_AIRBORNE),
        "airborne_label_R": int(state_r == STATE_AIRBORNE),
    }
    for target in TARGETS:
        y = np.asarray(truth(target, t), dtype=float)
        data[target] = y
        data[f"pred_{target}"] = y + offset
        data[f"mask_{target}"] = 1
    return pd.DataFrame(data)


def wave(target, t):
    """Varying reference; Fz_L + Fz_R is constant at 600 N (the Standing weight)."""
    s = np.sin(t / 10.0)
    if target == "Fz_L":
        return 300.0 + 40.0 * s
    if target == "Fz_R":
        return 300.0 - 40.0 * s
    return 20.0 * s + 5.0 * np.cos(t / 7.0)


def write_run(root: Path, frames) -> Path:
    """One run = one or more fold dirs, each with a predictions.csv (engine output layout)."""
    by_fold = {}
    for frame in frames:
        by_fold.setdefault(frame["test_group"].iloc[0], []).append(frame)
    for participant, parts in by_fold.items():
        fold = root / f"fold_{participant}"
        fold.mkdir(parents=True)
        pd.concat(parts).to_csv(fold / "predictions.csv", index=False)
    return root


def evaluate(tmp_path, runs, *extra) -> Path:
    out = tmp_path / "evaluation"
    argv = []
    for name, path in runs.items():
        argv += ["--run", f"{name}={path}"]
    main(argv + ["--output-dir", str(out), *extra])
    return out


def per_participant(out: Path) -> pd.DataFrame:
    return pd.read_csv(out / "per_participant.csv")


def summary(out: Path) -> pd.DataFrame:
    return pd.read_csv(out / "summary.csv")


def test_constant_offset_gives_perfect_correlation_and_offset_only_error(tmp_path):
    run = write_run(
        tmp_path / "run",
        [
            trial("T02_balance", "T02", "balance", wave, offset=5.0),
            trial("T02_squat", "T02", "squat", wave, offset=5.0),
        ],
    )
    rows = per_participant(evaluate(tmp_path, {"mlp": run}))
    rows = rows[(rows.subset == "measured") & (rows.task_family == "ALL")]
    assert sorted(rows.target) == sorted(TARGETS)
    assert rows.rmse.to_numpy() == pytest.approx(5.0)
    assert rows.pcc_trial_mean.to_numpy() == pytest.approx(1.0)
    assert rows.pcc_pooled.to_numpy() == pytest.approx(1.0)
    assert rows.offset_share.to_numpy() == pytest.approx(1.0)
    assert rows.dynamic_share.to_numpy() == pytest.approx(0.0, abs=1e-9)


def test_force_rmse_in_percent_body_weight_uses_balance_trials(tmp_path):
    def heavy_squat(target, t):  # Fz_L + Fz_R = 1000 N outside balance: must not set the body weight
        return wave(target, t) + (200.0 if target.startswith("Fz") else 0.0)

    run = write_run(
        tmp_path / "run",
        [
            trial("T02_balance", "T02", "balance", wave, offset=6.0),
            trial("T02_squat", "T02", "squat", heavy_squat, offset=6.0),
        ],
    )
    out = evaluate(tmp_path, {"mlp": run})
    bw = pd.read_csv(out / "body_weight.csv")
    assert bw.set_index("subject").loc["T02", "body_weight_N"] == pytest.approx(600.0)
    rows = per_participant(out)
    rows = rows[(rows.subset == "measured") & (rows.task_family == "ALL")].set_index("target")
    assert rows.loc[FORCES, "rmse_pct_bw"].to_numpy() == pytest.approx(1.0)  # 6 N of 600 N
    assert rows.loc[[t for t in TARGETS if t not in FORCES], "rmse_pct_bw"].isna().all()


def test_measured_and_airborne_subsets_are_never_mixed(tmp_path):
    zero = lambda target, t: np.zeros_like(t, dtype=float)  # noqa: E731
    jump_airborne = trial("T02_jump", "T02", "jump", zero, offset=7.0, state_l=STATE_AIRBORNE, state_r=STATE_AIRBORNE)
    masked = trial("T02_walk", "T02", "walk", wave, offset=99.0, state_l=0, state_r=0)
    for target in TARGETS:
        masked[target] = np.nan
        masked[f"mask_{target}"] = 0
    run = write_run(
        tmp_path / "run",
        [trial("T02_balance", "T02", "balance", wave, offset=2.0), jump_airborne, masked],
    )
    rows = per_participant(evaluate(tmp_path, {"mlp": run}))
    pooled = rows[rows.task_family == "ALL"]
    measured = pooled[pooled.subset == "measured"]
    airborne = pooled[pooled.subset == "airborne"]
    assert measured.rmse.to_numpy() == pytest.approx(2.0)
    assert airborne.rmse.to_numpy() == pytest.approx(7.0)
    assert sorted(airborne.target) == sorted(TARGETS)
    assert set(rows[rows.subset == "measured"].task_family) == {"ALL", "balance"}
    assert set(rows[rows.subset == "airborne"].task_family) == {"ALL", "jump"}
    assert "walk" not in set(rows.task_family)  # Masked samples are never evaluated


def test_aggregation_is_per_participant_then_mean_and_sd(tmp_path):
    run = write_run(
        tmp_path / "run",
        [
            trial("T02_balance", "T02", "balance", wave, offset=1.0),
            trial("T03_balance", "T03", "balance", wave, offset=3.0),
            trial("T03_squat", "T03", "squat", wave, offset=3.0),  # more T03 samples must not weigh more
        ],
    )
    out = evaluate(tmp_path, {"mlp": run})
    rows = summary(out)
    rmse = rows[
        (rows.run == "mlp")
        & (rows.subset == "measured")
        & (rows.task_family == "ALL")
        & (rows.target == "Fz_L")
        & (rows.metric == "rmse")
    ].iloc[0]
    assert rmse["mean"] == pytest.approx(2.0)  # pooled samples would give sqrt((1 + 2 * 9) / 3)
    assert rmse["sd"] == pytest.approx(math.sqrt(2.0))
    assert rmse["n_participants"] == 2


def test_reference_range_and_variance_normalisations(tmp_path):
    ramp = lambda target, t: np.linspace(0.0, 1000.0, SAMPLES)  # noqa: E731
    run = write_run(tmp_path / "run", [trial("T02_squat", "T02", "squat", ramp, offset=9.9)])
    rows = per_participant(evaluate(tmp_path, {"mlp": run}))
    row = rows[(rows.subset == "measured") & (rows.task_family == "ALL") & (rows.target == "Fx_L")].iloc[0]
    # 0.5th..99.5th percentiles of a 0..1000 ramp span 990 N (linear interpolation); its variance is 1000^2 (n+1) / (12 (n-1))
    assert row.nrmse_range == pytest.approx(0.01)
    variance = 1000.0**2 * (SAMPLES + 1) / (12 * (SAMPLES - 1))
    assert row.nmse == pytest.approx(9.9**2 / variance)


def test_common_task_family_filter_restricts_every_run_to_shared_families(tmp_path):
    mlp = write_run(
        tmp_path / "mlp",
        [
            trial("T02_balance", "T02", "balance", wave, offset=1.0),
            trial("T02_squat", "T02", "squat", wave, offset=1.0),
        ],
    )
    transformer = write_run(
        tmp_path / "transformer",
        [
            trial("T02_balance", "T02", "balance", wave, offset=1.0),
            trial("T02_squat", "T02", "squat", wave, offset=1.0),
            trial("T02_walk", "T02", "walk", wave, offset=50.0),
        ],
    )
    runs = {"mlp": mlp, "transformer": transformer}

    unfiltered = summary(evaluate(tmp_path / "unfiltered", runs))
    assert "walk" in set(unfiltered[unfiltered.run == "transformer"].task_family)

    out = evaluate(tmp_path / "filtered", runs, "--common-task-families")
    rows = summary(out)
    for run in runs:
        assert set(rows[rows.run == run].task_family) == {"ALL", "balance", "squat"}
    pooled = rows[(rows.subset == "measured") & (rows.task_family == "ALL") & (rows.metric == "rmse")]
    assert pooled["mean"].to_numpy() == pytest.approx(1.0)
    report = json.loads((out / "evaluation.json").read_text())
    assert report["common_task_families"] == ["balance", "squat"]
    assert report["runs"]["transformer"]["task_families_present"] == ["balance", "squat", "walk"]
    assert report["runs"]["transformer"]["task_families_evaluated"] == ["balance", "squat"]


def test_sum_of_both_feet_forces_is_evaluated_where_both_feet_are_measured(tmp_path):
    def squat(target, t):  # the total vertical force varies (constant during the balance Trial)
        return wave(target, t) + (30.0 * np.cos(t / 5.0) if target == "Fz_R" else 0.0)

    run = write_run(
        tmp_path / "run",
        [
            trial("T02_balance", "T02", "balance", wave, offset=5.0),
            trial("T02_squat", "T02", "squat", squat, offset=5.0),
            # One foot in the air: no total wrench to compare, left out of the sum.
            trial("T02_lateral", "T02", "lateral_2supports", wave, offset=50.0, state_r=STATE_AIRBORNE),
        ],
    )
    rows = per_participant(evaluate(tmp_path, {"mlp": run}))
    both = rows[rows.subset == "both_feet_sum"]
    assert set(both.task_family) == {"ALL", "balance", "squat"}
    pooled = both[both.task_family == "ALL"].set_index("target")
    assert sorted(pooled.index) == ["Fx_sum", "Fy_sum", "Fz_sum"]
    assert (pooled.foot == "sum").all() and (pooled.unit == "N").all()
    assert pooled.rmse.to_numpy() == pytest.approx(10.0)  # 5 N per foot
    assert pooled.rmse_pct_bw.to_numpy() == pytest.approx(100 * 10.0 / 600.0)
    assert pooled.pcc_trial_mean.to_numpy() == pytest.approx(1.0)
    assert pooled.n_trials.to_numpy() == pytest.approx(2)


def write_dataset_metadata(directory: Path, weights: dict[str, float]) -> Path:
    """A prepared dataset's metadata.json holding only per-Subject body weights."""
    directory.mkdir(parents=True, exist_ok=True)
    subjects = {s: {"body_weight_N": w, "source": "balance_trials"} for s, w in weights.items()}
    (directory / "metadata.json").write_text(
        json.dumps({"body_weights": {"subjects": subjects}}), encoding="utf-8")
    return directory


def bw_run(tmp_path) -> Path:
    """T02 and T03, 6 N force error; their balance predictions give 600 N each."""
    return write_run(tmp_path / "run", [
        trial("T02_balance", "T02", "balance", wave, offset=6.0),
        trial("T03_balance", "T03", "balance", wave, offset=6.0),
    ])


def force_pct_bw(out: Path, participant: str) -> np.ndarray:
    rows = per_participant(out)
    rows = rows[(rows.subset == "measured") & (rows.task_family == "ALL")
                & (rows.participant == participant)].set_index("target")
    return rows.loc[FORCES, "rmse_pct_bw"].to_numpy()


def test_percent_body_weight_uses_the_dataset_body_weights_when_present(tmp_path):
    dataset = write_dataset_metadata(tmp_path / "dataset", {"T02": 1200.0})
    out = evaluate(tmp_path, {"mlp": bw_run(tmp_path)}, "--dataset-dir", str(dataset))
    assert force_pct_bw(out, "T02") == pytest.approx(0.5)  # 6 N of 1200 N (dataset)
    assert force_pct_bw(out, "T03") == pytest.approx(1.0)  # not in the dataset: 600 N estimate
    bw = pd.read_csv(out / "body_weight.csv").set_index("subject")
    assert bw.loc["T02", "body_weight_N"] == pytest.approx(1200.0)
    assert bw.loc["T02", "source"] == "dataset:balance_trials"
    assert bw.loc["T03", "source"] == "predictions_balance"


def test_dataset_body_weights_are_found_from_the_run_metadata(tmp_path):
    run = bw_run(tmp_path)
    dataset = write_dataset_metadata(tmp_path / "dataset", {"T02": 1200.0, "T03": 300.0})
    for fold in run.iterdir():  # the engine writes metadata.json next to predictions.csv
        (fold / "metadata.json").write_text(json.dumps({"data_dir": str(dataset)}), encoding="utf-8")
    out = evaluate(tmp_path, {"mlp": run})
    assert force_pct_bw(out, "T02") == pytest.approx(0.5)
    assert force_pct_bw(out, "T03") == pytest.approx(2.0)


def test_without_dataset_body_weights_the_predictions_estimate_is_kept(tmp_path):
    out = evaluate(tmp_path, {"mlp": bw_run(tmp_path)})
    assert force_pct_bw(out, "T02") == pytest.approx(1.0)
    assert set(pd.read_csv(out / "body_weight.csv")["source"]) == {"predictions_balance"}


# --- Label source (spec #12, #21) ---------------------------------------------------

LABEL_NONE, LABEL_PLATE, LABEL_FOOT_HEIGHT, LABEL_PROTOCOL_WINDOW = 0, 1, 2, 3


def write_label_sources(directory: Path, frames, sources) -> Path:
    """A prepared dataset dir: one NPZ per Trial with ``target_time_s`` and ``label_source_L/R``.

    ``sources[batch]`` = (left codes, right codes), one int per sample or one int for all.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "metadata.json").exists():
        (directory / "metadata.json").write_text("{}", encoding="utf-8")
    for frame in frames:
        batch = frame["test_batch"].iloc[0]
        n = len(frame)
        left, right = (np.broadcast_to(np.asarray(c, dtype=np.int8), (n,)) for c in sources[batch])
        np.savez(directory / f"{batch}.npz", target_time_s=frame["time_s"].to_numpy(dtype=np.float64),
                 label_source_L=left, label_source_R=right)
    return directory


def zero(target, t):
    return np.zeros_like(t, dtype=float)


def airborne_trial(name, offset, participant="T02", family="jump"):
    return trial(name, participant, family, zero, offset=offset, state_l=STATE_AIRBORNE, state_r=STATE_AIRBORNE)


def test_airborne_metrics_are_split_by_label_source(tmp_path):
    balance = trial("T02_balance", "T02", "balance", wave, offset=2.0)
    jump = airborne_trial("T02_jump", offset=7.0)
    # A second Airborne Trial whose labels come from another (non Foot height) source.
    walk = airborne_trial("T02_walk", offset=3.0, family="walk")
    frames = [balance, jump, walk]
    dataset = write_label_sources(tmp_path / "dataset", frames, {
        "T02_balance": (LABEL_PLATE, LABEL_PLATE),
        "T02_jump": (LABEL_FOOT_HEIGHT, LABEL_FOOT_HEIGHT),
        "T02_walk": (LABEL_PROTOCOL_WINDOW, LABEL_PROTOCOL_WINDOW),
    })
    run = write_run(tmp_path / "run", frames)
    out = evaluate(tmp_path, {"mlp": run}, "--dataset-dir", str(dataset), "--by-label-source")
    rows = per_participant(out)
    pooled = rows[rows.task_family == "ALL"]
    by_subset = {s: pooled[pooled.subset == s] for s in set(pooled.subset)}
    assert by_subset["airborne"].rmse.to_numpy() == pytest.approx(math.sqrt((49 + 9) / 2))  # unchanged
    assert by_subset["airborne:foot_height_fused"].rmse.to_numpy() == pytest.approx(7.0)
    assert sorted(by_subset["airborne:foot_height_fused"].target) == sorted(TARGETS)
    assert by_subset["airborne:protocol_window"].rmse.to_numpy() == pytest.approx(3.0)
    assert not any(s.startswith("measured:") for s in by_subset)  # only Airborne is split
    assert set(summary(out).subset) >= {"airborne:foot_height_fused", "airborne:protocol_window"}
    report = json.loads((out / "evaluation.json").read_text())
    assert report["label_source_split"] is True
    assert report["runs"]["mlp"]["label_source_dataset_dirs"] == [str(dataset)]


def test_label_source_is_joined_per_sample_and_per_foot_on_time(tmp_path):
    jump = airborne_trial("T02_jump", offset=4.0)
    # Half-microsecond times (real ones step by 0.1 us), parsed back from the CSV: rounding to the
    # microsecond would split some of them from their dataset sample.
    jump["time_s"] = 1.2345005 + np.arange(SAMPLES) * 0.005
    left = np.where(np.arange(SAMPLES) < 50, LABEL_FOOT_HEIGHT, LABEL_PROTOCOL_WINDOW)
    dataset = write_label_sources(tmp_path / "dataset", [jump], {"T02_jump": (left, LABEL_FOOT_HEIGHT)})
    # Rows in another order than the dataset: the join must use the time, not the row.
    run = write_run(tmp_path / "run", [jump.sample(frac=1.0, random_state=0)])
    rows = per_participant(evaluate(tmp_path, {"mlp": run}, "--dataset-dir", str(dataset), "--by-label-source"))
    pooled = rows[rows.task_family == "ALL"].set_index(["subset", "target"])
    assert pooled.loc[("airborne:foot_height_fused", "Fz_L"), "n"] == 50
    assert pooled.loc[("airborne:protocol_window", "Fz_L"), "n"] == SAMPLES - 50
    assert pooled.loc[("airborne:foot_height_fused", "Fz_R"), "n"] == SAMPLES
    assert ("airborne:protocol_window", "Fz_R") not in pooled.index


def test_label_sources_are_found_from_the_run_metadata(tmp_path):
    jump = airborne_trial("T02_jump", offset=7.0)
    dataset = write_label_sources(tmp_path / "dataset", [jump], {"T02_jump": (LABEL_FOOT_HEIGHT, LABEL_FOOT_HEIGHT)})
    run = write_run(tmp_path / "run", [jump])
    for fold in run.iterdir():
        (fold / "metadata.json").write_text(json.dumps({"data_dir": str(dataset)}), encoding="utf-8")
    rows = per_participant(evaluate(tmp_path, {"mlp": run}, "--by-label-source"))
    assert set(rows.subset) == {"airborne", "airborne:foot_height_fused"}


def test_a_dataset_without_label_sources_reports_them_as_unrecorded(tmp_path):
    jump = airborne_trial("T02_jump", offset=7.0)
    dataset = tmp_path / "v36"  # v3.6 layout: an NPZ without label_source keys
    dataset.mkdir()
    (dataset / "metadata.json").write_text("{}", encoding="utf-8")
    np.savez(dataset / "T02_jump.npz", target_time_s=jump["time_s"].to_numpy())
    run = write_run(tmp_path / "run", [jump])
    rows = per_participant(evaluate(tmp_path, {"mlp": run}, "--dataset-dir", str(dataset), "--by-label-source"))
    assert set(rows.subset) == {"airborne", "airborne:unrecorded"}


def test_without_the_new_options_outputs_are_unchanged(tmp_path):
    frames = [trial("T02_balance", "T02", "balance", wave, offset=2.0), airborne_trial("T02_jump", offset=7.0)]
    dataset = write_label_sources(tmp_path / "dataset", frames, {
        "T02_balance": (LABEL_PLATE, LABEL_PLATE), "T02_jump": (LABEL_FOOT_HEIGHT, LABEL_FOOT_HEIGHT)})
    run = write_run(tmp_path / "run", frames)
    out = evaluate(tmp_path, {"mlp": run}, "--dataset-dir", str(dataset))
    rows = per_participant(out)
    report = json.loads((out / "evaluation.json").read_text())
    assert set(rows.subset) == {"measured", "airborne", "both_feet_sum"}
    assert list(rows.columns) == ["run", "participant", "subset", "task_family", "target", "component",
                                  "foot", "unit", *report["metrics"]]
    assert "label_source_split" not in report and "same_label" not in report
    assert report["subsets"] == ["measured", "airborne", "both_feet_sum"]


def with_states(frame, left, right=STATE_MEASURED):
    """``frame`` with per-sample foot states (Masked targets unmasked out, as the engine writes them)."""
    frame = frame.copy()
    frame["state_L"] = np.broadcast_to(left, (len(frame),))
    frame["state_R"] = np.broadcast_to(right, (len(frame),))
    for target in TARGETS:
        masked = frame[f"state_{target[-1]}"].to_numpy() == 0
        frame.loc[masked, target] = np.nan
        frame.loc[masked, f"mask_{target}"] = 0
    return frame


def segments(*parts):
    """Per-sample codes from (count, code) pairs."""
    return np.concatenate([np.full(n, code) for n, code in parts])


def test_same_label_comparison_keeps_only_samples_whose_foot_state_matches(tmp_path):
    M, A, X = STATE_MEASURED, STATE_AIRBORNE, 0
    base = trial("T02_jump", "T02", "jump", wave, offset=2.0)
    v36 = with_states(base, segments((100, M), (50, A), (50, X)))
    v37 = with_states(base, segments((100, M), (20, X), (80, A)))
    runs = {"v36": write_run(tmp_path / "v36", [v36]), "v37": write_run(tmp_path / "v37", [v37])}

    unfiltered = per_participant(evaluate(tmp_path / "all", runs))
    pooled = unfiltered[(unfiltered.task_family == "ALL") & (unfiltered.target == "Fz_L")].set_index(["run", "subset"])
    assert pooled.loc[("v37", "airborne"), "n"] == 80

    out = evaluate(tmp_path / "same", runs, "--same-label")
    rows = per_participant(out)
    pooled = rows[(rows.task_family == "ALL") & (rows.target == "Fz_L")].set_index(["run", "subset"])
    for run in runs:
        assert pooled.loc[(run, "measured"), "n"] == 100
        assert pooled.loc[(run, "airborne"), "n"] == 30  # samples 120..149, Airborne in both
    right = rows[(rows.task_family == "ALL") & (rows.target == "Fz_R")].set_index(["run", "subset"])
    assert right.loc[("v37", "measured"), "n"] == SAMPLES  # the right foot agrees everywhere
    both = rows[(rows.subset == "both_feet_sum") & (rows.task_family == "ALL") & (rows.target == "Fz_sum")]
    assert both.n.to_numpy() == pytest.approx(100)

    dropped = json.loads((out / "evaluation.json").read_text())["same_label"]["runs"]
    assert dropped["v36"]["L"] == {
        "evaluated_samples": 150, "kept": 130, "dropped": 20, "dropped_missing_in_other_runs": 0,
        "dropped_state_mismatch": 20, "dropped_label_source_mismatch": 0}
    assert dropped["v37"]["L"]["evaluated_samples"] == 180
    assert dropped["v37"]["L"]["dropped"] == 50
    assert dropped["v37"]["R"]["dropped"] == 0


def test_same_label_comparison_requires_the_same_airborne_label_source(tmp_path):
    base = airborne_trial("T02_jump", offset=4.0)
    runs = {"a": write_run(tmp_path / "a", [base]), "b": write_run(tmp_path / "b", [base]),
            "v36": write_run(tmp_path / "v36", [base])}
    left_b = segments((60, LABEL_PROTOCOL_WINDOW), (SAMPLES - 60, LABEL_FOOT_HEIGHT))
    dataset_a = write_label_sources(tmp_path / "data_a", [base], {"T02_jump": (LABEL_FOOT_HEIGHT, LABEL_FOOT_HEIGHT)})
    dataset_b = write_label_sources(tmp_path / "data_b", [base], {"T02_jump": (left_b, LABEL_FOOT_HEIGHT)})
    dataset_v36 = tmp_path / "data_v36"  # no label sources recorded: matches any source
    dataset_v36.mkdir()
    out = evaluate(tmp_path, runs, "--same-label", "--run-dataset", f"a={dataset_a}",
                   "--run-dataset", f"b={dataset_b}", "--run-dataset", f"v36={dataset_v36}")
    rows = per_participant(out)
    pooled = rows[(rows.task_family == "ALL") & (rows.subset == "airborne")].set_index(["run", "target"])
    for run in runs:
        assert pooled.loc[(run, "Fz_L"), "n"] == SAMPLES - 60
        assert pooled.loc[(run, "Fz_R"), "n"] == SAMPLES
    dropped = json.loads((out / "evaluation.json").read_text())["same_label"]["runs"]
    assert dropped["a"]["L"]["dropped_label_source_mismatch"] == 60
    assert dropped["a"]["L"]["dropped_state_mismatch"] == 0
    assert dropped["v36"]["L"]["dropped"] == 60


def test_same_label_comparison_drops_samples_missing_from_another_run(tmp_path):
    base = trial("T02_balance", "T02", "balance", wave, offset=2.0)
    extra = trial("T02_squat", "T02", "squat", wave, offset=2.0)
    runs = {"a": write_run(tmp_path / "a", [base, extra]), "b": write_run(tmp_path / "b", [base.iloc[:150]])}
    out = evaluate(tmp_path, runs, "--same-label")
    rows = per_participant(out)
    pooled = rows[(rows.task_family == "ALL") & (rows.target == "Fz_L") & (rows.subset == "measured")]
    assert pooled.set_index("run").n.to_dict() == {"a": 150, "b": 150}
    assert set(rows[rows.run == "a"].task_family) == {"ALL", "balance"}
    dropped = json.loads((out / "evaluation.json").read_text())["same_label"]["runs"]
    assert dropped["a"]["L"]["dropped_missing_in_other_runs"] == SAMPLES + 50
    assert dropped["b"]["L"]["dropped"] == 0


def test_same_label_comparison_needs_two_runs(tmp_path):
    run = write_run(tmp_path / "run", [trial("T02_balance", "T02", "balance", wave, offset=2.0)])
    with pytest.raises(SystemExit):
        evaluate(tmp_path, {"mlp": run}, "--same-label")


def test_published_gzipped_predictions_score_like_the_engine_output(tmp_path):
    frames = [trial("T02_balance", "T02", "balance", wave, offset=5.0), trial("T03_squat", "T03", "squat", wave, offset=2.0)]
    plain = write_run(tmp_path / "plain", frames)
    gzipped = write_run(tmp_path / "gzipped", frames)
    for path in gzipped.rglob("predictions.csv"):
        pd.read_csv(path).to_csv(path.with_name("predictions.csv.gz"), index=False)
        path.unlink()

    expected = summary(evaluate(tmp_path / "a", {"mlp": plain}))
    pd.testing.assert_frame_equal(summary(evaluate(tmp_path / "b", {"mlp": gzipped})), expected)
    # A fold holding both forms (gunzip -k) is read once, not twice.
    for path in gzipped.rglob("predictions.csv.gz"):
        pd.read_csv(path).to_csv(path.with_name("predictions.csv"), index=False)
    pd.testing.assert_frame_equal(summary(evaluate(tmp_path / "c", {"mlp": gzipped})), expected)
