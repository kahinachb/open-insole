#!/usr/bin/env python3
"""Literature-comparable evaluation of GRFM predictions.

Reads the ``predictions.csv`` files written by ``train_lstm_two_feet_masked.py``
(one per fold directory) for one or more runs and computes, per test
Participant and then mean +/- SD over Participants, for each target (component
and foot):

- RMSE (N, N.m), MAE and mean error;
- force RMSE in %BW. The body weight of each Subject is read from the prepared
  dataset's ``metadata.json`` (``body_weights.subjects``) when the dataset
  provides it: ``--dataset-dir``, else the ``data_dir`` recorded in the run's
  ``metadata.json`` next to each ``predictions.csv``. A Subject missing there
  falls back to the median of Fz_L + Fz_R over its balance Trials in the
  predictions (both feet Measured);
- NMSE = MSE / variance of the reference;
- PCC per Trial (averaged over Trials) and pooled PCC;
- NRMSE = RMSE / (99.5th - 0.5th percentile of the reference);
- offset / dynamic split of the MSE: share of the squared error explained by
  the per-Trial mean error, and the remainder.

The Measured subset and the Airborne subset are evaluated separately and never
mixed. The ``both_feet_sum`` subset holds the sum of both feet's forces
(``Fx_sum``, ``Fy_sum``, ``Fz_sum``, foot ``sum``) on the samples where both feet
are Measured: the total force of one plate under both feet (Sabbah 2024). Every metric is reported for all evaluated task families pooled
(``task_family == "ALL"``) and for each task family. With
``--common-task-families`` every run is restricted to the task families present
in all compared runs.

Usage::

    python evaluate_grfm.py \
        --run mlp=results/results_mlp_continuous_comparison/raw \
        --run transformer=results/results_transformer_continuous_comparison/raw \
        --common-task-families --output-dir evaluation_out

Body weight provided / missing / shifted (#28): ``evaluate_body_weight_modes.py``
writes one predictions dir per mode and passes them here as one ``--run`` each,
so the modes come out side by side like runs.

Foot height (engine ``--foot-height-output``, columns ``foot_height_mm_<foot>``,
``pred_foot_height_mm_<foot>``, ``foot_height_valid_<foot>``) is scored in its
own tables, never mixed with the GRFM ones: RMSE, MAE and mean error in mm per
Participant, foot, foot state (``all``, ``measured``, ``airborne``, ``masked``)
and task family (``ALL`` pooled), on the samples where it is valid. Runs whose
predictions carry no Foot height are left out of these tables.

Label source (dataset v3.7 NPZ ``label_source_L/R``: none, plate, foot_height,
protocol_window) is not in ``predictions.csv``: it is joined per sample from the
run's prepared dataset (``--run-dataset NAME=DIR``, else ``--dataset-dir``, else
the run metadata's ``data_dir``) on (``test_batch``, ``time_s`` =
``target_time_s``); ``unrecorded`` where the dataset has none (v3.6).

- ``--by-label-source`` adds the subsets ``airborne:<source>`` (the Airborne
  subset split by label source; ``airborne`` itself is unchanged).
- ``--same-label`` (two or more runs, e.g. a v3.6 and a v3.7 run on the same
  folds) evaluates every run only on the samples present in all runs and, per
  foot, labelled the same way in all of them: same foot state and, for Airborne,
  the same label source among the runs that record one. The Measured/Airborne
  foot-samples dropped per run, foot and reason are in ``evaluation.json``
  (``same_label``). The Foot height tables are not filtered.

Competitor tables (#39): ``competitor_comparison.csv`` extracts, per run, the
numbers the competitors report: walk-only RMSE, range NRMSE and per-Trial
PCC per target and averaged over the 12 targets (``mean_12``), against Mahmud
2026 and Li 2025; and the sum of both feet per continuous task in N and %BW,
per force and averaged over the three (``forces_mean``), against Sabbah 2024.
Each value is mean +/- SD over Participants. ``protocol_table.csv`` lists the
competitors' protocols (``tools/competitor_protocols.json``) and one row per
run, read from the engine's ``metadata.json`` next to its predictions.

Outputs (all written by ``write_outputs``): ``per_participant.csv``, ``summary.csv``,
``body_weight.csv``, ``evaluation.json``, ``competitor_comparison.csv`` and
``protocol_table.csv`` in ``--output-dir``, plus
``foot_height_per_participant.csv`` and ``foot_height_summary.csv`` when a run
has Foot height predictions.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from grfm_schema import (FEET, LABEL_SOURCE_CODES, STATE_AIRBORNE, STATE_MASKED, STATE_MEASURED, TARGETS,
                         subject_of_trial)


SUBSETS = {"measured": STATE_MEASURED, "airborne": STATE_AIRBORNE}
# Sum of both feet (the total wrench a single plate measures, as in Sabbah 2024): forces
# only, on the samples where both feet are Measured; targets ``Fx_sum`` .. ``Fz_sum``.
SUM_SUBSET = "both_feet_sum"
SUM_FOOT = "sum"
SUM_COMPONENTS = ("Fx", "Fy", "Fz")
ALL_FAMILIES = "ALL"
BALANCE_FAMILY = "balance"
RANGE_PERCENTILES = (0.5, 99.5)
MIN_TRIAL_SAMPLES = 3
METRICS = [
    "n",
    "n_trials",
    "rmse",
    "rmse_pct_bw",
    "mae",
    "mean_error",
    "nmse",
    "nrmse_range",
    "pcc_trial_mean",
    "pcc_pooled",
    "offset_share",
    "dynamic_share",
]
META_COLUMNS = ["test_batch", "test_group", "task_family", "state_L", "state_R"]
# Foot height (#20): its own subsets (foot states), metrics and tables.
FOOT_HEIGHT_STATES = {
    "all": None,
    "measured": STATE_MEASURED,
    "airborne": STATE_AIRBORNE,
    "masked": STATE_MASKED,
}
FOOT_HEIGHT_METRICS = ["n", "n_trials", "rmse_mm", "mae_mm", "mean_error_mm"]
FOOT_HEIGHT_COLUMNS = [
    column
    for foot in FEET
    for column in (f"foot_height_mm_{foot}", f"pred_foot_height_mm_{foot}", f"foot_height_valid_{foot}")
]


def subject_of(trial: pd.Series) -> pd.Series:
    """Subject of each trial name (``grfm_schema.subject_of_trial``)."""
    return trial.map(subject_of_trial)


def prediction_files(run_path: Path) -> list[Path]:
    """Every fold's predictions under a run dir: ``predictions.csv``, or the published
    ``predictions.csv.gz`` when a fold has no plain file (never both, or rows would count twice)."""
    if run_path.is_file():
        return [run_path]
    plain = set(run_path.rglob("predictions.csv"))
    gzipped = {path for path in run_path.rglob("predictions.csv.gz") if path.with_suffix("") not in plain}
    files = sorted(plain | gzipped)
    if not files:
        raise FileNotFoundError(f"no predictions.csv under {run_path}")
    return files

def load_predictions(files: list[Path], with_time: bool = False) -> pd.DataFrame:
    columns = META_COLUMNS + (["time_s"] if with_time else [])
    columns += [c for t in TARGETS for c in (t, f"pred_{t}", f"mask_{t}")]
    frames = [pd.read_csv(f, usecols=columns) for f in files]
    return pd.concat(frames, ignore_index=True)


def load_foot_heights(files: list[Path]) -> pd.DataFrame | None:
    """Foot height columns (with the meta columns) of the files that have them, else None."""
    frames = []
    for f in files:
        header = pd.read_csv(f, nrows=0).columns
        if set(FOOT_HEIGHT_COLUMNS) <= set(header):
            frames.append(pd.read_csv(f, usecols=META_COLUMNS + FOOT_HEIGHT_COLUMNS))
    return pd.concat(frames, ignore_index=True) if frames else None


def evaluate_foot_height(name: str, frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (participant, foot, foot state or all, task family or ALL), errors in mm."""
    participants = frame["test_group"].astype(str).to_numpy()
    families = frame["task_family"].astype(str).to_numpy()
    trial_codes = pd.factorize(frame["test_batch"])[0]
    rows = []
    for foot in FEET:
        truth = frame[f"foot_height_mm_{foot}"].to_numpy(dtype=float)
        error = frame[f"pred_foot_height_mm_{foot}"].to_numpy(dtype=float) - truth
        valid = (frame[f"foot_height_valid_{foot}"].to_numpy() == 1) & np.isfinite(error)
        state = frame[f"state_{foot}"].to_numpy()
        for foot_state, state_value in FOOT_HEIGHT_STATES.items():
            in_state = valid if state_value is None else valid & (state == state_value)
            for participant in np.unique(participants[in_state]):
                in_participant = in_state & (participants == participant)
                groups = [(ALL_FAMILIES, in_participant)]
                groups += [(f, in_participant & (families == f)) for f in np.unique(families[in_participant])]
                for family, selected in groups:
                    values = error[selected]
                    rows.append({
                        "run": name,
                        "participant": participant,
                        "foot": foot,
                        "foot_state": foot_state,
                        "task_family": family,
                        "unit": "mm",
                        "n": int(len(values)),
                        "n_trials": int(len(np.unique(trial_codes[selected]))),
                        "rmse_mm": float(np.sqrt(np.mean(values**2))),
                        "mae_mm": float(np.mean(np.abs(values))),
                        "mean_error_mm": float(np.mean(values)),
                    })
    return pd.DataFrame(rows)


def summarise_foot_height(per_participant: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SD (ddof=1) over Participants of the Foot height metrics, long format."""
    keys = ["run", "foot", "foot_state", "task_family", "unit"]
    long = per_participant.melt(
        id_vars=keys + ["participant"], value_vars=FOOT_HEIGHT_METRICS, var_name="metric"
    ).dropna(subset=["value"])
    grouped = long.groupby(keys + ["metric"], sort=False)["value"]
    return grouped.agg(mean="mean", sd="std", n_participants="count").reset_index()


def task_families(files: list[Path]) -> list[str]:
    families: set[str] = set()
    for f in files:
        families.update(pd.read_csv(f, usecols=["task_family"])["task_family"].astype(str).unique())
    return sorted(families)


def body_weights(predictions: pd.DataFrame) -> pd.DataFrame:
    """Median of Fz_L + Fz_R over each Subject's balance samples with both feet Measured."""
    rows = predictions[
        (predictions["task_family"] == BALANCE_FAMILY)
        & (predictions["state_L"] == STATE_MEASURED)
        & (predictions["state_R"] == STATE_MEASURED)
        & (predictions["mask_Fz_L"] == 1)
        & (predictions["mask_Fz_R"] == 1)
    ]
    total = rows["Fz_L"] + rows["Fz_R"]
    grouped = total.groupby(subject_of(rows["test_batch"]))
    table = pd.DataFrame({"body_weight_N": grouped.median(), "n_samples": grouped.size()})
    table.index.name = "subject"
    return table.reset_index()


PREDICTIONS_BW_SOURCE = "predictions_balance"


def dataset_dirs(files: list[Path]) -> list[Path]:
    """Dataset dirs recorded by the engine in the run ``metadata.json`` next to each file."""
    found: list[Path] = []
    for f in files:
        run_metadata = f.resolve().parent / "metadata.json"
        if not run_metadata.is_file():
            continue
        try:
            data_dir = json.loads(run_metadata.read_text(encoding="utf-8")).get("data_dir")
        except (OSError, ValueError, AttributeError):
            continue
        if data_dir and Path(data_dir) not in found:
            found.append(Path(data_dir))
    return found


def dataset_body_weights(data_dirs: list[Path]) -> pd.DataFrame:
    """``body_weights.subjects`` of the prepared datasets' metadata (first dataset wins)."""
    rows: dict[str, dict] = {}
    for data_dir in data_dirs:
        path = data_dir / "metadata.json"
        if not path.is_file():
            continue
        subjects = json.loads(path.read_text(encoding="utf-8")).get("body_weights", {}).get("subjects", {})
        for subject, entry in subjects.items():
            rows.setdefault(subject, {
                "subject": subject,
                "body_weight_N": float(entry["body_weight_N"]),
                "n_samples": entry.get("n_samples"),
                "source": f"dataset:{entry.get('source', 'unknown')}",
            })
    return pd.DataFrame(list(rows.values()), columns=["subject", "body_weight_N", "n_samples", "source"])


def run_body_weights(predictions: pd.DataFrame, data_dirs: list[Path]) -> pd.DataFrame:
    """Dataset body weights where present, else the predictions' balance estimate."""
    estimate = body_weights(predictions).assign(source=PREDICTIONS_BW_SOURCE)
    dataset = dataset_body_weights(data_dirs)
    if dataset.empty:
        return estimate
    fallback = estimate[~estimate["subject"].isin(dataset["subject"])]
    return pd.concat([dataset, fallback], ignore_index=True).sort_values("subject", ignore_index=True)


# Label source (#21): per sample and foot, joined from the prepared dataset's NPZ
# (``label_source_<foot>``, codes in metadata ``labelling_rules.label_source_codes``) on
# (``test_batch``, ``time_s`` = the NPZ's ``target_time_s``). predictions.csv does not carry it.
# A dataset that records no names falls back to grfm_schema.LABEL_SOURCE_CODES.
LABEL_SOURCE_UNRECORDED = "unrecorded"  # no dataset NPZ, or a dataset without label sources (v3.6)
TIME_KEY_SCALE = 1e9  # time_s rounded to the nanosecond for the joins (times carry 0.1 us steps)


def time_keys(time_s) -> np.ndarray:
    return np.round(np.asarray(time_s, dtype=float) * TIME_KEY_SCALE).astype(np.int64)


def label_source_codes(data_dir: Path) -> dict[int, str]:
    path = data_dir / "metadata.json"
    try:
        codes = json.loads(path.read_text(encoding="utf-8")).get("labelling_rules", {}).get("label_source_codes")
    except (OSError, ValueError, AttributeError):
        codes = None
    return {int(code): name for code, name in codes.items()} if codes else dict(LABEL_SOURCE_CODES)


def attach_label_sources(predictions: pd.DataFrame, data_dirs: list[Path]) -> pd.DataFrame:
    """Add ``label_source_L/R`` (source names) joined from the first dataset holding each Trial."""
    sources = {foot: np.full(len(predictions), LABEL_SOURCE_UNRECORDED, dtype=object) for foot in FEET}
    keys = time_keys(predictions["time_s"])
    batches = predictions["test_batch"].astype(str).to_numpy()
    for batch in np.unique(batches):
        npz = next((d / f"{batch}.npz" for d in data_dirs if (d / f"{batch}.npz").is_file()), None)
        if npz is None:
            continue
        codes = label_source_codes(npz.parent)
        with np.load(npz) as values:
            if not all(f"label_source_{foot}" in values for foot in FEET):
                continue
            dataset_keys = time_keys(values["target_time_s"])
            dataset_sources = {foot: values[f"label_source_{foot}"] for foot in FEET}
        order = np.argsort(dataset_keys, kind="stable")
        rows = np.flatnonzero(batches == batch)
        position = np.clip(np.searchsorted(dataset_keys[order], keys[rows]), 0, len(order) - 1)
        found = dataset_keys[order][position] == keys[rows]
        for foot in FEET:
            names = np.array([codes.get(int(c), str(int(c))) for c in dataset_sources[foot][order][position]],
                             dtype=object)
            sources[foot][rows[found]] = names[found]
    return predictions.assign(**{f"label_source_{foot}": sources[foot] for foot in FEET})


# Same-label comparison (#21): a foot whose label differs across the compared runs gets
# this state, which no subset selects.
STATE_LABEL_MISMATCH = -1
EVALUATED_STATES = (STATE_MEASURED, STATE_AIRBORNE)


def same_label_filter(predictions: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    """Keep, in every run, only the samples present in all runs, and per foot only the labels
    that agree: same foot state and, for Airborne, the same label source among the runs that
    record one. Other evaluated feet (Measured or Airborne) are set to ``STATE_LABEL_MISMATCH``.

    Returns the filtered predictions and, per run and foot, the evaluated foot-samples before,
    kept and dropped (by reason).
    """
    keyed = {}
    for name, frame in predictions.items():
        key = pd.MultiIndex.from_arrays([frame["test_batch"].astype(str), time_keys(frame["time_s"])])
        if key.has_duplicates:
            raise SystemExit(f"run {name}: duplicate (test_batch, time_s) samples, cannot align runs")
        keyed[name] = frame.set_axis(key)
    common = None
    for frame in keyed.values():
        common = frame.index if common is None else common.intersection(frame.index, sort=False)
    aligned = {name: frame.loc[common] for name, frame in keyed.items()}

    agree = {}
    for foot in FEET:
        states = np.stack([f[f"state_{foot}"].to_numpy() for f in aligned.values()])
        same_state = (states == states[0]).all(axis=0)
        sources = np.stack([f[f"label_source_{foot}"].astype(str).to_numpy() for f in aligned.values()])
        recorded = sources != LABEL_SOURCE_UNRECORDED
        reference = np.where(recorded.any(axis=0), sources[recorded.argmax(axis=0), np.arange(sources.shape[1])], "")
        same_source = ((sources == reference) | ~recorded).all(axis=0) | (states[0] != STATE_AIRBORNE)
        agree[foot] = (same_state, same_state & same_source)

    filtered, report = {}, {}
    for name, frame in predictions.items():
        kept = aligned[name].copy()
        report[name] = {}
        for foot in FEET:
            same_state, same_label = agree[foot]
            evaluated_before = int(frame[f"state_{foot}"].isin(EVALUATED_STATES).sum())
            evaluated = kept[f"state_{foot}"].isin(EVALUATED_STATES).to_numpy()
            report[name][foot] = {
                "evaluated_samples": evaluated_before,
                "kept": int((evaluated & same_label).sum()),
                "dropped": int(evaluated_before - (evaluated & same_label).sum()),
                "dropped_missing_in_other_runs": int(evaluated_before - evaluated.sum()),
                "dropped_state_mismatch": int((evaluated & ~same_state).sum()),
                "dropped_label_source_mismatch": int((evaluated & same_state & ~same_label).sum()),
            }
            column = kept[f"state_{foot}"].to_numpy().copy()
            column[evaluated & ~same_label] = STATE_LABEL_MISMATCH
            kept[f"state_{foot}"] = column
        filtered[name] = kept.reset_index(drop=True)
    return filtered, report


def _ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, NaN when the denominator is not positive (e.g. Airborne zeros)."""
    return float(numerator / denominator) if denominator > 0 else math.nan


def metrics(y: np.ndarray, pred: np.ndarray, trial: np.ndarray, bw: np.ndarray | None) -> dict[str, float]:
    """Metrics of one (participant, subset, family, target) group.

    ``trial`` holds integer Trial codes; ``bw`` the body weight of each sample's
    Subject (None for moments).
    """
    error = pred - y
    n = len(y)
    mse = float(np.mean(error**2))
    rmse = math.sqrt(mse)
    variance = float(np.var(y))
    low, high = np.percentile(y, RANGE_PERCENTILES)
    reference_range = float(high - low)

    codes, trial = np.unique(trial, return_inverse=True)
    counts = np.bincount(trial).astype(float)
    trial_mean_error = np.bincount(trial, weights=error) / counts
    offset_mse = float(np.mean(trial_mean_error[trial] ** 2))
    offset_share = _ratio(offset_mse, mse)

    yc = y - (np.bincount(trial, weights=y) / counts)[trial]
    pc = pred - (np.bincount(trial, weights=pred) / counts)[trial]
    syy = np.bincount(trial, weights=yc * yc)
    spp = np.bincount(trial, weights=pc * pc)
    syp = np.bincount(trial, weights=yc * pc)
    valid = (counts >= MIN_TRIAL_SAMPLES) & (syy > 0) & (spp > 0)
    trial_pcc = syp[valid] / np.sqrt(syy[valid] * spp[valid])

    pooled_ok = variance > 0 and np.var(pred) > 0
    if bw is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rmse_pct_bw = 100.0 * math.sqrt(float(np.mean((error / bw) ** 2)))
    else:
        rmse_pct_bw = math.nan

    return {
        "n": n,
        "n_trials": len(codes),
        "rmse": rmse,
        "rmse_pct_bw": rmse_pct_bw,
        "mae": float(np.mean(np.abs(error))),
        "mean_error": float(np.mean(error)),
        "nmse": _ratio(mse, variance),
        "nrmse_range": _ratio(rmse, reference_range),
        "pcc_trial_mean": float(np.mean(trial_pcc)) if len(trial_pcc) else math.nan,
        "pcc_pooled": float(np.corrcoef(y, pred)[0, 1]) if pooled_ok else math.nan,
        "offset_share": offset_share,
        "dynamic_share": 1.0 - offset_share,
    }


def evaluate_run(
    name: str, predictions: pd.DataFrame, weights: pd.DataFrame, by_label_source: bool = False
) -> pd.DataFrame:
    """One row per (participant, subset, task family or ALL, target).

    With ``by_label_source`` (needs the ``label_source_<foot>`` columns of
    ``attach_label_sources``) the Airborne subset is also reported per label
    source, as subsets ``airborne:<source>``.
    """
    subjects = subject_of(predictions["test_batch"])
    bw_by_subject = dict(zip(weights["subject"], weights["body_weight_N"]))
    sample_bw = subjects.map(bw_by_subject).to_numpy(dtype=float)
    trial_codes = pd.factorize(predictions["test_batch"])[0]
    participants = predictions["test_group"].astype(str).to_numpy()
    families = predictions["task_family"].astype(str).to_numpy()

    rows = []

    def add_rows(subset, target, component, foot, y_all, pred_all, selected, is_force):
        for participant in np.unique(participants[selected]):
            in_participant = selected & (participants == participant)
            groups = [(ALL_FAMILIES, in_participant)]
            groups += [(f, in_participant & (families == f)) for f in np.unique(families[in_participant])]
            for family, mask in groups:
                values = metrics(
                    y_all[mask],
                    pred_all[mask],
                    trial_codes[mask],
                    sample_bw[mask] if is_force else None,
                )
                rows.append(
                    {
                        "run": name,
                        "participant": participant,
                        "subset": subset,
                        "task_family": family,
                        "target": target,
                        "component": component,
                        "foot": foot,
                        "unit": "N" if is_force else "N.m",
                        **values,
                    }
                )

    def column(target, prefix=""):
        return predictions[f"{prefix}{target}"].to_numpy(dtype=float)

    def valid(target):
        return (predictions[f"mask_{target}"].to_numpy() == 1) & np.isfinite(column(target)) & np.isfinite(
            column(target, "pred_")
        )

    for target in TARGETS:
        foot = target[-1]
        component = target[:-2]
        state = predictions[f"state_{foot}"].to_numpy()
        for subset, state_value in SUBSETS.items():
            selected = valid(target) & (state == state_value)
            if selected.any():
                add_rows(subset, target, component, foot, column(target), column(target, "pred_"), selected,
                         component.startswith("F"))
            if by_label_source and state_value == STATE_AIRBORNE:
                sources = predictions[f"label_source_{foot}"].astype(str).to_numpy()
                for source in np.unique(sources[selected]):
                    add_rows(f"{subset}:{source}", target, component, foot, column(target),
                             column(target, "pred_"), selected & (sources == source), component.startswith("F"))

    both_measured = (predictions["state_L"].to_numpy() == STATE_MEASURED) & (
        predictions["state_R"].to_numpy() == STATE_MEASURED
    )
    for component in SUM_COMPONENTS:
        left, right = f"{component}_L", f"{component}_R"
        selected = both_measured & valid(left) & valid(right)
        if selected.any():
            add_rows(SUM_SUBSET, f"{component}_{SUM_FOOT}", component, SUM_FOOT,
                     column(left) + column(right), column(left, "pred_") + column(right, "pred_"), selected, True)
    return pd.DataFrame(rows)


def summarise(per_participant: pd.DataFrame) -> pd.DataFrame:
    """Mean +/- SD (ddof=1) over Participants, long format: one row per metric."""
    keys = ["run", "subset", "task_family", "target", "component", "foot", "unit"]
    long = per_participant.melt(id_vars=keys + ["participant"], value_vars=METRICS, var_name="metric")
    long = long.dropna(subset=["value"])
    grouped = long.groupby(keys + ["metric"], sort=False)["value"]
    table = grouped.agg(mean="mean", sd="std", n_participants="count").reset_index()
    return table


CONTINUOUS_FAMILIES = ("balance", "squat", "hulahoop", "lateral_2supports")
WALK_FAMILY = "walk"
WALK_METRICS = ("rmse", "nrmse_range", "pcc_trial_mean")
SUM_METRICS = ("rmse", "rmse_pct_bw")
COMPETITOR_PROTOCOLS = Path(__file__).resolve().parent / "tools" / "competitor_protocols.json"


def competitor_comparison(per_participant: pd.DataFrame) -> pd.DataFrame:
    """Walk-only metrics and the sum of both feet per continuous task, mean +/- SD over Participants."""
    walk = per_participant[(per_participant["subset"] == "measured") & (per_participant["task_family"] == WALK_FAMILY)]
    walk_mean = walk.groupby(["run", "participant", "task_family"], as_index=False)[list(WALK_METRICS)].mean()
    walk_mean["target"] = "mean_12"
    sums = per_participant[
        (per_participant["subset"] == SUM_SUBSET) & per_participant["task_family"].isin(CONTINUOUS_FAMILIES)
    ]
    sums_mean = sums.groupby(["run", "participant", "task_family"], as_index=False)[list(SUM_METRICS)].mean()
    sums_mean["target"] = "forces_mean"
    parts = [
        ("walk", pd.concat([walk, walk_mean], ignore_index=True), WALK_METRICS),
        ("both_feet_sum", pd.concat([sums, sums_mean], ignore_index=True), SUM_METRICS),
    ]
    rows = []
    for name, frame, reported in parts:
        if frame.empty:
            continue
        table = summarise_metric(frame, ["run", "task_family", "target"], reported)
        table.insert(1, "comparison", name)
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["run", "comparison", "task_family", "target", "metric", "mean", "sd", "n_participants"]
    )


def first_json(files: list[Path], name: str) -> dict:
    """The first ``name`` file next to one of the run's predictions (engine layout), else {}."""
    for path in files:
        candidate = path.parent / name
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def split_label(metadata: dict, manifest: dict, participants: list[str], folds: int) -> str:
    """Fixed split, Full CV (every Participant of the split tested exactly once) or partial CV.

    ``folds`` counts the run's fold predictions: one per tested Participant for a Full CV.
    """
    test, validation = metadata.get("test_groups"), metadata.get("validation_groups")
    if test or manifest.get("split") == "fixed":
        return f"Fixed split (test: {', '.join(test or participants)}; validation: {', '.join(validation or [])})"
    groups = manifest.get("groups")
    if not groups:
        return "unknown"
    if sorted(groups) == sorted(participants) and folds == len(groups):
        return f"Full CV ({len(groups)} folds, one per Participant)"
    return f"partial CV ({len(participants)} of {len(groups)} Participants tested)"


def run_protocol(name: str, files: list[Path], families: list[str], participants: list[str]) -> dict:
    """Our protocol row for one run, from the engine metadata and split manifest next to its predictions."""
    metadata = first_json(files, "metadata.json")
    causal = metadata.get("causal")
    return {
        "study": name,
        "outputs": "6D GRFM per foot (foot frame)",
        "hardware": "2 insoles, 15 FX29 Gauges + 4 IMUs each",
        "participants": f"{len(participants)} evaluated",
        "split": split_label(metadata, first_json(files, "split_manifest.json"), participants, len(files)),
        "causal": "unknown" if causal is None else ("yes" if causal else "no"),
        "filtering": "none (raw inputs, causal window)" if causal else "unknown",
        "tasks": ", ".join(families),
        "models": "one model, all tasks",
    }


def protocol_table(runs: list[dict]) -> pd.DataFrame:
    studies = json.loads(COMPETITOR_PROTOCOLS.read_text(encoding="utf-8"))["studies"]
    return pd.DataFrame(studies + runs)


def summarise_metric(frame: pd.DataFrame, keys: list[str], reported: tuple[str, ...]) -> pd.DataFrame:
    """Mean +/- SD (ddof=1) over Participants of the ``reported`` metrics, per ``keys``."""
    long = frame.melt(id_vars=keys + ["participant"], value_vars=list(reported), var_name="metric")
    grouped = long.dropna(subset=["value"]).groupby(keys + ["metric"], sort=False)["value"]
    return grouped.agg(mean="mean", sd="std", n_participants="count").reset_index()


def parse_run(text: str) -> tuple[str, Path]:
    if "=" in text:
        name, path = text.split("=", 1)
        return name, Path(path)
    path = Path(text)
    return "/".join(path.resolve().parts[-2:]), path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="run to evaluate: a directory searched recursively for predictions.csv "
        "(or one predictions.csv). Repeat to compare runs.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--common-task-families",
        action="store_true",
        help="restrict every run to the task families present in all runs",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        help="prepared dataset whose metadata.json body weights set %%BW (default: the "
        "data_dir recorded in each run's metadata.json, if any)",
    )
    parser.add_argument(
        "--run-dataset",
        action="append",
        default=[],
        metavar="NAME=DIR",
        help="prepared dataset of one run (overrides --dataset-dir for that run), e.g. to "
        "compare a v3.6 run with a v3.7 run. Repeat per run.",
    )
    parser.add_argument(
        "--by-label-source",
        action="store_true",
        help="also report Airborne metrics per label source (subsets airborne:<source>), "
        "joined from the dataset NPZ label_source_L/R; 'unrecorded' when absent (v3.6)",
    )
    parser.add_argument(
        "--same-label",
        action="store_true",
        help="compare the runs only on samples labelled the same way in all of them: same "
        "foot state and, for Airborne, the same label source (where recorded); the dropped "
        "counts are in evaluation.json under same_label",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runs = dict(parse_run(text) for text in args.run)
    if len(runs) != len(args.run):
        raise SystemExit("duplicate run names")

    files = {name: prediction_files(path) for name, path in runs.items()}
    present = {name: task_families(f) for name, f in files.items()}
    common = sorted(set.intersection(*(set(f) for f in present.values())))
    if args.common_task_families and not common:
        raise SystemExit("the runs share no task family")

    run_datasets = dict(parse_run(text) for text in args.run_dataset)
    unknown = sorted(set(run_datasets) - set(runs))
    if unknown:
        raise SystemExit(f"--run-dataset names unknown runs: {', '.join(unknown)}")
    if args.same_label and len(runs) < 2:
        raise SystemExit("--same-label needs at least two runs")
    with_sources = args.by_label_source or args.same_label

    loaded, run_data_dirs, weights = {}, {}, []
    for name, run_files in files.items():
        predictions = load_predictions(run_files, with_time=with_sources)
        if name in run_datasets:
            data_dirs = [run_datasets[name]]
        else:
            data_dirs = [args.dataset_dir] if args.dataset_dir else dataset_dirs(run_files)
        run_weights = run_body_weights(predictions, data_dirs)
        weights.append(run_weights)
        if args.common_task_families:
            predictions = predictions[predictions["task_family"].isin(common)]
        if with_sources:
            predictions = attach_label_sources(predictions, data_dirs)
        loaded[name] = (predictions, run_weights)
        run_data_dirs[name] = data_dirs

    same_label = None
    if args.same_label:
        filtered, same_label = same_label_filter({name: frame for name, (frame, _) in loaded.items()})
        loaded = {name: (filtered[name], w) for name, (_, w) in loaded.items()}

    per_participant = []
    foot_height_rows = []
    report_runs = {}
    protocols = []
    for name, run_files in files.items():
        predictions, run_weights = loaded[name]
        per_participant.append(evaluate_run(name, predictions, run_weights, args.by_label_source))
        foot_heights = load_foot_heights(run_files)
        if foot_heights is not None:
            if args.common_task_families:
                foot_heights = foot_heights[foot_heights["task_family"].isin(common)]
            foot_height_rows.append(evaluate_foot_height(name, foot_heights))
        protocols.append(run_protocol(
            name, run_files, sorted(predictions["task_family"].astype(str).unique()),
            sorted(predictions["test_group"].astype(str).unique()),
        ))
        report_runs[name] = {
            "path": str(runs[name]),
            "prediction_files": [str(f) for f in run_files],
            "participants": sorted(predictions["test_group"].astype(str).unique()),
            "task_families_present": present[name],
            "task_families_evaluated": sorted(predictions["task_family"].astype(str).unique()),
        }
        if with_sources:
            report_runs[name]["label_source_dataset_dirs"] = [str(d) for d in run_data_dirs[name]]

    table = pd.concat(per_participant, ignore_index=True)
    bw = pd.concat(weights, ignore_index=True).drop_duplicates("subject").sort_values("subject")
    report = {
        "runs": report_runs,
        "common_task_families_filter": bool(args.common_task_families),
        "common_task_families": common,
        "subsets": [*SUBSETS, SUM_SUBSET],
        "all_families_label": ALL_FAMILIES,
        "metrics": METRICS,
        "range_percentiles": list(RANGE_PERCENTILES),
        "sd_ddof": 1,
    }
    if args.by_label_source:
        report["label_source_split"] = True
        report["label_source_subsets"] = sorted(s for s in table["subset"].unique() if ":" in s)
    if same_label is not None:
        report["same_label"] = {
            "rule": "samples present in every run; per foot, the same foot state in every run "
            "and, for Airborne, the same label source among the runs that record one. Other "
            "Measured/Airborne feet are dropped. Foot height tables are not filtered.",
            "runs": same_label,
        }
    foot_height = pd.concat(foot_height_rows, ignore_index=True) if foot_height_rows else None
    report["foot_height"] = {
        "runs": sorted(foot_height["run"].unique()) if foot_height is not None else [],
        "foot_states": list(FOOT_HEIGHT_STATES),
        "metrics": FOOT_HEIGHT_METRICS,
        "unit": "mm",
    }
    write_outputs(args.output_dir, table, summarise(table), bw, report, foot_height, protocol_table(protocols))
    return 0


def write_outputs(
    output_dir: Path,
    per_participant: pd.DataFrame,
    summary: pd.DataFrame,
    bw: pd.DataFrame,
    report: dict,
    foot_height: pd.DataFrame | None = None,
    protocols: pd.DataFrame | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    competitor_comparison(per_participant).to_csv(output_dir / "competitor_comparison.csv", index=False)
    if protocols is not None:
        protocols.to_csv(output_dir / "protocol_table.csv", index=False)
    if foot_height is not None:
        foot_height.to_csv(output_dir / "foot_height_per_participant.csv", index=False)
        summarise_foot_height(foot_height).to_csv(output_dir / "foot_height_summary.csv", index=False)
    per_participant.to_csv(output_dir / "per_participant.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    bw.to_csv(output_dir / "body_weight.csv", index=False)
    (output_dir / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
