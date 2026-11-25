#!/usr/bin/env python3
"""Sabbah 2024 (BioRob, RFR_IS) retrained on our dataset under Full CV, offline as published (#46).

The method is kept as published (workspace ADR 0005):
- a scikit-learn multi-output Random Forest: 100 trees, max depth 10, min samples split 5;
- one model per task family: the task of a test trial is known in advance, as in the paper;
- a 5th-order zero-phase Butterworth low-pass at 10 Hz on inputs and training targets. It is
  non-causal: the method is offline;
- the Body weight (kg) as an input, his subject weight;
- one instantaneous sample per prediction, no window.

Our data forces a few changes:
- the targets are the 6D GRFM of each foot. One model per foot is fitted on the samples where
  that foot's 6 targets are valid (Measured or Airborne);
- the training samples are taken every ``--train-stride`` samples (default 4). This is our
  choice, not the paper's: at 200 Hz after a 10 Hz filter, neighbours are near-duplicates, and
  stride 4 gives about Sabbah's training size (about 150 k samples per model, his 4 x 45 k);
- each fold trains on its 12 training Participants. Nothing is selected, so the 2 validation
  Participants are unused;
- predictions are scored against our unfiltered targets, on the samples our causal MLP is
  scored on (``competitor_folds``).

Inputs (``--inputs``):
- ``loadsol``: per foot the 4 loadsol-like zones and the heel-Patch IMU, the equivalent of his
  shin IMU;
- ``raw``: our 78 channels.

Each foot's forest reads both insoles, as his model of the total wrench under both feet did.

Targets (``--target``):
- ``foot``: the 12 targets, one forest per foot;
- ``total``: his own target, the total force under both feet (the 3 force components summed),
  one forest per task family, fitted where both feet's forces are valid. His moments are taken
  about the plate origin, so they have no counterpart here. To be read by ``evaluate_grfm``, each
  foot is written half the total, and the moments 0: only its ``both_feet_sum`` rows mean anything.

Usage::

    python competitor_sabbah.py DATA_DIR --inputs loadsol --output-dir RESULTS/sabbah_rf_loadsol/all
    python competitor_sabbah.py DATA_DIR --inputs raw --output-dir ... --participant P01
    python competitor_sabbah.py DATA_DIR --inputs loadsol --target total --output-dir RESULTS/sabbah_rf_loadsol_total/all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import RandomForestRegressor

from competitor_folds import fold_of, load_folds, sampling_rate_hz, scored_indexes, segment_runs, subset, write_fold
from train_lstm_two_feet_masked import Acquisition, loadsol_inputs

FILTER_ORDER = 5
FILTER_CUTOFF_HZ = 10.0
FOREST = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}
TARGETS_PER_FOOT = 6
FORCES_PER_FOOT = 3
GRAVITY = 9.81


def zero_phase(values: np.ndarray, sos: np.ndarray, runs: list[slice]) -> np.ndarray:
    """Zero-phase low-pass of ``values`` (samples x columns) within each run; runs too short stay raw."""
    out = values.astype(np.float64, copy=True)
    for run in runs:
        length = run.stop - run.start
        if length < 2:
            continue
        padlen = min(3 * (2 * len(sos) + 1), length - 1)
        out[run] = sosfiltfilt(sos, values[run], axis=0, padlen=padlen)
    return out


def inputs(item: Acquisition, feature_names: list[str], mode: str, weights_kg: dict[str, float], sos) -> np.ndarray:
    """Filtered inputs of one trial, the Body weight (kg) last."""
    x = loadsol_inputs(item.x, feature_names, heel_imu=True)[0] if mode == "loadsol" else item.x
    x = zero_phase(x, sos, segment_runs(item.segment_id))
    weight = np.full((len(x), 1), weights_kg[item.subject_session])
    return np.concatenate((x, weight), axis=1)


def foot_targets(item: Acquisition, foot: int, sos) -> tuple[np.ndarray, np.ndarray]:
    """Filtered targets of one foot and where all 6 are valid; filtered within valid runs only."""
    columns = foot_columns(foot)
    valid = item.mask[:, columns].all(axis=1)
    y = np.where(valid[:, None], item.y[:, columns], 0.0)
    return zero_phase(y, sos, segment_runs(item.segment_id, valid)), valid


def total_targets(item: Acquisition, sos) -> tuple[np.ndarray, np.ndarray]:
    """Filtered total force under both feet and where both feet's forces are valid."""
    left, right = force_columns(0), force_columns(1)
    valid = item.mask[:, left].all(axis=1) & item.mask[:, right].all(axis=1)
    y = np.where(valid[:, None], item.y[:, left] + item.y[:, right], 0.0)
    return zero_phase(y, sos, segment_runs(item.segment_id, valid)), valid


def force_columns(foot: int) -> slice:
    """Force columns (Fx, Fy, Fz) of one foot, the first 3 of its 6 targets."""
    start = foot * TARGETS_PER_FOOT
    return slice(start, start + FORCES_PER_FOOT)


def foot_columns(foot: int) -> slice:
    """Target columns of one foot: the dataset orders the 6 left targets, then the 6 right ones."""
    return slice(foot * TARGETS_PER_FOOT, (foot + 1) * TARGETS_PER_FOOT)


def body_weights_kg(metadata: dict) -> dict[str, float]:
    """Body weight of each Subject in kg (dataset metadata, both load conditions share it)."""
    subjects = metadata.get("body_weights", {}).get("subjects", {})
    return {name: float(entry["body_weight_N"]) / GRAVITY for name, entry in subjects.items()
            if entry.get("body_weight_N") and float(entry["body_weight_N"]) > 0}


def run_fold(acquisitions, metadata, targets, groups, fold, args, sos) -> Path:
    feature_names = list(metadata["feature_names"])
    weights_kg = body_weights_kg(metadata)
    train, test = subset(acquisitions, fold, "train"), subset(acquisitions, fold, "test")
    predictions = {item.name: np.zeros((len(scored_indexes(item)), len(targets))) for item in test}
    for family in sorted({item.task_family for item in test}):
        family_train = [item for item in train if item.task_family == family]
        family_test = [item for item in test if item.task_family == family]
        x_train = [inputs(item, feature_names, args.inputs, weights_kg, sos) for item in family_train]
        x_test = [inputs(item, feature_names, args.inputs, weights_kg, sos)[scored_indexes(item)] for item in family_test]
        for model in ("total",) if args.target == "total" else (0, 1):
            rows_x, rows_y = [], []
            for item, x in zip(family_train, x_train):
                y, valid = total_targets(item, sos) if model == "total" else foot_targets(item, model, sos)
                keep = np.flatnonzero(valid)[:: args.train_stride]
                rows_x.append(x[keep])
                rows_y.append(y[keep])
            forest = RandomForestRegressor(**FOREST, n_jobs=args.jobs, random_state=args.seed)
            forest.fit(np.concatenate(rows_x), np.concatenate(rows_y))
            for item, x in zip(family_test, x_test):
                if not len(x):
                    continue
                if model == "total":
                    half = forest.predict(x) / 2
                    predictions[item.name][:, force_columns(0)] = half
                    predictions[item.name][:, force_columns(1)] = half
                else:
                    predictions[item.name][:, foot_columns(model)] = forest.predict(x)
            name = "total" if model == "total" else f"foot {'LR'[model]}"
            print(f"  {fold['test_group']} {family} {name}: {sum(len(r) for r in rows_y)} training samples", flush=True)
    fold_dir = args.output_dir / f"fold_{fold['test_group']}"
    write_fold(
        fold_dir, args.data_dir, groups, fold, targets,
        [(item, scored_indexes(item), predictions[item.name]) for item in test],
        {
            "model_type": "sabbah_rf",
            "competitor": "Sabbah 2024 (RFR_IS)",
            "causal": False,
            "inputs": args.inputs,
            "filter": f"zero-phase Butterworth, order {FILTER_ORDER}, {FILTER_CUTOFF_HZ} Hz, inputs and training targets",
            "target": args.target,
            "models": ("one Random Forest per task family on the total force under both feet; each foot is "
                       "written half of it and the moments 0, so only the both-feet sum is meaningful"
                       if args.target == "total" else "one Random Forest per task family and foot"),
            "forest": FOREST,
            "train_stride": args.train_stride,
            "seed": args.seed,
            "task_set": "all",
        },
    )
    return fold_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="run dir; one fold_<Participant> per fold")
    parser.add_argument("--inputs", choices=("loadsol", "raw"), required=True)
    parser.add_argument("--target", choices=("foot", "total"), default="foot",
                        help="foot: 12 targets, one forest per foot; total: the summed forces of both feet (his target)")
    parser.add_argument("--participant", nargs="*", default=None, help="test Participants (default: every fold)")
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.train_stride < 1:
        parser.error("--train-stride must be positive")
    acquisitions, metadata, targets, groups, folds = load_folds(args.data_dir)
    missing = sorted({item.subject_session for item in acquisitions} - set(body_weights_kg(metadata)))
    if missing:
        parser.error(f"no body weight in metadata.json for {', '.join(missing)}")
    sos = butter(FILTER_ORDER, FILTER_CUTOFF_HZ, fs=sampling_rate_hz(acquisitions), output="sos")
    for participant in args.participant or groups:
        if (args.output_dir / f"fold_{participant}" / "predictions.csv").exists():
            print(f"{participant}: already done, skipped")
            continue
        start = time.time()
        fold_dir = run_fold(acquisitions, metadata, targets, groups, fold_of(folds, participant), args, sos)
        print(f"{participant}: {fold_dir} ({time.time() - start:.0f} s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
