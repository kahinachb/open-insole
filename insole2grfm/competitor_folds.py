"""Full CV folds, scored samples and fold outputs shared by the competitor reproductions (#46, #47).

A competitor run must be scored exactly like our engine's runs (workspace ADR 0005):
- the same folds: each test Participant with the same 2 validation Participants and 12 training ones;
- the same samples: those our causal 21-sample MLP predicts (``scored_indexes``);
- the same files: the engine's fold layout (``predictions.csv``, ``metadata.json``,
  ``split_manifest.json``), so that ``evaluate_grfm.py`` reads it unchanged.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from train_lstm_two_feet_masked import (
    Acquisition,
    build_split_manifest,
    load_acquisitions,
    prediction_csv_fields,
    prediction_rows,
)

# Window of the MLP whose scored samples every competitor is scored on.
REFERENCE_WINDOW = 21
SPLIT_LEVEL = "subject_identity"
VALIDATION_GROUP_COUNT = 2


def load_folds(data_dir: Path, task_set: str = "all") -> tuple[list[Acquisition], dict, tuple[str, ...], list[str], list[dict]]:
    """Every trial, the dataset metadata, the targets, the Participants and the engine's Full CV folds."""
    acquisitions, metadata, targets, _ = load_acquisitions(data_dir, None, None, task_set)
    groups, folds = build_split_manifest(acquisitions, SPLIT_LEVEL, VALIDATION_GROUP_COUNT)
    return acquisitions, metadata, targets, groups, folds


def fold_of(folds: list[dict], participant: str) -> dict:
    for fold in folds:
        if fold["test_group"] == participant:
            return fold
    raise ValueError(f"no fold tests {participant}")


def subset(acquisitions: list[Acquisition], fold: dict, name: str) -> list[Acquisition]:
    """The trials of ``name`` (train, validation or test) in ``fold``."""
    names = set(fold[f"{name}_batches"])
    return [item for item in acquisitions if item.name in names]


def scored_indexes(acquisition: Acquisition, window: int = REFERENCE_WINDOW) -> np.ndarray:
    """Samples the engine scores with a causal ``window``: a supervised sample whose window stays in one segment."""
    left = window - 1
    index = np.arange(left, len(acquisition.x))
    segment = acquisition.segment_id
    same = segment[index - left] == segment[index]
    return index[same & acquisition.mask[index].any(axis=1)]


def segment_runs(segment_id: np.ndarray, valid: np.ndarray | None = None) -> list[slice]:
    """Contiguous runs of one segment (and, with ``valid``, of valid samples only)."""
    keep = np.ones(len(segment_id), dtype=bool) if valid is None else valid.astype(bool)
    runs, start = [], None
    for row in range(len(segment_id) + 1):
        boundary = row == len(segment_id) or not keep[row] or (start is not None and segment_id[row] != segment_id[start])
        if start is not None and boundary:
            runs.append(slice(start, row))
            start = None
        if row < len(segment_id) and keep[row] and start is None:
            start = row
    return runs


def sampling_rate_hz(acquisitions: list[Acquisition]) -> float:
    """Median sampling rate of the trials (the dataset is on one native grid, about 200 Hz)."""
    steps = np.concatenate([np.diff(item.time_s) for item in acquisitions if len(item.time_s) > 1])
    return float(1.0 / np.median(steps[steps > 0]))


def write_fold(
    fold_dir: Path,
    data_dir: Path,
    groups: list[str],
    fold: dict,
    targets: tuple[str, ...],
    predictions: list[tuple[Acquisition, np.ndarray, np.ndarray]],
    metadata: dict,
) -> None:
    """Write the engine's fold layout.

    ``predictions`` holds, per test trial, its scored sample indexes and the predicted
    targets there (physical units, columns in ``targets`` order).
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "split": "full_cv",
        "split_level": SPLIT_LEVEL,
        "validation_group_count": VALIDATION_GROUP_COUNT,
        "groups": groups,
        "folds": [fold],
    }
    (fold_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (fold_dir / "metadata.json").write_text(
        json.dumps({"data_dir": str(data_dir), **metadata}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # Written under a temporary name first: a complete predictions.csv marks a finished fold.
    partial = fold_dir / "predictions.csv.partial"
    with open(partial, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_csv_fields(targets, False, False))
        writer.writeheader()
        for item, rows, predicted in predictions:
            outputs = (
                item.y[rows], predicted, item.mask[rows], item.time_s[rows],
                item.state_l[rows], item.state_r[rows], item.airborne_l[rows], item.airborne_r[rows],
                None, None,
            )
            writer.writerows(prediction_rows(item, fold["test_group"], targets, outputs))
    partial.replace(fold_dir / "predictions.csv")
