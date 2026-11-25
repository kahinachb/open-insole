#!/usr/bin/env python3
"""Errors of a Full CV run with and without the backpack, and around foot contact and lift-off.

    python paper/scripts/loading_and_transitions.py RUN_DIR OUTPUT_DIR [--window-ms 50]

RUN_DIR holds the folds' ``predictions.csv`` (or ``.csv.gz``) of the Full CV, e.g.
``results/full_cv_v3.7.1/mlp_sensor_dropout_w21/all`` (the Full CV predictions, not distributed). Writes in OUTPUT_DIR:

- ``loading.csv``: per Participant, condition (``unloaded``; ``backpack`` for a Subject whose code
  ends in ``b``) and target, the RMSE on Measured samples and their count;
- ``transitions.csv``: per Participant, foot and event, the vertical force error (pred - truth, N)
  of the samples within the window of a contact (Airborne -> Measured) or lift-off (Measured ->
  Airborne), and of the rest of stance and of the air phase;
- ``airborne_by_family.csv``: per Participant, foot and task family, the vertical force predicted
  while the foot is in the air (the truth is zero there).

Only samples labelled Measured (state 1) or Airborne (state 2) are scored. A contact or lift-off is a
change between them across at most ``MAX_TRANSITION_GAP_S`` of unscored samples or dropped rows; a
longer interval starts a new segment and is not a transition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MEASURED, AIRBORNE = 1, 2
TARGETS = [f"{c}_{f}" for f in "LR" for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]
SAMPLE_PERIOD_S = 0.005


def condition_of(batch: str) -> str:
    """``P03b_walk2`` (the loaded Subject of P03) carries the backpack; ``P03_walk2`` does not."""
    return "backpack" if batch.split("_")[0].endswith("b") else "unloaded"


MAX_TRANSITION_GAP_S = 0.2  # longest unscored interval (Excluded samples or dropped rows) inside a transition


def _boundaries(states: np.ndarray, time_s: np.ndarray) -> list[tuple[int, int, str]]:
    """Each change between Measured and Airborne: (last sample before, first sample after, event).

    The two samples are consecutive scored samples of different states, separated by at most
    ``MAX_TRANSITION_GAP_S``. Between them the foot is usually Excluded for a few samples (near the
    ground but not yet, or no longer, on the plate), or the rows are missing because both feet were
    Excluded; such a change is still a contact or a lift-off. A longer interval is a new segment.
    """
    states, time_s = np.asarray(states), np.asarray(time_s)
    scored = np.flatnonzero(np.isin(states, (MEASURED, AIRBORNE)))
    out = []
    for before, after in zip(scored[:-1], scored[1:]):
        if states[before] != states[after] and time_s[after] - time_s[before] <= MAX_TRANSITION_GAP_S:
            out.append((int(before), int(after), "lift_off" if states[after] == AIRBORNE else "contact"))
    return out


def transition_labels(states: np.ndarray, time_s: np.ndarray, window: int) -> np.ndarray:
    """Per sample: ``contact``/``lift_off`` within ``window`` samples of a change of state (before the
    last sample of the old state and after the first of the new one), else ``stance``/``air``; ``""``
    for samples that are neither Measured nor Airborne."""
    states = np.asarray(states)
    labels = np.full(states.size, "", dtype=object)
    labels[states == MEASURED] = "stance"
    labels[states == AIRBORNE] = "air"
    scored = np.isin(states, (MEASURED, AIRBORNE))
    for before, after, event in _boundaries(states, time_s):
        for span in (slice(max(before - window + 1, 0), before + 1), slice(after, min(after + window, states.size))):
            labels[span] = np.where(scored[span], event, labels[span])
    return labels


def _events(states: np.ndarray, time_s: np.ndarray) -> dict[str, int]:
    events = [event for _, _, event in _boundaries(states, time_s)]
    return {"lift_off": events.count("lift_off"), "contact": events.count("contact")}


def transition_errors(predictions: pd.DataFrame, window: int) -> pd.DataFrame:
    """Vertical force error per Participant, foot and event (see the module docstring)."""
    parts, events = [], []
    for batch, frame in predictions.groupby("test_batch", sort=False):
        participant = frame.test_group.iloc[0]
        time_s = frame.time_s.to_numpy()
        for foot in "LR":
            states = frame[f"state_{foot}"].to_numpy()
            labels = transition_labels(states, time_s, window)
            keep = labels != ""
            error = (frame[f"pred_Fz_{foot}"] - frame[f"Fz_{foot}"]).to_numpy()[keep]
            parts.append(pd.DataFrame({"participant": participant, "foot": foot, "event": labels[keep], "error": error}))
            for event, count in _events(states, time_s).items():
                events.append({"participant": participant, "foot": foot, "event": event, "n_events": count})
    errors = pd.concat(parts, ignore_index=True)
    table = errors.groupby(["participant", "foot", "event"]).error.agg(
        rmse_N=lambda e: float(np.sqrt(np.mean(np.square(e)))), mean_error_N="mean", n_samples="size"
    ).reset_index()
    counts = pd.DataFrame(events).groupby(["participant", "foot", "event"]).n_events.sum().reset_index()
    return table.merge(counts, on=["participant", "foot", "event"], how="left")


def loading_errors(predictions: pd.DataFrame, targets: list[str] = TARGETS) -> pd.DataFrame:
    """RMSE per Participant, condition and target, on Measured samples."""
    frame = predictions.assign(condition=predictions.test_batch.map(condition_of))
    rows = []
    for (participant, condition), group in frame.groupby(["test_group", "condition"]):
        for target in targets:
            measured = group[group[f"mask_{target}"] == 1]
            if f"state_{target[-1]}" in measured:
                measured = measured[measured[f"state_{target[-1]}"] == MEASURED]
            error = measured[f"pred_{target}"] - measured[target]
            rows.append({"participant": participant, "condition": condition, "target": target,
                         "rmse": float(np.sqrt(np.mean(np.square(error)))) if len(error) else float("nan"),
                         "n": len(error)})
    return pd.DataFrame(rows)


def airborne_by_family(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean and RMSE of the vertical force predicted in the air, per Participant, foot and task family."""
    rows = []
    for foot in "LR":
        air = predictions[predictions[f"state_{foot}"] == AIRBORNE]
        error = air[f"pred_Fz_{foot}"] - air[f"Fz_{foot}"]
        table = error.groupby([air.test_group, air.task_family]).agg(
            mean_error_N="mean", rmse_N=lambda e: float(np.sqrt(np.mean(np.square(e)))), n_samples="size"
        )
        rows.append(table.reset_index().rename(columns={"test_group": "participant"}).assign(foot=foot))
    return pd.concat(rows, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--window-ms", type=float, default=50.0)
    args = parser.parse_args(argv)
    window = int(round(args.window_ms / 1000 / SAMPLE_PERIOD_S))
    files = sorted(args.run_dir.glob("fold_*/predictions.csv*"))
    if not files:
        parser.error(f"{args.run_dir}: no fold_*/predictions.csv")
    loading, transitions, airborne = [], [], []
    for path in files:  # one fold at a time: all folds together do not fit in memory
        predictions = pd.read_csv(path)
        loading.append(loading_errors(predictions))
        transitions.append(transition_errors(predictions, window))
        airborne.append(airborne_by_family(predictions))
        print(f"{path.parent.name}: {len(predictions)} samples", file=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(loading).to_csv(args.output_dir / "loading.csv", index=False)
    pd.concat(transitions).to_csv(args.output_dir / "transitions.csv", index=False)
    pd.concat(airborne).to_csv(args.output_dir / "airborne_by_family.csv", index=False)
    run = "/".join(args.run_dir.resolve().parts[-2:])
    (args.output_dir / "README.md").write_text(
        f"Made by paper/scripts/loading_and_transitions.py from {run}, window {args.window_ms:g} ms.\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
