"""Mahmud 2026's metrics on our walk trials: PCC and NRMSE between mean stance curves.

Mahmud 2026 (Table I) does not score samples. Each gait cycle is resampled to 100 points and
the cycles are averaged; the metrics then compare the mean measured curve with the mean
predicted one. NRMSE is the RMSE of both curves range-normalised to 0-1.

On our walk trials each foot steps on its own plate, so a cycle here is one stance on a plate:
from the strike (the state turns Measured after >= 0.1 s off the plate) to the toe-off (it
stops being Measured). Each stance is resampled to 101 points. Per Participant, foot and
component, the stances are averaged. Both mean curves are normalised by the measured curve's
range, then PCC and NRMSE are computed, as in Mahmud's eq. 12 and 13. The results are averaged
over feet, then mean ± SD over Participants.

Usage: python mean_cycle_metrics.py OUTPUT_CSV --run mlp=RUN_DIR --run mahmud=RUN_DIR [...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COMPONENTS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
FEET = ["L", "R"]
POINTS = 101
MEASURED = 1
QUIET_S = 0.1
MIN_STANCE_S = 0.3


def walk_rows(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.glob("fold_*/predictions.csv")):
        for chunk in pd.read_csv(path, chunksize=500_000):
            frames.append(chunk[chunk["task_family"] == "walk"])
    return pd.concat(frames, ignore_index=True)


def stances(state: np.ndarray, time_s: np.ndarray) -> list[tuple[int, int]]:
    """(start, end) of each full stance: a strike after a quiet spell, then a toe-off."""
    on = state == MEASURED
    starts = np.flatnonzero(~on[:-1] & on[1:]) + 1
    out = []
    for start in starts:
        previous = np.flatnonzero(on[:start])
        if len(previous) and time_s[start] - time_s[previous[-1]] < QUIET_S:
            continue
        after = np.flatnonzero(~on[start:])
        if not len(after):
            continue
        end = start + after[0]
        if time_s[end - 1] - time_s[start] >= MIN_STANCE_S:
            out.append((start, end))
    return out


def resample(values: np.ndarray) -> np.ndarray:
    return np.interp(np.linspace(0, 1, POINTS), np.linspace(0, 1, len(values)), values)


def mean_curves(frame: pd.DataFrame) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, int]]:
    """(participant, foot, component) -> (mean measured curve, mean predicted curve, stance count)."""
    store: dict[tuple[str, str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for (participant, _), trial in frame.groupby(["test_group", "test_batch"]):
        trial = trial.sort_values("time_s")
        time_s = trial["time_s"].to_numpy()
        for foot in FEET:
            for start, end in stances(trial[f"state_{foot}"].to_numpy(), time_s):
                for component in COMPONENTS:
                    name = f"{component}_{foot}"
                    if (trial[f"mask_{name}"].to_numpy()[start:end] <= 0).any():
                        continue
                    store.setdefault((participant, foot, component), []).append(
                        (resample(trial[name].to_numpy()[start:end]), resample(trial[f"pred_{name}"].to_numpy()[start:end]))
                    )
    return {key: (np.mean([t for t, _ in pairs], axis=0), np.mean([p for _, p in pairs], axis=0), len(pairs))
            for key, pairs in store.items()}


def metrics(truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    low, span = truth.min(), np.ptp(truth)
    nrmse = float(np.sqrt(np.mean(((truth - low) / span - (pred - low) / span) ** 2)))
    return float(np.corrcoef(truth, pred)[0, 1]), nrmse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR (fold_*/predictions.csv)")
    args = parser.parse_args()
    rows = []
    for spec in args.run:
        name, path = spec.split("=", 1)
        for (participant, foot, component), (truth, pred, count) in mean_curves(walk_rows(Path(path))).items():
            pcc, nrmse = metrics(truth, pred)
            rows.append({"run": name, "participant": participant, "foot": foot, "component": component,
                         "stances": count, "pcc": pcc, "nrmse": nrmse})
        print(f"{name}: done", flush=True)
    table = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
