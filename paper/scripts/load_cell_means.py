#!/usr/bin/env python3
"""Mean force of each load cell while its foot is loaded, per Participant (input of the Sensor-study numbers).

A sample counts as loaded when the load-cell sum of its foot exceeds 50 N. Every Trial of a Participant,
with and without the backpack, is pooled; each Trial weighs by its number of loaded samples.

    python paper/scripts/load_cell_means.py DATASET_DIR paper/evaluation/sensor_study/load_cells.csv
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LAYOUT = Path(__file__).resolve().parents[2] / "insole2grfm" / "tools" / "insole_layout.json"
LOADED_N = 50.0


def participant(trial: Path) -> str:
    """``P05b_squat`` (loaded Subject) and ``P05_squat`` are the same Participant."""
    return re.sub(r"b$", "", trial.stem.split("_")[0])


def load_cell_means(dataset: Path) -> pd.DataFrame:
    layout = json.loads(LAYOUT.read_text())
    names, patch = layout["feature_names"], {}
    for foot in layout["feet"].values():
        patch.update({s["channel"]: s["patch"] for s in foot["sensors"]})
    sums: dict[tuple, list[float]] = {}
    for trial in sorted(dataset.glob("*.npz")):
        data = np.load(trial)
        x = data["X"]
        for foot in "LR":
            loaded = data[f"insole_gauge_sum_n_{foot}"] > LOADED_N
            if not loaded.any():
                continue
            for i, name in enumerate(names):
                if name.startswith(f"{foot}_") and "force" in name:
                    entry = sums.setdefault((participant(trial), foot, name), [0.0, 0.0])
                    entry[0] += float(x[loaded, i].sum())
                    entry[1] += float(loaded.sum())
    return pd.DataFrame(
        [
            {"participant": p, "foot": f, "channel": c, "patch": patch[c], "mean_force_n": s / n, "n_samples": n}
            for (p, f, c), (s, n) in sorted(sums.items())
        ]
    )


if __name__ == "__main__":
    dataset, output = Path(sys.argv[1]), Path(sys.argv[2])
    load_cell_means(dataset).to_csv(output, index=False)
    print(f"load-cell means -> {output}")
