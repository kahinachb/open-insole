"""Duration and sample count of each trial of the dataset, for the paper.

Usage: python paper/trial_durations.py DATASET_DIR [OUTPUT_CSV]

Writes one row per trial (no participant name), read by make_artefacts.py.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parent


def trial_durations(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(dataset_dir).glob("*.npz")):
        time_s = np.load(path)["target_time_s"]
        rows.append({"duration_s": round(float(time_s[-1] - time_s[0]), 3), "n_samples": len(time_s)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    output = Path(sys.argv[2]) if len(sys.argv) > 2 else PAPER / "evaluation" / "trial_durations.csv"
    trial_durations(Path(sys.argv[1])).to_csv(output, index=False)
