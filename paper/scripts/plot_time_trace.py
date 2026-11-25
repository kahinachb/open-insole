#!/usr/bin/env python3
"""A time trace of the 12 outputs of a Full CV run, for the median participant: prediction against truth.

    python paper/scripts/plot_time_trace.py RUN_DIR PER_PARTICIPANT_CSV OUTPUT.pdf [--run mlp]
        [--family lateral_1_appui] [--seconds 8] [--start-fraction 0.3]

RUN_DIR holds the folds' ``predictions.csv(.gz)``. The participant is the median one by the run's
score in PER_PARTICIPANT_CSV (mean NRMSE of the 12 targets, Measured samples, all tasks), so the
trace is neither the best nor the worst case. The window is taken in that participant's first
unloaded trial of FAMILY, starting at START_FRACTION of the trial. Samples off the force plates have
no truth; the air phase (true force zero) is shaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import paper_style  # noqa: E402

paper_style.apply()
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK_2, GRID, AIR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#efeeea"
PREDICTION = "#2a78d6"
COMPONENTS = {"Fx": "N", "Fy": "N", "Fz": "N", "Mx": "N·m", "My": "N·m", "Mz": "N·m"}
AIRBORNE = 2


def median_participant(per_participant: Path, run: str) -> str:
    table = pd.read_csv(per_participant)
    rows = table[(table.run == run) & (table.subset == "measured") & (table.task_family == "ALL")]
    scores = rows.groupby("participant").nrmse_range.mean().sort_values()
    return scores.index[len(scores) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("per_participant", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run", default="mlp")
    parser.add_argument("--family", default="lateral_1_appui")
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--start-fraction", type=float, default=0.3)
    args = parser.parse_args()

    participant = median_participant(args.per_participant, args.run)
    path = next(args.run_dir.glob(f"fold_{participant}/predictions.csv*"))
    predictions = pd.read_csv(path)
    trials = predictions[(predictions.task_family == args.family) & (predictions.test_group == participant)]
    unloaded = [b for b in trials.test_batch.unique() if not b.split("_")[0].endswith("b")]
    trial = trials[trials.test_batch == unloaded[0]].reset_index(drop=True)
    start = trial.time_s.iloc[int(len(trial) * args.start_fraction)]
    window = trial[(trial.time_s >= start) & (trial.time_s < start + args.seconds)]
    t = (window.time_s - start).to_numpy()

    fig, axes = plt.subplots(6, 2, figsize=(7.2, 8.0), sharex=True, facecolor=SURFACE)
    for col, foot in enumerate("LR"):
        air = (window[f"state_{foot}"] == AIRBORNE).to_numpy()
        for row, (component, unit) in enumerate(COMPONENTS.items()):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            edges = np.flatnonzero(np.diff(np.r_[0, air.astype(int), 0]))
            for begin, end in zip(edges[::2], edges[1::2]):
                ax.axvspan(t[begin], t[min(end, len(t) - 1)], color=AIR, lw=0, zorder=0)
            target = f"{component}_{foot}"
            truth = window[target].where(window[f"mask_{target}"] == 1)
            ax.plot(t, truth, color=INK, lw=1.2, zorder=2, label="Force plates (truth)")
            ax.plot(t, window[f"pred_{target}"], color=PREDICTION, lw=1.2, zorder=3, label="Our MLP (prediction)")
            ax.set_ylabel(f"${component[0]}_{{{component[1]}}}$ ({unit})", color=INK, fontsize=8)
            ax.grid(True, axis="y", color=GRID, lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=7)
        axes[0, col].set_title(f"{'Left' if foot == 'L' else 'Right'} foot", color=INK, fontsize=9)
        axes[-1, col].set_xlabel("Time (s)", color=INK, fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=AIR))
    labels.append("Foot in the air")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(f"{participant} {unloaded[0]} from {start:.1f} s")


if __name__ == "__main__":
    main()
