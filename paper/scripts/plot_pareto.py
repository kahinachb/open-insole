"""Pareto front of the sensor configurations: error against the number of sensor channels.

Each fold of the Full CV scores the 4 095 configurations of the 12 sensor groups with the
sensor-dropout MLP, on its 2 validation participants (evaluate_sensor_configurations.py; ADR 0004).
A configuration's score is taken relative to the full insole of the same fold, so that the folds
can be pooled. Grey dots: each configuration, median over the folds. Thin lines: the Pareto front
of each fold (best score at each channel count or fewer). Thick line: median of the fold fronts.
Markers: the 28- and 34-channel configurations of the paper (median over the folds); dashed line:
the 10 % tolerance of the selection.

Usage: python plot_pareto.py CONFIGURATIONS_CSV OUTPUT_PDF
  CONFIGURATIONS_CSV: test_participant, configuration, channels, mean_nrmse (one row per fold and configuration)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style  # noqa: E402

paper_style.apply()

FULL = 78
TOLERANCE_PCT = 10
MARKED = {  # sensor groups of the paper's configurations
    "28 ch.": {"heel_gauges", "forefoot_gauges", "toes_gauges", "midfoot_acc"},
    "34 ch.": {"heel_gauges", "forefoot_gauges", "toes_gauges", "midfoot_acc", "toes_acc"},
}
INK, INK_2, GRID, ACCENT, POINTS = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#b9b8b3"
MARKER_COLOURS = {"28 ch.": "#d1495b", "34 ch.": "#eda100"}


def relative_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Each configuration's score in percent of the full insole of the same fold."""
    full = frame[frame.channels == FULL].set_index("test_participant").mean_nrmse
    frame = frame.copy()
    frame["change_pct"] = 100 * (frame.mean_nrmse / frame.test_participant.map(full) - 1)
    frame["groups"] = frame.configuration.str.split().map(frozenset)
    return frame


def pareto_front(scores: pd.DataFrame) -> pd.Series:
    """Lowest change at each channel count, using that many channels or fewer (non-increasing)."""
    best = scores.groupby("channels").change_pct.min().sort_index()
    return best.cummin()


def main() -> None:
    source, output = Path(sys.argv[1]), Path(sys.argv[2])
    frame = relative_scores(pd.read_csv(source))
    folds = sorted(frame.test_participant.unique())
    median = frame.groupby(["configuration", "channels"]).change_pct.median().reset_index()

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    ax.scatter(median.channels, median.change_pct, s=3, color=POINTS, lw=0, alpha=0.6, zorder=1,
               label="Configurations (median over folds)")
    fronts = pd.DataFrame({fold: pareto_front(frame[frame.test_participant == fold]) for fold in folds})
    fronts = fronts.sort_index().ffill()
    for fold in folds:
        ax.plot(fronts.index, fronts[fold], color=ACCENT, lw=0.4, alpha=0.35, zorder=2)
    ax.plot(fronts.index, fronts.median(axis=1), color=ACCENT, lw=1.6, zorder=3, label="Pareto front (median of folds)")
    ax.axhline(TOLERANCE_PCT, color=INK_2, lw=0.7, ls="--", zorder=2)
    ax.text(FULL, TOLERANCE_PCT + 1.5, f"+{TOLERANCE_PCT} %", ha="right", va="bottom", fontsize=7, color=INK_2)
    for label, groups in MARKED.items():
        rows = frame[frame.groups == frozenset(groups)]
        if rows.empty:
            raise ValueError(f"configuration {label} not found: {sorted(groups)}")
        ax.scatter(rows.channels.iloc[0], rows.change_pct.median(), s=28, marker="D", color=MARKER_COLOURS[label],
                   edgecolor="white", lw=0.6, zorder=4, label=f"{label} configuration")
    ax.scatter(FULL, 0, s=40, marker="*", color=INK, zorder=4, label="Full insole (78 ch.)")
    ax.set_xlabel("Sensor channels (both insoles)", fontsize=8)
    ax.set_ylabel("NRMSE vs full insole (%)", fontsize=8)
    ax.set_ylim(-5, 100)
    ax.set_xlim(0, 82)
    ax.tick_params(labelsize=7, length=2)
    ax.grid(color=GRID, lw=0.4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=6, frameon=False, loc="upper right")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
