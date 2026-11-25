#!/usr/bin/env python3
"""Each participant's score under each method: the spread behind the mean +- SD of the main table.

    python paper/scripts/plot_participant_scores.py EVALUATION_DIR OUTPUT.pdf

EVALUATION_DIR is the release's ``paper/evaluation``. A participant's score is the mean NRMSE of the
12 targets (Measured samples, all tasks). Participants are ordered by our MLP's score, so that each
column compares the methods on the same person.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import paper_style  # noqa: E402

paper_style.apply()
import pandas as pd  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHODS = {  # run: (evaluation dir, label, colour, marker); colours as in plot_speed_vs_error.py
    "mlp": ("competitors_mahmud", "Ours: MLP, causal", "#2a78d6", "o"),
    "sabbah_loadsol": ("competitors_mahmud", "Random Forest, loadsol-like", "#eb6834", "s"),
    "sabbah_raw": ("competitors_sabbah", "Random Forest, 78 ch.", "#e87ba4", "D"),
    "mahmud_gauges": ("competitors_mahmud", "Ke2KeNet, load cells", "#1baf7a", "^"),
    "mahmud_raw": ("competitors_mahmud", "Ke2KeNet, 78 ch.", "#eda100", "v"),
}


def scores(evaluation: Path) -> pd.DataFrame:
    """Participant x method: mean NRMSE (%) of the 12 targets."""
    columns = {}
    for run, (directory, *_) in METHODS.items():
        table = pd.read_csv(evaluation / directory / "per_participant.csv")
        rows = table[(table.run == run) & (table.subset == "measured") & (table.task_family == "ALL")]
        columns[run] = rows.groupby("participant").nrmse_range.mean() * 100
    return pd.DataFrame(columns).sort_values("mlp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("evaluation", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    table = scores(args.evaluation)
    x = range(len(table))

    fig, ax = plt.subplots(figsize=(7.2, 3.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for run, (_, label, colour, marker) in METHODS.items():
        ours = run == "mlp"
        ax.scatter(x, table[run], marker=marker, s=46 if ours else 32, color=colour, edgecolor=SURFACE,
                   linewidth=1.2, zorder=4 if ours else 3, label=label)
    ax.set_xticks(list(x), table.index, fontsize=8)
    ax.set_xlabel("Test participant, ordered by the score of our MLP", color=INK)
    ax.set_ylabel("NRMSE, mean of 12 outputs (%)", color=INK)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, frameon=False, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(table.round(2).to_string())


if __name__ == "__main__":
    main()
