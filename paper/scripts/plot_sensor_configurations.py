"""Selection score per Sensor configuration (Full CV): one thin line per Participant, mean ± SD on top.

Reads the configurations.json written by evaluate_sensor_configurations.py, one per model.

Usage: python plot_sensor_configurations.py OUTPUT_PNG LABEL=CONFIGURATIONS_JSON [LABEL=CONFIGURATIONS_JSON]
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np

ORDER = ["patches_1", "minimal", "patches_2", "patches_3", "patches_4", "full"]
TICKS = {"patches_1": "1 Patch\n(forefoot)", "minimal": "Minimal\n(2 Patches, 22 ch)",
         "patches_2": "2 Patches", "patches_3": "3 Patches", "patches_4": "4 Patches", "full": "Full\n(4 Patches)"}
COLORS = ["#2a6fdb", "#d1603d"]


def main(argv: list[str]) -> int:
    output, runs = argv[0], [argument.split("=", 1) for argument in argv[1:]]
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    x = np.arange(len(ORDER))
    for index, (label, path) in enumerate(runs):
        configurations = json.load(open(path, encoding="utf-8"))["configurations"]
        participants = sorted(configurations["full"]["participants"])
        scores = np.array([[100 * configurations[c]["participants"][p] for c in ORDER] for p in participants])
        offset = (index - (len(runs) - 1) / 2) * 0.12
        for row in scores:
            ax.plot(x + offset, row, color=COLORS[index], alpha=0.18, lw=0.8)
        mean, sd = scores.mean(axis=0), scores.std(axis=0, ddof=1)
        ax.errorbar(x + offset, mean, yerr=sd, color=COLORS[index], lw=2, marker="o", ms=6, capsize=3, label=label)
        for xi, m in zip(x + offset, mean):
            ax.annotate(f"{m:.2f}", (xi, m), textcoords="offset points", xytext=(-28 if offset < 0 else 8, -3), fontsize=7, color="#333")
    ax.set_xticks(x, [TICKS[c] for c in ORDER], fontsize=8)
    ax.set_ylabel("Mean NRMSE of the 12 targets (%)")
    ax.set_title("Sensor dropout at test time, Full CV (15 test Participants)", fontsize=10)
    ax.grid(axis="y", color="#e5e5e5")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.text(0.01, 0.01, "Thin lines: one per Participant. Dots: mean ± SD over Participants. "
             "One model per line, trained once with Sensor dropout; the left-out groups are zeroed at test time.",
             fontsize=6, color="#555")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
