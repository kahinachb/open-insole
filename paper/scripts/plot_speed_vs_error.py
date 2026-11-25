"""Speed vs error of our MLP and the competitors: predictions per second against the force error in %BW.

x: RMSE in %BW of Fx, Fy and Fz, per foot on Measured samples, averaged over feet, then over the
15 test Participants (``evaluate_grfm`` per_participant.csv). y: predictions per second in
streaming, one CPU thread, from ``benchmark_inference.py`` (log scale); Mahmud's GPU speed is
added as open markers. The dashed line is the sensor rate: above it, a method keeps up in real time.

Usage: python plot_speed_vs_error.py PER_PARTICIPANT_CSV SPEED_CSV OUTPUT_PNG
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paper_style  # noqa: E402

paper_style.apply()
import pandas as pd
from matplotlib.lines import Line2D

SAMPLE_RATE_HZ = 200
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHODS = {  # run name: (label, colour), categorical slots in fixed order
    "mlp": ("Ours: MLP, causal 100 ms", "#2a78d6"),
    "sabbah_loadsol": ("Sabbah 2024: Random Forest, offline", "#eb6834"),
    "mahmud_gauges": ("Mahmud 2026: Ke2KeNet, load cells, offline", "#1baf7a"),
    "mahmud_raw": ("Mahmud 2026: Ke2KeNet, 78 channels, offline", "#eda100"),
}
COMPONENTS = {"Fx": "o", "Fy": "s", "Fz": "^"}


def errors(per_participant: Path) -> pd.DataFrame:
    frame = pd.read_csv(per_participant)
    rows = frame[(frame["subset"] == "measured") & (frame["task_family"] == "ALL")
                 & frame["component"].isin(list(COMPONENTS)) & frame["run"].isin(list(METHODS))]
    per = rows.groupby(["run", "participant", "component"])["rmse_pct_bw"].mean()
    return per.groupby(["run", "component"]).mean().unstack()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("per_participant", type=Path)
    parser.add_argument("speed", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    error = errors(args.per_participant)
    speed = pd.read_csv(args.speed).set_index(["method", "device"])["predictions_per_second"]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axhspan(SAMPLE_RATE_HZ, 1e5, color="#2a78d6", alpha=0.05, lw=0)
    ax.axhline(SAMPLE_RATE_HZ, color=INK_2, lw=1, ls="--")
    ax.text(5.25, SAMPLE_RATE_HZ * 1.25, f"real time: sensor rate {SAMPLE_RATE_HZ} Hz", color=INK_2, fontsize=8, ha="right")
    for run, (label, colour) in METHODS.items():
        y = speed[(run, "cpu")]
        xs = [error.loc[run, c] for c in COMPONENTS]
        ax.plot([min(xs), max(xs)], [y, y], color=colour, lw=2, alpha=0.5, zorder=2)
        for component, marker in COMPONENTS.items():
            ax.scatter(error.loc[run, component], y, marker=marker, s=70, color=colour,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
        if (run, "cuda") in speed.index and run.startswith("mahmud"):
            gpu = speed[(run, "cuda")]
            for component, marker in COMPONENTS.items():
                ax.scatter(error.loc[run, component], gpu, marker=marker, s=70, facecolor="none",
                           edgecolor=colour, linewidth=1.5, zorder=3)
            ax.plot([min(xs), max(xs)], [gpu, gpu], color=colour, lw=1, ls=":", zorder=2)
    ax.set_yscale("log")
    ax.set_ylim(0.2, 3e4)
    ax.set_xlim(1.3, 5.3)
    ax.set_xlabel("Force error, RMSE (% body weight)", color=INK)
    ax.set_ylabel("Predictions per second, streaming (log)", color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2)
    method_handles = [Line2D([], [], color=c, lw=6, label=label) for label, c in METHODS.values()]
    component_handles = [Line2D([], [], marker=m, ls="", color=INK_2, label=c) for c, m in COMPONENTS.items()]
    device_handles = [Line2D([], [], marker="o", ls="", color=INK_2, label="1 CPU thread"),
                      Line2D([], [], marker="o", ls="", markerfacecolor="none", color=INK_2, label="GPU (RTX 4070)")]
    first = ax.legend(handles=method_handles, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2,
                      frameon=False, fontsize=8)
    ax.add_artist(first)
    ax.legend(handles=component_handles + device_handles, loc="upper left", bbox_to_anchor=(0, -0.27), ncol=5,
              frameon=False, fontsize=8)
    ax.set_title("Speed vs accuracy: only our method is real time", color=INK, fontsize=11, loc="left")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(error.round(2).to_string())


if __name__ == "__main__":
    main()
