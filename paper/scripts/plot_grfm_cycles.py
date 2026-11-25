"""Cycle-averaged 6D GRFM per task family: ground truth vs model, mean ± SD over test Participants.

For each trial, cycles are cut on the ground-truth (plate) signal that oscillates most clearly
for that task family; the prediction is cut at the same instants. Each cycle is resampled to
0-100 %, cycles are averaged per Participant, then mean ± SD is taken over Participants.
Masked samples are excluded (NaN); Airborne samples keep their zero target.

Optional --compare overlays a second prediction of the same samples (e.g. body weight missing) as a
dashed line, cut at the same instants.

Usage: python plot_grfm_cycles.py PREDICTIONS_CSV OUTPUT_DIR [--label "MLP (all)"]
           [--compare PREDICTIONS_CSV --compare-label LABEL --compare-run RUN [--panel-tags MAIN COMPARE]]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

COMPONENTS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
UNITS = {"Fx": "N", "Fy": "N", "Fz": "N", "Mx": "N·m", "My": "N·m", "Mz": "N·m"}
FEET = ["L", "R"]
FAMILY_TITLES = {
    "hulahoop": "Hula hoop",
    "squat": "Squat",
    "balance": "Balance (ankle / hip strategies)",
    "lateral_2supports": "Lateral double-support",
    "lateral_1_appui": "Lateral single-support",
    "jump": "Jump",
    "walk": "Walk",
}
# Candidate reference signals for cycle detection (ground truth only).
CANDIDATES = {
    "Fz sum": lambda d: d["Fz_L"] + d["Fz_R"],
    "Fz L": lambda d: d["Fz_L"],
    "Fx sum": lambda d: d["Fx_L"] + d["Fx_R"],
    "Fy L": lambda d: d["Fy_L"],
    "Fy sum": lambda d: d["Fy_L"] + d["Fy_R"],
}
POINTS = 101
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRUTH, MODEL, COMPARE = "#2a78d6", "#eb6834", "#1baf7a"


def lowpass(signal: np.ndarray, rate: float, cutoff: float = 2.0) -> np.ndarray:
    b, a = butter(2, cutoff / (rate / 2))
    return filtfilt(b, a, signal)


def dominant_period(signal: np.ndarray, rate: float) -> tuple[float, float]:
    """Period (s) of the strongest 0.15-2 Hz component and its share of the signal power."""
    centred = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(centred)) ** 2
    freqs = np.fft.rfftfreq(len(centred), 1 / rate)
    band = (freqs >= 0.15) & (freqs <= 2.0)
    if not band.any() or spectrum[1:].sum() == 0:
        return np.nan, 0.0
    peak = np.argmax(np.where(band, spectrum, 0))
    window = (freqs > freqs[peak] * 0.8) & (freqs < freqs[peak] * 1.2)
    return 1 / freqs[peak], spectrum[window].sum() / spectrum[1:].sum()


def trial_arrays(trial: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {}
    for foot in FEET:
        for c in COMPONENTS:
            name = f"{c}_{foot}"
            valid = trial[f"mask_{name}"].to_numpy() > 0
            out[name] = np.where(valid, trial[name].to_numpy(), np.nan)
            out[f"pred_{name}"] = np.where(valid, trial[f"pred_{name}"].to_numpy(), np.nan)
            if f"alt_{name}" in trial:
                out[f"alt_{name}"] = np.where(valid, trial[f"alt_{name}"].to_numpy(), np.nan)
    return out


def choose_reference(trials: list[pd.DataFrame], rate: float) -> str:
    scores = {}
    for name, build in CANDIDATES.items():
        shares = []
        for trial in trials:
            ref = build(trial).to_numpy()
            if np.isnan(ref).mean() > 0.2 or len(ref) < rate * 4:
                continue
            ref = pd.Series(ref).interpolate(limit_direction="both").to_numpy()
            shares.append(dominant_period(ref, rate)[1])
        scores[name] = np.mean(shares) if shares else 0.0
    return max(scores, key=scores.get)


def cut_cycles(trial: pd.DataFrame, reference: str, rate: float) -> list[tuple[int, int]]:
    ref = CANDIDATES[reference](trial).to_numpy()
    if np.isnan(ref).mean() > 0.2 or len(ref) < rate * 4:
        return []
    ref = lowpass(pd.Series(ref).interpolate(limit_direction="both").to_numpy(), rate)
    period, _ = dominant_period(ref, rate)
    if not np.isfinite(period):
        return []
    minima, _ = find_peaks(-ref, distance=max(1, int(0.6 * period * rate)))
    cycles = []
    for start, end in zip(minima[:-1], minima[1:]):
        if 0.6 * period * rate <= end - start <= 1.6 * period * rate:
            cycles.append((start, end))
    return cycles


def resample(values: np.ndarray) -> np.ndarray | None:
    valid = ~np.isnan(values)
    if valid.mean() < 0.8:
        return None
    x = np.linspace(0, 1, len(values))
    return np.interp(np.linspace(0, 1, POINTS), x[valid], values[valid])


LANDING_WINDOW_S = (-0.6, 1.0)


def landing_curves(frame: pd.DataFrame, rate: float):
    """Jump: windows aligned on each landing (both feet back on the plates after an Airborne phase)."""
    before, after = (int(round(s * rate)) for s in LANDING_WINDOW_S)
    per_participant: dict[str, dict[str, list[np.ndarray]]] = {}
    counts: dict[str, int] = {}
    for (participant, _), trial in frame.groupby(["test_group", "test_batch"]):
        trial = trial.sort_values("time_s")
        arrays = trial_arrays(trial)
        airborne = ((trial["state_L"] == 2) & (trial["state_R"] == 2)).to_numpy()
        measured = ((trial["state_L"] == 1) & (trial["state_R"] == 1)).to_numpy()
        flight_end = np.flatnonzero(airborne[:-1] & ~airborne[1:]) + 1
        for index in flight_end:
            later = np.flatnonzero(measured[index:index + int(0.2 * rate)])
            if len(later) == 0:
                continue
            landing = index + later[0]
            start, end = landing + before, landing + after
            if start < 0 or end > len(trial):
                continue
            store = per_participant.setdefault(participant, {})
            for name, values in arrays.items():
                store.setdefault(name, []).append(values[start:end])
            counts[participant] = counts.get(participant, 0) + 1
    means = {p: {n: np.nanmean(c, axis=0) for n, c in s.items()} for p, s in per_participant.items()}
    return "landing (both feet back on the plates)", means, counts


STRIKE_WINDOW_S = (-0.4, 1.2)


def strike_curves(frame: pd.DataFrame, rate: float):
    """Walk: each foot aligned on its own plate strikes (state turns Measured after >= 0.1 s off the plate)."""
    before, after = (int(round(s * rate)) for s in STRIKE_WINDOW_S)
    quiet = int(0.1 * rate)
    per_participant: dict[str, dict[str, list[np.ndarray]]] = {}
    counts: dict[str, int] = {}
    for (participant, _), trial in frame.groupby(["test_group", "test_batch"]):
        trial = trial.sort_values("time_s")
        arrays = trial_arrays(trial)
        for foot in FEET:
            on_plate = (trial[f"state_{foot}"] == 1).to_numpy()
            strikes = np.flatnonzero(~on_plate[:-1] & on_plate[1:]) + 1
            for strike in strikes:
                start, end = strike + before, strike + after
                if start < 0 or end > len(trial) or on_plate[strike - quiet:strike].any():
                    continue
                store = per_participant.setdefault(participant, {})
                for name, values in arrays.items():
                    if name.endswith(f"_{foot}"):
                        store.setdefault(name, []).append(values[start:end])
                key = f"{participant} {foot}"
                counts[key] = counts.get(key, 0) + 1
    means = {p: {n: np.nanmean(c, axis=0) for n, c in s.items()} for p, s in per_participant.items()}
    return "each foot's plate strike", means, counts


EVENT_WINDOWS = {"jump": LANDING_WINDOW_S, "walk": STRIKE_WINDOW_S}
EVENT_AXIS = {"jump": "Time from landing (s)", "walk": "Time from plate strike of that foot (s)"}


def family_curves(frame: pd.DataFrame, rate: float):
    if frame["task_family"].iloc[0] == "jump":
        return landing_curves(frame, rate)
    if frame["task_family"].iloc[0] == "walk":
        return strike_curves(frame, rate)
    trials = {key: t.sort_values("time_s") for key, t in frame.groupby(["test_group", "test_batch"])}
    reference = choose_reference(list(trials.values()), rate)
    per_participant: dict[str, dict[str, list[np.ndarray]]] = {}
    counts: dict[str, int] = {}
    for (participant, _), trial in trials.items():
        arrays = trial_arrays(trial)
        for start, end in cut_cycles(trial, reference, rate):
            store = per_participant.setdefault(participant, {})
            kept = {}
            for name, values in arrays.items():
                curve = resample(values[start:end])
                if curve is None and not name.startswith(("pred_", "alt_")):
                    kept = None
                    break
                kept[name] = curve
            if kept is None:
                continue
            counts[participant] = counts.get(participant, 0) + 1
            for name, curve in kept.items():
                if curve is not None:
                    store.setdefault(name, []).append(curve)
    participant_means = {
        p: {name: np.mean(curves, axis=0) for name, curves in store.items()} for p, store in per_participant.items()
    }
    return reference, participant_means, counts


def panel_metrics(summary: pd.DataFrame, run: str, family: str) -> dict[str, str]:
    """RMSE, range-NRMSE and per-trial PCC per target, from evaluate_grfm's summary (Measured subset)."""
    rows = summary[(summary["run"] == run) & (summary["subset"] == "measured") & (summary["task_family"] == family)]
    out = {}
    for foot in FEET:
        for c in COMPONENTS:
            name = f"{c}_{foot}"
            values = rows[rows["target"] == name].set_index("metric")["mean"]
            rmse, nrmse, pcc = (values.get(m, np.nan) for m in ("rmse", "nrmse_range", "pcc_trial_mean"))
            out[name] = f"RMSE {rmse:.2f} {UNITS[c]} · NRMSE {100 * nrmse:.1f} % · PCC {pcc:.2f}"
    return out


def plot_family(family, frame, rate, label, output_dir, split_note, metrics, compare_label=None, compare_metrics=None,
                title_label=None, panel_tags=("known", "missing")):
    reference, means, counts = family_curves(frame, rate)
    if len(means) == 0:
        print(f"{family}: no usable cycle, skipped")
        return None
    event_aligned = family in EVENT_WINDOWS
    length = len(next(iter(next(iter(means.values())).values())))
    phase = np.linspace(*EVENT_WINDOWS[family], length) if event_aligned else np.linspace(0, 100, POINTS)
    fig, axes = plt.subplots(6, 2, figsize=(11, 14), sharex=True, facecolor=SURFACE)
    for col, foot in enumerate(FEET):
        for row, c in enumerate(COMPONENTS):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            name = f"{c}_{foot}"
            series = [(name, TRUTH, True), (f"pred_{name}", MODEL, True)]
            if compare_label:
                series.append((f"alt_{name}", COMPARE, False))
            for key, colour, band in series:
                stack = np.array([m[key] for m in means.values() if key in m])
                mean = np.nanmean(stack, axis=0)
                sd = np.nanstd(stack, axis=0, ddof=1) if len(stack) > 1 else np.zeros_like(mean)
                if band:
                    ax.fill_between(phase, mean - sd, mean + sd, color=colour, alpha=0.18, linewidth=0)
                    ax.plot(phase, mean, color=colour, linewidth=2)
                else:
                    ax.plot(phase, mean, color=colour, linewidth=1.6, linestyle=(0, (4, 2)))
            ax.grid(True, color=GRID, linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_2, labelsize=8)
            ax.set_ylabel(f"{c} ({UNITS[c]})", color=INK, fontsize=9)
            title = metrics[name]
            if compare_label:
                title = f"{panel_tags[0]}: {title}\n{panel_tags[1]}: {compare_metrics[name]}"
            ax.set_title(title, loc="right", fontsize=7 if compare_label else 7.5, color=INK_2, pad=3)
            if row == 0:
                ax.set_title("Left foot" if foot == "L" else "Right foot", loc="left", color=INK, fontsize=11,
                             fontweight="bold", pad=3)
            if event_aligned:
                ax.axvline(0, color=INK_2, linewidth=0.8, linestyle=":")
            if row == 5:
                ax.set_xlabel(EVENT_AXIS[family] if event_aligned else "Cycle (%)", color=INK, fontsize=9)
    handles = [plt.Line2D([], [], color=TRUTH, lw=2), plt.Line2D([], [], color=MODEL, lw=2)]
    labels = ["Ground truth (force plate)", label]
    if compare_label:
        handles.append(plt.Line2D([], [], color=COMPARE, lw=1.6, linestyle=(0, (4, 2))))
        labels.append(compare_label)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=10, labelcolor=INK)
    cycles = ", ".join(f"{p}: {n} cycles" for p, n in sorted(counts.items()))
    fig.suptitle(f"{FAMILY_TITLES.get(family, family)}: 6D GRFM, ground truth vs {title_label or label}",
                 color=INK, fontsize=13, y=0.99)
    if event_aligned:
        how = f"aligned on {reference}; Masked samples (no plate reference) excluded"
        cycles = cycles.replace("cycles", "landings" if family == "jump" else "strikes")
    else:
        how = f"cycle-averaged, cycles cut on ground-truth {reference}"
    band = "Mean (line) ± SD (band) over test Participants" + (", dashed line: mean only" if compare_label else "")
    fig.text(0.5, 0.005,
             f"{band}, {how}. {cycles}. {split_note.rstrip('.')}. "
             "Panel metrics: Measured samples of this task, per Participant then mean; NRMSE = RMSE / (p99.5 − p0.5 "
             "range); PCC = per-trial Pearson r, averaged.",
             ha="center", va="bottom", fontsize=7.5, color=INK_2, wrap=True)
    fig.tight_layout(rect=(0, 0.04 if compare_label else 0.025, 1, 0.95))
    path = output_dir / f"{family}.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"{family}: reference {reference}, {cycles} -> {path}")
    return path


def read_predictions(path: Path, columns: list[str]) -> pd.DataFrame:
    """One predictions CSV, or every fold's ``predictions.csv[.gz]`` under a run dir, pooled."""
    if path.is_file():
        return pd.read_csv(path, usecols=columns)
    plain = set(path.rglob("predictions.csv"))  # a fold holding both forms is read once
    files = sorted(plain | {f for f in path.rglob("predictions.csv.gz") if f.with_suffix("") not in plain})
    if not files:
        raise FileNotFoundError(f"no predictions.csv under {path}")
    return pd.concat([pd.read_csv(f, usecols=columns) for f in files], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, help="predictions CSV, or a run dir whose fold predictions are pooled")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--label", default="MLP")
    parser.add_argument("--split-note", default="")
    parser.add_argument("--families", nargs="+", default=list(FAMILY_TITLES))
    parser.add_argument("--summary", type=Path, required=True, help="evaluate_grfm summary.csv holding this run")
    parser.add_argument("--run", required=True, help="run name in that summary")
    parser.add_argument("--compare", type=Path, help="second predictions CSV of the same samples, drawn dashed")
    parser.add_argument("--compare-label", default="Compare")
    parser.add_argument("--compare-run", help="run name of --compare in --summary")
    parser.add_argument("--title-label", help="model name in the figure title (default: --label)")
    parser.add_argument("--panel-tags", nargs=2, default=("known", "missing"), metavar=("MAIN", "COMPARE"),
                        help="short names of the two models in the per-panel metrics (default: known missing)")
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["test_batch", "test_group", "task_family", "time_s", "state_L", "state_R"]
    for foot in FEET:
        for c in COMPONENTS:
            columns += [f"{c}_{foot}", f"pred_{c}_{foot}", f"mask_{c}_{foot}"]
    frame = read_predictions(args.predictions, columns)
    if args.compare:
        if not args.compare_run:
            parser.error("--compare needs --compare-run")
        preds = [f"pred_{c}_{foot}" for foot in FEET for c in COMPONENTS]
        other = read_predictions(args.compare, ["test_batch", "time_s", *preds])
        # Runs with different windows predict different trial starts: keep the samples both cover.
        other = other.rename(columns={p: p.replace("pred_", "alt_") for p in preds})
        frame = frame.merge(other, on=["test_batch", "time_s"], how="inner", validate="one_to_one")
        if frame.empty:
            parser.error("--compare shares no sample with the predictions")
    step = frame.sort_values(["test_batch", "time_s"]).groupby("test_batch")["time_s"].diff().median()
    rate = float(np.clip(1 / step, 50, 2000))
    for family in args.families:
        subset = frame[frame["task_family"] == family]
        if subset.empty:
            print(f"{family}: not in predictions")
            continue
        plot_family(family, subset, rate, args.label, args.output_dir, args.split_note,
                    panel_metrics(summary, args.run, family),
                    args.compare_label if args.compare else None,
                    panel_metrics(summary, args.compare_run, family) if args.compare else None,
                    args.title_label, tuple(args.panel_tags))


if __name__ == "__main__":
    main()
