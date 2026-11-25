#!/usr/bin/env python3
"""Metrological characterisation of the insole, from data already recorded (#35-#37).

Each subcommand writes tables (CSV), a ``summary.json`` and figures to
``--output-dir``. Everything is computed on raw counts, before negative loads
are clipped when preparing the training data, so negative offsets stay
visible.

``tare``: every Tare recording of the acquisition repo's ``manip`` folder is
decoded to per-Gauge Tare counts (median over the recording); their spread
across Tares measures the stability of the Tare between sessions. Each
feet-raised control recording (``dataSemelle.bin``) gives, after subtracting
the Tare assigned to its Subject, the residual offset the model actually saw.
``--compare SUBJECT:TARE`` reports how much that Subject's loads would shift
with another Tare (per Gauge and per foot, in N).

``gain``: in each static trial (quiet standing on the plates), each insole's
gauge sum (median raw counts over the middle half of the recording, minus
the assigned Tare, nominal sensitivity) against the plates' vertical force
(median over the C3D, ``static_plate_fz.csv``). The total gain (both feet)
needs no pairing; per foot, the recordings do not say which plate was under
which foot, so the heavier insole is paired with the heavier plate
(``plate_assignment = inferred``). A robust (Theil-Sen) line through the
trials, insole = gain x plate + intercept, gives the gain, the intercept and
the residual, for both feet together and for every foot pooled (the per-foot
pairing is inferred, so left and right are not fitted apart). One static
trial per Subject is too few for a line per Participant: only gains.

``sync``: the synchronisation reports of every trial
(``synchronized_v2/rapport_synchronisation.json``): left/right pairing
time difference, sampling rate, lost and duplicated packets, and the
insole-to-plate offset with its spread over contacts.

``drift``: relative drift of the two insole clocks within each trial of the
prepared dataset. After alignment on the plates, each paired sample carries
both insoles' times on the plate clock; the slope of their difference over
the trial is the relative drift of the two clocks, and slope x duration the
drift accumulated over the trial. Only aggregates are written: the trial
names of the dataset are not pseudonymised.

``latency``: per-window inference time of the MLP and the patch Transformer
(batch of one, after warm-up), on CPU and, when available, GPU.

``allan``: overlapping Allan deviation of each gyroscope axis on the static
gyroscope recordings; angle random walk (deviation at 1 s) and bias
instability (minimum deviation / 0.664).

``thermal``: each Gauge's Tare against the mean temperature of its Patch in
the same recording (barometer temperature, same frames); the slope is the
thermal sensitivity of the Tare between sessions.

The binary frames are decoded with the acquisition repo's own Tare decoder
(``--decoder-dir``), not re-implemented here.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grfm_schema import participant_of_subject

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MANIP = WORKSPACE / "manip"
DEFAULT_DECODER = WORKSPACE / "acquisition" / "insole_data"
DEFAULT_ASSIGNMENTS = WORKSPACE / "acquisition" / "tare_assignments.json"
DEFAULT_OUTPUT = WORKSPACE / "docs" / "analysis" / "characterisation"
TARE_FILES = {"gauche": "tare_fx29_gauche.bin", "droite": "tare_fx29_droit.bin"}
# Inference latency is compared with the insoles' sample period (200 Hz).
SAMPLE_PERIOD_MS = 5.0
CONTROL_FILES = ("dataSemelle.bin", "dataSemell.bin")
SIDES = {"gauche": "L", "droite": "R"}


def load_module(directory: Path, name: str):
    """A module of the acquisition repo (``name``.py in ``directory``)."""
    if not (directory / f"{name}.py").exists():
        raise FileNotFoundError(f"{name}.py absent de {directory}")
    sys.path.insert(0, str(directory))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(directory))


def load_decoder(decoder_dir: Path):
    """The acquisition repo's FX29 Tare decoder module."""
    return load_module(decoder_dir, "calculerTareFX29")


def tare_folders(manip_dir: Path):
    """(folder name, {foot: recording}) of every folder holding both Tare recordings."""
    for folder in sorted(path for path in manip_dir.iterdir() if path.is_dir()):
        paths = {foot: folder / name for foot, name in TARE_FILES.items()}
        if all(path.exists() for path in paths.values()):
            yield folder.name, paths


def write_summary(output_dir: Path, summary: dict) -> int:
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results: {output_dir.resolve()}")
    return 0


def describe(values: pd.Series, *quantiles: float) -> dict[str, float]:
    """Median, max (and min, or the given quantiles) of a column, as plain floats."""
    out = {"median": float(values.median()), "max": float(values.max())}
    if quantiles:
        out.update({f"p{round(100 * q)}": float(values.quantile(q)) for q in quantiles})
    else:
        out["min"] = float(values.min())
    return out


def gauges(decoder):
    """(foot, patch identifier, gauge index, channel name) of every installed Gauge."""
    for foot, identifiers in decoder.PATCHES.items():
        for identifier in identifiers:
            for index in range(4):
                if index not in decoder.MISSING.get(identifier, set()):
                    yield foot, identifier, index, f"{SIDES[foot]}_p{identifier[0]}_force_{index + 1}_N"


def newtons_per_count(decoder, identifier, index) -> float:
    return decoder.capacity_lbf(identifier, index) * decoder.LBF_TO_N / decoder.NOMINAL_SPAN


def median_counts(decoder, samples) -> dict[str, float]:
    """Per-Gauge median raw counts of a decoded recording."""
    return {
        name: float(statistics.median(samples[identifier][index]))
        for _, identifier, index, name in gauges(decoder)
        if samples[identifier][index]
    }


def decode_tares(decoder, manip_dir: Path) -> dict[str, dict[str, float]]:
    """Tare counts per Gauge of every Tare folder."""
    tares = {}
    for name, paths in tare_folders(manip_dir):
        counts = {}
        for foot, path in paths.items():
            samples, _ = decoder.read_samples(path)
            counts.update({
                channel: value for channel, value in median_counts(decoder, samples).items()
                if channel.startswith(SIDES[foot])
            })
        tares[name] = counts
    return tares


def tare_tables(decoder, tares) -> tuple[pd.DataFrame, pd.DataFrame]:
    scale = {name: newtons_per_count(decoder, identifier, index) for _, identifier, index, name in gauges(decoder)}
    readings = pd.DataFrame(
        [
            {"tare": tare, "channel": name, "tare_counts": value, "tare_N_from_nominal": (value - decoder.NOMINAL_ZERO) * scale[name]}
            for tare, values in tares.items()
            for name, value in values.items()
        ]
    )
    grouped = readings.groupby("channel", sort=False)["tare_counts"]
    spread = pd.DataFrame({
        "tares": grouped.count(),
        "mean_counts": grouped.mean(),
        "std_counts": grouped.std(ddof=0),
        "range_counts": grouped.max() - grouped.min(),
    })
    spread["N_per_count"] = [scale[name] for name in spread.index]
    for column in ("std", "range"):
        spread[f"{column}_N"] = spread[f"{column}_counts"] * spread["N_per_count"]
    return readings, spread.reset_index()


def control_residuals(decoder, manip_dir: Path, tares, assignments) -> pd.DataFrame:
    rows = []
    for folder in sorted(path for path in manip_dir.iterdir() if path.is_dir()):
        control = next((folder / name for name in CONTROL_FILES if (folder / name).exists()), None)
        if control is None:
            continue
        tare = assignments.get(folder.name)
        if tare not in tares:
            print(f"SKIP {folder.name}: tare affectee {tare!r} introuvable", file=sys.stderr)
            continue
        samples, _ = decoder.read_samples(control)
        for _, identifier, index, name in gauges(decoder):
            values = np.asarray(samples[identifier][index], float)
            if not len(values) or name not in tares[tare]:
                continue
            residual = values - tares[tare][name]
            scale = newtons_per_count(decoder, identifier, index)
            rows.append({
                "subject": folder.name,
                "tare": tare,
                "channel": name,
                "samples": len(values),
                "residual_counts": float(np.median(residual)),
                "residual_p5_counts": float(np.percentile(residual, 5)),
                "residual_p95_counts": float(np.percentile(residual, 95)),
                "residual_N": float(np.median(residual)) * scale,
            })
    return pd.DataFrame(rows)


def tare_comparison(decoder, tares, assignments, subject: str, other: str) -> dict:
    assigned = assignments.get(subject)
    if assigned not in tares or other not in tares:
        raise ValueError(f"--compare {subject}:{other}: tare introuvable ({assigned!r} ou {other!r})")
    shifts, feet = {}, {"L": 0.0, "R": 0.0}
    for _, identifier, index, name in gauges(decoder):
        if name in tares[assigned] and name in tares[other]:
            shift = (tares[other][name] - tares[assigned][name]) * newtons_per_count(decoder, identifier, index)
            shifts[name] = shift
            feet[name[0]] += shift
    return {
        "subject": subject,
        "assigned_tare": assigned,
        "other_tare": other,
        # Positive: the assigned Tare reads lower, so the subject's loads were over-estimated by this much.
        "gauge_shift_N": shifts,
        "foot_sum_shift_N": feet,
    }


def middle_half_median(values) -> float:
    values = np.asarray(values, float)
    quarter = len(values) // 4
    return float(np.median(values[quarter : len(values) - quarter] if quarter else values))


def static_gains(decoder, manip_dir: Path, tares, assignments, plates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    plate_medians = plates.groupby(["subject", "trial"])[["force_z_p1_global_N", "force_z_p2_global_N"]].median()
    for recording in sorted(manip_dir.glob("*/static*/dataSemelle.bin")):
        subject, trial = recording.parent.parent.name, recording.parent.name
        tare = assignments.get(subject)
        if tare not in tares or (subject, trial) not in plate_medians.index:
            print(f"SKIP {subject}/{trial}: tare {tare!r} ou plateformes absentes", file=sys.stderr)
            continue
        samples, _ = decoder.read_samples(recording)
        insole = {"L": 0.0, "R": 0.0}
        for _, identifier, index, name in gauges(decoder):
            if samples[identifier][index] and name in tares[tare]:
                counts = middle_half_median(samples[identifier][index]) - tares[tare][name]
                insole[name[0]] += counts * newtons_per_count(decoder, identifier, index)
        plate = dict(zip(("p1", "p2"), plate_medians.loc[(subject, trial)].tolist()))
        heavy_insole = max(insole, key=insole.get)
        heavy_plate = max(plate, key=plate.get)
        pairing = {heavy_insole: heavy_plate, ("R" if heavy_insole == "L" else "L"): ("p2" if heavy_plate == "p1" else "p1")}
        rows.append({
            "subject": subject,
            "participant": participant_of_subject(subject),
            "trial": trial,
            "tare": tare,
            "insole_L_N": insole["L"],
            "insole_R_N": insole["R"],
            "insole_total_N": insole["L"] + insole["R"],
            "plate_p1_N": plate["p1"],
            "plate_p2_N": plate["p2"],
            "plate_total_N": plate["p1"] + plate["p2"],
            "gain_total": (insole["L"] + insole["R"]) / (plate["p1"] + plate["p2"]),
            "plate_L": pairing["L"],
            "plate_R": pairing["R"],
            "plate_assignment": "inferred",
            "gain_L": insole["L"] / plate[pairing["L"]],
            "gain_R": insole["R"] / plate[pairing["R"]],
        })
    return pd.DataFrame(rows)


def gain_fit(insole: pd.Series, plate: pd.Series) -> dict[str, float]:
    """Robust line insole = gain x plate + intercept (Theil-Sen) and the RMS / median absolute residual."""
    from scipy.stats import theilslopes

    gain, intercept, _, _ = theilslopes(insole.to_numpy(float), plate.to_numpy(float))
    residual = insole.to_numpy(float) - (gain * plate.to_numpy(float) + intercept)
    return {
        "gain": float(gain), "intercept_N": float(intercept),
        "residual_rms_N": float(np.sqrt(np.mean(residual**2))),
        "residual_median_abs_N": float(np.median(np.abs(residual))),
    }


def per_foot(gains: pd.DataFrame) -> pd.DataFrame:
    """One row per (trial, foot): insole and plate of that foot (pairing inferred)."""
    rows = []
    for side in ("L", "R"):
        frame = gains[["subject", "participant", "trial"]].copy()
        frame["foot"] = side
        frame["insole_N"] = gains[f"insole_{side}_N"]
        frame["plate_N"] = [row[f"plate_{row[f'plate_{side}']}_N"] for _, row in gains.iterrows()]
        frame["gain"] = gains[f"gain_{side}"]
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def plot_gains(gains: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(gains["plate_total_N"], gains["insole_total_N"], "o")
    for _, row in gains.iterrows():
        axis.annotate(row["subject"], (row["plate_total_N"], row["insole_total_N"]), fontsize=6)
    limit = float(max(gains["plate_total_N"].max(), gains["insole_total_N"].max())) * 1.05
    axis.plot([0, limit], [0, limit], "k:", linewidth=0.8, label="gain 1")
    axis.set_xlabel("plates, vertical force (N)")
    axis.set_ylabel("both insoles, gauge sum, nominal sensitivity (N)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_gain(args) -> int:
    decoder = load_decoder(args.decoder_dir)
    assignments = json.loads(args.tare_assignments.read_text(encoding="utf-8"))["assignments"]
    tares = decode_tares(decoder, args.manip_dir)
    gains = static_gains(decoder, args.manip_dir, tares, assignments, pd.read_csv(args.plate_fz))
    if gains.empty:
        raise ValueError("aucun trial statique exploitable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gains.to_csv(args.output_dir / "static_gain.csv", index=False)
    plot_gains(gains, args.output_dir / "static_gain.png")
    feet = per_foot(gains)
    feet.to_csv(args.output_dir / "static_gain_per_foot.csv", index=False)
    by_participant = feet.groupby(["participant", "foot"])["gain"].mean().unstack("foot")
    by_participant["total"] = gains.groupby("participant")["gain_total"].mean()
    by_participant.round(4).to_csv(args.output_dir / "static_gain_per_participant.csv")
    return write_summary(args.output_dir, {
        "trials": len(gains),
        "gain_total": describe(gains["gain_total"]),
        "gain_per_foot_inferred_pairing": describe(feet["gain"]),
        "fit_total": gain_fit(gains["insole_total_N"], gains["plate_total_N"]),
        "fit_per_foot_inferred_pairing": gain_fit(feet["insole_N"], feet["plate_N"]),
        "gain_per_participant": {
            participant: {key: round(float(value), 4) for key, value in row.items()}
            for participant, row in by_participant.iterrows()
        },
    })


def sync_table(manip_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(manip_dir.glob("*/*/synchronized_v2/rapport_synchronisation.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        quality = report.get("quality_control", {})
        offsets = report.get("offset_estimation", {}).get("per_side", {})
        row = {
            "subject": path.parents[2].name,
            "trial": path.parents[1].name,
            "status": report.get("status"),
            "left_right_max_abs_ms": quality.get("left_right_synchronization", {}).get("max_abs_time_difference_ms"),
            "missing_packets": report.get("packet_loss_assessment", {}).get("total_missing_packets"),
            "median_contact_correlation": report.get("offset_estimation", {}).get("median_contact_correlation"),
        }
        for side, key in (("L", "left_insole"), ("R", "right_insole")):
            insole = report.get(key, {})
            row[f"frequency_{side}_hz"] = insole.get("frequency_hz")
            row[f"duplicates_{side}"] = insole.get("duplicates_removed")
            row[f"packets_{side}"] = insole.get("packets_after_cleaning")
            estimate = offsets.get(side, {})
            row[f"plate_offset_{side}_ms"] = estimate.get("offset_ms")
            if estimate.get("q25_ms") is not None and estimate.get("q75_ms") is not None:
                row[f"plate_offset_iqr_{side}_ms"] = estimate["q75_ms"] - estimate["q25_ms"]
        rows.append(row)
    return pd.DataFrame(rows)


def run_sync(args) -> int:
    table = sync_table(args.manip_dir)
    if table.empty:
        raise ValueError(f"aucun rapport de synchronisation dans {args.manip_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "sync_reports.csv", index=False)
    packets = float(table[["packets_L", "packets_R"]].sum().sum())
    summary = {
        "trials": len(table),
        "left_right_max_abs_ms": describe(table["left_right_max_abs_ms"], 0.95),
        "frequency_hz": {
            "median": float(pd.concat([table["frequency_L_hz"], table["frequency_R_hz"]]).median()),
        },
        "packet_loss_rate": float(table["missing_packets"].sum()) / packets if packets else None,
        "duplicate_rate": float(table[["duplicates_L", "duplicates_R"]].sum().sum()) / packets if packets else None,
        "plate_offset_abs_ms": {
            "median": float(pd.concat([table["plate_offset_L_ms"], table["plate_offset_R_ms"]]).abs().median()),
        },
        "plate_offset_iqr_ms": {
            "median": float(pd.concat([table.get("plate_offset_iqr_L_ms"), table.get("plate_offset_iqr_R_ms")]).median()),
        },
        "median_contact_correlation": float(table["median_contact_correlation"].median()),
    }
    return write_summary(args.output_dir, summary)


def clock_drift(dataset_dir: Path, min_duration_s: float = 10.0) -> pd.DataFrame:
    """Per trial of the prepared dataset: duration, relative clock drift (ms/min) and drift accumulated (ms)."""
    rows = []
    for path in sorted(Path(dataset_dir).glob("*.npz")):
        with np.load(path, allow_pickle=True) as data:
            if "target_time_qtm_s_L" not in data.files:
                continue
            left, right = data["target_time_qtm_s_L"], data["target_time_qtm_s_R"]
        elapsed = left - left[0]
        if len(elapsed) < 10 or elapsed[-1] < min_duration_s:
            continue
        slope_ms_per_s = float(np.polyfit(elapsed, (right - left) * 1000.0, 1)[0])
        rows.append({
            "duration_s": float(elapsed[-1]),
            "drift_ms_per_min": 60.0 * slope_ms_per_s,
            "accumulated_drift_ms": abs(slope_ms_per_s) * float(elapsed[-1]),
        })
    return pd.DataFrame(rows)


def run_drift(args) -> int:
    table = clock_drift(args.dataset_dir, args.min_duration)
    if table.empty:
        raise ValueError(f"aucun essai avec les temps des deux semelles dans {args.dataset_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "clock_drift.csv", index=False)
    return write_summary(args.output_dir, {
        "trials": len(table),
        "drift_ms_per_min_abs": describe(table["drift_ms_per_min"].abs(), 0.95),
        "accumulated_drift_ms": describe(table["accumulated_drift_ms"], 0.95),
    })


LATENCY_MODELS = {
    # The frozen baseline MLP and patch Transformer v1 (78 channels + body weight and its flag).
    "mlp": dict(model_type="mlp", hidden_size=256, layers=2),
    "patch_transformer": dict(
        model_type="patch_transformer", hidden_size=128, layers=3, transformer_heads=4,
        transformer_feedforward_size=512, patch_samples=5,
    ),
}


def measure_latency(model_type: str, window: int, device: str, repeats: int, warmup: int, **options) -> dict:
    import time

    import torch

    from sensor_tokens import DEFAULT_LAYOUT
    from train_lstm_two_feet_masked import BODY_WEIGHT_FEATURE, BODY_WEIGHT_MISSING_FEATURE, build_model

    layout = json.loads(DEFAULT_LAYOUT.read_text(encoding="utf-8"))
    names = list(layout["feature_names"]) + [BODY_WEIGHT_FEATURE, BODY_WEIGHT_MISSING_FEATURE]
    model = build_model(model_type, len(names), window, dropout=0.0, output_size=12, feature_names=names, **options)
    model = model.to(device).eval()
    x = torch.randn(1, window, len(names), device=device)
    timings = []
    with torch.no_grad():
        for repeat in range(warmup + repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            if repeat >= warmup:
                timings.append((time.perf_counter() - start) * 1e3)
    return {
        "model": model_type, "window_samples": window, "device": device,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "median_ms": float(np.median(timings)), "p95_ms": float(np.percentile(timings, 95)),
    }


def run_latency(args) -> int:
    import torch

    devices = [device for device in args.devices if device != "cuda" or torch.cuda.is_available()]
    rows = [
        measure_latency(window=window, device=device, repeats=args.repeats, warmup=args.warmup, **options)
        for options in LATENCY_MODELS.values()
        for window in args.windows
        for device in devices
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "latency.csv", index=False)
    return write_summary(args.output_dir, {"rows": rows, "sample_period_ms": SAMPLE_PERIOD_MS})


def allan_deviation(rate: np.ndarray, period_s: float, cluster_sizes) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping Allan deviation of a rate signal at the given cluster sizes (samples)."""
    taus, deviations = [], []
    for size in cluster_sizes:
        averages = np.convolve(rate, np.ones(size) / size, mode="valid")
        differences = averages[size:] - averages[:-size]
        if len(differences) < 2:
            break
        taus.append(size * period_s)
        deviations.append(np.sqrt(0.5 * np.mean(np.square(differences))))
    return np.asarray(taus), np.asarray(deviations)


def allan_table(recordings: list[Path], period_s: float) -> tuple[pd.DataFrame, dict]:
    rows, curves = [], {}
    for path in recordings:
        data = pd.read_csv(path)
        foot = "L" if int(data["ID1"].min()) < 9 else "R"
        for patch, samples in data.groupby("ID1"):
            sizes = np.unique(np.logspace(0, np.log10(max(len(samples) // 9, 2)), 40).astype(int))
            for axis in ("X", "Y", "Z"):
                rate = samples[f"gy{axis} (°/s)"].to_numpy(float)
                taus, deviations = allan_deviation(rate - rate.mean(), period_s, sizes)
                curves[f"{foot}_p{patch}_{axis}"] = (taus, deviations)
                at_one = float(np.interp(1.0, taus, deviations)) if taus[0] <= 1.0 <= taus[-1] else math.nan
                rows.append({
                    "recording": path.parent.name, "foot": foot, "patch": int(patch), "axis": axis,
                    "samples": len(rate), "duration_s": len(rate) * period_s,
                    # Rate noise density in deg/s/sqrt(Hz) equals the deviation at tau = 1 s; x 60 for deg/sqrt(h).
                    "angle_random_walk_deg_per_sqrt_h": at_one * 60.0,
                    "bias_instability_deg_per_h": float(deviations.min()) / 0.664 * 3600.0,
                    "bias_instability_tau_s": float(taus[int(np.argmin(deviations))]),
                })
    return pd.DataFrame(rows), curves


def run_allan(args) -> int:
    recordings = sorted(args.gyro_dir.glob("session_gyro_*/initial_static.csv"))
    if not recordings:
        raise ValueError(f"aucun enregistrement statique gyroscope dans {args.gyro_dir}")
    table, curves = allan_table(recordings, 1.0 / args.frequency)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "allan.csv", index=False)
    figure, axis = plt.subplots(figsize=(6, 5))
    for name, (taus, deviations) in curves.items():
        axis.loglog(taus, deviations, linewidth=0.7, label=name)
    axis.set_xlabel("tau (s)")
    axis.set_ylabel("Allan deviation (deg/s)")
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=4, ncol=3)
    figure.tight_layout()
    figure.savefig(args.output_dir / "allan.png", dpi=160)
    plt.close(figure)
    return write_summary(args.output_dir, {
        "recordings": [str(path.parent.name) for path in recordings],
        "angle_random_walk_deg_per_sqrt_h": describe(table["angle_random_walk_deg_per_sqrt_h"]),
        "bias_instability_deg_per_h": describe(table["bias_instability_deg_per_h"]),
    })


def patch_temperatures(decoder, path: Path, get_temperature) -> dict[int, float]:
    """Mean temperature (degC) of each Patch over a recording, from the barometer bytes of its frames."""
    data = path.read_bytes()
    raw: dict[int, list[int]] = {}
    index = 0
    while index + decoder.PACKET_SIZE <= len(data):
        if data[index] in (0, 1) and decoder.packet_is_valid(data, index + 1):
            start = index + 1
        elif decoder.packet_is_valid(data, index):
            start = index
        else:
            index += 1
            continue
        for frame in range(4):
            offset = start + frame * decoder.FRAME_SIZE
            patch = data[offset]
            raw.setdefault(patch, []).append(data[offset + 26] * 65536 + (data[offset + 27] << 8 | data[offset + 28]))
        index = start + decoder.PACKET_SIZE
    return {patch: float(np.mean([get_temperature(value, patch) for value in values])) for patch, values in raw.items()}


def run_thermal(args) -> int:
    decoder = load_decoder(args.decoder_dir)
    get_temperature = load_module(args.decoder_dir, "fonctionsBarometreConversion").getTemp
    rows = []
    for tare, paths in tare_folders(args.manip_dir):
        for foot, path in paths.items():
            samples, _ = decoder.read_samples(path)
            counts = median_counts(decoder, samples)
            temperatures = patch_temperatures(decoder, path, get_temperature)
            for gauge_foot, identifier, index, name in gauges(decoder):
                if gauge_foot == foot and name in counts and identifier[0] in temperatures:
                    rows.append({
                        "tare": tare, "channel": name, "patch": identifier[0],
                        "temperature_C": temperatures[identifier[0]],
                        "tare_N": counts[name] * newtons_per_count(decoder, identifier, index),
                    })
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError(f"aucune tare dans {args.manip_dir}")
    slopes = []
    for name, group in table.groupby("channel", sort=False):
        spread = float(group["temperature_C"].max() - group["temperature_C"].min())
        slope = float(np.polyfit(group["temperature_C"], group["tare_N"], 1)[0]) if spread > 0.0 else math.nan
        slopes.append({"channel": name, "tares": len(group), "temperature_range_C": spread, "slope_N_per_C": slope})
    slopes = pd.DataFrame(slopes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "tare_temperatures.csv", index=False)
    slopes.to_csv(args.output_dir / "thermal_slopes.csv", index=False)
    return write_summary(args.output_dir, {
        "temperature_C": {"min": float(table["temperature_C"].min()), "max": float(table["temperature_C"].max())},
        "slope_abs_N_per_C": describe(slopes["slope_N_per_C"].abs()),
    })


def plot_spread(readings: pd.DataFrame, path: Path) -> None:
    channels = list(dict.fromkeys(readings["channel"]))
    figure, axis = plt.subplots(figsize=(max(8, 0.35 * len(channels)), 4))
    mean = readings.groupby("channel")["tare_counts"].mean()
    for tare, rows in readings.groupby("tare"):
        centred = rows.set_index("channel")["tare_counts"] - mean
        axis.plot([channels.index(name) for name in centred.index], centred.values, "o", markersize=3, label=tare)
    axis.set_xticks(range(len(channels)), [name.removesuffix("_N") for name in channels], rotation=90, fontsize=7)
    axis.set_ylabel("Tare - mean over Tares (counts)")
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.legend(fontsize=6, ncol=4)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_tare(args) -> int:
    decoder = load_decoder(args.decoder_dir)
    assignments = json.loads(args.tare_assignments.read_text(encoding="utf-8"))["assignments"]
    tares = decode_tares(decoder, args.manip_dir)
    if not tares:
        raise ValueError(f"aucune tare dans {args.manip_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    readings, spread = tare_tables(decoder, tares)
    residuals = control_residuals(decoder, args.manip_dir, tares, assignments)
    comparisons = [tare_comparison(decoder, tares, assignments, subject, other) for subject, other in args.compare]
    readings.to_csv(args.output_dir / "tare_readings.csv", index=False)
    spread.to_csv(args.output_dir / "tare_spread.csv", index=False)
    residuals.to_csv(args.output_dir / "control_residuals.csv", index=False)
    plot_spread(readings, args.output_dir / "tare_spread.png")
    summary = {
        "tares": sorted(tares),
        "controls": sorted(residuals["subject"].unique()) if len(residuals) else [],
        "tare_range_N": describe(spread["range_N"]),
        "tare_std_N": describe(spread["std_N"]),
        "control_residual_N": (
            {
                "median_abs": float(residuals["residual_N"].abs().median()),
                "max_abs": float(residuals["residual_N"].abs().max()),
            }
            if len(residuals) else None
        ),
        "comparisons": comparisons,
    }
    return write_summary(args.output_dir, summary)


def subject_and_tare(text: str) -> tuple[str, str]:
    subject, separator, tare = text.partition(":")
    if not (subject and separator and tare):
        raise argparse.ArgumentTypeError(f"attendu SUBJECT:TARE, recu {text!r}")
    return subject, tare


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = root.add_subparsers(dest="command", required=True)
    recordings = argparse.ArgumentParser(add_help=False)
    recordings.add_argument("--manip-dir", type=Path, default=DEFAULT_MANIP)
    recordings.add_argument("--decoder-dir", type=Path, default=DEFAULT_DECODER)
    assigned = argparse.ArgumentParser(add_help=False, parents=[recordings])
    assigned.add_argument("--tare-assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    tare = commands.add_parser(
        "tare", parents=[assigned], help="stabilite des tares et offset residuel (controles pieds leves)"
    )
    tare.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "tare")
    tare.add_argument(
        "--compare", action="append", default=[], type=subject_and_tare, metavar="SUBJECT:TARE",
        help="decalage des charges de SUBJECT si on lui appliquait TARE au lieu de sa tare affectee",
    )
    tare.set_defaults(run=run_tare)
    gain = commands.add_parser("gain", parents=[assigned], help="erreur de gain de la sensibilite nominale, trials statiques")
    gain.add_argument("--plate-fz", type=Path, default=Path(__file__).with_name("static_plate_fz.csv"))
    gain.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "gain")
    gain.set_defaults(run=run_gain)
    sync = commands.add_parser("sync", help="synchronisation, paquets perdus, alignement semelles/plateformes")
    sync.add_argument("--manip-dir", type=Path, default=DEFAULT_MANIP)
    sync.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "sync")
    sync.set_defaults(run=run_sync)
    drift = commands.add_parser("drift", help="derive relative des horloges des deux semelles, par essai")
    drift.add_argument("--dataset-dir", type=Path, default=WORKSPACE / "dataset_v3.7.1")
    drift.add_argument("--min-duration", type=float, default=10.0, help="essais plus courts ignores (s)")
    drift.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "drift")
    drift.set_defaults(run=run_drift)
    latency = commands.add_parser("latency", help="latence d'inference par fenetre")
    latency.add_argument("--windows", type=int, nargs="+", default=[21, 51, 101])
    latency.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    latency.add_argument("--repeats", type=int, default=500)
    latency.add_argument("--warmup", type=int, default=50)
    latency.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "latency")
    latency.set_defaults(run=run_latency)
    allan = commands.add_parser("allan", help="variance d'Allan des gyroscopes (enregistrements statiques)")
    allan.add_argument("--gyro-dir", type=Path, default=WORKSPACE / "calibration_imu" / "data")
    allan.add_argument("--frequency", type=float, default=200.0, help="frequence par Patch (Hz)")
    allan.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "allan")
    allan.set_defaults(run=run_allan)
    thermal = commands.add_parser(
        "thermal", parents=[recordings], help="tare des Gauges contre la temperature des Patches"
    )
    thermal.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "thermal")
    thermal.set_defaults(run=run_thermal)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return args.run(args)
    except (OSError, ValueError, KeyError) as exc:
        print(f"erreur: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
