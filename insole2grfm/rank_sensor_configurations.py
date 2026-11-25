#!/usr/bin/env python3
"""Rank every Sensor configuration of a Sensor-dropout checkpoint on its validation Participants (#42, ADR 0004).

Reloads the fold checkpoint of a run trained with ``--sensor-dropout``
(``metadata.json`` + ``models/*.pt``), rebuilds the inputs of its validation
trials exactly as the engine does, and evaluates the model on all 4 095
non-empty Sensor configurations by inference only: the groups left out are
zeroed and their presence columns set to 0, the kept groups keep the full
configuration's values (Inactive sensors removed). The test Participants are
never read (ADR 0001).

Per configuration (``configurations.csv``): the groups kept, the Patches
physically present, the sensor channels (both feet), and on the Measured
subset the mean range-normalised NRMSE of the 12 targets and the force RMSE in
%BW. Each validation Participant is scored on its own samples and range, then
the Participants are averaged (``mean_nrmse_<Participant>`` keeps each one). The
engine's physical NMSE on the Valid subset is pooled, as the engine computes it:
with ``--window-stride 1`` the full configuration reproduces the checkpoint's
``validation_physical_nmse``; the default stride of 5 keeps one window in five,
so its scores are a close estimate.

``pareto.json``: the full configuration and the Minimal configuration, the
fewest channels whose mean NRMSE is within ``--tolerance`` (relative, 10 %) of
the full configuration's, the lower NRMSE breaking ties. ``pareto.png``: mean
NRMSE against channels, with the Pareto front, the full and Minimal
configurations, and any ``--reference NAME=CHANNELS:NRMSE`` (e.g. the
loadsol-like model, ``loadsol-like=8:0.12``).

Usage::

    python rank_sensor_configurations.py RUN_DIR --output-dir OUT [--device cuda] [--window-stride 5]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate_body_weight_modes import fold_inputs, load_model, run_arguments
from grfm_schema import FORCES
from train_lstm_two_feet_masked import (
    SENSOR_CHANNEL,
    SENSOR_PATCHES,
    STATE_PLATFORM,
    MaskedWindowDataset,
    SensorDropout,
)

RANGE_PERCENTILES = (0.5, 99.5)


def windows(saved: dict, trials: list, zero_weight: float, stride: int, full: SensorDropout):
    """Validation windows (full configuration), their standardised targets and weights, and sample keys."""
    dataset = MaskedWindowDataset(
        trials, saved["x_mean"], saved["x_std"], saved["y_mean"], saved["y_std"], list(range(len(saved["targets"]))),
        saved["window_samples"], not saved["causal"], zero_weight, target_scaling=saved["target_scaling"],
        stride=stride, sensor_dropout=full,
    )
    batches = list(DataLoader(dataset, batch_size=4096, shuffle=False))
    x = torch.cat([batch[0] for batch in batches])
    y = torch.cat([batch[1] for batch in batches])
    weight = torch.cat([batch[2] for batch in batches])
    acquisition = torch.cat([batch[3] for batch in batches]).numpy()
    sample = torch.cat([batch[4] for batch in batches]).numpy()
    return x, y, weight, acquisition, sample


def configurations(group_names: tuple[str, ...]):
    for size in range(len(group_names), 0, -1):
        yield from itertools.combinations(range(len(group_names)), size)


def predict_all(model, x: torch.Tensor, dropped_columns: list[int], device, batch_size: int) -> torch.Tensor:
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start : start + batch_size].to(device)
            if dropped_columns:
                batch = batch.clone()
                batch[:, :, dropped_columns] = 0.0
            outputs.append(model(batch).float().cpu())
    return torch.cat(outputs)


def participant_nrmse(error: torch.Tensor, mask: torch.Tensor, truth: torch.Tensor,
                      participant: torch.Tensor) -> torch.Tensor:
    """NRMSE of each target for each Participant (rows in code order), on that Participant's own samples and range."""
    rows = []
    for code in torch.unique(participant):
        own = participant == code
        e, m, y = error[own], mask[own], truth[own]
        rmse = torch.sqrt((e.square() * m).sum(0) / m.sum(0).clamp(min=1))
        ranges = torch.tensor([
            float(np.subtract(*np.percentile(y[m[:, index] > 0, index].numpy(), RANGE_PERCENTILES[::-1])))
            for index in range(y.shape[1])
        ])
        rows.append(rmse / ranges)
    return torch.stack(rows)


def pareto_summary(table: pd.DataFrame, tolerance: float) -> dict:
    """Full and Minimal configurations (ADR 0004): the fewest channels within the tolerance, lower NRMSE on ties."""
    def row(frame):
        return frame.iloc[0][["configuration", "group_count", "patches", "channels", "mean_nrmse"]].to_dict()

    ordered = table.sort_values(["channels", "mean_nrmse"])
    full = table[table["group_count"] == table["group_count"].max()]
    limit = float(full["mean_nrmse"].iloc[0]) * (1.0 + tolerance)
    within = ordered[ordered["mean_nrmse"] <= limit]
    return {
        "tolerance": tolerance,
        "mean_nrmse_limit": limit,
        "full": row(full),
        "minimal": row(within) if len(within) else None,
    }


def pareto_front(table: pd.DataFrame) -> pd.DataFrame:
    """Configurations no other beats on both channel count and mean NRMSE (lower NRMSE breaks ties)."""
    ordered = table.sort_values(["channels", "mean_nrmse"])
    return ordered[ordered["mean_nrmse"] < ordered["mean_nrmse"].cummin().shift(fill_value=np.inf)]


def plot(table: pd.DataFrame, summary: dict, references: dict[str, dict], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    x = table["channels"]
    axis.scatter(x, table["mean_nrmse"], s=3, alpha=0.25, color="grey", label="configurations")
    front = pareto_front(table)
    axis.plot(x[front.index], front["mean_nrmse"], "o-", markersize=3, label="Pareto front")
    for name, marker, entry in (("full configuration", "*", summary["full"]),
                                ("minimal configuration", "s", summary["minimal"])):
        if entry:
            index = table.index[table["configuration"] == entry["configuration"]][0]
            axis.plot(x[index], entry["mean_nrmse"], marker, markersize=10, fillstyle="none", label=name)
    axis.axhline(summary["mean_nrmse_limit"], color="black", linestyle=":", linewidth=0.8, label="full + tolerance")
    for name, reference in references.items():
        axis.plot(reference["channels"], reference["mean_nrmse"], "D", markersize=7, label=name)
    axis.set_xlabel("sensor channels (both insoles)")
    axis.set_ylabel("mean NRMSE of the 12 targets (validation, Measured)")
    axis.legend(fontsize=7)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def rank(run_dir: Path, data_dir: Path | None, device: torch.device, stride: int, batch_size: int):
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    (checkpoint_path,) = sorted((run_dir / "models").glob("*.pt"))
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not saved.get("sensor_dropout"):
        raise ValueError(f"{checkpoint_path.name}: entraine sans --sensor-dropout")
    if saved["target_scaling"] != "standard":
        raise ValueError("seul --target-scaling standard est pris en charge")
    run = run_arguments(metadata)
    data_dir = data_dir or Path(metadata["data_dir"])
    trials, targets, _, weights = fold_inputs(run, saved, data_dir, saved["validation_batches"])
    if not weights:
        # Model without the body-weight input: %BW still uses the dataset's body weights.
        subjects = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8")).get("body_weights", {}).get("subjects", {})
        weights = {subject: entry["body_weight_N"] for subject, entry in subjects.items() if entry.get("body_weight_N")}
    names = list(saved["feature_names"])
    full = SensorDropout.from_feature_names(names)
    model = load_model(run, saved, device)
    y_mean, y_std = torch.as_tensor(saved["y_mean"]), torch.as_tensor(saved["y_std"])
    physical = torch.as_tensor(saved["physical_y_std"])
    measured = windows(saved, trials, 0.0, stride, full)
    valid = windows(saved, trials, saved["zero_target_weight"], stride, full)
    truth = measured[1] * y_std + y_mean
    participants = sorted({trial.subject_identity for trial in trials})
    participant = torch.tensor([participants.index(trials[int(index)].subject_identity) for index in measured[3]])
    body_weight = torch.tensor([
        weights.get(trials[int(index)].subject_session, np.nan) for index in measured[3]
    ], dtype=torch.float32)
    forces = [targets.index(name) for name in FORCES]
    patch_of = {
        group: SENSOR_PATCHES.index(group.rsplit("_", 1)[0]) for group in full.group_names
    }
    rows = []
    for kept in configurations(full.group_names):
        dropped = [index for index in range(len(full.groups)) if index not in kept]
        columns = [column for index in dropped for column in full.groups[index]] + [full.presence_columns[index] for index in dropped]
        prediction = predict_all(model, measured[0], columns, device, batch_size) * y_std + y_mean
        error, mask = prediction - truth, (measured[2] > 0).float()
        nrmse = participant_nrmse(error, mask, truth, participant)
        force_pct = torch.stack([
            100.0 * torch.sqrt(
                ((error[own][:, forces] / body_weight[own, None]).square() * mask[own][:, forces]).sum(0)
                / mask[own][:, forces].sum(0).clamp(min=1)
            ).mean()
            for own in (participant == code for code in range(len(participants)))
        ])
        valid_prediction = predict_all(model, valid[0], columns, device, batch_size) * y_std + y_mean
        valid_truth = valid[1] * y_std + y_mean
        physical_nmse = float((((valid_prediction - valid_truth) / physical).square() * valid[2]).sum() / valid[2].sum())
        groups = [full.group_names[index] for index in kept]
        rows.append({
            "configuration": " ".join(groups),
            "group_count": len(groups),
            "patches": len({patch_of[group] for group in groups}),
            "channels": sum(
                1 for index in kept for column in full.groups[index] if SENSOR_CHANNEL.match(names[column])
            ),
            "mean_nrmse": float(nrmse.mean(1).mean()),
            "force_rmse_pct_bw": float(force_pct.mean()),
            "physical_nmse": physical_nmse,
            **{f"mean_nrmse_{name}": float(value) for name, value in zip(participants, nrmse.mean(1))},
            **{f"nrmse_{target}": float(value) for target, value in zip(targets, nrmse.mean(0))},
        })
    return pd.DataFrame(rows), participants, saved


def parse_reference(text: str) -> tuple[str, dict]:
    """``NAME=CHANNELS:NRMSE``: a model to place on the figure at its own sensor cost."""
    name, _, cost = text.partition("=")
    channels, _, value = cost.partition(":")
    try:
        if not name:
            raise ValueError
        return name, {"channels": int(channels), "mean_nrmse": float(value)}
    except ValueError:
        raise argparse.ArgumentTypeError(f"attendu NOM=CANAUX:NRMSE, recu {text!r}") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", type=Path, help="engine output dir of one fold (metadata.json, models/*.pt)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, help="prepared dataset (default: the run's data_dir)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--window-stride", type=int, default=5, help="one validation window every N samples")
    parser.add_argument("--tolerance", type=float, default=0.10, help="Minimal configuration: within this of full")
    parser.add_argument("--reference", action="append", default=[], type=parse_reference, metavar="NAME=CHANNELS:NRMSE")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.window_stride < 1 or args.tolerance < 0.0:
        parser.error("--window-stride >= 1 et --tolerance >= 0")
    try:
        table, participants, saved = rank(args.run_dir, args.data_dir, torch.device(args.device),
                                          args.window_stride, args.batch_size)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "configurations.csv", index=False)
    summary = {
        "run_dir": str(args.run_dir.resolve()),
        "validation_participants": participants,
        "window_stride": args.window_stride,
        "reproduces_checkpoint_score": args.window_stride == 1,
        "checkpoint_validation_physical_nmse": saved.get("validation_physical_nmse"),
        **pareto_summary(table, args.tolerance),
        "references": dict(args.reference),
    }
    (args.output_dir / "pareto.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(table, summary, dict(args.reference), args.output_dir / "pareto.png")
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
