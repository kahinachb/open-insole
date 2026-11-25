#!/usr/bin/env python3
"""Mahmud 2026 (JBHI, Ke2KeNet) retrained on our dataset under Full CV, offline as published (#47).

The method is kept as published (workspace ADR 0005), following the paper and the authors'
``configs/insole.json`` and ``kinetics/runner.py``:
- Ke2KeNet (``ke2kenet.py``): depth 5, 32 base filters, deep supervision;
- non-causal segments of 1 024 samples; training segments do not overlap (``train_stride`` 1024);
- one model per target, 12 here. The paper uses 8 per foot (16 in total);
- min-max normalisation of inputs and targets to 0-1;
- Adam at 3e-4, batch 16, MAE loss, auxiliary heads weighted 0.5;
- the learning rate divided by 10 after 10 epochs without validation improvement, early stopping
  after 20 epochs with the best weights restored, at most 500 epochs.

Our data forces a few changes:
- the targets are the 6D GRFM of each foot. The loss only counts valid samples (Measured or
  Airborne), and there is no CoP;
- the normalisation is fitted on the training Participants only (the repository's corrected
  ``train_minmax``). The paper normalised each segment;
- validation uses the fold's 2 validation Participants, where the paper used a random 10 % of
  the training samples;
- the sampling rate is ours, about 200 Hz: a segment lasts 5.1 s, against 4 s at 256 Hz;
- every sample of a test trial is predicted by the first segment covering it. The score uses
  the samples our causal MLP is scored on (``competitor_folds``).

The paper reports 5.22 M parameters, but the public code at the paper's depth and width has
about 57 M, and neither the code nor the paper's equations give 5.22 M. We port the code; see
``ke2kenet.py``.

Inputs (``--inputs``):
- ``gauges``: the 15 gauges of the predicted foot, as the paper's 16 Moticon sensors of that foot;
- ``raw``: our 78 channels.

Each finished target is saved under ``fold_<P>/targets/``, so a stopped fold resumes where it
stopped. ``predictions.csv`` is written once all 12 targets are done.

Usage::

    python competitor_mahmud.py DATA_DIR --inputs gauges --output-dir RESULTS/mahmud_ke2kenet_gauges/all --participant P01
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import ke2kenet
from competitor_folds import fold_of, load_folds, scored_indexes, segment_runs, subset, write_fold
from train_lstm_two_feet_masked import Acquisition, gauge_indexes

WINDOW = 1024
TRAIN_STRIDE = 1024
BATCH_SIZE = 16
LEARNING_RATE = 3e-4
EPOCHS = 500
PATIENCE = 20
LR_PATIENCE = 10
LR_FACTOR = 0.1
AUXILIARY_WEIGHT = 0.5


def input_columns(feature_names: list[str], mode: str, target: str) -> list[int]:
    """Input columns for ``target``: the gauges of its foot, or all 78 channels."""
    return gauge_indexes(feature_names, target[-1]) if mode == "gauges" else list(range(len(feature_names)))


def segment_starts(length: int, window: int, stride: int, cover: bool) -> list[int]:
    """Window starts in a run of ``length`` samples; ``cover`` adds a last window ending at the run's end."""
    if length < window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    if cover and starts[-1] + window < length:
        starts.append(length - window)
    return starts


def windows(items, columns, target, x_range, y_range, window, stride, cover):
    """Normalised windows (inputs, target, mask) and where each came from: (trial index, run start, start, length).

    A run shorter than ``window`` is padded with its last sample, and the padding is masked.
    """
    xs, ys, ms, origin = [], [], [], []
    for trial, item in enumerate(items):
        x = (item.x[:, columns] - x_range[0]) / x_range[1]
        y = np.nan_to_num((item.y[:, target] - y_range[0]) / y_range[1])
        m = item.mask[:, target].astype(np.float32)
        for run in segment_runs(item.segment_id):
            length = run.stop - run.start
            for start in segment_starts(length, window, stride, cover):
                rows = np.arange(run.start + start, run.start + start + window)
                valid = rows < run.stop
                rows = np.minimum(rows, run.stop - 1)
                xs.append(x[rows].T)
                ys.append(y[rows])
                ms.append(m[rows] * valid)
                origin.append((trial, run.start, start, min(window, length)))
    if not xs:
        return None
    return (np.stack(xs).astype(np.float32), np.stack(ys)[:, None].astype(np.float32),
            np.stack(ms)[:, None].astype(np.float32), origin)


def minmax(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum and range of each column (range 1 where constant)."""
    low, high = values.min(axis=0), values.max(axis=0)
    span = high - low
    return low, np.where(span > 0, span, 1.0)


def head_loss(outputs: list[torch.Tensor], y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Masked MAE of every head; coarse heads compare with the masked mean of the target over their bins."""
    total = 0.0
    for index, out in enumerate(outputs):
        factor = y.shape[-1] // out.shape[-1]
        weight = F.avg_pool1d(m, factor) if factor > 1 else m
        target = F.avg_pool1d(y * m, factor) / weight.clamp(min=1e-6) if factor > 1 else y
        error = ((out - target).abs() * (weight > 0)).sum() / (weight > 0).sum().clamp(min=1)
        total = total + error * (1.0 if index == len(outputs) - 1 else AUXILIARY_WEIGHT)
    return total


def evaluate(model, data, device) -> float:
    model.eval()
    losses, counts = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(data[0]), BATCH_SIZE):
            x, y, m = (torch.from_numpy(a[start:start + BATCH_SIZE]).to(device) for a in data[:3])
            losses += head_loss(model(x), y, m).item() * len(x)
            counts += len(x)
    return losses / max(counts, 1)


def train_target(train, validation, args, device, seed) -> tuple[ke2kenet.Ke2KeNet, list[dict]]:
    """One Ke2KeNet for one target, with the authors' optimiser, schedule and early stopping."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ke2kenet.build(train[0].shape[1], length=args.window, depth=args.depth, width=args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=LR_FACTOR, patience=LR_PATIENCE)
    best, best_state, history, waited = float("inf"), None, [], 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = rng.permutation(len(train[0]))
        total = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            rows = order[start:start + BATCH_SIZE]
            x, y, m = (torch.from_numpy(a[rows]).to(device) for a in train[:3])
            loss = head_loss(model(x), y, m)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(rows)
        validation_loss = evaluate(model, validation, device)
        scheduler.step(validation_loss)
        history.append({"epoch": epoch, "loss": total / len(order), "val_loss": validation_loss,
                        "lr": optimizer.param_groups[0]["lr"]})
        if validation_loss < best:
            best, waited = validation_loss, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    if best_state is None:
        raise FloatingPointError("the validation loss was never finite")
    model.load_state_dict(best_state)
    return model, history


def predict(model, data, items, y_range, device) -> list[np.ndarray]:
    """Physical-unit prediction of every sample of every trial (first covering window wins)."""
    model.eval()
    out = [np.full(len(item.x), np.nan) for item in items]
    with torch.no_grad():
        for start in range(0, len(data[0]), BATCH_SIZE):
            x = torch.from_numpy(data[0][start:start + BATCH_SIZE]).to(device)
            predicted = model(x)[-1][:, 0].cpu().numpy() * y_range[1] + y_range[0]
            for row, values in zip(data[3][start:start + BATCH_SIZE], predicted):
                trial, run_start, window_start, length = row
                rows = np.arange(run_start + window_start, run_start + window_start + length)
                todo = np.isnan(out[trial][rows])
                out[trial][rows[todo]] = values[:length][todo]
    return out


def run_fold(acquisitions, metadata, targets, groups, fold, args, device) -> Path | None:
    feature_names = list(metadata["feature_names"])
    train, validation = subset(acquisitions, fold, "train"), subset(acquisitions, fold, "validation")
    test = subset(acquisitions, fold, "test")
    fold_dir = args.output_dir / f"fold_{fold['test_group']}"
    (fold_dir / "targets").mkdir(parents=True, exist_ok=True)
    for index, target in enumerate(targets):
        if args.targets and target not in args.targets:
            continue
        target_file = fold_dir / "targets" / f"{target}.npz"
        if target_file.exists():
            continue
        start = time.time()
        columns = input_columns(feature_names, args.inputs, target)
        x_range = minmax(np.concatenate([item.x[:, columns] for item in train]))
        valid_y = np.concatenate([item.y[item.mask[:, index], index] for item in train])
        y_range = minmax(valid_y[:, None])
        y_range = (float(y_range[0][0]), float(y_range[1][0]))
        train_data = windows(train, columns, index, x_range, y_range, args.window, args.train_stride, False)
        validation_data = windows(validation, columns, index, x_range, y_range, args.window, args.window, True)
        model, history = train_target(train_data, validation_data, args, device, args.seed + index)
        test_data = windows(test, columns, index, x_range, y_range, args.window, args.window, True)
        predicted = predict(model, test_data, test, y_range, device)
        np.savez(target_file, **{item.name: values for item, values in zip(test, predicted)})
        (fold_dir / "targets" / f"{target}_history.json").write_text(json.dumps(history), encoding="utf-8")
        print(f"  {fold['test_group']} {target}: {len(history)} epochs, best val {min(h['val_loss'] for h in history):.4f}"
              f" ({time.time() - start:.0f} s)", flush=True)
    if not all((fold_dir / "targets" / f"{target}.npz").exists() for target in targets):
        return None
    per_target = [np.load(fold_dir / "targets" / f"{target}.npz") for target in targets]
    predictions = []
    for item in test:
        rows = scored_indexes(item)
        predictions.append((item, rows, np.stack([values[item.name][rows] for values in per_target], axis=1)))
    write_fold(
        fold_dir, args.data_dir, groups, fold, targets, predictions,
        {
            "model_type": "mahmud_ke2kenet",
            "competitor": "Mahmud 2026 (Ke2KeNet)",
            "causal": False,
            "inputs": args.inputs,
            "models": "one Ke2KeNet per target",
            "window_samples": args.window,
            "train_stride": args.train_stride,
            "depth": args.depth,
            "width": args.width,
            "parameters": ke2kenet.parameter_count(
                ke2kenet.build(len(input_columns(feature_names, args.inputs, targets[0])), args.window,
                               depth=args.depth, width=args.width)),
            "seed": args.seed,
            "task_set": "all",
        },
    )
    return fold_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True, help="run dir; one fold_<Participant> per fold")
    parser.add_argument("--inputs", choices=("gauges", "raw"), required=True)
    parser.add_argument("--participant", nargs="*", default=None, help="test Participants (default: every fold)")
    parser.add_argument("--fold-index", type=int, default=None, help="index in the sorted Participants ($SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--targets", nargs="*", default=None, help="only these targets (default: all 12)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    # The paper's values; smaller ones only make the tests fast.
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--train-stride", type=int, default=TRAIN_STRIDE)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args(argv)
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    acquisitions, metadata, targets, groups, folds = load_folds(args.data_dir)
    unknown = sorted(set(args.targets or ()) - set(targets))
    if unknown:
        parser.error(f"unknown targets: {', '.join(unknown)} (known: {', '.join(targets)})")
    participants = args.participant or groups
    if args.fold_index is not None:
        if not 0 <= args.fold_index < len(groups):
            parser.error(f"--fold-index outside 0..{len(groups) - 1}")
        participants = [groups[args.fold_index]]
    for participant in participants:
        if (args.output_dir / f"fold_{participant}" / "predictions.csv").exists():
            print(f"{participant}: already done, skipped")
            continue
        fold_dir = run_fold(acquisitions, metadata, targets, groups, fold_of(folds, participant), args, device)
        print(f"{participant}: {fold_dir or 'targets left, predictions.csv not written'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
