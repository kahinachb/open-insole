"""Inference speed of our MLP and the two competitors, one prediction per new sample (streaming).

Each method gets a new sample and must give the wrench at that instant. The time measured is
the computation of one prediction, batch 1, after a warm-up; the median over the repeats.
- MLP: one forward pass on the 21-sample causal window, all 12 targets (the trained checkpoint).
- MLP, one per target (``--per-target``): one network per target, 12 in all, run one after the
  other on the same window. The trained Fz checkpoint gives both feet; it is run once per foot, so
  that the time is that of 12 networks of the MLP's size.
- Sabbah 2024: the two forests (one per foot) of the trial's task family on one sample. The
  forests are fitted on random data of our shape (100 trees, depth 10, about 150 k samples):
  the prediction cost depends on the trees' size, not on the data. Its zero-phase filter needs
  the whole trial and cannot run in streaming, so the time is a lower bound.
- Mahmud 2026: the 12 networks (one per target) on the 1 024-sample segment that ends with the
  new sample, random weights of the trained shape. In streaming each new sample needs a new
  pass; the time is also a lower bound, since the method was trained to read future samples.
  Offline, one pass covers 1 024 samples: the amortised time per sample is also written.

Usage: python benchmark_inference.py CHECKPOINT OUTPUT_CSV [--per-target CHECKPOINT ...] [--repeats 200]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor

REPO = Path(__file__).resolve().parents[2] / "insole2grfm"
sys.path.insert(0, str(REPO))
import ke2kenet  # noqa: E402
from competitor_sabbah import FOREST  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from train_lstm_two_feet_masked import model_from_args  # noqa: E402

# Sabbah's inputs: loadsol-like (per foot 4 zones and the 6-axis heel IMU, both feet) or our 78 channels,
# each with the Body weight.
SABBAH_FEATURES = {"loadsol": 21, "raw": 79}
SABBAH_SAMPLES = 150_000
MAHMUD_TARGETS = 12
MAHMUD_CHANNELS = {"gauges": 15, "raw": 78}
SEGMENT = 1024


def median_seconds(call, repeats: int, warmup: int = 5) -> float:
    for _ in range(warmup):
        call()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def load_mlp(checkpoint: Path, device: str) -> tuple[torch.nn.Module, dict]:
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # The checkpoint holds its training options, the vertical-share wrapper included.
    model = model_from_args(SimpleNamespace(**saved), saved["input_size"], len(saved["targets"]),
                            feature_names=list(saved["feature_names"]), targets=list(saved["targets"]))
    model.load_state_dict(saved["state_dict"])
    return model.to(device).eval(), saved


def mlp(checkpoints: list[Path], device: str, repeats: int, per_target: bool = False) -> float:
    """One prediction of all the targets of ``checkpoints``, the models run one after the other.

    With ``per_target``, a model is run once per target it gives: one network per target.
    """
    loaded = [load_mlp(checkpoint, device) for checkpoint in checkpoints]
    models = [model for model, s in loaded for _ in (s["targets"] if per_target else [None])]
    saved = loaded[0][1]
    targets = [target for _, s in loaded for target in s["targets"]]
    if len(targets) != len(set(targets)):
        raise ValueError(f"checkpoints predict a target twice: {targets}")
    x = torch.randn(1, saved["window_samples"], saved["input_size"], device=device)

    def call():
        with torch.no_grad():
            for model in models:
                model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return median_seconds(call, repeats)


def sabbah(features: int, repeats: int) -> float:
    rng = np.random.default_rng(0)
    x, y = rng.normal(size=(SABBAH_SAMPLES, features)), rng.normal(size=(SABBAH_SAMPLES, 6))
    forests = [RandomForestRegressor(**FOREST, n_jobs=1, random_state=foot).fit(x, y) for foot in range(2)]
    sample = x[:1]
    return median_seconds(lambda: [forest.predict(sample) for forest in forests], repeats)


def mahmud(channels: int, device: str, repeats: int) -> float:
    model = ke2kenet.build(channels).to(device).eval()
    x = torch.randn(1, channels, SEGMENT, device=device)

    def call():
        with torch.no_grad():
            for _ in range(MAHMUD_TARGETS):  # 12 networks of one shape: the same cost each
                model(x)
        if device == "cuda":
            torch.cuda.synchronize()
    return median_seconds(call, repeats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path, help="a trained MLP fold checkpoint (.pt)")
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--per-target", nargs="+", type=Path, default=[],
                        help="the per-target MLP checkpoints that together give the 12 targets")
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = []

    def add(method, device, seconds, amortised=None):
        rows.append({"method": method, "device": device, "seconds_per_prediction": seconds,
                     "predictions_per_second": 1 / seconds,
                     "offline_seconds_per_sample": amortised if amortised is not None else seconds})
        print(f"{method:15} {device:5} {seconds * 1e3:10.3f} ms  {1 / seconds:12.1f} /s", flush=True)

    add("mlp", "cpu", mlp([args.checkpoint], "cpu", args.repeats))
    if args.per_target:
        add("mlp_per_target", "cpu", mlp(args.per_target, "cpu", args.repeats, per_target=True))
    for variant, features in SABBAH_FEATURES.items():
        add(f"sabbah_{variant}", "cpu", sabbah(features, args.repeats))
    for variant, channels in MAHMUD_CHANNELS.items():
        cpu = mahmud(channels, "cpu", max(3, args.repeats // 50))
        add(f"mahmud_{variant}", "cpu", cpu, cpu / SEGMENT)
    if torch.cuda.is_available():
        add("mlp", "cuda", mlp([args.checkpoint], "cuda", args.repeats))
        if args.per_target:
            add("mlp_per_target", "cuda", mlp(args.per_target, "cuda", args.repeats, per_target=True))
        for variant, channels in MAHMUD_CHANNELS.items():
            gpu = mahmud(channels, "cuda", max(10, args.repeats // 10))
            add(f"mahmud_{variant}", "cuda", gpu, gpu / SEGMENT)
    frame = pd.DataFrame(rows)
    frame["cpu"] = "1 thread, " + (open("/proc/cpuinfo").read().split("model name")[1].split(":")[1].split("\n")[0].strip()
                                   if Path("/proc/cpuinfo").exists() else "unknown")
    frame["gpu"] = torch.cuda.get_device_name() if torch.cuda.is_available() else ""
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
