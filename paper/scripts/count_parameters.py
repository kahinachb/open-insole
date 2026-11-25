"""Number of trainable parameters of each method, for the 12 targets.

- MLP: the trained checkpoint (one network for the 12 targets).
- MLP, one per target: one network per target, 12 in all; the trained Fz checkpoint, which gives
  both feet, counts once per foot, as in benchmark_inference.py.
- Ke2KeNet: 12 networks of the trained shape (one per target), on the load cells of one foot (15
  channels) or on the 78 channels; the parameters do not depend on the weights.

Usage: python count_parameters.py MLP_CHECKPOINT OUTPUT_CSV --per-target CHECKPOINT ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_inference import MAHMUD_CHANNELS, MAHMUD_TARGETS, SEGMENT, ke2kenet, load_mlp  # noqa: E402


def count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--per-target", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    rows = [{"method": "mlp", "parameters": count(load_mlp(args.checkpoint, "cpu")[0])}]
    per_target = [load_mlp(path, "cpu") for path in args.per_target]
    rows.append({"method": "mlp_per_target",
                 "parameters": sum(count(model) * len(saved["targets"]) for model, saved in per_target)})
    for variant, channels in MAHMUD_CHANNELS.items():
        model = ke2kenet.build(channels).eval()
        with torch.no_grad():
            model(torch.zeros(1, channels, SEGMENT))  # materialise the lazy layers
        rows.append({"method": f"mahmud_{variant}", "parameters": count(model) * MAHMUD_TARGETS})
    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
