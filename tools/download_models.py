"""Download the trained models of the paper from the GitHub release and unpack them into models/.

    python tools/download_models.py                  # the main model (MLP with sensor dropout)
    python tools/download_models.py --all            # the four neural runs of the paper
    python tools/download_models.py mlp_minimal_28ch_w21

Each run is the Full CV of the paper: one model per fold, trained without the participant it is named
after (``models/<run>/all/fold_P05/models/mlp_test_P05.pt`` never saw P05). Any fold of
``mlp_sensor_dropout_w21`` can drive the live demo; P05 is the median fold. Every archive is checked
against its SHA-256 before it is unpacked. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "https://github.com/kahinachb/open-insole/releases/download/v1.0"
ROOT = Path(__file__).resolve().parents[1]

RUNS = {  # run: (SHA-256 of its archive, what it is in the paper)
    "mlp_sensor_dropout_w21": ("ae8f9b94e3fc44fe163208ffac66466d5f8a725808396d9502d1ceda903c3a3f",
                               "main model: MLP, 78 channels, sensor dropout"),
    "mlp_minimal_28ch_w21": ("68c2efe72cfcf6021ea66c83fa5b5619b024698e18cdc190929f8c68958f5a64",
                             "MLP on the selected 28-channel configuration"),
    "mlp_minimal_34ch_w21": ("39e1bf97edfb8ae09409da2ed9ec7593b9b2080858ecd47a5ae73c7e4c9daf5d",
                             "MLP on the 34-channel configuration"),
    "patch_transformer_v8": ("caf8256908e503c474ddda2342eabfa394ba46257b76894daff4c8f7fb0f9913",
                             "causal patch Transformer, 500 ms window"),
}
DEFAULT = "mlp_sensor_dropout_w21"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(run: str, out: Path) -> None:
    expected, _ = RUNS[run]
    archive = out / f"{run}.zip"
    if not (archive.exists() and sha256(archive) == expected):
        print(f"downloading {run}.zip", file=sys.stderr)
        urllib.request.urlretrieve(f"{RELEASE}/{run}.zip", archive)
    if sha256(archive) != expected:
        archive.unlink()
        raise SystemExit(f"{run}.zip: checksum mismatch, archive removed; try again")
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)
    print(f"{run}: {len(list((out / run).glob('all/fold_*/models/*.pt')))} models in {out / run}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog="runs: " + "; ".join(f"{k} ({v[1]})" for k, v in RUNS.items()))
    parser.add_argument("runs", nargs="*", metavar="RUN", help=f"default: {DEFAULT}")
    parser.add_argument("--all", action="store_true", help="the four runs")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    unknown = [run for run in args.runs if run not in RUNS]
    if unknown:
        parser.error(f"unknown run {', '.join(unknown)}; choose from {', '.join(RUNS)}")
    args.out.mkdir(parents=True, exist_ok=True)
    for run in list(RUNS) if args.all else args.runs or [DEFAULT]:
        fetch(run, args.out)


if __name__ == "__main__":
    main()
