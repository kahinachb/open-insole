"""Download the trained models of the paper from the GitHub releases and unpack them into models/.

    python tools/download_models.py                  # the main model: MLP, 78 channels
    python tools/download_models.py --all            # the four models
    python tools/download_models.py mlp-28ch mlp-per-target

Each model has its own release (tag ``mlp-78ch``, ``mlp-28ch``, ``mlp-34ch``, ``mlp-per-target``), holding one
archive: the Full CV of the paper, one network per fold, each trained without the participant it is named after
(``models/mlp_sensor_dropout_w21/all/fold_P05/models/mlp_test_P05.pt`` never saw P05). Any fold of the 78-channel
MLP can drive the live demo; P05 is the median fold.

Every archive is checked against its SHA-256 before it is unpacked. The download is anonymous when the
repository is public; otherwise it goes through the GitHub CLI (``gh``), logged in with an account that has
access to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = "kahinachb/open-insole"
ROOT = Path(__file__).resolve().parents[1]

MODELS = {  # release tag: (archive name, SHA-256 of the archive, what it is in the paper)
    "mlp-78ch": ("mlp_sensor_dropout_w21", "ae8f9b94e3fc44fe163208ffac66466d5f8a725808396d9502d1ceda903c3a3f",
                 "main model: MLP on the 78 channels, sensor dropout"),
    "mlp-28ch": ("mlp_minimal_28ch_w21", "68c2efe72cfcf6021ea66c83fa5b5619b024698e18cdc190929f8c68958f5a64",
                 "MLP on the selected 28-channel configuration"),
    "mlp-34ch": ("mlp_minimal_34ch_w21", "39e1bf97edfb8ae09409da2ed9ec7593b9b2080858ecd47a5ae73c7e4c9daf5d",
                 "MLP on the 34-channel configuration"),
    "mlp-per-target": ("mlp_per_target_dropout_w21",
                       "a643098ad582567f49ba8d1c2413ef5429133c9ed02caf9fd54bc953f88310ef",
                       "one MLP per component (11 networks, both vertical forces share one), sensor dropout; "
                       "one folder per component"),
}
DEFAULT = "mlp-78ch"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(tag: str, name: str, out: Path) -> None:
    """Anonymous download (public repository), else through ``gh`` (private repository)."""
    url = f"https://github.com/{REPOSITORY}/releases/download/{tag}/{name}.zip"
    try:
        urllib.request.urlretrieve(url, out / f"{name}.zip")
        return
    except urllib.error.HTTPError as error:
        if error.code != 404 or shutil.which("gh") is None:
            raise SystemExit(f"{url}: {error}; if the repository is private, install and log in the GitHub CLI (gh)")
    subprocess.run(["gh", "release", "download", tag, "--repo", REPOSITORY, "--pattern", f"{name}.zip",
                    "--dir", str(out), "--clobber"], check=True)


def fetch(tag: str, out: Path) -> None:
    name, expected, _ = MODELS[tag]
    archive = out / f"{name}.zip"
    if not (archive.exists() and sha256(archive) == expected):
        print(f"downloading {tag} ({name}.zip)", file=sys.stderr)
        download(tag, name, out)
    if sha256(archive) != expected:
        archive.unlink()
        raise SystemExit(f"{name}.zip: checksum mismatch, archive removed; try again")
    with zipfile.ZipFile(archive) as z:
        z.extractall(out)
    count = len(list((out / name).glob("**/fold_*/models/*.pt")))
    print(f"{tag}: {count} networks in {out / name}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     epilog="models: " + "; ".join(f"{k} ({v[2]})" for k, v in MODELS.items()))
    parser.add_argument("models", nargs="*", metavar="MODEL", help=f"release tag, default: {DEFAULT}")
    parser.add_argument("--all", action="store_true", help="the four models")
    parser.add_argument("--out", type=Path, default=ROOT / "models")
    args = parser.parse_args()
    unknown = [tag for tag in args.models if tag not in MODELS]
    if unknown:
        parser.error(f"unknown model {', '.join(unknown)}; choose from {', '.join(MODELS)}")
    args.out.mkdir(parents=True, exist_ok=True)
    for tag in list(MODELS) if args.all else args.models or [DEFAULT]:
        fetch(tag, args.out)


if __name__ == "__main__":
    main()
