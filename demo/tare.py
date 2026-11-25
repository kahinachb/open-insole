#!/usr/bin/env python3
"""Tare both insoles before the demo: insoles in their sandals, not worn, for a few seconds.

    python -m demo.tare --port /dev/ttyACM0 --output demo/tare.json

Writes one ``fx29_tare_v1`` file holding both feet (median counts of every
load cell), read by ``python -m demo.live --tare-file``.
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from demo.stream import SerialSource, codec, raw_counts

PATCH_IDS = ("1", "3", "5", "7", "9", "11", "13", "15")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--output", type=Path, default=Path("demo/tare.json"))
    args = parser.parse_args()

    print(f"Tare: insoles in their sandals, not worn, still for {args.seconds:.0f} s...")
    counts = raw_counts(SerialSource(args.port, args.baud), args.seconds + 0.5)
    missing = [patch for patch in PATCH_IDS if patch not in counts]
    if missing:
        raise SystemExit(f"no packet from Patches {', '.join(missing)}: is each insole on?")

    tares, spread = {}, {}
    for patch in PATCH_IDS:
        tares[patch], spread[patch] = [], []
        for number, values in enumerate(counts[patch], start=1):
            if (int(patch), number) in codec.CAPTEURS_ABSENTS:
                tares[patch].append(None)
                spread[patch].append(None)
                continue
            median = float(statistics.median(values))
            tares[patch].append(median)
            spread[patch].append(float(statistics.median(abs(value - median) for value in values)))

    document = {
        "format": "fx29_tare_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "foot": "both",
        "seconds": args.seconds,
        "tares": tares,
        "mad_counts": spread,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    samples = len(counts["1"][0])
    print(f"{args.output}: {samples} packets per foot, largest MAD "
          f"{max(v for p in spread.values() for v in p if v is not None):.1f} counts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
