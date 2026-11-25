#!/usr/bin/env python3
"""Record the raw serial stream of both insoles, without cameras.

Writes the bytes of the receiver as they arrive, the same ``dataSemelle.bin``
as ``record_trial.py``: ``insole_data/decodageToCSV.py`` decodes it,
``insole_data/calculerTareFX29.py`` takes a Tare from it, and
``python -m demo.live --replay`` plays it back.

    python acquisition/record_insoles.py --output manip/test/dataSemelle.bin
    python acquisition/record_insoles.py --output tare.bin --duration 5

Stop with Ctrl+C (or after ``--duration`` seconds). Every second the console
shows the packet rate of each insole (about 200 Hz when all is well).
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import serial

from live_plot_forces import LivePacketDecoder

FEET = {1: "left", 9: "right"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="/dev/ttyACM0", help="serial port of the receiver")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--output", type=Path,
                        help="file to write (default: insoles_<date>.bin in the current directory)")
    parser.add_argument("--duration", type=float, help="stop after this many seconds")
    parser.add_argument("--overwrite", action="store_true", help="replace --output if it exists")
    args = parser.parse_args()

    output = args.output or Path(f"insoles_{datetime.now():%Y-%m-%d_%H%M%S}.bin")
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists (use --overwrite to replace it)")
    output.parent.mkdir(parents=True, exist_ok=True)

    decoder = LivePacketDecoder()
    totals = {foot: 0 for foot in FEET.values()}
    window = dict(totals)
    written = 0
    print(f"Recording {args.port} to {output}. Ctrl+C to stop.")
    with serial.Serial(args.port, args.baud, timeout=0.05) as port, open(output, "wb") as file:
        port.reset_input_buffer()
        start = last_report = time.monotonic()
        try:
            while args.duration is None or time.monotonic() - start < args.duration:
                chunk = port.read(max(1, min(port.in_waiting, 65_536)))
                if chunk:
                    file.write(chunk)
                    written += len(chunk)
                    for _signal, packet in decoder.feed(chunk):
                        foot = FEET[packet[0]]
                        totals[foot] += 1
                        window[foot] += 1
                now = time.monotonic()
                if now - last_report >= 1.0:
                    rates = "  ".join(f"{foot} {count / (now - last_report):5.0f} Hz" for foot, count in window.items())
                    print(f"\r{now - start:7.1f} s  {rates}  {written / 1e6:6.1f} MB", end="", flush=True)
                    window = {foot: 0 for foot in FEET.values()}
                    last_report = now
        except KeyboardInterrupt:
            pass
        elapsed = time.monotonic() - start

    print(f"\n{output}: {elapsed:.1f} s, {written / 1e6:.1f} MB")
    for foot, count in totals.items():
        print(f"  {foot:5s} {count} packets, {count / max(elapsed, 1e-9):.0f} Hz on average")
    if decoder.rejected_bytes:
        print(f"  {decoder.rejected_bytes} bytes outside any valid packet")
    if not all(totals.values()):
        print("Warning: one insole sent nothing. Is it switched on?", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
