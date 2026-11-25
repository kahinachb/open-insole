"""Synthetic raw insole recordings in the firmware's binary format, for the characterisation tests.

Each packet is one insole: 4 Patch frames of 36 bytes (identifiers, frame
counter, 4 Gauges as big-endian uint16, IMU, barometer pressure and
temperature as 24-bit values, magnetometer), then a uint32 timestamp, behind a
one-byte sync signal. The acquisition repo's own decoder reads them back.
"""

from __future__ import annotations

import struct
from pathlib import Path

DECODER = Path(__file__).resolve().parents[2] / "acquisition" / "insole_data"
HAS_DECODER = (DECODER / "calculerTareFX29.py").exists()
PATCH_IDS = {"gauche": ((1, 2), (3, 4), (5, 6), (7, 8)), "droite": ((9, 10), (11, 12), (13, 14), (15, 16))}
NOMINAL_ZERO = 1000
N_PER_COUNT_100LBF = 100 * 4.44822 / 14000


def frame(identifier, gauges, raw_temperature: int = 0) -> bytes:
    body = bytes(identifier) + b"\x00" + struct.pack(">4H", *gauges) + bytes(12)
    return body + bytes(3) + bytes([raw_temperature >> 16]) + struct.pack(">H", raw_temperature & 0xFFFF) + bytes(7)


def write_recording(path: Path, counts, samples: int = 40, raw_temperature: int = 0, feet=tuple(PATCH_IDS)) -> None:
    """``counts(patch_id, gauge_index, sample)`` gives each raw reading; packets alternate between ``feet``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b""
    for sample in range(samples):
        for foot in feet:
            frames = b"".join(
                frame(identifier, [counts(identifier[0], index, sample) for index in range(4)], raw_temperature)
                for identifier in PATCH_IDS[foot]
            )
            data += b"\x00" + frames + struct.pack("<I", 5000)
    path.write_bytes(data)
