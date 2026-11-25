"""A tiny C3D with two force platforms, for the static extraction CLI (#26).

Needs ezc3d, so it runs in the ``c3d-io`` env (``C3D_IO_PYTHON``), as a script::

    python synthetic_c3d.py OUT.c3d '{"rate_hz": 1000, "fz": [[...plate 1...], [...plate 2...]]}'

The two platforms are type 2 (six channels each, no calibration matrix), so
ezc3d's force-platform filter returns the analog Fz channel as the platform's
force z unchanged. The number of samples must be a multiple of 10 (10 analog
samples per point frame at 1/10 of the analog rate).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

C3D_IO_PYTHON = Path(os.environ.get("C3D_IO_PYTHON", "envs/c3d-io/bin/python"))
ANALOG_PER_FRAME = 10


def write_c3d(path: Path, rate_hz: float, fz: list[list[float]]) -> Path:
    """Write ``path`` through the ``c3d-io`` interpreter (call from any env)."""
    spec = json.dumps({"rate_hz": rate_hz, "fz": fz})
    subprocess.run([str(C3D_IO_PYTHON), str(Path(__file__).resolve()), str(path), spec], check=True)
    return path


def _write(path: str, rate_hz: float, fz: list[list[float]]) -> None:
    import numpy as np
    import ezc3d

    samples = len(fz[0])
    if samples % ANALOG_PER_FRAME:
        raise ValueError(f"{samples} samples: not a multiple of {ANALOG_PER_FRAME}")
    c3d = ezc3d.c3d()
    c3d["parameters"]["POINT"]["RATE"]["value"] = [rate_hz / ANALOG_PER_FRAME]
    c3d["parameters"]["POINT"]["LABELS"]["value"] = ("marker",)
    c3d["parameters"]["ANALOG"]["RATE"]["value"] = [rate_hz]
    c3d["parameters"]["ANALOG"]["LABELS"]["value"] = tuple(
        f"{component}{plate}" for plate in (1, 2) for component in ("Fx", "Fy", "Fz", "Mx", "My", "Mz"))
    corners = np.zeros((3, 4, 2))
    for index, x0 in enumerate((0.0, 600.0)):
        corners[:, :, index] = [[x0 + 500, x0, x0, x0 + 500], [600, 600, 0, 0], [0, 0, 0, 0]]
    parameters = {"USED": [2], "TYPE": [2, 2], "ZERO": [1, 0],
                  "CHANNEL": np.arange(1, 13, dtype=float).reshape(2, 6).T,
                  "CORNERS": corners, "ORIGIN": np.zeros((3, 2)), "CAL_MATRIX": np.zeros((6, 6, 2))}
    for name, value in parameters.items():
        c3d.add_parameter("FORCE_PLATFORM", name, value)
    for name in ("USED", "TYPE", "ZERO", "CHANNEL"):  # integer parameters in the C3D standard
        c3d["parameters"]["FORCE_PLATFORM"][name]["type"] = ezc3d.ezc3d.INT
    c3d["data"]["points"] = np.ones((4, 1, samples // ANALOG_PER_FRAME))
    analogs = np.zeros((1, 12, samples))
    analogs[0, 2], analogs[0, 8] = fz
    c3d["data"]["analogs"] = analogs
    c3d.write(path)


if __name__ == "__main__":
    out, spec = sys.argv[1], json.loads(sys.argv[2])
    _write(out, spec["rate_hz"], spec["fz"])
