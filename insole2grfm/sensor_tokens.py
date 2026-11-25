"""Insole channels as Sensor tokens in the Canonical insole frame.

Each foot is 15 gauge tokens + 4 IMU tokens (heel, midfoot, forefoot, toes);
both feet give 38 tokens per instant, left foot first:

    0..14  L gauges (feature order)   15..18 L IMUs heel->toes
    19..33 R gauges (feature order)   34..37 R IMUs heel->toes

A token has a dynamic value vector of ``len(VALUE_CHANNELS)`` = 7 slots and a
static descriptor (``SensorTokenizer.descriptors``):

- gauge token: slot 0 = load (N), slots 1..6 = 0;
- IMU token: slot 0 = 0, slots 1..6 = acc (g) and gyro (dps) rotated into the
  canonical frame.

Canonical insole frame = left-foot frame: x anterior, y lateral, z up. The
right foot is mirrored (y -> -y): positions and tilt directions negate y,
accelerations (polar) negate y, angular rates (axial) negate x and z. IMU chip
axes are first rotated into the physical insole frame with the per-board
``chip_to_insole`` matrix of ``tools/insole_layout.json``.

Geometry comes from ``tools/insole_layout.json`` (see
``tools/build_insole_layout.py``). Inputs may be numpy arrays or torch
tensors; outputs follow the input type.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from grfm_schema import FEET, TARGETS  # noqa: F401  (TARGETS re-exported: part of this module's API)

PATCH_NAMES = ("heel", "midfoot", "forefoot", "toes")
GAUGES_PER_FOOT = 15
IMUS_PER_FOOT = 4
TOKENS_PER_FOOT = GAUGES_PER_FOOT + IMUS_PER_FOOT
TOKEN_COUNT = 2 * TOKENS_PER_FOOT
VALUE_CHANNELS = ("load_N", "acc_x_g", "acc_y_g", "acc_z_g", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps")
IMU_SUFFIXES = (
    "acc_x_cal_g", "acc_y_cal_g", "acc_z_cal_g",
    "gyro_x_cal_dps", "gyro_y_cal_dps", "gyro_z_cal_dps",
)
INACTIVE_P99_N = 5.0
FOURIER_WAVELENGTHS_MM = (20.0, 40.0, 80.0, 160.0, 320.0, 640.0)
# Mirroring the right foot (y -> -y) flips Fy (polar) and Mx, Mz (axial).
TARGET_MIRROR_SIGN = np.array([1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1], dtype=np.float32)
DEFAULT_LAYOUT = Path(__file__).resolve().parent / "tools" / "insole_layout.json"


@dataclass(frozen=True)
class SensorTokenizer:
    """Static token geometry plus the raw-samples -> (tokens, mask) map."""

    feature_names: tuple
    token_names: tuple          # 38 names: gauge channel or "{S}_p{P}_imu"
    gauge_columns: np.ndarray   # (2, 15) feature indexes
    imu_columns: np.ndarray     # (2, 4, 6) feature indexes
    acc_rotation: np.ndarray    # (2, 4, 3, 3) chip acc -> canonical frame
    gyro_rotation: np.ndarray   # (2, 4, 3, 3) chip gyro -> canonical frame
    positions_mm: np.ndarray    # (38, 2) canonical x, y (mm)
    tilt_vectors: np.ndarray    # (38, 2) cleat tilt direction x sin(tilt), canonical; 0 for IMUs
    token_types: np.ndarray     # (38,) 0 gauge, 1 IMU
    token_feet: np.ndarray      # (38,) 0 left, 1 right
    descriptor_names: tuple     # names of the descriptor columns
    descriptors: np.ndarray     # (38, D) float32: Fourier(x, y), tilt, type, foot

    @classmethod
    def from_layout(cls, layout_path: Path = DEFAULT_LAYOUT) -> "SensorTokenizer":
        layout = json.loads(Path(layout_path).read_text(encoding="utf-8"))
        names = list(layout["feature_names"])
        gauge_columns, imu_columns, token_names, acc_rotation, gyro_rotation = [], [], [], [], []
        positions, tilts = [], []
        for side in FEET:
            foot = layout["feet"][side]
            sign = 1.0 if side == "L" else -1.0
            outline_y = np.asarray(foot["outline"], float)[:, 1]
            centre = 0.5 * (outline_y.min() + outline_y.max())
            chip = np.array([imu["chip_to_insole"] for imu in foot["imus"]], float)  # (4, 3, 3)
            # mirror y: polar vectors negate y, axial vectors negate x and z
            acc_rotation.append(np.diag([1.0, sign, 1.0]) @ chip)
            gyro_rotation.append(np.diag([sign, 1.0, sign]) @ chip)
            gauges = sorted((s for s in foot["sensors"]), key=lambda s: names.index(s["channel"]))
            gauge_columns.append([names.index(s["channel"]) for s in gauges])
            imu_columns.append([
                [names.index(f"{imu['channel_prefix']}_{suffix}") for suffix in IMU_SUFFIXES]
                for imu in foot["imus"]
            ])
            token_names += [s["channel"] for s in gauges] + [f"{imu['channel_prefix']}_imu" for imu in foot["imus"]]
            for item in gauges + foot["imus"]:
                positions.append((item["x_mm"], sign * (item["y_mm"] - centre)))
            for item in gauges:
                scale = np.sin(np.radians(item["tilt_deg"]))
                tilts.append((scale * item["tilt_direction"][0], sign * scale * item["tilt_direction"][1]))
            tilts += [(0.0, 0.0)] * len(foot["imus"])
        positions = np.asarray(positions, np.float32)
        tilts = np.asarray(tilts, np.float32)
        types = np.tile(np.r_[np.zeros(GAUGES_PER_FOOT), np.ones(IMUS_PER_FOOT)], 2).astype(np.int64)
        feet = np.repeat([0, 1], TOKENS_PER_FOOT).astype(np.int64)
        descriptor_names, descriptors = _descriptors(positions, tilts, types, feet)
        return cls(
            feature_names=tuple(names),
            token_names=tuple(token_names),
            gauge_columns=np.asarray(gauge_columns),
            imu_columns=np.asarray(imu_columns),
            acc_rotation=np.asarray(acc_rotation, np.float32),
            gyro_rotation=np.asarray(gyro_rotation, np.float32),
            positions_mm=positions,
            tilt_vectors=tilts,
            token_types=types,
            token_feet=feet,
            descriptor_names=descriptor_names,
            descriptors=descriptors,
        )

    def mirror_symmetric(self) -> "SensorTokenizer":
        """Same tokenizer with the mirror counterparts sharing one geometry.

        Token i of L and token i + 19 of R get the mean of their canonical
        positions and tilt vectors (the two boards differ by a few mm), so a
        model reading these descriptors treats both feet identically: swapping
        the feet (:meth:`swap_feet`) then exactly swaps its per-foot outputs.
        IMU rotations and columns are unchanged.
        """
        def symmetric(values):
            mean = 0.5 * (values[:TOKENS_PER_FOOT] + values[TOKENS_PER_FOOT:])
            return np.concatenate([mean, mean]).astype(np.float32)

        positions, tilts = symmetric(self.positions_mm), symmetric(self.tilt_vectors)
        _, descriptors = _descriptors(positions, tilts, self.token_types, self.token_feet)
        return replace(self, positions_mm=positions, tilt_vectors=tilts, descriptors=descriptors)

    def swap_feet(self, x) -> np.ndarray:
        """Raw samples ``(..., 78)`` with the feet exchanged in the canonical frame.

        The result's left-foot tokens equal the input's right-foot tokens and
        vice versa (gauge i of one foot <-> gauge i of the other, IMU chip axes
        re-expressed through the other board's rotation). Numpy only.
        """
        x = np.asarray(x)
        out = x.copy()
        for foot, other in ((0, 1), (1, 0)):
            out[..., self.gauge_columns[foot]] = x[..., self.gauge_columns[other]]
            for patch in range(IMUS_PER_FOOT):
                for start, rotation in ((0, self.acc_rotation), (3, self.gyro_rotation)):
                    source = self.imu_columns[other, patch, start : start + 3]
                    canonical = x[..., source] @ rotation[other, patch].T.astype(np.float64)
                    chip = canonical @ np.linalg.inv(rotation[foot, patch].astype(np.float64)).T
                    out[..., self.imu_columns[foot, patch, start : start + 3]] = chip
        return out

    @property
    def gauge_tokens(self) -> np.ndarray:
        """Token indexes of the 30 gauges (L then R, feature order)."""
        return np.concatenate([np.arange(GAUGES_PER_FOOT) + f * TOKENS_PER_FOOT for f in range(2)])

    @property
    def imu_tokens(self) -> np.ndarray:
        """Token indexes of the 8 IMUs (L heel..toes, then R)."""
        return np.concatenate([np.arange(GAUGES_PER_FOOT, TOKENS_PER_FOOT) + f * TOKENS_PER_FOOT for f in range(2)])

    def inactive(self, trial, threshold_n: float = INACTIVE_P99_N) -> np.ndarray:
        """Inactive-sensor flag of one trial ``(T, 78)`` -> ``(38,)`` bool.

        A gauge is inactive when its 99th percentile over the whole trial is
        below ``threshold_n`` (5 N). IMU tokens are never flagged.
        """
        trial = np.asarray(trial)
        flags = np.zeros(TOKEN_COUNT, dtype=bool)
        p99 = np.percentile(trial[:, self.gauge_columns.ravel()], 99, axis=0)
        flags[self.gauge_tokens] = p99 < threshold_n
        return flags

    def __call__(self, x, inactive=None):
        """Raw samples ``(..., 78)`` -> tokens ``(..., 38, 7)`` and mask ``(..., 38)``.

        ``mask`` is True for a token the model may attend to (torch's
        ``key_padding_mask`` wants the opposite: pass ``~mask``). ``inactive``
        is a per-trial flag from :meth:`inactive`, shape ``(38,)`` or
        ``(B, 38)`` for windows ``(B, W, 78)`` (its leading axes align with the
        leading axes of ``x``); inactive tokens keep their reading and are
        masked out.
        """
        backend = _Backend(x)
        gauges = x[..., self.gauge_columns]                       # (..., 2, 15)
        imu = x[..., self.imu_columns]                            # (..., 2, 4, 6)
        acc = backend.einsum("fpij,...fpj->...fpi", backend.tensor(self.acc_rotation), imu[..., :3])
        gyro = backend.einsum("fpij,...fpj->...fpi", backend.tensor(self.gyro_rotation), imu[..., 3:])
        gauge_tokens = backend.cat([gauges[..., None], backend.zeros(gauges.shape + (6,))])
        imu_tokens = backend.cat([backend.zeros(imu.shape[:-1] + (1,)), acc, gyro])
        tokens = backend.cat([gauge_tokens, imu_tokens], axis=-2)  # (..., 2, 19, 7)
        tokens = tokens.reshape(tuple(tokens.shape[:-3]) + (TOKEN_COUNT, len(VALUE_CHANNELS)))
        mask = backend.ones_bool(tuple(tokens.shape[:-1]))
        if inactive is not None:
            inactive = backend.tensor(np.asarray(inactive) if not backend.is_native(inactive) else inactive, bool)
            missing = (mask.ndim - 1) - (inactive.ndim - 1)
            inactive = inactive.reshape(tuple(inactive.shape[:-1]) + (1,) * missing + (TOKEN_COUNT,))
            mask = mask & ~inactive
        return tokens, mask


def _descriptors(positions, tilts, types, feet) -> tuple[tuple, np.ndarray]:
    """Static descriptor columns: Fourier(x, y), tilt, token type, foot."""
    columns, names = [], []
    for axis, coordinate in zip("xy", positions.T):
        for wavelength in FOURIER_WAVELENGTHS_MM:
            phase = 2 * np.pi * coordinate / wavelength
            columns += [np.sin(phase), np.cos(phase)]
            names += [f"sin_{axis}_{wavelength:g}mm", f"cos_{axis}_{wavelength:g}mm"]
    columns += [tilts[:, 0], tilts[:, 1], types == 0, types == 1, feet == 0, feet == 1]
    names += ["tilt_x", "tilt_y", "is_gauge", "is_imu", "is_left", "is_right"]
    return tuple(names), np.column_stack(columns).astype(np.float32)


def targets_to_canonical(y):
    """12 targets ``(..., 12)`` in the subject frame -> canonical frame (right Fy, Mx, Mz negated)."""
    return y * _Backend(y).tensor(TARGET_MIRROR_SIGN)


def targets_from_canonical(y):
    """Inverse of :func:`targets_to_canonical` (the mirror is its own inverse)."""
    return targets_to_canonical(y)


class _Backend:
    """The few array operations the tokenizer needs, for numpy or torch inputs."""

    def __init__(self, like):
        self.like = like
        self.torch = type(like).__module__.split(".")[0] == "torch"
        if self.torch:
            import torch

            self.lib = torch

    def is_native(self, value) -> bool:
        return isinstance(value, self.lib.Tensor) if self.torch else isinstance(value, np.ndarray)

    def tensor(self, array, dtype=None):
        if self.torch:
            dtype = self.lib.bool if dtype is bool else self.like.dtype
            return self.lib.as_tensor(array, dtype=dtype, device=self.like.device)
        return np.asarray(array).astype(dtype or self.like.dtype, copy=False)

    def zeros(self, shape):
        if self.torch:
            return self.lib.zeros(tuple(shape), dtype=self.like.dtype, device=self.like.device)
        return np.zeros(shape, dtype=self.like.dtype)

    def ones_bool(self, shape):
        if self.torch:
            return self.lib.ones(tuple(shape), dtype=self.lib.bool, device=self.like.device)
        return np.ones(shape, dtype=bool)

    def einsum(self, equation, *operands):
        return (self.lib.einsum if self.torch else np.einsum)(equation, *operands)

    def cat(self, items, axis=-1):
        return self.lib.cat(items, dim=axis) if self.torch else np.concatenate(items, axis=axis)
