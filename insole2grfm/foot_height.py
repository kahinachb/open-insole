"""Foot height: height of a foot's lowest marker above the ground, per sample (mm).

Used by the prepared-dataset builder (v3.7). Three facts of the lab shape it:

- the ground level depends on position: the two force plates sit proud of the
  floor (plate tops at z ~ 8-11 mm, the floor ~ 55-60 mm lower and not quite flat),
  so each marker is measured against the plate top plane when it lies over a
  plate footprint and against the floor elsewhere;
- a marker sits above the sole (marker radius plus shoe, ~ 5-20 mm depending on
  the marker), so each marker's height above the ground is corrected by its
  offset, the median height of that marker above its plate while the foot is in
  plate contact (calibrated per Subject over all its trials). A flat foot then
  reads ~ 0 whichever markers are visible, and heel-off with the toes on the
  ground still reads ~ 0;
- the floor level is estimated per Subject from the feet seen off the plates: per
  250 mm xy cell, the most common offset-corrected lowest-marker height (stance;
  frames with the foot in the air spread above it), then a plane through the cells (the real floor
  tilts by ~ 13 mm over the 3 m walkway).

Known limit: at push-off the shoe's toe spring flattens and the toe marker drops
up to ~ 15 mm below its flat-stance height, so Foot height reads slightly negative
there (walk, P01b: on-plate q05 ~ -12 to -15 mm).

Foot height is ``min_i (z_i - offset_i - ground(x_i, y_i))`` over the visible,
calibrated markers of the foot, and is invalid (NaN) when fewer than
``MIN_VISIBLE_MARKERS`` are visible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation_utils import marker_arrays

MIN_VISIBLE_MARKERS = 2
DEFAULT_FALLBACK_FLOOR_Z_MM = -55.0
MIN_OFFSET_SAMPLES = 50
FLOOR_CELL_MM = 250.0
MIN_FLOOR_CELL_SAMPLES = 100
FLOOR_LEVEL_BIN_MM = 1.0
FLOOR_LEVEL_WINDOW_MM = 3.0
FLOOR_SLOPE_MIN_SPAN_MM = 1000.0
FLOOR_OUTLIER_MM = 10.0


@dataclass(frozen=True)
class PlateSurface:
    platform: int
    corners_xy: np.ndarray  # (4, 2), mm
    plane: np.ndarray  # z = a + b x + c y, mm

    def contains(self, xy: np.ndarray) -> np.ndarray:
        """Whether each point (..., 2) lies over the plate footprint (convex polygon)."""
        corners = self.corners_xy
        crosses = []
        for start, end in zip(corners, np.roll(corners, -1, axis=0)):
            edge, relative = end - start, xy - start
            crosses.append(edge[0] * relative[..., 1] - edge[1] * relative[..., 0])
        crosses = np.stack(crosses, axis=-1)
        same_side = np.all(crosses >= 0, axis=-1) | np.all(crosses <= 0, axis=-1)
        return same_side & np.isfinite(xy).all(axis=-1)


def plate_surfaces(platform_geometry: list[dict]) -> list[PlateSurface]:
    """Plate top planes from the synchronisation report's ``platform_geometry``."""
    surfaces = []
    for item in platform_geometry:
        corners = np.asarray(item["corners_global_mm"], dtype=np.float64)
        design = np.column_stack([np.ones(len(corners)), corners[:, :2]])
        plane = np.linalg.lstsq(design, corners[:, 2], rcond=None)[0]
        surfaces.append(PlateSurface(int(item["platform"]), corners[:, :2], plane))
    return surfaces


def _plane_z(plane: np.ndarray, xy: np.ndarray) -> np.ndarray:
    return plane[0] + plane[1] * xy[..., 0] + plane[2] * xy[..., 1]


@dataclass(frozen=True)
class Ground:
    plates: list[PlateSurface]
    floor: np.ndarray  # z = a + b x + c y, mm

    def z(self, xy: np.ndarray) -> np.ndarray:
        """Ground level under each point (..., 2): plate top over a plate, else floor."""
        ground = _plane_z(self.floor, xy)
        for plate in self.plates:
            over = plate.contains(xy)
            ground = np.where(over, _plane_z(plate.plane, xy), ground)
        return ground

    def over_plate(self, xy: np.ndarray) -> np.ndarray:
        over = np.zeros(xy.shape[:-1], dtype=bool)
        for plate in self.plates:
            over |= plate.contains(xy)
        return over


@dataclass
class FootHeightCalibration:
    """Per-Subject marker offsets (per foot) and floor level."""

    subject: str
    offsets_mm: dict[str, dict[str, float]] = field(default_factory=dict)
    offset_samples: dict[str, dict[str, int]] = field(default_factory=dict)
    floor: np.ndarray = field(
        default_factory=lambda: np.array([DEFAULT_FALLBACK_FLOOR_Z_MM, 0.0, 0.0])
    )
    floor_source: str = "fallback_constant"
    floor_samples: int = 0
    floor_fit_rmse_mm: float | None = None
    trials: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        return {
            "trials_used": self.trials,
            "marker_offsets_mm": self.offsets_mm,
            "marker_offset_samples": self.offset_samples,
            "floor_plane_mm": {
                "z0": float(self.floor[0]),
                "slope_x": float(self.floor[1]),
                "slope_y": float(self.floor[2]),
            },
            "floor_source": self.floor_source,
            "floor_samples": self.floor_samples,
            "floor_fit_rmse_mm": self.floor_fit_rmse_mm,
        }


def _contact_frames(
    times: np.ndarray, contacts: list[dict], side: str
) -> dict[int, np.ndarray]:
    """Mocap frames inside that foot's plate contacts, per platform."""
    frames: dict[int, np.ndarray] = {}
    for contact in contacts:
        if contact.get("matched_side") != side:
            continue
        inside = (times >= float(contact["contact_start_s"])) & (
            times <= float(contact["contact_end_s"])
        )
        platform = int(contact["platform"])
        frames[platform] = frames.get(platform, np.zeros(len(times), dtype=bool)) | inside
    return frames


def _contact_level(values: np.ndarray) -> float:
    """Most common value (smoothed 1 mm histogram), refined by the median around it."""
    edges = np.arange(np.floor(values.min()), np.ceil(values.max()) + 2 * FLOOR_LEVEL_BIN_MM,
                      FLOOR_LEVEL_BIN_MM)
    counts, edges = np.histogram(values, edges)
    smooth = np.convolve(counts, np.ones(3), mode="same")
    peak = edges[np.argmax(smooth)] + FLOOR_LEVEL_BIN_MM / 2
    return float(np.median(values[np.abs(values - peak) <= FLOOR_LEVEL_WINDOW_MM]))


def _fit_floor(xy: np.ndarray, z: np.ndarray, fallback_z_mm: float):
    """Floor plane through the per-xy-cell stance level of off-plate foot heights."""
    if len(z) < MIN_FLOOR_CELL_SAMPLES:
        return np.array([fallback_z_mm, 0.0, 0.0]), "fallback_constant", None
    cells = np.floor(xy / FLOOR_CELL_MM).astype(np.int64)
    keys, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    centres, levels, weights = [], [], []
    for index, count in enumerate(counts):
        if count < MIN_FLOOR_CELL_SAMPLES:
            continue
        values = z[inverse == index]
        centres.append((keys[index] + 0.5) * FLOOR_CELL_MM)
        levels.append(_contact_level(values))
        weights.append(count)
    if not levels:
        return np.array([fallback_z_mm, 0.0, 0.0]), "fallback_constant", None
    centres = np.asarray(centres, dtype=np.float64)
    levels = np.asarray(levels, dtype=np.float64)
    weights = np.sqrt(np.asarray(weights, dtype=np.float64))
    # A slope only along an axis the off-plate data actually spans.
    slope_axes = [
        axis for axis in (0, 1)
        if np.ptp(centres[:, axis]) >= FLOOR_SLOPE_MIN_SPAN_MM
    ]
    keep = np.ones(len(levels), dtype=bool)
    for _ in range(3):
        design = np.column_stack([np.ones(len(levels))] + [centres[:, a] for a in slope_axes])
        coefficients = np.linalg.lstsq(
            design[keep] * weights[keep, None], levels[keep] * weights[keep], rcond=None
        )[0]
        residuals = levels - design @ coefficients
        new_keep = np.abs(residuals) <= FLOOR_OUTLIER_MM
        if new_keep.sum() < max(1, len(slope_axes) + 1) or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    plane = np.zeros(3)
    plane[0] = coefficients[0]
    for position, axis in enumerate(slope_axes, start=1):
        plane[1 + axis] = coefficients[position]
    rmse = float(np.sqrt(np.mean(residuals[keep] ** 2)))
    return plane, f"off_plate_stance_mode_{len(levels)}_cells", rmse


def calibrate_subject(
    subject: str,
    trial_dirs: list[Path],
    fallback_floor_z_mm: float = DEFAULT_FALLBACK_FLOOR_Z_MM,
) -> FootHeightCalibration:
    """Marker offsets and floor level of one Subject, from all its synchronised trials.

    Offsets: per marker, the median of (z - plate top) over the mocap frames inside
    that foot's plate contacts (report ``contact_assignments``) where the marker is
    over that plate. Floor: per frame where a foot's visible markers are all off the
    plates, the lowest offset-corrected marker z; the floor is the stance
    level (histogram mode) of those values per xy cell, fitted with a plane.
    """
    calibration = FootHeightCalibration(subject)
    residuals: dict[str, dict[str, list[np.ndarray]]] = {"L": {}, "R": {}}
    loaded = []
    for trial in trial_dirs:
        sync = trial / "synchronized_v2"
        mocap_path = sync / "mocap_synchronisee.csv"
        if not mocap_path.is_file():
            continue
        report = json.loads((sync / "rapport_synchronisation.json").read_text(encoding="utf-8"))
        if "platform_geometry" not in report:
            continue
        mocap = pd.read_csv(mocap_path)
        surfaces = plate_surfaces(report["platform_geometry"])
        times = mocap["time_qtm_s"].to_numpy(dtype=np.float64)
        markers = {}
        for side in ("L", "R"):
            try:
                markers[side] = marker_arrays(mocap, side)
            except ValueError:
                continue
            frames = _contact_frames(times, report.get("contact_assignments", []), side)
            for name, positions in markers[side].items():
                for plate in surfaces:
                    selected = frames.get(plate.platform)
                    if selected is None:
                        continue
                    points = positions[selected]
                    over = plate.contains(points[:, :2]) & np.isfinite(points[:, 2])
                    residuals[side].setdefault(name, []).append(
                        points[over, 2] - _plane_z(plate.plane, points[over, :2])
                    )
        loaded.append((surfaces, markers))
        calibration.trials.append(trial.name)

    for side in ("L", "R"):
        calibration.offsets_mm[side] = {}
        calibration.offset_samples[side] = {}
        for name, chunks in sorted(residuals[side].items()):
            values = np.concatenate(chunks) if chunks else np.empty(0)
            calibration.offset_samples[side][name] = int(len(values))
            if len(values) >= MIN_OFFSET_SAMPLES:
                calibration.offsets_mm[side][name] = float(np.median(values))

    floor_xy, floor_z = [], []
    for surfaces, markers in loaded:
        ground = Ground(surfaces, np.zeros(3))
        for side, by_name in markers.items():
            names = [name for name in by_name if name in calibration.offsets_mm[side]]
            if len(names) < MIN_VISIBLE_MARKERS:
                continue
            points = np.stack([by_name[name] for name in names], axis=1)
            offsets = np.array([calibration.offsets_mm[side][name] for name in names])
            visible = np.isfinite(points).all(axis=2)
            off_plate = (~ground.over_plate(points[..., :2]) | ~visible).all(axis=1)
            usable = off_plate & (visible.sum(axis=1) >= MIN_VISIBLE_MARKERS)
            corrected = np.where(visible, points[..., 2] - offsets, np.inf)[usable]
            lowest = np.argmin(corrected, axis=1)
            floor_z.append(corrected[np.arange(len(lowest)), lowest])
            floor_xy.append(points[usable][np.arange(len(lowest)), lowest, :2])
    xy = np.concatenate(floor_xy) if floor_xy else np.empty((0, 2))
    z = np.concatenate(floor_z) if floor_z else np.empty(0)
    calibration.floor, calibration.floor_source, calibration.floor_fit_rmse_mm = _fit_floor(
        xy, z, fallback_floor_z_mm
    )
    calibration.floor_samples = int(len(z))
    return calibration


def foot_height(
    markers: dict[str, np.ndarray],
    offsets_mm: dict[str, float],
    ground: Ground,
    min_visible: int = MIN_VISIBLE_MARKERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Foot height (mm, NaN where invalid) and validity, from (N, 3) marker positions.

    Only markers with a calibrated offset count; the foot is valid where at least
    ``min_visible`` of them are visible.
    """
    names = [name for name in markers if name in offsets_mm]
    if not names:
        length = len(next(iter(markers.values()))) if markers else 0
        return np.full(length, np.nan), np.zeros(length, dtype=bool)
    points = np.stack([markers[name] for name in names], axis=1)
    offsets = np.array([offsets_mm[name] for name in names])
    visible = np.isfinite(points).all(axis=2)
    above = points[..., 2] - offsets - ground.z(points[..., :2])
    above = np.where(visible, above, np.inf)
    valid = visible.sum(axis=1) >= min_visible
    height = np.where(valid, above.min(axis=1), np.nan)
    return height, valid


def marker_visibility(
    markers: dict[str, np.ndarray], offsets_mm: dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Per sample: number of visible calibrated markers, and whether the toe marker is one.

    The toe marker is the calibrated marker whose name contains ``toe`` (every real
    marker set has exactly one per foot). These are the inputs of the Airborne
    marker-reliability rule (#24): at walking push-off the toe and lateral markers
    are hidden while the foot is still loaded on its toe.
    """
    names = [name for name in markers if name in offsets_mm]
    length = len(next(iter(markers.values()))) if markers else 0
    if not names:
        return np.zeros(length, dtype=np.int8), np.zeros(length, dtype=bool)
    visible = np.stack(
        [np.isfinite(markers[name]).all(axis=1) for name in names], axis=1
    )
    toe = np.array(["toe" in name.lower() for name in names])
    return visible.sum(axis=1).astype(np.int8), visible[:, toe].any(axis=1)
