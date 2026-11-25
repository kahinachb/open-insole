"""Synthetic synchronised trials in the real ``synchronized_v2`` format, for CPU tests.

One level earlier in the pipeline than ``synthetic_prepared``: this writes the
input of the prepared-dataset builder, ``<root>/<Subject>/<trial>/synchronized_v2/``
with the synchronisation report and the three synchronised CSVs (insoles, plates,
mocap), using the real column layout and the real foot marker names (4 left,
5 right).

Each Participant ``Name`` has two Subjects, ``Name`` (unloaded) and ``Name2``
(loaded, heavier). Every Subject gets the trials in ``TRIALS``; the Subjects in
``SUBJECTS_WITHOUT_STATIC`` have no static trial.

A trial is a timeline of ``Segment``s per foot. Each segment puts the foot in one
``Pose``:

- ``plate``: flat on its own plate (left on plate 1, right on plate 2), the plate
  carries the foot's load and the insole gauges read it;
- ``lifted``: ``LIFT_MM`` above its plate, plate empty, gauges at the empty-shoe
  preload;
- ``hover``: only ``HOVER_MM`` above its plate (below the v3.6 cluster-origin
  threshold), plate empty, gauges at the preload;
- ``floor``: on the floor off the plates, plate empty, gauges loaded;
- ``heel_off``: on its plate, pitched ``HEEL_OFF_PITCH_DEG`` about the toe tip,
  heel up and toes still on the plate, the plate carries the load.

Marker dropouts (``Dropout``) hide some markers of a foot for a while; they are
optional and per trial name (``write_synthetic_synchronized(dropouts=...)``).
Gauge overrides (``GaugeOverride``) force the insole gauges of a foot to a given
reading for a while, whatever the pose (``write_synthetic_synchronized(gauges=...)``).

Poses change as steps; the report's ``contact_assignments`` list exactly the
on-plate segments (``plate`` and ``heel_off``, back-to-back ones merged). The fixture returns the timeline, so tests can state which
foot state each pose should produce.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PARTICIPANTS = ("T01", "T03", "T04")
SUBJECTS_WITHOUT_STATIC = ("T04b",)
DURATION_S = 2.0
SAMPLE_PERIOD_S = 0.005  # insoles and mocap at 200 Hz
PLATE_PERIOD_S = 0.001  # plates at 1000 Hz
INSOLE_LAG_S = {"L": 0.001, "R": 0.002}  # insole packet time after the mocap frame
SIDE_OFFSET_MS = {"L": 20.0, "R": 25.0}
PATCHES = {"L": (1, 3, 5, 7), "R": (9, 11, 13, 15)}

# Plates as in the real report (rounded): flat tops at z 7.7 and 10.9 mm.
PLATE_CENTERS_MM = {1: (241.6, 296.4, 7.7), 2: (757.7, 300.7, 10.9)}
PLATE_HALF_SIZE_MM = (236.0, 290.0)
PLATE_OF_SIDE = {"L": 1, "R": 2}
FLOOR_Z_MM = -50.0  # the floor sits below the plate tops
FLOOR_Y_MM = -350.0  # a foot on the floor stands in front of the plates (y < 0)
LIFT_MM = 80.0
HOVER_MM = 10.0
HEEL_OFF_PITCH_DEG = 25.0
TOE_TIP_FORWARD_MM = 120.0  # heel-off pivot: the sole under the toe marker
ON_PLATE_POSES = ("plate", "heel_off")
GROUNDED_POSES = ON_PLATE_POSES + ("floor",)
GAUGE_PRELOAD_N = 3.0  # per gauge: 15 gauges, about 45 N per empty shoe

# Foot-frame marker positions (x lateral-positive, y heel to toe, z up from the sole).
# The right foot has two lateral markers, as in the real marker set.
MARKERS = {
    "L": {
        "L_medialM": (-45.0, 40.0, 12.0),
        "L_heelM": (0.0, -110.0, 20.0),
        "L_toeM": (0.0, 120.0, 8.0),
        "L_lateralM": (45.0, 40.0, 12.0),
    },
    "R": {
        "R_lateral1": (45.0, 40.0, 12.0),
        "R_medial": (-45.0, 40.0, 12.0),
        "R_toe": (0.0, 120.0, 8.0),
        "R_heel": (0.0, -110.0, 20.0),
        "R_lateral2": (45.0, -60.0, 14.0),
    },
}
# Medial points to the body midline (x = 500), between the plates.
LATERAL_SIGN = {"L": -1.0, "R": 1.0}


@dataclass(frozen=True)
class Segment:
    start_s: float
    end_s: float
    pose: str  # "plate" | "lifted" | "hover" | "floor" | "heel_off"


@dataclass(frozen=True)
class Dropout:
    """These markers of a foot are not visible (empty mocap fields) in [start, end)."""

    start_s: float
    end_s: float
    markers: tuple[str, ...]


@dataclass(frozen=True)
class GaugeOverride:
    """Every gauge of that foot reads ``per_gauge_n`` in [start, end), whatever the pose."""

    start_s: float
    end_s: float
    per_gauge_n: float


def _timeline(*poses: tuple[float, str]) -> tuple[Segment, ...]:
    """``(end_s, pose)`` pairs, back to back from 0 to ``DURATION_S``."""
    segments, start = [], 0.0
    for end, pose in poses:
        segments.append(Segment(start, end, pose))
        start = end
    return tuple(segments)


STANDING = _timeline((DURATION_S, "plate"))
# trial name -> per-foot timeline. Both feet flight (300 ms) then hover (150 ms) in
# jump; in walk the left foot is in the air for 300 ms then 30 ms (too short to be
# Airborne) while the right foot stays on the floor.
TRIALS = {
    "balance1": {"L": STANDING, "R": STANDING},
    "jump1": {
        side: _timeline((0.6, "plate"), (0.9, "lifted"), (1.3, "plate"), (1.45, "hover"),
                        (DURATION_S, "plate"))
        for side in ("L", "R")
    },
    "walk": {
        "L": _timeline((0.8, "plate"), (1.1, "lifted"), (1.4, "plate"), (1.43, "lifted"),
                       (DURATION_S, "plate")),
        "R": _timeline((DURATION_S, "floor")),
    },
}
STATIC_TRIAL = "static"


@dataclass(frozen=True)
class SyntheticTrial:
    subject: str
    trial: str
    body_weight_n: float
    feet: dict[str, tuple[Segment, ...]]
    dropouts: dict[str, tuple[Dropout, ...]] = field(default_factory=dict)
    gauges: dict[str, tuple[GaugeOverride, ...]] = field(default_factory=dict)

    @property
    def batch(self) -> str:
        return f"{self.subject}_{self.trial}"

    def segment_at(self, side: str, times_s: np.ndarray) -> list[Segment | None]:
        """The segment of that foot holding each time."""
        found: list[Segment | None] = [None] * len(times_s)
        for segment in self.feet[side]:
            for index in np.flatnonzero(self._inside(segment, side, times_s)):
                found[index] = segment
        return found

    def pose_at(self, side: str, times_s: np.ndarray) -> np.ndarray:
        """The pose of that foot at each time; the last pose holds past the end."""
        poses = np.empty(len(times_s), dtype=object)
        for segment in self.feet[side]:
            poses[self._inside(segment, side, times_s)] = segment.pose
        return poses

    def visible_marker_count(self, side: str, times_s: np.ndarray) -> np.ndarray:
        """How many of that foot's markers are visible at each time."""
        hidden = np.zeros((len(times_s), len(MARKERS[side])), dtype=bool)
        for dropout in self.dropouts.get(side, ()):
            inside = (times_s >= dropout.start_s) & (times_s < dropout.end_s)
            for column, name in enumerate(MARKERS[side]):
                if name in dropout.markers:
                    hidden[inside, column] = True
        return len(MARKERS[side]) - hidden.sum(axis=1)

    def _inside(self, segment: Segment, side: str, times_s: np.ndarray) -> np.ndarray:
        end = np.inf if segment is self.feet[side][-1] else segment.end_s
        return (times_s >= segment.start_s) & (times_s < end)

    def distance_to_pose_change_s(self, side: str, times_s: np.ndarray) -> np.ndarray:
        edges = np.array([segment.start_s for segment in self.feet[side][1:]])
        if not len(edges):
            return np.full(len(times_s), np.inf)
        return np.min(np.abs(times_s[:, None] - edges[None, :]), axis=1)


@dataclass(frozen=True)
class SyntheticSynchronized:
    root: Path
    participants: tuple[str, ...]
    subjects: tuple[str, ...]
    trials: tuple[SyntheticTrial, ...]

    def trial(self, batch: str) -> SyntheticTrial:
        return next(trial for trial in self.trials if trial.batch == batch)


def body_weight_n(participant_index: int, loaded: bool) -> float:
    return 600.0 + 50.0 * participant_index + (150.0 if loaded else 0.0)


def _foot_loads(trial: SyntheticTrial, times: np.ndarray) -> dict[str, np.ndarray]:
    """Vertical load per foot: the weight is shared by the feet on the ground."""
    grounded = {
        side: np.isin(trial.pose_at(side, times), GROUNDED_POSES) for side in ("L", "R")
    }
    count = grounded["L"].astype(float) + grounded["R"]
    share = np.divide(trial.body_weight_n, count, out=np.zeros_like(count), where=count > 0)
    return {side: np.where(grounded[side], share, 0.0) for side in ("L", "R")}


def _foot_position(side: str, poses: np.ndarray) -> np.ndarray:
    """Sole position (x, y, z) in mm, global frame, for each pose."""
    x, y, top = PLATE_CENTERS_MM[PLATE_OF_SIDE[side]]
    position = np.tile([x, y, top], (len(poses), 1)).astype(float)
    position[poses == "lifted", 2] += LIFT_MM
    position[poses == "hover", 2] += HOVER_MM
    floor = poses == "floor"
    position[floor, 1] = FLOOR_Y_MM
    position[floor, 2] = FLOOR_Z_MM
    return position


def _marker_offsets(side: str, name: str, poses: np.ndarray) -> np.ndarray:
    """Marker offset from the sole position (x, y, z), mm, for each pose."""
    lateral, forward, up = MARKERS[side][name]
    offsets = np.tile([LATERAL_SIGN[side] * lateral, forward, up], (len(poses), 1)).astype(float)
    pitch = np.radians(HEEL_OFF_PITCH_DEG)
    heel_off = poses == "heel_off"
    # Rotate (forward, up) about the toe tip on the sole: the heel goes up.
    back = forward - TOE_TIP_FORWARD_MM
    offsets[heel_off, 1] = TOE_TIP_FORWARD_MM + back * np.cos(pitch) - up * np.sin(pitch)
    offsets[heel_off, 2] = -back * np.sin(pitch) + up * np.cos(pitch)
    return offsets


def _mocap(trial: SyntheticTrial) -> pd.DataFrame:
    times = np.arange(round(DURATION_S / SAMPLE_PERIOD_S) + 2) * SAMPLE_PERIOD_S
    columns: dict[str, np.ndarray] = {
        "time_qtm_s": times,
        "time_qtm_absolute_s": times + 0.02,
        "frame_index_c3d": np.arange(len(times)) + 4,
    }
    for side in ("R", "L"):  # real column order: right markers first
        poses = trial.pose_at(side, times)
        sole = _foot_position(side, poses)
        for name in MARKERS[side]:
            position = sole + _marker_offsets(side, name, poses)
            for dropout in trial.dropouts.get(side, ()):
                if name in dropout.markers:
                    position[(times >= dropout.start_s) & (times < dropout.end_s)] = np.nan
            for axis_index, axis in enumerate("xyz"):
                columns[f"{name}_{axis}_mm"] = position[:, axis_index]
    return pd.DataFrame(columns)


def _plates(trial: SyntheticTrial) -> pd.DataFrame:
    times = np.arange(round(DURATION_S / PLATE_PERIOD_S) + 10) * PLATE_PERIOD_S
    loads = _foot_loads(trial, times)
    columns: dict[str, np.ndarray] = {
        "time_qtm_s": times,
        "time_qtm_absolute_s": times + 0.02,
        "sample_index_c3d": np.arange(len(times)) + 20,
    }
    global_columns: dict[str, np.ndarray] = {}
    for side in ("L", "R"):
        plate = PLATE_OF_SIDE[side]
        center = np.asarray(PLATE_CENTERS_MM[plate])
        on_plate = np.isin(trial.pose_at(side, times), ON_PLATE_POSES)
        fz = np.where(on_plate, loads[side], 0.0)
        # The CoP sits under the foot (its sole centre, the plate centre here).
        cop = np.tile(center, (len(times), 1))
        lever = cop - center
        force = np.column_stack([np.zeros_like(fz), np.zeros_like(fz), fz])
        moment = np.cross(lever, force)  # about the surface centre, N.mm
        columns.update({
            f"force_x_p{plate}_N": -force[:, 1], f"force_y_p{plate}_N": -force[:, 0],
            f"force_z_p{plate}_N": -force[:, 2],  # local z points down
            f"moment_x_p{plate}_Nmm": -moment[:, 1], f"moment_y_p{plate}_Nmm": -moment[:, 0],
            f"moment_z_p{plate}_Nmm": -moment[:, 2],
        })
        for axis_index, axis in enumerate("xyz"):
            global_columns[f"force_{axis}_p{plate}_global_N"] = force[:, axis_index]
            global_columns[f"moment_{axis}_p{plate}_global_Nmm"] = moment[:, axis_index]
            global_columns[f"cop_{axis}_p{plate}_global_mm"] = cop[:, axis_index]
            global_columns[f"free_moment_{axis}_p{plate}_global_Nmm"] = np.zeros(len(times))
    return pd.DataFrame({**columns, **global_columns})


def _insoles(trial: SyntheticTrial, rng: np.random.Generator) -> pd.DataFrame:
    frames = np.arange(round(DURATION_S / SAMPLE_PERIOD_S))
    blocks = []
    for side in ("R", "L"):
        times = frames * SAMPLE_PERIOD_S + INSOLE_LAG_S[side]
        poses = trial.pose_at(side, times)
        load = _foot_loads(trial, times)[side]
        per_gauge = np.where(np.isin(poses, GROUNDED_POSES), load / 15.0, GAUGE_PRELOAD_N)
        for override in trial.gauges.get(side, ()):
            per_gauge[(times >= override.start_s) & (times < override.end_s)] = (
                override.per_gauge_n
            )
        offset_ms = SIDE_OFFSET_MS[side]
        receiver_us = np.round((times - offset_ms / 1000.0) * 1e6).astype(np.int64) + 3_500_000_000
        for patch_index, patch in enumerate(PATCHES[side]):
            count = len(frames)
            block = {
                "time_qtm_s": times,
                "time_qtm_absolute_s": times + 0.02,
                "signal": np.ones(count, dtype=np.int64),
                "side": side,
                "patch_id": patch,
                "patch_id2": patch + 1,
                "packet_timestamp_unwrapped_us": receiver_us,
                "patch_counter": (frames + 100) % 256,
            }
            gauges = per_gauge[:, None] + rng.normal(scale=0.5, size=(count, 4))
            if patch_index == 3:  # patches 7 and 15 have no gauge 1
                gauges[:, 0] = rng.normal(scale=0.5, size=count)
            for k in range(1, 5):
                block[f"force_{k}_raw_counts"] = np.full(count, 1000 + 10 * k)
            for axis in "xyz":
                block[f"acc_{axis}_raw_counts"] = np.full(count, 4084 if axis == "z" else 10)
                block[f"gyro_{axis}_raw_counts"] = np.zeros(count, dtype=np.int64)
                block[f"mag_{axis}_raw_counts"] = np.full(count, 130000)
            for k in range(1, 5):
                block[f"force_{k}_nominal_N_no_tare"] = gauges[:, k - 1] + 1.0
                block[f"force_{k}_N"] = gauges[:, k - 1]
            for axis in "xyz":
                block[f"acc_{axis}_uncal_g"] = np.full(count, 1.0 if axis == "z" else 0.0)
                block[f"gyro_{axis}_uncal_dps"] = np.zeros(count)
                block[f"mag_{axis}_uncal_uT"] = np.full(count, 30.0)
            acc = rng.normal(scale=0.02, size=(count, 3)) + [0.0, 0.0, 1.0]
            gyro = rng.normal(scale=1.0, size=(count, 3))
            for axis_index, axis in enumerate("xyz"):
                block[f"acc_{axis}_cal_g"] = acc[:, axis_index]
            for axis_index, axis in enumerate("xyz"):
                block[f"gyro_{axis}_cal_dps"] = gyro[:, axis_index]
            for axis in "xyz":
                block[f"mag_{axis}_cal_uT"] = np.full(count, 30.0)
            block["mag_norm_cal_uT"] = np.full(count, 52.0)
            block["side_offset_ms"] = np.full(count, offset_ms)
            blocks.append(pd.DataFrame(block))
    insoles = pd.concat(blocks, ignore_index=True)
    return insoles.sort_values(["time_qtm_s", "side", "patch_id"], kind="stable")


def _report(trial: SyntheticTrial) -> dict:
    geometry = []
    for plate, (x, y, z) in PLATE_CENTERS_MM.items():
        half_x, half_y = PLATE_HALF_SIZE_MM
        corners = [[x + sx * half_x, y + sy * half_y, z] for sx, sy in
                   ((-1, -1), (-1, 1), (1, 1), (1, -1))]
        center = [x, y, z]
        geometry.append({
            "platform": plate,
            "corners_global_mm": corners,
            "surface_center_global_mm": center,
            "surface_center_from_measurement_origin_local_mm": [0.0, 0.0, 0.0],
            "measurement_origin_global_mm": center,
            "origin_global_mm": center,
            "moment_reference_global_mm": center,
            "rotation_local_to_global": [[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        })
    contacts = []
    for side in ("L", "R"):
        for segment in trial.feet[side]:
            if segment.pose not in ON_PLATE_POSES:
                continue
            previous = contacts[-1] if contacts else None
            if previous and previous["matched_side"] == side \
                    and previous["contact_end_s"] == segment.start_s:
                previous["contact_end_s"] = segment.end_s
                continue
            contacts.append({
                "platform": PLATE_OF_SIDE[side],
                "contact_start_s": segment.start_s,
                "contact_end_s": segment.end_s,
                "matched_side": side,
                "correlation": 0.9,
            })
    contacts.sort(key=lambda contact: contact["contact_start_s"])
    insole = {"missing_packets_from_counter": 0, "frequency_hz": 1.0 / SAMPLE_PERIOD_S}
    return {
        "pipeline_version": "synchronized_v2.2_subject_tare_contact_loss",
        "status": "OK",
        "left_insole": dict(insole),
        "right_insole": dict(insole),
        "v2_export_checks": {"overall_status": "OK", "failed_checks": []},
        "packet_loss_assessment": {"status": "NO_LOSS", "total_missing_packets": 0},
        "c3d": {"point_rate_hz": 200, "analog_rate_hz": 1000, "marker_count": 9,
                "platform_count": 2},
        "platform_geometry": geometry,
        "contact_assignments": contacts,
        "common_interval": {"qtm_absolute_start_s": 0.02,
                            "qtm_absolute_end_s": 0.02 + DURATION_S, "duration_s": DURATION_S},
        "marker_visibility": {
            side: {"median_visible": float(len(MARKERS[side])), "fraction_at_least_3": 1.0}
            for side in ("L", "R")
        },
        "outputs": {"semelles": "semelles_synchronisees.csv",
                    "plateformes": "plateformes_synchronisees.csv",
                    "mocap": "mocap_synchronisee.csv", "plots": []},
    }


def write_trial(root: Path, trial: SyntheticTrial, rng: np.random.Generator) -> Path:
    sync = root / trial.subject / trial.trial / "synchronized_v2"
    sync.mkdir(parents=True, exist_ok=True)
    float_format = "%.9f"
    _insoles(trial, rng).to_csv(sync / "semelles_synchronisees.csv", index=False,
                                float_format=float_format)
    _plates(trial).to_csv(sync / "plateformes_synchronisees.csv", index=False,
                          float_format=float_format)
    _mocap(trial).to_csv(sync / "mocap_synchronisee.csv", index=False,
                         float_format=float_format, na_rep="")
    (sync / "rapport_synchronisation.json").write_text(
        json.dumps(_report(trial), indent=2), encoding="utf-8"
    )
    return sync


def write_synthetic_synchronized(
    root: Path,
    participants: tuple[str, ...] = PARTICIPANTS,
    seed: int = 0,
    trials: dict[str, dict[str, tuple[Segment, ...]]] | None = None,
    dropouts: dict[str, dict[str, tuple[Dropout, ...]]] | None = None,
    gauges: dict[str, dict[str, tuple[GaugeOverride, ...]]] | None = None,
) -> SyntheticSynchronized:
    """Write ``<root>/<Subject>/<trial>/synchronized_v2/`` for every Subject and trial.

    ``trials`` replaces ``TRIALS`` (the static trial is still added); ``dropouts``
    maps a trial name to the marker dropouts of each foot in that trial, and
    ``gauges`` to the gauge overrides of each foot.
    """
    rng = np.random.default_rng(seed)
    plan_trials = TRIALS if trials is None else trials
    dropouts = dropouts or {}
    gauges = gauges or {}
    written, subjects = [], []
    for index, participant in enumerate(participants):
        for subject, loaded in ((participant, False), (f"{participant}b", True)):
            subjects.append(subject)
            weight = body_weight_n(index, loaded)
            plan = dict(plan_trials)
            if subject not in SUBJECTS_WITHOUT_STATIC:
                plan[STATIC_TRIAL] = {"L": STANDING, "R": STANDING}
            for name, feet in plan.items():
                trial = SyntheticTrial(subject, name, weight, feet, dropouts.get(name, {}),
                                       gauges.get(name, {}))
                write_trial(root, trial, rng)
                written.append(trial)
    return SyntheticSynchronized(root, tuple(participants), tuple(subjects), tuple(written))
