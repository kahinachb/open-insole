#!/usr/bin/env python3
"""Entraine un LSTM, BiLSTM, MLP, TCN ou Transformer avec loss masquee.

Entree attendue : ``prepare_npy_joint_masked_v3.py``. La loss ne voit que les
composantes dont ``target_mask`` vaut vrai. Les cibles nulles des phases
aeriennes peuvent etre sous-ponderees afin qu'elles ne dominent pas les
mesures reelles des plateformes. Les options historiques restent les valeurs
par defaut; les variantes d'architecture et de features servent aux sweeps.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import re
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib
# Keep training headless by default, without overriding an interactive backend
# already selected by an importing application (for example NPZ inference with
# --show).
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from grfm_schema import BODY_WEIGHT_FEATURE, BODY_WEIGHT_MISSING_FEATURE, participant_of_subject
from sensor_token_model import SensorTokenTransformer, prepare_sensor_token_inputs


STATE_UNKNOWN = 0
STATE_PLATFORM = 1
STATE_AIRBORNE = 2
FEET = ("L", "R")
FOOT_HEIGHT_KEYS = tuple(f"foot_height_mm_{side}" for side in FEET)
FOOT_HEIGHT_VALID_KEYS = tuple(f"foot_height_valid_{side}" for side in FEET)
TASK_SETS = {
    "continuous": {"hulahoop", "balance", "lateral_2supports", "squat"},
    "continuous_walk": {
        "hulahoop",
        "balance",
        "lateral_2supports",
        "squat",
        "walk",
    },
}


@dataclass
class Acquisition:
    name: str
    evaluation_group: str
    subject_session: str
    subject_identity: str
    task_family: str
    x: np.ndarray
    y: np.ndarray
    mask: np.ndarray
    time_s: np.ndarray
    segment_id: np.ndarray
    state_l: np.ndarray
    state_r: np.ndarray
    airborne_l: np.ndarray
    airborne_r: np.ndarray
    load_estimate_n: float = 1.0
    # Foot height (mm, columns L, R; NaN where invalid) and its validity, v3.7 NPZ keys (#14).
    foot_height: np.ndarray | None = None
    foot_height_valid: np.ndarray | None = None
    # Input columns of its Inactive sensors (``--sensor-dropout``), zeroed in every window.
    inactive_columns: tuple[int, ...] = ()


def subject_session_from_batch(batch: str) -> str:
    """Return the acquisition folder name encoded at the start of a batch."""
    return batch.split("_", 1)[0]


def subject_identity_from_session(session: str) -> str:
    """Merge the loaded Subject ``P01b`` with the unloaded ``P01`` (their Participant)."""
    return participant_of_subject(session)


def infer_task_family(batch: str) -> str:
    """Fallback for datasets created before task_family was stored."""
    name = batch.casefold()
    if "__lateral_2supports" in name:
        return "lateral_2supports"
    if "__lateral_1support" in name or "1_appui" in name:
        return "lateral_1_appui"
    if "walk" in name:
        return "walk"
    if "jump" in name:
        return "jump"
    if "lateral" in name:
        return "lateral_2supports"
    for family in ("hulahoop", "balance", "squat"):
        if family in name:
            return family
    return "other"


def load_acquisitions(
    data_dir: Path,
    selected_targets: tuple[str, ...] | None,
    subjects: list[str] | None,
    task_set: str,
) -> tuple[list[Acquisition], dict, tuple[str, ...], list[int]]:
    metadata = json.loads((data_dir / "metadata.json").read_text(encoding="utf-8"))
    if not str(metadata.get("pipeline_version", "")).startswith("prepared_joint_masked_v"):
        raise ValueError("dataset incompatible: executer prepare_npy_joint_masked_v3.py")
    all_targets = tuple(metadata["targets"])
    targets = selected_targets or all_targets
    unknown = sorted(set(targets) - set(all_targets))
    if unknown:
        raise ValueError(f"cibles inconnues: {unknown}")
    target_indexes = [all_targets.index(target) for target in targets]
    names = list(metadata["batches"])
    evaluation_groups = {
        str(report["batch"]): str(report.get("evaluation_group", report["batch"]))
        for report in metadata.get("reports", [])
    }
    task_families = {
        str(report["batch"]): str(
            report.get("task_family", infer_task_family(str(report["batch"])))
        )
        for report in metadata.get("reports", [])
    }
    if subjects:
        prefixes = tuple(f"{subject}_" for subject in subjects)
        names = [name for name in names if name.startswith(prefixes)]
    if task_set != "all":
        allowed = TASK_SETS[task_set]
        names = [
            name
            for name in names
            if task_families.get(name, infer_task_family(name)) in allowed
        ]
    acquisitions = []
    for name in names:
        path = data_dir / f"{name}.npz"
        subject_session = subject_session_from_batch(name)
        with np.load(path) as values:
            state_l = values["state_L"].astype(np.int8)
            state_r = values["state_R"].astype(np.int8)
            has_foot_height = all(key in values for key in FOOT_HEIGHT_KEYS + FOOT_HEIGHT_VALID_KEYS)
            acquisitions.append(
                Acquisition(
                    name=name,
                    evaluation_group=evaluation_groups.get(name, name),
                    subject_session=subject_session,
                    subject_identity=subject_identity_from_session(subject_session),
                    task_family=task_families.get(name, infer_task_family(name)),
                    x=values["X"].astype(np.float32),
                    y=values["y"][:, target_indexes].astype(np.float32),
                    mask=values["target_mask"][:, target_indexes].astype(bool),
                    time_s=values["target_time_s"].astype(np.float64),
                    segment_id=values["segment_id"].astype(np.int64),
                    state_l=state_l,
                    state_r=state_r,
                    airborne_l=(
                        values["airborne_label_L"].astype(bool)
                        if "airborne_label_L" in values
                        else values["airborne_gate_L"].astype(bool)
                        if "airborne_gate_L" in values
                        else state_l == STATE_AIRBORNE
                    ),
                    airborne_r=(
                        values["airborne_label_R"].astype(bool)
                        if "airborne_label_R" in values
                        else values["airborne_gate_R"].astype(bool)
                        if "airborne_gate_R" in values
                        else state_r == STATE_AIRBORNE
                    ),
                    foot_height=(
                        np.stack([values[key] for key in FOOT_HEIGHT_KEYS], axis=1).astype(np.float32)
                        if has_foot_height
                        else None
                    ),
                    foot_height_valid=(
                        np.stack([values[key] for key in FOOT_HEIGHT_VALID_KEYS], axis=1).astype(bool)
                        if has_foot_height
                        else None
                    ),
                )
            )
    if len(acquisitions) < 3:
        raise ValueError("au moins 3 essais sont necessaires (train/validation/test)")
    return acquisitions, metadata, targets, target_indexes


def gauge_indexes(feature_names: list[str], side: str | None = None) -> list[int]:
    """Return force-gauge columns, optionally restricted to one foot."""
    prefix = f"{side}_" if side else None
    return [
        index
        for index, name in enumerate(feature_names)
        if "_force_" in name and (prefix is None or name.startswith(prefix))
    ]


def estimate_insole_load(
    acquisition: Acquisition,
    force_indexes: list[int],
    calibration_seconds: float,
) -> float:
    """Estimate the carried load using only an initial causal insole interval.

    The scalar is the median of the sum of all strain-gauge channels during the
    first ``calibration_seconds``. It therefore uses neither platform nor mocap
    data and can be reproduced from a short standing calibration at deployment.
    """
    elapsed = acquisition.time_s - acquisition.time_s[0]
    selected = elapsed <= calibration_seconds
    if not selected.any():
        selected = np.ones(len(elapsed), dtype=bool)
    total = acquisition.x[selected][:, force_indexes].sum(axis=1, dtype=np.float64)
    finite = total[np.isfinite(total)]
    if not len(finite):
        raise ValueError(f"charge semelle non estimable pour {acquisition.name}")
    estimate = float(np.median(finite))
    if estimate <= 1.0:
        raise ValueError(
            f"charge semelle non physique pour {acquisition.name}: {estimate:.3f} N"
        )
    return estimate


LOADSOL_ZONES = ("heel", "medial", "lateral", "total")


def loadsol_zone_columns(feature_names: list[str]) -> dict[str, dict[str, list[int]]]:
    """Gauge columns of each loadsol-like zone, per foot (``--input-features loadsol``).

    Heel: the heel Patch's Gauges; medial / lateral: the forefoot and toes
    Gauges on either side of the foot's midline (sign of the lateral coordinate
    in the Canonical insole frame, from the insole layout); total: every Gauge
    (the midfoot ones count only there).
    """
    from sensor_tokens import SensorTokenizer

    geometry = SensorTokenizer.from_layout()
    lateral = {
        name: float(position[1])
        for name, position in zip(geometry.token_names, geometry.positions_mm)
    }
    groups = sensor_groups(feature_names)

    def gauges(side, *patches):
        return [
            column for patch in patches for column in groups[f"{patch}_gauges"]
            if feature_names[column].startswith(f"{side}_")
        ]

    zones = {}
    for side in FEET:
        front = gauges(side, "forefoot", "toes")
        zones[side] = {
            "heel": gauges(side, "heel"),
            "medial": [column for column in front if lateral[feature_names[column]] < 0.0],
            "lateral": [column for column in front if lateral[feature_names[column]] >= 0.0],
            "total": gauges(side, *SENSOR_PATCHES),
        }
    return zones


def loadsol_inputs(
    x: np.ndarray, feature_names: list[str], heel_imu: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Loadsol-like inputs: 4 vertical forces per foot (zonal gauge sums), optionally the heel-patch IMU."""
    zones = loadsol_zone_columns(feature_names)
    columns, names = [], []
    for side in FEET:
        for zone in LOADSOL_ZONES:
            columns.append(x[:, zones[side][zone]].sum(axis=1, dtype=np.float32))
            names.append(f"{side}_zone_{zone}_N")
    if heel_imu:
        groups = sensor_groups(feature_names)
        imu = sorted(groups["heel_acc"] + groups["heel_gyro"])
        columns.extend(x[:, column] for column in imu)
        names.extend(feature_names[column] for column in imu)
    return np.stack(columns, axis=1).astype(np.float32), names


def prepare_input_features(
    acquisitions: list[Acquisition],
    feature_names: list[str],
    mode: str,
    proportion_epsilon_n: float,
    load_calibration_seconds: float,
    estimate_load: bool = False,
    loadsol_heel_imu: bool = False,
) -> list[str]:
    """Estimate trial load and optionally append gauge totals/proportions.

    ``loadsol`` replaces the inputs by the loadsol-like zonal sums (see ``loadsol_inputs``).
    """
    if mode == "loadsol":
        for acquisition in acquisitions:
            acquisition.x, _ = loadsol_inputs(acquisition.x, list(feature_names), loadsol_heel_imu)
        return loadsol_inputs(np.zeros((1, len(feature_names)), np.float32), list(feature_names), loadsol_heel_imu)[1]
    if mode == "raw" and not estimate_load:
        # Nothing to derive: raw inputs need no gauge (a Sensor configuration may have none).
        return list(feature_names)
    left = gauge_indexes(feature_names, "L")
    right = gauge_indexes(feature_names, "R")
    if not left or not right:
        raise ValueError("colonnes de jauges gauche/droite introuvables")
    all_force = left + right
    for acquisition in acquisitions:
        if estimate_load:
            acquisition.load_estimate_n = estimate_insole_load(
                acquisition, all_force, load_calibration_seconds
            )
        if mode == "raw":
            continue
        additions = []
        for indexes in (left, right):
            values = acquisition.x[:, indexes]
            total = values.sum(axis=1, keepdims=True, dtype=np.float32)
            safe = np.where(
                np.abs(total) >= proportion_epsilon_n,
                total,
                np.nan,
            )
            proportions = np.divide(
                values,
                safe,
                out=np.zeros_like(values, dtype=np.float32),
                where=np.isfinite(safe),
            )
            additions.extend((total.astype(np.float32), proportions))
        acquisition.x = np.concatenate((acquisition.x, *additions), axis=1)
    if mode == "raw":
        return list(feature_names)
    enriched = list(feature_names)
    for side, indexes in (("L", left), ("R", right)):
        enriched.append(f"{side}_gauge_sum_N")
        enriched.extend(f"{feature_names[index]}_fraction" for index in indexes)
    return enriched


def append_body_weight_input(
    acquisitions: list[Acquisition], feature_names: list[str], metadata: dict
) -> tuple[list[str], dict[str, float]]:
    """Append the Subject's body weight (N, dataset metadata #16) as a constant input column.

    The column is standardised like every input, with the training split's
    statistics only. Returns the new feature names and the body weight of
    each Subject loaded.
    """
    subjects = metadata.get("body_weights", {}).get("subjects", {})
    weights = {}
    for item in acquisitions:
        entry = subjects.get(item.subject_session)
        if entry is None or entry.get("body_weight_N") is None:
            raise ValueError(
                f"--body-weight-input: poids du Subject {item.subject_session} absent de "
                "metadata.json (body_weights.subjects, dataset v3.7)"
            )
        weight = float(entry["body_weight_N"])
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"poids non physique pour {item.subject_session}: {weight}")
        weights[item.subject_session] = weight
        column = np.full((len(item.x), 1), weight, dtype=item.x.dtype)
        item.x = np.concatenate((item.x, column), axis=1)
    return list(feature_names) + [BODY_WEIGHT_FEATURE], dict(sorted(weights.items()))


def append_body_weight_missing_flag(acquisitions: list[Acquisition], feature_names: list[str]) -> list[str]:
    """Append the "body weight missing" flag column (``--body-weight-masking``), 0 in the data.

    The windows set it to 1 where they replace the body weight by the default
    value (``BodyWeightMode``); it is never standardised.
    """
    for item in acquisitions:
        item.x = np.concatenate((item.x, np.zeros((len(item.x), 1), dtype=item.x.dtype)), axis=1)
    return list(feature_names) + [BODY_WEIGHT_MISSING_FEATURE]


@dataclass(frozen=True)
class BodyWeightMode:
    """How the windows of a dataset present the body-weight input (``--body-weight-masking``).

    On standardised windows: ``missing`` (or, per window, with probability
    ``mask_probability``) replaces the body-weight column by ``default``
    (the standardised training mean) and sets the flag column to 1; otherwise
    the flag stays 0 and the body weight is shifted by ``shift_n`` N. Only these
    two columns change.
    """

    column: int
    flag_column: int | None
    default: float
    scale_n: float
    mask_probability: float = 0.0
    missing: bool = False
    shift_n: float = 0.0

    @classmethod
    def from_statistics(cls, feature_names, x_mean, x_std, **options) -> BodyWeightMode:
        names = list(feature_names)
        column = names.index(BODY_WEIGHT_FEATURE)
        flag = names.index(BODY_WEIGHT_MISSING_FEATURE) if BODY_WEIGHT_MISSING_FEATURE in names else None
        # The default value is the training mean, i.e. 0 once standardised.
        return cls(column, flag, 0.0, float(x_std[column]), **options)

    def apply(self, x: np.ndarray) -> np.ndarray:
        missing = self.missing or (
            self.mask_probability > 0.0 and float(torch.rand(())) < self.mask_probability
        )
        if missing:
            x[:, self.column] = self.default
            if self.flag_column is not None:
                x[:, self.flag_column] = 1.0
        elif self.shift_n:
            x[:, self.column] += self.shift_n / self.scale_n
        return x


SENSOR_PATCHES = ("heel", "midfoot", "forefoot", "toes")
SENSOR_MODALITIES = {"force": "gauges", "acc": "acc", "gyro": "gyro"}
SENSOR_CHANNEL = re.compile(r"^([LR])_p(\d+)_(force|acc|gyro)_")
PRESENCE_PREFIX = "presence_"
INACTIVE_GAUGE_P99_N = 5.0
# ``--sensor-dropout`` defaults: share of windows keeping the full configuration, per-channel drop.
SENSOR_FULL_PROBABILITY = 0.3
SENSOR_CHANNEL_DROPOUT = 0.02


def sensor_groups(feature_names: list[str]) -> dict[str, list[int]]:
    """Input columns of each Sensor group (both feet), heel to toes, gauges / acc / gyro.

    A foot's Patches are ranked by their number (``p1 p3 p5 p7`` left, ``p9 ...
    p15`` right), so both feet map to the same twelve groups.
    """
    channels = [(column, SENSOR_CHANNEL.match(name)) for column, name in enumerate(feature_names)]
    channels = [(column, match) for column, match in channels if match]
    patches = {
        side: sorted({int(match[2]) for _, match in channels if match[1] == side}) for side in FEET
    }
    groups = {f"{patch}_{modality}": [] for patch in SENSOR_PATCHES for modality in SENSOR_MODALITIES.values()}
    for column, match in channels:
        patch = SENSOR_PATCHES[patches[match[1]].index(int(match[2]))]
        groups[f"{patch}_{SENSOR_MODALITIES[match[3]]}"].append(column)
    return groups


def check_sensor_groups(feature_names: list[str], configuration) -> dict[str, list[int]]:
    """The Sensor groups of ``feature_names``; every name of ``configuration`` must be one of them."""
    groups = sensor_groups(feature_names)
    unknown = sorted(set(configuration or ()) - set(groups))
    if unknown:
        raise ValueError(f"Sensor groups inconnus: {', '.join(unknown)} (connus: {', '.join(groups)})")
    return groups


def append_sensor_presence_names(feature_names: list[str]) -> list[str]:
    return list(feature_names) + [f"{PRESENCE_PREFIX}{group}" for group in sensor_groups(feature_names)]


def append_sensor_presence(acquisitions: list[Acquisition], feature_names: list[str]) -> list[str]:
    """Append the twelve Sensor group presence columns (``--sensor-dropout``), 0 in the data.

    The windows write them (``SensorDropout``), after standardisation: the
    fraction of the group's channels present. Each trial also records its
    Inactive sensors, from the raw gauge readings.
    """
    names = append_sensor_presence_names(feature_names)
    for item in acquisitions:
        item.inactive_columns = inactive_gauge_columns(item.x, feature_names)
        presence = np.zeros((len(item.x), len(names) - len(feature_names)), dtype=item.x.dtype)
        item.x = np.concatenate((item.x, presence), axis=1)
    return names


def restrict_to_sensor_configuration(
    acquisitions: list[Acquisition], feature_names: list[str], configuration: list[str]
) -> list[str]:
    """Keep only the channels of the named Sensor groups (``--sensor-configuration``).

    The other channels are removed, not zeroed: the model has exactly the
    inputs of an insole carrying only these sensors.
    """
    groups = check_sensor_groups(feature_names, configuration)
    kept = sorted(column for group in configuration for column in groups[group])
    for item in acquisitions:
        item.x = item.x[:, kept]
    return [feature_names[column] for column in kept]


def inactive_gauge_columns(x: np.ndarray, feature_names: list[str]) -> tuple[int, ...]:
    """Gauge columns whose 99th percentile over the trial stays below 5 N (Inactive sensors)."""
    gauges = [column for column, name in enumerate(feature_names) if (match := SENSOR_CHANNEL.match(name)) and match[3] == "force"]
    p99 = np.percentile(x[:, gauges], 99, axis=0)
    return tuple(column for column, value in zip(gauges, p99) if value < INACTIVE_GAUGE_P99_N)


@dataclass(frozen=True)
class SensorDropout:
    """Which Sensor groups the windows of a dataset present (``--sensor-dropout``).

    On standardised windows. With ``configuration`` the named groups are kept,
    and without ``random`` the full configuration; with ``random`` each window keeps the full configuration with probability
    ``full_probability``, or else draws a size k uniformly in 1..12 and k groups
    at random (both feet always together). Kept channels are then dropped one by
    one with probability ``channel_probability``, and Inactive sensors always.
    Removed channels become 0 (the training mean); each presence column holds
    the fraction of its group's channels present (1 kept whole, 0 absent, in
    between when single channels are missing), so channel drops and Inactive
    sensors are told to the model as precisely as group drops.
    """

    groups: tuple[tuple[int, ...], ...]
    group_names: tuple[str, ...]
    presence_columns: tuple[int, ...]
    full_probability: float = SENSOR_FULL_PROBABILITY
    channel_probability: float = 0.0
    configuration: tuple[str, ...] | None = None
    random: bool = False

    @classmethod
    def from_feature_names(cls, feature_names, configuration=None, presence=True, **options) -> SensorDropout:
        """Without ``presence`` (a model trained without Sensor dropout) the removed channels are
        zeroed and no presence column is written."""
        names = list(feature_names)
        groups = check_sensor_groups(names, configuration)
        return cls(
            tuple(tuple(columns) for columns in groups.values()),
            tuple(groups),
            tuple(names.index(f"{PRESENCE_PREFIX}{group}") for group in groups) if presence else (),
            configuration=None if configuration is None else tuple(configuration),
            **options,
        )

    def kept(self) -> list[int]:
        if self.configuration is not None:
            return sorted(self.group_names.index(name) for name in self.configuration)
        if not self.random or float(torch.rand(())) < self.full_probability:
            return list(range(len(self.groups)))
        size = int(torch.randint(1, len(self.groups) + 1, ()))
        return sorted(torch.randperm(len(self.groups))[:size].tolist())

    def apply(self, x: np.ndarray, inactive_columns=()) -> np.ndarray:
        kept = set(self.kept())
        inactive = set(inactive_columns)
        for index, columns in enumerate(self.groups):
            present = np.zeros(len(columns), dtype=bool)
            if index in kept:
                present = np.array([column not in inactive for column in columns])
                if self.random and self.channel_probability > 0.0:
                    present &= (torch.rand(len(columns)) >= self.channel_probability).numpy()
            x[:, [column for column, keep in zip(columns, present) if not keep]] = 0.0
            if self.presence_columns:
                x[:, self.presence_columns[index]] = present.mean()
        return x


class BodyWeightMissingCounter:
    """Loader wrapper counting the windows whose body weight is missing (flag 1) in each pass."""

    def __init__(self, loader: DataLoader, flag_column: int) -> None:
        self.loader, self.flag_column = loader, flag_column
        self.dataset = loader.dataset
        self.missing, self.seen = 0, 0

    def __iter__(self):
        self.missing, self.seen = 0, 0
        for batch in self.loader:
            self.missing += int((batch[0][:, -1, self.flag_column] > 0.5).sum())
            self.seen += len(batch[0])
            yield batch

    def __len__(self) -> int:
        return len(self.loader)

    @property
    def fraction(self) -> float:
        return self.missing / self.seen if self.seen else math.nan


def split_group(acquisition: Acquisition, split_level: str) -> str:
    if split_level == "subject_identity":
        return acquisition.subject_identity
    return acquisition.evaluation_group


def build_split_manifest(
    acquisitions: list[Acquisition], split_level: str, validation_group_count: int = 1
) -> tuple[list[str], list[dict]]:
    """Build deterministic, leakage-free train/validation/test folds."""
    groups = sorted({split_group(item, split_level) for item in acquisitions})
    if len(groups) < 3:
        raise ValueError("au moins 3 groupes sont necessaires (train/validation/test)")
    if not 1 <= validation_group_count <= len(groups) - 2:
        raise ValueError(
            "validation_group_count doit laisser au moins un groupe train et un test"
        )
    folds = []
    for test_index, test_group in enumerate(groups):
        # Rotation deterministe: les groupes suivant le test servent a la
        # validation. Chaque identite est ainsi utilisee le meme nombre de fois.
        validation_groups = [
            groups[(test_index + offset) % len(groups)]
            for offset in range(1, validation_group_count + 1)
        ]
        train_groups = [
            group
            for group in groups
            if group != test_group and group not in validation_groups
        ]
        folds.append(
            split_fold(acquisitions, split_level, [test_group], validation_groups, train_groups)
        )
    return groups, folds


def split_fold(
    acquisitions: list[Acquisition],
    split_level: str,
    test_groups: list[str],
    validation_groups: list[str],
    train_groups: list[str],
) -> dict:
    """One fold of the split manifest: its groups and the trials of each subset."""
    fold = {
        # Nom du pli: le groupe test (CV complete) ou les groupes test joints par "+".
        "test_group": "+".join(test_groups),
        "test_groups": list(test_groups),
        # Champ historique conserve pour les anciens outils/checkpoints.
        "validation_group": validation_groups[0],
        "validation_groups": list(validation_groups),
        "train_groups": list(train_groups),
    }
    for subset, selected_groups in (
        ("train", set(train_groups)),
        ("validation", set(validation_groups)),
        ("test", set(test_groups)),
    ):
        selected = [
            item
            for item in acquisitions
            if split_group(item, split_level) in selected_groups
        ]
        fold[f"{subset}_batches"] = [item.name for item in selected]
        fold[f"{subset}_sessions"] = sorted(
            {item.subject_session for item in selected}
        )
        fold[f"{subset}_identities"] = sorted(
            {item.subject_identity for item in selected}
        )
    return fold


def build_fixed_split(
    acquisitions: list[Acquisition],
    split_level: str,
    test_groups: list[str],
    validation_groups: list[str],
) -> tuple[list[str], list[dict]]:
    """Fixed split: one fold with the given test and validation groups, the rest train."""
    groups = sorted({split_group(item, split_level) for item in acquisitions})
    unknown = sorted((set(test_groups) | set(validation_groups)) - set(groups))
    if unknown:
        raise ValueError(f"groupes inconnus {unknown}; disponibles: {groups}")
    if set(test_groups) & set(validation_groups):
        raise ValueError(
            f"groupes a la fois test et validation: {sorted(set(test_groups) & set(validation_groups))}"
        )
    train_groups = [
        group for group in groups if group not in test_groups and group not in validation_groups
    ]
    if not train_groups:
        raise ValueError("le split fixe ne laisse aucun groupe d'entrainement")
    return groups, [
        split_fold(
            acquisitions, split_level, sorted(set(test_groups)), sorted(set(validation_groups)), train_groups
        )
    ]


def statistics(
    acquisitions: list[Acquisition],
    target_indexes: list[int],
    zero_weight: float,
    target_scaling: str = "standard",
) -> tuple[np.ndarray, ...]:
    x = np.concatenate([item.x for item in acquisitions])
    x_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    x_std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-8] = 1.0
    target_count = acquisitions[0].y.shape[1]
    y_mean = np.zeros(target_count, dtype=np.float32)
    y_std = np.ones(target_count, dtype=np.float32)
    for index in range(target_count):
        source_index = target_indexes[index]
        selected_values = []
        selected_weights = []
        for item in acquisitions:
            selected = item.mask[:, index].copy()
            state = item.state_l if source_index < 6 else item.state_r
            weights = np.ones(len(state), dtype=np.float64)
            weights[state == STATE_AIRBORNE] = zero_weight
            selected &= weights > 0.0
            scale = item.load_estimate_n if target_scaling == "subject_load" else 1.0
            selected_values.append(item.y[selected, index] / scale)
            selected_weights.append(weights[selected])
        values = np.concatenate(selected_values)
        weights = np.concatenate(selected_weights)
        if not len(values):
            raise ValueError(f"aucune cible valide d'indice {index} dans le train")
        mean = float(np.average(values.astype(np.float64), weights=weights))
        y_mean[index] = mean
        deviation = float(
            np.sqrt(np.average(np.square(values.astype(np.float64) - mean), weights=weights))
        )
        y_std[index] = deviation if deviation >= 1e-8 else 1.0
    return x_mean, x_std, y_mean, y_std


def physical_target_std(
    acquisitions: list[Acquisition],
    target_indexes: list[int],
    zero_weight: float,
) -> np.ndarray:
    """Physical train-target scales shared by all target-scaling variants."""
    result = np.ones(acquisitions[0].y.shape[1], dtype=np.float32)
    for index, source_index in enumerate(target_indexes):
        values, weights = [], []
        for item in acquisitions:
            selected = item.mask[:, index]
            state = item.state_l if source_index < 6 else item.state_r
            item_weights = np.ones(len(state), dtype=np.float64)
            item_weights[state == STATE_AIRBORNE] = zero_weight
            selected = selected & (item_weights > 0.0)
            values.append(item.y[selected, index].astype(np.float64))
            weights.append(item_weights[selected])
        values_array = np.concatenate(values)
        weights_array = np.concatenate(weights)
        mean = float(np.average(values_array, weights=weights_array))
        deviation = float(
            np.sqrt(
                np.average(
                    np.square(values_array - mean), weights=weights_array
                )
            )
        )
        result[index] = deviation if deviation >= 1e-8 else 1.0
    return result


def foot_height_statistics(acquisitions: list[Acquisition]) -> tuple[np.ndarray, np.ndarray]:
    """Mean and std (mm) of each foot's valid Foot height over the given (train) trials."""
    missing = [item.name for item in acquisitions if item.foot_height is None]
    if missing:
        raise ValueError(f"Foot height absente (cles {', '.join(FOOT_HEIGHT_KEYS)}): {missing[:3]}")
    mean = np.zeros(len(FEET), dtype=np.float32)
    std = np.ones(len(FEET), dtype=np.float32)
    for column, side in enumerate(FEET):
        values = np.concatenate([
            item.foot_height[item.foot_height_valid[:, column], column] for item in acquisitions
        ]).astype(np.float64)
        if not len(values):
            raise ValueError(f"aucune Foot height valide ({side}) dans le train")
        mean[column] = values.mean()
        deviation = float(values.std())
        std[column] = deviation if deviation >= 1e-8 else 1.0
    return mean, std


class MaskedWindowDataset(Dataset):
    """Windows ending (causal) or centred on a sample with a valid GRFM target.

    With Foot height statistics (``--foot-height-output``), a window is also kept
    when only its Foot height is valid (Masked samples included), and each item
    gains the standardised Foot height (L, R) and its validity weight.
    """

    def __init__(
        self,
        acquisitions: list[Acquisition],
        x_mean: np.ndarray,
        x_std: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        target_indexes: list[int],
        window: int,
        centered: bool,
        zero_weight: float,
        target_scaling: str = "standard",
        stride: int = 1,
        foot_height_mean: np.ndarray | None = None,
        foot_height_std: np.ndarray | None = None,
        body_weight: BodyWeightMode | None = None,
        sensor_dropout: SensorDropout | None = None,
    ) -> None:
        self.acquisitions = acquisitions
        self.body_weight = body_weight
        self.sensor_dropout = sensor_dropout
        self.foot_height = foot_height_mean is not None
        self.foot_height_mean, self.foot_height_std = foot_height_mean, foot_height_std
        self.x_mean, self.x_std = x_mean, x_std
        self.y_mean, self.y_std = y_mean, y_std
        self.target_indexes = target_indexes
        self.window, self.centered = window, centered
        self.zero_weight = zero_weight
        self.target_scaling = target_scaling
        self.index: list[tuple[int, int]] = []
        left = window // 2 if centered else window - 1
        right = window - left - 1
        for acquisition_index, acquisition in enumerate(acquisitions):
            supervised = acquisition.mask.any(axis=1)
            if self.foot_height:
                supervised = supervised | acquisition.foot_height_valid.any(axis=1)
            for target_index in range(left, len(acquisition.x) - right, stride):
                start, end = target_index - left, target_index + right + 1
                same_segment = (
                    acquisition.segment_id[start]
                    == acquisition.segment_id[target_index]
                    == acquisition.segment_id[end - 1]
                )
                if same_segment and supervised[target_index]:
                    self.index.append((acquisition_index, target_index))

    def __len__(self) -> int:
        return len(self.index)

    def both_feet_measured(self, acquisition_indexes, target_indexes) -> torch.Tensor:
        """Whether both feet are Measured at each (acquisition, sample) of a batch."""
        if not hasattr(self, "_both_measured"):
            both = [
                (item.state_l == STATE_PLATFORM) & (item.state_r == STATE_PLATFORM)
                for item in self.acquisitions
            ]
            self._both_offsets = np.cumsum([0, *(len(values) for values in both[:-1])])
            self._both_measured = np.concatenate(both)
        positions = self._both_offsets[np.asarray(acquisition_indexes)] + np.asarray(target_indexes)
        return torch.from_numpy(self._both_measured[positions])

    def bounds(self, target_index: int) -> tuple[int, int]:
        start = target_index - self.window // 2 if self.centered else target_index - self.window + 1
        return start, start + self.window

    def __getitem__(self, item: int):
        acquisition_index, target_index = self.index[item]
        acquisition = self.acquisitions[acquisition_index]
        start, end = self.bounds(target_index)
        x = (acquisition.x[start:end] - self.x_mean) / self.x_std
        if self.body_weight is not None:
            x = self.body_weight.apply(x)
        if self.sensor_dropout is not None:
            x = self.sensor_dropout.apply(x, acquisition.inactive_columns)
        scale = (
            acquisition.load_estimate_n
            if self.target_scaling == "subject_load"
            else 1.0
        )
        y = (acquisition.y[target_index] / scale - self.y_mean) / self.y_std
        weight = acquisition.mask[target_index].astype(np.float32)
        for output_index, source_index in enumerate(self.target_indexes):
            state = acquisition.state_l[target_index] if source_index < 6 else acquisition.state_r[target_index]
            if state == STATE_AIRBORNE:
                weight[output_index] *= self.zero_weight
        item = (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(weight),
            acquisition_index,
            target_index,
        )
        if not self.foot_height:
            return item
        valid = acquisition.foot_height_valid[target_index]
        height = (acquisition.foot_height[target_index] - self.foot_height_mean) / self.foot_height_std
        height = np.where(valid, height, 0.0).astype(np.float32)
        return (*item, torch.from_numpy(height), torch.from_numpy(valid.astype(np.float32)))


def input_norm(size: int, enabled: bool) -> nn.Module:
    """``--input-layer-norm`` (auto-correction #9): LayerNorm of the first representation.

    This is how the spec's "pre-norm" correction is implemented. The
    Transformer models are already pre-norm (``norm_first=True`` in the
    temporal Transformer, pre-norm spatial layers in the sensor-token model),
    so for them the flag normalises the embedding entering the layers; the
    MLP, LSTM/BiLSTM and TCN have no residual attention blocks to make pre-norm,
    so the flag normalises their inputs per sample.

    Identity by default, so the default architectures and their checkpoints
    are unchanged.
    """
    return nn.LayerNorm(size) if enabled else nn.Identity()


def foot_height_head(input_size: int, hidden_size: int) -> nn.Sequential:
    """Foot height head (``--foot-height-output``): one standardised value per foot (L, R).

    Each model attaches it to the representation its GRFM head reads (see
    ``forward_with_foot_height``). It is created last in the constructor, so
    the other parameters are initialised exactly as without the head.
    """
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, len(FEET)),
    )


class GRFMLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        layers: int,
        dropout: float,
        output_size: int,
        bidirectional: bool = False,
        input_layer_norm: bool = False,
        foot_height_output: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.input_norm = input_norm(input_size, input_layer_norm)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout if layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        encoded_size = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(encoded_size),
            nn.Linear(encoded_size, max(hidden_size // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(hidden_size // 2, 1), output_size),
        )
        # Foot height head: on the LSTM encoding read by the GRFM head.
        self.foot_height_head = (
            foot_height_head(encoded_size, max(hidden_size // 2, 1)) if foot_height_output else None
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(self.input_norm(x))
        return (
            torch.cat((hidden[-2], hidden[-1]), dim=1)
            if self.bidirectional
            else hidden[-1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def forward_with_foot_height(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)
        return self.head(encoded), self.foot_height_head(encoded)

    def last_shared_layer(self) -> nn.Module:
        """Derniere couche partagee par les 12 cibles (GradNorm)."""
        return self.head[1]


class GRFMMLP(nn.Module):
    """Temporal MLP operating on the same flattened causal window."""

    def __init__(
        self,
        input_size: int,
        window_samples: int,
        hidden_size: int,
        layers: int,
        dropout: float,
        output_size: int,
        input_layer_norm: bool = False,
        foot_height_output: bool = False,
    ):
        super().__init__()
        self.input_norm = input_norm(input_size, input_layer_norm)
        blocks: list[nn.Module] = [nn.Flatten()]
        width = input_size * window_samples
        for _ in range(layers):
            blocks.extend(
                (
                    nn.Linear(width, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
            width = hidden_size
        blocks.append(nn.Linear(width, output_size))
        self.network = nn.Sequential(*blocks)
        # Foot height head: on the last hidden layer, which the GRFM output layer reads.
        self.foot_height_head = (
            foot_height_head(width, max(width // 2, 1)) if foot_height_output else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(self.input_norm(x))

    def forward_with_foot_height(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.network[:-1](self.input_norm(x))
        return self.network[-1](hidden), self.foot_height_head(hidden)

    def last_shared_layer(self) -> nn.Module:
        """Derniere couche cachee lineaire, partagee par les 12 cibles (GradNorm)."""
        if len(self.network) < 6:
            raise ValueError("GradNorm requiert au moins une couche cachee (--layers >= 1)")
        return self.network[-5]


class CausalConv1d(nn.Module):
    """One-dimensional convolution padded only on the causal (left) side."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            input_channels,
            output_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convolution(F.pad(x, (self.left_padding, 0)))


class ChannelLayerNorm(nn.Module):
    """Layer-normalize channels independently at every time sample."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalization(x.transpose(1, 2)).transpose(1, 2)


class TCNResidualBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(
            input_channels, output_channels, kernel_size, dilation
        )
        self.norm1 = ChannelLayerNorm(output_channels)
        self.conv2 = CausalConv1d(
            output_channels, output_channels, kernel_size, dilation
        )
        self.norm2 = ChannelLayerNorm(output_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Conv1d(input_channels, output_channels, kernel_size=1)
            if input_channels != output_channels
            else nn.Identity()
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        encoded = self.dropout(self.activation(self.norm1(self.conv1(x))))
        encoded = self.dropout(self.norm2(self.conv2(encoded)))
        return self.activation(encoded + residual)


class GRFMTCN(nn.Module):
    """Residual causal temporal convolutional network, read at current time."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        blocks: int,
        dropout: float,
        output_size: int,
        kernel_size: int = 3,
        input_layer_norm: bool = False,
        foot_height_output: bool = False,
    ) -> None:
        super().__init__()
        self.input_norm = input_norm(input_size, input_layer_norm)
        residual_blocks: list[nn.Module] = []
        channels = input_size
        for block_index in range(blocks):
            residual_blocks.append(
                TCNResidualBlock(
                    channels,
                    hidden_size,
                    kernel_size,
                    dilation=2**block_index,
                    dropout=dropout,
                )
            )
            channels = hidden_size
        self.temporal = nn.Sequential(*residual_blocks)
        head_size = max(hidden_size // 2, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, output_size),
        )
        # Foot height head: on the temporal encoding at the current sample.
        self.foot_height_head = (
            foot_height_head(hidden_size, head_size) if foot_height_output else None
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.input_norm(x).transpose(1, 2))[:, :, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def forward_with_foot_height(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)
        return self.head(encoded), self.foot_height_head(encoded)

    def last_shared_layer(self) -> nn.Module:
        """Derniere couche partagee par les 12 cibles (GradNorm)."""
        return self.head[1]


class GRFMTransformer(nn.Module):
    """Small strictly causal Transformer read at the current time sample.

    Pre-norm encoder layers (``norm_first=True``); ``--input-layer-norm`` adds a
    LayerNorm on the input embedding (see :func:`input_norm`).
    """

    def __init__(
        self,
        input_size: int,
        window_samples: int,
        hidden_size: int,
        layers: int,
        dropout: float,
        output_size: int,
        heads: int = 4,
        feedforward_size: int = 128,
        input_layer_norm: bool = False,
        foot_height_output: bool = False,
    ) -> None:
        super().__init__()
        self.window_samples = window_samples
        self.input_projection = nn.Linear(input_size, hidden_size)
        # The encoder layers are already pre-norm; the option normalises the embedding.
        self.input_norm = input_norm(hidden_size, input_layer_norm)
        self.position_embedding = nn.Parameter(
            torch.empty(1, window_samples, hidden_size)
        )
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            norm=nn.LayerNorm(hidden_size),
        )
        causal_mask = torch.triu(
            torch.ones(window_samples, window_samples, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        head_size = max(hidden_size // 2, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, head_size),
            nn.ReLU(),
            nn.Linear(head_size, output_size),
        )
        # Foot height head: on the encoding at the current sample, like the GRFM head.
        self.foot_height_head = (
            foot_height_head(hidden_size, head_size) if foot_height_output else None
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        if sequence_length > self.window_samples:
            raise ValueError(
                f"sequence de {sequence_length} echantillons superieure a la "
                f"fenetre configuree ({self.window_samples})"
            )
        encoded = self.input_norm(self.input_projection(x))
        encoded = encoded + self.position_embedding[:, :sequence_length]
        return self.encoder(
            encoded,
            mask=self.causal_mask[:sequence_length, :sequence_length],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x)[:, -1])

    def forward_with_foot_height(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)[:, -1]
        return self.head(encoded), self.foot_height_head(encoded)

    def last_shared_layer(self) -> nn.Module:
        """Derniere couche partagee par les 12 cibles (GradNorm)."""
        return self.head[0]


def sine_cosine_positions(count: int, size: int) -> torch.Tensor:
    """(1, count, size) fixed sine-cosine position embedding (sines then cosines)."""
    positions = torch.arange(count, dtype=torch.float32)[:, None]
    frequencies = 1.0 / (10000.0 ** (torch.arange(size // 2, dtype=torch.float32) / max(size // 2, 1)))
    angles = positions * frequencies[None, :]
    embedding = torch.zeros(count, size)
    embedding[:, : size // 2] = torch.sin(angles)
    embedding[:, size // 2 : 2 * (size // 2)] = torch.cos(angles)
    return embedding[None]


class GRFMPatchTransformer(nn.Module):
    """Transformer over temporal patches, read by one learned query per foot.

    The window is cut into patches of ``patch_samples`` samples ending at the
    current sample (edge-padded on the left when the window is not a multiple);
    one patch of every channel is one token, with fixed sine-cosine positions.
    The encoder attends freely across the patches: they are all present or past
    samples, so the model stays causal for Deployment. Two learned queries (the
    Readout tokens, L then R) read the encoded patches through a decoder; the
    conditioning inputs (``condition_columns``: body weight and its missing
    flag, at the current sample) are projected and added to both queries. The
    head reads both queries and starts near zero, i.e. at the mean standardised
    target (and, under ``--vertical-share``, at the body weight shared equally).
    """

    def __init__(
        self,
        input_size: int,
        window_samples: int,
        hidden_size: int,
        layers: int,
        dropout: float,
        output_size: int,
        heads: int = 4,
        feedforward_size: int = 128,
        patch_samples: int = 5,
        decoder_layers: int = 1,
        condition_columns: tuple[int, ...] = (),
        input_layer_norm: bool = False,
        foot_height_output: bool = False,
    ) -> None:
        super().__init__()
        self.window_samples = window_samples
        self.patch_samples = patch_samples
        self.patch_count = math.ceil(window_samples / patch_samples)
        self.patch_embedding = nn.Linear(input_size * patch_samples, hidden_size)
        # The encoder layers are pre-norm; the option normalises the patch embedding.
        self.input_norm = input_norm(hidden_size, input_layer_norm)
        self.register_buffer(
            "position_embedding", sine_cosine_positions(self.patch_count, hidden_size), persistent=False
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=heads, dim_feedforward=feedforward_size, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=layers,
            norm=nn.LayerNorm(hidden_size),
            enable_nested_tensor=False,
        )
        self.queries = nn.Parameter(torch.randn(1, len(FEET), hidden_size) * 0.02)
        self.condition_columns = list(condition_columns)
        self.condition = (
            nn.Sequential(
                nn.Linear(len(self.condition_columns), hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
            )
            if self.condition_columns
            else None
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_size, nhead=heads, dim_feedforward=feedforward_size, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=decoder_layers,
            norm=nn.LayerNorm(hidden_size),
        )
        readout_size = hidden_size * len(FEET)
        head_size = max(hidden_size // 2, 1)
        self.head = nn.Sequential(
            nn.Linear(readout_size, head_size),
            nn.GELU(),
            nn.Linear(head_size, output_size),
        )
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.head[-1].bias)
        # Foot height head: on both Readout tokens, like the GRFM head.
        self.foot_height_head = (
            foot_height_head(readout_size, head_size) if foot_height_output else None
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        sequence_length = x.shape[1]
        if sequence_length > self.window_samples:
            raise ValueError(
                f"sequence de {sequence_length} echantillons superieure a la "
                f"fenetre configuree ({self.window_samples})"
            )
        patch_count = math.ceil(sequence_length / self.patch_samples)
        padding = patch_count * self.patch_samples - sequence_length
        padded = torch.cat((x[:, :1].expand(-1, padding, -1), x), dim=1) if padding else x
        patches = padded.reshape(x.shape[0], patch_count, -1)
        tokens = self.input_norm(self.patch_embedding(patches)) + self.position_embedding[:, :patch_count]
        queries = self.queries.expand(x.shape[0], -1, -1)
        if self.condition is not None:
            queries = queries + self.condition(x[:, -1, self.condition_columns]).unsqueeze(1)
        return self.decoder(queries, self.encoder(tokens)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))

    def forward_with_foot_height(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encode(x)
        return self.head(encoded), self.foot_height_head(encoded)

    def last_shared_layer(self) -> nn.Module:
        """Derniere couche partagee par les 12 cibles (GradNorm)."""
        return self.head[0]


# Inputs that condition the Readout tokens of the patch Transformer, when present (with the
# Sensor group presence columns).
CONDITION_FEATURES = (BODY_WEIGHT_FEATURE, BODY_WEIGHT_MISSING_FEATURE)


def build_model(
    model_type: str,
    input_size: int,
    window_samples: int,
    hidden_size: int,
    layers: int,
    dropout: float,
    output_size: int,
    tcn_kernel_size: int = 3,
    transformer_heads: int = 4,
    transformer_feedforward_size: int = 128,
    feature_names: list[str] | None = None,
    token_dropout: float = 0.1,
    input_layer_norm: bool = False,
    foot_height_output: bool = False,
    patch_samples: int = 5,
    decoder_layers: int = 1,
) -> nn.Module:
    if model_type == "patch_transformer":
        names = list(feature_names or [])
        return GRFMPatchTransformer(
            input_size,
            window_samples,
            hidden_size,
            layers,
            dropout,
            output_size,
            heads=transformer_heads,
            feedforward_size=transformer_feedforward_size,
            patch_samples=patch_samples,
            decoder_layers=decoder_layers,
            condition_columns=tuple(
                index for index, name in enumerate(names)
                if name in CONDITION_FEATURES or name.startswith(PRESENCE_PREFIX)
            ),
            input_layer_norm=input_layer_norm,
            foot_height_output=foot_height_output,
        )
    if model_type == "sensor_token":
        if feature_names is None or len(feature_names) != input_size:
            raise ValueError("sensor_token a besoin des noms des features d'entree")
        return SensorTokenTransformer(
            feature_names,
            window_samples,
            hidden_size,
            layers,
            dropout,
            output_size,
            heads=transformer_heads,
            feedforward_size=transformer_feedforward_size,
            token_dropout=token_dropout,
            input_layer_norm=input_layer_norm,
            foot_height_output=foot_height_output,
        )
    if model_type == "mlp":
        return GRFMMLP(
            input_size,
            window_samples,
            hidden_size,
            layers,
            dropout,
            output_size,
            input_layer_norm=input_layer_norm,
            foot_height_output=foot_height_output,
        )
    if model_type == "tcn":
        return GRFMTCN(
            input_size,
            hidden_size,
            layers,
            dropout,
            output_size,
            kernel_size=tcn_kernel_size,
            input_layer_norm=input_layer_norm,
            foot_height_output=foot_height_output,
        )
    if model_type == "transformer":
        return GRFMTransformer(
            input_size,
            window_samples,
            hidden_size,
            layers,
            dropout,
            output_size,
            heads=transformer_heads,
            feedforward_size=transformer_feedforward_size,
            input_layer_norm=input_layer_norm,
            foot_height_output=foot_height_output,
        )
    return GRFMLSTM(
        input_size,
        hidden_size,
        layers,
        dropout,
        output_size,
        bidirectional=model_type == "bilstm",
        input_layer_norm=input_layer_norm,
        foot_height_output=foot_height_output,
    )


# ``--vertical-share`` (#27): the two outputs that become total + share.
VERTICAL_TARGETS = ("Fz_L", "Fz_R")
# ``--vertical-loss decomposed`` (#29): the two dedicated output units, after the 12 targets, and
# the two GradNorm tasks that replace the Fz_L, Fz_R terms (in their slots).
VERTICAL_HEADS = ("vertical_total_delta", "vertical_share_offset")
VERTICAL_TASKS = ("vertical_total", "vertical_share")
# Share term: only where the measured total exceeds this (N); below it the split is meaningless.
SHARE_MIN_TOTAL_N = 50.0


def straight_through_clamp(value: torch.Tensor, low: float | None, high: float | None) -> torch.Tensor:
    """``clamp(value, low, high)`` in the forward pass, identity gradient in the backward pass.

    The forward value is exactly the clamped one (exact zeros stay exact), but a
    saturated sample still passes the loss gradient to ``value``, so a wrong
    prediction stuck at the bound can be corrected.
    """
    return value + (torch.clamp(value, low, high) - value).detach()


class VerticalShareModel(nn.Module):
    """Total + left/right share parametrisation of the vertical force (``--vertical-share``).

    Wraps any GRFM model. Its two raw vertical outputs ``a_L``, ``a_R`` (the
    ``Fz_L``, ``Fz_R`` columns) become, in N::

        total = BW * max(1 + (a_L + a_R) / 2, 0)     i.e. BW * (1 + delta), delta >= -1
        s     = clamp(0.5 + (a_L - a_R) / 2, 0, 1)
        (both clamps straight-through: exact in the forward pass, identity
        gradient in the backward pass)
        Fz_L  = s * total,  Fz_R = (1 - s) * total

    so Fz_L + Fz_R = total for every sample and s is in [0, 1]. ``BW`` is the
    ``body_weight_N`` input column (``--body-weight-input``, de-standardised)
    of the window's last sample, so an untrained model predicts about BW
    shared equally. The clamps let one foot take exactly no load (Airborne,
    all the load on the other foot) and the total reach 0 (both feet
    Airborne); a sigmoid share would only approach 0. A plain clamp would also
    stop the gradient of every saturated sample, wrong ones included; the
    straight-through clamp keeps it. ``saturation`` holds the last batch's
    (total saturated, share saturated) boolean masks. Swapping a_L and a_R
    swaps Fz_L and Fz_R (mirror-consistent).
    Fz_L, Fz_R are re-standardised with the train target statistics, so the 12
    outputs keep their meaning: the per-target loss, GradNorm, predictions and
    evaluation are unchanged. Shear and moments pass through untouched.

    With ``decomposed=True`` (``--vertical-loss decomposed``, #29) ``delta`` and
    the share come from two dedicated output units (``VERTICAL_HEADS``) that the
    base model emits after its 12 target columns: total = BW * max(1 + delta, 0),
    s = clamp(0.5 + u, 0, 1). The base's own Fz_L, Fz_R columns are then unused
    (overwritten, no gradient). ``vertical`` holds the last forward pass's
    (total N, share) for the decomposed loss.

    The wrapped model's other attributes (``last_shared_layer``,
    ``foot_height_head``, ...) are reached through this module.
    """

    def __init__(
        self, base: nn.Module, feature_names: list[str], targets: list[str], decomposed: bool = False
    ) -> None:
        super().__init__()
        self.base = base
        self.decomposed = decomposed
        self.target_count = len(targets)
        self.body_weight_column = list(feature_names).index(BODY_WEIGHT_FEATURE)
        self.vertical_indexes = [list(targets).index(target) for target in VERTICAL_TARGETS]
        self.register_buffer("body_weight_mean", torch.zeros(()))
        self.register_buffer("body_weight_std", torch.ones(()))
        self.register_buffer("vertical_mean", torch.zeros(len(VERTICAL_TARGETS)))
        self.register_buffer("vertical_std", torch.ones(len(VERTICAL_TARGETS)))
        self.saturation: tuple[torch.Tensor, torch.Tensor] | None = None
        self.vertical: tuple[torch.Tensor, torch.Tensor] | None = None

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "base":
                raise
            return getattr(super().__getattr__("base"), name)

    def set_vertical_share_statistics(self, x_mean, x_std, y_mean, y_std) -> None:
        """Train statistics of the body-weight input and of Fz_L, Fz_R."""
        column, indexes = self.body_weight_column, self.vertical_indexes
        self.body_weight_mean.fill_(float(x_mean[column]))
        self.body_weight_std.fill_(float(x_std[column]))
        self.vertical_mean.copy_(torch.as_tensor(np.asarray(y_mean)[indexes], dtype=torch.float32))
        self.vertical_std.copy_(torch.as_tensor(np.asarray(y_std)[indexes], dtype=torch.float32))

    def total_and_share(self, x: torch.Tensor, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted total vertical force (N) and left share of a batch of raw outputs."""
        body_weight = x[:, -1, self.body_weight_column] * self.body_weight_std + self.body_weight_mean
        if self.decomposed:
            delta, offset = raw[:, self.target_count], raw[:, self.target_count + 1]
            relative_total, raw_share = 1.0 + delta, 0.5 + offset
        else:
            left, right = raw[:, self.vertical_indexes[0]], raw[:, self.vertical_indexes[1]]
            relative_total = 1.0 + 0.5 * (left + right)
            raw_share = 0.5 + 0.5 * (left - right)
        self.saturation = ((relative_total < 0.0).detach(), ((raw_share < 0.0) | (raw_share > 1.0)).detach())
        total = body_weight * straight_through_clamp(relative_total, 0.0, None)
        share = straight_through_clamp(raw_share, 0.0, 1.0)
        self.vertical = (total, share)
        return total, share

    def reparametrise(self, x: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        total, share = self.total_and_share(x, raw)
        vertical = torch.stack((share * total, (1.0 - share) * total), dim=1)
        output = raw[:, : self.target_count].clone()
        output[:, self.vertical_indexes] = (vertical - self.vertical_mean) / self.vertical_std
        return output

    def physical_vertical(self, standardised: torch.Tensor) -> torch.Tensor:
        """(B, 2) Fz_L, Fz_R in N from standardised outputs or targets."""
        return standardised[:, self.vertical_indexes] * self.vertical_std + self.vertical_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.reparametrise(x, self.base(x))

    def forward_with_foot_height(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw, height = self.base.forward_with_foot_height(x)
        return self.reparametrise(x, raw), height

    def vertical_total_and_share(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted total (N) and left share for input windows ``x``."""
        return self.total_and_share(x, self.base(x))


class VerticalSaturation:
    """Counts the epoch's samples whose predicted total or share sits on its clamp.

    Every sample of the batches enters, whatever the foot state; models without
    the total/share parametrisation (``saturation`` absent) count nothing.
    """

    def __init__(self) -> None:
        self.samples, self.total_saturated, self.share_saturated = 0, 0, 0

    def count_saturation(self, model) -> None:
        saturation = getattr(model, "saturation", None)
        if saturation is not None:
            total_saturated, share_saturated = saturation
            self.samples += len(total_saturated)
            self.total_saturated += int(total_saturated.sum())
            self.share_saturated += int(share_saturated.sum())

    @property
    def saturated_fractions(self) -> tuple[float, float]:
        """(total, share) fractions of the epoch's samples on the clamp; NaN without samples."""
        if not self.samples:
            return math.nan, math.nan
        return self.total_saturated / self.samples, self.share_saturated / self.samples


class VerticalSumLoss(VerticalSaturation):
    """Loss on Fz_L + Fz_R against the measured sum.

    Used with ``--vertical-share`` (#27) and, on a model without the
    parametrisation, with ``--vertical-loss sum`` (#29 diagnostic).
    Only samples where both feet are Measured enter it. Squared error of the
    sum in N divided by sqrt(std(Fz_L)^2 + std(Fz_R)^2) of the train targets
    (the per-target terms' scale). It is added to the training objective with
    ``weight``, outside GradNorm, and never enters the selection loss.
    ``value`` is NaN when the epoch had no such sample.
    ``indexes``: output columns of Fz_L, Fz_R; ``mean``, ``std``: their train
    target statistics (N).
    """

    def __init__(self, weight: float, indexes, mean, std) -> None:
        super().__init__()
        self.weight = weight
        self.indexes = list(indexes)
        self.mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        self.std = torch.as_tensor(np.asarray(std), dtype=torch.float32)
        self.squared_sum, self.count = 0.0, 0.0

    def term(self, model, prediction, y, weight, both_measured) -> torch.Tensor | None:
        self.count_saturation(model)
        indexes = self.indexes
        mean, std = self.mean.to(prediction.device), self.std.to(prediction.device)
        selected = both_measured & (weight[:, indexes] > 0.0).all(dim=1)
        scale = torch.sqrt(torch.square(std).sum())
        predicted_sum = (prediction[:, indexes] * std + mean).sum(dim=1)
        true_sum = (y[:, indexes] * std + mean).sum(dim=1)
        error = (predicted_sum - true_sum) / scale
        numerator = (torch.square(error) * selected).sum()
        count = float(selected.sum())
        self.squared_sum += float(numerator.detach())
        self.count += count
        if count == 0.0 or self.weight <= 0.0:
            return None
        return self.weight * numerator / count

    @property
    def value(self) -> float:
        return self.squared_sum / self.count if self.count else math.nan


def decomposed_vertical_scales(
    acquisitions: list[Acquisition],
    targets: list[str],
    target_indexes: list[int],
    zero_weight: float,
) -> dict[str, float]:
    """Normalisations (N) of the decomposed vertical loss, from the train trials only.

    * ``sigma_sum``: std of the measured total Fz_L* + Fz_R* over the samples
      where both feet have a valid vertical target (Measured, or Airborne with a
      non-zero ``--zero-target-weight``), weighted like the total term;
    * ``sigma_share``: std of (Fz_L* - Fz_R*) / 2 = (s* - 0.5) * total* over the
      samples of the share term (both feet valid, total* > ``SHARE_MIN_TOTAL_N``):
      the spread of the left/right split in N, the scale of the share term's
      error (s - s*) * total*.

    A scale below 1e-8 N (or without samples) is 1.
    """
    columns = [list(targets).index(target) for target in VERTICAL_TARGETS]
    totals, total_weights, halves, half_weights = [], [], [], []
    for item in acquisitions:
        feet = []
        for column in columns:
            state = item.state_l if target_indexes[column] < 6 else item.state_r
            weights = item.mask[:, column].astype(np.float64)
            weights[state == STATE_AIRBORNE] *= zero_weight
            feet.append(weights)
        both = np.minimum(*feet)
        left, right = (item.y[:, column].astype(np.float64) for column in columns)
        total = left + right
        valid = both > 0.0
        totals.append(total[valid])
        total_weights.append(both[valid])
        shared = valid & (total > SHARE_MIN_TOTAL_N)
        halves.append(0.5 * (left - right)[shared])
        half_weights.append(both[shared])

    def weighted_std(values, weights) -> float:
        values, weights = np.concatenate(values), np.concatenate(weights)
        if not len(values) or weights.sum() <= 0.0:
            return 1.0
        mean = np.average(values, weights=weights)
        deviation = float(np.sqrt(np.average(np.square(values - mean), weights=weights)))
        return deviation if deviation >= 1e-8 else 1.0

    return {
        "sigma_sum": weighted_std(totals, total_weights),
        "sigma_share": weighted_std(halves, half_weights),
    }


class DecomposedVerticalLoss(VerticalSaturation):
    """Total + share vertical loss of ``--vertical-loss decomposed`` (#29).

    Replaces the two per-foot Fz terms of the masked loss by two orthogonal
    terms on the model's dedicated heads (``VerticalShareModel(decomposed=True)``):

    * total (Fz_L slot, GradNorm task ``vertical_total``):
      (total - total*)^2 / sigma_sum^2 where both feet have a valid vertical
      target (Measured or Airborne), total* = Fz_L* + Fz_R*;
    * share (Fz_R slot, task ``vertical_share``):
      (s - s*)^2 * total*^2 / sigma_share^2, s* = Fz_L* / total*, on the same
      samples when total* > ``SHARE_MIN_TOTAL_N``. With one foot Airborne and
      the other Measured, s* is 0 or 1 and total* the Measured foot's Fz, so
      both terms apply; two Airborne feet (total* = 0) get the total term only.

    total* and s* come from the targets only (constants), so the total term
    reaches only the total head and the share term only the share head.
    Where exactly one foot is Masked, the valid foot's ordinary per-foot term
    (standardised squared error) goes into the total task; that term, Fz = s *
    total or (1 - s) * total, reaches both heads.
    A sample's weight is the smaller of the two feet's target weights (Airborne
    weight = ``--zero-target-weight``). The other 10 terms are unchanged, so
    GradNorm balances 12 tasks: 10 shear/moment + total + share.
    ``indexes``, ``mean``, ``std``: output columns of Fz_L, Fz_R and their train
    target statistics (N); ``sigma_sum``, ``sigma_share`` from
    ``decomposed_vertical_scales``.
    """

    def __init__(self, indexes, mean, std, sigma_sum: float, sigma_share: float) -> None:
        super().__init__()
        self.indexes = list(indexes)
        self.mean = torch.as_tensor(np.asarray(mean), dtype=torch.float32)
        self.std = torch.as_tensor(np.asarray(std), dtype=torch.float32)
        self.sigma_sum, self.sigma_share = float(sigma_sum), float(sigma_share)
        self.total_sum, self.total_count, self.share_sum, self.share_count = 0.0, 0.0, 0.0, 0.0

    def terms(self, model, prediction, y, weight) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-task (numerators, denominators), the Fz_L / Fz_R slots holding total / share.

        ``model.vertical`` must hold the (total N, share) of the forward pass
        that gave ``prediction``.
        """
        self.count_saturation(model)
        numerators = (torch.square(prediction - y) * weight).sum(dim=0)
        denominators = weight.sum(dim=0)
        total, share = model.vertical
        left, right = self.indexes
        mean, std = self.mean.to(y.device), self.std.to(y.device)
        truth = (y[:, self.indexes] * std + mean).detach()
        feet = weight[:, self.indexes]
        valid = feet > 0.0
        both = valid.all(dim=1)
        one = (valid.any(dim=1) & ~both).unsqueeze(1)
        zeros = torch.zeros_like(feet[:, 0])
        sample_weight = torch.where(both, feet.min(dim=1).values, zeros)
        true_total = truth.sum(dim=1)
        total_error = torch.square(total - true_total) / self.sigma_sum**2
        shared = both & (true_total > SHARE_MIN_TOTAL_N)
        share_weight = torch.where(shared, sample_weight, zeros)
        true_share = torch.where(shared, truth[:, 0] / true_total.clamp(min=SHARE_MIN_TOTAL_N), 0.5)
        share_error = torch.square(share - true_share) * torch.square(true_total) / self.sigma_share**2
        total_numerator = torch.where(both, total_error * sample_weight, zeros).sum()
        share_numerator = torch.where(shared, share_error * share_weight, zeros).sum()
        single = torch.where(one, torch.square(prediction[:, self.indexes] - y[:, self.indexes]) * feet, 0.0)
        single_weight = torch.where(one, feet, 0.0).sum()
        self.total_sum += float(total_numerator.detach())
        self.total_count += float(sample_weight.sum())
        self.share_sum += float(share_numerator.detach())
        self.share_count += float(share_weight.sum())
        numerators = torch.cat((
            numerators[:left], (total_numerator + single.sum()).reshape(1), numerators[left + 1:]
        ))
        numerators = torch.cat((numerators[:right], share_numerator.reshape(1), numerators[right + 1:]))
        denominators = denominators.clone()
        denominators[left] = sample_weight.sum() + single_weight
        denominators[right] = share_weight.sum()
        return numerators, denominators

    @property
    def total_value(self) -> float:
        """Epoch mean of the total term (both feet valid); NaN without such sample."""
        return self.total_sum / self.total_count if self.total_count else math.nan

    @property
    def share_value(self) -> float:
        """Epoch mean of the share term over its samples; NaN without such sample."""
        return self.share_sum / self.share_count if self.share_count else math.nan


def gradnorm_tasks(targets, vertical_loss: str) -> list[str]:
    """GradNorm task names: the targets, with total / share in the Fz_L / Fz_R slots
    under ``--vertical-loss decomposed``."""
    if vertical_loss != "decomposed":
        return list(targets)
    replaced = dict(zip(VERTICAL_TARGETS, VERTICAL_TASKS))
    return [replaced.get(target, target) for target in targets]


def task_terms(model, prediction, y, weight, decomposed: DecomposedVerticalLoss | None):
    """Per-task numerators and denominators of the training objective."""
    if decomposed is not None:
        return decomposed.terms(model, prediction, y, weight)
    return (torch.square(prediction - y) * weight).sum(dim=0), weight.sum(dim=0)


def vertical_sum_term(vertical_sum, model, loader, batch, prediction, y, weight, device):
    """Batch term of ``vertical_sum`` (None without the option or without both-Measured samples)."""
    if vertical_sum is None:
        return None
    both = loader.dataset.both_feet_measured(batch[3], batch[4]).to(device)
    return vertical_sum.term(model, prediction, y, weight, both)


def add_terms(loss, *terms):
    for term in terms:
        if term is not None:
            loss = term if loss is None else loss + term
    return loss


class GradNormBalancer:
    """GradNorm (Chen et al. 2018) sur les termes par cible de la loss masquee.

    La loss d'entrainement devient ``sum_i w_i * L_i`` ou ``L_i`` est le terme
    de la cible i dans la loss masquee (meme denominateur commun), de sorte
    que des poids unitaires redonnent exactement la loss d'origine. Les
    normes de gradient sont mesurees sur le poids de la derniere couche
    partagee, declaree par chaque modele via ``model.last_shared_layer()``.
    Apres chaque pas, les poids sont renormalises pour sommer au nombre de
    cibles.
    """

    def __init__(
        self,
        model: nn.Module,
        task_count: int,
        alpha: float = 1.5,
        learning_rate: float = 0.025,
    ) -> None:
        declare = getattr(model, "last_shared_layer", None)
        if declare is None:
            raise ValueError(
                f"{type(model).__name__} ne declare pas last_shared_layer() pour GradNorm"
            )
        self.shared_parameter = declare().weight
        device = self.shared_parameter.device
        self.task_count = task_count
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(task_count, device=device))
        self.optimizer = torch.optim.Adam([self.weights], lr=learning_rate)
        self.initial_losses = torch.full((task_count,), math.nan, device=device)

    def current_weights(self) -> list[float]:
        return [float(value) for value in self.weights.detach().cpu()]

    def training_loss(
        self,
        task_losses: torch.Tensor,
        task_mean_losses: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        """Met a jour les poids GradNorm et renvoie la loss ponderee a retropropager.

        ``task_losses``: termes L_i de la loss masquee (denominateur commun).
        ``task_mean_losses``: MSE par cible (denominateur propre), pour les
        ratios de loss L_i(t)/L_i(0). ``active``: cibles ayant au moins un
        poids non nul dans le batch; les autres gardent leur poids.
        """
        weighted_loss = (self.weights.detach().clone() * task_losses).sum()
        mean_losses = task_mean_losses.detach()
        first = active & torch.isnan(self.initial_losses)
        self.initial_losses[first] = mean_losses[first]
        usable = active & (self.initial_losses > 0)
        if int(usable.sum()) < 2:
            return weighted_loss
        indexes = torch.nonzero(usable).flatten()
        norms = torch.stack([
            self.weights[index]
            * torch.autograd.grad(
                task_losses[index], self.shared_parameter, retain_graph=True
            )[0].norm()
            for index in indexes.tolist()
        ])
        ratios = mean_losses[indexes] / self.initial_losses[indexes]
        target = (norms.mean() * (ratios / ratios.mean()).pow(self.alpha)).detach()
        balance_loss = torch.abs(norms - target).sum()
        self.optimizer.zero_grad(set_to_none=True)
        (self.weights.grad,) = torch.autograd.grad(balance_loss, self.weights)
        self.optimizer.step()
        with torch.no_grad():
            self.weights.clamp_(min=1e-6)
            self.weights.mul_(self.task_count / self.weights.sum())
        return weighted_loss


def model_outputs(model, batch, device):
    """GRFM prediction of a batch and, when the batch carries Foot height and the
    model has the head, ``(predicted, target, weight)`` of the standardised Foot height."""
    x = batch[0].to(device)
    if len(batch) <= 5 or getattr(model, "foot_height_head", None) is None:
        return model(x), None
    prediction, height = model.forward_with_foot_height(x)
    return prediction, (height, batch[5].to(device), batch[6].to(device))


def foot_height_terms(height) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum of the weighted squared Foot height errors and of the weights (both feet)."""
    predicted, target, weight = height
    return (torch.square(predicted - target) * weight).sum(), weight.sum()


class FootHeightLoss:
    """Accumulates the Foot height loss of an epoch; ``value`` is NaN without Foot height."""

    def __init__(self, weight: float) -> None:
        self.weight = weight
        self.squared_sum, self.weight_sum, self.seen = 0.0, 0.0, False

    def term(self, height) -> torch.Tensor | None:
        """Weighted batch term to add to the training loss (None if nothing to learn)."""
        if height is None:
            return None
        numerator, denominator = foot_height_terms(height)
        self.seen = True
        self.squared_sum += float(numerator.detach())
        self.weight_sum += float(denominator.detach())
        if float(denominator.detach()) == 0.0 or self.weight <= 0.0:
            return None
        return self.weight * numerator / denominator

    @property
    def value(self) -> float:
        return self.squared_sum / max(self.weight_sum, 1.0) if self.seen else math.nan


def run_epoch(
    model,
    loader,
    device,
    optimizer=None,
    gradient_clip: float = 1.0,
    balancer: GradNormBalancer | None = None,
    foot_height: FootHeightLoss | None = None,
    vertical_sum: VerticalSumLoss | None = None,
    decomposed: DecomposedVerticalLoss | None = None,
    guard: AnomalousBatchGuard | None = None,
) -> float:
    """One pass; returns the masked GRFM loss only (the selection loss).

    The Foot height term, when the batches carry it, is added to the training
    objective with ``foot_height.weight`` and accumulated in ``foot_height``;
    likewise the vertical sum term (``--vertical-share``, ``--vertical-loss sum``)
    in ``vertical_sum``. Neither enters the returned loss, hence neither early
    stopping nor the scheduler. With ``decomposed`` (``--vertical-loss
    decomposed``) the objective's Fz_L, Fz_R terms are its total and share
    terms; the returned loss stays the per-target masked loss of the 12
    reported outputs, comparable with every other run. A batch rejected by
    ``guard`` takes no optimizer step and is left out of the returned loss.
    """
    training = optimizer is not None
    foot_height = foot_height or FootHeightLoss(0.0)
    if training and balancer is not None:
        return run_gradnorm_epoch(
            model, loader, device, optimizer, gradient_clip, balancer, foot_height, vertical_sum, decomposed,
            guard,
        )
    model.train(training)
    squared_sum, weight_sum = 0.0, 0.0
    for batch in loader:
        y, weight = batch[1].to(device), batch[2].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            prediction, height = model_outputs(model, batch, device)
            numerator = (torch.square(prediction - y) * weight).sum()
            denominator = weight.sum()
            loss = add_terms(
                foot_height.term(height),
                vertical_sum_term(vertical_sum, model, loader, batch, prediction, y, weight, device),
            )
            objective, objective_denominator = numerator, denominator
            if decomposed is not None:
                task_numerators, task_denominators = decomposed.terms(model, prediction, y, weight)
                objective, objective_denominator = task_numerators.sum(), task_denominators.sum()
            if float(objective_denominator.detach()) != 0.0:
                grfm_loss = objective / objective_denominator
                loss = grfm_loss if loss is None else grfm_loss + loss
            if loss is None:
                continue
        if training:
            if guard is not None and guard.rejects(loss):
                continue
            loss.backward()
            if gradient_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
        squared_sum += float(numerator.detach())
        weight_sum += float(denominator.detach())
    return squared_sum / max(weight_sum, 1.0)


def run_gradnorm_epoch(
    model,
    loader,
    device,
    optimizer,
    gradient_clip,
    balancer,
    foot_height: FootHeightLoss | None = None,
    vertical_sum: VerticalSumLoss | None = None,
    decomposed: DecomposedVerticalLoss | None = None,
    guard: AnomalousBatchGuard | None = None,
) -> float:
    """Epoque d'entrainement avec GradNorm; renvoie la loss masquee non ponderee.

    GradNorm equilibre les 12 cibles GRFM seulement (avec ``decomposed``: les 10
    cibles de cisaillement/moment + total + part); les termes Foot height et
    somme verticale (``--vertical-share``, ``--vertical-loss sum``) sont ajoutes
    hors balance, chacun avec son propre poids. Un batch rejete par ``guard``
    (teste sur la loss non ponderee) ne met a jour ni le modele ni les poids GradNorm.
    """
    foot_height = foot_height or FootHeightLoss(0.0)
    model.train(True)
    squared_sum, weight_sum = 0.0, 0.0
    for batch in loader:
        y, weight = batch[1].to(device), batch[2].to(device)
        prediction, height = model_outputs(model, batch, device)
        task_numerators, task_denominators = task_terms(model, prediction, y, weight, decomposed)
        denominator = task_denominators.sum()
        # Returned loss: the per-target masked loss of the 12 reported outputs (as without decomposed).
        squared_error = (torch.square(prediction - y) * weight).detach().sum(dim=0)
        loss = add_terms(
            foot_height.term(height),
            vertical_sum_term(vertical_sum, model, loader, batch, prediction, y, weight, device),
        )
        if guard is not None and guard.rejects(
            add_terms(loss, task_numerators.sum() / denominator if float(denominator.detach()) != 0.0 else None)
        ):
            continue
        if float(denominator.detach()) != 0.0:
            grfm_loss = balancer.training_loss(
                task_numerators / denominator,
                task_numerators / task_denominators.clamp(min=1e-12),
                task_denominators > 0,
            )
            loss = grfm_loss if loss is None else grfm_loss + loss
        if loss is None:
            continue
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if gradient_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
        optimizer.step()
        squared_sum += float(squared_error.sum())
        weight_sum += float(weight.sum(dim=0).sum())
    return squared_sum / max(weight_sum, 1.0)


def physical_normalized_mse(
    model,
    dataset,
    loader,
    y_mean,
    y_std,
    physical_y_std,
    device,
) -> float:
    """Comparable validation score after conversion back to physical units."""
    model.eval()
    squared_sum, weight_sum = 0.0, 0.0
    scale_tensor = torch.as_tensor(physical_y_std, device=device)
    mean_tensor = torch.as_tensor(y_mean, device=device)
    std_tensor = torch.as_tensor(y_std, device=device)
    with torch.no_grad():
        for batch in loader:
            x, y, weight, acquisition_indexes = (
                batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3]
            )
            prediction = model(x) * std_tensor + mean_tensor
            truth = y * std_tensor + mean_tensor
            if dataset.target_scaling == "subject_load":
                loads = torch.as_tensor(
                    [
                        dataset.acquisitions[int(index)].load_estimate_n
                        for index in acquisition_indexes
                    ],
                    dtype=prediction.dtype,
                    device=device,
                ).unsqueeze(1)
                prediction = prediction * loads
                truth = truth * loads
            numerator = (
                torch.square((prediction - truth) / scale_tensor) * weight
            ).sum()
            squared_sum += float(numerator)
            weight_sum += float(weight.sum())
    return squared_sum / max(weight_sum, 1.0)


def predict(
    model,
    dataset,
    loader,
    y_mean,
    y_std,
    device,
    source_target_indexes,
    hard_airborne_zero,
):
    """Test predictions in physical units.

    The ninth element is None, or with the Foot height head ``(truth, predicted,
    valid)`` of the Foot height in mm, ``(N, 2)`` for L, R, truth NaN where invalid.
    The last is None, or with ``--vertical-share`` ``(total_N, share_L)``, ``(N,)`` each.
    """
    model.eval()
    vertical_share = isinstance(model, VerticalShareModel)
    totals, shares = [], []
    truth, predicted, mask, times = [], [], [], []
    state_l, state_r, airborne_l, airborne_r = [], [], [], []
    height_truth, height_predicted, height_valid = [], [], []
    with torch.no_grad():
        for batch in loader:
            y, acquisition_indexes, row_indexes = batch[1], batch[3], batch[4]
            prediction, height = model_outputs(model, batch, device)
            if vertical_share:
                total, share = model.vertical_total_and_share(batch[0].to(device))
                totals.append(total.cpu().numpy())
                shares.append(share.cpu().numpy())
            output = prediction.cpu().numpy() * y_std + y_mean
            batch_truth = y.numpy() * y_std + y_mean
            if height is not None:
                height_predicted.append(
                    height[0].cpu().numpy() * dataset.foot_height_std + dataset.foot_height_mean
                )
            batch_state_l, batch_state_r = [], []
            batch_airborne_l, batch_airborne_r = [], []
            for acquisition_index, target_index in zip(acquisition_indexes, row_indexes):
                acquisition = dataset.acquisitions[int(acquisition_index)]
                row = int(target_index)
                if dataset.target_scaling == "subject_load":
                    output[len(batch_state_l)] *= acquisition.load_estimate_n
                    batch_truth[len(batch_state_l)] *= acquisition.load_estimate_n
                mask.append(acquisition.mask[row])
                times.append(acquisition.time_s[row])
                batch_state_l.append(acquisition.state_l[row])
                batch_state_r.append(acquisition.state_r[row])
                batch_airborne_l.append(acquisition.airborne_l[row])
                batch_airborne_r.append(acquisition.airborne_r[row])
                if height is not None:
                    height_truth.append(acquisition.foot_height[row])
                    height_valid.append(acquisition.foot_height_valid[row])
            truth.append(batch_truth)
            batch_state_l = np.asarray(batch_state_l)
            batch_state_r = np.asarray(batch_state_r)
            batch_airborne_l = np.asarray(batch_airborne_l, dtype=bool)
            batch_airborne_r = np.asarray(batch_airborne_r, dtype=bool)
            if hard_airborne_zero:
                for output_index, source_index in enumerate(source_target_indexes):
                    airborne = batch_airborne_l if source_index < 6 else batch_airborne_r
                    output[airborne, output_index] = 0.0
            predicted.append(output)
            state_l.extend(batch_state_l.tolist())
            state_r.extend(batch_state_r.tolist())
            airborne_l.extend(batch_airborne_l.tolist())
            airborne_r.extend(batch_airborne_r.tolist())
    return (
        np.vstack(truth), np.vstack(predicted), np.vstack(mask), np.asarray(times),
        np.asarray(state_l), np.asarray(state_r), np.asarray(airborne_l), np.asarray(airborne_r),
        (
            (
                np.where(np.vstack(height_valid), np.vstack(height_truth), np.nan),
                np.vstack(height_predicted),
                np.vstack(height_valid),
            )
            if height_valid
            else None
        ),
        (np.concatenate(totals), np.concatenate(shares)) if vertical_share else None,
    )


PREDICTION_META_FIELDS = (
    "test_batch", "test_group", "task_family", "time_s", "state_L", "state_R",
    "airborne_label_L", "airborne_label_R",
)


def prediction_csv_fields(targets, foot_height: bool, vertical_share: bool, extra=()) -> list[str]:
    """Columns of ``predictions.csv`` (``extra`` columns last)."""
    fields = list(PREDICTION_META_FIELDS)
    for target in targets:
        fields.extend((target, f"pred_{target}", f"mask_{target}"))
    if foot_height:
        for side in FEET:
            fields.extend((f"foot_height_mm_{side}", f"pred_foot_height_mm_{side}", f"foot_height_valid_{side}"))
    if vertical_share:
        fields.extend(("pred_vertical_total_N", "pred_vertical_share_L"))
    return fields + list(extra)


def prediction_rows(test_item, test_group: str, targets, outputs, extra: dict | None = None):
    """``predictions.csv`` rows of one test trial from the ``predict`` outputs (``extra`` on every row)."""
    truth, predicted, mask, times, state_l, state_r, airborne_l, airborne_r, heights, vertical = outputs
    for row_index in range(len(truth)):
        row = {
            "test_batch": test_item.name,
            # Participant test de l'essai (= le pli en CV complete).
            "test_group": test_group,
            "task_family": test_item.task_family,
            "time_s": times[row_index],
            "state_L": int(state_l[row_index]),
            "state_R": int(state_r[row_index]),
            "airborne_label_L": int(airborne_l[row_index]),
            "airborne_label_R": int(airborne_r[row_index]),
        }
        for index, target in enumerate(targets):
            row[target] = truth[row_index, index] if mask[row_index, index] else math.nan
            row[f"pred_{target}"] = predicted[row_index, index]
            row[f"mask_{target}"] = int(mask[row_index, index])
        if heights is not None:
            height_truth, height_predicted, height_valid = heights
            for column, side in enumerate(FEET):
                row[f"foot_height_mm_{side}"] = height_truth[row_index, column]
                row[f"pred_foot_height_mm_{side}"] = height_predicted[row_index, column]
                row[f"foot_height_valid_{side}"] = int(height_valid[row_index, column])
        if vertical is not None:
            row["pred_vertical_total_N"] = vertical[0][row_index]
            row["pred_vertical_share_L"] = vertical[1][row_index]
        yield {**row, **(extra or {})}


def metric_rows(
    truth,
    prediction,
    mask,
    state_l,
    state_r,
    model,
    batch,
    task_family,
    targets,
    target_indexes,
):
    rows = []
    subsets = {
        "all_valid": lambda state: state != STATE_UNKNOWN,
        "platform": lambda state: state == STATE_PLATFORM,
        "airborne_zero": lambda state: state == STATE_AIRBORNE,
    }
    for output_index, (target, source_index) in enumerate(zip(targets, target_indexes)):
        state = state_l if source_index < 6 else state_r
        for subset, state_filter in subsets.items():
            selected = mask[:, output_index] & state_filter(state)
            if not selected.any():
                continue
            reference = truth[selected, output_index]
            estimate = prediction[selected, output_index]
            error = reference - estimate
            ss_res = float(np.square(error).sum())
            ss_tot = float(np.square(reference - reference.mean()).sum())
            correlation = (
                float(np.corrcoef(reference, estimate)[0, 1])
                if len(reference) > 1 and np.std(reference) > 0 and np.std(estimate) > 0
                else math.nan
            )
            rows.append(
                {
                    "model": model,
                    "test_batch": batch,
                    "task_family": task_family,
                    "subset": subset,
                    "target": target,
                    "n": int(len(reference)),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "mae": float(np.mean(np.abs(error))),
                    "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan,
                    "correlation": correlation,
                }
            )
    return rows


def save_plot(
    path: Path, batch: str, time_s, truth, prediction, mask, targets, model_label: str
):
    rows = math.ceil(len(targets) / 2)
    figure, axes = plt.subplots(rows, 2, figsize=(16, max(4, 3.2 * rows)), sharex=True, squeeze=False)
    for index, (axis, target) in enumerate(zip(axes.ravel(), targets)):
        reference = truth[:, index].copy()
        estimate = prediction[:, index].copy()
        reference[~mask[:, index]] = np.nan
        estimate[~mask[:, index]] = np.nan
        axis.plot(time_s, reference, label="reference valide", linewidth=1.0)
        axis.plot(time_s, estimate, label=model_label, linewidth=0.9, alpha=0.85)
        axis.set_title(target)
        axis.grid(alpha=0.25)
        axis.legend()
    for axis in axes.ravel()[len(targets):]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("Temps (s)")
    figure.suptitle(f"{model_label} masque — test {batch}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def model_from_args(
    args,
    input_size: int,
    output_size: int,
    dropout: float | None = None,
    feature_names: list[str] | None = None,
    targets: list[str] | None = None,
) -> nn.Module:
    """Construit le modele de ``--model-type`` a partir des options CLI.

    Avec ``--vertical-share``, le modele est enveloppe dans ``VerticalShareModel``
    (il faut alors ``feature_names`` avec ``body_weight_N`` et ``targets``).
    Avec ``--vertical-loss decomposed``, le modele de base sort en plus les deux
    unites dediees ``VERTICAL_HEADS`` (``output_size + 2``).
    """
    decomposed = getattr(args, "vertical_loss", "default") == "decomposed"
    model = build_model(
        args.model_type,
        input_size,
        args.window_samples,
        args.hidden_size,
        args.layers,
        args.dropout if dropout is None else dropout,
        output_size + len(VERTICAL_HEADS) if decomposed else output_size,
        tcn_kernel_size=args.tcn_kernel_size,
        transformer_heads=args.transformer_heads,
        transformer_feedforward_size=args.transformer_feedforward_size,
        feature_names=feature_names,
        token_dropout=args.token_dropout,
        input_layer_norm=args.input_layer_norm,
        foot_height_output=getattr(args, "foot_height_output", False),
        patch_samples=getattr(args, "patch_samples", 5),
        decoder_layers=getattr(args, "decoder_layers", 1),
    )
    if getattr(args, "vertical_share", False):
        model = VerticalShareModel(model, feature_names, targets, decomposed=decomposed)
    return model


def fit_model_statistics(
    model, train, x_mean, x_std, y_mean, y_std, foot_height_mean=None, foot_height_std=None
) -> None:
    """Statistiques internes des modeles qui en ont (sensor_token, total/part), sur le train."""
    if isinstance(model, VerticalShareModel):
        model.set_vertical_share_statistics(x_mean, x_std, y_mean, y_std)
        model = model.base
    if isinstance(model, SensorTokenTransformer):
        model.set_statistics([item.x for item in train], x_mean, x_std, y_mean, y_std)
        if model.foot_height_head is not None:
            model.set_foot_height_statistics(foot_height_mean, foot_height_std)


def add_linear_warmup(optimizer, warmup_steps: int) -> None:
    """Rampe lineaire du learning rate sur les ``warmup_steps`` premiers pas.

    Le pas k (0-indexe) utilise lr * (k + 1) / warmup_steps; ensuite le
    learning rate n'est plus touche, ce qui laisse ReduceLROnPlateau libre.
    """
    if warmup_steps <= 0:
        return
    base_rates = [float(group["lr"]) for group in optimizer.param_groups]
    state = {"step": 0}

    def scale(factor: float) -> None:
        for group, rate in zip(optimizer.param_groups, base_rates):
            group["lr"] = rate * factor

    def before_step(*_):
        if state["step"] < warmup_steps:
            scale((state["step"] + 1) / warmup_steps)
            state["step"] += 1

    scale(1.0 / warmup_steps)
    optimizer.register_step_pre_hook(before_step)


def add_cosine_schedule(optimizer, warmup_steps: int, total_steps: int, min_learning_rate: float) -> None:
    """Warmup lineaire puis decroissance cosinus jusqu'a ``min_learning_rate``, pas par pas.

    Le pas k (0-indexe) utilise lr * (k + 1) / warmup_steps pendant le warmup,
    puis min_lr + (lr - min_lr) * (1 + cos(pi * t)) / 2, t allant de 0 a 1 sur
    les pas restants; au-dela de ``total_steps`` le learning rate reste a min_lr.
    """
    base_rates = [float(group["lr"]) for group in optimizer.param_groups]
    state = {"step": 0}

    def apply(step: int) -> None:
        for group, rate in zip(optimizer.param_groups, base_rates):
            if step < warmup_steps:
                group["lr"] = rate * (step + 1) / warmup_steps
            else:
                progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
                group["lr"] = min_learning_rate + (rate - min_learning_rate) * 0.5 * (1.0 + math.cos(math.pi * progress))

    def before_step(*_):
        apply(state["step"])
        state["step"] += 1

    apply(0)
    optimizer.register_step_pre_hook(before_step)


class WeightAverage:
    """Moyenne exponentielle (EMA) des parametres, mise a jour apres chaque pas d'optimiseur.

    Le decay suit un warmup, min(max_decay, (1 + n) / (10 + n)) a la mise a jour n,
    pour que les premiers pas ne figent pas l'initialisation. ``applied()``
    charge temporairement les poids moyens dans le modele (buffers inchanges).
    """

    def __init__(self, model: nn.Module, max_decay: float) -> None:
        self.parameters = list(model.parameters())
        self.shadow = [parameter.detach().clone() for parameter in self.parameters]
        self.max_decay, self.updates = max_decay, 0

    def attach(self, optimizer) -> None:
        optimizer.register_step_post_hook(lambda *_: self.update())

    @torch.no_grad()
    def update(self) -> None:
        decay = min(self.max_decay, (1 + self.updates) / (10 + self.updates))
        for average, parameter in zip(self.shadow, self.parameters):
            average.lerp_(parameter.detach(), 1.0 - decay)
        self.updates += 1

    @contextmanager
    def applied(self):
        raw = [parameter.detach().clone() for parameter in self.parameters]
        with torch.no_grad():
            for parameter, average in zip(self.parameters, self.shadow):
                parameter.copy_(average)
        try:
            yield
        finally:
            with torch.no_grad():
                for parameter, value in zip(self.parameters, raw):
                    parameter.copy_(value)


class AnomalousBatchGuard:
    """Rejette un batch dont la loss d'entrainement est non finie ou depasse ``threshold``."""

    def __init__(self, threshold: float) -> None:
        self.threshold, self.skipped = threshold, 0

    def rejects(self, loss: torch.Tensor | None) -> bool:
        if loss is None:
            return False
        value = float(loss.detach())
        if math.isfinite(value) and value <= self.threshold:
            return False
        self.skipped += 1
        return True


OVERFIT_TASK_FAMILIES = tuple(sorted(TASK_SETS["continuous"]))
LOAD_CONDITIONS = ("unloaded", "loaded")
OVERFIT_EXIT_FAIL = 3


def load_condition(acquisition: Acquisition) -> str:
    """``P01`` = unloaded, ``P01b`` = loaded (backpack)."""
    return "unloaded" if acquisition.subject_session == acquisition.subject_identity else "loaded"


def disable_dropout(model: nn.Module) -> None:
    """Annule toute forme de dropout du modele (overfit test)."""
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.p = 0.0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0.0
        if isinstance(getattr(module, "token_dropout", None), float):
            module.token_dropout = 0.0


class WindowRecordingLoader:
    """Iterable sur un DataLoader qui note les fenetres vues a chaque passe."""

    def __init__(self, loader: DataLoader) -> None:
        self.loader = loader
        self.seen: list[tuple[int, int]] = []

    def __iter__(self):
        self.seen = []
        for batch in self.loader:
            self.seen.extend(zip(batch[3].tolist(), batch[4].tolist()))
            yield batch

    def __len__(self) -> int:
        return len(self.loader)


def windows_digest(windows) -> str:
    text = ";".join(f"{a}:{t}" for a, t in sorted(windows))
    return hashlib.sha1(text.encode("ascii")).hexdigest()[:16]


def sample_overfit_windows(
    dataset: MaskedWindowDataset, per_stratum: int, seed: int
) -> tuple[list[int], dict[str, int], dict[str, int]]:
    """Tire un ensemble fixe de fenetres, ``per_stratum`` par famille x charge.

    Renvoie les positions dans ``dataset``, le nombre tire et le nombre
    disponible par strate ``famille/charge``.
    """
    strata: dict[str, list[int]] = {
        f"{family}/{condition}": []
        for family in OVERFIT_TASK_FAMILIES
        for condition in LOAD_CONDITIONS
    }
    for position, (acquisition_index, _) in enumerate(dataset.index):
        item = dataset.acquisitions[acquisition_index]
        key = f"{item.task_family}/{load_condition(item)}"
        if key in strata:
            strata[key].append(position)
    rng = np.random.default_rng(seed)
    selected, drawn, available = [], {}, {}
    for key, positions in strata.items():
        count = min(per_stratum, len(positions))
        chosen = rng.choice(len(positions), size=count, replace=False) if count else []
        selected.extend(sorted(positions[int(index)] for index in chosen))
        drawn[key], available[key] = count, len(positions)
    return selected, drawn, available


OVERFIT_FINAL_WINDOW = 10


def final_overfit_nmse(history: list[np.ndarray], window: int = OVERFIT_FINAL_WINDOW) -> np.ndarray:
    """Per-target median NMSE over the last ``window`` epochs: the overfit verdict's final NMSE.

    A median, so one oscillating epoch at constant learning rate does not decide
    the verdict, while a target that stays above the threshold still fails it.
    """
    return np.median(np.stack(history[-window:]), axis=0)


def per_target_nmse(model, loader, device, task_count: int) -> np.ndarray:
    """NMSE par cible (MSE masquee / variance ponderee de la reference)."""
    model.eval()
    squared = torch.zeros(task_count, dtype=torch.float64)
    weights = torch.zeros(task_count, dtype=torch.float64)
    sums = torch.zeros(task_count, dtype=torch.float64)
    sums_sq = torch.zeros(task_count, dtype=torch.float64)
    with torch.no_grad():
        for batch in loader:
            x, y, weight = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            prediction = model(x)
            squared += (torch.square(prediction - y) * weight).sum(dim=0).double().cpu()
            weights += weight.sum(dim=0).double().cpu()
            sums += (y * weight).sum(dim=0).double().cpu()
            sums_sq += (torch.square(y) * weight).sum(dim=0).double().cpu()
    with np.errstate(invalid="ignore", divide="ignore"):
        w = weights.numpy()
        mean = sums.numpy() / w
        variance = np.maximum(sums_sq.numpy() / w - np.square(mean), 1e-12)
        nmse = squared.numpy() / w / variance
    nmse[w <= 0] = np.nan
    return nmse


def run_overfit_test(
    args, acquisitions, targets, target_indexes, fold, device, feature_names=None
) -> int:
    """Overfit test: apprend par coeur un ensemble fixe de fenetres stratifiees.

    Ecrit dans ``--output-dir``: ``overfit_windows.csv``, ``overfit_curves.csv``,
    ``gradnorm_weights.csv`` et ``overfit_verdict.json``. Renvoie 0 si le
    verdict est positif, ``OVERFIT_EXIT_FAIL`` sinon.
    """
    started = time.monotonic()
    output_dir = args.output_dir
    train_groups = set(fold["train_groups"])
    pool = [
        item for item in acquisitions
        if split_group(item, args.split_level) in train_groups
        and item.task_family in OVERFIT_TASK_FAMILIES
    ]
    verdict = {
        "schema_version": 1,
        "mode": "overfit_test",
        "passed": False,
        "cause": None,
        "detail": "",
        "failing_targets": [],
        "final_nmse": {target: None for target in targets},
        "nmse_threshold": args.overfit_nmse_threshold,
        "divergence_factor": args.overfit_divergence_factor,
        "initial_mean_nmse": None,
        "final_mean_nmse": None,
        "epochs_requested": args.overfit_epochs,
        "epochs_completed": 0,
        "stopped_epoch": None,
        "model_type": args.model_type,
        "targets": list(targets),
        "fold": {"test_group": fold["test_group"], "train_groups": fold["train_groups"]},
        "windows": 0,
        "windows_per_stratum": args.overfit_windows_per_stratum,
        "strata": {},
        "strata_available": {},
        "windows_digest": None,
        "final_gradnorm_weights": None,
        "optimizer": {
            "name": "AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": 0.0,
            "warmup_steps": args.warmup_steps,
            "gradient_clip": args.gradient_clip,
            "batch_size": args.batch_size,
            "lr_scheduler": "none",
        },
        "dropout": 0.0,
        "gradnorm": {"alpha": args.gradnorm_alpha, "learning_rate": args.gradnorm_learning_rate},
        "seed": args.seed,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "artifacts": {
            "windows": "overfit_windows.csv",
            "curves": "overfit_curves.csv",
            "gradnorm_weights": "gradnorm_weights.csv",
        },
        "elapsed_s": None,
    }
    verdict_path = output_dir / "overfit_verdict.json"

    def finish(cause: str | None, detail: str) -> int:
        verdict["passed"] = cause is None
        verdict["cause"] = cause
        verdict["detail"] = detail
        verdict["elapsed_s"] = round(time.monotonic() - started, 3)
        verdict_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"overfit_test {'PASS' if cause is None else 'FAIL'}: {detail}")
        print(f"Verdict: {verdict_path.resolve()}")
        return 0 if cause is None else OVERFIT_EXIT_FAIL

    if not pool:
        return finish("missing_strata", "aucun essai continu chez les participants d'entrainement")
    x_mean, x_std, y_mean, y_std = statistics(
        pool, target_indexes, args.zero_target_weight, target_scaling=args.target_scaling
    )
    dataset = MaskedWindowDataset(
        pool, x_mean, x_std, y_mean, y_std, target_indexes,
        window=args.window_samples, centered=not args.causal,
        zero_weight=args.zero_target_weight, target_scaling=args.target_scaling,
        # Memorisation is checked on the full Sensor configuration.
        sensor_dropout=SensorDropout.from_feature_names(feature_names) if args.sensor_dropout else None,
    )
    selected, drawn, available = sample_overfit_windows(
        dataset, args.overfit_windows_per_stratum, args.seed
    )
    verdict.update(windows=len(selected), strata=drawn, strata_available=available)
    with (output_dir / "overfit_windows.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["stratum", "task_family", "load_condition", "subject", "batch", "target_index", "time_s"])
        for position in selected:
            acquisition_index, target_index = dataset.index[position]
            item = pool[acquisition_index]
            writer.writerow([
                f"{item.task_family}/{load_condition(item)}", item.task_family,
                load_condition(item), item.subject_session, item.name, target_index,
                float(item.time_s[target_index]),
            ])
    missing = [key for key, count in drawn.items() if count == 0]
    if missing:
        return finish("missing_strata", "strates sans fenetre: " + ", ".join(missing))
    verdict["windows_digest"] = windows_digest(dataset.index[position] for position in selected)

    fixed_set = torch.utils.data.Subset(dataset, selected)
    loader_options = dict(num_workers=args.num_workers, pin_memory=device.type == "cuda")
    train_loader = WindowRecordingLoader(DataLoader(
        fixed_set, args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed), **loader_options,
    ))
    evaluation_loader = DataLoader(fixed_set, args.batch_size, shuffle=False, **loader_options)
    model = model_from_args(
        args, pool[0].x.shape[1], len(targets), dropout=0.0, feature_names=feature_names,
        targets=list(targets),
    ).to(device)
    fit_model_statistics(model, pool, x_mean, x_std, y_mean, y_std)
    disable_dropout(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    add_linear_warmup(optimizer, args.warmup_steps)
    balancer = GradNormBalancer(
        model, len(targets), alpha=args.gradnorm_alpha, learning_rate=args.gradnorm_learning_rate
    )
    # --vertical-loss decomposed: the gate trains the same total + share objective as the run.
    decomposed = None
    if getattr(args, "vertical_loss", "default") == "decomposed":
        columns = [list(targets).index(target) for target in VERTICAL_TARGETS]
        decomposed = DecomposedVerticalLoss(
            columns, y_mean[columns], y_std[columns],
            **decomposed_vertical_scales(pool, list(targets), target_indexes, args.zero_target_weight),
        )
    tasks = gradnorm_tasks(targets, getattr(args, "vertical_loss", "default"))

    curve_fields = [
        "epoch", "train_loss", "mean_nmse", "max_nmse", "learning_rate", "windows_seen",
        "windows_digest", *(f"nmse_{target}" for target in targets),
    ]
    curves_path = output_dir / "overfit_curves.csv"
    gradnorm_path = output_dir / "gradnorm_weights.csv"
    gradnorm_fields = ["test_group", "epoch", *tasks]
    with curves_path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=curve_fields).writeheader()
    with gradnorm_path.open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=gradnorm_fields).writeheader()

    def log_epoch(epoch, train_loss, nmse, learning_rate, seen) -> None:
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "mean_nmse": float(np.nanmean(nmse)) if np.isfinite(nmse).any() else math.nan,
            "max_nmse": float(np.nanmax(nmse)) if np.isfinite(nmse).any() else math.nan,
            "learning_rate": learning_rate,
            "windows_seen": len(seen),
            "windows_digest": windows_digest(seen),
            **{f"nmse_{target}": float(value) for target, value in zip(targets, nmse)},
        }
        with curves_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=curve_fields).writerow(row)
        verdict["final_nmse"] = {
            target: (float(value) if np.isfinite(value) else None)
            for target, value in zip(targets, nmse)
        }
        verdict["final_mean_nmse"] = row["mean_nmse"] if np.isfinite(row["mean_nmse"]) else None

    nmse = per_target_nmse(model, evaluation_loader, device, len(targets))
    log_epoch(0, math.nan, nmse, float(optimizer.param_groups[0]["lr"]),
              [dataset.index[position] for position in selected])
    supervised = ~np.isnan(nmse)
    if not np.isfinite(nmse[supervised]).all() or not supervised.any():
        verdict["stopped_epoch"] = 0
        return finish("nan", "NMSE non finie avant entrainement (entrees ou statistiques)")
    initial_mean = float(nmse[supervised].mean())
    verdict["initial_mean_nmse"] = initial_mean
    # Divergence = a target's NMSE above factor x its own untrained NMSE, or the mean
    # above factor x the untrained mean (the per-target test catches one target
    # blowing up while the others still learn).
    divergence_limit = args.overfit_divergence_factor * initial_mean
    target_limits = args.overfit_divergence_factor * nmse
    history = []
    for epoch in range(1, args.overfit_epochs + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        train_loss = run_epoch(
            model, train_loader, device, optimizer,
            gradient_clip=args.gradient_clip, balancer=balancer, decomposed=decomposed,
        )
        nmse = per_target_nmse(model, evaluation_loader, device, len(targets))
        history.append(nmse)
        log_epoch(epoch, train_loss, nmse, learning_rate, train_loader.seen)
        epoch_weights = balancer.current_weights()
        verdict["final_gradnorm_weights"] = dict(zip(tasks, epoch_weights))
        with gradnorm_path.open("a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=gradnorm_fields).writerow(
                {"test_group": fold["test_group"], "epoch": epoch, **dict(zip(tasks, epoch_weights))}
            )
        verdict["epochs_completed"] = epoch
        values = nmse[supervised]
        if (
            not math.isfinite(train_loss)
            or not np.isfinite(values).all()
            or not np.isfinite(epoch_weights).all()
        ):
            verdict["stopped_epoch"] = epoch
            bad = [t for t, v in zip(targets, nmse) if not np.isfinite(v)]
            return finish("nan", f"valeur non finie a l'epoque {epoch}" + (f" ({', '.join(bad)})" if bad else ""))
        mean_nmse = float(values.mean())
        diverging = [
            target for target, value, limit, used in zip(targets, nmse, target_limits, supervised)
            if used and value > limit
        ]
        if diverging or mean_nmse > divergence_limit:
            verdict["stopped_epoch"] = epoch
            verdict["failing_targets"] = diverging
            per_target = ", ".join(
                f"{target}={verdict['final_nmse'][target]:.4g}" for target in diverging
            )
            return finish(
                "divergence",
                f"a l'epoque {epoch}: NMSE moyenne {mean_nmse:.4g} (initiale {initial_mean:.4g}); "
                f"cibles au-dela de {args.overfit_divergence_factor:g} x leur NMSE initiale: "
                f"{per_target or 'aucune'}",
            )
        if epoch % 50 == 0 or epoch == args.overfit_epochs:
            print(f"overfit epoch={epoch:04d} train={train_loss:.6f} mean_nmse={mean_nmse:.6f}")
    # Verdict on the median of the last epochs; ``final_nmse`` stays the last epoch's values.
    final = final_overfit_nmse(history)
    verdict["median_final_nmse"] = {
        target: (float(value) if np.isfinite(value) else None) for target, value in zip(targets, final)
    }
    verdict["median_window_epochs"] = min(OVERFIT_FINAL_WINDOW, len(history))
    failing = [
        target for target, value in zip(targets, final)
        if np.isfinite(value) and value >= args.overfit_nmse_threshold
    ]
    verdict["failing_targets"] = failing
    if failing:
        return finish(
            "nmse_above_threshold",
            f"NMSE finale (mediane des {verdict['median_window_epochs']} dernieres epoques) >= "
            f"{args.overfit_nmse_threshold:g} pour " + ", ".join(
                f"{target}={verdict['median_final_nmse'][target]:.4g}" for target in failing
            ),
        )
    return finish(None, f"toutes les cibles sous NMSE {args.overfit_nmse_threshold:g}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results_lstm_two_feet_masked"))
    parser.add_argument("--window-samples", type=int, default=21)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--hard-airborne-zero",
        action="store_true",
        help=(
            "ancien post-traitement: force les sorties a zero avec le label "
            "airborne hors ligne; ne pas utiliser pour un modele deployable"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument(
        "--model-type",
        choices=("lstm", "bilstm", "mlp", "tcn", "transformer", "patch_transformer", "sensor_token"),
        default="lstm",
        help="architecture temporelle; toutes utilisent la meme fenetre",
    )
    parser.add_argument(
        "--tcn-kernel-size",
        type=int,
        default=3,
        help="taille du noyau des convolutions TCN causales",
    )
    parser.add_argument(
        "--transformer-heads",
        type=int,
        default=4,
        help="nombre de tetes d'attention du Transformer causal",
    )
    parser.add_argument(
        "--transformer-feedforward-size",
        type=int,
        default=128,
        help="largeur de la couche feed-forward du Transformer",
    )
    parser.add_argument(
        "--sensor-dropout",
        action="store_true",
        help=(
            "Sensor dropout: chaque fenetre d'entrainement garde une Sensor configuration tiree au "
            "hasard (taille uniforme 1..12, memes groupes aux deux pieds) et 12 indicateurs de presence"
        ),
    )
    parser.add_argument(
        "--loadsol-heel-imu",
        action="store_true",
        help="avec --input-features loadsol: ajoute l'IMU du Patch heel (equivalent de l'IMU tibiale de Sabbah)",
    )
    parser.add_argument(
        "--sensor-configuration",
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "ne garde que les canaux de ces Sensor groups (ex. heel_gauges forefoot_acc), aux deux "
            "pieds; les autres canaux sont retires des entrees"
        ),
    )
    parser.add_argument(
        "--sensor-full-probability",
        type=float,
        default=SENSOR_FULL_PROBABILITY,
        help="avec --sensor-dropout: part des fenetres gardant la configuration complete",
    )
    parser.add_argument(
        "--sensor-channel-dropout",
        type=float,
        default=SENSOR_CHANNEL_DROPOUT,
        help="avec --sensor-dropout: probabilite de retirer en plus chaque canal garde",
    )
    parser.add_argument(
        "--patch-samples",
        type=int,
        default=5,
        help="patch_transformer: echantillons par token temporel",
    )
    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=1,
        help="patch_transformer: couches du decodeur des Readout tokens",
    )
    parser.add_argument(
        "--token-dropout",
        type=float,
        default=0.1,
        help="sensor_token: fraction de Sensor tokens masques au hasard en entrainement",
    )
    parser.add_argument(
        "--input-layer-norm",
        action="store_true",
        help=(
            "LayerNorm de la premiere representation (auto-correction #9): entrees "
            "par echantillon (mlp, tcn, lstm), embedding (transformer, deja pre-norm), "
            "embeddings des Sensor tokens avant l'attention (sensor_token)"
        ),
    )
    parser.add_argument(
        "--input-features",
        choices=("raw", "gauge_augmented", "loadsol"),
        default="raw",
        help="raw ou jauges absolues + sommes par pied + proportions",
    )
    parser.add_argument(
        "--body-weight-input",
        action="store_true",
        help=(
            "ajoute le poids du Subject (metadata body_weights, N) en entree, "
            "constant par Subject et standardise sur le train"
        ),
    )
    parser.add_argument(
        "--body-weight-masking",
        type=float,
        nargs="?",
        const=0.3,
        default=0.0,
        metavar="P",
        help=(
            "poids du corps optionnel: avec la probabilite P (0.3 sans valeur), tiree par fenetre "
            "d'entrainement, le poids est remplace par la moyenne du train et une entree "
            "'body weight missing' passe a 1 (demande --body-weight-input; 0 = desactive)"
        ),
    )
    parser.add_argument(
        "--body-weight-mode",
        choices=("provided", "missing"),
        default="provided",
        help="poids du corps en validation et en test: fourni, ou absent (valeur par defaut, drapeau a 1)",
    )
    parser.add_argument(
        "--vertical-share",
        action="store_true",
        help=(
            "Fz total + part gauche/droite: total = BW * (1 + delta) appris, part s dans "
            "[0, 1], Fz_L = s * total, Fz_R = (1 - s) * total, plus une loss sur "
            "Fz_L + Fz_R la ou les deux pieds sont Measured, sauf avec --vertical-loss decomposed "
            "(demande --body-weight-input)"
        ),
    )
    parser.add_argument(
        "--vertical-sum-loss-weight",
        type=float,
        default=1.0,
        help="poids de la loss sur Fz_L + Fz_R (avec --vertical-share ou --vertical-loss sum), hors GradNorm",
    )
    parser.add_argument(
        "--vertical-loss",
        choices=("default", "decomposed", "sum"),
        default="default",
        help=(
            "loss de la force verticale. default: termes Fz_L, Fz_R par pied (plus la loss de somme "
            "avec --vertical-share). decomposed (demande --vertical-share): tetes dediees delta_total "
            "et part, termes total et part a la place des termes Fz_L, Fz_R (GradNorm a 12 taches). "
            "sum (diagnostic, sans --vertical-share): ajoute la loss de somme Fz_L + Fz_R a un modele "
            "non reparametrise"
        ),
    )
    parser.add_argument(
        "--target-scaling",
        choices=("standard", "subject_load"),
        default="standard",
        help="normalisation standard ou division prealable par la charge semelle estimee",
    )
    parser.add_argument("--load-calibration-seconds", type=float, default=3.0)
    parser.add_argument("--gauge-proportion-epsilon-n", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "plateau", "cosine"),
        default="none",
        help=(
            "plateau: reduit le learning rate quand la loss de validation stagne; "
            "cosine: decroissance cosinus pas par pas jusqu'a --min-learning-rate sur --epochs"
        ),
    )
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=4)
    parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="norme maximale des gradients; 0 desactive le clipping",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="rampe lineaire du learning rate sur N pas d'optimiseur; 0 desactive",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="weight decay d'AdamW (regularisation)",
    )
    parser.add_argument(
        "--warmup-fraction",
        type=float,
        default=0.0,
        help="avec --lr-scheduler cosine: warmup sur cette fraction des pas totaux (au lieu de --warmup-steps)",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help=(
            "decay maximal de la moyenne exponentielle des poids (avec warmup); a chaque epoque "
            "la selection garde les poids bruts ou moyens, les meilleurs en validation; 0 desactive"
        ),
    )
    parser.add_argument(
        "--anomalous-loss-threshold",
        type=float,
        default=0.0,
        help="saute le pas d'un batch dont la loss est non finie ou au-dessus du seuil; 0 desactive",
    )
    parser.add_argument("--zero-target-weight", type=float, default=0.5)
    parser.add_argument(
        "--selection-subset",
        choices=("valid", "measured"),
        default="valid",
        help=(
            "loss de validation de l'early stopping et du scheduler: valid = Measured + "
            "Airborne (spec #1, v3.6), measured = Measured seulement; Airborne reste "
            "dans la loss d'entrainement"
        ),
    )
    parser.add_argument(
        "--foot-height-output",
        action="store_true",
        help=(
            "sortie auxiliaire Foot height (L, R; NPZ foot_height_mm_L/R, v3.7): loss sur "
            "tout echantillon a Foot height valide, Masked compris, standardisee sur le "
            "train, hors GradNorm et hors loss d'early stopping"
        ),
    )
    parser.add_argument(
        "--foot-height-loss-weight",
        type=float,
        default=1.0,
        help="poids de la loss Foot height ajoutee a la loss GRFM (avec --foot-height-output)",
    )
    parser.add_argument(
        "--overfit-test",
        action="store_true",
        help=(
            "overfit test: ensemble fixe stratifie (familles continues x charge) "
            "des participants train du premier pli, sans dropout ni weight decay, "
            "GradNorm actif; ecrit overfit_verdict.json (code 0 = pass, 3 = fail)"
        ),
    )
    parser.add_argument("--overfit-epochs", type=int, default=1000)
    parser.add_argument("--overfit-windows-per-stratum", type=int, default=256)
    parser.add_argument("--overfit-nmse-threshold", type=float, default=0.05)
    parser.add_argument(
        "--overfit-divergence-factor",
        type=float,
        default=10.0,
        help="divergence si la NMSE d'une cible (ou la moyenne) depasse ce facteur x sa NMSE initiale",
    )
    parser.add_argument(
        "--gradnorm",
        action="store_true",
        help="equilibre les termes par cible de la loss masquee avec GradNorm",
    )
    parser.add_argument(
        "--gradnorm-alpha",
        type=float,
        default=1.5,
        help="force de rappel GradNorm (alpha)",
    )
    parser.add_argument(
        "--gradnorm-learning-rate",
        type=float,
        default=0.025,
        help="taux d'apprentissage (Adam) des poids GradNorm",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-level",
        choices=("evaluation_group", "subject_identity"),
        default="evaluation_group",
        help="niveau independant utilise pour train/validation/test",
    )
    parser.add_argument(
        "--validation-group-count",
        type=int,
        default=1,
        help="nombre de groupes/sujets distincts utilises pour la validation",
    )
    parser.add_argument(
        "--test-groups",
        nargs="+",
        help=(
            "split fixe: un seul pli avec ces participants test (avec --validation-groups, "
            "--split-level subject_identity); sans l'option, CV complete"
        ),
    )
    parser.add_argument(
        "--validation-groups",
        nargs="+",
        help="split fixe: participants de validation (early stopping, scheduler)",
    )
    parser.add_argument(
        "--only-test-batch",
        "--only-test-group",
        dest="only_test_group",
        help="limite l'execution au pli contenant cet essai ou ce groupe",
    )
    parser.add_argument(
        "--dry-run-split",
        action="store_true",
        help="ecrit et affiche les plis sans entrainer de modele",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help=(
            "hyperparametres: conserve seulement les losses train/validation; "
            "ne calcule ni predictions test, ni figures, ni checkpoints"
        ),
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="ne trace pas les figures par essai test (accelere les tests)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument(
        "--task-set",
        choices=("continuous", "continuous_walk", "all"),
        default="all",
        help=(
            "continuous=hulahoop/balance/lateral 2 appuis/squat; "
            "continuous_walk=les memes + walk; all=toutes les taches"
        ),
    )
    args = parser.parse_args(argv)
    if bool(args.test_groups) != bool(args.validation_groups):
        parser.error("--test-groups et --validation-groups vont ensemble (split fixe)")
    if args.test_groups and args.split_level != "subject_identity":
        parser.error("le split fixe demande --split-level subject_identity (participants)")
    if args.window_samples < 2 or args.epochs < 1 or args.batch_size < 1:
        parser.error("window, epochs et batch-size doivent etre positifs")
    if args.hidden_size < 1 or args.layers < 1 or not 0.0 <= args.dropout < 1.0:
        parser.error("hidden-size/layers positifs et dropout dans [0,1[")
    if args.tcn_kernel_size < 2:
        parser.error("tcn-kernel-size doit etre >= 2")
    if args.transformer_heads < 1 or args.transformer_feedforward_size < 1:
        parser.error("les dimensions du Transformer doivent etre positives")
    if not 0.0 <= args.sensor_full_probability <= 1.0 or not 0.0 <= args.sensor_channel_dropout < 1.0:
        parser.error("--sensor-full-probability dans [0, 1], --sensor-channel-dropout dans [0, 1[")
    if not args.sensor_dropout and (
        args.sensor_full_probability != SENSOR_FULL_PROBABILITY
        or args.sensor_channel_dropout != SENSOR_CHANNEL_DROPOUT
    ):
        parser.error("--sensor-full-probability / --sensor-channel-dropout demandent --sensor-dropout")
    if args.loadsol_heel_imu and args.input_features != "loadsol":
        parser.error("--loadsol-heel-imu demande --input-features loadsol")
    if args.input_features == "loadsol" and (
        args.sensor_dropout or args.target_scaling != "standard" or args.model_type == "sensor_token"
    ):
        parser.error("--input-features loadsol: incompatible avec --sensor-dropout, subject_load et sensor_token")
    if args.sensor_configuration:
        if args.sensor_dropout:
            parser.error("--sensor-configuration et --sensor-dropout sont incompatibles")
        if args.input_features != "raw" or args.target_scaling != "standard":
            parser.error("--sensor-configuration demande --input-features raw et --target-scaling standard")
        if args.model_type == "sensor_token":
            parser.error("--sensor-configuration: sensor_token attend les 78 canaux")
    if args.sensor_dropout and args.model_type == "sensor_token":
        parser.error("--sensor-dropout: sensor_token masque deja ses tokens, non supporte")
    if args.patch_samples < 1 or args.decoder_layers < 1:
        parser.error("--patch-samples et --decoder-layers doivent etre positifs")
    if (
        args.model_type in {"transformer", "patch_transformer"}
        and args.hidden_size % args.transformer_heads != 0
    ):
        parser.error("hidden-size doit etre divisible par transformer-heads")
    if args.load_calibration_seconds <= 0 or args.gauge_proportion_epsilon_n <= 0:
        parser.error("les parametres de charge/proportion doivent etre positifs")
    if args.learning_rate <= 0 or args.min_learning_rate <= 0:
        parser.error("les learning rates doivent etre positifs")
    if not 0.0 < args.lr_factor < 1.0 or args.lr_patience < 1:
        parser.error("lr-factor doit etre dans ]0,1[ et lr-patience positif")
    if args.gradient_clip < 0:
        parser.error("gradient-clip doit etre positif ou nul")
    if not args.causal and args.window_samples % 2 == 0:
        parser.error("une fenetre centree doit avoir une longueur impaire")
    if args.model_type in {"tcn", "transformer", "patch_transformer", "sensor_token"} and not args.causal:
        parser.error("le TCN/Transformer de deploiement doit utiliser --causal")
    if args.model_type == "sensor_token":
        if args.hidden_size % args.transformer_heads != 0:
            parser.error("hidden-size doit etre divisible par transformer-heads")
        if args.input_features != "raw":
            parser.error("sensor_token lit les canaux bruts: --input-features raw")
        if not 0.0 <= args.token_dropout < 1.0:
            parser.error("--token-dropout doit etre dans [0,1[")
    if args.foot_height_loss_weight < 0.0:
        parser.error("--foot-height-loss-weight doit etre positif ou nul")
    if args.vertical_sum_loss_weight < 0.0:
        parser.error("--vertical-sum-loss-weight doit etre positif ou nul")
    if not 0.0 <= args.body_weight_masking < 1.0:
        parser.error("--body-weight-masking doit etre dans [0, 1[")
    if (args.body_weight_masking > 0.0 or args.body_weight_mode != "provided") and not args.body_weight_input:
        parser.error("--body-weight-masking / --body-weight-mode demandent --body-weight-input")
    if args.vertical_share:
        if not args.body_weight_input:
            parser.error("--vertical-share ancre le total sur le poids du corps: ajouter --body-weight-input")
        if args.target_scaling != "standard":
            parser.error("--vertical-share demande --target-scaling standard (Fz en N)")
        if args.hard_airborne_zero:
            parser.error("--vertical-share et --hard-airborne-zero sont incompatibles")
    if args.vertical_loss == "decomposed":
        if not args.vertical_share:
            parser.error("--vertical-loss decomposed demande --vertical-share")
        if args.model_type == "sensor_token":
            parser.error("--vertical-loss decomposed: sensor_token n'a pas de tetes dediees (12 sorties fixes)")
    if args.vertical_loss == "sum":
        if args.vertical_share:
            parser.error("--vertical-loss sum est le diagnostic sans --vertical-share (qui a deja la loss de somme)")
        if args.target_scaling != "standard":
            parser.error("--vertical-loss sum demande --target-scaling standard (Fz en N)")
    if not 0.0 <= args.zero_target_weight <= 1.0:
        parser.error("--zero-target-weight doit etre entre 0 et 1")
    if args.gradnorm_alpha < 0.0:
        parser.error("--gradnorm-alpha doit etre positif ou nul")
    if args.gradnorm_learning_rate < 0.0:
        parser.error("--gradnorm-learning-rate doit etre positif ou nul")
    if args.warmup_steps < 0:
        parser.error("--warmup-steps doit etre positif ou nul")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay doit etre positif ou nul")
    if not 0.0 <= args.warmup_fraction < 1.0:
        parser.error("--warmup-fraction doit etre dans [0, 1[")
    if args.warmup_fraction > 0.0 and (args.lr_scheduler != "cosine" or args.warmup_steps > 0):
        parser.error("--warmup-fraction demande --lr-scheduler cosine et remplace --warmup-steps")
    if args.lr_scheduler == "cosine" and args.min_learning_rate > args.learning_rate:
        parser.error("--lr-scheduler cosine: --min-learning-rate doit etre <= --learning-rate")
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema-decay doit etre dans [0, 1[")
    if args.anomalous_loss_threshold < 0.0:
        parser.error("--anomalous-loss-threshold doit etre positif ou nul")
    if args.overfit_epochs < 1 or args.overfit_windows_per_stratum < 1:
        parser.error("--overfit-epochs et --overfit-windows-per-stratum doivent etre positifs")
    if args.overfit_nmse_threshold <= 0 or args.overfit_divergence_factor <= 1.0:
        parser.error("--overfit-nmse-threshold > 0 et --overfit-divergence-factor > 1")
    selected_targets = tuple(args.targets) if args.targets else None
    if (
        (args.vertical_share or args.vertical_loss != "default")
        and selected_targets
        and not set(VERTICAL_TARGETS) <= set(selected_targets)
    ):
        parser.error(f"--vertical-share / --vertical-loss demandent les cibles {', '.join(VERTICAL_TARGETS)}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else (args.device if args.device != "auto" else "cpu")
    device = torch.device(device_name)
    try:
        acquisitions, source_metadata, targets, target_indexes = load_acquisitions(
            args.data_dir, selected_targets, args.subjects, args.task_set
        )
        source_feature_names = list(source_metadata["feature_names"])
        if args.sensor_configuration:
            source_feature_names = restrict_to_sensor_configuration(
                acquisitions, source_feature_names, args.sensor_configuration
            )
        model_feature_names = prepare_input_features(
            acquisitions,
            source_feature_names,
            args.input_features,
            args.gauge_proportion_epsilon_n,
            args.load_calibration_seconds,
            estimate_load=args.target_scaling == "subject_load",
            loadsol_heel_imu=args.loadsol_heel_imu,
        )
        body_weights_n = {}
        if args.body_weight_input:
            model_feature_names, body_weights_n = append_body_weight_input(
                acquisitions, model_feature_names, source_metadata
            )
            if args.body_weight_masking > 0.0:
                model_feature_names = append_body_weight_missing_flag(acquisitions, model_feature_names)
        if args.sensor_dropout:
            model_feature_names = append_sensor_presence(acquisitions, model_feature_names)
        if args.model_type == "sensor_token":
            model_feature_names = prepare_sensor_token_inputs(
                acquisitions, model_feature_names, targets
            )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    if args.foot_height_output:
        missing = [item.name for item in acquisitions if item.foot_height is None]
        if missing:
            parser.error(
                "--foot-height-output demande les cles NPZ "
                f"{', '.join(FOOT_HEIGHT_KEYS + FOOT_HEIGHT_VALID_KEYS)} (dataset v3.7); "
                f"absentes de {', '.join(missing[:3])}"
            )
    if args.hard_airborne_zero and str(
        source_metadata.get("pipeline_version", "")
    ).startswith("prepared_joint_masked_v3"):
        parser.error(
            "--hard-airborne-zero utiliserait un label mocap indisponible au deploiement; "
            "laisser le LSTM apprendre les zeros supervises"
        )
    names = [item.name for item in acquisitions]
    task_counts = {
        family: sum(item.task_family == family for item in acquisitions)
        for family in sorted({item.task_family for item in acquisitions})
    }
    print(
        f"model={args.model_type} features={args.input_features} "
        f"target_scaling={args.target_scaling} input_size={len(model_feature_names)} "
        f"task_set={args.task_set} acquisitions={len(acquisitions)} "
        + " ".join(f"{family}={count}" for family, count in task_counts.items())
    )
    try:
        groups, split_manifest = (
            build_fixed_split(acquisitions, args.split_level, args.test_groups, args.validation_groups)
            if args.test_groups
            else build_split_manifest(acquisitions, args.split_level, args.validation_group_count)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.only_test_group:
        if args.only_test_group in names:
            requested_group = split_group(
                acquisitions[names.index(args.only_test_group)], args.split_level
            )
        elif args.only_test_group in groups or any(
            fold["test_group"] == args.only_test_group for fold in split_manifest
        ):
            requested_group = args.only_test_group
        else:
            parser.error(f"essai ou groupe test inconnu: {args.only_test_group}")
    else:
        requested_group = None

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "split": "fixed" if args.test_groups else "full_cv",
        "split_level": args.split_level,
        "validation_group_count": args.validation_group_count,
        "task_set": args.task_set,
        "task_counts": task_counts,
        "identity_rule": "remove terminal 2 from session folder"
        if args.split_level == "subject_identity"
        else None,
        "groups": groups,
        "folds": split_manifest,
    }
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for fold in split_manifest:
        print(
            f"split test={','.join(fold['test_groups'])} "
            f"validation={','.join(fold['validation_groups'])} "
            f"train={','.join(fold['train_groups'])}"
        )
    if args.dry_run_split:
        print(f"Split manifest: {(args.output_dir / 'split_manifest.json').resolve()}")
        return 0
    if args.overfit_test:
        overfit_fold = next(
            (fold for fold in split_manifest if requested_group in (fold["test_group"], *fold["test_groups"])),
            split_manifest[0],
        )
        return run_overfit_test(
            args, acquisitions, targets, target_indexes, overfit_fold, device,
            feature_names=model_feature_names,
        )
    if not args.selection_only:
        if not args.no_plots:
            (args.output_dir / "plots").mkdir(exist_ok=True)
        (args.output_dir / "models").mkdir(exist_ok=True)
    metrics, histories = [], {}
    body_weight_modes = args.body_weight_masking > 0.0 or args.body_weight_mode != "provided"
    prediction_fields = prediction_csv_fields(
        targets, args.foot_height_output, args.vertical_share, ["body_weight_mode"] if body_weight_modes else []
    )
    body_weight_defaults_n = {}
    predictions_path = args.output_dir / "predictions.csv"
    if not args.selection_only:
        with predictions_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=prediction_fields)
            writer.writeheader()
    gradnorm_path = args.output_dir / "gradnorm_weights.csv"
    tasks = gradnorm_tasks(targets, args.vertical_loss)
    gradnorm_fields = ["test_group", "epoch", *tasks]
    vertical_loss_scales_n = {}
    if args.gradnorm:
        with gradnorm_path.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=gradnorm_fields).writeheader()
    selected_folds = [
        fold for fold in split_manifest
        if requested_group is None or requested_group in (fold["test_group"], *fold["test_groups"])
    ]
    for fold in selected_folds:
        test_group = fold["test_group"]
        test_groups = set(fold.get("test_groups", [test_group]))
        validation_groups = set(
            fold.get("validation_groups", [fold["validation_group"]])
        )
        train_groups = set(fold["train_groups"])
        test = [
            item for item in acquisitions
            if split_group(item, args.split_level) in test_groups
        ]
        train = [
            item
            for item in acquisitions
            if split_group(item, args.split_level) in train_groups
        ]
        validation = [
            item for item in acquisitions
            if split_group(item, args.split_level) in validation_groups
        ]
        try:
            x_mean, x_std, y_mean, y_std = statistics(
                train,
                target_indexes,
                args.zero_target_weight,
                target_scaling=args.target_scaling,
            )
            physical_y_std = physical_target_std(
                train, target_indexes, args.zero_target_weight
            )
            foot_height_mean, foot_height_std = (
                foot_height_statistics(train) if args.foot_height_output else (None, None)
            )
        except ValueError as exc:
            print(f"SKIP {test_group}: {exc}")
            continue
        evaluation_body_weight = train_body_weight = None
        if body_weight_modes:
            body_weight_defaults_n[test_group] = float(x_mean[model_feature_names.index(BODY_WEIGHT_FEATURE)])
            evaluation_body_weight = BodyWeightMode.from_statistics(
                model_feature_names, x_mean, x_std, missing=args.body_weight_mode == "missing"
            )
            train_body_weight = BodyWeightMode.from_statistics(
                model_feature_names, x_mean, x_std, mask_probability=args.body_weight_masking
            )
        # Evaluation: the full configuration (Inactive sensors still removed); training: random draws.
        evaluation_sensor_dropout = train_sensor_dropout = None
        if args.sensor_dropout:
            evaluation_sensor_dropout = SensorDropout.from_feature_names(model_feature_names)
            train_sensor_dropout = SensorDropout.from_feature_names(
                model_feature_names,
                random=True,
                full_probability=args.sensor_full_probability,
                channel_probability=args.sensor_channel_dropout,
            )
        common = dict(
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            target_indexes=target_indexes,
            window=args.window_samples,
            centered=not args.causal,
            zero_weight=args.zero_target_weight,
            target_scaling=args.target_scaling,
            # With the head, windows whose only target is Foot height (Masked) are kept too.
            foot_height_mean=foot_height_mean,
            foot_height_std=foot_height_std,
            body_weight=evaluation_body_weight,
            sensor_dropout=evaluation_sensor_dropout,
        )
        train_set = MaskedWindowDataset(
            train, **{**common, "body_weight": train_body_weight, "sensor_dropout": train_sensor_dropout}
        )
        validation_set = MaskedWindowDataset(validation, **common)
        # Measured-only selection: a zero Airborne weight leaves the Measured subset.
        measured_validation_set = (
            MaskedWindowDataset(validation, **{**common, "zero_weight": 0.0})
            if args.selection_subset == "measured"
            else None
        )
        test_sets = (
            []
            if args.selection_only
            else [(item, MaskedWindowDataset([item], **common)) for item in test]
        )
        if (
            not len(train_set)
            or not len(validation_set)
            or (
                not args.selection_only
                and (
                    not test_sets
                    or any(not len(test_set) for _, test_set in test_sets)
                )
            )
        ):
            print(f"SKIP {test_group}: aucun centre de fenetre avec cible valide")
            continue
        loader_options = dict(
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        generator = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train_set, args.batch_size, shuffle=True, generator=generator, **loader_options)
        validation_loader = DataLoader(validation_set, args.batch_size, shuffle=False, **loader_options)
        measured_validation_loader = (
            DataLoader(measured_validation_set, args.batch_size, shuffle=False, **loader_options)
            if measured_validation_set is not None
            else None
        )
        missing_counters = {}
        if body_weight_modes and train_body_weight.flag_column is not None:
            # The flag shows which windows saw the default value.
            flag_column = train_body_weight.flag_column
            train_loader = missing_counters["train"] = BodyWeightMissingCounter(train_loader, flag_column)
            validation_loader = missing_counters["validation"] = BodyWeightMissingCounter(
                validation_loader, flag_column
            )
        test_loaders = [
            (
                item,
                test_set,
                DataLoader(test_set, args.batch_size, shuffle=False, **loader_options),
            )
            for item, test_set in test_sets
        ]
        model = model_from_args(
            args, train[0].x.shape[1], len(targets), feature_names=model_feature_names,
            targets=list(targets),
        ).to(device)
        fit_model_statistics(
            model, train, x_mean, x_std, y_mean, y_std, foot_height_mean, foot_height_std
        )
        vertical_columns = (
            [list(targets).index(target) for target in VERTICAL_TARGETS]
            if args.vertical_share or args.vertical_loss != "default"
            else None
        )
        vertical_scales = None
        if args.vertical_loss == "decomposed":
            vertical_scales = decomposed_vertical_scales(
                train, list(targets), target_indexes, args.zero_target_weight
            )
            vertical_loss_scales_n[test_group] = vertical_scales
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        if args.lr_scheduler == "cosine":
            total_steps = args.epochs * len(train_loader)
            warmup_steps = round(args.warmup_fraction * total_steps) if args.warmup_fraction > 0.0 else args.warmup_steps
            add_cosine_schedule(optimizer, warmup_steps, total_steps, args.min_learning_rate)
        else:
            add_linear_warmup(optimizer, args.warmup_steps)
        weight_average = WeightAverage(model, args.ema_decay) if args.ema_decay > 0.0 else None
        if weight_average is not None:
            weight_average.attach(optimizer)
        guard = AnomalousBatchGuard(args.anomalous_loss_threshold) if args.anomalous_loss_threshold > 0.0 else None
        scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=args.lr_factor,
                patience=args.lr_patience,
                min_lr=args.min_learning_rate,
            )
            if args.lr_scheduler == "plateau"
            else None
        )
        balancer = (
            GradNormBalancer(
                model,
                len(targets),
                alpha=args.gradnorm_alpha,
                learning_rate=args.gradnorm_learning_rate,
            )
            if args.gradnorm
            else None
        )
        best_state, best_loss, stale = None, math.inf, 0
        history = []
        for epoch in range(1, args.epochs + 1):
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_foot_height = FootHeightLoss(args.foot_height_loss_weight)
            validation_foot_height = FootHeightLoss(args.foot_height_loss_weight)
            sum_loss = args.vertical_loss == "sum" or (args.vertical_share and args.vertical_loss == "default")
            train_vertical_sum, validation_vertical_sum = (
                [
                    VerticalSumLoss(
                        args.vertical_sum_loss_weight,
                        vertical_columns,
                        y_mean[vertical_columns],
                        y_std[vertical_columns],
                    )
                    for _ in range(2)
                ]
                if sum_loss
                else (None, None)
            )
            train_decomposed, validation_decomposed = (
                [
                    DecomposedVerticalLoss(
                        vertical_columns, y_mean[vertical_columns], y_std[vertical_columns], **vertical_scales
                    )
                    for _ in range(2)
                ]
                if vertical_scales is not None
                else (None, None)
            )
            train_loss = run_epoch(
                model,
                train_loader,
                device,
                optimizer,
                gradient_clip=args.gradient_clip,
                balancer=balancer,
                foot_height=train_foot_height,
                vertical_sum=train_vertical_sum,
                decomposed=train_decomposed,
                guard=guard,
            )
            # GRFM loss only: the Foot height and vertical sum terms stay out of early
            # stopping and the scheduler.
            validation_loss = run_epoch(
                model, validation_loader, device, foot_height=validation_foot_height,
                vertical_sum=validation_vertical_sum, decomposed=validation_decomposed,
            )
            history_row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_lr,
            }
            if args.foot_height_output:
                # Standardised Foot height MSE (unweighted), reported next to the GRFM losses.
                history_row["train_foot_height_loss"] = train_foot_height.value
                history_row["validation_foot_height_loss"] = validation_foot_height.value
            if sum_loss:
                # Fz_L + Fz_R error where both feet are Measured (standardised, unweighted).
                history_row["train_vertical_sum_loss"] = train_vertical_sum.value
                history_row["validation_vertical_sum_loss"] = validation_vertical_sum.value
            if train_decomposed is not None:
                # Total and share terms of the decomposed loss (normalised by sigma_sum, sigma_share).
                for split, collector in (("train", train_decomposed), ("validation", validation_decomposed)):
                    history_row[f"{split}_vertical_total_loss"] = collector.total_value
                    history_row[f"{split}_vertical_share_loss"] = collector.share_value
            if args.vertical_share:
                # Samples whose total (both feet Airborne) or share (one foot Airborne) is clamped.
                collectors = (
                    (("train", train_decomposed), ("validation", validation_decomposed))
                    if train_decomposed is not None
                    else (("train", train_vertical_sum), ("validation", validation_vertical_sum))
                )
                for split, collector in collectors:
                    total_fraction, share_fraction = collector.saturated_fractions
                    history_row[f"{split}_vertical_total_saturated_fraction"] = total_fraction
                    history_row[f"{split}_vertical_share_saturated_fraction"] = share_fraction
            for split, counter in missing_counters.items():
                history_row[f"{split}_body_weight_missing_fraction"] = counter.fraction
            selection_loss = validation_loss
            if measured_validation_loader is not None:
                selection_loss = run_epoch(model, measured_validation_loader, device)
                history_row["validation_measured_loss"] = selection_loss
            selected_weights = "raw"
            if weight_average is not None:
                # Same GRFM loss (and same subset) on the averaged weights; the better of the two is kept.
                with weight_average.applied():
                    history_row["validation_ema_loss"] = average_loss = run_epoch(model, validation_loader, device)
                    if measured_validation_loader is not None:
                        history_row["validation_ema_measured_loss"] = average_loss = run_epoch(
                            model, measured_validation_loader, device
                        )
                if average_loss < selection_loss:
                    selection_loss, selected_weights = average_loss, "ema"
            history_row["selection_loss"] = selection_loss
            history_row["selected_weights"] = selected_weights
            if guard is not None:
                history_row["skipped_anomalous_batches"] = guard.skipped
            if balancer is not None:
                epoch_weights = dict(zip(tasks, balancer.current_weights()))
                history_row["gradnorm_weights"] = epoch_weights
                with gradnorm_path.open("a", newline="", encoding="utf-8") as stream:
                    csv.DictWriter(stream, fieldnames=gradnorm_fields).writerow(
                        {"test_group": test_group, "epoch": epoch, **epoch_weights}
                    )
            history.append(history_row)
            print(
                f"test={test_group} epoch={epoch:03d} train={train_loss:.6f} "
                f"val={validation_loss:.6f} lr={current_lr:.3g}"
            )
            if selection_loss < best_loss - 1e-6:
                if selected_weights == "ema":
                    with weight_average.applied():
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    best_state = copy.deepcopy(model.state_dict())
                best_loss, stale = selection_loss, 0
            else:
                stale += 1
                if stale >= args.patience:
                    break
            if scheduler is not None:
                scheduler.step(selection_loss)
        if best_state is None:
            continue
        model.load_state_dict(best_state)
        comparable_validation_score = physical_normalized_mse(
            model,
            validation_set,
            validation_loader,
            y_mean,
            y_std,
            physical_y_std,
            device,
        )
        best_history_row = min(history, key=lambda row: row["selection_loss"])
        best_history_row["validation_physical_nmse"] = comparable_validation_score
        print(
            f"test={test_group} best_epoch={best_history_row['epoch']:03d} "
            f"val_physical_nmse={comparable_validation_score:.6f}"
        )
        histories[test_group] = history
        if args.selection_only:
            continue
        for test_item, test_set, test_loader in test_loaders:
            outputs = predict(
                model,
                test_set,
                test_loader,
                y_mean,
                y_std,
                device,
                target_indexes,
                args.hard_airborne_zero,
            )
            truth, predicted, mask, times, state_l, state_r = outputs[:6]
            metrics.extend(metric_rows(
                truth, predicted, mask, state_l, state_r,
                f"{args.model_type}_two_feet_masked",
                test_item.name, test_item.task_family, targets, target_indexes,
            ))
            with predictions_path.open("a", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=prediction_fields)
                writer.writerows(prediction_rows(
                    test_item, split_group(test_item, args.split_level), targets, outputs,
                    {"body_weight_mode": args.body_weight_mode} if body_weight_modes else None,
                ))
            if args.no_plots:
                continue
            save_plot(
                args.output_dir / "plots" / f"{args.model_type}_{test_item.name}.png",
                test_item.name,
                times,
                truth,
                predicted,
                mask,
                targets,
                args.model_type.upper(),
            )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_type": args.model_type,
                "input_size": train[0].x.shape[1],
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "dropout": args.dropout,
                "tcn_kernel_size": args.tcn_kernel_size,
                "transformer_heads": args.transformer_heads,
                "transformer_feedforward_size": args.transformer_feedforward_size,
                "patch_samples": args.patch_samples,
                "decoder_layers": args.decoder_layers,
                "token_dropout": args.token_dropout,
                "input_layer_norm": args.input_layer_norm,
                "learning_rate": args.learning_rate,
                "lr_scheduler": args.lr_scheduler,
                "warmup_fraction": args.warmup_fraction,
                "weight_decay": args.weight_decay,
                "ema_decay": args.ema_decay,
                "selected_weights": best_history_row["selected_weights"],
                "anomalous_loss_threshold": args.anomalous_loss_threshold,
                "selection_subset": args.selection_subset,
                "gradient_clip": args.gradient_clip,
                "window_samples": args.window_samples,
                "causal": args.causal,
                "hard_airborne_zero": args.hard_airborne_zero,
                "zero_target_weight": args.zero_target_weight,
                "gradnorm": args.gradnorm,
                "gradnorm_alpha": args.gradnorm_alpha,
                "gradnorm_final_weights": (
                    balancer.current_weights() if balancer is not None else None
                ),
                "foot_height_output": args.foot_height_output,
                "foot_height_loss_weight": args.foot_height_loss_weight,
                "foot_height_mean_mm": foot_height_mean,
                "foot_height_std_mm": foot_height_std,
                "vertical_share": args.vertical_share,
                "vertical_sum_loss_weight": args.vertical_sum_loss_weight,
                "vertical_loss": args.vertical_loss,
                "vertical_heads": list(VERTICAL_HEADS) if args.vertical_loss == "decomposed" else None,
                "vertical_loss_scales_N": vertical_scales,
                "gradnorm_tasks": tasks if args.gradnorm else None,
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "physical_y_std": physical_y_std,
                "validation_physical_nmse": comparable_validation_score,
                "feature_names": model_feature_names,
                "source_feature_names": source_feature_names,
                "sensor_configuration": args.sensor_configuration,
                "input_features": args.input_features,
                "loadsol_heel_imu": args.loadsol_heel_imu,
                "body_weight_input": args.body_weight_input,
                "body_weight_masking": args.body_weight_masking,
                "body_weight_mode": args.body_weight_mode,
                "sensor_dropout": (
                    {
                        "full_probability": args.sensor_full_probability,
                        "channel_probability": args.sensor_channel_dropout,
                        "groups": list(train_sensor_dropout.group_names),
                    }
                    if args.sensor_dropout
                    else None
                ),
                "body_weight_default_N": body_weight_defaults_n.get(test_group),
                "body_weights_N": {
                    item.subject_session: body_weights_n[item.subject_session]
                    for item in train + validation + test
                    if item.subject_session in body_weights_n
                },
                "gauge_proportion_epsilon_n": args.gauge_proportion_epsilon_n,
                "target_scaling": args.target_scaling,
                "load_calibration_seconds": args.load_calibration_seconds,
                "load_estimates_n": {
                    item.name: item.load_estimate_n
                    for item in train + validation + test
                },
                "targets": targets,
                "target_indexes": target_indexes,
                "target_frame": source_metadata.get("target_frame"),
                "contact_supervision": source_metadata.get("contact_supervision"),
                "source_pipeline_version": source_metadata.get("pipeline_version"),
                "sample_grid": source_metadata.get("sample_grid"),
                "test_group": test_group,
                "test_groups": sorted(test_groups),
                "test_batches": [item.name for item in test],
                "validation_group": sorted(validation_groups)[0],
                "validation_groups": sorted(validation_groups),
                "validation_batches": [item.name for item in validation],
                "train_groups": sorted(train_groups),
                "train_batches": [item.name for item in train],
                "split_level": args.split_level,
                "task_set": args.task_set,
                "task_counts": task_counts,
            },
            args.output_dir / "models" / f"{args.model_type}_test_{test_group}.pt",
        )

    if not histories:
        print("Aucun pli exploitable.", file=sys.stderr)
        return 1
    if not args.selection_only:
        if not metrics:
            print("Aucune metrique test disponible.", file=sys.stderr)
            return 1
        with (args.output_dir / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
    (args.output_dir / "training_history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    run_metadata = vars(args).copy()
    for key in ("data_dir", "output_dir"):
        run_metadata[key] = str(run_metadata[key].resolve())
    run_metadata.update({
        "device_used": str(device),
        "targets": list(targets),
        "target_frame": source_metadata.get("target_frame"),
        "source_pipeline_version": source_metadata.get("pipeline_version"),
        "sample_grid": source_metadata.get("sample_grid"),
        "contact_supervision": source_metadata.get("contact_supervision"),
        "body_weights_N": body_weights_n if args.body_weight_input else None,
        # Default body weight (training mean, N) per fold, with --body-weight-masking / --body-weight-mode.
        "body_weight_default_N": body_weight_defaults_n or None,
        # --vertical-loss decomposed: per fold, sigma_sum and sigma_share (N) from the train trials.
        "vertical_loss_scales_N": vertical_loss_scales_n or None,
        "airborne_zero_learned_from_insole_inputs": True,
        "masked_loss": True,
    })
    (args.output_dir / "metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Results: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
