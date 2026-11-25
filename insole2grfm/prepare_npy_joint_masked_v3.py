#!/usr/bin/env python3
"""Prepare les donnees 2 semelles a leur frequence native pour un LSTM causal.

Les entrees sont les paires de paquets gauche/droit alignees sur le temps QTM
corrige de chaque semelle, sans grille mocap, interpolation ni repetition. Le
mocap et les plateformes ne servent qu'a construire les cibles d'apprentissage.
Ils ne font jamais partie de ``X`` et ne seront donc pas necessaires au
deploiement.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from data_preparation_utils import (
    ABSENT_FORCE_SENSORS,
    FORCE_COLUMNS,
    IMU_COLUMNS,
    PATCH_IDS,
    STATE_AIRBORNE,
    STATE_PLATFORM,
    STATE_UNKNOWN,
    TARGET_AXES,
    active_platforms,
    discover_synchronized_trials,
    filter_and_interpolate_platform,
    marker_arrays,
    nearest_indices,
    rigid_foot_frames,
)
from grfm_schema import (  # per-sample, per-foot label source (``label_source_L/R`` in each NPZ)
    LABEL_SOURCE_CODES,
    LABEL_SOURCE_FOOT_HEIGHT,
    LABEL_SOURCE_NONE,
    LABEL_SOURCE_PLATE,
    LABEL_SOURCE_PROTOCOL_WINDOW,
    MEASURED_LABEL_SOURCES,
)
from static_body_weight import (
    BALANCE_SOURCE,
    DEFAULT_SEX_CONFIG,
    body_weights_block,
    load_sex,
    refresh_body_weights,
    static_standing_weights,
)
from foot_height import (
    DEFAULT_FALLBACK_FLOOR_Z_MM,
    MIN_VISIBLE_MARKERS,
    FootHeightCalibration,
    Ground,
    calibrate_subject,
    foot_height,
    marker_visibility,
    plate_surfaces,
)


@dataclass
class NativeInsole:
    receiver_time_s: np.ndarray
    qtm_time_s: np.ndarray
    packet_timestamp_us: np.ndarray
    x: np.ndarray
    feature_names: list[str]
    force_saturated: np.ndarray
    accelerometer_saturated: np.ndarray
    gyroscope_saturated: np.ndarray


@dataclass
class LateralSplit:
    switch_qtm_s: float
    active_start_qtm_s: float
    active_end_qtm_s: float
    double_support_before: float
    single_support_before: float
    double_support_after: float
    single_support_after: float
    score: float


DEFAULT_EXCLUDE_CONFIG = Path(__file__).with_name("excluded_trials.json")
DATASET_VERSION = "3.7"
# Names the foot-state labelling rules; the version string is
# ``prepared_joint_masked_v{dataset_version}_{LABELLING_RULES_TAG}``. The training
# engine relies on the ``prepared_joint_masked_v3`` prefix, hence 3.x versions only.
LABELLING_RULES_TAG = "plate_reliable_markers_insole_veto_airborne"
DATASET_VERSION_PATTERN = re.compile(r"3\.[0-9]+[A-Za-z0-9.-]*")

# Airborne threshold on Foot height (#15), calibrated on real walk, jump and
# lateral_1_appui trials (P01b, P07, P13, P15b): plate-empty samples
# whose own gauges read > 300 N (foot loaded on the floor) have Foot height
# q99.5 <= 5.2 mm, and the ground mode of plate-empty samples ends at ~+5 mm.
AIRBORNE_FOOT_HEIGHT_MM = 15.0
AIRBORNE_THRESHOLD_CALIBRATION = (
    "15 mm, calibrated on real walk, jump and lateral_1_appui trials of P01b, "
    "P07, P13 and P15b (11 trials, 2 feet): the Foot height of plate-empty "
    "samples has a ground mode from about -10 to +5 mm (push-off and floor stance) "
    "and a flat plateau above it with the foot in the air; plate-empty samples whose own gauges read more "
    "than 300 N (foot loaded on the floor, a check only, not a labelling rule) have "
    "Foot height q99.5 <= 5.2 mm per trial and foot and none above 15 mm except "
    "0.15 % on one foot. 15 mm keeps a >= 10 mm margin over the ground and recovers "
    "more of the foot in the air than 20 mm (Airborne share of foot samples at 15 vs 20 mm: walk "
    "13.0 vs 10.5 %, lateral_1_appui 23.7 vs 20.3 %, jump 3.7 vs 3.3 %); the "
    "gauge sum of the resulting Airborne samples has a median of 42-58 N (empty-shoe "
    "preload) and exceeds 150 N on 0-0.2 % of them."
)
# Airborne marker reliability (#24): Foot height counts for Airborne only where the
# foot's toe marker is visible or at least this many calibrated markers are.
AIRBORNE_MIN_MARKERS_WITHOUT_TOE = 3
AIRBORNE_MARKER_CALIBRATION = (
    "toe visible or >= 3 markers, calibrated on 86 real walk, jump and "
    "lateral_1_appui batches of 12 Subjects (P01b, P02, P03, P05b, "
    "P06b, P07, P07b, P09, P09b, P13, P14b, P15b) built "
    "with the first v3.7 rules: every jump and lateral_1_appui Airborne sample has "
    "its toe marker visible; in walk, the toe-hidden 2-marker samples are 4.8 % of "
    "Airborne but hold 709 of the 848 samples whose gauges read > 150 N (gauge-sum "
    "q99 652 N), while toe-hidden samples with 3 markers (127) and 4+ markers (8) "
    "hold none (q99 77 N). Requiring 4 or 5 markers removes no further suspect."
)
# Insole veto (#24): a foot whose gauge sum exceeds this load cannot be Airborne
# (it becomes Masked). The insole never creates an Airborne label.
AIRBORNE_INSOLE_VETO_N = 120.0
AIRBORNE_VETO_CALIBRATION = (
    "120 N, calibrated on the same 86 batches: the empty-shoe preload varies by "
    "Subject and foot (median gauge sum of reliable-marker Airborne samples in jump "
    "and lateral_1_appui 32-76 N; P03 reads 66-76 N), and the highest gauge sum "
    "of any jump or lateral_1_appui Airborne sample of the whole first v3.7 build "
    "(all Subjects) is 116.2 N, so 120 N vetoes no flight of those families (100 N "
    "would veto 13 in the 12 Subjects); in walk, the samples left above 150 N once the marker "
    "rule applies (139, toe visible, Foot height 15-22 mm, gauges up to 437 N) are "
    "all vetoed, at a cost of 0.1 % of walk Airborne."
)
GRAVITY_M_S2 = 9.81
BODY_WEIGHT_SOURCE_FAMILY = "balance"


def is_static_trial(trial_name: str) -> bool:
    """Static (quiet-standing) trials are not dataset trials from v3.7 on."""
    return trial_name.casefold().startswith("static")


def subject_body_weights(
    plate_fz_by_subject: dict[str, list[tuple[str, np.ndarray]]]
) -> dict[str, dict[str, object]]:
    """Body weight per Subject: median total plate Fz over its balance samples.

    ``plate_fz_by_subject`` maps each Subject to ``(batch, total plate Fz in N)``
    pairs, one per balance batch, holding only the samples with both feet Measured.
    The samples of all of a Subject's balance batches are pooled.
    """
    weights: dict[str, dict[str, object]] = {}
    for subject, parts in sorted(plate_fz_by_subject.items()):
        values = np.concatenate([fz for _, fz in parts]) if parts else np.empty(0)
        values = values[np.isfinite(values)]
        if not len(values):
            continue
        weight = float(np.median(values))
        weights[subject] = {
            "body_weight_N": weight,
            "body_weight_kg": weight / GRAVITY_M_S2,
            "source": BALANCE_SOURCE,
            "n_samples": int(len(values)),
            "trials": sorted(batch for batch, fz in parts if len(fz)),
            # Per-trial medians, to check that a Subject's balance trials agree.
            "trial_medians_N": {
                batch: float(np.median(fz[np.isfinite(fz)]))
                for batch, fz in sorted(parts)
                if np.isfinite(fz).any()
            },
        }
    return weights


def finite_or_none(value: float) -> float | None:
    """A JSON-safe threshold: an infinite one (rule disabled) is written as null."""
    return value if math.isfinite(value) else None


def pipeline_version(dataset_version: str) -> str:
    return f"prepared_joint_masked_v{dataset_version}_{LABELLING_RULES_TAG}"


def existing_dataset_version(output_dir: Path) -> str | None:
    """``pipeline_version`` of the dataset already in ``output_dir``, if any."""
    metadata = output_dir / "metadata.json"
    if not metadata.is_file():
        return None
    try:
        return str(json.loads(metadata.read_text(encoding="utf-8")).get("pipeline_version"))
    except (OSError, ValueError, AttributeError):
        return "<metadata.json illisible>"


def load_batch_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"{path}: une liste JSON de noms d'essais est attendue")
    return set(values)


def detect_lateral_support_switch(
    platforms: pd.DataFrame,
    contact_threshold_n: float = 15.0,
    active_total_threshold_n: float = 150.0,
    lowpass_hz: float = 15.0,
    minimum_phase_s: float = 8.0,
) -> LateralSplit:
    """Detecte le passage 2 appuis -> 1 appui a partir des plateformes.

    Le seuil bas sert uniquement a distinguer une plateforme reellement
    dechargee d'une plateforme encore en contact. Il n'est jamais applique
    aux jauges de la semelle. Une valeur de 15 N conserve donc les appuis
    legers observes pendant les transferts lateraux, contrairement a
    l'ancien seuil de 40 N.
    """
    time = platforms["time_qtm_s"].to_numpy(dtype=np.float64)
    if len(time) < 3 or np.any(np.diff(time) <= 0):
        raise ValueError("horloge plateformes invalide")
    rate_hz = 1.0 / float(np.median(np.diff(time)))
    if lowpass_hz >= rate_hz / 2.0:
        raise ValueError("frequence de coupure laterale invalide")
    sos = butter(4, lowpass_hz, fs=rate_hz, output="sos")
    forces: list[np.ndarray] = []
    for platform_number in (1, 2):
        global_column = f"force_z_p{platform_number}_global_N"
        local_column = f"force_z_p{platform_number}_N"
        if global_column in platforms:
            vertical = platforms[global_column].to_numpy(dtype=np.float64)
        else:
            # Dans les exports QTM utilises ici, l'axe z local est oppose a
            # la reaction verticale globale.
            vertical = -platforms[local_column].to_numpy(dtype=np.float64)
        forces.append(np.maximum(sosfiltfilt(sos, vertical), 0.0))

    force_1, force_2 = forces
    active = (force_1 + force_2) > active_total_threshold_n
    contact_1 = force_1 > contact_threshold_n
    contact_2 = force_2 > contact_threshold_n
    single_support = active & np.logical_xor(contact_1, contact_2)
    double_support = active & contact_1 & contact_2

    one_second = max(1, int(round(rate_hz)))
    active_fraction = np.convolve(
        active.astype(np.float64), np.ones(one_second) / one_second, mode="same"
    )
    active_indexes = np.flatnonzero(active_fraction > 0.8)
    if not len(active_indexes):
        raise ValueError("aucune periode active stable detectee")
    start, end = int(active_indexes[0]), int(active_indexes[-1])
    minimum_samples = int(round(minimum_phase_s * rate_hz))
    if end - start < 2 * minimum_samples:
        raise ValueError("essai trop court pour separer les deux phases")

    # Cherche la rupture qui maximise a la fois l'augmentation du temps en
    # appui simple et la diminution du temps en double appui.
    step = max(1, int(round(0.1 * rate_hz)))
    best: tuple[float, int, float, float, float, float] | None = None
    for split in range(start + minimum_samples, end - minimum_samples, step):
        single_before = float(single_support[start:split].mean())
        single_after = float(single_support[split : end + 1].mean())
        double_before = float(double_support[start:split].mean())
        double_after = float(double_support[split : end + 1].mean())
        score = (single_after - single_before) + (double_before - double_after)
        candidate = (
            score,
            split,
            single_before,
            single_after,
            double_before,
            double_after,
        )
        if best is None or candidate[0] > best[0]:
            best = candidate
    assert best is not None
    score, split, single_before, single_after, double_before, double_after = best
    if double_before < 0.75 or single_after < 0.45 or score < 0.70:
        raise ValueError(
            "bascule 2 appuis -> 1 appui insuffisamment nette "
            f"(double avant={double_before:.2f}, simple apres={single_after:.2f}, "
            f"score={score:.2f})"
        )
    return LateralSplit(
        switch_qtm_s=float(time[split]),
        active_start_qtm_s=float(time[start]),
        active_end_qtm_s=float(time[end]),
        double_support_before=double_before,
        single_support_before=single_before,
        double_support_after=double_after,
        single_support_after=single_after,
        score=score,
    )


def pivot_native_insole(insoles: pd.DataFrame, side: str) -> NativeInsole:
    """Produit exactement une ligne par paquet, dans l'ordre de la semelle."""
    values = list(FORCE_COLUMNS + IMU_COLUMNS)
    rows = insoles.loc[insoles["side"] == side].copy()
    packet_column = "packet_timestamp_unwrapped_us"
    if rows.duplicated([packet_column, "patch_id"]).any():
        raise ValueError(f"doublons paquet/patch pour le pied {side}")

    rows["receiver_time_s"] = rows["time_qtm_s"] - rows["side_offset_ms"] / 1000.0
    metadata = rows.groupby(packet_column, sort=True).agg(
        qtm_time_s=("time_qtm_s", "first"),
        qtm_time_count=("time_qtm_s", "nunique"),
        receiver_time_s=("receiver_time_s", "first"),
        receiver_time_count=("receiver_time_s", "nunique"),
        patch_count=("patch_id", "nunique"),
    )
    if (
        (metadata["qtm_time_count"] != 1).any()
        or (metadata["receiver_time_count"] != 1).any()
        or (metadata["patch_count"] != 4).any()
    ):
        raise ValueError(f"paquets incomplets ou temps incoherents pour le pied {side}")

    wide = rows.pivot(index=packet_column, columns="patch_id", values=values)
    wide.columns = [f"p{int(patch)}_{column}" for column, patch in wide.columns]
    wide = wide.sort_index()
    expected: list[str] = []
    for patch in PATCH_IDS[side]:
        for column in values:
            if column == "force_1_N" and (patch, 1) in ABSENT_FORCE_SENSORS:
                continue
            expected.append(f"p{patch}_{column}")
    missing = sorted(set(expected) - set(wide.columns))
    if missing:
        raise ValueError(f"colonnes semelle manquantes pour {side}: {missing}")
    x = wide[expected].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"valeurs semelle non finies pour {side}")
    force_raw = [f"force_{index}_raw_counts" for index in range(1, 5)]
    acc_raw = [f"acc_{axis}_raw_counts" for axis in "xyz"]
    gyro_raw = [f"gyro_{axis}_raw_counts" for axis in "xyz"]
    packet_qc = pd.DataFrame(
        {
            "force_saturated": (rows[force_raw].to_numpy(dtype=np.int64) >= 16383).any(axis=1),
            "accelerometer_saturated": (
                np.abs(rows[acc_raw].to_numpy(dtype=np.int64)) >= 32760
            ).any(axis=1),
            "gyroscope_saturated": (
                np.abs(rows[gyro_raw].to_numpy(dtype=np.int64)) >= 32760
            ).any(axis=1),
            packet_column: rows[packet_column].to_numpy(dtype=np.int64),
        }
    ).groupby(packet_column, sort=True).max()
    packet_qc = packet_qc.loc[wide.index]
    metadata = metadata.loc[wide.index]
    receiver_time = metadata["receiver_time_s"].to_numpy(dtype=np.float64)
    if len(receiver_time) < 2 or np.any(np.diff(receiver_time) <= 0):
        raise ValueError(f"horloge semelle non strictement croissante pour {side}")
    return NativeInsole(
        receiver_time_s=receiver_time,
        qtm_time_s=metadata["qtm_time_s"].to_numpy(dtype=np.float64),
        packet_timestamp_us=wide.index.to_numpy(dtype=np.int64),
        x=x,
        feature_names=expected,
        force_saturated=packet_qc["force_saturated"].to_numpy(dtype=bool),
        accelerometer_saturated=packet_qc["accelerometer_saturated"].to_numpy(dtype=bool),
        gyroscope_saturated=packet_qc["gyroscope_saturated"].to_numpy(dtype=bool),
    )


def pair_native_packets(
    left_time: np.ndarray, right_time: np.ndarray, tolerance_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Appairage monotone 1:1, sans interpolation ni reutilisation de paquet."""
    left_indexes: list[int] = []
    right_indexes: list[int] = []
    left = right = 0
    while left < len(left_time) and right < len(right_time):
        difference = right_time[right] - left_time[left]
        if abs(difference) <= tolerance_s:
            left_indexes.append(left)
            right_indexes.append(right)
            left += 1
            right += 1
        elif difference < -tolerance_s:
            right += 1
        else:
            left += 1
    return np.asarray(left_indexes, dtype=np.int64), np.asarray(right_indexes, dtype=np.int64)


def is_continuous_double_support_task(trial_name: str) -> bool:
    """Tasks whose recorded movement is performed with both feet on the plates."""
    name = trial_name.casefold()
    if "walk" in name or "jump" in name or "1_appui" in name:
        return False
    return any(task in name for task in ("balance", "squat", "hulahoop", "lateral"))


def task_family(trial_name: str, task_phase: str | None) -> str:
    """Canonical task name used to build reproducible training subsets."""
    if task_phase == "lateral_2supports":
        return "lateral_2supports"
    if task_phase == "lateral_1support":
        return "lateral_1_appui"
    name = trial_name.casefold()
    if "walk" in name:
        return "walk"
    if "jump" in name:
        return "jump"
    if "lateral" in name and "1_appui" in name:
        return "lateral_1_appui"
    if "lateral" in name:
        return "lateral_2supports"
    for family in ("hulahoop", "balance", "squat"):
        if family in name:
            return family
    return "other"


def sustained_true(mask: np.ndarray, times: np.ndarray, minimum_duration_s: float) -> np.ndarray:
    """Keep only true runs long enough to represent a physical flight phase."""
    result = np.zeros(len(mask), dtype=bool)
    transitions = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    for start, stop in transitions.reshape(-1, 2):
        if stop - start < 2:
            continue
        duration = float(times[stop - 1] - times[start])
        if duration >= minimum_duration_s:
            result[start:stop] = True
    return result


def dominant_platform(
    contacts: list[dict[str, object]], side: str, minimum_correlation: float
) -> int:
    """Return the platform most often assigned to one foot."""
    durations = {1: 0.0, 2: 0.0}
    for contact in contacts:
        if contact.get("matched_side") != side:
            continue
        correlation = contact.get("correlation")
        spatially_validated = bool(contact.get("spatially_validated", False))
        if not spatially_validated and (
            correlation is not None
            and np.isfinite(correlation)
            and correlation < minimum_correlation
        ):
            continue
        platform = int(contact["platform"])
        durations[platform] += max(
            0.0,
            float(contact["contact_end_s"]) - float(contact["contact_start_s"]),
        )
    return max(durations, key=durations.get) if max(durations.values()) > 0 else 0


def spatially_reassign_contacts(
    contacts: list[dict[str, object]],
    times_by_side: dict[str, np.ndarray],
    origins: dict[str, np.ndarray],
    platform_values: dict[str, dict[str, np.ndarray]],
    platform_centers: dict[int, np.ndarray],
    correlation_bonus_mm: float,
    continuity_bonus_mm: float,
    maximum_assignment_distance_mm: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Reassign platform contacts using mocap/CoP geometry first.

    Correlation-only assignment is ambiguous for synchronous bilateral jumps.
    The CoP should lie below the contacting foot, while the platform centre and
    the previous event on that platform provide robust fallbacks. The original
    gauge correlation is retained as a small tie-breaker, never as the primary
    spatial evidence.
    """
    reassigned: list[dict[str, object]] = []
    changed_events: list[dict[str, object]] = []
    previous_side: dict[int, str] = {}
    spatial_events = 0
    retained_without_geometry = 0
    margins: list[float] = []

    for source in sorted(contacts, key=lambda item: float(item["contact_start_s"])):
        contact = dict(source)
        platform = int(contact["platform"])
        original = str(contact.get("matched_side", ""))
        correlation = contact.get("correlation")
        correlation_value = (
            float(correlation)
            if correlation is not None and np.isfinite(correlation)
            else 0.0
        )
        scores: dict[str, float] = {}
        details: dict[str, dict[str, float | None]] = {}
        for side in ("L", "R"):
            times = times_by_side[side]
            inside = (times >= float(contact["contact_start_s"])) & (
                times <= float(contact["contact_end_s"])
            )
            valid_origin = inside & np.isfinite(origins[side]).all(axis=1)
            if not valid_origin.any():
                details[side] = {"cop_distance_mm": None, "center_distance_mm": None}
                continue

            center = platform_centers.get(platform)
            center_distance = (
                float(
                    np.median(
                        np.linalg.norm(origins[side][valid_origin, :2] - center[:2], axis=1)
                    )
                )
                if center is not None
                else math.nan
            )
            cop_columns = [
                f"cop_{axis}_p{platform}_global_mm" for axis in ("x", "y")
            ]
            cop_distance = math.nan
            if all(column in platform_values[side] for column in cop_columns):
                cop = np.column_stack(
                    [platform_values[side][column] for column in cop_columns]
                )
                valid_cop = valid_origin & np.isfinite(cop).all(axis=1)
                if valid_cop.any():
                    cop_distance = float(
                        np.median(
                            np.linalg.norm(
                                origins[side][valid_cop, :2] - cop[valid_cop], axis=1
                            )
                        )
                    )

            if np.isfinite(cop_distance) and np.isfinite(center_distance):
                score = 0.8 * cop_distance + 0.2 * center_distance
            elif np.isfinite(cop_distance):
                score = cop_distance
            elif np.isfinite(center_distance):
                score = center_distance
            else:
                details[side] = {"cop_distance_mm": None, "center_distance_mm": None}
                continue
            if side == original:
                score -= correlation_bonus_mm * max(0.0, min(1.0, correlation_value))
            if previous_side.get(platform) == side:
                score -= continuity_bonus_mm
            scores[side] = score
            details[side] = {
                "cop_distance_mm": cop_distance if np.isfinite(cop_distance) else None,
                "center_distance_mm": (
                    center_distance if np.isfinite(center_distance) else None
                ),
            }

        spatially_validated = False
        if scores:
            ordered = sorted(scores, key=scores.get)
            chosen = ordered[0]
            chosen_distances = details[chosen]
            chosen_cop_distance = chosen_distances["cop_distance_mm"]
            chosen_center_distance = chosen_distances["center_distance_mm"]
            spatial_distance = (
                float(chosen_cop_distance)
                if chosen_cop_distance is not None
                else (
                    float(chosen_center_distance)
                    if chosen_center_distance is not None
                    else math.inf
                )
            )
            # Never switch to the only geometrically available foot when that
            # foot is itself far away from the measured contact. Missing
            # markers on the originally matched foot are not evidence that the
            # opposite foot owns the platform contact.
            if chosen != original and spatial_distance > maximum_assignment_distance_mm:
                chosen = original
            if chosen in details:
                final_distances = details[chosen]
                final_primary_distance = (
                    final_distances["cop_distance_mm"]
                    if final_distances["cop_distance_mm"] is not None
                    else final_distances["center_distance_mm"]
                )
                spatially_validated = (
                    final_primary_distance is not None
                    and float(final_primary_distance) <= maximum_assignment_distance_mm
                )
            margin = (
                float(scores[ordered[1]] - scores[ordered[0]])
                if len(ordered) > 1
                else math.nan
            )
            spatial_events += 1
            if np.isfinite(margin):
                margins.append(margin)
        else:
            chosen = original
            margin = math.nan
            retained_without_geometry += 1
        if chosen in ("L", "R"):
            previous_side[platform] = chosen
            contact["matched_side"] = chosen
        contact["original_matched_side"] = original
        contact["assignment_method"] = "cop_mocap_center_continuity_correlation"
        contact["spatially_validated"] = spatially_validated
        if chosen != original:
            changed_events.append(
                {
                    "platform": platform,
                    "contact_start_s": float(contact["contact_start_s"]),
                    "contact_end_s": float(contact["contact_end_s"]),
                    "original_side": original,
                    "assigned_side": chosen,
                    "confidence_margin_mm": margin if np.isfinite(margin) else None,
                    "distances": details,
                    "original_correlation": (
                        correlation_value if np.isfinite(correlation_value) else None
                    ),
                }
            )
        reassigned.append(contact)

    diagnostics = {
        "method": "cop_mocap_center_continuity_correlation",
        "events": len(contacts),
        "events_with_spatial_evidence": spatial_events,
        "events_retained_without_geometry": retained_without_geometry,
        "changed_event_count": len(changed_events),
        "changed_events": changed_events,
        "confidence_margin_mm_median": float(np.median(margins)) if margins else None,
        "confidence_margin_mm_min": float(np.min(margins)) if margins else None,
        "correlation_tie_break_bonus_mm": correlation_bonus_mm,
        "continuity_bonus_mm": continuity_bonus_mm,
        "maximum_assignment_distance_mm": maximum_assignment_distance_mm,
    }
    return reassigned, diagnostics


def fused_contact_labels(
    times_by_side: dict[str, np.ndarray],
    origins: dict[str, np.ndarray],
    foot_heights: dict[str, tuple[np.ndarray, np.ndarray]],
    contacts: list[dict[str, object]],
    minimum_correlation: float,
    continuous_double_support: bool,
    continuous_interval: tuple[float, float] | None,
    airborne_foot_height_mm: float,
    airborne_minimum_duration_s: float,
    marker_reliable: dict[str, np.ndarray] | None = None,
    gauge_sums: dict[str, np.ndarray] | None = None,
    insole_veto_n: float = math.inf,
) -> tuple[
    dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]
]:
    """Fuse platform contacts, Foot height and the acquisition protocol (v3.7 rules).

    - Platform contacts provide the Measured labels (plate assignment unchanged).
    - Airborne (fused, #24): the foot's plate is empty (and not ambiguous), its
      Foot height is valid and above ``airborne_foot_height_mm``, its marker set
      is reliable (``marker_reliable``: toe visible or enough markers), and its
      insole gauge sum does not exceed ``insole_veto_n`` (a veto only: the insole
      never creates an Airborne label), all for at least
      ``airborne_minimum_duration_s``.
    - For protocol-defined double-support tasks, low-load holes in the 15 N
      platform detector are filled with the dominant plate inside the recorded
      active interval, and no sample of that interval is Airborne; apparent
      Foot height elevations there are only counted for QC.

    Returns the platform per sample, the Airborne mask, the protocol-filled mask
    (Measured from the protocol window, not from the contact detector) and the
    diagnostics.
    """
    platforms: dict[str, np.ndarray] = {}
    ambiguous_by_side: dict[str, np.ndarray] = {}
    airborne_by_side: dict[str, np.ndarray] = {}
    dominant: dict[str, int] = {}
    vetoed_by_side = {"L": 0, "R": 0}

    for side in ("L", "R"):
        platform, ambiguous = active_platforms(
            times_by_side[side], contacts, side, minimum_correlation
        )
        platforms[side] = platform
        ambiguous_by_side[side] = ambiguous
        dominant[side] = dominant_platform(contacts, side, minimum_correlation)
        height, height_valid = foot_heights[side]
        # NaN never passes: the valid flag gates the comparison explicitly.
        above = np.zeros(len(platform), dtype=bool)
        above[height_valid] = height[height_valid] > airborne_foot_height_mm
        candidate = (platform == 0) & ~ambiguous & height_valid & above
        if marker_reliable is not None:
            candidate &= marker_reliable[side]
        if gauge_sums is not None:
            vetoed = candidate & (gauge_sums[side] > insole_veto_n)
            vetoed_by_side[side] = int(vetoed.sum())
            candidate &= ~vetoed
        airborne_by_side[side] = sustained_true(
            candidate, times_by_side[side], airborne_minimum_duration_s
        )

    paired_time = (times_by_side["L"] + times_by_side["R"]) / 2.0
    protocol_filled = {
        side: np.zeros(len(paired_time), dtype=bool) for side in ("L", "R")
    }
    rescued = {"L": 0, "R": 0}
    protocol_disagreements = {"L": 0, "R": 0}
    protocol_start_s: float | None = None
    protocol_end_s: float | None = None
    if continuous_double_support and dominant["L"] and dominant["R"]:
        if continuous_interval is None:
            both_confirmed = (
                (platforms["L"] == dominant["L"])
                & (platforms["R"] == dominant["R"])
                & ~ambiguous_by_side["L"]
                & ~ambiguous_by_side["R"]
            )
            indexes = np.flatnonzero(both_confirmed)
            protocol_zone = np.zeros(len(paired_time), dtype=bool)
            if len(indexes):
                protocol_zone[indexes[0] : indexes[-1] + 1] = True
                protocol_start_s = float(paired_time[indexes[0]])
                protocol_end_s = float(paired_time[indexes[-1]])
        else:
            start, stop = continuous_interval
            protocol_zone = (paired_time >= start) & (paired_time < stop)
            protocol_start_s = float(start)
            protocol_end_s = float(stop)

        for side in ("L", "R"):
            frame_valid = np.isfinite(origins[side]).all(axis=1)
            protocol_disagreements[side] = int(
                np.sum(protocol_zone & airborne_by_side[side])
            )
            fill = (
                protocol_zone
                & (platforms[side] == 0)
                & ~ambiguous_by_side[side]
                & frame_valid
            )
            platforms[side][fill] = dominant[side]
            protocol_filled[side] = fill
            # The protocol is authoritative inside this phase. An apparent Foot
            # height elevation is retained as a QC count, not converted to zero GRFM.
            airborne_by_side[side][protocol_zone] = False
            # A run cut by the zone edge must still last the minimum duration.
            airborne_by_side[side] = sustained_true(
                airborne_by_side[side], times_by_side[side], airborne_minimum_duration_s
            )
            rescued[side] = int(fill.sum())

    diagnostics = {
        "method": "platform_foot_height_protocol_fusion",
        "dominant_platform_L": dominant["L"],
        "dominant_platform_R": dominant["R"],
        "airborne_foot_height_mm": airborne_foot_height_mm,
        "airborne_samples_L": int(airborne_by_side["L"].sum()),
        "airborne_samples_R": int(airborne_by_side["R"].sum()),
        "insole_veto_n": finite_or_none(insole_veto_n),
        "insole_vetoed_samples_L": vetoed_by_side["L"],
        "insole_vetoed_samples_R": vetoed_by_side["R"],
        "protocol_rescued_samples_L": rescued["L"],
        "protocol_rescued_samples_R": rescued["R"],
        "protocol_foot_height_disagreement_samples_L": protocol_disagreements["L"],
        "protocol_foot_height_disagreement_samples_R": protocol_disagreements["R"],
        "protocol_active_start_qtm_s": protocol_start_s,
        "protocol_active_end_qtm_s": protocol_end_s,
        "continuous_double_support": continuous_double_support,
    }
    return platforms, airborne_by_side, protocol_filled, diagnostics


def make_targets_native(
    times_by_side: dict[str, np.ndarray],
    foot_heights: dict[str, tuple[np.ndarray, np.ndarray]],
    rotations: dict[str, np.ndarray],
    origins: dict[str, np.ndarray],
    platform_values: dict[str, dict[str, np.ndarray]],
    moment_references: dict[int, np.ndarray],
    platform_centers: dict[int, np.ndarray],
    contacts: list[dict[str, object]],
    minimum_correlation: float,
    continuous_double_support: bool,
    continuous_interval: tuple[float, float] | None,
    airborne_foot_height_mm: float,
    airborne_minimum_duration_s: float,
    assignment_correlation_bonus_mm: float,
    assignment_continuity_bonus_mm: float,
    maximum_assignment_distance_mm: float,
    max_transport_reference_distance_mm: float,
    max_effective_lever_arm_mm: float,
    minimum_force_for_distant_contact_n: float,
    marker_reliable: dict[str, np.ndarray] | None = None,
    gauge_sums: dict[str, np.ndarray] | None = None,
    insole_veto_n: float = math.inf,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    sample_count = len(times_by_side["L"])
    y = np.zeros((sample_count, 12), dtype=np.float32)
    mask = np.zeros_like(y, dtype=bool)
    states: dict[str, np.ndarray] = {}
    reassigned_contacts, assignment_diagnostics = spatially_reassign_contacts(
        contacts,
        times_by_side,
        origins,
        platform_values,
        platform_centers,
        assignment_correlation_bonus_mm,
        assignment_continuity_bonus_mm,
        maximum_assignment_distance_mm,
    )
    (
        platforms_by_side,
        airborne_by_side,
        protocol_filled,
        fusion_diagnostics,
    ) = fused_contact_labels(
        times_by_side,
        origins,
        foot_heights,
        reassigned_contacts,
        minimum_correlation,
        continuous_double_support,
        continuous_interval,
        airborne_foot_height_mm,
        airborne_minimum_duration_s,
        marker_reliable,
        gauge_sums,
        insole_veto_n,
    )
    transport_reference_distance: dict[str, np.ndarray] = {
        side: np.full(sample_count, np.nan, dtype=np.float32) for side in ("L", "R")
    }
    effective_lever_arm: dict[str, np.ndarray] = {
        side: np.full(sample_count, np.nan, dtype=np.float32) for side in ("L", "R")
    }
    transport_violation: dict[str, np.ndarray] = {
        side: np.zeros(sample_count, dtype=bool) for side in ("L", "R")
    }
    transport_distance_warning: dict[str, np.ndarray] = {
        side: np.zeros(sample_count, dtype=bool) for side in ("L", "R")
    }

    for side_index, side in enumerate(("L", "R")):
        target_slice = slice(side_index * 6, side_index * 6 + 6)
        platform = platforms_by_side[side]
        airborne = airborne_by_side[side]
        state = np.full(sample_count, STATE_UNKNOWN, dtype=np.int8)
        state[(platform == 0) & airborne] = STATE_AIRBORNE
        frame_valid = np.isfinite(rotations[side]).all(axis=(1, 2)) & np.isfinite(
            origins[side]
        ).all(axis=1)

        for platform_number in sorted(moment_references):
            selected = (platform == platform_number) & frame_valid
            if not selected.any():
                continue
            values = platform_values[side]
            force_global = np.column_stack(
                [
                    values[f"force_{axis}_p{platform_number}_global_N"]
                    for axis in ("x", "y", "z")
                ]
            )[selected]
            moment_global = np.column_stack(
                [
                    values[f"moment_{axis}_p{platform_number}_global_Nmm"]
                    for axis in ("x", "y", "z")
                ]
            )[selected]
            # Transport du moment du centre de la surface de la plateforme P
            # vers l'origine du pied Q : M_Q = M_P + (P - Q) x F.
            reference_to_foot_mm = (
                moment_references[platform_number] - origins[side][selected]
            )
            moment_at_foot_global = moment_global + np.cross(
                reference_to_foot_mm, force_global
            )
            rotation = rotations[side][selected]
            force_foot = np.einsum("nij,ni->nj", rotation, force_global)
            moment_foot_nm = (
                np.einsum("nij,ni->nj", rotation, moment_at_foot_global) / 1000.0
            )
            selected_indexes = np.flatnonzero(selected)
            reference_distance_mm = np.linalg.norm(reference_to_foot_mm, axis=1)
            force_norm_squared = np.einsum("ni,ni->n", force_global, force_global)
            force_norm = np.sqrt(force_norm_squared)
            effective_mm = np.full(len(force_global), np.nan, dtype=np.float64)
            force_valid = force_norm_squared > 25.0
            effective_mm[force_valid] = (
                np.linalg.norm(
                    np.cross(force_global[force_valid], moment_at_foot_global[force_valid] / 1000.0),
                    axis=1,
                )
                / force_norm_squared[force_valid]
                * 1000.0
            )
            distance_warning = reference_distance_mm > max_transport_reference_distance_mm
            violation = distance_warning & (
                (
                    np.isfinite(effective_mm)
                    & (effective_mm > max_effective_lever_arm_mm)
                )
                | (force_norm < minimum_force_for_distant_contact_n)
            )
            transport_reference_distance[side][selected_indexes] = reference_distance_mm
            effective_lever_arm[side][selected_indexes] = effective_mm
            transport_violation[side][selected_indexes] = violation
            transport_distance_warning[side][selected_indexes] = distance_warning
            if violation.any():
                selected_indexes = selected_indexes[~violation]
                selected = np.zeros(sample_count, dtype=bool)
                selected[selected_indexes] = True
                force_foot = force_foot[~violation]
                moment_foot_nm = moment_foot_nm[~violation]
            y[selected, target_slice] = np.column_stack(
                [force_foot, moment_foot_nm]
            ).astype(np.float32)
            mask[selected, target_slice] = True
            state[selected] = STATE_PLATFORM

        rejected = transport_violation[side]
        state[rejected] = STATE_UNKNOWN
        platforms_by_side[side][rejected] = 0
        zero = state == STATE_AIRBORNE
        mask[zero, target_slice] = True
        states[side] = state

    same_platform = (
        (platforms_by_side["L"] > 0)
        & (platforms_by_side["L"] == platforms_by_side["R"])
    )
    if same_platform.any():
        mask[same_platform] = False
        states["L"][same_platform] = STATE_UNKNOWN
        states["R"][same_platform] = STATE_UNKNOWN
    label_sources: dict[str, np.ndarray] = {}
    for side in ("L", "R"):
        source = np.full(sample_count, LABEL_SOURCE_NONE, dtype=np.int8)
        measured = states[side] == STATE_PLATFORM
        source[measured & ~protocol_filled[side]] = LABEL_SOURCE_PLATE
        source[measured & protocol_filled[side]] = LABEL_SOURCE_PROTOCOL_WINDOW
        source[states[side] == STATE_AIRBORNE] = LABEL_SOURCE_FOOT_HEIGHT
        label_sources[side] = source
    extras = {
        "state_L": states["L"],
        "state_R": states["R"],
        "label_source_L": label_sources["L"],
        "label_source_R": label_sources["R"],
        "platform_L": platforms_by_side["L"],
        "platform_R": platforms_by_side["R"],
        "airborne_label_L": states["L"] == STATE_AIRBORNE,
        "airborne_label_R": states["R"] == STATE_AIRBORNE,
        "transport_reference_distance_mm_L": transport_reference_distance["L"],
        "transport_reference_distance_mm_R": transport_reference_distance["R"],
        "effective_lever_arm_mm_L": effective_lever_arm["L"],
        "effective_lever_arm_mm_R": effective_lever_arm["R"],
        "transport_qc_violation_L": transport_violation["L"],
        "transport_qc_violation_R": transport_violation["R"],
        "transport_distance_warning_L": transport_distance_warning["L"],
        "transport_distance_warning_R": transport_distance_warning["R"],
        "fusion_diagnostics": fusion_diagnostics,
        "assignment_diagnostics": assignment_diagnostics,
        "transport_qc": {
            "max_transport_reference_distance_mm": max_transport_reference_distance_mm,
            "max_effective_lever_arm_mm": max_effective_lever_arm_mm,
            "minimum_force_for_distant_contact_n": minimum_force_for_distant_contact_n,
            "masked_violation_samples_L": int(transport_violation["L"].sum()),
            "masked_violation_samples_R": int(transport_violation["R"].sum()),
            "distance_warning_samples_L": int(transport_distance_warning["L"].sum()),
            "distance_warning_samples_R": int(transport_distance_warning["R"].sum()),
            "maximum_reference_distance_mm_L": (
                float(np.nanmax(transport_reference_distance["L"]))
                if np.isfinite(transport_reference_distance["L"]).any()
                else None
            ),
            "maximum_reference_distance_mm_R": (
                float(np.nanmax(transport_reference_distance["R"]))
                if np.isfinite(transport_reference_distance["R"]).any()
                else None
            ),
            "maximum_effective_lever_arm_mm_L": (
                float(np.nanmax(effective_lever_arm["L"]))
                if np.isfinite(effective_lever_arm["L"]).any()
                else None
            ),
            "maximum_effective_lever_arm_mm_R": (
                float(np.nanmax(effective_lever_arm["R"]))
                if np.isfinite(effective_lever_arm["R"]).any()
                else None
            ),
        },
    }
    return y, mask, extras


def foot_height_summary(height: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    values = height[valid]
    return {
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
        "median_mm": float(np.median(values)) if len(values) else None,
        "q05_mm": float(np.percentile(values, 5)) if len(values) else None,
        "q95_mm": float(np.percentile(values, 95)) if len(values) else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manip_roots", type=Path, nargs="*")
    parser.add_argument(
        "--static-plate-fz",
        type=Path,
        default=None,
        help=(
            "CSV des Fz plateformes des essais statiques (extract_static_plate_fz.py, "
            "env c3d-io): source du Body weight; sans lui, les essais balance"
        ),
    )
    parser.add_argument(
        "--participant-sex",
        type=Path,
        default=DEFAULT_SEX_CONFIG,
        help="JSON Participant -> female|male, pour le sac nominal (+3/+6 kg)",
    )
    parser.add_argument(
        "--refresh-body-weights",
        type=Path,
        default=None,
        metavar="DATASET_DIR",
        help=(
            "ne reconstruit rien: recalcule seulement body_weights dans "
            "DATASET_DIR/metadata.json (valeurs balance existantes conservees)"
        ),
    )
    parser.add_argument(
        "--dataset-version",
        default=DATASET_VERSION,
        help=(
            "version du dataset prepare (3.x), enregistree dans metadata.json "
            f"(defaut: {DATASET_VERSION})"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "dossier du dataset prepare (defaut: "
            "DATA/prepared_npy_joint_masked_v<dataset-version>); un dossier contenant "
            "un dataset d'une autre version n'est jamais ecrit"
        ),
    )
    parser.add_argument("--pair-tolerance-ms", type=float, default=5.0)
    parser.add_argument("--frame-tolerance-ms", type=float, default=3.0)
    parser.add_argument("--segment-gap-ms", type=float, default=10.0)
    parser.add_argument("--platform-lowpass-hz", type=float, default=50.0)
    parser.add_argument(
        "--lateral-contact-threshold-n",
        type=float,
        default=15.0,
        help="seuil plateforme bas pour reconnaitre un appui lateral leger",
    )
    parser.add_argument(
        "--lateral-active-total-threshold-n",
        type=float,
        default=150.0,
        help="charge totale minimale pour rechercher la bascule laterale",
    )
    parser.add_argument(
        "--airborne-foot-height-mm",
        type=float,
        default=AIRBORNE_FOOT_HEIGHT_MM,
        help=(
            "Foot height minimale (mm) d'un pied Airborne, plateforme vide "
            f"(defaut {AIRBORNE_FOOT_HEIGHT_MM:g}, calibre sur les donnees)"
        ),
    )
    parser.add_argument(
        "--airborne-min-markers-without-toe",
        type=int,
        default=AIRBORNE_MIN_MARKERS_WITHOUT_TOE,
        help=(
            "marqueurs visibles minimum, sans le marqueur de l'orteil, pour que la "
            "Foot height puisse donner Airborne (orteil visible: toujours fiable; "
            f"defaut {AIRBORNE_MIN_MARKERS_WITHOUT_TOE})"
        ),
    )
    parser.add_argument(
        "--airborne-insole-veto-n",
        type=float,
        default=AIRBORNE_INSOLE_VETO_N,
        help=(
            "somme des jauges (N) au-dela de laquelle un pied ne peut etre Airborne "
            f"(veto seulement, devient Masked; defaut {AIRBORNE_INSOLE_VETO_N:g}, "
            "inf desactive)"
        ),
    )
    parser.add_argument(
        "--airborne-minimum-duration-ms",
        type=float,
        default=50.0,
        help="duree minimale au-dessus du seuil pour superviser un pied en l'air",
    )
    parser.add_argument("--min-assignment-correlation", type=float, default=0.0)
    parser.add_argument(
        "--assignment-correlation-bonus-mm",
        type=float,
        default=10.0,
        help="petit bonus spatial donne au cote choisi par correlation",
    )
    parser.add_argument(
        "--assignment-continuity-bonus-mm",
        type=float,
        default=15.0,
        help="bonus favorisant la continuite du pied sur une plateforme",
    )
    parser.add_argument(
        "--max-assignment-distance-mm",
        type=float,
        default=250.0,
        help=(
            "distance CoP-pied maximale pour autoriser une reaffectation; "
            "evite de choisir l'autre pied lorsque les marqueurs du premier manquent"
        ),
    )
    parser.add_argument(
        "--max-transport-reference-distance-mm",
        type=float,
        default=300.0,
        help="distance centre plateforme-pied declenchant le controle du transport",
    )
    parser.add_argument(
        "--max-effective-lever-arm-mm",
        type=float,
        default=200.0,
        help="bras de levier effectif maximal si la plateforme est aussi eloignee",
    )
    parser.add_argument(
        "--min-force-for-distant-contact-n",
        type=float,
        default=5.0,
        help=(
            "masque une association distante si la norme de force est aussi "
            "quasi nulle (bordure de contact non exploitable)"
        ),
    )
    parser.add_argument(
        "--foot-height-min-markers",
        type=int,
        default=MIN_VISIBLE_MARKERS,
        help="marqueurs visibles minimum pour une Foot height valide",
    )
    parser.add_argument(
        "--fallback-floor-z-mm",
        type=float,
        default=DEFAULT_FALLBACK_FLOOR_Z_MM,
        help=(
            "niveau du sol hors plateformes (z global, mm) pour un sujet dont aucun "
            "essai ne montre de pied hors plateformes"
        ),
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        help="limite optionnellement la preparation a ces dossiers sujets",
    )
    parser.add_argument(
        "--trials",
        nargs="+",
        help="limite optionnellement la preparation a ces noms d'essais",
    )
    parser.add_argument(
        "--exclude-subjects",
        nargs="*",
        default=["pilot1", "pilot2"],
        help="sujets ignores meme si synchronized_v2 existe",
    )
    parser.add_argument(
        "--lateral-split-config",
        type=Path,
        default=Path(__file__).with_name("lateral_split_trials.json"),
        help=(
            "liste JSON des essais a separer automatiquement en phases "
            "lateral_2supports et lateral_1support"
        ),
    )
    parser.add_argument(
        "--exclude-config",
        type=Path,
        default=None,
        help=(
            "liste JSON des essais a exclure explicitement de la preparation "
            "(defaut: excluded_trials.json a cote du script s'il existe, sinon aucune)"
        ),
    )
    args = parser.parse_args(argv)
    try:
        sex = load_sex(args.participant_sex)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if args.refresh_body_weights is not None:
        if not (args.refresh_body_weights / "metadata.json").is_file():
            parser.error(f"{args.refresh_body_weights}/metadata.json absent")
        weights = refresh_body_weights(
            args.refresh_body_weights, args.static_plate_fz, args.participant_sex
        )
        for participant, entry in weights["participants"].items():
            print(f"BODY WEIGHT {participant}: {entry['body_weight_N']:.1f} N "
                  f"({entry['source']})")
        return 0
    if not args.manip_roots:
        parser.error("au moins un dossier manip est requis")
    if args.static_plate_fz is None:
        print("ATTENTION: sans --static-plate-fz, Body weight = essais balance (#16)")
    if args.exclude_config is None and DEFAULT_EXCLUDE_CONFIG.is_file():
        args.exclude_config = DEFAULT_EXCLUDE_CONFIG
    if not DATASET_VERSION_PATTERN.fullmatch(args.dataset_version):
        parser.error(
            f"--dataset-version {args.dataset_version!r}: version 3.x attendue "
            "(le moteur d'entrainement exige le prefixe prepared_joint_masked_v3)"
        )
    version = pipeline_version(args.dataset_version)
    if args.output_dir is None:
        args.output_dir = Path(f"DATA/prepared_npy_joint_masked_v{args.dataset_version}")
    existing = existing_dataset_version(args.output_dir)
    if existing is not None and existing != version:
        parser.error(
            f"{args.output_dir} contient deja le dataset {existing}; "
            f"choisir un autre --output-dir pour {version}"
        )
    if args.foot_height_min_markers < 1:
        parser.error("--foot-height-min-markers doit etre >= 1")
    if args.airborne_min_markers_without_toe < 1:
        parser.error("--airborne-min-markers-without-toe doit etre >= 1")
    if not args.airborne_insole_veto_n > 0:
        parser.error("--airborne-insole-veto-n doit etre > 0")
    if min(
        args.pair_tolerance_ms,
        args.frame_tolerance_ms,
        args.segment_gap_ms,
        args.airborne_foot_height_mm,
        args.airborne_minimum_duration_ms,
        args.max_transport_reference_distance_mm,
        args.max_effective_lever_arm_mm,
    ) <= 0:
        parser.error("les tolerances temporelles doivent etre positives")
    try:
        split_batches = load_batch_list(args.lateral_split_config)
        excluded_batches = load_batch_list(args.exclude_config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batches: list[str] = []
    summaries: list[dict[str, object]] = []
    final_feature_names: list[str] | None = None
    static_batches: list[str] = []
    plate_fz_by_subject: dict[str, list[tuple[str, np.ndarray]]] = {}

    selected_subjects = set(args.subjects or [])
    selected_trials = set(args.trials or [])
    discovered = discover_synchronized_trials(args.manip_roots)
    # Foot height is calibrated per Subject on all its synchronised trials, whatever
    # the --trials filter, so building one trial gives the same values as a full build.
    trials_by_subject: dict[str, list[Path]] = {}
    for _, trial in discovered:
        trials_by_subject.setdefault(trial.parent.name, []).append(trial)
    calibrations: dict[str, FootHeightCalibration] = {}
    for batch, trial in discovered:
        if selected_subjects and trial.parent.name not in selected_subjects:
            continue
        if selected_trials and trial.name not in selected_trials and batch not in selected_trials:
            continue
        if trial.parent.name in set(args.exclude_subjects):
            print(f"IGNORE {batch}: sujet exclu")
            continue
        if batch in excluded_batches:
            print(f"IGNORE {batch}: liste d'exclusion explicite")
            continue
        if is_static_trial(trial.name):
            print(f"IGNORE {batch}: essai statique (hors dataset)")
            static_batches.append(batch)
            continue
        sync = trial / "synchronized_v2"
        report = json.loads((sync / "rapport_synchronisation.json").read_text(encoding="utf-8"))
        if report.get("status") != "OK" or report["v2_export_checks"]["overall_status"] != "OK":
            print(f"IGNORE {batch}: rapport v2 non OK")
            continue
        lost_packets = int(report["left_insole"]["missing_packets_from_counter"]) + int(
            report["right_insole"]["missing_packets_from_counter"]
        )
        if lost_packets:
            loss_assessment = report.get("packet_loss_assessment", {})
            if loss_assessment.get("status") != "ACCEPTED_OUTSIDE_CONTACT":
                print(f"IGNORE {batch}: perte de paquets non validee hors contact")
                continue

        insoles = pd.read_csv(sync / "semelles_synchronisees.csv")
        platforms = pd.read_csv(sync / "plateformes_synchronisees.csv")
        mocap = pd.read_csv(sync / "mocap_synchronisee.csv")
        split_info: LateralSplit | None = None
        if batch in split_batches:
            try:
                split_info = detect_lateral_support_switch(
                    platforms,
                    contact_threshold_n=args.lateral_contact_threshold_n,
                    active_total_threshold_n=args.lateral_active_total_threshold_n,
                )
            except ValueError as exc:
                print(f"IGNORE {batch}: separation laterale impossible: {exc}")
                continue
        try:
            native = {side: pivot_native_insole(insoles, side) for side in ("L", "R")}
            foot_frames = {side: rigid_foot_frames(mocap, side) for side in ("L", "R")}
        except ValueError as exc:
            print(f"IGNORE {batch}: {exc}")
            continue

        # The two ESP32 clocks are independent.  Pairing their clocks rebased at
        # Signal=1 can silently associate samples acquired at different physical
        # instants when the side-specific QTM offsets differ (up to about 80 ms
        # in this dataset).  synchronized_v2 already estimated one offset per
        # side from the insole/platform contacts, so the one-to-one pairing must
        # be performed on that common physical QTM clock.
        pair_l, pair_r = pair_native_packets(
            native["L"].qtm_time_s,
            native["R"].qtm_time_s,
            args.pair_tolerance_ms / 1000.0,
        )
        if len(pair_l) < 10:
            print(f"IGNORE {batch}: moins de 10 paires natives")
            continue
        pair_indexes = {"L": pair_l, "R": pair_r}
        x_by_side = {side: native[side].x[pair_indexes[side]] for side in ("L", "R")}
        qtm_times = {
            side: native[side].qtm_time_s[pair_indexes[side]] for side in ("L", "R")
        }
        receiver_times = {
            side: native[side].receiver_time_s[pair_indexes[side]] for side in ("L", "R")
        }
        sample_times = (qtm_times["L"] + qtm_times["R"]) / 2.0
        sample_times -= sample_times[0]

        mocap_times = mocap["time_qtm_s"].to_numpy(dtype=np.float64)
        frame_indexes = {
            side: nearest_indices(
                mocap_times, qtm_times[side], args.frame_tolerance_ms / 1000.0
            )
            for side in ("L", "R")
        }
        rotations: dict[str, np.ndarray] = {}
        origins: dict[str, np.ndarray] = {}
        for side in ("L", "R"):
            rotations[side] = np.full((len(sample_times), 3, 3), np.nan, dtype=np.float64)
            origins[side] = np.full((len(sample_times), 3), np.nan, dtype=np.float64)
            valid = frame_indexes[side] >= 0
            rotations[side][valid] = foot_frames[side][0][frame_indexes[side][valid]]
            origins[side][valid] = foot_frames[side][1][frame_indexes[side][valid]]

        subject = trial.parent.name
        if subject not in calibrations:
            calibrations[subject] = calibrate_subject(
                subject, trials_by_subject[subject], args.fallback_floor_z_mm
            )
        calibration = calibrations[subject]
        ground = Ground(plate_surfaces(report["platform_geometry"]), calibration.floor)
        foot_heights: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        visible_markers: dict[str, np.ndarray] = {}
        toe_visible: dict[str, np.ndarray] = {}
        marker_reliable: dict[str, np.ndarray] = {}
        gauge_sums: dict[str, np.ndarray] = {}
        for side in ("L", "R"):
            # Markers at the same mocap frame as the foot frame of each sample.
            valid = frame_indexes[side] >= 0
            sample_markers = {}
            for name, positions in marker_arrays(mocap, side).items():
                at_samples = np.full((len(sample_times), 3), np.nan, dtype=np.float64)
                at_samples[valid] = positions[frame_indexes[side][valid]]
                sample_markers[name] = at_samples
            foot_heights[side] = foot_height(
                sample_markers,
                calibration.offsets_mm.get(side, {}),
                ground,
                args.foot_height_min_markers,
            )
            visible_markers[side], toe_visible[side] = marker_visibility(
                sample_markers, calibration.offsets_mm.get(side, {})
            )
            marker_reliable[side] = toe_visible[side] | (
                visible_markers[side] >= args.airborne_min_markers_without_toe
            )
            gauges = [
                index
                for index, name in enumerate(native[side].feature_names)
                if "_force_" in f"_{name}"
            ]
            gauge_sums[side] = x_by_side[side][:, gauges].astype(np.float64).sum(axis=1)

        platform_values = {
            side: filter_and_interpolate_platform(
                platforms, qtm_times[side], args.platform_lowpass_hz
            )
            for side in ("L", "R")
        }
        moment_references = {
            int(item["platform"]): np.asarray(
                item.get(
                    "moment_reference_global_mm",
                    item.get("surface_center_global_mm", item["origin_global_mm"]),
                ),
                dtype=np.float64,
            )
            for item in report["platform_geometry"]
        }
        platform_centers = {
            int(item["platform"]): np.asarray(
                item.get("surface_center_global_mm", item["origin_global_mm"]),
                dtype=np.float64,
            )
            for item in report["platform_geometry"]
        }
        names_by_side = {side: native[side].feature_names for side in ("L", "R")}
        continuous_double_support = is_continuous_double_support_task(trial.name)
        continuous_interval = (
            (split_info.active_start_qtm_s, split_info.switch_qtm_s)
            if split_info is not None
            else None
        )
        y, target_mask, extras = make_targets_native(
            qtm_times,
            foot_heights,
            rotations,
            origins,
            platform_values,
            moment_references,
            platform_centers,
            report["contact_assignments"],
            args.min_assignment_correlation,
            continuous_double_support,
            continuous_interval,
            args.airborne_foot_height_mm,
            args.airborne_minimum_duration_ms / 1000.0,
            args.assignment_correlation_bonus_mm,
            args.assignment_continuity_bonus_mm,
            args.max_assignment_distance_mm,
            args.max_transport_reference_distance_mm,
            args.max_effective_lever_arm_mm,
            args.min_force_for_distant_contact_n,
            marker_reliable,
            gauge_sums,
            args.airborne_insole_veto_n,
        )
        x = np.concatenate([x_by_side["L"], x_by_side["R"]], axis=1).astype(np.float32)
        feature_names = [f"L_{name}" for name in names_by_side["L"]] + [
            f"R_{name}" for name in names_by_side["R"]
        ]
        if final_feature_names is None:
            final_feature_names = feature_names
        elif final_feature_names != feature_names:
            raise ValueError(f"ordre de features incoherent pour {batch}")

        gaps = np.diff(sample_times)
        segment_id = np.zeros(len(sample_times), dtype=np.int64)
        segment_id[1:] = np.cumsum(gaps > args.segment_gap_ms / 1000.0)
        frame_valid = {
            side: np.isfinite(rotations[side]).all(axis=(1, 2)) for side in ("L", "R")
        }
        pair_error_ms = np.abs(qtm_times["L"] - qtm_times["R"]) * 1000.0
        arrays = {
            "X": x,
            "y": y,
            "target_mask": target_mask,
            "target_time_s": sample_times,
            "source_trial_time_s": sample_times,
            "target_time_qtm_s_L": qtm_times["L"],
            "target_time_qtm_s_R": qtm_times["R"],
            "insole_time_s_L": receiver_times["L"] - receiver_times["L"][0],
            "insole_time_s_R": receiver_times["R"] - receiver_times["R"][0],
            "packet_timestamp_us_L": native["L"].packet_timestamp_us[pair_l],
            "packet_timestamp_us_R": native["R"].packet_timestamp_us[pair_r],
            # Input-quality diagnostics only; saturated values remain valid
            # deployment inputs because the same sensor limits exist at runtime.
            "force_input_saturated_L": native["L"].force_saturated[pair_l],
            "force_input_saturated_R": native["R"].force_saturated[pair_r],
            "accelerometer_input_saturated_L": native[
                "L"
            ].accelerometer_saturated[pair_l],
            "accelerometer_input_saturated_R": native[
                "R"
            ].accelerometer_saturated[pair_r],
            "gyroscope_input_saturated_L": native["L"].gyroscope_saturated[
                pair_l
            ],
            "gyroscope_input_saturated_R": native["R"].gyroscope_saturated[
                pair_r
            ],
            "segment_id": segment_id,
            "state_L": extras["state_L"],
            "state_R": extras["state_R"],
            # Supervised offline labels. They are never runtime inputs.
            "airborne_label_L": extras["airborne_label_L"],
            "airborne_label_R": extras["airborne_label_R"],
            # Per-sample label source (metadata labelling_rules.label_source_codes).
            "label_source_L": extras["label_source_L"],
            "label_source_R": extras["label_source_R"],
            "platform_L": extras["platform_L"],
            "platform_R": extras["platform_R"],
            "transport_reference_distance_mm_L": extras[
                "transport_reference_distance_mm_L"
            ],
            "transport_reference_distance_mm_R": extras[
                "transport_reference_distance_mm_R"
            ],
            "effective_lever_arm_mm_L": extras["effective_lever_arm_mm_L"],
            "effective_lever_arm_mm_R": extras["effective_lever_arm_mm_R"],
            "transport_qc_violation_L": extras["transport_qc_violation_L"],
            "transport_qc_violation_R": extras["transport_qc_violation_R"],
            "transport_distance_warning_L": extras[
                "transport_distance_warning_L"
            ],
            "transport_distance_warning_R": extras[
                "transport_distance_warning_R"
            ],
            "foot_frame_valid_L": frame_valid["L"],
            "foot_frame_valid_R": frame_valid["R"],
            "foot_height_mm_L": foot_heights["L"][0].astype(np.float32),
            "foot_height_mm_R": foot_heights["R"][0].astype(np.float32),
            "foot_height_valid_L": foot_heights["L"][1],
            "foot_height_valid_R": foot_heights["R"][1],
            # Airborne evidence diagnostics (#24), offline labels only.
            "foot_visible_markers_L": visible_markers["L"],
            "foot_visible_markers_R": visible_markers["R"],
            "toe_marker_visible_L": toe_visible["L"],
            "toe_marker_visible_R": toe_visible["R"],
            "insole_gauge_sum_n_L": gauge_sums["L"].astype(np.float32),
            "insole_gauge_sum_n_R": gauge_sums["R"].astype(np.float32),
        }
        # Total vertical load on both plates, for the Subject's body weight only
        # (not stored in the NPZ). Resampled on the left insole clock.
        total_plate_fz = (
            platform_values["L"]["force_z_p1_global_N"]
            + platform_values["L"]["force_z_p2_global_N"]
        )

        phases: list[tuple[str, str | None, np.ndarray]]
        if split_info is not None:
            paired_qtm_time = (qtm_times["L"] + qtm_times["R"]) / 2.0
            # Setup tails are not part of either requested movement phase.
            before = (
                (paired_qtm_time >= split_info.active_start_qtm_s)
                & (paired_qtm_time < split_info.switch_qtm_s)
            )
            after = (
                (paired_qtm_time >= split_info.switch_qtm_s)
                & (paired_qtm_time <= split_info.active_end_qtm_s)
            )
            if before.sum() < 10 or after.sum() < 10:
                print(f"IGNORE {batch}: une phase laterale contient moins de 10 paires")
                continue
            phases = [
                (f"{batch}__lateral_2supports", "lateral_2supports", before),
                (f"{batch}__lateral_1support", "lateral_1support", after),
            ]
            old_unsplit = args.output_dir / f"{batch}.npz"
            if old_unsplit.exists():
                old_unsplit.unlink()
        else:
            fusion = extras["fusion_diagnostics"]
            if continuous_double_support and fusion["protocol_active_start_qtm_s"] is not None:
                paired_qtm_time = (qtm_times["L"] + qtm_times["R"]) / 2.0
                active = (
                    (paired_qtm_time >= float(fusion["protocol_active_start_qtm_s"]))
                    & (paired_qtm_time <= float(fusion["protocol_active_end_qtm_s"]))
                )
                phases = [(batch, "continuous_2supports", active)]
            else:
                phases = [(batch, None, np.ones(len(x), dtype=bool))]

        for output_batch, task_phase, selected in phases:
            part = {name: values[selected].copy() for name, values in arrays.items()}
            part["target_time_s"] -= part["target_time_s"][0]
            part["insole_time_s_L"] -= part["insole_time_s_L"][0]
            part["insole_time_s_R"] -= part["insole_time_s_R"][0]
            part["segment_id"] -= part["segment_id"].min()
            part_gaps = np.diff(part["target_time_s"])
            np.savez_compressed(args.output_dir / f"{output_batch}.npz", **part)
            if task_family(trial.name, task_phase) == BODY_WEIGHT_SOURCE_FAMILY:
                both_measured = (part["state_L"] == STATE_PLATFORM) & (
                    part["state_R"] == STATE_PLATFORM
                )
                plate_fz_by_subject.setdefault(trial.parent.name, []).append(
                    (output_batch, total_plate_fz[selected][both_measured])
                )

            measured = int(
                np.sum(part["state_L"] == STATE_PLATFORM)
                + np.sum(part["state_R"] == STATE_PLATFORM)
            )
            airborne = int(
                np.sum(part["state_L"] == STATE_AIRBORNE)
                + np.sum(part["state_R"] == STATE_AIRBORNE)
            )
            unknown = int(
                np.sum(part["state_L"] == STATE_UNKNOWN)
                + np.sum(part["state_R"] == STATE_UNKNOWN)
            )
            summary = {
                "batch": output_batch,
                "source_batch": batch,
                "evaluation_group": batch,
                "task_phase": task_phase,
                "task_family": task_family(trial.name, task_phase),
                "source_trial_start_s": float(part["source_trial_time_s"][0]),
                "source_trial_end_s": float(part["source_trial_time_s"][-1]),
                "samples": int(selected.sum()),
                "paired_fraction_of_shorter_stream": float(
                    len(x) / min(len(native["L"].x), len(native["R"].x))
                ),
                "median_rate_hz": float(1.0 / np.median(part_gaps)),
                "max_gap_ms": float(part_gaps.max() * 1000.0),
                "max_pair_error_ms": float(pair_error_ms[selected].max()),
                "segments": int(part["segment_id"].max() + 1),
                "force_input_saturated_samples": int(
                    part["force_input_saturated_L"].sum()
                    + part["force_input_saturated_R"].sum()
                ),
                "accelerometer_input_saturated_samples": int(
                    part["accelerometer_input_saturated_L"].sum()
                    + part["accelerometer_input_saturated_R"].sum()
                ),
                "gyroscope_input_saturated_samples": int(
                    part["gyroscope_input_saturated_L"].sum()
                    + part["gyroscope_input_saturated_R"].sum()
                ),
                "measured_foot_samples": measured,
                "airborne_zero_foot_samples": airborne,
                "masked_unknown_foot_samples": unknown,
                "valid_target_fraction": float(part["target_mask"].mean()),
                "foot_frame_L": foot_frames["L"][2],
                "foot_frame_R": foot_frames["R"][2],
                **{
                    f"foot_height_{side}": foot_height_summary(
                        part[f"foot_height_mm_{side}"], part[f"foot_height_valid_{side}"]
                    )
                    for side in ("L", "R")
                },
                # Foot samples per label source (per foot), in this batch.
                "label_sources": {
                    side: {
                        name: int(np.sum(part[f"label_source_{side}"] == code))
                        for code, name in LABEL_SOURCE_CODES.items()
                    }
                    for side in ("L", "R")
                },
                "contact_fusion": extras["fusion_diagnostics"],
                "platform_assignment": extras["assignment_diagnostics"],
                "moment_transport_qc": extras["transport_qc"],
            }
            if split_info is not None:
                summary["lateral_switch_qtm_s"] = split_info.switch_qtm_s
                summary["lateral_switch_trial_s"] = float(
                    split_info.switch_qtm_s
                    - (qtm_times["L"][0] + qtm_times["R"][0]) / 2.0
                )
                summary["lateral_double_support_before"] = split_info.double_support_before
                summary["lateral_single_support_after"] = split_info.single_support_after
                summary["lateral_split_score"] = split_info.score
            summaries.append(summary)
            batches.append(output_batch)
            print(
                f"{output_batch}: {len(part['X'])} paires, "
                f"{summary['median_rate_hz']:.2f} Hz, segments={summary['segments']}, "
                f"cibles valides={100 * part['target_mask'].mean():.1f}%"
            )

    if not batches:
        print("Aucun essai v2 exploitable.", file=sys.stderr)
        return 1

    targets = [f"{axis}_L" for axis in TARGET_AXES] + [
        f"{axis}_R" for axis in TARGET_AXES
    ]
    metadata = {
        "pipeline_version": version,
        "dataset_version": args.dataset_version,
        "source_roots": [str(path.resolve()) for path in args.manip_roots],
        "source_synchronization": "synchronized_v2 (labels only for C3D/mocap)",
        "batches": batches,
        "joint": True,
        "sample_grid": "native_one_to_one_paired_on_side_corrected_qtm_clock",
        "pairing_note": (
            "left/right packets remain native and are not interpolated; they are "
            "paired after applying each side's synchronized_v2 physical-time offset"
        ),
        "deployment_inputs": "two_insoles_only_no_mocap_no_force_platform",
        "input_saturation_qc": {
            "force_raw_max_counts": 16383,
            "imu_raw_limit_counts": 32767,
            "imu_detection_threshold_abs_counts": 32760,
            "action": (
                "diagnostic flags are retained; samples are not masked because "
                "the same physical sensor saturation can occur at deployment"
            ),
        },
        "feature_names": final_feature_names,
        "feature_count": len(final_feature_names or []),
        "targets": targets,
        "target_units": {
            target: ("N" if target.startswith("F") else "N.m") for target in targets
        },
        "target_frame": "each_foot_frame_about_rigid_cluster_origin",
        "platform_wrench_convention": {
            "global_extraction": "ezc3d_force_platform_filter",
            "input_moment_reference": "force_platform_working_surface_center",
            "transport_to_foot": "M_foot = M_surface + (r_surface - r_foot) x F",
            "foot_projection": "global_to_foot_rotation_transpose",
            "assignment": "CoP/mocap proximity + platform centre + temporal continuity; gauge correlation tie-break only",
            "transport_qc": (
                "mask when both platform-reference distance and effective force-line lever arm exceed thresholds"
            ),
        },
        "target_frame_runtime_note": (
            "the model directly predicts this learned convention; fixed marker/sensor "
            "placement is assumed and mocap is not a runtime input"
        ),
        "labelling_rules": {
            "version": LABELLING_RULES_TAG,
            "label_source_npz_keys": ["label_source_L", "label_source_R"],
            "label_source_dtype": "int8",
            "label_source_codes": {
                str(code): name for code, name in LABEL_SOURCE_CODES.items()
            },
            # The insole gauges are read by one rule only: the Airborne veto (#24).
            # They never create a label, so no label comes from the model's input
            # alone and no zero target is trained where the gauges read a load.
            "state_rules_read_insole_channels": True,
            "insole_channel_role": "airborne_veto_only",
            "states": {
                "measured": {
                    "rule": (
                        "the foot's plate detects contact: inside a contact of the "
                        "synchronisation report's contact_assignments (15 N platform "
                        "threshold) assigned to that foot, with a valid foot frame and "
                        "a moment transport passing the QC; in a continuous task, also "
                        "a plate hole filled by the protocol window"
                    ),
                    "label_source": list(MEASURED_LABEL_SOURCES),
                },
                "airborne": {
                    "rule": (
                        "fused rule (#24), on every sample of a run lasting at least "
                        "minimum_duration_ms (and at least 2 samples): the foot's "
                        "plate is empty (no contact and not ambiguous; a plate that "
                        "detects contact is never Airborne), its Foot height is valid "
                        "and above foot_height_threshold_mm, its marker set is "
                        "reliable (marker_reliability), and its insole gauge sum does "
                        "not exceed insole_veto.gauge_sum_threshold_n; target GRFM = 0"
                    ),
                    "label_source": LABEL_SOURCE_CODES[LABEL_SOURCE_FOOT_HEIGHT],
                    "marker_reliability": {
                        "rule": (
                            "the toe marker of the foot is visible, or at least "
                            "min_visible_markers_without_toe of its calibrated markers "
                            "are visible; otherwise Foot height cannot make the foot "
                            "Airborne (at walking push-off the toe and lateral markers "
                            "are hidden while the foot is loaded on its toe, and heel + "
                            "medial alone read a high Foot height)"
                        ),
                        "toe_marker_visible_suffices": True,
                        "min_visible_markers_without_toe": (
                            args.airborne_min_markers_without_toe
                        ),
                        "npz_keys": [
                            "foot_visible_markers_L", "foot_visible_markers_R",
                            "toe_marker_visible_L", "toe_marker_visible_R",
                        ],
                        "calibration": (
                            AIRBORNE_MARKER_CALIBRATION
                            if args.airborne_min_markers_without_toe
                            == AIRBORNE_MIN_MARKERS_WITHOUT_TOE
                            else "set on the command line "
                            "(--airborne-min-markers-without-toe)"
                        ),
                    },
                    "insole_veto": {
                        "rule": (
                            "a foot whose insole gauge sum (its 15 *_force_* channels, "
                            "N) exceeds gauge_sum_threshold_n cannot be Airborne and "
                            "is Masked; the insole acts as a veto only and never "
                            "creates an Airborne (or any other) label"
                        ),
                        "gauge_sum_threshold_n": finite_or_none(args.airborne_insole_veto_n),
                        "reads_insole_channels": True,
                        "creates_airborne_labels": False,
                        "vetoed_state": "masked",
                        "npz_keys": ["insole_gauge_sum_n_L", "insole_gauge_sum_n_R"],
                        "calibration": (
                            AIRBORNE_VETO_CALIBRATION
                            if args.airborne_insole_veto_n == AIRBORNE_INSOLE_VETO_N
                            else "set on the command line (--airborne-insole-veto-n)"
                        ),
                    },
                    "foot_height_threshold_mm": args.airborne_foot_height_mm,
                    "minimum_duration_ms": args.airborne_minimum_duration_ms,
                    "foot_height_npz_keys": [
                        "foot_height_mm_L", "foot_height_mm_R",
                        "foot_height_valid_L", "foot_height_valid_R",
                    ],
                    "threshold_calibration": (
                        AIRBORNE_THRESHOLD_CALIBRATION
                        if args.airborne_foot_height_mm == AIRBORNE_FOOT_HEIGHT_MM
                        else "set on the command line (--airborne-foot-height-mm)"
                    ),
                },
                "masked": {
                    "rule": (
                        "everything else: a foot on the ground off the plates, a plate "
                        "empty with Foot height invalid or not above the threshold for "
                        "long enough, an unreliable marker set, a gauge sum above the "
                        "insole veto, an ambiguous plate (two contacts), both feet on "
                        "the same plate, an invalid foot frame on a plate, or a moment "
                        "transport QC violation; excluded from training and evaluation"
                    ),
                    "label_source": "none",
                },
            },
            "plate_assignment": (
                "unchanged from v3.6: each report contact goes to the foot whose "
                "marker-cluster origin is closest to the plate CoP (0.8) and plate "
                "centre (0.2), with a temporal-continuity bonus "
                f"({args.assignment_continuity_bonus_mm:g} mm), a switch refused beyond "
                f"{args.max_assignment_distance_mm:g} mm, and the synchronisation "
                "report's insole/plate correlation used only as a tie-break bonus of at "
                f"most {args.assignment_correlation_bonus_mm:g} mm; contacts that are "
                "not spatially validated and have a correlation below "
                f"{args.min_assignment_correlation:g} are dropped"
            ),
            "protocol_windows": {
                "continuous_tasks": ["balance", "squat", "hulahoop", "lateral_2supports"],
                "window": (
                    "from the first to the last sample with both feet confirmed on their "
                    "dominant plates (the active interval before the switch for a split "
                    "lateral trial); continuous trials are cropped to it"
                ),
                "action": (
                    "plate holes (no detected contact, valid foot frame) are filled with "
                    "the dominant plate (label source protocol_window) and no sample is "
                    "Airborne; Foot height elevations there are counted in "
                    "contact_fusion.protocol_foot_height_disagreement_samples_*"
                ),
                "continuous_tasks_have_no_airborne": True,
            },
            "removed_v36_rules": {
                "gauge_sum_hysteresis": (
                    "v3.6 jump and lateral single-support rescues made a foot Airborne "
                    "from its insole gauge sum (70/100 N hysteresis); removed, the "
                    "insole never creates a label (since #24 it only vetoes Airborne)"
                ),
                "cluster_origin_height": (
                    "v3.6 made a foot Airborne when its marker-cluster origin rose 20 mm "
                    "above its on-plate q99; removed, the origin rises at heel-off"
                ),
            },
            "runtime_note": (
                "mocap and force platforms supervise training only; the model must "
                "learn zero GRFM from insole inputs"
            ),
        },
        "lateral_split": {
            "config": str(args.lateral_split_config.resolve()),
            "requested_source_batches": sorted(split_batches),
            "method": "offline_change_point_from_two_force_platform_vertical_loads",
            "contact_threshold_n": args.lateral_contact_threshold_n,
            "active_total_threshold_n": args.lateral_active_total_threshold_n,
            "minimum_phase_s": 8.0,
            "note": "platforms are used only to label and split training data, never at runtime",
        },
        "explicitly_excluded_batches": sorted(excluded_batches),
        "foot_height": {
            "npz_keys": {
                "height": "foot_height_mm_L / foot_height_mm_R (float32, NaN where invalid)",
                "valid": "foot_height_valid_L / foot_height_valid_R (bool)",
            },
            "unit": "mm",
            "definition": (
                "per foot and per sample: min over the foot's visible markers of "
                "(marker z - marker offset - ground level under that marker), mocap "
                "global frame (z up), markers taken at the same mocap frame as the "
                "foot frame of the sample"
            ),
            "validity": (
                f"invalid where fewer than {args.foot_height_min_markers} markers of "
                "that foot (with a calibrated offset) are visible, or where no mocap "
                "frame lies within the frame tolerance"
            ),
            "min_visible_markers": args.foot_height_min_markers,
            "marker_sets": "all markers of each foot (4 on the left, 5 on the right)",
            "ground_level_source": {
                "over_a_plate": (
                    "plate top plane fitted through corners_global_mm of the "
                    "synchronisation report's platform_geometry, used where the "
                    "marker's xy lies inside that plate footprint"
                ),
                "elsewhere": (
                    "floor plane per Subject: stance level (histogram mode per 250 mm "
                    "xy cell) of the offset-corrected lowest marker z over the frames "
                    "where the foot's visible markers are all off the plates, fitted "
                    "with a plane (slope only along an axis spanned by >= 1 m of data); "
                    "fallback_floor_z_mm when a Subject never leaves the plates"
                ),
                "fallback_floor_z_mm": args.fallback_floor_z_mm,
            },
            "marker_offset": (
                "per Subject, foot and marker: median of (marker z - plate top) over "
                "the mocap frames inside that foot's plate contacts (report "
                "contact_assignments) with the marker over that plate, pooled over "
                "all the Subject's synchronised trials; it removes the marker's height "
                "above the sole, so a flat foot reads ~0 and heel-off with the toes "
                "down stays ~0; at push-off the toe marker can drop below its "
                "flat-stance height (shoe toe spring), so values down to ~-15 mm occur"
            ),
            "subjects": {
                subject: calibrations[subject].summary() for subject in sorted(calibrations)
            },
        },
        "static_trials_excluded": sorted(static_batches),
        "body_weights": body_weights_block(
            sorted({batch.split("_", 1)[0] for batch in batches}),
            subject_body_weights(plate_fz_by_subject),
            static_standing_weights(args.static_plate_fz),
            sex,
            (
                "per Subject, median total plate Fz (force_z_p1 + force_z_p2, global, "
                f"{args.platform_lowpass_hz:g} Hz low-pass) over the samples of its "
                "balance trials (protocol active window) where both feet are Measured; "
                "samples of all its balance trials pooled"
            ),
            str(args.static_plate_fz) if args.static_plate_fz else None,
        ),
        "target_mask_codes": {
            "0": "unknown/masked",
            "1": "platform_measured",
            "2": "airborne_zero",
        },
        "parameters": {
            "pair_tolerance_ms": args.pair_tolerance_ms,
            "frame_tolerance_ms": args.frame_tolerance_ms,
            "segment_gap_ms": args.segment_gap_ms,
            "platform_lowpass_hz": args.platform_lowpass_hz,
            "lateral_contact_threshold_n": args.lateral_contact_threshold_n,
            "lateral_active_total_threshold_n": args.lateral_active_total_threshold_n,
            "airborne_foot_height_mm": args.airborne_foot_height_mm,
            "airborne_min_markers_without_toe": args.airborne_min_markers_without_toe,
            "airborne_insole_veto_n": finite_or_none(args.airborne_insole_veto_n),
            "airborne_minimum_duration_ms": args.airborne_minimum_duration_ms,
            "min_assignment_correlation": args.min_assignment_correlation,
            "assignment_correlation_bonus_mm": args.assignment_correlation_bonus_mm,
            "assignment_continuity_bonus_mm": args.assignment_continuity_bonus_mm,
            "max_assignment_distance_mm": args.max_assignment_distance_mm,
            "max_transport_reference_distance_mm": (
                args.max_transport_reference_distance_mm
            ),
            "max_effective_lever_arm_mm": args.max_effective_lever_arm_mm,
            "min_force_for_distant_contact_n": (
                args.min_force_for_distant_contact_n
            ),
        },
        "reports": summaries,
        "training_task_sets": {
            "continuous": ["hulahoop", "balance", "lateral_2supports", "squat"],
            "continuous_walk": [
                "hulahoop",
                "balance",
                "lateral_2supports",
                "squat",
                "walk",
            ],
            "all": "all available task families",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame(summaries).drop(columns=["foot_frame_L", "foot_frame_R"]).to_csv(
        args.output_dir / "preparation_summary.csv", index=False
    )
    print(f"Prepared data: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
