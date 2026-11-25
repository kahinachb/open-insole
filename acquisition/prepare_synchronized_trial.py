#!/usr/bin/env python3
"""Prepare one synchronized insole, force-platform and mocap trial.

The export keeps decoded integer counts and adds uncalibrated physical units.
No measured FX29 tare, IMU calibration or magnetometer calibration is applied.
Platform and marker data are read from one C3D and share the QTM clock.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import ezc3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from insole_data.decodageToCSV import (
    CAPTEURS_ABSENTS,
    LIVRES_VERS_NEWTONS,
    PLAGES_FX29_LB,
    SPAN_NOMINAL_FX29_COUNTS,
    coeffsConversion,
    load_all_imu_calibrations,
    matrix_vector_product,
)


FRAME_SIZE = 36
PACKET_SIZE = 148
TIMESTAMP_MODULUS = 1 << 32
EXPECTED_PATCHES = {
    "L": (1, 3, 5, 7),
    "R": (9, 11, 13, 15),
}
PACKET_IDENTIFIERS = {
    (1, 2): ((1, 2), (3, 4), (5, 6), (7, 8)),
    (9, 10): ((9, 10), (11, 12), (13, 14), (15, 16)),
}
FRAME_FORMAT = ">BBBHHHHhhhhhhBHBHBBBBBBB"
FX29_NOMINAL_ZERO_COUNTS = 1000.0

# Chemins par defaut des fichiers de calibration/tare produits en amont
# (calculerTareFX29.py, calibrate_accelerometers.py/calibrate_gyroscopes.py,
# calibrateMagnetometer.py). Utilises par prepare_synchronized_trial.py et
# batch_process_manip.py, avec possibilite de les surcharger en CLI.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARE_LEFT = REPO_ROOT / "manip/tare1/tare_fx29_gauche.json"
DEFAULT_TARE_RIGHT = REPO_ROOT / "manip/tare1/tare_fx29_droit.json"
DEFAULT_ACC_CALIBRATION_LEFT = (
    REPO_ROOT
    / "calibration_imu/data/session_20260728_155627_gauche/calibration_acc/accelerometer_calibration.json"
)
DEFAULT_GYRO_CALIBRATION_LEFT = (
    REPO_ROOT
    / "calibration_imu/data/session_gyro_20260803_124950_gauche/calibration_gyro_std015_cv/gyroscope_calibration.json"
)
DEFAULT_ACC_CALIBRATION_RIGHT = (
    REPO_ROOT
    / "calibration_imu/data/session_20260914_181836_droit/calibration_acc/accelerometer_calibration.json"
)
DEFAULT_GYRO_CALIBRATION_RIGHT = (
    REPO_ROOT
    / "calibration_imu/data/session_gyro_20260915_113614_droit/calibration_gyro/gyroscope_calibration.json"
)
DEFAULT_MAG_CALIBRATION_LEFT = (
    REPO_ROOT / "calibration_magnetometer/calibration_gauche_20260916/magnetometer_calibration.json"
)
DEFAULT_MAG_CALIBRATION_RIGHT = (
    REPO_ROOT / "calibration_magnetometer/calibration_droite_20260916/magnetometer_calibration.json"
)


def load_fx29_tares(path_left: Path | None, path_right: Path | None) -> dict[str, list[float | None]]:
    """Fusionne les tares FX29 gauche/droite (format 'fx29_tare_v1')."""
    tares: dict[str, list[float | None]] = {}
    for path in (path_left, path_right):
        if path is None:
            continue
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("format") != "fx29_tare_v1":
            raise ValueError(f"format de fichier de tare non reconnu : {path}")
        for patch_id, values in document["tares"].items():
            if patch_id in tares:
                raise ValueError(f"tare du patch {patch_id} presente dans plusieurs fichiers")
            tares[patch_id] = values
    return tares


def load_magnetometer_calibrations(
    path_left: Path | None, path_right: Path | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fusionne les calibrations hard-iron/soft-iron gauche/droite.

    Suit la meme formule que calibration_magnetometer/applyMagnetometerCalibration.py :
    m_cal = soft_iron_matrix @ (m_raw - hard_iron_bias_uT), indexee par patch
    "ID1-ID2".
    """
    calibrations: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for path in (path_left, path_right):
        if path is None:
            continue
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        for patch_label, values in document.get("patches", {}).items():
            bias = np.asarray(values["hard_iron_bias_uT"], dtype=float)
            matrix = np.asarray(values["soft_iron_matrix"], dtype=float)
            if bias.shape != (3,) or matrix.shape != (3, 3):
                raise ValueError(f"coefficients magnetometre invalides pour le patch {patch_label}")
            if patch_label in calibrations:
                raise ValueError(f"patch {patch_label} present dans plusieurs calibrations magnetometre")
            calibrations[patch_label] = (bias, matrix)
    return calibrations


def calibrated_force_n(
    raw_counts: float, patch_id: int, force_index: int, tares: dict[str, list[float | None]]
) -> float:
    """Force FX29 en N, tare mesuree appliquee (memes regles que decodageToCSV.getForce)."""
    if (patch_id, force_index) in CAPTEURS_ABSENTS:
        return 0.0
    range_lb = PLAGES_FX29_LB[patch_id][force_index - 1]
    if range_lb is None:
        return math.nan
    tare_patch = tares.get(str(patch_id))
    zero_counts = tare_patch[force_index - 1] if tare_patch else FX29_NOMINAL_ZERO_COUNTS
    if zero_counts is None:
        return math.nan
    newtons_per_count = range_lb * LIVRES_VERS_NEWTONS / SPAN_NOMINAL_FX29_COUNTS
    force_n = (float(raw_counts) - float(zero_counts)) * newtons_per_count
    return max(0.0, force_n)


def calibrated_imu(
    acc_uncal_g: list[float],
    gyro_uncal_dps: list[float],
    patch_id: int,
    imu_calibrations: dict[int, dict[str, tuple]],
) -> tuple[list[float], list[float]]:
    """Applique la correction matricielle acc/gyro (memes regles que decodageToCSV.apply_imu_calibration)."""
    calibration = imu_calibrations.get(patch_id)
    if calibration is None:
        return [math.nan] * 3, [math.nan] * 3
    acc_matrix, additive_bias = calibration["acc"]
    calibrated_acc = matrix_vector_product(
        acc_matrix,
        [value + float(bias) for value, bias in zip(acc_uncal_g, additive_bias)],
    )
    gyro_matrix, subtractive_bias = calibration["gyro"]
    calibrated_gyro = matrix_vector_product(
        gyro_matrix,
        [value - float(bias) for value, bias in zip(gyro_uncal_dps, subtractive_bias)],
    )
    return calibrated_acc, calibrated_gyro


def calibrated_magnetometer(
    mag_uncal_uT: list[float],
    patch_id: int,
    patch_id2: int,
    mag_calibrations: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[float]:
    """m_cal = soft_iron_matrix @ (m_raw - hard_iron_bias_uT), cf. load_magnetometer_calibrations."""
    calibration = mag_calibrations.get(f"{patch_id}-{patch_id2}")
    if calibration is None:
        return [math.nan] * 3
    bias, matrix = calibration
    raw = np.asarray(mag_uncal_uT, dtype=float)
    return (matrix @ (raw - bias)).tolist()


@dataclass(frozen=True)
class RawPacket:
    side: str
    signal: int
    timestamp_us: int
    frames: tuple[dict[str, int], ...]


def packet_is_valid(data: bytes, start: int) -> bool:
    if start + PACKET_SIZE > len(data):
        return False
    sequence = PACKET_IDENTIFIERS.get(tuple(data[start : start + 2]))
    if sequence is None:
        return False
    for patch_index, identifier in enumerate(sequence):
        frame_start = start + patch_index * FRAME_SIZE
        if tuple(data[frame_start : frame_start + 2]) != identifier:
            return False
        forces = struct.unpack(">HHHH", data[frame_start + 3 : frame_start + 11])
        if any(force > 0x3FFF for force in forces):
            return False
    return True


def decode_frame(frame: bytes) -> dict[str, int]:
    values = struct.unpack(FRAME_FORMAT, frame)
    (
        id1,
        id2,
        counter,
        force_1,
        force_2,
        force_3,
        force_4,
        acc_x,
        acc_y,
        acc_z,
        gyro_x,
        gyro_y,
        gyro_z,
        pressure_msb,
        pressure_lsb,
        temperature_msb,
        temperature_lsb,
        mag_x_0,
        mag_x_1,
        mag_y_0,
        mag_y_1,
        mag_z_0,
        mag_z_1,
        mag_xyz_ext,
    ) = values
    mag_x = (mag_x_0 << 10) | (mag_x_1 << 2) | ((mag_xyz_ext >> 6) & 0x03)
    mag_y = (mag_y_0 << 10) | (mag_y_1 << 2) | ((mag_xyz_ext >> 4) & 0x03)
    mag_z = (mag_z_0 << 10) | (mag_z_1 << 2) | ((mag_xyz_ext >> 2) & 0x03)
    return {
        "id1": id1,
        "id2": id2,
        "counter": counter,
        "force_1": force_1,
        "force_2": force_2,
        "force_3": force_3,
        "force_4": force_4,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyro_x": gyro_x,
        "gyro_y": gyro_y,
        "gyro_z": gyro_z,
        "mag_x": mag_x,
        "mag_y": mag_y,
        "mag_z": mag_z,
        "pressure": (pressure_msb << 16) | pressure_lsb,
        "temperature": (temperature_msb << 16) | temperature_lsb,
    }


def decode_binary(path: Path) -> tuple[list[RawPacket], dict[str, int]]:
    data = path.read_bytes()
    packets: list[RawPacket] = []
    signal = 0
    index = 0
    skipped_bytes = 0
    while index + PACKET_SIZE <= len(data):
        if data[index] in (0, 1) and packet_is_valid(data, index + 1):
            signal = int(data[index])
            packet_start = index + 1
        elif packet_is_valid(data, index):
            packet_start = index
        else:
            index += 1
            skipped_bytes += 1
            continue

        frames = tuple(
            decode_frame(
                data[
                    packet_start + patch_index * FRAME_SIZE :
                    packet_start + (patch_index + 1) * FRAME_SIZE
                ]
            )
            for patch_index in range(4)
        )
        side = "L" if frames[0]["id1"] == 1 else "R"
        timestamp_us = struct.unpack(">I", data[packet_start + 144 : packet_start + 148])[0]
        packets.append(RawPacket(side, signal, timestamp_us, frames))
        index = packet_start + PACKET_SIZE

    return packets, {
        "file_bytes": len(data),
        "valid_packets": len(packets),
        "skipped_bytes": skipped_bytes,
        "trailing_bytes": len(data) - index,
    }


def unwrap_timestamps(values: list[int]) -> np.ndarray:
    if not values:
        return np.empty(0, dtype=np.int64)
    result = np.empty(len(values), dtype=np.int64)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = result[index - 1] + (
            (values[index] - values[index - 1]) % TIMESTAMP_MODULUS
        )
    return result


def active_segment(packets: list[RawPacket], side: str) -> list[RawPacket]:
    side_packets = [packet for packet in packets if packet.side == side]
    active_indices = [
        index for index, packet in enumerate(side_packets) if packet.signal == 1
    ]
    if not active_indices:
        raise ValueError(f"Aucun paquet Signal=1 pour la semelle {side}")
    first, last = active_indices[0], active_indices[-1]
    if any(packet.signal != 1 for packet in side_packets[first : last + 1]):
        raise ValueError(
            f"Plusieurs segments Signal=1 pour la semelle {side}; "
            "traitez-les comme des essais separes"
        )
    return side_packets[first : last + 1]


def clean_side_packets(
    packets: list[RawPacket], side: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    active = active_segment(packets, side)
    expected_ids = EXPECTED_PATCHES[side]
    unique: list[RawPacket] = []
    seen: set[tuple[object, ...]] = set()
    duplicates = 0
    for packet in active:
        ids = tuple(frame["id1"] for frame in packet.frames)
        if ids != expected_ids:
            raise ValueError(f"Sequence invalide {ids} pour la semelle {side}")
        key = (
            packet.timestamp_us,
            *(frame["counter"] for frame in packet.frames),
        )
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(packet)

    timestamps = unwrap_timestamps([packet.timestamp_us for packet in unique])
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"Timestamps non strictement croissants pour la semelle {side}")
    time_s = (timestamps - timestamps[0]) / 1_000_000.0

    rows: list[dict[str, float | int]] = []
    for packet, timestamp, relative_time in zip(unique, timestamps, time_s):
        row: dict[str, float | int] = {
            f"{side}_time_s": float(relative_time),
            f"{side}_timestamp_unwrapped_us": int(timestamp),
        }
        for frame in packet.frames:
            prefix = f"{side}_P{frame['id1']}"
            row[f"{prefix}_id2"] = frame["id2"]
            row[f"{prefix}_counter"] = frame["counter"]
            for channel in ("force_1", "force_2", "force_3", "force_4"):
                row[f"{prefix}_{channel}_raw_counts"] = frame[channel]
            for sensor in ("acc", "gyro", "mag"):
                for axis in ("x", "y", "z"):
                    row[f"{prefix}_{sensor}_{axis}_raw_counts"] = frame[
                        f"{sensor}_{axis}"
                    ]
        rows.append(row)

    intervals = np.diff(time_s)
    median_period = float(np.median(intervals))
    missing_by_patch: dict[str, int] = {}
    repeated_counters_by_patch: dict[str, int] = {}
    for patch_index, patch_id in enumerate(expected_ids):
        counters = [packet.frames[patch_index]["counter"] for packet in unique]
        counter_steps = [
            (current - previous) % 256
            for previous, current in zip(counters, counters[1:])
        ]
        missing_by_patch[str(patch_id)] = int(
            sum(max(0, step - 1) for step in counter_steps)
        )
        repeated_counters_by_patch[str(patch_id)] = int(
            sum(step == 0 for step in counter_steps)
        )
    # Un paquet ESP-NOW transporte les quatre patchs. Un saut simultane des
    # quatre compteurs correspond donc a un paquet perdu, pas a quatre.
    missing_from_counter = max(missing_by_patch.values(), default=0)
    stats: dict[str, object] = {
        "active_packets_before_deduplication": len(active),
        "duplicates_removed": duplicates,
        "packets_after_cleaning": len(unique),
        "duration_s": float(time_s[-1]),
        "frequency_hz": float((len(unique) - 1) / time_s[-1]),
        "median_period_ms": median_period * 1000.0,
        "missing_packets_from_counter": missing_from_counter,
        "missing_counter_steps_by_patch": missing_by_patch,
        "repeated_counters_after_deduplication_by_patch": (
            repeated_counters_by_patch
        ),
    }
    return pd.DataFrame(rows), stats


def align_insole_clocks(
    left: pd.DataFrame, right: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rebase chaque semelle sur son propre debut d'acquisition active.

    Les timestamps contenus dans les paquets proviennent des deux ESP32 des
    semelles, et non d'une horloge commune au recepteur. Leur origine peut donc
    differer de plusieurs minutes (ou etre fortuitement proche). Les comparer
    directement cree un faux decalage gauche/droite. Le passage de ``Signal`` a
    1 est la seule origine commune fiable; le retard residuel de chaque semelle
    vers QTM est estime ensuite par correlation avec les plateformes.
    """
    left = left.copy()
    right = right.copy()
    left["L_time_s"] = (
        left["L_timestamp_unwrapped_us"]
        - int(left["L_timestamp_unwrapped_us"].iloc[0])
    ) / 1_000_000.0
    right["R_time_s"] = (
        right["R_timestamp_unwrapped_us"]
        - int(right["R_timestamp_unwrapped_us"].iloc[0])
    ) / 1_000_000.0
    return left, right


def pair_insoles(
    left: pd.DataFrame,
    right: pd.DataFrame,
    tolerance_ms: float,
    offset_ms: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    left = left.sort_values("L_time_s")
    right = right.sort_values("R_time_s")
    paired = pd.merge_asof(
        left,
        right,
        left_on="L_time_s",
        right_on="R_time_s",
        direction="nearest",
        tolerance=tolerance_ms / 1000.0,
    )
    paired = paired.dropna(subset=["R_time_s"]).copy()
    paired["R_timestamp_unwrapped_us"] = paired["R_timestamp_unwrapped_us"].astype(
        np.int64
    )
    before_reuse_removal = len(paired)
    paired = paired.drop_duplicates("R_timestamp_unwrapped_us", keep="first").copy()
    paired["left_right_dt_ms"] = (
        paired["R_time_s"] - paired["L_time_s"]
    ) * 1000.0
    paired["time_qtm_absolute_s"] = (
        (paired["L_time_s"] + paired["R_time_s"]) / 2.0 + offset_ms / 1000.0
    )
    paired.insert(0, "signal", 1)
    columns = [
        "time_qtm_absolute_s",
        "signal",
        "L_time_s",
        "R_time_s",
        "left_right_dt_ms",
        *[
            column
            for column in paired.columns
            if column
            not in {
                "time_qtm_absolute_s",
                "signal",
                "L_time_s",
                "R_time_s",
                "left_right_dt_ms",
            }
        ],
    ]
    paired = paired[columns].sort_values("time_qtm_absolute_s").reset_index(drop=True)
    absolute_dt = paired["left_right_dt_ms"].abs()
    stats: dict[str, float | int] = {
        "paired_samples": len(paired),
        "left_samples_without_pair": len(left) - before_reuse_removal,
        "reused_right_samples_removed": before_reuse_removal - len(paired),
        "median_abs_left_right_dt_ms": float(absolute_dt.median()),
        "p99_abs_left_right_dt_ms": float(absolute_dt.quantile(0.99)),
        "max_abs_left_right_dt_ms": float(absolute_dt.max()),
    }
    return paired, stats


def build_insole_quality_control(
    left_stats: dict[str, object],
    right_stats: dict[str, object],
    pair_stats: dict[str, float | int],
    pair_tolerance_ms: float,
) -> dict[str, object]:
    """Produit des statuts et messages explicites pour le rapport JSON."""
    insole_checks: dict[str, object] = {}
    for side_name, label, stats in (
        ("left", "gauche", left_stats),
        ("right", "droite", right_stats),
    ):
        missing = int(stats["missing_packets_from_counter"])
        duplicates = int(stats["duplicates_removed"])
        repeated_by_patch = stats[
            "repeated_counters_after_deduplication_by_patch"
        ]
        counters_continuous = missing == 0 and all(
            int(value) == 0 for value in repeated_by_patch.values()
        )
        status = "OK" if counters_continuous else "WARNING"
        loss_text = (
            "aucun paquet perdu detecte"
            if missing == 0
            else f"{missing} paquet(s) perdu(s) estime(s)"
        )
        duplicate_text = (
            "aucun doublon detecte"
            if duplicates == 0
            else f"{duplicates} doublon(s) detecte(s) et supprime(s)"
        )
        insole_checks[side_name] = {
            "status": status,
            "signal_value_checked": 1,
            "packets_before_deduplication": int(
                stats["active_packets_before_deduplication"]
            ),
            "duplicates_detected_and_removed": duplicates,
            "packets_after_cleaning": int(stats["packets_after_cleaning"]),
            "estimated_lost_packets": missing,
            "estimated_lost_packets_by_patch": stats[
                "missing_counter_steps_by_patch"
            ],
            "counter_continuity_ok": counters_continuous,
            "message": f"Semelle {label}: {loss_text}; {duplicate_text}.",
        }

    paired_samples = int(pair_stats["paired_samples"])
    left_packets = int(left_stats["packets_after_cleaning"])
    right_packets = int(right_stats["packets_after_cleaning"])
    left_coverage = 100.0 * paired_samples / left_packets
    right_coverage = 100.0 * paired_samples / right_packets
    duration_difference_ms = abs(
        float(left_stats["duration_s"]) - float(right_stats["duration_s"])
    ) * 1000.0
    synchronization_ok = (
        float(pair_stats["p99_abs_left_right_dt_ms"]) <= pair_tolerance_ms
        and left_coverage >= 99.0
        and right_coverage >= 99.0
    )
    synchronization_status = "OK" if synchronization_ok else "WARNING"
    synchronization_message = (
        "Synchronisation gauche/droite OK"
        if synchronization_ok
        else "Synchronisation gauche/droite a verifier"
    )
    synchronization_check = {
        "status": synchronization_status,
        "message": synchronization_message,
        "acceptance_pair_tolerance_ms": pair_tolerance_ms,
        "minimum_pairing_coverage_percent": 99.0,
        "paired_unique_samples": paired_samples,
        "left_pairing_coverage_percent": left_coverage,
        "right_pairing_coverage_percent": right_coverage,
        "median_abs_time_difference_ms": float(
            pair_stats["median_abs_left_right_dt_ms"]
        ),
        "p99_abs_time_difference_ms": float(
            pair_stats["p99_abs_left_right_dt_ms"]
        ),
        "max_abs_time_difference_ms": float(
            pair_stats["max_abs_left_right_dt_ms"]
        ),
        "active_duration_difference_ms": duration_difference_ms,
    }

    overall_ok = (
        insole_checks["left"]["status"] == "OK"
        and insole_checks["right"]["status"] == "OK"
        and synchronization_ok
    )
    return {
        "overall_status": "OK" if overall_ok else "WARNING",
        "overall_message": (
            "Donnees des deux semelles OK pendant Signal=1."
            if overall_ok
            else "Anomalie detectee dans les donnees semelles pendant Signal=1."
        ),
        "signal_1_integrity": insole_checks,
        "left_right_synchronization": synchronization_check,
    }


def make_long_insole_table(
    left: pd.DataFrame,
    right: pd.DataFrame,
    offset_ms: float,
    tares: dict[str, list[float | None]] | None = None,
    imu_calibrations: dict[int, dict[str, tuple]] | None = None,
    mag_calibrations: dict[str, tuple] | None = None,
) -> pd.DataFrame:
    """Une observation par patch, sans interpolation ni repetition de paquet.

    En plus des colonnes nominales/non calibrees (conservees pour
    compatibilite), ajoute quand les fichiers correspondants sont fournis :
    force_{i}_N (tare mesuree appliquee), acc/gyro_{axis}_cal (matrice de
    correction), mag_{axis}_cal_uT + mag_norm_cal_uT (hard/soft iron). Colonne
    NaN quand la calibration du patch concerne est absente.
    """
    tares = tares or {}
    imu_calibrations = imu_calibrations or {}
    mag_calibrations = mag_calibrations or {}
    output_rows: list[dict[str, float | int | str]] = []
    channels = (
        "force_1",
        "force_2",
        "force_3",
        "force_4",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
        "mag_x",
        "mag_y",
        "mag_z",
    )
    for side, frame, patch_ids in (
        ("L", left, EXPECTED_PATCHES["L"]),
        ("R", right, EXPECTED_PATCHES["R"]),
    ):
        side_time = f"{side}_time_s"
        side_timestamp = f"{side}_timestamp_unwrapped_us"
        for source_row in frame.to_dict(orient="records"):
            for patch_id in patch_ids:
                prefix = f"{side}_P{patch_id}"
                output_row: dict[str, float | int | str] = {
                    "time_qtm_absolute_s": float(source_row[side_time])
                    + offset_ms / 1000.0,
                    "signal": 1,
                    "side": side,
                    "patch_id": patch_id,
                    "patch_id2": int(source_row[f"{prefix}_id2"]),
                    "packet_timestamp_unwrapped_us": int(source_row[side_timestamp]),
                    "patch_counter": int(source_row[f"{prefix}_counter"]),
                }
                for channel in channels:
                    output_row[f"{channel}_raw_counts"] = int(
                        source_row[f"{prefix}_{channel}_raw_counts"]
                    )
                for force_index, range_lb in enumerate(
                    PLAGES_FX29_LB[patch_id], start=1
                ):
                    physical_column = f"force_{force_index}_nominal_N_no_tare"
                    if range_lb is None:
                        output_row[physical_column] = math.nan
                    else:
                        raw_force = output_row[f"force_{force_index}_raw_counts"]
                        output_row[physical_column] = (
                            (float(raw_force) - FX29_NOMINAL_ZERO_COUNTS)
                            * range_lb
                            * LIVRES_VERS_NEWTONS
                            / SPAN_NOMINAL_FX29_COUNTS
                        )
                    output_row[f"force_{force_index}_N"] = calibrated_force_n(
                        output_row[f"force_{force_index}_raw_counts"],
                        patch_id,
                        force_index,
                        tares,
                    )

                acc_uncal_g = []
                gyro_uncal_dps = []
                mag_uncal_uT = []
                for axis in ("x", "y", "z"):
                    acc_value = (
                        float(output_row[f"acc_{axis}_raw_counts"])
                        * coeffsConversion["acc_xyz"]
                    )
                    gyro_value = (
                        float(output_row[f"gyro_{axis}_raw_counts"])
                        * coeffsConversion["gy_xyz"]
                    )
                    mag_value = (
                        (
                            float(output_row[f"mag_{axis}_raw_counts"])
                            - coeffsConversion["m_xyz"]
                        )
                        * coeffsConversion["f_xyz"]
                    )
                    acc_uncal_g.append(acc_value)
                    gyro_uncal_dps.append(gyro_value)
                    mag_uncal_uT.append(mag_value)
                    output_row[f"acc_{axis}_uncal_g"] = acc_value
                    output_row[f"gyro_{axis}_uncal_dps"] = gyro_value
                    output_row[f"mag_{axis}_uncal_uT"] = mag_value

                acc_cal, gyro_cal = calibrated_imu(
                    acc_uncal_g, gyro_uncal_dps, patch_id, imu_calibrations
                )
                mag_cal = calibrated_magnetometer(
                    mag_uncal_uT,
                    patch_id,
                    output_row["patch_id2"],
                    mag_calibrations,
                )
                for axis, value in zip(("x", "y", "z"), acc_cal):
                    output_row[f"acc_{axis}_cal_g"] = value
                for axis, value in zip(("x", "y", "z"), gyro_cal):
                    output_row[f"gyro_{axis}_cal_dps"] = value
                for axis, value in zip(("x", "y", "z"), mag_cal):
                    output_row[f"mag_{axis}_cal_uT"] = value
                output_row["mag_norm_cal_uT"] = (
                    float(np.linalg.norm(mag_cal))
                    if all(math.isfinite(value) for value in mag_cal)
                    else math.nan
                )
                output_rows.append(output_row)
    return pd.DataFrame(output_rows).sort_values(
        ["time_qtm_absolute_s", "side", "patch_id"]
    ).reset_index(drop=True)


def unique_channel_names(labels: list[str], units: list[str]) -> list[str]:
    if len(labels) % 6 != 0:
        raise ValueError(
            "Le C3D doit contenir des blocs de 6 voies par plateforme "
            "(3 forces puis 3 moments)"
        )
    names: list[str] = []
    for index, (label, unit) in enumerate(zip(labels, units)):
        platform = index // 6 + 1
        base = label.strip().lower().replace(" ", "_")
        clean_unit = unit.strip().replace("/", "_per_") or "unit"
        names.append(f"{base}_p{platform}_{clean_unit}")
    return names


def platform_local_to_global_rotation(
    corners: np.ndarray, platform_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation locale->globale d'une plateforme a partir de ses 4 coins C3D.

    corners: tableau (xyz, coin, plateforme) issu de FORCE_PLATFORM:CORNERS,
    deja exprime dans le repere global (mocap). La convention C3D numerote
    les coins (+x,+y), (-x,+y), (-x,-y), (+x,-y). L'axe +x va donc du coin 2
    au coin 1 et l'axe +y du coin 4 au coin 1. Les colonnes de la matrice
    retournee sont les axes locaux exprimes dans le repere global.
    """
    corner_points = corners[:, :, platform_index].T  # (4, 3), mm, repere global
    center = corner_points.mean(axis=0)
    x_hat = corner_points[0] - corner_points[1]
    x_hat = x_hat / np.linalg.norm(x_hat)
    y_raw = corner_points[0] - corner_points[3]
    y_raw = y_raw - np.dot(y_raw, x_hat) * x_hat
    y_hat = y_raw / np.linalg.norm(y_raw)
    z_hat = np.cross(x_hat, y_hat)
    z_hat = z_hat / np.linalg.norm(z_hat)
    y_hat = np.cross(z_hat, x_hat)
    y_hat = y_hat / np.linalg.norm(y_hat)
    rotation = np.column_stack([x_hat, y_hat, z_hat])
    if np.linalg.det(rotation) < 0.999:
        raise ValueError("repere plateforme invalide ou non direct")
    return rotation, center


def add_global_frame_platform_columns(platform: pd.DataFrame, c3d: "ezc3d.c3d") -> None:
    """Ajoute le torseur et le CoP globaux calcules par le filtre ezc3d.

    Les canaux analogiques locaux sont conserves tels quels. ``ezc3d`` applique
    la convention C3D (CORNERS, ORIGIN, TYPE et CHANNEL) et transporte les
    moments sur le plan de travail de la plateforme. Cela evite de dupliquer
    imparfaitement les conventions propres aux differents types de plateformes.
    """
    platforms = c3d["data"].get("platform", [])
    if not platforms:
        raise ValueError(
            "donnees plateforme ezc3d absentes; lire le C3D avec "
            "extract_forceplat_data=True"
        )
    axes = ("x", "y", "z")
    for platform_index, force_platform in enumerate(platforms):
        platform_number = platform_index + 1
        arrays = {
            "force": np.asarray(force_platform["force"], dtype=float).T,
            "moment": np.asarray(force_platform["moment"], dtype=float).T,
            "cop": np.asarray(force_platform["center_of_pressure"], dtype=float).T,
            "free_moment": np.asarray(force_platform["Tz"], dtype=float).T,
        }
        if any(values.shape != (len(platform), 3) for values in arrays.values()):
            raise ValueError(
                f"dimensions ezc3d inattendues pour la plateforme {platform_number}"
            )
        for axis_index, axis in enumerate(axes):
            platform[f"force_{axis}_p{platform_number}_global_N"] = arrays["force"][:, axis_index]
            platform[f"moment_{axis}_p{platform_number}_global_Nmm"] = arrays["moment"][:, axis_index]
            platform[f"cop_{axis}_p{platform_number}_global_mm"] = arrays["cop"][:, axis_index]
            platform[f"free_moment_{axis}_p{platform_number}_global_Nmm"] = arrays["free_moment"][:, axis_index]


def read_c3d(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    c3d = ezc3d.c3d(str(path), extract_forceplat_data=True)
    point_rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    analog_rate = float(c3d["parameters"]["ANALOG"]["RATE"]["value"][0])

    analog = np.asarray(c3d["data"]["analogs"], dtype=float)
    analog_values = analog.transpose(2, 0, 1).reshape(-1, analog.shape[1])
    analog_labels = list(c3d["parameters"]["ANALOG"]["LABELS"]["value"])
    analog_units = list(c3d["parameters"]["ANALOG"]["UNITS"]["value"])
    platform = pd.DataFrame(
        analog_values,
        columns=unique_channel_names(analog_labels, analog_units),
    )
    platform.insert(0, "time_qtm_absolute_s", np.arange(len(platform)) / analog_rate)
    platform.insert(1, "sample_index_c3d", np.arange(len(platform), dtype=np.int64))
    add_global_frame_platform_columns(platform, c3d)

    points = np.asarray(c3d["data"]["points"], dtype=float)
    marker_labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    marker_data: dict[str, np.ndarray] = {}
    for marker_index, label in enumerate(marker_labels):
        visible = points[3, marker_index, :] >= 0
        for axis_index, axis in enumerate(("x", "y", "z")):
            values = points[axis_index, marker_index, :].copy()
            values[~visible] = np.nan
            marker_data[f"{label.strip()}_{axis}_mm"] = values
    mocap = pd.DataFrame(marker_data)
    mocap.insert(0, "time_qtm_absolute_s", np.arange(len(mocap)) / point_rate)
    mocap.insert(1, "frame_index_c3d", np.arange(len(mocap), dtype=np.int64))

    event_times = []
    event = c3d["parameters"].get("EVENT", {})
    if "TIMES" in event:
        times = np.asarray(event["TIMES"]["value"], dtype=float)
        if times.ndim == 2 and times.shape[0] >= 2:
            event_times = times[0, :] * 60.0 + times[1, :]
    metadata = {
        "point_rate_hz": point_rate,
        "analog_rate_hz": analog_rate,
        "marker_count": len(marker_labels),
        "platform_count": len(c3d["data"].get("platform", [])),
        "platform_global_source": "ezc3d_force_platform_filter",
        "platform_moment_reference": "working_surface_plane",
        "qtm_event_times_s": [float(value) for value in event_times],
    }
    return platform, mocap, metadata


def crop_and_rebase(
    frame: pd.DataFrame, start_s: float, end_s: float
) -> pd.DataFrame:
    result = frame.loc[
        frame["time_qtm_absolute_s"].between(start_s, end_s, inclusive="both")
    ].copy()
    if result.empty:
        raise ValueError("Aucune donnee C3D dans l'intervalle commun")
    result.insert(
        0,
        "time_qtm_s",
        result["time_qtm_absolute_s"] - start_s,
    )
    return result.reset_index(drop=True)


def lowpass(values: np.ndarray, rate_hz: float, cutoff_hz: float = 15.0) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) < 30:
        return values
    if not np.all(np.isfinite(values)):
        values = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    sos = butter(4, cutoff_hz, btype="lowpass", fs=rate_hz, output="sos")
    return sosfiltfilt(sos, values)


def insole_force_sum(
    frame: pd.DataFrame, side: str
) -> tuple[np.ndarray, np.ndarray]:
    """Somme des 4 capteurs FX29, tare mesuree appliquee (force_{i}_N)."""
    force_columns = [f"force_{index}_N" for index in range(1, 5)]
    selected = frame.loc[frame["side"] == side].copy()
    selected["force_sum"] = selected[force_columns].sum(axis=1, skipna=True)
    grouped = selected.groupby("time_qtm_s", sort=True)["force_sum"].sum()
    return grouped.index.to_numpy(dtype=float), grouped.to_numpy(dtype=float)


def wide_insole_force_sum(
    frame: pd.DataFrame, side: str, tares: dict[str, list[float | None]]
) -> tuple[np.ndarray, np.ndarray]:
    """Somme des 4 capteurs FX29 des 4 patchs d'un pied, tare mesuree appliquee."""
    total = np.zeros(len(frame), dtype=float)
    for patch_id in EXPECTED_PATCHES[side]:
        for force_index, range_lb in enumerate(PLAGES_FX29_LB[patch_id], start=1):
            if range_lb is None:
                continue
            raw = frame[
                f"{side}_P{patch_id}_force_{force_index}_raw_counts"
            ].to_numpy(dtype=float)
            total += np.array(
                [calibrated_force_n(value, patch_id, force_index, tares) for value in raw]
            )
    return frame[f"{side}_time_s"].to_numpy(dtype=float), total


def vertical_platform_columns(platform: pd.DataFrame) -> list[str]:
    """Voies Fz locales (repere capteur, pas les colonnes _global_ ajoutees)."""
    columns = [
        column
        for column in platform
        if column.startswith("force_z_p") and not column.endswith("_global_N")
    ]
    if len(columns) < 2:
        raise ValueError("Deux voies de force verticale sont requises dans le C3D")
    return columns[:2]


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    valid = np.isfinite(first) & np.isfinite(second)
    if np.sum(valid) < 20:
        return math.nan
    first = first[valid] - np.mean(first[valid])
    second = second[valid] - np.mean(second[valid])
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    return float(np.dot(first, second) / denominator) if denominator > 0 else math.nan


def contact_segments(
    vertical_force: np.ndarray,
    rate_hz: float,
    threshold_n: float,
    margin_s: float = 0.15,
    minimum_duration_s: float = 0.05,
) -> list[tuple[int, int, int, int]]:
    contact = np.asarray(vertical_force) > threshold_n
    transitions = np.flatnonzero(np.diff(np.r_[False, contact, False]))
    segments: list[tuple[int, int, int, int]] = []
    margin = int(round(margin_s * rate_hz))
    minimum_samples = int(round(minimum_duration_s * rate_hz))
    for start, stop in transitions.reshape(-1, 2):
        if stop - start < minimum_samples:
            continue
        segments.append(
            (
                int(start),
                int(stop),
                max(0, int(start) - margin),
                min(len(contact), int(stop) + margin),
            )
        )
    return segments


def estimate_insole_offset(
    left: pd.DataFrame,
    right: pd.DataFrame,
    platform: pd.DataFrame,
    platform_rate_hz: float,
    search_ms: float,
    contact_threshold_n: float,
    tares: dict[str, list[float | None]],
    max_event_duration_s: float = 2.0,
    min_reliable_contacts: int = 2,
) -> tuple[float, dict[str, object]]:
    """Estime le retard par correlation, contact par contact.

    Chaque contact est compare aux deux semelles. La semelle et le retard qui
    maximisent la correlation sont retenus, puis la mediane des retards des
    contacts fiables fournit l'offset global robuste.

    Seul le debut de chaque contact (jusqu'a max_event_duration_s) sert a la
    correlation : sur un appui statique prolonge (essais d'equilibre), le
    plateau de force est quasi constant et sa correlation est dominee par le
    bruit/la derive, alors que le front de montee au moment ou le pied se
    pose reste un evenement net et fiable, y compris pour un seul contact par
    plateforme.
    """
    insole_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side, frame in (("L", left), ("R", right)):
        time_s, force_n = wide_insole_force_sum(frame, side, tares)
        insole_signals[side] = (time_s, lowpass(force_n, 200.0))

    platform_time = platform["time_qtm_absolute_s"].to_numpy(dtype=float)
    candidates_s = np.arange(-search_ms, search_ms + 0.5, 1.0) / 1000.0
    event_results: list[dict[str, object]] = []
    for platform_index, column in enumerate(vertical_platform_columns(platform), start=1):
        vertical_force = lowpass(
            -platform[column].to_numpy(dtype=float), platform_rate_hz
        )
        for start, stop, window_start, window_stop in contact_segments(
            vertical_force,
            platform_rate_hz,
            contact_threshold_n,
        ):
            window_stop = min(
                window_stop, start + int(round(max_event_duration_s * platform_rate_hz))
            )
            qtm_window = platform_time[window_start:window_stop]
            platform_window = vertical_force[window_start:window_stop]
            best_correlation = -math.inf
            best_side = ""
            best_offset_s = math.nan
            for side, (insole_time, insole_force) in insole_signals.items():
                for offset_s in candidates_s:
                    interpolated = np.interp(
                        qtm_window - offset_s,
                        insole_time,
                        insole_force,
                        left=np.nan,
                        right=np.nan,
                    )
                    correlation = pearson_correlation(
                        platform_window, interpolated
                    )
                    if np.isfinite(correlation) and correlation > best_correlation:
                        best_correlation = correlation
                        best_side = side
                        best_offset_s = float(offset_s)
            if np.isfinite(best_offset_s):
                event_results.append(
                    {
                        "platform": platform_index,
                        "contact_start_qtm_s": float(platform_time[start]),
                        "contact_end_qtm_s": float(platform_time[stop - 1]),
                        "matched_side": best_side,
                        "best_offset_ms": best_offset_s * 1000.0,
                        "correlation": best_correlation,
                    }
                )

    reliable = [event for event in event_results if event["correlation"] >= 0.85]
    if len(reliable) < min_reliable_contacts:
        raise ValueError(
            f"Estimation automatique du retard impossible: moins de "
            f"{min_reliable_contacts} contacts plateforme/semelle fiables. "
            "Fournissez --insole-offset-ms."
        )
    offsets = np.asarray([event["best_offset_ms"] for event in reliable], dtype=float)
    if np.any(np.abs(offsets) >= search_ms - 1.0):
        raise ValueError(
            "Le meilleur retard atteint la limite de recherche; augmentez "
            "--offset-search-ms ou fournissez --insole-offset-ms."
        )
    estimated_ms = float(np.median(offsets))
    report: dict[str, object] = {
        "mode": "automatic_contact_correlation",
        "estimated_offset_ms": estimated_ms,
        "search_range_ms": [-search_ms, search_ms],
        "contact_threshold_n": contact_threshold_n,
        "max_event_duration_s": max_event_duration_s,
        "min_reliable_contacts": min_reliable_contacts,
        "reliable_contact_count": len(reliable),
        "offset_q25_ms": float(np.quantile(offsets, 0.25)),
        "offset_q75_ms": float(np.quantile(offsets, 0.75)),
        "median_contact_correlation": float(
            np.median([event["correlation"] for event in reliable])
        ),
        "contacts": event_results,
    }
    return estimated_ms, report


def shoe_mean_height(mocap: pd.DataFrame, side: str) -> np.ndarray:
    # Les labels de marqueurs QTM ne sont pas toujours dans la meme casse
    # d'un projet a l'autre (ex: "L_toe" pour pilot1, "l_toe" pour pilot2) :
    # comparaison insensible a la casse pour couvrir les deux.
    columns = [
        column
        for column in mocap
        if column.lower().startswith(f"{side.lower()}_") and column.endswith("_z_mm")
    ]
    if not columns:
        return np.full(len(mocap), np.nan)
    return mocap[columns].mean(axis=1, skipna=True).to_numpy(dtype=float)


def _best_side(correlations: dict[str, float]) -> str:
    return max(
        correlations,
        key=lambda side: correlations[side] if np.isfinite(correlations[side]) else -math.inf,
    )


def classify_platform_contacts(
    insoles: pd.DataFrame,
    platform: pd.DataFrame,
    platform_rate_hz: float,
    contact_threshold_n: float,
) -> list[dict[str, object]]:
    """Associe chaque contact plateforme au pied le plus correle.

    Quand les deux plateformes sont en contact au meme instant (reception
    bilaterale d'un saut, double appui en marche), ce sont necessairement
    deux pieds differents : la correlation avec sa propre semelle peut alors
    etre quasi identique pour les deux pieds (charges quasi simultanees et
    similaires), rendant le choix independant plateforme par plateforme
    instable. On resout donc ces paires conjointement (les deux options
    L/R <-> P1/P2 sont comparees ensemble) plutot qu'independamment.
    """
    insole_signals = {
        side: insole_force_sum(insoles, side) for side in ("L", "R")
    }
    platform_time = platform["time_qtm_s"].to_numpy(dtype=float)

    platform_segments: dict[int, list[dict[str, object]]] = {}
    for platform_index, column in enumerate(vertical_platform_columns(platform), start=1):
        vertical_force = lowpass(
            -platform[column].to_numpy(dtype=float), platform_rate_hz
        )
        segments = []
        for start, stop, window_start, window_stop in contact_segments(
            vertical_force,
            platform_rate_hz,
            contact_threshold_n,
        ):
            time_window = platform_time[window_start:window_stop]
            force_window = vertical_force[window_start:window_stop]
            correlations: dict[str, float] = {}
            for side, (insole_time, insole_force) in insole_signals.items():
                interpolated = np.interp(
                    time_window,
                    insole_time,
                    lowpass(insole_force, 200.0),
                    left=np.nan,
                    right=np.nan,
                )
                correlations[side] = pearson_correlation(force_window, interpolated)
            segments.append(
                {
                    "start_s": float(platform_time[start]),
                    "end_s": float(platform_time[stop - 1]),
                    "correlations": correlations,
                }
            )
        platform_segments[platform_index] = segments

    segments_p1 = platform_segments.get(1, [])
    segments_p2 = platform_segments.get(2, [])
    used_p2: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for i1, seg1 in enumerate(segments_p1):
        best_j, best_overlap = None, 0.0
        for j2, seg2 in enumerate(segments_p2):
            if j2 in used_p2:
                continue
            overlap = min(seg1["end_s"], seg2["end_s"]) - max(seg1["start_s"], seg2["start_s"])
            if overlap > best_overlap:
                best_overlap, best_j = overlap, j2
        if best_j is not None:
            pairs.append((i1, best_j))
            used_p2.add(best_j)

    assignments: list[dict[str, object]] = []
    paired_p1 = {i1 for i1, _ in pairs}
    paired_p2 = {j2 for _, j2 in pairs}

    def safe(value: float) -> float:
        return value if np.isfinite(value) else -math.inf

    for i1, j2 in pairs:
        seg1, seg2 = segments_p1[i1], segments_p2[j2]
        option_r1_l2 = safe(seg1["correlations"]["R"]) + safe(seg2["correlations"]["L"])
        option_l1_r2 = safe(seg1["correlations"]["L"]) + safe(seg2["correlations"]["R"])
        side1, side2 = ("R", "L") if option_r1_l2 >= option_l1_r2 else ("L", "R")
        assignments.append(
            {
                "platform": 1,
                "contact_start_s": seg1["start_s"],
                "contact_end_s": seg1["end_s"],
                "matched_side": side1,
                "correlation": seg1["correlations"][side1],
            }
        )
        assignments.append(
            {
                "platform": 2,
                "contact_start_s": seg2["start_s"],
                "contact_end_s": seg2["end_s"],
                "matched_side": side2,
                "correlation": seg2["correlations"][side2],
            }
        )

    for platform_index, segments, paired in (
        (1, segments_p1, paired_p1),
        (2, segments_p2, paired_p2),
    ):
        for idx, seg in enumerate(segments):
            if idx in paired:
                continue
            matched_side = _best_side(seg["correlations"])
            assignments.append(
                {
                    "platform": platform_index,
                    "contact_start_s": seg["start_s"],
                    "contact_end_s": seg["end_s"],
                    "matched_side": matched_side,
                    "correlation": seg["correlations"][matched_side],
                }
            )

    assignments.sort(key=lambda item: (item["contact_start_s"], item["platform"]))
    return assignments


def make_plots(
    insoles: pd.DataFrame,
    platform: pd.DataFrame,
    mocap: pd.DataFrame,
    output_dir: Path,
    insole_rate_hz: float,
    platform_rate_hz: float,
    contact_threshold_n: float,
    contact_assignments: list[dict[str, object]],
) -> tuple[Path, Path, Path, Path]:
    time_platform = platform["time_qtm_s"].to_numpy()
    time_mocap = mocap["time_qtm_s"].to_numpy()
    time_left, left_values = insole_force_sum(insoles, "L")
    time_right, right_values = insole_force_sum(insoles, "R")
    force_left = lowpass(left_values, insole_rate_hz)
    force_right = lowpass(right_values, insole_rate_hz)
    pf_columns = vertical_platform_columns(platform)
    pf_1 = lowpass(-platform[pf_columns[0]].to_numpy(), platform_rate_hz)
    pf_2 = lowpass(-platform[pf_columns[1]].to_numpy(), platform_rate_hz)

    full_path = output_dir / "synchronisation_complete.png"
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    force_axis = axes[0]
    force_axis.plot(time_left, force_left, label="Semelle gauche", color="#1f77b4")
    force_axis.plot(time_right, force_right, label="Semelle droite", color="#ff7f0e")
    force_axis.plot(time_platform, pf_1, label="Plateforme 1", color="#2ca02c", alpha=0.8)
    force_axis.plot(time_platform, pf_2, label="Plateforme 2", color="#d62728", alpha=0.8)
    force_axis.set_ylabel("Force verticale / somme FX29 nominale (N)")
    force_axis.grid(alpha=0.25)
    force_axis.legend(ncol=4, loc="upper right")
    force_axis.set_title(
        "Controle temporel des forces (filtre 15 Hz pour la figure uniquement)"
    )

    axes[1].plot(time_mocap, shoe_mean_height(mocap, "L"), label="Chaussure gauche")
    axes[1].plot(time_mocap, shoe_mean_height(mocap, "R"), label="Chaussure droite")
    contact = (pf_1 > contact_threshold_n) | (pf_2 > contact_threshold_n)
    axes[1].fill_between(
        time_platform,
        0,
        1,
        where=contact,
        transform=axes[1].get_xaxis_transform(),
        color="0.8",
        alpha=0.35,
        label=f"Contact plateforme > {contact_threshold_n:g} N",
    )
    axes[1].set_ylabel("Hauteur moyenne des marqueurs (mm)")
    axes[1].set_xlabel("Temps QTM recale (s)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3, loc="best")
    figure.tight_layout()
    figure.savefig(full_path, dpi=160)
    plt.close(figure)

    active_times = time_platform[contact]
    zoom_start = max(0.0, float(active_times[0]) - 0.5) if len(active_times) else 0.0
    zoom_end = min(float(time_platform[-1]), zoom_start + 5.2)
    zoom_path = output_dir / "synchronisation_zoom.png"
    figure, force_axis = plt.subplots(figsize=(15, 5.5))
    force_axis.plot(time_left, force_left, label="Semelle gauche", color="#1f77b4")
    force_axis.plot(time_right, force_right, label="Semelle droite", color="#ff7f0e")
    force_axis.plot(time_platform, pf_1, label="Plateforme 1", color="#2ca02c", alpha=0.8)
    force_axis.plot(time_platform, pf_2, label="Plateforme 2", color="#d62728", alpha=0.8)
    force_axis.set_xlim(zoom_start, zoom_end)
    force_axis.set_xlabel("Temps QTM recale (s)")
    force_axis.set_ylabel("Force verticale / somme FX29 nominale (N)")
    force_axis.grid(alpha=0.25)
    force_axis.legend(ncol=4, loc="upper right")
    force_axis.set_title("Zoom sur les premiers contacts")
    figure.tight_layout()
    figure.savefig(zoom_path, dpi=160)
    plt.close(figure)

    side_colors = {"L": "#1f77b4", "R": "#ff7f0e"}
    side_labels = {"L": "gauche", "R": "droite"}
    side_signals = {
        "L": (time_left, force_left),
        "R": (time_right, force_right),
    }
    comparison_path = output_dir / "semelles_par_plateforme.png"
    figure, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for platform_index, (axis, platform_force) in enumerate(
        zip(axes, (pf_1, pf_2)), start=1
    ):
        axis.plot(
            time_platform,
            platform_force,
            color="black",
            linewidth=1.3,
            label=f"Plateforme {platform_index} (-Fz)",
        )
        for side, (insole_time, insole_force) in side_signals.items():
            axis.plot(
                insole_time,
                insole_force,
                color=side_colors[side],
                alpha=0.35,
                linewidth=1.0,
                label=f"Semelle {side_labels[side]} (tout l'essai)",
            )
        for assignment in contact_assignments:
            if assignment["platform"] != platform_index:
                continue
            side = str(assignment["matched_side"])
            start = float(assignment["contact_start_s"])
            stop = float(assignment["contact_end_s"])
            insole_time, insole_force = side_signals[side]
            selected = (insole_time >= start) & (insole_time <= stop)
            axis.axvspan(start, stop, color=side_colors[side], alpha=0.10)
            axis.plot(
                insole_time[selected],
                insole_force[selected],
                color=side_colors[side],
                linewidth=2.8,
                label=f"Contact associe a la semelle {side_labels[side]}",
            )
        handles, labels = axis.get_legend_handles_labels()
        unique_legend = dict(zip(labels, handles))
        axis.legend(unique_legend.values(), unique_legend.keys(), ncol=3, loc="upper right")
        axis.set_ylabel("Force (N)")
        axis.set_title(f"Plateforme {platform_index} et semelles associees")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Temps QTM recale (s)")
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=160)
    plt.close(figure)

    components_path = output_dir / "composantes_plateformes.png"
    figure, axes = plt.subplots(2, 2, figsize=(17, 9), sharex=True)
    component_colors = ("#1f77b4", "#ff7f0e", "#2ca02c")
    for platform_index in (1, 2):
        force_component_axis = axes[0, platform_index - 1]
        moment_component_axis = axes[1, platform_index - 1]
        for axis_name, color in zip(("x", "y", "z"), component_colors):
            force_component_axis.plot(
                time_platform,
                platform[f"force_{axis_name}_p{platform_index}_N"],
                color=color,
                label=f"F{axis_name}",
            )
            moment_component_axis.plot(
                time_platform,
                platform[f"moment_{axis_name}_p{platform_index}_Nmm"],
                color=color,
                label=f"M{axis_name}",
            )
        for assignment in contact_assignments:
            if assignment["platform"] == platform_index:
                side = str(assignment["matched_side"])
                for axis in (force_component_axis, moment_component_axis):
                    axis.axvspan(
                        float(assignment["contact_start_s"]),
                        float(assignment["contact_end_s"]),
                        color=side_colors[side],
                        alpha=0.08,
                    )
        force_component_axis.set_title(f"Plateforme {platform_index} - forces")
        moment_component_axis.set_title(f"Plateforme {platform_index} - moments")
        force_component_axis.set_ylabel("Force (N)")
        moment_component_axis.set_ylabel("Moment (Nmm)")
        moment_component_axis.set_xlabel("Temps QTM recale (s)")
        force_component_axis.legend(ncol=3)
        moment_component_axis.legend(ncol=3)
        force_component_axis.grid(alpha=0.25)
        moment_component_axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(components_path, dpi=160)
    plt.close(figure)
    return full_path, zoom_path, comparison_path, components_path


def count_visible_markers(mocap: pd.DataFrame, side: str) -> np.ndarray:
    # cf. shoe_mean_height : comparaison insensible a la casse.
    columns = [
        column
        for column in mocap
        if column.lower().startswith(f"{side.lower()}_") and column.endswith("_x_mm")
    ]
    if not columns:
        return np.zeros(len(mocap), dtype=int)
    return mocap[columns].notna().sum(axis=1).to_numpy(dtype=int)


def find_one(folder: Path, pattern: str, label: str) -> Path:
    matches = sorted(folder.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"{label}: {len(matches)} fichier(s) trouve(s) avec {pattern}; "
            f"utilisez l'option correspondante pour choisir le fichier"
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Decode les deux semelles sans calibration/tare, supprime les "
            "doublons, les synchronise avec un C3D QTM et exporte 3 CSV."
        )
    )
    parser.add_argument("trial_dir", type=Path, help="dossier de l'acquisition")
    parser.add_argument("--bin", dest="binary_path", type=Path)
    parser.add_argument("--c3d", dest="c3d_path", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--insole-offset-ms",
        type=float,
        help=(
            "retard commun semelles vers QTM; par defaut il est estime "
            "automatiquement sur les contacts"
        ),
    )
    parser.add_argument(
        "--offset-search-ms",
        type=float,
        default=250.0,
        help="demi-largeur de recherche automatique (defaut: 250 ms)",
    )
    parser.add_argument(
        "--contact-threshold-n",
        type=float,
        default=40.0,
        help="seuil de detection d'un contact plateforme (defaut: 40 N)",
    )
    parser.add_argument(
        "--contact-max-window-s",
        type=float,
        default=2.0,
        help=(
            "duree maximale (depuis le debut du contact) utilisee pour la "
            "correlation ; sur un appui statique prolonge, seul le front de "
            "montee est fiable (defaut: 2.0 s)"
        ),
    )
    parser.add_argument(
        "--min-reliable-contacts",
        type=int,
        default=2,
        help="nombre minimal de contacts fiables requis (defaut: 2)",
    )
    parser.add_argument(
        "--pair-tolerance-ms",
        type=float,
        default=4.5,
        help="ecart maximal gauche/droite accepte (defaut: 4.5 ms)",
    )
    parser.add_argument("--tare-left", type=Path, default=DEFAULT_TARE_LEFT)
    parser.add_argument("--tare-right", type=Path, default=DEFAULT_TARE_RIGHT)
    parser.add_argument("--acc-calibration-left", type=Path, default=DEFAULT_ACC_CALIBRATION_LEFT)
    parser.add_argument("--gyro-calibration-left", type=Path, default=DEFAULT_GYRO_CALIBRATION_LEFT)
    parser.add_argument("--acc-calibration-right", type=Path, default=DEFAULT_ACC_CALIBRATION_RIGHT)
    parser.add_argument("--gyro-calibration-right", type=Path, default=DEFAULT_GYRO_CALIBRATION_RIGHT)
    parser.add_argument("--mag-calibration-left", type=Path, default=DEFAULT_MAG_CALIBRATION_LEFT)
    parser.add_argument("--mag-calibration-right", type=Path, default=DEFAULT_MAG_CALIBRATION_RIGHT)
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="desactive tare/calibration IMU/magnetometre (colonnes _N/_cal en NaN)",
    )
    args = parser.parse_args()

    trial_dir = args.trial_dir.resolve()
    binary_path = (args.binary_path or find_one(trial_dir, "*.bin", "Semelles")).resolve()
    c3d_path = (args.c3d_path or find_one(trial_dir, "*.c3d", "C3D")).resolve()
    output_dir = (args.out_dir or trial_dir / "synchronized").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_paths: dict[str, str | None] = {
        "tare_left": None,
        "tare_right": None,
        "acc_calibration_left": None,
        "gyro_calibration_left": None,
        "acc_calibration_right": None,
        "gyro_calibration_right": None,
        "mag_calibration_left": None,
        "mag_calibration_right": None,
    }
    if args.no_calibration:
        tares: dict[str, list[float | None]] = {}
        imu_calibrations: dict[int, dict[str, tuple]] = {}
        mag_calibrations: dict[str, tuple] = {}
    else:
        tares = load_fx29_tares(args.tare_left, args.tare_right)
        imu_calibrations = load_all_imu_calibrations(
            args.acc_calibration_left,
            args.gyro_calibration_left,
            args.acc_calibration_right,
            args.gyro_calibration_right,
        )
        mag_calibrations = load_magnetometer_calibrations(
            args.mag_calibration_left, args.mag_calibration_right
        )
        calibration_paths = {
            "tare_left": str(args.tare_left),
            "tare_right": str(args.tare_right),
            "acc_calibration_left": str(args.acc_calibration_left),
            "gyro_calibration_left": str(args.gyro_calibration_left),
            "acc_calibration_right": str(args.acc_calibration_right),
            "gyro_calibration_right": str(args.gyro_calibration_right),
            "mag_calibration_left": str(args.mag_calibration_left),
            "mag_calibration_right": str(args.mag_calibration_right),
        }

    packets, binary_stats = decode_binary(binary_path)
    left, left_stats = clean_side_packets(packets, "L")
    right, right_stats = clean_side_packets(packets, "R")
    left, right = align_insole_clocks(left, right)
    platform, mocap, c3d_stats = read_c3d(c3d_path)
    if args.insole_offset_ms is None:
        insole_offset_ms, offset_report = estimate_insole_offset(
            left,
            right,
            platform,
            platform_rate_hz=float(c3d_stats["analog_rate_hz"]),
            search_ms=args.offset_search_ms,
            contact_threshold_n=args.contact_threshold_n,
            tares=tares,
            max_event_duration_s=args.contact_max_window_s,
            min_reliable_contacts=args.min_reliable_contacts,
        )
    else:
        insole_offset_ms = float(args.insole_offset_ms)
        offset_report = {
            "mode": "manual",
            "estimated_offset_ms": insole_offset_ms,
            "contact_threshold_n": args.contact_threshold_n,
        }
    _, pair_stats = pair_insoles(
        left,
        right,
        tolerance_ms=args.pair_tolerance_ms,
        offset_ms=insole_offset_ms,
    )
    quality_control = build_insole_quality_control(
        left_stats,
        right_stats,
        pair_stats,
        pair_tolerance_ms=args.pair_tolerance_ms,
    )
    insoles = make_long_insole_table(
        left,
        right,
        insole_offset_ms,
        tares=tares,
        imu_calibrations=imu_calibrations,
        mag_calibrations=mag_calibrations,
    )
    if insoles.empty:
        raise ValueError("Aucune paire gauche/droite apres nettoyage")

    common_start = max(
        float(insoles["time_qtm_absolute_s"].iloc[0]),
        float(platform["time_qtm_absolute_s"].iloc[0]),
        float(mocap["time_qtm_absolute_s"].iloc[0]),
    )
    common_end = min(
        float(insoles["time_qtm_absolute_s"].iloc[-1]),
        float(platform["time_qtm_absolute_s"].iloc[-1]),
        float(mocap["time_qtm_absolute_s"].iloc[-1]),
    )
    if common_end <= common_start:
        raise ValueError("Les trois systemes n'ont aucun intervalle temporel commun")

    insoles = crop_and_rebase(insoles, common_start, common_end)
    platform = crop_and_rebase(platform, common_start, common_end)
    mocap = crop_and_rebase(mocap, common_start, common_end)

    insole_path = output_dir / "semelles_synchronisees.csv"
    platform_path = output_dir / "plateformes_synchronisees.csv"
    mocap_path = output_dir / "mocap_synchronisee.csv"
    insoles.to_csv(insole_path, index=False, float_format="%.9f")
    platform.to_csv(platform_path, index=False, float_format="%.9f")
    mocap.to_csv(mocap_path, index=False, float_format="%.9f", na_rep="")

    insole_rate = float(
        np.mean([left_stats["frequency_hz"], right_stats["frequency_hz"]])
    )
    contact_assignments = classify_platform_contacts(
        insoles,
        platform,
        platform_rate_hz=float(c3d_stats["analog_rate_hz"]),
        contact_threshold_n=args.contact_threshold_n,
    )
    full_plot, zoom_plot, comparison_plot, components_plot = make_plots(
        insoles,
        platform,
        mocap,
        output_dir,
        insole_rate_hz=insole_rate,
        platform_rate_hz=float(c3d_stats["analog_rate_hz"]),
        contact_threshold_n=args.contact_threshold_n,
        contact_assignments=contact_assignments,
    )

    left_visible = count_visible_markers(mocap, "L")
    right_visible = count_visible_markers(mocap, "R")
    report = {
        "inputs": {"binary": str(binary_path), "c3d": str(c3d_path)},
        "parameters": {
            "insole_offset_ms": insole_offset_ms,
            "pair_tolerance_ms": args.pair_tolerance_ms,
            "insole_export": (
                "raw integer counts, uncalibrated nominal physical units "
                "(force_*_nominal_N_no_tare, *_uncal_*), plus calibrated "
                "columns (force_*_N with measured tare, *_cal_* with IMU/"
                "magnetometer correction) when calibration files are found; "
                "NaN in the calibrated columns otherwise"
            ),
            "calibration_files": calibration_paths,
            "platform_export": (
                "force_{x,y,z}_p{n}_N / moment_{...}_p{n}_Nmm: repere local "
                "du capteur de chaque plateforme (tel qu'enregistre par QTM). "
                "force_{x,y,z}_p{n}_global_N / moment_..._global_Nmm: memes "
                "grandeurs tournees dans le repere mocap global (via "
                "FORCE_PLATFORM:CORNERS), Fz_global positif = force de "
                "reaction vers le haut pendant l'appui. Rotation validee "
                "empiriquement par recoupement centre-de-pression/marqueurs "
                "du pied (ecart residuel ~10-80mm)."
            ),
        },
        "quality_control": quality_control,
        "offset_estimation": offset_report,
        "binary": binary_stats,
        "left_insole": left_stats,
        "right_insole": right_stats,
        "pairing": pair_stats,
        "c3d": c3d_stats,
        "common_interval": {
            "qtm_absolute_start_s": common_start,
            "qtm_absolute_end_s": common_end,
            "duration_s": common_end - common_start,
            "insole_rows": len(insoles),
            "platform_rows": len(platform),
            "mocap_rows": len(mocap),
        },
        "mocap_visibility": {
            "left_frames_with_at_least_3_markers": int(np.sum(left_visible >= 3)),
            "right_frames_with_at_least_3_markers": int(np.sum(right_visible >= 3)),
            "total_frames": len(mocap),
        },
        "platform_contact_assignments": contact_assignments,
        "outputs": {
            "insoles_csv": str(insole_path),
            "platforms_csv": str(platform_path),
            "mocap_csv": str(mocap_path),
            "full_plot": str(full_plot),
            "zoom_plot": str(zoom_plot),
            "platform_insole_plot": str(comparison_plot),
            "platform_components_plot": str(components_plot),
        },
    }
    report_path = output_dir / "rapport_synchronisation.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Semelles : {len(insoles)} lignes -> {insole_path}")
    print(f"Plateformes : {len(platform)} lignes -> {platform_path}")
    print(f"Mocap : {len(mocap)} lignes -> {mocap_path}")
    print(
        "Doublons retires : "
        f"gauche={left_stats['duplicates_removed']}, "
        f"droite={right_stats['duplicates_removed']}"
    )
    print(quality_control["signal_1_integrity"]["left"]["message"])
    print(quality_control["signal_1_integrity"]["right"]["message"])
    print(quality_control["left_right_synchronization"]["message"])
    print(
        "Statut global semelles : "
        f"{quality_control['overall_status']} - "
        f"{quality_control['overall_message']}"
    )
    print(
        "Ecart gauche/droite absolu : "
        f"mediane={pair_stats['median_abs_left_right_dt_ms']:.3f} ms, "
        f"p99={pair_stats['p99_abs_left_right_dt_ms']:.3f} ms, "
        f"max={pair_stats['max_abs_left_right_dt_ms']:.3f} ms"
    )
    print(
        f"Intervalle commun : {common_end - common_start:.6f} s; "
        f"decalage semelles/QTM applique : {insole_offset_ms:+.3f} ms "
        f"({offset_report['mode']})"
    )
    print(f"Rapport : {report_path}")
    print(
        f"Figures : {full_plot}, {zoom_plot}, {comparison_plot}, "
        f"{components_plot}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
