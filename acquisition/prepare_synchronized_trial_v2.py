#!/usr/bin/env python3
"""Synchronisation v2 stricte d'un essai semelles/C3D.

Differences principales avec la premiere version :
- politique configurable des pertes, avec localisation par rapport aux
  contacts plateforme pour les essais de marche ;
- estimation d'un retard semelle->QTM distinct pour le pied gauche et droit ;
- controles bloquants sur calibrations, completude des patchs et valeurs finies ;
- export de la geometrie des plateformes necessaire au transport des moments ;
- ecriture dans ``synchronized_v2`` sans modifier les sorties historiques.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import ezc3d
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import prepare_synchronized_trial as legacy
from insole_data.decodageToCSV import load_all_imu_calibrations


PIPELINE_VERSION = "synchronized_v2.2_subject_tare_contact_loss"
PACKET_LOSS_MARGIN_S = 0.05


def json_safe(value):
    """Convertit scalaires NumPy et NaN en JSON strict."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def platform_geometry(c3d_path: Path) -> list[dict[str, object]]:
    c3d = ezc3d.c3d(str(c3d_path))
    parameters = c3d["parameters"]["FORCE_PLATFORM"]
    corners = np.asarray(parameters["CORNERS"]["value"], dtype=float)
    origins = np.asarray(parameters["ORIGIN"]["value"], dtype=float)
    geometry = []
    for index in range(corners.shape[2]):
        rotation, center = legacy.platform_local_to_global_rotation(corners, index)
        origin_local = origins[:, index]
        # Pour une plateforme C3D TYPE-2, ORIGIN va de l'origine mecanique de
        # mesure vers le centre de la surface de travail. Le filtre ezc3d
        # transporte ensuite le moment sur ce plan de travail.
        measurement_origin_global = center - rotation @ origin_local
        geometry.append(
            {
                "platform": index + 1,
                "corners_global_mm": corners[:, :, index].T.tolist(),
                "surface_center_global_mm": center.tolist(),
                "surface_center_from_measurement_origin_local_mm": origin_local.tolist(),
                "measurement_origin_global_mm": measurement_origin_global.tolist(),
                # Alias historique conserve pour les anciens lecteurs.
                "origin_global_mm": measurement_origin_global.tolist(),
                "moment_reference_global_mm": center.tolist(),
                "rotation_local_to_global": rotation.tolist(),
            }
        )
    return geometry


def cop_quality_control(
    platform: pd.DataFrame,
    geometry: list[dict[str, object]],
    contact_threshold_n: float,
    boundary_tolerance_mm: float = 5.0,
) -> dict[str, object]:
    """Verifie le CoP ezc3d depuis le torseur exporte et la geometrie C3D."""
    checks: dict[str, object] = {}
    for item in geometry:
        number = int(item["platform"])
        rotation = np.asarray(item["rotation_local_to_global"], dtype=float)
        center = np.asarray(item["surface_center_global_mm"], dtype=float)
        corners = np.asarray(item["corners_global_mm"], dtype=float)
        force_columns = [
            f"force_{axis}_p{number}_global_N" for axis in ("x", "y", "z")
        ]
        moment_columns = [
            f"moment_{axis}_p{number}_global_Nmm" for axis in ("x", "y", "z")
        ]
        cop_columns = [
            f"cop_{axis}_p{number}_global_mm" for axis in ("x", "y", "z")
        ]
        required = force_columns + moment_columns + cop_columns
        if not all(column in platform for column in required):
            checks[str(number)] = {
                "status": "FAIL",
                "reason": "colonnes force/moment/CoP globales absentes",
            }
            continue

        force_local = platform[force_columns].to_numpy(dtype=float) @ rotation
        moment_local = platform[moment_columns].to_numpy(dtype=float) @ rotation
        cop_local = (
            platform[cop_columns].to_numpy(dtype=float) - center[None, :]
        ) @ rotation
        valid = (
            np.isfinite(force_local).all(axis=1)
            & np.isfinite(moment_local).all(axis=1)
            & np.isfinite(cop_local).all(axis=1)
            & (np.abs(force_local[:, 2]) >= contact_threshold_n)
        )
        count = int(valid.sum())
        if count == 0:
            checks[str(number)] = {
                "status": "FAIL",
                "reason": "aucun echantillon charge avec CoP fini",
                "loaded_sample_count": 0,
            }
            continue

        reconstructed = np.column_stack(
            [
                -moment_local[valid, 1] / force_local[valid, 2],
                moment_local[valid, 0] / force_local[valid, 2],
            ]
        )
        reconstruction_error = np.linalg.norm(
            reconstructed - cop_local[valid, :2], axis=1
        )
        plane_error = np.abs(cop_local[valid, 2])
        corners_local = (corners - center[None, :]) @ rotation
        lower = corners_local[:, :2].min(axis=0) - boundary_tolerance_mm
        upper = corners_local[:, :2].max(axis=0) + boundary_tolerance_mm
        inside = np.all(
            (cop_local[valid, :2] >= lower[None, :])
            & (cop_local[valid, :2] <= upper[None, :]),
            axis=1,
        )
        error_p95 = float(np.quantile(reconstruction_error, 0.95))
        plane_p95 = float(np.quantile(plane_error, 0.95))
        inside_fraction = float(inside.mean())
        status = (
            "OK"
            if error_p95 <= 1.0 and plane_p95 <= 1.0 and inside_fraction >= 0.95
            else "FAIL"
        )
        checks[str(number)] = {
            "status": status,
            "loaded_sample_count": count,
            "contact_threshold_n": float(contact_threshold_n),
            "cop_reconstruction_error_median_mm": float(
                np.median(reconstruction_error)
            ),
            "cop_reconstruction_error_p95_mm": error_p95,
            "cop_plane_distance_p95_mm": plane_p95,
            "cop_inside_platform_fraction": inside_fraction,
            "boundary_tolerance_mm": float(boundary_tolerance_mm),
        }
    failed = [number for number, item in checks.items() if item["status"] != "OK"]
    return {
        "overall_status": "OK" if not failed else "FAIL",
        "failed_platforms": failed,
        "platforms": checks,
    }


def make_cop_validation_plot(
    platform: pd.DataFrame,
    geometry: list[dict[str, object]],
    output_dir: Path,
    contact_threshold_n: float,
) -> Path:
    """Trace le CoP charge dans le plan local de chaque plateforme."""
    output_path = output_dir / "verification_cop_plateformes.png"
    figure, axes = legacy.plt.subplots(1, len(geometry), figsize=(7 * len(geometry), 6))
    axes = np.atleast_1d(axes)
    for axis, item in zip(axes, geometry):
        number = int(item["platform"])
        rotation = np.asarray(item["rotation_local_to_global"], dtype=float)
        center = np.asarray(item["surface_center_global_mm"], dtype=float)
        corners = np.asarray(item["corners_global_mm"], dtype=float)
        force = platform[
            [f"force_{name}_p{number}_global_N" for name in ("x", "y", "z")]
        ].to_numpy(dtype=float)
        cop = platform[
            [f"cop_{name}_p{number}_global_mm" for name in ("x", "y", "z")]
        ].to_numpy(dtype=float)
        force_local = force @ rotation
        cop_local = (cop - center[None, :]) @ rotation
        valid = (
            np.isfinite(cop_local).all(axis=1)
            & np.isfinite(force_local).all(axis=1)
            & (np.abs(force_local[:, 2]) >= contact_threshold_n)
        )
        indexes = np.flatnonzero(valid)
        if len(indexes) > 5000:
            indexes = indexes[np.linspace(0, len(indexes) - 1, 5000).astype(int)]
        corners_local = (corners - center[None, :]) @ rotation
        polygon = np.vstack([corners_local[:, :2], corners_local[0, :2]])
        axis.plot(polygon[:, 0], polygon[:, 1], "k-", linewidth=2, label="plateforme")
        if len(indexes):
            colors = platform["time_qtm_s"].to_numpy(dtype=float)[indexes]
            points = axis.scatter(
                cop_local[indexes, 0],
                cop_local[indexes, 1],
                c=colors,
                s=4,
                alpha=0.45,
                cmap="viridis",
            )
            figure.colorbar(points, ax=axis, label="Temps (s)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x plateforme (mm)")
        axis.set_ylabel("y plateforme (mm)")
        axis.set_title(f"Plateforme {number} — CoP pour |Fz| ≥ {contact_threshold_n:g} N")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    legacy.plt.close(figure)
    return output_path


def side_offsets(
    global_offset_ms: float,
    offset_report: dict[str, object],
    search_ms: float,
) -> tuple[dict[str, float], dict[str, object]]:
    reliable = [
        item
        for item in offset_report.get("contacts", [])
        if float(item.get("correlation", -math.inf)) >= 0.85
        and abs(float(item["best_offset_ms"])) < search_ms - 1.0
    ]
    offsets: dict[str, float] = {}
    details: dict[str, object] = {}
    for side in ("L", "R"):
        values = np.asarray(
            [float(item["best_offset_ms"]) for item in reliable if item["matched_side"] == side],
            dtype=float,
        )
        if len(values):
            offsets[side] = float(np.median(values))
            details[side] = {
                "method": "median_of_reliable_contacts",
                "offset_ms": offsets[side],
                "reliable_contact_count": int(len(values)),
                "q25_ms": float(np.quantile(values, 0.25)),
                "q75_ms": float(np.quantile(values, 0.75)),
            }
        else:
            offsets[side] = float(global_offset_ms)
            details[side] = {
                "method": "global_median_fallback_no_contact_for_side",
                "offset_ms": offsets[side],
                "reliable_contact_count": 0,
            }
    return offsets, details


def apply_side_offsets(insoles: pd.DataFrame, offsets_ms: dict[str, float]) -> pd.DataFrame:
    result = insoles.copy()
    result["side_offset_ms"] = result["side"].map(offsets_ms).astype(float)
    result["time_qtm_absolute_s"] += result["side_offset_ms"] / 1000.0
    return result.sort_values(["time_qtm_absolute_s", "side", "patch_id"]).reset_index(drop=True)


def validate_export(
    insoles: pd.DataFrame,
    left_stats: dict,
    right_stats: dict,
    *,
    tolerate_packet_loss: bool = False,
) -> dict[str, object]:
    checks: dict[str, object] = {}
    losses = {
        "L": int(left_stats["missing_packets_from_counter"]),
        "R": int(right_stats["missing_packets_from_counter"]),
    }
    checks["packet_loss"] = {
        "status": (
            "OK" if not any(losses.values())
            else "WARNING" if tolerate_packet_loss
            else "FAIL"
        ),
        "counts": losses,
    }

    expected = {side: set(values) for side, values in legacy.EXPECTED_PATCHES.items()}
    observed = {
        side: set(insoles.loc[insoles["side"] == side, "patch_id"].astype(int).unique())
        for side in ("L", "R")
    }
    patches_ok = observed == expected
    checks["patch_ids"] = {
        "status": "OK" if patches_ok else "FAIL",
        "expected": {key: sorted(value) for key, value in expected.items()},
        "observed": {key: sorted(value) for key, value in observed.items()},
    }

    count_table = insoles.groupby(["side", "packet_timestamp_unwrapped_us"]).size()
    complete = bool(len(count_table) and (count_table == 4).all())
    checks["four_patches_per_packet"] = {
        "status": "OK" if complete else "FAIL",
        "bad_packet_count": int((count_table != 4).sum()),
    }

    calibrated_columns = [
        column
        for column in insoles
        if column.endswith("_N") or column.endswith("_cal_g") or column.endswith("_cal_dps")
    ]
    calibrated_finite = bool(calibrated_columns) and bool(
        np.isfinite(insoles[calibrated_columns].to_numpy(dtype=float)).all()
    )
    checks["calibrated_force_and_imu_finite"] = {
        "status": "OK" if calibrated_finite else "FAIL",
        "column_count": len(calibrated_columns),
    }
    failures = [name for name, value in checks.items() if value["status"] == "FAIL"]
    checks["overall_status"] = "OK" if not failures else "FAIL"
    checks["failed_checks"] = failures
    return checks


def packet_gaps(insoles: pd.DataFrame, side: str) -> list[dict[str, object]]:
    """Localise les sauts de compteur d'une semelle dans le temps QTM recale."""
    patch_id = legacy.EXPECTED_PATCHES[side][0]
    rows = insoles.loc[
        (insoles["side"] == side) & (insoles["patch_id"] == patch_id),
        ["time_qtm_s", "patch_counter"],
    ].sort_values("time_qtm_s")
    times = rows["time_qtm_s"].to_numpy(dtype=float)
    counters = rows["patch_counter"].to_numpy(dtype=int)
    gaps: list[dict[str, object]] = []
    for index, step in enumerate(np.diff(counters) % 256):
        missing = int(step) - 1
        if missing <= 0:
            continue
        gaps.append(
            {
                "side": side,
                "time_before_qtm_s": float(times[index]),
                "time_after_qtm_s": float(times[index + 1]),
                "center_qtm_s": float((times[index] + times[index + 1]) / 2.0),
                "missing_packets": missing,
            }
        )
    return gaps


def assess_packet_loss(
    insoles: pd.DataFrame,
    assignments: list[dict[str, object]],
    policy: str,
) -> dict[str, object]:
    """Distingue, pour la marche, les pertes hors et pendant appui plateforme."""
    gaps = packet_gaps(insoles, "L") + packet_gaps(insoles, "R")
    critical: list[dict[str, object]] = []
    for gap in gaps:
        for contact in assignments:
            if contact.get("matched_side") != gap["side"]:
                continue
            if (
                float(gap["time_before_qtm_s"]) - PACKET_LOSS_MARGIN_S
                <= float(contact["contact_end_s"])
                and float(gap["time_after_qtm_s"]) + PACKET_LOSS_MARGIN_S
                >= float(contact["contact_start_s"])
            ):
                critical.append(
                    {
                        **gap,
                        "platform": int(contact["platform"]),
                        "contact_start_s": float(contact["contact_start_s"]),
                        "contact_end_s": float(contact["contact_end_s"]),
                    }
                )
                break
    total_missing = int(sum(int(gap["missing_packets"]) for gap in gaps))
    if not gaps:
        status = "NO_LOSS"
    elif policy == "allow":
        status = "ALLOWED"
    elif critical:
        status = "REJECTED_DURING_CONTACT"
    else:
        status = "ACCEPTED_OUTSIDE_CONTACT"
    return {
        "policy": policy,
        "status": status,
        "overlap_margin_s": PACKET_LOSS_MARGIN_S,
        "total_missing_packets": total_missing,
        "gap_count": len(gaps),
        "critical_gap_count": len(critical),
        "gaps": gaps,
        "critical_gaps": critical,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial_dir", type=Path)
    parser.add_argument("--bin", dest="binary_path", type=Path)
    parser.add_argument("--c3d", dest="c3d_path", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--offset-search-ms", type=float, default=250.0)
    parser.add_argument(
        "--contact-threshold-n",
        type=float,
        default=15.0,
        help="seuil bas de detection des appuis plateforme, y compris legers",
    )
    parser.add_argument(
        "--cop-threshold-n",
        type=float,
        default=40.0,
        help="seuil plus eleve reserve au controle numerique du CoP",
    )
    parser.add_argument("--contact-max-window-s", type=float, default=2.0)
    parser.add_argument("--min-reliable-contacts", type=int, default=2)
    parser.add_argument("--pair-tolerance-ms", type=float, default=4.5)
    parser.add_argument("--insole-offset-ms", type=float, help="offset commun manuel")
    parser.add_argument("--insole-offset-left-ms", type=float)
    parser.add_argument("--insole-offset-right-ms", type=float)
    parser.add_argument("--allow-packet-loss", action="store_true")
    parser.add_argument(
        "--packet-loss-policy",
        choices=("reject-any", "reject-during-contact", "allow"),
        default="reject-any",
    )
    parser.add_argument("--tare-left", type=Path, default=legacy.DEFAULT_TARE_LEFT)
    parser.add_argument("--tare-right", type=Path, default=legacy.DEFAULT_TARE_RIGHT)
    parser.add_argument("--acc-calibration-left", type=Path, default=legacy.DEFAULT_ACC_CALIBRATION_LEFT)
    parser.add_argument("--gyro-calibration-left", type=Path, default=legacy.DEFAULT_GYRO_CALIBRATION_LEFT)
    parser.add_argument("--acc-calibration-right", type=Path, default=legacy.DEFAULT_ACC_CALIBRATION_RIGHT)
    parser.add_argument("--gyro-calibration-right", type=Path, default=legacy.DEFAULT_GYRO_CALIBRATION_RIGHT)
    parser.add_argument("--mag-calibration-left", type=Path, default=legacy.DEFAULT_MAG_CALIBRATION_LEFT)
    parser.add_argument("--mag-calibration-right", type=Path, default=legacy.DEFAULT_MAG_CALIBRATION_RIGHT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packet_loss_policy = "allow" if args.allow_packet_loss else args.packet_loss_policy
    trial_dir = args.trial_dir.resolve()
    binary_path = (args.binary_path or legacy.find_one(trial_dir, "*.bin", "Semelles")).resolve()
    c3d_path = (args.c3d_path or legacy.find_one(trial_dir, "*.c3d", "C3D")).resolve()

    calibration_paths = {
        "tare_left": args.tare_left.resolve(),
        "tare_right": args.tare_right.resolve(),
        "acc_calibration_left": args.acc_calibration_left.resolve(),
        "gyro_calibration_left": args.gyro_calibration_left.resolve(),
        "acc_calibration_right": args.acc_calibration_right.resolve(),
        "gyro_calibration_right": args.gyro_calibration_right.resolve(),
        "mag_calibration_left": args.mag_calibration_left.resolve(),
        "mag_calibration_right": args.mag_calibration_right.resolve(),
    }
    missing_calibrations = [str(path) for path in calibration_paths.values() if not path.is_file()]
    if missing_calibrations:
        raise ValueError(f"calibration(s) absente(s): {missing_calibrations}")

    tares = legacy.load_fx29_tares(args.tare_left, args.tare_right)
    imu_calibrations = load_all_imu_calibrations(
        args.acc_calibration_left,
        args.gyro_calibration_left,
        args.acc_calibration_right,
        args.gyro_calibration_right,
    )
    mag_calibrations = legacy.load_magnetometer_calibrations(
        args.mag_calibration_left, args.mag_calibration_right
    )

    packets, binary_stats = legacy.decode_binary(binary_path)
    left, left_stats = legacy.clean_side_packets(packets, "L")
    right, right_stats = legacy.clean_side_packets(packets, "R")
    losses = int(left_stats["missing_packets_from_counter"]) + int(
        right_stats["missing_packets_from_counter"]
    )
    if losses and packet_loss_policy == "reject-any":
        print(
            "EXCLUDED_PACKET_LOSS "
            f"L={left_stats['missing_packets_from_counter']} "
            f"R={right_stats['missing_packets_from_counter']}",
            file=sys.stderr,
        )
        return 2

    left, right = legacy.align_insole_clocks(left, right)
    platform, mocap, c3d_stats = legacy.read_c3d(c3d_path)
    if args.insole_offset_left_ms is not None or args.insole_offset_right_ms is not None:
        if args.insole_offset_left_ms is None or args.insole_offset_right_ms is None:
            raise ValueError("fournir les deux offsets manuels gauche et droit")
        offsets = {"L": float(args.insole_offset_left_ms), "R": float(args.insole_offset_right_ms)}
        offset_report: dict[str, object] = {
            "mode": "manual_side_specific",
            "per_side": {
                side: {"method": "manual", "offset_ms": value} for side, value in offsets.items()
            },
        }
    elif args.insole_offset_ms is not None:
        offsets = {"L": float(args.insole_offset_ms), "R": float(args.insole_offset_ms)}
        offset_report = {
            "mode": "manual_common",
            "per_side": {
                side: {"method": "manual_common", "offset_ms": value} for side, value in offsets.items()
            },
        }
    else:
        global_offset, raw_offset_report = legacy.estimate_insole_offset(
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
        offsets, per_side = side_offsets(global_offset, raw_offset_report, args.offset_search_ms)
        offset_report = {
            **raw_offset_report,
            "mode": "automatic_side_specific_contact_correlation",
            "global_fallback_offset_ms": global_offset,
            "per_side": per_side,
        }

    _, pair_stats = legacy.pair_insoles(left, right, args.pair_tolerance_ms, 0.0)
    quality_control = legacy.build_insole_quality_control(
        left_stats, right_stats, pair_stats, args.pair_tolerance_ms
    )
    insoles = legacy.make_long_insole_table(
        left,
        right,
        0.0,
        tares=tares,
        imu_calibrations=imu_calibrations,
        mag_calibrations=mag_calibrations,
    )
    insoles = apply_side_offsets(insoles, offsets)
    export_checks = validate_export(
        insoles,
        left_stats,
        right_stats,
        tolerate_packet_loss=packet_loss_policy != "reject-any",
    )
    if export_checks["overall_status"] != "OK":
        raise ValueError(f"controle qualite bloquant: {export_checks['failed_checks']}")

    starts = [
        float(insoles.loc[insoles["side"] == side, "time_qtm_absolute_s"].min())
        for side in ("L", "R")
    ]
    ends = [
        float(insoles.loc[insoles["side"] == side, "time_qtm_absolute_s"].max())
        for side in ("L", "R")
    ]
    common_start = max(*starts, float(platform["time_qtm_absolute_s"].min()), float(mocap["time_qtm_absolute_s"].min()))
    common_end = min(*ends, float(platform["time_qtm_absolute_s"].max()), float(mocap["time_qtm_absolute_s"].max()))
    if common_end <= common_start:
        raise ValueError("aucun intervalle commun aux deux semelles et au C3D")
    insoles = legacy.crop_and_rebase(insoles, common_start, common_end)
    platform = legacy.crop_and_rebase(platform, common_start, common_end)
    mocap = legacy.crop_and_rebase(mocap, common_start, common_end)

    assignments = legacy.classify_platform_contacts(
        insoles,
        platform,
        float(c3d_stats["analog_rate_hz"]),
        args.contact_threshold_n,
    )
    loss_assessment = assess_packet_loss(insoles, assignments, packet_loss_policy)
    if loss_assessment["status"] == "REJECTED_DURING_CONTACT":
        print(
            "EXCLUDED_PACKET_LOSS_DURING_CONTACT "
            + json.dumps(json_safe(loss_assessment["critical_gaps"]), ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    geometry_report = platform_geometry(c3d_path)
    cop_checks = cop_quality_control(
        platform,
        geometry_report,
        args.cop_threshold_n,
    )
    output_dir = (args.out_dir or trial_dir / "synchronized_v2").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    insoles.to_csv(output_dir / "semelles_synchronisees.csv", index=False, float_format="%.9f")
    platform.to_csv(output_dir / "plateformes_synchronisees.csv", index=False, float_format="%.9f")
    mocap.to_csv(output_dir / "mocap_synchronisee.csv", index=False, float_format="%.9f", na_rep="")

    insole_rate = float(np.mean([left_stats["frequency_hz"], right_stats["frequency_hz"]]))
    plots = list(legacy.make_plots(
        insoles,
        platform,
        mocap,
        output_dir,
        insole_rate_hz=insole_rate,
        platform_rate_hz=float(c3d_stats["analog_rate_hz"]),
        contact_threshold_n=args.contact_threshold_n,
        contact_assignments=assignments,
    ))
    plots.append(
        make_cop_validation_plot(
            platform,
            geometry_report,
            output_dir,
            args.cop_threshold_n,
        )
    )
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "status": (
            "OK"
            if export_checks["overall_status"] == "OK"
            and cop_checks["overall_status"] == "OK"
            else "WARNING"
        ),
        "inputs": {
            "binary": str(binary_path),
            "binary_sha256": sha256(binary_path),
            "c3d": str(c3d_path),
            "c3d_sha256": sha256(c3d_path),
        },
        "parameters": {
            "contact_threshold_n": args.contact_threshold_n,
            "cop_threshold_n": args.cop_threshold_n,
            "pair_tolerance_ms": args.pair_tolerance_ms,
            "packet_loss_policy": packet_loss_policy,
            "calibration_files": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in calibration_paths.items()
            },
        },
        "binary": binary_stats,
        "left_insole": left_stats,
        "right_insole": right_stats,
        "quality_control": quality_control,
        "v2_export_checks": export_checks,
        "platform_cop_quality": cop_checks,
        "packet_loss_assessment": loss_assessment,
        "offset_estimation": offset_report,
        "c3d": c3d_stats,
        "platform_geometry": geometry_report,
        "contact_assignments": assignments,
        "common_interval": {
            "qtm_absolute_start_s": common_start,
            "qtm_absolute_end_s": common_end,
            "duration_s": common_end - common_start,
        },
        "marker_visibility": {
            side: {
                "median_visible": float(np.median(legacy.count_visible_markers(mocap, side))),
                "fraction_at_least_3": float(np.mean(legacy.count_visible_markers(mocap, side) >= 3)),
            }
            for side in ("L", "R")
        },
        "outputs": {
            "semelles": "semelles_synchronisees.csv",
            "plateformes": "plateformes_synchronisees.csv",
            "mocap": "mocap_synchronisee.csv",
            "plots": [path.name for path in plots],
        },
    }
    (output_dir / "rapport_synchronisation.json").write_text(
        json.dumps(json_safe(report), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(f"OK {trial_dir} -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
