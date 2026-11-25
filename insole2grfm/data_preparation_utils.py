"""Fonctions partagees pour construire les datasets GRFM deux pieds.

Ce module ne constitue pas un pipeline executable. Il regroupe uniquement les
constantes, la reconstruction des reperes pied, l'interpolation plateforme et
les fonctions de masque reutilisees par la preparation native et les audits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


TARGET_AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
FORCE_COLUMNS = ("force_1_N", "force_2_N", "force_3_N", "force_4_N")
IMU_COLUMNS = (
    "acc_x_cal_g",
    "acc_y_cal_g",
    "acc_z_cal_g",
    "gyro_x_cal_dps",
    "gyro_y_cal_dps",
    "gyro_z_cal_dps",
)
PATCH_IDS = {"L": (1, 3, 5, 7), "R": (9, 11, 13, 15)}
ABSENT_FORCE_SENSORS = {(7, 1), (15, 1)}
STATE_UNKNOWN = np.int8(0)
STATE_PLATFORM = np.int8(1)
STATE_AIRBORNE = np.int8(2)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIP = PROJECT_DIR.parent / "manip"
EXCLUDED_SUBJECTS = {"pilot1", "pilot2"}  # pilot sessions, not in the dataset
TARE2_SUBJECTS = {"P01", "P01b"}


def discover_synchronized_trials(roots: list[Path]) -> list[tuple[str, Path]]:
    """Trouve les essais possedant un rapport ``synchronized_v2``."""
    trials: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        candidates = [root, *(path for path in root.rglob("*") if path.is_dir())]
        for trial in candidates:
            report = trial / "synchronized_v2" / "rapport_synchronisation.json"
            if trial in seen or not report.is_file():
                continue
            seen.add(trial)
            subject = trial.parent.name
            trials.append((f"{subject}_{trial.name}", trial))
    return sorted(trials)


def choose_c3d(trial: Path) -> Path:
    """Choisit le C3D gap-filled, sinon exige un unique C3D brut."""
    candidates = sorted(trial.glob("*.c3d"))
    gapfilled = [path for path in candidates if path.stem.casefold().endswith("_gapf")]
    if len(gapfilled) == 1:
        return gapfilled[0]
    if len(gapfilled) > 1:
        raise ValueError(
            f"plusieurs C3D gap-filled: {[path.name for path in gapfilled]}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"{len(candidates)} C3D trouves: {[path.name for path in candidates]}"
        )
    return candidates[0]


def tare_paths(manip_root: Path, subject: str) -> tuple[str, Path, Path]:
    """Retourne la paire de tares attendue pour un sujet de la campagne."""
    tare_name = "tare2" if subject in TARE2_SUBJECTS else "tare1"
    folder = manip_root / tare_name
    return (
        tare_name,
        folder / "tare_fx29_gauche.json",
        folder / "tare_fx29_droit.json",
    )


def discover_raw_trials(
    manip_root: Path,
    selected_subjects: set[str] | None = None,
    include_static: bool = False,
) -> list[tuple[str, Path]]:
    """Enumere les essais bruts de la campagne, hors sujets exclus."""
    trials: list[tuple[str, Path]] = []
    for subject_dir in sorted(path for path in manip_root.iterdir() if path.is_dir()):
        subject = subject_dir.name
        if subject.casefold() in EXCLUDED_SUBJECTS or subject in {"tare1", "tare2"}:
            continue
        if selected_subjects and subject not in selected_subjects:
            continue
        for trial in sorted(path for path in subject_dir.iterdir() if path.is_dir()):
            if not include_static and trial.name.casefold().startswith("static"):
                continue
            if list(trial.glob("*.bin")) or list(trial.glob("*.c3d")):
                trials.append((subject, trial))
    return trials


def nearest_indices(
    reference: np.ndarray, query: np.ndarray, tolerance_s: float
) -> np.ndarray:
    """Index du voisin temporel le plus proche, ou -1 hors tolerance."""
    if len(reference) < 2:
        return np.full(len(query), -1, dtype=np.int64)
    positions = np.searchsorted(reference, query)
    positions = np.clip(positions, 1, len(reference) - 1)
    left = positions - 1
    choose_left = np.abs(query - reference[left]) <= np.abs(
        query - reference[positions]
    )
    result = np.where(choose_left, left, positions).astype(np.int64)
    result[np.abs(reference[result] - query) > tolerance_s] = -1
    return result


def marker_arrays(mocap: pd.DataFrame, side: str) -> dict[str, np.ndarray]:
    """Regroupe les colonnes mocap XYZ par marqueur d'un pied."""
    suffixes = {"_x_mm": 0, "_y_mm": 1, "_z_mm": 2}
    grouped: dict[str, list[str | None]] = {}
    for column in mocap.columns:
        lowered = column.lower()
        if not lowered.startswith(f"{side.lower()}_"):
            continue
        for suffix, axis in suffixes.items():
            if lowered.endswith(suffix):
                base = lowered[: -len(suffix)]
                grouped.setdefault(base, [None, None, None])[axis] = column
    result = {}
    for base, columns in grouped.items():
        if all(column is not None for column in columns):
            result[base] = mocap[columns].to_numpy(dtype=np.float64)
    if len(result) < 3:
        raise ValueError(f"moins de 3 marqueurs trouves pour {side}")
    return result


def anatomical_seed(
    markers: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construit les axes anatomiques initiaux heel-toe/medial-lateral."""

    def group(keyword: str) -> np.ndarray:
        arrays = [values for name, values in markers.items() if keyword in name]
        if not arrays:
            raise ValueError(f"marqueur '{keyword}' absent")
        stack = np.stack(arrays, axis=1)
        count = np.sum(np.isfinite(stack), axis=1)
        total = np.nansum(stack, axis=1)
        return np.divide(
            total, count, out=np.full_like(total, np.nan), where=count > 0
        )

    heel, toe = group("heel"), group("toe")
    medial, lateral = group("medial"), group("lateral")
    x_axis = toe - heel
    x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    z_axis = np.cross(x_axis, lateral - medial)
    z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)
    z_axis[z_axis[:, 2] < 0] *= -1.0
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=2)
    valid = np.isfinite(rotation).all(axis=(1, 2)) & np.isfinite(heel).all(axis=1)
    return rotation, heel, valid


def rigid_foot_frames(
    mocap: pd.DataFrame, side: str
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Reconstruit le repere avec Kabsch et n'importe quels >=3 marqueurs visibles."""
    markers = marker_arrays(mocap, side)
    seed_rotation, seed_origin, seed_valid = anatomical_seed(markers)
    if int(seed_valid.sum()) < 3:
        raise ValueError(f"repere anatomique initial insuffisant pour {side}")

    template: dict[str, np.ndarray] = {}
    for name, observed in markers.items():
        valid = seed_valid & np.isfinite(observed).all(axis=1)
        if not valid.any():
            continue
        local = np.einsum(
            "nij,ni->nj", seed_rotation[valid], observed[valid] - seed_origin[valid]
        )
        template[name] = np.nanmedian(local, axis=0)
    if len(template) < 3:
        raise ValueError(f"gabarit rigide insuffisant pour {side}")
    template_centroid = np.mean(np.stack(list(template.values())), axis=0)
    template = {name: value - template_centroid for name, value in template.items()}

    count = len(mocap)
    rotations = np.full((count, 3, 3), np.nan, dtype=np.float64)
    origins = np.full((count, 3), np.nan, dtype=np.float64)
    residuals = np.full(count, np.nan, dtype=np.float64)
    names = sorted(template)
    for frame in range(count):
        visible = [name for name in names if np.isfinite(markers[name][frame]).all()]
        if len(visible) < 3:
            continue
        local = np.stack([template[name] for name in visible])
        global_points = np.stack([markers[name][frame] for name in visible])
        local_center, global_center = local.mean(axis=0), global_points.mean(axis=0)
        local_centered = local - local_center
        global_centered = global_points - global_center
        if np.linalg.matrix_rank(local_centered, tol=1e-6) < 2:
            continue
        u, _, vt = np.linalg.svd(local_centered.T @ global_centered)
        rotation = vt.T @ u.T
        if np.linalg.det(rotation) < 0:
            vt[-1] *= -1
            rotation = vt.T @ u.T
        origin = global_center - rotation @ local_center
        fitted = local @ rotation.T + origin
        rotations[frame] = rotation
        origins[frame] = origin
        residuals[frame] = float(
            np.sqrt(np.mean(np.square(fitted - global_points)))
        )
    valid = np.isfinite(rotations).all(axis=(1, 2)) & (residuals <= 20.0)
    rotations[~valid] = np.nan
    origins[~valid] = np.nan
    info = {
        "template_marker_count": len(template),
        "origin_definition": "rigid_marker_cluster_centroid",
        "valid_fraction": float(valid.mean()),
        "median_rigid_fit_rmse_mm": (
            float(np.nanmedian(residuals[valid])) if valid.any() else None
        ),
        "max_allowed_fit_rmse_mm": 20.0,
    }
    return rotations, origins, info


def filter_and_interpolate_platform(
    platforms: pd.DataFrame, query_times: np.ndarray, cutoff_hz: float
) -> dict[str, np.ndarray]:
    """Filtre les GRFM globaux puis les interpole aux temps demandes."""
    source_time = platforms["time_qtm_s"].to_numpy(dtype=np.float64)
    rate = 1.0 / np.median(np.diff(source_time))
    sos = butter(4, cutoff_hz, btype="lowpass", fs=rate, output="sos")
    result = {}
    columns = [
        column
        for column in platforms
        if (
            column.startswith("force_")
            or column.startswith("moment_")
            or column.startswith("cop_")
        )
        and "_global_" in column
    ]
    for column in columns:
        values = platforms[column].to_numpy(dtype=np.float64)
        # A very short NaN tail can occur when the common C3D/insole interval
        # ends between two analogue samples (observed on P02b/walk).  A
        # single NaN would otherwise make sosfiltfilt return NaN for the whole
        # trial.  Interpolate only from finite force-platform samples before
        # filtering; np.interp holds the nearest finite value at an edge.
        finite = np.isfinite(values) & np.isfinite(source_time)
        if finite.sum() < 2:
            raise ValueError(f"signal plateforme insuffisant: {column}")
        if not finite.all():
            values = np.interp(source_time, source_time[finite], values[finite])
        # CoP is undefined off-platform and must not be low-pass filtered
        # across separate contacts. It is only interpolated here and later
        # consumed inside an already detected contact interval. Forces and
        # moments, on the other hand, receive the common label filter.
        filtered = (
            values
            if column.startswith("cop_")
            else sosfiltfilt(sos, values) if len(values) > 30 else values
        )
        result[column] = np.interp(query_times, source_time, filtered)
    return result


def hysteresis_airborne(
    force: np.ndarray, low_n: float, high_n: float
) -> np.ndarray:
    """Detecte l'etat pied en l'air avec un hysteresis sur les jauges."""
    airborne = np.zeros(len(force), dtype=bool)
    state = bool(force[0] <= low_n)
    for index, value in enumerate(force):
        if state and value >= high_n:
            state = False
        elif not state and value <= low_n:
            state = True
        airborne[index] = state
    return airborne


def active_platforms(
    times: np.ndarray,
    contacts: list[dict[str, object]],
    side: str,
    minimum_correlation: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Associe chaque instant d'un pied a une plateforme, avec ambiguite explicite."""
    platform = np.zeros(len(times), dtype=np.int8)
    ambiguous = np.zeros(len(times), dtype=bool)
    for contact in contacts:
        if contact.get("matched_side") != side:
            continue
        correlation = contact.get("correlation")
        # A spatially validated assignment is authoritative. Correlation with
        # the gauges is only a complementary tie-breaker and must not discard
        # a real low-load or poorly calibrated platform contact afterwards.
        spatially_validated = bool(contact.get("spatially_validated", False))
        if not spatially_validated and (
            correlation is not None
            and np.isfinite(correlation)
            and correlation < minimum_correlation
        ):
            continue
        inside = (times >= float(contact["contact_start_s"])) & (
            times <= float(contact["contact_end_s"])
        )
        already = inside & (platform != 0) & (platform != int(contact["platform"]))
        ambiguous |= already
        platform[inside & (platform == 0)] = int(contact["platform"])
    platform[ambiguous] = 0
    return platform, ambiguous
