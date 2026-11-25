#!/usr/bin/env python3
"""Utilitaires partages par les scripts de traitement et synchronisation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Iterable


OUTPUT_DIRECTORIES = {"synchronized", "synchronized_v2"}
_TARE_BUILD_LOCK = Lock()


def ensure_fx29_tare_json(
    folder: Path, foot: str, *, materialize: bool = True
) -> Path:
    """Retourne la tare JSON d'un pied, en la calculant depuis le BIN si besoin.

    Seuls les fichiers ``tare_fx29_*.bin`` sont consideres. Une acquisition
    ``dataSemelle.bin``/``dataSemell.bin`` n'est donc jamais utilisee comme
    tare; elle reste un controle independant pieds leves.
    """
    if foot not in {"gauche", "droit"}:
        raise ValueError(f"pied de tare inconnu: {foot}")
    output = folder / f"tare_fx29_{foot}.json"
    if output.is_file() or not materialize:
        return output
    binary = folder / f"tare_fx29_{foot}.bin"
    if not binary.is_file():
        raise FileNotFoundError(f"tare {foot} absente: {binary}")

    # Plusieurs essais d'un meme sujet sont traites en parallele. Le verrou
    # evite que plusieurs workers generent simultanement le meme JSON.
    with _TARE_BUILD_LOCK:
        if output.is_file():
            return output
        from insole_data.calculerTareFX29 import compute_tares, read_samples

        samples, packet_counts = read_samples(binary)
        source_foot = "gauche" if foot == "gauche" else "droite"
        if packet_counts[source_foot] == 0:
            raise ValueError(f"aucun paquet {source_foot} dans {binary}")
        tares, statistics = compute_tares(samples, source_foot)
        document = {
            "format": "fx29_tare_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_file": str(binary.resolve()),
            "foot": source_foot,
            "frequency_hz": 200.0,
            "packet_count": packet_counts[source_foot],
            "nominal_span_counts": 14000.0,
            "tares": tares,
            "statistics": statistics,
            "generated_by": "process_manip_v2.ensure_fx29_tare_json",
        }
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output)
    return output


def find_many(folder: Path, pattern: str) -> list[Path]:
    """Retourne les fichiers correspondant au motif, dans un ordre stable."""
    return sorted(folder.glob(pattern))


def resolve_sync_dir(trial_dir: Path, requested: str | None = None) -> Path:
    """Trouve la sortie synchronisee, avec priorite a la version v2."""
    if requested:
        path = trial_dir / requested
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    for name in ("synchronized_v2", "synchronized"):
        path = trial_dir / name
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"aucun dossier synchronized_v2/ ou synchronized/ dans {trial_dir}"
    )


def choose_c3d(trial_dir: Path) -> Path:
    """Choisit le C3D gap-filled quand il existe, sinon l'unique C3D."""
    candidates = find_many(trial_dir, "*.c3d")
    gapfilled = [path for path in candidates if path.stem.casefold().endswith("_gapf")]
    if len(gapfilled) == 1:
        return gapfilled[0]
    if len(gapfilled) > 1:
        raise ValueError(
            f"plusieurs C3D gap-filled dans {trial_dir}: "
            f"{[path.name for path in gapfilled]}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"{len(candidates)} C3D trouves dans {trial_dir}: "
            f"{[path.name for path in candidates]}"
        )
    return candidates[0]


def discover_trials(
    root: Path,
    *,
    subjects: Iterable[str] | None = None,
    trials: Iterable[str] | None = None,
    excluded_subjects: Iterable[str] = (),
    include_static: bool = False,
) -> list[Path]:
    """Decouvre les essais bruts et applique les filtres sujet/essai.

    ``root`` peut etre le dossier ``manip``, un dossier sujet ou directement
    un dossier essai. Un filtre ``--trial`` peut etre un nom simple
    (``walk``) ou un identifiant ``Sujet/walk``.
    """
    root = root.resolve()
    subject_filter = {value.casefold() for value in subjects or ()}
    trial_filter = {value.strip("/").casefold() for value in trials or ()}
    excluded = {value.casefold() for value in excluded_subjects}
    candidates = [root]
    candidates.extend(path for path in root.rglob("*") if path.is_dir())
    selected: list[Path] = []
    for path in candidates:
        if path.name in OUTPUT_DIRECTORIES or path.name.startswith("."):
            continue
        binaries = find_many(path, "*.bin")
        if len(binaries) != 1 or not find_many(path, "*.c3d"):
            continue
        subject = path.parent.name
        if subject.casefold() in excluded:
            continue
        if subject_filter and subject.casefold() not in subject_filter:
            continue
        if not include_static and path.name.casefold().startswith("static"):
            continue
        if trial_filter:
            relative = str(path.relative_to(root)).replace("\\", "/").casefold()
            subject_trial = f"{subject}/{path.name}".casefold()
            if (
                path.name.casefold() not in trial_filter
                and relative not in trial_filter
                and subject_trial not in trial_filter
            ):
                continue
        selected.append(path)
    return sorted(set(selected))


def insole_active_duration_s(report: dict) -> float:
    """Duree moyenne Signal=1, avec repli sur l'intervalle commun."""
    if "left_insole" in report and "right_insole" in report:
        return float(
            (report["left_insole"]["duration_s"] + report["right_insole"]["duration_s"])
            / 2.0
        )
    return float(report["common_interval"]["duration_s"])


def load_trigger_on_time(trigger_log_path: Path) -> float | None:
    """Lit la premiere transition vers Signal=1 du journal d'acquisition."""
    with trigger_log_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["switch_value"]) == 1:
                return float(row["timestamp_s"])
    return None


def align_video_frames(
    video_path: Path,
    trial_dir: Path,
    output_dir: Path,
    insole_duration_s: float,
) -> dict[str, object]:
    """Ecrit le mapping frame video -> temps QTM d'un essai synchronise."""
    timestamps_path = video_path.with_name(video_path.stem + "_timestamps.csv")
    trigger_log_path = trial_dir / "trigger_log.csv"
    output_csv = output_dir / f"{video_path.stem}_frames_synchronises.csv"

    if timestamps_path.is_file() and trigger_log_path.is_file():
        trigger_on_s = load_trigger_on_time(trigger_log_path)
        if trigger_on_s is None:
            raise ValueError(f"aucune transition Signal=1 dans {trigger_log_path}")
        with timestamps_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        with output_csv.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_index", "time_qtm_absolute_s"])
            for row in rows:
                writer.writerow(
                    [int(row["frame_index"]), f"{float(row['timestamp_s']) - trigger_on_s:.6f}"]
                )
        return {
            "method": "measured_timestamps",
            "confidence": "high",
            "frame_count": len(rows),
            "timestamps_file": str(timestamps_path),
            "mapping_file": output_csv.name,
            "note": (
                "Horodatage reel de chaque frame recale sur la premiere "
                "transition Signal=1 de trigger_log.csv."
            ),
        }

    # Import paresseux: OpenCV n'est necessaire que pour les anciennes videos
    # depourvues de timestamps mesures.
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_nominal = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    if frame_count <= 1 or insole_duration_s <= 0:
        raise ValueError(f"impossible d'estimer le temps des frames de {video_path}")
    fps_effective = (frame_count - 1) / insole_duration_s
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["frame_index", "time_qtm_absolute_s"])
        for frame_index in range(frame_count):
            writer.writerow([frame_index, f"{frame_index / fps_effective:.6f}"])
    return {
        "method": "fallback_constant_fps_estimate",
        "confidence": "low",
        "frame_count": frame_count,
        "insole_active_duration_s": insole_duration_s,
        "fps_effective_hz": fps_effective,
        "fps_nominal_declared_hz": fps_nominal,
        "mapping_file": output_csv.name,
        "note": (
            "Pas de timestamps mesures; temps estime uniformement a partir "
            "du nombre de frames et de la duree active des semelles."
        ),
    }


def align_trial_videos(trial_dir: Path, output_dir: Path, report: dict) -> dict[str, dict]:
    """Genere les mappings de toutes les videos MP4 presentes dans un essai."""
    duration_s = insole_active_duration_s(report)
    result = {
        video.name: align_video_frames(video, trial_dir, output_dir, duration_s)
        for video in find_many(trial_dir, "*.mp4")
    }
    (output_dir / "synchronisation_video.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
