#!/usr/bin/env python3
"""Construit les dossiers ``synchronized_v2`` et le mapping des videos.

Le lanceur accepte un dossier campagne, sujet ou essai. Il permet de filtrer
par sujet et/ou essai, prefere les C3D ``*_gapf.c3d`` et applique la tare du
sujet: celle de son dossier ``<sujet>/tare/`` quand il existe (dataset publie),
sinon la tare dediee declaree dans ``tare_assignments.json``. Toute perte exclut les taches
en appui continu; pour la marche, seules les pertes qui chevauchent l'appui
du pied concerne sur une plateforme sont bloquantes.

Exemples::

    python process_data/process_manip_v2.py manip
    python process_data/process_manip_v2.py manip --subject P12
    python process_data/process_manip_v2.py manip --subject P12 --trial lateral

Le mapping ``frame_index -> time_qtm_absolute_s`` est ecrit directement dans
``synchronized_v2``. Les timestamps mesures sont utilises quand ils existent;
le repli a frequence constante est explicitement marque comme peu fiable.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from utils import (
    align_trial_videos,
    choose_c3d,
    discover_trials,
    ensure_fx29_tare_json,
    find_many,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SYNC_SCRIPT = SCRIPT_DIR / "prepare_synchronized_trial_v2.py"
DEFAULT_EXCLUDED_SUBJECTS = ("pilot1", "pilot2")
DEFAULT_TARE_MANIFEST = SCRIPT_DIR / "tare_assignments.json"


def find_campaign_root(root: Path) -> Path | None:
    """Trouve l'ancetre contenant tare1/ et tare2/."""
    for candidate in (root.resolve(), *root.resolve().parents):
        if (candidate / "tare1").is_dir() and (candidate / "tare2").is_dir():
            return candidate
    return None


def tare_paths(
    campaign_root: Path | None,
    subject: str,
    requested_set: str,
    assignments: dict[str, str],
    *,
    materialize: bool,
) -> tuple[str | None, Path | None, Path | None]:
    """Resout la paire de tares dediee a partir du manifeste de campagne."""
    if requested_set == "none":
        return None, None, None
    if campaign_root is None:
        raise ValueError(
            "dossiers tare1/tare2 introuvables; utilisez --tare-root, "
            "--tare-set none ou fournissez une racine de campagne"
        )
    name = requested_set
    if name == "auto":
        try:
            name = assignments[subject.casefold()]
        except KeyError as exc:
            raise ValueError(
                f"aucune tare declaree pour {subject}; completez le manifeste"
            ) from exc
    folder = campaign_root / name
    if not folder.is_dir():
        raise ValueError(f"dossier de tare {name} absent: {folder}")
    left = ensure_fx29_tare_json(folder, "gauche", materialize=materialize)
    right = ensure_fx29_tare_json(folder, "droit", materialize=materialize)
    if materialize and (not left.is_file() or not right.is_file()):
        raise ValueError(f"fichiers de tare {name} absents dans {folder}")
    return name, left, right


def load_tare_assignments(path: Path) -> dict[str, str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    values = document.get("assignments")
    if not isinstance(values, dict) or not values:
        raise ValueError(f"{path}: objet 'assignments' absent ou vide")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in values.items()):
        raise ValueError(f"{path}: les affectations doivent etre sujet -> dossier")
    return {key.casefold(): value for key, value in values.items()}


def run_sync(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True)


def process_trial(
    trial: Path,
    root: Path,
    *,
    campaign_root: Path | None,
    tare_set: str,
    tare_assignments: dict[str, str],
    walking_loss_policy: str,
    force: bool,
    refresh_video_map: bool,
    map_videos: bool,
    passthrough: list[str],
    dry_run: bool,
) -> tuple[str, dict[str, object], str]:
    subject = trial.parent.name
    key = str(trial.relative_to(root)) if trial != root else f"{subject}/{trial.name}"
    output = trial / "synchronized_v2"
    temporary = trial / ".synchronized_v2_build"
    item: dict[str, object] = {"subject": subject, "trial": trial.name}
    try:
        binaries = find_many(trial, "*.bin")
        if len(binaries) != 1:
            raise ValueError(f"{len(binaries)} fichiers .bin dans {trial}")
        c3d = choose_c3d(trial)
        own_tare = trial.parent / "tare"
        if tare_set == "auto" and any(own_tare.glob("tare_fx29_gauche.*")):
            # Released dataset: each Subject folder holds the tare it was processed with.
            selected_tare = "tare"
            tare_left = ensure_fx29_tare_json(own_tare, "gauche", materialize=not dry_run)
            tare_right = ensure_fx29_tare_json(own_tare, "droit", materialize=not dry_run)
        else:
            selected_tare, tare_left, tare_right = tare_paths(
                campaign_root,
                subject,
                tare_set,
                tare_assignments,
                materialize=not dry_run,
            )
        packet_loss_policy = (
            walking_loss_policy
            if trial.name.casefold().startswith("walk")
            else "reject-any"
        )
        item.update(
            binary=str(binaries[0]),
            c3d=str(c3d),
            tare_set=selected_tare,
            tare_left=str(tare_left) if tare_left else None,
            tare_right=str(tare_right) if tare_right else None,
            packet_loss_policy=packet_loss_policy,
        )

        if dry_run:
            item["status"] = "dry_run"
            return key, item, f"DRY {key}: {c3d.name}, {selected_tare or 'tares par defaut'}"

        report_path = output / "rapport_synchronisation.json"
        if report_path.is_file() and not force:
            video_report = output / "synchronisation_video.json"
            if map_videos and (refresh_video_map or not video_report.is_file()):
                report = json.loads(report_path.read_text(encoding="utf-8"))
                videos = align_trial_videos(trial, output, report)
                item["video_mapping"] = {
                    name: info["method"] for name, info in videos.items()
                }
            item["status"] = "already_done"
            return key, item, f"DEJA {key}"

        if temporary.exists():
            shutil.rmtree(temporary)
        command = [
            sys.executable,
            str(SYNC_SCRIPT),
            str(trial),
            "--bin",
            str(binaries[0]),
            "--c3d",
            str(c3d),
            "--out-dir",
            str(temporary),
            "--packet-loss-policy",
            packet_loss_policy,
            *passthrough,
        ]
        if tare_left is not None and tare_right is not None:
            command.extend(["--tare-left", str(tare_left), "--tare-right", str(tare_right)])

        completed = run_sync(command)
        combined = completed.stdout + "\n" + completed.stderr
        offset_search_ms = 250.0
        if completed.returncode == 1 and "limite de recherche" in combined:
            if temporary.exists():
                shutil.rmtree(temporary)
            offset_search_ms = 1000.0
            completed = run_sync([*command, "--offset-search-ms", "1000"])
            combined = completed.stdout + "\n" + completed.stderr

        if completed.returncode == 2 and "EXCLUDED_PACKET_LOSS" in combined:
            if temporary.exists():
                shutil.rmtree(temporary)
            if output.exists():
                shutil.rmtree(output)
            item.update(status="excluded_packet_loss", reason=combined.strip())
            return key, item, f"EXCLU {key}: perte de paquets"
        if completed.returncode != 0:
            raise RuntimeError(combined.strip())

        report = json.loads(
            (temporary / "rapport_synchronisation.json").read_text(encoding="utf-8")
        )
        if report.get("status") != "OK":
            raise RuntimeError("rapport synchronized_v2 non OK")
        if report.get("v2_export_checks", {}).get("overall_status") != "OK":
            raise RuntimeError("controle export synchronized_v2 non OK")
        if report.get("platform_cop_quality", {}).get("overall_status") != "OK":
            raise RuntimeError("controle CoP plateformes synchronized_v2 non OK")

        videos: dict[str, dict] = {}
        if map_videos:
            videos = align_trial_videos(trial, temporary, report)
        if output.exists():
            shutil.rmtree(output)
        temporary.replace(output)
        item.update(
            status="ok",
            output=str(output),
            offset_search_ms=offset_search_ms,
            video_mapping={name: info["method"] for name, info in videos.items()},
        )
        return key, item, f"OK {key}: {c3d.name}, {selected_tare or 'tares par defaut'}"
    except Exception as exc:
        if temporary.exists():
            shutil.rmtree(temporary)
        item.update(status="error", reason=str(exc))
        last_line = str(exc).strip().splitlines()[-1] if str(exc).strip() else repr(exc)
        return key, item, f"ERREUR {key}: {last_line}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, nargs="?", default=Path("manip"))
    parser.add_argument("--subject", action="append", help="sujet a traiter (repetable)")
    parser.add_argument(
        "--trial", action="append", help="essai ou Sujet/essai a traiter (repetable)"
    )
    parser.add_argument(
        "--exclude-subject",
        action="append",
        help="sujet a ignorer; par defaut: pilot1 et pilot2",
    )
    parser.add_argument("--include-static", action="store_true")
    parser.add_argument("--force", action="store_true", help="reconstruit les CSV v2")
    parser.add_argument(
        "--refresh-video-map",
        action="store_true",
        help="regenere le mapping video sans forcement refaire les CSV",
    )
    parser.add_argument("--no-video-map", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="chemin du manifeste (sinon global pour une campagne complete, selection sinon)",
    )
    parser.add_argument(
        "--tare-root",
        type=Path,
        help="dossier contenant tare1/ et tare2/ (auto-detecte par defaut)",
    )
    parser.add_argument(
        "--tare-set",
        choices=("auto", "tare1", "tare2", "none"),
        default="auto",
    )
    parser.add_argument(
        "--tare-manifest",
        type=Path,
        default=DEFAULT_TARE_MANIFEST,
        help="manifeste JSON sujet -> dossier de tare",
    )
    parser.add_argument(
        "--walking-loss-policy",
        choices=("reject-any", "reject-during-contact", "allow"),
        default="reject-during-contact",
        help="politique des pertes pour walk/walk1/walk2/...",
    )
    args, passthrough = parser.parse_known_args()
    if args.workers < 1:
        parser.error("--workers doit etre >= 1")

    root = args.root.resolve()
    excluded = args.exclude_subject or list(DEFAULT_EXCLUDED_SUBJECTS)
    tare_assignments = load_tare_assignments(args.tare_manifest.resolve())
    campaign_root = args.tare_root.resolve() if args.tare_root else find_campaign_root(root)
    trials = discover_trials(
        root,
        subjects=args.subject,
        trials=args.trial,
        excluded_subjects=excluded,
        include_static=args.include_static,
    )
    if not trials:
        parser.error(f"aucun essai correspondant dans {root}")

    results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_trial,
                trial,
                root,
                campaign_root=campaign_root,
                tare_set=args.tare_set,
                tare_assignments=tare_assignments,
                walking_loss_policy=args.walking_loss_policy,
                force=args.force,
                refresh_video_map=args.refresh_video_map,
                map_videos=not args.no_video_map,
                passthrough=passthrough,
                dry_run=args.dry_run,
            )
            for trial in trials
        ]
        for future in as_completed(futures):
            key, item, message = future.result()
            results[key] = item
            print(message, flush=True)

    counts = Counter(str(item["status"]) for item in results.values())
    if not args.dry_run:
        filtered_run = bool(args.subject or args.trial) or (
            campaign_root is not None and root != campaign_root
        )
        manifest = args.manifest or (
            root
            / (
                "synchronized_v2_manifest_selection.json"
                if filtered_run
                else "synchronized_v2_manifest.json"
            )
        )
        manifest.write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Manifest: {manifest}")
    print("Resume: " + ", ".join(f"{name}={counts[name]}" for name in sorted(counts)))
    return 1 if counts.get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
