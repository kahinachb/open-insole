"""Body weight per Participant from the unloaded static trial (ticket #26).

Body weight (GLOSSARY) is the Participant's own weight, without the backpack. It is
read from the force plates during the unloaded Subject's static trial, from the CSV
written by ``extract_static_plate_fz.py`` (run in the ``c3d-io`` env, since the main
env has no ezc3d). Both Subjects of the Participant (``P01`` and ``P01b``) get it.

Quiet-standing window of a static trial (``quiet_standing_weight``), on the raw
1 kHz global plate Fz:

- both feet on the plates: total Fz >= ``MIN_TOTAL_N`` and each plate >= ``MIN_PLATE_SHARE``
  of the total, over the whole centred ``WINDOW_S`` window around the sample;
- still: the standard deviation of the total Fz over that window <= ``MAX_STD_N``;
- the trial is usable when at least ``MIN_QUIET_S`` of samples pass; the value is the
  median total Fz over them.

Fallbacks when a Participant's unloaded static is missing or unusable, in order:
(a) the loaded Subject's static minus the nominal backpack (``BACKPACK_KG``, by the
Participant's sex from ``participant_sex.json``); (b) the unloaded Subject's
balance-trial median (#16). The source is flagged per Participant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from grfm_schema import LOADED_SUFFIX, participant_of_subject

GRAVITY_M_S2 = 9.81
WINDOW_S = 0.5
MAX_STD_N = 5.0
MIN_PLATE_SHARE = 0.10
MIN_TOTAL_N = 100.0
MIN_QUIET_S = 1.0
BACKPACK_KG = {"female": 3.0, "male": 6.0}  # GLOSSARY "Load condition"
DEFAULT_SEX_CONFIG = Path(__file__).with_name("participant_sex.json")

SOURCE_UNLOADED_STATIC = "unloaded_static"
SOURCE_LOADED_STATIC = "loaded_static_minus_backpack"
SOURCE_UNLOADED_BALANCE = "unloaded_balance_trials"  # Participant fallback (b): its unloaded Subject's value
BALANCE_SOURCE = "balance_trials"  # the one source tag of the #16 per-Subject balance values

WINDOW_RULE = (
    f"raw global plate Fz (C3D, ezc3d force-platform filter, no low-pass); a sample is "
    f"quiet standing when, over the centred {WINDOW_S:g} s window around it, the total "
    f"Fz (plate 1 + plate 2) is always >= {MIN_TOTAL_N:g} N with each plate >= "
    f"{MIN_PLATE_SHARE:.0%} of it (both feet on their plates) and its standard "
    f"deviation is <= {MAX_STD_N:g} N; the trial is usable with >= {MIN_QUIET_S:g} s "
    "of quiet samples, and its standing weight is their median total Fz"
)


def participant_of(subject: str) -> str:
    """``P01b`` (loaded) and ``P01`` (unloaded) are the Participant ``P01``."""
    return participant_of_subject(subject)


def quiet_standing_weight(time_s, fz1, fz2) -> dict[str, object]:
    """Standing weight of one static trial; ``usable`` False with a ``reason`` otherwise."""
    time_s = np.asarray(time_s, dtype=float)
    fz1 = np.asarray(fz1, dtype=float)
    fz2 = np.asarray(fz2, dtype=float)
    total = fz1 + fz2
    if len(total) < 2:
        return {"usable": False, "reason": "no plate samples"}
    window = max(1, int(round(WINDOW_S / float(np.median(np.diff(time_s))))))
    with np.errstate(invalid="ignore"):
        on_plates = (
            np.isfinite(total)
            & (total >= MIN_TOTAL_N)
            & (fz1 >= MIN_PLATE_SHARE * total)
            & (fz2 >= MIN_PLATE_SHARE * total)
        )
    rolling = dict(window=window, center=True, min_periods=window)
    always_on = pd.Series(on_plates.astype(float)).rolling(**rolling).min().to_numpy() == 1.0
    spread = pd.Series(np.where(on_plates, total, np.nan)).rolling(**rolling).std().to_numpy()
    quiet = always_on & (spread <= MAX_STD_N)
    duration = float(quiet.sum() * np.median(np.diff(time_s)))
    summary: dict[str, object] = {
        "quiet_window": {
            "n_samples": int(quiet.sum()),
            "duration_s": duration,
            "trial_duration_s": float(time_s[-1] - time_s[0]),
            "start_s": float(time_s[quiet][0]) if quiet.any() else None,
            "end_s": float(time_s[quiet][-1]) if quiet.any() else None,
        }
    }
    if duration < MIN_QUIET_S:
        return {"usable": False, "reason": f"quiet standing {duration:.2f} s < {MIN_QUIET_S:g} s",
                **summary}
    weight = float(np.median(total[quiet]))
    return {
        "usable": True,
        "standing_weight_N": weight,
        "plate_medians_N": [float(np.median(fz1[quiet])), float(np.median(fz2[quiet]))],
        **summary,
    }


def static_standing_weights(path: Path | None) -> dict[str, dict[str, dict[str, object]]]:
    """Subject -> static batch -> ``quiet_standing_weight`` of each trial of the CSV."""
    if path is None:
        return {}
    table = pd.read_csv(path)
    found: dict[str, dict[str, dict[str, object]]] = {}
    for (subject, trial), rows in table.groupby(["subject", "trial"], sort=True):
        rows = rows.sort_values("time_s")
        result = quiet_standing_weight(
            rows["time_s"], rows["force_z_p1_global_N"], rows["force_z_p2_global_N"]
        )
        result["c3d_file"] = str(rows["c3d_file"].iloc[0])
        found.setdefault(str(subject), {})[f"{subject}_{trial}"] = result
    return found


def load_sex(path: Path | None) -> dict[str, str]:
    if path is None or not Path(path).is_file():
        return {}
    sex = json.loads(Path(path).read_text(encoding="utf-8"))
    unknown = {value for value in sex.values() if value not in BACKPACK_KG}
    if unknown:
        raise ValueError(f"{path}: sexe inconnu {sorted(unknown)} (attendu {sorted(BACKPACK_KG)})")
    return sex


def _best_static(trials: dict[str, dict[str, object]]) -> tuple[str | None, dict | None]:
    usable = {batch: r for batch, r in trials.items() if r["usable"]}
    if not usable:
        return None, None
    batch = max(sorted(usable), key=lambda b: usable[b]["quiet_window"]["duration_s"])
    return batch, usable[batch]


def body_weights_block(
    subjects: list[str],
    balance: dict[str, dict[str, object]],
    statics: dict[str, dict[str, dict[str, object]]],
    sex: dict[str, str],
    balance_rule: str,
    static_source: str | None,
) -> dict[str, object]:
    """The metadata ``body_weights`` block: per Participant, and shared by its Subjects.

    ``subjects`` are the dataset's Subjects, ``balance`` the #16 per-Subject balance
    values (``subject_body_weights``), ``statics`` the ``static_standing_weights``.
    """
    by_participant: dict[str, list[str]] = {}
    for subject in sorted(set(subjects) | set(balance)):
        by_participant.setdefault(participant_of(subject), []).append(subject)
    participants: dict[str, dict[str, object]] = {}
    for participant, members in sorted(by_participant.items()):
        unloaded, loaded = participant, f"{participant}{LOADED_SUFFIX}"
        rejected = {
            batch: result["reason"]
            for subject in (unloaded, loaded)
            for batch, result in sorted(statics.get(subject, {}).items())
            if not result["usable"]
        }
        entry: dict[str, object] = {"subjects": sorted(members)}
        batch, result = _best_static(statics.get(unloaded, {}))
        loaded_batch, loaded_result = _best_static(statics.get(loaded, {}))
        backpack = BACKPACK_KG.get(sex.get(participant, ""))
        if result is not None:
            weight = float(result["standing_weight_N"])
            entry.update(source=SOURCE_UNLOADED_STATIC, static_trial=batch,
                         quiet_window=result["quiet_window"],
                         plate_medians_N=result["plate_medians_N"])
        elif loaded_result is not None and backpack is not None:
            weight = float(loaded_result["standing_weight_N"]) - backpack * GRAVITY_M_S2
            entry.update(source=SOURCE_LOADED_STATIC, static_trial=loaded_batch,
                         quiet_window=loaded_result["quiet_window"],
                         plate_medians_N=loaded_result["plate_medians_N"],
                         loaded_standing_weight_N=float(loaded_result["standing_weight_N"]),
                         backpack_kg=backpack, sex=sex[participant])
        elif unloaded in balance:
            weight = float(balance[unloaded]["body_weight_N"])
            entry.update(source=SOURCE_UNLOADED_BALANCE, static_trial=None,
                         balance_trials=balance[unloaded].get("trials"))
        else:
            continue
        if loaded_result is not None and backpack is None and result is None:
            entry["loaded_static_skipped"] = f"sexe de {participant} inconnu (backpack nominal)"
        if rejected:
            entry["rejected_statics"] = rejected
        participants[participant] = {"body_weight_N": weight,
                                     "body_weight_kg": weight / GRAVITY_M_S2, **entry}
    subjects_out = {
        subject: {
            "body_weight_N": participants[participant_of(subject)]["body_weight_N"],
            "body_weight_kg": participants[participant_of(subject)]["body_weight_kg"],
            "source": participants[participant_of(subject)]["source"],
            "participant": participant_of(subject),
        }
        for subject in sorted(set(subjects) | set(balance))
        if participant_of(subject) in participants
    }
    return {
        "rule": (
            "Body weight per Participant (no backpack), shared by its unloaded and loaded "
            "Subjects: the standing weight of the unloaded Subject's static trial; "
            f"fallback (a) the loaded Subject's static minus the nominal backpack "
            f"({BACKPACK_KG['female']:g} kg women, {BACKPACK_KG['male']:g} kg men, "
            f"g = {GRAVITY_M_S2} m/s2); fallback (b) the unloaded Subject's balance-trial "
            "median (balance_trials)"
        ),
        "window_rule": WINDOW_RULE,
        "sources": {
            SOURCE_UNLOADED_STATIC: "static trial C3D of the unloaded Subject",
            SOURCE_LOADED_STATIC: "static trial C3D of the loaded Subject minus the backpack",
            SOURCE_UNLOADED_BALANCE: "balance trials of the unloaded Subject (#16 rule)",
        },
        "static_plate_fz_file": static_source,
        "static_trials": {
            batch: result
            for subject in sorted(statics)
            if participant_of(subject) in by_participant
            for batch, result in sorted(statics[subject].items())
        },
        "gravity_m_s2": GRAVITY_M_S2,
        "units": "N",
        "participants": participants,
        "subjects": subjects_out,
        "balance_rule": balance_rule,
        "balance_trials": balance,
    }


def existing_balance(body_weights: dict) -> tuple[dict[str, dict[str, object]], str]:
    """The #16 per-Subject balance values of an existing metadata ``body_weights``."""
    if "balance_trials" in body_weights:
        return body_weights["balance_trials"], body_weights.get("balance_rule", "")
    balance = {
        subject: entry for subject, entry in body_weights.get("subjects", {}).items()
        if entry.get("source") == BALANCE_SOURCE
    }
    return balance, body_weights.get("rule", "")


def refresh_body_weights(data_dir: Path, static_plate_fz: Path | None,
                         sex_config: Path | None) -> dict[str, object]:
    """Rewrite only ``body_weights`` in ``data_dir/metadata.json`` (no rebuild)."""
    from grfm_schema import subject_of_trial

    path = Path(data_dir) / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    balance, balance_rule = existing_balance(metadata.get("body_weights", {}))
    subjects = sorted({subject_of_trial(batch) for batch in metadata.get("batches", [])})
    metadata["body_weights"] = body_weights_block(
        subjects, balance, static_standing_weights(static_plate_fz), load_sex(sex_config),
        balance_rule, str(static_plate_fz) if static_plate_fz else None,
    )
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")
    return metadata["body_weights"]
