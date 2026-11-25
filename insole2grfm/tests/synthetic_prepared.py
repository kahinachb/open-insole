"""Synthetic mini prepared dataset in the v3.6 format, for CPU tests.

Builds a directory that the training engine accepts like the real prepared
dataset: one ``{batch}.npz`` per trial plus ``metadata.json`` with the real
78 feature names and the 12 target names. Each Participant ``Name`` has two
Subjects (``Name`` unloaded, ``Name2`` loaded), each with one trial per task
family. Foot states cover Measured everywhere, Airborne + Masked in ``jump``
and Masked in ``walk``. Optionally (``foot_height=True``) each trial also
carries the v3.7 Foot height keys (#14), valid on every foot state except a
short marker dropout, and (``walk_both_feet_masked=True``) ``walk`` holds a
stretch where both feet are Masked, i.e. samples with no GRFM target at all.
With ``one_foot_airborne=True`` the right foot lands early in ``jump``: the
first ``SINGLE_SUPPORT_SAMPLES`` of the flight have the left foot Airborne and
the right foot Measured (single support), and the left insole's gauges read
``SINGLE_SUPPORT_GAUGE_N`` there, so the inputs show which foot is unloaded.

Targets are a deterministic linear function of the inputs, so a model can
learn them; Fz is positive (about 350 N per foot) where the foot is Measured.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grfm_schema import STATE_AIRBORNE, STATE_MASKED, STATE_MEASURED
from grfm_schema import TARGETS as SCHEMA_TARGETS

REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE = REPO_ROOT / "train_lstm_two_feet_masked.py"
FEATURE_NAMES = json.loads(
    (REPO_ROOT / "tools" / "insole_layout.json").read_text(encoding="utf-8")
)["feature_names"]
# A list: pandas reads ``frame[TARGETS]`` as a column selection (a tuple would be one key).
TARGETS = list(SCHEMA_TARGETS)
PIPELINE_VERSION = "prepared_joint_masked_v3.6_fused_dynamic_airborne"
PARTICIPANTS = ("T01", "T03", "T04", "T05")
# (task_family, trial task name); the first four form the `continuous` task set.
FAMILIES = (
    ("balance", "balance1"),
    ("squat", "squat"),
    ("hulahoop", "hulahoop"),
    ("lateral_2supports", "lateral"),
    ("jump", "jump1"),
    ("walk", "walk"),
)
MEASURED_FZ_N = 350.0
# Foot height (mm) by foot state: flat on the plate, lifted, somewhere off the plates.
FOOT_HEIGHT_BASE_MM = {STATE_MEASURED: 0.0, STATE_AIRBORNE: 60.0, STATE_MASKED: 25.0}
# Samples of every trial whose Foot height is invalid (too few visible markers).
FOOT_HEIGHT_DROPOUT = slice(90, 95)
# ``one_foot_airborne``: jump samples (from the flight start) with only the left foot Airborne.
SINGLE_SUPPORT_SAMPLES = 25
SINGLE_SUPPORT_GAUGE_N = -5.0


@dataclass(frozen=True)
class SyntheticPrepared:
    data_dir: Path
    participants: tuple[str, ...]
    subjects: tuple[str, ...]
    batches: tuple[str, ...]


def foot_states(
    task_family: str, samples: int, walk_both_feet_masked: bool = False, one_foot_airborne: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    state_l = np.full(samples, STATE_MEASURED, dtype=np.int8)
    state_r = state_l.copy()
    third = samples // 3
    if task_family == "jump":
        state_l[third : 2 * third] = STATE_AIRBORNE
        state_r[third : 2 * third] = STATE_AIRBORNE
        state_l[2 * third : 2 * third + 10] = STATE_MASKED
        if one_foot_airborne:
            state_r[third : third + SINGLE_SUPPORT_SAMPLES] = STATE_MEASURED
    elif task_family == "walk":
        state_r[third : third + 30] = STATE_MASKED
        if walk_both_feet_masked:
            state_l[third : third + 30] = STATE_MASKED
    return state_l, state_r


def write_synthetic_prepared(
    data_dir: Path,
    participants: tuple[str, ...] = PARTICIPANTS,
    samples: int = 120,
    seed: int = 0,
    gauge_offset_n: float = 0.0,
    mirror_symmetric: bool = False,
    loaded_samples: int | None = None,
    body_weights_n: dict[str, float] | None = None,
    foot_height: bool = False,
    walk_both_feet_masked: bool = False,
    one_foot_airborne: bool = False,
) -> SyntheticPrepared:
    """Write a prepared-dataset directory and return what it contains.

    ``gauge_offset_n`` is added to every gauge channel after the targets are
    computed; with the default 0 the N(0, 1) gauges are all Inactive sensors
    (p99 < 5 N) for the sensor tokens. ``mirror_symmetric`` makes the right
    foot's targets the mirror image of the left foot's function applied to
    the feet-swapped inputs (like real physics), which a mirror-consistent
    model can fit. ``loaded_samples`` sets a different trial length for the
    loaded Subjects (``Name2``), so the two load conditions hold unequal
    numbers of windows. ``body_weights_n`` (Subject -> N) writes the v3.7
    metadata ``body_weights`` block (#16) for those Subjects; absent by default.
    ``foot_height`` writes ``foot_height_mm_L/R`` (``FOOT_HEIGHT_BASE_MM`` of
    the foot state plus a linear function of the inputs, NaN where invalid)
    and ``foot_height_valid_L/R`` (False on ``FOOT_HEIGHT_DROPOUT``).
    ``walk_both_feet_masked`` makes both feet Masked on the same ``walk``
    stretch. ``one_foot_airborne`` adds single support (left Airborne, right
    Measured, left gauges at ``SINGLE_SUPPORT_GAUGE_N``) at the start of the
    ``jump`` flight. Foot height draws from its own RNG, so ``foot_height`` leaves the
    other arrays unchanged.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    mixing = rng.normal(scale=10.0, size=(len(FEATURE_NAMES), len(TARGETS)))
    if mirror_symmetric:
        from sensor_tokens import TARGET_MIRROR_SIGN, SensorTokenizer

        swap = SensorTokenizer.from_layout().swap_feet(np.eye(len(FEATURE_NAMES)))
        mixing[:, 6:] = (swap @ mixing[:, :6]) * TARGET_MIRROR_SIGN[6:]
    foot_height_mixing = np.random.default_rng(seed + 1).normal(scale=3.0, size=(len(FEATURE_NAMES), 2))
    fz_columns = [TARGETS.index("Fz_L"), TARGETS.index("Fz_R")]
    gauge_columns = [index for index, name in enumerate(FEATURE_NAMES) if "_force_" in name]
    left_gauge_columns = [index for index in gauge_columns if FEATURE_NAMES[index].startswith("L_")]
    batches, reports, subjects = [], [], []
    for participant in participants:
        for subject, length in (
            (participant, samples),
            (f"{participant}b", samples if loaded_samples is None else loaded_samples),
        ):
            subjects.append(subject)
            for task_family, task in FAMILIES:
                batch = f"{subject}_{task}"
                x = rng.normal(size=(length, len(FEATURE_NAMES))).astype(np.float32)
                y = x @ mixing
                x[:, gauge_columns] += gauge_offset_n
                y[:, fz_columns] += MEASURED_FZ_N
                state_l, state_r = foot_states(
                    task_family, length, walk_both_feet_masked, one_foot_airborne
                )
                if one_foot_airborne:
                    single = (state_l == STATE_AIRBORNE) & (state_r == STATE_MEASURED)
                    x[np.ix_(single, left_gauge_columns)] = SINGLE_SUPPORT_GAUGE_N
                foot_state = np.repeat(np.stack([state_l, state_r], axis=1), 6, axis=1)
                y[foot_state != STATE_MEASURED] = 0.0
                extra = {}
                if foot_height:
                    height = x @ foot_height_mixing
                    for side, column, state in (("L", 0, state_l), ("R", 1, state_r)):
                        valid = np.ones(length, dtype=bool)
                        valid[FOOT_HEIGHT_DROPOUT] = False
                        values = height[:, column] + np.vectorize(FOOT_HEIGHT_BASE_MM.get)(state)
                        extra[f"foot_height_mm_{side}"] = np.where(valid, values, np.nan).astype(np.float32)
                        extra[f"foot_height_valid_{side}"] = valid
                np.savez(
                    data_dir / f"{batch}.npz",
                    **extra,
                    X=x,
                    y=y.astype(np.float32),
                    target_mask=foot_state != STATE_MASKED,
                    target_time_s=np.arange(length, dtype=np.float64) * 0.005,
                    segment_id=np.zeros(length, dtype=np.int64),
                    state_L=state_l,
                    state_R=state_r,
                    airborne_label_L=state_l == STATE_AIRBORNE,
                    airborne_label_R=state_r == STATE_AIRBORNE,
                )
                batches.append(batch)
                reports.append(
                    {"batch": batch, "evaluation_group": batch, "task_family": task_family}
                )
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "feature_names": FEATURE_NAMES,
        "feature_count": len(FEATURE_NAMES),
        "targets": TARGETS,
        "target_mask_codes": {"0": "unknown/masked", "1": "platform_measured", "2": "airborne_zero"},
        "batches": batches,
        "reports": reports,
    }
    if body_weights_n is not None:
        metadata["body_weights"] = {
            "units": "N",
            "subjects": {
                subject: {"body_weight_N": float(value), "source": "balance_trials"}
                for subject, value in body_weights_n.items()
            },
        }
    (data_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return SyntheticPrepared(data_dir, tuple(participants), tuple(subjects), tuple(batches))


# One fold of the engine on this dataset: test T01, validation T03, train T04 and T05; 2 epochs.
ONE_FOLD = ("--causal", "--split-level", "subject_identity", "--only-test-group", "T01", "--epochs", "2")
# One-fold plain MLP run shared by several test modules (through the ``trained_once`` fixture).
PLAIN_MLP_ONE_FOLD = ("--model-type", "mlp", "--layers", "2", *ONE_FOLD)


def checkpoint(out: Path, model_type: str = "mlp", fold: str = "T01") -> dict:
    """The engine's saved checkpoint of ``fold`` in run dir ``out`` (``models/<type>_test_<fold>.pt``)."""
    import torch

    return torch.load(out / "models" / f"{model_type}_test_{fold}.pt", weights_only=False)


# Small, CPU-only defaults; callers append/override flags (argparse keeps the last).
FAST_TRAINING_ARGS = (
    "--device", "cpu",
    "--epochs", "1",
    "--batch-size", "64",
    "--hidden-size", "8",
    "--num-workers", "0",
    "--no-plots",
)


def run_training_cli(
    data_dir: Path, output_dir: Path, *args: str, check: bool = True
) -> subprocess.CompletedProcess:
    """Run the training engine CLI with fast CPU defaults.

    In-process through the engine's ``main(argv)`` (same argument parsing,
    same outputs, exit code = its return value or ``SystemExit`` code, 1 on an
    uncaught exception), so the tests do not pay a Python + torch start-up per
    call. The engine seeds every RNG from ``--seed`` at the start of ``main``.
    """
    argv = [str(data_dir), "--output-dir", str(output_dir), *FAST_TRAINING_ARGS, *args]
    engine = importlib.import_module(ENGINE.stem)
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            returncode = engine.main(argv)
        except SystemExit as exit_:
            code = exit_.code
            returncode = code if isinstance(code, int) else (0 if code is None else 1)
        except Exception:  # noqa: BLE001 - reported like a crashed process
            traceback.print_exc()
            returncode = 1
    result = subprocess.CompletedProcess(
        [sys.executable, str(ENGINE), *argv], returncode or 0, stdout.getvalue(), stderr.getvalue()
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"training CLI failed ({result.returncode}):\n{result.stdout}\n{result.stderr}"
        )
    return result
