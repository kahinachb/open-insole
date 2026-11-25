#!/usr/bin/env python3
"""Resumable overnight job queue (spec #1): every model under one protocol.

Jobs run in priority order (MLP, temporal Transformer, sensor-token model on
the ``all`` task set, then the same on ``continuous``). The plan's ``split``
sets the folds of every job. The Fixed split (the default plan, #23:
``{"test": [...], "validation": [...]}``) is one fold, ``fold_<T1+T2+...>``:
one model tested on the named test Participants and early-stopped on the
validation ones. Full CV (``"split": "full_cv"``, a plan without ``split``, or
``--full-cv``) is one engine run
(``train_lstm_two_feet_masked.py --only-test-group P``) per test Participant,
each in its own fold directory. The split is part of the config signature, so
the folds of one split are never taken as done for the other. Before its first job, each model goes through
the entry gate: the engine's overfit test (missing verdict = failure). Baselines
(plan key ``baseline``: the MLP and the temporal Transformer) stop there; the
other models (the sensor-token model) then face the validation guard against
the MLP and, on failure, the bounded auto-correction (:mod:`auto_correction`)
whose adopted flags are appended to every command.

Resumability: a fold whose ``metadata.json`` and checkpoint exist and whose
config signature (``queue_fold.json``; folds without one = plan config without
adopted overrides) equals the current one is never retrained. A finished fold
of another config (after an adopted override) is moved to ``stale/`` and
retrained; any other fold directory is an interrupted run and is wiped and
restarted. A job whose model type the engine does not know yet is skipped with
a reason; a job whose model fails the gate is failed with the reason; the queue
always moves on to the next job. A lock file prevents two runners at once.

Results tree (``--results-root``)::

    queue_status.json              every job: pending/running/done/skipped/failed + reason,
                                   gates, adopted overrides, correction budget used per model
    queue.lock                     flock held by the running queue (holds its PID)
    auto_correction.jsonl          every gate check, skipped correction, adoption,
                                   exclusion, invalidated job and stale fold (one JSON per line)
    <model>/overfit/               entry gate: overfit_verdict.json, overfit_curves.csv,
                                   gradnorm_weights.csv, overfit_windows.csv,
                                   split_manifest.json, engine.log, gate_result.json
    <model>/overfit_attempt_<N>/   same, for auto-correction attempt N
    <model>/guard[_attempt_<N>]/   validation guard run (--selection-only): training_history.json,
                                   engine.log, guard_run.json
    mlp/guard_reference/           short MLP run, only when its first fold is not finished
    <model>/<task_set>/            split_manifest.json, queue_config.json
    <model>/<task_set>/fold_<P>/   (Fixed split: fold_<T1+T2+...>) predictions.csv, fold_metrics.csv, training_history.json,
                                   gradnorm_weights.csv, metadata.json, split_manifest.json,
                                   models/<model_type>_test_<P>.pt, plots/, engine.log,
                                   queue_fold.json (config signature it was trained with)
    stale/<model>/<task_set>/<signature>/fold_<P>/   folds superseded by an adopted override

Usage::

    python overnight_queue.py ../dataset --results-root ../results/overnight_v3.6 \
        --device cuda [--jobs mlp_continuous] [--full-cv [--folds P01 P07]]

Named plans (``--named-plan``): ``default`` (spec #1) and ``v3.7`` (spec #12, the
v3.7 queue as launched; ``--named-plan v3.7 --print-plan`` writes its plan file)::

    python overnight_queue.py ../dataset_v3.7 --named-plan v3.7 \
        --results-root ../results/fixed_split_v3.7 --device cuda
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

from auto_correction import auto_correct_gate
from queue_common import (
    AUTO_CORRECTION_LOG,
    GATE_TASK_SET,
    last_value,
    now,
    read_json,
    write_json,
)

ENGINE = Path(__file__).resolve().with_name("train_lstm_two_feet_masked.py")
DEFAULT_RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results" / "overnight_v3.6"
STATUS_FILE = "queue_status.json"
LOCK_FILE = "queue.lock"
STATUS_SCHEMA_VERSION = 2
EXIT_LOCKED = 4
FOLD_CONFIG_FILE = "queue_fold.json"
FOLD_TIMINGS = ("started_at", "finished_at", "elapsed_s")
STALE_DIR = "stale"
ENGINE_DEFAULT_LEARNING_RATE = "1e-3"
MAX_PASSES = 6
CHILD_STOP_TIMEOUT_S = 30.0  # on interrupt: SIGTERM the engine, SIGKILL it if still alive after this
# --engine-arg flags left out of the config signature: they do not change what a fold learns.
OPERATIONAL_ENGINE_ARGS = {"--no-plots": 0, "--num-workers": 1}

# Shared protocol of every job = the existing comparison runs (spec user stories
# 3-4) plus GradNorm and full CV. Model-specific flags come after and win.
COMMON_ARGS = [
    "--causal", "--window-samples", "21",
    "--split-level", "subject_identity", "--validation-group-count", "2",
    "--input-features", "raw", "--target-scaling", "standard",
    "--learning-rate", "3e-4", "--lr-scheduler", "plateau", "--lr-factor", "0.5",
    "--lr-patience", "4", "--min-learning-rate", "1e-5", "--gradient-clip", "1.0",
    "--epochs", "120", "--patience", "12", "--batch-size", "256",
    "--zero-target-weight", "0.25", "--num-workers", "4", "--seed", "42",
    "--gradnorm",
]

TRANSFORMER_ARGS = [
    "--hidden-size", "64", "--layers", "2", "--dropout", "0.1",
    "--transformer-heads", "4", "--transformer-feedforward-size", "128",
]

# #23 (owner, 2026-10-24): one model per experiment instead of full CV. A Participant's
# two Subjects always stay together (the engine splits by Participant).
# #25 (owner, 2026-10-25), replacing #23's P01 + P07 + P13 / P02 + P14:
# test P13 + P04 (typical, in distribution), validation P07 (both Subjects; the heaviest
# Participant, out of distribution), training the other 12 Participants.
FIXED_SPLIT = {"test": ["P04", "P13"], "validation": ["P07"]}

# Data-driven plan; ``--plan FILE`` replaces it (same JSON structure). ``"baseline": true``
# marks a model whose entry gate is the overfit test only: no validation guard, no
# auto-correction (see ``Queue.is_baseline`` for the default when the key is absent).
DEFAULT_PLAN = {
    "common_args": COMMON_ARGS,
    "split": FIXED_SPLIT,
    "models": {
        "mlp": {
            "model_type": "mlp",
            "engine_args": ["--hidden-size", "256", "--layers", "2", "--dropout", "0.2"],
            "baseline": True,
        },
        "transformer": {"model_type": "transformer", "engine_args": TRANSFORMER_ARGS, "baseline": True},
        # #8: the temporal Transformer settings plus 10 % token dropout (needs --causal, raw inputs).
        "sensor_token": {
            "model_type": "sensor_token",
            "engine_args": [*TRANSFORMER_ARGS, "--token-dropout", "0.1"],
        },
    },
    "jobs": [
        {"name": "mlp_all", "model": "mlp", "task_set": "all"},
        {"name": "transformer_all", "model": "transformer", "task_set": "all"},
        {"name": "sensor_token_all", "model": "sensor_token", "task_set": "all"},
        {"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"},
        {"name": "transformer_continuous", "model": "transformer", "task_set": "continuous"},
        {"name": "sensor_token_continuous", "model": "sensor_token", "task_set": "continuous"},
    ],
}

# Spec #12 / #22: the v3.7 queue as launched on 2026-10-25 (owner decisions: the sensor-token
# model is left out; the temporal Transformer on v3.7 is not planned yet). MLP only, on the
# Fixed split, Measured-only selection: continuous without / with the body-weight input, then
# all without / with the Foot height output. ``--named-plan v3.7 --print-plan`` writes it.
MLP_ARGS = ["--hidden-size", "256", "--layers", "2", "--dropout", "0.2"]
BODY_WEIGHT_MASKING_ARGS = ["--body-weight-masking", "0.3"]
V37_PLAN = {
    "common_args": [*COMMON_ARGS, "--selection-subset", "measured"],
    "split": FIXED_SPLIT,
    "models": {
        "mlp": {"model_type": "mlp", "engine_args": MLP_ARGS, "baseline": True},
        "mlp_bw": {"model_type": "mlp", "engine_args": [*MLP_ARGS, "--body-weight-input"], "baseline": True},
        "mlp_fh": {"model_type": "mlp", "engine_args": [*MLP_ARGS, "--foot-height-output"], "baseline": True},
        # #27: MLP + body weight + total/share parametrisation of the vertical force.
        "mlp_share": {
            "model_type": "mlp",
            "engine_args": [*MLP_ARGS, "--body-weight-input", "--vertical-share"],
            "baseline": True,
        },
        # #28: optional body weight (random masking, p = 0.3, with the "body weight missing" flag),
        # without and with total/share.
        "mlp_bw_mask": {
            "model_type": "mlp",
            "engine_args": [*MLP_ARGS, "--body-weight-input", *BODY_WEIGHT_MASKING_ARGS],
            "baseline": True,
        },
        "mlp_share_mask": {
            "model_type": "mlp",
            "engine_args": [*MLP_ARGS, "--body-weight-input", "--vertical-share", *BODY_WEIGHT_MASKING_ARGS],
            "baseline": True,
        },
        # #29: total/share with dedicated heads and the decomposed total + share loss; and the
        # diagnostic: the sum loss on a plain body-weight MLP (loss effect without the anchoring).
        "mlp_share_decomposed": {
            "model_type": "mlp",
            "engine_args": [*MLP_ARGS, "--body-weight-input", "--vertical-share", "--vertical-loss", "decomposed"],
            "baseline": True,
        },
        "mlp_bw_sumloss": {
            "model_type": "mlp",
            "engine_args": [*MLP_ARGS, "--body-weight-input", "--vertical-loss", "sum"],
            "baseline": True,
        },
    },
    "jobs": [
        {"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"},
        {"name": "mlp_bw_continuous", "model": "mlp_bw", "task_set": "continuous"},
        {"name": "mlp_all", "model": "mlp", "task_set": "all"},
        {"name": "mlp_fh_all", "model": "mlp_fh", "task_set": "all"},
        {"name": "mlp_share_continuous", "model": "mlp_share", "task_set": "continuous"},
        {"name": "mlp_bw_mask_continuous", "model": "mlp_bw_mask", "task_set": "continuous"},
        {"name": "mlp_share_mask_continuous", "model": "mlp_share_mask", "task_set": "continuous"},
        {"name": "mlp_share_decomposed_continuous", "model": "mlp_share_decomposed", "task_set": "continuous"},
        {"name": "mlp_bw_sumloss_continuous", "model": "mlp_bw_sumloss", "task_set": "continuous"},
        {"name": "mlp_share_decomposed_all", "model": "mlp_share_decomposed", "task_set": "all"},
        {"name": "mlp_bw_sumloss_all", "model": "mlp_bw_sumloss", "task_set": "all"},
    ],
}
NAMED_PLANS = {"default": DEFAULT_PLAN, "v3.7": V37_PLAN}
# Plan models without a ``baseline`` key (plan files written from the default plan
# before the key existed) are baselines when they are one in the default plan.
DEFAULT_BASELINES = {name for name, spec in DEFAULT_PLAN["models"].items() if spec.get("baseline")}
# Checks of an entry gate, recorded in each gate record (``checks``).
BASELINE_CHECKS = ["overfit"]
CANDIDATE_CHECKS = ["overfit", "guard"]


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ENGINE.parent), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return (result.stdout.strip() or None) if result.returncode == 0 else None


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def empty_overrides() -> dict:
    return {
        "all_models": [],
        "per_model": {},
        "learning_rate_factor": 1.0,
        "corrections": {"all_models": [], "per_model": {}},
    }


def training_engine_args(engine_args: list[str]) -> list[str]:
    """``--engine-arg`` tokens without the operational flags (and their values)."""
    kept, skip = [], 0
    for token in engine_args:
        if skip:
            skip -= 1
            continue
        flag = token.split("=", 1)[0]
        if flag in OPERATIONAL_ENGINE_ARGS:
            skip = 0 if "=" in token else OPERATIONAL_ENGINE_ARGS[flag]
            continue
        kept.append(token)
    return kept


def signature_of(config_args: list[str]) -> str:
    return hashlib.sha256(json.dumps(config_args).encode("utf-8")).hexdigest()[:12]


# Engine flags whose model reads the dataset's per-Subject body weight (#19, #27).
BODY_WEIGHT_FLAGS = ("--body-weight-input", "--vertical-share")


def body_weights_digest(metadata: dict) -> str | None:
    """Digest of the per-Subject body weights the engine reads (``body_weights.subjects[S].body_weight_N``).

    ``--refresh-body-weights`` (#26) rewrites them without changing the
    ``pipeline_version``, so the signature of a body-weight job needs them too.
    None when the dataset has no body weights.
    """
    subjects = ((metadata or {}).get("body_weights") or {}).get("subjects")
    if not subjects:
        return None
    values = {subject: (entry or {}).get("body_weight_N") for subject, entry in sorted(subjects.items())}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def uses_body_weight(args: list[str]) -> bool:
    return any(token.split("=", 1)[0] in BODY_WEIGHT_FLAGS for token in args)


# #28: the engine's default body weight (masked windows, "missing" mode) is the fold's training
# mean; its rule goes in the signature of the masked jobs (the probability is in their args).
BODY_WEIGHT_MASKING_FLAG = "--body-weight-masking"
BODY_WEIGHT_DEFAULT_RULE = "body_weight_default=train_mean"


def uses_body_weight_masking(args: list[str]) -> bool:
    return any(token.split("=", 1)[0] == BODY_WEIGHT_MASKING_FLAG for token in args)


def fold_is_done(fold_dir: Path, model_type: str, fold: str) -> bool:
    # metadata.json is the engine's last write, after the checkpoint.
    return (fold_dir / "metadata.json").is_file() and (
        fold_dir / "models" / f"{model_type}_test_{fold}.pt"
    ).is_file()


def overfit_gate(queue: "Queue", model: str) -> dict:
    """Entry gate of a baseline (and default gate): one overfit test with the model's configuration.

    Same record shape as ``auto_correction.auto_correct_gate`` (``overfit``,
    ``guard``); no validation guard, no correction. The gate of the other
    models is injected (``Queue(entry_gate=...)``, #9).
    """
    overfit = queue.run_overfit(model)
    return {
        "model": model,
        "passed": overfit["passed"],
        "cause": overfit["cause"],
        "reason": overfit["reason"],
        "overfit": overfit,
        "guard": None,
        "elapsed_s": overfit["elapsed_s"],
        "finished_at": now(),
    }


class Queue:
    def __init__(
        self,
        args: argparse.Namespace,
        plan: dict,
        entry_gate: Callable[["Queue", str], dict] = overfit_gate,
    ) -> None:
        self.args = args
        self.plan = plan
        dataset_metadata = read_json(args.data_dir / "metadata.json") or {}
        self.dataset_version = dataset_metadata.get("pipeline_version")
        self.body_weights_digest = body_weights_digest(dataset_metadata)
        self.root: Path = args.results_root
        self.status_path = self.root / STATUS_FILE
        self.entry_gate = entry_gate
        self.child: subprocess.Popen | None = None
        previous = read_json(self.status_path) or {}
        # Adopted by auto-correction (#9), appended to every command:
        # ``all_models`` = loss/optimiser flags, ``learning_rate_factor`` = global
        # multiplier of each model's own learning rate (lr / 3), ``per_model[model]``
        # = architecture flags; ``corrections`` = names of the adopted corrections.
        # A schema-1 status (no factor / corrections) reads as "nothing adopted".
        self.adopted_overrides = empty_overrides()
        for key, value in previous.get("adopted_overrides", {}).items():
            self.adopted_overrides[key] = value
        self.correction_log = self.root / AUTO_CORRECTION_LOG
        # Auto-correction attempts and seconds already spent per model on this
        # results root ({model: {attempts, elapsed_s}}): the per-model budget holds
        # across queue passes, gate re-runs and restarts. Absent in older statuses.
        self.correction_budget: dict[str, dict] = previous.get("correction_budget", {})
        self.timed_out = False
        previous_jobs = {job["name"]: job for job in previous.get("jobs", [])}
        # Gate records survive a restart; one is reused only while the model's
        # config signature is unchanged (see ``gate``).
        self.gates: dict[str, dict] = {
            model: record for model, record in previous.get("gates", {}).items()
            if model in plan["models"] and record.get("passed") is not None
        }
        self.jobs: list[dict] = []
        for job in plan["jobs"]:
            model = plan["models"][job["model"]]
            record = {
                "name": job["name"],
                "model": job["model"],
                "model_type": model["model_type"],
                "task_set": job["task_set"],
                "status": "pending",
                "reason": None,
                "output_dir": str(self.root / job["model"] / job["task_set"]),
                "config_signature": None,
                "folds": {},
                "folds_total": None,
                "folds_done": 0,
                "started_at": None,
                "finished_at": None,
            }
            kept = previous_jobs.get(job["name"])
            if args.jobs and job["name"] not in args.jobs:
                record = kept if kept else {**record, "reason": "not selected in this run"}
            elif kept and (kept.get("model"), kept.get("task_set")) == (job["model"], job["task_set"]):
                # Relaunch: the done folds keep their records (timings), the job its start
                # and end; ``run_job`` re-derives every fold's status from the disk.
                done = {fold: item for fold, item in (kept.get("folds") or {}).items() if item.get("status") == "done"}
                if done:
                    record.update(
                        folds=done, folds_total=kept.get("folds_total"), folds_done=len(done),
                        config_signature=kept.get("config_signature"),
                        started_at=kept.get("started_at"), finished_at=kept.get("finished_at"),
                    )
            self.jobs.append(record)
        self.state = {
            "schema_version": STATUS_SCHEMA_VERSION,
            "results_root": str(self.root),
            "data_dir": str(args.data_dir.resolve()),
            "dataset_version": None,
            "engine": str(ENGINE),
            "engine_commit": git_commit(),
            "device": args.device,
            "selected_jobs": args.jobs or [job["name"] for job in plan["jobs"]],
            "selected_folds": args.folds,
            "runner": {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "state": "running",
                "started_at": now(),
                "finished_at": None,
            },
            "updated_at": now(),
            "adopted_overrides": self.adopted_overrides,
            "auto_correction_log": str(self.correction_log),
            "gates": self.gates,
            "correction_budget": self.correction_budget,
            "jobs": self.jobs,
        }

    # -- bookkeeping -------------------------------------------------------------
    def save(self) -> None:
        self.state["updated_at"] = now()
        write_json(self.status_path, self.state)

    def log(self, message: str) -> None:
        print(f"{now()} {message}", flush=True)

    def log_event(self, event: str, **fields) -> None:
        """One line of the auto-correction log (JSON Lines, append-only)."""
        self.root.mkdir(parents=True, exist_ok=True)
        append_jsonl(self.correction_log, {"event": event, "at": now(), **fields})

    # -- commands ----------------------------------------------------------------
    def engine_command(self, model: str, task_set: str, output_dir: Path, *extra: str) -> list[str]:
        spec = self.plan["models"][model]
        return [
            self.args.python, "-u", str(ENGINE), str(self.args.data_dir),
            "--output-dir", str(output_dir),
            "--task-set", task_set,
            "--model-type", spec["model_type"],
            *self.plan.get("common_args", COMMON_ARGS),
            *spec.get("engine_args", []),
            *self.split_args(),
            "--device", self.args.device,
            *self.override_args(model),
            *extra,
            *self.args.engine_arg,
        ]

    # -- split (#23) ----------------------------------------------------------------
    def split(self) -> dict | None:
        """The plan's Fixed split ({"test": [...], "validation": [...]}), None for full CV."""
        split = self.plan.get("split")
        return None if split in (None, "full_cv") else split

    def split_args(self) -> list[str]:
        split = self.split()
        if split is None:
            return []
        return ["--test-groups", *split["test"], "--validation-groups", *split["validation"]]

    def split_matches(self, manifest: dict | None) -> bool:
        """Whether an engine split manifest was written for the plan's split."""
        if not manifest:
            return False
        split = self.split()
        if split is None:
            return manifest.get("split", "full_cv") == "full_cv"
        folds = manifest.get("folds") or []
        return manifest.get("split") == "fixed" and len(folds) == 1 and (
            folds[0].get("test_groups") == sorted(split["test"])
            and folds[0].get("validation_groups") == sorted(split["validation"])
        )

    def guard_task_set(self, model: str) -> str:
        """Task set of the validation guard of ``model``: that of its first job in the plan.

        Candidate and reference (MLP) are compared on this task set and on the
        first fold of the split; the MLP job's finished fold is reused when current.
        """
        return next((job["task_set"] for job in self.plan["jobs"] if job["model"] == model), GATE_TASK_SET)

    # -- effective configuration (#9) ----------------------------------------------
    def base_args(self, model: str) -> list[str]:
        return [*self.plan.get("common_args", COMMON_ARGS), *self.plan["models"][model].get("engine_args", [])]

    def learning_rate(self, model: str) -> float:
        """Learning rate ``model`` trains with (its own flag x the adopted factor)."""
        command = self.base_args(model) + self.override_args(model)
        return float(last_value(command, "--learning-rate") or ENGINE_DEFAULT_LEARNING_RATE)

    def override_args(self, model: str) -> list[str]:
        """Adopted auto-correction flags of ``model`` (empty when nothing adopted)."""
        overrides = self.adopted_overrides
        factor = float(overrides.get("learning_rate_factor", 1.0))
        rate = []
        if factor != 1.0:
            base = float(last_value(self.base_args(model), "--learning-rate") or ENGINE_DEFAULT_LEARNING_RATE)
            rate = ["--learning-rate", format(base * factor, ".6g")]
        return [*overrides.get("all_models", []), *rate, *overrides.get("per_model", {}).get(model, [])]

    def config_args(self, model: str, task_set: str, adopted: bool = True) -> list[str]:
        """Effective training configuration of ``model`` on ``task_set``.

        Everything that defines what a fold learns: dataset version (its
        ``pipeline_version``, #23: folds of another dataset are never reused),
        task set, model type, plan args, split, adopted overrides and
        ``--engine-arg`` (minus the flags that do not change training,
        ``OPERATIONAL_ENGINE_ARGS``); for a model that reads the body weight
        (``BODY_WEIGHT_FLAGS``), the digest of the dataset's body weights too,
        so a body-weight refresh retrains it, and for a masked job
        (``--body-weight-masking``) the default-value rule; not the interpreter, engine path,
        data/output paths or device.
        """
        args = [
            "--task-set", task_set,
            "--model-type", self.plan["models"][model]["model_type"],
            *self.base_args(model),
            *self.split_args(),
            *(self.override_args(model) if adopted else []),
            *training_engine_args(self.args.engine_arg),
        ]
        weights = [f"body_weights={self.body_weights_digest}"] if uses_body_weight(args) else []
        if uses_body_weight_masking(args):
            weights.append(BODY_WEIGHT_DEFAULT_RULE)
        return [f"dataset_version={self.dataset_version}", *weights, *args]

    def config_signature(self, model: str, task_set: str, adopted: bool = True) -> str:
        return signature_of(self.config_args(model, task_set, adopted))

    def fold_signature(self, fold_dir: Path, model: str, task_set: str) -> str:
        """Signature of the configuration a finished fold was trained with.

        Folds trained before #9 have no ``queue_fold.json``: nothing could be
        adopted then, so they count as trained with the plan's configuration
        without adopted overrides.
        """
        recorded = read_json(fold_dir / FOLD_CONFIG_FILE)
        if recorded and recorded.get("config_signature"):
            return recorded["config_signature"]
        return self.config_signature(model, task_set, adopted=False)

    def fold_is_current(self, model: str, task_set: str, fold: str) -> bool:
        """Fold finished and trained with the current configuration."""
        fold_dir = self.root / model / task_set / f"fold_{fold}"
        model_type = self.plan["models"][model]["model_type"]
        return fold_is_done(fold_dir, model_type, fold) and (
            self.fold_signature(fold_dir, model, task_set) == self.config_signature(model, task_set)
        )

    def stale_folds(self, job: dict) -> dict[str, str]:
        """Finished folds of ``job`` trained with another configuration: {fold: signature}."""
        current = self.config_signature(job["model"], job["task_set"])
        stale = {}
        for fold_dir in sorted(Path(job["output_dir"]).glob("fold_*")):
            fold = fold_dir.name.removeprefix("fold_")
            if not self.split_matches(read_json(fold_dir / "split_manifest.json")):
                continue  # a fold of the other split (Fixed / full CV) is never retrained for this one
            if fold_is_done(fold_dir, job["model_type"], fold):
                signature = self.fold_signature(fold_dir, job["model"], job["task_set"])
                if signature != current:
                    stale[fold] = signature
        return stale

    def invalidate_stale_jobs(self, by_model: str, corrections: list[str]) -> None:
        """After an adoption: every job with finished folds of another config is redone.

        Fairness (spec story 22): a global loss/optimiser change makes the
        finished folds of the other models stale; they are retrained (old
        outputs moved to ``<root>/stale/``) the next time their job runs.
        """
        for job in self.jobs:
            stale = self.stale_folds(job)
            if not stale:
                continue
            signature = self.config_signature(job["model"], job["task_set"])
            reason = (
                f"config changed: {', '.join(corrections)} adopted by {by_model}; "
                f"{len(stale)} finished fold(s) trained with {', '.join(sorted(set(stale.values())))} "
                f"to retrain with {signature}"
            )
            self.log(f"{job['name']}: {reason}")
            self.log_event(
                "invalidated", job=job["name"], model=job["model"], by_model=by_model,
                corrections=corrections, folds=sorted(stale), trained_signatures=stale,
                new_signature=signature,
            )
            if job["status"] != "running":
                job.update(status="pending", reason=reason)
            for fold in stale:
                if fold in job["folds"]:
                    job["folds"][fold].update(status="pending", reason=reason)
            job["folds_done"] = sum(item["status"] == "done" for item in job["folds"].values())
        self.save()

    def adopt(self, model: str, corrections: list[dict]) -> None:
        """Record adopted corrections ``[{name, scope, flags}]`` (scope all_models | per_model)."""
        overrides = self.adopted_overrides
        names = overrides["corrections"]
        for correction in corrections:
            if correction["scope"] == "all_models":
                if correction["name"] == "lr_div3":
                    overrides["learning_rate_factor"] = float(overrides["learning_rate_factor"]) / 3.0
                else:
                    overrides["all_models"].extend(correction["flags"])
                names["all_models"].append(correction["name"])
            else:
                overrides["per_model"].setdefault(model, []).extend(correction["flags"])
                names["per_model"].setdefault(model, []).append(correction["name"])
        self.log_event(
            "adopted", model=model, corrections=corrections,
            adopted_overrides=copy.deepcopy(overrides),
            config_signature=self.config_signature(model, GATE_TASK_SET),
        )
        self.save()
        self.invalidate_stale_jobs(model, [correction["name"] for correction in corrections])

    def correction_usage(self, model: str) -> dict:
        """Auto-correction attempts and seconds ``model`` already used: ``{attempts, elapsed_s}``."""
        return self.correction_budget.setdefault(model, {"attempts": 0, "elapsed_s": 0.0})

    def spend_correction(self, model: str, elapsed_s: float) -> None:
        """Count one correction attempt of ``model`` (persisted at once)."""
        usage = self.correction_usage(model)
        usage["attempts"] += 1
        usage["elapsed_s"] = round(usage["elapsed_s"] + elapsed_s, 1)
        self.save()

    def run_logged(self, command: list[str], log_path: Path, timeout_s: float | None = None) -> int:
        """Run an engine command, its output timestamped into ``log_path``.

        With ``timeout_s`` the child is terminated when the time is up
        (``self.timed_out`` then reads True).
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
        self.timed_out = False
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"{now()} $ {shlex.join(command)}\n")
            log.flush()
            self.child = subprocess.Popen(
                command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=environment, cwd=ENGINE.parent,
            )
            timer = None
            if timeout_s is not None:
                child = self.child

                def expire() -> None:
                    self.timed_out = True
                    child.terminate()

                timer = threading.Timer(max(timeout_s, 0.0), expire)
                # Daemon and cancelled on any exit (SIGTERM raises SystemExit here): a
                # pending timer must never keep the interpreter alive after an interrupt.
                timer.daemon = True
                timer.start()
            try:
                for line in self.child.stdout:
                    log.write(f"{now()} {line}")
                    log.flush()
                returncode = self.child.wait()
            finally:
                if timer is not None:
                    timer.cancel()
            suffix = " (time budget exhausted)" if self.timed_out else ""
            log.write(f"{now()} exit code {returncode}{suffix}\n")
        self.child = None
        return returncode

    def model_type_available(self, model_type: str) -> bool:
        # argparse rejects an unknown --model-type choice (exit 2) before --help.
        result = subprocess.run(
            [self.args.python, str(ENGINE), "--model-type", model_type, "--help"],
            capture_output=True, text=True, cwd=ENGINE.parent,
        )
        return result.returncode == 0

    # -- entry gate --------------------------------------------------------------
    def run_overfit(
        self, model: str, extra_args: tuple[str, ...] = (), output_dir: Path | None = None,
        timeout_s: float | None = None,
    ) -> dict:
        """Overfit test of ``model``; reused when the same command already ran.

        Returns ``{passed, reason, cause, returncode, verdict, command, ...}``.
        """
        output_dir = output_dir or self.root / model / "overfit"
        command = self.engine_command(model, GATE_TASK_SET, output_dir, "--overfit-test", *extra_args)
        result_path = output_dir / "gate_result.json"
        cached = read_json(result_path)
        if cached and cached.get("command") == command:
            self.log(f"gate {model}: reuse {result_path}")
            return cached
        if output_dir.exists():
            shutil.rmtree(output_dir)
        self.log(f"gate {model}: overfit test -> {output_dir}")
        started = time.time()
        returncode = self.run_logged(command, output_dir / "engine.log", timeout_s)
        timed_out = self.timed_out
        verdict_path = output_dir / "overfit_verdict.json"
        verdict = read_json(verdict_path)
        if timed_out:
            cause, detail = "budget_exhausted", f"stopped after {timeout_s:.0f} s (auto-correction time budget)"
        elif verdict is None:
            cause, detail = "missing_verdict", f"no overfit_verdict.json (exit code {returncode})"
        else:
            cause, detail = verdict.get("cause"), verdict.get("detail")
            if returncode != 0 and cause is None:
                cause = "engine_error"
                detail = f"exit code {returncode}"
        passed = returncode == 0 and not timed_out and verdict is not None and verdict.get("passed") is True
        result = {
            "model": model,
            "passed": passed,
            "cause": None if passed else cause,
            "reason": None if passed else f"overfit gate failed: {cause}: {detail}",
            "returncode": returncode,
            "verdict": str(verdict_path) if verdict is not None else None,
            "output_dir": str(output_dir),
            "elapsed_s": round(time.time() - started, 1),
            "finished_at": now(),
            "command": command,
        }
        write_json(result_path, result)
        return result

    def is_baseline(self, model: str) -> bool:
        """Baseline = entry gate is the overfit test only (plan key ``baseline``).

        Without the key, a model is a baseline when it is one in ``DEFAULT_PLAN``
        (``mlp`` and ``transformer``), so a plan file written from the default
        plan before the key existed keeps its sensor-token model the only one
        under the validation guard and auto-correction.
        """
        return bool(self.plan["models"][model].get("baseline", model in DEFAULT_BASELINES))

    def gate_checks(self, model: str) -> list[str]:
        return BASELINE_CHECKS if self.is_baseline(model) else CANDIDATE_CHECKS

    def gate_is_current(self, model: str) -> bool:
        record = self.gates.get(model)
        if not record or record.get("passed") is None:
            return False
        # Records without ``checks`` (before baselines): a guard result = overfit + guard.
        checks = record.get("checks") or (CANDIDATE_CHECKS if record.get("guard") else BASELINE_CHECKS)
        if checks != self.gate_checks(model):
            return False
        if "config_signature" in record:
            return record["config_signature"] == self.config_signature(model, GATE_TASK_SET)
        # Schema-1 record (queue before #9): same overfit command up to the interpreter/engine path.
        current = self.engine_command(model, GATE_TASK_SET, self.root / model / "overfit", "--overfit-test")
        return record["passed"] is True and (
            training_engine_args((record.get("command") or [])[3:]) == training_engine_args(current[3:])
        )

    def gate(self, model: str) -> dict:
        """Entry gate of ``model``; re-run whenever its config signature or its checks changed.

        A baseline runs the overfit test only (``overfit_gate``): it never runs
        the guard and never triggers or adopts a correction.
        """
        if not self.gate_is_current(model):
            self.gates[model] = {"model": model, "passed": None, "reason": "running"}
            self.save()
            entry_gate = overfit_gate if self.is_baseline(model) else self.entry_gate
            record = entry_gate(self, model)
            record["checks"] = self.gate_checks(model)
            # Signature after any adoption = the config the gate passed (or failed) with.
            record["config_signature"] = self.config_signature(model, GATE_TASK_SET)
            self.gates[model] = record
            self.save()
        return self.gates[model]

    # -- jobs --------------------------------------------------------------------
    def manifest_groups(self, model: str, task_set: str) -> list[str]:
        """Fold names of the split (``<model>/<task_set>/split_manifest.json``, cached).

        Full CV: the test Participants. Fixed split: one fold, its test Participants
        joined by ``+``. The cache is rebuilt when the plan's split changed.
        """
        job_dir = self.root / model / task_set
        manifest_path = job_dir / "split_manifest.json"
        if not self.split_matches(read_json(manifest_path)):
            command = self.engine_command(model, task_set, job_dir, "--dry-run-split")
            returncode = self.run_logged(command, job_dir / "split.log")
            if returncode != 0:
                raise RuntimeError(f"split failed (exit code {returncode}), see {job_dir / 'split.log'}")
        return [fold["test_group"] for fold in read_json(manifest_path)["folds"]]

    def split_folds(self, job: dict) -> list[str]:
        groups = self.manifest_groups(job["model"], job["task_set"])
        if not self.args.folds:
            return list(groups)
        unknown = sorted(set(self.args.folds) - set(groups))
        if unknown:
            raise RuntimeError(f"unknown folds {unknown}; participants: {groups}")
        return [group for group in groups if group in self.args.folds]

    def write_job_config(self, job: dict, folds: list[str]) -> None:
        sample = self.engine_command(
            job["model"], job["task_set"], Path(job["output_dir"]) / "fold_<P>", "--only-test-group", "<P>"
        )
        write_json(Path(job["output_dir"]) / "queue_config.json", {
            "job": job["name"],
            "model": job["model"],
            "model_type": job["model_type"],
            "task_set": job["task_set"],
            "data_dir": self.state["data_dir"],
            "dataset_version": self.state["dataset_version"],
            "folds": folds,
            "split": self.split() or "full_cv",
            "seed": last_value(sample, "--seed"),
            "engine_commit": self.state["engine_commit"],
            "adopted_overrides": self.adopted_overrides,
            "config_signature": self.config_signature(job["model"], job["task_set"]),
            "config_args": self.config_args(job["model"], job["task_set"]),
            "fold_command": sample,
            "written_at": now(),
        })

    def run_job(self, job: dict) -> int:
        """Train the folds of ``job`` not done with the current config; returns how many ran."""
        job_dir = Path(job["output_dir"])
        folds = self.split_folds(job)
        signature = self.config_signature(job["model"], job["task_set"])
        job["folds_total"] = len(folds)
        job["config_signature"] = signature
        previous = job.get("folds") or {}
        job["folds"] = {fold: {"status": "pending", "reason": None} for fold in folds}
        self.write_job_config(job, folds)
        ran = 0
        for fold in folds:
            fold_dir = job_dir / f"fold_{fold}"
            record = job["folds"][fold]
            if fold_is_done(fold_dir, job["model_type"], fold):
                trained = self.fold_signature(fold_dir, job["model"], job["task_set"])
                if trained == signature:
                    earlier = previous.get(fold) or {}
                    if earlier.get("status") == "done" and earlier.get("config_signature", signature) == signature:
                        record.update({key: earlier[key] for key in FOLD_TIMINGS if key in earlier})
                    record.update(status="done", config_signature=signature)
                    continue
                # Trained with another config (adopted override): set it aside and retrain.
                archive = self.root / STALE_DIR / job["model"] / job["task_set"] / trained / fold_dir.name
                if archive.exists():
                    shutil.rmtree(archive)
                archive.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(fold_dir), str(archive))
                record["retrained_from"] = trained
                self.log(f"{job['name']} fold {fold}: trained with config {trained} != {signature}, retrain")
                self.log_event(
                    "stale_fold", job=job["name"], model=job["model"], fold=fold,
                    trained_signature=trained, current_signature=signature, archived_to=str(archive),
                )
            elif fold_dir.exists():
                self.log(f"{job['name']} fold {fold}: interrupted run, restart")
                shutil.rmtree(fold_dir)
            record.update(status="running", started_at=now())
            ran += 1
            self.save()
            self.log(f"{job['name']} fold {fold}: train -> {fold_dir}")
            started = time.time()
            command = self.engine_command(
                job["model"], job["task_set"], fold_dir, "--only-test-group", fold
            )
            returncode = self.run_logged(command, fold_dir / "engine.log")
            record.update(finished_at=now(), elapsed_s=round(time.time() - started, 1))
            if returncode == 0 and fold_is_done(fold_dir, job["model_type"], fold):
                # Ties the finished fold to the configuration it was trained with.
                write_json(fold_dir / FOLD_CONFIG_FILE, {
                    "schema_version": 1,
                    "job": job["name"],
                    "model": job["model"],
                    "task_set": job["task_set"],
                    "fold": fold,
                    "config_signature": signature,
                    "config_args": self.config_args(job["model"], job["task_set"]),
                    "adopted_overrides": copy.deepcopy(self.adopted_overrides),
                    "command": command,
                    "finished_at": now(),
                })
                record.update(status="done", config_signature=signature)
            else:
                record.update(status="failed", reason=f"engine exit code {returncode}, see engine.log")
            job["folds_done"] = sum(item["status"] == "done" for item in job["folds"].values())
            self.save()
        job["folds_done"] = sum(item["status"] == "done" for item in job["folds"].values())
        failed = [fold for fold, item in job["folds"].items() if item["status"] == "failed"]
        if failed:
            job.update(status="failed", reason=f"folds failed: {', '.join(failed)}")
        else:
            job.update(status="done", reason=None)
        return ran

    def run(self) -> int:
        self.state["dataset_version"] = self.dataset_version
        self.save()
        availability: dict[str, bool] = {}
        # An adoption during a pass changes the config of jobs already run in it:
        # run the selected jobs again (folds of the current config are skipped,
        # stale ones retrained). Each correction is adopted at most once.
        for _ in range(MAX_PASSES):
            before = json.dumps(self.adopted_overrides, sort_keys=True)
            self.run_pass(availability)
            if json.dumps(self.adopted_overrides, sort_keys=True) == before:
                break
            self.log("adopted overrides changed during the pass: run the selected jobs again")
        self.state["runner"].update(state="finished", finished_at=now())
        self.save()
        return 0

    def run_pass(self, availability: dict[str, bool]) -> None:
        for job in self.jobs:
            if self.args.jobs and job["name"] not in self.args.jobs:
                continue
            # A job already started keeps its start; a done job where nothing trained keeps its end.
            finished_before, ran = job["finished_at"], 0
            job.update(status="running", reason=None, started_at=job["started_at"] or now(), finished_at=None)
            self.save()
            model_type = job["model_type"]
            if model_type not in availability:
                availability[model_type] = self.model_type_available(model_type)
            if not availability[model_type]:
                job.update(status="skipped", reason=f"model type {model_type!r} not available in the engine")
            else:
                gate = self.gate(job["model"])
                if not gate["passed"]:
                    job.update(status="failed", reason=gate["reason"])
                else:
                    try:
                        ran = self.run_job(job)
                    except RuntimeError as exc:
                        job.update(status="failed", reason=str(exc))
            keep_end = job["status"] == "done" and not ran and finished_before
            job["finished_at"] = finished_before if keep_end else now()
            self.log(f"{job['name']}: {job['status']}" + (f" ({job['reason']})" if job["reason"] else ""))
            self.save()

    def interrupt(self, signum: int, _frame) -> None:
        """SIGTERM / SIGINT: stop and reap the engine child, record the interruption, exit."""
        child = self.child
        if child is not None:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=CHILD_STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        self.state["runner"].update(state="interrupted", finished_at=now())
        self.save()
        raise SystemExit(128 + signum)


def load_plan(path: Path | None, name: str = "default") -> dict:
    if path is None:
        return copy.deepcopy(NAMED_PLANS[name])
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan.setdefault("common_args", COMMON_ARGS)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path, nargs="?", help="prepared dataset (not needed with --print-plan)")
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--jobs", nargs="+", help="run only these jobs (default: all, in priority order)")
    parser.add_argument(
        "--folds", nargs="+",
        help="restrict every job to these folds (full CV: test Participants; Fixed split: its one fold)",
    )
    parser.add_argument(
        "--full-cv", action="store_true",
        help="full CV (one fold per Participant) instead of the plan's Fixed split",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    plans = parser.add_mutually_exclusive_group()
    plans.add_argument("--plan", type=Path, help="JSON plan replacing the built-in one (models, jobs, common_args)")
    plans.add_argument(
        "--named-plan", choices=sorted(NAMED_PLANS), default="default",
        help="built-in plan: default (spec #1) or v3.7 (spec #12, the launched v3.7 queue)",
    )
    parser.add_argument(
        "--print-plan", action="store_true",
        help="print the plan as JSON (the plan file format) and exit, running nothing",
    )
    parser.add_argument(
        "--engine-arg", action="append", default=[],
        help="flag appended to every engine command, e.g. --engine-arg=--no-plots",
    )
    parser.add_argument("--python", default=sys.executable, help="interpreter running the engine")
    parser.add_argument(
        "--correction-max-attempts", type=int, default=4,
        help="auto-correction: corrections tried per model after a failed gate (default 4)",
    )
    parser.add_argument(
        "--correction-budget-s", type=float, default=3600.0,
        help="auto-correction: wall-clock budget of the corrections per model, in s (default 3600)",
    )
    parser.add_argument(
        "--guard-epochs", type=int, default=3,
        help="validation guard: first-fold epochs compared with the MLP (default 3)",
    )
    args = parser.parse_args(argv)

    plan = load_plan(args.plan, args.named_plan)
    if args.full_cv:
        plan["split"] = "full_cv"
    if args.print_plan:
        sys.stdout.write(json.dumps(plan, indent=1))
        return 0
    if args.data_dir is None or not (args.data_dir / "metadata.json").is_file():
        parser.error(f"dataset invalide: {args.data_dir}")
    split = plan.get("split")
    if split not in (None, "full_cv") and not (
        isinstance(split, dict) and split.get("test") and split.get("validation")
    ):
        parser.error(f'plan "split": "full_cv" or {{"test": [...], "validation": [...]}}, not {split!r}')
    names = [job["name"] for job in plan["jobs"]]
    if args.jobs and set(args.jobs) - set(names):
        parser.error(f"jobs inconnus {sorted(set(args.jobs) - set(names))}; disponibles: {names}")
    args.results_root.mkdir(parents=True, exist_ok=True)

    lock = (args.results_root / LOCK_FILE).open("a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        print(f"another queue runner holds {lock.name} (pid {lock.read().strip() or '?'})", file=sys.stderr)
        return EXIT_LOCKED
    lock.seek(0)
    lock.truncate()
    lock.write(f"{os.getpid()}\n")
    lock.flush()

    queue = Queue(args, plan, entry_gate=auto_correct_gate)
    signal.signal(signal.SIGTERM, queue.interrupt)
    signal.signal(signal.SIGINT, queue.interrupt)
    return queue.run()


if __name__ == "__main__":
    raise SystemExit(main())
