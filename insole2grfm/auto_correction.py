"""Entry gate of the overnight queue with validation guard and bounded auto-correction (#9).

Applies to the non-baseline models only (the sensor-token model); a baseline
(plan key ``baseline``, see ``Queue.is_baseline``) runs the overfit test alone.
A model enters the queue when its configuration passes

1. the engine's overfit test (``--overfit-test``), and
2. the validation guard (every model except the reference, the MLP): a short
   training of the first fold (``--only-test-group <first> --epochs k
   --selection-only``, on the task set of the candidate's first plan job and the
   first fold of the split, the Fixed split's one fold by default) whose validation
   loss at epoch k must be lower than the MLP's at the same epoch of the same fold
   and task set. The MLP curve comes from the MLP
   job's own ``fold_<first>/training_history.json`` when that fold is finished
   with the current configuration, else from a short reference run of the MLP
   (``<root>/mlp/guard_reference/``) with the current configuration. When the
   candidate is judged with loss/optimiser corrections (lr / 3, warmup,
   clipping, GradNorm alpha), the MLP reference is trained with the same ones
   (same fold, same epochs, ``mlp/guard_reference_attempt_<N>/``);
   architecture corrections (input LayerNorm) do not apply to the reference.

If the check fails, corrections are stacked in this order, each one
re-checked: lr / 3, 500-step warmup, gradient clip 0.5, input LayerNorm
(``--input-layer-norm``, the architectures are already pre-norm), GradNorm
alpha 0.5. A correction already in effect for the model is skipped. At most
``--correction-max-attempts`` (4) attempts and ``--correction-budget-s``
(3600 s) of correction time per model for the whole results root, across
queue passes, gate re-runs after an adoption and restarts (the consumption is
persisted in ``queue_status.json`` ``correction_budget``); a running attempt is
stopped when the budget runs out. The first passing stack is adopted: loss/optimiser
corrections for every model of the queue (``adopted_overrides.all_models`` /
``learning_rate_factor``; finished folds of the other configuration are
retrained), architecture ones for this model only. Decisions read only overfit
verdicts and validation losses, never test predictions or metrics (the guard
runs with ``--selection-only``: no test predictions are even computed).

Every check, skipped correction, adoption and exclusion is one JSON line of
``<root>/auto_correction.jsonl`` (see ``Queue.log_event``).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

from queue_common import GATE_TASK_SET, GUARD_REFERENCE, last_value, now, read_json, write_json

if TYPE_CHECKING:  # the queue injects this gate; no import of overnight_queue at run time
    from overnight_queue import Queue

# (name, scope, flags); lr / 3 is computed from the model's effective learning rate.
CORRECTIONS = (
    ("lr_div3", "all_models", None),
    ("warmup_500", "all_models", ("--warmup-steps", "500")),
    ("gradient_clip_0.5", "all_models", ("--gradient-clip", "0.5")),
    ("input_layer_norm", "per_model", ("--input-layer-norm",)),
    ("gradnorm_alpha_0.5", "all_models", ("--gradnorm-alpha", "0.5")),
)
CORRECTION_FLAGS = {name: flags for name, _, flags in CORRECTIONS}


def correction_flags(queue: Queue, model: str, name: str, flags) -> list[str]:
    if name == "lr_div3":
        return ["--learning-rate", format(queue.learning_rate(model) / 3.0, ".6g")]
    return list(flags)


def already_in_effect(queue: Queue, model: str, name: str, flags) -> str | None:
    """Why ``name`` is not tried for ``model`` (None = try it)."""
    adopted = queue.adopted_overrides["corrections"]
    if name in adopted["all_models"] or name in adopted["per_model"].get(model, []):
        return "already adopted"
    if flags is None:
        return None
    command = queue.engine_command(model, GATE_TASK_SET, Path("-"))
    if len(flags) == 1 and flags[0] in command:
        return "already in the model configuration"
    if len(flags) == 2 and last_value(command, flags[0]) is not None:
        if float(last_value(command, flags[0])) == float(flags[1]):
            return "already in the model configuration"
    return None


def guard_reference(queue: Queue) -> str:
    return queue.plan.get("guard_reference", GUARD_REFERENCE)


def history_curve(path: Path, fold: str) -> list[float] | None:
    history = read_json(path)
    if not history or fold not in history:
        return None
    return [float(row["validation_loss"]) for row in history[fold]]


def guard_run(queue: Queue, model: str, task_set: str, output_dir: Path, fold: str, extra: list[str],
              timeout_s) -> dict:
    """Short first-fold training (validation only); cached while the command is identical."""
    command = queue.engine_command(
        model, task_set, output_dir, "--only-test-group", fold,
        "--epochs", str(queue.args.guard_epochs), "--selection-only", *extra,
    )
    result_path = output_dir / "guard_run.json"
    cached = read_json(result_path)
    if cached and cached.get("command") == command:
        return cached
    if output_dir.exists():
        shutil.rmtree(output_dir)
    started = time.time()
    returncode = queue.run_logged(command, output_dir / "engine.log", timeout_s)
    result = {
        "returncode": returncode,
        "timed_out": queue.timed_out,
        "validation_loss": history_curve(output_dir / "training_history.json", fold),
        "output_dir": str(output_dir),
        "elapsed_s": round(time.time() - started, 1),
        "command": command,
    }
    write_json(result_path, result)
    return result


def reference_flags(queue: Queue, stack: list[dict]) -> list[str]:
    """Global (loss/optimiser) corrections of ``stack`` as the reference model trains with them.

    Fairness: a candidate judged with lr / 3, warmup, clipping or GradNorm alpha
    faces an MLP trained with the same corrections (lr / 3 of the MLP's own lr);
    architecture corrections (``per_model``) do not apply to the reference.
    """
    reference = guard_reference(queue)
    if reference not in queue.plan["models"]:
        return []  # no reference: the guard is skipped
    return [
        flag for item in stack if item["scope"] == "all_models"
        for flag in correction_flags(queue, reference, item["name"], CORRECTION_FLAGS[item["name"]])
    ]


def reference_curve(queue: Queue, task_set: str, fold: str, extra: list[str], suffix: str,
                    timeout_s) -> tuple[list[float] | None, str, str]:
    """MLP validation curve of the first fold with the global corrections ``extra``.

    Returns (curve, source kind, source path). Without corrections the MLP job's
    own finished fold is used when it has the current configuration; otherwise a
    short reference run (``<reference>/guard_reference<suffix>/``) on the same
    fold and epochs as the candidate.
    """
    reference = guard_reference(queue)
    fold_dir = queue.root / reference / task_set / f"fold_{fold}"
    if not extra and queue.fold_is_current(reference, task_set, fold):
        path = fold_dir / "training_history.json"
        return history_curve(path, fold), "job_history", str(path)
    run = guard_run(queue, reference, task_set, queue.root / reference / f"guard_reference{suffix}", fold, extra,
                    timeout_s)
    return run["validation_loss"], "reference_run", str(Path(run["output_dir"]) / "training_history.json")


def validation_guard(queue: Queue, model: str, extra: list[str], output_dir: Path, timeout_s,
                     reference_extra: list[str] = (), suffix: str = "") -> dict:
    """Candidate validation loss vs the MLP's at the same epoch of the first fold.

    ``reference_extra`` = the global corrections the candidate is judged with
    (see ``reference_flags``), applied to the reference run too.
    """
    reference = guard_reference(queue)
    if reference not in queue.plan["models"]:
        return {"passed": True, "skipped": True, "cause": None,
                "reason": f"no reference model {reference!r} in the plan: guard skipped"}
    # Same task set (the candidate's first job) and same first fold of the split for both models.
    task_set = queue.guard_task_set(model)
    fold = queue.manifest_groups(reference, task_set)[0]
    started = time.time()
    reference_extra = list(reference_extra)
    reference_loss, source, source_path = reference_curve(queue, task_set, fold, reference_extra, suffix, timeout_s)
    remaining = None if timeout_s is None else timeout_s - (time.time() - started)
    run = guard_run(queue, model, task_set, output_dir, fold, extra, remaining)
    candidate_loss = run["validation_loss"]
    record = {
        "task_set": task_set,
        "fold": fold,
        "epochs": queue.args.guard_epochs,
        "epoch": None,
        "candidate_validation_loss": candidate_loss,
        "reference_model": reference,
        "reference_validation_loss": reference_loss,
        "reference_source": source,
        "reference_history": source_path,
        "reference_flags": reference_extra,
        "output_dir": run["output_dir"],
        "command": run["command"],
    }
    if run["timed_out"]:
        cause, reason = "budget_exhausted", "guard run stopped by the auto-correction time budget"
    elif not reference_loss:
        cause, reason = "no_reference", f"no {reference} validation curve for fold {fold} ({source_path})"
    elif run["returncode"] != 0 or not candidate_loss:
        cause, reason = "engine_error", f"guard run exit code {run['returncode']}, see {run['output_dir']}/engine.log"
    else:
        epoch = min(queue.args.guard_epochs, len(candidate_loss), len(reference_loss))
        candidate, mlp = candidate_loss[epoch - 1], reference_loss[epoch - 1]
        record["epoch"] = epoch
        if candidate < mlp:
            cause, reason = None, None
        else:
            cause = "not_better_than_reference"
            reason = (f"validation loss {candidate:.6g} at epoch {epoch} of fold {fold} "
                      f"is not below {reference}'s {mlp:.6g}")
    return {"passed": cause is None, "skipped": False, "cause": cause, "reason": reason, **record}


def check(queue: Queue, model: str, attempt: int, extra: list[str], timeout_s,
          reference_extra: list[str] = ()) -> dict:
    """Overfit test then (if it passed and the model is not the reference) the guard."""
    suffix = "" if attempt == 0 else f"_attempt_{attempt}"
    overfit = queue.run_overfit(model, tuple(extra), queue.root / model / f"overfit{suffix}", timeout_s)
    guard = None
    if overfit["passed"] and model != guard_reference(queue):
        remaining = None if timeout_s is None else timeout_s - overfit["elapsed_s"]
        guard = validation_guard(queue, model, extra, queue.root / model / f"guard{suffix}", remaining,
                                 reference_extra, suffix)
    if not overfit["passed"]:
        passed, cause, reason = False, overfit["cause"], overfit["reason"]
    elif guard is not None and not guard["passed"]:
        passed, cause, reason = False, guard["cause"], f"validation guard failed: {guard['cause']}: {guard['reason']}"
    else:
        passed, cause, reason = True, None, None
    return {"passed": passed, "cause": cause, "reason": reason, "overfit": overfit, "guard": guard}


def summary(result: dict) -> dict:
    """Compact form of a check for the log and the status file."""
    overfit, guard = result["overfit"], result["guard"]
    return {
        "passed": result["passed"],
        "cause": result["cause"],
        "reason": result["reason"],
        "overfit": {key: overfit.get(key) for key in ("passed", "cause", "reason", "output_dir", "verdict", "elapsed_s")},
        "guard": None if guard is None else {
            key: guard.get(key) for key in (
                "passed", "skipped", "cause", "reason", "task_set", "fold", "epoch", "candidate_validation_loss",
                "reference_validation_loss", "reference_source", "reference_history", "reference_flags",
                "output_dir",
            )
        },
    }


def auto_correct_gate(queue: Queue, model: str) -> dict:
    """``Queue`` entry gate: check, then stacked corrections within the budget."""
    settings = {
        "max_attempts": queue.args.correction_max_attempts,
        "budget_s": queue.args.correction_budget_s,
        "guard_epochs": queue.args.guard_epochs,
        "guard_reference": guard_reference(queue),
    }
    started = time.time()
    attempts: list[dict] = []
    stack: list[dict] = []

    def record_check(attempt: int, correction: str | None, extra: list[str], result: dict) -> dict:
        entry = {"attempt": attempt, "correction": correction,
                 "corrections": [item["name"] for item in stack], "flags": extra, **summary(result)}
        attempts.append(entry)
        queue.log(f"gate {model}: attempt {attempt} ({correction or 'initial'}) "
                  + ("passed" if result["passed"] else f"failed: {result['reason']}"))
        queue.log_event("check", model=model, **entry)
        return entry

    def finish(passed: bool, last: dict, reason: str | None, diagnosis: str | None) -> dict:
        return {
            "model": model,
            "passed": passed,
            "cause": last["cause"],
            "reason": reason,
            "diagnosis": diagnosis,
            "excluded": not passed,
            "attempts": attempts,
            "adopted": [item["name"] for item in stack] if passed else [],
            "settings": settings,
            "overfit": last["overfit"],
            "guard": last["guard"],
            "elapsed_s": round(time.time() - started, 1),
            "finished_at": now(),
        }

    result = check(queue, model, 0, [], None)
    record_check(0, None, [], result)
    if result["passed"]:
        return finish(True, result, None, None)

    # The budget is per model for the whole results root: attempts and seconds
    # spent by earlier gate runs of this model (previous pass, restart) count.
    usage = queue.correction_usage(model)
    correction_started = time.time()
    deadline = correction_started + settings["budget_s"] - usage["elapsed_s"]
    stop = None
    for name, scope, flags in CORRECTIONS:
        if usage["attempts"] >= settings["max_attempts"]:
            stop = f"maximum of {settings['max_attempts']} attempts reached ({usage['attempts']} used for {model})"
            break
        why = already_in_effect(queue, model, name, flags)
        if why:
            queue.log_event("skipped_correction", model=model, correction=name, why=why)
            continue
        remaining = deadline - time.time()
        if remaining <= 0:
            stop = f"time budget of {settings['budget_s']:.0f} s exhausted"
            break
        stack.append({"name": name, "scope": scope, "flags": correction_flags(queue, model, name, flags)})
        extra = [flag for item in stack for flag in item["flags"]]
        attempt_started = time.time()
        result = check(queue, model, len(attempts), extra, remaining, reference_flags(queue, stack))
        queue.spend_correction(model, time.time() - attempt_started)
        record_check(len(attempts), name, extra, result)
        if result["passed"]:
            queue.adopt(model, stack)
            return finish(True, result, None, None)
        if result["cause"] == "budget_exhausted":
            stop = f"time budget of {settings['budget_s']:.0f} s exhausted"
            break
    tried = len(attempts) - 1
    stop = stop or "no correction left to try"
    diagnosis = (
        f"{tried} correction attempts ({', '.join(item['name'] for item in stack) or 'none'}) "
        f"in {time.time() - correction_started:.0f} s, {stop}; initial failure: {attempts[0]['reason']}; "
        f"last failure: {attempts[-1]['reason']}"
    )
    queue.log_event("excluded", model=model, diagnosis=diagnosis, attempts=tried,
                    elapsed_s=round(time.time() - started, 1), settings=settings)
    return finish(False, result, f"excluded by auto-correction: {diagnosis}", diagnosis)
