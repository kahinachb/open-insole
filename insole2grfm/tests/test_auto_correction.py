"""Validation guard and bounded auto-correction of the overnight queue (#9).

Driven through the queue CLI on the synthetic dataset. The overfit NMSE
threshold of the fast settings is huge, so an overfit test fails only on NaN
or divergence (mean NMSE > 10 x the untrained one).
"""

from __future__ import annotations

import json

from test_overnight_queue import jobs_by_name, run_queue, status, write_plan

# MLP hidden 8 on the synthetic set (20 overfit epochs, no clipping): lr 3
# diverges at the first epoch, lr 1 = 3 / 3 trains.
OVERFIT_FAST = ("--engine-arg=--overfit-epochs=20",)
# lr inf: NaN whatever the warmup, clipping or normalisation.
BROKEN = ["--layers", "1", "--learning-rate", "inf", "--gradient-clip", "0"]


def log_events(root, event=None):
    path = root / "auto_correction.jsonl"
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return [item for item in events if event is None or item["event"] == event]


FRAGILE = ["--layers", "1", "--learning-rate", "3", "--gradient-clip", "0"]  # lr 3 diverges, lr 1 trains


def test_broken_model_is_excluded_after_4_stacked_corrections_for_the_whole_night(synthetic_prepared, tmp_path):
    plan = write_plan(
        tmp_path / "plan.json",
        models={"broken": {"model_type": "mlp", "engine_args": BROKEN},
                "fragile": {"model_type": "mlp", "engine_args": FRAGILE}},
        jobs=[
            {"name": "broken_continuous", "model": "broken", "task_set": "continuous"},
            {"name": "fragile_continuous", "model": "fragile", "task_set": "continuous"},
        ],
    )
    root = tmp_path / "results"
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--folds", "T01", *OVERFIT_FAST)

    checks = [event for event in log_events(root, "check") if event["model"] == "broken"]
    # First gate run: the initial check, then at most 4 attempts (GradNorm alpha 0.5,
    # the 5th, is never reached). Corrections stack.
    first = checks[:5]
    assert [event["attempt"] for event in first] == [0, 1, 2, 3, 4]
    assert [event["correction"] for event in first] == [
        None, "lr_div3", "warmup_500", "gradient_clip_0.5", "input_layer_norm",
    ]
    assert first[-1]["corrections"] == ["lr_div3", "warmup_500", "gradient_clip_0.5", "input_layer_norm"]
    assert all(event["passed"] is False and event["overfit"]["cause"] == "nan" for event in checks)
    assert first[2]["flags"][-2:] == ["--warmup-steps", "500"]
    # fragile then adopts lr / 3 for every model: broken's config changes and its gate runs
    # again in the next pass, but its 4 attempts of the night are spent: initial check only.
    assert [event["correction"] for event in checks[5:]] == [None]
    excluded = [event for event in log_events(root, "excluded") if event["model"] == "broken"]
    assert len(excluded) == 2
    assert "4 correction attempts" in excluded[0]["diagnosis"] and "nan" in excluded[0]["diagnosis"]
    assert "maximum of 4 attempts reached" in excluded[1]["diagnosis"]
    budget = status(root)["correction_budget"]
    assert budget["broken"]["attempts"] == 4 and budget["broken"]["elapsed_s"] > 0
    assert budget["fragile"]["attempts"] == 1

    job = jobs_by_name(root)["broken_continuous"]
    assert job["status"] == "failed"
    assert job["reason"] == status(root)["gates"]["broken"]["reason"]
    assert "excluded" in job["reason"] and "nan" in job["reason"]
    assert not (root / "broken" / "continuous" / "fold_T01").exists()
    # Nothing adopted from a failure: only fragile's lr / 3.
    assert [event["model"] for event in log_events(root, "adopted")] == ["fragile"]
    assert status(root)["adopted_overrides"]["all_models"] == []
    assert jobs_by_name(root)["fragile_continuous"]["status"] == "done"
    assert status(root)["auto_correction_log"] == str(root / "auto_correction.jsonl")


def metadata(fold_dir):
    return json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))


def test_fixable_model_is_recovered_and_its_global_change_retrains_every_model(synthetic_prepared, tmp_path):
    plan = write_plan(
        tmp_path / "plan.json",
        models={
            # Reference of the guard; a tiny lr keeps its validation loss high.
            "mlp": {"model_type": "mlp", "engine_args": ["--layers", "1", "--learning-rate", "1e-6"]},
            "fragile": {"model_type": "mlp", "engine_args": FRAGILE},
        },
        jobs=[
            {"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"},
            {"name": "fragile_continuous", "model": "fragile", "task_set": "continuous"},
        ],
    )
    root = tmp_path / "results"
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--folds", "T01", *OVERFIT_FAST)

    checks = log_events(root, "check")
    fragile = [event for event in checks if event["model"] == "fragile"]
    assert [(event["correction"], event["passed"]) for event in fragile] == [(None, False), ("lr_div3", True)]
    assert fragile[0]["overfit"]["cause"] == "divergence"
    assert fragile[1]["flags"] == ["--learning-rate", "1"]
    guard = fragile[1]["guard"]
    # Same fold (first of the split) and epoch as the MLP; judged with lr / 3, the candidate
    # faces an MLP trained with lr / 3 too, not the MLP job's finished fold at the old lr.
    assert guard["passed"] is True and guard["fold"] == "T01" and guard["epoch"] == 2
    assert guard["reference_source"] == "reference_run"
    assert guard["reference_flags"] == ["--learning-rate", "3.33333e-07"]
    assert guard["candidate_validation_loss"][1] < guard["reference_validation_loss"][1]
    # The MLP (reference) never runs the guard.
    assert all(event["guard"] is None for event in checks if event["model"] == "mlp")

    adopted = log_events(root, "adopted")
    assert [(event["model"], [c["name"] for c in event["corrections"]]) for event in adopted] == [
        ("fragile", ["lr_div3"])
    ]
    overrides = status(root)["adopted_overrides"]
    assert overrides["corrections"]["all_models"] == ["lr_div3"]
    assert abs(overrides["learning_rate_factor"] - 1 / 3) < 1e-12
    gates = status(root)["gates"]
    assert gates["fragile"]["adopted"] == ["lr_div3"]

    # MLP folds finished before the adoption are invalidated and retrained with lr / 3.
    invalidated = log_events(root, "invalidated")
    assert [(event["job"], event["folds"]) for event in invalidated] == [("mlp_continuous", ["T01"])]
    stale = {event["fold"]: event for event in log_events(root, "stale_fold")}
    assert sorted(stale) == ["T01"]
    archived = root / "stale" / "mlp" / "continuous" / stale["T01"]["trained_signature"] / "fold_T01"
    assert metadata(archived)["learning_rate"] == 1e-6
    jobs = jobs_by_name(root)
    for name, model, learning_rate in (("mlp_continuous", "mlp", 1e-6 / 3), ("fragile_continuous", "fragile", 1.0)):
        job = jobs[name]
        assert job["status"] == "done" and job["folds_done"] == 1
        for fold in ("T01",):
            fold_dir = root / model / "continuous" / f"fold_{fold}"
            assert abs(metadata(fold_dir)["learning_rate"] / learning_rate - 1) < 1e-5
            recorded = json.loads((fold_dir / "queue_fold.json").read_text(encoding="utf-8"))
            assert recorded["config_signature"] == job["config_signature"] == job["folds"][fold]["config_signature"]
    # The MLP re-passed its overfit gate under the new configuration.
    assert gates["mlp"]["passed"] is True
    assert "3.33333e-07" in gates["mlp"]["overfit"]["command"]

    # No decision reads test data: guard runs are validation-only.
    for guard_dir in root.glob("*/guard*"):
        assert not list(guard_dir.rglob("predictions.csv")) and not list(guard_dir.rglob("fold_metrics.csv"))


def test_model_losing_to_the_mlp_fails_the_guard_and_is_excluded_within_the_attempt_limit(
    synthetic_prepared, tmp_path
):
    plan = write_plan(
        tmp_path / "plan.json",
        models={
            "mlp": {"model_type": "mlp", "engine_args": ["--layers", "1", "--learning-rate", "3e-3"]},
            # Learns nothing in 2 epochs: trains fine (overfit passes), loses to the MLP.
            "slow": {"model_type": "mlp", "engine_args": ["--layers", "1", "--learning-rate", "1e-9"]},
        },
        # The guard runs on the candidate's first job's task set (`all`), against the MLP on `all`.
        jobs=[
            {"name": "mlp_all", "model": "mlp", "task_set": "all"},
            {"name": "slow_all", "model": "slow", "task_set": "all"},
        ],
    )
    root = tmp_path / "results"
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--folds", "T01",
              "--correction-max-attempts", "1", *OVERFIT_FAST)

    checks = [event for event in log_events(root, "check") if event["model"] == "slow"]
    assert [(event["correction"], event["cause"]) for event in checks] == [
        (None, "not_better_than_reference"), ("lr_div3", "not_better_than_reference"),
    ]
    # Uncorrected: the MLP curve is the MLP job's own finished first fold (same config).
    guard = checks[0]["guard"]
    assert guard["reference_source"] == "job_history" and guard["fold"] == "T01"
    assert guard["task_set"] == "all"
    assert guard["reference_history"] == str(root / "mlp" / "all" / "fold_T01" / "training_history.json")
    assert guard["candidate_validation_loss"][1] >= guard["reference_validation_loss"][1]
    assert guard["reference_flags"] == []
    # Judged with lr / 3 (a global correction), the candidate faces an MLP trained with
    # its own lr / 3 on the same fold, not the MLP of the uncorrected configuration.
    corrected = checks[1]["guard"]
    assert corrected["reference_flags"] == ["--learning-rate", "0.001"]
    assert corrected["reference_source"] == "reference_run" and corrected["fold"] == "T01"
    reference_dir = root / "mlp" / "guard_reference_attempt_1"
    assert corrected["reference_history"] == str(reference_dir / "training_history.json")
    assert metadata(reference_dir)["learning_rate"] == 1e-3
    assert corrected["reference_validation_loss"] != guard["reference_validation_loss"]
    excluded = log_events(root, "excluded")
    assert "maximum of 1 attempts" in excluded[0]["diagnosis"]
    assert metadata(reference_dir)["task_set"] == "all"
    job = jobs_by_name(root)["slow_all"]
    assert job["status"] == "failed" and "validation guard failed" in job["reason"]
    assert not log_events(root, "adopted")
    assert jobs_by_name(root)["mlp_all"]["status"] == "done"


def test_exhausted_time_budget_excludes_the_model_without_corrections(synthetic_prepared, tmp_path):
    plan = write_plan(
        tmp_path / "plan.json",
        models={"broken": {"model_type": "mlp", "engine_args": BROKEN}},
        jobs=[{"name": "broken_continuous", "model": "broken", "task_set": "continuous"}],
    )
    root = tmp_path / "results"
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--correction-budget-s", "0", *OVERFIT_FAST)

    assert [event["attempt"] for event in log_events(root, "check")] == [0]
    diagnosis = log_events(root, "excluded")[0]["diagnosis"]
    assert "time budget of 0 s exhausted" in diagnosis and "nan" in diagnosis
    assert jobs_by_name(root)["broken_continuous"]["status"] == "failed"


def test_folds_and_gate_of_a_schema_1_status_still_count_as_done(synthetic_prepared, tmp_path):
    plan = write_plan(tmp_path / "plan.json", {"mlp": {"model_type": "mlp", "engine_args": ["--layers", "1"]}},
                      [{"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"}])
    root = tmp_path / "results"
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--folds", "T01")
    # Rewind to what the queue before #9 leaves on disk: no fold config record,
    # schema-1 status (no signatures, no adopted factor / corrections).
    fold_dir = root / "mlp" / "continuous" / "fold_T01"
    (fold_dir / "queue_fold.json").unlink()
    legacy = status(root)
    legacy["schema_version"] = 1
    legacy["adopted_overrides"] = {"all_models": [], "per_model": {}}
    del legacy["auto_correction_log"]
    for record in legacy["gates"].values():
        for key in ("config_signature", "attempts", "adopted", "settings", "excluded", "diagnosis", "guard"):
            record.pop(key, None)
        record.update(record.pop("overfit"))
    (root / "queue_status.json").write_text(json.dumps(legacy), encoding="utf-8")
    (root / "auto_correction.jsonl").unlink(missing_ok=True)  # a baseline gate logs no check
    checkpoint = fold_dir / "models" / "mlp_test_T01.pt"
    mtime = checkpoint.stat().st_mtime_ns

    # Operational engine flags (plots, workers) are not part of the config signature.
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--folds", "T01",
              "--engine-arg=--num-workers", "--engine-arg=0", "--engine-arg=--no-plots")

    job = jobs_by_name(root)["mlp_continuous"]
    assert job["status"] == "done" and job["folds"]["T01"]["status"] == "done"
    assert checkpoint.stat().st_mtime_ns == mtime
    assert not (root / "auto_correction.jsonl").exists()  # gate reused, nothing re-checked
    assert status(root)["schema_version"] == 2


def test_baselines_only_run_the_overfit_gate_and_a_failed_guard_record_is_not_reused(synthetic_prepared, tmp_path):
    reference = {"model_type": "mlp", "engine_args": ["--layers", "1", "--learning-rate", "3e-3"]}
    # Loses to the MLP in 2 epochs (see the guard test above).
    slow = {"model_type": "mlp", "engine_args": ["--layers", "1", "--learning-rate", "1e-9"]}
    jobs = [{"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"},
            {"name": "transformer_continuous", "model": "transformer", "task_set": "continuous"}]
    root = tmp_path / "results"
    # As a candidate it fails the guard; no correction left to try.
    plan = write_plan(tmp_path / "candidate.json", {"mlp": reference, "transformer": {**slow, "baseline": False}}, jobs)
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--folds", "T01",
              "--correction-max-attempts", "0", *OVERFIT_FAST)
    assert "validation guard failed" in jobs_by_name(root)["transformer_continuous"]["reason"]

    # A plan without the key (written from the old default plan): mlp and transformer are
    # baselines. The failed guard record is not reused: the overfit test alone decides.
    plan = write_plan(
        tmp_path / "old_plan.json",
        {"mlp": reference, "transformer": slow, "broken": {"model_type": "mlp", "engine_args": BROKEN, "baseline": True}},
        [*jobs, {"name": "broken_continuous", "model": "broken", "task_set": "continuous"}],
    )
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--folds", "T01", *OVERFIT_FAST)

    jobs = jobs_by_name(root)
    gates = status(root)["gates"]
    assert jobs["transformer_continuous"]["status"] == "done"
    assert gates["transformer"]["passed"] is True and gates["transformer"]["checks"] == ["overfit"]
    # A baseline failing the overfit test is failed with its reason, without any correction.
    assert jobs["broken_continuous"]["status"] == "failed"
    assert jobs["broken_continuous"]["reason"].startswith("overfit gate failed: nan")
    assert not list((root / "broken").glob("*attempt*")) and not (root / "broken" / "guard").exists()
    assert "broken" not in status(root)["correction_budget"]
    assert [event["model"] for event in log_events(root, "check")] == ["transformer"]  # run 1 only
