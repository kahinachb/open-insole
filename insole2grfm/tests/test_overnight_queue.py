"""Resumable overnight job queue, driven through its CLI on the synthetic dataset."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from synthetic_prepared import REPO_ROOT

QUEUE = REPO_ROOT / "overnight_queue.py"
# Fast CPU settings appended to every engine command (argparse keeps the last).
# The huge NMSE threshold makes the overfit gate pass unless training blows up.
FAST = (
    "--device", "cpu",
    "--engine-arg=--epochs=2", "--engine-arg=--patience=2",
    "--engine-arg=--batch-size=64", "--engine-arg=--hidden-size=8",
    "--engine-arg=--num-workers=0", "--engine-arg=--no-plots",
    "--engine-arg=--overfit-epochs=2", "--engine-arg=--overfit-windows-per-stratum=8",
    "--engine-arg=--overfit-nmse-threshold=1e9",
)


def run_queue(data_dir, root, *args, check=True):
    command = [sys.executable, str(QUEUE), str(data_dir), "--results-root", str(root), *FAST, *args]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise AssertionError(f"queue failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
    return result


def status(root):
    return json.loads((root / "queue_status.json").read_text(encoding="utf-8"))


def jobs_by_name(root):
    return {job["name"]: job for job in status(root)["jobs"]}


def write_plan(path, models, jobs):
    path.write_text(json.dumps({"models": models, "jobs": jobs}), encoding="utf-8")
    return path


def test_failed_gate_and_unavailable_model_do_not_stop_the_queue(synthetic_prepared, tmp_path):
    plan = write_plan(
        tmp_path / "plan.json",
        models={
            # An infinite learning rate makes the overfit test end on NaN.
            "broken": {"model_type": "mlp", "engine_args": ["--learning-rate", "inf", "--gradient-clip", "0"]},
            "future": {"model_type": "not_in_the_engine_yet", "engine_args": []},
            "tcn": {"model_type": "tcn", "engine_args": ["--layers", "1"]},
        },
        jobs=[
            {"name": "broken_continuous", "model": "broken", "task_set": "continuous"},
            {"name": "future_continuous", "model": "future", "task_set": "continuous"},
            {"name": "tcn_continuous", "model": "tcn", "task_set": "continuous"},
            {"name": "broken_all", "model": "broken", "task_set": "all"},
        ],
    )
    root = tmp_path / "results"
    # No auto-correction attempt: only the gate's verdict matters here.
    result = run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan), "--folds", "T01",
                       "--correction-max-attempts", "0", check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    jobs = jobs_by_name(root)
    assert [job["name"] for job in status(root)["jobs"]] == [
        "broken_continuous", "future_continuous", "tcn_continuous", "broken_all",
    ]
    for name in ("broken_continuous", "broken_all"):
        assert jobs[name]["status"] == "failed"
        assert "overfit" in jobs[name]["reason"] and "nan" in jobs[name]["reason"]
    assert jobs["future_continuous"]["status"] == "skipped"
    assert "not_in_the_engine_yet" in jobs["future_continuous"]["reason"]
    assert jobs["tcn_continuous"]["status"] == "done"
    assert jobs["tcn_continuous"]["reason"] is None

    # The failed model never trained; its overfit artefacts are kept for the report.
    assert (root / "broken" / "overfit" / "overfit_verdict.json").is_file()
    assert not (root / "broken" / "continuous" / "fold_T01").exists()
    fold = root / "tcn" / "continuous" / "fold_T01"
    for name in ("predictions.csv", "fold_metrics.csv", "training_history.json",
                 "gradnorm_weights.csv", "metadata.json", "models/tcn_test_T01.pt"):
        assert (fold / name).is_file(), name
    assert (root / "tcn" / "continuous" / "split_manifest.json").is_file()
    metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["gradnorm"] is True and metadata["only_test_group"] == "T01"


MLP_ONLY = {"mlp": {"model_type": "mlp", "engine_args": ["--layers", "1"]}}


def test_killed_queue_resumes_without_retraining_finished_folds(synthetic_prepared, tmp_path):
    plan = write_plan(tmp_path / "plan.json", MLP_ONLY,
                      [{"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"}])
    root = tmp_path / "results"
    job_dir = root / "mlp" / "continuous"
    command = [sys.executable, str(QUEUE), str(synthetic_prepared.data_dir), "--results-root", str(root),
               *FAST, "--plan", str(plan)]
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    # Kill the whole process group (queue + engine) while the second fold trains.
    deadline = time.time() + 60
    while not (job_dir / "fold_T03" / "split_manifest.json").exists():
        assert process.poll() is None and time.time() < deadline, "queue ended before the second fold"
        time.sleep(0.02)
    os.killpg(process.pid, signal.SIGKILL)
    process.wait()

    killed = jobs_by_name(root)["mlp_continuous"]
    assert killed["status"] == "running"
    assert killed["folds"]["T01"]["status"] == "done"
    assert killed["folds"]["T03"]["status"] == "running"
    assert not (job_dir / "fold_T03" / "metadata.json").exists()
    finished = job_dir / "fold_T01" / "models" / "mlp_test_T01.pt"
    finished_mtime = finished.stat().st_mtime_ns
    timings = ("started_at", "finished_at", "elapsed_s")
    alice = {key: killed["folds"]["T01"][key] for key in timings}
    assert alice["elapsed_s"] > 0

    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan))

    job = jobs_by_name(root)["mlp_continuous"]
    assert job["status"] == "done" and job["folds_done"] == job["folds_total"] == 4
    assert {fold["status"] for fold in job["folds"].values()} == {"done"}
    assert finished.stat().st_mtime_ns == finished_mtime  # never retrained
    # The relaunch keeps the timings of what was already done.
    assert {key: job["folds"]["T01"][key] for key in timings} == alice
    assert job["started_at"] == killed["started_at"]
    finished_job = {key: job[key] for key in ("started_at", "finished_at")}
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan))
    again = jobs_by_name(root)["mlp_continuous"]
    assert {key: again[key] for key in ("started_at", "finished_at")} == finished_job
    assert {key: again["folds"]["T03"][key] for key in timings} == {key: job["folds"]["T03"][key] for key in timings}
    assert (job_dir / "fold_T03" / "models" / "mlp_test_T03.pt").is_file()  # restarted
    assert status(root)["runner"]["state"] == "finished"
    config = json.loads((job_dir / "queue_config.json").read_text(encoding="utf-8"))
    assert config["folds"] == ["T01", "T03", "T04", "T05"]
    assert config["seed"] == "42"
    assert config["dataset_version"].startswith("prepared_joint_masked_v3.6")


def test_fixed_split_plan_runs_one_fold_and_never_mixes_with_full_cv(synthetic_prepared, tmp_path):
    jobs = [{"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"}]
    fixed_plan = tmp_path / "fixed.json"
    fixed_plan.write_text(json.dumps({
        "models": MLP_ONLY, "jobs": jobs, "split": {"test": ["T01", "T03"], "validation": ["T04"]},
    }), encoding="utf-8")
    root = tmp_path / "results"
    job_dir = root / "mlp" / "continuous"

    run_queue(synthetic_prepared.data_dir, root, "--plan", str(fixed_plan))
    job = jobs_by_name(root)["mlp_continuous"]
    assert job["status"] == "done" and job["folds_total"] == job["folds_done"] == 1
    assert list(job["folds"]) == ["T01+T03"]
    fixed_model = job_dir / "fold_T01+T03" / "models" / "mlp_test_T01+T03.pt"
    fixed_mtime = fixed_model.stat().st_mtime_ns
    manifest = json.loads((job_dir / "fold_T01+T03" / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "fixed"
    assert manifest["folds"][0]["validation_groups"] == ["T04"]
    config = json.loads((job_dir / "queue_config.json").read_text(encoding="utf-8"))
    assert config["folds"] == ["T01+T03"]
    assert config["split"] == {"test": ["T01", "T03"], "validation": ["T04"]}
    fixed_signature = config["config_signature"]

    # Same results root, full CV of one Participant: a fold of its own, the Fixed-split fold untouched.
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(fixed_plan), "--full-cv", "--folds", "T01")
    job = jobs_by_name(root)["mlp_continuous"]
    assert list(job["folds"]) == ["T01"] and job["folds"]["T01"]["status"] == "done"
    assert job["config_signature"] != fixed_signature
    assert (job_dir / "fold_T01" / "models" / "mlp_test_T01.pt").is_file()
    assert json.loads((job_dir / "fold_T01" / "split_manifest.json").read_text(encoding="utf-8"))["split"] == "full_cv"
    assert fixed_model.stat().st_mtime_ns == fixed_mtime
    full_cv_mtime = (job_dir / "fold_T01" / "models" / "mlp_test_T01.pt").stat().st_mtime_ns

    # Back to the Fixed split: resumed, nothing retrained, the full-CV fold kept as it is.
    run_queue(synthetic_prepared.data_dir, root, "--plan", str(fixed_plan))
    job = jobs_by_name(root)["mlp_continuous"]
    assert list(job["folds"]) == ["T01+T03"] and job["status"] == "done"
    assert job["config_signature"] == fixed_signature
    assert fixed_model.stat().st_mtime_ns == fixed_mtime
    assert (job_dir / "fold_T01" / "models" / "mlp_test_T01.pt").stat().st_mtime_ns == full_cv_mtime

    # Another dataset version, same results root: the fold is stale and retrained, not reused.
    other = tmp_path / "other_dataset"
    shutil.copytree(synthetic_prepared.data_dir, other)
    metadata = json.loads((other / "metadata.json").read_text(encoding="utf-8"))
    metadata["pipeline_version"] = "prepared_joint_masked_v3.7_test"
    (other / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    run_queue(other, root, "--plan", str(fixed_plan))
    job = jobs_by_name(root)["mlp_continuous"]
    assert job["status"] == "done" and job["config_signature"] != fixed_signature
    assert job["folds"]["T01+T03"]["retrained_from"] == fixed_signature
    assert (root / "stale" / "mlp" / "continuous" / fixed_signature / "fold_T01+T03").is_dir()


def test_a_new_fixed_split_never_reuses_a_fold_of_the_old_one(synthetic_prepared, tmp_path):
    """#25: the split changed (test P13 + P04, validation P07); old folds never count as done."""
    jobs = [{"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"}]
    root = tmp_path / "results"
    job_dir = root / "mlp" / "continuous"

    def run_split(test, validation):
        plan = tmp_path / f"plan_{'+'.join(test)}_{'+'.join(validation)}.json"
        plan.write_text(json.dumps({"models": MLP_ONLY, "jobs": jobs,
                                    "split": {"test": test, "validation": validation}}), encoding="utf-8")
        run_queue(synthetic_prepared.data_dir, root, "--plan", str(plan))
        return jobs_by_name(root)["mlp_continuous"]

    old = run_split(["T01", "T03"], ["T04"])
    old_model = job_dir / "fold_T01+T03" / "models" / "mlp_test_T01+T03.pt"
    old_mtime = old_model.stat().st_mtime_ns

    # A former test Participant becomes the validation one: a new fold, trained from scratch.
    new = run_split(["T03"], ["T01"])
    assert new["status"] == "done" and list(new["folds"]) == ["T03"]
    assert new["config_signature"] != old["config_signature"]
    manifest = json.loads((job_dir / "fold_T03" / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["split"] == "fixed" and manifest["folds"][0]["validation_groups"] == ["T01"]
    assert old_model.stat().st_mtime_ns == old_mtime  # the old fold is left alone, not reused

    # Same test Participants, other validation: same fold name, still retrained (stale), not reused.
    same_name = run_split(["T01", "T03"], ["T05"])
    assert same_name["folds"]["T01+T03"]["retrained_from"] == old["config_signature"]
    assert old_model.stat().st_mtime_ns != old_mtime
    manifest = json.loads((job_dir / "fold_T01+T03" / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["folds"][0]["validation_groups"] == ["T05"]


def test_default_plan_is_the_fixed_split_all_task_set_first():
    from overnight_queue import DEFAULT_PLAN

    assert DEFAULT_PLAN["split"] == {"test": ["P04", "P13"], "validation": ["P07"]}
    assert [job["name"] for job in DEFAULT_PLAN["jobs"]] == [
        "mlp_all", "transformer_all", "sensor_token_all",
        "mlp_continuous", "transformer_continuous", "sensor_token_continuous",
    ]
    sensor_token_all = next(job for job in DEFAULT_PLAN["jobs"] if job["name"] == "sensor_token_all")
    assert (sensor_token_all["model"], sensor_token_all["task_set"]) == ("sensor_token", "all")


def processes_mentioning(text):
    found = []
    for cmdline in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            if text in cmdline.read_bytes().decode(errors="replace"):
                found.append(cmdline.parent.name)
        except OSError:
            pass
    return found


def test_sigterm_during_a_timed_correction_stops_the_engine_and_exits_promptly(synthetic_prepared, tmp_path):
    plan = write_plan(tmp_path / "plan.json",
                      {"broken": {"model_type": "mlp", "engine_args": ["--learning-rate", "inf", "--gradient-clip", "0"]}},
                      [{"name": "broken_continuous", "model": "broken", "task_set": "continuous"}])
    root = tmp_path / "results"
    command = [sys.executable, str(QUEUE), str(synthetic_prepared.data_dir), "--results-root", str(root),
               *FAST, "--plan", str(plan)]
    process = subprocess.Popen(command, cwd=REPO_ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # The first correction attempt runs under the auto-correction time budget (a timer).
        deadline = time.time() + 60
        while not (root / "broken" / "overfit_attempt_1" / "engine.log").exists():
            assert process.poll() is None and time.time() < deadline, "queue ended before the first attempt"
            time.sleep(0.02)
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=30) == 128 + signal.SIGTERM
    finally:
        process.kill()
    assert status(root)["runner"]["state"] == "interrupted"
    assert not processes_mentioning(str(root))  # the engine child is gone too


def test_second_runner_is_refused_while_the_lock_is_held(synthetic_prepared, tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    with (root / "queue.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = run_queue(synthetic_prepared.data_dir, root, check=False)
    assert result.returncode == 4
    assert "another queue runner" in result.stderr
    assert not (root / "queue_status.json").exists()


# The plan of the real v3.7 queue (spec #12, #22) as launched on 2026-10-25: a byte copy of
# results/fixed_split_v3.7/plan_v37.json, with the split of #25 (test P04 + P13, validation P07), the #27 mlp_share job
# the #28 masked body-weight jobs and the #29 decomposed / sum-loss jobs.
LAUNCHED_V37_PLAN = Path(__file__).parent / "data" / "plan_v37.json"


def print_plan(*args):
    result = subprocess.run([sys.executable, str(QUEUE), "--print-plan", *args], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_named_v37_plan_is_exactly_the_launched_plan():
    assert print_plan("--named-plan", "v3.7") == LAUNCHED_V37_PLAN.read_text(encoding="utf-8")
    assert json.loads(print_plan("--named-plan", "v3.7"))["split"] == {"test": ["P04", "P13"], "validation": ["P07"]}
    assert json.loads(print_plan())["jobs"][0]["name"] == "mlp_all"  # the default plan is unchanged


def test_plan_file_and_named_plan_are_exclusive():
    result = subprocess.run([sys.executable, str(QUEUE), "--print-plan", "--named-plan", "v3.7",
                             "--plan", str(LAUNCHED_V37_PLAN)], cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 2 and "not allowed with" in result.stderr


def test_v37_plan_runs_on_the_mini_dataset_and_resumes(tmp_path):
    from synthetic_prepared import write_synthetic_prepared

    subjects = [f"{p}{suffix}" for p in ("T01", "T03", "T04", "T05") for suffix in ("", "b")]
    dataset = write_synthetic_prepared(tmp_path / "v37", foot_height=True,
                                       body_weights_n={s: 600.0 + 10 * i for i, s in enumerate(subjects)})
    metadata = json.loads((dataset.data_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata["pipeline_version"] = "prepared_joint_masked_v3.7_plate_foot_height_airborne"
    (dataset.data_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    plan = json.loads(print_plan("--named-plan", "v3.7"))
    plan["split"] = {"test": ["T01", "T03"], "validation": ["T04"]}  # the mini-dataset's Participants
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan), encoding="utf-8")
    root = tmp_path / "results"

    run_queue(dataset.data_dir, root, "--plan", str(plan_file))
    jobs = jobs_by_name(root)
    assert list(jobs) == ["mlp_continuous", "mlp_bw_continuous", "mlp_all", "mlp_fh_all", "mlp_share_continuous",
                          "mlp_bw_mask_continuous", "mlp_share_mask_continuous",
                          "mlp_share_decomposed_continuous", "mlp_bw_sumloss_continuous",
                          "mlp_share_decomposed_all", "mlp_bw_sumloss_all"]
    assert {job["status"] for job in jobs.values()} == {"done"}
    folds = {name: root / model / task_set / "fold_T01+T03" for name, model, task_set in (
        ("mlp_continuous", "mlp", "continuous"), ("mlp_bw_continuous", "mlp_bw", "continuous"),
        ("mlp_all", "mlp", "all"), ("mlp_fh_all", "mlp_fh", "all"),
        ("mlp_share_continuous", "mlp_share", "continuous"),
        ("mlp_bw_mask_continuous", "mlp_bw_mask", "continuous"),
        ("mlp_share_mask_continuous", "mlp_share_mask", "continuous"),
        ("mlp_share_decomposed_continuous", "mlp_share_decomposed", "continuous"),
        ("mlp_bw_sumloss_continuous", "mlp_bw_sumloss", "continuous"),
        ("mlp_share_decomposed_all", "mlp_share_decomposed", "all"),
        ("mlp_bw_sumloss_all", "mlp_bw_sumloss", "all"))}
    meta = {name: json.loads((fold / "metadata.json").read_text(encoding="utf-8")) for name, fold in folds.items()}
    assert {m["selection_subset"] for m in meta.values()} == {"measured"}
    assert {m["gradnorm"] for m in meta.values()} == {True}
    assert [name for name, m in meta.items() if m["body_weight_input"]] == [
        "mlp_bw_continuous", "mlp_share_continuous", "mlp_bw_mask_continuous", "mlp_share_mask_continuous",
        "mlp_share_decomposed_continuous", "mlp_bw_sumloss_continuous", "mlp_share_decomposed_all",
        "mlp_bw_sumloss_all"]
    assert [name for name, m in meta.items() if m["vertical_share"]] == [
        "mlp_share_continuous", "mlp_share_mask_continuous", "mlp_share_decomposed_continuous",
        "mlp_share_decomposed_all"]
    assert {name: m["vertical_loss"] for name, m in meta.items() if m["vertical_loss"] != "default"} == {
        "mlp_share_decomposed_continuous": "decomposed", "mlp_bw_sumloss_continuous": "sum",
        "mlp_share_decomposed_all": "decomposed", "mlp_bw_sumloss_all": "sum"}
    # The option is in the queue's config signature (through the engine args).
    signatures = {name: job["config_signature"] for name, job in jobs.items()}
    assert len({signatures["mlp_share_continuous"], signatures["mlp_share_decomposed_continuous"]}) == 2
    assert len({signatures["mlp_bw_continuous"], signatures["mlp_bw_sumloss_continuous"]}) == 2
    assert [name for name, m in meta.items() if m.get("body_weight_masking")] == [
        "mlp_bw_mask_continuous", "mlp_share_mask_continuous"]
    assert [name for name, m in meta.items() if m["foot_height_output"]] == ["mlp_fh_all"]

    def checkpoints():
        return {name: next((fold / "models").glob("*.pt")).stat().st_mtime_ns for name, fold in folds.items()}

    trained = checkpoints()
    run_queue(dataset.data_dir, root, "--plan", str(plan_file))  # resumed: nothing retrained
    assert {job["status"] for job in jobs_by_name(root).values()} == {"done"}
    assert checkpoints() == trained


def test_a_body_weight_refresh_makes_only_the_body_weight_folds_stale(tmp_path):
    """Review of spec #12: `--refresh-body-weights` keeps the pipeline_version, so the signature must see it."""
    from synthetic_prepared import write_synthetic_prepared

    subjects = [f"{p}{suffix}" for p in ("T01", "T03", "T04", "T05") for suffix in ("", "b")]
    dataset = write_synthetic_prepared(tmp_path / "data",
                                       body_weights_n={s: 600.0 + 10 * i for i, s in enumerate(subjects)})
    models = {**MLP_ONLY, "mlp_bw": {"model_type": "mlp", "engine_args": ["--layers", "1", "--body-weight-input"]}}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"models": models, "split": {"test": ["T01", "T03"], "validation": ["T04"]},
                                "jobs": [{"name": "mlp_continuous", "model": "mlp", "task_set": "continuous"},
                                         {"name": "mlp_bw_continuous", "model": "mlp_bw",
                                          "task_set": "continuous"}]}), encoding="utf-8")
    root = tmp_path / "results"
    run_queue(dataset.data_dir, root, "--plan", str(plan))
    before = jobs_by_name(root)
    plain = root / "mlp" / "continuous" / "fold_T01+T03" / "models" / "mlp_test_T01+T03.pt"
    plain_mtime = plain.stat().st_mtime_ns

    run_queue(dataset.data_dir, root, "--plan", str(plan))  # same body weights: nothing is stale
    assert {job["config_signature"] for job in jobs_by_name(root).values()} == {
        job["config_signature"] for job in before.values()}

    metadata_path = dataset.data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["body_weights"]["subjects"]["T01"]["body_weight_N"] += 12.5  # a refreshed static value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    run_queue(dataset.data_dir, root, "--plan", str(plan))

    after = jobs_by_name(root)
    assert after["mlp_bw_continuous"]["config_signature"] != before["mlp_bw_continuous"]["config_signature"]
    assert after["mlp_bw_continuous"]["folds"]["T01+T03"]["retrained_from"] == \
        before["mlp_bw_continuous"]["config_signature"]
    assert after["mlp_continuous"]["config_signature"] == before["mlp_continuous"]["config_signature"]
    assert plain.stat().st_mtime_ns == plain_mtime  # the model without body weight is not retrained
