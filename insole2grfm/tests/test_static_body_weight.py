"""Ticket #26: Body weight from the unloaded static C3D trials, shared by both Subjects.

Seams: the extraction CLI (``extract_static_plate_fz.py`` in the ``c3d-io`` env, on
small C3D files, and its output file), the builder CLI (``metadata.json``
``body_weights``), the builder's metadata-only refresh of an existing prepared
dataset, and the evaluation CLI's ``body_weight.csv`` / %BW.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_dataset_builder_cli import run_builder_cli
from synthetic_c3d import C3D_IO_PYTHON, write_c3d
from synthetic_prepared import REPO_ROOT, write_synthetic_prepared
from synthetic_synchronized import body_weight_n, write_synthetic_synchronized
from test_evaluate_grfm import evaluate, wave, write_run
from test_evaluate_grfm import trial as predicted_trial

G = 9.81
RATE_HZ = 1000.0


def standing_plate_fz(weight_n: float, seconds: float = 6.0, step_on_s: float = 1.5,
                      seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Empty plates, a step-on with one foot then the other, quiet standing, a step-off."""
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE_HZ)
    t = np.arange(n) / RATE_HZ
    p1 = np.zeros(n)
    p2 = np.zeros(n)
    first = t >= step_on_s * 0.5
    both = (t >= step_on_s) & (t < seconds - 1.0)
    p1[first & (t < step_on_s)] = weight_n * 1.1  # all the weight on one foot, moving
    p1[both] = weight_n * 0.48 + 3.0 * np.sin(2 * np.pi * 0.4 * t[both])
    p2[both] = weight_n * 0.52 - 3.0 * np.sin(2 * np.pi * 0.4 * t[both])
    p1 += rng.normal(0.0, 0.5, n)
    p2 += rng.normal(0.0, 0.5, n)
    return p1, p2


def write_static_csv(path: Path, trials: dict[tuple[str, str], float]) -> Path:
    """The extraction CLI's file, for (Subject, static trial) -> standing weight in N."""
    frames = []
    for index, ((subject, trial), weight) in enumerate(sorted(trials.items())):
        p1, p2 = standing_plate_fz(weight, seed=index)
        frames.append(pd.DataFrame({
            "subject": subject, "trial": trial, "c3d_file": f"{subject}/{trial}/{trial}.c3d",
            "time_s": np.arange(len(p1)) / RATE_HZ,
            "force_z_p1_global_N": p1, "force_z_p2_global_N": p2,
        }))
    pd.concat(frames).to_csv(path, index=False)
    return path


# --- extraction CLI ---------------------------------------------------------------------

@pytest.mark.skipif(not C3D_IO_PYTHON.is_file(), reason="needs the c3d-io env (ezc3d)")
def test_extraction_cli_writes_both_plates_of_every_static_c3d(tmp_path):
    manip = tmp_path / "manip"
    for subject, trial in [("T01", "static"), ("T01b", "static2"), ("T03", "static"), ("T03", "walk1")]:
        (manip / subject / trial).mkdir(parents=True)
    (manip / "T03" / "static" / "dataSemelle.bin").write_bytes(b"")  # a static dir without C3D
    ramp = [float(i) for i in range(10)]
    write_c3d(manip / "T01" / "static" / "static.c3d", RATE_HZ, [[1.0] * 10, [2.0] * 10])
    write_c3d(manip / "T01" / "static" / "static_gapf.c3d", RATE_HZ, [ramp, [300.0] * 10])  # preferred
    write_c3d(manip / "T01b" / "static2" / "static2.c3d", RATE_HZ, [[400.0] * 10, [410.0] * 10])
    write_c3d(manip / "T03" / "walk1" / "walk1.c3d", RATE_HZ, [[5.0] * 10, [5.0] * 10])  # not a static trial
    out = tmp_path / "static_plate_fz.csv"
    result = subprocess.run([str(C3D_IO_PYTHON), str(REPO_ROOT / "extract_static_plate_fz.py"), str(manip),
                             "--output", str(out)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    table = pd.read_csv(out)
    assert list(table.columns) == ["subject", "trial", "c3d_file", "time_s",
                                   "force_z_p1_global_N", "force_z_p2_global_N"]
    assert set(table["subject"]) == {"T01", "T01b"}
    alice = table[table["subject"] == "T01"]
    assert alice["c3d_file"].unique().tolist() == ["T01/static/static_gapf.c3d"]
    assert alice["trial"].unique().tolist() == ["static"]
    assert alice["time_s"].tolist() == pytest.approx([i / RATE_HZ for i in range(10)])
    assert alice["force_z_p1_global_N"].tolist() == pytest.approx(ramp)
    assert alice["force_z_p2_global_N"].tolist() == pytest.approx([300.0] * 10)
    assert "SANS C3D T03/static" in result.stdout


# --- builder ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synchronized(tmp_path_factory):
    # Participants T01 (0), T03 (1), T04 (2); T04b has no static trial.
    return write_synthetic_synchronized(tmp_path_factory.mktemp("manip_bw"))


def sex_file(path: Path) -> Path:
    path.write_text(json.dumps({"T01": "female", "T03": "male", "T04": "female"}),
                    encoding="utf-8")
    return path


def build(synchronized, work: Path, *extra: str) -> dict:
    no_split = work / "no_split.json"
    no_split.write_text("[]", encoding="utf-8")
    out = work / "data"
    code, output = run_builder_cli(
        str(synchronized.root), "--lateral-split-config", str(no_split),
        "--output-dir", str(out), "--participant-sex", str(sex_file(work / "sex.json")), *extra,
        "--trials", "balance1", cwd=work,
    )
    assert code == 0, output
    return json.loads((out / "metadata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def built(synchronized, tmp_path_factory):
    """T01: unloaded static; T03: only the loaded static; T04: no static at all."""
    work = tmp_path_factory.mktemp("build_bw")
    statics = write_static_csv(work / "static_plate_fz.csv", {
        ("T01", "static"): 612.0,
        ("T01b", "static2"): 770.0,
        ("T03b", "static"): 830.0,
    })
    return build(synchronized, work, "--static-plate-fz", str(statics))


def test_unloaded_static_trial_gives_the_body_weight_of_both_subjects(built):
    weights = built["body_weights"]
    alice = weights["participants"]["T01"]
    assert alice["source"] == "unloaded_static"
    assert alice["body_weight_N"] == pytest.approx(612.0, abs=1.0)
    assert alice["body_weight_kg"] == pytest.approx(612.0 / G, abs=0.1)
    assert alice["static_trial"] == "T01_static"
    assert alice["subjects"] == ["T01", "T01b"]
    assert alice["quiet_window"]["duration_s"] > 2.5
    for subject in ("T01", "T01b"):  # the loaded Subject shares it
        entry = weights["subjects"][subject]
        assert entry["body_weight_N"] == alice["body_weight_N"]
        assert entry["source"] == "unloaded_static"
        assert entry["participant"] == "T01"


def test_without_an_unloaded_static_the_loaded_static_minus_the_backpack_is_used(built):
    bob = built["body_weights"]["participants"]["T03"]
    assert bob["source"] == "loaded_static_minus_backpack"
    assert bob["static_trial"] == "T03b_static"
    assert bob["backpack_kg"] == 6.0  # male
    assert bob["body_weight_N"] == pytest.approx(830.0 - 6.0 * G, abs=1.0)
    assert built["body_weights"]["subjects"]["T03b"]["body_weight_N"] == bob["body_weight_N"]


def test_without_any_static_the_unloaded_balance_trials_are_used(built):
    chloe = built["body_weights"]["participants"]["T04"]
    assert chloe["source"] == "unloaded_balance_trials"
    assert chloe["body_weight_N"] == pytest.approx(body_weight_n(2, loaded=False), rel=0.005)
    assert chloe["static_trial"] is None
    assert built["body_weights"]["subjects"]["T04b"]["body_weight_N"] == chloe["body_weight_N"]


def test_the_balance_values_of_every_subject_stay_as_a_cross_check(built):
    balance = built["body_weights"]["balance_trials"]
    assert set(balance) == {"T01", "T01b", "T03", "T03b", "T04", "T04b"}
    assert balance["T03b"]["body_weight_N"] == pytest.approx(body_weight_n(1, True), rel=0.005)
    assert balance["T03b"]["trials"] == ["T03b_balance1"]


def test_an_unusable_unloaded_static_falls_back_and_says_why(synchronized, tmp_path):
    statics = write_static_csv(tmp_path / "static_plate_fz.csv", {
        ("T01", "static"): 612.0, ("T01b", "static2"): 770.0,
    })
    table = pd.read_csv(statics)
    unloaded = table["subject"] == "T01"
    table.loc[unloaded, "force_z_p2_global_N"] = 0.0  # one foot never on its plate
    table.to_csv(statics, index=False)
    alice = build(synchronized, tmp_path, "--static-plate-fz", str(statics))[
        "body_weights"]["participants"]["T01"]
    assert alice["source"] == "loaded_static_minus_backpack"
    assert alice["backpack_kg"] == 3.0  # female
    assert alice["body_weight_N"] == pytest.approx(770.0 - 3.0 * G, abs=1.0)
    assert "T01_static" in alice["rejected_statics"]


def test_without_static_plate_data_every_participant_uses_its_balance_trials(synchronized, tmp_path):
    weights = build(synchronized, tmp_path)["body_weights"]
    assert {p["source"] for p in weights["participants"].values()} == {"unloaded_balance_trials"}
    assert weights["subjects"]["T01b"]["body_weight_N"] == pytest.approx(
        body_weight_n(0, False), rel=0.005)


# --- metadata-only refresh --------------------------------------------------------------

def test_refresh_rewrites_only_the_body_weights_of_an_existing_dataset(tmp_path):
    data = write_synthetic_prepared(tmp_path / "data")
    metadata_path = data.data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    balance = {s: 700.0 + i for i, s in enumerate(data.subjects)}
    metadata["body_weights"] = {"units": "N", "subjects": {
        s: {"body_weight_N": v, "body_weight_kg": v / G, "source": "balance_trials",
            "n_samples": 50, "trials": [f"{s}_balance"], "trial_medians_N": {}}
        for s, v in balance.items()}}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    npz_before = {p.name: p.read_bytes() for p in data.data_dir.glob("*.npz")}
    first, second = data.participants[:2]
    statics = write_static_csv(tmp_path / "static.csv", {(first, "static"): 640.0})
    sex = tmp_path / "sex.json"
    sex.write_text(json.dumps({p: "male" for p in data.participants}), encoding="utf-8")

    for _ in range(2):  # idempotent: the balance values survive a second refresh
        code, output = run_builder_cli(
            "--refresh-body-weights", str(data.data_dir), "--static-plate-fz", str(statics),
            "--participant-sex", str(sex), cwd=tmp_path)
        assert code == 0, output
        refreshed = json.loads(metadata_path.read_text(encoding="utf-8"))
        weights = refreshed["body_weights"]
        assert weights["participants"][first]["source"] == "unloaded_static"
        assert weights["subjects"][f"{first}b"]["body_weight_N"] == pytest.approx(640.0, abs=1.0)
        assert weights["participants"][second]["source"] == "unloaded_balance_trials"
        assert weights["subjects"][f"{second}b"]["body_weight_N"] == balance[second]
        assert weights["balance_trials"][f"{second}b"]["body_weight_N"] == balance[f"{second}b"]
    assert {k: v for k, v in refreshed.items() if k != "body_weights"} == \
        {k: v for k, v in metadata.items() if k != "body_weights"}
    assert {p.name: p.read_bytes() for p in data.data_dir.glob("*.npz")} == npz_before


def test_evaluation_reads_the_participant_body_weight_for_both_subjects(tmp_path):
    data = write_synthetic_prepared(tmp_path / "data")
    first = data.participants[0]
    statics = write_static_csv(tmp_path / "static.csv", {(first, "static"): 640.0})
    code, output = run_builder_cli("--refresh-body-weights", str(data.data_dir),
                                   "--static-plate-fz", str(statics), cwd=tmp_path)
    assert code == 0, output
    # A run testing both of the Participant's Subjects, a 6.4 N error on every force.
    run = write_run(tmp_path / "run", [predicted_trial(f"{subject}_balance", first, "balance", wave, offset=6.4)
                                       for subject in (first, f"{first}b")])
    out = evaluate(tmp_path, {"mlp": run}, "--dataset-dir", str(data.data_dir))
    weights = pd.read_csv(out / "body_weight.csv").set_index("subject")
    assert weights.loc[first, "body_weight_N"] == weights.loc[f"{first}b", "body_weight_N"]
    assert weights.loc[f"{first}b", "body_weight_N"] == pytest.approx(640.0, abs=1.0)
    assert weights.loc[f"{first}b", "source"] == "dataset:unloaded_static"
    rows = pd.read_csv(out / "per_participant.csv")
    forces = rows[(rows.subset == "measured") & (rows.task_family == "ALL") & rows.target.str.startswith("F")]
    assert forces.rmse_pct_bw.to_numpy() == pytest.approx(1.0, abs=0.01)  # 6.4 N of 640 N
