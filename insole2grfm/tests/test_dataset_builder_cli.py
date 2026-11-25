"""Seam 1 of spec #12: the prepared-dataset builder CLI on synthetic synchronised trials."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic_prepared import REPO_ROOT, run_training_cli
from synthetic_synchronized import (
    DURATION_S,
    HOVER_MM,
    LIFT_MM,
    STANDING,
    GAUGE_PRELOAD_N,
    Dropout,
    GaugeOverride,
    _timeline,
    write_synthetic_synchronized,
)

BUILDER = REPO_ROOT / "prepare_npy_joint_masked_v3.py"


def run_builder_cli(*args: str, cwd: Path | None = None) -> tuple[int, str]:
    """Run the builder's ``main(argv)`` in-process; returns (exit code, stdout + stderr)."""
    builder = importlib.import_module(BUILDER.stem)
    output = io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output), \
            contextlib.chdir(cwd or Path.cwd()):
        try:
            code = builder.main(list(args))
        except SystemExit as exit_:
            code = exit_.code if isinstance(exit_.code, int) else 1
    return code or 0, output.getvalue()


@pytest.fixture(scope="module")
def synchronized(tmp_path_factory):
    return write_synthetic_synchronized(tmp_path_factory.mktemp("manip"))


@pytest.fixture(scope="module")
def built(synchronized, tmp_path_factory):
    """The builder run with its defaults, except an empty lateral-split list."""
    work = tmp_path_factory.mktemp("build")
    no_split = work / "no_split.json"
    no_split.write_text("[]", encoding="utf-8")
    code, output = run_builder_cli(
        str(synchronized.root), "--lateral-split-config", str(no_split), cwd=work
    )
    assert code == 0, output
    (data_dir,) = (work / "DATA").iterdir()
    return data_dir


def test_fixture_writes_the_real_synchronized_v2_layout(synchronized):
    assert len(synchronized.participants) >= 3
    assert {f"{p}{suffix}" for p in synchronized.participants for suffix in ("", "b")} == set(
        synchronized.subjects
    )
    trial = synchronized.trials[0]
    sync = synchronized.root / trial.subject / trial.trial / "synchronized_v2"
    assert sorted(path.name for path in sync.iterdir()) == [
        "mocap_synchronisee.csv", "plateformes_synchronisees.csv",
        "rapport_synchronisation.json", "semelles_synchronisees.csv",
    ]
    header = (sync / "mocap_synchronisee.csv").read_text(encoding="utf-8").splitlines()[0]
    markers = {column.rsplit("_", 2)[0] for column in header.split(",")[3:]}
    assert markers == {"R_lateral1", "R_medial", "R_toe", "R_heel", "R_lateral2",
                       "L_medialM", "L_heelM", "L_toeM", "L_lateralM"}
    statics = {t.subject for t in synchronized.trials if t.trial == "static"}
    assert statics and statics != set(synchronized.subjects)


def test_builder_writes_a_prepared_dataset_the_training_cli_accepts(synchronized, built, tmp_path):
    metadata = json.loads((built / "metadata.json").read_text(encoding="utf-8"))
    assert sorted(metadata["batches"]) == sorted(
        t.batch for t in synchronized.trials if t.trial != "static"
    )
    for batch in metadata["batches"]:
        assert (built / f"{batch}.npz").is_file()
    assert metadata["explicitly_excluded_batches"] == []

    out = tmp_path / "run"
    run_training_cli(
        built, out, "--model-type", "mlp", "--causal", "--layers", "2",
        "--split-level", "subject_identity", "--only-test-group", "T01",
    )
    predictions = pd.read_csv(out / "predictions.csv")
    assert set(predictions["test_group"]) == {"T01"}
    assert set(predictions["test_batch"]) == {t.batch for t in synchronized.trials
                                               if t.subject in ("T01", "T01b")
                                               and t.trial != "static"}


def test_static_trials_are_not_dataset_trials(synchronized, built):
    metadata = json.loads((built / "metadata.json").read_text(encoding="utf-8"))
    statics = [t.batch for t in synchronized.trials if t.trial == "static"]
    assert statics
    assert not set(statics) & set(metadata["batches"])
    assert not any((built / f"{batch}.npz").exists() for batch in statics)
    assert {r["task_family"] for r in metadata["reports"]} == {"balance", "jump", "walk"}


def test_every_subject_keeps_its_balance_trial_value_as_a_cross_check(synchronized, built):
    # #26: the Body weight itself is per Participant (tests/test_static_body_weight.py).
    metadata = json.loads((built / "metadata.json").read_text(encoding="utf-8"))
    weights = metadata["body_weights"]["balance_trials"]
    assert set(weights) == set(synchronized.subjects)  # T04b has no static trial
    for subject in synchronized.subjects:
        expected = next(t.body_weight_n for t in synchronized.trials if t.subject == subject)
        entry = weights[subject]
        assert entry["body_weight_N"] == pytest.approx(expected, rel=0.005), subject
        assert entry["body_weight_kg"] == pytest.approx(expected / 9.81, rel=0.005)
        assert entry["source"] == "balance_trials"
        assert entry["trials"] == [f"{subject}_balance1"]
        assert entry["trial_medians_N"][f"{subject}_balance1"] == pytest.approx(expected, rel=0.005)
        assert entry["n_samples"] > 100
    # T01b carries the backpack: about 150 N more than T01 in the fixture.
    assert weights["T01b"]["body_weight_N"] - weights["T01"]["body_weight_N"] == \
        pytest.approx(150.0, abs=3.0)


def test_default_build_is_v37_in_its_own_directory(built):
    assert built.name == "prepared_npy_joint_masked_v3.7"
    metadata = json.loads((built / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_version"] == "3.7"
    # The engine refuses --hard-airborne-zero on any "prepared_joint_masked_v3" dataset.
    assert metadata["pipeline_version"].startswith("prepared_joint_masked_v3.7")


def test_version_and_output_directory_can_be_set(synchronized, tmp_path):
    out = tmp_path / "dataset_v3.7b"
    code, output = run_builder_cli(
        str(synchronized.root), "--subjects", "T01", "--trials", "balance1",
        "--lateral-split-config", str(write_empty_list(tmp_path)),
        "--dataset-version", "3.7b", "--output-dir", str(out),
    )
    assert code == 0, output
    metadata = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["dataset_version"] == "3.7b"
    assert metadata["pipeline_version"].startswith("prepared_joint_masked_v3.7b")
    assert metadata["batches"] == ["T01_balance1"]


def test_a_version_outside_v3_is_refused(synchronized, tmp_path):
    code, output = run_builder_cli(
        str(synchronized.root), "--dataset-version", "4.0", "--output-dir", str(tmp_path / "out"),
    )
    assert code != 0
    assert "--dataset-version" in output
    assert not (tmp_path / "out").exists()


def test_a_directory_holding_another_dataset_version_is_never_written(synchronized, tmp_path):
    v36 = tmp_path / "dataset"
    v36.mkdir()
    old_metadata = json.dumps({"pipeline_version": "prepared_joint_masked_v3.6_fused_dynamic_airborne"})
    (v36 / "metadata.json").write_text(old_metadata, encoding="utf-8")
    code, output = run_builder_cli(
        str(synchronized.root), "--subjects", "T01", "--trials", "balance1",
        "--lateral-split-config", str(write_empty_list(tmp_path)), "--output-dir", str(v36),
    )
    assert code != 0
    assert "v3.6" in output
    assert sorted(path.name for path in v36.iterdir()) == ["metadata.json"]
    assert (v36 / "metadata.json").read_text(encoding="utf-8") == old_metadata


MEASURED, AIRBORNE, MASKED = 1, 2, 0
# Per-sample label source (``label_source_L/R``), named in the metadata.
SOURCE_NAMES = {0: "none", 1: "plate", 2: "foot_height_fused", 3: "protocol_window"}
EDGE_MARGIN_S = 0.010  # skip samples next to a pose change (plate filter, frame matching)


def expected_v37_state(pose: str, duration_s: float) -> int:
    """Foot state the v3.7 rules give each pose of the fixture (written from the spec)."""
    if pose in ("plate", "heel_off"):  # the plate detects contact
        return MEASURED
    if pose == "lifted":  # plate empty, Foot height 80 mm, for >= 50 ms
        return AIRBORNE if duration_s >= 0.05 else MASKED
    # "hover" (10 mm, below the threshold) and "floor" (on the ground off the plates)
    return MASKED


EXPECTED_SOURCE = {MEASURED: "plate", AIRBORNE: "foot_height_fused", MASKED: "none"}


def test_labels_follow_the_v37_rules(synchronized, built):
    metadata = json.loads((built / "metadata.json").read_text(encoding="utf-8"))
    families = {report["batch"]: report["task_family"] for report in metadata["reports"]}
    dataset_trials = [t for t in synchronized.trials if t.trial != "static"]
    assert {families[t.batch] for t in dataset_trials} == {"balance", "jump", "walk"}
    seen = set()
    for trial in dataset_trials:
        data = np.load(built / f"{trial.batch}.npz")
        for side_index, side in enumerate(("L", "R")):
            times = data[f"target_time_qtm_s_{side}"]
            segments = trial.segment_at(side, times)
            state = data[f"state_{side}"]
            source = data[f"label_source_{side}"]
            assert source.dtype == np.int8 and set(np.unique(source)) <= set(SOURCE_NAMES)
            targets = slice(6 * side_index, 6 * side_index + 6)
            near_edge = trial.distance_to_pose_change_s(side, times) < EDGE_MARGIN_S
            for index, segment in enumerate(segments):
                if segment is None:
                    continue
                duration = segment.end_s - segment.start_s
                expected = expected_v37_state(segment.pose, duration)
                if near_edge[index] and duration >= 0.05:  # short segments: every sample
                    continue
                assert state[index] == expected, (trial.batch, side, segment, times[index])
                assert SOURCE_NAMES[int(source[index])] == EXPECTED_SOURCE[expected], (
                    trial.batch, side, segment, times[index])
                seen.add((trial.trial, segment.pose, int(expected)))
            # Every sample: Airborne <=> Foot height source, Masked <=> no source.
            assert np.array_equal(state == AIRBORNE, source == 2), (trial.batch, side)
            assert np.array_equal(state == MASKED, source == 0), (trial.batch, side)
            assert np.array_equal(data[f"airborne_label_{side}"], state == AIRBORNE)
            assert np.array_equal(data["target_mask"][:, targets].all(axis=1), state != MASKED)
            assert not data["target_mask"][state == MASKED][:, targets].any()
            assert np.all(data["y"][state == AIRBORNE][:, targets] == 0.0)
            # Measured: the foot's plate load in the foot frame (flat foot: z up). In
            # every Measured pose of the fixture the other foot is also on the ground.
            fz = data["y"][:, 6 * side_index + 2]
            quiet = (state == MEASURED) & (trial.distance_to_pose_change_s(side, times) > 0.05)
            assert np.allclose(fz[quiet], trial.body_weight_n / 2, rtol=0.02), trial.batch
    # The 30 ms walk lift is too short; the 10 mm jump hover (v3.6: Airborne through
    # the gauge rescue) is below the Foot height threshold.
    assert ("jump1", "lifted", AIRBORNE) in seen and ("walk", "lifted", AIRBORNE) in seen
    assert ("walk", "lifted", MASKED) in seen and ("jump1", "hover", MASKED) in seen
    assert ("walk", "floor", MASKED) in seen


def test_metadata_states_the_labelling_rules_and_the_threshold(built):
    metadata = json.loads((built / "metadata.json").read_text(encoding="utf-8"))
    assert "fused_dynamic_airborne" not in metadata["pipeline_version"]
    # Not the first v3.7 build's rules (#24), so queue results never mix the two.
    assert "plate_foot_height_airborne" not in metadata["pipeline_version"]
    rules = metadata["labelling_rules"]
    assert rules["label_source_npz_keys"] == ["label_source_L", "label_source_R"]
    assert {int(code): name for code, name in rules["label_source_codes"].items()} == SOURCE_NAMES
    # The insole is read, as a veto on Airborne only (#24).
    assert rules["state_rules_read_insole_channels"] is True
    assert rules["insole_channel_role"] == "airborne_veto_only"
    airborne = rules["states"]["airborne"]
    assert airborne["label_source"] == "foot_height_fused"
    reliability = airborne["marker_reliability"]
    assert reliability["toe_marker_visible_suffices"] is True
    assert reliability["min_visible_markers_without_toe"] == 3
    assert reliability["calibration"]
    veto = airborne["insole_veto"]
    assert 69.0 < veto["gauge_sum_threshold_n"] < 150.0  # above the preload q95
    assert veto["creates_airborne_labels"] is False
    assert veto["vetoed_state"] == "masked"
    assert veto["calibration"]
    assert metadata["parameters"]["airborne_insole_veto_n"] == veto["gauge_sum_threshold_n"]
    assert metadata["parameters"]["airborne_min_markers_without_toe"] == 3
    assert 15.0 <= airborne["foot_height_threshold_mm"] <= 20.0
    assert airborne["minimum_duration_ms"] == 50.0
    assert airborne["threshold_calibration"]
    assert rules["states"]["measured"]["label_source"] == ["plate", "protocol_window"]
    assert rules["states"]["masked"]["label_source"] == "none"
    assert rules["protocol_windows"]["continuous_tasks_have_no_airborne"] is True
    assert set(rules["removed_v36_rules"]) == {"gauge_sum_hysteresis", "cluster_origin_height"}
    # The v3.6 claim about the gauge gate is gone, and so are its parameters.
    assert "legacy_gauge_gate_diagnostic" not in metadata
    assert not {"unloaded_threshold_n", "loaded_threshold_n", "airborne_height_mm"} & set(
        metadata["parameters"])
    assert metadata["parameters"]["airborne_foot_height_mm"] == airborne["foot_height_threshold_mm"]
    for report in metadata["reports"]:
        counts = report["label_sources"]
        assert set(counts) == {"L", "R"}
        assert set(counts["L"]) == set(SOURCE_NAMES.values())
        assert sum(counts["L"].values()) == report["samples"]


# A single-support trial (#24 fused Airborne rule). The left foot is lifted 80 mm
# from 0.3 to 1.7 s with, in turn: the toe and lateral markers hidden (heel +
# medial left, the walking push-off defect), the heel and medial hidden (toe +
# lateral left), the toe alone hidden (3 markers left), gauges at ~180 N (above
# the insole veto) and gauges at ~75 N (empty-shoe preload range). The right foot
# stands on the floor off the plates with the gauges of an empty shoe.
GAUGE_TRIALS = {
    "balance1": {"L": STANDING, "R": STANDING},
    "lateral_1_appui": {
        "L": _timeline((0.3, "plate"), (1.7, "lifted"), (DURATION_S, "plate")),
        "R": _timeline((DURATION_S, "floor")),
    },
}
GAUGES = {
    "lateral_1_appui": {
        "L": (GaugeOverride(1.3, 1.5, 12.0),  # 15 gauges x 12 N = 180 N on a lifted foot
              GaugeOverride(1.5, 1.65, 5.0)),  # 75 N: above the preload, below the veto
        "R": (GaugeOverride(0.0, DURATION_S, GAUGE_PRELOAD_N),),  # ~45 N on the floor
    },
}
LIFT_DROPOUT = {
    "lateral_1_appui": {"L": (
        Dropout(0.4, 0.6, ("L_toeM", "L_lateralM")),
        Dropout(0.7, 0.9, ("L_heelM", "L_medialM")),
        Dropout(1.0, 1.2, ("L_toeM",)),
    )}
}
# (start, end, expected state of the lifted left foot)
LIFT_PIECES = (
    (0.4, 0.6, MASKED),  # 2 markers, toe hidden: unreliable marker set
    (0.7, 0.9, AIRBORNE),  # 2 markers with the toe
    (1.0, 1.2, AIRBORNE),  # 3 markers without the toe
    (1.3, 1.5, MASKED),  # insole veto
    (1.5, 1.65, AIRBORNE),  # gauges below the veto
)


@pytest.fixture(scope="module")
def gauge_built(tmp_path_factory):
    work = tmp_path_factory.mktemp("gauges")
    synchronized = write_synthetic_synchronized(
        work / "manip", participants=("T06",), trials=GAUGE_TRIALS, dropouts=LIFT_DROPOUT,
        gauges=GAUGES,
    )
    out = work / "dataset"
    code, output = run_builder_cli(
        str(synchronized.root), "--lateral-split-config", str(write_empty_list(work)),
        "--output-dir", str(out),
    )
    assert code == 0, output
    return synchronized, out


def test_airborne_needs_reliable_markers_and_unloaded_gauges(gauge_built):
    synchronized, out = gauge_built
    for subject in synchronized.subjects:
        trial = synchronized.trial(f"{subject}_lateral_1_appui")
        data = np.load(out / f"{trial.batch}.npz")
        times = data["target_time_qtm_s_L"]
        for start, end, expected in LIFT_PIECES:
            inside = (times > start + EDGE_MARGIN_S) & (times < end - EDGE_MARGIN_S)
            assert inside.sum() > 20, (start, end)
            assert np.all(data["state_L"][inside] == expected), (subject, start, end)
            assert np.all(data["foot_height_valid_L"][inside]), (start, end)
            assert np.all(data["label_source_L"][inside] == (2 if expected == AIRBORNE else 0))
        # Fully visible, preload gauges: Airborne between the pieces.
        clear = (times > 0.3 + EDGE_MARGIN_S) & (times < 0.4 - EDGE_MARGIN_S)
        assert clear.sum() > 10 and np.all(data["state_L"][clear] == AIRBORNE)
        # The lightly loaded foot on the floor is never Airborne.
        assert np.all(data["state_R"] == MASKED)
        assert np.all(data["label_source_R"] == 0)


# Foot height (#14): lowest foot marker above the ground, mm, NaN where invalid.
FOOT_HEIGHT_TOLERANCE_MM = 2.0
EXPECTED_FOOT_HEIGHT_MM = {"plate": 0.0, "lifted": LIFT_MM, "hover": HOVER_MM, "floor": 0.0,
                           "heel_off": 0.0}


def test_foot_height_is_zero_flat_and_the_imposed_lift_in_the_air(synchronized, built):
    seen = set()
    for trial in synchronized.trials:
        if trial.trial == "static":  # not a dataset trial
            continue
        data = np.load(built / f"{trial.batch}.npz")
        for side in ("L", "R"):
            times = data[f"target_time_qtm_s_{side}"]
            height = data[f"foot_height_mm_{side}"]
            valid = data[f"foot_height_valid_{side}"]
            assert height.dtype == np.float32 and valid.dtype == bool
            assert valid.all(), (trial.batch, side)  # every marker is visible in this fixture
            quiet = trial.distance_to_pose_change_s(side, times) >= EDGE_MARGIN_S
            poses = trial.pose_at(side, times)
            for pose, expected in EXPECTED_FOOT_HEIGHT_MM.items():
                selected = quiet & (poses == pose)
                if selected.any():
                    assert np.allclose(height[selected], expected,
                                       atol=FOOT_HEIGHT_TOLERANCE_MM), (trial.batch, side, pose)
                    seen.add(pose)
    # "floor": the foot stands on the floor off the plates, ~60 mm below their tops.
    assert seen == {"plate", "lifted", "hover", "floor"}


def test_left_and_right_marker_sets_give_the_same_foot_height_for_mirrored_motion(
    synchronized, built
):
    jumps = [trial for trial in synchronized.trials if trial.trial == "jump1"]
    assert jumps and all(trial.feet["L"] == trial.feet["R"] for trial in jumps)
    for trial in jumps:
        data = np.load(built / f"{trial.batch}.npz")
        quiet = np.ones(len(data["X"]), dtype=bool)
        for side in ("L", "R"):
            times = data[f"target_time_qtm_s_{side}"]
            quiet &= trial.distance_to_pose_change_s(side, times) >= EDGE_MARGIN_S
        assert np.allclose(data["foot_height_mm_L"][quiet], data["foot_height_mm_R"][quiet],
                           atol=1.0), trial.batch


def test_metadata_documents_foot_height(synchronized, built):
    metadata = json.loads((built / "metadata.json").read_text(encoding="utf-8"))
    foot_height = metadata["foot_height"]
    assert foot_height["unit"] == "mm"
    assert foot_height["min_visible_markers"] == 2
    assert "foot_height_mm_L" in foot_height["npz_keys"]["height"]
    assert "foot_height_valid_R" in foot_height["npz_keys"]["valid"]
    assert "platform_geometry" in foot_height["ground_level_source"]["over_a_plate"]
    subjects = foot_height["subjects"]
    assert set(subjects) == set(synchronized.subjects)
    alice = subjects["T01"]
    assert set(alice["marker_offsets_mm"]["L"]) == {"l_medialm", "l_heelm", "l_toem",
                                                    "l_lateralm"}
    assert len(alice["marker_offsets_mm"]["R"]) == 5
    # The fixture's toe marker sits 8 mm above the sole; its floor is at z = -50 mm.
    assert alice["marker_offsets_mm"]["L"]["l_toem"] == pytest.approx(8.0, abs=0.5)
    assert alice["floor_plane_mm"]["z0"] == pytest.approx(-50.0, abs=1.0)
    for report in metadata["reports"]:
        assert report["foot_height_L"]["valid_fraction"] == 1.0


# A walk with heel-off and marker dropouts, on its own small dataset: the right foot
# keeps 1 then 2 of its 5 markers, the left foot 2 then 1 of its 4.
HEEL_OFF_WALK = {
    "balance1": {"L": STANDING, "R": STANDING},
    "walk": {
        "L": _timeline((0.5, "plate"), (0.9, "heel_off"), (1.2, "lifted"), (DURATION_S, "plate")),
        "R": STANDING,
    },
}
DROPOUTS = {
    "walk": {
        "L": (Dropout(1.4, 1.6, ("L_toeM", "L_heelM")),
              Dropout(1.7, 1.9, ("L_toeM", "L_heelM", "L_medialM"))),
        "R": (Dropout(0.2, 0.4, ("R_toe", "R_heel", "R_lateral2", "R_medial")),
              Dropout(0.6, 0.8, ("R_toe", "R_heel", "R_lateral2"))),
    },
}


@pytest.fixture(scope="module")
def heel_off_built(tmp_path_factory):
    work = tmp_path_factory.mktemp("heel_off")
    synchronized = write_synthetic_synchronized(
        work / "manip", participants=("T09",), trials=HEEL_OFF_WALK, dropouts=DROPOUTS
    )
    out = work / "dataset"
    code, output = run_builder_cli(
        str(synchronized.root), "--lateral-split-config", str(write_empty_list(work)),
        "--output-dir", str(out),
    )
    assert code == 0, output
    return synchronized, out


def test_heel_off_with_the_toes_down_does_not_raise_foot_height(heel_off_built):
    synchronized, out = heel_off_built
    for subject in synchronized.subjects:
        trial = synchronized.trial(f"{subject}_walk")
        data = np.load(out / f"{trial.batch}.npz")
        times = data["target_time_qtm_s_L"]
        quiet = trial.distance_to_pose_change_s("L", times) >= EDGE_MARGIN_S
        heel_off = quiet & (trial.pose_at("L", times) == "heel_off")
        assert heel_off.sum() > 50
        assert np.all(np.abs(data["foot_height_mm_L"][heel_off]) <= FOOT_HEIGHT_TOLERANCE_MM)
        lifted = quiet & (trial.pose_at("L", times) == "lifted")
        assert np.allclose(data["foot_height_mm_L"][lifted], LIFT_MM,
                           atol=FOOT_HEIGHT_TOLERANCE_MM)


def test_foot_height_is_invalid_where_fewer_than_two_markers_are_visible(heel_off_built):
    synchronized, out = heel_off_built
    seen = set()
    for subject in synchronized.subjects:
        trial = synchronized.trial(f"{subject}_walk")
        data = np.load(out / f"{trial.batch}.npz")
        for side in ("L", "R"):
            times = data[f"target_time_qtm_s_{side}"]
            # Insole samples map to the nearest mocap frame: skip the dropout edges.
            visible = trial.visible_marker_count(side, times)
            stable = (visible == trial.visible_marker_count(side, times - EDGE_MARGIN_S)) & (
                visible == trial.visible_marker_count(side, times + EDGE_MARGIN_S)
            )
            height = data[f"foot_height_mm_{side}"]
            valid = data[f"foot_height_valid_{side}"]
            assert np.array_equal(valid[stable], visible[stable] >= 2), (trial.batch, side)
            assert np.all(np.isnan(height[~valid])) and np.all(np.isfinite(height[valid]))
            # With 2 markers left (not the lowest one), a flat foot still reads ~0.
            flat = stable & (visible == 2) & (trial.pose_at(side, times) == "plate")
            assert flat.sum() > 20
            assert np.all(np.abs(height[flat]) <= FOOT_HEIGHT_TOLERANCE_MM)
            seen.update(int(count) for count in np.unique(visible[stable]))
    assert {1, 2, 4, 5} <= seen


def write_empty_list(directory: Path) -> Path:
    path = directory / "empty_list.json"
    path.write_text("[]", encoding="utf-8")
    return path


def protocol_labels(interval: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Airborne mask and plate per sample of the left foot, from the labelling function's returned arrays.

    200 Hz, 1 s, double support on the plates except the left foot's 200 ms off its
    plate (0.4-0.6 s) at a Foot height of 40 mm; the recorded protocol window
    (lateral split, #15) is ``interval``.
    """
    builder = importlib.import_module(BUILDER.stem)
    times = np.arange(200) * 0.005
    lifted = (times > 0.4) & (times < 0.6)
    contacts = [
        {"matched_side": "L", "platform": 1, "contact_start_s": 0.0, "contact_end_s": 0.4,
         "spatially_validated": True},
        {"matched_side": "L", "platform": 1, "contact_start_s": 0.6, "contact_end_s": 1.0,
         "spatially_validated": True},
        {"matched_side": "R", "platform": 2, "contact_start_s": 0.0, "contact_end_s": 1.0,
         "spatially_validated": True},
    ]
    heights = {"L": (np.where(lifted, 40.0, 0.0), np.ones(200, dtype=bool)),
               "R": (np.zeros(200), np.ones(200, dtype=bool))}
    platforms, airborne, _, _ = builder.fused_contact_labels(
        {"L": times, "R": times}, {"L": np.zeros((200, 3)), "R": np.zeros((200, 3))}, heights, contacts,
        minimum_correlation=0.3, continuous_double_support=True, continuous_interval=interval,
        airborne_foot_height_mm=15.0, airborne_minimum_duration_s=0.05)
    return airborne["L"], platforms["L"]


def test_protocol_window_never_leaves_an_airborne_fragment_shorter_than_50_ms():
    times = np.arange(200) * 0.005
    # The window ends 25 ms before the foot lands: the 25 ms left over is too short to be Airborne.
    airborne, platform = protocol_labels((0.0, 0.575))
    assert not airborne.any()
    after = (times >= 0.575) & (times < 0.6)
    assert (platform[after] == 0).all()  # neither plate nor Airborne: Masked
    # The window ends 100 ms before landing: that part stays Airborne.
    airborne, _ = protocol_labels((0.0, 0.5))
    assert airborne[(times >= 0.5) & (times < 0.6)].all() and not airborne[times < 0.5].any()
