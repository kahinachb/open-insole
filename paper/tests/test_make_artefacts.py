"""Tests of paper/make_artefacts.py at its two seams: number_macros and latex_table."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_artefacts as ma  # noqa: E402

TARGETS = [f"{c}_{f}" for f in "LR" for c in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]


def rows(run, participant, nrmse, pct_bw, pcc, subset="measured", family="ALL"):
    """One Participant's 12 targets: every target gets the same NRMSE, %BW and PCC."""
    return [
        {
            "run": run,
            "participant": participant,
            "subset": subset,
            "task_family": family,
            "target": t,
            "component": t[:2],
            "foot": t[-1],
            "nrmse_range": nrmse,
            "rmse_pct_bw": pct_bw if t.startswith("F") else float("nan"),
            "pcc_trial_mean": pcc,
        }
        for t in TARGETS
    ]


def write_evaluation(directory: Path, records) -> Path:
    directory.mkdir(parents=True)
    pd.DataFrame(records).to_csv(directory / "per_participant.csv", index=False)
    return directory


def test_score_is_the_mean_over_participants_of_each_participants_mean_over_the_12_targets(tmp_path):
    records = rows("mlp", "P01", 0.06, 2.0, 0.90) + rows("mlp", "P02", 0.08, 3.0, 0.80)
    # Airborne samples and single task families never enter the score.
    records += rows("mlp", "P01", 0.50, 9.0, 0.10, subset="airborne") + rows(
        "mlp", "P02", 0.50, 9.0, 0.1, family="walk"
    )
    evaluation = write_evaluation(tmp_path / "full_configuration", records)

    score = ma.method_score(evaluation, "mlp")

    assert score.nrmse_pct == pytest.approx(7.0)
    assert score.nrmse_sd_pct == pytest.approx(1.4142, abs=1e-4)  # sample SD of 6 and 8
    assert score.force_pct_bw == pytest.approx(2.5)
    assert score.pcc == pytest.approx(0.85)


def test_number_macros_print_the_scores_with_two_decimals(tmp_path):
    evaluation = write_evaluation(
        tmp_path / "full_configuration",
        rows("mlp", "P01", 0.0726, 2.834, 0.88) + rows("mlp", "P02", 0.0726, 2.834, 0.88),
    )

    macros = ma.number_macros({"Main": (evaluation, "mlp")})

    assert r"\newcommand{\nrmseMain}{7.26}" in macros
    assert r"\newcommand{\nrmseSdMain}{0.00}" in macros
    assert r"\newcommand{\forceBwMain}{2.83}" in macros
    assert r"\newcommand{\pccMain}{0.88}" in macros


def test_paired_comparison_counts_the_participants_where_the_reference_wins_and_gives_wilcoxons_p(tmp_path):
    records = []
    for i in range(6):
        records += rows("mlp", f"P0{i + 1}", 0.07, 2.8, 0.9) + rows("rival", f"P0{i + 1}", 0.08 + 0.001 * i, 2.8, 0.9)
    evaluation = write_evaluation(tmp_path / "competitors", records)

    wins, p = ma.paired_comparison(evaluation, "rival", reference="mlp")

    assert wins == 6
    assert p == pytest.approx(2 / 2**6)  # exact two-sided signed-rank p when all 6 differences share a sign


def test_comparison_table_has_one_row_per_method_with_mean_sd_and_the_paired_test_against_the_first(tmp_path):
    records = []
    for i in range(6):
        records += rows("mlp", f"P0{i + 1}", 0.06 + 0.004 * i, 2.5, 0.85) + rows("rival", f"P0{i + 1}", 0.09, 3.0, 0.75)
    evaluation = write_evaluation(tmp_path / "competitors", records)

    table = ma.comparison_table([("Ours", evaluation, "mlp"), ("Rival", evaluation, "rival")])

    assert r"Ours & $7.00 \pm 0.75$ & 2.50 & 0.85 & -- & -- \\" in table
    assert r"Rival & $9.00 \pm 0.00$ & 3.00 & 0.75 & 6/6 & 0.031 \\" in table
    assert table.startswith(r"\begin{tabular}") and table.rstrip().endswith(r"\end{tabular}")


def test_p_values_below_one_in_a_thousand_are_written_as_a_bound():
    assert ma.format_p(0.00006) == "$<0.001$"
    assert ma.format_p(0.4212) == "0.42"
    assert ma.format_p(0.00116) == "0.001"
    assert ma.format_p(0.046) == "0.046"  # never rounded up to the 0.05 threshold


def test_speeds_are_rounded_to_what_the_text_can_say():
    assert ma.format_speed(6138.24) == r"6\,138"
    assert ma.format_speed(143.76) == "144"
    assert ma.format_speed(12.252) == "12.3"
    assert ma.format_speed(0.471) == "0.5"


def test_paired_comparison_can_take_the_reference_from_another_evaluation(tmp_path):
    dedicated = write_evaluation(
        tmp_path / "dedicated", [r for i in range(6) for r in rows("minimal", f"P0{i + 1}", 0.08 + 0.001 * i, 3.0, 0.9)]
    )
    restricted = write_evaluation(
        tmp_path / "restricted", [r for i in range(6) for r in rows("minimal", f"P0{i + 1}", 0.07, 3.0, 0.9)]
    )

    wins, p = ma.paired_comparison(dedicated, "minimal", reference="minimal", reference_dir=restricted)

    assert (wins, p) == (6, pytest.approx(2 / 2**6))


def test_dataset_macros_count_participants_by_sex_body_weight_in_kg_and_trials_per_family(tmp_path):
    evaluation = write_evaluation(tmp_path / "full_configuration", [])
    pd.DataFrame({"subject": ["P01", "P01b", "P02", "P02b"], "body_weight_N": [588.6, 588.6, 784.8, 784.8]}).to_csv(
        evaluation / "body_weight.csv", index=False
    )
    trials = [
        {"run": "mlp", "participant": p, "subset": "measured", "task_family": f, "target": "Fz_L", "n_trials": n}
        for p, f, n in [
            ("P01", "walk", 2),
            ("P02", "walk", 3),
            ("P01", "squat", 1),
            ("P01", "ALL", 3),
            ("P02", "ALL", 3),
        ]
    ]
    pd.DataFrame(trials).to_csv(evaluation / "per_participant.csv", index=False)
    sex = tmp_path / "participant_sex.json"
    sex.write_text('{"P01": "female", "P02": "male"}')
    anthropometry = tmp_path / "participant_anthropometry.json"
    anthropometry.write_text(
        '{"P01": {"age_years": 24, "height_m": 1.69}, "P02": {"age_years": 28, "height_m": 1.81}}'
    )

    macros = ma.dataset_macros(evaluation, sex, anthropometry)

    assert r"\newcommand{\nParticipants}{2}" in macros
    assert r"\newcommand{\nWomen}{1}" in macros
    assert r"\newcommand{\nMen}{1}" in macros
    assert r"\newcommand{\bodyMassKg}{70.0}" in macros  # (60 + 80) / 2 with g = 9.81
    assert r"\newcommand{\bodyMassSdKg}{14.1}" in macros
    assert r"\newcommand{\ageYears}{26.0}" in macros
    assert r"\newcommand{\ageSdYears}{2.8}" in macros
    assert r"\newcommand{\heightM}{1.75}" in macros
    assert r"\newcommand{\heightSdM}{0.08}" in macros
    assert r"\newcommand{\nTrials}{6}" in macros
    assert r"\newcommand{\nTrialsWalk}{5}" in macros
    assert r"\newcommand{\nTrialsSquat}{1}" in macros


def test_characterisation_macros_read_the_summaries_of_characterise_insole(tmp_path):
    import json

    summaries = {
        "sync": {
            "trials": 395,
            "frequency_hz": {"median": 199.418},
            "packet_loss_rate": 2.48e-5,
            "duplicate_rate": 0.00117,
            "left_right_max_abs_ms": {"max": 4.4999},
            "plate_offset_abs_ms": {"median": 10.0},
            "plate_offset_iqr_ms": {"median": 1.0},
            "median_contact_correlation": 0.99637,
        },
        "thermal": {
            "temperature_C": {"min": 21.24, "max": 29.88},
            "slope_abs_N_per_C": {"median": 0.0221, "max": 0.081},
        },
    }
    for name, summary in summaries.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(summary))
    pd.DataFrame({"channel": ["a", "b"], "temperature_range_C": [8.0, 6.0], "slope_N_per_C": [0.05, -0.081]}).to_csv(
        tmp_path / "thermal" / "thermal_slopes.csv", index=False
    )

    macros = ma.characterisation_macros(tmp_path)

    assert r"\newcommand{\sampleRateHz}{199.4}" in macros
    assert r"\newcommand{\packetLossPpm}{25}" in macros
    assert r"\newcommand{\duplicatePct}{0.12}" in macros
    assert r"\newcommand{\pairingMaxMs}{4.5}" in macros
    assert r"\newcommand{\contactCorrelation}{0.996}" in macros
    assert r"\newcommand{\thermalSlopeMaxNPerC}{0.081}" in macros
    # Largest drift a single Gauge saw over its own temperature range: 0.081 x 6 = 0.49 N, not 0.081 x 8.64.
    assert r"\newcommand{\thermalDriftMaxN}{0.5}" in macros


def six_participants(run, nrmse_of, pct_bw=3.0, pcc=0.9):
    """Rows of P01..P06, Participant i scoring ``nrmse_of(i)``."""
    return [r for i in range(6) for r in rows(run, f"P0{i + 1}", nrmse_of(i), pct_bw, pcc)]


def test_main_table_gives_the_speed_and_marks_a_pending_row(tmp_path):
    records = six_participants("mlp", lambda i: 0.06 + 0.004 * i, 2.5, 0.85) + six_participants("rival", lambda i: 0.09)
    evaluation = write_evaluation(tmp_path / "competitors", records)

    table = ma.comparison_table(
        [("Ours", evaluation, "mlp"), ("Ours, fewer inputs", None, None), ("Rival", evaluation, "rival")],
        speeds={"Ours": r"6\,138", "Rival": None},
    )

    assert r"Ours & $7.00 \pm 0.75$ & 2.50 & 0.85 & -- & -- & 6\,138 \\" in table
    assert r"Rival & $9.00 \pm 0.00$ & 3.00 & 0.90 & 6/6 & 0.031 & n/a \\" in table
    assert r"Ours, fewer inputs & \multicolumn{6}{c}{\textit{running}} \\" in table


def test_configuration_table_gives_the_change_from_full_and_the_paired_test(tmp_path):
    evaluation = write_evaluation(
        tmp_path / "configurations",
        six_participants("full", lambda i: 0.07) + six_participants("gauges_only", lambda i: 0.077 + 0.001 * i),
    )
    rows_ = [ma.ConfigurationRow("All sensors", 78, evaluation, "full"),
             ma.ConfigurationRow("Gauges only", 30, evaluation, "gauges_only")]

    table = ma.configuration_table(rows_, reference=(evaluation, "full"))

    assert r"All sensors & 78 & $7.00 \pm 0.00$ & -- & 0.90 & -- \\" in table
    # mean 7.95 %, +13.6 % from 7.00; worse on 6/6, exact p = 2/64
    assert r"Gauges only & 30 & $7.95 \pm 0.19$ & +13.6\,\% & 0.90 & 0.031 \\" in table


def test_robustness_table_puts_both_models_side_by_side_each_against_its_own_full_configuration(tmp_path):
    without = write_evaluation(
        tmp_path / "without",
        six_participants("full", lambda i: 0.06) + six_participants("gauges_only", lambda i: 0.12 + 0.001 * i),
    )
    with_ = write_evaluation(
        tmp_path / "with",
        six_participants("full", lambda i: 0.07) + six_participants("gauges_only", lambda i: 0.077),
    )

    table = ma.robustness_table(
        [ma.RobustnessRow("All sensors", 78, "full", with_), ma.RobustnessRow("Gauges only", 30, "gauges_only", with_)],
        without=without,
    )

    # NRMSE with sensor dropout, then without it
    assert r"All sensors & 78 & 7.00 & 6.00 \\" in table
    assert r"Gauges only & 30 & 7.70 & 12.25 \\" in table
    assert "vs all" not in table


def test_configuration_and_robustness_macros_carry_the_numbers_the_text_cites(tmp_path):
    without = write_evaluation(
        tmp_path / "without",
        six_participants("full", lambda i: 0.06) + six_participants("gauges_only", lambda i: 0.12 + 0.001 * i),
    )
    with_ = write_evaluation(
        tmp_path / "with", six_participants("full", lambda i: 0.07) + six_participants("gauges_only", lambda i: 0.077)
    )
    full, gauges = ma.RobustnessRow("All", 78, "full", with_), ma.RobustnessRow("Gauges", 30, "gauges_only", with_)

    configuration = ma.configuration_macros("Gauges", ma.ConfigurationRow("Gauges", 30, with_, "gauges_only"), (with_, "full"))
    robust = ma.robustness_macros("Gauges", gauges, full, without)

    assert r"\newcommand{\changeCfgGauges}{+10.0\,\%}" in configuration
    assert r"\newcommand{\worseCfgGauges}{6}" in configuration
    assert r"\newcommand{\changeNoDropoutGauges}{+104.2\,\%}" in robust
    assert r"\newcommand{\betterDropoutGauges}{6}" in robust


def target_rows(run, participant, subset, family, values):
    """Rows of the 12 targets; ``values`` maps a component to (rmse, pct_bw, nrmse, pcc, mean_error)."""
    out = []
    for t in TARGETS:
        rmse, pct_bw, nrmse, pcc, bias = values[t[:2]]
        out.append({"run": run, "participant": participant, "subset": subset, "task_family": family, "target": t,
                    "component": t[:2], "foot": t[-1], "rmse": rmse, "rmse_pct_bw": pct_bw if t[0] == "F" else float("nan"),
                    "nrmse_range": nrmse, "pcc_trial_mean": pcc, "mean_error": bias, "n_trials": 4})
    return out


COMPONENT_VALUES = {"Fx": (10.0, 1.5, 0.08, 0.8, 0.0), "Fy": (12.0, 1.8, 0.07, 0.8, 0.0), "Fz": (40.0, 5.5, 0.05, 0.98, 1.0),
                    "Mx": (3.0, 0, 0.09, 0.7, 0.0), "My": (6.0, 0, 0.06, 0.9, 0.0), "Mz": (1.5, 0, 0.10, 0.6, 0.0)}


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = pd.Series([6.0, 7.0, 8.0, 7.5, 6.5, 7.2])
    low, high = ma.bootstrap_ci(values)
    assert low < values.mean() < high
    assert (low, high) == ma.bootstrap_ci(values)


def test_component_table_gives_each_component_per_foot_with_the_airborne_residual(tmp_path):
    records = []
    for i in range(6):
        records += target_rows("mlp", f"P0{i + 1}", "measured", "ALL", COMPONENT_VALUES)
        records += target_rows("mlp", f"P0{i + 1}", "airborne", "ALL",
                               {c: (2.0 + 10 * (c == "Fz"), 0.5, float("nan"), float("nan"), 3.0 * (c == "Fz")) for c in COMPONENT_VALUES})
    evaluation = write_evaluation(tmp_path / "full", records)

    table = ma.component_table(evaluation, "mlp")

    # Per foot, left then right: RMSE, NRMSE, PCC on measured samples; RMSE and mean error on airborne ones.
    assert r"$F_z$ (N) & 40.0 & 5.00 & 0.98 & 40.0 & 5.00 & 0.98 & 12.0 & $+3.0$ & 12.0 & $+3.0$ \\" in table
    assert r"$M_z$ (N$\cdot$m) & 1.50 & 10.00 & 0.60 & 1.50 & 10.00 & 0.60 & 2.00 & $+0.00$ & 2.00 & $+0.00$ \\" in table
    assert "Left" in table and "Right" in table


def test_family_table_has_one_row_per_task_family_and_the_pooled_row_last_without_trial_counts(tmp_path):
    records = []
    for i in range(6):
        for family, scale in (("walk", 1.0), ("squat", 0.5), ("ALL", 0.8)):
            # A family's own NRMSE is inflated by its narrower range (x 1.5 here); the table rescales by the ALL range.
            values = {c: (v[0] * scale / 0.8, v[1] * scale, v[2] * scale * 1.5, v[3], v[4]) for c, v in COMPONENT_VALUES.items()}
            if family == "ALL":
                values = {c: (v[0], v[1] * scale, v[2] * scale, v[3], v[4]) for c, v in COMPONENT_VALUES.items()}
            records += target_rows("mlp", f"P0{i + 1}", "measured", family, values)
    evaluation = write_evaluation(tmp_path / "full", records)

    table = ma.family_table(evaluation, "mlp", {"squat": "Squat", "walk": "Walking"})

    lines = table.splitlines()
    squat = next(l for l in lines if l.startswith("Squat"))
    assert squat == r"Squat & $3.75 \pm 0.00$ & [3.75, 3.75] & 2.75 & 0.80 \\"
    assert lines.index(squat) < lines.index(next(l for l in lines if l.startswith("Walking")))
    assert "Trials" not in table
    assert next(l for l in lines if l.startswith("All tasks")).startswith(r"All tasks & $6.00")


def test_loading_table_normalises_both_conditions_by_the_all_task_range_and_tests_the_backpack(tmp_path):
    evaluation = write_evaluation(
        tmp_path / "full", [r for i in range(6) for r in target_rows("mlp", f"P0{i + 1}", "measured", "ALL", COMPONENT_VALUES)]
    )
    pd.DataFrame({"subject": [f"P0{i + 1}" for i in range(6)], "body_weight_N": 800.0}).to_csv(
        evaluation / "body_weight.csv", index=False
    )
    # ALL row: range = rmse / nrmse, e.g. Fz 40 / 0.05 = 800 N; unloaded errs as ALL, backpack 1.2x more
    loading = pd.DataFrame(
        [{"participant": f"P0{i + 1}", "condition": cond, "target": t, "rmse": COMPONENT_VALUES[t[:2]][0] * k, "n": 10}
         for i in range(6) for cond, k in (("unloaded", 1.0), ("backpack", 1.2)) for t in TARGETS]
    )

    table = ma.loading_table(loading, evaluation, "mlp")

    assert r"Unloaded & $7.50 \pm 0.00$ & 40.0 & 5.00 & -- \\" in table
    assert r"Backpack & $9.00 \pm 0.00$ & 48.0 & 6.00 & 0.031 \\" in table


def test_transition_table_averages_the_feet_per_participant(tmp_path):
    pd.DataFrame({"subject": ["P01", "P02"], "body_weight_N": [500.0, 1000.0]}).to_csv(tmp_path / "bw.csv", index=False)
    transitions = pd.DataFrame(
        [{"participant": p, "foot": f, "event": e, "rmse_N": r, "mean_error_N": b, "n_samples": 100, "n_events": n}
         for p in ("P01", "P02") for f in "LR"
         for e, r, b, n in (("stance", 30.0, 0.0, 0), ("contact", 60.0, 10.0, 5), ("lift_off", 50.0, 20.0, 5), ("air", 25.0, 12.0, 0))]
    )

    table = ma.transition_table(transitions, tmp_path / "bw.csv")

    # 60 N of 500 and 1000 N: 12 and 6 %BW, whose mean is 9 and sample SD 4.24
    assert r"Contact, $\pm$50\,ms & $60.0 \pm 0.0$ & $9.00 \pm 4.24$ & $+10.0 \pm 0.0$ \\" in table
    assert table.index("Stance") < table.index("Contact") < table.index("Lift-off") < table.index("In the air")


def test_imu_calibration_macros_summarise_both_insoles(tmp_path):
    import json

    def write(session, kind, patches, metrics, model):
        directory = tmp_path / session / f"calibration_{kind}"
        directory.mkdir(parents=True)
        name = "accelerometer_calibration.json" if kind == "acc" else "gyroscope_calibration.json"
        body = {"model": model, "patches": {str(i): {"scale_factors": {"x": s, "y": 1.0, "z": 1.0},
                                                     "misalignment_parameters_rad": {"a": m}, "metrics": metrics}
                                            for i, (s, m) in enumerate(patches)}}
        (directory / name).write_text(json.dumps(body))

    write("session_a_gauche", "acc", [(0.996, 0.007)], {"raw_norm_rmse_g": 0.0185, "calibrated_fit_norm_rmse_g": 0.0003}, "acc")
    write("session_b_droit", "acc", [(0.999, 0.001)], {"raw_norm_rmse_g": 0.0125, "calibrated_fit_norm_rmse_g": 0.0001}, "acc")
    write("session_gyro_c_gauche", "gyro", [(0.98, 0.0086)], {"raw_angle_rmse_deg": 0.87, "calibrated_fit_angle_rmse_deg": 0.17}, "gyro")
    write("session_gyro_d_droit", "gyro", [(1.02, 0.001)], {"raw_angle_rmse_deg": 1.96, "calibrated_fit_angle_rmse_deg": 0.79}, "gyro")

    macros = ma.imu_calibration_macros(tmp_path)

    assert r"\newcommand{\accScaleDevMaxPct}{0.4}" in macros  # |0.996 - 1|
    assert r"\newcommand{\accRawNormMg}{12.5--18.5}" in macros
    assert r"\newcommand{\accCalNormMg}{0.1--0.3}" in macros
    assert r"\newcommand{\gyroScaleDevMaxPct}{2.0}" in macros
    assert r"\newcommand{\gyroMisalignMaxDeg}{0.49}" in macros
    assert r"\newcommand{\gyroRawAngleDeg}{0.87--1.96}" in macros
    assert r"\newcommand{\gyroCalAngleDeg}{0.17--0.79}" in macros


import sensor_study as ss  # noqa: E402


def component_rows(run, participant, values):
    """One Participant's 12 targets; ``values`` maps a component to its NRMSE (fraction), same on both feet."""
    return [
        {"run": run, "participant": participant, "subset": "measured", "task_family": "ALL", "target": f"{c}_{f}",
         "component": c, "foot": f, "nrmse_range": values[c], "rmse_pct_bw": 1.0, "pcc_trial_mean": 0.9}
        for f in "LR" for c in ss.COMPONENTS
    ]


def test_component_change_reads_each_component_against_the_full_configuration(tmp_path):
    full = dict.fromkeys(ss.COMPONENTS, 0.10)
    imus = {**full, "Fz": 0.25}
    records = []
    for i, participant in enumerate(["P01", "P02", "P03", "P04", "P05", "P06"]):
        records += component_rows("full", participant, {c: v + i / 1000 for c, v in full.items()})
        records += component_rows("imus_only", participant, {c: v + i / 1000 for c, v in imus.items()})
    evaluation = write_evaluation(tmp_path / "study", records)

    change, worse, _ = ss.component_change(evaluation, "imus_only")

    assert change["Fz"] == pytest.approx(100 * (0.2525 / 0.1025 - 1))
    assert change["Fx"] == pytest.approx(0.0)
    assert worse["Fz"] == 6


def test_load_shares_sum_the_load_cells_of_each_patch_over_both_feet():
    load_cells = pd.DataFrame(
        [{"participant": "P01", "foot": f, "channel": f"{f}{p}", "patch": p, "mean_force_n": n}
         for f in "LR" for p, n in (("heel", 40), ("midfoot", 10), ("forefoot", 30), ("toes", 20))]
    )

    shares = ss.load_shares(load_cells)

    assert shares.loc["P01", "heel"] == pytest.approx(40.0)
    assert shares.loc["P01", "midfoot"] == pytest.approx(10.0)


def test_cost_of_one_insole_includes_its_communication_module_and_the_minimal_one_drops_seven_sensors():
    bom = pd.DataFrame(
        [{"unit_price_eur": 20.0, "quantity_per_insole": 15, "role": "load_cell"},
         {"unit_price_eur": 2.0, "quantity_per_insole": 4, "role": "imu"},
         {"unit_price_eur": 12.0, "quantity_per_insole": 1, "role": "patch"},
         {"unit_price_eur": 80.0, "quantity_per_insole": 1, "role": "com_module"}]
    )

    macros = ss.cost_macros(bom)

    usd = ss.EUR_TO_USD
    assert rf"\newcommand{{\costInsole}}{{{round(400 * usd, -1):.0f}}}" in macros  # 300 + 8 + 12 + 80 EUR
    assert rf"\newcommand{{\costLoadCells}}{{{round(300 * usd, -1):.0f}}}" in macros
    assert rf"\newcommand{{\costInsoleMinimal}}{{{round(314 * usd, -1):.0f}}}" in macros  # 400 - 80 - 6 EUR
    assert r"\newcommand{\costSavingMinimal}{21\,\%}" in macros  # 86 / 400 = 21.5 %


def test_robustness_row_can_read_the_model_without_dropout_from_its_own_dir(tmp_path):
    p = ["P01", "P02", "P03", "P04", "P05", "P06"]
    without = write_evaluation(tmp_path / "without", sum((rows("full", x, 0.07, 2, 0.9) for x in p), []))
    lost = write_evaluation(
        tmp_path / "lost_without",
        sum((rows("full", x, 0.07, 2, 0.9) + rows("no_heel", x, 0.14 + i / 1000, 2, 0.9) for i, x in enumerate(p)), []),
    )
    with_dropout = write_evaluation(
        tmp_path / "with",
        sum((rows("full", x, 0.07, 2, 0.9) + rows("no_heel", x, 0.08 + i / 1000, 2, 0.9) for i, x in enumerate(p)), []),
    )
    full = ma.RobustnessRow("All sensors", 78, "full", with_dropout)
    row = ma.RobustnessRow("Heel patch lost", 58, "no_heel", with_dropout, lost)

    r = ma.robustness(row, full, without)

    assert r.change_without_pct == pytest.approx(100 * (0.1425 / 0.07 - 1))
    assert r.dropout_better == 6


def test_per_target_component_macros_compare_the_per_target_mlp_with_our_mlp_and_ke2kenet(tmp_path):
    # Three Participants; the per-target MLP beats our MLP everywhere, Ke2KeNet beats it on two.
    records = []
    for participant, (main, per_target, mahmud) in {
        "P01": (0.08, 0.06, 0.05),
        "P02": (0.09, 0.07, 0.06),
        "P03": (0.10, 0.08, 0.09),
    }.items():
        records += rows("mlp", participant, main, 2.0, 0.9)
        records += rows(ma.PER_TARGET_RUN, participant, per_target, 2.0, 0.9)
        records += rows("mahmud_raw", participant, mahmud, 2.0, 0.9)
    evaluation = write_evaluation(tmp_path / "per_target", records)

    macros = ma.per_target_component_macros(evaluation)

    assert r"\newcommand{\nrmseCompMyMain}{9.00}" in macros
    assert r"\newcommand{\nrmseCompMyPerTarget}{7.00}" in macros
    assert r"\newcommand{\nrmseCompMyMahmudRaw}{6.67}" in macros
    assert r"\newcommand{\changeCompMyPerTarget}{-22.2\,\%}" in macros
    assert r"\newcommand{\winsCompMyPerTarget}{3}" in macros
    assert r"\newcommand{\winsCompMyMahmudOverPerTarget}{2}" in macros
    assert r"\newcommand{\winsCompAllPerTarget}{3}" in macros
    assert macros.count(r"\newcommand") == 7 * 8


def test_results_table_gives_each_target_per_method_and_bolds_the_lowest_nrmse(tmp_path):
    records = []
    for participant in ("P01", "P02"):
        records += rows("mlp", participant, 0.10, 2.0, 0.9) + rows("rf", participant, 0.095, 2.0, 0.8)
    evaluation = write_evaluation(tmp_path / "e", [dict(r, rmse=3.0) for r in records])

    table = ma.results_table([ma.ResultsColumn("MLP (ours)", evaluation, "mlp", 0.2, True, ours=True),
                              ma.ResultsColumn("B net", evaluation, "rf", None, False)])

    lines = table.splitlines()
    assert sum(line.startswith("$") for line in lines) == 12  # 6 components x 2 feet
    # bold: lowest among the real-time methods (A); underlined: lowest overall (B, offline)
    assert r"$F_{x,L}$ (N) & 3.0 & \textbf{10.00} & 0.90 & 3.0 & \underline{9.50} & 0.80 \\" in lines
    assert r"\underline{9.50} $\pm$ 0.00" in table
    assert lines[0] == r"\begin{tabular}{l" + ma.OURS_SHADE * 3 + "ccc}"  # our columns shaded
    shaded = r"\multicolumn{3}{" + ma.OURS_SHADE + "}"
    # gaps against the first method (ours): B is 5 % below it
    assert r"NRMSE vs MLP (ours) & " + shaded + r"{--} & \multicolumn{3}{c}{-5.0\,\%" in table
    assert r"Inference time per frame, 12 targets (ms) & " + shaded + r"{\textbf{0.20}} & \multicolumn{3}{c}{n/a}" in table
    assert r"Real time & " + shaded + r"{yes} & \multicolumn{3}{c}{no}" in table


def test_airborne_table_gives_the_rmse_and_mean_error_of_each_foot(tmp_path):
    records = [dict(r, rmse=rmse, mean_error=0.5) for participant, rmse in (("P01", 1.0), ("P02", 3.0))
               for r in rows("mlp", participant, 0.1, 1.0, 0.9, subset="airborne")]
    evaluation = write_evaluation(tmp_path / "e", records)

    table = ma.airborne_table(evaluation, "mlp")

    # mean and sample SD over the Participants of 1 and 3
    assert r"$F_z$ (N) & $2.0 \pm 1.4$ & $+0.5 \pm 0.0$ & $2.0 \pm 1.4$ & $+0.5 \pm 0.0$ \\" in table
    assert r"$M_y$ (N$\cdot$m) & $2.00 \pm 1.41$ & $+0.50 \pm 0.00$ & $2.00 \pm 1.41$ & $+0.50 \pm 0.00$ \\" in table


def test_measured_share_macros_give_each_family_its_percentage_of_the_measured_samples(tmp_path):
    records = []
    for family, n in (("ALL", 100), ("walk", 4), ("balance", 96)):
        records += [dict(r, n=n) for r in rows("mlp", "P01", 0.1, 1.0, 0.9, family=family)]
    evaluation = write_evaluation(tmp_path / "e", records)

    macros = ma.measured_share_macros(evaluation, "mlp")

    assert r"\newcommand{\shareMeasuredWalk}{4.0}" in macros
    assert r"\newcommand{\shareMeasuredBalance}{96.0}" in macros


def test_parameter_macros_give_millions_and_the_ratio_to_our_mlp(tmp_path):
    path = tmp_path / "parameters.csv"
    pd.DataFrame({"method": ["mlp", "mlp_per_target", "mahmud_gauges", "mahmud_raw"],
                  "parameters": [564_748, 6_743_566, 685_151_496, 685_990_656]}).to_csv(path, index=False)

    macros = ma.parameter_macros(path)

    assert r"\newcommand{\paramsMain}{0.56}" in macros
    assert r"\newcommand{\paramsPerTarget}{6.74}" in macros
    assert r"\newcommand{\paramsMahmudGauges}{685}" in macros
    assert r"\newcommand{\ratioParamsMahmudGauges}{1\,200}" in macros
    assert r"\newcommand{\ratioParamsPerTarget}{12}" in macros


def test_parts_bought_once_per_pair_are_reported_apart():
    bom = pd.DataFrame(
        {"unit_price_eur": [20.0, 2.0, 3.0, 6.0], "quantity_per_insole": [15, 4, 4, 1],
         "role": ["load_cell", "imu", "patch", "com_module"]}
    )
    pair = pd.DataFrame({"unit_price_eur": [6.0, 15.0], "quantity_per_pair": [1, 1]})

    macros = ss.cost_macros(bom, pair)

    assert rf"\newcommand{{\costPerPair}}{{{round(21 * ss.EUR_TO_USD, -1):.0f}}}" in macros  # reception module + filament
    # Two insoles (15 x 20 + 4 x 2 + 4 x 3 + 6 = 326 EUR each) and the parts bought once per pair, rounded first.
    usd = ss.EUR_TO_USD
    assert rf"\newcommand{{\costPair}}{{{2 * round(326 * usd, -1) + round(21 * usd, -1):.0f}}}" in macros


def test_duration_macros_give_median_trial_total_hours_and_million_samples(tmp_path):
    durations = tmp_path / "trial_durations.csv"
    pd.DataFrame({"duration_s": [60.0, 90.0, 3450.0], "n_samples": [12000, 18000, 690000]}).to_csv(
        durations, index=False
    )

    macros = ma.duration_macros(durations)

    assert r"\newcommand{\trialDurationMedianS}{90}" in macros
    assert r"\newcommand{\datasetHours}{1.0}" in macros
    assert r"\newcommand{\nSamplesMillion}{0.7}" in macros


def test_sabbah_total_macros_compare_the_sum_of_both_feet_on_the_continuous_tasks(tmp_path):
    records = []
    for i, participant in enumerate(("P01", "P02", "P03", "P04", "P05", "P06")):
        for run, error in (("mlp", 2.0), ("sabbah_loadsol", 3.0), ("sabbah_total", 4.0 + 0.1 * i)):
            for family in ("balance", "squat", "lateral_2supports", "hulahoop", "walk"):
                for component in ("Fx", "Fy", "Fz", "Mz"):
                    # walk and the moments never enter the comparison
                    value = error if family != "walk" and component != "Mz" else 99.0
                    records.append({"run": run, "participant": participant, "subset": "both_feet_sum",
                                    "task_family": family, "component": component, "foot": "sum",
                                    "rmse_pct_bw": value})
    evaluation = write_evaluation(tmp_path / "competitors_sabbah_total", records)

    macros = ma.sabbah_total_macros(evaluation)

    assert r"\newcommand{\bwSumMain}{2.00}" in macros
    assert r"\newcommand{\bwSumSabbahLoadsol}{3.00}" in macros
    assert r"\newcommand{\bwSumSabbahTotal}{4.25}" in macros
    assert r"\newcommand{\winsSumSabbahTotal}{6}" in macros
    assert r"\newcommand{\pSumSabbahTotal}{0.031}" in macros  # exact two-sided p for 6/6
