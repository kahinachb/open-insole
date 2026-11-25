"""Characterisation: synchronisation, inference latency, Allan deviation, thermal drift of the Tare (#37)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import characterise_insole as characterise
from synthetic_insole import DECODER, HAS_DECODER, N_PER_COUNT_100LBF, NOMINAL_ZERO, write_recording


def report(offset_l, offset_r, missing=0, max_lr=2.0):
    return {
        "status": "OK",
        "left_insole": {"frequency_hz": 200.0, "duplicates_removed": 1, "packets_after_cleaning": 1000},
        "right_insole": {"frequency_hz": 199.0, "duplicates_removed": 0, "packets_after_cleaning": 1000},
        "quality_control": {"left_right_synchronization": {"max_abs_time_difference_ms": max_lr}},
        "packet_loss_assessment": {"total_missing_packets": missing},
        "offset_estimation": {
            "median_contact_correlation": 0.95,
            "per_side": {
                "L": {"offset_ms": offset_l, "q25_ms": -1.0, "q75_ms": 1.0},
                "R": {"offset_ms": offset_r, "q25_ms": -2.0, "q75_ms": 2.0},
            },
        },
    }


def test_sync_summarises_the_synchronisation_reports(tmp_path):
    manip = tmp_path / "manip"
    for trial, content in (("walk", report(0.0, -1.0, missing=4, max_lr=4.0)), ("squat", report(2.0, 3.0))):
        path = manip / "P04" / trial / "synchronized_v2" / "rapport_synchronisation.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(content), encoding="utf-8")
    out = tmp_path / "sync"
    assert characterise.main(["sync", "--manip-dir", str(manip), "--output-dir", str(out)]) == 0
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["trials"] == 2
    assert summary["packet_loss_rate"] == pytest.approx(4 / 4000)
    assert summary["duplicate_rate"] == pytest.approx(2 / 4000)
    assert summary["left_right_max_abs_ms"]["max"] == 4.0
    assert summary["plate_offset_abs_ms"]["median"] == pytest.approx(1.5)
    assert summary["plate_offset_iqr_ms"]["median"] == pytest.approx(3.0)


def test_latency_is_measured_per_model_window_and_device(tmp_path):
    out = tmp_path / "latency"
    assert characterise.main([
        "latency", "--windows", "21", "--devices", "cpu", "--repeats", "5", "--warmup", "1", "--output-dir", str(out),
    ]) == 0
    table = pd.read_csv(out / "latency.csv")
    assert set(table["model"]) == {"mlp", "patch_transformer"}
    assert (table["median_ms"] > 0).all() and (table["p95_ms"] >= table["median_ms"]).all()


def test_allan_recovers_the_angle_random_walk_of_white_rate_noise(tmp_path):
    rng = np.random.default_rng(0)
    frequency, density = 200.0, 0.01  # deg/s/sqrt(Hz)
    samples = 24000
    rows = []
    for patch in (1, 3, 5, 7):
        noise = rng.normal(scale=density * np.sqrt(frequency), size=(samples, 3))
        rows.append(pd.DataFrame({"ID1": patch, "gyX (°/s)": noise[:, 0], "gyY (°/s)": noise[:, 1], "gyZ (°/s)": noise[:, 2]}))
    path = tmp_path / "gyro" / "session_gyro_x_gauche" / "initial_static.csv"
    path.parent.mkdir(parents=True)
    pd.concat(rows).to_csv(path, index=False)
    out = tmp_path / "allan"
    assert characterise.main(["allan", "--gyro-dir", str(tmp_path / "gyro"), "--output-dir", str(out)]) == 0
    table = pd.read_csv(out / "allan.csv")
    assert len(table) == 12
    assert table["angle_random_walk_deg_per_sqrt_h"].median() == pytest.approx(density * 60.0, rel=0.1)


@pytest.mark.skipif(not HAS_DECODER, reason="acquisition repo absent")
def test_thermal_slope_of_the_tare_is_recovered_from_the_tares(tmp_path):
    get_temperature = characterise.load_module(DECODER, "fonctionsBarometreConversion").getTemp
    manip = tmp_path / "manip"
    for position, raw in enumerate((7_100_000, 7_300_000, 7_500_000)):
        celsius = get_temperature(raw, 1)
        # Heel Gauge 1 of the left insole drifts by 0.5 N per degC; everything else is flat.
        heel = NOMINAL_ZERO + round(0.5 * celsius / N_PER_COUNT_100LBF)

        def counts(patch, index, sample, heel=heel):
            return heel if (patch, index) == (1, 0) else NOMINAL_ZERO

        for foot, name in (("gauche", "tare_fx29_gauche.bin"), ("droite", "tare_fx29_droit.bin")):
            write_recording(manip / f"tare{position}" / name, counts, samples=20, raw_temperature=raw, feet=(foot,))
    out = tmp_path / "thermal"
    assert characterise.main(["thermal", "--manip-dir", str(manip), "--decoder-dir", str(DECODER), "--output-dir", str(out)]) == 0
    slopes = pd.read_csv(out / "thermal_slopes.csv").set_index("channel")
    assert slopes.loc["L_p1_force_1_N", "slope_N_per_C"] == pytest.approx(0.5, abs=0.05)
    assert abs(slopes.loc["L_p3_force_1_N", "slope_N_per_C"]) < 1e-6


def test_clock_drift_is_the_slope_of_the_left_right_time_difference(tmp_path):
    elapsed = np.arange(0.0, 60.0, 0.005)
    # The right clock gains 1 ms per minute on the left one, from a half-period offset.
    np.savez(tmp_path / "trial.npz", target_time_qtm_s_L=elapsed, target_time_qtm_s_R=elapsed + 0.0025 + elapsed / 60.0 * 0.001)
    np.savez(tmp_path / "short.npz", target_time_qtm_s_L=elapsed[:100], target_time_qtm_s_R=elapsed[:100])
    np.savez(tmp_path / "other.npz", X=np.zeros(3))

    table = characterise.clock_drift(tmp_path)

    assert len(table) == 1
    assert table.drift_ms_per_min.iloc[0] == pytest.approx(1.0, rel=1e-6)
    assert table.accumulated_drift_ms.iloc[0] == pytest.approx(elapsed[-1] / 60.0, rel=1e-6)
