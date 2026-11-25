"""Characterisation of the Tare: spread across Tares and residual offset on feet-raised controls (#35)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import characterise_insole as characterise
from synthetic_insole import DECODER, HAS_DECODER, NOMINAL_ZERO, write_recording

pytestmark = pytest.mark.skipif(not HAS_DECODER, reason="acquisition repo absent")


def tare_of(tare: str):
    # Each Tare shifts every Gauge's reading by a known amount; L_p1_force_1 differs most between Tares.
    shift = {"tare1": 0, "P04": 20, "tare2": 40}[tare]
    return lambda patch, index, sample: NOMINAL_ZERO + 10 * patch + index + (shift if (patch, index) == (1, 0) else 0)


@pytest.fixture
def manip(tmp_path):
    root = tmp_path / "manip"
    for tare in ("tare1", "P04", "tare2"):
        for foot in ("gauche", "droit"):
            write_recording(root / tare / f"tare_fx29_{foot}.bin", tare_of(tare))
    # P04's feet-raised control reads 30 counts below her Tare on L_p1_force_1 (a negative offset).
    p04_tare = tare_of("P04")
    write_recording(
        root / "P04" / "dataSemelle.bin",
        lambda patch, index, sample: p04_tare(patch, index, sample) - (30 if (patch, index) == (1, 0) else 0),
    )
    assignments = tmp_path / "tare_assignments.json"
    assignments.write_text(json.dumps({"assignments": {"P04": "P04", "P13": "tare1"}}), encoding="utf-8")
    return root, assignments


def run(manip, tmp_path, *extra):
    root, assignments = manip
    out = tmp_path / "out"
    code = characterise.main([
        "tare", "--manip-dir", str(root), "--tare-assignments", str(assignments),
        "--decoder-dir", str(DECODER), "--output-dir", str(out), *extra,
    ])
    assert code == 0
    return out


def test_every_tare_is_decoded_to_per_gauge_readings_and_their_spread(manip, tmp_path):
    out = run(manip, tmp_path)
    readings = pd.read_csv(out / "tare_readings.csv")
    assert set(readings["tare"]) == {"tare1", "P04", "tare2"}
    assert len(readings[readings["tare"] == "tare1"]) == 30  # 15 Gauges per insole
    row = readings[(readings["tare"] == "tare2") & (readings["channel"] == "L_p1_force_1_N")].iloc[0]
    assert row["tare_counts"] == 1050
    spread = pd.read_csv(out / "tare_spread.csv").set_index("channel")
    assert spread.loc["L_p1_force_1_N", "range_counts"] == 40
    assert spread.loc["L_p3_force_1_N", "range_counts"] == 0
    # FX29 heel Gauges: 100 lbf over a 14000-count span.
    assert spread.loc["L_p1_force_1_N", "range_N"] == pytest.approx(40 / 14000 * 100 * 4.44822)


def test_feet_raised_controls_show_the_residual_offset_including_negative_ones(manip, tmp_path):
    out = run(manip, tmp_path)
    residual = pd.read_csv(out / "control_residuals.csv")
    row = residual[(residual["subject"] == "P04") & (residual["channel"] == "L_p1_force_1_N")].iloc[0]
    assert row["tare"] == "P04"
    assert row["residual_counts"] == -30
    assert row["residual_N"] < 0.0
    others = residual[(residual["subject"] == "P04") & (residual["channel"] != "L_p1_force_1_N")]
    assert (others["residual_counts"] == 0).all()


def test_a_tare_comparison_reports_the_load_shift_per_gauge_and_per_foot(manip, tmp_path):
    out = run(manip, tmp_path, "--compare", "P13:tare2")
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    (comparison,) = summary["comparisons"]
    assert comparison["subject"] == "P13"
    assert comparison["assigned_tare"] == "tare1" and comparison["other_tare"] == "tare2"
    assert comparison["foot_sum_shift_N"]["L"] == pytest.approx(40 / 14000 * 100 * 4.44822)
    assert comparison["foot_sum_shift_N"]["R"] == 0.0
    assert (out / "tare_spread.png").exists()


def test_a_malformed_comparison_is_rejected_by_the_parser(manip, tmp_path):
    with pytest.raises(SystemExit) as exit_:
        run(manip, tmp_path, "--compare", "P13")
    assert exit_.value.code == 2
