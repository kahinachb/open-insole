"""Characterisation: gain error of the nominal Gauge sensitivity in static trials (#36)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import characterise_insole as characterise
from synthetic_insole import DECODER, HAS_DECODER, N_PER_COUNT_100LBF, NOMINAL_ZERO, write_recording

pytestmark = pytest.mark.skipif(not HAS_DECODER, reason="acquisition repo absent")


def heel_only(load_n: dict[str, float]):
    """Raw counts with ``load_n[side]`` on the heel Gauge 1 of that insole (100 lbf Gauges), zero elsewhere."""
    def counts(patch, index, sample):
        side = "L" if patch < 9 else "R"
        if patch in (1, 9) and index == 0:
            return NOMINAL_ZERO + round(load_n[side] / N_PER_COUNT_100LBF)
        return NOMINAL_ZERO
    return counts


@pytest.fixture
def manip(tmp_path):
    root = tmp_path / "manip"
    for foot in ("gauche", "droit"):
        write_recording(root / "tare1" / f"tare_fx29_{foot}.bin", lambda patch, index, sample: NOMINAL_ZERO)
    # P04: insoles read 80 % of the plate on the left foot and 90 % on the right foot.
    write_recording(root / "P04" / "static" / "dataSemelle.bin", heel_only({"L": 240.0, "R": 360.0}))
    # T03, T08 and T05 lie on the line insole = 0.9 x plate + 20 N (both feet together).
    write_recording(root / "T03" / "static" / "dataSemelle.bin", heel_only({"L": 290.0, "R": 360.0}))
    write_recording(root / "T08" / "static" / "dataSemelle.bin", heel_only({"L": 380.0, "R": 450.0}))
    write_recording(root / "T05" / "static" / "dataSemelle.bin", heel_only({"L": 340.0, "R": 400.0}))
    plates = tmp_path / "static_plate_fz.csv"
    pd.DataFrame({
        "subject": ["P04", "T03", "T08", "T05"], "trial": ["static"] * 4,
        "force_z_p1_global_N": [400.0, 400.0, 500.0, 450.0], "force_z_p2_global_N": [300.0, 300.0, 400.0, 350.0],
    }).to_csv(plates, index=False)
    assignments = tmp_path / "tare_assignments.json"
    assignments.write_text(json.dumps({"assignments": {name: "tare1" for name in ("P04", "T03", "T08", "T05")}}), encoding="utf-8")
    return root, assignments, plates


def test_static_trials_give_the_total_and_per_foot_gain_against_the_plates(manip, tmp_path):
    root, assignments, plates = manip
    out = tmp_path / "gain"
    assert characterise.main([
        "gain", "--manip-dir", str(root), "--tare-assignments", str(assignments), "--plate-fz", str(plates),
        "--decoder-dir", str(DECODER), "--output-dir", str(out),
    ]) == 0
    table = pd.read_csv(out / "static_gain.csv").set_index("subject", drop=False)
    row = table.loc["P04"]
    assert row["subject"] == "P04" and row["participant"] == "P04"
    assert row["insole_total_N"] == pytest.approx(600.0, abs=0.1)
    assert row["gain_total"] == pytest.approx(600.0 / 700.0, abs=1e-3)
    # The heavier insole (right, 360 N) is paired with the heavier plate (400 N): an inferred assignment.
    assert row["plate_R"] == "p1" and row["plate_assignment"] == "inferred"
    assert row["gain_R"] == pytest.approx(360.0 / 400.0, abs=1e-3)
    assert row["gain_L"] == pytest.approx(240.0 / 300.0, abs=1e-3)
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    # The robust line follows T03, T08 and T05, not P04 (600 N on 700 N, 30 N below it).
    fit = summary["fit_total"]
    assert fit["gain"] == pytest.approx(0.9, abs=0.01) and fit["intercept_N"] == pytest.approx(20.0, abs=5.0)
    assert fit["residual_median_abs_N"] < 5.0
    assert summary["gain_per_participant"]["P04"]["R"] == pytest.approx(0.9, abs=1e-3)
    assert set(pd.read_csv(out / "static_gain_per_foot.csv")["foot"]) == {"L", "R"}
    assert (out / "static_gain.png").exists()
