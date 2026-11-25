"""Holm adjustment of the paper's p values (pvalues.py)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pvalues  # noqa: E402


def test_holm_multiplies_by_the_remaining_count_keeps_the_order_and_caps_at_one():
    # Sorted 0.01, 0.02, 0.04, 0.5: x4, x3, x2, x1, then non-decreasing and capped.
    assert pvalues.holm([0.04, 0.01, 0.5, 0.02]) == pytest.approx([0.08, 0.04, 0.5, 0.06])
    assert pvalues.holm([0.3, 0.4]) == pytest.approx([0.6, 0.6])
    assert pvalues.holm([0.9, 0.8]) == pytest.approx([1.0, 1.0])


def test_a_single_comparison_is_not_adjusted():
    assert pvalues.holm([0.041]) == pytest.approx([0.041])


def test_two_passes_return_raw_then_adjusted_values_within_each_family():
    p = pvalues.PValues()
    names = pvalues.FAMILIES["per-target vs Ke2KeNet components"]
    raw = [0.85, 0.56, 0.018, 0.28, 0.022, 0.80]
    assert [p(name, value) for name, value in zip(names, raw)] == raw
    p("pTransformer", 0.041)
    p.adjust()
    assert p("pCompFzPerTargetMahmud", 0.018) == pytest.approx(0.108)
    assert p("pCompMyPerTargetMahmud", 0.022) == pytest.approx(0.11)
    assert p("pTransformer", 0.041) == pytest.approx(0.041)


def test_aliases_share_one_value_and_must_agree():
    p = pvalues.PValues()
    p("pPerTarget", 0.0004)
    p("pCompAllPerTarget", 0.0004)
    p.adjust()
    assert p("pCompAllPerTarget", 0.0004) == p("pPerTarget", 0.0004)

    p = pvalues.PValues()
    p("pPerTarget", 0.0004)
    p("pCompAllPerTarget", 0.02)
    with pytest.raises(ValueError, match="computed twice"):
        p.adjust()


def test_a_p_value_outside_every_family_is_refused():
    p = pvalues.PValues()
    p("pSomethingNew", 0.01)
    with pytest.raises(ValueError, match="outside any family"):
        p.adjust()
