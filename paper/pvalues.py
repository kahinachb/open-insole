"""Holm adjustment of the paper's p values, within each family of comparisons.

Every p value of the paper is a paired two-sided Wilcoxon test over the Participants. The paper reports
them Holm-adjusted within the family of comparisons that answer one question (``FAMILIES``), so that each
adjusted p can be compared with 0.05 directly. ``make_artefacts.py`` makes the artefacts twice: the first
pass records every raw p under its macro name, ``P_VALUES.adjust()`` adjusts them, and the second pass
writes the adjusted values. The raw values stay in the released evaluation tables.

This module holds the shared state on its own, so that ``make_artefacts`` run as a script and
``sensor_study`` (which imports ``make_artefacts``) see the same instance.
"""

from __future__ import annotations

import math

COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")

FAMILIES = {
    # Our MLP against the other estimators of Section V-B and the supplementary material.
    "methods": [
        "pPerTarget", "pSabbahRaw", "pSabbahLoadsol", "pMahmudRaw", "pMahmudGauges",
        "pPerTargetMahmudRaw", "pSumSabbahTotal",
    ],
    # Per-target MLP against our MLP, and against Ke2KeNet (78 ch.), component by component.
    "per-target components": [f"pComp{c}PerTarget" for c in COMPONENTS],
    "per-target vs Ke2KeNet components": [f"pComp{c}PerTargetMahmud" for c in COMPONENTS],
    # Sensor-dropout model against the model trained without it, one test per row of the robustness table.
    "sensor dropout": [
        f"pDropout{name}" for name in (
            "All", "NoHeel", "NoMidfoot", "NoForefoot", "NoToes", "GaugesAcc", "GaugesGyro", "Gauges", "Imus",
            "ThirtyFourCh", "TwentyEightCh",
        )
    ],
    # Each sensor configuration against the full insole.
    "configurations": [
        f"pCfg{name}" for name in (
            "GaugesAcc", "GaugesGyro", "Gauges", "Imus",
            "DropoutThirtyFourCh", "DedicatedThirtyFourCh", "DropoutTwentyEightCh", "DedicatedTwentyEightCh",
        )
    ],
    # Which modality carries which component: each component without the load cells, then without the IMUs.
    "modalities": [f"pComp{c}Without{removed}" for removed in ("LoadCells", "Imus") for c in COMPONENTS],
    # Choice of the minimal configuration.
    "minimal configuration": [
        "pThirtyFourOverTwentyEight", "pDropoutTwentyEight", "pDropoutThirtyFour", "pWithoutMidfootLoadCells",
    ],
}
# Single comparisons, each its own question: the choice of architecture, and the load carriage.
SINGLE = ("pTransformer", "pBackpack")
# The same comparison computed in two places: one name is kept.
ALIASES = {
    "pCompAllPerTarget": "pPerTarget",
    "pCompAllPerTargetMahmud": "pPerTargetMahmudRaw",
    "pDedicatedTwentyEight": "pCfgDedicatedTwentyEightCh",
    "pDedicatedThirtyFour": "pCfgDedicatedThirtyFourCh",
}


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p values, in the order given: the k-th smallest of m is multiplied by m - k + 1,
    then made non-decreasing and capped at 1."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [math.nan] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


class PValues:
    """Records the raw p values by macro name (first pass), then returns their adjusted values."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.raw: dict[str, list[float]] = {}
        self.adjusted: dict[str, float] | None = None

    def __call__(self, name: str, p: float) -> float:
        name = ALIASES.get(name, name)
        if self.adjusted is None:
            self.raw.setdefault(name, []).append(float(p))
            return p
        return self.adjusted[name]

    def adjust(self) -> dict[str, float]:
        families = {**FAMILIES, **{name: [name] for name in SINGLE}}
        known = {name for names in families.values() for name in names}
        if unknown := sorted(set(self.raw) - known):
            raise ValueError(f"p values outside any family: {unknown}")
        raw = {}
        for name, values in self.raw.items():
            if not all(math.isclose(v, values[0], rel_tol=1e-9, abs_tol=1e-12) for v in values):
                raise ValueError(f"{name} computed twice with different values: {values}")
            raw[name] = values[0]
        self.adjusted = {}
        for names in families.values():
            present = [name for name in names if name in raw]
            self.adjusted |= dict(zip(present, holm([raw[name] for name in present])))
        return self.adjusted


P_VALUES = PValues()
