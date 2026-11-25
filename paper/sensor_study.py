"""Numbers and the supplementary table of the Sensor study that the configuration tables do not give.

Inputs, under ``paper/evaluation/sensor_study``:
- ``no_dropout/`` and ``sensor_dropout/``: ``evaluate_grfm.py`` scores of the MLP trained without and with
  Sensor dropout, under lost-patch and other configurations (``evaluate_sensor_configurations.py``);
- ``load_cells.csv``: mean force of each load cell while its foot is loaded (``scripts/load_cell_means.py``);
- ``flat_feet.json``: the Participants with flat feet, and the typical foot shown next to them.
The bill of materials is ``paper/bom/bom_insole.csv`` (a copy of the one of the hardware repository).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from make_artefacts import format_change, format_p, method_score, paired_comparison, read_table
from pvalues import P_VALUES

COMPONENTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
PATCHES = ("heel", "midfoot", "forefoot", "toes")
LOST_PATCH_RUNS = {f"no_{p}": p for p in PATCHES}
MINIMAL_KEPT = {"load_cell": ("heel", "forefoot", "toes"), "imu": ("midfoot",)}  # 28 channels


def component_scores(evaluation_dir: Path, run: str) -> pd.DataFrame:
    """NRMSE (%) per Participant (rows) and component (columns), both feet averaged."""
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == run) & (table.subset == "measured") & (table.task_family == "ALL")
                 & table.foot.isin(["L", "R"])]
    return rows.groupby(["participant", "component"]).nrmse_range.mean().unstack() * 100


def component_change(evaluation_dir: Path, run: str, reference: str = "full") -> tuple[pd.Series, pd.Series, pd.Series]:
    """Per component: change of the mean (%) from ``reference``, Participants worse, and Wilcoxon p."""
    from scipy.stats import wilcoxon

    a, b = component_scores(evaluation_dir, reference), component_scores(evaluation_dir, run)
    change = 100 * (b.mean() / a.mean() - 1)
    worse = (b > a).sum()
    p = pd.Series({c: wilcoxon(a[c], b[c]).pvalue for c in a.columns})
    return change, worse, p


def modality_component_table(evaluation_dir: Path) -> str:
    """Supplementary table: NRMSE per component with all sensors, load cells only and IMUs only."""
    full = component_scores(evaluation_dir, "full").mean()
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r" & All sensors & \multicolumn{2}{c}{Load cells only} & \multicolumn{2}{c}{IMUs only} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"Component & NRMSE (\%) & NRMSE (\%) & vs all & NRMSE (\%) & vs all \\",
        r"\midrule",
    ]
    cells = {run: component_scores(evaluation_dir, run).mean() for run in ("gauges_only", "imus_only")}
    for c in COMPONENTS:
        g, i = cells["gauges_only"][c], cells["imus_only"][c]
        lines.append(
            rf"${c[0]}_{c[1]}$ & {full[c]:.2f} & {g:.2f} & {format_change(100 * (g / full[c] - 1))} "
            rf"& {i:.2f} & {format_change(100 * (i / full[c] - 1))} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def macro(name: str, value: str) -> str:
    return rf"\newcommand{{\{name}}}{{{value}}}" + "\n"


def pct(value: float, digits: int = 0) -> str:
    return rf"{value:.{digits}f}\,\%"


def modality_macros(dropout: Path) -> str:
    """Which modality carries which component: the change when the other is removed."""
    out = ""
    for run, removed in (("imus_only", "LoadCells"), ("gauges_only", "Imus")):
        change, worse, p = component_change(dropout, run)
        for c in COMPONENTS:
            out += macro(f"changeComp{c}Without{removed}", rf"{change[c]:+.0f}\,\%")
            out += macro(f"worseComp{c}Without{removed}", f"{worse[c]}")
            out += macro(f"pComp{c}Without{removed}", format_p(P_VALUES(f"pComp{c}Without{removed}", p[c])))
    return out


def lost_patch_macros(no_dropout: Path, dropout: Path) -> str:
    """Mean and range over the four lost patches of each model's change from its own full score."""
    out = ""
    for directory, name in ((no_dropout, "NoDropout"), (dropout, "Dropout")):
        full = method_score(directory, "full").nrmse_pct
        changes = {p: 100 * (method_score(directory, run).nrmse_pct / full - 1) for run, p in LOST_PATCH_RUNS.items()}
        series = pd.Series(changes)
        out += macro(f"changeLostPatch{name}", rf"{series.mean():+.0f}\,\%")
        out += macro(f"changeLostPatchMin{name}", rf"{series.min():+.0f}")
        out += macro(f"changeLostPatchMax{name}", rf"{series.max():+.0f}\,\%")
        out += macro(f"worstLostPatch{name}", series.idxmax())
    better, worst_p = [], 0.0
    for run in LOST_PATCH_RUNS:
        wins, p = paired_comparison(no_dropout, run, reference=run, reference_dir=dropout)
        better.append(wins)
        p = P_VALUES("pDropout" + "".join(part.capitalize() for part in run.split("_")), p)  # e.g. pDropoutNoHeel
        worst_p = max(worst_p, p)
    out += macro("betterDropoutLostPatchMin", f"{min(better)}")
    out += macro("pDropoutLostPatchMax", format_p(worst_p))
    for directory, name in ((no_dropout, "NoDropout"), (dropout, "Dropout")):
        for run, label in (("full", "All"), ("no_heel", "NoHeel")):
            out += macro(f"fz{name}{label}", f"{component_scores(directory, run).mean()['Fz']:.1f}")
    return out


def load_shares(load_cells: pd.DataFrame) -> pd.DataFrame:
    """Share (%) of each Participant's insole load carried by each patch, both feet pooled."""
    per_patch = load_cells.groupby(["participant", "patch"]).mean_force_n.sum().unstack()
    return per_patch.div(per_patch.sum(axis=1), axis=0) * 100


def medial_midfoot_force(load_cells: pd.DataFrame, layout: dict) -> pd.Series:
    """Mean force (N) of the medial midfoot load cells per Participant (left foot: y < 0; right: y > 0)."""
    medial = {
        s["channel"]
        for side, foot in layout["feet"].items()
        for s in foot["sensors"]
        if s["patch"] == "midfoot" and (s["y_mm"] < 0 if side == "L" else s["y_mm"] > 0)
    }
    return load_cells[load_cells.channel.isin(medial)].groupby("participant").mean_force_n.mean()


# Prices are bought in EUR and reported in USD at this rate (ECB reference and market quotes, early
# November 2026, rounded). Change it here only.
EUR_TO_USD = 1.12


def cost_macros(bom: pd.DataFrame, pair: pd.DataFrame | None = None) -> str:
    """Cost of one insole with its communication module, full and minimal (28 channels: 4 midfoot load cells
    and 3 IMUs fewer), in USD (``EUR_TO_USD``) rounded to 10, and, with ``pair``, the parts bought once per
    pair (reception module, 3D-printing filament) and the cost of a complete pair."""
    lines = bom.unit_price_eur * bom.quantity_per_insole
    insole = lines.sum()
    price = bom.groupby("role").unit_price_eur.first()
    minimal = insole - 4 * price["load_cell"] - 3 * price["imu"]
    usd = EUR_TO_USD
    text = (
        macro("eurToUsd", f"{usd:.2f}")
        + macro("costInsole", f"{round(insole * usd, -1):.0f}")
        + macro("costLoadCells", f"{round(lines[bom.role == 'load_cell'].sum() * usd, -1):.0f}")
        + macro("costInsoleMinimal", f"{round(minimal * usd, -1):.0f}")
        + macro("costSavingMinimal", pct(100 * (1 - minimal / insole)))
    )
    if pair is not None:
        per_pair = (pair.unit_price_eur * pair.quantity_per_pair).sum()
        text += macro("costPerPair", f"{round(per_pair * usd, -1):.0f}")
        # A complete pair: two insoles with their communication modules, and the parts bought once per pair.
        # Summed from the rounded values, so that the text adds up.
        text += macro("costPair", f"{2 * round(insole * usd, -1) + round(per_pair * usd, -1):.0f}")
    return text


def minimal_macros(load_cells: pd.DataFrame, dedicated: Path) -> str:
    shares = load_shares(load_cells)
    kept = shares[list(MINIMAL_KEPT["load_cell"])].sum(axis=1)
    wins, p = paired_comparison(dedicated, "minimal_28ch", reference="minimal_34ch")
    return (
        macro("shareKeptPatches", pct(kept.mean()))
        + macro("shareMidfoot", pct(shares["midfoot"].mean()))
        + macro("shareMidfootMin", f"{shares['midfoot'].min():.0f}")
        + macro("shareMidfootMax", pct(shares["midfoot"].max()))
        + macro("winsThirtyFourOverTwentyEight", f"{wins}")
        + macro("pThirtyFourOverTwentyEight", format_p(P_VALUES("pThirtyFourOverTwentyEight", p)))
    )


NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def minimal_spread_macros(dropout: Path, target_pct: float = 10.0) -> str:
    """How the 28-channel configuration varies between Participants: how many exceed the relative target
    (as a word below ten), and the largest increase with its Participant."""
    full = method_score(dropout, "full").per_participant_pct
    minimal = method_score(dropout, "minimal_28ch").per_participant_pct.reindex(full.index)
    change = 100 * (minimal / full - 1)
    over = int((change > target_pct).sum())
    return (
        macro("minimalOverTargetCount", NUMBER_WORDS[over] if over < len(NUMBER_WORDS) else f"{over}")
        + macro("minimalMaxChangePct", f"{change.max():.1f}")
        + macro("minimalMaxChangeCode", str(change.idxmax()))
    )


def flat_foot_macros(dropout: Path, minimal_dir: Path, load_cells: pd.DataFrame, flat: dict, layout: dict) -> str:
    """The midfoot load cells and the Participants with flat feet."""
    flat_feet, typical = flat["flat_feet"], flat["typical_foot"]
    medial = medial_midfoot_force(load_cells, layout)
    others = medial.drop(flat_feet)
    full = method_score(dropout, "full").per_participant_pct
    without = method_score(dropout, "full_minus_midgauges").per_participant_pct
    per_participant = 100 * (without / full - 1)
    worse, p = paired_comparison(dropout, "full_minus_midgauges", reference="full")
    out = (
        macro("ratioMedialMidfootFlat", f"{medial[flat_feet].mean() / others.mean():.1f}")
        + macro("changeWithoutMidfootLoadCells", pct(100 * (without.mean() / full.mean() - 1)))
        + macro("worseWithoutMidfootLoadCells", f"{worse}")
        + macro("pWithoutMidfootLoadCells", format_p(P_VALUES("pWithoutMidfootLoadCells", p)))
        + macro("codeTypicalFoot", typical)
    )
    minimal = method_score(dropout, "minimal_28ch").per_participant_pct
    plus = method_score(dropout, "minimal_plus_midgauges").per_participant_pct
    thirty_four = method_score(minimal_dir, "minimal_34ch").per_participant_pct
    comp_min, comp_plus = component_scores(dropout, "minimal_28ch"), component_scores(dropout, "minimal_plus_midgauges")
    for index, code in enumerate(flat_feet):
        name = "AB"[index]
        out += macro(f"codeFlatFoot{name}", code)
        out += macro(f"changeWithoutMidfootLoadCellsFlat{name}", rf"{per_participant[code]:+.1f}\,\%")
        out += macro(f"nrmseFlat{name}", f"{full[code]:.2f}")
        out += macro(f"nrmseMinimalFlat{name}", f"{minimal[code]:.2f}")
        out += macro(f"nrmseMinimalMidfootFlat{name}", f"{plus[code]:.2f}")
        out += macro(f"nrmseThirtyFourFlat{name}", f"{thirty_four[code]:.2f}")
        for c in ("Fy", "Mx"):
            change = 100 * (comp_min.loc[code, c] / comp_plus.loc[code, c] - 1)
            out += macro(f"change{c}WithoutMidfootFlat{name}", rf"{change:+.0f}\,\%")
    return out


def sensor_study_macros(evaluation: Path, bom_csv: Path, layout_json: Path, minimal_dir: Path, dedicated: Path) -> str:
    study = evaluation / "sensor_study"
    no_dropout, dropout = study / "no_dropout", study / "sensor_dropout"
    load_cells = pd.read_csv(study / "load_cells.csv")
    flat = json.loads((study / "flat_feet.json").read_text())
    layout = json.loads(layout_json.read_text())
    return "".join(
        [
            "% Sensor study (paper/sensor_study.py)\n",
            modality_macros(dropout),
            lost_patch_macros(no_dropout, dropout),
            minimal_macros(load_cells, dedicated),
            cost_macros(pd.read_csv(bom_csv), pd.read_csv(Path(bom_csv).with_name("bom_pair.csv"))),
            flat_foot_macros(dropout, minimal_dir, load_cells, flat, layout),
            minimal_spread_macros(dropout),
        ]
    )
