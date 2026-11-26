"""Write the paper's numbers, tables and figures from the published evaluation tables.

    python paper/make_artefacts.py [--evaluation paper/evaluation] [--output paper/artefacts]
                                   [--predictions RESULTS_DIR]

Every number the paper cites comes from ``numbers.tex`` (one LaTeX macro per number), so no
number is typed by hand. The score of a method is the protocol's: for each Participant, the
mean NRMSE of the 12 targets on Measured samples (all task families pooled), then the mean
and SD over the Participants.
"""

from __future__ import annotations

import argparse
import functools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Score:
    nrmse_pct: float
    nrmse_sd_pct: float
    force_pct_bw: float
    pcc: float
    per_participant_pct: pd.Series


@functools.lru_cache(maxsize=None)
def _read_table(path: str, modified_ns: int) -> pd.DataFrame:
    return pd.read_csv(path)


def read_table(path: Path) -> pd.DataFrame:
    """A table of the evaluation, read once per run of this script (unless the file changes)."""
    return _read_table(str(path), path.stat().st_mtime_ns)


def method_score(evaluation_dir: Path, run: str) -> Score:
    """Score of ``run`` in an ``evaluate_grfm.py`` output directory."""
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == run) & (table.subset == "measured") & (table.task_family == "ALL")]
    if rows.empty:
        raise ValueError(f"no measured ALL rows for run {run!r} in {evaluation_dir}")
    by_participant = rows.groupby("participant")
    nrmse = by_participant.nrmse_range.mean() * 100
    forces = rows[rows.component.str.startswith("F")].groupby("participant").rmse_pct_bw.mean()
    return Score(
        nrmse_pct=nrmse.mean(),
        nrmse_sd_pct=nrmse.std(),
        force_pct_bw=forces.mean(),
        pcc=by_participant.pcc_trial_mean.mean().mean(),
        per_participant_pct=nrmse,
    )


def paired_comparison(
    evaluation_dir: Path, run: str, reference: str = "mlp", reference_dir: Path | None = None
) -> tuple[int, float]:
    """Participants on which ``reference`` has the lower score, and the two-sided paired Wilcoxon p.

    ``reference`` is read from ``reference_dir`` when given, else from ``evaluation_dir``.
    """
    from scipy.stats import wilcoxon

    ours = method_score(reference_dir or evaluation_dir, reference).per_participant_pct
    theirs = method_score(evaluation_dir, run).per_participant_pct.reindex(ours.index)
    if theirs.isna().any():
        raise ValueError(f"{run} lacks Participants of {reference}: {list(ours.index[theirs.isna()])}")
    return int((ours < theirs).sum()), float(wilcoxon(ours, theirs).pvalue)


from pvalues import P_VALUES  # noqa: E402  (Holm adjustment, shared with sensor_study)


def format_p(p: float) -> str:
    """Two decimals, or three below 0.1, so that no p crosses 0.05 or 0.01 by rounding."""
    if p < 0.001:
        return "$<0.001$"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def format_speed(per_second: float) -> str:
    """Predictions per second: whole numbers from 100 up, with a thin space between thousands."""
    if per_second >= 100:
        return f"{per_second:,.0f}".replace(",", r"\,")
    return f"{per_second:.1f}"


def comparison_table(
    methods: list[tuple[str, Path | None, str | None]], speeds: dict[str, str | None] | None = None
) -> str:
    """LaTeX tabular of (label, evaluation dir, run); the first method is the reference of the paired tests.

    Each paired test reads the reference run from the other method's evaluation dir, so that
    both come from the same ``evaluate_grfm.py`` call. A method without an evaluation dir is still
    running: its row says so. With ``speeds`` (label -> predictions per second, ``None`` when not
    measured), a last column gives the speed.
    """
    _, _, reference_run = methods[0]
    columns = 6 if speeds is not None else 5
    header = r"Method & NRMSE (\%) & Forces (\%BW) & PCC & Ours better on & $p$"
    if speeds is not None:
        header += r" & Pred./s"
    lines = [rf"\begin{{tabular}}{{l{'c' * columns}}}", r"\toprule", header + r" \\", r"\midrule"]
    for index, (label, evaluation_dir, run) in enumerate(methods):
        if evaluation_dir is None:
            lines.append(rf"{label} & \multicolumn{{{columns}}}{{c}}{{\textit{{running}}}} \\")
            continue
        score = method_score(evaluation_dir, run)
        if index == 0:
            versus = "-- & --"
        else:
            wins, p = paired_comparison(evaluation_dir, run, reference=reference_run)
            versus = f"{wins}/{score.per_participant_pct.size} & {format_p(p)}"
        speed = ""
        if speeds is not None:
            speed = " & " + (speeds.get(label) or "n/a")
        lines.append(
            rf"{label} & ${score.nrmse_pct:.2f} \pm {score.nrmse_sd_pct:.2f}$ & {score.force_pct_bw:.2f} "
            rf"& {score.pcc:.2f} & {versus}{speed} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def format_change(change_pct: float) -> str:
    """Relative change in percent, signed, with one decimal."""
    return rf"{change_pct:+.1f}\,\%"


@dataclass(frozen=True)
class ConfigurationRow:
    """A row of a configuration table: a run of an evaluation dir, with its channel count (both feet)."""

    label: str
    channels: int
    evaluation_dir: Path
    run: str


@dataclass(frozen=True)
class Versus:
    """A score against a reference: relative change of the mean (%), and the paired test."""

    change_pct: float
    wins: int  # Participants on which the reference has the lower score
    n: int
    p: float


def versus(evaluation_dir: Path, run: str, reference: tuple[Path, str]) -> Versus:
    """``run`` against ``reference`` (evaluation dir, run)."""
    reference_dir, reference_run = reference
    score = method_score(evaluation_dir, run)
    wins, p = paired_comparison(evaluation_dir, run, reference=reference_run, reference_dir=reference_dir)
    change = 100 * (score.nrmse_pct / method_score(reference_dir, reference_run).nrmse_pct - 1)
    return Versus(change, wins, score.per_participant_pct.size, p)


def is_reference(evaluation_dir: Path, run: str, reference: tuple[Path, str]) -> bool:
    """Whether (``evaluation_dir``, ``run``) is the reference itself."""
    return (Path(evaluation_dir), run) == (Path(reference[0]), reference[1])


def configuration_table(rows: list[ConfigurationRow], reference: tuple[Path, str]) -> str:
    """LaTeX tabular of Sensor configurations, each against ``reference`` (evaluation dir, run).

    Per row: channels, score, relative change of the mean from the reference, PCC, and the paired
    test (Wilcoxon p). The
    reference's own row shows dashes. No %BW: the configuration evaluations estimate the body weight
    from the predictions, so their %BW is not comparable with the main table's.
    """
    lines = [
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Sensors & Ch. & NRMSE (\%) & vs all & PCC & $p$ \\",
        r"\midrule",
    ]
    for row in rows:
        score = method_score(row.evaluation_dir, row.run)
        if is_reference(row.evaluation_dir, row.run, reference):
            change, tested = "--", "--"
        else:
            v = versus(row.evaluation_dir, row.run, reference)
            change, tested = format_change(v.change_pct), format_p(v.p)
        lines.append(
            rf"{row.label} & {row.channels} & ${score.nrmse_pct:.2f} \pm {score.nrmse_sd_pct:.2f}$ & {change} "
            rf"& {score.pcc:.2f} & {tested} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def configuration_macros(name: str, row: ConfigurationRow, reference: tuple[Path, str]) -> str:
    """\\nrmseCfg<name>, \\changeCfg<name>, \\worseCfg<name> and \\pCfg<name> of a configuration row."""
    v = versus(row.evaluation_dir, row.run, reference)
    return (
        rf"\newcommand{{\nrmseCfg{name}}}{{{method_score(row.evaluation_dir, row.run).nrmse_pct:.2f}}}" + "\n"
        rf"\newcommand{{\changeCfg{name}}}{{{format_change(v.change_pct)}}}" + "\n"
        rf"\newcommand{{\worseCfg{name}}}{{{v.wins}}}" + "\n"
        rf"\newcommand{{\pCfg{name}}}{{{format_p(P_VALUES(f'pCfg{name}', v.p))}}}" + "\n"
    )


@dataclass(frozen=True)
class RobustnessRow:
    """One configuration scored on the model trained without and on the one trained with Sensor dropout."""

    label: str
    channels: int
    run: str
    with_dropout_dir: Path
    without_dropout_dir: Path | None = None  # default: the table's ``without`` dir


@dataclass(frozen=True)
class Robustness:
    """Both models' scores (%) under one configuration, and their paired test."""

    without_pct: float
    with_pct: float
    change_without_pct: float  # from the model's own full configuration
    change_with_pct: float
    dropout_better: int  # Participants on which the Sensor-dropout model has the lower score
    n: int
    p: float


def robustness(row: RobustnessRow, full: RobustnessRow, without: Path) -> Robustness:
    """``row`` scored on the model without Sensor dropout (``without``) and on the one with it."""
    without = row.without_dropout_dir or without
    a = method_score(without, row.run)
    b = method_score(row.with_dropout_dir, row.run)
    better, p = paired_comparison(without, row.run, reference=row.run, reference_dir=row.with_dropout_dir)
    return Robustness(
        a.nrmse_pct,
        b.nrmse_pct,
        100 * (a.nrmse_pct / method_score(without, full.run).nrmse_pct - 1),
        100 * (b.nrmse_pct / method_score(full.with_dropout_dir, full.run).nrmse_pct - 1),
        better,
        b.per_participant_pct.size,
        p,
    )


SUM_FAMILIES = ("balance", "squat", "lateral_2supports", "hulahoop")


def sabbah_total_macros(evaluation_dir: Path) -> str:
    """Total force of both feet on the continuous tasks: our MLP, Sabbah's per-foot forests summed, and his
    forest trained on the total (his own target). %BW averaged over the 3 force axes and the 4 tasks per
    Participant, then over the Participants; paired Wilcoxon of our MLP against the forest trained on the total."""
    from scipy.stats import wilcoxon

    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.subset == "both_feet_sum") & table.task_family.isin(SUM_FAMILIES)
                 & table.component.isin(["Fx", "Fy", "Fz"])]
    scores = rows.groupby(["run", "participant"]).rmse_pct_bw.mean().unstack("run")
    ours, total = scores["mlp"], scores["sabbah_total"]
    values = {
        "bwSumMain": f"{ours.mean():.2f}",
        "bwSumSabbahLoadsol": f"{scores['sabbah_loadsol'].mean():.2f}",
        "bwSumSabbahTotal": f"{total.mean():.2f}",
        "winsSumSabbahTotal": f"{int((ours < total).sum())}",
        "pSumSabbahTotal": format_p(P_VALUES("pSumSabbahTotal", float(wilcoxon(ours, total).pvalue))),
    }
    return "\n".join(rf"\newcommand{{\{name}}}{{{value}}}" for name, value in values.items())


def robustness_table(rows: list[RobustnessRow], without: Path) -> str:
    """LaTeX tabular of the Sensor-dropout MLP under each configuration of sensors present at test time.

    Beside it, the same configuration scored with the model trained without Sensor dropout. The changes
    and paired tests the text cites come from ``robustness_macros``.
    """
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{NRMSE (\%)} \\",
        r"\cmidrule(lr){3-4}",
        r"Sensors & Ch. & With dropout & Without \\",
        r"\midrule",
    ]
    for row in rows:
        r = robustness(row, rows[0], without)
        lines.append(rf"{row.label} & {row.channels} & {r.with_pct:.2f} & {r.without_pct:.2f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def robustness_macros(name: str, row: RobustnessRow, full: RobustnessRow, without: Path) -> str:
    """\\nrmseNoDropout<name>, \\nrmseDropout<name>, their changes, \\betterDropout<name> and \\pDropout<name>."""
    r = robustness(row, full, without)
    values = {
        f"nrmseNoDropout{name}": f"{r.without_pct:.2f}",
        f"nrmseDropout{name}": f"{r.with_pct:.2f}",
        f"changeNoDropout{name}": format_change(r.change_without_pct),
        f"changeDropout{name}": format_change(r.change_with_pct),
        f"betterDropout{name}": f"{r.dropout_better}",
        f"pDropout{name}": format_p(P_VALUES(f"pDropout{name}", r.p)),
    }
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in values.items())


def bootstrap_ci(values: pd.Series, resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """95 % percentile bootstrap interval of the mean of ``values`` (one value per Participant)."""
    import numpy as np

    array = np.asarray(values, dtype=float)
    draws = np.random.default_rng(seed).choice(array, size=(resamples, array.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def format_ci(values: pd.Series) -> str:
    low, high = bootstrap_ci(values)
    return f"[{low:.2f}, {high:.2f}]"


COMPONENTS = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
ANTHROPOMETRY = Path(__file__).resolve().parents[1] / "insole2grfm" / "participant_anthropometry.json"


def normalised(rows: pd.DataFrame, evaluation_dir: Path, column: str = "rmse",
               anthropometry: Path = ANTHROPOMETRY) -> pd.Series:
    """``column`` of each row as a percentage of body weight for a force (%BW) and of body weight times height
    for a moment (%BW.H), with the Participant's unloaded body weight (``body_weight.csv``) and height."""
    import json

    weight = rows.participant.map(body_weight_n(Path(evaluation_dir) / "body_weight.csv"))
    heights = {code: entry["height_m"] for code, entry in json.loads(Path(anthropometry).read_text()).items()}
    height = rows.participant.map(heights).where(~rows.component.str.startswith("F"), 1.0)
    return 100 * rows[column] / (weight * height)


def normalised_digits(component: str) -> int:
    """Decimals of a normalised error: two for a force (%BW), three for a moment (%BW.H, about ten times smaller)."""
    return 2 if component.startswith("F") else 3


def normalised_unit(component: str) -> str:
    return r"\%BW" if component.startswith("F") else r"\%BW$\cdot$H"


def physical_unit(component: str) -> str:
    return "N" if component.startswith("F") else r"N$\cdot$m"


@dataclass(frozen=True)
class ComponentStats:
    """One force or moment component, both feet averaged per Participant (Measured samples, all tasks)."""

    rmse: pd.Series  # N or N m, per Participant
    pct_bw: float | None  # forces only
    nrmse_pct: pd.Series
    pcc: float
    airborne_rmse: float | None  # error where the truth is zero
    airborne_bias: float | None  # the mean residual force or moment


def component_stats(evaluation_dir: Path, run: str) -> dict[str, ComponentStats]:
    """Per component, the statistics of ``component_table``."""
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == run) & (table.task_family == "ALL")]
    measured, airborne = rows[rows.subset == "measured"], rows[rows.subset == "airborne"]

    def per_participant(frame, component, column):
        return frame[frame.component == component].groupby("participant")[column].mean()

    stats = {}
    for component in COMPONENTS:
        force = component.startswith("F")
        stats[component] = ComponentStats(
            rmse=per_participant(measured, component, "rmse"),
            pct_bw=per_participant(measured, component, "rmse_pct_bw").mean() if component.startswith("F") else None,
            nrmse_pct=per_participant(measured, component, "nrmse_range") * 100,
            pcc=per_participant(measured, component, "pcc_trial_mean").mean(),
            airborne_rmse=per_participant(airborne, component, "rmse").mean() if not airborne.empty else None,
            airborne_bias=per_participant(airborne, component, "mean_error").mean() if not airborne.empty else None,
        )
    return stats


def component_table(evaluation_dir: Path, run: str) -> str:
    """LaTeX tabular of each force and moment component, for the left and the right foot.

    Per component and foot, averaged over Participants (all tasks): on Measured samples, RMSE in N or
    N m, NRMSE and PCC; on Airborne samples, where the truth is zero (so NRMSE and PCC are undefined),
    the RMSE and the mean error, i.e. the residual force or moment.
    """
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == run) & (table.task_family == "ALL")]
    measured, airborne = rows[rows.subset == "measured"], rows[rows.subset == "airborne"]

    def mean(frame, component, foot, column):
        values = frame[(frame.component == component) & (frame.foot == foot)].groupby("participant")[column].mean()
        return values.mean() if len(values) else float("nan")

    lines = [
        r"\begin{tabular}{lcccccccccc}",
        r"\toprule",
        r" & \multicolumn{6}{c}{Measured samples} & \multicolumn{4}{c}{Airborne samples} \\",
        r"\cmidrule(lr){2-7}\cmidrule(lr){8-11}",
        r" & \multicolumn{3}{c}{Left} & \multicolumn{3}{c}{Right} & \multicolumn{2}{c}{Left} & \multicolumn{2}{c}{Right} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
        r"Comp. & RMSE & NRMSE (\%) & PCC & RMSE & NRMSE (\%) & PCC & RMSE & Mean & RMSE & Mean \\",
        r"\midrule",
    ]
    for component in COMPONENTS:
        force = component.startswith("F")
        digits = 1 if force else 2
        cells = []
        for foot in ("L", "R"):
            cells += [f"{mean(measured, component, foot, 'rmse'):.{digits}f}",
                      f"{100 * mean(measured, component, foot, 'nrmse_range'):.2f}",
                      f"{mean(measured, component, foot, 'pcc_trial_mean'):.2f}"]
        for foot in ("L", "R"):
            rmse = mean(airborne, component, foot, "rmse")
            bias = mean(airborne, component, foot, "mean_error")
            cells += ["--", "--"] if pd.isna(rmse) else [f"{rmse:.{digits}f}", f"${bias:+.{digits}f}$"]
        lines.append(rf"${component[0]}_{component[1]}$ ({'N' if force else r'N$\cdot$m'}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


FEET = {"L": "left", "R": "right"}


@dataclass(frozen=True)
class ResultsColumn:
    """A method of the results table."""

    label: str
    evaluation_dir: Path
    run: str
    ms_per_frame: float | None  # one prediction of the 12 targets on one CPU thread; None: not measured
    real_time: bool  # causal and faster than the sample period
    ours: bool = False  # our methods: their columns are shaded
    name: str = ""  # macro suffix of METHODS: its gap p is adjusted as \p<name>


OURS_SHADE = r">{\columncolor[gray]{0.92}}c"  # needs colortbl in the paper


def format_ms(ms: float) -> str:
    """Milliseconds: two decimals below 10, whole numbers with a thin space between thousands above."""
    return f"{ms:.2f}" if ms < 10 else f"{ms:,.0f}".replace(",", r"\,")


def results_table(methods: list[ResultsColumn], anthropometry: Path = ANTHROPOMETRY) -> str:
    """LaTeX tabular: RMSE, NRMSE and PCC of each method for each component of each foot.

    Measured samples, all tasks; each value is the mean over the Participants. The RMSE of a force is in
    %BW and that of a moment in %BW.H (``normalised``). In each row, the lowest
    NRMSE among the real-time methods is in bold and the lowest overall is underlined; our columns are
    shaded. Below: the score of each method (mean $\pm$ SD of the Participants' mean over the 12
    targets), its gap to the first method (relative change and paired Wilcoxon p), the time per frame and
    whether it runs in real time.
    """
    def measured(evaluation_dir, run):
        table = read_table(Path(evaluation_dir) / "per_participant.csv")
        rows = table[(table.run == run) & (table.subset == "measured") & (table.task_family == "ALL")]
        if rows.empty:
            raise ValueError(f"no measured ALL rows for run {run!r} in {evaluation_dir}")
        return rows

    frames = [measured(m.evaluation_dir, m.run) for m in methods]
    frames = [frame.assign(rmse_norm=normalised(frame, m.evaluation_dir, anthropometry=anthropometry))
              for frame, m in zip(frames, methods)]
    n = len(methods)
    spec = "".join((OURS_SHADE * 3) if m.ours else "ccc" for m in methods)

    def span(method, cell):
        return rf"\multicolumn{{3}}{{{OURS_SHADE if method.ours else 'c'}}}{{{cell}}}"

    def mark(values: list[float], digits: int = 2) -> list[str]:
        """Format; bold the lowest among real-time methods, underline the lowest overall."""
        text = [f"{v:.{digits}f}" for v in values]
        overall = min(text, key=float)
        real_time = min((t for t, m in zip(text, methods) if m.real_time), key=float, default=None)
        out = []
        for t in text:
            cell = rf"\textbf{{{t}}}" if t == real_time else t
            out.append(rf"\underline{{{cell}}}" if t == overall else cell)
        return out

    lines = [
        rf"\begin{{tabular}}{{l{spec}}}",
        r"\toprule",
        " & " + " & ".join(span(m, m.label) for m in methods) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(n)),
        r"Target & " + " & ".join([r"RMSE & NRMSE (\%) & PCC"] * n) + r" \\",
        r"\midrule",
    ]
    for component in COMPONENTS:
        for foot in FEET:
            stats = []
            for frame in frames:
                rows = frame[(frame.component == component) & (frame.foot == foot)]
                stats.append((rows.rmse_norm.mean(), 100 * rows.nrmse_range.mean(), rows.pcc_trial_mean.mean()))
            nrmses = mark([nrmse for _, nrmse, _ in stats])
            cells = []
            for (rmse, _, pcc), nrmse in zip(stats, nrmses):
                cells += [f"{rmse:.{normalised_digits(component)}f}", nrmse, f"{pcc:.2f}"]
            lines.append(
                rf"${component[0]}_{{{component[1]},{foot}}}$ ({normalised_unit(component)}) & " + " & ".join(cells) + r" \\"
            )
    lines.append(r"\midrule")
    scores = [method_score(m.evaluation_dir, m.run) for m in methods]
    means = mark([score.nrmse_pct for score in scores])
    cells = []
    for score, mean in zip(scores, means):
        cells += ["--", rf"{mean} $\pm$ {score.nrmse_sd_pct:.2f}", f"{score.pcc:.2f}"]
    lines.append(r"All targets & " + " & ".join(cells) + r" \\")
    reference = methods[0]

    def row(label, cells):
        lines.append(label + " & " + " & ".join(span(m, cell) for m, cell in zip(methods, cells)) + r" \\")

    gaps = []
    for method, score in zip(methods[1:], scores[1:]):
        _, p = paired_comparison(method.evaluation_dir, method.run, reference=reference.run,
                                 reference_dir=reference.evaluation_dir)
        p = P_VALUES(f"p{method.name}", p) if method.name else p
        change = format_change(100 * (score.nrmse_pct / scores[0].nrmse_pct - 1))
        gaps.append(rf"{change} ($p$ {format_p(p)})" if format_p(p).startswith("$") else rf"{change} ($p = {format_p(p)}$)")
    row(rf"NRMSE vs {reference.label}", ["--"] + gaps)
    fastest = min(m.ms_per_frame for m in methods if m.ms_per_frame)
    row(r"Inference time per frame, 12 targets (ms)", [
        "n/a" if not m.ms_per_frame
        else rf"\textbf{{{format_ms(m.ms_per_frame)}}}" if m.ms_per_frame == fastest else format_ms(m.ms_per_frame)
        for m in methods
    ])
    row(r"Real time", ["yes" if m.real_time else "no" for m in methods])
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def rmse_table(methods: list[ResultsColumn]) -> str:
    """LaTeX tabular: RMSE of each method for each component of each foot, in N or N m.

    Measured samples, all tasks; mean $\pm$ SD over the Participants. Our columns are shaded.
    """
    spec = "".join(OURS_SHADE if m.ours else "c" for m in methods)
    lines = [
        rf"\begin{{tabular}}{{l{spec}}}",
        r"\toprule",
        "Target & " + " & ".join(m.label for m in methods) + r" \\",
        r"\midrule",
    ]
    frames = []
    for m in methods:
        table = read_table(Path(m.evaluation_dir) / "per_participant.csv")
        frames.append(table[(table.run == m.run) & (table.subset == "measured") & (table.task_family == "ALL")])
    for component in COMPONENTS:
        digits = 1 if component.startswith("F") else 2
        for foot in FEET:
            cells = []
            for frame in frames:
                rmse = frame[(frame.component == component) & (frame.foot == foot)].groupby("participant").rmse.mean()
                cells.append(rf"${rmse.mean():.{digits}f} \pm {rmse.std():.{digits}f}$")
            lines.append(rf"${component[0]}_{{{component[1]},{foot}}}$ ({physical_unit(component)}) & "
                         + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def airborne_table(evaluation_dir: Path, run: str, anthropometry: Path = ANTHROPOMETRY) -> str:
    """LaTeX tabular of the airborne samples, where the truth is zero: RMSE and mean error per component and foot,
    in %BW for the forces and %BW.H for the moments."""
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == run) & (table.subset == "airborne") & (table.task_family == "ALL")]
    rows = rows.assign(rmse=normalised(rows, evaluation_dir, anthropometry=anthropometry),
                       mean_error=normalised(rows, evaluation_dir, "mean_error", anthropometry=anthropometry))
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Left} & \multicolumn{2}{c}{Right} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Comp. & RMSE & Mean & RMSE & Mean \\",
        r"\midrule",
    ]
    for component in COMPONENTS:
        digits = normalised_digits(component)
        cells = []
        for foot in FEET:
            values = rows[(rows.component == component) & (rows.foot == foot)].groupby("participant")
            rmse, bias = values.rmse.mean(), values.mean_error.mean()
            cells += ["--", "--"] if rmse.empty else [
                rf"${rmse.mean():.{digits}f} \pm {rmse.std():.{digits}f}$",
                rf"${bias.mean():+.{digits}f} \pm {bias.std():.{digits}f}$",
            ]
        lines.append(rf"${component[0]}_{component[1]}$ ({normalised_unit(component)}) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def component_macros(evaluation_dir: Path, run: str) -> str:
    """\\nrmseComp<C>, \\pccComp<C> per component; \\bwComp<C>, \\airborneRmse<C>, \\airborneBias<C> per force."""
    lines = []
    for component, c in component_stats(evaluation_dir, run).items():
        lines += [rf"\newcommand{{\nrmseComp{component}}}{{{c.nrmse_pct.mean():.2f}}}",
                  rf"\newcommand{{\pccComp{component}}}{{{c.pcc:.2f}}}"]
        if c.pct_bw is not None:
            lines.append(rf"\newcommand{{\bwComp{component}}}{{{c.pct_bw:.2f}}}")
        if c.airborne_rmse is not None:
            digits = 1 if component.startswith("F") else 2
            lines += [rf"\newcommand{{\airborneRmse{component}}}{{{c.airborne_rmse:.{digits}f}}}",
                      rf"\newcommand{{\airborneBias{component}}}{{${c.airborne_bias:+.{digits}f}$}}"]
    return "\n".join(lines) + "\n"


def per_foot_rmse_macros(evaluation_dir: Path, run: str) -> str:
    """\\rmseComp<C>Left and \\rmseComp<C>Right: RMSE of each component and foot on measured samples, mean over the
    Participants, as in the results table (forces in N with one decimal, moments in N m with two)."""
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == run) & (table.subset == "measured") & (table.task_family == "ALL")]
    lines = []
    for (component, foot), rmse in rows.groupby(["component", "foot"]).rmse.mean().items():
        if foot not in ("L", "R"):
            continue
        digits = 1 if component.startswith("F") else 2
        side = "Left" if foot == "L" else "Right"
        lines.append(rf"\newcommand{{\rmseComp{component}{side}}}{{{rmse:.{digits}f}}}")
    return "\n".join(lines) + "\n"


def per_foot_normalised_macros(evaluation_dir: Path, run: str, anthropometry: Path = ANTHROPOMETRY) -> str:
    """\\bwComp<C>Left and \\bwComp<C>Right: RMSE of each component and foot on measured samples in %BW (forces)
    or %BW.H (moments), mean over the Participants, as in the results table; \\airborneBw<C>: the RMSE of each
    component on airborne samples, both feet, likewise normalised."""
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == run) & (table.task_family == "ALL")]
    rows = rows.assign(rmse_norm=normalised(rows, evaluation_dir, anthropometry=anthropometry))
    lines = []
    measured = rows[rows.subset == "measured"]
    for (component, foot), value in measured.groupby(["component", "foot"]).rmse_norm.mean().items():
        if foot in ("L", "R"):
            lines.append(rf"\newcommand{{\bwComp{component}{'Left' if foot == 'L' else 'Right'}}}"
                         rf"{{{value:.{normalised_digits(component)}f}}}")
    airborne = rows[(rows.subset == "airborne") & rows.foot.isin(["L", "R"])]
    for component, value in airborne.groupby(["component", "participant"]).rmse_norm.mean().groupby("component").mean().items():
        lines.append(rf"\newcommand{{\airborneBw{component}}}{{{value:.{normalised_digits(component)}f}}}")
    return "\n".join(lines) + "\n"


# Sensitivity of the reference GRFM to the foot frame (Section V-A): a load normal to the foot, a pitch error of
# the marker-defined frame and a lateral shift of its origin. The changes they cause are computed, not typed.
SENSITIVITY_LOAD_N = 700.0
SENSITIVITY_PITCH_DEG = 1.0
SENSITIVITY_SHIFT_MM = 2.0


def reference_sensitivity_macros() -> str:
    import math

    values = {
        "sensitivityLoadN": f"{SENSITIVITY_LOAD_N:.0f}",
        "sensitivityPitchDeg": f"{SENSITIVITY_PITCH_DEG:.0f}",
        "sensitivityShiftMm": f"{SENSITIVITY_SHIFT_MM:.0f}",
        "sensitivityFxN": f"{SENSITIVITY_LOAD_N * math.sin(math.radians(SENSITIVITY_PITCH_DEG)):.1f}",
        "sensitivityMxNm": f"{SENSITIVITY_LOAD_N * SENSITIVITY_SHIFT_MM / 1000:.1f}",
    }
    return "".join(rf"\newcommand{{\{name}}}{{{value}}}" + "\n" for name, value in values.items())


@dataclass(frozen=True)
class FamilyStats:
    trials: int
    nrmse_pct: pd.Series  # per Participant, normalised by the all-task range
    fz_pct_bw: float
    pcc: float


def family_stats(evaluation_dir: Path, run: str, families: list[str]) -> dict[str, FamilyStats]:
    """Per task family (and ``ALL``), the statistics of ``family_table``.

    Every family's RMSE is normalised by the target's range over all tasks (``rmse / nrmse_range`` of
    the ALL row), not over the family alone: a family's own range is narrower, so its own NRMSE would
    be higher than the all-task score and not comparable with it.
    """
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == run) & (table.subset == "measured")].copy()
    pooled = rows[rows.task_family == "ALL"].set_index(["participant", "target"])
    rows = rows.join((pooled.rmse / pooled.nrmse_range).rename("target_range"), on=["participant", "target"])
    rows["nrmse_range"] = rows.rmse / rows.target_range
    stats = {}
    for family in families:
        frame = rows[rows.task_family == family]
        if frame.empty:
            continue
        by_participant = frame.groupby("participant")
        stats[family] = FamilyStats(
            trials=int(by_participant.n_trials.max().sum()),
            nrmse_pct=by_participant.nrmse_range.mean() * 100,
            fz_pct_bw=frame[frame.component == "Fz"].groupby("participant").rmse_pct_bw.mean().mean(),
            pcc=by_participant.pcc_trial_mean.mean().mean(),
        )
    return stats


def family_table(evaluation_dir: Path, run: str, families: dict[str, str]) -> str:
    """LaTeX tabular of the score per task family (``families``: family -> label, in table order), then all tasks.

    Per row: NRMSE (mean +- SD over Participants, see ``family_stats``) and its 95 % bootstrap
    interval, vertical force RMSE in %BW (both feet), PCC.
    """
    labels = {**families, "ALL": "All tasks"}
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Task & NRMSE (\%) & 95\,\% CI & $F_z$ (\%BW) & PCC \\",
        r"\midrule",
    ]
    for family, f in family_stats(evaluation_dir, run, list(labels)).items():
        if family == "ALL":
            lines.append(r"\midrule")
        lines.append(
            rf"{labels[family]} & ${f.nrmse_pct.mean():.2f} \pm {f.nrmse_pct.std():.2f}$ "
            rf"& {format_ci(f.nrmse_pct)} & {f.fz_pct_bw:.2f} & {f.pcc:.2f} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def measured_share_macros(evaluation_dir: Path, run: str) -> str:
    """\\shareMeasured<Family>: percentage of the measured samples (one target, all Participants) in each task family."""
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == run) & (table.subset == "measured") & (table.target == "Fz_L")]
    counts = rows.groupby("task_family").n.sum()
    total = counts.pop("ALL")
    return "".join(
        rf"\newcommand{{\shareMeasured{FAMILY_MACROS[family]}}}{{{100 * count / total:.1f}}}" + "\n"
        for family, count in counts.items() if family in FAMILY_MACROS
    )


def family_macros(evaluation_dir: Path, run: str) -> str:
    """\\nrmseFamily<F> and \\fzBwFamily<F> per task family (suffixes of FAMILY_MACROS), and the CI of all tasks."""
    stats = family_stats(evaluation_dir, run, [*FAMILY_MACROS, "ALL"])
    lines = []
    for family, name in FAMILY_MACROS.items():
        if family in stats:
            lines += [rf"\newcommand{{\nrmseFamily{name}}}{{{stats[family].nrmse_pct.mean():.2f}}}",
                      rf"\newcommand{{\fzBwFamily{name}}}{{{stats[family].fz_pct_bw:.2f}}}"]
    low, high = bootstrap_ci(stats["ALL"].nrmse_pct)
    lines += [rf"\newcommand{{\ciLowMain}}{{{low:.2f}}}", rf"\newcommand{{\ciHighMain}}{{{high:.2f}}}"]
    return "\n".join(lines) + "\n"


def body_weight_n(path: Path) -> pd.Series:
    """Body weight (N) per Participant code, from an evaluation's ``body_weight.csv`` (unloaded Subjects)."""
    weights = pd.read_csv(path)
    return weights[~weights.subject.str.fullmatch(r"[A-Z]\d{2,}b")].set_index("subject").body_weight_N


@dataclass(frozen=True)
class LoadingStats:
    nrmse_pct: pd.Series  # per Participant, normalised by the all-task range
    fz_rmse_n: pd.Series
    fz_pct_bw: pd.Series


def loading_stats(loading: pd.DataFrame, evaluation_dir: Path, run: str) -> dict[str, LoadingStats]:
    """Per condition (``unloaded``, ``backpack``), the statistics of ``loading_table``.

    Each target's RMSE is divided by its range over all tasks and conditions (``rmse / nrmse_range``
    of the evaluation's ALL row), so both conditions share one scale; %BW uses the unloaded body weight.
    """
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    pooled = table[(table.run == run) & (table.subset == "measured") & (table.task_family == "ALL")]
    pooled = pooled.set_index(["participant", "target"])
    rows = loading.join((pooled.rmse / pooled.nrmse_range).rename("target_range"), on=["participant", "target"])
    rows = rows.assign(nrmse=rows.rmse / rows.target_range * 100)
    weight = body_weight_n(Path(evaluation_dir) / "body_weight.csv")
    stats = {}
    for condition in ("unloaded", "backpack"):
        frame = rows[rows.condition == condition]
        fz = frame[frame.target.str.startswith("Fz")].groupby("participant").rmse.mean()
        stats[condition] = LoadingStats(
            nrmse_pct=frame.groupby("participant").nrmse.mean(),
            fz_rmse_n=fz,
            fz_pct_bw=fz / weight.reindex(fz.index) * 100,
        )
    return stats


def backpack_comparison(stats: dict[str, LoadingStats]) -> tuple[int, int, float]:
    """Participants whose backpack NRMSE is higher than their unloaded one, their number, and Wilcoxon's p."""
    from scipy.stats import wilcoxon

    unloaded, backpack = stats["unloaded"].nrmse_pct, stats["backpack"].nrmse_pct
    common = unloaded.index.intersection(backpack.index)
    return int((backpack[common] > unloaded[common]).sum()), common.size, float(wilcoxon(unloaded[common], backpack[common]).pvalue)


def loading_macros(loading: pd.DataFrame, evaluation_dir: Path, run: str) -> str:
    """\\nrmseUnloaded, \\nrmseBackpack, \\fzBwUnloaded, \\fzBwBackpack, \\worseBackpack, \\pBackpack."""
    stats = loading_stats(loading, evaluation_dir, run)
    worse, _, p = backpack_comparison(stats)
    values = {"worseBackpack": f"{worse}", "pBackpack": format_p(P_VALUES("pBackpack", p))}
    for condition, name in (("unloaded", "Unloaded"), ("backpack", "Backpack")):
        values[f"nrmse{name}"] = f"{stats[condition].nrmse_pct.mean():.2f}"
        values[f"fzBw{name}"] = f"{stats[condition].fz_pct_bw.mean():.2f}"
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in values.items())


def transition_macros(transitions: pd.DataFrame, body_weight_csv: Path, airborne: pd.DataFrame) -> str:
    """\\transitionRmse<E> (N), \\transitionBw<E> (%BW) and \\transitionBias<E> per event; \\airborneBiasFamily<F> per task family."""
    stats = transition_stats(transitions, body_weight_csv)
    names = {"stance": "Stance", "contact": "Contact", "lift_off": "LiftOff", "air": "Air"}
    lines = []
    for event, name in names.items():
        lines += [rf"\newcommand{{\transitionRmse{name}}}{{{stats.loc[event].rmse:.1f}}}",
                  rf"\newcommand{{\transitionBw{name}}}{{{stats.loc[event].pct_bw:.2f}}}",
                  rf"\newcommand{{\transitionBias{name}}}{{${stats.loc[event].bias:+.1f}$}}"]
    by_family = airborne.groupby(["task_family", "participant"]).mean_error_N.mean().groupby("task_family").mean()
    for family, name in FAMILY_MACROS.items():
        if family in by_family:
            lines.append(rf"\newcommand{{\airborneBiasFamily{name}}}{{${by_family[family]:+.1f}$}}")
    return "\n".join(lines) + "\n"


def loading_table(loading: pd.DataFrame, evaluation_dir: Path, run: str) -> str:
    """LaTeX tabular of the trials without and with the backpack (``loading.csv`` of loading_and_transitions.py).

    Per condition: NRMSE (see ``loading_stats``), vertical force RMSE in %BW; for the backpack,
    the paired Wilcoxon p against unloaded.
    """
    stats = loading_stats(loading, evaluation_dir, run)
    worse, n, p = backpack_comparison(stats)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Load condition & NRMSE (\%) & $F_z$ (\%BW) & $p$ \\",
        r"\midrule",
    ]
    for condition, label, tested in (("unloaded", "Unloaded", "--"), ("backpack", "Backpack", format_p(P_VALUES("pBackpack", p)))):
        s = stats[condition]
        lines.append(
            rf"{label} & ${s.nrmse_pct.mean():.2f} \pm {s.nrmse_pct.std():.2f}$ & {s.fz_pct_bw.mean():.2f} & {tested} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


TRANSITION_EVENTS = {  # event of transitions.csv -> label, in table order
    "stance": "Stance, away from transitions",
    "contact": r"Contact, $\pm$50\,ms",
    "lift_off": r"Lift-off, $\pm$50\,ms",
    "air": "In the air, away from transitions",
}


def transition_stats(transitions: pd.DataFrame, body_weight_csv: Path) -> pd.DataFrame:
    """Per event: events counted over Participants and feet, vertical force RMSE (N, mean and SD over
    Participants of the mean of both feet), RMSE in %BW and mean error (N)."""
    weight = body_weight_n(body_weight_csv)
    per_participant = transitions.groupby(["event", "participant"]).agg(
        rmse=("rmse_N", "mean"), bias=("mean_error_N", "mean"), events=("n_events", "sum")
    ).reset_index()
    per_participant["pct_bw"] = per_participant.rmse / per_participant.participant.map(weight) * 100
    return per_participant.groupby("event").agg(
        events=("events", "sum"), rmse=("rmse", "mean"), rmse_sd=("rmse", "std"),
        pct_bw=("pct_bw", "mean"), pct_bw_sd=("pct_bw", "std"), bias=("bias", "mean"), bias_sd=("bias", "std"),
    )


def transition_table(transitions: pd.DataFrame, body_weight_csv: Path) -> str:
    """LaTeX tabular of the vertical force error in stance, around contact and lift-off, and in the air."""
    stats = transition_stats(transitions, body_weight_csv)
    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Samples & RMSE (\%BW) & Mean error (N) \\",
        r"\midrule",
    ]
    for event, label in TRANSITION_EVENTS.items():
        s = stats.loc[event]
        lines.append(rf"{label} & ${s.pct_bw:.2f} \pm {s.pct_bw_sd:.2f}$ & ${s.bias:+.1f} \pm {s.bias_sd:.1f}$ \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def imu_calibration_macros(calibration_dir: Path) -> str:
    """Macros of the IMU calibrations of both insoles (Tedaldi et al. multi-position method).

    Per sensor, over the Patches of both insoles: largest scale-factor deviation from 1 (%), largest
    misalignment angle (deg), and the range over the two insoles of the calibration error before and
    after correction (accelerometer: RMSE of the norm of gravity, mg; gyroscope: RMSE of the rotation
    angle between static poses, deg).
    """
    import json

    def load(pattern):
        return [json.loads(p.read_text()) for p in sorted(Path(calibration_dir).glob(pattern))]

    def span(values, scale, digits):
        low, high = min(values) * scale, max(values) * scale
        return f"{low:.{digits}f}--{high:.{digits}f}"

    values = {}
    for name, pattern, raw_key, cal_key, scale, digits, unit in (
        ("acc", "*/calibration_acc*/accelerometer_calibration.json", "raw_norm_rmse_g", "calibrated_fit_norm_rmse_g", 1000, 1, "Mg"),
        ("gyro", "*/calibration_gyro*/gyroscope_calibration.json", "raw_angle_rmse_deg", "calibrated_fit_angle_rmse_deg", 1, 2, "Deg"),
    ):
        files = load(pattern)
        if not files:
            continue
        patches = [p for f in files for p in f["patches"].values()]
        scales = [abs(v - 1) * 100 for p in patches for v in p["scale_factors"].values()]
        angles = [abs(v) * 180 / 3.141592653589793 for p in patches for v in p["misalignment_parameters_rad"].values()]
        per_insole_raw = [sum(p["metrics"][raw_key] for p in f["patches"].values()) / len(f["patches"]) for f in files]
        per_insole_cal = [sum(p["metrics"][cal_key] for p in f["patches"].values()) / len(f["patches"]) for f in files]
        label = "Norm" if name == "acc" else "Angle"
        values |= {
            f"{name}ScaleDevMaxPct": f"{max(scales):.1f}",
            f"{name}MisalignMaxDeg": f"{max(angles):.2f}",
            f"{name}Raw{label}{unit}": span(per_insole_raw, scale, digits),
            f"{name}Cal{label}{unit}": span(per_insole_cal, scale, digits),
        }
    return "".join(rf"\newcommand{{\{k}}}{{{v}}}" + "\n" for k, v in values.items())


def per_target_component_macros(evaluation_dir: Path) -> str:
    """NRMSE per component of our MLP, the per-target MLP and Ke2KeNet (78 ch.), and their paired tests.

    Both feet are averaged per Participant; ``All`` averages the 12 targets. ``winsComp<C>PerTarget`` counts the Participants on which
    the per-target MLP beats our MLP; ``winsComp<C>MahmudOverPerTarget`` those on which Ke2KeNet
    beats the per-target MLP.
    """
    from scipy.stats import wilcoxon

    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    table = table[(table.subset == "measured") & (table.task_family == "ALL")]
    scores = table.groupby(["run", "component", "participant"]).nrmse_range.mean() * 100
    pooled = table.assign(component="All").groupby(["run", "component", "participant"]).nrmse_range.mean() * 100
    scores = pd.concat([scores, pooled])
    lines = []
    for component in ("All", "Fx", "Fy", "Fz", "Mx", "My", "Mz"):
        main, per_target, mahmud = (scores.loc[run, component] for run in ("mlp", PER_TARGET_RUN, "mahmud_raw"))
        per_target, mahmud = per_target.reindex(main.index), mahmud.reindex(main.index)
        values = {
            f"nrmseComp{component}Main": f"{main.mean():.2f}",
            f"nrmseComp{component}PerTarget": f"{per_target.mean():.2f}",
            f"nrmseComp{component}MahmudRaw": f"{mahmud.mean():.2f}",
            f"changeComp{component}PerTarget": format_change(100 * (per_target.mean() / main.mean() - 1)),
            f"winsComp{component}PerTarget": str(int((per_target < main).sum())),
            f"pComp{component}PerTarget": format_p(P_VALUES(f"pComp{component}PerTarget", wilcoxon(per_target, main).pvalue)),
            f"winsComp{component}MahmudOverPerTarget": str(int((mahmud < per_target).sum())),
            f"pComp{component}PerTargetMahmud": format_p(
                P_VALUES(f"pComp{component}PerTargetMahmud", wilcoxon(per_target, mahmud).pvalue)
            ),
        }
        lines += [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in values.items()]
    return "\n".join(lines) + "\n"


def parameter_macros(parameters_csv: Path) -> str:
    """\\params<Method> in millions, and \\ratioParams<Method>: how many times our MLP's count, to two significant digits."""
    counts = pd.read_csv(parameters_csv).set_index("method").parameters
    names = {"mlp": "Main", "mlp_per_target": "PerTarget", "mahmud_gauges": "MahmudGauges", "mahmud_raw": "MahmudRaw"}
    lines = []
    for method, name in names.items():
        millions = counts[method] / 1e6
        lines.append(rf"\newcommand{{\params{name}}}{{{millions:.2f}}}" if millions < 10
                     else rf"\newcommand{{\params{name}}}{{{millions:.0f}}}")
        if method != "mlp":
            ratio = round(float(f"{counts[method] / counts['mlp']:.2g}"))
            lines.append(rf"\newcommand{{\ratioParams{name}}}{{{f'{ratio:,}'.replace(',', chr(92) + ',')}}}")
    return "\n".join(lines) + "\n"


def number_macros(methods: dict[str, tuple[Path, str]]) -> str:
    """LaTeX macros of each method's score; ``methods`` maps a macro suffix to (evaluation dir, run)."""
    lines = []
    for name, (evaluation_dir, run) in methods.items():
        score = method_score(evaluation_dir, run)
        lines += [
            rf"\newcommand{{\nrmse{name}}}{{{score.nrmse_pct:.2f}}}",
            rf"\newcommand{{\nrmseSd{name}}}{{{score.nrmse_sd_pct:.2f}}}",
            rf"\newcommand{{\forceBw{name}}}{{{score.force_pct_bw:.2f}}}",
            rf"\newcommand{{\pcc{name}}}{{{score.pcc:.2f}}}",
        ]
    return "\n".join(lines) + "\n"


GRAVITY_M_S2 = 9.81
FAMILY_MACROS = {  # task family in the tables -> macro suffix
    "balance": "Balance",
    "squat": "Squat",
    "hulahoop": "Hulahoop",
    "lateral_2supports": "LateralDouble",
    "lateral_1_appui": "LateralSingle",
    "walk": "Walk",
    "jump": "Jump",
}


def dataset_macros(evaluation_dir: Path, sex_file: Path, anthropometry_file: Path) -> str:
    """Participants by sex, age, height, body mass (mean and SD over Participants) and Trials per task family."""
    import json

    sex = json.loads(Path(sex_file).read_text())
    body = pd.DataFrame(json.loads(Path(anthropometry_file).read_text())).T
    weights = pd.read_csv(Path(evaluation_dir) / "body_weight.csv")
    unloaded = weights[~weights.subject.str.fullmatch(r"[A-Z]\d{2,}b")]  # P01, not its loaded Subject P01b
    weights = unloaded.set_index("subject").body_weight_N / GRAVITY_M_S2
    table = read_table(Path(evaluation_dir) / "per_participant.csv")
    rows = table[(table.run == table.run.iloc[0]) & (table.subset == "measured")]
    # A trial counts once if any of its 12 targets has a Measured sample.
    trials = rows.groupby(["task_family", "participant"]).n_trials.max().groupby("task_family").sum()
    lines = [
        rf"\newcommand{{\nParticipants}}{{{len(sex)}}}",
        rf"\newcommand{{\nWomen}}{{{sum(v == 'female' for v in sex.values())}}}",
        rf"\newcommand{{\nMen}}{{{sum(v == 'male' for v in sex.values())}}}",
        rf"\newcommand{{\bodyMassKg}}{{{weights.mean():.1f}}}",
        rf"\newcommand{{\bodyMassSdKg}}{{{weights.std():.1f}}}",
        rf"\newcommand{{\ageYears}}{{{body.age_years.mean():.1f}}}",
        rf"\newcommand{{\ageSdYears}}{{{body.age_years.std():.1f}}}",
        rf"\newcommand{{\heightM}}{{{body.height_m.mean():.2f}}}",
        rf"\newcommand{{\heightSdM}}{{{body.height_m.std():.2f}}}",
        rf"\newcommand{{\nTrials}}{{{int(trials['ALL'])}}}",
    ]
    lines += [
        rf"\newcommand{{\nTrials{name}}}{{{int(trials[family])}}}"
        for family, name in FAMILY_MACROS.items()
        if family in trials
    ]
    return "\n".join(lines) + "\n"


def duration_macros(durations_file: Path) -> str:
    """Median trial duration, total recording time and total samples per foot."""
    trials = pd.read_csv(durations_file)
    lines = [
        rf"\newcommand{{\trialDurationMedianS}}{{{trials.duration_s.median():.0f}}}",
        rf"\newcommand{{\datasetHours}}{{{trials.duration_s.sum() / 3600:.1f}}}",
        rf"\newcommand{{\nSamplesMillion}}{{{trials.n_samples.sum() / 1e6:.1f}}}",
    ]
    return "\n".join(lines) + "\n"


def characterisation_macros(characterisation_dir: Path) -> str:
    """Macros of the insole characterisation (``characterise_insole.py``); a missing part is skipped."""
    import json

    root = Path(characterisation_dir)

    def summary(part: str) -> dict | None:
        path = root / part / "summary.json"
        return json.loads(path.read_text()) if path.exists() else None

    values: dict[str, str] = {}
    if sync := summary("sync"):
        values |= {
            "nSyncTrials": f"{sync['trials']}",
            "sampleRateHz": f"{sync['frequency_hz']['median']:.1f}",
            "packetLossPpm": f"{sync['packet_loss_rate'] * 1e6:.0f}",
            "duplicatePct": f"{sync['duplicate_rate'] * 100:.2f}",
            "pairingMaxMs": f"{sync['left_right_max_abs_ms']['max']:.1f}",
            "plateOffsetMs": f"{sync['plate_offset_abs_ms']['median']:.0f}",
            "plateOffsetIqrMs": f"{sync['plate_offset_iqr_ms']['median']:.0f}",
            "contactCorrelation": f"{sync['median_contact_correlation']:.3f}",
        }
    if tare := summary("tare"):
        values |= {
            "nTares": f"{len(tare['tares'])}",
            "tareRangeMedianN": f"{tare['tare_range_N']['median']:.2f}",
            "tareRangeMaxN": f"{tare['tare_range_N']['max']:.1f}",
            "tareSdMedianN": f"{tare['tare_std_N']['median']:.2f}",
        }
        residuals = root / "tare" / "control_residuals.csv"
        if residuals.exists():
            frame = pd.read_csv(residuals)
            per_foot = frame.groupby([frame.subject, frame.channel.str[0]]).residual_N.sum()
            values |= {
                "preloadMinN": f"{per_foot.min():.0f}",
                "preloadMaxN": f"{per_foot.max():.0f}",
                "nPreloadControls": f"{frame.subject.nunique()}",
            }
    if gain := summary("gain"):
        values |= {
            "nGainTrials": f"{gain['trials']}",
            "gainMedian": f"{gain['gain_total']['median']:.2f}",
            "gainFitResidualN": f"{gain['fit_total']['residual_median_abs_N']:.0f}",
        }
        # Spread of the static load ratio over the trials (both insoles), and its median per foot.
        if (trials := root / "gain" / "static_gain.csv").exists():
            total = pd.read_csv(trials).gain_total
            values |= {"gainLowerQuartile": f"{total.quantile(0.25):.2f}", "gainUpperQuartile": f"{total.quantile(0.75):.2f}"}
        if (feet := root / "gain" / "static_gain_per_foot.csv").exists():
            medians = pd.read_csv(feet).groupby("foot").gain.median()
            values |= {"gainMedianLeft": f"{medians['L']:.2f}", "gainMedianRight": f"{medians['R']:.2f}"}
    if drift := summary("drift"):
        # Relative drift of the two insole clocks accumulated over a trial (characterise_insole.py drift).
        values |= {
            "clockDriftMedianMs": f"{drift['accumulated_drift_ms']['median']:.1f}",
            "clockDriftMaxMs": f"{drift['accumulated_drift_ms']['max']:.1f}",
        }
    if allan := summary("allan"):
        values |= {
            "arwMedian": f"{allan['angle_random_walk_deg_per_sqrt_h']['median']:.2f}",
            "arwMax": f"{allan['angle_random_walk_deg_per_sqrt_h']['max']:.2f}",
            "biasInstabilityMedian": f"{allan['bias_instability_deg_per_h']['median']:.1f}",
            "biasInstabilityMax": f"{allan['bias_instability_deg_per_h']['max']:.1f}",
        }
    if thermal := summary("thermal"):
        # Largest drift one Gauge saw over its own temperature range (not the pooled range).
        slopes = pd.read_csv(root / "thermal" / "thermal_slopes.csv")
        drift = (slopes.slope_N_per_C.abs() * slopes.temperature_range_C).max()
        values |= {
            "tempMinC": f"{thermal['temperature_C']['min']:.1f}",
            "tempMaxC": f"{thermal['temperature_C']['max']:.1f}",
            "thermalSlopeMedianNPerC": f"{thermal['slope_abs_N_per_C']['median']:.3f}",
            "thermalSlopeMaxNPerC": f"{thermal['slope_abs_N_per_C']['max']:.3f}",
            "thermalDriftMaxN": f"{drift:.1f}",
        }
    return "".join(rf"\newcommand{{\{name}}}{{{value}}}" + "\n" for name, value in values.items())


PAPER = Path(__file__).resolve().parent
PER_TARGET = "per_target"
PER_TARGET_RUN = "mlp_per_target_dropout"

# Macro suffix: (evaluation dir under --evaluation, run, reference run, reference dir or None).
# The reference is the model a method is tested against (paired Wilcoxon over the Participants).
METHODS = {
    "Main": ("full_configuration", "mlp", None, None),
    "Transformer": ("full_configuration", "v8", "mlp", None),
    "SabbahLoadsol": ("competitors_mahmud", "sabbah_loadsol", "mlp", None),
    "SabbahRaw": ("competitors_sabbah", "sabbah_raw", "mlp", None),
    "MahmudGauges": ("competitors_mahmud", "mahmud_gauges", "mlp", None),
    "MahmudRaw": ("competitors_mahmud", "mahmud_raw", "mlp", None),
    # One MLP per target (11 networks give the 12 targets), against our MLP and against Ke2KeNet:
    # winsPerTargetMahmudRaw counts the Participants on which the per-target MLP has the lower score.
    "PerTarget": ("per_target", PER_TARGET_RUN, "mlp", None),
    "PerTargetMain": ("per_target", "mlp", None, None),
    "PerTargetMahmudRaw": ("per_target", "mahmud_raw", PER_TARGET_RUN, None),
    # Dedicated models of the minimal configurations, against the full-configuration MLP ...
    "DedicatedTwentyEight": ("minimal_dedicated", "minimal_28ch", "mlp_full", None),
    "DedicatedThirtyFour": ("minimal_dedicated", "minimal_34ch", "mlp_full", None),
    # ... and the Sensor-dropout model restricted to the same channels, against the dedicated one.
    "DropoutTwentyEight": (
        "configurations_minimal_28_34/evaluation",
        "minimal_28ch",
        "minimal_28ch",
        "minimal_dedicated",
    ),
    "DropoutThirtyFour": (
        "configurations_minimal_28_34/evaluation",
        "minimal_34ch",
        "minimal_34ch",
        "minimal_dedicated",
    ),
}
# Rows of the main comparison table: (label, macro suffix of METHODS, or None while the run is running).
# Sabbah's forest trained on the total force of both feet is left out: it predicts no per-foot target.
# Columns of the results table (per component and foot): label, macro suffix of METHODS.
# The last column is the reference of the gap row. Real time: causal and faster than the sample period;
# the Random Forest filters with a zero-phase filter over the whole trial, Ke2KeNet reads future samples.
# In the paper the competitors get the same 78 channels as our MLP; the subset of our sensors closest
# to their own insole is in the supplementary material (EMULATED_TABLE). The gap row is against the
# first column, our MLP.
RESULTS_TABLE = [
    ("MLP (ours)", "Main", True),
    ("MLP, one per target (ours)", "PerTarget", True),
    ("Random Forest [Sabbah], 78 ch.", "SabbahRaw", False),
    ("Ke2KeNet [Mahmud], 78 ch.", "MahmudRaw", False),
]
EMULATED_TABLE = [
    ("MLP (ours)", "Main", True),
    ("Random Forest, loadsol-like", "SabbahLoadsol", False),
    ("Ke2KeNet, load cells", "MahmudGauges", False),
]
MAIN_TABLE = [
    ("MLP (ours)", "Main"),
    ("Random Forest [Sabbah], loadsol-like", "SabbahLoadsol"),
    ("Random Forest [Sabbah], 78 ch.", "SabbahRaw"),
    ("Ke2KeNet [Mahmud], load cells", "MahmudGauges"),
    ("Ke2KeNet [Mahmud], 78 ch.", "MahmudRaw"),
    ("MLP, one per target (ours)", "PerTarget"),
]
SPEEDS = {"Main": "mlp", "PerTarget": "mlp_per_target", "SabbahRaw": "sabbah_raw", "SabbahLoadsol": "sabbah_loadsol", "MahmudGauges": "mahmud_gauges", "MahmudRaw": "mahmud_raw"}
CITES = {"[Sabbah]": r"\cite{sabbah2024grfm}", "[Mahmud]": r"\cite{mahmud2026ke2kenet}"}

# Sensor configurations of the Sensor-dropout MLP, scored by inference only (evaluate_sensor_configurations.py).
# Macro suffix: (label, channels, evaluation dir, run). The first is the reference of the others.
GAUGES_IMUS = "configurations_gauges_imus/mlp_sensor_dropout_w21/evaluation"
MODALITIES_2 = "configurations_modalities_2/mlp_sensor_dropout_w21/evaluation"
MINIMAL = "configurations_minimal_28_34/evaluation"
MODALITY_TABLE = {
    "All": ("All sensors", 78, GAUGES_IMUS, "full"),
    "GaugesAcc": ("Load cells + accelerometers", 54, MODALITIES_2, "gauges_acc"),
    "GaugesGyro": ("Load cells + gyroscopes", 54, MODALITIES_2, "gauges_gyro"),
    "Gauges": ("Load cells only", 30, GAUGES_IMUS, "gauges_only"),
    "Imus": ("IMUs only", 48, GAUGES_IMUS, "imus_only"),
}
MINIMAL_TABLE = {
    "FullMlp": ("All sensors", 78, "minimal_dedicated", "mlp_full"),
    "DropoutThirtyFourCh": ("34 ch., sensor-dropout model", 34, MINIMAL, "minimal_34ch"),
    "DedicatedThirtyFourCh": ("34 ch., dedicated model", 34, "minimal_dedicated", "minimal_34ch"),
    "DropoutTwentyEightCh": ("28 ch., sensor-dropout model", 28, MINIMAL, "minimal_28ch"),
    "DedicatedTwentyEightCh": ("28 ch., dedicated model", 28, "minimal_dedicated", "minimal_28ch"),
}
# The MLP trained without Sensor dropout, scored under the same configurations (--without-sensor-dropout).
NO_DROPOUT = "robustness_no_dropout/evaluation"
# Lost patches (paper/sensor_study.py): both models' scores in their own dirs.
SENSOR_STUDY = "sensor_study/sensor_dropout"
SENSOR_STUDY_NO_DROPOUT = "sensor_study/no_dropout"
# In the order of the text: robustness to a lost patch, then the sensor modalities, then the minimal configurations.
ROBUSTNESS_TABLE = {
    "All": ("All sensors", 78, "full", GAUGES_IMUS),
    "NoHeel": ("Heel patch lost", 58, "no_heel", SENSOR_STUDY),
    "NoMidfoot": ("Midfoot patch lost", 58, "no_midfoot", SENSOR_STUDY),
    "NoForefoot": ("Forefoot patch lost", 58, "no_forefoot", SENSOR_STUDY),
    "NoToes": ("Toes patch lost", 60, "no_toes", SENSOR_STUDY),
    "GaugesAcc": ("Load cells + accelerometers", 54, "gauges_acc", MODALITIES_2),
    "GaugesGyro": ("Load cells + gyroscopes", 54, "gauges_gyro", MODALITIES_2),
    "Gauges": ("Load cells only", 30, "gauges_only", GAUGES_IMUS),
    "Imus": ("IMUs only", 48, "imus_only", GAUGES_IMUS),
    "ThirtyFourCh": ("Minimal, 34 ch.", 34, "minimal_34ch", MINIMAL),
    "TwentyEightCh": ("Minimal, 28 ch.", 28, "minimal_28ch", MINIMAL),
}


# paper/scripts/loading_and_transitions.py on the main run's predictions.
LOADING_TRANSITIONS = "loading_transitions"
FAMILY_LABELS = {  # task family -> label of the per-family table, in the order of the Dataset section
    "balance": "Balance",
    "squat": "Squat",
    "hulahoop": "Hula hoop",
    "lateral_2supports": "Lateral double-support",
    "lateral_1_appui": "Lateral single-support",
    "walk": "Walking",
    "jump": "Jumping",
}


def configuration_rows(evaluation: Path, table: dict) -> dict[str, ConfigurationRow]:
    """The rows of MODALITY_TABLE or MINIMAL_TABLE, their dirs under ``evaluation``."""
    return {name: ConfigurationRow(label, ch, evaluation / d, run) for name, (label, ch, d, run) in table.items()}


def robustness_rows(evaluation: Path) -> dict[str, RobustnessRow]:
    """The rows of ROBUSTNESS_TABLE, their Sensor-dropout dirs under ``evaluation``."""
    return {
        name: RobustnessRow(
            label, ch, run, evaluation / d, evaluation / SENSOR_STUDY_NO_DROPOUT if d == SENSOR_STUDY else None
        )
        for name, (label, ch, run, d) in ROBUSTNESS_TABLE.items()
    }


def with_cites(label: str) -> str:
    """``label`` with its [Author] tags replaced by citations."""
    for key, cite in CITES.items():
        label = label.replace(key, cite)
    return label


CYCLE_RUN = "mlp_sensor_dropout_w21"


def write_numbers(evaluation: Path, output: Path) -> None:
    lines = ["% Generated by paper/make_artefacts.py from paper/evaluation: do not edit.", ""]
    lines.append(characterisation_macros(PAPER / "characterisation"))
    lines.append(imu_calibration_macros(PAPER.parent / "calibration_imu" / "data"))
    lines.append(
        dataset_macros(
            evaluation / "full_configuration",
            PAPER.parent / "insole2grfm" / "participant_sex.json",
            PAPER.parent / "insole2grfm" / "participant_anthropometry.json",
        )
    )
    lines.append(duration_macros(evaluation / "trial_durations.csv"))
    lines.append(number_macros({name: (evaluation / d, run) for name, (d, run, _, _) in METHODS.items()}))
    for name, (directory, run, reference, reference_dir) in METHODS.items():
        if reference is None:
            continue
        wins, p = paired_comparison(
            evaluation / directory,
            run,
            reference=reference,
            reference_dir=evaluation / reference_dir if reference_dir else None,
        )
        lines += [rf"\newcommand{{\wins{name}}}{{{wins}}}", rf"\newcommand{{\p{name}}}{{{format_p(P_VALUES(f'p{name}', p))}}}"]
    main_dir = evaluation / METHODS["Main"][0]
    lines += [component_macros(main_dir, METHODS["Main"][1]), per_foot_rmse_macros(main_dir, METHODS["Main"][1]),
              per_foot_normalised_macros(main_dir, METHODS["Main"][1]),
              reference_sensitivity_macros(), family_macros(main_dir, METHODS["Main"][1]),
              measured_share_macros(main_dir, METHODS["Main"][1])]
    extra = evaluation / LOADING_TRANSITIONS
    lines += [
        loading_macros(pd.read_csv(extra / "loading.csv"), main_dir, METHODS["Main"][1]),
        transition_macros(pd.read_csv(extra / "transitions.csv"), main_dir / "body_weight.csv",
                          pd.read_csv(extra / "airborne_by_family.csv")),
    ]
    for table in (MODALITY_TABLE, MINIMAL_TABLE):
        rows = configuration_rows(evaluation, table)
        reference = next(iter(rows.values()))
        reference = (reference.evaluation_dir, reference.run)
        lines += [configuration_macros(name, row, reference) for name, row in list(rows.items())[1:]]
    if (evaluation / "competitors_sabbah_total").exists():
        lines.append(sabbah_total_macros(evaluation / "competitors_sabbah_total"))
    if (evaluation / NO_DROPOUT).exists():
        rows = robustness_rows(evaluation)
        full = rows["All"]
        lines += [robustness_macros(name, row, full, evaluation / NO_DROPOUT) for name, row in rows.items()]
    if (evaluation / "sensor_study").exists():
        import sensor_study

        lines.append(
            sensor_study.sensor_study_macros(
                evaluation, PAPER / "bom" / "bom_insole.csv",
                PAPER.parent / "insole2grfm" / "tools" / "insole_layout.json",
                evaluation / MINIMAL, evaluation / "minimal_dedicated",
            )
        )
    speed = pd.read_csv(evaluation / "inference_speed" / "speed.csv").set_index(["method", "device"])
    for name, method in (("Main", "mlp"), ("PerTarget", "mlp_per_target")):
        per_prediction_ms = 1000 / speed.loc[(method, "cpu"), "predictions_per_second"]
        lines.append(rf"\newcommand{{\msPerPrediction{name}}}{{{per_prediction_ms:.2f}}}")
    for name, method in SPEEDS.items():
        lines.append(
            rf"\newcommand{{\speedCpu{name}}}{{{format_speed(speed.loc[(method, 'cpu'), 'predictions_per_second'])}}}"
        )
    if ("mahmud_gauges", "cuda") in speed.index:  # measured only on a machine with a GPU
        lines.append(
            rf"\newcommand{{\speedGpuMahmudGauges}}{{{format_speed(speed.loc[('mahmud_gauges', 'cuda'), 'predictions_per_second'])}}}"
        )
    lines.append(rf"\newcommand{{\speedCpuName}}{{{speed.cpu.iloc[0].removeprefix('1 thread, ').replace('(R)', '').replace('(TM)', '')}}}")
    lines.append(per_target_component_macros(evaluation / PER_TARGET))
    lines.append(parameter_macros(evaluation / "inference_speed" / "parameters.csv"))
    (output / "numbers.tex").write_text("\n".join(lines) + "\n")


def write_tables(evaluation: Path, tables: Path) -> None:
    """The four result tables of the paper (robustness only once its evaluation is published)."""
    speed = pd.read_csv(evaluation / "inference_speed" / "speed.csv").set_index(["method", "device"])
    speeds = {
        with_cites(label): format_speed(speed.loc[(SPEEDS[name], "cpu"), "predictions_per_second"])
        if name in SPEEDS
        else None
        for label, name in MAIN_TABLE
    }
    main = [
        (with_cites(label), evaluation / METHODS[name][0], METHODS[name][1]) if name else (with_cites(label), None, None)
        for label, name in MAIN_TABLE
    ]
    (tables / "main_comparison.tex").write_text(comparison_table(main, speeds=speeds))
    main_run = (evaluation / METHODS["Main"][0], METHODS["Main"][1])
    (tables / "components.tex").write_text(component_table(*main_run))
    speed_of = {"PerTargetMain": "Main", "PerTargetMahmudRaw": "MahmudRaw"}

    def columns(table):
        result = []
        for label, name, real_time in table:
            method = SPEEDS.get(speed_of.get(name, name))
            ms = 1000 / speed.loc[(method, "cpu"), "predictions_per_second"] if method in speed.index.get_level_values(0) else None
            result.append(ResultsColumn(with_cites(label), evaluation / METHODS[name][0], METHODS[name][1], ms, real_time,
                                        ours="(ours)" in label, name=name))
        return result

    (tables / "results.tex").write_text(results_table(columns(RESULTS_TABLE)))
    (tables / "results_emulated.tex").write_text(results_table(columns(EMULATED_TABLE)))
    (tables / "results_rmse.tex").write_text(rmse_table(columns(RESULTS_TABLE)))
    (tables / "airborne.tex").write_text(airborne_table(*main_run))
    (tables / "task_families.tex").write_text(family_table(*main_run, FAMILY_LABELS))
    extra = evaluation / LOADING_TRANSITIONS
    (tables / "loading.tex").write_text(loading_table(pd.read_csv(extra / "loading.csv"), *main_run))
    (tables / "transitions.tex").write_text(
        transition_table(pd.read_csv(extra / "transitions.csv"), main_run[0] / "body_weight.csv")
    )
    for filename, table in (("modality_ablation.tex", MODALITY_TABLE), ("minimal_configuration.tex", MINIMAL_TABLE)):
        rows = list(configuration_rows(evaluation, table).values())
        (tables / filename).write_text(configuration_table(rows, reference=(rows[0].evaluation_dir, rows[0].run)))
    if (evaluation / NO_DROPOUT).exists():
        (tables / "robustness.tex").write_text(
            robustness_table(list(robustness_rows(evaluation).values()), without=evaluation / NO_DROPOUT)
        )
    if (evaluation / "sensor_study").exists():
        import sensor_study

        (tables / "modality_components.tex").write_text(
            sensor_study.modality_component_table(evaluation / SENSOR_STUDY)
        )


def write_provenance(output: Path) -> None:
    """Record, next to the artefacts, the release commit and inputs they were made from."""
    repo = PAPER.parent

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True).stdout.strip()

    # Uncommitted changes to what the artefacts are made of: tracked or untracked, code or tables.
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", "paper", "insole2grfm/participant_sex.json",
                "insole2grfm/participant_anthropometry.json")
    dirty = [line for line in dirty.splitlines() if " paper/artefacts/" not in line]
    lines = [f"commit: {git('rev-parse', 'HEAD') or 'unknown'}"]
    lines += [f"uncommitted: {line}" for line in dirty]
    (output / "PROVENANCE").write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--evaluation", type=Path, default=PAPER / "evaluation")
    parser.add_argument("--output", type=Path, default=PAPER / "artefacts")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Full CV results root holding the main model's predictions (not distributed); "
        "when given, the per-task-family cycle figures are drawn too",
    )
    args = parser.parse_args(argv)
    evaluation, output = args.evaluation, args.output
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)

    # Two passes: the first records every raw p value, the second writes them Holm-adjusted (pvalues.py).
    # The tables the paper does not use (main_comparison, modality_ablation, minimal_configuration) keep raw p.
    P_VALUES.reset()
    write_numbers(evaluation, output)
    write_tables(evaluation, output / "tables")
    P_VALUES.adjust()
    write_numbers(evaluation, output)
    write_tables(evaluation, output / "tables")
    scripts = PAPER / "scripts"
    subprocess.run(
        [
            sys.executable,
            str(scripts / "plot_speed_vs_error.py"),
            str(evaluation / "competitors_mahmud" / "per_participant.csv"),
            str(evaluation / "inference_speed" / "speed.csv"),
            str(output / "figures" / "speed_vs_error.pdf"),
        ],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(scripts / "plot_pareto.py"), str(evaluation / "ranking" / "configurations.csv.gz"),
         str(output / "figures" / "pareto.pdf")],
        check=True,
    )
    subprocess.run(
        [sys.executable, str(scripts / "plot_participant_scores.py"), str(evaluation),
         str(output / "figures" / "participant_scores.pdf")],
        check=True,
    )
    if args.predictions:
        subprocess.run(
            [sys.executable, str(scripts / "plot_time_trace.py"), str(args.predictions / CYCLE_RUN / "all"),
             str(evaluation / "full_configuration" / "per_participant.csv"), str(output / "figures" / "time_trace.pdf")],
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(scripts / "plot_grfm_cycles.py"),
                str(args.predictions / CYCLE_RUN),
                str(output / "figures" / "cycles"),
                "--summary",
                str(evaluation / "full_configuration" / "summary.csv"),
                "--run",
                "mlp",
            ],
            check=True,
        )
    write_provenance(output)
    print(f"artefacts -> {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
