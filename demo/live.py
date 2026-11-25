#!/usr/bin/env python3
"""Real-time demo: insole acquisition, per-foot GRFM estimation and pressure heatmaps, in Rerun.

``$CHECKPOINT`` is a trained model, e.g. the Full CV fold of P05 (the median fold)::

    python tools/download_models.py
    CHECKPOINT=models/mlp_sensor_dropout_w21/all/fold_P05/models/mlp_test_P05.pt

Live, from the receiver (Tare first with ``python -m demo.tare``)::

    python -m demo.live --port /dev/ttyACM0 --tare-file demo/tare.json --body-weight 650 \\
        --checkpoint $CHECKPOINT

Every live session is saved to ``demo/sessions/<date>``: the raw serial capture
(``.bin``) and the Rerun recording (``.rrd``). Replay a capture without the
insoles (same Tare file)::

    python -m demo.live --replay demo/sessions/<date>.bin --tare-file demo/tare.json --body-weight 650 --checkpoint ...

Or play a prepared dataset trial, with its force-plate GRFM drawn under the estimate::

    python -m demo.live --trial <dataset>/P05_walk.npz --body-weight 744 --checkpoint ...
"""

from __future__ import annotations

import os

# One thread each for numpy and torch: the heatmap and the 200 Hz model would
# otherwise fight over every core.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import argparse
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from demo.estimator import GRFMEstimator
from demo.stream import (
    CaptureSource,
    SerialSource,
    imu_calibrations,
    load_tares,
    packet_samples,
    threaded,
    trial_samples,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "insole2grfm"))
from insole_heatmap import TRAIL_S, InsoleHeatmap  # noqa: E402

APP = "open_insole_live"
SESSIONS = Path(__file__).resolve().parent / "sessions"
HEATMAP_HZ = 15.0  # wall clock: the heatmap never holds the GRFM back
WINDOW_S = 10.0
FEET = {"L": "left", "R": "right"}
# The insole blue for the estimate (as the heatmap), neutral greys for the references.
ESTIMATE = (0, 113, 227)
PLATE = (29, 29, 31)
INSOLE_SUM = (174, 174, 178)
COMPONENTS = {
    "Fx": (255, 69, 58), "Fy": (48, 209, 88), "Fz": ESTIMATE,
    "Mx": (255, 159, 10), "My": (191, 90, 242), "Mz": (100, 210, 255),
}


def blueprint() -> rrb.Blueprint:
    recent = rrb.VisibleTimeRange(
        "time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-WINDOW_S),
        end=rrb.TimeRangeBoundary.cursor_relative(),
    )

    def series(path: str, name: str) -> rrb.TimeSeriesView:
        return rrb.TimeSeriesView(
            origin=path, name=name, time_ranges=recent,
            axis_y=rrb.ScalarAxis(zoom_lock=False), plot_legend=rrb.PlotLegend(corner="LeftTop"),
        )

    def foot(side: str) -> rrb.Vertical:
        label = FEET[side].capitalize()
        return rrb.Vertical(
            series(f"/{FEET[side]}/Fz", f"{label} foot · vertical force Fz (N)"),
            series(f"/{FEET[side]}/forces", f"{label} · shear forces (N)"),
            series(f"/{FEET[side]}/moments", f"{label} · moments (N·m)"),
            row_shares=[2, 1, 1],
        )

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(origin="/insoles", name="Insole pressure", background=(255, 255, 255)),
            foot("L"),
            foot("R"),
            column_shares=[1.1, 1, 1],
        ),
        rrb.TimePanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.BlueprintPanel(state="collapsed"),
        collapse_panels=True,
    )


def log_styles(with_plate: bool) -> None:
    for side, foot in FEET.items():
        rr.log(f"{foot}/Fz/estimate", rr.SeriesLines(colors=ESTIMATE, widths=2.5, names="insoles + model"), static=True)
        rr.log(f"{foot}/Fz/insole_sum", rr.SeriesLines(colors=INSOLE_SUM, widths=1.5, names="sum of load cells"),
               static=True)
        if with_plate:
            rr.log(f"{foot}/Fz/plate", rr.SeriesLines(colors=PLATE, widths=1.5, names="force plate"), static=True)
        for group, names in (("forces", ("Fx", "Fy")), ("moments", ("Mx", "My", "Mz"))):
            for name in names:
                rr.log(f"{foot}/{group}/{name}", rr.SeriesLines(colors=COMPONENTS[name], widths=2, names=name),
                       static=True)
                if with_plate:
                    rr.log(f"{foot}/{group}/{name}_plate",
                           rr.SeriesLines(colors=COMPONENTS[name] + (90,), widths=1, names=f"{name} plate"),
                           static=True)


def log_grfm(targets: list[str], grfm: np.ndarray, suffix: str = "") -> None:
    for target, value in zip(targets, grfm):
        if not np.isfinite(value):
            continue
        component, side = target.split("_")
        foot = FEET[side]
        if component == "Fz":
            path = f"{foot}/Fz/{'plate' if suffix else 'estimate'}"
        else:
            group = "forces" if component.startswith("F") else "moments"
            path = f"{foot}/{group}/{component}{suffix}"
        rr.log(path, rr.Scalars(float(value)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port", help="serial port of the receiver, e.g. /dev/ttyACM0")
    source.add_argument("--replay", type=Path,
                        help="capture (.bin) of an earlier session, or an acquisition's dataSemelle.bin")
    source.add_argument("--trial", type=Path, help="prepared dataset trial (.npz), with its force plates")
    parser.add_argument("--checkpoint", type=Path, required=True, help="trained model (.pt)")
    parser.add_argument("--body-weight", type=float, required=True, help="body weight of the wearer, in N")
    parser.add_argument("--tare-file", type=Path, nargs="+",
                        help="fx29_tare_v1 file of both feet (demo.tare), or one file per foot")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed (0: as fast as possible)")
    parser.add_argument("--no-save", action="store_true", help="do not save the live session")
    parser.add_argument("--save", type=Path, help="Rerun recording (.rrd) to write")
    parser.add_argument("--headless", action="store_true", help="no viewer: only write --save")
    args = parser.parse_args()

    if args.body_weight <= 0:
        parser.error("--body-weight is in N and must be > 0 (e.g. 65 kg -> 638)")
    if args.trial is None:
        if args.tare_file is None:
            parser.error("--tare-file is needed with the insoles (python -m demo.tare)")
        load_tares(args.tare_file)

    estimator = GRFMEstimator(args.checkpoint, args.body_weight)
    heatmap = InsoleHeatmap(mirror=True)
    layout_names = list(heatmap_feature_names())

    rr.init(APP, spawn=False)
    sinks = []
    if not args.headless:
        # The viewer installed with rerun-sdk, even when the venv is not activated.
        viewer = Path(sys.executable).with_name("rerun")
        rr.spawn(connect=False, default_blueprint=blueprint(),
                 executable_path=str(viewer) if viewer.exists() else None)
        sinks.append(rr.GrpcSink())
    save = args.save
    if args.port and not args.no_save and save is None:
        SESSIONS.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        save, capture = SESSIONS / f"{stamp}.rrd", SESSIONS / f"{stamp}.bin"
    else:
        capture = None
    if save is not None:
        sinks.append(rr.FileSink(str(save)))
    rr.set_sinks(*sinks, default_blueprint=blueprint())
    log_styles(with_plate=args.trial is not None)

    status: dict = {}
    if args.trial:
        samples = trial_samples(args.trial, estimator.channels, args.speed)
    else:
        chunks = SerialSource(args.port, args.baud, capture) if args.port else CaptureSource(args.replay, args.speed)
        samples = ((at, channels, None) for at, channels in packet_samples(chunks, imu_calibrations(), status))
    items = threaded(samples)

    trail_length = int(TRAIL_S * 200)
    trails = {side: deque(maxlen=trail_length) for side in FEET}
    loads = {side: np.zeros(len(heatmap.columns[side])) for side in FEET}
    last_frame, last_report, count = 0.0, time.monotonic(), 0
    print(f"Model: {args.checkpoint.name}, {len(estimator.targets)} outputs, window {estimator.window}"
          + (f"\nSaving the session to {save}" + (f" and {capture}" if capture else "") if save else ""))
    while True:
        item = items.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        at, channels, truth = item
        count += 1
        rr.set_time("time", duration=at)
        row = np.array([channels.get(name, 0.0) for name in layout_names], np.float32)
        for side, foot in FEET.items():
            loads[side] = heatmap.loads(row, side)
            trails[side].append(heatmap.cop_mm(loads[side][None], side)[0])
            rr.log(f"{foot}/Fz/insole_sum", rr.Scalars(float(loads[side].sum())))
        grfm = estimator.push(channels)
        if grfm is not None:
            log_grfm(estimator.targets, grfm)
        if truth is not None:
            log_grfm(estimator.targets, truth, suffix="_plate")
        if time.monotonic() - last_frame >= 1.0 / HEATMAP_HZ:
            last_frame = time.monotonic()
            frame = heatmap.render(loads, {side: np.array(trails[side]) for side in FEET})
            rr.log("insoles", rr.Image(frame).compress(jpeg_quality=90))
        now = time.monotonic()
        if now - last_report > 5.0:
            rate = count / (now - last_report)
            print(f"{rate:5.0f} samples/s, backlog {items.qsize()}"
                  + (f", unpaired packets dropped {status.get('dropped', 0)}" if status else ""))
            last_report, count = now, 0
    print("End of the source.")
    return 0


def heatmap_feature_names() -> list[str]:
    import json

    layout = Path(__file__).resolve().parents[1] / "insole2grfm" / "tools" / "insole_layout.json"
    return json.loads(layout.read_text())["feature_names"]


if __name__ == "__main__":
    raise SystemExit(main())
