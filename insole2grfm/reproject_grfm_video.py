#!/usr/bin/env python3
"""Reproject the reference and predicted GRFM of a trial onto its video.

On every video frame, for each foot, the ground reaction force is drawn as an
arrow from its Centre of pressure on the Sole plane (1 body weight = 0.5 m),
and its Free moment as an arc around the foot's vertical: the reference (force
plates) in blue, the model (insoles) in orange. The reference is drawn only
where the foot is Measured or Airborne; a Masked foot gets "no reference".
Nothing is drawn under 20 N of vertical force (no Centre of pressure).

Two commands, writing under ``--output-dir``:

- ``calibrate SESSION --camera camN``: calibrates the session camera
  (``video_calibration.py``) from ``calibration/<SESSION>_<camN>_corners.json``
  (``tools/click_plate_corners.py``) and the InstantHMR keypoints
  ``keypoints/<SESSION>_<trial>_<camN>.npz`` (``tools/instanthmr_feet.py``) of
  its trials, into ``calibration/<SESSION>_<camN>.json`` plus a check image;
- ``render TRIAL --camera camN``: the fold checkpoint of the filmed
  Participant (full CV) predicts every sample of the trial, Masked ones
  included (``<TRIAL>/predictions_all_samples.csv``), then writes
  ``<TRIAL>/<TRIAL>_<camN>.mp4`` (video + Fz / Free-moment strip) and, with
  ``--stills T [T ...]`` (trial time, s), cropped stills and their row.

Example::

    python reproject_grfm_video.py calibrate P02 --camera cam2 \
      --output-dir ../docs/analysis/figures/2026-10-30-grfm-video-mlp
    python reproject_grfm_video.py render P02_squat --camera cam2 \
      --run-dir ../results/full_cv_v3.7.1/mlp_sensor_dropout_w21/all --data-dir ../dataset_v3.7.1 \
      --output-dir ../docs/analysis/figures/2026-10-30-grfm-video-mlp
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluate_body_weight_modes import fold_inputs, load_model, run_arguments
from grfm_schema import FEET, STATE_AIRBORNE, STATE_MASKED, STATE_MEASURED
from train_lstm_two_feet_masked import (
    BodyWeightMode,
    MaskedWindowDataset,
    SensorDropout,
    predict,
    subject_identity_from_session,
    subject_session_from_batch,
)
from video_calibration import (
    Camera, TrialVideo, calibrate, calibrate_from_feet, nearest_rows, plate_corners, trial_mocap,
)

WORKSPACE = Path(__file__).resolve().parents[1]
MANIP_DIR = WORKSPACE / "manip"
TARGETS = [f"{component}_{side}" for side in "LR" for component in ("Fx", "Fy", "Fz", "Mx", "My", "Mz")]


def trial_dir(trial: str) -> Path:
    """``P04b_jump2`` -> ``manip/P04b/jump2``; a segment ``P14b_lateral1__lateral_1support`` -> its
    recording ``manip/P14b/lateral1``."""
    session = subject_session_from_batch(trial)
    return MANIP_DIR / session / trial[len(session) + 1:].split("__")[0]


def predict_trial(trial: str, run_dir: Path, data_dir: Path | None, device: torch.device) -> pd.DataFrame:
    """Reference and prediction of every sample of ``trial``, Masked ones included.

    The checkpoint is the fold whose test Participant is the trial's, with the
    full Sensor configuration and the body weight provided, as in the run's own
    ``predictions.csv``. Columns: ``time_qtm_s``, ``state_L``, ``state_R``, then
    ``<target>`` (NaN where not a target) and ``pred_<target>`` for the 12 targets.
    """
    session = subject_session_from_batch(trial)
    fold = run_dir / f"fold_{subject_identity_from_session(session)}"
    metadata = json.loads((fold / "metadata.json").read_text(encoding="utf-8"))
    run = run_arguments(metadata)
    saved = torch.load(next((fold / "models").glob("*.pt")), map_location="cpu", weights_only=False)
    data_dir = data_dir or Path(metadata["data_dir"])
    items, targets, target_indexes, _ = fold_inputs(run, saved, data_dir, [trial])
    if not items or targets != TARGETS:
        raise ValueError(f"{trial}: not found in {data_dir} or unexpected targets {targets}")
    item = items[0]
    truth_mask = item.mask.copy()
    # Every sample becomes a window centre: the model also predicts the Masked ones.
    item.mask = np.ones_like(item.mask)
    body_weight = BodyWeightMode.from_statistics(saved["feature_names"], saved["x_mean"], saved["x_std"])
    dataset = MaskedWindowDataset(
        [item], saved["x_mean"], saved["x_std"], saved["y_mean"], saved["y_std"], target_indexes,
        saved["window_samples"], not saved["causal"], saved["zero_target_weight"],
        target_scaling=saved["target_scaling"], body_weight=body_weight,
        # The engine's evaluation: full configuration, Inactive sensors removed.
        sensor_dropout=(
            SensorDropout.from_feature_names(saved["feature_names"]) if metadata["sensor_dropout"] else None
        ),
    )
    model = load_model(run, saved, device)
    outputs = predict(
        model, dataset, DataLoader(dataset, 1024, shuffle=False), saved["y_mean"], saved["y_std"],
        device, target_indexes, saved["hard_airborne_zero"],
    )
    rows = np.array([row for _, row in dataset.index])
    times = np.load(data_dir / f"{trial}.npz")
    frame = pd.DataFrame({
        "time_s": item.time_s[rows],
        "time_qtm_s": (times["target_time_qtm_s_L"][rows] + times["target_time_qtm_s_R"][rows]) / 2.0,
        "state_L": item.state_l[rows],
        "state_R": item.state_r[rows],
    })
    truth, predicted = outputs[0], outputs[1]
    for index, target in enumerate(targets):
        frame[target] = np.where(truth_mask[rows, index], truth[:, index], np.nan)
        frame[f"pred_{target}"] = predicted[:, index]
    return frame


# Okabe-Ito blue / orange, BGR.
REFERENCE_COLOUR = (178, 114, 0)
MODEL_COLOUR = (0, 159, 230)
OUTLINE_COLOUR = (25, 25, 25)
# (source, column prefix, colour, legend, line thickness): the thin model line is drawn over the
# thick reference, so both stay visible where they agree.
SOURCES = (
    ("ref", "", REFERENCE_COLOUR, "Reference (force plates)", 8),
    ("mlp", "pred_", MODEL_COLOUR, "MLP (insoles)", 3),
)
ARROW_MM_PER_BODY_WEIGHT = 500.0
MIN_VERTICAL_FORCE_N = 20.0
ARC_RADIUS_MM = 130.0
ARC_DEGREES_PER_NM = 40.0
SAMPLE_TOLERANCE_S = 0.010
STRIP_HEIGHT = 300
STRIP_WINDOW_S = 6.0
CALIBRATION_TRIALS = (
    "squat", "squat1", "squat2", "walk", "walk1", "walk2", "jump", "jump1", "jump2", "lateral_1_appui", "lateral2_1_appui",
)
BODY_WEIGHT_CSV = WORKSPACE / "results" / "full_cv_v3.7.1" / "evaluation" / "full_configuration" / "body_weight.csv"


def calibrate_session(session: str, camera: str, output_dir: Path) -> None:
    """``calibration/<session>_<camera>.json`` from every trial with keypoints and a synchronised mocap."""
    trials = [
        name for name in CALIBRATION_TRIALS
        if (output_dir / "keypoints" / f"{session}_{name}_{camera}.npz").is_file()
        and (MANIP_DIR / session / name / "synchronized_v2" / "mocap_synchronisee.csv").is_file()
    ]
    if not trials:
        raise SystemExit(f"{session} {camera}: no trial with both keypoints and synchronized_v2")
    calibration = output_dir / "calibration"
    result, local, offsets, report = calibrate(
        [MANIP_DIR / session / name for name in trials], camera,
        [output_dir / "keypoints" / f"{session}_{name}_{camera}.npz" for name in trials],
        calibration / f"{session}_{camera}_corners.json",
    )
    result.save(
        calibration / f"{session}_{camera}.json", video_offset_s=offsets, trials=trials,
        keypoint_local_mm=local.tolist(), report=report,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "hypotheses"}))
    check_image(MANIP_DIR / session / trials[0], camera, result, offsets[trials[0]],
                calibration / f"{session}_{camera}_check.jpg")


def calibrate_second_camera(session: str, camera: str, primary: str, output_dir: Path, extra: list[str]) -> None:
    """``calibration/<session>_<camera>.json`` without clicks, from the feet seen by both cameras.

    Uses the trials of ``session`` (and of ``extra`` sessions of the same day,
    filmed by the same still cameras) that have InstantHMR data on ``camera``
    in ``mesh2/`` or ``keypoints/``, the keypoint offsets of the ``primary`` camera calibration,
    and its video offsets as starting points.
    """
    calibration = output_dir / "calibration"
    videos, offsets, local = [], {}, None
    for current in [session, *extra]:
        saved = json.loads((calibration / f"{current}_{primary}.json").read_text(encoding="utf-8"))
        local = np.asarray(saved["keypoint_local_mm"]) if local is None else local
        paths = {path.name: path for folder in ("mesh2", "keypoints")
                 for path in sorted((output_dir / folder).glob(f"{current}_*_{camera}.npz"))}
        for path in paths.values():
            name = path.name[len(current) + 1: -len(f"_{camera}.npz")]
            if name not in saved["video_offset_s"]:
                continue
            key = name if current == session else f"{current}/{name}"
            videos.append(TrialVideo.load(MANIP_DIR / current / name, camera, path, key))
            offsets[key] = float(saved["video_offset_s"][name])
    if not videos:
        raise SystemExit(f"{session} {camera}: no trial in mesh2/ calibrated on {primary}")
    capture = cv2.VideoCapture(str(MANIP_DIR / session / videos[0].name.split("/")[-1] / f"{camera}_manip.mp4"))
    image_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    focal = json.loads((calibration / f"{session}_{primary}.json").read_text(encoding="utf-8"))["focal_px"]
    result, found, report = calibrate_from_feet(videos, local, offsets, image_size, focal)
    trials = [name for name in found if "/" not in name]
    result.save(
        calibration / f"{session}_{camera}.json", video_offset_s=found, trials=list(found),
        keypoint_local_mm=local.tolist(), report=report, calibrated_from=primary,
    )
    print(json.dumps({"focal_px": round(result.focal_px, 1), **report}))
    check_image(MANIP_DIR / session / trials[0], camera, result, found[trials[0]],
                calibration / f"{session}_{camera}_check.jpg")


def check_image(trial_path: Path, camera: str, result: Camera, offset: float, path: Path) -> None:
    """Middle frame of a trial with the projected plate corners (green) and foot markers (red L, blue R)."""
    from data_preparation_utils import marker_arrays

    sync = trial_path / "synchronized_v2"
    mocap = pd.read_csv(sync / "mocap_synchronisee.csv")
    video = pd.read_csv(sync / f"{camera}_manip_frames_synchronises.csv")
    frame = len(video) // 2
    row = nearest_rows(mocap["time_qtm_absolute_s"].to_numpy(),
                       np.array([video["time_qtm_absolute_s"][frame] - offset]), 0.02)[0]
    capture = cv2.VideoCapture(str(trial_path / f"{camera}_manip.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    _, image = capture.read()
    for point in result.project(plate_corners(trial_path)):
        cv2.circle(image, tuple(np.round(point).astype(int)), 4, (0, 220, 0), -1)
    for side, colour in zip(FEET, ((0, 0, 255), (255, 0, 0))):
        for values in marker_arrays(mocap, side).values():
            point = result.project(values[row])
            if row >= 0 and np.isfinite(point).all():
                cv2.circle(image, tuple(np.round(point).astype(int)), 3, colour, -1)
    cv2.imwrite(str(path), image)


def sole_offset_mm(predictions: pd.DataFrame, origin: np.ndarray, plates: np.ndarray) -> float:
    """Height of the marker-cluster origin above the plate surface while Measured (median, both feet)."""
    heights = [
        origin[predictions[f"state_{side}"].to_numpy() == STATE_MEASURED, index, 2] for index, side in enumerate(FEET)
    ]
    return float(np.nanmedian(np.concatenate(heights))) - plates[:, 2].mean()


def centre_of_pressure(force: np.ndarray, moment_nm: np.ndarray, sole_mm: float):
    """Centre of pressure (mm, foot frame, on z = -sole_mm) and Free moment (N m) of a GRFM about the origin.

    With P = (x, y, -h): M_P = M_O - P x F has no x, y component at the CoP.
    NaN under ``MIN_VERTICAL_FORCE_N``.
    """
    fx, fy, fz = force.T
    mx, my, mz = (moment_nm * 1000.0).T
    loaded = fz >= MIN_VERTICAL_FORCE_N
    safe = np.where(loaded, fz, np.nan)
    x = -(my + sole_mm * fx) / safe
    y = (mx - sole_mm * fy) / safe
    free = (mz - (x * fy - y * fx)) / 1000.0
    return np.column_stack((x, y, np.full_like(x, -sole_mm))), free


def grfm_in_lab(predictions: pd.DataFrame, prefix: str, side: str, rotation, origin, sole_mm: float):
    """Lab CoP (n, 3), lab force (n, 3) and Free moment (n,) of one foot for one source."""
    force = predictions[[f"{prefix}{name}_{side}" for name in ("Fx", "Fy", "Fz")]].to_numpy()
    moment = predictions[[f"{prefix}{name}_{side}" for name in ("Mx", "My", "Mz")]].to_numpy()
    cop, free = centre_of_pressure(force, moment, sole_mm)
    cop_lab = np.einsum("nij,nj->ni", rotation, cop) + origin
    return cop_lab, np.einsum("nij,nj->ni", rotation, force), free


def draw_arrow(image, start, end, colour, thickness):
    if not (np.isfinite(start).all() and np.isfinite(end).all()):
        return
    start, end = tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int))
    length = max(np.hypot(end[0] - start[0], end[1] - start[1]), 1.0)
    tip = min(0.3, (12.0 + 2.0 * thickness) / length)
    cv2.arrowedLine(image, start, end, OUTLINE_COLOUR, thickness + 3, cv2.LINE_AA, tipLength=tip)
    cv2.arrowedLine(image, start, end, colour, thickness, cv2.LINE_AA, tipLength=tip)


def draw_free_moment(image, camera: Camera, centre, rotation, free_nm, colour, thickness):
    """Arc around the foot vertical at the CoP, its angle proportional to the Free moment."""
    span = np.clip(free_nm * ARC_DEGREES_PER_NM, -330.0, 330.0)
    if not np.isfinite(span) or abs(span) < 30.0:
        return
    angles = np.radians(np.linspace(0.0, span, max(int(abs(span) / 6), 4)))
    local = np.column_stack((ARC_RADIUS_MM * np.cos(angles), ARC_RADIUS_MM * np.sin(angles), np.zeros_like(angles)))
    points = camera.project(local @ rotation.T + centre)
    if not np.isfinite(points).all():
        return
    polyline = np.round(points).astype(np.int32)
    # Thinner than the force arrows: the arc is seen almost edge-on from the cameras.
    thickness = max(thickness // 2, 2)
    cv2.polylines(image, [polyline], False, OUTLINE_COLOUR, thickness + 2, cv2.LINE_AA)
    cv2.polylines(image, [polyline], False, colour, thickness, cv2.LINE_AA)
    if abs(span) >= 60.0:
        draw_arrow(image, points[-2], points[-1] + 2.0 * (points[-1] - points[-2]), colour, thickness)


def put_label(image, text, origin, colour=(255, 255, 255), scale=0.6):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, OUTLINE_COLOUR, 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1 if scale < 0.7 else 2, cv2.LINE_AA)


def draw_legend(image):
    for row, (_, _, colour, label, thickness) in enumerate(SOURCES):
        y = 30 + 28 * row
        cv2.line(image, (20, y - 6), (60, y - 6), OUTLINE_COLOUR, thickness + 3, cv2.LINE_AA)
        cv2.line(image, (20, y - 6), (60, y - 6), colour, thickness, cv2.LINE_AA)
        put_label(image, label, (70, y))


class Scene:
    """Everything one trial needs per video frame, in arrays over the prediction samples."""

    def __init__(self, trial: str, camera_name: str, output_dir: Path, predictions: pd.DataFrame,
                 offset: float | None = None):
        """``offset``: the video offset (s) to use where the calibration has none for this trial."""
        session = subject_session_from_batch(trial)
        self.trial_path = trial_dir(trial)
        saved = json.loads((output_dir / "calibration" / f"{session}_{camera_name}.json").read_text())
        self.camera = Camera.load(output_dir / "calibration" / f"{session}_{camera_name}.json")
        name = self.trial_path.name
        if name not in saved["video_offset_s"] and offset is None:
            raise SystemExit(f"{trial}: no video offset in the {session} {camera_name} calibration; extract its "
                             "keypoints (tools/instanthmr_feet.py) and calibrate the session again")
        self.offset = float(saved["video_offset_s"].get(name, offset))
        sync = self.trial_path / "synchronized_v2"
        mocap = pd.read_csv(sync / "mocap_synchronisee.csv", usecols=["time_qtm_s", "time_qtm_absolute_s"])
        mocap_times, rotation, origin, _ = trial_mocap(self.trial_path)
        self.predictions = predictions
        # Sample times on the QTM absolute clock of the video map.
        self.times = np.interp(predictions["time_qtm_s"], mocap["time_qtm_s"], mocap["time_qtm_absolute_s"])
        rows = nearest_rows(mocap_times, self.times, SAMPLE_TOLERANCE_S)
        valid = rows >= 0
        self.rotation = np.full((len(rows), 2, 3, 3), np.nan)
        self.origin = np.full((len(rows), 2, 3), np.nan)
        self.rotation[valid], self.origin[valid] = rotation[rows[valid]], origin[rows[valid]]
        self.sole_mm = sole_offset_mm(predictions, self.origin, plate_corners(self.trial_path))
        weights = pd.read_csv(BODY_WEIGHT_CSV).set_index("subject")["body_weight_N"]
        self.scale = ARROW_MM_PER_BODY_WEIGHT / float(weights[session])
        self.place_grfm()
        video = pd.read_csv(sync / f"{camera_name}_manip_frames_synchronises.csv")
        self.frame_times = video["time_qtm_absolute_s"].to_numpy() - self.offset
        self.frame_samples = nearest_rows(self.times, self.frame_times, SAMPLE_TOLERANCE_S)
        self.video_path = self.trial_path / f"{camera_name}_manip.mp4"

    def place_grfm(self) -> None:
        """Centre of pressure, force and Free moment of each source and foot in the lab, from the
        foot frames (again after a caller has filled foot frames the mocap lacks)."""
        self.grfm = {}
        for source, prefix, *_ in SOURCES:
            for index, side in enumerate(FEET):
                cop, force, free = grfm_in_lab(
                    self.predictions, prefix, side, self.rotation[:, index], self.origin[:, index], self.sole_mm
                )
                if source == "ref":
                    shown = np.isin(self.predictions[f"state_{side}"].to_numpy(), (STATE_MEASURED, STATE_AIRBORNE))
                    cop[~shown], force[~shown], free[~shown] = np.nan, np.nan, np.nan
                self.grfm[source, side] = (cop, force, free)

    def draw(self, image, frame: int) -> None:
        sample = self.frame_samples[frame] if frame < len(self.frame_samples) else -1
        draw_legend(image)
        if sample < 0:
            return
        for index, side in enumerate(FEET):
            rotation = self.rotation[sample, index]
            if not np.isfinite(rotation).all():
                continue
            for source, _, colour, _, thickness in SOURCES:
                cop, force, free = (values[sample] for values in self.grfm[source, side])
                if not np.isfinite(cop).all():
                    continue
                start, end = self.camera.project(np.stack((cop, cop + self.scale * force)))
                draw_free_moment(image, self.camera, cop, rotation, free, colour, thickness)
                draw_arrow(image, start, end, colour, thickness)
            if self.predictions[f"state_{side}"].iat[sample] == STATE_MASKED:
                anchor = self.camera.project(self.origin[sample, index])
                if np.isfinite(anchor).all():
                    put_label(image, "no reference", tuple(np.round(anchor + (-50, 40)).astype(int)), scale=0.55)


def series(scene: Scene, source: str, side: str, quantity: str) -> np.ndarray:
    if quantity == "Fz":
        values = scene.predictions[f"{'' if source == 'ref' else 'pred_'}Fz_{side}"].to_numpy().copy()
        if source == "ref":
            values[~np.isin(scene.predictions[f"state_{side}"], (STATE_MEASURED, STATE_AIRBORNE))] = np.nan
        return values
    return scene.grfm[source, side][2]


def strip_panels(scene: Scene) -> list[tuple[str, str, str, float, float]]:
    """(title, side, quantity, low, high) of the four strip panels."""
    panels = []
    for quantity, unit in (("Fz", "N"), ("Tz", "N m")):
        values = np.concatenate([series(scene, source, side, quantity) for source, *_ in SOURCES for side in FEET])
        low, high = np.nanpercentile(values, [0.5, 99.5])
        pad = 0.1 * max(high - low, 1.0)
        for side in FEET:
            name = "vertical force" if quantity == "Fz" else "free moment"
            panels.append((f"{'Left' if side == 'L' else 'Right'} foot {name} ({unit})", side, quantity, low - pad, high + pad))
    return panels


def draw_strip(scene: Scene, panels, width: int, now: float) -> np.ndarray:
    strip = np.full((STRIP_HEIGHT, width, 3), 255, np.uint8)
    panel_width, panel_height = width // 2, STRIP_HEIGHT // 2
    window = (scene.times >= now - STRIP_WINDOW_S / 2) & (scene.times <= now + STRIP_WINDOW_S / 2)
    for number, (title, side, quantity, low, high) in enumerate(panels):
        x0, y0 = (number % 2) * panel_width, (number // 2) * panel_height
        left, right, top, bottom = x0 + 50, x0 + panel_width - 15, y0 + 24, y0 + panel_height - 12

        def to_px(times, values):
            u = left + (times - (now - STRIP_WINDOW_S / 2)) / STRIP_WINDOW_S * (right - left)
            v = bottom - (values - low) / (high - low) * (bottom - top)
            return np.column_stack((u, v))

        state = scene.predictions[f"state_{side}"].to_numpy()[window]
        times = scene.times[window]
        for code, shade in ((STATE_MASKED, (215, 215, 215)), (STATE_AIRBORNE, (245, 232, 220))):
            for start, stop in runs(state == code):
                u = to_px(times[[start, stop - 1]], np.zeros(2))[:, 0].astype(int)
                cv2.rectangle(strip, (u[0], top), (u[1], bottom), shade, -1)
        cv2.rectangle(strip, (left, top), (right, bottom), (120, 120, 120), 1)
        if low < 0 < high:
            zero = int(to_px(np.array([now]), np.array([0.0]))[0, 1])
            cv2.line(strip, (left, zero), (right, zero), (170, 170, 170), 1)
        for source, _, colour, _, thickness in SOURCES:
            values = series(scene, source, side, quantity)[window]
            for start, stop in runs(np.isfinite(values)):
                points = np.round(to_px(times[start:stop], values[start:stop])).astype(np.int32)
                cv2.polylines(strip, [points], False, colour, max(thickness // 2, 1), cv2.LINE_AA)
        cv2.line(strip, ((left + right) // 2, top), ((left + right) // 2, bottom), (40, 40, 40), 1)
        cv2.putText(strip, title, (left, y0 + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        for value in (low, high):
            cv2.putText(strip, f"{value:.0f}", (x0 + 4, int(to_px(np.array([now]), np.array([value]))[0, 1]) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    cv2.putText(strip, "grey: masked (no reference)   tinted: airborne", (width - 360, STRIP_HEIGHT - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1, cv2.LINE_AA)
    return strip


def runs(mask: np.ndarray):
    """(start, stop) of each run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0]))
    return list(zip(edges[::2], edges[1::2]))


def render(scene: Scene, path: Path) -> None:
    capture = cv2.VideoCapture(str(scene.video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    panels = strip_panels(scene)
    raw = path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height + STRIP_HEIGHT))
    first, last = np.flatnonzero(scene.frame_samples >= 0)[[0, -1]]
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(first))
    for frame in range(int(first), int(last) + 1):
        ok, image = capture.read()
        if not ok:
            break
        scene.draw(image, frame)
        writer.write(np.vstack((image, draw_strip(scene, panels, width, scene.frame_times[frame]))))
    writer.release()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
         str(path)], check=True,
    )
    raw.unlink()


def stills(scene: Scene, seconds: list[float], directory: Path, name: str, margin: int = 70) -> None:
    """Frames at ``seconds`` of trial time (the predictions' ``time_s``), cropped around the drawn
    forces with the same height for all, and their row."""
    capture = cv2.VideoCapture(str(scene.video_path))
    shots = []
    for second in seconds:
        sample = int(np.argmin(np.abs(scene.predictions["time_s"].to_numpy() - second)))
        frame = int(np.nanargmin(np.abs(scene.frame_times - scene.times[sample])))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
        if not ok:
            continue
        clean = image.copy()
        scene.draw(image, frame)
        changed = np.argwhere(np.abs(image.astype(int) - clean).sum(axis=2) > 30)
        changed = changed[changed[:, 0] > 100]  # the legend stays out of the crop
        if not len(changed):
            # Nothing drawn (both feet airborne): frame the projected feet.
            feet = scene.camera.project(scene.origin[scene.frame_samples[frame]])
            changed = np.round(feet[:, ::-1]).astype(int) if np.isfinite(feet).all() else changed
        if len(changed):
            shots.append((second, image, changed.min(axis=0), changed.max(axis=0)))
    if not shots:
        return
    # One crop size for every still: the union of the drawn regions, centred on each still's own.
    size = np.max([high - low for _, _, low, high in shots], axis=0) + 2 * margin
    crops = []
    for second, image, low, high in shots:
        centre = (low + high) // 2
        top, left = np.clip(centre - size // 2, 0, np.array(image.shape[:2]) - size)
        crop = image[top:top + size[0], left:left + size[1]]
        cv2.imwrite(str(directory / f"{name}_t{second:05.2f}.png"), crop)
        crops.append(crop)
    row = [cv2.copyMakeBorder(crop, 0, 0, 0, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for crop in crops]
    cv2.imwrite(str(directory / f"{name}_stills.png"), np.hstack(row)[:, :-8])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    calibration = commands.add_parser("calibrate", help="calibrate one session camera")
    calibration.add_argument("session", help="acquisition folder, e.g. P02 or P08b")
    calibration.add_argument("--from-camera", choices=("cam1", "cam2"),
                             help="calibrate without clicks from this already calibrated camera of the session")
    calibration.add_argument("--extra-sessions", nargs="*", default=[],
                             help="with --from-camera: sessions of the same day to pool, e.g. P03b")
    rendering = commands.add_parser("render", help="predict and render one trial")
    rendering.add_argument("trial", help="prepared trial name, e.g. P02_squat")
    rendering.add_argument("--run-dir", type=Path, required=True, help="full-CV run dir holding fold_<Participant>/")
    rendering.add_argument("--data-dir", type=Path, help="prepared dataset (default: the run's data_dir)")
    rendering.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    rendering.add_argument("--stills", type=float, nargs="*", default=[], help="trial times (s, as time_s) of the stills")
    rendering.add_argument("--no-video", action="store_true", help="stills only")
    for command in (calibration, rendering):
        command.add_argument("--camera", choices=("cam1", "cam2"), required=True)
        command.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "calibrate":
        if args.from_camera:
            calibrate_second_camera(args.session, args.camera, args.from_camera, args.output_dir, args.extra_sessions)
        else:
            calibrate_session(args.session, args.camera, args.output_dir)
        return 0
    out = args.output_dir / args.trial
    out.mkdir(parents=True, exist_ok=True)
    predictions_path = out / "predictions_all_samples.csv"
    if not predictions_path.is_file():
        predict_trial(args.trial, args.run_dir, args.data_dir, torch.device(args.device)).to_csv(
            predictions_path, index=False
        )
    scene = Scene(args.trial, args.camera, args.output_dir, pd.read_csv(predictions_path))
    print(f"{args.trial}: sole plane {scene.sole_mm:.1f} mm below the cluster origin, video offset {scene.offset:.3f} s")
    if args.stills:
        stills(scene, args.stills, out, f"{args.trial}_{args.camera}")
    if not args.no_video:
        render(scene, out / f"{args.trial}_{args.camera}.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
