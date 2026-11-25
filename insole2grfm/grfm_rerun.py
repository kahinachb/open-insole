#!/usr/bin/env python3
"""Rerun recording of a trial for the paper: pseudo-ground-truth body, insoles and GRFM.

In the QTM lab frame (z up, metres), on the trial time line (``time_s`` of the
predictions):

- the BODY is the pseudo ground truth of sam3d-student-release
  (``datasets_pipeline/semelle/label.py rows``): sam-3d-body run on every 5th
  video frame (8 fps), decoded with the MHR rig and carried to its LOD 3 mesh
  (4 899 vertices) by a fixed barycentric map. It is taken to the lab by the
  camera pose, scaled about the camera centre by the trial's median ratio of
  the mocap feet to its foot keypoints (the teacher's per-trial depth error,
  which a scale about the camera leaves invisible in the image), interpolated
  to every video frame (cubic spline within runs of keyframes at most
  ``MAX_GAP_S`` apart) and shifted per frame onto the mocap feet, the feet in
  contact (insole force, ``label.py``'s threshold) weighing most: the foot
  lock. With two calibrated cameras, the one whose feet sit closest to the
  mocap feet gives the body;
- the force plates, and for each foot the force arrow from its Centre of
  pressure and its Free moment arc: force plates (reference, only where
  Measured or Airborne) in graphite, insoles (MLP) in blue;
- each camera video, with the same arrows and arcs projected on it;
- the two insoles from above (``insole_heatmap.py``): load density, total
  load and Centre of pressure of the gauges;
- the six GRFM components and the Free moment per foot, in a 6 s window
  following the time cursor.

The videos (cameras and insoles) are embedded as H.264, so a recording stays
small. The look (backgrounds, grid, eye, curve ranges) is the recording's
default blueprint.

Example::

    python grfm_rerun.py P02_squat \\
      --output-dir ../docs/analysis/figures/2026-11-01-grfm-rerun [--spawn]
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rerun as rr
import rerun.blueprint as rrb
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, savgol_filter

from grfm_schema import COMPONENTS, FEET, STATE_AIRBORNE, STATE_MEASURED, TARGETS
from insole_heatmap import FONT, HEIGHT as HEATMAP_HEIGHT, TRAIL_S, WIDTH as HEATMAP_WIDTH, InsoleHeatmap
from reproject_grfm_video import (
    ARC_DEGREES_PER_NM, ARC_RADIUS_MM, Scene, predict_trial, subject_session_from_batch, trial_dir,
)
from video_calibration import FootObservations, keypoints_lab, nearest_rows, plate_corners

WORKSPACE = Path(__file__).resolve().parents[1]
STUDENT_ROOT = WORKSPACE.parent / "sam3d-student-release"
ANNOTATIONS = STUDENT_ROOT / "data" / "semelle_work" / "rows"
# fill_gaps.py: the participant tracked through the frames label.py skipped for want of mocap feet.
FILLED = STUDENT_ROOT / "data" / "semelle_work" / "rows_fill"
# label.py's log: the video offset each annotation was made with, also where the calibration has none.
LABEL_LOG = STUDENT_ROOT / "data" / "semelle_work" / "label_rows.log"
LOD3_FBX = STUDENT_ROOT / "models" / "mhr_assets" / "assets" / "lod3.fbx"
CALIBRATION_DIR = WORKSPACE / "docs" / "analysis" / "figures" / "2026-10-30-grfm-video-mlp"
RUN_DIR = WORKSPACE / "results" / "full_cv_v3.7.1" / "mlp_sensor_dropout_w21" / "all"
DATA_DIR = WORKSPACE / "dataset_v3.7.1"
CAMERAS = ("cam1", "cam2")

# MHR70 indexes of the foot keypoints: ankle, big toe, small toe, heel; L then R (label.py's FOOT_KP).
FOOT_JOINT_INDEXES = (13, 15, 16, 17, 14, 18, 19, 20)
SWAP = np.r_[4:8, 0:4]
MAX_GAP_S = 0.30
OUTLIER_MM = 150.0
SWING_WEIGHT = 0.15
LOCK_WINDOW, LOCK_ORDER = 9, 2  # 225 ms at 40 fps
ARROW_3D_FACTOR = 1.2
OPACITY_3D = 166  # arcs in the 3D view, 0.65
# Rerun draws a translucent arrow with its back faces, which darkens it: the 3D arrows are opaque, in the
# colour they would have at OPACITY_3D over the canvas.
ARC_LIFT_M = 0.025  # 3D Free moment arcs above the Sole plane, clear of the plates
NATIVE_HZ = 200.0
FOOT_FRAME_TOLERANCE_S = 0.03
LOADED_N = 50.0
LOW_PASS_HZ, LOW_PASS_ORDER = 15.0, 4  # zero-phase (forward-backward) Butterworth: no lag

# Palette: Apple-like neutrals, one colour per source.
INK = (29, 29, 31)  # force plates
ACCENT = (0, 113, 227)  # insoles
CANVAS = (250, 250, 252)
WHITE = (255, 255, 255)
GRID = (230, 230, 235)
PLATE = (232, 232, 237)
PLATE_EDGE = (174, 174, 178)
BODY = (112, 112, 120, 120)
# (source, column prefix, colour, legend, 3D arrow radius m, 2D arrow radius ui points)
SOURCES = (
    ("ref", "", INK, "Force plates", 0.016, 1.7),
    ("mlp", "pred_", ACCENT, "Insoles (MLP)", 0.010, 1.2),
)
QUANTITIES = {
    "Fz": ("Vertical force Fz", "N"),
    "Fx": ("Anteroposterior force Fx", "N"),
    "Fy": ("Mediolateral force Fy", "N"),
    "Free": ("Free moment", "N m"),
    "Mx": ("Moment Mx about the cluster", "N m"),
    "My": ("Moment My about the cluster", "N m"),
    "Mz": ("Moment Mz about the cluster", "N m"),
}
SIDE_NAMES = {"L": "Left", "R": "Right"}


@functools.cache
def label_offsets() -> dict[str, float]:
    """``<session>_<trial>_<camN>`` -> video offset (s) of the annotations."""
    offsets = {}
    if LABEL_LOG.is_file():
        for match in re.finditer(r"^(\S+): offset (-?[0-9.]+) s", LABEL_LOG.read_text(), re.MULTILINE):
            offsets[match[1]] = float(match[2])
    return offsets


@dataclass
class View:
    """One calibrated camera of the trial: its scene (calibration, video offset, GRFM) and pose."""

    name: str
    scene: Scene
    rotation: np.ndarray
    translation: np.ndarray

    @classmethod
    def load(cls, trial: str, camera: str, calibration_dir: Path, predictions: pd.DataFrame) -> View | None:
        if not (calibration_dir / "calibration" / f"{subject_session_from_batch(trial)}_{camera}.json").is_file():
            return None
        try:
            recording = trial_dir(trial)
            scene = Scene(trial, camera, calibration_dir, predictions,
                          label_offsets().get(f"{recording.parent.name}_{recording.name}_{camera}"))
        except SystemExit:  # no video offset for this trial on this camera
            return None
        rotation, _ = cv2.Rodrigues(np.asarray(scene.camera.rvec, dtype=np.float64))
        return cls(camera, scene, rotation, np.asarray(scene.camera.tvec))

    def to_lab(self, points_cam_mm: np.ndarray) -> np.ndarray:
        return (points_cam_mm - self.translation) @ self.rotation

    def to_camera(self, points_lab_mm: np.ndarray) -> np.ndarray:
        return points_lab_mm @ self.rotation.T + self.translation

    @property
    def frames(self) -> np.ndarray:
        """Video frames inside the predictions."""
        shown = np.flatnonzero(self.scene.frame_samples >= 0)
        return np.arange(shown[0], shown[-1] + 1)

    def trial_times(self, frames: np.ndarray) -> np.ndarray:
        scene = self.scene
        return np.interp(scene.frame_times[frames], scene.times, scene.predictions["time_s"].to_numpy())

    def mocap_feet(self, frames: np.ndarray, local: np.ndarray) -> np.ndarray:
        """(n, 8, 3) lab mm of the mocap foot keypoints at ``frames`` (NaN without mocap)."""
        samples = self.scene.frame_samples[np.minimum(frames, len(self.scene.frame_samples) - 1)]
        rotation = np.where(samples[:, None, None, None] >= 0, self.scene.rotation[samples], np.nan)
        origin = np.where(samples[:, None, None] >= 0, self.scene.origin[samples], np.nan)
        return keypoints_lab(FootObservations(rotation, origin, None, {}), local)


class Lod3:
    """Fixed barycentric map from the MHR rig mesh (LOD 1, 18 439 vertices) to its LOD 3 mesh."""

    def __init__(self, mhr):
        import pymomentum.geometry
        import trimesh

        character = pymomentum.geometry.Character.load_fbx(str(LOD3_FBX))
        low = np.asarray(character.mesh.vertices) / 100.0 * np.array([1.0, -1.0, -1.0])  # cm, y up -> m, vision
        rest, _ = mhr.forward_full(np.zeros(204, np.float32), np.zeros(45, np.float32))
        closest, distance, triangle = trimesh.proximity.closest_point(trimesh.Trimesh(rest, mhr.faces, process=False), low)
        if np.median(distance) > 1e-3:
            raise SystemExit(f"{LOD3_FBX} does not lie on the rig mesh (median {np.median(distance) * 1e3:.1f} mm)")
        self.corners = mhr.faces[triangle]
        self.weights = trimesh.triangles.points_to_barycentric(rest[self.corners], closest).astype(np.float32)
        self.faces = np.asarray(character.mesh.faces, dtype=np.int32)

    def __call__(self, vertices: np.ndarray) -> np.ndarray:
        return np.einsum("...vkc,vk->...vc", vertices[..., self.corners, :], self.weights)


@dataclass
class Body:
    """One camera's pseudo-ground-truth keyframes in the lab (m), after the depth scale."""

    view: View
    times: np.ndarray  # trial time of each keyframe
    vertices: np.ndarray  # (n, V, 3)
    feet: np.ndarray  # (n, 8, 3), mocap side order
    scale: float
    residual_mm: float  # median foot keypoint distance to the mocap feet

    @classmethod
    def load(cls, view: View, mhr, lod: Lod3, local: np.ndarray) -> Body | None:
        trial_path = view.scene.trial_path
        name = f"{trial_path.parent.name}_{trial_path.name}_{view.name}.parquet"
        if not (ANNOTATIONS / name).is_file():
            return None
        rows = pd.concat([pd.read_parquet(folder / name, columns=["image", "model_params", "shape_params"])
                          for folder in (ANNOTATIONS, FILLED) if (folder / name).is_file()])
        frames = rows["image"].str.extract(r"_(\d{6})\.jpg$")[0].astype(int).to_numpy()
        rows, frames = rows.iloc[np.argsort(frames, kind="stable")], np.sort(frames, kind="stable")
        keep = (frames < len(view.scene.frame_samples)) & (view.scene.frame_samples[np.minimum(
            frames, len(view.scene.frame_samples) - 1)] >= 0)
        if keep.sum() < 4:
            return None
        rows, frames = rows[keep], frames[keep]
        params = np.stack(rows["model_params"].map(np.asarray)).astype(np.float32)
        shapes = np.stack(rows["shape_params"].map(np.asarray)).astype(np.float32)
        vertices, feet = [], []
        for start in range(0, len(params), 128):
            mesh, keypoints = mhr.forward_full(params[start:start + 128], shapes[start:start + 128])
            vertices.append(lod(mesh))
            feet.append(keypoints[:, list(FOOT_JOINT_INDEXES)])
        vertices_cam, feet_cam = np.concatenate(vertices) * 1000.0, np.concatenate(feet) * 1000.0
        mocap_cam = view.to_camera(view.mocap_feet(frames, local))

        # The keypoint side order of each keyframe, then one depth scale about the camera centre.
        swapped = np.zeros(len(frames), dtype=bool)
        ratios = np.full(len(frames), np.nan)
        for index in np.flatnonzero(np.isfinite(mocap_cam).all(axis=(1, 2))):
            best = np.inf
            for swap, order in ((False, np.arange(8)), (True, SWAP)):
                source = feet_cam[index][order]
                ratio = np.sum(source * mocap_cam[index]) / np.sum(source * source)
                error = np.median(np.linalg.norm(ratio * source - mocap_cam[index], axis=1))
                if error < best:
                    best, swapped[index], ratios[index] = error, swap, ratio
        if not np.isfinite(ratios).any():
            return None
        scale = float(np.nanmedian(ratios))
        feet_cam[swapped] = feet_cam[swapped][:, SWAP]
        error = np.linalg.norm(scale * feet_cam - mocap_cam, axis=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # keyframes without mocap
            per_frame = np.nanmedian(error, axis=1)
        good = ~(per_frame > OUTLIER_MM)  # keyframes without mocap are kept
        return cls(
            view=view,
            times=view.trial_times(frames[good]),
            vertices=(view.to_lab(scale * vertices_cam[good]) / 1000.0).astype(np.float32),
            feet=view.to_lab(scale * feet_cam[good]) / 1000.0,
            scale=scale,
            residual_mm=float(np.nanmedian(per_frame[good])),
        )


def foot_contact(trial: str, trial_path: Path, times_abs: np.ndarray) -> np.ndarray:
    """(n, 2) weight of each foot in the foot lock: 1 in contact, ``SWING_WEIGHT`` in the air, 0.5 unsure."""
    sys.path.insert(0, str(STUDENT_ROOT / "datasets_pipeline" / "semelle"))
    from label import trial_contact

    weights = np.full((len(times_abs), 2), 0.5)
    # The prepared trial (a segment of the recording for the split lateral tasks) and its recording.
    contact = trial_contact(trial_path.parent.name, trial[len(trial_path.parent.name) + 1:], trial_path)
    if contact is None:
        return weights
    rows = nearest_rows(contact[0], times_abs, 0.010)
    found = rows >= 0
    on, valid = contact[1][rows[found]] > 0, contact[2][rows[found]] > 0
    weights[found] = np.where(valid, np.where(on, 1.0, SWING_WEIGHT), 0.5)
    return weights


def yaw(rotation: np.ndarray) -> np.ndarray:
    """Heading (rad) of the foot frame's x axis in the horizontal plane."""
    return np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])


def about_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def fill_foot_frames(scene: Scene, times: np.ndarray, feet_mm: np.ndarray, local: np.ndarray) -> int:
    """Foot frames of the samples the mocap lacks, then the GRFM placed again; returns how many.

    The body's foot keypoints (``feet_mm`` at ``times``, the nearest frame within
    ``FOOT_FRAME_TOLERANCE_S``) give the heading and horizontal position of the foot; its tilt and,
    under load, its height are the foot's median ones over the loaded samples with mocap (the body's
    keypoints do not match the mocap ones well enough to tilt the foot, and a tilt moves the Centre of
    pressure by its lever arm). Returns the number of foot frames filled."""
    rows = nearest_rows(times, scene.predictions["time_s"].to_numpy(), FOOT_FRAME_TOLERANCE_S)
    count = 0
    for index, side in enumerate(FEET):
        loaded = scene.predictions[f"pred_Fz_{side}"].to_numpy() > LOADED_N
        known = np.isfinite(scene.rotation[:, index]).all(axis=(1, 2))
        stance = known & loaded
        if not stance.any():
            continue
        level = np.einsum("nji,njk->nik", np.stack([about_z(a) for a in yaw(scene.rotation[stance, index])]),
                          scene.rotation[stance, index]).mean(axis=0)
        u, _, vt = np.linalg.svd(level)
        tilt = u @ np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))]) @ vt
        stance_height = np.median(scene.origin[stance, index, 2])
        source = local[index] @ tilt.T  # keypoints in a frame with the foot's tilt and no heading
        # The body's heading and horizontal position of the foot at every sample it covers.
        heading = np.full(len(rows), np.nan)
        position = np.full((len(rows), 3), np.nan)
        for sample in np.flatnonzero(rows >= 0):
            target = feet_mm[rows[sample], 4 * index: 4 * index + 4]
            if np.isfinite(target).all():
                a, b = source[:, :2] - source[:, :2].mean(axis=0), target[:, :2] - target[:, :2].mean(axis=0)
                heading[sample] = np.arctan2(np.sum(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), np.sum(a * b))
                position[sample] = target.mean(axis=0) - about_z(heading[sample]) @ source.mean(axis=0)
        # Where both exist, the mocap's offset from the body; across a gap, blended linearly between the
        # offsets on its two sides, so the foot does not jump when the mocap drops out or comes back.
        both = known & np.isfinite(heading)
        if not both.any():
            continue
        turn_offset = np.angle(np.exp(1j * (yaw(scene.rotation[:, index]) - heading)))
        shift_offset = scene.origin[:, index] - position
        samples = np.arange(len(rows))
        turn_offset = np.interp(samples, samples[both], np.unwrap(turn_offset[both]))
        shift_offset = np.stack([np.interp(samples, samples[both], shift_offset[both, k]) for k in range(3)], axis=1)
        for sample in np.flatnonzero(~known & np.isfinite(heading)):
            origin = position[sample] + shift_offset[sample]
            if loaded[sample]:
                origin[2] = stance_height
            scene.rotation[sample, index] = about_z(heading[sample] + turn_offset[sample]) @ tilt
            scene.origin[sample, index] = origin
            count += 1
    scene.place_grfm()
    return count


def runs(times: np.ndarray, gap: float) -> list[tuple[int, int]]:
    """[start, stop) index ranges of ``times`` without a step above ``gap``."""
    cuts = np.flatnonzero(np.diff(times) > gap) + 1
    edges = np.r_[0, cuts, len(times)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a >= 2]


def body_track(body: Body, times: np.ndarray, mocap_feet_m: np.ndarray, weights: np.ndarray):
    """(n, V, 3) body vertices and (n, 8, 3) foot keypoints at ``times`` (NaN outside the keyframe
    runs), foot-locked, and the median foot keypoint distance to the mocap feet (mm) before and after
    the lock."""
    vertices = np.full((len(times), body.vertices.shape[1], 3), np.nan, np.float32)
    feet = np.full((len(times), 8, 3), np.nan)
    for start, stop in runs(body.times, MAX_GAP_S):
        inside = (times >= body.times[start]) & (times <= body.times[stop - 1])
        if stop - start >= 4:
            vertices[inside] = CubicSpline(body.times[start:stop], body.vertices[start:stop], axis=0)(times[inside])
            feet[inside] = CubicSpline(body.times[start:stop], body.feet[start:stop], axis=0)(times[inside])
        else:
            for target, source in ((vertices, body.vertices), (feet, body.feet)):
                flat = source[start:stop].reshape(stop - start, -1)
                target[inside] = np.stack([np.interp(times[inside], body.times[start:stop], column) for column in flat.T],
                                          axis=1).reshape(-1, *source.shape[1:])
    present = np.isfinite(vertices[:, 0, 0])
    # Foot lock: per frame and axis, the weighted mean step from the body's foot keypoints to their
    # targets: the mocap ones; without them, for a foot in contact (insole force), the height its
    # keypoints have over the frames where mocap shows that foot in contact.
    step = mocap_feet_m - feet
    before = np.nanmedian(np.linalg.norm(step, axis=2)) * 1000.0 if np.isfinite(step).any() else np.nan
    contact = np.repeat(weights >= 1.0, 4, axis=1)
    mocap = np.isfinite(mocap_feet_m).all(axis=2)
    height = np.array([np.median(mocap_feet_m[mocap[:, k] & contact[:, k], k, 2])
                       if (mocap[:, k] & contact[:, k]).any() else np.nan for k in range(8)])
    floor = ~mocap & contact & np.isfinite(height)
    step[floor, 2] = height[np.nonzero(floor)[1]] - feet[floor, 2]
    per_keypoint = np.repeat(weights, 4, axis=1)[..., None] * np.isfinite(step)
    total = per_keypoint.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.where(total > 0, np.einsum("nkc,nkc->nc", per_keypoint, np.nan_to_num(step)) / total, np.nan)
    for start, stop in runs(np.flatnonzero(present).astype(float), 1.5):
        frames = np.flatnonzero(present)[start:stop]
        run = shift[frames]
        for axis in range(3):
            known = np.isfinite(run[:, axis])
            if known.any():
                run[:, axis] = np.interp(np.arange(len(run)), np.flatnonzero(known), run[known, axis])
        if not np.isfinite(run).all():
            continue
        if len(run) >= LOCK_WINDOW:
            run = savgol_filter(run, LOCK_WINDOW, LOCK_ORDER, axis=0)
        vertices[frames] += run[:, None, :].astype(np.float32)
        feet[frames] += run[:, None, :]
    after = np.nanmedian(np.linalg.norm(mocap_feet_m - feet, axis=2)) * 1000.0 if np.isfinite(step).any() else np.nan
    return vertices, feet, float(before), float(after)


def encode_video(frames, size: tuple[int, int], fps: float, path: Path, pixel_format: str = "bgr24") -> None:
    """H.264 without B-frames (the viewer scrubs it frame by frame) from an iterator of images."""
    command = [
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", pixel_format, "-s", f"{size[0]}x{size[1]}",
        "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "21", "-bf", "0", "-g",
        f"{int(fps)}", "-pix_fmt", "yuv420p", str(path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    for image in frames:
        process.stdin.write(np.ascontiguousarray(image).tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed on {path}")


def video_frames(path: Path, first: int, last: int):
    """Frames ``first..last`` read in order (seeking mpeg4 is slow and inexact); the last one repeats
    past the end of the file."""
    capture = cv2.VideoCapture(str(path))
    image = None
    for frame in range(last + 1):
        if frame < first:
            capture.grab()
            continue
        ok, read = capture.read()
        image = read if ok else image
        yield image


def legend(size: tuple[int, int], font_px: int, background: tuple | None) -> np.ndarray:
    """RGBA legend of the two sources, a line swatch and a label each: on ``background``, one
    left-aligned row of ``size``; with ``background`` None, a translucent white card sized to its content."""
    font = ImageFont.truetype(FONT, font_px)
    pad, swatch, gap = font_px * 0.7, font_px * 2.0, font_px * 0.6
    widths = [font.getlength(label) for _, _, _, label, _, _ in SOURCES]
    card = (int(2 * pad + swatch + gap + max(widths)), int(2 * pad + len(SOURCES) * font_px * 1.5 - font_px * 0.5))
    image = Image.new("RGBA", size if background else card, (*background, 255) if background else (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    if background is None:
        draw.rounded_rectangle((0, 0, card[0] - 1, card[1] - 1), radius=font_px * 0.5, fill=(255, 255, 255, 215))
    x = pad
    for row, (_, _, colour, label, _, points) in enumerate(SOURCES):
        # A card stacks the sources; a strip (``background`` given) lines them up.
        y = size[1] / 2 if background else pad + row * font_px * 1.5 + font_px * 0.5
        left = x if background else pad
        draw.line((left, y, left + swatch, y), fill=colour, width=max(int(round(points * font_px / 9)), 2))
        draw.text((left + swatch + gap, y), label, fill=INK, font=font, anchor="lm")
        x = left + swatch + gap + widths[row] + 2.5 * font_px
    return np.asarray(image)


def with_legend(frames, legend_rgba: np.ndarray, margin: int):
    """BGR frames with ``legend_rgba`` pasted in their top-right corner."""
    height, width = legend_rgba.shape[:2]
    alpha = legend_rgba[..., 3:4].astype(np.float32) / 255.0
    colour = legend_rgba[..., 2::-1].astype(np.float32)
    for image in frames:
        if image is not None:
            image = image.copy()
            region = image[margin:margin + height, -margin - width:-margin]
            region[:] = (region * (1 - alpha) + colour * alpha).astype(np.uint8)
        yield image


def low_pass(predictions: pd.DataFrame) -> pd.DataFrame:
    """The reference and predicted GRFM, low-passed forward and backward (no lag) over each run of
    finite samples; runs too short for the filter stay as they are."""
    filtered = predictions.copy()
    rate = 1.0 / np.median(np.diff(predictions["time_s"].to_numpy()))
    b, a = butter(LOW_PASS_ORDER, LOW_PASS_HZ, fs=rate)
    for column in [f"{prefix}{target}" for prefix in ("", "pred_") for target in TARGETS]:
        values = filtered[column].to_numpy(dtype=np.float64).copy()
        finite = np.isfinite(values)
        edges = np.flatnonzero(np.diff(np.r_[0, finite.astype(np.int8), 0]))
        for start, stop in zip(edges[::2], edges[1::2]):
            if stop - start > 3 * max(len(a), len(b)):
                values[start:stop] = filtfilt(b, a, values[start:stop])
        filtered[column] = values
    return filtered


def log_video(entity: str, path: Path, times: np.ndarray) -> None:
    video = rr.AssetVideo(path=path)
    rr.log(entity, video, static=True)
    stamps = video.read_frame_timestamps_nanos()
    count = min(len(stamps), len(times))
    rr.send_columns(entity, indexes=[rr.TimeColumn("trial_time", duration=times[:count])],
                    columns=rr.VideoFrameReference.columns_nanos(stamps[:count]))


def send_instances(entity: str, archetype, times: np.ndarray, counts, **fields) -> None:
    """``archetype``'s fields for ``times``, ``counts[i]`` instances at time i (0 hides it)."""
    if not np.any(counts):
        return
    rr.send_columns(entity, indexes=[rr.TimeColumn("trial_time", duration=times)],
                    columns=archetype.columns(**fields).partition(counts))


def over_canvas(colour) -> tuple[int, int, int]:
    """``colour`` at ``OPACITY_3D`` over ``CANVAS``."""
    alpha = OPACITY_3D / 255.0
    return tuple(int(round(alpha * c + (1 - alpha) * b)) for c, b in zip(colour, CANVAS))


def free_moment_arc(centre, rotation, free_nm) -> np.ndarray | None:
    """Lab mm points of the Free moment arc around the foot vertical (``reproject_grfm_video``'s)."""
    span = np.clip(free_nm * ARC_DEGREES_PER_NM, -330.0, 330.0)
    if not np.isfinite(span) or abs(span) < 30.0:
        return None
    angles = np.radians(np.linspace(0.0, span, max(int(abs(span) / 6), 4)))
    local = np.column_stack((ARC_RADIUS_MM * np.cos(angles), ARC_RADIUS_MM * np.sin(angles), np.zeros_like(angles)))
    return local @ rotation.T + centre


def grfm_shapes(scene: Scene, samples: np.ndarray, source: str, side: str):
    """Per sample: force arrow (origin, tip) and Free moment arc in lab mm, None where not drawn."""
    index = FEET.index(side)
    cop, force, free = scene.grfm[source, side]
    shapes = []
    for sample in samples:
        rotation = scene.rotation[sample, index]
        if sample < 0 or not (np.isfinite(cop[sample]).all() and np.isfinite(rotation).all()):
            shapes.append((None, None))
            continue
        tip = cop[sample] + ARROW_3D_FACTOR * scene.scale * force[sample]
        shapes.append(((cop[sample], tip), free_moment_arc(cop[sample], rotation, free[sample])))
    return shapes


def log_world(views: list[View], times: np.ndarray, samples: np.ndarray, vertices: np.ndarray | None,
              faces: np.ndarray) -> None:
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    corners = plate_corners(views[0].scene.trial_path) / 1000.0
    for plate in range(len(corners) // 4):
        quad = corners[4 * plate: 4 * plate + 4]
        rr.log(f"world/plates/plate_{plate + 1}", rr.Mesh3D(
            vertex_positions=quad, triangle_indices=[[0, 1, 2], [0, 2, 3]], albedo_factor=PLATE,
        ), static=True)
        rr.log(f"world/plates/plate_{plate + 1}/edge", rr.LineStrips3D(
            [np.vstack((quad, quad[:1]))], radii=0.0012, colors=[PLATE_EDGE],
        ), static=True)
    if vertices is not None:
        present = np.isfinite(vertices[:, 0, 0])
        rr.log("world/body", rr.Mesh3D.from_fields(albedo_factor=BODY), static=True)
        send_instances("world/body", rr.Mesh3D, times[present], np.full(int(present.sum()), vertices.shape[1]),
                       vertex_positions=vertices[present].reshape(-1, 3))
        # The faces follow the presence of the body: without them a stale mesh is not drawn.
        toggles = np.flatnonzero(np.diff(np.r_[0, present.astype(np.int8)]) != 0)
        if len(toggles):
            counts = np.where(present[toggles], len(faces), 0)
            send_instances("world/body", rr.Mesh3D, times[toggles], counts,
                           triangle_indices=np.concatenate([faces] * int(present[toggles].sum()) or [np.zeros((0, 3))]))
    scene = views[0].scene
    for side in FEET:
        for source, _, colour, _, radius, _ in SOURCES:
            path = f"world/grfm/{side}/{source}"
            shapes = grfm_shapes(scene, samples, source, side)
            arrows = [arrow for arrow, _ in shapes if arrow is not None]
            rr.log(f"{path}/force", rr.Arrows3D.from_fields(radii=radius, colors=[over_canvas(colour)]), static=True)
            send_instances(f"{path}/force", rr.Arrows3D, times, [int(arrow is not None) for arrow, _ in shapes],
                           origins=np.array([a for a, _ in arrows]).reshape(-1, 3) / 1000.0,
                           vectors=np.array([b - a for a, b in arrows]).reshape(-1, 3) / 1000.0)
            arcs = [arc for _, arc in shapes if arc is not None]
            rr.log(f"{path}/free_moment", rr.LineStrips3D.from_fields(radii=radius * 0.4, colors=[(*colour, OPACITY_3D)]), static=True)
            send_instances(f"{path}/free_moment", rr.LineStrips3D, times, [int(arc is not None) for _, arc in shapes],
                           strips=[arc / 1000.0 + [0.0, 0.0, ARC_LIFT_M] for arc in arcs])


def log_camera(view: View, size: tuple[int, int], times: np.ndarray, frames: np.ndarray) -> None:
    """Pinhole, and the force arrows and Free moment arcs projected on the video."""
    camera, entity = view.scene.camera, f"world/{view.name}"
    rr.log(entity, rr.Transform3D(translation=camera.centre() / 1000.0, mat3x3=view.rotation.T), static=True)
    rr.log(entity, rr.Pinhole(
        focal_length=camera.focal_px, principal_point=[camera.cx, camera.cy], resolution=list(size),
        camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.4,
    ), static=True)
    samples = view.scene.frame_samples[frames]
    for side in FEET:
        for source, _, colour, _, _, points in SOURCES:
            shapes = grfm_shapes(view.scene, samples, source, side)
            arrows, arcs = [], []
            for arrow, arc in shapes:
                ends = camera.project(np.stack(arrow)) if arrow is not None else None
                arrows.append(ends if ends is not None and np.isfinite(ends).all() else None)
                pixels = camera.project(arc) if arc is not None else None
                arcs.append(pixels[np.isfinite(pixels).all(axis=1)] if pixels is not None else None)
            arc_counts = [int(arc is not None and len(arc) > 1) for arc in arcs]
            for layer_colour, width, order in ((colour, points, 21.0),):
                path = f"{entity}/overlay/{side}/{source}"
                rr.log(f"{path}/force", rr.Arrows2D.from_fields(
                    radii=rr.Radius.ui_points(width), colors=[layer_colour], draw_order=order + (source == "mlp"),
                ), static=True)
                kept = [arrow for arrow in arrows if arrow is not None]
                send_instances(f"{path}/force", rr.Arrows2D, times, [int(arrow is not None) for arrow in arrows],
                               origins=np.array([a[0] for a in kept]).reshape(-1, 2),
                               vectors=np.array([a[1] - a[0] for a in kept]).reshape(-1, 2))
                rr.log(f"{path}/free_moment", rr.LineStrips2D.from_fields(
                    radii=rr.Radius.ui_points(width * 0.7), colors=[layer_colour], draw_order=order + (source == "mlp"),
                ), static=True)
                send_instances(f"{path}/free_moment", rr.LineStrips2D, times, arc_counts,
                               strips=[arc for arc, count in zip(arcs, arc_counts) if count])


def series_values(scene: Scene, source: str, side: str, quantity: str) -> np.ndarray:
    """A curve of the trial, the reference only where Measured or Airborne."""
    prefix = "" if source == "ref" else "pred_"
    if quantity == "Free":
        values = scene.grfm[source, side][2].copy()
    else:
        values = scene.predictions[f"{prefix}{quantity}_{side}"].to_numpy(dtype=np.float64).copy()
    if source == "ref":
        values[~np.isin(scene.predictions[f"state_{side}"], (STATE_MEASURED, STATE_AIRBORNE))] = np.nan
    return values


def curve_ranges(scene: Scene) -> dict[str, tuple[float, float]]:
    """Vertical range of each quantity, shared by both feet and both sources."""
    ranges = {}
    for quantity in QUANTITIES:
        values = np.concatenate([series_values(scene, source, side, quantity) for source, *_ in SOURCES for side in FEET])
        low, high = np.nanpercentile(values, [0.2, 99.8])
        low, high = min(low, 0.0), max(high, 0.0)
        pad = 0.08 * max(high - low, 1.0)
        ranges[quantity] = (float(low - pad), float(high + pad))
    return ranges


def log_series(scene: Scene) -> None:
    times = rr.TimeColumn("trial_time", duration=scene.predictions["time_s"].to_numpy())
    for quantity in QUANTITIES:
        for source, _, colour, label, _, _ in SOURCES:
            for side in FEET:
                path = f"plots/{quantity}/{side}/{source}"
                rr.log(path, rr.SeriesLines(colors=[colour], widths=[1.6 if source == "ref" else 1.4], names=[label]),
                       static=True)
                rr.send_columns(path, indexes=[times],
                                columns=rr.Scalars.columns(scalars=series_values(scene, source, side, quantity)))


def log_insoles(trial: str, view: View, frames: np.ndarray, times: np.ndarray, data_dir: Path, workdir: Path,
                fps: float) -> None:
    """The insole video, one frame per video frame of ``view``."""
    data = np.load(data_dir / f"{trial}.npz")
    rows_time = (data["target_time_qtm_s_L"] + data["target_time_qtm_s_R"]) / 2.0
    sample_time = view.scene.predictions["time_qtm_s"].to_numpy()[view.scene.frame_samples[frames]]
    rows = np.clip(np.searchsorted(rows_time, sample_time), 0, len(rows_time) - 1)
    heatmap = InsoleHeatmap()
    loads = {side: heatmap.loads(data["X"], side) for side in FEET}
    cop = {side: heatmap.cop_mm(loads[side], side) for side in FEET}
    trail = int(TRAIL_S * NATIVE_HZ)

    def images():
        for row in rows:
            yield heatmap.render({side: loads[side][row] for side in FEET},
                                 {side: cop[side][max(row - trail, 0):row + 1:2] for side in FEET})

    path = workdir / "insoles.mp4"
    encode_video(images(), (HEATMAP_WIDTH, HEATMAP_HEIGHT), fps, path, "rgb24")
    log_video("insoles/video", path, times)


def eye(views: list[View]) -> tuple[np.ndarray, np.ndarray]:
    """A three-quarter view of the plates from the side of the first camera, raised."""
    corners = plate_corners(views[0].scene.trial_path) / 1000.0
    target = np.r_[corners[:, :2].mean(axis=0), 0.9]
    direction = views[0].scene.camera.centre()[:2] / 1000.0 - target[:2]
    angle = np.radians(32.0)
    direction = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) @ direction
    direction /= np.linalg.norm(direction)
    return np.r_[target[:2] + 2.7 * direction, 1.6], target


def blueprint(trial: str, views: list[View], ranges: dict, eye_position, eye_target) -> rrb.Blueprint:
    window = rrb.VisibleTimeRange(
        "trial_time",
        start=rrb.TimeRangeBoundary.cursor_relative(seconds=-3.0),
        end=rrb.TimeRangeBoundary.cursor_relative(seconds=3.0),
    )

    def curve(quantity: str, side: str, legend: bool = False) -> rrb.TimeSeriesView:
        name, unit = QUANTITIES[quantity]
        return rrb.TimeSeriesView(
            name=f"{SIDE_NAMES[side]} · {name} ({unit})",
            origin=f"plots/{quantity}/{side}",
            axis_y=rrb.ScalarAxis(range=ranges[quantity], zoom_lock=True),
            plot_legend=rrb.PlotLegend(corner="LeftTop", visible=legend),
            background=rrb.archetypes.PlotBackground(color=WHITE, show_grid=True),
            time_ranges=window,
        )

    def stack(name: str, *quantities: str) -> rrb.Vertical:
        return rrb.Vertical(*[curve(q, side, legend=(q, side) == (quantities[0], "L"))
                              for q in quantities for side in FEET], name=name)

    cameras = [
        rrb.Spatial2DView(name=f"Camera {view.name[-1]}", origin=f"world/{view.name}",
                          contents=[f"world/{view.name}/**"],
                          background=rrb.Background(kind="SolidColor", color=CANVAS))
        for view in views
    ]
    insoles = rrb.Spatial2DView(name="Insoles · load density (kPa)", origin="insoles",
                                 background=rrb.Background(kind="SolidColor", color=WHITE))
    key = rrb.Spatial2DView(name=trial.replace("_", " "), origin="legend",
                            background=rrb.Background(kind="SolidColor", color=CANVAS))
    lab = rrb.Spatial3DView(
        name="Lab \u00b7 3D",
        origin="world",
        contents=["+ world/**", *[f"- world/{view.name}/**" for view in views]],
        background=rrb.Background(kind="SolidColor", color=CANVAS),
        line_grid=rrb.LineGrid3D(visible=True, spacing=0.5, plane=rr.components.Plane3D.XY, stroke_width=1.0,
                                 color=GRID),
        eye_controls=rrb.EyeControls3D(kind=rrb.Eye3DKind.Orbital, position=eye_position, look_target=eye_target,
                                       eye_up=[0.0, 0.0, 1.0]),
    )
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(*cameras, insoles, row_shares=[1.0] * len(cameras) + [2.0]),
            rrb.Vertical(key, lab, row_shares=[0.1, 1.0]),
            rrb.Tabs(
                stack("Vertical force \u00b7 Free moment", "Fz", "Free"),
                stack("Shear forces", "Fx", "Fy"),
                stack("Moments about the cluster", "Mx", "My", "Mz"),
                active_tab=0,
            ),
            column_shares=[1.0, 2.1, 1.1],
        ),
        rrb.BlueprintPanel(state="collapsed"),
        rrb.SelectionPanel(state="collapsed"),
        rrb.TimePanel(state="collapsed", timeline="trial_time"),
    )


def load_predictions(trial: str, calibration_dir: Path, output_dir: Path, data_dir: Path) -> pd.DataFrame:
    """The trial's predictions: the video render's, else this run's cache, else computed (CPU)."""
    for path in (calibration_dir / trial / "predictions_all_samples.csv", output_dir / "predictions" / f"{trial}.csv"):
        if path.is_file():
            return pd.read_csv(path)
    path = output_dir / "predictions" / f"{trial}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = predict_trial(trial, RUN_DIR, data_dir, torch.device("cpu"))
    predictions.to_csv(path, index=False)
    return predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trial", help="prepared trial name, e.g. P02_squat")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, default=CALIBRATION_DIR,
                        help="holds calibration/ (and the video renders' predictions_all_samples.csv)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spawn", action="store_true", help="open the .rrd in the Rerun viewer when written")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = low_pass(load_predictions(args.trial, args.calibration_dir, args.output_dir, args.data_dir))
    views = [view for view in (View.load(args.trial, name, args.calibration_dir, predictions) for name in CAMERAS) if view]
    if not views:
        raise SystemExit(f"{args.trial}: no calibrated camera with a video offset")
    session = subject_session_from_batch(args.trial)
    local = {view.name: np.asarray(json.loads(
        (args.calibration_dir / "calibration" / f"{session}_{view.name}.json").read_text())["keypoint_local_mm"])
        for view in views}

    sys.path.insert(0, str(STUDENT_ROOT))
    from instanthmr.mhr_renderer import DEFAULT_MHR_SCRIPT, MHRScriptRenderer

    mhr = MHRScriptRenderer(DEFAULT_MHR_SCRIPT, device=args.device)
    lod = Lod3(mhr)
    bodies = [body for body in (Body.load(view, mhr, lod, local[view.name]) for view in views) if body]
    if not bodies:
        raise SystemExit(f"{args.trial}: no pseudo-ground-truth annotation in {ANNOTATIONS}")
    # The body comes from the camera whose feet fit the mocap best, among those that see most of the trial.
    most = max(len(body.times) for body in bodies)
    body = min((body for body in bodies if len(body.times) >= 0.7 * most), key=lambda body: body.residual_mm)
    views = [body.view] + [view for view in views if view is not body.view]
    primary = body.view

    frames = primary.frames
    times = primary.trial_times(frames)
    samples = primary.scene.frame_samples[frames]
    weights = foot_contact(args.trial, primary.scene.trial_path, primary.scene.frame_times[frames])
    vertices, feet, before, after = body_track(body, times, primary.mocap_feet(frames, local[primary.name]) / 1000.0,
                                               weights)
    filled = [fill_foot_frames(view.scene, times, feet * 1000.0, local[primary.name]) for view in views]
    stats = {
        "trial": args.trial, "body_camera": primary.name, "cameras": [view.name for view in views],
        "keyframes": len(body.times), "frames": len(frames),
        "frames_with_body": int(np.isfinite(vertices[:, 0, 0]).sum()),
        "depth_scale": round(body.scale, 4), "foot_keypoint_to_mocap_mm_before_lock": round(before, 1),
        "foot_keypoint_to_mocap_mm_after_lock": round(after, 1),
        "foot_frames_from_body": filled[0],
    }
    print(json.dumps(stats))

    rrd = args.output_dir / f"{args.trial}.rrd"
    rr.init(f"grfm_gt_{args.trial}")
    fps = cv2.VideoCapture(str(primary.scene.video_path)).get(cv2.CAP_PROP_FPS)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as workdir:
        sizes = {}
        for view in views:
            capture = cv2.VideoCapture(str(view.scene.video_path))
            sizes[view.name] = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        rr.save(rrd, default_blueprint=blueprint(args.trial, views, curve_ranges(primary.scene), *eye(views)))
        log_world(views, times, samples, vertices, lod.faces)
        # Wider than its strip, the legend fills the strip's width, so it starts at its left edge.
        rr.log("legend", rr.Image(legend((1200, 40), 22, CANVAS)[..., :3]), static=True)
        for view in views:
            view_frames = view.frames
            view_times = view.trial_times(view_frames)
            path = Path(workdir) / f"{view.name}.mp4"
            card = legend(sizes[view.name], max(sizes[view.name][1] // 26, 14), None)
            encode_video(with_legend(video_frames(view.scene.video_path, int(view_frames[0]), int(view_frames[-1])),
                                     card, sizes[view.name][1] // 30), sizes[view.name], fps, path)
            log_video(f"world/{view.name}/video", path, view_times)
            log_camera(view, sizes[view.name], view_times, view_frames)
        log_insoles(args.trial, primary, frames, times, args.data_dir, Path(workdir), fps)
        log_series(primary.scene)
        rr.disconnect()
    (args.output_dir / f"{args.trial}.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(f"{rrd}: {rrd.stat().st_size / 1e6:.0f} MB")
    if args.spawn:
        subprocess.Popen(["rerun", str(rrd)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
