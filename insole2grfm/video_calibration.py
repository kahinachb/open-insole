"""Calibrate a session camera in the QTM lab frame from plate corners and feet.

One camera stays still through a session, so it is calibrated once per
session from all its trials that have InstantHMR foot keypoints
(``tools/instanthmr_feet.py``):

- the clicked force-plate corners (``tools/click_plate_corners.py``) are
  matched to the eight ``platform_geometry`` corners (the click order is free)
  and give a first pose by planar PnP;
- each InstantHMR foot keypoint (ankle, big toe tip, small toe tip, heel) is a
  point fixed in its foot frame (``rigid_foot_frames`` of the mocap), at an
  unknown offset from the markers; those 8 offsets are estimated with the
  camera (focal, principal point, two radial distortion terms, pose) by robust
  least squares over all frames, the corners keeping the ground in place.

In each frame the person and the left / right assignment that best fit the
current camera are chosen, so bystanders and InstantHMR side swaps drop out.
Each trial also gets its own video offset (video timestamp minus mocap time of
the same instant), searched over the whole overlap: the synchronisation map
of ``process_manip_v2.py`` is sometimes aligned on the wrong trigger edge.
Lengths are in mm, the lab frame is the QTM one (z up).
"""

from __future__ import annotations

import itertools
import warnings
import json
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares, linear_sum_assignment

from data_preparation_utils import marker_arrays, rigid_foot_frames

SIDES = ("L", "R")
# InstantHMR keypoints per foot, in the order of ``tools/instanthmr_feet.py`` (L then R).
FOOT_KEYPOINTS = ("ankle", "big_toe_tip", "small_toe_tip", "heel")
CORNER_SIGMA_PX = 2.0
KEYPOINT_SIGMA_PX = 8.0
MOCAP_TOLERANCE_S = 0.010
# The cameras stand on tripods or shelves: their centre is above the floor.
MIN_CAMERA_HEIGHT_MM = 300.0
COARSE_OFFSET_STEP_S = 0.1
COARSE_FRAME_STEP = 12


@dataclass
class Camera:
    """Pinhole camera with two radial terms; ``rvec``, ``tvec`` map lab (mm) to camera."""

    focal_px: float
    cx: float
    cy: float
    k1: float
    k2: float
    rvec: list[float]
    tvec: list[float]

    def vector(self) -> np.ndarray:
        return np.array([self.focal_px, self.cx, self.cy, self.k1, self.k2, *self.rvec, *self.tvec])

    @classmethod
    def from_vector(cls, values) -> Camera:
        values = [float(value) for value in values]
        return cls(*values[:5], values[5:8], values[8:11])

    def to_camera(self, points: np.ndarray) -> np.ndarray:
        rotation, _ = cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64))
        return points @ rotation.T + np.asarray(self.tvec)

    def project(self, points: np.ndarray) -> np.ndarray:
        """Pixel coordinates of lab points ``(..., 3)``; NaN behind the camera."""
        camera = self.to_camera(points)
        depth = np.where(camera[..., 2] > 1.0, camera[..., 2], np.nan)
        x, y = camera[..., 0] / depth, camera[..., 1] / depth
        r2 = x * x + y * y
        distortion = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        return np.stack(
            (self.focal_px * x * distortion + self.cx, self.focal_px * y * distortion + self.cy), axis=-1
        )

    def centre(self) -> np.ndarray:
        """Optical centre in the lab frame (mm)."""
        rotation, _ = cv2.Rodrigues(np.asarray(self.rvec, dtype=np.float64))
        return -rotation.T @ np.asarray(self.tvec)

    def save(self, path: Path, **extra) -> None:
        path.write_text(json.dumps({**asdict(self), **extra}, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Camera:
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{name: values[name] for name in cls.__dataclass_fields__})


def plate_corners(trial_dir: Path) -> np.ndarray:
    """The eight top-surface corners (8, 3) of the two plates, lab frame, mm."""
    report = json.loads((trial_dir / "synchronized_v2" / "rapport_synchronisation.json").read_text())
    return np.concatenate([np.asarray(plate["corners_global_mm"]) for plate in report["platform_geometry"]])


def distinct_clicks(clicks: np.ndarray, tolerance_px: float = 3.0) -> np.ndarray:
    """The clicks without repeats (a double click lands twice on one corner)."""
    kept: list[np.ndarray] = []
    for click in clicks:
        if all(np.linalg.norm(click - other) > tolerance_px for other in kept):
            kept.append(click)
    return np.array(kept)


def widest_quadrilateral(points: np.ndarray) -> list[int]:
    """Indexes of the four points whose convex hull has the largest area.

    Far-apart points are not enough: three of them can lie on one plate edge,
    and four points with three collinear make planar PnP degenerate.
    """
    return list(max(
        itertools.combinations(range(len(points)), 4),
        key=lambda chosen: cv2.contourArea(cv2.convexHull(points[list(chosen)].astype(np.float32))),
    ))


def corner_hypotheses(clicks: np.ndarray, corners: np.ndarray, image_size) -> list[tuple[float, Camera, list[int]]]:
    """Plausible (rms px, camera, click -> corner index) for free-order clicks.

    The four clicks spanning the widest quadrilateral are tried against every ordered choice of four corners,
    over a grid of focal lengths; the remaining clicks go to the projected
    corners by optimal one-to-one assignment. Corners symmetric on the floor give several hypotheses
    (plates swapped): the feet decide between them.
    """
    width, height = image_size
    base = widest_quadrilateral(clicks)
    hypotheses: dict[tuple[int, ...], tuple[float, Camera]] = {}
    # IPPE needs exactly coplanar points (the corners differ by a few mm in height), and
    # on four points off the z = 0 plane OpenCV's IPPE returns wrong poses: solve at
    # z = 0, then lift the pose to the corners' mean height.
    floor = np.array([0.0, 0.0, corners[:, 2].mean()])
    flat = corners.copy()
    flat[:, 2] = 0.0
    for focal in np.geomspace(400.0, 2500.0, 15):
        matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        for chosen in itertools.permutations(range(len(corners)), 4):
            # Both IPPE solutions: the planar pose is ambiguous up to a mirror through the floor.
            count, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                flat[list(chosen)], clicks[base], matrix, None, flags=cv2.SOLVEPNP_IPPE
            )
            for rvec, tvec in zip(rvecs[:count], tvecs[:count]):
                tvec = tvec.ravel() - cv2.Rodrigues(rvec)[0] @ floor
                camera = Camera(focal, width / 2, height / 2, 0.0, 0.0, rvec.ravel().tolist(), tvec.tolist())
                if camera.centre()[2] < MIN_CAMERA_HEIGHT_MM:
                    continue
                projected = camera.project(corners)
                if not np.isfinite(projected).all():
                    continue
                distance = np.linalg.norm(clicks[:, None] - projected[None], axis=2)
                rows, assignment = linear_sum_assignment(distance)
                rms = float(np.sqrt(np.mean(distance[rows, assignment] ** 2)))
                key = tuple(assignment.tolist())
                if key not in hypotheses or rms < hypotheses[key][0]:
                    hypotheses[key] = (rms, camera)
    ranked = sorted((rms, camera, list(key)) for key, (rms, camera) in hypotheses.items())
    best = ranked[0][0]
    return [item for item in ranked if item[0] <= max(3.0 * best, 6.0)][:6]


@dataclass
class FootObservations:
    """Matched video frames of the session: foot frames and InstantHMR keypoints."""

    rotation: np.ndarray  # (n, 2, 3, 3) local -> lab, L then R
    origin: np.ndarray  # (n, 2, 3) mm
    keypoints: np.ndarray  # (n, persons, 8, 2) px, L then R
    marker_local: dict[str, np.ndarray]  # side -> (4, 3) initial local points from the markers


def foot_frames(mocap: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Rigid foot frames of both feet, ``(n, 2, 3, 3)`` and ``(n, 2, 3)``."""
    frames = [rigid_foot_frames(mocap, side) for side in SIDES]
    return np.stack([frame[0] for frame in frames], axis=1), np.stack([frame[1] for frame in frames], axis=1)


def marker_local_points(mocap: pd.DataFrame, rotation: np.ndarray, origin: np.ndarray) -> dict[str, np.ndarray]:
    """First guess of the keypoints in each foot frame: ankle above the heel, toes, heel."""
    result = {}
    for index, side in enumerate(SIDES):
        local = {}
        for name, values in marker_arrays(mocap, side).items():
            points = np.einsum("nij,ni->nj", rotation[:, index], values - origin[:, index])
            local[name] = np.nanmedian(points, axis=0)

        def find(keyword):
            return np.mean([value for name, value in local.items() if keyword in name], axis=0)

        heel, toe, lateral = find("heel"), find("toe"), find("lateral")
        result[side] = np.stack((heel + np.array([40.0, 0.0, 60.0]), toe, lateral, heel))
    return result


@cache
def trial_mocap(trial_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """QTM absolute times, foot frames and first-guess keypoint offsets of one trial."""
    mocap = pd.read_csv(trial_dir / "synchronized_v2" / "mocap_synchronisee.csv")
    rotation, origin = foot_frames(mocap)
    return mocap["time_qtm_absolute_s"].to_numpy(), rotation, origin, marker_local_points(mocap, rotation, origin)


def nearest_rows(reference_times: np.ndarray, times: np.ndarray, tolerance_s: float) -> np.ndarray:
    """Index of the nearest ``reference_times`` sample to each of ``times``; -1 beyond ``tolerance_s``."""
    rows = np.clip(np.searchsorted(reference_times, times), 1, len(reference_times) - 1)
    rows = np.where(np.abs(reference_times[rows - 1] - times) < np.abs(reference_times[rows] - times), rows - 1, rows)
    return np.where(np.abs(reference_times[rows] - times) < tolerance_s, rows, -1)


@dataclass
class TrialVideo:
    """One trial of the session: its mocap, the video timestamps of its map and its keypoints."""

    name: str
    mocap_times: np.ndarray
    rotation: np.ndarray
    origin: np.ndarray
    marker_local: dict[str, np.ndarray]
    frame_times: np.ndarray  # QTM absolute time of each video frame in the synchronisation map
    keypoints: np.ndarray  # (frames, persons, 8, 2)

    @classmethod
    def load(cls, trial_dir: Path, camera: str, keypoint_file: Path, name: str | None = None) -> TrialVideo:
        mocap_times, rotation, origin, local = trial_mocap(trial_dir)
        video = pd.read_csv(trial_dir / "synchronized_v2" / f"{camera}_manip_frames_synchronises.csv")
        detected = np.load(keypoint_file)["joints_2d"].astype(np.float64)
        count = min(len(video), len(detected))
        return cls(name or trial_dir.name, mocap_times, rotation, origin, local,
                   video["time_qtm_absolute_s"].to_numpy()[:count], detected[:count])

    def observations(self, offset_s: float, step: int) -> FootObservations:
        """Foot frames at each ``step``-th video frame, a frame stamped T showing the mocap at T - ``offset_s``."""
        frames = np.arange(0, len(self.frame_times), step)
        rows = nearest_rows(self.mocap_times, self.frame_times[frames] - offset_s, MOCAP_TOLERANCE_S)
        valid = rows >= 0
        valid[valid] = np.isfinite(self.origin[rows[valid]]).all(axis=(1, 2))
        return FootObservations(
            self.rotation[rows[valid]], self.origin[rows[valid]], self.keypoints[frames[valid]], self.marker_local
        )

    def offset_range(self) -> tuple[float, float]:
        """Offsets for which video and mocap overlap."""
        return self.frame_times[0] - self.mocap_times[-1], self.frame_times[-1] - self.mocap_times[0]


def session_observations(videos: list[TrialVideo], offsets: list[float], step: int) -> FootObservations:
    parts = [video.observations(offset, step) for video, offset in zip(videos, offsets)]
    return FootObservations(
        np.concatenate([part.rotation for part in parts]), np.concatenate([part.origin for part in parts]),
        np.concatenate([part.keypoints for part in parts]),
        {side: np.mean([video.marker_local[side] for video in videos], axis=0) for side in SIDES},
    )


def coarse_offset(video: TrialVideo, camera: Camera, local: np.ndarray, gate_px: float) -> tuple[float, int]:
    """Offset (s, 0.1 s grid over the whole overlap) keeping the most frames, and that count.

    The synchronisation map may be off by whole trigger intervals (tens of
    seconds), so the whole range where video and mocap overlap is searched.
    """
    low, high = video.offset_range()
    best = (0.0, -1)
    for offset in np.arange(np.floor(low * 10) / 10, high, COARSE_OFFSET_STEP_S):
        observations = video.observations(offset, COARSE_FRAME_STEP)
        if len(observations.origin) < 20:
            continue
        _, kept = select_people(observations, camera, local, gate_px)
        count = round(kept * len(observations.origin))
        if count > best[1]:
            best = (float(offset), count)
    return best


def fine_offset(video: TrialVideo, camera: Camera, local: np.ndarray, centre_s: float, step: int) -> float:
    """Offset within 0.2 s of ``centre_s`` (5 ms grid) minimising the mean keypoint error."""
    scores = []
    for offset in np.arange(centre_s - 0.2, centre_s + 0.2001, 0.005):
        observations = video.observations(offset, step)
        chosen, _ = select_people(observations, camera, local, 25.0)
        error = np.linalg.norm(camera.project(keypoints_lab(observations, local)) - chosen, axis=2)
        scores.append((float(np.nanmean(error)), float(offset)))
    return min(scores)[1]


def keypoints_lab(observations: FootObservations, local: np.ndarray) -> np.ndarray:
    """Lab position (n, 8, 3) of the 8 keypoints, ``local`` (2, 4, 3) in their foot frames."""
    points = np.einsum("nsij,skj->nski", observations.rotation, local) + observations.origin[:, :, None]
    return points.reshape(len(points), 8, 3)


def select_people(observations: FootObservations, camera: Camera, local: np.ndarray, gate_px: float):
    """Per frame, the person and side assignment closest to the projected feet.

    Returns the chosen keypoints (n, 8, 2) (NaN where nothing fits within
    ``gate_px`` median distance) and the fraction of frames kept.
    """
    projected = camera.project(keypoints_lab(observations, local))  # (n, 8, 2)
    swap = np.r_[4:8, 0:4]
    best = np.full(projected.shape, np.nan)
    best_cost = np.full(len(projected), np.inf)
    for person in range(observations.keypoints.shape[1]):
        for order in (np.arange(8), swap):
            candidate = observations.keypoints[:, person][:, order]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # frames without that person
                cost = np.nanmedian(np.linalg.norm(candidate - projected, axis=2), axis=1)
            better = np.isfinite(cost) & (cost < best_cost)
            best[better], best_cost[better] = candidate[better], cost[better]
    best[~(best_cost < gate_px)] = np.nan
    return best, float(np.mean(best_cost < gate_px))


def fit(observations, clicks, target_corners, camera: Camera, local, chosen, image_size, free_camera, free_local,
        focal_prior: float | None = None):
    """Robust least squares of the ``free_camera`` parameters (indexes of ``Camera.vector``) and,
    with ``free_local``, the keypoint offsets; ``chosen`` None fits the corners alone, ``clicks``
    None the keypoints alone. ``focal_prior`` (px) holds the focal within about 10 %.

    The corners are weighted as much as all the keypoints together, so the many
    keypoints refine the camera without dragging the ground away.
    """
    width, height = image_size
    base = np.concatenate((camera.vector(), local.ravel()))
    free = np.array(list(free_camera) + (list(range(11, 35)) if free_local else []), dtype=int)
    prior = local.ravel().copy()
    used = None if chosen is None else np.isfinite(chosen).all(axis=2)
    corner_weight = 0.0 if clicks is None else 1.0 if used is None else max(1.0, np.sqrt(used.sum() / len(clicks)))
    # A prior weighted like a handful of frames would vanish under thousands of keypoints.
    corner_weight_free = 1.0 if used is None else max(1.0, np.sqrt(used.sum() / 8.0))

    def unpack(values):
        full = base.copy()
        full[free] = values
        return Camera.from_vector(full[:11]), full[11:].reshape(2, 4, 3)

    def residuals(values):
        current, points = unpack(values)
        parts = [] if clicks is None else [
            corner_weight * (current.project(target_corners) - clicks).ravel() / CORNER_SIGMA_PX
        ]
        if used is not None:
            projected = current.project(keypoints_lab(observations, points))
            parts.append(((projected - chosen)[used]).ravel() / KEYPOINT_SIGMA_PX)
            parts.append((points.ravel() - prior) / 40.0)
        parts.append([(current.cx - width / 2) / 20.0, (current.cy - height / 2) / 20.0, current.k1 / 0.1, current.k2 / 0.1])
        if focal_prior is not None:
            parts.append([corner_weight_free * (current.focal_px - focal_prior) / (0.1 * focal_prior)])
        return np.nan_to_num(np.concatenate(parts), nan=1e3)

    result = least_squares(residuals, base[free], loss="soft_l1", f_scale=3.0, x_scale="jac", max_nfev=100)
    return unpack(result.x)


FOCAL_POSE = (0, 5, 6, 7, 8, 9, 10)
ALL_CAMERA = tuple(range(11))


def calibrate(trial_dirs, camera_name, keypoint_files, clicks_file: Path, step: int = 6):
    """Camera, keypoint offsets, per-trial video offsets (s) and a quality report for one session camera.

    Stages: (1) focal and pose from the corners alone, the corner hypothesis
    chosen by how many frames its feet explain once each trial's video offset
    is searched coarsely; (2) keypoint offsets with that camera, the people
    re-chosen under tightening gates, then each video offset refined to 5 ms
    (it holds the camera latency, tens of ms, and any error of the
    synchronisation map); (3) everything together, kept only if the corners
    still fit within 3 px rms.
    """
    clicked = json.loads(clicks_file.read_text(encoding="utf-8"))
    clicks, image_size = distinct_clicks(np.asarray(clicked["corners_px"], dtype=np.float64)), tuple(clicked["image_size"])
    corners = plate_corners(trial_dirs[0])
    videos = [TrialVideo.load(trial, camera_name, keypoints) for trial, keypoints in zip(trial_dirs, keypoint_files)]
    initial_local = np.stack([
        np.mean([video.marker_local[side] for video in videos], axis=0) for side in SIDES
    ])
    reference = session_observations(videos, [0.0] * len(videos), step)

    candidates = []
    for corner_rms, camera, assignment in corner_hypotheses(clicks, corners, image_size):
        camera, _ = fit(reference, clicks, corners[assignment], camera, initial_local, None, image_size,
                        FOCAL_POSE, False)
        found = [coarse_offset(video, camera, initial_local, 60.0) for video in videos]
        candidates.append((sum(count for _, count in found), corner_rms, camera, assignment, [o for o, _ in found]))
    _, corner_rms, camera, assignment, offsets = max(candidates, key=lambda item: (item[0], -item[1]))
    target_corners = corners[assignment]

    local = initial_local
    observations = session_observations(videos, offsets, step)
    for gate in (80.0, 40.0, 25.0):
        chosen, kept = select_people(observations, camera, local, gate)
        _, local = fit(observations, clicks, target_corners, camera, local, chosen, image_size, (), True)
    offsets = [fine_offset(video, camera, local, offset, step) for video, offset in zip(videos, offsets)]
    observations = session_observations(videos, offsets, step)
    for gate in (40.0, 25.0):
        chosen, kept = select_people(observations, camera, local, gate)
        _, local = fit(observations, clicks, target_corners, camera, local, chosen, image_size, (), True)
    joint_camera, joint_local = fit(
        observations, clicks, target_corners, camera, local, chosen, image_size, ALL_CAMERA, True
    )
    joint_rms = float(np.sqrt(np.mean(np.sum((joint_camera.project(target_corners) - clicks) ** 2, axis=1))))
    joint_used = joint_rms <= 3.0
    if joint_used:
        camera, local = joint_camera, joint_local
    offsets = {
        video.name: round(fine_offset(video, camera, local, offset, step), 3) for video, offset in zip(videos, offsets)
    }
    observations = session_observations(videos, list(offsets.values()), step)
    chosen, kept = select_people(observations, camera, local, 25.0)
    projected = camera.project(keypoints_lab(observations, local))
    errors = np.linalg.norm(projected - chosen, axis=2)
    corner_errors = np.linalg.norm(camera.project(corners[assignment]) - clicks, axis=1)
    report = {
        "video_offset_s": offsets,
        "map_start_s": {video.name: round(float(video.frame_times[0]), 3) for video in videos},
        "corner_assignment": assignment,
        "joint_refinement_used": joint_used,
        "joint_corner_rms_px": round(joint_rms, 2),
        "corner_error_px": [round(float(value), 2) for value in corner_errors],
        "keypoint_median_error_px": float(np.nanmedian(errors)),
        "keypoint_p90_error_px": float(np.nanpercentile(errors, 90)),
        "frames": int(len(chosen)),
        "frames_kept": float(kept),
        "keypoint_local_mm": {side: local[index].round(1).tolist() for index, side in enumerate(SIDES)},
        "hypotheses": [
            {"frames_explained": item[0], "corner_rms_px": round(item[1], 2), "assignment": item[3]}
            for item in candidates
        ],
    }
    return camera, local, offsets, report


def calibrate_from_feet(videos: list[TrialVideo], local: np.ndarray, offsets: dict[str, float], image_size,
                        focal_prior: float, step: int = 4):
    """Calibrate a second camera of a session without clicks: the feet are the target.

    The keypoint offsets ``local`` (2, 4, 3) in the foot frames come from the
    first camera's calibration, so every frame gives eight known lab points.
    Both cameras are the same model, so the focal is held near the first
    one's (``focal_prior``): with feet that barely move (squat) nothing else
    pins it. (1) PnP RANSAC on every person and side order of every frame
    (bystanders and swapped sides are outliers) at that focal, with each
    trial's video offset taken from the first camera; (2) the people
    re-chosen and the whole camera fitted under tightening gates; (3) each
    video offset refined, then the camera fitted again. The plate corners are
    not used, so their reprojection checks the result.
    """
    width, height = image_size
    observations = session_observations(videos, [offsets[video.name] for video in videos], step)
    lab = keypoints_lab(observations, local)
    swap = np.r_[4:8, 0:4]
    object_points, image_points = [], []
    for person in range(observations.keypoints.shape[1]):
        for order in (np.arange(8), swap):
            pixels = observations.keypoints[:, person][:, order]
            valid = np.isfinite(pixels).all(axis=2) & np.isfinite(lab).all(axis=2)
            object_points.append(lab[valid])
            image_points.append(pixels[valid])
    object_points, image_points = np.concatenate(object_points), np.concatenate(image_points)
    best = (-1, None)
    for focal in (focal_prior,):
        matrix = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points, image_points, matrix, None, iterationsCount=3000, reprojectionError=12.0,
            confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
        )
        if ok and inliers is not None and len(inliers) > best[0]:
            best = (len(inliers), Camera(focal, width / 2, height / 2, 0.0, 0.0, rvec.ravel().tolist(),
                                         tvec.ravel().tolist()))
    camera = best[1]
    for gate in (60.0, 40.0, 25.0):
        chosen, kept = select_people(observations, camera, local, gate)
        camera, _ = fit(observations, None, None, camera, local, chosen, image_size, ALL_CAMERA, False, focal_prior)
    found = {video.name: fine_offset(video, camera, local, offsets[video.name], step) for video in videos}
    observations = session_observations(videos, list(found.values()), step)
    for gate in (25.0, 25.0):
        chosen, kept = select_people(observations, camera, local, gate)
        camera, _ = fit(observations, None, None, camera, local, chosen, image_size, ALL_CAMERA, False, focal_prior)
    found = {name: round(fine_offset(video, camera, local, offset, step), 3)
             for video, (name, offset) in zip(videos, found.items())}
    observations = session_observations(videos, list(found.values()), step)
    chosen, kept = select_people(observations, camera, local, 25.0)
    errors = np.linalg.norm(camera.project(keypoints_lab(observations, local)) - chosen, axis=2)
    report = {
        "video_offset_s": found,
        "pnp_inliers": best[0],
        "keypoint_median_error_px": float(np.nanmedian(errors)),
        "keypoint_p90_error_px": float(np.nanpercentile(errors, 90)),
        "frames": int(len(chosen)),
        "frames_kept": float(kept),
    }
    return camera, found, report
