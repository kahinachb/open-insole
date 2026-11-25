#!/usr/bin/env python3
"""Top view of the two insoles: load density of the gauges, total load and Centre of pressure.

Each gauge load is spread by a normalised Gaussian kernel (sigma = 14 mm, as
``tools/insole_viewer.html``), so the density reads in kPa and integrates to
the gauge sum. Toes up, both feet seen from above, left foot on the left. The
colour scale is fixed across trials (square root, 0 to ``DENSITY_MAX_KPA``) so
that frames of different trials and Participants compare.

``InsoleHeatmap.render`` returns one RGB frame; ``grfm_rerun.py`` encodes them
as the insole video of a trial.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from grfm_schema import FEET

LAYOUT = Path(__file__).resolve().parent / "tools" / "insole_layout.json"
FONT = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
SIGMA_MM = 14.0
DENSITY_MAX_KPA = 200.0
MIN_COP_LOAD_N = 20.0
TRAIL_S = 0.6

WIDTH, HEIGHT = 640, 720
PX_PER_MM = 2.0
TOP_PX, SOLE_PX = 64, 560
CENTRES_PX = {"L": 190, "R": 400}
BAR_X, BAR_W = 548, 10

# Neutral canvas, the insole blue for everything the insoles measure.
CANVAS = (255, 255, 255)
SOLE = (242, 242, 247)
OUTLINE = (199, 199, 204)
SENSOR = (174, 174, 178)
INK = (29, 29, 31)
MUTED = (134, 134, 139)
ACCENT = (0, 113, 227)
RAMP = np.array([
    (242, 242, 247), (204, 224, 248), (125, 178, 240), (0, 113, 227), (0, 64, 160), (8, 32, 92),
], float)


def ramp(values: np.ndarray) -> np.ndarray:
    """[0, 1] -> uint8 RGB along ``RAMP``."""
    position = np.clip(values, 0.0, 1.0) * (len(RAMP) - 1)
    low = np.floor(position).astype(int).clip(0, len(RAMP) - 2)
    weight = (position - low)[..., None]
    return ((1 - weight) * RAMP[low] + weight * RAMP[low + 1]).astype(np.uint8)


class InsoleHeatmap:
    """Renderer of the two insoles for the gauge columns of a prepared trial's ``X``."""

    def __init__(self, layout: Path = LAYOUT, mirror: bool = False):
        """``mirror`` flips each insole about its own long axis (left foot still on the left)."""
        sign = -1.0 if mirror else 1.0
        spec = json.loads(layout.read_text())
        names = spec["feature_names"]
        self.columns, self.kernels, self.boxes, self.sensors, self.to_px, self.outline = {}, {}, {}, {}, {}, {}
        alpha = np.zeros((HEIGHT, WIDTH), np.float32)
        for side in FEET:
            foot = spec["feet"][side]
            sensors = np.array([[s["x_mm"], s["y_mm"]] for s in foot["sensors"]])
            outline = np.array(foot["outline"], float)
            self.columns[side] = [names.index(s["channel"]) for s in foot["sensors"]]
            middle = (outline[:, 1].max() + outline[:, 1].min()) / 2.0
            centre, base = CENTRES_PX[side], TOP_PX + SOLE_PX

            def to_px(points, centre=centre, middle=middle, base=base):
                points = np.asarray(points, float)
                return np.stack((centre - sign * (points[..., 1] - middle) * PX_PER_MM, base - points[..., 0] * PX_PER_MM), -1)

            self.to_px[side] = to_px
            polygon = to_px(outline)
            # Anti-aliased sole mask: the polygon at 4x, averaged down.
            fine = np.zeros((HEIGHT * 4, WIDTH * 4), np.uint8)
            cv2.fillPoly(fine, [np.round(polygon * 4).astype(np.int32)], 255)
            mask = cv2.resize(fine, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            alpha = np.maximum(alpha, mask)
            x0, y0 = np.floor(polygon.min(axis=0)).astype(int) - 1
            x1, y1 = np.ceil(polygon.max(axis=0)).astype(int) + 2
            self.boxes[side] = (x0, y0, x1, y1)
            # Lab mm of every pixel of the box, then the kernel matrix (pixels, sensors) in 1/mm^2.
            u, v = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
            x_mm = (base - v) / PX_PER_MM
            y_mm = middle - sign * (u - centre) / PX_PER_MM
            distance = (x_mm[..., None] - sensors[:, 0]) ** 2 + (y_mm[..., None] - sensors[:, 1]) ** 2
            self.kernels[side] = (np.exp(-distance / (2 * SIGMA_MM ** 2)) / (2 * np.pi * SIGMA_MM ** 2)).reshape(
                -1, len(sensors)).astype(np.float32)
            self.sensors[side] = (to_px(sensors), sensors)
            self.outline[side] = polygon
        self.mask = alpha
        self.background = self._background()
        self.fonts = {size: ImageFont.truetype(FONT, size) for size in (21, 24, 40)}

    def loads(self, x: np.ndarray, side: str) -> np.ndarray:
        """(n, sensors) gauge loads (N, negatives clipped) of ``X`` rows."""
        return np.clip(np.asarray(x)[..., self.columns[side]], 0.0, None)

    def cop_mm(self, loads: np.ndarray, side: str) -> np.ndarray:
        """(n, 2) insole-frame Centre of pressure of the gauges, NaN under ``MIN_COP_LOAD_N``."""
        total = loads.sum(axis=-1, keepdims=True)
        cop = loads @ self.sensors[side][1] / np.maximum(total, 1e-6)
        cop[total[..., 0] < MIN_COP_LOAD_N] = np.nan
        return cop

    def _background(self) -> np.ndarray:
        """The static layer: captions, empty soles and the colour bar."""
        image = np.full((HEIGHT, WIDTH, 3), CANVAS, np.uint8)
        image = (image * (1 - self.mask[..., None]) + np.array(SOLE) * self.mask[..., None]).astype(np.uint8)
        top, bottom = TOP_PX + 40, TOP_PX + SOLE_PX - 40
        bar = ramp(np.linspace(1.0, 0.0, bottom - top)[:, None].repeat(BAR_W, 1))
        image[top:bottom, BAR_X:BAR_X + BAR_W] = bar
        return image

    def render(self, loads: dict, trails: dict) -> np.ndarray:
        """One frame: ``loads[side]`` (sensors,) N; ``trails[side]`` (k, 2) insole mm, oldest first."""
        image = self.background.astype(np.float32)
        for side in FEET:
            x0, y0, x1, y1 = self.boxes[side]
            density = (self.kernels[side] @ loads[side].astype(np.float32)).reshape(y1 - y0, x1 - x0) * 1000.0
            colour = ramp(np.sqrt(density / DENSITY_MAX_KPA)).astype(np.float32)
            alpha = self.mask[y0:y1, x0:x1, None]
            image[y0:y1, x0:x1] = image[y0:y1, x0:x1] * (1 - alpha) + colour * alpha
        canvas = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(canvas, "RGBA")
        for side in FEET:
            draw.line([tuple(p) for p in np.vstack((self.outline[side], self.outline[side][:1]))],
                      fill=OUTLINE, width=2, joint="curve")
            for u, v in self.sensors[side][0]:
                radius = 9.85 * PX_PER_MM
                draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=SENSOR + (90,), width=1)
            trail = trails[side]
            trail = trail[np.isfinite(trail).all(axis=1)]
            if len(trail):
                points = self.to_px[side](trail)
                for k in range(1, len(points)):
                    shade = int(40 + 180 * k / len(points))
                    draw.line([tuple(points[k - 1]), tuple(points[k])], fill=ACCENT + (shade,), width=3)
                u, v = points[-1]
                draw.ellipse((u - 8, v - 8, u + 8, v + 8), fill=(255, 255, 255, 255))
                draw.ellipse((u - 6, v - 6, u + 6, v + 6), fill=ACCENT + (255,))
            centre = CENTRES_PX[side]
            draw.text((centre, TOP_PX - 24), "Left" if side == "L" else "Right", fill=MUTED,
                      font=self.fonts[24], anchor="ms")
            total = float(loads[side].sum())
            draw.text((centre, TOP_PX + SOLE_PX + 56), f"{total:,.0f}".replace(",", " "), fill=INK,
                      font=self.fonts[40], anchor="rs")
            draw.text((centre + 8, TOP_PX + SOLE_PX + 56), "N", fill=MUTED, font=self.fonts[24], anchor="ls")
        top, bottom = TOP_PX + 40, TOP_PX + SOLE_PX - 40
        for value in (0, 50, 100, 150, 200):
            v = bottom - np.sqrt(value / DENSITY_MAX_KPA) * (bottom - top)
            draw.line([(BAR_X + BAR_W, v), (BAR_X + BAR_W + 4, v)], fill=MUTED, width=1)
            draw.text((BAR_X + BAR_W + 8, v), f"{value}", fill=MUTED, font=self.fonts[21], anchor="lm")
        draw.text((BAR_X + BAR_W / 2, top - 18), "kPa", fill=MUTED, font=self.fonts[21], anchor="ms")
        return np.asarray(canvas)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz", type=Path, help="prepared trial, e.g. ../dataset_v3.7.1/P02_squat.npz")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True, help="PNG")
    args = parser.parse_args()
    heatmap = InsoleHeatmap()
    x = np.load(args.npz)["X"]
    loads = {side: heatmap.loads(x[args.row], side) for side in FEET}
    trail = int(TRAIL_S * 200)
    trails = {side: heatmap.cop_mm(heatmap.loads(x[max(args.row - trail, 0):args.row + 1], side), side)
              for side in FEET}
    cv2.imwrite(str(args.output), cv2.cvtColor(heatmap.render(loads, trails), cv2.COLOR_RGB2BGR))
