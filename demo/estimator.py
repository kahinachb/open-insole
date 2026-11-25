"""Deployment of a trained GRFM model: one sample in, the 12 GRFM components out.

The checkpoint carries everything the inputs need: the channel order, the
standardisation statistics and the training options. At Deployment the
full Sensor configuration is present (every presence column 1) and the body
weight is known (missing flag 0), so the window is built as in training with
no Sensor dropout. Inactive sensors are not detected live.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "insole2grfm"))

from grfm_schema import BODY_WEIGHT_FEATURE, BODY_WEIGHT_MISSING_FEATURE  # noqa: E402
from train_lstm_two_feet_masked import PRESENCE_PREFIX, model_from_args  # noqa: E402


class GRFMEstimator:
    def __init__(self, checkpoint: Path, body_weight_n: float):
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not saved.get("causal", False):
            raise ValueError("the demo needs a causal model (it only sees past samples)")
        if saved.get("input_features") != "raw":
            raise ValueError(f"unsupported input representation: {saved.get('input_features')}")
        self.feature_names = list(saved["feature_names"])
        self.channels = list(saved["source_feature_names"])
        self.targets = list(saved["targets"])
        self.window = int(saved["window_samples"])
        self.model = model_from_args(
            SimpleNamespace(**saved), saved["input_size"], len(self.targets),
            feature_names=self.feature_names, targets=self.targets,
        )
        self.model.load_state_dict(saved["state_dict"])
        self.model.eval()
        torch.set_num_threads(1)

        x_mean, x_std = np.asarray(saved["x_mean"], np.float32), np.asarray(saved["x_std"], np.float32)
        self.y_mean, self.y_std = np.asarray(saved["y_mean"], np.float32), np.asarray(saved["y_std"], np.float32)
        self.channel_columns = np.array([self.feature_names.index(name) for name in self.channels])
        self.channel_mean, self.channel_std = x_mean[self.channel_columns], x_std[self.channel_columns]

        # The constant part of every row: body weight, its missing flag, the presence columns.
        self.constant = np.zeros(len(self.feature_names), np.float32)
        if BODY_WEIGHT_FEATURE in self.feature_names:
            column = self.feature_names.index(BODY_WEIGHT_FEATURE)
            self.constant[column] = (body_weight_n - x_mean[column]) / x_std[column]
        if BODY_WEIGHT_MISSING_FEATURE in self.feature_names:
            self.constant[self.feature_names.index(BODY_WEIGHT_MISSING_FEATURE)] = 0.0
        for column, name in enumerate(self.feature_names):
            if name.startswith(PRESENCE_PREFIX):
                self.constant[column] = 1.0
        self.rows: deque[np.ndarray] = deque(maxlen=self.window)

    def push(self, channels: dict[str, float]) -> np.ndarray | None:
        """Add one sample; the GRFM (12, N and N·m, in ``targets`` order) once the window is full."""
        row = self.constant.copy()
        values = np.array([channels[name] for name in self.channels], np.float32)
        row[self.channel_columns] = (values - self.channel_mean) / self.channel_std
        self.rows.append(row)
        if len(self.rows) < self.window:
            return None
        with torch.no_grad():
            output = self.model(torch.from_numpy(np.stack(self.rows))[None])[0].numpy()
        return output * self.y_std + self.y_mean
