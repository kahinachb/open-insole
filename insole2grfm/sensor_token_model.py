"""Sensor-token Transformer ("model B") for the masked training engine.

Pipeline for a causal window ``x (B, W, F)`` standardised by the engine:

1. undo the engine's input standardisation (buffers ``x_mean``/``x_std``) and
   turn the 78 raw channels into 38 Sensor tokens in the Canonical insole
   frame (:mod:`sensor_tokens`); the per-trial Inactive-sensor flags travel as
   extra input columns ``inactive_<gauge channel>`` (see
   :func:`append_inactive_flags`);
2. a shared causal temporal encoder (dilated causal 1D convolutions over the
   window) per token, read at the current sample;
3. spatial attention over the 38 Sensor tokens + 2 Readout tokens (one per
   foot), masking inactive tokens and, during training, 10 % randomly dropped
   tokens. Foot membership enters attention only as a same-foot / other-foot
   relation, and descriptors use the mirror-symmetric geometry, so the model
   treats both feet with the same function;
4. each Readout token is decoded into its foot's 6 GRFM components in the
   canonical frame, mirrored back to the subject frame (Fy, Mx, Mz of the
   right foot negated) and re-standardised with the engine's target
   statistics, so the engine's masked loss and predictions are unchanged.

With the engine's ``--body-weight-input``, the standardised body-weight
column (``BODY_WEIGHT_FEATURE``, constant per Subject) is a global input to
both Readout tokens: a linear projection of it is added to the Readout
embedding before the spatial attention. The same term goes to both feet, so
the model stays mirror consistent. Without the column, nothing is created.
With ``--body-weight-masking``, the "body weight missing" flag column
(``BODY_WEIGHT_MISSING_FEATURE``) is a second input of that projection.

The per-foot Readout embeddings (:meth:`SensorTokenTransformer.readout_embeddings`)
are the extension point for extra per-foot outputs. With the engine's
``--foot-height-output``, ``foot_height_head`` (``nn.Linear(hidden, 1)``, one
function shared by both feet) sits next to ``grfm_decoder`` on each foot's
projected Readout token and gives that foot's Foot height, scaled with the
train statistics pooled over both feet (mirror consistent) and re-standardised
with the engine's per-foot statistics. Without the option, nothing is created.

Design decisions:

- **No absolute foot one-hot (deliberate departure from spec story 7).** Story
  7 gives each Sensor token its foot as an input feature. Here the
  ``is_left`` / ``is_right`` descriptors are dropped (``FOOT_DESCRIPTORS``)
  and foot membership only enters attention as a learned same-foot /
  other-foot relation (:class:`FootRelationAttention`). With an absolute
  one-hot the model could learn a different function per foot; with the
  relative relation and mirror-symmetric descriptors it is exactly mirror
  consistent: swapping the feet (``SensorTokenizer.swap_feet``) mirrors the
  predictions. The cost: it can only fit mirror-symmetric mappings.
- **Pre-norm.** The spatial layers (:class:`SpatialEncoderLayer`) are pre-norm
  by construction (LayerNorm before attention and feed-forward), as the
  temporal Transformer is through ``norm_first=True``. The auto-correction's
  "pre-norm / input LayerNorm" step is therefore implemented as the engine
  flag ``--input-layer-norm``, which here normalises the Sensor-token
  embeddings entering the spatial layers (and, in the MLP / LSTM / TCN, the
  inputs; in the temporal Transformer, the input embedding).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from grfm_schema import BODY_WEIGHT_FEATURE, BODY_WEIGHT_MISSING_FEATURE, COMPONENTS, TARGETS
from sensor_tokens import (
    TARGET_MIRROR_SIGN,
    TOKEN_COUNT,
    VALUE_CHANNELS,
    SensorTokenizer,
)

READOUT_COUNT = 2
COMPONENTS_PER_FOOT = len(COMPONENTS)
INACTIVE_PREFIX = "inactive_"
# Foot membership reaches the model through the attention relation only.
FOOT_DESCRIPTORS = ("is_left", "is_right")


def inactive_flag_names(tokenizer: SensorTokenizer) -> list[str]:
    return [f"{INACTIVE_PREFIX}{tokenizer.token_names[i]}" for i in tokenizer.gauge_tokens]


def append_inactive_flags(acquisitions, feature_names: list[str], tokenizer: SensorTokenizer | None = None) -> list[str]:
    """Append the per-trial Inactive-sensor flags (one column per gauge) to each ``x``.

    The flag is computed once per trial on the raw channels (gauge p99 < 5 N)
    and repeated on every sample, so it reaches the model through the
    engine's windows. Returns the new feature names.
    """
    tokenizer = tokenizer or SensorTokenizer.from_layout()
    columns = _source_columns(feature_names, tokenizer)
    for item in acquisitions:
        flags = tokenizer.inactive(item.x[:, columns])[tokenizer.gauge_tokens]
        repeated = np.broadcast_to(flags.astype(item.x.dtype), (len(item.x), len(flags)))
        item.x = np.concatenate([item.x, repeated], axis=1)
    return list(feature_names) + inactive_flag_names(tokenizer)


def prepare_sensor_token_inputs(acquisitions, feature_names: list[str], targets) -> list[str]:
    """Engine hook: check the 12 targets and append the Inactive-sensor flags."""
    if list(targets) != list(TARGETS):
        raise ValueError(f"le modele sensor_token predit les 12 cibles dans l'ordre {list(TARGETS)}")
    return append_inactive_flags(acquisitions, feature_names)


def _source_columns(feature_names, tokenizer: SensorTokenizer) -> list[int]:
    missing = [name for name in tokenizer.feature_names if name not in feature_names]
    if missing:
        raise ValueError(f"canaux bruts absents pour les sensor tokens: {missing[:3]}...")
    return [list(feature_names).index(name) for name in tokenizer.feature_names]


class CausalTemporalEncoder(nn.Module):
    """Dilated causal convolutions shared by all tokens, read at the last sample."""

    def __init__(self, input_size: int, hidden_size: int, window_samples: int, dropout: float, kernel_size: int = 3):
        super().__init__()
        dilations, receptive = [1], kernel_size
        while receptive < window_samples:
            dilations.append(dilations[-1] * 2)
            receptive += (kernel_size - 1) * dilations[-1]
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.convolutions = nn.ModuleList(
            nn.Conv1d(input_size if i == 0 else hidden_size, hidden_size, kernel_size, dilation=d)
            for i, d in enumerate(dilations)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x (N, C, W)`` -> ``(N, hidden)`` at the last (current) sample."""
        h = x
        for index, (convolution, dilation) in enumerate(zip(self.convolutions, self.dilations)):
            out = F.gelu(convolution(F.pad(h, ((self.kernel_size - 1) * dilation, 0))))
            h = self.dropout(out) if index == 0 else h + self.dropout(out)
        return h[:, :, -1]


class FootRelationAttention(nn.Module):
    """Multi-head self-attention with a learned same-foot / other-foot relation.

    The relation adds a per-head key and value embedding (Shaw et al. 2018)
    for each of the two relations, so tokens know which foot a key belongs to
    relative to themselves, never in absolute L/R terms.
    """

    def __init__(self, hidden_size: int, heads: int, dropout: float):
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden-size doit etre divisible par transformer-heads")
        self.heads, self.head_size = heads, hidden_size // heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.relation_key = nn.Parameter(torch.zeros(2, heads, self.head_size))
        self.relation_value = nn.Parameter(torch.zeros(2, heads, self.head_size))
        nn.init.normal_(self.relation_key, std=0.02)
        nn.init.normal_(self.relation_value, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, same_foot: torch.Tensor, attend: torch.Tensor) -> torch.Tensor:
        batch, count, _ = h.shape
        q, k, v = self.qkv(h).view(batch, count, 3, self.heads, self.head_size).permute(2, 0, 3, 1, 4)
        scale = 1.0 / math.sqrt(self.head_size)
        relation = torch.einsum("bhnd,rhd->bhnr", q, self.relation_key)
        logits = (q @ k.transpose(-1, -2)) * scale + torch.where(
            same_foot, relation[..., :1], relation[..., 1:]
        ) * scale
        logits = logits.masked_fill(~attend[:, None, None, :], float("-inf"))
        weights = self.dropout(torch.softmax(logits, dim=-1))
        same = (weights * same_foot).sum(-1, keepdim=True)
        out = weights @ v + same * self.relation_value[0, :, None] + (weights.sum(-1, keepdim=True) - same) * self.relation_value[1, :, None]
        return self.output(out.transpose(1, 2).reshape(batch, count, -1))


class SpatialEncoderLayer(nn.Module):
    """Pre-norm Transformer layer over the tokens of one instant."""

    def __init__(self, hidden_size: int, heads: int, feedforward_size: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = FootRelationAttention(hidden_size, heads, dropout)
        self.feedforward_norm = nn.LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, same_foot, attend):
        h = h + self.dropout(self.attention(self.attention_norm(h), same_foot, attend))
        return h + self.dropout(self.feedforward(self.feedforward_norm(h)))


class SensorTokenTransformer(nn.Module):
    """Sensor tokens -> causal temporal encoder -> spatial attention -> per-foot readouts."""

    def __init__(
        self,
        feature_names: list[str],
        window_samples: int,
        hidden_size: int,
        layers: int,
        dropout: float,
        output_size: int,
        heads: int = 4,
        feedforward_size: int = 128,
        token_dropout: float = 0.1,
        tokenizer: SensorTokenizer | None = None,
        input_layer_norm: bool = False,
        foot_height_output: bool = False,
    ) -> None:
        super().__init__()
        if output_size != len(TARGETS):
            raise ValueError(f"le modele sensor_token predit les 12 cibles {TARGETS}")
        if not 0.0 <= token_dropout < 1.0:
            raise ValueError("token-dropout doit etre dans [0,1[")
        self.tokenizer = (tokenizer or SensorTokenizer.from_layout()).mirror_symmetric()
        self.source_columns = _source_columns(feature_names, self.tokenizer)
        flags = inactive_flag_names(self.tokenizer)
        missing = [name for name in flags if name not in feature_names]
        if missing:
            raise ValueError("colonnes inactive_* absentes: utiliser append_inactive_flags")
        self.flag_columns = [list(feature_names).index(name) for name in flags]
        self.gauge_tokens = torch.as_tensor(self.tokenizer.gauge_tokens)
        self.token_dropout = token_dropout
        input_size = len(feature_names)
        keep = [i for i, n in enumerate(self.tokenizer.descriptor_names) if n not in FOOT_DESCRIPTORS]
        self.register_buffer("descriptors", torch.as_tensor(self.tokenizer.descriptors[:, keep]))
        feet = np.r_[self.tokenizer.token_feet, np.arange(READOUT_COUNT)]
        self.register_buffer("same_foot", torch.as_tensor(feet[:, None] == feet[None, :]), persistent=False)
        self.register_buffer("mirror_sign", torch.as_tensor(TARGET_MIRROR_SIGN), persistent=False)
        # Statistics, filled by set_statistics() from the training split.
        self.register_buffer("x_mean", torch.zeros(input_size))
        self.register_buffer("x_std", torch.ones(input_size))
        self.register_buffer("y_mean", torch.zeros(output_size))
        self.register_buffer("y_std", torch.ones(output_size))
        self.register_buffer("value_mean", torch.zeros(TOKEN_COUNT, len(VALUE_CHANNELS)))
        self.register_buffer("value_std", torch.ones(TOKEN_COUNT, len(VALUE_CHANNELS)))
        self.register_buffer("canonical_mean", torch.zeros(COMPONENTS_PER_FOOT))
        self.register_buffer("canonical_std", torch.ones(COMPONENTS_PER_FOOT))

        self.temporal = CausalTemporalEncoder(
            len(VALUE_CHANNELS) + self.descriptors.shape[1], hidden_size, window_samples, dropout
        )
        self.descriptor_projection = nn.Linear(self.descriptors.shape[1], hidden_size)
        # --input-layer-norm (auto-correction #9): the spatial layers are already
        # pre-norm; the option normalises the Sensor-token embeddings entering them.
        self.input_norm = nn.LayerNorm(hidden_size) if input_layer_norm else nn.Identity()
        self.readout_embedding = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.readout_embedding, std=0.02)
        # Optional global input of the Readout tokens (engine --body-weight-input).
        self.body_weight_column = (
            list(feature_names).index(BODY_WEIGHT_FEATURE) if BODY_WEIGHT_FEATURE in feature_names else None
        )
        # The "body weight missing" flag (engine --body-weight-masking) follows the body weight.
        self.body_weight_columns = (
            [self.body_weight_column]
            + ([list(feature_names).index(BODY_WEIGHT_MISSING_FEATURE)]
               if BODY_WEIGHT_MISSING_FEATURE in feature_names else [])
            if self.body_weight_column is not None
            else []
        )
        self.body_weight_projection = (
            nn.Linear(len(self.body_weight_columns), hidden_size) if self.body_weight_columns else None
        )
        self.layers = nn.ModuleList(
            SpatialEncoderLayer(hidden_size, heads, feedforward_size, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.readout_projection = nn.Linear(hidden_size, hidden_size)
        self.grfm_decoder = nn.Linear(hidden_size, COMPONENTS_PER_FOOT)
        # Optional Foot height output from the Readout tokens (engine --foot-height-output).
        self.foot_height_head = nn.Linear(hidden_size, 1) if foot_height_output else None
        if foot_height_output:
            # Engine statistics (mm, per foot L, R), filled by set_foot_height_statistics().
            self.register_buffer("foot_height_mean", torch.zeros(READOUT_COUNT))
            self.register_buffer("foot_height_std", torch.ones(READOUT_COUNT))

    def set_statistics(self, train_x: list[np.ndarray], x_mean, x_std, y_mean, y_std) -> None:
        """Copy the engine's statistics and fit the per-slot token scaling.

        ``train_x``: raw input arrays ``(T, F)`` of the training trials. Gauge
        loads are scaled with one mean/std over the active gauges of both
        feet, each IMU axis with one mean/std over the 8 IMUs (canonical
        frame), and each canonical GRFM component with statistics pooled over
        the two feet, so that both feet share the same scaling.
        """
        device = self.x_mean.device
        for name, value in (("x_mean", x_mean), ("x_std", x_std), ("y_mean", y_mean), ("y_std", y_std)):
            getattr(self, name).copy_(torch.as_tensor(np.asarray(value), device=device))
        sums = np.zeros((2, len(VALUE_CHANNELS)))
        squares = np.zeros_like(sums)
        counts = np.zeros_like(sums)
        gauge = np.zeros(TOKEN_COUNT, bool)
        gauge[self.tokenizer.gauge_tokens] = True
        for trial in train_x:
            for start in range(0, len(trial), 65536):
                chunk = trial[start : start + 65536]
                inactive = np.zeros(TOKEN_COUNT, bool)
                inactive[self.tokenizer.gauge_tokens] = chunk[0, self.flag_columns] > 0.5
                tokens, mask = self.tokenizer(chunk[:, self.source_columns].astype(np.float64), inactive=inactive)
                for row, selected in ((0, gauge), (1, ~gauge)):
                    values = tokens[:, selected][mask[:, selected]]
                    sums[row] += values.sum(axis=0)
                    squares[row] += np.square(values).sum(axis=0)
                    counts[row] += len(values)
        mean = sums / np.maximum(counts, 1.0)
        std = np.sqrt(np.maximum(squares / np.maximum(counts, 1.0) - mean**2, 0.0))
        std[std < 1e-8] = 1.0
        value_mean = np.zeros((TOKEN_COUNT, len(VALUE_CHANNELS)))
        value_std = np.ones_like(value_mean)
        value_mean[gauge, 0], value_std[gauge, 0] = mean[0, 0], std[0, 0]
        value_mean[~gauge, 1:], value_std[~gauge, 1:] = mean[1, 1:], std[1, 1:]
        self.value_mean.copy_(torch.as_tensor(value_mean, dtype=torch.float32, device=device))
        self.value_std.copy_(torch.as_tensor(value_std, dtype=torch.float32, device=device))
        # Canonical target scaling pooled over feet (right Fy, Mx, Mz mirrored).
        y_mean = np.asarray(y_mean, np.float64).reshape(2, COMPONENTS_PER_FOOT)
        y_std = np.asarray(y_std, np.float64).reshape(2, COMPONENTS_PER_FOOT)
        sign = TARGET_MIRROR_SIGN.reshape(2, COMPONENTS_PER_FOOT)
        self.canonical_mean.copy_(torch.as_tensor((y_mean * sign).mean(axis=0), dtype=torch.float32))
        self.canonical_std.copy_(torch.as_tensor(np.sqrt(np.square(y_std).mean(axis=0)), dtype=torch.float32))

    def set_foot_height_statistics(self, mean, std) -> None:
        """Copy the engine's train Foot height statistics (mm, per foot L, R)."""
        device = self.foot_height_mean.device
        self.foot_height_mean.copy_(torch.as_tensor(np.asarray(mean), dtype=torch.float32, device=device))
        self.foot_height_std.copy_(torch.as_tensor(np.asarray(std), dtype=torch.float32, device=device))

    def readout_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Standardised window ``(B, W, F)`` -> per-foot Readout embeddings ``(B, 2, hidden)``."""
        raw = x * self.x_std + self.x_mean
        inactive = torch.zeros(x.shape[0], TOKEN_COUNT, dtype=torch.bool, device=x.device)
        inactive[:, self.gauge_tokens] = raw[:, -1, self.flag_columns] > 0.5
        tokens, attend = self.tokenizer(raw[..., self.source_columns], inactive=inactive)
        attend = attend[:, -1]                                   # (B, 38), per-trial flags
        if self.training and self.token_dropout > 0.0:
            attend = attend & (torch.rand(attend.shape, device=x.device) >= self.token_dropout)
        values = (tokens - self.value_mean) / self.value_std     # (B, W, 38, 7)
        batch, window = values.shape[:2]
        descriptors = self.descriptors.expand(batch, window, -1, -1)
        temporal_input = torch.cat([values, descriptors], dim=-1).permute(0, 2, 3, 1)
        encoded = self.temporal(temporal_input.reshape(batch * TOKEN_COUNT, -1, window))
        h = self.input_norm(encoded.view(batch, TOKEN_COUNT, -1) + self.descriptor_projection(self.descriptors))
        readout = self.readout_embedding.expand(batch, READOUT_COUNT, -1)
        if self.body_weight_projection is not None:
            readout = readout + self.body_weight_projection(x[:, -1, self.body_weight_columns])[:, None]
        h = torch.cat([h, readout], dim=1)
        attend = torch.cat([attend, attend.new_ones(batch, READOUT_COUNT)], dim=1)
        for layer in self.layers:
            h = layer(h, self.same_foot, attend)
        return self.norm(h[:, TOKEN_COUNT:])

    def _grfm(self, shared: torch.Tensor) -> torch.Tensor:
        canonical = self.grfm_decoder(shared) * self.canonical_std + self.canonical_mean  # (B, 2, 6)
        physical = canonical.reshape(shared.shape[0], -1) * self.mirror_sign
        return (physical - self.y_mean) / self.y_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._grfm(F.gelu(self.readout_projection(self.readout_embeddings(x))))

    def forward_with_foot_height(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardised GRFM ``(B, 12)`` and standardised Foot height ``(B, 2)`` (L, R)."""
        shared = F.gelu(self.readout_projection(self.readout_embeddings(x)))
        # Same function for both feet, on the scale pooled over the feet (mirror consistent).
        pooled_mean = self.foot_height_mean.mean()
        pooled_std = torch.sqrt(torch.square(self.foot_height_std).mean())
        height = self.foot_height_head(shared).squeeze(-1) * pooled_std + pooled_mean
        return self._grfm(shared), (height - self.foot_height_mean) / self.foot_height_std

    def last_shared_layer(self) -> nn.Module:
        """Derniere couche partagee par les 12 cibles (GradNorm), avant le decodeur par pied."""
        return self.readout_projection
