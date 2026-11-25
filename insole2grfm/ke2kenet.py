"""Ke2KeNet (Mahmud 2026, JBHI), ported to PyTorch from the authors' Keras code (#47).

Source: github.com/Sakib1263/Insole-to-3DGRFMCoP, ``models/MLMRSNet.py``, class
``MLMRSNet``, method ``IMUNet``, the network of the paper's configuration
(``configs/insole.json``: depth 5, width 32, kernel 3, cardinality 5, mix pooling,
deep supervision, 1 024-sample segments). The authors' README notes that the class
name is not certified to be the paper's Ke2KeNet.

Parameter count: the public code at the paper's d = 5 and n = 32 has about 57 M
parameters, and the port reproduces it (``tests/test_competitor_mahmud.py``). Its
bottleneck and first decoder level have 1 024 filters (multiplier 2^5), beyond the
paper's eq. 2 (f = 32 x 2^m, m in 0:4). Capping them at 512 still gives about 25 M,
and no depth or width at n = 32 gives the 5.22 M the paper reports. Its "6.19 MB" does
not match 5.22 M float32 parameters either. Neither the code nor eq. 2 reproduces that
figure, so we port the code.

Where the code and the paper differ, the port follows the code:
- upsampling repeats samples (Keras ``UpSampling1D``), where the paper says bilinear;
- mix pooling adds max and average pooling and then a 1x1 convolution, where the paper
  weights them with alpha = 0.5 and has no projection.

The Keras behaviours kept here:
- 'same' padding: pooling outputs ceil(L / s) samples, average pooling excludes the
  padding, and a stride-s transposed convolution outputs L * s samples;
- BatchNormalization with momentum 0.99 and epsilon 1e-3;
- Glorot-uniform kernels and zero biases.

Layout is (batch, channels, time), PyTorch's, against Keras's (batch, time, channels).
Lazy layers infer their input channels on the first forward pass, as Keras does.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def same_pool(x: torch.Tensor, stride: int, kind: str, size: int = 3) -> torch.Tensor:
    """Keras ``MaxPooling1D`` / ``AveragePooling1D`` with ``padding="same"``."""
    length = x.shape[-1]
    out = math.ceil(length / stride)
    pad = max((out - 1) * stride + size - length, 0)
    left, right = pad // 2, pad - pad // 2
    if kind == "max":
        return F.max_pool1d(F.pad(x, (left, right), value=float("-inf")), size, stride)
    ones = F.pad(torch.ones_like(x[:, :1]), (left, right))
    return F.avg_pool1d(F.pad(x, (left, right)), size, stride) / F.avg_pool1d(ones, size, stride)


def upsample(x: torch.Tensor, size: int) -> torch.Tensor:
    """Keras ``UpSampling1D``: each sample repeated ``size`` times."""
    return x.repeat_interleave(size, dim=-1)


class ConvBlock(nn.Module):
    """Conv1D 'same', optional BatchNormalization, ReLU (``Conv_Block``)."""

    def __init__(self, filters: int, kernel: int, batchnorm: bool = True) -> None:
        super().__init__()
        self.conv = nn.LazyConv1d(filters, kernel, padding=kernel // 2)
        self.norm = nn.LazyBatchNorm1d(eps=1e-3, momentum=0.01) if batchnorm else nn.Identity()

    def forward(self, x):
        return torch.relu(self.norm(self.conv(x)))


class TransConv(nn.Module):
    """Keras ``Conv1DTranspose(filters, 4, strides=s, padding="same")``, BatchNormalization, ReLU."""

    def __init__(self, filters: int, stride: int, kernel: int = 4) -> None:
        super().__init__()
        self.stride, self.kernel = stride, kernel
        self.conv = nn.LazyConvTranspose1d(filters, kernel, stride=stride, bias=False)
        self.bias = nn.Parameter(torch.zeros(filters))
        self.norm = nn.LazyBatchNorm1d(eps=1e-3, momentum=0.01)

    def forward(self, x):
        target = x.shape[-1] * self.stride
        y = self.conv(x)
        if self.kernel >= self.stride:
            start = (self.kernel - self.stride) // 2
            y = y[..., start:start + target]
        else:
            y = F.pad(y, (0, target - y.shape[-1]))
        return torch.relu(self.norm(y + self.bias[:, None]))


class MixPool(nn.Module):
    """Max + average pooling, then a 1x1 projection keeping the channels (``mix_pool_layer``)."""

    def __init__(self, stride: int) -> None:
        super().__init__()
        self.stride = stride
        # Keeps the input channels, unknown until the first call (``build`` makes that call).
        self.proj: nn.Conv1d | None = None

    def forward(self, x):
        if self.proj is None:
            self.proj = nn.Conv1d(x.shape[1], x.shape[1], 1).to(x.device)
        return self.proj(same_pool(x, self.stride, "max") + same_pool(x, self.stride, "avg"))


class MSPUnit(nn.Module):
    """Pool one temporal scope, restore its length by two paths, fuse (``MSP_Unit``, mix pooling)."""

    def __init__(self, level: int, filters: int, multiplier: int) -> None:
        super().__init__()
        self.level = level
        self.pool = MixPool(level)
        self.conv = ConvBlock(filters * multiplier, 1)
        self.trans = TransConv(filters * multiplier, level)
        self.fuse = ConvBlock(filters, 1, batchnorm=False)

    def forward(self, x):
        x = self.conv(self.pool(x))
        return self.fuse(torch.cat([self.trans(x), upsample(x, self.level)], dim=1))


class MRPBlock(nn.Module):
    """The input and its pooled scopes, parallel 3/5/7 convolutions, a 1x1 fusion (``MRP_Block``)."""

    def __init__(self, filters: int, multiplier: int, cardinality: int) -> None:
        super().__init__()
        width = filters * multiplier
        self.scopes = nn.ModuleList(MSPUnit(2**ii, filters, multiplier) for ii in range(cardinality))
        self.branches = nn.ModuleList(ConvBlock(width, kernel, batchnorm=False) for kernel in (3, 5, 7))
        self.out = ConvBlock(width, 1)

    def forward(self, x):
        if self.scopes:
            x = torch.cat([x, *(scope(x) for scope in self.scopes)], dim=1)
        return self.out(torch.cat([branch(x) for branch in self.branches], dim=1))


class Ke2KeNet(nn.Module):
    """``MLMRSNet.IMUNet``: multiscale encoder, full-scale decoder aggregation, deep supervision.

    ``forward`` returns the heads from coarsest to finest, with the full-resolution
    output ``out`` last. Keras orders the auxiliary heads the other way; the loss
    matches each head to the target by its length, so the order does not matter.
    """

    def __init__(self, length: int = 1024, depth: int = 5, width: int = 32, kernel: int = 3,
                 outputs: int = 1, deep_supervision: bool = True) -> None:
        super().__init__()
        if length % 2**depth:
            raise ValueError("Signal length must be divisible by 2**depth.")
        self.depth, self.deep_supervision = depth, deep_supervision
        self.encoder = nn.ModuleList(MRPBlock(width, 2 ** (i - 1), depth - i + 1) for i in range(1, depth + 1))
        self.bottleneck = MRPBlock(width, 2**depth, 0)
        self.skips = nn.ModuleList(
            nn.ModuleList(ConvBlock(width, kernel) for _ in range(depth - j - 1)) for j in range(depth)
        )
        self.pre_up = nn.ModuleList(MRPBlock(width, 1, 0) for _ in range(depth))
        self.decoder = nn.ModuleList(MRPBlock(width, 2 ** (depth - j), 0) for j in range(depth))
        # Deep-supervision heads: the bottleneck, then each decoder level but the last.
        self.levels = nn.ModuleList(nn.LazyConv1d(1, 1) for _ in range(depth if deep_supervision else 0))
        self.out = nn.LazyConv1d(outputs, 1)

    def forward(self, x):
        convs, pool = [], x
        for block in self.encoder:
            conv = block(pool)
            convs.append(conv)
            pool = F.max_pool1d(conv, 2)
        deconv = self.bottleneck(pool)
        levels = [self.levels[0](deconv)] if self.deep_supervision else []
        deconvs = []
        for j in range(self.depth):
            skip_all = [convs[self.depth - j - 1]]
            for k, block in enumerate(self.skips[j]):
                skip_all.append(block(F.max_pool1d(convs[k], 2 ** (self.depth - j - k - 1))))
            parts = [*skip_all, torch.sigmoid(upsample(self.pre_up[j](deconv), 2))]
            parts += [torch.sigmoid(upsample(deconvs[m], 2 ** (j - m))) for m in range(j)]
            deconv = self.decoder[j](torch.cat(parts, dim=1))
            deconvs.append(deconv)
            if self.deep_supervision and j < self.depth - 1:
                levels.append(self.levels[j + 1](deconv))
        return [*levels, self.out(deconv)]


def build(channels: int, length: int = 1024, outputs: int = 1, **kwargs) -> Ke2KeNet:
    """A materialised Ke2KeNet for ``channels`` inputs, with Keras's Glorot-uniform kernels and zero biases."""
    model = Ke2KeNet(length, outputs=outputs, **kwargs)
    model.eval()
    with torch.no_grad():
        model(torch.zeros(2, channels, length))
    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    model.train()
    return model


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
