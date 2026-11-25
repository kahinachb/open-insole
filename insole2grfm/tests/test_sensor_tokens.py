"""Behaviour of the sensor-token representation in the canonical insole frame."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sensor_tokens import SensorTokenizer

LAYOUT = Path(__file__).resolve().parents[1] / "tools" / "insole_layout.json"


@pytest.fixture(scope="module")
def tokenizer() -> SensorTokenizer:
    return SensorTokenizer.from_layout(LAYOUT)


@pytest.fixture(scope="module")
def names() -> list[str]:
    return json.loads(LAYOUT.read_text(encoding="utf-8"))["feature_names"]


def frame(names, count=1, **values) -> np.ndarray:
    x = np.zeros((count, len(names)), dtype=np.float32)
    for name, value in values.items():
        x[:, names.index(name)] = value
    return x


def test_window_batch_gives_38_tokens_per_instant(tokenizer, names):
    x = np.random.default_rng(0).normal(size=(5, 21, len(names))).astype(np.float32)
    tokens, mask = tokenizer(x)
    assert tokens.shape == (5, 21, 38, 7)
    assert mask.shape == (5, 21, 38) and mask.dtype == bool and mask.all()
    assert tokenizer.descriptors.shape[0] == 38
    assert len(tokenizer.token_names) == 38


# KiCad footprint rotation of each patch IMU (deg) and the chip -> physical insole
# axes it implies, verified on walking data (insole angle = KiCad rot + 180 deg).
KICAD_ROT = {
    "L_p1": -90, "L_p3": 180, "L_p5": -90, "L_p7": 0,
    "R_p9": -90, "R_p11": 180, "R_p13": -90, "R_p15": -90,  # R forefoot: from the left board
}
CHIP_FOR_INSOLE = {  # rot: insole in-plane (ix, iy) -> chip in-plane (cx, cy)
    -90: lambda ix, iy: (iy, -ix),   # x_ins = -chip_y, y_ins = +chip_x
    180: lambda ix, iy: (ix, iy),    # identity
    0: lambda ix, iy: (-ix, -iy),    # x_ins = -chip_x, y_ins = -chip_y
}
IMU_TOKENS = dict(zip(KICAD_ROT, (15, 16, 17, 18, 34, 35, 36, 37)))


def chip_values(physical: dict) -> dict:
    """Raw channels whose IMUs read ``physical[prefix] = (acc, gyro)`` in the physical insole frame."""
    values = {}
    for prefix, (acc, gyro) in physical.items():
        to_chip = CHIP_FOR_INSOLE[KICAD_ROT[prefix]]
        for kind, vector, unit in (("acc", acc, "g"), ("gyro", gyro, "dps")):
            cx, cy = to_chip(vector[0], vector[1])
            for axis, value in zip("xyz", (cx, cy, vector[2])):
                values[f"{prefix}_{kind}_{axis}_cal_{unit}"] = value
    return values


@pytest.mark.parametrize("pitch_deg", [0.0, 20.0])
def test_quiet_standing_gravity_maps_to_up_on_every_imu(tokenizer, names, pitch_deg):
    # heel raised by pitch_deg: the accelerometer feels +1 g along the tilted vertical
    up = (-np.sin(np.radians(pitch_deg)), 0.0, np.cos(np.radians(pitch_deg)))
    x = frame(names, **chip_values({prefix: (up, (0.0, 0.0, 0.0)) for prefix in KICAD_ROT}))
    tokens, _ = tokenizer(x)
    for prefix, index in IMU_TOKENS.items():
        np.testing.assert_allclose(tokens[0, index, 1:4], up, atol=1e-6, err_msg=prefix)


LEFT_TO_RIGHT_PATCH = {"p1": "p9", "p3": "p11", "p5": "p13", "p7": "p15"}


def mirror_physical(acc, gyro):
    """Mirror image of a motion through the sagittal plane (y -> -y)."""
    (ax, ay, az), (gx, gy, gz) = acc, gyro
    return (ax, -ay, az), (-gx, gy, -gz)


def test_right_foot_mirrored_from_left_foot_gives_the_same_tokens(tokenizer, names):
    rng = np.random.default_rng(1)
    values = {}
    physical = {}
    for name in names:
        if name.startswith("L_") and "_force_" in name:
            load = float(rng.uniform(0, 300))
            values[name] = load
            patch = name.split("_")[1]
            values[name.replace(f"L_{patch}", f"R_{LEFT_TO_RIGHT_PATCH[patch]}")] = load
    for left, right in LEFT_TO_RIGHT_PATCH.items():
        acc, gyro = tuple(rng.normal(size=3)), tuple(rng.normal(scale=100, size=3))
        physical[f"L_{left}"] = (acc, gyro)
        physical[f"R_{right}"] = mirror_physical(acc, gyro)
    tokens, mask = tokenizer(frame(names, **values, **chip_values(physical)))
    np.testing.assert_allclose(tokens[0, 19:], tokens[0, :19], atol=1e-4)
    assert np.abs(tokens[0, :19]).sum() > 0
    # gait flexion (rotation about the physical +y axis) is canonical +y on both feet
    flexion = {prefix: ((0, 0, 1), (0, 50.0, 0)) for prefix in KICAD_ROT}
    tokens, _ = tokenizer(frame(names, **chip_values(flexion)))
    for prefix, index in IMU_TOKENS.items():
        np.testing.assert_allclose(tokens[0, index, 4:], (0, 50.0, 0), atol=1e-5, err_msg=prefix)


def test_inactive_gauge_is_masked_not_zero_filled(tokenizer, names):
    # a trial where the midfoot postero-medial gauge stays below 5 N (p99) and reads noise
    rng = np.random.default_rng(2)
    trial = frame(names, count=400, L_p1_force_1_N=200.0, R_p9_force_1_N=150.0)
    dead = names.index("L_p3_force_4_N")
    trial[:, dead] = rng.uniform(-2.0, 3.0, size=len(trial))
    trial[:3, dead] = 40.0  # a few spikes do not make the gauge active (p99)
    inactive = tokenizer.inactive(trial)
    assert inactive.shape == (38,) and inactive.dtype == bool
    dead_token = tokenizer.token_names.index("L_p3_force_4_N")
    # gauges reading zero all trial are inactive too; IMU tokens never are
    assert inactive[dead_token]
    assert not inactive[tokenizer.token_names.index("L_p1_force_1_N")]
    assert not inactive[[15, 16, 17, 18, 34, 35, 36, 37]].any()

    windows = np.stack([trial[:21], trial[100:121]])  # (2, 21, 78)
    tokens, mask = tokenizer(windows, inactive=np.stack([inactive, np.zeros(38, bool)]))
    assert mask.shape == (2, 21, 38)
    assert not mask[0, :, dead_token].any() and mask[1, :, dead_token].all()
    assert (mask[0] == ~inactive).all()
    # the reading is kept as is (the mask, not a zero, removes it)
    np.testing.assert_allclose(tokens[0, :, dead_token, 0], trial[:21, dead])


def test_static_descriptors_describe_each_token_in_the_canonical_frame(tokenizer):
    index = tokenizer.token_names.index
    # sensor centres from the KiCad boards; right foot mirrored so lateral is +y
    np.testing.assert_allclose(tokenizer.positions_mm[index("L_p1_force_1_N")], (72.76, 11.31))
    np.testing.assert_allclose(tokenizer.positions_mm[index("R_p9_force_1_N")], (74.31, 8.67))
    np.testing.assert_allclose(tokenizer.positions_mm[index("L_p1_imu")], (56.76, -0.32))
    # the right forefoot IMU is missing from its board: mirrored left one
    np.testing.assert_allclose(tokenizer.positions_mm[index("R_p13_imu")], (198.63, 7.27))
    # cleat tilt vector = direction x sin(tilt); C1 tilts 6 deg; IMUs have none
    sin6 = np.sin(np.radians(6.0))
    np.testing.assert_allclose(tokenizer.tilt_vectors[index("L_p1_force_1_N")], (-sin6, 0.0), atol=1e-4)
    np.testing.assert_allclose(tokenizer.tilt_vectors[index("L_p3_force_1_N")], (-0.966 * sin6, -0.259 * sin6), atol=1e-3)
    np.testing.assert_allclose(tokenizer.tilt_vectors[index("R_p11_force_1_N")], (-0.966 * sin6, -0.259 * sin6), atol=1e-3)
    assert (tokenizer.tilt_vectors[tokenizer.imu_tokens] == 0).all()
    assert list(tokenizer.token_types) == ([0] * 15 + [1] * 4) * 2   # 0 gauge, 1 IMU
    assert list(tokenizer.token_feet) == [0] * 19 + [1] * 19         # 0 left, 1 right

    d = tokenizer.descriptors
    names = tokenizer.descriptor_names
    assert d.shape == (38, len(names)) and d.dtype == np.float32
    # Fourier features of x / y (mm): 20 mm shortest wavelength
    assert d[index("L_p1_force_1_N"), names.index("sin_x_20mm")] == pytest.approx(np.sin(2 * np.pi * 72.76 / 20), abs=1e-5)
    assert d[index("L_p1_force_1_N"), names.index("cos_y_20mm")] == pytest.approx(np.cos(2 * np.pi * 11.31 / 20), abs=1e-5)
    for name in ("tilt_x", "tilt_y", "is_gauge", "is_imu", "is_left", "is_right"):
        assert name in names
    assert d[index("R_p15_imu"), names.index("is_imu")] == 1 and d[index("R_p15_imu"), names.index("is_right")] == 1
    assert d[index("L_p7_force_2_N"), names.index("is_gauge")] == 1 and d[index("L_p7_force_2_N"), names.index("is_left")] == 1
    assert len(np.unique(d, axis=0)) == 38  # no two tokens share a descriptor


def test_right_foot_targets_flip_fy_mx_mz_into_the_canonical_frame_and_back():
    from sensor_tokens import targets_from_canonical, targets_to_canonical

    # Fx_L Fy_L Fz_L Mx_L My_L Mz_L Fx_R Fy_R Fz_R Mx_R My_R Mz_R
    y = np.arange(1.0, 13.0)[None].repeat(4, axis=0)
    canonical = targets_to_canonical(y)
    np.testing.assert_array_equal(canonical[0], [1, 2, 3, 4, 5, 6, 7, -8, 9, -10, 11, -12])
    np.testing.assert_array_equal(targets_from_canonical(canonical), y)


def test_torch_batches_give_the_same_tokens_as_numpy(tokenizer, names):
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(3).normal(size=(2, 21, len(names))).astype(np.float32)
    inactive = np.zeros((2, 38), bool)
    inactive[1, 5] = True
    tokens_np, mask_np = tokenizer(x, inactive=inactive)
    x_t = torch.from_numpy(x).requires_grad_(True)
    tokens_t, mask_t = tokenizer(x_t, inactive=torch.from_numpy(inactive))
    assert isinstance(tokens_t, torch.Tensor) and mask_t.dtype == torch.bool
    np.testing.assert_allclose(tokens_t.detach().numpy(), tokens_np, atol=1e-5)
    np.testing.assert_array_equal(mask_t.numpy(), mask_np)
    tokens_t.sum().backward()
    assert x_t.grad is not None
    from sensor_tokens import targets_to_canonical

    y = torch.arange(1.0, 13.0)
    assert targets_to_canonical(y)[7].item() == -8.0


def test_swap_feet_exchanges_the_feet_tokens(tokenizer, names):
    x = np.random.default_rng(1).normal(size=(4, len(names)))
    tokens, _ = tokenizer(x)
    swapped, _ = tokenizer(tokenizer.swap_feet(x))
    np.testing.assert_allclose(swapped[:, :19], tokens[:, 19:], atol=1e-9)
    np.testing.assert_allclose(swapped[:, 19:], tokens[:, :19], atol=1e-9)


def test_mirror_symmetric_tokenizer_shares_geometry_between_counterparts(tokenizer):
    symmetric = tokenizer.mirror_symmetric()
    np.testing.assert_array_equal(symmetric.descriptors[:19, :-2], symmetric.descriptors[19:, :-2])
    assert np.abs(symmetric.positions_mm[:19] - tokenizer.positions_mm[:19]).max() < 10.0
