"""Insole samples from the receiver's serial port, from a raw capture, or from a prepared trial.

A *sample* is one row of the model's 78 input channels (forces in N after the
Tare, IMU calibrated), with the left and right packets paired. Every source
yields ``(time_s, x)`` with ``x`` keyed by the channel names of the dataset
(``L_p1_force_1_N`` ... ``R_p15_gyro_z_cal_dps``).

A demo capture (``.bin``) starts with ``CAPTURE_MAGIC`` and stores the serial
chunks as they arrived, each as ``<d I`` (seconds since the start, length)
followed by the bytes, so a session replays exactly as it ran. An acquisition
of ``record_trial.py`` (``dataSemelle.bin``, the bare byte stream) replays too,
paced at ``PACKETS_PER_S``.
"""

from __future__ import annotations

import json
import struct
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "acquisition"))

from insole_data import decodageToCSV as codec  # noqa: E402
from live_plot_forces import LivePacketDecoder  # noqa: E402

CALIBRATION = REPO / "calibration_imu" / "data"
IMU_CALIBRATION = {
    "acc_left": CALIBRATION / "session_20260728_155627_gauche/calibration_acc/accelerometer_calibration.json",
    "gyro_left": CALIBRATION / "session_gyro_20260803_124950_gauche/calibration_gyro_std015_cv/gyroscope_calibration.json",
    "acc_right": CALIBRATION / "session_20260914_181836_droit/calibration_acc/accelerometer_calibration.json",
    "gyro_right": CALIBRATION / "session_gyro_20260915_113614_droit/calibration_gyro/gyroscope_calibration.json",
}
CHUNK_HEADER = struct.Struct("<dI")
CAPTURE_MAGIC = b"OPEN-INSOLE-CAPTURE-1\n"
PACKETS_PER_S = 400.0  # 200 Hz per foot
# A foot whose packets stop is reported after this many unpaired packets of the other foot.
MAX_UNPAIRED = 8


def load_tares(paths: list[Path]) -> None:
    """Use the Tares of ``paths`` (``fx29_tare_v1``, one or both feet each) in the force conversion."""
    tares = {}
    for path in paths:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("format") != "fx29_tare_v1":
            raise ValueError(f"{path}: not a fx29_tare_v1 Tare file")
        tares.update(document["tares"])
    missing = {"1", "3", "5", "7", "9", "11", "13", "15"} - set(tares)
    if missing:
        raise ValueError(f"no Tare for Patches {sorted(missing, key=int)}: give the Tare of both feet")
    codec.taresFX29.clear()
    codec.taresFX29.update(tares)


def foot_channels(packet: bytes, signal: int, calibrations: dict) -> tuple[str, dict[str, float]]:
    """The foot (``L``/``R``) of one packet and its 39 channels, named as in the dataset."""
    sequence = codec.SEQUENCES_PATCHS[tuple(packet[:2])]
    side = "L" if sequence[0][0] == 1 else "R"
    timestamp = packet[144:148]
    channels: dict[str, float] = {}
    for index, (identifier, _) in enumerate(sequence):
        start = index * codec.TAILLE_TRAME
        frame = codec.decode(packet[start:start + codec.TAILLE_TRAME], signal, timestamp)
        frame = codec.apply_imu_calibration(frame, calibrations)
        prefix = f"{side}_p{identifier}"
        for number in range(1, 5):
            if (identifier, number) not in codec.CAPTEURS_ABSENTS:
                channels[f"{prefix}_force_{number}_N"] = frame[3 + number]
        for axis, value in zip("xyz", frame[21:24]):
            channels[f"{prefix}_acc_{axis}_cal_g"] = value
        for axis, value in zip("xyz", frame[24:27]):
            channels[f"{prefix}_gyro_{axis}_cal_dps"] = value
    return side, channels


class Pairer:
    """Pairs left and right packets in arrival order into samples."""

    def __init__(self) -> None:
        self.waiting = {"L": deque(), "R": deque()}
        self.dropped = 0

    def push(self, side: str, received_s: float, channels: dict) -> list[tuple[float, dict]]:
        self.waiting[side].append((received_s, channels))
        samples = []
        while self.waiting["L"] and self.waiting["R"]:
            (t_left, left), (t_right, right) = self.waiting["L"].popleft(), self.waiting["R"].popleft()
            samples.append((max(t_left, t_right), {**left, **right}))
        for queue in self.waiting.values():
            while len(queue) > MAX_UNPAIRED:
                queue.popleft()
                self.dropped += 1
        return samples


class SerialSource:
    """Samples of the live receiver; the raw chunks are also written to ``capture`` when given."""

    def __init__(self, port: str, baud: int = 2_000_000, capture: Path | None = None):
        self.port, self.baud, self.capture = port, baud, capture

    def chunks(self) -> Iterator[tuple[float, bytes]]:
        import serial

        start = time.monotonic()
        writer = open(self.capture, "wb") if self.capture else None
        if writer:
            writer.write(CAPTURE_MAGIC)
        try:
            with serial.Serial(self.port, self.baud, timeout=0.02) as port:
                while True:
                    chunk = port.read(max(1, min(port.in_waiting, 65_536)))
                    if not chunk:
                        continue
                    now = time.monotonic() - start
                    if writer:
                        writer.write(CHUNK_HEADER.pack(now, len(chunk)) + chunk)
                    yield now, chunk
        finally:
            if writer:
                writer.close()


class CaptureSource:
    """The chunks of a capture, at their recorded pace (``speed`` 0: as fast as possible)."""

    def __init__(self, path: Path, speed: float = 1.0):
        self.path, self.speed = Path(path), speed

    def recorded(self, data: bytes) -> Iterator[tuple[float, bytes]]:
        offset = len(CAPTURE_MAGIC)
        while offset + CHUNK_HEADER.size <= len(data):
            at, length = CHUNK_HEADER.unpack_from(data, offset)
            offset += CHUNK_HEADER.size
            yield at, data[offset:offset + length]
            offset += length

    @staticmethod
    def bare(data: bytes) -> Iterator[tuple[float, bytes]]:
        size = 4 * codec.TAILLE_PAQUET
        for offset in range(0, len(data), size):
            yield offset / codec.TAILLE_PAQUET / PACKETS_PER_S, data[offset:offset + size]

    def chunks(self) -> Iterator[tuple[float, bytes]]:
        data = self.path.read_bytes()
        chunks = self.recorded(data) if data.startswith(CAPTURE_MAGIC) else self.bare(data)
        start = time.monotonic()
        for at, chunk in chunks:
            if self.speed > 0:
                delay = at / self.speed - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)
            yield at, chunk


def packet_samples(source, calibrations: dict, status: dict | None = None) -> Iterator[tuple[float, dict]]:
    """Paired samples of a chunk source (serial or capture)."""
    decoder, pairer = LivePacketDecoder(), Pairer()
    for at, chunk in source.chunks():
        for signal, packet in decoder.feed(chunk):
            side, channels = foot_channels(packet, signal, calibrations)
            if status is not None:
                status[side] = at
            yield from pairer.push(side, at, channels)
        if status is not None:
            status["dropped"] = pairer.dropped
            status["rejected_bytes"] = decoder.rejected_bytes


def trial_samples(
    npz: Path, channels: list[str], speed: float = 1.0, rate_hz: float = 200.0
) -> Iterator[tuple[float, dict, np.ndarray]]:
    """Rows of a prepared dataset trial (columns ``channels``), with its plate GRFM (NaN where not Measured)."""
    trial = np.load(npz, allow_pickle=True)
    x, y, mask = trial["X"], trial["y"], trial["target_mask"].astype(bool)
    truth = np.where(mask, y, np.nan)
    start = time.monotonic()
    for row in range(len(x)):
        at = row / rate_hz
        if speed > 0:
            delay = at / speed - (time.monotonic() - start)
            if delay > 0:
                time.sleep(delay)
        yield at, dict(zip(channels, x[row].tolist())), truth[row]


def imu_calibrations() -> dict:
    paths = IMU_CALIBRATION
    return codec.load_all_imu_calibrations(paths["acc_left"], paths["gyro_left"], paths["acc_right"], paths["gyro_right"])


def raw_counts(source, seconds: float) -> dict[str, list[list[int]]]:
    """Raw load-cell counts per Patch ID over the first ``seconds`` of a chunk source."""
    decoder = LivePacketDecoder()
    counts: dict[str, list[list[int]]] = {}
    for at, chunk in source.chunks():
        if at > seconds:
            break
        for _signal, packet in decoder.feed(chunk):
            for index in range(4):
                start = index * codec.TAILLE_TRAME
                identifier = packet[start]
                values = struct.unpack(">HHHH", packet[start + 3:start + 11])
                patch = counts.setdefault(str(identifier), [[], [], [], []])
                for number, value in enumerate(values):
                    patch[number].append(value)
    return counts


def threaded(iterator: Iterator, maxsize: int = 4096):
    """Run ``iterator`` in a thread; returns a queue fed with its items, then ``None``."""
    import queue

    items: queue.Queue = queue.Queue(maxsize)

    def worker():
        try:
            for item in iterator:
                items.put(item)
        except Exception as exc:  # reported by the consumer
            items.put(exc)
        items.put(None)

    threading.Thread(target=worker, daemon=True).start()
    return items
