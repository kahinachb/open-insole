#!/usr/bin/env python3
"""Plot the Gauge forces of both insoles live, from the receiver's serial port.

A quick test of an insole: plug the receiver into the PC, switch the insole on,
then press each zone of the insole; the matching Gauges should rise.

    python acquisition/live_plot_forces.py --port /dev/ttyACM0
    python acquisition/live_plot_forces.py --port /dev/ttyACM0 --view total

Without --tare-file, every Gauge uses a default Tare of 1000 counts, so the
forces are only approximate.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import serial

from insole_data.decodageToCSV import (
    SEQUENCES_PATCHS,
    TAILLE_PAQUET,
    TAILLE_TRAME,
    decode,
    packet_is_valid,
    taresFX29,
)


PATCHES = {
    "left": ((1, 2), (3, 4), (5, 6), (7, 8)),
    "right": ((9, 10), (11, 12), (13, 14), (15, 16)),
}


class LivePacketDecoder:
    """Resynchronise the serial stream and return the complete packets."""

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.signal = 0
        self.rejected_bytes = 0

    def feed(self, chunk: bytes) -> list[tuple[int, bytes]]:
        self.buffer.extend(chunk)
        packets: list[tuple[int, bytes]] = []

        while len(self.buffer) >= TAILLE_PAQUET:
            if packet_is_valid(self.buffer, 0):
                packet = bytes(self.buffer[:TAILLE_PAQUET])
                del self.buffer[:TAILLE_PAQUET]
                packets.append((self.signal, packet))
            elif self.buffer[0] in (0, 1) and len(self.buffer) < TAILLE_PAQUET + 1:
                # The last byte of a packet preceded by its Signal byte may still be missing.
                break
            elif (
                self.buffer[0] in (0, 1)
                and packet_is_valid(self.buffer, 1)
            ):
                self.signal = self.buffer[0]
                packet = bytes(self.buffer[1 : 1 + TAILLE_PAQUET])
                del self.buffer[: TAILLE_PAQUET + 1]
                packets.append((self.signal, packet))
            else:
                del self.buffer[0]
                self.rejected_bytes += 1

        return packets


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the Gauge forces of the insoles in near real time."
    )
    parser.add_argument("--port", required=True, help="serial port of the receiver")
    parser.add_argument("--baud", type=int, default=2_000_000)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--plot-hz", type=float, default=20.0)
    parser.add_argument(
        "--view",
        choices=("patches", "total"),
        default="patches",
        help="forces per Patch, or total force of each insole",
    )
    parser.add_argument("--tare-file", help="JSON written by insole_data/calculerTareFX29.py")
    return parser


def load_tares(path: str | None) -> None:
    if path is None:
        return
    with open(path, encoding="utf-8") as tare_file:
        document = json.load(tare_file)
    if document.get("format") != "fx29_tare_v1":
        raise ValueError("unknown Tare file format")
    taresFX29.clear()
    taresFX29.update(document["tares"])


def autoscale_y(axis: plt.Axes, values: list[float]) -> None:
    if not values:
        return
    low = min(0.0, min(values))
    high = max(values)
    padding = max(1.0, (high - low) * 0.1)
    axis.set_ylim(low - padding, high + padding)


def serial_worker(
    port: str,
    baud: int,
    stop_event: threading.Event,
    samples: queue.Queue,
    status: queue.Queue,
    decoder: LivePacketDecoder,
) -> None:
    """Read and decode the serial port without ever blocking the GUI."""
    try:
        with serial.Serial(port, baud, timeout=0.05) as serial_port:
            status.put(("ready", None))
            while not stop_event.is_set():
                chunk = serial_port.read(max(1, min(serial_port.in_waiting, 65_536)))
                if not chunk:
                    continue
                for signal, packet in decoder.feed(chunk):
                    received_at = time.monotonic()
                    sequence = SEQUENCES_PATCHS[tuple(packet[:2])]
                    timestamp = packet[144:148]
                    patch_samples = []
                    for patch_index, identifier in enumerate(sequence):
                        start = patch_index * TAILLE_TRAME
                        decoded = decode(
                            packet[start : start + TAILLE_TRAME], signal, timestamp
                        )
                        patch_samples.append((identifier, tuple(decoded[4:8])))
                    foot = "left" if sequence[0][0] == 1 else "right"
                    foot_total = sum(
                        sum(forces) for _identifier, forces in patch_samples
                    )
                    try:
                        samples.put_nowait(
                            (received_at, foot, foot_total, tuple(patch_samples))
                        )
                    except queue.Full:
                        # Keep acquiring; the GUI reports the loss.
                        try:
                            status.put_nowait(("dropped", 1))
                        except queue.Full:
                            pass
    except Exception as exc:
        status.put(("error", exc))


def main() -> int:
    args = build_argparser().parse_args()
    if args.baud <= 0 or args.window_seconds <= 0 or args.plot_hz <= 0:
        print("--baud, --window-seconds and --plot-hz must be > 0", file=sys.stderr)
        return 2

    try:
        load_tares(args.tare_file)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Tare error: {exc}", file=sys.stderr)
        return 2

    histories = {
        identifier: {
            "times": deque(),
            "forces": tuple(deque() for _ in range(4)),
        }
        for identifiers in PATCHES.values()
        for identifier in identifiers
    }
    foot_histories = {
        foot: {"times": deque(), "totals": deque()} for foot in PATCHES
    }

    patch_lines: dict[tuple[int, int], list[plt.Line2D]] = {}
    foot_lines: dict[str, plt.Line2D] = {}
    if args.view == "patches":
        figure, axes = plt.subplots(4, 2, figsize=(14, 10), sharex=True)
    else:
        figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    figure.suptitle("Insole Gauge forces, live")

    if args.view == "patches":
        for column, (foot, identifiers) in enumerate(PATCHES.items()):
            for row, identifier in enumerate(identifiers):
                axis = axes[row][column]
                patch_lines[identifier] = [
                    axis.plot([], [], linewidth=0.9, alpha=0.7, label=f"F{i}")[0]
                    for i in range(1, 5)
                ]
                patch_lines[identifier].append(
                    axis.plot([], [], color="black", linewidth=2.0, label="Total")[0]
                )
                axis.set_title(
                    f"{foot.capitalize()}, Patch {identifier[0]}/{identifier[1]}"
                )
                axis.set_ylabel("Force (N)")
                axis.set_xlim(-args.window_seconds, 0.0)
                axis.grid(True, alpha=0.3)
                axis.legend(loc="upper left", ncol=5, fontsize=8)
        axes[-1][0].set_xlabel("Time (s, latest sample at 0)")
        axes[-1][1].set_xlabel("Time (s, latest sample at 0)")
    else:
        for axis, foot in zip(axes, PATCHES):
            foot_lines[foot] = axis.plot(
                [], [], color="black", linewidth=2.0, label="Total force"
            )[0]
            axis.set_title(f"{foot.capitalize()} insole")
            axis.set_ylabel("Force (N)")
            axis.set_xlim(-args.window_seconds, 0.0)
            axis.grid(True, alpha=0.3)
            axis.legend(loc="upper left")
        axes[-1].set_xlabel("Time (s, latest sample at 0)")
    plt.tight_layout()
    plt.show(block=False)

    decoder = LivePacketDecoder()
    sample_queue: queue.Queue = queue.Queue(maxsize=16_384)
    status_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    reader = threading.Thread(
        target=serial_worker,
        name="insole-serial-reader",
        daemon=True,
        args=(args.port, args.baud, stop_event, sample_queue, status_queue, decoder),
    )
    redraw_period = 1.0 / args.plot_hz
    last_redraw = 0.0
    last_autoscale = 0.0
    packet_count = 0
    dropped_samples = 0

    try:
        reader.start()
        while plt.fignum_exists(figure.number):
            while True:
                try:
                    kind, value = status_queue.get_nowait()
                except queue.Empty:
                    break
                if kind == "error":
                    raise value
                if kind == "dropped":
                    dropped_samples += int(value)

            while True:
                try:
                    received_at, foot, foot_total, patch_samples = sample_queue.get_nowait()
                except queue.Empty:
                    break
                foot_histories[foot]["times"].append(received_at)
                foot_histories[foot]["totals"].append(foot_total)
                for identifier, forces in patch_samples:
                    history = histories[identifier]
                    history["times"].append(received_at)
                    for index, force in enumerate(forces):
                        history["forces"][index].append(force)
                packet_count += 1

            now = time.monotonic()
            cutoff = now - args.window_seconds
            for history in histories.values():
                while history["times"] and history["times"][0] < cutoff:
                    history["times"].popleft()
                    for force_values in history["forces"]:
                        force_values.popleft()
            for history in foot_histories.values():
                while history["times"] and history["times"][0] < cutoff:
                    history["times"].popleft()
                    history["totals"].popleft()

            if now - last_redraw >= redraw_period:
                if args.view == "patches":
                    for identifier, history in histories.items():
                        x = [sample_time - now for sample_time in history["times"]]
                        all_values: list[float] = []
                        force_lists = [list(values) for values in history["forces"]]
                        for line, values in zip(
                            patch_lines[identifier][:4], force_lists
                        ):
                            line.set_data(x, values)
                            all_values.extend(values)
                        totals = [sum(values) for values in zip(*force_lists)]
                        patch_lines[identifier][4].set_data(x, totals)
                        all_values.extend(totals)
                        if now - last_autoscale >= 1.0:
                            autoscale_y(patch_lines[identifier][0].axes, all_values)
                else:
                    for foot, history in foot_histories.items():
                        x = [sample_time - now for sample_time in history["times"]]
                        totals = list(history["totals"])
                        foot_lines[foot].set_data(x, totals)
                        if now - last_autoscale >= 1.0:
                            autoscale_y(foot_lines[foot].axes, totals)
                figure.suptitle(
                    "Insole Gauge forces, live | "
                    f"packets={packet_count} | "
                    f"rejected bytes={decoder.rejected_bytes} | "
                    f"samples dropped by the GUI={dropped_samples}"
                )
                figure.canvas.draw_idle()
                last_redraw = now
                if now - last_autoscale >= 1.0:
                    last_autoscale = now

            plt.pause(0.001)

        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, serial.SerialException) as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Acquisition error: {exc}", file=sys.stderr)
        return 2
    finally:
        stop_event.set()
        if reader.is_alive():
            reader.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
