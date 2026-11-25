"""Record one trial: both insoles and two cameras, started by the receiver's switch.

Writes to OUTPUT_DIR the files that the acquisition pipeline reads:

    dataSemelle.bin                 raw serial stream of the receiver
    trigger_log.csv                 PC time of every switch transition
    cam1_manip.mp4, cam2_manip.mp4  videos, recorded only while the switch is on
    cam{1,2}_manip_timestamps.csv   PC time of every video frame

The insoles are recorded continuously; the switch on D7 of the receiver
(hardware/firmware/receiver) prefixes each packet with Signal = 0 or 1. Its
transitions, stamped with the PC clock, are the common anchor that aligns the
videos with the insoles. Press Enter to stop; a diagnostic of packet loss and
frame rate follows.

    python acquisition/record_trial.py --output-dir manip/P01/squat
"""

import argparse
import csv
import multiprocessing as mp
import queue
import struct
import threading
import time
from pathlib import Path
from threading import BrokenBarrierError

import cv2
import serial

BAUD_RATE = 2_000_000
BARRIER_TIMEOUT_S = 2.0
WARMUP_FRAMES = 15


def listen_receiver(port, switch_state, stop_flag, trigger_log_path, bin_path):
    """Write the raw serial stream to disk and log the switch transitions."""
    ser = serial.Serial(port, BAUD_RATE, timeout=0.01)
    buffer = bytearray()
    trigger_events = []
    last_logged = None

    with open(bin_path, "wb") as f_bin:
        while not stop_flag.value:
            if ser.in_waiting > 0:
                chunk = ser.read(ser.in_waiting)
                f_bin.write(chunk)
                f_bin.flush()

                buffer.extend(chunk)
                if len(buffer) > 500:
                    buffer = buffer[-500:]

                # The latest packet start: a Signal byte (0 or 1) followed by
                # the identifier of the first Patch of either insole.
                for i in range(len(buffer) - 3, -1, -1):
                    if buffer[i] in (0, 1):
                        if (buffer[i + 1] == 1 and buffer[i + 2] == 2) or (
                            buffer[i + 1] == 9 and buffer[i + 2] == 10
                        ):
                            value = buffer[i]
                            with switch_state.get_lock():
                                switch_state.value = value

                            # Only transitions are logged, with the PC wall
                            # clock, as the cameras stamp their frames.
                            if value != last_logged:
                                trigger_events.append((time.time(), value))
                                last_logged = value

                            buffer = buffer[i:]
                            break
    ser.close()

    with open(trigger_log_path, "w", newline="") as f_log:
        writer = csv.writer(f_log)
        writer.writerow(["timestamp_s", "switch_value"])
        writer.writerows(trigger_events)


def capture_camera(device, video_path, switch_state, stop_flag, barrier):
    """Record one camera while the switch is on.

    The barrier is shared by the two camera processes only: it makes both
    call cap.grab() at the same instant. The receiver process does not use it.
    Everything sits in try/finally so that a camera that fails to open still
    aborts the barrier, instead of leaving the other one waiting.
    """
    cap = None
    out = None
    writer_thread = None
    frame_queue: "queue.Queue" = queue.Queue(maxsize=150)
    frame_timestamps = []
    dropped_frames = 0

    def writer_loop():
        frame_index = 0
        while True:
            item = frame_queue.get()
            if item is None:
                break
            frame, timestamp = item
            out.write(frame)
            frame_timestamps.append((frame_index, timestamp))
            frame_index += 1

    try:
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {device}")
        # MJPG must be requested before the frame rate: the uncompressed
        # default (YUYV) caps these cameras at 30 fps.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        # Resolution and format are renegotiated at every opening, so they
        # are requested explicitly. The default driver buffer is kept: a
        # buffer of 1 capped the rate at about 20 fps, and the barrier keeps
        # both cameras in step anyway.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        w, h = int(cap.get(3)), int(cap.get(4))
        if w == 0 or h == 0:
            raise RuntimeError(f"invalid resolution from {device} ({w}x{h})")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(video_path, fourcc, CAMERA_FPS, (w, h))
        if not out.isOpened():
            raise RuntimeError(f"cannot create video file {video_path}")

        # The first frames after opening are slow or incomplete: they go
        # through the barrier, to stay in step, but are neither written nor
        # stamped.
        for _ in range(WARMUP_FRAMES):
            if stop_flag.value:
                break
            try:
                barrier.wait(timeout=BARRIER_TIMEOUT_S)
            except BrokenBarrierError:
                break
            cap.grab()
            try:
                barrier.wait(timeout=BARRIER_TIMEOUT_S)
            except BrokenBarrierError:
                break
            cap.retrieve()

        # Encoding runs in its own thread, so that a slow disk never delays
        # the capture and desynchronises the cameras.
        writer_thread = threading.Thread(target=writer_loop, daemon=True)
        writer_thread.start()

        while not stop_flag.value:
            # Both cameras grab together; the slower decoding comes after.
            try:
                barrier.wait(timeout=BARRIER_TIMEOUT_S)
            except BrokenBarrierError:
                break
            cap.grab()
            try:
                barrier.wait(timeout=BARRIER_TIMEOUT_S)
            except BrokenBarrierError:
                break

            ret, frame = cap.retrieve()
            if not ret:
                continue

            with switch_state.get_lock():
                switch_on = switch_state.value == 1

            if switch_on:
                # Stamped at capture with the PC wall clock, comparable to
                # trigger_log.csv.
                timestamp = time.time()
                try:
                    frame_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    # The disk cannot keep up: drop the frame rather than
                    # stall the capture.
                    dropped_frames += 1
    except Exception as exc:  # noqa: BLE001
        print(f"[{video_path}] ERROR: {exc}")
    finally:
        barrier.abort()

        if writer_thread is not None:
            frame_queue.put(None)
            writer_thread.join()

        if cap is not None:
            cap.release()
        if out is not None:
            out.release()

        with open(timestamps_path_of(video_path), "w", newline="") as f_ts:
            writer = csv.writer(f_ts)
            writer.writerow(["frame_index", "timestamp_s"])
            writer.writerows(frame_timestamps)

        if dropped_frames:
            print(f"[{video_path}] warning: {dropped_frames} frame(s) dropped (disk too slow)")
        print(f"[{video_path}] {len(frame_timestamps)} frames recorded.")


def timestamps_path_of(video_path):
    path = Path(video_path)
    return path.with_name(path.stem + "_timestamps.csv")


# ---------------------------------------------------------------------------
# End-of-recording diagnostic, computed after the stop from the written files.
# ---------------------------------------------------------------------------
FRAME_SIZE = 36
PACKET_SIZE = 148
SERIAL_RECORD_SIZE = PACKET_SIZE + 1
PACKET_SEQUENCES = {
    (1, 2): ("left", ((1, 2), (3, 4), (5, 6), (7, 8))),
    (9, 10): ("right", ((9, 10), (11, 12), (13, 14), (15, 16))),
}
CAMERA_FPS = 40.0
# Largest tolerated difference between the left and right packet counts (%)
LEFT_RIGHT_GAP_MAX_PCT = 2.0


def inspect_insole_packet(data, start):
    """Return the foot if a complete ESP-NOW packet starts at ``start``."""
    if start + PACKET_SIZE > len(data):
        return None
    sequence_info = PACKET_SEQUENCES.get(tuple(data[start : start + 2]))
    if sequence_info is None:
        return None
    foot, sequence = sequence_info
    for patch_index, identifier in enumerate(sequence):
        frame_start = start + patch_index * FRAME_SIZE
        if tuple(data[frame_start : frame_start + 2]) != identifier:
            return None
        forces = struct.unpack(">HHHH", data[frame_start + 3 : frame_start + 11])
        if any(force > 0x3FFF for force in forces):
            return None
    return foot


def insole_diagnostic(bin_path):
    """Count valid packets, losses, duplicates and fragments in the .bin."""
    stats = {
        "bytes_written": 0, "packets": 0, "left_packets": 0, "right_packets": 0,
        "leading_bytes": 0, "internal_fragment_bytes": 0, "trailing_bytes": 0,
        "lost_packets": 0, "duplicate_packets": 0, "signal_changes": 0,
    }
    # The same counters, restricted to packets received while Signal = 1.
    stats_on = {
        "packets": 0, "left_packets": 0, "right_packets": 0,
        "internal_fragment_bytes": 0, "lost_packets": 0, "duplicate_packets": 0,
    }
    data = Path(bin_path).read_bytes()
    stats["bytes_written"] = len(data)
    last_cpt = {}
    last_cpt_on = {}
    last_signal = None
    have_seen_packet = False
    pos = 0

    def record(foot, packet_start, marker):
        nonlocal last_signal, have_seen_packet
        if marker is not None:
            if last_signal is not None and marker != last_signal:
                stats["signal_changes"] += 1
            last_signal = marker
        cpt = data[packet_start + 2]
        previous_cpt = last_cpt.get(foot)
        if previous_cpt is not None:
            step = (cpt - previous_cpt) & 0xFF
            if step == 0:
                stats["duplicate_packets"] += 1
            elif 1 < step < 128:
                stats["lost_packets"] += step - 1
        last_cpt[foot] = cpt
        stats["packets"] += 1
        stats["left_packets" if foot == "left" else "right_packets"] += 1
        have_seen_packet = True

        if last_signal == 1:
            # Losses are counted only between two consecutive packets of the
            # same foot both received while Signal = 1.
            previous_cpt_on = last_cpt_on.get(foot)
            if previous_cpt_on is not None:
                step = (cpt - previous_cpt_on) & 0xFF
                if step == 0:
                    stats_on["duplicate_packets"] += 1
                elif 1 < step < 128:
                    stats_on["lost_packets"] += step - 1
            last_cpt_on[foot] = cpt
            stats_on["packets"] += 1
            stats_on["left_packets" if foot == "left" else "right_packets"] += 1
        else:
            last_cpt_on.clear()

    while len(data) - pos >= PACKET_SIZE:
        foot = inspect_insole_packet(data, pos)
        if foot is not None:
            record(foot, pos, None)
            pos += PACKET_SIZE
            continue
        if data[pos] in (0, 1):
            if len(data) - pos < SERIAL_RECORD_SIZE:
                break
            foot = inspect_insole_packet(data, pos + 1)
            if foot is not None:
                record(foot, pos + 1, data[pos])
                pos += SERIAL_RECORD_SIZE
                continue
        if have_seen_packet:
            stats["internal_fragment_bytes"] += 1
            if last_signal == 1:
                stats_on["internal_fragment_bytes"] += 1
        else:
            stats["leading_bytes"] += 1
        pos += 1

    stats["trailing_bytes"] = len(data) - pos
    return stats, stats_on


def read_active_periods(trigger_log_path):
    """Intervals [start, end] during which the switch was on."""
    periods = []
    start = None
    with open(trigger_log_path, newline="") as f:
        for row in csv.DictReader(f):
            t, v = float(row["timestamp_s"]), int(row["switch_value"])
            if v == 1 and start is None:
                start = t
            elif v == 0 and start is not None:
                periods.append((start, t))
                start = None
    if start is not None:
        periods.append((start, float("inf")))
    return periods


def camera_diagnostic(timestamps_path, periods):
    """Effective frame rate and gaps within each switch-on period (pauses with
    the switch off are not losses)."""
    with open(timestamps_path, newline="") as f:
        ts = [float(row["timestamp_s"]) for row in csv.DictReader(f)]
    res = {"frames": len(ts), "fps": 0.0, "max_gap_ms": 0.0, "gaps": 0, "missing_frames_est": 0}
    if len(ts) < 2:
        return res
    nominal_period = 1.0 / CAMERA_FPS
    total_duration = 0.0
    n_intervals = 0
    for i in range(1, len(ts)):
        dt = ts[i] - ts[i - 1]
        same_period = any(a <= ts[i - 1] and ts[i] <= b for a, b in periods) if periods else True
        if not same_period:
            continue
        total_duration += dt
        n_intervals += 1
        res["max_gap_ms"] = max(res["max_gap_ms"], dt * 1000)
        if dt > 1.5 * nominal_period:
            res["gaps"] += 1
            res["missing_frames_est"] += round(dt / nominal_period) - 1
    if total_duration > 0:
        res["fps"] = n_intervals / total_duration
    return res


def print_diagnostic(bin_path, trigger_log_path, video_paths):
    print("\n==================== DIAGNOSTIC ====================")
    try:
        stats, stats_on = insole_diagnostic(bin_path)
        print(
            "Insoles: "
            f"{stats['bytes_written']} bytes, {stats['packets']} valid packets "
            f"(L={stats['left_packets']}, R={stats['right_packets']})"
        )
        print(
            "Insole diagnostic: "
            f"internal fragments={stats['internal_fragment_bytes']} byte(s), "
            f"lost packets={stats['lost_packets']}, "
            f"duplicates={stats['duplicate_packets']}, "
            f"Signal changes={stats['signal_changes']}, "
            f"partial start={stats['leading_bytes']} byte(s), "
            f"partial end={stats['trailing_bytes']} byte(s)"
        )
        print(
            "Insoles while Signal=1: "
            f"{stats_on['packets']} valid packets "
            f"(L={stats_on['left_packets']}, R={stats_on['right_packets']}), "
            f"internal fragments={stats_on['internal_fragment_bytes']} byte(s), "
            f"lost packets={stats_on['lost_packets']}, "
            f"duplicates={stats_on['duplicate_packets']}"
        )
        if stats_on["packets"] == 0:
            print("  -> WARNING: no packet received with Signal=1")
        elif stats_on["left_packets"] == 0 or stats_on["right_packets"] == 0:
            missing = "left" if stats_on["left_packets"] == 0 else "right"
            print(f"  -> WARNING: no packet from the {missing} insole while Signal=1")
        elif stats_on["internal_fragment_bytes"] or stats_on["lost_packets"] or stats_on["duplicate_packets"]:
            print("  -> WARNING: losses or corruption during the recording (Signal=1)")
        else:
            print("  -> OK: no loss during the recording (Signal=1)")

        n_left, n_right = stats_on["left_packets"], stats_on["right_packets"]
        if n_left and n_right:
            gap = n_left - n_right
            gap_pct = 100.0 * abs(gap) / max(n_left, n_right)
            print(
                f"Left/right difference (Signal=1): L-R={gap:+d} packet(s) "
                f"({gap_pct:.2f} %, threshold {LEFT_RIGHT_GAP_MAX_PCT:.1f} %)"
            )
            if gap_pct > LEFT_RIGHT_GAP_MAX_PCT:
                fewer = "left" if n_left < n_right else "right"
                print(f"  -> WARNING: the {fewer} insole sent clearly fewer packets")
    except Exception as exc:  # noqa: BLE001
        print(f"Insoles: diagnostic failed ({exc})")

    try:
        periods = read_active_periods(trigger_log_path)
        print(f"Trigger: {len(periods)} recording period(s) (switch on)")
    except Exception as exc:  # noqa: BLE001
        periods = []
        print(f"Trigger: cannot read ({exc})")

    for video_path in video_paths:
        try:
            r = camera_diagnostic(timestamps_path_of(video_path), periods)
            print(
                f"{Path(video_path).name}: {r['frames']} frames, "
                f"effective fps={r['fps']:.1f} (nominal {CAMERA_FPS:.0f}), "
                f"gaps={r['gaps']}, estimated missing frames={r['missing_frames_est']}, "
                f"largest gap={r['max_gap_ms']:.1f} ms"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"{Path(video_path).name}: diagnostic failed ({exc})")
    print("====================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", default=".", help="trial directory, e.g. manip/P01/squat")
    parser.add_argument("--serial-port", default="/dev/ttyACM0", help="port of the reception ESP32")
    parser.add_argument(
        "--cameras", nargs=2, default=["/dev/video2", "/dev/video4"], metavar=("CAM1", "CAM2"),
        help="video devices of the two cameras",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    switch_state = mp.Value("i", 0)
    stop_flag = mp.Value("b", False)
    trigger_log_path = str(output_dir / "trigger_log.csv")
    bin_path = str(output_dir / "dataSemelle.bin")
    video_paths = [str(output_dir / "cam1_manip.mp4"), str(output_dir / "cam2_manip.mp4")]

    print("Starting the acquisition...")
    camera_barrier = mp.Barrier(2)
    processes = [
        mp.Process(
            target=listen_receiver,
            args=(args.serial_port, switch_state, stop_flag, trigger_log_path, bin_path),
        )
    ] + [
        mp.Process(target=capture_camera, args=(device, path, switch_state, stop_flag, camera_barrier))
        for device, path in zip(args.cameras, video_paths)
    ]
    for process in processes:
        process.start()

    print("\n---------------------------------------------------")
    print(f" RECORDING INTO {output_dir}:")
    print(" - Insoles:    dataSemelle.bin")
    print(" - Videos:     cam1_manip.mp4 & cam2_manip.mp4")
    print(" - Timestamps: trigger_log.csv, cam1_manip_timestamps.csv, cam2_manip_timestamps.csv")
    print(" (the receiver's switch starts and pauses the videos)")
    print("---------------------------------------------------")

    try:
        input("Press Enter to stop...\n")
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping the processes and closing the files...")
        with stop_flag.get_lock():
            stop_flag.value = True
        for process in processes:
            process.join()
        print("Done: the files are ready for the pipeline.")
        print_diagnostic(bin_path, trigger_log_path, video_paths)
