#!/usr/bin/env python3
"""Calcule et visualise la tare individuelle des capteurs FX29."""

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


FRAME_SIZE = 36
PACKET_SIZE = 148
PATCHES = {
    "gauche": ((1, 2), (3, 4), (5, 6), (7, 8)),
    "droite": ((9, 10), (11, 12), (13, 14), (15, 16)),
}
SEQUENCES = {patches[0]: patches for patches in PATCHES.values()}
MISSING = {(7, 8): {0}, (15, 16): {0}}
LBF_TO_N = 4.44822
NOMINAL_ZERO = 1000.0
NOMINAL_SPAN = 14000.0


def capacity_lbf(identifier, sensor_index):
    """Capacité installée, identique à la table du décodeur principal."""
    id1 = identifier[0]
    if id1 == 1 or id1 >= 9:
        return 100.0
    if id1 in (3, 7):
        return 50.0
    if id1 == 5:
        return 100.0 if sensor_index == 3 else 50.0
    raise ValueError(f"identifiant de patch inconnu : {identifier}")


def packet_is_valid(data, start):
    if start + PACKET_SIZE > len(data):
        return False
    sequence = SEQUENCES.get(tuple(data[start:start + 2]))
    if sequence is None:
        return False
    return all(
        tuple(data[start + index * FRAME_SIZE:start + index * FRAME_SIZE + 2])
        == identifier
        for index, identifier in enumerate(sequence)
    )


def read_samples(binary_path):
    data = binary_path.read_bytes()
    samples = {
        identifier: [[] for _ in range(4)]
        for patches in PATCHES.values() for identifier in patches
    }
    packet_counts = {"gauche": 0, "droite": 0}
    index = 0

    while index + PACKET_SIZE <= len(data):
        if data[index] in (0, 1) and packet_is_valid(data, index + 1):
            packet_start = index + 1
        elif packet_is_valid(data, index):
            packet_start = index
        else:
            index += 1
            continue

        first_identifier = tuple(data[packet_start:packet_start + 2])
        foot_name = "gauche" if first_identifier == (1, 2) else "droite"
        packet_counts[foot_name] += 1
        for patch_index, identifier in enumerate(PATCHES[foot_name]):
            frame_start = packet_start + patch_index * FRAME_SIZE
            for sensor_index in range(4):
                offset = frame_start + 3 + 2 * sensor_index
                raw = (data[offset] << 8) | data[offset + 1]
                if raw <= 0x3FFF:
                    samples[identifier][sensor_index].append(raw)
        index = packet_start + PACKET_SIZE

    return samples, packet_counts


def compute_tares(samples, foot_name):
    tares = {}
    stats = {}
    for identifier in PATCHES[foot_name]:
        key = str(identifier[0])
        tares[key] = []
        stats[key] = []
        for sensor_index, values in enumerate(samples[identifier]):
            if sensor_index in MISSING.get(identifier, set()):
                tares[key].append(None)
                stats[key].append(None)
                continue
            if not values:
                raise RuntimeError(
                    f"aucune mesure pour ID {identifier[0]}, F{sensor_index + 1}"
                )
            median = float(statistics.median(values))
            deviations = [abs(value - median) for value in values]
            mad = float(statistics.median(deviations))
            tares[key].append(median)
            stats[key].append({
                "samples": len(values),
                "median_counts": median,
                "mad_counts": mad,
                "min_counts": min(values),
                "max_counts": max(values),
            })
    return tares, stats


def raw_to_newtons(raw, zero, capacity):
    return (raw - zero) / NOMINAL_SPAN * capacity * LBF_TO_N


def create_plots(samples, tares, foot_name, frequency_hz, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for patch_number, identifier in enumerate(PATCHES[foot_name], start=1):
        active = [index for index in range(4)
                  if index not in MISSING.get(identifier, set())]
        figure, axes = plt.subplots(
            len(active), 2, figsize=(14, 2.6 * len(active)), squeeze=False,
            sharex="col",
        )
        figure.suptitle(
            f"Tare FX29 — semelle {foot_name}, patch {patch_number} "
            f"(ID {identifier[0]}/{identifier[1]})"
        )
        for row, sensor_index in enumerate(active):
            values = samples[identifier][sensor_index]
            time_s = [sample / frequency_hz for sample in range(len(values))]
            capacity = capacity_lbf(identifier, sensor_index)
            before = [raw_to_newtons(raw, NOMINAL_ZERO, capacity) for raw in values]
            zero = tares[str(identifier[0])][sensor_index]
            after = [raw_to_newtons(raw, zero, capacity) for raw in values]

            axes[row][0].plot(time_s, before, linewidth=0.8)
            axes[row][1].plot(time_s, after, linewidth=0.8)
            axes[row][0].set_ylabel(f"F{sensor_index + 1} (N)")
            axes[row][0].grid(True, alpha=0.3)
            axes[row][1].grid(True, alpha=0.3)
            axes[row][1].axhline(0.0, color="black", linestyle=":", linewidth=0.8)
            axes[row][1].text(
                0.01, 0.92, f"tare = {zero:.1f} counts",
                transform=axes[row][1].transAxes, va="top", fontsize="small",
            )

        axes[0][0].set_title("Avant tare (zéro nominal = 1000 counts)")
        axes[0][1].set_title("Après tare individuelle")
        axes[-1][0].set_xlabel("Temps estimé (s)")
        axes[-1][1].set_xlabel("Temps estimé (s)")
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        path = output_dir / f"tare_{foot_name}_patch_{patch_number}.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Calcule les tares FX29 depuis une acquisition sans charge."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--foot", choices=PATCHES, default="gauche")
    parser.add_argument("--frequency", type=float, default=200.0)
    parser.add_argument("--output", type=Path, default=Path("tare_fx29.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("plots_tare_fx29"))
    args = parser.parse_args()

    if not args.input_file.is_file():
        parser.error(f"fichier introuvable : {args.input_file}")
    if args.frequency <= 0:
        parser.error("--frequency doit être strictement positive")

    samples, packet_counts = read_samples(args.input_file)
    if packet_counts[args.foot] == 0:
        parser.error(f"aucun paquet de semelle {args.foot} trouvé")

    tares, stats = compute_tares(samples, args.foot)
    document = {
        "format": "fx29_tare_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(args.input_file),
        "foot": args.foot,
        "frequency_hz": args.frequency,
        "packet_count": packet_counts[args.foot],
        "nominal_span_counts": NOMINAL_SPAN,
        "tares": tares,
        "statistics": stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    paths = create_plots(
        samples, tares, args.foot, args.frequency, args.plots_dir
    )

    print(f"Paquets utilisés : {packet_counts[args.foot]}")
    for id1, values in tares.items():
        display = ["absent" if value is None else f"{value:.1f}" for value in values]
        print(f"ID {id1}: " + ", ".join(
            f"F{index + 1}={value}" for index, value in enumerate(display)
        ))
    print(f"Fichier de tare créé : {args.output}")
    for path in paths:
        print(f"Figure créée : {path}")


if __name__ == "__main__":
    main()
