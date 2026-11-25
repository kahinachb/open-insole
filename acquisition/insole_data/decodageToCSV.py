#!/usr/bin/env python3
import csv
import os
import math
import struct
import argparse
import json
from pathlib import Path

# Une transmission ESP-NOW contient les quatre trames d'une semelle :
# 4 × 36 octets, suivis du timestamp sur 4 octets.
TAILLE_TRAME = 36
TAILLE_PAQUET = 148
SEQUENCES_PATCHS = {
    (1, 2): ((1, 2), (3, 4), (5, 6), (7, 8)),
    (9, 10): ((9, 10), (11, 12), (13, 14), (15, 16)),
}

# Coefficients pour la conversion de données brutes en données physiques
coeffsConversion = {
    "acc_xyz": 1/4096,
    "gy_xyz": 1/16.384,
    # MMC5983MA en mode 18 bits : zéro à 131072 counts,
    # sensibilité de 16384 counts/G et 1 G = 100 µT.
    "m_xyz": 131072,
    "f_xyz": 100.0/16384.0
}

taresFX29 = {}
LIVRES_VERS_NEWTONS = 4.44822
SPAN_NOMINAL_FX29_COUNTS = 14000.0
CAPTEURS_ABSENTS = {(7, 1), (15, 1)}

# Plage nominale de chaque FX29 monté dans les semelles. La tare corrige le
# zéro de chaque capteur, tandis que cette plage fournit l'estimation de sa
# sensibilité en N/count.
PLAGES_FX29_LB = {
    1: (100, 100, 100, 100),
    3: (50, 50, 50, 50),
    5: (50, 50, 50, 100),
    7: (None, 50, 50, 50),
    9: (100, 100, 100, 100),
    11: (100, 100, 100, 100),
    13: (100, 100, 100, 100),
    15: (None, 100, 100, 100),
}

conserver_forces_negatives = False

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ACC_CALIBRATION = (
    SCRIPT_DIR.parents[1]
    / "calibration_imu/data/session_20260728_155627_gauche/calibration_acc/accelerometer_calibration.json"
)
DEFAULT_GYRO_CALIBRATION = (
    SCRIPT_DIR.parents[1]
    / "calibration_imu/data/session_gyro_20260803_124950_gauche/calibration_gyro_std015_cv/gyroscope_calibration.json"
)
ACC_CAL_COLUMNS = ["accX_cal (g)", "accY_cal (g)", "accZ_cal (g)"]
GYRO_CAL_COLUMNS = ["gyX_cal (°/s)", "gyY_cal (°/s)", "gyZ_cal (°/s)"]


def load_imu_calibration(acc_path, gyro_path, label):
    """Charge une paire de calibrations acc/gyro indexée par ID1."""
    try:
        with open(acc_path, encoding="utf-8") as file:
            acc_document = json.load(file)
        with open(gyro_path, encoding="utf-8") as file:
            gyro_document = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"impossible de charger la calibration IMU {label} : {exc}"
        ) from exc

    calibrations = {}
    try:
        for patch in acc_document["patches"].values():
            identifier = int(patch["ID1"])
            calibrations.setdefault(identifier, {})["acc"] = (
                patch["correction_matrix"],
                [patch["additive_bias_g"][axis] for axis in "xyz"],
            )
        for patch in gyro_document["patches"].values():
            identifier = int(patch["ID1"])
            calibrations.setdefault(identifier, {})["gyro"] = (
                patch["correction_matrix"],
                [patch["bias_dps"][axis] for axis in "xyz"],
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"format de calibration IMU {label} invalide : {exc}"
        ) from exc

    for identifier, sensors in calibrations.items():
        if set(sensors) != {"acc", "gyro"}:
            raise ValueError(f"calibration IMU incomplète pour ID1={identifier}")
        for sensor, (matrix, bias) in sensors.items():
            if (
                len(matrix) != 3
                or any(len(row) != 3 for row in matrix)
                or len(bias) != 3
            ):
                raise ValueError(
                    f"dimensions de calibration {sensor} invalides pour ID1={identifier}"
                )
            values = [float(value) for row in matrix for value in row]
            values.extend(float(value) for value in bias)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"valeur non finie dans la calibration {sensor}, ID1={identifier}"
                )
    return calibrations


def load_all_imu_calibrations(
    acc_left_path,
    gyro_left_path,
    acc_right_path=None,
    gyro_right_path=None,
):
    """Fusionne les calibrations gauche et droite sans écraser un ID1."""
    if (acc_right_path is None) != (gyro_right_path is None):
        raise ValueError(
            "les calibrations IMU droites accéléromètre et gyroscope "
            "doivent être fournies ensemble"
        )

    groups = [
        ("gauche", acc_left_path, gyro_left_path, {1, 3, 5, 7}),
    ]
    if acc_right_path is not None:
        groups.append(
            ("droite", acc_right_path, gyro_right_path, {9, 11, 13, 15})
        )

    combined = {}
    for label, acc_path, gyro_path, expected_ids in groups:
        calibration = load_imu_calibration(acc_path, gyro_path, label)
        identifiers = set(calibration)
        missing = expected_ids - identifiers
        unexpected = identifiers - expected_ids
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"ID1 manquants {sorted(missing)}")
            if unexpected:
                details.append(f"ID1 inattendus {sorted(unexpected)}")
            raise ValueError(
                f"calibration IMU {label} incorrecte : {', '.join(details)}"
            )
        duplicates = set(combined) & identifiers
        if duplicates:
            raise ValueError(
                f"ID1 présents dans plusieurs calibrations : {sorted(duplicates)}"
            )
        combined.update(calibration)
    return combined


def matrix_vector_product(matrix, vector):
    return [
        sum(float(matrix[row][column]) * vector[column] for column in range(3))
        for row in range(3)
    ]


def apply_imu_calibration(decoded_frame, calibrations):
    """Ajoute acc_cal et gyro_cal à une trame déjà convertie en unités physiques."""
    identifier = int(decoded_frame[1])
    calibration = calibrations.get(identifier)
    if calibration is None:
        raise ValueError(f"aucune calibration IMU disponible pour ID1={identifier}")

    raw_acc = [float(value) for value in decoded_frame[8:11]]
    acc_matrix, additive_bias = calibration["acc"]
    calibrated_acc = matrix_vector_product(
        acc_matrix,
        [value + float(bias) for value, bias in zip(raw_acc, additive_bias)],
    )

    raw_gyro = [float(value) for value in decoded_frame[11:14]]
    gyro_matrix, subtractive_bias = calibration["gyro"]
    calibrated_gyro = matrix_vector_product(
        gyro_matrix,
        [value - float(bias) for value, bias in zip(raw_gyro, subtractive_bias)],
    )
    return decoded_frame + calibrated_acc + calibrated_gyro


def write_csv(list_of_lists, filepath):
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        for sublist in list_of_lists:
            writer.writerow(sublist)


def getForce(forceBrute, ID, numero):
    """Estime la compression d'un FX29 en N à partir des counts et de sa tare."""
    if (ID, numero) in CAPTEURS_ABSENTS:
        return 0.0

    plages_patch = PLAGES_FX29_LB.get(ID)
    if plages_patch is None:
        raise ValueError(f"ID de patch inconnu pour la conversion FX29 : {ID}")
    plage_lb = plages_patch[numero - 1]
    if plage_lb is None:
        return 0.0

    tare_patch = taresFX29.get(str(ID))
    zero_counts = tare_patch[numero - 1] if tare_patch else 1000.0
    if zero_counts is None:
        raise ValueError(f"tare absente pour le capteur présent ID={ID}, F{numero}")

    newtons_par_count = (
        plage_lb * LIVRES_VERS_NEWTONS / SPAN_NOMINAL_FX29_COUNTS
    )
    force_n = (forceBrute - float(zero_counts)) * newtons_par_count
    if conserver_forces_negatives:
        return force_n
    return max(0.0, force_n)


def decode(frame, signal, temps):
    raw_data = bytes([signal]) + bytes(frame) + bytes(temps)
    unpacked_data = struct.unpack('>BBBBHHHHhhhhhhBHBHBBBBBBBI', raw_data)
    (
        signal, ID1, ID2, CPT,
        F1_raw, F2_raw, F3_raw, F4_raw,
        accX_raw, accY_raw, accZ_raw,
        gyX_raw, gyY_raw, gyZ_raw,
        press_raw_part1, press_raw_part2,
        temp_raw_part1, temp_raw_part2,
        x0, x1, y0, y1, z0, z1, xyz_ext,
        temps,
    ) = unpacked_data

    forces_raw = [F1_raw, F2_raw, F3_raw, F4_raw]
    forces = [getForce(forces_raw[i], ID1, i + 1) for i in range(4)]
    accXYZ_raw = [accX_raw, accY_raw, accZ_raw]
    accXYZ = [acc * coeffsConversion["acc_xyz"] for acc in accXYZ_raw]
    gyXYZ_raw = [gyX_raw, gyY_raw, gyZ_raw]
    gyXYZ = [gy * coeffsConversion["gy_xyz"] for gy in gyXYZ_raw]

    temp_raw = temp_raw_part1 * 65536 + temp_raw_part2 
    press_raw = press_raw_part1 * 65536 + press_raw_part2

    xyz_raw = [
        (x0 * 2**10) | (x1 * 2**2) | ((xyz_ext // 2**6) & 0x03),
        (y0 * 2**10) | (y1 * 2**2) | ((xyz_ext // 2**4) & 0x03),
        (z0 * 2**10) | (z1 * 2**2) | ((xyz_ext // 2**2) & 0x03),
    ]
    m_xyz = [xyz - coeffsConversion["m_xyz"] for xyz in xyz_raw]
    f_xyz = [m * coeffsConversion["f_xyz"] for m in m_xyz]
    norm = math.sqrt(f_xyz[0]**2 + f_xyz[1]**2 + f_xyz[2]**2)

    if ID1 != 7 and ID1 != 15:
        temps = 0
    decodedFrame = [
        signal, ID1, ID2, CPT,
        forces[0], forces[1], forces[2], forces[3],
        accXYZ[0], accXYZ[1], accXYZ[2],
        gyXYZ[0], gyXYZ[1], gyXYZ[2],
        press_raw, temp_raw,
        f_xyz[0], f_xyz[1], f_xyz[2],
        norm, temps,
    ]
    return decodedFrame


def packet_is_valid(data, start):
    if start + TAILLE_PAQUET > len(data):
        return False

    sequence = SEQUENCES_PATCHS.get(tuple(data[start:start + 2]))
    if sequence is None:
        return False

    for patch_index, identifier in enumerate(sequence):
        frame_start = start + patch_index * TAILLE_TRAME
        if tuple(data[frame_start:frame_start + 2]) != identifier:
            return False

        raw_forces = struct.unpack('>HHHH', data[frame_start + 3:frame_start + 11])
        if any(force > 0x3FFF for force in raw_forces):
            return False

    return True


def get_converted_data(data, imu_calibrations=None):
    entete = [
        "Signal", "ID1", "ID2", "CPT",
        "F1 (N)", "F2 (N)", "F3 (N)", "F4 (N)",
        "accX (g)", "accY (g)", "accZ (g)",
        "gyX (°/s)", "gyY (°/s)", "gyZ (°/s)",
        "Press brute", "Temp brute",
        "magX (µT)", "magY (µT)", "magZ (µT)", "Norme (µT)",
        "Temps",
    ]
    if imu_calibrations is not None:
        entete.extend(ACC_CAL_COLUMNS + GYRO_CAL_COLUMNS)
    L = [entete]
    signal = 0
    packet_count = 0
    index = 0

    while index + TAILLE_PAQUET <= len(data):
        if data[index] in (0, 1) and packet_is_valid(data, index + 1):
            signal = data[index]
            packet_start = index + 1
            index += 1
        elif packet_is_valid(data, index):
            packet_start = index
        else:
            index += 1
            continue

        timestamp = data[packet_start + 144:packet_start + 148]
        for patch_index in range(4):
            frame_start = packet_start + patch_index * TAILLE_TRAME
            frame = data[frame_start:frame_start + TAILLE_TRAME]
            decoded_frame = decode(frame, signal, timestamp)
            if imu_calibrations is not None:
                decoded_frame = apply_imu_calibration(
                    decoded_frame, imu_calibrations
                )
            L.append(decoded_frame)

        packet_count += 1
        index = packet_start + TAILLE_PAQUET

    print("Nombre de paquets ESP-NOW valides :", packet_count)
    return L


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Conversion des données des semelles depuis un fichier binaire en CSV'
    )
    parser.add_argument('input_file', type=str, help='Le fichier binaire .bin en entrée')
    parser.add_argument('output_file', type=str, help='Le fichier .csv en sortie')

    # Arguments de tare assouplis : permet 1 ou plusieurs fichiers
    parser.add_argument(
        '--tare-file', type=str, nargs='+',
        help='Un ou plusieurs fichiers JSON de tare (ex: gauche puis droite)',
    )
    parser.add_argument(
        '--tare-file-left', type=str,
        help='Fichier JSON de tare pour la semelle gauche',
    )
    parser.add_argument(
        '--tare-file-right', type=str,
        help='Fichier JSON de tare pour la semelle droite',
    )

    parser.add_argument(
        '--signed-forces', action='store_true',
        help='Conserve les forces négatives pour diagnostiquer la tare',
    )
    parser.add_argument(
        '--calibrate-imu', action='store_true',
        help='Ajoute les colonnes accéléromètre et gyroscope calibrées',
    )
    parser.add_argument(
        '--acc-calibration', '--acc-calibration-left',
        dest='acc_calibration', type=Path, default=DEFAULT_ACC_CALIBRATION,
        help='JSON de calibration des accéléromètres gauches',
    )
    parser.add_argument(
        '--gyro-calibration', '--gyro-calibration-left',
        dest='gyro_calibration', type=Path, default=DEFAULT_GYRO_CALIBRATION,
        help='JSON de calibration des gyroscopes gauches',
    )
    parser.add_argument(
        '--acc-calibration-right', type=Path,
        help='JSON de calibration des accéléromètres droits',
    )
    parser.add_argument(
        '--gyro-calibration-right', type=Path,
        help='JSON de calibration des gyroscopes droits',
    )
    args = parser.parse_args()
    conserver_forces_negatives = args.signed_forces

    # Chargement de tous les fichiers de tare spécifiés
    tare_files = []
    if args.tare_file:
        tare_files.extend(args.tare_file)
    if args.tare_file_left:
        tare_files.append(args.tare_file_left)
    if args.tare_file_right:
        tare_files.append(args.tare_file_right)

    for path in tare_files:
        try:
            with open(path, encoding='utf-8') as f:
                tare_document = json.load(f)
            if tare_document.get('format') != 'fx29_tare_v1':
                parser.error(f"format de tare non reconnu pour {path}")
            taresFX29.update(tare_document['tares'])
        except OSError as exc:
            parser.error(f"impossible de lire le fichier de tare {path} : {exc}")

    if taresFX29:
        print(f"Tares chargées pour les patchs ID : {sorted(map(int, taresFX29.keys()))}")

    imu_calibrations = None
    if args.calibrate_imu:
        try:
            imu_calibrations = load_all_imu_calibrations(
                args.acc_calibration,
                args.gyro_calibration,
                args.acc_calibration_right,
                args.gyro_calibration_right,
            )
        except ValueError as exc:
            parser.error(str(exc))

    with open(args.input_file, 'rb') as file:
        data_bytes = file.read()
    try:
        conv_data = get_converted_data(data_bytes, imu_calibrations)
    except ValueError as exc:
        parser.error(str(exc))

    print("Nombre de trames converties :", len(conv_data) - 1)
    write_csv(conv_data, args.output_file)